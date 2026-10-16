"""Web shell smoke tests that do not run the expensive retrieval pipeline."""

import pytest
from fastapi.testclient import TestClient

from agent.service import AgentBusyError
from online_retrieval.api import app


client = TestClient(app)


def test_home_page_contains_question_interface() -> None:
    """The root URL should return the human-facing form, not a JSON response."""

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="question-form"' in response.text
    assert 'id="query-mode"' in response.text
    assert "法律检索助手" in response.text
    assert "从法条原文中查找答案" in response.text


def test_frontend_static_assets_are_served() -> None:
    """FastAPI must expose both files required for layout and interaction."""

    css_response = client.get("/static/styles.css")
    js_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]


def test_agent_endpoint_uses_separate_route(monkeypatch) -> None:
    """Agent API 不覆盖固定 RAG 路由，并能隐藏逐节点 debug 事件。"""

    async def fake_run_agent(question: str, top_k: int, request_id: str | None = None, history=None, mode="agent"):
        return {
            "request_id": request_id,
            "question": question,
            "answer": "测试回答",
            "agent_trace": {"events": [{"event": "test"}]},
        }

    monkeypatch.setattr("online_retrieval.api.run_agent", fake_run_agent)
    response = client.post(
        "/api/v1/agent/questions",
        json={"question": "测试问题", "top_k": 3, "debug_trace": False},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "测试回答"
    assert "events" not in response.json()["agent_trace"]
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_request_id_is_sanitized_and_liveness_is_dependency_free() -> None:
    """危险请求 ID 不应原样进入日志；存活检查不访问模型或数据库。"""

    response = client.get("/live", headers={"X-Request-ID": "bad id with spaces"})
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert response.headers["X-Request-ID"] != "bad id with spaces"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_question_request_rejects_whitespace_and_oversized_input() -> None:
    """输入边界应在调用检索和模型之前由 API Schema 拦截。"""

    whitespace = client.post(
        "/api/v1/agent/questions",
        json={"question": "   ", "top_k": 3},
    )
    oversized = client.post(
        "/api/v1/agent/questions",
        json={"question": "法" * 2001, "top_k": 3},
    )
    assert whitespace.status_code == 422
    assert oversized.status_code == 422


def test_readiness_distinguishes_core_ready_from_agent_degraded(monkeypatch) -> None:
    """关闭Agent时仍可提供基础RAG。"""

    monkeypatch.setattr(
        "online_retrieval.api._knowledge_health",
        lambda: {"status": "ok", "ready": True},
    )

    async def degraded_agent():
        return {"status": "unavailable", "agent_ready": False}

    monkeypatch.setattr("online_retrieval.api.agent_health", degraded_agent)
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["ready"] is True


def test_readiness_returns_503_when_core_index_is_unavailable(monkeypatch) -> None:
    """核心索引不可用时，负载均衡器应能从HTTP状态码识别未就绪实例。"""

    monkeypatch.setattr(
        "online_retrieval.api._knowledge_health",
        lambda: {"status": "unavailable", "ready": False},
    )

    async def healthy_agent():
        return {"status": "ok", "agent_ready": True}

    monkeypatch.setattr("online_retrieval.api.agent_health", healthy_agent)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_agent_capacity_exhaustion_returns_retryable_503(monkeypatch) -> None:
    """并发舱壁满时不能返回400或堆积请求，应明确告诉客户端稍后重试。"""

    async def busy_agent(*_args, **_kwargs):
        raise AgentBusyError("Agent 当前并发已达到 2，请稍后重试")

    monkeypatch.setattr("online_retrieval.api.run_agent", busy_agent)
    response = client.post(
        "/api/v1/agent/questions",
        json={"question": "供应商条件", "top_k": 3},
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "2"


@pytest.mark.parametrize("route", ["/api/v1/questions", "/api/v1/agent/questions"])
def test_real_http_question_refusal_and_evidence_contract(route):
    response = client.post(route, json={"question": "政府采购法第二十二条"}, headers={"X-Request-ID": "http-e2e"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert data["request_id"] == response.headers["X-Request-ID"] == "http-e2e"
    assert data["agent_trace"]["llm_call_count"] == 0
    evidence = client.get("/api/v1/evidence/" + data["citations"][0]["evidence_id"])
    assert evidence.status_code == 200
    assert evidence.json()["text"] == data["citations"][0]["quote"]
    refused = client.post(route, json={"question": "政府采购法第九十九条"}).json()
    assert refused["status"] == "refused" and refused["citations"] == []


@pytest.mark.parametrize("route", ["/api/v1/questions", "/api/v1/agent/questions"])
def test_all_question_routes_handle_capacity_and_unexpected_errors(route, monkeypatch):
    async def busy(*args, **kwargs):
        raise AgentBusyError("当前请求较多")
    monkeypatch.setattr("online_retrieval.api.run_agent", busy)
    assert client.post(route, json={"question": "政府采购问题"}).status_code == 503

    async def broken(*args, **kwargs):
        raise Exception("private-key-and-user-data")
    monkeypatch.setattr("online_retrieval.api.run_agent", broken)
    response = client.post(route, json={"question": "政府采购问题"})
    assert response.status_code == 502
    assert "private-key-and-user-data" not in response.text


def test_history_is_bounded_and_can_resolve_a_followup():
    data = client.post("/api/v1/agent/questions", json={"question": "那第十三条呢？", "history": [
        {"role": "user", "content": "《政府采购质疑和投诉办法》第十条"}]}).json()
    assert [row["article_or_position"] for row in data["citations"]] == ["第十三条"]
    response = client.post("/api/v1/agent/questions", json={"question": "政府采购问题", "history": [
        {"role": "user", "content": "政府采购问题"}] * 7})
    assert response.status_code == 422
    assert client.get("/api/v1/evidence/nonexistent").status_code == 404
