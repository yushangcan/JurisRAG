"""使用本地模拟端点验证兼容模型 HTTP 链路，不调用真实供应商。"""

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.service import run_agent
from shared.config import get_llm_settings
from shared.llm import CompatibleChatModel


@contextmanager
def endpoint(*, fail=False, wire_api="chat", response_status="completed", refusal=False):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers["Authorization"], payload))
            if fail:
                self.send_error(503, "private-provider-diagnostic")
                return
            prompt = payload["input"][0]["content"] if wire_api == "responses" else payload["messages"][0]["content"]
            state = json.loads(prompt.split("\n状态：", 1)[1])
            if not state["candidates"]:
                action = {"type": "search", "queries": ["单一来源采购"],
                          "references": [{"document_title": "政府采购法", "article_number": "第三十一条"}]}
            elif not state["evidence"]:
                row = next(row for row in state["candidates"] if row["article"] == "第三十一条")
                action = {"type": "read_evidence", "evidence_ids": [row["evidence_id"]]}
            else:
                claims = [{"text": row["text"], "quote": row["text"], "evidence_id": row["evidence_id"],
                           "issues": state["required_issues"]} for row in state["evidence"]]
                action = {"type": "answer", "claims": claims}
            content = json.dumps({"action": action})
            if wire_api == "responses":
                output = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "private-model-reasoning"}]},
                          {"type": "message", "role": "assistant", "content": [
                              {"type": "refusal", "refusal": "private-refusal"} if refusal else {"type": "output_text", "text": content}]}]
                data = {"status": response_status, "output": output,
                        "usage": {"input_tokens": 100, "output_tokens": 60, "total_tokens": 160}}
            else:
                data = {"choices": [{"message": {"content": content}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160}}
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    model = CompatibleChatModel({"provider": "local-test", "model": "local-test", "api_key": "test-key",
                                 "wire_api": wire_api, "reasoning_effort": "max", "disable_response_storage": True,
                                 "base_url": f"http://127.0.0.1:{server.server_port}/v1"}, timeout=2)
    try:
        yield model, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("wire_api", ["chat", "responses"])
def test_local_http_generation_keeps_grounding_and_supplier_usage(wire_api):
    with endpoint(wire_api=wire_api) as (model, received):
        result = asyncio.run(run_agent("哪些情形可以采用单一来源采购？", 1, model=model))
    assert result["generation_mode"] == "agent_llm_loop"
    assert result["agent_trace"]["llm_usage"]["total_tokens"] == 480
    assert len(received) == 3
    assert received[0][0] == ("/v1/responses" if wire_api == "responses" else "/v1/chat/completions")
    assert received[0][1] == "Bearer test-key"
    for _, _, payload in received:
        assert payload["store"] is False
        if wire_api == "responses":
            assert payload["reasoning"] == {"effort": "max"}
            assert payload["text"] == {"format": {"type": "json_object"}}
            assert not {"temperature", "messages", "max_tokens", "response_format"} & payload.keys()
        else:
            assert payload["response_format"] == {"type": "json_object"}
    assert result["citations"][0]["quote"] in result["answer"]
    assert "private-model-reasoning" not in json.dumps(result)


@pytest.mark.parametrize("wire_api", ["chat", "responses"])
def test_provider_http_error_falls_back_without_private_error_body(wire_api):
    with endpoint(fail=True, wire_api=wire_api) as (model, _):
        result = asyncio.run(run_agent("哪些情形可以采用单一来源采购？", 1, model=model))
    assert result["generation_mode"] == "agent_evidence_fallback"
    assert result["status"] == "answered"
    assert "private-provider-diagnostic" not in json.dumps(result)
    assert "test-key" not in json.dumps(result)


@pytest.mark.parametrize("response_status,refusal", [("incomplete", False), ("failed", False), ("completed", True)])
def test_responses_failure_cannot_execute_partial_action_and_preserves_usage(response_status, refusal):
    with endpoint(wire_api="responses", response_status=response_status, refusal=refusal) as (model, received):
        result = asyncio.run(run_agent("哪些情形可以采用单一来源采购？", 1, model=model))
    trace = result["agent_trace"]
    assert trace["controller"] == "rule_fallback"
    assert len(received) == 1
    assert not any(row["event"] == "model_action" for row in trace["events"])
    assert trace["llm_usage"]["total_tokens"] == 160
    assert trace["usage_events"][0]["error_type"] == "ModelUnavailable"
    assert "private-refusal" not in json.dumps(result)


def test_codez_uses_responses_with_requested_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "codez")
    for name in ("CODEZ_MODEL", "LLM_WIRE_API", "LLM_REASONING_EFFORT", "DISABLE_RESPONSE_STORAGE"):
        monkeypatch.delenv(name, raising=False)
    settings = get_llm_settings()
    assert settings["model"] == "gpt-6.1-sol"
    assert settings["wire_api"] == "responses"
    assert settings["reasoning_effort"] == "max"
    assert settings["disable_response_storage"] is True
