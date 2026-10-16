"""FastAPI HTTP layer kept thin so business logic remains easy to test."""

from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agent.config import get_agent_settings
from agent.schemas import HistoryMessage
from agent.service import AgentBusyError, agent_health, run_agent
from online_retrieval.retriever import LegalRetriever
from shared.store import IndexUnavailable, LegalStore


# Web 页面与 API 同进程提供，前端无需构建工具。
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
LOGGER = logging.getLogger("jurisrag.api")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MAX_QUESTION_CHARS = get_agent_settings().max_question_chars

app = FastAPI(title="JurisRAG 轻量法律检索 Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class QuestionRequest(BaseModel):
    """Request body sent by the browser or an API client."""

    model_config = {"extra": "forbid"}

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="用户要查询的法规问题",
    )
    top_k: int = Field(default=5, ge=1, le=10, description="最多返回多少个去重后的父法条/章节")

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """去掉首尾空白，并在进入检索/模型前拒绝纯空白问题。"""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("问题不能为空")
        return cleaned


class AgentQuestionRequest(QuestionRequest):
    """Agent 模式请求；debug_trace=false 时可以隐藏逐节点事件。"""

    debug_trace: bool = Field(default=True, description="是否返回可公开的 Agent 执行事件")
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6, description="最近对话，仅用于承接问题，不能充当法律证据")


@app.middleware("http")
async def attach_request_context(request: Request, call_next):
    """为每个请求附加安全的追踪 ID 和耗时响应头。

    客户端可以传 ``X-Request-ID`` 关联自己的日志，但只接受短的字母数字 ID，
    防止换行或超长内容污染日志；不合法时由服务端重新生成 UUID。日志只记录
    方法、路径、状态和耗时，不记录用户问题，也不会记录 API Key。
    """

    supplied = request.headers.get("X-Request-ID", "").strip()
    request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    LOGGER.info(
        "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def _knowledge_health() -> dict[str, object]:
    """轻量模式只依赖已生成的 SQLite，不依赖模型或 MCP 进程。"""
    return LegalRetriever().health()


@app.get("/", include_in_schema=False)
def web_index() -> FileResponse:
    """Return the human-facing Q&A page; ``/docs`` remains the API console."""

    return FileResponse(WEB_DIR / "index.html")


@app.get("/live")
def liveness() -> dict[str, object]:
    """只说明 API 进程能响应，不访问数据库、模型或 MCP。"""

    return {"status": "alive", "checked_at": datetime.now(UTC).isoformat()}


@app.get("/health")
def health() -> dict[str, object]:
    """检查索引和已启用的可选模型文件；不触发模型下载或网络调用。"""

    return _knowledge_health()


@app.get("/ready")
async def readiness() -> JSONResponse:
    """检索依赖缺失返回503；仅关闭Agent时仍能提供基础RAG。"""

    knowledge = _knowledge_health()
    agent = await agent_health()
    core_ready = bool(knowledge["ready"])
    status = "ready" if core_ready and agent["agent_ready"] else "degraded" if core_ready else "unavailable"
    body = {
        "status": status,
        "ready": core_ready,
        "checked_at": datetime.now(UTC).isoformat(),
        "knowledge": knowledge,
        "agent": agent,
    }
    return JSONResponse(body, status_code=200 if core_ready else 503)


@app.post("/api/v1/questions")
async def ask_question(request: QuestionRequest, http_request: Request) -> dict[str, object]:
    """Answer one question and expose every evidence citation."""

    return await _answer(request, http_request, mode="rag")


@app.get("/api/v1/agent/health")
async def get_agent_health() -> dict[str, object]:
    """返回轻量Agent、可选模型配置及请求容量。"""

    return await agent_health()


@app.post("/api/v1/agent/questions")
async def ask_agent_question(request: AgentQuestionRequest, http_request: Request) -> dict[str, object]:
    """运行受约束 Agent；固定 RAG 接口继续保留用于基线与降级。"""

    return await _answer(request, http_request, mode="agent")


async def _answer(request: QuestionRequest, http_request: Request, *, mode: str) -> dict[str, object]:
    try:
        kwargs = {"mode": mode, "request_id": http_request.state.request_id}
        if isinstance(request, AgentQuestionRequest):
            kwargs["history"] = [message.model_dump() for message in request.history]
        result = await run_agent(
            request.question,
            request.top_k,
            **kwargs,
        )
        if isinstance(request, AgentQuestionRequest) and not request.debug_trace:
            result.get("agent_trace", {}).pop("events", None)
        return result
    except AgentBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "2"},
        ) from exc
    except IndexUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.error(
            "question_failed request_id=%s error_type=%s", http_request.state.request_id, type(exc).__name__
        )
        raise HTTPException(
            status_code=502,
            detail=f"问答暂不可用，请稍后重试（request_id={http_request.state.request_id}）",
        ) from exc


@app.get("/api/v1/cases")
def list_cases() -> dict[str, object]:
    """向页面展示支持的业务场景，不将场景规则当作法规正文。"""
    from agent.cases import RULES
    return {"domain": "政府采购与招投标", "cases": [{"id": rule.key, "name": rule.label, "issue": rule.issue} for rule in RULES]}


@app.get("/api/v1/evidence/{evidence_id}")
def full_evidence(evidence_id: str) -> dict[str, object]:
    if len(evidence_id) > 300:
        raise HTTPException(status_code=400, detail="证据ID过长")
    try:
        rows = LegalStore().fetch([evidence_id], "parent")
    except IndexUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not rows:
        raise HTTPException(status_code=404, detail="没有找到该证据")
    return rows[0]
