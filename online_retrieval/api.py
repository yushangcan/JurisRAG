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
from agent.service import AgentBusyError, agent_health, run_agent
from online_retrieval.answerer import answer_question
from shared.chroma_store import get_collection, get_parent_collection
from shared.config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL_PATH,
    RERANKER_BACKEND,
    RERANKER_MODEL_PATH,
)


# Web 页面与 API 由同一个 FastAPI 进程提供。这样浏览器访问页面时不会遇到
# 跨域问题，也不需要为这个简历项目额外安装 Node.js 或前端构建工具。
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
LOGGER = logging.getLogger("jurisrag.api")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MAX_QUESTION_CHARS = get_agent_settings().max_question_chars

app = FastAPI(title="招标采购法规知识库 RAG", version="0.4.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class QuestionRequest(BaseModel):
    """Request body sent by the browser or an API client."""

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
    """检查固定 RAG 的必要依赖；故障时返回结构化状态而不是直接抛 500。"""

    try:
        child_collection = get_collection()
        parent_collection = get_parent_collection()
        child_count = child_collection.count()
        parent_count = parent_collection.count()
        index_ready = child_count > 0 and parent_count > 0
        error = None
    except Exception as exc:
        child_count = 0
        parent_count = 0
        index_ready = False
        # 健康接口只暴露异常类型，详细堆栈留在服务日志，避免泄漏磁盘路径。
        error = type(exc).__name__
        LOGGER.exception("knowledge_health_failed")
    embedding_ready = EMBEDDING_BACKEND == "hash" or EMBEDDING_MODEL_PATH.exists()
    reranker_ready = RERANKER_BACKEND == "none" or RERANKER_MODEL_PATH.exists()
    ready = index_ready and embedding_ready and reranker_ready
    return {
        "status": "ok" if ready else "unavailable",
        "ready": ready,
        "indexed_chunks": child_count,
        "indexed_children": child_count,
        "indexed_parents": parent_count,
        "index_ready": index_ready,
        "embedding_ready": embedding_ready,
        "reranker_ready": reranker_ready,
        "error_type": error,
    }


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
    """Return index and local-model readiness for a deployment check.

    仅检查 Chroma 数量会漏掉一种常见问题：索引可以打开，但查询时需要的
    Embedding 或 Reranker 模型文件已经被移动。这里同时检查模型目录，让 Web
    页面在用户提问之前就能显示真实运行状态。
    """

    return _knowledge_health()


@app.get("/ready")
async def readiness() -> JSONResponse:
    """聚合固定 RAG 和 Agent 依赖，核心索引不可用时返回 HTTP 503。

    MCP/LLM 不可用但固定 RAG 正常时仍返回 HTTP 200 和 ``degraded``，因为系统
    可以提供有明确标记的基线服务；调用方可根据 ``agent.agent_ready`` 决定是否
    暂时隐藏 Agent 模式。
    """

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
def ask_question(request: QuestionRequest) -> dict[str, object]:
    """Answer one question and expose every evidence citation."""

    try:
        return answer_question(request.question, request.top_k)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"问答处理失败: {exc}") from exc


@app.get("/api/v1/agent/health")
async def get_agent_health() -> dict[str, object]:
    """检查 Agent 配置、回答模型和九个 MCP 工具是否就绪。"""

    return await agent_health()


@app.post("/api/v1/agent/questions")
async def ask_agent_question(request: AgentQuestionRequest, http_request: Request) -> dict[str, object]:
    """运行受约束 Agent；固定 RAG 接口继续保留用于基线与降级。"""

    try:
        result = await run_agent(
            request.question,
            request.top_k,
            request_id=http_request.state.request_id,
        )
        if not request.debug_trace:
            result.get("agent_trace", {}).pop("events", None)
        return result
    except AgentBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "2"},
        ) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception(
            "agent_request_failed request_id=%s", http_request.state.request_id
        )
        raise HTTPException(
            status_code=502,
            detail=f"Agent 问答暂不可用，请稍后重试（request_id={http_request.state.request_id}）",
        ) from exc
