"""Agent 对 FastAPI 和测试暴露的稳定业务入口。"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from agent.config import AgentSettings, get_agent_settings
from agent.graph import LegalAgentWorkflow
from agent.mcp_client import LegalToolClient, StreamableHTTPMCPClient
from agent.state import initial_agent_state
from online_retrieval.answerer import answer_question
from shared.llm import build_chat_model


class AgentBusyError(RuntimeError):
    """进程内 Agent 并发已满；API 应返回 503 而不是继续堆积任务。"""


class AgentResponseContractError(RuntimeError):
    """最终响应与直接/条件式/拒答契约不一致。"""


class AgentCapacityGuard:
    """很小的进程内舱壁，保护 CPU Reranker、模型和 MCP 连接。

    这是单机项目的并发保护，不冒充分布式限流。多个 API 进程分别拥有自己的
    计数器；若以后部署多副本，应由网关或共享存储再增加全局限流。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0

    @contextmanager
    def slot(self, limit: int):
        with self._lock:
            if self._active >= limit:
                raise AgentBusyError(f"Agent 当前并发已达到 {limit}，请稍后重试")
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)

    def snapshot(self, limit: int) -> dict[str, int]:
        with self._lock:
            return {
                "active_requests": self._active,
                "max_concurrent_requests": limit,
                "available_slots": max(0, limit - self._active),
            }


_CAPACITY_GUARD = AgentCapacityGuard()
_CLIENT_LOCK = threading.Lock()
_SHARED_CLIENT: StreamableHTTPMCPClient | None = None
_SHARED_CLIENT_KEY: tuple[Any, ...] | None = None


def _get_shared_mcp_client(settings: AgentSettings) -> StreamableHTTPMCPClient:
    """同一进程复用客户端对象，使连接熔断状态能够跨请求生效。"""

    global _SHARED_CLIENT, _SHARED_CLIENT_KEY
    key = (
        settings.mcp_server_url,
        settings.mcp_timeout_seconds,
        settings.mcp_retry_attempts,
        settings.mcp_circuit_failure_threshold,
        settings.mcp_circuit_recovery_seconds,
    )
    with _CLIENT_LOCK:
        if _SHARED_CLIENT is None or _SHARED_CLIENT_KEY != key:
            _SHARED_CLIENT = StreamableHTTPMCPClient(
                settings.mcp_server_url,
                settings.mcp_timeout_seconds,
                retry_attempts=settings.mcp_retry_attempts,
                circuit_failure_threshold=settings.mcp_circuit_failure_threshold,
                circuit_recovery_seconds=settings.mcp_circuit_recovery_seconds,
            )
            _SHARED_CLIENT_KEY = key
        return _SHARED_CLIENT


def _validate_agent_response(response: dict[str, Any], settings: AgentSettings) -> None:
    """最终出站前再次检查状态一致性，失败时由调用方转成安全拒答。"""

    answerability = response.get("answerability")
    insufficient = response.get("insufficient_evidence") is True
    citations = response.get("citations")
    if answerability not in {"direct", "conditional", "insufficient"}:
        raise AgentResponseContractError("answerability 不在允许枚举中")
    if not isinstance(response.get("answer"), str) or not response["answer"].strip():
        raise AgentResponseContractError("最终回答不能为空")
    if insufficient != (answerability == "insufficient"):
        raise AgentResponseContractError("answerability 与 insufficient_evidence 不一致")
    if not isinstance(citations, list) or len(citations) > settings.max_citations:
        raise AgentResponseContractError("引用不是数组或超过引用预算")
    if insufficient and citations:
        raise AgentResponseContractError("证据不足拒答不能携带看似支持结论的引用")
    if not insufficient and not citations:
        raise AgentResponseContractError("直接/条件式回答至少需要一个可追溯引用")
    seen: set[str] = set()
    for citation in citations:
        if not isinstance(citation, dict):
            raise AgentResponseContractError("引用项必须是对象")
        evidence_id = str(citation.get("evidence_id", "")).strip()
        if (
            not evidence_id
            or not str(citation.get("document_title", "")).strip()
            or not str(citation.get("quote", "")).strip()
        ):
            raise AgentResponseContractError("引用缺少 evidence_id、法规名或原文")
        if evidence_id in seen:
            raise AgentResponseContractError("最终引用存在重复 evidence_id")
        seen.add(evidence_id)


def _agent_response(state: dict[str, Any], settings: AgentSettings) -> dict[str, Any]:
    titles = list(
        dict.fromkeys(
            citation.get("document_title", "")
            for citation in state.get("citations", [])
            if citation.get("document_title")
        )
    )
    response = {
        "request_id": state["request_id"],
        "question": state["original_question"],
        "answer": state.get("final_answer", ""),
        "answerability": state.get("answerability", "insufficient"),
        "insufficient_evidence": bool(state.get("insufficient_evidence")),
        "stop_reason": state.get("stop_reason", ""),
        "refusal_reason": state.get("stop_reason") if state.get("insufficient_evidence") else None,
        "domain": state.get("domain", "uncertain"),
        "intent": state.get("intent", ""),
        "normalized_question": state.get("normalized_question", state["original_question"]),
        "terminology_map": state.get("terminology_map", {}),
        "legal_issues": state.get("legal_issues", []),
        "missing_user_facts": state.get("missing_user_facts", []),
        "citations": state.get("citations", []),
        "source_count": len(titles),
        "source_titles": titles,
        "generation_mode": state.get("generation_mode", "agent"),
        "warning": " | ".join(state.get("warnings", [])) or None,
        "agent_trace": {
            "iterations": state.get("iteration", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "llm_call_count": state.get("llm_call_count", 0),
            "correction_count": state.get("correction_count", 0),
            "actions": state.get("action_history", []),
            "issue_coverage": state.get("issue_coverage", []),
            "validation_result": state.get("validation_result", {}),
            "validation_feedback": state.get("validation_feedback", []),
            "events": state.get("public_trace", []),
            "budgets": {
                "max_iterations": settings.max_iterations,
                "max_tool_calls": settings.max_tool_calls,
                "max_llm_calls": settings.max_llm_calls,
                "max_corrections": settings.max_corrections,
                "timeout_seconds": settings.timeout_seconds,
                "request_timeout_seconds": settings.request_timeout_seconds,
                "llm_timeout_seconds": settings.llm_timeout_seconds,
                "mcp_timeout_seconds": settings.mcp_timeout_seconds,
            },
        },
        # 兼容当前 Web 的字段名；Agent 页面会优先读取 agent_trace。
        "retrieval_trace": {"mode": "agent_mcp", "evidence_sufficient": not state.get("insufficient_evidence")},
        "disclaimer": "回答仅基于当前知识库和受约束 Agent 检索结果，不构成法律意见；法规时效请核验官方最新记录。",
    }
    _validate_agent_response(response, settings)
    return response


def _operational_abstention(
    state: dict[str, Any], settings: AgentSettings, reason: str, warning: str
) -> dict[str, Any]:
    """依赖超时或响应损坏时返回格式稳定的失败关闭结果。"""

    safe_state = {
        **state,
        "final_answer": (
            "当前服务未能在约束时间和响应契约内完成可靠检索，因此本次停止作答。"
            "请稍后重试；如问题紧急，请直接核验官方法规原文。"
        ),
        "answerability": "insufficient",
        "insufficient_evidence": True,
        "citations": [],
        "generation_mode": "agent_operational_abstention",
        "stop_reason": reason,
        "warnings": [*state.get("warnings", []), warning],
    }
    return _agent_response(safe_state, settings)


async def _run_agent_once(
    question: str,
    top_k: int,
    settings: AgentSettings,
    client: LegalToolClient,
    model: Any | None,
    request_id: str | None,
) -> dict[str, Any]:
    """执行状态图和固定 RAG 降级；外层再负责并发与端到端超时。"""

    workflow = LegalAgentWorkflow(client, model, settings)
    initial = initial_agent_state(question, top_k, settings, request_id=request_id)
    try:
        state = await asyncio.wait_for(
            workflow.graph.ainvoke(initial, {"recursion_limit": 40}),
            timeout=settings.timeout_seconds + 2,
        )
    except TimeoutError:
        state = {
            **initial,
            "use_baseline_fallback": True,
            "stop_reason": "time_budget_exhausted",
            "warnings": ["Agent 图达到时间预算，已切换固定 RAG"],
        }

    if state.get("use_baseline_fallback"):
        try:
            baseline = await asyncio.wait_for(
                asyncio.to_thread(answer_question, question, top_k),
                timeout=settings.baseline_timeout_seconds,
            )
        except TimeoutError:
            return _operational_abstention(
                state,
                settings,
                "baseline_timeout",
                "固定 RAG 降级也达到独立时间预算",
            )
        baseline["request_id"] = state["request_id"]
        baseline["answerability"] = (
            "insufficient" if baseline.get("insufficient_evidence") else "conditional"
        )
        baseline["stop_reason"] = state.get("stop_reason", "mcp_unavailable")
        baseline["domain"] = state.get("domain", "uncertain")
        baseline["intent"] = state.get("intent", "")
        baseline["normalized_question"] = state.get("normalized_question", question)
        baseline["terminology_map"] = state.get("terminology_map", {})
        baseline["legal_issues"] = state.get("legal_issues", [])
        baseline["missing_user_facts"] = state.get("missing_user_facts", [])
        baseline["generation_mode"] = "agent_baseline_fallback"
        baseline["agent_trace"] = {
            "iterations": state.get("iteration", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "llm_call_count": state.get("llm_call_count", 0),
            "correction_count": state.get("correction_count", 0),
            "actions": state.get("action_history", []),
            "issue_coverage": state.get("issue_coverage", []),
            "events": state.get("public_trace", []),
            "fallback": "fixed_adaptive_rag",
        }
        return baseline
    try:
        return _agent_response(state, settings)
    except AgentResponseContractError as exc:
        return _operational_abstention(
            state,
            settings,
            "response_contract_failed",
            f"最终响应契约校验失败: {exc}",
        )


async def run_agent(
    question: str,
    top_k: int = 5,
    *,
    client: LegalToolClient | None = None,
    model: Any | None = None,
    settings: AgentSettings | None = None,
    request_id: str | None = None,
    capacity_guard: AgentCapacityGuard | None = None,
) -> dict[str, Any]:
    """运行一次无跨会话记忆的 Agent，并提供并发舱壁和端到端超时。"""

    settings = settings or get_agent_settings()
    if not settings.enabled:
        raise RuntimeError("Agent 功能已通过 ENABLE_AGENT=false 关闭")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k 必须在 1 到 10 之间")
    if not question.strip():
        raise ValueError("问题不能为空")
    if len(question.strip()) > settings.max_question_chars:
        raise ValueError(f"问题不能超过 {settings.max_question_chars} 字")
    client = client or _get_shared_mcp_client(settings)
    model = build_chat_model() if model is None else model
    guard = capacity_guard or _CAPACITY_GUARD
    with guard.slot(settings.max_concurrent_requests):
        try:
            return await asyncio.wait_for(
                _run_agent_once(
                    question.strip(), top_k, settings, client, model, request_id
                ),
                timeout=settings.request_timeout_seconds,
            )
        except TimeoutError:
            initial = initial_agent_state(
                question.strip(), top_k, settings, request_id=request_id
            )
            return _operational_abstention(
                initial,
                settings,
                "request_timeout",
                "Agent 达到端到端请求时间预算",
            )


async def agent_health(settings: AgentSettings | None = None) -> dict[str, Any]:
    """轻量检查 Agent 配置、LLM 和 MCP 工具发现。"""

    settings = settings or get_agent_settings()
    started = time.monotonic()
    degradation_reasons: list[str] = []
    try:
        llm_ready = build_chat_model() is not None
    except Exception as exc:
        llm_ready = False
        degradation_reasons.append(f"LLM配置不可用: {type(exc).__name__}")
    result: dict[str, Any] = {
        "status": "checking",
        "checked_at": datetime.now(UTC).isoformat(),
        "agent_enabled": settings.enabled,
        "agent_ready": False,
        "llm_ready": llm_ready,
        "agent_mode": "disabled" if not settings.enabled else "checking",
        "mcp_ready": False,
        "mcp_tool_count": 0,
        "required_tools_missing": [],
        "capacity": _CAPACITY_GUARD.snapshot(settings.max_concurrent_requests),
        "degradation_reasons": degradation_reasons,
    }
    if not settings.enabled:
        result["status"] = "disabled"
        result["check_duration_ms"] = round((time.monotonic() - started) * 1000, 2)
        return result
    client = _get_shared_mcp_client(settings)
    try:
        result.update(
            await asyncio.wait_for(
                client.health(), timeout=settings.health_timeout_seconds
            )
        )
    except Exception as exc:
        result["mcp_error"] = f"{type(exc).__name__}: {exc}"
        result["mcp_circuit"] = client.circuit_status()
        degradation_reasons.append("MCP工具服务不可用")
    # 没有外部 LLM 时仍可使用确定性意图、工具路由和原文回答，因此 Agent
    # 处于 degraded 而不是 unavailable；MCP 才是 Agent 工具闭环的必要依赖。
    result["agent_ready"] = bool(result["mcp_ready"])
    result["agent_mode"] = (
        "full_llm"
        if result["agent_ready"] and result["llm_ready"]
        else "deterministic_degraded"
        if result["agent_ready"]
        else "unavailable"
    )
    if not result["llm_ready"] and result["agent_ready"]:
        degradation_reasons.append("LLM不可用，使用确定性规划与原文回答")
    result["status"] = (
        "ok"
        if result["agent_mode"] == "full_llm"
        else "degraded"
        if result["agent_ready"]
        else "unavailable"
    )
    result["degradation_reasons"] = list(dict.fromkeys(degradation_reasons))
    result["check_duration_ms"] = round((time.monotonic() - started) * 1000, 2)
    return result
