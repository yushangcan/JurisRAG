"""Agent 的集中配置与边界校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass

# 导入 shared.config 会先加载项目根目录的 .env。
from shared.config import PROJECT_ROOT  # noqa: F401


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


@dataclass(frozen=True)
class AgentSettings:
    """一次 Agent 请求使用的预算；冻结后节点不能在运行时偷偷改参数。"""

    enabled: bool
    mcp_server_url: str
    max_iterations: int
    max_tool_calls: int
    max_llm_calls: int
    max_corrections: int
    max_child_evidence: int
    max_parent_evidence: int
    max_citations: int
    timeout_seconds: int
    request_timeout_seconds: int
    llm_timeout_seconds: int
    baseline_timeout_seconds: int
    mcp_timeout_seconds: int
    mcp_retry_attempts: int
    mcp_circuit_failure_threshold: int
    mcp_circuit_recovery_seconds: int
    health_timeout_seconds: int
    no_progress_limit: int
    tool_top_k: int
    max_question_chars: int
    max_concurrent_requests: int


def get_agent_settings() -> AgentSettings:
    """读取 Agent 配置，并在程序启动时尽早拒绝不合理参数。"""

    settings = AgentSettings(
        enabled=os.getenv("ENABLE_AGENT", "true").strip().lower() == "true",
        mcp_server_url=os.getenv(
            "MCP_SERVER_URL", "http://127.0.0.1:8001/mcp"
        ).strip(),
        max_iterations=_positive_int("AGENT_MAX_ITERATIONS", 3),
        max_tool_calls=_positive_int("AGENT_MAX_TOOL_CALLS", 8),
        max_llm_calls=_positive_int("AGENT_MAX_LLM_CALLS", 7),
        max_corrections=_positive_int("AGENT_MAX_CORRECTIONS", 2),
        max_child_evidence=_positive_int("AGENT_MAX_CHILD_EVIDENCE", 20),
        max_parent_evidence=_positive_int("AGENT_MAX_PARENT_EVIDENCE", 8),
        max_citations=_positive_int("AGENT_MAX_CITATIONS", 5),
        timeout_seconds=_positive_int("AGENT_TIMEOUT_SECONDS", 90),
        # 图预算、单次模型预算和固定 RAG 降级预算彼此独立。最外层请求预算
        # 保证即使某个依赖没有正确响应，API 也能在有限时间内停止。
        request_timeout_seconds=_positive_int("AGENT_REQUEST_TIMEOUT_SECONDS", 120),
        llm_timeout_seconds=_positive_int("AGENT_LLM_TIMEOUT_SECONDS", 35),
        baseline_timeout_seconds=_positive_int("AGENT_BASELINE_TIMEOUT_SECONDS", 25),
        mcp_timeout_seconds=_positive_int("AGENT_MCP_TIMEOUT_SECONDS", 30),
        mcp_retry_attempts=_positive_int("AGENT_MCP_RETRY_ATTEMPTS", 2),
        mcp_circuit_failure_threshold=_positive_int(
            "AGENT_MCP_CIRCUIT_FAILURE_THRESHOLD", 3
        ),
        mcp_circuit_recovery_seconds=_positive_int(
            "AGENT_MCP_CIRCUIT_RECOVERY_SECONDS", 30
        ),
        health_timeout_seconds=_positive_int("AGENT_HEALTH_TIMEOUT_SECONDS", 10),
        no_progress_limit=_positive_int("AGENT_NO_PROGRESS_LIMIT", 2),
        tool_top_k=_positive_int("AGENT_TOOL_TOP_K", 5),
        max_question_chars=_positive_int("AGENT_MAX_QUESTION_CHARS", 2000),
        max_concurrent_requests=_positive_int("AGENT_MAX_CONCURRENT_REQUESTS", 2),
    )
    if not settings.mcp_server_url.startswith(("http://", "https://")):
        raise ValueError("MCP_SERVER_URL 必须是 http/https 地址")
    if settings.tool_top_k > 10:
        raise ValueError("AGENT_TOOL_TOP_K 第一版不能大于 10")
    if settings.max_tool_calls < settings.max_iterations * 2:
        raise ValueError("AGENT_MAX_TOOL_CALLS 至少应允许每轮一次检索和一次覆盖检查")
    if settings.request_timeout_seconds <= settings.timeout_seconds:
        raise ValueError("AGENT_REQUEST_TIMEOUT_SECONDS 必须大于 AGENT_TIMEOUT_SECONDS")
    if settings.llm_timeout_seconds >= settings.request_timeout_seconds:
        raise ValueError("AGENT_LLM_TIMEOUT_SECONDS 必须小于端到端请求超时")
    if settings.mcp_timeout_seconds >= settings.request_timeout_seconds:
        raise ValueError("AGENT_MCP_TIMEOUT_SECONDS 必须小于端到端请求超时")
    if settings.mcp_retry_attempts > 3:
        raise ValueError("AGENT_MCP_RETRY_ATTEMPTS 第一版不能大于 3")
    if settings.health_timeout_seconds >= settings.request_timeout_seconds:
        raise ValueError("AGENT_HEALTH_TIMEOUT_SECONDS 必须小于端到端请求超时")
    if settings.max_question_chars > 10000:
        raise ValueError("AGENT_MAX_QUESTION_CHARS 第一版不能大于 10000")
    return settings
