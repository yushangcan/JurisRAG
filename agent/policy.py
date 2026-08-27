"""工具白名单、参数、预算和重复调用策略。"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.config import AgentSettings
from agent.schemas import (
    ExactSearchArgs,
    HydeSearchArgs,
    KeywordSearchArgs,
    MultiQueryArgs,
    ScopeLookupArgs,
    TextSearchArgs,
    ToolAction,
)
from agent.state import LegalAgentState


class AgentPolicyError(ValueError):
    """动作违反 Agent 的确定性安全规则。"""


ARGUMENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "exact_legal_search": ExactSearchArgs,
    "keyword_search": KeywordSearchArgs,
    "semantic_search": TextSearchArgs,
    "hyde_search": HydeSearchArgs,
    "multi_query_search": MultiQueryArgs,
    "lookup_document_scope": ScopeLookupArgs,
}


def action_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """标准化参数后计算签名，标点/字典顺序不会制造伪新动作。"""

    canonical = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{tool_name}:{canonical}".encode("utf-8")).hexdigest()


def time_budget_exhausted(state: LegalAgentState, settings: AgentSettings) -> bool:
    return time.monotonic() - state["started_at"] >= settings.timeout_seconds


def search_budget_exhausted(state: LegalAgentState, settings: AgentSettings) -> bool:
    return (
        state["iteration"] >= settings.max_iterations
        or state["tool_call_count"] >= settings.max_tool_calls
        or state["no_progress_rounds"] >= settings.no_progress_limit
        or time_budget_exhausted(state, settings)
    )


def validate_action(
    action: ToolAction,
    state: LegalAgentState,
    settings: AgentSettings,
) -> tuple[ToolAction, str]:
    """把模型建议变成允许执行的规范动作，或明确拒绝。"""

    if search_budget_exhausted(state, settings):
        raise AgentPolicyError("Agent 检索预算已经耗尽")
    schema = ARGUMENT_SCHEMAS.get(action.tool_name)
    if schema is None:
        raise AgentPolicyError(f"工具不在 Agent 搜索白名单: {action.tool_name}")
    try:
        normalized = schema.model_validate(action.arguments).model_dump()
    except ValidationError as exc:
        raise AgentPolicyError(f"工具参数不合法: {exc}") from exc

    # 精确工具至少需要一个精确信号，不能拿它执行空泛自然语言搜索。
    if action.tool_name == "exact_legal_search" and not any(
        normalized.get(key)
        for key in ("document_title", "article_number", "document_number")
    ):
        raise AgentPolicyError("精确检索至少需要法规名、条号或文号之一")
    if action.tool_name == "multi_query_search":
        cleaned = list(
            dict.fromkeys(query.strip() for query in normalized["queries"] if query.strip())
        )
        if not cleaned:
            raise AgentPolicyError("多查询工具至少需要一个非空查询")
        normalized["queries"] = cleaned[:5]

    normalized["top_k"] = min(int(normalized.get("top_k", settings.tool_top_k)), 10)
    approved = action.model_copy(update={"arguments": normalized})
    signature = action_signature(approved.tool_name, approved.arguments)
    if signature in state["attempted_action_signatures"]:
        raise AgentPolicyError("相同工具和参数已经执行过，禁止重复调用")
    return approved, signature

