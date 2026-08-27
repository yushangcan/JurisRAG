"""LangGraph 使用的请求内 Agent 状态。"""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from agent.config import AgentSettings


class LegalAgentState(TypedDict, total=False):
    request_id: str
    original_question: str
    normalized_question: str
    top_k: int
    intent: str
    in_scope: bool
    domain: str
    domain_signals: list[str]
    terminology_map: dict[str, str]
    legal_issues: list[str]
    required_case_facts: list[str]
    exact_references: list[dict[str, str]]
    initial_queries: list[str]

    iteration: int
    tool_call_count: int
    llm_call_count: int
    llm_usage_events: list[dict[str, Any]]
    correction_count: int
    no_progress_rounds: int
    started_at: float

    planned_action: dict[str, Any] | None
    attempted_action_signatures: list[str]
    action_history: list[dict[str, Any]]
    last_tool_result: dict[str, Any] | None
    last_tool_error: str | None

    child_evidence: dict[str, dict[str, Any]]
    parent_evidence: dict[str, dict[str, Any]]
    scope_documents: dict[str, dict[str, Any]]
    evidence_issue_map: dict[str, list[str]]
    issue_coverage: list[dict[str, Any]]
    conflicting_evidence: list[dict[str, Any]]

    answerability: str
    should_continue: bool
    missing_evidence: list[str]
    missing_user_facts: list[str]
    stop_reason: str
    route: str

    draft_answer: str
    draft_citation_ids: list[str]
    citations: list[dict[str, Any]]
    validation_result: dict[str, Any]
    validation_feedback: list[str]
    final_answer: str
    generation_mode: str
    insufficient_evidence: bool
    use_baseline_fallback: bool
    exact_fast_path: bool
    warnings: list[str]
    public_trace: list[dict[str, Any]]


def initial_agent_state(
    question: str,
    top_k: int,
    settings: AgentSettings,
    *,
    request_id: str | None = None,
) -> LegalAgentState:
    """创建字段齐全的初始状态，节点不需要反复处理缺失键。"""

    return {
        # HTTP 请求会把响应头中的 X-Request-ID 传进来；命令行和评测入口没有
        # 请求头时仍在这里生成 UUID。这样日志、响应正文和 Agent trace 使用同一 ID。
        "request_id": request_id or str(uuid.uuid4()),
        "original_question": question,
        "normalized_question": question,
        "top_k": top_k,
        "intent": "",
        "in_scope": True,
        "domain": "uncertain",
        "domain_signals": [],
        "terminology_map": {},
        "legal_issues": [],
        "required_case_facts": [],
        "exact_references": [],
        "initial_queries": [],
        "iteration": 0,
        "tool_call_count": 0,
        "llm_call_count": 0,
        "llm_usage_events": [],
        "correction_count": 0,
        "no_progress_rounds": 0,
        "started_at": time.monotonic(),
        "planned_action": None,
        "attempted_action_signatures": [],
        "action_history": [],
        "last_tool_result": None,
        "last_tool_error": None,
        "child_evidence": {},
        "parent_evidence": {},
        "scope_documents": {},
        "evidence_issue_map": {},
        "issue_coverage": [],
        "conflicting_evidence": [],
        "answerability": "insufficient",
        "should_continue": True,
        "missing_evidence": [],
        "missing_user_facts": [],
        "stop_reason": "",
        "route": "analyze",
        "draft_answer": "",
        "draft_citation_ids": [],
        "citations": [],
        "validation_result": {},
        "validation_feedback": [],
        "final_answer": "",
        "generation_mode": "agent",
        "insufficient_evidence": False,
        "use_baseline_fallback": False,
        "exact_fast_path": False,
        "warnings": [],
        "public_trace": [
            {
                "event": "request_started",
                "request_id": "pending",
                "budgets": {
                    "max_iterations": settings.max_iterations,
                    "max_tool_calls": settings.max_tool_calls,
                    "max_llm_calls": settings.max_llm_calls,
                    "max_corrections": settings.max_corrections,
                    "timeout_seconds": settings.timeout_seconds,
                },
            }
        ],
    }
