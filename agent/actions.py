"""检索动作的模型提示词与确定性降级策略。"""

from __future__ import annotations

import json

from agent.config import AgentSettings
from agent.intent import compact_keywords
from agent.policy import AgentPolicyError, validate_action
from agent.schemas import ToolAction
from agent.state import LegalAgentState


SEARCH_TOOL_GUIDE = {
    "exact_legal_search": "用户明确给出法规名、条号或文号时使用",
    "keyword_search": "数字、期限、否决、资格和准确术语使用",
    "semantic_search": "自然语言、口语或同义表达使用",
    "hyde_search": "普通问法与法规语言差异大且其他检索不足时使用",
    "multi_query_search": "两个以上法律争点需要一起覆盖时使用",
    "lookup_document_scope": "已经知道法规名/文号且需要效力或范围元数据时使用",
}


def action_prompt(state: LegalAgentState) -> str:
    """只让模型选择一个搜索动作；coverage 和 parent 由状态图固定控制。"""

    evidence_digest = [
        {
            "id": item.get("chunk_id"),
            "title": item.get("title"),
            "article": item.get("article_number"),
            "parent_id": item.get("parent_id"),
        }
        for item in list(state["child_evidence"].values())[:8]
    ]
    return (
        "你是受约束法律检索 Agent 的工具规划器，不回答法律结论。只输出 JSON 对象，"
        "顶层字段为action；action字段必须为tool_name、arguments、target_issues、"
        "expected_evidence、reason_summary。每次只选一个工具。\n"
        f"可选工具：{json.dumps(SEARCH_TOOL_GUIDE, ensure_ascii=False)}\n"
        "不得选择coverage、parent或metadata工具，它们由状态图自动执行。"
        "不得重复已执行动作；不得凭记忆增加法规条号。reason_summary只写一句公开摘要。\n"
        f"原问题：{state['original_question']}\n"
        f"标准问题：{state['normalized_question']}\n"
        f"意图：{state['intent']}；领域：{state['domain']}\n"
        f"法律争点：{json.dumps(state['legal_issues'], ensure_ascii=False)}\n"
        f"缺失证据：{json.dumps(state['missing_evidence'], ensure_ascii=False)}\n"
        f"当前证据摘要：{json.dumps(evidence_digest, ensure_ascii=False)}\n"
        f"已执行动作：{json.dumps(state['action_history'], ensure_ascii=False)[:4000]}"
    )


def deterministic_actions(state: LegalAgentState, settings: AgentSettings) -> list[ToolAction]:
    """按法律检索常识生成有序候选，模型失败或建议非法动作时使用。"""

    top_k = min(state["top_k"], settings.tool_top_k, 10)
    candidates: list[ToolAction] = []
    for reference in state.get("exact_references", []):
        candidates.append(
            ToolAction(
                tool_name="exact_legal_search",
                arguments={**reference, "top_k": top_k},
                target_issues=state["legal_issues"],
                expected_evidence="用户明确引用的法规原文",
                reason_summary="先核对用户给出的法规、条号或文号",
            )
        )

    queries = list(
        dict.fromkeys(
            [
                *state.get("missing_evidence", []),
                *state.get("legal_issues", []),
                *state.get("initial_queries", []),
                state["normalized_question"],
            ]
        )
    )
    if state["intent"] in {"legality_assessment", "multi_issue"} or len(state["legal_issues"]) > 1:
        candidates.append(
            ToolAction(
                tool_name="multi_query_search",
                arguments={"queries": queries[:5], "top_k": top_k},
                target_issues=state["legal_issues"],
                expected_evidence="分别覆盖设置权限、公平性和法律后果的法规",
                reason_summary="复杂评价问题需要从多个法律争点同时召回",
            )
        )
    candidates.extend(
        [
            ToolAction(
                tool_name="semantic_search",
                arguments={"query": state["normalized_question"], "top_k": top_k},
                target_issues=state["legal_issues"],
                expected_evidence="与自然语言问题同义的法规表达",
                reason_summary="使用中文语义检索处理口语和同义表达",
            ),
            ToolAction(
                tool_name="keyword_search",
                arguments={"keywords": compact_keywords(queries), "top_k": top_k},
                target_issues=state["legal_issues"],
                expected_evidence="包含准确主体、行为、数字和法律后果的法条",
                reason_summary="使用关键词补充精确术语和数字召回",
            ),
            ToolAction(
                tool_name="hyde_search",
                arguments={"question": state["original_question"], "top_k": top_k},
                target_issues=state["legal_issues"],
                expected_evidence="采用法规语言表达的相关规则",
                reason_summary="其他表达不足时使用中性假设答案扩展语义召回",
            ),
        ]
    )
    title = next(
        (
            reference.get("document_title", "")
            for reference in state.get("exact_references", [])
            if reference.get("document_title")
        ),
        "",
    )
    if title:
        candidates.append(
            ToolAction(
                tool_name="lookup_document_scope",
                arguments={"title_or_number": title, "top_k": top_k},
                target_issues=state["legal_issues"],
                expected_evidence="法规类别、效力状态和官方来源元数据",
                reason_summary="核对已知法规的适用范围辅助信息",
            )
        )
    return candidates


def first_allowed_deterministic_action(
    state: LegalAgentState, settings: AgentSettings
) -> tuple[ToolAction, str] | None:
    """优先尝试尚未使用的通道，再考虑同工具的新查询参数。"""

    candidates = deterministic_actions(state, settings)
    used_tools = {
        row.get("tool_name")
        for row in state.get("action_history", [])
        if row.get("status") == "success"
    }
    ordered = [item for item in candidates if item.tool_name not in used_tools]
    ordered.extend(item for item in candidates if item.tool_name in used_tools)
    for candidate in ordered:
        try:
            return validate_action(candidate, state, settings)
        except AgentPolicyError:
            continue
    return None
