"""LangGraph 实现的受约束法律检索循环。"""

from __future__ import annotations

import json
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.actions import action_prompt, first_allowed_deterministic_action
from agent.config import AgentSettings
from agent.evidence import (
    answer_evidence,
    citation_from_evidence,
    evidence_id,
    merge_child_results,
    merge_parent_results,
    select_parent_ids,
)
from agent.intent import deterministic_intent, intent_prompt
from agent.mcp_client import MCPClientError, MCPConnectionError, LegalToolClient
from agent.modeling import invoke_schema
from agent.policy import AgentPolicyError, search_budget_exhausted, validate_action
from agent.schemas import ActionDecision, AnswerDraft, AnswerValidation, IntentAnalysis
from agent.state import LegalAgentState
from online_retrieval.answerer import (
    _unsupported_generated_facts,
    _unsupported_legal_references,
)


class LegalAgentWorkflow:
    """持有模型和 MCP 客户端，并编译一张请求内无持久记忆的状态图。"""

    def __init__(self, client: LegalToolClient, model: Any | None, settings: AgentSettings) -> None:
        self.client = client
        self.model = model
        self.settings = settings
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(LegalAgentState)
        graph.add_node("validate_input", self.validate_input)
        graph.add_node("analyze_intent", self.analyze_intent)
        graph.add_node("select_action", self.select_action)
        graph.add_node("execute_tool", self.execute_tool)
        graph.add_node("observe", self.observe)
        graph.add_node("check_coverage", self.check_coverage)
        graph.add_node("fetch_context", self.fetch_context)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("verify_answer", self.verify_answer)
        graph.add_node("safe_evidence_fallback", self.safe_evidence_fallback)
        graph.add_node("finalize", self.finalize)
        graph.add_node("refuse", self.refuse)
        graph.add_node("baseline_fallback", self.baseline_fallback)

        graph.add_edge(START, "validate_input")
        graph.add_conditional_edges(
            "validate_input", self.route, {"analyze": "analyze_intent", "refuse": "refuse"}
        )
        graph.add_conditional_edges(
            "analyze_intent", self.route, {"select": "select_action", "refuse": "refuse"}
        )
        graph.add_conditional_edges(
            "select_action",
            self.route,
            {"execute": "execute_tool", "coverage": "check_coverage", "fetch": "fetch_context", "refuse": "refuse"},
        )
        graph.add_conditional_edges(
            "execute_tool",
            self.route,
            {
                "observe": "observe",
                "select": "select_action",
                "coverage": "check_coverage",
                "baseline": "baseline_fallback",
                "refuse": "refuse",
            },
        )
        graph.add_edge("observe", "check_coverage")
        graph.add_conditional_edges(
            "check_coverage",
            self.route,
            {"select": "select_action", "fetch": "fetch_context", "refuse": "refuse", "baseline": "baseline_fallback"},
        )
        graph.add_conditional_edges(
            "fetch_context", self.route, {"generate": "generate_answer", "refuse": "refuse"}
        )
        graph.add_conditional_edges(
            "generate_answer", self.route, {"verify": "verify_answer", "refuse": "refuse"}
        )
        graph.add_conditional_edges(
            "verify_answer",
            self.route,
            {
                "finish": "finalize",
                "rewrite": "generate_answer",
                "select": "select_action",
                "safe": "safe_evidence_fallback",
                "refuse": "refuse",
            },
        )
        graph.add_edge("safe_evidence_fallback", "finalize")
        graph.add_edge("finalize", END)
        graph.add_edge("refuse", END)
        graph.add_edge("baseline_fallback", END)
        return graph.compile()

    @staticmethod
    def route(state: LegalAgentState) -> str:
        return state["route"]

    def _trace(self, state: LegalAgentState, event: str, **values: Any) -> list[dict[str, Any]]:
        return [*state["public_trace"], {"event": event, **values}]

    def validate_input(self, state: LegalAgentState) -> dict[str, Any]:
        question = state["original_question"].strip()
        trace = list(state["public_trace"])
        trace[0] = {**trace[0], "request_id": state["request_id"]}
        if not question:
            return {
                "route": "refuse",
                "stop_reason": "invalid_input",
                "missing_evidence": ["问题不能为空"],
                "public_trace": [*trace, {"event": "input_rejected", "reason": "empty"}],
            }
        if len(question) > self.settings.max_question_chars:
            return {
                "route": "refuse",
                "stop_reason": "invalid_input",
                "missing_evidence": [
                    f"问题超过 {self.settings.max_question_chars} 字，请缩短后重试"
                ],
                "public_trace": [*trace, {"event": "input_rejected", "reason": "too_long"}],
            }
        return {
            "original_question": question,
            "normalized_question": question,
            "route": "analyze",
            "public_trace": [*trace, {"event": "input_validated"}],
        }

    async def analyze_intent(self, state: LegalAgentState) -> dict[str, Any]:
        fallback = deterministic_intent(state["original_question"])
        analysis = fallback
        warnings = list(state["warnings"])
        llm_calls = state["llm_call_count"]
        llm_usage_events = list(state["llm_usage_events"])
        if self.model is not None and llm_calls < self.settings.max_llm_calls:
            try:
                analysis = await invoke_schema(
                    self.model,
                    IntentAnalysis,
                    intent_prompt(state["original_question"], fallback),
                    timeout_seconds=self.settings.llm_timeout_seconds,
                    node="analyze_intent",
                    usage_events=llm_usage_events,
                )
                llm_calls += 1
                # 用户明确给出的正式引用不能被模型删除或替换。
                if fallback.exact_references:
                    analysis.exact_references = fallback.exact_references
                # 没有任何领域信号时必须保留不确定性。
                if not fallback.domain_signals:
                    analysis.domain = "uncertain"
            except Exception as exc:
                llm_calls += 1
                warnings.append(f"意图模型失败，已使用确定性分析: {type(exc).__name__}")

        route = "select" if analysis.in_scope and analysis.primary_intent != "out_of_scope" else "refuse"
        return {
            "in_scope": analysis.in_scope,
            "intent": analysis.primary_intent,
            "normalized_question": analysis.normalized_question,
            "domain": analysis.domain,
            "domain_signals": analysis.domain_signals,
            "terminology_map": analysis.terminology_map,
            "legal_issues": analysis.legal_issues or fallback.legal_issues,
            "required_case_facts": analysis.required_case_facts,
            "missing_user_facts": analysis.required_case_facts,
            "exact_references": [item.model_dump() for item in analysis.exact_references],
            "initial_queries": analysis.initial_queries or fallback.initial_queries,
            "llm_call_count": llm_calls,
            "llm_usage_events": llm_usage_events,
            "warnings": warnings,
            "route": route,
            "stop_reason": "out_of_scope" if route == "refuse" else "",
            "public_trace": self._trace(
                state,
                "intent_analyzed",
                intent=analysis.primary_intent,
                domain=analysis.domain,
                terminology_map=analysis.terminology_map,
                legal_issues=analysis.legal_issues,
                reason_summary=analysis.reason_summary,
            ),
        }

    async def select_action(self, state: LegalAgentState) -> dict[str, Any]:
        if search_budget_exhausted(state, self.settings):
            if state["child_evidence"]:
                return {"route": "coverage"}
            return {"route": "refuse", "stop_reason": self._budget_stop_reason(state)}

        approved = None
        signature = ""
        planned_by = "deterministic_policy"
        warnings = list(state["warnings"])
        llm_calls = state["llm_call_count"]
        llm_usage_events = list(state["llm_usage_events"])
        # 第一轮的正式条号/问题类型路由由确定性信号完成，减少一次无必要模型调用；
        # 第二轮开始模型根据上一轮真实证据缺口选择工具。
        if (
            state["iteration"] > 0
            and self.model is not None
            and llm_calls < self.settings.max_llm_calls
        ):
            try:
                decision = await invoke_schema(
                    self.model,
                    ActionDecision,
                    action_prompt(state),
                    timeout_seconds=self.settings.llm_timeout_seconds,
                    node="select_action",
                    usage_events=llm_usage_events,
                )
                llm_calls += 1
                approved, signature = validate_action(decision.action, state, self.settings)
                planned_by = "llm"
            except Exception as exc:
                llm_calls += 1
                warnings.append(f"工具规划未通过约束，已使用确定性动作: {type(exc).__name__}")

        if approved is None:
            fallback = first_allowed_deterministic_action(state, self.settings)
            if fallback is None:
                if state["child_evidence"]:
                    return {
                        "route": "coverage",
                        "warnings": warnings,
                        "llm_call_count": llm_calls,
                        "llm_usage_events": llm_usage_events,
                    }
                return {
                    "route": "refuse",
                    "stop_reason": "no_valid_action",
                    "warnings": warnings,
                    "llm_call_count": llm_calls,
                    "llm_usage_events": llm_usage_events,
                }
            approved, signature = fallback

        action = {**approved.model_dump(), "signature": signature}
        return {
            "planned_action": action,
            "route": "execute",
            "warnings": warnings,
            "llm_call_count": llm_calls,
            "llm_usage_events": llm_usage_events,
            "public_trace": self._trace(
                state,
                "action_planned",
                iteration=state["iteration"] + 1,
                tool_name=approved.tool_name,
                arguments=approved.arguments,
                target_issues=approved.target_issues,
                reason_summary=approved.reason_summary,
                planned_by=planned_by,
            ),
        }

    async def execute_tool(self, state: LegalAgentState) -> dict[str, Any]:
        action = state.get("planned_action")
        if not action:
            return {"route": "refuse", "stop_reason": "no_valid_action"}
        started = time.monotonic()
        calls = state["tool_call_count"] + 1
        try:
            result = await self.client.call_tool(action["tool_name"], action["arguments"])
            return {
                "last_tool_result": result,
                "last_tool_error": None,
                "tool_call_count": calls,
                "route": "observe",
                "public_trace": self._trace(
                    state,
                    "tool_called",
                    tool_name=action["tool_name"],
                    duration_ms=round((time.monotonic() - started) * 1000, 2),
                    result_count=result.get("count", 0),
                ),
            }
        except MCPClientError as exc:
            history = [
                *state["action_history"],
                {"tool_name": action["tool_name"], "arguments": action["arguments"], "status": "error", "error": type(exc).__name__},
            ]
            warnings = [*state["warnings"], f"MCP 工具失败: {type(exc).__name__}: {exc}"]
            attempted = list(dict.fromkeys([*state["attempted_action_signatures"], action["signature"]]))
            if isinstance(exc, MCPConnectionError):
                route = "baseline"
                stop_reason = "mcp_unavailable"
            elif state["correction_count"] < self.settings.max_corrections and not search_budget_exhausted(state, self.settings):
                route = "select"
                stop_reason = ""
            elif state["child_evidence"]:
                route = "coverage"
                stop_reason = ""
            else:
                route = "refuse"
                stop_reason = "tool_failure"
            return {
                "last_tool_result": None,
                "last_tool_error": str(exc),
                "tool_call_count": calls,
                "correction_count": state["correction_count"] + 1,
                "action_history": history,
                "attempted_action_signatures": attempted,
                "warnings": warnings,
                "planned_action": None,
                "route": route,
                "stop_reason": stop_reason,
                "use_baseline_fallback": route == "baseline",
                "public_trace": self._trace(state, "tool_failed", tool_name=action["tool_name"], error_type=type(exc).__name__),
            }

    def observe(self, state: LegalAgentState) -> dict[str, Any]:
        action = state["planned_action"]
        result = state["last_tool_result"] or {}
        child = state["child_evidence"]
        scope = dict(state["scope_documents"])
        new_count = 0
        if action["tool_name"] == "lookup_document_scope":
            for row in result.get("results", []):
                key = str(row.get("document_id") or row.get("title", ""))
                if key and key not in scope:
                    scope[key] = row
                    new_count += 1
        else:
            child, new_count = merge_child_results(
                child, result.get("results", []), self.settings.max_child_evidence
            )
        no_progress = 0 if new_count else state["no_progress_rounds"] + 1
        llm_calls = state["llm_call_count"]
        if action["tool_name"] == "hyde_search" and result.get("generated_by") == "llm":
            llm_calls += 1
        history = [
            *state["action_history"],
            {
                "tool_name": action["tool_name"],
                "arguments": action["arguments"],
                "target_issues": action.get("target_issues", []),
                "status": "success",
                "result_count": result.get("count", 0),
                "new_evidence_count": new_count,
            },
        ]
        return {
            "child_evidence": child,
            "scope_documents": scope,
            "iteration": state["iteration"] + 1,
            "llm_call_count": llm_calls,
            "no_progress_rounds": no_progress,
            "attempted_action_signatures": list(
                dict.fromkeys([*state["attempted_action_signatures"], action["signature"]])
            ),
            "action_history": history,
            "planned_action": None,
            "last_tool_result": None,
            "route": "coverage",
            "public_trace": self._trace(
                state,
                "evidence_observed",
                new_evidence_count=new_count,
                total_child_evidence=len(child),
                no_progress_rounds=no_progress,
            ),
        }

    async def check_coverage(self, state: LegalAgentState) -> dict[str, Any]:
        if not state["child_evidence"]:
            if search_budget_exhausted(state, self.settings):
                return {"route": "refuse", "stop_reason": self._budget_stop_reason(state)}
            return {"route": "select", "answerability": "insufficient", "should_continue": True}

        ids = list(state["child_evidence"])[:20]
        calls = state["tool_call_count"] + 1
        llm_calls = state["llm_call_count"]
        warnings = list(state["warnings"])
        try:
            result = await self.client.call_tool(
                "check_issue_coverage",
                {"question": state["original_question"], "evidence_ids": ids},
            )
            answerability = result.get("answerability", "insufficient")
            if result.get("planner_used"):
                llm_calls += 1
            should_continue = bool(result.get("should_continue", True))
            legal_issues = list(
                dict.fromkeys([*state["legal_issues"], *result.get("legal_issues", [])])
            )
            missing = result.get("missing_evidence", []) or result.get("uncovered_issues", [])
            coverage = result.get("issue_coverage", [])
            id_map = {
                int(row["candidate_id"]): row["evidence_id"]
                for row in result.get("evidence_index", [])
                if str(row.get("candidate_id", "")).isdigit()
            }
            issue_map: dict[str, list[str]] = {}
            for row in coverage:
                issue_map[str(row.get("issue", ""))] = [
                    id_map[number]
                    for number in row.get("candidate_ids", [])
                    if number in id_map
                ]
            # “设置条件是否合法”和“违反条件能否否决”是不同命题。模型不能拿只写
            # 公平性/文件内容的条文证明否决后果。高风险后果争点必须在候选正文中
            # 真实出现否决、无效、废标等表达，否则强制继续寻找后果依据。
            consequence_markers = ("否决", "无效", "废标", "不予受理", "不合格")
            evidence_consequence_terms = (
                "否决",
                "否决投标",
                "投标无效",
                "无效投标",
                "无效响应",
                "作无效处理",
                "废标",
                "不予受理",
                "不合格投标",
            )
            for required_issue in state["legal_issues"]:
                if not any(marker in required_issue for marker in consequence_markers):
                    continue
                supporting_ids = [
                    evidence_id_value
                    for evidence_id_value, evidence_row in state["child_evidence"].items()
                    if any(
                        term in str(evidence_row.get("text", ""))
                        for term in evidence_consequence_terms
                    )
                ]
                issue_map[required_issue] = supporting_ids
                if not supporting_ids:
                    missing = [*missing, required_issue]
                    should_continue = True
            missing = list(dict.fromkeys(str(value) for value in missing if str(value).strip()))
            if result.get("warning"):
                warnings.append(str(result["warning"]))
        except MCPConnectionError as exc:
            return {
                "tool_call_count": calls,
                "llm_call_count": llm_calls,
                "warnings": [*warnings, f"覆盖检查无法连接 MCP: {exc}"],
                "use_baseline_fallback": True,
                "route": "baseline",
                "stop_reason": "mcp_unavailable",
            }
        except MCPClientError as exc:
            warnings.append(f"覆盖检查失败，使用保守降级: {type(exc).__name__}: {exc}")
            answerability = "direct" if state["intent"] == "exact_reference" else "conditional"
            should_continue = state["iteration"] < self.settings.max_iterations
            legal_issues = state["legal_issues"]
            missing = state["missing_evidence"]
            coverage = []
            issue_map = {}

        exhausted = search_budget_exhausted(
            {**state, "tool_call_count": calls}, self.settings
        )
        if answerability in {"direct", "conditional"} and not should_continue:
            route = "fetch"
            stop_reason = f"{answerability}_evidence_complete"
        elif exhausted:
            route = "fetch"
            stop_reason = self._budget_stop_reason({**state, "tool_call_count": calls})
            if answerability == "insufficient":
                answerability = "conditional"
                warnings.append("检索预算已用尽，现有一般规则仅用于条件式回答")
        else:
            route = "select"
            stop_reason = ""
        return {
            "tool_call_count": calls,
            "llm_call_count": llm_calls,
            "answerability": answerability,
            "should_continue": should_continue,
            "legal_issues": legal_issues,
            "missing_evidence": missing,
            "issue_coverage": coverage,
            "evidence_issue_map": issue_map,
            "warnings": warnings,
            "route": route,
            "stop_reason": stop_reason,
            "public_trace": self._trace(
                state,
                "coverage_checked",
                answerability=answerability,
                should_continue=should_continue,
                covered_issues=sum(bool(value) for value in issue_map.values()),
                missing_evidence=missing,
            ),
        }

    async def fetch_context(self, state: LegalAgentState) -> dict[str, Any]:
        parent_ids = select_parent_ids(
            state["child_evidence"], min(state["top_k"], self.settings.max_parent_evidence)
        )
        if not parent_ids:
            return {"route": "refuse", "stop_reason": "knowledge_base_gap"}
        warnings = list(state["warnings"])
        calls = state["tool_call_count"] + 1
        try:
            result = await self.client.call_tool(
                "fetch_parent_context", {"parent_ids": parent_ids}
            )
            parents, _ = merge_parent_results(
                state["parent_evidence"], result.get("results", []), self.settings.max_parent_evidence
            )
            if result.get("missing_ids"):
                warnings.append("部分 parent ID 未找到，已保留可用证据")
        except MCPClientError as exc:
            parents = state["parent_evidence"]
            warnings.append(f"父块回填失败，回答将使用 child 证据: {type(exc).__name__}: {exc}")
        return {
            "parent_evidence": parents,
            "tool_call_count": calls,
            "warnings": warnings,
            "route": "generate",
            "public_trace": self._trace(
                state,
                "parent_context_fetched",
                requested=len(parent_ids),
                returned=len(parents),
            ),
        }

    def _evidence_prompt(self, state: LegalAgentState, evidence: list[dict[str, Any]]) -> str:
        blocks = []
        for row in evidence:
            metadata = row.get("metadata", {})
            blocks.append(
                f"证据ID={evidence_id(row)}\n法规={metadata.get('title', row.get('title', ''))}"
                f" {metadata.get('article_number', row.get('article_number', ''))}\n正文={row.get('text', '')}"
            )
        feedback = "；".join(state.get("validation_feedback", []))
        return (
            "你是谨慎的法规检索回答器，只能依据给定证据。只输出JSON对象，字段为"
            "answer、citation_ids、missing_user_facts。citation_ids只能填写给定证据ID。"
            "第一段直接作答；个案事实或领域不明时必须条件式回答，区分政府采购与工程"
            "建设招投标。不能把写入招标文件自动等同于合法，不能把没有专门数字规定"
            "自动等同于合法或违法，不得新增法规、条号、数字和行业惯例。必须分别回答"
            "条件本身能否设置，以及违反该条件能否产生否决后果；若证据只支持前者，"
            "必须明确后者尚不能从当前证据确定。\n"
            f"问题：{state['original_question']}\n领域：{state['domain']}\n"
            f"可回答性：{state['answerability']}\n法律争点：{json.dumps(state['legal_issues'], ensure_ascii=False)}\n"
            f"需补充事实：{json.dumps(state['missing_user_facts'], ensure_ascii=False)}\n"
            f"上次验证反馈：{feedback or '无'}\n证据：\n" + "\n\n".join(blocks)
        )

    async def generate_answer(self, state: LegalAgentState) -> dict[str, Any]:
        evidence = answer_evidence(
            state["parent_evidence"], state["child_evidence"], self.settings.max_parent_evidence
        )
        if not evidence:
            return {"route": "refuse", "stop_reason": "knowledge_base_gap"}
        available_ids = [evidence_id(row) for row in evidence]
        llm_calls = state["llm_call_count"]
        llm_usage_events = list(state["llm_usage_events"])
        warnings = list(state["warnings"])
        draft: AnswerDraft | None = None
        if self.model is not None and llm_calls < self.settings.max_llm_calls:
            try:
                draft = await invoke_schema(
                    self.model,
                    AnswerDraft,
                    self._evidence_prompt(state, evidence),
                    timeout_seconds=self.settings.llm_timeout_seconds,
                    node="generate_answer",
                    usage_events=llm_usage_events,
                )
                llm_calls += 1
            except Exception as exc:
                llm_calls += 1
                warnings.append(f"Agent 回答模型失败，使用原文证据回答: {type(exc).__name__}")
        if draft is None:
            lines = ["基于当前知识库，检索到以下可核验依据："]
            for index, row in enumerate(evidence, 1):
                metadata = row.get("metadata", {})
                lines.append(
                    f"{index}. 《{metadata.get('title', row.get('title', '未命名法规'))}》"
                    f"{metadata.get('article_number', row.get('article_number', ''))}：{row.get('text', '')}"
                )
            lines.append("当前未使用生成模型归纳结论，请结合项目类型和招标文件原文判断。")
            draft = AnswerDraft(
                answer="\n".join(lines),
                citation_ids=available_ids,
                missing_user_facts=state["missing_user_facts"],
            )
            mode = "agent_evidence_fallback"
        else:
            mode = "agent_llm"
        citation_ids = [value for value in draft.citation_ids if value in available_ids]
        if not citation_ids:
            citation_ids = available_ids[: self.settings.max_citations]
        citation_ids = citation_ids[: self.settings.max_citations]
        citations = []
        for row in evidence:
            row_id = evidence_id(row)
            if row_id not in citation_ids:
                continue
            child_ids = [
                child_id
                for child_id, child in state["child_evidence"].items()
                if child.get("parent_id") == row_id or child_id == row_id
            ]
            supports = [
                issue
                for issue, evidence_ids in state["evidence_issue_map"].items()
                if any(child_id in evidence_ids for child_id in child_ids)
            ]
            citations.append(citation_from_evidence(row, supports))
        return {
            "draft_answer": draft.answer,
            "draft_citation_ids": citation_ids,
            "missing_user_facts": list(
                dict.fromkeys([*state["missing_user_facts"], *draft.missing_user_facts])
            ),
            "citations": citations,
            "llm_call_count": llm_calls,
            "llm_usage_events": llm_usage_events,
            "generation_mode": mode,
            "warnings": warnings,
            "route": "verify",
            "public_trace": self._trace(
                state, "answer_drafted", generation_mode=mode, citation_count=len(citations)
            ),
        }

    async def verify_answer(self, state: LegalAgentState) -> dict[str, Any]:
        evidence = answer_evidence(
            state["parent_evidence"], state["child_evidence"], self.settings.max_parent_evidence
        )
        evidence_text = "\n".join(str(row.get("text", "")) for row in evidence)
        shaped = [
            {"text": row.get("text", ""), "metadata": row.get("metadata", {})}
            for row in evidence
        ]
        deterministic_issues = [
            *_unsupported_generated_facts(
                state["draft_answer"], state["original_question"], evidence_text
            ),
            *_unsupported_legal_references(state["draft_answer"], shaped),
        ]
        available_ids = {evidence_id(row) for row in evidence}
        invalid_ids = [value for value in state["draft_citation_ids"] if value not in available_ids]
        if invalid_ids:
            deterministic_issues.append("引用 ID 不属于当前证据")
        warnings = list(state["warnings"])
        llm_calls = state["llm_call_count"]
        llm_usage_events = list(state["llm_usage_events"])
        validation = AnswerValidation(
            passed=not deterministic_issues,
            unsupported_claims=deterministic_issues,
            correction_action="rewrite_answer" if deterministic_issues else "none",
            reason_summary="确定性法规、条号、数字和引用检查",
        )
        if not deterministic_issues and self.model is not None and llm_calls < self.settings.max_llm_calls:
            prompt = (
                "你是法律RAG证据审查器，不补充外部知识。只输出JSON对象，字段为passed、"
                "unsupported_claims、overstated_claims、domain_mismatches、missing_issues、"
                "citation_errors、correction_action、reason_summary。逐项检查回答是否由证据"
                "支持，是否混淆政府采购与工程招标，是否把条件式规则写成绝对结论。"
                "引用对象由系统根据citation_ids单独附加，不要求回答正文出现[证据N]标记；"
                "只要条件式回答区分了规则、成立条件和事实缺口，不得仅因没有给绝对结论"
                "而判为失败。"
                "correction_action只能为none、rewrite_answer、retrieve_more、conditional_answer、refuse。\n"
                f"问题：{state['original_question']}\n领域：{state['domain']}\n"
                f"法律争点：{json.dumps(state['legal_issues'], ensure_ascii=False)}\n"
                f"回答：{state['draft_answer']}\n引用ID：{state['draft_citation_ids']}\n证据：{evidence_text}"
            )
            try:
                validation = await invoke_schema(
                    self.model,
                    AnswerValidation,
                    prompt,
                    timeout_seconds=self.settings.llm_timeout_seconds,
                    node="verify_answer",
                    usage_events=llm_usage_events,
                )
                llm_calls += 1
            except Exception as exc:
                llm_calls += 1
                warnings.append(f"LLM 蕴含检查失败，保留确定性校验结果: {type(exc).__name__}")

        feedback = [
            *validation.unsupported_claims,
            *validation.overstated_claims,
            *validation.domain_mismatches,
            *validation.missing_issues,
            *validation.citation_errors,
        ]
        if validation.passed and not feedback:
            route = "finish"
            final = state["draft_answer"]
            correction_count = state["correction_count"]
        elif state["correction_count"] >= self.settings.max_corrections:
            # 检索覆盖已经允许 direct/conditional 时，审查失败只说明模型归纳不可靠，
            # 不等于完全没有法规证据。撤销模型结论并回退到带引用的原文模式。
            route = "safe"
            final = ""
            correction_count = state["correction_count"]
        elif validation.correction_action == "retrieve_more" and not search_budget_exhausted(state, self.settings):
            route = "select"
            final = ""
            correction_count = state["correction_count"] + 1
        elif validation.correction_action in {"rewrite_answer", "conditional_answer"} or deterministic_issues:
            route = "rewrite"
            final = ""
            correction_count = state["correction_count"] + 1
        else:
            route = "refuse"
            final = ""
            correction_count = state["correction_count"] + 1
        return {
            "validation_result": validation.model_dump(),
            "validation_feedback": feedback,
            "llm_call_count": llm_calls,
            "llm_usage_events": llm_usage_events,
            "correction_count": correction_count,
            "warnings": warnings,
            "final_answer": final,
            "route": route,
            "stop_reason": state["stop_reason"] or ("validation_failed" if route == "refuse" else ""),
            "public_trace": self._trace(
                state,
                "answer_verified",
                passed=validation.passed and not feedback,
                correction_action=validation.correction_action,
                problem_count=len(feedback),
            ),
        }

    def safe_evidence_fallback(self, state: LegalAgentState) -> dict[str, Any]:
        """模型归纳不稳定时保留真实证据，但不保留未通过审查的法律结论。"""

        evidence = answer_evidence(
            state["parent_evidence"], state["child_evidence"], self.settings.max_parent_evidence
        )
        lines = [
            "模型归纳结果未通过证据一致性审查。以下仅提供本次检索到的可核验法规原文，"
            "不据此替你作出具体项目结论："
        ]
        for index, row in enumerate(evidence, 1):
            metadata = row.get("metadata", {})
            lines.append(
                f"{index}. 《{metadata.get('title', row.get('title', '未命名法规'))}》"
                f"{metadata.get('article_number', row.get('article_number', ''))}：{row.get('text', '')}"
            )
        if state["missing_user_facts"]:
            lines.append("仍需补充：" + "；".join(state["missing_user_facts"]) + "。")
        return {
            "final_answer": "\n".join(lines),
            "generation_mode": "agent_evidence_fallback",
            "insufficient_evidence": False,
            "answerability": "conditional",
            "warnings": [*state["warnings"], "模型回答审查未通过，已撤销结论并保留原文证据"],
            "route": "finish",
            "stop_reason": state["stop_reason"] or "conditional_evidence_complete",
            "public_trace": self._trace(
                state,
                "answer_replaced_by_safe_evidence",
                validation_feedback=state["validation_feedback"],
            ),
        }

    def finalize(self, state: LegalAgentState) -> dict[str, Any]:
        return {
            "insufficient_evidence": False,
            "route": "finish",
            "stop_reason": state["stop_reason"] or f"{state['answerability']}_evidence_complete",
            "public_trace": self._trace(
                state,
                "request_finished",
                stop_reason=state["stop_reason"] or f"{state['answerability']}_evidence_complete",
                iterations=state["iteration"],
                tool_calls=state["tool_call_count"],
                llm_calls=state["llm_call_count"],
            ),
        }

    def refuse(self, state: LegalAgentState) -> dict[str, Any]:
        reason = state["stop_reason"] or "knowledge_base_gap"
        details = "；".join(state.get("missing_evidence", []))
        answer = (
            "当前知识库和受约束多轮检索仍不足以支持可靠结论，因此系统停止作答。"
            + (f"仍缺少：{details}。" if details else "")
            + "请补充项目类型、招标文件完整条款、法规名称或文号，并以官方最新文本为准。"
        )
        return {
            "final_answer": answer,
            "citations": [],
            "generation_mode": "agent_abstention",
            "answerability": "insufficient",
            "insufficient_evidence": True,
            "route": "refuse",
            "stop_reason": reason,
            "public_trace": self._trace(state, "request_refused", stop_reason=reason),
        }

    def baseline_fallback(self, state: LegalAgentState) -> dict[str, Any]:
        return {
            "route": "baseline",
            "use_baseline_fallback": True,
            "public_trace": self._trace(state, "baseline_fallback_requested", reason=state["stop_reason"]),
        }

    def _budget_stop_reason(self, state: LegalAgentState) -> str:
        if time.monotonic() - state["started_at"] >= self.settings.timeout_seconds:
            return "time_budget_exhausted"
        if state["no_progress_rounds"] >= self.settings.no_progress_limit:
            return "no_progress"
        if state["tool_call_count"] >= self.settings.max_tool_calls:
            return "tool_budget_exhausted"
        return "iteration_budget_exhausted"
