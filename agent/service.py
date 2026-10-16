"""轻量法律检索 Agent：规划 → 检索 → 观察缺口 → 有限补查 → 带证据回答。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import uuid
from contextlib import contextmanager
from typing import Any

from agent.config import AgentSettings, get_agent_settings
from agent.intent import contextual_question, plan_question
from agent.modeling import invoke_schema
from agent.schemas import GroundedAnswer, QueryPlan, SearchPlan
from online_retrieval.answerer import DISCLAIMER, answer_prompt, citation_from_evidence, evidence_answer, validate_claims
from online_retrieval.retriever import LegalRetriever, matches_source
from shared.config import RETRIEVAL_MODE, get_llm_settings
from shared.llm import build_chat_model


class AgentBusyError(RuntimeError):
    pass


class AgentCapacityGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0

    @contextmanager
    def slot(self, limit: int):
        with self._lock:
            if self._active >= limit:
                raise AgentBusyError("当前检索请求较多，请稍后重试")
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1

    def snapshot(self, limit: int) -> dict:
        with self._lock:
            return {"active_requests": self._active, "max_concurrent_requests": limit, "available_slots": max(0, limit - self._active)}


_CAPACITY_GUARD = AgentCapacityGuard()
_AUTO_MODEL = object()


def _summarize_llm_usage(events: list[dict]) -> dict:
    complete = bool(events) and all(isinstance(event.get(key), int) for event in events for key in ("input_tokens", "output_tokens", "total_tokens"))
    return {"call_count": len(events), "token_reporting_complete": complete,
            **{key: sum(event[key] for event in events) if complete else None for key in ("input_tokens", "output_tokens", "total_tokens")},
            "events": events}


def _response(plan: QueryPlan, trace: dict, request_id: str, *, answer: str, evidence: list[dict] | None = None,
              reason: str = "", mode: str = "agent_abstention", clarification: bool = False) -> dict:
    evidence = evidence or []
    insufficient = not evidence
    status = "needs_clarification" if clarification or (evidence and plan.required_case_facts) else "refused" if insufficient else "answered"
    return {
        "request_id": request_id, "question": plan.normalized_question, "normalized_question": plan.normalized_question,
        "answer": answer, "status": status, "answerability": "insufficient" if insufficient else "conditional" if status == "needs_clarification" else "direct",
        "insufficient_evidence": insufficient, "refusal_reason": reason if insufficient else None,
        "missing_user_facts": plan.required_case_facts, "clarifying_questions": plan.required_case_facts if status == "needs_clarification" else [],
        "citations": [citation_from_evidence(row, [need.issue for need in plan.needs if LegalRetriever.supports(need, row)]) for row in evidence],
        "source_count": len({row["title"] for row in evidence}), "source_titles": list(dict.fromkeys(row["title"] for row in evidence)),
        "intent": plan.primary_intent, "domain": plan.domain, "cases": plan.cases,
        "legal_issues": [need.issue for need in plan.needs], "generation_mode": mode.replace("agent_", "rag_") if trace.get("mode") == "rag" else mode,
        "stop_reason": reason or "evidence_complete",
        "warning": "; ".join(trace.get("warnings", [])) or None, "disclaimer": DISCLAIMER,
        "agent_trace": {**trace, "llm_call_count": len(trace["usage_events"]), "llm_usage": _summarize_llm_usage(trace["usage_events"])},
    }


async def _execute(question: str, top_k: int, history: list | None, retriever: LegalRetriever, model: Any,
                   settings: AgentSettings, request_id: str, mode: str, trace: dict) -> dict:
    plan = plan_question(contextual_question(question, history))
    if plan.boundary_reason:
        return _response(plan, trace, request_id, answer=plan.boundary_reason, reason="scope_or_capability_boundary",
                         clarification=plan.boundary_action == "clarify")
    if plan.safe_exact:
        plan.needs = []  # 原文查询只核验用户明确要求的引用，不扩展成其他场景。
    if model is _AUTO_MODEL:
        try:
            model = None if plan.safe_exact else build_chat_model(timeout=settings.llm_timeout_seconds)
        except (ValueError, RuntimeError):
            model = None
            trace["warnings"].append("模型配置不可用，本次使用法规原文回答")
    references = list(plan.exact_references)
    if mode == "agent" and not plan.safe_exact:
        references.extend(source for need in plan.needs for source in need.sources)
    queries = [plan.normalized_question]
    if mode == "agent" and not plan.safe_exact:
        queries.extend(need.query for need in plan.needs)
    accumulated: dict[str, dict] = {}
    attempted: set[str] = set()
    assessment: dict = {"sufficient": False, "missing_evidence": []}
    stop_reason = "iteration_budget_exhausted"
    max_iterations = 1 if mode == "rag" or plan.safe_exact else settings.max_iterations
    for iteration in range(max_iterations):
        queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        signature = hashlib.sha256(json.dumps(sorted(queries), ensure_ascii=False).encode()).hexdigest()
        if signature in attempted:
            stop_reason = "no_new_query"
            break
        attempted.add(signature)
        rows = await asyncio.to_thread(retriever.search, queries, references=references)
        if plan.safe_exact:
            rows = [row for row in rows if any(matches_source(row, ref) for ref in plan.exact_references)]
        new_count = sum(row["chunk_id"] not in accumulated for row in rows)
        for row in rows:
            old = accumulated.get(row["chunk_id"])
            if old is None or float(row.get("score", 0)) > float(old.get("score", 0)):
                accumulated[row["chunk_id"]] = row
        trace["iterations"] = iteration + 1
        trace["tool_call_count"] += 1
        trace["actions"].append({"tool_name": "exact_legal_search" if plan.safe_exact else "search_legal_evidence", "status": "ok",
                                 "signature": signature, "result_count": len(rows), "new_evidence_count": new_count})
        assessment = retriever.assess(plan, list(accumulated.values()))
        trace["events"].append({"event": "evidence_checked", "iteration": iteration + 1, **assessment})
        if assessment["sufficient"]:
            break
        # 明确引用不存在时不能用语义相似的其他法条替代，也不花模型预算尝试补写。
        if plan.safe_exact or (plan.exact_references and not assessment["formal_reference_satisfied"]):
            stop_reason = "formal_reference_missing"
            break
        if not new_count:
            stop_reason = "no_new_evidence"
            break
        missing = [need for need in plan.needs if need.issue in assessment["missing_evidence"]]
        queries = [need.query for need in missing] or [plan.normalized_question]
        if model is not None and iteration + 1 < max_iterations and len(trace["usage_events"]) < settings.max_llm_calls - 1:
            try:
                search = await invoke_schema(model, SearchPlan,
                    "只输出JSON {\"queries\":[\"检索词\"]}，最多3条。根据问题与缺少证据改写检索词，不回答问题，不新增法规条号。\n"
                    f"问题：{plan.normalized_question}\n缺口：{assessment['missing_evidence']}",
                    timeout_seconds=settings.llm_timeout_seconds, node="refine_search", usage_events=trace["usage_events"])
                queries = [query.strip()[:300] for query in search.queries if query.strip()] or queries
            except Exception as exc:
                trace["warnings"].append(f"检索改写不可用，使用规则查询（{type(exc).__name__}）")
    trace["issue_coverage"] = assessment.get("issue_coverage", [])
    if not assessment["sufficient"]:
        details = "；".join(assessment["missing_evidence"]) or "没有新增且足够支持问题的证据"
        return _response(plan, trace, request_id, answer=f"当前知识库证据不足，停止作答。缺少：{details}。请提供准确法规、官方文件或补充检索范围。",
                         reason=stop_reason)
    parents = await asyncio.to_thread(retriever.parents, plan, list(accumulated.values()), top_k)
    trace["tool_call_count"] += 1
    if not parents or not retriever.assess(plan, parents)["sufficient"]:
        return _response(plan, trace, request_id, answer="最终上下文未覆盖全部争点或指定法源，停止作答。可增加返回证据数或缩小问题范围。",
                         reason="final_context_incomplete")
    answer = evidence_answer(plan, parents)
    generation_mode = "agent_exact_reference" if plan.safe_exact else "agent_evidence_fallback"
    if model is not None and not plan.safe_exact and len(trace["usage_events"]) < settings.max_llm_calls:
        try:
            draft = await invoke_schema(model, GroundedAnswer, answer_prompt(plan, parents),
                                        timeout_seconds=settings.llm_timeout_seconds, node="generate_answer", usage_events=trace["usage_events"])
            errors = validate_claims(draft, parents, plan)
            if errors:
                trace["warnings"].append("模型归纳未通过引用核验，已使用原文依据：" + "；".join(errors))
            else:
                citation_numbers = {row["parent_id"]: index for index, row in enumerate(parents, 1)}
                answer = "\n\n".join(f"{claim.text} [{citation_numbers[claim.evidence_id]}]\n原文：{claim.quote}" for claim in draft.claims)
                if plan.required_case_facts:
                    answer = "当前只能解释一般规则，补充事实后才能判断具体项目。\n\n" + answer + "\n\n请补充：" + "；".join(plan.required_case_facts)
                generation_mode = "agent_llm"
        except Exception as exc:
            trace["warnings"].append(f"模型暂不可用，已使用原文依据（{type(exc).__name__}）")
    return _response(plan, trace, request_id, answer=answer, evidence=parents, mode=generation_mode)


async def run_agent(question: str, top_k: int = 5, *, history: list | None = None, retriever: LegalRetriever | None = None,
                    model: Any = _AUTO_MODEL, settings: AgentSettings | None = None, request_id: str | None = None,
                    capacity_guard: AgentCapacityGuard | None = None, mode: str = "agent") -> dict:
    settings = settings or get_agent_settings()
    if not settings.enabled and mode == "agent":
        raise RuntimeError("Agent 已关闭")
    if not question.strip() or len(question) > settings.max_question_chars or not 1 <= top_k <= 10:
        raise ValueError("问题不能为空、不能超过2000字，top_k必须在1到10之间")
    if mode not in {"agent", "rag"}:
        raise ValueError("执行模式不合法")
    request_id = request_id or str(uuid.uuid4())
    retriever = retriever or LegalRetriever()
    trace: dict = {"mode": mode, "iterations": 0, "tool_call_count": 0, "actions": [], "events": [], "usage_events": [], "warnings": [], "runtime": "in_process"}
    with (capacity_guard or _CAPACITY_GUARD).slot(settings.max_concurrent_requests):
        try:
            return await asyncio.wait_for(_execute(question.strip(), top_k, history, retriever, model, settings, request_id, mode, trace),
                                          timeout=settings.request_timeout_seconds)
        except TimeoutError:
            plan = plan_question(contextual_question(question, history))
            return _response(plan, trace, request_id, answer="请求达到时间预算，已停止等待本次结果。请缩小问题后重试。", reason="request_timeout")


async def agent_health(settings: AgentSettings | None = None) -> dict:
    settings = settings or get_agent_settings()
    knowledge = LegalRetriever().health()
    try:
        llm = get_llm_settings()
        llm_configured = bool(llm["enabled"] and llm["api_key"] and llm["model"])
    except ValueError:
        llm_configured = False
    return {"agent_ready": settings.enabled and knowledge["ready"], "enabled": settings.enabled,
            "runtime": "in_process", "retrieval_mode": RETRIEVAL_MODE, "requires_mcp": False,
            "llm_enabled": llm_configured, "llm_required": False,
            "budgets": {"iterations": settings.max_iterations, "model_calls": settings.max_llm_calls},
            "capacity": _CAPACITY_GUARD.snapshot(settings.max_concurrent_requests)}
