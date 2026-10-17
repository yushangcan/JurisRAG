"""真实冻结语料 + 可控模型决策，验证观察、动作、补查和校验反馈的闭环。"""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.config import get_agent_settings
from agent.service import run_agent
from online_retrieval.retriever import LegalRetriever


class ScriptedModel:
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.states = []

    async def ainvoke(self, prompt):
        state = json.loads(prompt.split("\n状态：", 1)[1])
        self.states.append(state)
        action = next(self.actions)
        action = action(state) if callable(action) else action
        return SimpleNamespace(content=json.dumps({"action": action}, ensure_ascii=False),
                               usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def search(query="单一来源采购", **kwargs):
    return {"type": "search", "queries": [query], **kwargs}


def read_article(article):
    def action(state):
        row = next(row for row in state["candidates"] if row["article"] == article)
        return {"type": "read_evidence", "evidence_ids": [row["evidence_id"]]}
    return action


def answer(state):
    return {"type": "answer", "claims": [{"text": row["text"], "quote": row["text"],
        "evidence_id": row["evidence_id"], "issues": state["required_issues"]} for row in state["evidence"]]}


def ask(model, question="哪些情形可以采用单一来源采购？", **kwargs):
    return asyncio.run(run_agent(question, model=model, **kwargs))


def test_model_observes_partial_evidence_then_chooses_another_search():
    class FocusedRetriever(LegalRetriever):
        def search(self, queries, *, references):
            assert references == []  # 模型工具参数没有自动混入场景条号。
            title, article = ("政府采购法", "第二十二条") if queries == ["资格条件"] else ("政府采购法实施条例", "第十七条")
            return [{**row, "score": 1} for row in self.store.exact(title, article)]

    model = ScriptedModel(search("资格条件"), read_article("第二十二条"), search("资格证明材料"), read_article("第十七条"), answer)
    result = ask(model, "供应商资格条件和证明材料有哪些？", retriever=FocusedRetriever(), top_k=2)
    assert result["status"] == "answered"
    assert result["generation_mode"] == "agent_llm_loop"
    assert {row["article_or_position"] for row in result["citations"]} == {"第二十二条", "第十七条"}
    trace = result["agent_trace"]
    assert trace["controller"] == "llm_loop"
    assert trace["iterations"] == 2 and trace["decision_steps"] == 5
    assert trace["llm_usage"]["total_tokens"] == 75
    assert [row["tool_name"] for row in trace["actions"]] == [
        "search_legal_evidence", "fetch_parent_context", "search_legal_evidence", "fetch_parent_context"]
    assert model.states[1]["observation"]["rule_assessment"]["missing_evidence"] == ["资格证明材料"]
    assert len(model.states[2]["evidence"]) == 1


def test_fabricated_citation_is_reported_to_model_and_can_be_repaired():
    def fabricated(state):
        action = answer(state)
        action["claims"][0]["evidence_id"] = "invented-parent"
        return action

    model = ScriptedModel(search(references=[{"document_title": "政府采购法", "article_number": "第三十一条"}]),
                          read_article("第三十一条"), fabricated, answer)
    result = ask(model, top_k=1)
    assert result["generation_mode"] == "agent_llm_loop"
    assert model.states[-1]["observation"]["event"] == "answer_rejected"
    assert "invented-parent" not in json.dumps(result, ensure_ascii=False)
    assert len(result["agent_trace"]["usage_events"]) == 4


@pytest.mark.parametrize("tamper", ["number", "quote", "title"])
def test_loop_does_not_accept_laundered_claims_and_allows_correction(tamper):
    def invalid(state):
        action = answer(state)
        claim = action["claims"][0]
        if tamper == "number":
            claim["text"] += "审批期限为999个工作日。"
        elif tamper == "quote":
            claim["quote"] = "采购人可以不受限制随意采购。"
        else:
            claim["text"] += "《政府采购法实施条例》第三十一条也如此规定。"
        return action

    model = ScriptedModel(search(references=[{"document_title": "政府采购法", "article_number": "第三十一条"}]),
                          read_article("第三十一条"), invalid, answer)
    result = ask(model, top_k=1)
    assert result["generation_mode"] == "agent_llm_loop"
    assert "999" not in result["answer"]
    assert model.states[-1]["observation"]["event"] == "answer_rejected"


def test_model_cannot_cite_a_candidate_without_reading_parent():
    def unread(state):
        row = state["candidates"][0]
        return {"type": "answer", "claims": [{"text": row["snippet"], "quote": row["snippet"],
            "evidence_id": row["evidence_id"], "issues": state["required_issues"]}]}

    model = ScriptedModel(search(references=[{"document_title": "政府采购法", "article_number": "第三十一条"}]),
                          unread, read_article("第三十一条"), answer)
    result = ask(model, top_k=1)
    assert result["generation_mode"] == "agent_llm_loop"
    assert model.states[2]["observation"]["event"] == "answer_rejected"


def test_model_cannot_read_ids_outside_its_search_results():
    model = ScriptedModel({"type": "read_evidence", "evidence_ids": ["unobserved"]},
                          {"type": "refuse", "reason": "尚未找到可引用的法源。"})
    result = ask(model)
    assert model.states[1]["observation"]["event"] == "invalid_action"
    assert result["agent_trace"]["tool_call_count"] == 0
    assert result["citations"] == []


def test_model_can_add_unmapped_issue_and_retrieve_it():
    def new_issue_answer(state):
        action = answer(state)
        action["claims"][0]["issues"] = state["required_issues"]
        return action

    model = ScriptedModel(search("采购文件保存期限", issues=["采购档案保管起算点"],
        references=[{"document_title": "政府采购法", "article_number": "第四十二条"}]), read_article("第四十二条"), new_issue_answer)
    result = ask(model, "政府采购记录应保管多久？", top_k=1)
    assert result["generation_mode"] == "agent_llm_loop"
    assert "采购档案保管起算点" in result["legal_issues"]
    assert "采购档案保管起算点" in result["citations"][0]["supports_issues"]
    coverage = {row["issue"]: row["evidence_ids"] for row in result["agent_trace"]["issue_coverage"]}
    assert coverage["采购档案保管起算点"] == [result["citations"][0]["evidence_id"]]


def test_missing_original_issue_cannot_be_removed_by_model():
    def incomplete(state):
        action = answer(state)
        action["claims"][0]["issues"] = [state["required_issues"][0]]
        return action

    model = ScriptedModel(search("供应商资格", references=[{"document_title": "政府采购法", "article_number": "第二十二条"}]),
        read_article("第二十二条"), incomplete, {"type": "refuse", "reason": "尚缺证明材料依据。"})
    result = ask(model, "供应商资格条件和证明材料有哪些？")
    assert result["status"] == "refused" and result["citations"] == []
    assert any("资格证明材料" in error for error in model.states[-1]["observation"]["errors"])


def test_duplicate_model_search_stops_without_reexecuting_tool():
    model = ScriptedModel({"type": "search", "queries": ["单一来源", "采购条件"]},
                          {"type": "search", "queries": ["采购条件", "单一来源"]})
    result = ask(model)
    assert result["stop_reason"] == "repeated_action"
    assert result["agent_trace"]["tool_call_count"] == 1
    assert result["citations"] == []


def test_no_new_evidence_stops_after_two_different_queries():
    class EmptyRetriever(LegalRetriever):
        def search(self, *args, **kwargs):
            return []
    result = ask(ScriptedModel(search("检索一"), search("检索二")), retriever=EmptyRetriever())
    assert result["stop_reason"] == "no_new_evidence"
    assert result["agent_trace"]["iterations"] == 2
    assert result["citations"] == []


def test_call_budget_never_promotes_unreviewed_results():
    result = ask(ScriptedModel(search()), settings=replace(get_agent_settings(), max_llm_calls=1))
    assert result["stop_reason"] == "model_budget_exhausted"
    assert result["citations"] == []


def test_search_budget_is_visible_and_cannot_be_exceeded():
    model = ScriptedModel(search("单一来源"), search("其他依据"), {"type": "refuse", "reason": "证据不足。"})
    result = ask(model, settings=replace(get_agent_settings(), max_iterations=1))
    assert model.states[-1]["observation"]["event"] == "search_budget_exhausted"
    assert result["agent_trace"]["iterations"] == 1
    assert result["citations"] == []


def test_model_can_request_user_facts_without_asserting_a_legal_conclusion():
    result = ask(ScriptedModel({"type": "clarify", "questions": ["请提供招标文件的完整要求"]}),
                 "我们政府采购投标被否决了，怎么办？")
    assert result["status"] == "needs_clarification"
    assert "请提供招标文件的完整要求" in result["clarifying_questions"]
    assert result["generation_mode"] == "agent_llm_clarification"
    assert result["citations"] == []


def test_disabled_model_exposes_rule_controller_instead_of_claiming_model_loop():
    result = ask(None)
    assert result["agent_trace"]["controller"] == "rule_workflow"
    assert result["agent_trace"]["decision_steps"] == 0


def test_explicit_articles_with_analysis_do_not_bypass_model_decisions():
    result = ask(ScriptedModel({"type": "clarify", "questions": ["请说明需要比较的具体问题"]}),
                 "《政府采购法》第二十二条与第二十三条有什么关系？")
    assert result["agent_trace"]["controller"] == "llm_loop"
    assert result["agent_trace"]["llm_call_count"] == 1


def test_invalid_answer_at_last_step_refuses_instead_of_leaking_draft():
    def invalid(state):
        action = answer(state)
        action["claims"][0]["text"] += "审批期限为999个工作日。"
        return action
    model = ScriptedModel(search(references=[{"document_title": "政府采购法", "article_number": "第三十一条"}]),
                          read_article("第三十一条"), invalid)
    result = ask(model, settings=replace(get_agent_settings(), max_llm_calls=3))
    assert result["stop_reason"] == "model_budget_exhausted"
    assert result["citations"] == []
    assert "999" not in result["answer"]


def test_model_failure_does_not_reset_or_exceed_search_budget():
    model = ScriptedModel(search("采购条件"))  # 下一次模型调用失败，进入规则回退。
    result = ask(model, settings=replace(get_agent_settings(), max_iterations=1))
    assert result["agent_trace"]["controller"] == "rule_fallback"
    assert result["agent_trace"]["iterations"] == 1
    assert result["stop_reason"] == "iteration_budget_exhausted"
    assert result["citations"] == []


@pytest.mark.parametrize("content,error_type", [
    ('{"action":', "JSONDecodeError"),
    ('{"action":{"type":"search","queries":["单一来源"]}]}', "JSONDecodeError"),
    ('{"action":{"type":"clarify","questions":[]}}', "ValidationError"),
])
def test_malformed_json_can_be_corrected_and_usage_is_not_lost(content, error_type):
    class RepairingModel(ScriptedModel):
        async def ainvoke(self, prompt):
            if not self.states:
                self.states.append({})
                return SimpleNamespace(content=content,
                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
            return await super().ainvoke(prompt)

    model = RepairingModel(search(references=[{"document_title": "政府采购法", "article_number": "第三十一条"}]),
                           read_article("第三十一条"), answer)
    result = ask(model, top_k=1)
    assert result["generation_mode"] == "agent_llm_loop"
    assert model.states[1]["observation"]["event"] == "invalid_decision"
    usage = result["agent_trace"]["llm_usage"]
    assert usage["total_tokens"] == 60
    assert usage["events"][0]["success"] is False
    assert usage["events"][0]["error_type"] == error_type
