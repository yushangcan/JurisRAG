"""验证补检、模型校验及总体预算的实际行为。"""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.config import get_agent_settings
from agent.service import run_agent
from online_retrieval.retriever import LegalRetriever
from shared.store import LegalStore


def ask(question, **kwargs):
    return asyncio.run(run_agent(question, model=kwargs.pop("model", None), **kwargs))


def test_second_search_can_complete_missing_issue_and_replace_noise():
    store = LegalStore()
    qualification = store.exact("政府采购法", "第二十二条")[0]
    materials = store.exact("政府采购法实施条例", "第十七条")[0]
    noise = store.exact("政府采购法", "第一条")[0]

    class ProgressiveRetriever(LegalRetriever):
        count = 0

        def search(self, queries, **kwargs):
            self.count += 1
            return [{**qualification, "score": 1}, {**noise, "score": 2}] if self.count == 1 else [{**materials, "score": 3}]

    result = ask("供应商资格条件及证明材料有哪些？", retriever=ProgressiveRetriever(), top_k=2)
    assert result["status"] == "answered"
    assert result["agent_trace"]["iterations"] == 2
    assert {row["article_or_position"] for row in result["citations"]} == {"第二十二条", "第十七条"}
    assert "第一条" not in {row["article_or_position"] for row in result["citations"]}


def test_identical_search_is_not_repeated():
    class OneResultRetriever(LegalRetriever):
        count = 0

        def search(self, queries, **kwargs):
            self.count += 1
            return self.store.exact("政府采购法", "第一条")

    retriever = OneResultRetriever()
    result = ask("政府采购量子传送设备专属审批期限是什么？", retriever=retriever,
                 settings=replace(get_agent_settings(), max_iterations=3))
    assert result["status"] == "refused"
    assert result["stop_reason"] == "no_new_query"
    assert retriever.count == 1


def test_request_timeout_preserves_model_usage_and_releases_capacity():
    class SlowModel:
        async def ainvoke(self, prompt):
            await asyncio.sleep(1)

    result = ask("哪些情形可以采用单一来源采购？", model=SlowModel(),
                 settings=replace(get_agent_settings(), request_timeout_seconds=0.25, llm_timeout_seconds=2))
    assert result["stop_reason"] == "request_timeout"
    assert result["citations"] == []
    assert result["agent_trace"]["iterations"] == 0  # 模型先决策，超时前尚未执行检索。
    assert result["agent_trace"]["llm_call_count"] == 1
    assert result["agent_trace"]["llm_usage"]["events"][0]["error_type"] == "CancelledError"
    assert ask("政府采购法第二十二条")["status"] == "answered"


def test_valid_model_output_retains_real_quotes_and_reported_tokens():
    class QuotingModel:
        async def ainvoke(self, prompt):
            passages = json.loads(prompt.split("证据：", 1)[1])
            claims = [{"text": row["text"], "evidence_id": row["evidence_id"], "quote": row["text"]} for row in passages]
            return SimpleNamespace(content=json.dumps({"claims": claims}), usage_metadata={"input_tokens": 100, "output_tokens": 80, "total_tokens": 180})

    result = ask("哪些情形可以采用单一来源采购？", model=QuotingModel(), top_k=1, mode="rag")
    assert result["generation_mode"] == "rag_llm"
    assert result["agent_trace"]["llm_usage"]["total_tokens"] == 180
    assert "原文：" in result["answer"]


@pytest.mark.parametrize("tamper", ["number", "chinese_number", "quote", "title"])
def test_real_evidence_id_cannot_launder_unsupported_claims(tamper):
    class TamperingModel:
        async def ainvoke(self, prompt):
            row = json.loads(prompt.split("证据：", 1)[1])[0]
            claim = {"text": row["text"], "evidence_id": row["evidence_id"], "quote": row["text"]}
            if tamper == "number":
                claim["text"] += "审批期限为999个工作日。"
            elif tamper == "chinese_number":
                claim["text"] += "审批期限为九百九十九个工作日。"
            elif tamper == "quote":
                claim["quote"] = "采购人可以不受限制随意采购。"
            else:
                claim["text"] += "《政府采购法实施条例》第三十一条也如此规定。"
            return SimpleNamespace(content=json.dumps({"claims": [claim]}))

    result = ask("哪些情形可以采用单一来源采购？", model=TamperingModel(), top_k=1, mode="rag")
    assert result["generation_mode"] == "rag_evidence_fallback"
    assert "999" not in result["answer"] and "九百九十九" not in result["answer"]


def test_broken_optional_model_config_does_not_break_retrieval(monkeypatch):
    def broken_model(**kwargs):
        raise ValueError("bad model configuration")
    monkeypatch.setattr("agent.service.build_chat_model", broken_model)
    result = asyncio.run(run_agent("哪些情形可以采用单一来源采购？"))
    assert result["status"] == "answered"
    assert result["generation_mode"] == "agent_evidence_fallback"


def test_followup_changes_article_without_retaining_the_old_article():
    result = ask("那第十三条呢？", history=[
        {"role": "user", "content": "《政府采购质疑和投诉办法》第十条规定什么？"},
        {"role": "user", "content": "那第十二条呢？"},
    ])
    assert result["status"] == "answered"
    assert [row["article_or_position"] for row in result["citations"]] == ["第十三条"]


def test_mixed_title_notation_keeps_each_article_with_its_law():
    result = ask("《政府采购法》第五十二条和政府采购质疑和投诉办法第十三条")
    assert {(row["document_title"], row["article_or_position"]) for row in result["citations"]} == {
        ("中华人民共和国政府采购法", "第五十二条"), ("政府采购质疑和投诉办法", "第十三条")}


@pytest.mark.parametrize("number", ["财政部令第87号", "财政部令87号", "财政部87号令"])
def test_document_number_variants_identify_the_article(number):
    result = ask(number + "第三十一条规定什么？")
    assert result["status"] == "answered"
    assert {(row["document_title"], row["article_or_position"]) for row in result["citations"]} == {
        ("政府采购货物和服务招标投标管理办法", "第三十一条")}


def test_new_document_number_does_not_inherit_previous_law_title():
    result = ask("《政府采购法》第二十二条和财政部87号令第三十一条分别规定什么？")
    assert {(row["document_title"], row["article_or_position"]) for row in result["citations"]} == {
        ("中华人民共和国政府采购法", "第二十二条"), ("政府采购货物和服务招标投标管理办法", "第三十一条")}


@pytest.mark.parametrize("question, article", [
    ("政府采购法第二十二条的供应商资格条件及相应证明材料是什么？", "第十七条"),
    ("政府采购法第二十二条规定什么，以及如何投诉？", "第十七条"),
])
def test_explicit_article_does_not_erase_additional_issues(question, article):
    result = ask(question)
    assert result["status"] == "answered"
    assert result["generation_mode"] == "agent_evidence_fallback"
    assert article in {row["article_or_position"] for row in result["citations"]}


def test_clarification_resolves_legal_regime_without_repeating_that_question():
    result = ask("我们是政府采购货物项目。", history=[{"role": "user", "content": "政府采购还是工程建设招标，投标文件超过150页可以否决吗？"}])
    assert result["domain"] == "government_procurement"
    assert result["status"] == "needs_clarification"
    assert "项目属于政府采购还是工程建设招标" not in result["clarifying_questions"]


def test_too_many_issues_requests_split_instead_of_silently_dropping_them():
    result = ask("政府采购中供应商资格条件及证明材料、同一品牌、质疑及质疑答复、投诉、中小企业政策分别是什么？")
    assert result["status"] == "needs_clarification"
    assert result["citations"] == []
    assert "拆分" in result["answer"]


def test_hybrid_health_does_not_treat_lexical_index_as_vector_ready(monkeypatch):
    monkeypatch.setattr("online_retrieval.retriever.RETRIEVAL_MODE", "hybrid")
    state = LegalRetriever().health()
    assert state["ready"] is False
    assert state["embedding_ready"] is False
