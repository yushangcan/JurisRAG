"""意图、领域和口语术语的确定性降级测试。"""

from agent.intent import deterministic_intent
from agent.schemas import IntentAnalysis


def test_unbracketed_law_and_article_are_preserved() -> None:
    analysis = deterministic_intent("政府采购法第二十二条规定了哪些条件？")
    assert analysis.primary_intent == "exact_reference"
    assert analysis.domain == "government_procurement"
    assert analysis.exact_references[0].document_title == "政府采购法"
    assert analysis.exact_references[0].article_number == "第二十二条"


def test_tender_words_alone_do_not_force_a_domain() -> None:
    analysis = deterministic_intent("招标文件约定投标文件超过150页可以否决，合法吗？")
    assert analysis.primary_intent == "legality_assessment"
    assert analysis.domain == "uncertain"
    assert len(analysis.legal_issues) >= 3
    assert "项目属于政府采购还是工程建设招标" in analysis.required_case_facts


def test_party_a_is_mapped_but_not_forced_to_one_legal_subject() -> None:
    analysis = deterministic_intent("招标文件里可以列甲方内部的制度吗？")
    assert analysis.domain == "uncertain"
    assert analysis.terminology_map["甲方"] == "采购人或招标人，具体取决于项目场景"


def test_clearly_unrelated_question_stops_before_retrieval() -> None:
    analysis = deterministic_intent("明天的天气怎么样？")
    assert analysis.in_scope is False
    assert analysis.primary_intent == "out_of_scope"


def test_intent_schema_normalizes_list_terminology_explanations() -> None:
    analysis = IntentAnalysis.model_validate(
        {
            "primary_intent": "rule_explanation",
            "normalized_question": "测试",
            "terminology_map": {"甲方": ["采购人", "招标人"]},
        }
    )
    assert analysis.terminology_map["甲方"] == "采购人；招标人"
