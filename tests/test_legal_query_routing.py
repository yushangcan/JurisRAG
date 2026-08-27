"""Fast unit tests for legal-question routing; no model or index is loaded."""

from online_retrieval.answerer import (
    _unsupported_generated_facts,
    _unsupported_legal_references,
)
from online_retrieval.query_expansion import legal_issue_variants
from online_retrieval.retriever import _requires_normative_expansion


def test_legality_question_forces_issue_expansion() -> None:
    """A high-similarity hit must not stop a question that needs legal analysis."""

    assert _requires_normative_expansion(
        "招标文件中约定投标文件超过150页可以否决其投标，合法吗？"
    )
    assert not _requires_normative_expansion("政府采购法第二十二条有哪些条件？")


def test_tender_document_expansion_contains_both_legal_sides() -> None:
    """Expansion should retrieve both rule-setting and rejection provisions."""

    text = " ".join(
        variant.text
        for variant in legal_issue_variants("投标文件超过150页能否否决？")
    )

    assert "项目特点" in text
    assert "实际需要" in text
    assert "实质性要求" in text
    assert "否决投标" in text


def test_unsupported_numeric_claim_is_detected() -> None:
    """The model must not invent an industry page-range absent from evidence."""

    unsupported = _unsupported_generated_facts(
        "行业惯例通常为30-80页，超过150页可以否决。",
        "投标文件超过150页可以否决吗？",
        "证据仅规定不得以不合理条件限制投标人。",
    )

    assert "30-80页" in unsupported
    assert "150页" not in unsupported


def test_uncited_article_is_detected() -> None:
    """An answer cannot cite an article absent from the returned evidence."""

    evidence = [
        {
            "metadata": {
                "title": "中华人民共和国招标投标法实施条例",
                "article_number": "第三十二条",
            }
        }
    ]

    unsupported = _unsupported_legal_references(
        "依据《招标投标法实施条例》第三十二条处理，而第五十一条另有规定。",
        evidence,
    )

    assert "未引用条号第五十一条" in unsupported
    assert "未引用条号第三十二条" not in unsupported
