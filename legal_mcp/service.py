"""可选 MCP 适配器复用应用的检索业务，主 Agent 不需要 MCP 服务。"""

from __future__ import annotations

from typing import Any

from agent.intent import plan_question
from online_retrieval.retriever import LegalRetriever
from shared.store import LegalStore


def _limit(top_k: int) -> None:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k 必须在 1 到 20 之间")


def _text(value: str) -> str:
    if not value.strip() or len(value) > 2000:
        raise ValueError("检索文本不能为空或超过2000字")
    return value.strip()


def _result(tool: str, rows: list[dict], **extra) -> dict:
    return {"tool": tool, "count": len(rows), "results": rows, **extra}


def exact_legal_search(document_title: str = "", article_number: str = "", document_number: str = "", top_k: int = 5) -> dict[str, Any]:
    _limit(top_k)
    if any(len(value) > 300 for value in (document_title, article_number, document_number)):
        raise ValueError("精确引用字段过长")
    rows = LegalStore().exact(document_title.strip(), article_number.strip(), document_number.strip(), top_k)
    return _result("exact_legal_search", rows, required_matches=[key for key, value in
                   (("document_title", document_title), ("article_number", article_number), ("document_number", document_number)) if value])


def keyword_search(keywords: str, top_k: int = 5) -> dict[str, Any]:
    _limit(top_k)
    return _result("keyword_search", LegalStore().search(_text(keywords), top_k))


def semantic_search(query: str, top_k: int = 5) -> dict[str, Any]:
    _limit(top_k)
    return _result("semantic_search", LegalRetriever().semantic(_text(query), top_k))


def multi_query_search(queries: list[str], top_k: int = 5) -> dict[str, Any]:
    _limit(top_k)
    if not 1 <= len(queries) <= 5:
        raise ValueError("多查询需要1到5条查询")
    return _result("multi_query_search", LegalRetriever().search([_text(query) for query in queries], limit=top_k))


def fetch_parent_context(parent_ids: list[str]) -> dict[str, Any]:
    if not 1 <= len(parent_ids) <= 20 or any(not value.strip() or len(value) > 300 for value in parent_ids):
        raise ValueError("需要1到20个有效证据ID")
    rows = LegalStore().fetch(parent_ids, "parent")
    found = {row["parent_id"] for row in rows}
    return _result("fetch_parent_context", rows, missing_ids=[value for value in parent_ids if value not in found])


def lookup_document_scope(title_or_number: str, top_k: int = 5) -> dict[str, Any]:
    _limit(top_k)
    return _result("lookup_document_scope", LegalStore().documents(_text(title_or_number))[:top_k])


def check_issue_coverage(question: str, evidence_ids: list[str]) -> dict[str, Any]:
    if not 1 <= len(evidence_ids) <= 20:
        raise ValueError("需要1到20个证据ID")
    plan = plan_question(_text(question))
    rows = LegalStore().fetch(evidence_ids)
    assessment = LegalRetriever().assess(plan, rows)
    if plan.boundary_reason:
        assessment["sufficient"] = False
        assessment["missing_evidence"] = [plan.boundary_reason]
    return {"tool": "check_issue_coverage", **assessment,
            "answerability": "direct" if assessment["sufficient"] else "insufficient",
            "should_continue": not assessment["sufficient"]}
