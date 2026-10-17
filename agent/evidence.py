"""引用转换与确定性事实检查；模型的归纳不能替代原文证据。"""
from __future__ import annotations
import re
from typing import Any

from shared.legal_text import DOCUMENT_NUMBER_PATTERN, article_value, number_key, title_matches

_QUANTIFIED_FACT_PATTERN = re.compile(
    r"[一二三四五六七八九十百千万零〇两0-9]+(?:\.\d+)?(?:\s*[—–-]\s*[一二三四五六七八九十百千万零〇两0-9]+)?\s*"
    r"(?:个工作日|工作日|万元|亿元|分钟|小时|页|天|日|年|元|%|％|倍|人)|[一二三四五六七八九十百0-9]+分之[一二三四五六七八九十百0-9]+"
)
_ARTICLE_REFERENCE_PATTERN = re.compile(r"第\s*[一二三四五六七八九十百千万零〇两0-9]+\s*条")


def _unsupported_generated_facts(
    answer: str,
    question: str,
    evidence_text: str,
) -> list[str]:
    """找出问题和证据中都没有出现的量化事实或占位内容。

    这不是完整的事实核验器，但能拦截法律回答中风险很高的“模型自创期限、
    金额、页数区间”和 ``XXX`` 占位文号。法条编号由引用约束负责。
    """

    supported_text = f"{question}\n{evidence_text}".replace(" ", "")
    unsupported = [
        match.group(0)
        for match in _QUANTIFIED_FACT_PATTERN.finditer(answer)
        if match.group(0).replace(" ", "") not in supported_text
    ]
    if "XXX" in answer.upper():
        unsupported.append("XXX占位内容")
    if "封闭式列举" in answer and "封闭式列举" not in evidence_text:
        unsupported.append("证据未支持否决情形属于封闭式列举")
    return list(dict.fromkeys(unsupported))


def _compact_reference(value: str) -> str:
    """删除引用中的空格和常见书名装饰，用于同一性比较。"""

    return re.sub(r"[\s《》]", "", value).lower()


def _unsupported_legal_references(
    answer: str,
    evidence: list[dict[str, Any]],
) -> list[str]:
    """检查回答中的法规标题和条号是否真实存在于本次引用集合。"""

    allowed_titles = {
        _compact_reference(str(item.get("metadata", {}).get("title", "")))
        for item in evidence
    }
    # 条文正文可能真实援引另一部法规或其他条号，不能把照录原文误判为编造。
    allowed_titles.update(_compact_reference(title) for item in evidence
                          for title in re.findall(r"《([^》]+)》", str(item.get("text", ""))))
    allowed_articles = {
        _compact_reference(str(item.get("metadata", {}).get("article_number", "")))
        for item in evidence
        if item.get("metadata", {}).get("article_number")
    }
    allowed_articles.update(_compact_reference(article) for item in evidence
                            for article in _ARTICLE_REFERENCE_PATTERN.findall(str(item.get("text", ""))))
    unsupported: list[str] = []
    for title in re.findall(r"《([^》]+)》", answer):
        compact_title = _compact_reference(title)
        if not any(
            title_matches(compact_title, allowed)
            for allowed in allowed_titles
            if allowed
        ):
            unsupported.append(f"未引用法规《{title}》")
    for article in _ARTICLE_REFERENCE_PATTERN.findall(answer):
        if article_value(article) not in {article_value(value) for value in allowed_articles}:
            unsupported.append(f"未引用条号{article}")
    allowed_numbers = {number_key(item.get("metadata", {}).get("document_number", "")) for item in evidence}
    allowed_numbers.update(number_key(match.group(0)) for item in evidence
                           for match in DOCUMENT_NUMBER_PATTERN.finditer(str(item.get("text", ""))))
    for match in DOCUMENT_NUMBER_PATTERN.finditer(answer):
        if number_key(match.group(0)) not in allowed_numbers:
            unsupported.append("未引用文号" + match.group(0))
    return list(dict.fromkeys(unsupported))


def citation_from_evidence(row: dict[str, Any], supports_issues: list[str]) -> dict[str, Any]:
    """转换为兼容当前 Web 的 citation，同时保留 Agent 争点映射。"""

    metadata = row.get("metadata", {})
    text = str(row.get("text", ""))
    return {
        "evidence_id": str(row.get("parent_id") or row.get("chunk_id", "")),
        "document_title": metadata.get("title", row.get("title", "")),
        "document_number": metadata.get("document_number", ""),
        "promulgated_date": metadata.get("promulgated_date", ""),
        "effective_date": metadata.get("effective_date", ""),
        "legal_status": metadata.get("legal_status", ""),
        "status_verified_at": metadata.get("status_verified_at", ""),
        "raw_sha256": metadata.get("sha256", ""),
        "category": metadata.get("category", ""),
        "article_or_position": (
            metadata.get("article_number")
            or metadata.get("section_path")
            or metadata.get("section_title")
            or metadata.get("chunk_index", "")
        ),
        "structure_type": metadata.get("structure_type", ""),
        "section_title": metadata.get("section_title", ""),
        "section_path": metadata.get("section_path", row.get("section_path", "")),
        "chapter": metadata.get("chapter", ""),
        "page_number": metadata.get("page_number", ""),
        "page_start": metadata.get("page_start", row.get("page_start", "")),
        "page_end": metadata.get("page_end", row.get("page_end", "")),
        "page_numbers": metadata.get("page_numbers", ""),
        "quote": text,
        "source_portal": metadata.get("source_portal", ""),
        "source_url": metadata.get("source_url", row.get("source_url", "")),
        "attachment_url": metadata.get("attachment_url", ""),
        "parent_id": row.get("parent_id", ""),
        "matched_child_chunk_id": row.get("matched_child_chunk_id", ""),
        "matched_child_quote": row.get("matched_child_text", ""),
        "retrieval_channels": row.get("retrieval_channels", []),
        "exact_matches": row.get("exact_matches", []),
        "hybrid_score": row.get("scores", {}).get("hybrid_score", 0.0),
        "reranker_score": row.get("reranker_score"),
        "supports_issues": supports_issues,
    }

