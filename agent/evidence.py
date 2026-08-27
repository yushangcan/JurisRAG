"""证据去重、父子选择和前端引用转换。"""

from __future__ import annotations

from typing import Any


def merge_child_results(
    existing: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    limit: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    """按 child ID 去重，并合并同一证据命中的检索通道。"""

    merged = {key: dict(value) for key, value in existing.items()}
    new_count = 0
    for row in results:
        chunk_id = str(row.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        if chunk_id not in merged:
            merged[chunk_id] = dict(row)
            new_count += 1
        else:
            old_channels = merged[chunk_id].get("retrieval_channels", [])
            new_channels = row.get("retrieval_channels", [])
            merged[chunk_id]["retrieval_channels"] = list(
                dict.fromkeys([*old_channels, *new_channels])
            )
    # Python 字典保留插入顺序；工具已经按相关性排序，先到的证据优先保留。
    return dict(list(merged.items())[:limit]), new_count


def merge_parent_results(
    existing: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    limit: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    merged = {key: dict(value) for key, value in existing.items()}
    new_count = 0
    for row in results:
        parent_id = str(row.get("parent_id", "")).strip()
        if not parent_id:
            continue
        if parent_id not in merged:
            merged[parent_id] = dict(row)
            new_count += 1
    return dict(list(merged.items())[:limit]), new_count


def select_parent_ids(
    child_evidence: dict[str, dict[str, Any]],
    limit: int,
) -> list[str]:
    """按 child 排名选取不同 parent，防止同一法条重复占满上下文。"""

    selected: list[str] = []
    for row in child_evidence.values():
        parent_id = str(row.get("parent_id", "")).strip()
        if parent_id and parent_id not in selected:
            selected.append(parent_id)
        if len(selected) >= limit:
            break
    return selected


def answer_evidence(
    parent_evidence: dict[str, dict[str, Any]],
    child_evidence: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """回答优先用完整 parent；MCP 父块失败时显式降级到 child。"""

    values = list(parent_evidence.values()) or list(child_evidence.values())
    return values[:limit]


def evidence_id(row: dict[str, Any]) -> str:
    return str(row.get("parent_id") or row.get("chunk_id", ""))


def citation_from_evidence(row: dict[str, Any], supports_issues: list[str]) -> dict[str, Any]:
    """转换为兼容当前 Web 的 citation，同时保留 Agent 争点映射。"""

    metadata = row.get("metadata", {})
    text = str(row.get("text", ""))
    return {
        "evidence_id": evidence_id(row),
        "document_title": metadata.get("title", row.get("title", "")),
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
        "reranker_score": row.get("scores", {}).get("reranker_score"),
        "supports_issues": supports_issues,
    }

