"""供 MCP 工具调用的法律检索业务函数。

这一层刻意不依赖 MCP SDK。这样做有两个好处：

1. 普通 Python 单元测试可以直接调用函数，不必启动协议服务器；
2. 将来把同一组能力接入 LangGraph、HTTP API 或命令行时无需复制检索代码。

所有函数都是只读操作，不会修改法规原文或 Chroma 数据库。
"""

from __future__ import annotations

from typing import Any

from online_retrieval.query_expansion import build_retry_variants, plan_retrieval
from online_retrieval.retriever import (
    _cached_lexical_index,
    _compact,
    _fuse_rankings,
    _rank_lexical_index,
    _rerank_fused,
    _vector_rank,
)
from shared.chroma_store import get_collection, get_parent_collection
from shared.config import HYBRID_CANDIDATE_MULTIPLIER, RERANK_CANDIDATES


# Agent 如果一次索取过多全文，会迅速挤占模型上下文。这里在工具边界统一限制，
# 而不是依赖每个 Agent 都能自觉传入合理参数。
MAX_TOP_K = 20
MAX_MULTI_QUERIES = 5
MAX_LOOKUP_IDS = 20


def _validate_text(value: str, field_name: str) -> str:
    """去掉首尾空白，并用对小白友好的错误说明拒绝空参数。"""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name}不能为空")
    return cleaned


def _validate_top_k(top_k: int) -> int:
    """限制单次返回数量，防止 Agent 误传很大的数字。"""

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k 必须在 1 到 {MAX_TOP_K} 之间")
    return top_k


def _candidate_k(corpus_size: int, top_k: int) -> int:
    """先多召回候选，再从中选择最终结果。"""

    return min(corpus_size, max(top_k * HYBRID_CANDIDATE_MULTIPLIER, 20))


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """只复制可序列化元数据，避免把 Chroma 内部对象交给 MCP。"""

    return {str(key): value for key, value in metadata.items()}


def _normalized_document_title(value: str) -> str:
    """统一法规标题写法，但不把“法律”和“实施条例”误认为同一文件。"""

    compact = _compact(value)
    return compact.removeprefix("中华人民共和国")


def _child_evidence(row: dict[str, Any], rank: int) -> dict[str, Any]:
    """把不同检索通道的内部行统一为 Agent 易理解的证据结构。"""

    metadata = _safe_metadata(row.get("metadata", {}))
    parent_id = str(
        metadata.get("parent_article_id")
        or metadata.get("parent_section_id")
        or row.get("chunk_id", "")
    )
    return {
        "rank": rank,
        "chunk_id": str(row.get("chunk_id", "")),
        "parent_id": parent_id,
        "text": str(row.get("text", "")),
        "title": str(metadata.get("title", "")),
        "article_number": str(metadata.get("article_number", "")),
        "section_path": str(metadata.get("section_path", "")),
        "page_start": str(metadata.get("page_start", "")),
        "page_end": str(metadata.get("page_end", "")),
        "source_url": str(metadata.get("source_url", "")),
        "document_number": str(metadata.get("document_number", "")),
        "legal_status": str(metadata.get("legal_status", "")),
        "scores": {
            "vector_distance": row.get("distance"),
            "lexical_score": row.get("lexical_score"),
            "exact_score": row.get("exact_score"),
            "reranker_score": row.get("reranker_score"),
            "hybrid_score": row.get("hybrid_score"),
        },
        "exact_matches": list(row.get("exact_matches", [])),
        "retrieval_channels": list(row.get("retrieval_channels", [])),
        "metadata": metadata,
    }


def _parent_evidence(row: dict[str, Any], rank: int) -> dict[str, Any]:
    """统一父块输出，同时保留是哪个细粒度 child 命中的。"""

    metadata = _safe_metadata(row.get("metadata", {}))
    return {
        "rank": rank,
        "parent_id": str(row.get("parent_id") or row.get("chunk_id", "")),
        "text": str(row.get("text", "")),
        "title": str(metadata.get("title", "")),
        "article_number": str(metadata.get("article_number", "")),
        "section_path": str(metadata.get("section_path", "")),
        "page_start": str(metadata.get("page_start", "")),
        "page_end": str(metadata.get("page_end", "")),
        "source_url": str(metadata.get("source_url", "")),
        "matched_child_chunk_id": str(row.get("matched_child_chunk_id", "")),
        "matched_child_text": str(row.get("matched_child_text", "")),
        "metadata": metadata,
    }


def exact_legal_search(
    document_title: str = "",
    article_number: str = "",
    document_number: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    """按法规名称、条号或文号精确查找 child 法条。

    与语义检索不同，这个工具不会用近义表达猜测结果。调用者填写了哪些字段，
    候选就必须同时满足哪些字段，因此适合“《政府采购法》第二十二条”一类查询。
    """

    top_k = _validate_top_k(top_k)
    document_title = document_title.strip()
    article_number = article_number.strip()
    document_number = document_number.strip()
    if not any((document_title, article_number, document_number)):
        raise ValueError("法规名称、条号和文号至少填写一个")

    # 书名号使现有精确信号提取器明确知道这部分是法规名称。
    parts = [f"《{document_title}》" if document_title else "", article_number, document_number]
    query = " ".join(part for part in parts if part)
    collection = get_collection()
    ranked = _rank_lexical_index(query, _cached_lexical_index(collection))

    required_matches: set[str] = set()
    if document_title:
        required_matches.add("document_title")
    if article_number:
        required_matches.add("article_number")
    if document_number:
        required_matches.add("document_number")
    matches: list[dict[str, Any]] = []
    for row in ranked:
        metadata = row.get("metadata", {})
        if not required_matches.issubset(set(row.get("exact_matches", []))):
            continue
        # 底层召回允许标题包含关系，以提高普通自然语言检索的召回率；但 MCP
        # 的“精确检索”必须再做一次标题等值检查。例如查询《政府采购法》时，
        # 不能把《政府采购法实施条例》的同号条文当成精确结果。
        if document_title and _normalized_document_title(
            str(metadata.get("title", ""))
        ) != _normalized_document_title(document_title):
            continue
        if document_number and _compact(
            str(metadata.get("document_number", ""))
        ) != _compact(document_number):
            continue
        row["retrieval_channels"] = ["exact_lexical"]
        matches.append(row)
        if len(matches) >= top_k:
            break
    return {
        "tool": "exact_legal_search",
        "query": query,
        "required_matches": sorted(required_matches),
        "count": len(matches),
        "results": [_child_evidence(row, rank) for rank, row in enumerate(matches, 1)],
    }


def keyword_search(keywords: str, top_k: int = 5) -> dict[str, Any]:
    """使用本地 BM25 与精确短语加分进行关键词检索。"""

    keywords = _validate_text(keywords, "keywords")
    top_k = _validate_top_k(top_k)
    collection = get_collection()
    ranked = _rank_lexical_index(keywords, _cached_lexical_index(collection))
    # 零分意味着候选与关键词没有任何可解释的词法交集，不应返回给 Agent。
    matches = [row for row in ranked if float(row.get("lexical_score", 0.0)) > 0.0][:top_k]
    return {
        "tool": "keyword_search",
        "query": keywords,
        "count": len(matches),
        "results": [_child_evidence(row, rank) for rank, row in enumerate(matches, 1)],
    }


def semantic_search(query: str, top_k: int = 5) -> dict[str, Any]:
    """使用中文 BGE Embedding 搜索自然语言近义表达。"""

    query = _validate_text(query, "query")
    top_k = _validate_top_k(top_k)
    collection = get_collection()
    rows = _vector_rank(query, collection, _candidate_k(collection.count(), top_k))[:top_k]
    for row in rows:
        row["retrieval_channels"] = ["semantic_vector"]
    return {
        "tool": "semantic_search",
        "query": query,
        "embedding_role": "query",
        "count": len(rows),
        "results": [_child_evidence(row, rank) for rank, row in enumerate(rows, 1)],
    }


def hyde_search(question: str, top_k: int = 5) -> dict[str, Any]:
    """生成中性假设答案，并用它做向量检索，再按原问题重排。

    HyDE 的文本仅作为检索查询，绝不能直接作为用户答案或法律证据。模型不可用
    时，``build_retry_variants`` 会返回不包含虚构条号的确定性降级表达。
    """

    question = _validate_text(question, "question")
    top_k = _validate_top_k(top_k)
    variants, warning = build_retry_variants(question)
    answer_variant = next((item for item in variants if item.style == "answer"), None)
    if answer_variant is None:
        return {
            "tool": "hyde_search",
            "question": question,
            "hypothetical_query": "",
            "warning": warning or "未生成可用的 HyDE 假设答案",
            "count": 0,
            "results": [],
        }

    collection = get_collection()
    candidate_k = _candidate_k(collection.count(), top_k)
    rows = _vector_rank(answer_variant.text, collection, candidate_k)
    for row in rows:
        row["retrieval_channels"] = ["hyde_vector"]
    reranked, reranker_used, rerank_count, reranker_warning = _rerank_fused(
        question,
        rows,
        top_k=top_k,
        limit=RERANK_CANDIDATES,
    )
    selected = reranked[:top_k]
    warnings = [value for value in (warning, reranker_warning) if value]
    return {
        "tool": "hyde_search",
        "question": question,
        "hypothetical_query": answer_variant.text,
        "generated_by": answer_variant.generated_by,
        "reranker_used": reranker_used,
        "rerank_candidates": rerank_count,
        "warning": " | ".join(warnings) or None,
        "count": len(selected),
        "results": [_child_evidence(row, rank) for rank, row in enumerate(selected, 1)],
    }


def multi_query_search(queries: list[str], top_k: int = 5) -> dict[str, Any]:
    """合并最多五个争点查询的 BM25 与向量结果。

    查询由上层 Agent 根据法律争点提供。本工具只执行检索、RRF 融合和重排，
    不替 Agent 擅自扩展问题，因而每一条检索词都可以被审计。
    """

    top_k = _validate_top_k(top_k)
    cleaned = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
    if not cleaned:
        raise ValueError("queries 至少需要一个非空查询")
    if len(cleaned) > MAX_MULTI_QUERIES:
        raise ValueError(f"queries 最多允许 {MAX_MULTI_QUERIES} 个查询")

    collection = get_collection()
    lexical_index = _cached_lexical_index(collection)
    candidate_k = _candidate_k(collection.count(), top_k)
    ranked_lists: list[tuple[str, list[dict[str, Any]]]] = []
    for index, query in enumerate(cleaned, 1):
        ranked_lists.append(
            (f"query_{index}_keyword", _rank_lexical_index(query, lexical_index)[:candidate_k])
        )
        ranked_lists.append(
            (f"query_{index}_semantic", _vector_rank(query, collection, candidate_k))
        )
    fused = _fuse_rankings(ranked_lists)
    reranked, reranker_used, rerank_count, warning = _rerank_fused(
        cleaned[0], fused, top_k=top_k, limit=RERANK_CANDIDATES
    )
    selected = reranked[:top_k]
    return {
        "tool": "multi_query_search",
        "queries": cleaned,
        "fusion": "RRF",
        "reranker_query": cleaned[0],
        "reranker_used": reranker_used,
        "rerank_candidates": rerank_count,
        "warning": warning,
        "count": len(selected),
        "results": [_child_evidence(row, rank) for rank, row in enumerate(selected, 1)],
    }


def fetch_parent_context(parent_ids: list[str]) -> dict[str, Any]:
    """根据 child 返回的 parent_id 读取完整法条或章节。"""

    cleaned = list(dict.fromkeys(value.strip() for value in parent_ids if value.strip()))
    if not cleaned:
        raise ValueError("parent_ids 至少需要一个 ID")
    if len(cleaned) > MAX_LOOKUP_IDS:
        raise ValueError(f"parent_ids 最多允许 {MAX_LOOKUP_IDS} 个 ID")

    parent_collection = get_parent_collection()
    raw = parent_collection.get(ids=cleaned, include=["documents", "metadatas"])
    found: dict[str, dict[str, Any]] = {}
    for index, identifier in enumerate(raw.get("ids", [])):
        found[str(identifier)] = {
            "chunk_id": str(identifier),
            "parent_id": str(identifier),
            "text": raw.get("documents", [])[index],
            "metadata": raw.get("metadatas", [])[index] or {},
        }
    ordered = [found[value] for value in cleaned if value in found]
    return {
        "tool": "fetch_parent_context",
        "requested_ids": cleaned,
        "missing_ids": [value for value in cleaned if value not in found],
        "count": len(ordered),
        "results": [_parent_evidence(row, rank) for rank, row in enumerate(ordered, 1)],
    }


def lookup_document_scope(title_or_number: str, top_k: int = 5) -> dict[str, Any]:
    """查询法规的适用范围辅助元数据，如类别、效力状态和生效日期。

    这里只返回语料中真实保存的元数据；缺失字段保持空字符串，不让 Agent 猜测。
    """

    title_or_number = _validate_text(title_or_number, "title_or_number")
    top_k = _validate_top_k(top_k)
    collection = get_collection()
    ranked = _rank_lexical_index(title_or_number, _cached_lexical_index(collection))
    query_key = _compact(title_or_number)
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        metadata = _safe_metadata(row.get("metadata", {}))
        title = str(metadata.get("title", ""))
        number = str(metadata.get("document_number", ""))
        if query_key not in _compact(title) and query_key not in _compact(number):
            continue
        document_id = str(metadata.get("document_id") or title or number)
        if document_id in seen:
            continue
        seen.add(document_id)
        documents.append(
            {
                "document_id": document_id,
                "title": title,
                "document_number": number,
                "category": str(metadata.get("category", "")),
                "legal_status": str(metadata.get("legal_status", "")),
                "promulgated_date": str(metadata.get("promulgated_date", "")),
                "effective_date": str(metadata.get("effective_date", "")),
                "revision_note": str(metadata.get("revision_note", "")),
                "source_portal": str(metadata.get("source_portal", "")),
                "source_url": str(metadata.get("source_url", "")),
                "verification_url": str(metadata.get("verification_url", "")),
            }
        )
        if len(documents) >= top_k:
            break
    return {
        "tool": "lookup_document_scope",
        "query": title_or_number,
        "count": len(documents),
        "results": documents,
        "notice": "适用范围仍需结合具体项目事实判断；本工具只返回库内元数据。",
    }


def _fetch_evidence_rows(chunk_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """同时查询 child/parent 两个集合，并按请求顺序还原证据。"""

    found: dict[str, dict[str, Any]] = {}
    for collection in (get_collection(), get_parent_collection()):
        raw = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        for index, identifier in enumerate(raw.get("ids", [])):
            identifier = str(identifier)
            if identifier in found:
                continue
            found[identifier] = {
                "chunk_id": identifier,
                "text": raw.get("documents", [])[index],
                "metadata": raw.get("metadatas", [])[index] or {},
            }
    return [found[value] for value in chunk_ids if value in found], [
        value for value in chunk_ids if value not in found
    ]


def get_source_metadata(chunk_ids: list[str]) -> dict[str, Any]:
    """读取证据的法规名、法条、页码、官方链接和效力状态。"""

    cleaned = list(dict.fromkeys(value.strip() for value in chunk_ids if value.strip()))
    if not cleaned:
        raise ValueError("chunk_ids 至少需要一个 ID")
    if len(cleaned) > MAX_LOOKUP_IDS:
        raise ValueError(f"chunk_ids 最多允许 {MAX_LOOKUP_IDS} 个 ID")
    rows, missing = _fetch_evidence_rows(cleaned)
    results = []
    for row in rows:
        metadata = _safe_metadata(row.get("metadata", {}))
        results.append(
            {
                "chunk_id": row["chunk_id"],
                "title": str(metadata.get("title", "")),
                "article_number": str(metadata.get("article_number", "")),
                "section_path": str(metadata.get("section_path", "")),
                "page_start": str(metadata.get("page_start", "")),
                "page_end": str(metadata.get("page_end", "")),
                "page_numbers": str(metadata.get("page_numbers", "")),
                "document_number": str(metadata.get("document_number", "")),
                "legal_status": str(metadata.get("legal_status", "")),
                "source_portal": str(metadata.get("source_portal", "")),
                "source_url": str(metadata.get("source_url", "")),
                "verification_url": str(metadata.get("verification_url", "")),
                "raw_file": str(metadata.get("raw_file", "")),
                "markdown_file": str(metadata.get("markdown_file", "")),
            }
        )
    return {
        "tool": "get_source_metadata",
        "requested_ids": cleaned,
        "missing_ids": missing,
        "count": len(results),
        "results": results,
    }


def check_issue_coverage(question: str, evidence_ids: list[str]) -> dict[str, Any]:
    """让规划模型逐项检查现有证据能否覆盖问题的法律争点。

    模型只能引用传入的证据序号。返回的 ``should_continue`` 与
    ``answerability`` 会被上层 Agent 用来决定继续检索、条件式回答或拒答。
    """

    question = _validate_text(question, "question")
    cleaned = list(dict.fromkeys(value.strip() for value in evidence_ids if value.strip()))
    if not cleaned:
        raise ValueError("evidence_ids 至少需要一个 ID")
    if len(cleaned) > MAX_LOOKUP_IDS:
        raise ValueError(f"evidence_ids 最多允许 {MAX_LOOKUP_IDS} 个 ID")
    evidence, missing = _fetch_evidence_rows(cleaned)
    plan, variants, warning = plan_retrieval(question, evidence)
    # 规划器的 candidate_ids 使用从 1 开始的短编号以减少提示词长度。这里显式
    # 返回编号与真实 Chroma ID 的映射，Agent 无需依赖隐含的列表顺序。
    evidence_index = [
        {
            "candidate_id": index,
            "evidence_id": row["chunk_id"],
            "title": str(row.get("metadata", {}).get("title", "")),
            "article_number": str(row.get("metadata", {}).get("article_number", "")),
        }
        for index, row in enumerate(evidence, 1)
    ]
    return {
        "tool": "check_issue_coverage",
        "question": question,
        "evidence_ids": [row["chunk_id"] for row in evidence],
        "evidence_index": evidence_index,
        "missing_ids": missing,
        "planner_used": bool(plan.get("planner_used")),
        "should_continue": bool(plan.get("should_continue", True)),
        "answerability": str(plan.get("answerability", "unknown")),
        "domain": str(plan.get("domain", "unknown")),
        "legal_issues": list(plan.get("legal_issues", [])),
        "issue_coverage": list(plan.get("issue_coverage", [])),
        "uncovered_issues": list(plan.get("uncovered_issues", [])),
        "missing_evidence": list(plan.get("missing_evidence", [])),
        "reason": str(plan.get("reason", "")),
        "suggested_queries": [
            {"style": item.style, "text": item.text, "generated_by": item.generated_by}
            for item in variants
        ],
        "warning": warning,
    }
