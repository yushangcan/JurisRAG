"""混合检索、父子回填和可选 HyDE。

一次在线检索的完整链路是：

1. 子块集合执行向量检索，找到语义上相近的细粒度证据；
2. 同一个子块集合被全部扫描做轻量词法检索，优先保证“第二十二条”、
   “财库〔2020〕10号”和法规专有名词不会被哈希向量冲掉；
3. 原始问题和（可选的）HyDE 假设答案通过 RRF 合并；
4. 按 ``parent_article_id``/``parent_section_id`` 去重，再从父块集合取回
   完整法条或章节。返回给回答器的证据因此既精细命中，又不丢完整上下文。

本实现不依赖额外 BM25 包。当前语料规模下，内存中计算字符二元组 BM25
足够透明、可离线运行；以后语料达到百万级时，可以把这一层替换成倒排索引，
而不改变外层检索合同。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from shared.chroma_store import get_collection, get_parent_collection
from shared.config import (
    DEFAULT_TOP_K,
    ENABLE_MULTI_RECALL,
    EVIDENCE_MIN_CHANNEL_AGREEMENT,
    EVIDENCE_MIN_RERANKER_SCORE,
    EVIDENCE_MIN_WEIGHTED_COVERAGE,
    HYBRID_CANDIDATE_MULTIPLIER,
    RERANK_CANDIDATES,
    RERANKER_BACKEND,
    RETRY_RERANKER_SCORE,
    RRF_K,
)
from shared.embedding import embed_query
from shared.reranker import rerank_children
from online_retrieval.query_expansion import plan_retrieval, variants_for_trace


CHINESE_NUMBER = "一二三四五六七八九十百千万零〇两0-9"
ARTICLE_PATTERN = re.compile(rf"第\s*([{CHINESE_NUMBER}]+)\s*条")
DOCUMENT_NUMBER_PATTERN = re.compile(
    r"(?:[\u4e00-\u9fff]{1,8}\s*(?:〔|\[|\()\s*\d{4}\s*(?:〕|\]|\))\s*\d+\s*号"
    r"|[\u4e00-\u9fff]{1,12}(?:令|公告)\s*第?\s*\d+\s*号)"
)
TITLE_PATTERN = re.compile(r"《([^》]+)》")

# key 是 ``(Chroma 集合名, 子块数量)``。API 进程第一次查询时构建一次，
# 后续问题直接复用分词和文档频率；离线重建导致数量变化时会自动换 key。
_LEXICAL_INDEX_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def _compact(value: str) -> str:
    """去掉版式空格和常见标点，方便比较“第 九 条”和“第九条”。"""

    return re.sub(r"[\s\u3000，。！？；：、（）()《》“”\"'·-]", "", value).lower()


def _chinese_number(value: str) -> int | None:
    """把常见中文条号转换成整数；无法识别时返回 None 而不是猜测。"""

    value = value.strip()
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    section = 0
    number = 0
    found = False
    for character in value:
        if character in digits:
            number = digits[character]
            found = True
        elif character in units:
            found = True
            unit = units[character]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section, number = 0, 0
            else:
                section += (number or 1) * unit
                number = 0
        else:
            return None
    return total + section + number if found else None


def _article_value(article_number: str) -> int | None:
    """提取元数据中的“第X条”并统一中英文数字表达。"""

    match = ARTICLE_PATTERN.search(article_number)
    return _chinese_number(match.group(1)) if match else None


def _query_signals(query: str) -> dict[str, Any]:
    """提取需要精确处理的条号、文号和书名号标题。"""

    articles = [
        value
        for value in (_chinese_number(match.group(1)) for match in ARTICLE_PATTERN.finditer(query))
        if value is not None
    ]
    return {
        "article_values": set(articles),
        "document_numbers": {_compact(match.group(0)) for match in DOCUMENT_NUMBER_PATTERN.finditer(query)},
        "titles": {_compact(match.group(1)) for match in TITLE_PATTERN.finditer(query)},
    }


def _is_natural_language_query(query: str) -> bool:
    """区分自然语言问题与“竞争性磋商”这类直接检索词。

    这不是意图分类模型，只是一条可解释的路由规则。即使判断不完美，原始
    查询的词法和向量通道始终保留，因此 HyDE 不会成为唯一召回来源。
    """

    question_markers = ("什么", "哪些", "如何", "怎么", "是否", "能否", "可以", "需要", "为何", "为什么", "谁", "多少")
    return len(_compact(query)) > 12 or "?" in query or "？" in query or any(marker in query for marker in question_markers)


def _requires_normative_expansion(query: str) -> bool:
    """判断问题是否要求对某个条款或行为作合法性评价。

    相关性高不代表论证材料完整。例如“超过150页能否否决”很容易召回
    “否决投标”条文，却仍缺少“该页数条件能否设置”的依据。评价型问题因此
    必须检索一次法律争点，而不能仅依赖第一次分数门槛。
    """

    compact = _compact(query)
    markers = (
        "合法吗",
        "是否合法",
        "违法吗",
        "是否违法",
        "是否合规",
        "合规吗",
        "能否否决",
        "可以否决",
        "可否否决",
        "是否合理",
        "有效吗",
        "是否有效",
    )
    return any(marker in compact for marker in markers)


def _tokens(text: str) -> list[str]:
    """中文使用二/三元字符片段，数字和英文使用完整词，兼顾专有名词。"""

    compact = _compact(text)
    tokens: list[str] = []
    for word in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", compact):
        if re.fullmatch(r"[a-z0-9]+", word):
            tokens.append(word)
            continue
        tokens.extend(word[index : index + 2] for index in range(len(word) - 1))
        tokens.extend(word[index : index + 3] for index in range(len(word) - 2))
        if len(word) <= 4:
            tokens.append(word)
    return tokens


def _exact_score(query: str, metadata: dict[str, Any], signals: dict[str, Any]) -> tuple[float, list[str]]:
    """给条号、法规标题和文号明确加分，并记录命中的理由。"""

    score = 0.0
    reasons: list[str] = []
    article_value = _article_value(str(metadata.get("article_number", "")))
    if signals["article_values"] and article_value in signals["article_values"]:
        score += 8.0
        reasons.append("article_number")

    title = _compact(str(metadata.get("title", "")))
    query_compact = _compact(query)
    title_match = any(
        mentioned and (mentioned in title or title in mentioned)
        for mentioned in signals["titles"]
    )
    if not title_match and title:
        short_title = title.replace("中华人民共和国", "")
        title_match = len(short_title) >= 4 and short_title in query_compact
    if title_match:
        score += 12.0
        reasons.append("document_title")

    metadata_number = _compact(str(metadata.get("document_number", "")))
    if metadata_number and any(
        number in metadata_number or metadata_number in number
        for number in signals["document_numbers"]
    ):
        score += 16.0
        reasons.append("document_number")
    # 短查询通常就是用户直接输入的专有名词，例如“竞争性磋商”。若该
    # 完整短语出现在子块中，给予明确但低于“法规名+条号”的加分。
    if 2 <= len(query_compact) <= 12 and query_compact in _compact(str(metadata.get("search_text", ""))):
        score += 5.0
        reasons.append("exact_phrase")
    if len(reasons) >= 2:
        score += 10.0
    return score, reasons


def _build_lexical_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """预计算与查询无关的词频、文档长度和文档频率。"""

    document_counters = [Counter(_tokens(str(row.get("text", "")))) for row in rows]
    lengths = [sum(counter.values()) for counter in document_counters]
    average_length = sum(lengths) / max(len(lengths), 1)
    document_frequency: Counter[str] = Counter()
    for counter in document_counters:
        document_frequency.update(counter.keys())
    return {
        "rows": rows,
        "document_counters": document_counters,
        "lengths": lengths,
        "average_length": average_length,
        "document_frequency": document_frequency,
    }


def _cached_lexical_index(collection) -> dict[str, Any]:
    """读取 Chroma 子块并缓存词法索引；同名旧缓存会被及时移除。"""

    key = (str(collection.name), collection.count())
    if key in _LEXICAL_INDEX_CACHE:
        return _LEXICAL_INDEX_CACHE[key]
    corpus = collection.get(include=["documents", "metadatas"])
    rows = [
        {
            "chunk_id": identifier,
            "text": corpus["documents"][index],
            "distance": None,
            "metadata": corpus["metadatas"][index] or {},
        }
        for index, identifier in enumerate(corpus.get("ids", []))
    ]
    for stale_key in [cached_key for cached_key in _LEXICAL_INDEX_CACHE if cached_key[0] == key[0]]:
        _LEXICAL_INDEX_CACHE.pop(stale_key, None)
    _LEXICAL_INDEX_CACHE[key] = _build_lexical_index(rows)
    return _LEXICAL_INDEX_CACHE[key]


def _rank_lexical_index(query: str, index: dict[str, Any]) -> list[dict[str, Any]]:
    """在预计算词法索引上评分，并叠加精确条号/文号/短语分。"""

    rows = index["rows"]
    if not rows:
        return []
    query_counter = Counter(_tokens(query))
    document_counters = index["document_counters"]
    lengths = index["lengths"]
    average_length = index["average_length"]
    document_frequency = index["document_frequency"]
    signals = _query_signals(query)
    k1, b = 1.2, 0.75

    ranked: list[dict[str, Any]] = []
    for row, counter, length in zip(rows, document_counters, lengths):
        lexical_score = 0.0
        for token in query_counter:
            term_frequency = counter.get(token, 0)
            if not term_frequency:
                continue
            idf = math.log((len(rows) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5) + 1.0)
            normalization = 1.0 - b + b * length / max(average_length, 1.0)
            lexical_score += idf * (term_frequency * (k1 + 1.0) / (term_frequency + k1 * normalization))
        scoring_metadata = dict(row.get("metadata", {}))
        scoring_metadata["search_text"] = row.get("text", "")
        exact_score, reasons = _exact_score(query, scoring_metadata, signals)
        enriched = dict(row)
        enriched["lexical_score"] = lexical_score + exact_score
        enriched["exact_score"] = exact_score
        enriched["exact_matches"] = reasons
        ranked.append(enriched)
    return sorted(ranked, key=lambda row: (-row["lexical_score"], row.get("chunk_id", "")))


def _lexical_rank(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """无缓存兼容接口，主要供小型单元测试直接传入候选行。"""

    return _rank_lexical_index(query, _build_lexical_index(rows))


def _vector_rank(query: str, collection, candidate_k: int) -> list[dict[str, Any]]:
    """执行一次向量召回，并统一为内部行格式。"""

    result = collection.query(query_embeddings=[embed_query(query)], n_results=candidate_k)
    rows: list[dict[str, Any]] = []
    identifiers = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for index, chunk_id in enumerate(identifiers):
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": documents[index],
                "distance": distances[index] if index < len(distances) else None,
                "metadata": metadatas[index] or {},
            }
        )
    return rows


def _fuse_rankings(ranked_lists: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """用 Reciprocal Rank Fusion 合并原问题、HyDE 和词法排名。"""

    merged: dict[str, dict[str, Any]] = {}
    for channel, rows in ranked_lists:
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row["chunk_id"])
            target = merged.setdefault(chunk_id, {**row, "hybrid_score": 0.0, "retrieval_channels": []})
            target["hybrid_score"] += 1.0 / (RRF_K + rank)
            if channel not in target["retrieval_channels"]:
                target["retrieval_channels"].append(channel)
            if row.get("exact_score", 0.0) > target.get("exact_score", 0.0):
                target["exact_score"] = row["exact_score"]
                target["exact_matches"] = row.get("exact_matches", [])
            if target.get("distance") is None and row.get("distance") is not None:
                target["distance"] = row["distance"]
    return sorted(
        merged.values(),
        key=lambda row: (-row.get("exact_score", 0.0), -row["hybrid_score"], row["chunk_id"]),
    )


def _parent_id(row: dict[str, Any]) -> str:
    """获取文章/章节父 ID；旧索引没有父 ID 时退回自身。"""

    metadata = row.get("metadata", {})
    return str(
        metadata.get("parent_article_id")
        or metadata.get("parent_section_id")
        or row.get("chunk_id", "")
    )


def _promote_to_parents(ranked_children: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """子块命中后按父 ID 去重，并回填完整父块正文。"""

    if not ranked_children:
        return []
    parent_ids: list[str] = []
    best_children: dict[str, dict[str, Any]] = {}
    for child in ranked_children:
        parent_id = _parent_id(child)
        if parent_id not in best_children:
            parent_ids.append(parent_id)
            best_children[parent_id] = child
        if len(parent_ids) >= top_k:
            break

    parent_collection = get_parent_collection()
    result = parent_collection.get(ids=parent_ids, include=["documents", "metadatas"])
    parent_by_id: dict[str, dict[str, Any]] = {}
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    for index, identifier in enumerate(result.get("ids", [])):
        if index < len(documents):
            parent_by_id[str(identifier)] = {
                "text": documents[index],
                "metadata": metadatas[index] or {},
            }

    promoted: list[dict[str, Any]] = []
    for parent_id in parent_ids:
        child = best_children[parent_id]
        parent = parent_by_id.get(parent_id)
        if parent is None:
            promoted.append({**child, "parent_id": parent_id, "parent_missing": True})
            continue
        promoted.append(
            {
                "chunk_id": parent_id,
                "parent_id": parent_id,
                "text": parent["text"],
                "distance": child.get("distance"),
                "metadata": parent["metadata"],
                "hybrid_score": child.get("hybrid_score", 0.0),
                "exact_score": child.get("exact_score", 0.0),
                "exact_matches": child.get("exact_matches", []),
                "retrieval_channels": child.get("retrieval_channels", []),
                "reranker_score": child.get("reranker_score"),
                "matched_child_chunk_id": child.get("chunk_id", ""),
                "matched_child_text": child.get("text", ""),
            }
        )
    return promoted


def _combine_parent_evidence(
    issue_results: list[dict[str, Any]],
    direct_results: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """为评价型问题同时保留“规则是否合法”和“能否否决”两侧证据。

    普通全局排序很容易返回五条内容相近的否决规定。这里先给法律争点检索
    保留最多两个位置，再用原问题的直接检索补齐。仍按 parent ID 去重，最终
    数量不超过用户请求的 Top-K。
    """

    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    issue_slots = min(2, top_k)
    for row in [*issue_results[:issue_slots], *direct_results]:
        parent_id = str(row.get("parent_id") or row.get("chunk_id", ""))
        if parent_id in seen:
            continue
        seen.add(parent_id)
        combined.append(row)
        if len(combined) >= top_k:
            break
    return combined


def _rerank_fused(
    query: str,
    fused: list[dict[str, Any]],
    *,
    top_k: int,
    limit: int,
) -> tuple[list[dict[str, Any]], bool, int, str | None]:
    """精排 fused 头部，并把失败转成显式 trace，而不是中断检索。"""

    rerank_count = min(len(fused), max(limit, top_k))
    try:
        reranked_head = rerank_children(query, fused[:rerank_count])
        used = RERANKER_BACKEND != "none" and bool(reranked_head)
        return reranked_head + fused[rerank_count:], used, rerank_count if used else 0, None
    except Exception as exc:
        warning = (
            "Reranker 运行失败，已明确退回混合召回顺序: "
            f"{type(exc).__name__}: {exc}"
        )
        return fused, False, 0, warning


def _merge_reranked_candidates(
    initial_rows: list[dict[str, Any]],
    expanded_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并两轮候选，确保二次召回只扩充、不覆盖第一次结果。

    两轮都使用原问题计算 Reranker 分数，因此分数可以直接比较。同一 child
    在更多通道命中时合并通道名称，保留更高的精确匹配与混合检索分数。
    """

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in [*initial_rows, *expanded_rows]:
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id not in merged:
            merged[chunk_id] = dict(row)
            order.append(chunk_id)
            continue
        target = merged[chunk_id]
        target["retrieval_channels"] = list(
            dict.fromkeys(
                [
                    *target.get("retrieval_channels", []),
                    *row.get("retrieval_channels", []),
                ]
            )
        )
        if float(row.get("exact_score", 0.0)) > float(target.get("exact_score", 0.0)):
            target["exact_score"] = row.get("exact_score", 0.0)
            target["exact_matches"] = row.get("exact_matches", [])
        target["hybrid_score"] = max(
            float(target.get("hybrid_score", 0.0)),
            float(row.get("hybrid_score", 0.0)),
        )
        if row.get("reranker_score") is not None:
            target["reranker_score"] = row["reranker_score"]

    rows = [merged[chunk_id] for chunk_id in order]
    scored = [row for row in rows if row.get("reranker_score") is not None]
    unscored = [row for row in rows if row.get("reranker_score") is None]
    scored.sort(
        key=lambda row: (
            -int(float(row.get("exact_score", 0.0)) >= 16.0),
            -float(row.get("reranker_score", float("-inf"))),
            -float(row.get("hybrid_score", 0.0)),
            str(row.get("chunk_id", "")),
        )
    )
    return scored + unscored


def _weighted_query_coverage(
    query: str,
    ranked_children: list[dict[str, Any]],
    lexical_index: dict[str, Any],
) -> float:
    """计算问题高信息量词项被候选证据覆盖的加权比例。

    语料里罕见的词（如“注册资本”）权重更高，常见的“政府采购”权重较低；
    因此只命中领域通用词，不能轻易通过证据门。
    """

    query_tokens = set(_tokens(query))
    if not query_tokens or not ranked_children:
        return 0.0
    evidence_tokens = set(
        _tokens(" ".join(str(row.get("text", "")) for row in ranked_children[:3]))
    )
    document_count = max(len(lexical_index.get("rows", [])), 1)
    document_frequency = lexical_index.get("document_frequency", {})
    weights = {
        token: math.log((document_count + 1.0) / (float(document_frequency.get(token, 0)) + 1.0))
        + 1.0
        for token in query_tokens
    }
    total = sum(weights.values())
    matched = sum(weight for token, weight in weights.items() if token in evidence_tokens)
    return matched / total if total else 0.0


def _formal_reference_satisfied(
    signals: dict[str, Any], ranked_children: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    """精确法规名、条号和文号必须在同一个候选上真实匹配。"""

    required: set[str] = set()
    if signals["article_values"]:
        required.add("article_number")
    if signals["titles"]:
        required.add("document_title")
    if signals["document_numbers"]:
        required.add("document_number")
    # “政府采购法第22条”没有书名号，但精确打分能识别短法规名。只要某个
    # 候选识别到了标题，就把标题也纳入同候选组合校验，避免只对上同号条文。
    if required and any(
        "document_title" in row.get("exact_matches", []) for row in ranked_children[:10]
    ):
        required.add("document_title")
    if not required:
        return False, []
    satisfied = any(
        required.issubset(set(row.get("exact_matches", [])))
        for row in ranked_children[:10]
    )
    return satisfied, sorted(required)


def _assess_evidence(
    query: str,
    ranked_children: list[dict[str, Any]],
    lexical_index: dict[str, Any],
    signals: dict[str, Any],
) -> dict[str, Any]:
    """组合多个可解释信号，判断是否足以把候选交给回答器。"""

    if not ranked_children:
        return {
            "sufficient": False,
            "needs_retry": True,
            "reason": "没有召回任何候选证据",
            "top_reranker_score": None,
            "weighted_query_coverage": 0.0,
            "channel_agreement": 0,
            "formal_reference_required": [],
            "formal_reference_satisfied": False,
        }

    formal_requested = bool(
        signals["article_values"] or signals["document_numbers"] or signals["titles"]
    )
    formal_satisfied, formal_required = _formal_reference_satisfied(signals, ranked_children)
    top_score = ranked_children[0].get("reranker_score")
    coverage = _weighted_query_coverage(query, ranked_children, lexical_index)
    agreement = max(
        (len(row.get("retrieval_channels", [])) for row in ranked_children[:3]),
        default=0,
    )

    if formal_requested:
        sufficient = formal_satisfied
        reason = (
            "精确法规引用已在同一候选中匹配"
            if sufficient
            else "问题包含法规名、条号或文号，但召回结果未完成同候选精确匹配"
        )
        needs_retry = not sufficient
    else:
        score_ok = (
            top_score is None
            or float(top_score) >= EVIDENCE_MIN_RERANKER_SCORE
        )
        coverage_ok = coverage >= EVIDENCE_MIN_WEIGHTED_COVERAGE
        agreement_ok = agreement >= EVIDENCE_MIN_CHANNEL_AGREEMENT
        # 高分结果仍至少需要一部分高信息量词覆盖；不能只凭模型主观分数放行。
        high_confidence = (
            top_score is not None
            and float(top_score) >= RETRY_RERANKER_SCORE
            and coverage >= EVIDENCE_MIN_WEIGHTED_COVERAGE * 0.60
            and agreement_ok
        )
        sufficient = (score_ok and coverage_ok and agreement_ok) or high_confidence
        needs_retry = not sufficient
        reason = (
            "Reranker、加权词项覆盖和多通道一致性达到证据门"
            if sufficient
            else "候选相关性、问题词项覆盖或多通道一致性不足"
        )

    return {
        "sufficient": sufficient,
        "needs_retry": needs_retry,
        "reason": reason,
        "top_reranker_score": float(top_score) if top_score is not None else None,
        "weighted_query_coverage": round(coverage, 4),
        "channel_agreement": agreement,
        "formal_reference_required": formal_required,
        "formal_reference_satisfied": formal_satisfied,
    }


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """执行混合召回并返回证据与可审计的检索轨迹。"""

    if not query.strip():
        raise ValueError("问题不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    collection = get_collection()
    corpus_size = collection.count()
    if corpus_size == 0:
        raise RuntimeError("知识库为空，请先运行离线入库命令")
    candidate_k = min(corpus_size, max(top_k * HYBRID_CANDIDATE_MULTIPLIER, 20))

    lexical_index = _cached_lexical_index(collection)
    lexical_rows = _rank_lexical_index(query, lexical_index)[:candidate_k]
    ranked_lists: list[tuple[str, list[dict[str, Any]]]] = [("lexical", lexical_rows)]
    ranked_lists.append(("vector", _vector_rank(query, collection, candidate_k)))

    signals = _query_signals(query)
    formal_reference = bool(
        signals["article_values"] or signals["document_numbers"] or signals["titles"]
    )
    short_exact_term = any(row.get("exact_score", 0.0) >= 5.0 for row in lexical_rows[:3])
    exact_query = formal_reference or (
        short_exact_term and not _is_natural_language_query(query)
    )

    initial_fused = _fuse_rankings(ranked_lists)
    # 初次检索保持上一版的20候选精排，保证新增拒答逻辑不会悄悄改变已有基线。
    initial_ranked, initial_reranker_used, initial_rerank_count, initial_reranker_warning = (
        _rerank_fused(
            query,
            initial_fused,
            top_k=top_k,
            limit=RERANK_CANDIDATES,
        )
    )
    initial_assessment = _assess_evidence(
        query,
        initial_ranked,
        lexical_index,
        signals,
    )

    initial_plan, planned_variants, planning_warning = plan_retrieval(
        query, initial_ranked
    )
    normative_expansion = _requires_normative_expansion(query)
    planner_used = bool(initial_plan["planner_used"])
    retry_triggered = bool(
        ENABLE_MULTI_RECALL
        and (
            initial_plan["should_continue"]
            if planner_used
            else initial_assessment["needs_retry"] or normative_expansion
        )
    )
    variants = planned_variants if retry_triggered else []
    expansion_warning: str | None = planning_warning
    final_ranked = initial_ranked
    final_reranker_used = initial_reranker_used
    final_rerank_count = initial_rerank_count
    final_reranker_warning = initial_reranker_warning
    issue_ranked: list[dict[str, Any]] = []
    issue_rerank_count = 0
    issue_reranker_warning: str | None = None

    if retry_triggered:
        issue_ranked_lists: list[tuple[str, list[dict[str, Any]]]] = []
        issue_query = next(
            (
                variant.text
                for variant in variants
                if variant.style == "question"
                and variant.generated_by == "legal_issue_rule"
            ),
            next(
                (
                    variant.text
                    for variant in variants
                    if variant.style == "question"
                ),
                query,
            ),
        )
        for variant in variants:
            if variant.style == "question":
                lexical_variant_rows = _rank_lexical_index(
                    variant.text, lexical_index
                )[:candidate_k]
                vector_variant_rows = _vector_rank(
                    variant.text, collection, candidate_k
                )
                ranked_lists.append(
                    ("question_lexical", lexical_variant_rows)
                )
                ranked_lists.append(
                    ("question_vector", vector_variant_rows)
                )
                if variant.generated_by == "legal_issue_rule":
                    issue_ranked_lists.extend(
                        [
                            ("issue_question_lexical", lexical_variant_rows),
                            ("issue_question_vector", vector_variant_rows),
                        ]
                    )
            elif variant.style == "answer":
                ranked_lists.append(
                    ("answer_vector", _vector_rank(variant.text, collection, candidate_k))
                )
            elif variant.style == "keywords":
                keyword_variant_rows = _rank_lexical_index(
                    variant.text, lexical_index
                )[:candidate_k]
                ranked_lists.append(
                    ("keyword_lexical", keyword_variant_rows)
                )
                if variant.generated_by == "legal_issue_rule":
                    issue_ranked_lists.append(
                        ("issue_keyword_lexical", keyword_variant_rows)
                    )

        expanded_fused = _fuse_rankings(ranked_lists)
        expanded_ranked, expanded_reranker_used, expanded_rerank_count, final_reranker_warning = (
            _rerank_fused(
                query,
                expanded_fused,
                top_k=top_k,
                limit=RERANK_CANDIDATES,
            )
        )
        # 只合并两轮真正经过同一原问题打分的头部；后面的未打分候选按原顺序
        # 补在末尾。这样扩展通道不会把第一次已经找到的高质量 child 覆盖掉。
        final_ranked = _merge_reranked_candidates(
            initial_ranked[:initial_rerank_count] if initial_rerank_count else initial_ranked,
            expanded_ranked[:expanded_rerank_count] if expanded_rerank_count else expanded_ranked,
        )
        final_reranker_used = initial_reranker_used or expanded_reranker_used
        final_rerank_count = len(
            [row for row in final_ranked if row.get("reranker_score") is not None]
        )
        if issue_ranked_lists:
            issue_fused = _fuse_rankings(issue_ranked_lists)
            issue_ranked, _, issue_rerank_count, issue_reranker_warning = _rerank_fused(
                issue_query,
                issue_fused,
                top_k=min(2, top_k),
                limit=RERANK_CANDIDATES,
            )

    final_assessment = _assess_evidence(
        query,
        final_ranked,
        lexical_index,
        signals,
    )
    warnings = [
        warning
        for warning in (
            initial_reranker_warning,
            expansion_warning,
            final_reranker_warning
            if final_reranker_warning != initial_reranker_warning
            else None,
            issue_reranker_warning,
        )
        if warning
    ]
    hyde_used = any(variant.style == "answer" for variant in variants)
    direct_results = _promote_to_parents(final_ranked, top_k)
    issue_results = (
        _promote_to_parents(issue_ranked, min(2, top_k)) if issue_ranked else []
    )
    results = (
        _combine_parent_evidence(issue_results, direct_results, top_k)
        if retry_triggered and issue_results
        else direct_results
    )
    final_plan = initial_plan
    final_planning_warning: str | None = None
    if retry_triggered and planner_used:
        final_plan, _, final_planning_warning = plan_retrieval(query, results)
        if final_planning_warning:
            warnings.append(final_planning_warning)
    evidence_sufficient = (
        final_plan.get("answerability") in {"direct", "conditional"}
        if final_plan["planner_used"]
        else bool(final_assessment["sufficient"])
    )
    return {
        "results": results,
        "trace": {
            "mode": (
                "hybrid_multi_recall_rerank"
                if retry_triggered and final_reranker_used
                else "hybrid_multi_recall"
                if retry_triggered
                else "hybrid_rerank"
                if final_reranker_used
                else "hybrid"
            ),
            "exact_query": exact_query,
            "hyde_used": hyde_used,
            "multi_recall_enabled": ENABLE_MULTI_RECALL,
            "retry_triggered": retry_triggered,
            "retry_reason": (
                "llm_planner_requested_more_evidence"
                if retry_triggered and planner_used
                else "deterministic_fallback"
                if retry_triggered
                else None
            ),
            "initial_retrieval_plan": initial_plan,
            "final_retrieval_plan": final_plan,
            "retrieval_round_limit_reached": bool(
                retry_triggered
                and final_plan.get("planner_used")
                and final_plan.get("should_continue")
            ),
            "initial_assessment": initial_assessment,
            "final_assessment": final_assessment,
            "evidence_sufficient": evidence_sufficient,
            "refusal_reason": (
                None
                if evidence_sufficient
                else final_plan.get("reason")
                if final_plan.get("planner_used")
                else final_assessment["reason"]
            ),
            "query_variants": [query] + [variant.text for variant in variants],
            "query_variant_details": variants_for_trace(variants),
            "channels": [channel for channel, _ in ranked_lists],
            "reranker_backend": RERANKER_BACKEND,
            "reranker_used": final_reranker_used,
            "initial_rerank_candidates": initial_rerank_count,
            "rerank_candidates": final_rerank_count,
            "issue_rerank_candidates": issue_rerank_count,
            "reranker_warning": final_reranker_warning,
            "thresholds": {
                "retry_reranker_score": RETRY_RERANKER_SCORE,
                "minimum_reranker_score": EVIDENCE_MIN_RERANKER_SCORE,
                "minimum_weighted_coverage": EVIDENCE_MIN_WEIGHTED_COVERAGE,
                "minimum_channel_agreement": EVIDENCE_MIN_CHANNEL_AGREEMENT,
            },
            "warning": " | ".join(warnings) or None,
        },
    }


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """兼容旧接口，只返回证据列表；回答器可使用 ``hybrid_search`` 的轨迹。"""

    return hybrid_search(query, top_k=top_k)["results"]
