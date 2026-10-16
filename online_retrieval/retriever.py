"""精确检索、中文 BM25 与可选向量融合；所有轮次使用同一证据评分方式。"""

from __future__ import annotations

import math
from importlib.util import find_spec
from array import array
from typing import Any

from agent.schemas import EvidenceNeed, QueryPlan
from shared.config import EMBEDDING_BACKEND, EMBEDDING_MODEL_PATH, RETRIEVAL_MODE, RERANKER_BACKEND, RERANKER_MODEL_PATH
from shared.legal_text import article_value, number_key, title_matches, tokens
from shared.store import LegalStore

STOP_TERMS = set(tokens("请问如何什么哪些是否可以应该根据规定政府采购招标投标供应商法律法规情况问题通常需要以及同时"))


def query_terms(question: str) -> set[str]:
    return set(tokens(question)) - STOP_TERMS


def coverage(question: str, evidence: str) -> float:
    terms = query_terms(question)
    return len(terms & set(tokens(evidence))) / max(len(terms), 1)


def matches_source(row: dict[str, Any], source) -> bool:
    return ((not source.document_title or title_matches(source.document_title, row["title"]))
            and (not source.article_number or article_value(source.article_number) == article_value(row.get("article_number", "")))
            and (not source.document_number or number_key(source.document_number) == number_key(row["metadata"].get("document_number", ""))))


class LegalRetriever:
    def __init__(self, store: LegalStore | None = None) -> None:
        self.store = store or LegalStore()

    def health(self) -> dict[str, Any]:
        """只检查配置、依赖和本地文件；模型是否成功推理由实际查询验证。"""
        state = self.store.health()
        embedding_ready = reranker_ready = True
        if RETRIEVAL_MODE == "hybrid":
            embedding_ready = state.get("embedding_backend") == EMBEDDING_BACKEND and bool(state.get("vectors_complete"))
            if EMBEDDING_BACKEND != "hash":
                model_present = (EMBEDDING_MODEL_PATH / "model_optimized.onnx").is_file() or any(EMBEDDING_MODEL_PATH.glob("snapshots/*/model_optimized.onnx"))
                embedding_ready = embedding_ready and model_present and find_spec("fastembed") is not None
            if RERANKER_BACKEND != "none":
                reranker_ready = (RERANKER_BACKEND == "bge_reranker_base_onnx"
                    and (RERANKER_MODEL_PATH / "onnx/model_quantized.onnx").is_file()
                    and (RERANKER_MODEL_PATH / "tokenizer.json").is_file()
                    and all(find_spec(name) is not None for name in ("numpy", "onnxruntime", "tokenizers")))
        ready = bool(state["ready"] and embedding_ready and reranker_ready)
        return {**state, "ready": ready, "status": "ok" if ready else "unavailable", "retrieval_mode": RETRIEVAL_MODE,
                "embedding_ready": embedding_ready, "reranker_ready": reranker_ready, "model_inference_checked": False}

    def search(self, queries: list[str], *, references: list | None = None, limit: int = 30) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for query in queries[:6]:
            for rank, row in enumerate(self.store.search(query, limit), start=1):
                value = fused.setdefault(row["chunk_id"], {**row, "score": 0.0})
                value["score"] += 1 / (40 + rank)
                value["query_coverage"] = max(value.get("query_coverage", 0), coverage(query, row["text"]))
        for ref in references or []:
            for row in self.store.exact(ref.document_title, ref.article_number, ref.document_number, limit):
                value = fused.setdefault(row["chunk_id"], {**row, "score": 0.0})
                value["score"] += 1.0 if ref.article_number or ref.document_number else 0.02
                value["retrieval_channels"] = list(dict.fromkeys([*value.get("retrieval_channels", []), "exact"]))
                value["exact_matches"] = row["exact_matches"]
        if RETRIEVAL_MODE == "hybrid":
            for rank, row in enumerate(self.semantic(queries[0], limit), start=1):
                value = fused.setdefault(row["chunk_id"], {**row, "score": 0.0})
                value["score"] += 1 / (40 + rank)
                value["retrieval_channels"] = list(dict.fromkeys([*value.get("retrieval_channels", []), "vector"]))
        rows = sorted(fused.values(), key=lambda row: (-row["score"], row["chunk_id"]))[:limit]
        if RERANKER_BACKEND != "none" and rows:
            from shared.reranker import rerank_children
            rows = rerank_children(queries[0], rows)
            rows = [{**row, "fusion_score": row["score"], "score": 1 / (40 + rank)} for rank, row in enumerate(rows, 1)]
        return rows

    def semantic(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if RETRIEVAL_MODE != "hybrid":
            raise ValueError("语义检索为可选能力，请启用 hybrid 并重建索引")
        from shared.embedding import embed_query
        vector = embed_query(query)
        with self.store.connect() as db:
            backend = db.execute("SELECT value FROM info WHERE key='embedding_backend'").fetchone()
            if not backend or backend[0] != EMBEDDING_BACKEND:
                raise ValueError("索引向量模型与当前配置不一致，请重建索引")
            scored = []
            norm = math.sqrt(sum(value * value for value in vector))
            for row in db.execute("SELECT * FROM nodes WHERE kind='child' AND vector IS NOT NULL"):
                other = array("f")
                other.frombytes(row["vector"])
                if len(other) != len(vector):
                    raise ValueError("索引向量维度不匹配")
                denominator = norm * math.sqrt(sum(value * value for value in other))
                score = sum(a * b for a, b in zip(vector, other)) / denominator if denominator else 0.0
                scored.append((score, self.store.row(row)))
        rows = []
        for score, row in sorted(scored, key=lambda item: -item[0])[:limit]:
            rows.append({**row, "vector_score": score, "retrieval_channels": ["vector"]})
        return rows

    def assess(self, plan: QueryPlan, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """正式引用逐条满足；场景逐争点核验；一般问题至少达到内容覆盖门槛。"""
        missing_refs = [ref.model_dump() for ref in plan.exact_references if not any(matches_source(row, ref) for row in rows)]
        issue_coverage = []
        for need in plan.needs:
            ids = [row["chunk_id"] for row in rows if self.supports(need, row)]
            issue_coverage.append({"issue": need.issue, "evidence_ids": ids})
        missing = [row["issue"] for row in issue_coverage if not row["evidence_ids"]]
        missing.extend("未找到指定法规引用：" + str(ref) for ref in missing_refs)
        return {"sufficient": bool(rows) and not missing, "missing_evidence": missing,
                "issue_coverage": issue_coverage, "formal_reference_satisfied": not missing_refs}

    @staticmethod
    def supports(need: EvidenceNeed, row: dict[str, Any]) -> bool:
        if need.sources:
            return any(matches_source(row, source) and (source.article_number or not need.query
                       or (coverage(need.query, row["text"]) >= 0.34 and len(query_terms(need.query) & set(tokens(row["text"]))) >= 2))
                       for source in need.sources)
        return coverage(need.query, row["text"]) >= 0.34 and len(query_terms(need.query) & set(tokens(row["text"]))) >= 2

    def parents(self, plan: QueryPlan, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """先覆盖各争点和正式引用，再补高分结果；后续轮次的新证据可替换旧结果。"""
        selected: list[str] = []
        requirements = [EvidenceNeed(issue="正式引用", query="", sources=[ref]) for ref in plan.exact_references]
        requirements.extend(plan.needs)
        relevant = [row for row in rows if any(self.supports(need, row) for need in requirements)]
        ranked = sorted(relevant, key=lambda row: (-float(row.get("score", 0)), row["chunk_id"]))
        for need in requirements:
            candidates = [row for row in ranked if self.supports(need, row)]
            if any(row["parent_id"] in selected for row in candidates):
                continue
            if candidates and candidates[0]["parent_id"] not in selected:
                selected.append(candidates[0]["parent_id"])
        required_count = len(selected)
        if required_count > top_k:
            return []  # 不把未放入最终上下文的来源算作已经覆盖。
        for row in ranked:
            if row["parent_id"] not in selected:
                selected.append(row["parent_id"])
            if len(selected) >= top_k:
                break
        parents = self.store.fetch(selected[:top_k], "parent")
        for parent in parents:
            child_rows = [row for row in ranked if row["parent_id"] == parent["parent_id"]]
            parent["retrieval_channels"] = list(dict.fromkeys(channel for row in child_rows for channel in row.get("retrieval_channels", [])))
            parent["score"] = max((float(row.get("score", 0)) for row in child_rows), default=0)
        return parents


def hybrid_search(question: str, top_k: int = 5) -> dict[str, Any]:
    """固定 RAG 基线：使用同一存储和证据门，只进行一轮检索。"""
    from agent.intent import plan_question
    plan = plan_question(question)
    retriever = LegalRetriever()
    if plan.boundary_reason:
        return {"results": [], "trace": {"mode": "sqlite_fts5", "evidence_sufficient": False, "refusal_reason": plan.boundary_reason}}
    rows = retriever.search([question], references=plan.exact_references)
    assessment = retriever.assess(plan, rows)
    parents = retriever.parents(plan, rows, top_k) if assessment["sufficient"] else []
    return {"results": parents, "trace": {"mode": "sqlite_fts5", "evidence_sufficient": bool(parents),
            "final_assessment": assessment, "refusal_reason": None if parents else "当前证据未覆盖问题所需的全部规则"}}
