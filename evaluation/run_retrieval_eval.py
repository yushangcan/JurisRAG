r"""Run a reproducible retrieval baseline with deterministic and RAGAS metrics.

This first evaluation stage does not ask an LLM to judge legal correctness.
It uses human-labelled article IDs and RAGAS 0.4.3 ID-based context metrics,
then stores every retrieved result so low-scoring cases remain inspectable.

Run from the project root with the independent evaluation interpreter::

    .\.venv-eval\Scripts\python.exe -m evaluation.run_retrieval_eval
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import ragas
from ragas import SingleTurnSample

# RAGAS 0.4.3 documents these two classes under ``metrics.collections``, but
# the published 0.4.3 wheel only exposes them through its compatibility export.
# The classes themselves are real RAGAS implementations; suppress only this
# known import deprecation so the result output stays readable.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

from online_retrieval.retriever import hybrid_search
from shared.chroma_store import get_collection
from shared.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_BACKEND,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    MANIFEST_PATH,
    RERANKER_BACKEND,
    RERANK_CANDIDATES,
    get_llm_settings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "legal_rag_gold_v2.jsonl"
DEFAULT_RESULT = PROJECT_ROOT / "evaluation" / "results" / "legal_rag_gold_v2_top5.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read JSONL gold cases and reject duplicate identifiers."""

    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("评测集存在重复 case_id")
    if not cases:
        raise ValueError("评测集不能为空")
    return cases


def git_commit() -> str:
    """Return the evaluated code revision without modifying Git state."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    ).strip()


def git_is_dirty() -> bool:
    """记录评测是否包含未提交代码，避免只写 HEAD 造成可复现性误解。"""

    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    )
    return bool(status.strip())


def percentile(values: list[float], fraction: float) -> float:
    """Compute a small-sample linear percentile without adding NumPy logic."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_index_metadata() -> list[dict[str, Any]]:
    """Load all Chroma metadata so gold article labels can resolve to real IDs."""

    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("向量索引为空，请先运行 offline_ingestion.build_index")
    result = collection.get(limit=collection.count(), include=["metadatas"])
    return list(result["metadatas"])


def resolve_gold_parent_ids(case: dict[str, Any], all_metadata: list[dict[str, Any]]) -> list[str]:
    """Resolve human raw-file/article labels to IDs actually stored in Chroma."""

    resolved: list[str] = []
    for evidence in case["gold_evidence"]:
        matches = {
            str(metadata.get("parent_article_id", ""))
            for metadata in all_metadata
            if metadata.get("raw_file") == evidence["raw_file"]
            and metadata.get("article_number") == evidence["article_number"]
            and metadata.get("parent_article_id")
        }
        if len(matches) != 1:
            raise ValueError(
                f"{case['case_id']} 的 gold 证据无法唯一解析: "
                f"{evidence['raw_file']} {evidence['article_number']} -> {sorted(matches)}"
            )
        resolved.extend(matches)
    return list(dict.fromkeys(resolved))


def keys_for_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Extract parent/article/document IDs from ranked retrieval rows."""

    return {
        "parents": [str(row["metadata"].get("parent_article_id", "")) for row in rows],
        "articles": [
            f"{row['metadata'].get('raw_file', '')}::{row['metadata'].get('article_number', '')}"
            for row in rows
        ],
        "documents": [str(row["metadata"].get("raw_file", "")) for row in rows],
    }


def mean(values: list[float]) -> float:
    """Return a JSON-friendly arithmetic mean."""

    return statistics.fmean(values) if values else 0.0


def run_case(
    case: dict[str, Any], all_metadata: list[dict[str, Any]], top_k: int
) -> dict[str, Any]:
    """Retrieve one question and calculate inspectable per-case metrics."""

    gold_parent_ids = resolve_gold_parent_ids(case, all_metadata)
    gold_parent_set = set(gold_parent_ids)
    gold_articles = {
        f"{evidence['raw_file']}::{evidence['article_number']}" for evidence in case["gold_evidence"]
    }
    gold_documents = {evidence["raw_file"] for evidence in case["gold_evidence"]}

    started = perf_counter()
    retrieval = hybrid_search(case["question"], top_k=top_k)
    rows = retrieval["results"]
    latency_ms = (perf_counter() - started) * 1000
    ranked = keys_for_rows(rows)

    metrics: dict[str, float] = {}
    for k in (1, 3, 5):
        parent_set = set(ranked["parents"][:k])
        article_set = set(ranked["articles"][:k])
        document_set = set(ranked["documents"][:k])
        metrics[f"parent_recall_at_{k}"] = len(parent_set & gold_parent_set) / len(gold_parent_set)
        metrics[f"parent_hit_at_{k}"] = float(bool(parent_set & gold_parent_set))
        metrics[f"article_hit_at_{k}"] = float(bool(article_set & gold_articles))
        metrics[f"document_hit_at_{k}"] = float(bool(document_set & gold_documents))

    first_rank = next(
        (
            rank
            for rank, parent_id in enumerate(ranked["parents"][:5], start=1)
            if parent_id in gold_parent_set
        ),
        None,
    )
    metrics["reciprocal_rank_at_5"] = 1.0 / first_rank if first_rank is not None else 0.0

    sample = SingleTurnSample(
        retrieved_context_ids=ranked["parents"],
        reference_context_ids=gold_parent_ids,
    )
    metrics["ragas_id_context_precision_at_5"] = float(
        IDBasedContextPrecision().single_turn_score(sample)
    )
    metrics["ragas_id_context_recall_at_5"] = float(IDBasedContextRecall().single_turn_score(sample))

    retrieved = []
    for row in rows:
        metadata = row["metadata"]
        retrieved.append(
            {
                "chunk_id": row["chunk_id"],
                "parent_article_id": metadata.get("parent_article_id", ""),
                "title": metadata.get("title", ""),
                "article_number": metadata.get("article_number", ""),
                "page_numbers": metadata.get("page_numbers", ""),
                "distance": row["distance"],
                "hybrid_score": row.get("hybrid_score"),
                "reranker_score": row.get("reranker_score"),
                "exact_matches": row.get("exact_matches", []),
                "text": row["text"],
            }
        )
    return {
        **case,
        "gold_parent_article_ids": gold_parent_ids,
        "retrieved": retrieved,
        # 本脚本只评检索，不重复调用回答器。否则每题会执行两次检索，延迟
        # 与 Reranker 开销都会被错误放大。
        "retrieval_trace": retrieval["trace"],
        "evidence_sufficient": bool(retrieval["trace"].get("evidence_sufficient")),
        "generation_mode": "not_run_retrieval_only",
        "latency_ms": latency_ms,
        "metrics": metrics,
    }


def summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics while preserving the number of passed cases."""

    metric_names = list(case_results[0]["metrics"])
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = [float(case["metrics"][name]) for case in case_results]
        summary[name] = mean(values)
        summary[f"{name}_passed_cases"] = sum(value == 1.0 for value in values)
    latencies = [float(case["latency_ms"]) for case in case_results]
    summary.update(
        {
            "case_count": len(case_results),
            "evidence_sufficient_rate": mean(
                [float(case["evidence_sufficient"]) for case in case_results]
            ),
            "false_refusal_count": sum(
                not case["evidence_sufficient"] for case in case_results
            ),
            "multi_recall_retry_rate": mean(
                [
                    float(case["retrieval_trace"].get("retry_triggered", False))
                    for case in case_results
                ]
            ),
            "latency_ms_mean": mean(latencies),
            "latency_ms_p50": percentile(latencies, 0.50),
            "latency_ms_p95": percentile(latencies, 0.95),
        }
    )
    return summary


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    """Write a compact human-readable report next to the complete JSON result."""

    summary = result["summary"]
    lines = [
        "# 法规 RAG 检索评测报告",
        "",
        f"- 运行时间：{result['run']['timestamp']}",
        f"- Git commit：`{result['run']['git_commit']}`",
        f"- Git 工作树：{'有未提交实验改动' if result['run']['git_dirty'] else 'clean'}",
        f"- 数据集：`{result['run']['dataset_version']}`，{summary['case_count']} 条人工 gold",
        f"- 语料：{result['run'].get('indexed_document_count', '未知')} 份唯一文档",
        f"- 检索配置：{result['run'].get('retrieval', 'vector')}，Top-K={result['run']['top_k']}",
        f"- Embedding：`{result['run']['embedding_backend']}` / `{result['run']['embedding_model']}`",
        f"- Reranker：`{result['run']['reranker_backend']}`，候选数={result['run']['rerank_candidates']}",
        f"- HyDE：{'已启用' if result['run'].get('hyde_enabled') else '未启用'}",
        f"- RAGAS：{result['run']['ragas_version']}，仅运行 ID-based 非 LLM 指标",
        "",
        "## 汇总",
        "",
        "| 指标 | 分数 | 满分题数 |",
        "|---|---:|---:|",
    ]
    for metric in (
        "parent_recall_at_1",
        "parent_recall_at_3",
        "parent_recall_at_5",
        "reciprocal_rank_at_5",
        "document_hit_at_5",
        "article_hit_at_5",
        "ragas_id_context_precision_at_5",
        "ragas_id_context_recall_at_5",
    ):
        passed = summary.get(f"{metric}_passed_cases", "-")
        lines.append(f"| {metric} | {summary[metric]:.4f} | {passed}/{summary['case_count']} |")
    lines.extend(
        [
            "",
            f"检索延迟：mean={summary['latency_ms_mean']:.2f} ms，"
            f"P50={summary['latency_ms_p50']:.2f} ms，P95={summary['latency_ms_p95']:.2f} ms。",
            f"冷启动预热：{result['run']['warmup_latency_ms']:.2f} ms（不计入上述分位数）。",
            f"有答案题放行率：{summary['evidence_sufficient_rate']:.4f}，"
            f"错误拒答 {summary['false_refusal_count']}/{summary['case_count']}。",
            f"二次多路召回触发率：{summary['multi_recall_retry_rate']:.4f}。",
            "",
            "## 逐题结果",
            "",
            "| case_id | gold 法条 | Top-1 | Recall@5 | MRR@5 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for case in result["cases"]:
        gold = "、".join(item["article_number"] for item in case["gold_evidence"])
        top1 = ""
        if case["retrieved"]:
            top = case["retrieved"][0]
            top1 = f"《{top['title']}》{top['article_number']}"
        lines.append(
            f"| {case['case_id']} | {gold} | {top1} | "
            f"{case['metrics']['parent_recall_at_5']:.1f} | "
            f"{case['metrics']['reciprocal_rank_at_5']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 评测边界",
            "",
            "- v2 共50题，gold 的法条定位和原文摘录已由 `evaluation.audit_gold_dataset` 逐条核验；它仍是开发集，不等于生产准确率。",
            "- 本报告只评检索，没有运行答案生成，也没有运行 Faithfulness、Factual Correctness 等 LLM 裁判指标。",
            "- RAGAS ID Context Precision 衡量 Top-5 中 gold 父法条占比；它不是语义相关性的 LLM 判断。",
            "- “满分题数”表示该题此项指标等于 1.0；单个 gold 法条配合 5 个不同检索结果时，ID Precision 命中一次通常只有 0.2。",
            "- 低分案例应回查检索结果和源法条，不应修改 gold 迎合模型预测。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI entrypoint: execute cases and persist JSON plus Markdown evidence."""

    parser = argparse.ArgumentParser(description="Evaluate the legal RAG retrieval baseline")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if args.top_k < 5:
        raise ValueError("当前报告固定计算 Recall@1/3/5，因此 top-k 不能小于 5")
    if ragas.__version__ != "0.4.3":
        raise RuntimeError(f"评测固定使用 ragas==0.4.3，当前为 {ragas.__version__}")

    cases = load_cases(args.dataset)
    all_metadata = load_index_metadata()
    # 第一次查询会加载 ONNX 模型并建立内存 BM25 索引。把它单独记为冷启动，
    # 后续逐题延迟更接近常驻 API 服务的真实请求。
    warmup_started = perf_counter()
    hybrid_search(cases[0]["question"], top_k=args.top_k)
    warmup_latency_ms = (perf_counter() - warmup_started) * 1000
    case_results = [run_case(case, all_metadata, args.top_k) for case in cases]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    indexed_document_count = len(
        {metadata.get("document_id") for metadata in all_metadata if metadata.get("document_id")}
    )
    result = {
        "run": {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "dataset_version": args.dataset.stem,
            "corpus_version": (
                f"{manifest['dataset_name']}+verified_collection_{indexed_document_count}_unique_documents"
            ),
            "indexed_document_count": indexed_document_count,
            "ragas_version": ragas.__version__,
            "ragas_install_note": "ID metrics verified; scikit-network omitted on Windows Python 3.14 because no wheel/build tools",
            "top_k": args.top_k,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_backend": EMBEDDING_BACKEND,
            "embedding_model": (
                f"character_sha256_hash_{EMBEDDING_DIMENSION}d"
                if EMBEDDING_BACKEND == "hash"
                else EMBEDDING_MODEL_NAME
            ),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "reranker_backend": RERANKER_BACKEND,
            "rerank_candidates": RERANK_CANDIDATES if RERANKER_BACKEND != "none" else 0,
            "retrieval": "parent_child_bm25_exact_vector_rrf_optional_reranker",
            "hyde_enabled": bool(get_llm_settings()["enabled"] and get_llm_settings()["api_key"]),
            "generation_mode": "not_run_retrieval_only",
            "warmup_latency_ms": warmup_latency_ms,
            "llm_judge_metrics": "not_run_no_dedicated_ragas_evaluator_key",
        },
        "summary": summarize(case_results),
        "cases": case_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.output.with_suffix(".md")
    write_markdown_report(result, report_path)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"完整结果: {args.output}")
    print(f"评测报告: {report_path}")


if __name__ == "__main__":
    main()
