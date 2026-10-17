"""离线评测检索、拒答、时延和模型调用次数；不把检索命中当成回答正确率。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from agent.service import run_agent
from agent.config import get_agent_settings
from evaluation.audit_gold_dataset import _compact_whitespace, load_verified_parents
from shared.config import INDEX_PATH, PROJECT_ROOT, RETRIEVAL_MODE
from shared.llm import build_chat_model


def retrieval_metrics(parent_ids: list[str], gold_ids: list[str]) -> dict:
    keys = ("parent_recall_at_1", "parent_recall_at_3", "parent_recall_at_5", "mrr_at_5", "all_gold_hit_at_5")
    if not gold_ids:
        return dict.fromkeys(keys)
    gold = set(gold_ids)
    return {**{f"parent_recall_at_{k}": len(set(parent_ids[:k]) & gold) / len(gold) for k in (1, 3, 5)},
            "mrr_at_5": next((1 / rank for rank, value in enumerate(parent_ids[:5], 1) if value in gold), 0.0),
            "all_gold_hit_at_5": float(gold.issubset(set(parent_ids[:5])))}


def implementation_hash() -> str:
    digest = hashlib.sha256()
    for folder in ("agent", "shared", "online_retrieval", "offline_ingestion", "evaluation"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
            digest.update(path.read_text(encoding="utf-8").encode("utf-8"))
    return digest.hexdigest()


async def evaluate(dataset: Path, *, top_k: int = 5, with_llm: bool = False) -> dict:
    model = build_chat_model(timeout=get_agent_settings().llm_timeout_seconds) if with_llm else None
    if with_llm and model is None:
        raise ValueError("--with-llm 要求显式启用并完整配置模型；本次未执行模型评测")
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("题集为空或 case_id 重复")
    lookup = load_verified_parents()
    for case in cases:
        for gold in case.get("gold_evidence", []):
            matches = lookup.get((gold["raw_file"], gold["article_number"]), [])
            if len(matches) != 1 or _compact_whitespace(gold["evidence_quote"]) not in _compact_whitespace(matches[0]["text"]):
                raise ValueError(f"gold 来源审计失败：{case['case_id']}")
    records = []
    for case in cases:
        gold_ids = [lookup[(gold["raw_file"], gold["article_number"])][0]["parent_id"] for gold in case.get("gold_evidence", [])]
        for mode in ("rag", "agent"):
            started = perf_counter()
            result = await run_agent(case["question"], top_k, mode=mode, model=model)
            ids = [row["parent_id"] for row in result["citations"]]
            records.append({
                "case_id": case["case_id"], "mode": mode, "question": case["question"], "should_answer": case.get("should_answer", True),
                "status": result["status"], "refused": result["insufficient_evidence"], "stop_reason": result["stop_reason"],
                "category": case.get("category", "retrieval"), "gold_parent_ids": gold_ids, "returned_parent_ids": ids,
                "metrics": retrieval_metrics(ids, gold_ids), "latency_ms": round((perf_counter() - started) * 1000, 2),
                "llm_call_count": result["agent_trace"]["llm_call_count"], "iterations": result["agent_trace"]["iterations"],
                "llm_usage": result["agent_trace"]["llm_usage"], "generation_mode": result["generation_mode"], "warning": result["warning"],
                "controller": result["agent_trace"]["controller"], "decision_steps": result["agent_trace"]["decision_steps"],
                "actions": result["agent_trace"]["actions"],
                "answer": result["answer"], "citations": result["citations"],
            })
    summary = {}
    for mode in ("rag", "agent"):
        rows = [row for row in records if row["mode"] == mode]
        answerable = [row for row in rows if row["should_answer"]]
        boundary = [row for row in rows if not row["should_answer"]]
        fusion = [row for row in answerable if len(set(row["gold_parent_ids"])) > 1]
        mean = lambda values: statistics.fmean(values) if values else None
        summary[mode] = {
            "case_count": len(rows), "answerable_count": len(answerable), "boundary_count": len(boundary),
            "false_refusal_count": sum(row["refused"] for row in answerable),
            "correct_boundary_refusal_count": sum(row["refused"] and not row["returned_parent_ids"] for row in boundary),
            "retrieval": {key: mean([row["metrics"][key] for row in answerable if row["metrics"][key] is not None])
                          for key in ("parent_recall_at_1", "parent_recall_at_3", "parent_recall_at_5", "mrr_at_5")},
            "multi_file_all_hit_at_5": mean([row["metrics"]["all_gold_hit_at_5"] for row in fusion]),
            "mean_latency_ms": mean([row["latency_ms"] for row in rows]),
            "max_latency_ms": max(row["latency_ms"] for row in rows),
            "model_calls": sum(row["llm_call_count"] for row in rows),
            "controller_counts": {controller: sum(row["controller"] == controller for row in rows)
                                  for controller in sorted({row["controller"] for row in rows})},
        }
    model_events = [event for record in records for event in record["llm_usage"]["events"]]
    model_degraded = any(not event["success"] for event in model_events) or any(record["warning"] for record in records)
    status = "offline_complete" if not with_llm else "model_not_exercised" if not model_events else "model_run_degraded" if model_degraded else "model_run_complete"
    return {"run": {"timestamp": datetime.now(timezone.utc).isoformat(), "status": status,
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, encoding="utf-8").strip()),
                    "dataset": dataset.name, "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    "implementation_sha256": implementation_hash(), "index_sha256": hashlib.sha256(INDEX_PATH.read_bytes()).hexdigest(),
                    "top_k": top_k, "retrieval_mode": RETRIEVAL_MODE, "llm_requested": with_llm,
                    "agent_settings": asdict(get_agent_settings()),
                    "answer_quality_judged": False, "dataset_role": "development_regression"},
            "summary": summary, "cases": records}


def write_markdown(result: dict, path: Path) -> None:
    run = result["run"]
    rows = [
        "# 轻量法律检索 Agent 开发集回归", "",
        f"- 状态：{run['status']}；题集：{run['dataset']}；Top-K={run['top_k']}；检索模式：{run['retrieval_mode']}",
        f"- Git：{run['git_commit']}；存在未提交文件：{run['git_dirty']}",
        f"- 数据集 SHA-256：{run['dataset_sha256']}",
        f"- 实现 SHA-256：{run['implementation_sha256']}",
        f"- 索引 SHA-256：{run['index_sha256']}；时间 UTC：{run['timestamp']}",
        f"- 显式请求外部模型：{run['llm_requested']}；没有评判模型回答的法律正确性。耗时是本机顺序运行测量，不代表并发能力。",
        "", "| 指标 | 基础 RAG | 法律检索 Agent |", "| --- | ---: | ---: |",
    ]
    for key, label in (("parent_recall_at_1", "Parent Recall@1"), ("parent_recall_at_3", "Parent Recall@3"),
                       ("parent_recall_at_5", "Parent Recall@5"), ("mrr_at_5", "MRR@5")):
        values = [result["summary"][mode]["retrieval"][key] for mode in ("rag", "agent")]
        rows.append(f"| {label} | " + " | ".join("-" if value is None else f"{value:.4f}" for value in values) + " |")
    for key, label in (("false_refusal_count", "有答案题误拒答数"), ("correct_boundary_refusal_count", "边界题正确拒答数"),
                       ("multi_file_all_hit_at_5", "多法源全部命中@5"), ("mean_latency_ms", "平均耗时 ms"), ("model_calls", "外部模型调用数")):
        values = [result["summary"][mode][key] for mode in ("rag", "agent")]
        rows.append(f"| {label} | " + " | ".join("-" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value) for value in values) + " |")
    rows.extend(["", "指标基于最终正式引用的父法条ID。证据不足时清空引用，因此召回结果也体现保守停止的代价。", "",
                 "| case_id | 模式 | 状态 | Recall@5 | 停止原因 |", "| --- | --- | --- | ---: | --- |"])
    for row in result["cases"]:
        score = row["metrics"]["parent_recall_at_5"]
        rows.append(f"| {row['case_id']} | {row['mode']} | {row['status']} | {'-' if score is None else f'{score:.4f}'} | {row['stop_reason']} |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation/datasets/legal_agent_rag_comparison_v1.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation/results/lightweight_agent_v2.json")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--with-llm", action="store_true", help="显式调用已配置的模型；默认不会调用")
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.dataset, top_k=args.top_k, with_llm=args.with_llm))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, args.output.with_suffix(".md"))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
