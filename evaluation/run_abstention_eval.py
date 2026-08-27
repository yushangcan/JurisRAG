"""评测证据不足问题是否被系统拒答，不调用 LLM 裁判。

这里的10题专门覆盖不存在条号、境外法规、缺失地方文件、错误数值前提和
个案预测。指标是确定性的拒答分类结果，不把模型主观评分伪装成准确率。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from online_retrieval.answerer import answer_question
from shared.config import PROJECT_ROOT


DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "legal_rag_unanswerable_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "abstention_unanswerable_v1.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(row.get("should_answer") is not False for row in rows):
        raise ValueError("拒答评测集必须包含 should_answer=false 的案例")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("拒答评测集存在重复 case_id")
    return rows


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    """写出小白也能直接阅读的逐题拒答报告。"""

    summary = result["summary"]
    lines = [
        "# 法规 RAG 无答案拒答评测",
        "",
        f"- 运行时间：{result['run']['timestamp']}",
        f"- 案例数：{summary['case_count']}",
        f"- 正确拒答：{summary['refused_count']}/{summary['case_count']}",
        f"- 不安全回答：{summary['unsafe_answer_count']}",
        "- LLM 裁判：未使用",
        "",
        "| case_id | 类型 | 是否拒答 | 最终原因 |",
        "|---|---|---:|---|",
    ]
    for case in result["cases"]:
        reason = str(case.get("refusal_reason") or "").replace("|", "\\|")
        lines.append(
            f"| {case['case_id']} | {case['category']} | "
            f"{'是' if case['refused'] else '否'} | {reason} |"
        )
    lines.extend(
        [
            "",
            "该结果只适用于当前10题开发集，不能外推为生产环境100%拒答准确率。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate legal RAG abstention behavior")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    results: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        response = answer_question(case["question"], top_k=args.top_k)
        results.append(
            {
                **case,
                "refused": bool(response["insufficient_evidence"]),
                "generation_mode": response["generation_mode"],
                "refusal_reason": response.get("refusal_reason"),
                "latency_ms": (perf_counter() - started) * 1000,
                "retrieval_trace": response["retrieval_trace"],
            }
        )

    refused_count = sum(result["refused"] for result in results)
    output = {
        "run": {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": args.dataset.name,
            "top_k": args.top_k,
            "llm_judge": "not_used",
        },
        "summary": {
            "case_count": len(results),
            "refused_count": refused_count,
            "refusal_rate": refused_count / len(results),
            "unsafe_answer_count": len(results) - refused_count,
            "category_counts": dict(Counter(result["category"] for result in results)),
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.output.with_suffix(".md")
    write_markdown_report(output, report_path)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"完整结果: {args.output}")
    print(f"评测报告: {report_path}")


if __name__ == "__main__":
    main()
