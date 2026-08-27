"""Run RAGAS LLM-judged Context Precision/Recall on a saved retrieval run.

The current system uses ``retrieval_fallback`` answers, so generation metrics
such as Faithfulness and Factual Correctness would mostly reward copying the
retrieved passages.  This script intentionally evaluates context quality only.

No key is printed or written to the result.  The evaluator uses the dedicated
RAGAS variables when present, otherwise the already configured DeepSeek key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "results" / "baseline_hash_top5.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "baseline_hash_top5_ragas_llm.json"


def evaluator_settings() -> tuple[str, str, str]:
    """Resolve evaluator settings without logging the secret key."""

    key = os.getenv("RAGAS_EVALUATOR_API_KEY", "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = os.getenv("RAGAS_EVALUATOR_MODEL", "").strip() or os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-chat"
    base_url = (
        os.getenv("RAGAS_EVALUATOR_BASE_URL", "").strip()
        or os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or "https://api.deepseek.com"
    )
    if not key:
        raise RuntimeError("缺少 RAGAS_EVALUATOR_API_KEY 或 DEEPSEEK_API_KEY，无法运行 LLM 裁判")
    return key, model, base_url


async def score_case(case: dict[str, Any], llm, semaphore: asyncio.Semaphore) -> dict[str, Any]:
    """Score one case; preserve failures instead of aborting the whole run."""

    contexts = [item["text"] for item in case["retrieved"]]

    async def score_metric(metric, **kwargs):
        async with semaphore:
            return await metric.ascore(**kwargs)

    try:
        precision_result, recall_result = await asyncio.gather(
            score_metric(
                ContextPrecision(llm=llm),
                user_input=case["question"],
                reference=case["reference_answer"],
                retrieved_contexts=contexts,
            ),
            score_metric(
                ContextRecall(llm=llm),
                user_input=case["question"],
                reference=case["reference_answer"],
                retrieved_contexts=contexts,
            ),
        )
        return {
            "case_id": case["case_id"],
            "context_precision": precision_result.value,
            "context_recall": recall_result.value,
            "precision_reason": precision_result.reason,
            "recall_reason": recall_result.reason,
            "error": None,
        }
    except Exception as exc:
        # Exception text is kept for diagnosis, but settings and request headers
        # are never included because they may contain credentials.
        return {
            "case_id": case["case_id"],
            "context_precision": None,
            "context_recall": None,
            "precision_reason": None,
            "recall_reason": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_report(result: dict[str, Any], path: Path) -> None:
    """Write a concise report and keep the JSON as the detailed artifact."""

    summary = result["summary"]
    lines = [
        "# RAGAS LLM 上下文评测",
        "",
        f"- 评测模型：`{result['run']['evaluator_model']}`",
        f"- 成功案例：{summary['successful_cases']}/{summary['case_count']}",
        "- 仅评测 Context Precision 和 Context Recall。",
        "- 业务回答仍为 retrieval_fallback，未运行生成质量指标。",
        "",
        "| 指标 | 平均分 |",
        "|---|---:|",
        f"| RAGAS Context Precision | {summary['context_precision_mean']:.4f} |",
        f"| RAGAS Context Recall | {summary['context_recall_mean']:.4f} |",
        "",
        "## 逐题结果",
        "",
        "| case_id | Context Precision | Context Recall | 状态 |",
        "|---|---:|---:|---|",
    ]
    for case in result["cases"]:
        cp = "-" if case["context_precision"] is None else f"{case['context_precision']:.4f}"
        cr = "-" if case["context_recall"] is None else f"{case['context_recall']:.4f}"
        status = "成功" if case["error"] is None else case["error"].split(":", 1)[0]
        lines.append(f"| {case['case_id']} | {cp} | {cr} | {status} |")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这些分数来自 LLM 裁判，必须结合人工 gold Recall 和具体失败案例阅读。",
            "- Context Precision 可把语义相关但非 gold 法条的上下文判为有用，因此可能高于 ID Precision。",
            "- Context Recall 依赖人工参考答案的完整性，不等同于确定性的法条 Recall@K。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(input_path: Path, output_path: Path, concurrency: int) -> None:
    """Create the evaluator, score all cases and persist aggregate evidence."""

    baseline = json.loads(input_path.read_text(encoding="utf-8"))
    key, model, base_url = evaluator_settings()
    # RAGAS 0.4.3's current wheel requires an explicit async client.  Passing
    # api_key directly to llm_factory raises a migration error.
    client = AsyncOpenAI(api_key=key, base_url=base_url)
    evaluator_llm = llm_factory(model, client=client)
    semaphore = asyncio.Semaphore(concurrency)
    case_results = await asyncio.gather(
        *(score_case(case, evaluator_llm, semaphore) for case in baseline["cases"])
    )
    successful = [case for case in case_results if case["error"] is None]
    if not successful:
        raise RuntimeError("所有 RAGAS LLM 评测均失败，请检查逐题错误")
    result = {
        "run": {
            "baseline_result": str(input_path.relative_to(PROJECT_ROOT)),
            "ragas_version": baseline["run"]["ragas_version"],
            "evaluator_model": model,
            "concurrency": concurrency,
            "generation_metrics": "not_run_retrieval_fallback_mode",
        },
        "summary": {
            "case_count": len(case_results),
            "successful_cases": len(successful),
            "context_precision_mean": fmean(float(case["context_precision"]) for case in successful),
            "context_recall_mean": fmean(float(case["context_recall"]) for case in successful),
        },
        "cases": case_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output_path.with_suffix(".md")
    write_report(result, report_path)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"完整结果: {output_path}")
    print(f"评测报告: {report_path}")


def main() -> None:
    """Parse CLI options and run the asynchronous evaluator."""

    parser = argparse.ArgumentParser(description="Run RAGAS LLM context metrics")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency 必须大于 0")
    asyncio.run(run(args.input, args.output, args.concurrency))


if __name__ == "__main__":
    main()
