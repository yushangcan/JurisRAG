"""运行受约束 Agent 的工程行为评测。

本脚本检查领域路由、可回答性、首个工具、停止预算、重复调用和引用完整性。
它不把这些工程指标冒充法律结论正确率，也不调用 RAGAS 语义评分。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from agent.service import run_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "legal_agent_smoke_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "legal_agent_smoke_v1.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case["case_id"] for case in cases]
    if not cases:
        raise ValueError("Agent 评测集不能为空")
    if len(ids) != len(set(ids)):
        raise ValueError("Agent 评测集存在重复 case_id")
    required = {
        "case_id",
        "question",
        "expected_domain",
        "acceptable_answerability",
        "acceptable_first_tools",
        "expect_tool_call",
        "must_not_claim",
    }
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"{case.get('case_id', 'unknown')} 缺少字段: {sorted(missing)}")
    return cases


def _duplicate_action_count(actions: list[dict[str, Any]]) -> int:
    signatures = [
        json.dumps(
            {"tool": row.get("tool_name"), "arguments": row.get("arguments", {})},
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in actions
        if row.get("status") == "success"
    ]
    return len(signatures) - len(set(signatures))


def score_case(case: dict[str, Any], result: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    trace = result.get("agent_trace", {})
    actions = trace.get("actions", [])
    first_tool = actions[0].get("tool_name") if actions else None
    citations = result.get("citations", [])
    forbidden = [text for text in case["must_not_claim"] if text in result.get("answer", "")]
    citation_ids_valid = all(
        citation.get("evidence_id") or citation.get("parent_id")
        for citation in citations
    )
    citation_sources_valid = all(citation.get("source_url") for citation in citations)
    metrics = {
        "domain_correct": result.get("domain") == case["expected_domain"],
        "answerability_acceptable": result.get("answerability") in case["acceptable_answerability"],
        "first_tool_acceptable": (
            first_tool in case["acceptable_first_tools"]
            if case["expect_tool_call"]
            else first_tool is None
        ),
        "tool_call_expectation_met": bool(actions) == bool(case["expect_tool_call"]),
        "bounded_iterations": int(trace.get("iterations", 0)) <= int(
            trace.get("budgets", {}).get("max_iterations", 3)
        ),
        "bounded_tool_calls": int(trace.get("tool_call_count", 0)) <= int(
            trace.get("budgets", {}).get("max_tool_calls", 8)
        ),
        "no_duplicate_actions": _duplicate_action_count(actions) == 0,
        "citation_ids_valid": citation_ids_valid,
        "citation_sources_valid": citation_sources_valid,
        "forbidden_claims_absent": not forbidden,
    }
    return {
        **case,
        "result_summary": {
            "domain": result.get("domain"),
            "answerability": result.get("answerability"),
            "stop_reason": result.get("stop_reason"),
            "generation_mode": result.get("generation_mode"),
            "first_tool": first_tool,
            "iterations": trace.get("iterations", 0),
            "tool_call_count": trace.get("tool_call_count", 0),
            "llm_call_count": trace.get("llm_call_count", 0),
            "citation_count": len(citations),
            "latency_ms": latency_ms,
            "forbidden_claims_found": forbidden,
        },
        "metrics": metrics,
        "passed": all(metrics.values()),
        "answer": result.get("answer", ""),
        "citations": citations,
        "agent_trace": trace,
    }


async def run_case(case: dict[str, Any], top_k: int) -> dict[str, Any]:
    started = perf_counter()
    result = await run_agent(case["question"], top_k=top_k)
    return score_case(case, result, (perf_counter() - started) * 1000)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = list(results[0]["metrics"])
    latencies = [row["result_summary"]["latency_ms"] for row in results]
    return {
        "case_count": len(results),
        "passed_cases": sum(row["passed"] for row in results),
        "pass_rate": statistics.fmean(float(row["passed"]) for row in results),
        **{
            name: statistics.fmean(float(row["metrics"][name]) for row in results)
            for name in metric_names
        },
        "latency_ms_mean": statistics.fmean(latencies),
        "average_iterations": statistics.fmean(
            row["result_summary"]["iterations"] for row in results
        ),
        "average_tool_calls": statistics.fmean(
            row["result_summary"]["tool_call_count"] for row in results
        ),
        "average_llm_calls": statistics.fmean(
            row["result_summary"]["llm_call_count"] for row in results
        ),
    }


async def main_async(args: argparse.Namespace) -> None:
    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    results = []
    # 法律 Agent 单题会多次调用模型。第一版顺序执行，避免并发掩盖限流和延迟。
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['case_id']}")
        try:
            results.append(await run_case(case, args.top_k))
        except Exception as exc:
            results.append(
                {
                    **case,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                    "result_summary": {"latency_ms": 0, "iterations": 0, "tool_call_count": 0, "llm_call_count": 0},
                }
            )
    successful = [row for row in results if row.get("metrics")]
    payload = {
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(args.dataset),
        "top_k": args.top_k,
        "summary": summarize(successful) if successful else {"case_count": 0},
        "failed_to_run": sum(not row.get("metrics") for row in results),
        "cases": results,
        "metric_boundary": "工程行为评测，不代表法律权威正确率或完整 RAGAS 评测",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"结果已写入 {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JurisRAG Agent 工程行为评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题；0 表示全部")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))

