"""在同一冻结题集上对比固定流程 RAG 与受约束 LangGraph Agent。

评测刻意分成三层，避免把不同含义的数字混在一起：

1. 确定性检索指标：根据人工 gold 父法条计算 Recall@K、MRR 和多文件全命中率；
2. RAGAS 回答指标：只对有答案题计算上下文精确率/召回率、忠实度、回答
   相关性和事实正确性；
3. 边界与 Agent 工程指标：正确拒答、错误回答、伪引用、工具次数、循环次数、
   重复动作、纠正次数和降级率。

脚本按系统和题目即时保存缓存。缓存签名包含数据集、模型、Top-K、提示版本、
代码状态、语料清单与索引摘要；签名变化后不会误用旧分数。API Key 只从环境变量
读取，永远不会写入结果、日志或 Git。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from agent.mcp_client import InProcessMCPClient
from agent.service import run_agent
from evaluation.audit_agent_rag_comparison import (
    DEFAULT_DATASET,
    audit_dataset,
    load_comparison_dataset,
)
from evaluation.audit_gold_dataset import load_verified_parents
from legal_mcp.server import create_server
from online_retrieval.answerer import answer_question
from shared.config import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    MANIFEST_PATH,
    PARENT_CHROMA_COLLECTION_NAME,
    PROJECT_ROOT,
    RERANKER_BACKEND,
    RERANK_CANDIDATES,
    get_llm_settings,
)
from shared.llm import build_chat_model


SYSTEMS = ("fixed_rag", "agent")
SYSTEM_LABELS = {
    "fixed_rag": "固定流程 RAG",
    "agent": "LangGraph Agent",
}
RAGAS_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
)
ANSWER_PROMPT_VERSIONS = {
    "fixed_rag": "current_answerer_evidence_gate_v1",
    "agent": "current_langgraph_policy_validation_v1",
}
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "agent_rag_comparison_v1.json"
SAFE_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


def sha256_file(path: Path) -> str | None:
    """流式计算文件摘要；文件不存在时显式返回空，而不是伪造摘要。"""

    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_paths(paths: Iterable[Path]) -> str:
    """为影响两套系统行为的 Python 文件生成稳定组合摘要。"""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value).lower()):
        digest.update(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def implementation_sha256() -> str:
    roots = ("agent", "legal_mcp", "online_retrieval", "shared")
    paths = [
        path
        for root in roots
        for path in (PROJECT_ROOT / root).rglob("*.py")
        if path.is_file()
    ]
    return sha256_paths(paths)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def git_is_dirty() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    )
    # 用户本地工具目录不属于本次实验，不能让它们污染可复现实验状态。
    relevant = [
        line
        for line in output.splitlines()
        if not any(name in line for name in (".workbuddy/", ".resume_task/"))
    ]
    return bool(relevant)


def safe_error(exc: Exception) -> str:
    """保留可诊断的异常类型，但遮蔽意外出现在异常文本中的密钥。"""

    message = SAFE_KEY_PATTERN.sub("[REDACTED_API_KEY]", str(exc))
    return f"{type(exc).__name__}: {message}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.fmean(rows) if rows else None


def duplicate_action_count(actions: list[dict[str, Any]]) -> int:
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


def gold_parent_lookup() -> dict[tuple[str, str], str]:
    """建立人工标签到稳定 parent_id 的唯一映射。"""

    parents = load_verified_parents()
    result: dict[tuple[str, str], str] = {}
    for key, values in parents.items():
        if len(values) == 1:
            result[key] = values[0]["parent_id"]
    return result


def resolve_gold_parent_ids(
    case: dict[str, Any], lookup: dict[tuple[str, str], str]
) -> list[str]:
    result: list[str] = []
    for row in case["gold_evidence"]:
        key = (str(row["raw_file"]), str(row["article_number"]))
        if key not in lookup:
            raise ValueError(f"{case['case_id']} 的 gold 无法解析为 parent_id: {key}")
        if lookup[key] not in result:
            result.append(lookup[key])
    return result


def citation_parent_ids(citations: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in citations:
        value = str(row.get("parent_id") or row.get("evidence_id") or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def retrieval_metrics(parent_ids: list[str], gold_ids: list[str]) -> dict[str, Any]:
    """只基于稳定 ID 评分，不让 LLM 主观判断代替真实命中。"""

    if not gold_ids:
        return {
            "parent_recall_at_1": None,
            "parent_recall_at_3": None,
            "parent_recall_at_5": None,
            "mrr_at_5": None,
            "all_gold_hit_at_5": None,
            "any_gold_hit_at_5": None,
        }
    gold = set(gold_ids)
    metrics: dict[str, Any] = {}
    for k in (1, 3, 5):
        metrics[f"parent_recall_at_{k}"] = len(set(parent_ids[:k]) & gold) / len(gold)
    first_rank = next(
        (rank for rank, value in enumerate(parent_ids[:5], 1) if value in gold),
        None,
    )
    metrics["mrr_at_5"] = 1 / first_rank if first_rank else 0.0
    metrics["all_gold_hit_at_5"] = float(gold.issubset(set(parent_ids[:5])))
    metrics["any_gold_hit_at_5"] = float(bool(gold & set(parent_ids[:5])))
    return metrics


def normalize_result(
    system: str,
    case: dict[str, Any],
    result: dict[str, Any],
    latency_ms: float,
    lookup: dict[tuple[str, str], str],
) -> dict[str, Any]:
    citations = list(result.get("citations") or [])
    parents = citation_parent_ids(citations)
    refused = bool(result.get("insufficient_evidence")) or (
        system == "agent" and result.get("answerability") == "insufficient"
    )
    gold_ids = resolve_gold_parent_ids(case, lookup) if case["should_answer"] else []
    trace = dict(result.get("agent_trace") or {})
    actions = list(trace.get("actions") or [])
    first_tool = actions[0].get("tool_name") if actions else None
    llm_usage = dict(trace.get("llm_usage") or {})
    return {
        "case_id": case["case_id"],
        "system": system,
        "question": case["question"],
        "answer": str(result.get("answer") or ""),
        "citations": citations,
        "contexts": [str(row.get("quote") or "") for row in citations if row.get("quote")],
        "citation_parent_ids": parents,
        "gold_parent_ids": gold_ids,
        "refused": refused,
        "correct_boundary_refusal": bool(not case["should_answer"] and refused),
        "false_refusal": bool(case["should_answer"] and refused),
        "unsafe_boundary_answer": bool(not case["should_answer"] and not refused),
        "pseudo_citation_on_boundary": bool(not case["should_answer"] and citations),
        "latency_ms": latency_ms,
        "generation_mode": result.get("generation_mode"),
        "refusal_reason": result.get("refusal_reason") or result.get("stop_reason"),
        "answerability": result.get("answerability"),
        "domain": result.get("domain"),
        "retrieval_metrics": retrieval_metrics(parents, gold_ids),
        "agent_behavior": (
            {
                "first_tool": first_tool,
                "iterations": int(trace.get("iterations", 0)),
                "tool_call_count": int(trace.get("tool_call_count", 0)),
                "llm_call_count": int(trace.get("llm_call_count", 0)),
                "duplicate_action_count": duplicate_action_count(actions),
                "correction_count": int(trace.get("correction_count", 0)),
                "stop_reason": result.get("stop_reason"),
                "baseline_fallback": result.get("generation_mode") == "agent_baseline_fallback",
                "token_reporting_complete": llm_usage.get("token_reporting_complete"),
                "input_tokens": llm_usage.get("input_tokens"),
                "output_tokens": llm_usage.get("output_tokens"),
                "total_tokens": llm_usage.get("total_tokens"),
                "actions": actions,
            }
            if system == "agent"
            else None
        ),
        "raw_trace": (
            result.get("agent_trace")
            if system == "agent"
            else result.get("retrieval_trace")
        ),
        "execution_error": None,
    }


def failure_result(system: str, case: dict[str, Any], exc: Exception, latency_ms: float) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "system": system,
        "question": case["question"],
        "answer": "",
        "citations": [],
        "contexts": [],
        "citation_parent_ids": [],
        "gold_parent_ids": [],
        "refused": False,
        "correct_boundary_refusal": False,
        "false_refusal": False,
        "unsafe_boundary_answer": False,
        "pseudo_citation_on_boundary": False,
        "latency_ms": latency_ms,
        "generation_mode": "execution_failed",
        "refusal_reason": None,
        "answerability": None,
        "domain": None,
        "retrieval_metrics": {},
        "agent_behavior": None,
        "raw_trace": None,
        "execution_error": safe_error(exc),
    }


def load_cache(path: Path, signature: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = read_json(path)
    if value.get("signature") != signature:
        print(f"缓存签名已变化，将重新生成：{path.name}", flush=True)
        return {}
    return {str(row["case_id"]): row for row in value.get("cases", [])}


def save_cache(
    path: Path,
    signature: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    case_order: list[str],
) -> None:
    write_json(
        path,
        {
            "signature": signature,
            "cases": [rows[key] for key in case_order if key in rows],
        },
    )


async def run_system_cases(
    system: str,
    cases: list[dict[str, Any]],
    lookup: dict[tuple[str, str], str],
    top_k: int,
    cache_path: Path,
    signature: dict[str, Any],
    retry_failures: bool,
) -> list[dict[str, Any]]:
    """顺序执行，避免两套系统争用本地 ONNX 模型和外部限流额度。"""

    case_order = [str(case["case_id"]) for case in cases]
    cached = load_cache(cache_path, signature)
    client = InProcessMCPClient(create_server()) if system == "agent" else None
    for index, case in enumerate(cases, 1):
        previous = cached.get(case["case_id"])
        if previous and not (retry_failures and previous.get("execution_error")):
            continue
        started = perf_counter()
        try:
            if system == "fixed_rag":
                raw = await asyncio.to_thread(answer_question, case["question"], top_k)
            else:
                raw = await run_agent(case["question"], top_k=top_k, client=client)
            row = normalize_result(
                system,
                case,
                raw,
                (perf_counter() - started) * 1000,
                lookup,
            )
        except Exception as exc:
            row = failure_result(
                system, case, exc, (perf_counter() - started) * 1000
            )
        cached[case["case_id"]] = row
        save_cache(cache_path, signature, cached, case_order)
        status = "失败" if row["execution_error"] else "完成"
        print(f"{SYSTEM_LABELS[system]} {index}/{len(cases)} {case['case_id']}：{status}", flush=True)
    return [cached[case["case_id"]] for case in cases if case["case_id"] in cached]


def evaluator_settings() -> tuple[str, str, str]:
    key = os.getenv("RAGAS_EVALUATOR_API_KEY", "").strip()
    model = os.getenv("RAGAS_EVALUATOR_MODEL", "").strip()
    base_url = os.getenv("RAGAS_EVALUATOR_BASE_URL", "").strip()
    if not key or not model or not base_url:
        raise RuntimeError(
            "RAGAS 需要 RAGAS_EVALUATOR_API_KEY、RAGAS_EVALUATOR_MODEL 和 "
            "RAGAS_EVALUATOR_BASE_URL；密钥只能通过环境变量传入"
        )
    return key, model, base_url


async def preflight_answer_model() -> None:
    """在昂贵评测前做一次最小真实调用，防止401后悄悄评到降级答案。"""

    model = build_chat_model()
    if model is None:
        raise RuntimeError("回答模型工厂没有返回可用模型")
    try:
        response = await model.ainvoke(
            "这是评测连通性检查。请只回复 OK。",
            config={"run_name": "jurisrag_evaluation_preflight"},
        )
    except Exception as exc:
        raise RuntimeError(f"回答模型预检失败：{safe_error(exc)}") from exc
    content = response.content if isinstance(response.content, str) else str(response.content)
    if not content.strip():
        raise RuntimeError("回答模型预检返回空内容")


async def run_ragas_scores(
    cases: list[dict[str, Any]],
    system_rows: dict[str, list[dict[str, Any]]],
    cache_path: Path,
    signature: dict[str, Any],
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """对两套系统的有答案输出运行相同 RAGAS 裁判并逐题落盘。"""

    import ragas
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    if ragas.__version__ != "0.4.3":
        raise RuntimeError(f"评测固定使用 ragas==0.4.3，当前为 {ragas.__version__}")
    from evaluation.run_ragas_full_eval import ProjectBgeEmbedding, score_one

    key, model, base_url = evaluator_settings()
    client = AsyncOpenAI(
        api_key=key,
        base_url=base_url,
        default_headers={
            "User-Agent": os.getenv("LLM_HTTP_USER_AGENT", "Mozilla/5.0")
        },
    )
    evaluator_llm = llm_factory(model, client=client, temperature=0, max_tokens=4096)
    embeddings = ProjectBgeEmbedding()
    semaphore = asyncio.Semaphore(concurrency)
    cached = load_cache(cache_path, signature)
    case_map = {str(case["case_id"]): case for case in cases}
    ordered_keys = [
        f"{system}::{case['case_id']}"
        for system in SYSTEMS
        if system in system_rows
        for case in cases
        if case["should_answer"]
    ]

    for system in SYSTEMS:
        if system not in system_rows:
            continue
        for row in system_rows[system]:
            case = case_map[row["case_id"]]
            if not case["should_answer"]:
                continue
            key_name = f"{system}::{case['case_id']}"
            previous = cached.get(key_name)
            if previous and all(
                previous.get("metrics", {}).get(name, {}).get("value") is not None
                for name in RAGAS_METRICS
            ):
                continue
            if row.get("execution_error") or not row.get("answer"):
                metrics = {
                    name: {
                        "value": None,
                        "reason": None,
                        "error": "系统执行失败或回答为空，未调用 RAGAS",
                    }
                    for name in RAGAS_METRICS
                }
                scored = {"metrics": metrics}
            else:
                ragas_case = {
                    "case_id": key_name,
                    "question": case["question"],
                    "reference_answer": case["reference_answer"],
                    "retrieved": [{"text": text} for text in row["contexts"]],
                }
                generation = {
                    "response": row["answer"],
                    "error": None,
                }
                scored = await score_one(
                    ragas_case,
                    generation,
                    evaluator_llm,
                    embeddings,
                    semaphore,
                )
            cached[key_name] = {
                "case_id": key_name,
                "system": system,
                "source_case_id": case["case_id"],
                "metrics": scored["metrics"],
            }
            # save_cache 期望 case_id 唯一，因此这里的复合 ID 就是缓存主键。
            save_cache(cache_path, signature, cached, ordered_keys)
            print(f"RAGAS {SYSTEM_LABELS[system]} {case['case_id']}：完成", flush=True)
    return cached


def aggregate_rows(
    rows: list[dict[str, Any]],
    case_map: dict[str, dict[str, Any]],
    ragas_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("execution_error")]
    answerable = [row for row in successful if case_map[row["case_id"]]["should_answer"]]
    boundary = [row for row in successful if not case_map[row["case_id"]]["should_answer"]]
    multi = [
        row
        for row in answerable
        if case_map[row["case_id"]]["category"] == "multi_file_fusion"
    ]
    latencies = [float(row["latency_ms"]) for row in successful]
    retrieval = {
        name: mean(
            float(row["retrieval_metrics"][name])
            for row in answerable
            if row.get("retrieval_metrics", {}).get(name) is not None
        )
        for name in (
            "parent_recall_at_1",
            "parent_recall_at_3",
            "parent_recall_at_5",
            "mrr_at_5",
        )
    }
    ragas_summary: dict[str, Any] = {}
    system = rows[0]["system"] if rows else "unknown"
    for metric in RAGAS_METRICS:
        values = [
            float(score["metrics"][metric]["value"])
            for key, score in ragas_scores.items()
            if key.startswith(f"{system}::")
            and score.get("metrics", {}).get(metric, {}).get("value") is not None
        ]
        ragas_summary[metric] = {
            "mean": mean(values),
            "successful_cases": len(values),
            "expected_cases": len(answerable),
        }
    summary: dict[str, Any] = {
        "case_count": len(rows),
        "execution_success_count": len(successful),
        "execution_failure_count": len(rows) - len(successful),
        "answerable_count": len(answerable),
        "false_refusal_count": sum(row["false_refusal"] for row in answerable),
        "false_refusal_rate": mean(float(row["false_refusal"]) for row in answerable),
        "boundary_count": len(boundary),
        "correct_boundary_refusal_count": sum(
            row["correct_boundary_refusal"] for row in boundary
        ),
        "correct_boundary_refusal_rate": mean(
            float(row["correct_boundary_refusal"]) for row in boundary
        ),
        "unsafe_boundary_answer_count": sum(
            row["unsafe_boundary_answer"] for row in boundary
        ),
        "pseudo_citation_on_boundary_count": sum(
            row["pseudo_citation_on_boundary"] for row in boundary
        ),
        "retrieval": retrieval,
        "multi_file": {
            "case_count": len(multi),
            "all_gold_hit_at_5": mean(
                float(row["retrieval_metrics"]["all_gold_hit_at_5"]) for row in multi
            ),
            "any_gold_hit_at_5": mean(
                float(row["retrieval_metrics"]["any_gold_hit_at_5"]) for row in multi
            ),
        },
        "latency_ms": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "ragas": ragas_summary,
    }
    if system == "agent":
        behavior = [row["agent_behavior"] for row in successful if row.get("agent_behavior")]
        summary["agent_behavior"] = {
            "average_iterations": mean(float(row["iterations"]) for row in behavior),
            "average_tool_calls": mean(float(row["tool_call_count"]) for row in behavior),
            "average_llm_calls": mean(float(row["llm_call_count"]) for row in behavior),
            "duplicate_action_count": sum(row["duplicate_action_count"] for row in behavior),
            "correction_count": sum(row["correction_count"] for row in behavior),
            "baseline_fallback_count": sum(row["baseline_fallback"] for row in behavior),
            "first_tool_counts": dict(Counter(str(row["first_tool"]) for row in behavior)),
            "stop_reason_counts": dict(Counter(str(row["stop_reason"]) for row in behavior)),
            "token_complete_cases": sum(
                row["token_reporting_complete"] is True for row in behavior
            ),
            "total_tokens": (
                sum(int(row["total_tokens"]) for row in behavior)
                if behavior
                and all(isinstance(row.get("total_tokens"), int) for row in behavior)
                else None
            ),
        }
    return summary


def grouped_summaries(
    rows: list[dict[str, Any]],
    case_map: dict[str, dict[str, Any]],
    ragas_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for field in ("category", "difficulty"):
        values = sorted({str(case[field]) for case in case_map.values()})
        groups[field] = {}
        for value in values:
            selected = [row for row in rows if case_map[row["case_id"]][field] == value]
            groups[field][value] = aggregate_rows(selected, case_map, ragas_scores)
    return groups


def fmt(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def markdown_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    systems = result["systems"]
    run = result["run"]
    summaries = {name: value["summary"] for name, value in systems.items()}
    lines = [
        "# JurisRAG 固定流程 RAG 与 Agent 对比评测",
        "",
        f"- 状态：**{run['status']}**",
        f"- 运行时间：{run['timestamp']}",
        f"- Git commit：`{run['git_commit']}`（工作树{'有实验改动' if run['git_dirty'] else '干净'}）",
        f"- 数据集：`{run['dataset_name']}`，SHA-256 `{run['dataset_sha256']}`",
        f"- 题目：{run['case_count']} 道，其中可回答 {run['answerable_count']}、多文件融合 {run['multi_file_count']}、边界 {run['boundary_count']}",
        f"- 公平配置：同一回答模型 `{run['answer_model']}`，同一 Top-K={run['top_k']}，同一语料、Embedding 和 Reranker",
        f"- RAGAS：{run['ragas_status']}；评测模型 `{run.get('evaluator_model') or '-'}`",
        "",
        "> 这里的“固定流程 RAG”是项目当前非 Agent 基线，包含父子切块、BM25/向量/精确匹配、RRF、可选重排与证据门。它不是单纯向量 Top-K，报告不把它误称为裸向量 RAG。",
        "",
        "## 总体对比",
        "",
        "| 指标 | 固定流程 RAG | LangGraph Agent |",
        "| --- | ---: | ---: |",
    ]
    metric_rows = [
        ("Parent Recall@1", lambda s: s["retrieval"]["parent_recall_at_1"]),
        ("Parent Recall@3", lambda s: s["retrieval"]["parent_recall_at_3"]),
        ("Parent Recall@5", lambda s: s["retrieval"]["parent_recall_at_5"]),
        ("MRR@5", lambda s: s["retrieval"]["mrr_at_5"]),
        ("多文件全部命中@5", lambda s: s["multi_file"]["all_gold_hit_at_5"]),
        ("边界正确拒答率", lambda s: s["correct_boundary_refusal_rate"]),
        ("有答案题误拒答率", lambda s: s["false_refusal_rate"]),
        ("平均延迟(ms)", lambda s: s["latency_ms"]["mean"]),
        ("P95延迟(ms)", lambda s: s["latency_ms"]["p95"]),
    ]
    for label, getter in metric_rows:
        lines.append(
            f"| {label} | {fmt(getter(summaries.get('fixed_rag', {})), 2 if 'ms' in label else 4)} | "
            f"{fmt(getter(summaries.get('agent', {})), 2 if 'ms' in label else 4)} |"
        )
    for metric in RAGAS_METRICS:
        lines.append(
            f"| RAGAS {metric} | {fmt(summaries.get('fixed_rag', {}).get('ragas', {}).get(metric, {}).get('mean'))} | "
            f"{fmt(summaries.get('agent', {}).get('ragas', {}).get(metric, {}).get('mean'))} |"
        )

    if "agent" in summaries:
        behavior = summaries["agent"].get("agent_behavior", {})
        lines.extend(
            [
                "",
                "## Agent 行为与成本",
                "",
                f"- 平均迭代：{fmt(behavior.get('average_iterations'), 2)}",
                f"- 平均工具调用：{fmt(behavior.get('average_tool_calls'), 2)}",
                f"- 平均 LLM 调用：{fmt(behavior.get('average_llm_calls'), 2)}",
                f"- 重复成功动作：{behavior.get('duplicate_action_count', '-')} 次",
                f"- 验证后纠正：{behavior.get('correction_count', '-')} 次",
                f"- 固定 RAG 降级：{behavior.get('baseline_fallback_count', '-')} 次",
                f"- Token 完整上报：{behavior.get('token_complete_cases', 0)}/{summaries['agent']['execution_success_count']} 题；总 Token：{behavior.get('total_tokens') if behavior.get('total_tokens') is not None else '供应商未完整返回，未估算'}",
            ]
        )

    case_map = {row["case_id"]: row for row in result["dataset_cases"]}
    row_maps = {
        name: {row["case_id"]: row for row in system["cases"]}
        for name, system in systems.items()
    }
    score_map = result.get("ragas_scores", {})
    lines.extend(
        [
            "",
            "## 逐题结果",
            "",
            "| case_id | 类型 | 固定RAG状态 | Agent状态 | 固定RAG Recall@5 | Agent Recall@5 |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for case in result["dataset_cases"]:
        fixed = row_maps.get("fixed_rag", {}).get(case["case_id"], {})
        agent = row_maps.get("agent", {}).get(case["case_id"], {})
        def status(row: dict[str, Any]) -> str:
            if not row:
                return "未运行"
            if row.get("execution_error"):
                return "执行失败"
            return "拒答" if row.get("refused") else "作答"
        lines.append(
            f"| {case['case_id']} | {case['category']} | {status(fixed)} | {status(agent)} | "
            f"{fmt(fixed.get('retrieval_metrics', {}).get('parent_recall_at_5'))} | "
            f"{fmt(agent.get('retrieval_metrics', {}).get('parent_recall_at_5'))} |"
        )

    lines.extend(["", "## 逐题答案、引用与 RAGAS", ""])
    for case in result["dataset_cases"]:
        lines.extend([f"### {case['case_id']}", "", f"问题：{case['question']}", ""])
        for system in SYSTEMS:
            row = row_maps.get(system, {}).get(case["case_id"])
            if not row:
                continue
            lines.append(f"**{SYSTEM_LABELS[system]}**")
            lines.append("")
            if row.get("execution_error"):
                lines.append(f"- 执行错误：`{markdown_escape(row['execution_error'])}`")
            else:
                lines.append(f"- 回答：{markdown_escape(row['answer'])}")
                citation_text = "；".join(
                    f"《{item.get('document_title', '')}》{item.get('article_or_position', '')}"
                    for item in row.get("citations", [])
                ) or "无"
                lines.append(f"- 引用：{markdown_escape(citation_text)}")
                score = score_map.get(f"{system}::{case['case_id']}", {})
                if score:
                    metric_text = "，".join(
                        f"{name}={fmt(score['metrics'][name].get('value'))}"
                        for name in RAGAS_METRICS
                    )
                    lines.append(f"- RAGAS：{metric_text}")
            lines.append("")

    lines.extend(
        [
            "## 解释边界",
            "",
            "- RAGAS 是可复现的自动化 LLM 评测框架，但不是司法机关或法律专家认证。",
            "- Recall/MRR 衡量检索命中，Faithfulness 衡量回答是否受上下文支持，二者不能互相替代。",
            "- 边界题没有支持性 gold 法条，不用 RAGAS Context Recall 硬打分，而使用正确拒答、错误回答和伪引用指标。",
            "- 多文件全部命中表示 Top-5 同时包含该题全部人工 gold 父法条；只命中一个文件不算融合成功。",
            "- 延迟包含本地检索、重排、模型和 Agent 工具链时间；不同网络时段会波动。",
            "- 未完整返回的 Token 不做估算，避免制造虚假成本数据。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_signature(
    dataset: Path,
    top_k: int,
    answer_model: str,
    evaluator_model: str | None,
) -> dict[str, Any]:
    index_db = INDEX_DIR / "chroma.sqlite3"
    return {
        "dataset_sha256": sha256_file(dataset),
        "git_commit": git_commit(),
        "implementation_sha256": implementation_sha256(),
        "top_k": top_k,
        "answer_model": answer_model,
        "evaluator_model": evaluator_model,
        "answer_prompt_versions": ANSWER_PROMPT_VERSIONS,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "index_sha256": sha256_file(index_db),
        "embedding_backend": EMBEDDING_BACKEND,
        "reranker_backend": RERANKER_BACKEND,
    }


async def async_main(args: argparse.Namespace) -> None:
    if args.top_k != 5:
        raise ValueError("本次冻结对比固定使用 top_k=5，避免不同 K 值造成不公平")
    audit = audit_dataset(args.dataset)
    cases = load_comparison_dataset(args.dataset)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
        missing = wanted - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"未知 case_id：{sorted(missing)}")
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("筛选后没有评测题目")

    requested_systems = tuple(value.strip() for value in args.systems.split(",") if value.strip())
    if not requested_systems or any(value not in SYSTEMS for value in requested_systems):
        raise ValueError(f"--systems 只能从 {SYSTEMS} 中选择")

    llm = get_llm_settings()
    if not llm["enabled"] or not llm["api_key"]:
        raise RuntimeError("回答模型未启用或缺少 API Key，不能进行真实系统对比")
    answer_model = f"{llm['provider']}:{llm['model']}"
    await preflight_answer_model()
    evaluator_model: str | None = None
    if not args.skip_ragas:
        _, evaluator_model, _ = evaluator_settings()
    signature = build_signature(args.dataset, args.top_k, answer_model, evaluator_model)
    lookup = gold_parent_lookup()
    system_rows: dict[str, list[dict[str, Any]]] = {}
    for system in requested_systems:
        cache_path = DEFAULT_RESULTS_DIR / f"agent_rag_comparison_v1_{system}_cache.json"
        system_rows[system] = await run_system_cases(
            system,
            cases,
            lookup,
            args.top_k,
            cache_path,
            {**signature, "system": system},
            args.retry_failures,
        )

    ragas_scores: dict[str, dict[str, Any]] = {}
    ragas_error: str | None = None
    if not args.skip_ragas:
        try:
            ragas_scores = await run_ragas_scores(
                cases,
                system_rows,
                DEFAULT_RESULTS_DIR / "agent_rag_comparison_v1_ragas_cache.json",
                {**signature, "system": "shared_ragas"},
                args.ragas_concurrency,
            )
        except Exception as exc:
            ragas_error = safe_error(exc)
            print(f"RAGAS 未完成：{ragas_error}", flush=True)

    case_map = {str(case["case_id"]): case for case in cases}
    systems_output: dict[str, Any] = {}
    for system, rows in system_rows.items():
        systems_output[system] = {
            "summary": aggregate_rows(rows, case_map, ragas_scores),
            "groups": grouped_summaries(rows, case_map, ragas_scores),
            "cases": rows,
        }

    execution_errors = [
        row["execution_error"]
        for rows in system_rows.values()
        for row in rows
        if row.get("execution_error")
    ]
    # 仅有 RAGAS 缓存文件并不代表所有指标都评测成功。
    # 例如评测模型额度耗尽时，缓存仍会保存 error，但 value 为 None。
    # 这里按“系统 × 可回答题 × 五项指标”检查有效值，避免把部分结果误报为 complete。
    expected_ragas_values = (
        sum(1 for case in cases if case["should_answer"])
        * len(system_rows)
        * len(RAGAS_METRICS)
    )
    valid_ragas_values = sum(
        1
        for score in ragas_scores.values()
        for metric in RAGAS_METRICS
        if score.get("metrics", {}).get(metric, {}).get("value") is not None
    )
    ragas_partial = bool(
        not args.skip_ragas
        and not ragas_error
        and ragas_scores
        and valid_ragas_values < expected_ragas_values
    )
    if ragas_partial:
        ragas_error = (
            f"RAGAS 仅完成 {valid_ragas_values}/{expected_ragas_values} 个指标值；"
            "其余题目请查看 ragas_scores 中的 error 字段。"
        )
    status = (
        "generation_complete_ragas_skipped"
        if not execution_errors and args.skip_ragas
        else "complete"
        if not execution_errors and ragas_scores and not ragas_error
        else "incomplete"
    )
    run = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "git_commit": signature["git_commit"],
        "git_dirty": git_is_dirty(),
        "dataset_name": args.dataset.name,
        "dataset_sha256": signature["dataset_sha256"],
        "case_count": len(cases),
        "answerable_count": sum(case["should_answer"] for case in cases),
        "multi_file_count": sum(case["category"] == "multi_file_fusion" for case in cases),
        "boundary_count": sum(not case["should_answer"] for case in cases),
        "top_k": args.top_k,
        "systems": list(requested_systems),
        "answer_model": answer_model,
        "evaluator_model": evaluator_model,
        "ragas_version": "0.4.3" if not args.skip_ragas else None,
        "ragas_status": (
            "skipped_by_user"
            if args.skip_ragas
            else "partial"
            if ragas_partial
            else "failed"
            if ragas_error
            else "complete"
        ),
        "ragas_error": ragas_error,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_backend": RERANKER_BACKEND,
        "rerank_candidates": RERANK_CANDIDATES,
        "child_collection": CHROMA_COLLECTION_NAME,
        "parent_collection": PARENT_CHROMA_COLLECTION_NAME,
        "manifest_sha256": signature["manifest_sha256"],
        "index_sha256": signature["index_sha256"],
        "implementation_sha256": signature["implementation_sha256"],
        "api_key_persisted": False,
    }
    result = {
        "run": run,
        "dataset_audit": audit,
        "dataset_cases": cases,
        "systems": systems_output,
        "ragas_scores": ragas_scores,
    }
    write_json(args.output, result)
    report_path = args.output.with_suffix(".md")
    write_markdown_report(result, report_path)
    print(json.dumps({name: value["summary"] for name, value in systems_output.items()}, ensure_ascii=False, indent=2))
    print(f"完整结果：{args.output}")
    print(f"中文报告：{report_path}")
    if status == "incomplete":
        raise RuntimeError("对比评测未完整成功；请查看结果中的 run 状态和错误字段")


def main() -> None:
    parser = argparse.ArgumentParser(description="同题对比固定 RAG、Agent 与 RAGAS")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--systems", default="fixed_rag,agent")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ragas-concurrency", type=int, default=1)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于0")
    if args.ragas_concurrency <= 0:
        raise ValueError("--ragas-concurrency 必须大于0")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
