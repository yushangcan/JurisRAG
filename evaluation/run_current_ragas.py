"""冻结当前运行配置，断点续跑 30 题 RAG / Agent 与真正的 RAGAS 评分。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from agent.config import get_agent_settings
from agent.service import run_agent
from evaluation.audit_agent_rag_comparison import DEFAULT_DATASET, audit_dataset, load_comparison_dataset
from evaluation.audit_gold_dataset import load_verified_parents
from evaluation.run_lightweight_eval import retrieval_metrics
from online_retrieval.retriever import LegalRetriever
from shared.config import INDEX_PATH, PROJECT_ROOT, RETRIEVAL_MODE, get_llm_settings
from shared.llm import build_chat_model

METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy", "factual_correctness")
MODES = ("rag", "agent")
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/results/current_codez_ragas_20261017.json"
DEFAULT_EMBEDDING = PROJECT_ROOT / "data/models/fastembed/models--Qdrant--bge-small-zh-v1.5/snapshots/46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hash(folders: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for folder in folders:
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def environment() -> dict:
    versions = {}
    for name in ("ragas", "pydantic", "openai", "instructor", "fastembed", "onnxruntime", "numpy", "langchain-community"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": versions}


def safe_signature(dataset: Path, embedding_path: Path, top_k: int) -> dict:
    settings = get_llm_settings()
    if not settings["enabled"] or not settings["api_key"]:
        raise ValueError("未启用或未配置模型，不能执行本次测评")
    return {"dataset_sha256": sha256(dataset), "index_sha256": sha256(INDEX_PATH),
            "runtime_sha256": source_hash(("agent", "shared", "online_retrieval", "offline_ingestion")),
            "model": {key: settings[key] for key in ("provider", "base_url", "model", "wire_api", "reasoning_effort", "disable_response_storage")},
            "agent_settings": asdict(get_agent_settings()), "retrieval_mode": RETRIEVAL_MODE, "top_k": top_k,
            "ragas_version": "0.4.3", "answer_relevancy_strictness": 3,
            "factual_correctness": {"mode": "f1", "beta": 1.0, "atomicity": "low", "coverage": "low"},
            "embedding": {"model": "BAAI/bge-small-zh-v1.5", "role": "query_embed_for_both_questions",
                          "files": {name: sha256(embedding_path / name) for name in
                                    ("model_optimized.onnx", "config.json", "tokenizer.json", "tokenizer_config.json")}},
            "context_policy": "full_final_citation_quotes_in_returned_order",
            "judge_prompt_adapter": "stock_ragas_plus_chinese_output_and_json_schema_v1",
            "judge_timeout_seconds": 120, "judge_attempts_per_call": 2}


def write_checkpoint(state: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)


def valid_score(score: dict | None) -> float | None:
    if not score or score.get("status") != "ok":
        return None
    value = score.get("value")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def usage_summary(events: list[dict]) -> dict:
    totals = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        known = [event[key] for event in events if isinstance(event.get(key), int)]
        totals[key] = sum(known) if len(known) == len(events) else None
        totals["known_" + key] = sum(known) if known else (0 if not events else None)
        totals[key + "_reported_calls"] = len(known)
    return {"call_count": len(events), "failed_calls": sum(not e["success"] for e in events), **totals}


async def generate(state: dict, cases: list[dict], output: Path) -> None:
    settings = get_agent_settings()
    model = build_chat_model(timeout=settings.llm_timeout_seconds)
    if model is None:
        raise ValueError("模型不可用，未生成测评答案")
    lookup = load_verified_parents()
    retriever = LegalRetriever()
    semaphore = asyncio.Semaphore(min(2, settings.max_concurrent_requests))
    state.setdefault("generation_environment", environment())
    state.setdefault("generation_runner_sha256", sha256(Path(__file__)))
    state["generation_concurrency"] = min(2, settings.max_concurrent_requests)

    async def one(case: dict, mode: str):
        key = f"{case['case_id']}:{mode}"
        if key in state["records"]:
            return
        async with semaphore:
            started = perf_counter()
            # 生成只接收原问题；gold / reference 不传入生产链路。
            record = {"case_id": case["case_id"], "mode": mode, "question": case["question"],
                      "should_answer": case["should_answer"], "category": case["category"],
                      "reference_answer": case["reference_answer"],
                      "gold_parent_ids": [lookup[(g["raw_file"], g["article_number"])][0]["parent_id"]
                                          for g in case["gold_evidence"]], "scores": {}}
            try:
                result = await run_agent(case["question"], state["signature"]["top_k"], mode=mode,
                                         model=model, settings=settings)
                record["result"] = result
                ids = [citation["parent_id"] for citation in result["citations"]]
                record["retrieval"] = retrieval_metrics(ids, record["gold_parent_ids"])
                parents = {p["parent_id"]: p for p in await asyncio.to_thread(retriever.store.fetch, ids, "parent")}
                record["invalid_citation_count"] = sum(
                    c["parent_id"] not in parents or c["quote"] != parents[c["parent_id"]]["text"]
                    for c in result["citations"])
                record["generation_status"] = "ok"
            except Exception as exc:
                record.update(generation_status="failed", error_type=type(exc).__name__)
            record["latency_ms"] = round((perf_counter() - started) * 1000, 2)
            state["records"][key] = record
            write_checkpoint(state, output)
            result = record.get("result", {})
            print(f"GENERATED {len(state['records'])}/{state['expected_generations']} {key} "
                  f"{result.get('status', record['generation_status'])} "
                  f"{result.get('generation_mode', '')} {record['latency_ms'] / 1000:.1f}s", flush=True)

    await asyncio.gather(*(one(case, mode) for case in cases for mode in MODES))


async def score(state: dict, cases: list[dict], output: Path, embedding_path: Path,
                metrics: list[str], concurrency: int, retry_failed: bool) -> None:
    print("Loading RAGAS and local BGE ...", flush=True)
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness, FactualCorrectness
    from evaluation.ragas_adapters import ChineseQuestionEmbeddings, ResponsesJudge

    if importlib.metadata.version("ragas") != state["signature"]["ragas_version"]:
        raise ValueError("RAGAS 版本与冻结配置不符")
    state.setdefault("scoring_environment", environment())
    if state["scoring_environment"] != environment():
        raise ValueError("评分环境改变，拒绝混用不同依赖版本")
    state.setdefault("scoring_concurrency", concurrency)
    state.setdefault("scoring_source_sha256", source_hash(("evaluation",)))
    embeddings = ChineseQuestionEmbeddings(embedding_path)
    await embeddings.aembed_text("政府采购应遵循哪些原则？")
    model = build_chat_model(timeout=state["signature"]["judge_timeout_seconds"])
    semaphore = asyncio.Semaphore(concurrency)
    case_ids = {case["case_id"] for case in cases}
    factories = {"context_precision": ContextPrecision, "context_recall": ContextRecall,
                 "faithfulness": Faithfulness, "answer_relevancy": AnswerRelevancy, "factual_correctness": FactualCorrectness}

    async def one(key: str, record: dict, name: str):
        previous = record["scores"].get(name)
        if previous and (previous["status"] in {"ok", "not_applicable"}
                         or (previous["status"] == "failed" and not retry_failed)):
            return
        checkpoint = record["scores"].setdefault(name, {"value": None, "status": "pending", "calls": {}})
        if record["generation_status"] != "ok":
            checkpoint.update(status="not_applicable", reason="generation_failed")
            write_checkpoint(state, output)
            return
        result = record["result"]
        contexts = [citation["quote"] for citation in result["citations"]]
        if name in {"context_precision", "context_recall", "faithfulness"} and not contexts:
            checkpoint.update(status="not_applicable", reason="empty_final_contexts")
            write_checkpoint(state, output)
            return
        async with semaphore:
            checkpoint["status"] = "running"
            started = perf_counter()
            judge = ResponsesJudge(model, checkpoint, lambda: write_checkpoint(state, output),
                                   timeout=state["signature"]["judge_timeout_seconds"],
                                   attempts=state["signature"]["judge_attempts_per_call"])
            kwargs = {"llm": judge}
            if name == "answer_relevancy":
                kwargs.update(embeddings=embeddings, strictness=3)
            elif name == "factual_correctness":
                kwargs.update(state["signature"]["factual_correctness"])
            metric = factories[name](**kwargs)
            arguments = {"response": result["answer"], "reference": record["reference_answer"]} if name == "factual_correctness" else {
                "user_input": record["question"], "reference": record["reference_answer"], "retrieved_contexts": contexts}
            if name == "faithfulness":
                arguments = {"user_input": record["question"], "response": result["answer"], "retrieved_contexts": contexts}
            elif name == "answer_relevancy":
                arguments = {"user_input": record["question"], "response": result["answer"]}
            try:
                value = float((await metric.ascore(**arguments)).value)
                if not math.isfinite(value):
                    raise ValueError("RAGAS returned non-finite score")
                checkpoint.update(status="ok", value=value, error_type=None)
            except Exception as exc:
                checkpoint.update(status="failed", value=None, error_type=type(exc).__name__)
            checkpoint["last_attempt_duration_ms"] = round((perf_counter() - started) * 1000, 2)
            write_checkpoint(state, output)
            done = sum(s["status"] in {"ok", "failed", "not_applicable"}
                       for row in state["records"].values() for s in row["scores"].values())
            print(f"SCORED {done}/{state['expected_scores']} {key} {name} "
                  f"{checkpoint['status']} {checkpoint['value']}", flush=True)

    await asyncio.gather(*(one(key, record, name) for key, record in state["records"].items()
                           if record["case_id"] in case_ids and record["should_answer"] for name in metrics))


def summarize(state: dict) -> dict:
    summary = {}
    mean = lambda values: statistics.fmean(values) if values else None
    records = list(state["records"].values())
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        valid = [row for row in rows if row["generation_status"] == "ok"]
        all_answerable = [row for row in rows if row["should_answer"]]
        answerable = [row for row in valid if row["should_answer"]]
        boundary = [row for row in valid if not row["should_answer"]]
        generation_events = [event for row in valid for event in row["result"]["agent_trace"]["llm_usage"]["events"]]
        judge_events = [event for row in rows for metric in row["scores"].values()
                        for call in metric["calls"].values() for attempt in call["attempts"] for event in attempt["events"]]
        summary[mode] = {"generated_count": len(rows), "generation_failed": len(rows) - len(valid),
                         "answerable_count": len(answerable), "boundary_count": len(boundary),
                         "answerable_with_evidence": sum(not row["result"]["insufficient_evidence"] for row in answerable),
                         "false_refusal_count": sum(row["result"]["insufficient_evidence"] for row in answerable),
                         "correct_boundary_refusal_count": sum(row["result"]["insufficient_evidence"] and not row["result"]["citations"] for row in boundary),
                         "invalid_citation_count": sum(row["invalid_citation_count"] for row in valid),
                         "warning_count": sum(bool(row["result"]["warning"]) for row in valid),
                         "generation_modes": dict(Counter(row["result"]["generation_mode"] for row in valid)),
                         "controllers": dict(Counter(row["result"]["agent_trace"]["controller"] for row in valid)),
                         "mean_latency_ms": mean([row["latency_ms"] for row in valid]),
                         "generation_usage": usage_summary(generation_events), "judge_usage": usage_summary(judge_events),
                         "retrieval": {name: mean([row["retrieval"][name] for row in answerable]) for name in
                                       ("parent_recall_at_1", "parent_recall_at_3", "parent_recall_at_5", "mrr_at_5", "all_gold_hit_at_5")},
                         "scores": {}}
        for name in METRICS:
            values = [valid_score(row["scores"].get(name)) for row in all_answerable]
            values = [value for value in values if value is not None]
            statuses = Counter(row["scores"].get(name, {}).get("status", "pending") for row in all_answerable)
            summary[mode]["scores"][name] = {"mean": mean(values), "valid_n": len(values),
                                               "eligible_n": state["audit"]["answerable_count"], "statuses": dict(statuses)}
    summary["paired"] = {}
    for name in METRICS:
        paired = []
        for row in records:
            other = state["records"].get(row["case_id"] + ":agent")
            if row["mode"] == "rag" and row["should_answer"] and other:
                left, right = valid_score(row["scores"].get(name)), valid_score(other["scores"].get(name))
                if left is not None and right is not None:
                    paired.append((left, right))
        summary["paired"][name] = {"n": len(paired), "rag": mean([p[0] for p in paired]),
                                    "agent": mean([p[1] for p in paired]), "agent_minus_rag": mean([p[1] - p[0] for p in paired])}
    return summary


def write_report(state: dict, output: Path) -> None:
    summary = state["summary"] = summarize(state)
    records = state["records"]
    terminal = [s for row in records.values() for s in row["scores"].values()
                if s["status"] in {"ok", "failed", "not_applicable"}]
    state["status"] = "incomplete" if len(records) != state["expected_generations"] or len(terminal) != state["expected_scores"] else (
        "complete" if all(s["status"] == "ok" for s in terminal) else "complete_with_missing_scores")
    fmt = lambda value: "—" if value is None else f"{value:.4f}"
    lines = ["# 当前 CodeZ 版本 RAGAS 实测", "", f"- 运行状态：`{state['status']}`；生成 {len(records)}/{state['expected_generations']}；评分已结束 {len(terminal)}/{state['expected_scores']}。",
             f"- 生产版本：`{state['git_commit']}`；数据集：`{state['dataset']}`（开发回归集）。",
             f"- 模型：`{state['signature']['model']['provider']}/{state['signature']['model']['model']}`，Responses，reasoning effort=max，store=false。回答与裁判使用同一模型。",
             "- 30 道题 × 2 条链路；21 道有依据题 × 5 个 RAGAS 指标；9 道边界题只计算拒答与引用检查。参考答案和 gold 不进入生成链路。",
             "- RAGAS 0.4.3 collections 原始指标；仅补充 JSON Schema 与中文输出要求。AnswerRelevancy strictness=3，使用本地 BAAI/bge-small-zh-v1.5 真实语义向量（512维）；两种问题均 query_embed。FactualCorrectness 为 F1、beta=1、atomicity=low、coverage=low。",
             "- 上下文为最终正式引用的完整父证据，保持返回顺序；不是所有检索候选，也不等于模型曾看到的截断片段。所有交付答案原样评分，包括原文回退。无上下文的证据指标记为不适用，调用失败或非有限值留空。",
             "", "| RAGAS 指标 | 固定 RAG 均值（有效/有依据题） | Agent 均值（有效/有依据题） | 配对有效题 | 配对 Agent−RAG |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in METRICS:
        cells = [f"{fmt(summary[mode]['scores'][name]['mean'])} ({summary[mode]['scores'][name]['valid_n']}/{summary[mode]['scores'][name]['eligible_n']})" for mode in MODES]
        paired = summary["paired"][name]
        lines.append(f"| {name} | {' | '.join(cells)} | {paired['n']} | {fmt(paired['agent_minus_rag'])} |")
    lines.extend(["", "| 系统交付与开销 | 固定 RAG | Agent |", "| --- | ---: | ---: |"])
    for key, label in (("answerable_with_evidence", "有依据题返回证据数 / 21"), ("false_refusal_count", "有依据题误拒答数 / 21"),
                       ("correct_boundary_refusal_count", "边界题正确拒答数 / 9"), ("invalid_citation_count", "引用ID或原文不匹配数"),
                       ("warning_count", "含告警的请求数"), ("mean_latency_ms", "平均请求耗时 ms（并发最多2，非吞吐测试）")):
        lines.append(f"| {label} | " + " | ".join(fmt(summary[m][key]) if isinstance(summary[m][key], float) else str(summary[m][key]) for m in MODES) + " |")
    for name in ("parent_recall_at_5", "mrr_at_5", "all_gold_hit_at_5"):
        lines.append(f"| 最终引用 {name} | " + " | ".join(fmt(summary[m]["retrieval"][name]) for m in MODES) + " |")
    for group, label in (("generation_usage", "生成"), ("judge_usage", "裁判")):
        lines.append(f"| {label}模型调用数（含失败） | " + " | ".join(str(summary[m][group]["call_count"]) for m in MODES) + " |")
        lines.append(f"| {label}失败调用数 | " + " | ".join(str(summary[m][group]["failed_calls"]) for m in MODES) + " |")
        lines.append(f"| {label}已知 total tokens（报告调用数/全部调用数） | " + " | ".join(
            f"{summary[m][group]['known_total_tokens']} ({summary[m][group]['total_tokens_reported_calls']}/{summary[m][group]['call_count']})" for m in MODES) + " |")
    lines.extend(["", "## 解释与限制", "", "- Context Precision/Recall 衡量最终交付证据对参考答案的支持；Faithfulness 衡量答案能否由这些证据支持；Answer Relevancy 衡量相关性；Factual Correctness 衡量与参考答案的事实一致性。它们不能替代法律专家或现行法效核验。",
                  "- 每项同时报告有效评分数与相同题目的配对差值；不同分母的均值不可直接作为优劣结论。拒答并未被删除：证据指标缺失，相关性和事实一致性继续评分。有效评分均值不能脱离返回率理解。",
                  "- 同模型作为回答者与裁判可能存在偏好；题集参与过开发，且样本较小。这次不是独立泛化测试，也没有统计显著性或法律正确性保证。参考答案的覆盖范围仍会影响 F1。",
                  "- 精确原文快路径保留在系统结果中，可能不调用模型。Agent 的校验失败、预算停止、超时和规则回退均保留；没有调整生产行为来追求分数。",
                  "- Token 只来自供应商 usage；未知用量留空，已知 token 可能只是下界。没有核验价格，不换算人民币成本。裁判开销单独列出，不算作线上业务成本。",
                  "", "## 控制器与输出类型", ""])
    for mode in MODES:
        lines.extend([f"- {mode} controllers：`{json.dumps(summary[mode]['controllers'], ensure_ascii=False)}`",
                      f"- {mode} generation_modes：`{json.dumps(summary[mode]['generation_modes'], ensure_ascii=False)}`"])
    lines.extend(["", "## 逐题结果", "", "CP/CR/F/AR/FC 分别为上面五项指标；`—` 表示不适用、未完成或失败，详见完整 JSON。", "",
                  "| case_id | 模式 | 交付状态 | 输出类型 | CP | CR | F | AR | FC |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for key in sorted(records):
        row = records[key]
        result = row.get("result", {})
        cells = " | ".join(fmt(valid_score(row["scores"].get(name))) for name in METRICS)
        lines.append(f"| {row['case_id']} | {row['mode']} | {result.get('status', 'generation_failed')} | {result.get('generation_mode', '—')} | {cells} |")
    failures = [(key, name, s) for key, row in records.items() for name, s in row["scores"].items() if s["status"] != "ok"]
    lines.extend(["", "## 缺失与失败", ""])
    lines.extend([f"- {key} / {name}：{s['status']}，{s.get('reason') or s.get('error_type', 'pending')}。" for key, name, s in failures] or ["无缺失或失败评分。"])
    lines.extend(["", "## 复现与冻结信息", "", f"- 开始 UTC：{state['started_utc']}；最后保存 UTC：{state.get('updated_utc', '—')}。",
                  f"- 完整答案、证据、子调用缓存与 usage：`{output.name}`（本机保存，遵循仓库规则不提交 JSON）。",
                  "- 同一路径重跑会检查数据/索引/生产代码/模型/评分参数签名，已有答案不会重新生成；成功评分子调用会复用，三个反问采样使用不同编号。`--retry-failed` 只重试失败评分，不重抽业务答案。",
                  "", "```powershell", ".\\.venv-lite\\Scripts\\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase generate",
                  ".\\.venv-eval\\Scripts\\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase score",
                  ".\\.venv-lite\\Scripts\\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase report", "```", "", "```json",
                  json.dumps({"signature": state["signature"], "generation_environment": state.get("generation_environment"),
                              "scoring_environment": state.get("scoring_environment"), "generation_concurrency": state.get("generation_concurrency"),
                              "scoring_concurrency": state.get("scoring_concurrency"),
                              "generation_runner_sha256": state.get("generation_runner_sha256"),
                              "scoring_source_sha256": state.get("scoring_source_sha256")}, ensure_ascii=False, indent=2), "```", ""])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    write_checkpoint(state, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "score", "report", "all"), default="all")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-path", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--case-id", action="append", help="预检指定题；默认处理全部30题")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--scoring-concurrency", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    audit = audit_dataset(args.dataset)
    cases = load_comparison_dataset(args.dataset)
    signature = safe_signature(args.dataset, args.embedding_path, args.top_k)
    if args.output.exists():
        state = json.loads(args.output.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("运行签名改变，不能混用缓存；请指定新的输出文件")
    else:
        state = {"started_utc": datetime.now(timezone.utc).isoformat(), "signature": signature,
                 "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "dataset": args.dataset.name, "audit": audit, "expected_generations": len(cases) * 2,
                 "expected_scores": sum(case["should_answer"] for case in cases) * 2 * len(METRICS), "records": {}}
    selected = [case for case in cases if not args.case_id or case["case_id"] in args.case_id]
    if not selected:
        raise ValueError("没有匹配的题目")
    print(f"Phase={args.phase}; selected_cases={len(selected)}; cached_generations={len(state['records'])}", flush=True)
    if args.phase in {"generate", "all"}:
        asyncio.run(generate(state, selected, args.output))
    if args.phase in {"score", "all"}:
        asyncio.run(score(state, selected, args.output, args.embedding_path, args.metrics,
                          args.scoring_concurrency, args.retry_failed))
    write_report(state, args.output)
    print(f"Saved {args.output.name}: {state['status']}", flush=True)


if __name__ == "__main__":
    main()
