"""对最终检索结果运行完整的 RAGAS 生成质量评测。

评测分为两个阶段：

1. 只根据已保存的 Top-K 法规证据生成答案，不把参考答案泄露给回答模型。
2. 使用 RAGAS 0.4.3 评估上下文精确率、上下文召回率、忠实度、
   回答相关性和事实正确性。

生成答案、指标分数和错误都会分批保存，因此网络中断后可以继续运行。
密钥只从环境变量读取，任何结果文件都不会保存密钥或请求头。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from pathlib import Path
from statistics import fmean
from typing import Any, Awaitable, Callable

import ragas
from openai import AsyncOpenAI
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from shared.embedding import embed_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_INPUT = RESULTS_DIR / "bge_multi_recall_refusal_v2_top5.json"
DEFAULT_OUTPUT = RESULTS_DIR / "ragas_full_final_v2.json"
DEFAULT_GENERATION_CACHE = RESULTS_DIR / "ragas_full_final_v2_generations.json"
DEFAULT_SCORE_CACHE = RESULTS_DIR / "ragas_full_final_v2_scores.json"
METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
)


def provider_settings() -> tuple[str, str, str]:
    """读取评测服务配置，但不记录或打印密钥。"""

    key = os.getenv("RAGAS_EVALUATOR_API_KEY", "").strip() or os.getenv(
        "DEEPSEEK_API_KEY", ""
    ).strip()
    model = (
        os.getenv("RAGAS_EVALUATOR_MODEL", "").strip()
        or os.getenv("DEEPSEEK_MODEL", "").strip()
        or "deepseek-chat"
    )
    base_url = (
        os.getenv("RAGAS_EVALUATOR_BASE_URL", "").strip()
        or os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or "https://api.deepseek.com"
    )
    if not key:
        raise RuntimeError(
            "缺少 RAGAS_EVALUATOR_API_KEY 或 DEEPSEEK_API_KEY，无法运行 LLM 评测"
        )
    return key, model, base_url


def read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON，并让错误信息保留具体文件路径。"""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取评测文件: {path}") from exc


def write_json(path: Path, value: dict[str, Any]) -> None:
    """以稳定的 UTF-8 格式写入可检查的中间结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def safe_error(exc: Exception) -> str:
    """只记录异常类型和消息，不拼接客户端配置或请求头。"""

    return f"{type(exc).__name__}: {exc}"


def evidence_prompt(case: dict[str, Any]) -> str:
    """构造只包含问题和检索证据的回答提示，不包含人工参考答案。"""

    contexts = []
    for index, item in enumerate(case.get("retrieved", []), start=1):
        title = str(item.get("title", ""))
        article = str(item.get("article_number", ""))
        text = str(item.get("text", ""))
        contexts.append(f"证据{index}：《{title}》{article}\n{text}")
    evidence = "\n\n".join(contexts)
    return (
        "你是政府采购与招标投标法规检索助手。只能依据下列证据回答，"
        "不得新增证据中没有的法规名称、条号、数字、期限或法律结论。"
        "如果不同证据分别支持结论，请分点说明；回答末尾不要编造来源。\n\n"
        f"问题：{case['question']}\n\n检索证据：\n{evidence}"
    )


async def generate_one(
    case: dict[str, Any],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """为一条检索结果生成证据约束答案，并保留失败状态。"""

    if not case.get("retrieved"):
        return {
            "case_id": case["case_id"],
            "question": case["question"],
            "response": None,
            "error": "没有检索上下文，未生成答案",
        }
    try:
        async with semaphore:
            completion = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=800,
                messages=[
                    {
                        "role": "system",
                        "content": "你只能根据用户消息中提供的法规证据作答。",
                    },
                    {"role": "user", "content": evidence_prompt(case)},
                ],
            )
        response = completion.choices[0].message.content
        if not response or not response.strip():
            raise ValueError("回答模型返回空内容")
        return {
            "case_id": case["case_id"],
            "question": case["question"],
            "response": response.strip(),
            "error": None,
        }
    except Exception as exc:
        return {
            "case_id": case["case_id"],
            "question": case["question"],
            "response": None,
            "error": safe_error(exc),
        }


async def build_generations(
    cases: list[dict[str, Any]],
    cache_path: Path,
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """生成缺失答案，每批完成后立即保存，支持断点续跑。"""

    cached_rows: list[dict[str, Any]] = []
    if cache_path.exists():
        cached_rows = list(read_json(cache_path).get("cases", []))
    cached = {str(row["case_id"]): row for row in cached_rows}

    def needs_generation(case: dict[str, Any]) -> bool:
        """新题或上次生成失败的题重新调用回答模型。"""

        row = cached.get(case["case_id"])
        if row is None:
            return True
        return not row.get("response")

    pending = [case for case in cases if needs_generation(case)]
    semaphore = asyncio.Semaphore(concurrency)
    batch_size = max(concurrency * 2, 1)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        rows = await asyncio.gather(
            *(generate_one(case, client, model, semaphore) for case in batch)
        )
        cached.update({str(row["case_id"]): row for row in rows})
        ordered = [cached[case["case_id"]] for case in cases if case["case_id"] in cached]
        write_json(
            cache_path,
            {
                "run": {
                    "model": model,
                    "answer_prompt": "evidence_only_legal_answer_v1",
                    "reference_answer_exposed_to_generator": False,
                },
                "cases": ordered,
            },
        )
        print(f"答案生成进度: {len(ordered)}/{len(cases)}", flush=True)
    return cached


class ProjectBgeEmbedding(BaseRagasEmbedding):
    """让 RAGAS AnswerRelevancy 复用项目本地 BGE 查询向量。"""

    def __init__(self) -> None:
        super().__init__()
        # FastEmbed 的 ONNX session 不需要并行执行这些很短的评测文本；
        # 串行化能避免多个 RAGAS 协程同时访问同一 session。
        self._lock = threading.Lock()

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        del kwargs
        with self._lock:
            return embed_query(text)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text, **kwargs)


async def score_metric(
    metric_name: str,
    call: Callable[[], Awaitable[Any]],
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any]]:
    """执行一个 RAGAS 指标，并把单指标失败转换为结构化结果。"""

    try:
        async with semaphore:
            result = await call()
        return metric_name, {
            "value": float(result.value),
            "reason": getattr(result, "reason", None),
            "error": None,
        }
    except Exception as exc:
        return metric_name, {
            "value": None,
            "reason": None,
            "error": safe_error(exc),
        }


async def score_one(
    case: dict[str, Any],
    generation: dict[str, Any],
    evaluator_llm: Any,
    embeddings: ProjectBgeEmbedding,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """对一题运行五个指标；生成失败时仍评估上下文指标。"""

    contexts = [str(item["text"]) for item in case.get("retrieved", [])]
    response = generation.get("response")
    calls: list[tuple[str, Callable[[], Awaitable[Any]]]] = [
        (
            "context_precision",
            lambda: ContextPrecision(llm=evaluator_llm).ascore(
                user_input=case["question"],
                reference=case["reference_answer"],
                retrieved_contexts=contexts,
            ),
        ),
        (
            "context_recall",
            lambda: ContextRecall(llm=evaluator_llm).ascore(
                user_input=case["question"],
                reference=case["reference_answer"],
                retrieved_contexts=contexts,
            ),
        ),
    ]
    if response:
        calls.extend(
            [
                (
                    "faithfulness",
                    lambda: Faithfulness(llm=evaluator_llm).ascore(
                        user_input=case["question"],
                        response=response,
                        retrieved_contexts=contexts,
                    ),
                ),
                (
                    "answer_relevancy",
                    lambda: AnswerRelevancy(
                        llm=evaluator_llm, embeddings=embeddings, strictness=3
                    ).ascore(user_input=case["question"], response=response),
                ),
                (
                    "factual_correctness",
                    lambda: FactualCorrectness(
                        llm=evaluator_llm,
                        mode="f1",
                        atomicity="high",
                        coverage="high",
                    ).ascore(response=response, reference=case["reference_answer"]),
                ),
            ]
        )

    scored_pairs = await asyncio.gather(
        *(score_metric(name, call, semaphore) for name, call in calls)
    )
    metrics = dict(scored_pairs)
    if not response:
        for name in ("faithfulness", "answer_relevancy", "factual_correctness"):
            metrics[name] = {
                "value": None,
                "reason": None,
                "error": f"答案生成失败: {generation.get('error')}",
            }
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "response": response,
        "generation_error": generation.get("error"),
        "metrics": metrics,
    }


async def build_scores(
    cases: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
    cache_path: Path,
    evaluator_llm: Any,
    embeddings: ProjectBgeEmbedding,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """评估缺失题目，并在每个小批次后保存分数。"""

    cached_rows: list[dict[str, Any]] = []
    if cache_path.exists():
        cached_rows = list(read_json(cache_path).get("cases", []))
    cached = {str(row["case_id"]): row for row in cached_rows}

    def needs_scoring(case: dict[str, Any]) -> bool:
        """新题或存在失败指标的题都重新评分，避免失败缓存永久固化。"""

        row = cached.get(case["case_id"])
        if row is None:
            return True
        return any(
            row.get("metrics", {}).get(name, {}).get("value") is None
            for name in METRIC_NAMES
        )

    pending = [case for case in cases if needs_scoring(case)]
    semaphore = asyncio.Semaphore(concurrency)
    # 一题包含五个 LLM 指标；一次只启动少量题，避免瞬间制造过多请求。
    batch_size = max(concurrency, 1)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        rows = await asyncio.gather(
            *(
                score_one(
                    case,
                    generations[case["case_id"]],
                    evaluator_llm,
                    embeddings,
                    semaphore,
                )
                for case in batch
            )
        )
        cached.update({str(row["case_id"]): row for row in rows})
        ordered = [cached[case["case_id"]] for case in cases if case["case_id"] in cached]
        write_json(cache_path, {"cases": ordered})
        print(f"RAGAS 评分进度: {len(ordered)}/{len(cases)}", flush=True)
    return cached


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按指标忽略失败题求均值，同时公开成功数，禁止静默丢失。"""

    summary: dict[str, Any] = {
        "case_count": len(rows),
        "generation_success_count": sum(not row.get("generation_error") for row in rows),
        "fully_scored_count": 0,
        "metrics": {},
    }
    for name in METRIC_NAMES:
        values = [
            float(row["metrics"][name]["value"])
            for row in rows
            if row["metrics"][name]["value"] is not None
        ]
        summary["metrics"][name] = {
            "mean": fmean(values) if values else None,
            "successful_cases": len(values),
            "failed_cases": len(rows) - len(values),
        }
    summary["fully_scored_count"] = sum(
        all(row["metrics"][name]["value"] is not None for name in METRIC_NAMES)
        for row in rows
    )
    error_messages = [
        str(row["metrics"][name]["error"])
        for row in rows
        for name in METRIC_NAMES
        if row["metrics"][name]["error"]
    ]
    summary["evaluation_complete"] = summary["fully_scored_count"] == len(rows)
    summary["incomplete_reason"] = (
        "评测接口余额不足（HTTP 402），缺失指标未计入均值"
        if any("402" in message and "Insufficient Balance" in message for message in error_messages)
        else "部分指标调用失败，缺失指标未计入均值"
        if error_messages
        else None
    )
    return summary


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    """生成中文汇总，并保留逐题分数和错误状态。"""

    summary = result["summary"]
    retrieval = result["retrieval_summary"]
    descriptions = {
        "context_precision": "召回上下文中与问题和参考答案相关的比例",
        "context_recall": "参考答案所需信息被召回上下文覆盖的程度",
        "faithfulness": "生成答案中的陈述能否由检索证据支持",
        "answer_relevancy": "生成答案是否直接回答用户问题",
        "factual_correctness": "生成答案与人工参考答案的事实一致程度",
    }
    title = (
        "# JurisRAG 最终版本 RAGAS 完整评测"
        if summary["evaluation_complete"]
        else "# JurisRAG 最终版本 RAGAS 评测（部分完成）"
    )
    lines = [
        title,
        "",
        f"- RAGAS版本：`{result['run']['ragas_version']}`",
        f"- 评测模型：`{result['run']['evaluator_model']}`",
        f"- 数据集：`{result['run']['dataset_version']}`",
        f"- 总题数：{summary['case_count']}",
        f"- 答案生成成功：{summary['generation_success_count']}/{summary['case_count']}",
        f"- 五项指标全部成功：{summary['fully_scored_count']}/{summary['case_count']}",
    ]
    if not summary["evaluation_complete"]:
        lines.extend(
            [
                f"- **状态：未完成。{summary['incomplete_reason']}。**",
                "- 下表每个均值只使用该指标成功返回的题目，失败题不是0分。",
            ]
        )
    lines.extend([
        "",
        "## RAGAS 汇总",
        "",
        "| 指标 | 平均分 | 成功题数 | 失败题数 | 中文含义 |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for name in METRIC_NAMES:
        metric = summary["metrics"][name]
        mean_text = "-" if metric["mean"] is None else f"{metric['mean']:.4f}"
        lines.append(
            f"| {name} | {mean_text} | {metric['successful_cases']} | "
            f"{metric['failed_cases']} | {descriptions[name]} |"
        )
    lines.extend(
        [
            "",
            "## 确定性检索指标",
            "",
            "| 指标 | 结果 |",
            "| --- | ---: |",
            f"| Parent Recall@1 | {retrieval.get('parent_recall_at_1', 0):.4f} |",
            f"| Parent Recall@3 | {retrieval.get('parent_recall_at_3', 0):.4f} |",
            f"| Parent Recall@5 | {retrieval.get('parent_recall_at_5', 0):.4f} |",
            f"| MRR@5 | {retrieval.get('reciprocal_rank_at_5', 0):.4f} |",
            f"| RAGAS ID Context Precision@5 | {retrieval.get('ragas_id_context_precision_at_5', 0):.4f} |",
            f"| RAGAS ID Context Recall@5 | {retrieval.get('ragas_id_context_recall_at_5', 0):.4f} |",
            "",
            "## 逐题结果",
            "",
            "| case_id | CP | CR | 忠实度 | 回答相关性 | 事实正确性 | 状态 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["cases"]:
        values = []
        errors = []
        for name in METRIC_NAMES:
            metric = row["metrics"][name]
            values.append("-" if metric["value"] is None else f"{metric['value']:.4f}")
            if metric["error"]:
                errors.append(name)
        status = "成功" if not errors else "失败: " + ",".join(errors)
        lines.append(
            f"| {row['case_id']} | {' | '.join(values)} | {status} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- Recall@K、MRR和ID指标基于人工gold ID，是确定性检索指标。",
            "- 五项RAGAS语义指标由LLM裁判给出，不是司法机关或法律专家认证。",
            "- Faithfulness高只表示答案有上下文依据，不自动代表依据是现行有效法规。",
            "- Factual Correctness依赖当前人工参考答案的完整性。",
            "- 本数据集仍是开发集，不能将结果外推为生产环境法律准确率。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    """运行答案生成、五指标评分和报告汇总。"""

    if ragas.__version__ != "0.4.3":
        raise RuntimeError(f"评测固定使用 ragas==0.4.3，当前为 {ragas.__version__}")
    baseline = read_json(args.input)
    cases = list(baseline.get("cases", []))
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("输入结果没有可评测案例")

    if args.report_only:
        cached_rows = list(read_json(args.score_cache).get("cases", []))
        scores = {str(row["case_id"]): row for row in cached_rows}
        missing = [case["case_id"] for case in cases if case["case_id"] not in scores]
        if missing:
            raise ValueError(f"评分缓存缺少案例，无法生成报告: {missing}")
        ordered = [scores[case["case_id"]] for case in cases]
        old_result = read_json(args.output) if args.output.exists() else {}
        old_run = dict(old_result.get("run", {}))
        generation_cache = read_json(args.generation_cache)
        model = str(generation_cache.get("run", {}).get("model", "unknown"))
        result = {
            "run": {
                **old_run,
                "ragas_version": ragas.__version__,
                "evaluator_model": model,
                "dataset_version": baseline.get("run", {}).get("dataset_version", "unknown"),
                "retrieval_input": str(args.input),
                "answer_prompt": "evidence_only_legal_answer_v1",
                "reference_answer_exposed_to_generator": False,
                "answer_relevancy_strictness": 3,
                "factual_correctness_mode": "f1",
                "evaluator_max_tokens": 4096,
            },
            "retrieval_summary": baseline.get("summary", {}),
            "summary": summarize(ordered),
            "cases": ordered,
        }
        write_json(args.output, result)
        report_path = args.output.with_suffix(".md")
        write_markdown_report(result, report_path)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        print(f"已从缓存重建报告: {report_path}", flush=True)
        return

    key, model, base_url = provider_settings()
    client = AsyncOpenAI(api_key=key, base_url=base_url)
    generations = await build_generations(
        cases, args.generation_cache, client, model, args.concurrency
    )
    # Faithfulness 和 FactualCorrectness 会先把较长法律回答拆成原子陈述。
    # RAGAS 默认上限可能截断结构化 JSON，因此明确提供足够的输出空间。
    evaluator_llm = llm_factory(
        model,
        client=client,
        temperature=0,
        max_tokens=4096,
    )
    embeddings = ProjectBgeEmbedding()
    scores = await build_scores(
        cases,
        generations,
        args.score_cache,
        evaluator_llm,
        embeddings,
        args.concurrency,
    )
    ordered = [scores[case["case_id"]] for case in cases]
    result = {
        "run": {
            "ragas_version": ragas.__version__,
            "evaluator_model": model,
            "dataset_version": baseline.get("run", {}).get("dataset_version", "unknown"),
            "retrieval_input": str(args.input),
            "answer_prompt": "evidence_only_legal_answer_v1",
            "reference_answer_exposed_to_generator": False,
            "answer_relevancy_strictness": 3,
            "factual_correctness_mode": "f1",
            "evaluator_max_tokens": 4096,
        },
        "retrieval_summary": baseline.get("summary", {}),
        "summary": summarize(ordered),
        "cases": ordered,
    }
    write_json(args.output, result)
    report_path = args.output.with_suffix(".md")
    write_markdown_report(result, report_path)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"完整结果: {args.output}", flush=True)
    print(f"中文报告: {report_path}", flush=True)


def main() -> None:
    """解析命令行参数并启动异步评测。"""

    parser = argparse.ArgumentParser(description="运行 JurisRAG 最终版本完整 RAGAS 评测")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-cache", type=Path, default=DEFAULT_GENERATION_CACHE)
    parser.add_argument("--score-cache", type=Path, default=DEFAULT_SCORE_CACHE)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="不调用模型，只根据现有 score cache 重建 JSON 和 Markdown 报告",
    )
    args = parser.parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency 必须大于0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit 必须大于0")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
