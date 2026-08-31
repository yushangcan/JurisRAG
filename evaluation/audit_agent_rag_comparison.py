"""审计固定 RAG 与 Agent 对比题集，防止 gold 和边界标签凭空编写。

本脚本只读取冻结题集和本地结构化 Markdown，不调用检索器或模型。可回答题的
每条证据必须通过 ``raw_file + article_number`` 唯一定位到父法条，摘录也必须
真实出现在该法条中；边界题则必须没有伪造的 gold 证据，并说明拒答原因。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.audit_gold_dataset import _compact_whitespace, load_verified_parents
from shared.config import PROJECT_ROOT


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "legal_agent_rag_comparison_v1.jsonl"
)
REQUIRED_FIELDS = {
    "case_id",
    "question",
    "reference_answer",
    "gold_evidence",
    "should_answer",
    "category",
    "question_type",
    "difficulty",
    "expected_domain",
    "required_source_count",
}
ALLOWED_CATEGORIES = {"single_file", "multi_file_fusion", "boundary"}
ALLOWED_DIFFICULTIES = {"basic", "medium", "hard"}
ALLOWED_DOMAINS = {"government_procurement", "construction_bidding", "uncertain"}


def load_comparison_dataset(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并尽早暴露重复 ID、空文本或 Schema 错误。"""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第{line_number}行不是合法 JSON") from exc
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"第{line_number}行缺少字段：{sorted(missing)}")
        if not str(row["case_id"]).strip():
            raise ValueError(f"第{line_number}行 case_id 为空")
        if not str(row["question"]).strip() or not str(row["reference_answer"]).strip():
            raise ValueError(f"{row['case_id']} 的问题或参考答案为空")
        if not isinstance(row["should_answer"], bool):
            raise ValueError(f"{row['case_id']} 的 should_answer 必须是布尔值")
        if row["category"] not in ALLOWED_CATEGORIES:
            raise ValueError(f"{row['case_id']} 的 category 不合法")
        if row["difficulty"] not in ALLOWED_DIFFICULTIES:
            raise ValueError(f"{row['case_id']} 的 difficulty 不合法")
        if row["expected_domain"] not in ALLOWED_DOMAINS:
            raise ValueError(f"{row['case_id']} 的 expected_domain 不合法")
        if not isinstance(row["gold_evidence"], list):
            raise ValueError(f"{row['case_id']} 的 gold_evidence 必须是数组")
        if not isinstance(row["required_source_count"], int):
            raise ValueError(f"{row['case_id']} 的 required_source_count 必须是整数")
        rows.append(row)

    if not rows:
        raise ValueError("对比题集不能为空")
    counts = Counter(str(row["case_id"]) for row in rows)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"case_id 重复：{duplicates}")
    return rows


def audit_dataset(path: Path) -> dict[str, Any]:
    """逐题核验来源和边界标签，返回可供 CI 断言的摘要。"""

    cases = load_comparison_dataset(path)
    parents = load_verified_parents()
    evidence_count = 0
    resolved_parent_ids: set[str] = set()

    for case in cases:
        evidence_rows = case["gold_evidence"]
        if case["should_answer"]:
            if case["category"] == "boundary" or not evidence_rows:
                raise ValueError(f"{case['case_id']} 是可回答题但缺少有效 gold")
            unique_sources = {str(row.get("raw_file", "")) for row in evidence_rows}
            if case["required_source_count"] != len(unique_sources):
                raise ValueError(
                    f"{case['case_id']} 的 required_source_count 与 gold 文件数不一致"
                )
            expected_category = (
                "multi_file_fusion" if len(unique_sources) > 1 else "single_file"
            )
            if case["category"] != expected_category:
                raise ValueError(
                    f"{case['case_id']} 的 category 应为 {expected_category}"
                )
        else:
            if case["category"] != "boundary" or evidence_rows:
                raise ValueError(f"{case['case_id']} 的边界题不能携带 gold 法条")
            if case["required_source_count"] != 0:
                raise ValueError(f"{case['case_id']} 的边界题 required_source_count 必须为0")
            if not str(case.get("rationale", "")).strip():
                raise ValueError(f"{case['case_id']} 的边界题缺少 rationale")
            continue

        for evidence in evidence_rows:
            evidence_count += 1
            for field in ("raw_file", "article_number", "evidence_quote"):
                if not str(evidence.get(field, "")).strip():
                    raise ValueError(f"{case['case_id']} 的 gold 缺少 {field}")
            key = (str(evidence["raw_file"]), str(evidence["article_number"]))
            matches = parents.get(key, [])
            if len(matches) != 1:
                raise ValueError(
                    f"{case['case_id']} 的 gold 不能唯一定位：{key}，命中{len(matches)}个"
                )
            parent = matches[0]
            quote = _compact_whitespace(str(evidence["evidence_quote"]))
            if quote not in _compact_whitespace(parent["text"]):
                raise ValueError(
                    f"{case['case_id']} 的 evidence_quote 不在原文中："
                    f"{parent['title']} {evidence['article_number']}"
                )
            resolved_parent_ids.add(parent["parent_id"])

    category_counts = Counter(str(row["category"]) for row in cases)
    should_answer_counts = Counter(bool(row["should_answer"]) for row in cases)
    if category_counts["multi_file_fusion"] < 5:
        raise ValueError("多文件融合题少于5道，不能覆盖融合能力")
    if should_answer_counts[False] < 5:
        raise ValueError("边界题少于5道，不能覆盖拒答能力")

    return {
        "dataset": path.name,
        "case_count": len(cases),
        "answerable_count": should_answer_counts[True],
        "boundary_count": should_answer_counts[False],
        "evidence_count": evidence_count,
        "unique_gold_parent_count": len(resolved_parent_ids),
        "category_counts": dict(sorted(category_counts.items())),
        "question_type_counts": dict(
            sorted(Counter(str(row["question_type"]) for row in cases).items())
        ),
        "difficulty_counts": dict(
            sorted(Counter(str(row["difficulty"]) for row in cases).items())
        ),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计固定 RAG 与 Agent 对比题集")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
