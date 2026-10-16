"""逐题审计法律检索 gold，防止评测标签凭空编写。

这个脚本不调用检索器，也不看模型预测结果。它直接重新读取66份已审计
Markdown，确认每个 ``raw_file + article_number`` 只能定位到一个父法条，
并确认人工摘录的 ``evidence_quote`` 确实存在于该法条原文。只有先通过这道
独立质量门，后续 Recall/MRR 才有解释意义。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from offline_ingestion.corpus import load_corpus_entries
from offline_ingestion.chunker import build_markdown_nodes
from shared.config import CHUNK_OVERLAP, CHUNK_SIZE, PROJECT_ROOT


DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "legal_rag_gold_v2.jsonl"


def _compact_whitespace(value: str) -> str:
    """忽略 PDF 换行和排版空格，但保留文字、数字与标点本身。"""

    return re.sub(r"\s+", "", value)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并在缺字段或 case_id 重复时立即失败。"""

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "case_id",
        "question",
        "reference_answer",
        "gold_evidence",
        "question_type",
        "difficulty",
    }
    for line_number, row in enumerate(rows, start=1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"第{line_number}行缺少字段：{sorted(missing)}")
        if not row["question"].strip() or not row["reference_answer"].strip():
            raise ValueError(f"{row['case_id']} 的问题或参考答案为空")
        if not row["gold_evidence"]:
            raise ValueError(f"{row['case_id']} 没有 gold_evidence")
    case_ids = [row["case_id"] for row in rows]
    duplicates = [key for key, count in Counter(case_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"case_id 重复：{duplicates}")
    return rows


def load_verified_parents() -> dict[tuple[str, str], list[dict[str, str]]]:
    """从源 Markdown 重新生成 parent，而不是相信某个既有向量索引。"""

    entries, _ = load_corpus_entries()
    lookup: dict[tuple[str, str], list[dict[str, str]]] = {}
    for entry in entries:
        parents, _ = build_markdown_nodes(
            entry.markdown_path,
            entry.document,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )
        for parent in parents:
            article_number = str(parent.article_number or "")
            if not article_number:
                continue
            key = (entry.document.raw_file, article_number)
            lookup.setdefault(key, []).append(
                {
                    "parent_id": parent.chunk_id,
                    "document_id": entry.document.document_id,
                    "title": entry.document.title,
                    "text": parent.text,
                }
            )
    return lookup


def audit_dataset(path: Path) -> dict[str, Any]:
    """核验全部标签并返回适合保存或打印的审计摘要。"""

    cases = load_dataset(path)
    parents = load_verified_parents()
    evidence_count = 0
    resolved_parent_ids: set[str] = set()
    for case in cases:
        for evidence in case["gold_evidence"]:
            evidence_count += 1
            for field in ("raw_file", "article_number", "evidence_quote"):
                if not str(evidence.get(field, "")).strip():
                    raise ValueError(f"{case['case_id']} 的 gold 缺少 {field}")
            key = (evidence["raw_file"], evidence["article_number"])
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
                    f"{parent['title']} {evidence['article_number']} -> {evidence['evidence_quote']}"
                )
            resolved_parent_ids.add(parent["parent_id"])

    return {
        "dataset": path.name,
        "case_count": len(cases),
        "evidence_count": evidence_count,
        "unique_gold_parent_count": len(resolved_parent_ids),
        "question_type_counts": dict(sorted(Counter(row["question_type"] for row in cases).items())),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in cases).items())),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit source grounding of legal RAG gold cases")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
