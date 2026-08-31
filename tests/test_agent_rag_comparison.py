"""对比评测器自身的确定性单元测试，不发起模型网络请求。"""

from __future__ import annotations

from evaluation.audit_agent_rag_comparison import DEFAULT_DATASET, audit_dataset
from evaluation.run_agent_rag_ragas_comparison import (
    duplicate_action_count,
    retrieval_metrics,
)


def test_comparison_dataset_is_source_grounded() -> None:
    result = audit_dataset(DEFAULT_DATASET)
    assert result["status"] == "passed"
    assert result["case_count"] == 30
    assert result["answerable_count"] == 21
    assert result["boundary_count"] == 9
    assert result["category_counts"]["multi_file_fusion"] == 7


def test_retrieval_metrics_require_all_gold_for_fusion() -> None:
    result = retrieval_metrics(["p1", "noise", "p2"], ["p1", "p2"])
    assert result["parent_recall_at_1"] == 0.5
    assert result["parent_recall_at_5"] == 1.0
    assert result["all_gold_hit_at_5"] == 1.0
    assert result["mrr_at_5"] == 1.0


def test_duplicate_action_count_only_counts_successful_repeats() -> None:
    actions = [
        {"tool_name": "keyword_search", "arguments": {"keywords": "质疑"}, "status": "success"},
        {"tool_name": "keyword_search", "arguments": {"keywords": "质疑"}, "status": "success"},
        {"tool_name": "keyword_search", "arguments": {"keywords": "质疑"}, "status": "error"},
    ]
    assert duplicate_action_count(actions) == 1
