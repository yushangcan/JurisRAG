from evaluation.audit_agent_rag_comparison import DEFAULT_DATASET, audit_dataset
from evaluation.run_lightweight_eval import retrieval_metrics

import asyncio
import pytest
from evaluation.run_lightweight_eval import evaluate


def test_frozen_comparison_dataset_still_has_grounded_gold():
    report = audit_dataset(DEFAULT_DATASET)
    assert report["status"] == "passed"
    assert report["case_count"] == 30


def test_multisource_metrics_require_all_gold():
    partial = retrieval_metrics(["p1", "noise"], ["p1", "p2"])
    assert partial["parent_recall_at_5"] == 0.5
    assert partial["all_gold_hit_at_5"] == 0
    full = retrieval_metrics(["p1", "noise", "p2"], ["p1", "p2"])
    assert full["all_gold_hit_at_5"] == 1
    assert retrieval_metrics([], [])["parent_recall_at_5"] is None


def test_model_evaluation_cannot_silently_run_without_model(monkeypatch):
    monkeypatch.setattr("evaluation.run_lightweight_eval.build_chat_model", lambda: None)
    with pytest.raises(ValueError, match="未执行模型评测"):
        asyncio.run(evaluate(DEFAULT_DATASET, with_llm=True))
