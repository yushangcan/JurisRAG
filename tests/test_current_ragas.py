"""评测完整性：缺失不补零、配对分母一致、断点不压缩三次采样。"""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from evaluation.run_current_ragas import summarize, usage_summary, valid_score
from evaluation.run_lightweight_eval import retrieval_metrics


def test_missing_score_and_unknown_usage_do_not_become_zero():
    assert valid_score({"status": "failed", "value": 0}) is None
    assert valid_score({"status": "ok", "value": float("nan")}) is None
    assert valid_score({"status": "ok", "value": 0}) == 0
    usage = usage_summary([{"success": True, "total_tokens": 10}, {"success": False}])
    assert usage["total_tokens"] is None
    assert usage["known_total_tokens"] == 10
    assert usage["total_tokens_reported_calls"] == 1


def test_partial_scores_use_full_denominator_and_common_pairs():
    def record(case_id, mode, value, status="ok"):
        return {"case_id": case_id, "mode": mode, "should_answer": True, "generation_status": status,
                "invalid_citation_count": 0, "latency_ms": 1,
                "result": {"insufficient_evidence": False, "warning": None, "generation_mode": "test",
                           "agent_trace": {"controller": "test", "llm_usage": {"events": []}}},
                "retrieval": retrieval_metrics(["p1"], ["p1"]),
                "scores": {"context_recall": {"value": value, "status": "ok" if value is not None else "failed", "calls": {}}}}

    records = {"one:rag": record("one", "rag", 1), "one:agent": record("one", "agent", 0.5),
               "two:rag": record("two", "rag", 0), "two:agent": record("two", "agent", None, "failed")}
    result = summarize({"records": records, "audit": {"answerable_count": 2}})
    assert result["agent"]["scores"]["context_recall"]["eligible_n"] == 2
    assert result["agent"]["scores"]["context_recall"]["valid_n"] == 1
    assert result["rag"]["scores"]["context_recall"]["mean"] == 0.5
    assert result["paired"]["context_recall"] == {"n": 1, "rag": 1.0, "agent": 0.5, "agent_minus_rag": -0.5}


def test_responses_judge_resumes_subcalls_without_collapsing_repeated_prompts():
    pytest.importorskip("ragas")
    from evaluation.ragas_adapters import ResponsesJudge

    class Output(BaseModel):
        answer: int

    class Model:
        calls = 0

        async def ainvoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(content=f'{{"answer":{self.calls}}}', response_metadata={})

    async def run():
        model, checkpoint = Model(), {}
        judge = ResponsesJudge(model, checkpoint, lambda: None)
        first = await judge.agenerate("same prompt", Output)
        second = await judge.agenerate("same prompt", Output)
        assert (first.answer, second.answer, model.calls) == (1, 2, 2)
        resumed = ResponsesJudge(model, checkpoint, lambda: None)
        assert (await resumed.agenerate("same prompt", Output)).answer == 1
        assert (await resumed.agenerate("same prompt", Output)).answer == 2
        assert model.calls == 2
        changed = ResponsesJudge(model, checkpoint, lambda: None)
        with pytest.raises(ValueError, match="输入改变"):
            await changed.agenerate("changed prompt", Output)

    asyncio.run(run())
