"""Agent 评测数据和纯聚合逻辑测试，不触发真实模型调用。"""

from evaluation.run_agent_eval import DEFAULT_DATASET, load_cases, score_case


def test_agent_smoke_dataset_is_well_formed() -> None:
    cases = load_cases(DEFAULT_DATASET)
    assert len(cases) == 10
    assert len({case["case_id"] for case in cases}) == len(cases)


def test_score_case_checks_tool_budget_and_forbidden_claims() -> None:
    case = load_cases(DEFAULT_DATASET)[0]
    result = {
        "domain": case["expected_domain"],
        "answerability": case["acceptable_answerability"][0],
        "answer": "基于证据回答",
        "citations": [
            {"evidence_id": "parent-1", "source_url": "https://example.gov/law"}
        ],
        "agent_trace": {
            "iterations": 1,
            "tool_call_count": 3,
            "llm_call_count": 2,
            "actions": [
                {
                    "tool_name": case["acceptable_first_tools"][0],
                    "arguments": {"query": case["question"]},
                    "status": "success",
                }
            ],
            "budgets": {"max_iterations": 3, "max_tool_calls": 8},
        },
    }
    scored = score_case(case, result, 100.0)
    assert scored["passed"] is True
