"""Agent 工具边界和重复动作测试。"""

import pytest

from agent.config import get_agent_settings
from agent.policy import AgentPolicyError, validate_action
from agent.schemas import ToolAction
from agent.state import initial_agent_state


def test_policy_rejects_duplicate_action() -> None:
    settings = get_agent_settings()
    state = initial_agent_state("供应商条件", 3, settings)
    action = ToolAction(
        tool_name="semantic_search",
        arguments={"query": "供应商条件", "top_k": 3},
    )
    approved, signature = validate_action(action, state, settings)
    state["attempted_action_signatures"] = [signature]
    with pytest.raises(AgentPolicyError, match="禁止重复"):
        validate_action(approved, state, settings)


def test_policy_rejects_oversized_top_k() -> None:
    settings = get_agent_settings()
    state = initial_agent_state("供应商条件", 3, settings)
    action = ToolAction(
        tool_name="semantic_search",
        arguments={"query": "供应商条件", "top_k": 100},
    )
    with pytest.raises(AgentPolicyError, match="参数不合法"):
        validate_action(action, state, settings)


def test_exact_search_requires_a_formal_signal() -> None:
    settings = get_agent_settings()
    state = initial_agent_state("供应商条件", 3, settings)
    action = ToolAction(tool_name="exact_legal_search", arguments={"top_k": 3})
    with pytest.raises(AgentPolicyError, match="至少需要"):
        validate_action(action, state, settings)


def test_tool_action_normalizes_oversized_target_issues() -> None:
    """模型多给争点时应清洗并截断，而不是让整个 API 返回 502。"""

    action = ToolAction(
        tool_name="semantic_search",
        arguments={"query": "页数限制与否决投标", "top_k": 3},
        target_issues=[
            " 页数限制 ",
            "否决依据",
            "比例原则",
            "限制竞争",
            "公平原则",
            "诚实信用",
            "否决依据",  # 重复项不会占用六项预算。
            "救济途径",
        ],
    )

    assert action.target_issues == [
        "页数限制",
        "否决依据",
        "比例原则",
        "限制竞争",
        "公平原则",
        "诚实信用",
    ]
