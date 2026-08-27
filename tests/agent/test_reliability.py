"""Agent 高可用边界：并发舱壁、MCP响应、熔断、模型超时和出站契约。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anyio
import pytest

from agent.config import get_agent_settings
from agent.mcp_client import (
    MCPCircuitOpenError,
    MCPResponseError,
    StreamableHTTPMCPClient,
    _structured_result,
)
from agent.modeling import invoke_schema
from agent.schemas import AnswerDraft
from agent.service import (
    AgentBusyError,
    AgentCapacityGuard,
    AgentResponseContractError,
    _agent_response,
)
from agent.state import initial_agent_state


def _mcp_result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(isError=False, structuredContent=payload, content=[])


def test_mcp_structured_result_rejects_count_mismatch() -> None:
    """JSON对象仍须满足工具名、count和results的一致性。"""

    with pytest.raises(MCPResponseError, match="count"):
        _structured_result(
            "keyword_search",
            _mcp_result(
                {
                    "tool": "keyword_search",
                    "count": 2,
                    "results": [
                        {"chunk_id": "c1", "parent_id": "p1", "text": "法规原文"}
                    ],
                }
            ),
        )


def test_mcp_structured_result_rejects_untraceable_evidence() -> None:
    """缺少ID或正文的候选不能进入正式证据状态。"""

    with pytest.raises(MCPResponseError, match="缺少 ID 或正文"):
        _structured_result(
            "semantic_search",
            _mcp_result(
                {
                    "tool": "semantic_search",
                    "count": 1,
                    "results": [{"chunk_id": "", "text": ""}],
                }
            ),
        )


def test_mcp_circuit_opens_after_consecutive_connection_failures() -> None:
    """达到阈值后新调用快速失败，不再逐个等待网络超时。"""

    client = StreamableHTTPMCPClient(
        "http://127.0.0.1:65530/mcp",
        circuit_failure_threshold=2,
        circuit_recovery_seconds=30,
    )
    client._record_connection_failure()
    assert client.circuit_status()["state"] == "closed"
    client._record_connection_failure()
    assert client.circuit_status()["state"] == "open"
    with pytest.raises(MCPCircuitOpenError, match="熔断器开启"):
        client._before_call()


def test_capacity_guard_fails_fast_and_releases_slot() -> None:
    """并发已满时返回可重试错误，退出后必须释放容量。"""

    guard = AgentCapacityGuard()
    with guard.slot(1):
        assert guard.snapshot(1)["available_slots"] == 0
        with pytest.raises(AgentBusyError, match="并发"):
            with guard.slot(1):
                pass
    assert guard.snapshot(1)["available_slots"] == 1


def test_final_response_contract_fails_closed_without_citation() -> None:
    """声称可回答却没有引用时，不能直接把矛盾响应发给用户。"""

    settings = get_agent_settings()
    state = initial_agent_state("测试问题", 3, settings)
    state.update(
        {
            "final_answer": "这是一个没有引用的结论。",
            "answerability": "conditional",
            "insufficient_evidence": False,
            "citations": [],
        }
    )
    with pytest.raises(AgentResponseContractError, match="至少需要一个"):
        _agent_response(state, settings)


def test_single_llm_call_has_independent_timeout() -> None:
    """模型长期挂起时应抛出超时，让节点有机会执行确定性降级。"""

    class SlowModel:
        async def ainvoke(self, _prompt: str):
            await asyncio.sleep(0.05)
            return SimpleNamespace(content='{"answer":"ok"}')

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await invoke_schema(
                SlowModel(),
                AnswerDraft,
                "prompt",
                timeout_seconds=0.001,
            )

    anyio.run(scenario)
