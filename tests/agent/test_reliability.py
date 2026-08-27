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
    _summarize_llm_usage,
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


def test_llm_usage_supports_langchain_standard_metadata() -> None:
    """标准 usage_metadata 应按节点记录，且记录中不出现 prompt。"""

    class UsageModel:
        _llm_type = "mock-provider"
        model_name = "mock-model"

        async def ainvoke(self, _prompt: str):
            return SimpleNamespace(
                content='{"answer":"ok"}',
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            )

    async def scenario() -> None:
        events: list[dict] = []
        await invoke_schema(
            UsageModel(),
            AnswerDraft,
            "secret prompt must not be stored",
            node="generate_answer",
            usage_events=events,
        )
        assert events[0]["node"] == "generate_answer"
        assert events[0]["provider"] == "mock-provider"
        assert events[0]["model"] == "mock-model"
        assert events[0]["total_tokens"] == 13
        assert "prompt" not in events[0]
        assert "secret" not in str(events[0])

    anyio.run(scenario)


def test_llm_usage_supports_openai_metadata_and_aggregates() -> None:
    """OpenAI兼容 token_usage 可以归一化，并对多次完整调用求和。"""

    class OpenAIStyleModel:
        model = "compatible-model"

        async def ainvoke(self, _prompt: str):
            return SimpleNamespace(
                content='{"answer":"ok"}',
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                    }
                },
            )

    async def scenario() -> None:
        events: list[dict] = []
        for node in ("generate_answer", "verify_answer"):
            await invoke_schema(
                OpenAIStyleModel(), AnswerDraft, "prompt", node=node, usage_events=events
            )
        summary = _summarize_llm_usage(events)
        assert summary["call_count"] == 2
        assert summary["token_reporting_complete"] is True
        assert summary["input_tokens"] == 14
        assert summary["output_tokens"] == 4
        assert summary["total_tokens"] == 18

    anyio.run(scenario)


def test_llm_usage_keeps_missing_provider_tokens_null() -> None:
    """供应商不报告 Token 时必须为 null，不能用字符数伪造。"""

    class NoUsageModel:
        async def ainvoke(self, _prompt: str):
            return SimpleNamespace(content='{"answer":"ok"}')

    async def scenario() -> None:
        events: list[dict] = []
        await invoke_schema(
            NoUsageModel(), AnswerDraft, "prompt", node="generate_answer", usage_events=events
        )
        summary = _summarize_llm_usage(events)
        assert summary["token_reporting_complete"] is False
        assert summary["input_tokens"] is None
        assert summary["output_tokens"] is None
        assert summary["total_tokens"] is None

    anyio.run(scenario)


def test_failed_llm_call_is_observed_without_error_message() -> None:
    """失败只记录异常类型，不保存可能包含敏感内容的异常消息。"""

    class FailingModel:
        async def ainvoke(self, _prompt: str):
            raise RuntimeError("api_key=do-not-record")

    async def scenario() -> None:
        events: list[dict] = []
        with pytest.raises(RuntimeError):
            await invoke_schema(
                FailingModel(), AnswerDraft, "prompt", node="verify_answer", usage_events=events
            )
        assert events[0]["success"] is False
        assert events[0]["error_type"] == "RuntimeError"
        assert "do-not-record" not in str(events[0])

    anyio.run(scenario)
