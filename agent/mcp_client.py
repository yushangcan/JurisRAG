"""官方 MCP Client 封装和错误分类。"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from typing import Any, Protocol

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.shared.exceptions import McpError


REQUIRED_TOOLS = {
    "exact_legal_search",
    "keyword_search",
    "semantic_search",
    "hyde_search",
    "multi_query_search",
    "fetch_parent_context",
    "lookup_document_scope",
    "get_source_metadata",
    "check_issue_coverage",
}


class MCPClientError(RuntimeError):
    pass


class MCPConnectionError(MCPClientError):
    pass


class MCPToolNotFoundError(MCPClientError):
    pass


class MCPToolExecutionError(MCPClientError):
    pass


class MCPResponseError(MCPClientError):
    pass


class MCPTimeoutError(MCPConnectionError):
    pass


class MCPCircuitOpenError(MCPConnectionError):
    """连续连接失败后快速失败，避免每个请求都等待完整网络超时。"""


class LegalToolClient(Protocol):
    async def list_tools(self) -> list[str]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


RESULT_LIST_TOOLS = REQUIRED_TOOLS - {"check_issue_coverage"}
EVIDENCE_RESULT_TOOLS = {
    "exact_legal_search",
    "keyword_search",
    "semantic_search",
    "hyde_search",
    "multi_query_search",
    "fetch_parent_context",
}
MAX_STRUCTURED_RESULTS = 20


def _validate_structured_payload(name: str, structured: dict[str, Any]) -> None:
    """验证 MCP 返回的最小契约，防止损坏或无界结果进入 Agent 状态。

    MCP 的 ``structuredContent`` 只是说明返回了 JSON 对象，不代表字段一定可靠。
    这里再验证工具名、结果数量和证据标识。验证失败属于响应契约错误，不进行
    网络重连，也不会把半结构化内容当作正式法律证据。
    """

    if structured.get("tool") != name:
        raise MCPResponseError(
            f"MCP 工具返回名称与调用名称不一致: {structured.get('tool')!r}"
        )
    if name in RESULT_LIST_TOOLS:
        results = structured.get("results")
        count = structured.get("count")
        if not isinstance(results, list) or any(not isinstance(row, dict) for row in results):
            raise MCPResponseError(f"MCP 工具 {name} 的 results 必须是对象数组")
        if len(results) > MAX_STRUCTURED_RESULTS:
            raise MCPResponseError(
                f"MCP 工具 {name} 返回 {len(results)} 项，超过 {MAX_STRUCTURED_RESULTS} 项上限"
            )
        if not isinstance(count, int) or isinstance(count, bool) or count != len(results):
            raise MCPResponseError(f"MCP 工具 {name} 的 count 与 results 数量不一致")
        if name in EVIDENCE_RESULT_TOOLS:
            for row in results:
                evidence_id = str(row.get("parent_id") or row.get("chunk_id") or "").strip()
                if not evidence_id or not str(row.get("text", "")).strip():
                    raise MCPResponseError(f"MCP 工具 {name} 返回了缺少 ID 或正文的证据")

    if name == "check_issue_coverage":
        answerability = structured.get("answerability")
        if answerability not in {"direct", "conditional", "insufficient", "unknown"}:
            raise MCPResponseError("覆盖检查返回了未知 answerability")
        if not isinstance(structured.get("should_continue"), bool):
            raise MCPResponseError("覆盖检查缺少布尔 should_continue")
        evidence_index = structured.get("evidence_index")
        if not isinstance(evidence_index, list) or len(evidence_index) > MAX_STRUCTURED_RESULTS:
            raise MCPResponseError("覆盖检查 evidence_index 不合法或超过上限")
        for row in evidence_index:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("candidate_id"), int)
                or not str(row.get("evidence_id", "")).strip()
            ):
                raise MCPResponseError("覆盖检查 evidence_index 缺少候选编号或证据 ID")


def _structured_result(name: str, result: Any) -> dict[str, Any]:
    if getattr(result, "isError", False):
        messages = [getattr(item, "text", "") for item in getattr(result, "content", [])]
        raise MCPToolExecutionError(f"MCP 工具 {name} 执行失败: {' '.join(messages).strip()}")
    structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, dict):
        raise MCPResponseError(f"MCP 工具 {name} 没有返回 structuredContent")
    _validate_structured_payload(name, structured)
    return structured


class StreamableHTTPMCPClient:
    """无状态 HTTP MCP 会话，加有限重连和进程内连接熔断。

    熔断器只统计连接/超时故障，不统计工具参数错误和响应 Schema 错误。达到阈值
    后，冷却期内新请求会立即降级到固定 RAG，避免 MCP 已宕机时所有请求都等待
    30 秒。冷却期结束后允许一次探测；探测成功即恢复，失败则重新进入冷却。
    """

    def __init__(
        self,
        url: str,
        timeout_seconds: int = 30,
        *,
        retry_attempts: int = 2,
        circuit_failure_threshold: int = 3,
        circuit_recovery_seconds: int = 30,
    ) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_recovery_seconds = circuit_recovery_seconds
        self._circuit_lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_until = 0.0
        self._probe_in_flight = False

    def _before_call(self) -> None:
        now = time.monotonic()
        with self._circuit_lock:
            if self._opened_until <= 0:
                return
            if now < self._opened_until:
                remaining = max(1, round(self._opened_until - now))
                raise MCPCircuitOpenError(f"MCP 熔断器开启，约 {remaining} 秒后允许探测")
            if self._probe_in_flight:
                raise MCPCircuitOpenError("MCP 熔断器正在执行半开探测")
            self._probe_in_flight = True

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures = 0
            self._opened_until = 0.0
            self._probe_in_flight = False

    def _record_connection_failure(self) -> None:
        with self._circuit_lock:
            self._probe_in_flight = False
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.circuit_failure_threshold:
                self._opened_until = time.monotonic() + self.circuit_recovery_seconds

    def circuit_status(self) -> dict[str, Any]:
        """返回不含 URL 密钥等敏感信息的熔断状态，供健康检查展示。"""

        now = time.monotonic()
        with self._circuit_lock:
            remaining = max(0.0, self._opened_until - now)
            state = "open" if remaining > 0 else "half_open" if self._probe_in_flight else "closed"
            return {
                "state": state,
                "consecutive_failures": self._consecutive_failures,
                "recovery_in_seconds": round(remaining, 2),
                "failure_threshold": self.circuit_failure_threshold,
            }

    async def _with_session(self, operation):
        self._before_call()
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                async def run_once():
                    async with streamable_http_client(self.url) as (read, write, _):
                        async with ClientSession(
                            read,
                            write,
                            read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
                        ) as session:
                            await session.initialize()
                            return await operation(session)

                value = await asyncio.wait_for(run_once(), timeout=self.timeout_seconds)
                self._record_success()
                return value
            except asyncio.CancelledError:
                # 上层总超时或客户端断开时必须立即传播取消，不能误当网络故障重试。
                raise
            except McpError as exc:
                self._record_success()
                raise MCPToolExecutionError(f"MCP 协议返回工具错误: {exc}") from exc
            except TimeoutError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
            if attempt + 1 < self.retry_attempts:
                # 很短的线性退避可以避开瞬时断连，又不会明显吃掉 Agent 总预算。
                await asyncio.sleep(0.15 * (attempt + 1))
        self._record_connection_failure()
        if isinstance(last_error, TimeoutError):
            raise MCPTimeoutError(f"MCP 调用超过 {self.timeout_seconds} 秒") from last_error
        raise MCPConnectionError(
            f"无法连接 MCP Server: {type(last_error).__name__ if last_error else 'unknown'}"
        ) from last_error

    async def list_tools(self) -> list[str]:
        response = await self._with_session(lambda session: session.list_tools())
        return [tool.name for tool in response.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in REQUIRED_TOOLS:
            raise MCPToolNotFoundError(f"工具不在 JurisRAG MCP 白名单: {name}")
        result = await self._with_session(
            lambda session: session.call_tool(name, arguments)
        )
        return _structured_result(name, result)

    async def health(self) -> dict[str, Any]:
        names = set(await self.list_tools())
        missing = sorted(REQUIRED_TOOLS - names)
        return {
            "mcp_ready": not missing,
            "mcp_tool_count": len(names),
            "required_tools_missing": missing,
            "mcp_circuit": self.circuit_status(),
        }


class InProcessMCPClient:
    """测试使用的真实协议客户端，不占网络端口。"""

    def __init__(self, server: Any) -> None:
        self.server = server

    async def list_tools(self) -> list[str]:
        async with create_connected_server_and_client_session(self.server) as session:
            response = await session.list_tools()
            return [tool.name for tool in response.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with create_connected_server_and_client_session(self.server) as session:
            result = await session.call_tool(name, arguments)
            return _structured_result(name, result)
