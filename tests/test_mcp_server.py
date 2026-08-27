"""MCP 协议层测试：验证工具可被发现并返回结构化结果。"""

from __future__ import annotations

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from legal_mcp import service
from legal_mcp.server import create_server


EXPECTED_TOOLS = {
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


def test_mcp_lists_all_read_only_legal_tools() -> None:
    """真实建立内存 MCP 会话，而不是只检查 Python 函数是否存在。"""

    async def scenario() -> None:
        async with create_connected_server_and_client_session(create_server()) as session:
            response = await session.list_tools()
            tools = {tool.name: tool for tool in response.tools}
            assert set(tools) == EXPECTED_TOOLS
            assert all(tool.annotations.readOnlyHint is True for tool in tools.values())
            assert all(tool.annotations.destructiveHint is False for tool in tools.values())

    anyio.run(scenario)


def test_mcp_tool_call_returns_structured_content(monkeypatch) -> None:
    """通过 MCP 的 tools/call 调用，确认结果不是一段无法解析的自由文本。"""

    expected = {
        "tool": "exact_legal_search",
        "query": "《政府采购法》 第二十二条",
        "required_matches": ["article_number", "document_title"],
        "count": 0,
        "results": [],
    }
    monkeypatch.setattr(service, "exact_legal_search", lambda **_: expected)

    async def scenario() -> None:
        async with create_connected_server_and_client_session(create_server()) as session:
            result = await session.call_tool(
                "exact_legal_search",
                {"document_title": "政府采购法", "article_number": "第二十二条"},
            )
            assert result.isError is False
            assert result.structuredContent == expected

    anyio.run(scenario)
