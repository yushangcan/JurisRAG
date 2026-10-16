"""可选 MCP 只读出口：安装 requirements-mcp.txt 后独立运行。"""

from __future__ import annotations

from legal_mcp import service


def create_server(host: str = "127.0.0.1", port: int = 8001):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    from shared.config import RETRIEVAL_MODE

    server = FastMCP("JurisRAG", host=host, port=port)
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    functions = [service.exact_legal_search, service.keyword_search, service.multi_query_search,
                 service.fetch_parent_context, service.lookup_document_scope, service.check_issue_coverage]
    if RETRIEVAL_MODE == "hybrid":
        functions.append(service.semantic_search)
    for function in functions:
        server.tool(annotations=annotations, structured_output=True)(function)
    return server
