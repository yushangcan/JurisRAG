"""使用官方 Python SDK 注册 JurisRAG MCP 工具。

MCP（Model Context Protocol）解决的是“Agent 如何发现并调用工具”。本文件只做
协议适配：工具的实际业务逻辑全部在 ``legal_mcp.service``，便于测试和复用。
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from legal_mcp import service


# 所有工具都只读取本地知识库：只读、幂等、不访问未声明的外部业务系统。
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def create_server(host: str = "127.0.0.1", port: int = 8001) -> FastMCP:
    """创建一个可用于 stdio 或 Streamable HTTP 的 MCP Server。"""

    mcp = FastMCP(
        name="JurisRAG Legal Retrieval",
        instructions=(
            "只在政府采购与招标投标法律知识库中检索。先用精确/关键词/语义工具"
            "找到 child，再按 parent_id 获取完整法条；法律结论必须引用返回证据。"
        ),
        host=host,
        port=port,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
    )

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def exact_legal_search(
        document_title: str = "",
        article_number: str = "",
        document_number: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """按法规名称、条号、文号精确查法条；填写的多个字段必须同时命中。"""

        return service.exact_legal_search(
            document_title=document_title,
            article_number=article_number,
            document_number=document_number,
            top_k=top_k,
        )

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def keyword_search(keywords: str, top_k: int = 5) -> dict[str, Any]:
        """用 BM25 检索准确术语、数字、否决用语和专有名词。"""

        return service.keyword_search(keywords, top_k)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def semantic_search(query: str, top_k: int = 5) -> dict[str, Any]:
        """用中文 BGE Embedding 检索自然语言问题和同义表达。"""

        return service.semantic_search(query, top_k)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def hyde_search(question: str, top_k: int = 5) -> dict[str, Any]:
        """生成仅用于检索的中性假设答案，向量召回后按原问题重排。"""

        return service.hyde_search(question, top_k)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def multi_query_search(queries: list[str], top_k: int = 5) -> dict[str, Any]:
        """对最多五个法律争点查询执行 BM25+向量检索、RRF 融合和重排。"""

        return service.multi_query_search(queries, top_k)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def fetch_parent_context(parent_ids: list[str]) -> dict[str, Any]:
        """根据检索结果中的 parent_id 取回完整法条或章节上下文。"""

        return service.fetch_parent_context(parent_ids)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def lookup_document_scope(title_or_number: str, top_k: int = 5) -> dict[str, Any]:
        """查询法规类别、效力状态、生效日期和官方来源等适用范围辅助信息。"""

        return service.lookup_document_scope(title_or_number, top_k)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def get_source_metadata(chunk_ids: list[str]) -> dict[str, Any]:
        """查询证据的法规名、条号、页码、文件路径和官方来源链接。"""

        return service.get_source_metadata(chunk_ids)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def check_issue_coverage(question: str, evidence_ids: list[str]) -> dict[str, Any]:
        """逐项检查证据对法律争点的覆盖，并建议继续检索或停止。"""

        return service.check_issue_coverage(question, evidence_ids)

    return mcp


# 这个对象方便测试器或支持直接导入 MCP 对象的客户端使用。
mcp = create_server()
