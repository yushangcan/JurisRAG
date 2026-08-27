"""JurisRAG 的 MCP 工具层。

这个包只负责把已有法律检索能力暴露为标准 MCP 工具。检索算法仍然放在
``online_retrieval`` 中，MCP 客户端、LangGraph 或其他 Agent 框架都不需要
了解 Chroma、Embedding 和 BM25 的具体实现。
"""

