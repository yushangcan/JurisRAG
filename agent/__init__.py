"""受约束法律检索 Agent。

Agent 通过 MCP 调用 ``legal_mcp`` 工具，不直接操作 Chroma。这样现有 RAG
仍是独立的证据引擎，Agent 只负责规划、观察、停止和回答验证。
"""

