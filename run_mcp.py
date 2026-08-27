"""启动 JurisRAG MCP Server。

示例：

    # 本地 Agent 子进程最常用，不占网络端口
    python run_mcp.py --transport stdio

    # 跨进程或调试客户端使用，地址为 http://127.0.0.1:8001/mcp
    python run_mcp.py --transport streamable-http --port 8001
"""

from __future__ import annotations

import argparse

from legal_mcp.server import create_server


def main() -> None:
    """解析启动参数并交给官方 MCP SDK 管理协议生命周期。"""

    parser = argparse.ArgumentParser(description="JurisRAG 法律检索 MCP Server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio 适合本地 Agent；streamable-http 适合独立服务",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址")
    parser.add_argument("--port", type=int, default=8001, help="HTTP 监听端口")
    args = parser.parse_args()
    server = create_server(host=args.host, port=args.port)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()

