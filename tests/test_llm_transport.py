"""使用本地模拟端点验证兼容模型 HTTP 链路，不调用真实供应商。"""

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent.service import run_agent
from shared.llm import CompatibleChatModel


@contextmanager
def endpoint(*, fail=False):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers["Authorization"], payload))
            if fail:
                self.send_error(503, "private-provider-diagnostic")
                return
            state = json.loads(payload["messages"][0]["content"].split("\n状态：", 1)[1])
            if not state["candidates"]:
                action = {"type": "search", "queries": ["单一来源采购"],
                          "references": [{"document_title": "政府采购法", "article_number": "第三十一条"}]}
            elif not state["evidence"]:
                row = next(row for row in state["candidates"] if row["article"] == "第三十一条")
                action = {"type": "read_evidence", "evidence_ids": [row["evidence_id"]]}
            else:
                claims = [{"text": row["text"], "quote": row["text"], "evidence_id": row["evidence_id"],
                           "issues": state["required_issues"]} for row in state["evidence"]]
                action = {"type": "answer", "claims": claims}
            body = json.dumps({"choices": [{"message": {"content": json.dumps({"action": action})}}],
                               "usage": {"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    model = CompatibleChatModel({"provider": "local-test", "model": "local-test", "api_key": "test-key",
                                 "base_url": f"http://127.0.0.1:{server.server_port}/v1"}, timeout=2)
    try:
        yield model, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_http_generation_keeps_grounding_and_supplier_usage():
    with endpoint() as (model, received):
        result = asyncio.run(run_agent("哪些情形可以采用单一来源采购？", 1, model=model))
    assert result["generation_mode"] == "agent_llm_loop"
    assert result["agent_trace"]["llm_usage"]["total_tokens"] == 480
    assert len(received) == 3
    assert received[0][0] == "/v1/chat/completions"
    assert received[0][1] == "Bearer test-key"
    assert all(request[2]["response_format"] == {"type": "json_object"} for request in received)
    assert result["citations"][0]["quote"] in result["answer"]


def test_provider_http_error_falls_back_without_private_error_body():
    with endpoint(fail=True) as (model, _):
        result = asyncio.run(run_agent("哪些情形可以采用单一来源采购？", 1, model=model))
    assert result["generation_mode"] == "agent_evidence_fallback"
    assert result["status"] == "answered"
    assert "private-provider-diagnostic" not in json.dumps(result)
    assert "test-key" not in json.dumps(result)
