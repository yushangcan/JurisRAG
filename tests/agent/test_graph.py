"""使用可控 MCP 替身验证 LangGraph 路由、循环和停止。"""

from __future__ import annotations

import anyio

from agent.config import get_agent_settings
from agent.graph import LegalAgentWorkflow
from agent.mcp_client import MCPConnectionError
from agent.state import initial_agent_state


CHILD = {
    "rank": 1,
    "chunk_id": "law-article-22-child-0",
    "parent_id": "law-article-22",
    "text": "第二十二条 供应商参加政府采购活动应当具备相应条件；未实质性响应的投标应当否决。",
    "title": "中华人民共和国政府采购法",
    "article_number": "第二十二条",
    "source_url": "https://example.gov/law",
    "retrieval_channels": ["exact_lexical"],
    "metadata": {
        "title": "中华人民共和国政府采购法",
        "article_number": "第二十二条",
        "source_url": "https://example.gov/law",
        "category": "法律",
    },
}

PARENT = {
    "rank": 1,
    "parent_id": "law-article-22",
    "text": "第二十二条 供应商参加政府采购活动应当具备相应条件；未实质性响应的投标应当否决。",
    "title": "中华人民共和国政府采购法",
    "article_number": "第二十二条",
    "source_url": "https://example.gov/law",
    "metadata": CHILD["metadata"],
}


class FakeMCPClient:
    def __init__(self, continue_once: bool = False) -> None:
        self.calls: list[str] = []
        self.coverage_calls = 0
        self.continue_once = continue_once

    async def list_tools(self) -> list[str]:
        return []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append(name)
        if name == "check_issue_coverage":
            self.coverage_calls += 1
            should_continue = self.continue_once and self.coverage_calls == 1
            return {
                "tool": name,
                "planner_used": True,
                "answerability": "conditional" if should_continue else "direct",
                "should_continue": should_continue,
                "legal_issues": ["供应商法定条件"],
                "issue_coverage": [
                    {"issue": "供应商法定条件", "candidate_ids": [1]}
                ],
                "uncovered_issues": [],
                "missing_evidence": ["进一步规则"] if should_continue else [],
                "evidence_index": [
                    {"candidate_id": 1, "evidence_id": CHILD["chunk_id"]}
                ],
            }
        if name == "fetch_parent_context":
            return {"tool": name, "count": 1, "results": [PARENT], "missing_ids": []}
        return {"tool": name, "count": 1, "results": [CHILD]}


def test_graph_completes_exact_query_with_traceable_parent() -> None:
    async def scenario() -> None:
        settings = get_agent_settings()
        client = FakeMCPClient()
        workflow = LegalAgentWorkflow(client, None, settings)
        state = await workflow.graph.ainvoke(
            initial_agent_state("《中华人民共和国政府采购法》第二十二条规定什么？", 3, settings),
            {"recursion_limit": 40},
        )
        assert state["iteration"] == 1
        assert state["answerability"] == "direct"
        assert state["insufficient_evidence"] is False
        assert state["stop_reason"] == "direct_evidence_complete"
        assert client.calls == [
            "exact_legal_search",
            "check_issue_coverage",
            "fetch_parent_context",
        ]
        assert state["citations"][0]["parent_id"] == "law-article-22"

    anyio.run(scenario)


def test_graph_changes_search_strategy_when_coverage_requests_more() -> None:
    async def scenario() -> None:
        settings = get_agent_settings()
        client = FakeMCPClient(continue_once=True)
        workflow = LegalAgentWorkflow(client, None, settings)
        state = await workflow.graph.ainvoke(
            initial_agent_state("招标文件设置与项目无关的条件合法吗？", 3, settings),
            {"recursion_limit": 40},
        )
        search_calls = [name for name in client.calls if name not in {"check_issue_coverage", "fetch_parent_context"}]
        assert len(search_calls) == 2
        assert search_calls[0] != search_calls[1]
        assert state["iteration"] == 2
        assert state["insufficient_evidence"] is False

    anyio.run(scenario)


def test_connection_failure_requests_fixed_rag_fallback() -> None:
    class FailingClient(FakeMCPClient):
        async def call_tool(self, name: str, arguments: dict) -> dict:
            raise MCPConnectionError("offline")

    async def scenario() -> None:
        settings = get_agent_settings()
        workflow = LegalAgentWorkflow(FailingClient(), None, settings)
        state = await workflow.graph.ainvoke(
            initial_agent_state("政府采购法第二十二条", 3, settings),
            {"recursion_limit": 40},
        )
        assert state["use_baseline_fallback"] is True
        assert state["stop_reason"] == "mcp_unavailable"

    anyio.run(scenario)


def test_safe_evidence_fallback_keeps_citations_after_answer_review_failure() -> None:
    """回答模型失败不应把已经通过覆盖检查的真实法规证据一起清空。"""

    settings = get_agent_settings()
    workflow = LegalAgentWorkflow(FakeMCPClient(), None, settings)
    state = initial_agent_state("某项要求是否合法？", 3, settings)
    state.update(
        {
            "parent_evidence": {PARENT["parent_id"]: PARENT},
            "citations": [
                {
                    "parent_id": PARENT["parent_id"],
                    "document_title": PARENT["title"],
                }
            ],
            "answerability": "conditional",
            "missing_user_facts": ["项目类型"],
            "validation_feedback": ["结论表述过强"],
            "stop_reason": "conditional_evidence_complete",
        }
    )
    update = workflow.safe_evidence_fallback(state)
    assert update["insufficient_evidence"] is False
    assert update["generation_mode"] == "agent_evidence_fallback"
    assert "可核验法规原文" in update["final_answer"]
    assert "项目类型" in update["final_answer"]
