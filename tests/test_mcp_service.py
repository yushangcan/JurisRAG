"""法律 MCP 业务层的最小真实数据库测试。"""

from __future__ import annotations

import pytest

from legal_mcp.service import exact_legal_search, fetch_parent_context, keyword_search


def test_exact_search_does_not_mix_law_with_implementation_regulation() -> None:
    """《政府采购法》第二十二条不能混入《实施条例》第二十二条。"""

    result = exact_legal_search("中华人民共和国政府采购法", "第二十二条", top_k=3)
    assert result["count"] >= 1
    assert all(item["title"] == "中华人民共和国政府采购法" for item in result["results"])
    assert all(item["article_number"] == "第二十二条" for item in result["results"])


def test_parent_context_keeps_traceable_metadata() -> None:
    """从 child 命中到 parent 全文后仍能看到法规名、条号和来源链接。"""

    child = exact_legal_search("中华人民共和国政府采购法", "第二十二条", top_k=1)
    parent_id = child["results"][0]["parent_id"]
    parent = fetch_parent_context([parent_id])
    assert parent["missing_ids"] == []
    assert parent["results"][0]["article_number"] == "第二十二条"
    assert parent["results"][0]["source_url"].startswith("http")
    assert "供应商参加政府采购活动" in parent["results"][0]["text"]


def test_tool_boundaries_reject_unbounded_or_empty_calls() -> None:
    """工具边界能阻止 Agent 误传空查询或一次拉取过多证据。"""

    with pytest.raises(ValueError, match="不能为空"):
        keyword_search("   ")
    with pytest.raises(ValueError, match="1 到 20"):
        keyword_search("供应商", top_k=100)
