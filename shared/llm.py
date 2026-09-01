"""Create the configured chat model without exposing provider secrets.

Both query expansion and final answering need the same provider selection.  A
single factory prevents one stage from accidentally using DeepSeek while the
other stage uses DashScope.
"""

from __future__ import annotations

import os
from typing import Any

from shared.config import get_llm_settings


def build_chat_model() -> Any | None:
    """Return the enabled chat model, or ``None`` when no usable key exists."""

    settings = get_llm_settings()
    if not settings["enabled"] or not settings["api_key"]:
        return None

    if settings["provider"] == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=settings["model"],
            api_key=settings["api_key"],
            base_url=settings["base_url"] or None,
            temperature=0,
            # 部分 OpenAI-compatible 网关会拦截 SDK 默认 User-Agent；允许通过
            # 环境变量覆盖，默认使用兼容性更好的普通浏览器标识。
            default_headers={
                "User-Agent": os.getenv("LLM_HTTP_USER_AGENT", "Mozilla/5.0")
            },
        )

    # DashScope exposes an OpenAI-compatible endpoint.  ChatOpenAI is only the
    # protocol client here; the actual model remains the configured Qwen model.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings["model"],
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        temperature=0,
        default_headers={
            "User-Agent": os.getenv("LLM_HTTP_USER_AGENT", "Mozilla/5.0")
        },
    )
