"""跨 DeepSeek/DashScope 的结构化模型调用辅助函数。"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def parse_json_object(content: str) -> dict[str, Any]:
    """兼容纯 JSON 和 Markdown JSON 围栏，但拒绝非对象输出。"""

    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # 部分兼容端点会在 JSON 前后附一句说明。只截取最外层对象后仍由
        # Pydantic 严格验证字段，不能因为容错而接受任意自由文本。
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return parsed


async def invoke_schema(
    model: Any,
    schema: type[SchemaT],
    prompt: str,
    *,
    timeout_seconds: int | None = None,
) -> SchemaT:
    """调用聊天模型并用 Pydantic 验证，同时限制单次调用时间。

    总图超时只能限制整条链路。如果某一次模型调用长期挂起，后续检索、降级和
    拒答都没有机会执行。因此这里再增加单次调用超时；超时异常由上层节点捕获，
    节点会使用确定性策略或证据原文降级，不会无限等待供应商接口。
    """

    invocation = model.ainvoke(prompt)
    response = (
        await asyncio.wait_for(invocation, timeout=timeout_seconds)
        if timeout_seconds is not None
        else await invocation
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    return schema.model_validate(parse_json_object(content))
