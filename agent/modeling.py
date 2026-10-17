"""跨 DeepSeek/DashScope 的结构化模型调用辅助函数。"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, TypeVar

from pydantic import BaseModel


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMCallUsage(BaseModel):
    """一次模型调用的安全观测记录。

    这里只保存排障和性能分析需要的元数据，不保存 prompt、模型回复或 API Key。
    Token 数完全以供应商响应为准；供应商没有返回时保留 ``None``，避免把估算值
    当成真实账单用量。
    """

    node: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: float
    success: bool
    error_type: str | None = None


def _safe_model_identity(model: Any) -> tuple[str, str]:
    """只读取供应商和模型名，不序列化可能含密钥的整个模型对象。"""

    provider = str(getattr(model, "_llm_type", "") or "").strip()
    if not provider:
        provider = type(model).__module__.split(".")[0] or "unknown"
    model_name = str(
        getattr(model, "model_name", "") or getattr(model, "model", "") or "unknown"
    ).strip()
    return provider[:80], model_name[:160]


def _non_negative_int(value: Any) -> int | None:
    """把供应商返回的整数安全归一化；布尔值和负数都视为无效。"""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _response_tokens(response: Any) -> tuple[int | None, int | None, int | None]:
    """兼容 LangChain 标准字段和 OpenAI 兼容端点的 token_usage。"""

    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        metadata = getattr(response, "response_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    return (
        _non_negative_int(usage.get("input_tokens", usage.get("prompt_tokens"))),
        _non_negative_int(usage.get("output_tokens", usage.get("completion_tokens"))),
        _non_negative_int(usage.get("total_tokens")),
    )


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
    node: str = "unspecified",
    usage_events: list[dict[str, Any]] | None = None,
) -> SchemaT:
    """调用聊天模型并用 Pydantic 验证，同时限制单次调用时间。

    总请求超时只能限制整条链路。如果某一次模型调用长期挂起，后续检索、降级和
    拒答都没有机会执行。因此这里再增加单次调用超时；超时异常由上层节点捕获，
    节点会使用确定性策略或证据原文降级，不会无限等待供应商接口。
    """

    provider, model_name = _safe_model_identity(model)
    started = time.monotonic()
    input_tokens = output_tokens = total_tokens = None
    try:
        invocation = model.ainvoke(prompt)
        response = (
            await asyncio.wait_for(invocation, timeout=timeout_seconds)
            if timeout_seconds is not None
            else await invocation
        )
        input_tokens, output_tokens, total_tokens = _response_tokens(response)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = schema.model_validate(parse_json_object(content))
    except BaseException as exc:
        # CancelledError 也必须留下失败记录后继续向上抛出，不能吞掉任务取消信号。
        if usage_events is not None:
            usage_events.append(
                LLMCallUsage(
                    node=node,
                    provider=provider,
                    model=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    duration_ms=round((time.monotonic() - started) * 1000, 2),
                    success=False,
                    error_type=type(exc).__name__,
                ).model_dump()
            )
        raise

    if usage_events is not None:
        usage_events.append(
            LLMCallUsage(
                node=node,
                provider=provider,
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                duration_ms=round((time.monotonic() - started) * 1000, 2),
                success=True,
            ).model_dump()
        )
    return parsed
