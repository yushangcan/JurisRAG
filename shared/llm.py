"""用标准库访问 OpenAI 兼容端点，不在主服务安装 LangChain 或供应商 SDK。"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from shared.config import get_llm_settings


class ModelUnavailable(RuntimeError):
    def __init__(self, message: str, *, usage: dict | None = None) -> None:
        super().__init__(message)
        self.usage_metadata = usage


class CompatibleChatModel:
    def __init__(self, settings: dict, timeout: float = 20) -> None:
        self._llm_type = settings["provider"]
        self.model_name = settings["model"]
        self._api_key = settings["api_key"]
        self._wire_api = settings.get("wire_api", "chat")
        if self._wire_api not in {"responses", "chat"}:
            raise ValueError("模型协议只能是 responses 或 chat")
        self._reasoning_effort = settings.get("reasoning_effort", "")
        self._disable_response_storage = settings.get("disable_response_storage", True)
        self._url = str(settings["base_url"]).rstrip("/") + ("/responses" if self._wire_api == "responses" else "/chat/completions")
        self._timeout = timeout
        if urlparse(self._url).scheme not in {"http", "https"}:
            raise ValueError("模型端点必须使用 http/https")

    def invoke(self, prompt: str):
        if self._wire_api == "responses":
            body = {"model": self.model_name, "input": [{"role": "user", "content": prompt}],
                    "store": not self._disable_response_storage, "stream": False, "max_output_tokens": 8192,
                    "text": {"format": {"type": "json_object"}}}
            if self._reasoning_effort:
                body["reasoning"] = {"effort": self._reasoning_effort}
        else:
            body = {"model": self.model_name, "temperature": 0, "max_tokens": 1800,
                    "response_format": {"type": "json_object"}, "store": not self._disable_response_storage,
                    "messages": [{"role": "user", "content": prompt}]}
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(self._url, data=payload, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self._api_key,
            "User-Agent": os.getenv("LLM_HTTP_USER_AGENT", "JurisRAG/1.0"),
        })
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ModelUnavailable("模型响应超过允许大小")
            data = json.loads(raw)
            usage = data.get("usage", {})
            if self._wire_api == "responses":
                if data.get("status") != "completed":
                    raise ModelUnavailable("模型响应未完成或执行失败", usage=usage)
                content = "\n".join(part["text"] for item in data.get("output", []) if item.get("type") == "message"
                                    and item.get("role") == "assistant" for part in item.get("content", [])
                                    if part.get("type") == "output_text" and isinstance(part.get("text"), str))
            else:
                content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ModelUnavailable("模型未返回可用文本", usage=usage)
            return SimpleNamespace(content=content, response_metadata={"usage": usage})
        except HTTPError as exc:
            raise ModelUnavailable(f"模型接口返回 HTTP {exc.code}") from None
        except (URLError, TimeoutError, KeyError, IndexError, ValueError):
            raise ModelUnavailable("模型接口不可用或响应格式错误") from None

    async def ainvoke(self, prompt: str):
        return await asyncio.to_thread(self.invoke, prompt)


def build_chat_model(*, timeout: float = 20) -> CompatibleChatModel | None:
    settings = get_llm_settings()
    if not settings["enabled"] or not settings["api_key"]:
        return None
    if not settings["model"]:
        raise ValueError("启用模型时必须指定模型名")
    return CompatibleChatModel(settings, timeout)
