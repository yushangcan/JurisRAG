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
    pass


class CompatibleChatModel:
    def __init__(self, settings: dict, timeout: float = 20) -> None:
        self._llm_type = settings["provider"]
        self.model_name = settings["model"]
        self._api_key = settings["api_key"]
        self._url = str(settings["base_url"]).rstrip("/") + "/chat/completions"
        self._timeout = timeout
        if urlparse(self._url).scheme not in {"http", "https"}:
            raise ValueError("模型端点必须使用 http/https")

    def invoke(self, prompt: str):
        payload = json.dumps({"model": self.model_name, "temperature": 0, "max_tokens": 1800,
                              "response_format": {"type": "json_object"},
                              "messages": [{"role": "user", "content": prompt}]}, ensure_ascii=False).encode("utf-8")
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
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ModelUnavailable("模型未返回可用文本")
            return SimpleNamespace(content=content, response_metadata={"usage": data.get("usage", {})})
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
