"""RAGAS 0.4 的 Responses / 本地中文向量适配；不修改业务链路。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path

from pydantic import BaseModel
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

from agent.modeling import invoke_schema


class ResponsesJudge(InstructorBaseRagasLLM):
    """每个 metric 单独编号子调用，保留 strictness=3 的三次独立采样。"""

    def __init__(self, model, checkpoint: dict, save, *, timeout: int = 120, attempts: int = 2):
        self.model, self.checkpoint, self.save = model, checkpoint, save
        self.timeout, self.attempts, self.ordinal = timeout, attempts, 0

    def generate(self, prompt: str, response_model: type[BaseModel]):
        return asyncio.run(self.agenerate(prompt, response_model))

    async def agenerate(self, prompt: str, response_model: type[BaseModel]):
        schema = response_model.model_json_schema()
        request = (prompt + "\nFor Chinese input, write generated questions and statements in Chinese. "
                   "Return only one JSON object matching this JSON Schema; do not include Markdown:\n"
                   + json.dumps(schema, ensure_ascii=False))
        signature = hashlib.sha256(request.encode("utf-8")).hexdigest()
        key, self.ordinal = str(self.ordinal), self.ordinal + 1
        call = self.checkpoint.setdefault("calls", {}).setdefault(key, {"signature": signature, "attempts": []})
        if call["signature"] != signature:
            raise ValueError("评分子调用输入改变，拒绝复用缓存")
        if "output" in call:
            return response_model.model_validate(call["output"])
        for attempt in range(self.attempts):
            events = []
            try:
                output = await invoke_schema(self.model, response_model, request,
                                             timeout_seconds=self.timeout + 5,
                                             node=response_model.__name__, usage_events=events)
            except Exception as exc:
                # ModelUnavailable 的消息由共享客户端生成，不包含供应商正文或密钥。
                message = str(exc) if type(exc).__name__ == "ModelUnavailable" else type(exc).__name__
                call["attempts"].append({"events": events, "error_type": type(exc).__name__, "error": message})
                self.save()
                if attempt + 1 == self.attempts:
                    raise
                await asyncio.sleep(2)
            else:
                call["attempts"].append({"events": events, "error_type": None})
                call["output"] = output.model_dump(mode="json")
                self.save()
                return output


class ChineseQuestionEmbeddings(BaseRagasEmbedding):
    """仅为 AnswerRelevancy 编码问题；绝不退回字符 hash。"""

    def __init__(self, path: Path):
        super().__init__()
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5",
                                   specific_model_path=str(path.resolve()),
                                   local_files_only=True, threads=2)
        self.lock = threading.Lock()
        self.vectors: dict[str, list[float]] = {}

    def embed_text(self, text: str, **kwargs) -> list[float]:
        with self.lock:
            if text not in self.vectors:
                vector = next(iter(self.model.query_embed([text]))).tolist()
                if len(vector) != 512 or not any(vector):
                    raise ValueError("中文语义向量无效")
                self.vectors[text] = vector
            return self.vectors[text]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)
