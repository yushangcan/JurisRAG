"""可切换的中文语义Embedding与可复现实验用hash基线。

默认使用ONNX版 ``BAAI/bge-small-zh-v1.5``。旧字符hash实现继续保留，
不是作为生产方案，而是为了同一语料、同一检索器下做可复现实验对照。
文档和查询分别调用 ``passage_embed``/``query_embed``，避免未来换成需要
query前缀的模型时把两种角色混在一起。
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Iterable

from shared.config import (
    EMBEDDING_BACKEND,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
)


def _hash_embed(text: str) -> list[float]:
    """把文字转成确定性字符hash向量，仅用于旧基线与单元测试。"""

    vector = [0.0] * EMBEDDING_DIMENSION
    if not text:
        return vector

    for character in text:
        digest = hashlib.sha256(character.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


@lru_cache(maxsize=1)
def _semantic_model():
    """懒加载本地ONNX模型；API进程内只创建一次推理会话。"""

    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"缺少中文Embedding模型: {EMBEDDING_MODEL_PATH}。"
            "请先下载 BAAI/bge-small-zh-v1.5，或设置 EMBEDDING_BACKEND=hash 运行基线。"
        )
    from fastembed import TextEmbedding

    # FastEmbed 的缓存根目录包含 ``snapshots/<revision>/``，而
    # ``specific_model_path`` 需要直接指向有 model_optimized.onnx 的目录。
    # 允许配置写缓存根目录，换模型 revision 时也不必手工改代码。
    model_path = EMBEDDING_MODEL_PATH
    direct_model = model_path / "model_optimized.onnx"
    if not direct_model.exists():
        snapshots = sorted(
            (
                candidate.parent
                for candidate in model_path.glob("snapshots/*/model_optimized.onnx")
                if candidate.is_file()
            ),
            key=lambda path: str(path),
        )
        if snapshots:
            model_path = snapshots[-1]
    if not (model_path / "model_optimized.onnx").exists():
        raise FileNotFoundError(
            f"模型目录中缺少 model_optimized.onnx: {EMBEDDING_MODEL_PATH}。"
        )

    return TextEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        specific_model_path=str(model_path),
        threads=None,
    )


def _as_float_lists(vectors: Iterable) -> list[list[float]]:
    """Chroma需要普通Python列表，不能直接保存NumPy数组。"""

    return [vector.astype("float32", copy=False).tolist() for vector in vectors]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """批量编码child正文，离线入库应优先调用这个函数。"""

    if EMBEDDING_BACKEND == "hash":
        return [_hash_embed(text) for text in texts]
    return _as_float_lists(
        _semantic_model().passage_embed(texts, batch_size=EMBEDDING_BATCH_SIZE)
    )


def embed_query(text: str) -> list[float]:
    """编码一个用户查询或HyDE文本。"""

    if EMBEDDING_BACKEND == "hash":
        return _hash_embed(text)
    return _as_float_lists(_semantic_model().query_embed([text]))[0]


def embed_text(text: str) -> list[float]:
    """兼容旧调用方：单条文字按查询角色编码。"""

    return embed_query(text)
