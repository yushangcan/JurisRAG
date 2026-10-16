"""用 Cross-Encoder 对混合召回候选做第二阶段精排。

第一阶段的向量检索和 BM25 负责“尽量别漏掉”，Reranker 则把用户问题与
每个 child 正文成对阅读，负责“把更正确的候选排到前面”。这里选择 ONNX
版本的 ``bge-reranker-base``，这样 Windows/Python 3.14 的 CPU 环境不需要
安装 PyTorch。模型按需加载，一次 API 进程只创建一个推理会话。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from shared.config import RERANKER_BACKEND, RERANKER_MODEL_PATH


@lru_cache(maxsize=1)
def _runtime():
    """加载 tokenizer 和 ONNX 会话，并返回模型实际需要的输入名称。

    显式启用重排后需要模型文件完整；健康检查会报告缺失，查询不会冒充已精排。
    """

    model_file = RERANKER_MODEL_PATH / "onnx" / "model_quantized.onnx"
    tokenizer_file = RERANKER_MODEL_PATH / "tokenizer.json"
    missing = [path for path in (model_file, tokenizer_file) if not path.is_file()]
    if missing:
        missing_text = "、".join(str(path) for path in missing)
        raise FileNotFoundError(f"缺少 Reranker 模型文件：{missing_text}")

    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    # Cross-Encoder 最大上下文为 512 token。动态 padding 只补到当前 batch
    # 最长序列，避免所有短法条都无谓计算 512 个位置。
    tokenizer.enable_truncation(max_length=512)
    pad_token = "[PAD]" if tokenizer.token_to_id("[PAD]") is not None else "<pad>"
    pad_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(pad_id=pad_id or 0, pad_token=pad_token)

    session = ort.InferenceSession(
        str(model_file),
        providers=["CPUExecutionProvider"],
    )
    input_names = {item.name for item in session.get_inputs()}
    output_name = session.get_outputs()[0].name
    return tokenizer, session, input_names, output_name


def _score_batch(query: str, texts: list[str]) -> list[float]:
    """一次计算一批 ``(问题, child正文)`` 的相关性原始分数。"""

    if not texts:
        return []
    tokenizer, session, input_names, output_name = _runtime()
    encodings = tokenizer.encode_batch([(query, text) for text in texts])

    possible_inputs = {
        "input_ids": np.asarray([encoding.ids for encoding in encodings], dtype=np.int64),
        "attention_mask": np.asarray(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        ),
        "token_type_ids": np.asarray(
            [encoding.type_ids for encoding in encodings], dtype=np.int64
        ),
    }
    model_inputs = {
        name: possible_inputs[name]
        for name in input_names
        if name in possible_inputs
    }
    missing_inputs = input_names - model_inputs.keys()
    if missing_inputs:
        raise RuntimeError(f"Reranker 出现未支持的 ONNX 输入：{sorted(missing_inputs)}")

    logits = session.run([output_name], model_inputs)[0]
    # 排序只需要单调的 logit，无需额外 sigmoid；保留原始分更便于调试。
    return [float(value) for value in np.asarray(logits).reshape(-1)]


def rerank_children(
    query: str,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """给 child 候选加入 ``reranker_score`` 并返回新的排序。

    法规名+条号或文号属于确定性强信号。精确命中的候选被
    放在普通语义候选之前，避免 Cross-Encoder 把准确编号降级；同一层内再
    按 Reranker 分数排序，混合召回分数作为稳定的并列裁决。
    """

    if RERANKER_BACKEND == "none" or not rows:
        return [dict(row) for row in rows]
    if RERANKER_BACKEND != "bge_reranker_base_onnx":
        raise ValueError(f"不支持的 RERANKER_BACKEND：{RERANKER_BACKEND}")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    scores: list[float] = []
    texts = [str(row.get("text", "")) for row in rows]
    for start in range(0, len(texts), batch_size):
        scores.extend(_score_batch(query, texts[start : start + batch_size]))

    enriched = [
        {**row, "reranker_score": score}
        for row, score in zip(rows, scores, strict=True)
    ]
    return sorted(
        enriched,
        key=lambda row: (
            -int("exact" in row.get("retrieval_channels", [])),
            -float(row.get("reranker_score", float("-inf"))),
            -float(row.get("score", 0.0)),
            str(row.get("chunk_id", "")),
        ),
    )
