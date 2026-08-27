"""下载并校验本项目需要的两个 ONNX 模型。

模型总计约 380 MB，保存在 ``data/models`` 且不会提交 Git。国内网络可先在
PowerShell 设置 ``$env:HF_ENDPOINT='https://hf-mirror.com'``；下载支持断点
续传。revision 固定后，同一份代码不会悄悄换成不同权重。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

from shared.config import PROJECT_ROOT


EMBEDDING_REVISION = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
RERANKER_REVISION = "280bcc27a84e0b898c251e06fddb25171bd9b101"
RERANKER_SHA256 = "dd98f3e67837d23210a6b7550c08cced4f61845b940ac45be3565840a10f3244"


def _sha256(path: Path) -> str:
    """分块计算大文件哈希，避免一次把 279 MB 全读进内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    models_root = PROJECT_ROOT / "data" / "models"
    embedding_cache = models_root / "fastembed"
    reranker_root = models_root / "bge-reranker-base"

    print("1/2 下载 BGE-small-zh-v1.5（中文语义 Embedding）……")
    snapshot_download(
        repo_id="Qdrant/bge-small-zh-v1.5",
        revision=EMBEDDING_REVISION,
        cache_dir=embedding_cache,
        allow_patterns=[
            "config.json",
            "model_optimized.onnx",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )

    print("2/2 下载 bge-reranker-base（量化 ONNX）……")
    snapshot_download(
        repo_id="Xenova/bge-reranker-base",
        revision=RERANKER_REVISION,
        local_dir=reranker_root,
        allow_patterns=[
            "config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "onnx/model_quantized.onnx",
        ],
    )
    reranker_file = reranker_root / "onnx" / "model_quantized.onnx"
    actual_hash = _sha256(reranker_file)
    if actual_hash != RERANKER_SHA256:
        raise ValueError(
            "Reranker SHA-256 校验失败："
            f"expected={RERANKER_SHA256}, actual={actual_hash}"
        )
    print(f"模型准备完成：{models_root}")


if __name__ == "__main__":
    main()
