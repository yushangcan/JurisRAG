"""Central project paths and environment settings.

Keeping paths here means the offline and online programs use exactly the same
index location.  A beginner can change the data directory in one place instead
of hunting through several scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(env_name: str, default: Path) -> Path:
    """读取可选路径参数，并把相对路径统一解释为项目根目录下的路径。"""

    configured = Path(os.getenv(env_name, str(default))).expanduser()
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


# 原文件、派生 Markdown 和数据库彼此分开，防止把程序生成内容误当原文。
RAW_LAWS_DIR = _project_path(
    "RAW_LAWS_DIR", PROJECT_ROOT / "data" / "source" / "primary_laws"
)
PROCESSED_MARKDOWN_DIR = _project_path(
    "PROCESSED_MARKDOWN_DIR", PROJECT_ROOT / "data" / "processed" / "markdown"
)
INDEX_DIR = _project_path(
    "INDEX_DIR", PROJECT_ROOT / "data" / "index" / "legal_knowledge"
)
LEGAL_COLLECTION_SOURCE_DIR = _project_path(
    "LEGAL_COLLECTION_SOURCE_DIR",
    PROJECT_ROOT / "data" / "source" / "legal_collection",
)
LEGAL_COLLECTION_MARKDOWN_DIR = _project_path(
    "LEGAL_COLLECTION_MARKDOWN_DIR",
    PROJECT_ROOT / "data" / "processed" / "legal_collection",
)
MANIFEST_PATH = RAW_LAWS_DIR / "manifest.json"
LEGAL_COLLECTION_AUDIT_PATH = LEGAL_COLLECTION_MARKDOWN_DIR / "转换审计报告.json"
INCLUDE_LEGAL_COLLECTION = os.getenv("INCLUDE_LEGAL_COLLECTION", "true").strip().lower() == "true"

# v6 支持 hash 与中文 BGE 两套可切换向量。集合名必须带模型标识，避免
# 128维旧向量和512维新向量落进同一个Chroma集合。
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "bge_small_zh_v1_5").strip().lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5").strip()
EMBEDDING_MODEL_PATH = Path(
    os.getenv(
        "EMBEDDING_MODEL_PATH",
        str(PROJECT_ROOT / "data" / "models" / "fastembed" / "models--Qdrant--bge-small-zh-v1.5"),
    )
)
if not EMBEDDING_MODEL_PATH.is_absolute():
    EMBEDDING_MODEL_PATH = PROJECT_ROOT / EMBEDDING_MODEL_PATH
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
if EMBEDDING_BACKEND not in {"hash", "bge_small_zh_v1_5"}:
    raise ValueError(f"不支持的 EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")
if EMBEDDING_BATCH_SIZE <= 0:
    raise ValueError("EMBEDDING_BATCH_SIZE 必须大于 0")

_EMBEDDING_COLLECTION_SUFFIX = "hash128" if EMBEDDING_BACKEND == "hash" else "bge_small_zh_512"
CHROMA_COLLECTION_NAME = f"government_procurement_law_children_v6_{_EMBEDDING_COLLECTION_SUFFIX}"
PARENT_CHROMA_COLLECTION_NAME = f"government_procurement_law_parents_v6_{_EMBEDDING_COLLECTION_SUFFIX}"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
EMBEDDING_DIMENSION = 128 if EMBEDDING_BACKEND == "hash" else 512
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE 必须大于 0")
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP 必须大于等于 0 且小于 CHUNK_SIZE")
if DEFAULT_TOP_K <= 0:
    raise ValueError("DEFAULT_TOP_K 必须大于 0")

RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "bge_reranker_base_onnx").strip().lower()
RERANKER_MODEL_PATH = Path(
    os.getenv("RERANKER_MODEL_PATH", str(PROJECT_ROOT / "data" / "models" / "bge-reranker-base"))
)
if not RERANKER_MODEL_PATH.is_absolute():
    RERANKER_MODEL_PATH = PROJECT_ROOT / RERANKER_MODEL_PATH
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
if RERANKER_BACKEND not in {"none", "bge_reranker_base_onnx"}:
    raise ValueError(f"不支持的 RERANKER_BACKEND: {RERANKER_BACKEND}")
if RERANK_CANDIDATES <= 0:
    raise ValueError("RERANK_CANDIDATES 必须大于 0")

# 第一次召回不够可信时，才启动问题式/答案式/关键词式扩展。证据门不会只看
# 一个未校准的模型分数，而是结合精确编号、词项覆盖和多通道一致性。
ENABLE_MULTI_RECALL = os.getenv("ENABLE_MULTI_RECALL", "true").strip().lower() == "true"
RETRY_RERANKER_SCORE = float(os.getenv("RETRY_RERANKER_SCORE", "5.0"))
EVIDENCE_MIN_RERANKER_SCORE = float(os.getenv("EVIDENCE_MIN_RERANKER_SCORE", "3.0"))
EVIDENCE_MIN_WEIGHTED_COVERAGE = float(
    os.getenv("EVIDENCE_MIN_WEIGHTED_COVERAGE", "0.20")
)
EVIDENCE_MIN_CHANNEL_AGREEMENT = int(os.getenv("EVIDENCE_MIN_CHANNEL_AGREEMENT", "2"))
if EVIDENCE_MIN_CHANNEL_AGREEMENT <= 0:
    raise ValueError("EVIDENCE_MIN_CHANNEL_AGREEMENT 必须大于 0")
if not 0.0 <= EVIDENCE_MIN_WEIGHTED_COVERAGE <= 1.0:
    raise ValueError("EVIDENCE_MIN_WEIGHTED_COVERAGE 必须在 0 到 1 之间")

# 每一路检索先多取一些候选，再通过 RRF 融合并按 parent ID 去重。
# 66份语料下 Top-K=5 时每路取40个候选。开发集只改变这个变量的对照中，
# 20个候选命中7/10，40/60/80个都命中8/10，因此选择更小的40个。
HYBRID_CANDIDATE_MULTIPLIER = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "8"))
RRF_K = int(os.getenv("RRF_K", "60"))
if HYBRID_CANDIDATE_MULTIPLIER <= 0:
    raise ValueError("HYBRID_CANDIDATE_MULTIPLIER 必须大于 0")
if RRF_K <= 0:
    raise ValueError("RRF_K 必须大于 0")


def get_llm_settings() -> dict[str, str | bool]:
    """Read one configured answer-model provider without exposing its key.

    ``LLM_PROVIDER`` controls which group of environment variables is used.
    The project currently supports DeepSeek and Alibaba Cloud DashScope's
    OpenAI-compatible API.  Keeping provider selection here means answer code
    does not need to scatter environment-variable names across the project.
    """

    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    provider_options = {
        "deepseek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY", "").strip(),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "").strip(),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
        },
        "dashscope": {
            "api_key": os.getenv("DASHSCOPE_API_KEY", "").strip(),
            "base_url": os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).strip(),
            "model": os.getenv("DASHSCOPE_MODEL", "qwen-plus").strip(),
        },
    }
    if provider not in provider_options:
        supported = "、".join(provider_options)
        raise ValueError(f"不支持的 LLM_PROVIDER: {provider}，可选值为 {supported}")

    return {
        "enabled": os.getenv("ENABLE_LLM", "false").strip().lower() == "true",
        "provider": provider,
        **provider_options[provider],
    }
