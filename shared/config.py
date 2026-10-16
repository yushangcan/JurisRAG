"""项目路径与轻量运行配置；可选模型不会在模块导入时加载。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


RAW_LAWS_DIR = _project_path("RAW_LAWS_DIR", "data/source/primary_laws")
PROCESSED_MARKDOWN_DIR = _project_path("PROCESSED_MARKDOWN_DIR", "data/processed/markdown")
LEGAL_COLLECTION_SOURCE_DIR = _project_path("LEGAL_COLLECTION_SOURCE_DIR", "data/source/legal_collection")
LEGAL_COLLECTION_MARKDOWN_DIR = _project_path("LEGAL_COLLECTION_MARKDOWN_DIR", "data/processed/legal_collection")
INDEX_DIR = _project_path("INDEX_DIR", "data/index/legal_knowledge")
INDEX_PATH = INDEX_DIR / "knowledge.sqlite3"
MANIFEST_PATH = RAW_LAWS_DIR / "manifest.json"
LEGAL_COLLECTION_AUDIT_PATH = LEGAL_COLLECTION_MARKDOWN_DIR / "转换审计报告.json"
MARKDOWN_CHECKSUMS_PATH = PROJECT_ROOT / "data/processed/markdown_checksums.json"
INCLUDE_LEGAL_COLLECTION = os.getenv("INCLUDE_LEGAL_COLLECTION", "true").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
if CHUNK_SIZE <= 0 or not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("切块长度必须为正数，且重叠长度小于切块长度")

# lexical 是完整的 BM25 检索增强流程。hybrid 才需要下载中文向量模型。
# 旧 .env 中的模型路径不会阻止默认轻量模式启动。
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "lexical").strip().lower()
if RETRIEVAL_MODE not in {"lexical", "hybrid"}:
    raise ValueError("RETRIEVAL_MODE 只能是 lexical 或 hybrid")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "bge_small_zh_v1_5").lower()
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_MODEL_PATH = _project_path("EMBEDDING_MODEL_PATH", "data/models/fastembed/models--Qdrant--bge-small-zh-v1.5")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_DIMENSION = 128 if EMBEDDING_BACKEND == "hash" else 512
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "none").lower() if RETRIEVAL_MODE == "hybrid" else "none"
RERANKER_MODEL_PATH = _project_path("RERANKER_MODEL_PATH", "data/models/bge-reranker-base")
RERANK_CANDIDATES = 20


def get_llm_settings() -> dict[str, str | bool]:
    """仅供模型工厂读取，调用方不得记录整个对象或其中的密钥。"""
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    options = {
        "deepseek": ("DEEPSEEK", "https://api.deepseek.com/v1", "deepseek-chat"),
        "dashscope": ("DASHSCOPE", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        "openai": ("OPENAI", "https://api.openai.com/v1", ""),
    }
    if provider not in options:
        raise ValueError("LLM_PROVIDER 只支持 deepseek、dashscope、openai")
    prefix, base_url, model = options[provider]
    return {
        "enabled": os.getenv("ENABLE_LLM", "false").lower() == "true",
        "provider": provider,
        "api_key": os.getenv(f"{prefix}_API_KEY", "").strip(),
        "base_url": os.getenv(f"{prefix}_BASE_URL", "").strip() or base_url,
        "model": os.getenv(f"{prefix}_MODEL", model).strip(),
    }
