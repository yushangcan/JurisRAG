"""测试使用独立 SQLite；不读取密钥、不覆盖本地运行索引。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_WORKSPACE = tempfile.TemporaryDirectory(prefix="jurisrag-tests-")
os.environ["INDEX_DIR"] = str(Path(_WORKSPACE.name) / "index")
os.environ["ENABLE_LLM"] = "false"
os.environ["RETRIEVAL_MODE"] = "lexical"


@pytest.fixture(scope="session", autouse=True)
def knowledge_index():
    from offline_ingestion.build_index import build_index
    from shared.config import INDEX_PATH
    summary = build_index()
    assert summary["documents"] == 66
    yield INDEX_PATH
    _WORKSPACE.cleanup()
