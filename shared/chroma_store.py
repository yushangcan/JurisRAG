"""One place for opening the persistent Chroma collection."""

from __future__ import annotations

import os

import chromadb

from shared.config import (
    CHROMA_COLLECTION_NAME,
    INDEX_DIR,
    PARENT_CHROMA_COLLECTION_NAME,
    PROJECT_ROOT,
)


def _chroma_persist_path() -> str:
    """返回 Chroma 可稳定重开的持久化路径。

    Chroma 1.5.9 在 Windows 上用“包含中文的绝对路径”创建较大的 HNSW
    索引后，下一进程可能无法重新加载二进制段。数据库仍然保存在项目的
    ``data/index``，这里只把传给 Chroma 的字符串改为 ASCII 相对路径。
    Chroma 后台线程也依赖当前目录，因此进程会固定在项目根目录运行。
    """

    resolved_index = INDEX_DIR.resolve()
    if os.name == "nt" and any(ord(character) > 127 for character in str(resolved_index)):
        try:
            relative_index = resolved_index.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            # 用户把 INDEX_DIR 配到项目之外时，无法安全构造项目内相对路径；
            # 此时保留用户显式给出的绝对路径。
            return str(resolved_index)
        os.chdir(PROJECT_ROOT)
        return str(relative_index)
    return str(resolved_index)


def _get_named_collection(name: str, description: str, *, reset: bool = False):
    """打开一个命名集合；重建索引时只删除明确指定的这个集合。"""

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=_chroma_persist_path())
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    # Both ingestion and retrieval pass vectors explicitly.  Therefore Chroma
    # only stores/searches vectors here and does not need to own an embedding
    # model.  This also makes replacing the baseline embedding much easier.
    return client.get_or_create_collection(
        name=name,
        metadata={"description": description},
    )


def get_collection(*, reset: bool = False):
    """打开参与向量/词法混合检索的细粒度子块集合。"""

    return _get_named_collection(
        CHROMA_COLLECTION_NAME,
        "Fine-grained child chunks for hybrid legal retrieval",
        reset=reset,
    )


def get_parent_collection(*, reset: bool = False):
    """打开按父 ID 回填完整法条或普通章节的父块集合。"""

    return _get_named_collection(
        PARENT_CHROMA_COLLECTION_NAME,
        "Full parent articles and sections for parent-child retrieval",
        reset=reset,
    )
