"""从已审计 Markdown 原子重建 SQLite；不需要向量模型、Chroma 或 PDF 依赖。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from array import array
from pathlib import Path

from offline_ingestion.chunker import build_markdown_nodes
from offline_ingestion.corpus import load_corpus_entries, verify_file_hash
from offline_ingestion.markdown_metadata import read_front_matter
from shared.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_BACKEND, INDEX_PATH, MARKDOWN_CHECKSUMS_PATH, PROJECT_ROOT, RETRIEVAL_MODE
from shared.legal_text import article_value, tokens
from shared.store import SCHEMA_VERSION


def verify_markdown(path: Path, raw_sha256: str) -> None:
    """校验冻结的派生文件摘要，避免每次建库重新加载 PDF 解析器。"""
    records = json.loads(MARKDOWN_CHECKSUMS_PATH.read_text(encoding="utf-8"))["files"]
    key = path.relative_to(PROJECT_ROOT).as_posix()
    expected = records.get(key)
    if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Markdown 校验失败: {path.name}；重新转换并运行 offline_ingestion.verify_snapshots --refresh")
    if read_front_matter(path).get("raw_sha256") != raw_sha256:
        raise ValueError(f"Markdown 与原文件哈希不一致: {path.name}")


def build_index(*, path: Path = INDEX_PATH) -> dict[str, int]:
    """先验证全部输入，再写临时库；失败时保留原有可用索引。"""
    entries, duplicates = load_corpus_entries()
    for entry in entries:
        verify_file_hash(entry.source_path, entry.document.sha256)
        verify_markdown(entry.markdown_path, str(entry.document.sha256))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building-" + uuid.uuid4().hex)
    db = sqlite3.connect(temporary)
    summary = {"documents": len(entries), "skipped_duplicates": duplicates, "parents": 0, "children": 0}
    try:
        db.executescript("""
            CREATE TABLE info(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents(id TEXT PRIMARY KEY, title TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE nodes(id TEXT PRIMARY KEY, parent_id TEXT NOT NULL, kind TEXT NOT NULL,
                               text TEXT NOT NULL, metadata TEXT NOT NULL, article_value INTEGER, vector BLOB);
            CREATE INDEX nodes_parent ON nodes(parent_id, kind);
            CREATE INDEX nodes_article ON nodes(article_value, kind);
            CREATE VIRTUAL TABLE search_index USING fts5(title, body);
        """)
        for entry in entries:
            parents, children = build_markdown_nodes(entry.markdown_path, entry.document, CHUNK_SIZE, CHUNK_OVERLAP)
            db.execute("INSERT INTO documents VALUES(?,?,?)", (entry.document.document_id, entry.document.title,
                       json.dumps(entry.document.to_metadata(), ensure_ascii=False)))
            vectors = [None] * len(children)
            if RETRIEVAL_MODE == "hybrid":
                from shared.embedding import embed_documents
                vectors = [array("f", row).tobytes() for row in embed_documents([node.text for node in children])]
            for kind, nodes in (("parent", parents), ("child", children)):
                for index, node in enumerate(nodes):
                    parent_id = node.chunk_id if kind == "parent" else node.parent_article_id or node.parent_section_id
                    cursor = db.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?,?)", (node.chunk_id, parent_id, kind, node.text,
                                        json.dumps(node.to_metadata(), ensure_ascii=False), article_value(node.article_number),
                                        vectors[index] if kind == "child" else None))
                    if kind == "child":
                        db.execute("INSERT INTO search_index(rowid,title,body) VALUES(?,?,?)", (cursor.lastrowid,
                                   " ".join(tokens(entry.document.title)), " ".join(tokens(node.text))))
                summary["parents" if kind == "parent" else "children"] += len(nodes)
        metadata = {**summary, "embedding_backend": EMBEDDING_BACKEND if RETRIEVAL_MODE == "hybrid" else "none",
                    "markdown_snapshot_sha256": hashlib.sha256(MARKDOWN_CHECKSUMS_PATH.read_bytes()).hexdigest()}
        db.executemany("INSERT INTO info VALUES(?,?)", [(key, str(value)) for key, value in metadata.items()])
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        db.execute("INSERT INTO search_index(search_index) VALUES('optimize')")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite 索引完整性检查失败")
        db.commit()
        db.close()
        os.replace(temporary, path)
    except BaseException:
        db.close()
        temporary.unlink(missing_ok=True)
        raise
    return summary


if __name__ == "__main__":
    print(json.dumps(build_index(), ensure_ascii=False, indent=2))
