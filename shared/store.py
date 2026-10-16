"""SQLite + FTS5 证据存储。每次操作使用独立连接，不创建远端数据库。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from shared.config import INDEX_PATH
from shared.legal_text import article_value, number_key, title_matches, tokens

SCHEMA_VERSION = 1


class IndexUnavailable(RuntimeError):
    pass


class LegalStore:
    def __init__(self, path: Path = INDEX_PATH) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """只读打开；索引缺失时拒绝创建空库冒充就绪。"""
        if not self.path.is_file():
            raise IndexUnavailable("知识库尚未建立，请先运行 python -m offline_ingestion.build_index")
        try:
            connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)
        except sqlite3.DatabaseError as exc:
            raise IndexUnavailable("知识库无法打开，请检查文件权限或重建索引") from exc
        connection.row_factory = sqlite3.Row
        try:
            if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise IndexUnavailable("知识库格式不兼容，请重新建立索引")
            yield connection
        except sqlite3.DatabaseError as exc:
            raise IndexUnavailable("知识库不可读取，请检查或重建索引") from exc
        finally:
            connection.close()

    @staticmethod
    def row(value: sqlite3.Row) -> dict[str, Any]:
        metadata = json.loads(value["metadata"])
        return {"chunk_id": value["id"], "parent_id": value["parent_id"], "text": value["text"],
                "title": metadata.get("title", ""), "article_number": metadata.get("article_number", ""),
                "source_url": metadata.get("source_url", ""), "metadata": metadata}

    def health(self) -> dict[str, Any]:
        try:
            with self.connect() as db:
                counts = dict(db.execute("SELECT kind, count(*) FROM nodes GROUP BY kind"))
                metadata = dict(db.execute("SELECT key, value FROM info"))
                fts_count = db.execute("SELECT count(*) FROM search_index").fetchone()[0]
                vector_count = db.execute("SELECT count(*) FROM nodes WHERE kind='child' AND vector IS NOT NULL").fetchone()[0]
            ready = bool(counts.get("child") and counts.get("parent") and fts_count == counts["child"])
            return {"status": "ok" if ready else "unavailable", "ready": ready,
                    "index_ready": ready, "backend": "sqlite_fts5", "indexed_children": counts.get("child", 0),
                    "indexed_parents": counts.get("parent", 0), "indexed_chunks": counts.get("child", 0),
                    "documents": int(metadata.get("documents", 0)), "vectors_complete": vector_count == counts.get("child", 0),
                    "embedding_backend": metadata.get("embedding_backend", "none"), "error_type": None}
        except (IndexUnavailable, sqlite3.DatabaseError):
            return {"status": "unavailable", "ready": False, "index_ready": False,
                    "backend": "sqlite_fts5", "indexed_children": 0, "indexed_parents": 0,
                    "error_type": "IndexUnavailable"}

    def fetch(self, ids: list[str], kind: str | None = None) -> list[dict[str, Any]]:
        if not ids:
            return []
        if len(ids) > 40:
            raise ValueError("一次最多读取40条证据")
        sql = f"SELECT * FROM nodes WHERE id IN ({','.join('?' for _ in ids)})"
        params: list[Any] = list(ids)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        with self.connect() as db:
            found = {row["id"]: self.row(row) for row in db.execute(sql, params)}
        return [found[key] for key in dict.fromkeys(ids) if key in found]

    def exact(self, title: str = "", article: str = "", number: str = "", limit: int = 10) -> list[dict[str, Any]]:
        if not any((title, article, number)):
            raise ValueError("精确检索至少需要法规名、条号或文号")
        with self.connect() as db:
            sql = "SELECT * FROM nodes WHERE kind='child'"
            params: list[Any] = []
            if article:
                sql += " AND article_value=?"
                params.append(article_value(article))
            rows = [self.row(row) for row in db.execute(sql, params)]
        rows = [row for row in rows if (not title or title_matches(title, row["title"]))
                and (not number or number_key(number) == number_key(row["metadata"].get("document_number", "")))]
        for row in rows:
            row.update(retrieval_channels=["exact"], exact_matches=[name for name, value in
                       (("document_title", title), ("article_number", article), ("document_number", number)) if value])
        return rows[:limit]

    def search(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        terms = list(dict.fromkeys(tokens(query)))[:100]
        if not terms:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        with self.connect() as db:
            rows = db.execute("SELECT n.*, bm25(search_index, 2.0, 1.0) AS rank FROM search_index "
                              "JOIN nodes n ON n.rowid=search_index.rowid WHERE search_index MATCH ? "
                              "ORDER BY rank, n.id LIMIT ?", (expression, limit)).fetchall()
        result = []
        for row in rows:
            item = self.row(row)
            item.update(lexical_score=-float(row["rank"]), retrieval_channels=["bm25"])
            result.append(item)
        return result

    def documents(self, title_or_number: str = "") -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = [json.loads(row[0]) for row in db.execute("SELECT metadata FROM documents ORDER BY title")]
        if title_or_number:
            rows = [row for row in rows if title_matches(title_or_number, row["title"])
                    or number_key(title_or_number) == number_key(row.get("document_number", ""))]
        return rows
