"""Audit the legal corpus before building a vector index.

Run with ``python -m offline_ingestion.audit_corpus``.  The report catches
missing/corrupted files and suspiciously short extraction results.  It does
not decide whether a law is currently effective; that still requires checking
the official legal source and recording the result in ``manifest.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from offline_ingestion.build_index import verify_file_hash
from offline_ingestion.document_loader import read_source_file
from shared.config import MANIFEST_PATH, RAW_LAWS_DIR


ARTICLE_PATTERN = re.compile(r"第\s*[一二三四五六七八九十百零〇两0-9]+\s*条")


def normalize_for_match(text: str) -> str:
    """Remove whitespace so titles still match text extracted from PDF lines."""

    return "".join(text.split())


def audit_document(record: dict[str, object]) -> dict[str, object]:
    """Check one manifest record and return machine-readable audit fields."""

    path = RAW_LAWS_DIR / str(record["raw_file"])
    verify_file_hash(path, str(record.get("sha256") or ""))
    text = read_source_file(path)
    title_found = normalize_for_match(str(record["title"])) in normalize_for_match(text)
    article_count = len(ARTICLE_PATTERN.findall(text))
    warnings: list[str] = []
    if len(text) < 500:
        warnings.append("extracted_text_too_short")
    if not title_found:
        warnings.append("title_not_found")
    if article_count == 0:
        warnings.append("no_article_marker_found")
    return {
        "title": record["title"],
        "raw_file": path.name,
        "bytes": path.stat().st_size,
        "characters": len(text),
        "article_markers": article_count,
        "warnings": warnings,
    }


def audit_corpus() -> dict[str, object]:
    """Audit every indexed source and report raw files not referenced by it."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = manifest["documents"]
    results = [audit_document(record) for record in records]
    referenced = {str(record["raw_file"]) for record in records}
    for record in records:
        notice_raw_file = record.get("notice_raw_file")
        if notice_raw_file:
            notice_path = RAW_LAWS_DIR / str(notice_raw_file)
            verify_file_hash(notice_path, str(record.get("notice_sha256") or ""))
            referenced.add(str(notice_raw_file))
    raw_candidates = {path.name for path in RAW_LAWS_DIR.iterdir() if path.name != MANIFEST_PATH.name}
    return {
        "document_count": len(results),
        "warning_count": sum(bool(result["warnings"]) for result in results),
        "documents": results,
        "unindexed_raw_files": sorted(raw_candidates - referenced),
    }


if __name__ == "__main__":
    report = audit_corpus()
    for item in report["documents"]:
        print(
            f"{item['title']} | chars={item['characters']} | "
            f"articles={item['article_markers']} | warnings={item['warnings']}"
        )
    print(f"语料审计完成: documents={report['document_count']}, warnings={report['warning_count']}")
    if report["unindexed_raw_files"]:
        print(f"未入库原始文件（可能是通知页或历史保留件）: {report['unindexed_raw_files']}")
