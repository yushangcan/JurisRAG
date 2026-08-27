"""Audit generated Markdown before it is allowed into the vector index.

The converter is deterministic, but an explicit audit gives the learner a
separate quality gate.  It checks provenance and structure; it does not claim
that a regulation is currently effective.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from offline_ingestion.build_index import load_manifest
from offline_ingestion.chunker import read_markdown_articles
from offline_ingestion.document_loader import read_source_file
from offline_ingestion.markdown_metadata import read_front_matter
from offline_ingestion.markdown_converter.converter import CONVERTER_NAME, CONVERTER_VERSION, markdown_filename
from shared.config import PROCESSED_MARKDOWN_DIR, RAW_LAWS_DIR


def audit_markdown_document(document) -> dict[str, object]:
    """Audit one derived file and return warnings instead of hiding problems."""

    raw_path = RAW_LAWS_DIR / document.raw_file
    markdown_path = PROCESSED_MARKDOWN_DIR / markdown_filename(document)
    warnings: list[str] = []
    if not markdown_path.exists():
        return {"title": document.title, "markdown_file": markdown_path.name, "warnings": ["markdown_missing"]}

    metadata = read_front_matter(markdown_path)
    markdown_text = markdown_path.read_text(encoding="utf-8")
    articles = read_markdown_articles(markdown_path)
    article_numbers = [article.article_number for article in articles]
    segment_pages = [segment.page_number for article in articles for segment in article.segments]
    duplicate_numbers = sorted(number for number, count in Counter(article_numbers).items() if count > 1)
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw_text = read_source_file(raw_path)
    body_characters = sum(len(article.body) for article in articles)

    if metadata.get("raw_sha256") != document.sha256 or raw_hash != document.sha256:
        warnings.append("raw_sha256_mismatch")
    if metadata.get("title") != document.title or f"# {document.title}" not in markdown_text:
        warnings.append("title_mismatch")
    if metadata.get("raw_file") != document.raw_file:
        warnings.append("raw_file_mismatch")
    if metadata.get("converter") != CONVERTER_NAME or metadata.get("converter_version") != CONVERTER_VERSION:
        warnings.append("converter_version_mismatch")
    if not articles:
        warnings.append("no_articles")
    if duplicate_numbers:
        warnings.append("duplicate_article_numbers")
    if metadata.get("source_format") == "pdf":
        page_count = int(metadata.get("source_page_count") or 0)
        if any(page is None for page in segment_pages):
            warnings.append("pdf_segment_missing_page")
        if any(page is not None and not 1 <= page <= page_count for page in segment_pages):
            warnings.append("pdf_page_out_of_range")
    # Markdown drops web navigation, headers and formatting whitespace, so an
    # exact character comparison is inappropriate.  Less than 45% is only a
    # conservative warning that asks a human to inspect the result.
    if raw_text and body_characters / len(raw_text) < 0.45:
        warnings.append("body_suspiciously_short")

    return {
        "title": document.title,
        "markdown_file": markdown_path.name,
        "source_format": metadata.get("source_format"),
        "articles": len(articles),
        "body_characters": body_characters,
        "duplicate_article_numbers": duplicate_numbers,
        "pages_in_body": sorted({page for page in segment_pages if page is not None}),
        "warnings": warnings,
    }


def audit_markdown_corpus() -> dict[str, object]:
    """Audit all manifest outputs and report derived files not in the manifest."""

    documents = load_manifest()
    results = [audit_markdown_document(document) for document in documents]
    expected = {markdown_filename(document) for document in documents}
    actual = {path.name for path in PROCESSED_MARKDOWN_DIR.glob("*.md")}
    return {
        "document_count": len(results),
        "article_count": sum(int(result.get("articles", 0)) for result in results),
        "warning_count": sum(bool(result["warnings"]) for result in results),
        "documents": results,
        "unexpected_markdown_files": sorted(actual - expected),
    }


if __name__ == "__main__":
    report = audit_markdown_corpus()
    for item in report["documents"]:
        print(
            f"{item['title']} | articles={item.get('articles', 0)} | "
            f"body_chars={item.get('body_characters', 0)} | warnings={item['warnings']}"
        )
    print(
        f"Markdown 审计完成: documents={report['document_count']}, "
        f"articles={report['article_count']}, warnings={report['warning_count']}"
    )
    if report["warning_count"] or report["unexpected_markdown_files"]:
        raise SystemExit(1)
