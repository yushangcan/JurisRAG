"""Combine source extraction, legal parsing and Markdown serialization."""

from __future__ import annotations

import json
from pathlib import Path

from offline_ingestion.markdown_converter.source_extractor import ExtractedSource, extract_source
from offline_ingestion.markdown_converter.structure_parser import ParsedLegalText, parse_legal_structure
from shared.models import SourceDocument


CONVERTER_NAME = "deterministic_legal_markdown"
CONVERTER_VERSION = "1.1.0"


def markdown_filename(document: SourceDocument) -> str:
    """Return a stable output name based on the immutable raw-file name."""

    return f"{Path(document.raw_file).stem}.md"


def _yaml_scalar(value: object) -> str:
    """Serialize a scalar safely without adding a YAML dependency.

    JSON strings are valid YAML strings.  ``null`` is also valid YAML and makes
    missing legal metadata explicit instead of silently inventing a value.
    """

    return json.dumps(value, ensure_ascii=False)


def render_markdown(document: SourceDocument, source: ExtractedSource, parsed: ParsedLegalText) -> str:
    """Render front matter, one document title, chapter headings and articles."""

    metadata = {
        "document_id": document.document_id,
        "title": document.title,
        "document_number": document.document_number,
        "category": document.category,
        "promulgated_date": document.promulgated_date,
        "effective_date": document.effective_date,
        "legal_status": document.legal_status,
        "source_portal": document.source_portal,
        "source_url": document.source_url,
        "attachment_url": document.attachment_url,
        "raw_file": document.raw_file,
        "raw_sha256": document.sha256,
        "source_format": source.source_format,
        "source_page_count": source.page_count,
        "converter": CONVERTER_NAME,
        "converter_version": CONVERTER_VERSION,
    }
    lines = ["---", *(f"{key}: {_yaml_scalar(value)}" for key, value in metadata.items()), "---", "", f"# {document.title}", ""]
    previous_page = None
    for paragraph in parsed.preamble:
        if paragraph.page_number and paragraph.page_number != previous_page:
            lines.extend([f"<!-- page: {paragraph.page_number} -->", ""])
            previous_page = paragraph.page_number
        lines.extend([paragraph.text, ""])

    previous_chapter = None
    for article in parsed.articles:
        if article.chapter and article.chapter != previous_chapter:
            lines.extend([f"## {article.chapter}", ""])
            previous_chapter = article.chapter
        if article.page_number and article.page_number != previous_page:
            lines.extend([f"<!-- page: {article.page_number} -->", ""])
            previous_page = article.page_number
        lines.extend([f"### {article.article_number}", ""])
        for paragraph in article.paragraphs:
            # Page markers may now appear inside one article.  This is crucial
            # for an article that starts near the bottom of one PDF page and
            # continues on the next page.
            if paragraph.page_number and paragraph.page_number != previous_page:
                lines.extend([f"<!-- page: {paragraph.page_number} -->", ""])
                previous_page = paragraph.page_number
            lines.extend([paragraph.text, ""])
    return "\n".join(lines).rstrip() + "\n"


def convert_document(document: SourceDocument, source_path: Path, output_path: Path) -> dict[str, object]:
    """Convert one manifest document and return a small machine-readable summary."""

    source = extract_source(source_path)
    parsed = parse_legal_structure(source, document.title)
    if not parsed.articles:
        raise ValueError(f"没有识别到任何条文，停止生成 Markdown: {source_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(document, source, parsed), encoding="utf-8")
    return {
        "title": document.title,
        "source_format": source.source_format,
        "markdown_file": output_path.name,
        "articles": len(parsed.articles),
        "pages": source.page_count,
    }
