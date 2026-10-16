"""Convert the independent government-procurement collection to Markdown.

This command is intentionally separate from ``convert_corpus.py``:

1. ``convert_corpus.py`` serves the RAG project's current main manifest.
2. This file reads ``data/source/legal_collection`` and writes matching
   structured Markdown to ``data/processed/legal_collection``.
3. Conversion does not build the retrieval index. A human can audit the
   Markdown first and decide later which files enter the index.

The implementation is verbose on purpose: a beginner should be able to follow
the path from checksum verification, through extraction and legal parsing, to
the final Markdown and audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from offline_ingestion.markdown_converter.source_extractor import ExtractedSource, SourceBlock, extract_source
from offline_ingestion.markdown_converter.structure_parser import (
    CHAPTER_PATTERN,
    ParsedLegalText,
    parse_legal_structure,
)
from shared.models import SourceDocument
from shared.config import (
    LEGAL_COLLECTION_MARKDOWN_DIR,
    LEGAL_COLLECTION_SOURCE_DIR,
    RAW_LAWS_DIR,
)


DEFAULT_COLLECTION_ROOT = LEGAL_COLLECTION_SOURCE_DIR
DEFAULT_OUTPUT_ROOT = LEGAL_COLLECTION_MARKDOWN_DIR
OUTPUT_DIRECTORY_NAME = "03_结构化Markdown"
AUDIT_REPORT_NAME = "转换审计报告.json"
SUPPORTED_SUFFIXES = {".docx", ".html", ".htm", ".pdf"}
SOURCE_CATEGORY_DIRECTORIES = ("01_政府采购核心", "02_工程招投标衔接", "03_配套附件")
SUPPLEMENT_MANIFEST_GLOB = "补充文件清单_*.json"

CONVERTER_NAME = "government_procurement_collection_markdown"
CONVERTER_VERSION = "1.1.0"

CHINESE_NUMBER = r"[一二三四五六七八九十百千万零〇两0-9]+"
SECTION_PATTERN = re.compile(rf"^(第\s*{CHINESE_NUMBER}\s*节)\s*(.*)$")
TOP_LEVEL_PATTERN = re.compile(r"^([一二三四五六七八九十百]+、)\s*(.+)$")
SECOND_LEVEL_PATTERN = re.compile(r"^([（(][一二三四五六七八九十百0-9]+[）)])\s*(.+)$")
THIRD_LEVEL_PATTERN = re.compile(r"^([0-9]+[.、])\s*(.+)$")
PAGE_MARKER_PATTERN = re.compile(r"<!-- page: (\d+) -->")


@dataclass(frozen=True)
class CollectionMetadata:
    """One source record plus provenance fields specific to this collection."""

    document: SourceDocument
    source_role: str
    source_page_published_at: str | None = None
    verification_url: str | None = None


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    """Read one UTF-8 JSON file and show its path if the data is malformed."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取元数据 JSON: {path}") from exc


def _find_collection_manifest(collection_root: Path) -> Path | None:
    """Find the one collection-level supplement manifest, if it exists.

    The raw collection lives outside this Git repository.  Keeping its
    manifest beside the downloaded files means a future collection can be
    copied to another drive without changing Python code.  We fail loudly when
    two dated manifests are present because silently choosing one could attach
    the wrong title or checksum to a legal source.
    """

    candidates = sorted(
        path
        for path in collection_root.glob(SUPPLEMENT_MANIFEST_GLOB)
        if path.is_file()
    )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"法律数据目录存在多个补充清单，请只保留一个: {names}")
    return candidates[0] if candidates else None


def _infer_category(title: str, document_number: str | None) -> str:
    """Infer only a broad document type when the download manifest omits it."""

    number = document_number or ""
    if title.startswith("中华人民共和国") and title.endswith("法"):
        return "法律"
    if title.endswith("实施条例"):
        return "行政法规"
    if "令第" in number:
        return "部门规章"
    if number.startswith("国办发"):
        return "国务院办公厅文件"
    if number.startswith("财办库"):
        return "部门文件"
    return "部门规范性文件"


def _metadata_from_collection_manifest(collection_root: Path) -> dict[str, CollectionMetadata]:
    """Read metadata for files downloaded into the external collection.

    The main RAG manifest intentionally remains unchanged: these files are
    being audited before they enter the vector index.  This adapter reads the
    collection's own JSON, indexes it by SHA-256, and lets the conversion loop
    prove that the metadata belongs to the exact bytes on disk.
    """

    path = _find_collection_manifest(collection_root)
    if path is None:
        return {}

    payload = _read_json(path)
    raw_documents = payload.get("documents", [])
    if not isinstance(raw_documents, list):
        raise ValueError(f"补充文件清单的 documents 必须是数组: {path}")

    records: dict[str, CollectionMetadata] = {}
    seen_relative_paths: set[str] = set()
    document_ids: set[str] = set()
    parent_links: list[tuple[str, str]] = []
    for raw_document in raw_documents:
        if not isinstance(raw_document, dict):
            raise ValueError(f"补充文件清单包含非对象记录: {path}")
        record = dict(raw_document)
        sha256 = str(record.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"补充文件清单包含非法 SHA-256: {record.get('raw_file')}")
        if not record.get("raw_file") or not record.get("title"):
            raise ValueError(f"补充文件清单缺少 title/raw_file: {record}")

        relative_key = str(record["raw_file"]).replace("\\", "/")
        if relative_key in seen_relative_paths:
            raise ValueError(f"补充文件清单重复记录同一路径: {relative_key}")
        seen_relative_paths.add(relative_key)

        # Resolve the manifest path before trusting it.  This catches a typo in
        # the JSON even when SHA256SUMS.txt happens to contain the real file.
        source_path = (collection_root / Path(relative_key)).resolve()
        if not source_path.is_relative_to(collection_root.resolve()):
            raise ValueError(f"补充文件路径越出数据目录: {relative_key}")
        if not source_path.is_file():
            raise FileNotFoundError(f"补充文件清单中的源文件不存在: {relative_key}")
        if _sha256(source_path) != sha256:
            raise ValueError(f"补充文件清单 SHA-256 校验失败: {relative_key}")

        document = SourceDocument.from_manifest_record(record)
        if document.document_id in document_ids:
            raise ValueError(f"补充文件清单存在重复 document_id: {document.document_id}")
        document_ids.add(document.document_id)
        if document.parent_document_id:
            parent_links.append((document.document_id, document.parent_document_id))
        published_at = record.get("source_page_published_at")
        records[sha256] = CollectionMetadata(
            document=document,
            source_role=str(record.get("source_role", "official_source")),
            source_page_published_at=str(published_at) if published_at else None,
            verification_url=record.get("verification_url"),
        )

    # All current attachments and their notices live in this supplement.  A
    # dangling relation would make citations point to a non-existent mother
    # document, so stop before producing Markdown.
    for document_id, parent_document_id in parent_links:
        if parent_document_id not in document_ids:
            raise ValueError(
                f"补充附件 {document_id} 的 parent_document_id 不存在: {parent_document_id}"
            )
    return records


def _metadata_from_main_manifest() -> dict[str, CollectionMetadata]:
    """Index existing official-source metadata by immutable source hash."""

    path = RAW_LAWS_DIR / "manifest.json"
    payload = _read_json(path)
    records: dict[str, CollectionMetadata] = {}
    for raw_record in payload.get("documents", []):
        record = dict(raw_record)
        sha256 = record.get("sha256")
        if not sha256:
            continue
        # A web-page publication timestamp is not automatically the legal
        # document's promulgation date.  Keep the two concepts separate.
        source_page_published_at = record.get("source_page_published_at")
        if "promulgated_date" not in record:
            record["promulgated_date"] = None
        document = SourceDocument.from_manifest_record(record)
        role = "official_attachment" if document.attachment_url else "official_full_text_page"
        records[str(sha256)] = CollectionMetadata(
            document=document,
            source_role=role,
            source_page_published_at=str(source_page_published_at) if source_page_published_at else None,
        )
    return records


def _metadata_from_additions_manifest() -> dict[str, CollectionMetadata]:
    """Expand the additions manifest so every retained attachment has metadata."""

    path = (
        RAW_LAWS_DIR
        / "government_procurement_core_additions_2026-08-15"
        / "download_manifest.json"
    )
    payload = _read_json(path)
    records: dict[str, CollectionMetadata] = {}
    for raw_document in payload.get("documents", []):
        base = dict(raw_document)
        for raw_source in base.get("source_files", []):
            source = dict(raw_source)
            sha256 = str(source["sha256"])
            attachment_url = source.get("source_url") if "attachment" in str(source.get("role", "")) else None
            manifest_record = {
                "document_id": f"collection-{sha256[:16]}",
                "title": base["title"],
                "document_number": base.get("document_number"),
                "category": _infer_category(str(base["title"]), base.get("document_number")),
                "promulgated_date": base.get("promulgated_date"),
                "effective_date": base.get("effective_date"),
                "legal_status": base.get("legal_status"),
                "source_portal": base["source_portal"],
                # The notice page explains context; the optional attachment URL
                # points to the exact PDF that was retained.
                "source_url": base["source_url"],
                "attachment_url": attachment_url,
                "raw_file": source["raw_file"],
                "sha256": sha256,
            }
            records[sha256] = CollectionMetadata(
                document=SourceDocument.from_manifest_record(manifest_record),
                source_role=str(source.get("role", "official_source")),
            )
    return records


def _manual_metadata() -> dict[str, CollectionMetadata]:
    """Describe the two later additions that are not in either project manifest."""

    records = [
        (
            "4798cfd7f97c12bd8646ce0dddaa9eb03eac757e9a5e807c47dd8d7153d348e9",
            {
                "document_id": "mof-treasury-2022-3",
                "title": "财政部关于《中华人民共和国政府采购法实施条例》第十九条第一款“较大数额罚款”具体适用问题的意见",
                "document_number": "财库〔2022〕3号",
                "category": "部门规范性文件",
                "promulgated_date": "2022-01-05",
                "effective_date": "2022-02-08",
                "source_portal": "中国政府采购网",
                "source_url": "https://www.ccgp.gov.cn/zcfg/mof/202201/t20220113_17517412.htm",
                "raw_file": "placeholder.html",
            },
            "official_full_text_page",
            "2022-01-13",
            None,
        ),
        (
            "732c39b4e0106fcc42e04270d0cb976b3b2e049d84c2d7a94537fc4e45208948",
            {
                "document_id": "mof-office-2015-352",
                "title": "关于转发国务院法制办公室《对政府采购工程项目法律适用及申领施工许可证问题的答复》的通知",
                "document_number": "财办库〔2015〕352号",
                "category": "部门文件",
                "promulgated_date": "2015-12-02",
                "source_portal": "湖南省公共资源交易中心",
                "source_url": "https://ggzy.hunan.gov.cn/ggzy/xxgk/xxgkml/zcfg/gfxwj/201802/t20180227_21175986.html",
                "revision_note": "财政部公开通知页可用于交叉核对标题、文号和附件。",
                "raw_file": "placeholder.html",
            },
            "public_full_text_copy_verified_against_mof_notice",
            "2018-02-27",
            "https://www.mof.gov.cn/gp/xxgkml/gks/201512/t20151210_2511462.htm",
        ),
    ]

    result: dict[str, CollectionMetadata] = {}
    for sha256, raw_record, source_role, source_page_published_at, verification_url in records:
        record = dict(raw_record)
        record["sha256"] = sha256
        result[sha256] = CollectionMetadata(
            document=SourceDocument.from_manifest_record(record),
            source_role=source_role,
            source_page_published_at=source_page_published_at,
            verification_url=verification_url,
        )
    return result


def load_metadata_by_hash(collection_root: Path | None = None) -> dict[str, CollectionMetadata]:
    """Combine repository metadata with the external collection manifest.

    Existing records come from the RAG project's manifests; newly downloaded
    records come from ``补充文件清单_*.json`` beside the raw collection.  A
    duplicate hash is acceptable only when its title agrees, which protects
    against accidentally reusing one file for two different legal documents.
    """

    combined: dict[str, CollectionMetadata] = {}
    groups = [
        _metadata_from_main_manifest(),
        _metadata_from_additions_manifest(),
        _manual_metadata(),
    ]
    if collection_root is not None:
        groups.append(_metadata_from_collection_manifest(collection_root.resolve()))

    for group in groups:
        for sha256, metadata in group.items():
            if sha256 in combined and combined[sha256].document.title != metadata.document.title:
                raise ValueError(f"同一 SHA-256 对应两个不同标题: {sha256}")
            combined[sha256] = metadata
    return combined


def read_expected_checksums(collection_root: Path) -> dict[str, str]:
    """Read ``SHA256SUMS.txt`` into ``relative/path -> digest`` mappings."""

    checksum_path = collection_root / "SHA256SUMS.txt"
    expected: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) \*(.+)", line)
        if not match:
            raise ValueError(f"SHA256SUMS.txt 第 {line_number} 行格式错误: {line}")
        relative_path = match.group(2).replace("\\", "/")
        if relative_path in expected:
            raise ValueError(f"SHA256SUMS.txt 重复记录路径: {relative_path}")
        expected[relative_path] = match.group(1).lower()
    return expected


def discover_source_files(collection_root: Path) -> list[Path]:
    """Find supported source files in all three data categories.

    Notes, JSON manifests and the previous Markdown output are deliberately
    outside these directories, so they cannot be mistaken for legal sources.
    """

    paths: list[Path] = []
    for directory_name in SOURCE_CATEGORY_DIRECTORIES:
        directory = collection_root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"缺少法律分类目录: {directory}")
        paths.extend(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
    return sorted(paths, key=lambda path: path.relative_to(collection_root).as_posix())


def _yaml_scalar(value: object) -> str:
    """Encode a scalar as JSON; every JSON scalar is also valid YAML."""

    return json.dumps(value, ensure_ascii=False)


def _front_matter(metadata: CollectionMetadata, source: ExtractedSource) -> list[str]:
    """Build transparent provenance metadata for one Markdown document."""

    document = metadata.document
    values = {
        "document_id": document.document_id,
        "title": document.title,
        "document_number": document.document_number,
        "category": document.category,
        "promulgated_date": document.promulgated_date,
        "effective_date": document.effective_date,
        # ``null`` means not verified; it must never be guessed from publication.
        "legal_status": document.legal_status,
        "status_verified_at": document.status_verified_at,
        "source_page_published_at": metadata.source_page_published_at,
        "source_portal": document.source_portal,
        "source_url": document.source_url,
        "attachment_url": document.attachment_url,
        "parent_document_id": document.parent_document_id,
        "verification_url": metadata.verification_url,
        "source_role": metadata.source_role,
        "revision_note": document.revision_note,
        "raw_file": document.raw_file,
        "raw_sha256": document.sha256,
        "source_format": source.source_format,
        "source_page_count": source.page_count,
        "converter": CONVERTER_NAME,
        "converter_version": CONVERTER_VERSION,
    }
    return ["---", *(f"{key}: {_yaml_scalar(value)}" for key, value in values.items()), "---"]


def _append_page_marker(lines: list[str], block: SourceBlock, previous_page: int | None) -> int | None:
    """Insert a PDF page marker exactly when the original source page changes."""

    if block.page_number is not None and block.page_number != previous_page:
        lines.extend([f"<!-- page: {block.page_number} -->", ""])
        return block.page_number
    return previous_page


def _general_markdown_line(text: str) -> str:
    """Promote obvious short legal headings while leaving substantive text intact."""

    compact = "".join(text.split())
    chapter = CHAPTER_PATTERN.match(text)
    if chapter and len(text) <= 80:
        return f"## {chapter.group(1).replace(' ', '')} {chapter.group(2).strip()}".rstrip()
    section = SECTION_PATTERN.match(text)
    if section and len(text) <= 80:
        return f"### {section.group(1).replace(' ', '')} {section.group(2).strip()}".rstrip()
    if compact in {"说明", "附件"}:
        return f"## {compact}"
    if text.startswith("附件") and len(text) <= 50:
        return f"## {text}"
    if TOP_LEVEL_PATTERN.match(text) and len(text) <= 80:
        return f"## {text}"
    if SECOND_LEVEL_PATTERN.match(text) and len(text) <= 60:
        return f"### {text}"
    if THIRD_LEVEL_PATTERN.match(text) and len(text) <= 50:
        return f"#### {text}"
    return text


def _table_markdown_row(text: str) -> tuple[str, bool] | None:
    """Convert a catalog extractor row and flag whether it is the header row."""

    cells = text.split(" | ")
    if len(cells) != 3:
        return None
    normalized_header = ["".join(cell.split()) for cell in cells]
    is_header = normalized_header == ["编码", "品目名称", "说明"]
    escaped = [cell.replace("|", "\\|") for cell in cells]
    return f"| {' | '.join(escaped)} |", is_header


def render_collection_markdown(
    metadata: CollectionMetadata,
    source: ExtractedSource,
    parsed: ParsedLegalText,
) -> tuple[str, str]:
    """Render an article-aware document or a structured general notice/catalog."""

    document = metadata.document
    lines = [*_front_matter(metadata, source), "", f"# {document.title}", ""]
    previous_page: int | None = None

    if not parsed.articles:
        # Notices, opinions and the 147-page procurement catalog do not use
        # ``第×条``.  They are still valid source documents, so they use a
        # general heading hierarchy rather than being rejected or mislabeled.
        normalized_title = "".join(document.title.split())
        in_markdown_table = False
        for index, block in enumerate(source.blocks):
            if block.page_number is not None and block.page_number != previous_page and in_markdown_table:
                lines.append("")
                in_markdown_table = False
            previous_page = _append_page_marker(lines, block, previous_page)
            if index < 10 and "".join(block.text.split()) == normalized_title:
                continue
            table_row = _table_markdown_row(block.text)
            if table_row:
                row, is_header = table_row
                if is_header:
                    lines.extend([row, "| --- | --- | --- |"])
                    in_markdown_table = True
                else:
                    lines.append(row)
                continue
            if in_markdown_table:
                lines.append("")
                in_markdown_table = False
            lines.extend([_general_markdown_line(block.text), ""])
        return "\n".join(lines).rstrip() + "\n", "general_sections"

    for paragraph in parsed.preamble:
        block = SourceBlock(paragraph.text, paragraph.page_number)
        previous_page = _append_page_marker(lines, block, previous_page)
        lines.extend([paragraph.text, ""])

    previous_chapter: str | None = None
    for article in parsed.articles:
        if article.chapter and article.chapter != previous_chapter:
            lines.extend([f"## {article.chapter}", ""])
            previous_chapter = article.chapter
        article_block = SourceBlock(article.article_number, article.page_number)
        previous_page = _append_page_marker(lines, article_block, previous_page)
        lines.extend([f"### {article.article_number}", ""])
        for paragraph in article.paragraphs:
            paragraph_block = SourceBlock(paragraph.text, paragraph.page_number)
            previous_page = _append_page_marker(lines, paragraph_block, previous_page)
            lines.extend([paragraph.text, ""])
    return "\n".join(lines).rstrip() + "\n", "legal_articles"


def _audit_output(
    source_path: Path,
    output_path: Path,
    source: ExtractedSource,
    parsed: ParsedLegalText,
    markdown: str,
    expected_sha256: str,
    structure_mode: str,
) -> dict[str, object]:
    """Apply per-file structural, provenance and PDF-page quality checks."""

    warnings: list[str] = []
    source_characters = sum(len(block.text) for block in source.blocks)
    markdown_characters = len(markdown)
    page_markers = {int(number) for number in PAGE_MARKER_PATTERN.findall(markdown)}
    source_pages = {block.page_number for block in source.blocks if block.page_number is not None}
    table_count = markdown.count("| --- | --- | --- |")
    table_rows = sum(
        line.startswith("| ")
        and line != "| --- | --- | --- |"
        and "编 码" not in line
        and "编码" not in line
        for line in markdown.splitlines()
    )

    if not output_path.exists() or output_path.stat().st_size == 0:
        warnings.append("markdown_missing_or_empty")
    if f"# " not in markdown:
        warnings.append("title_heading_missing")
    if expected_sha256 not in markdown:
        warnings.append("sha256_missing_from_front_matter")
    if source_characters and markdown_characters < source_characters * 0.8:
        warnings.append("markdown_suspiciously_short")
    if source.source_format == "pdf" and page_markers != source_pages:
        warnings.append("pdf_page_markers_do_not_match_extracted_pages")
    if source.source_format != "pdf" and page_markers:
        warnings.append("non_pdf_has_fabricated_page_markers")

    return {
        "source_file": source_path.name,
        "markdown_file": output_path.name,
        # 派生 Markdown 也要可核验，防止转换完成后被手工改写却仍沿用旧报告。
        # 必须对实际落盘字节计算；Windows 可能把 \n 写成 \r\n，内存字符串
        # 哈希不能代表用户之后真正交给入库程序的文件。
        "markdown_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "source_format": source.source_format,
        "structure_mode": structure_mode,
        "source_blocks": len(source.blocks),
        "source_characters": source_characters,
        "markdown_characters": markdown_characters,
        "articles": len(parsed.articles),
        "source_page_count": source.page_count,
        "page_markers": sorted(page_markers),
        "markdown_table_count": table_count,
        "markdown_table_data_rows": table_rows,
        "sha256": expected_sha256,
        "warnings": warnings,
    }


def convert_collection(
    collection_root: Path = DEFAULT_COLLECTION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    """Verify, convert and audit every retained source file in the collection."""

    collection_root = collection_root.resolve()
    output_root = output_root.resolve()
    expected_checksums = read_expected_checksums(collection_root)
    metadata_by_hash = load_metadata_by_hash(collection_root)
    source_paths = discover_source_files(collection_root)

    discovered_relatives = {path.relative_to(collection_root).as_posix() for path in source_paths}
    if discovered_relatives != set(expected_checksums):
        missing = sorted(set(expected_checksums) - discovered_relatives)
        unexpected = sorted(discovered_relatives - set(expected_checksums))
        raise ValueError(f"正文文件与 SHA256SUMS.txt 不一致: missing={missing}, unexpected={unexpected}")

    results: list[dict[str, object]] = []
    for source_path in source_paths:
        relative = source_path.relative_to(collection_root)
        relative_key = relative.as_posix()
        expected_sha256 = expected_checksums[relative_key]
        actual_sha256 = _sha256(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"原文 SHA-256 校验失败: {relative_key}")
        if actual_sha256 not in metadata_by_hash:
            raise ValueError(f"找不到官方来源元数据: {relative_key} ({actual_sha256})")

        base_metadata = metadata_by_hash[actual_sha256]
        document = replace(base_metadata.document, raw_file=relative_key, sha256=actual_sha256)
        metadata = replace(base_metadata, document=document)
        source = extract_source(source_path)
        parsed = parse_legal_structure(source, document.title)
        markdown, structure_mode = render_collection_markdown(metadata, source, parsed)

        output_path = output_root / relative.parent / f"{source_path.stem}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        result = _audit_output(
            source_path,
            output_path,
            source,
            parsed,
            markdown,
            expected_sha256,
            structure_mode,
        )
        result["source_relative_path"] = relative_key
        result["source_category"] = relative.parts[0]
        result["source_role"] = metadata.source_role
        result["parent_document_id"] = document.parent_document_id
        # 审计字段保留迁移前的虚拟前缀，入库程序会把它映射到独立的
        # processed 目录。这样旧报告和新报告都使用同一套读取逻辑。
        result["markdown_relative_path"] = (
            Path(OUTPUT_DIRECTORY_NAME) / relative.parent / f"{source_path.stem}.md"
        ).as_posix()
        results.append(result)
        print(
            f"已转换: {relative_key} -> {output_path.name} "
            f"(mode={structure_mode}, articles={len(parsed.articles)}, pages={source.page_count})"
        )

    warning_count = sum(len(result["warnings"]) for result in results)
    summary: dict[str, object] = {
        "collection_root": str(collection_root),
        "output_root": str(output_root),
        "converter": CONVERTER_NAME,
        "converter_version": CONVERTER_VERSION,
        "document_count": len(results),
        "markdown_count": len(results),
        "category_counts": {
            directory_name: sum(result["source_category"] == directory_name for result in results)
            for directory_name in SOURCE_CATEGORY_DIRECTORIES
        },
        "source_role_counts": {
            role: sum(result["source_role"] == role for result in results)
            for role in sorted({str(result["source_role"]) for result in results})
        },
        "article_document_count": sum(result["structure_mode"] == "legal_articles" for result in results),
        "general_document_count": sum(result["structure_mode"] == "general_sections" for result in results),
        "article_count": sum(int(result["articles"]) for result in results),
        "pdf_document_count": sum(result["source_format"] == "pdf" for result in results),
        "warning_count": warning_count,
        "documents": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / AUDIT_REPORT_NAME
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"转换完成: documents={summary['document_count']}, markdown={summary['markdown_count']}, "
        f"articles={summary['article_count']}, warnings={summary['warning_count']}"
    )
    if warning_count:
        raise ValueError(f"转换审计发现 {warning_count} 个警告，请检查 {report_path}")
    return summary


def main() -> None:
    """Parse the optional collection path and run the conversion command."""

    parser = argparse.ArgumentParser(description="将政府采购法律数据汇集转换为结构化 Markdown")
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=DEFAULT_COLLECTION_ROOT,
        help=f"法律数据汇集根目录（默认: {DEFAULT_COLLECTION_ROOT}）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"结构化 Markdown 输出目录（默认: {DEFAULT_OUTPUT_ROOT}）",
    )
    args = parser.parse_args()
    convert_collection(args.collection_root, args.output_root)


if __name__ == "__main__":
    main()
