"""Deterministically extract ordered text blocks from DOCX, HTML and PDF.

The module does *not* understand legal chapters or articles.  Its only job is
to turn three file formats into the same small data structure.  Legal structure
is added later by ``structure_parser.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
import pdfplumber
from pypdf import PdfReader


@dataclass(frozen=True)
class SourceBlock:
    """One source text block and the PDF page on which it starts, if known."""

    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class ExtractedSource:
    """Format-neutral extraction result consumed by the legal parser."""

    source_format: str
    blocks: list[SourceBlock]
    page_count: int | None = None


def _clean_text(text: str) -> str:
    """Normalize invisible spaces without flattening paragraph boundaries."""

    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _iter_docx_blocks(document: DocxDocument):
    """Yield paragraphs and tables in the order in which Word stores them.

    ``python-docx`` exposes ``document.paragraphs`` and ``document.tables`` as
    separate lists.  Walking the underlying body lets us preserve interleaving
    when a future legal document contains a table between two paragraphs.
    """

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def extract_docx(path: Path) -> ExtractedSource:
    """Read Word paragraphs and table rows while keeping document order."""

    document = Document(path)
    blocks: list[SourceBlock] = []
    for item in _iter_docx_blocks(document):
        if isinstance(item, Paragraph):
            text = _clean_text(item.text)
            if text:
                blocks.append(SourceBlock(text))
            continue

        for row in item.rows:
            # Markdown tables are not guessed here.  A tab-separated row keeps
            # cell order deterministically and remains readable in Markdown.
            cells = [_clean_text(cell.text) for cell in row.cells]
            if any(cells):
                blocks.append(SourceBlock(" | ".join(cells)))
    return ExtractedSource(source_format="docx", blocks=blocks)


HTML_CONTENT_SELECTORS = (
    "#UCAP-CONTENT",  # 中国政府网正文
    ".TRS_Editor",  # 中国政府采购网正文
    ".detailcont",  # 国家发展改革委规章页
    "#j-show-body",  # 湖南省公共资源交易中心政策正文
    ".main_con_zw",  # 同上：旧页面模板的正文 class
    "article",
    "main",
)

# 这些标签在浏览器中会开启新的内容块。遇到它们时才换段，
# 而 ``span``/``strong``/``a`` 等行内标签应继续留在同一段。
HTML_BLOCK_TAGS = {
    "address",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "section",
    "table",
    "tbody",
    "thead",
    "tfoot",
    "tr",
    "ul",
}


def _choose_html_container(soup: BeautifulSoup):
    """Choose the first known main-content container, then fall back to body."""

    for selector in HTML_CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node and len(_clean_text(node.get_text(" ", strip=True))) >= 200:
            return node
    return soup.body or soup


def _extract_html_blocks(container: Tag) -> list[SourceBlock]:
    """Turn one HTML正文容器 into readable blocks without splitting inline text.

    ``Tag.get_text("\\n")`` inserts a newline at *every* nested tag.  Official
    pages often wrap a year or document number in several ``span`` elements,
    which used to turn ``财库〔2024〕13号`` into six unrelated chunks.
    This small browser-like walker only flushes at real block boundaries and
    ``br`` tags.  Table rows are kept as one pipe-separated Markdown-friendly
    block so cell order is not lost.
    """

    blocks: list[SourceBlock] = []
    text_parts: list[str] = []

    def flush() -> None:
        """Store the currently accumulated visible text, if it is non-empty."""

        text = _clean_text("".join(text_parts))
        text_parts.clear()
        if text:
            blocks.append(SourceBlock(text))

    def walk(node) -> None:
        """Recursively visit text while treating inline and block tags differently."""

        if isinstance(node, NavigableString):
            text_parts.append(str(node))
            return
        if not isinstance(node, Tag):
            return
        if node.name == "br":
            flush()
            return
        if node.name == "tr":
            # A table row is one logical record.  Walking every cell as a block
            # would lose the relationship between its code, name and remarks.
            flush()
            cells = [_clean_text(cell.get_text("", strip=True)) for cell in node.find_all(["th", "td"], recursive=False)]
            if any(cells):
                blocks.append(SourceBlock(" | ".join(cells)))
            return

        is_block = node.name in HTML_BLOCK_TAGS
        if is_block:
            flush()
        for child in node.children:
            # ``tr`` already serializes all of its cells, so its descendants
            # must not be visited a second time by the outer table walker.
            walk(child)
        if is_block:
            flush()

    walk(container)
    flush()
    return blocks


def extract_html(path: Path) -> ExtractedSource:
    """Extract visible lines from an official web page's main content only."""

    # Passing bytes lets BeautifulSoup honor the page's own charset declaration.
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()
    container = _choose_html_container(soup)
    blocks = _extract_html_blocks(container)
    return ExtractedSource(source_format="html", blocks=blocks)


PDF_NOISE_PATTERNS = (
    re.compile(r"^[-—]?\s*\d+\s*[-—]?$"),
    re.compile(r"^国家发展和改革委员会(?:行政规范性文件|规章)$"),
    re.compile(r"^国家发展和改革委员会发布$"),
    re.compile(r"^[XＸ]$"),
)


def _is_pdf_noise(line: str) -> bool:
    """Return true for known standalone page numbers and repeated page headers."""

    return any(pattern.fullmatch(line) for pattern in PDF_NOISE_PATTERNS)


def _is_procurement_catalog(reader: PdfReader) -> bool:
    """Identify the official catalog whose正文 is a three-column table.

    Most legal PDFs are continuous prose and should continue through pypdf's
    line extractor.  The procurement catalog is different: pages 11-147 are
    ruled tables, so preserving only reading-order lines would lose which
    description belongs to which code.
    """

    first_pages = "".join((page.extract_text() or "") for page in reader.pages[:5])
    return "政府采购品目分类目录" in "".join(first_pages.split())


def _extract_catalog_rows(page) -> list[str]:
    """Return Markdown-friendly cells from one ruled catalog page, if present."""

    tables = page.extract_tables()
    if not tables:
        return []
    table = tables[0]
    if not table or len(table[0]) != 3:
        return []
    header = [re.sub(r"\s+", "", str(cell or "")) for cell in table[0]]
    if header != ["编码", "品目名称", "说明"]:
        return []

    rows: list[str] = []
    for raw_row in table:
        # pdfplumber keeps visual line wraps inside a cell as ``\n``.  A space
        # is enough in Markdown and retains every word without inventing rows.
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in raw_row]
        rows.append(" | ".join(cells))
    return rows


def extract_pdf(path: Path) -> ExtractedSource:
    """Extract embedded PDF text in page order; scanned PDFs are rejected.

    No OCR or image model is used.  Keeping the page number on every line makes
    later article citations traceable even after visual line wrapping is merged.
    """

    reader = PdfReader(path)
    blocks: list[SourceBlock] = []
    extracted_characters = 0
    is_catalog = _is_procurement_catalog(reader)
    # Only the catalog needs geometry-aware table extraction.  Keeping this
    # scope narrow avoids changing the reading order of ordinary legal PDFs.
    plumber_document = pdfplumber.open(path) if is_catalog else None
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            catalog_rows = _extract_catalog_rows(plumber_document.pages[page_number - 1]) if plumber_document else []
            if catalog_rows:
                for row in catalog_rows:
                    blocks.append(SourceBlock(row, page_number=page_number))
                    extracted_characters += len(row)
                continue

            for raw_line in (page.extract_text() or "").splitlines():
                line = _clean_text(raw_line)
                if line and not _is_pdf_noise(line):
                    blocks.append(SourceBlock(line, page_number=page_number))
                    extracted_characters += len(line)
    finally:
        if plumber_document is not None:
            plumber_document.close()
    if extracted_characters < 100:
        raise ValueError(f"PDF 几乎没有可提取文字，当前版本不支持扫描件/OCR: {path.name}")
    return ExtractedSource(source_format="pdf", blocks=blocks, page_count=len(reader.pages))


def extract_source(path: Path) -> ExtractedSource:
    """Dispatch one supported file to its deterministic extractor."""

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_docx(path)
    if suffix in {".html", ".htm"}:
        return extract_html(path)
    if suffix == ".pdf":
        return extract_pdf(path)
    raise ValueError(f"仅支持 DOCX、HTML 和可提取文本的 PDF: {path.suffix}")
