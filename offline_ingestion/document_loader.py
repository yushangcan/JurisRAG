"""Read supported source files into plain text for the raw-corpus audit.

The vector index no longer consumes these flattened strings.  Production of
structured Markdown lives in ``offline_ingestion.markdown_converter``; this
module remains useful for quick raw-file integrity checks and early lessons.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader


def read_docx(path: Path) -> str:
    """Read paragraph text from a Word document in its original order."""

    document = Document(path)
    return "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())


def read_html(path: Path) -> str:
    """Remove HTML tags and return visible text with readable line breaks."""

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def read_pdf(path: Path) -> str:
    """Extract embedded text from every PDF page in page order.

    This baseline supports text PDFs, such as official ministry documents.
    A scanned PDF may return almost no text; the corpus audit must flag that
    case instead of silently indexing an empty document.
    """

    reader = PdfReader(path)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text.strip() for text in page_texts if text.strip())


def read_source_file(path: Path) -> str:
    """Dispatch to the correct parser based on the file extension."""

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix in {".html", ".htm"}:
        return read_html(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"暂不支持的文件类型: {path.suffix}")
