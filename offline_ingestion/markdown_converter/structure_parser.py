"""Recognize Chinese legal chapters/articles and render structured Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass

from offline_ingestion.markdown_converter.source_extractor import ExtractedSource, SourceBlock


CHINESE_NUMBER = r"[一二三四五六七八九十百千万零〇两0-9]+"
CHAPTER_PATTERN = re.compile(rf"^(第\s*{CHINESE_NUMBER}\s*章)\s*(.*)$")
ARTICLE_PATTERN = re.compile(rf"^(第\s*{CHINESE_NUMBER}\s*条)\s*(.*)$")
ITEM_PATTERN = re.compile(r"^(?:[（(][一二三四五六七八九十0-9]+[）)]|[0-9]+[.、])")


@dataclass(frozen=True)
class LegalParagraph:
    """One logical paragraph and the PDF page that supplied its text.

    DOCX and HTML do not have stable source pages, so ``page_number`` is
    intentionally ``None`` for those formats instead of a guessed value.
    """

    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class LegalArticle:
    """One parsed legal article with page-aware body paragraphs."""

    chapter: str
    article_number: str
    paragraphs: list[LegalParagraph]
    page_number: int | None = None


@dataclass(frozen=True)
class ParsedLegalText:
    """The structural result shared by Markdown rendering and tests."""

    # The introductory notice, addressee and signature can also be useful for
    # citations.  Keeping ``LegalParagraph`` here means a PDF preamble retains
    # its real page instead of silently dropping that provenance.
    preamble: list[LegalParagraph]
    articles: list[LegalArticle]


def _compact_identifier(text: str) -> str:
    """Remove source-layout spaces inside a chapter/article identifier."""

    return "".join(text.split())


def _prepare_blocks(source: ExtractedSource, title: str) -> list[SourceBlock]:
    """Remove duplicate title lines and a DOCX table of contents.

    NPC Word files put ``目录`` and a list of chapters immediately before the
    real first chapter.  The final chapter line before the first article is the
    real chapter heading, so it is retained while the earlier list is skipped.
    """

    blocks = list(source.blocks)
    normalized_title = "".join(title.split())
    while blocks and "".join(blocks[0].text.split()) == normalized_title:
        blocks.pop(0)

    toc_index = next((i for i, block in enumerate(blocks) if "".join(block.text.split()) == "目录"), None)
    if toc_index is None:
        return blocks
    first_article = next((i for i in range(toc_index + 1, len(blocks)) if ARTICLE_PATTERN.match(blocks[i].text)), None)
    if first_article is None:
        return blocks
    last_chapter = next(
        (i for i in range(first_article - 1, toc_index, -1) if CHAPTER_PATTERN.match(blocks[i].text)),
        None,
    )
    keep_from = last_chapter if last_chapter is not None else first_article
    return blocks[:toc_index] + blocks[keep_from:]


def _append_pdf_continuation(paragraphs: list[LegalParagraph], block: SourceBlock) -> None:
    """Join PDF visual wraps without joining text across a page boundary.

    A sentence can continue onto the next PDF page.  We deliberately keep that
    continuation as a separate logical segment so the later chunker can assign
    an exact page to every chunk.
    """

    if not paragraphs:
        paragraphs.append(LegalParagraph(block.text, block.page_number))
        return
    previous = paragraphs[-1]
    starts_new = (
        block.page_number != previous.page_number
        or bool(ITEM_PATTERN.match(block.text))
        or previous.text.endswith(("。", "；", "！", "？", "："))
    )
    if starts_new:
        paragraphs.append(LegalParagraph(block.text, block.page_number))
    else:
        # Chinese PDF lines normally wrap without a semantic space.  A space is
        # retained only between two ASCII alphanumeric characters.
        separator = " " if previous.text[-1:].isascii() and block.text[:1].isascii() else ""
        paragraphs[-1] = LegalParagraph(previous.text + separator + block.text, previous.page_number)


def parse_legal_structure(source: ExtractedSource, title: str) -> ParsedLegalText:
    """Parse blocks into preamble plus articles with inherited chapter context."""

    preamble: list[LegalParagraph] = []
    articles: list[LegalArticle] = []
    current_chapter = ""
    current_number = ""
    current_page: int | None = None
    current_paragraphs: list[LegalParagraph] = []

    def finish_article() -> None:
        """Store the current mutable article, if one has started."""

        nonlocal current_number, current_page, current_paragraphs
        if current_number:
            articles.append(
                LegalArticle(
                    chapter=current_chapter,
                    article_number=current_number,
                    paragraphs=current_paragraphs,
                    page_number=current_page,
                )
            )
        current_number = ""
        current_page = None
        current_paragraphs = []

    for block in _prepare_blocks(source, title):
        chapter_match = CHAPTER_PATTERN.match(block.text)
        if chapter_match:
            finish_article()
            number = _compact_identifier(chapter_match.group(1))
            name = chapter_match.group(2).strip()
            current_chapter = f"{number} {name}".strip()
            continue

        article_match = ARTICLE_PATTERN.match(block.text)
        if article_match:
            finish_article()
            current_number = _compact_identifier(article_match.group(1))
            current_page = block.page_number
            body = article_match.group(2).strip()
            current_paragraphs = [LegalParagraph(body, block.page_number)] if body else []
            continue

        if current_number:
            if source.source_format == "pdf":
                _append_pdf_continuation(current_paragraphs, block)
            else:
                current_paragraphs.append(LegalParagraph(block.text))
        else:
            preamble.append(LegalParagraph(block.text, block.page_number))

    finish_article()
    return ParsedLegalText(preamble=preamble, articles=articles)
