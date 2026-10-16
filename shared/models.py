"""Small, explicit data structures shared by the two application halves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SourceDocument:
    """Metadata from one official source record in ``manifest.json``."""

    document_id: str
    title: str
    category: str
    raw_file: str
    source_portal: str
    source_url: str
    attachment_url: str | None = None
    source_role: str | None = None
    verification_url: str | None = None
    source_page_published_at: str | None = None
    # 配套附件不是一部独立法规；这个字段把附件连接回发布它的通知。
    # 保留连接关系后，检索结果可以同时展示“附件正文”和“母文件通知”。
    parent_document_id: str | None = None
    notice_raw_file: str | None = None
    promulgated_date: str | None = None
    effective_date: str | None = None
    document_number: str | None = None
    legal_status: str | None = None
    status_verified_at: str | None = None
    revision_note: str | None = None
    sha256: str | None = None

    @classmethod
    def from_manifest_record(cls, record: dict[str, Any]) -> "SourceDocument":
        """Convert a manifest dictionary and tolerate optional fields."""

        return cls(
            document_id=record.get("document_id", record.get("raw_file", "unknown")),
            title=record.get("title", record.get("raw_file", "未命名法规")),
            category=record.get("category", "未分类"),
            raw_file=record["raw_file"],
            source_portal=record.get("source_portal", "未知来源平台"),
            source_url=record.get("source_url", ""),
            attachment_url=record.get("attachment_url"),
            source_role=record.get("source_role"),
            verification_url=record.get("verification_url"),
            source_page_published_at=record.get("source_page_published_at"),
            parent_document_id=record.get("parent_document_id"),
            notice_raw_file=record.get("notice_raw_file"),
            promulgated_date=record.get("promulgated_date", record.get("source_page_published_at")),
            effective_date=record.get("effective_date"),
            document_number=record.get("document_number"),
            legal_status=record.get("legal_status"),
            status_verified_at=record.get("status_verified_at"),
            revision_note=record.get("revision_note"),
            sha256=record.get("sha256"),
        )

    def to_metadata(self) -> dict[str, str]:
        """将可选值统一成字符串，保持来源和派生文本的元数据格式一致。"""

        values = asdict(self)
        return {key: str(value or "") for key, value in values.items()}


@dataclass
class TextChunk:
    """One article-aware passage and the source metadata needed for citations.

    Most articles produce one chunk.  A long article or a PDF article crossing
    a page boundary produces multiple child chunks.  They all keep the same
    ``parent_article_id`` so citations can regroup them as one legal article.
    """

    chunk_id: str
    text: str
    chunk_index: int
    document: SourceDocument
    markdown_file: str = ""
    # ``node_type`` 用来区分参与检索的 child 与只负责回填全文的 parent。
    # 同表保留角色，父块用于回填，子块用于检索。
    node_type: str = "child"
    chapter: str = ""
    article_number: str = ""
    # ``structure_type`` 说明这个切片来自法条、普通章节还是正文前言。
    # 普通通知和合同没有“第X条”，不能再把这些字段伪装成法条。
    structure_type: str = ""
    section_title: str = ""
    section_path: str = ""
    parent_article_id: str = ""
    parent_section_id: str = ""
    child_index: int = 0
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    page_numbers: str = ""
    source_format: str = ""

    def to_metadata(self) -> dict[str, str]:
        metadata = self.document.to_metadata()
        metadata.update(
            {
                "chunk_id": self.chunk_id,
                "chunk_index": str(self.chunk_index),
                "markdown_file": self.markdown_file,
                "node_type": self.node_type,
                "chapter": self.chapter,
                "article_number": self.article_number,
                "structure_type": self.structure_type,
                "section_title": self.section_title,
                "section_path": self.section_path,
                "parent_article_id": self.parent_article_id,
                "parent_section_id": self.parent_section_id,
                "child_index": str(self.child_index),
                "page_number": str(self.page_number or ""),
                "page_start": str(self.page_start or ""),
                "page_end": str(self.page_end or ""),
                "page_numbers": self.page_numbers,
                "source_format": self.source_format,
            }
        )
        return metadata
