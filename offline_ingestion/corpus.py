"""读取已审计的法规语料清单；不加载数据库或模型。"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

from shared.config import (
    INCLUDE_LEGAL_COLLECTION,
    LEGAL_COLLECTION_AUDIT_PATH,
    LEGAL_COLLECTION_MARKDOWN_DIR,
    LEGAL_COLLECTION_SOURCE_DIR,
    MANIFEST_PATH,
    PROCESSED_MARKDOWN_DIR,
    RAW_LAWS_DIR,
)
from shared.models import SourceDocument
from offline_ingestion.markdown_metadata import read_front_matter


@dataclass(frozen=True)
class CorpusEntry:
    """一份可入库文档及其原文、Markdown和来源集合位置。"""

    document: SourceDocument
    source_path: Path
    markdown_path: Path
    origin: str


def load_manifest() -> list[SourceDocument]:
    """Load official source metadata before reading any raw file."""

    records = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["documents"]
    for record in records:
        record.setdefault(
            "source_role",
            "official_attachment" if record.get("attachment_url") else "official_full_text_page",
        )
    return [SourceDocument.from_manifest_record(record) for record in records]


def verify_file_hash(path: Path, expected_sha256: str | None) -> None:
    """Stop ingestion when a raw source no longer matches the manifest.

    Legal answers must be traceable to the reviewed source file.  Silently
    indexing a changed or corrupted download would break that evidence chain.
    """

    if not expected_sha256:
        raise ValueError(f"清单缺少 SHA-256: {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"文件哈希与清单不一致: {path.name}")


def load_primary_entries() -> list[CorpusEntry]:
    """把仓库内20份基线 manifest 转换成统一入库计划。"""

    entries: list[CorpusEntry] = []
    for document in load_manifest():
        source_path = RAW_LAWS_DIR / document.raw_file
        entries.append(
            CorpusEntry(
                document=document,
                source_path=source_path,
                markdown_path=PROCESSED_MARKDOWN_DIR / f"{source_path.stem}.md",
                origin="primary_manifest",
            )
        )
    return entries


def load_external_collection_entries() -> list[CorpusEntry]:
    """读取独立60份法律汇集，并把审计报告作为强制质量门。

    原始文件与派生 Markdown 分目录保存。报告、front matter 与原文件
    SHA-256 必须三方一致，任何一个文件被改动都会阻止入库。
    """

    if not INCLUDE_LEGAL_COLLECTION:
        return []
    if not LEGAL_COLLECTION_AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"缺少法律汇集审计报告: {LEGAL_COLLECTION_AUDIT_PATH}。"
            "可先运行 convert_legal_collection，或设置 INCLUDE_LEGAL_COLLECTION=false。"
        )
    report = json.loads(LEGAL_COLLECTION_AUDIT_PATH.read_text(encoding="utf-8"))
    if report.get("warning_count") != 0:
        raise ValueError("法律汇集转换审计存在警告，拒绝进入正式索引")
    records = report.get("documents", [])
    if report.get("document_count") != len(records) or report.get("markdown_count") != len(records):
        raise ValueError("法律汇集审计报告的文档数量不一致")

    entries: list[CorpusEntry] = []
    for result in records:
        source_path = LEGAL_COLLECTION_SOURCE_DIR / str(result["source_relative_path"])
        markdown_relative = Path(str(result["markdown_relative_path"]))
        # 迁移前的审计报告会记录“03_结构化Markdown/分类/文件.md”。
        # 迁移后 Markdown 单独放在 data/processed/legal_collection，所以
        # 去掉这一层虚拟前缀；也兼容以后直接记录分类相对路径的报告。
        if markdown_relative.parts and markdown_relative.parts[0] == "03_结构化Markdown":
            markdown_relative = Path(*markdown_relative.parts[1:])
        markdown_path = LEGAL_COLLECTION_MARKDOWN_DIR / markdown_relative
        if not source_path.exists() or not markdown_path.exists():
            raise FileNotFoundError(f"法律汇集文件缺失: source={source_path}, markdown={markdown_path}")
        front_matter = read_front_matter(markdown_path)
        report_hash = str(result["sha256"])
        front_hash = str(front_matter.get("raw_sha256") or "")
        if report_hash != front_hash:
            raise ValueError(f"审计报告与 Markdown SHA-256 不一致: {markdown_path.name}")
        expected_markdown_hash = str(result.get("markdown_sha256") or "")
        actual_markdown_hash = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
        if not expected_markdown_hash or expected_markdown_hash != actual_markdown_hash:
            raise ValueError(f"结构化 Markdown 哈希与审计报告不一致: {markdown_path.name}")
        if str(front_matter.get("raw_file")) != str(result["source_relative_path"]):
            raise ValueError(f"Markdown 原文路径与审计报告不一致: {markdown_path.name}")
        if str(front_matter.get("source_role")) != str(result.get("source_role")):
            raise ValueError(f"Markdown 来源角色与审计报告不一致: {markdown_path.name}")
        record = dict(front_matter)
        record["sha256"] = front_hash
        entries.append(
            CorpusEntry(
                document=SourceDocument.from_manifest_record(record),
                source_path=source_path,
                markdown_path=markdown_path,
                origin="verified_legal_collection",
            )
        )
    return entries


def load_corpus_entries() -> tuple[list[CorpusEntry], int]:
    """合并主语料与外部汇集，按稳定 ID 和原文哈希去重。

    同一 document_id 如果正文哈希或标题不同，说明两个清单发生冲突，必须
    人工核验；只有三者一致时才把外部副本作为重复项跳过。
    """

    primary = load_primary_entries()
    accepted = list(primary)
    by_id = {entry.document.document_id: entry for entry in primary}
    by_hash = {entry.document.sha256: entry for entry in primary if entry.document.sha256}
    skipped_duplicates = 0
    for entry in load_external_collection_entries():
        existing_id = by_id.get(entry.document.document_id)
        if existing_id is not None:
            same_hash = existing_id.document.sha256 == entry.document.sha256
            same_title = "".join(existing_id.document.title.split()) == "".join(entry.document.title.split())
            if not same_hash or not same_title:
                raise ValueError(f"同一 document_id 对应不同法规内容: {entry.document.document_id}")
            skipped_duplicates += 1
            continue
        if entry.document.sha256 and entry.document.sha256 in by_hash:
            skipped_duplicates += 1
            continue
        accepted.append(entry)
        by_id[entry.document.document_id] = entry
        if entry.document.sha256:
            by_hash[entry.document.sha256] = entry
    return accepted, skipped_duplicates
