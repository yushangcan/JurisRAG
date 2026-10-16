"""派生文件校验；刷新摘要前必须完成完整原文审计，需要安装入库可选依赖。"""

from __future__ import annotations

import argparse
import hashlib
import json

from offline_ingestion.corpus import load_corpus_entries, verify_file_hash
from shared.config import MARKDOWN_CHECKSUMS_PATH, PROJECT_ROOT


def verify_snapshots(*, refresh: bool = False) -> dict[str, int]:
    if refresh:
        from offline_ingestion.markdown_converter.audit_markdown import audit_markdown_corpus
        report = audit_markdown_corpus()
        if report["warning_count"] or report["unexpected_markdown_files"]:
            raise ValueError("原文与 Markdown 审计未通过，不能刷新摘要")
    entries, _ = load_corpus_entries()
    for entry in entries:
        verify_file_hash(entry.source_path, entry.document.sha256)
    if refresh:
        files = {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted((PROJECT_ROOT / "data/processed").rglob("*.md"))}
        MARKDOWN_CHECKSUMS_PATH.write_text(json.dumps({"version": 1, "files": files}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    from offline_ingestion.build_index import verify_markdown
    for entry in entries:
        verify_markdown(entry.markdown_path, str(entry.document.sha256))
    return {"verified_documents": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_snapshots(refresh=args.refresh), ensure_ascii=False))
