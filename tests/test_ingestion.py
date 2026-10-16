"""入库失败不能破坏旧库；派生文件变更必须先通过审核。"""

import hashlib
import json

import pytest

from offline_ingestion import build_index as ingestion


@pytest.mark.parametrize("stage", ["validation", "writing"])
def test_failed_rebuild_keeps_previous_index(stage, tmp_path, monkeypatch):
    path = tmp_path / "knowledge.sqlite3"
    previous = b"previous-index-snapshot"
    path.write_bytes(previous)

    def fail(*args, **kwargs):
        raise ValueError("simulated ingestion failure")

    monkeypatch.setattr(ingestion, "verify_markdown" if stage == "validation" else "build_markdown_nodes", fail)
    with pytest.raises(ValueError, match="simulated"):
        ingestion.build_index(path=path)
    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.building-*")) == []


def test_modified_markdown_cannot_reuse_frozen_source_hash(tmp_path, monkeypatch):
    path = tmp_path / "law.md"
    original = "原始法条，不得擅自改变。"
    path.write_text(original + "采购人可以任意否决。", encoding="utf-8")
    checksums = tmp_path / "checksums.json"
    checksums.write_text(json.dumps({"files": {"law.md": hashlib.sha256(original.encode()).hexdigest()}}), encoding="utf-8")
    monkeypatch.setattr(ingestion, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ingestion, "MARKDOWN_CHECKSUMS_PATH", checksums)
    with pytest.raises(ValueError, match="Markdown 校验失败"):
        ingestion.verify_markdown(path, "unchanged-raw-sha")
