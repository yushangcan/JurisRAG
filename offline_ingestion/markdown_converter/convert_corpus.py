"""CLI: convert every manifest source into structured Markdown.

Run this separately from indexing so a human can audit ``data/processed``
before any derived text enters Chroma.
"""

from __future__ import annotations

from offline_ingestion.build_index import load_manifest, verify_file_hash
from offline_ingestion.markdown_converter.converter import convert_document, markdown_filename
from shared.config import PROCESSED_MARKDOWN_DIR, RAW_LAWS_DIR


def convert_corpus() -> dict[str, object]:
    """Verify raw hashes, convert all documents, and return conversion counts."""

    documents = load_manifest()
    results: list[dict[str, object]] = []
    for document in documents:
        source_path = RAW_LAWS_DIR / document.raw_file
        verify_file_hash(source_path, document.sha256)
        output_path = PROCESSED_MARKDOWN_DIR / markdown_filename(document)
        result = convert_document(document, source_path, output_path)
        results.append(result)
        print(f"已转换: {document.title} -> {output_path.name} ({result['articles']} 条)")
    summary = {
        "document_count": len(results),
        "article_count": sum(int(item["articles"]) for item in results),
        "documents": results,
    }
    print(f"Markdown 转换完成: documents={summary['document_count']}, articles={summary['article_count']}")
    return summary


if __name__ == "__main__":
    convert_corpus()
