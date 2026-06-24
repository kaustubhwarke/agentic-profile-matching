"""Tests for resume loading (no embeddings/network required)."""

from __future__ import annotations

from profile_matching.rag.ingestion import load_resume_documents


def test_load_resume_documents_reads_text(tmp_settings) -> None:
    (tmp_settings.resume_dir / "jane_doe.txt").write_text(
        "Jane Doe\nSenior Engineer\nReact, TypeScript", encoding="utf-8"
    )
    (tmp_settings.resume_dir / "ignore.png").write_bytes(b"not a resume")

    docs = load_resume_documents(tmp_settings)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["candidate_id"] == "jane_doe"
    assert doc.metadata["name"] == "Jane Doe"
    assert "TypeScript" in doc.page_content


def test_empty_directory_returns_no_docs(tmp_settings) -> None:
    assert load_resume_documents(tmp_settings) == []
