"""Security and behaviour tests for the sandboxed file-system tools."""

from __future__ import annotations

from profile_matching.tools.filesystem import list_files, read_file, write_file


def test_read_file_within_sandbox(tmp_settings) -> None:
    (tmp_settings.resume_dir / "cv.txt").write_text("hello world", encoding="utf-8")
    assert read_file.invoke({"path": "cv.txt"}) == "hello world"


def test_write_file_within_sandbox(tmp_settings) -> None:
    result = write_file.invoke({"path": "report.md", "content": "# Report"})
    assert "Wrote" in result
    assert (tmp_settings.report_dir / "report.md").read_text(encoding="utf-8") == "# Report"


def test_path_traversal_is_rejected(tmp_settings) -> None:
    result = read_file.invoke({"path": "../../etc/passwd"})
    assert result.startswith("Error")
    assert "Access denied" in result


def test_absolute_escape_is_rejected(tmp_settings) -> None:
    result = read_file.invoke({"path": "/etc/hostname"})
    assert result.startswith("Error")


def test_list_files_shows_roots(tmp_settings) -> None:
    (tmp_settings.resume_dir / "a.txt").write_text("x", encoding="utf-8")
    listing = list_files.invoke({"directory": ""})
    assert "resumes/" in listing
    assert "a.txt" in listing
