"""Deep review composes complete books and consumes revision-bound coverage."""

import hashlib
import json

import pytest

from ouroboros import deep_self_review as deep
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.reference_books import compose_book, load_reference_book
from tests.test_deep_review_slot import _native_row, _ScriptedLLM, _tool_call


def _corpus(root, *, newline="\n"):
    files = {
        "BIBLE.md": "# Constitution\n\nThe constitutional source.\n",
        "docs/CHECKLISTS.md": "# Checklists\n\n## Review\n\nCheck the actual contract.\n",
        "worker.py": "def work(): return True\n",
    }
    for book_id, entry in (("architecture", "docs/ARCHITECTURE.md"), ("development", "docs/DEVELOPMENT.md")):
        files[entry] = f"# {book_id.title()}\n\nPurpose of the {book_id} book.\n\n## Chapters\n\n- [Flow]({book_id}/flow.md)\n- [State]({book_id}/state.md)\n"
        for name in ("flow", "state"):
            files[f"docs/{book_id}/{name}.md"] = (
                f"# {name.title()}\n\nIntroduction to {book_id} {name}.\n\n## Contract\n\nExact {book_id} {name} contract body.\n")
    files = {rel: text.replace("\n", newline) for rel, text in files.items()}
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    return files


def _required(root, rel):
    raw = (root / rel).read_bytes()
    normalized = raw.decode().replace("\r\n", "\n").replace("\r", "\n")
    return {"root": "system_repo", "path": rel, "source_revision": hashlib.sha256(raw).hexdigest(),
            "complete_sha256": hashlib.sha256(normalized.encode()).hexdigest(), "complete_chars": len(normalized),
            "range_basis": "unicode_text_universal_newlines"}


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_packed_chaptered_books_are_complete_once_and_stable_before_atlas(tmp_path, monkeypatch, newline):
    monkeypatch.setattr(deep, "get_context_mode", lambda: "max")
    monkeypatch.setattr(deep, "_compute_graph_centrality", lambda *a: {})
    prefixes = []
    for name in ("first", "second"):
        repo, data = tmp_path / name, tmp_path / f"{name}-data"
        files = _corpus(repo, newline=newline)
        monkeypatch.setattr(deep, "_dulwich_tracked_paths", lambda *a: (list(files), []))
        monkeypatch.chdir(tmp_path)
        pack, stats = deep.build_review_pack(repo, data)
        for rel, text in files.items():
            if rel.startswith(("docs/architecture/", "docs/development/")):
                assert pack.count(text) == 1
        prefix = "\n".join(f"## Reference book: docs/{book_id.upper()}.md\n\n" + compose_book(load_reference_book(repo, book_id))
                           for book_id in ("architecture", "development"))
        assert pack.startswith(prefix)
        assert str(repo) not in prefix
        assert all(view["delivery"] == "full" for view in stats["context_manifest"]["reference_book_views"])
        prefixes.append(prefix)
    assert prefixes[0] == prefixes[1]


def test_low_packed_architecture_overview_does_not_reinline_omitted_chapters(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    files = _corpus(repo)
    monkeypatch.setattr(deep, "get_context_mode", lambda: "low")
    monkeypatch.setattr(deep, "_compute_graph_centrality", lambda *a: {})
    monkeypatch.setattr(deep, "_dulwich_tracked_paths", lambda *a: (list(files), []))
    pack, stats = deep.build_review_pack(repo, data)
    assert "Introduction to architecture flow." in pack
    assert "Exact architecture flow contract body." not in pack
    assert "Exact development flow contract body." in pack
    assert "docs/architecture/flow.md" in pack
    row = next(r for r in stats["context_manifest"]["coverage"] if r["path"] == "docs/architecture/flow.md")
    assert "overview only" in row["reason"]


def test_missing_declared_chapter_never_becomes_a_successful_partial_packed_book(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    files = _corpus(repo)
    (repo / "docs/architecture/state.md").unlink()
    monkeypatch.setattr(deep, "_dulwich_tracked_paths", lambda *a: (list(files), []))
    pack, stats = deep.build_review_pack(repo, tmp_path / "data")
    assert pack == "" and "book unavailable" in stats["skipped"][0]
    assert "state.md" in stats["skipped"][0]


def test_retrieving_book_navigation_uses_only_physical_chapter_addresses(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    _corpus(repo)
    # This legacy mapper must not be fed a composed book with invented entrypoint lines.
    # The deep-review packet reads the chapter-addressed view (context_layout.book_navigation);
    # the legacy monolith mapper is no longer imported here at all.
    assert not hasattr(deep, "generate_doc_nav_map")
    task, _facts = deep._retrieving_task(repo, data)
    assert "Source: `docs/architecture/flow.md`" in task
    assert "Source: `docs/development/state.md`" in task
    assert "Introduction to architecture state." in task
    assert "Exact architecture state contract body." not in task
    assert "file you open" in task
    assert task.index("Introduction to architecture flow.") < task.index("## Memory")


def test_exact_coverage_overrides_complete_legacy_lines_without_rebinding_current_file(tmp_path):
    files = _corpus(tmp_path)
    row = {**_required(tmp_path, "BIBLE.md"), "status": "incomplete", "covered_chars": 7,
           "missing_ranges": [[7, len(files["BIBLE.md"])]]}
    usage = {"native_read_coverage": {"status": "incomplete", "sources": [row]},
             "native_tool_receipts": [{"tool": "read_file", "outcome": "executed", "path": "BIBLE.md",
                                       "root": "system_repo", "start_line": 1, "end_line": 99, "total_lines": 99}]}
    (tmp_path / "BIBLE.md").write_text("Changed after the observed review source.\n")
    detail = deep._native_read_coverage(usage, tmp_path)["BIBLE.md"]
    assert detail["state"] == "partial" and detail["covered_chars"] == 7
    assert detail["source_revision"] == row["source_revision"]
    assert detail["evidence_basis"] == "source_ranges" and "covered_lines" not in detail
    assert deep._delivery_incomplete("native_tool_rounds", usage) == "required_source_coverage_incomplete"


def test_legacy_line_evidence_stays_explicit_and_unsent_reads_never_complete_it(tmp_path):
    receipt = {"tool": "read_file", "outcome": "executed", "path": "BIBLE.md", "root": "system_repo",
               "start_line": 1, "end_line": 5, "total_lines": 5}
    usage = {"native_tool_receipts": [receipt]}
    detail = deep._native_read_coverage(usage, tmp_path)["BIBLE.md"]
    assert detail["state"] == "read" and detail["evidence_basis"] == "legacy_lines"
    assert "source_revision" not in detail
    receipt["delivered"] = False
    assert deep._native_read_coverage(usage, tmp_path)["BIBLE.md"]["state"] == "missing"


def test_native_deep_review_reports_exact_chapter_gap_without_changing_the_finding(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    _corpus(repo)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    report = "Read: Constitution.\n\nCRITICAL: a real consumer violates the contract."
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "BIBLE.md"})]}, {"content": report}])
    required = [_required(repo, "BIBLE.md"), _required(repo, "docs/architecture/flow.md")]
    text, usage = deep.run_deep_self_review(repo, data, llm, lambda m: None,
        slot=_native_row(), task_id="book-review", required_sources=required)
    assert text.endswith(report)
    assert usage["deep_review_coverage_basis"] == "source_ranges"
    assert usage["native_incomplete"] == "required_source_coverage_incomplete"
    assert "docs/architecture/flow.md NOT read" in text
    history = json.loads(read_actor_source_bytes(data, "book-review", usage["native_history_source"]))
    assert history["required_sources"] == required
