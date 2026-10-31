"""File/patch/script/media usability: bounded tree listing, directory reads, patch
ambiguity, run_script source custody, PDF page selection and per-call video frames."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import types

import pytest

from ouroboros.tools import core_file_tools, media
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.tool_result import LegacyTextResultAdapter


def _ctx(tmp_path: pathlib.Path) -> ToolContext:
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir(parents=True, exist_ok=True)
    drive.mkdir(parents=True, exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=drive)


def _tree(text: str) -> dict:
    assert not text.startswith("⚠️"), text
    return json.loads(text)


def _gap(listing: dict, kind: str) -> dict:
    return next(row for row in listing["gaps"] if row["kind"] == kind)


# ---------------------------------------------------------------------------
# list_files / read_file
# ---------------------------------------------------------------------------

def test_list_files_default_is_the_unchanged_single_level_listing(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.repo_dir / "pkg" / "deep").mkdir(parents=True)
    (ctx.repo_dir / "pkg" / "deep" / "x.py").write_text("x", encoding="utf-8")
    (ctx.repo_dir / "a.txt").write_text("a", encoding="utf-8")

    assert json.loads(core_file_tools._list_files(ctx, ".")) == ["a.txt", "pkg/"]
    assert core_file_tools._list_files(ctx, ".") == core_file_tools._list_files(ctx, ".", depth=1, include="")


def test_tree_walk_is_breadth_first_filtered_and_complete_when_nothing_is_hidden(tmp_path):
    ctx = _ctx(tmp_path)
    for rel in ("a.py", "b.txt", "pkg/c.py", "pkg/sub/d.py", "pkg/sub/deeper/e.py"):
        (ctx.repo_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (ctx.repo_dir / rel).write_text("x", encoding="utf-8")

    full = _tree(core_file_tools._list_files(ctx, ".", depth=3))
    assert full["entries"] == ["a.py", "b.txt", "pkg/", "pkg/c.py", "pkg/sub/", "pkg/sub/d.py", "pkg/sub/deeper/"]
    assert full["complete"] is False and full["scanned"] == 7
    assert _gap(full, "depth_limit")["paths"] == ["pkg/sub/deeper/"]

    matched = _tree(core_file_tools._list_files(ctx, "pkg", depth=5, include="*.{py,md}"))
    assert matched["entries"] == ["pkg/c.py", "pkg/sub/d.py", "pkg/sub/deeper/e.py"]
    assert matched["path"] == "active_workspace:pkg" and matched["include"] == "*.{py,md}"
    assert matched["complete"] is True and matched["gaps"] == []


def test_sparse_walk_bounds_inspected_entries_not_only_matches(tmp_path):
    ctx = _ctx(tmp_path)
    noise = ctx.repo_dir / "noise"
    noise.mkdir()
    for index in range(60):
        (noise / f"n{index:02d}.txt").write_text("", encoding="utf-8")
    (ctx.repo_dir / "zz").mkdir()
    (ctx.repo_dir / "zz" / "needle.py").write_text("", encoding="utf-8")

    capped = _tree(core_file_tools._list_files(ctx, ".", depth=3, include="*.py", max_scan=25))
    assert capped["entries"] == [] and capped["scanned"] == 25 and capped["complete"] is False
    stop = _gap(capped, "scan_cap")
    assert stop["detail"] == ("stopped at max_scan after examining 25 entries; the rest of noise/ was not read; "
                              "1 queued director(ies) were not opened")
    assert "not a snapshot" in capped["order"]

    found = _tree(core_file_tools._list_files(ctx, ".", depth=3, include="*.py", max_scan=200))
    assert found["entries"] == ["zz/needle.py"] and found["complete"] is True

    entry_cap = _tree(core_file_tools._list_files(ctx, "noise", depth=2, max_entries=5))
    assert len(entry_cap["entries"]) == 5 and entry_cap["complete"] is False
    assert "the rest of noise/ was not examined" in _gap(entry_cap, "entry_cap")["detail"]


def test_scan_cap_stops_filesystem_enumeration_itself(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    big = ctx.repo_dir / "big"
    big.mkdir()
    for index in range(500):
        (big / f"f{index:03d}.txt").write_text("", encoding="utf-8")
    real_scandir, pulled = core_file_tools.os.scandir, {}

    class CountingScandir:  # the real streaming iterator, counting every entry the filesystem yields
        def __init__(self, path):
            self._stream, self._key = real_scandir(path), str(path)
            pulled.setdefault(self._key, 0)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self._stream.close()

        def __iter__(self):
            return self

        def __next__(self):
            item = next(self._stream)
            pulled[self._key] += 1
            return item

    monkeypatch.setattr(core_file_tools.os, "scandir", CountingScandir)
    listing = _tree(core_file_tools._list_files(ctx, "big", depth=2, include="*.py", max_scan=10))
    assert listing["scanned"] == 10 and listing["complete"] is False
    assert pulled[str(big)] == 11  # the cap plus the one look-ahead entry; the other 489 are never enumerated
    assert "the rest of big/ was not read" in _gap(listing, "scan_cap")["detail"]

    pulled.clear()
    whole = _tree(core_file_tools._list_files(ctx, "big", depth=2, max_scan=500))
    assert whole["complete"] is True and len(whole["entries"]) == 500 and pulled[str(big)] == 500
    assert whole["entries"][:2] == ["big/f000.txt", "big/f001.txt"]  # the examined subset is sorted


def test_tree_names_symlinks_generated_and_protected_dirs_without_opening_them(tmp_path):
    ctx = _ctx(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("", encoding="utf-8")
    (ctx.repo_dir / "link").symlink_to(outside, target_is_directory=True)
    (ctx.repo_dir / "node_modules" / "lib").mkdir(parents=True)
    (ctx.repo_dir / "node_modules" / "lib" / "i.py").write_text("", encoding="utf-8")
    (ctx.repo_dir / "black_box").mkdir()
    (ctx.repo_dir / "black_box" / "inner.py").write_text("", encoding="utf-8")
    ctx.task_metadata = {"task_contract": {"resource_policy": {"protected_artifacts": [
        {"id": "ref", "role": "black_box_reference", "paths": ["./black_box"], "allow": ["execute"]}]}}}

    listing = _tree(core_file_tools._list_files(ctx, ".", depth=4))
    assert listing["entries"] == ["black_box/", "link", "node_modules/"]
    assert _gap(listing, "symlink_not_followed")["paths"] == ["link"]
    assert _gap(listing, "generated_not_descended")["paths"] == ["node_modules/"]
    assert _gap(listing, "protected_not_descended")["paths"] == ["black_box/"]
    assert listing["complete"] is False
    assert "secret.py" not in json.dumps(listing) and "inner.py" not in json.dumps(listing)
    # The protected directory itself keeps refusing a direct listing.
    assert "RESOURCE_POLICY_BLOCKED" in core_file_tools._list_files(ctx, "black_box", depth=2)


def test_tree_walk_refuses_a_target_escaping_its_root_and_keeps_misses_discovery(tmp_path):
    ctx = _ctx(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (ctx.repo_dir / "escape").symlink_to(outside, target_is_directory=True)
    (ctx.repo_dir / "file.txt").write_text("x", encoding="utf-8")

    escaped = core_file_tools._list_files(ctx, "escape", depth=2)
    assert escaped.startswith("⚠️") and "file.txt" not in escaped
    missing = core_file_tools._list_files(ctx, "nope", depth=2)
    assert missing.startswith("⚠️ LIST_FILES_NOT_FOUND: Directory not found")
    not_dir = core_file_tools._list_files(ctx, "file.txt", depth=2)
    assert not_dir.startswith("⚠️ LIST_FILES_NOT_FOUND: Not a directory")
    assert core_file_tools._list_files(ctx, ".", depth="deep").startswith("⚠️ TOOL_ARG_ERROR")


def test_runtime_data_tree_hides_the_project_store_and_counts_it(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.drive_root / "projects" / "p1").mkdir(parents=True)
    (ctx.drive_root / "projects" / "p1" / "facts.md").write_text("", encoding="utf-8")
    (ctx.drive_root / "logs").mkdir()
    (ctx.drive_root / "logs" / "a.jsonl").write_text("", encoding="utf-8")

    listing = _tree(core_file_tools._list_files(ctx, ".", root="runtime_data", depth=3))
    assert "logs/a.jsonl" in listing["entries"]
    assert not any(entry.startswith("projects") for entry in listing["entries"])
    assert _gap(listing, "policy_hidden")["count"] >= 1
    assert listing["complete"] is False


def test_read_file_on_an_admitted_directory_returns_the_listing_call(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.repo_dir / "src").mkdir()

    text = core_file_tools._read_file(ctx, "src")
    assert text.startswith("⚠️ NOT_FOUND: active_workspace:src is a directory, not a file.")
    assert f'list_files(root="active_workspace", path={json.dumps(str(ctx.repo_dir / "src"))})' in text
    assert LegacyTextResultAdapter.from_text("read_file", text).code == "LEGACY_WARNING"
    assert ctx.last_read_view is None


def test_skill_payload_directory_hint_keeps_the_skill_identity(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    payload = data / "skills" / "native" / "user-native"
    (payload / "scripts").mkdir(parents=True)
    (payload / "SKILL.md").write_text("# user-native\n", encoding="utf-8")
    (payload / "scripts" / "run.py").write_text("print(1)\n", encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    selector = {"root": "skill_payload", "bucket": "external", "skill_name": "user-native"}

    hint = registry.execute("read_file", {**selector, "path": "scripts"})
    assert hint.startswith("⚠️ NOT_FOUND: skill_payload:scripts is a directory, not a file.")
    assert f'list_files(root="skill_payload", path={json.dumps(str(payload / "scripts"))}, bucket="external", skill_name="user-native")' in hint
    # The recipe is usable as written: the same identity lists the directory.
    assert json.loads(registry.execute("list_files", {**selector, "path": str(payload / "scripts")})) == ["scripts/run.py"]


def test_read_file_directory_hint_never_precedes_the_read_policy(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.repo_dir / "black_box").mkdir()
    ctx.task_metadata = {"task_contract": {"resource_policy": {"protected_artifacts": [
        {"id": "ref", "role": "black_box_reference", "paths": ["./black_box"], "allow": ["execute"]}]}}}
    blocked = core_file_tools._read_file(ctx, "black_box")
    assert "RESOURCE_POLICY_BLOCKED" in blocked and "list_files(" not in blocked


# ---------------------------------------------------------------------------
# apply_patch ambiguity
# ---------------------------------------------------------------------------

_REPEATED = "def a():\n    x = 1\n\ndef b():\n    x = 1\n\ndef c():\n    x = 1\n"


def _patch(tmp_path, monkeypatch, patch: str):
    from ouroboros.tools.edit_ops import _apply_patch

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    ctx = _ctx(tmp_path)
    target = ctx.repo_dir / "s.py"
    target.write_text(_REPEATED, encoding="utf-8")
    return ctx, target, _apply_patch(ctx, patch)


def test_ambiguous_hunk_reports_every_placement_and_writes_nothing(tmp_path, monkeypatch):
    _ctx_, target, result = _patch(tmp_path, monkeypatch, "*** Update File: s.py\n-    x = 1\n+    x = 2\n")
    assert result.startswith("⚠️ APPLY_PATCH_ERROR: validation failed; NOTHING was written.")
    assert "context is ambiguous in s.py — 3 placements:" in result
    for line, owner in ((2, "a"), (5, "b"), (8, "c")):
        assert f"line {line}: before 'def {owner}():'" in result
    assert "lower bound (its FIRST hit)" in result
    assert target.read_text(encoding="utf-8") == _REPEATED


def test_anchor_lower_bound_narrows_but_cannot_select_a_middle_placement(tmp_path, monkeypatch):
    _ctx_, target, inside = _patch(tmp_path, monkeypatch, "*** Update File: s.py\n@@ def b\n-    x = 1\n+    x = 2\n")
    assert "at or after the @@ anchor's first hit (line 4) — 2 placements:" in inside
    assert "line 5: before 'def b():'" in inside and "line 8: before 'def c():'" in inside
    assert "line 2:" not in inside  # the placement before the anchor is outside its bound
    assert target.read_text(encoding="utf-8") == _REPEATED

    _ctx2, target2, last = _patch(tmp_path / "again", monkeypatch,
                                  "*** Update File: s.py\n@@ def c\n-    x = 1\n+    x = 3\n")
    assert last.startswith("✅ Updated s.py")
    assert target2.read_text(encoding="utf-8") == _REPEATED.rsplit("x = 1", 1)[0] + "x = 3\n"


def test_ambiguity_omissions_are_counted_and_named(tmp_path):
    from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch

    hunks = _parse_patch("*** Update File: s.py\n-x\n+y\n")[0][0].hunks
    new, _notes, error = _apply_hunks_to_text("x\n" * 40, hunks, "s.py")
    assert new is None and "40 placements:" in error
    assert "… 35 more not shown, starting at line(s) 6, 7," in error and "25 and 15 more" in error


def test_atomic_cumulative_chain_stays_atomic_when_a_later_hunk_is_ambiguous(tmp_path, monkeypatch):
    patch = ("*** Update File: s.py\n@@ def c\n-    x = 1\n+    x = 3\n"
             "*** Update File: s.py\n-    x = 1\n+    x = 9\n")
    _ctx_, target, result = _patch(tmp_path, monkeypatch, patch)
    assert "NOTHING was written" in result and "2 placements:" in result
    assert target.read_text(encoding="utf-8") == _REPEATED

    chained = ("*** Update File: s.py\n@@ def c\n-    x = 1\n+    x = 3\n"
               "*** Update File: s.py\n@@ def b\n-    x = 1\n+    x = 2\n")
    _ctx2, target2, ok = _patch(tmp_path / "chain", monkeypatch, chained)
    assert ok.startswith("✅ Updated s.py") and "Context after s.py:" in ok
    assert target2.read_text(encoding="utf-8") == "def a():\n    x = 1\n\ndef b():\n    x = 2\n\ndef c():\n    x = 3\n"


def test_ambiguous_pure_insertion_anchor_lists_its_placements():
    from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch

    hunks = _parse_patch("*** Update File: s.py\n@@ x = 1\n+    y = 0\n")[0][0].hunks
    new, _notes, error = _apply_hunks_to_text(_REPEATED, hunks, "s.py")
    assert new is None and "@@ anchor 'x = 1' is ambiguous in s.py — 3 placements:" in error
    assert "make the anchor unique" in error


# ---------------------------------------------------------------------------
# run_script source custody
# ---------------------------------------------------------------------------

@pytest.fixture
def script_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_id = "task1"
    return registry, repo, data


def _script_path(text: str) -> pathlib.Path:
    line = next(row for row in text.splitlines() if row.startswith("# script_path="))
    return pathlib.Path(line.removeprefix("# script_path=").split(" (temporary file removed", 1)[0])


def _dispatch(data: pathlib.Path, arguments: dict) -> dict:
    from ouroboros.tool_call_log import new_invocation, persist_dispatch_source

    invocation = new_invocation("call_1", {}, 1)
    persist_dispatch_source({"budget_drive_root": str(data)}, data / "logs", "task1", invocation, "run_script", arguments)
    return invocation


def test_run_script_cites_the_retained_redacted_dispatch_record_after_temp_removal(script_registry):
    from ouroboros.observability import read_call_payload
    from ouroboros.task_results import write_task_result
    from ouroboros.tool_call_log import invocation_scope

    registry, _repo, data = script_registry
    arguments = {"script": "print('kept-source-marker')"}
    invocation = _dispatch(data, arguments)
    assert invocation["args_source_status"] == "ready"
    with invocation_scope(invocation):
        result = registry.execute_result("run_script", arguments)

    ref = invocation["args_source_ref"]
    assert result.status == "ok" and "kept-source-marker" in result.text
    assert f"call_id={ref['call_id']!r}, manifest_sha256={ref['sha256']!r}" in result.text
    source_fact = result.meta["script_source"]
    assert source_fact["status"] == "ready"
    assert source_fact["call_id"] == ref["call_id"] and source_fact["sha256"] == ref["sha256"]
    recipe = source_fact["source_read"]
    assert recipe == {"tool": "get_task_result", "arguments": {
        "task_id": "task1",
        "call_id": ref["call_id"], "manifest_sha256": ref["sha256"],
    }}
    assert not _script_path(result.text).exists()
    _manifest, payload, verified = read_call_payload(data, task_id="task1", call_id=ref["call_id"])
    assert payload["arguments"]["script"] == arguments["script"] and verified["manifest_ref"] == ref

    # The public citation itself is sufficient after temporary-file deletion:
    # metadata first, then the advertised bounded character range. The task
    # result does not duplicate or separately publish the dispatch manifest.
    write_task_result(data, "task1", "completed", result="done")
    metadata_result = registry.execute_result(recipe["tool"], recipe["arguments"])
    metadata = json.loads(metadata_result.text)
    assert metadata_result.status == "ok"
    assert metadata["call_source"]["call_type"] == "tool_dispatch"
    assert "range_read" not in source_fact
    ranged_arguments = {**recipe["arguments"], "source_start_char": 0,
                        "source_end_char": metadata["call_source"]["complete_chars"]}
    recovered_result = registry.execute_result(recipe["tool"], ranged_arguments)
    recovered = json.loads(recovered_result.text)
    assert recovered_result.status == "ok"
    assert "kept-source-marker" in recovered["call_source"]["text"]
    # No second script store: the only copy is the existing dispatch record.
    assert not list(data.rglob("*.py"))


def test_run_script_nonzero_and_timeout_keep_their_own_first_line(script_registry):
    from ouroboros.tool_call_log import invocation_scope

    registry, _repo, data = script_registry
    failing = {"script": "import sys\nprint('partial')\nsys.exit(3)"}
    with invocation_scope(_dispatch(data, failing)):
        result = registry.execute_result("run_script", failing)
    assert result.text.startswith("⚠️ SHELL_EXIT_ERROR: command exited with exit_code=3")
    assert "partial" in result.text and "get_task_result(task_id='task1', call_id='tool_dispatch_" in result.text
    assert result.meta["exit_code"] == 3 and result.meta["script_source"]["status"] == "ready"

    slow = {"script": "import time\ntime.sleep(30)", "timeout_sec": 1}
    with invocation_scope(_dispatch(data, slow)):
        timed_out = registry.execute("run_script", slow)
    assert timed_out.startswith("⚠️ TOOL_TIMEOUT (run_command)")
    assert "get_task_result(task_id='task1', call_id='tool_dispatch_" in timed_out and not _script_path(timed_out).exists()


def test_run_script_capture_failure_never_erases_the_command_effect(script_registry, monkeypatch):
    from ouroboros.tool_call_log import invocation_scope

    registry, repo, _data = script_registry
    arguments = {"script": "from pathlib import Path\nPath('made.txt').write_text('effect')\nprint('done')"}
    with invocation_scope({"invocation_id": "x", "args_source_status": "unavailable:OSError"}):
        result = registry.execute_result("run_script", arguments)
    assert result.status == "ok" and "done" in result.text
    assert "retention is unavailable or unknown (unavailable:OSError)" in result.text
    assert result.meta["script_source"] == {"status": "unavailable:OSError"}
    assert (repo / "made.txt").read_text(encoding="utf-8") == "effect"

    def broken():
        raise RuntimeError("ledger offline")

    monkeypatch.setattr("ouroboros.tool_call_log.current_args_source", broken)
    again = registry.execute_result("run_script", {"script": "print('still here')"})
    assert (again.status, again.code) == (result.status, result.code) == ("ok", "OK") and "still here" in again.text
    assert "retention is unavailable or unknown (unavailable:RuntimeError)" in again.text


def test_run_script_preparation_failure_claims_no_script_path(script_registry, monkeypatch):
    from ouroboros.tools import shell

    registry, _repo, _data = script_registry

    def refuse(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(shell.tempfile, "mkdtemp", refuse)
    result = registry.execute_result("run_script", {"script": "print('never')"})
    assert result.text.startswith("⚠️ RUN_SCRIPT_BLOCKED: script preparation failed: disk full")
    assert "# script_path=" not in result.text and "never" not in result.text
    assert result.text.endswith(
        "# script_source: nothing ran; its text was not retained (no dispatch record for this call)"
    )
    assert result.meta.get("operation_outcome") == "completed_no_effect"
    assert result.meta["script_source"] == {"status": "no dispatch record for this call"}


def test_run_script_preparation_failure_cites_an_existing_dispatch_record(script_registry, monkeypatch):
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools import shell

    registry, repo, data = script_registry
    arguments = {"script": "print('never')"}
    invocation = _dispatch(data, arguments)

    def refuse(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(shell.tempfile, "mkdtemp", refuse)
    with invocation_scope(invocation):
        result = registry.execute_result("run_script", arguments)
    ref = invocation["args_source_ref"]
    assert result.text.startswith("⚠️ RUN_SCRIPT_BLOCKED: script preparation failed: disk full")
    assert f"call_id={ref['call_id']!r}, manifest_sha256={ref['sha256']!r}" in result.text
    assert "get_task_result(" in result.text and "8192-character" not in result.text
    assert "# script_path=" not in result.text and not (repo / ".ouroboros").exists()
    assert result.meta["operation_outcome"] == "completed_no_effect"
    assert result.meta["script_source"]["source_read"]["arguments"]["manifest_sha256"] == ref["sha256"]


# ---------------------------------------------------------------------------
# PDF text layer
# ---------------------------------------------------------------------------

def _write_pdf(path: pathlib.Path, page_texts: list[str]) -> pathlib.Path:
    """A minimal PDF: a page with text has a text layer; an empty page has none."""
    objects: list[bytes] = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in page_texts:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
                        f"<< /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>").encode())
        kids.append(len(objects))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def pdf(tmp_path, monkeypatch):
    pytest.importorskip("pypdf")
    path = _write_pdf(tmp_path / "doc.pdf", ["Alpha page", "", "Gamma page", "Delta page"])
    monkeypatch.setattr(media, "_resolve_local_file", lambda c, p, **k: (path, ""))
    return path


def test_pdf_page_ranges_deliver_selected_pages_and_name_every_omission(pdf):
    out = media._ocr_pdf(None, "doc.pdf", pages="2-3")
    assert out.startswith("PDF text layer of doc.pdf (embedded text, not OCR): page(s) 3 of 4 delivered.")
    assert "--- page 3 ---\nGamma page" in out and "Alpha" not in out and "Delta" not in out
    assert "- 1 page(s) 2: no text layer (scanned/image-only?)" in out
    assert "- 2 page(s) 1,4: not requested" in out

    capped = media._ocr_pdf(None, "doc.pdf", 1)  # the historical positional max_pages still works
    assert "--- page 1 ---\nAlpha page" in capped
    assert '- 3 page(s) 2-4: requested but not read (max_pages=1); continue with pages="2-4"' in capped

    tail = media._ocr_pdf(None, "doc.pdf", pages="3-, 9")
    assert "page(s) 3-4 of 4 delivered" in tail and "- page(s) 9: requested past the last page (4)" in tail
    huge = media._ocr_pdf(None, "doc.pdf", pages="1-999999999")  # bounded by the document, not the spec
    assert "page(s) 1,3-4 of 4 delivered" in huge and "- page(s) 5-999999999: requested past the last page (4)" in huge


def test_pdf_scanned_selection_and_bad_page_specs_are_typed(pdf):
    scanned = media._ocr_pdf(None, "doc.pdf", pages="2")
    assert scanned.startswith("⚠️ OCR_PDF_SCANNED_UNAVAILABLE: no extractable text layer on 1 page(s) 2 of 4")
    assert "does not OCR" in scanned and "- 3 page(s) 1,3-4: not requested" in scanned
    assert media._ocr_pdf(None, "doc.pdf", pages="7-").startswith("⚠️ TOOL_ARG_ERROR: pages 7 are past the last page (4).")
    assert media._ocr_pdf(None, "doc.pdf", pages="two").startswith("⚠️ TOOL_ARG_ERROR: pages must look like")
    assert media._ocr_pdf(None, "doc.pdf", pages="5-2").startswith("⚠️ TOOL_ARG_ERROR: page range '5-2'")


def test_pdf_whole_scanned_document_keeps_the_compatibility_marker(tmp_path, monkeypatch):
    pytest.importorskip("pypdf")
    path = _write_pdf(tmp_path / "scan.pdf", ["", ""])
    monkeypatch.setattr(media, "_resolve_local_file", lambda c, p, **k: (path, ""))
    out = media._ocr_pdf(None, "scan.pdf")
    assert out.startswith("⚠️ OCR_PDF_SCANNED_UNAVAILABLE: no extractable text layer on 2 page(s) 1-2 of 2")
    capped = media._ocr_pdf(None, "scan.pdf", max_pages=1)
    assert "on 1 page(s) 1 of 2" in capped
    assert '- 1 page(s) 2: requested but not read (max_pages=1); continue with pages="2"' in capped
    assert LegacyTextResultAdapter.from_text("extract_pdf_text", out).status == "unavailable"


class _Page:
    def __init__(self, text=None, error=None):
        self._text, self._error = text, error

    def extract_text(self):
        if self._error is not None:
            raise self._error
        return self._text


def _fake_pdf(tmp_path, monkeypatch, pages):
    path = tmp_path / "f.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(media, "_resolve_local_file", lambda c, p, **k: (path, ""))
    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=lambda *_a, **_k: types.SimpleNamespace(pages=pages)))


def test_pdf_extraction_failures_name_every_failing_page(tmp_path, monkeypatch):
    pages = [_Page("one"), *[_Page(error=KeyError("x")) for _ in range(12)], _Page(""), _Page(error=ValueError("y"))]
    _fake_pdf(tmp_path, monkeypatch, pages)
    out = media._ocr_pdf(None, "f.pdf")
    assert "page(s) 1 of 15 delivered" in out
    assert "- 12 page(s) 2-13: extraction failed (KeyError)" in out  # no silent cut at ten failures
    assert "- 1 page(s) 15: extraction failed (ValueError)" in out and "- 1 page(s) 14: no text layer" in out


def test_pdf_all_failed_prefix_never_claims_the_whole_selection(tmp_path, monkeypatch):
    _fake_pdf(tmp_path, monkeypatch, [_Page(error=KeyError("x")) for _ in range(15)])
    out = media._ocr_pdf(None, "f.pdf", max_pages=12, pages="2-15")
    assert out.startswith("⚠️ OCR_PDF_UNAVAILABLE: text extraction failed on each of the 12 page(s) 2-13 attempted (of 15)")
    assert "- 12 page(s) 2-13: extraction failed (KeyError)" in out
    assert '- 2 page(s) 14-15: requested but not read (max_pages=12); continue with pages="14-15"' in out
    assert "- 1 page(s) 1: not requested" in out


def test_pdf_char_cap_names_the_cut_page_and_the_unread_rest(pdf, monkeypatch):
    monkeypatch.setattr(media, "_OCR_PDF_MAX_CHARS", 12)
    out = media._ocr_pdf(None, "doc.pdf")
    assert "Characters [0:10) of 10 extracted characters.\n--- page 1 ---\nAlpha page" in out
    assert "Characters [0:2) of 10 extracted characters.\n--- page 3 ---\nGa\n\nGaps:" in out
    assert "- page 3: text cut at the 12-char cap; the rest of this page is not delivered" in out
    assert '- 1 page(s) 4: requested but not read (the 12-char cap); continue with pages="4"' in out


def test_extract_pdf_text_shares_the_ocr_pdf_handler_and_contract():
    entries = {entry.name: entry for entry in media.get_tools()}
    ocr, extract = entries["ocr_pdf"], entries["extract_pdf_text"]
    assert ocr.handler is extract.handler is media._ocr_pdf
    assert ocr.schema["parameters"] == extract.schema["parameters"]
    assert set(extract.schema["parameters"]["properties"]) == {"path", "pages", "max_pages", "start_char"}
    assert "not OCR" in ocr.schema["description"] and "not OCR" in extract.schema["description"]


# ---------------------------------------------------------------------------
# video frames
# ---------------------------------------------------------------------------

def test_repeated_frame_extraction_never_overwrites_the_first_set(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    out_root = tmp_path / "artifacts"
    monkeypatch.setattr(media, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(media, "_resolve_local_file", lambda c, p, **k: (video, ""))
    monkeypatch.setattr("ouroboros.tool_access.resource_root_path", lambda ctx, root: out_root)
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        target = pathlib.Path(cmd[-1])
        assert not target.exists() and "-n" in cmd and "-y" not in cmd
        target.write_bytes(f"call{len(calls)}".encode())
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_run)
    first = media._extract_video_frames(None, "clip.mp4", timestamps="0 1.5", max_frames=2)
    second = media._extract_video_frames(None, "clip.mp4", timestamps="0 1.5", max_frames=2)
    first_paths = [pathlib.Path(p) for p in first.splitlines()[1:]]
    second_paths = [pathlib.Path(p) for p in second.splitlines()[1:]]
    assert [p.name for p in first_paths] == ["frame_01_0s.jpg", "frame_02_1.5s.jpg"]
    assert first_paths[0].parent != second_paths[0].parent
    assert first_paths[0].parent.parent == second_paths[0].parent.parent == out_root / "video_frames"
    assert [p.read_bytes() for p in first_paths] == [b"call1", b"call2"]
    assert [p.read_bytes() for p in second_paths] == [b"call3", b"call4"]
    assert "earlier extractions are kept" in second

    bad = media._extract_video_frames(None, "clip.mp4", timestamps="nan -1")
    assert bad.startswith("⚠️ EXTRACT_VIDEO_FRAMES_UNAVAILABLE: invalid timestamp 'nan'")
    assert len(list((out_root / "video_frames").iterdir())) == 2  # a failed call leaves no empty directory


@pytest.mark.skipif(media._resolve_ffmpeg() is None, reason="no local ffmpeg")
def test_real_ffmpeg_frames_are_unique_per_call(tmp_path, monkeypatch):
    ffmpeg = media._resolve_ffmpeg()
    video = tmp_path / "gen.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=duration=2:size=64x64:rate=5", "-pix_fmt", "yuv420p", str(video)],
                   check=True, timeout=60)
    monkeypatch.setattr(media, "_resolve_local_file", lambda c, p, **k: (video, ""))
    monkeypatch.setattr("ouroboros.tool_access.resource_root_path", lambda ctx, root: tmp_path / "artifacts")
    first = media._extract_video_frames(None, "gen.mp4", timestamps="0.5")
    second = media._extract_video_frames(None, "gen.mp4", timestamps="0.5")
    one, two = pathlib.Path(first.splitlines()[1]), pathlib.Path(second.splitlines()[1])
    assert one != two and one.is_file() and two.is_file() and one.stat().st_size > 0


def test_pdf_oversized_single_page_has_lossless_continuation(tmp_path, monkeypatch):
    original = "abcdefghijklmnopqrstuvwxyz"
    _fake_pdf(tmp_path, monkeypatch, [_Page(original)])
    monkeypatch.setattr(media, "_OCR_PDF_MAX_CHARS", 10)
    parts = []
    for offset in (0, 10, 20):
        response = media._ocr_pdf(None, "f.pdf", pages="1", start_char=offset)
        parts.append(response.split("--- page 1 ---\n", 1)[1].split("\n\nGaps:", 1)[0])
        if offset < 20:
            assert f'continue with pages="1", start_char={offset + 10}' in response
    assert "".join(parts) == original
    assert "TOOL_ARG_ERROR" in media._ocr_pdf(None, "f.pdf", pages="1", start_char=100)
    _fake_pdf(tmp_path, monkeypatch, [_Page(original), _Page("another")])
    assert "exactly one selected page" in media._ocr_pdf(None, "f.pdf", pages="1-2", start_char=10)


def test_registered_tree_does_not_materialize_packed_index(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    folder = ctx.drive_root / "task_results/artifacts/testtask/source_handles/context_checkpoints"
    folder.mkdir(parents=True)
    (folder / "loose.txt").write_text("visible", encoding="utf-8")
    monkeypatch.setattr("ouroboros.evidence_packs.packed_handle_names_at", lambda *_a:
                        (_ for _ in ()).throw(AssertionError("unbounded index was read")))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    result = registry.execute_result("list_files", {
        "root": "runtime_data", "path": str(folder), "depth": 2, "max_scan": 10,
    })
    assert result.status == "ok", result.text
    body = json.loads(result.text)
    assert body["complete"] is False and body["scanned"] == 1
    assert _gap(body, "packed_index_not_scanned")["paths"]
    assert "depth=1" in _gap(body, "packed_index_not_scanned")["detail"]


def test_pdf_suffix_discloses_exact_range_and_preserves_extracted_whitespace(tmp_path, monkeypatch):
    text = "  alpha\n beta  "
    _fake_pdf(tmp_path, monkeypatch, [_Page(text)])
    full = media._ocr_pdf(None, "f.pdf", pages="1")
    assert full.split("--- page 1 ---\n", 1)[1] == text
    suffix = media._ocr_pdf(None, "f.pdf", pages="1", start_char=4)
    assert f"Characters [4:{len(text)}) of {len(text)} extracted characters." in suffix
    assert "Prefix [0:4) omitted" in suffix
    assert suffix.split("--- page 1 ---\n", 1)[1] == text[4:]
    end = media._ocr_pdf(None, "f.pdf", pages="1", start_char=len(text))
    assert "Empty end-of-page slice; no new characters delivered" in end
    assert f"[{len(text)}:{len(text)})" in end


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_failed_frame_removes_only_its_new_partial_file(tmp_path, monkeypatch, failure):
    import subprocess
    from ouroboros.tools import shell
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    frames = tmp_path / "artifacts"
    previous = frames / "video_frames" / "older" / "frame.jpg"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"old-good")
    monkeypatch.setattr(media, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(media, "_resolve_local_file", lambda *_a, **_k: (video, ""))
    monkeypatch.setattr("ouroboros.tool_access.resource_root_path", lambda *_a: frames)
    def fail(cmd, **kwargs):
        pathlib.Path(cmd[-1]).write_bytes(b"incomplete")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        return types.SimpleNamespace(returncode=1, stderr="failed")
    monkeypatch.setattr(shell, "_tracked_subprocess_run", fail)
    result = media._extract_video_frames(None, str(video), max_frames=1)
    assert result.startswith("⚠️ EXTRACT_VIDEO_FRAMES_UNAVAILABLE")
    assert list((frames / "video_frames").iterdir()) == [previous.parent]
    assert previous.read_bytes() == b"old-good"


def test_run_script_early_handler_refusal_keeps_dispatch_source(script_registry, monkeypatch):
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools import shell
    registry, _repo, data = script_registry
    arguments = {"script": "print('never')", "cwd": "bad"}
    def bad_binding(*_a, **_k):
        raise ValueError("fixture invalid cwd")
    monkeypatch.setattr(shell, "build_resolved_resource_binding", bad_binding)
    with invocation_scope(_dispatch(data, arguments)):
        result = shell._run_script(registry._ctx, **arguments)
    assert "nothing ran" in result and "tool_dispatch_" in result
    assert "# script_path=" not in result


@pytest.mark.parametrize("gap", ["depth_limit", "symlink_not_followed"])
def test_tree_with_any_gap_is_incomplete(tmp_path, gap):
    ctx = _ctx(tmp_path)
    folder = ctx.repo_dir / "nested"
    folder.mkdir()
    (folder / "file.py").write_text("x")
    if gap == "symlink_not_followed":
        (ctx.repo_dir / "link").symlink_to(folder, target_is_directory=True)
    result = _tree(core_file_tools._list_files(ctx, ".", depth=1 if gap == "depth_limit" else 3, include="*.py"))
    assert _gap(result, gap) and result["complete"] is False


def test_user_file_tree_descends_generated_directory_names(tmp_path, monkeypatch):
    from dataclasses import replace
    from ouroboros.tool_access import build_resolved_resource_binding
    ctx = _ctx(tmp_path)
    (ctx.repo_dir / "build" / "dist").mkdir(parents=True)
    (ctx.repo_dir / "build" / "dist" / "report.txt").write_text("user document")
    binding = build_resolved_resource_binding(ctx, root="active_workspace", operation="list", path=".")
    code = _tree(core_file_tools._tree_list(ctx, binding, 4, "", 500, 2000))
    assert _gap(code, "generated_not_descended") and not code["complete"]
    monkeypatch.setattr(core_file_tools, "user_files_path_block_reason", lambda *_a, **_kw: "")
    user = _tree(core_file_tools._tree_list(ctx, replace(binding, root="user_files"), 4, "", 500, 2000))
    assert user["complete"] and "build/dist/report.txt" in user["entries"] and not user["gaps"]
