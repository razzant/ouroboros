"""Focused tests for the v6.54.3 runtime-reliability commit (Commit 1).

Covers: the file-API root-label hybrid (1.1), the safety parse-fix + owner-only
OUROBOROS_SAFETY_MODE guard set (1.2), the light-mode read-vs-write mention-scan
refinement + actionable path messages + attachment abs_path (1.3), the deadline
package (1.4), plan_task deadline scaling (1.5), and schedule_subagent slot
visibility (1.6).
"""

from __future__ import annotations

import pathlib
import queue

import pytest

from ouroboros import config as config_mod
from ouroboros import safety as safety_mod
from ouroboros.tools.registry import ToolContext
from tests.test_plan_task_tool import _record, _write
from tests.test_plan_task_tool import harness as _plan_task_harness

harness = _plan_task_harness  # the plan_task fixture (real ToolContext and plan record, fake review substrate)


# ---------------------------------------------------------------------------
# helpers


def _ctx(tmp_path: pathlib.Path, *, task_id: str = "t-v655", meta: dict | None = None) -> ToolContext:
    system = tmp_path / "system"
    data = tmp_path / "data"
    for p in (system, data):
        p.mkdir(exist_ok=True)
    return ToolContext(
        repo_dir=system,
        drive_root=data,
        task_id=task_id,
        task_metadata=dict(meta or {}),
        event_queue=queue.Queue(),
    )


# ---------------------------------------------------------------------------
# 1.2 — safety mode config: normalize / ratchet / owner-only merge-skip


# ---------------------------------------------------------------------------
# 1.2 — safety gate behavior per mode


# ---------------------------------------------------------------------------
# 1.2 — parse-failure classification


def test_classify_safety_parse_failure_classes():
    fn = safety_mod._classify_safety_parse_failure
    assert fn({"content": ""}, {"completion_tokens": 0}) == "empty"
    assert fn({"content": None}, None) == "empty"
    max_tokens = config_mod.get_safety_max_tokens()
    assert fn({"content": '{"status": "SA'}, {"completion_tokens": max_tokens}) == "truncated"
    assert fn({"content": "prose without json"}, {"completion_tokens": 10}) == "unparseable"


# ---------------------------------------------------------------------------
# 1.2 — self-lowering detectors (shell + browser JS)


# ---------------------------------------------------------------------------
# 1.2 — response_format is optional intent the model catalogue may report unsupported


# ---------------------------------------------------------------------------
# 1.3 — read-vs-write mention-scan refinement


# The commands name their paths with forward slashes (``as_posix``): a Windows path such as
# D:\a\_temp\...\b0\t\notes.txt inside a Python literal decodes \a, \b, \t and \n, so that
# command would not name the file at all, and the guard rightly lets it through.
def _guard(tmp_path, raw_cmd, *, writeish):
    from ouroboros.tools.shell_guards import runtime_data_guard_targets

    drive = tmp_path / "data"
    allowed = drive / "task_results" / "artifacts" / "t-v655"
    scratch = drive / "task_drives" / "t-v655"
    for p in (allowed, scratch):
        p.mkdir(parents=True, exist_ok=True)
    return runtime_data_guard_targets(
        raw_cmd,
        writeish=writeish,
        drive_root=drive,
        work_dir=scratch,
        allowed_roots=[scratch, allowed],
    )


def test_pure_read_of_cross_project_store_still_blocks(tmp_path):
    """Adversarial r1 #1: the read relaxation must mirror read_file(root=runtime_data)
    — a pure-read interpreter mention of another project's facts store
    (projects/<id>/...) blocks, matching the file API (no cross-project peeking)."""
    other = tmp_path / "data" / "projects" / "OTHER" / "knowledge" / "facts.md"
    cmd = f"python3 -c \"print(open('{other.as_posix()}').read())\""
    blocked = _guard(tmp_path, cmd, writeish=False)
    assert blocked and str(other) in blocked[0]


def test_opaque_write_primitives_do_not_slip_the_write_guard(tmp_path):
    """Adversarial r2 #1: opaque/unmodeled write-capable calls the write regex
    did not name — subprocess exec (rm/mv/dd), tarfile/zipfile extractall,
    shutil.unpack_archive, sqlite3.connect — must NOT be treated as pure reads.
    A write to a runtime_data path outside the task roots blocks (base behavior),
    while the AST models none of them (they fall to the conservative full scan)."""
    victim = tmp_path / "data" / "memory" / "identity.md"
    mem = tmp_path / "data" / "memory"
    db = tmp_path / "data" / "state" / "notes.db"
    for cmd in (
        f"python3 -c \"import subprocess; subprocess.run(['rm','-rf','{victim.as_posix()}'])\"",
        f"python3 -c \"import tarfile; tarfile.open('/tmp/x.tar').extractall('{mem.as_posix()}')\"",
        f"python3 -c \"import zipfile; zipfile.ZipFile('/tmp/x.zip').extractall('{mem.as_posix()}')\"",
        f"python3 -c \"import shutil; shutil.unpack_archive('/tmp/x.tar','{mem.as_posix()}')\"",
        f"python3 -c \"import os; os.system('rm -rf {victim.as_posix()}')\"",
        f"python3 -c \"import sqlite3; sqlite3.connect('{db.as_posix()}')\"",
    ):
        blocked = _guard(tmp_path, cmd, writeish=False)
        assert blocked, f"opaque write not blocked: {cmd}"


def test_python_write_targets_windows_shape_and_degenerate_unknown():
    """The mkdir/touch modeling of a windows-shaped literal must resolve the REAL
    parent (deterministic on every host via PureWindowsPath); and a derivation
    that still collapses to '.'/'' is UNKNOWN → the caller keeps the conservative
    full mention scan instead of trusting a cwd-shaped target."""
    import pathlib

    from ouroboros.tools.shell_guards import _python_write_targets_and_unknown

    win = "C:\\Users\\u\\AppData\\Temp\\home\\Ouroboros\\data\\uploads\\touch-report.html"
    code = (
        "from pathlib import Path\n"
        f"p = Path({win!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.touch()\n"
    )
    targets, unknown = _python_write_targets_and_unknown(code)
    assert str(pathlib.PureWindowsPath(win).parent) in targets

    degenerate = (
        "from pathlib import Path\n"
        "p = Path('relative.html')\n"
        "p.parent.mkdir(exist_ok=True)\n"
    )
    targets, unknown = _python_write_targets_and_unknown(degenerate)
    assert unknown and "." not in targets


# ---------------------------------------------------------------------------
# 1.3 — actionable messages carry REAL resolved paths


# ---------------------------------------------------------------------------
# 1.1 — resolve_user_file_path absolute-outside-home rejection


# ---------------------------------------------------------------------------
# 1.1 — dispatch root-label hybrid: reads auto-route, writes redirect


def test_dispatch_auto_routes_user_files_read_under_workspace(tmp_path):
    from ouroboros.tools.registry import _normalize_dispatch_path_args

    ctx = _ctx(tmp_path)
    ctx.task_metadata = {"work_kind": "owner"}
    target = tmp_path / "system" / "src" / "x.py"
    args = {"root": "user_files", "path": str(target)}
    note = _normalize_dispatch_path_args(ctx, "read_file", args)
    assert note.startswith("⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE")
    assert args["root"] == "active_workspace"


# ---------------------------------------------------------------------------
# 1.4 — deadline-clamped outer timeouts for network/long tools


# ---------------------------------------------------------------------------
# 1.5 — plan_task deadline scaling


def _deadline_ctx(harness, seconds):
    from datetime import datetime, timedelta, timezone

    ctx = harness.make_ctx()
    if seconds is not None:
        ctx.task_metadata["deadline_at"] = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
    return ctx


def _deadline_skip_events(events):
    drained = []
    while not events.empty():
        drained.append(events.get_nowait())
    return [e for e in drained if e.get("type") == "plan_task_deadline_skip"]


@pytest.mark.parametrize("seconds", [60, -5], ids=["inside-reserve", "expired"])
def test_plan_task_skips_only_a_spent_deadline(harness, monkeypatch, seconds):
    # A deadline spent down to its finalization reserve leaves no dispatch: the ask is the
    # paid act, so the whole write+ask call is refused before anything is written or sent.
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    substrate = harness.install({})
    text, result = _write(_deadline_ctx(harness, seconds), ask="council")
    assert "PLAN_ASK_SKIPPED_DEADLINE" in text and "PLAN_TASK_SKIPPED_DEADLINE" in text
    assert result.code == "CAPABILITY_UNAVAILABLE" and "Nothing was written" in text
    assert substrate.calls == []
    assert _record(harness)["revisions"] == [] and _record(harness)["asks"] == []
    [event] = _deadline_skip_events(harness.events)
    assert event["task_id"] == "task-1" and event["reserve_sec"] == 120.0
    assert "min_useful_sec" not in event and "scaled_ceiling_sec" not in event


# Retired pin (R4): 400 s left was refused as "a 100 s planning window under the 300 s useful
# floor" of the quarter-window wait that no longer exists; 1199 s and 600 s are council time.
@pytest.mark.parametrize("seconds", [None, 1_199, 400],
                         ids=["no-deadline", "1199s", "400s"])
def test_plan_task_no_deadline_does_not_skip(harness, monkeypatch, seconds):
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    substrate = harness.install({})
    text, result = _write(_deadline_ctx(harness, seconds), ask="council")
    assert result.status == "ok" and "SKIPPED_DEADLINE" not in text
    assert len(substrate.calls) == 1  # one ask, one dispatch
    record = _record(harness)
    assert [r["n"] for r in record["revisions"]] == [1] and [a["ask_id"] for a in record["asks"]] == ["a1"]
    assert _deadline_skip_events(harness.events) == []


# ---------------------------------------------------------------------------
# 1.6 — schedule_subagent reads pool counts, and does not decide from them


# ---------------------------------------------------------------------------
# review round 1 regressions


def test_list_files_hard_failure_is_first_class_error(tmp_path, monkeypatch):
    """Review round 3: an iterdir/permission failure inside a listing helper must
    surface as the first-class LIST_FILES_ERROR string, never ok-shaped JSON."""
    from ouroboros.tools import core_file_tools as core_mod

    ctx = _ctx(tmp_path)
    ctx.task_metadata = {"work_kind": "owner"}
    boom = tmp_path / "data" / "task_drives" / "t-v655"
    boom.mkdir(parents=True, exist_ok=True)

    def _explode(*a, **k):
        raise PermissionError("simulated iterdir failure")

    monkeypatch.setattr(core_mod, "_list_dir", _explode)
    out = core_mod._list_files(ctx, path=".", root="task_drive")
    assert out.startswith("⚠️ LIST_FILES_ERROR (PermissionError)")
