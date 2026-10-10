"""Phase 3 (v6.39): durable swarm coordination + honest finalization.

Covers F3 (journal fail-loud on unknown kind), F2 (mirror task-tree coordination into the
durable project journal), and I-b (stat child artifact pointers as a ground-truth fact).
"""

from __future__ import annotations

import pathlib

import ouroboros.tools.project_journal as pj
from ouroboros.task_status import _artifact_stat_marker, _child_artifact_pointers


# ----------------------------------------------------------- I-b: artifact stat

def test_artifact_stat_marker(tmp_path):
    present = tmp_path / "out.txt"
    present.write_text("hello", encoding="utf-8")
    assert "✓ present" in _artifact_stat_marker(str(present))
    assert "5 bytes" in _artifact_stat_marker(str(present))
    assert _artifact_stat_marker(str(tmp_path / "nope.txt")) == "[⚠ MISSING]"
    # A relative pointer cannot be resolved here -> not falsely flagged missing.
    assert _artifact_stat_marker("rel/path.txt") == "[? unresolved path]"


def test_safe_name_bounds_long_dir_component():
    # I-a genesis naming: an arbitrary-length project name must not produce an
    # ENAMETOOLONG path component, yet two long names sharing a prefix stay distinct.
    from ouroboros.subagent_worktrees import _safe_name
    safe = _safe_name("x" * 500)
    assert len(safe) <= 64
    other = _safe_name("x" * 55 + "y" * 445)
    assert len(other) <= 64
    assert safe != other  # hash suffix keeps same-prefix long names unique
    # A short name is unchanged.
    assert _safe_name("cyber-racing") == "cyber-racing"


def test_provision_genesis_collision_counts_up(tmp_path):
    # I-a: two genesis projects sharing a display-name-derived dir must NOT clobber and the
    # second must count up (<name>_1), never FileExistsError.
    from ouroboros.subagent_worktrees import provision_genesis_project
    repo = tmp_path / "repo"
    repo.mkdir()
    projects = tmp_path / "projects"
    data = tmp_path / "data"
    data.mkdir()
    h1 = provision_genesis_project(repo_dir=repo, task_id="t1", projects_root=projects,
                                   data_dir=data, dir_name="myproj")
    h2 = provision_genesis_project(repo_dir=repo, task_id="t2", projects_root=projects,
                                   data_dir=data, dir_name="myproj")
    assert pathlib.Path(h1.path).name == "myproj"
    assert pathlib.Path(h2.path).name == "myproj_1"
    assert h1.path != h2.path
    # Binding identity stays the task_id even though the dir was named from dir_name.
    assert h1.task_id == "t1" and h2.task_id == "t2"


def test_artifact_stat_marker_relative_existing_is_unresolved(tmp_path, monkeypatch):
    # A relative pointer that HAPPENS to exist under the parent's cwd must still read as
    # unresolved (the absorbing parent's cwd is not the child's) — never a false ✓ present.
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert _artifact_stat_marker("README.md") == "[? unresolved path]"


def test_child_artifact_pointers_flags_missing(tmp_path):
    present = tmp_path / "deliverable.html"
    present.write_text("<html/>", encoding="utf-8")
    child = {"artifact_bundle": {"artifacts": [
        {"name": "page", "abs_path": str(present)},
        {"name": "ghost", "abs_path": str(tmp_path / "ghost.html")},
    ]}}
    pointers = _child_artifact_pointers(child)
    assert any("page" in p and "✓ present" in p for p in pointers)
    assert any("ghost" in p and "⚠ MISSING" in p for p in pointers)


# ----------------------------------------------------- F3: journal fail-loud

def test_journal_unknown_kind_warns_but_records(monkeypatch, caplog, tmp_path):
    captured = {}
    monkeypatch.setattr(pj, "append_jsonl", lambda path, row: captured.update(row))
    monkeypatch.setattr(pj, "project_journal_path", lambda pid: tmp_path / "journal.jsonl")
    # append_journal_milestone imports projects_registry.touch_project INTERNALLY, so the
    # real source must be patched (not pj.touch_project) to keep writes off live state.
    monkeypatch.setattr("ouroboros.projects_registry.touch_project", lambda *a, **k: None)
    with caplog.at_level("WARNING"):
        pj.append_journal_milestone("proj1", "weird_kind", "hello", task_id="t1")
    # Recorded (not lost) but coerced to note, and the unknown kind was surfaced loudly.
    assert captured.get("kind") == "note"
    assert captured.get("text") == "hello"
    assert "unknown kind" in caplog.text


def test_journal_empty_kind_defaults_to_note_quietly(monkeypatch, caplog, tmp_path):
    captured = {}
    monkeypatch.setattr(pj, "append_jsonl", lambda path, row: captured.update(row))
    monkeypatch.setattr(pj, "project_journal_path", lambda pid: tmp_path / "journal.jsonl")
    # append_journal_milestone imports projects_registry.touch_project INTERNALLY, so the
    # real source must be patched (not pj.touch_project) to keep writes off live state.
    monkeypatch.setattr("ouroboros.projects_registry.touch_project", lambda *a, **k: None)
    with caplog.at_level("WARNING"):
        pj.append_journal_milestone("proj1", "", "hi", task_id="t1")
    assert captured.get("kind") == "note"
    assert "unknown kind" not in caplog.text  # an omitted kind is a quiet default


# --------------------------------------------- F2: tree -> journal mirror

def test_mirror_only_high_signal_kinds(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        pj, "append_journal_milestone",
        lambda pid, kind, text, task_id="": recorded.append((kind, text)),
    )
    monkeypatch.setattr(pj, "sanitize_project_id", lambda p: "proj1")
    rows = [
        {"kind": "blocker", "text": "build broke", "role": "builder"},
        {"kind": "interface_contract", "text": "API v2", "role": "arch"},
        {"kind": "question", "text": "which db?", "task_id": "abcdef1234"},
        {"kind": "contract", "text": "shared frame", "role": "lead"},
        {"kind": "note", "text": "noise", "role": "x"},          # skipped
        {"kind": "fact", "text": "noise", "role": "x"},          # skipped
        {"kind": "milestone", "text": "noise", "role": "x"},     # skipped
        {"kind": "partial_finding", "text": "noise", "role": "x"},  # skipped
        {"kind": "blocker", "text": "", "role": "x"},            # empty text -> skipped
    ]
    monkeypatch.setattr("ouroboros.task_tree_ledger.tree_ledger_rows", lambda rid: rows)
    pj.mirror_tree_coordination_to_journal("proj1", "root-1", task_id="root-1")
    kinds = [k for k, _ in recorded]
    # blocker -> blocked; question/interface_contract/contract -> note. Noise excluded.
    assert kinds == ["blocked", "note", "note", "note"]
    assert all("[swarm " in text for _, text in recorded)


def test_mirror_noops_without_project_or_root(monkeypatch):
    called = []
    monkeypatch.setattr(pj, "append_journal_milestone", lambda *a, **k: called.append(1))
    pj.mirror_tree_coordination_to_journal("", "root-1")
    monkeypatch.setattr(pj, "sanitize_project_id", lambda p: "proj1")
    pj.mirror_tree_coordination_to_journal("proj1", "")
    assert not called


# ---------------------------------------- F2 wiring: gated on swarm root (no parent)

class _FakeEnv:
    def __init__(self, root):
        self.drive_root = root

    def drive_path(self, sub):
        p = self.drive_root / sub
        p.mkdir(parents=True, exist_ok=True)
        return p


class _FakeMemory:
    def load_identity(self):
        return "id"


class _FakeCtx:
    pending_restart_reason = None


def _run_emit(tmp_path, task, monkeypatch):
    import time
    import ouroboros.agent_task_pipeline as atp
    drive_root = tmp_path / "data"
    for sub in ("logs", "memory", "task_results"):
        (drive_root / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(atp, "_run_scratchpad_consolidation", lambda *a, **k: None)
    monkeypatch.setattr(atp, "_run_post_task_processing_async", lambda *a, **k: None)
    # Hermetic: isolate the live project-journal/state side effects (append + mirror both
    # write under the canonical DATA_DIR otherwise). We only assert the F2 gate here.
    monkeypatch.setattr(pj, "append_journal_milestone", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(pj, "mirror_tree_coordination_to_journal",
                        lambda pid, root_id, task_id="": calls.append((pid, root_id, task_id)))

    class _LLM:
        def chat(self, **k):
            return {"content": "s"}, {"cost": 0}

    atp.emit_task_results(
        env=_FakeEnv(drive_root), memory=_FakeMemory(), llm=_LLM(),
        pending_events=[], task=task, text="reply",
        usage={"cost": 0.0, "rounds": 1, "prompt_tokens": 1, "completion_tokens": 1},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=time.time() - 0.1, drive_logs=drive_root / "logs", ctx=_FakeCtx(),
    )
    return calls


def test_mirror_wired_for_root_project_task(tmp_path, monkeypatch):
    task = {"id": "root1", "type": "task", "chat_id": 1, "text": "hi", "project_id": "proj1"}
    calls = _run_emit(tmp_path, task, monkeypatch)
    assert calls and calls[0][0] == "proj1"  # mirrored once for the swarm root


def test_mirror_skipped_for_child_task(tmp_path, monkeypatch):
    task = {"id": "c1", "type": "task", "chat_id": 1, "text": "hi",
            "project_id": "proj1", "parent_task_id": "root1"}
    calls = _run_emit(tmp_path, task, monkeypatch)
    assert not calls  # a subagent/child does not re-mirror; the root absorbs the tree


# ------------------------------- Q8: typed off-registry work-location journal row

def _capture_journal(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(pj, "append_jsonl", lambda path, row: captured.append(row))
    monkeypatch.setattr(pj, "project_journal_path", lambda pid: tmp_path / "journal.jsonl")
    monkeypatch.setattr("ouroboros.projects_registry.touch_project", lambda *a, **k: None)
    return captured


def test_work_location_row_written_for_off_registry_tree(tmp_path, monkeypatch):
    """A finished root whose effective tree differs from the registry working_dir
    leaves ONE typed 'work lives at <path> @ <sha>' row — from facts the task
    record already holds (no git subprocess)."""
    captured = _capture_journal(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "ouroboros.projects_registry.get_project",
        lambda root, pid: {"working_dir": str(tmp_path / "registered")},
    )
    task = {
        "id": "r1",
        "workspace_root": str(tmp_path / "elsewhere"),
        "metadata": {"workspace_preflight": {"git": {"head": "abc123"}}},
    }

    pj._record_work_location("proj1", task)

    assert len(captured) == 1
    row = captured[0]
    assert row["type"] == "work_location"
    assert row["path"] == str(tmp_path / "elsewhere")
    assert row["sha"] == "abc123"
    assert row["registered_working_dir"] == str(tmp_path / "registered")
    assert "work lives at" in row["text"] and "@ abc123" in row["text"]
    assert row["task_id"] == "r1"


def test_work_location_row_written_without_sha_when_unknown(tmp_path, monkeypatch):
    captured = _capture_journal(monkeypatch, tmp_path)
    monkeypatch.setattr("ouroboros.projects_registry.get_project", lambda root, pid: None)
    task = {"id": "r2", "workspace_root": str(tmp_path / "tree")}

    pj._record_work_location("proj1", task)

    assert len(captured) == 1
    assert captured[0]["sha"] == ""
    assert "@" not in captured[0]["text"]  # path recorded without a fabricated sha


def test_work_location_row_skipped_when_tree_matches_registry(tmp_path, monkeypatch):
    captured = _capture_journal(monkeypatch, tmp_path)
    tree = tmp_path / "tree"
    monkeypatch.setattr(
        "ouroboros.projects_registry.get_project",
        lambda root, pid: {"working_dir": str(tree)},
    )

    pj._record_work_location("proj1", {"id": "r3", "workspace_root": str(tree)})
    pj._record_work_location("proj1", {"id": "r4"})  # no working tree at all

    assert captured == []
