"""Phase 3 (v6.39): durable swarm coordination + honest finalization.

Covers F3 (journal fail-loud on unknown kind), F2 (mirror task-tree coordination into the
durable project journal), and I-b (stat child artifact pointers as a ground-truth fact).
"""

from __future__ import annotations

import json
import os

import pytest

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


@pytest.mark.skipif(os.name == "nt", reason="a Windows name cannot end in a space")
def test_child_artifact_pointers_stat_the_exact_registered_path(tmp_path):
    """A registered path is the file's identity: padding is part of the name, so the stat reads that very file,
    never its trimmed sibling, which is another file of another size."""
    padded = tmp_path / "note.txt "
    padded.write_text("hello", encoding="utf-8")
    (tmp_path / "note.txt").write_text("the trimmed sibling is longer", encoding="utf-8")
    assert _child_artifact_pointers({"artifact_bundle": {"artifacts": [{"name": "kept", "abs_path": str(padded)}]}}) == [
        f"kept -> {padded} [✓ present, 5 bytes]"]


def test_child_artifact_pointers_keep_a_valid_unicode_name_as_it_is(tmp_path):
    present = tmp_path / "отчёт-雪.txt"
    present.write_text("данные", encoding="utf-8")
    pointers = _child_artifact_pointers({"artifact_bundle": {"artifacts": [{"name": present.name, "abs_path": str(present)}]}})
    assert pointers == [f"{present.name} -> {present} [✓ present, {len('данные'.encode())} bytes]"]


@pytest.mark.skipif(os.name != "posix", reason="native POSIX filename bytes")
def test_a_finished_handoff_names_a_registered_byte_artifact_by_an_exact_literal(tmp_path):
    """The parent reads a child's artifact pointer as UTF-8 text: a POSIX byte name is a JSON string literal there,
    distinct from a real name that spells the same escape. The marker is the stat of the registered path, whatever
    this volume answers for a name it cannot hold."""
    from ouroboros.task_status import format_subagent_absorption_message

    byte_name = b"report-\xff.txt".decode("utf-8", "surrogateescape")  # constructed: the fixture volume need not store it
    spelled = r"report-\udcff.txt"
    child = {"task_id": "c1", "parent_task_id": "P", "status": "completed", "role": "a", "result": "Report written.",
             "artifact_bundle": {"artifacts": [{"name": byte_name, "abs_path": str(tmp_path / byte_name)},
                                               {"name": spelled, "abs_path": str(tmp_path / spelled)}]}}
    handoff = format_subagent_absorption_message([child], parent_task_id="P")
    handoff.encode("utf-8")
    [line] = [row for row in handoff.splitlines() if row.startswith("artifacts: ")]
    decoder, body = json.JSONDecoder(), line.removeprefix("artifacts: ")
    shown_name, end = decoder.raw_decode(body)
    shown_path, end = decoder.raw_decode(body, end + len(" -> "))
    assert (shown_name, shown_path) == (byte_name, str(tmp_path / byte_name))
    assert f"; {spelled} -> {tmp_path / spelled} " in line


# ----------------------------------------------------- F3: journal fail-loud


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


# ------------------------------- Q8: typed off-registry work-location journal row

def _capture_journal(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(pj, "append_jsonl", lambda path, row: captured.append(row))
    monkeypatch.setattr(pj, "project_journal_path", lambda pid: tmp_path / "journal.jsonl")
    monkeypatch.setattr("ouroboros.projects_registry.touch_project", lambda *a, **k: None)
    return captured


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
