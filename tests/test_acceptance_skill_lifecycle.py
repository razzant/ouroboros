"""The per-skill lifecycle facts of a task's related skills (visibility only).

A task that authored or repaired a skill used to leave no way to see whether
that skill actually reached a reviewed, ready, enabled state. These facts close
that gap; the task's completion observations carry them to post-task synthesis.
They are facts, never a gate on the execution route.
"""

from __future__ import annotations

import json

import pathlib

from ouroboros.skill_loader import find_skill
from ouroboros.skill_readiness import (
    _skill_names_from_review_history,
    acceptance_skill_lifecycle,
    skill_names_touched_by_trace,
    skill_readiness_for_execution,
)


def _write_skill(drive_root: pathlib.Path, name: str) -> pathlib.Path:
    skill_dir = drive_root / "skills" / "external" / name
    skill_dir.mkdir(parents=True)
    (drive_root / "state" / "skills" / name).mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ntype: instruction\nversion: 0.1.0\n---\nbody\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_record(drive_root: pathlib.Path, name: str, root_task_id: str) -> str:
    """One settled ``surface="skill"`` ledger record of ``name`` under ``root_task_id`` (the wave merges the
    frozen subject's own identity before writing, as ``review_change._finish_record`` does)."""
    from ouroboros import review_ledger
    from tests.test_review_ledger import _answer, _facts, _raw

    rows = [_raw("s1", "openai/gpt-5", parts=("skill",), answers={"skill": _answer("skill")})]
    subject = {"root_kind": "skill_payload", "root": f"/skills/external/{name}", "kind": "skill", "base": "", "head": "",
               "tree_sha": "h" * 64, "diff_sha": "d" * 64, "checkout": ""}
    record = review_ledger.build_wave_record(_facts(rows, task_id=root_task_id, subject=subject), surface="skill",
                                             drive_root=drive_root)
    record.subject = {**dict(record.subject), "skill": name, "content_hash": "h" * 64}
    review_ledger.write_record(drive_root, record)
    return record.record_id


def test_the_lifecycle_scan_joins_trace_edits_and_this_root_task_history(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_skill(drive_root, "edited")
    _write_skill(drive_root, "child_authored")
    _write_skill(drive_root, "someone_elses")
    _write_record(drive_root, "child_authored", "root-1")
    _write_record(drive_root, "someone_elses", "root-other")

    trace = {"tool_calls": [
        {"tool": "write_file", "args": {"path": "data/skills/external/edited/SKILL.md"}},
    ]}
    rows = acceptance_skill_lifecycle(drive_root, trace, "root-1")
    names = [row["name"] for row in rows]
    assert names == ["edited", "child_authored"]
    assert "someone_elses" not in names


def test_the_lifecycle_facts_match_the_readiness_predicate(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_skill(drive_root, "alpha")
    trace = {"tool_calls": [
        {"tool": "write_file", "args": {"bucket": "external", "skill_name": "alpha",
                                        "path": "SKILL.md"}},
    ]}
    row = acceptance_skill_lifecycle(drive_root, trace, "")[0]
    skill = find_skill(drive_root, "alpha")
    readiness = skill_readiness_for_execution(drive_root, skill)

    assert row["name"] == "alpha"
    assert row["source"] == skill.source
    assert row["review_status"] == skill.review.status
    assert row["ready"] is readiness.ready
    assert row["blockers"] == readiness.blockers
    assert row["enabled"] is bool(skill.enabled)
    # An unreviewed, disabled self-authored skill is exactly what must stay
    # visible, so the row must be honest rather than empty.
    assert row["ready"] is False
    assert row["blockers"]


def test_a_skill_lifecycle_tool_call_names_the_skill_without_a_payload_edit(tmp_path):
    """A free delegation lane integrates a patch and never calls write_file, so
    the lifecycle tools are the only carrier of the name in that shape."""
    trace = {"tool_calls": [
        {"tool": "skill_review", "args": {"skill": "delegated"}},
        {"tool": "skill_preflight", "args": {"skill": "probed"}},
        {"tool": "skill_exec", "args": {"skill": "executed", "script": "scripts/run.py"}},
        {"tool": "toggle_skill", "args": {"skill": "toggled", "enabled": True}},
        {"tool": "submit_skill_to_hub", "args": {
            "skill": "published", "confirm_public_submission": True,
        }},
        {"tool": "edit_text", "args": {
            "root": "skill_payload", "bucket": "user_repo",
            "skill_name": "user-repo-skill", "path": "SKILL.md",
            "old_text": "old", "new_text": "new",
        }},
        {"tool": "run_command", "args": {"cmd": "ls"}},
        {"tool": "run_command", "args": {
            "cmd": ["python3", "repair.py"], "cwd": "skill_payload/scripts",
            "bucket": "external", "skill_name": "command-edited",
        }},
        {"tool": "run_script", "args": {
            "script": "repair()", "cwd": "skill_payload",
            "bucket": "clawhub", "skill_name": "script-edited",
        }},
        {"tool": "delegate_start", "args": {
            "prompt": "repair", "root": "skill_payload",
            "bucket": "user_repo", "skill_name": "delegate-edited",
        }},
        {"tool": "delegate_start", "args": {
            "prompt": "ordinary workspace", "skill_name": "not-selected",
        }},
    ]}
    assert skill_names_touched_by_trace(trace) == [
        "delegated", "probed", "executed", "toggled", "published",
        "user-repo-skill", "command-edited", "script-edited", "delegate-edited",
    ]


def test_real_skill_payload_selectors_feed_lifecycle_rows(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    for name in ("command-edited", "script-edited", "delegate-edited"):
        _write_skill(drive_root, name)
    trace = {"tool_calls": [
        {"tool": "run_command", "args": {
            "cmd": ["python3", "repair.py"], "cwd": "skill_payload/scripts",
            "bucket": "external", "skill_name": "command-edited",
        }},
        {"tool": "run_script", "args": {
            "script": "repair()", "cwd": "skill_payload",
            "bucket": "external", "skill_name": "script-edited",
        }},
        {"tool": "delegate_start", "args": {
            "prompt": "repair", "root": "skill_payload",
            "bucket": "external", "skill_name": "delegate-edited",
        }},
    ]}

    assert [row["name"] for row in acceptance_skill_lifecycle(
        drive_root, trace, "root-1",
    )] == ["command-edited", "script-edited", "delegate-edited"]


def test_no_touched_skill_yields_no_lifecycle_rows(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    assert acceptance_skill_lifecycle(drive_root, {"tool_calls": []}, "") == []
    assert acceptance_skill_lifecycle(None, {"tool_calls": []}, "") == []


def test_bounded_history_projection_discloses_both_omission_limits(tmp_path):
    """The root task's skills come from the ledger's HOT index alone (one bounded read); rows of other roots are
    skipped, and rotated index segments the hot read never opens are disclosed as truncation."""
    from ouroboros import review_ledger

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_skill(drive_root, "visible")
    for index in range(3):
        _write_record(drive_root, f"skill-{index}", "other")
    _write_record(drive_root, "visible", "root-1")

    history = _skill_names_from_review_history(drive_root, "root-1")
    assert history["names"] == ["visible"]
    assert history["coverage"]["rows_scanned"] == 1 and history["coverage"]["truncated"] is False  # the root's own rows
    assert history["coverage"]["complete"] is True and history["coverage"]["gap_reasons"] == []
    assert history["coverage"]["source_ref"] == {
        "kind": "review_ledger_index", "path": "state/review_ledger/index.jsonl", "reader": "read_file",
    }

    # An archived segment beside the hot index: older records live where the bounded read never looks — a cut
    # for a task whose start is unknown or precedes the index's newest rotation (opus R1-8 / R2-3, subagent 2);
    # a task that started after that rotation has every row of its own in the hot index, and says so. The
    # rotation's time is the segment's name: a row carried into the fresh hot index (its record no longer
    # resolves) can be far older than the rows the rotation archived, so the hot index's first row says nothing.
    (review_ledger.ledger_dir(drive_root) / "index.20260101T000000.jsonl").write_text("", encoding="utf-8")
    assert review_ledger.newest_rotation_ts(drive_root) == "2026-01-01T00:00:00+00:00"
    history = _skill_names_from_review_history(drive_root, "root-1")
    assert history["names"] == ["visible"] and history["coverage"]["truncated"] is True
    assert history["coverage"]["complete"] is False and history["coverage"]["gap_reasons"] == ["tail_bytes_truncated"]
    before = _skill_names_from_review_history(drive_root, "root-1", task_started_at="2000-01-01T00:00:00+00:00")
    assert before["coverage"]["truncated"] is True and before["coverage"]["complete"] is False
    since = _skill_names_from_review_history(drive_root, "root-1", task_started_at="2026-01-02T00:00:00+00:00")
    assert since["names"] == ["visible"] and since["coverage"]["truncated"] is False
    assert since["coverage"]["complete"] is True and since["coverage"]["gap_reasons"] == []
    # A later rotation archived rows of a task that started before it: a cut, whatever the hot head holds —
    # here an old commit row carried into the fresh hot index ahead of everything (its record no longer resolves).
    (review_ledger.ledger_dir(drive_root) / "index.20261021T000000.jsonl").write_text("", encoding="utf-8")
    hot = review_ledger.index_path(drive_root)
    carried = json.dumps({"record_id": "rl-carried", "surface": "commit_gate", "state": "settled",
                          "ts": "2025-01-01T00:00:00+00:00", "subject": {}}) + "\n"
    hot.write_text(carried + hot.read_text(encoding="utf-8"), encoding="utf-8")
    spanning = _skill_names_from_review_history(drive_root, "root-1", task_started_at="2026-06-01T00:00:00+00:00")
    assert spanning["coverage"]["truncated"] is True and spanning["coverage"]["complete"] is False
