"""The acceptance packet's per-skill lifecycle facts (visibility only).

A task that authored or repaired a skill used to hand the panel no way to see
whether that skill actually reached a reviewed, ready, enabled state. These
facts close that gap. They are facts, never a gate: acceptance judges quality,
never the execution route.
"""

from __future__ import annotations

import json
import pathlib

from ouroboros.review_evidence import build_task_acceptance_evidence
from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary
from ouroboros.skill_loader import find_skill
from ouroboros.skill_readiness import (
    _skill_names_from_review_history,
    acceptance_skill_lifecycle,
    skill_names_touched_by_trace,
    skill_readiness_for_execution,
)
from ouroboros.tools.registry import ToolContext


def _write_skill(drive_root: pathlib.Path, name: str) -> pathlib.Path:
    skill_dir = drive_root / "skills" / "external" / name
    skill_dir.mkdir(parents=True)
    (drive_root / "state" / "skills" / name).mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ntype: instruction\nversion: 0.1.0\n---\nbody\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_history(drive_root: pathlib.Path, name: str, root_task_id: str) -> None:
    path = drive_root / "state" / "skills" / name / "review_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ts": "2026-09-03T00:00:00+00:00", "status": "pass",
                    "root_task_id": root_task_id}) + "\n",
        encoding="utf-8",
    )
    projection = drive_root / "state" / "skill_review_root_tasks.jsonl"
    with projection.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"root_task_id": root_task_id, "skill": name}) + "\n")


def _ctx(drive_root: pathlib.Path, tmp_path: pathlib.Path, task_id: str) -> ToolContext:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=drive_root, task_id=task_id)


def test_the_lifecycle_scan_joins_trace_edits_and_this_root_task_history(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_skill(drive_root, "edited")
    _write_skill(drive_root, "child_authored")
    _write_skill(drive_root, "someone_elses")
    _write_history(drive_root, "child_authored", "root-1")
    _write_history(drive_root, "someone_elses", "root-other")

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
    # An unreviewed, disabled self-authored skill is exactly what the panel
    # needs to see, so the row must be honest rather than empty.
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


def test_split_root_packet_reads_skill_lifecycle_from_the_canonical_root(tmp_path):
    canonical = tmp_path / "canonical"
    execution = tmp_path / "execution"
    canonical.mkdir()
    execution.mkdir()
    _write_skill(canonical, "canonical-skill")
    ctx = ToolContext(
        repo_dir=tmp_path, drive_root=execution, budget_drive_root=canonical,
        task_id="task-split-skill",
    )

    packet = build_task_acceptance_evidence(
        ctx,
        llm_trace={"tool_calls": [{
            "tool": "skill_review", "args": {"skill": "canonical-skill"},
        }]},
        drive_root=execution,
        task_id="task-split-skill",
    )

    assert packet["skill_lifecycle"][0]["name"] == "canonical-skill"
    assert packet["skill_lifecycle"][0].get("present", True) is True


def test_the_packet_carries_the_section_and_the_vocabulary_resolves_it(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_skill(drive_root, "alpha")
    ctx = _ctx(drive_root, tmp_path, "task-skill")
    trace = {"tool_calls": [
        {"tool": "write_file", "args": {"path": "data/skills/external/alpha/SKILL.md"},
         "status": "ok"},
    ]}
    packet = build_task_acceptance_evidence(
        ctx, llm_trace=trace, drive_root=drive_root, task_id="task-skill",
    )
    assert packet["skill_lifecycle"][0]["name"] == "alpha"
    assert packet["__provenance__"]["skill_lifecycle"] == "host_attested"
    assert acceptance_evidence_ref_vocabulary(packet)["skill_lifecycle"] == "packet_section"


def test_no_touched_skill_adds_no_section(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    ctx = _ctx(drive_root, tmp_path, "task-plain")
    packet = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": [{"tool": "run_command", "status": "ok"}]},
        drive_root=drive_root, task_id="task-plain",
    )
    assert "skill_lifecycle" not in packet
    assert acceptance_skill_lifecycle(drive_root, {"tool_calls": []}, "") == []
    assert acceptance_skill_lifecycle(None, {"tool_calls": []}, "") == []


def test_bounded_history_projection_discloses_both_omission_limits(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_skill(drive_root, "visible")
    projection = drive_root / "state" / "skill_review_root_tasks.jsonl"
    projection.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps({
            "root_task_id": "other", "skill": f"skill-{index}",
            "padding": "x" * 1500,
        })
        for index in range(800)
    ]
    rows.append(json.dumps({"root_task_id": "root-1", "skill": "visible"}))
    projection.write_text("\n".join(rows) + "\n", encoding="utf-8")

    history = _skill_names_from_review_history(drive_root, "root-1")
    assert history["names"] == ["visible"]
    assert history["coverage"]["rows_scanned"] <= 512
    assert history["coverage"]["truncated"] is True
    assert set(history["coverage"]["gap_reasons"]) >= {
        "tail_bytes_truncated", "max_entries_truncated",
    }
    assert history["coverage"]["source_ref"] == {
        "kind": "canonical_jsonl",
        "path": "state/skill_review_root_tasks.jsonl",
        "reader": "read_file",
    }

    ctx = _ctx(drive_root, tmp_path, "task-bounded-history")
    packet = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": [{
            "tool": "skill_review", "args": {"skill": "visible"},
        }]}, drive_root=drive_root, task_id="task-bounded-history",
    )
    assert packet["skill_lifecycle_history_coverage"]["truncated"] is True
    assert packet["skill_lifecycle_complete"] is False
    assert acceptance_evidence_ref_vocabulary(packet)["skill_lifecycle"] == "partial"
    partial = next(
        row for row in packet["__unresolved_partial_artifacts__"]
        if row["tool"] == "skill_lifecycle"
    )
    assert partial["status"] == "not_materialized_for_reviewer"
    assert partial["source_ref"] == history["coverage"]["source_ref"]
