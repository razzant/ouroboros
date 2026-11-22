"""The self-authored skill finalization gate of ``ouroboros.loop``.

Split out of ``tests/test_loop_misc.py`` when that module was divided by
theme; every moved block is verbatim.
"""
from __future__ import annotations

import json

from ouroboros.loop_nudges import _skill_finalization_message, _skill_names_touched_by_trace
from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    save_enabled,
    save_review_state,
)


# ---------------------------------------------------------------------------
# Skill finalization gate (self-authored skills must reach ready+enabled
# before the loop accepts a final text response)
# ---------------------------------------------------------------------------


def _write_self_authored_skill(drive_root, name: str = "alpha"):
    skill_dir = drive_root / "skills" / "external" / name
    state_dir = drive_root / "state" / "skills" / name
    skill_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ntype: instruction\nversion: 0.1.0\n---\nbody\n",
        encoding="utf-8",
    )
    marker = {
        "schema_version": 1,
        "origin": "self_authored",
        "task_id": "task-1",
        "created_at": "2026-05-07T00:00:00+00:00",
    }
    (skill_dir / ".self_authored.json").write_text(json.dumps(marker), encoding="utf-8")
    (state_dir / "self_authored.json").write_text(json.dumps(marker), encoding="utf-8")
    return skill_dir

def test_skill_names_touched_by_trace_detects_data_skill_edits():
    trace = {
        "tool_calls": [
            {"tool": "write_file", "args": {"path": "skills/external/alpha/plugin.py"}},
            {"tool": "edit_text", "args": {"path": "data/skills/external/beta/SKILL.md"}},
            {"tool": "write_file", "args": {"path": "SKILL.md", "bucket": "external", "skill_name": "delta"}},
        ]
    }

    assert _skill_names_touched_by_trace(trace) == ["alpha", "beta", "delta"]

def test_skill_finalization_message_blocks_unreviewed_self_authored_skill(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_self_authored_skill(drive_root)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "skills/external/alpha/SKILL.md"}}]}

    message = _skill_finalization_message(drive_root, trace)

    assert "SKILL_NOT_FINALIZED" in message
    assert "alpha" in message

def test_skill_finalization_message_sees_real_skill_payload_selectors(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_self_authored_skill(drive_root)

    for tool, selector in (
        ("run_command", {"cmd": ["true"], "cwd": "skill_payload"}),
        ("run_script", {"script": "pass", "cwd": "skill_payload/scripts"}),
        ("delegate_start", {"prompt": "repair", "root": "skill_payload"}),
    ):
        trace = {"tool_calls": [{"tool": tool, "args": {
            **selector, "bucket": "external", "skill_name": "alpha",
        }}]}
        assert "alpha" in _skill_finalization_message(drive_root, trace)

def test_skill_finalization_message_allows_ready_self_authored_skill(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    skill_dir = _write_self_authored_skill(drive_root)
    content_hash = compute_content_hash(skill_dir)
    save_review_state(drive_root, "alpha", SkillReviewState(status="pass", content_hash=content_hash))
    save_enabled(drive_root, "alpha", True)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "skills/external/alpha/SKILL.md"}}]}

    assert _skill_finalization_message(drive_root, trace) == ""


def test_a_skill_changed_during_the_work_gets_one_neutral_fact(tmp_path, monkeypatch):
    """B6″ (owner 1A, 21.11): a skill enabled at the work's first view whose files changed gets one fact read from its
    gate in this mode: outside Cyber Pro it names ``review_stale``, never «its keys still work»; in Cyber Pro the keys
    that still work and the permission that waits. An unchanged skill, and an acceptance in the same work, leave
    none; a work without a first view (a delegate) reads the skills its trace touched."""
    from types import SimpleNamespace

    from ouroboros import config
    from ouroboros.skill_loader import find_skill, save_skill_grants
    from ouroboros.skill_readiness import note_enabled_skill_digests

    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    cyber = {"on": False}
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: cyber["on"])
    (tmp_path / "settings.json").write_text(json.dumps({"FIXTURE_KEY": "example"}), encoding="utf-8")
    folder = tmp_path / "skills" / "external" / "demo"
    folder.mkdir(parents=True)
    (folder / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    (folder / "SKILL.md").write_text(
        "---\nname: demo\ndescription: d\nversion: 1.0.0\ntype: extension\nentry: plugin.py\nplugin_api: \"2.0\"\n"
        "permissions: [notify_owner]\nenv_from_settings: [FIXTURE_KEY]\n---\nFixture.\n", encoding="utf-8")

    def accept() -> None:
        current = find_skill(tmp_path, "demo")
        save_review_state(tmp_path, "demo", SkillReviewState(status="clean", content_hash=current.content_hash))
        save_skill_grants(tmp_path, "demo", ["FIXTURE_KEY"], content_hash=current.content_hash,
                          requested_keys=["FIXTURE_KEY"], granted_permissions=["notify_owner"],
                          requested_permissions=["notify_owner"])

    accept()
    save_enabled(tmp_path, "demo", True, actor="owner_ui")
    ctx = SimpleNamespace()
    note_enabled_skill_digests(ctx, [find_skill(tmp_path, "demo")])
    read_only = {"tool_calls": [{"tool": "skill_preflight", "args": {"skill": "demo"}}]}
    assert _skill_finalization_message(tmp_path, read_only, ctx) == "", "an unchanged skill owes nothing"
    (folder / "notes.md").write_text("edited\n", encoding="utf-8")
    message = _skill_finalization_message(tmp_path, read_only, ctx)
    assert "files of skill demo changed during this work" in message and "review_stale" in message
    assert "still work" not in message
    cyber["on"] = True
    message = _skill_finalization_message(tmp_path, read_only, ctx)
    assert "its keys (FIXTURE_KEY) still work" in message and "(notify_owner) wait for the check" in message
    delegate = _skill_finalization_message(tmp_path, read_only, SimpleNamespace())
    assert "files of skill demo were touched by this work" in delegate
    assert _skill_finalization_message(tmp_path, {"tool_calls": []}, SimpleNamespace()) == ""
    accept()
    assert _skill_finalization_message(tmp_path, read_only, ctx) == "", "accepted in the same work: nothing is owed"
