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
    """B6″ (owner 1A, 21.11): a skill enabled at the work's first view whose files changed by the end, a hub
    install no self-authored check would see, gets one fact: its keys still work, its permissions on the
    owner's behalf wait for the check of the new version; the decision is the mind's. An untouched skill and
    a skill turned off during the work stay silent."""
    import types

    from ouroboros import config
    from ouroboros.skill_grant_scope import install_identity
    from ouroboros.skill_install_record import write_install_record
    from ouroboros.skill_loader import discover_skills, find_skill, save_skill_grants
    from ouroboros.skill_readiness import note_enabled_skill_digests

    monkeypatch.setattr(config, "load_settings", lambda: {"CAL_TOKEN": "x"})
    drive_root = tmp_path / "drive"
    skill_dir = drive_root / "skills" / "ouroboroshub" / "cal"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: cal\ndescription: d\nversion: 1.0.0\ntype: extension\nentry: plugin.py\n"
        "env_from_settings: [CAL_TOKEN]\npermissions: [notify_owner]\n---\n\n# cal\n", encoding="utf-8")
    (skill_dir / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    write_install_record(drive_root, "cal", {"source": {"id": "hub", "kind": "hub"},
                                             "release": {"digest": "a" * 64}, "basis": "trusted"})
    loaded = find_skill(drive_root, "cal")
    save_review_state(drive_root, "cal", SkillReviewState(status="pass", content_hash=loaded.content_hash))
    save_enabled(drive_root, "cal", True)
    save_skill_grants(drive_root, "cal", ["CAL_TOKEN"], content_hash=loaded.content_hash, requested_keys=["CAL_TOKEN"],
                      granted_permissions=["notify_owner"], requested_permissions=["notify_owner"],
                      install_identity=install_identity(drive_root, find_skill(drive_root, "cal")))
    ctx = types.SimpleNamespace()
    note_enabled_skill_digests(ctx, discover_skills(drive_root))
    assert ctx._skill_digests_at_start == {"cal": loaded.content_hash}
    assert _skill_finalization_message(drive_root, {"tool_calls": []}, ctx) == ""  # untouched: silent

    (skill_dir / "plugin.py").write_text("def register(api):\n    return None\n", encoding="utf-8")  # a shell edit
    message = _skill_finalization_message(drive_root, {"tool_calls": []}, ctx)
    assert "files of skill cal changed during this work (by this work or another participant)" in message
    assert "its keys (CAL_TOKEN) still work" in message
    assert "(notify_owner) wait for the check of the new version" in message
    # A delegate's work has no first view: its trace names the skill it touched.
    touched = {"tool_calls": [{"tool": "write_file", "args": {"path": "skills/ouroboroshub/cal/plugin.py"}}]}
    delegate = types.SimpleNamespace(_skill_digests_at_start={})
    assert "differ from the version its permissions were granted for" in _skill_finalization_message(
        drive_root, touched, delegate)
    assert _skill_finalization_message(drive_root, {"tool_calls": []}, delegate) == ""
    save_enabled(drive_root, "cal", False)  # turned off during the work
    assert _skill_finalization_message(drive_root, {"tool_calls": []}, ctx) == ""
