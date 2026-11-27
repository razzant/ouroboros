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
    """Observed file differences, activity without proof, a nanny's quiet notice and explicit Off stay distinct."""
    from types import SimpleNamespace

    from ouroboros import config
    from ouroboros.context import build_llm_messages
    from ouroboros.loop_nudges import _maybe_inject_finalization_nudges
    from ouroboros.skill_loader import discover_skills, find_skill, save_skill_grants
    from ouroboros.skill_readiness import changed_skills_notice, note_enabled_skill_digests
    from ouroboros.tools.registry import ToolContext
    from tests.test_cache_optimization import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    drive = env.drive_root
    # End install-ownership lookup at this fixture, not an ancestor of pytest's scratch.
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    cyber = {"on": False}
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: cyber["on"])
    (drive / "settings.json").write_text(json.dumps({"FIXTURE_KEY": "example"}), encoding="utf-8")
    folder = drive / "skills" / "external" / "demo"
    folder.mkdir(parents=True)
    (folder / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    (folder / "SKILL.md").write_text(
        "---\nname: demo\ndescription: d\nversion: 1.0.0\ntype: extension\nentry: plugin.py\nplugin_api: \"2.0\"\n"
        "permissions: [notify_owner]\nenv_from_settings: [FIXTURE_KEY]\n---\nFixture.\n", encoding="utf-8")

    def accept() -> None:
        current = find_skill(drive, "demo")
        save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=current.content_hash))
        save_skill_grants(drive, "demo", ["FIXTURE_KEY"], content_hash=current.content_hash,
                          requested_keys=["FIXTURE_KEY"], granted_permissions=["notify_owner"],
                          requested_permissions=["notify_owner"])

    def work() -> ToolContext:  # a new work: the context assembly records its first view
        ctx = ToolContext(repo_dir=env.repo_dir, drive_root=drive)
        build_llm_messages(env=env, memory=memory, task={"id": "t1", "type": "task", "text": "hi"}, ctx=ctx)
        assert ctx._skill_digests_at_start == {"demo": find_skill(drive, "demo").content_hash}
        return ctx

    def edit(text: str) -> None:
        (folder / "notes.md").write_text(text + "\n", encoding="utf-8")

    accept()
    save_enabled(drive, "demo", True, actor="owner_ui")
    read_only = {"tool_calls": [{"tool": "skill_preflight", "args": {"skill": "demo"}}]}
    untouched = work()
    assert changed_skills_notice(drive, read_only, untouched) == "", "an unchanged skill owes nothing"
    ctx = work()
    edit("edited")
    assert changed_skills_notice(drive, read_only, untouched) == "", "an empty comparison is not repeated either"
    message = changed_skills_notice(drive, read_only, ctx)
    assert "files of skill demo changed since this work's first observation (author not established)" in message
    assert "review_stale" in message and "still work" not in message and "SKILL_NOT_FINALIZED" not in message
    assert changed_skills_notice(drive, read_only, ctx) == "", "compared once per work"
    cyber["on"] = True
    ctx = work()
    ctx._nanny_route_dispatched = True
    edit("edited again")
    note_enabled_skill_digests(ctx, discover_skills(drive))  # another existing catalogue read saw the change
    messages, progress = [], []
    trace = {"reasoning_notes": []}

    def no_scan(*_args, **_kwargs):
        raise AssertionError("finalization must not discover every payload")

    with monkeypatch.context() as patch:
        patch.setattr("ouroboros.skill_loader.discover_skills", no_scan)
        assert _maybe_inject_finalization_nudges(SimpleNamespace(_ctx=ctx), drive, "t1", trace, "Done", messages,
                                                 progress.append)
    message = messages[-1]["content"]
    assert ctx._completion_hold_cause == "pre_delivery_notice"
    assert not progress and trace["reasoning_notes"], "a nanny retains the observation but sends no progress row"
    assert "author not established" in message and "SKILL_NOT_FINALIZED" not in message
    assert "missing_grants" in message and "FIXTURE_KEY and notify_owner were granted to an earlier version" in message
    assert "Settings values stay saved." in message and "still work" not in message
    # An attempt names a subject; neither a tool's name nor a toggle proves a payload write.
    assert changed_skills_notice(drive, read_only, SimpleNamespace()) == ""
    for call in (
        {"tool": "write_file", "args": {"path": "skills/external/demo/notes.md"}, "is_error": True},
        {"tool": "toggle_skill", "args": {"skill": "demo", "enabled": True}, "is_error": False},
    ):
        message = changed_skills_notice(drive, {"tool_calls": [call]}, SimpleNamespace())
        assert "current state of skill demo (file changes not verified)" in message and "missing_grants" in message
    ctx = work()
    edit("edited and accepted")
    accept()
    assert changed_skills_notice(drive, read_only, ctx) == "", "accepted in the same work: nothing is owed"
    ctx = work()
    edit("edited and off")
    save_enabled(drive, "demo", False, actor="owner_ui")
    assert changed_skills_notice(drive, read_only, ctx) == ""


def test_the_end_of_work_fact_names_an_extension_load_failure(tmp_path, monkeypatch):
    """A current, in-process load attempt is evidence, not a cross-process assertion or a functional key check."""
    from types import SimpleNamespace

    from ouroboros import config, extension_loader
    from ouroboros.skill_loader import find_skill, save_skill_grants
    from ouroboros.skill_readiness import changed_skills_notice, note_enabled_skill_digests
    from tests._extension_loader_shared import _prepare_extension
    from tests._shared import clean_extension_runtime_state

    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: True)
    clean_extension_runtime_state()
    loaded, repo, drive = _prepare_extension(tmp_path, "mailer", "def register(api):\n    pass\n", ["read_settings"],
                                             env_from_settings=["MAILER_TOKEN"])
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo))
    values = {"MAILER_TOKEN": "fixture"}
    monkeypatch.setattr(config, "load_settings", lambda *_a, **_k: dict(values))
    save_skill_grants(drive, "mailer", ["MAILER_TOKEN"], content_hash=loaded.content_hash,
                      requested_keys=["MAILER_TOKEN"])
    plugin = repo / "mailer" / "plugin.py"

    def work() -> SimpleNamespace:
        ctx = SimpleNamespace()
        note_enabled_skill_digests(ctx, [find_skill(drive, "mailer", repo_path=str(repo))])
        return ctx

    def reconcile() -> dict:
        return extension_loader.reconcile_extension("mailer", drive, lambda: dict(values), repo_path=str(repo))

    trace = {"tool_calls": [{"tool": "skill_review", "args": {"skill": "mailer"}}]}

    def grant_current() -> None:
        current = find_skill(drive, "mailer", repo_path=str(repo))
        save_skill_grants(drive, "mailer", ["MAILER_TOKEN"], content_hash=current.content_hash,
                          requested_keys=["MAILER_TOKEN"])

    try:
        assert reconcile()["live_loaded"]
        ctx = work()
        plugin.write_text("raise RuntimeError('broken on purpose')\n", encoding="utf-8")
        grant_current()  # no grant gap to conceal an independent load failure
        assert not reconcile()["live_loaded"]
        message = changed_skills_notice(drive, trace, ctx)
        assert "a load attempt for the current extension failed in this process" in message
        assert "broken on purpose" in message and "still work" not in message
        ctx = work()
        plugin.write_text("def register(api):\n    return None\n", encoding="utf-8")
        grant_current()
        assert reconcile()["live_loaded"]
        message = changed_skills_notice(drive, trace, ctx)
        assert "no recorded acceptance" in message and "still work" not in message
        ctx = work()
        plugin.write_text("def register(api):\n    return 'accepted'\n", encoding="utf-8")
        current = find_skill(drive, "mailer", repo_path=str(repo))
        save_review_state(drive, "mailer", SkillReviewState(status="clean", content_hash=current.content_hash))
        grant_current()
        assert reconcile()["live_loaded"]
        assert changed_skills_notice(drive, trace, ctx) == "", "accepted and loading: nothing is owed"
    finally:
        clean_extension_runtime_state()


def test_the_end_of_work_fact_reads_the_server_view_of_an_extension(tmp_path, monkeypatch):
    """The last server error is not tied to this payload and a component's failure is not a load failure."""
    from types import SimpleNamespace

    from ouroboros import config
    from ouroboros.extension_health import BROKEN, COMPANION_RESTART_EXHAUSTED, record_extension_health
    from ouroboros.skill_loader import find_skill, save_skill_grants
    from ouroboros.skill_readiness import changed_skills_notice, note_enabled_skill_digests

    drive = tmp_path / "data"
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    monkeypatch.setattr("ouroboros.skill_review_status.cyber_authority", lambda: True)
    monkeypatch.setattr("ouroboros.extension_liveness._process_role", lambda: "worker")
    folder = drive / "skills" / "external" / "mailer"
    folder.mkdir(parents=True)
    (folder / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    (folder / "SKILL.md").write_text(
        "---\nname: mailer\ndescription: d\nversion: 1.0.0\ntype: extension\nentry: plugin.py\nplugin_api: \"2.0\"\n"
        "permissions: [read_settings]\nenv_from_settings: [MAILER_TOKEN]\n---\nFixture.\n", encoding="utf-8")
    (drive / "settings.json").write_text(json.dumps({"MAILER_TOKEN": "fixture"}), encoding="utf-8")
    save_review_state(drive, "mailer", SkillReviewState(status="clean", content_hash=find_skill(drive, "mailer").content_hash))
    save_skill_grants(drive, "mailer", ["MAILER_TOKEN"], content_hash=find_skill(drive, "mailer").content_hash,
                      requested_keys=["MAILER_TOKEN"])
    save_enabled(drive, "mailer", True, actor="owner_ui")

    edits = iter(range(1, 10))

    def changed() -> str:  # a work whose first view saw the skill, then an edit of its files
        ctx = SimpleNamespace()
        note_enabled_skill_digests(ctx, [find_skill(drive, "mailer")])
        (folder / "notes.md").write_text(f"edit {next(edits)}\n", encoding="utf-8")
        save_skill_grants(drive, "mailer", ["MAILER_TOKEN"], content_hash=find_skill(drive, "mailer").content_hash,
                          requested_keys=["MAILER_TOKEN"])
        trace = {"tool_calls": [{"tool": "skill_review", "args": {"skill": "mailer"}}]}
        return changed_skills_notice(drive, trace, ctx)

    record_extension_health(drive, "mailer", status=BROKEN, reason="load_error", process="server",
                            load_error="registration refused by the server\n" + "details " * 60)
    message = changed()
    assert "last recorded server extension load failure (payload revision not recorded)" in message
    assert "registration refused by the server\ndetails" in message and "OMISSION NOTE" in message
    assert "kind='skill_state', ids=['mailer'], state_file='health'" in message
    record_extension_health(drive, "mailer", status=BROKEN, reason=f"{COMPANION_RESTART_EXHAUSTED}:sync",
                            process="server", load_error="calendar sync ended")
    message = changed()
    assert "last recorded server companion failure (sync)" in message and "calendar sync ended" in message
    assert "extension load failure" not in message and "not ready" not in message
    monkeypatch.setattr("ouroboros.deployment_extensions.trust_gap", lambda *_a: "changed")
    message = changed()
    assert "deployment_untrusted: Its files differ from the version your organization" in message, message
    assert "companion failure" not in message


def test_finish_reports_an_unavailable_script_without_blocking_its_separate_mcp(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import config
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_readiness import changed_skills_notice

    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "get_skills_repo_path", lambda: "")
    # The gap under test is the absent script file, not this host's Node. Where the reviewed Node is not prepared (a
    # fresh checkout, CI) the interpreter reason is long and the 300-character preview cuts the separate-MCP clause
    # (disclosed by its omission note; the whole note stays at list_skills): that host fact is not asserted here.
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _runtime: ("/node", ""))
    folder = tmp_path / "skills" / "external" / "mixed"
    folder.mkdir(parents=True)
    (folder / "tool.js").write_text("// fixture, never started\n", encoding="utf-8")
    simple = ("---\nname: mixed\nversion: 1.0.0\ntype: script\nruntime: node\n"
              "scripts: [{name: missing.js}]\n")
    manifest = folder / "SKILL.md"
    trace = {"tool_calls": [{"tool": "skill_review", "args": {"skill": "mixed"}}]}
    save_enabled(tmp_path, "mixed", True, actor="owner_ui")
    for extra, expected in (
        ("mcp_servers: [{id: fixture, entry: tool.js, runtime: node}]\n", "its MCP tools are separate"),
        ("", "not ready: entry_unavailable:"),
    ):
        manifest.write_text(simple + extra + "---\nFixture.\n", encoding="utf-8")
        current = find_skill(tmp_path, "mixed")
        save_review_state(tmp_path, "mixed", SkillReviewState(status="clean", content_hash=current.content_hash))
        message = changed_skills_notice(tmp_path, trace, SimpleNamespace())
        assert expected in message
        if extra:
            assert "skill_exec unavailable" in message and "not ready" not in message
