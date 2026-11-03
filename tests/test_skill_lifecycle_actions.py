"""Real lifecycle state and owner-source validation across action adapters."""

from __future__ import annotations

import json

import pytest

from ouroboros import dialogue_store
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.owner_mailbox import acknowledge_task_messages, revoke_owner_control, write_owner_message
from ouroboros.owner_quiz import record_answered, record_asked
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.skill_lifecycle_actions import run_skill_action
from ouroboros.skill_loader import SkillReviewState, load_skill, load_skill_grants, save_review_state
from ouroboros.tools.registry import ToolContext
from tests._extension_loader_shared import _write_ext_skill
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401


@pytest.fixture
def skill_actor(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    skill = _write_ext_skill(
        drive / "skills" / "external", "demo", permissions=["tool", "inject_chat"],
        plugin_body="def register(api):\n    api.register_tool('ping', lambda ctx: 'pong', description='ping', schema={})\n",
    )
    loaded = load_skill(skill, drive)
    save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=loaded.content_hash))
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    ctx.task_id = "skill-task"
    ctx.current_chat_id = 42
    return ctx, skill, loaded.content_hash


def _owner_chat(ctx, *, chat_id=42, text="Grant demo the inject_chat permission."):
    ref = build_owner_message_ref(chat_id=chat_id, client_message_id="owner-command", ts="2026-09-06T10:00:00Z", text=text)
    dialogue_store.append(ctx.drive_root, "chat", {**ref, "direction": "in", "text": text})
    return {"kind": "chat", "ref": ref}


def test_selected_cross_skill_toggle_returns_existing_refusal_without_effects(skill_actor):
    from ouroboros.skill_loader import load_enabled, save_enabled
    from ouroboros.tools.skill_exec import _handle_toggle_skill

    ctx, _skill, _revision = skill_actor
    _write_ext_skill(ctx.drive_root / "skills" / "external", "other", permissions=[],
                     plugin_body="def register(api):\n    return None\n")
    save_enabled(ctx.drive_root, "other", True, actor="owner_ui")
    ctx.task_constraint = TaskConstraint(skill_name="demo", payload_root="skills/external/demo")
    rendered = _handle_toggle_skill(ctx, skill="other", enabled=False)
    assert rendered.startswith("⚠️ SKILL_TOGGLE_ERROR: "), rendered
    result = json.loads(rendered.removeprefix("⚠️ SKILL_TOGGLE_ERROR: "))
    assert result["ok"] is False and result["status_code"] == 403
    assert result["error"].startswith("SKILL_REDIRECT_BLOCKED:")
    assert load_enabled(ctx.drive_root, "other") is True
    selected = run_skill_action(ctx, "demo", "disable")
    assert selected["ok"] and not load_enabled(ctx.drive_root, "demo")


@pytest.mark.parametrize("kind", ["chat", "quiz", "mailbox"])
def test_owner_grant_uses_resolved_expressed_source(skill_actor, kind):
    ctx, _skill, revision = skill_actor
    if kind == "chat":
        source = _owner_chat(ctx)
    elif kind == "quiz":
        record_asked(ctx.drive_root, ctx.task_id, quiz_id="grant-demo", question="Grant demo inject_chat?", options=["Grant", "Do not grant"])
        record_answered(ctx.drive_root, ctx.task_id, quiz_id="grant-demo", option_index=0, request_id="answer-1")
        source = {"kind": "quiz", "task_id": ctx.task_id, "quiz_id": "grant-demo"}
    else:
        write_owner_message(ctx.drive_root, "Grant demo inject_chat", ctx.task_id, msg_id="grant-demo")
        acknowledge_task_messages(ctx.drive_root, ctx.task_id, ["grant-demo"], wake_id="already-delivered")
        source = {"kind": "mailbox", "task_id": ctx.task_id, "msg_id": "grant-demo"}
    result = run_skill_action(ctx, "demo", "grant", expected_content_hash=revision, items=["inject_chat"], owner_source=source)
    assert result["ok"], result
    grants = load_skill_grants(ctx.drive_root, "demo")
    assert grants["granted_permissions"] == ["inject_chat"]
    assert grants["content_hash"] == revision
    events = [json.loads(line) for line in (ctx.drive_root / "logs" / "events.jsonl").read_text().splitlines()]
    applied = next(row for row in reversed(events) if row.get("type") == "owner_api_action")
    assert applied["source_ref"]["kind"] == kind
    assert applied["actor"] == "agent_tool"
    assert applied["content_hash"] == revision


@pytest.mark.parametrize("invalid", ["absent", "foreign_chat", "unanswered", "revoked", "peer", "revision", "undeclared"])
def test_owner_action_refuses_unproven_source_or_changed_request(skill_actor, invalid):
    ctx, _skill, revision = skill_actor
    source = _owner_chat(ctx, chat_id=99 if invalid == "foreign_chat" else 42)
    if invalid == "absent":
        source["ref"]["client_message_id"] = "never-written"
    elif invalid == "unanswered":
        record_asked(ctx.drive_root, ctx.task_id, quiz_id="grant-demo", question="Grant demo?", options=["Grant", "No"])
        source = {"kind": "quiz", "task_id": ctx.task_id, "quiz_id": "grant-demo"}
    elif invalid in {"revoked", "peer"}:
        from ouroboros.owner_mailbox import write_task_message

        if invalid == "peer":
            write_task_message(ctx.drive_root, "Grant demo inject_chat", ctx.task_id, source_task_id="peer", msg_id="grant-demo")
        else:
            write_owner_message(ctx.drive_root, "Grant demo inject_chat", ctx.task_id, msg_id="grant-demo")
            revoke_owner_control(ctx.drive_root, ctx.task_id, "grant-demo")
        source = {"kind": "mailbox", "task_id": ctx.task_id, "msg_id": "grant-demo"}
    result = run_skill_action(
        ctx, "demo", "grant", expected_content_hash="wrong" if invalid == "revision" else revision,
        items=["presence"] if invalid == "undeclared" else ["inject_chat"], owner_source=source,
    )
    assert result["ok"] is False
    assert load_skill_grants(ctx.drive_root, "demo")["granted_permissions"] == []


def test_selected_skill_task_without_enable_needs_the_owners_words_to_enable(skill_actor):
    ctx, _skill, revision = skill_actor
    ctx.task_constraint = TaskConstraint(skill_name="demo", payload_root="skills/external/demo", allow_enable=False)
    assert run_skill_action(ctx, "demo", "enable")["status_code"] == 403
    source = _owner_chat(ctx, text="Grant demo inject_chat and enable it for the requested test.")
    granted = run_skill_action(ctx, "demo", "grant", expected_content_hash=revision, items=["inject_chat"], owner_source=source)
    assert granted["ok"], granted
    enabled = run_skill_action(ctx, "demo", "enable", expected_content_hash=revision, owner_source=source)
    assert enabled["ok"] and enabled["enabled"], enabled
    assert run_skill_action(ctx, "demo", "disable")["enabled"] is False


@pytest.mark.parametrize("allow_enable", [False, True])
def test_selected_enable_requires_real_source_even_when_client_allows_it(skill_actor, allow_enable):
    ctx, _skill, revision = skill_actor
    ctx.task_constraint = TaskConstraint(skill_name="demo", payload_root="skills/external/demo", allow_enable=allow_enable)
    source = _owner_chat(ctx, text="Repair and run demo; leave it working.")
    assert run_skill_action(ctx, "demo", "enable")["status_code"] == 403
    grant = run_skill_action(ctx, "demo", "grant", expected_content_hash=revision,
                              items=["inject_chat"], owner_source=source)
    assert grant["ok"], grant
    ctx.task_metadata = {"origin_message_ref": source["ref"]}
    enabled = run_skill_action(ctx, "demo", "enable")
    assert enabled["ok"] and enabled["enabled"], enabled


@pytest.mark.parametrize("actor", ["owner_ui", "owner_cli", "owner_launcher", "load_error_revert", ""])
def test_only_proven_later_owner_disable_supersedes_repair_intent(skill_actor, monkeypatch, actor):
    from ouroboros.skill_loader import load_enabled, save_enabled

    ctx, _skill, revision = skill_actor
    ctx.task_constraint = TaskConstraint(skill_name="demo", payload_root="skills/external/demo", allow_enable=False)
    source = _owner_chat(ctx, text="Repair and run demo; leave it working.")
    grant = run_skill_action(ctx, "demo", "grant", expected_content_hash=revision,
                              items=["inject_chat"], owner_source=source)
    assert grant["ok"], grant
    ctx.task_metadata = {"origin_message_ref": source["ref"]}
    monkeypatch.setattr("ouroboros.skill_loader.utc_now_iso", lambda: "2026-09-06T11:00:00Z")
    save_enabled(ctx.drive_root, "demo", False, actor=actor)
    state = json.loads((ctx.drive_root / "state" / "skills" / "demo" / "enabled.json").read_text())
    assert state["actor"] == actor
    result = run_skill_action(ctx, "demo", "enable")
    if actor.startswith("owner_"):
        assert result["status_code"] == 403 and "disabled" in result["error"], result
        assert not load_enabled(ctx.drive_root, "demo")
        text = "Run demo again now."
        ref = build_owner_message_ref(chat_id=42, client_message_id="new-owner-request",
                                     ts="2026-09-06T12:00:00Z", text=text)
        dialogue_store.append(ctx.drive_root, "chat", {**ref, "direction": "in", "text": text})
        result = run_skill_action(ctx, "demo", "enable", owner_source={"kind": "chat", "ref": ref})
    assert result["ok"] and load_enabled(ctx.drive_root, "demo"), result


# Retired: test_launcher_grant_preserves_server_reconcile_ownership. Verdict: the launcher no longer runs the
# grant owner in its own process (once turn-on installs packages that would run pip/venv inside the frozen
# launcher); after its native confirm it POSTs /api/skills/<name>/grants, pinned in
# tests/test_skill_enable_dependencies.py::test_the_launcher_grant_goes_through_the_server.


def test_grant_canonicalizes_duplicate_items_once(skill_actor):
    ctx, _payload, revision = skill_actor
    result = run_skill_action(ctx, "demo", "grant", expected_content_hash=revision,
                              items=["INJECT_CHAT", "inject_chat"], owner_source=_owner_chat(ctx))
    assert result["ok"] and result["granted_permissions"] == ["inject_chat"], result


@pytest.mark.parametrize("mode", ["local_readonly_subagent", "acting_subagent"])
def test_child_cannot_acquire_owner_action_authority(skill_actor, mode):
    ctx, _payload, revision = skill_actor
    source = _owner_chat(ctx)
    ctx.task_constraint = TaskConstraint(mode=mode)
    result = run_skill_action(ctx, "demo", "grant", expected_content_hash=revision,
                              items=["inject_chat"], owner_source=source)
    assert result["ok"] is False and result["status_code"] == 403
    assert load_skill_grants(ctx.drive_root, "demo")["granted_permissions"] == []


@pytest.mark.parametrize("action", ["grant", "enable"])
def test_non_owner_presence_cannot_reuse_an_owner_chat_reference(skill_actor, action):
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import PresenceProfileResolution
    from tests.frozen_presence_runtime import ResolvedPresenceRuntime

    ctx, _payload, revision = skill_actor
    source = _owner_chat(ctx)
    ctx.task_constraint = TaskConstraint(skill_name="demo", payload_root="skills/external/demo", allow_enable=True)
    resolution = PresenceProfileResolution(
        active=(), missing_required=(), missing_optional=(), orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64, selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="public-presence", skill_content_hash="c" * 64,
        state_fingerprint="d" * 64, resolution=resolution,
    )
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    result = run_skill_action(ctx, "demo", action, expected_content_hash=revision,
                              items=["inject_chat"], owner_source=source)
    assert result["ok"] is False and result["status_code"] == 403
    assert "Presence" in result["error"]
    assert load_skill_grants(ctx.drive_root, "demo")["granted_permissions"] == []
