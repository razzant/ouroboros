"""Chat-first owner controls for generic presence bindings and selections."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros.presence_capabilities import load_presence_state
from ouroboros.utils import atomic_write_json
from ouroboros.tools.presence import get_tools
from ouroboros.tools.registry import ToolContext


def _configure(ctx: ToolContext, action: str, **params):
    entry = next(item for item in get_tools() if item.name == "configure_presence")
    return entry.handler(ctx, action, **params)


def _ctx(tmp_path) -> ToolContext:
    repo = tmp_path / "repo"
    repo.mkdir()
    return ToolContext(repo_dir=repo, drive_root=tmp_path)


def test_configure_presence_creates_lists_and_disables_binding(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    created = json.loads(_configure(
        ctx,
        "create",
        transport_skill="telegram-bot",
        behavior_skill="community-helper",
        transport="telegram",
        account_id="bot-1",
        conversation_id="room-1",
        thread_id="topic-1",
    ))

    listed = json.loads(_configure(ctx, "list"))
    assert listed["bindings"][0]["binding_id"] == created["binding_id"]
    assert listed["bindings"][0]["origin"] == {
        "transport": "telegram",
        "account_id": "bot-1",
        "conversation_id": "room-1",
        "thread_id": "topic-1",
    }

    disabled = json.loads(_configure(ctx, "disable", binding_id=created["binding_id"]))
    assert disabled == {"ok": True, "binding_id": created["binding_id"], "enabled": False}


def test_configure_presence_inspects_and_selects_exact_tool(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    skill_dir = tmp_path / "skills" / "external" / "community-helper"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: community-helper\n"
        "description: Neutral presence fixture.\n"
        "version: 0.1.0\n"
        "type: instruction\n"
        "presence:\n"
        "  instructions: Participate helpfully.\n"
        "  capability_requests:\n"
        "    - id: history\n"
        "      kind: tool\n"
        "      required: true\n"
        "      purpose: Read prior dialogue.\n"
        "---\n"
        "# Community helper\n",
        encoding="utf-8",
    )

    inspected = json.loads(_configure(
        ctx, "inspect", behavior_skill="community-helper",
    ))
    assert inspected["requests"][0]["id"] == "history"
    assert inspected["selections"] == []

    selected = json.loads(_configure(
        ctx,
        "select",
        behavior_skill="community-helper",
        request_id="history",
        target_type="tool",
        tool_kind="builtin",
        target_name="chat_history",
    ))
    state = load_presence_state(tmp_path, "community-helper")
    assert selected["ok"] is True
    assert state.selections[0].target.name == "chat_history"

    runtime = json.loads(_configure(
        ctx,
        "runtime",
        behavior_skill="community-helper",
        model_slot="light",
        inline_max_rounds=6,
    ))
    assert runtime["runtime_overrides"] == {
        "model_slot": "light",
        "inline_max_rounds": 6,
    }
    assert load_presence_state(tmp_path, "community-helper").runtime_overrides.model_slot == "light"


def test_configure_presence_rejects_unimplemented_resource_argument_projection(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    skill_dir = tmp_path / "skills" / "external" / "community-helper"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: community-helper\n"
        "description: Neutral presence fixture.\n"
        "version: 0.1.0\n"
        "type: instruction\n"
        "presence:\n"
        "  instructions: Participate helpfully.\n"
        "  capability_requests:\n"
        "    - id: sender\n"
        "      kind: tool\n"
        "      required: true\n"
        "      purpose: Send one bounded result.\n"
        "---\n",
        encoding="utf-8",
    )

    result = _configure(
        ctx,
        "select",
        behavior_skill="community-helper",
        request_id="sender",
        target_type="tool",
        tool_kind="builtin",
        target_name="send_user_message",
        argument_bindings=[{
            "argument_path": ["path"],
            "source": "resource",
            "resource_request_id": "workspace",
        }],
    )
    assert result == "ERROR: PRESENCE_RESOURCE_ARGUMENT_BINDING_UNSUPPORTED"


def test_initiate_presence_resolves_binding_and_reports_actual_delivery(monkeypatch, tmp_path) -> None:
    ctx = _ctx(tmp_path)
    created = json.loads(_configure(
        ctx,
        "create",
        transport_skill="telegram-bot",
        behavior_skill="community-helper",
        transport="telegram",
        account_id="bot-1",
        conversation_id="room-1",
    ))
    captured = {}

    def admit(**kwargs):
        captured["admission_args"] = kwargs
        return SimpleNamespace()

    def run(**kwargs):
        captured["event"] = kwargs["event"]
        return SimpleNamespace(
            outcome="tool_delivered", text="", task_id="turn-1", work_ref="",
        )

    monkeypatch.setattr("ouroboros.presence_admission.admit_presence_turn", admit)
    monkeypatch.setattr("ouroboros.presence_runner.run_presence_turn", run)
    entry = next(item for item in get_tools() if item.name == "initiate_presence")
    result = json.loads(entry.handler(
        ctx, created["binding_id"], "Check whether the room needs an update.", "wake-1",
    ))

    assert result["delivered"] is True
    assert captured["admission_args"]["authenticated_transport_skill"] == "telegram-bot"
    assert captured["event"].conversation_id == "room-1"
    assert captured["event"].actor["kind"] == "proactive_initiation"


def test_presence_cancel_work_accepts_only_same_binding_and_conversation(monkeypatch, tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {
        "presence": {
            "binding_id": "1" * 32,
            "event": {"conversation_key": "telegram:bot-1:room-1"},
        }
    }
    atomic_write_json(
        tmp_path / "task_results" / "presence-work-1.json",
        {
            "_schema_version": 1,
            "task_id": "presence-work-1",
            "status": "running",
            "metadata": {
                "presence": {
                    "binding_id": "1" * 32,
                    "event": {"conversation_key": "telegram:bot-1:room-1"},
                }
            },
        },
    )
    monkeypatch.setattr(
        "ouroboros.tools.join_ledger._cancel_task",
        lambda _ctx, task_id, reason="": f"cancel:{task_id}:{reason}",
    )
    entry = next(item for item in get_tools() if item.name == "presence_cancel_work")

    assert entry.handler(ctx, "presence-work-1", "no longer needed") == (
        "cancel:presence-work-1:no longer needed"
    )
    ctx.task_metadata["presence"]["event"]["conversation_key"] = "telegram:bot-1:other"
    assert entry.handler(ctx, "presence-work-1") == "ERROR: PRESENCE_WORK_NOT_CORRELATED"
