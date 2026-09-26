import json
from types import SimpleNamespace

from ouroboros.consolidator import _format_entries_for_block
from ouroboros.dialogue_provenance import (
    dialogue_author,
    presence_provenance_from_task,
)
from ouroboros.memory import Memory


def _row():
    return {
        "ts": "2026-08-21T10:00:00+00:00",
        "direction": "in",
        "text": "hello",
        "sender_label": "Alex",
        "source": "presence:telegram",
        "transport": {
            "provider": "telegram",
            "account_id": "bot-1",
            "conversation_id": "room-1",
            "thread_id": "topic-1",
            "actor": {"platform_actor_id": "user-7"},
        },
    }


def test_presence_provenance_survives_recent_and_consolidated_rendering():
    expected = "Alex [provider=telegram; account=bot-1; conversation=room-1; thread=topic-1]"
    assert dialogue_author(_row()) == expected
    assert expected in Memory._format_chat_line(_row(), compact=False)
    assert expected in _format_entries_for_block([_row()])


def _presence_task():
    return {
        "id": "presence-1",
        "type": "presence",
        "chat_id": 17,
        "_presence_turn": True,
        "metadata": {
            "presence": {
                "binding_id": "b" * 32,
                "transport_skill": "telegram-bot",
                "behavior_skill": "community-helper",
                "profile_fingerprint": "p" * 64,
                "event": {
                    "source_event_id": "event-7",
                    "provider": "telegram",
                    "account_id": "bot-1",
                    "conversation_id": "room-1",
                    "thread_id": "topic-1",
                    "conversation_key": "telegram:bot-1:room-1:topic-1",
                    "actor": {
                        "platform_actor_id": "user-7",
                        "display_name": "Alex",
                    },
                },
            }
        },
        "task_contract": {
            "capability_ceiling": {
                "profile_fingerprint": "a" * 64,
                "state_fingerprint": "s" * 64,
                "selection_fingerprint": "c" * 64,
            }
        },
    }


def test_normalized_presence_task_provenance_uses_event_and_ceiling_facts():
    assert presence_provenance_from_task(_presence_task()) == {
        "binding_id": "b" * 32,
        "transport_skill": "telegram-bot",
        "behavior_skill": "community-helper",
        "profile_fingerprint": "a" * 64,
        "state_fingerprint": "s" * 64,
        "selection_fingerprint": "c" * 64,
        "source_event_id": "event-7",
        "conversation_key": "telegram:bot-1:room-1:topic-1",
        "provider": "telegram",
        "account_id": "bot-1",
        "conversation_id": "room-1",
        "thread_id": "topic-1",
        "actor_id": "user-7",
    }
    assert presence_provenance_from_task({"type": "user"}) == {}


def test_presence_task_facts_row_persists_normalized_provenance(tmp_path):
    from ouroboros.agent_task_pipeline import _record_task_facts

    logs = tmp_path / "logs"
    logs.mkdir()
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    _record_task_facts(
        env,
        _presence_task(),
        {"rounds": 1, "cost": 0.0},
        {"tool_calls": []},
        logs,
    )
    row = json.loads((logs / "chat.jsonl").read_text(encoding="utf-8"))
    assert row["type"] == "task_summary"
    assert row["presence_provenance"] == presence_provenance_from_task(_presence_task())


def test_presence_reflection_entry_is_stamped_before_append(tmp_path, monkeypatch):
    import ouroboros.reflection as reflection
    from ouroboros.agent_task_pipeline import _run_reflection

    captured = []
    monkeypatch.setattr(reflection, "should_generate_reflection", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        reflection,
        "generate_reflection",
        lambda *args, **kwargs: {"reflection": "useful", "memory_actions": []},
    )
    monkeypatch.setattr(
        reflection,
        "append_reflection_routed",
        lambda env, task, entry: captured.append(dict(entry)),
    )
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    entry = _run_reflection(
        env,
        object(),
        _presence_task(),
        {"rounds": 2, "cost": 0.0},
        {"tool_calls": []},
        {},
    )
    assert entry == captured[0]
    assert captured[0]["presence_provenance"] == presence_provenance_from_task(_presence_task())
