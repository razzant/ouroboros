
from tests import _dialogue as dialogue_test
import json
from types import SimpleNamespace

from ouroboros.dialogue_provenance import (
    dialogue_author,
    presence_provenance_from_task,
)
from ouroboros.memory import Memory


def _chronicle_view(row):
    """One chat row as ``memory_read`` renders it — the successor of the retired block formatter."""
    from ouroboros import chat_chain
    from ouroboros.tools.chronicle import _row_line

    header, text = _row_line(chat_chain.row_address(row), row, 0, {})
    return f"{header} {text}"


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
    assert expected in _chronicle_view(_row())


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
    row = json.loads(dialogue_test.log_text(logs / "chat.jsonl"))
    assert row["type"] == "task_summary"
    assert row["presence_provenance"] == presence_provenance_from_task(_presence_task())


def test_one_row_grammar_serves_memory_read_and_the_view_and_only_indents_later_lines():
    from ouroboros import chat_chain
    from ouroboros.dialogue_provenance import memory_row_header, render_memory_row, row_author

    row = {**_row(), "text": "first line\n## not a section"}
    address = chat_chain.row_address(row)
    author = row_author(row)
    header = memory_row_header(address, row, author=author)
    assert header == (f"[2026-08-21T10:00:00+00:00; {author['label']}; {chat_chain.format_address(address)}]")
    assert render_memory_row(address, row, author=author) == _chronicle_view(row)  # memory_read prints it as is
    indented = render_memory_row(address, row, author=author, indent="  ")
    assert indented == header + " first line\n  ## not a section"
    assert "\n## " in _chronicle_view(row) and "\n## " not in indented
    assert memory_row_header(address, {"ts": ""}, author={}).startswith("[time not recorded; author not recorded; ")


def test_memory_and_evidence_keep_a_relays_quoted_owner_words_and_what_a_reply_answers(tmp_path):
    """F3: every text reader of a row (memory rows, chat_history lines, room evidence) shows the relations the
    host resolved: the owner's words a relay quotes, with where they were said and what they answered."""
    from ouroboros import dialogue_store
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.chat_references import reference_from_row
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.dialogue_provenance import render_memory_row, row_author
    from ouroboros.owner_relay import owner_quote, relay_envelope
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    beta = create_project(tmp_path, "beta", name="Beta")
    write_task_result(tmp_path, "recipient", "running", chat_id=1, title="recipient")
    bind_task_to_project(tmp_path, "recipient", beta["id"], beta["chat_id"], origin={"absent": "system"})
    notice = {"chat_id": 1, "direction": "out", "type": "main_notice", "task_id": "teller",
              "ts": "2026-10-01T00:00:00Z", "text": "A specific project observation."}
    dialogue_store.append(tmp_path, "chat", notice)
    owner = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:01:00Z", "client_message_id": "owner-reply",
             "text": "Why is that here?", "reply_to": reference_from_row(notice)}
    dialogue_store.append(tmp_path, "chat", owner)
    relay_row = {"chat_id": beta["chat_id"], "direction": "out", "type": "relay_entry", "task_id": "recipient",
                 "ts": "2026-10-01T00:02:00Z", "client_message_id": "relay:quoted:recipient",
                 "text": "Please answer the owner.", "relay": relay_envelope("quoted", [owner_quote(tmp_path, owner)], chat_id=1)}
    dialogue_store.append(tmp_path, "chat", relay_row)
    notice_address = format_address(row_address(notice))
    for text in (render_memory_row(row_address(relay_row), relay_row, author=row_author(relay_row)),
                 Memory._format_chat_line(relay_row, compact=False)):
        assert "Please answer the owner." in text and "Why is that here?" in text
        assert format_address(row_address(owner)) in text and notice_address in text
    owner_text = render_memory_row(row_address(owner), owner, author=row_author(owner))
    assert owner_text.startswith("[2026-10-01T00:01:00Z; ") and "Why is that here?" in owner_text
    assert notice_address in owner_text and "A specific project observation." in owner_text
    evidence = read_room_source(tmp_path, beta["chat_id"], task_id="recipient")
    assert "Why is that here?" in evidence["text"] and notice_address in evidence["text"]
    plain = {"chat_id": 1, "direction": "out", "ts": "2026-10-01T00:03:00Z", "text": "No relations here."}
    assert Memory._format_chat_line(plain, compact=False) == "→ [2026-10-01T00:03] No relations here."
