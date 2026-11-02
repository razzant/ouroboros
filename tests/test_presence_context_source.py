"""Later Presence turns can read the exact context observed with earlier speech.

The runner, memory views, selected registry reader and pack/sweep are real. Only
the model is scripted; retaining a source does not prove semantic understanding.
"""
from __future__ import annotations

from tests import _dialogue as dialogue_test

from dataclasses import replace
import json

import pytest

from ouroboros import chat_chain, evidence_packer, memory_view
from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from ouroboros.dialogue_provenance import dialogue_text
from ouroboros.presence_context import build_presence_context_section
from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_presence_cognitive_baseline import _ceiling
from tests.test_presence_runner import _admission, _event, _terminal


QUESTION = "Which approach would you choose for our service?"
CANCELLED = "We have already chosen; there is no open request now."


def _observed_event():
    # Context is provider-owned data, with its own observation time and gap facts.
    # It includes messages newer than the triggering event, not a historical replay.
    return replace(_event(), text="That is the service.", conversation={
        "title": "Community", "thread_history": {
            "status": "partial", "observed_at": "2026-10-15T11:00:00Z",
            "scope": "current_accessible_thread",
            "data": {"messages": [
                {"id": "earlier", "actor_id": "person-a", "text": QUESTION},
                {"id": "later", "actor_id": "person-b", "text": CANCELLED},
            ], "complete": False, "next_cursor": "remaining-page"},
            "error": {"code": "rate_limited"},
        }}, message={"message_id": "42", "sent_at": "2026-10-15T10:00:00Z"})


def _rows(root):
    return [row for _address, row, _pos in chat_chain.iter_rows(root) if row.get("direction") in {"in", "out"}]


def _context_ref(text):
    prefix = "memory_read(source_ref="
    assert prefix in text
    return json.JSONDecoder().raw_decode(text.split(prefix, 1)[1])[0]


@pytest.mark.parametrize("packed", [False, True], ids=["loose", "packed"])
def test_next_presence_reads_the_same_source_through_its_selected_memory_tool(tmp_path, monkeypatch, packed):
    repo, data, fork = (tmp_path / name for name in ("repo", "data", "fork"))
    for path in (repo, data, fork):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    admission = replace(_admission(), capability_ceiling=_ceiling())
    event = _observed_event()
    tasks, reads = [], []

    class Agent:
        def handle_task(self, task):
            tasks.append(task)
            if len(tasks) == 1:
                initial = build_presence_context_section(data, task["metadata"]["presence"])
                assert QUESTION in initial and CANCELLED in initial and "remaining-page" in initial
            else:
                # A new actor on a forked drive uses the canonical memory root, not
                # a copied source or the predecessor's capability/context object.
                ctx = ToolContext(repo_dir=repo, drive_root=fork, task_id=task["id"],
                    current_chat_id=task["chat_id"], task_contract=task["task_contract"],
                    task_metadata={**task["metadata"], "budget_drive_root": str(data)})
                registry = ToolRegistry(repo_dir=repo, drive_root=fork)
                registry.set_context(ctx)
                assert "memory_read" in {schema["function"]["name"] for schema in registry.schemas()}
                spec = memory_view.view_spec_for_task(task, data, ctx=ctx)
                room = memory_view.render_room(memory_view.capture_memory_view(data, task, spec))
                history = registry.execute("chat_history", {"count": 100})
                rows = registry.execute("memory_read", {"rows": True})
                refs = [_context_ref(view) for view in (room, history, rows)]
                assert refs == [{key: value for key, value in source_ref.items()
                                 if key not in {"read", "canonical_root"}}] * 3
                for view in (room, history, rows):
                    assert "not new speech or current provider truth" in view
                source = registry.execute("memory_read", {"source_ref": refs[0]})
                assert QUESTION in source and CANCELLED in source and "rate_limited" in source
                assert "remaining-page" in source and "2026-10-15T10:00:00Z" in source
                observed = json.loads(source[source.index("{"):])
                assert observed["transport"]["conversation"] == event.conversation
                reads.append(source)
            _terminal(data, task, "Observed without a reply.")
            return [{"type": "presence_result", "outcome": "silent", "text": "", "work_ref": ""}]

    def turn(observation):
        return run_presence_turn(admission=admission, event=observation, repo_dir=repo,
            drive_root=data, agent_factory=lambda **_kwargs: Agent(), gate=PresenceTurnGate(2))

    first = turn(event)
    assert first.outcome == "silent"
    original = _rows(data)[0]
    source_ref = original["context_source_ref"]
    raw = read_actor_source_bytes(data, tasks[0]["id"], source_ref)
    payload = json.loads(raw)
    assert payload == {"recorded_at": original["ts"], "source_event_id": event.source_event_id,
                       "text": event.text, "transport": original["transport"]}
    assert len(raw) < len(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    assert original["ts"] != event.message["sent_at"]  # chat-chain address stays the host's row time
    address = chat_chain.row_address(original)
    assert original["text"] == event.text and len(_rows(data)) == 1  # history is not new speech
    if packed:
        from ouroboros.terminal_projection import SETTLEMENT_SETTLED, settle_terminal_projection

        assert settle_terminal_projection(data, tasks[0]["id"], task=tasks[0]) == SETTLEMENT_SETTLED
        result = evidence_packer.run_pass(data)  # the row stays addressable in the store; chat is never rotated
        assert result["packed"] == 1 and result["faults"] == 0, result
        source_path = task_artifact_dir_path(data, tasks[0]["id"], create=False) / source_ref["path"]
        assert not source_path.exists(), "the test must actually read through the packed-source resolver"
    # Same-id replay neither re-logs nor replaces the frozen context with a new provider observation.
    replay = turn(replace(event, conversation={"title": "Changed provider state"}))
    assert replay.task_id == first.task_id and len(tasks) == 1 and len(_rows(data)) == 1
    assert read_actor_source_bytes(data, tasks[0]["id"], source_ref) == raw
    second = turn(replace(event, source_event_id="telegram:bot-1:43", text="A later observation.",
                          conversation={"title": "Community"}, message={"message_id": "43"}))
    assert second.outcome == "silent" and second.task_id != first.task_id and len(reads) == 1
    assert chat_chain.resolve_row(data, address)[0] == original
    assert [row["text"] for row in _rows(data)] == [event.text, "A later observation."]


def test_failed_source_retention_publishes_no_address_and_runs_no_actor(tmp_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("source storage unavailable")

    monkeypatch.setattr(chat_chain, "retain_memory_source", unavailable)
    with pytest.raises(OSError, match="source storage unavailable"):
        run_presence_turn(admission=_admission(), event=_observed_event(), repo_dir=tmp_path,
            drive_root=tmp_path, agent_factory=lambda **_kwargs: pytest.fail("actor ran before retention"),
            gate=PresenceTurnGate(1))
    assert not dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")


def test_ordinary_and_outgoing_rows_do_not_gain_incoming_context_speech():
    assert dialogue_text({"direction": "in", "text": "ordinary owner input"}) == "ordinary owner input"
    assert dialogue_text({"direction": "out", "text": "I replied.",
                          "context_source_ref": {"kind": "task_source"}}) == "I replied."
