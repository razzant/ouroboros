"""A promoted Place root is told where its chosen final goes; nothing else is.

Real Place admission and focus construction (as ``tests.test_presence_promises._turn``), the real
``start_task`` producer and supervisor promote admission, the real scheduler for a fired promise,
``build_llm_messages`` and the registered ``finish_task`` schema. The section is rendered only when
the same ``reply_carrier`` that speaks a root's final exists and no FOCUS frame or schedule rule
already governs it. Synthetic fixture data only; nothing is sent.
"""
from __future__ import annotations

import json
import types

import pytest

from tests.test_presence_promises import _due_promise, _turn
from tests.test_presence_promises import fired_context as fired_context  # fixture


HEADING = "## Where your final answer goes (host-authored facts)"
THREAD = "t-42"


def _source_turn(root, thread_id):
    """Real admission and focus construction, including the known empty (no-thread) value."""
    from dataclasses import replace

    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import PresenceTurnEvent, _build_task
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="team", conversation_id="k1", thread_id=thread_id)
    event = PresenceTurnEvent("evt-k1-t42", "chat", "team", "k1", thread_id, conversation_key("chat", "team", "k1", thread_id),
                              {"platform_actor_id": "person"}, {}, {}, "hello", facts=("mention",))
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}), event,
                       drive_root=root, staged_files=())
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=task["id"], task_metadata=dict(task["metadata"]),
                      task_contract=task["task_contract"], is_direct_chat=True, current_chat_id=task["chat_id"])
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry.set_context(ctx)
    return registry, task


def _promote(root, monkeypatch, turn_task, registry):
    """The focus turn's own start_task, admitted by the supervisor's promote handler."""
    import supervisor.queue as queue
    import supervisor.workers as workers
    from ouroboros.tools.control_routing import _start_task
    from supervisor.events import _handle_promote_chat_to_task
    from tests.test_promote_chat_flow import _confirm_promote

    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue, "DRIVE_ROOT", str(root))
    _confirm_promote(monkeypatch)
    pending, emitted = [], []
    handler = types.SimpleNamespace(
        DRIVE_ROOT=root, WORKERS={0: types.SimpleNamespace()}, PENDING=pending, bridge=None,
        append_jsonl=lambda *_a, **_k: None, persist_queue_snapshot=lambda **_k: True,
        enqueue_task=lambda task: pending.append(dict(task)) or pending[-1], load_state=lambda: {"owner_chat_id": 1})
    ctx = registry._ctx
    ctx.event_queue = types.SimpleNamespace(
        put_nowait=lambda event: emitted.append(event) or _handle_promote_chat_to_task(event, handler))
    out = _start_task(ctx, "Compile the requested report", title="Report", destination="place")
    assert out.startswith("OK: task"), out
    [evt], [promoted] = emitted, pending
    assert evt["presence"]["event"] == turn_task["metadata"]["presence"]["event"]  # the speaker's carrier
    assert promoted["metadata"]["presence"]["event"] == turn_task["metadata"]["presence"]["event"]
    return promoted


def _section(text):
    return text.count(HEADING)


@pytest.mark.parametrize("thread_id", [THREAD, ""])
def test_a_promoted_place_root_names_its_retained_conversation(fired_context, monkeypatch, thread_id):
    from ouroboros.place_reply import reply_carrier

    root, _pending, first_request = fired_context
    registry, turn = _source_turn(root, thread_id)
    route = json.dumps(dict(account_id="team", conversation_id="k1", provider="chat", thread_id=thread_id), sort_keys=True)
    key = f"chat:team:k1:{thread_id or '0'}"
    assert turn["metadata"]["presence"]["event"]["thread_id"] == thread_id  # the producer's source thread
    promoted = _promote(root, monkeypatch, turn, registry)
    retained = promoted["metadata"]["presence"]["event"]
    assert (retained["thread_id"], retained["conversation_key"]) == (thread_id, key)  # admission
    assert not promoted.get("_presence_turn") and reply_carrier(promoted["metadata"])["conversation_key"] == key
    rendered = first_request(promoted)  # the request messages as JSON text
    assert _section(rendered) == 1
    body = rendered.split(HEADING, 1)[1].split("##", 1)[0]
    # The exact source thread, never the place's configured filter/destination or a guessed DM.
    assert route.replace('"', '\\"') in body and "unrecorded" not in body
    assert promoted["metadata"]["presence"]["event"]["origin"]["conversation_id"] == "*"
    assert '\\"*\\"' not in body  # the admitting filter's wildcard is not an address
    for words in ("new speech there, even after earlier sends", "selecting it is not delivering it",
                  "not a private channel instead of the conversation", "does not discharge a reply someone is owed",
                  "not proof of who owns this work"):
        assert words in body, words
    assert "A wake is your attention" not in rendered  # no FOCUS frame on a promoted root

    # Cold rebuild: the persisted (JSON, underscore-free) task renders the same section.
    cold = {key: value for key, value in json.loads(json.dumps(promoted)).items() if not key.startswith("_")}
    cold_text = first_request(cold)
    assert _section(cold_text) == 1 and route.replace('"', '\\"') in cold_text.split(HEADING, 1)[1].split("##", 1)[0]

    # Negative control: the actual builder without the context wiring omits it on this same root.
    import ouroboros.presence_context as presence_context
    real = presence_context.build_presence_context_section
    monkeypatch.setattr(presence_context, "build_presence_context_section",
                        lambda *a, answers_conversation=False, **k: real(*a, **k))
    assert _section(first_request(promoted)) == 0


def test_inline_scheduled_authority_only_and_missing_carriers_get_no_audience_section(fired_context, monkeypatch):
    from supervisor.queue import check_scheduled_tasks

    root, pending, first_request = fired_context
    registry, turn, binding = _turn(root, "team", "k1")
    inline = first_request(turn)
    assert "A wake is your attention" in inline and _section(inline) == 0  # FOCUS already frames it; no duplicate

    _due_promise(registry, "Check back on the report")
    check_scheduled_tasks()
    [fired] = pending
    scheduled = first_request(fired)
    assert "nothing it finishes with is sent to the conversation by itself" in scheduled and _section(scheduled) == 0

    child = {"id": "child-1", "delegation_role": "subagent", "parent_task_id": turn["id"], "root_task_id": turn["id"],
             "type": "task", "text": "Count the rows", "metadata": {"presence_binding_authority": {"binding_id": binding}}}
    assert _section(first_request(child)) == 0
    lost = json.loads(json.dumps(turn["metadata"]["presence"]))
    lost["event"].pop("conversation_key")
    for presence in (lost, {"place_id": turn["metadata"]["presence"]["place_id"]}):
        assert _section(first_request({"id": "root-x", "type": "task", "text": "Report",
                                       "metadata": {"presence": presence}})) == 0


def test_missing_route_fields_stay_explicit(tmp_path):
    from ouroboros.presence_context import build_presence_context_section

    value = {"place_id": "place-x", "event": {"provider": "chat", "conversation_id": "k1", "conversation_key": "chat::k1:0"}}
    section = build_presence_context_section(tmp_path, value, "root-y", answers_conversation=True)
    assert '{"account_id": "unrecorded", "conversation_id": "k1", "provider": "chat", "thread_id": "unrecorded"}' in section
    assert section.count(HEADING) == 1 and build_presence_context_section(tmp_path, value, "root-y") == ""


def test_the_registered_finish_schema_does_not_claim_an_owner_only_reader(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    [schema] = [s for s in ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path).schemas()
                if s["function"]["name"] == "finish_task"]
    answer = schema["function"]["parameters"]["properties"]["answer"]["description"]
    assert "a root's owner reads it as your reply" not in answer
    assert "a child's parent reads it as your report" in answer
    assert "the conversation a Presence root answers" in answer
