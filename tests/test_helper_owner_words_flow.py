"""The owner's words ride down the task tree by value, and keep arriving while the tree works.

``schedule_subagent`` computes them once from what the parent holds and carries them
in the existing tree-origin channel (``origin_metadata``) and the child's requested
record. A child passes on exactly what it inherited, so a grandchild sees the root's
words unchanged. A row that reaches the root later is carried by the host to every
live member at the root's drain, by identity, once (``owner_relay.carry_to_members``).
The child's own owner corpus stays its assignment: the root's words are a fact in its
view, never its owner requirements or an owner-door stamp. Every rule is checked in
both directions.
"""
from __future__ import annotations

import ast
import json
import pathlib
import queue
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store, owner_words as ow
from ouroboros.chat_chain import format_address, row_address
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.dialogue_provenance import run_origin
from ouroboros.loop_delivery import _effective_delivery_criteria
from ouroboros.loop_messages import _initialize_owner_directives, _record_owner_directive
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.memory_view import _owner_words_block
from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message
from ouroboros.owner_relay import held_owner_messages, owner_quote, write_relay
from ouroboros.task_results import load_task_result
from ouroboros.tools import control
from ouroboros.tools.control_scheduling import _schedule_task
from ouroboros.tools.registry import ToolContext
from supervisor.task_dispatch import build_scheduled_task_payload
from tests.test_available_subagents_runtime import _api_row, _settings

REPO = pathlib.Path(__file__).resolve().parents[1]
ASKED = "Build the quarterly report\nwith the regional split"
LATER = "Add the charts too"
LATEST = "Use last week's numbers"
ORIGIN_REF = {"chat_id": 1, "ts": "2026-10-04T05:00:00+00:00", "client_message_id": "msg-1"}
CARRIED = "[Owner words quoted verbatim; {ts}; Main; carried to this work by the host ({fact}); handle {address}]"


@pytest.fixture(autouse=True)
def _one_api_actor(monkeypatch):
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row()))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")


def _root(tmp_path, *, metadata=None, owner_said=True):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_task_type="task")
    ctx.task_id, ctx.event_queue = "root", queue.Queue()
    ctx.task_metadata = {"root_task_id": "root", **(metadata if metadata is not None else
                                                      {"origin_message_ref": dict(ORIGIN_REF)})}
    ctx.task_contract = build_task_contract({"id": "root", "type": "task", "text": ASKED})
    if owner_said:
        _initialize_owner_directives(ctx, [{"role": "user", "content": ASKED}])
        _record_owner_directive(ctx, source="owner_mailbox", content=LATER, msg_id="mail-1")
    return ctx


def _schedule(ctx, objective="Collect the Q3 figures", subagent_id="api-builder", **extra):
    """One real schedule: the tool's event, its requested record and the supervisor's payload."""
    result = _schedule_task(ctx, subagent_id=subagent_id, objective=objective,
                            expected_output="figures", memory_mode="empty", **extra)
    assert not result.startswith("⚠️"), result
    event = ctx.event_queue.get_nowait()
    stored = load_task_result(ctx.budget_drive_root or ctx.drive_root, event["task_id"])
    payload = build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": ctx.task_id,
                                            "text": event["objective"], "desc": event["objective"]})
    return event, stored, payload


def _child(tmp_path, payload):
    """The child's run as its worker builds it: metadata from the payload, corpus from its first user turn."""
    ctx = ToolContext(repo_dir=tmp_path, drive_root=pathlib.Path(payload["drive_root"]),
                      budget_drive_root=str(tmp_path), current_task_type="task")
    ctx.task_id, ctx.task_depth, ctx.event_queue = payload["id"], payload["depth"], queue.Queue()
    ctx.task_metadata, ctx.task_contract = dict(payload["metadata"]), payload["task_contract"]
    _initialize_owner_directives(ctx, [{"role": "user", "content": payload["text"]}])
    return ctx


def _late_row(tmp_path, text=LATEST, client_message_id="msg-2", ts="2026-10-04T06:00:00+00:00"):
    """A later message of my human: its chat row saved, its delivery in the root's mailbox; its address."""
    row = {"chat_id": 1, "direction": "in", "ts": ts, "text": text, "client_message_id": client_message_id}
    dialogue_store.append(tmp_path, "chat", row)
    write_owner_message(tmp_path, text, "root", msg_id=f"mail-{client_message_id}", client_message_id=client_message_id)
    return row, format_address(row_address(row))


def _drain(ctx, messages=None):
    """One mailbox drain of ``ctx`` as its loop runs it; ``(messages, controls)``."""
    messages = [] if messages is None else messages
    seen = getattr(ctx, "_loop_mailbox_seen_ids", None)
    seen = seen if isinstance(seen, set) else set()
    return messages, _drain_incoming_messages(messages, queue.Queue(), ctx.drive_root, ctx.task_id, None, seen,
                                              owner_ctx=ctx)


def _mail(ctx):
    return drain_owner_entries(ctx.drive_root, ctx.task_id, include_acknowledged=True)


def _carried_line(row, address, fact="reached this work"):
    return CARRIED.format(ts=row["ts"], fact=fact, address=address)


def _texts(rows):
    return [row["text"] for row in rows]


def _holds(value, text):
    """Whether ``text`` appears in the JSON form of ``value`` (its own JSON escaping, newlines included)."""
    return json.dumps(text, ensure_ascii=False)[1:-1] in json.dumps(value, ensure_ascii=False, default=str)


def test_the_owner_words_ride_the_event_the_record_and_the_payload(tmp_path):
    event, stored, payload = _schedule(_root(tmp_path))
    assert _texts(event["origin_metadata"][ow.FIELD]) == [ASKED, LATER]
    assert ow.ABSENT_FIELD not in event["origin_metadata"]
    assert _texts(stored[ow.FIELD]) == [ASKED, LATER]  # the requested record any reader projects
    assert _texts(payload["metadata"][ow.FIELD]) == [ASKED, LATER]  # task.metadata of the child
    assert stored[ow.FIELD] == event["origin_metadata"][ow.FIELD] == payload["metadata"][ow.FIELD]
    first = stored[ow.FIELD][0]
    assert first["source"] == "initial_user" and first["task_id"] == "root" and first["ts"] == ORIGIN_REF["ts"]
    # The child's view renders exactly these rows; a child scheduled without the field shows nothing.
    from ouroboros.memory_view import _owner_words_block

    block = _owner_words_block(payload)
    assert block.startswith("## My human's words for this work (verbatim, dated, host-attested)\n") and "  " + LATER in block
    bare = {**payload, "metadata": {key: value for key, value in payload["metadata"].items() if key != ow.FIELD}}
    assert _owner_words_block(bare) == ""


def test_a_grandchild_receives_the_root_words_and_what_its_parent_received_since(tmp_path):
    root = _root(tmp_path)
    _event, _stored, payload = _schedule(root)
    child = _child(tmp_path, payload)
    grand_event, grand_stored, grand_payload = _schedule(child, objective="Check one region")
    assert grand_payload["metadata"][ow.FIELD] == payload["metadata"][ow.FIELD]
    assert grand_stored[ow.FIELD] == grand_event["origin_metadata"][ow.FIELD] == payload["metadata"][ow.FIELD]
    assert not _holds(grand_payload["metadata"][ow.FIELD], "Collect the Q3 figures")
    # The child's own corpus would give its assignment; what it passes on is what it inherited.
    assert _texts(ow.directive_owner_rows(child)) == []
    child.task_metadata.pop(ow.FIELD)
    child.task_metadata[ow.ABSENT_FIELD] = "not_recorded"
    _e, _s, orphan = _schedule(child, objective="Check another region")
    assert ow.FIELD not in orphan["metadata"] and orphan["metadata"][ow.ABSENT_FIELD] == "not_recorded"
    # A row the root drains later reaches the child at its own drain and rides on to a helper scheduled after it.
    child.task_metadata = dict(payload["metadata"])
    _row, address = _late_row(tmp_path)
    _drain(root)
    _drain(child)
    _e, _s, later = _schedule(child, objective="Check the latest numbers")
    assert _texts(later["metadata"][ow.FIELD]) == [ASKED, LATER, LATEST]
    assert later["metadata"][ow.FIELD][2]["ref"] == "chat 1 / msg-2" and later["metadata"][ow.FIELD][2]["carrier"] == "held"
    assert _texts(ow.directive_owner_rows(child)) == []  # still nothing of its own
    assert not _mail(_child(tmp_path, later))  # born with the row: the child's next drain writes it nothing
    _drain(child)
    assert not _mail(_child(tmp_path, later))


def test_a_late_row_reaches_a_running_child_and_grandchild_once_each_outside_their_corpus(tmp_path):
    """Both are written at the root's drain; each shows the row once at its own next drain, dated and labelled,
    with its corpus, its delivery criteria and every earlier message untouched."""
    root = _root(tmp_path)
    _e, _s, payload = _schedule(root)
    child = _child(tmp_path, payload)
    _e, _s, grand_payload = _schedule(child, objective="Check one region")
    grand = _child(tmp_path, grand_payload)
    row, address = _late_row(tmp_path)
    _drain(root)
    assert [entry["msg_id"] for entry in _mail(child)] == [f"relay:carry:{address}:{child.task_id}"]
    assert [entry["msg_id"] for entry in _mail(grand)] == [f"relay:carry:{address}:{grand.task_id}"]
    for member, request in ((child, payload), (grand, grand_payload)):
        corpus, criteria = json.dumps(member._owner_directives, default=str), _effective_delivery_criteria(member)
        messages = [{"role": "user", "content": request["text"]}, {"role": "assistant", "content": "Working."}]
        earlier = json.dumps(messages)
        _messages, controls = _drain(member, messages)
        assert controls["owner_carried_addressed"] == [address] and len(messages) == 3
        assert json.dumps(messages[:2]) == earlier
        assert str(messages[2]["content"]) == f"{_carried_line(row, address)}\n{LATEST}"
        assert "owner_quote_handles" not in str(messages) and "[Message from Ouroboros" not in str(messages)
        assert json.dumps(member._owner_directives, default=str) == corpus
        assert _effective_delivery_criteria(member) == criteria
        assert held_owner_messages(member)[address]["carried"]["from"] == "root"
        _messages, controls = _drain(member, messages)
        assert "owner_carried_addressed" not in controls and len(messages) == 3
    assert len(_mail(grand)) == 1  # the child's drain over its own children wrote the grandchild nothing more


def test_the_first_drain_shows_nothing_the_first_request_holds_and_an_absent_row_whatever_its_age(tmp_path):
    """Identity, never time: a row whose reference stands in the member's first request renders no words (its
    room's stamp earns the one host fact line); a row absent from that request is shown once even when it is
    older than the member's start."""
    root = _root(tmp_path)
    row, address = _late_row(tmp_path)
    _drain(root)  # the root read the row: a helper scheduled now is born with it
    _e, _s, payload = _schedule(root)
    born_with = _child(tmp_path, payload)
    request = [{"role": "system", "content": _owner_words_block(payload, tmp_path)},
               {"role": "user", "content": payload["text"]}]
    assert LATEST in request[0]["content"] and "chat 1 / msg-2" in request[0]["content"]
    _drain(root)  # a root alive across the update: nothing written for a row the member was born with
    assert not _mail(born_with)
    write_relay(born_with.drive_root, born_with.task_id, relay_id=f"carry:{address}", note="", chat_id=1,
                quotes=[owner_quote(tmp_path, row, address=address)], teller_task_id="root", canonical_root=tmp_path,
                host_carried={"went_to": ""})
    messages, controls = _drain(born_with, list(request))
    assert len(messages) == 2 and "owner_carried_addressed" not in controls
    assert held_owner_messages(born_with)[address]["carried"]["went_to"] == ""
    # Under its room's stamp (a birth block that read the row from the room): the one host fact line, no words.
    room_read = _child(tmp_path, payload)
    room_read.task_id = payload["id"] + "-room"
    write_relay(room_read.drive_root, room_read.task_id, relay_id=f"carry:{address}", note="", chat_id=1,
                quotes=[owner_quote(tmp_path, row, address=address)], teller_task_id="root", canonical_root=tmp_path,
                host_carried={"went_to": ""})
    stamped = ow.render_owner_words([ow._row(LATEST, source="room_message", carrier="room", ts=row["ts"],
                                             ref="chat 1 / msg-2")], audience="child", root_task_id="root")
    messages, controls = _drain(room_read, [{"role": "system", "content": stamped}, {"role": "user", "content": payload["text"]}])
    assert [m["content"] for m in messages[2:]] == [
        f"[Host fact: the row {address} in your starting words was addressed to this work]"]
    assert controls["owner_carried_addressed"] == [address]
    # The other direction: an older row absent from the first request is shown once, with no time compared.
    older, older_address = _late_row(tmp_path, "Before anything else", "msg-0", "2026-10-04T04:00:00+00:00")
    _drain(root)
    assert [entry["msg_id"] for entry in _mail(born_with)][-1] == f"relay:carry:{older_address}:{born_with.task_id}"
    messages, controls = _drain(born_with, list(request))
    assert controls["owner_carried_addressed"] == [older_address] and len(messages) == 3
    assert str(messages[2]["content"]) == f"{_carried_line(older, older_address)}\nBefore anything else"


def test_a_parent_writes_one_entry_to_each_member_born_without_the_row_and_raises_nothing(tmp_path):
    """A grandchild that already has the entry from the root's drain and a sibling born since (before the child
    read the row) each hold exactly one entry after the child's drain; the sibling shows it under the root's name."""
    root = _root(tmp_path)
    _e, _s, payload = _schedule(root)
    child = _child(tmp_path, payload)
    _e, _s, grand_payload = _schedule(child, objective="Check one region")
    grand = _child(tmp_path, grand_payload)
    row, address = _late_row(tmp_path)
    _drain(root)
    _e, _s, sibling_payload = _schedule(child, objective="Check a second region")  # the child has not read the row
    sibling = _child(tmp_path, sibling_payload)
    assert not _holds(sibling_payload["metadata"][ow.FIELD], LATEST) and not _mail(sibling)
    messages, controls = _drain(child)
    assert controls["owner_carried_addressed"] == [address]
    assert [entry["msg_id"] for entry in _mail(grand)] == [f"relay:carry:{address}:{grand.task_id}"]
    assert [entry["msg_id"] for entry in _mail(sibling)] == [f"relay:carry:{address}:{sibling.task_id}"]
    assert _mail(sibling)[0]["source_task_id"] == "root"
    _drain(child)
    assert len(_mail(grand)) == 1 and len(_mail(sibling)) == 1
    messages, controls = _drain(sibling, [{"role": "user", "content": sibling_payload["text"]}])
    assert controls["owner_carried_addressed"] == [address] and len(messages) == 2
    assert str(messages[1]["content"]) == f"{_carried_line(row, address)}\n{LATEST}"


def _carried_mail(tmp_path, task_id, row, address, *, went_to=""):
    return write_relay(tmp_path, task_id, relay_id=f"{'carry-context' if went_to else 'carry'}:{address}", note="",
                       quotes=[owner_quote(tmp_path, row, address=address)], chat_id=1, teller_task_id="root",
                       canonical_root=tmp_path, host_carried={"went_to": went_to})


@pytest.mark.parametrize("standing", ["addressed", "context"])
def test_a_closing_helper_holds_its_final_once_for_an_addressed_row_and_not_for_a_context_row(tmp_path, monkeypatch, standing):
    """The delivery close reads a carried row: addressed, the final is held once and one round reads it, with the
    corpus, its hash and the seal untouched; context, nothing holds. A sealed close leaves the row unread mail."""
    import threading
    from ouroboros.loop_messages import owner_source_sha256
    from tests.test_completion_hold_observation import holds, runner, sha
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    turn = runner(tmp_path, monkeypatch, task_id="helper")
    turn.registry._ctx.is_direct_chat = False
    turn.registry._ctx.task_metadata = {"delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root"}
    turn.registry._ctx.owner_message_admission_lock = threading.Lock()
    turn.registry._ctx.owner_message_admission_agent = SimpleNamespace(_busy=True, _current_task_id="helper",
                                                                        _accepting_owner_messages=True)
    row, address = _late_row(tmp_path)
    hashes = []

    def first():
        hashes.append(owner_source_sha256(turn.registry._ctx))
        _carried_mail(tmp_path, "helper", row, address, went_to="" if standing == "addressed" else "sibling")
        return {"role": "assistant", "content": "Initial draft"}

    text, _usage, _trace = turn.run([first, {"role": "assistant", "content": "Draft after reading"}])
    assert owner_source_sha256(turn.registry._ctx) == hashes[0]  # the acknowledgement hash never moved
    if standing == "addressed":
        assert text == "Draft after reading" and len(turn.calls) == 2
        [row_held] = holds(tmp_path)
        assert (row_held["reason"], row_held["subreason"], row_held["answer_sha256"]) == (
            "owner_followup", "owner_carried", sha("Initial draft"))
        assert any(_carried_line(row, address) in str(m.get("content")) for m in turn.calls[1])
    else:
        assert text == "Initial draft" and len(turn.calls) == 1 and not holds(tmp_path)
    assert not drain_owner_entries(tmp_path, "helper")  # read: the entry is acknowledged either way
    # A sealed close leaves an addressed row as unread mail: not shown, not acknowledged, read by the next drain.
    sealed = _child(tmp_path, {"id": "sealed", "depth": 1, "drive_root": str(tmp_path), "text": "Assignment",
                               "metadata": {"delegation_role": "subagent", "root_task_id": "root"}, "task_contract": {}})
    _carried_mail(tmp_path, "sealed", row, address)
    sealed._task_acceptance_sealed_fence_token = "sealed"
    messages, controls = _drain(sealed)
    assert messages == [] and "owner_carried_addressed" not in controls
    assert [entry["msg_id"] for entry in drain_owner_entries(tmp_path, "sealed")] == [f"relay:carry:{address}:sealed"]
    sealed._task_acceptance_sealed_fence_token = None
    messages, controls = _drain(sealed)
    assert controls["owner_carried_addressed"] == [address] and len(messages) == 1
    assert not drain_owner_entries(tmp_path, "sealed")


@pytest.mark.parametrize("rail", ["one_turn", "two_turns", "provider_outcome_unknown"])
def test_the_forced_rail_never_replaces_a_helpers_final_with_the_fallback_for_a_carried_row(tmp_path, monkeypatch, rail):
    """A carried row read after the forced call is no directive: a one-turn rail keeps the forced final, a two-turn
    rail asks once more with the row in view, and a two-turn rail whose forced call left the provider outcome
    unknown keeps its draft; none buys the resume or the no-resend fallback, and the corpus does not grow."""
    from tests.test_delivery_forced_finalization import _forced_test_context
    incoming = queue.Queue()
    loop, registry, ctx, _trace = _forced_test_context(tmp_path, incoming=incoming)
    registry._ctx.task_metadata["delegation_role"] = "subagent"
    row, address = _late_row(tmp_path)
    calls = 0

    def forced_model(_llm, messages, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            _carried_mail(tmp_path, "parent1", row, address)
            if rail == "provider_outcome_unknown":
                ctx.accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
        else:
            assert any(_carried_line(row, address) in str(m.get("content")) for m in messages)
            assert "[FORCED_OWNER_REFRESH]" in str(messages[-1]["content"])
        return {"role": "assistant", "content": f"forced draft {calls}"}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)
    from ouroboros.loop_messages import owner_source_sha256
    corpus = owner_source_sha256(registry._ctx)
    text, _usage, returned_trace = loop._forced_final_answer(
        ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
        single_semantic_turn=rail == "one_turn")
    assert calls == (2 if rail == "two_turns" else 1) and text == f"forced draft {calls}"
    assert "Resume the task" not in text and "blind resend" not in text
    assert returned_trace["forced_finalization"].get("source") not in {"late_owner_directive_requires_resume",
                                                                        "provider_outcome_unknown_no_resend"}
    assert owner_source_sha256(registry._ctx) == corpus and not getattr(registry._ctx, "_owner_directives", None)
    assert held_owner_messages(registry._ctx)[address]["carried"]["went_to"] == ""


def test_a_consciousness_root_carries_the_raw_marker_and_no_words(tmp_path):
    ctx = _root(tmp_path, metadata={"initiator": "consciousness", "usage_category": "consciousness_task"},
                owner_said=False)
    _initialize_owner_directives(ctx, [{"role": "user", "content": "A wake decided to look at the logs"}])
    event, stored, payload = _schedule(ctx)
    assert event["origin_metadata"][ow.ABSENT_FIELD] == "initiator=consciousness"
    assert ow.FIELD not in event["origin_metadata"] and ow.FIELD not in stored
    assert payload["metadata"][ow.ABSENT_FIELD] == "initiator=consciousness"
    assert payload["metadata"]["initiator"] == "consciousness"  # the consciousness origin still rides along
    assert ow.ABSENT_FIELD not in _schedule(_root(tmp_path))[2]["metadata"]


def test_a_swarm_roots_words_stay_exact_with_ignored_legacy_metadata(tmp_path, monkeypatch):
    """The root and child receive the door's exact text, without the retired host initiative."""
    from ouroboros.context import build_user_content
    from ouroboros.review_evidence_sections import _owner_content_projection

    door = {"origin_message_ref": dict(ORIGIN_REF), "origin_message_text": ASKED,
            "subagents_mode": "swarm", "force_plan": True, "force_plan_source": "swarm"}
    turn = build_user_content({"text": ASKED, "metadata": door})
    assert _owner_content_projection(turn) == ASKED
    swarm = _root(tmp_path, metadata=door, owner_said=False)
    _initialize_owner_directives(swarm, [{"role": "user", "content": turn}])
    _event, stored, payload = _schedule(swarm)
    assert _texts(payload["metadata"][ow.FIELD]) == [ASKED] and _texts(stored[ow.FIELD]) == [ASKED]
    assert payload["metadata"][ow.FIELD][0]["ref"] == "chat 1 / msg-1"  # still the door's stamp
    assert "[SWARM_INITIATIVE]" not in ow.owner_words_text(swarm)
    assert not _holds(swarm._owner_directives, "[SWARM_INITIATIVE]")
    # The other side: a door that kept no text leaves the corpus row's own projection.
    bare = _root(tmp_path, metadata={key: value for key, value in door.items() if key != "origin_message_text"},
                 owner_said=False)
    _initialize_owner_directives(bare, [{"role": "user", "content": turn}])
    assert _texts(ow.directive_owner_rows(bare)) == [_owner_content_projection(turn)]


def test_the_child_corpus_stays_its_assignment_and_no_owner_door(tmp_path):
    root = _root(tmp_path)
    _event, _stored, payload = _schedule(root)
    child = _child(tmp_path, payload)
    assert [row["source"] for row in child._owner_directives] == ["initial_text"]
    assert not _holds(child._owner_directives, ASKED) and _holds(child._owner_directives, "Collect the Q3 figures")
    assert run_origin({"metadata": child.task_metadata})["owner_ingress"] is False
    assert run_origin({"metadata": root.task_metadata})["owner_ingress"] is True
    assert not _holds(_effective_delivery_criteria(child), ASKED)
    assert not _holds(_effective_delivery_criteria(child), LATER)
    assert _holds(_effective_delivery_criteria(root), ASKED) and _holds(_effective_delivery_criteria(root), LATER)
    # The words a reviewer or a session of this run sees: the child's inherited, the root's own corpus.
    assert ASKED in ow.owner_words_text(child) and LATER in ow.owner_words_text(child)
    assert "Collect the Q3 figures" not in ow.owner_words_text(child)
    session = ow.owner_words_text(root, audience="session")
    assert ASKED in ow.owner_words_text(root) and session.startswith(ow._HEADINGS["session"] + "\n" + ow._FRAMES["session"])
    assert "of task root ·" in session and "from task root" not in session  # the stamp names the task, the frame no longer


def test_the_parent_reads_the_childs_words_from_the_childs_record(tmp_path):
    from ouroboros.subagent_work_order import _source_task_from_context
    from ouroboros.task_results import write_task_result

    root = _root(tmp_path)
    _event, stored, _payload = _schedule(root)
    seen = _source_task_from_context(root, stored["task_id"])
    assert seen[ow.FIELD] == stored[ow.FIELD]  # the parent's own metadata carries no field
    root.task_metadata[ow.FIELD] = [{"text": "The parent's own inherited words", "source": "initial_user"}]
    assert _source_task_from_context(root, stored["task_id"])[ow.FIELD] == stored[ow.FIELD]
    write_task_result(tmp_path, "nofield", "requested", description="a child scheduled before the field")
    assert ow.FIELD not in _source_task_from_context(_root(tmp_path), "nofield")


def test_schedule_task_stays_within_the_function_ceiling():
    tree = ast.parse((REPO / "ouroboros" / "tools" / "control_scheduling.py").read_text(encoding="utf-8"))
    node = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == "_schedule_task")
    assert node.end_lineno - node.lineno + 1 <= 300
