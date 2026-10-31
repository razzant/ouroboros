"""The routing rail tells the truth about what it did (14.09 incident).

A Project root steered the Main root; the supervisor wrote a refusal receipt and
the model was told `STEER_UNCONFIRMED`. Four separate lies produced that one
sentence, and each is pinned here against the PRODUCER's own output:

* the tool waited on an EMPTY receipt id, so a task-authored steer could never be
  confirmed — refused or delivered;
* the wait's accept-set had no `rejected`, so a refusal the host had already
  written durably was polled to the deadline and reported as a timeout;
* the promote/route refusal sentences were not in the identifier register (and
  the promotion ones carried no marker at all), so a routing act that scheduled
  nothing counted as a SUCCESSFUL tool call in the outcome classifier;
* an admitted promote named the project it ASKED for rather than the one
  admission put the task in.

Plus the sibling projection (#902): after an ADOPTING conversion the message's
other live cards still offered a second "turn into project".
"""

from __future__ import annotations

import time
import types

import pytest


def _tool_ctx(tmp_path, metadata=None, **kwargs):
    """A routing turn with no supervisor: events land in ``pending_events``."""
    return types.SimpleNamespace(
        pending_events=[],
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_id="turn-1",
        task_metadata=dict(metadata or {}),
        last_owner_delivery=None,
        **kwargs,
    )


def _supervisor_ctx(tmp_path, notices, running=None):
    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING=dict(running or {}),
        PENDING=[],
        send_with_budget=lambda _chat_id, text, *a, **k: notices.append(text),
    )


def _live_tool_ctx(tmp_path, supervisor_ctx, emitted, metadata=None):
    """A routing turn wired to the REAL supervisor handler, so the tool's durable
    receipt wait sees exactly what the handler persisted."""
    from supervisor.events import _handle_steer_task

    ctx = _tool_ctx(tmp_path, metadata=metadata)
    ctx.event_queue = types.SimpleNamespace(
        put_nowait=lambda event: (emitted.append(event), _handle_steer_task(event, supervisor_ctx))[0],
    )
    return ctx


# --- (a) a settled rejection is returned as one, never as a timeout -----------

def test_the_steer_wait_returns_a_written_rejection_instead_of_timing_out(tmp_path):
    """`rejected` is what the cancel-pending refusal writes. While the accept-set
    held only delivered/needs_manual_target/unconfirmed, that durable refusal was
    polled for the full window and then answered `confirmation_timeout` — the
    caller was told the host could not confirm the very thing it had confirmed."""
    from ouroboros.project_dialogue import append_chat_annotation
    from ouroboros.routing_wait import wait_for_routing_annotation

    append_chat_annotation(
        tmp_path, "cm-1", action="steer_task", target="t-target",
        status="rejected", routing_token="tok-1", reason="cancel_pending",
    )

    started = time.monotonic()
    receipt = wait_for_routing_annotation(tmp_path, "cm-1", "tok-1", timeout_sec=30.0)
    elapsed = time.monotonic() - started

    assert receipt["status"] == "rejected"
    assert receipt["reason"] == "cancel_pending"
    assert elapsed < 5.0, "a settled refusal must not be waited out"


def test_the_steer_wait_still_times_out_when_no_receipt_exists(tmp_path):
    """The timeout keeps its one meaning: nothing matching has been written."""
    from ouroboros.routing_wait import wait_for_routing_annotation

    receipt = wait_for_routing_annotation(tmp_path, "cm-1", "tok-1", timeout_sec=0.05)

    assert receipt == {"status": "unconfirmed", "reason": "confirmation_timeout"}


def test_a_newer_attempts_receipt_is_not_read_as_this_ones_rejection(tmp_path):
    """Token binding still governs: another attempt's row answers nobody."""
    from ouroboros.project_dialogue import append_chat_annotation
    from ouroboros.routing_wait import wait_for_routing_annotation

    append_chat_annotation(
        tmp_path, "cm-1", action="steer_task", target="t-target",
        status="rejected", routing_token="tok-OTHER", reason="cancel_pending",
    )

    receipt = wait_for_routing_annotation(tmp_path, "cm-1", "tok-1", timeout_sec=0.05)

    assert receipt == {"status": "unconfirmed", "reason": "confirmation_timeout"}


# --- (b) a task-authored steer has a receipt of its own -----------------------

def test_a_steer_with_no_owner_rail_is_refused_readably_not_unconfirmed(tmp_path):
    """The incident itself. A managed task steering another root carries no chat
    ingress id, so the steer went out with an empty `client_message_id`: the
    supervisor wrote no annotation, the wait answered `client_message_id_missing`
    before the handler had even run, and the model saw STEER_UNCONFIRMED for a
    steer the host had REFUSED. The act now keys its own synthetic receipt."""
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.tools.control import _tell_task

    notices, emitted = [], []
    ctx = _live_tool_ctx(tmp_path, _supervisor_ctx(tmp_path, notices), emitted, metadata={})

    out = _tell_task(ctx, "t-gone", "pick the smaller PR")

    assert out.startswith("TELL_REJECTED: t-gone: target_unknown")
    assert "UNCONFIRMED" not in out
    receipt_id = emitted[0]["client_message_id"]
    assert receipt_id == f"agent-steer:{emitted[0]['routing_token']}"
    row = latest_chat_annotations(tmp_path)[receipt_id]
    assert (row["action"], row["status"], row["reason"]) == (
        "steer_task", "needs_manual_target", "target_unknown",
    )


def test_a_task_authored_steer_refused_by_a_pending_cancel_reports_the_cause(
    tmp_path, monkeypatch,
):
    """The other refusal producer writes `rejected` rather than
    `needs_manual_target`; both must read as the same typed refusal, with the
    reason the owner and the model need to act on."""
    import ouroboros.cancel_intents as cancel_intents
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.tools.control import _tell_task

    monkeypatch.setattr(
        cancel_intents, "cancel_pending", lambda _root, task_id, **_k: task_id == "t-target",
    )
    notices, emitted = [], []
    supervisor = _supervisor_ctx(
        tmp_path, notices, running={"t-target": {"task": {"id": "t-target", "chat_id": 1}}},
    )
    ctx = _live_tool_ctx(tmp_path, supervisor, emitted, metadata={})

    out = _tell_task(ctx, "t-target", "stop after the current file")

    assert out.startswith("TELL_REJECTED: t-target: cancel_pending")
    receipt_id = emitted[0]["client_message_id"]
    assert latest_chat_annotations(tmp_path)[receipt_id]["status"] == "rejected"


def test_a_delivered_task_authored_steer_is_confirmed_under_its_own_id(tmp_path, monkeypatch):
    """The capability half of the same fix: the synthetic id must confirm a
    LANDED delivery too, or every task-authored steer would invite a retry of a
    message that already arrived. A task speaking for itself is told its message
    was WRITTEN (wave 2: it lands as a task message, never as owner text)."""
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools.control import _tell_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    notices, emitted = [], []
    supervisor = _supervisor_ctx(
        tmp_path, notices, running={"t-target": {"task": {"id": "t-target", "chat_id": 1}}},
    )
    ctx = _live_tool_ctx(tmp_path, supervisor, emitted, metadata={})

    out = _tell_task(ctx, "t-target", "keep the PR small")

    assert out.startswith("Message written to t-target")
    assert "UNCONFIRMED" not in out
    [entry] = drain_owner_entries(tmp_path, "t-target")
    assert (entry["text"], entry["kind"], entry["source_task_id"]) == (
        "keep the PR small", "relay_entry", "turn-1",
    )


def test_an_unconfirmed_act_is_kept_by_its_exact_ids_and_a_settled_one_is_not(tmp_path):
    """A steer whose receipt the wait did not see may still land, or already has: the turn
    keeps its exact identity, so a later acting round reports the recorded outcome
    (``loop_model_call._append_routing_receipts``), and the text says so instead of inviting
    a second delivery. A refused act is settled and kept nowhere."""
    from ouroboros.tools.control import _tell_task

    ctx = _tool_ctx(tmp_path, metadata={}, unconfirmed_routing_acts=[])
    out = _tell_task(ctx, "t-target", "keep the PR small")
    assert out.startswith("TELL_UNCONFIRMED: t-target:")
    assert "The request may still land." in out
    assert "Check its receipt" in out and "sending the words again" in out
    [event] = ctx.pending_events
    assert ctx.unconfirmed_routing_acts == [{
        "action": "steer_task", "client_message_id": event["client_message_id"],
        "routing_token": event["routing_token"], "target": "t-target"}]
    notices, emitted = [], []
    live = _live_tool_ctx(tmp_path, _supervisor_ctx(tmp_path, notices), emitted, metadata={})
    live.unconfirmed_routing_acts = []
    assert _tell_task(live, "t-gone", "pick the smaller PR").startswith("TELL_REJECTED")
    assert live.unconfirmed_routing_acts == []


def test_a_resumed_turn_keeps_its_own_unconfirmed_acts(tmp_path):
    """A cold continuation (an owner wait across a restart) restores the exact ids, so the
    resumed turn still reports a late outcome; a state saved before the key restores none."""
    import json

    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    act = {"action": "steer_task", "client_message_id": "agent-steer:tok", "routing_token": "tok", "target": "t-1"}
    state = json.loads(json.dumps(continuation_state(
        types.SimpleNamespace(task_id="turn-1", unconfirmed_routing_acts=[act]), [], {}, {}, 1, [], set())))
    restored = types.SimpleNamespace(task_id="turn-1", unconfirmed_routing_acts=[])  # the resumed turn's own context
    restore_continuation_state(types.SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert restored.unconfirmed_routing_acts == [act]
    state.pop("unconfirmed_routing_acts")
    restore_continuation_state(types.SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert restored.unconfirmed_routing_acts == []


# --- (c) a routing refusal is a recorded failure, not a successful call -------


def _produced_refusals(tmp_path, monkeypatch):
    from ouroboros.tools import control_events, control_routing
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda ctx: "crash_storm")
    disabled = _start_task(_tool_ctx(tmp_path), "Audit", title="Audit")
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda ctx: "")
    monkeypatch.setattr(control_events, "_wait_for_promotion_admission",
                        lambda *a, **kw: {"status": "rejected", "reason": "duplicate_task_id"})
    rejected = _start_task(_tool_ctx(tmp_path), "Audit", title="Audit")
    return {"pool_disabled": disabled, "promote_rejected": rejected}


@pytest.mark.parametrize("key,identifier", [
    ("pool_disabled", "PROMOTE_REJECTED"),
    ("promote_rejected", "PROMOTE_REJECTED"),
])
def test_a_routing_act_that_scheduled_nothing_is_not_a_successful_call(
    tmp_path, monkeypatch, key, identifier,
):
    """Each sentence is the producer's own, classified by the ONE classifier.

    Before the register carried this family, the two promotion receipts opened
    line 1 with no warning marker at all and the route/picker ones ended in a
    suffix the generic chain does not read, so every one of them was `ok`: a
    refused promotion and a scheduled task were the same fact to the outcome
    classifier and the acceptance packet.
    """
    from ouroboros.tools.tool_result import TOOL_CODE_SPECS, LegacyTextResultAdapter

    text = _produced_refusals(tmp_path, monkeypatch)[key]
    assert text.startswith(f"⚠️ {identifier}"), text

    typed = LegacyTextResultAdapter.from_text("start_task", text)
    assert (typed.status, typed.code) == ("error", "TOOL_REPORTED_FAILURE")
    assert TOOL_CODE_SPECS[typed.code].outcome_bucket == "tool_reported_failure"


def test_a_routing_refusal_is_recorded_without_degrading_execution_health(tmp_path, monkeypatch):
    """The other half of the homing, asserted where it is consumed: the agent SEES
    the refusal (is_error, a policy-denial row) and the execution axis stays clean,
    because the host refused — the agent did not fail. Same contract the steer
    receipts already had; a promote or route must not be judged differently."""
    from ouroboros._outcome_tool_errors import _classify_tool_errors
    from ouroboros.tools.tool_result import TOOL_CODE_SPECS, LegacyTextResultAdapter

    produced = _produced_refusals(tmp_path, monkeypatch)
    calls = [
        {
            "tool": "start_task",
            "status": TOOL_CODE_SPECS[
                LegacyTextResultAdapter.from_text("start_task", text).code
            ].outcome_bucket,
            "is_error": True,
            "result": text,
        }
        for text in produced.values()
    ]

    buckets = _classify_tool_errors({"tool_calls": calls})

    assert len(buckets["policy_denials"]) == len(calls)
    assert buckets["unresolved"] == []


# --- (d) an admitted promote names where the task actually landed -------------

def test_an_admitted_promote_names_the_destination_admission_returned(tmp_path, monkeypatch):
    """An implicit promote's project is re-resolved by the admission handler under
    the origin claim lock, so a sibling card converted in the emit → admission
    window moves the root into a project the tool never named. Reporting the
    caller's own argument back told the owner the work was in a room it is not in."""
    from ouroboros.projects_registry import create_project
    from ouroboros.tools import control_events
    from ouroboros.tools.control import _start_task

    create_project(tmp_path, "dinosaurs", name="Dinosaurs")
    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": "dinosaurs"},
    )

    out = _start_task(
        _tool_ctx(tmp_path), "Audit the issue", title="Issue audit",
    )

    assert out.startswith("OK: task")
    assert "in project 'Dinosaurs' (dinosaurs)" in out


def test_a_promote_admitted_into_no_project_claims_none(tmp_path, monkeypatch):
    """The symmetric direction: a requested project that admission did not grant
    must not be reported as the destination either."""
    from ouroboros.tools import control_events
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": ""},
    )

    out = _start_task(
        _tool_ctx(tmp_path), "Audit the issue", title="Issue audit", project_name="Dinosaurs",

    )

    assert out.startswith("OK: task")
    assert "project" not in out.split("accepted")[0]


def test_an_unconfirmed_promote_names_what_was_requested_as_a_request(tmp_path, monkeypatch):
    """No receipt means no known destination, so the sentence says what was ASKED
    for and that the outcome is unknown — never a room the task may not be in."""
    from ouroboros.tools import control_events
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )

    out = _start_task(
        _tool_ctx(tmp_path), "Audit the issue", title="Issue audit", project_name="Dinosaurs",

    )

    assert out.startswith("Accepted, waiting to start:")
    assert "admission to new project 'Dinosaurs'" in out
    assert "is pending" in out


# --- (e) one owner message keeps one convertible unit (#902) ------------------

_OWNER_TEXT = "turn the seven skills into a project"


def _origin_ref(client_message_id="cm-1", chat_id=1):
    """An ingress-shaped owner-message ref; the binding validates its integrity."""
    from ouroboros.project_dialogue import build_owner_message_ref

    return build_owner_message_ref(
        chat_id=chat_id, client_message_id=client_message_id,
        ts="2026-09-15T00:00:00+00:00", text=_OWNER_TEXT,
    )


def _request(tmp_path):
    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path)),
    )


def test_binding_projection_needs_no_live_origin_queue(tmp_path):
    from ouroboros.gateway.state import _task_bindings_safe
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "dinosaurs", name="Dinosaurs")
    bind_task_to_project(tmp_path, "clicked-root", "dinosaurs", project["chat_id"],
                         origin={"ref": _origin_ref(), "text": _OWNER_TEXT})
    assert _task_bindings_safe(_request(tmp_path)) == {
        "clicked-root": {"project_id": "dinosaurs", "chat_id": project["chat_id"], "project_name": "Dinosaurs"},
    }


# --- (f) receipts: one owner message, several acts, every receipt readable ---

def test_two_roots_under_one_owner_message_keep_two_readable_receipts_after_compaction(tmp_path):
    """One owner message becomes a root, then a later steer relays the same
    message: two acts, two tokens. The retained set is keyed per (message,
    token), so compaction keeps the older act's receipt readable by its token
    while the message's LATEST row stays what the UI paints."""
    import json

    from ouroboros.project_dialogue import (
        _COMPACT_AT_BYTES, append_chat_annotation, chat_annotation_receipt, latest_chat_annotations,
    )

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text(
        json.dumps({"direction": "in", "chat_id": 1, "client_message_id": "cm-1", "text": "go"}) + "\n",
        encoding="utf-8",
    )
    assert append_chat_annotation(
        tmp_path, "cm-1", action="promote_chat_to_task", target="root-a", status="scheduled",
        routing_token="tok-promote",
    )
    assert append_chat_annotation(
        tmp_path, "cm-1", action="steer_task", target="root-b", status="delivered",
        routing_token="tok-steer",
    )
    annotations = logs / "chat_annotations.jsonl"
    filler = {"ts": "2026-09-15T00:00:00Z", "type": "chat_annotation", "client_message_id": "msg-expired",
              "action": "routed", "status": "delivered", "detail": "x" * 400}
    with annotations.open("a", encoding="utf-8") as stream:
        while annotations.stat().st_size < _COMPACT_AT_BYTES:
            stream.write(json.dumps(filler) + "\n")
    assert append_chat_annotation(tmp_path, "cm-1", action="steer_task", target="root-b",
                                  status="delivered", routing_token="tok-steer-2")

    assert annotations.stat().st_size < _COMPACT_AT_BYTES  # it compacted
    assert chat_annotation_receipt(tmp_path, "cm-1", "tok-promote")["target"] == "root-a"
    assert chat_annotation_receipt(tmp_path, "cm-1", "tok-steer")["target"] == "root-b"
    assert chat_annotation_receipt(tmp_path, "cm-1", "tok-steer-2")["status"] == "delivered"
    assert chat_annotation_receipt(tmp_path, "cm-1", "tok-unknown") == {}
    assert latest_chat_annotations(tmp_path)["cm-1"]["routing_token"] == "tok-steer-2"
    assert "msg-expired" not in latest_chat_annotations(tmp_path)


# --- (e) a relayed owner steer refused by a pending cancel: receipt, no bubble ---

def test_a_relayed_owner_steer_refused_by_a_pending_cancel_gets_no_standalone_bubble(
    tmp_path, monkeypatch,
):
    """R15: an act that wears an owner message shows its cause on that message's
    receipt line, once its ACK is handed over; only an UNLABELLED owner act (a
    synthetic receipt id, so no chat row can show the refusal) still gets the
    standalone cancel-pending notice."""
    import ouroboros.cancel_intents as cancel_intents
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX, latest_chat_annotations
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(
        cancel_intents, "cancel_pending", lambda _root, task_id, **_k: task_id == "t-target",
    )
    notices, acks = [], []
    supervisor = _supervisor_ctx(
        tmp_path, notices, running={"t-target": {"task": {"id": "t-target", "chat_id": 1}}},
    )
    supervisor.bridge = types.SimpleNamespace(send_routing_ack=lambda _chat_id, **kwargs: acks.append(kwargs))
    base = {"type": "steer_task", "target_task_id": "t-target", "message": "stop after this file",
            "chat_id": 1}

    _handle_steer_task({**base, "client_message_id": "cm-owner-1", "routing_token": "tok-1"}, supervisor)

    row = latest_chat_annotations(tmp_path)["cm-owner-1"]
    assert (row["status"], row["reason"]) == ("rejected", "cancel_pending")
    assert row["cause"] == "Not delivered: that task is being stopped"
    assert [(ack["cause"], ack["routing_notice_carrier"]) for ack in acks] == [(row["cause"], "routing_ack")]
    assert notices == []

    unlabelled = f"{AGENT_RECEIPT_ID_PREFIX}tok-2"
    _handle_steer_task({**base, "client_message_id": unlabelled, "routing_token": "tok-2"}, supervisor)

    assert latest_chat_annotations(tmp_path)[unlabelled]["cause"] == "Not delivered: that task is being stopped"
    assert len(notices) == 1
