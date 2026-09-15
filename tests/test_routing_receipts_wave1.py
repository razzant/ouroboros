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
    from ouroboros.tools.control import _steer_task

    notices, emitted = [], []
    ctx = _live_tool_ctx(tmp_path, _supervisor_ctx(tmp_path, notices), emitted, metadata={})

    out = _steer_task(ctx, "t-gone", "pick the smaller PR")

    assert out.startswith("⚠️ STEER_REJECTED: task t-gone was not steered (target_unknown)")
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
    from ouroboros.tools.control import _steer_task

    monkeypatch.setattr(
        cancel_intents, "cancel_pending", lambda _root, task_id, **_k: task_id == "t-target",
    )
    notices, emitted = [], []
    supervisor = _supervisor_ctx(
        tmp_path, notices, running={"t-target": {"task": {"id": "t-target", "chat_id": 1}}},
    )
    ctx = _live_tool_ctx(tmp_path, supervisor, emitted, metadata={})

    out = _steer_task(ctx, "t-target", "stop after the current file")

    assert out.startswith("⚠️ STEER_REJECTED: task t-target was not steered (cancel_pending)")
    receipt_id = emitted[0]["client_message_id"]
    assert latest_chat_annotations(tmp_path)[receipt_id]["status"] == "rejected"


def test_a_delivered_task_authored_steer_is_confirmed_under_its_own_id(tmp_path, monkeypatch):
    """The capability half of the same fix: the synthetic id must confirm a
    LANDED delivery too, or every task-authored steer would invite a retry of a
    message that already arrived. A task speaking for itself is told its message
    was WRITTEN (wave 2: it lands as a task message, never as owner text)."""
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import KIND_TASK_MESSAGE, drain_owner_entries
    from ouroboros.tools.control import _steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    notices, emitted = [], []
    supervisor = _supervisor_ctx(
        tmp_path, notices, running={"t-target": {"task": {"id": "t-target", "chat_id": 1}}},
    )
    ctx = _live_tool_ctx(tmp_path, supervisor, emitted, metadata={})

    out = _steer_task(ctx, "t-target", "keep the PR small")

    assert out.startswith("✉️ Message to task t-target written to its mailbox (durably confirmed")
    assert "UNCONFIRMED" not in out
    [entry] = drain_owner_entries(tmp_path, "t-target")
    assert (entry["text"], entry["kind"], entry["provenance"], entry["source_task_id"]) == (
        "keep the PR small", KIND_TASK_MESSAGE, "independent_task", "turn-1",
    )


# --- (c) a routing refusal is a recorded failure, not a successful call -------

def _produced_refusals(tmp_path, monkeypatch):
    """Every routing refusal sentence, taken from the PRODUCER that composes it."""
    from ouroboros.projects_registry import create_project
    from ouroboros.tools import control_events, control_routing
    from ouroboros.tools.control import _promote_chat_to_task, _route_to_project

    create_project(tmp_path, "dinosaurs")
    produced: dict = {}

    monkeypatch.setattr(
        control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "crash_storm",
    )
    produced["pool_disabled"] = _promote_chat_to_task(
        _tool_ctx(tmp_path), "Audit the issue", predecessor_task_id="",
    )
    monkeypatch.setattr(
        control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "",
    )

    for key, admission in (
        ("promote_rejected", {"status": "rejected", "reason": "duplicate_task_id"}),
        ("promote_unconfirmed", {"status": "unconfirmed", "reason": "confirmation_timeout"}),
    ):
        monkeypatch.setattr(
            control_events, "_wait_for_promotion_admission", lambda *_a, _r=admission, **_k: dict(_r),
        )
        produced[key] = _promote_chat_to_task(
            _tool_ctx(tmp_path), "Audit the issue", project_name="Dinosaurs",
            predecessor_task_id="",
        )

    for key, receipt in (
        ("route_rejected", {"status": "rejected", "reason": "project_routing_fence"}),
        ("route_unconfirmed", {"status": "unconfirmed", "reason": "confirmation_timeout"}),
    ):
        monkeypatch.setattr(
            control_events, "_wait_for_promotion_admission", lambda *_a, _r=receipt, **_k: dict(_r),
        )
        produced[key] = _route_to_project(
            _tool_ctx(tmp_path), project_id="dinosaurs", message="continue there",
            predecessor_task_id="",
        )

    for key, receipt in (
        ("needs_manual_target", {"status": "needs_manual_target", "options": []}),
        ("routing_unconfirmed", {"status": "unconfirmed", "reason": "confirmation_timeout"}),
    ):
        monkeypatch.setattr(
            control_events, "_wait_for_routing_annotation", lambda *_a, _r=receipt, **_k: dict(_r),
        )
        produced[key] = _route_to_project(
            _tool_ctx(tmp_path, metadata={"client_message_id": "cm-1"}),
            project_id="", message="somewhere", predecessor_task_id="",
        )
    return produced


@pytest.mark.parametrize("key,identifier", [
    ("pool_disabled", "PROMOTE_REJECTED"),
    ("promote_rejected", "PROMOTE_REJECTED"),
    ("promote_unconfirmed", "PROMOTE_UNCONFIRMED"),
    ("route_rejected", "ROUTE_REJECTED"),
    ("route_unconfirmed", "ROUTE_UNCONFIRMED"),
    ("needs_manual_target", "NEEDS_MANUAL_TARGET"),
    ("routing_unconfirmed", "ROUTING_UNCONFIRMED"),
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

    typed = LegacyTextResultAdapter.from_text("promote_chat_to_task", text)
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
            "tool": "route_to_project",
            "status": TOOL_CODE_SPECS[
                LegacyTextResultAdapter.from_text("route_to_project", text).code
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
    from ouroboros.tools.control import _promote_chat_to_task

    create_project(tmp_path, "dinosaurs", name="Dinosaurs")
    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": "dinosaurs"},
    )

    out = _promote_chat_to_task(
        _tool_ctx(tmp_path), "Audit the issue", predecessor_task_id="",
    )

    assert out.startswith("OK: task")
    assert "in project 'Dinosaurs' (dinosaurs)" in out


def test_a_promote_admitted_into_no_project_claims_none(tmp_path, monkeypatch):
    """The symmetric direction: a requested project that admission did not grant
    must not be reported as the destination either."""
    from ouroboros.tools import control_events
    from ouroboros.tools.control import _promote_chat_to_task

    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": ""},
    )

    out = _promote_chat_to_task(
        _tool_ctx(tmp_path), "Audit the issue", project_name="Dinosaurs",
        predecessor_task_id="",
    )

    assert out.startswith("OK: task")
    assert "project" not in out.split("accepted")[0]


def test_an_unconfirmed_promote_names_what_was_requested_as_a_request(tmp_path, monkeypatch):
    """No receipt means no known destination, so the sentence says what was ASKED
    for and that the outcome is unknown — never a room the task may not be in."""
    from ouroboros.tools import control_events
    from ouroboros.tools.control import _promote_chat_to_task

    monkeypatch.setattr(
        control_events, "_wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )

    out = _promote_chat_to_task(
        _tool_ctx(tmp_path), "Audit the issue", project_name="Dinosaurs",
        predecessor_task_id="",
    )

    assert out.startswith("⚠️ PROMOTE_UNCONFIRMED")
    assert "the requested destination was new project 'Dinosaurs'" in out
    assert "the effective one is unknown" in out


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


def test_a_live_task_whose_owner_message_has_a_project_offers_no_second_conversion(
    tmp_path, monkeypatch,
):
    """An ADOPTING conversion (#900) binds only the clicked card. The sibling task
    of the SAME owner message stayed task-unbound, so the client gate — which is
    task-keyed — kept offering "Turn into project" for work that already has one."""
    import supervisor.workers as workers
    from ouroboros.gateway.state import _task_bindings_safe
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "dinosaurs", name="Dinosaurs")
    bind_task_to_project(
        tmp_path, "clicked-root", "dinosaurs", project["chat_id"],
        origin={"ref": _origin_ref(), "text": _OWNER_TEXT},
    )
    monkeypatch.setattr(workers, "PENDING", [
        {"id": "sibling-root", "origin_message_ref": _origin_ref()},
        {"id": "other-work", "origin_message_ref": _origin_ref("cm-OTHER")},
        {"id": "child", "delegation_role": "subagent", "origin_message_ref": _origin_ref()},
    ])
    monkeypatch.setattr(workers, "RUNNING", {})

    bindings = _task_bindings_safe(_request(tmp_path))

    assert bindings["clicked-root"] == {
        "project_id": "dinosaurs", "chat_id": project["chat_id"],
    }
    assert bindings["sibling-root"] == {
        "project_id": "dinosaurs", "chat_id": project["chat_id"], "origin_bound": True,
    }
    # A different message's work keeps its own convert button, and a delegated
    # child is never bound itself — it inherits its root's project by lineage.
    assert "other-work" not in bindings and "child" not in bindings


def test_an_origin_bound_card_is_not_moved_out_of_the_chat_it_runs_in(tmp_path, monkeypatch):
    """A DURABLE binding re-homes a converted card; an origin-bound row is a gate
    fact only. Moving the live card into the project room would strand it from its
    own chat rows, which no conversion ever wrote."""
    import supervisor.queue as queue_mod
    from ouroboros.gateway.state import _chat_activities_snapshot_safe

    monkeypatch.setattr(queue_mod, "PENDING", [{
        "id": "sibling-root", "root_task_id": "sibling-root",
        "delegation_role": "root", "chat_id": 1,
    }])
    monkeypatch.setattr(queue_mod, "RUNNING", {})
    bindings = {
        "sibling-root": {"project_id": "dinosaurs", "chat_id": 77, "origin_bound": True},
    }

    rows = _chat_activities_snapshot_safe(tmp_path, bindings, direct_turns=[])

    [row] = [entry for entry in rows if entry["activity_id"] == "sibling-root"]
    assert (row["chat_id"], row["project_id"]) == (1, "")


def test_an_unreadable_live_queue_still_answers_the_durable_bindings(tmp_path, monkeypatch):
    """The enrichment fails OPEN: its residual is the stray button it was added to
    remove, never a wrong or missing durable binding."""
    import ouroboros.projects_registry as registry
    from ouroboros.gateway.state import _task_bindings_safe
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "dinosaurs", name="Dinosaurs")
    bind_task_to_project(
        tmp_path, "clicked-root", "dinosaurs", project["chat_id"],
        origin={"ref": _origin_ref(), "text": _OWNER_TEXT},
    )

    def _boom():
        raise OSError("supervisor unreadable")

    monkeypatch.setattr(registry, "live_origin_lanes", _boom)

    assert _task_bindings_safe(_request(tmp_path)) == {
        "clicked-root": {"project_id": "dinosaurs", "chat_id": project["chat_id"]},
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

