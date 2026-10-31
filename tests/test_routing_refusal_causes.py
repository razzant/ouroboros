"""Host-owned cause sentences for refused routing acts (Q3=A; R6/R7/R11/R12/R14).

The receipt line under the owner's message, the host-initiated System row and
the picker's 409 toast all read ONE host table; the browser renders the
sentence verbatim. These pins keep the table complete against every producer
of a refusal reason, keep the phrases factual and short, keep the prefix an
honest statement of the act's outcome, and keep the admission-notice rows in
the chat the owner wrote in.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from ouroboros.project_dialogue import (
    ORIGIN_ADDRESSED_NOTICE_TYPES,
    ROUTING_REFUSAL_CAUSES,
    WORK_SPEECH_ELSEWHERE_TYPES,
    room_membership,
    routing_refusal_cause,
)

REPO = pathlib.Path(__file__).resolve().parents[1]

# Every typed refusal reason that reaches `_emit_routing_receipt`, the promote
# outcome (the host-initiated notice) or the steer receipt — one row each.
REFUSAL_REASONS = (
    # promote / route admission (worker_promotion, events_project_routing,
    # queue.enqueue_task `_admission_blocked`, task_admission.reserve_task_admission)
    "workspace_unusable", "workspace_provisioning_failed", "worker_pool_unavailable",
    "worker_pool_state_unavailable", "duplicate_task_id", "admission_reservation_owned",
    "admission_reservation_lost", "admission_reservation_failed", "admission_fence",
    "admission_rejected", "invalid_admission_reservation", "task_id_lookup_failed",
    "empty_objective", "project_routing_fence", "project_routing_fence_changed", "project_routing_fence_lookup_failed",
    "project_binding_failed", "project_registration_failed", "project_source_error",
    "attachment_admission_rejected", "staging_unavailable",
    "queue_snapshot_persist_unavailable", "queue_snapshot_persist_failed",
    "invalid_skill_repair_constraint", "skill_repair_payload_missing",
    "skill_repair_payload_unreadable", "skill_repair_admission_unwritable",
    "repair_promotion_failed", "task_acceptance_fence", "invalid_task_depth",
    # the unconfirmed family (handler crash, receipt persistence, routing_wait)
    "promotion_persistence_failed", "routing_receipt_persist_failed",
    "routing_annotation_persist_failed", "source_continuation_publish_failed",
    "confirmation_timeout",
    # steer (supervisor/steering.py)
    "target_unknown", "direct_chat_turn", "subagent_target",
    "cancel_pending", "target_closed", "target_finished", "acceptance_fence_sealed",
    "mailbox_write_failed",
    # ensure_project_scope (worker_promotion, events_project_routing)
    "project_scope_conflict", "missing_task_or_project", "ensure_project_scope_failed",
    # the parallel receipt producer (ouroboros/server_owner_routing.py)
    "project_unavailable",
    # route_to_project's typed abstention (ouroboros/tools/control_routing.py)
    "target_unspecified", "invalid_project_id", "target_not_found",
)

PRODUCERS = (
    "supervisor/worker_promotion.py",
    "supervisor/events_project_routing.py",
    "supervisor/steering.py",
    "supervisor/queue.py",
    "supervisor/task_admission.py",
    "ouroboros/server_owner_routing.py",
    "ouroboros/tools/control_routing.py",
)
# The scan sees `"reason": "<lit>"` / `reason="<lit>"` only; a code chosen by a
# conditional expression (control_routing.py `failure = ("a" if … else "b")`) is
# invisible to it and is pinned by REFUSAL_REASONS instead.
_REASON_LITERAL = re.compile(r'(?:"reason":\s*|\breason=)"([a-z][a-z0-9_]*)"')

# Literals the producer scan finds that are NOT owner-facing refusal reasons.
EXEMPT = {
    # snapshot-persist / rollback reasons (a persist call's audit label, never a receipt)
    "promote_chat_to_task_rejected", "promote_chat_to_task_failed", "promote_chat_to_task",
    "drain_all_pending", "acceptance_fence_owner_message", "evolve_off", "deep_self_review_enqueued",
    "project_hold_revalidated", "schedule_subagent_admitted",
    "agent_restart_admission", "agent_restart_admission_unconfirmed",
    # the bind-failure events.jsonl row's reason (project_binding_unreadable), never a receipt
    "project_binding_unreadable",
    # success / pseudo reasons on a delivered ensure_project_scope receipt
    "created", "adopted", "attached", "delivered", "renamed", "name_unchanged", "rename_failed",
    # the picker's claim/closing pseudo-reasons (`claimed_option:N` / `answered_option:N`)
    "answered_option", "claimed_option",
    # internal wait/transport outcomes that never reach an owner surface
    "client_message_id_missing", "event_serialization_failed", "handler_returned_no_outcome",
    # StoreBusy's acquisition cause; the same unsent owner retries it off-lock.
    "contention",
}


def test_every_refusal_reason_has_a_short_factual_phrase():
    assert set(REFUSAL_REASONS) == set(ROUTING_REFUSAL_CAUSES)
    for reason, phrase in ROUTING_REFUSAL_CAUSES.items():
        assert phrase == phrase.strip() and len(phrase) <= 60, (reason, phrase)
        assert "_" not in phrase and not phrase.endswith("."), (reason, phrase)
        assert phrase[0].islower() and not phrase.startswith("Not "), (reason, phrase)


@pytest.mark.parametrize(
    "action, status, expected",
    [
        ("promote_chat_to_task", "needs_manual_target", "Not started"),
        ("route_to_project", "needs_manual_target", "Not started"),
        ("promote_chat_to_task", "rejected", "Not started"),
        ("steer_task", "needs_manual_target", "Not delivered"),
        ("steer_task", "rejected", "Not delivered"),
        ("ensure_project_scope", "rejected", "Not moved"),
        ("promote_chat_to_task", "unconfirmed", "Not confirmed"),
        ("steer_task", "unconfirmed", "Not confirmed"),
        ("ensure_project_scope", "unconfirmed", "Not confirmed"),
    ],
)
def test_prefix_follows_the_act_and_its_outcome(action, status, expected):
    """The prefix states only what the outcome proves: an UNCONFIRMED act reads
    «Not confirmed» whatever it was; a refused steer «Not delivered»; a refused
    scope bind «Not moved»; every other refused act «Not started»."""
    sentence = routing_refusal_cause(action, status, "project_binding_failed")
    assert sentence == f"{expected}: the project could not be set up"


def test_landed_rows_and_the_picker_carry_no_cause():
    for status in ("scheduled", "delivered", "pending", "dispatch_pending", "accepted"):
        assert routing_refusal_cause("promote_chat_to_task", status, "workspace_unusable") == ""
    # A refusal WITH options is the picker: «Choose a target · A / B» stays.
    assert routing_refusal_cause("route_decision", "needs_manual_target", "target_unspecified",
                                 [{"action": "steer_task", "task_id": "t1"}]) == ""
    # Without options nothing can be chosen, so the row states the cause.
    assert routing_refusal_cause("route_decision", "needs_manual_target", "target_unspecified", []) == (
        "Not started: no destination was chosen"
    )


def test_unknown_reasons_stay_raw_and_unconfirmed_stays_honest():
    """DESIGN sanctions raw over invented: a reason with no row is shown as its
    code (the raw fallback is exempt from the no-underscore rule), while an
    unconfirmed act with an unknown reason reads as the honest uncertainty."""
    assert routing_refusal_cause("promote_chat_to_task", "needs_manual_target", "x_y") == "Not started (x_y)"
    assert routing_refusal_cause("steer_task", "rejected", "x_y") == "Not delivered (x_y)"
    assert routing_refusal_cause("promote_chat_to_task", "unconfirmed", "x_y") == (
        "Not confirmed: the task may or may not have started"
    )
    assert routing_refusal_cause("promote_chat_to_task", "needs_manual_target", "") == "Not started"
    assert routing_refusal_cause("promote_chat_to_task", "needs_manual_target", "workspace_unusable") == (
        "Not started: the working folder can't be used"
    )


def test_producer_literals_are_all_covered_or_exempt():
    """Completeness against the code: every `"reason": "<literal>"` /
    `reason="<literal>"` a producer writes has a sentence or an explicit exemption."""
    found = {}
    for relative in PRODUCERS:
        source = (REPO / relative).read_text(encoding="utf-8")
        for match in _REASON_LITERAL.finditer(source):
            found.setdefault(match.group(1), set()).add(relative)
    assert found, "the producer scan found nothing — the regex or the paths drifted"
    missing = {
        literal: sorted(files) for literal, files in found.items()
        if literal not in ROUTING_REFUSAL_CAUSES and literal not in EXEMPT
    }
    assert missing == {}, f"refusal reasons without a host sentence: {missing}"
    assert not (set(EXEMPT) & set(ROUTING_REFUSAL_CAUSES)), "a reason cannot be both exempt and worded"


def test_origin_addressed_notice_types_cover_admission_and_steering():
    assert ORIGIN_ADDRESSED_NOTICE_TYPES == frozenset({
        "task_not_started", "task_start_unconfirmed", "steer_not_delivered", "relay_reply",
    })


@pytest.mark.parametrize("notice_type", sorted(ORIGIN_ADDRESSED_NOTICE_TYPES))
def test_origin_addressed_notice_rows_stay_in_the_owners_chat(notice_type):
    """R12: the refused task id is bound to the project it never started in, so
    binding lineage would move the notice out of Main into that project on
    reload. The notice is addressed to the chat the OWNER wrote in. A relay's answer
    is its work's speech too: it stays in that chat and also joins its work's Project."""
    rooms = lambda task_id, parent="", root="": 77 if "refused-task" in (task_id, parent, root) else 0  # noqa: E731
    main = room_membership(1, {77}, [], rooms)
    project = room_membership(77, {77}, [], rooms)
    notice = {"direction": "system", "type": notice_type, "task_id": "refused-task", "chat_id": 1}

    assert main(1, notice) is True
    assert project(1, notice) is (notice_type in WORK_SPEECH_ELSEWHERE_TYPES)
    assert project(77, {**notice, "chat_id": 77}) is True
    # An ordinary bound row keeps today's behaviour: lineage moves it to its room.
    ordinary = {"direction": "out", "type": "chat", "task_id": "refused-task", "chat_id": 1}
    assert main(1, ordinary) is False
    assert project(1, ordinary) is True


@pytest.mark.parametrize("chat_id", [1, 0, -1001])
@pytest.mark.parametrize("labelled", [False, True])
@pytest.mark.parametrize("after_lock", [False, True])
def test_cancel_refusal_is_one_system_notice_only_when_no_owner_receipt_can_tell_it(
    tmp_path, monkeypatch, chat_id, labelled, after_lock,
):
    import types
    import ouroboros.cancel_intents as cancel_intents
    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor.steering import _handle_steer_task

    checks = iter([False, True] if after_lock else [True])
    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda *_a, **_k: next(checks))
    notices = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, PENDING=[],
        RUNNING={"target": {"task": {"id": "target", "title": "Review", "chat_id": chat_id,
                                    "drive_root": str(tmp_path)}}},
        send_with_budget=lambda cid, text, **kw: notices.append((cid, text, kw)),
    )
    _handle_steer_task({
        "target_task_id": "target", "message": "Continue", "chat_id": chat_id,
        "client_message_id": "owner-message" if labelled else "agent-steer:unlabelled",
        "issuer": {"kind": "owner_turn"},
    }, ctx)

    assert drain_owner_entries(tmp_path, "target") == []
    if chat_id < 0:
        assert notices == []
    else:
        # No bridge handed the ACK over, so no owner receipt can tell it, labelled or not: this row is the
        # notice's carrier, answering no owner row (the host accepted none under "owner-message" here).
        assert notices == [(chat_id, "Review · " + routing_refusal_cause("steer_task", "rejected", "cancel_pending"), {
            "role": "system", "system_type": "steer_not_delivered", "task_id": "target",
            "progress_meta": {"routing_notice_carrier": "system", "reply_to": {}},
        })]


def test_the_parallel_receipt_producer_reads_the_same_host_table(tmp_path):
    """`server_owner_routing._record_routing_receipt` is the SECOND producer of a
    routing receipt. Its one refusal — an owner message addressed to a reserved
    project that is no longer available — used to leave the owner line to a
    hardcoded browser label; it now carries the host sentence on the durable row
    and on the live ack. A landed receipt still carries none."""
    import types

    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.server_owner_routing import _record_routing_receipt

    acks = []
    bridge = types.SimpleNamespace(send_routing_ack=lambda chat_id, **kw: acks.append((chat_id, kw)))
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path)

    _record_routing_receipt(
        bridge, ctx, chat_id=1, client_message_id="owner-pu",
        action="project_route", target="gone", status="project_unavailable",
        reason="project_unavailable",
    )
    row = latest_chat_annotations(tmp_path)["owner-pu"]
    assert (row["status"], row["reason"]) == ("project_unavailable", "project_unavailable")
    assert row["cause"] == "Not started: the project is no longer available"
    assert acks[0][1]["cause"] == row["cause"]

    _record_routing_receipt(
        bridge, ctx, chat_id=1, client_message_id="owner-ok",
        action="mailbox_delivery", target="t-1", status="delivered",
    )
    assert "cause" not in latest_chat_annotations(tmp_path)["owner-ok"]
    assert "cause" not in acks[1][1]
