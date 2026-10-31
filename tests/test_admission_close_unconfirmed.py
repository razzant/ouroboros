"""Owner decision 2A after task acceptance: the note old decisions carry still renders, and the
root admission seal at an author stop reads the queue's typed answer.

The vocabulary half: one reason token, one owner sentence in BOTH twins
(``project_dialogue.TASK_CAUSE_PHRASES`` / ``web/modules/task_reason_words.js``), pinned by
``web/tests/fixtures/outcome_phase_parity.json`` on both sides of the boundary, and a
reducer that keeps an accepted decision out of BLOCKED. Nothing writes the note now (the
acceptance decision writer is gone); stored task cards still carry it. The behaviour half:
the seal (``loop_delivery._seal_admission_before_delivery``, right before an author stop is
delivered) reads a ``sealed`` refusal as its own earlier seal and delivers over a silent
supervisor or any other refusal with ``admission_released=False``; only owner facts, a locally
seen owner change or owner mail that arrived during the silent wait, start the owner-revision round.
"""

from __future__ import annotations

import pathlib
import queue as stdqueue
import threading
from types import SimpleNamespace

import pytest

from ouroboros.outcomes import (
    ACCEPTANCE_ACCEPTED, ACCEPTANCE_FINALIZED_UNACCEPTED, OBJECTIVE_PASS, OUTCOME_TIER_BLOCKED,
    OUTCOME_TIER_SOLVED, _objective_axis,
)
from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict, outcome_phase
from tests.test_acceptance_fence_transport import _pooled_agent

NOTE = "admission_close_unconfirmed"
SENTENCE = "Reviewers approved this answer; the supervisor did not confirm that task admission was closed."


def _record(status: str, reason: str, *, enforcement: str = "blocking") -> dict:
    review = {"status": "pass", "outcome_tier": OUTCOME_TIER_SOLVED,
              "acceptance_decision": {"status": status, "reason": reason, "enforcement": enforcement}}
    return {"status": "completed", "reason_code": "final_message",
            "outcome_axes": {"execution": {"status": "ok"}, "review": review, "objective": _objective_axis(review)}}


def test_the_note_has_one_sentence_in_both_twins():
    """Old acceptance decisions carry the note; their task cards keep saying it."""
    assert TASK_CAUSE_PHRASES[NOTE] == SENTENCE
    twin = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "task_reason_words.js").read_text(encoding="utf-8")
    assert f'{NOTE}: "{SENTENCE}",' in twin, "the browser twin carries the byte-identical sentence"


def test_an_accepted_answer_with_the_note_is_done_and_never_blocked():
    """Both directions of the reducer: the note rides an ACCEPTED decision and keeps the
    objective a PASS at the reviewer's tier, while the honest unaccepted cells that a
    transport gap must NOT be laundered into still terminalize BLOCKED under blocking."""
    record = _record(ACCEPTANCE_ACCEPTED, NOTE)
    objective = record["outcome_axes"]["objective"]
    assert objective["status"] == OBJECTIVE_PASS and objective["outcome_tier"] == OUTCOME_TIER_SOLVED
    assert outcome_phase(record, {}) == "done" and outcome_phase({}, record) == "done"
    assert _completion_verdict(record, {}) == _completion_verdict({}, record) == SENTENCE
    for refusal in ("infra_failure", "review_degraded"):
        blocked = _record(ACCEPTANCE_FINALIZED_UNACCEPTED, refusal)["outcome_axes"]["objective"]
        assert blocked["outcome_tier"] == OUTCOME_TIER_BLOCKED and blocked["reason"] == refusal
    # A clean accepted decision keeps rendering nothing: the note is the only accepted cell that speaks here.
    assert _completion_verdict(_record(ACCEPTANCE_ACCEPTED, "clean_pass"), {}) == ""


# --- the behaviour half: the final seal at an author stop reads the queue's typed answer -------


def _gap(**_kwargs):
    raise TimeoutError("supervisor did not acknowledge acceptance fence")


def _seal_context(tmp_path, owner_changed=False):
    ctx = SimpleNamespace(
        task_id="root-1", drive_root=tmp_path, task_metadata={"root_task_id": "root-1", "budget_drive_root": str(tmp_path)},
        _task_acceptance_fence_token=None, _task_acceptance_sealed_fence_token=None, _task_acceptance_fence_generation=None,
        _task_acceptance_fence_generation_mismatch=owner_changed, _owner_directives=[],
        _completion_selected={"action": "stop", "rationale": "The rest needs the owner's decision."},
    )
    return SimpleNamespace(_ctx=ctx), SimpleNamespace(task_id="root-1")


def _stopped_trace():
    """The trace as ``record_stopped_completion`` leaves it right before the seal runs."""
    return {"review_decision": {}, "review_runs": [],
            "task_completion": {"action": "stop", "rationale": "The rest needs the owner's decision.",
                                "answer_sha256": "a" * 64, "source": "finish_task"}}


@pytest.mark.parametrize("answer,owner_changed,delivered", [
    ("unknown", False, True),  # a gap, never a refusal: delivered, the release disclosed
    ("sealed", False, True),  # the worker's own earlier seal
    ("active", False, True),  # another holder is not owner mail: delivered, the release disclosed
    ("unknown", True, False),  # a locally seen owner change is real
])
def test_the_final_seal_reads_the_queues_typed_answer(tmp_path, answer, owner_changed, delivered):
    from ouroboros.loop_delivery import _seal_admission_before_delivery

    tools, limit_ctx = _seal_context(tmp_path, owner_changed)
    tools._ctx.begin_acceptance_fence = _gap if answer == "unknown" else (lambda **_kw: (_ for _ in ()).throw(RuntimeError(answer)))
    llm_trace = _stopped_trace()
    assert _seal_admission_before_delivery(tools, limit_ctx, llm_trace) is delivered
    # The owner-revision round voids the selected completion; a delivered stop keeps its record.
    assert ("task_completion" in llm_trace) is delivered
    if delivered and answer in {"unknown", "active"}:
        assert llm_trace["review_decision"]["admission_released"] is False


def test_owner_mail_that_arrived_during_the_silent_wait_is_not_delivered_over(tmp_path):
    """The wait for a silent supervisor is long enough for the owner to write, and their mail is
    durable before any generation moves: a gap delivers only after the local mailbox was read once
    more. Quiet direction: the same gap with an empty mailbox delivers (table above)."""
    from ouroboros.loop_delivery import _seal_admission_before_delivery
    from ouroboros.owner_mailbox import write_owner_message

    tools, limit_ctx = _seal_context(tmp_path, False)
    tools._ctx.begin_acceptance_fence = _gap
    assert write_owner_message(tmp_path, "Use the blue variant instead", "root-1", msg_id="owner-blue")
    llm_trace = _stopped_trace()
    assert _seal_admission_before_delivery(tools, limit_ctx, llm_trace) is False
    assert "task_completion" not in llm_trace


def test_a_locally_seen_owner_change_survives_a_silent_end(tmp_path):
    """`refused(generation_mismatch)` is a real owner follow-up even when the transport went
    silent: the local comparison decides the flag, not the missing ack. The quiet direction:
    an unchanged owner generation with the same silent end leaves the flag down."""
    from ouroboros.loop import _end_task_acceptance_fence

    def ctx(generation):
        return SimpleNamespace(
            task_metadata={"root_task_id": "root-1"}, task_id="root-1", drive_root=tmp_path,
            _task_acceptance_fence_token="token-1", _task_acceptance_sealed_fence_token=None,
            _task_acceptance_fence_generation=0, _task_acceptance_queue_descendants=[],
            _task_acceptance_owner_generation=0, owner_message_admission_lock=threading.RLock(),
            owner_message_admission_agent=SimpleNamespace(_owner_message_generation=generation),
            end_acceptance_fence=_gap,
        )

    changed, same = ctx(1), ctx(0)
    assert _end_task_acceptance_fence(changed, outcome="terminal").status == "unknown"
    assert changed._task_acceptance_fence_generation_mismatch is True
    assert _end_task_acceptance_fence(same, outcome="terminal").status == "unknown"
    assert same._task_acceptance_fence_generation_mismatch is False


@pytest.mark.parametrize("op,ack,reason", [
    ("begin", {"ok": False, "status": "sealed", "error": "acceptance fence already sealed for root root-1"}, "sealed"),
    ("begin", {"ok": False, "status": "error", "error": "invalid acceptance fence event"}, "invalid acceptance fence event"),
    ("inspect", {"ok": True, "status": "released", "token": "t", "row_absent": True}, "released"),
])
def test_the_seam_refuses_with_the_rows_typed_status(tmp_path, monkeypatch, op, ack, reason):
    """A refusal carries the row's typed state (`sealed`, `active`, `released`) as its reason;
    only a malformed request keeps the error text. The worker reads the state, never the prose."""
    from supervisor import queue as task_queue

    agent = _pooled_agent(tmp_path, stdqueue.Queue())
    agent._fence_in_process = lambda: True
    monkeypatch.setattr(task_queue, "transition_acceptance_fence", lambda **_kwargs: ack)
    with pytest.raises(RuntimeError) as refused:
        if op == "begin":
            agent._begin_acceptance_fence(root_task_id="root-1", task_id="root-1")
        else:
            agent._inspect_acceptance_fence(token="t")
    assert str(refused.value) == reason
