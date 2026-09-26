"""A failed parent retains already admitted work without reviving stale speech."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
from ouroboros.presence_runner import PresenceTurnError, PresenceTurnGate, presence_result_from_stored, run_presence_turn
from ouroboros.task_results import load_task_result
from ouroboros.tools.control_routing import _finish_swarm_handoff
from ouroboros.tools.presence import _finish_presence
from tests.test_presence_runner import _admission, _event


def _failed_parent(tmp_path, *, outcome="deferred", admission="scheduled", usage=None, accepted=False,
                   request_text="Outdated proposed reply", refusal=""):
    """Real completion/admission producers and durable terminal pipeline.

    ``refusal`` names the typed Host refusal an unproven attempt raises instead of a result,
    on the first call and again on replay; the returned error carries the admitted ``work_ref``.
    """
    created = []
    text = "Current partial result after the parent stopped."
    terminal_usage = {"execution_status": "infra_failed", "reason_code": "provider_unavailable",
                      "terminal_origin": "model_final",
                      "terminal_provider_notice": "Provider unavailable; child work remains admitted."} if usage is None else usage

    class Agent:
        def handle_task(self, task):
            created.append(task["id"])
            task["_skip_post_task_synthesis"] = True
            ctx = SimpleNamespace(task_contract=task["task_contract"], task_metadata=task["metadata"])
            _finish_presence(ctx, outcome, request_text)
            ctx._presence_completion_accepted = accepted
            if admission:
                _finish_swarm_handoff(ctx, {"task_id": "managed-work"}, "Admission receipt", status=admission)
            pending = []
            pipeline.emit_task_results(
                SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None, pending, task, text,
                dict(terminal_usage), {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs", ctx=ctx,
            )
            return pending

    kwargs = dict(admission=_admission(), event=replace(_event(), delivery_reporting_version=1),
                  repo_dir=tmp_path, drive_root=tmp_path, agent_factory=lambda **_kw: Agent(),
                  gate=PresenceTurnGate(1))
    if refusal:
        refused = [_refused_turn(kwargs, refusal) for _ in range(2)]
        assert len(created) == 1 and refused[1].work_ref == refused[0].work_ref
        return refused[0], load_task_result(tmp_path, refused[0].turn_ref), text
    first = run_presence_turn(**kwargs)
    assert run_presence_turn(**kwargs) == first
    assert len(created) == 1
    return first, load_task_result(tmp_path, first.task_id), text


def _refused_turn(kwargs, code):
    with pytest.raises(PresenceTurnError) as raised:
        run_presence_turn(**kwargs)
    assert raised.value.code == code
    return raised.value


_UNKNOWN_FENCE = pytest.mark.parametrize("usage", [
    # The forced rail's own fallback text under the fence, worded provider_unavailable.
    {"execution_status": "infra_failed", "reason_code": "provider_unavailable", "terminal_origin": "host_notice",
     "_last_llm_error_kind": "provider_outcome_unknown"},
    # A round-one draft the rail salvaged as model_final behind the same fence.
    {"execution_status": "infra_failed", "reason_code": "provider_unavailable", "terminal_origin": "model_final",
     "_best_effort_extracted": True, "_last_llm_error_kind": "provider_outcome_unknown"},
    # A round still holding a transport-death record, whatever the later sticky kind.
    {"execution_status": "infra_failed", "reason_code": "provider_unavailable", "terminal_origin": "model_final",
     "_best_effort_extracted": True, "_last_llm_error_kind": "server_error",
     TRANSPORT_DEATHS_KEY: {"round_id": "round-1", "count": 1, "backoff_sec": 2.0}},
], ids=["host_fallback", "salvaged_draft", "death_record"])


@pytest.mark.parametrize("outcome", ["deferred", "message", "silent", "tool_delivered"])
@pytest.mark.parametrize("request_text", ["", "Outdated proposed reply"])
def test_failed_parent_and_cached_result_keep_admitted_child_pollable(tmp_path, outcome, request_text):
    result, stored, text = _failed_parent(tmp_path, outcome=outcome, request_text=request_text)
    assert result.outcome == "deferred" and result.work_ref == "managed-work"
    assert result.delivery_reporting_version == 1
    assert result.text == text
    assert stored["terminal_provider_notice"] == "Provider unavailable; child work remains admitted."
    assert "Outdated proposed reply" not in result.text
    assert stored["metadata"]["presence_outcome"] == "deferred"
    assert stored["metadata"]["presence_result_text"] == result.text
    assert stored["status"] == "failed" and stored["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert stored["reason_code"] == "provider_unavailable"


@_UNKNOWN_FENCE
@pytest.mark.parametrize("admission", ["scheduled", "unconfirmed"])
def test_unresolved_attempt_refuses_and_keeps_only_admitted_child_custody(tmp_path, usage, admission):
    """The same failed parent behind the unknown-outcome fence answered nothing: the Host refuses
    the event back to its transport (first call and replay), the real pipeline's marker is the
    durable reason, and an admitted child alone stays pollable — never the draft."""
    refused, stored, text = _failed_parent(tmp_path, admission=admission, accepted=True, usage=usage,
                                           refusal="presence_attempt_outcome_unknown")
    work_ref = "managed-work" if admission == "scheduled" else ""
    assert refused.work_ref == work_ref
    assert stored["status"] == "failed" and stored["result"] == text
    assert stored["reason_code"] == "provider_unavailable"
    assert stored["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert stored["metadata"]["presence_unknown_outcome"]["source"] == "provider_outcome_unknown_no_resend"
    assert stored["metadata"].get("presence_work_ref", "") == work_ref
    assert "presence_retry_proof" not in stored["metadata"]
    view = presence_result_from_stored(stored, refused.turn_ref)
    assert (view.outcome, view.text, view.work_ref) == ("silent", "", work_ref)


@pytest.mark.parametrize("admission", ["scheduled", "unconfirmed", "rejected", ""])
def test_forced_best_effort_requires_positive_admission_and_keeps_current_body(tmp_path, admission):
    usage = {"execution_status": "failed", "reason_code": "round_limit", "_best_effort_extracted": True,
             "terminal_origin": "model_final"}
    result, stored, text = _failed_parent(tmp_path, outcome="silent", admission=admission,
                                        usage=usage, accepted=True)
    assert result.outcome == ("deferred" if admission == "scheduled" else "message")
    assert result.work_ref == ("managed-work" if admission == "scheduled" else "")
    assert result.text == text
    assert stored["outcome_axes"]["execution"]["status"] == "best_effort"
    assert stored["reason_code"] == "round_limit"


@pytest.mark.parametrize("outcome", ["deferred", "silent", "tool_delivered"])
def test_successful_replacement_answer_does_not_inherit_old_outcome(tmp_path, outcome):
    result, stored, text = _failed_parent(tmp_path, outcome=outcome, usage={"terminal_origin": "model_final"})
    assert result.outcome == "message" and result.text == text
    assert result.work_ref == "managed-work"
    assert stored["status"] == "completed" and stored["outcome_axes"]["execution"]["status"] == "ok"


@pytest.mark.parametrize("outcome", ["silent", "tool_delivered"])
def test_accepted_successful_nonmessage_remains_nonmessage(tmp_path, outcome):
    result, stored, _text = _failed_parent(tmp_path, outcome=outcome, usage={"terminal_origin": "model_final"}, accepted=True)
    assert result.outcome == outcome and result.text == ""
    assert stored["status"] == "completed"
