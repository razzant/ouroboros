"""A run held by its job id still owns its own engine detail.

``POST /v2/runs`` answers 202 with only ``jobId`` when the run has not bound inside the
engine's start timeout, and the host keeps that job id as the run's handle
(``test_a_202_handle_without_a_run_id_is_a_live_run_not_a_failure``). The engine gives a run
both names and addresses it by either (control-api ``findRun``), but collection compared the
detail's ``runId`` alone: on 21 October a finished Astra research held as ``job-74ac4b509812``
was refused twice as "another run's" observation, and its custody never settled. A detail
naming another job and run is still refused.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_collection as body
from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_progress as progress
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _LiveRunStub,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
)
from tests._review_session_route_shared import fake_route  # noqa: F401 -- thread_gateway's fixture
from tests.test_delegate_body_collection import money, started, terminal  # noqa: F401
from tests.test_review_checking import author  # noqa: F401
from tests.test_review_thread_checking import _run, thread_gateway  # noqa: F401

JOB, RUN = "job-74ac4b509812", "run-c6cd4df3fc3a"


def _named(detail, *, job=JOB, run=RUN):
    """The engine's detail of one run: both of its names, as GET /v2/runs/:id serves them."""
    detail["summary"].update(jobId=job, runId=run)
    return detail


@pytest.mark.parametrize("held", [JOB, RUN])
def test_a_run_observes_and_collects_its_own_answer_under_either_name(tmp_path, money, held):  # noqa: F811
    entry = started(tmp_path, rid=held)
    detail = _named(terminal(RUN, "the research answer", seq=222))
    assert body.observe_run(tmp_path, entry, detail)
    assert current.run(tmp_path, held).terminal_state == "succeeded"
    result = body.collect_terminal(tmp_path, held, detail=detail)
    now = current.run(tmp_path, held)
    assert result["status"] == "collected" and now.settled and now.ledger_recorded and len(money) == 1
    assert body.retained_payload(tmp_path, now)["primary_output"]["text"] == "the research answer"


@pytest.mark.parametrize("held, job, run", [
    (JOB, "job-0123456789ab", "run-0123456789ab"),  # another job and the run it backs
    (JOB, "", RUN),                                  # a run naming no job is not this job's
    (RUN, JOB, "run-0123456789ab"),                  # held by run id: that job backs another run
], ids=["another-pair", "no-job-name", "job-of-another-run"])
def test_a_detail_naming_another_run_is_still_refused(tmp_path, held, job, run):
    entry = started(tmp_path, rid=held)
    detail = terminal(run)
    if job:
        detail["summary"]["jobId"] = job
    with pytest.raises(ValueError, match="another run"):
        body.observe_run(tmp_path, entry, detail)
    assert not current.run(tmp_path, held).terminal_state


def test_questions_of_a_run_held_by_its_job_id_are_observed(tmp_path):
    entry = started(tmp_path, rid=JOB)
    detail = {"summary": {"jobId": JOB, "runId": RUN, "state": "running"}, "lastSeq": 2,
              "pendingInteractions": [{"interactionId": "q1", "runId": RUN,
                                       "questions": [{"id": "file", "question": "which file?"}]}]}
    assert body.observe_run(tmp_path, entry, detail)
    assert current.run(tmp_path, JOB).pending_interaction_ids == ["q1"]


def test_delegate_wait_returns_and_settles_the_answer_of_a_run_held_by_its_job_id(tmp_path, monkeypatch):
    """The 21 October shape end to end: started as a 202 job, finished, collected by the wait."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    asked = []

    class _Finished(_LiveRunStub):
        def get_run(self, rid, **_kw):
            asked.append(rid)
            text = "the research answer"
            return {"lastSeq": 222, "primaryOutput": {"text": text, "bytes": len(text), "truncated": False},
                    "summary": {"jobId": JOB, "runId": RUN, "state": "succeeded", "spendUsd": 0.0,
                                "effectiveAccess": "readonly"}}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Finished())
    delegate._CUSTODY.clear()
    assert custody.record_started(tmp_path, delegate._RunCustody(
        run_id=JOB, task_id="t-a", route_id="r", model="m", project_id="p", project_owned=False,
        root_task_id="t-a", ledger_root=str(tmp_path)))
    out = json.loads(delegate._delegate_wait(_nanny_ctx(tmp_path), JOB, wait_sec=1))
    delegate._CUSTODY.clear()
    assert out["settlement"]["settled"] is True and out["settlement"]["ledger_recorded"] is True
    assert "the research answer" in json.dumps(out)
    assert current.run(tmp_path, JOB).settled
    assert set(asked) == {JOB}, "the engine is addressed by the name the host holds"


def test_the_live_executor_of_a_run_held_by_its_job_id_is_observed():
    from tests.test_executor_observation import _detail, _entry, _observation

    detail = _detail()
    detail["summary"]["jobId"] = "job-1"
    ctx = SimpleNamespace(task_id="child", task_attempt=0)
    advance = progress.WindowObservations().record(detail, detail["lastSeq"], 0)
    held = _entry(run_id="job-1")
    assert progress.executor_observation(ctx, "job-1", advance, detail, held) == _observation(run_id="job-1")
    detail["summary"]["jobId"] = "job-2"
    assert progress.executor_observation(ctx, "job-1", advance, detail, held) == {}


def test_the_final_attempt_of_a_run_held_by_its_job_id_is_read_and_no_other_runs(tmp_path):
    from ouroboros.gateways.claudexor import final_attempt_facts
    from tests.test_claudexor_observed_attempt import _write_telemetry

    attempts = [{"attempt_id": "a02", "harness_id": "codex", "observed_model": "gpt-6-astra"}]
    detail = _named(_write_telemetry(tmp_path / "own", attempts, run_id=RUN))
    assert final_attempt_facts(detail, JOB)["model"] == final_attempt_facts(detail, RUN)["model"] == "gpt-6-astra"
    other = "run-0123456789ab"
    stray = _named(_write_telemetry(tmp_path / "stray", attempts, run_id=other))
    assert final_attempt_facts(stray, JOB) == {}  # this run's detail, another run's telemetry
    foreign = _named(_write_telemetry(tmp_path / "foreign", attempts, run_id=other), job="job-0123456789ab", run=other)
    assert final_attempt_facts(foreign, JOB) == {}  # another run's detail lends no telemetry


def test_a_thread_turn_finds_the_run_its_job_id_backs_and_no_other():
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.review_thread_continuity import review_thread_receipt

    class Gateway:
        def get_thread(self, _thread_id):
            return {"thread": {"headRunId": RUN}, "turns": [{"id": "turn-1", "runId": RUN}], "sessions": []}

        def get_run_artifact(self, _run_id, _path):
            return b""

    receipt = review_thread_receipt(Gateway(), "thread-1", JOB, "turn-1", run_names={JOB, RUN})
    assert receipt["turn_id"] == "turn-1" and receipt["run_id"] == JOB
    for names in ((), {"job-0123456789ab", "run-0123456789ab"}):
        with pytest.raises(ReviewRouteUnavailable) as excinfo:
            review_thread_receipt(Gateway(), "thread-1", JOB, "turn-1", run_names=names)
        assert excinfo.value.code == "review_thread_receipt_missing"


def test_a_review_turn_answered_before_its_run_bound_keeps_its_thread_receipt(author, tmp_path, thread_gateway):  # noqa: F811
    """The thread turn handle carries only the job id; the turn records the run id."""
    physical = {}
    start, get = thread_gateway.start_thread_turn, thread_gateway.get_run

    def queued(thread_id, request, *, idempotency_key):
        handle = start(thread_id, request, idempotency_key=idempotency_key)
        job = "job-" + handle["turnId"]
        physical[job] = handle["runId"]
        return {"jobId": job, "threadId": thread_id, "turnId": handle["turnId"], "state": "queued"}

    def named(run_id, **kw):
        detail = get(run_id, **kw)
        detail["summary"].update(jobId=run_id, runId=physical[run_id])
        return detail

    thread_gateway.start_thread_turn, thread_gateway.get_run = queued, named
    thread_gateway.engine_version = "3.3.7"
    result = _run(author, tmp_path / "drive", None, access="readonly")
    assert result["run_id"] == "job-turn-1" and result["turn_id"] == "turn-1"
    assert result["thread_receipt"]["turn_id"] == "turn-1"


def test_a_continued_try_of_the_same_run_names_no_other_run():
    from ouroboros.delegate_continuation import terminal_continuity

    row = {"tryIndex": 2, "attemptId": "a02", "carrier": "native_moved", "cause": "vendor_limit",
           "from": {"runId": RUN, "attemptId": "a01", "profileId": "p1"}, "to": {"profileId": "p2"}}
    summary = {"jobId": JOB, "runId": RUN, "continuity": [row]}
    assert "of run" not in terminal_continuity(JOB, summary)["continuity"][0]
    summary["continuity"] = [{**row, "from": {**row["from"], "runId": "run-0123456789ab"}}]
    assert "of run run-0123456789ab" in terminal_continuity(JOB, summary)["continuity"][0]


def test_a_name_that_is_not_a_string_is_refused_without_raising(tmp_path):
    """Membership in the names must stay as tolerant as the old ``!=`` was."""
    from ouroboros.delegate_continuation import terminal_continuity
    from ouroboros.gateways.claudexor import final_attempt_facts
    from tests.test_claudexor_observed_attempt import _write_telemetry

    attempts = [{"attempt_id": "a02", "harness_id": "codex", "observed_model": "gpt-6-astra"}]
    assert final_attempt_facts(_named(_write_telemetry(tmp_path, attempts, run_id=[RUN])), JOB) == {}
    row = {"tryIndex": 2, "carrier": "native", "cause": "quota", "from": {"runId": {"id": RUN}}, "to": {}}
    assert "of run" in terminal_continuity(JOB, {"jobId": JOB, "runId": RUN, "continuity": [row]})["continuity"][0]
