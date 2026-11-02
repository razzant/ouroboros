"""Input custody and late result observation are independent of a failed turn."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from ouroboros.gateway import host_service
from ouroboros.presence_runner import PresenceTurnGate, presence_turn_task_id, run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_host_service_responsiveness import _event, _presence_app, _request, _turn


def _poll(app, binding, reference):
    request = _request(app)
    request.method = "GET"
    request.path_params = {"work_ref": reference}
    request.query_params = {"binding_id": binding}
    response = asyncio.run(host_service._api_presence_work(request))
    return response.status_code, json.loads(response.body)


def _host(tmp_path, *, cause="unknown", child=False, proof=False):
    started = []

    class Agent:
        def handle_task(self, task):
            started.append(task["id"])
            meta = dict(task["metadata"])
            if meta["presence"]["event"]["source_event_id"] == "old":
                if child:
                    meta["presence_work_ref"] = "child"
                    write_task_result(tmp_path, "child", "running", metadata={"presence": meta["presence"]})
                reason = "resource_refusal_no_resend" if cause == "quota" else "provider_unavailable"
                if cause == "unknown":
                    meta["presence_unknown_outcome"] = {"source": "provider_outcome_unknown_no_resend"}
                if proof:
                    meta["presence_retry_proof"] = {
                        "kind": "first_round_engine_not_started", "task_id": task["id"],
                        "event_identity": meta["presence_event_identity"],
                        "reset_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                    }
                write_task_result(tmp_path, task["id"], "failed", metadata=meta, reason_code=reason,
                                  terminal_origin="model_final", result="unselected draft")
            else:
                meta.update(presence_outcome="message", presence_result_text="new reply")
                write_task_result(tmp_path, task["id"], "completed", metadata=meta,
                                  terminal_origin="model_final", result="new reply")
            return [{"type": "presence_result", "outcome": "message", "text": "new reply"}]

    app, binding, ctx = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, agent_factory=lambda **kw: Agent(),
        gate=PresenceTurnGate(1), **kwargs))
    return app, binding, ctx, started


def test_retained_receipt_releases_next_event_preserves_original_and_late_child(tmp_path):
    app, binding, ctx, started = _host(tmp_path, child=True)
    old = presence_turn_task_id(binding, "old")
    first = asyncio.run(host_service._api_presence_turn(_request(app, {
        "binding_id": binding, "event": _event("old"), "delivery_reporting_version": 1})))
    receipt = json.loads(first.body)
    assert first.status_code == 409 and receipt["disposition"] == "retained"
    assert receipt["turn_ref"] == old and receipt["work_ref"] == "child"
    assert receipt["delivery_reporting_version"] == 1 and "outcome" not in receipt
    assert ctx.presence_turns.live() == []
    later = asyncio.run(_turn(app, binding, "new"))
    assert json.loads(later.body)["text"] == "new reply"
    again = asyncio.run(_turn(app, binding, "old"))
    assert json.loads(again.body)["disposition"] == "retained"
    assert started == [old, presence_turn_task_id(binding, "new")]
    status, ended = _poll(app, binding, old)
    assert status == 200 and ended["status"] == "failed"  # it can no longer change: observation ends
    assert ended["turn_ref"] == ended["work_ref"] == old and ended["child_work_ref"] == "child"
    assert (ended["outcome"], ended["text"]) == ("silent", "") and ended["code"] == "presence_attempt_outcome_unknown"
    assert _poll(app, binding, "child")[1]["status"] == "pending"
    child = load_task_result(tmp_path, "child")
    write_task_result(tmp_path, "child", "completed", metadata={**child["metadata"],
                      "presence_outcome": "message", "presence_result_text": "child answer"},
                      terminal_origin="model_final", result="child answer")
    assert _poll(app, binding, "child")[1]["text"] == "child answer"
    assert _poll(app, binding, old)[1]["status"] == "failed"
    # Projection guarantee only: a canonical answer stored on the row is served first (no production
    # writer adds one after an attempt ended; conforming transports stopped at the terminal above).
    # A canonical late publication may enrich the existing failed result with its
    # source-bound selected answer; task lifecycle remains monotonic. Observation
    # neither regenerates the old event nor rewrites it as a silent completion.
    previous = load_task_result(tmp_path, old)
    metadata = dict(previous["metadata"])
    metadata["presence_unknown_outcome"] = None  # canonical producer supersedes the unknown marker
    metadata.update(presence_outcome="message", presence_result_text="late original")
    write_task_result(tmp_path, old, "failed", metadata=metadata, reason_code="",
                      terminal_origin="model_final", result="late original")
    assert _poll(app, binding, old)[1]["text"] == "late original"
    assert len(started) == 2


@pytest.mark.parametrize("proof,disposition", [(False, "retained"), (True, "retry")])
def test_only_source_bound_no_start_proof_keeps_quota_event_retryable(tmp_path, proof, disposition):
    app, binding, _, started = _host(tmp_path, cause="quota", proof=proof)
    for _ in range(2):
        response = asyncio.run(_turn(app, binding, "old"))
        body = json.loads(response.body)
        assert response.status_code == 409 and body["disposition"] == disposition
        assert body["code"] == "presence_resources_unavailable"
    assert len(started) == 1



def test_work_poll_distinguishes_live_then_retained_lost_turn_and_late_terminal(tmp_path):
    """One real Host author: live pending, lost retained uncertainty, then its canonical late answer."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from starlette.testclient import TestClient
    from tests.test_presence_orphan_replay import _answering_agent, _complete, _host as orphan_host, _host_body

    agents, calls = [], []
    app, binding = orphan_host(tmp_path, agents)
    body = _host_body(binding)
    reference = presence_turn_task_id(binding, body["event"]["source_event_id"])
    started, release = Event(), Event()

    def hold(_task):
        started.set()
        assert release.wait(60)

    agents.append(_answering_agent(calls, "", tmp_path, during=hold, lost=True))
    with TestClient(app) as client:
        headers = {"X-Skill-Token": "presence-token"}
        def post():
            return client.post("/presence/turn", headers=headers, json=body)
        def poll():
            return client.get(f"/presence/work/{reference}", headers=headers, params={"binding_id": binding})

        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(post)
            try:
                assert started.wait(60)
                live = poll()
                assert live.status_code == 202 and live.json()["status"] == "pending"
                assert "presence_recovery_recorded" not in load_task_result(tmp_path, reference)
            finally:
                release.set()
            assert running.result(timeout=60).status_code == 500

        refused = post()
        assert refused.status_code == 409 and refused.json()["disposition"] == "retained"
        lost = load_task_result(tmp_path, reference)
        assert lost["status"] == "running" and lost["presence_recovery_recorded"]
        ended = poll()
        facts = ended.json()
        assert ended.status_code == 200 and facts["status"] == "failed"  # the gate proved it lost
        assert facts["code"] == "presence_attempt_outcome_unknown" and "disposition" not in facts
        assert facts["turn_ref"] == facts["work_ref"] == reference
        assert facts["source_event_id"] == body["event"]["source_event_id"]
        assert facts["recorded_status"] == "running" and facts["delivery_reporting_version"] == 1
        assert (facts["outcome"], facts["text"]) == ("silent", "")
        assert poll().json() == facts

        # Projection guarantee only, as above: a canonical answer stored after the end is served first.
        _complete(calls[0], "Late canonical answer", tmp_path)
        completed = poll()
        assert completed.status_code == 200 and completed.json()["status"] == "completed"
        assert completed.json()["outcome"] == "message" and completed.json()["text"] == "Late canonical answer"
        assert poll().json() == completed.json()
        assert post().json()["text"] == "Late canonical answer"
        assert load_task_result(tmp_path, reference)["presence_recovery_recorded"] == lost["presence_recovery_recorded"]
    assert len(calls) == 1 and agents == []


@pytest.fixture(autouse=True)
def _focus_admission_enabled(monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available", "remaining_usd": 10})
