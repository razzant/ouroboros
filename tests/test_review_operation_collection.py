"""Pure collection of an already-paid review seat, and the exact bytes a task emitted.

Collection branches BEFORE the ordinary runner (``review_operation._collect_pending_row``,
which ``collect_recorded_review_actor`` reaches once the original prompt is proven): exact
producer CAS, a live local worker, or an attach-only read of a proven delegated run, parsed
locally. Spies prove no model call, send, re-post, daemon ensure, cancel or retirement.
The send handler's receipts (``terminal_answer_receipts``) are the only proof of which
answer bytes a task emitted.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import review_operation
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
from ouroboros.task_results import write_task_result
from ouroboros.triad_review import parse_review_findings
from tests._cancel_intents_shared import DeliveryBus

TASK = "collect-root"
# A live array-contract surface: the collector is the same for every surface that
# collects a recorded seat (commit gate, plan review, skill review).
SURFACE = "scope_review"
PASS = json.dumps([{"item": "the reviewed answer", "verdict": "PASS", "severity": "advisory",
                    "reason": "Independent review passes"}])


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.fixture
def forbid_effects(monkeypatch):
    """Every path that could buy, send, start, cancel, acknowledge or retire fails loudly."""
    def forbidden(name):
        def fail(*_args, **_kwargs):
            pytest.fail(f"pure collection reached {name}")
        return fail

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", forbidden("the ordinary runner"))
    monkeypatch.setattr("ouroboros.review_substrate.ReviewCoordinator.run", forbidden("the coordinator"))
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", forbidden("a model call"))
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", forbidden("daemon ensure/start"))
    monkeypatch.setattr("ouroboros.review_verdict_extraction._extract_verdict_via_light_model",
                        forbidden("Light extraction"))


def _ctx(root):
    return SimpleNamespace(task_id=TASK, task_attempt=1, drive_root=root, budget_drive_root=root,
                           task_metadata={}, pending_events=[], event_queue=queue.Queue())


def _collect(run, root, usage_ctx, controller=None):
    """Collect the run's one recorded seat purely, from its recorded request, roster row
    and frozen actor row: returns the actor row and its typed collection fact."""
    request = ReviewRequest(**copy.deepcopy(run["request"]))
    roster = copy.deepcopy(run["slot_roster"][0])
    slot = ReviewSlot(**{**roster, "route": ReviewRouteKind(roster["route"])})
    actor, fact = review_operation._collect_pending_row(
        root, request, slot, copy.deepcopy(run["actors"][0]), usage_ctx, controller)
    return dataclasses.asdict(actor), fact


def _verdict(actor):
    """The verdict a panel reads from a collected answer."""
    return parse_review_findings(actor["raw_text"])[2]


def _session_run(slot_id="s", operation_id="op-session", controller=None):
    request = ReviewRequest(surface=SURFACE, task_id=TASK, goal="goal", subject="reviewed answer A",
                            evidence={"requirement": "exact"}, retry_key="wave-session")
    slot = ReviewSlot(slot_id=slot_id, model="codex", route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex")
    return json.loads(json.dumps({
        "request": dataclasses.asdict(request), "slot_roster": [dataclasses.asdict(slot)],
        "actors": [{"slot_id": slot_id, "model": "codex", "status": "error", "operation_id": operation_id,
                    "operation_state": "pending_dispatch", "late_result_pending": True,
                    "usage": {"review_controller": controller or {"pid": _dead_pid(), "session": "gone"}}}]}))


class ObservingGateway:
    """The owned daemon as the collector may see it: reads only."""

    def __init__(self, detail, calls):
        self.detail, self.calls = detail, calls

    def get_run(self, run_id, *, timeout_sec=None):
        self.calls.append(("get_run", run_id))
        return self.detail

    def get_run_artifact(self, run_id, path):
        self.calls.append(("artifact", path))
        return b""

    def close(self):
        self.calls.append(("close",))

    def __getattr__(self, name):
        pytest.fail(f"pure collection called gateway.{name}")


def _started(root, *, operation_id="op-session", slot_id="s", run_id=""):
    custody.emit(root, custody.START_REQUESTED, {
        "invocation_id": f"inv-{operation_id}", "task_id": TASK, "operation_id": operation_id, "slot_id": slot_id,
        "surface": SURFACE, "root_task_id": TASK, "request": {"prompt": "review", "harnesses": ["codex"]}})
    if run_id:
        custody.emit(root, custody.STARTED, {"run_id": run_id, "invocation_id": f"inv-{operation_id}",
                                             "task_id": TASK, "surface": SURFACE})


def _observe(monkeypatch, detail):
    calls = []
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway",
                        lambda: (calls.append(("handshake",)), ObservingGateway(detail, calls))[1])
    return calls


def test_pending_invocation_without_run_is_unknown_and_never_reposted(tmp_path, forbid_effects, monkeypatch):
    calls = _observe(monkeypatch, {})
    _started(tmp_path)
    actor, fact = _collect(_session_run(), tmp_path, _ctx(tmp_path))
    assert actor["operation_state"] == "custody_lost" and "never re-posted" in actor["error"]
    assert fact == "unavailable"
    assert calls == [], "an unbound invocation is never even observed, let alone re-POSTed"
    assert actor["operation_state"] not in {"pending_dispatch", "in_flight"}, "a gap is not a live wait"


def test_absent_start_request_is_unknown_and_only_a_definite_refusal_proves_no_send(tmp_path, forbid_effects, monkeypatch):
    calls = _observe(monkeypatch, {})
    actor, fact = _collect(_session_run(), tmp_path, _ctx(tmp_path))
    assert actor["operation_state"] == "custody_lost" and "cannot prove nothing was sent" in actor["error"]
    assert fact == "unavailable" and calls == [], "absence is never proof"
    # The daemon's own definite refusal of this exact invocation is a positive receipt.
    _started(tmp_path)
    custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "inv-op-session", "task_id": TASK,
                                                  "definite": True, "error": "route refused"})
    refused, fact = _collect(_session_run(), tmp_path, _ctx(tmp_path))
    assert refused["operation_state"] == "not_dispatched"
    assert fact == "never_dispatched" and calls == []


def test_an_unverifiable_controller_is_never_presumed_dead(tmp_path, forbid_effects, monkeypatch):
    """A live pid whose birth was never recorded may still be the controller: only
    exact proofs settle its rows; nothing becomes an unknown-outcome gap early."""
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        unverifiable = {"pid": worker.pid, "session": "other"}
        calls = _observe(monkeypatch, {"summary": {"state": "running"}})
        _actor, fact = _collect(_session_run(controller=unverifiable), tmp_path, _ctx(tmp_path))
        assert fact == "deferred" and calls == [], "no start on record: still owed"
        _started(tmp_path, run_id="run-proven")
        _actor, fact = _collect(_session_run(controller=unverifiable), tmp_path, _ctx(tmp_path))
        assert fact == "deferred" and ("get_run", "run-proven") in calls
        api = _session_run(controller=unverifiable)
        api["slot_roster"][0]["route"] = ReviewRouteKind.API_CHAT.value
        api["actors"][0]["operation_id"] = "op-api"
        assert _collect(api, tmp_path, _ctx(tmp_path))[1] == "deferred"
    finally:
        worker.kill()
        worker.wait()


@pytest.mark.parametrize("answer, method", [(PASS, "strict"), ("Looks fine overall, no blocking issues.", "parse_unavailable")])
def test_proven_existing_run_is_read_attach_only_and_parsed_locally(tmp_path, forbid_effects, monkeypatch, answer, method):
    detail = {"summary": {"state": "succeeded", "outputConformance": ""},
              "primaryOutput": {"text": answer, "truncated": False}}
    calls = _observe(monkeypatch, detail)
    _started(tmp_path, run_id="run-live")
    actor, fact = _collect(_session_run(), tmp_path, _ctx(tmp_path))
    assert calls == [("handshake",), ("get_run", "run-live"), ("close",)], \
        "one disclosed protocol handshake, then reads only"
    assert actor["usage"]["collection"] == "attach_only_observation"
    assert actor["usage"]["verdict_method"] == method and actor["raw_text"] == answer, "the answer is kept whole"
    if method == "strict":
        assert _verdict(actor) == "PASS" and fact == "collected"
    else:
        assert actor["parse_status"] == "malformed" and not actor.get("semantic_verdict")
    assert actor["operation_state"] not in {"pending_dispatch", "in_flight"}


@pytest.mark.parametrize('telemetry', ['observed', 'missing', 'wrong_run', 'duplicate_attempt'])
def test_cold_collection_preserves_only_final_attempt_identity(tmp_path, forbid_effects, monkeypatch, telemetry):
    from ouroboros.review_projection import compact_review_projection

    final = tmp_path / 'run' / 'final'
    final.mkdir(parents=True)
    last = {'attempt_id': 'a02', 'harness_id': 'actual-harness',
            'observed_model': 'actual-model', 'profile_id': 'actual-profile'}
    record = {'run_id': 'wrong' if telemetry == 'wrong_run' else 'run-live', 'final_attempt_id': 'a02',
              'attempts': [{'attempt_id': 'a01', 'harness_id': 'prior-harness', 'observed_model': 'prior-model'}, last]}
    if telemetry == 'duplicate_attempt':
        record['attempts'].append(last)
    if telemetry != 'missing':
        (final / 'telemetry.yaml').write_text(json.dumps(record), encoding='utf-8')
    detail = {'summary': {'state': 'succeeded', 'runDir': str(final.parent), 'model': 'request-echo'},
              'primaryOutput': {'text': PASS, 'truncated': False}}
    calls = _observe(monkeypatch, detail)
    _started(tmp_path, run_id='run-live')
    run = _session_run()
    run['slot_roster'][0].update(model='codex=requested-model', session_target='codex=requested-model')
    actor, _fact = _collect(run, tmp_path, _ctx(tmp_path))
    usage = actor['usage']
    assert _verdict(actor) == 'PASS'
    expected = telemetry == 'observed'
    assert usage['resolved_model'] == ('actual-model' if expected else '')
    assert usage['applied_profile'] == ('actual-profile' if expected else '')
    assert usage['delegated_route'] == ('actual-harness' if expected else '')
    assert usage['observed_attempt'] == ({'attempt_id': 'a02', 'harness_id': 'actual-harness',
                                          'model': 'actual-model', 'profile_id': 'actual-profile'} if expected else {})
    reasons = {delta['reason'] for delta in usage['capability_delta']}
    assert reasons == ({'session_ran_off_pinned_route', 'session_route_resolves_its_own_model'} if expected
                       else {'session_route_observation_unavailable'})
    # The panel projection a task result stores names only the final attempt's facts.
    projected = compact_review_projection([{'request': run['request'], 'actors': [actor]}])['panels'][0]['actors'][0]
    assert projected['model'] == ('actual-model' if expected else '')
    assert projected['executions'] == ([{'kind': 'harness', 'harness_id': 'actual-harness', 'model': 'actual-model'}]
                                       if expected else [{'kind': 'harness'}])
    assert calls == [('handshake',), ('get_run', 'run-live'), ('close',)]


def test_a_run_still_in_flight_stays_deferred_and_is_never_cancelled(tmp_path, forbid_effects, monkeypatch):
    calls = _observe(monkeypatch, {"summary": {"state": "running"}})
    _started(tmp_path, run_id="run-busy")
    run = _session_run()
    actor, fact = _collect(run, tmp_path, _ctx(tmp_path))
    assert fact == "deferred" and actor["operation_state"] in {"pending_dispatch", "in_flight"}
    assert ("get_run", "run-busy") in calls and not [c for c in calls if c[0] not in {"handshake", "get_run", "close"}]
    assert actor["usage"]["review_controller"] == run["actors"][0]["usage"]["review_controller"]


def test_live_process_controller_keeps_the_row_deferred_without_reading_anything(tmp_path, forbid_effects, monkeypatch):
    from ouroboros.platform_layer import process_start_time

    calls = _observe(monkeypatch, {})
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _started(tmp_path, run_id="run-owned-elsewhere")
        exact = {"pid": worker.pid, "birth": process_start_time(worker.pid), "session": "other"}
        _actor, fact = _collect(_session_run(controller=exact), tmp_path, _ctx(tmp_path))
        _reused, reused_fact = _collect(_session_run(controller={**exact, "birth": "another process"}),
                                        tmp_path, _ctx(tmp_path))
    finally:
        worker.kill()
        worker.wait()
    assert fact == "deferred"
    # Only the reused-pid row (its recorded controller proven gone) was read, attach-only;
    # the live exact controller's row was not touched at all.
    assert calls == [("handshake",), ("get_run", "run-owned-elsewhere"), ("close",)]
    assert reused_fact == "deferred", "the run is still running: deferred, never cancelled"


class _HeldModel:
    def __init__(self):
        self.release, self.calls = threading.Event(), 0

    def chat(self, **_kwargs):
        self.calls += 1
        assert self.release.wait(10)
        return {"content": PASS}, {"prompt_tokens": 1, "completion_tokens": 1}


def _released_panel(root, ctx, model, *, retry_key="wave-cas", subject="reviewed answer A"):
    request = ReviewRequest(surface=SURFACE, task_id=TASK, goal="goal", subject=subject,
                            evidence={"requirement": "exact"}, retry_key=retry_key, drain_deadline=time.monotonic())
    first = run_review_request(request, slots=[ReviewSlot(slot_id="a", model="model/a", timeout_sec=20)],
                               drive_root=root, usage_ctx=ctx, llm=model)
    return json.loads(json.dumps(dataclasses.asdict(first)))


def _settle(model, run):
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK

    model.release.set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            if not any(entry.operation_id == run["actors"][0]["operation_id"] for entry in _ACTIVE.values()):
                return
        time.sleep(0.01)
    raise AssertionError("worker did not settle")


def test_exact_cas_collection_crosses_processes_and_a_lost_api_worker_is_unknown(tmp_path, monkeypatch):
    model = _HeldModel()
    write_task_result(tmp_path, TASK, "running", chat_id=3)
    run = _released_panel(tmp_path, _ctx(tmp_path), model)
    assert run["actors"][0]["usage"]["review_controller"]["pid"] > 0, "a still-owed row names its controller"
    _settle(model, run)
    forbidden = pytest.fail
    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", lambda *a, **k: forbidden("runner"))
    # A cold context of another process: the exact producer CAS is the whole answer.
    cold = SimpleNamespace(task_id=TASK, task_attempt=1, drive_root=tmp_path, budget_drive_root=tmp_path, task_metadata={})
    actor, fact = _collect(run, tmp_path, cold)
    assert _verdict(actor) == "PASS" and fact == "collected"
    # An API worker killed with its process left no producer: unknown, never re-sent.
    lost = json.loads(json.dumps(run))
    lost["actors"][0].update(operation_id="op-killed", recovery_binding={})
    lost["actors"][0]["usage"]["review_controller"] = {"pid": _dead_pid(), "session": "gone"}
    gap, _fact = _collect(lost, tmp_path, cold)
    assert gap["operation_state"] == "custody_lost" and "never re-sent" in gap["error"]
    assert model.calls == 1


@pytest.fixture
def fresh_sends(monkeypatch):
    """The send handler's process-local dedupe starts empty for each test."""
    import collections

    from supervisor import events_chat_delivery as chat

    monkeypatch.setattr(chat, "_DELIVERED_MESSAGE_IDS", collections.deque(maxlen=256))


def _send_ctx(root, sends, *, fail=False):
    from ouroboros.utils import append_jsonl

    def send(chat_id, text, **_kwargs):
        if fail:
            raise RuntimeError("chat transport down")
        sends.append((chat_id, text))

    return SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, send_with_budget=send, append_jsonl=append_jsonl)


def _answer_event(text):
    from supervisor.terminal_delivery import delivery_id_for

    return {"type": "send_message", "chat_id": 3, "task_id": TASK, "text": text, "format": "markdown",
            "delivery_id": delivery_id_for(TASK, text)}


@pytest.mark.parametrize("sent", ["answer B", "answer A"])
def test_the_send_receipt_names_the_exact_sent_bytes_and_the_routed_room(tmp_path, monkeypatch, fresh_sends, sent):
    """The real send handler is the producer: owed is not delivered; after the send its
    receipt names the exact text and the ROUTED room, retained in the immutable store."""
    from ouroboros.artifacts import read_actor_source_bytes
    from supervisor import events_chat_delivery as chat
    from supervisor.terminal_delivery import register_pending_delivery, terminal_answer_receipts

    # The mutable result points at A whatever was sent; it is never byte proof.
    write_task_result(tmp_path, TASK, "completed", chat_id=3, result="answer A")
    event = _answer_event(sent)
    assert register_pending_delivery(tmp_path, event)
    owed = terminal_answer_receipts(tmp_path, TASK)
    assert owed["state"] == "owed" and owed["delivered"] == []
    # The task was bound to a project room after admission: the send goes there.
    monkeypatch.setattr(chat, "_bound_project_chat_id", lambda *_args: 42)
    sends = []
    chat._handle_send_message(dict(event), _send_ctx(tmp_path, sends))
    assert sends == [(42, sent)]
    emitted = terminal_answer_receipts(tmp_path, TASK)
    assert emitted["state"] == "delivered" and emitted["owed_delivery_ids"] == []
    receipt = emitted["delivered"][0]
    assert receipt["chat_id"] == 42 and receipt["text_sha256"] == hashlib.sha256(sent.encode()).hexdigest()
    retained = json.loads(read_actor_source_bytes(tmp_path, TASK, receipt["source_ref"]))
    assert (retained["text"], retained["chat_id"], retained["delivery_id"]) == (sent, 42, event["delivery_id"])


def test_an_unconfirmed_or_uncaptured_send_proves_no_bytes_and_a_replay_captures_the_same(tmp_path, monkeypatch,
                                                                                         fresh_sends):
    from supervisor import events_chat_delivery as chat
    from supervisor import terminal_delivery as registry
    from supervisor.terminal_delivery import terminal_answer_receipts

    write_task_result(tmp_path, TASK, "completed", chat_id=3, result="answer B")
    event = _answer_event("answer B")
    registry.register_pending_delivery(tmp_path, event)
    # The send raised: nothing is registered, the answer stays owed, the version unknown.
    chat._handle_send_message(dict(event), _send_ctx(tmp_path, [], fail=True))
    assert terminal_answer_receipts(tmp_path, TASK)["state"] == "owed"
    # The send landed but capture failed: retain the exact sent obligation, without proof yet.
    sends = []
    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.artifacts.store_actor_source_bytes",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("store unavailable")))
        chat._handle_send_message(dict(event), _send_ctx(tmp_path, sends))
    fact = terminal_answer_receipts(tmp_path, TASK)
    assert sends == [(3, "answer B")] and not registry.already_delivered(tmp_path, event["delivery_id"])
    assert fact["state"] == "owed" and fact["unverified_delivery_ids"] == []
    outbox = DeliveryBus()
    assert registry.replay_pending_deliveries(tmp_path, event_queue=outbox) == [event["delivery_id"]]
    assert outbox.empty() and sends == [(3, "answer B")], "the duty completes capture without another send"
    captured = terminal_answer_receipts(tmp_path, TASK)
    assert captured["state"] == "delivered"
    assert captured["delivered"][0]["text_sha256"] == hashlib.sha256(b"answer B").hexdigest()
    # An owed row replayed through the ordinary outbox reaches the same handler and receipt.
    other = TASK + "-replayed"
    write_task_result(tmp_path, other, "completed", chat_id=3, result="answer C")
    replay = {**_answer_event("answer C"), "task_id": other,
              "delivery_id": registry.delivery_id_for(other, "answer C")}
    registry.register_pending_delivery(tmp_path, replay)
    monkeypatch.setattr(registry, "_replay_due", lambda _row: True)
    outbox = DeliveryBus()  # replay admits a copy only through the bus's membership
    assert registry.replay_pending_deliveries(tmp_path, event_queue=outbox) == [replay["delivery_id"]]
    chat._handle_send_message(outbox.get_nowait(), _send_ctx(tmp_path, sends))
    replayed = terminal_answer_receipts(tmp_path, other)
    assert replayed["state"] == "delivered"
    assert replayed["delivered"][0]["text_sha256"] == hashlib.sha256(b"answer C").hexdigest()
    assert terminal_answer_receipts(tmp_path, TASK) == captured, "receipts are per task and survive rewrites"


@pytest.mark.parametrize("state,error_count", [("cancelled", 0), ("interrupted", 0), ("failed", 1)])
def test_cold_terminal_loss_keeps_its_type_without_blaming_restart_on_reviewer(
        tmp_path, forbid_effects, monkeypatch, state, error_count):
    from ouroboros.review_execution_projection import review_actor_progress_text
    from ouroboros.review_records import ReviewActorRecord
    from ouroboros.observability import read_call_payload

    _observe(monkeypatch, {"summary": {"state": state}})
    _started(tmp_path, run_id="run-ended")
    actor, fact = _collect(_session_run(), tmp_path, _ctx(tmp_path))
    assert fact == "collected" and actor["outcome"]["operation_id"] == "op-session"
    assert read_call_payload(tmp_path, **actor["outcome"]["source_ref"])[0]["producer_complete"]
    events = tmp_path / "logs" / "events.jsonl"
    errors = [json.loads(line) for line in events.read_text().splitlines()
              if json.loads(line).get("type") == "review_model_error"] if events.exists() else []
    assert len(errors) == error_count
    if state in {"cancelled", "interrupted"}:
        from ouroboros.review_substrate import compact_review_projection

        assert actor["failure_code"] == "review_run_cancelled"
        assert actor["transport_status"] == actor["outcome"]["status"] == "interrupted"
        card = compact_review_projection([{"request": {"surface": SURFACE}, "actors": [actor]}])["panels"][0]["actors"][0]
        assert card["transport_status"] == "interrupted" and card["parse_status"] == "unavailable"
        assert not card["semantic_verdict"]
        text = review_actor_progress_text(SURFACE, "finished", ReviewSlot("s", "codex"), ReviewActorRecord(**actor))
        assert "answer may be lost" in text and "didn't answer" not in text
    else:
        assert actor["failure_code"] == "review_run_failed"
        assert actor["transport_status"] == "provider_transport_error"


def test_cold_collection_reads_saved_question_and_never_current_contract(tmp_path, forbid_effects, monkeypatch):
    from ouroboros.observability import persist_call
    from ouroboros import triad_review
    from ouroboros.review_dispatch import review_operation_binding, review_reconciliation_identity

    run = _session_run()
    request = ReviewRequest(**run["request"])
    request.policy = {"parts": ["result"], "output_shape": "array"}
    slot = ReviewSlot(**{**run["slot_roster"][0], "route": ReviewRouteKind.AGENT_SESSION})
    request.reconciliation_identity = review_reconciliation_identity(request, [slot], root_task_id=TASK)
    binding = review_operation_binding(request, slot, "op-session")
    persist_call(tmp_path, task_id=TASK, call_id="op-session_prompt", call_type="review_prompt",
                 payload={"request": dataclasses.asdict(request), "slot": dataclasses.asdict(slot)},
                 manifest={"review_operation_binding": binding})
    _started(tmp_path, run_id="run-result")
    _observe(monkeypatch, {"summary": {"state": "succeeded"}, "primaryOutput": {"text": "[]\nNO_FINDINGS"}})
    monkeypatch.setitem(triad_review.REVIEW_OUTPUT_SHAPES,
                        SURFACE, "two_part")
    ctx = _ctx(tmp_path)
    actor = review_operation.collect_recorded_review_actor(tmp_path, task_id=TASK, surface=SURFACE,
        row=run["actors"][0], retry_key=request.retry_key, usage_ctx=ctx)
    assert ctx._last_review_slot_executions["s"]["status"] == "ok"
    assert actor["outcome"]["parts"] == ["result"]
    assert actor["outcome"]["answers"]["result"]["verdict"] == "PASS"
    assert actor["parse_status"] == "valid" and actor["usage"]["collection"] == "attach_only_observation"
