"""One declared producer contract drives durable records, gates and projections."""
from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros import review_custody, triad_review
from ouroboros.observability import read_call_payload
from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot
from ouroboros.review_substrate import ReviewCoordinator, compact_review_projection
from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS


def _body(critical=False):
    return json.dumps({"change": [{"item": "rollback", "verdict": "FAIL", "severity": "critical",
                                   "reason": "restore loses the prior value"}] if critical else [],
                       "change_clean": not critical,
                       "coupling": [{"item": item, "verdict": "PASS", "severity": "advisory",
                                     "reason": "checked callers"} for item in sorted(SCOPE_REQUIRED_ITEMS)]})  # sorted: the id must not depend on the hash seed


def _errors(root):
    path = root / "logs" / "events.jsonl"
    return [row for line in path.read_text().splitlines() if (row := json.loads(line))["type"] == "review_model_error"] if path.exists() else []


def _produce(root, text, parts, *, cost=None):
    class Answer:
        def chat(self, **_kwargs):
            return {"content": text}, {"provider": "openai", "resolved_model": "observed/model", "cost": cost}

    ctx = SimpleNamespace(drive_root=root, task_id="truth", task_metadata={}, event_queue=queue.Queue(), pending_events=[])
    request = ReviewRequest(surface="multi_model_review", task_id="truth", goal="review", policy={"parts": parts})
    slot = ReviewSlot("seat", "requested/model", role_hint="Independent reviewer", native_retrieval_override=False)
    run = ReviewCoordinator(llm=Answer(), drive_root=root, usage_ctx=ctx).run(request, [slot])
    return run, ctx


@pytest.mark.parametrize("text,parts,verdict,parse,errors", [
    (_body(True), ["change", "coupling"], "FAIL", "valid", 0),
    (_body(), ["change", "coupling"], "PASS", "valid", 0),
    ("[]\nNO_FINDINGS", ["result"], "PASS", "valid", 0),
    ("[]\nNO_FINDINGS", ["skill"], "PASS", "valid", 0),
    ("not an answer", ["result"], "DEGRADED", "malformed", 1),
])
def test_producer_consumer_and_cas_have_one_contract_truth(tmp_path, monkeypatch, text, parts, verdict, parse, errors):
    run, ctx = _produce(tmp_path, text, parts)
    [actor] = run.actors
    outcome = actor["outcome"]
    assert run.aggregate_signal == verdict
    assert actor["parse_status"] == outcome["parse"] == parse
    assert actor["transport_status"] == outcome["transport"] == "success"
    assert actor["model"] == "observed/model" and actor["provider"] == "openai"
    assert actor["actor_role"] == "Independent reviewer"
    assert actor["quorum_contribution"] == (parse == "valid")
    assert actor["enforcement_impact"] == ("veto" if verdict == "FAIL" else "supports_pass" if verdict == "PASS" else "abstains")
    assert {"outcome_id", "operation_id", "slot_id", "contract_id", "parts", "answers", "findings", "source_ref", "usage_ref"} <= outcome.keys()
    manifest, payload, _ = read_call_payload(tmp_path, **outcome["source_ref"])
    assert manifest["producer_complete"] is True
    assert payload["producer_outcome"]["outcome"] == outcome
    assert ctx._last_review_slot_executions["seat"]["effective"]["model"] == "requested/model"
    assert len(_errors(tmp_path)) == errors

    def no_parse(*_a, **_kw):
        pytest.fail("a consumer reparsed the producer's answer")
    monkeypatch.setattr(triad_review, "extract_json_array", no_parse)
    monkeypatch.setattr(triad_review, "parse_two_part_answer", no_parse)
    parsed = triad_review.parse_seat_answers({"results": [{**actor, "text": "malformed new settings"}]}, {"seat": ["change"]})
    assert parsed.actor_records[0].answers == outcome["answers"]
    assert parsed.actor_records[0].parts == parts
    triad_review.emit_review_model_error_events(ctx, parsed, source="multi_model_review")
    assert len(_errors(tmp_path)) == errors
    [projected] = compact_review_projection([asdict(run)])["panels"][0]["actors"]
    assert projected["parse_status"] == parse
    assert projected["semantic_verdict"] == (verdict if parse == "valid" else "")
    assert projected["provider"] == actor["provider"]


@pytest.mark.parametrize("state,transport,error_count", [
    ("in_flight", "", 0), ("pending_dispatch", "", 0),
    ("late_settled", "interrupted", 0), ("custody_lost", "provider_transport_error", 1),
    ("settled", "provider_transport_error", 1),
])
def test_pending_and_interruption_are_not_model_losses(tmp_path, state, transport, error_count):
    from ouroboros.review_seat_outcome import finalize_seat_outcome

    actor = ReviewActorRecord(slot_id="seat", model="m", status="error", operation_id="operation",
                              operation_state=state, late_result_pending=state in {"in_flight", "pending_dispatch", "custody_lost"},
                              transport_status=transport, error="no completed answer")
    request = ReviewRequest(surface="multi_model_review", task_id="truth", goal="g", policy={"parts": ["result"]})
    finalize_seat_outcome(request, ReviewSlot("seat", "m"), actor)
    triad_review.emit_review_run_errors(None, request, [actor], drive_root=tmp_path)
    triad_review.emit_review_run_errors(None, request, [actor], drive_root=tmp_path)
    assert len(_errors(tmp_path)) == error_count
    if state in {"in_flight", "pending_dispatch"}:
        assert actor.outcome["status"] == "pending" and actor.parse_status == "awaiting"
    if transport == "interrupted":
        assert actor.outcome["status"] == "interrupted" and actor.parse_status == "unavailable"


@pytest.mark.parametrize("paused", [True, False])
def test_unsent_pause_refusal_is_not_a_reviewer_failure(tmp_path, paused):
    from ouroboros.review_seat_outcome import finalize_seat_outcome

    actor = ReviewActorRecord("seat", "m", "not_dispatched", operation_id="refused",
        operation_state="not_dispatched", usage={"owner_pause_preparing": paused}, error="unsent")
    request = ReviewRequest(surface="multi_model_review", goal="g", policy={"parts": ["result"]})
    finalize_seat_outcome(request, ReviewSlot("seat", "m"), actor)
    triad_review.emit_review_run_errors(None, request, [actor], drive_root=tmp_path)
    assert len(_errors(tmp_path)) == (0 if paused else 1)


def test_quorum_uses_assigned_seats_and_ignores_additional():
    from ouroboros.review_ledger import reduce_verdict

    records = [triad_review.ReviewActorRecord(model_id=str(i), status="responded" if i == 0 else "error",
                                             raw_text="", slot_id=str(i), parts=["result"],
                                             answers={"result": {"status": "responded", "verdict": "PASS"}} if i == 0 else {})
               for i in range(3)]
    panel = triad_review.ParsedTriadReview([], ["0"], records)
    assert not panel.quorum_met
    records[1].status, records[1].answers = "responded", records[0].answers
    assert panel.quorum_met
    records[1].additional = True
    rows = [record.to_dict() for record in records if not record.additional]
    quorum = reduce_verdict(rows)["quorum"]
    assert panel.quorum_met == (quorum["responded"] >= max(1, quorum["required"]))


def test_answer_progress_precedes_packet_parse(tmp_path, monkeypatch):
    from ouroboros.review_execution_projection import _reviewer_verb
    from ouroboros import review_seat_outcome

    observed = []
    parse = review_seat_outcome.finalize_seat_outcome
    monkeypatch.setattr(review_seat_outcome, "answer_received", lambda *_a: observed.append("received"))
    def finish(*args):
        if not args[-1].outcome:
            assert observed == ["received"]
            observed.append("parsed")
        return parse(*args)
    monkeypatch.setattr(review_seat_outcome, "finalize_seat_outcome", finish)
    _produce(tmp_path, "[]\nNO_FINDINGS", ["result"])
    assert observed == ["received", "parsed"]
    actor = ReviewActorRecord("s", "m", "ok")
    assert _reviewer_verb("multi_model_review", "finished", actor) == "answer received, verdict being read"
    actor.parse_status = "valid"
    assert _reviewer_verb("multi_model_review", "finished", actor) == "answered"
    actor.parse_status = "malformed"
    assert _reviewer_verb("multi_model_review", "finished", actor) == "returned an unreadable verdict"


@pytest.mark.parametrize("kind", ["session", "native"])
def test_retrieving_answer_progress_precedes_verdict_extraction(monkeypatch, kind):
    from ouroboros import review_execution, review_native_episode

    module = review_execution if kind == "session" else review_native_episode
    cls = module.AgentSessionReviewExecutor if kind == "session" else module.NativeToolRoundReviewExecutor
    observed = []
    def canonicalize(text, **_kwargs):
        assert observed == ["received"]
        observed.append("read")
        return text, "strict", {}
    monkeypatch.setattr(module, "canonicalize_session_verdict", canonicalize)
    executor = SimpleNamespace(_raw_transcript="[]\nNO_FINDINGS", _conformance_passed=False,
        answer_observer=lambda: observed.append("received"), _output_contract=lambda: "array",
        llm=object(), assignment=SimpleNamespace(request=ReviewRequest(surface="multi_model_review", goal="g"),
                                                 slot=ReviewSlot("s", "m")),
        _session_usage={}, _episode_usage={}, _deltas=[], _episode_deltas=[], _run_id="run")
    result = cls._verdict_result(executor)
    assert result.raw_text == "[]\nNO_FINDINGS" and observed == ["received", "read"]


def test_gate_keeps_the_canonical_additional_seat_assignment():
    from ouroboros.review_seat_outcome import finalize_seat_outcome
    from ouroboros.tools.review import _collect_review_findings

    actor = ReviewActorRecord("adviser", "m", "ok", raw_text="[]\nNO_FINDINGS", operation_id="op")
    request = ReviewRequest(surface="multi_model_review", goal="g", policy={"parts": ["result"], "additional": True})
    finalize_seat_outcome(request, ReviewSlot("adviser", "m"), actor)
    result = _collect_review_findings(SimpleNamespace(), [{**asdict(actor), "text": actor.raw_text}],
                                     {"slot_ids": ["adviser"], "parts": [["coupling"]], "additional": [False]})
    [record] = result[3]
    assert record["additional"] is True and record["parts"] == ["result"]
    assert record["answers"] == actor.outcome["answers"]


def test_no_detach_waits_for_settlement_and_outer_exception_is_durable_before_ready(tmp_path, monkeypatch):
    from ouroboros.review_seat_outcome import publish_seat_execution

    entered, release, published = threading.Event(), threading.Event(), threading.Event()
    observed = []
    def fail(self, request, slot, **kw):
        entered.set()
        assert release.wait(10)
        raise ValueError("outer worker failed before executor preparation")
    monkeypatch.setattr(ReviewCoordinator, "_run_slot", fail)
    def publish(request, slot, actor, ctx, **kw):
        entry = next((e for e in review_custody._ACTIVE.values() if e.operation_id == actor.operation_id), None)
        observed.append(entry is None or entry.event.is_set())
        assert read_call_payload(tmp_path, **actor.outcome["source_ref"])[0]["producer_complete"]
        published.set()
        return publish_seat_execution(request, slot, actor, ctx, **kw)
    monkeypatch.setattr("ouroboros.review_seat_outcome.publish_seat_execution", publish)
    request = ReviewRequest(surface="multi_model_review", task_id="outer", goal="g", policy={"parts": ["result"]})
    coordinator = ReviewCoordinator(llm=object(), drive_root=tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(coordinator.run, request, [ReviewSlot("seat", "m")])
            assert entered.wait(10)
            assert not future.done(), "ordinary whole-wave waiting stays intact"
            release.set()
            actor = future.result(timeout=10).actors[0]
        assert published.is_set() and observed == [False, True]
        assert actor["outcome"]["status"] == "error" and actor["response_ref"]
    finally:
        release.set()


def test_failed_durable_publication_never_marks_the_entry_ready(tmp_path):
    import queue
    from ouroboros.review_wave import ReviewWave

    wave = ReviewWave(1)
    entry = review_custody.ActiveReviewAttempt("write-failed", "write-failed")
    actor = ReviewActorRecord("seat", "m", "ok", raw_text="[]\nNO_FINDINGS")
    def refuse(*_args):
        raise OSError("disk unavailable")
    with pytest.raises(OSError, match="disk unavailable"):
        with wave.worker(entry):
            review_custody._settle_review_attempt(entry, ReviewSlot("seat", "m"), actor,
                usage_ctx=None, request=ReviewRequest(surface="multi_model_review", goal="g"), task_id="t",
                result_queue=queue.Queue(), settle_actor=refuse)
    assert not entry.event.is_set() and entry.actor is actor and wave.running == 1
