"""Physical failure, exact price refinement and real writer concurrency consumers."""
import json
import subprocess
import sys

import pytest

from ouroboros import usage_accounting as ua, usage_store, usage_journal
from ouroboros.observability import read_blob_ref
from ouroboros.openrouter_cost import apply_retained_receipt, binding_for_target, retain_generation_receipt


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    return root


def request(root, **kwargs):
    return ua.AttemptRequest(model="synthetic-fixture", provider="openrouter", reservation_usd=2,
        drive_root=root, task_id="child", root_task_id="parent", provider_receipt_binding=binding_for_target({
            "provider": "openrouter", "base_url": "https://openrouter.invalid/api/v1", "api_key": "fake-key"}), **kwargs)


def row(root, attempt_id):
    with usage_store.hold(root) as txn:
        return txn.attempt(attempt_id)


def receipt(root, attempt_id, cost=.25):
    current = row(root, attempt_id)
    return retain_generation_receipt(root, current, {"data": {
        "id": current["provider_receipt_binding"]["generation_id"], "total_cost": cost, "usage": 0.0}})


class BodyFailure(RuntimeError):
    status_code = 400

    def __init__(self, body):
        super().__init__("synthetic physical failure")
        self.body = body


def test_failure_then_success_then_abandon_then_receipt_keeps_exact_attempt(root):
    failure = BodyFailure({"id": "gen-A", "error": {"message": "original private failure"}})
    with ua.capture_attempt_ids() as attempts:
        with pytest.raises(BodyFailure):
            ua.execute_physical_attempt(request(root), lambda: (_ for _ in ()).throw(failure))
        before_retry = row(root, attempts[0])
        assert before_retry["physical_failure"]["stage"] == "raised_exception"
        ua.execute_physical_attempt(request(root), lambda: {"id": "gen-B", "usage": {"cost": .8}})
    first, second = attempts
    reservation = ua.AttemptReservation(first, root, "synthetic-fixture", "openrouter", 2)
    ua.terminalize_abandoned_attempt(reservation, reason="owner_task_terminal")
    abandoned = row(root, first)
    assert abandoned["reason"] == "owner_task_terminal"
    assert "BodyFailure" in abandoned["unresolved_reason"]
    assert abandoned["physical_failure"] == before_retry["physical_failure"]
    evidence = read_blob_ref(root, abandoned["physical_failure"]["evidence_ref"])
    assert evidence["body"]["error"]["message"] == "original private failure"
    assert apply_retained_receipt(root, first, receipt(root, second))["status"] == "ineligible"
    assert apply_retained_receipt(root, first, receipt(root, first))["status"] == "applied"
    assert row(root, first)["physical_failure"] == before_retry["physical_failure"]
    assert row(root, second)["cost_usd"] == .8
    summary = ua.usage_breakdown(root, root_task_id="parent")
    assert summary["confirmed_usd"] == 1.05
    assert summary["physical_calls"] == 2


@pytest.mark.parametrize("body_error", [False, True])
def test_actual_retry_ladder_retains_first_failure_before_second_send(root, monkeypatch, body_error):
    from ouroboros.llm import LLMClient

    class Response:
        def __init__(self, payload):
            self.payload = payload
        def model_dump(self):
            return self.payload

    monkeypatch.setattr(ua, "_reservation_cost", lambda _: 2)
    client = LLMClient(api_key="fake-key")
    target = {"provider": "openrouter", "usage_model": "synthetic-fixture", "resolved_model": "synthetic-fixture",
              "supports_openrouter_extensions": True, "base_url": "https://openrouter.invalid/api/v1", "api_key": "fake-key"}
    body = {"id": "gen-retry-A", "error": {"code": 500 if body_error else 400, "message": "original rejection"}}
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if body_error:
                return Response(body)
            raise BodyFailure(body)
        first = row(root, attempts[0])
        assert read_blob_ref(root, first["physical_failure"]["evidence_ref"])["body"] == body
        assert first["provider_receipt_binding"]["generation_id"] == "gen-retry-A"
        return Response({"id": "gen-retry-B", "choices": [{"message": {"content": "answer"}}], "usage": {"cost": .4}})

    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="retry-task")), ua.capture_attempt_ids() as attempts:
        response = client._create_chat_completion_with_retries(create, {
            "model": "synthetic-fixture", "max_tokens": 16,
            "messages": [{"role": "assistant", "content": "previous", "reasoning": "previous reasoning",
                          "reasoning_details": [{"type": "reasoning.encrypted", "data": "synthetic-seal"}]},
                         {"role": "user", "content": "continue"}],
        }, target)
    assert len(calls) == len(attempts) == 2
    assert response.model_dump()["usage"]["cost"] == .4
    assert row(root, attempts[0]).get("cost_usd") is None
    assert row(root, attempts[1])["cost_usd"] == .4


@pytest.mark.parametrize("usage,expected,final", [
    ({"cost": 1.25}, 1.25, True), ({"cost": 0}, 0, True), ({}, None, False),
    ({"cost": "nan"}, None, False), ({"cost": True}, None, False),
    ({"cost": 1.25, "prompt_tokens": "broken", "completion_tokens": "inf"}, 1.25, True),
    ({"cost": 1.25, "cache_creation": {"ephemeral_5m_input_tokens": "broken"},
      "cache_write_tokens_by_ttl": "broken"}, 1.25, True),
    ({"prompt_tokens": 0, "completion_tokens": 0}, 0, True),
    ({"prompt_tokens": 0, "completion_tokens": 0, "cost": -1}, None, False),
])
def test_returned_body_failure_preserves_explicit_money(root, usage, expected, final):
    body = {"id": "gen-body", "error": {"code": 500, "message": "private failure"}, "usage": usage}
    assert ua.execute_physical_attempt(request(root), lambda: body) is body
    current = row(root, ua.last_physical_attempt_capture().attempt_id)
    assert current["cost_usd"] == expected and current["cost_final"] is final
    assert current["physical_failure"]["stage"] == "response_body_error"
    assert read_blob_ref(root, current["physical_failure"]["evidence_ref"])["body"] == body


@pytest.mark.parametrize("cost", [0, 1.25])
@pytest.mark.parametrize("via_stream", [False, True])
def test_received_cost_precedes_existing_free_exception_classification(root, cost, via_stream):
    error = BodyFailure({"id": "gen-free", "error": {"message": "No endpoints found"}, "usage": {"cost": cost}})
    error.status_code = 404
    error.args = ("No endpoints found",)
    if via_stream:
        error.body.pop("usage")
        error.stream_usage = {"cost": cost}
    with pytest.raises(BodyFailure):
        ua.execute_physical_attempt(request(root), lambda: (_ for _ in ()).throw(error))
    current = row(root, ua.last_physical_attempt_capture().attempt_id)
    assert current["cost_usd"] == cost and current["cost_final"] is True


def test_price_only_success_refinement_preserves_all_nonmoney_facts_and_ownership(root):
    known = {"prompt_tokens": 17, "completion_tokens": 4, "cached_tokens": 8,
             "cache_write_tokens": 2, "prompt_cache_ttl": "1h", "effort": {"reported": "high"},
             "processing": {"observed": "standard"}, "service_tier": "default", "cost_evidence": {"original": True}}
    ua.execute_physical_attempt(request(root), lambda: {"id": "gen-success"},
                                extractor=lambda _: (known, 1.1, False))
    attempt_id = ua.last_physical_attempt_capture().attempt_id
    before = row(root, attempt_id)
    with usage_store.hold(root) as txn:
        assert txn.open_attempts(root_task_id="parent") == []
    retained = receipt(root, attempt_id)
    assert apply_retained_receipt(root, attempt_id, retained, expected_revision=before["revision"] - 1)["status"] == "stale"
    assert apply_retained_receipt(root, attempt_id, retained, expected_revision=before["revision"])["status"] == "applied"
    after = row(root, attempt_id)
    for key, value in before.items():
        if key not in {"seq", "ts", "revision", "cost_usd", "cost_final", "settle_reason"}:
            assert after[key] == value, key
    assert apply_retained_receipt(root, attempt_id, retained, expected_revision=before["revision"])["status"] == "duplicate"
    assert apply_retained_receipt(root, attempt_id, receipt(root, attempt_id, .9))["status"] == "conflict"
    assert row(root, attempt_id) == after


def test_missing_or_mismatched_source_and_generation_conflict_cannot_settle(root):
    def send():
        ua.bind_provider_generation("gen-first")
        first = row(root, ua.last_physical_attempt_capture().attempt_id)
        ua.bind_provider_generation("gen-first")
        assert row(root, first["attempt_id"])["revision"] == first["revision"]
        ua.bind_provider_generation("gen-second")
        return {"usage": {}}
    ua.execute_physical_attempt(request(root), send)
    attempt_id = ua.last_physical_attempt_capture().attempt_id
    assert row(root, attempt_id)["provider_receipt_binding"]["conflict"]
    assert apply_retained_receipt(root, attempt_id, {"cost_usd": 0})["status"] == "ineligible"
    assert receipt(root, attempt_id)["status"] == "binding_unavailable"


def test_receipt_credential_mismatch_and_missing_cas_are_noops(root):
    ua.execute_physical_attempt(request(root), lambda: {"id": "gen-identity"})
    attempt_id = ua.last_physical_attempt_capture().attempt_id
    held = receipt(root, attempt_id)
    assert apply_retained_receipt(root, attempt_id, {**held, "credential_sha256": "other"})["status"] == "ineligible"
    assert apply_retained_receipt(root, attempt_id, {**held, "evidence_ref": {}})["status"] == "ineligible"
    assert row(root, attempt_id)["cost_final"] is False


def test_retention_failure_keeps_body_outcome_and_explicit_gap(root, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("synthetic storage outage")
    monkeypatch.setattr("ouroboros.observability.write_blob", fail)
    body = {"error": {"code": 500}, "usage": {"cost": .3}}
    assert ua.execute_physical_attempt(request(root), lambda: body) is body
    current = row(root, ua.last_physical_attempt_capture().attempt_id)
    assert current["cost_usd"] == .3
    assert current["physical_failure"]["retention_gap"] == "OSError"


def test_failure_body_is_shared_by_evidence_and_settlement(root):
    """Large SDK failure bodies need one decode per accounting/capture boundary."""
    body = {"id": "gen-decoded", "error": {"code": 500}, "usage": {"cost": .3}}
    reads = []

    class Response:
        status_code = 500

        def json(self):
            reads.append(True)
            return body

    error = RuntimeError("provider failure")
    error.response = Response()
    with pytest.raises(RuntimeError) as caught:
        ua.execute_physical_attempt(request(root), lambda: (_ for _ in ()).throw(error))
    assert caught.value is error
    current = row(root, ua.last_physical_attempt_capture().attempt_id)
    assert current["cost_usd"] == .3 and current["cost_final"] is True
    assert read_blob_ref(root, current["physical_failure"]["evidence_ref"])["body"] == body
    assert len(reads) == 2  # Evidence/settlement share one; exception capture reads one.


def test_sdk_body_is_shared_without_losing_response_only_price(root):
    reads = []

    class Response:
        total_cost_usd = .4  # SDK-only attribute, outside the serialized body.

        def model_dump(self):
            reads.append(True)
            return {"id": "gen-sdk", "choices": [{"message": {"content": "paid answer"}}]}

    response = Response()
    assert ua.execute_physical_attempt(request(root), lambda: response) is response
    current = row(root, ua.last_physical_attempt_capture().attempt_id)
    assert current["cost_usd"] == .4 and current["cost_final"] is True
    assert current["provider_receipt_binding"]["generation_id"] == "gen-sdk"
    assert len(reads) == 1


def test_evidence_cas_preserves_first_failure_and_money(root):
    attempt = ua.reserve_attempt(request(root))
    ua.mark_dispatched(attempt)
    before = ua.usage_projection(root)
    with usage_store.hold(root) as txn:
        revision = txn.attempt(attempt.attempt_id)["revision"]
        assert txn.record_evidence(attempt.attempt_id, {"physical_failure": {"exception_type": "first"}}, expected_revision=revision)
        assert not txn.record_evidence(attempt.attempt_id, {"physical_failure": {"exception_type": "stale"}}, expected_revision=revision)
        assert txn.record_evidence(attempt.attempt_id, {"physical_failure": {"exception_type": "replacement"}}, expected_revision=revision + 1)
    assert row(root, attempt.attempt_id)["physical_failure"]["exception_type"] == "first"
    assert ua.usage_projection(root) == before


def test_shared_writer_accepts_provider_neutral_exact_attempt_fact(root):
    # A different provider's proof producer uses its own opaque identity. The
    # common writer must neither require nor parse OpenRouter's private fields.
    binding = {"receipt_namespace": "fixture-provider", "opaque_request": "request-one"}
    ua.execute_physical_attempt(
        ua.AttemptRequest(model="fixture-model", provider="fixture-provider", reservation_usd=2,
                          drive_root=root, provider_receipt_binding=binding),
        lambda: {"usage": {}}, extractor=lambda _: ({"prompt_tokens": 7}, 1.5, False))
    attempt_id = ua.last_physical_attempt_capture().attempt_id
    before = row(root, attempt_id)
    fact = {"attempt_id": attempt_id, "provider": "fixture-provider", "binding": binding,
            "cost_usd": 0.25, "evidence_ref": {"sha256": "fixture-provider-proved-source"}}
    assert ua.apply_provider_price_receipt(root, attempt_id, {
        **fact, "binding": {**binding, "opaque_request": "different"}})["status"] == "ineligible"
    assert row(root, attempt_id) == before
    applied = ua.apply_provider_price_receipt(root, attempt_id, fact, expected_revision=before["revision"])
    assert applied["status"] == "applied"
    assert applied["row"]["cost_usd"] == 0.25 and applied["row"]["prompt_tokens"] == 7
    assert ua.apply_provider_price_receipt(root, attempt_id, fact)["status"] == "duplicate"
    assert ua.apply_provider_price_receipt(root, attempt_id, {**fact, "cost_usd": 0.5})["status"] == "conflict"


def test_journal_accepts_nonfinal_price_refinement_but_not_release(root):
    ua.execute_physical_attempt(request(root), lambda: {"id": "gen-journal", "usage": {"prompt_tokens": 12}})
    attempt_id = ua.last_physical_attempt_capture().attempt_id
    before = row(root, attempt_id)
    held = receipt(root, attempt_id)
    applied = apply_retained_receipt(root, attempt_id, held)["row"]
    # Validate the actual transition, independently of export's minimal chain.
    base = {key: value for key, value in before.items() if key not in {"seq", "revision"}}
    rows = [{**base, "state": "reserved", "seq": 1}, {**base, "state": "dispatched", "seq": 2},
            {**base, "seq": 3}, {**applied, "seq": 4}]
    usage_journal._validate_records(rows)
    with pytest.raises(ua.UsageLedgerCorrupt):
        usage_journal._validate_records(rows[:3] + [{**base, "state": "released", "seq": 4,
                                                    "reason": "before_dispatch_failed:forged"}])
    assert usage_store.export_journal(root)["attempts"] == 1
    assert row(root, attempt_id)["provider_price_receipt"]["evidence_ref"] == held["evidence_ref"]
    assert apply_retained_receipt(root, attempt_id, held)["status"] == "duplicate"


@pytest.mark.serial
def test_two_process_receipt_application_changes_money_once(root):
    ua.execute_physical_attempt(request(root), lambda: {"id": "gen-race"})
    attempt_id = ua.last_physical_attempt_capture().attempt_id
    held = receipt(root, attempt_id)
    code = """import json, sys
from ouroboros.openrouter_cost import apply_retained_receipt
print(json.dumps(apply_retained_receipt(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))['status']))
"""
    command = [sys.executable, "-c", code, str(root), attempt_id, json.dumps(held)]
    # Bounded children inside the safe_test-owned environment; no provider HTTP.
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        results = [process.communicate(timeout=30) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    assert [process.returncode for process in processes] == [0, 0], results
    assert sorted(json.loads(out) for out, _ in results) == ["applied", "duplicate"]
    summary = ua.usage_breakdown(root)
    assert summary["confirmed_usd"] == .25 and summary["physical_calls"] == 1
