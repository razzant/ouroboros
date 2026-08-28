from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import usage_accounting as ua


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    return root


def _request(data_root, **overrides):
    values = {
        "model": "openai/gpt-5.2",
        "provider": "openai",
        "reservation_usd": 1.0,
        "drive_root": data_root,
        "task_id": "child",
        "root_task_id": "root",
        "source": "test",
    }
    values.update(overrides)
    return ua.AttemptRequest(**values)


def _ledger(data_root):
    path = data_root / ua.LEDGER_REL
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_attempt_lifecycle_and_root_projection(data_root):
    reservation = ua.reserve_attempt(_request(data_root, root_limit_usd=2.0))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(
        reservation,
        {"prompt_tokens": 10, "completion_tokens": 5},
        cost_usd=0.25,
        cost_final=True,
    )

    projection = ua.usage_projection(data_root)
    assert projection["settled_usd"] == 0.25
    assert projection["confirmed_usd"] == 0.25
    assert projection["cost_final"] is True
    assert projection["by_root"]["root"]["settled_usd"] == 0.25
    assert projection["by_root"]["root"]["limit_usd"] == 2.0
    rows = _ledger(data_root)
    assert [row["state"] for row in rows] == ["reserved", "dispatched", "settled"]
    assert [row["seq"] for row in rows] == [1, 2, 3]


def test_projection_uses_explicit_runtime_limit_over_environment(data_root):
    assert ua.usage_projection(data_root, global_limit_usd=7.5)["limit_usd"] == 7.5


def test_breakdown_uses_final_rows_and_keeps_unattributed_explicit(data_root):
    reservation = ua.reserve_attempt(_request(
        data_root, category="review", prompt_tokens_estimate=10,
    ))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(
        reservation,
        {"prompt_tokens": 10, "completion_tokens": 3, "cached_tokens": 2},
        cost_usd=0.2,
        cost_final=True,
    )
    external_id = ua.record_unmetered_external_dispatch(
        "skill-call-1",
        drive_root=data_root,
        provider="external-skill",
        category="skill",
        prompt_tokens=4,
        completion_tokens=1,
    )

    breakdown = ua.usage_breakdown(data_root)
    assert breakdown["physical_calls"] == 2
    assert breakdown["prompt_tokens"] == 14
    assert breakdown["completion_tokens"] == 4
    assert breakdown["confirmed_usd"] == 0.2
    assert breakdown["unknown_unmetered"] == 1
    assert breakdown["by_model"]["openai/gpt-5.2"]["physical_calls"] == 1
    assert breakdown["by_provider"]["external-skill"]["unknown_unmetered"] == 1
    assert breakdown["by_category"]["skill"]["physical_calls"] == 1
    assert breakdown["unattributed"]["model"]["physical_calls"] == 1
    assert external_id.startswith("external-")


def test_external_unmetered_dispatch_is_idempotent_and_conflict_checked(data_root):
    first = ua.record_unmetered_external_dispatch(
        "stable-id", drive_root=data_root, provider="skill", task_id="t",
    )
    second = ua.record_unmetered_external_dispatch(
        "stable-id", drive_root=data_root, provider="skill", task_id="t",
    )
    assert first == second
    assert len(_ledger(data_root)) == 1
    with pytest.raises(ua.UsageAccountingError, match="conflicting"):
        ua.record_unmetered_external_dispatch(
            "stable-id", drive_root=data_root, provider="different", task_id="t",
        )


def test_provider_failure_remains_unresolved(data_root):
    sends = 0

    def fail():
        nonlocal sends
        sends += 1
        raise TimeoutError("transport timeout")

    with pytest.raises(TimeoutError) as exc_info:
        ua.execute_physical_attempt(_request(data_root), fail)
    assert sends == 1
    capture = ua.physical_attempt_capture_from_exception(exc_info.value)
    assert capture is not None
    assert capture.state == "unresolved"
    assert capture.provider_response_observed is False
    assert capture.provider_status_code is None
    assert ua.physical_provider_outcome_unknown(exc_info.value) is True
    projection = ua.usage_projection(data_root)
    assert projection["unresolved_upper_bound_usd"] == 1.0
    assert _ledger(data_root)[-1]["state"] == "unresolved"


def test_typed_http_failure_is_not_an_unknown_provider_outcome(data_root):
    from ouroboros.loop_llm_call import classify_llm_exception

    class Response:
        status_code = 503

        @staticmethod
        def json():
            return {"error": {"code": "overloaded"}}

    class ProviderError(Exception):
        def __init__(self):
            super().__init__("provider unavailable")
            self.response = Response()

    error = ProviderError()
    with pytest.raises(ProviderError) as exc_info:
        ua.execute_physical_attempt(
            _request(data_root),
            lambda: (_ for _ in ()).throw(error),
        )

    capture = ua.physical_attempt_capture_from_exception(exc_info.value)
    assert capture is not None
    assert capture.state == "unresolved"  # conservative cost custody
    assert capture.provider_response_observed is True
    assert capture.provider_status_code == 503
    assert capture.provider_code == "overloaded"
    assert ua.physical_provider_outcome_unknown(exc_info.value) is False
    classification = classify_llm_exception(exc_info.value)
    assert classification.kind == "provider_transient"
    assert classification.retry_same_request is True


@pytest.mark.parametrize(("attribute", "value", "expected"), [
    ("status", 408, 408),
    ("code", 503, 503),
    ("code", "503", 503),
])
def test_typed_http_status_aliases_preserve_safe_retry(
    data_root, attribute, value, expected,
):
    from ouroboros.loop_llm_call import classify_llm_exception

    class ProviderError(Exception):
        pass

    error = ProviderError("typed provider response")
    setattr(error, attribute, value)
    with pytest.raises(ProviderError) as exc_info:
        ua.execute_physical_attempt(
            _request(data_root),
            lambda: (_ for _ in ()).throw(error),
        )

    capture = ua.physical_attempt_capture_from_exception(exc_info.value)
    assert capture.provider_response_observed is True
    assert capture.provider_status_code == expected
    assert capture.logical_call_attempt_ids == (capture.attempt_id,)
    classification = classify_llm_exception(exc_info.value)
    assert classification.kind == "provider_transient"
    assert classification.retry_same_request is True


@pytest.mark.parametrize("provider_code", ["insufficient_quota", "invalid_api_key"])
def test_unknown_outcome_outranks_unobserved_provider_code(provider_code, data_root):
    from ouroboros.loop_llm_call import classify_llm_exception

    class ProviderError(ConnectionError):
        code = provider_code

    with pytest.raises(ProviderError) as exc_info:
        ua.execute_physical_attempt(
            _request(data_root),
            lambda: (_ for _ in ()).throw(ProviderError("connection lost after send")),
        )

    assert ua.physical_provider_outcome_unknown(exc_info.value) is True
    classification = classify_llm_exception(exc_info.value)
    assert classification.kind == "provider_outcome_unknown"
    assert classification.retry_same_request is False


def test_async_provider_failure_carries_unknown_outcome_capture(data_root):
    async def fail():
        raise ConnectionError("connection dropped")

    with pytest.raises(ConnectionError) as exc_info:
        asyncio.run(ua.execute_physical_attempt_async(_request(data_root), fail))

    capture = ua.physical_attempt_capture_from_exception(exc_info.value)
    assert capture is not None
    assert capture.state == "unresolved"
    assert capture.provider_response_observed is False
    assert ua.physical_provider_outcome_unknown(exc_info.value) is True


def test_async_typed_status_alias_preserves_safe_retry(data_root):
    from ouroboros.loop_llm_call import classify_llm_exception

    class ProviderError(Exception):
        status = "503"

    async def fail():
        raise ProviderError("typed async provider response")

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(ua.execute_physical_attempt_async(_request(data_root), fail))

    capture = ua.physical_attempt_capture_from_exception(exc_info.value)
    assert capture.provider_response_observed is True
    assert capture.provider_status_code == 503
    classification = classify_llm_exception(exc_info.value)
    assert classification.kind == "provider_transient"
    assert classification.retry_same_request is True


def test_lock_failure_is_fail_closed_before_send(data_root, monkeypatch):
    import ouroboros.platform_layer as platform

    monkeypatch.setattr(platform, "acquire_exclusive_file_lock", lambda *args, **kwargs: None)
    sends = 0

    def send():
        nonlocal sends
        sends += 1

    with pytest.raises(ua.UsageAccountingError):
        ua.execute_physical_attempt(_request(data_root), send)
    assert sends == 0


def test_relative_or_mock_like_drive_root_never_writes_cwd(data_root):
    with pytest.raises(ua.UsageAccountingError, match="must be absolute"):
        ua.reserve_attempt(_request(data_root, drive_root="."))


def test_paid_response_survives_settlement_storage_failure(data_root, monkeypatch):
    def broken_settle(*args, **kwargs):
        raise OSError("disk full after response")

    monkeypatch.setattr(ua, "settle_attempt", broken_settle)
    response = {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    assert ua.execute_physical_attempt(_request(data_root), lambda: response) is response
    assert _ledger(data_root)[-1]["state"] == "unresolved"
    assert ua.usage_projection(data_root)["unresolved_upper_bound_usd"] == 1.0


def test_paid_response_survives_usage_extractor_failure(data_root):
    response = object()

    def broken_extractor(_response):
        raise ValueError("malformed provider usage")

    assert ua.execute_physical_attempt(
        _request(data_root), lambda: response, extractor=broken_extractor,
    ) is response
    assert _ledger(data_root)[-1]["state"] == "unresolved"


def test_async_paid_response_survives_usage_extractor_failure(data_root):
    response = object()

    async def send():
        return response

    def broken_extractor(_response):
        raise ValueError("malformed provider usage")

    result = asyncio.run(ua.execute_physical_attempt_async(
        _request(data_root), send, extractor=broken_extractor,
    ))
    assert result is response
    assert _ledger(data_root)[-1]["state"] == "unresolved"


def test_provider_reported_zero_cost_is_final_not_missing(data_root):
    response = {
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0},
    }
    ua.execute_physical_attempt(_request(data_root), lambda: response)
    projection = ua.usage_projection(data_root)
    assert projection["confirmed_usd"] == 0
    assert projection["unknown_unmetered"] == 0
    assert projection["cost_final"] is True
    assert _ledger(data_root)[-1]["cost_usd"] == 0


def test_torn_final_row_is_quarantined_but_midstream_corruption_fails(data_root):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(reservation)
    ledger = data_root / ua.LEDGER_REL
    with ledger.open("ab") as handle:
        handle.write(b'{"seq":')

    projection = ua.usage_projection(data_root)
    assert projection["attempt_counts"]["released"] == 1
    assert projection["integrity_degraded"] is True
    assert projection["cost_final"] is False
    breakdown = ua.usage_breakdown(data_root)
    assert breakdown["integrity_degraded"] is True
    assert breakdown["cost_final"] is False
    assert (data_root / ua.QUARANTINE_REL).is_file()
    repaired = ledger.read_bytes()
    assert b'{"seq":' not in repaired


@pytest.mark.parametrize(
    "field,value",
    (("seq", "not-a-number"), ("prompt_tokens", "not-a-number")),
)
def test_structurally_invalid_numeric_tail_is_quarantined(data_root, field, value):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(reservation)
    ledger = data_root / ua.LEDGER_REL
    row = {
        "seq": 3,
        "ts": "2026-01-01T00:00:00Z",
        "attempt_id": "tail-attempt",
        "kind": "attempt",
        "state": "reserved",
        "reservation_upper_bound_usd": 1.0,
        field: value,
    }
    with ledger.open("a") as handle:
        handle.write(json.dumps(row) + "\n")

    projection = ua.usage_projection(data_root)
    assert projection["integrity_degraded"] is True
    assert projection["cost_final"] is False
    assert projection["attempt_counts"]["released"] == 1


def test_quarantined_dispatch_tail_makes_replay_evidence_degraded(data_root):
    reservation = ua.reserve_attempt(_request(data_root, task_id="replay-risk"))
    ledger = data_root / ua.LEDGER_REL
    corrupt_dispatch = {
        **_ledger(data_root)[-1],
        "seq": 2,
        "state": "dispatched",
        "prompt_tokens": "torn",
    }
    with ledger.open("a") as handle:
        handle.write(json.dumps(corrupt_dispatch) + "\n")

    evidence = ua.usage_breakdown(data_root, task_id="replay-risk")
    assert evidence["physical_calls"] == 0
    assert evidence["integrity_degraded"] is True
    ua.release_attempt(reservation)

    lines = ledger.read_text().splitlines()
    ledger.write_text(lines[0] + "\nnot-json\n" + lines[1] + "\n")
    with pytest.raises(ua.UsageLedgerCorrupt):
        ua.usage_projection(data_root)


def test_structurally_invalid_final_row_is_quarantined_but_midstream_is_fatal(data_root):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(reservation)
    ledger = data_root / ua.LEDGER_REL
    bad = {
        "seq": 999,
        "kind": "attempt",
        "attempt_id": "bad-tail",
        "state": "dispatched",
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(bad) + "\n")

    assert ua.usage_projection(data_root)["attempt_counts"] == {"released": 1}
    assert all(row.get("attempt_id") != "bad-tail" for row in _ledger(data_root))
    assert (data_root / ua.QUARANTINE_REL).is_file()

    lines = ledger.read_text().splitlines()
    bad["seq"] = 2
    ledger.write_text(lines[0] + "\n" + json.dumps(bad) + "\n" + lines[1] + "\n")
    with pytest.raises(ua.UsageLedgerCorrupt):
        ua.usage_projection(data_root)


def test_concurrent_writers_keep_monotonic_sequence(data_root):
    def one(index):
        reservation = ua.reserve_attempt(_request(data_root, task_id=f"t{index}"))
        ua.mark_dispatched(reservation)
        ua.settle_attempt(reservation, cost_usd=0.01, cost_final=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, range(16)))
    rows = _ledger(data_root)
    assert [row["seq"] for row in rows] == list(range(1, len(rows) + 1))
    assert ua.usage_projection(data_root)["settled_usd"] == 0.16


def test_known_reservation_is_checked_before_dispatch(data_root):
    first = ua.reserve_attempt(_request(data_root, reservation_usd=0.6, global_limit_usd=1.0))
    with pytest.raises(ua.BudgetExceeded):
        ua.reserve_attempt(_request(data_root, reservation_usd=0.5, global_limit_usd=1.0))
    assert [row["state"] for row in _ledger(data_root)] == ["reserved"]
    ua.release_attempt(first)


def test_live_openrouter_catalog_produces_known_reservation(data_root, monkeypatch):
    from ouroboros import pricing

    # Isolate every process-local catalog state carrier.  A prior failed fetch
    # may leave a short retry cooldown (or an in-progress marker), which must
    # not suppress this test's deterministic synthetic provider response.
    monkeypatch.setattr(pricing, "_cached_pricing", {})
    monkeypatch.setattr(pricing, "_pricing_fetched_at", {})
    monkeypatch.setattr(pricing, "_pricing_retry_after", {})
    monkeypatch.setattr(pricing, "_pricing_fetch_in_progress", set())
    monkeypatch.setattr(
        "ouroboros.llm.fetch_openrouter_pricing",
        lambda **kwargs: {"openai/gpt-new": (2.0, None, None, 8.0)},
    )
    reservation = ua.reserve_attempt(_request(
        data_root,
        model="openai/gpt-new",
        provider="openrouter",
        reservation_usd=None,
        prompt_tokens_estimate=1_000,
        max_completion_tokens=500,
    ))
    row = _ledger(data_root)[-1]
    # OpenAI-family reservations retain the tokenizer envelope: 1,100 input.
    assert row["reservation_upper_bound_usd"] == 0.0062
    assert row["reservation_basis"] == "linear_pricing"
    ua.release_attempt(reservation)


def test_explicit_reservation_is_not_inflated_by_tokenizer_margin(data_root):
    reservation = ua.reserve_attempt(_request(
        data_root,
        model="openai/gpt-5.5-pro",
        provider="openrouter",
        reservation_usd=2.5,
        prompt_tokens_estimate=460_332,
        max_completion_tokens=65_536,
    ))
    assert _ledger(data_root)[-1]["reservation_upper_bound_usd"] == 2.5
    ua.release_attempt(reservation)

    opaque = ua.reserve_attempt(_request(
        data_root,
        model="openai/gpt-5.5-pro",
        provider="openrouter",
        reservation_usd=None,
        max_budget_usd=3.25,
        prompt_tokens_estimate=460_332,
        max_completion_tokens=65_536,
    ))
    assert _ledger(data_root)[-1]["reservation_upper_bound_usd"] == 3.25
    ua.release_attempt(opaque)


def test_known_hold_does_not_override_provider_reported_settlement(data_root):
    reservation = ua.reserve_attempt(_request(
        data_root,
        model="openai/gpt-new",
        provider="openrouter",
        reservation_usd=8.01278,
    ))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(
        reservation,
        {"prompt_tokens": 477_909, "completion_tokens": 7_585},
        cost_usd=5.120415,
        cost_final=True,
    )
    row = _ledger(data_root)[-1]
    assert row["cost_usd"] == 5.120415
    assert row["reservation_upper_bound_usd"] == 8.01278


def test_scope_runtime_limit_is_enforced_without_provider_retry(data_root):
    sends = 0

    def send():
        nonlocal sends
        sends += 1

    scope = ua.UsageScope(drive_root=data_root, global_limit_usd=0.5)
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded):
        ua.execute_physical_attempt(_request(data_root, reservation_usd=0.6), send)
    assert sends == 0


def test_llm_retry_machine_does_not_classify_budget_rail_as_provider_failure(data_root, monkeypatch):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="unused")
    sends = 0

    def create(**kwargs):
        nonlocal sends
        sends += 1

    def forbidden(*args, **kwargs):
        raise AssertionError("local accounting rail reached provider retry logic")

    monkeypatch.setattr(client, "_retry_without_optional_sampling", forbidden)
    monkeypatch.setattr(client, "_openrouter_signature_retry_kwargs", forbidden)
    monkeypatch.setattr(client, "_reroute_kwargs_for_body_error", forbidden)
    target = {
        "provider": "openai",
        "usage_model": "openai/gpt-5.2",
        "resolved_model": "gpt-5.2",
    }
    with ua.usage_scope(ua.UsageScope(drive_root=data_root, global_limit_usd=0)):
        with pytest.raises(ua.BudgetExceeded):
            client._create_chat_completion_with_retries(
                create,
                {"model": "gpt-5.2", "messages": [{"role": "user", "content": "x"}], "max_tokens": 10},
                target,
            )
    assert sends == 0


def test_web_search_does_not_cascade_on_accounting_rail(data_root, monkeypatch):
    from ouroboros.tools import search

    class Ctx:
        task_id = "t"
        task_metadata = {"budget_drive_root": str(data_root)}

    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openrouter")
    monkeypatch.setattr(
        search,
        "_web_search_openrouter",
        lambda *args, **kwargs: (_ for _ in ()).throw(ua.BudgetExceeded("rail")),
    )
    with pytest.raises(ua.BudgetExceeded):
        search._web_search(Ctx(), "query")


def test_unknown_pricing_is_not_reported_as_zero(data_root, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "0")
    reservation = ua.reserve_attempt(
        _request(
            data_root,
            model="unknown/vendor-model",
            reservation_usd=None,
            max_completion_tokens=100,
        )
    )
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, {})
    projection = ua.usage_projection(data_root)
    assert projection["settled_usd"] == 0
    assert projection["unresolved_upper_bound_usd"] == 0
    assert projection["unknown_unmetered"] == 1
    assert projection["cost_final"] is False
    assert _ledger(data_root)[-1]["cost_usd"] is None
    assert _ledger(data_root)[-1]["cost_final"] is False


def test_legacy_metadata_gap_is_count_only_not_monetary_unknown():
    summary = ua._summary([{
        "kind": "legacy_metadata",
        "attempt_id": "legacy-gap",
        "state": "settled",
        "ambiguous_call_count": 7,
    }])
    assert summary["attempt_counts"]["metadata_only"] == 7
    assert summary["unknown_unmetered"] == 0
    assert summary["cost_final"] is True


def test_opaque_operation_without_max_budget_reserves_unknown(data_root, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "0")
    reservation = ua.reserve_attempt(
        _request(
            data_root,
            model="anthropic/claude-opus-4.8",
            reservation_usd=None,
            prompt_tokens_estimate=1000,
            force_unknown_reservation=True,
        )
    )
    row = _ledger(data_root)[-1]
    assert row["reservation_upper_bound_usd"] is None
    assert row["pricing_known"] is False
    assert row["reservation_basis"] == "opaque_unknown"
    projection = ua.usage_projection(data_root)
    assert projection["reserved_usd"] == 0
    assert projection["unknown_unmetered"] == 1
    ua.release_attempt(reservation)


def test_unknown_pricing_is_fail_open_under_finite_global_and_root_limits(data_root, monkeypatch):
    monkeypatch.setattr("ouroboros.llm.fetch_openrouter_pricing", lambda **kwargs: {})
    first = ua.reserve_attempt(_request(
        data_root, model="unknown/vendor-model", reservation_usd=None,
    ))
    first_row = _ledger(data_root)[-1]
    assert first_row["reservation_upper_bound_usd"] is None
    assert first_row["pricing_known"] is False
    ua.release_attempt(first)

    second = ua.reserve_attempt(_request(
        data_root,
        model="unknown/vendor-model",
        reservation_usd=None,
        global_limit_usd=float("inf"),
        root_limit_usd=2.0,
    ))
    assert _ledger(data_root)[-1]["reservation_upper_bound_usd"] is None
    ua.release_attempt(second)


def test_live_pricing_lookup_finishes_before_ledger_lock(data_root, monkeypatch):
    lock_active = False
    lookup_called = False
    original_locked = ua._locked

    @contextlib.contextmanager
    def tracked_lock(root):
        nonlocal lock_active
        with original_locked(root):
            lock_active = True
            try:
                yield
            finally:
                lock_active = False

    def pricing_lookup(*args, **kwargs):
        nonlocal lookup_called
        lookup_called = True
        assert lock_active is False
        return None

    monkeypatch.setattr(ua, "_locked", tracked_lock)
    monkeypatch.setattr(ua, "estimate_cost_optional", pricing_lookup)

    reservation = ua.reserve_attempt(_request(
        data_root,
        model="openai/gpt-future",
        provider="openrouter",
        reservation_usd=None,
    ))
    assert lookup_called is True
    assert reservation.reservation_upper_bound_usd is None
    ua.release_attempt(reservation)


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openrouter", "openai/gpt-brand-new"),
        ("openai", "openai::gpt-brand-new"),
        ("openai-compatible", "openai-compatible::vendor-model"),
    ],
)
def test_unknown_new_model_dispatches_when_catalog_is_unavailable(
    data_root, monkeypatch, provider, model,
):
    monkeypatch.setattr("ouroboros.llm.fetch_openrouter_pricing", lambda **kwargs: {})
    sends = 0

    def send():
        nonlocal sends
        sends += 1
        return {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    response = ua.execute_physical_attempt(
        _request(
            data_root,
            model=model,
            provider=provider,
            reservation_usd=None,
            global_limit_usd=10.0,
        ),
        send,
    )
    assert response["usage"]["prompt_tokens"] == 3
    assert sends == 1
    final = _ledger(data_root)[-1]
    assert final["state"] == "settled"
    assert final["cost_usd"] is None
    assert ua.usage_breakdown(data_root)["physical_calls"] == 1


def test_direct_chat_budget_exhaustion_requires_budget_change_before_retry():
    from ouroboros.agent import (
        _budget_exhausted_message,
        _budget_resume_policy,
        _queued_budget_exhausted_message,
    )

    text = _budget_exhausted_message().lower()
    assert "increase or reset" in text
    assert "starting a new run before changing" in text
    assert _budget_resume_policy(replay_safe=False, direct_chat=True) == (
        "increase_or_reset_budget_then_retry"
    )
    assert _budget_resume_policy(replay_safe=True, direct_chat=True) == (
        "increase_or_reset_budget_then_retry"
    )
    assert "cancel it or start a new run" in _queued_budget_exhausted_message().lower()
    assert _budget_resume_policy(replay_safe=False, direct_chat=False) == "cancel_or_new_run"


def test_direct_chat_loop_budget_exhaustion_never_suggests_new_run(data_root, monkeypatch):
    from types import SimpleNamespace

    from ouroboros.loop import _handle_budget_exceeded, _LoopExitContext

    monkeypatch.setattr(
        ua,
        "usage_breakdown",
        lambda *args, **kwargs: {"physical_calls": 1, "integrity_degraded": False},
    )
    exit_ctx = _LoopExitContext(
        tools=SimpleNamespace(_ctx=SimpleNamespace(
            budget_drive_root=data_root,
            drive_root=data_root,
            is_direct_chat=True,
        )),
        drive_root=data_root,
        task_id="direct-chat",
        event_queue=None,
        drive_logs=data_root / "logs",
        accumulated_usage={},
        llm_trace={},
    )

    text, usage, trace = _handle_budget_exceeded(
        ua.BudgetExceeded("known budget exhausted"),
        exit_ctx,
    )

    assert "increase or reset" in text.lower()
    assert "starting a new run before changing" in text.lower()
    assert usage["resource_limit"]["resume_policy"] == "increase_or_reset_budget_then_retry"
    assert trace["resource_limit"]["replay_safe"] is False


def test_zero_bound_cannot_dispatch_after_finite_limit_is_reached(data_root):
    first = ua.reserve_attempt(_request(data_root, reservation_usd=1.0, global_limit_usd=1.0))
    ua.mark_dispatched(first)
    ua.settle_attempt(first, {}, cost_usd=1.0, cost_final=True)

    with pytest.raises(ua.BudgetExceeded):
        ua.reserve_attempt(_request(
            data_root, task_id="next", reservation_usd=None,
            max_budget_usd=0.0, global_limit_usd=1.0,
        ))


def test_legacy_state_projection_cannot_regress_under_reordered_writers(
    data_root, monkeypatch,
):
    from supervisor import state

    state.init(data_root, total_budget_limit=0.0)
    first_started = threading.Event()
    release_first = threading.Event()
    calls = []

    def breakdown(_root):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(2.0)
            value = 1.0
        else:
            value = 2.0
        return {
            "accounted_usd": value, "physical_calls": int(value),
            "prompt_tokens": int(value), "completion_tokens": 0, "cached_tokens": 0,
            "settled_usd": value, "confirmed_usd": value, "estimated_usd": 0.0,
            "reserved_usd": 0.0, "unresolved_upper_bound_usd": 0.0,
            "unknown_unmetered": 0, "cost_final": True, "attempt_counts": {},
        }

    monkeypatch.setattr(ua, "ensure_legacy_imported", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(ua, "usage_breakdown", breakdown)
    older = threading.Thread(target=state.update_budget_from_usage, args=({},))
    newer = threading.Thread(target=state.update_budget_from_usage, args=({},))
    older.start()
    assert first_started.wait(2.0)
    newer.start()
    time.sleep(0.1)
    assert calls == [1]
    release_first.set()
    older.join(2.0)
    newer.join(2.0)

    assert calls == [1, 2]
    assert state.load_state()["spent_usd"] == 2.0


def test_legacy_budget_projection_accepts_nullable_usage_cost(data_root):
    from supervisor import state

    state.init(data_root, total_budget_limit=0.0)
    reservation = ua.reserve_attempt(_request(
        data_root,
        provider="openai",
        model="openai::future-model",
        reservation_usd=None,
    ))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(
        reservation,
        {"prompt_tokens": 1, "completion_tokens": 1},
        cost_usd=None,
        cost_final=False,
    )
    state.update_budget_from_usage({"cost": None, "prompt_tokens": 1})

    stored = state.load_state()
    assert stored["spent_usd"] == 0.0
    assert stored["usage_accounting"]["unknown_unmetered"] == 1
    assert stored["usage_accounting"]["cost_final"] is False


def test_legacy_import_is_resumable_and_preserves_delta(data_root):
    events = data_root / "logs" / "events.jsonl"
    events.parent.mkdir(parents=True)
    usage = {
        "type": "llm_usage",
        "ts": "2026-01-01T00:00:00Z",
        "task_id": "t",
        "model": "openai/gpt-5.2",
        "provider": "openai",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "cost": 0.1,
    }
    events.write_text(
        "\n".join(
            (
                json.dumps(usage),
                json.dumps(usage),
                json.dumps({"type": "llm_round", "ts": "2026-01-01T00:00:01Z", "task_id": "ambiguous"}),
            )
        )
        + "\n"
    )
    (data_root / "state" / "state.json").write_text(
        json.dumps(
            {
                "spent_usd": 0.4,
                "spent_calls": 3,
            }
        )
    )
    settings = data_root / "settings.json"
    settings.write_text('{"secret":"unchanged"}\n')
    before = settings.read_bytes()

    first = ua.ensure_legacy_imported(data_root)
    row_count = len(_ledger(data_root))
    second = ua.ensure_legacy_imported(data_root)

    assert first["legacy_usage_count"] == 1
    assert first["legacy_metadata_count"] == 2
    assert first["legacy_delta_usd"] == 0.3
    assert first["legacy_baseline_source"] == "state.json"
    assert second == first
    assert len(_ledger(data_root)) == row_count
    projection = ua.usage_projection(data_root)
    assert projection["settled_usd"] == 0.4
    assert projection["unknown_unmetered"] == 0
    assert projection["attempt_counts"]["metadata_only"] == 2
    assert settings.read_bytes() == before
    manifests = list((data_root / "archive" / "usage_import").glob("*/sha256.json"))
    assert len(manifests) == 1
    archive = manifests[0].parent
    archived_hashes = json.loads(manifests[0].read_text())
    for name, expected in first["source_sha256"].items():
        assert archived_hashes[name] == expected
        if expected and name != "settings.json":
            assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == expected
    assert not (archive / "settings.json").exists()
    assert first["quarantined_test_operator_rows"] == 0
    assert first["test_operator_quarantine_policy"] == "typed_evidence_only_no_inference"


def test_completed_import_is_immutable_without_a_second_repair_api(data_root):
    events = data_root / "logs" / "events.jsonl"
    events.parent.mkdir(parents=True)
    usage_rows = [
        {
            "type": "llm_usage",
            "task_id": f"t{index}",
            "model": "openai/gpt-5.2",
            "provider": "openai",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cost": 0.1,
        }
        for index in range(2)
    ]
    events.write_text("\n".join(json.dumps(row) for row in usage_rows) + "\n")
    (data_root / "state" / "state.json").write_text(
        json.dumps({"spent_usd": 0, "spent_calls": 1})
    )
    (data_root / "settings.json").write_text('{"secret":"unchanged"}\n')

    incomplete = ua.ensure_legacy_imported(data_root)
    original_ledger = (data_root / ua.LEDGER_REL).read_bytes()
    original_watermark = (data_root / ua.IMPORT_REL).read_bytes()
    assert incomplete["legacy_baseline_source"] == "state.json"
    assert incomplete["legacy_usage_count"] == 2
    assert incomplete["legacy_metadata_count"] == 0

    assert ua.ensure_legacy_imported(data_root) == incomplete
    assert (data_root / ua.LEDGER_REL).read_bytes() == original_ledger
    assert (data_root / ua.IMPORT_REL).read_bytes() == original_watermark


def test_concurrent_legacy_importers_share_one_exact_snapshot(data_root, monkeypatch):
    events = data_root / "logs" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        json.dumps(
            {
                "type": "llm_usage",
                "task_id": "t",
                "model": "openai/gpt-5.2",
                "provider": "openai",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "cost": 0.1,
            }
        )
        + "\n"
    )
    (data_root / "state" / "state.json").write_text(json.dumps({"spent_usd": 0.1, "spent_calls": 1}))
    (data_root / "settings.json").write_text('{"secret":"unchanged"}\n')

    original = ua._legacy_snapshot
    calls = 0
    calls_lock = threading.Lock()
    barrier = threading.Barrier(4)

    def snapshot(root):
        nonlocal calls
        assert not (root / "state" / "usage_attempts.lock").exists()
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return original(root)

    def import_once(_index):
        barrier.wait()
        return ua.ensure_legacy_imported(data_root)

    monkeypatch.setattr(ua, "_legacy_snapshot", snapshot)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(import_once, range(4)))

    assert calls == 1
    assert all(result == results[0] for result in results)
    assert results[0]["legacy_usage_count"] == 1
    assert len(_ledger(data_root)) == 1


def test_actor_limit_blocks_third_retry_before_provider_send(data_root, monkeypatch):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="unused")
    sends = 0

    def create(**kwargs):
        nonlocal sends
        sends += 1
        raise RuntimeError(f"provider failure {sends}")

    monkeypatch.setattr(
        client,
        "_retry_without_optional_sampling",
        lambda kwargs, model, exc: {**kwargs, "temperature": None},
    )
    monkeypatch.setattr(
        client,
        "_openrouter_signature_retry_kwargs",
        lambda target, kwargs, exc: {**kwargs, "messages": []},
    )
    target = {
        "provider": "openai",
        "usage_model": "openai/gpt-5.2",
        "resolved_model": "gpt-5.2",
    }
    with ua.physical_attempt_limit(2), pytest.raises(ua.PhysicalAttemptLimitExceeded):
        client._create_chat_completion_with_retries(
            create,
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "x"}]},
            target,
        )

    assert sends == 2
    assert ua.usage_projection(data_root)["attempt_counts"] == {
        "unresolved": 2,
        "released": 1,
    }


def test_env_zero_is_unbounded_but_explicit_zero_is_a_hard_rail(data_root, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "0")
    request = ua.AttemptRequest(
        model="local/test",
        provider="local",
        drive_root=data_root,
    )
    reservation = ua.reserve_attempt(request)
    ua.release_attempt(reservation)
    assert "limit_usd" not in ua.usage_projection(data_root)

    with pytest.raises(ua.BudgetExceeded) as exc_info:
        ua.reserve_attempt(
            ua.AttemptRequest(
                model="local/test",
                provider="local",
                drive_root=data_root,
                global_limit_usd=0,
            )
        )
    assert exc_info.value.limit_scope == "global"


def test_explicit_zero_root_limit_blocks_only_that_root(data_root, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "0")
    with pytest.raises(ua.BudgetExceeded) as exc_info:
        ua.reserve_attempt(
            ua.AttemptRequest(
                model="local/test",
                provider="local",
                drive_root=data_root,
                task_id="task-a",
                root_task_id="root-a",
                root_limit_usd=0,
            )
        )
    assert exc_info.value.limit_scope == "root"
    assert exc_info.value.root_task_id == "root-a"


def test_claude_sdk_reserves_max_budget_and_settles_actual(data_root, monkeypatch):
    from ouroboros.gateways import claude_code as cc

    class Result:
        session_id = "session"
        total_cost_usd = 0.12
        usage = {"input_tokens": 20, "output_tokens": 4}
        subtype = "success"

    class Client:
        def __init__(self, options):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def query(self, prompt):
            return None

        async def receive_response(self):
            yield Result()

    monkeypatch.setattr(cc, "ClaudeAgentOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(cc, "ClaudeSDKClient", Client)
    monkeypatch.setattr(cc, "ResultMessage", Result)

    result = asyncio.run(cc._run_edit_async("do work", str(data_root), budget=2.0))
    assert result.success is True
    assert result.usage["ledger_attempt_ids"]
    projection = ua.usage_projection(data_root)
    assert projection["confirmed_usd"] == 0.12
    rows = _ledger(data_root)
    assert rows[0]["reservation_upper_bound_usd"] == 2.0


def test_body_error_zero_usage_settles_confirmed_zero():
    # A top-level provider body-error (OpenRouter passes 429/5xx through the body
    # of an HTTP-200) that billed zero tokens is a request rejected before
    # generation — settle a confirmed $0, not an unknown cost that holds the bound.
    normalized, cost, final = ua.usage_from_response(
        {"error": {"code": 429, "message": "rate limited"}, "choices": None, "usage": None}
    )
    assert cost == 0.0
    assert final is True
    assert normalized["prompt_tokens"] == 0
    assert normalized["completion_tokens"] == 0


def test_billed_tokens_with_error_field_keep_the_bound():
    # A partial stream / real completion that ALSO carries an error field but billed
    # tokens must NOT be zeroed — it keeps the normal cost path (and its bound).
    normalized, cost, final = ua.usage_from_response(
        {"error": {"code": 500}, "usage": {"prompt_tokens": 40, "completion_tokens": 8}}
    )
    assert cost is None  # no cost field -> unknown, falls through (not forced to 0)
    assert final is False
    assert normalized["prompt_tokens"] == 40


def test_body_error_storm_does_not_phantom_exhaust_budget(data_root):
    # Full-path regression on a PRICED model: seven body-error attempts (the shape
    # that killed SWE-Pro tasks) must release their reservation bounds instead of
    # accumulating a phantom unresolved sum that exhausts the finite budget.
    class _BodyErrResp:
        def model_dump(self):
            return {"error": {"code": 429, "message": "rate limited"}, "usage": None}

    for i in range(7):
        ua.execute_physical_attempt(
            _request(
                data_root,
                task_id=f"storm{i}",
                model="openai/gpt-5.5",
                prompt_tokens_estimate=200000,
                max_completion_tokens=4000,
                global_limit_usd=25.0,
                root_limit_usd=25.0,
            ),
            lambda: _BodyErrResp(),
        )

    projection = ua.usage_projection(data_root, global_limit_usd=25.0)
    assert projection["unresolved_upper_bound_usd"] == 0.0
    assert projection["settled_usd"] == 0.0
    # a subsequent real reservation still fits the untouched budget
    reservation = ua.reserve_attempt(
        _request(data_root, task_id="after", model="openai/gpt-5.5", global_limit_usd=25.0)
    )
    assert reservation is not None


def test_cache_bearing_sends_are_measured_on_every_cache_inclusive_route(tmp_path):
    """A cached send must still teach density on routes whose prompt_tokens is a TOTAL.

    The skip exists for routes that report cache tokens OUTSIDE prompt_tokens, where a
    partially cached call would look falsely cheap and LOOSEN the review-pack cap. It must
    not fire on routes that already fold cache reads and writes in — every review surface
    marks a stable prefix, so skipping those would make the measurement path vacuous and
    freeze every pack at the cold-start density forever.

    Pinned as a CONTRACT, not as the current membership list: direct-Anthropic became
    cache-inclusive in v6.77.0 (`llm.py` folds cache_read/cache_creation into
    prompt_tokens) but was left out of the set until v6.81.0, which silently killed
    measurement on the main and heavy slots. Nothing failed, because nothing tested it."""
    from ouroboros.capability_evidence import _DENSITY_MEMO, get_token_density
    from ouroboros.provider_models import normalize_model_identity

    cached_usage = {
        "prompt_tokens": 1_500_000,   # already INCLUDES the cache reads below
        "cached_tokens": 900_000,
        "cache_write_tokens": 100_000,
    }

    for provider, model in (
        ("anthropic", "anthropic/claude-sonnet-5"),
        ("openrouter", "openrouter/some-model"),
        ("openai", "openai/gpt-5.5"),
        ("openai-compatible", "compat/some-model"),
        ("cloudru", "cloudru/some-model"),
        ("local", "local/some-model"),
    ):
        _DENSITY_MEMO.clear()
        root = tmp_path / provider
        ua._observe_token_density(
            ua.AttemptRequest(
                model=model,
                provider=provider,
                prompt_tokens_estimate=1_000_000,
                drive_root=root,
            ),
            dict(cached_usage),
        )
        measured = get_token_density(root, normalize_model_identity(model))
        assert abs(measured - 1.5) < 1e-6, f"{provider} must be measured, got {measured}"

    # The other direction: a route whose cache-token semantics are undocumented is still
    # skipped, because there an under-measured density loosens the cap.
    _DENSITY_MEMO.clear()
    root = tmp_path / "gigachat"
    ua._observe_token_density(
        ua.AttemptRequest(
            model="gigachat/some-model",
            provider="gigachat",
            prompt_tokens_estimate=1_000_000,
            drive_root=root,
        ),
        dict(cached_usage),
    )
    assert get_token_density(root, normalize_model_identity("gigachat/some-model")) == 0.0

    # And an UNCACHED send on that same route is measured normally — the skip is about
    # cache accounting, not about distrusting the provider.
    _DENSITY_MEMO.clear()
    root = tmp_path / "gigachat_uncached"
    ua._observe_token_density(
        ua.AttemptRequest(
            model="gigachat/some-model",
            provider="gigachat",
            prompt_tokens_estimate=1_000_000,
            drive_root=root,
        ),
        {"prompt_tokens": 1_500_000},
    )
    measured = get_token_density(root, normalize_model_identity("gigachat/some-model"))
    assert abs(measured - 1.5) < 1e-6
