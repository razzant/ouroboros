"""A fallback's failure cannot release or borrow a previous backend's attempt."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros import net_transport, usage_accounting as ua, usage_ledger as ledger
from ouroboros.tools import search
from tests._usage_store_testing import ledger_rows, request, root as root


@pytest.fixture
def search_calls(root, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "auto")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "")
    monkeypatch.setattr("ouroboros.search_routes._package_discoverable", lambda name: False)
    create, fallback = Mock(), Mock()
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    ctx = SimpleNamespace(task_id="child", task_metadata={"root_task_id": "dominant"}, pending_events=[])
    monkeypatch.setattr(net_transport, "web_search_openai_client", lambda **k: client)
    monkeypatch.setattr(search, "_web_search_openrouter", fallback)
    monkeypatch.setattr(search, "_responses_search_candidate", lambda *a: (
        {"provider": "openai"}, {}, request(root, provider="openai"), lambda r: None))
    return ctx, create, fallback


@pytest.mark.parametrize("settlement_fails", [False, True])
def test_empty_completed_search_keeps_paid_liability_on_fallback_error(root, search_calls, monkeypatch, settlement_fails):
    ctx, create, fallback = search_calls
    empty = SimpleNamespace(usage=None, output=[])
    create.return_value = iter([SimpleNamespace(type="response.completed", response=empty)])
    refusal = ledger.UsageLockUnavailable("fallback's own refusal", reason="contention")
    own_capture = ua.PhysicalAttemptCapture(attempt_id="fallback-attempt", model="stub", provider="openrouter",
                                            state="reserved", candidate_measurement_kind="opaque")
    refusal.physical_attempt_capture = own_capture
    fallback.side_effect = refusal
    if settlement_fails:
        monkeypatch.setattr(search, "settle_attempt", Mock(side_effect=ledger.UsageLockUnavailable(
            "settlement unavailable", reason="contention")))
    with ua.physical_attempt_limit(1):
        with pytest.raises(ledger.UsageLockUnavailable) as error:
            search._web_search(ctx, "synthetic query")
        assert error.value is refusal and error.value.physical_attempt_capture is own_capture
        [row] = ledger_rows(root)
        assert row["attempt_id"] != own_capture.attempt_id
        assert ua._PHYSICAL_LIMIT.get().claimed_ids == {row["attempt_id"]}
        assert ua._PHYSICAL_LIMIT.get().used == 1
    create.assert_called_once()
    fallback.assert_called_once()
    assert (row["state"], row["revision"]) == ("unresolved" if settlement_fails else "settled", 3)
    assert row.get("cost_usd") is None and row["reservation_upper_bound_usd"] == 1
    assert ua.usage_projection(root)["accounted_usd"] == 1


def test_fallback_lock_error_keeps_its_own_capture_after_typed_refusal(root, search_calls):
    ctx, create, fallback = search_calls
    authentication_error = RuntimeError("credentials rejected")
    authentication_error.status_code = 401
    authentication_error.body = {"error": {"type": "authentication_error"}}
    create.side_effect = authentication_error
    refusal = ledger.UsageLockUnavailable("fallback's own refusal", reason="contention")
    own_capture = ua.PhysicalAttemptCapture(attempt_id="fallback-attempt", model="stub", provider="openrouter",
                                            state="reserved", candidate_measurement_kind="opaque")
    refusal.physical_attempt_capture = own_capture
    fallback.side_effect = refusal
    with ua.physical_attempt_limit(1):
        with pytest.raises(ledger.UsageLockUnavailable) as error:
            search._web_search(ctx, "synthetic query")
        assert error.value is refusal
        assert error.value.physical_attempt_capture is own_capture
        [row] = ledger_rows(root)
        assert row["state"] == "settled" and row["attempt_id"] != own_capture.attempt_id
        assert ua._PHYSICAL_LIMIT.get().claimed_ids == {row["attempt_id"]}
        assert ua._PHYSICAL_LIMIT.get().used == 1
    create.assert_called_once()
    fallback.assert_called_once()
    assert ua.usage_projection(root)["accounted_usd"] == 1
