"""A fallback's failure cannot release or borrow a previous backend's paid attempt."""
from types import SimpleNamespace

import pytest

from ouroboros import net_transport, usage_accounting as ua, usage_ledger as ledger
from ouroboros.tools import search
from tests._usage_store_testing import ledger_rows, request, root as root

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("settlement_fails", [False, True])
def test_empty_completed_search_keeps_paid_liability_on_fallback_error(root, monkeypatch, settlement_fails):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "auto")
    calls = []
    def create(**kwargs):
        calls.append(1)
        return iter([SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None, output=[]))])
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    ctx = SimpleNamespace(task_id="child", task_metadata={"root_task_id": "dominant"}, pending_events=[])
    monkeypatch.setattr(search, "_web_search_backend_pin", lambda: "auto")
    monkeypatch.setattr(search, "_resolve_openai_client_settings", lambda: ("synthetic", "https://invalid.example/v1", "openai", "synthetic"))
    monkeypatch.setattr(net_transport, "web_search_openai_client", lambda **k: client)
    monkeypatch.setattr(search, "_responses_search_candidate", lambda *a: (
        None, {}, request(root, provider="openai"), lambda r: None))
    refusal = ledger.UsageLockUnavailable("fallback's own refusal", reason="contention")
    own_capture = SimpleNamespace(attempt_id="fallback-attempt", state="reserved")
    refusal.physical_attempt_capture = own_capture
    def fallback(*args, **kwargs):
        raise refusal
    monkeypatch.setattr(search, "_web_search_openrouter", fallback)
    if settlement_fails:
        monkeypatch.setattr(search, "settle_attempt", fallback)
    with ua.physical_attempt_limit(1):
        with pytest.raises(ledger.UsageLockUnavailable) as error:
            search._web_search(ctx, "synthetic query")
        assert error.value is refusal
        assert error.value.physical_attempt_capture is own_capture
        assert ua._PHYSICAL_LIMIT.get().used == 1
    # One current row per attempt; its revision counts reserve, dispatch, terminal.
    rows = ledger_rows(root)
    assert [(row["state"], row["revision"]) for row in rows] == [
        ("unresolved" if settlement_fails else "settled", 3)]
    assert calls == [1]
    assert ua.usage_projection(root)["accounted_usd"] == 1
