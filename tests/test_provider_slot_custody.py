"""The physical sender retains capacity after its logical receiver leaves."""
import threading

import pytest

from ouroboros import model_concurrency as mc, owner_pause
from tests.test_model_concurrency import ROUTE, _records, _until
from tests.test_pause_model_receiver import _call, scope as scope
from tests._usage_store_testing import attempt_rows_in_start_order

# Holders below take the same model key as the wrapped physical send (ROUTE names no
# usage_model), so they contend for that model's slots on this route.

pytestmark = pytest.mark.serial


def test_abandoned_receiver_does_not_release_physical_slot(tmp_path, monkeypatch, scope):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC", "30")
    sent, release = threading.Event(), threading.Event()
    outcome = {}

    def send():
        sent.set()
        assert release.wait(60)
        return {"choices": [{"message": {"content": "late answer"}}]}

    thread, _owner = _call(tmp_path, scope, mc.limit_physical_send(ROUTE, send), outcome)
    try:
        assert sent.wait(30)
        assert len(_records(tmp_path, "slot")) == 1
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        _until(lambda: "returned_at" in outcome)
        assert getattr(outcome["error"], "receiver_abandoned", False)
        assert len(_records(tmp_path, "slot")) == 1
        with pytest.raises(mc.ModelSlotTimeout):
            with mc.model_call_slot("", route=ROUTE, deadline_ts=0):
                pytest.fail("receiver abandonment released a live sender's slot")
    finally:
        release.set()
        thread.join(30)
        assert not thread.is_alive()
    _until(lambda: not _records(tmp_path, "slot"))
    _until(lambda: attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled")
    with mc.model_call_slot("", route=ROUTE):
        assert len(_records(tmp_path, "slot")) == 1
    assert not _records(tmp_path, "slot")


def test_capacity_timeout_records_no_dispatch_then_a_free_slot_sends(tmp_path, monkeypatch, scope):
    from ouroboros import usage_accounting as ua
    from tests.test_pause_model_receiver import _request, _extract

    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    sends = []
    wrapped = mc.limit_physical_send(ROUTE, lambda: sends.append(True) or {"answer": "ok"})
    with ua.usage_scope(scope):
        with mc.model_call_slot("", route=ROUTE):
            with mc.model_call_scope("waiting", deadline_ts=0):
                with pytest.raises(mc.ModelSlotTimeout) as refusal:
                    ua.execute_physical_attempt(_request(), wrapped, extractor=_extract)
            assert refusal.value.physical_attempt_capture.state == "released"
            assert sends == []
        assert ua.execute_physical_attempt(_request(), wrapped, extractor=_extract) == {"answer": "ok"}
    assert sends == [True]
    assert [row["state"] for row in attempt_rows_in_start_order(tmp_path)] == ["released", "settled"]


def test_owner_pause_cancels_a_queued_sender_before_physical_dispatch(tmp_path, monkeypatch, scope):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC", "10")
    sends, outcome = [], {}
    wrapped = mc.limit_physical_send(ROUTE, lambda: sends.append(True) or {"answer": "ok"})
    with mc.model_call_slot("", route=ROUTE):
        thread, _owner = _call(tmp_path, scope, wrapped, outcome, abandonable=False)
        try:
            _until(lambda: bool(_records(tmp_path, "ticket")))
            owner_pause.install_fence(tmp_path, "root", request_id="queued-pause")
            _until(lambda: "returned_at" in outcome)
            assert getattr(outcome["error"], "control_reason", "") == "owner_pause"
            assert not sends and not _records(tmp_path, "ticket")
            assert len(_records(tmp_path, "slot")) == 1
            assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "released"
        finally:
            thread.join(30)
            assert not thread.is_alive()
    owner_pause.release_fence(tmp_path, "root", reason="test_resume")
    resumed = {}
    thread, _owner = _call(tmp_path, scope, wrapped, resumed, abandonable=False)
    thread.join(30)
    assert not thread.is_alive()
    assert resumed.get("result") == {"answer": "ok"} and sends == [True]
    assert [row["state"] for row in attempt_rows_in_start_order(tmp_path)] == ["released", "settled"]
