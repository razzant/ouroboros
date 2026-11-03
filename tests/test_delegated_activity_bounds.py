"""Journal reads share the wait's remaining allowance and existing owner controls."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import errno
import json
import os
from pathlib import Path
import stat

import httpx
import pytest

from ouroboros import cancel_intents, deadline_utils, delegate_activity, model_wait
from ouroboros.gateways import claudexor_run_events as run_events
from ouroboros.task_results import write_task_result
from tests.test_delegated_activity import Daemon, _ctx, _gateway, _harness, _records

pytestmark = pytest.mark.serial


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _transport(tmp_path, monkeypatch, on_request=None):
    clock = Clock()
    monkeypatch.setattr(delegate_activity, "time", clock)
    monkeypatch.setattr(run_events, "time", clock)
    daemon = Daemon([[_harness(1, "message", text="retained words")]])
    gateway = _gateway(daemon)
    daemon.revealed = 1
    requests = []

    def handler(request):
        requests.append(request)
        if on_request:
            on_request(request, clock)
        return daemon.handler(request)

    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler))
    return _ctx(tmp_path, []), gateway, clock, requests


def _no_read(record, requests, reason):
    assert requests == [], "a spent or controlled operation must not open catalog or SSE"
    assert record["source"]["provisional"] is True
    assert record["gaps"] == [{"after_seq": 0, "through_seq": 1, "reason": reason,
                               "final": True, "where": "run_journal"}]
    assert "ref" not in record["source"], "nothing was read or retained"
    assert "the exact range was not read" in delegate_activity.progress_text(record)


def test_zero_drain_opens_nothing_and_leaves_the_range_for_a_later_wait(tmp_path, monkeypatch):
    ctx, gateway, _clock, requests = _transport(tmp_path, monkeypatch)
    record, = _records(ctx, gateway, 1, drain=0)
    _no_read(record, requests, "read_bound:time")
    later, = _records(ctx, gateway, 1, after=1, drain=1)
    assert later["after_seq"] == 0 and later["source"]["ended"]
    assert later["parts"][0]["text"] == "retained words"


@pytest.mark.parametrize("control", ["cancelled", "panic", "owner_restart"])
def test_durable_stop_precedes_the_first_catalog_read(tmp_path, monkeypatch, control):
    ctx, gateway, _clock, requests = _transport(tmp_path, monkeypatch)
    if control == "cancelled":
        write_task_result(tmp_path, "child", "running")
        cancel_intents.request_cancel(tmp_path, "child", source="owner")
    else:
        (tmp_path / "state").mkdir(exist_ok=True)
        if control == "panic":
            from ouroboros import model_wait
            monkeypatch.setattr(model_wait, "current_model_wait", lambda: SimpleNamespace(control_reason=lambda: "panic"))
        else:
            from tests._work_stop_helpers import live_restart
            live_restart(tmp_path, monkeypatch)
    record, = _records(ctx, gateway, 1, drain=30)
    _no_read(record, requests, "owner_cancelled" if control == "cancelled" else "owner_panic" if control == "panic" else "owner_owner_restart")


@pytest.mark.parametrize("control", ["cancelled", "absolute_ceiling", "finalize_requested"])
def test_native_owner_control_precedes_any_journal_read(tmp_path, monkeypatch, control):
    ctx, gateway, _clock, requests = _transport(tmp_path, monkeypatch)
    with model_wait.task_model_wait_scope(task={"id": "child"}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=False,
                                          owner_control=lambda: control):
        record, = _records(ctx, gateway, 1, drain=30)
    _no_read(record, requests, f"owner_{control}")


@pytest.mark.parametrize("spent_in_catalog", [False, True])
def test_catalog_and_sse_share_the_actual_remaining_drain(tmp_path, monkeypatch, spent_in_catalog):
    def advance(request, clock):
        if request.url.path == "/v2/operations":
            clock.now += 0.025 if spent_in_catalog else 0.010

    ctx, gateway, _clock, requests = _transport(tmp_path, monkeypatch, advance)
    record, = _records(ctx, gateway, 1, drain=0.025)
    assert requests[0].extensions["timeout"]["read"] == pytest.approx(0.025)
    if spent_in_catalog:
        assert [r.url.path for r in requests] == ["/v2/operations"]
        assert record["gaps"][0]["reason"] == "read_bound:time"
    else:
        assert [r.url.path for r in requests] == ["/v2/operations", "/v2/runs/run-1/events"]
        assert requests[1].extensions["timeout"]["read"] == pytest.approx(0.015)
        assert record["source"]["ended"] is True


def test_native_control_arriving_during_catalog_prevents_sse(tmp_path, monkeypatch):
    control = [None]

    def stop(request, _clock):
        if request.url.path == "/v2/operations":
            control[0] = "cancelled"

    ctx, gateway, _clock, requests = _transport(tmp_path, monkeypatch, stop)
    with model_wait.task_model_wait_scope(task={"id": "child"}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=False,
                                          owner_control=lambda: control[0]):
        record, = _records(ctx, gateway, 1, drain=30)
    assert [r.url.path for r in requests] == ["/v2/operations"]
    assert record["gaps"][0]["reason"] == "owner_cancelled"


@pytest.mark.parametrize("left", [0, 0.025])
def test_task_deadline_narrows_the_drain_without_a_positive_floor(tmp_path, monkeypatch, left):
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    ctx, gateway, _clock, requests = _transport(tmp_path, monkeypatch)
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: now)
    ctx.task_metadata["deadline_at"] = (now + timedelta(
        seconds=effective_finalization_reserve_sec(ctx) + left)).isoformat()
    record, = _records(ctx, gateway, 1, drain=30)
    if left == 0:
        _no_read(record, requests, "owner_deadline")
    else:
        assert all(r.extensions["timeout"]["read"] == pytest.approx(left) for r in requests)
        assert record["source"]["ended"] is True


def test_inherited_execution_deadline_prevents_new_reads(tmp_path, monkeypatch):
    ctx, gateway, _clock, requests = _transport(tmp_path, monkeypatch)
    with model_wait.execution_deadline_scope(0):
        record, = _records(ctx, gateway, 1, drain=30)
    _no_read(record, requests, "owner_deadline")


def test_zero_reader_allowance_opens_no_stream(tmp_path, monkeypatch):
    _ctx_value, gateway, _clock, requests = _transport(tmp_path, monkeypatch)
    read = run_events.read_run_events(gateway, "run-1", after_seq=0, through_seq=None, timeout_sec=0)
    assert requests == [] and read.through_seq == 0 and read.stop == "time" and not read.events


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("spent_in", ["detail", "source_lookup"])
def test_spent_wait_admits_no_new_activity_read(tmp_path, monkeypatch, terminal, spent_in):
    from ouroboros.delegate_custody import RunCustody
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    def spend(request, clock):
        if request.url.path == "/v2/runs/run-1" and spent_in == "detail":
            clock.now += 1

    _unused, gateway, clock, requests = _transport(tmp_path, monkeypatch, spend)
    if not terminal:
        original = gateway._client

        def running(request):
            response = original.send(request)
            if request.url.path == "/v2/runs/run-1":
                body = response.json()
                body["summary"]["state"] = "running"
                return httpx.Response(200, json=body)
            return response

        gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(running))
    ctx = _nanny_ctx(tmp_path, task_id="child")
    activities = []
    ctx.emit_progress_fn = lambda _text, **meta: activities.append(meta["delegated_activity"])
    monkeypatch.setitem(delegate._CUSTODY, "run-1", RunCustody(
        task_id="child", run_id="run-1", route_id="claude", model="m", access="readonly"))
    monkeypatch.setattr(delegate, "_capture_terminal_patch", lambda *a, **k: None)
    monkeypatch.setattr(delegate, "time", clock)
    if spent_in == "source_lookup":
        real = delegate_activity._retained_range

        def spend_lookup(*args):
            clock.now += 1
            return real(*args)

        monkeypatch.setattr(delegate_activity, "_retained_range", spend_lookup)
    result = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1, since_seq=0, gateway=gateway))
    assert result["status"] == ("terminal" if terminal else "progress")
    assert not any(r.url.path.endswith(("/operations", "/events")) for r in requests)
    assert activities[0]["gaps"][0]["reason"] == "read_bound:time"
    assert activities[0]["gaps"][0].get("final", False) is terminal


@pytest.mark.parametrize("errors", ["native", "windows"])
@pytest.mark.parametrize("flag_name", ["panic_stop.flag", "owner_restart_no_resume.flag"])
@pytest.mark.parametrize("entry", ["dangling", "unreadable", "obstructed_parent"])
def test_installation_input_uncertainty_is_not_a_delegated_task_stop(tmp_path, monkeypatch, errors, flag_name, entry):
    from tests.test_state_authority import _windows_shaped_not_found

    ctx, gateway, _clock, requests = _transport(tmp_path, monkeypatch)
    if errors == "windows":
        _windows_shaped_not_found(monkeypatch)
    flag = tmp_path / "state" / flag_name
    if entry == "obstructed_parent":
        (tmp_path / "state").write_bytes(b"obstruction")
    else:
        real = os.lstat

        def lstat(path, *args, **kwargs):
            if Path(path) == flag:
                if entry == "unreadable":
                    raise PermissionError(errno.EACCES, "unreadable", str(flag))
                return os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 1, 0, 0, 0))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(os, "lstat", lstat)
    from ouroboros import cancel_intents
    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda *a: False)
    assert delegate_activity._owner_stop(ctx, "child") == ""
