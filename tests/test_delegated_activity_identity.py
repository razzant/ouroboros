"""Run/seq identity survives Unicode, preview bounds and an omitted wait cursor."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_activity
from ouroboros.delegate_custody import RunCustody
from tests.test_delegated_activity import Daemon, _agent, _ctx, _gateway, _harness, _observe



def test_producer_fragment_offsets_are_unicode_code_points(tmp_path):
    daemon = Daemon([[_harness(seq, "message", text=text, delta=True)
                      for seq, text in enumerate(("😀", "ok", "🛰️"), 1)]])
    daemon.revealed = 1
    record = _observe(_ctx(tmp_path, []), _gateway(daemon), 3)
    part = record["parts"][0]
    assert (part["text"], part["cuts"], part["chars"]) == ("😀ok🛰️", [[2, 1], [3, 3]], 5)
    assert [part["text"][offset:] for _, offset in part["cuts"]] == ["ok🛰️", "🛰️"]


def test_identity_budgets_disclose_the_bound_without_dropping_the_original(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    events = [_harness(seq, "message", text="a", delta=True) for seq in range(1, 1003)]
    daemon = Daemon([events])
    daemon.revealed = 1
    record = _observe(_ctx(tmp_path, []), _gateway(daemon), 1002)
    part = record["parts"][0]
    assert part["cuts_truncated"] and len(part["cuts"]) == 1000
    assert part["text"] == "a" * 1002 and part["last_seq"] == 1002
    assert len(read_actor_source_bytes(tmp_path, "child", record["source"]["ref"]).splitlines()) == 1002
    technical = delegate_activity._technical([
        {"seq": seq, "actor": "", "label": "Read", "detail": ""} for seq in range(1, 1002, 2)
    ])
    assert technical["count"] == 501 and technical["seqs_truncated"] and "seqs" not in technical


@pytest.mark.parametrize("first_terminal", [True, False])
def test_wait_without_since_seq_keeps_messages_already_present_on_first_poll(tmp_path, monkeypatch, first_terminal):
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    steps = [[_harness(1, "message", text="Already said before the wait.")]]
    if not first_terminal:
        steps.append([_harness(2, "message", text="Said while waiting.")])
    gateway = _gateway(Daemon(steps))
    ctx = _nanny_ctx(tmp_path, task_id="child")
    ctx.task_attempt = 0
    _, events = _agent(ctx)
    monkeypatch.setitem(delegate._CUSTODY, "run-1", RunCustody(
        task_id="child", run_id="run-1", route_id="claude", model="m", access="readonly"))
    monkeypatch.setattr(delegate, "_capture_terminal_patch", lambda *a, **k: None)
    clock = SimpleNamespace(now=0.0)
    with monkeypatch.context() as timing:
        timing.setattr(delegate.time, "monotonic", lambda: clock.now)
        timing.setattr(delegate.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
        result = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=60, gateway=gateway))
    assert result["status"] == "terminal"
    frames = [events.get_nowait() for _ in range(events.qsize())]
    records = [frame["progress_meta"]["delegated_activity"] for frame in frames if frame["type"] == "send_message"]
    assert [part["text"] for record in records for part in record["parts"]] == [
        event["payload"]["text"] for step in steps for event in step]
    assert records[0]["after_seq"] == 0
