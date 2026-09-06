"""``IsolatedServer.wait_for_absorb`` may answer ``absorbed=False`` early only with PROOF that no cycle is
pending (rc.15 stand, adversarial finding of 2026-09-06): a cycle that committed keeps its campaign transaction
as ``waiting_for_restart`` through the synchronous supervisor restart (queue idle, counter unchanged) and the
re-exec'd server answers ``/api/state`` with zero counts before its supervisor is ready; one idle sample used to
end the wait as ``no_promotion`` right there. The typed idle reasons come from the durable campaign state."""
from __future__ import annotations

import json
import pathlib

import pytest

from devtools.benchmarks.common import server_runner


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += float(seconds)


def _server(tmp_path: pathlib.Path, states: list, monkeypatch, *, sha: str = "aaaaaaaa") -> server_runner.IsolatedServer:
    """``states`` feeds the idle poll only (one read per poll after the grace); the served sha is fixed."""
    clock = _Clock()
    monkeypatch.setattr(server_runner, "time", clock)
    srv = object.__new__(server_runner.IsolatedServer)
    srv.data_root = tmp_path
    srv.base_url = "http://127.0.0.1:1"
    calls = {"n": 0}

    def _state(timeout: float = 5) -> dict:
        st = states[min(calls["n"], len(states) - 1)]
        calls["n"] += 1
        if isinstance(st, Exception):
            raise st
        return st

    srv._state = _state
    srv.current_sha = lambda: sha
    srv.wait_for_health = lambda timeout=180: True
    return srv


def _campaign(root: pathlib.Path, **fields) -> None:
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state" / "evolution_campaign.json").write_text(json.dumps({"schema_version": 1, **fields}), encoding="utf-8")


IDLE = {"sha": "aaaaaaaa", "pending_count": 0, "running_count": 0, "supervisor_ready": True}
BOOTING = {"sha": "", "pending_count": 0, "running_count": 0, "supervisor_ready": False}
BUSY = {"sha": "aaaaaaaa", "pending_count": 0, "running_count": 1, "supervisor_ready": True}


def test_an_idle_queue_with_a_transaction_waiting_for_restart_is_not_a_declined_promotion(tmp_path, monkeypatch):
    _campaign(tmp_path, status="active", source="post_task", absorbed_cycles_done=0,
              active_transaction={"commit_sha": "c" * 40, "cycle_outcome": "waiting_for_restart"})
    srv = _server(tmp_path, [IDLE], monkeypatch)
    out = srv.wait_for_absorb("aaaaaaaa", 0, timeout=200, idle_grace=10, idle_polls=2)
    assert out["reason"] == "timeout" and out["absorbed"] is False and out["campaign"]["active_transaction"] is True


def test_the_booting_reexecd_server_and_a_single_idle_sample_do_not_end_the_wait(tmp_path, monkeypatch):
    _campaign(tmp_path, status="active", source="post_task", absorbed_cycles_done=0, transaction_history=[])
    # after the grace: booting, idle, busy, idle, idle -> the streak of two idle polls only forms on the fifth poll
    srv = _server(tmp_path, [BOOTING, IDLE, BUSY, IDLE, IDLE], monkeypatch)
    out = srv.wait_for_absorb("aaaaaaaa", 0, timeout=10_000, idle_grace=10, idle_polls=2)
    assert out["reason"] == "cycle_not_enqueued" and out["absorbed"] is False
    assert server_runner.time.now == 35   # polls at t=15,20,25,30,35: the streak completes on the fifth


def test_a_pending_request_file_blocks_the_early_exit(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "post_task_evolution_request.json").write_text("{}", encoding="utf-8")
    srv = _server(tmp_path, [IDLE], monkeypatch)
    assert srv.wait_for_absorb("aaaaaaaa", 0, timeout=100, idle_grace=10, idle_polls=1)["reason"] == "timeout"


def test_the_absorb_is_confirmed_when_the_counter_and_the_served_sha_move(tmp_path, monkeypatch):
    _campaign(tmp_path, status="active", source="post_task", absorbed_cycles_done=1,
              transaction_history=[{"cycle_outcome": "absorbed"}])
    srv = _server(tmp_path, [IDLE], monkeypatch, sha="bbbbbbbb")
    out = srv.wait_for_absorb("aaaaaaaa", 0, timeout=100, idle_grace=10)
    assert out["absorbed"] is True and out["new_sha"] == "bbbbbbbb" and out["campaign"]["newest_outcome"] == "absorbed"


@pytest.mark.parametrize("campaign,counter,expected", [
    (None, 1, "no_promotion"),                                            # an every_n tick ran, nothing promoted
    (None, 0, "no_decision"),                                             # no tick recorded (llm cadence / ineligible)
    ({"status": "active", "transaction_history": [{"cycle_outcome": "no_op"}]}, 1, "cycle_no_op"),
    ({"status": "active", "transaction_history": [{"cycle_outcome": "abandoned"}]}, 1, "cycle_not_absorbed"),
    ({"status": "paused", "transaction_history": [{"cycle_outcome": "no_op"}]}, 1, "campaign_paused"),
    ({"status": "active"}, 1, "cycle_not_enqueued"),
])
def test_idle_reasons_are_typed_from_the_durable_campaign_state(tmp_path, monkeypatch, campaign, counter, expected):
    """The cycle written DURING the wait decides ``cycle_*``: the campaign file is created empty before the wait
    and the cycle's transaction lands after it started (a fresh one-shot campaign); a paused status wins."""
    (tmp_path / "state").mkdir(exist_ok=True)
    if counter:
        (tmp_path / "state" / "post_task_evolution_counter.json").write_text(json.dumps({"n": counter}), encoding="utf-8")
    srv = _server(tmp_path, [IDLE], monkeypatch)
    if campaign is not None:
        # the campaign appears after the wait's first summary: the first poll writes it, so its history is "new"
        original = srv._state

        def _state_then_campaign(timeout: float = 5) -> dict:
            _campaign(tmp_path, **campaign)
            return original(timeout)

        srv._state = _state_then_campaign
    out = srv.wait_for_absorb("aaaaaaaa", 0, timeout=1_000, idle_grace=10, idle_polls=3)
    assert out["reason"] == expected and out["absorbed"] is False
    assert out["campaign"]["post_task_counter"] == counter and out["campaign"]["present"] is (campaign is not None)


def test_older_cycles_of_a_resumed_campaign_never_speak_for_this_boundary(tmp_path, monkeypatch):
    """A campaign resumed across instances (CLB stateful, evolve_smoke --tasks N) carries the previous cycle's
    ``no_op``/``absorbed`` outcome in its history: a boundary that attaches no new cycle is ``cycle_not_enqueued``,
    not the older cycle's outcome."""
    _campaign(tmp_path, status="active", source="benchmark", absorbed_cycles_done=1,
              transaction_history=[{"cycle_outcome": "absorbed"}, {"cycle_outcome": "no_op"}])
    srv = _server(tmp_path, [IDLE], monkeypatch)
    out = srv.wait_for_absorb("aaaaaaaa", 1, timeout=1_000, idle_grace=10, idle_polls=2)
    assert out["reason"] == "cycle_not_enqueued" and out["campaign"]["history_len"] == 2
