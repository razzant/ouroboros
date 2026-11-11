"""The conversation lane's carriers in the vitals (owner decision 4A, review W6).

No number bounds concurrent carriers: the bound is spawn success, and the vitals show the carriers and each
one's RSS. A carrier measures its own size in-process and reports it on its turn's heartbeat; the server
keeps the newest report on the slot; the per-minute ``supervisor_load`` row lists every carrier, and the
vitals line the mind reads renders that row.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from types import SimpleNamespace

MIB = 1048576
CARRIED = {"kind": "direct", "lane": "conversation", "executor": "worker"}


def _stamp(seconds_ago: float = 0.0) -> str:
    return (dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(seconds=seconds_ago)).isoformat()


def _slot(wid, *, lane="conversation", busy=None):
    return SimpleNamespace(wid=wid, lane=lane, executor="worker", busy_task_id=busy,
                           proc=SimpleNamespace(pid=1000 + wid))


def test_a_carried_turns_heartbeat_reports_its_carriers_own_size_to_the_lane_census(tmp_path, monkeypatch):
    from ouroboros import cost_projection, update_generation_facts
    from supervisor import conversation_lane, workers
    from supervisor.events_worker_reports import _handle_task_heartbeat
    from supervisor.log_addressing import task_event_queue
    from supervisor.worker_chat_lane import CarrierTurn

    monkeypatch.setattr(update_generation_facts, "_rss", lambda: (812 * MIB, "peak"))
    monkeypatch.setattr(cost_projection, "live_root_cost_projection", lambda *a, **k: {})
    task = {"id": "turn-v", "type": "task", "chat_id": 1, "project_id": "p", "_execution_owner": dict(CARRIED),
            "_conversation_prepare": {"admitted_at": time.time(), "reply_expected": False}}
    sent: list = []
    carrier = CarrierTurn(task, tmp_path, SimpleNamespace(put=sent.append))
    try:  # the carrier side: the agent's beat passes the turn's endpoint, which the carrier observes
        task_event_queue(SimpleNamespace(put=sent.append), task, **carrier.hooks()).put(
            {"type": "task_heartbeat", "task_id": "turn-v", "phase": "running", "ts": _stamp()})
    finally:
        carrier.ended()
    [beat] = [frame for frame in sent if frame.get("type") == "task_heartbeat"]
    assert beat["_carrier_rss"] == {"rss_bytes": 812 * MIB, "rss_measure": "peak"}

    # The server side: the newest beat's report stays on its worker's slot, a project slot's too (memory admission
    # reads every live worker's own size, ``pool_admission.typical_size``).
    carrier_slot, spare, project = _slot(7, busy="turn-v"), _slot(8), _slot(1, lane="project", busy="job")
    pushed: list = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, WORKERS={7: carrier_slot, 8: spare, 1: project},
                          bridge=SimpleNamespace(push_log=pushed.append), RUNNING={
                              "turn-v": {"task": task, "worker_id": 7, "started_at": time.time() - 30},
                              "job": {"task": {"id": "job", "type": "task"}, "worker_id": 1,
                                      "started_at": time.time() - 30}})
    _handle_task_heartbeat(beat, ctx)
    _handle_task_heartbeat({**beat, "ts": _stamp(5), "_carrier_rss": {"rss_bytes": 3, "rss_measure": "peak"}}, ctx)
    _handle_task_heartbeat({"type": "task_heartbeat", "task_id": "job", "phase": "running", "ts": _stamp(),
                            "_carrier_rss": {"rss_bytes": 9 * MIB, "rss_measure": "peak"}}, ctx)
    assert carrier_slot.rss == {"rss_bytes": 812 * MIB, "rss_measure": "peak"}  # an older beat never overwrites
    assert project.rss == {"rss_bytes": 9 * MIB, "rss_measure": "peak"} and len(pushed) == 3

    # The census the per-minute row carries: every carrier, the spare before its first turn's beat included.
    monkeypatch.setattr(workers, "WORKERS", ctx.WORKERS)
    assert conversation_lane.carrier_vitals() == {"carriers": 2, "idle": 1, "rss_mib": [812, None],
                                                  "rss_measure": ["peak"]}
    from ouroboros import server_liveness

    assert server_liveness._load_sections()["conversation_lane"] is conversation_lane.carrier_vitals


def _journal(root, *rows):
    path = root / "logs" / "supervisor.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_the_vitals_show_the_lanes_carriers_and_their_rss_from_the_newest_load_row(tmp_path, monkeypatch):
    from ouroboros.context_runtime_facts import vitals_fact, vitals_line

    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")  # the mind's own carrier builds the vitals from the journal
    load = {"ts": "2026-11-02T19:59:00+00:00", "type": "supervisor_load", "window_sec": 60.0,
            "max_event_lag_sec": 0.2, "handlers": {}}
    _journal(tmp_path, {**load, "conversation_lane": {"carriers": 3, "idle": 1, "rss_mib": [812, None, 640],
                                                      "rss_measure": ["peak"]}})
    fact = vitals_fact(tmp_path, observed_at="2026-11-02T20:00:00+00:00")
    assert fact["conversation_carriers"] == "19:59: carriers 3, idle 1; rss 812 · ? · 640 MiB (peak)"
    assert "conversation carriers 19:59: carriers 3, idle 1; rss 812 · ? · 640 MiB (peak)" in vitals_line(fact)

    _journal(tmp_path, {**load, "conversation_lane": {"carriers": 1, "idle": 1, "rss_mib": [None],
                                                      "rss_measure": []}})
    assert vitals_fact(tmp_path, observed_at="2026-11-02T20:00:00+00:00")["conversation_carriers"] == \
        "19:59: carriers 1, idle 1"  # no report yet: no size is invented
    _journal(tmp_path, load)  # a row from before the section: no key, never "none"
    assert "conversation_carriers" not in vitals_fact(tmp_path, observed_at="2026-11-02T20:00:00+00:00")
