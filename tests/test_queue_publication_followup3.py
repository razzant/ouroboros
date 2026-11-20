"""Queue publication, review follow-up 3 (RV_MAIN2 N1, N9, N10, N11 and the census NIT).

N1: the published pool's availability counts every started process (project slots and lane
carriers) and a disabled pool still refuses; pinned over a REAL published document at idle,
unstubbed. N9: a failed publication is retried by the next tick and ``queue.alive`` never
re-certifies the stale document meanwhile. N10: a newer capture always wins over an older one,
whatever their revisions. N11: the retry-custody publication repeats only when the claim's
facts change. NIT: the census carries only the model waits that are still waiting.
"""

import json
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status
from supervisor import queue, queue_publication, queue_snapshot


@pytest.fixture
def published(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    for name, value in (("PENDING", []), ("RUNNING", {}), ("BUDGET_ROOT_FENCES", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    monkeypatch.setattr(queue_publication, "utc_now_iso", lambda: datetime.fromtimestamp(now[0], timezone.utc).isoformat())
    monkeypatch.setattr(queue_publication, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None))
    for module in (queue_observation, task_status):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(queue_observation, "_sleep", lambda _sec: None)
    queue_publication.reset()
    yield tmp_path, now
    queue_publication.reset()


def _read(root, name):
    return json.loads((root / "state" / name).read_text(encoding="utf-8"))


def _slot(wid, lane, busy=None):
    return types.SimpleNamespace(wid=wid, lane=lane, busy_task_id=busy, active_capacity=True,
                                 reaping=False, readiness_exhausted=False, process_birth="b",
                                 ready_identity=(100 + wid, "b"), proc=types.SimpleNamespace(pid=100 + wid, is_alive=lambda: True))


def _idle_pool(monkeypatch, *, disabled=""):
    """The pool as boot leaves it and as every quiet period returns it: the lane's spare alone."""
    from supervisor import worker_pool_lifecycle, workers
    from supervisor.worker_pool_lifecycle import LANE_CONVERSATION

    monkeypatch.setattr(workers, "WORKERS", {0: _slot(0, LANE_CONVERSATION)})
    monkeypatch.setattr(workers, "_worker_pool_execution_state", lambda: {"disabled_reason": disabled})
    monkeypatch.setattr(worker_pool_lifecycle, "_slot_pid", lambda s: s.proc.pid)


# ---- N1: availability over the published document at idle ---------------------------------------

def test_an_idle_pool_with_the_lanes_spare_is_available_and_a_disabled_pool_refuses(published, monkeypatch):
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.tools.control_events import _promotion_pool_disabled_from_snapshot

    root, _now = published
    _idle_pool(monkeypatch)
    assert queue.persist_queue_snapshot("boot") is True
    document = _read(root, "queue_snapshot.json")
    assert document["worker_total"] == 0 and document["lane_carrier_total"] == 1, "the real idle shape"
    ctx = types.SimpleNamespace(drive_root=root, budget_drive_root="")
    assert _promotion_pool_disabled_from_snapshot(ctx) == "", "a started process answers; a quiet project pool is not absence"
    fact = _queue_context_fact({"budget_drive_root": str(root)})
    assert fact["worker_total"] == 0 and fact["lane_carrier_total"] == 1
    # A disabled or exhausted pool still refuses, whatever is started.
    _idle_pool(monkeypatch, disabled="crash_storm")
    assert queue.persist_queue_snapshot("storm") is True
    assert _promotion_pool_disabled_from_snapshot(ctx) == "crash_storm"
    # No process at all is absence.
    from supervisor import workers
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "_worker_pool_execution_state", lambda: {"disabled_reason": "no_workers"})
    assert queue.persist_queue_snapshot("empty") is True
    assert _promotion_pool_disabled_from_snapshot(ctx) == "no_workers"


class _EventSink:
    """The supervisor's live event queue that answers nothing (``test_promote_admission_pending._DeadQueue``)."""

    def __init__(self) -> None:
        self.events = []

    def put_nowait(self, event) -> None:
        self.events.append(event)


def test_a_host_thread_alone_is_no_pool_in_the_document_the_certificate_and_start_task(published, monkeypatch):
    """#3149: a Host thread surviving a failed spawn runs its turn on a negative pseudo-slot. Its owner counts no
    pool there (``no_workers``); the document, the certificate naming it and the readers deciding on them say the
    same: a carrier's ``pool_available``, and ``start_task``'s own door, which refuses typed before any effect. A
    real spare beside it is a started pool again: the same door emits its one event (nothing answers it here)."""
    from ouroboros import config, utils
    from ouroboros.model_detach import pool_available
    from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import control_events
    from ouroboros.tools.control_routing import _start_task
    from supervisor import workers
    from supervisor.worker_pool_lifecycle import LANE_CONVERSATION

    root, _now = published
    host = _slot(-2, LANE_CONVERSATION, busy="presence-turn")
    host.executor = EXECUTOR_INLINE_DEGRADED
    monkeypatch.setattr(workers, "WORKERS", {-2: host})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0.05)  # the unanswered wait, kept short
    monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_POLL_SEC", 0.005)

    def published_facts(reason):
        assert queue.persist_queue_snapshot(reason) is True
        document, alive = _read(root, "queue_snapshot.json"), _read(root, "queue.alive")
        assert alive["generation"] == document["generation"], "the certificate names this document"
        with monkeypatch.context() as carrier:  # a conversation carrier reads the published document
            carrier.setattr(utils, "in_worker_process", lambda: True)
            detach = pool_available()["disabled_reason"]
        return (workers._worker_pool_execution_state()["disabled_reason"],  # the owner's own answer
                document["lane_carrier_total"], alive["pool"]["lane_carrier_total"], detach)

    def start_task():  # the real door, with the context ``test_promote_admission_pending._promote_ctx`` uses
        ctx = types.SimpleNamespace(pending_events=[], event_queue=_EventSink(), current_chat_id=1, drive_root=root,
                                    budget_drive_root=str(root), project_id="", task_metadata={}, task_id="")
        return _start_task(ctx, "Look into the report", title="Report check", workspace="none"), ctx

    def refused_before_any_effect(out, ctx):
        assert out.startswith("⚠️ PROMOTE_REJECTED: task ") and "(worker_pool_unavailable: no_workers)" in out, out
        task_id = out.split(" was not scheduled", 1)[0].rsplit(" ", 1)[1]
        assert ctx.event_queue.events == [] and ctx.pending_events == [], "no event emitted"
        assert not load_task_result(root, task_id), "no durable intent written"

    assert published_facts("host_thread") == ("no_workers", 0, 0, "no_workers")
    refused_before_any_effect(*start_task())
    workers.WORKERS[0] = _slot(0, LANE_CONVERSATION)  # a real spare starts beside it
    assert published_facts("spare") == ("", 1, 1, "")
    out, ctx = start_task()
    assert out.startswith("Not confirmed yet: separate task "), out
    [event] = ctx.event_queue.events
    assert event["type"] == "promote_chat_to_task"
    assert load_task_result(root, event["task_id"])["promotion_admission"]["status"] == "emitted"
    del workers.WORKERS[0]  # the spare retires while the Host thread is still unwinding
    assert published_facts("spare_retired") == ("no_workers", 0, 0, "no_workers")
    refused_before_any_effect(*start_task())


# ---- a failed pool measurement is published as unknown, never as an empty pool -------------------

def test_a_failed_pool_measurement_publishes_unknown_and_every_reader_keeps_it_unknown(published, monkeypatch):
    from ouroboros import config, utils
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.engine_facts import engine_pool_line
    from ouroboros.model_detach import pool_available
    from ouroboros.tools.control_events import _promotion_pool_disabled_from_snapshot
    from supervisor import worker_pool_lifecycle

    root, _now = published
    _idle_pool(monkeypatch)

    def broken(*_a, **_kw):
        raise RuntimeError("pool table mid-rebind")

    with monkeypatch.context() as failing:
        failing.setattr(worker_pool_lifecycle, "pool_slot_counts", broken)
        assert queue.persist_queue_snapshot("introspection_failed") is True
    document = _read(root, "queue_snapshot.json")
    assert document["pool_observation_status"] == "unknown"
    assert all(document[key] is None for key in queue_publication._POOL_COUNT_KEYS), "never zeros"
    assert "project_demand" not in document  # no forecast is published; the counters say what acts and waits
    # The control projections refuse as the old zero-count form did, under the honest reason.
    ctx = types.SimpleNamespace(drive_root=root, budget_drive_root="")
    assert _promotion_pool_disabled_from_snapshot(ctx) == "unknown"
    with monkeypatch.context() as carrier:  # a conversation carrier reads the published document
        carrier.setattr(utils, "in_worker_process", lambda: True)
        carrier.setattr(config, "DATA_DIR", root)
        state = pool_available()
    assert (state["available"], state["disabled_reason"], state["worker_count"]) == (False, "unknown", None)
    fact = _queue_context_fact({"budget_drive_root": str(root)})
    assert (fact["pool_observation_status"], fact["free_worker_slots"], fact["free_worker_slots_basis"],
            fact["worker_total"]) == ("unknown", None, "unknown_pool_observation", None)
    assert ": unknown (introspection failed)" in engine_pool_line(root)
    # The measurement back: an observed idle pool again.
    assert queue.persist_queue_snapshot("measured") is True
    assert _read(root, "queue_snapshot.json")["pool_observation_status"] == "observed"
    assert _promotion_pool_disabled_from_snapshot(ctx) == ""


def test_a_document_from_before_the_status_reads_its_failed_measurement_as_unknown(tmp_path):
    """The earlier publisher's failure form: zeros with ``worker_pool_disabled_reason: unknown``."""
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.queue_observation import pool_observation_status
    from ouroboros.tools.control_events import _promotion_pool_disabled_from_snapshot

    legacy = {"running": [], "pending": [], "pending_count": 0, "running_count": 0, "worker_total": 0,
              "active_worker_count": 0, "assignable_idle_workers": 0, "lane_carrier_total": 0,
              "worker_pool_disabled_reason": "unknown"}
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert pool_observation_status(legacy) == "unknown"
    assert pool_observation_status({**legacy, "worker_pool_disabled_reason": ""}) == "observed"
    assert _promotion_pool_disabled_from_snapshot(types.SimpleNamespace(drive_root=tmp_path, budget_drive_root="")) \
        == "unknown"
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    assert fact["free_worker_slots"] is None
    assert all(fact[key] is None for key in ("worker_total", "acting_worker_count", "reaping_count", "lane_carrier_total", "memory_admission"))


# ---- NIT: the census carries only the waits that still wait ---------------------------------------
