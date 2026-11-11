"""Physical admission of new project processes: the machine is the limit (pool-capacity D, owner Q1 = D).

``worker_pool_lifecycle.admissible_starts`` answers with this rule: a new project process starts while one more of
ours fits, ``available - R >= T * (B + 1)``, and the macOS kernel reports no critical pressure (warn refuses
nothing). ``available`` is ``host_memory.physical_memory`` (within this process's own cgroup limits), ``R`` the host
reserve (``config.HOST_MEMORY_RESERVE_FRACTION`` of the total), ``T`` the typical size of one of our workers (the mean
of the live workers' own heartbeat reports, both lanes; else this generation's last ``worker_ready`` size; else
unknown) and ``B`` the processes of both lanes still starting. With no project process acting or starting one start is
admitted even under pressure (the floor), within the owner's cap and readiness fence;
with measured memory but unknown ``T``, only that floor admits. Readiness parks since the last successful ``worker_ready`` stop growth only when they
outnumber live ready project processes (finiteness); an unknown reading admits what the owner's cap leaves when set
(``max(floor, room)``, ``room = cap - acting - idle - booting``), else one starting process at a time (beyond the
floor, both lanes' booting processes must finish), and says why (``memory_unmeasured: ...``); the owner's
``OUROBOROS_MAX_WORKERS``, when set, caps acting tasks and claims.

Reuse of an idle process, a warm return and a crash replacement never ask; the conversation lane's starts stay
outside (4A: spawn success). The machine is read at most once a pass, outside Q and the lifecycle serializer, only
when chosen work needs a new process (``dispatch_choice.run_pass``), and never kept.
Workers report RSS, never probe host memory. Pressure stops or cools nobody;
it refuses a new process. Dated admission facts are published with the pool's counts (``facts``); the queue census
hashes only whether starts are admitted and why. Disclosed residuals: growth after a start is not predicted; neighbours
on the host share memory without a common ledger; Windows Job and commit limits are not read; running out of memory
is not prevented, only starting a process into a measured shortage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from ouroboros.host_memory import MemorySnapshot, physical_memory
from ouroboros.utils import utc_now_iso


@dataclass(frozen=True)
class Reading:
    """One pass's reading of the machine: ``memory`` is None when unknown and ``unmeasured`` then says why."""

    memory: Optional[MemorySnapshot]
    unmeasured: str
    at: str


PROBE = Reading(None, "", "")  # before the pass reads: whether a start could be admitted at all
_READY: Dict[str, Any] = {"seq": 0, "generation": None, "rss_bytes": None}  # successful openings; the newest size
_FACTS: Dict[str, Any] = {}  # the last reading's facts, published with the pool's counts (written and read under Q)


def read() -> Reading:
    """The machine now. Called outside Q and the lifecycle serializer; never kept beyond its pass."""
    try:
        memory = physical_memory()
    except Exception as exc:  # the probe answers None for its own failures; anything else is unknown too
        return Reading(None, f"memory_unmeasured: {type(exc).__name__}", utc_now_iso())
    why = "" if memory is not None else "memory_unmeasured: the host memory probe gave no reading"
    return Reading(memory, why, utc_now_iso())


def note_ready(row: Dict[str, Any]) -> None:
    """Under Q: a slot opened on its own ``worker_ready`` row; its size is this generation's newest fresh size."""
    from supervisor import workspace_preparation

    _READY["seq"] += 1
    if type(row.get("rss_bytes")) is int and row["rss_bytes"] > 0:
        _READY.update(generation=workspace_preparation.current_generation(), rss_bytes=row["rss_bytes"])


def note_park(slot: Any) -> None:
    """Under Q: ``slot`` was parked for readiness after the openings counted so far."""
    slot.parked_after_ready = _READY["seq"]


def parked(slots: Iterable[Any]) -> int:
    """Slots parked for readiness since the last successful ``worker_ready`` (one marked by hand counts)."""
    return sum(getattr(slot, "readiness_exhausted", False)
               and getattr(slot, "parked_after_ready", _READY["seq"]) == _READY["seq"] for slot in slots)


def waits_on_its_process(meta: Any, now: Optional[float] = None) -> bool:
    """A running row that holds its process without acting: an accepted owner wait or warm sleep (``owner_wait`` in
    state ``waiting``) or a live ``in_slot_wait`` lease (``model_sleep.declare_in_slot_wait``). Every other row a
    busy slot holds acts, a launch claim whose row is not running yet included."""
    if not isinstance(meta, dict):
        return False
    wait = meta.get("owner_wait")
    if isinstance(wait, dict) and wait.get("state") == "waiting":
        return True
    leases, now = meta.get("active_operation_leases"), time.time() if now is None else now
    return isinstance(leases, dict) and any(
        isinstance(row, dict) and row.get("kind") == "in_slot_wait" and _number(row.get("until_ts")) > now
        for row in leases.values())


def acting(slot: Any, meta: Any, now: Optional[float] = None) -> bool:
    """A busy slot or projected launch acts unless its slot or running row says it waits on its process."""
    return bool(getattr(slot, "active_capacity", True) and not waits_on_its_process(meta, now))


def cold_sleeping(task: Any) -> bool:
    """A pending row paused for sleep holds no place at the scheduling soft cap."""
    return isinstance(task, dict) and isinstance(pause := task.get("_budget_pause"), dict) and pause.get("reason") == "sleep"


def acting_children(root_task_id: str, running: Iterable[Any]) -> int:
    """The acting subagents of one root (``OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT`` counts these): a child that
    waits on its process or sleeps without one does not act; a row chosen in this pass does."""
    from supervisor import workers

    now = time.time()
    return sum(1 for meta in running if isinstance(meta, dict) and isinstance(task := meta.get("task"), dict)
               and task.get("delegation_role") == "subagent" and str(task.get("root_task_id") or "") == root_task_id
               and root_task_id and acting(workers.WORKERS.get(meta.get("worker_id")), meta, now))


def typical_size(slots: Iterable[Any]) -> Tuple[Optional[int], str]:
    """``T`` and its basis: the mean of the live workers' own reports (both lanes, from their tasks' heartbeats),
    else this generation's last ``worker_ready`` size, else None (only the floor admits)."""
    from supervisor import workspace_preparation
    from supervisor.worker_pool_lifecycle import is_worker_slot  # an inline executor's report is the server's size

    sizes = [report["rss_bytes"] for slot in slots if isinstance(report := getattr(slot, "rss", None), dict)
             and type(report.get("rss_bytes")) is int and is_worker_slot(slot) and slot.proc.is_alive()]
    if sizes:
        return sum(sizes) // len(sizes), "heartbeat_mean"
    if _READY["rss_bytes"] and _READY["generation"] is workspace_preparation.current_generation():
        return _READY["rss_bytes"], "worker_ready"
    return None, "unknown"


def admit(reading: Optional[Reading], counts: Dict[str, Any], slots: Iterable[Any], *, parked_now: int) -> int:
    """Under Q: how many new project processes may start, from ``counts`` (``pool_slot_counts``) and the pass's
    ``reading`` (None: not read, nothing starts; ``PROBE``: 1 when a reading could admit one). A real reading
    records its dated facts for publication; finiteness publishes its refusal even without a memory read."""
    from supervisor.worker_pool_lifecycle import is_project_slot, is_worker_slot, worker_ready

    slots = list(slots)
    cap, acting, booting = counts["max_workers"], counts["acting_worker_count"], counts["booting_count"]
    room = None if cap is None else cap - acting - counts["assignable_idle_workers"] - booting
    if parked_now and parked_now > sum(is_project_slot(slot) and is_worker_slot(slot) and worker_ready(slot) for slot in slots):
        _FACTS.clear()
        _FACTS.update(read_at=utc_now_iso(), status="not_read", starts_admitted=0, reason="worker_readiness_finiteness")
        return 0
    if _FACTS.get("reason") == "worker_readiness_finiteness":
        _FACTS.clear()
    if room is not None and room <= 0 or reading is None:
        return 0  # the owner's cap or a pass that has not read the machine
    if reading is PROBE:
        return 1
    floor = int(acting == 0 and booting == 0)
    starting = sum(1 for slot in slots if getattr(slot, "booting", False))
    typical, basis = typical_size(slots)
    memory = reading.memory
    if memory is None:
        starts, why = max(floor, room if room is not None else int(starting == 0)), reading.unmeasured
    elif memory.pressure:
        starts, why = floor, "memory_pressure_critical"
    elif not typical:
        starts, why = floor, "worker_size_unknown"
    else:
        fits = memory.room_bytes // typical - starting
        starts, why = max(floor, fits), "memory_room" if fits > 0 else "memory_full"
    starts = starts if room is None else min(starts, room)
    _FACTS.clear()
    _FACTS.update({
        "read_at": reading.at, "status": "unknown" if memory is None else "measured",
        "total_bytes": memory and memory.total_bytes, "available_bytes": memory and memory.available_bytes,
        "reserve_bytes": memory and memory.reserve_bytes,
        "pressure": memory and ("critical" if memory.pressure else "not_critical"),
        "typical_worker_bytes": typical, "typical_basis": basis, "starting_processes": starting,
        "starts_admitted": starts, "reason": why})
    return starts


def facts() -> Dict[str, Any]:
    """Under Q: dated admission facts, including finiteness without a memory read (``{}`` before the first)."""
    return dict(_FACTS)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
