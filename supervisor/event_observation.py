"""The supervisor event bus: one Manager-backed queue, two lanes, one consumer.

The bus is the ordinary ``queue.Queue`` the process-lifetime manager serves,
with these facts kept under its own mutex and nothing else:

* **Two lanes.** Every event kind rides the lane its ``EventDisposition``
  declares (``supervisor/event_taxonomy.py``): ``control`` for chat sends,
  task start/stop, admission and controls; ``telemetry`` for heartbeats, usage
  rows and ``log_event`` frames. ``get`` serves the oldest control event while
  any exists, else the oldest telemetry event, FIFO inside each lane, so a
  backlog of worker telemetry can never sit in front of an answer or a control
  (owner decision, 14 Oct 2026). Producers keep one ``put``; the single
  supervisor consumer keeps one bounded ``get`` loop. Nothing is re-put.
* **Pending idle inputs.** The bounded drain cannot conclude idleness while a
  fact that may change that decision is still queued, in either lane. This
  index counts only pending envelopes; it neither folds activity nor grants a
  lease. The consumer finishes dispatch before checking timeouts. A later
  enqueue is ordinary concurrent activity, not a claim about the earlier
  snapshot.
* **Outstanding deliveries.** A ``send_message`` carrying a ``delivery_id`` is
  outstanding from its put until the consumer releases it after the handler
  returned or raised (``ouroboros/server_liveness.py``). The replay duty asks
  this membership before it enqueues a copy (``put_unless_outstanding``), so a
  queued or in-flight original is never duplicated and never counted as a
  retry. Volatile: a new server generation starts empty; the durable outbox
  stays the authority on what is owed.
* **Lane depth and age.** Each envelope keeps the manager process's own
  monotonic stamp of its put; ``lane_facts`` answers per lane how many wait
  and how long the oldest has (the per-minute ``supervisor_load`` row reads
  it), one process and one clock. A re-put envelope ages from its new put.
"""
from __future__ import annotations

import logging
import queue
import time
from collections import Counter, deque
from multiprocessing.managers import SyncManager
from typing import Any

from supervisor.event_taxonomy import CONTROL_LANE, TELEMETRY_LANE, disposition_for

log = logging.getLogger(__name__)


def idle_input_addresses(event: Any) -> frozenset[str]:
    """Routing addresses of an existing idle-input event, without interpreting work.

    ``log_event`` is telemetry, even when its payload names a lifecycle event:
    its handler does not fold activity. Heartbeats and host narration likewise
    never change the idle inputs. Exact attempts remain the handlers' authority;
    a stale frame can delay observation only until it leaves this queue.
    """
    if not isinstance(event, dict):
        return frozenset()
    kind = str(event.get("type") or "")
    disposition = disposition_for(kind)
    if disposition is None or not disposition.idle_input:
        return frozenset()
    if kind == "send_message" and (not event.get("is_progress") or event.get("host_narration")):
        return frozenset()
    return frozenset(str(event[key]) for key in ("task_id", "parent_task_id", "root_task_id")
                     if event.get(key))


def event_lane(event: Any) -> str:
    """The lane one event rides, read from its declared disposition; undeclared kinds are control."""
    kind = str(event.get("type") or "") if isinstance(event, dict) else ""
    disposition = disposition_for(kind)
    return TELEMETRY_LANE if disposition is not None and disposition.lane == TELEMETRY_LANE else CONTROL_LANE


def outstanding_delivery_id(event: Any) -> str:
    """The delivery identity a queued send carries ('' for anything else)."""
    if isinstance(event, dict) and str(event.get("type") or "") == "send_message":
        return str(event.get("delivery_id") or "")
    return ""


class ObservedEventQueue(queue.Queue):
    """The manager's ordinary FIFO with two lanes and two indexes, not another bus."""

    def _init(self, maxsize: int) -> None:
        self._lanes: dict[str, deque] = {CONTROL_LANE: deque(), TELEMETRY_LANE: deque()}
        self._pending: Counter[str] = Counter()
        self._outstanding: Counter[str] = Counter()

    def _qsize(self) -> int:
        return sum(len(lane) for lane in self._lanes.values())

    def _put(self, item: Any) -> None:
        addresses = idle_input_addresses(item)
        # Retain the addresses by value: an in-process producer may keep or
        # mutate its original dict, while the manager transport copies it.
        self._lanes[event_lane(item)].append((item, addresses, time.monotonic()))
        self._pending.update(addresses)
        delivery_id = outstanding_delivery_id(item)
        if delivery_id:
            self._outstanding[delivery_id] += 1

    def _get(self) -> Any:
        lane = self._lanes[CONTROL_LANE] or self._lanes[TELEMETRY_LANE]
        item, addresses, _put_at = lane.popleft()
        for address in addresses:
            self._pending[address] -= 1
            if not self._pending[address]:
                del self._pending[address]
        return item

    def pending_idle_inputs(self) -> set[str]:
        """One atomic snapshot of addresses with an unconsumed idle input, both lanes."""
        with self.mutex:
            return set(self._pending)

    def lane_facts(self) -> dict[str, dict[str, Any]]:
        """Per lane: the envelopes waiting and the age in seconds of the oldest (None when empty)."""
        with self.mutex:
            now = time.monotonic()
            return {name: {"depth": len(lane), "oldest_age_sec": round(now - lane[0][2], 1) if lane else None}
                    for name, lane in self._lanes.items()}

    def release_delivery(self, delivery_id: str) -> None:
        """The consumer's receipt: one queued send of this id left the handler."""
        delivery_id = str(delivery_id or "")
        with self.mutex:
            if self._outstanding[delivery_id] > 1:
                self._outstanding[delivery_id] -= 1
            else:
                self._outstanding.pop(delivery_id, None)

    def delivery_outstanding(self, delivery_id: str) -> bool:
        with self.mutex:
            return bool(self._outstanding.get(str(delivery_id or "")))

    def put_unless_outstanding(self, event: Any) -> bool:
        """Enqueue atomically unless a send of the same ``delivery_id`` is queued or in flight.

        The ``put_nowait`` path of ``queue.Queue`` under the same condition, with
        the membership check inside it: a check followed by an ordinary put would
        still race the consumer. ``queue.Full`` propagates like ``put_nowait``.
        """
        with self.not_full:
            delivery_id = outstanding_delivery_id(event)
            if delivery_id and self._outstanding.get(delivery_id):
                return False
            if 0 < self.maxsize <= self._qsize():
                raise queue.Full
            self._put(event)
            self.unfinished_tasks += 1
            self.not_empty.notify()
            return True


class EventQueueManager(SyncManager):
    """Same process-lifetime manager, with the supervisor's queue factory."""


EventQueueManager.register("Queue", ObservedEventQueue)


def pending_idle_observations(event_queue: Any) -> set[str] | None:
    """Read outside Q; missing/broken observation is unknown, never empty.

    No actor clock or progress fact is advanced. The domain timeout owner uses
    this only to defer its idle inference, never deadline/ceiling/owner controls.
    """
    try:
        pending = event_queue.pending_idle_inputs()
        if not isinstance(pending, (set, frozenset)) or any(not isinstance(v, str) for v in pending):
            raise ValueError("invalid supervisor event observation")
        return set(pending)
    except Exception:
        log.warning("Supervisor idle inputs are unobservable; idle decisions deferred", exc_info=True)
        return None


def delivery_outstanding(event_queue: Any, delivery_id: str) -> bool | None:
    """Is a send of this id queued or being handled? ``None`` = unobservable.

    Mirrors ``pending_idle_observations``: a bus without the membership method
    or a broken proxy answers unknown, and the caller treats unknown as "maybe
    still queued" (no copy, no attempt counted), never as absence.
    """
    try:
        outstanding = event_queue.delivery_outstanding(str(delivery_id or ""))
        if not isinstance(outstanding, bool):
            raise ValueError("invalid supervisor delivery observation")
        return outstanding
    except Exception:
        log.warning("Supervisor delivery membership is unobservable for %s", delivery_id, exc_info=True)
        return None


def lane_observations(event_queue: Any) -> dict[str, dict[str, Any]] | None:
    """``lane_facts`` read outside Q; a bus that cannot answer is unknown (None), never empty."""
    try:
        facts = event_queue.lane_facts()
        if not isinstance(facts, dict):
            raise ValueError("invalid supervisor lane observation")
        return facts
    except Exception:
        log.debug("Supervisor lane depths are unobservable", exc_info=True)
        return None
