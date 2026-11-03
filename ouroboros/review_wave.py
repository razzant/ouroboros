"""One wave's author detach latch and physical-worker capacity.

Logical drains and free rejoins never own permits. The first observed detach is
shared even after the author consumes the mailbox event that caused it.
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager

_CURRENT = contextvars.ContextVar("review_wave", default=None)


class ReviewWave:
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.running = 0
        self.entries = {}
        self.detached = False
        self.reason = ""
        self.operation = None
        self.condition = threading.Condition(threading.RLock())

    def _inherit_detach(self, entry):
        entry.author_left_for = self.reason
        if self.reason:
            return
        from ouroboros.owner_pause import episode_armed, launch_lock
        from ouroboros.review_pause import _author_coordinates

        root, task, _ = _author_coordinates(self.operation.wait)
        with launch_lock(root, task):
            recovered = not entry.started_at and any(entry.retry_state.get(key) for key in
                                                    ("pending_invocation_id", "delegated_run_id"))
            if not recovered and not entry.event.is_set() and not episode_armed(entry.episode):
                entry.pause_before_launch = True
                if isinstance(entry.episode, dict):
                    entry.episode["pause_before_launch"] = True

    def register(self, entry):
        with self.condition:
            self.entries[id(entry)] = entry
            if self.detached:
                self._inherit_detach(entry)
            self.condition.notify_all()

    def observe_detach(self, operation, observer):
        with self.condition:
            if not self.detached and observer(operation, list(self.entries.values())):
                self.detached, self.operation = True, operation
                self.reason = next((getattr(entry, "author_left_for", "") for entry in self.entries.values()), "")
                self.condition.notify_all()
            return self.detached

    @contextmanager
    def worker(self, entry):
        """Hold capacity through durable settlement, including after author detach."""
        with self.condition:
            while self.capacity is not None and self.running >= self.capacity and not entry.pause_before_launch:
                self.condition.wait()
            admitted = not entry.pause_before_launch
            if admitted:
                self.running += 1
        try:
            yield admitted
        finally:
            with self.condition:
                # A failed durable write cannot publish ready or free its custody.
                if admitted and entry.event.is_set():
                    self.running -= 1
                self.condition.notify_all()


@contextmanager
def review_wave_scope(capacity=None):
    wave = _CURRENT.get()
    if wave is None:
        wave = ReviewWave(capacity)
    elif capacity is not None and wave.capacity is None:
        wave.capacity = capacity
    token = _CURRENT.set(wave)
    try:
        yield wave
    finally:
        _CURRENT.reset(token)


def current_review_wave():
    return _CURRENT.get()
