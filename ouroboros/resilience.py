"""
Ouroboros — Resilience: Circuit Breakers.

Graceful degradation for external service dependencies.
Prevents cascading failures when OpenRouter, Drive, or GitHub are down.

Bible alignment:
  P0 (Agency): An agent that crashes on external failure has no agency.
  P5 (Minimalism): 80 lines, stdlib only, no dependencies.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker for external service calls.

    States:
        CLOSED  — Normal operation. Calls pass through.
        OPEN    — Service is failing. Calls are short-circuited.
        HALF_OPEN — Cooldown expired. One test call is allowed.

    After `failure_threshold` consecutive failures, the circuit opens
    for `cooldown_sec` seconds. After cooldown, one test call is allowed.
    If it succeeds, the circuit closes. If it fails, the circuit re-opens.

    Thread-safe via a simple lock.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_sec: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec

        self._lock = threading.Lock()
        self._failures = 0
        self._state = "CLOSED"
        self._opened_at = 0.0
        self._last_failure_msg = ""

    @property
    def state(self) -> str:
        with self._lock:
            # Check if OPEN should transition to HALF_OPEN
            if self._state == "OPEN":
                if time.time() - self._opened_at > self.cooldown_sec:
                    self._state = "HALF_OPEN"
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == "OPEN"

    def allow_call(self) -> bool:
        """Check if a call should be attempted."""
        state = self.state  # Property handles OPEN → HALF_OPEN transition
        return state != "OPEN"

    def record_success(self) -> None:
        """Record a successful call. Closes the circuit."""
        with self._lock:
            if self._failures > 0 or self._state != "CLOSED":
                log.info("Circuit '%s' recovered (was %s, %d failures)",
                         self.name, self._state, self._failures)
            self._failures = 0
            self._state = "CLOSED"
            self._last_failure_msg = ""

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call. May open the circuit."""
        with self._lock:
            self._failures += 1
            self._last_failure_msg = str(error)[:200] if error else ""

            if self._failures >= self.failure_threshold:
                if self._state != "OPEN":
                    log.warning(
                        "Circuit '%s' OPENED after %d failures: %s",
                        self.name, self._failures, self._last_failure_msg,
                    )
                self._state = "OPEN"
                self._opened_at = time.time()

    def status_dict(self) -> Dict[str, Any]:
        """Return status for health invariant display."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "failures": self._failures,
                "threshold": self.failure_threshold,
                "last_failure": self._last_failure_msg,
                "cooldown_remaining": max(
                    0,
                    self.cooldown_sec - (time.time() - self._opened_at)
                ) if self._state == "OPEN" else 0,
            }


# ---------------------------------------------------------------------------
# Global circuit breakers for Ouroboros external services
# ---------------------------------------------------------------------------

_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(
    name: str,
    failure_threshold: int = 3,
    cooldown_sec: float = 60.0,
) -> CircuitBreaker:
    """Get or create a named circuit breaker (singleton per name)."""
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                cooldown_sec=cooldown_sec,
            )
        return _breakers[name]


def all_breaker_statuses() -> list:
    """Return status of all circuit breakers for health monitoring."""
    with _breakers_lock:
        return [b.status_dict() for b in _breakers.values()]


def format_breakers_for_health() -> str:
    """One-liner for health invariants section.

    Returns empty string if all breakers are closed (don't clutter context).
    Only surfaces problems.
    """
    statuses = all_breaker_statuses()
    open_breakers = [s for s in statuses if s["state"] != "CLOSED"]

    if not open_breakers:
        return ""  # All good — don't add noise

    lines = []
    for s in open_breakers:
        if s["state"] == "OPEN":
            lines.append(
                f"WARNING: SERVICE DOWN — '{s['name']}' circuit OPEN "
                f"({s['failures']} failures, "
                f"retry in {s['cooldown_remaining']:.0f}s): "
                f"{s['last_failure']}"
            )
        elif s["state"] == "HALF_OPEN":
            lines.append(
                f"INFO: SERVICE RECOVERING — '{s['name']}' testing recovery"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-configured breakers for common services
# ---------------------------------------------------------------------------

def openrouter_breaker() -> CircuitBreaker:
    """Circuit breaker for OpenRouter API."""
    return get_breaker("openrouter", failure_threshold=3, cooldown_sec=60.0)


def drive_breaker() -> CircuitBreaker:
    """Circuit breaker for Google Drive FUSE operations."""
    return get_breaker("google_drive", failure_threshold=5, cooldown_sec=30.0)


def github_breaker() -> CircuitBreaker:
    """Circuit breaker for GitHub API."""
    return get_breaker("github_api", failure_threshold=3, cooldown_sec=120.0)


def playwright_breaker() -> CircuitBreaker:
    """Circuit breaker for Playwright browser."""
    return get_breaker("playwright", failure_threshold=2, cooldown_sec=90.0)
