"""Atomic companion status and event-loop heartbeat."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from platform_support import machine_architecture, path_is_link_or_reparse


STATUS_SCHEMA = 2
STATUS_NAME = "status.json"
HEARTBEAT_INTERVAL_SEC = 10.0

_STATES = {
    "starting",
    "waiting_owner",
    "waiting_token",
    "reconnecting",
    "verifying",
    "syncing_menu",
    "ready",
    "degraded",
    "disabled",
    "blocked",
    "reconciling",
    "stopping",
    "stopped",
    "rollback_pending",
    "error",
}
_SECURITY_COUNTERS = {
    "auth_success",
    "auth_rejected",
    "auth_rate_limited",
    "auth_busy",
    "active_sessions",
    "active_websockets",
}


class RuntimeStatusError(RuntimeError):
    pass


def _bounded(value: Any, maximum: int) -> str:
    return str(value or "").strip().replace("\r", " ").replace("\n", " ")[:maximum]


class RuntimeStatus:
    def __init__(
        self,
        state_dir: Path,
        *,
        cloudflared_version: str,
        metrics: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        self._state_dir = Path(state_dir).resolve()
        self._path = self._state_dir / STATUS_NAME
        try:
            self._path.relative_to(self._state_dir)
        except ValueError as exc:
            raise RuntimeStatusError("Status path escaped private skill state.") from exc
        if path_is_link_or_reparse(self._path):
            raise RuntimeStatusError("Status path is an unsafe link.")
        self._cloudflared_version = _bounded(cloudflared_version, 64)
        self._metrics = metrics
        self._instance_id = uuid.uuid4().hex
        self._state = "starting"
        self._reason_code = "initializing"
        self._message = "Starting Telegram Mini App companion."
        self._public_url = ""
        self._attempt = 0
        self._next_retry_at_epoch = 0
        self._last_ready_at_epoch = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def reason_code(self) -> str:
        return self._reason_code

    def set_metrics_provider(self, provider: Callable[[], dict[str, int]] | None) -> None:
        self._metrics = provider

    def transition(
        self,
        state: str,
        message: str,
        *,
        reason_code: str,
        public_url: str = "",
        attempt: int = 0,
        next_retry_at_epoch: int = 0,
    ) -> None:
        state = _bounded(state, 32)
        if state not in _STATES:
            raise RuntimeStatusError("Companion attempted an unknown status transition.")
        self._state = state
        self._reason_code = _bounded(reason_code, 64) or "unknown"
        self._message = _bounded(message, 300)
        self._public_url = _bounded(public_url, 2048) if state == "ready" else ""
        self._attempt = max(0, min(int(attempt), 1_000_000))
        self._next_retry_at_epoch = max(0, int(next_retry_at_epoch))
        if state == "ready":
            self._last_ready_at_epoch = int(time.time())
        self.publish()

    def _security_snapshot(self) -> dict[str, int]:
        if self._metrics is None:
            return {}
        try:
            values = self._metrics()
        except Exception:
            return {}
        if not isinstance(values, dict):
            return {}
        result: dict[str, int] = {}
        for key in sorted(_SECURITY_COUNTERS):
            try:
                result[key] = max(0, min(int(values.get(key) or 0), 2_147_483_647))
            except (TypeError, ValueError):
                continue
        return result

    def publish(self) -> None:
        now = int(time.time())
        payload: dict[str, Any] = {
            "schema": STATUS_SCHEMA,
            "state": self._state,
            "reason_code": self._reason_code,
            "message": self._message,
            "cloudflared_version": self._cloudflared_version,
            "pid": os.getpid(),
            "instance_id": self._instance_id,
            "updated_at_epoch": now,
            "last_ready_at_epoch": self._last_ready_at_epoch,
            "attempt": self._attempt,
            "next_retry_at_epoch": self._next_retry_at_epoch,
            "platform": f"{platform.system().lower()}-{machine_architecture()}",
        }
        if self._public_url:
            payload["public_url"] = self._public_url
        security = self._security_snapshot()
        if security:
            payload["security"] = security
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if len(encoded.encode("utf-8")) > 16_384:
            raise RuntimeStatusError("Companion status is unexpectedly large.")
        tmp = self._path.with_name(f".{self._path.name}.tmp-{os.getpid()}")
        try:
            tmp.write_text(encoded, encoding="utf-8")
            with contextlib.suppress(OSError):
                tmp.chmod(0o600)
            os.replace(tmp, self._path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise RuntimeStatusError("Could not persist companion status.") from exc

    async def heartbeat(
        self,
        stop_event: asyncio.Event,
        *,
        failure_event: asyncio.Event | None = None,
    ) -> None:
        try:
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=HEARTBEAT_INTERVAL_SEC)
                except asyncio.TimeoutError:
                    self.publish()
        except Exception:
            if failure_event is not None:
                failure_event.set()
            raise


__all__ = [
    "HEARTBEAT_INTERVAL_SEC",
    "RuntimeStatus",
    "RuntimeStatusError",
    "STATUS_NAME",
    "STATUS_SCHEMA",
]
