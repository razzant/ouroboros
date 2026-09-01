"""Gateway cancellation custody for the CyberGym executor.

Split out of ``cybergym_lifecycle.py`` so the lifecycle layer stays inside
its size-ratchet band.  ``_CustodyMixin`` holds the cancellation request and
the post-cancellation custody poll — the seam that decides when a gateway
task the launcher stopped waiting for is truly terminal — mixed into
``CyberGymExecutor`` and dispatched on ``self`` at runtime.  Everything here
is gateway HTTP plus checkpoint persistence; it never touches containers or
child processes.
"""
from __future__ import annotations

import pathlib
import time
import urllib.parse
from collections.abc import Mapping
from typing import Any

from devtools.benchmarks.cybergym.cybergym_adapter import _TERMINAL_GATEWAY_STATUSES
from devtools.benchmarks.cybergym.cybergym_docker import _write_json
from devtools.benchmarks.cybergym.cybergym_wire import (
    ExecutorFailure,
    GatewayTransportError,
    HttpStatusError,
    _CostGraceTracker,
    _cost_is_pending,
    _gateway_path,
    _response_status,
    _unwrap_http_json,
    _valid_cost_grace,
)


class _CustodyMixin:
    """Cancellation and terminal-custody methods for the CyberGym executor."""

    def _poll_gateway_custody(
        self,
        task_id: str,
        checkpoint: pathlib.Path,
        *,
        cancel_response: Mapping[str, Any] | None,
        cancel_status_code: int | None = None,
        custody_seconds: float,
    ) -> Mapping[str, Any]:
        """Poll an already admitted task until a terminal custody frame.

        This helper is shared by the normal cancellation response and the
        gateway's 503/404 cancellation races.  A ``completed`` frame with
        pending cost accounting is not terminal for this adapter unless the
        bounded abandoned-residue grace releases it: the outer campaign
        ledger must receive a final/upper-bound frame, never an intermediate
        cost snapshot.
        """

        deadline = time.monotonic() + custody_seconds
        cost_grace = _CostGraceTracker()
        cancel_frame = dict(cancel_response) if isinstance(cancel_response, Mapping) else None
        latest: Mapping[str, Any] = cancel_response or {}
        status_url = _gateway_path(
            self.config.ouroboros_url,
            "/api/tasks/" + urllib.parse.quote(task_id, safe=""),
        )
        while time.monotonic() < deadline:
            try:
                latest = _unwrap_http_json(
                    self.config.http_runner(
                        "GET", status_url, timeout=30
                    ),
                    operation="Ouroboros cancellation custody status",
                )
                returned_id = str(latest.get("task_id") or "").strip()
                if returned_id and returned_id != task_id:
                    raise ExecutorFailure("cancellation status belongs to a different task")
                status = _response_status(latest)
                if status == "completed" and _cost_is_pending(latest):
                    latest = (
                        cost_grace.accept(latest, now=time.monotonic(), wall_now=time.time())
                        or latest
                    )
                frame: dict[str, Any] = {
                    "gateway_task_id": task_id,
                    "status": status or "cancel_pending",
                    "result": dict(latest),
                }
                if cancel_status_code is not None:
                    frame["cancel_status_code"] = cancel_status_code
                if cancel_frame is not None:
                    frame["cancel_response"] = cancel_frame
                _write_json(checkpoint, frame)
                if status in _TERMINAL_GATEWAY_STATUSES and not (
                    status == "completed"
                    and _cost_is_pending(latest)
                    and _valid_cost_grace(latest) is None
                ):
                    self._terminalize_gateway_attempt(task_id)
                    return latest
            except ExecutorFailure:
                # HTTP/auth/transport failures remain typed failures and keep
                # the attempt registered for manual custody recovery.
                raise
            except Exception as exc:
                frame = {
                    "gateway_task_id": task_id,
                    "status": "cancel_poll_error",
                    "cancel_error": type(exc).__name__,
                }
                if cancel_status_code is not None:
                    frame["cancel_status_code"] = cancel_status_code
                if cancel_frame is not None:
                    frame["cancel_response"] = cancel_frame
                _write_json(checkpoint, frame)
            self.config.sleep(max(0.5, float(self.config.poll_interval_sec)))
        raise ExecutorFailure("Ouroboros task cancellation custody did not settle")

    def _cancel_gateway_task(
        self, task_id: str, checkpoint: pathlib.Path
    ) -> Mapping[str, Any]:
        """Request cancellation and retain custody until a terminal status.

        A caller-side polling deadline is not proof that the worker stopped.
        The cancel response and the subsequent short custody poll are written
        to the same checkpoint, so an operator can later inspect/reattach
        without making a duplicate paid attempt.  A 503 (durable cancel
        intent, asynchronous teardown) or a 404 (the task already left the
        active set — a long-terminal task still answers GET from its durable
        result) allows a GET-only recovery of the existing terminal task
        result.  Other HTTP statuses and transport failures are not converted
        into apparent task results.
        """
        cancel_url = _gateway_path(
            self.config.ouroboros_url,
            "/api/tasks/" + urllib.parse.quote(task_id, safe="") + "/cancel",
        )
        custody_seconds = min(
            180.0, max(30.0, float(self.config.poll_interval_sec) * 8.0 + 10.0)
        )
        try:
            cancel_response = _unwrap_http_json(
                self.config.http_runner(
                    "POST", cancel_url, body={}, timeout=30
                ),
                operation="Ouroboros task cancellation",
                accepted_statuses=(200, 202, 204),
            )
        except HttpStatusError as exc:
            _write_json(
                checkpoint,
                {
                    "gateway_task_id": task_id,
                    "status": "cancel_request_failed",
                    "cancel_error": type(exc).__name__,
                    "cancel_status_code": exc.status_code,
                },
            )
            if exc.status_code in (503, 404):
                # Only a later terminal GET can turn a cancellation race into
                # an adapter outcome; absent that frame we retain the
                # original custody block.
                return self._poll_gateway_custody(
                    task_id,
                    checkpoint,
                    cancel_response=None,
                    cancel_status_code=exc.status_code,
                    custody_seconds=custody_seconds,
                )
            raise ExecutorFailure("Ouroboros task cancellation request failed") from exc
        except GatewayTransportError as exc:
            _write_json(
                checkpoint,
                {
                    "gateway_task_id": task_id,
                    "status": "cancel_request_failed",
                    "cancel_error": type(exc).__name__,
                },
            )
            raise
        except Exception as exc:
            _write_json(
                checkpoint,
                {
                    "gateway_task_id": task_id,
                    "status": "cancel_request_failed",
                    "cancel_error": type(exc).__name__,
                },
            )
            raise ExecutorFailure("Ouroboros task cancellation request failed") from exc
        _write_json(
            checkpoint,
            {
                "gateway_task_id": task_id,
                "status": _response_status(cancel_response) or "cancel_requested",
                "cancel_response": dict(cancel_response),
            },
        )
        return self._poll_gateway_custody(
            task_id,
            checkpoint,
            cancel_response=cancel_response,
            custody_seconds=custody_seconds,
        )
