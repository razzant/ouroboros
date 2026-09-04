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
    GATEWAY_TRANSPORT_RETRY_BUDGET_SEC,
    FINALIZATION_GRACE_SEC,
    _CostGraceTracker,
    _cost_is_pending,
    _gateway_finalizing,
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
        # A frame that shows the worker finished and artifacts finalizing is a
        # paid result in flight, not a stuck cancellation: custody stays open
        # for it up to the finalization grace (once).
        finalization_deadline = time.monotonic() + max(custody_seconds, FINALIZATION_GRACE_SEC)
        finalization_extended = False
        transport_deadline: float | None = None
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
                if not finalization_extended and _gateway_finalizing(latest):
                    finalization_extended = True
                    deadline = max(deadline, finalization_deadline)
                    _write_json(checkpoint, {**frame, "custody_basis": "finalization_grace"})
                transport_deadline = None
            except (GatewayTransportError, HttpStatusError) as exc:
                if isinstance(exc, HttpStatusError) and exc.status_code != 503:
                    raise
                # Cancellation waves can starve both the cancel POST and the
                # follow-up GET on the same event loop.  Keep custody through
                # that transient outage, bounded by both the custody window
                # and the shared transport retry budget.
                now = time.monotonic()
                if transport_deadline is None:
                    transport_deadline = min(
                        deadline, now + GATEWAY_TRANSPORT_RETRY_BUDGET_SEC
                    )
                frame = {
                    "gateway_task_id": task_id,
                    "status": "cancel_poll_error",
                    "cancel_error": type(exc).__name__,
                }
                if isinstance(exc, HttpStatusError):
                    frame["cancel_poll_status_code"] = exc.status_code
                if cancel_status_code is not None:
                    frame["cancel_status_code"] = cancel_status_code
                if cancel_frame is not None:
                    frame["cancel_response"] = cancel_frame
                _write_json(checkpoint, frame)
                if now >= transport_deadline:
                    raise
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
        # The post-cancel custody poll must outlast a deadline-wave settle:
        # when a full 64-lane wave hits its 2 h deadline together, the isolate
        # takes minutes to settle each cancellation, and the previous ~34 s
        # bound wrote off 18 paid tasks in one wave as "custody did not
        # settle" even though every cancel had already landed.
        custody_seconds = min(
            600.0, max(300.0, float(self.config.poll_interval_sec) * 8.0 + 10.0)
        )
        cancel_response: Mapping[str, Any] | None = None
        transport_deadline: float | None = None
        while cancel_response is None:
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
                # Transient: the cancel intent usually lands server-side and
                # only the response is starved (an isolate event-loop stall),
                # and a duplicate cancel POST is idempotent.  Ride out the
                # stall within a bounded budget before writing the attempt
                # off; exhaustion keeps the original fail-closed behaviour.
                now = time.monotonic()
                if transport_deadline is None:
                    transport_deadline = now + GATEWAY_TRANSPORT_RETRY_BUDGET_SEC
                if now >= transport_deadline:
                    _write_json(
                        checkpoint,
                        {
                            "gateway_task_id": task_id,
                            "status": "cancel_request_failed",
                            "cancel_error": type(exc).__name__,
                        },
                    )
                    raise
                self.config.sleep(max(0.5, float(self.config.poll_interval_sec)))
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
