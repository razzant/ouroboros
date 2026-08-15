"""Owner-only HTTP surface over the remote connection store (RWS v2, D6).

Transport-thin handlers over ``ouroboros/connection_store.py`` (the owner-state
authority for path/schema/lock/atomic writer) plus the live-state projection of
the remote workspace broker. Durable trust/lifecycle mutations go through the
store module; this module never re-implements its persistence.

Every route here lives under ``/api/owner/connections`` and is owner-gated by
``server_auth.NetworkAuthGate`` even on loopback (see
``server_auth.is_owner_connections_path``).
"""

from __future__ import annotations

import asyncio
import inspect
import pathlib
import threading
import time
from functools import partial
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.connection_store import (
    add_connection,
    get_connection,
    list_connections,
    pin_connection_host,
    record_bootstrap,
    retire_connection,
    retrust_connection,
)
from ouroboros.gateway._helpers import wire_error_code
from ouroboros.gateway.contracts import CONNECTION_STATUSES
from ouroboros.remote_contracts import CONTRACT_SET_VERSION
from ouroboros.remote_refusal_actions import (
    ACTION_BOOTSTRAP,
    ACTION_CANCEL_TASKS,
    ACTION_CONFIRM_RETRUST,
    ACTION_READMIT_PROJECT,
    ACTION_RELOAD_CONNECTIONS,
    ACTION_RESTART,
    ACTION_RETRUST,
    ACTION_TEST,
    REFUSAL_ACTIONS,
    connection_blocker,
)
# THE wire/log scrubber for this surface. It is the same function the transport
# runs over every envelope it emits, so a diagnostic cannot be scrubbed one way on
# the way out of the target and another way on the way to the browser.
from ouroboros.workspace_diagnostics import sanitize_execution_text as _sanitize_live_text

_LIVE_TEXT_LIMIT = 4000
_LIVE_COLLECTION_LIMIT = 32
_CONNECTION_HEALTH_FRESH_SEC = 300.0
_RUNTIME_EVIDENCE_LOCK = threading.RLock()
_RUNTIME_EVIDENCE: dict[tuple[str, str], dict[str, Any]] = {}

# The wire code for the whole "this build/host cannot serve a remote request"
# class. The CLI maps it (with the broker's own typed codes) to exit 4, so an
# owner script can tell "not serviceable here" from "refused" without parsing
# prose. Kept as a NAME rather than a literal because it crosses three surfaces.
REMOTE_TRANSPORT_UNAVAILABLE = "remote_transport_unavailable"


def _connection_store_path(request: Request) -> pathlib.Path:
    from ouroboros.connection_store import connection_store_path

    state = getattr(request.app, "state", None)
    injected = getattr(state, "remote_connections_path", None) if state is not None else None
    return pathlib.Path(injected) if injected else connection_store_path()


def _record_runtime_health(
    path: pathlib.Path,
    connection_id: str,
    result: dict[str, Any],
) -> None:
    """Record the process-local half of the evidence: WHEN health was last observed.

    Freshness is the only part that genuinely cannot be persisted — "the target
    answered within the last few minutes" is a claim about this run, and a monotonic
    clock does not survive a restart. Bootstrap compatibility is the durable half and
    lives in the owner store (``connection_store.record_bootstrap``), which is why
    this function no longer refuses to remember health for a connection it has not
    seen bootstrapped in THIS process: that guard is what made a restart leave the
    New Project picker permanently empty, since the flag it keyed on could only exist
    if a bootstrap had happened since the last start.
    """

    status = str(result.get("status") or "").strip().lower()
    if status not in {"ready", "degraded", "disconnected", "unknown"}:
        return
    key = (
        str(pathlib.Path(path).resolve(strict=False)),
        str(connection_id),
    )
    with _RUNTIME_EVIDENCE_LOCK:
        _RUNTIME_EVIDENCE[key] = {
            "status": status,
            "health_checked_monotonic": time.monotonic(),
        }


def _runtime_evidence_fields(
    path: pathlib.Path,
    connection_id: str,
    row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The Home-owned admission evidence: durable compatibility + live freshness.

    ``row`` is the connection's own store record when the caller already holds it;
    otherwise it is read here. The two halves stay structurally separate — a broker
    status projection can never manufacture either (``api_connections_list`` layers
    this LAST for exactly that reason).
    """
    if row is None:
        row = get_connection(str(connection_id), path) or {}
    compatible = bool(row.get("bootstrapped_at")) and row.get("lifecycle", "active") == "active"
    if not compatible:
        return {
            "bootstrap_compatible": False,
            "health_fresh": False,
            "execd_outdated": False,
            "required_contract_set": CONTRACT_SET_VERSION,
            "bootstrap_contract_set": int(row.get("bootstrap_contract_set") or 0),
        }
    with _RUNTIME_EVIDENCE_LOCK:
        live = dict(
            _RUNTIME_EVIDENCE.get(
                (
                    str(pathlib.Path(path).resolve(strict=False)),
                    str(connection_id),
                ),
                {},
            )
        )
    checked = float(live.get("health_checked_monotonic") or 0.0)
    fresh = checked > 0 and time.monotonic() - checked <= _CONNECTION_HEALTH_FRESH_SEC
    # The bootstrap claim, CHECKED. `bootstrapped_at` said a bootstrap happened; it
    # could not say whether the executor it installed can still talk to this build —
    # which is the fact the owner actually needed, and the one whose absence let a
    # target running an execd from a different contract set look perfectly healthy in
    # Connections while every remote tool call failed. Compared against the CONTRACT
    # SET rather than the release id on purpose (`connection_store.record_bootstrap`):
    # most releases change no shared contract, so release comparison would mark every
    # connection stale after every upgrade.
    #
    # It rides beside `bootstrap_compatible` instead of folding into it, because
    # "never bootstrapped" and "bootstrapped against an older contract set" are
    # different sentences with the same next step, and a surface that says only
    # "incompatible" cannot tell the owner which one they are looking at.
    outdated = int(row.get("bootstrap_contract_set") or 0) != CONTRACT_SET_VERSION
    result: dict[str, Any] = {
        "bootstrap_compatible": True,
        "health_fresh": fresh,
        "execd_outdated": outdated,
        "required_contract_set": CONTRACT_SET_VERSION,
        "bootstrap_contract_set": int(row.get("bootstrap_contract_set") or 0),
    }
    if fresh and live.get("status"):
        result["status"] = str(live["status"])
    if row.get("bootstrap_build"):
        result["build"] = str(row["bootstrap_build"])
    return result


def _drop_runtime_evidence(path: pathlib.Path, connection_id: str) -> None:
    with _RUNTIME_EVIDENCE_LOCK:
        _RUNTIME_EVIDENCE.pop(
            (
                str(pathlib.Path(path).resolve(strict=False)),
                str(connection_id),
            ),
            None,
        )


def _remote_service(request: Request) -> Any:
    state = getattr(request.app, "state", None)
    injected = (
        getattr(state, "remote_workspace_service", None)
        if state is not None
        else None
    )
    if injected is not None:
        return injected
    try:
        from ouroboros.remote_workspace import get_remote_workspace_service

        return get_remote_workspace_service()
    except (ImportError, RuntimeError):
        return None


def _service_missing_response(connection_id: str = "") -> JSONResponse:
    """No broker is running in THIS server, so the honest hint is a restart.

    The transport module is part of the build, so there is no longer a "not in this
    build" arm to take; `REMOTE_TRANSPORT_UNAVAILABLE` survives as the CLI-facing
    code for the class (`cli_connections`), not as a branch here.
    """
    return _typed_error(
        "remote workspace service is unavailable", 503,
        error_code="remote_service_unavailable", action=ACTION_RESTART,
        connection_id=connection_id,
    )


async def _service_call(service: Any, method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    fn = getattr(service, method, None)
    if not callable(fn):
        return {
            "ok": False,
            "error": "remote workspace service is unavailable",
            "error_code": "remote_service_unavailable",
            "phase": "connect",
            "action": ACTION_RESTART,
        }
    try:
        result = await asyncio.to_thread(fn, *args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        result = _public_live_fields(exc, default_phase="connect")
        return {
            **result,
            "ok": False,
            "error": _sanitize_live_text(f"{type(exc).__name__}: {exc}")[:2000],
        }
    if isinstance(result, dict):
        # The ONE door every service answer enters through, so the wire spelling of
        # its typed code is settled here rather than at each response site.
        row = dict(result)
        if row.get("error_code") is not None:
            row["error_code"] = wire_error_code(row["error_code"])
        return row
    return {
        "ok": False,
        "error": "remote workspace service returned an invalid response",
        "error_code": "remote_service_invalid_response",
        "phase": "connect",
        "action": ACTION_RESTART,
    }


def _typed_error(
    error: str,
    status_code: int,
    *,
    error_code: str,
    phase: str = "",
    action: str = "",
    connection_id: str = "",
) -> JSONResponse:
    payload = {
        "ok": False,
        "error": str(error or error_code),
        "error_code": wire_error_code(error_code),
    }
    if phase:
        payload["phase"] = phase
    if action:
        payload["action"] = action
    if connection_id:
        payload["connection_id"] = connection_id
    return JSONResponse(payload, status_code=status_code)


def _service_status_code(result: dict[str, Any]) -> int:
    code = wire_error_code(result.get("error_code"))
    if code in {"connection_retired", "connection_conflict", "active_lease"}:
        return 409
    if code in {"connection_not_found", "workspace_not_found"}:
        return 404
    if code in {"host_identity_changed", "owner_action_required", "auth_required"}:
        return 409
    if code in {
        "invalid_request",
        "invalid_remote_root",
        "workspace_not_git",
        "workspace_not_git_root",
        "workspace_root_mismatch",
    }:
        return 400
    return 503


#: What a bounded collection says about the part it dropped. A truncation that
#: leaves no trace is indistinguishable from data that was never there, which is
#: precisely the wrong thing to be ambiguous about in a forensic payload — the
#: depth guard already announces itself with ``[truncated]``, and breadth now does
#: the same (the pattern `export_policy_contract` uses for its exclusion lists:
#: whatever is capped, the count is disclosed beside it).
_OMITTED_KEY = "…"


def _omission_note(total: int, kept: int) -> str:
    return f"{total - kept} of {total} entries omitted (bounded live projection)"


def _bounded_live_value(value: Any, *, depth: int = 0) -> Any:
    """Keep live diagnostics useful without turning WS frames into log dumps."""

    if depth >= 4:
        return "[truncated]"
    if isinstance(value, str):
        return _sanitize_live_text(value)[:_LIVE_TEXT_LIMIT]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            _sanitize_live_text(str(key))[:128]: _bounded_live_value(
                item, depth=depth + 1
            )
            for key, item in items[:_LIVE_COLLECTION_LIMIT]
        }
        if len(items) > _LIVE_COLLECTION_LIMIT:
            bounded[_OMITTED_KEY] = _omission_note(len(items), _LIVE_COLLECTION_LIMIT)
        return bounded
    if isinstance(value, (list, tuple)):
        items = list(value)
        bounded = [
            _bounded_live_value(item, depth=depth + 1)
            for item in items[:_LIVE_COLLECTION_LIMIT]
        ]
        if len(items) > _LIVE_COLLECTION_LIMIT:
            bounded.append(_omission_note(len(items), _LIVE_COLLECTION_LIMIT))
        return bounded
    return _sanitize_live_text(str(value))[:_LIVE_TEXT_LIMIT]


def _public_live_fields(
    value: Any,
    *,
    default_phase: str = "",
) -> dict[str, Any]:
    """Bound live mappings and project typed exceptions through one projection."""

    if isinstance(value, BaseException):
        code = wire_error_code(
            getattr(value, "code", None)
            or getattr(value, "error_code", None)
            or "remote_service_error"
        )
        row: dict[str, Any] = {
            "status": "degraded",
            "error_code": code,
            "phase": str(getattr(value, "phase", None) or default_phase),
            "completion": str(
                getattr(value, "completion", None) or "not_started"
            ),
            # A typed refusal names its own action (and derives it from its code —
            # `workspace_diagnostics.RemoteWorkspaceError`). An UNTYPED exception has
            # no action attribute at all, and the fallback used to be the bare literal
            # `"retry"` for every one of them, so a plain `ValueError` carrying a
            # contract-drift message reached the owner as "try again". The register
            # answers for the code instead; `retry` survives only where a retry is
            # genuinely what moves it.
            "action": str(
                getattr(value, "action", None)
                or REFUSAL_ACTIONS.get(code)
                or "retry"
            ),
        }
        diagnostic_fn = getattr(value, "diagnostic", None)
        if callable(diagnostic_fn):
            try:
                diagnostic = diagnostic_fn()
                # A MISSING field reads as null instead of raising: a typed
                # diagnostic whose shape drifts by one attribute must not make
                # the whole forensic payload vanish from the owner's response.
                row["diagnostic"] = {
                    key: getattr(diagnostic, key, None)
                    for key in (
                        "domain",
                        "code",
                        "message",
                        "phase",
                        "request_id",
                        "operation_id",
                        "completion",
                        "retryable",
                        "errno",
                        "details",
                    )
                }
            except Exception:
                pass
    else:
        row = dict(value) if isinstance(value, dict) else {}
    if not row.get("platform") and row.get("system"):
        row["platform"] = row["system"]
    if not row.get("architecture") and row.get("machine"):
        row["architecture"] = row["machine"]
    result = {
        key: str(row[key])[:512]
        for key in (
            "status",
            "phase",
            "task_id",
            "project_id",
            "platform",
            "architecture",
            "build",
            "completion",
            "error_code",
            "action",
            # The blocker rides on the live frame too, or a Settings row that learned
            # about a remaining block from `GET /connections` would forget it on the
            # next `connection_state` push (the browser drops any field absent from
            # both whitelists) — the exact way the outdated-executor badge used to
            # flicker off.
            "blocked_by",
            "blocker_action",
            "blocker_hint",
        )
        if key in row and row[key] is not None
    }
    if "status" in result and result["status"].strip().lower() not in CONNECTION_STATUSES:
        # A one-sided guard closed. The exception branch above hardcodes `degraded`;
        # this branch passed a MAPPING's `status` through untouched, and those mappings
        # come from the broker, the transport's own `health()` and the service envelope
        # — several producers, one closed five-word contract, and no boundary enforcing
        # it. A sixth word is not additive but invisible: `_record_runtime_health`
        # silently drops a status it does not know and the browser reads one as *no
        # typed status at all*, falling back to derivation and withdrawing the Reconnect
        # button at the one moment a reconnect is the fix. `unknown` is the contract's
        # own word for "no typed statement", so clamping says the same thing the frame
        # would have meant, in a spelling every consumer already handles.
        result["status"] = "unknown"
    if isinstance(row.get("blocker_rank"), int) and not isinstance(row.get("blocker_rank"), bool):
        result["blocker_rank"] = int(row["blocker_rank"])
    if "error_code" in result:
        # ONE normalization for the whole surface: the code leaves here in the wire
        # spelling every consumer compares against, whatever case its authority used.
        result["error_code"] = wire_error_code(result["error_code"])
    for key in ("bootstrap_compatible", "health_fresh", "execd_outdated"):
        if key in row:
            result[key] = bool(row[key])
    for key in ("required_contract_set", "bootstrap_contract_set"):
        if isinstance(row.get(key), int) and not isinstance(row.get(key), bool):
            result[key] = int(row[key])
    if isinstance(row.get("diagnostic"), dict):
        result["diagnostic"] = _bounded_live_value(row["diagnostic"])
    # A capped list plus the TOTAL it was capped from. One number, not a number and
    # a boolean: a separate `*_truncated` flag can disagree with its own count, and
    # "truncated" is exactly `count > len(list)`. Without the count, a surface that
    # shows four of nine warnings is indistinguishable from one that shows all four,
    # which is how a real transport warning silently stopped reaching the owner.
    for name in ("log_refs", "warnings"):
        if isinstance(row.get(name), list):
            rows = [item for item in row[name] if isinstance(item, dict)]
            result[name] = [
                _bounded_live_value(item) for item in rows[:_LIVE_COLLECTION_LIMIT]
            ]
            result[f"{name}_count"] = len(rows)
    return result


def _broadcast_connection_state(connection_id: str, result: dict[str, Any]) -> None:
    """Emit the connection-wide frame, then one TASK-SCOPED frame per live task on it.

    The contract declares `task_id`/`project_id` on this envelope and both browser
    consumers were written for them — but no producer ever set them, so Chat's
    `remoteTaskStates` map (which is filled ONLY from these frames) stayed empty
    forever and its whole remote section, card, per-task Reconnect/Cancel and
    typing-indicator suppression, never rendered once. The Activity subtab escaped
    only because it seeds its map from durable rows and so had something for a
    connection-wide frame to fan out to.

    Home genuinely knows the relation: the queue holds each task's sealed placement,
    and the same walk already answers "is this connection busy?" for the owner
    mutations. A task-scoped frame also carries `project_id`, which is what lets Chat
    scope a frame to its own Project thread instead of rendering foreign work.

    The connection-wide frame is still sent FIRST and unconditionally: a surface that
    knows about a connection but not about its tasks (Settings → Connections) needs
    it, and it is the only frame there is when nothing is running.
    """
    try:
        from supervisor.message_bus import get_bridge

        bridge = get_bridge()
        public = _public_live_fields(result)
        bridge.broadcast({
            "type": "connection_state",
            "connection_id": connection_id,
            **public,
        })
        for row in _live_connection_tasks(connection_id) or []:
            bridge.broadcast({
                "type": "connection_state",
                "connection_id": connection_id,
                **public,
                # The task's OWN identity wins over anything the transport put in
                # the result: the placement is the authority on which task this is.
                "task_id": row["task_id"],
                **({"project_id": row["project_id"]} if row["project_id"] else {}),
            })
    except Exception:
        pass


def _observed_host_id(result: dict[str, Any]) -> str:
    """The host identity a live answer observed, in the three places it can appear.

    Exactly three, and each has a producer: the envelope's own ``host_id``
    (``remote_ssh.probe``/``bootstrap``), the ``handshake`` block
    (``remote_ssh._handshake``, ``remote_session_admission``), and a typed
    ``diagnostic`` whose details name the host that answered. A fourth arm read
    ``admission_evidence``, which nothing anywhere produces — and its browser and CLI
    twins (``connections_ui.js::observedHostId``, ``cli_connections._observed_cli_host_id``)
    had only three, so the extra arm was both dead and a mirror divergence. Retrust
    depends on all three agreeing across the surfaces, so they are kept in step.
    """
    for source in (
        result,
        result.get("handshake") if isinstance(result.get("handshake"), dict) else {},
        result.get("diagnostic") if isinstance(result.get("diagnostic"), dict) else {},
    ):
        host_id = str(source.get("host_id") or source.get("observed_host_id") or "").strip()
        if host_id:
            return host_id
    return ""


def _live_connection_tasks(connection_id: str) -> list[dict[str, str]] | None:
    """``[{task_id, project_id}]`` for queued/running tasks placed on this connection.

    ONE walk of the queue, read two ways by design. The owner mutations need "is
    anything live here?" and must fail CLOSED, while the live broadcast needs "which
    tasks does this news concern?" and must fail EMPTY — so the ambiguous case is
    returned as ``None`` ("could not tell") instead of being resolved here for both
    callers at once. A second walk would be a second answer to one question.

    Home task state comes from the queue; the PRE-QUEUE admission window belongs to
    the broker's own ``has_active_lease``, because the broker is the only thing that
    knows an admission is in flight.
    """
    try:
        from ouroboros.workspace_ref import workspace_ref_for
        from supervisor import queue as supervisor_queue
        from supervisor.queue import PENDING, RUNNING

        rows: list[dict[str, str]] = []
        with supervisor_queue._queue_lock:
            tasks = list(PENDING)
            tasks.extend(
                row.get("task")
                for row in RUNNING.values()
                if isinstance(row, dict)
            )
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                try:
                    ref = workspace_ref_for(task)
                except ValueError:
                    continue
                if (
                    ref is not None
                    and ref.kind == "ssh"
                    and ref.connection_id == connection_id
                ):
                    task_id = str(task.get("id") or task.get("task_id") or "")
                    if task_id:
                        rows.append({
                            "task_id": task_id,
                            "project_id": str(task.get("project_id") or ""),
                        })
        return rows
    except Exception:
        return None


def _connection_busy(service: Any, connection_id: str) -> bool:
    """Fail-closed live-task plus broker-lease check for owner mutations."""

    live = _live_connection_tasks(connection_id)
    if live is None or live:
        # Lifecycle lookup failure must not make a destructive trust/lifecycle
        # operation look safe.
        return True
    probe = getattr(service, "has_active_lease", None)
    if not callable(probe):
        return True
    try:
        return bool(probe(connection_id))
    except Exception:
        return True


async def api_connections_list(request: Request) -> JSONResponse:
    path = _connection_store_path(request)
    try:
        rows = list_connections(path, include_retired=True)
    except Exception as exc:
        return _typed_error(
            str(exc), 500, error_code="connection_store_unavailable",
            action=REFUSAL_ACTIONS["connection_store_unavailable"],
        )
    service = _remote_service(request)
    live: dict[str, dict[str, Any]] = {}
    if service is not None:
        status = await _service_call(service, "status", None)
        items = status.get("connections") if isinstance(status, dict) else None
        if isinstance(items, list):
            live = {
                str(item.get("connection_id") or item.get("id") or ""):
                    _public_live_fields(item)
                for item in items
                if (
                    isinstance(item, dict)
                    and str(item.get("connection_id") or item.get("id") or "")
                )
            }
        elif isinstance(items, dict):
            live = {
                str(key): _public_live_fields(value)
                for key, value in items.items()
                if str(key)
            }
        elif isinstance(status, dict):
            connection_id = str(status.get("connection_id") or "")
            if connection_id:
                live = {connection_id: _public_live_fields(status)}
    projected = []
    for row in rows:
        connection_id = str(row.get("id") or "")
        live_row = live.get(connection_id, {})
        if live_row:
            _record_runtime_health(path, connection_id, live_row)
        evidence = _runtime_evidence_fields(path, connection_id, row)
        merged = {
            **row,
            "status": "unknown" if row.get("lifecycle") == "active" else "disconnected",
            **live_row,
            # Bootstrap compatibility (durable, owner store) and health freshness
            # (process-local) are Home's OWN admission evidence. A transport status
            # projection must never manufacture or override either.
            **evidence,
        }
        # …and LAST, the one thing standing in front of this connection, derived from
        # the merged evidence rather than guessed at by whoever renders it. Before
        # this, the New Project picker composed its own empty-state sentence listing
        # every conceivable cure ("run Bootstrap (or Test to refresh health)"), so an
        # owner blocked by an outdated executor pressed Test, got `ok`, and stayed
        # blocked — Test cannot write the contract-set stamp. `connection_blocker`
        # returns `{}` exactly when the row is selectable, which is why the browser's
        # `isSelectableRemoteConnection` and this ladder are one predicate.
        projected.append({**merged, **connection_blocker(merged)})
    return JSONResponse({"connections": projected})


async def api_connections_add(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        return _typed_error(
            "body must be a JSON object", 400, error_code="invalid_request",
        )
    try:
        row = add_connection(
            name=body.get("name"),
            ssh_alias=body.get("ssh_alias"),
            path=_connection_store_path(request),
        )
    except ValueError as exc:
        return _typed_error(
            str(exc), 409 if "already uses" in str(exc) else 400,
            error_code="connection_conflict" if "already uses" in str(exc) else "invalid_request",
        )
    except Exception as exc:
        return _typed_error(
            str(exc), 500, error_code="connection_store_unavailable",
        )
    return JSONResponse({"ok": True, "connection": {**row, "status": "unknown"}}, status_code=201)


async def _connection_action(request: Request, method: str) -> JSONResponse:
    connection_id = str(request.path_params.get("connection_id") or "").strip()
    path = _connection_store_path(request)
    row = get_connection(connection_id, path)
    if row is None:
        return _typed_error(
            "connection not found", 404, error_code="connection_not_found",
            connection_id=connection_id,
        )
    if row.get("lifecycle") != "active":
        return _typed_error(
            "connection is retired", 409, error_code="connection_retired",
            action=REFUSAL_ACTIONS["connection_retired"], connection_id=connection_id,
        )
    service = _remote_service(request)
    if service is None:
        return _service_missing_response(connection_id)
    from ouroboros.config import get_ssh_timeout_sec

    timeout_kind = {
        "bootstrap": "bootstrap",
        "reconnect_connection": "reconcile",
    }.get(method, "connect")
    result = await _service_call(
        service,
        method,
        row,
        timeout_sec=float(get_ssh_timeout_sec(timeout_kind)),
    )
    if not result.get("platform") and result.get("system"):
        result["platform"] = result["system"]
    if not result.get("architecture") and result.get("machine"):
        result["architecture"] = result["machine"]
    observed = _observed_host_id(result)
    expected = str(row.get("expected_host_id") or "").strip()
    if (
        result.get("ok")
        and observed
        and expected
        and observed != expected
    ):
        result = {
            **result,
            "ok": False,
            "status": "degraded",
            "error": "remote host identity differs from the pinned identity",
            "error_code": "host_identity_changed",
            "action": ACTION_RETRUST,
        }
    if method == "bootstrap" and result.get("ok") and not observed:
        result = {
            **result,
            "ok": False,
            "status": "degraded",
            "error": "bootstrap did not return a remote host identity",
            "error_code": "host_identity_missing",
            "action": ACTION_BOOTSTRAP,
        }
    if method == "bootstrap" and result.get("ok") and observed:
        try:
            row = pin_connection_host(
                connection_id, observed, path=path,
            )
            # The DURABLE half of the admission evidence, written by the only
            # operation that establishes it. It outlives this process, which is the
            # whole point: the compatible executor stays installed on that host when
            # Ouroboros restarts, so the picker must not go blind.
            row = record_bootstrap(
                connection_id,
                build=str(result.get("build") or ""),
                contract_set=int(result.get("contract_set") or 0),
                path=path,
            )
        except ValueError as exc:
            result = {
                **result,
                "ok": False,
                "status": "degraded",
                "error": str(exc),
                "error_code": "host_identity_changed",
                "action": ACTION_RETRUST,
            }
        except KeyError:
            # The row was retired or removed between the read and the pin —
            # a successful bootstrap must not report a trust pin that no
            # durable record carries.
            result = {
                **result,
                "ok": False,
                "status": "disconnected",
                "error": "connection disappeared while bootstrap was pinning it",
                "error_code": "connection_not_found",
                "action": ACTION_RELOAD_CONNECTIONS,
            }
    if method == "reconnect_connection" and str(result.get("status") or "") != "ready":
        result["ok"] = False
        result.setdefault("error", "remote project session was not reconnected")
        result.setdefault("error_code", "remote_session_disconnected")
        result.setdefault("action", ACTION_READMIT_PROJECT)
    _record_runtime_health(path, connection_id, result)
    # `row` is re-read by the store on every mutation above, so this sees the
    # bootstrap just recorded — and a plain Test sees the one recorded in a
    # previous run, which is how Test alone brings a connection back into the picker.
    result.update(_runtime_evidence_fields(path, connection_id, row))
    # What is STILL in the way after this action, in the same answer that reports the
    # action succeeded. This is the seam the owner's dead end ran through: a Test on a
    # connection with an outdated executor answered `ok` with `health_fresh: true` and
    # said nothing about the block that remained, so the surface had no way to correct
    # the advice it had already given. `{}` when nothing is in the way.
    result.update(connection_blocker({**row, **result}))
    result.setdefault("connection_id", connection_id)
    result.setdefault("connection", row)
    _broadcast_connection_state(connection_id, result)
    return JSONResponse(
        result,
        status_code=200 if result.get("ok") else _service_status_code(result),
    )


api_connection_test = partial(_connection_action, method="test_connection")
api_connection_test.__name__ = "api_connection_test"
api_connection_bootstrap = partial(_connection_action, method="bootstrap")
api_connection_bootstrap.__name__ = "api_connection_bootstrap"
api_connection_reconnect = partial(
    _connection_action,
    method="reconnect_connection",
)
api_connection_reconnect.__name__ = "api_connection_reconnect"


async def api_connection_dirs(request: Request) -> JSONResponse:
    connection_id = str(request.path_params.get("connection_id") or "").strip()
    row = get_connection(connection_id, _connection_store_path(request))
    if row is None:
        return _typed_error(
            "connection not found", 404, error_code="connection_not_found",
            connection_id=connection_id,
        )
    if row.get("lifecycle") != "active":
        return _typed_error(
            "connection is retired", 409, error_code="connection_retired",
            action=REFUSAL_ACTIONS["connection_retired"], connection_id=connection_id,
        )
    service = _remote_service(request)
    if service is None:
        return _service_missing_response(connection_id)
    from ouroboros.config import get_ssh_timeout_sec

    result = await _service_call(
        service,
        "list_directories",
        row,
        remote_root=str(request.query_params.get("path") or ""),
        timeout_sec=float(get_ssh_timeout_sec("connect")),
    )
    if not result.get("ok", True):
        return JSONResponse(result, status_code=_service_status_code(result))
    dirs = [
        {
            "name": str(item.get("name") or ""),
            "path": str(item.get("path") or ""),
            "is_git": bool(item.get("is_git")),
        }
        for item in (result.get("dirs") or [])
        if isinstance(item, dict)
    ][:500]
    return JSONResponse({
        "connection_id": connection_id,
        "path": str(result.get("path") or ""),
        "parent": str(result.get("parent") or ""),
        "dirs": dirs,
        "truncated": bool(result.get("truncated")) or len(result.get("dirs") or []) > 500,
    })


async def api_connection_retrust(request: Request) -> JSONResponse:
    connection_id = str(request.path_params.get("connection_id") or "").strip()
    path = _connection_store_path(request)
    row = get_connection(connection_id, path)
    if row is None:
        return _typed_error(
            "connection not found", 404, error_code="connection_not_found",
            connection_id=connection_id,
        )
    if row.get("lifecycle", "active") != "active":
        return _typed_error(
            "connection is retired", 409, error_code="connection_retired",
            action=REFUSAL_ACTIONS["connection_retired"], connection_id=connection_id,
        )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict) or body.get("confirm") is not True:
        return _typed_error(
            "explicit retrust confirmation is required", 400,
            error_code="confirmation_required", action=ACTION_CONFIRM_RETRUST,
            connection_id=connection_id,
        )
    service = _remote_service(request)
    if service is None:
        return _service_missing_response(connection_id)
    if _connection_busy(service, connection_id):
        return _typed_error(
            "connection has an active task or lease", 409,
            error_code="active_lease", action=ACTION_CANCEL_TASKS,
            connection_id=connection_id,
        )
    from ouroboros.config import get_ssh_timeout_sec

    probe = await _service_call(
        service,
        "test_connection",
        row,
        timeout_sec=float(get_ssh_timeout_sec("connect")),
    )
    observed = _observed_host_id(probe)
    if not probe.get("ok") and not observed:
        probe.setdefault("connection_id", connection_id)
        return JSONResponse(probe, status_code=_service_status_code(probe))
    expected_old = str(body.get("old_host_id") or "").strip()
    expected_new = str(body.get("new_host_id") or "").strip()
    current = str(row.get("expected_host_id") or "")
    if not observed or expected_old != current or expected_new != observed:
        return _typed_error(
            "host identity changed while retrust was being confirmed", 409,
            error_code="host_identity_confirmation_stale", action=ACTION_TEST,
            connection_id=connection_id,
        )
    try:
        from supervisor import queue as supervisor_queue

        with supervisor_queue._queue_lock:
            if _connection_busy(service, connection_id):
                return _typed_error(
                    "connection has an active task or lease", 409,
                    error_code="active_lease", action=ACTION_CANCEL_TASKS,
                    connection_id=connection_id,
                )
            trusted = retrust_connection(
                connection_id,
                observed,
                path=path,
                has_active_lease=False,
            )
    except ValueError as exc:
        return _typed_error(
            str(exc), 409, error_code="connection_conflict",
            connection_id=connection_id,
        )
    except KeyError:
        return _typed_error(
            "connection not found", 404, error_code="connection_not_found",
            connection_id=connection_id,
        )
    _drop_runtime_evidence(path, connection_id)
    result = {
        "ok": True,
        "connection_id": connection_id,
        "connection": trusted,
        "status": "unknown",
        "phase": "retrust",
        "completion": "retrusted",
        "action": REFUSAL_ACTIONS["connection_not_bootstrapped"],
        "bootstrap_compatible": False,
        "health_fresh": False,
    }
    # Retrust clears the durable bootstrap claim on purpose, so the connection is
    # blocked the instant it succeeds. Saying so in the SAME answer is what lets the
    # surface report "trusted" and "still needs Bootstrap" as one truth instead of a
    # success message that happens to have the right next step written into it.
    result.update(connection_blocker({**trusted, **result}))
    _broadcast_connection_state(connection_id, result)
    return JSONResponse(result)


async def api_connection_retire(request: Request) -> JSONResponse:
    connection_id = str(request.path_params.get("connection_id") or "").strip()
    path = _connection_store_path(request)
    row = get_connection(connection_id, path)
    if row is None:
        return _typed_error(
            "connection not found", 404, error_code="connection_not_found",
            connection_id=connection_id,
        )
    service = _remote_service(request)
    if service is None:
        return _service_missing_response(connection_id)
    try:
        from supervisor import queue as supervisor_queue

        with supervisor_queue._queue_lock:
            if _connection_busy(service, connection_id):
                return _typed_error(
                    "connection has an active task or lease", 409,
                    error_code="active_lease", action=ACTION_CANCEL_TASKS,
                    connection_id=connection_id,
                )
            retired = retire_connection(
                connection_id,
                path=path,
                has_active_lease=False,
            )
    except ValueError as exc:
        return _typed_error(
            str(exc), 409, error_code="connection_conflict",
            connection_id=connection_id,
        )
    except KeyError:
        return _typed_error(
            "connection not found", 404, error_code="connection_not_found",
            connection_id=connection_id,
        )
    cancel_connection = getattr(service, "cancel_connection", None)
    if callable(cancel_connection):
        try:
            await asyncio.to_thread(cancel_connection, connection_id)
        except Exception:
            pass
    result = {
        "ok": True,
        "connection_id": connection_id,
        "connection": retired,
        "status": "disconnected",
        "completion": "retired",
    }
    _drop_runtime_evidence(path, connection_id)
    _broadcast_connection_state(connection_id, result)
    return JSONResponse(result)


__all__ = [
    "REMOTE_TRANSPORT_UNAVAILABLE",
    "api_connection_bootstrap",
    "api_connection_dirs",
    "api_connection_reconnect",
    "api_connection_retire",
    "api_connection_retrust",
    "api_connection_test",
    "api_connections_add",
    "api_connections_list",
]
