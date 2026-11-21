"""Send this process's ERROR log records to a Sentry-compatible error tracker (example).

The extension has no declared dependencies and no binary files, so it loads in process in the server and in every
pool worker. The SDK comes from Ouroboros's own environment. The destination is the ``SENTRY_DSN`` setting, read
through this extension's grant (a deployment policy can fix it); without that setting the SDK reads the process's
``SENTRY_DSN`` variable itself, as it always did, and without either it sends nothing.

Each process records only what its own transport observed in a file of its own: reports queued, accepted by the
receiver (an HTTP 2xx answer: the receiver took the request, which says nothing about storing the event), refused,
rate limited, unreachable, or dropped before sending. The ``status`` route serves recently updated records with the
process that made each, says how many older ones it leaves out, and calls a process whose start time it cannot read
of unknown liveness; the widget draws them. Recording never raises and never logs above DEBUG, so it cannot report
itself. The hooks are sentry-sdk 2.71.0's transport methods; a version without one reports less.
"""

import heapq
import json
import logging
import os
import threading
import time

import sentry_sdk
from sentry_sdk.integrations.atexit import AtexitIntegration
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.transport import HttpTransport

from ouroboros.observability import redact_projection

log = logging.getLogger(__name__)

# The parts of an error event a tracker needs. The logging integration also copies a
# record's `extra` dict, which holds whatever a caller logged, so it is not forwarded.
_EVENT_FIELDS = ("event_id", "timestamp", "level", "logger", "platform", "sdk", "server_name", "release",
                 "environment", "tags", "fingerprint", "logentry", "message", "exception")
_OUTCOMES = ("queued", "accepted", "refused", "rate_limited", "unreachable", "dropped")
_SHOWN_PROCESSES = 16


def _before_send(event, hint):
    return redact_projection({key: event[key] for key in _EVENT_FIELDS if key in event}).value


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _birth(pid):
    try:
        from ouroboros.platform_layer import process_start_time

        return str(process_start_time(pid) or "")
    except Exception:
        return ""


def _liveness(record):
    """``running`` (its pid has the start time it recorded), ``ended`` (provably gone, or its pid now belongs to a
    later process), else ``unknown``: a start time that cannot be read is never taken for an end."""
    pid, started = int(record.get("pid") or 0), str(record.get("started") or "")
    try:
        from ouroboros.platform_layer import pid_provably_gone

        if pid > 0 and pid_provably_gone(pid):
            return "ended"
    except Exception:
        return "unknown"
    now = _birth(pid) if pid > 0 else ""
    if started and now:
        return "running" if now == started else "ended"
    return "unknown"


class _Delivery:
    """This process's observations, written whole to its own file after each one."""

    def __init__(self, path, facts):
        self._path, self._lock = path, threading.Lock()
        self._state = {**facts, "counts": dict.fromkeys(_OUTCOMES, 0), "last": None}
        self._write()

    def note(self, outcome, detail=""):
        with self._lock:
            self._state["counts"][outcome] += 1
            if outcome != "queued":
                self._state["last"] = {"outcome": outcome, "detail": detail, "at": _now()}
            self._write()

    def _write(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            partial = f"{self._path}.{os.getpid()}.tmp"
            with open(partial, "w", encoding="utf-8") as handle:
                json.dump(self._state, handle)
            os.replace(partial, self._path)
        except Exception:
            log.debug("error_tracker could not record a delivery outcome", exc_info=True)


def _carries_event(envelope):
    return envelope is not None and any(item.data_category == "error" for item in envelope.items)


def _transport(delivery):
    """sentry-sdk's HTTP transport with this process's outcomes recorded (a class: the SDK instantiates it)."""
    dropping = threading.local()

    class ObservedTransport(HttpTransport):
        def capture_envelope(self, envelope):
            dropping.full = False
            super().capture_envelope(envelope)  # on a full queue it calls on_dropped_event first
            if _carries_event(envelope) and not dropping.full:
                delivery.note("queued")

        def _handle_response(self, response, envelope):
            super()._handle_response(response=response, envelope=envelope)
            if _carries_event(envelope):
                status = int(getattr(response, "status", 0) or 0)
                outcome = "accepted" if 200 <= status < 300 else "rate_limited" if status == 429 else "refused"
                delivery.note(outcome, f"HTTP {status}")

        def _handle_request_error(self, envelope, loss_reason="network", record_reason="network_error"):
            if loss_reason == "network" and _carries_event(envelope):
                delivery.note("unreachable", "no HTTP answer")
            super()._handle_request_error(envelope, loss_reason=loss_reason, record_reason=record_reason)

        def on_dropped_event(self, reason):
            if reason == "full_queue":
                dropping.full = True
                delivery.note("dropped", "the send queue was full")
            super().on_dropped_event(reason)

    return ObservedTransport


def _destination(api):
    """``(dsn passed to the SDK, where it comes from)``; the SDK reads the environment's variable itself."""
    configured = str(api.get_settings(["SENTRY_DSN"]).get("SENTRY_DSN") or "").strip()
    if configured:
        return configured, "settings"
    return None, "environment" if os.environ.get("SENTRY_DSN", "").strip() else "none"


def _status_of(state_dir):
    """The newest process records, each said as what that process observed; no DSN, no event text. A sentence
    covers only the processes shown and never takes a process it cannot see for ended or healthy."""
    folder = os.path.join(state_dir, "delivery")
    recent, total = [], 0
    # Bound the JSON reads and retained sort state, not the directory's metadata walk. An older server that
    # reports again belongs in this recent-activity view; startup age alone would hide it behind old workers.
    if os.path.isdir(folder):
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
                        continue
                    item = (entry.stat(follow_symlinks=False).st_mtime_ns, entry.name)
                except OSError:
                    continue
                total += 1
                heapq.heappush(recent, item) if len(recent) < _SHOWN_PROCESSES else heapq.heappushpop(recent, item)
    shown = []
    for _updated, name in sorted(recent, reverse=True):
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                record = json.load(handle)
            if isinstance(record, dict):
                shown.append(record)
        except (OSError, ValueError):
            continue
    omitted = total - len(shown)
    rows, problems, accepted = [], [], 0
    for record in shown:
        liveness = _liveness(record)
        counts, last = record.get("counts") or {}, record.get("last") or {}
        process = (f"{record.get('role')} · pid {record.get('pid')} · "
                   f"{'liveness unknown' if liveness == 'unknown' else liveness}")
        rows.append({"process": process, "loaded": record.get("loaded_at", ""), "sdk": record.get("sdk", ""),
                     "destination": {"settings": "Settings", "environment": "Environment",
                                     "none": "Not configured"}.get(record.get("source"), "Unknown"),
                     **{outcome: counts.get(outcome, 0) for outcome in _OUTCOMES},
                     "last": f"{last.get('outcome')} ({last.get('detail')}) at {last.get('at')}" if last else "—"})
        if liveness != "ended" and last and last.get("outcome") != "accepted":
            problems.append(f"{process}: {last.get('outcome')} ({last.get('detail')}) at {last.get('at')}")
        accepted += int(counts.get("accepted") or 0)
    configured = any(record.get("source") != "none" for record in shown)
    return {
        "processes": rows,
        "problem": "Last outcome not accepted in " + "; ".join(problems) if problems else "",
        "ok": "" if problems or not accepted else ("The receiver answered HTTP 2xx to the last report of each "
                                                   "process shown here that sent one and is not known to have ended. "
                                                   "That means it took the request, not that it stored the event."),
        "idle": "" if problems or accepted else ("Configured; nothing has been sent yet." if configured else
                                                 "Not configured: no SENTRY_DSN, so nothing is sent."),
        "note": (f"{omitted} process record{'s are' if omitted != 1 else ' is'} not shown and not counted "
                 "above.") if omitted else "",
    }


def register(api):
    role = "worker" if os.environ.get("OUROBOROS_IN_WORKER") == "1" else "server"
    dsn, source = _destination(api)
    state_dir = api.get_state_dir()
    delivery = _Delivery(os.path.join(state_dir, "delivery", f"{role}-{os.getpid()}.json"), {
        "role": role, "pid": os.getpid(), "started": _birth(os.getpid()), "loaded_at": _now(),
        "source": source, "sdk": sentry_sdk.VERSION})
    sentry_sdk.init(
        **({"dsn": dsn} if dsn else {}),  # otherwise the SDK reads SENTRY_DSN from the environment, as before
        transport=_transport(delivery),
        include_local_variables=False,  # frame locals hold prompts, keys and file contents
        include_source_context=False,
        send_default_pii=False,
        default_integrations=False,  # no auto-instrumented model clients, no trace headers on their requests
        integrations=[
            LoggingIntegration(level=None, event_level=logging.ERROR),
            DedupeIntegration(),
            AtexitIntegration(),
        ],
        max_request_body_size="never",
        trace_propagation_targets=[],
        before_send=_before_send,
        server_name=f"ouroboros-{role}",  # not the machine's hostname
        release=f"ouroboros@{api.get_runtime_info().get('app_version', '')}",
    )
    api.on_unload(lambda: sentry_sdk.get_client().close(timeout=1))
    api.register_route("status", lambda _request: _status_of(state_dir), methods=("GET",))
    api.register_ui_tab("delivery", title="Error tracker", render={
        "kind": "declarative", "schema_version": 1, "components": [
            {"type": "poll", "label": "Refresh", "route": "status", "auto_start": True, "interval_ms": 15000,
             "max_ticks": 40},
            {"type": "callout", "tone": "warning", "path": "problem", "condition_key": "problem"},
            {"type": "callout", "tone": "success", "path": "ok", "condition_key": "ok"},
            {"type": "callout", "tone": "info", "path": "idle", "condition_key": "idle"},
            {"type": "callout", "tone": "info", "path": "note", "condition_key": "note"},
            {"type": "table", "path": "processes", "columns": [
                {"label": "Process", "path": "process"}, {"label": "Destination", "path": "destination"},
                {"label": "Queued", "path": "queued", "presentation": "number"},
                {"label": "Accepted", "path": "accepted", "presentation": "number"},
                {"label": "Refused", "path": "refused", "presentation": "number"},
                {"label": "Unreachable", "path": "unreachable", "presentation": "number"},
                {"label": "Last outcome", "path": "last"}]},
        ]})
