"""Background thinking loop with scoped tools and no silent context drops."""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import inspect
import json
import logging
import os
import pathlib
import threading
import traceback
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from ouroboros.tools.registry import ToolRegistry

from ouroboros.config import get_consciousness_model, resolve_effort
from ouroboros.context import (
    build_governance_sections,
    build_health_invariants,
    build_knowledge_sections,
    build_memory_sections,
    build_recent_sections,
    build_runtime_section,
    safe_read,
)
from ouroboros.context_budget import (
    BG_CONTEXT_MAX_CHARS,
    BG_CONTEXT_WARN_CHARS,
    BG_STATE_JSON_WARN_CHARS,
)
from ouroboros.llm import LLMClient, add_usage
from ouroboros.loop_tool_execution import StatefulToolExecutor, _truncate_tool_result
from ouroboros.memory import Memory
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
from ouroboros.pricing import infer_provider_from_model
from ouroboros.settings_setup_contract import resolve_total_budget_usd
from ouroboros.utils import (
    append_jsonl,
    emit_log_event,
    jsonl_append_lock_path,
    read_text,
    sanitize_tool_args_for_log,
    sanitize_tool_result_for_log,
    truncate_for_log,
    utc_now_iso,
)

_OBSERVATIONS_REL = pathlib.Path("state") / "consciousness_observations.jsonl"
_OBSERVATION_SOURCE_REF = (
    "read_file(root='runtime_data', "
    "path='state/consciousness_observations.jsonl')"
)
_OBSERVATION_RENDER_LIMIT = 10
_OBSERVATION_RENDER_CHARS = 12_000

log = logging.getLogger(__name__)


class BackgroundConsciousness:
    """Persistent background thinking loop for Ouroboros."""

    def __init__(
        self,
        drive_root: pathlib.Path,
        repo_dir: pathlib.Path,
        event_queue: Any,
        owner_chat_id_fn: Callable[[], Optional[int]],
    ):
        self._drive_root = drive_root
        self._repo_dir = repo_dir
        self._event_queue = event_queue
        self._owner_chat_id_fn = owner_chat_id_fn

        self._max_bg_rounds = int(os.environ.get("OUROBOROS_BG_MAX_ROUNDS", "10"))
        self._wakeup_min = int(os.environ.get("OUROBOROS_BG_WAKEUP_MIN", "30"))
        self._wakeup_max = int(os.environ.get("OUROBOROS_BG_WAKEUP_MAX", "7200"))

        self._llm = LLMClient()
        self._registry = self._build_registry()
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._wakeup_event = threading.Event()
        self._next_wakeup_sec: float = 300.0
        # Observations are an append-only inbox.  The lock only serializes the
        # read/dedupe/append transaction inside this process; the JSONL helper
        # supplies the durable writer lock shared with other processes.
        self._observation_lock = threading.RLock()
        self._observation_state_cache: Optional[Dict[str, Any]] = None
        self._observation_store_signature: Optional[tuple] = None
        self._deferred_events: list = []
        self._tool_executor = StatefulToolExecutor()
        self._identity_source_requirements: Dict[str, pathlib.Path] = {}
        self._identity_source_reads: Dict[str, str] = {}
        self._identity_unresolved_sources: set[str] = set()

        self._bg_spent_usd: float = 0.0
        self._bg_budget_pct: float = float(
            os.environ.get("OUROBOROS_BG_BUDGET_PCT", "10")
        )
        self._last_cycle_started_at: str = ""
        self._last_cycle_finished_at: str = ""
        self._last_idle_reason: str = "stopped"
        self._last_error: str = ""

    @property
    def is_running(self) -> bool:
        thread = getattr(self, "_thread", None)
        return bool(getattr(self, "_running", False) and thread is not None and thread.is_alive())

    @property
    def is_paused(self) -> bool:
        return bool(getattr(self, "_paused", False))

    def _observation_lock_for_instance(self) -> threading.RLock:
        """Lazily restore observation fields for object.__new__ overlap tests."""

        lock = getattr(self, "_observation_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._observation_lock = lock
        return lock

    def _stop_requested(self) -> bool:
        event = getattr(self, "_stop_event", None)
        return bool(event is not None and event.is_set())

    @contextlib.contextmanager
    def _observation_writer_lock(self, path: pathlib.Path):
        """Use the same sidecar lock seam as append_jsonl for store transactions."""

        lock_path = jsonl_append_lock_path(path)
        lock_fd = acquire_exclusive_file_lock(
            lock_path,
            timeout_sec=2.0,
            stale_sec=10.0,
            poll_sec=0.01,
        )
        if lock_fd is None:
            yield False
            return
        try:
            yield True
        finally:
            release_exclusive_file_lock(lock_path, lock_fd)

    @staticmethod
    def _append_observation_line_locked(path: pathlib.Path, row: Dict[str, Any]) -> bool:
        """Append one row while the shared JSONL writer lock is held."""

        try:
            data = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        return False
                    view = view[written:]
            finally:
                os.close(fd)
            return True
        except Exception:
            log.warning("Failed to append background observation row", exc_info=True)
            return False

    def _mark_cycle_settlement_gap(self, reason: str) -> None:
        self._cycle_settlement_failed = True
        # A missing/unknown durable receipt means this cognition cycle cannot
        # truthfully settle its observation snapshot, even if the model later
        # returns a final message.
        self._cycle_ack_allowed = False
        reasons = getattr(self, "_cycle_settlement_reasons", None)
        if reasons is None:
            reasons = []
            self._cycle_settlement_reasons = reasons
        if reason not in reasons:
            reasons.append(reason)
        log.error("Background consciousness durable settlement gap: %s", reason)

    def _append_cycle_receipt(
        self,
        path: pathlib.Path,
        row: Dict[str, Any],
        *,
        label: str,
    ) -> bool:
        """Persist a cycle receipt and latch an explicit writer failure."""

        try:
            written = append_jsonl(path, row)
        except Exception as exc:
            self._mark_cycle_settlement_gap(f"{label}: {type(exc).__name__}")
            return False
        # A receipt is authoritative only when the writer reports truthy
        # success.  ``None`` from a callback is an unknown write, not proof.
        if not written:
            self._mark_cycle_settlement_gap(f"{label}: append_jsonl returned false")
            return False
        return True

    @property
    def _model(self) -> str:
        return get_consciousness_model()

    def status_snapshot(self) -> Dict[str, Any]:
        with self._observation_lock_for_instance():
            state = self._read_observation_state()
            pending_count = int(state.get("pending_count") or 0)
            oldest = str(state.get("oldest_pending_at") or "")
            gap_count = int(
                state.get("gap_count", len(state.get("gap_reasons") or ())) or 0
            )
        return {
            "running": bool(self.is_running),
            "paused": bool(self.is_paused),
            "next_wakeup_sec": int(getattr(self, "_next_wakeup_sec", 300) or 300),
            "last_cycle_started_at": getattr(self, "_last_cycle_started_at", ""),
            "last_cycle_finished_at": getattr(self, "_last_cycle_finished_at", ""),
            "last_idle_reason": getattr(self, "_last_idle_reason", "stopped"),
            "last_error": getattr(self, "_last_error", ""),
            # Status is deliberately content-free: it gives the operator a
            # truthful horizon and a resolvable store, never observation text.
            "pending_observation_count": pending_count,
            "oldest_observation_at": oldest or "",
            "observation_source": _OBSERVATION_SOURCE_REF,
            "observation_source_complete": gap_count == 0,
            "observation_gap_count": gap_count,
        }

    def start(self) -> str:
        if self.is_running:
            return "Background consciousness is already running."
        self._running = True
        self._paused = False
        self._last_idle_reason = "starting"
        self._last_error = ""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return "Background consciousness started."

    def stop(self) -> str:
        if not self.is_running:
            return "Background consciousness is not running."
        self._running = False
        self._last_idle_reason = "stopping"
        self._stop_event.set()
        self._wakeup_event.set()  # Unblock sleep
        try:
            self._tool_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.debug("Failed to shutdown consciousness tool executor", exc_info=True)
        return "Background consciousness stopping."

    def pause(self) -> None:
        """Pause during foreground task execution."""
        self._paused = True
        self._last_idle_reason = "paused_by_active_task"

    def resume(self) -> None:
        """Resume after a task and flush deferred events first."""
        if self._deferred_events and self._event_queue is not None:
            for evt in self._deferred_events:
                self._event_queue.put(evt)
            self._deferred_events.clear()
        self._paused = False
        self._last_idle_reason = "waking"
        self._wakeup_event.set()

    def inject_observation(
        self,
        text: Any,
        *,
        observation_id: Optional[str] = None,
        source: str = "runtime",
        kind: str = "text",
        observed_at: Optional[str] = None,
        payload: Any = None,
        ref: Any = None,
    ) -> bool:
        """Persist one observation before waking the background loop.

        Existing producers pass a string only.  Structured callers may provide
        a stable ``observation_id`` plus source/kind/time/payload/ref.  A
        duplicate stable ID is a no-op, which makes retries and concurrent
        producers idempotent without introducing a second queue subsystem.
        """
        if isinstance(text, dict):
            record = dict(text)
            payload = record.get("payload", record.get("text", payload))
            observation_id = observation_id or record.get("id") or record.get("observation_id")
            source = str(record.get("source") or source)
            kind = str(record.get("kind") or kind)
            observed_at = observed_at or record.get("time") or record.get("observed_at")
            ref = ref if ref is not None else record.get("ref")
        if payload is None:
            payload = text
        identifier = str(observation_id or uuid.uuid4().hex)
        row = {
            "id": identifier,
            "source": str(source or "runtime"),
            "kind": str(kind or "text"),
            "time": str(observed_at or utc_now_iso()),
            "payload": payload,
            "ref": ref,
        }
        path = self._observation_store_path()
        with self._observation_lock_for_instance():
            # The process lock protects this instance; the shared sidecar lock
            # protects sibling processes. Re-read while holding both so the
            # stable-ID check and append are one transaction.
            with self._observation_writer_lock(path) as locked:
                if not locked:
                    log.error("Failed to lock background observation store %s", path)
                    return False
                state = self._read_observation_state(force=True)
                if identifier in state["rows"]:
                    return False
                if not self._append_observation_line_locked(path, {"op": "enqueue", **row}):
                    log.error("Failed to durably enqueue background observation %s", identifier)
                    return False
                state["rows"][identifier] = row
                self._refresh_observation_summary(state)
                self._refresh_observation_signature(path)
        self._wakeup_event.set()
        return True

    def _observation_store_path(self) -> pathlib.Path:
        return self._drive_root / _OBSERVATIONS_REL

    def _read_observation_state(self, *, force: bool = False) -> Dict[str, Any]:
        """Read append-only observations into a cached stable-ID index.

        A process rebuilds from the durable source once, then only performs a
        cheap inode/size/mtime check on status/cycle calls.  A new process (or
        an external writer changing that signature) rebuilds naturally.
        """
        path = self._observation_store_path()
        signature = self._observation_signature(path)
        cached = getattr(self, "_observation_state_cache", None)
        cached_signature = getattr(self, "_observation_store_signature", None)
        if not force and cached is not None and signature == cached_signature:
            return cached

        rows: Dict[str, Dict[str, Any]] = {}
        acked = set()
        gap_reasons: List[str] = []
        try:
            raw_lines = path.open("rb")
        except FileNotFoundError:
            raw_lines = None
        except OSError as exc:
            raw_lines = None
            gap_reasons.append(f"unreadable observation store: {exc}")
        if raw_lines is not None:
            with raw_lines:
                for line_no, raw in enumerate(raw_lines, 1):
                    if not raw.strip():
                        continue
                    try:
                        item = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                        gap_reasons.append(f"malformed observation row at line {line_no}")
                        continue
                    if not isinstance(item, dict):
                        gap_reasons.append(f"non-object observation row at line {line_no}")
                        continue
                    if not self._validate_observation_row(item, line_no, gap_reasons):
                        continue
                    if item.get("op") == "ack":
                        identifier = item.get("id", item.get("observation_id"))
                        if identifier not in rows:
                            gap_reasons.append(
                                f"ack row references unknown observation "
                                f"{identifier!r} at line {line_no}"
                            )
                            # Do not pre-ack an enqueue that may appear later.
                            continue
                    self._index_observation_row(item, rows, acked)

        self._observation_state_cache = {
            "rows": rows,
            "acked": acked,
            "gap_reasons": gap_reasons,
            "gap_count": len(gap_reasons),
        }
        self._refresh_observation_summary(self._observation_state_cache)
        self._observation_store_signature = signature
        return self._observation_state_cache

    @staticmethod
    def _validate_observation_row(
        item: Dict[str, Any],
        line_no: int,
        gap_reasons: List[str],
    ) -> bool:
        """Validate the small durable row contract before indexing it."""

        op = item.get("op")
        if op not in (None, "enqueue", "observation", "ack"):
            gap_reasons.append(f"unknown observation op at line {line_no}")
            return False
        identifier = item.get("id", item.get("observation_id"))
        if not isinstance(identifier, str) or not identifier.strip():
            gap_reasons.append(f"observation row missing id at line {line_no}")
            return False
        if op == "ack":
            return True
        # Canonical enqueue rows are strict. Legacy/old observation rows retain
        # aliases and an optional ref so existing producers remain readable.
        if op == "enqueue":
            required = ("source", "kind", "time", "payload", "ref")
            missing = [key for key in required if key not in item]
            if missing:
                gap_reasons.append(
                    f"enqueue row missing {','.join(missing)} at line {line_no}"
                )
                return False
        elif op == "observation":
            if not str(item.get("source") or "").strip():
                gap_reasons.append(f"observation row missing source at line {line_no}")
                return False
            if not str(item.get("kind") or "").strip():
                gap_reasons.append(f"observation row missing kind at line {line_no}")
                return False
            if not str(item.get("time") or item.get("observed_at") or "").strip():
                gap_reasons.append(f"observation row missing time at line {line_no}")
                return False
            if "payload" not in item and "text" not in item:
                gap_reasons.append(f"observation row missing payload at line {line_no}")
                return False
        return True

    @staticmethod
    def _refresh_observation_summary(state: Dict[str, Any]) -> None:
        rows = state.get("rows") or {}
        acked = state.get("acked") or set()
        pending_count = 0
        oldest = ""
        for identifier, row in rows.items():
            if identifier in acked:
                continue
            pending_count += 1
            if not oldest:
                oldest = str(row.get("time") or "")
        state["pending_count"] = pending_count
        state["oldest_pending_at"] = oldest

    @staticmethod
    def _index_observation_row(
        item: Dict[str, Any],
        rows: Dict[str, Dict[str, Any]],
        acked: set,
    ) -> None:
        """Index one validated JSONL row without changing first-write order."""
        op = item.get("op")
        identifier = str(item.get("id") or item.get("observation_id") or "")
        if not identifier:
            return
        if op == "ack":
            acked.add(identifier)
            return
        if op not in (None, "enqueue", "observation"):
            return
        # First enqueue wins. A retry with the same ID cannot replace the
        # original source or payload after it entered the inbox.
        rows.setdefault(identifier, {
            "id": identifier,
            "source": str(item.get("source") or "runtime"),
            "kind": str(item.get("kind") or "text"),
            "time": str(item.get("time") or item.get("observed_at") or ""),
            "payload": item.get("payload", item.get("text", "")),
            "ref": item.get("ref"),
        })

    @staticmethod
    def _observation_signature(path: pathlib.Path) -> tuple:
        try:
            stat = path.stat()
        except OSError:
            return (None, 0, None)
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _refresh_observation_signature(self, path: pathlib.Path) -> None:
        self._observation_store_signature = self._observation_signature(path)

    def _snapshot_pending_observations(self) -> List[Dict[str, Any]]:
        """Return a non-destructive, insertion-ordered pending snapshot."""
        with self._observation_lock_for_instance():
            state = self._read_observation_state()
            return [row for identifier, row in state["rows"].items()
                    if identifier not in state["acked"]]

    def _ack_observations(self, observations: Sequence[Dict[str, Any]]) -> bool:
        """Append acknowledgements after a settled successful cognition cycle."""
        if not observations:
            return True
        if getattr(self, "_cycle_settlement_failed", False):
            log.error("Cannot acknowledge observations while cycle settlement has a durable gap")
            return False
        path = self._observation_store_path()
        with self._observation_lock_for_instance():
            with self._observation_writer_lock(path) as locked:
                if not locked:
                    log.error("Failed to lock background observation store %s", path)
                    return False
                state = self._read_observation_state(force=True)
                if state.get("gap_reasons"):
                    log.error(
                        "Cannot acknowledge observations with unreadable inbox rows: %s",
                        state["gap_reasons"][:3],
                    )
                    return False
                for observation in observations:
                    identifier = str(observation.get("id") or "")
                    if not identifier or identifier in state["acked"]:
                        continue
                    if not self._append_observation_line_locked(path, {
                        "op": "ack",
                        "id": identifier,
                        "time": utc_now_iso(),
                    }):
                        log.error("Failed to acknowledge background observation %s", identifier)
                        return False
                    state["acked"].add(identifier)
                self._refresh_observation_summary(state)
                self._refresh_observation_signature(path)
        return True

    def _render_observations(self, observations: Sequence[Dict[str, Any]]) -> str:
        """Render a bounded truthful view with a working durable source ref."""
        total = len(observations)
        shown = list(observations[-_OBSERVATION_RENDER_LIMIT:])
        omitted = max(0, total - len(shown))
        source = _OBSERVATION_SOURCE_REF
        with self._observation_lock_for_instance():
            gaps = list((self._read_observation_state().get("gap_reasons") or ()))
        projection_incomplete = bool(omitted)
        lines = [
            f"## Pending observations (total={total}; showing={len(shown)}; "
            f"omitted={omitted}; source={source}; "
            f"source_complete=False; gaps={len(gaps)})",
        ]
        for index, row in enumerate(shown):
            payload = json.dumps(row.get("payload"), ensure_ascii=False, sort_keys=True)
            if len(payload) > 800:
                projection_incomplete = True
                payload = payload[:700] + f"…[payload omitted; read source id={row.get('id')}]"
            item_line = (
                f"- id={row.get('id')} source={row.get('source')} kind={row.get('kind')} "
                f"time={row.get('time')}: {payload}"
            )
            if len("\n".join(lines)) + len(item_line) + 1 > _OBSERVATION_RENDER_CHARS:
                projection_incomplete = True
                lines.append(
                    f"- [projection truncated; omitted={len(shown) - index} "
                    f"more; read source={source} from id={row.get('id')}]"
                )
                break
            lines.append(item_line)
            if row.get("ref") is not None:
                ref_line = f"  ref={json.dumps(row.get('ref'), ensure_ascii=False, sort_keys=True)}"
                if len("\n".join(lines)) + len(ref_line) + 1 <= _OBSERVATION_RENDER_CHARS:
                    lines.append(ref_line)
                else:
                    projection_incomplete = True
                    lines.append(
                        f"  ref=[projection truncated; read source={source} from id={row.get('id')}]"
                    )
        # The source is complete only when the actor saw every row and every
        # rendered field in the bounded projection.  Keep the existing durable
        # source reference usable for materialization, and expose this result
        # through the same identity-completeness envelope as malformed gaps.
        source_complete = not gaps and not projection_incomplete
        lines[0] = (
            f"## Pending observations (total={total}; showing={len(shown)}; "
            f"omitted={omitted}; source={source}; source_complete={source_complete}; "
            f"gaps={len(gaps)})"
        )
        self._observation_projection_incomplete = not source_complete
        return "\n".join(lines)

    def _emit_live_log(self, event_type: str, **fields: Any) -> None:
        emit_log_event(
            self._event_queue,
            {
                "type": event_type,
                "ts": utc_now_iso(),
                "task_id": "bg-consciousness",
                "task_type": "consciousness",
                **fields,
            },
            blocking=True,
            log_label="consciousness live",
        )

    def _emit_cycle_idle(self, state: str) -> None:
        """Signal that a background-thinking cycle ended, so the web UI can retire
        the bg-consciousness live card instead of leaving it in a perpetual
        "thinking" phase.

        Background consciousness writes no task_result, so the renderer has no
        terminal signal of its own. This emits a structured marker
        (``consciousness_state``) consumed by ``web/modules/log_events.js`` — never
        a text-matched one. Replay after reload is handled separately in
        ``gateway/history.py``.
        """
        self._emit_live_log(
            "consciousness_status",
            is_progress=True,
            consciousness_state=state,
        )

    def _loop(self) -> None:
        """Daemon thread: sleep, wake, think, repeat."""
        while not self._stop_event.is_set():
            self._wakeup_event.clear()
            self._wakeup_event.wait(timeout=self._next_wakeup_sec)

            if self._stop_event.is_set():
                break

            if self._paused:
                self._last_idle_reason = "paused_by_active_task"
                continue

            if not self._check_budget():
                self._last_idle_reason = "budget_blocked"
                self._next_wakeup_sec = self._wakeup_max
                continue

            try:
                self._last_cycle_started_at = utc_now_iso()
                self._last_idle_reason = "thinking"
                self._last_error = ""
                cycle_completed = self._think()
                self._last_cycle_finished_at = utc_now_iso()
                # Preserve distinct overflow/LLM error statuses set inside _think().
                if cycle_completed and not self._stop_event.is_set() and not self._paused:
                    self._last_idle_reason = "sleeping"
                # Retire the live card now that this cycle is done (skip while paused:
                # a real task is active and owns the status).
                if not self._paused:
                    self._emit_cycle_idle(self._last_idle_reason)
            except Exception as e:
                self._last_cycle_finished_at = utc_now_iso()
                self._last_idle_reason = "error_backoff"
                self._last_error = repr(e)
                append_jsonl(self._drive_root / "logs" / "events.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "consciousness_error",
                    "error": repr(e),
                    "traceback": traceback.format_exc()[:1500],
                })
                self._emit_cycle_idle("error_backoff")
                self._next_wakeup_sec = min(
                    self._next_wakeup_sec * 2, self._wakeup_max
                )
        self._last_idle_reason = "stopped"
        self._emit_cycle_idle("stopped")

    def _check_budget(self) -> bool:
        """Return whether background consciousness is within its budget."""
        try:
            from ouroboros.usage_accounting import usage_projection

            total_budget = resolve_total_budget_usd()
            if total_budget is None:
                return True
            max_bg = total_budget * (self._bg_budget_pct / 100.0)
            projection = usage_projection(
                self._drive_root, root_task_id="bg-consciousness",
            )
            accounted = float(projection.get("accounted_usd") or 0.0)
            self._bg_spent_usd = float(projection.get("settled_usd") or 0.0)
            return accounted < max_bg
        except Exception:
            log.warning("Failed to check background consciousness budget", exc_info=True)
            return False

    def _think(self) -> bool:
        """Bind each wakeup to the global ledger and its background sub-budget."""
        from ouroboros.usage_accounting import UsageScope, usage_scope

        total_budget = resolve_total_budget_usd()
        root_limit = total_budget * (self._bg_budget_pct / 100.0) if total_budget else None

        with usage_scope(UsageScope(
            drive_root=self._drive_root,
            task_id="bg-consciousness",
            root_task_id="bg-consciousness",
            category="consciousness",
            source="background_consciousness",
            global_limit_usd=total_budget,
            root_limit_usd=root_limit,
        )):
            return self._think_scoped()

    def _think_scoped(self) -> bool:
        """Run one context/LLM/tools cycle; False preserves skip/error status."""
        self._cycle_settlement_failed = False
        self._cycle_settlement_reasons = []
        self._cycle_ack_allowed = True
        if not hasattr(self, "_deferred_events"):
            self._deferred_events = []
        observation_snapshot = self._snapshot_pending_observations()
        try:
            context = self._build_cycle_context(observation_snapshot)
        except OverflowError as exc:
            # P1: skip the cycle rather than silently truncating cognitive context.
            log.warning("consciousness: wakeup cycle skipped: %s", exc)
            self._last_idle_reason = "context_overflow"
            append_jsonl(self._drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "consciousness_context_overflow",
                "error": str(exc),
            })
            return False
        model = self._model

        tools = self._tool_schemas()
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": "Wake up. Think."},
        ]
        _use_local_consciousness = os.environ.get(
            "USE_LOCAL_CONSCIOUSNESS", ""
        ).lower() in ("true", "1")
        effort = resolve_effort("consciousness")
        total_cost = 0.0
        cost_final = True
        cycle_usage: Dict[str, Any] = {}
        final_content = ""
        round_idx = 0
        all_pending_events = []

        try:
            target = (
                self._llm._resolve_remote_target(model)
                if not _use_local_consciousness else None
            )
            for round_idx in range(1, self._max_bg_rounds + 1):
                if self.is_paused:
                    self._cycle_ack_allowed = False
                    break
                if target is not None:
                    from ouroboros.openai_chat_dispatch import projected_context_size_bytes

                    physical_chars = projected_context_size_bytes(
                        messages,
                        tools,
                        provider=str(target.get("provider") or ""),
                        reasoning_effort=effort,
                    )
                    if physical_chars > BG_CONTEXT_MAX_CHARS:
                        error = (
                            "Background consciousness physical context too large "
                            f"({physical_chars:,} bytes including tools). "
                            "Groom memory to continue."
                        )
                        self._last_idle_reason = "context_overflow"
                        self._append_cycle_receipt(self._drive_root / "logs" / "events.jsonl", {
                            "ts": utc_now_iso(),
                            "type": "consciousness_context_overflow",
                            "error": error,
                        }, label="context overflow")
                        return False
                    if physical_chars > BG_CONTEXT_WARN_CHARS:
                        log.warning(
                            "consciousness: physical context is large "
                            "(%d bytes including tools)",
                            physical_chars,
                        )
                self._emit_live_log(
                    "llm_round_started",
                    round=round_idx,
                    attempt=1,
                    model=model,
                    reasoning_effort=effort,
                    use_local=bool(_use_local_consciousness),
                )
                from ouroboros.llm_observability import chat_observed

                msg, usage = chat_observed(
                    self._llm,
                    drive_root=self._drive_root,
                    task_id="consciousness",
                    call_type="consciousness_round",
                    messages=messages,
                    model=model,
                    tools=tools,
                    reasoning_effort=effort,
                    max_tokens=65536,
                    use_local=_use_local_consciousness,
                )
                from ouroboros.openai_chat_dispatch import (
                    custom_validation_by_call_id,
                    pop_custom_validation_receipts,
                )

                wire_validation = pop_custom_validation_receipts(
                    usage,
                    msg.get("tool_calls") or [],
                )
                validation_by_id = custom_validation_by_call_id(wire_validation)
                cost = float(usage["cost"]) if usage.get("cost") is not None else None
                if cost is None:
                    cost_final = False
                else:
                    total_cost += cost
                    self._bg_spent_usd += cost
                add_usage(cycle_usage, usage)

                # Global budget updates via events.py; direct updates would double-count.

                if not self._check_budget():
                    self._last_idle_reason = "budget_blocked"
                    self._cycle_ack_allowed = False
                    self._append_cycle_receipt(self._drive_root / "logs" / "events.jsonl", {
                        "ts": utc_now_iso(),
                        "type": "bg_budget_exceeded_mid_cycle",
                        "round": round_idx,
                    }, label="budget blocked")
                    break

                if self._event_queue is not None:
                    provider = "local" if _use_local_consciousness else str(usage.get("provider") or infer_provider_from_model(model))
                    resolved_model = str(usage.get("resolved_model") or model)
                    model_name = f"{model} (local)" if _use_local_consciousness else resolved_model
                    self._event_queue.put({
                        "type": "llm_usage",
                        "provider": provider,
                        "model": model_name,
                        "usage": usage,
                        "cost": cost,
                        "source": "consciousness",
                        "ts": utc_now_iso(),
                        "category": "consciousness",
                    })

                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                self._emit_live_log(
                    "llm_round_finished",
                    round=round_idx,
                    attempt=1,
                    model=model,
                    reasoning_effort=effort,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    cached_tokens=int(usage.get("cached_tokens") or 0),
                    cache_write_tokens=int(usage.get("cache_write_tokens") or 0),
                    cost_usd=cost,
                    response_kind="tool_calls" if tool_calls else "message",
                    tool_call_count=len(tool_calls),
                    has_text=bool(content.strip()),
                )

                self._emit_progress(content)

                if self.is_paused:
                    self._cycle_ack_allowed = False
                    break

                if content and not tool_calls:
                    final_content = content
                    break

                if tool_calls:
                    messages.append(msg)
                    for tc in tool_calls:
                        result = self._execute_tool(
                            tc,
                            all_pending_events,
                            validation_by_id.get(str(tc.get("id") or "")),
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": result,
                        })
                    continue

                break

            if all_pending_events and self._event_queue is not None:
                if self.is_paused:
                    self._deferred_events.extend(all_pending_events)
                else:
                    for evt in all_pending_events:
                        self._event_queue.put(evt)

            thought_written = self._append_cycle_receipt(self._drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "consciousness_thought",
                "thought_preview": (final_content or "")[:300],
                "cost_usd": total_cost if cost_final else None,
                "cost_final": cost_final,
                "rounds": round_idx,
                "model": model,
                **{
                    key: cycle_usage[key]
                    for key in (
                        "request_wire", "request_wire_history",
                        "request_wire_history_omitted",
                    )
                    if key in cycle_usage
                },
            }, label="thought receipt")
            # A paused/stopped cycle may have performed no complete cognition;
            # leave its snapshot pending for the next wake.  Acknowledgement is
            # the final durable step, after the thought receipt and all tool
            # state writes above have settled.
            if self.is_paused or self._stop_requested():
                self._cycle_ack_allowed = False
            if not final_content.strip():
                self._cycle_ack_allowed = False
            if not self.is_paused and not self._stop_requested():
                if (
                    not thought_written
                    or not self._cycle_ack_allowed
                    or not self._ack_observations(observation_snapshot)
                ):
                    self._last_idle_reason = "observation_ack_pending"
                    return False
            elif observation_snapshot:
                self._last_idle_reason = "observation_ack_pending"
                return False

        except Exception as e:
            self._cycle_ack_allowed = False
            self._emit_live_log("llm_round_error", round=round_idx, model=model, error=repr(e))
            self._append_cycle_receipt(self._drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "consciousness_llm_error",
                "error": repr(e),
            }, label="llm error")
            self._last_idle_reason = "llm_error"
            # Back off persistent provider/tool failures.
            self._next_wakeup_sec = min(self._next_wakeup_sec * 2, self._wakeup_max)
            return False

        return True

    def _emit_progress(self, content: str) -> None:
        if not content or not content.strip():
            return
        chat_id = self._owner_chat_id_fn()
        entry = {
            "type": "send_message",
            "chat_id": chat_id,
            "text": f"💬 {content.strip()}",
            "format": "markdown",
            "ts": utc_now_iso(),
            "task_id": "bg-consciousness",
            "content": content.strip(),
            "is_progress": True,
        }
        persist_locally = self._event_queue is None or chat_id is None
        if self._event_queue is not None and chat_id is not None:
            try:
                if self.is_paused:
                    self._deferred_events.append(entry)
                else:
                    self._event_queue.put(entry)
            except Exception:
                log.warning("Failed to emit progress event", exc_info=True)
                persist_locally = False
        if persist_locally:
            append_jsonl(self._drive_root / "logs" / "progress.jsonl", entry)

    def _build_cycle_context(
        self,
        observations: Sequence[Dict[str, Any]],
    ) -> str:
        """Call context builders across the pre-observation compatibility seam.

        A few overlap tests construct this class with ``object.__new__`` and
        provide an older zero-argument builder.  Inspect the callable before
        invoking it so a real ``TypeError`` raised *inside* a current builder
        is never mistaken for an unsupported keyword.
        """
        builder = self._build_context
        try:
            parameters = inspect.signature(builder).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        accepts_observations = any(
            parameter.name == "observations"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if accepts_observations:
            return builder(observations=observations)
        return builder()

    def _load_bg_prompt(self) -> str:
        """Load consciousness system prompt."""
        prompt_path = self._repo_dir / "prompts" / "CONSCIOUSNESS.md"
        if prompt_path.exists():
            return read_text(prompt_path)
        return "You are Ouroboros in background consciousness mode. Think."

    def _build_context(
        self,
        *,
        observations: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        from ouroboros.agent import Env
        env = Env(repo_dir=self._repo_dir, drive_root=self._drive_root)
        memory = Memory(drive_root=self._drive_root, repo_dir=self._repo_dir)
        bg_task = {"id": "bg-consciousness", "type": "consciousness"}
        # Per-cycle proof only.  A read from an older decision envelope cannot
        # authorize a later identity rewrite after the source changed.
        self._identity_source_requirements = {}
        self._identity_source_reads = {}
        self._identity_unresolved_sources = set()

        parts = [self._load_bg_prompt()]

        if not (self._repo_dir / "docs" / "ARCHITECTURE.md").is_file():
            logging.getLogger(__name__).warning(
                "consciousness: docs/ARCHITECTURE.md not found or empty"
            )
        parts.extend(build_governance_sections(env, warn_large=True, warn_label="consciousness"))

        durable_dialogue_gaps: List[Dict[str, Any]] = []
        parts.extend(build_memory_sections(
            memory, durable_dialogue_gaps_out=durable_dialogue_gaps,
        ))
        for gap in durable_dialogue_gaps:
            gap_id = str(gap.get("gap_id") or f"block-{gap.get('block_index', '?')}")
            self._identity_unresolved_sources.add(f"dialogue-gap:{gap_id}")

        parts.extend(
            build_knowledge_sections(
                env,
                warn_large=True,
                pattern_header="## Pattern Register",
            )
        )

        try:
            from ouroboros.improvement_backlog import (
                backlog_path,
                format_backlog_digest,
                load_backlog_items,
            )

            full_backlog_digest = format_backlog_digest(
                self._drive_root, limit=8, max_chars=2_147_483_647,
            )
            backlog_digest = format_backlog_digest(self._drive_root, limit=8, max_chars=4000)
            if backlog_digest:
                open_items = [
                    item for item in load_backlog_items(self._drive_root)
                    if str(item.get("status") or "open").lower() == "open"
                ]
                if len(open_items) > 8 or backlog_digest != full_backlog_digest:
                    self._identity_source_requirements["improvement-backlog"] = backlog_path(
                        self._drive_root
                    )
                    backlog_digest += (
                        "\n- complete_source: call knowledge_read(topic=\"improvement-backlog\") "
                        "and receive the complete current record before update_identity; "
                        "if unavailable, abstain from that rewrite"
                    )
                parts.append(backlog_digest)
        except Exception:
            try:
                from ouroboros.improvement_backlog import backlog_path

                path = backlog_path(self._drive_root)
                if path.exists():
                    self._identity_source_requirements["improvement-backlog"] = path
                    parts.append(
                        "## Improvement Backlog — source unavailable\n\n"
                        "The named current source could not be materialized. "
                        "Abstain from update_identity in this cycle."
                    )
            except Exception:
                pass
            log.debug("Failed to include improvement backlog in consciousness context", exc_info=True)

        health_section = build_health_invariants(env)
        if health_section:
            parts.append(health_section)

        # Full drive state: no clip_text here.
        state_json = safe_read(env.drive_path("state/state.json"), fallback="{}")
        if len(state_json) > BG_STATE_JSON_WARN_CHARS:
            log.warning(
                "consciousness: drive state JSON is large (%d chars)", len(state_json)
            )
        parts.append("## Drive state\n\n" + state_json)

        scheduled_tasks_digest: Dict[str, Any] = {}
        parts.append(build_runtime_section(
            env, bg_task, scheduled_tasks_digest_out=scheduled_tasks_digest,
        ))
        if int(scheduled_tasks_digest.get("omitted_count") or 0) > 0:
            self._identity_unresolved_sources.add("scheduled-tasks")

        # Empty task_id includes recent sections across tasks.  The typed facts
        # below are the exact same facts rendered into the decision envelope.
        recent_chat_coverage: Dict[str, Any] = {}
        parts.extend(build_recent_sections(
            memory, env, task_id="", chat_coverage_out=recent_chat_coverage,
        ))
        if (
            recent_chat_coverage.get("gaps")
            or int(recent_chat_coverage.get("omitted_matching_rows") or 0) > 0
            or bool(recent_chat_coverage.get("omitted_matching_rows_unknown"))
        ):
            self._identity_unresolved_sources.add("recent-chat")
        if observations is None:
            observations = self._snapshot_pending_observations()
        with self._observation_lock_for_instance():
            observation_state = self._read_observation_state()
        observation_gaps = list(observation_state.get("gap_reasons") or ())
        if observation_gaps:
            # The bounded observation view is still useful, but a malformed or
            # otherwise unreadable row means the actor has only a partial source.
            # Reuse the existing identity completeness envelope rather than
            # adding a second approval or policy gate.
            self._identity_unresolved_sources.add("background-observations")

        observation_rendered = ""
        if observations or observation_gaps:
            observation_rendered = self._render_observations(observations)
            if getattr(self, "_observation_projection_incomplete", False):
                # A bounded projection is useful for thought, but it is not a
                # complete source for a destructive identity rewrite.  The
                # existing actor-readable ref above is the recovery seam.
                self._identity_unresolved_sources.add("background-observations")

        if self._identity_unresolved_sources:
            parts.append(
                "## Identity update completeness\n\n"
                "Named unresolved source(s): "
                + ", ".join(sorted(self._identity_unresolved_sources))
                + ". Existing readers may inspect surviving data, but this cycle cannot "
                "prove complete unchanged sources; direct update_identity must abstain."
            )

        if observation_rendered:
            parts.append(observation_rendered)

        bg_info_lines = [
            f"BG budget spent: ${self._bg_spent_usd:.4f}",
            f"Current wakeup interval: {self._next_wakeup_sec}s",
            f"Current model: {self._model}",
        ]
        parts.append("## Background consciousness info\n\n" + "\n".join(bg_info_lines))

        # P1 guard: warn when large, fail the wakeup instead of truncating artifacts.
        _BG_TOTAL_WARN_CHARS = BG_CONTEXT_WARN_CHARS   # ~150K tokens — warn but proceed
        _BG_TOTAL_MAX_CHARS = BG_CONTEXT_MAX_CHARS  # ~300K tokens — fail fast (P1 compliance)
        full_text = "\n\n".join(parts)
        if len(full_text) > _BG_TOTAL_MAX_CHARS:
            log.warning(
                "consciousness: context too large (%d chars > %d limit) — "
                "skipping wakeup cycle; groom memory (knowledge, patterns, scratchpad) "
                "to reduce size",
                len(full_text), _BG_TOTAL_MAX_CHARS,
            )
            raise OverflowError(
                f"Background consciousness context too large ({len(full_text):,} chars). "
                "Groom memory to continue."
            )
        if len(full_text) > _BG_TOTAL_WARN_CHARS:
            log.warning(
                "consciousness: context is large (%d chars) — consider grooming memory",
                len(full_text),
            )
        return full_text

    _BG_TOOL_WHITELIST = frozenset({
        "send_user_message", "update_scratchpad",
        "update_identity", "set_next_wakeup",
        "knowledge_read", "knowledge_write", "knowledge_list",
        "web_search", "read_file", "list_files", "search_code", "query_code",
        "chat_history", "recent_tasks",
        "initiate_presence",
        "list_github_issues", "get_github_issue",
    })

    def _build_registry(self) -> "ToolRegistry":
        """Create a ToolRegistry scoped to background-allowed tools."""
        from ouroboros.tools.registry import ToolEntry, ToolRegistry

        registry = ToolRegistry(repo_dir=self._repo_dir, drive_root=self._drive_root)

        def _set_next_wakeup(ctx: Any, seconds: int = 300) -> str:
            self._next_wakeup_sec = max(self._wakeup_min, min(self._wakeup_max, int(seconds)))
            return f"OK: next wakeup in {self._next_wakeup_sec}s"

        registry.register(ToolEntry("set_next_wakeup", {
            "name": "set_next_wakeup",
            "description": "Set how many seconds until your next thinking cycle. "
                           f"Default 300. Range: {self._wakeup_min}-{self._wakeup_max} (clamped).",
            "parameters": {"type": "object", "properties": {
                "seconds": {"type": "integer",
                            "description": f"Seconds until next wakeup ({self._wakeup_min}-{self._wakeup_max})"},
            }, "required": ["seconds"]},
        }, _set_next_wakeup))

        return registry

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas filtered to the background whitelist."""
        return [
            s for s in self._registry.schemas()
            if s.get("function", {}).get("name") in self._BG_TOOL_WHITELIST
        ]

    def _execute_tool(
        self,
        tc: Dict[str, Any],
        all_pending_events: List[Dict[str, Any]],
        custom_validation: Any = None,
    ) -> str:
        """Execute a background tool call with timeout."""
        fn_name = tc.get("function", {}).get("name", "")
        if custom_validation is not None and not custom_validation.allows_execution:
            from ouroboros.openai_chat_dispatch import custom_tool_argument_error

            return custom_tool_argument_error(fn_name, custom_validation)
        if fn_name not in self._BG_TOOL_WHITELIST:
            return f"Tool {fn_name} not available in background mode."
        try:
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except (json.JSONDecodeError, ValueError):
            return "Failed to parse arguments."

        if fn_name == "update_identity":
            pending = list(self._identity_unresolved_sources)
            for topic, path in self._identity_source_requirements.items():
                try:
                    current_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception:
                    current_sha = ""
                if not current_sha or self._identity_source_reads.get(topic) != current_sha:
                    pending.append(topic)
            if pending:
                return (
                    "⚠️ IDENTITY_UPDATE_ABSTAINED: direct update_identity authority remains "
                    "available, but this decision envelope omitted named source(s) that are "
                    "not completely materialized in this cycle: " + ", ".join(sorted(pending))
                    + ". Materialize each resolvable source with its existing reader; "
                    "unresolved typed gaps require abstention."
                )

        self._emit_live_log(
            "tool_call_started",
            tool=fn_name,
            args=sanitize_tool_args_for_log(fn_name, args if isinstance(args, dict) else {}),
            timeout_sec=self._registry.get_timeout(fn_name),
        )

        chat_id = self._owner_chat_id_fn()
        self._registry._ctx.current_chat_id = chat_id
        self._registry._ctx.pending_events = []
        self._registry._ctx.event_queue = self._event_queue
        self._registry._ctx.task_id = "bg-consciousness"
        from ouroboros.tool_capabilities import BACKGROUND_DELEGATION_ROLE

        self._registry._ctx.task_metadata = {
            "root_task_id": "bg-consciousness",
            "session_id": "background-consciousness",
            "actor_id": "background-consciousness",
            # Owner-delivery gating keys on this role: BG frames must stay
            # cycle-end deferred (pause discipline), never live mid-cycle.
            "delegation_role": BACKGROUND_DELEGATION_ROLE,
        }

        timeout_sec = self._registry.get_timeout(fn_name)
        result = None
        error = None
        timed_out = False

        def _run_tool():
            nonlocal result, error
            try:
                result = self._registry.execute(fn_name, args)
            except Exception as e:
                error = e

        future = self._tool_executor.submit(_run_tool)
        try:
            future.result(timeout=timeout_sec)
        except (TimeoutError, concurrent.futures.TimeoutError):
            self._tool_executor.reset()
            timed_out = True
            self._cycle_ack_allowed = False
            result = f"[TIMEOUT after {timeout_sec}s]"
            self._emit_live_log(
                "tool_call_timeout",
                tool=fn_name,
                args=sanitize_tool_args_for_log(fn_name, args if isinstance(args, dict) else {}),
                timeout_sec=timeout_sec,
            )
            self._append_cycle_receipt(self._drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "consciousness_tool_timeout",
                "tool": fn_name,
                "timeout_sec": timeout_sec,
            }, label=f"tool timeout:{fn_name}")

        if error is not None:
            self._cycle_ack_allowed = False
            self._emit_live_log(
                "tool_call_finished",
                tool=fn_name,
                args=sanitize_tool_args_for_log(fn_name, args if isinstance(args, dict) else {}),
                is_error=True,
                result_preview=repr(error),
            )
            self._append_cycle_receipt(self._drive_root / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "consciousness_tool_error",
                "tool": fn_name,
                "error": repr(error),
            }, label=f"tool error:{fn_name}")
            result = f"Error: {repr(error)}"

        for evt in self._registry._ctx.pending_events:
            all_pending_events.append(evt)

        result_str = _truncate_tool_result(
            result,
            tool_name=fn_name,
            tool_args=args if isinstance(args, dict) else {},
        )

        if (
            fn_name == "knowledge_read"
            and error is None
            and not timed_out
            and isinstance(args, dict)
        ):
            topic = str(args.get("topic") or "").strip()
            path = self._identity_source_requirements.get(topic)
            if path is not None:
                try:
                    raw = path.read_bytes()
                    current = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                    if str(result) == current and result_str == current:
                        self._identity_source_reads[topic] = hashlib.sha256(raw).hexdigest()
                except Exception:
                    pass

        args_for_log = sanitize_tool_args_for_log(fn_name, args)
        if error is None and result is not None and not timed_out:
            self._emit_live_log(
                "tool_call_finished",
                tool=fn_name,
                args=args_for_log,
                is_error=False,
                result_preview=sanitize_tool_result_for_log(truncate_for_log(result_str, 500)),
            )
        self._append_cycle_receipt(self._drive_root / "logs" / "tools.jsonl", {
            "ts": utc_now_iso(),
            "tool": fn_name,
            "source": "consciousness",
            "args": args_for_log,
            "result_preview": sanitize_tool_result_for_log(truncate_for_log(result_str, 2000)),
        }, label=f"tool receipt:{fn_name}")

        return result_str


def compact_acknowledged_observations(
    drive_root: Any,
    retention_days: Optional[int] = None,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Fold old ACKNOWLEDGED observation rows into an archive segment (CPL4-C23).

    Contract: unacknowledged rows are NEVER pruned — they must survive
    restart and overflow verbatim. An acknowledged enqueue older than the
    unified GC retention moves, together with every ack row naming it, into
    ``archive/consciousness_observations_<ts>.jsonl`` (durable history,
    never GC'd). STRICTLY fail-closed: any malformed line, invalid row or
    ghost ack skips the whole fold — the same gap classes that block a live
    ack — and the archive segment is written BEFORE the live rewrite, so a
    crash can only duplicate rows into forensic history, never lose them.
    Runs at server startup, before Background Consciousness starts, under
    the store's own writer lock.
    """
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.retention import age_cutoff, get_gc_retention_days

    path = pathlib.Path(drive_root) / _OBSERVATIONS_REL
    report: Dict[str, Any] = {"folded": 0, "skipped": "", "archive": ""}
    if not path.exists():
        return report
    if retention_days is None:
        retention_days = get_gc_retention_days()
    cutoff = age_cutoff(retention_days, now)
    lock_path = jsonl_append_lock_path(path)
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0)
    if lock_fd is None:
        report["skipped"] = "lock_unavailable"
        return report
    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
        parsed: List[tuple] = []  # (raw_bytes, row_dict, identifier, is_ack)
        gap_reasons: List[str] = []
        enqueued_ids: set = set()
        acked_ids: set = set()
        for line_no, raw in enumerate(raw_lines, 1):
            stripped = raw.strip()
            if not stripped:
                report["skipped"] = "blank_line"
                return report
            try:
                row = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                report["skipped"] = "malformed_row"
                return report
            if not isinstance(row, dict) or not BackgroundConsciousness._validate_observation_row(
                row, line_no, gap_reasons,
            ) or gap_reasons:
                report["skipped"] = "invalid_row"
                return report
            identifier = str(row.get("id", row.get("observation_id")) or "")
            is_ack = row.get("op") == "ack"
            if is_ack:
                if identifier not in enqueued_ids:
                    report["skipped"] = "ghost_ack"
                    return report
                acked_ids.add(identifier)
            else:
                enqueued_ids.add(identifier)
            parsed.append((raw, row, identifier, is_ack))
        fold_ids: set = set()
        for _raw, row, identifier, is_ack in parsed:
            if is_ack or identifier not in acked_ids:
                continue
            enqueued_at = parse_deadline_ts(str(row.get("time") or row.get("observed_at") or ""))
            if enqueued_at is not None and enqueued_at.timestamp() < cutoff:
                fold_ids.add(identifier)
        if not fold_ids:
            return report
        keep: List[bytes] = []
        fold: List[bytes] = []
        for raw, _row, identifier, _is_ack in parsed:
            (fold if identifier in fold_ids else keep).append(raw)
        ts = utc_now_iso().replace("-", "").replace(":", "").split(".")[0]
        archive_dir = pathlib.Path(drive_root) / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        segment = archive_dir / f"consciousness_observations_{ts}.jsonl"
        suffix = 0
        while segment.exists():
            suffix += 1
            segment = archive_dir / f"consciousness_observations_{ts}_{suffix}.jsonl"
        # Archive FIRST: a crash between the two writes duplicates rows into
        # forensic history instead of destroying the owner's inbox.
        with segment.open("wb") as handle:
            handle.write(b"".join(fold))
            handle.flush()
            os.fsync(handle.fileno())
        tmp = path.with_name(path.name + ".compact.tmp")
        tmp.write_bytes(b"".join(keep))
        os.replace(tmp, path)
        report["folded"] = len(fold_ids)
        report["archive"] = segment.name
        return report
    except OSError:
        report["skipped"] = "io_error"
        return report
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)
