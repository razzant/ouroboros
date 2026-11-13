"""``/api/state``'s display publication: one producer and one encoded body per revision (cost law L6).

Every ``GET /api/state`` used to run the whole census (a queue copy under Q, a hot-fact snapshot, strict
document loads, usage reads, every held custody run, the project registry) in the default executor,
with no single-flight; a client that gave up left its build running and its retry started another.

Now the serving app holds ONE publication of its data root. A request with no refresh in flight starts one executor job
that probes the revision (``probe``: cheap reads) and, only when it differs from the held
publication's, produces the body (``gateway/state.py:_state_snapshot`` plus ``produced_fields``) and
encodes it once. Requests that arrive meanwhile await that same job; a cancelled waiter leaves without
starting anything, and the shared job finishes for the others (a thread is never abandoned midway;
its publication serves the next reader).

The revision is read BEFORE the production, so a body is never older than its key (a change landing
mid-production makes the next probe differ: one more production, never a stale body kept):

``queue``    the queue's revision (``supervisor.queue_publication.queue_revision``, bumped under Q by every
             mark, so a change is seen before the tick publishes it), the PENDING/RUNNING ids and
             each RUNNING row's live model activity, which the documents do not carry;
``stores``   ``meta.commit_seq`` of the data root's current store and of every other root the last
             production read (a task's ``budget_drive_root``); ``usage`` the usage ledger's
             ``publication_marker`` of the same roots;
``dialogue`` one aggregate over the dialogue store's rooms (unread/activity facts, Main's revision);
``bindings`` ``projects_registry.bindings_revision`` (bindings and registry files: the census names Projects too);
``files``    the identities of ``state.json`` (and its last-good copy), ``settings.json`` and the
             evolution campaign; ``checkout`` the runtime checkout's identity;
``live``     the in-memory census inputs no revision covers: the direct turns (``_direct_turns_source``,
             the one seam for their source), the post-task owners still running, the budget limit;
``presence`` the live Presence authors' witnesses (``conversation_lease.live_presence_roots``): who holds one,
             and whether every held one could be read (a released unreadable one makes the census whole again),
             and the cancel intents (``state/cancel_intents.json``): a recorded Stop a live turn has not taken yet
             is its work's ``stop_requested``, which no queue or store revision announces.

A component that cannot prove itself unchanged (a bindings file written within the clock tick, a
service file likewise) makes the key unequal to every other, so it is never reused. A store that
cannot be read is a value (its failure kind): the production reads it as unavailable too.

Process facts (uptime, workers, supervisor readiness, the boot document, served UI, runtime/context/
safety modes, effort range) are assembled per request without I/O and spliced in front of the encoded
body; ``task_bindings`` follows the request's ``bindings_since``, compared with the map's own digest (``map_revision``). ``boot.jobs`` is the store's job rows
as the last production read them: the in-memory value ``/api/health`` serves without a store read.

Display only: no mutation authority is derived from a publication (every door re-reads its own);
its generation and age ride the response headers. Residual: a fact that changes by clock passage
alone (the consciousness "due" wording, the rolling 24 h allowance) refreshes at the next revision move.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

GENERATION_HEADER = "X-Ouroboros-State-Generation"
AGE_HEADER = "X-Ouroboros-State-Age"


@dataclass(frozen=True)
class Publication:
    """One produced display body and the revision it was produced at."""

    key: Tuple[Any, ...]
    generation: int
    published_at: float
    body: bytes  # the encoded JSON object of every produced field
    bindings: bytes  # the encoded task_bindings map
    bindings_revision: str
    roots: frozenset = frozenset()  # other data roots this production read: the next probe stamps them too
    jobs: Dict[str, Any] = field(default_factory=dict)  # the store's named-job rows this production read


def encode(value: Any) -> bytes:
    """The same bytes ``JSONResponse`` renders (NaN refused, so a bad value fails as it did)."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")


def map_revision(bindings: Any) -> str:
    """The revision a reader holds its ``task_bindings`` map by: a digest of the bytes it was sent. The map itself
    names its revision, so it stands while those bytes would be the same, whatever else was written meanwhile (a
    Project touched, the serving process restarted), and it moves exactly when a reader's map would differ."""
    return hashlib.sha256(encode(bindings)).hexdigest()[:16]


def response_body(publication: Publication, live: Dict[str, Any], *, bindings_unchanged: bool) -> bytes:
    """The response bytes: the request's bindings answer and the per-request process facts (``live``) in front of
    the publication's encoded fields. Both sides always hold fields and their keys never meet (``produced_fields``
    owns the rest), so the join is the bytes of one object; no produced byte is decoded or encoded again."""
    head = encode({**live, "task_bindings_unchanged": bool(bindings_unchanged)})
    bindings = b"null" if bindings_unchanged else publication.bindings
    return b'{"task_bindings":' + bindings + b"," + head[1:-1] + b"," + publication.body[1:]


# ---- the revision ------------------------------------------------------------------------------

def _unprovable() -> object:
    """A component that cannot prove itself unchanged: a fresh object equals no other key."""
    return object()


def _failure(exc: BaseException) -> Tuple[str, str]:
    return ("unavailable", type(exc).__name__)


def _queue_revision() -> Tuple[Any, ...]:
    """The queue's revision, the live ids, and each RUNNING row's model activity, sent attempt context and start
    step (``turn_phase``, #2862: in-memory facts the documents omit). Read without Q: ``list()`` copies a dict or
    list whole."""
    from supervisor import queue
    from supervisor.queue_publication import queue_revision

    pending = [str(task.get("id") or "") for task in list(queue.PENDING) if isinstance(task, dict)]
    running = []
    for task_id, meta in list(queue.RUNNING.items()):
        active = (meta.get("active_llm_call") or {}) if isinstance(meta, dict) else {}
        running.append((str(task_id), meta.get("attempt") if isinstance(meta, dict) else None,
                        json.dumps([active.get("model_activity"), meta.get("context_attempt") if isinstance(
                            meta, dict) else None, meta.get("lead") if isinstance(meta, dict) else None,
                            meta.get("_lead_request_id") if isinstance(meta, dict) else None], sort_keys=True, default=str),
                        str(meta.get("turn_phase") or "") if isinstance(meta, dict) else ""))
    return (queue_revision(), tuple(pending), tuple(running))


def _store_revision(root: str) -> Any:
    from ouroboros import store

    try:
        with store.open_store(root, write=False) as txn:
            return txn.revision()
    except Exception as exc:
        return _failure(exc)


def _usage_revision(root: str) -> Any:
    from ouroboros import usage_store

    try:
        with usage_store.read(root, allow_stale=True) as txn:
            return tuple(txn.marker())
    except Exception as exc:
        return _failure(exc)


def _dialogue_revision(root: str) -> Any:
    """Every room's revision, read revision, last row and activity, folded: each only grows; and the generation
    and revision of the rooms' message times (``room_messages``), which dating the history moves alone."""
    from ouroboros import dialogue_store, room_messages

    try:
        with dialogue_store.transaction(root, write=False) as conn:
            record = conn.execute("SELECT count(*), total(revision), total(read_revision), max(last_seq), "
                                  "max(activity_at) FROM rooms").fetchone()
            dated = tuple(dialogue_store._meta(conn, key) for key in (room_messages.GENERATION_KEY,
                                                                      room_messages.REVISION_KEY))
        return tuple(record) + dated
    except Exception as exc:
        return _failure(exc)


def _bindings_revision(root: str) -> Any:
    from ouroboros.projects_registry import bindings_revision

    try:
        return bindings_revision(root) or _unprovable()
    except Exception as exc:
        return _failure(exc)


def _file_stamps(paths) -> Tuple[Any, ...]:
    """The identity of each service file the body reads; one changed within the clock tick proves nothing."""
    from ouroboros import utils

    started, stamps = time.time_ns(), []
    for path in paths:
        try:
            info = pathlib.Path(path).stat()
        except FileNotFoundError:
            stamps.append(None)
            continue
        except OSError as exc:
            stamps.append(_failure(exc))
            continue
        if max(info.st_mtime_ns, info.st_ctime_ns) >= started - utils._RACY_NS:
            return (_unprovable(),)
        stamps.append(utils.file_identity(info))
    return tuple(stamps)


def _service_files() -> Tuple[Any, ...]:
    from ouroboros import config
    from supervisor import state as supervisor_state
    from supervisor.evolution_lifecycle import _evolution_campaign_path

    return (supervisor_state.STATE_PATH, supervisor_state.STATE_LAST_GOOD_PATH, config.SETTINGS_PATH,
            _evolution_campaign_path())


def _direct_turns_source() -> Tuple[list, bool]:
    """The direct turns the census starts from, with whether the read was complete: the one seam for their
    source (today the process registry; the conversation lane's RUNNING rows replace it)."""
    from ouroboros.gateway import state as gateway_state

    availability: Dict[str, Any] = {"complete": True}
    rows = gateway_state._direct_turns_snapshot_safe(availability=availability)
    return rows, availability["complete"] is True


def _post_task_owners(root: str) -> Tuple[Any, ...]:
    """This process's post-task work on ``root`` still running, with each live owner's model waits."""
    from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK

    resolved = str(pathlib.Path(root).resolve(strict=False))
    with POST_TASK_SYNTHESIS_LOCK:
        owners = [(task, owner) for (path, task), owner in POST_TASK_SYNTHESIS_INFLIGHT.items() if path == resolved]
    facts = []
    for task, owner in sorted(owners, key=lambda item: str(item[0])):
        waits = None
        if owner is not None and not getattr(owner, "closed", False):
            try:
                waits = json.dumps(owner.snapshot(), sort_keys=True, default=str)
            except Exception as exc:
                waits = _failure(exc)
        facts.append((str(task), waits))
    return tuple(facts)


def _presence_authors(root: str) -> Any:
    """The author census the body reads beside the direct turns (``gateway.state._presence_turns_snapshot_safe``)."""
    from ouroboros.conversation_lease import live_presence_roots

    try:
        census = live_presence_roots(root)
        return tuple(sorted((str(row["task_id"]), bool(row["local"])) for row in census["roots"])), census["complete"]
    except Exception as exc:
        return _failure(exc)


def probe(drive_root: str, roots=frozenset(), *, direct: Optional[Tuple[list, bool]] = None) -> Tuple[Any, ...]:
    """The revision of everything the body reads (module docstring), read before the production."""
    from ouroboros.cancel_intents import _intents_path_read
    from ouroboros.config import REPO_DIR
    from ouroboros.gateway.state import _git_checkout_identity
    from supervisor.state import TOTAL_BUDGET_LIMIT

    every = sorted({str(drive_root), *map(str, roots)})
    direct_rows, complete = direct if direct is not None else _direct_turns_source()
    return (
        ("queue", _queue_revision()),
        ("stores", tuple((root, _store_revision(root)) for root in every)),
        ("usage", tuple((root, _usage_revision(root)) for root in every)),
        ("dialogue", _dialogue_revision(str(drive_root))),
        ("bindings", _bindings_revision(str(drive_root))),
        ("files", _file_stamps(_service_files())),
        ("checkout", _git_checkout_identity(REPO_DIR)),
        ("live", json.dumps(direct_rows, sort_keys=True, default=str), complete, _post_task_owners(str(drive_root)),
         TOTAL_BUDGET_LIMIT),
        ("presence", _presence_authors(str(drive_root)), _file_stamps([_intents_path_read(drive_root)])),
    )


# ---- the single-flight publisher -------------------------------------------------------------

class Publisher:
    """One data root's held publication and its one refresh in flight."""

    def __init__(self, drive_root: str) -> None:
        self.drive_root = str(drive_root)
        self.held: Optional[Publication] = None
        self._lock = threading.Lock()  # held/generation, between the executor job and the loop
        self._generation = 0
        self._flight: Optional[Tuple[asyncio.AbstractEventLoop, asyncio.Future]] = None
        self.productions = 0  # bodies produced in this process (the reuse is observable)

    def refresh(self, produce: Callable[..., Dict[str, Any]]) -> Publication:
        """Probe, then reuse the held publication or produce a new one (the executor job)."""
        held = self.held
        direct = _direct_turns_source()
        key = probe(self.drive_root, held.roots if held else frozenset(), direct=direct)
        if held is not None and held.key == key:
            return held
        produced = produce(direct)
        bindings = produced.pop("task_bindings") or {}
        with self._lock:
            self._generation += 1
            self.productions += 1
            publication = Publication(
                key=key, generation=self._generation, published_at=time.time(), body=encode(produced["fields"]),
                bindings=encode(bindings), bindings_revision=str(produced.get("task_bindings_revision") or ""),
                roots=frozenset(map(str, produced.get("roots") or ())) - {self.drive_root},
                jobs=dict(produced.get("jobs") or {}))
            self.held = publication
        return publication

    async def current(self, produce: Callable[..., Dict[str, Any]]) -> Publication:
        """The publication valid for the revision observed now: await the refresh in flight, else start one."""
        loop = asyncio.get_running_loop()
        flight = self._flight
        if flight is None or flight[0] is not loop or flight[1].done():
            future = asyncio.ensure_future(asyncio.to_thread(self.refresh, produce))
            self._flight = (loop, future)
            future.add_done_callback(_retrieve)
        else:
            future = flight[1]
        return await asyncio.shield(future)  # a cancelled waiter leaves; the shared refresh goes on


def _retrieve(future: asyncio.Future) -> None:
    """A refresh every waiter left still settles quietly; its failure is not an unretrieved one."""
    if not future.cancelled() and future.exception() is not None:
        log.debug("State publication refresh failed", exc_info=future.exception())


_ATTACH_LOCK = threading.Lock()


def publisher(app_state: Any, drive_root: Any) -> Publisher:
    """The serving app's publisher of ``drive_root``: one per app, so a second app (a test, a new generation's
    app) never reads another's publication; an app without a state object shares nothing."""
    root = str(pathlib.Path(drive_root).resolve(strict=False))
    if app_state is None:
        return Publisher(root)
    with _ATTACH_LOCK:
        held = getattr(app_state, "state_publication", None)
        if not isinstance(held, Publisher) or held.drive_root != root:
            held = Publisher(root)
            setattr(app_state, "state_publication", held)
        return held


def published_jobs(app_state: Any) -> Dict[str, Any]:
    """The store's named-job rows as the app's last publication read them (``{}`` before the first one)."""
    held = getattr(app_state, "state_publication", None) if app_state is not None else None
    publication = held.held if isinstance(held, Publisher) else None
    return dict(publication.jobs) if publication is not None else {}


# ---- the per-request process facts -----------------------------------------------------------

def _worker_counts() -> Tuple[int, int]:
    """(alive, total) of every worker slot of both lanes, project pool and conversation carriers, from this
    process's table (no I/O; a liveness check per process). Not the project ceiling or assignable capacity:
    the queue snapshot publishes those (``queue_publication._pool_counts``)."""
    try:
        from supervisor.workers import WORKERS

        workers = list(WORKERS.values())
        return sum(1 for worker in workers if worker.proc.is_alive()), len(workers)
    except Exception:
        return 0, 0


def live_fields(request: Any, jobs: Dict[str, Any]) -> Dict[str, Any]:
    """The process facts of ``StateResponse``, read per request without I/O: never part of the publication."""
    from ouroboros.config import get_context_mode, get_runtime_mode, get_safety_mode, get_skills_repo_path
    from ouroboros.gateway import state as gateway_state
    from ouroboros.settings_scales import effort_range

    supervisor_ready = gateway_state._state_attr(request, "supervisor_ready_event")
    get_supervisor_error = gateway_state._state_attr(request, "get_supervisor_error")
    app_start = float(gateway_state._state_attr(request, "app_start", time.time()) or time.time())
    alive, total = _worker_counts()
    return {
        "uptime": int(time.time() - app_start),
        "workers_alive": alive,
        "workers_total": total,
        "supervisor_ready": bool(supervisor_ready.is_set()) if supervisor_ready else False,
        "supervisor_error": get_supervisor_error() if callable(get_supervisor_error) else None,
        "boot": gateway_state._live_boot(request, jobs),  # the boot phases the header paints (Starting… → Online)
        "runtime_mode": get_runtime_mode(),
        "context_mode": get_context_mode(),
        # Frozen one-window compatibility field. Persistent auto-Low is retired.
        "context_mode_auto_low": False,
        "effort_range": effort_range(),  # the owner's range, the tolerant read
        "safety_mode": get_safety_mode(),
        "skills_repo_configured": bool(get_skills_repo_path()),
        "served_ui": gateway_state._served_ui_state(request),
    }


# ---- the produced fields ---------------------------------------------------------------------

def _accounting_block(accounting: Dict[str, Any], available: bool, limit: float) -> Dict[str, Any]:
    def money(name: str) -> Optional[float]:
        return float(accounting.get(name) or 0.0) if available else None

    return {
        "available": available,
        "authority": "physical_attempt_ledger",
        "settled_usd": money("settled_usd"),
        "confirmed_usd": money("confirmed_usd"),
        "estimated_usd": money("estimated_usd"),
        "reserved_usd": money("reserved_usd"),
        "unresolved_upper_bound_usd": money("unresolved_upper_bound_usd"),
        "accounted_usd": money("accounted_usd"),
        "unknown_unmetered": int(accounting.get("unknown_unmetered") or 0) if available else None,
        "cost_final": bool(accounting.get("cost_final")) if available else False,
        "integrity_degraded": bool(accounting.get("integrity_degraded")) if available else True,
        "attempt_counts": dict(accounting.get("attempt_counts") or {}),
        "limit_usd": limit,
        "remaining_known_usd": (float(accounting.get("remaining_known_usd") or 0.0)
                                if available and limit > 0 else None),
        **({"error_code": "ledger_unavailable"} if not available else {}),
    }


def produced_fields(snap: Dict[str, Any]) -> Dict[str, Any]:
    """Every ``StateResponse`` field the production owns, from one ``_state_snapshot``: all but the process
    facts and the bindings map, which ``live_fields`` and the request add at serve time."""
    from supervisor.state import STATE_READ_KEY, control_value

    st = snap["st"]
    runtime_branch, runtime_sha = snap["runtime_identity"]
    limit, spent, available = snap["limit"], snap["spent"], snap["accounting_available"]
    # A read that is not current never renders its controls as known (#1307).
    state_read = st.get(STATE_READ_KEY) if isinstance(st.get(STATE_READ_KEY), dict) else {}
    evolution_known, evolution_enabled = control_value(st, "evolution_mode_enabled")
    bg_known, bg_requested = control_value(st, "bg_consciousness_enabled")
    return {
        "pending_count": snap["pending_count"],
        "running_count": snap["running_count"],
        "spent_usd": round(spent, 4) if spent is not None else None,
        "budget_limit": limit,
        "budget_pct": round((spent / limit * 100) if limit > 0 else 0, 1) if spent is not None else None,
        # W2-F3: state values when the supervisor stamped them, else the actual runtime checkout
        # (source-mode installs never stamp state); a guessed branch is not identity.
        "branch": runtime_branch,
        "sha": (runtime_sha or "")[:8],
        "evolution_enabled": bool(evolution_enabled) if evolution_known else None,
        "bg_consciousness_enabled": bool(bg_requested) if bg_known else None,
        "state_quality": {"quality": str(state_read.get("quality") or "current"),
                          "source": str(state_read.get("source") or "primary"),
                          "unconfirmed": list(state_read.get("unconfirmed") or [])},
        "evolution_cycle": int(st.get("evolution_cycle") or 0),
        "evolution_state": snap["evolution_state"],
        "bg_consciousness_state": snap.get("bg_state") or {},
        "spent_calls": int(snap["breakdown"].get("physical_calls") or 0) if available else None,
        "subagents_mode": snap["subagents_mode"],
        "subagents_availability": snap["subagents_availability"],
        "main_route": snap["main_route"],
        "github_token_configured": snap["github_token_configured"],
        "accounting": _accounting_block(snap["accounting"], available, limit),
        "projects": snap["projects"],
        "project_messages_revision": snap.get("project_messages_revision"),
        "project_listable_count": snap.get("project_listable_count"),
        "project_chat_ids": snap["project_chat_ids"],
        "main_visible_revision": snap.get("main_visible_revision"),
        "task_bindings_revision": snap["task_bindings_revision"],
        "active_direct_turns": snap.get("active_direct_turns") or [],
        "active_chat_activities": snap.get("active_chat_activities") or [],
        "active_chat_activities_complete": snap.get("active_chat_activities_complete") is True,
        "current_questions": snap.get("current_questions") or [],
        "current_questions_complete": snap.get("current_questions_complete") is True,
        "current_questions_gaps": snap.get("current_questions_gaps") or [],
        "main_attention": snap.get("main_attention") or [],
        "main_attention_complete": snap.get("main_attention_complete") is True,
        "main_attention_gaps": snap.get("main_attention_gaps") or [],
        "works": snap.get("works") or [],
        "rooms": snap.get("rooms") or {},
        "work_summary": snap.get("work_summary") or {},
        "works_complete": snap.get("works_complete") is True,
        "works_gaps": snap.get("works_gaps") or [],
    }
