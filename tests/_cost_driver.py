"""Boot the real server over a scale root in THIS fresh interpreter and measure every boundary.

Run by tests/test_cost_contract.py as ``python -m tests._cost_driver SPEC.json OUT.json`` with
``OUROBOROS_DATA_DIR`` naming the root that tests/_scale_root.py built. The meter is installed
before production code is imported, so a measurement is cold by construction: nothing a
previous request, boot or test cached in memory is present.

Only what would leave the machine or start real work is replaced: the provider check says a
provider is ready (no model is called; a call would be recorded and refused), a name lookup of
any host but this one is refused and counted (``network_refused``), workers are not
spawned, the supervisor's git bootstrap (checkout reset + import-test child process) and the
boot-time update check do not run, the owned Claudexor daemon is not warmed, the starter-skill
installer does not run (the boot of this fresh root with a ready provider schedules it; it would
clone the skill sources from GitHub and install skills while the boundaries are measured; the
request is recorded as ``starter_requests``), and the host service listens on an ephemeral
loopback port. Everything else — the ASGI lifespan, the store import, the supervisor's init, the
gateway handlers, the WebSocket chat door and the Stop path — is production code.

Boundaries (tags in the output):

- ``B0``: the canonical path of the configured root (``install_identity.canonical_root``), resolved
  before the lifespan. It is a process-wide memo filled on first use, and the health request, the
  store import and the boot coordinator reach it concurrently, so the scheduler decides which
  boundary below would pay the walk over the root's ancestors. That order does not depend on the
  history. It is resolved once here and held to the same equality;
- ``B1``: the lifespan up to ``lifespan.startup.complete`` (the work ``/api/health`` waits for);
  ``B1h``: the first ``/api/health`` request; ``popen_before_health``: child processes on that
  path (the lifespan and the request); ``B1c``: the boot coordinator thread that runs the remaining
  boot steps after the HTTP yield, when the tree has one (joined once readiness is known);
- ``B2``: the threads that publish the store and initialize the supervisor (``thread_rules``) and
  every step that waits on the store's publication (``store_switch.when_ready``) wherever it runs,
  snapshotted when the chat bus is initialized (``B2_dialogue``) and when the supervisor reports
  ready (``B2_ready``); the supervisor is held there, before its first tick, while the requests
  below are measured. ``B2_child_headers`` separately counts the boot's required directory/header
  compatibility pass; both publication snapshots retain it for its own bounded-work guard;
- ``B4``/``B4w``: an owner chat frame over ``/ws`` on the cold process, then a second one;
- ``B3``/``B3w``: ``GET /api/state``, cold then warm;
- ``R_active``/``R_late`` (+``w``): ``GET /api/chat/history`` of the active and the late-bound room;
- ``B5``: ``POST /api/tasks/<root>/cancel`` with ``{"cascade": true}`` on the pending tree;
- ``P:*`` (B8, the producer and loop boundaries, each on its FIRST pass in this process):
  ``P:metrics_run`` a delegated run's measurement publication with the metrics cursor ABSENT,
  ``P:metrics_review`` a review publication with the cursor STALE (behind the first segment),
  ``P:send`` the loop's ``send_message`` of a terminal answer, ``P:allowance`` one consciousness
  tick's allowance read;
- ``D:<duty>:first|rest|work`` (B7, the rest law): every pass of the reconciled duty inventory
  (``server_maintenance.HOST_DUTIES``) and the delegated body's tick (``delegated-body``), each run
  on this thread under its own latch: ``first`` over the root as the boot left it, ``rest`` a second
  pass with the duty's own clock advanced (the body's 20 s retry window, a requested child look),
  ``work`` a third pass after one addressed input changed (``companion``: a run whose terminal was
  just observed, a late body for a ``result_absent`` owner, a settled task owing its terminal
  projection, an owed late-review row naming nothing, a scheduled child job, a question or a
  delivery still owed, an open cancel intent). A pass's count includes the threads it started, the
  pool work it submitted and the wakes it caused (a child look it requested, the early delivery
  replay), all joined or run here under its tag before the snapshot. The daemon is absent (every
  gateway read refuses typed), the maintenance child's spawn is recorded instead of started (its
  jobs run here, and what they report stays on the pass: ``child_reports``), and the body's driver
  thread stays parked while its tick is driven here;
- ``T:*`` (B10, last): one owner message as the supervisor loop handles it (the state read, the
  canonical inbound row, the routing pre-pass) and the production direct turn it starts, its body once
  that handling has returned (``_owner_turn``), counted
  up to the turn's first model request (``T:<tag>@request``), where the refused model call ends
  it (a turn the product ends before it keeps the supervisor log's record of how: ``ended``):
  ``T:main`` the process's first owner turn, ``T:mainw`` the next, ``T:main:foreign`` Main
  after a busy install's activity elsewhere, ``T:room`` a project room's turn; at k=1 also
  ``T:main:broad`` after 40 more live rooms and 10 running roots, ``T:main:rows`` after 650
  more open rows over those rooms and ``T:main:archive`` after 40 archived knowledge notes and
  40 old rescue snapshots (the turn after ``T:main:archive:first``, which meets them first). The
  history they must not reread is laid after the import, untagged, and the install brought to the
  steady state its maintenance child and its boot keep (the stored tail facts, the recorded OpenRouter
  catalogue, the archive listing; tests/_turn_shapes.py).

A spec's ``plan`` measures something else on the same booted process instead of these boundaries: ``live_set`` every recurring pass
as the live set grows (tests/_live_set.py), ``stand`` the production shape and the owner's turns over it (tests/_stand.py).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import json
import math
import os
import pathlib
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

from tests._cost_meter import BOUNDARY, CostMeter, tag_of, tagged
from tests._scale_root import CURRENT, force_store_journal, heartbeats, money, spend_limit_usd, terminal_detail

READY_TIMEOUT_SEC = 240.0  # the transport bound of this measurement, not a judgment about the boot
# The supervisor is held before its first tick, and the body's driver parked, for the whole measurement;
# the parent's child timeout (tests/test_cost_contract.py) is the bound that ends a runaway driver.
HOLD_TIMEOUT_SEC = 1800.0
STATEMENT_SAMPLE = 80
# B10: what a turn's request must carry (a row the step before it wrote), by tag.
NEEDLES: Dict[str, List[str]] = {}
# The clocks a measured process meets beside the work it counts: production bounds that the drive's own duration or the
# stand's age crosses, so they split identical work by elapsed time, never by history (counting is not timing). Each is
# held, before the server is imported, where the stand's intended state holds at every size: an install just restarted,
# its supervisor alive. A spec's ``clocks`` replaces any of them in its own unit: the instrument's controls drive the
# production values and see the work move.
CLOCKS: Dict[str, float] = {
    # ``utils``, ``skill_identity`` and ``skill_discovery_reuse`` (a copy made at import) ``_RACY_NS``, 2 s: a file read
    # within a clock tick of its write is read again, and each restore of the stand leaves its rewritten files that
    # young (#2961: B2's 16 reads of the project bindings, a stat each or an open each). A file's identity (inode, size,
    # nanosecond times) still names every write the process meets. Held below zero, not at it: with 0 a file written
    # in its reader's own tick still reads as just written where ``time.time_ns`` ticks at 15.6 ms (Windows), and how
    # many reads fall in that tick is timing (T:main's capability evidence: one or two more opens). It holds every
    # boundary through ``hold_clocks`` and the stand's turns; the live set resets only ``utils`` to 0 (``_live_set``).
    "racy_ns": -2_000_000_000,
    # ``task_status._SNAPSHOT_OWNERSHIP_FRESH_SEC``, 10 s: the held supervisor never refreshes its queue stamp, so a
    # pass that outlasts it takes every queued task for live (B7: three evidence tasks wait, 336 VM steps and 16
    # statements each).
    "queue_alive_fresh_sec": math.inf,
    # ``restore_pending_from_snapshot``'s ``max_age_sec``, 900 s: the boot restores the stand's queue only while the
    # fixture's stamp is younger, so a stand restored 15 minutes after its build boots another way (B2, B3, B5, the
    # cancel sweep's first pass).
    "restore_max_age_sec": math.inf,
}


def hold_clocks(replaced: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Set ``CLOCKS`` (with ``replaced`` over them) where the product reads them; returns what was held, as the result
    records it: JSON has no infinity, so an unbounded clock is written ``"inf"``, which a spec may hand back."""
    from ouroboros import skill_discovery_reuse, skill_identity, task_status, utils
    from supervisor import queue

    clocks = {name: float(value) if isinstance(value, str) else value  # a recorded "inf" handed back
              for name, value in {**CLOCKS, **(replaced or {})}.items()}
    utils._RACY_NS = skill_identity._RACY_NS = skill_discovery_reuse._RACY_NS = int(clocks["racy_ns"])
    task_status._SNAPSHOT_OWNERSHIP_FRESH_SEC = clocks["queue_alive_fresh_sec"]
    restore = getattr(queue.restore_pending_from_snapshot, "func", queue.restore_pending_from_snapshot)
    queue.restore_pending_from_snapshot = functools.partial(restore, max_age_sec=clocks["restore_max_age_sec"])
    return {name: value if math.isfinite(value) else str(value) for name, value in clocks.items()}


class _FirstRequest(BaseException):
    """The turn reached its first model request: its count is taken there and the turn ends (test-only)."""


def refused(report: Any, *, rest: bool = False) -> List[str]:
    """What a maintenance pass's own report (``maintenance.run``) says it did not do: the pass or a job failed (a job
    raised, or returned ``failed``), or there is no report (its thread died; the child process reports a raised pass
    as ``failed``: ``maintenance.main``). A refused job did no work, so its cheapness proves nothing (#2465: a pass
    pointed at another install has every job refused). ``pending`` (contention or resumable work, alike) is owed
    work, not refused, except at ``rest``: there every job that ran finished (``done``); a resting job is absent."""
    if not isinstance(report, dict):
        return [f"no report: {report!r}"]
    raised = f" ({report['raised']})" if report.get("raised") else ""
    out = [] if report.get("status") == "completed" else [f"pass {report.get('status')}{raised}"]
    return out + [f"{name}: {(outcome or {}).get('state')}" for name, outcome in (report.get("jobs") or {}).items()
                  if (outcome or {}).get("state") == "failed" or (rest and (outcome or {}).get("state") != "done")]


def name_owed(where: str, *reports: Any) -> None:
    """Work the reports left unfinished where that is no refusal (outside a counted rest): named as debt."""
    if owed := [line for report in reports for line in refused(report, rest=True)]:
        warnings.warn(f"{where}: owed work left pending {owed}", stacklevel=2)


def schedule_child_job(root: pathlib.Path) -> str:
    """The maintenance child's addressed input change: the job its boot schedules (``legacy_reconcile``) is pending
    again, its cursor kept, so its run resumes past every drive and records it done. Returns the job's name."""
    from ouroboros import maintenance_jobs

    maintenance_jobs.record(root, "legacy_reconcile", state="pending")
    return "legacy_reconcile"


def _summary(meter: CostMeter, tag: str) -> Dict[str, Any]:
    snap = meter.snapshot(tag)
    statements = sorted(snap["statements"].items(), key=lambda item: -item[1])[:STATEMENT_SAMPLE]
    return {"counts": snap["counts"], "paths": snap["paths"], "names": snap["names"],
            "statements": dict(statements), "callers": snap["callers"]}


def _all_counts(meter: CostMeter) -> Dict[str, Dict[str, int]]:
    return {tag: meter.snapshot(tag)["counts"] for tag in meter.tags()}


class _Tagger:
    """ASGI wrapper: the lifespan runs as ``B1``; a request names its boundary in a header."""

    def __init__(self, app: Any, meter: CostMeter, result: Dict[str, Any]) -> None:
        self.app, self.meter, self.result = app, meter, result

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            # Work-count fixture, not a contention benchmark: consecutive requests
            # must share the per-thread connection cache for cold/warm to be stable.
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
            token = BOUNDARY.set("B1")

            async def tagged_send(message):
                if message.get("type") == "lifespan.startup.complete":
                    _mark(self.result, "lifespan_complete")
                    self.result["B1"] = _summary(self.meter, "B1")
                    self.result["process_at_B1"] = _all_counts(self.meter)
                return await send(message)

            try:
                return await self.app(scope, receive, tagged_send)
            finally:
                BOUNDARY.reset(token)
        headers = dict(scope.get("headers") or [])
        tag = headers.get(b"x-cost-boundary", b"").decode() or None
        if tag is None:
            return await self.app(scope, receive, send)
        token = BOUNDARY.set(tag)
        try:
            return await self.app(scope, receive, send)
        finally:
            BOUNDARY.reset(token)


def _stub_external_effects(srv: Any, result: Dict[str, Any], meter: CostMeter) -> None:
    import socket

    from ouroboros import claudexor_daemon, llm, skill_starter
    from supervisor import workers

    srv.has_startup_ready_provider = lambda _settings: True
    srv._bootstrap_supervisor_repo = lambda _settings, git_ops_module=None: (True, "cost contract: no git bootstrap")
    srv._boot_managed_update_tasks = lambda: None
    workers.spawn_workers = lambda *_a, **_k: None
    claudexor_daemon.warm_owned_daemon = lambda *_a, **_k: None
    skill_starter.schedule_starter = lambda _root, _ids, *, origin, repo_dir=None: (
        result.setdefault("starter_requests", []).append(origin))

    def refuse_model(*args, **kwargs):
        tag = tag_of()
        result.setdefault("model_calls", []).append(tag)
        if tag.startswith("T:") and f"{tag}@request" not in result:
            # The owner's wait ends at the turn's first model request: the count is taken there; the turn
            # then unwinds through its finally blocks (a BaseException passes its failure handlers).
            request = json.dumps([args[1:], kwargs], default=str, ensure_ascii=False)
            result[f"{tag}@request"] = {**_summary(meter, tag),
                                        "carries": [needle for needle in NEEDLES.get(tag, ()) if needle in request]}
            _mark(result, f"{tag}@request")
            raise _FirstRequest(tag)
        raise RuntimeError("cost contract: no model call is expected on these boundaries")

    llm.LLMClient.chat = refuse_model
    llm.LLMClient.chat_async = refuse_model
    socket.getaddrinfo = loopback_only(meter, socket.getaddrinfo)


def loopback_only(meter: CostMeter, resolve: Callable[..., Any]) -> Callable[..., Any]:
    """``socket.getaddrinfo`` for the driver: this machine's names resolve; any other host is refused and
    counted under the boundary that asked (``network_refused``), so nothing measured here leaves the machine."""
    def lookup(host, *args, **kwargs):
        if host not in (None, "", "localhost", "::1") and not str(host).startswith("127."):
            meter.note_name("network_refused", str(host))
            raise OSError(f"cost contract: no network ({host})")
        return resolve(host, *args, **kwargs)
    return lookup


class _Park:
    """The body's driver thread enters its tick and waits here; the pass drives the real tick itself."""

    def __init__(self, tick: Callable[..., Any]) -> None:
        self.tick, self.release = tick, threading.Event()

    def parked(self) -> Callable[[Any], None]:
        return lambda _runtime: self.release.wait(HOLD_TIMEOUT_SEC)


class _Clock:
    """``time`` for ``delegate_collection``: a monotonic clock the pass advances past the body's retry window."""

    def __init__(self) -> None:
        self.now = time.monotonic()

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class _NoProcess:
    pid = 0

    def poll(self) -> int:
        return 0

    def kill(self) -> None:
        pass


def _stub_duty_effects(result: Dict[str, Any]) -> Dict[str, Any]:
    """What the duty passes must not reach: the daemon (every gateway read refuses typed, as an
    install without one answers), the maintenance child's process (its spawn is recorded), the body's
    own driver clock and the lifespan's reconcile pickup (parked; a pass drives them and advances the
    clock itself) and a process behind the event bus."""
    from ouroboros import claudexor_daemon, delegate_collection, extension_reconcile_queue, maintenance
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from supervisor import workers
    from supervisor.event_observation import ObservedEventQueue

    async def pickup_parked(*_args, **_kwargs):
        await asyncio.Event().wait()  # until the lifespan cancels it: a pass measures the markers, not a race with its loop

    extension_reconcile_queue.extension_reconcile_pickup_loop = pickup_parked

    def no_daemon(*_args, **_kwargs):
        raise ClaudexorUnavailable("daemon_unreachable", "cost contract: no daemon on these boundaries")

    # The bus the spawned pool would have created: without it the duties that hand work to it fail before their first
    # read, and a duty that cannot run rests by default. Nothing drains it here: the supervisor is held.
    workers._EVENT_Q = ObservedEventQueue()
    claudexor_daemon.read_owned_gateway = no_daemon
    claudexor_daemon.ensure_owned_gateway = no_daemon
    maintenance.spawn_maintenance_job = lambda data_root, repo_dir, purpose, **_k: (
        result.setdefault("child_spawns", []).append(tag_of()) or _NoProcess())
    maintenance.finish_maintenance_job = lambda *_args, **_kwargs: None
    park, clock = _Park(delegate_collection.BodyRuntime.tick), _Clock()
    delegate_collection.BodyRuntime.tick = park.parked()
    delegate_collection.time = clock
    return {"park": park, "clock": clock}


def _mark(result: Dict[str, Any], name: str) -> None:
    result.setdefault("marks", {})[name] = round(time.monotonic() - result["_started"], 3)


def _install_checkpoints(meter: CostMeter, srv: Any, result: Dict[str, Any], root: pathlib.Path) -> Dict[str, threading.Event]:
    import functools

    from ouroboros import owned_shutdown, server_liveness, store_journal, store_switch, usage_store
    from supervisor import message_bus
    from tests._cost_meter import tagged

    events = {"ready": threading.Event(), "release": threading.Event()}
    # B1c's usage import enumerates state/ for unfinished usage.sqlite imports. Hold the two background
    # writers until it finishes: otherwise that census may meet current.sqlite-wal/-shm and
    # owned_processes.json (or the writers' temporary files). They still run and count under their own tags.
    usage_imported = threading.Event()

    def after_usage_import(call):
        def run(*args, **kwargs):
            usage_imported.wait()
            return call(*args, **kwargs)
        return run

    store_switch._run = after_usage_import(store_switch._run)
    owned_shutdown.import_inherited_records = after_usage_import(owned_shutdown.import_inherited_records)
    migrate_usage = usage_store.migrate_from_journal

    def usage_first(*args, **kwargs):
        try:
            return migrate_usage(*args, **kwargs)
        finally:
            usage_imported.set()

    usage_store.migrate_from_journal = usage_first
    when_ready = store_switch.when_ready
    prepare_children = store_journal.prepare_children

    def child_headers(root):
        # Compatibility must precede producers. Measure its directory/header work
        # separately; test_cost_contract bounds it and still forbids history payloads.
        return tagged("B2_child_headers", lambda: prepare_children(root))

    store_journal.prepare_children = child_headers

    def publication_cost():
        return {**_summary(meter, "B2"), "child_headers": _summary(meter, "B2_child_headers")}

    def publication_steps(root, *steps):
        # What waits on the store's publication is B2 wherever it runs (the product's ``when_ready`` runs
        # it on a background thread, the import's or ``store-ready-steps``, never the caller's).
        result.setdefault("when_ready_threads", []).append(threading.current_thread().name)
        return when_ready(root, *[functools.partial(tagged, "B2", step) for step in steps])

    store_switch.when_ready = publication_steps
    bus_init = message_bus.init

    def init(*args, **kwargs):
        value = bus_init(*args, **kwargs)
        _mark(result, "dialogue")
        result["B2_dialogue"] = publication_cost()
        result["B2_dialogue_tag"] = BOUNDARY.get() or getattr(threading.current_thread(), "_cost_tag", None)
        return value

    message_bus.init = init
    noted = server_liveness.note_supervisor_ready

    def ready(*args, **kwargs):
        _mark(result, "ready")
        result["B2_ready"] = publication_cost()
        result["B2_ready_tag"] = getattr(threading.current_thread(), "_cost_tag", None)
        result["B2_ready_flag"] = srv._supervisor_ready.is_set()
        events["ready"].set()
        events["release"].wait(HOLD_TIMEOUT_SEC)  # held before the first tick while the boundaries are measured
        return noted(*args, **kwargs)

    server_liveness.note_supervisor_ready = ready
    recover = srv._run_startup_task_recovery

    def recover_live(*args, **kwargs):
        # The root holds the previous process's running tasks as live ones, and counting is not timing: a row older than
        # task_status._ORPHAN_RUNNING_GRACE_SECONDS sends the boot of one size down the stale-orphan check (it parses the
        # events tail) and not the other's. The bench stamps their heartbeats as its own step, at the doorstep of the
        # recovery that reads them; the check, the grace and the tail read stay the product's.
        tagged("heartbeats", lambda: heartbeats(root))
        return recover(*args, **kwargs)

    srv._run_startup_task_recovery = recover_live
    return events


def _request(meter: CostMeter, client: Any, result: Dict[str, Any], tag: str, method: str, path: str,
             **kwargs: Any) -> None:
    response = client.request(method, path, headers={"x-cost-boundary": tag}, **kwargs)
    meter.join_spawned(tag, timeout=10.0)
    _mark(result, tag)
    result[tag] = _summary(meter, tag)
    result.setdefault("responses", {})[tag] = {"status": response.status_code, "bytes": len(response.content)}
    if tag in {"B3", "B3w"}:
        payload = response.json()
        result["responses"][tag]["decisions"] = {
            "current_questions": sorted(
                [{key: value for key, value in row.items() if key != "ts"} for row in payload["current_questions"]],
                key=lambda row: (row["task_id"], row["quiz_id"])),
            "current_questions_complete": payload["current_questions_complete"],
            "current_questions_gaps": payload["current_questions_gaps"],
        }


def _chat_frame(meter: CostMeter, client: Any, result: Dict[str, Any], tag: str, message_id: str) -> None:
    frame = {"type": "chat", "content": f"owner message {message_id}", "chat_id": 1,
             "client_message_id": message_id, "sender_session_id": "cost-contract"}
    seen: List[str] = []
    with client.websocket_connect("/ws", headers={"x-cost-boundary": tag}) as socket:
        socket.send_text(json.dumps(frame))
        while True:
            data = json.loads(socket.receive_text())
            seen.append(str(data.get("type") or ""))
            if data.get("client_message_id") == message_id:
                result.setdefault("responses", {})[tag] = {"echo": data.get("system_type") or data.get("role") or "chat",
                                                           "frames": seen}
                break
    meter.join_spawned(tag, timeout=10.0)
    _mark(result, tag)
    result[tag] = _summary(meter, tag)


# ---- B8: producer and loop boundaries, B7: the duties ----------------------------------------------

def _event_ctx(srv: Any) -> Any:
    """The supervisor's event context as ``server._run_supervisor`` builds it, from the same owners, the
    production direct turn included."""
    from ouroboros.utils import append_jsonl
    from supervisor import queue, workers
    from supervisor.message_bus import send_with_budget, try_get_bridge
    from supervisor.state import load_state, save_state, update_state
    from supervisor.worker_chat_lane import handle_chat_direct

    return SimpleNamespace(
        DRIVE_ROOT=srv.DATA_DIR, REPO_DIR=srv.REPO_DIR, bridge=try_get_bridge(), WORKERS=workers.WORKERS,
        PENDING=queue.PENDING, RUNNING=queue.RUNNING, send_with_budget=send_with_budget,
        load_state=load_state, save_state=save_state, update_state=update_state,
        append_jsonl=append_jsonl, enqueue_task=queue.enqueue_task, persist_queue_snapshot=queue.persist_queue_snapshot,
        sort_pending=queue.sort_pending, consciousness=None, handle_chat_direct=handle_chat_direct,
        request_restart=lambda *_a, **_k: None, kill_workers=lambda *_a, **_k: None,
        spawn_workers=lambda *_a, **_k: None, safe_restart=lambda *_a, **_k: None)


def _boundary(meter: CostMeter, result: Dict[str, Any], tag: str, call: Callable[[], Any], *,
              timeout: float = 180.0) -> None:
    try:
        response = tagged(tag, call)
    except Exception as exc:  # the boundary's own refusal is a fact of the measurement, not a driver error
        response = f"{type(exc).__name__}: {exc}"
    alive = meter.join_spawned(tag, timeout=timeout)
    if alive:
        result["errors"].append(f"{tag}: threads still alive after {timeout}s: {alive}")
    _mark(result, tag)
    result[tag] = _summary(meter, tag)
    result.setdefault("responses", {})[tag] = {"status": 200 if not isinstance(response, str) else 500,
                                               "echo": response if isinstance(response, str) else ""}


def _stale_cursor(root: pathlib.Path) -> None:
    """A metrics cursor behind the first segment of the event chain: the suffix after it is k-sized."""
    from ouroboros import store
    from ouroboros.jsonl_tail import JsonlChainSnapshot
    from ouroboros.subagent_metrics import CURSOR

    through = JsonlChainSnapshot(root / "logs/events.jsonl").ends[0]
    with store.open_store(root, write=True) as txn:
        txn.set_meta(CURSOR, {"through": through, "source_root": str(root.resolve()), "missing_producers": [],
                              "coverage": {"state": "observed_window", "stream": "incomplete"}})


def _publish_run(root: pathlib.Path, run_id: str) -> None:
    from ouroboros import delegate_custody_current as current
    from ouroboros.subagent_history import record_session_execution

    entry = current.run(root, run_id)
    record_session_execution(root, entry, terminal_detail(run_id), {"model": "test", "profile_id": ""})


def _publish_review(root: pathlib.Path, task_id: str) -> None:
    from ouroboros.subagent_metric_events import review_execution

    request = SimpleNamespace(surface="code_review", task_id=task_id)
    slot = SimpleNamespace(route=SimpleNamespace(value="api_chat"), model="test", slot_id="seat-1", subagent_id="",
                           session_target="", session_profile="", effort="", processing_preference="", session_access="")
    actor = SimpleNamespace(outcome={"outcome_id": "outcome-cost-1", "operation_id": "op-cost-1", "status": "responded",
                                     "parse": "valid", "transport": "success", "source_ref": {"task_id": task_id},
                                     "usage_ref": {}},
                            usage={}, raw_text="LGTM", duration_sec=1.5, failure_code="", reset_at=None,
                            operation_id="op-cost-1", awaiting_since=None)
    review_execution(request, slot, actor, SimpleNamespace(), fallback_root=root)


def _send_answer(ctx: Any, task_id: str, text: str) -> Any:
    from supervisor.events_chat_delivery import _handle_send_message
    from supervisor.terminal_delivery import delivery_id_for

    return _handle_send_message({"type": "send_message", "chat_id": 1, "task_id": task_id, "text": text,
                                 "delivery_id": delivery_id_for(task_id, text), "role": "assistant",
                                 "terminal_origin": "model_final"}, ctx)


def _measure_boundaries(meter: CostMeter, srv: Any, result: Dict[str, Any], root: pathlib.Path,
                        facts: Dict[str, Any]) -> None:
    from ouroboros.consciousness_allowance import allowance_window

    ctx = _event_ctx(srv)
    held = facts["held_runs"]
    _boundary(meter, result, "P:metrics_run", lambda: _publish_run(root, held[0]["run_id"]))
    _stale_cursor(root)
    _boundary(meter, result, "P:metrics_review", lambda: _publish_review(root, held[1]["task_id"]))
    _boundary(meter, result, "P:send", lambda: _send_answer(ctx, CURRENT["running"][0], "the final answer of the cost contract"))
    _boundary(meter, result, "P:allowance", lambda: allowance_window(root))


def _addressed_stream(root: pathlib.Path, result: Dict[str, Any]) -> None:
    """``B9``: the memory's addressed chat reader over the whole stream (``chat_chain.iter_rows``: every
    room's open rows come through it, and Main's view reads them on each turn). It reads rows in proportion
    to the stream it is asked for; tests/test_cost_contract.py holds its read transactions to a few a page."""
    from ouroboros import chat_chain

    result["B9_rows"] = sum(1 for _entry in chat_chain.iter_rows(root))


# ---- B10: the owner's turn, from the loop's handling of the message to the turn's first model request ----

_UPDATE_IDS = iter(range(9_000_000, 10_000_000))


def no_carrier_can_start() -> None:
    """No worker starts in this measurement, and a turn's body must run where it is counted: the physical fact the
    conversation lane's disclosed degraded carrier answers to (a tree without the lane runs every turn here anyway). Called
    once the boot is over: a pool generation starts with no refusal on record."""
    try:
        from supervisor import conversation_lane
    except ImportError:
        return
    conversation_lane._SPAWN_REFUSED[0] = "cost contract: no worker process here"


def _lane_rows_ended() -> None:
    """A turn this driver ended at its first request sends no ``task_done``: its lane row, if the tree has lanes, ends here,
    or every later turn would count it among the live roots."""
    try:
        from supervisor.conversation_lane import conversation_row
    except ImportError:
        return
    from supervisor import queue

    with queue._queue_lock:
        for task_id, meta in list(queue.RUNNING.items()):
            if conversation_row(meta):
                queue.RUNNING.pop(task_id, None)


def _owner_turn(srv: Any, ctx: Any, chat_id: int, message_id: str) -> int:
    """One owner message as the supervisor loop handles it (``server._handle_bridge_update_batch``): the state
    read, the canonical inbound row, the routing pre-pass and the direct turn it starts on its own thread, which
    inherits this boundary up to its first model request. The turn's body runs once that handling has returned. A
    live install runs the body in a lane worker, a process whose memos the loop's reads never fill; no worker starts
    here, so it runs on an inline carrier's thread of this process, and nothing else would order the loop's last
    records of the new turn against the body's first writes of its task: whether the loop's lineage read finds the
    task absent, as its placeholder (``create_only``, no room yet) or admitted, and whether it runs before the turn's
    request at all, would be the scheduler's choice, and this process keeps what that read found
    (``projects_registry._TASK_LINEAGE``). The hold is class-wide: any carrier built while the handling runs waits
    for it, not only the turn's; in the stand no other carrier is built then. Neither wait has a timeout: the handling
    must not wait on the body it starts, and the usage hold (``_install_checkpoints``) assumes no boot step before the
    usage import waits for the store's publication; both hold in the product today."""
    from supervisor.inline_carrier import InlineCarrier

    update = {"update_id": next(_UPDATE_IDS), "message": {
        "chat": {"id": chat_id}, "from": {"id": 7}, "text": f"owner turn {message_id}", "source": "web",
        "client_message_id": message_id, "sender_session_id": "cost-contract"}}
    handled, carrier = threading.Event(), InlineCarrier.__init__

    def after_handling(self: Any, target: Callable[[], None], *args: Any, **kwargs: Any) -> None:
        def body() -> None:
            handled.wait()
            target()
        carrier(self, body, *args, **kwargs)

    InlineCarrier.__init__ = after_handling  # any carrier built while the handling runs starts its body after it
    try:
        return srv._handle_bridge_update_batch(ctx.bridge, [update], 0, ctx, [0])
    finally:
        InlineCarrier.__init__ = carrier
        handled.set()


def _turn_ends(log: pathlib.Path, start: int) -> List[Dict[str, Any]]:
    """The direct turns' ends the supervisor log recorded after byte ``start`` (a refused turn's
    ``direct_chat_error`` with its error: ``worker_chat_lane._report_direct_chat_error``), read after the boundary."""
    rows: List[Any] = []
    if not log.exists():
        return rows
    with open(log, "rb") as handle:
        handle.seek(start)
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return [{key: row.get(key) for key in ("type", "error")} for row in rows
            if isinstance(row, dict) and str(row.get("type") or "").startswith("direct_chat_")]


def _view_receipt(root: pathlib.Path) -> Dict[str, Any]:
    """The newest ``context_memory_view`` receipt of ``logs/events.jsonl``, read after the turn that wrote it (one
    build per turn up to its first model request): the floor it chose and the work it counted (``work``)."""
    receipt: Dict[str, Any] = {}
    path = root / "logs" / "events.jsonl"
    if path.exists():
        with open(path, "rb") as handle:
            for line in handle:
                if b'"context_memory_view"' in line:
                    with contextlib.suppress(ValueError):
                        row = json.loads(line)
                        receipt = row if row.get("type") == "context_memory_view" else receipt
    floor = receipt.get("floor") or {}
    return {"work": receipt.get("work") or {}, "mode": floor.get("mode"), "mode_switch": floor.get("mode_switch"),
            "steps": floor.get("steps") or {}, "window_tokens": floor.get("window_tokens"),
            "blocks": receipt.get("blocks") or {}, "max_room": receipt.get("max_room") or {}}


def _boot_census(srv: Any, root: pathlib.Path) -> None:
    """What precedes the owner's first turn on a live install and is not that turn's: the boot's first worker
    verifies the server generation and stamps it (no worker is spawned here), and the health block's stray-server
    probe holds its answer for 15 minutes. Both count the machine's processes, one ``ps`` per process named like
    an install (#2025's census), which follow the host, not the history (tests/test_cost_host.py holds that axis)."""
    from ouroboros import agent, context_health

    env = agent.Env(repo_dir=pathlib.Path(srv.REPO_DIR), drive_root=root)
    agent.OuroborosAgent._log_worker_boot_once(SimpleNamespace(env=env))
    agent._worker_boot_logged = False  # this process's own first turn still logs its boot and reads the stamp
    context_health._stray_server_note(env)


def _measure_turns(meter: CostMeter, srv: Any, result: Dict[str, Any], root: pathlib.Path,
                   facts: Dict[str, Any]) -> None:
    """B10: the history (untagged) and the steady state a long-lived install keeps over it, then each turn up to
    its first model request; at k=1 also breadth, rows and archived history."""
    from tests import _turn_shapes as shapes

    no_carrier_can_start()
    ctx, k = _event_ctx(srv), int(facts.get("k") or 1)

    def breadth() -> Dict[str, Any]:
        from supervisor import queue

        laid = shapes.grow_breadth(root, shapes.BREADTH)
        queue.persist_queue_snapshot(reason="breadth")  # the tick publishes what it holds: a roster reads the document
        return laid

    def activity() -> Dict[str, Any]:
        _completion(root, "turn", k)
        return shapes.foreign_activity(root)

    def rows() -> Dict[str, Any]:
        grown = shapes.grow_rows(root, shapes.MORE_ROWS)
        NEEDLES["T:main:rows"] = [grown["main_newest"]]
        return grown

    shape_steps = {"history": lambda: shapes.turn_history(root, k),
                   "steady": lambda: shapes.steady_state(root, refused=refused),
                   "activity": activity, "breadth": breadth, "rows": rows,
                   "archive": lambda: shapes.archive_history(root, shapes.ARCHIVED)}
    plan = ["history", "steady", "census", ("T:main", 1), ("T:mainw", 1), "activity", ("T:main:foreign", 1),
            ("T:room", int(facts["late_chat_id"]))]
    if k == 1:  # the breadth, rows and archive axes need one size only
        plan += ["breadth", ("T:main:broad", 1), "rows", ("T:main:rows", 1),
                 "archive", ("T:main:archive:first", 1), ("T:main:archive", 1)]
    for index, step in enumerate(plan):
        if step == "census":
            _boot_census(srv, root)
        elif isinstance(step, str):
            started, cpu = time.monotonic(), time.process_time()
            laid = shape_steps[step]()
            result.setdefault("turn_shapes", {})[step] = {**laid, "cpu": round(time.process_time() - cpu, 2),
                                                          "seconds": round(time.monotonic() - started, 2)}
        else:
            tag, chat_id = step
            log = root / "logs" / "supervisor.jsonl"
            start = log.stat().st_size if log.exists() else 0
            _boundary(meter, result, tag, lambda chat_id=chat_id, index=index: _owner_turn(
                srv, ctx, chat_id, f"cost-turn-{index}"), timeout=600.0)
            if f"{tag}@request" in result:  # the build of this turn, as its own receipt counted it
                result[f"{tag}@request"]["view"] = _view_receipt(root)
            else:  # how it ended instead, as the product recorded it
                result["responses"][tag]["ended"] = _turn_ends(log, start)
            _lane_rows_ended()


def _duty_rows(sm: Any) -> List[Any]:
    """The inventory the dispatcher starts from. An older body without the declaration (the negative
    control on the body that broke the owner's install) is read by the latch and runner names its
    dispatcher used; on a body that has it this shim never applies (tests/test_duty_inventory.py)."""
    rows = getattr(sm, "HOST_DUTIES", None)
    if rows is not None:
        return list(rows)
    legacy = (("cancel_intent_sweep", "_CANCEL_INTENT_SWEEP_LOCK", "_run_terminal_duty"),
              ("terminal_delivery_replay", "_DELIVERY_SWEEP_LOCK", "_run_terminal_duty"),
              ("question_delivery_replay", "_QUESTION_SWEEP_LOCK", "_run_terminal_duty"),
              ("pause_notice_replay", "_PAUSE_NOTICE_SWEEP_LOCK", "_run_terminal_duty"),
              ("abandoned_usage_reconciliation", "_USAGE_SWEEP_LOCK", "_run_terminal_duty"),
              ("post_task_synthesis_recovery", "_SYNTHESIS_SWEEP_LOCK", "_run_terminal_duty"),
              ("held_continuation_recheck", "_CONTINUATION_SWEEP_LOCK", "_run_terminal_duty"),
              ("owner_pause_settlement", "_OWNER_PAUSE_SWEEP_LOCK", "_run_terminal_duty"),
              ("cold-custody", "_COLD_CUSTODY_SWEEP_LOCK", "_run_cold_custody"),
              ("custody-maintenance", "_CUSTODY_SWEEP_LOCK", "_run_periodic_custody_sweep"),
              ("reconcile-maintenance", "_RECONCILE_SWEEP_LOCK", "_run_periodic_reconcile_sweep"),
              ("maintenance-child", "", "_tick_maintenance_child"),
              ("schedule-admission", "_SCHEDULE_SWEEP_LOCK", "_run_schedule_duty"))
    return [SimpleNamespace(name=name, latch=getattr(sm, latch) if latch else None, run=getattr(sm, runner))
            for name, latch, runner in legacy]


def _one_pass(sm: Any, duty: Any, root: pathlib.Path, stop: threading.Event) -> None:
    """One pass of a declared duty on this thread, under its latch, with the arguments the tick gives it."""
    run, latch = duty.run, duty.latch
    if run is sm._tick_maintenance_child:
        run()
        return
    if not latch.acquire(blocking=False):
        raise RuntimeError(f"{duty.name}: its latch is held")
    if run is sm._run_terminal_duty:
        run(root, duty.name, latch, stop)
    elif run is sm._run_periodic_reconcile_sweep:
        run([0.0], stop, latch)
    else:  # cold custody, the custody sweep, schedule admission
        run(stop, latch)


def _measured_pass(meter: CostMeter, tag: str, one: Callable[[], None], sm: Any, root: pathlib.Path,
                   stop: threading.Event, result: Dict[str, Any]) -> Dict[str, Any]:
    """One pass under ``tag``, everything it dispatched (``_drain``), then its count. The spawn
    baseline is read BEFORE the pass: a look whose thread ends inside the pass has its spawn
    recorded before the drain begins, and a baseline read then would take it for an old one and
    leave the child's jobs unmeasured."""
    spawned = len(result.get("child_spawns", ()))
    try:
        tagged(tag, one)
    except Exception as exc:
        result["errors"].append(f"{tag}: {type(exc).__name__}: {exc}")
    _drain(meter, tag, sm, root, stop, result, spawned)
    _mark(result, tag)
    return _summary(meter, tag)


def _drain(meter: CostMeter, tag: str, sm: Any, root: pathlib.Path, stop: threading.Event,
           result: Dict[str, Any], spawned: int) -> None:
    """Everything the pass dispatched or woke, before its count is read: its threads (the child's
    look, the daemon latch retry, a reaper), the child look it requested (run here under its tag),
    the child's jobs a look it caused spawned — every spawn recorded since ``spawned``, the count
    before the pass — (run under its tag, as B6's job runner does: a child process the meter cannot
    see would hide the work a pass dispatched; on a thread of its own, as the child process starts:
    it inherits no connection pool, so the connections the jobs open are the same at every size
    instead of depending on what the pass's thread leased before) and the early delivery replay it woke. What the
    child reports stays on the pass's tag (``child_reports``), where B7 reads it: a thread does not lose it."""
    from ouroboros import maintenance

    def child_pass() -> None:
        done: Dict[str, Any] = {}

        def jobs() -> None:
            try:
                done["report"] = maintenance.run(root, should_stop=lambda: False)
            except Exception as exc:  # what the child process reports for a pass that raised (maintenance.main)
                done["report"] = {"status": "failed", "jobs": {}, "raised": f"{type(exc).__name__}: {exc}"}

        worker = threading.Thread(target=jobs, name="maintenance-child-jobs")
        worker.start()
        worker.join(600.0)
        if worker.is_alive():
            result["errors"].append(f"{tag}: the child's jobs did not finish")
        else:
            result.setdefault("child_reports", {}).setdefault(tag, []).append(done.get("report"))

    for _round in range(3):
        alive = meter.join_spawned(tag, timeout=180.0)
        if alive:
            result["errors"].append(f"{tag}: threads still alive: {alive}")
        woke = False
        if len(result.get("child_spawns", ())) > spawned:
            spawned, woke = len(result["child_spawns"]), True
            tagged(tag, child_pass)
        if maintenance.child.owed():
            woke = True
            tagged(tag, sm._tick_maintenance_child)
        if sm._DELIVERY_WAKE.is_set() and not tag.startswith("D:terminal_delivery_replay"):
            woke = True

            def replay():
                if sm._DELIVERY_SWEEP_LOCK.acquire(blocking=False):
                    sm._run_terminal_duty(root, "terminal_delivery_replay", sm._DELIVERY_SWEEP_LOCK, stop)
            tagged(tag, replay)
        if not woke:
            return


def _body_pass(runtime: Any, tick: Callable[..., Any]) -> None:
    """The body's tick, then every future it submitted, then the tick that records their completion
    (the body's own next tick: a fact changed by the work is due again at once), until a tick submits
    nothing. The pass owns everything the work it dispatched caused."""
    for _round in range(4):
        tick(runtime)
        with runtime.lock:
            jobs = [*runtime.reads.values(), *runtime.collections.values()]
        if not jobs:
            return
        concurrent.futures.wait(jobs, timeout=300.0)


def _companions(srv: Any, root: pathlib.Path, facts: Dict[str, Any]) -> Dict[str, Callable[[], str]]:
    """One addressed input change per duty, after whose arrival the duty has work again (the live set's own six in
    tests/_live_set.py). Each returns what it changed."""
    from ouroboros import delegate_custody as custody
    from ouroboros import owner_quiz, store
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.delegate_collection import observe_run
    from ouroboros.late_review_marks import OWED_SET
    from ouroboros.task_results import write_task_result
    from supervisor.terminal_delivery import delivery_id_for, register_pending_delivery

    companion = facts["companion"]

    def body() -> str:
        entry = custody.RunCustody(
            run_id=companion["run_id"], task_id=companion["task_id"], root_task_id=companion["task_id"],
            invocation_id="inv-" + companion["run_id"], route_id="test", model="test", access="readonly", mode="ask",
            ledger_root=str(root), snapshot_id="snap-" + companion["run_id"], execution_root=companion["area"],
            baseline_sha=companion["baseline_sha"], target_root=companion["area"], authority_source="investigation")
        custody.record_started(root, entry)
        observe_run(root, entry, terminal_detail(companion["run_id"]))
        return "a terminal observed for a run with no capture yet"

    def late_body() -> str:
        write_task_result(root, "hist-absent-0-0", "cancelled", chat_id=1, result="the late body")
        return "the result row of a result_absent usage owner landed"

    def owed_projection() -> str:
        write_task_result(root, "cur-owes-projection", "completed", chat_id=1, result="done without a projection")
        return "a settled task without its terminal projection"

    def owed_mark() -> str:
        with store.open_store(root, write=True) as txn:
            txn.put_row(OWED_SET, "late-review-naming-nothing", {"after_end": True})
        return "an owed late-review row that names nothing it can owe"

    def child_job() -> str:
        schedule_child_job(root)
        srv_child = __import__("ouroboros.maintenance", fromlist=["child"]).child
        srv_child.request()
        return "legacy_reconcile scheduled and the child look requested"

    def question() -> str:
        write_task_result(root, "cur-asks-later", "running", chat_id=1, text="asks a question")
        owner_quiz.record_asked(root, "cur-asks-later", quiz_id="quiz-late-1", question="Which one now?",
                                options=["A", "B"], chat_id=1)
        return "an asked question not yet published"

    def delivery() -> str:
        from datetime import datetime, timedelta, timezone

        text = "an answer owed to the chat"
        lost = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()  # a send presumed lost an hour ago
        register_pending_delivery(root, {"delivery_id": delivery_id_for("cur-owes-delivery", text),
                                         "task_id": "cur-owes-delivery", "chat_id": 1, "text": text,
                                         "role": "assistant", "type": "send_message", "last_replay_at": lost})
        return "a terminal delivery whose send was lost"

    def cancel() -> str:
        from supervisor import task_lifecycle

        task_lifecycle._INTENT_WATCHDOG_MIN_AGE_SEC = 0.0  # the in-band event had its tick (the loop is held here)
        write_task_result(root, "cur-cancel-me", "running", chat_id=1, text="to be cancelled")
        request_cancel(root, "cur-cancel-me", reason="cost contract", source="owner", requested_by="owner")
        return "an open cancel intent for a task nobody runs"

    from tests._live_set import companions as live_set_companions

    return {"delegated-body": body, "abandoned_usage_reconciliation": late_body, "custody-maintenance": owed_projection,
            "reconcile-maintenance": owed_mark, "maintenance-child": child_job, "question_delivery_replay": question,
            "terminal_delivery_replay": delivery, "cancel_intent_sweep": cancel, **live_set_companions(root)}


def _completion(root: pathlib.Path, who: str, _k: int) -> None:
    """One ordinary task nobody here owns completes: a terminal row write, the terminal-write counter moves."""
    from ouroboros.task_results import write_task_result

    write_task_result(root, f"cur-unrelated-{who}", "completed", chat_id=1, result="an ordinary finished turn")


def _money_write(root: pathlib.Path, who: str, k: int) -> None:
    """One task at work records one LLM attempt through the real ledger, as every model call on a live
    install does: the usage store's publication marker moves and the task is a dirty usage owner."""
    from ouroboros.task_results import write_task_result

    task_id = f"cur-money-{who}"
    write_task_result(root, task_id, "running", chat_id=1, text="a task at work, making one model call")
    money(root, task_id, limit=spend_limit_usd(k))


def _new_drive(root: pathlib.Path, who: str, _k: int) -> None:
    """One new child drive appears for a task at work (a forked or empty-memory task, an external-workspace
    task), as admission creates it (``headless.prepare_task_drive``, then the row naming the drive): the
    layout directory's mtime moves."""
    from ouroboros.headless import prepare_task_drive
    from ouroboros.task_results import write_task_result

    task_id = f"cur-drive-{who}"
    child = prepare_task_drive(root, task_id, "empty")
    write_task_result(root, task_id, "running", chat_id=1, text="a task at work on its own drive",
                      child_drive_root=str(child))


# The ordinary activities of an install that is never idle, one between two passes of every duty and job
# (tests/test_cost_contract.py: the pass after each costs the same at every size). Each moves one
# install-wide fact a rest could wrongly key on: the terminal-write counter, the usage ledger's marker,
# a layout directory's mtime. Each takes the root, the name of the duty or job it happens beside, and k.
ACTIVITIES: Dict[str, Callable[[pathlib.Path, str, int], None]] = {
    "unrelated": _completion, "money": _money_write, "drive": _new_drive}


def _measure_duties(meter: CostMeter, srv: Any, result: Dict[str, Any], root: pathlib.Path, facts: Dict[str, Any],
                    stubs: Dict[str, Any]) -> None:
    from ouroboros import server_maintenance as sm
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from ouroboros.delegate_collection import body_runtime
    from ouroboros.headless import DRIVE_SETTLEMENTS_PER_PASS
    from ouroboros.maintenance import child

    stop = threading.Event()
    k = int(facts.get("k") or 1)
    companions = _companions(srv, root, facts)
    duties = result.setdefault("duties", {})
    runtime, park, clock = body_runtime(root), stubs["park"], stubs["clock"]
    if runtime is None:
        result["errors"].append("the delegated body is not running")
        return

    def one_pass(tag: str, one: Callable[[], None]) -> Dict[str, Any]:
        return _measured_pass(meter, tag, one, sm, root, stop, result)

    # A duty that works through a budgeted lap (drive settlement: DRIVE_SETTLEMENTS_PER_PASS attempts per
    # pass and layout) settles the root's drives over as many passes as its layouts need, then one more
    # pass records that the lap found nothing to do.
    settle_passes = 2 + -(-int(facts.get("child_drives", 0)) // DRIVE_SETTLEMENTS_PER_PASS)

    def passes(name: str, one: Callable[[], None], before_rest: Callable[[], None]) -> None:
        """``first`` over the root as the boot left it; then, while the previous pass wrote (a duty
        whose first pass changed its own inputs, e.g. recorded a member's rest after writing its
        row, or a budgeted lap still settling), up to ``settle_passes`` ``settle`` passes; then
        ``rest``, the pass over unchanged inputs the law holds to; then one pass after each ordinary
        activity of ``ACTIVITIES`` nobody here owns (the install is never idle: a pass may do the
        addressed work the activity owes it, never history-sized work); then ``work`` after the
        companion's input change."""
        entry: Dict[str, Any] = {"first": one_pass(f"D:{name}:first", one)}
        settled = entry["first"]
        for attempt in range(1, settle_passes + 1):
            if not settled["counts"].get("writes"):
                break
            before_rest()
            settled = entry[f"settle{attempt}"] = one_pass(f"D:{name}:settle{attempt}", one)
        before_rest()
        entry["rest"] = one_pass(f"D:{name}:rest", one)
        for phase, happen in ACTIVITIES.items():
            happen(root, name, k)
            before_rest()
            entry[phase] = one_pass(f"D:{name}:{phase}", one)
        companion = companions.get(name)
        if companion is None:
            entry["companion"] = None
        else:
            entry["companion"] = companion()
            before_rest()
            entry["work"] = one_pass(f"D:{name}:work", one)
        duties[name] = entry

    def advance_body_clock() -> None:
        clock.now += BODY_MAINTENANCE_INTERVAL_SEC + 1.0

    # The child's own pass over the root as the boot left it (what the install's child would have run
    # by now), until no job is due; then the process's first look at the job facts, owed to the boot.
    from ouroboros import maintenance

    for _attempt in range(3):
        if not maintenance.due(root):
            break
        report = tagged("D:maintenance-child:jobs", lambda: maintenance.run(root, should_stop=lambda: False))
        if refused(report):  # every duty below would be measured on an install whose maintenance never ran
            result["errors"].append(f"D:maintenance-child:jobs refused: {refused(report)}")
    result["child_jobs"] = {**_summary(meter, "D:maintenance-child:jobs"), "still_due": maintenance.due(root)}
    tagged("D:maintenance-child:boot", sm._tick_maintenance_child)
    meter.join_spawned("D:maintenance-child:boot", timeout=180.0)
    passes("delegated-body", lambda: _body_pass(runtime, park.tick), advance_body_clock)
    for duty in _duty_rows(sm):
        passes(duty.name, lambda duty=duty: _one_pass(sm, duty, root, stop),
               child.request if duty.name == "maintenance-child" else (lambda: None))


def _measure_contract(meter: CostMeter, srv: Any, client: Any, result: Dict[str, Any], root: pathlib.Path,
                      facts: Dict[str, Any], stubs: Dict[str, Any]) -> None:
    _chat_frame(meter, client, result, "B4", "cost-contract-1")
    _request(meter, client, result, "B3", "GET", "/api/state")
    _request(meter, client, result, "R_active", "GET", f"/api/chat/history?chat_id={facts['active_chat_id']}")
    _request(meter, client, result, "R_late", "GET", f"/api/chat/history?chat_id={facts['late_chat_id']}")
    _chat_frame(meter, client, result, "B4w", "cost-contract-2")
    _request(meter, client, result, "B3w", "GET", "/api/state")
    _request(meter, client, result, "R_activew", "GET", f"/api/chat/history?chat_id={facts['active_chat_id']}")
    _request(meter, client, result, "B5", "POST", f"/api/tasks/{CURRENT['stop_root']}/cancel", json={"cascade": True})
    _boundary(meter, result, "B9", lambda: _addressed_stream(root, result))
    if facts.get("held_runs"):  # a root with the duties' shapes: B8 first, then B7
        heartbeats(root)
        _measure_boundaries(meter, srv, result, root, facts)
        _measure_duties(meter, srv, result, root, facts, stubs)
    _measure_turns(meter, srv, result, root, facts)  # B10 last: no earlier law's inputs change


def run(spec: Dict[str, Any]) -> Dict[str, Any]:
    root = pathlib.Path(spec["root"])
    result: Dict[str, Any] = {"facts": spec.get("facts", {}), "errors": [], "_started": time.monotonic()}
    meter = CostMeter(root, execution_root=spec.get("facts", {}).get("execution_root"),
                      thread_rules=[("current-store-import", "B2"), ("supervisor-main", "B2"),
                                    ("boot-coordinator", "B1c")],
                      lineage={"B1": "B1:spawned", "B2": "B2:spawned", "B1c": "B1c:spawned"})
    meter.__enter__()
    stubs: Dict[str, Any] = {}
    try:
        result["clocks"] = hold_clocks(spec.get("clocks"))  # before the server's first read
        from starlette.testclient import TestClient

        import server as srv

        force_store_journal()
        _mark(result, "imported")
        _stub_external_effects(srv, result, meter)
        stubs = _stub_duty_effects(result)
        # The install root is resolved once per process, in whichever boundary's thread asks first:
        # a race the history size tips, not history work. Resolve it before any boundary starts.
        from ouroboros import install_identity

        install_identity.canonical_root()
        events = _install_checkpoints(meter, srv, result, root)
        from ouroboros import install_identity

        tagged("B0", install_identity.canonical_root)
        result["B0"] = _summary(meter, "B0")
        client = TestClient(_Tagger(srv.app, meter, result))
        with client:
            response = client.get("/api/health", headers={"x-cost-boundary": "B1h"})
            _mark(result, "health")
            result["B1h"] = _summary(meter, "B1h")
            # On the path the first answer waits for: the lifespan up to its yield and the request itself.
            result["popen_before_health"] = (result["B1"]["counts"].get("popen", 0)
                                             + result["B1h"]["counts"].get("popen", 0))
            result.setdefault("responses", {})["B1h"] = {"status": response.status_code}
            if not events["ready"].wait(READY_TIMEOUT_SEC):
                result["errors"].append("the supervisor never reported ready")
            if meter.spawned.get("B1c"):  # the boot coordinator (work after the yield) ends once readiness is known
                if meter.join_spawned("B1c", timeout=READY_TIMEOUT_SEC):
                    result["errors"].append("the boot coordinator did not finish")
                _mark(result, "coordinator")
                result["B1c"] = _summary(meter, "B1c")
            # The one-time archive import runs in the maintenance child on a real install; the
            # boundaries below measure the steady state after it (a room page, an owner message).
            token = BOUNDARY.set("import")
            try:
                from ouroboros.dialogue_import import run_dialogue_import

                result["import"] = {"state": run_dialogue_import(root).get("state")}
            finally:
                BOUNDARY.reset(token)
            try:
                facts = spec.get("facts", {})
                if spec.get("plan") == "live_set":  # the live-set axis grows the set on this booted root instead
                    from tests import _live_set

                    _live_set.measure(meter, srv, client, result, root, spec, stubs)
                elif spec.get("plan") == "stand":  # the production-shaped stand: that set, the rooms and the turns
                    from tests import _stand

                    _stand.measure(meter, srv, client, result, root, spec, stubs)
                else:
                    _measure_contract(meter, srv, client, result, root, facts, stubs)
            finally:
                result["process_at_end"] = _all_counts(meter)
                if stubs:
                    stubs["park"].release.set()
                srv._supervisor_stop.set()  # the held supervisor leaves before its first tick
                events["release"].set()
    except Exception:
        result["errors"].append(traceback.format_exc())
    finally:
        meter.__exit__(None, None, None)
    return result


def main(argv: List[str]) -> int:
    spec = json.loads(pathlib.Path(argv[0]).read_text(encoding="utf-8"))
    if pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR", "")).resolve() != pathlib.Path(spec["root"]).resolve():
        raise SystemExit("OUROBOROS_DATA_DIR must name the measured root")
    started = time.monotonic()
    result = run(spec)
    result.pop("_started", None)
    result["driver_seconds"] = round(time.monotonic() - started, 2)
    pathlib.Path(argv[1]).write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
