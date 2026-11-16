"""The boot coordinator's steps of the server lifespan.

What the lifespan ran before it served now runs here, in the same order, on the boot
coordinator's thread while the socket already answers (``server_liveness.start_boot_coordinator``;
the step list and the pinned seams stay in ``server.py::lifespan``). A step does its work or
raises, and the coordinator records the outcome in the boot document; ``on_loop`` and
``stop_extension_loop`` are helpers, not steps. One step, ``prepare_event_queue``, gives the event
bus a head start on its own thread while the extensions load; the supervisor waits for it to end
(``wait_event_queue``) before it initializes its workers. Nothing here decides readiness; the two
waits at the end only observe the store's publication and the supervisor's outcome for the
document's phase.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import threading
from concurrent.futures import Future, wait
from typing import Any, Callable, Optional

import uvicorn

from ouroboros import store_switch
from ouroboros.server_entrypoint import bound_service_socket
from ouroboros.server_liveness import BOOT, BootDocument
from ouroboros.server_process import _embedded_uvicorn_server, log


def seed_skills(root: pathlib.Path, settings: dict, *, provider_ready: bool) -> None:
    """The skills part of startup: the built-in source's work on this data root, then the starter
    installer for a new install that skips onboarding (``skill_starter.skills_startup``)."""
    from ouroboros.skill_starter import skills_startup

    skills_startup(root, settings, provider_ready=provider_ready)


def reconcile_projects(root: pathlib.Path) -> None:
    """Register any pre-existing data/projects/<id>/ store whose row is missing, so an inherited
    project's raw chat is partitioned (reserved_project_chat_ids, the multi-project isolation SSOT)
    before the supervisor routes a turn. Idempotent and never prunes."""
    from ouroboros.projects_registry import reconcile_projects as reconcile

    reconcile(root)


def remember_log_listings(root: pathlib.Path) -> threading.Thread:
    """The archive listing a process's first owner turn would make for its task's recent tools and
    events, made once off the turn, so that turn starts from it (``Memory.remember_log_listings``).
    It grows with the archive, so a daemon thread makes it, off the steps readiness waits for, as the
    catalogue is observed; a turn that comes first lists for itself."""
    from ouroboros.memory import Memory

    thread = threading.Thread(target=Memory(root).remember_log_listings, daemon=True, name="log-listings")
    thread.start()
    return thread


def prepare_pinned_node(root: pathlib.Path) -> Optional[threading.Thread]:
    """Boot step, on disk only: before skills, extensions, MCP servers and companions, this process names Playwright's
    driver Node (``node_runtime.select_local_playwright_node``), so in-process extensions and every child it builds an
    environment for never run the SDK's own Node. Preparing the pin's Node, when it still has to be copied or
    downloaded (an operator's explicit driver does not change that: skills and the gate's Node lane use the pin's
    too), is ``request_pinned_node``'s, off the steps readiness waits for, as the owned daemon's warm-up is."""
    from ouroboros.node_runtime import select_local_playwright_node

    return request_pinned_node(root) if select_local_playwright_node() else None


_PREPARING = threading.Lock()  # one preparation thread per process; the install lock orders processes


def request_pinned_node(root: "pathlib.Path | None" = None) -> Optional[threading.Thread]:
    """Prepare the pin's Node off the caller's path (``finish_pinned_node`` on a daemon thread), single-flight: a
    request while one runs in this process is that one. Asked by the boot step and by an acting consumer that found the
    pin unprepared after a failed attempt (a skill or companion start, the attach role, the gate's Node lane), never by
    a lookup, an environment built for a child, a status read or read-only work. No timer: the next acting consumer
    asks again. ``OUROBOROS_PYTEST_ACTIVE=1`` starts none, as it blocks dependency installs. ``root``: the data root
    whose extensions are asked again, by default this process's."""
    if os.environ.get("OUROBOROS_PYTEST_ACTIVE") == "1" or not _PREPARING.acquire(blocking=False):
        return None
    if root is None:
        from ouroboros.config import DATA_DIR as root

    def _prepare() -> None:
        try:
            finish_pinned_node(root)
        finally:
            _PREPARING.release()

    thread = threading.Thread(target=_prepare, daemon=True, name="pinned-node")
    thread.start()
    return thread


def finish_pinned_node(root: pathlib.Path) -> str:
    """Prepare the pin's Node, Node only (never the engine or an account), under the install lock a browser call joins;
    then this process's automatic selection names it, and ``requeue_waiting_extensions`` asks again. A failure is
    logged: the next acting consumer (``request_pinned_node``), the browser tool and the next boot try again."""
    from ouroboros.claudexor_runtime import get_runtime_manager
    from ouroboros.node_runtime import select_playwright_node

    try:
        get_runtime_manager().reviewed_node(allow_install=True)
        node = select_playwright_node()
    except Exception as exc:
        log.warning("The pinned Node for Playwright's driver was not prepared: %s", exc)
        return ""
    requeue_waiting_extensions(root)
    return node


def requeue_waiting_extensions(root: pathlib.Path) -> None:
    """After the pin's Node was prepared: each enabled extension not live in this process is asked again through the
    server's reconcile queue (``request_extension_reconcile``, file-backed, so a worker's request reaches it too): one
    whose load failed on the planned path, as one starting Playwright while loading does, or one still loading, whose
    reconcile waits for that load (its lifecycle lock). A disabled one is never asked; a live one is not reloaded."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.extension_reconcile_queue import request_extension_reconcile
    from ouroboros.extension_registry_state import _extensions, _lock
    from ouroboros.skill_loader import discover_skills

    skills = discover_skills(root, repo_path=get_skills_repo_path() or None)
    with _lock:
        live = set(_extensions)
    for name in sorted(s.name for s in skills if s.manifest.is_extension() and s.enabled and s.name not in live):
        request_extension_reconcile(root, name, reason="pinned Node prepared", source="server")


def start_background_services(settings: dict, *, local_model: bool, boot_managed_update: Callable[[], Any],
                              warm_daemon: bool) -> None:
    """P2: finalize a pending managed merge update (post-boot smoke / boot-loop rollback) and run a
    one-shot boot-time update check (check-on-restart) so the main-screen Update badge reflects
    availability; both fail-soft. Local autostart goes FIRST: a local-only install's boot check
    may write its update letter through the local model, and the git fetch ahead of that call is
    the head start the model server gets (no readiness wait — a letter that still finds the model
    loading fails typed and is rewritten by the next check). The owned daemon is warmed, and an
    OpenRouter install's model catalogue observed unless a process recorded a current one with prices, for
    provisioned homes only (``warm_daemon``): background work off the startup path, so a request
    reads the catalogue and never waits for it. A managed execution domain's dedicated containers
    are measured for Panic, also off the startup path (``execution_domain``). A sign-in entry naming
    a copy of the app that is gone is re-pointed at this one (``desktop_autostart.follow_moved_app``, fail-soft)."""
    if local_model and settings.get("LOCAL_MODEL_SOURCE"):
        from ouroboros.local_model_autostart import auto_start_local_model

        threading.Thread(target=auto_start_local_model, args=(settings,), daemon=True, name="local-model-autostart").start()
    threading.Thread(target=boot_managed_update, daemon=True, name="boot-managed-update").start()
    if warm_daemon:
        from ouroboros.claudexor_daemon import warm_owned_daemon

        warm_owned_daemon()
    from ouroboros.execution_domain import measure_generation_identity

    measure_generation_identity()  # a managed domain's dedicated containers, for this generation's Panic
    from ouroboros.provider_models import configured_kinds

    if warm_daemon and "openrouter" in configured_kinds(settings):  # any OpenRouter key, built-in or added, that serves
        from ouroboros.llm import LLMClient

        threading.Thread(target=LLMClient.prefetch_openrouter_catalog, daemon=True, name="openrouter-catalog").start()
    from ouroboros.desktop_autostart import follow_moved_app

    follow_moved_app()


async def serve_host_service(root: pathlib.Path, boot: Any) -> None:
    """Start the shared extension loop, then the companions' Host Service on the ASGI loop.

    Supervised tasks and async subscribers use the extension loop. The Host Service
    task, server and bound socket stay on this loop and land on ``boot`` for teardown.
    """
    from ouroboros.event_bus import init_global_event_bus
    from ouroboros.extension_companion import init_global_supervisor
    from ouroboros.gateway.host_service import DEFAULT_HOST_SERVICE_HOST, create_host_service_app, host_service_port

    init_global_event_bus().start_loop()
    init_global_supervisor(root)
    host_service_app = create_host_service_app(root)
    host_port = host_service_port()
    # Bind before starting the asyncio task: uvicorn's bind-error SystemExit otherwise escapes
    # run_forever and kills the main server. Keep that socket, removing the former
    # probe-close/rebind race as well.
    host_socket = boot.host_service_listener.enter_context(bound_service_socket(
        root, "host_service", DEFAULT_HOST_SERVICE_HOST, host_port))
    config = uvicorn.Config(host_service_app, host=DEFAULT_HOST_SERVICE_HOST, port=host_port,
                            log_level="warning", log_config=None)  # uvicorn loggers propagate to the root handlers
    boot.host_service_server = _embedded_uvicorn_server(config)
    boot.host_service_task = asyncio.create_task(boot.host_service_server.serve(sockets=[host_socket]), name="host-service-api")
    boot.host_service_task.add_done_callback(lambda _task: boot.host_service_listener.close())
    log.info("Host Service API listening on %s:%d", DEFAULT_HOST_SERVICE_HOST, host_port)


def stop_extension_loop(timeout: float = 5.0) -> None:
    """Cancel the extension loop so a skill ``finally`` still runs on shutdown."""
    try:
        from ouroboros.event_bus import get_global_event_bus

        get_global_event_bus().stop_loop(timeout)
    except Exception:
        log.debug("extension loop did not stop", exc_info=True)


def on_loop(loop: asyncio.AbstractEventLoop, coroutine) -> Any:
    """Run a loop-bound step from the coordinator's thread and wait for its outcome."""
    return asyncio.run_coroutine_threadsafe(coroutine, loop).result()


def reconcile_review_jobs(root: pathlib.Path) -> None:
    from ouroboros.skill_review_runner import reconcile_stale_review_jobs

    reconcile_stale_review_jobs(root)


def mcp_startup() -> None:
    from ouroboros.config import load_settings
    from ouroboros.mcp_client import get_manager, configuration_changed

    get_manager().become_lister()
    # Read the saved document when the step runs, never the early boot snapshot,
    # and without waiting for a Save: one that changes MCP applies its own event.
    configuration_changed(load_settings(), reason="startup")


def wait_store_published(root: pathlib.Path, stop: threading.Event, document: BootDocument = BOOT) -> None:
    """The import's outcome: a boot whose store did not publish cannot become ready."""
    outcome = store_switch.wait_ready(root, timeout=0.5)
    while outcome.get("status") == "importing" and not stop.is_set():
        outcome = store_switch.wait_ready(root, timeout=0.5)
    if outcome.get("status") not in ("published", "not_started", "importing"):
        document.fail(f"current-facts store not published: {outcome.get('error') or outcome.get('status')}")
        raise RuntimeError(f"store not published: {outcome.get('status')}")


def wait_supervisor_outcome(ready: threading.Event, error: Callable[[], Optional[str]], stop: threading.Event,
                            document: BootDocument = BOOT) -> None:
    """The generation is ready once the supervisor (or the providerless recovery) is; its failure
    is this step's failure. The lifespan names the step ``supervisor_outcome``: ``supervisor`` is the
    generation's own step with its startup sub-steps, which this observation never overwrites."""
    while not (ready.wait(0.5) or error() or stop.is_set() or document.phase() == "failed"):
        pass
    if error():
        raise RuntimeError(error())


_head_start: Optional[Future] = None  # the latest provider-ready lifespan's (prepare_event_queue); a done one is inert


def prepare_event_queue(root: pathlib.Path, stop_requested: Callable[[], bool],
                        document: BootDocument = BOOT) -> Future:
    """Start the process-lifetime event bus on its own thread while the extensions load: a head
    start, never workers or recovery, which still wait for the registered subscribers.

    The supervisor waits for it to end (``wait_event_queue``), then makes its ordinary
    ``get_event_q`` call: that returns this queue, or after a failure (this step's, in the boot
    document) ends the attempt's never-published manager and creates anew, as without the head
    start. Stop/Panic is rechecked under the queue's lock, and fences the manager's start."""
    global _head_start
    future = _head_start = Future()

    def prepare():
        try:
            outcome = store_switch.wait_ready(root, timeout=0.1)
            while outcome.get("status") == "importing" and not stop_requested():
                outcome = store_switch.wait_ready(root, timeout=0.1)
            if stop_requested():
                future.cancel()
                return
            if outcome.get("status") not in ("published", "not_started"):
                raise RuntimeError(f"store not published: {outcome.get('status')}")
            from supervisor.workers import get_event_q

            with document.step("startup:event_queue"):
                queue = get_event_q(drive_root=root, stop_requested=stop_requested)
            future.set_result(queue)
        except BaseException as exc:
            log.warning("The event bus head start failed; unless the boot stops, the supervisor creates the bus itself",
                        exc_info=True)
            future.set_exception(exc)

    try:
        threading.Thread(target=prepare, daemon=True, name="boot-event-queue").start()
    except BaseException as exc:
        future.set_exception(exc)
    return future


def wait_event_queue(stop_requested: Callable[[], bool]) -> bool:
    """Wait until the head start ended, never past a Stop; False once stopping. Its outcome is not
    re-raised: the caller's own ``get_event_q`` call owns creation, as without a head start."""
    while _head_start is not None and not _head_start.done() and not stop_requested():
        wait([_head_start], timeout=0.1)
    return not stop_requested()
