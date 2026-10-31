"""The boot coordinator's steps of the server lifespan.

What the lifespan ran before it served now runs here, in the same order, on the boot
coordinator's thread while the socket already answers (``server_liveness.start_boot_coordinator``;
the step list and the pinned seams stay in ``server.py::lifespan``). Each function is one named
step: it does its work or raises, and the coordinator records the outcome in the boot document.
Nothing here decides readiness; the two waits at the end only observe the store's publication
and the supervisor's outcome for the document's phase.
"""

from __future__ import annotations

import asyncio
import pathlib
import threading
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


def start_background_services(settings: dict, *, local_model: bool, boot_managed_update: Callable[[], Any],
                              warm_daemon: bool) -> None:
    """P2: finalize a pending managed merge update (post-boot smoke / boot-loop rollback) and run a
    one-shot boot-time update check (check-on-restart) so the main-screen Update badge reflects
    availability; both fail-soft. Local autostart goes FIRST: a local-only install's boot check
    may write its update letter through the local model, and the git fetch ahead of that call is
    the head start the model server gets (no readiness wait — a letter that still finds the model
    loading fails typed and is rewritten by the next check). The owned daemon is warmed, and an
    OpenRouter install's model catalogue observed unless a process recorded a current one, for
    provisioned homes only (``warm_daemon``): background work off the startup path, so a request
    reads the catalogue and never waits for it. A sign-in entry naming a copy of the app that is
    gone is re-pointed at this one (``desktop_autostart.follow_moved_app``, fail-soft)."""
    if local_model and settings.get("LOCAL_MODEL_SOURCE"):
        from ouroboros.local_model_autostart import auto_start_local_model

        threading.Thread(target=auto_start_local_model, args=(settings,), daemon=True, name="local-model-autostart").start()
    threading.Thread(target=boot_managed_update, daemon=True, name="boot-managed-update").start()
    if warm_daemon:
        from ouroboros.claudexor_daemon import warm_owned_daemon

        warm_owned_daemon()
    from ouroboros.provider_models import configured_kinds

    if warm_daemon and "openrouter" in configured_kinds(settings):  # any OpenRouter key, built-in or added, that serves
        from ouroboros.llm import LLMClient

        threading.Thread(target=LLMClient.prefetch_openrouter_catalog, daemon=True, name="openrouter-catalog").start()
    from ouroboros.desktop_autostart import follow_moved_app

    follow_moved_app()


async def serve_host_service(root: pathlib.Path, boot: Any) -> None:
    """Start the companions' Host Service on the main loop; its task, server and bound socket
    land on ``boot`` for the lifespan's teardown."""
    from ouroboros.event_bus import init_global_event_bus
    from ouroboros.extension_companion import init_global_supervisor
    from ouroboros.gateway.host_service import DEFAULT_HOST_SERVICE_HOST, create_host_service_app, host_service_port

    init_global_event_bus().set_loop(asyncio.get_running_loop())
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
    is this step's failure."""
    while not (ready.wait(0.5) or error() or stop.is_set() or document.phase() == "failed"):
        pass
    if error():
        raise RuntimeError(error())
