"""One supervised lifecycle for sidecar, Quick Tunnel, and Telegram button."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import os
import random
import signal
import socket
import threading
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import httpx
import uvicorn

from cloudflare_tunnel import (
    CLOUDFLARED_VERSION,
    CloudflaredError,
    QuickTunnel,
    ensure_cloudflared,
)
from sidecar import SidecarSeams, TelegramProxySidecar
from platform_support import (
    IS_WINDOWS,
    PlatformSupportError,
    acquire_file_lock,
    path_is_link_or_reparse,
    process_alive,
    release_file_lock,
    require_isolated_process_group as _require_isolated_process_group,
    start_parent_lifeline,
    write_lock_owner,
)
from runtime_status import RuntimeStatus, RuntimeStatusError
from public_gateway import (
    PublicProbeOutcome,
    PublicGatewayError,
    probe_public_gateway_once,
)
from telegram_settings import (
    TelegramSettingsError,
    TelegramSettingsObserver,
    miniapp_enabled,
    telegram_api_base,
    telegram_proxy,
)
from telegram_menu import (
    TelegramMenuConflictError,
    TelegramMenuError,
    TelegramMenuManager,
    TelegramMenuRejectedError,
    TelegramMenuTransportError,
    snapshot_owner_chat_id,
)


_CONFIG_NAME = "runtime_config.json"
_STATUS_NAME = "status.json"
_CONFIG_SCHEMA = 2
_INITIAL_PARENT_PID = os.getppid()
# The host's ouroboros.process_custody.PARENT_PID_ENV: the spawner's own PID.
_PARENT_PID_ENV = "OUROBOROS_PARENT_PID"
_RETRY_BASE_SEC = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0)
_STABLE_READY_RESET_SEC = 300.0
_CORE_HEALTH_INTERVAL_SEC = 30.0
_PUBLIC_HEALTH_INTERVAL_SEC = 45.0
_MENU_HEALTH_INTERVAL_SEC = 300.0
_OWNER_HEALTH_INTERVAL_SEC = 30.0
_MONITOR_TICK_SEC = 2.0
_SIDECAR_STOP_GRACE_SEC = 0.75
_SHUTDOWN_TELEGRAM_REQUEST_TIMEOUT_SEC = 0.7
_STUCK_CLEANUP_EXIT_CODE = 78
_T = TypeVar("_T")

class CompanionError(RuntimeError):
    pass


class ShutdownRequested(CompanionError):
    pass


class SidecarStoppedError(CompanionError):
    pass


def acquire_singleton(state_dir: Path, *, timeout_sec: float = 8.0) -> int:
    """Hold one non-blocking process lease for the complete companion lifetime."""
    path = state_dir / ".companion.lock"
    try:
        fd = acquire_file_lock(path, timeout_sec=timeout_sec)
        write_lock_owner(fd, os.getpid())
        return fd
    except PlatformSupportError as exc:
        raise CompanionError("Another Mini App companion generation is still shutting down.") from exc


def release_singleton(fd: int | None) -> None:
    release_file_lock(fd)


def _pid_alive(pid: int) -> bool:
    return process_alive(pid)


def _server_parent_pid() -> int:
    """The spawning server's PID, or 0 once this companion is already orphaned.

    A bare container server is PID 1, but PID 1 also adopts orphans. A host
    that names itself settles it on POSIX: the parent captured at import and
    the current one must both be that PID, so neither init nor a subreaper
    passes for the server. A legacy host names nobody and PID 1 stays
    refused; Windows never reparents a child.
    """
    parent = _INITIAL_PARENT_PID
    try:
        named = int(os.environ.get(_PARENT_PID_ENV) or 0)
    except ValueError:
        named = 0
    if named and not IS_WINDOWS:
        return parent if named == parent == os.getppid() else 0
    return parent if parent > 1 else 0


def require_running(stop_event: asyncio.Event) -> None:
    if stop_event.is_set():
        raise ShutdownRequested("Companion shutdown was requested.")


def require_isolated_process_group() -> None:
    try:
        _require_isolated_process_group()
    except PlatformSupportError as exc:
        raise CompanionError(str(exc)) from exc


async def _run_startup_operation(
    operation: Awaitable[_T],
    *,
    stop_event: asyncio.Event | None,
    timeout_sec: float,
) -> _T:
    """Bound one startup operation and make graceful shutdown win every race."""
    operation_task = asyncio.create_task(operation)
    stop_task = asyncio.create_task(stop_event.wait()) if stop_event is not None else None
    waiters: set[asyncio.Task[Any]] = {operation_task}
    if stop_task is not None:
        waiters.add(stop_task)
    try:
        done, _pending = await asyncio.wait(
            waiters,
            timeout=max(0.01, float(timeout_sec)),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task is not None and stop_task in done:
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            require_running(stop_event)
        if operation_task in done:
            result = await operation_task
            if stop_event is not None:
                require_running(stop_event)
            return result
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise asyncio.TimeoutError
    finally:
        if stop_task is not None:
            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)


def _safe_state_dir(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    try:
        raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise CompanionError("Could not create private skill state.") from exc
    if path_is_link_or_reparse(raw) or not raw.is_dir():
        raise CompanionError("Skill state must be a real directory.")
    try:
        return raw.resolve(strict=True)
    except OSError as exc:
        raise CompanionError("Could not resolve private skill state.") from exc


def _state_file(state_dir: Path, name: str) -> Path:
    path = state_dir / name
    try:
        path.parent.resolve(strict=True).relative_to(state_dir)
    except (OSError, ValueError) as exc:
        raise CompanionError("Companion state path escaped private skill state.") from exc
    if path_is_link_or_reparse(path):
        raise CompanionError("Companion state path is an unsafe link.")
    return path


def load_runtime_config(state_dir: Path) -> dict[str, Any]:
    path = _state_file(state_dir, _CONFIG_NAME)
    if path.is_symlink() or not path.is_file():
        raise CompanionError("Runtime configuration is missing or unsafe.")
    try:
        if path.stat().st_size > 16_384:
            raise CompanionError("Runtime configuration is oversized.")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except CompanionError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise CompanionError("Runtime configuration is unreadable.") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "core_port",
        "owner_chat_id",
        "button_text",
        "tunnel",
    }:
        raise CompanionError("Runtime configuration has an unexpected shape.")
    if raw.get("schema") != _CONFIG_SCHEMA or raw.get("tunnel") != "cloudflare_quick":
        raise CompanionError("Runtime configuration requests an unsupported transport.")
    try:
        core_port = int(raw["core_port"])
        owner_chat_id = int(raw["owner_chat_id"])
    except (TypeError, ValueError) as exc:
        raise CompanionError("Runtime configuration contains invalid IDs.") from exc
    button_text = str(raw.get("button_text") or "").strip()
    if not 1 <= core_port <= 65_535 or owner_chat_id < 0:
        raise CompanionError("Runtime configuration contains out-of-range IDs.")
    if not button_text or len(button_text) > 64 or any(c in button_text for c in "\r\n"):
        raise CompanionError("Runtime configuration contains invalid button text.")
    return {
        "core_port": core_port,
        "owner_chat_id": owner_chat_id,
        "button_text": button_text,
    }


async def probe_core(core_port: int, *, timeout_sec: float = 2.0) -> bool:
    async with httpx.AsyncClient(
        timeout=timeout_sec,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        try:
            response = await client.get(
                f"http://127.0.0.1:{core_port}/api/health",
                headers={"Accept": "application/json"},
            )
        except (httpx.HTTPError, OSError):
            return False
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


async def start_sidecar(sidecar: TelegramProxySidecar) -> tuple[uvicorn.Server, asyncio.Task[None], int]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server: uvicorn.Server | None = None
    task: asyncio.Task[None] | None = None
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        port = int(listener.getsockname()[1])
        config = uvicorn.Config(
            sidecar.app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            proxy_headers=False,
            server_header=False,
            date_header=False,
            lifespan="off",
            ws_max_size=8 * 1024 * 1024,
            ws_max_queue=16,
            ws_per_message_deflate=False,
            limit_concurrency=128,
            backlog=128,
            timeout_keep_alive=10,
            timeout_graceful_shutdown=3,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        task = asyncio.create_task(server.serve(sockets=[listener]))
        for _ in range(100):
            if server.started:
                return server, task, port
            if task.done():
                try:
                    await task
                except Exception as exc:
                    raise CompanionError("Authentication sidecar failed during startup.") from exc
                raise CompanionError("Authentication sidecar stopped during startup.")
            await asyncio.sleep(0.05)
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, timeout=3)
        raise CompanionError("Authentication sidecar did not start in time.")
    except BaseException:
        if server is not None:
            server.should_exit = True
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        with contextlib.suppress(OSError):
            listener.close()
        raise


async def stop_sidecar(
    server: uvicorn.Server | None,
    task: asyncio.Task[None] | None,
    sidecar: TelegramProxySidecar | None,
) -> None:
    if sidecar is not None:
        sidecar.clear_public_url()
    if server is not None:
        server.should_exit = True

    def consume_terminal_result(done_task: asyncio.Task[None]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            done_task.exception()

    async def finish_cleanup() -> None:
        cleanup_stuck = False
        if task is not None:
            done, _pending = await asyncio.wait(
                {task},
                timeout=_SIDECAR_STOP_GRACE_SEC,
            )
            if task not in done:
                if server is not None:
                    with contextlib.suppress(Exception):
                        server.force_exit = True
                task.cancel()
                done, _pending = await asyncio.wait(
                    {task},
                    timeout=_SIDECAR_STOP_GRACE_SEC,
                )
            if task in done:
                consume_terminal_result(task)
            else:
                task.add_done_callback(consume_terminal_result)
                cleanup_stuck = True
        if sidecar is not None:
            close_task = asyncio.create_task(sidecar.aclose())
            done, _pending = await asyncio.wait(
                {close_task},
                timeout=_SIDECAR_STOP_GRACE_SEC,
            )
            if close_task not in done:
                close_task.cancel()
                done, _pending = await asyncio.wait(
                    {close_task},
                    timeout=_SIDECAR_STOP_GRACE_SEC,
                )
            if close_task in done:
                consume_terminal_result(close_task)
            else:
                close_task.add_done_callback(consume_terminal_result)
                cleanup_stuck = True
        if cleanup_stuck:
            _hard_exit_stuck_cleanup()

    # Keep the bounded cleanup transaction alive across cancellation at any
    # await point.  The caller receives CancelledError only after aclose.
    caller_cancelled = False
    cleanup_task = asyncio.create_task(finish_cleanup())
    cleanup_error: BaseException | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            if cleanup_task.done():
                cleanup_error = exc
                break
            caller_cancelled = True
        except Exception as exc:
            cleanup_error = exc
            break
    if cleanup_error is None and cleanup_task.done():
        try:
            cleanup_error = cleanup_task.exception()
        except asyncio.CancelledError as exc:
            cleanup_error = exc
    if caller_cancelled:
        raise asyncio.CancelledError
    if cleanup_error is not None:
        raise cleanup_error


def _hard_exit_stuck_cleanup() -> None:
    """End the companion if an in-process cleanup coroutine cannot be reaped."""

    os._exit(_STUCK_CLEANUP_EXIT_CODE)


def _safe_error(exc: BaseException) -> str:
    if isinstance(
        exc,
        (
            CompanionError,
            CloudflaredError,
            TelegramMenuError,
            TelegramSettingsError,
            RuntimeStatusError,
            PlatformSupportError,
        ),
    ):
        return str(exc)[:300]
    return f"Unexpected companion failure ({type(exc).__name__})."


class _GenerationResult(enum.Enum):
    RECONNECT = "reconnect"
    OWNER_CHANGED = "owner_changed"
    MENU_BLOCKED = "menu_blocked"
    TELEGRAM_BLOCKED = "telegram_blocked"
    SETTINGS_BLOCKED = "settings_blocked"
    DISABLED = "disabled"


@dataclass(frozen=True)
class _GenerationContext:
    menu: TelegramMenuManager
    settings_observer: TelegramSettingsObserver
    owner: int
    button_text: str
    core_port: int
    status: RuntimeStatus
    stop_event: asyncio.Event
    server_task: asyncio.Task[None]


def _retry_delay(failure_count: int) -> float:
    index = max(0, min(int(failure_count) - 1, len(_RETRY_BASE_SEC) - 1))
    base = _RETRY_BASE_SEC[index]
    return min(60.0, max(0.1, base * random.uniform(0.8, 1.2)))


async def _wait_or_stop(
    delay: float,
    stop_event: asyncio.Event,
    *,
    server_task: asyncio.Task[None] | None = None,
) -> None:
    stop_task = asyncio.create_task(stop_event.wait())
    waiters: set[asyncio.Task[Any]] = {stop_task}
    if server_task is not None:
        waiters.add(server_task)
    try:
        done, _pending = await asyncio.wait(
            waiters,
            timeout=max(0.0, float(delay)),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if server_task is not None and server_task in done:
            try:
                await server_task
            except asyncio.CancelledError as exc:
                raise SidecarStoppedError("Authentication sidecar was cancelled unexpectedly.") from exc
            except Exception as exc:
                raise SidecarStoppedError("Authentication sidecar failed unexpectedly.") from exc
            raise SidecarStoppedError("Authentication sidecar stopped unexpectedly.")
        if stop_task in done and stop_event.is_set():
            raise ShutdownRequested("Companion shutdown was requested.")
    finally:
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)


async def _wait_for_owner(
    settings_observer: TelegramSettingsObserver,
    token: str,
    state_dir: Path,
    telegram_client: httpx.AsyncClient,
    status: RuntimeStatus,
    stop_event: asyncio.Event,
) -> tuple[int, TelegramMenuManager]:
    failures = 0
    while True:
        require_running(stop_event)
        if not miniapp_enabled(state_dir):
            status.transition(
                "disabled",
                "Telegram Mini App is off; the text bridge remains active.",
                reason_code="miniapp_disabled",
            )
            await _wait_or_stop(5.0, stop_event)
            continue
        try:
            owner = settings_observer.owner_chat_id()
        except TelegramSettingsError as exc:
            status.transition(
                "blocked",
                _safe_error(exc),
                reason_code="owner_binding_conflict",
            )
            await _wait_or_stop(30.0, stop_event)
            continue
        if owner <= 0:
            status.transition(
                "waiting_owner",
                "Open the private Telegram bot chat once to pin its owner.",
                reason_code="waiting_private_owner",
            )
            await _wait_or_stop(5.0, stop_event)
            continue
        menu = TelegramMenuManager(token, owner, state_dir, client=telegram_client,
                                   api_root=telegram_api_base(state_dir))
        try:
            await _run_startup_operation(
                menu.verify_private_owner(),
                stop_event=stop_event,
                timeout_sec=15.0,
            )
            if not miniapp_enabled(state_dir):
                continue
            return owner, menu
        except (TelegramMenuTransportError, asyncio.TimeoutError) as exc:
            failures += 1
            delay = _retry_delay(failures)
            status.transition(
                "degraded",
                _safe_error(exc),
                reason_code="telegram_owner_unobservable",
                attempt=failures,
                next_retry_at_epoch=int(time.time() + delay),
            )
            await _wait_or_stop(delay, stop_event)
        except TelegramMenuRejectedError as exc:
            status.transition(
                "blocked",
                _safe_error(exc),
                reason_code="telegram_owner_rejected",
            )
            await _wait_or_stop(30.0, stop_event)
        except TelegramMenuError as exc:
            status.transition(
                "blocked",
                _safe_error(exc),
                reason_code="telegram_owner_invalid",
            )
            await _wait_or_stop(30.0, stop_event)


async def _wait_for_core_and_owner(
    settings_observer: TelegramSettingsObserver,
    owner: int,
    core_port: int,
    status: RuntimeStatus,
    stop_event: asyncio.Event,
) -> bool | _GenerationResult:
    failures = 0
    while True:
        require_running(stop_event)
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        try:
            if settings_observer.owner_chat_id() != owner:
                return False
        except TelegramSettingsError as exc:
            status.transition(
                "blocked",
                _safe_error(exc),
                reason_code="owner_binding_conflict",
            )
            await _wait_or_stop(30.0, stop_event)
            continue
        if not await probe_core(core_port):
            if not miniapp_enabled(settings_observer.state_dir):
                return _GenerationResult.DISABLED
            failures += 1
            delay = _retry_delay(failures)
            status.transition(
                "degraded",
                "Local Ouroboros is not ready; the Mini App remains private and will retry.",
                reason_code="core_unavailable",
                attempt=failures,
                next_retry_at_epoch=int(time.time() + delay),
            )
            await _wait_or_stop(delay, stop_event)
            continue
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        try:
            if not settings_observer.safe_for_exposure(owner):
                raise TelegramSettingsError("Telegram owner or Mini App setting changed.")
            return True
        except TelegramSettingsError as exc:
            status.transition(
                "blocked",
                _safe_error(exc),
                reason_code="telegram_settings_error",
            )
            await _wait_or_stop(30.0, stop_event)


async def _wait_for_cloudflared(
    state_dir: Path,
    status: RuntimeStatus,
    stop_event: asyncio.Event,
    *,
    server_task: asyncio.Task[None] | None = None,
) -> Path | _GenerationResult:
    failures = 0
    while True:
        require_running(stop_event)
        if not miniapp_enabled(state_dir):
            return _GenerationResult.DISABLED
        try:
            binary = await _run_startup_operation(
                ensure_cloudflared(state_dir),
                stop_event=stop_event,
                timeout_sec=45.0,
            )
            return binary if miniapp_enabled(state_dir) else _GenerationResult.DISABLED
        except (CloudflaredError, asyncio.TimeoutError) as exc:
            if not miniapp_enabled(state_dir):
                return _GenerationResult.DISABLED
            failures += 1
            delay = _retry_delay(failures)
            status.transition(
                "reconnecting",
                _safe_error(exc),
                reason_code="cloudflared_unavailable",
                attempt=failures,
                next_retry_at_epoch=int(time.time() + delay),
            )
            await _wait_or_stop(delay, stop_event, server_task=server_task)


async def _wait_for_public_verification(
    public_url: str,
    tunnel: QuickTunnel,
    settings_observer: TelegramSettingsObserver,
    owner: int,
    status: RuntimeStatus,
    stop_event: asyncio.Event,
    server_task: asyncio.Task[None],
) -> _GenerationResult | None:
    confirmed_bad = 0
    while True:
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        try:
            if settings_observer.owner_chat_id() != owner:
                return _GenerationResult.OWNER_CHANGED
            if not settings_observer.safe_for_exposure(owner):
                return _GenerationResult.SETTINGS_BLOCKED
        except TelegramSettingsError:
            return _GenerationResult.SETTINGS_BLOCKED
        outcome = await probe_public_gateway_once(
            public_url,
            tunnel,
            stop_event=stop_event,
        )
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        if outcome is PublicProbeOutcome.READY:
            return None
        if outcome is PublicProbeOutcome.TUNNEL_EXITED:
            return _GenerationResult.RECONNECT
        if outcome is PublicProbeOutcome.UNHEALTHY_RESPONSE:
            confirmed_bad += 1
            status.transition(
                "verifying",
                "The public endpoint responded without the exact Mini App marker.",
                reason_code="public_marker_unhealthy",
                attempt=confirmed_bad,
            )
            if confirmed_bad >= 3:
                return _GenerationResult.RECONNECT
        else:
            status.transition(
                "degraded",
                "This machine cannot currently observe the public tunnel; it will not rotate blindly.",
                reason_code="public_observer_unavailable",
            )
        await _wait_or_stop(2.0, stop_event, server_task=server_task)


async def _sync_menu_until_confirmed(
    public_url: str,
    tunnel: QuickTunnel,
    context: _GenerationContext,
) -> _GenerationResult | None:
    menu = context.menu
    settings_observer = context.settings_observer
    owner = context.owner
    button_text = context.button_text
    status = context.status
    stop_event = context.stop_event
    server_task = context.server_task
    failures = 0
    while True:
        require_running(stop_event)
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        if tunnel.returncode is not None:
            return _GenerationResult.RECONNECT
        try:
            if settings_observer.owner_chat_id() != owner:
                return _GenerationResult.OWNER_CHANGED
            if not settings_observer.safe_for_exposure(owner):
                return _GenerationResult.SETTINGS_BLOCKED
        except TelegramSettingsError as exc:
            failures += 1
            if failures >= 3:
                return _GenerationResult.SETTINGS_BLOCKED
            delay = _retry_delay(failures)
            status.transition(
                "degraded",
                _safe_error(exc),
                reason_code="owner_or_settings_unobservable",
                attempt=failures,
                next_retry_at_epoch=int(time.time() + delay),
            )
            await _wait_or_stop(delay, stop_event, server_task=server_task)
            continue
        try:
            await _run_startup_operation(
                menu.install(public_url, button_text),
                stop_event=stop_event,
                timeout_sec=15.0,
            )
            if not miniapp_enabled(settings_observer.state_dir):
                return _GenerationResult.DISABLED
            if tunnel.returncode is not None:
                return _GenerationResult.RECONNECT
            if settings_observer.owner_chat_id() != owner:
                return _GenerationResult.OWNER_CHANGED
            if not settings_observer.safe_for_exposure(owner):
                return _GenerationResult.SETTINGS_BLOCKED
            return None
        except TelegramSettingsError:
            return _GenerationResult.SETTINGS_BLOCKED
        except TelegramMenuConflictError:
            return _GenerationResult.MENU_BLOCKED
        except TelegramMenuRejectedError:
            return _GenerationResult.TELEGRAM_BLOCKED
        except TelegramMenuTransportError as exc:
            if not miniapp_enabled(settings_observer.state_dir):
                return _GenerationResult.DISABLED
            failures += 1
            delay = _retry_delay(failures)
            status.transition(
                "syncing_menu",
                _safe_error(exc),
                reason_code="menu_transport_unavailable",
                attempt=failures,
                next_retry_at_epoch=int(time.time() + delay),
            )
            await _wait_or_stop(delay, stop_event, server_task=server_task)
        except asyncio.TimeoutError:
            if not miniapp_enabled(settings_observer.state_dir):
                return _GenerationResult.DISABLED
            failures += 1
            delay = _retry_delay(failures)
            status.transition(
                "syncing_menu",
                "Telegram menu operation timed out.",
                reason_code="menu_transport_unavailable",
                attempt=failures,
                next_retry_at_epoch=int(time.time() + delay),
            )
            await _wait_or_stop(delay, stop_event, server_task=server_task)
        except TelegramMenuError:
            return _GenerationResult.MENU_BLOCKED


async def _monitor_generation(
    public_url: str,
    tunnel: QuickTunnel,
    context: _GenerationContext,
) -> tuple[_GenerationResult, bool]:
    menu = context.menu
    settings_observer = context.settings_observer
    owner = context.owner
    button_text = context.button_text
    core_port = context.core_port
    status = context.status
    stop_event = context.stop_event
    server_task = context.server_task
    loop = asyncio.get_running_loop()
    ready_since = loop.time()
    next_core = loop.time() + _CORE_HEALTH_INTERVAL_SEC
    next_public = loop.time() + _PUBLIC_HEALTH_INTERVAL_SEC
    next_menu = loop.time() + _MENU_HEALTH_INTERVAL_SEC
    next_owner = loop.time() + _OWNER_HEALTH_INTERVAL_SEC
    confirmed_bad = 0
    degraded_reasons: set[str] = set()
    tunnel_wait = asyncio.create_task(tunnel.wait())
    stop_wait = asyncio.create_task(stop_event.wait())
    try:
        while True:
            if not miniapp_enabled(settings_observer.state_dir):
                return _GenerationResult.DISABLED, False
            done, _pending = await asyncio.wait(
                {tunnel_wait, stop_wait, server_task},
                timeout=_MONITOR_TICK_SEC,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_wait in done and stop_event.is_set():
                raise ShutdownRequested("Companion shutdown was requested.")
            if server_task in done:
                try:
                    await server_task
                except asyncio.CancelledError as exc:
                    raise SidecarStoppedError(
                        "Authentication sidecar was cancelled unexpectedly."
                    ) from exc
                except Exception as exc:
                    raise SidecarStoppedError(
                        "Authentication sidecar failed unexpectedly."
                    ) from exc
                raise SidecarStoppedError("Authentication sidecar stopped unexpectedly.")
            if tunnel_wait in done:
                return _GenerationResult.RECONNECT, loop.time() - ready_since >= _STABLE_READY_RESET_SEC
            now = loop.time()
            if now >= next_owner:
                next_owner = now + _OWNER_HEALTH_INTERVAL_SEC
                try:
                    if not miniapp_enabled(settings_observer.state_dir):
                        return _GenerationResult.DISABLED, False
                    if settings_observer.owner_chat_id() != owner:
                        return _GenerationResult.OWNER_CHANGED, False
                    if not settings_observer.safe_for_exposure(owner):
                        return _GenerationResult.SETTINGS_BLOCKED, False
                except TelegramSettingsError:
                    return _GenerationResult.SETTINGS_BLOCKED, False
                else:
                    degraded_reasons.discard("owner_or_settings_unobservable")
            if now >= next_core:
                next_core = now + _CORE_HEALTH_INTERVAL_SEC
                if await probe_core(core_port):
                    degraded_reasons.discard("core_unavailable")
                else:
                    degraded_reasons.add("core_unavailable")
            if now >= next_public:
                next_public = now + _PUBLIC_HEALTH_INTERVAL_SEC
                outcome = await probe_public_gateway_once(
                    public_url,
                    tunnel,
                    stop_event=stop_event,
                )
                if outcome is PublicProbeOutcome.TUNNEL_EXITED:
                    return (
                        _GenerationResult.RECONNECT,
                        loop.time() - ready_since >= _STABLE_READY_RESET_SEC,
                    )
                if outcome is PublicProbeOutcome.READY:
                    confirmed_bad = 0
                    degraded_reasons.discard("public_observer_unavailable")
                    degraded_reasons.discard("public_marker_unhealthy")
                elif outcome is PublicProbeOutcome.UNHEALTHY_RESPONSE:
                    confirmed_bad += 1
                    degraded_reasons.discard("public_observer_unavailable")
                    degraded_reasons.add("public_marker_unhealthy")
                    if confirmed_bad >= 3:
                        return (
                            _GenerationResult.RECONNECT,
                            loop.time() - ready_since >= _STABLE_READY_RESET_SEC,
                        )
                else:
                    degraded_reasons.discard("public_marker_unhealthy")
                    degraded_reasons.add("public_observer_unavailable")
            if now >= next_menu:
                next_menu = now + _MENU_HEALTH_INTERVAL_SEC
                try:
                    await _run_startup_operation(
                        menu.check_owned(public_url, button_text),
                        stop_event=stop_event,
                        timeout_sec=15.0,
                    )
                    degraded_reasons.discard("menu_transport_unavailable")
                except TelegramMenuConflictError:
                    return _GenerationResult.MENU_BLOCKED, False
                except TelegramMenuRejectedError:
                    return _GenerationResult.TELEGRAM_BLOCKED, False
                except (TelegramMenuTransportError, asyncio.TimeoutError):
                    degraded_reasons.add("menu_transport_unavailable")
                except TelegramMenuError:
                    return _GenerationResult.MENU_BLOCKED, False
            if degraded_reasons:
                reason = sorted(degraded_reasons)[0]
                if status.state != "degraded" or status.reason_code != reason:
                    status.transition(
                        "degraded",
                        "Mini App remains exposed through the last verified URL while a health signal recovers.",
                        reason_code=reason,
                    )
            elif status.state != "ready":
                status.transition(
                    "ready",
                    "Existing Ouroboros SPA is available from the private Telegram bot menu.",
                    reason_code="healthy",
                    public_url=public_url,
                )
    finally:
        stop_wait.cancel()
        if not tunnel_wait.done():
            tunnel_wait.cancel()
        await asyncio.gather(stop_wait, tunnel_wait, return_exceptions=True)


async def _wait_blocked_state(
    reason: _GenerationResult,
    menu: TelegramMenuManager,
    settings_observer: TelegramSettingsObserver,
    owner: int,
    status: RuntimeStatus,
    stop_event: asyncio.Event,
    server_task: asyncio.Task[None],
) -> _GenerationResult:
    messages = {
        _GenerationResult.MENU_BLOCKED: (
            "Telegram menu changed outside this skill; public exposure is stopped.",
            "menu_drift",
        ),
        _GenerationResult.TELEGRAM_BLOCKED: (
            "Telegram rejected the bot configuration; disable and re-enable after fixing it.",
            "telegram_rejected",
        ),
        _GenerationResult.SETTINGS_BLOCKED: (
            "Telegram owner or Mini App setting changed; public exposure is stopped.",
            "telegram_settings_drift",
        ),
    }
    message, reason_code = messages[reason]
    while True:
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        status.transition("blocked", message, reason_code=reason_code)
        await _wait_or_stop(30.0, stop_event, server_task=server_task)
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        try:
            if settings_observer.owner_chat_id() != owner:
                return _GenerationResult.OWNER_CHANGED
            if reason is _GenerationResult.MENU_BLOCKED:
                await _run_startup_operation(
                    menu.check_snapshot_owned(),
                    stop_event=stop_event,
                    timeout_sec=15.0,
                )
                return _GenerationResult.RECONNECT
            if reason is _GenerationResult.SETTINGS_BLOCKED:
                if settings_observer.safe_for_exposure(owner):
                    return _GenerationResult.RECONNECT
        except (TelegramSettingsError, TelegramMenuTransportError, asyncio.TimeoutError):
            continue
        except TelegramMenuError:
            continue


async def _run_tunnel_generations(
    binary: Path,
    state_dir: Path,
    sidecar_port: int,
    sidecar: TelegramProxySidecar,
    context: _GenerationContext,
) -> _GenerationResult:
    menu = context.menu
    settings_observer = context.settings_observer
    owner = context.owner
    status = context.status
    stop_event = context.stop_event
    server_task = context.server_task
    failures = 0
    blocked: _GenerationResult | None = None
    # External tunnel outages are reconciled in-process with bounded backoff and
    # visible status. Host max_restarts owns process crashes, not network uptime.
    while True:
        if blocked is not None:
            blocked_result = await _wait_blocked_state(
                blocked,
                menu,
                settings_observer,
                owner,
                status,
                stop_event,
                server_task,
            )
            if blocked_result in {_GenerationResult.OWNER_CHANGED, _GenerationResult.DISABLED}:
                return blocked_result
            blocked = None
        require_running(stop_event)
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        if failures:
            binary_result = await _wait_for_cloudflared(
                state_dir,
                status,
                stop_event,
                server_task=server_task,
            )
            if binary_result is _GenerationResult.DISABLED:
                return binary_result
            binary = binary_result
        tunnel = QuickTunnel(binary, state_dir, sidecar_port)
        result = _GenerationResult.RECONNECT
        stable_ready = False
        failure_message = ""
        try:
            status.transition(
                "reconnecting" if failures else "starting",
                "Starting a pinned Cloudflare Quick Tunnel to the authentication sidecar.",
                reason_code="starting_tunnel",
                attempt=failures,
            )
            try:
                await _run_startup_operation(
                    tunnel.start(),
                    stop_event=stop_event,
                    timeout_sec=10.0,
                )
                public_url = await _run_startup_operation(
                    tunnel.wait_url(timeout_sec=30),
                    stop_event=stop_event,
                    timeout_sec=31.0,
                )
            except asyncio.TimeoutError as exc:
                raise CloudflaredError("Tunnel startup operation timed out.") from exc
            sidecar.set_public_url(public_url)
            status.transition(
                "verifying",
                "Verifying the public authentication boundary.",
                reason_code="verifying_public_gateway",
            )
            verification_result = await _wait_for_public_verification(
                public_url,
                tunnel,
                settings_observer,
                owner,
                status,
                stop_event,
                server_task,
            )
            if verification_result is not None:
                result = verification_result
            else:
                status.transition(
                    "syncing_menu",
                    "Synchronizing the private Telegram Mini App button.",
                    reason_code="syncing_menu",
                )
                sync_result = await _sync_menu_until_confirmed(
                    public_url,
                    tunnel,
                    context,
                )
                if sync_result is not None:
                    result = sync_result
                else:
                    status.transition(
                        "ready",
                        "Existing Ouroboros SPA is available from the private Telegram bot menu.",
                        reason_code="healthy",
                        public_url=public_url,
                    )
                    result, stable_ready = await _monitor_generation(
                        public_url,
                        tunnel,
                        context,
                    )
        except (CloudflaredError, CompanionError, PublicGatewayError) as exc:
            if isinstance(exc, (ShutdownRequested, SidecarStoppedError)):
                raise
            if not miniapp_enabled(settings_observer.state_dir):
                result = _GenerationResult.DISABLED
            else:
                result = _GenerationResult.RECONNECT
                failure_message = _safe_error(exc)
                status.transition(
                    "reconnecting",
                    failure_message,
                    reason_code="tunnel_generation_failed",
                    attempt=failures + 1,
                )
        finally:
            sidecar.clear_public_url()
            await tunnel.stop()
        if result is _GenerationResult.OWNER_CHANGED:
            return result
        if result is _GenerationResult.DISABLED:
            return result
        if result in {
            _GenerationResult.MENU_BLOCKED,
            _GenerationResult.TELEGRAM_BLOCKED,
            _GenerationResult.SETTINGS_BLOCKED,
        }:
            blocked = result
            continue
        failures = 0 if stable_ready else failures + 1
        if not miniapp_enabled(settings_observer.state_dir):
            return _GenerationResult.DISABLED
        delay = _retry_delay(max(1, failures))
        status.transition(
            "reconnecting",
            failure_message or "The tunnel generation ended; retrying inside the same companion.",
            reason_code=(
                "tunnel_generation_failed" if failure_message else "tunnel_reconnect_backoff"
            ),
            attempt=failures,
            next_retry_at_epoch=int(time.time() + delay),
        )
        await _wait_or_stop(delay, stop_event, server_task=server_task)


async def _reconcile_owner_change(
    menu: TelegramMenuManager | None,
    _settings_observer: TelegramSettingsObserver,
    status: RuntimeStatus,
    stop_event: asyncio.Event,
) -> None:
    """Restore the durable menu snapshot before accepting a new owner."""

    if menu is None:
        return
    failures = 0
    while True:
        try:
            await _run_startup_operation(
                menu.restore(),
                stop_event=stop_event,
                timeout_sec=15.0,
            )
            return
        except ShutdownRequested:
            raise
        except (TelegramMenuTransportError, asyncio.TimeoutError) as exc:
            failures += 1
            delay = _retry_delay(failures)
            error = exc
        except TelegramMenuError as exc:
            failures += 1
            delay = 30.0
            error = exc
        status.transition(
            "reconciling",
            _safe_error(error),
            reason_code="owner_change_rollback_pending",
            attempt=failures,
            next_retry_at_epoch=int(time.time() + delay),
        )
        await _wait_or_stop(delay, stop_event)

def _prior_owner_reconciliation(
    token: str,
    current_owner: int,
    state_dir: Path,
    telegram_client: httpx.AsyncClient,
    _settings_observer: TelegramSettingsObserver,
) -> tuple[bool, TelegramMenuManager | None]:
    menu_owner = snapshot_owner_chat_id(state_dir)
    if not isinstance(menu_owner, int) or menu_owner <= 0 or menu_owner == current_owner:
        return False, None
    prior_menu = TelegramMenuManager(
        token,
        menu_owner,
        state_dir,
        client=telegram_client,
        api_root=telegram_api_base(state_dir),
    )
    return True, prior_menu


async def run() -> int:
    state_dir: Path | None = None
    sidecar: TelegramProxySidecar | None = None
    server: uvicorn.Server | None = None
    server_task: asyncio.Task[None] | None = None
    menu: TelegramMenuManager | None = None
    settings_observer: TelegramSettingsObserver | None = None
    telegram_client: httpx.AsyncClient | None = None
    lifeline_cancel: threading.Event | None = None
    lifeline_thread: threading.Thread | None = None
    clean_shutdown = False
    rollback_complete = False
    fatal_error_message = ""
    stop_event = asyncio.Event()
    heartbeat_stop = asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None
    status: RuntimeStatus | None = None
    singleton_fd: int | None = None
    sidecar_failures = 0

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop_event.set)

    try:
        require_isolated_process_group()
        state_raw = os.environ.get("OUROBOROS_SKILL_STATE_DIR", "")
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not state_raw:
            raise CompanionError("OUROBOROS_SKILL_STATE_DIR is missing.")
        state_dir = _safe_state_dir(state_raw)
        singleton_fd = acquire_singleton(state_dir)
        status = RuntimeStatus(state_dir, cloudflared_version=CLOUDFLARED_VERSION)
        status.transition(
            "starting",
            "Checking local Ouroboros and Telegram owner.",
            reason_code="initializing",
        )
        heartbeat_task = asyncio.create_task(
            status.heartbeat(heartbeat_stop, failure_event=stop_event)
        )
        if not token:
            raise CompanionError("TELEGRAM_BOT_TOKEN is missing or not granted.")
        parent_pid = _server_parent_pid()
        if not parent_pid or not _pid_alive(parent_pid):
            raise CompanionError("Ouroboros server parent is unavailable.")
        lifeline_cancel, lifeline_thread = start_parent_lifeline(
            parent_pid,
            loop,
            stop_event,
        )
        config = load_runtime_config(state_dir)

        telegram_client = httpx.AsyncClient(
            timeout=12,
            follow_redirects=False,
            trust_env=False,
            proxy=telegram_proxy(state_dir),
        )
        settings_observer = TelegramSettingsObserver(state_dir, config["core_port"])
        while True:
            owner, candidate_menu = await _wait_for_owner(
                settings_observer,
                token,
                state_dir,
                telegram_client,
                status,
                stop_event,
            )
            try:
                reconcile_prior_owner, prior_menu = _prior_owner_reconciliation(
                    token,
                    owner,
                    state_dir,
                    telegram_client,
                    settings_observer,
                )
            except (TelegramMenuError, TelegramSettingsError) as exc:
                status.transition(
                    "blocked",
                    _safe_error(exc),
                    reason_code="menu_snapshot_owner_invalid",
                )
                await _wait_or_stop(30.0, stop_event)
                continue
            if reconcile_prior_owner:
                # Keep the previous owner in the outer cleanup slot so a stop
                # during compensation retries the correct Telegram chat.
                menu = prior_menu
                await _reconcile_owner_change(menu, settings_observer, status, stop_event)
                menu = None
                # Owner settings may have changed again while compensation was
                # in flight. Re-read and re-verify before any new exposure.
                continue
            menu = candidate_menu
            # Complete the potentially slow first download before exposure.
            binary_result = await _wait_for_cloudflared(state_dir, status, stop_event)
            if binary_result is _GenerationResult.DISABLED:
                menu = None
                continue
            binary = binary_result
            owner_result = await _wait_for_core_and_owner(
                settings_observer,
                owner,
                config["core_port"],
                status,
                stop_event,
            )
            if owner_result is _GenerationResult.DISABLED:
                menu = None
                continue
            if not owner_result:
                menu = None
                continue
            sidecar = TelegramProxySidecar(
                token,
                owner,
                config["core_port"],
                seams=SidecarSeams(
                    exposure_guard=lambda: settings_observer.safe_for_exposure(owner)
                ),
            )
            status.set_metrics_provider(sidecar.diagnostics)
            try:
                server, server_task, sidecar_port = await _run_startup_operation(
                    start_sidecar(sidecar),
                    stop_event=stop_event,
                    timeout_sec=6.0,
                )
            except (CompanionError, OSError, asyncio.TimeoutError) as exc:
                if isinstance(exc, ShutdownRequested):
                    raise
                sidecar_failures += 1
                delay = _retry_delay(sidecar_failures)
                status.transition(
                    "reconnecting",
                    _safe_error(exc),
                    reason_code="sidecar_start_failed",
                    attempt=sidecar_failures,
                    next_retry_at_epoch=int(time.time() + delay),
                )
                await stop_sidecar(server, server_task, sidecar)
                status.set_metrics_provider(None)
                sidecar = None
                server = None
                server_task = None
                await _wait_or_stop(delay, stop_event)
                continue
            sidecar_started_at = time.monotonic()
            sidecar_failure: SidecarStoppedError | None = None
            try:
                try:
                    generation_context = _GenerationContext(
                        menu=menu,
                        settings_observer=settings_observer,
                        owner=owner,
                        button_text=config["button_text"],
                        core_port=config["core_port"],
                        status=status,
                        stop_event=stop_event,
                        server_task=server_task,
                    )
                    result = await _run_tunnel_generations(
                        binary,
                        state_dir,
                        sidecar_port,
                        sidecar,
                        generation_context,
                    )
                except SidecarStoppedError as exc:
                    sidecar_failure = exc
            finally:
                sidecar.clear_public_url()
                await stop_sidecar(server, server_task, sidecar)
                status.set_metrics_provider(None)
                sidecar = None
                server = None
                server_task = None
            if sidecar_failure is not None:
                if time.monotonic() - sidecar_started_at >= _STABLE_READY_RESET_SEC:
                    sidecar_failures = 0
                sidecar_failures += 1
                delay = _retry_delay(sidecar_failures)
                status.transition(
                    "reconnecting",
                    _safe_error(sidecar_failure),
                    reason_code="sidecar_restarting",
                    attempt=sidecar_failures,
                    next_retry_at_epoch=int(time.time() + delay),
                )
                await _wait_or_stop(delay, stop_event)
                continue
            if result is _GenerationResult.OWNER_CHANGED:
                # Rebind only after the old owner's menu and settings state are
                # restored. Any conflict remains visible and fail-closed.
                await _reconcile_owner_change(menu, settings_observer, status, stop_event)
                sidecar_failures = 0
                menu = None
                continue
            if result is _GenerationResult.DISABLED:
                await _reconcile_owner_change(menu, settings_observer, status, stop_event)
                menu = None
                continue
    except ShutdownRequested:
        heartbeat_failure: BaseException | None = None
        if heartbeat_task is not None and heartbeat_task.done() and not heartbeat_task.cancelled():
            with contextlib.suppress(asyncio.CancelledError):
                heartbeat_failure = heartbeat_task.exception()
        if heartbeat_failure is not None:
            fatal_error_message = _safe_error(heartbeat_failure)
            if status is not None:
                with contextlib.suppress(Exception):
                    status.transition(
                        "error",
                        fatal_error_message,
                        reason_code="status_heartbeat_failed",
                    )
            return 1
        clean_shutdown = True
        return 0
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        fatal_error_message = _safe_error(exc)
        if status is not None and singleton_fd is not None:
            with contextlib.suppress(Exception):
                status.transition(
                    "error",
                    _safe_error(exc),
                    reason_code="fatal_companion_error",
                )
        return 1
    finally:
        if status is not None:
            with contextlib.suppress(Exception):
                status.transition(
                    "stopping",
                    "Stopping public exposure and reconciling owned settings.",
                    reason_code="shutdown",
                )
        if sidecar is not None:
            sidecar.clear_public_url()

        async def restore_menu() -> bool:
            if menu is None:
                return True
            # The host allows five seconds before hard-killing a companion.
            # A restore may issue get/set/get, so bound each request separately
            # and let the complete transaction fit inside the 2.5s envelope.
            await menu.restore(
                request_timeout_sec=_SHUTDOWN_TELEGRAM_REQUEST_TIMEOUT_SEC,
            )
            return True

        async def close_sidecar() -> None:
            await stop_sidecar(server, server_task, sidecar)

        cleanup_results = await asyncio.gather(
            asyncio.wait_for(restore_menu(), timeout=2.5),
            asyncio.wait_for(close_sidecar(), timeout=0.75),
            return_exceptions=True,
        )
        rollback_complete = cleanup_results[0] is True
        if telegram_client is not None:
            with contextlib.suppress(Exception):
                await telegram_client.aclose()
        if clean_shutdown and status is not None:
            with contextlib.suppress(Exception):
                if rollback_complete:
                    status.transition(
                        "stopped",
                        "Mini App stopped; the prior Telegram button is restored.",
                        reason_code="clean_shutdown",
                    )
                else:
                    status.transition(
                        "rollback_pending",
                        "Mini App stopped; durable rollback state remains for the next enable.",
                        reason_code="rollback_pending",
                    )
        elif fatal_error_message and status is not None:
            with contextlib.suppress(Exception):
                status.transition(
                    "error",
                    fatal_error_message,
                    reason_code=(
                        "fatal_error" if rollback_complete else "fatal_error_rollback_pending"
                    ),
                )
        heartbeat_stop.set()
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if lifeline_cancel is not None:
            lifeline_cancel.set()
        if lifeline_thread is not None:
            lifeline_thread.join(timeout=0.5)
        release_singleton(singleton_fd)


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
