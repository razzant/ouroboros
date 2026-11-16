"""Playwright browser tools: one browser generation per ToolContext, owned by its own event loop.

The tools are synchronous facades: generation pins, retirement, refusals and results live
here on the calling thread; each call's Playwright work is one coroutine on its generation's
loop (``browser_session``), which also keeps answering the page's requests between calls.
"""

from __future__ import annotations

import base64
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import functools
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urlsplit

try:
    from playwright_stealth import Stealth
    _HAS_STEALTH = True
except ImportError:
    _HAS_STEALTH = False

from ouroboros import browser_policy, chromium_runtime
from ouroboros.tools import browser_session
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter, ToolResult, _compose_execute_result, _publish_tool_result,
    append_published_text,
)

log = logging.getLogger(__name__)

_playwright_ready = False
_playwright_ready_engines: set[tuple[str, str]] = set()
_playwright_browsers_path_managed = False
_SUPPORTED_BROWSER_ENGINES = frozenset({"chromium", "webkit"})


@dataclasses.dataclass
class _BrowserCall:
    ctx: ToolContext
    generation: Any


_browser_call: contextvars.ContextVar[_BrowserCall | None] = contextvars.ContextVar("browser_call", default=None)


@contextlib.contextmanager
def browser_call_scope(ctx: ToolContext, generation: Any = None):
    """An invocation's expectation, advanced only by its own replacements.

    The loop binds the submitted generation before safety checks; direct calls
    bind at entry. Nested browser helpers share the cell, concurrent calls do
    not. Resetting also keeps a reused worker from inheriting a finished call.
    """
    current = _browser_call.get()
    if generation is None and current is not None and current.ctx is ctx:
        yield current
        return
    call = _BrowserCall(ctx, generation if generation is not None else ctx.browser_state)
    token = _browser_call.set(call)
    try:
        yield call
    finally:
        _browser_call.reset(token)


def _bound_browser_call(fn):
    @functools.wraps(fn)
    def bound(ctx, *args, **kwargs):
        with browser_call_scope(ctx):
            return fn(ctx, *args, **kwargs)
    return bound


_runtime_mode_for_browser = browser_policy.browser_runtime_mode  # one owner for every browser surface


def _normalize_browser_engine(engine: str = "") -> str:
    value = str(engine or "chromium").strip().lower()
    if value not in _SUPPORTED_BROWSER_ENGINES:
        raise ValueError("browser engine must be 'chromium' or 'webkit'")
    return value


# Reuse the effective child contract: Cyber acting children retain agency;
# explicitly readonly children keep their assigned access contract.
from ouroboros.tools.core import is_restricted_subagent_profile as _readonly_subagent  # noqa: E402


def _has_platform_browser(local_browsers_dir: pathlib.Path, engine: str = "chromium") -> bool:
    """Return True when this root holds a usable browser: WebKit for this platform, or the
    exact pinned Chromium (never Playwright's own Chromium build)."""
    engine = _normalize_browser_engine(engine)
    if not local_browsers_dir.is_dir():
        return False
    if engine == "webkit":
        for webkit_dir in local_browsers_dir.iterdir():
            if not webkit_dir.name.startswith("webkit-"):
                continue
            for executable_name in (
                "pw_run.sh",
                "MiniBrowser",
                "MiniBrowser.exe",
                "Playwright.exe",
                "WebKitWebProcess",
                "WebKitWebProcess.exe",
            ):
                if any(candidate.is_file() for candidate in webkit_dir.rglob(executable_name)):
                    return True
        return False
    try:
        return chromium_runtime.installed_executable(local_browsers_dir, chromium_runtime.host_pin()) is not None
    except chromium_runtime.ChromiumRefusal:
        return False


def _has_platform_chromium(local_browsers_dir: pathlib.Path) -> bool:
    return _has_platform_browser(local_browsers_dir, "chromium")


def _has_platform_webkit(local_browsers_dir: pathlib.Path) -> bool:
    return _has_platform_browser(local_browsers_dir, "webkit")


def _set_playwright_browsers_path_if_bundled() -> None:
    """Use bundled Playwright browsers in packaged builds; respect explicit env override."""
    global _playwright_browsers_path_managed, _playwright_ready
    if "PLAYWRIGHT_BROWSERS_PATH" in os.environ:
        return
    try:
        import playwright as _pw_pkg
        pkg_root = pathlib.Path(_pw_pkg.__file__).parent
        local_browsers = pkg_root / "driver" / "package" / ".local-browsers"
        bundled = [
            engine for engine in sorted(_SUPPORTED_BROWSER_ENGINES)
            if _has_platform_browser(local_browsers, engine)
        ]
        if bundled:
            if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") != "0":
                _playwright_ready_engines.clear()
                _playwright_ready = False
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
            _playwright_browsers_path_managed = True
            log.debug("Bundled Playwright browsers detected (%s) — set PLAYWRIGHT_BROWSERS_PATH=0", ", ".join(bundled))
    except Exception:
        pass  # non-fatal; fall through to standard cache lookup


_set_playwright_browsers_path_if_bundled()


def _ensure_playwright_installed(*, engine: str = "chromium", allow_install: bool = True):
    """Install Playwright and make the engine launchable: Chromium returns the pinned
    headless shell's executable path (``_pinned_chromium``); WebKit stays Playwright's
    own build and returns None."""
    global _playwright_browsers_path_managed, _playwright_ready
    engine = _normalize_browser_engine(engine)

    try:
        import playwright  # noqa: F401
    except ImportError:
        if not allow_install:
            raise RuntimeError("Browser tools are unavailable in local_readonly_subagent mode because Playwright is not already installed.")
        if getattr(sys, 'frozen', False):
            raise RuntimeError(
                "Browser tools require Playwright, which is not bundled. "
                f"Install manually: pip3 install playwright && python3 -m playwright install {engine}"
            )
        from ouroboros.deployment_policy import body_replacement_refusal

        if refusal := body_replacement_refusal():  # the package would land in the serving interpreter
            raise RuntimeError(f"Browser tools require Playwright, which is not installed. {refusal}")
        log.info("Playwright not found, installing...")
        from ouroboros.platform_layer import pip_install_target_args

        _install_beside_calls("playwright-package", "the Playwright package", lambda _progress: _diagnostic_install(
            [sys.executable, "-m", "pip", "install", *pip_install_target_args(sys.executable), "playwright"]))

    # Before every SDK start and installer subprocess, either engine: the driver runs on the pinned Node,
    # never silently on the SDK's own copy (an older signed bundle or a pip-installed SDK carries one). A local
    # one is selected at once; preparing it (~50 MB download) runs beside the calls, as a browser's does.
    from ouroboros.claudexor_runtime import ClaudexorRuntimeError, get_runtime_manager
    from ouroboros.node_runtime import select_playwright_node

    try:
        try:
            select_playwright_node()
        except ClaudexorRuntimeError:
            if not allow_install:
                raise
            node_version = getattr(get_runtime_manager().pin, "node_version", "")
            _install_beside_calls("pinned-node", f"the pinned Node {node_version} for Playwright's driver",
                                  lambda _progress: _prepare_driver_node(select_playwright_node))
    except ClaudexorRuntimeError as exc:
        if exc.code == "runtime_install_in_progress":  # another process prepares it (a boot after an Update)
            raise chromium_runtime.ChromiumRefusal(chromium_runtime.PREPARING, (
                "the pinned Node for Playwright's driver is still being installed (waiting for another Ouroboros "
                "process that is installing it); this call opened no browser. A later call uses it.")) from None
        raise RuntimeError(f"Browser tools need the reviewed Node for Playwright's driver: {exc}. "
                           "Set PLAYWRIGHT_NODEJS_PATH to choose a Node explicitly.") from exc

    if engine == "chromium":
        return _pinned_chromium(allow_install=allow_install)

    current_browser_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if not (current_browser_path and current_browser_path != "0" and not _playwright_browsers_path_managed):
        try:
            import playwright as _pw_pkg
            local_browsers = pathlib.Path(_pw_pkg.__file__).parent / "driver" / "package" / ".local-browsers"
        except Exception:
            local_browsers = None
        if local_browsers is not None and _has_platform_browser(local_browsers, engine):
            if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") != "0":
                _playwright_ready_engines.clear()
                _playwright_ready = False
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
            _playwright_browsers_path_managed = True
        elif current_browser_path == "0" or _playwright_browsers_path_managed:
            from ouroboros.config import resolve_data_dir

            data_dir = resolve_data_dir()  # the configured data root, APP_ROOT's when only it is set
            target = data_dir / "playwright-browsers"
            if target.exists() and _has_platform_browser(target, engine):
                target_str = str(target)
                if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") != target_str:
                    _playwright_ready_engines.clear()
                    _playwright_ready = False
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = target_str
                _playwright_browsers_path_managed = True
            if allow_install:
                target.mkdir(parents=True, exist_ok=True)
                target_str = str(target)
                if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") != target_str:
                    _playwright_ready_engines.clear()
                    _playwright_ready = False
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = target_str
                _playwright_browsers_path_managed = True
                log.warning("Bundled %s is unavailable; using Playwright browser cache %s", engine, target)

    key = (engine, str(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or ""))
    if _playwright_ready and key in _playwright_ready_engines:
        return
    install_key = ":".join(key)

    try:
        from playwright.sync_api import sync_playwright
        from ouroboros.browser_output import driver_diagnostics
        with driver_diagnostics(), sync_playwright() as pw:
            executable_path = pathlib.Path(str(getattr(pw, engine).executable_path))
        if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") == "0":
            try:
                import playwright as _pw_pkg
                local_browsers = pathlib.Path(_pw_pkg.__file__).parent / "driver" / "package" / ".local-browsers"
            except Exception:
                local_browsers = None
            if local_browsers is None or not _has_platform_browser(local_browsers, engine):
                raise RuntimeError(f"bundled Playwright {engine} is missing")
        elif not executable_path.exists() or install_key in _installs:  # still being installed: join it
            raise RuntimeError(f"Playwright {engine} binary not found at {executable_path}")
        log.info("Playwright %s binary found", engine)
    except Exception:
        if not allow_install:
            raise RuntimeError(
                f"Browser tools are unavailable in local_readonly_subagent mode because {engine} is not already installed."
            )
        install_python = _playwright_python()

        def install(_progress: Dict[str, Any]) -> None:  # Playwright's own installer locks its browsers root
            log.info("Installing Playwright %s dependencies and binary...", engine)
            _repair_host_dependencies(engine, install_python)
            _diagnostic_install([install_python, "-m", "playwright", "install", engine])

        _install_beside_calls(install_key, f"Playwright {engine}", install)

    _playwright_ready_engines.add((engine, str(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "")))
    _playwright_ready = True


def _playwright_python() -> str:
    """The interpreter that runs Playwright's CLI: this one, or a packaged app's embedded one."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    try:
        from ouroboros.platform_layer import embedded_python_candidates
        bases: list[pathlib.Path] = []
        frozen_base = getattr(sys, "_MEIPASS", None)
        if frozen_base:
            bases.append(pathlib.Path(frozen_base))
        exe_parent = pathlib.Path(sys.executable).resolve().parent
        bases.extend([exe_parent, exe_parent.parent])
        for base in bases:
            for candidate in embedded_python_candidates(base):
                if candidate.exists():
                    return str(candidate)
    except Exception:
        pass
    raise RuntimeError(
        "Playwright browser install requires the embedded python-standalone interpreter, "
        "but it was not found in this packaged app."
    )


def _diagnostic_install(command: list[str]) -> None:
    """Installer progress is diagnostics, unlike the browser driver's protocol."""
    from ouroboros.process_logging import relayed_child_output

    with relayed_child_output() as output:
        subprocess.check_call(command, **({"stdout": output, "stderr": output} if output is not None else {}))


def _repair_host_dependencies(engine: str, install_python: str = "") -> None:
    """Best effort: Playwright installs the engine's host libraries where it can."""
    try:
        _diagnostic_install([install_python or _playwright_python(), "-m", "playwright", "install-deps", engine])
    except Exception as exc:
        log.warning("Playwright system dependency repair failed; continuing with browser download: %s", exc)


# One preparation per browser engine and root at a time, beside the calls that need it: the first
# Chromium of a new pin (~100 MB) once kept four calls past their windows (04.11) while each started
# its own download. A call waits only what is left of its own window and then says where it stands.
_installs: Dict[str, Tuple[concurrent.futures.Future, Dict[str, Any]]] = {}
_installs_lock = threading.Lock()


def _prepare_driver_node(select) -> str:
    """Prepare and select the pin's Node for the driver; then the extensions that waited for it are asked again, as
    after the boot's preparation (``server_boot.requeue_waiting_extensions``; a failure there only logs)."""
    node = select(allow_install=True)
    try:
        from ouroboros.config import DATA_DIR
        from ouroboros.server_boot import requeue_waiting_extensions

        requeue_waiting_extensions(DATA_DIR)
    except Exception:
        log.warning("Extensions waiting for the pinned Node were not asked again", exc_info=True)
    return node


def _install_beside_calls(key: str, what: str, install: Callable[[Dict[str, Any]], Any], hint: str = "") -> Any:
    """Join or start the one background preparation for ``key``; wait at most this call's own window.

    The worker's dispatch deadline less the settlement margin bounds the wait, so the call answers
    with a typed fact before its outer wait ends; the preparation continues and a later call joins it
    or finds its result. Without a deadline the call waits as before. The starting call physically
    ends with its preparation (``hold_invocation``), so its worker is not retired mid-download. A
    failure reaches its waiters and is never sticky: the next call starts a fresh preparation."""
    from ouroboros.model_wait import dispatch_deadline_remaining_sec
    from ouroboros.runtime_limits import NESTED_SETTLEMENT_MARGIN_SEC
    from ouroboros.tool_custody import hold_invocation

    with _installs_lock:
        entry = _installs.get(key)
        if entry is None:
            entry, release = (concurrent.futures.Future(), {"started": time.monotonic()}), hold_invocation()
            try:
                threading.Thread(target=_run_install, args=(key, entry, install, release),
                                 name=f"browser-install {key}", daemon=True).start()
            except BaseException:
                release()  # nothing started: no entry for later calls to wait on
                raise
            _installs[key] = entry
    future, progress = entry
    remaining = dispatch_deadline_remaining_sec()
    try:
        return future.result(timeout=None if remaining is None else max(0.0, remaining - NESTED_SETTLEMENT_MARGIN_SEC))
    except concurrent.futures.TimeoutError:
        state = ("waiting for another Ouroboros process that is installing it" if progress.get("waiting")
                 else f"{progress['bytes']} of {progress.get('size')} bytes downloaded" if "bytes" in progress
                 else "started")
        raise chromium_runtime.ChromiumRefusal(chromium_runtime.PREPARING, (
            f"{what} is still being installed ({state}, {int(time.monotonic() - progress['started'])} s so far); "
            f"this call opened no browser. The install continues in the background and a later call uses it.{hint}"
        )) from None


def _run_install(key: str, entry: Tuple[concurrent.futures.Future, Dict[str, Any]],
                 install: Callable[[Dict[str, Any]], Any], release: Callable[[], None]) -> None:
    future, progress = entry
    outcome, failure = None, None
    try:
        outcome = install(progress)
    except BaseException as exc:  # noqa: BLE001 - every waiter receives the install's own failure
        failure = exc
        log.warning("Browser install %s failed: %s", key, exc)  # its callers may have answered already
    with _installs_lock:  # settled before any waiter wakes: its next call starts afresh or finds the result
        if _installs.get(key) is entry:
            del _installs[key]
    try:
        release()
    finally:
        future.set_exception(failure) if failure is not None else future.set_result(outcome)


@contextlib.contextmanager
def _machine_lock(path: pathlib.Path, progress: Dict[str, Any]):
    """One installer per browsers root on this machine (tasks run in separate worker processes);
    the OS frees the lock with its process. The lock file stays: one inode, one lock. Where no
    lock can be taken (no write access, a filesystem without locks) the install runs unlocked, as
    before: its atomic publication still keeps two installs from colliding."""
    from ouroboros.platform_layer import file_lock_exclusive, file_unlock

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+b")
    except OSError:
        yield
        return
    with handle:
        progress["waiting"] = True
        try:
            file_lock_exclusive(handle.fileno())
            locked = True
        except OSError:
            locked = False
        progress.pop("waiting", None)
        try:
            yield
        finally:
            if locked:
                file_unlock(handle.fileno())


def _pinned_chromium(*, allow_install: bool) -> str:
    """The pinned headless shell's executable: found, else installed and verified, else refused.

    The exact pin is looked up under an operator's ``PLAYWRIGHT_BROWSERS_PATH`` directory, the
    Playwright package's bundled ``.local-browsers`` and the data root's ``playwright-browsers``;
    a missing pin installs into the operator's directory, else that cache, never the package."""
    from ouroboros.config import resolve_data_dir

    pin = chromium_runtime.host_pin()
    chromium_runtime.require_supported_host(pin)
    configured = "" if _playwright_browsers_path_managed else os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    operator = pathlib.Path(configured) if configured not in ("", "0") else None
    cache = resolve_data_dir() / "playwright-browsers"
    roots = [root for root in (operator, chromium_runtime.browsers_root("0"), cache) if root is not None]
    found = chromium_runtime.find_installed(roots, pin)
    if found is not None:
        return str(found)
    if not allow_install:
        raise chromium_runtime.ChromiumRefusal(chromium_runtime.UNAVAILABLE, (
            f"the pinned Chromium {chromium_runtime.CHROMIUM_VERSION} is not installed, and this read-only "
            "actor may not install it; a Chromium browser call by its parent installs it."))
    target = operator or cache

    def install(progress: Dict[str, Any]) -> str:
        with _machine_lock(target / f".{pin.directory}.lock", progress):
            if (published := chromium_runtime.installed_executable(target, pin)) is not None:
                return str(published)  # a peer process installed it while this one waited
            _repair_host_dependencies("chromium")
            log.info("Installing pinned Chromium %s into %s", chromium_runtime.CHROMIUM_VERSION, target)
            progress["size"] = pin.archive_size
            return str(chromium_runtime.install(target, pin, progress=progress))

    return _install_beside_calls(f"chromium:{target}", f"the pinned Chromium {chromium_runtime.CHROMIUM_VERSION}",
                                 install, " engine='webkit' is available where it is installed.")


async def _launch_browser_with_fallback(pw_instance: Any, *, engine: str = "chromium",
                                        executable_path: Any = None, sandbox: Any = None) -> Any:
    """Launch the engine with no fallback: Chromium only as the pinned executable, inside its
    sandbox unless the administrator opted out, and only when that build answers."""
    engine = _normalize_browser_engine(engine)
    browser_type = getattr(pw_instance, engine)
    if engine != "chromium":
        return await browser_type.launch(headless=True)
    if not executable_path:
        raise chromium_runtime.ChromiumRefusal(chromium_runtime.UNAVAILABLE, (
            "no pinned Chromium executable was resolved; Playwright's own Chromium build is never "
            "launched for browser tools."))
    sandbox = chromium_runtime.sandbox_enabled() if sandbox is None else bool(sandbox)
    if not sandbox:
        log.warning("Chromium sandbox is OFF by administrator opt-out %s=off", chromium_runtime.SANDBOX_ENV)
    try:
        browser = await browser_type.launch(
            headless=True,
            executable_path=str(executable_path),
            chromium_sandbox=sandbox,  # without it Playwright itself adds --no-sandbox
            args=[
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--window-size=1920,1080",
                # J (v6.39): software-GL via ANGLE+SwiftShader so WebGL/canvas/3D actually
                # RENDER in headless (the headless shell has no GPU) instead of a black
                # frame; ignore the GPU blocklist so SwiftShader is used.
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
            ],
        )
    except Exception as exc:
        refusal = chromium_runtime.sandbox_refusal(exc) if sandbox else None
        if refusal is not None:
            raise refusal from exc
        raise
    version = str(getattr(browser, "version", "") or "")
    if version != chromium_runtime.CHROMIUM_VERSION:
        with contextlib.suppress(Exception):
            await browser.close()
        raise chromium_runtime.ChromiumRefusal(chromium_runtime.UNAVAILABLE, (
            f"{executable_path} answered as Chromium {version or 'of an unknown version'}, not the pinned "
            f"{chromium_runtime.CHROMIUM_VERSION}; it was closed."))
    return browser


def _device_context_options(pw_instance: Any, device: str = "") -> Tuple[Dict[str, Any], str]:
    device_name = str(device or "").strip()
    if not device_name:
        return {}, ""
    devices = getattr(pw_instance, "devices", {}) or {}
    resolved = device_name
    if resolved not in devices:
        matches = [name for name in devices if str(name).lower() == device_name.lower()]
        if matches:
            resolved = matches[0]
        else:
            samples = ", ".join(list(devices)[:6])
            raise ValueError(f"Unknown Playwright device descriptor {device_name!r}. Examples: {samples}")
    return dict(devices[resolved]), resolved


def _default_context_options(engine: str) -> Dict[str, Any]:
    options: Dict[str, Any] = {"viewport": {"width": 1920, "height": 1080}}
    if engine == "chromium":
        options["user_agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    return options


@_bound_browser_call
def _ensure_browser(ctx: ToolContext, *, engine: str = "chromium", device: str = ""):
    """Create or reuse this context's browser; no module-level Playwright state.

    Returns ``(page, generation)`` — the CALL works with the generation that
    owns its page for the rest of its life (screenshots, guards), so a
    mid-call timeout retirement replacing the shared slot can never rebind
    the abandoned call onto the NEXT command's state."""
    engine = _normalize_browser_engine(engine)
    requested_device = str(device or "").strip()
    call = _browser_call.get()
    bs = call.generation
    if ctx.browser_state is not bs:
        # The submit wrapper pinned the generation this call is bound to; a
        # mismatch here means the call's own timeout retired it before the
        # session ever opened — refuse without touching the NEXT command's
        # replacement (the check-then-run window of the wrapper is closed at
        # the exact point the generation is captured).
        raise RuntimeError(
            "⚠️ BROWSER_SESSION_RETIRED: this call was abandoned by its "
            "timeout before the session opened; nothing ran.")
    stored_engine = getattr(bs, "_browser_engine", "")
    stored_device = getattr(bs, "_browser_device", "")
    if bs.browser is not None and (
        stored_engine != engine
        or stored_device.lower() != requested_device.lower()
    ):
        log.info("Browser engine/device changed (%s/%s -> %s/%s); recreating context.",
                 stored_engine or "chromium", stored_device, engine, requested_device)
        bs = cleanup_browser(ctx, expected=bs)

    if bs is None:
        raise RuntimeError(_SESSION_RETIRED_VOID)
    call.generation = bs

    if bs.browser is not None:
        if _session_alive(bs, page=False):
            return bs.page, bs
        bs = cleanup_browser(ctx, expected=bs)
        if bs is None:
            raise RuntimeError(_SESSION_RETIRED_VOID)
        call.generation = bs

    # Preparation stays on the calling thread, outside the session's loop.
    readonly_subagent = _readonly_subagent(ctx)
    executable_path = _ensure_playwright_installed(engine=engine, allow_install=not readonly_subagent)
    sandbox = chromium_runtime.sandbox_enabled() if engine == "chromium" else None

    if bs.pw_instance is None:
        backlog = _live_retired_generations(ctx)
        if len(backlog) >= _RETIRED_GENERATIONS_MAX:
            raise RuntimeError(
                "⚠️ BROWSER_BACKLOG_RETIRED_SESSIONS: "
                f"{len(backlog)} earlier browser sessions of this task are not "
                "confirmed closed (a call hung past its timeout, or a close hung "
                "or failed); refusing to open another. Stop retrying browser "
                "tools in this task, or finish without them — a hung call's "
                "session is closed when the call settles or the task ends."
            )
    owner = browser_session.owner_of(bs)
    try:
        # From an empty context: the driver's reader and every callback of this session start there.
        owner.run(_open_session(ctx, bs, owner, engine, requested_device, executable_path, sandbox), clean=True)
    except BaseException:
        # A failed setup (launch, version, sandbox, page) retires and closes its generation as every
        # close does: a confirmed one leaves the backlog, an unconfirmed one stays counted in it. The
        # call continues on the fresh, empty generation, so its caller decides as before.
        fresh = cleanup_browser(ctx, expected=bs)
        if fresh is None:  # its own timeout retired it meanwhile: closed like every retired one
            cleanup_browser_handles(bs)
        else:
            call.generation = fresh
        raise
    return bs.page, bs


async def _open_session(ctx: ToolContext, bs: Any, owner: Any, engine: str, requested_device: str,
                        executable_path: Any, sandbox: Any) -> None:
    """Start Playwright and the browser, then the one page, its observer and the request guard."""
    if bs.pw_instance is None:
        await _start_driver(bs)
    bs.browser = await _launch_browser_with_fallback(bs.pw_instance, engine=engine,
                                                     executable_path=executable_path, sandbox=sandbox)
    device_options, resolved_device = _device_context_options(bs.pw_instance, requested_device)
    context_options = device_options or _default_context_options(engine)
    bs_context = await bs.browser.new_context(**context_options)
    setattr(bs, "_browser_context", bs_context)
    setattr(bs, "_browser_engine", engine)
    setattr(bs, "_browser_device", resolved_device or requested_device)
    bs.page = await bs_context.new_page()

    if _HAS_STEALTH:
        # The iframe.contentWindow evasion intercepts every page's iframe.srcdoc
        # setter and never calls the native one, so srcdoc frames (module widgets)
        # render empty (#1606). Every other evasion stays.
        stealth = Stealth(iframe_content_window=False)
        await stealth.apply_stealth_async(bs.page)

    bs.page.set_default_timeout(30000)
    page = bs.page
    setattr(bs, "navigations", [])

    def observe_navigation(response: Any) -> None:
        # Native redirect hops never reach the route callback, so a navigation
        # response that ends a redirect chain is kept for the chain check
        # (_navigation_block_reason) that runs before any page result. A
        # main-frame navigation starts a new document and drops the old rows;
        # an unredirected navigation was already judged by the route callback.
        try:
            request = response.request
            if not request.is_navigation_request():
                return
            rows = [] if response.frame == page.main_frame else list(getattr(bs, "navigations", []))
            if request.redirected_from is not None:
                rows.append(response)
            setattr(bs, "navigations", rows)
        except Exception:
            log.debug("Browser navigation observation failed", exc_info=True)

    page.on("response", observe_navigation)
    # Requests exposed by Playwright share the target decision, between calls too.
    # Native HTTP redirect hops can bypass route callbacks and are checked after
    # navigation; that check withholds content, not the already-sent request.
    # Existing owner POST shapes apply at the actual Ouroboros origin; a dev app
    # reusing their pathname is not Ouroboros.
    await bs_context.route("**/*", browser_session.request_guard(owner, ctx, bs))
    # Facts of THIS generation, set only once it is fully set up: a reused
    # generation reports its own, a retired one never borrows a newer one's.
    version = str(getattr(bs.browser, "version", "") or "unknown version")
    supply = "Playwright" if sandbox is None else f"Playwright; {chromium_runtime.conditions(sandbox)}"
    setattr(bs, "browser_conditions", f"Browser conditions: {engine} {version} ({supply}); " + (
        "playwright-stealth evasions applied except iframe.contentWindow." if _HAS_STEALTH
        else "playwright-stealth is not installed, so no stealth evasions were applied."))


async def _start_driver(bs: Any) -> None:
    """Own the transport before starting it; a close during the handshake takes the same driver. The driver's
    stderr joins the process buffer for this task's spawn alone (``driver_diagnostics``); stdout stays protocol."""
    from ouroboros.browser_output import driver_diagnostics

    with driver_diagnostics(task=True):
        driver = browser_session.PlaywrightDriver()
        with _retired_generations_lock:
            setattr(bs, "_pw_driver", driver)
        playwright = await driver.start()
    with _retired_generations_lock:
        if getattr(bs, "_pw_driver", None) is driver:
            bs.pw_instance = playwright
            return
    raise browser_session.BrowserSessionClosed(browser_session.CLOSED)


def _take_handles(bs: Any) -> tuple:
    """Read and clear a generation's Playwright handles at once, a driver start in flight
    included: a second reader finds none."""
    with _retired_generations_lock:
        handles = (getattr(bs, "page", None), getattr(bs, "_browser_context", None),
                   getattr(bs, "browser", None), getattr(bs, "pw_instance", None), getattr(bs, "_pw_driver", None))
        bs.page = None
        setattr(bs, "_browser_context", None)
        bs.browser = None
        bs.pw_instance = None
        setattr(bs, "_pw_driver", None)
        setattr(bs, "_browser_engine", "")
        setattr(bs, "_browser_device", "")
    return handles


async def _alive(browser: Any, page: Any) -> bool:
    return bool(browser.is_connected()) and (page is None or not page.is_closed())


def _session_alive(bs: Any, *, page: bool = True) -> bool:
    """Whether the generation's browser (and, by default, its page) still answers, read on its loop."""
    try:
        return browser_session.owner_of(bs).run(_alive(bs.browser, bs.page if page else None))
    except Exception:
        log.debug("Browser connection check failed", exc_info=True)
        return False


def _observed(ctx: ToolContext, bs: Any, text: str) -> str:
    """One successful observation of the call's own generation, with its host notes.

    The generation's browser conditions (and every refused request not shown yet,
    between calls included) reach the model for screenshots and DOM/evaluate reads
    alike; the extracted data stays unannotated in ``producer_text`` for programmatic readers."""
    notes = tuple(note for note in (browser_session.take_block_notes(bs),
                                    str(getattr(bs, "browser_conditions", "") or "")) if note)
    if not notes:
        return text
    composed = text
    for note in notes:
        composed = _compose_execute_result(composed, note, "")
    return _publish_tool_result(ctx, dataclasses.replace(
        LegacyTextResultAdapter.from_text("", text), text=composed, producer_text=text, host_annotations=notes))


# The in-process bound on retired browser generations not confirmed closed (a
# hung call's queued cleanup has not run, or a close hung or failed).
# Structural cap, not an owner knob: hitting it means repeated hung calls or
# failed closes in ONE task, and the honest move is a typed refusal of the next
# session rather than unbounded Chromium accumulation (the true fix — a
# process-isolated browser worker — is a disclosed future design).
_RETIRED_GENERATIONS_MAX = 3

# One process-wide lock for every read-modify-write of a context's
# ``_retired_browser_generations`` list and for every swap of its
# ``browser_state`` slot: the timeout path (main thread) and the worker race
# otherwise, and a lost update silently drops an unclosed generation from the
# backlog bound or replaces the next command's state.
_retired_generations_lock = threading.Lock()

# The void result of a call whose generation its own timeout retired.
_SESSION_RETIRED_VOID = ("⚠️ BROWSER_SESSION_RETIRED: this call timed out and its "
                         "browser session was retired; the result is void.")


def _live_retired_generations(ctx: ToolContext) -> list:
    """Retired-but-unclosed generations of this context (closed ones pruned)."""
    with _retired_generations_lock:
        rows = [g for g in getattr(ctx, "_retired_browser_generations", [])
                if not getattr(g, "_cleanup_done", False)]
        setattr(ctx, "_retired_browser_generations", rows)
        return list(rows)


def _detach_browser(ctx: ToolContext, expected: Any = None) -> tuple[Any, Any]:
    """Retire the context's browser GENERATION without touching Playwright.

    The shared ``ctx.browser_state`` is REPLACED with a fresh object; returns
    ``(retired, fresh)``. An abandoned worker that still holds a local
    reference keeps writing into ITS retired object (even handles it creates
    after this call land there), so closing the retired object later reaps
    everything that worker ever owned — while the next command gets a
    brand-new state no stale thread can reach. Callers that continue working
    (the _ensure_browser internal recreations) use FRESH directly and never
    re-read the shared slot, so an external timeout retirement landing in
    between cannot rebind them onto the next command's state.

    With ``expected`` the swap is a compare-and-swap: it happens only while the
    slot still holds that generation, else ``(None, None)`` and nothing changes.
    Every swap of the slot is this one, under one lock. A retired page starts no
    new request between calls; its hung call's own requests are still decided."""
    from ouroboros.tools.registry import BrowserState  # lazy: avoids the import cycle

    fresh = BrowserState()
    with _retired_generations_lock:
        retired = ctx.browser_state
        if expected is not None and retired is not expected:
            return None, None
        ctx.browser_state = fresh
        setattr(retired, "_retired", True)
        rows = list(getattr(ctx, "_retired_browser_generations", []))
        rows.append(retired)
        setattr(ctx, "_retired_browser_generations", rows)
    return retired, fresh


def cleanup_browser_handles(retired: Any) -> None:
    """Close a retired generation's Playwright, from any thread but its own loop's.

    Its owner elects one close: the winner reads and clears the handles on the loop
    itself (those an abandoned call opened after detachment included) and closes
    them there before the loop stops; a concurrent or later call closes nothing of
    its own and reports that outcome, and the closed owner starts nothing again.
    The timeout path queues this behind the hung call on the retiring worker; the
    task's end may close the same generation meanwhile, and one of them closes it."""
    if retired is None:
        return
    owner = browser_session.owner_of(retired)
    confirmed = owner.close(functools.partial(_close_generation, retired))
    if not owner.started:  # no loop ever held anything: closed only if nothing is open
        confirmed = confirmed and not any(_take_handles(retired))
    # Settlement truth for the backlog bound: stamped only when the close is confirmed —
    # a driver stop that hung or failed, or a loop that did not stop, keeps it counted.
    if confirmed:
        setattr(retired, "_cleanup_done", True)


async def _close_generation(bs: Any) -> bool:
    return await browser_session.close_handles(*_take_handles(bs))


def cleanup_browser(ctx: ToolContext, expected: Any = None) -> Any:
    """Retire and close the context's CURRENT browser generation; return the FRESH
    one so a continuing caller never re-reads the shared slot (with ``expected``,
    None when the slot no longer holds it). Without ``expected`` the task or turn
    is done with its browser: the generations its timed-out calls retired close
    too, each on its own loop, whether or not those calls settled."""
    retired, fresh = _detach_browser(ctx, expected)
    if fresh is not None:
        cleanup_browser_handles(retired)
        if expected is None:
            for generation in _live_retired_generations(ctx):
                cleanup_browser_handles(generation)
    return fresh


def _is_infrastructure_error(obj: Any) -> bool:
    """Detect generation-state, context-state or legacy string-based failures."""
    if hasattr(obj, "browser_state") or hasattr(obj, "pw_instance"):
        bs = getattr(obj, "browser_state", obj)
        if bs.browser is None or bs.pw_instance is None:
            return True
        return not _session_alive(bs)

    msg = str(obj).lower()
    return any(token in msg for token in (
        "browser has been closed",
        "page has been closed",
        "connection closed",
    ))


_MARKDOWN_JS = """() => {
    const walk = (el) => {
        let out = '';
        for (const child of el.childNodes) {
            if (child.nodeType === 3) {
                const t = child.textContent.trim();
                if (t) out += t + ' ';
            } else if (child.nodeType === 1) {
                const tag = child.tagName;
                if (['SCRIPT','STYLE','NOSCRIPT'].includes(tag)) continue;
                if (['H1','H2','H3','H4','H5','H6'].includes(tag))
                    out += '\\n' + '#'.repeat(parseInt(tag[1])) + ' ';
                if (tag === 'P' || tag === 'DIV' || tag === 'BR') out += '\\n';
                if (tag === 'LI') out += '\\n- ';
                if (tag === 'A') out += '[';
                out += walk(child);
                if (tag === 'A') out += '](' + (child.href||'') + ')';
            }
        }
        return out;
    };
    return walk(document.body);
}"""


def _inject_native_screenshot(ctx: ToolContext, b64: str) -> str:
    """Add a fresh screenshot to the conversation as a native image block.

    The screenshot is saved to ``data/uploads/screenshots/<ts>.png`` (the
    re-view path used by eviction placeholders) and merged into the canonical
    user turn via the existing multipart-preserving merge, whatever the route:
    the send-time image policy (``vision_routing``) decides whether this route
    receives its pixels, a caption or a marker. The TOOL result stays a plain
    string — the tool-message contract is unchanged.
    """
    try:
        messages = getattr(ctx, "messages", None)
        if not isinstance(messages, list):
            return ""
        from ouroboros.utils import utc_now_iso

        ts = utc_now_iso().replace(":", "").replace("-", "")[:15]
        shot_dir = pathlib.Path(ctx.drive_root) / "uploads" / "screenshots"
        shot_dir.mkdir(parents=True, exist_ok=True)
        shot_path = shot_dir / f"{ts}.png"
        shot_path.write_bytes(base64.b64decode(b64))
        from ouroboros.maintenance_retirement import artifact_changed
        artifact_changed(ctx.drive_root)
        caption = f"[browser screenshot {ts}]"
        from ouroboros.loop import _append_or_merge_user_content

        _append_or_merge_user_content(messages, [
            {"type": "text", "text": caption},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
                "_caption": caption,
                "_source_path": str(shot_path),
            },
        ], stamp=ctx)  # a batch eviction of sent images names its origin
        return ("The screenshot is in your context as an image; the image-input mode and this route decide "
                "whether you receive its pixels, a caption or a marker. ")
    except Exception:
        log.debug("native screenshot injection failed", exc_info=True)
        return ""


async def _page_health_snapshot(page: Any) -> str:
    """Cheap textual page diagnostics (native Playwright, zero LLM cost) so a
    screenshot caller can tell whether the page actually loaded/rendered even when
    no vision model is available. Each probe is independently defensive."""
    parts: list = []
    try:
        parts.append(f"url={page.url}")
    except Exception:
        pass
    try:
        parts.append("title=" + repr((await page.title() or "")[:120]))
    except Exception:
        pass
    try:
        counts = await _evaluate_bounded(
            page,
            "() => ({canvas: document.querySelectorAll('canvas').length,"
            " img: document.querySelectorAll('img').length,"
            " scripts: document.querySelectorAll('script').length,"
            " bodyChars: (document.body && document.body.innerText || '').length})",
        )
        if isinstance(counts, dict):
            parts.append("elements=" + ",".join(f"{k}:{v}" for k, v in counts.items()))
    except Exception:
        pass
    try:
        body = (await page.inner_text("body") or "").strip().replace("\n", " ")
        if body:
            parts.append("bodyText=" + repr(body[:200]))
    except Exception:
        pass
    return " | ".join(parts)


async def _wait_for_page_paint(page: Any, timeout_ms: int = 3000) -> None:
    """J (v6.39): let the page paint before a screenshot — wait for document.readyState then
    two requestAnimationFrames (the second fires AFTER the paint), so a freshly-rendered
    canvas/WebGL frame is not captured black/blank. Best-effort + bounded; never blocks."""
    try:
        await page.wait_for_function("document.readyState === 'complete'", timeout=min(int(timeout_ms or 3000), 3000))
    except Exception:
        pass
    try:
        # Schedule a paint flag via double-rAF (the second fires AFTER the paint), then wait
        # for it with a HARD Playwright timeout. The flag-set evaluate returns immediately
        # (it does not await a page-owned promise), and wait_for_function's own timeout
        # bounds the wait — so a page that suppresses requestAnimationFrame can never hang
        # the capture (the page's own timers are never trusted to unblock us).
        await page.evaluate(
            "() => { window.__obo_painted = false;"
            " requestAnimationFrame(() => requestAnimationFrame(() => { window.__obo_painted = true; })); }"
        )
        await page.wait_for_function("window.__obo_painted === true", timeout=500)
    except Exception:
        pass


def _evaluate_source(expression: str) -> str:
    """Select function-body syntax before execution, never by a runtime error.

    The existing JavaScript grammar distinguishes a top-level return from one
    inside a function. If grammar support is unavailable, retain the literal
    expression; a failed evaluation is never replayed to try another form.
    """
    try:
        from tree_sitter_language_pack import get_parser

        tree = get_parser("javascript").parse(expression.encode("utf-8"))
        pending = [tree.root_node]
        functions = {"function_declaration", "function_expression", "arrow_function",
                     "generator_function", "generator_function_declaration", "method_definition"}
        while pending:
            node = pending.pop()
            if node.type in functions:
                continue
            if node.type == "return_statement" and not tree.root_node.has_error:
                return "(() => {\n" + expression + "\n})()"
            pending.extend(node.named_children)
    except Exception:
        pass
    return expression


async def _evaluate_bounded(page: Any, expression: str, timeout_ms: int = 30000) -> Any:
    """``page.evaluate`` with an in-page deadline.

    Playwright's ``evaluate`` accepts no timeout kwarg and ignores
    ``set_default_timeout``, so an awaited never-resolving promise used to hang
    until the outer tool timeout. Racing the expression against an in-page
    ``setTimeout`` rejection bounds those ASYNC hangs; it CANNOT interrupt a
    synchronous event-loop block (``while(true){}``) — the outer tool timeout
    remains the backstop for that.

    The expression is evaluated with the EXACT semantics of Playwright's
    raw-string evaluate: the driver's ``UtilityScript`` runs
    ``global.eval(expression)`` and — because the Python client leaves
    ``isFunction`` unset for a string — INVOKES a function-valued result
    (that is how ``page.evaluate("() => {...}")`` and ``_MARKDOWN_JS`` work).
    So statement lists, a trailing ``;``, completion values AND
    function-expression invocation all behave identically to an un-bounded
    evaluate. A parenthesised ``const __x = (expr)`` wrapper broke the former
    (SyntaxError → silent ``undefined``); a bare ``eval`` broke the latter
    (function returned uninvoked → serialized ``undefined``).
    """
    ms = max(int(timeout_ms or 0), 1)
    src = json.dumps(str(expression))
    # INDIRECT eval ("(0, eval)") like the driver's global.eval: the snippet
    # runs in GLOBAL scope, so `var`/function declarations persist across
    # evaluate calls (cross-call parity) and the wrapper's own lexical
    # `__obo_result` binding is invisible to user code (a direct eval exposed
    # it as a TDZ ReferenceError).
    wrapped = (
        "() => Promise.race(["
        "Promise.resolve().then(() => { const __obo_result = (0, eval)(" + src + "); "
        "return typeof __obo_result === 'function' ? __obo_result() : __obo_result; }), "
        "new Promise((_, reject) => setTimeout(() => reject(new Error("
        "'evaluate timed out after " + str(ms) + "ms')), " + str(ms) + "))])"
    )
    return await page.evaluate(wrapped)


async def _page_output(page: Any, output: str) -> Any:
    """The page read for ``output``: text, HTML, markdown, or the screenshot's bytes with page health."""
    if output == "screenshot":
        await _wait_for_page_paint(page)  # J: paint before capture (shared with browser_action)
        return await page.screenshot(type="png", full_page=False), await _page_health_snapshot(page)
    # The whole page text goes on: the exact-source delivery keeps it readable and bounds only the view.
    if output == "html":
        return await page.content()
    if output == "markdown":
        return await _evaluate_bounded(page, _MARKDOWN_JS)
    return await page.inner_text("body")


def _compose_output(ctx: ToolContext, bs: Any, output: str, payload: Any) -> str:
    """The page result composed on the calling thread. A screenshot lands in the CALL's
    own generation ``bs``, never in a replacement the shared slot may already hold
    after a timeout retirement."""
    if output != "screenshot":
        return payload
    data, health = payload
    b64 = base64.b64encode(data).decode()
    bs.last_screenshot_b64 = b64
    health_note = (f"Page health: {health}. " if health else "")
    if _readonly_subagent(ctx):
        return (
            f"Screenshot captured ({len(b64)} bytes base64). "
            + health_note
            + "Use analyze_screenshot to inspect it."
        )
    native_note = _inject_native_screenshot(ctx, b64)
    return (
        f"Screenshot captured ({len(b64)} bytes base64). "
        + health_note
        + (native_note or "Use analyze_screenshot to inspect it. ")
        + "Call send_photo(image_base64='__last_screenshot__') to deliver it to the user."
    )


_WAIT_STATES = frozenset({"attached", "visible", "hidden", "detached"})
_WAIT_ARGUMENTS = "Error: wait requires a selector and state attached/visible/hidden/detached."


async def _wait_for_page_state(page: Any, selector: str, state: str, timeout: int) -> tuple[bool, str]:
    """Observe a selector on this page, without navigation or model polling (callers check arguments)."""
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    reached = True
    try:
        await page.wait_for_selector(selector, state=state, timeout=max(int(timeout or 30000), 1))
    except PlaywrightTimeoutError:
        reached = False
    locator = page.locator(selector)
    count = await locator.count()
    return reached, json.dumps({
        "status": "reached" if reached else "timeout", "selector": selector,
        "requested_state": state, "url": page.url, "matched_elements": count,
        "first_visible": await locator.first.is_visible() if count else False,
    }, ensure_ascii=False)


async def _navigation_block_reason(page: Any, bs: Any, ctx: ToolContext, restricted: bool,
                                   response: Any = None) -> str:
    """Apply the target decision to the current URL and every observed redirect hop.

    ``response`` is the navigation result of the call's own ``goto``; ``bs`` carries
    every navigation response observed for the current document (subframes
    included). Each ``request.redirected_from`` chain is walked with the same
    predicate as direct navigation, so an allowed→blocked→allowed redirect
    withholds the content it produced. The forbidden hop's request was already
    dispatched natively by then — a disclosed residual, not a pre-request guard.
    The hops are read on the loop; the decision (DNS, service identity) runs off it.
    """
    targets = [str(getattr(page, "url", "") or "")]
    for root in (response, *getattr(bs, "navigations", [])):
        request = getattr(root, "request", None) if root is not None else None
        while request is not None:
            targets.append(str(request.url))
            request = request.redirected_from
    return await browser_session.owner_of(bs).decide(_first_url_block, list(dict.fromkeys(targets)), ctx, restricted)


def _first_url_block(targets: List[str], ctx: ToolContext, restricted: bool) -> str:
    for target in targets:
        if reason := browser_policy.browser_url_block_reason(
            target, ctx, restricted=restricted, runtime_mode=_runtime_mode_for_browser(ctx)):
            return reason
    return ""


@_bound_browser_call
def _browse_page(ctx: ToolContext, url: str, output: str = "text",
                 wait_for: str = "", timeout: int = 30000,
                 viewport: str = "", engine: str = "chromium", device: str = "", state: str = "visible") -> str:
    readonly_subagent = _readonly_subagent(ctx)
    if reason := browser_policy.browser_url_block_reason(
        str(url or ""), ctx, restricted=readonly_subagent,
        runtime_mode=_runtime_mode_for_browser(ctx)):
        return "⚠️ " + reason
    call = _browser_call.get()
    entry, mark = call.generation, browser_session.block_mark(call.generation)
    load = functools.partial(_load_page, ctx, _observer_url(url, ctx), output, wait_for, timeout, viewport, engine,
                             device, state, readonly_subagent)
    try:
        return load()
    except Exception as e:
        own = call.generation
        if ctx.browser_state is not own:
            # Only this invocation's legitimate replacements advance its
            # expectation; a later call opening the shared slot never does.
            cleanup_browser_handles(own)
            return _SESSION_RETIRED_VOID
        if isinstance(e, chromium_runtime.ChromiumRefusal):
            return _chromium_refused(ctx, own, e)
        if reason := browser_session.take_block_since(own, mark if own is entry else 0):
            return "⚠️ " + reason
        if _is_infrastructure_error(own):
            log.warning("Browser infrastructure error: %s. Cleaning up and retrying...", e)
            # The ONE deliberate retry replaces the call's OWN generation and
            # moves the pin to the replacement it made, as _ensure_browser's
            # own replacements do. A timeout retirement that took the slot
            # during the checks above wins the swap: the call is abandoned.
            fresh = cleanup_browser(ctx, expected=own)
            if fresh is None:
                cleanup_browser_handles(own)
                return _SESSION_RETIRED_VOID
            call.generation = fresh
            return load()
        raise


def _observer_url(url: Any, ctx: ToolContext) -> Any:
    """``url`` opened as an observer view when it is this installation's own UI page (DESIGN "Unread"):
    what the builtin reads there is not the owner's read. Only the UI document (``/``) under a ``main``
    proof from ``browser_policy.runtime_service_kind`` gets ``observer=1``, its spelling kept (an
    ``observer=0`` it was given leaves no exit); an API or file path, any other target and an unknown or
    unreadable identity open as given."""
    text = str(url)
    try:
        if urlsplit(text).path not in ("", "/") or browser_policy.runtime_service_kind(text, ctx) != "main":
            return url
    except Exception:
        log.debug("Own-UI identity could not be read; the page opens as given", exc_info=True)
        return url
    head, mark, fragment = text.partition("#")
    path, _, query = head.partition("?")
    if "observer=1" in query.split("&"):
        return url
    query = f"{query}&observer=1" if query and not query.endswith("&") else f"{query}observer=1"
    return f"{path}?{query}{mark}{fragment}"


def _load_page(ctx: ToolContext, url: str, output: str, wait_for: str, timeout: int, viewport: str,
               engine: str, device: str, state: str, readonly: bool) -> str:
    """One navigation of the call's own generation: its Playwright work is one command there."""
    page, generation = _ensure_browser(ctx, engine=engine, device=device)
    kind, payload = browser_session.owner_of(generation).run(
        _navigate(ctx, page, generation, url, output, wait_for, timeout, viewport, state, readonly))
    if kind == "blocked":
        return "⚠️ " + payload
    if kind == "arguments":
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=payload))
    if kind == "waited":
        return payload
    return _observed(ctx, generation, _compose_output(ctx, generation, output, payload))


async def _navigate(ctx: ToolContext, page: Any, generation: Any, url: str, output: str, wait_for: str,
                    timeout: int, viewport: str, state: str, readonly: bool) -> tuple[str, Any]:
    # The caller's page-load timeout is also the session default so the
    # extraction calls that honor the default (screenshot, inner_text,
    # content) share the same bound instead of a stale prior value.
    page.set_default_timeout(int(timeout) if timeout else 30000)
    if viewport:
        await _apply_viewport(page, viewport)
    response = await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
    waited = (True, "")
    if wait_for:
        waited = await _wait_for_page_state(page, wait_for, state, timeout) if state in _WAIT_STATES else None
    if reason := await _navigation_block_reason(page, generation, ctx, readonly, response):
        return "blocked", reason
    if waited is None:
        return "arguments", _WAIT_ARGUMENTS
    if not waited[0]:
        return "waited", waited[1]
    return "page", await _page_output(page, output)


def _chromium_refused(ctx: ToolContext, own: Any, refusal: "chromium_runtime.ChromiumRefusal") -> str:
    """A refused Chromium launch started no browser: no retry and no unknown outcome.

    The half-open generation (a Playwright driver without a browser) is retired; the
    typed text names what to change."""
    if cleanup_browser(ctx, expected=own) is None:
        cleanup_browser_handles(own)
    log.warning("Chromium launch refused: %s", refusal.tool_text())
    return refusal.tool_text()


async def _apply_viewport(page: Any, viewport: str) -> None:
    """Parse a 'WxH' string and resize the browser viewport."""
    try:
        parts = viewport.lower().split("x")
        w, h = int(parts[0]), int(parts[1])
        await page.set_viewport_size({"width": max(320, w), "height": max(480, h)})
    except (ValueError, IndexError):
        log.warning("Invalid viewport '%s', expected WxH (e.g. '375x812')", viewport)


@_bound_browser_call
def _browser_action(ctx: ToolContext, action: str, selector: str = "",
                    value: str = "", timeout: int = 5000,
                    engine: str = "", device: str = "", state: str = "visible") -> str:
    normalized_action = str(action or "").strip().lower()
    readonly_subagent = _readonly_subagent(ctx)
    # The JavaScript rule's owner (and its rationale) is browser_policy, shared with MCP task sessions.
    if normalized_action == "evaluate" and (script_block := browser_policy.browser_script_block_reason(ctx)):
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="ACCESS_BLOCKED", text=f"⚠️ {script_block}"))

    call = _browser_call.get()
    entry, mark = call.generation, browser_session.block_mark(call.generation)
    script = _evaluate_source(value) if normalized_action == "evaluate" and value else ""
    try:
        page, generation = _ensure_browser(
            ctx,
            engine=engine or getattr(call.generation, "_browser_engine", "chromium") or "chromium",
            device=device or getattr(call.generation, "_browser_device", "") or "",
        )
        # The caller timeout (default 5000) keeps its explicit click/fill/select
        # wait semantics, but as the SESSION default it is floored at 30000 so
        # the 5s action default cannot strangle the default-honoring capture
        # calls (screenshot); an explicitly larger timeout widens them.
        default_ms = max(int(timeout or 0), 30000)
        kind, payload = browser_session.owner_of(generation).run(_act(
            ctx, page, generation, normalized_action, str(action), selector, value, script, timeout, state,
            default_ms, readonly_subagent))
        return _action_result(ctx, generation, kind, payload, readonly_subagent)
    except Exception as e:
        own = call.generation
        if ctx.browser_state is not own:
            cleanup_browser_handles(own)
            return _SESSION_RETIRED_VOID
        if isinstance(e, chromium_runtime.ChromiumRefusal):
            return _chromium_refused(ctx, own, e)
        if reason := browser_session.take_block_since(own, mark if own is entry else 0):
            return "⚠️ " + reason
        if _is_infrastructure_error(own):
            log.warning("Browser action connection failed; preserving unknown outcome: %s", e)
            if cleanup_browser(ctx, expected=own) is None:
                cleanup_browser_handles(own)
                return _SESSION_RETIRED_VOID
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR", text=(
                "⚠️ BROWSER_ACTION_OUTCOME_UNKNOWN: the browser connection failed. "
                "The action may already have taken effect and was not repeated. "
                f"Inspect the target before deciding what to do next. Detail: {e}")))
        raise


_SCROLLS = {"down": "window.scrollBy(0, 600)", "up": "window.scrollBy(0, -600)", "top": "window.scrollTo(0, 0)",
            "bottom": "window.scrollTo(0, document.body.scrollHeight)"}


async def _act(ctx: ToolContext, page: Any, generation: Any, action: str, raw_action: str, selector: str,
               value: str, script: str, timeout: int, state: str, default_ms: int, readonly: bool) -> tuple[str, Any]:
    page.set_default_timeout(default_ms)
    if reason := await _navigation_block_reason(page, generation, ctx, readonly):
        return "blocked", reason
    if action == "wait":
        if not selector or state not in _WAIT_STATES:
            return "arguments", _WAIT_ARGUMENTS
        return "receipt", (await _wait_for_page_state(page, selector, state, timeout))[1]
    if action in ("click", "fill", "select") and not selector:
        return "arguments", f"Error: selector required for {action}"
    if action == "click":
        await page.click(selector, timeout=timeout)
        await page.wait_for_timeout(500)
        return "receipt", f"Clicked: {selector}"
    if action == "fill":
        await page.fill(selector, value, timeout=timeout)
        return "receipt", f"Filled {selector} with: {value}"
    if action == "select":
        await page.select_option(selector, value, timeout=timeout)
        return "receipt", f"Selected {value} in {selector}"
    if action == "screenshot":
        await _wait_for_page_paint(page, int(timeout or 3000))  # J: paint before capture
        return "screenshot", await page.screenshot(type="png", full_page=False)
    if action == "evaluate":
        if not value:
            return "arguments", "Error: value (JS code) required for evaluate"
        return "observation", str(await _evaluate_bounded(page, script, default_ms))
    if action == "scroll":
        direction = value or "down"
        if direction in _SCROLLS:
            await _evaluate_bounded(page, _SCROLLS[direction], default_ms)
        return "receipt", f"Scrolled {direction}"
    return "arguments", f"Unknown action: {raw_action}. Use: click, fill, select, screenshot, evaluate, scroll, wait"


def _action_result(ctx: ToolContext, generation: Any, kind: str, payload: Any, readonly: bool) -> str:
    """An action's outcome on the calling thread: observations carry the generation's notes and
    conditions; receipts, refusals and argument errors carry its refusal notes."""
    if kind == "screenshot":
        b64 = base64.b64encode(payload).decode()
        # Land the capture in the CALL's own generation, never in a
        # replacement the shared slot may hold after a timeout retirement.
        generation.last_screenshot_b64 = b64
        return _observed(ctx, generation, f"Screenshot captured ({len(b64)} bytes base64). " + (
            "Use analyze_screenshot to inspect it." if readonly
            else "Call send_photo(image_base64='__last_screenshot__') to deliver it to the user."))
    if kind == "observation":
        return _observed(ctx, generation, payload)
    text = ("⚠️ " + payload if kind == "blocked" else payload if kind == "receipt"
            else _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=payload)))
    note = browser_session.take_block_notes(generation)
    return append_published_text(ctx, text, f"\n\n{note}" if note else "")


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="browse_page",
            schema={
                "name": "browse_page",
                "description": (
                    "Open a URL in this task's own headless Playwright browser (not the owner's signed-in Chrome) "
                    "and return its content as text, html, markdown, or screenshot (base64 PNG). "
                    "Browser persists across calls within a task. "
                    "For screenshots: use send_photo tool to deliver the image to the user. "
                    "Use viewport to test mobile layouts (e.g. '375x812'). "
                    "Use engine='webkit' plus a Playwright iPhone device descriptor "
                    "for iOS Safari-grade mobile verification."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to open"},
                        "output": {
                            "type": "string",
                            "enum": ["text", "html", "markdown", "screenshot"],
                            "description": "Output format (default: text)",
                        },
                        "wait_for": {
                            "type": "string",
                            "description": "CSS selector to wait for before extraction",
                        },
                        "state": {"type": "string", "enum": ["attached", "visible", "hidden", "detached"],
                                  "description": "State for wait_for (default visible). Timeout returns observed selector facts."},
                        "timeout": {
                            "type": "integer",
                            "description": "Page load timeout in ms (default: 30000)",
                        },
                        "viewport": {
                            "type": "string",
                            "description": "Viewport size as WxH (e.g. '375x812' for mobile, '1920x1080' for desktop). Default: current viewport.",
                        },
                        "engine": {
                            "type": "string",
                            "enum": ["chromium", "webkit"],
                            "description": "Browser engine. Default: chromium. Use webkit for iOS Safari-style checks.",
                        },
                        "device": {
                            "type": "string",
                            "description": "Optional Playwright device descriptor, e.g. 'iPhone 15 Pro' or 'iPhone 13'.",
                        },
                    },
                    "required": ["url"],
                },
            },
            handler=_browse_page,
            timeout_sec=180,
        ),
        ToolEntry(
            name="browser_action",
            schema={
                "name": "browser_action",
                "description": (
                    "Act on the page open in this task's headless browser (from browse_page). Actions: "
                    "click (selector), fill (selector + value), select (selector + value), "
                    "screenshot (base64 PNG), evaluate (JS code in value), "
                    "scroll (value: up/down/top/bottom), wait (selector + state, without navigation)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["click", "fill", "select", "screenshot", "evaluate", "scroll", "wait"],
                            "description": "Action to perform",
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for click/fill/select/wait",
                        },
                        "state": {"type": "string", "enum": ["attached", "visible", "hidden", "detached"],
                                  "description": "Wait state (default visible). Timeout reports current match/visibility; hidden also accepts absence."},
                        "value": {
                            "type": "string",
                            "description": "Value for fill/select, JS for evaluate, direction for scroll",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Action timeout in ms (default: 5000)",
                        },
                        "engine": {
                            "type": "string",
                            "enum": ["chromium", "webkit"],
                            "description": "Optional engine for a new browser session. Existing pages keep their current engine.",
                        },
                        "device": {
                            "type": "string",
                            "description": "Optional Playwright device descriptor for a new browser session.",
                        },
                    },
                    "required": ["action"],
                },
            },
            handler=_browser_action,
            timeout_sec=180,
        ),
    ]
