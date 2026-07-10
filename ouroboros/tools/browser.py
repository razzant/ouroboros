"""Playwright browser tools with per-ToolContext lifecycle/thread affinity."""

from __future__ import annotations

import base64
import ipaddress
import logging
import os
import pathlib
import re
import socket
import subprocess
import sys
import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

try:
    from playwright_stealth import Stealth
    _HAS_STEALTH = True
except ImportError:
    _HAS_STEALTH = False

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.server_auth import is_loopback_host
from ouroboros.config import AGENT_SERVER_PORT

log = logging.getLogger(__name__)

_playwright_ready = False
_playwright_ready_engines: set[tuple[str, str]] = set()
_playwright_browsers_path_managed = False
_MISSING_EXECUTABLE_RE = re.compile(r"Executable doesn't exist at ([^\n]+)")
_NONSTANDARD_NUMERIC_IPV4_RE = re.compile(r"^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}$", re.I)
_SUPPORTED_BROWSER_ENGINES = frozenset({"chromium", "webkit"})


def _normalize_browser_engine(engine: str = "") -> str:
    value = str(engine or "chromium").strip().lower()
    if value not in _SUPPORTED_BROWSER_ENGINES:
        raise ValueError("browser engine must be 'chromium' or 'webkit'")
    return value


# Subagent browse restrictions (no loopback/private/non-HTTP) apply to ALL
# delegated subagents — read-only, acting, and fail-closed missing-constraint.
# Same fail-closed predicate as secret/control READ denials (SSOT in tools.core).
from ouroboros.tools.core import is_restricted_subagent_profile as _readonly_subagent


def _is_subagent_blocked_browser_url(url: str, ctx: Any = None) -> bool:
    parsed = urlparse(str(url or ""))
    scheme = parsed.scheme
    if scheme == "file":
        # Readonly/acting subagents may open their OWN built files for visual
        # checks, scoped to the task's explicit workspace root only — never the
        # data root, so secrets like data/settings.json stay unreachable.
        return not _file_url_under_workspace(parsed, ctx)
    if scheme not in {"http", "https"}:
        return True
    host = (parsed.hostname or "").strip().rstrip(".").lower()
    if not host:
        return True
    if is_loopback_host(host) or host == "localhost":
        # Local app verification is allowed EXCEPT the Ouroboros control-plane
        # ports: loopback API is unauthenticated (server_auth bypasses auth for
        # loopback), so a subagent must never reach it.
        return _is_blocked_loopback_port(parsed)
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if _NONSTANDARD_NUMERIC_IPV4_RE.match(host):
            return True
        return _hostname_resolves_to_blocked_ip(host)
    if ip.is_loopback:
        return _is_blocked_loopback_port(parsed)
    return _is_blocked_subagent_ip(ip)


def _control_plane_loopback_ports() -> set[int]:
    """Ouroboros loopback control-plane ports a subagent must never reach: the three live
    defaults agent-API (8765), local-model (8765+1=8766) and host-service (8765+2=8767);
    the configured LOCAL_MODEL_PORT; the ACTUAL bound server port (find_free_port may fall
    back, recorded in state/server_port); and any isolated-run server's EXPLICIT
    OUROBOROS_SERVER_PORT / OUROBOROS_HOST_SERVICE_PORT. The +1/+2 above are the fixed
    default ports, NOT adjacency guesses — configured/bound ports are blocked EXACTLY (the
    isolated server sets both env ports independently, so no neighbor needs guessing)."""
    ports = {AGENT_SERVER_PORT, AGENT_SERVER_PORT + 1, AGENT_SERVER_PORT + 2}
    for env in ("OUROBOROS_SERVER_PORT", "OUROBOROS_HOST_SERVICE_PORT", "LOCAL_MODEL_PORT"):
        value = os.environ.get(env, "").strip()
        if value.isdigit():
            ports.add(int(value))
    # The server may bind a fallback port (find_free_port) recorded only in state.
    try:
        from ouroboros.config import DATA_DIR

        port_text = (DATA_DIR / "state" / "server_port").read_text(encoding="utf-8").strip()
        if port_text.isdigit():
            ports.add(int(port_text))
    except (OSError, ValueError):
        pass
    return ports


def _is_blocked_loopback_port(parsed: Any) -> bool:
    try:
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return True
    return int(port) in _control_plane_loopback_ports()


def _file_url_under_workspace(parsed: Any, ctx: Any) -> bool:
    """True only when a file:// path resolves under the task's EXPLICIT workspace
    root, so a subagent can view its own built app but not the data root/secrets."""
    if ctx is None:
        return False
    ws = str(getattr(ctx, "workspace_root", "") or "").strip()
    if not ws:
        return False
    try:
        from urllib.request import url2pathname

        path = pathlib.Path(url2pathname(parsed.path)).resolve(strict=False)
        base = pathlib.Path(ws).resolve(strict=False)
        path.relative_to(base)
        return True
    except (ValueError, OSError):
        return False


def _is_blocked_subagent_ip(ip: ipaddress._BaseAddress) -> bool:
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
    )


# AWS IMDSv6 endpoint; the IPv4 metadata services all live in 169.254.0.0/16
# (link-local), which ``is_link_local`` covers including decimal/hex URL
# spellings once ipaddress normalizes the resolved address.
_METADATA_IPV6_ADDRESSES = frozenset({ipaddress.ip_address("fd00:ec2::254")})


def _is_metadata_ip(ip: ipaddress._BaseAddress) -> bool:
    # Unwrap IPv4-mapped IPv6 (http://[::ffff:169.254.169.254]/) so the
    # link-local check sees the real IPv4 — mirrors mcp_client's guard.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_link_local) or ip in _METADATA_IPV6_ADDRESSES


def _is_metadata_blocked_browser_url(url: str) -> bool:
    """Main-agent guard: True only for link-local/cloud-metadata destinations."""
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip().rstrip(".").lower()
    if not host:
        return False
    try:
        return _is_metadata_ip(ipaddress.ip_address(host))
    except ValueError:
        pass
    if _NONSTANDARD_NUMERIC_IPV4_RE.match(host):
        # Decimal/hex IPv4 spellings (e.g. http://2852039166/) bypass naive
        # string checks; resolve via inet_aton normalization below.
        try:
            packed = socket.inet_aton(host)
            return _is_metadata_ip(ipaddress.ip_address(socket.inet_ntoa(packed)))
        except OSError:
            return True
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False  # unresolvable hosts fail naturally at fetch time
    for info in infos:
        try:
            if _is_metadata_ip(ipaddress.ip_address(str(info[4][0]))):
                return True
        except ValueError:
            continue
    return False


def _hostname_resolves_to_blocked_ip(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return True
    if not infos:
        return True
    for info in infos:
        try:
            sockaddr = info[4]
            ip = ipaddress.ip_address(str(sockaddr[0]))
        except Exception:
            return True
        if _is_blocked_subagent_ip(ip):
            return True
    return False


def _has_platform_browser(local_browsers_dir: pathlib.Path, engine: str = "chromium") -> bool:
    """Return True when a platform-matching bundled browser executable exists."""
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
    plat = sys.platform
    if plat == "darwin":
        candidates = ["chrome-mac", "chrome-headless-shell-mac"]
    elif plat.startswith("win"):
        candidates = ["chrome-win", "chrome-headless-shell-win"]
    else:
        candidates = ["chrome-linux", "chrome-headless-shell-linux"]
    for chromium_dir in local_browsers_dir.iterdir():
        if not (
            chromium_dir.name.startswith("chromium-")
            or chromium_dir.name.startswith("chromium_headless_shell-")
        ):
            continue
        for sub in chromium_dir.iterdir():
            if not any(sub.name.startswith(c) for c in candidates):
                continue
            # Avoid treating partial downloads as usable browser bundles.
            if (
                (plat == "darwin" and (
                    (sub / "Chromium.app" / "Contents" / "MacOS" / "Chromium").exists()
                    or (sub / "chrome-headless-shell").exists()
                ))
                or (plat.startswith("win") and (
                    (sub / "chrome.exe").exists()
                    or (sub / "chrome-headless-shell.exe").exists()
                ))
                or (not plat.startswith(("darwin", "win")) and (
                    (sub / "chrome").exists()
                    or (sub / "chrome-headless-shell").exists()
                ))
            ):
                return True
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
    """Install Playwright and the requested browser engine if not already available."""
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
        log.info("Playwright not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])

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
            data_dir = pathlib.Path(
                os.environ.get("OUROBOROS_DATA_DIR") or pathlib.Path.home() / "Ouroboros" / "data"
            )
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

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            executable_path = pathlib.Path(str(getattr(pw, engine).executable_path))
        if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") == "0":
            try:
                import playwright as _pw_pkg
                local_browsers = pathlib.Path(_pw_pkg.__file__).parent / "driver" / "package" / ".local-browsers"
            except Exception:
                local_browsers = None
            if local_browsers is None or not _has_platform_browser(local_browsers, engine):
                raise RuntimeError(f"bundled Playwright {engine} is missing")
        elif not executable_path.exists():
            raise RuntimeError(f"Playwright {engine} binary not found at {executable_path}")
        log.info("Playwright %s binary found", engine)
    except Exception:
        if not allow_install:
            raise RuntimeError(
                f"Browser tools are unavailable in local_readonly_subagent mode because {engine} is not already installed."
            )
        if not getattr(sys, "frozen", False):
            install_python = sys.executable
        else:
            install_python = ""
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
                            install_python = str(candidate)
                            break
                    if install_python:
                        break
            except Exception:
                install_python = ""
            if not install_python:
                raise RuntimeError(
                    "Playwright browser install requires the embedded python-standalone interpreter, "
                    "but it was not found in this packaged app."
                )
        log.info("Installing Playwright %s dependencies and binary...", engine)
        try:
            subprocess.check_call([install_python, "-m", "playwright", "install-deps", engine])
        except Exception as exc:
            log.warning("Playwright system dependency repair failed; continuing with browser download: %s", exc)
        subprocess.check_call([install_python, "-m", "playwright", "install", engine])

    _playwright_ready_engines.add((engine, str(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "")))
    _playwright_ready = True


def _maybe_alias_playwright_binary(exc: Exception) -> bool:
    """Bridge x64->arm64 browser cache lookups on Apple Silicon when possible."""
    match = _MISSING_EXECUTABLE_RE.search(str(exc))
    if not match:
        return False

    missing_path = pathlib.Path(match.group(1).strip())
    missing_dir = missing_path.parent
    if "-mac-x64" not in str(missing_dir):
        return False

    alternate_dir = pathlib.Path(str(missing_dir).replace("-mac-x64", "-mac-arm64"))
    alternate_binary = alternate_dir / missing_path.name
    if not alternate_binary.exists():
        return False

    try:
        if missing_dir.exists():
            return missing_path.exists()
        missing_dir.symlink_to(alternate_dir, target_is_directory=True)
        log.info("Aliased Playwright browser cache %s -> %s", missing_dir, alternate_dir)
        return True
    except OSError:
        log.debug("Failed to alias Playwright browser cache", exc_info=True)
        return False


def _launch_browser_with_fallback(pw_instance: Any, *, engine: str = "chromium", allow_cache_write: bool = True) -> Any:
    engine = _normalize_browser_engine(engine)
    launch_kwargs = {
        "headless": True,
    }
    if engine == "chromium":
        launch_kwargs["args"] = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--disable-features=site-per-process",
            "--window-size=1920,1080",
            # J (v6.39): software-GL via ANGLE+SwiftShader so WebGL/canvas/3D actually
            # RENDER in headless (bundled chrome-headless-shell has no GPU) instead of a
            # black frame; ignore the GPU blocklist so SwiftShader is used.
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
        ]
    browser_type = getattr(pw_instance, engine)
    try:
        return browser_type.launch(**launch_kwargs)
    except Exception as exc:
        if engine == "chromium" and allow_cache_write and _maybe_alias_playwright_binary(exc):
            return browser_type.launch(**launch_kwargs)
        raise


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


def _ensure_browser(ctx: ToolContext, *, engine: str = "chromium", device: str = ""):
    """Create or reuse this context's browser; no module-level Playwright state."""
    engine = _normalize_browser_engine(engine)
    requested_device = str(device or "").strip()
    bs = ctx.browser_state
    current_thread_id = threading.get_ident()
    stored_thread_id = getattr(bs, "_thread_id", None)
    stored_engine = getattr(bs, "_browser_engine", "")
    stored_device = getattr(bs, "_browser_device", "")

    if stored_thread_id is not None and stored_thread_id != current_thread_id:
        log.info("Thread switch detected (old=%s, new=%s). Tearing down browser for this context.",
                 stored_thread_id, current_thread_id)
        cleanup_browser(ctx)
    elif bs.browser is not None and (
        stored_engine != engine
        or stored_device.lower() != requested_device.lower()
    ):
        log.info("Browser engine/device changed (%s/%s -> %s/%s); recreating context.",
                 stored_engine or "chromium", stored_device, engine, requested_device)
        cleanup_browser(ctx)

    if bs.browser is not None:
        try:
            if bs.browser.is_connected():
                return bs.page
        except Exception:
            log.debug("Browser connection check failed", exc_info=True)
        cleanup_browser(ctx)

    readonly_subagent = _readonly_subagent(ctx)
    _ensure_playwright_installed(engine=engine, allow_install=not readonly_subagent)

    if bs.pw_instance is None:
        from playwright.sync_api import sync_playwright
        bs.pw_instance = sync_playwright().start()
        setattr(bs, "_thread_id", current_thread_id)
        log.info("Created Playwright instance in thread %s", current_thread_id)

    bs.browser = _launch_browser_with_fallback(bs.pw_instance, engine=engine, allow_cache_write=not readonly_subagent)
    device_options, resolved_device = _device_context_options(bs.pw_instance, requested_device)
    context_options = device_options or _default_context_options(engine)
    bs_context = bs.browser.new_context(**context_options)
    setattr(bs, "_browser_context", bs_context)
    setattr(bs, "_browser_engine", engine)
    setattr(bs, "_browser_device", resolved_device or requested_device)
    bs.page = bs_context.new_page()

    if _HAS_STEALTH:
        stealth = Stealth()
        stealth.apply_stealth_sync(bs.page)

    bs.page.set_default_timeout(30000)
    # Browser tools are agent-controlled. They may inspect the UI, but must not
    # use clicks/fetches to change the owner-controlled context horizon.
    bs_context.route("**/api/owner/context-mode", _block_context_mode_owner_post)
    bs_context.route("**/api/owner/scope-review-floor", _block_scope_review_floor_owner_post)
    # Broad glob (any /api/owner/** path): the glob matches the RAW URL, so a
    # percent-encoded `safety%2Dmode` would slip a literal pattern — the handler
    # URL-DECODES and aborts only the safety-mode POST (review round 6).
    bs_context.route("**/api/owner/**", _block_safety_mode_owner_post)
    # Broad glob (any /api/owner/skills/** path) so a percent-encoded `attest%2Dreview`
    # still routes to the handler, which then URL-DECODES and precisely aborts the
    # attestation POST (the glob matches the RAW URL, so it must not assume the literal).
    bs_context.route("**/api/owner/skills/**", _block_owner_skill_attest_post)
    # Owner-only self-modification toggles ride /api/settings; block the browser
    # click+Save path (POST /api/settings) for them, not just evaluate-JS. Applies to
    # every browser session (root + subagents).
    bs_context.route("**/api/settings", _block_owner_settings_post)
    if readonly_subagent:
        bs_context.route(
            "**/*",
            lambda route: route.abort()
            if _is_subagent_blocked_browser_url(route.request.url, ctx)
            else _route_fallback(route),
        )
    else:
        # Main-agent SSRF guard (conservative): block ONLY link-local /
        # cloud-metadata endpoints (169.254.0.0/16 incl. decimal/hex spellings,
        # fd00:ec2::254). Private/LAN stays reachable — owners legitimately
        # browse their own LAN services. Route interception re-validates every
        # hop, so redirects cannot smuggle a metadata fetch.
        bs_context.route(
            "**/*",
            lambda route: route.abort()
            if _is_metadata_blocked_browser_url(route.request.url)
            else _route_fallback(route),
        )
    return bs.page


def cleanup_browser(ctx: ToolContext) -> None:
    """Close page/browser and stop the Playwright instance."""
    bs = ctx.browser_state
    try:
        if bs.page is not None:
            bs.page.close()
    except Exception:
        log.debug("Failed to close browser page during cleanup", exc_info=True)
    try:
        browser_context = getattr(bs, "_browser_context", None)
        if browser_context is not None:
            browser_context.close()
    except Exception:
        log.debug("Failed to close browser context during cleanup", exc_info=True)
    try:
        if bs.browser is not None:
            bs.browser.close()
    except Exception:
        log.debug("Failed to close browser during cleanup", exc_info=True)
    try:
        if bs.pw_instance is not None:
            bs.pw_instance.stop()
    except Exception:
        log.debug("Failed to stop Playwright instance during cleanup", exc_info=True)
    bs.page = None
    bs.browser = None
    bs.pw_instance = None
    setattr(bs, "_thread_id", None)
    setattr(bs, "_browser_context", None)
    setattr(bs, "_browser_engine", "")
    setattr(bs, "_browser_device", "")


def _is_infrastructure_error(obj: Any) -> bool:
    """Detect context-state or legacy string-based browser infrastructure failures."""
    if hasattr(obj, "browser_state"):
        bs = obj.browser_state
        if bs.browser is None or bs.pw_instance is None:
            return True
        try:
            if not bs.browser.is_connected():
                return True
        except Exception:
            return True
        if bs.page is not None:
            try:
                if bs.page.is_closed():
                    return True
            except Exception:
                return True
        return False

    msg = str(obj).lower()
    return any(token in msg for token in (
        "green thread",
        "different thread",
        "browser has been closed",
        "page has been closed",
        "connection closed",
    ))


def _blocks_context_mode_self_lowering_js(value: str) -> bool:
    low = str(value or "").lower()
    return "low" in low and (
        "/api/owner/context-mode" in low
        or ("ouroboros_context_mode" in low and ("settings.json" in low or "save_settings" in low))
    )


def _blocks_scope_review_floor_self_lowering_js(value: str) -> bool:
    """Block browser JS that tries to weaken the owner-only scope-review floor
    (CW1, v6.34.0) — the click+fetch bypass of the dedicated owner endpoint."""
    low = str(value or "").lower()
    return (
        "/api/owner/scope-review-floor" in low
        or ("ouroboros_scope_review_floor" in low and (
            "settings.json" in low or "save_settings" in low or "/api/settings" in low
        ))
    )


def _blocks_safety_mode_self_lowering_js(value: str) -> bool:
    """Block browser JS that tries to change the owner-only LLM-safety coverage mode
    (v6.54.3) — the click+fetch bypass of the dedicated owner endpoint. URL-decode
    first so a percent-encoded path (``safety%2Dmode``) cannot slip the literal
    match (review round 6; mirrors the owner-attestation guard)."""
    import urllib.parse

    low = str(value or "").lower()
    decoded = urllib.parse.unquote(urllib.parse.unquote(low)).lower()
    text = f"{low} {decoded}"
    return (
        "/api/owner/safety-mode" in text
        or ("ouroboros_safety_mode" in text and (
            "settings.json" in text or "save_settings" in text or "/api/settings" in text
        ))
    )


def _blocks_mutative_toggle_js(value: str) -> bool:
    """Block browser JS that tries to enable the owner-only mutative-subagents toggle."""
    low = str(value or "").lower()
    return "ouroboros_allow_mutative_subagents" in low and (
        "settings.json" in low or "save_settings" in low or "/api/settings" in low
    )


def _blocks_post_task_evolution_js(value: str) -> bool:
    """Block browser JS that tries to set an owner-only self-evolution control (the
    post-task evolution toggle or the persistent evolution-objective steer)."""
    low = str(value or "").lower()
    return (
        "ouroboros_post_task_evolution" in low
        or "ouroboros_evolution_persistent_objective" in low
    ) and (
        "settings.json" in low or "save_settings" in low or "/api/settings" in low
    )


def _blocks_owner_skill_attest_js(value: str) -> bool:
    """Block browser JS that tries to loopback-POST the OWNER-ONLY skill-attestation
    endpoint (C1, v6.39) — owner-attestation skips the LLM skill review and must be
    owner-issued, never agent self-callable from a browser fetch. URL-decode first so a
    percent-encoded path (``%61ttest-review``) cannot slip past the literal match."""
    import urllib.parse
    low = str(value or "").lower()
    decoded = urllib.parse.unquote(urllib.parse.unquote(low)).lower()
    text = f"{low} {decoded}"
    return "/api/owner/skills/" in text and "attest-review" in text


def _route_fallback(route: Any) -> None:
    """Pass a non-matching request DOWN the guard chain (review round 8).

    Playwright runs route handlers in REVERSE registration order and
    ``route.continue_()`` FINALIZES the request — a later-registered catch-all
    that continued would silently skip every earlier-registered owner-endpoint
    block (they were dead, not merely ordered oddly). ``route.fallback()``
    defers to the next matching handler and plain-continues when none remain,
    so every guard in the chain actually evaluates. Older Playwright fakes/
    versions without ``fallback`` degrade to ``continue_`` (the historical
    behavior)."""
    fallback = getattr(route, "fallback", None)
    if callable(fallback):
        fallback()
        return
    route.continue_()


def _is_context_mode_owner_post(request: Any) -> bool:
    try:
        parsed = urlparse(str(request.url or ""))
        method = str(request.method or "").upper()
    except Exception:
        return False
    return method == "POST" and parsed.path.rstrip("/") == "/api/owner/context-mode"


def _block_context_mode_owner_post(route: Any) -> None:
    if _is_context_mode_owner_post(route.request):
        route.abort()
        return
    _route_fallback(route)


def _is_scope_review_floor_owner_post(request: Any) -> bool:
    try:
        parsed = urlparse(str(request.url or ""))
        method = str(request.method or "").upper()
    except Exception:
        return False
    return method == "POST" and parsed.path.rstrip("/") == "/api/owner/scope-review-floor"


def _block_scope_review_floor_owner_post(route: Any) -> None:
    if _is_scope_review_floor_owner_post(route.request):
        route.abort()
        return
    _route_fallback(route)


def _is_safety_mode_owner_post(request: Any) -> bool:
    """POST to the owner safety-mode endpoint — decoded, so a percent-encoded
    path cannot slip past (the broad ``**/api/owner/**`` route registration
    feeds RAW URLs here; Starlette decodes server-side, so we must too)."""
    import urllib.parse

    try:
        parsed = urlparse(str(request.url or ""))
        method = str(request.method or "").upper()
    except Exception:
        return False
    path = urllib.parse.unquote(urllib.parse.unquote(parsed.path)).rstrip("/")
    return method == "POST" and path == "/api/owner/safety-mode"


def _block_safety_mode_owner_post(route: Any) -> None:
    if _is_safety_mode_owner_post(route.request):
        route.abort()
        return
    _route_fallback(route)


def _is_owner_skill_attest_post(request: Any) -> bool:
    """A browser POST to the owner-only skill owner-attestation endpoint — the click/form
    bypass of the evaluate-only JS guard (C1, v6.39)."""
    try:
        import urllib.parse
        parsed = urlparse(str(request.url or ""))
        method = str(request.method or "").upper()
        # Decode so a percent-encoded path (which the server decodes before routing) is
        # matched the same way the route is registered.
        path = urllib.parse.unquote(urllib.parse.unquote(parsed.path)).rstrip("/").lower()
    except Exception:
        return False
    return method == "POST" and path.startswith("/api/owner/skills/") and path.endswith("/attest-review")


def _block_owner_skill_attest_post(route: Any) -> None:
    if _is_owner_skill_attest_post(route.request):
        route.abort()
        return
    _route_fallback(route)


def _is_owner_settings_self_elevation_post(request: Any) -> bool:
    """A browser POST /api/settings carrying an owner-only self-modification toggle —
    the click+Save bypass of the evaluate-only JS guards."""
    try:
        if str(request.method or "").upper() != "POST":
            return False
        parsed = urlparse(str(request.url or ""))
        if parsed.path.rstrip("/") != "/api/settings":
            return False
        body = str(request.post_data or "").lower()
    except Exception:
        return False
    return (
        "ouroboros_post_task_evolution" in body
        or "ouroboros_allow_mutative_subagents" in body
        or "ouroboros_evolution_persistent_objective" in body
    )


def _block_owner_settings_post(route: Any) -> None:
    if _is_owner_settings_self_elevation_post(route.request):
        route.abort()
        return
    _route_fallback(route)


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
    """Hand a fresh screenshot to a vision-capable active model natively.

    The screenshot is saved to ``data/uploads/screenshots/<ts>.png`` (the
    re-view path used by eviction placeholders) and injected as a user-role
    image block via the existing multipart-preserving merge. The TOOL result
    stays a plain string — the tool-message contract is unchanged. Non-vision
    models keep the analyze_screenshot/vlm_query flow.
    """
    try:
        from ouroboros.provider_models import supports_vision

        # Resolve the model THIS task is actually running on (the loop publishes
        # ctx.active_model each round, incl. switch_model / per-task overrides);
        # fall back to the per-task override, then the global env default. Reading
        # OUROBOROS_MODEL alone misclassified vision when the live model differed.
        active_model = (
            str(getattr(ctx, "active_model", "") or "")
            or str(getattr(ctx, "task_model_override", "") or "")
            or str(os.environ.get("OUROBOROS_MODEL", "") or "")
        )
        if not supports_vision(active_model):
            return ""
        messages = getattr(ctx, "messages", None)
        if not isinstance(messages, list):
            return ""
        from ouroboros.utils import utc_now_iso

        ts = utc_now_iso().replace(":", "").replace("-", "")[:15]
        shot_dir = pathlib.Path(ctx.drive_root) / "uploads" / "screenshots"
        shot_dir.mkdir(parents=True, exist_ok=True)
        shot_path = shot_dir / f"{ts}.png"
        shot_path.write_bytes(base64.b64decode(b64))
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
        ])
        return "The screenshot is attached to your context natively (vision model). "
    except Exception:
        log.debug("native screenshot injection failed", exc_info=True)
        return ""


def _page_health_snapshot(page: Any) -> str:
    """Cheap textual page diagnostics (native Playwright, zero LLM cost) so a
    screenshot caller can tell whether the page actually loaded/rendered even when
    no vision model is available. Each probe is independently defensive."""
    parts: list = []
    try:
        parts.append(f"url={page.url}")
    except Exception:
        pass
    try:
        parts.append("title=" + repr((page.title() or "")[:120]))
    except Exception:
        pass
    try:
        counts = page.evaluate(
            "() => ({canvas: document.querySelectorAll('canvas').length,"
            " img: document.querySelectorAll('img').length,"
            " scripts: document.querySelectorAll('script').length,"
            " bodyChars: (document.body && document.body.innerText || '').length})"
        )
        if isinstance(counts, dict):
            parts.append("elements=" + ",".join(f"{k}:{v}" for k, v in counts.items()))
    except Exception:
        pass
    try:
        body = (page.inner_text("body") or "").strip().replace("\n", " ")
        if body:
            parts.append("bodyText=" + repr(body[:200]))
    except Exception:
        pass
    return " | ".join(parts)


def _wait_for_page_paint(page: Any, timeout_ms: int = 3000) -> None:
    """J (v6.39): let the page paint before a screenshot — wait for document.readyState then
    two requestAnimationFrames (the second fires AFTER the paint), so a freshly-rendered
    canvas/WebGL frame is not captured black/blank. Best-effort + bounded; never blocks."""
    try:
        page.wait_for_function("document.readyState === 'complete'", timeout=min(int(timeout_ms or 3000), 3000))
    except Exception:
        pass
    try:
        # Schedule a paint flag via double-rAF (the second fires AFTER the paint), then wait
        # for it with a HARD Playwright timeout. The flag-set evaluate returns immediately
        # (it does not await a page-owned promise), and wait_for_function's own timeout
        # bounds the wait — so a page that suppresses requestAnimationFrame can never hang
        # the capture (the page's own timers are never trusted to unblock us).
        page.evaluate(
            "() => { window.__obo_painted = false;"
            " requestAnimationFrame(() => requestAnimationFrame(() => { window.__obo_painted = true; })); }"
        )
        page.wait_for_function("window.__obo_painted === true", timeout=500)
    except Exception:
        pass


def _extract_page_output(page: Any, output: str, ctx: ToolContext) -> str:
    """Extract page content in the requested format."""
    if output == "screenshot":
        _wait_for_page_paint(page)  # J: paint before capture (shared with browser_action)
        data = page.screenshot(type="png", full_page=False)
        b64 = base64.b64encode(data).decode()
        ctx.browser_state.last_screenshot_b64 = b64
        health = _page_health_snapshot(page)
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
    elif output == "html":
        html = page.content()
        return html[:50000] + ("... [truncated]" if len(html) > 50000 else "")
    elif output == "markdown":
        text = page.evaluate(_MARKDOWN_JS)
        return text[:30000] + ("... [truncated]" if len(text) > 30000 else "")
    else:  # text
        text = page.inner_text("body")
        return text[:30000] + ("... [truncated]" if len(text) > 30000 else "")


def _browse_page(ctx: ToolContext, url: str, output: str = "text",
                 wait_for: str = "", timeout: int = 30000,
                 viewport: str = "", engine: str = "chromium", device: str = "") -> str:
    readonly_subagent = _readonly_subagent(ctx)
    if readonly_subagent and _is_subagent_blocked_browser_url(str(url or ""), ctx):
        return "⚠️ BROWSER_LOCAL_READONLY_BLOCKED: subagents may browse external HTTP(S), localhost (non-Ouroboros ports), and file:// under their workspace — not the Ouroboros API ports, private/link-local IPs, or other schemes."
    try:
        page = _ensure_browser(ctx, engine=engine, device=device)
        if viewport:
            _apply_viewport(page, viewport)
        page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        if wait_for:
            page.wait_for_selector(wait_for, timeout=timeout)
        if readonly_subagent and _is_subagent_blocked_browser_url(str(getattr(page, "url", "") or ""), ctx):
            return "⚠️ BROWSER_LOCAL_READONLY_BLOCKED: subagents may browse external HTTP(S), localhost (non-Ouroboros ports), and file:// under their workspace — not the Ouroboros API ports, private/link-local IPs, or other schemes."
        return _extract_page_output(page, output, ctx)
    except Exception as e:
        if _is_infrastructure_error(ctx):
            log.warning("Browser infrastructure error: %s. Cleaning up and retrying...", e)
            cleanup_browser(ctx)
            page = _ensure_browser(ctx, engine=engine, device=device)
            if viewport:
                _apply_viewport(page, viewport)
            page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            if wait_for:
                page.wait_for_selector(wait_for, timeout=timeout)
            if readonly_subagent and _is_subagent_blocked_browser_url(str(getattr(page, "url", "") or ""), ctx):
                return "⚠️ BROWSER_LOCAL_READONLY_BLOCKED: subagents may browse external HTTP(S), localhost (non-Ouroboros ports), and file:// under their workspace — not the Ouroboros API ports, private/link-local IPs, or other schemes."
            return _extract_page_output(page, output, ctx)
        raise


def _apply_viewport(page: Any, viewport: str) -> None:
    """Parse a 'WxH' string and resize the browser viewport."""
    try:
        parts = viewport.lower().split("x")
        w, h = int(parts[0]), int(parts[1])
        page.set_viewport_size({"width": max(320, w), "height": max(480, h)})
    except (ValueError, IndexError):
        log.warning("Invalid viewport '%s', expected WxH (e.g. '375x812')", viewport)


def _browser_action(ctx: ToolContext, action: str, selector: str = "",
                    value: str = "", timeout: int = 5000,
                    engine: str = "", device: str = "") -> str:
    normalized_action = str(action or "").strip().lower()
    readonly_subagent = _readonly_subagent(ctx)
    if readonly_subagent and normalized_action == "evaluate":
        return "⚠️ BROWSER_LOCAL_READONLY_BLOCKED: subagents cannot run arbitrary browser JavaScript."

    def _do_action():
        page = _ensure_browser(
            ctx,
            engine=engine or getattr(ctx.browser_state, "_browser_engine", "chromium") or "chromium",
            device=device or getattr(ctx.browser_state, "_browser_device", "") or "",
        )
        if readonly_subagent and _is_subagent_blocked_browser_url(str(getattr(page, "url", "") or ""), ctx):
            return "⚠️ BROWSER_LOCAL_READONLY_BLOCKED: subagents may act on external HTTP(S), localhost (non-Ouroboros ports), and file:// under their workspace — not the Ouroboros API ports or private/link-local pages."

        if normalized_action == "click":
            if not selector:
                return "Error: selector required for click"
            page.click(selector, timeout=timeout)
            page.wait_for_timeout(500)
            return f"Clicked: {selector}"
        elif normalized_action == "fill":
            if not selector:
                return "Error: selector required for fill"
            page.fill(selector, value, timeout=timeout)
            return f"Filled {selector} with: {value}"
        elif normalized_action == "select":
            if not selector:
                return "Error: selector required for select"
            page.select_option(selector, value, timeout=timeout)
            return f"Selected {value} in {selector}"
        elif normalized_action == "screenshot":
            _wait_for_page_paint(page, int(timeout or 3000))  # J: paint before capture
            data = page.screenshot(type="png", full_page=False)
            b64 = base64.b64encode(data).decode()
            ctx.browser_state.last_screenshot_b64 = b64
            if readonly_subagent:
                return (
                    f"Screenshot captured ({len(b64)} bytes base64). "
                    "Use analyze_screenshot to inspect it."
                )
            return (
                f"Screenshot captured ({len(b64)} bytes base64). "
                f"Call send_photo(image_base64='__last_screenshot__') to deliver it to the user."
            )
        elif normalized_action == "evaluate":
            if not value:
                return "Error: value (JS code) required for evaluate"
            if _blocks_context_mode_self_lowering_js(value):
                return (
                    "⚠️ CONTEXT_MODE_SELF_LOWERING_BLOCKED: browser JavaScript "
                    "looks like an attempt to lower OUROBOROS_CONTEXT_MODE. "
                    "Context mode is owner-controlled — ask the owner to use "
                    "the Low/Max toggle."
                )
            if _blocks_scope_review_floor_self_lowering_js(value):
                return (
                    "⚠️ SCOPE_REVIEW_FLOOR_SELF_LOWERING_BLOCKED: browser JavaScript "
                    "looks like an attempt to weaken OUROBOROS_SCOPE_REVIEW_FLOOR. "
                    "The scope-review floor gates the BIBLE P3 blocking review — it is "
                    "owner-controlled, and the agent must not lower it."
                )
            if _blocks_safety_mode_self_lowering_js(value):
                return (
                    "⚠️ SAFETY_MODE_SELF_LOWERING_BLOCKED: browser JavaScript "
                    "looks like an attempt to change OUROBOROS_SAFETY_MODE. "
                    "LLM-safety coverage is owner-controlled (BIBLE P3) — the agent "
                    "must not reduce its own supervision."
                )
            if _blocks_mutative_toggle_js(value):
                return (
                    "⚠️ ELEVATION_BLOCKED: browser JavaScript looks like an attempt to enable "
                    "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS. This master toggle is owner-controlled — "
                    "the agent must not self-enable mutative subagents."
                )
            if _blocks_post_task_evolution_js(value):
                return (
                    "⚠️ ELEVATION_BLOCKED: browser JavaScript looks like an attempt to enable "
                    "OUROBOROS_POST_TASK_EVOLUTION. Post-task self-evolution is owner-controlled — "
                    "the agent must not self-enable it."
                )
            if _blocks_owner_skill_attest_js(value):
                return (
                    "⚠️ OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED: browser JavaScript looks like an "
                    "attempt to POST /api/owner/skills/<skill>/attest-review. Owner-attestation skips "
                    "the LLM skill review and is owner-only — the agent must not self-attest its own skill."
                )
            try:
                result = page.evaluate(value)
            except Exception as eval_err:  # noqa: BLE001
                msg = str(eval_err)
                if "SyntaxError" not in msg:
                    raise
                # J (v6.39): a statement-style snippet (a top-level `return`, or several
                # statements) is a SyntaxError for a raw evaluate EXPRESSION; retry the same
                # code wrapped in an IIFE function body before surfacing a real parse error.
                try:
                    result = page.evaluate("(() => {\n" + value + "\n})()")
                except Exception as iife_err:  # noqa: BLE001
                    # The IIFE PARSED but threw a RUNTIME error (e.g. ReferenceError) -> that
                    # is the real result; surface it like the raw path, not as a parse error.
                    if "SyntaxError" not in str(iife_err):
                        raise
                    snippet = value.strip()[:80]
                    return (
                        "⚠️ BROWSER_EVALUATE_SYNTAX_ERROR: the JS failed to parse "
                        f"({msg.splitlines()[0][:160]}). First 80 chars: {snippet!r}. "
                        "Check for stray git conflict markers (<<<<<<<) or shell "
                        "heredocs (<<EOF) leaked into the value."
                    )
            out = str(result)
            return out[:20000] + ("... [truncated]" if len(out) > 20000 else "")
        elif normalized_action == "scroll":
            direction = value or "down"
            if direction == "down":
                page.evaluate("window.scrollBy(0, 600)")
            elif direction == "up":
                page.evaluate("window.scrollBy(0, -600)")
            elif direction == "top":
                page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            return f"Scrolled {direction}"
        else:
            return f"Unknown action: {action}. Use: click, fill, select, screenshot, evaluate, scroll"

    try:
        return _do_action()
    except Exception as e:
        if _is_infrastructure_error(ctx):
            log.warning("Browser infrastructure error: %s. Cleaning up and retrying...", e)
            cleanup_browser(ctx)
            return _do_action()
        raise


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="browse_page",
            schema={
                "name": "browse_page",
                "description": (
                    "Open a URL in headless browser. Returns page content as text, "
                    "html, markdown, or screenshot (base64 PNG). "
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
                    "Perform action on current browser page. Actions: "
                    "click (selector), fill (selector + value), select (selector + value), "
                    "screenshot (base64 PNG), evaluate (JS code in value), "
                    "scroll (value: up/down/top/bottom)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["click", "fill", "select", "screenshot", "evaluate", "scroll"],
                            "description": "Action to perform",
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for click/fill/select",
                        },
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
