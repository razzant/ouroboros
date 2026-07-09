"""Isolated-server runner for evolution benchmark drivers (B-full, production-faithful).

Spawns a REAL isolated Ouroboros ``server.py`` against a throwaway repo clone + data
root on a free port, so a benchmark drives the ACTUAL supervisor loop (post-task
evolution -> reviewed commit -> os.execvpe restart -> verify_restart absorb) instead
of a headless ``ouroboros run`` that would attach to whatever server is on the
default port. The live Ouroboros is never touched: a unique port, an isolated clone,
and an isolated data root keep it fully separate.

Why a server (not headless): the post-task evolution signal is only consumed by the
supervisor tick inside ``server.py`` (``apply_pending_request`` +
``enqueue_evolution_task_if_needed``); ``ouroboros run`` is a thin HTTP client.

Model: tests/test_ui_smoke_playwright.py::direct_server_with_data +
devtools/benchmarks/terminal_bench/harbor_installed_agent.py.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from ouroboros.platform_layer import (
    kill_pid_tree,
    subprocess_new_group_kwargs,
    terminate_process_tree,
)

_FINAL_STATUSES = {"completed", "failed", "cancelled", "rejected_duplicate"}

# Live/managed runtime env keys that must NEVER leak into an isolated benchmark server:
# the sanitized settings.json is the source of truth, so an inherited value here would
# silently route the throwaway server through the LIVE local-model/runtime/host config.
# SSOT for BOTH IsolatedServer._env() (process env) and the drivers' _seed_settings()
# (copied settings.json) so the two sanitizations can never drift apart.
STALE_INHERITED_ENV_KEYS = (
    "OUROBOROS_SERVER_HOST", "OUROBOROS_SERVER_PORT", "OUROBOROS_HOST_SERVICE_PORT",
    "OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH",
    "OUROBOROS_URL", "OUROBOROS_MANAGED_BY_LAUNCHER",
    # The parent-pinned runtime-mode baseline is exported to subprocesses and is PREFERRED
    # over settings.json (config.initialize_runtime_mode_baseline / get_runtime_mode), so an
    # inherited value would boot the isolated server in the LIVE mode instead of its own
    # advanced sandbox — strip it so the sanitized settings win.
    "OUROBOROS_BOOT_RUNTIME_MODE",
    "USE_LOCAL_MAIN", "USE_LOCAL_CODE", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
    # Owner/control SECRETS must never leak into the isolated server's env (untrusted
    # benchmark tasks run here). Provider creds are loaded from the sanitized settings.json.
    "GITHUB_TOKEN", "GITHUB_REPO", "OUROBOROS_NETWORK_PASSWORD",
)

# Allowlist for seeding an ISOLATED benchmark settings.json from live settings: ONLY provider
# credentials/endpoints, model slots, effort, and budget. Owner/control secrets and knobs
# (GITHUB_TOKEN, OUROBOROS_NETWORK_PASSWORD, transport/skill secrets, owner chat ids, etc.)
# are NEVER copied — the isolated data root is readable by untrusted benchmark tasks.
# Model-slot / effort / local-model / gigachat-provider key families (all are model or
# provider config — safe to copy). GIGACHAT_* covers its credentials+endpoint+scope.
_ISO_SETTINGS_ALLOW_PREFIX = ("OUROBOROS_MODEL", "OUROBOROS_EFFORT", "LOCAL_MODEL_", "GIGACHAT_")
# EXPLICIT provider creds/endpoints + review/budget keys. Deliberately NOT a `*_API_KEY`
# pattern: a custom skill secret could be named `<x>_API_KEY` and must NOT be copied.
_ISO_SETTINGS_ALLOW_EXACT = frozenset({
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
    "ANTHROPIC_API_KEY",
    "OUROBOROS_WEBSEARCH_MODEL", "OUROBOROS_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL",
    # Review policy knobs (non-secret): must propagate so settings.json's task-acceptance
    # self-review config is honored by isolated benchmark servers (else it silently
    # falls back to the "auto" default and the end-of-task review never runs).
    "OUROBOROS_TASK_REVIEW_MODE", "OUROBOROS_REVIEW_ENFORCEMENT",
    "CLAUDE_CODE_MODEL", "CLAUDE_AGENT_SDK_MODEL",
    "TOTAL_BUDGET", "OUROBOROS_PER_TASK_COST_USD", "OUROBOROS_CONTEXT_MODE",
})


# Provider credentials the isolated agent legitimately needs in its env (kept); every other
# secret-shaped inherited env var is stripped so untrusted benchmark tasks (which inherit the
# server env via shell tools) cannot read owner/skill secrets like TELEGRAM_BOT_TOKEN.
_PROVIDER_ENV_KEYS = frozenset({
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "ANTHROPIC_API_KEY",
    "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD",
})


def _is_secret_env_key(key: str) -> bool:
    """A non-provider secret-shaped env var (token/secret/password/api-key/credentials)."""
    ku = str(key).upper()
    if ku in _PROVIDER_ENV_KEYS:
        return False
    return (
        "TOKEN" in ku or "SECRET" in ku or "PASSWORD" in ku
        or ku.endswith("_API_KEY") or ku.endswith("_CREDENTIALS")
    )


def build_isolated_settings(live_cfg: dict, **overrides) -> dict:
    """Build an isolated benchmark settings.json from live settings, copying ONLY the
    EXPLICIT provider/model/budget allowlist above (never owner/control secrets like
    GITHUB_TOKEN / OUROBOROS_NETWORK_PASSWORD / transport / skill secrets / owner knobs),
    then applying explicit isolated overrides. The isolated data root is reachable by
    untrusted benchmark tasks, so this is the hermetic 'provider keys + model slots only' seed."""
    out: dict = {}
    for key, value in (live_cfg or {}).items():
        ks = str(key)
        if ks in _ISO_SETTINGS_ALLOW_EXACT or ks.startswith(_ISO_SETTINGS_ALLOW_PREFIX):
            out[ks] = value
    out.update(overrides)
    return out


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _api(base_url: str, method: str, path: str, payload: dict | None = None, timeout: float = 60) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(base_url + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def seed_owner_state(data_root: pathlib.Path, *, evolution_enabled: bool = False) -> None:
    """Pre-seed state.json so the evolution loop's owner_chat_id gate passes (the
    /api/tasks path never binds owner_chat_id). Optionally pre-enable the campaign."""
    state_path = pathlib.Path(data_root) / "state" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    st: dict = {}
    if state_path.exists():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
    st["owner_chat_id"] = 1
    if evolution_enabled:
        st["evolution_mode_enabled"] = True
    state_path.write_text(json.dumps(st), encoding="utf-8")


def absorbed_cycles_done(data_root: pathlib.Path) -> int:
    """Read absorbed self-evolution cycle count from evolution_campaign.json."""
    path = pathlib.Path(data_root) / "state" / "evolution_campaign.json"
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("absorbed_cycles_done") or 0)
    except (OSError, ValueError, TypeError):
        return 0


class IsolatedServer:
    """A throwaway Ouroboros server bound to an isolated clone + data root + port."""

    def __init__(self, clone: pathlib.Path, data_root: pathlib.Path, settings_path: pathlib.Path,
                 *, host: str = "127.0.0.1") -> None:
        self.clone = pathlib.Path(clone)
        self.data_root = pathlib.Path(data_root)
        self.settings_path = pathlib.Path(settings_path)
        self.host = host
        self.port = free_port()
        self.host_service_port = free_port()
        self.base_url = f"http://{host}:{self.port}"
        self.proc: subprocess.Popen | None = None

    def _env(self) -> dict:
        env = dict(os.environ)
        # Strip ALL stale live/managed runtime keys FIRST, so an Ouroboros-managed launch
        # environment cannot REINTRODUCE values that _seed_settings stripped from the copied
        # settings (hermetic isolation: the sanitized settings.json is the source of truth;
        # a leaked USE_LOCAL_*/host/path here would route the throwaway server through live
        # config). This includes OUROBOROS_MANAGED_BY_LAUNCHER (direct self-re-exec, not
        # launcher-managed) and OUROBOROS_URL (never point the in-process CLI at another server).
        for key in STALE_INHERITED_ENV_KEYS:
            env.pop(key, None)
        for key in list(env):
            if _is_secret_env_key(key):
                env.pop(key, None)
        # Then apply the isolated overrides explicitly (these win over anything inherited).
        env.update({
            "OUROBOROS_APP_ROOT": str(self.clone.parent),
            "OUROBOROS_REPO_DIR": str(self.clone),
            "OUROBOROS_DATA_DIR": str(self.data_root),
            "OUROBOROS_SETTINGS_PATH": str(self.settings_path),
            "OUROBOROS_SERVER_HOST": self.host,
            "OUROBOROS_SERVER_PORT": str(self.port),
            "OUROBOROS_HOST_SERVICE_PORT": str(self.host_service_port),
        })
        return env

    def _patch_settings_ports(self) -> None:
        """Write the chosen free ports INTO settings.json. The server applies
        settings.json over env at startup (apply_settings_to_env), so the host-service
        port must live in settings or it falls back to the default 8767 and collides
        with the live server."""
        cfg: dict = {}
        try:
            if self.settings_path.exists():
                cfg = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        cfg["OUROBOROS_SERVER_HOST"] = self.host
        cfg["OUROBOROS_SERVER_PORT"] = self.port
        cfg["OUROBOROS_HOST_SERVICE_PORT"] = self.host_service_port
        self.settings_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    def start(self, ready_timeout: float = 180) -> "IsolatedServer":
        self._patch_settings_ports()
        # Own process group/session so a hung server + its worker children can be
        # killed as a tree (platform_layer), not orphaned past graceful SIGTERM.
        self.proc = subprocess.Popen(
            [sys.executable, "server.py"], cwd=str(self.clone), env=self._env(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            **subprocess_new_group_kwargs(),
        )
        try:
            self._wait_ready(ready_timeout)
        except BaseException:
            # NEVER orphan the spawned server/worker tree if readiness fails (timeout, etc.):
            # via __enter__ a raise here would skip __exit__, leaking the process group.
            self.stop()
            raise
        return self

    def _state(self, timeout: float = 5) -> dict:
        return _api(self.base_url, "GET", "/api/state", timeout=timeout)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"isolated server exited early (rc={self.proc.returncode})")
            try:
                st = self._state()
                if st.get("supervisor_ready") and int(st.get("workers_total") or 0) > 0:
                    return
                last = f"supervisor_ready={st.get('supervisor_ready')} workers={st.get('workers_total')}"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last = repr(exc)
            time.sleep(2)
        raise RuntimeError(f"isolated server not ready in {timeout}s ({last})")

    def current_sha(self) -> str:
        try:
            return str(self._state(timeout=10).get("sha") or "")
        except (urllib.error.URLError, OSError, ValueError):
            return ""

    def submit(self, description: str, *, workspace_root: str = "",
               memory_mode: str = "forked", timeout_sec: int = 1800) -> str:
        body: dict = {
            "description": description,
            "memory_mode": memory_mode,
            "actor_id": "evolve-driver",
            "source": "evolve-driver",
            "timeout_sec": timeout_sec,
            "metadata": {"source": "evolve-driver", "delegation_role": "root"},
        }
        if workspace_root:
            body["workspace_root"] = str(workspace_root)
            body["workspace_mode"] = "external"
        created = _api(self.base_url, "POST", "/api/tasks", body, timeout=60)
        return str(created.get("task_id") or "")

    def wait_task(self, task_id: str, timeout: float = 2400) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = _api(self.base_url, "GET", "/api/tasks/" + urllib.parse.quote(task_id), timeout=30)
                if str(result.get("status") or "") in _FINAL_STATUSES:
                    return result
            except (urllib.error.URLError, OSError, ValueError):
                pass  # transient (e.g. server re-exec restart) — keep polling
            time.sleep(3)
        return {"status": "timeout"}

    def cancel_task(self, task_id: str) -> None:
        """Best-effort cancel of a still-running task (used when wait_task hits its own
        deadline) so the worker stops before the driver captures/continues."""
        try:
            _api(self.base_url, "POST",
                 "/api/tasks/" + urllib.parse.quote(task_id) + "/cancel", {}, timeout=30)
        except (urllib.error.URLError, OSError, ValueError):
            pass

    def wait_for_health(self, timeout: float = 180) -> bool:
        """Wait for /api/state to answer with supervisor ready again (after a
        self-evolution os.execvpe re-exec the same PID restarts on new code)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                st = self._state(timeout=5)
                if st.get("supervisor_ready") and int(st.get("workers_total") or 0) > 0:
                    return True
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(2)
        return False

    def wait_for_absorb(self, prev_sha: str, prev_absorbed: int, timeout: float = 1800,
                        idle_grace: float = 90) -> dict:
        """Between instances, wait for an absorbed self-evolution cycle: the server
        re-execs onto a new SHA and `absorbed_cycles_done` increments. Returns
        {absorbed, new_sha, cycles, reason}. When the LLM legitimately declines to
        promote (the common path), this returns absorbed=False EARLY — once the queue
        is idle, no post_task_evolution_request.json is pending, and no cycle absorbed
        within a short grace — instead of stalling the full timeout."""
        deadline = time.time() + timeout
        start = time.time()
        request_path = self.data_root / "state" / "post_task_evolution_request.json"
        while time.time() < deadline:
            cycles = absorbed_cycles_done(self.data_root)
            sha = self.current_sha()
            if cycles > prev_absorbed and sha and sha != prev_sha:
                self.wait_for_health(timeout=180)
                return {"absorbed": True, "new_sha": sha, "cycles": cycles, "reason": "absorbed"}
            if time.time() - start > idle_grace and cycles == prev_absorbed:
                try:
                    st = self._state(timeout=5)
                    idle = int(st.get("pending_count") or 0) == 0 and int(st.get("running_count") or 0) == 0
                except (urllib.error.URLError, OSError, ValueError):
                    idle = False
                if idle and not request_path.exists():
                    return {"absorbed": False, "new_sha": sha, "cycles": cycles, "reason": "no_promotion"}
            time.sleep(5)
        return {"absorbed": False, "new_sha": self.current_sha(),
                "cycles": absorbed_cycles_done(self.data_root), "reason": "timeout"}

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            pid = self.proc.pid
            terminate_process_tree(self.proc)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                kill_pid_tree(pid)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    def __enter__(self) -> "IsolatedServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
