"""Harbor installed-agent entrypoint for evaluating full Ouroboros in Terminal-Bench.

This adapter intentionally does not translate Ouroboros decisions into shell
commands. Harbor starts a task container, this class installs Ouroboros inside
that container, starts the normal Ouroboros server/supervisor, and submits the
Terminal-Bench instruction as an external workspace task rooted at ``/app``.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path
from typing import Any

from devtools.benchmarks.common.manifests import repo_provenance, write_json

try:  # Harbor is an optional benchmark dependency.
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except Exception:  # pragma: no cover - exercised when Harbor is absent.
    BaseInstalledAgent = object  # type: ignore[assignment]
    BaseEnvironment = Any  # type: ignore[assignment]
    AgentContext = Any  # type: ignore[assignment]


_CONTAINER_SRC = "/opt/ouroboros-src"
_CONTAINER_VENV = "/opt/ouroboros-venv"
# Optional host-mounted pip wheel cache (mount a host dir here via Harbor --mounts to make the
# per-trial Ouroboros pip install offline-fast and resilient to mirror/network drops). Safe by
# default: if nothing is mounted at this path it is just an ephemeral in-container cache dir, so
# behavior is unchanged. pip keys cached wheels by (name, version, python-tag, platform-tag), so a
# single shared cache is correct across heterogeneous task images (py3.11/3.12, different glibc) and
# concurrency-safe (atomic-rename writes of identical content). See run_tb.py OBO_TB_PIP_CACHE.
_CONTAINER_PIP_CACHE = "/opt/ouro-pip-cache"
_CONTAINER_DATA = "/logs/agent/ouroboros-data"
_CONTAINER_WORKSPACE = "/app"
_SERVER_URL = "http://127.0.0.1:8765"
_CONTAINER_SECRET_OPT_IN = "OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS"
_SECRET_ENV_KEYS = frozenset({
    "ANTHROPIC_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_PASSWORD",
    "GIGACHAT_USER",
    "OPENAI_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENROUTER_API_KEY",
})
log = logging.getLogger(__name__)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_host_settings_path() -> Path:
    return Path(os.environ.get("OUROBOROS_SETTINGS_PATH") or _workspace_root() / "data" / "settings.json")


def _json_load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _secret_shaped_source_name(name: str) -> bool:
    lower = name.lower()
    secret_extensions = (".json", ".yaml", ".yml", ".toml", ".ini", ".txt")
    if lower.startswith(".env") or lower.endswith(".env") or ".env." in lower:
        return True
    if lower.endswith((".key", ".pem", ".pfx", ".p12")):
        return True
    if lower in {".git-credentials", ".netrc", ".npmrc", ".pypirc", "id_ed25519", "id_rsa"}:
        return True
    if lower in {"credentials.json", "token.json", "secrets.json", "secrets.yaml", "secrets.yml", "secrets.toml"}:
        return True
    if any(token in lower for token in ("secret", "credential", "token", "service-account")):
        return lower.endswith(secret_extensions)
    return False


def _copy_clean_source(source: Path, target: Path) -> None:
    excluded_dirs = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "data",
        "data_evaluated",
        "dist",
        "node-standalone",
        "node_modules",
        "python-standalone",
        "venv",
    }
    excluded_suffixes = {".pyc", ".pyo"}
    excluded_names = {
        ".DS_Store",
        ".env.example",
        ".release_notes.md",
        "repo.bundle",
        "repo_bundle_manifest.json",
    }

    def ignore(_: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in excluded_dirs or name in excluded_names:
                ignored.add(name)
                continue
            if _secret_shaped_source_name(name):
                ignored.add(name)
                continue
            if any(name.endswith(suffix) for suffix in excluded_suffixes):
                ignored.add(name)
        return ignored

    shutil.copytree(source, target, ignore=ignore, symlinks=True)


def _tree_digest(root: Path) -> dict[str, Any]:
    digest = sha256()
    file_count = 0
    symlink_count = 0
    byte_count = 0
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        rel = str(path.relative_to(root)).replace(os.sep, "/")
        if path.is_symlink():
            symlink_count += 1
            digest.update(f"L\0{rel}\0{os.readlink(path)}\0".encode("utf-8", errors="replace"))
            continue
        if not path.is_file():
            continue
        file_count += 1
        size = path.stat().st_size
        byte_count += int(size)
        digest.update(f"F\0{rel}\0{size}\0".encode("utf-8", errors="replace"))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "sha256": digest.hexdigest(),
        "files": file_count,
        "symlinks": symlink_count,
        "bytes": byte_count,
    }


def _source_copy_provenance(source: Path, copied_tree: Path | None = None) -> dict[str, Any]:
    provenance = repo_provenance(source)
    provenance["copy_policy"] = {
        "schema": "ouroboros.benchmark.source_copy.v1",
        "git_dir_copy_allowed": False,
        "runtime_data_copy_allowed": False,
        "secret_shaped_file_copy_allowed": False,
        "copy_target": _CONTAINER_SRC,
    }
    if copied_tree is not None:
        provenance["copied_tree"] = _tree_digest(copied_tree)
    return provenance


class OuroborosTerminalBenchAgent(BaseInstalledAgent):
    """Install and run full Ouroboros inside the Terminal-Bench task container."""

    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str = "ouroboros-inside",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        workspace_dir = str(kwargs.pop("workspace_dir", _CONTAINER_WORKSPACE))
        host_settings_path = str(kwargs.pop("host_settings_path", ""))
        install_timeout_sec = int(kwargs.pop("install_timeout_sec", 900))
        server_start_timeout_sec = int(kwargs.pop("server_start_timeout_sec", 180))
        task_timeout_sec = kwargs.pop("task_timeout_sec", None)
        openrouter_min_credit_usd = float(kwargs.pop("openrouter_min_credit_usd", os.environ.get("OUROBOROS_BENCH_OPENROUTER_MIN_CREDIT_USD", 5.0)))
        # plan_task needs >=2 workers (planning scouts run as pooled subagents);
        # 1 worker forced the capacity-degraded inline fallback on every run.
        max_workers = int(kwargs.pop("max_workers", 4))  # v6.55.0: 3-4 subagent slots (root takes one); 10 would blow container memory (full python proc per worker)
        runtime_mode = str(kwargs.pop("runtime_mode", "pro"))
        review_enforcement = str(kwargs.pop("review_enforcement", "blocking"))
        # Safety mode: configurable (full|light|off). Default light keeps the v6.55.0
        # scaffold behavior (LLM safety for integration tools only); off disables the
        # LLM safety pass entirely for a fully-disposable jail. Deterministic guards
        # are unaffected either way.
        safety_mode = str(kwargs.pop("safety_mode", "light")).strip().lower()
        if safety_mode not in ("full", "light", "off"):
            safety_mode = "light"
        task_review_mode = str(kwargs.pop("task_review_mode", "required"))
        disable_agent_web = str(kwargs.pop("disable_agent_web", "true")).strip().lower() not in (
            "0", "false", "no", "off", "",
        )
        ouroboros_model = str(kwargs.pop("ouroboros_model", ""))
        ouroboros_light_model = str(kwargs.pop("ouroboros_light_model", "google/gemini-3.5-flash"))
        leave_server_running_for_verifier = bool(kwargs.pop("leave_server_running_for_verifier", True))
        try:
            super().__init__(*args, logs_dir=logs_dir, model_name=model_name, **kwargs)
        except TypeError:
            super().__init__()
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
        self.workspace_dir = workspace_dir
        self.host_settings_path = Path(
            host_settings_path
            or os.environ.get("OUROBOROS_SETTINGS_PATH")
            or _default_host_settings_path()
        ).expanduser()
        self.install_timeout_sec = int(install_timeout_sec)
        self.server_start_timeout_sec = int(server_start_timeout_sec)
        self.task_timeout_sec = (
            int(task_timeout_sec)
            if task_timeout_sec is not None and int(task_timeout_sec) > 0
            else None
        )
        self.openrouter_min_credit_usd = float(openrouter_min_credit_usd)
        self.max_workers = int(max_workers)
        self.runtime_mode = runtime_mode
        self.review_enforcement = review_enforcement
        self.safety_mode = safety_mode
        self.task_review_mode = task_review_mode
        self.disable_agent_web = bool(disable_agent_web)
        self.ouroboros_model = ouroboros_model
        self.ouroboros_light_model = ouroboros_light_model
        self.leave_server_running_for_verifier = bool(leave_server_running_for_verifier)
        self._run_summary: dict[str, Any] = {}
        # Monotonic timestamp of run() start, so the deadline we hand the agent accounts for the
        # install/server time already consumed inside Harbor's external per-task wall-clock cap.
        self._run_started_monotonic: float | None = None

    @staticmethod
    def name() -> str:
        return "Ouroboros Installed"

    def version(self) -> str | None:
        return "0.2.0"

    def _host_settings(self) -> dict[str, Any]:
        return _json_load(self.host_settings_path)

    def _container_secret_injection_allowed(self, settings: dict[str, Any]) -> bool:
        value = os.environ.get(_CONTAINER_SECRET_OPT_IN)
        if value is None:
            value = settings.get(_CONTAINER_SECRET_OPT_IN)
        return str(value or "").strip().lower() in {"1", "true", "yes", "allow"}

    def _available_host_secret_keys(self, settings: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        for key in sorted(_SECRET_ENV_KEYS):
            if str(os.environ.get(key) or settings.get(key) or "").strip():
                keys.append(key)
        return keys

    def _openrouter_credit_preflight(self, settings: dict[str, Any]) -> None:
        key = str(os.environ.get("OPENROUTER_API_KEY") or settings.get("OPENROUTER_API_KEY") or "").strip()
        if not key:
            return
        threshold = max(0.0, float(self.openrouter_min_credit_usd or 0.0))
        if threshold <= 0:
            return
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError("OpenRouter credit preflight failed: token is unauthorized") from exc
            # Do not block on transient credit endpoint failures; model calls will
            # still surface provider errors, and publishable runs preserve logs.
            (self.logs_dir / "openrouter-credit-preflight.json").write_text(
                json.dumps({"ok": False, "status": exc.code, "warning": "non-blocking"}, indent=2),
                encoding="utf-8",
            )
            return
        except Exception as exc:
            (self.logs_dir / "openrouter-credit-preflight.json").write_text(
                json.dumps({"ok": False, "error": type(exc).__name__, "warning": "non-blocking"}, indent=2),
                encoding="utf-8",
            )
            return
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        total = float(data.get("total_credits") or data.get("credits") or data.get("credit_limit") or 0.0)
        used = float(data.get("total_usage") or data.get("usage") or 0.0)
        remaining = total - used if total else float(data.get("remaining_credits") or data.get("remaining") or 0.0)
        (self.logs_dir / "openrouter-credit-preflight.json").write_text(
            json.dumps({"ok": True, "remaining_usd": remaining, "threshold_usd": threshold}, indent=2),
            encoding="utf-8",
        )
        if remaining < threshold:
            raise RuntimeError(
                f"OpenRouter credit preflight failed: remaining ${remaining:.2f} below threshold ${threshold:.2f}"
            )

    def _enforce_container_secret_policy(self, env: dict[str, str]) -> None:
        settings = self._host_settings()
        blocked = self._available_host_secret_keys(settings)
        if blocked and not self._container_secret_injection_allowed(settings):
            names = ", ".join(blocked)
            raise RuntimeError(
                "Terminal-Bench installed-container mode refuses to inject long-lived provider "
                f"credentials into task containers by default ({names}). Use a host-mediated LLM "
                "bridge when available, or set OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1 only for "
                "trusted local smoke runs where the task container and logs are under operator control."
            )
        if not blocked and not any(key in env for key in _SECRET_ENV_KEYS):
            return

    def _container_env(self) -> dict[str, str]:
        settings = self._host_settings()
        allow_secrets = self._container_secret_injection_allowed(settings)
        keys = [
            "OPENAI_BASE_URL",
            "OPENAI_COMPATIBLE_BASE_URL",
            "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
            "GIGACHAT_SCOPE",
            "GIGACHAT_BASE_URL",
            "GIGACHAT_VERIFY_SSL_CERTS",
            "GIGACHAT_PROFANITY_CHECK",
            "OUROBOROS_MODEL",
            "OUROBOROS_MODEL_HEAVY",  # v6.39 slot rename (legacy OUROBOROS_MODEL_CODE -> _HEAVY)
            "OUROBOROS_MODEL_LIGHT",
            # OUROBOROS_MODEL_FALLBACK is deliberately NOT forwarded: the
            # benchmark metric must stay single-model (a host-configured
            # fallback would silently contaminate the measurement).
            "OUROBOROS_WEBSEARCH_MODEL",
            "OUROBOROS_REVIEW_MODELS",
            "OUROBOROS_SCOPE_REVIEW_MODELS",
            "OUROBOROS_SCOPE_REVIEW_MODEL",
            "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
            "CLAUDE_CODE_MODEL",
            "OUROBOROS_EFFORT_TASK",
            "OUROBOROS_EFFORT_REVIEW",
            "OUROBOROS_EFFORT_SCOPE_REVIEW",
            "OUROBOROS_EFFORT_DEEP_SELF_REVIEW",
            "OUROBOROS_RETURN_REASONING",
            "TOTAL_BUDGET",
            "OUROBOROS_PER_TASK_COST_USD",
            "OUROBOROS_SOFT_TIMEOUT_SEC",
            "OUROBOROS_HARD_TIMEOUT_SEC",
            "OUROBOROS_TOOL_TIMEOUT_SEC",
        ]
        if allow_secrets:
            keys.extend(sorted(_SECRET_ENV_KEYS))
        env: dict[str, str] = {}
        for key in keys:
            value = os.environ.get(key)
            if value is None:
                value = settings.get(key)
            if value not in (None, ""):
                env[key] = str(value)

        if self.ouroboros_model:
            env["OUROBOROS_MODEL"] = self.ouroboros_model
            env["OUROBOROS_MODEL_HEAVY"] = self.ouroboros_model
        if self.ouroboros_light_model:
            env["OUROBOROS_MODEL_LIGHT"] = self.ouroboros_light_model

        # Pin the fallback to the EFFECTIVE main model: the container has no
        # settings.json, so leaving the key unset resurrects the
        # SETTINGS_DEFAULTS fallback (a DIFFERENT model) and contaminates the
        # single-model metric; an empty-string env value is skipped by
        # load_settings the same way. Resolution order: explicit kwarg ->
        # forwarded host main model -> packaged default main model.
        fallback_pin = self.ouroboros_model or env.get("OUROBOROS_MODEL", "")
        if not fallback_pin:
            try:
                from ouroboros.config import SETTINGS_DEFAULTS as _DEFAULTS

                fallback_pin = str(_DEFAULTS.get("OUROBOROS_MODEL") or "")
            except Exception:
                fallback_pin = ""
        if fallback_pin:
            # Pin BOTH the legacy singular AND the current plural key. config.parse_fallback_chain
            # reads OUROBOROS_MODEL_FALLBACKS (plural) BEFORE the legacy singular, and the container's
            # SETTINGS_DEFAULTS plural is a DIFFERENT model (the shipped cross-model chain). Leaving
            # the plural unset lets that default shadow the singular pin and contaminate the
            # single-model metric, so we pin the plural to the main model too.
            env["OUROBOROS_MODEL_FALLBACK"] = fallback_pin
            env["OUROBOROS_MODEL_FALLBACKS"] = fallback_pin

        env.update(
            {
                "OUROBOROS_REPO_DIR": _CONTAINER_SRC,
                "OUROBOROS_DATA_DIR": _CONTAINER_DATA,
                "OUROBOROS_SETTINGS_PATH": f"{_CONTAINER_DATA}/settings.json",
                "OUROBOROS_PID_FILE": "/logs/agent/ouroboros.pid",
                "OUROBOROS_PORT_FILE": f"{_CONTAINER_DATA}/state/server_port",
                "OUROBOROS_SERVER_HOST": "127.0.0.1",
                "OUROBOROS_SERVER_PORT": "8765",
                "OUROBOROS_WORKER_START_METHOD": "spawn",
                "OUROBOROS_RUNTIME_MODE": self.runtime_mode,
                "OUROBOROS_REVIEW_ENFORCEMENT": self.review_enforcement,
                "OUROBOROS_TASK_REVIEW_MODE": self.task_review_mode,
                "OUROBOROS_MAX_WORKERS": str(self.max_workers),
                # v6.55.0: the container is an isolated jail — the LLM safety layer
                # adds cost/latency without protecting anything the deterministic
                # guards don't (34% of all LLM calls in the k=5 run); light keeps
                # the LLM check for integration tools only (Owner decision #14).
                # Configurable via the safety_mode agent-kwarg (full|light|off).
                "OUROBOROS_SAFETY_MODE": self.safety_mode,
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env

    async def _append_log(self, environment: BaseEnvironment, message: str) -> None:
        safe = json.dumps(message)
        await environment.exec(
            command=f"mkdir -p /logs/agent && python3 - <<'PY'\n"
            "from pathlib import Path\n"
            f"Path('/logs/agent/ouroboros-install.log').open('a', encoding='utf-8').write({safe} + '\\n')\n"
            "PY",
            user="root",
        )

    async def _upload_source(self, environment: BaseEnvironment) -> None:
        source = _repo_root()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ouroboros-tb-src-") as tmp:
            clean = Path(tmp) / "repo"
            _copy_clean_source(source, clean)
            write_json(self.logs_dir / "source-provenance.json", _source_copy_provenance(source, clean))
            await environment.exec(
                command=f"rm -rf {_CONTAINER_SRC} && mkdir -p {_CONTAINER_SRC}",
                user="root",
            )
            await environment.upload_dir(clean, _CONTAINER_SRC)

    async def install(self, environment: BaseEnvironment) -> None:
        started = time.monotonic()
        await self._append_log(environment, "install: starting source upload")
        await self._upload_source(environment)
        await self._append_log(environment, "install: source uploaded")

        install_cmd = textwrap.dedent(
            f"""
            set -euo pipefail
            mkdir -p /logs/agent {_CONTAINER_DATA}/logs {_CONTAINER_DATA}/state
            {{
              echo "install: installing system dependencies"
              if command -v apt-get >/dev/null 2>&1; then
                export DEBIAN_FRONTEND=noninteractive
                apt-get update
                apt-get install -y --no-install-recommends git curl bash ca-certificates procps python3 python3-venv python3-pip
              elif command -v apk >/dev/null 2>&1; then
                apk add --no-cache git curl bash ca-certificates procps python3 py3-pip py3-virtualenv
              elif command -v yum >/dev/null 2>&1; then
                yum install -y git curl bash ca-certificates procps python3 python3-pip
              else
                echo "install: no known package manager; assuming required tools already exist"
              fi

              PYTHON_BIN="$(command -v python3 || command -v python)"
              PY_OK="$("$PYTHON_BIN" - <<'PY'
import sys
print(1 if sys.version_info >= (3, 10) else 0)
PY
)"
              if [ "$PY_OK" != "1" ]; then
                echo "install: system Python is too old; installing Python 3.12 with uv"
                curl -LsSf https://astral.sh/uv/install.sh | sh
                export PATH="$HOME/.local/bin:$PATH"
                uv python install 3.12
                PYTHON_BIN="$(uv python find 3.12)"
              fi
              echo "install: using $PYTHON_BIN"
              "$PYTHON_BIN" -m venv {_CONTAINER_VENV} || {{
                "$PYTHON_BIN" -m pip install --break-system-packages --user virtualenv || "$PYTHON_BIN" -m pip install --user virtualenv
                "$PYTHON_BIN" -m virtualenv {_CONTAINER_VENV}
              }}

              . {_CONTAINER_VENV}/bin/activate
              export PIP_CACHE_DIR={_CONTAINER_PIP_CACHE}
              mkdir -p "$PIP_CACHE_DIR" 2>/dev/null || true
              python -m pip install --upgrade pip setuptools wheel
              python -m pip install -r {_CONTAINER_SRC}/requirements.txt || {{
                echo "install: requirements install failed; retrying without optional tree-sitter code-intel deps (lazy runtime import, degrades gracefully)"
                grep -ivE 'tree[-_]sitter' {_CONTAINER_SRC}/requirements.txt > /tmp/ouro_reqs_no_treesitter.txt
                python -m pip install -r /tmp/ouro_reqs_no_treesitter.txt
              }}
              python -m pip install -e {_CONTAINER_SRC} --no-deps
              # ffmpeg in the AGENT prefix (v6.56.0, P0-1): task images rarely ship
              # ffmpeg, so extract_video_frames was dead in TB tasks. The wheel binary
              # is found by media._resolve_ffmpeg via imageio_ffmpeg.get_ffmpeg_exe();
              # a mirror hiccup degrades gracefully (typed UNAVAILABLE + cv2 hint).
              python -m pip install imageio-ffmpeg || echo "install: imageio-ffmpeg failed (extract_video_frames degrades to the cv2 workaround)"
              chmod -R a+rX {_CONTAINER_SRC} {_CONTAINER_VENV} /logs/agent
              {_CONTAINER_VENV}/bin/python -c 'import importlib.metadata; print("ouroboros", importlib.metadata.version("ouroboros"))'
              echo "install: complete"
            }} 2>&1 | tee -a /logs/agent/ouroboros-install.log
            """
        ).strip()
        await self.exec_as_root(
            environment,
            command=install_cmd,
            timeout_sec=self.install_timeout_sec,
        )
        elapsed = time.monotonic() - started
        await self._append_log(environment, f"install: elapsed_sec={elapsed:.1f}")

    async def _ensure_workspace_git_root(self, environment: BaseEnvironment) -> None:
        workspace_dir = shlex.quote(self.workspace_dir)
        command = textwrap.dedent(
            f"""
            set -euo pipefail
            workspace_dir={workspace_dir}
            cd "$workspace_dir"
            if git rev-parse --show-toplevel >/tmp/ouroboros-git-root 2>/dev/null; then
              root="$(cat /tmp/ouroboros-git-root)"
              if [ "$root" != "$workspace_dir" ]; then
                echo "workspace git root is $root, expected $workspace_dir" >&2
                exit 2
              fi
            else
              git init
              git config user.email ouroboros-bench@example.invalid
              git config user.name "Ouroboros Bench"
            fi
            """
        ).strip()
        result = await environment.exec(command=command, cwd=self.workspace_dir, timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(
                f"failed to prepare {self.workspace_dir} as git workspace: "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )

    async def _resolve_workspace_dir(self, environment: BaseEnvironment) -> None:
        """Use /app when present, but support tasks whose Dockerfile uses /workspace."""
        requested = self.workspace_dir
        quoted_requested = shlex.quote(requested)
        result = await environment.exec(command=f"test -d {quoted_requested}", timeout_sec=10)
        if result.return_code == 0:
            return
        if requested == _CONTAINER_WORKSPACE:
            fallback = await environment.exec(command="test -d /workspace", timeout_sec=10)
            if fallback.return_code == 0:
                self.workspace_dir = "/workspace"
                await self._append_log(environment, "workspace: /app missing, using /workspace")
                return
        create = await environment.exec(command=f"mkdir -p {quoted_requested}", user="root", timeout_sec=10)
        if create.return_code != 0:
            raise RuntimeError(
                f"failed to create workspace {requested}: stdout={create.stdout!r} stderr={create.stderr!r}"
            )
        await self._append_log(environment, f"workspace: created {requested}")

    async def _start_server(self, environment: BaseEnvironment, env: dict[str, str]) -> None:
        start_cmd = textwrap.dedent(
            f"""
            set -euo pipefail
            mkdir -p {_CONTAINER_DATA}/logs {_CONTAINER_DATA}/state /logs/agent
            rm -f /logs/agent/ouroboros.pid
            cd {_CONTAINER_SRC}
            nohup {_CONTAINER_VENV}/bin/python server.py --host 127.0.0.1 --port 8765 \
              > /logs/agent/ouroboros-server.stdout.log \
              2> /logs/agent/ouroboros-server.stderr.log &
            echo "$!" > /logs/agent/ouroboros.pid
            """
        ).strip()
        result = await environment.exec(command=start_cmd, env=env, timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(f"failed to start Ouroboros server: {result.stdout}\n{result.stderr}")

        wait_cmd = textwrap.dedent(
            f"""
            {_CONTAINER_VENV}/bin/python - <<'PY'
            import json
            import pathlib
            import sys
            import time
            import urllib.request

            deadline = time.time() + {self.server_start_timeout_sec}
            last_error = ""
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen("{_SERVER_URL}/api/state", timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    pathlib.Path("/logs/agent/ouroboros-state.json").write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if data.get("supervisor_ready"):
                        print(json.dumps({{"ready": True, "state": data}}, ensure_ascii=False))
                        sys.exit(0)
                    last_error = "server responded but supervisor_ready=false"
                except Exception as exc:
                    last_error = repr(exc)
                time.sleep(2)
            print(json.dumps({{"ready": False, "error": last_error}}, ensure_ascii=False))
            sys.exit(1)
            PY
            """
        ).strip()
        result = await environment.exec(command=wait_cmd, env=env, timeout_sec=self.server_start_timeout_sec + 20)
        if result.return_code != 0:
            raise RuntimeError(f"Ouroboros server did not become ready: {result.stdout}\n{result.stderr}")

    async def _network_preflight(self, environment: BaseEnvironment, env: dict[str, str]) -> None:
        provider_url = ""
        provider_name = ""
        if env.get("OPENROUTER_API_KEY"):
            provider_url = "https://openrouter.ai/api/v1/models"
            provider_name = "openrouter"
        elif env.get("OPENAI_COMPATIBLE_API_KEY"):
            base_url = str(env.get("OPENAI_COMPATIBLE_BASE_URL") or env.get("OPENAI_BASE_URL") or "").strip()
            provider_name = "openai_compatible"
            if base_url:
                provider_url = base_url.rstrip("/") + "/models"
            else:
                (self.logs_dir / "network-preflight.txt").write_text(
                    "openai_compatible_preflight_error missing OPENAI_COMPATIBLE_BASE_URL\n",
                    encoding="utf-8",
                )
                raise RuntimeError("OPENAI_COMPATIBLE_API_KEY requires OPENAI_COMPATIBLE_BASE_URL for container preflight")
        elif env.get("OPENAI_API_KEY"):
            provider_url = "https://api.openai.com/v1/models"
            provider_name = "openai"
        elif env.get("ANTHROPIC_API_KEY"):
            provider_url = "https://api.anthropic.com/v1/models"
            provider_name = "anthropic"
        elif env.get("CLOUDRU_FOUNDATION_MODELS_API_KEY"):
            provider_url = (env.get("CLOUDRU_FOUNDATION_MODELS_BASE_URL") or "https://foundation-models.api.cloud.ru/v1").rstrip("/") + "/models"
            provider_name = "cloudru"
        elif env.get("GIGACHAT_CREDENTIALS") or (env.get("GIGACHAT_USER") and env.get("GIGACHAT_PASSWORD")):
            provider_url = (env.get("GIGACHAT_BASE_URL") or "https://gigachat.devices.sberbank.ru/api/v1").rstrip("/") + "/models"
            provider_name = "gigachat"
        if not provider_url:
            (self.logs_dir / "network-preflight.txt").write_text(
                "provider preflight skipped: no provider API key was injected; "
                "Ouroboros runtime will surface provider configuration errors.\n",
                encoding="utf-8",
            )
            return
        command = textwrap.dedent(
            f"""
            python3 - <<'PY'
            import sys
            import urllib.error
            import urllib.request
            req = urllib.request.Request({provider_url!r}, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    print({provider_name!r} + "_preflight_status", resp.status)
                    sys.exit(0 if 200 <= resp.status < 500 else 1)
            except urllib.error.HTTPError as exc:
                print({provider_name!r} + "_preflight_status", exc.code)
                sys.exit(0 if 200 <= exc.code < 500 else 1)
            except Exception as exc:
                print({provider_name!r} + "_preflight_error", type(exc).__name__)
                sys.exit(1)
            PY
            """
        ).strip()
        result = await environment.exec(command=command, timeout_sec=20)
        (self.logs_dir / "network-preflight.txt").write_text(
            f"stdout:\n{result.stdout or ''}\nstderr:\n{result.stderr or ''}\nreturn_code={result.return_code}\n",
            encoding="utf-8",
        )
        if result.return_code != 0:
            raise RuntimeError(f"container cannot reach configured provider endpoint ({provider_name})")

    def _disabled_tools(self) -> list[str]:
        # Reward-hacking guard: faithful TB2.1 runs give the task FULL container network
        # (every task.toml declares allow_internet=true; tasks like build-cython-ext/caffe-cifar-10
        # require `git clone`), so we must NOT block shell egress. We only withhold the agent's OWN
        # LLM-powered web/search/browser/VLM tools (which a reference shell agent wouldn't have) via
        # the declarative `disabled_tools` tool-policy. This leaves allowed_resources at its permissive
        # default (network/git/pip available) and never trips the web<->network cross-implication in
        # the registry resource gate. (Previously this set allowed_resources.network=false, which
        # wrongly blocked `git clone` even though the container had working network.)
        # The web group mirrors the registry's `_WEB_TOOLS` set (web_search/
        # browse_page/browser_action/youtube_transcript — the transcript tool joined
        # `_WEB_TOOLS` in v6.52.1 and the adapter's list had silently drifted until
        # v6.55.0; a sync test now pins the mirror). On top of it, web-off runs also
        # withhold the DELEGATED-vision tools (analyze_screenshot/vlm_query): they
        # route through an LLM/VLM lookup a reference shell agent would not have.
        # `view_image` is intentionally NOT disabled: it is a LOCAL image-to-model
        # tool registered OUTSIDE `_WEB_TOOLS` (it injects a local file into the
        # agent's own model context, no web/second-model call), so local-image tasks
        # (e.g. financial-document/code-from-image) keep a legitimate vision
        # affordance a reference agent could also have.
        # v6.55.0: claude_code_edit is disabled in EVERY bench run regardless of the
        # web gate — benches measure Ouroboros as a single-model harness; the embedded
        # Claude-Code delegate is a separate future experiment.
        disabled = ["claude_code_edit"]
        if getattr(self, "disable_agent_web", True):
            disabled = list(self._WEB_TOOLS_MIRROR) + list(self._DELEGATED_VISION_TOOLS) + disabled
        return disabled

    async def _run_ouroboros_task(self, environment: BaseEnvironment, env: dict[str, str]) -> dict[str, Any]:
        workspace_root = json.dumps(self.workspace_dir)
        disabled_tools_line = f'"disabled_tools": {json.dumps(self._disabled_tools())},'

        runner = textwrap.dedent(
            f"""
            import json
            import os
            import pathlib
            import sys
            import time
            import urllib.parse
            import urllib.request

            instruction = pathlib.Path("/logs/agent/instruction.txt").read_text(encoding="utf-8")
            started = time.time()
            run_log = pathlib.Path("/logs/agent/ouroboros-run.jsonl")
            stderr_log = pathlib.Path("/logs/agent/ouroboros-run.stderr.log")
            task_id_path = pathlib.Path("/logs/agent/ouroboros-current-task-id.txt")

            def api(method, path, body=None, timeout=30):
                data = None
                headers = {{"Accept": "application/json"}}
                if body is not None:
                    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                    headers["Content-Type"] = "application/json"
                req = urllib.request.Request("{_SERVER_URL}" + path, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw.strip() else {{}}

            def emit(event):
                with run_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\\n")

            task_body = {{
                "description": instruction,
                "workspace_root": {workspace_root},
                "workspace_mode": "external",
                "memory_mode": "empty",
                "service_teardown": "keep",
                "actor_id": "harbor-terminal-bench",
                "source": "terminal-bench",
                "metadata": {{"source": "terminal-bench", "delegation_role": "root"}},
                {disabled_tools_line}
            }}
            task_timeout = {int(self._effective_task_timeout_sec())}
            if task_timeout > 0:
                task_body["timeout_sec"] = task_timeout
            created = api("POST", "/api/tasks", task_body)
            task_id = str(created.get("task_id") or "")
            if not task_id:
                stderr_log.write_text(f"task creation did not return task_id: {{created!r}}\\n", encoding="utf-8")
                print(json.dumps({{"return_code": 1, "elapsed_sec": round(time.time() - started, 3), "status": "create_failed"}}))
                sys.exit(1)
            task_id_path.write_text(task_id, encoding="utf-8")
            emit({{"type": "task_created", "task_id": task_id, "data": created}})

            latest = {{}}
            seen_events = set()
            final_statuses = {{"completed", "failed", "cancelled", "rejected_duplicate"}}
            while True:
                result = api("GET", "/api/tasks/" + urllib.parse.quote(task_id), timeout=30)
                for event in result.get("events") or []:
                    key = (str(event.get("type") or ""), str(event.get("ts") or event.get("seq") or ""))
                    if key in seen_events:
                        continue
                    seen_events.add(key)
                    emit({{"type": "task_event", "task_id": task_id, "data": event}})
                status = str(result.get("status") or "")
                if status in final_statuses:
                    latest = result
                    pathlib.Path("/logs/agent/ouroboros-task-result.json").write_text(
                        json.dumps(latest, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    emit({{"type": "final", "task_id": task_id, "result": latest}})
                    break
                time.sleep(2)

            status = str(latest.get("status") or "")
            reason_code = str(latest.get("reason_code") or "")
            axes = latest.get("outcome_axes") if isinstance(latest.get("outcome_axes"), dict) else {{}}
            execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {{}}
            infra_failed = (
                reason_code == "llm_api_error"
                or str(execution.get("status") or "") == "infra_failed"
                or str(execution.get("reason_code") or "") == "llm_api_error"
            )
            summary = {{
                "return_code": 2 if infra_failed else 0,
                "task_status_code": 0 if status == "completed" else 1,
                "elapsed_sec": round(time.time() - started, 3),
                "task_id": latest.get("task_id") or latest.get("id"),
                "status": status,
                "reason_code": reason_code,
                "infra_failed": infra_failed,
                "cost_usd": latest.get("cost_usd"),
                "prompt_tokens": latest.get("prompt_tokens"),
                "completion_tokens": latest.get("completion_tokens"),
                "total_rounds": latest.get("total_rounds"),
            }}
            pathlib.Path("/logs/agent/ouroboros-run-summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False))
            sys.exit(2 if infra_failed else 0)
            """
        ).strip()
        command = "cat > /tmp/run_ouroboros_task.py <<'PY'\n" + runner + "\nPY\n" + (
            f"{_CONTAINER_VENV}/bin/python /tmp/run_ouroboros_task.py"
        )
        result = await environment.exec(
            command=command,
            env=env,
            cwd=self.workspace_dir,
            timeout_sec=(self.task_timeout_sec + 60 if self.task_timeout_sec is not None else None),
        )
        parsed: dict[str, Any] | None = None
        try:
            candidate = json.loads((result.stdout or "").strip().splitlines()[-1])
            if isinstance(candidate, dict):
                parsed = candidate
        except Exception:
            parsed = None
        # The runner exits 2 (not 0) purely to SIGNAL a terminal `infra_failed` result — the task
        # still reached a terminal /api/tasks state (status completed/failed). That is a real terminal
        # outcome, NOT a Harbor wall-clock interruption, so treat it as a returned result (caller sets
        # reached_terminal_result=True and the captured summary is NOT mislabeled
        # captured_after_cancellation). Only a nonzero exit that produced NO terminal summary (e.g. a
        # genuine runner crash / create_failed) is a real failure to raise on.
        if parsed is not None and str(parsed.get("status") or "") in ("completed", "failed"):
            return parsed
        if result.return_code != 0:
            raise RuntimeError(f"Ouroboros task runner failed: {result.stdout}\n{result.stderr}")
        return parsed if parsed is not None else {"raw_stdout": result.stdout or "", "raw_stderr": result.stderr or ""}

    async def _emit_trajectory(self, environment: BaseEnvironment, env: dict[str, str]) -> None:
        """Build /logs/agent/trajectory.json in-container from the trial logs.

        Uses the stdlib-only builder shipped with the uploaded source tree, so
        the exact same mapping serves live runs and the offline backfill
        converter (build_atif_trajectories.py).
        """
        builder = f"{_CONTAINER_SRC}/devtools/benchmarks/terminal_bench/atif.py"
        model = json.dumps(self.ouroboros_model or "")
        result = await environment.exec(
            command=(
                f"{_CONTAINER_VENV}/bin/python {builder} /logs/agent --model {model}"
            ),
            env=env,
            timeout_sec=120,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"trajectory builder exited {result.return_code}: {result.stderr}"
            )

    async def _stop_server(self, environment: BaseEnvironment) -> None:
        await environment.exec(
            command=(
                "if [ -s /logs/agent/ouroboros-current-task-id.txt ]; then "
                "TASK_ID=$(cat /logs/agent/ouroboros-current-task-id.txt); "
                "export TASK_ID; "
                f"{_CONTAINER_VENV}/bin/python - <<'PY' || true\n"
                "import os, urllib.parse, urllib.request\n"
                "task_id = os.environ.get('TASK_ID', '')\n"
                "if task_id:\n"
                f"    urllib.request.urlopen(urllib.request.Request('{_SERVER_URL}/api/tasks/' + urllib.parse.quote(task_id) + '/cancel', data=b'{{}}', method='POST'), timeout=5).read()\n"
                "PY\n"
                "fi; "
                "if [ -f /logs/agent/ouroboros.pid ]; then "
                "kill $(cat /logs/agent/ouroboros.pid) 2>/dev/null || true; "
                "fi; "
                "pkill -TERM -f '/opt/ouroboros-src|/opt/ouroboros-venv/bin/ouroboros' 2>/dev/null || true"
            ),
            timeout_sec=10,
        )

    async def _capture_current_task_summary(self, environment: BaseEnvironment, interrupted: bool = True) -> None:
        """Persist best-effort task state. ``interrupted`` records whether Harbor cancelled
        agent.run mid-exec (True) vs a routine post-terminal snapshot (False); it is written as
        ``captured_after_cancellation`` so the disclosure ledger can tell a real cancellation
        apart from a normal terminal finish."""
        captured_after_cancellation = "True" if interrupted else "False"
        command = textwrap.dedent(
            f"""
            if [ ! -s /logs/agent/ouroboros-current-task-id.txt ]; then
              exit 0
            fi
            {_CONTAINER_VENV}/bin/python - <<'PY'
            import json
            import pathlib
            import time
            import urllib.parse
            import urllib.request

            task_id = pathlib.Path("/logs/agent/ouroboros-current-task-id.txt").read_text(encoding="utf-8").strip()
            if not task_id:
                raise SystemExit(0)
            try:
                with urllib.request.urlopen("{_SERVER_URL}/api/tasks/" + urllib.parse.quote(task_id), timeout=10) as resp:
                    latest = json.loads(resp.read().decode("utf-8", errors="replace"))
            except Exception as exc:
                pathlib.Path("/logs/agent/ouroboros-run.stderr.log").open("a", encoding="utf-8").write(
                    "best-effort task summary failed: " + repr(exc) + "\\n"
                )
                raise SystemExit(0)

            pathlib.Path("/logs/agent/ouroboros-task-result.json").write_text(
                json.dumps(latest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            status = str(latest.get("status") or "")
            summary = {{
                "return_code": 0,
                "task_status_code": 0 if status == "completed" else 1,
                "elapsed_sec": None,
                "task_id": latest.get("task_id") or latest.get("id") or task_id,
                "status": status,
                "cost_usd": latest.get("cost_usd"),
                "prompt_tokens": latest.get("prompt_tokens"),
                "completion_tokens": latest.get("completion_tokens"),
                "total_rounds": latest.get("total_rounds"),
                "captured_after_cancellation": {captured_after_cancellation},
                "captured_at": time.time(),
            }}
            pathlib.Path("/logs/agent/ouroboros-run-summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with pathlib.Path("/logs/agent/ouroboros-run.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({{"type": "final", "task_id": task_id, "result": latest, "captured_after_cancellation": {captured_after_cancellation}}}, ensure_ascii=False) + "\\n")
            PY
            """
        ).strip()
        await environment.exec(command=command, timeout_sec=30)

    @staticmethod
    def _context_task_timeout_sec(context: Any) -> int | None:
        """Best-effort per-task timeout from the harbor AgentContext.

        Harbor's AgentContext does not currently expose the task.toml timeout
        (verified against harbor docs 2026-06: tokens/cost/rollout/metadata
        only), so this probe usually returns None today. If a future harbor
        adds a timeout field, the deadline pass-through (milestone nudges +
        run_command cap inside Ouroboros) lights up without an adapter change.
        """
        for attr in ("agent_timeout_sec", "task_timeout_sec", "timeout_sec", "max_agent_timeout_sec"):
            raw = getattr(context, attr, None)
            if raw is None and isinstance(getattr(context, "metadata", None), dict):
                raw = context.metadata.get(attr)
            try:
                value = int(raw) if raw is not None else 0
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    # Buffer between the agent's own deadline and Harbor's hard external kill, so the
    # loop's graceful self-finalize (which itself fires get_finalization_grace_sec before
    # the deadline) completes and the partial artifact is written before Harbor terminates.
    # v6.55.0: 30s let gpt2-codegolf overrun the deadline by 26.5s (a 351s
    # provider-recovery gap + a final round); 105s covers the measured
    # finalization overhead with margin (owner decision #15, range 90-120).
    _DEADLINE_SAFETY_SEC = 105

    # Mirror of ouroboros/tools/registry.py::_WEB_TOOLS (the adapter must stay
    # importable without the runtime package on the harbor host; a sync test in
    # tests/test_devtools_benchmarks.py pins this against the real set).
    _WEB_TOOLS_MIRROR = ("web_search", "browse_page", "browser_action", "youtube_transcript")
    _DELEGATED_VISION_TOOLS = ("analyze_screenshot", "vlm_query")

    def _resolve_task_timeout_from_dataset(self, context: Any) -> int | None:
        """Read the per-task agent wall-clock cap from the cached task.toml.

        Harbor's AgentContext does not expose the task.toml timeout, so derive the task
        name from the trial path (logs_dir = .../<task>__<trialhash>/agent) or context, then
        read ``[agent].timeout_sec`` from the cached dataset task.toml. Best-effort: returns
        None on any failure (agent then runs deadline-blind, as before — safe fallback)."""
        task_name = ""
        try:
            parent = Path(self.logs_dir).resolve().parent.name  # "<task>__<trialhash>"
            if "__" in parent:
                task_name = parent.rsplit("__", 1)[0]
        except Exception:
            task_name = ""
        if not task_name:
            for attr in ("task_id", "task_name", "task", "name"):
                raw = getattr(context, attr, None)
                if isinstance(raw, str) and raw.strip():
                    task_name = raw.strip().split("/")[-1].rsplit("__", 1)[0]
                    break
        if not task_name:
            return None
        try:
            import glob as _glob
            base = Path.home() / ".cache" / "harbor" / "tasks" / "packages" / "terminal-bench" / task_name
            matches = _glob.glob(str(base / "**" / "task.toml"), recursive=True)
            if not matches:
                return None
            # Pick the newest matching task.toml (avoid a stale cached package version).
            chosen = max(matches, key=lambda p: os.path.getmtime(p))
            text = Path(chosen).read_text(encoding="utf-8")
        except Exception:
            return None
        # Parse [agent].timeout_sec without a toml dependency (the field is a simple float/int).
        import re as _re
        section = None
        cap = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                section = s[1:-1].strip()
                continue
            if section == "agent":
                m = _re.match(r"timeout_sec\s*=\s*([0-9]+(?:\.[0-9]+)?)", s)
                if m:
                    try:
                        cap = int(float(m.group(1)))
                    except (TypeError, ValueError):
                        cap = None
                    break
        return cap if (cap and cap > 0) else None

    def _effective_task_timeout_sec(self) -> int:
        """The deadline (sec from task creation) handed to the agent: the per-task Harbor cap
        minus the install/server time already consumed and a small safety buffer. 0 means no
        deadline (agent runs as before). The agent uses this to pace and self-finalize a partial
        result before Harbor's hard external kill."""
        cap = self.task_timeout_sec
        if not cap or int(cap) <= 0:
            return 0
        elapsed = 0.0
        if self._run_started_monotonic is not None:
            elapsed = max(0.0, time.monotonic() - self._run_started_monotonic)
        effective = float(int(cap)) - elapsed - float(self._DEADLINE_SAFETY_SEC)
        # Cap IS known here (guard above returned for unknown caps). If install/server already ate
        # the budget, hand the agent a 1s deadline so it enters graceful finalization immediately
        # rather than running blind into Harbor's hard kill with an empty result.
        return int(effective) if effective > 0 else 1

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._run_started_monotonic = time.monotonic()
        if self.task_timeout_sec is None:
            probed = self._context_task_timeout_sec(context)
            if probed:
                self.task_timeout_sec = probed
        if self.task_timeout_sec is None:
            self.task_timeout_sec = self._resolve_task_timeout_from_dataset(context)
        (self.logs_dir / "instruction.txt").write_text(instruction, encoding="utf-8")
        await environment.upload_file(self.logs_dir / "instruction.txt", "/logs/agent/instruction.txt")

        env = self._container_env()
        reached_terminal_result = False
        try:
            self._enforce_container_secret_policy(env)
            self._openrouter_credit_preflight(self._host_settings())
            await self._network_preflight(environment, env)
            await self._resolve_workspace_dir(environment)
            await self._ensure_workspace_git_root(environment)
            await self._start_server(environment, env)
            self._run_summary = await self._run_ouroboros_task(environment, env)
            reached_terminal_result = True
        finally:
            try:
                # Only mark the captured summary as a cancellation when run() did NOT reach a
                # terminal result (i.e. Harbor actually interrupted mid-exec). On a normal terminal
                # finish this is a routine post-run snapshot, not a cancellation — so the disclosure
                # ledger does not misread a genuine terminal `provider_unavailable` as a wall-clock
                # cancellation (run_tb._failure_category keys on captured_after_cancellation).
                await self._capture_current_task_summary(
                    environment, interrupted=not reached_terminal_result
                )
            except Exception as exc:
                (getattr(self, "logger", None) or log).warning("Failed to capture in-container Ouroboros task summary: %s", exc)
            try:
                # Leaderboard submissions require an ATIF trajectory for every
                # passing trial (harbor static validation); emit it while the
                # trial logs are still in the container.
                await self._emit_trajectory(environment, env)
            except Exception as exc:
                (getattr(self, "logger", None) or log).warning("Failed to emit ATIF trajectory: %s", exc)
            if not self.leave_server_running_for_verifier or not reached_terminal_result:
                try:
                    await self._stop_server(environment)
                except Exception as exc:
                    (getattr(self, "logger", None) or log).warning("Failed to stop in-container Ouroboros cleanly: %s", exc)

        cost = self._run_summary.get("cost_usd")
        prompt_tokens = self._run_summary.get("prompt_tokens")
        completion_tokens = self._run_summary.get("completion_tokens")
        context.cost_usd = float(cost) if cost is not None else None
        context.n_input_tokens = int(prompt_tokens) if prompt_tokens is not None else None
        context.n_output_tokens = int(completion_tokens) if completion_tokens is not None else None
        context.metadata = {
            "adapter_mode": "installed_ouroboros",
            "workspace_dir": self.workspace_dir,
            "runtime_mode": self.runtime_mode,
            "review_enforcement": self.review_enforcement,
            "summary": self._run_summary,
        }


InstalledOuroborosTerminalBenchAgent = OuroborosTerminalBenchAgent

__all__ = ["OuroborosTerminalBenchAgent", "InstalledOuroborosTerminalBenchAgent"]
