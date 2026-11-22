"""Local llama-cpp-python server lifecycle, download, and health manager."""

from __future__ import annotations

import logging
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from ouroboros.local_model_server import LOCAL_LAUNCH_ENV, LOCAL_LAUNCH_HEADER
from ouroboros.platform_layer import (
    IS_MACOS, IS_WINDOWS, terminate_process_tree, kill_process_tree,
    subprocess_new_group_kwargs, merge_hidden_kwargs,
)

log = logging.getLogger(__name__)

_LOCAL_MODEL_DEFAULT_PORT = 8766
# start_server appends this pair last, so a model path cannot supply it.
_SERVING_CONTEXT_ARGV_RE = re.compile(r"\s--n_ctx[ =](\d+)$")
# start_server places the artifact first: `--model <path> --port <port>`.
_SERVING_MODEL_ARGV_RE = re.compile(r"\s--model (.+?) --port (\d+)(?:\s|$)")
# huggingface_hub's cache layout: <cache>/models--<org>--<repo>/snapshots/<revision>/<path in repo>.
_HF_CACHE_PATH_RE = re.compile(r"[/\\]models--([^/\\]+)[/\\]snapshots[/\\][^/\\]+[/\\](.+)$")


def local_dispatch_port() -> int:
    return int(os.environ.get("LOCAL_MODEL_PORT", _LOCAL_MODEL_DEFAULT_PORT))


def local_artifact_key(source: str = "", filename: str = "", *, model_path: str = "") -> str:
    """One identity for a configured artifact and the file ``download_model`` serves for it.

    A local file is its expanded path (the filename is unused there); a Hugging Face artifact is
    ``hf:<repo>:<repo-relative path>`` from settings or its cache path. Basename selections
    use the downloader's metadata resolver, never infer equivalence by dropping subfolders.
    This resolves metadata only; it never downloads an artifact."""
    if model_path:
        cached = _HF_CACHE_PATH_RE.search(str(model_path))
        return (f"hf:{cached.group(1).replace('--', '/')}:{cached.group(2).replace(chr(92), '/')}"
                if cached else os.path.expanduser(str(model_path)))
    source, filename = str(source or "").strip(), str(filename or "").strip()
    if not source or os.path.isfile(source) or source.startswith(("/", "~")):
        return os.path.expanduser(source)
    filename = filename.replace(chr(92), '/')
    resolved = LocalModelManager._resolve_hf_path(source, filename) if filename else filename
    return f"hf:{source}:{resolved}"


# What the owned server imports at start (``local_model_server`` runs ``llama_cpp.server``).
RUNTIME_PROBE = "import llama_cpp, llama_cpp.server.app"

def local_model_settings(values: dict) -> dict:
    """Effective launch values, shared by execution and saved/applied comparison."""
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    default = int(SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"])
    context = int(values.get("LOCAL_MODEL_CONTEXT_LENGTH", default))
    return {
        "LOCAL_MODEL_SOURCE": str(values.get("LOCAL_MODEL_SOURCE") or "").strip(),
        "LOCAL_MODEL_FILENAME": str(values.get("LOCAL_MODEL_FILENAME") or "").strip(),
        "LOCAL_MODEL_PORT": int(values.get("LOCAL_MODEL_PORT", _LOCAL_MODEL_DEFAULT_PORT)),
        "LOCAL_MODEL_N_GPU_LAYERS": int(values.get("LOCAL_MODEL_N_GPU_LAYERS", 0)),
        "LOCAL_MODEL_CONTEXT_LENGTH": context if context > 0 else default,
        "LOCAL_MODEL_CHAT_FORMAT": str(values.get("LOCAL_MODEL_CHAT_FORMAT") or "").strip(),
    }

def _get_install_command() -> list:
    """Return the llama-cpp-python pip install command."""
    from ouroboros.platform_layer import pip_install_target_args

    return [sys.executable, "-m", "pip", "install",
            *pip_install_target_args(sys.executable), "--upgrade", "llama-cpp-python[server]"]


def _get_install_env() -> dict:
    """Return pip install env, enabling Metal flags on macOS."""
    env = os.environ.copy()
    if IS_MACOS:
        env.update(CMAKE_ARGS="-DGGML_METAL=on", FORCE_CMAKE="1")
    return env


class LocalModelStartConflict(RuntimeError):
    """The requested local-model operation conflicts with one already in progress."""


class LocalRuntimeMissing(RuntimeError):
    """A failed server import and the installation advice this deployment permits."""

    def __init__(self, detail: str = "") -> None:
        from ouroboros.deployment_policy import body_replacement_refusal

        message = "The local model runtime could not load llama-cpp-python."
        self.hint = ""
        if body_replacement_refusal():
            message += " Your administrator installs and updates the local runtime."
        else:
            flags = 'CMAKE_ARGS="-DGGML_METAL=on" ' if IS_MACOS else ""
            self.hint = flags + "pip install llama-cpp-python[server]"
        super().__init__(f"{message}\n{detail[-500:]}" if detail else message)



def _run_runtime_probe() -> subprocess.CompletedProcess:
    """The server application's import probe, shared by the runtime check and start_server."""
    return subprocess.run(
        [sys.executable, "-c", RUNTIME_PROBE],
        **_with_hidden_subprocess({"capture_output": True, "text": True, "timeout": 15}),
    )


def _read_tail(stream, limit: int) -> bytes:
    """Read a pipe to EOF, retaining only its last ``limit`` bytes."""
    buf = b""
    try:
        while chunk := os.read(stream.fileno(), 4096):
            buf = (buf + chunk)[-limit:]
    except Exception:
        pass
    return buf


_manager: Optional[LocalModelManager] = None
_manager_lock = threading.Lock()


def get_manager(*, create: bool = True) -> Optional[LocalModelManager]:
    global _manager
    if _manager is not None or not create:
        return _manager
    with _manager_lock:
        if _manager is None:
            _manager = LocalModelManager()
        return _manager


class LocalModelManager:
    """Lifecycle manager for a llama-cpp-python server subprocess."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._service_binding: Optional[dict] = None
        self._install_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._pending_start: Optional[object] = None
        self._status = "offline"
        self._error: Optional[str] = None
        self._model_path: Optional[str] = None
        self._port: int = _LOCAL_MODEL_DEFAULT_PORT
        self._context_length: int = 0

        self._serving_context_length: int = 0
        self._measurement_route: bool = False
        self._model_name: str = ""
        self._launch_settings: dict = {}
        self._applied_settings: dict = {}
        self._download_progress: float = 0.0
        self._stderr_buf = bytearray()
        # Runtime (llama-cpp-python) install state
        self._runtime_status: str = "unknown"  # Last import or installation result.
        self._runtime_install_log: str = ""
        self._runtime_probe: Optional[Future] = None
        self._runtime_probe_error = ""
        self._installing = False
        self._install_cancelled = threading.Event()
        self._panic_requested = False
        self._grants: Dict[str, str] = {}  # a deferred Start's receipts (boot's, an install's): every Stop revokes them

    def get_status(self) -> str:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> str:
        if self._status in ("stopping", "stop_unconfirmed"):
            return "stopping"  # Stop alone settles this child and its binding.
        if self._proc is not None and (code := self._proc.poll()) is not None:
            if self._status == "loading":
                self._status = "startup_exit"
                self._error = f"Server process exited during startup (code {code})"
            else:
                self._status = "error"
                self._error = f"Server exited with code {code}"
            self._proc = None
        return "error" if self._status == "startup_exit" else self._status

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_running(self) -> bool:
        return self.get_status() == "ready"

    def status_dict(self, *, check_runtime: bool = False) -> Dict[str, Any]:
        if check_runtime:
            self._runtime_import(passive=True)
        with self._lock:
            result = {
                "status": self._status_locked(),
                "error": self._error or self._runtime_probe_error or None,
                "model_path": self._model_path,
                "model_name": self._model_name,
                "context_length": self._context_length,
                "port": self._port,
                "download_progress": self._download_progress,
                "runtime_status": "installing" if self._installing else self._runtime_status,
                "runtime_install_log": self._runtime_install_log[-500:] if self._runtime_install_log else "",
            }
            if self._status == "startup_exit":
                result["error_stage"] = "startup_exit"
                if self._stderr_buf:
                    result["error"] = f"{self._error}: {self._stderr_buf.decode('utf-8', errors='replace')[-500:]}"
            return result

    def settings_application(self, settings: dict) -> dict:
        """A ready owned process applies its captured inputs; other states do not."""
        desired = local_model_settings(settings)
        with self._lock:
            status = self._status_locked()
            applied = dict(self._applied_settings) if status == "ready" else {}
        pending = sorted(key for key in desired if key in applied and desired[key] != applied[key])
        unknown = sorted(set(desired) - set(applied)) if status == "ready" else []
        if pending:
            action = "Stop, then Start" if desired["LOCAL_MODEL_SOURCE"] else "Stop"
            summary = f"Saved local model settings differ from the running model. Use {action} to apply them."
        elif status not in ("ready", "stopping") and desired["LOCAL_MODEL_SOURCE"]:
            summary = "Local model settings are not applied to a ready model. Use the local model controls to start it."
        elif unknown:
            summary = "Some running local model settings were not reported."
        else:
            summary = ""
        return {"status": status, "pending_keys": pending, "unknown_keys": unknown, "summary": summary}

    @staticmethod
    def _probe_runtime() -> subprocess.CompletedProcess:
        kwargs = merge_hidden_kwargs({"capture_output": True, "text": True, "timeout": 15})
        return subprocess.run([sys.executable, "-c", RUNTIME_PROBE], **kwargs)

    def _runtime_import(self, *, passive: bool = False, start: object | None = None) -> Optional[subprocess.CompletedProcess]:
        """Pollers share one import, including an inconclusive result. An explicit Start checks again."""
        with self._lock:
            if start is not None:
                self._check_start_locked(start)
            if passive and (self._installing or self._panic_requested or self._runtime_status != "unknown"
                            or self._status_locked() != "offline"):
                return None
            probe = self._runtime_probe
            run = probe is None or (probe.done() and not passive)
            if run:
                self._runtime_probe = probe = Future()
                self._runtime_probe_error = ""

        if run:
            result, error = None, ""
            try:
                result = self._probe_runtime()
            except Exception as exc:
                error = f"Could not check the local model runtime: {exc}"
            with self._lock:
                if self._runtime_probe is probe and not self._installing:
                    self._runtime_probe_error = error
                    self._runtime_status = "unknown" if error else ("ok" if result.returncode == 0 else "missing")
                probe.set_result((result, error))
        result, error = probe.result()
        if error and not passive:
            raise RuntimeError(error)
        return result

    def check_runtime(self, *, start: object | None = None) -> bool:
        """Probe the server application and its dependencies in the interpreter used at launch."""
        return self._runtime_import(start=start).returncode == 0

    def boot_grant(self) -> str:
        """Boot's receipt for its staged Start: a Stop revokes it, and ``_starting`` then refuses that Start."""
        with self._lock:
            return self._grants.setdefault("boot", uuid4().hex)

    def install_runtime(self) -> tuple[str, str]:
        """Reserve the interpreter until pip and its verification finish: (status, install_id, "" if joined)."""
        with self._lock:
            if self._installing:
                return "already_installing", ""
            if self._panic_requested:
                raise RuntimeError("Local runtime install cancelled by Panic")
            stopping = self._status_locked() == "stopping"
            if stopping or self._pending_start is not None or self._proc is not None:
                raise LocalModelStartConflict("Stop the local model before installing the runtime.")
            self._install_cancelled.clear()
            self._grants["install"] = install_id = uuid4().hex
            self._runtime_probe = None
            self._runtime_probe_error = self._runtime_install_log = ""
            self._installing = True
        try:
            threading.Thread(target=self._run_install, daemon=True, name="llama-install").start()
        except Exception as exc:
            with self._lock:
                self._installing = False
                self._runtime_status, self._runtime_install_log = "install_error", str(exc)
            raise
        return "installing", install_id

    def _run_install(self) -> None:
        status, output = "install_error", ""
        try:
            if self._install_cancelled.is_set() or self._panic_requested:
                return
            cmd = _get_install_command()
            log.info("Installing llama-cpp-python: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd, env=_get_install_env(),
                **merge_hidden_kwargs({
                    "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                    **subprocess_new_group_kwargs(),  # Stop signals the install's tree, never this server (#2353).
                }),
            )
            with self._lock:
                self._install_proc = proc
                stopped_during_spawn = self._install_cancelled.is_set()
            if self._panic_requested:
                self.panic_stop(request_only=True)
            elif stopped_during_spawn:
                self._stop_install_process(proc)

            tail = b""
            try:
                if proc.stdout:
                    with proc.stdout:
                        while chunk := os.read(proc.stdout.fileno(), 4096):
                            tail = (tail + chunk)[-4096:]
            except (OSError, ValueError):
                log.debug("Could not read pip output", exc_info=True)
            finally:
                proc.wait()
            output = tail.decode("utf-8", errors="replace")
            with self._lock:
                self._install_proc = None
                self._runtime_probe = None  # Only an import after pip exits can verify this install.
            if proc.returncode == 0 and not self._install_cancelled.is_set():
                if self.check_runtime():
                    status = "install_ok"
                    log.info("llama-cpp-python installed successfully")
                else:
                    output += "\nInstall succeeded but import still fails."
                    log.warning("llama-cpp-python install ran but import still fails")
            elif proc.returncode and not self._install_cancelled.is_set():
                log.error("llama-cpp-python install failed (rc=%s)", proc.returncode)
        except Exception as exc:
            output += f"\n{exc}"
            log.exception("Local runtime install failed")
        finally:
            with self._lock:
                self._install_proc = self._runtime_probe = None
                self._runtime_status = "unknown" if self._install_cancelled.is_set() else status
                self._runtime_install_log = output
                self._installing = False

    @staticmethod
    def _resolve_hf_path(source: str, filename: str) -> str:
        """Resolve omitted HF subfolders; fail open so original errors propagate."""
        if "/" in filename:
            return filename
        try:
            from huggingface_hub import list_repo_files
            all_paths = list(list_repo_files(source))
        except Exception:
            return filename  # fail-open: caller will get the original 404 error
        matches = [p for p in all_paths if p.endswith("/" + filename) or p == filename]
        if not matches:
            return filename
        if len(matches) > 1:
            paths_str = "\n  ".join(matches)
            raise ValueError(
                f"Ambiguous filename '{filename}' in repo '{source}' — found in multiple locations:\n"
                f"  {paths_str}\n"
                f"Specify the full path including subfolder, "
                f"e.g. '{matches[0]}'"
            )
        resolved = matches[0]
        if resolved != filename:
            log.info(
                "Auto-resolved HF path: '%s' → '%s' (subfolder detected automatically)",
                filename, resolved,
            )
        return resolved

    def _check_start_locked(self, start: object | None) -> None:
        if self._panic_requested:
            raise LocalModelStartConflict("Local model startup cancelled by Panic")
        if self._pending_start is not start:
            raise LocalModelStartConflict("Local model startup cancelled by Stop")

    @contextmanager
    def _starting(self, grant: Optional[str] = None):
        with self._lock:
            stopping = self._status_locked() == "stopping"
            if self._panic_requested:
                raise LocalModelStartConflict("Local model startup cancelled by Panic")
            if grant is not None and grant not in self._grants.values():
                raise LocalModelStartConflict("Local model startup cancelled: its install was stopped or replaced")
            if self._installing:
                raise LocalModelStartConflict("The local runtime is being installed.")
            if stopping or self._pending_start is not None or self._proc is not None:
                raise LocalModelStartConflict("The local model server is already starting, running, or stopping.")
            self._pending_start = start = object()
            self._status, self._error, self._download_progress = "loading", None, 0.0
        try:
            yield start
            with self._lock:
                self._check_start_locked(start)
        except Exception as exc:
            with self._lock:
                self._check_start_locked(start)
                if self._proc is None and self._status not in ("error", "startup_exit"):
                    self._status, self._error = "error", str(exc)
            raise
        finally:
            with self._lock:
                # Stop may already have admitted another model.
                if self._pending_start is start:
                    self._pending_start = None

    @staticmethod
    def _normalize_hf_filename(filename: str):
        """Split an HF filename into (subfolder, basename)."""
        filename = filename.strip()
        if "/" in filename:
            subfolder, basename = filename.rsplit("/", 1)
            return subfolder.strip() or None, basename.strip()
        return None, filename

    @staticmethod
    def _detect_shard_info(basename: str):
        """Return split-GGUF shard metadata, preserving original digit widths."""
        import re
        m = re.match(r"^(.*?)-(\d+)-of-(\d+)(\.gguf)$", basename, re.IGNORECASE)
        if m:
            prefix, shard_str, total_str, suffix = m.groups()
            shard_num = int(shard_str)
            total_shards = int(total_str)
            if total_shards >= 1 and 1 <= shard_num <= total_shards:
                return prefix, shard_num, total_shards, len(shard_str), len(total_str), suffix
        return None

    @staticmethod
    def _all_shard_basenames(
        prefix: str,
        total_shards: int,
        suffix: str,
        shard_width: int = 5,
        total_width: int = 5,
    ):
        """Yield split-GGUF shard basenames using the original digit widths."""
        for i in range(1, total_shards + 1):
            yield f"{prefix}-{str(i).zfill(shard_width)}-of-{str(total_shards).zfill(total_width)}{suffix}"

    def download_model(
        self, source: str, filename: str = "", progress_cb: Optional[Callable[[float], None]] = None,
        *, _start: object | None = None,
    ) -> str:
        """Resolve/download a .gguf; split GGUFs download all shards from shard 1."""
        with self._lock:
            self._check_start_locked(_start)
            if self._status == "startup_exit":
                self._status, self._error = "offline", None
        path = os.path.expanduser(source)
        if os.path.isfile(path):
            log.info("Using local model file: %s", path)
            return path
        if source.startswith(("/", "~")):
            raise FileNotFoundError(f"Local model file not found: {path}")

        try:
            from huggingface_hub import hf_hub_download
            from tqdm.auto import tqdm
        except ImportError:
            raise RuntimeError("Downloading a model requires huggingface_hub. You can also open a local model file.")
        if not filename:
            raise ValueError("filename is required when source is a HuggingFace repo ID. "
                             "Example: filename='model-Q4_K_M.gguf' or 'quant/model-00001-of-00003.gguf'")

        filename = self._resolve_hf_path(source, filename)
        folder, basename = self._normalize_hf_filename(filename)
        names, count = (basename,), 1
        if shards := self._detect_shard_info(basename):
            prefix, number, count, shard_width, total_width, suffix = shards
            names = self._all_shard_basenames(prefix, count, suffix, shard_width, total_width)
            if number != 1:
                first = "/".join(filter(None, (folder, next(names))))
                raise ValueError(f"Split GGUF: shard {number} of {count} specified. "
                                 f"Please use the first shard to start the server. Change filename to: '{first}'")

        def progress(fraction):
            with self._lock:
                self._check_start_locked(_start)
                self._download_progress = fraction
            if progress_cb is not None:
                progress_cb(fraction)

        def progress_bar(index):
            class DownloadProgress(tqdm):
                def update(self, n=1):
                    result = super().update(n)
                    try:
                        if self.total and self.total > 0:
                            progress((index + min(self.n / self.total, 1.0)) / count)
                    except LocalModelStartConflict:
                        raise
                    except Exception:
                        pass
                    return result
            return DownloadProgress

        first_path = None
        try:
            for index, name in enumerate(names):
                with self._lock:
                    self._check_start_locked(_start)
                    self._status = "downloading"
                    self._download_progress = index / count
                log.info("Downloading %s/%s (%s/%s)", source, name, index + 1, count)
                path = hf_hub_download(repo_id=source, filename=name, subfolder=folder or None,
                                       resume_download=True, tqdm_class=progress_bar(index))
                if index == 0:
                    first_path = path
            progress(1.0)
            log.info("Model downloaded to: %s", first_path)
            return first_path
        except Exception as exc:
            with self._lock:
                self._check_start_locked(_start)
                self._status, self._error = "error", f"Download failed: {exc}"
            raise

    def start_model(self, source: str, filename: str, *, grant: Optional[str] = None, **options: Any) -> str:
        """Reserve one Start across the import probe, download and child publication; a deferred one names its grant."""
        with self._starting(grant) as start:
            if not self.check_runtime(start=start):
                raise LocalRuntimeMissing()
            path = self.download_model(source, filename, _start=start)
            self.start_server(path, source=source, filename=filename, _start=start, **options)
            return path

    def start_server(
        self,
        model_path: str,
        port: int = _LOCAL_MODEL_DEFAULT_PORT,
        n_gpu_layers: int = -1,
        n_ctx: int = 0,
        chat_format: str = "",
        source: str | None = None,
        filename: str | None = None,
        *, _start: object | None = None,
    ) -> None:
        """Start the server; rechecks runtime as a safety net before Popen."""
        with (self._starting() if _start is None else nullcontext(_start)) as start:
            probe = self._runtime_import(start=start)
            with self._lock:
                self._check_start_locked(start)
                if probe.returncode != 0:
                    raise LocalRuntimeMissing((probe.stderr or probe.stdout or "").strip())
                self._model_path = model_path
                self._status = "loading"
                self._error = None
                launch = local_model_settings({"LOCAL_MODEL_SOURCE": source, "LOCAL_MODEL_FILENAME": filename,
                    "LOCAL_MODEL_PORT": port, "LOCAL_MODEL_N_GPU_LAYERS": n_gpu_layers,
                    "LOCAL_MODEL_CONTEXT_LENGTH": n_ctx, "LOCAL_MODEL_CHAT_FORMAT": chat_format})
                self._launch_settings = {key: value for key, value in launch.items()
                                        if not ((key == "LOCAL_MODEL_SOURCE" and source is None)
                                                or (key == "LOCAL_MODEL_FILENAME" and filename is None))}
                self._applied_settings = {}
                self._port = port = launch["LOCAL_MODEL_PORT"]
                n_gpu_layers, chat_format = launch["LOCAL_MODEL_N_GPU_LAYERS"], launch["LOCAL_MODEL_CHAT_FORMAT"]

                # A Windows venv's python.exe is a redirector that runs the base interpreter as its child: start that
                # interpreter, naming the venv (as extension_companion does), so the PID held here is the server's own.
                base = getattr(sys, "_base_executable", None) or sys.executable
                redirected = IS_WINDOWS and os.path.normcase(base) != os.path.normcase(sys.executable)
                cmd = [
                    base if redirected else sys.executable, "-m", "ouroboros.local_model_server",
                    "--model", model_path,
                    "--port", str(port),
                    "--n_gpu_layers", str(n_gpu_layers),
                ]
                if chat_format:
                    cmd.extend(["--chat_format", chat_format])
                effective_ctx = launch["LOCAL_MODEL_CONTEXT_LENGTH"]
                self._context_length = self._serving_context_length = effective_ctx
                self._measurement_route = True
                cmd.extend(["--n_ctx", str(effective_ctx)])

                log.info("Starting local model server: %s", " ".join(cmd))

                try:
                    launch_id = uuid4().hex
                    _popen_kwargs = dict(
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.DEVNULL,
                        env={**os.environ, **({"__PYVENV_LAUNCHER__": sys.executable} if redirected else {}),
                             LOCAL_LAUNCH_ENV: launch_id, "PYTHONPATH": os.pathsep.join(filter(None, (
                            str(pathlib.Path(__file__).resolve().parent.parent), os.environ.get("PYTHONPATH", ""))))},
                    )
                    _popen_kwargs.update(subprocess_new_group_kwargs())
                    self._proc = subprocess.Popen(cmd, **merge_hidden_kwargs(_popen_kwargs))
                    if self._panic_requested:
                        self.panic_stop(request_only=True)
                    try:
                        from ouroboros.config import DATA_DIR
                        from ouroboros.process_custody import record_process

                        record_process(
                            pathlib.Path(DATA_DIR),
                            pid=self._proc.pid,
                            cmd=cmd,
                            purpose="local_model_server",
                            scope="session",
                        )
                    except Exception:
                        log.debug("local model custody record failed", exc_info=True)
                except FileNotFoundError:
                    self._status = "error"
                    self._error = "Python executable not found. Cannot start local model server."
                    raise RuntimeError(self._error)

                self._stderr_buf = bytearray()
                proc = self._proc
                threading.Thread(
                    target=self._drain_stderr, args=(proc, self._stderr_buf), daemon=True, name="local-model-stderr"
                ).start()

            threading.Thread(target=self._wait_for_healthy, kwargs={"proc": proc, "port": port, "launch_id": launch_id},
                             daemon=True, name="local-model-health").start()

    def _drain_stderr(self, proc: subprocess.Popen, buf: bytearray) -> None:
        """Drain stderr to avoid pipe deadlock and retain a diagnostic tail."""
        if proc.stderr is None:
            return
        try:
            fd = proc.stderr.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                buf[:] = (buf + chunk)[-2048:]
        except Exception:
            pass

    def _wait_for_healthy(self, *, proc: subprocess.Popen, port: int, launch_id: str, timeout: float = 300.0) -> None:
        """Only a model list from the current live launch can establish readiness."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._proc is not proc or self._status_locked() != "loading" or self._panic_requested:
                    return
                if time.monotonic() >= deadline:
                    self._status = "error"
                    self._error = f"Server failed to become healthy within {timeout}s"
                    log.error(self._error)
                    return
            try:
                health = self.health_check(port=port, launch_id=launch_id)
            except Exception:
                health = {}
            if not health.get("ok"):
                time.sleep(2.0)
                continue

            from ouroboros.config import DATA_DIR
            from ouroboros.server_process import record_service_binding

            with self._lock:
                if self._proc is not proc or self._status_locked() != "loading" or self._panic_requested:
                    return
                try:
                    self._service_binding = record_service_binding(
                        DATA_DIR, "local_model", "127.0.0.1", port, pid=proc.pid)
                except Exception:
                    self._service_binding = None
                    log.warning("Local model service binding unavailable", exc_info=True)
                if self._status_locked() != "loading" or self._panic_requested:
                    return
                self._applied_settings = dict(self._launch_settings)
                self._context_length, self._model_name = health["context_length"], health["model_name"]
                self._status = "ready"
            log.info("Local model server ready (ctx=%s, model=%s)", health["context_length"], health["model_name"])
            return

    def panic_stop(self, *, request_only: bool = False) -> list:
        """Signal our captured children before manager locks or cleanup waits."""
        from ouroboros.platform_layer import request_process_tree_kill

        self._panic_requested = True
        self._install_cancelled.set()
        requests = [request_process_tree_kill(proc) for proc in (self._proc, self._install_proc)
                    if proc is not None]
        if not request_only:
            self.stop_server()
        return requests

    @staticmethod
    def _stop_install_process(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            terminate_process_tree(proc)
            proc.wait(timeout=5)
        except Exception:
            try:
                kill_process_tree(proc)
            except Exception:
                log.warning("Could not stop the local runtime installer", exc_info=True)

    def stop_server(self) -> bool:
        """Confirm model exit; any installer keeps its independent cancellation and settlement."""
        with self._lock:
            self._grants.clear()  # every Stop request, a repeat included, ends the Starts that wait on a receipt
            if self._status == "stopping":  # Another Stop is waiting on this child.
                return False
            self._install_cancelled.set()
            if self._pending_start is not None:
                self._pending_start = self._runtime_probe = None
            proc, install_proc = self._proc, self._install_proc
            self._status, self._error = "stopping", None

        exited = proc is None
        try:
            if proc is not None:
                for signal_child, timeout in ((terminate_process_tree, 10), (kill_process_tree, 5)):
                    try:
                        exited = proc.poll() is not None
                        if not exited:
                            signal_child(proc)
                            exited = proc.wait(timeout=timeout) is not None
                        if exited:
                            break
                    except (OSError, subprocess.SubprocessError):
                        log.debug("Local model stop attempt (pid=%s) unconfirmed", proc.pid, exc_info=True)
        finally:
            with self._lock:
                if exited:
                    self._proc = None
                    binding, self._service_binding = self._service_binding, None
                    self._status, self._error = "offline", None
                    self._applied_settings, self._stderr_buf = {}, bytearray()
                    self._context_length = self._serving_context_length = 0
                    self._download_progress = 0.0
                    self._measurement_route, self._model_name = False, ""
                else:
                    self._status, self._error = "stop_unconfirmed", "Server exit is not confirmed"
        if not exited:
            return False
        if binding is not None:
            from ouroboros.config import DATA_DIR
            from ouroboros.server_process import clear_service_binding

            clear_service_binding(DATA_DIR, "local_model", binding)
        if install_proc is not None:
            self._stop_install_process(install_proc)
        return True

    def health_check(self, *, port: int | None = None, launch_id: str | None = None) -> Dict[str, Any]:
        """Read loopback metadata; readiness also requires the specified launch's own reply."""
        import requests

        url = f"http://127.0.0.1:{self._port if port is None else port}/v1/models"
        with requests.Session() as session:
            session.trust_env = False  # Owned loopback never needs proxy discovery, in any process.
            resp = session.get(url, timeout=5, allow_redirects=launch_id is None)
        resp.raise_for_status()
        if launch_id is not None and (not launch_id or resp.status_code != 200
                                      or resp.headers.get(LOCAL_LAUNCH_HEADER) != launch_id):
            return {"ok": False, "error": "Model list did not identify the launched server"}
        data = resp.json()
        models = data.get("data", [])
        if not models:
            return {"ok": False, "error": "No models loaded"}

        model_info = models[0]
        ctx = model_info.get("meta", {}).get("n_ctx_train", 0)
        if not ctx:
            ctx = model_info.get("context_window", 0)

        return {"ok": True, "model_name": model_info.get("id", "unknown"), "context_length": ctx}

    def serving_context_evidence(self, *, bind_endpoint: bool = False) -> Dict[str, Any]:
        """The live owned server's configured window, distinct from training metadata."""
        from ouroboros.utils import in_worker_process

        if in_worker_process():  # A forked worker's inherited _proc is not its child.
            return self._published_serving_evidence()
        process = self._proc
        capacity = int(getattr(self, "_serving_context_length", 0) or 0)
        known = bool(self._status == "ready" and process is not None and process.poll() is None and capacity > 0)
        evidence = {"context_window": capacity if known else None, "confirmed": known,
                    "source": "owned_server_arguments" if known else "serving_window_unobserved",
                    "process_id": process.pid if known else None}
        if bind_endpoint and known:
            evidence["port"] = int(self._port)  # ready health outlives a failed binding write (``_wait_for_healthy``)
        return evidence

    def serving_artifact(self) -> Dict[str, Any]:
        """The model file and port the live server was launched with, else ``{}``.

        Saved settings stay pending until Stop/Start (``settings_application``), so a fact
        about the serving model binds this launch identity, never the desired settings.
        A worker reads the published, liveness-checked binding's own arguments."""
        from ouroboros.utils import in_worker_process

        if not in_worker_process():
            process = self._proc
            live = self._status == "ready" and process is not None and process.poll() is None
            return {"model_path": str(self._model_path), "port": int(self._port)} if live and self._model_path else {}
        evidence = self._published_serving_evidence()
        if not evidence.get("confirmed"):
            return {}
        from ouroboros.platform_layer import process_command

        match = _SERVING_MODEL_ARGV_RE.search(process_command(int(evidence["process_id"])))
        if not match or int(match.group(2)) != int(evidence.get("port") or 0):
            return {}
        return {"model_path": match.group(1).strip('"'), "port": int(match.group(2))}

    def _published_serving_evidence(self) -> Dict[str, Any]:
        """A worker's view of the server its server process owns.

        Workers never own the server: the owner publishes the ready instance's service binding, which counts
        only while that exact process (pid and platform fingerprint) is live on this worker's loopback port;
        its window is that process's own ``--n_ctx`` argument.
        """
        unknown = {"context_window": None, "confirmed": False,
                   "source": "serving_window_unobserved", "process_id": None}
        try:
            from ouroboros.config import DATA_DIR
            from ouroboros.platform_layer import process_command
            from ouroboros.server_process import read_service_bindings, service_binding_is_live

            binding = read_service_bindings(pathlib.Path(DATA_DIR)).get("local_model") or {}
            # The endpoint this worker's LLMClient._get_local_client dispatches to.
            port = local_dispatch_port()
            if (binding.get("host"), binding.get("port")) != ("127.0.0.1", port):
                return unknown
            # Read argv before the identity check: a pid that still matches its
            # recorded birth afterwards already belonged to that process here.
            match = _SERVING_CONTEXT_ARGV_RE.search(process_command(int(binding["pid"])))
            window = int(match.group(1)) if match else 0
            if window <= 0 or not service_binding_is_live(binding):
                return unknown
        except Exception:
            return unknown
        return {"context_window": window, "confirmed": True, "source": "published_server_arguments",
                "process_id": int(binding["pid"]), "port": port}

    def measure_prepared_input(self, payload: Dict[str, Any], *, expected_port: Optional[int] = None) -> Dict[str, Any]:
        """Bounded read-only I/O to the same owned server; it never generates tokens."""
        from ouroboros.local_model_server import input_fingerprint

        evidence = self.serving_context_evidence(bind_endpoint=True)
        unknown = {"supported": False, "input_is_exact": False, "reason": "measurement_unavailable"}
        # A published instance was launched by start_server, which always serves this route.
        published = evidence.get("source") == "published_server_arguments"
        if not evidence["confirmed"] or not (published or getattr(self, "_measurement_route", False)):
            return unknown
        expected_pid, port = evidence["process_id"], evidence.get("port")
        try:
            if port is None or port != (local_dispatch_port() if expected_port is None else expected_port):
                return {**unknown, "reason": "measurement_endpoint_mismatch"}
            import requests

            with requests.Session() as session:
                session.trust_env = False  # Owned loopback must not use an external proxy.
                response = session.post(f"http://127.0.0.1:{port}/extras/measure_chat",
                                        json=payload, timeout=5.0, allow_redirects=False)
                if response.status_code != 200:
                    return unknown
                measured = response.json()
            current = self.serving_context_evidence(bind_endpoint=True)
            if (not isinstance(measured, dict) or measured.get("process_id") != expected_pid
                    or not current.get("confirmed")
                    or any(current.get(key) != evidence.get(key) for key in ("source", "process_id", "port", "context_window"))
                    or measured.get("native_input_sha256") != input_fingerprint(payload)
                    or measured.get("context_window") != current.get("context_window")):
                return {**unknown, "reason": "measurement_route_changed"}
            return measured
        except Exception as error:
            return {**unknown, "reason": f"measurement_unavailable:{type(error).__name__}"}

    def get_context_length(self) -> int:
        """Metadata (launch value while loading), or 0; not serving capacity or health."""
        try:
            self._context_length = max(0, int(
                self._context_length or self.health_check().get("context_length") or 0))
        except Exception:
            self._context_length = 0
        return self._context_length

    def test_tool_calling(self) -> Dict[str, Any]:
        """Run basic chat and tool-call checks against the local server.

        Both probes are ordinary generations of the built-in model: each takes the server's own queue slot
        (``model_concurrency.built_in_route``) inside its physical sender, as the local lane's sends do, so a
        probe never opens beside a generation and the server's own count cannot enter while one is open. The
        send time starts after that slot, so a queued probe reports its own generation, not its wait.

        The slot names THIS server's endpoint (the port read once at probe start, the one ``start_server``
        accepted) because the client below sends there: ``built_in_route`` otherwise names the lane's configured
        port, and a server listening elsewhere is a different one whose queue this probe must neither join nor
        block. One read also keeps a host that moves the port between them from splitting the address the SDK
        calls from the queue it holds onto.
        """
        from openai import OpenAI
        from ouroboros.model_concurrency import built_in_route, limit_physical_send
        from ouroboros.usage_accounting import (
            AttemptRequest,
            UsageAccountingError,
            UsageScope,
            current_usage_scope,
            execute_physical_attempt,
            usage_scope,
        )

        port = int(self._port)  # one read: the SDK's address and the queue below must name one endpoint
        client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="local",
            max_retries=0,
        )
        # The queue this server's generations and its own count share, at the endpoint the client below reaches.
        route = built_in_route(port)

        def _dispatch_probe(payload: Dict[str, Any]) -> Any:
            request = AttemptRequest(
                model="local-model",
                provider="local",
                prompt_tokens_estimate=max(0, len(str(payload)) // 4),
                max_completion_tokens=int(payload.get("max_tokens") or 0),
                source="capability_probe.local_model",
            )
            elapsed = {"send": 0.0}

            def _send() -> Any:
                started = time.monotonic()
                try:
                    return client.chat.completions.create(**payload)
                finally:
                    elapsed["send"] = time.monotonic() - started

            def _attempt() -> Any:
                return execute_physical_attempt(request, limit_physical_send(route, _send))

            if current_usage_scope() is not None:
                response = _attempt()
            else:
                with usage_scope(UsageScope(
                    task_id="system:capability_probe",
                    root_task_id="system:capability_probe",
                    category="capability_probe",
                    non_task_operation=True,
                    source="capability_probe.local_model",
                )):
                    response = _attempt()
            return response, elapsed["send"]

        result: Dict[str, Any] = {
            "success": False,
            "chat_ok": False,
            "tool_call_ok": False,
            "details": "",
            "tokens_per_sec": 0.0,
        }

        try:
            resp, elapsed = _dispatch_probe({
                "model": "local-model",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 32,
            })
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            tokens = resp.usage.completion_tokens if resp.usage else len(text.split())
            result["chat_ok"] = bool(text.strip())
            if elapsed > 0 and tokens > 0:
                result["tokens_per_sec"] = round(tokens / elapsed, 1)
        except UsageAccountingError:
            raise
        except Exception as e:
            result["details"] = f"Basic chat failed: {e}"
            return result

        try:
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Returns the current time.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]
            resp, _elapsed = _dispatch_probe({
                "model": "local-model",
                "messages": [{"role": "user", "content": "What time is it? Use the get_time tool."}],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 256,
            })
            msg = resp.choices[0].message if resp.choices else None
            tool_calls = list(getattr(msg, "tool_calls", None) or []) if msg else []
            if msg and not tool_calls and getattr(msg, "content", None):
                from ouroboros.llm import LLMClient

                parsed = LLMClient._parse_tool_calls_from_content(
                    {
                        "content": msg.content,
                        "tool_calls": [],
                    },
                    {"get_time"},
                )
                tool_calls = parsed.get("tool_calls") or []
            if tool_calls:
                result["tool_call_ok"] = True
            else:
                result["details"] = "Model returned text instead of tool_call"
        except UsageAccountingError:
            raise
        except Exception as e:
            result["details"] = f"Tool call test failed: {e}"
            result["success"] = result["chat_ok"]
            return result

        result["success"] = result["chat_ok"] and result["tool_call_ok"]
        if result["success"]:
            result["details"] = "All tests passed"
        elif result["chat_ok"] and not result["tool_call_ok"]:
            result["details"] = (
                "Chat works but tool calling failed. "
                "This model may not work for main agent tasks. "
                "Consider using it for Light/Consciousness only."
            )
        return result
