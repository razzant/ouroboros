"""Local llama-cpp-python server lifecycle, download, and health manager."""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from ouroboros.platform_layer import (
    IS_MACOS, terminate_process_tree, kill_process_tree,
    subprocess_new_group_kwargs, subprocess_hidden_kwargs,
)

log = logging.getLogger(__name__)

_LOCAL_MODEL_DEFAULT_PORT = 8766

def _get_install_command() -> list:
    """Return the llama-cpp-python pip install command."""
    from ouroboros.platform_layer import pip_install_target_args

    return [sys.executable, "-m", "pip", "install",
            *pip_install_target_args(sys.executable), "--upgrade", "llama-cpp-python[server]"]


def _get_install_env() -> dict:
    """Return pip install env, enabling Metal flags on macOS."""
    env = os.environ.copy()
    if IS_MACOS:
        env["CMAKE_ARGS"] = "-DGGML_METAL=on"
        env["FORCE_CMAKE"] = "1"
    return env


def _get_runtime_hint() -> str:
    """Return a human-readable llama-cpp-python install hint."""
    if IS_MACOS:
        return 'CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python[server]'
    return "pip install llama-cpp-python[server]"


def _with_hidden_subprocess(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Add platform hidden-window flags to subprocess kwargs."""
    hidden = subprocess_hidden_kwargs()
    if hidden:
        kwargs = dict(kwargs)
        existing = kwargs.get("creationflags", 0)
        kwargs["creationflags"] = existing | hidden.get("creationflags", 0)
    return kwargs

_manager: Optional[LocalModelManager] = None
_manager_lock = threading.Lock()


def get_manager() -> LocalModelManager:
    global _manager
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
        self._status = "offline"
        self._error: Optional[str] = None
        self._model_path: Optional[str] = None
        self._port: int = _LOCAL_MODEL_DEFAULT_PORT
        self._context_length: int = 0
        self._model_name: str = ""
        self._download_progress: float = 0.0
        self._stderr_buf: bytes = b""
        # Runtime (llama-cpp-python) install state
        self._runtime_status: str = "unknown"  # unknown | ok | missing | installing | install_ok | install_error
        self._runtime_install_log: str = ""
        # Cancellation flag — set in stop_server() so _run_install() can abort
        # even before _install_proc is assigned, closing the panic-window race.
        self._install_cancelled = threading.Event()

    def get_status(self) -> str:
        if self._proc is not None and self._proc.poll() is not None:
            self._status = "error"
            self._error = f"Server exited with code {self._proc.returncode}"
            self._proc = None
        return self._status

    @property
    def port(self) -> int:
        return self._port

    @property
    def is_running(self) -> bool:
        return self.get_status() == "ready"

    def status_dict(self) -> Dict[str, Any]:
        return {
            "status": self.get_status(),
            "error": self._error,
            "model_path": self._model_path,
            "model_name": self._model_name,
            "context_length": self._context_length,
            "port": self._port,
            "download_progress": self._download_progress,
            "runtime_status": self._runtime_status,
            "runtime_install_log": self._runtime_install_log[-500:] if self._runtime_install_log else "",
        }

    def check_runtime(self) -> bool:
        """Check llama-cpp-python importability and update runtime status."""
        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import llama_cpp"],
                **_with_hidden_subprocess({
                    "capture_output": True,
                    "text": True,
                    "timeout": 15,
                }),
            )
            if probe.returncode == 0:
                self._runtime_status = "ok"
                return True
            else:
                self._runtime_status = "missing"
                return False
        except Exception as exc:
            log.warning("Runtime check failed: %s", exc)
            self._runtime_status = "missing"
            return False

    def install_runtime(self) -> None:
        """Install llama-cpp-python in a tracked background thread."""
        with self._lock:
            if self._runtime_status == "installing":
                log.info("Runtime install already in progress")
                return
            # stop_server may have cancelled a previous lifecycle.
            self._install_cancelled.clear()
            self._runtime_status = "installing"
            self._runtime_install_log = ""

        threading.Thread(
            target=self._run_install, daemon=True, name="llama-install"
        ).start()

    def _run_install(self) -> None:
        """Background install worker."""
        # Check before spawning to close the install_runtime -> Popen race.
        if self._install_cancelled.is_set():
            self._runtime_status = "missing"
            return

        cmd = _get_install_command()
        env = _get_install_env()
        log.info("Installing llama-cpp-python: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                **_with_hidden_subprocess({
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "stdin": subprocess.DEVNULL,
                }),
            )
            self._install_proc = proc

            # stop_server may have run during Popen; kill immediately if so.
            if self._install_cancelled.is_set():
                try:
                    terminate_process_tree(proc)
                except Exception:
                    pass
                self._install_proc = None
                self._runtime_status = "missing"
                return
            output_bytes = b""
            try:
                if proc.stdout:
                    fd = proc.stdout.fileno()
                    while True:
                        chunk = os.read(fd, 4096)
                        if not chunk:
                            break
                        output_bytes = (output_bytes + chunk)[-4096:]
            except Exception:
                pass
            proc.wait()
            self._install_proc = None
            output = output_bytes.decode("utf-8", errors="replace")
            self._runtime_install_log = output

            if proc.returncode == 0:
                if self.check_runtime():
                    self._runtime_status = "install_ok"
                    log.info("llama-cpp-python installed successfully")
                else:
                    self._runtime_status = "install_error"
                    self._runtime_install_log += "\nInstall succeeded but import still fails."
                    log.warning("llama-cpp-python install ran but import still fails")
            else:
                self._runtime_status = "install_error"
                log.error("llama-cpp-python install failed (rc=%d)", proc.returncode)
        except Exception as exc:
            self._install_proc = None
            self._runtime_status = "install_error"
            self._runtime_install_log = str(exc)
            log.error("llama-cpp-python install exception: %s", exc)

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
        self,
        source: str,
        filename: str = "",
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> str:
        """Resolve/download a .gguf; split GGUFs download all shards from shard 1."""
        if os.path.isfile(source):
            log.info("Using local model file: %s", source)
            return source

        if source.startswith("/") or source.startswith("~"):
            expanded = os.path.expanduser(source)
            if os.path.isfile(expanded):
                return expanded
            raise FileNotFoundError(f"Local model file not found: {expanded}")

        try:
            from huggingface_hub import hf_hub_download
            from tqdm.auto import tqdm as _base_tqdm
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for downloading models. "
                "Install with: pip install huggingface_hub"
            )

        if not filename:
            raise ValueError(
                "filename is required when source is a HuggingFace repo ID. "
                "Example: filename='model-Q4_K_M.gguf' or 'quant/model-00001-of-00003.gguf'"
            )

        # Common UX fix: resolve basename-only HF shard filenames to subfolders.
        filename = self._resolve_hf_path(source, filename)

        subfolder, basename = self._normalize_hf_filename(filename)
        shard_info = self._detect_shard_info(basename)

        if shard_info is not None:
            _prefix, shard_num, total_shards, _shard_w, _total_w, _suffix = shard_info
            if shard_num != 1:
                first_basename = next(self._all_shard_basenames(_prefix, total_shards, _suffix, _shard_w, _total_w))
                first_filename = f"{subfolder}/{first_basename}" if subfolder else first_basename
                raise ValueError(
                    f"Split GGUF: shard {shard_num} of {total_shards} specified. "
                    f"Please use the first shard to start the server. "
                    f"Change filename to: '{first_filename}'"
                )

        self._status = "downloading"
        self._download_progress = 0.0
        log.info("Downloading %s/%s from HuggingFace...", source, filename)

        # For split GGUFs, report global progress across all shards.
        manager_ref = self

        def _make_progress_tqdm(shard_index: int, total_shards_count: int):
            class _ProgressTqdm(_base_tqdm):
                def update(self, n=1):
                    result = super().update(n)
                    try:
                        if self.total and self.total > 0:
                            file_fraction = min(self.n / self.total, 1.0)
                            global_fraction = (shard_index - 1 + file_fraction) / total_shards_count
                            manager_ref._download_progress = global_fraction
                            if progress_cb is not None:
                                progress_cb(global_fraction)
                    except Exception:
                        pass
                    return result
            return _ProgressTqdm

        def _download_one(basename_: str, subfolder_: Optional[str], shard_idx: int, total: int) -> str:
            """Download one file; return the local path."""
            return hf_hub_download(
                repo_id=source,
                filename=basename_,
                subfolder=subfolder_ or None,
                resume_download=True,
                tqdm_class=_make_progress_tqdm(shard_idx, total),
            )

        try:
            if shard_info is not None:
                _prefix, _shard_num, total_shards, _shard_w, _total_w, _suffix = shard_info
                first_path: Optional[str] = None
                for idx, shard_basename in enumerate(
                    self._all_shard_basenames(_prefix, total_shards, _suffix, _shard_w, _total_w), start=1
                ):
                    log.info(
                        "Downloading shard %d/%d: %s/%s",
                        idx, total_shards, subfolder or "", shard_basename,
                    )
                    p = _download_one(shard_basename, subfolder, idx, total_shards)
                    if idx == 1:
                        first_path = p
                path = first_path  # type: ignore[assignment]
            else:
                path = _download_one(basename, subfolder, 1, 1)

            self._download_progress = 1.0
            if progress_cb:
                progress_cb(1.0)
            log.info("Model downloaded to: %s", path)
            return path
        except Exception as e:
            self._status = "error"
            self._error = f"Download failed: {e}"
            raise

    def start_server(
        self,
        model_path: str,
        port: int = _LOCAL_MODEL_DEFAULT_PORT,
        n_gpu_layers: int = -1,
        n_ctx: int = 0,
        chat_format: str = "",
    ) -> None:
        """Start the server; rechecks runtime as a safety net before Popen."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("Local model server is already running")

            self._model_path = model_path
            self._port = port
            self._status = "loading"
            self._error = None

            python = sys.executable
            cmd = [
                python, "-m", "llama_cpp.server",
                "--model", model_path,
                "--port", str(port),
                "--n_gpu_layers", str(n_gpu_layers),
            ]
            if chat_format:
                cmd.extend(["--chat_format", chat_format])
            effective_ctx = n_ctx if n_ctx > 0 else 16384
            self._context_length = effective_ctx
            cmd.extend(["--n_ctx", str(effective_ctx)])

            log.info("Starting local model server: %s", " ".join(cmd))

            try:
                probe = subprocess.run(
                    [python, "-c", "import llama_cpp"],
                    **_with_hidden_subprocess({
                        "capture_output": True,
                        "text": True,
                        "timeout": 15,
                    }),
                )
            except Exception as exc:
                self._status = "error"
                self._error = f"Failed to verify llama-cpp-python installation: {exc}"
                raise RuntimeError(self._error) from exc
            if probe.returncode != 0:
                self._status = "error"
                details = (probe.stderr or probe.stdout or "").strip()
                hint = _get_runtime_hint()
                self._error = f"llama-cpp-python is not installed or failed to import. Install with: {hint}"
                if details:
                    self._error += f": {details[-500:]}"
                raise RuntimeError(self._error)

            try:
                _popen_kwargs = dict(
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                )
                _popen_kwargs.update(subprocess_new_group_kwargs())
                self._proc = subprocess.Popen(cmd, **_with_hidden_subprocess(_popen_kwargs))
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

            self._stderr_buf = b""
            threading.Thread(
                target=self._drain_stderr, daemon=True, name="local-model-stderr"
            ).start()

        threading.Thread(
            target=self._wait_for_healthy, daemon=True, name="local-model-health"
        ).start()

    def _drain_stderr(self) -> None:
        """Drain stderr to avoid pipe deadlock and retain a diagnostic tail."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        buf = b""
        try:
            fd = proc.stderr.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                buf = (buf + chunk)[-2048:]
        except Exception:
            pass
        self._stderr_buf = buf

    def _wait_for_healthy(self, timeout: float = 300.0) -> None:
        """Poll the local server until healthy or timed out."""
        proc = self._proc
        start = time.time()
        while time.time() - start < timeout:
            if self._proc is None or self._proc.poll() is not None:
                self._status = "error"
                rc = self._proc.returncode if self._proc else "?"
                stderr_tail = ""
                if self._stderr_buf:
                    try:
                        stderr_tail = self._stderr_buf.decode("utf-8", errors="replace")[-500:]
                    except Exception:
                        pass
                self._error = f"Server process exited during startup (code {rc})"
                if stderr_tail:
                    self._error += f": {stderr_tail}"
                self._proc = None
                return
            try:
                health = self.health_check()
                if health.get("ok"):
                    from ouroboros.config import DATA_DIR
                    from ouroboros.server_process import record_service_binding

                    with self._lock:
                        # The health response belongs only to this live manager
                        # generation; a concurrent stop/restart cannot publish it.
                        if proc is None or self._proc is not proc or proc.poll() is not None:
                            return
                        try:
                            self._service_binding = record_service_binding(
                                DATA_DIR, "local_model", "127.0.0.1", self._port, pid=proc.pid)
                        except Exception:
                            self._service_binding = None
                            log.warning("Local model service binding unavailable; health remains ready", exc_info=True)
                        self._status = "ready"
                        self._context_length = health.get("context_length", 0)
                        self._model_name = health.get("model_name", "")
                    log.info(
                        "Local model server ready (ctx=%d, model=%s)",
                        self._context_length, self._model_name,
                    )
                    return
            except Exception:
                pass
            time.sleep(2.0)

        self._status = "error"
        self._error = f"Server failed to become healthy within {timeout}s"
        log.error(self._error)

    def stop_server(self) -> None:
        """Stop the local model server subprocess and any ongoing install."""
        # Abort install even before _install_proc exists.
        self._install_cancelled.set()

        with self._lock:
            proc = self._proc
            binding, self._service_binding = self._service_binding, None
            install_proc = self._install_proc
            self._proc = None
            self._install_proc = None
            self._status = "offline"
            self._error = None
            self._context_length = 0
            self._model_name = ""
            self._stderr_buf = b""

        if install_proc is not None:
            log.info("Terminating ongoing llama-cpp-python install (pid=%s)...", install_proc.pid)
            try:
                terminate_process_tree(install_proc)
                install_proc.wait(timeout=5)
            except Exception:
                try:
                    kill_process_tree(install_proc)
                except Exception:
                    pass

        if proc is None:
            return

        log.info("Stopping local model server (pid=%s)...", proc.pid)
        terminate_process_tree(proc)

        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("Local model server did not exit, force-killing")
            kill_process_tree(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if binding is not None and proc.poll() is not None:
            from ouroboros.config import DATA_DIR
            from ouroboros.server_process import clear_service_binding

            clear_service_binding(DATA_DIR, "local_model", binding)

    def health_check(self) -> Dict[str, Any]:
        """Query local server health and loaded-model info."""
        import requests

        from ouroboros.utils import in_worker_process

        url = f"http://127.0.0.1:{self._port}/v1/models"
        with requests.Session() as session:
            if in_worker_process():
                session.trust_env = False  # fork-safe + localhost never needs a proxy
            resp = session.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if not models:
            return {"ok": False, "error": "No models loaded"}

        model_info = models[0]
        ctx = model_info.get("meta", {}).get("n_ctx_train", 0)
        if not ctx:
            ctx = model_info.get("context_window", 0)

        return {
            "ok": True,
            "model_name": model_info.get("id", "unknown"),
            "context_length": ctx,
        }

    def get_context_length(self) -> int:
        """Return cached context length, querying the server if needed."""
        if self._context_length > 0:
            return self._context_length
        try:
            info = self.health_check()
            self._context_length = info.get("context_length", 4096)
        except Exception:
            self._context_length = 4096
        return self._context_length

    def test_tool_calling(self) -> Dict[str, Any]:
        """Run basic chat and tool-call checks against the local server."""
        from openai import OpenAI
        from ouroboros.usage_accounting import (
            AttemptRequest,
            UsageAccountingError,
            UsageScope,
            current_usage_scope,
            execute_physical_attempt,
            usage_scope,
        )

        client = OpenAI(
            base_url=f"http://127.0.0.1:{self._port}/v1",
            api_key="local",
            max_retries=0,
        )

        def _dispatch_probe(payload: Dict[str, Any]) -> Any:
            request = AttemptRequest(
                model="local-model",
                provider="local",
                prompt_tokens_estimate=max(0, len(str(payload)) // 4),
                max_completion_tokens=int(payload.get("max_tokens") or 0),
                source="capability_probe.local_model",
            )

            def _send() -> Any:
                return execute_physical_attempt(
                    request,
                    lambda: client.chat.completions.create(**payload),
                )

            if current_usage_scope() is not None:
                return _send()
            with usage_scope(UsageScope(
                task_id="system:capability_probe",
                root_task_id="system:capability_probe",
                category="capability_probe",
                source="capability_probe.local_model",
            )):
                return _send()

        result: Dict[str, Any] = {
            "success": False,
            "chat_ok": False,
            "tool_call_ok": False,
            "details": "",
            "tokens_per_sec": 0.0,
        }

        try:
            t0 = time.time()
            resp = _dispatch_probe({
                "model": "local-model",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 32,
            })
            elapsed = time.time() - t0
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
            resp = _dispatch_probe({
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
