"""Adapter-owned isolated Ouroboros server for CyberGym runs.

The CyberGym executor talks to an Ouroboros HTTP gateway.  This wrapper makes
that gateway an owned, throwaway server instead of silently attaching to the
operator's live process.  It deliberately reuses the common
``IsolatedServer`` lifecycle and changes only the adapter boundary: the
selected rootless Docker socket is injected into the isolated server process,
while the repository clone and settings/data roots remain run-local.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from devtools.benchmarks.cybergym.cybergym_sidecar import (
    DockerHostRef,
    is_placeholder_api_key,
    resolve_rootless_docker_host,
)


class CyberGymServerError(RuntimeError):
    """Typed refusal for isolated-server preparation or attestation."""


# Positive evidence only: ``--state-dir`` exists specifically to keep hot
# mutable lock/state traffic on a local disk. Unknown/FUSE/network types need
# the explicit risk override rather than silently recreating the CephFS stall.
_LOCAL_STATE_FS_TYPES = frozenset({
    "btrfs",
    "ext2",
    "ext3",
    "ext4",
    "hfs",
    "hfsplus",
    "ntfs",
    "ntfs3",
    "overlay",
    "xfs",
    "zfs",
})


def _mount_fs_type(path: pathlib.Path, mounts_text: str | None = None) -> str:
    """Return the filesystem type serving ``path`` per the kernel mount table.

    The longest mount-point prefix of the (already resolved) path wins, so a
    not-yet-created state dir is classified by its future location.  An empty
    string means the type could not be determined (no readable /proc/mounts,
    e.g. macOS); callers must treat that as unknown, not as proof of local.
    """
    if mounts_text is None:
        try:
            mounts_text = pathlib.Path("/proc/mounts").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return ""
    probe = str(path)
    best_length = 0
    best_type = ""
    for line in mounts_text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point = fields[1].replace("\\040", " ").rstrip("/") or "/"
        if mount_point != "/" and probe != mount_point and not probe.startswith(mount_point + "/"):
            continue
        if len(mount_point) > best_length:
            best_length = len(mount_point)
            best_type = fields[2]
    return best_type


GitRunner = Callable[[Sequence[str], pathlib.Path], int]
ServerFactory = Callable[..., Any]


def _run_git(argv: Sequence[str], cwd: pathlib.Path) -> int:
    """Run a fixed argv git operation without a shell or output logging."""
    try:
        result = subprocess.run(
            ["git", *list(argv)],
            cwd=str(cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return 127
    return int(result.returncode)


def _run_git_output(argv: Sequence[str], cwd: pathlib.Path) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *list(argv)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return 127, ""
    return int(result.returncode), (result.stdout or "").strip()


class _RootlessIsolatedServer:
    """Small subclass that adds the explicit rootless socket to IsolatedServer._env."""

    def __init__(
        self,
        *args: Any,
        docker_host: DockerHostRef,
        provider_key: str = "",
        provider_key_env: str = "OPENROUTER_API_KEY",
        worker_start_method: str = "spawn",
        expected_settings_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        from devtools.benchmarks.common.server_runner import IsolatedServer

        self._docker_host = docker_host
        self._provider_key = str(provider_key or "").strip()
        self._provider_key_env = str(provider_key_env or "").strip()
        self._worker_start_method = str(worker_start_method or "").strip().lower()
        if self._worker_start_method not in {"spawn", "forkserver"}:
            raise CyberGymServerError(
                "worker_start_method must avoid fork-from-thread in the isolated server"
            )
        self._delegate = IsolatedServer(
            *args,
            expected_settings_sha256=expected_settings_sha256,
            **kwargs,
        )
        # Keep the original bound method before ``start`` installs our
        # adapter-owned shim on the delegate.  Calling ``delegate._env`` from
        # the shim after that assignment would recurse forever.
        self._base_env = self._delegate._env  # noqa: SLF001 - lifecycle seam

    def _env(self) -> dict[str, str]:
        env = self._base_env()
        env["DOCKER_HOST"] = self._docker_host.value
        if self._provider_key:
            env[self._provider_key_env] = self._provider_key
        env["OUROBOROS_WORKER_START_METHOD"] = self._worker_start_method
        return env

    def start(self, *args: Any, **kwargs: Any) -> Any:
        # Bind the subclass environment method for the existing server
        # implementation without copying its process/attestation lifecycle.
        self._delegate._env = self._env  # type: ignore[method-assign]  # noqa: SLF001
        return self._delegate.start(*args, **kwargs)

    def stop(self) -> None:
        self._delegate.stop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class CyberGymIsolatedServer:
    """Prepare and own one isolated Ouroboros server for a CyberGym campaign.

    ``run_root`` must already be an adapter-approved fresh output directory.
    The wrapper refuses to reuse an existing clone/data child, preventing a
    resumed run from silently changing the server identity or settings.
    """

    def __init__(
        self,
        seed_repo: pathlib.Path | str,
        run_root: pathlib.Path | str,
        applied_settings: pathlib.Path | str,
        docker_host: str | DockerHostRef,
        *,
        expected_commit: str = "",
        provider_key: str = "",
        provider_key_env: str = "OPENROUTER_API_KEY",
        server_factory: ServerFactory | None = None,
        git_runner: GitRunner | None = None,
        expected_settings_sha256: str = "",
        state_dir: pathlib.Path | str | None = None,
        allow_network_state_dir: bool = False,
    ) -> None:
        self.seed_repo = pathlib.Path(seed_repo).expanduser().resolve(strict=False)
        self.run_root = pathlib.Path(run_root).expanduser().resolve(strict=False)
        self.applied_settings = pathlib.Path(applied_settings).expanduser().resolve(strict=False)
        self.docker_host = resolve_rootless_docker_host(docker_host)
        self.expected_commit = str(expected_commit or "").strip().lower()
        self.provider_key = str(provider_key or "").strip()
        self.provider_key_env = str(provider_key_env or "").strip()
        self.worker_start_method = "spawn"
        self.expected_settings_sha256 = str(expected_settings_sha256 or "").strip().lower()
        if self.expected_settings_sha256 and (
            len(self.expected_settings_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.expected_settings_sha256)
        ):
            raise CyberGymServerError("expected_settings_sha256 must be a SHA-256 digest")
        self._server_factory = server_factory
        self._git_runner = git_runner or _run_git
        self.clone_root = self.run_root / "ouroboros-clone"
        self._server: Any | None = None
        self._prepared = False
        self._started = False
        self.attestation: dict[str, Any] = {}
        self.state_export: dict[str, Any] = {}
        self._copied_settings_sha256: str | None = None

        if not self.seed_repo.is_dir():
            raise CyberGymServerError("seed_repo must be an existing directory")
        if not re.fullmatch(r"[0-9a-f]{40}", self.expected_commit):
            raise CyberGymServerError("expected_commit is required for isolated-server provenance")
        if self.provider_key_env != "OPENROUTER_API_KEY":
            raise CyberGymServerError("provider_key_env must be OPENROUTER_API_KEY")
        if self.provider_key and is_placeholder_api_key(self.provider_key):
            raise CyberGymServerError("provider_key must not be a placeholder")
        if not self.run_root.is_absolute() or self.run_root == pathlib.Path("/"):
            raise CyberGymServerError("run_root must be a non-root absolute path")
        try:
            self.run_root.relative_to(self.seed_repo)
            raise CyberGymServerError("run_root must not be inside seed_repo")
        except ValueError:
            pass
        try:
            self.seed_repo.relative_to(self.run_root)
            raise CyberGymServerError("run_root must not contain seed_repo")
        except ValueError:
            pass
        if state_dir is not None:
            # The mutable isolate state (state/, logs/, task_results/, locks)
            # may live on a fast local disk while the run root stays on the
            # durable artifact filesystem.  The state dir is operator-chosen;
            # keep the same overlap discipline as for run_root so a mistaken
            # path can never alias the seed or the artifact tree.
            resolved_state = pathlib.Path(state_dir).expanduser().resolve(strict=False)
            if not resolved_state.is_absolute() or resolved_state == pathlib.Path("/"):
                raise CyberGymServerError("state_dir must be a non-root absolute path")
            try:
                resolved_state.relative_to(self.seed_repo)
                raise CyberGymServerError("state_dir must not be inside seed_repo")
            except ValueError:
                pass
            try:
                self.seed_repo.relative_to(resolved_state)
                raise CyberGymServerError("state_dir must not contain seed_repo")
            except ValueError:
                pass
            try:
                resolved_state.relative_to(self.run_root)
                raise CyberGymServerError("state_dir must not be inside run_root")
            except ValueError:
                pass
            try:
                self.run_root.relative_to(resolved_state)
                raise CyberGymServerError("state_dir must not contain run_root")
            except ValueError:
                pass
            # The same discipline the executor applies to its roots: an
            # operator-chosen state dir must never alias the live data/repo
            # roots of this host.
            try:
                from devtools.benchmarks.common.run_roots import (
                    live_data_roots,
                    live_repo_roots,
                )

                live_roots = [
                    pathlib.Path(item).expanduser().resolve(strict=False)
                    for item in (*live_data_roots(), *live_repo_roots())
                ]
            except (ImportError, OSError, ValueError):
                live_roots = []
            for live_root in live_roots:
                try:
                    resolved_state.relative_to(live_root)
                    raise CyberGymServerError("state_dir must not be inside a live Ouroboros root")
                except ValueError:
                    pass
                try:
                    live_root.relative_to(resolved_state)
                    raise CyberGymServerError("state_dir must not contain a live Ouroboros root")
                except ValueError:
                    pass
            if not isinstance(allow_network_state_dir, bool):
                raise CyberGymServerError("allow_network_state_dir must be a boolean")
            state_fs_type = _mount_fs_type(resolved_state)
            if state_fs_type not in _LOCAL_STATE_FS_TYPES:
                if not allow_network_state_dir:
                    raise CyberGymServerError(
                        "state_dir must be on a known local filesystem; observed "
                        f"type {state_fs_type or 'unknown'!r} (pass --allow-network-state-dir to "
                        "accept the lock-latency risk)"
                    )
                print(
                    "[cybergym] WARNING: state_dir filesystem is "
                    f"{state_fs_type or 'unknown'!r}; isolated-server lock "
                    "latency there may stall runs",
                    file=sys.stderr,
                )
            self.state_dir: pathlib.Path | None = resolved_state
            self.data_root = resolved_state / "ouroboros-data"
        else:
            self.state_dir = None
            self.data_root = self.run_root / "ouroboros-data"
        self.settings_path = self.data_root / "settings.json"
        if not self.applied_settings.is_file():
            raise CyberGymServerError("applied_settings must name an existing JSON file")
        if self.clone_root.exists() or self.data_root.exists():
            raise CyberGymServerError("isolated server child paths already exist; use a fresh run root")

    def _git(self, argv: Sequence[str], cwd: pathlib.Path) -> int:
        return int(self._git_runner(argv, cwd))

    def _git_value(self, argv: Sequence[str], cwd: pathlib.Path, label: str) -> str:
        # Keep mutation injectable, but use a separate read-only probe for the
        # commit/status values that are part of provenance.
        code, value = _run_git_output(argv, cwd)
        if code != 0 or not value:
            raise CyberGymServerError(f"git probe failed: {label}")
        return value

    def _clone(self) -> None:
        if self._git(("clone", "--no-hardlinks", "--quiet", str(self.seed_repo), str(self.clone_root)), self.run_root) != 0:
            raise CyberGymServerError("unable to clone the pinned Ouroboros seed")
        # The local clone must not retain a remote back to the operator checkout.
        self._git(("remote", "remove", "origin"), self.clone_root)
        commit = self._git_value(("rev-parse", "HEAD"), self.clone_root, "HEAD").lower()
        if self.expected_commit and commit != self.expected_commit:
            raise CyberGymServerError("isolated Ouroboros clone commit does not match the pinned seed")
        code, status = _run_git_output(("status", "--porcelain=v1", "--untracked-files=all"), self.clone_root)
        if code != 0:
            raise CyberGymServerError("isolated Ouroboros clone status probe failed")
        if status:
            raise CyberGymServerError("isolated Ouroboros clone is dirty")

    def _copy_settings(self) -> None:
        try:
            raw = self.applied_settings.read_bytes()
        except OSError as exc:
            raise CyberGymServerError("applied settings are unreadable") from exc
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        if self.expected_settings_sha256 and observed_sha256 != self.expected_settings_sha256:
            raise CyberGymServerError("applied settings digest changed before isolated copy")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CyberGymServerError("applied settings are unreadable") from exc
        if not isinstance(payload, Mapping):
            raise CyberGymServerError("applied settings must be a JSON object")
        self.data_root.mkdir(parents=True, exist_ok=False)
        (self.data_root / "state").mkdir(parents=True, exist_ok=False)
        temporary = self.settings_path.with_name(self.settings_path.name + f".tmp.{os.getpid()}")
        try:
            # Preserve the producer's exact bytes. Re-serializing here would
            # create a new unaudited snapshot and reopen the producer→copy race.
            temporary.write_bytes(raw)
            os.replace(temporary, self.settings_path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise CyberGymServerError("unable to copy applied settings into isolated data") from exc
        self.settings_path.chmod(0o600)
        self._copied_settings_sha256 = observed_sha256
        try:
            from supervisor import state as supervisor_state

            sentinel = self.data_root / supervisor_state.ISOLATED_BENCHMARK_SENTINEL
        except (ImportError, AttributeError):
            sentinel = self.data_root / ".ouroboros_isolated_benchmark"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("isolated CyberGym server\n", encoding="utf-8")

    def prepare(self) -> "CyberGymIsolatedServer":
        if self._prepared:
            return self
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._clone()
        self._copy_settings()
        self._prepared = True
        return self

    def start(self, *, ready_timeout: float = 180) -> "CyberGymIsolatedServer":
        if self._started:
            return self
        self.prepare()
        factory = self._server_factory
        factory_kwargs = {
            "docker_host": self.docker_host,
            "provider_key": self.provider_key,
            "provider_key_env": self.provider_key_env,
        }
        if factory is None:
            factory = _RootlessIsolatedServer
            # CyberGym's applied snapshot is authoritative. The default wrapper
            # adds the selected OpenRouter key after the common runner removes
            # ambient provider/model settings. Custom test factories retain their
            # historical call contract and do not need this implementation detail.
            factory_kwargs["settings_authoritative_env"] = True
            factory_kwargs["expected_settings_sha256"] = self._copied_settings_sha256
            factory_kwargs["worker_start_method"] = self.worker_start_method
        self._server = factory(
            self.clone_root,
            self.data_root,
            self.settings_path,
            **factory_kwargs,
        )
        try:
            self._server.start(ready_timeout=ready_timeout)
            observed = dict(getattr(self._server, "attestation", {}) or {})
            observed_head = str(observed.get("repo_head") or "").lower()
            if self.expected_commit and observed_head != self.expected_commit:
                raise CyberGymServerError("isolated Ouroboros runtime attested a different commit")
            self.attestation = {
                "base_url": str(self._server.base_url),
                "docker_host": self.docker_host.value,
                "clone_root": str(self.clone_root),
                "data_root": str(self.data_root),
                "state_dir": str(self.state_dir) if self.state_dir is not None else "",
                "settings_path": str(self.settings_path),
                "settings_sha256": self._copied_settings_sha256,
                "worker_start_method": self.worker_start_method,
                "repo_head": observed_head,
                "runtime": observed,
            }
            self._started = True
            return self
        except BaseException:
            self.close()
            raise

    @property
    def base_url(self) -> str:
        if self._server is None or not self._started:
            raise CyberGymServerError("isolated Ouroboros server is not started")
        return str(self._server.base_url)

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()
        self._started = False

    _STATE_EXPORT_DIRS = ("state", "logs", "task_results", "memory")
    _STATE_EXPORT_FILES = ("settings.json",)

    @staticmethod
    def _export_ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
        }

    def export_state_snapshot(self) -> dict[str, Any]:
        """Best-effort mirror of the audit-relevant isolate state to the run root.

        With an external ``state_dir`` the mutable state lives outside the
        durable artifact tree.  After the server stops we mirror the small
        audit surface (state/, logs/, task_results/, memory/, settings.json)
        back to ``run_root/ouroboros-data`` so the append-only run root keeps
        a complete custody record.  Heavy wire evidence under
        ``observability/`` intentionally stays on the external state dir; the
        run manifest documents both locations.  The export never raises: a
        failure is reported in the returned receipt and in ``state_export``.
        """
        if self.state_dir is None:
            self.state_export = {
                "status": "not_needed",
                "ok": True,
                "reason": "state_dir_in_run_root",
            }
            return self.state_export
        destination = self.run_root / "ouroboros-data"
        report: dict[str, Any] = {
            "status": "exported",
            "ok": True,
            "source": str(self.data_root),
            "destination": str(destination),
            "items": {},
            "skipped": ["observability"],
        }
        try:
            if not self.data_root.is_dir():
                report.update(
                    {"status": "no_state", "ok": False, "reason": "data_root_missing"}
                )
                return report
            if destination.exists():
                report.update(
                    {"status": "skipped", "ok": False, "reason": "destination_exists"}
                )
                return report
            destination.mkdir(parents=True, exist_ok=False)
            for name in self._STATE_EXPORT_DIRS:
                source = self.data_root / name
                if not source.is_dir():
                    continue
                shutil.copytree(
                    source,
                    destination / name,
                    symlinks=True,
                    ignore=self._export_ignore,
                )
                report["items"][name] = "dir"
            for name in self._STATE_EXPORT_FILES:
                source = self.data_root / name
                if not source.is_file():
                    continue
                shutil.copy2(source, destination / name)
                (destination / name).chmod(0o600)
                report["items"][name] = "file"
            try:
                from supervisor import state as supervisor_state

                sentinel_name = supervisor_state.ISOLATED_BENCHMARK_SENTINEL
            except (ImportError, AttributeError):
                sentinel_name = ".ouroboros_isolated_benchmark"
            sentinel = self.data_root / sentinel_name
            if sentinel.is_file():
                shutil.copy2(sentinel, destination / sentinel_name)
                report["items"][sentinel_name] = "file"
        except OSError as exc:
            report.update(
                {"status": "error", "ok": False, "error_type": type(exc).__name__}
            )
        self.state_export = report
        return report

    def close(self) -> None:
        self.stop()
        if self.state_dir is not None and self._prepared:
            self.export_state_snapshot()

    def __enter__(self) -> "CyberGymIsolatedServer":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def state_layout_manifest(isolated_server: Any) -> dict[str, Any]:
    """Document where the isolate's mutable state lives for this run.

    With ``--state-dir`` the hot mutable state (state/, logs/, task_results/,
    locks, observability wire evidence) lives on the operator-chosen local
    disk while the append-only run root keeps the durable artifacts
    (workspaces/, checkpoints/, result_index.jsonl, run_manifest.json).  At
    finalize the server mirrors the small audit surface back to
    ``run_root/ouroboros-data`` on a best-effort basis; the receipt lands in
    ``extra.state_export`` after cleanup.
    """
    state_dir = getattr(isolated_server, "state_dir", None)
    data_root = getattr(isolated_server, "data_root", None)
    external = state_dir is not None
    return {
        "mode": "external_state_dir" if external else "run_root",
        "state_dir": str(state_dir) if external else "",
        "data_root": str(data_root) if data_root is not None else "",
        "run_root_surfaces": [
            "workspaces/",
            "checkpoints/",
            "attestations/",
            "private/",
            "result_index.jsonl",
            "claims.jsonl",
            "run_manifest.json",
        ],
        "state_dir_surfaces": (
            [
                "ouroboros-data/state/",
                "ouroboros-data/logs/",
                "ouroboros-data/task_results/",
                "ouroboros-data/memory/",
                "ouroboros-data/observability/",
                "ouroboros-data/settings.json",
            ]
            if external
            else []
        ),
        "finalize_mirror": (
            "best_effort_to_run_root/ouroboros-data" if external else "not_needed"
        ),
        "observability_blobs_mirrored": False,
    }


__all__ = ["CyberGymIsolatedServer", "CyberGymServerError", "state_layout_manifest"]
