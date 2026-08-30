"""Docker, container, and network machinery for the CyberGym executor.

Extracted from ``cybergym_executor.py`` (which re-imports every public name, so
existing imports keep working) to keep each module inside the size ratchet.
This is the lowest adapter layer below the lifecycle module and the executor
assembly: it imports only from the pinned wire, sidecar, and adapter helpers
and never from the lifecycle module or executor, so no import cycle is
introduced.  ``_DockerRuntimeMixin`` collects the docker/container/network/
workspace/attestation-probe methods that are mixed into ``CyberGymExecutor``;
they are dispatched on ``self`` at runtime, so instance state and executor-owned
methods remain available even though they are declared elsewhere.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from devtools.benchmarks.cybergym.cybergym_adapter import (
    OFFICIAL_MODEL,
    TaskSpec,
)
from devtools.benchmarks.cybergym.cybergym_sidecar import (
    API_KEY_ENV,
    CleanupPlan,
    DockerHostRef,
    NetworkPlan,
    WorkspaceCommandSpec,
    build_connectivity_probe_plan,
    build_network_create_argv,
    build_workspace_argv,
    cleanup_argv,
    validate_cleanup_observation,
)
from devtools.benchmarks.cybergym.cybergym_wire import (
    ExecutorFailure,
    _unwrap_http_json,
    urllib_json,
)


_HEX40 = re.compile(r"^[0-9a-f]{40}$")


_GATEWAY_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


_EXPECTED_MODEL = OFFICIAL_MODEL


_SAFE_ENV_NAMES = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "XDG_RUNTIME_DIR",
    "LANG",
    "LC_ALL",
    "TMPDIR",
)


_GENERATED_INPUT_EXCLUDES = (
    "/repo-vul.tar.gz",
    "/src-vul/",
    "/submissions/",
)


_GENERATED_TRACKED_INPUTS = ("README.md", "description.txt", "submit.sh")


_WORKSPACE_BACKEND_ALIAS_NAME = "workspace"


_WORKSPACE_BACKEND_ALIAS_TARGET = "."


_WORKSPACE_BACKEND_ALIAS_EXCLUDE = f"/{_WORKSPACE_BACKEND_ALIAS_NAME}"


_WORKSPACE_BACKEND_ALIAS_SCHEMA = "ouroboros.benchmark.cybergym.workspace_backend_alias.v1"


_SERVER_HTTP_SCRIPT = r'''
import base64, json, os, urllib.error, urllib.request

method = os.environ.get("CYBERGYM_HTTP_METHOD", "GET")
path = os.environ.get("CYBERGYM_HTTP_PATH", "")
port = os.environ.get("CYBERGYM_HTTP_PORT", "8666")
body_text = os.environ.get("CYBERGYM_HTTP_BODY_B64", "")
try:
    body = base64.b64decode(body_text.encode("ascii"), validate=True) if body_text else None
except Exception:
    print(json.dumps({"status_code": 0, "transport_error": "invalid_body"}))
    raise SystemExit(17)
headers = {"Accept": "application/json"}
if body is not None:
    headers["Content-Type"] = "application/json"
if os.environ.get("CYBERGYM_HTTP_AUTH") == "1":
    headers["X-API-Key"] = os.environ.get("CYBERGYM_API_KEY", "")
request = urllib.request.Request(
    "http://127.0.0.1:" + port + path,
    data=body,
    headers=headers,
    method=method,
)
try:
    response = urllib.request.urlopen(request, timeout=float(os.environ.get("CYBERGYM_HTTP_TIMEOUT", "30")))
    raw = response.read(4_000_000)
    status = int(response.status)
except urllib.error.HTTPError as exc:
    raw = exc.read(4_000_000)
    status = int(exc.code)
except Exception:
    print(json.dumps({"status_code": 0, "transport_error": "request_failed"}))
    raise SystemExit(18)
try:
    parsed = json.loads(raw.decode("utf-8", errors="replace"))
except Exception:
    parsed = {"non_json": True}
print(json.dumps({"status_code": status, "body": parsed}, separators=(",", ":")))
'''


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """Small subprocess result accepted by the injected command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult: ...


class HttpRunner(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        *,
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60,
    ) -> Any: ...


def run_command(
    argv: Sequence[str],
    *,
    cwd: pathlib.Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> CommandResult:
    """Run an argv list without a shell and return bounded text output."""

    try:
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(124, stdout, stderr or "timeout")
    except OSError as exc:
        return CommandResult(127, "", f"{type(exc).__name__}: {exc}")
    return CommandResult(proc.returncode, proc.stdout or "", proc.stderr or "")


def _json_command(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    cwd: pathlib.Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 60,
) -> Any:
    result = runner(argv, cwd=cwd, env=env, timeout=timeout)
    if result.returncode != 0:
        raise ExecutorFailure(f"command failed ({result.returncode}): {argv[0]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutorFailure(f"command returned invalid JSON: {argv[0]}") from exc


def _safe_abs(value: pathlib.Path | str, name: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve(strict=False)
    if not path.is_absolute() or path == pathlib.Path("/"):
        raise ExecutorFailure(f"{name} must be a non-root absolute path")
    return path


def _inside(path: pathlib.Path, root: pathlib.Path, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ExecutorFailure(f"{name} is outside its approved root") from exc


def _paths_overlap(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Return whether either resolved path contains the other."""

    left = pathlib.Path(left).expanduser().resolve(strict=False)
    right = pathlib.Path(right).expanduser().resolve(strict=False)
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _image_digest(value: str, name: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        raise ExecutorFailure(f"{name} must be a resolved sha256 digest")
    return text


def _pinned_image_ref(image: str, digest: str, name: str) -> str:
    """Return an immutable Docker image reference, rejecting digest drift."""

    text = str(image or "").strip()
    if not text or any(char in text for char in " \t\r\n;,"):
        raise ExecutorFailure(f"{name} is not a safe image reference")
    if "@" in text:
        base, supplied = text.rsplit("@", 1)
        if supplied != digest:
            raise ExecutorFailure(f"{name} digest conflicts with its configured digest")
        return text
    return f"{text}@{digest}"


def _observed_image_matches(observation: Mapping[str, Any], digest: str) -> bool:
    values: set[str] = set()
    config = observation.get("Config")
    if isinstance(config, Mapping):
        image = config.get("Image")
        if isinstance(image, str):
            values.add(image)
        repo_digests = config.get("RepoDigests")
        if isinstance(repo_digests, Sequence) and not isinstance(repo_digests, (str, bytes)):
            values.update(
                item.rsplit("@", 1)[-1]
                for item in repo_digests
                if isinstance(item, str) and "@" in item
            )
    image = observation.get("Image")
    if isinstance(image, str):
        values.add(image)
    repo_digests = observation.get("RepoDigests")
    if isinstance(repo_digests, Sequence) and not isinstance(repo_digests, (str, bytes)):
        values.update(
            item.rsplit("@", 1)[-1]
            for item in repo_digests
            if isinstance(item, str) and "@" in item
        )
    return digest in values


def _container_matches_image(
    container: Mapping[str, Any], image: Mapping[str, Any], digest: str
) -> bool:
    """Bind a running container to the image inspected by immutable digest.

    ``docker container inspect`` normally reports an image *ID* while the
    configured value is a registry manifest digest.  Merely copying
    ``RepoDigests`` from an independent image inspection would attest the
    wrong container, so require the container's reported ID/ref to correspond
    to the inspected image before enriching the redacted projection.
    """
    image_id = image.get("Id")
    observed: set[str] = set()
    raw_image = container.get("Image")
    if isinstance(raw_image, str):
        observed.add(raw_image)
    config = container.get("Config")
    if isinstance(config, Mapping) and isinstance(config.get("Image"), str):
        observed.add(str(config["Image"]))
    if isinstance(image_id, str) and image_id:
        # Docker's container inspect exposes the immutable image ID in
        # ``Image``.  When that field is present, a manifest digest appearing
        # in ``Config.Image`` is not enough: it can be a stale/caller-supplied
        # reference attached to an entirely different image.
        return image_id in observed
    return any(digest in value for value in observed)


def _bind_container_image(
    container: Mapping[str, Any],
    image: Mapping[str, Any] | None,
    digest: str,
    role: str,
) -> dict[str, Any]:
    """Require container/image identity binding before adding digest evidence."""
    if isinstance(image, Mapping):
        return dict(_enrich_verified_container_image(container, image, digest, role))
    if not _observed_image_matches(container, digest):
        raise ExecutorFailure(f"{role} container image digest attestation failed")
    return dict(container)


def _pid_from_observation(observation: Mapping[str, Any]) -> int | None:
    state = observation.get("State")
    value = state.get("Pid") if isinstance(state, Mapping) else observation.get("Pid")
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _enrich_verified_container_image(
    container: Mapping[str, Any], image: Mapping[str, Any], digest: str, name: str
) -> Mapping[str, Any]:
    """Add a digest to a container projection only after identity binding."""
    if not _container_matches_image(container, image, digest):
        raise ExecutorFailure(f"{name} container image identity does not match its immutable image")
    return {
        **dict(container),
        "RepoDigests": [f"verified@{digest}"],
    }


def _minimal_child_env(host: DockerHostRef, *, api_key: str = "") -> dict[str, str]:
    """Build an allow-listed environment for adapter-owned child processes.

    The launcher process may carry provider credentials and unrelated operator
    secrets.  Passing ``os.environ`` to the pinned generator or workspace would
    make those values readable from an otherwise isolated task.  Docker still
    receives the explicit rootless socket, and the server sidecar receives only
    the named CyberGym key required by its ``--env CYBERGYM_API_KEY`` contract.
    """

    env = {
        name: value
        for name in _SAFE_ENV_NAMES
        if (value := os.environ.get(name)) is not None
    }
    env["DOCKER_HOST"] = host.value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if api_key:
        env[API_KEY_ENV] = api_key
    return env


def _initialize_generated_workspace_git(
    workspace_root: pathlib.Path,
    *,
    runner: CommandRunner,
    host: DockerHostRef,
) -> str:
    """Create a tiny, deterministic Git anchor for official generated input.

    The gateway requires a Git worktree, whereas CyberGym emits a plain
    directory.  Only the small task-control files are tracked.  The pinned
    generated archive/source tree is ignored deliberately so each task does
    not duplicate hundreds of megabytes of Git blobs or publish benchmark
    input as an agent-authored patch.  Tool trajectories remain the authority
    for source reads/writes; new files such as ``final.poc`` stay unignored.
    """

    root = _safe_abs(workspace_root, "workspace_root")
    marker = root / ".git"
    if os.path.lexists(marker):
        raise ExecutorFailure("generated CyberGym workspace unexpectedly contains git metadata")

    git_env = _minimal_child_env(host)
    git_env.update({
        "GIT_AUTHOR_NAME": "CyberGym Input Anchor",
        "GIT_AUTHOR_EMAIL": "cybergym-input-anchor@invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_NAME": "CyberGym Input Anchor",
        "GIT_COMMITTER_EMAIL": "cybergym-input-anchor@invalid",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    })
    init = runner(
        ["git", "init", "--quiet", str(root)],
        cwd=root.parent,
        env=git_env,
        timeout=30,
    )
    if init.returncode != 0 or not marker.is_dir():
        raise ExecutorFailure("generated CyberGym workspace could not be made a git worktree")

    exclude = marker / "info" / "exclude"
    try:
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        lines = existing.splitlines()
        for pattern in _GENERATED_INPUT_EXCLUDES:
            if pattern not in lines:
                lines.append(pattern)
        exclude.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    except OSError as exc:
        raise ExecutorFailure("generated CyberGym git excludes could not be installed") from exc

    add = runner(
        ["git", "-C", str(root), "add", "--", *_GENERATED_TRACKED_INPUTS],
        cwd=root,
        env=git_env,
        timeout=30,
    )
    if add.returncode != 0:
        raise ExecutorFailure("generated CyberGym control files could not be anchored")
    commit = runner(
        [
            "git", "-C", str(root),
            "-c", "core.hooksPath=/dev/null",
            "-c", "commit.gpgsign=false",
            "commit", "--quiet", "--no-verify", "--no-gpg-sign",
            "-m", "Anchor official CyberGym generated inputs",
        ],
        cwd=root,
        env=git_env,
        timeout=30,
    )
    if commit.returncode != 0:
        raise ExecutorFailure("generated CyberGym input anchor commit failed")
    head = runner(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        cwd=root,
        env=git_env,
        timeout=30,
    )
    anchor = head.stdout.strip()
    if head.returncode != 0 or not _HEX40.fullmatch(anchor):
        raise ExecutorFailure("generated CyberGym input anchor identity is invalid")
    status = runner(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        env=git_env,
        timeout=30,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ExecutorFailure("generated CyberGym input anchor is not clean")
    return anchor


def _write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _install_workspace_backend_alias(workspace_root: pathlib.Path) -> pathlib.Path:
    """Install the confined host alias for the container's ``/workspace`` root.

    The generated task is a git worktree, so the alias is hidden through that
    worktree's local ``.git/info/exclude`` rather than by changing source files
    or the global patch policy.  A pre-existing entry is refused: replacing a
    task file/directory would silently alter the benchmark input.  Every
    fallible metadata operation completes before the final symlink creation;
    there is deliberately no post-create check-and-delete rollback, because
    that sequence cannot be made race-free against a child replacing the link.
    A failed preparation may leave its O_EXCL temporary under ``.git/info``;
    that metadata is ignored and the workspace is append-only disposable, so
    no path-based cleanup is attempted.  A failed or interrupted final
    creation therefore leaves either no alias or the alias in the unique task
    workspace for ordinary cleanup custody.
    """

    root = pathlib.Path(workspace_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ExecutorFailure("workspace backend alias requires a directory root")

    alias = root / _WORKSPACE_BACKEND_ALIAS_NAME
    if os.path.lexists(alias):
        raise ExecutorFailure(
            "generated workspace contains the reserved backend alias path"
        )

    git_dir = root / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise ExecutorFailure("workspace backend alias requires local git metadata")
    info_dir = git_dir / "info"
    # Do not let a generated workspace redirect the exclude update through a
    # symlinked (including dangling) metadata ancestor.  ``Path.is_file``
    # follows links, so inspect every relevant component explicitly first.
    if info_dir.is_symlink() or not info_dir.is_dir():
        raise ExecutorFailure("workspace backend alias requires local git info directory")
    exclude = info_dir / "exclude"
    if exclude.is_symlink() or not exclude.is_file():
        raise ExecutorFailure("workspace backend alias requires git info/exclude")

    temporary: pathlib.Path | None = None
    try:
        try:
            current = exclude.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ExecutorFailure("workspace git exclude file is unreadable") from exc
        if _WORKSPACE_BACKEND_ALIAS_EXCLUDE not in current.splitlines():
            separator = "" if not current or current.endswith(("\n", "\r")) else "\n"
            replacement = current + separator + _WORKSPACE_BACKEND_ALIAS_EXCLUDE + "\n"
            # ``NamedTemporaryFile`` creates the file with O_EXCL in the same
            # directory.  This prevents a stale/foreign predictable temp path
            # from being truncated before the atomic replacement.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=exclude.parent,
                prefix=f".{exclude.name}.tmp.",
                delete=False,
            ) as handle:
                temporary = pathlib.Path(handle.name)
                handle.write(replacement)
            os.replace(temporary, exclude)
    except (FileExistsError, OSError, RuntimeError, TypeError) as exc:
        raise ExecutorFailure("unable to install workspace backend alias") from exc

    # This must remain the final fallible operation.  In particular, do not
    # lstat/readlink/unlink here: a concurrent replacement could turn a
    # check-then-delete rollback into deletion of a task-owned object.
    try:
        os.symlink(
            _WORKSPACE_BACKEND_ALIAS_TARGET,
            alias,
            target_is_directory=True,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError) as exc:
        raise ExecutorFailure("unable to install workspace backend alias") from exc
    return alias


class _DockerRuntimeMixin:
    """Docker/container/network/workspace methods mixed into the executor."""

    def _docker(self, *args: str, timeout: float = 60) -> CommandResult:
        return self.config.command_runner(
            ["docker", "--host", self.host.value, *args],
            cwd=self.config.run_root,
            env=_minimal_child_env(self.host),
            timeout=timeout,
        )

    def _inspect(self, kind: str, name: str) -> Mapping[str, Any]:
        result = self._docker(kind, "inspect", name)
        if result.returncode != 0:
            raise ExecutorFailure(f"docker inspect failed for {kind} {name}")
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExecutorFailure("docker inspect returned invalid JSON") from exc
        if not isinstance(values, list) or not values or not isinstance(values[0], Mapping):
            raise ExecutorFailure("docker inspect returned no object")
        return values[0]

    def _inspect_optional(self, kind: str, name: str) -> Mapping[str, Any] | None:
        """Read one owned object, returning ``None`` only for a missing object."""
        result = self._docker(kind, "inspect", name)
        if result.returncode != 0:
            diagnostic = f"{result.stdout}\n{result.stderr}".lower()
            exact_not_found = f"{kind} {str(name).lower()} not found"
            if any(
                marker in diagnostic
                for marker in ("no such object", "no such container", "no such network", exact_not_found)
            ) or (kind in {"container", "network"} and diagnostic.rstrip().endswith(" not found")):
                return None
            raise ExecutorFailure(f"docker inspect failed for {kind} {name}")
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExecutorFailure("docker inspect returned invalid JSON") from exc
        if not isinstance(values, list) or not values or not isinstance(values[0], Mapping):
            raise ExecutorFailure("docker inspect returned no object")
        return values[0]

    def _inspect_image(self, image_ref: str, digest: str, name: str) -> Mapping[str, Any]:
        """Resolve an image by its immutable reference before any paid call."""
        result = self._docker("image", "inspect", image_ref)
        if result.returncode != 0:
            raise ExecutorFailure(f"docker image inspect failed for {name}")
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExecutorFailure(f"docker image inspect returned invalid JSON for {name}") from exc
        if not isinstance(values, list) or not values or not isinstance(values[0], Mapping):
            raise ExecutorFailure(f"docker image inspect returned no object for {name}")
        observed = values[0]
        repo_digests = observed.get("RepoDigests")
        if not isinstance(repo_digests, Sequence) or isinstance(repo_digests, (str, bytes)):
            repo_digests = ()
        digest_values = {
            item.rsplit("@", 1)[-1]
            for item in repo_digests
            if isinstance(item, str) and "@" in item
        }
        image_id = observed.get("Id")
        if isinstance(image_id, str):
            digest_values.add(image_id)
        if digest not in digest_values:
            raise ExecutorFailure(f"{name} does not resolve to its configured immutable digest")
        return observed

    def _inspect_daemon(self) -> dict[str, Any]:
        """Prove that the selected socket is a live rootless daemon.

        A path-shaped socket under ``/mnt/data`` is not identity evidence: a
        stale file or a rootful daemon can satisfy that lexical heuristic.  We
        retain only non-secret Docker info fields and require the daemon's
        explicit rootless security marker before any provider request.
        """
        result = self._docker("info", "--format", "{{json .}}", timeout=30)
        if result.returncode != 0:
            raise ExecutorFailure("selected Docker daemon info probe failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExecutorFailure("selected Docker daemon returned invalid info JSON") from exc
        if not isinstance(value, Mapping):
            raise ExecutorFailure("selected Docker daemon info is not an object")
        daemon_id = str(value.get("ID") or value.get("Id") or "").strip()
        version = str(value.get("ServerVersion") or "").strip()
        security = value.get("SecurityOptions")
        security_values = (
            [str(item).lower() for item in security if isinstance(item, str)]
            if isinstance(security, Sequence) and not isinstance(security, (str, bytes))
            else []
        )
        rootless_value = value.get("Rootless")
        rootless = rootless_value is True or any("rootless" in item for item in security_values)
        if not daemon_id or not version or not rootless:
            raise ExecutorFailure("selected Docker daemon is not attested as rootless")
        observation = {
            "status": "passed",
            "socket": self.host.value,
            "endpoint": self.host.value,
            "daemon_id": daemon_id,
            "server_version": version,
            "rootless": True,
            "security_options": sorted(security_values),
        }
        docker_root_dir = value.get("DockerRootDir") or value.get("docker_root_dir")
        if isinstance(docker_root_dir, str) and docker_root_dir:
            observation["docker_root_dir"] = docker_root_dir
        self.daemon_observation = observation
        _write_json(self.config.run_root / "docker_daemon.json", observation)
        return observation

    def _image_observation(self, container: Mapping[str, Any], image: Mapping[str, Any]) -> dict[str, Any]:
        """Merge image-level RepoDigests into a container inspect projection."""
        result = dict(container)
        repo_digests = image.get("RepoDigests")
        if isinstance(repo_digests, Sequence) and not isinstance(repo_digests, (str, bytes)):
            result["RepoDigests"] = list(repo_digests)
            config = result.get("Config")
            if isinstance(config, Mapping):
                config_copy = dict(config)
                config_copy["RepoDigests"] = list(repo_digests)
                result["Config"] = config_copy
        return result

    def _network(self) -> None:
        argv = build_network_create_argv(self.host, self._network_plan("campaign"))
        result = self.config.command_runner(
            argv, cwd=self.config.run_root, env=_minimal_child_env(self.host), timeout=60
        )
        if result.returncode == 0:
            self.network_id = result.stdout.strip()
            self._network_created = True
        else:
            # A campaign always owns a fresh network.  Reusing a same-named
            # network is ambiguous (and breaks parallel campaigns), even when
            # its labels happen to look compatible; leave it for an explicit
            # operator cleanup instead of attaching to stale containers.
            raise ExecutorFailure(
                "cybergym-internal already exists or could not be created; a fresh campaign network is required"
            )
        if not self.network_id:
            raise ExecutorFailure("network create did not return an id")
        info = self._inspect("network", "cybergym-internal")
        if info.get("Name") != "cybergym-internal" or info.get("Internal") is not False or info.get("Driver") != "bridge":
            raise ExecutorFailure("CyberGym network attestation failed")
        observed_id = str(info.get("Id") or "").strip()
        if observed_id != self.network_id:
            raise ExecutorFailure("CyberGym network id changed during startup")
        labels = info.get("Labels") if isinstance(info.get("Labels"), Mapping) else {}
        if labels.get("com.ouroboros.campaign") != self.config.campaign_id:
            raise ExecutorFailure("CyberGym network ownership label is missing or mismatched")
        attached = info.get("Containers")
        if isinstance(attached, Mapping) and attached:
            # A reused network must not silently inherit another campaign's
            # containers.  A newly created network should be empty at this
            # point; the server/workspace ids are added only after their own
            # inspections below.
            own_ids = {self.server_id, *self._task_containers.values()} - {""}
            foreign = {
                str(container_id)
                for container_id in attached
                if str(container_id) not in own_ids
            }
            if foreign:
                raise ExecutorFailure("CyberGym network has unknown attached containers")

    def _network_plan(self, task_id: str) -> NetworkPlan:
        # ``NetworkPlan`` rejects aliases containing the task token.  A
        # campaign-level server has no real task, so use an opaque bootstrap
        # token rather than the human word ``campaign`` (which commonly occurs
        # in the campaign id itself).
        plan_task_id = "bootstrap" if task_id == "campaign" else task_id
        plan = NetworkPlan(
            self.config.campaign_id,
            plan_task_id,
            int(self.config.server_port),
            int(self.config.verifier_host_port or self.config.server_port + 1),
            server_container_port=int(self.config.server_port),
        )
        if task_id != "campaign":
            # The server has one campaign alias; per-task plans only change the
            # opaque workspace alias/agent identity.
            campaign_plan = self._network_plan("campaign")
            plan = dataclasses.replace(plan, server_alias=campaign_plan.server_alias)
        return plan

    def _task_network_plan(self, task_id: str, agent_id: str) -> NetworkPlan:
        """Build a task plan whose workspace identity is unique to this attempt."""
        plan = self._network_plan(task_id)
        # Clear the derived alias as well: ``dataclasses.replace`` reruns the
        # validator, and retaining the previous task's alias would make the
        # Docker label/NO_PROXY identity disagree with the new attempt.
        return dataclasses.replace(plan, opaque_agent_id=agent_id, workspace_alias="")

    def _opaque_workspace_path(self, agent_id: str) -> pathlib.Path:
        """Return a host workspace path that carries no real task identifier."""

        if not re.fullmatch(r"agent-[0-9a-f]{24}", agent_id):
            raise ExecutorFailure("workspace agent id is not opaque")
        path = (self.config.run_root / "workspaces" / agent_id).resolve(strict=False)
        _inside(path, _safe_abs(self.config.run_root, "run_root"), "workspace_dir")
        return path

    def _wait_server(self, plan: NetworkPlan) -> None:
        # FastAPI's /docs is HTML; the JSON transport uses the equivalent
        # OpenAPI route so readiness does not mistake a healthy server for a
        # malformed JSON response.
        deadline = time.monotonic() + 120
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = _unwrap_http_json(
                    self._server_http("GET", "/openapi.json", timeout=10),
                    operation="CyberGym server readiness",
                )
                paths = response.get("paths")
                if not isinstance(paths, Mapping):
                    raise ExecutorFailure("CyberGym readiness response has no OpenAPI paths")
                required = {"/submit-vul", "/submit-fix", "/query-poc", "/verify-agent-pocs"}
                if not required.issubset(paths):
                    raise ExecutorFailure("CyberGym readiness response misses a required route")
                return
            except Exception as exc:  # transport/readiness only; do not leak body
                last_error = exc
            self.config.sleep(1)
        raise ExecutorFailure("CyberGym server did not expose its documented route") from last_error

    def _server_http(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60,
    ) -> Any:
        """Call the private server without exposing a port from an internal bridge.

        A caller-supplied HTTP runner remains the explicit test/integration
        seam.  The default transport executes a fixed stdlib client inside
        the server container, whose immutable id was attested at startup.
        Only the named API-key flag is passed to ``docker exec``; the key value
        is inherited from the server container environment.
        """
        method_text = str(method or "GET").strip().upper()
        if method_text not in {"GET", "POST"}:
            raise ExecutorFailure("server HTTP method is unsupported")
        path_text = str(path or "").strip()
        parsed = urllib.parse.urlsplit(path_text)
        custom_runner = self.config.http_runner is not urllib_json
        if parsed.scheme or parsed.netloc:
            # Preserve the injected runner's full URL contract, but normalize
            # production calls to a path so no untrusted host can be reached
            # from the server container.
            if custom_runner:
                return self.config.http_runner(
                    method_text,
                    path_text,
                    body=body,
                    headers=headers,
                    timeout=timeout,
                )
            path_text = parsed.path or "/"
            if parsed.query or parsed.fragment:
                path_text += "?" + parsed.query if parsed.query else ""
        if not path_text.startswith("/") or "\x00" in path_text or len(path_text) > 2048:
            raise ExecutorFailure("server HTTP path is unsafe")
        if custom_runner:
            plan = self._network_plan("campaign")
            return self.config.http_runner(
                method_text,
                f"http://127.0.0.1:{plan.verifier_host_port}{path_text}",
                body=body,
                headers=headers,
                timeout=timeout,
            )
        server_id = str(self.server_id or "").strip()
        if not server_id or not _GATEWAY_TASK_ID.fullmatch(server_id):
            raise ExecutorFailure("server HTTP requires an immutable server container id")
        encoded = ""
        if body is not None:
            try:
                encoded = base64.b64encode(json.dumps(body, ensure_ascii=False).encode("utf-8")).decode("ascii")
            except (TypeError, ValueError) as exc:
                raise ExecutorFailure("server HTTP body is not JSON serializable") from exc
        auth = bool(headers and any(str(key).lower() == "x-api-key" for key in headers))
        exec_argv = [
            "docker", "--host", self.host.value, "exec",
            "--env", f"CYBERGYM_HTTP_METHOD={method_text}",
            "--env", f"CYBERGYM_HTTP_PATH={path_text}",
            "--env", f"CYBERGYM_HTTP_PORT={int(self.config.server_port)}",
            "--env", f"CYBERGYM_HTTP_BODY_B64={encoded}",
            "--env", f"CYBERGYM_HTTP_TIMEOUT={max(1.0, float(timeout))}",
            "--env", f"CYBERGYM_HTTP_AUTH={'1' if auth else '0'}",
            server_id, "python", "-c", _SERVER_HTTP_SCRIPT,
        ]
        result = self.config.command_runner(
            exec_argv,
            cwd=self.config.run_root,
            env=_minimal_child_env(self.host),
            timeout=max(1.0, float(timeout) + 5.0),
        )
        if result.returncode != 0:
            raise ExecutorFailure("private server HTTP transport failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ExecutorFailure("private server HTTP transport returned invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ExecutorFailure("private server HTTP transport returned a non-object")
        if value.get("transport_error"):
            raise ExecutorFailure("private server HTTP transport failed")
        return value

    def _write_campaign_state(self, state: Mapping[str, Any]) -> None:
        _write_json(self.config.run_root / "sidecar_state.json", state)

    def _recover_workspace_custody(
        self, container_name: str, plan: NetworkPlan, reason: str
    ) -> bool:
        """Recover a container attached by a failed ``docker run`` by name.

        Docker can create and attach the container before the runner receives
        an id (for example, when the command times out).  Inspect the exact
        generated name while it is still an owned handle, then publish the
        immutable id only after the campaign/role/network/image checks pass.
        If inspection cannot prove custody, retain a typed name entry so close
        and attestation never silently treat the container as disposable.
        """
        observed: Mapping[str, Any] | None = None
        failure_reason = str(reason or "workspace start failed")
        try:
            observed = self._inspect_optional("container", container_name)
        except Exception as exc:
            failure_reason += f"; name inspect failed: {type(exc).__name__}"
        if observed is not None:
            observed_id = str(observed.get("Id") or "").strip()
            actual_name = str(observed.get("Name") or "").lstrip("/")
            config = observed.get("Config")
            labels = config.get("Labels", {}) if isinstance(config, Mapping) else {}
            networks = ((observed.get("NetworkSettings") or {}).get("Networks") or {})
            network = networks.get("cybergym-internal") if isinstance(networks, Mapping) else None
            try:
                bound = _bind_container_image(
                    observed,
                    self._workspace_image_observation,
                    self.config.workspace_image_digest,
                    "workspace",
                )
            except Exception as exc:
                bound = None
                failure_reason += f"; image custody failed: {type(exc).__name__}"
            if (
                observed_id
                and _GATEWAY_TASK_ID.fullmatch(observed_id)
                and actual_name == container_name
                and isinstance(labels, Mapping)
                and labels.get("com.ouroboros.campaign") == self.config.campaign_id
                and labels.get("com.ouroboros.role") == "workspace"
                and labels.get("com.ouroboros.agent_id") == plan.opaque_agent_id
                and isinstance(network, Mapping)
                and (not self.network_id or str(network.get("NetworkID") or "") == self.network_id)
                and bound is not None
            ):
                with self._registry_condition:
                    self._task_containers[container_name] = observed_id
                    self._workspace_observations[container_name] = bound
                    self._unresolved_workspace_custody.pop(container_name, None)
                return True
            failure_reason += "; inspected container did not prove ownership"
        with self._registry_condition:
            self._unresolved_workspace_custody[container_name] = failure_reason
        return False

    def _workspace(self, task: TaskSpec, task_dir: pathlib.Path, plan: NetworkPlan) -> str:
        container_name = f"cybergym-workspace-{plan.opaque_agent_id}"
        with self._registry_condition:
            if self._unresolved_workspace_custody:
                names = ", ".join(sorted(self._unresolved_workspace_custody))
                raise ExecutorFailure(f"workspace startup custody is unresolved: {names}")
        spec = WorkspaceCommandSpec(
            self.host,
            plan,
            _pinned_image_ref(self.config.workspace_image, self.config.workspace_image_digest, "workspace_image"),
            container_name,
            str(task_dir),
            command=self.config.command,
            labels={"com.ouroboros.image_digest": self.config.workspace_image_digest},
        )
        # A container is attached before ``docker run`` returns its id.  Mark
        # the startup before invoking Docker, but do not hold the registry lock
        # over the command: independent task lanes must retain parallel starts.
        with self._registry_condition:
            self._workspace_starting[container_name] = self._workspace_starting.get(container_name, 0) + 1
            self._unresolved_workspace_custody.pop(container_name, None)
        try:
            try:
                result = self.config.command_runner(
                    build_workspace_argv(spec), cwd=self.config.run_root,
                    env=_minimal_child_env(self.host), timeout=120,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    raise ExecutorFailure("CyberGym workspace failed to start")
                provisional_id = result.stdout.strip().splitlines()[-1].strip()
                if not provisional_id or not _GATEWAY_TASK_ID.fullmatch(provisional_id):
                    raise ExecutorFailure("workspace start returned an unsafe container id")
                # Publish the provisional id before inspect so a transport
                # failure after ``run`` still leaves exact cleanup custody.
                with self._registry_lock:
                    self._task_containers[container_name] = provisional_id
                observed = self._inspect("container", container_name)
                observed_id = str(observed.get("Id") or "").strip()
                if not observed_id:
                    raise ExecutorFailure("workspace inspect returned no immutable container id")
                if observed_id != provisional_id:
                    raise ExecutorFailure("workspace container id changed during startup")
                networks = ((observed.get("NetworkSettings") or {}).get("Networks") or {})
                if "cybergym-internal" not in networks:
                    raise ExecutorFailure("workspace is not on cybergym-internal")
                observed = _bind_container_image(
                    observed,
                    self._workspace_image_observation,
                    self.config.workspace_image_digest,
                    "workspace",
                )
                with self._registry_lock:
                    self._task_containers[container_name] = observed_id
                    self._workspace_observations[container_name] = observed
                    self._unresolved_workspace_custody.pop(container_name, None)
                return container_name
            except BaseException as exc:
                with self._registry_lock:
                    has_exact_id = bool(self._task_containers.get(container_name))
                if not has_exact_id:
                    has_exact_id = self._recover_workspace_custody(
                        container_name, plan, type(exc).__name__
                    )
                if has_exact_id:
                    # Failed attempt after create: release the docker slot.
                    # Logs/checkpoints remain custody, not a live container.
                    report = (
                        self.config.run_root
                        / "workspaces"
                        / f"{container_name}.startup_cleanup.json"
                    )
                    try:
                        self._cleanup_workspace_container(
                            container_name,
                            str(getattr(task, "task_id", "") or "startup"),
                            "startup",
                            report,
                        )
                    except Exception:
                        pass
                raise
        finally:
            with self._registry_condition:
                count = self._workspace_starting.get(container_name, 0)
                if count <= 1:
                    self._workspace_starting.pop(container_name, None)
                else:
                    self._workspace_starting[container_name] = count - 1
                self._registry_condition.notify_all()

    def _probe_from_workspace(self, container_id: str, script: str) -> bool | None:
        """Run one bounded, non-mutating connectivity probe in the agent container."""
        result = self._docker("exec", container_id, "sh", "-lc", script, timeout=30)
        if result.returncode == 127 and "not found" in (result.stderr or "").lower():
            return None
        return result.returncode == 0

    def _probe_workspace_http(
        self,
        container_id: str,
        url: str,
        *,
        method: str = "GET",
        expected_statuses: set[int] | None = None,
    ) -> dict[str, Any]:
        """Return redacted HTTP reachability/denial facts from the workspace."""
        method_text = str(method or "GET").strip().upper()
        if method_text not in {"GET", "POST"}:
            raise ExecutorFailure("workspace probe method is unsupported")
        request_flags = ["--request", method_text]
        if method_text == "POST":
            # The private CyberGym routes run their API-key dependency before
            # body validation.  This deliberately malformed JSON therefore
            # proves an unauthenticated denial without supplying multipart
            # data that could create or modify a PoC.
            request_flags.extend(
                [
                    "--header",
                    "Content-Type: application/json",
                    "--data-raw",
                    '{"agent_id":"cybergym-probe","task_id":"cybergym-probe"}',
                ]
            )
        script = (
            "curl --noproxy '*' --silent --show-error --output /dev/null "
            "--write-out '%{http_code}' --connect-timeout 5 --max-time 15 "
            + " ".join(shlex.quote(item) for item in request_flags)
            + " "
            + shlex.quote(url)
        )
        result = self._docker("exec", container_id, "sh", "-lc", script, timeout=30)
        if result.returncode == 127 and "not found" in (result.stderr or "").lower():
            return {"reachable": None, "denied": None, "mutating": None, "status_code": None}
        match = re.search(r"(?:^|\D)([1-5]\d\d)(?:\D|$)", result.stdout or "")
        status = int(match.group(1)) if match else None
        reachable = result.returncode == 0 and status is not None
        denied = status in (expected_statuses or set()) if status is not None else None
        return {
            "reachable": reachable,
            "denied": denied,
            "mutating": False if denied is True else None,
            "status_code": status,
        }

    def _probe_http_route(
        self,
        method: str,
        url: str,
        *,
        api_key: str = "",
    ) -> bool | None:
        """Distinguish an HTTP response from a dead transport without logging bodies."""
        headers = {"X-API-Key": api_key} if api_key else None
        try:
            parsed = urllib.parse.urlsplit(str(url))
            if self.config.http_runner is urllib_json and parsed.hostname in {"127.0.0.1", "localhost"}:
                response = self._server_http(
                    method,
                    parsed.path or "/",
                    body={"agent_id": "probe-agent", "task_id": "probe-task"},
                    headers=headers,
                    timeout=15,
                )
            else:
                response = self.config.http_runner(
                    method,
                    url,
                    body={"agent_id": "probe-agent", "task_id": "probe-task"},
                    headers=headers,
                    timeout=15,
                )
            if isinstance(response, Mapping):
                status = response.get("status_code", response.get("http_status"))
                if status is not None:
                    int(status)
        except ExecutorFailure as exc:
            message = str(exc).lower()
            if "http " in message and "transport failed" not in message:
                return True
            return None
        except Exception:
            return None
        return True

    def _connectivity_observation(
        self,
        plan: NetworkPlan,
        workspace_id: str,
        api_key: str,
    ) -> dict[str, Any]:
        """Collect route facts plus bounded hidden-artifact checks.

        The generic sidecar schema has five stable connectivity fields.  The
        adapter adds the path/environment checks here so they are preserved in
        the redacted report without making the core schema know CyberGym's
        private filenames.
        """
        probes = {item["name"]: item for item in build_connectivity_probe_plan(plan)}
        server_target = str(probes["agent_to_server"]["target"])
        public_target = str(probes["agent_to_public"]["target"])
        verifier_targets = tuple(probes["agent_to_verifier"]["targets"])
        tool_probe = self._docker(
            "exec", workspace_id, "sh", "-lc", "command -v sh >/dev/null && command -v curl >/dev/null", timeout=30
        )
        agent_probe_tools = (
            None
            if tool_probe.returncode == 127 and "not found" in (tool_probe.stderr or "").lower()
            else tool_probe.returncode == 0
        )
        positive_script = (
            "curl --noproxy '*' --silent --show-error --output /dev/null "
            "--write-out '%{http_code}' --connect-timeout 5 --max-time 15 "
            + shlex.quote(server_target)
        )
        public_script = (
            "curl --noproxy '*' --silent --show-error --output /dev/null "
            "--connect-timeout 3 --max-time 8 " + shlex.quote(public_target)
        )
        verifier_script = (
            "curl --noproxy '*' --silent --show-error --output /dev/null "
            "--connect-timeout 3 --max-time 8 " + shlex.quote(verifier_targets[0])
        )
        positive_result = self._docker("exec", workspace_id, "sh", "-lc", positive_script, timeout=30)
        if positive_result.returncode == 127 and "not found" in (positive_result.stderr or "").lower():
            agent_to_server: bool | None = None
        else:
            status_match = re.search(r"(?:^|\D)2\d\d(?:\D|$)", positive_result.stdout or "")
            agent_to_server = positive_result.returncode == 0 and status_match is not None
        public_value = self._probe_from_workspace(workspace_id, public_script)
        verifier_value = self._probe_from_workspace(workspace_id, verifier_script)
        socket_value = self._probe_from_workspace(workspace_id, "test ! -e /var/run/docker.sock")
        private_targets = tuple(probes["verifier_to_private"]["targets"])
        # Probe the protected routes with malformed POSTs.  The pinned server
        # checks its API-key dependency before parsing the body, so an
        # unauthenticated 404/401/403 proves both transport and the auth wall;
        # a 405 (wrong method) or 422 (body-only rejection) is insufficient.
        private_values = [self._probe_http_route("GET", target, api_key=api_key) for target in private_targets]
        private_reachable: bool | None
        if any(value is None for value in private_values):
            private_reachable = None
        else:
            private_reachable = all(value is True for value in private_values)
        protected_targets = tuple(probes["agent_to_server_protected"]["targets"])
        protected_observed = {
            target: self._probe_workspace_http(
                workspace_id,
                str(target),
                method="POST",
                expected_statuses={401, 403, 404},
            )
            for target in protected_targets
        }
        hidden_paths = (
            "/cybergym-server-data",
            "/cybergym-mask-map.json",
            "/cybergym-poc.db",
            "/cybergym-fixed",
        )
        hidden_artifacts: dict[str, bool | None] = {}
        for path in hidden_paths:
            hidden_artifacts[path] = self._probe_from_workspace(
                workspace_id, "test ! -e " + shlex.quote(path)
            )
        secret_env = self._probe_from_workspace(
            workspace_id,
            "test -z \"${CYBERGYM_API_KEY-}\" && test -z \"${DOCKER_HOST-}\"",
        )
        return {
            "agent_to_server": agent_to_server,
            "verifier_to_private": {"reachable": private_reachable},
            "agent_to_server_protected": {
                "targets": protected_observed,
                "reachable": all(item["reachable"] is True for item in protected_observed.values()),
                "denied": all(item["denied"] is True for item in protected_observed.values()),
                "mutating": any(item["mutating"] is True for item in protected_observed.values()),
            },
            "agent_to_public": public_value,
            "agent_to_verifier": verifier_value,
            "agent_socket_visible": None if socket_value is None else not socket_value,
            "agent_hidden_artifacts": hidden_artifacts,
            "agent_secret_env_absent": secret_env,
            "agent_probe_tools": agent_probe_tools,
        }

    def _cleanup_workspace_container(
        self,
        container_name: str,
        task_id: str,
        attempt_id: str,
        report_path: pathlib.Path,
    ) -> dict[str, Any]:
        """Reap one settled workspace by immutable id and verify its absence.

        The campaign network and server remain shared by other lanes, so this
        deliberately does not call the broader ``CleanupPlan``.  It performs
        the same ownership checks locally: inspect the stored id, reject a
        name replacement, remove the exact id, and inspect again.  A finished
        or failed attempt must release this slot; logs and result_index are
        the custody surface, not a live container.
        """
        with self._registry_lock:
            container_id = str(self._task_containers.get(container_name) or "").strip()
        if not container_id:
            raise ExecutorFailure("workspace cleanup has no immutable container id")
        observed = self._inspect_optional("container", container_id)
        if observed is None:
            replacement = self._inspect_optional("container", container_name)
            if replacement is not None and str(replacement.get("Id") or "").strip() != container_id:
                raise ExecutorFailure("workspace name was replaced before cleanup")
            report = {
                "schema": "ouroboros.benchmark.cybergym.workspace_cleanup.v1",
                "status": "verified",
                "ok": True,
                "container_id": container_id,
                "container_name": container_name,
                "network_id": self.network_id,
                "already_absent": True,
            }
            _write_json(report_path, report)
            with self._registry_lock:
                self._task_containers.pop(container_name, None)
                self._workspace_observations.pop(container_name, None)
            return report
        actual_id = str(observed.get("Id") or "").strip()
        actual_name = str(observed.get("Name") or "").lstrip("/")
        config = observed.get("Config")
        labels = config.get("Labels", {}) if isinstance(config, Mapping) else {}
        if (
            actual_id != container_id
            or actual_name != container_name
            or not isinstance(labels, Mapping)
            or labels.get("com.ouroboros.campaign") != self.config.campaign_id
            or labels.get("com.ouroboros.role") != "workspace"
        ):
            raise ExecutorFailure("workspace cleanup ownership attestation failed")
        networks = ((observed.get("NetworkSettings") or {}).get("Networks") or {})
        network = networks.get("cybergym-internal") if isinstance(networks, Mapping) else None
        if not isinstance(network, Mapping) or str(network.get("NetworkID") or "") != self.network_id:
            raise ExecutorFailure("workspace cleanup network identity attestation failed")
        result = self._docker("rm", "--force", container_id, timeout=60)
        if result.returncode not in {0, 1}:
            raise ExecutorFailure("workspace cleanup command failed")
        if self._inspect_optional("container", container_id) is not None:
            raise ExecutorFailure("workspace cleanup postcondition failed")
        replacement = self._inspect_optional("container", container_name)
        if replacement is not None and str(replacement.get("Id") or "").strip() != container_id:
            raise ExecutorFailure("workspace name was replaced during cleanup")
        report = {
            "schema": "ouroboros.benchmark.cybergym.workspace_cleanup.v1",
            "status": "verified",
            "ok": True,
            "container_id": container_id,
            "container_name": container_name,
            "network_id": self.network_id,
            "already_absent": False,
        }
        _write_json(report_path, report)
        with self._registry_lock:
            self._task_containers.pop(container_name, None)
            self._workspace_observations.pop(container_name, None)
        return report

    def _cleanup_owned_resources(self) -> dict[str, Any]:
        """Remove exact inspected ids and verify that no owned object remains."""
        with self._registry_condition:
            if self._workspace_starting:
                raise ExecutorFailure("cleanup custody is pending workspace startup")
            if self._unresolved_workspace_custody:
                names = ", ".join(sorted(self._unresolved_workspace_custody))
                raise ExecutorFailure(f"cleanup custody is unresolved for workspace names: {names}")
            workspace_items = tuple(self._task_containers.items())
        workspace_ids = tuple(container_id for _name, container_id in workspace_items)
        if not self.network_id and not self.server_id and not workspace_ids:
            return {"status": "not_needed", "ok": True}
        if not self.network_id:
            raise ExecutorFailure("cleanup custody is incomplete; refusing name-based removal")
        if not self._network_created:
            raise ExecutorFailure("campaign network was not created by this executor; refusing removal")
        if not self.server_id and not workspace_ids:
            network = self._inspect_optional("network", self.network_id)
            if network is not None:
                labels = network.get("Labels") if isinstance(network.get("Labels"), Mapping) else {}
                if str(network.get("Id") or "") != self.network_id or labels.get("com.ouroboros.campaign") != self.config.campaign_id:
                    raise ExecutorFailure("cleanup network ownership attestation failed")
            result = self.config.command_runner(
                ("docker", "--host", self.host.value, "network", "rm", self.network_id),
                cwd=self.config.run_root,
                env=_minimal_child_env(self.host),
                timeout=60,
            )
            if result.returncode not in {0, 1} or self._inspect_optional("network", self.network_id) is not None:
                raise ExecutorFailure("campaign network cleanup postcondition failed")
            report = {
                "schema": "ouroboros.benchmark.cybergym.cleanup.v1",
                "status": "verified",
                "ok": True,
                "network_id": self.network_id,
                "network_removed": True,
                "container_ids": [],
            }
            _write_json(self.config.run_root / "cleanup.json", report)
            return report
        # Inspect live objects before removal and require the campaign labels and
        # immutable ids to agree with our checkpoint.  A name may have been
        # replaced by an unrelated container since startup.
        owned_ids = set(workspace_ids) | {self.server_id}
        for name, container_id in [(self.server_name, self.server_id), *workspace_items]:
            observed = self._inspect_optional("container", container_id)
            if observed is None:
                continue
            actual_id = str(observed.get("Id") or "").strip()
            actual_name = str(observed.get("Name") or "").lstrip("/")
            labels = (observed.get("Config") or {}).get("Labels", {}) if isinstance(observed.get("Config"), Mapping) else {}
            if actual_id != container_id or actual_name != name or labels.get("com.ouroboros.campaign") != self.config.campaign_id:
                raise ExecutorFailure("cleanup ownership attestation failed")
        network = self._inspect_optional("network", self.network_id)
        if network is not None:
            if str(network.get("Id") or "") != self.network_id or network.get("Name") != "cybergym-internal":
                raise ExecutorFailure("cleanup network identity attestation failed")
            labels = network.get("Labels") if isinstance(network.get("Labels"), Mapping) else {}
            if labels.get("com.ouroboros.campaign") != self.config.campaign_id:
                raise ExecutorFailure("cleanup network ownership attestation failed")
        cleanup = CleanupPlan(
            self.host,
            self.config.campaign_id,
            server_container_id=self.server_id,
            workspace_container_ids=workspace_ids,
            network_id=self.network_id,
        )
        commands = cleanup_argv(cleanup)
        for command in commands:
            result = self.config.command_runner(
                command,
                cwd=self.config.run_root,
                env=_minimal_child_env(self.host),
                timeout=60,
            )
            if result.returncode not in {0, 1}:
                raise ExecutorFailure("campaign-owned cleanup command failed")
        removed_ids: list[str] = []
        for container_id in sorted(owned_ids):
            if self._inspect_optional("container", container_id) is None:
                removed_ids.append(container_id)
        network_removed = self._inspect_optional("network", self.network_id) is None
        observation = {
            "removed_container_ids": removed_ids,
            "network_removed": network_removed,
            "removed_network_id": self.network_id,
            "ownership": {
                "campaign_id": self.config.campaign_id,
                "owner_label": f"com.ouroboros.campaign={self.config.campaign_id}",
                "container_ids": sorted(owned_ids),
                "network_id": self.network_id,
            },
        }
        report = validate_cleanup_observation(observation, cleanup)
        _write_json(self.config.run_root / "cleanup.json", report)
        if not report.get("ok"):
            raise ExecutorFailure("campaign cleanup postcondition failed")
        return report
