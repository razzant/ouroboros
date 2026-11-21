"""Workspace executor path contracts and Docker declared-output probes.

General file/cwd mappings remain strict. Declared Docker outputs alone resolve in
container space and carry a confirmed bind view to the artifact policy owner.
"""
from __future__ import annotations

import json
import pathlib
import posixpath
import subprocess
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from ouroboros.tool_access import path_is_relative_to


@dataclass(frozen=True)
class PathMapping:
    host_path: pathlib.Path
    backend_path: str


class BackendPathNoMatch(ValueError):
    """The backend spelling does not select any configured mapping.

    Mapping callers use this narrow distinction to preserve their existing
    host-path fallback.  A matched mapping with an invalid descendant remains
    a plain ``ValueError`` and must be refused.
    """

class ManagedPathRefused(ValueError):
    """A managed deployment's execution domain (``execution_domain``) does not map this path: nothing starts, and
    nothing runs on the trusted service instead. A ``ValueError``, so every mapping caller keeps its handling."""

    process_not_started = True


@dataclass(frozen=True)
class ExecutorRef:
    kind: str
    executor_id: str
    network: str
    mappings: tuple[PathMapping, ...]
    container_name: str = ""
    managed: bool = False  # the deployment's pinned executor: an unmapped path refuses, never runs locally

def normalize_executor_ref(raw: dict[str, Any]) -> ExecutorRef | None:
    if not raw:
        return None
    kind = str(raw.get("type") or raw.get("kind") or "").strip().lower()
    if not kind:
        raise ValueError("executor_ref.type is required")
    if kind not in {"local", "docker_exec"}:
        raise ValueError(f"unsupported executor_ref.type: {kind}")
    network = str(raw.get("network") or "host").strip().lower()
    if network not in {"host", "none"}:
        raise ValueError("executor_ref.network must be 'host' or 'none'")
    if kind == "local" and network == "none":
        raise ValueError("local executor_ref cannot enforce network=none; use docker_exec")

    mappings: list[PathMapping] = []
    workspace_host = str(raw.get("workspace_host_path") or "").strip()
    workspace_backend = str(raw.get("workspace_backend_path") or "").strip()
    if workspace_host and workspace_backend:
        mappings.append(PathMapping(pathlib.Path(workspace_host).expanduser().resolve(strict=False), _normalize_backend_path(workspace_backend)))
    raw_mappings = raw.get("path_mappings") if "path_mappings" in raw else raw.get("mappings")
    if raw_mappings is None:
        raw_mappings = []
    if not isinstance(raw_mappings, list):
        raise ValueError("executor_ref.path_mappings must be a list")
    for item in raw_mappings:
        if not isinstance(item, dict):
            raise ValueError("executor_ref.path_mappings entries must be objects")
        host = str(item.get("host_path") or "").strip()
        backend = str(item.get("backend_path") or "").strip()
        if not host or not backend:
            raise ValueError("executor_ref.path_mappings entries require host_path and backend_path")
        mappings.append(PathMapping(pathlib.Path(host).expanduser().resolve(strict=False), _normalize_backend_path(backend)))
    if not mappings:
        raise ValueError("executor_ref requires at least one host/backend path mapping")
    if kind == "docker_exec" and not str(raw.get("container_name") or raw.get("container") or "").strip():
        raise ValueError("docker_exec executor_ref requires container_name")
    return ExecutorRef(
        kind=kind,
        executor_id=str(raw.get("id") or raw.get("container_name") or uuid.uuid4().hex[:12]),
        network=network,
        mappings=tuple(_dedupe_mappings(mappings)),
        container_name=str(raw.get("container_name") or raw.get("container") or "").strip(),
    )


def _normalize_backend_path(path_text: str) -> str:
    normalized = str(path_text or "").replace("\\", "/").strip()
    if not normalized.startswith("/"):
        raise ValueError("executor_ref backend_path must be an absolute backend path")
    raw_parts = [part for part in normalized.split("/") if part]
    if any(part in {".", ".."} for part in raw_parts):
        raise ValueError("executor_ref backend_path must not contain traversal segments")
    normalized = posixpath.normpath(normalized)
    if normalized in {"", ".", "/"}:
        raise ValueError("executor_ref backend_path must not be empty or backend root")
    return normalized


def _dedupe_mappings(mappings: list[PathMapping]) -> list[PathMapping]:
    seen: set[tuple[str, str]] = set()
    result: list[PathMapping] = []
    for mapping in mappings:
        key = (str(mapping.host_path), mapping.backend_path.rstrip("/"))
        if key in seen:
            continue
        seen.add(key)
        result.append(mapping)
    result.sort(key=lambda item: len(str(item.host_path)), reverse=True)
    return result


def map_host_path(executor: ExecutorRef, path: pathlib.Path) -> str:
    host = pathlib.Path(path).expanduser().resolve(strict=False)
    for mapping in executor.mappings:
        if not path_is_relative_to(host, mapping.host_path):
            continue
        rel = host.relative_to(mapping.host_path).as_posix()
        base = mapping.backend_path.rstrip("/")
        return base if not rel or rel == "." else f"{base}/{rel}"
    if executor.managed:
        raise ManagedPathRefused(f"{host} is outside this deployment's execution domain, which maps only its work "
                                 "paths; nothing was started, and nothing runs on the service instead")
    raise ValueError(f"path is outside executor mappings: {host}")


def map_backend_path(executor: ExecutorRef, path_text: str) -> pathlib.Path:
    return map_backend_path_lexical(executor, path_text).resolve(strict=False)


def map_backend_path_lexical(executor: ExecutorRef, path_text: str) -> pathlib.Path:
    """Map a backend path without resolving descendant symlinks.

    Policy callers need the lexical spelling to detect a path that originated
    under a configured output root before canonicalization can move it into a
    different allowed root.  Execution and ordinary path consumers continue to
    use ``map_backend_path`` and receive the resolved host path.
    """
    normalized = str(path_text or "").replace("\\", "/").rstrip("/")
    if not normalized.startswith("/"):
        raise BackendPathNoMatch(f"backend path is not absolute: {path_text}")
    for mapping in sorted(executor.mappings, key=lambda item: len(item.backend_path.rstrip("/")), reverse=True):
        base = str(mapping.backend_path or "").replace("\\", "/").rstrip("/")
        if not base:
            continue
        if normalized != base and not normalized.startswith(base + "/"):
            continue
        rel_text = normalized[len(base):].lstrip("/")
        rel_parts = [part for part in rel_text.split("/") if part and part != "."]
        if any(part == ".." for part in rel_parts):
            raise ValueError(f"backend path escapes executor mapping: {path_text}")
        return mapping.host_path.joinpath(*rel_parts)
    raise BackendPathNoMatch(f"path is outside executor backend mappings: {path_text}")


class OutputResolutionUnavailable(RuntimeError):
    """Docker could not establish an output address; no host fallback is valid."""


class DockerOutputAbsent(OutputResolutionUnavailable):
    """The original address is absent in the container and at its confirmed host bind."""


@dataclass(frozen=True)
class DockerOutputPath:
    requested: str
    backend_path: str
    source: pathlib.Path
    lexical_source: pathlib.Path
    # Relative mountpoint -> confirmed host source. These replace hidden host entries.
    mounts: tuple[tuple[str, pathlib.Path], ...] = ()


@dataclass(frozen=True)
class DockerOutputContext:
    executor: ExecutorRef
    backend_cwd: str
    paths: dict[str, DockerOutputPath]
    # A later capture (``pinned``): declared spelling -> why that path alone could not be addressed. Its consumer
    # leaves it unconfirmed and never resolves it on the host instead; its siblings are captured.
    unresolved: dict[str, str] = field(default_factory=dict)


# POSIX sh builtins plus ordinary readlink; no Python, realpath or GNU options.
# Missing components remain prospective, but returning through .. resumes physical
# traversal. Symlinks expand BEFORE a later ..; cd -P also respects mount parents.
# Export first tests the original address: prospective resolution can erase a
# missing prefix before .. or a final slash's directory requirement.
_OUTPUT_PATH_PROBE = r'''
set -f
fail() { printf '%s\n' "$*" >&2; exit 73; }
missing=
if [ "${2:-}" = existing ] && [ ! -e "$1" ]; then missing=1; fi
physical_dir() {
    resolved=$(CDPATH= cd -P "$1" && pwd -P && printf '.') || fail 'directory unavailable'
    resolved=${resolved%.}; resolved=${resolved%?}
}
command -v readlink >/dev/null 2>&1 || fail 'readlink capability unavailable'
cap=$(readlink /proc/self/cwd) || fail 'readlink capability unavailable'
case "$cap" in /*) ;; *) fail 'readlink capability unavailable';; esac
resolved=/
pending=${1#/}
links=0
while [ -n "$pending" ]; do
    case "$pending" in
        */*) part=${pending%%/*}; pending=${pending#*/} ;;
        *) part=$pending; pending= ;;
    esac
    case "$part" in
        ''|.) continue ;;
        ..)
            if [ -d "$resolved" ]; then
                physical_dir "$resolved/.."
            else
                resolved=${resolved%/*}; [ -n "$resolved" ] || resolved=/
            fi
            continue ;;
    esac
    [ ! -d "$resolved" ] || [ -x "$resolved" ] || fail 'directory not searchable'
    candidate=${resolved%/}/$part
    if [ -L "$candidate" ]; then
        links=$((links + 1)); [ "$links" -le 40 ] || fail 'too many symlinks'
        target=$(readlink "$candidate" && printf '.') || fail 'symlink unreadable'
        target=${target%.}; target=${target%?}
        [ -n "$target" ] || fail 'empty symlink'
        case "$target" in /*) resolved=/; target=${target#/} ;; esac
        pending=$target${pending:+/$pending}
    elif [ -d "$candidate" ]; then
        physical_dir "$candidate"
    elif [ -e "$candidate" ]; then
        [ -z "$pending" ] || fail 'non-directory path component'
        resolved=$candidate
    else
        resolved=$candidate
    fi
done
[ -z "$missing" ] || printf 'missing\000'
printf '%s\000' "$resolved"
'''


def _docker_output_probe(argv: list[str]) -> str:
    try:
        result = subprocess.run(["docker", *argv], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise OutputResolutionUnavailable(f"Docker output probe unavailable: {exc}") from exc
    if result.returncode:
        raise OutputResolutionUnavailable(f"Docker output probe failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode("utf-8", errors="surrogateescape")


def _output_container(executor: ExecutorRef) -> tuple[ExecutorRef, list[dict[str, Any]]]:
    text = _docker_output_probe([
        "inspect", "-f",
        '{"id":{{json .Id}},"running":{{json .State.Running}},'
        '"network":{{json .HostConfig.NetworkMode}},"mounts":{{json .Mounts}},'
        '"tmpfs":{{json (index .HostConfig "Tmpfs")}}}',
        executor.container_name,
    ])
    try:
        facts = json.loads(text)
        identity, mounts = facts["id"], facts["mounts"]
        if not isinstance(identity, str) or not identity or not isinstance(mounts, list) or not facts["running"]:
            raise ValueError("container identity/mounts unavailable or container stopped")
        if executor.network == "none" and facts["network"] != "none":
            raise ValueError("executor requires Docker network=none")
        # --tmpfs destinations are separate from .Mounts, but hide host binds too.
        tmpfs = facts.get("tmpfs")
        if tmpfs is not None:
            if not isinstance(tmpfs, dict):
                raise ValueError("container tmpfs facts incomplete")
            mounts.extend({"Type": "tmpfs", "Destination": dest} for dest in tmpfs)
        for mount in mounts:
            if not isinstance(mount, dict) or not str(mount.get("Destination", "")).startswith("/"):
                raise ValueError("container mount facts incomplete")
    except (KeyError, TypeError, ValueError) as exc:
        raise OutputResolutionUnavailable(f"Docker output inspection unavailable: {exc}") from exc
    return replace(executor, container_name=identity), mounts


def pinned_container(executor: ExecutorRef) -> ExecutorRef:
    """A launch without declared outputs has nothing else pin its container: the configured name resolved once,
    before any backend effect, to the running container's immutable id. Every later probe of the launch (custody
    census, replacement, retained capture) reads that id, so a name a successor container reuses cannot answer
    for it. Facts that cannot be read start nothing (``process_not_started``)."""
    try:
        return _output_container(executor)[0]
    except OutputResolutionUnavailable as exc:
        refused = OutputResolutionUnavailable(f"Docker container identity unavailable: {exc}")
        refused.process_not_started = True
        raise refused from exc


def _under_backend(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip("/") + "/")


def _confirmed_output_source(executor: ExecutorRef, path: str, mounts: list[dict[str, Any]]) -> pathlib.Path:
    covering = [m for m in mounts if _under_backend(path, m["Destination"])]
    if not covering:
        raise ValueError(f"Docker output has no confirmed bind mount: {path}")
    mount = max(covering, key=lambda m: len(m["Destination"]))
    if mount.get("Type") != "bind" or not mount.get("Source"):
        raise ValueError(f"Docker output mount is not a mapped host bind: {path}")
    # Never resolve the output's symlinks on the host: the container already did so.
    try:
        host = pathlib.Path(mount["Source"]).resolve(strict=False) / posixpath.relpath(path, mount["Destination"])
        physical_host = host.resolve(strict=False)
    except OSError as exc:
        raise OutputResolutionUnavailable(f"Docker output bind source unavailable: {path}: {exc}") from exc
    try:
        declared = map_backend_path_lexical(executor, path)
    except BackendPathNoMatch as exc:
        raise ValueError(f"Docker output is outside declared mappings: {path}") from exc
    if declared != host:
        raise ValueError(f"Docker output mapping disagrees with mounted source: {path}")
    if physical_host != host:
        raise OutputResolutionUnavailable(f"Docker output host view disagrees with physical container path: {path}")
    return host


def resolve_docker_output(executor: ExecutorRef, requested: str,
                         mounts: list[dict[str, Any]] | None = None, *, must_exist: bool = False) -> DockerOutputPath:
    if mounts is None:
        observed, mounts = _output_container(executor)
        if observed.container_name != executor.container_name:
            raise OutputResolutionUnavailable("Docker output container identity changed")
    resolved = _docker_output_probe(["exec", executor.container_name, "sh", "-c", _OUTPUT_PATH_PROBE,
                                     "output-path", requested, "existing" if must_exist else "prospective"])
    missing = must_exist and resolved.startswith("missing\0")
    if missing:
        resolved = resolved.removeprefix("missing\0")
    if not resolved.endswith("\0") or "\0" in resolved[:-1] or not resolved.startswith("/"):
        raise OutputResolutionUnavailable("Docker output path probe returned invalid framing")
    resolved = resolved[:-1]
    source = _confirmed_output_source(executor, resolved, mounts)
    if missing:
        try:
            source.lstat()
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise DockerOutputAbsent("missing output") from exc
        raise OutputResolutionUnavailable("missing output or unavailable original address")
    # Lexical provenance is only for policy, never for choosing exported bytes.
    try:
        lexical = map_backend_path_lexical(executor, posixpath.normpath(requested))
    except BackendPathNoMatch:
        lexical = source
    overlays = tuple(
        (posixpath.relpath(m["Destination"], resolved),
         _confirmed_output_source(executor, m["Destination"], mounts))
        for m in mounts if m["Destination"] != resolved and _under_backend(m["Destination"], resolved)
    )
    return DockerOutputPath(requested, resolved, source, lexical, overlays)


def prepare_docker_outputs(executor: ExecutorRef | None, work_dir: pathlib.Path,
                           outputs: list[str] | None, *, pinned: bool = False) -> DockerOutputContext | None:
    """Select only the Docker execution route; pin its identity before any effect.

    An absolute address in a declared backend mapping is a backend address; other
    absolute addresses and ~ keep explicit host semantics. Relative outputs always
    use the selected backend, including spellings without traversal components.
    One container inspection answers the whole operation. Before a launch one path
    that cannot be addressed refuses the start; a ``pinned`` capture of an ended
    launch records it (``unresolved``) and still addresses the others.
    """
    if not outputs or executor is None or executor.kind != "docker_exec":
        return None
    try:
        cwd = map_host_path(executor, work_dir)
    except ValueError:
        if pinned:
            raise
        return None  # The process consumer selects its ordinary host route too.
    observed, mounts = _output_container(executor)
    if pinned and observed.container_name != executor.container_name:
        raise OutputResolutionUnavailable("Docker output container identity changed")
    executor = observed
    paths, unresolved = {}, {}
    for item in outputs:
        text = str(item or "").strip()
        if not text or text.startswith("~"):
            continue
        try:
            if "\0" in text:
                raise ValueError("declared output contains NUL")
            if not text.startswith("/"):
                # A Windows host absolute address is still an explicit host address.
                if pathlib.PureWindowsPath(text).is_absolute():
                    continue
                requested = cwd.rstrip("/") + "/" + text
            elif any(_under_backend(text, m.backend_path) for m in executor.mappings):
                requested = text
            else:
                continue
            paths[text] = resolve_docker_output(executor, requested, mounts)
        except (OutputResolutionUnavailable, ValueError) as exc:
            if not pinned:
                raise
            unresolved[text] = type(exc).__name__
    return DockerOutputContext(executor, cwd, paths, unresolved)
