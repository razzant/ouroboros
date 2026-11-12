"""Work-domain effects for a managed deployment: what acts AS the work domain, never as the protected service.

Under a managed ``execution`` (``execution_domain``) a folder the pinned executor maps belongs to the work domain.
Every effect on a work path runs in the pinned container through C1's custody (``execution_domain.attached_launch``:
the executor's foreground record before the start, then settlement of the launch's whole process group; Stop now
and Panic reach it there): a process started in a work folder — Git included, whose configuration, attributes,
hooks, filters, fsmonitor, credential helpers and ssh commands are the repository's own — and a file written,
renamed or removed. Bytes cross on the CLI's standard streams; folders cross as tar streams the receiver checks.
Nothing that fails there is retried on the service: a refusal before the start says nothing started, an end the
container did not confirm keeps the executor's record and reaches every consumer as ``WorkDomainUnsettled`` (an
``OSError`` carrying what arrived), never as an ordinary success.

The service still reads work bytes for intake (a file a tool shows, a capture's members) only through
descriptor-confined opens anchored at the mapping's own root (``confined_files``): no descendant link is followed
and no pathname is reopened, so an ancestor swapped after a check reaches nothing of the service. A read that a
link or a permission interrupts is the work domain's own (``cat`` in the container).

Which domain a path belongs to is read from its spelling (``work_spelling``): lexically inside a mapping, or a
service path whose resolution lands in one; never a service-side ``resolve()`` that an employee's symlink could
steer. Unmanaged, and for every service path, each reader and each folder helper (``make_dirs``, ``remove_tree``,
``copy_file``, ``private_dir``, ``discard``) is exactly the local operation it replaces; the other writers take work
paths only and refuse any other before anything starts.
"""

from __future__ import annotations

import contextlib
import errno
import io
import logging
import os
import pathlib
import posixpath
import shutil
import stat
import subprocess
import tarfile
import tempfile
import uuid
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

_WRITE_TIMEOUT_SEC = 120.0  # one container file effect: the transport's bound, never a judgment
_READ_TIMEOUT_SEC = 120.0
# The variables a work-domain command may receive from the service: Git's and the locale's. Everything else the
# caller's environment holds is the service's own (credentials, launch facts) and never crosses.
_FORWARDED = ("GIT_", "LC_")
_FORWARDED_NAMES = frozenset({"LANG", "LANGUAGE", "TZ"})
_GIT_LOCATION_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                     "GIT_ALTERNATE_OBJECT_DIRECTORIES")


class WorkDomainRefused(ValueError):
    """Refused before anything started anywhere: the operation names what its domain cannot see."""

    process_not_started = True


class WorkDomainUnsettled(OSError):
    """A work-domain process ended without the container confirming its group ended. Its executor record is kept
    for settlement (Panic, shutdown); what it produced stands in ``result`` and its effect may have happened."""

    process_not_started = False

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


class Link(str):
    """A symbolic link's own text: a member ``import_tree`` places as a link, never a file of these bytes."""


def _confirmed(result: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
    """``result`` when the container confirmed its group ended; else ``WorkDomainUnsettled`` carrying it, so no
    consumer reads an unconfirmed end as an ordinary one."""
    if getattr(result, "backend_settled", True):
        return result
    raise WorkDomainUnsettled(f"the work domain's {what} ended (exit {result.returncode}) without the container "
                              "confirming its process group ended; its effect may have happened, and its executor "
                              "record is kept for settlement", result)


# --- which domain a path belongs to ---------------------------------------------------------------------------

def _domain() -> Any:
    from ouroboros import execution_domain

    return execution_domain.current()


def _lexical(path: Any) -> str:
    return posixpath.normpath(os.path.abspath(os.fspath(path)))


def _rest(path: str, root: str) -> Optional[str]:
    if path == root:
        return ""
    prefix = root if root.endswith("/") else root + "/"
    return path[len(prefix):] if path.startswith(prefix) else None


def _mapping_of(domain: Any, host: str) -> Optional[Tuple[str, str, str]]:
    """``(host root, container root, rest)`` of the most specific mapping holding ``host`` by spelling."""
    best = None
    for root, backend in domain.work_roots:
        rest = _rest(host, root)
        if rest is not None and (best is None or len(root) > len(best[0])):
            best = (root, backend, rest)
    return best


def managed() -> bool:
    return _domain() is not None


def work_spelling(path: Any) -> str:
    """The service's spelling of ``path`` in the work domain, else "" (unmanaged, or the service's own path).

    A path inside a mapping by spelling is the work's whatever its resolution says now (an employee may have made
    one of its folders a link); a service path that resolves into a mapping is the work's too, at that resolution."""
    domain = _domain()
    if domain is None or path is None or os.fspath(path) == "":
        return ""
    lexical = _lexical(path)
    if _mapping_of(domain, lexical) is not None:
        return lexical
    # A service path entering a mapping through a link: from the first resolved prefix inside it, the rest of the
    # spelling is the work's to resolve (a later link of the work's must not lead the service back out).
    current, parts = "/", [part for part in lexical.split("/") if part]
    for index, part in enumerate(parts):
        current = os.path.realpath(posixpath.join(current, part))
        if _mapping_of(domain, current) is not None:
            return posixpath.join(current, *parts[index + 1:])
    return ""


def is_work(path: Any) -> bool:
    return bool(work_spelling(path))


def settle(path: Any, *, identical: bool = False) -> pathlib.Path:
    """``path`` as a resource binding keeps it: a work path by its spelling (the container resolves it; the
    service must not), any other path resolved exactly as before. ``identical`` refuses, before anything starts, a
    work path the container spells otherwise (Git there, and a native run's engine, name its folders by that)."""
    spelling = work_spelling(path)
    if spelling and identical and not identity(spelling):
        raise WorkDomainRefused(f"{path} is mapped into the work domain at another address; this needs an identical "
                                "mapping (the work names its folders as the service does); nothing was started")
    return pathlib.Path(spelling) if spelling else pathlib.Path(path).resolve(strict=False)


def container_path(path: Any) -> str:
    """The container's spelling of a work path."""
    spelling = work_spelling(path)
    found = _mapping_of(_domain(), spelling) if spelling else None
    if found is None:
        raise WorkDomainRefused(f"{path} is not a work-domain path; nothing was started")
    _root, backend, rest = found
    return posixpath.join(backend, rest) if rest else backend


def identity(path: Any) -> bool:
    """Whether the container spells this work path as the service does (an identical mapping): what Git prints
    there, and what a native run's engine reads, is then the service's spelling too."""
    try:
        return container_path(path) == work_spelling(path)
    except WorkDomainRefused:
        return False


def staging_path(*parts: str) -> str:
    """``<staging_root>/<parts>``: the administrator's work folder for what the service prepares for the domain
    (private copies, staged inputs, session scratch). Refused, before anything starts, when the domain names none."""
    domain = _domain()
    if domain is None or not domain.staging_root:
        raise WorkDomainRefused("This deployment's execution domain names no staging_root, where the work keeps what "
                                "the service prepares for it (private copies, staged inputs, session scratch, skill "
                                "payloads); nothing was started, and nothing runs on the service instead")
    return posixpath.join(domain.staging_root, *[part.strip("/") for part in parts if part])


# --- one process in the work container, under the executor's custody ------------------------------------------

def _stdin_source(stdin: Any) -> Tuple[Any, Optional[BinaryIO]]:
    """``(Popen stdin, file this call owns)``: bytes ride an unlinked temporary file (never a pipe the wait could
    deadlock on), a readable file goes as it is, nothing is an empty stream."""
    if stdin is None or stdin is subprocess.DEVNULL:
        return subprocess.DEVNULL, None
    if isinstance(stdin, (bytes, bytearray, str)):
        owned = tempfile.TemporaryFile()
        owned.write(stdin.encode("utf-8") if isinstance(stdin, str) else bytes(stdin))
        owned.seek(0)
        return owned, owned
    return stdin, None


def run_in_work(argv: Sequence[Any], *, cwd: Any = None, env: Optional[Dict[str, str]] = None, stdin: Any = None,
                stdout: Any = subprocess.PIPE, stderr: Any = subprocess.PIPE, timeout_sec: Optional[float] = None,
                stop: Any = None) -> subprocess.CompletedProcess:
    """Run ``argv`` AS the work domain, under the executor's custody; returns its bytes.

    ``cwd`` is a work path the container enters itself (the service never resolves it); ``env`` holds the caller's
    own values in the container's spelling and rides argv (``env``), so it may hold no secret; ``stdin`` is bytes or
    a readable file; ``stdout``/``stderr`` PIPE (captured), a writable file (streamed) or DEVNULL. Before the start
    a refusal (no CLI, an unmapped folder, a ``network: none`` breach) says nothing started. After it the deadline
    or Stop now stops the launch's whole group in the container first (``AttachedLaunch.settle``), then the CLI;
    ``backend_settled`` on the result says whether the container confirmed the group ended (False keeps the
    executor's record, which Panic and shutdown settle)."""
    from ouroboros.execution_domain import attached_launch
    from ouroboros.platform_layer import subprocess_new_group_kwargs
    from ouroboros.tool_custody import external_handoff
    from ouroboros.tools.shell_process import (
        _kill_process_group,
        communicate_or_stop,
        contained_popen,
        retire_tracked,
        tracked_start,
    )

    words = [os.fspath(part) if isinstance(part, os.PathLike) else str(part) for part in argv]
    where = [container_path(cwd)] if cwd is not None else []
    script = ('cd -- "$1" 2>/dev/null || { printf "the work domain cannot enter %s\\n" "$1" >&2; exit 125; }; '
              'shift; exec env "$@"' if where else 'exec env "$@"')
    assignments = [f"{key}={value}" for key, value in (env or {}).items()]
    launch = attached_launch(["sh", "-c", script, "sh", *where, *assignments, *words], None, {})
    if launch is None:
        raise WorkDomainRefused("no managed execution domain: a work-domain process has nowhere to run")
    launch.require_cli()
    launch.adopt()
    confirmed = external_handoff("work_domain_process")
    source, owned = _stdin_source(stdin)
    proc = None
    try:
        with tracked_start() as publish:
            proc = contained_popen(subprocess.Popen, launch.argv, env=launch.env, stdin=source, stdout=stdout,
                                   stderr=stderr, **subprocess_new_group_kwargs())
            publish(proc)
    except BaseException:
        if proc is None:  # Popen returned nothing: positive proof that no CLI, so nothing in the container, ran
            confirmed()
            launch.abandon()
        if owned is not None:
            owned.close()
        raise
    settled: Optional[bool] = None

    def end() -> bool:  # Stop now, or the deadline: the container's group first, then the CLI
        nonlocal settled
        settled = launch.settle()
        _kill_process_group(proc)
        return settled

    try:
        try:
            out, err = communicate_or_stop(proc, timeout_sec, stop, end)
        except subprocess.TimeoutExpired:
            end()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            raise
    finally:
        retire_tracked(proc)
        if proc.poll() is not None:
            confirmed()
        if settled is None:  # the CLI's end is not the group's: confirmed in the container, else kept recorded
            settled = launch.settle()
        if owned is not None:
            owned.close()
    result = subprocess.CompletedProcess(words, proc.returncode, out, err)
    result.backend_settled = bool(settled)
    if not settled:
        log.warning("Work-domain process %s ended without the container confirming its group ended; its executor "
                    "record is kept for settlement", words[:3])
    return result


def _forwarded(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """The variables of a caller's environment that a work-domain command receives: only what the caller set or
    changed beyond the service's own environment, and only Git's and the locale's. A path-valued one must name a
    work path (translated to the container's spelling) or ``/dev/null``: a service path is refused."""
    if env is None:
        return {}
    chosen: Dict[str, str] = {}
    for key, value in env.items():
        value = str(value)
        if os.environ.get(key) == value or not (key.startswith(_FORWARDED) or key in _FORWARDED_NAMES):
            continue
        if value.startswith("/") and value != os.devnull:
            if not work_spelling(value):
                raise WorkDomainRefused(f"{key} names {value}, a service path the work domain does not see; "
                                        "nothing was started, and nothing runs on the service instead")
            value = container_path(value)
        chosen[key] = value
    return chosen


def _location_options(cmd: Sequence[str]) -> List[str]:
    """The folders a Git command names before its subcommand (``-C``, ``--git-dir``, ``--work-tree``)."""
    found, words = [], list(cmd[1:])
    while words:
        word = words.pop(0)
        if word in ("-C", "--git-dir", "--work-tree") and words:
            found.append(words.pop(0))
        elif word.startswith(("--git-dir=", "--work-tree=")):
            found.append(word.split("=", 1)[1])
        elif word == "-c" and words:
            words.pop(0)
        elif not word.startswith("-"):
            break
    return found


def _work_location(cmd: Sequence[Any], cwd: Any, env: Optional[Dict[str, str]]) -> str:
    """Where a command acts when that is the work domain's (the service spelling of its folder), else "".

    A folder it names that belongs to the other domain refuses before anything starts, either way: a work command
    cannot reach a service folder, and a service command must not touch a work one."""
    words = [str(part) for part in cmd]
    named = [*((cwd,) if cwd is not None else ()),
             *(_location_options(words) if words and posixpath.basename(words[0]) == "git" else ()),
             *(str(env[key]) for key in _GIT_LOCATION_ENV if env and env.get(key) and env.get(key) != os.environ.get(key))]
    if not named:
        return ""
    base = cwd if cwd is not None else os.getcwd()
    spellings = [work_spelling(item if os.path.isabs(os.fspath(item)) else os.path.join(os.fspath(base), os.fspath(item)))
                 for item in named]
    if not any(spellings):
        return ""
    if not all(spellings):
        raise WorkDomainRefused("this command names both a work folder and a service folder; nothing was started")
    return spellings[0] if cwd is not None else work_spelling(base) or spellings[0]


def run(cmd: Sequence[Any], **kwargs: Any) -> subprocess.CompletedProcess:
    """``subprocess.run`` for a command that may act on a work folder: unchanged (the very same call) unless its
    folder is the work domain's (``_work_location``); then it runs there (``run_in_work``) with its own options —
    ``input``/``stdin``, captured or streamed output, ``text``/``encoding``/``errors``, ``timeout`` and ``check`` —
    and only Git's and the locale's variables crossing (``_forwarded``). Absolute work paths in its arguments take
    the container's spelling; any other argument is the command's own to read there."""
    where = _work_location(cmd, kwargs.get("cwd"), kwargs.get("env"))
    if not where:
        return subprocess.run(cmd, **kwargs)
    capture = kwargs.pop("capture_output", False)
    stdout = subprocess.PIPE if capture else kwargs.pop("stdout", None)
    stderr = subprocess.PIPE if capture else kwargs.pop("stderr", None)
    text = bool(kwargs.pop("text", False) or kwargs.pop("universal_newlines", False))
    encoding, errors = kwargs.pop("encoding", None), kwargs.pop("errors", None)
    text = text or encoding is not None or errors is not None
    data = kwargs.pop("input", None)
    stdin = data if data is not None else kwargs.pop("stdin", None)
    check, timeout = kwargs.pop("check", False), kwargs.pop("timeout", None)
    env = _forwarded(kwargs.pop("env", None))
    kwargs.pop("cwd", None)
    stop = kwargs.pop("stop", None)
    for ignored in ("close_fds", "start_new_session", "creationflags", "startupinfo"):
        kwargs.pop(ignored, None)
    if kwargs:
        raise TypeError(f"work-domain run does not take {sorted(kwargs)}")
    words = [str(part) for part in cmd]
    words = [container_path(word) if word.startswith("/") and word != os.devnull and work_spelling(word) else word
             for word in words]
    result = run_in_work(words, cwd=where, env=env, stdin=stdin, stdout=stdout, stderr=stderr,
                         timeout_sec=timeout, stop=stop)
    out, err = result.stdout, result.stderr
    if text:
        decode = lambda raw: (raw.decode(encoding or "utf-8", errors or "strict").replace("\r\n", "\n").replace("\r", "\n")  # noqa: E731
                              if isinstance(raw, bytes) else raw)
        out, err = decode(out), decode(err)
    completed = subprocess.CompletedProcess(list(cmd), result.returncode, out, err)
    completed.backend_settled = result.backend_settled
    _confirmed(completed, " ".join(words[:2]))  # an unconfirmed end raises, its bytes in the exception's result
    if check and completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, list(cmd), out, err)
    return completed


def run_into(sink: BinaryIO, cmd: Sequence[Any], **kwargs: Any) -> Tuple[subprocess.CompletedProcess, int]:
    """``run`` whose standard output is appended to the service file ``sink`` once the command ended, through a
    service temporary (``sink`` keeps its own write order): ``(result, bytes appended)``. An unconfirmed end still
    appends what arrived, then raises (``WorkDomainUnsettled``, with ``written``)."""
    written, unsettled = 0, None
    with tempfile.TemporaryFile() as out:
        try:
            done = run(cmd, stdout=out, **kwargs)
        except WorkDomainUnsettled as exc:
            done, unsettled = exc.result, exc
        out.seek(0)
        while chunk := out.read(1024 * 128):
            sink.write(chunk)
            written += len(chunk)
    if unsettled is not None:
        unsettled.written = written
        raise unsettled
    return done, written


# --- files in the work domain ---------------------------------------------------------------------------------

def _effect(verb: str, path: str, argv: List[str], *, data: Any = None, timeout_sec: float = _WRITE_TIMEOUT_SEC) -> None:
    result = run_in_work(argv, stdin=data, timeout_sec=timeout_sec)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()[:300] or str(result.returncode)
        if not result.backend_settled:
            raise WorkDomainUnsettled(f"the work domain did not {verb} {path} ({detail}), and the container did not "
                                      "confirm its process group ended; its executor record is kept", result)
        raise OSError(f"the work domain did not {verb} {path} ({detail})")
    _confirmed(result, f"{verb} of {path}")


def write_file(path: Any, data: Any, *, append: bool = False, mode: Optional[int] = None) -> None:
    """Write ``data`` to a work file AS the work domain: its folder made there, an overwrite landing atomically (a
    sibling temporary, an existing file's mode kept, or exactly ``mode``, then ``mv``), an append growing the file.
    ``data`` is bytes, a readable binary file, or a service file's path: the latter two stream on stdin as they
    are, never read whole. The bytes never reach a shell; the container resolves the path, so an ancestor swap
    reaches only the work."""
    if not isinstance(data, (bytes, bytearray)) and not hasattr(data, "read"):
        if work_spelling(data):  # only the service's own file is opened here; a work file is the work's to copy
            raise WorkDomainRefused(f"{data} is a work path, not a service file; nothing was written")
        with open(data, "rb") as stream:
            write_file(path, stream, append=append, mode=mode)
        return
    target = container_path(path)
    if append:
        script = 'mkdir -p -- "${1%/*}" && cat >> "$1"'
        argv = ["sh", "-c", script, "sh", target]
    else:
        keep = ('chmod "$3" "$2"' if mode is not None else
                '{ if [ -e "$1" ]; then chmod --reference="$1" "$2" 2>/dev/null || true; fi; }')
        script = (f'mkdir -p -- "${{1%/*}}" && cat > "$2" && {keep} && mv -f -- "$2" "$1" || '
                  '{ rm -f -- "$2"; exit 1; }')
        argv = ["sh", "-c", script, "sh", target, f"{target}.ouroboros-{uuid.uuid4().hex}.tmp",
                format(mode & 0o7777, "o") if mode is not None else ""]
    _effect("write", str(path), argv, data=bytes(data) if isinstance(data, (bytes, bytearray)) else data)


def make_symlink(path: Any, link_target: str) -> None:
    """Place a link with text ``link_target`` at a work path AS the work domain (a sibling link, then ``mv``)."""
    target = container_path(path)
    script = 'mkdir -p -- "${1%/*}" && ln -s -- "$3" "$2" && mv -f -- "$2" "$1" || { rm -f -- "$2"; exit 1; }'
    _effect("link", str(path), ["sh", "-c", script, "sh", target, f"{target}.ouroboros-{uuid.uuid4().hex}.tmp",
                                link_target])


def write_text(path: Any, text: str, *, append: bool = False) -> None:
    write_file(path, text.encode("utf-8"), append=append)


def remove(path: Any) -> None:
    """Remove a work file (``rm -f``: an absent one is not an error) AS the work domain."""
    _effect("remove", str(path), ["rm", "-f", "--", container_path(path)])


def remove_tree(path: Any, *, local: Any = None) -> None:
    """Remove a folder and everything in it: a work folder AS the work domain (``rm -rf``), any other by the caller's
    own ``local`` removal (``shutil.rmtree`` by default)."""
    if not work_spelling(path):
        return (local or shutil.rmtree)(path)
    _effect("remove", str(path), ["rm", "-rf", "--", container_path(path)])


def copy_file(source: Any, target: Any) -> None:
    """``shutil.copy2`` (mode and times kept); between work paths ``cp -p`` AS the work domain."""
    if not (work_spelling(source) or work_spelling(target)):
        shutil.copy2(source, target)
        return
    _effect("copy", str(source), ["cp", "-p", "--", container_path(source), container_path(target)])


def remove_empty_dir(path: Any) -> None:
    """``rmdir`` of a work folder AS the work domain: an ``OSError`` when it still holds anything, as ``rmdir``."""
    _effect("remove", str(path), ["rmdir", "--", container_path(path)])


def make_dir(path: Any) -> None:
    """One new work folder AS the work domain (``mkdir`` without ``-p``: an existing entry is an ``OSError``)."""
    _effect("make", str(path), ["mkdir", "--", container_path(path)])


def make_dirs(path: Any) -> None:
    """``Path.mkdir(parents=True, exist_ok=True)``; a work folder's ``mkdir -p`` AS the work domain."""
    if not work_spelling(path):
        return pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    _effect("make", str(path), ["mkdir", "-p", "--", container_path(path)])


def private_dir(beside: Any, prefix: str) -> pathlib.Path:
    """A fresh private folder in ``beside``'s domain (``discard`` removes it): under the staging root's ``tmp``, made
    AS the work domain, for a work path; else ``tempfile.mkdtemp(prefix=prefix)`` as before."""
    if not work_spelling(beside):
        return pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    folder = pathlib.Path(staging_path("tmp", f"{prefix}{uuid.uuid4().hex}"))
    make_dirs(folder)
    return folder


def discard(folder: Any) -> None:
    """Best-effort removal of a private folder: ``shutil.rmtree(ignore_errors=True)``, or AS the work domain."""
    with contextlib.suppress(Exception):
        remove_tree(folder, local=lambda path: shutil.rmtree(path, ignore_errors=True))


def rename(source: Any, target: Any) -> None:
    _effect("rename", str(source), ["mv", "-f", "--", container_path(source), container_path(target)])


def read_in_work(path: Any, *, limit: Optional[int] = None) -> bytes:
    """A work file's bytes as the work domain reads it (its links followed in the container)."""
    target = container_path(path)
    argv = ["head", "-c", str(int(limit)), "--", target] if limit is not None else ["cat", "--", target]
    result = run_in_work(argv, timeout_sec=_READ_TIMEOUT_SEC)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        if "No such file" in detail or "not found" in detail.lower():
            raise FileNotFoundError(errno.ENOENT, "no such work file", str(path))
        if "Is a directory" in detail:
            raise IsADirectoryError(errno.EISDIR, "a work folder, not a file", str(path))
        raise OSError(f"the work domain did not read {path} ({detail[:300] or result.returncode})")
    return _confirmed(result, f"read of {path}").stdout


def _confined_link(rel: str, link: str) -> bool:
    """Whether a link at ``rel`` naming ``link`` stays inside the folder that holds ``rel``'s tree."""
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(rel), link))
    return bool(link) and not link.startswith("/") and joined != ".." and not joined.startswith("../")


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _received_parent(top: int, parts: Sequence[str], held: Dict[Tuple[str, ...], int]) -> int:
    """The held folder for a received member's parent ``parts``, descended from ``top`` by plain names: a missing
    folder is made there, an existing name opens only as a folder of its own (never through a link, so a received
    link, composed links or a swapped name refuse the intake instead of redirecting a write)."""
    folder = top
    for depth, part in enumerate(parts, 1):
        key = tuple(parts[:depth])
        if key in held:
            folder = held[key]
            continue
        with contextlib.suppress(FileExistsError):
            os.mkdir(part, 0o755, dir_fd=folder)
        try:
            folder = held[key] = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=folder)
        except OSError as exc:
            raise WorkDomainRefused(f"the work domain's export placed a member below {'/'.join(key)!r}, which is not "
                                    "a folder of its own") from exc
    return folder


# --- folders crossing as tar streams --------------------------------------------------------------------------

def import_tree(destination: Any, members: Sequence[Tuple[str, Any, int]]) -> None:
    """Materialize ``(relative path, bytes, a service file, a confined ``Link`` or None for a folder, mode)`` members
    under a work folder AS the work domain: one tar stream, spooled through a service temporary (never held whole), that the container
    unpacks (``tar -x``, no owner, no absolute or parent paths: the service wrote every member from its own bytes or
    files, and a link only where its text stays inside the folder)."""
    target = container_path(destination)
    with tempfile.TemporaryFile() as spool:
        with tarfile.open(fileobj=spool, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for rel, data, mode in members:
                parts = pathlib.PurePosixPath(rel).parts
                if not parts or rel.startswith("/") or ".." in parts:
                    raise WorkDomainRefused(f"{rel!r} is not a relative member path; nothing was sent")
                info = tarfile.TarInfo(rel)
                info.mode = (mode & 0o777) or 0o644
                if data is None:  # a folder of its own (an empty one, which no file member would make)
                    info.type, info.mode = tarfile.DIRTYPE, (mode & 0o777) or 0o755
                    archive.addfile(info)
                    continue
                if isinstance(data, Link):
                    if not _confined_link(rel, str(data)):
                        raise WorkDomainRefused(f"{rel!r} links outside its folder; nothing was sent")
                    info.type, info.linkname, info.mode = tarfile.SYMTYPE, str(data), 0o777
                    archive.addfile(info)
                    continue
                if isinstance(data, (bytes, bytearray)):
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
                    continue
                if work_spelling(data):  # only the service's own files are sent; a work file is never opened here
                    raise WorkDomainRefused(f"{data} is a work path; nothing was sent")
                with open(data, "rb") as handle:
                    info.size = os.fstat(handle.fileno()).st_size
                    archive.addfile(info, handle)
        spool.seek(0)
        script = 'mkdir -p -- "$1" && tar -x -o -f - -C "$1"'
        _effect("unpack into", str(destination), ["sh", "-c", script, "sh", target], data=spool,
                timeout_sec=_WRITE_TIMEOUT_SEC * 5)


def export_tree(root: Any, rels: Sequence[str], destination: Any, *, links: bool = False) -> List[Dict[str, Any]]:
    """Receive regular files ``rels`` of a work folder from the work domain (``tar -c`` there, its reads the work
    domain's) into the service folder ``destination``: every member must be one of ``rels`` (relative, no ``..``)
    and a regular file — a hard link to one already received is its bytes again — or, with ``links``, a link whose
    text stays inside the folder (placed as that link, never followed); anything else refuses the whole intake.
    Every member is written by plain names from the destination's held descriptor: no received link (nor composed
    links, nor a name swapped beside the intake) is ever followed by a write, a folder creation or a copy.
    Returns ``[{path, size, sha256}]`` (a link: ``{path, link}``) as received."""
    import hashlib

    wanted = [rel for rel in rels if rel]
    for rel in wanted:
        parts = pathlib.PurePosixPath(rel).parts
        if rel.startswith("/") or ".." in parts:
            raise WorkDomainRefused(f"{rel!r} is not a relative member path; nothing was read")
    if not wanted:
        return []
    received, allowed, target, done = [], set(wanted), pathlib.Path(destination), {}
    target.mkdir(parents=True, exist_ok=True)  # the service's own folder, chosen by the caller
    held: Dict[Tuple[str, ...], int] = {}
    top = os.open(target, os.O_RDONLY | _DIRECTORY)
    with contextlib.ExitStack() as cleanup, tempfile.TemporaryFile() as raw:
        cleanup.callback(lambda: [os.close(fd) for fd in [top, *held.values()]])
        # macOS tar would add AppleDouble ``._*`` metadata members (refused as unrequested); others ignore this
        result = run_in_work(["tar", "-c", "-f", "-", "-C", container_path(root), "--", *wanted],
                             env={"COPYFILE_DISABLE": "1"}, stdout=raw, timeout_sec=_READ_TIMEOUT_SEC * 5)
        if result.returncode != 0:
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()[:300]
            raise OSError(f"the work domain did not export {len(wanted)} file(s) from {root} ({detail or result.returncode})")
        raw.seek(0)
        with tarfile.open(fileobj=raw, mode="r|") as archive:
            for info in archive:
                name = info.name.rstrip("/")
                hardlink = info.islnk() and "sha256" in done.get(info.linkname.rstrip("/"), {})  # a regular one only
                symlink = links and info.issym() and _confined_link(name, info.linkname)
                if name not in allowed or name in done or not (info.isreg() or hardlink or symlink):
                    raise WorkDomainRefused(f"the work domain's export held {name!r}, not a requested regular file")
                parts = pathlib.PurePosixPath(name).parts
                if not parts or ".." in parts or name.startswith("/"):
                    raise WorkDomainRefused(f"the work domain's export held {name!r}, not a relative member path")
                folder = _received_parent(top, parts[:-1], held)
                if symlink:
                    os.symlink(info.linkname, parts[-1], dir_fd=folder)
                    done[name] = row = {"path": name, "link": info.linkname}
                    received.append(row)
                    continue
                out_fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o644, dir_fd=folder)
                out = cleanup.enter_context(os.fdopen(out_fd, "wb"))
                if hardlink:  # the same bytes under a second name: copied from the member already received
                    first = pathlib.PurePosixPath(info.linkname.rstrip("/")).parts
                    source_fd = os.open(first[-1], os.O_RDONLY | _NOFOLLOW,
                                        dir_fd=_received_parent(top, first[:-1], held))
                    with os.fdopen(source_fd, "rb") as source:
                        shutil.copyfileobj(source, out, 1024 * 1024)
                        os.fchmod(out.fileno(), os.fstat(source.fileno()).st_mode & 0o777)
                    out.close()
                    done[name] = row = {**done[info.linkname.rstrip("/")], "path": name}
                    received.append(row)
                    continue
                digest, size = hashlib.sha256(), 0
                with archive.extractfile(info) as member, out:  # in bounded chunks
                    for chunk in iter(lambda: member.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        out.write(chunk)
                    os.fchmod(out.fileno(), info.mode & 0o777)  # keep executable bits, never privilege bits
                done[name] = row = {"path": name, "size": size, "sha256": digest.hexdigest()}
                received.append(row)
    missing = allowed - set(done)
    if missing:
        raise OSError(f"the work domain's export lacked {sorted(missing)[:5]}")
    try:
        _confirmed(result, f"export from {root}")
    except WorkDomainUnsettled as exc:
        exc.received = received  # what arrived stands in the destination
        raise
    return received


# --- the service's own reads of work bytes: confined intake ---------------------------------------------------

def _anchor(spelling: str) -> Tuple[str, List[str]]:
    found = _mapping_of(_domain(), spelling)
    if found is None:
        raise WorkDomainRefused(f"{spelling} is not a work-domain path")
    root, _backend, rest = found
    return root, [part for part in rest.split("/") if part]


@contextlib.contextmanager
def _confined_parent(spelling: str) -> Iterator[Tuple[Any, str]]:
    """The held folder of a work path's parent (descended from the mapping's own root by plain names, no link
    followed) and the final name; the root itself has no final name."""
    from ouroboros.confined_files import confined_directory

    root, parts = _anchor(spelling)
    if not parts:
        raise IsADirectoryError(errno.EISDIR, "a mapping's root is a folder", spelling)
    with confined_directory(root, parts[:-1]) as directory:
        yield directory, parts[-1]


def _final_lstat(directory: Any, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=directory.handle, follow_symlinks=False)


def lstat(path: Any) -> os.stat_result:
    """A work path's own metadata (``lstat``), its folders descended without following a link; the mapping's own
    root from the descriptor the descent holds for it (the administrator's folder, as configured); any other path's
    ``os.lstat`` as before."""
    from ouroboros.confined_files import confined_directory

    spelling = work_spelling(path)
    if not spelling:
        return os.lstat(path)
    root, parts = _anchor(spelling)
    if not parts:
        with confined_directory(root, ()) as directory:
            return os.fstat(directory.handle)
    with _confined_parent(spelling) as (directory, name):
        return _final_lstat(directory, name)


def read_bytes(path: Any, *, limit: Optional[int] = None) -> bytes:
    """A file's bytes. Any other path reads as before (``store.read_file_bytes``). A work file is opened through
    the confined descent (regular files only, checked before and after the open); a link on its path, or a
    permission only the work domain holds, makes it the work domain's own read (``read_in_work``)."""
    spelling = work_spelling(path)
    if not spelling:
        from ouroboros import store

        return store.read_file_bytes(path, limit=limit)
    try:
        with _confined_parent(spelling) as (directory, name):
            info = _final_lstat(directory, name)
            if stat.S_ISLNK(info.st_mode):
                raise OSError(errno.ELOOP, "a link", spelling)
            if stat.S_ISDIR(info.st_mode):
                raise IsADirectoryError(errno.EISDIR, "a work folder, not a file", spelling)
            if not stat.S_ISREG(info.st_mode):
                raise OSError(errno.EINVAL, "not a regular file: the service never opens it", spelling)
            handle, observed = directory.open(name)
            with handle:
                if (observed.st_dev, observed.st_ino) != (info.st_dev, info.st_ino):
                    raise OSError(errno.ELOOP, "replaced while opened", spelling)
                return handle.read() if limit is None else handle.read(limit)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EACCES, errno.EPERM, errno.ENOTDIR):
            return read_in_work(spelling, limit=limit)
        raise


def measure(path: Any, sink: Any = None, *, strict: bool = False) -> Dict[str, Any]:
    """``{size, sha256}`` of a work file's bytes, copied into ``sink`` in bounded chunks: from the confined descent
    (regular files only, its identity checked after the open), else, for a link on its path, the work domain's own
    read streamed through a service temporary. ``strict`` (a policy-checked member) never falls back: a link or
    an unreadable folder on its path is the member's ``OSError``, as the service's own member check refuses it."""
    import hashlib

    spelling = work_spelling(path)
    if not spelling:
        raise WorkDomainRefused(f"{path} is not a work-domain path")
    digest, size = hashlib.sha256(), 0

    def pour(handle: Any) -> None:
        nonlocal size
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            if sink is not None:
                sink.write(chunk)

    try:
        with _confined_parent(spelling) as (directory, name):
            info = _final_lstat(directory, name)
            if stat.S_ISLNK(info.st_mode):
                raise OSError(errno.ELOOP, "a link", spelling)
            if not stat.S_ISREG(info.st_mode):
                raise OSError(errno.EINVAL, "not a regular file: the service never opens it", spelling)
            handle, observed = directory.open(name)
            with handle:
                if (observed.st_dev, observed.st_ino) != (info.st_dev, info.st_ino):
                    raise OSError(errno.ELOOP, "replaced while opened", spelling)
                pour(handle)
    except OSError as exc:
        if strict or exc.errno not in (errno.ELOOP, errno.EACCES, errno.EPERM, errno.ENOTDIR) or size:
            raise
        with tempfile.TemporaryFile() as spool:
            result = run_in_work(["cat", "--", container_path(spelling)], stdout=spool, timeout_sec=_READ_TIMEOUT_SEC)
            if result.returncode != 0:
                raise OSError(f"the work domain did not read {path}") from exc
            spool.seek(0)
            pour(spool)
            _confirmed(result, f"read of {path}")  # the bytes poured into ``sink`` stand either way
    return {"size": size, "sha256": digest.hexdigest()}


def read_text(path: Any, *, encoding: str = "utf-8", errors: str = "strict") -> str:
    """``Path.read_text`` for any other path; a work file's ``read_bytes`` decoded with the same universal newlines."""
    if not work_spelling(path):
        return pathlib.Path(path).read_text(encoding=encoding, errors=errors)
    return read_bytes(path).decode(encoding, errors).replace("\r\n", "\n").replace("\r", "\n")


def is_wal_index(path: Any) -> bool:
    """``store.is_wal_index``: a work entry is asked through its held parent (the store's own ``dir_fd`` rule:
    nothing there is followed), any other path as before."""
    from ouroboros.store import is_wal_index as rule

    spelling = work_spelling(path)
    if not spelling:
        return rule(path)
    try:
        with _confined_parent(spelling) as (directory, name):
            return rule(name, dir_fd=directory.handle)
    except OSError:
        return False


def exists(path: Any) -> bool:
    """Whether a path exists (a link counts as itself). A work path is asked through the confined descent, a link
    on its path being the work domain's to answer."""
    spelling = work_spelling(path)
    if not spelling:
        return pathlib.Path(path).exists()
    try:
        lstat(spelling)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EACCES):
            answer = run_in_work(["test", "-e", container_path(spelling)], timeout_sec=_READ_TIMEOUT_SEC)
            return _confirmed(answer, f"look at {path}").returncode == 0
        raise


def is_file(path: Any) -> bool:
    """``Path.is_file``; a work path by its own metadata (``kind``: a link there is not followed)."""
    return kind(path) == "file" if work_spelling(path) else pathlib.Path(path).is_file()


def is_dir(path: Any) -> bool:
    """``Path.is_dir``; a work path by its own metadata (``kind``: a link there is not followed)."""
    return kind(path) == "dir" if work_spelling(path) else pathlib.Path(path).is_dir()


def lexists(path: Any) -> bool:
    """``os.path.lexists``; a work path through the confined descent (``exists``: a link counts as itself)."""
    return exists(path) if work_spelling(path) else os.path.lexists(path)


def readlink(path: Any) -> str:
    """A link's own text (``os.readlink``), a work link read from its held parent."""
    spelling = work_spelling(path)
    if not spelling:
        return os.readlink(path)
    with _confined_parent(spelling) as (directory, name):
        return os.readlink(name, dir_fd=directory.handle)


def kind(path: Any) -> str:
    """``file``, ``dir``, ``link``, ``other`` or ``missing``, from a path's own metadata (never followed). A link
    on a work path's way (a folder made a link) makes it the work domain's own answer, resolved in its container."""
    try:
        mode = lstat(path).st_mode
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        spelling = work_spelling(path)
        if not spelling or exc.errno not in (errno.ELOOP, errno.ENOTDIR, errno.EACCES, errno.EPERM):
            raise
        script = ('if [ -L "$1" ]; then echo link; elif [ -d "$1" ]; then echo dir; elif [ -f "$1" ]; then echo file; '
                  'elif [ -e "$1" ]; then echo other; else echo missing; fi')
        result = _confirmed(run_in_work(["sh", "-c", script, "sh", container_path(spelling)],
                                        timeout_sec=_READ_TIMEOUT_SEC), f"look at {path}")
        answer = (result.stdout or b"").decode("utf-8", "replace").strip()
        if result.returncode != 0 or answer not in ("link", "dir", "file", "other", "missing"):
            raise OSError(f"the work domain did not report the kind of {path}") from exc
        return answer
    if stat.S_ISLNK(mode):
        return "link"
    return "dir" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode) else "other"


def entries(path: Any) -> List[Tuple[str, str]]:
    """A work folder's ``(name, kind)`` entries, sorted, from its held descriptor: links named, never followed."""
    from ouroboros.confined_files import confined_directory

    spelling = work_spelling(path)
    if not spelling:
        raise WorkDomainRefused(f"{path} is not a work-domain path")
    root, parts = _anchor(spelling)
    listed: List[Tuple[str, str]] = []
    with confined_directory(root, parts) as directory:
        for name in sorted(os.listdir(directory.handle)):
            mode = os.stat(name, dir_fd=directory.handle, follow_symlinks=False).st_mode
            listed.append((name, "link" if stat.S_ISLNK(mode) else "dir" if stat.S_ISDIR(mode)
                           else "file" if stat.S_ISREG(mode) else "other"))
    return listed


def _descend(path: Any, strict: bool, onerror: Any) -> Iterator[Tuple[str, List[Tuple[str, str]], List[str]]]:
    spelling = work_spelling(path)
    if not spelling:
        raise WorkDomainRefused(f"{path} is not a work-domain path")
    pending = [spelling]
    while pending:
        folder = pending.pop(0)
        try:
            listed = entries(folder)
        except OSError as exc:
            exc.filename = exc.filename or folder
            if onerror is not None:
                onerror(exc)
            if strict:
                raise
            continue
        dirs = [name for name, what in listed if what == "dir"]
        yield folder, listed, dirs
        pending[0:0] = [posixpath.join(folder, name) for name in dirs]  # what the caller left of ``dirs``


def walk(path: Any, *, strict: bool = False, onerror: Any = None) -> Iterator[Tuple[str, List[str], List[str]]]:
    """``os.walk`` over a work folder by held descriptors: ``(folder, dirs, files)`` with service spellings, a
    caller pruning ``dirs`` in place as with ``os.walk``; links and special files are neither descended nor listed.
    A folder that cannot be listed goes to ``onerror`` (as ``os.walk``'s, its ``filename`` the folder), then is
    skipped, or raises with ``strict``: never read as an empty folder."""
    for folder, listed, dirs in _descend(path, strict, onerror):
        yield folder, dirs, [name for name, what in listed if what == "file"]


def walk_entries(path: Any, *, strict: bool = False, onerror: Any = None) -> Iterator[Tuple[str, List[Tuple[str, str]]]]:
    """``(folder, [(name, kind)])`` for every folder under a work folder (``walk``'s descent): links and special
    files named with their kind, never descended; an unlistable folder goes to ``onerror``, raising with ``strict``."""
    for folder, listed, _dirs in _descend(path, strict, onerror):
        yield folder, listed
