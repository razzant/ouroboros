"""The body predicate: is a repository root Ouroboros's own body?

One structural fact decides which checklist layer a change review runs
(:func:`layer_for`): the universal core for any code Ouroboros works on, the
core plus the Ouroboros body layer when the subject IS the installed body —
the system repository, or a candidate whose upstream is this install's
configured update source or the install's own fork (owner clarification
2026-10-07). The fact is read from git and the install manifest only; no
directory, branch or remote NAME is matched by pattern (BIBLE P5). Where git
cannot answer, the fact is ``unknown`` and the review runs the core with the
fact recorded loudly; the mind may raise ``unknown`` to the body with
``treat_as_body=True`` (the raise is recorded), and nothing lowers a
recognized body.

Recognition order, and the ``how`` each step records:

1. ``dir`` — the root is the system repository or a path inside it.
2. ``git_common_dir`` — the root shares the system repository's common git
   dir: a worktree or nested copy of the body (``workspace_copies``).
3. ``remote_chain`` — a remote's fetch URL is a local path that reaches the
   body, followed recursively up to :data:`REMOTE_CHAIN_DEPTH` hops.
4. ``managed_remote`` — a fetch URL equals the install manifest's
   ``managed_remote_url``: the configured update source (official repository,
   fork or mirror) IS this install's body.
5. ``install_fork`` — a fetch URL equals the installed repository's ``origin``.
6. ``copy_origin`` — the task-owned copy registry binds the root to a source
   (``workspace_copies.admitted_copy_metadata``), consulted when git gave no
   answer.

A remote URL that reaches none of these makes the root ``false`` (a foreign
project). A git repository with no remote and no copy binding, a tree that is
not a git repository, and a git failure or timeout are ``unknown``. The
branch never takes part: it is a fact of the record, not of the body.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlsplit

from ouroboros.workspace_copies import admitted_copy_metadata, same_directory, source_is_system_repo

BODY_TRUE = "true"
BODY_FALSE = "false"
BODY_UNKNOWN = "unknown"
BODY_FACT_VALUES = (BODY_TRUE, BODY_FALSE, BODY_UNKNOWN)
HOW_VALUES = ("dir", "git_common_dir", "remote_chain", "managed_remote",
              "install_fork", "copy_origin", "unknown")
CORE_LAYER = "core"
BODY_LAYER = "body"
CHECKLIST_LAYERS = (CORE_LAYER, BODY_LAYER)

# How many local-path remotes may be followed in a row before the chain is
# declared exhausted (root → its remote → that remote's remote → ...).
REMOTE_CHAIN_DEPTH = 3
GIT_TIMEOUT_SEC = 10


@dataclass(frozen=True)
class BodyFact:
    """``body`` ∈ :data:`BODY_FACT_VALUES`; ``how`` ∈ :data:`HOW_VALUES`;
    ``detail`` is the human-readable evidence for the durable record."""

    body: str
    how: str
    detail: str = ""


def layer_for(fact: BodyFact) -> str:
    """The checklist layer this fact selects: the body layer only for a
    recognized (or explicitly raised) body, the universal core otherwise."""
    return BODY_LAYER if fact.body == BODY_TRUE else CORE_LAYER


class _GitUnavailable(Exception):
    """git could not answer at all: missing binary, timeout, unreadable root."""


@dataclass(frozen=True)
class _BodyIdentity:
    system: Path
    managed_url: str
    origin_url: str
    data_dir: Any


def _git(root: Path, *args: str) -> Optional[str]:
    """stdout of one read-only git call; ``None`` when git answered non-zero."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=GIT_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError) as exc:
        raise _GitUnavailable(f"git {args[0]} in {root}: {type(exc).__name__}: {exc}") from exc
    return None if proc.returncode else proc.stdout


def _is_local_path(url: str) -> bool:
    """A fetch URL that names a directory rather than a host."""
    text = str(url or "").strip().replace("\\", "/")
    if not text:
        return False
    if text.startswith("file://"):
        return True
    if "://" in text:
        return False
    head = text.split("/", 1)[0]
    drive = len(head) == 2 and head[1] == ":" and head[0].isalpha()
    return ":" not in head or drive  # ``user@host:path`` is scp-like, not local


def _local_path(url: str, base: Path) -> Path:
    text = str(url or "").strip()
    if text.startswith("file://"):
        text = text[len("file://"):]
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def normalize_remote_url(url: str) -> str:
    """One spelling for one remote: scheme and user info dropped, host
    lower-cased, ``.git`` and trailing slashes removed, so
    ``git@host:owner/repo.git``, ``ssh://git@host/owner/repo`` and
    ``https://host/owner/repo.git/`` are the same remote. A local path is
    returned as given (it is followed, never compared as a URL)."""
    text = str(url or "").strip()
    if not text:
        return ""
    if "://" in text:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
        if parts.port:
            host = f"{host}:{parts.port}"
        path = parts.path
    else:
        head, sep, rest = text.partition(":")
        if sep and "/" not in head and not (len(head) == 1 and head.isalpha()):
            host, path = head.rsplit("@", 1)[-1].lower(), "/" + rest
        else:
            host, path = "", text
    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4].rstrip("/")
    return f"{host}{path}" if host else path


def _remotes(root: Path) -> List[Tuple[str, str]]:
    """``(name, fetch URL)`` for every remote of ``root``."""
    out = _git(root, "remote", "-v") or ""
    remotes: List[Tuple[str, str]] = []
    for line in out.splitlines():
        if not line.endswith("(fetch)"):
            continue
        name, _tab, rest = line.partition("\t")
        url = rest[: -len("(fetch)")].strip()
        if name and url:
            remotes.append((name, url))
    return remotes


def _inside(root: Path, system: Path) -> bool:
    return same_directory(root, system) or any(same_directory(parent, system) for parent in root.parents)


def _copy_origin(root: Path, body: _BodyIdentity) -> Optional[BodyFact]:
    for candidate in dict.fromkeys((str(root), str(root.resolve()))):
        try:
            meta = admitted_copy_metadata(candidate, body.data_dir)
        except (ValueError, OSError, TypeError, KeyError):
            continue
        source = str(meta.get("source_root") or "")
        if meta.get("source_is_system_repo") is not False:
            return BodyFact(BODY_TRUE, "copy_origin",
                            f"task-owned copy of the body (registry binding; source {source or 'system repository'})")
        return BodyFact(BODY_FALSE, "copy_origin",
                        f"task-owned copy of a foreign source {source} (registry binding)")
    return None


def _git_fact(root: Path, body: _BodyIdentity, depth: int, seen: set) -> BodyFact:
    """Steps 1-5: the fact git alone establishes for ``root``."""
    if _inside(root, body.system):  # inside the body whether or not the path exists yet
        return BodyFact(BODY_TRUE, "dir", f"{root} is the system repository {body.system} or lies inside it")
    if not root.is_dir():
        return BodyFact(BODY_UNKNOWN, "unknown", f"{root} is not a directory")
    try:
        if _git(root, "rev-parse", "--git-common-dir") is None:
            return BodyFact(BODY_UNKNOWN, "unknown", f"{root} is not a git repository")
        if source_is_system_repo(root, body.system):
            return BodyFact(BODY_TRUE, "git_common_dir",
                            f"{root} shares the system repository's common git dir")
        remotes = _remotes(root)
    except _GitUnavailable as exc:
        return BodyFact(BODY_UNKNOWN, "unknown", str(exc))

    foreign: List[str] = []
    undecided: List[str] = []
    for name, url in remotes:
        if _is_local_path(url):
            target = _local_path(url, root)
            key = str(target)
            if depth + 1 > REMOTE_CHAIN_DEPTH:
                undecided.append(f"remote {name} → {target}: chain depth {REMOTE_CHAIN_DEPTH} exhausted")
                continue
            if key in seen:
                undecided.append(f"remote {name} → {target}: already followed")
                continue
            seen.add(key)
            sub = _recognize(target, body, depth + 1, seen)
            if sub.body == BODY_TRUE:
                return BodyFact(BODY_TRUE, "remote_chain", f"remote {name} → {target}: {sub.how} ({sub.detail})")
            (foreign if sub.body == BODY_FALSE else undecided).append(
                f"remote {name} → {target}: {sub.body} ({sub.detail})")
            continue
        normalized = normalize_remote_url(url)
        if normalized and normalized == body.managed_url:
            return BodyFact(BODY_TRUE, "managed_remote",
                            f"remote {name} {url} is the install's configured managed_remote_url")
        if normalized and normalized == body.origin_url:
            return BodyFact(BODY_TRUE, "install_fork",
                            f"remote {name} {url} is the installed repository's origin")
        foreign.append(f"remote {name} {url}: reaches neither the managed remote nor the install's origin")
    if foreign:
        return BodyFact(BODY_FALSE, "remote_chain", "; ".join(foreign + undecided))
    return BodyFact(BODY_UNKNOWN, "unknown",
                    "; ".join(undecided) if undecided else f"{root} has no remote")


def _recognize(root: Path, body: _BodyIdentity, depth: int, seen: set) -> BodyFact:
    fact = _git_fact(root, body, depth, seen)
    if fact.body != BODY_UNKNOWN:
        return fact
    copy = _copy_origin(root, body)
    if copy is not None:
        return copy
    return BodyFact(BODY_UNKNOWN, "unknown", f"{fact.detail}; no copy origin")


def body_fact(root: Any, *, system_repo: Any, manifest: Optional[dict] = None,
              data_dir: Any = None, treat_as_body: bool = False) -> BodyFact:
    """Is ``root`` Ouroboros's body? See the module docstring for the order.

    ``manifest`` is the install's managed-repo manifest
    (``launcher_bootstrap.load_repo_manifest``; read from ``system_repo`` when
    ``None``); an install without ``managed_remote_url`` is still the body by
    directory and by common git dir. ``data_dir`` addresses the copy registry.
    ``treat_as_body=True`` raises an ``unknown`` fact to ``true`` — ``how``
    keeps the original value and ``detail`` records the raise — and changes
    nothing for a recognized body or a recognized foreign root (the caller
    records that its argument was not needed or was ignored)."""
    system = Path(system_repo).expanduser().resolve()
    if manifest is None:
        from ouroboros.launcher_bootstrap import load_repo_manifest

        try:
            manifest = load_repo_manifest(system)
        except Exception:  # noqa: BLE001 — an unreadable manifest is "no configured source", not a crash
            manifest = {}
    try:
        origin = _git(system, "remote", "get-url", "origin") or ""
    except _GitUnavailable:
        origin = ""
    identity = _BodyIdentity(
        system=system,
        managed_url=normalize_remote_url(str((manifest or {}).get("managed_remote_url") or "")),
        origin_url=normalize_remote_url(origin.strip()),
        data_dir=data_dir,
    )
    start = Path(root).expanduser().resolve()
    fact = _recognize(start, identity, 0, {str(start)})
    if treat_as_body and fact.body == BODY_UNKNOWN:
        return BodyFact(BODY_TRUE, fact.how, f"{fact.detail}; raised to body by treat_as_body")
    return fact
