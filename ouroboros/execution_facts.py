"""Placement-neutral execution facts — the authorize phase's ONLY fact source.

RWS v2 §3.1 step 2 (prepare) and step 3 (authorize): guards must judge FACTS
about the execution target, never ``pathlib.Path`` objects they probe themselves.
A guard that calls ``Path.exists()`` is silently asserting "the target is this
Home filesystem"; for an SSH placement that assertion is wrong and, worse,
succeeds — a remote spelling read as a Home path is exactly the postmortem class
"one policy × N doors". So every fact a guard needs about the target is named
here, once, behind a narrow accessor:

* path canonicalization (``canonical_path``),
* path existence / kind / size / symlink-ness (``path_fact``, ``path_facts``),
* git worktree identity (``git_fact``),
* interpreter provability (``interpreter_fact``).

Two implementations, selected by the sealed placement (``workspace_ref_for``):

* :class:`LocalExecutionFacts` — direct filesystem/subprocess probes. This is
  the CURRENT behavior byte-for-byte: ``expanduser().resolve(strict=False)``
  canonicalization, ``exists()``/``is_dir()``-equivalent stat semantics
  (symlinks followed; a broken symlink reads as missing, as ``exists()`` does),
  and the same ``git rev-parse --show-toplevel`` probe the admission SSOT runs.
* :class:`RemoteExecutionFacts` — reads the block the target filled during the
  operation's ONE bundled prepare (plan §3.1 step 2, codex #17). Every method is
  a lookup, never an RPC; a fact the bundle does not carry fails LOUDLY with
  ``SSH_FACTS_UNAVAILABLE`` rather than falling back to a Home probe, because a
  Home probe would answer about the wrong filesystem and succeed.

Interpreter provability delegates to ``python_interpreter.usable_executable``
(lazy import) rather than re-implementing the venv-symlink-preserving probe:
one implementation, consulted through the placement-neutral door.

This module deliberately imports NO Home policy authority (``tool_access``,
``protected_artifacts``, ``workspace_executor``): the guards in those modules
import facts, so facts must not import them back.
"""

from __future__ import annotations

import dataclasses
import pathlib
import stat as stat_module
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from ouroboros.platform_layer import bootstrap_process_path
from ouroboros.workspace_native_contract import (
    NATIVE_FACT_GIT_TOPLEVELS,
    NATIVE_FACT_INTERPRETERS,
    NATIVE_FACT_PATH_STATS,
)
from ouroboros.workspace_ref import SshWorkspaceRef

# Typed unavailability code for target facts on an ssh placement. Distinct from
# workspace_executor.SSH_EXECUTOR_UNAVAILABLE on purpose: this one fires in the
# PREPARE phase (before authorize), that one in EXECUTE, and a diagnosis must be
# able to tell which phase refused.
SSH_FACTS_UNAVAILABLE = "SSH_FACTS_UNAVAILABLE"


class RemoteFactsUnavailableError(RuntimeError):
    """Raised when target facts are requested for an ssh placement before the
    prepare transport exists (Lane 1).

    ``action`` comes from the one register (``remote_refusal_actions``), like every
    other remote refusal: a fact the prepare bundle does not carry is a gap between
    the two builds' contract sets, which Bootstrap resolves and a retry does not.
    """

    code = SSH_FACTS_UNAVAILABLE

    def __init__(self, detail: str = "") -> None:
        from ouroboros.remote_refusal_actions import REFUSAL_ACTIONS

        message = f"{SSH_FACTS_UNAVAILABLE}: {detail}" if detail else SSH_FACTS_UNAVAILABLE
        self.action = REFUSAL_ACTIONS.get(SSH_FACTS_UNAVAILABLE.lower(), "retry")
        super().__init__(message)


PathKind = Literal["missing", "dir", "file", "other"]


@dataclasses.dataclass(frozen=True)
class PathFact:
    """One target path, as the guards are allowed to see it.

    ``kind`` follows symlinks (so ``is_dir`` matches ``Path.is_dir()`` exactly);
    ``symlink`` reports whether the REQUESTED spelling was itself a link, which
    is what the protected-path guards need to reason about escapes.
    """

    requested: str
    canonical: str
    kind: PathKind = "missing"
    symlink: bool = False
    size: int = -1

    @property
    def exists(self) -> bool:
        return self.kind != "missing"

    @property
    def is_dir(self) -> bool:
        return self.kind == "dir"

    @property
    def is_file(self) -> bool:
        return self.kind == "file"


@dataclasses.dataclass(frozen=True)
class GitFact:
    """Git worktree identity of a target directory (empty toplevel = not a worktree)."""

    cwd: str
    toplevel: str = ""

    @property
    def is_worktree(self) -> bool:
        return bool(self.toplevel)

    @property
    def is_worktree_root(self) -> bool:
        return bool(self.toplevel) and self.toplevel == self.cwd


@dataclasses.dataclass(frozen=True)
class InterpreterFact:
    """Whether an interpreter token resolves to a provable executable on the target."""

    requested: str
    resolved: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.resolved)


@runtime_checkable
class ExecutionFacts(Protocol):
    """The narrow fact door. ``placement`` is the discriminator guards may log
    but must never branch policy on — policy runs over facts, not placement."""

    placement: str

    def canonical_path(self, path: Any) -> str: ...

    def path_fact(self, path: Any) -> PathFact: ...

    def path_facts(self, paths: Iterable[Any]) -> tuple[PathFact, ...]: ...

    def git_fact(self, cwd: Any) -> GitFact: ...

    def interpreter_fact(self, requested: str) -> InterpreterFact: ...


class LocalExecutionFacts:
    """Facts about a Home-local target: direct FS/subprocess probes.

    Canonicalization is ``expanduser().resolve(strict=False)``. ``expanduser``
    is identity for every absolute/relative spelling, so this matches the bare
    ``resolve(strict=False)`` callers byte-for-byte and additionally matches the
    callers that already expand ``~`` themselves.
    """

    placement = "local"

    def canonical_path(self, path: Any) -> str:
        return str(pathlib.Path(str(path)).expanduser().resolve(strict=False))

    def path_fact(self, path: Any) -> PathFact:
        requested = str(path)
        expanded = pathlib.Path(requested).expanduser()
        canonical = str(expanded.resolve(strict=False))
        try:
            symlink = expanded.is_symlink()
        except OSError:
            symlink = False
        try:
            info = expanded.stat()
        except (OSError, ValueError):
            return PathFact(requested=requested, canonical=canonical, symlink=symlink)
        if stat_module.S_ISDIR(info.st_mode):
            kind: PathKind = "dir"
        elif stat_module.S_ISREG(info.st_mode):
            kind = "file"
        else:
            kind = "other"
        return PathFact(
            requested=requested,
            canonical=canonical,
            kind=kind,
            symlink=symlink,
            size=int(info.st_size),
        )

    def path_facts(self, paths: Iterable[Any]) -> tuple[PathFact, ...]:
        return tuple(self.path_fact(path) for path in paths)

    def git_fact(self, cwd: Any) -> GitFact:
        """The admission SSOT's worktree probe, as a fact.

        Same command, timeout and failure semantics as
        ``workspace_admission.validate_workspace_root`` used inline: a non-zero
        exit, a missing git, or a timeout all read as "not a worktree" (empty
        toplevel) rather than as an error.
        """
        root = str(pathlib.Path(str(cwd)).expanduser().resolve(strict=False))
        bootstrap_process_path()
        try:
            probe = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return GitFact(cwd=root)
        text = (probe.stdout or "").strip() if probe.returncode == 0 else ""
        toplevel = str(pathlib.Path(text).resolve(strict=False)) if text else ""
        return GitFact(cwd=root, toplevel=toplevel)

    def interpreter_fact(self, requested: str) -> InterpreterFact:
        from ouroboros.python_interpreter import usable_executable

        return InterpreterFact(requested=str(requested), resolved=usable_executable(requested))


class RemoteExecutionFacts:
    """SSH placement facts, READ from the operation's one bundled prepare response.

    Constructed with the ``native_facts`` block execd filled during prepare
    (``workspace_native_contract.bundle_prepared_facts``). Every method is a
    dictionary lookup: there is no RPC here, because a fact that needed one would
    mean the prepare bundle was incomplete and the guard is about to reason about
    a target nobody asked.

    A miss is a TYPED refusal, never a Home probe. That is the whole point of the
    class: the guards cannot tell which placement they are judging, so the one
    thing this must never do is quietly answer a question about the wrong
    filesystem. Two conditions produce the refusal:

    * no bundle at all — the accessor was built before prepare ran (or the token
      expired and the operation must prepare again);
    * a bundle that does not name the requested spelling — the operation did not
      declare that path, so no authorization may rest on it.
    """

    placement = "ssh"

    def __init__(self, ref: SshWorkspaceRef, native_facts: Mapping[str, Any] | None = None) -> None:
        self.ref = ref
        self.native_facts: dict[str, Any] = dict(native_facts or {})
        self.prepared = native_facts is not None

    def _unavailable(self, fact: str, detail: str = "") -> RemoteFactsUnavailableError:
        reason = detail or (
            "the operation has no prepare bundle yet"
            if not self.prepared
            else "the prepare bundle does not declare it"
        )
        return RemoteFactsUnavailableError(
            f"{fact} for ssh workspace {self.ref.workspace_id} "
            f"(connection_id={self.ref.connection_id}): {reason}"
        )

    def _block(self, key: str) -> dict[str, Any]:
        block = self.native_facts.get(key)
        return dict(block) if isinstance(block, Mapping) else {}

    def _stat_row(self, path: Any) -> dict[str, Any]:
        spelling = str(path)
        row = self._block(NATIVE_FACT_PATH_STATS).get(spelling)
        if not isinstance(row, Mapping):
            raise self._unavailable(f"path stat of {spelling!r}")
        return dict(row)

    def canonical_path(self, path: Any) -> str:
        return str(self._stat_row(path).get("canonical") or "")

    def path_fact(self, path: Any) -> PathFact:
        row = self._stat_row(path)
        kind = str(row.get("kind") or "missing")
        return PathFact(
            requested=str(path),
            canonical=str(row.get("canonical") or ""),
            kind=kind if kind in {"missing", "dir", "file", "other"} else "other",
            symlink=bool(row.get("symlink")),
            size=int(row.get("size", -1)),
        )

    def path_facts(self, paths: Iterable[Any]) -> tuple[PathFact, ...]:
        return tuple(self.path_fact(path) for path in paths)

    def git_fact(self, cwd: Any) -> GitFact:
        spelling = str(cwd)
        toplevels = self._block(NATIVE_FACT_GIT_TOPLEVELS)
        if spelling not in toplevels:
            raise self._unavailable(f"git worktree probe of {spelling!r}")
        return GitFact(cwd=spelling, toplevel=str(toplevels[spelling] or ""))

    def interpreter_fact(self, requested: str) -> InterpreterFact:
        token = str(requested)
        interpreters = self._block(NATIVE_FACT_INTERPRETERS)
        if token not in interpreters:
            raise self._unavailable(f"interpreter probe of {token!r}")
        return InterpreterFact(requested=token, resolved=str(interpreters[token] or ""))


# Stateless: the local accessor holds no per-operation state, so one instance
# serves every caller. Operation-scoped MEMOIZATION of facts is the prepare
# phase's job (§3.1 step 2), not this accessor's.
LOCAL_FACTS = LocalExecutionFacts()


def facts_for_ref(ref: Any, native_facts: Mapping[str, Any] | None = None) -> ExecutionFacts:
    """Return the fact accessor for an ALREADY-READ sealed placement.

    The ONLY constructor, and it takes a ref rather than a context on purpose: the
    prepare phase is the dispatch's single placement-resolution site, so every
    accessor is derived from that one read and a second, disagreeing placement
    cannot come into existence here. (A `facts_for(ctx)` convenience existed and was
    removed once nothing production-side could reach it — a second door into the
    placement is exactly what this module is shaped to prevent.) ``native_facts`` is
    the ssh operation's bundled prepare block; omitting it yields an accessor that
    refuses every fact, which is the correct answer BEFORE prepare has run.
    """
    if isinstance(ref, SshWorkspaceRef):
        return RemoteExecutionFacts(ref, native_facts)
    return LOCAL_FACTS


__all__ = [
    "SSH_FACTS_UNAVAILABLE",
    "ExecutionFacts",
    "GitFact",
    "InterpreterFact",
    "LOCAL_FACTS",
    "LocalExecutionFacts",
    "PathFact",
    "PathKind",
    "RemoteExecutionFacts",
    "RemoteFactsUnavailableError",
    "facts_for_ref",
]
