"""The target's path kernel: ONE door per intent, and each door asks ONE question.

Extracted from ``workspace_native`` so the confinement is a module a reader can hold
in their head — and so the rule cannot be re-derived per call site. Two doors only:

* :func:`native_target` — every READ-shaped resolution. Resolves the whole spelling,
  including the final component, and refuses anything that lands outside the root.
* :func:`native_mutation_target` — every WRITE-shaped resolution: open, append,
  atomic replace, restore, delete. Same refusal, plus the create case (a final
  component that does not exist yet is legal and stays unresolved).

The doors also APPLY THE EXPORT POLICY, and the title's old "no policy" is gone with
the reason: confinement and policy were decided in different places, so a spelling was
judged by the policy and then a DIFFERENT file was resolved and handed over. A symlink
`safe.txt -> .env` returned the secret that `.env` itself was refused, and on the write
side there was no policy check at all — a symlink alias overwrote a protected artifact.
Resolution is where both questions can be answered about the same file, so both are
answered here, and ``question`` is a REQUIRED keyword on both doors: a new door cannot
resolve a workspace path without stating which policy question its caller is asking,
which is the property that stops the third door re-opening this hole. In-root symlinks
stay traversable — that is correct and matches the local route — but the TARGET is what
gets judged.

Why two rather than one: a read must exist to be read, while a write is allowed to
name a path that does not exist yet, and the two need different confinement arithmetic
(a write confines the PARENT plus the final link; a read confines the whole spelling).
Neither door resolves strictly any more — existence is checked AFTER the policy, or the
refusal doubles as an existence oracle for every excluded name.
Why not more than two: the append hole (`workspace_native._write_file`, mode
``append``, opened ``"a"`` on a path whose final component was a symlink out of the
workspace) existed because the confinement decision was made at the open site, and one
open site out of two had made it differently. A rule spelled once cannot disagree
with itself.

The answers here MATCH the local route by construction, which is the property
``tests/test_route_refusal_parity.py`` pins: ``tool_access.resolve_resource_path``
returns ``(base / rel).resolve()`` and refuses on ``relative_to`` — so an out-of-root
link in the final component is refused on BOTH routes, and an in-root link is
followed (not replaced) on BOTH.
"""

from __future__ import annotations

import errno
import os
import pathlib
import tempfile
from typing import Any

from ouroboros.export_policy_contract import (
    QUESTION_MUTATION,
    QUESTION_NONE,
    refuse_excluded_target,
)
from ouroboros.workspace_native_contract import native_relative_spelling

__all__ = [
    "atomic_write",
    "native_cwd",
    "open_confined_source",
    "native_mutation_target",
    "native_target",
    "path_is_relative_to",
    "workspace_root_dir",
]

# The path canonicalization is the CONTRACT's (`native_relative_spelling`), not this
# kernel's: Home reproduces it to distinguish a tidied spelling from a substituted
# file, and two copies of the rule would be two answers waiting to disagree.
_relative_text = native_relative_spelling


def path_is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    # `RuntimeError` is in the tuple because `pathlib` raises it — not an `OSError` —
    # for a symlink LOOP, at every resolution depth and even with `strict=False`. An
    # unanswerable containment question is answered NO.
    try:
        pathlib.Path(path).resolve(strict=False).relative_to(
            pathlib.Path(root).resolve(strict=False)
        )
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def workspace_root_dir(value: pathlib.Path | str) -> pathlib.Path:
    root = pathlib.Path(value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    return root


# The ONE refusal both doors raise, so a caller cannot tell a refused read from a
# refused write by its text. `EACCES` on purpose: `workspace_diagnostics` maps it to the
# typed `permission_denied` diagnostic with `completion="not_started"`, which is what a
# refused mutation has to report — nothing ran. Spelled as a template rather than a
# helper function because the function budget is at its ceiling and a raise site that
# reads `raise PermissionError(errno.EACCES, _ESCAPE.format(...))` hides nothing.
_ESCAPE = "path escapes workspace through symlink: {value}"
# The same refusal for a path that cannot be RESOLVED at all. `pathlib` raises a bare
# `RuntimeError` on a symlink loop, which is neither an `OSError` nor typed: it escaped
# both doors uncaught, so `workspace_diagnostics` never saw it and a looped read reported
# an untyped crash instead of `permission_denied` with `completion="not_started"`.
_UNRESOLVABLE = "path cannot be resolved (symlink loop or too many levels): {value}"


def native_target(
    root: pathlib.Path,
    value: Any,
    *,
    question: str,
    facts: Any,
    must_exist: bool = False,
    channel: str = "",
) -> pathlib.Path:
    """Resolve a workspace-relative READ path: confined, then judged by the policy.

    ``question`` and ``facts`` are required, and that is the structural half of the
    fix. Confinement answers "is this under the root"; the policy answers "may these
    bytes leave the host", and the second question can only be asked honestly of the
    RESOLVED file. A caller that has no policy question to ask says so with
    ``question=QUESTION_NONE`` — spelled, greppable, and a decision rather than an
    omission.

    ``must_exist`` is enforced AFTER the policy, and the order is the rule rather than an
    implementation detail. Resolving with ``strict=True`` first made the refusal an
    EXISTENCE ORACLE: a present `.env` answered `ExportPolicyExcludedError` and an absent
    one answered `FileNotFoundError`, so the pair of answers told a caller which excluded
    files the workspace holds — a fact the refusal exists to withhold. A path the policy
    excludes now gets the same refusal whether or not it is there.
    """
    rel = _relative_text(value)
    candidate = root if rel == "." else root.joinpath(*rel.split("/"))
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise PermissionError(errno.EACCES, _UNRESOLVABLE.format(value=value)) from None
    if not path_is_relative_to(resolved, root):
        raise PermissionError(errno.EACCES, _ESCAPE.format(value=value))
    if question != QUESTION_NONE:
        refuse_excluded_target(
            root, resolved, rel, facts, question=question, channel=channel
        )
    if must_exist and not resolved.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(resolved))
    return resolved


def open_confined_source(
    root: pathlib.Path,
    value: Any,
    *,
    facts: Any,
    question: str,
    channel: str = "",
) -> tuple[pathlib.Path, int]:
    """Return ``(resolved path, open fd)`` where the fd IS the object the policy judged.

    The third door, and it exists because the other two cannot make this promise. They
    return a PATH, and the caller opens it afterwards, so the authorization is bound to a
    NAME and the read is bound to whatever that name means a moment later. A reviewer
    demonstrated the window with a deterministic stand-in: swap `frame.png` for a symlink
    to `.env` inside the applier call, and the media door exported
    ``b'SECRET_TOKEN=hunter2\\n'`` under ``mime: image/png``. It does not need to be won by
    timing to be a hole — a task can hold a concurrent writer in its own workspace
    (`start_service`, a background `run_command`), and "checked by name, used by name" is
    the shape, not the race.

    Three properties make the fd the judged object:

    * ``O_NOFOLLOW`` on the RESOLVED path. A realpath's final component is never a symlink,
      so this rejects exactly the case where something replaced it with one — with
      ``ELOOP``, which maps to the same typed refusal as any other confinement answer;
    * the policy is applied to ``os.fstat(fd)``, not to a fresh ``stat`` of the path, so
      the inode judged is the inode opened. A swap to a hardlink of `.env` is caught by the
      alias index against the DESCRIPTOR's identity;
    * the caller reads from the fd and never re-opens the path, so nothing that happens to
      the name afterwards can change the bytes.

    RESIDUAL, stated rather than implied: :func:`native_target` and
    :func:`native_mutation_target` still hand back a path, so every caller that opens one
    later keeps the window. This door is used by the two byte-returning single-source
    channels (`read_file` and the media/file bridge); converting the write side means
    ``openat``-style plumbing through every mutation site and is not done here.
    """

    # The ordinary door FIRST, with the policy and WITHOUT `must_exist`: an excluded name
    # has to be refused before existence is consulted, or the refusal is an existence
    # oracle again. The `open` below is what discovers a missing file, and it reports
    # `FileNotFoundError` exactly as the door used to.
    resolved = native_target(
        root, value, question=question, facts=facts, must_exist=False, channel=channel
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        handle = os.open(resolved, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise PermissionError(errno.EACCES, _ESCAPE.format(value=value)) from None
        raise
    try:
        refuse_excluded_target(
            root,
            resolved,
            _relative_text(value),
            facts,
            question=question,
            channel=channel,
            stat_result=os.fstat(handle),
        )
    except BaseException:
        os.close(handle)
        raise
    return resolved, handle


def native_mutation_target(
    root: pathlib.Path, value: Any, *, facts: Any
) -> pathlib.Path:
    """The ONE door every native mutation walks through: parent AND final link confined.

    Two rules, and the second one used to be missing. The PARENT is confined by
    resolution, as everywhere else. The FINAL component is then resolved only if it is
    itself a symlink — and a link that leaves the root is refused exactly as
    :func:`native_target` refuses it for a read.

    Leaving the final component unresolved was reasoned about only for
    :func:`atomic_write`, where ``os.replace`` substitutes the LINK and never follows
    it. That reasoning did not survive contact with a second write mode: ``append``
    opened the same path with ``"a"``, which DOES follow the link, so one write mode
    obeyed the confinement and its sibling wrote outside the workspace. The parent-only
    confinement was therefore correct for the caller it was written against and wrong
    as a rule — which is why the rule now lives here and not at the open sites.

    An in-root link is FOLLOWED, not replaced, because that is what the local route
    does (``tool_access.resolve_resource_path`` returns the resolved path and writes
    there). The two routes now agree in both directions.

    ``facts`` is required, because following that in-root link is exactly how a
    protected artifact was overwritten: the write side had NO policy applier at all,
    so `innocent.bin -> golden.bin` wrote `TAMPERED` into a path Home had declared
    protected, and the same alias let `edit_text` rewrite a `.env`. The link is still
    followed; the file at the far end is now judged, under ``QUESTION_MUTATION`` (the
    protected-artifact class only — a task writing a `.env` into its own workspace is
    ordinary on a local placement). ``facts=None`` states an UNBOUND operation, which
    carries no protected-path list to enforce.
    """
    rel = _relative_text(value)
    if rel == ".":
        return root
    candidate = root.joinpath(*rel.split("/"))
    try:
        parent = candidate.parent.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise PermissionError(errno.EACCES, _UNRESOLVABLE.format(value=value)) from None
    if not path_is_relative_to(parent, root):
        raise PermissionError(errno.EACCES, _ESCAPE.format(value=value))
    target = parent / candidate.name
    try:
        is_link = target.is_symlink()
    except OSError:
        # The confinement question could not be ANSWERED, so it is answered NO. A
        # mutation that proceeds on an unanswered confinement question is the defect
        # above with a different first step.
        raise PermissionError(errno.EACCES, _ESCAPE.format(value=value)) from None
    if is_link:
        if not path_is_relative_to(target, root):
            raise PermissionError(errno.EACCES, _ESCAPE.format(value=value))
        try:
            target = target.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            raise PermissionError(
                errno.EACCES, _UNRESOLVABLE.format(value=value)
            ) from None
    refuse_excluded_target(root, target, rel, facts, question=QUESTION_MUTATION)
    return target


def native_cwd(root: pathlib.Path, value: Any) -> pathlib.Path:
    text = str(value or "").strip().replace("\\", "/")
    if text.startswith("/"):
        normalized_root = root.as_posix().rstrip("/")
        if text == normalized_root:
            rel = "."
        elif text.startswith(normalized_root + "/"):
            rel = text[len(normalized_root) + 1 :]
        else:
            raise ValueError("cwd is outside the remote workspace")
    else:
        rel = text or "."
    # A command CWD exports nothing: no bytes leave the host because a process ran in
    # a directory, and judging it as a source would refuse `cd .git` on the target
    # while the local route allows it.
    target = native_target(root, rel, question=QUESTION_NONE, facts=None, must_exist=True)
    if not target.is_dir():
        raise NotADirectoryError(str(target))
    return target


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing_mode = None if path.is_symlink() else os.stat(path).st_mode & 0o7777
    except OSError:
        existing_mode = None
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, existing_mode if existing_mode is not None else 0o644)
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
