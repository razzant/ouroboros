"""How an applied patch result reaches a Git index, for both patch pipelines.

Each touched path stages by its index membership BEFORE the apply, and a failed
staging undoes the whole attempt: file results, the patch in the tree and exactly
the index entries and flags it changed. The callers own the lock, the apply and
every message; this module stages and undoes.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Any, Dict, List

from ouroboros import work_domain  # a work target's Git and files are the work domain's (its container)


def git_apply(target: pathlib.Path, patch_path: Any, flags: List[str], **options: Any) -> Any:
    """``git apply FLAGS <patch>`` in TARGET. A work target cannot see the service's patch file, so the open file
    streams on stdin there (``-``), never read whole; any other target reads the file by its path, as before."""
    if work_domain.is_work(target):
        with open(patch_path, "rb") as patch:
            return work_domain.run(["git", "apply", *flags, "-"], cwd=str(target), stdin=patch, capture_output=True,
                                   **{key: value for key, value in options.items() if key in ("text", "env")})
    return subprocess.run(["git", "apply", *flags, str(patch_path)], cwd=str(target), capture_output=True, **options)


def _git_paths(target: pathlib.Path, args: List[str], paths: List[str], prefix: bytes = b"") -> Any:
    """One Git command reading NUL-delimited paths on stdin: byte-safe, immune to argv limits."""
    return work_domain.run(["git", *args], cwd=str(target), capture_output=True, input=b"".join(
        prefix + path.encode("utf-8", errors="surrogateescape") + b"\0" for path in paths))


def _git_failure(proc: Any) -> str:
    """A failed Git command's own words; never empty, so a silent failure cannot read as success."""
    return ((proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
            or f"Git exited {proc.returncode} without a message")


# An intent-to-add entry lists as the empty blob (SHA-1 or SHA-256); only such entries need the second look.
_EMPTY_BLOBS = {"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
                "473a0f4c3be8a93681a267e3b1e9a7dcda1185436fe141f7749120a303721813"}


def index_entries(target: pathlib.Path, paths: List[str]) -> Dict[str, tuple]:
    """Each of PATHS the target's index holds: every stage as ``ls-files --stage`` prints it (and ``update-index
    --index-info`` takes back), then the flags Git keeps beside it, ``+assume-unchanged``, ``+skip-worktree``
    and ``+intent-to-add``. Raises when the index cannot be read: unknown is not untracked."""
    wanted, entries, flags = set(paths), {}, {}
    listed = work_domain.run(["git", "ls-files", "--stage", "-v", "-z"], cwd=str(target), capture_output=True, check=True)
    for line in listed.stdout.split(b"\0"):
        meta, _, raw = line.partition(b"\t")
        if (path := raw.decode("utf-8", errors="surrogateescape")) in wanted:
            tag, _, entry = meta.decode("ascii").partition(" ")
            entries.setdefault(path, []).append(entry)
            flags.setdefault(path, set()).update(
                {"+assume-unchanged"} if tag.islower() else set(), {"+skip-worktree"} if tag.upper() == "S" else set())
    # Porcelain status alone tells intent-to-add (index mode 000000) from a staged empty file; it writes no index.
    if candidates := [path for path, rows in entries.items() if len(rows) == 1 and rows[0].split()[1] in _EMPTY_BLOBS]:
        status = work_domain.run(["git", "--no-optional-locks", "--literal-pathspecs", "status", "--porcelain=v2", "-z",
                                 "--untracked-files=no", "--ignore-submodules=all", "--no-renames", "--", *candidates],
                                cwd=str(target), capture_output=True, check=True)
        for record in status.stdout.split(b"\0"):
            fields = record.split(b" ", 8)
            if len(fields) == 9 and fields[0] == b"1" and fields[4] == b"000000":
                flags.get(fields[8].decode("utf-8", errors="surrogateescape"), set()).add("+intent-to-add")
    return {path: (*entries[path], *sorted(flags[path])) for path in entries}


def _mark_flags(target: pathlib.Path, flagged: Dict[str, tuple], prefix: str = "") -> str:
    """Set (``prefix="no-"``: clear) the assume-unchanged and skip-worktree flags each record of FLAGGED holds,
    on entries the index holds now. A failed Git command's own words, else ""."""
    for option in ("assume-unchanged", "skip-worktree"):
        group = [path for path, record in flagged.items() if f"+{option}" in record]
        if group and (proc := _git_paths(target, ["update-index", "-z", f"--{prefix}{option}", "--stdin"], group)).returncode:
            return _git_failure(proc)
    return ""


def stage_applied(target: pathlib.Path, paths: List[str], indexed: Dict[str, tuple]) -> tuple[str, List[str]]:
    """Stage an applied result by each path's index membership BEFORE the apply (``indexed``).

    A path the index tracked stages its new state, deletion included, under any ignore rule:
    ``add -u`` never consults them, and tracking was the project's own choice. An assume-unchanged or
    skip-worktree entry would not stage that way (``add -u`` skips the first silently and refuses the
    second), so its flag is cleared, ``update-index`` stages it, and the recorded flag returns on the
    entry the index still holds, a three-way apply's rewritten one included. A new path stages
    ordinarily unless the target's ignore rules, as the apply left them, exclude it: it stays
    delivered in the working tree and out of the index (a three-way apply's own entry is removed),
    returned by name; forcing it in would rewrite the project's policy. A path neither present nor
    tracked (an untracked file the result deleted), or a deletion a three-way apply already staged,
    has nothing to stage. Pathspecs are literal. Returns ``(git_failure, delivered_unindexed)``."""
    tracked = [path for path in paths if path in indexed]
    if gone := [path for path in tracked if not os.path.lexists(target / path)]:
        try:
            held = index_entries(target, gone)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"the target's index could not be read: {exc}", []
        tracked = [path for path in tracked if path not in gone or path in held]
    flagged = {path: indexed[path] for path in tracked
               if {"+assume-unchanged", "+skip-worktree"} & set(indexed[path])}
    new = [path for path in paths if path not in indexed and os.path.lexists(target / path)]
    ignored: List[str] = []
    if new:  # check-ignore refuses literal pathspecs; a ./ spelling keeps a leading ':' a plain name
        probe = _git_paths(target, ["check-ignore", "--no-index", "-z", "--stdin"], new, b"./")
        if probe.returncode not in (0, 1):
            return _git_failure(probe), []
        ignored = sorted(item.decode("utf-8", errors="surrogateescape")[2:]
                         for item in probe.stdout.split(b"\0") if item)
    if failure := _mark_flags(target, flagged, "no-"):
        return failure, []
    stdin_specs = ["--pathspec-from-file=-", "--pathspec-file-nul"]
    for args, group in ((["update-index", "-z", "--add", "--remove", "--replace", "--stdin"], list(flagged)),
                        (["add", "-u", *stdin_specs], [path for path in tracked if path not in flagged]),
                        (["add", *stdin_specs], sorted(set(new) - set(ignored))),
                        (["update-index", "-z", "--force-remove", "--stdin"], ignored)):
        if group and (proc := _git_paths(target, ["--literal-pathspecs", *args], group)).returncode:
            return _git_failure(proc), []
    try:  # a flagged path the result deleted has no entry left to mark
        held = index_entries(target, list(flagged)) if flagged else {}
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the target's index could not be read: {exc}", []
    failure = _mark_flags(target, {path: record for path, record in flagged.items() if path in held})
    return failure, [] if failure else ignored


def restore_index(target: pathlib.Path, paths: List[str], indexed: Dict[str, tuple]) -> str:
    """Return every path of PATHS whose index record differs from INDEXED to it, then prove it: all stages in
    one ``--index-info`` write (absent before means removed), the written entries' stat cache refreshed for
    exactly those paths, then the flags Git kept beside them. An intent-to-add entry has no content to write:
    ``add -N`` recreates exactly that intent, ``-f`` only because ignore rules may cover the name. No other
    entry, staged or not, is written. "" once the index holds INDEXED again, else what is still wrong."""
    try:
        now = index_entries(target, paths)
        if not (changed := [path for path in paths if now.get(path) != indexed.get(path)]):
            return ""
        before = {path: indexed.get(path, ()) for path in changed}
        null = "0" * len((now.get(changed[0]) or indexed[changed[0]])[0].split()[1])
        lines = [f"0 {null}\t{path}" for path in changed]
        lines += [f"{entry}\t{path}" for path, record in before.items() if "+intent-to-add" not in record
                  for entry in record if not entry.startswith("+")]
        restore = work_domain.run(["git", "update-index", "-z", "--index-info"], cwd=str(target), capture_output=True,
                                 input="".join(f"{line}\0" for line in lines).encode("utf-8", errors="surrogateescape"))
        if restore.returncode:
            return _git_failure(restore)
        specs = ["--literal-pathspecs", "add", "--pathspec-from-file=-", "--pathspec-file-nul"]
        for args, group in (([*specs, "--refresh"], [path for path, record in before.items() if record
                                                     and not {"+intent-to-add", "+skip-worktree"} & set(record)]),
                            ([*specs, "-N", "-f"], [path for path, record in before.items() if "+intent-to-add" in record])):
            if group and (proc := _git_paths(target, args, group)).returncode:
                return _git_failure(proc)
        if failure := _mark_flags(target, before):
            return failure
        now = index_entries(target, changed)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return "" if all(now.get(path) == indexed.get(path) for path in changed) else "the index still differs"


def sparse_absent(target: pathlib.Path, paths: List[str], indexed: Dict[str, tuple]) -> List[str]:
    """Skip-worktree paths absent from the tree before the apply: a three-way apply materializes them."""
    return [path for path in paths if "+skip-worktree" in indexed.get(path, ()) and not (
        work_domain.exists(target / path) if work_domain.is_work(target) else os.path.lexists(target / path))]


def undo_applied(target: pathlib.Path, patch_path: Any, prepared: Any, paths: List[str], indexed: Dict[str, tuple],
                 sparse: List[str]) -> tuple[bool, str]:
    """Undo an applied result whose staging failed: file results, then the patch (PATCH_PATH, None when there
    is none) reversed in the tree after a check, the SPARSE files it materialized removed once that holds,
    then exactly the index entries and flags it changed. ``(reverted, problems)``: reverted only when all are
    back; PROBLEMS names what is not, for an honest partial."""
    problems: List[str] = []
    try:
        if prepared:
            prepared.rollback()
    except Exception as exc:
        problems.append(f"file rollback: {type(exc).__name__}: {exc}")
    tree_reverted = patch_path is None or all(
        git_apply(target, patch_path, [*check, "--reverse"]).returncode == 0 for check in (["--check"], []))
    if not tree_reverted:
        problems.append("the patch could not be reversed in the working tree")
    for path in sparse if tree_reverted else ():
        try:  # the attempt's own file: the sparse checkout had none
            work_domain.remove(target / path) if work_domain.is_work(target) else (target / path).unlink(missing_ok=True)
        except OSError as exc:
            problems.append(f"sparse file {path}: {exc}")
    if problem := restore_index(target, paths, indexed):
        problems.append(f"index restore: {problem}")
    return not problems, "; ".join(problems)


def staging_receipt(count: int, unindexed: List[str]) -> tuple[str, str, str]:
    """A receipt's staging words, delivery and indexing apart: the parenthetical count, a line naming the
    delivered files the target's ignore rules keep out of the index (the verdict lists every one), and how
    the closing sentence names what is staged."""
    if not unindexed:
        return f"{count} file(s), staged", "", "Changes are staged but NOT committed"
    shown = ", ".join(unindexed[:20]) + (f" (+{len(unindexed) - 20} more)" if len(unindexed) > 20 else "")
    return (f"{count} file(s), {count - len(unindexed)} staged",
            f"\nDelivered, not indexed by the project's ignore rules: {len(unindexed)} new file(s) stay "
            f"unstaged in the working tree, out of any commit unless added deliberately: {shown}.",
            "The other changes are staged but NOT committed; the files named above are not staged at all")
