"""Dependency-light query_code kernel shared by local and remote workspaces."""

from __future__ import annotations

import ast
import errno
import fnmatch
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import time
from collections.abc import Mapping
from typing import Any, Callable

from ouroboros.code_intelligence import (
    _TS_LANGUAGES,
    _language,
    build_code_inventory,
    impact_files,
    relevant_files,
    render_codebase_digest,
    symbol_callees,
    symbol_callers,
    symbol_definitions,
    symbol_references,
)
from ouroboros.workspace_diagnostics import (
    ExecutionDiagnostic,
    ProcessExecutionResult,
    ToolExecutionEnvelope,
)
from ouroboros.export_policy_contract import (
    MAX_DISCLOSED_EXCLUSIONS,
    QUESTION_EXPORT,
    REASON_EXCLUDED_DIRECTORY,
    REASON_PROTECTED_ARTIFACT,
    AliasIndex,
    export_disclosure_block,
    judged_exclusion,
    path_under_any,
    policy_filtered_note,
    export_policy_hash,
    policy_from_facts,
)
from ouroboros.workspace_native_contract import NativeOperationResult
from ouroboros.workspace_snapshot_native import (
    snapshot_integrity_ready,
    snapshot_policy,
    snapshot_workspace,
)

QUERY_OPERATION_ORDER = (
    "relevant_files",
    "symbols",
    "definition",
    "references",
    "callers",
    "callees",
    "impact",
    "structural",
    "digest",
)
QUERY_OPERATIONS = frozenset(QUERY_OPERATION_ORDER)
_MAX_LIMIT = 200
# How many symbol names one relevant_files row samples before it says how many remain.
_RELEVANT_FILE_SYMBOLS = 5
_STRUCTURAL_MAX_FILES = 20_000
_SEARCH_MAX_FILES = 20_000
_SEARCH_MAX_MATCHES = 200
_SEARCH_MAX_FILE_BYTES = 1024 * 1024
_SEARCH_EXCLUDED_DIRS = frozenset({
    ".git", ".ouroboros", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".venv", "venv", "env", "node_modules", "dist", "build",
    ".tox", ".eggs", "python-standalone", "assets",
})


def classify_workspace_path(
    root: pathlib.Path,
    args: Mapping[str, Any],
) -> ToolExecutionEnvelope:
    """Classify one absolute path against the target's canonical workspace."""

    raw = str(args.get("path") or "")
    if not raw.startswith("/"):
        raise ValueError("ambiguous workspace classifier requires an absolute path")
    try:
        resolved = pathlib.Path(raw).resolve(strict=False)
        resolved.relative_to(root)
        inside = True
    except (OSError, ValueError):
        resolved = pathlib.Path(raw)
        inside = False
    relative = resolved.relative_to(root).as_posix() if inside else ""
    payload = {
        "classification": "active_workspace" if inside else "outside_workspace",
        "inside_workspace": inside,
        "resolved_path": resolved.as_posix(),
        "relative_path": relative,
    }
    return ToolExecutionEnvelope(
        text=json.dumps(payload, sort_keys=True),
        trace={"completion": "complete", **payload},
    )
_SEARCH_SKIP_GLOBS = frozenset({
    "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll", "*.exe", "*.bin", "*.o",
    "*.a", "*.tar", "*.gz", "*.zip", "*.png", "*.jpg", "*.jpeg", "*.gif",
    "*.ico", "*.webp", "*.woff", "*.woff2", "*.ttf", "*.eot", "*.min.js",
    "*.min.css", "*.map", "*.db", "*.sqlite", "*.sqlite3", "*.lock",
})
_PATCH_MAX_BYTES = 64 * 1024 * 1024


def _structural_wall_budget() -> float:
    try:
        return max(
            5.0,
            float(os.environ.get("OUROBOROS_SEARCH_CODE_WALL_SEC", "45") or 45),
        )
    except Exception:
        return 45.0


def walk_candidate_files(
    scope: pathlib.Path,
    repo_root: pathlib.Path,
) -> tuple[list[pathlib.Path], str]:
    """Return a bounded, symlink-confined structural-query file list."""

    if scope.is_file():
        return [scope], ""
    root_resolved = repo_root.resolve(strict=False)
    deadline = time.monotonic() + _structural_wall_budget()
    files: list[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(scope, followlinks=False):
        if time.monotonic() > deadline:
            return files, (
                f"walk stopped after {_structural_wall_budget():.0f}s wall budget "
                "(narrow path=)"
            )
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if not (pathlib.Path(dirpath) / name).is_symlink()
        ]
        for name in sorted(filenames):
            candidate = pathlib.Path(dirpath) / name
            try:
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue
            files.append(candidate)
            if len(files) >= _STRUCTURAL_MAX_FILES:
                return files, (
                    f"walk stopped at {_STRUCTURAL_MAX_FILES} files (narrow path=)"
                )
    return files, ""


class _PolicyExclusions:
    """The exclusions ONE query walk made, so the walk can disclose them (D7).

    A query channel that filters silently makes the model reason from a false
    premise: `search_code("SECRET_TOKEN")` returned "No matches found … (2 files
    searched)" on a remote workspace whose `.env` held that very string, while the
    same query on a local workspace returned the line. The model concluded "the
    key is not here" — a cross-placement divergence (§9) manufactured by the
    filter itself, not a difference in the trees.

    ``REASON_EXCLUDED_DIRECTORY`` is deliberately NOT collected. Those are the
    infrastructure directories (`.git`, `__pycache__`, `.pytest_cache`, …) that
    BOTH placements skip — local `search_code` prunes them via `SKIP_DIRS` with no
    disclosure either — so naming them would be noise that buries the class where
    the placements genuinely differ, and would flip every ordinary query to
    `partial`. What is collected is the sensitive/protected class: the paths whose
    omission changes what a no-match answer means.
    """

    __slots__ = ("_rows", "_seen", "_root", "_aliases", "_admitted")

    def __init__(self, root: pathlib.Path, document: Mapping[str, Any]) -> None:
        self._rows: list[dict[str, str]] = []
        self._seen: set[str] = set()
        # ONE alias index for the whole walk, built on the first entry that actually has a
        # second name. A walk can only judge spellings, and a hardlink is a second
        # spelling for one inode — `search_code` used to return `SECRET_TOKEN=…` as a
        # matched line from a `notes.txt` sharing `.env`'s inode while disclosing `.env` as
        # excluded two lines below.
        #
        # The root and document are REQUIRED, not optional. They were defaulted to `None`,
        # and a collector constructed without them silently judged spellings only — a
        # second, weaker mechanic reachable by forgetting an argument.
        #
        # It is also no longer seeded UP FRONT, and that is the cost fix: seeding walked
        # the whole tree before anything was read, so `search_code(path="scope")` over one
        # file paid a 24 000-file traversal (44 ms measured) for a question about one path.
        # The up-front ordering existed to stop an alias sorting earlier than the file it
        # aliases from being admitted first; a per-entry question has no order to get wrong.
        self._root = root
        self._aliases = AliasIndex(root, document)
        # The ADMITTED paths, which this collector never kept. `disclosure()` therefore
        # emitted an empty `exported[]`, so Home's returned-manifest leak check re-evaluated
        # the policy over nothing for `search_code` and `query_code` — the same vacuous
        # backstop that was fixed for the read and declared-output channels, still open on
        # the two walk channels. A collector that records only what it DROPPED cannot say
        # what it handed over.
        self._admitted: list[str] = []

    def record(self, relative: str, reason: str, judged: str = "") -> None:
        if not relative or not reason or reason == REASON_EXCLUDED_DIRECTORY:
            return
        if relative in self._seen:
            return
        self._seen.add(relative)
        # `judged` is the spelling the policy excluded, which differs from `path` exactly
        # when an ALIAS was the finding — and Home, holding no workspace, cannot re-derive
        # it. Without it the row is a claim nothing can check.
        self._rows.append(
            {"path": relative, "reason": reason, "judged": judged or relative}
        )

    def judge(self, relative: str, document: Mapping[str, Any]) -> str:
        """Ask THE judge — spelling, resolved identity and alias, in one call.

        The same function every single-source door asks, which is the property that was
        missing: this collector used to hold its own two-step version of the identity
        question, and the doors held a different, weaker one.
        """

        reason, _sentence, judged = judged_exclusion(
            self._root,
            self._root.joinpath(*relative.split("/")),
            relative,
            document,
            question=QUESTION_EXPORT,
            aliases=self._aliases,
        )
        if reason:
            self.record(relative, reason, judged)
        elif relative not in self._seen:
            self._seen.add(relative)
            self._admitted.append(relative)
        return reason

    @property
    def rows(self) -> list[dict[str, str]]:
        return sorted(self._rows, key=lambda row: row["path"])

    def __bool__(self) -> bool:
        return bool(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def note(self, marker: str) -> str:
        """The owner-facing sentence. Exact count, bounded list, never a bare skip."""

        rows = self.rows
        shown = rows[:MAX_DISCLOSED_EXCLUSIONS]
        text = (
            f"⚠️ {marker}: {len(rows)} path{'s' if len(rows) != 1 else ''} "
            "excluded by the export policy and NOT read — a no-result answer is "
            "NOT authoritative for them:\n"
            + "\n".join(f"- {row['path']}: {row['reason']}" for row in shown)
        )
        if len(rows) > len(shown):
            text += (
                f"\n… and {len(rows) - len(shown)} more "
                f"(list bounded at {MAX_DISCLOSED_EXCLUSIONS}; the count is exact)"
            )
        return text

    def disclosure(self, document: Mapping[str, Any] | None) -> dict[str, Any]:
        """The same wire block declared outputs emit, over the same evaluator.

        With the ADMITTED paths, which it did not carry: Home's leak check derives its
        field list from `MANIFEST_EXPORTED_PATH_FIELDS`, found `exported` here and got an
        empty list, so every `search_code` and `query_code` passed the backstop on hash and
        arithmetic alone. This is that channel's half of the same fix.
        """

        return export_disclosure_block(
            {"export_policy": document} if document is not None else None,
            self.rows,
            sorted(self._admitted),
        )


def _search_skippable(path: pathlib.Path) -> bool:
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in _SEARCH_SKIP_GLOBS):
        return True
    stat_result = path.lstat()
    return (
        stat.S_ISLNK(stat_result.st_mode)
        or not stat.S_ISREG(stat_result.st_mode)
        or stat_result.st_size > _SEARCH_MAX_FILE_BYTES
    )


def search_workspace(
    workspace_root: pathlib.Path | str,
    args: Mapping[str, Any],
    *,
    path_allowed: Callable[[pathlib.Path], bool] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    """Run the case-sensitive public search_code contract on a workspace."""

    document = snapshot_policy(policy)
    root = pathlib.Path(workspace_root).resolve(strict=True)
    filtered = _PolicyExclusions(root, document)
    query = str(args.get("query") or "")
    if not query:
        return ToolExecutionEnvelope(
            text="⚠️ SEARCH_ERROR: query is required.",
            trace={"completion": "complete"},
        )
    raw_scope = str(args.get("path") or "").strip().replace("\\", "/")
    pure_scope = pathlib.PurePosixPath(raw_scope or ".")
    lexical_scope = (
        pure_scope.as_posix().removeprefix("./")
        if not pure_scope.is_absolute() and ".." not in pure_scope.parts
        else ""
    )
    scope_excluded = bool(
        lexical_scope not in {"", "."}
        and filtered.judge(lexical_scope, document)
    )
    try:
        if scope_excluded:
            rel = lexical_scope
            scope: pathlib.Path | None = None
        else:
            rel = _relative_scope(root, args.get("path"))
            scope = (root / (rel or ".")).resolve(strict=True)
            scope.relative_to(root)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise PermissionError(
            errno.EACCES,
            f"search path escapes the workspace: {exc}",
        ) from exc
    regex_mode = bool(args.get("regex", False))
    try:
        pattern = re.compile(query if regex_mode else re.escape(query))
    except re.error as exc:
        return ToolExecutionEnvelope(
            text=f"⚠️ SEARCH_ERROR: invalid regex: {exc}",
            trace={"completion": "complete"},
        )
    max_results = max(
        1,
        min(_SEARCH_MAX_MATCHES, int(args.get("max_results") or 200)),
    )
    include = str(args.get("include") or "")
    matches: list[str] = []
    unreadable: list[str] = []
    scanned = 0
    truncated = False
    scan_limit_hit = False
    metadata_checked: set[pathlib.Path] = set()
    paths: list[pathlib.Path] = []
    if scope is not None:
        try:
            if not _search_skippable(scope):
                paths = [scope]
            metadata_checked.add(scope)
        except OSError as exc:
            if len(unreadable) < 20:
                unreadable.append(
                    f"{rel or '.'}: {type(exc).__name__}: {exc}"
                )
            scope = None
    if scope is not None and not paths:
        for dirpath, dirnames, filenames in os.walk(
            scope,
            followlinks=False,
            onerror=lambda exc: unreadable.append(
                f"{getattr(exc, 'filename', scope)}: "
                f"{type(exc).__name__}: {exc}"
            )
            if len(unreadable) < 20
            else None,
        ):
            directory = pathlib.Path(dirpath)
            try:
                directory_rel = directory.relative_to(root)
            except ValueError:
                directory_rel = pathlib.Path(".")
            kept_dirs: list[str] = []
            for name in sorted(dirnames):
                child_rel = (directory_rel / name).as_posix()
                if child_rel.startswith("./"):
                    child_rel = child_rel[2:]
                if name in _SEARCH_EXCLUDED_DIRS:
                    continue
                if filtered.judge(child_rel, document):
                    continue
                kept_dirs.append(name)
            dirnames[:] = kept_dirs
            for name in sorted(filenames):
                path = pathlib.Path(dirpath) / name
                try:
                    relpath = path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if filtered.judge(relpath, document):
                    continue
                try:
                    if _search_skippable(path):
                        continue
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(root)
                except OSError as exc:
                    if len(unreadable) < 20:
                        try:
                            display = path.relative_to(root).as_posix()
                        except (OSError, ValueError):
                            display = path.as_posix()
                        unreadable.append(
                            f"{display}: {type(exc).__name__}: {exc}"
                        )
                    continue
                except ValueError:
                    continue
                paths.append(resolved)
                metadata_checked.add(resolved)
                if len(paths) >= _SEARCH_MAX_FILES:
                    scan_limit_hit = True
                    break
            if scan_limit_hit:
                break
    for path in paths:
        if include and not fnmatch.fnmatch(path.name, include):
            continue
        try:
            relpath = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if filtered.judge(relpath, document):
            continue
        try:
            if path not in metadata_checked and _search_skippable(path):
                continue
        except OSError as exc:
            if len(unreadable) < 20:
                try:
                    display = path.relative_to(root).as_posix()
                except (OSError, ValueError):
                    display = path.as_posix()
                unreadable.append(
                    f"{display}: {type(exc).__name__}: {exc}"
                )
            continue
        if path_allowed is not None and not path_allowed(path):
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            unreadable.append(
                f"{path.relative_to(root).as_posix()}: {type(exc).__name__}: {exc}"
            )
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                matches.append(
                    f"active_workspace:{relpath}:{line_no}: {line.rstrip()}"
                )
                if len(matches) >= max_results:
                    truncated = True
                    break
        if truncated:
            break
    # A policy exclusion belongs in `complete` for the same reason `unreadable`
    # does: both mean the walk did not cover the tree it was asked about, so a
    # no-match answer is not authoritative. Only `integrity_complete` (in the
    # disclosure block) stays true — the omission was a decision, not a failure.
    complete = (
        not unreadable
        and not scan_limit_hit
        and not truncated
        and not filtered
    )
    display_path = f"active_workspace:{rel or '.'}"
    if matches:
        header = (
            f"Found {len(matches)} match"
            f"{'es' if len(matches) != 1 else ''} in {display_path} "
            f"({scanned} files searched)"
        )
        if scan_limit_hit:
            header += (
                f" — scan stopped at {_SEARCH_MAX_FILES} files "
                "(narrow the path or glob)"
            )
        if truncated:
            header += f" — truncated at {max_results} results"
        text = header + "\n\n" + "\n".join(matches)
    else:
        text = (
            f"No matches found for {'regex' if regex_mode else 'literal'} "
            f"`{query}` in {display_path} ({scanned} files searched)."
        )
        if scan_limit_hit:
            text += (
                f" Scan stopped after {_SEARCH_MAX_FILES} files — "
                "narrow the path or glob."
            )
    if unreadable:
        text += "\n\n⚠️ SEARCH_PARTIAL: unreadable paths:\n" + "\n".join(
            unreadable[:20]
        )
    if filtered:
        text += "\n\n" + filtered.note("SEARCH_POLICY_FILTERED")
    return ToolExecutionEnvelope(
        text=text,
        trace={
            "completion": "complete" if complete else "partial",
            "scanned_files": scanned,
            "unreadable": unreadable[:20],
            "truncated": truncated,
            **filtered.disclosure(document),
        },
    )


def execute_workspace_query_operation(
    root: pathlib.Path,
    operation: str,
    args: Mapping[str, Any],
    native_facts: Mapping[str, Any],
) -> ToolExecutionEnvelope:
    """Apply the export policy to the two public query operations.

    ``search_code`` returns matched LINE CONTENT, so the query walk is a byte
    channel like any other and gets the same document. Before this it filtered
    only the explicitly protected paths, which meant a `.env` line matched a
    remote search and its bytes reached Home while the very same file was being
    excluded from the snapshot two channels over — the "one policy × N doors"
    class exactly.

    Filtering alone was still not enough. The first version of this dropped the
    excluded paths SILENTLY, so the channel stopped leaking bytes and started
    manufacturing a false negative instead: the model read "No matches found …"
    and concluded the key was absent. Both operations now route their exclusions
    through the same disclosure block declared outputs use — exact count, bounded
    list, `policy_scope=policy_filtered`, honest `complete=false` — so the answer
    the model reasons over says what it did not look at.
    """

    document = snapshot_policy(
        policy_from_facts(native_facts),
        tuple(str(item) for item in native_facts.get("protected_paths") or [] if str(item or "")),
    )
    if operation == "search_code":
        return search_workspace(root, args, policy=document)
    if operation != "query_code":
        raise ValueError(f"unsupported workspace query operation: {operation}")
    return query_workspace(root, args, policy=document)


def git_workspace(
    workspace_root: pathlib.Path | str,
    args: Mapping[str, Any],
    subcommand: list[str],
    *,
    excluded_paths: tuple[str, ...] = (),
    excluded_rows: tuple[Mapping[str, str], ...] = (),
    admitted_paths: tuple[str, ...] = (),
    native_facts: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    """Run one bounded read-only Git projection with public VCS rendering.

    The D7 block rides on BOTH outcomes, and its absence was a finding: this returned
    `trace={"completion": "complete"}` and nothing else, so a `vcs_diff` whose secret-
    carrying paths had been removed by pathspec read as an authoritative complete diff, the
    owner-facing omission note the other channels emit was missing, and Home's
    returned-manifest check found no fields and validated an empty set. A filtered answer
    that says nothing about its filtering is the exact premise the model then reasons from.
    """

    root = pathlib.Path(workspace_root).resolve(strict=True)
    path = _relative_scope(root, args.get("path"))
    cmd = ["git", *subcommand]
    if path or excluded_paths:
        cmd.extend(["--", path or "."])
        cmd.extend(
            f":(exclude,literal){item}"
            for item in sorted(set(excluded_paths))
            if item
        )
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    process = ProcessExecutionResult(
        proc.returncode,
        stdout,
        stderr,
        {"backend": "ssh_exec", "cwd": root.as_posix()},
        cmd,
    )
    rows = [dict(row) for row in excluded_rows]
    disclosure = export_disclosure_block(native_facts, rows, admitted_paths)
    if proc.returncode:
        detail = " ".join((stderr or f"git exited {proc.returncode}").split())
        return ToolExecutionEnvelope(
            text=f"⚠️ GIT_ERROR: {detail}",
            process=process,
            trace={"completion": "complete", **disclosure},
        )
    cap = int(args.get("max_chars") or 0)
    text = stdout
    if cap > 0 and len(text) > cap:
        text = (
            text[:cap]
            + f"\n⚠️ OUTPUT_TRUNCATED: git output limited to {cap} "
            "characters by max_chars."
        )
    if rows:
        text += "\n\n" + policy_filtered_note("VCS_POLICY_FILTERED", rows)
    return ToolExecutionEnvelope(
        text=text,
        process=process,
        # `partial`, because the projection does not cover the paths it was asked about —
        # the same completion a filtered listing reports, for the same reason.
        trace={"completion": "partial" if rows else "complete", **disclosure},
    )


def policy_excluded_git_paths(
    root: pathlib.Path,
    document: Mapping[str, Any],
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    """The EXCLUDED rows and the ADMITTED paths for every path this repo knows about.

    The rules are name shapes, not paths, and re-expressing them as git globs
    would create a second dialect of the same policy — the exact drift this module
    is here to end. So git is asked which paths exist and THE judge judges the
    answer: one policy, one implementation, expressed to git as literals.

    Judged with identity, not spelling: `vcs_status` names paths and `vcs_diff` carries
    bytes, so a hardlink alias in the index is a second name for content the policy
    excludes, and the spelling-only filter this used to be could not see it — a reviewer
    read `+SECRET_TOKEN=hunter2` out of `vcs_diff` through a TRACKED hardlink.

    Both halves are returned because both are owed. The excluded ROWS are the D7
    disclosure these two operations emitted nothing of: `git_workspace` returned
    `trace={"completion": "complete"}` and no policy block, so a filtered diff read as an
    authoritative empty one and Home's own check ran over nothing. The admitted PATHS are
    the evidence half — the set this projection was allowed to name, which is what Home
    re-evaluates.
    """

    known: set[str] = set()
    for argv in (["ls-files", "-z"], ["ls-files", "-z", "--others", "--exclude-standard"]):
        for item in _git_bytes(root, argv, allow=frozenset({0, 128})).split(b"\0"):
            if item:
                known.add(item.decode("utf-8", errors="replace"))
    aliases = AliasIndex(root, document)
    excluded: list[dict[str, str]] = []
    admitted: list[str] = []
    for rel in sorted(known):
        reason, _sentence, judged = judged_exclusion(
            root,
            root.joinpath(*rel.split("/")),
            rel,
            document,
            question=QUESTION_EXPORT,
            aliases=aliases,
        )
        if reason:
            excluded.append({"path": rel, "reason": reason, "judged": judged or rel})
        else:
            admitted.append(rel)
    return tuple(excluded), tuple(admitted)


def execute_git_workspace_operation(
    root: pathlib.Path,
    operation: str,
    args: Mapping[str, Any],
    native_facts: Mapping[str, Any],
) -> NativeOperationResult:
    document = snapshot_policy(
        policy_from_facts(native_facts),
        tuple(str(item) for item in native_facts.get("protected_paths") or [] if str(item or "")),
    )
    if bool(args.get("artifact_export", False)):
        return export_workspace_patch(root, args, policy=document)
    # `vcs_status`/`vcs_diff` render CONTENT (a porcelain line names the path; a
    # diff hunk carries the bytes), so both are byte channels and both are filtered
    # at the source by the same document.
    excluded_rows, admitted = policy_excluded_git_paths(root, document)
    excluded = tuple(row["path"] for row in excluded_rows)
    disclosure = {
        "excluded_rows": excluded_rows,
        "admitted_paths": admitted,
        "native_facts": native_facts,
    }
    if operation == "vcs_status":
        return NativeOperationResult(
            git_workspace(
                root,
                args,
                ["status", "--porcelain"],
                excluded_paths=excluded,
                **disclosure,
            )
        )
    if bool(args.get("recent_commit", False)):
        subcommand = [
            "show",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--stat",
            "-p",
            "HEAD",
        ]
    else:
        subcommand = ["diff"]
        if bool(args.get("staged", False)):
            subcommand.append("--staged")
        if bool(args.get("name_only", False)):
            subcommand.append("--name-only")
        elif bool(args.get("stat", False)):
            subcommand.append("--stat")
    return NativeOperationResult(
        git_workspace(root, args, subcommand, excluded_paths=excluded, **disclosure)
    )


def _git_bytes(
    root: pathlib.Path,
    argv: list[str],
    *,
    allow: frozenset[int] = frozenset({0}),
    input_bytes: bytes | None = None,
) -> bytes:
    proc = subprocess.run(
        ["git", *argv],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=input_bytes,
        timeout=60,
    )
    if proc.returncode not in allow:
        raise RuntimeError(
            proc.stderr.decode("utf-8", errors="replace")
            or f"git {' '.join(argv)} exited {proc.returncode}"
        )
    return bytes(proc.stdout or b"")


def export_workspace_patch(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    protected_paths: tuple[str, ...] = (),
    policy: Mapping[str, Any] | None = None,
) -> NativeOperationResult:
    """Export the workspace diff, MINUS whatever the export policy omits (D7).

    The donor refused the whole export as soon as one touched path looked
    sensitive, which is the bug D7 names: a single `.env` in the remote repo took
    plan review, the claude_code_edit bridge and the file bridge down with it. A
    policy exclusion is not a malfunction — the patch is produced without those
    paths and the omission is DISCLOSED (`complete=false`,
    `integrity_complete=true`, `policy_scope=policy_filtered`, exact
    `excluded_count`, bounded `excluded[]`).

    An IO failure or an unstable snapshot stays fail-closed, because those mean
    nobody knows what the workspace contained. The two conditions are kept apart
    deliberately: conflating them is what turned a policy decision into an outage.
    """

    document = snapshot_policy(policy, tuple(protected_paths))
    before, _ = snapshot_workspace(root, policy=document)
    if not snapshot_integrity_ready(before):
        raise RuntimeError(
            "remote workspace snapshot could not be observed (IO failure or "
            "unstable tree); this is fail-closed and is NOT a policy exclusion"
        )
    expected_head = str(args.get("expected_head") or "")
    current_head = _git_bytes(
        root,
        ["rev-parse", "--verify", "HEAD"],
        allow=frozenset({0, 128}),
    ).decode().strip()
    expected_present = bool(args.get("expected_head_present", bool(expected_head)))
    expected_known = bool(args.get("expected_admission_known", bool(expected_head)))
    if expected_present and expected_head != current_head:
        raise RuntimeError(
            f"workspace HEAD changed: expected={expected_head}, current={current_head}"
        )
    if expected_known and not expected_present and current_head:
        raise RuntimeError(
            f"workspace HEAD changed: expected=<unborn>, current={current_head}"
        )
    base_is_empty_tree = not bool(args.get("base_ref") or expected_head)
    base_ref = str(args.get("base_ref") or expected_head)
    if base_is_empty_tree:
        base_ref = _git_bytes(
            root,
            ["hash-object", "-t", "tree", "--stdin"],
            input_bytes=b"",
        ).decode().strip()
    tracked = [
        item.decode("utf-8", errors="replace")
        for item in _git_bytes(
            root,
            [
                "diff", "--name-only", "--no-renames", "-z", "--no-ext-diff",
                "--no-textconv", "--no-color", base_ref, "--",
            ],
        ).split(b"\0")
        if item
    ]
    untracked = [
        item.decode("utf-8", errors="replace")
        for item in _git_bytes(
            root,
            ["ls-files", "-z", "--others", "--exclude-standard"],
        ).split(b"\0")
        if item
    ]
    scratch_raw = args.get("scratch_fingerprints")
    scratch = (
        {str(path): str(digest) for path, digest in scratch_raw.items()}
        if isinstance(scratch_raw, Mapping)
        else {}
    )
    scratch_excluded: list[str] = []
    kept_untracked: list[str] = []
    for rel in untracked:
        expected = scratch.get(rel)
        candidate = (root / rel).resolve(strict=False)
        try:
            candidate.relative_to(root)
            confined = True
        except ValueError:
            confined = False
        if expected and confined and candidate.is_file():
            try:
                if hashlib.sha256(candidate.read_bytes()).hexdigest() == expected:
                    scratch_excluded.append(rel)
                    continue
            except OSError:
                pass
        kept_untracked.append(rel)
    untracked = kept_untracked
    omitted_paths = tuple(
        str(row.get("path") or "")
        for row in list(before.get("policy_exclusions") or [])
        if isinstance(row, dict) and str(row.get("path") or "")
    )
    # ONE policy evaluation over every path the patch would carry. Whatever it
    # names is removed BEFORE any diff is computed, so the excluded bytes are
    # never read, never hashed and never in the blob — filtering the patch after
    # it reached Home would already be the leak.
    excluded: list[dict[str, str]] = []
    aliases = AliasIndex(root, document)
    for rel in sorted({*tracked, *untracked}):
        reason, _sentence, judged = judged_exclusion(
            root,
            root.joinpath(*rel.split("/")),
            rel,
            document,
            question=QUESTION_EXPORT,
            aliases=aliases,
        )
        if not reason and path_under_any(rel, omitted_paths):
            reason, judged = REASON_PROTECTED_ARTIFACT, rel
        if reason:
            excluded.append({"path": rel, "reason": reason, "judged": judged or rel})
    excluded_set = {row["path"] for row in excluded}
    tracked = [rel for rel in tracked if rel not in excluded_set]
    untracked = [rel for rel in untracked if rel not in excluded_set]
    chunks: list[bytes] = []
    # With nothing excluded the argv is byte-identical to before (a whole-tree
    # diff). With exclusions it becomes pathspec-limited to the KEPT paths, which
    # is how the excluded files stay unopened rather than being stripped later.
    if not excluded or tracked:
        tracked_patch = _git_bytes(
            root,
            [
                "diff", "--binary", "--no-ext-diff", "--no-textconv",
                "--no-color", base_ref, "--", *(tracked if excluded else []),
            ],
        )
        if tracked_patch:
            chunks.append(tracked_patch)
    for rel in untracked:
        patch = _git_bytes(
            root,
            [
                "diff", "--no-index", "--binary", "--no-ext-diff",
                "--no-textconv", "--no-color", "--", os.devnull, rel,
            ],
            allow=frozenset({0, 1}),
        )
        if patch:
            chunks.append(patch)
    patch_bytes = b"\n".join(chunks)
    if len(patch_bytes) > _PATCH_MAX_BYTES:
        raise ValueError("remote workspace patch exceeds export limit")
    after, _ = snapshot_workspace(root, policy=document)
    if not snapshot_integrity_ready(after):
        raise RuntimeError(
            "remote workspace snapshot became unobservable while the patch was "
            "exported (IO failure or unstable tree); fail-closed, not a policy exclusion"
        )
    if before["fingerprint"] != after["fingerprint"]:
        raise RuntimeError("remote workspace changed while patch was exported")
    digest = hashlib.sha256(patch_bytes).hexdigest() if patch_bytes else ""
    status = "ready_with_changes" if patch_bytes else "ready_no_changes"
    artifact = (
        {
            "name": "workspace.patch", "blob_id": digest, "sha256": digest,
            "size": len(patch_bytes), "mime": "text/x-diff",
        }
        if patch_bytes
        else None
    )
    export = {
        "status": status,
        "base_ref": base_ref,
        "base_head": current_head,
        "base_is_empty_tree": base_is_empty_tree,
        "current_head": current_head,
        "tracked_changed": tracked,
        "untracked_included": untracked,
        "scratch_excluded": scratch_excluded,
        # D7 wire shape, identical to the snapshot manifest's: a policy-filtered
        # export is COMPLETE work with a disclosed omission, not a failure.
        "complete": not excluded,
        "integrity_complete": True,
        "policy_scope": "policy_filtered" if excluded else "full",
        "policy_hash": export_policy_hash(document),
        "excluded": excluded,
        "excluded_count": len(excluded),
        "patch_size": len(patch_bytes),
        "sha256": digest,
        "snapshot_fingerprint": before["fingerprint"],
    }
    return NativeOperationResult(
        ToolExecutionEnvelope(
            text=json.dumps(export, sort_keys=True),
            artifacts=(artifact,) if artifact else (),
            trace={"completion": "complete", "patch_export": export},
        ),
        {digest: patch_bytes} if patch_bytes else {},
    )


def _relative_scope(root: pathlib.Path, value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if text in {"", "."}:
        return ""
    candidate = pathlib.PurePosixPath(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"path escapes root: {value}")
    target = root.joinpath(*candidate.parts).resolve(strict=False)
    try:
        return target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path escapes root: {value}") from exc


def _inventory_rows(inventory: Any, options: Mapping[str, Any]) -> list[str]:
    op = str(options.get("op") or "")
    query = str(options.get("query") or "")
    path = str(options.get("path") or "")
    kind = str(options.get("kind") or "any")
    depth = int(options.get("depth") or 1)
    limit = int(options.get("limit") or 40)
    offset = int(options.get("offset") or 0)
    rows: list[str] = []
    if op in {"symbols", "definition"}:
        for file, symbol in symbol_definitions(
            inventory,
            query,
            path=path,
            kind=kind or "any",
        ):
            rows.append(
                f"{file.path}:{symbol.line_start} {symbol.kind} "
                f"{symbol.signature or symbol.name}"
            )
    elif op == "references":
        for file, ref in symbol_references(inventory, query, path=path):
            enclosing = f" in {ref.enclosing}" if ref.enclosing else ""
            rows.append(f"{file.path}:{ref.line} {query}{enclosing}")
    elif op in {"callers", "callees"}:
        iterator = (
            symbol_callers(inventory, query, path=path)
            if op == "callers"
            else symbol_callees(inventory, query, path=path)
        )
        for file, call in iterator:
            enclosing = f"{call.enclosing} -> " if call.enclosing else ""
            rows.append(f"{file.path}:{call.line} {enclosing}{call.name}")
    elif op == "impact":
        for file, reason in impact_files(inventory, path or query, depth=depth):
            rows.append(f"{file.path}  {reason}")
    elif op == "relevant_files":
        selected = relevant_files(
            inventory,
            query,
            limit=min(_MAX_LIMIT, offset + limit),
        )
        for index, (file, score, reason) in enumerate(selected, 1):
            # The symbol sample is bounded, and the bound says so: a model deciding
            # which file to open reasons about the names it can see, so "5 symbols" and
            # "5 of 40 symbols" are different facts (BIBLE P1, no silent elision).
            shown = list(file.symbols[:_RELEVANT_FILE_SYMBOLS])
            symbols = ", ".join(symbol.name for symbol in shown)
            if len(file.symbols) > len(shown):
                symbols += f", +{len(file.symbols) - len(shown)} more"
            suffix = f" symbols={symbols}" if symbols else ""
            rows.append(
                f"{index}. {file.path} score={score:.2f} reason={reason}{suffix}"
            )
    return rows


def _node_type(query: str) -> str:
    text = str(query or "").strip()
    if text.startswith("("):
        match = re.match(r"\(\s*([A-Za-z_][\w-]*)", text)
        return match.group(1) if match else ""
    return text


def _tree_rows(grammar: str, rel: str, text: str, node_type: str) -> list[str] | None:
    from ouroboros import code_intelligence

    parser = code_intelligence._ts_parser(grammar)
    if parser is None:
        return None
    try:
        tree = parser.parse(text.encode("utf-8", errors="replace"))
    except Exception:
        return None
    rows: list[str] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == node_type:
            rows.append(f"{rel}:{int(node.start_point[0]) + 1} {node.type}")
        stack.extend(reversed(list(node.children)))
    return rows


def _structural_rows(
    root: pathlib.Path,
    *,
    query: str,
    path: str,
    lang: Any,
    limit: int,
    visible: Callable[[str], bool] | None,
) -> tuple[list[str], list[str]]:
    wanted_type = _node_type(query)
    grammar_text = str(lang or "").strip().lower()
    wanted_grammar = (
        None
        if grammar_text in {"", "any"}
        else _TS_LANGUAGES.get(grammar_text, grammar_text)
    )
    rows: list[str] = []
    issues: list[str] = []
    unavailable: set[str] = set()
    scope = (root / (path or ".")).resolve(strict=False)
    candidates, walk_note = walk_candidate_files(scope, root)
    for file_path in candidates:
        if len(rows) >= limit:
            break
        try:
            relative = file_path.relative_to(root).as_posix()
        except ValueError:
            continue
        if visible is not None and not visible(relative):
            continue
        lang_id = _language(file_path)
        grammar = "python" if lang_id == "python" else _TS_LANGUAGES.get(lang_id)
        if wanted_grammar is not None and grammar != wanted_grammar:
            continue
        if grammar is None or not wanted_type:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            issues.append(f"unreadable:{relative}")
            continue
        tree_rows = _tree_rows(grammar, relative, text, wanted_type)
        if tree_rows or (tree_rows is not None and lang_id != "python"):
            rows.extend(tree_rows[: max(0, limit - len(rows))])
            continue
        if lang_id != "python":
            if lang_id not in unavailable:
                unavailable.add(lang_id)
                rows.append(
                    f"structural_unavailable:{lang_id} "
                    "(tree-sitter grammar not loaded)"
                )
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if node.__class__.__name__.casefold() == wanted_type.casefold():
                rows.append(
                    f"{relative}:{int(getattr(node, 'lineno', 0) or 0)} "
                    f"{node.__class__.__name__}"
                )
                if len(rows) >= limit:
                    break
    if walk_note and len(rows) < limit:
        issues.append(f"walk_truncated:{walk_note}")
    return rows, issues


def _empty_hint(op: str, label: str) -> str:
    if op in {"definition", "references", "callers", "callees", "impact"}:
        return (
            "Check the exact symbol name (these ops match a defined symbol, not "
            f"text). Use op=relevant_files query=\"{label}\" to find where to "
            "look, or op=symbols to list what's defined."
        )
    if op == "symbols":
        return (
            "Narrow with path= to a file/dir, or use op=relevant_files to locate "
            "the area first."
        )
    if op == "structural":
        return (
            "structural needs a node type, not free text — an AST class for "
            "Python (FunctionDef/ClassDef) or a tree-sitter node for other "
            "langs (function_declaration for Go, struct_item for Rust, etc.). "
            "Add lang=go|rust|... to filter by language."
        )
    if op == "relevant_files":
        return (
            "Rephrase the task in domain words, or use search_code for an exact "
            "string you expect in the source."
        )
    return "Verify the symbol/path; use search_code only for plain-text matches."


def _next_step_hint(op: str) -> str:
    return {
        "relevant_files": (
            "\n\nNext: read_file(...) the top hit, or "
            "query_code(op=symbols, path=...) to list its symbols."
        ),
        "symbols": (
            "\n\nNext: query_code(op=definition/references, query=<name>) on a "
            "symbol of interest."
        ),
        "definition": (
            "\n\nNext: query_code(op=references/callers, query=<name>) to see "
            "how it is used."
        ),
        "callers": (
            "\n\nNext: read_file(...) a caller, or "
            "query_code(op=impact, query=<name>) for blast radius."
        ),
        "callees": (
            "\n\nNext: query_code(op=definition, query=<callee>) to read what it calls."
        ),
    }.get(op, "")


def query_workspace(
    workspace_root: pathlib.Path | str,
    args: Mapping[str, Any],
    *,
    inventory: Any | None = None,
    policy: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    """Run a model-visible query_code operation without Home authority imports.

    Takes the policy DOCUMENT rather than an opaque visibility predicate on
    purpose: a bare predicate can only say no, and a channel that can only say no
    has nothing to disclose. With the document the walk can name the reason THE
    evaluator gave, which is what makes the omission reportable.
    """

    root = pathlib.Path(workspace_root).resolve(strict=True)
    document = snapshot_policy(policy) if policy is not None else None
    filtered = _PolicyExclusions(root, document)
    visible: Callable[[str], bool] | None = None
    exclude_paths: list[pathlib.Path] | None = None
    if document is not None:
        visible = lambda relative: not filtered.judge(relative, document)  # noqa: E731
        # EVERY path the policy excludes is kept out of the inventory walk, not only the
        # protected ones, and that is the order the reviewer found reversed: the builder
        # was handed `protected_paths` alone, so it READ AND PARSED every `id_rsa`,
        # `credentials.json` and `.netrc` in the tree — extracting symbols, imports and
        # routes from them — and `visible` removed the rows only afterwards. Derived
        # content from an excluded file had already been computed, which is a source-side
        # policy claim the source was not keeping.
        #
        # `policy_excluded_git_paths` is the authority because `build_code_inventory`
        # enumerates through `git ls-files --cached --others`, the same set: one walk, one
        # judge, no second dialect.
        exclude_paths = []
        for rel in (
            *(str(item) for item in document.get("protected_paths") or ()),
            *(row["path"] for row in policy_excluded_git_paths(root, document)[0]),
        ):
            target = root / rel
            if target in exclude_paths:
                continue
            exclude_paths.append(target)
            if target.exists():
                filtered.judge(rel, document)
    op = str(args.get("op") or "").strip()
    query = str(args.get("query") or "")
    if op not in QUERY_OPERATIONS:
        allowed = ", ".join(QUERY_OPERATION_ORDER)
        return ToolExecutionEnvelope(
            f"⚠️ TOOL_ARG_ERROR (query_code): op must be one of {allowed}."
        )
    if op not in {"symbols", "digest"} and not query.strip():
        return ToolExecutionEnvelope(
            f"⚠️ TOOL_ARG_ERROR (query_code): op '{op}' requires query."
        )
    try:
        path = _relative_scope(root, args.get("path"))
        limit = min(max(1, int(args.get("limit") or 40)), _MAX_LIMIT)
        offset = max(0, int(args.get("offset") or 0))
        options = {
            **dict(args),
            "op": op,
            "query": query,
            "path": path,
            "limit": limit,
            "offset": offset,
        }
        issues: list[str] = []
        direct_text = ""
        if op == "structural":
            rows, issues = _structural_rows(
                root,
                query=query,
                path=path,
                lang=args.get("lang"),
                limit=min(_MAX_LIMIT, offset + limit),
                visible=visible,
            )
        else:
            if inventory is None:
                inventory = build_code_inventory(
                    root,
                    persist=False,
                    exclude_paths=exclude_paths,
                )
            if visible is not None:
                inventory.files = [
                    fact for fact in inventory.files if visible(str(fact.path))
                ]
            issues = [
                f"{fact.disposition}:{fact.path}"
                for fact in inventory.files
                if fact.disposition.startswith("read_error:")
                or fact.disposition.startswith("structural_unavailable:")
            ]
            if op == "digest":
                direct_text = render_codebase_digest(inventory)
                rows = []
            else:
                rows = _inventory_rows(inventory, options)
    except Exception as exc:
        return ToolExecutionEnvelope(
            f"⚠️ QUERY_CODE_ERROR: {type(exc).__name__}: {exc}",
            trace={"op": op, "completion": "unknown"},
        )
    total = len(rows)
    shown = rows[offset : offset + limit]
    label = query or path or "."
    if direct_text:
        text = direct_text
    elif not shown:
        text = f"No results for op `{op}` `{label}`. {_empty_hint(op, label)}"
    else:
        text = f"{op} `{label}` — {len(shown)} of {total}"
        if offset + limit < total:
            text += f" — next offset={offset + limit}"
        text += "\n\n" + "\n".join(shown) + _next_step_hint(op)
    diagnostic = None
    completion = "complete"
    if filtered:
        completion = "partial"
        text += "\n\n" + filtered.note("QUERY_POLICY_FILTERED")
    if issues:
        completion = "partial"
        text += (
            "\n\n⚠️ QUERY_PARTIAL: some workspace files were not readable or "
            "structurally available; a no-result answer is not authoritative.\n"
            + "\n".join(issues[:20])
        )
        diagnostic = ExecutionDiagnostic(
            domain="filesystem",
            code="query_partial",
            message="Workspace query covered only a readable structural subset.",
            phase="execute",
            completion="unknown",
            retryable=True,
            details={"issues": issues[:20]},
        )
    return ToolExecutionEnvelope(
        text,
        diagnostic=diagnostic,
        trace={
            "op": op,
            "completion": completion,
            "issues": issues[:20],
            **filtered.disclosure(document),
        },
    )
