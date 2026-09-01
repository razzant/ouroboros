"""Ripgrep-backed search helper for the search_code tool.

This module is intentionally policy-agnostic: callers must post-filter every
returned path through Ouroboros's protected/secret gates before surfacing it.
"""

from __future__ import annotations

import fnmatch
import json
import os
import pathlib
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple

# Files larger than this are skipped by search (binary blobs, vendored data).
MAX_FILE_SIZE_BYTES = 1024 * 1024  # 1 MB
# Hard ceiling on files read by the Python search fallback so a search whose
# root resolves to a very large tree (e.g. ``user_files`` == ``/`` under a bench
# HOME=/) cannot walk unbounded. ripgrep (the primary path) streams and needs
# no such cap; this only guards the degraded fallback.
MAX_SEARCH_FILES_SCANNED = 50000
# Per-rg-invocation argv length budget (chars). Targets are batched so a single
# rg command line stays well under the OS limit — Windows' CreateProcess caps at
# ~32767 chars, far below POSIX ARG_MAX — so many long paths cannot WinError-206.
_ARGV_CHAR_BUDGET = 28000


def _search_wall_clock_sec() -> float:
    """Total wall-clock budget for one search_with_rg call (the config SSOT getter).

    The file-count cap bounds memory but NOT time: a walk over a very large root
    (``user_files`` == ``/`` under a bench HOME) can traverse for minutes filtering
    non-matching files, and the batched rg loop runs ``timeout=60`` PER batch
    (unbounded in total). This budget bounds the rg path's walk + batch loop; the
    Python fallback in tools/core.py shares the SAME budget from one start time so
    the WHOLE search_code call is bounded, not each stage. Overridable for slow hosts.
    """
    from ouroboros.config import get_search_code_wall_sec

    return get_search_code_wall_sec()


def search_skip_reason(path: pathlib.Path) -> str:
    """Typed reason a path is not searchable, or ``""`` (D3, capinv-447).

    Closed vocabulary (file-transport facts, not semantics): ``excluded_name``,
    ``symlink``, ``stat_error``, ``not_regular_file``, ``oversized``. Skips
    name-pattern excludes, oversized files, AND non-regular files — device
    nodes, FIFOs, sockets. ``/dev/zero`` and ``/proc`` pseudo-files report
    st_size 0 (so a size guard alone lets them through) and read_text() on
    them never terminates / grows memory without bound (the search_code OOM
    root cause when a root resolves to ``/``).
    """
    try:
        from ouroboros.code_intelligence import SEARCH_SKIP_GLOBS
    except Exception:
        SEARCH_SKIP_GLOBS = frozenset()
    name = path.name
    for glob_pat in SEARCH_SKIP_GLOBS:
        if fnmatch.fnmatch(name, glob_pat):
            return "excluded_name"
    try:
        if path.is_symlink():
            # Never follow symlinks: one inside an allowed root can point outside it,
            # letting search_code read bytes beyond the resource confinement boundary.
            return "symlink"
    except OSError:
        return "stat_error"
    try:
        st = path.stat()
    except OSError:
        return "stat_error"
    if not stat.S_ISREG(st.st_mode):
        return "not_regular_file"
    if st.st_size > MAX_FILE_SIZE_BYTES:
        return "oversized"
    return ""


def is_search_skippable(path: pathlib.Path) -> bool:
    """Boolean projection of :func:`search_skip_reason` for gate call sites."""
    return bool(search_skip_reason(path))


@dataclass(frozen=True)
class RgMatch:
    path: pathlib.Path
    line: int
    text: str


class RgSearchResult(NamedTuple):
    """Outcome of a ripgrep search: the matches plus two independent caps.

    truncated = the max_results cap was hit; file_capped = the file-scan cap
    (MAX_SEARCH_FILES_SCANNED) was hit so parts of the tree were not searched;
    deadline_hit = the wall-clock budget expired before all files/batches were
    searched (results — incl. a "no matches" — may be INCOMPLETE, distinct from
    the max_results cap). Grouping these keeps ``format_search_result`` within the
    parameter limit.
    """

    matches: list[RgMatch]
    truncated: bool
    file_capped: bool
    deadline_hit: bool = False


def _rg_binary() -> str:
    try:
        from ouroboros.platform_layer import resolve_bundled_ripgrep

        candidate = resolve_bundled_ripgrep()
        if candidate:
            return candidate
    except Exception:
        pass
    candidate = shutil.which("rg")
    return candidate or ""


def search_with_rg(
    search_targets: pathlib.Path | list[pathlib.Path],
    query: str,
    *,
    regex: bool,
    include: str = "",
    max_results: int = 200,
    path_allowed: Callable[[pathlib.Path], bool] | None = None,
) -> RgSearchResult:
    """Return an ``RgSearchResult`` (matches, truncated, file_capped)."""
    rg = _rg_binary()
    if not rg:
        raise FileNotFoundError("rg not found")
    base_cmd = [rg, "--json", "--line-number", "--color", "never"]
    if not regex:
        base_cmd.append("--fixed-strings")
    if include:
        base_cmd.extend(["--glob", include])

    # Build an EXPLICIT, gated file list and hand it to rg. Pre-filtering each
    # path through ``path_allowed`` (the protected/secret/skippable gate) BEFORE
    # rg sees it is a security property: rg must never READ a file the caller is
    # not permitted to see, even though matches are also post-filtered. To stay
    # memory- and ARG_MAX-bounded on a huge root (e.g. user_files == / under a
    # bench HOME=/), the walk prunes SKIP_DIRS and non-regular files (via
    # path_allowed -> _is_search_skippable), caps the file count, and the rg
    # calls are BATCHED instead of one giant argv (the previous os.walk-into-one-
    # exec approach E2BIG'd / OOM'd and was the search_code SIGKILL root cause).
    deadline = time.monotonic() + _search_wall_clock_sec()
    capped = False
    deadline_hit = False
    if isinstance(search_targets, list):
        targets = [p for p in search_targets if path_allowed is None or path_allowed(p)]
    elif search_targets.is_dir():
        try:
            from ouroboros.code_intelligence import SKIP_DIRS
        except Exception:
            SKIP_DIRS = frozenset()
        targets = []
        for dirpath, dirnames, filenames in os.walk(str(search_targets)):
            if time.monotonic() > deadline:
                deadline_hit = True  # ran out of time enumerating — NOT a file-count cap
                break
            dirnames[:] = [n for n in sorted(dirnames) if n not in SKIP_DIRS]
            for fname in sorted(filenames):
                if include and not fnmatch.fnmatch(fname, include):
                    continue
                path = pathlib.Path(dirpath) / fname
                if path_allowed is not None and not path_allowed(path):
                    continue
                targets.append(path)
                if len(targets) >= MAX_SEARCH_FILES_SCANNED:
                    capped = True
                    break
            if capped:
                break
    else:
        targets = [search_targets] if path_allowed is None or path_allowed(search_targets) else []
    if not targets:
        return RgSearchResult([], False, capped, deadline_hit)

    # Pack targets into batches bounded by total argv LENGTH (not a fixed count),
    # so N long paths cannot overflow the OS command-line limit. Always keep at
    # least one path per batch. Budget is the module-level _ARGV_CHAR_BUDGET.
    batches: list[list[pathlib.Path]] = []
    cur: list[pathlib.Path] = []
    cur_len = 0
    for _p in targets:
        plen = len(str(_p)) + 1
        if cur and cur_len + plen > _ARGV_CHAR_BUDGET:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(_p)
        cur_len += plen
    if cur:
        batches.append(cur)

    matches: list[RgMatch] = []
    result_truncated = False  # hit the max_results cap (distinct from file-scan cap)
    for batch in batches:
        if len(matches) >= max_results:
            result_truncated = True
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            deadline_hit = True  # wall-clock budget gone — distinct from the max_results cap
            break
        cmd = base_cmd + ["--", query] + [str(p) for p in batch]
        # Bound each rg call by the REMAINING wall-clock budget (<=60s) so the total
        # search cannot run 60s-per-batch unbounded.
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=min(60.0, max(1.0, remaining)))
        if proc.returncode not in (0, 1):
            detail = (proc.stderr or proc.stdout or "").strip()[:500]
            raise RuntimeError(detail or f"rg exited {proc.returncode}")
        for raw in proc.stdout.splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data") or {}
            path_text = ((data.get("path") or {}).get("text") or "").strip()
            if not path_text:
                continue
            path = pathlib.Path(path_text)
            if path_allowed is not None and not path_allowed(path):
                continue
            lines = data.get("lines") or {}
            text = str(lines.get("text") or "").rstrip("\n")
            line = int(data.get("line_number") or 0)
            if len(matches) >= max_results:
                result_truncated = True
                break
            matches.append(RgMatch(path=path, line=line, text=text.rstrip()))
        if result_truncated:
            break
    # Return result-cap, file-scan-cap, and deadline SEPARATELY so the caller can tell
    # an honest "no matches" from "scan/time stopped before the whole tree was seen".
    return RgSearchResult(matches, result_truncated, capped, deadline_hit)


def format_dropped_files_note(dropped: dict[str, int] | None) -> str:
    """Render the filter receipt for files present but excluded from a search.

    D3 (capinv-447): count-and-time caps were already disclosed; this discloses
    the ORDINARY exclusions (oversized, symlink, policy-filtered, unreadable)
    so a clean "No matches" is only ever claimed when nothing was dropped.
    """
    total = sum(int(v) for v in (dropped or {}).values() if v)
    if not total:
        return ""
    details = ", ".join(f"{key}={int(count)}" for key, count in sorted((dropped or {}).items()) if count)
    return f" {total} file(s) were present but not searched ({details})."


def format_search_result(
    *,
    display_path: str,
    root_name: str,
    root_path: pathlib.Path,
    query: str,
    regex: bool,
    max_results: int,
    result: RgSearchResult,
    dropped: dict[str, int] | None = None,
) -> str:
    matches, truncated, file_capped = result.matches, result.truncated, result.file_capped
    deadline_hit = result.deadline_hit
    cap_note = (
        f" Scan stopped after {MAX_SEARCH_FILES_SCANNED} files — large parts of "
        "the tree were not searched; narrow the path or glob."
        if file_capped else ""
    )
    # A deadline cutoff means even a "no matches" may be INCOMPLETE — surface it in BOTH
    # branches so the caller never reads a timed-out search as an authoritative empty result.
    deadline_note = (
        " Search stopped at the time budget before the whole tree was scanned — results "
        "may be incomplete; narrow the path or glob, or raise OUROBOROS_SEARCH_CODE_WALL_SEC."
        if deadline_hit else ""
    )
    dropped_note = format_dropped_files_note(dropped)
    rendered = [f"{root_name}:{m.path.relative_to(root_path).as_posix()}:{m.line}: {m.text}" for m in matches]
    if not rendered:
        # Surface the file-scan cap, the deadline, AND the filter receipt here: a
        # capped/timed-out/filtered huge-root search with zero matches must never
        # look like a clean "no matches" (the misleading case).
        return f"No matches found for {'regex' if regex else 'literal'} `{query}` in {display_path} (ripgrep).{cap_note}{deadline_note}{dropped_note}"
    header = f"Found {len(rendered)} match{'es' if len(rendered) != 1 else ''} in {display_path} (ripgrep)"
    if truncated:
        header += f" — truncated at {max_results} results"
    if file_capped:
        header += f" — scan stopped at {MAX_SEARCH_FILES_SCANNED} files (narrow the path/glob)"
    if deadline_hit:
        header += " — stopped at the time budget (results may be incomplete)"
    if dropped_note:
        header += " —" + dropped_note.rstrip(".")
    return header + "\n\n" + "\n".join(rendered)
