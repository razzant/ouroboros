"""Carrier-aware managed-update conflict resolution (owner-ratified: spec §1.9-10,
batch №8 answer 6=A; mandatory v7next return, owner answers 5.12-5.14=A).

ONE shared resolver serves all three managed-update insertion points in
``supervisor/update_merge_plan.py`` — the isolated-worktree planner merge, the
clean-plan base re-merge (both applied BEFORE write-tree) and the live assisted
materializer (applied BEFORE the mechanical M0 baseline is pinned, so reviewers
diff a baseline that already carries the span policy) — plus, with the opposite
preference, the operator rebase helper ``scripts/carrier_rebase_helper.py``.

A merge conflict in a release-carrier file is resolved ONLY when every conflict
in it sits inside a declared carrier span: each span in every stage is
substituted with the preferred side's span (the incoming official side for
managed updates), the remainder is re-merged as an ordinary textual 3-way, and
the file is staged iff that re-merge is clean. Anything else — a malformed or
duplicate span anchor, an unreadable, missing or non-UTF-8 stage, overlapping
spans, a conflict OUTSIDE the spans — leaves the file on the ordinary
assisted-conflict path: never a crash, never silent adoption, and never
whole-file theirs (only the spans themselves change sides).

The span descriptors AND the span-substitution primitive
(``substitute_carrier_spans``) are owned by ``ouroboros.tools.release_sync``
(the release-carrier SSOT; the commit-review pack cut reads the same
primitive), imported at call time so importing the update machinery never
drags the tool package in. Every git invocation here is BOUNDED (the
update-flow redesign's plumbing rule): the live-materializer insertion point
runs while the update lock is held, and a hung git must die with its whole
process tree instead of wedging the update flow. The resolver itself never runs
``git merge`` — it substitutes explicit index stages and re-merges them with
``git merge-file`` — so the engine is neutral to rerere by construction, in
line with the update flow's ``_MERGE_NEUTRAL_FLAGS`` discipline. Honest frame:
the FIRST pre-v7 upgrade is driven by the OLD updater, which never calls this
module; the policy targets steady state (7.0.0 -> 7.0.1 and beyond).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Index stages of a conflicted path: 1 = merge base, 2 = ours, 3 = theirs.
# During a managed update "theirs" is the official target in all three
# insertion points; during a rebase "ours" is the side being rebased onto.
_PREFER_STAGE = {"ours": 2, "theirs": 3}

_CARRIER_GIT_TIMEOUT_SEC = 300.0
"""Wall-clock bound for one carrier-resolution git invocation, mirroring
``supervisor.update_candidate._GIT_RUN_TIMEOUT_SEC``. Kept as a local constant
(and the runner below self-contained) so the operator rebase helper can import
this module standalone without dragging git_ops or the tool registry in."""


def _run_git(
    worktree: str, args: List[str], *, input_bytes: Optional[bytes] = None
) -> Tuple[int, bytes, bytes]:
    """Run git in *worktree* with byte-exact capture (no newline translation).

    BOUNDED per the update-flow plumbing rule: the process starts in its own
    group and the WHOLE TREE is killed on timeout (``platform_layer`` helpers,
    imported at call time to keep this module standalone-importable)."""
    from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs

    cmd = ["git", "-C", str(worktree), *args]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_bytes is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **subprocess_new_group_kwargs(),
        )
    except OSError as exc:
        return 127, b"", str(exc).encode("utf-8", "replace")
    try:
        stdout, stderr = proc.communicate(input=input_bytes, timeout=_CARRIER_GIT_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        message = (
            f"git timed out after {_CARRIER_GIT_TIMEOUT_SEC:g}s and was terminated: "
            + " ".join(cmd)
        )
        return 124, b"", message.encode("utf-8", "replace")
    return proc.returncode, stdout or b"", stderr or b""


def _stage_text(worktree: str, stage: int, path: str) -> Optional[str]:
    """UTF-8 text of one index stage, or None (missing stage / undecodable)."""
    rc, out, _err = _run_git(worktree, ["show", f":{stage}:{path}"])
    if rc != 0:
        return None
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _merge_span_substituted_texts(
    current: str, base: str, other: str
) -> Tuple[Optional[str], str]:
    """Ordinary textual 3-way over the span-substituted stages.

    Clean merge -> ``(merged_text, "")``. Remaining conflicts mean a conflict
    OUTSIDE the carrier spans -> ``(None, "conflict_outside_carrier_span")``."""
    with tempfile.TemporaryDirectory(prefix="ouro-carrier-merge-") as tmp:
        stage_paths: List[str] = []
        for name, content in (("current", current), ("base", base), ("other", other)):
            stage_path = os.path.join(tmp, name)
            with open(stage_path, "wb") as handle:
                handle.write(content.encode("utf-8"))
            stage_paths.append(stage_path)
        # merge-file needs no repository; -C pins the bounded runner to the
        # temp dir and the absolute stage paths stay valid either way.
        rc, out, _err = _run_git(tmp, ["merge-file", "-p", "--", *stage_paths])
        if rc == 0:
            try:
                return out.decode("utf-8"), ""
            except UnicodeDecodeError:
                return None, "merge_result_undecodable"
        # Positive exit = number of remaining conflicts; anything else = error.
        if 0 < rc <= 127:
            return None, "conflict_outside_carrier_span"
        return None, "merge_file_failed"


def resolve_carrier_conflict_file(
    worktree: str, path: str, prefer: str
) -> Tuple[bool, str]:
    """Resolve ONE conflicted carrier file in *worktree*; (resolved, reason)."""
    from ouroboros.tools.release_sync import carrier_spans_for, substitute_carrier_spans

    spans = carrier_spans_for(path)
    if not spans:
        return False, "not_a_carrier"
    stage_texts: Dict[int, str] = {}
    for stage in (1, 2, 3):
        text = _stage_text(worktree, stage, path)
        if text is None:
            return False, f"stage_{stage}_unavailable"
        stage_texts[stage] = text
    preferred_text = stage_texts[_PREFER_STAGE[prefer]]
    substituted: Dict[int, str] = {}
    for stage in (1, 2, 3):
        text, reason = substitute_carrier_spans(stage_texts[stage], spans, preferred_text)
        if text is None:
            return False, reason
        substituted[stage] = text
    merged, reason = _merge_span_substituted_texts(
        substituted[2], substituted[1], substituted[3]
    )
    if merged is None:
        return False, reason
    absolute = os.path.join(str(worktree), path.replace("/", os.sep))
    try:
        with open(absolute, "wb") as handle:
            handle.write(merged.encode("utf-8"))
    except OSError:
        return False, "worktree_write_failed"
    rc_add, _out, _err = _run_git(worktree, ["add", "--", path])
    if rc_add != 0:
        return False, "stage_failed"
    return True, ""


def resolve_carrier_conflicts(
    worktree: str, conflict_paths: List[str], *, prefer: str = "theirs"
) -> Dict[str, Any]:
    """Resolve carrier-span conflicts among *conflict_paths* in *worktree*.

    Returns ``{"resolved": [paths staged here], "kept": {path: reason}}``.
    ``prefer`` picks the winning side INSIDE the spans only: ``"theirs"`` for
    managed updates (the official target), ``"ours"`` for tactical rebases.
    Per-file failures degrade that file to the assisted path — this function
    never raises for a file it cannot resolve."""
    if prefer not in _PREFER_STAGE:
        raise ValueError(f"unsupported carrier preference: {prefer!r}")
    resolved: List[str] = []
    kept: Dict[str, str] = {}
    for raw_path in conflict_paths:
        path = str(raw_path).strip()
        if not path:
            continue
        try:
            ok, reason = resolve_carrier_conflict_file(worktree, path, prefer)
        except Exception:  # degrade, never crash the update machinery
            ok, reason = False, "resolver_error"
        if ok:
            resolved.append(path)
        else:
            kept[path] = reason
    return {"resolved": resolved, "kept": kept}
