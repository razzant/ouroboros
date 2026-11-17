"""``integrate_subagent_patch``: the parent's manifest-first integration tool.

A mutative (acting) subagent returns its changes as a ``workspace.patch`` artifact
(produced by headless finalization, a git diff against the child's base commit).
The parent decides what to do with it — apply one (best-of-N), synthesize several,
or reject. This tool applies isolated self_worktree patches to the parent's repo,
or verifies native external_workspace changes at the child’s recorded shared target
under current read authority, even when the parent works elsewhere. Nothing is
transferred into the parent’s folder. A shared result that no longer verifies stays
undecided until the parent explicitly accepts it as it stands (its judgment, recorded
beside each captured path's observed state) or rejects it.
The parent stays the sole committer: applying stages changes but never
commits; the parent reviews and runs ``commit_reviewed`` itself.

Only the immediate parent integrates a child; the one exception is a root the
owner's explicit Continue created, which may decide a finished direct child of
its recorded predecessor (``_continued_lineage``). A bound copy returns to its
recorded source under the deciding task's current write rights; legacy patches
target the active repo. Descendant patches still bubble up one level at a time.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import stat
import subprocess
import tempfile
import zipfile
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import append_published_text, completed_local_read
from ouroboros.tool_access_paths import entry_path
from ouroboros.task_results import cancellation_blocks_child_result, load_task_result
from ouroboros.task_status import effective_task_result
from ouroboros.tools.join_ledger import (
    _cancelled_result_facts, _cancelled_result_note, _cancelled_result_refusal, _child_result_sha256,
    _record_child_result_disposition,
)
from ouroboros.workspace_file_outputs import (
    file_output_changes, file_output_states, prepare_file_outputs, verify_file_outputs,
)
from ouroboros.review_state import invalidate_advisory_after_mutation
from ouroboros.runtime_mode_policy import (
    mode_allows_protected_write,
    protected_paths_in,
)
from ouroboros.workspace_copies import protected_apply_refusal
from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE
from ouroboros.config import get_runtime_mode
from ouroboros.headless import ARTIFACT_STATUS_READY_WITH_CHANGES

# Git's own retargeting variables: they would make a Git command read a FOREIGN repository
# (or turn a plain folder into one), so the reverse patch check strips them and lets the
# selected root alone decide.
_GIT_RETARGET_ENV = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE",
})

log = logging.getLogger(__name__)


def _record_integration_disposition(
    ctx: ToolContext,
    child_task_id: str,
    disposition: str,
    reason: str,
    result_sha: str,
    lineage: Dict[str, str] | None = None,
    verdict: str = "",
    unretained: List[str] = (),
) -> str:
    """Stamp a completed operation against the consumed result, never a newer CURRENT.

    An owner-Continue successor's decision is its own verdict, confirmed on the
    canonical custody marks the next successor reads; the task-tree row binds the
    child's own root/parent, which this task is not, so it stays unwritten.
    ``verdict`` is the writer's artifact path, empty when that write failed;
    ``unretained`` its failure facts, each reported beside the decision.
    """
    facts = [text for fact, text in _UNRETAINED.items() if fact in unretained]
    gap = f"\n⚠️ This decision's evidence is incomplete: {'; '.join(facts)}." if facts else ""
    if lineage:
        from ouroboros.tool_access_paths import canonical_data_root

        try:
            landed = _recorded_decision(canonical_data_root(ctx), [str(getattr(ctx, "task_id", "") or "")],
                                        child_task_id, lineage["child_result_sha256"])
        except Exception:
            landed = {}
        if not landed:
            retained = ("a later Continue successor can read the retained verdict artifact above"
                        if verdict else "the verdict artifact was not written either; no retained decision "
                        "has been confirmed")
            return ("\n⚠️ INTEGRATE_DISPOSITION_FAILED: this decision's custody record did not land or cannot "
                    f"be read, so acceptance evidence omits it; {retained}.")
        return (f"{gap}\nDecided as the owner-Continue successor of {lineage['original_parent_task_id']}: "
                "the verdict is the decision record; that task's tree ledger is unchanged.")
    recorded = _record_child_result_disposition(ctx, {
        "type": "child_result_disposition", "child_task_id": child_task_id,
        "disposition": disposition, "child_result_sha256": result_sha,
    }, reason)
    if recorded.startswith("OK:"):
        return gap
    return f"{gap}\n⚠️ INTEGRATE_DISPOSITION_FAILED: {recorded}"


# The verdict writer's failure facts (``write_patch_verdict(failures=...)``), as a decider reads them.
_UNRETAINED = {
    "verdict_artifact_write_failed": ("its verdict artifact was not written (verdict_artifact_write_failed), so that "
                                      "file keeps no record of it and an earlier verdict there, if any, still reads "
                                      "as before"),
    "custody_row_write_failed": ("its custody row did not land (custody_row_write_failed), so readers of the "
                                 "custody marks do not see it"),
}


def _candidate_drive_roots(ctx: ToolContext) -> List[pathlib.Path]:
    roots: List[pathlib.Path] = []
    seen = set()
    meta = getattr(ctx, "task_metadata", {})
    meta_budget = meta.get("budget_drive_root") if isinstance(meta, dict) else ""
    for raw in (
        getattr(ctx, "drive_root", None),
        getattr(ctx, "budget_drive_root", None),
        meta_budget,
    ):
        if not raw:
            continue
        key = str(raw)
        if key in seen:
            continue
        seen.add(key)
        roots.append(pathlib.Path(raw))
    return roots


def _locate_child_patch(
    ctx: ToolContext, child_task_id: str
) -> Union[str, Tuple[pathlib.Path, Dict[str, Any], Dict[str, Any]]]:
    roots, cancelled = _candidate_drive_roots(ctx), False
    for root in roots:
        try:
            from ouroboros.headless import current_workspace_manifest
            result = effective_task_result(root, load_task_result(root, child_task_id, strict=True) or {})
            manifest_path = current_workspace_manifest(root, child_task_id, result)
        except Exception as exc:
            return f"⚠️ INTEGRATE_RESULT_UNREADABLE: {child_task_id}: {type(exc).__name__}: {exc}."
        cancelled = cancelled or cancellation_blocks_child_result(result)
        if not manifest_path.exists():
            if result.get("native_capture_pending"):
                return manifest_path.parent / "workspace.patch", {"status": "failed", "note": result.get("artifact_error", "Capture unavailable")}, result
            if any(row.get("kind") == "workspace_patch_manifest" for row in result.get("artifacts", [])):
                return f"⚠️ INTEGRATE_MANIFEST_UNREADABLE: current capture is missing: {manifest_path}."
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"⚠️ INTEGRATE_MANIFEST_UNREADABLE: {manifest_path}: {type(exc).__name__}: {exc}."
        if not isinstance(manifest, dict):
            continue
        if result.get("native_capture_pending") or result.get("artifact_status") == "failed":
            manifest = {**manifest, "status": "failed", "note": result.get("artifact_error", "Native capture is not current.")}
        return manifest_path.parent / str(manifest.get("patch_name") or "workspace.patch"), manifest, result
    listed = ", ".join(str(r) for r in roots) or "(no drive roots resolved)"
    return (
        f"⚠️ INTEGRATE_PATCH_NOT_FOUND: no workspace_patch.json for child {child_task_id!r} under {listed}. "
        + ("It was cancelled without a retained capture and never finishes later, so nothing of it can be integrated."
           if cancelled else "Ensure the child finished and was a mutative subagent that returned a workspace patch "
           "(retrieve it with get_task_result/wait_task first).")
    )


def _sha256_file(path: pathlib.Path) -> str:
    from hashlib import sha256

    hasher = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Verdict writing lives in tools/patch_verdict.py (module-size ceiling);
# the historical private name stays importable for callers and tests.
from ouroboros.tools.patch_verdict import decision_reason_refusal, write_patch_verdict as _write_verdict  # noqa: E402
from ouroboros.tools.patch_staging import (  # noqa: E402
    index_entries, sparse_absent, stage_applied, staging_receipt, undo_applied,
)


def _child_write_root(child_result: Dict[str, Any]) -> str:
    constraint = child_result.get("task_constraint") if isinstance(child_result.get("task_constraint"), dict) else {}
    metadata = child_result.get("metadata") if isinstance(child_result.get("metadata"), dict) else {}
    for value in (
        constraint.get("write_root"),
        child_result.get("workspace_root"),
        metadata.get("workspace_root"),
        child_result.get("write_root"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _read_binding(ctx: ToolContext, root: str, path: pathlib.Path, entry: bool) -> Any:
    """The ordinary read binding of PATH; an entry's comes from its folder, its name rejoined unfollowed."""
    from dataclasses import replace
    from ouroboros.tool_access import build_resolved_resource_binding

    binding = build_resolved_resource_binding(ctx, root=root, operation="read", path=str(path.parent if entry else path))
    return replace(binding, target_path=pathlib.Path(binding.target_path) / path.name) if entry else binding


def _granted_root_bindings(ctx: ToolContext, ceiling: Any, binding: Any, entry: bool = False) -> Iterator[Any]:
    """Other ceiling-granted roots whose ordinary binding reaches this same physical file.

    Overlapping roots (Deliverables inside user_files) give one file several ordinary
    labels; the deepest containing one must not discard a grant on another. Each
    candidate passes the ordinary resolver and its confinement; nothing is aliased.
    """
    physical = pathlib.Path(binding.target_path)
    physical = physical if entry else physical.resolve(strict=False)
    for root in dict.fromkeys(g.root for g in ceiling.resource_grants if "read" in g.operations):
        if root == binding.root:
            continue
        try:
            other = _read_binding(ctx, root, physical, entry)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if (pathlib.Path(other.target_path) if entry else pathlib.Path(other.target_path).resolve(strict=False)) == physical:
            yield other


def _shared_read_refusal(ctx: ToolContext, target: pathlib.Path, operation: str = "", *, entry: bool = False) -> str:
    """Use current read authority; selecting a child's folder adds no grant. An ``entry`` (a captured
    path's `entry_path`) is judged as itself: its folder takes the ordinary binding, and the Presence
    ceiling and task policy still see its exact name, which is never followed to a destination."""
    from ouroboros.tool_access import build_resolved_resource_binding
    from ouroboros.tools.tool_resolution import _root_containing_absolute_path
    from ouroboros.presence_authority import presence_ceiling_from_context, presence_ceiling_allows_binding
    from ouroboros.protected_artifacts import _artifact_records, _operation_denied, block_reason_for_path

    try:
        root = _root_containing_absolute_path(ctx, "read_file", str(target.parent if entry else target))
        binding = _read_binding(ctx, root or "user_files", target, entry)
        ceiling = presence_ceiling_from_context(ctx)
        if ceiling is not None and not presence_ceiling_allows_binding(ceiling, binding, entry=entry) and not any(
                presence_ceiling_allows_binding(ceiling, other, entry=entry)
                for other in _granted_root_bindings(ctx, ceiling, binding, entry)):
            return "Presence resource ceiling does not allow reading this child target"
        if operation:
            # Only relative policy needs the caller's active root (including a
            # project room). B's read binding must never become the policy base.
            relative_policy = any(
                _operation_denied(record, operation)
                and any(str(path).strip() and not pathlib.Path(str(path).strip()).expanduser().is_absolute()
                        for path in record.get("paths") or [])
                for record in _artifact_records(ctx))
            policy_binding = (build_resolved_resource_binding(
                ctx, root="active_workspace", operation="read", path=".") if relative_policy else None)
            if refusal := block_reason_for_path(ctx, target, operation, policy_binding, entry=entry):
                return refusal
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return str(exc)
    return ""


def _shared_target(child_result: Dict[str, Any], manifest: Dict[str, Any], requested: str) -> pathlib.Path:
    """Canonical assignment selects B; present capture/assignment fields must agree.

    Historical captures may omit additive root/base fields. Neither an explicit
    target nor a manifest alone selects a destination. Shared live HEAD may move.
    """
    from ouroboros.workspace_copies import same_directory

    root = _child_write_root(child_result)
    if not root:
        raise ValueError("child result did not record write_root/workspace_root")
    target = pathlib.Path(root).expanduser().resolve(strict=False)
    constraint = child_result.get("task_constraint") if isinstance(child_result.get("task_constraint"), dict) else {}
    metadata = child_result.get("metadata") if isinstance(child_result.get("metadata"), dict) else {}
    for value in (constraint.get("write_root"), child_result.get("workspace_root"),
                  metadata.get("workspace_root"), child_result.get("write_root"),
                  manifest.get("workspace_root"), requested):
        if value and not same_directory(value, target):
            raise ValueError("recorded child target, capture workspace_root or explicit target_root disagree")
    base, captured = constraint.get("base_sha"), manifest.get("base_head")
    if base and captured and base != captured:
        raise ValueError("capture base does not match the child's admitted base")
    return target


def _verify_shared_external_workspace(
    target: pathlib.Path,
    patch_path: pathlib.Path,
    touched: List[str],
    file_rows: List[Dict[str, Any]] = (),
) -> tuple[bool, List[str], str]:
    invalid: List[str] = []
    resolved_target = target.resolve(strict=False)
    for rel in touched:
        text = str(rel or "").strip()
        try:  # confinement is the entry's folder: a link is compared as a link, never followed
            if text and not entry_path(target / text).is_relative_to(resolved_target):
                invalid.append(text)
        except (OSError, RuntimeError):
            continue  # an unexaminable folder: the caller names the path unknown; Git's reverse check decides
    if invalid:
        return False, invalid, ""
    if not (target / ".git").exists():
        return False, [], f"target {target} is not a git working tree"
    try:
        if not verify_file_outputs(file_rows, target):
            return False, [], "registered file outputs do not match the shared workspace"
    except (OSError, ValueError) as exc:
        return False, [], str(exc)
    if not patch_path.is_file() or not patch_path.stat().st_size:
        return (True, [], "") if file_rows else (False, [], "workspace patch and file outputs are absent")
    # Capture diffed B under the ambient Git configuration (CRLF conversion, filters);
    # the read-only reverse check must convert the same way, so it strips only the
    # variables that would redirect Git to another repository or index.
    proc = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=str(target),
        capture_output=True,
        text=True,
        env={**{k: v for k, v in os.environ.items() if k not in _GIT_RETARGET_ENV},
             "GIT_CEILING_DIRECTORIES": str(resolved_target.parent)},
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, [], detail[:600] or "reverse patch check failed"
    return True, [], ""


def _capped_self_repo_refusal(ctx: Any, child_task_id: str) -> str:
    """A tree under the light per-task cap (a consciousness Act/Observe tree) may not land a
    patch on the Ouroboros repository — protected paths or not, in every install mode. The
    empty string when the task is not capped."""
    from ouroboros.consciousness_authority import task_mode_capped_light

    if not task_mode_capped_light(getattr(ctx, "task_metadata", None)):
        return ""
    return (
        f"⚠️ INTEGRATE_CAPPED_TREE: child {child_task_id} produced a self_worktree patch (against "
        "the Ouroboros system repo), but this task's tree runs under a light cap (a consciousness "
        "Act/Observe tree): it may not land a patch on the Ouroboros repository, protected paths "
        "or not, in any runtime mode."
    )


def _integration_runtime_mode(ctx: Any) -> str:
    """The mode the protected-path gate of an integration reads: the stricter of the install
    mode and the task's own cap (a consciousness Act/Observe tree is light), so a capped task
    cannot land a system-repo patch the install mode alone would allow."""
    from ouroboros.consciousness_authority import effective_runtime_mode

    return effective_runtime_mode(get_runtime_mode(), getattr(ctx, "task_metadata", None))


def _patch_touched_paths(patch_path: pathlib.Path, target: pathlib.Path, env: Any = None) -> tuple[set[str], str]:
    """Every path the patch touches, parsed NUL-SAFELY from git's own reader.

    ``git apply --numstat -z`` is the machine-readable form: fields are
    NUL-terminated and pathnames are never munged (no quoting, no ``\\t``/``\\n``
    escapes). The previous text parse split on tabs and additionally regex-scanned
    ``diff --git a/… b/…`` headers, so a path containing a tab, a newline or a
    quote-triggering byte produced a corrupted pathspec — which then reached the
    protected-path gate and the staging step.

    Read in BOTH directions, because ``git apply --numstat`` names only the paths
    that direction WRITES: a rename reports its destination forward and its source
    in reverse, and the source is exactly the deletion the staging step must
    record. (This is where the regex scan earned its keep; the union of the two
    NUL-safe readings replaces it without the quoting bug.)
    """
    touched: set[str] = set()
    from ouroboros.tools.patch_staging import git_apply  # a work target reads it as the work domain (stdin)

    for direction in ([], ["-R"]):
        numstat = git_apply(target, patch_path, [*direction, "--numstat", "-z"], env=env)
        if numstat.returncode != 0:
            detail = (numstat.stderr or numstat.stdout or b"").decode("utf-8", errors="replace")
            return set(), detail.strip()[:600]
        tokens = (numstat.stdout or b"").split(b"\0")
        index = 0
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if not token:
                continue
            parts = token.split(b"\t", 2)
            if len(parts) < 3:
                continue
            path = parts[2]
            if path:
                touched.add(path.decode("utf-8", errors="surrogateescape"))
                continue
            # `git diff --numstat -z` spells a rename as an empty path field
            # followed by two more NUL-terminated fields (source, destination).
            for _ in range(2):
                if index < len(tokens):
                    extra = tokens[index]
                    index += 1
                    if extra:
                        touched.add(extra.decode("utf-8", errors="surrogateescape"))
    return {path for path in touched if path}, ""


def _baseline_drifted_paths(
    target: pathlib.Path, baseline_sha: str, touched: List[str], baseline_entries: Optional[Dict[str, str]] = None,
) -> tuple[List[str], str]:
    """Touched paths whose CURRENT target state differs from the run's baseline.

    A plain ``git apply`` relocates hunks by offset and ignores whole files whose
    context happens to still match, so "the target drifted since the snapshot"
    cannot be inferred from the apply's exit code — a moved target could be
    patched at a shifted position and silently accepted. Drift is therefore
    PROVEN first, against the baseline commit, with the same temp-index machinery
    the baseline itself was built with (identical filter/attribute treatment):
    seed a scratch index from the baseline tree, stage the current worktree state
    of exactly these paths into it, and ask git which entries now differ.

    Once the owner released a result's pins and Git collected the baseline,
    ``baseline_entries`` (the capture's recorded "mode blob" of each path B held)
    make the same proof without B: the current entries staged into an empty
    scratch index must equal them, and a path B lacked must still be absent.

    Returns ``(drifted_paths, error)``; a non-empty error means the comparison
    could not be made and the caller must refuse rather than apply blind.
    """
    if not touched:
        return [], ""
    if not str(baseline_sha or "").strip():
        return [], "the run's custody row carries no baseline commit"
    payload = b"\0".join(p.encode("utf-8", errors="surrogateescape") for p in touched) + b"\0"
    scratch = tempfile.mkdtemp(prefix="ouro_baseline_drift_")
    env = {**os.environ, "GIT_INDEX_FILE": str(pathlib.Path(scratch) / "index")}
    try:
        for args in (
            ["git", "read-tree", str(baseline_sha)],
            ["git", "update-index", "-z", "--add", "--remove", "--stdin"],
            ["git", "diff-index", "--cached", "--name-only", "-z", str(baseline_sha)],
        ):
            proc = subprocess.run(
                args, cwd=str(target), capture_output=True, env=env,
                input=payload if args[1] == "update-index" else None,
            )
            if proc.returncode != 0 and args[1] == "read-tree" and baseline_entries is not None:
                return _recorded_baseline_drift(target, env, payload, touched, baseline_entries)
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
                return [], f"{args[1]} failed: {detail.strip()[:300]}"
        drifted = sorted(
            chunk.decode("utf-8", errors="surrogateescape")
            for chunk in (proc.stdout or b"").split(b"\0") if chunk
        )
        return drifted, ""
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _recorded_baseline_drift(target: pathlib.Path, env: Dict[str, str], payload: bytes, touched: List[str],
                             entries: Dict[str, str]) -> tuple[List[str], str]:
    """The drift proof against recorded baseline entries. The scratch index is seeded with them first, as
    ``read-tree`` of B would be, so ``core.fileMode=false`` or ``core.symlinks=false`` keep B's modes."""
    seed = "".join(f"{entries[path]}\t{path}\0" for path in touched if path in entries)
    current: Dict[str, str] = {}
    for args, data in ((["update-index", "-z", "--index-info"], seed.encode("utf-8", "surrogateescape")),
                       (["update-index", "-z", "--add", "--remove", "--stdin"], payload),
                       (["ls-files", "--stage", "-z"], None)):
        proc = subprocess.run(["git", *args], cwd=str(target), capture_output=True, env=env, input=data)
        if proc.returncode != 0:
            return [], f"{args[0]} failed: {(proc.stderr or b'').decode('utf-8', errors='replace').strip()[:300]}"
    for line in proc.stdout.split(b"\0"):
        meta, _, path = line.partition(b"\t")
        if path:
            mode, blob, _stage = meta.decode("ascii").split()
            current[path.decode("utf-8", errors="surrogateescape")] = f"{mode} {blob}"
    return sorted(path for path in touched if current.get(path) != entries.get(path)), ""


def _drift_state(captured: Any, current: Any) -> str:
    """A captured path now: ``removed`` (captured present), ``recreated`` (captured absent)."""
    return ("unchanged" if captured == current else "recreated" if captured is None
            else "removed" if current is None else "changed")


def _entry_kind(target: pathlib.Path, rel: str) -> str:
    """REL under TARGET examined entry by entry and never followed: ``file``, ``dir``, ``link``, ``other``
    (a FIFO, socket or device), ``beyond_link`` (a folder above it is a link now and the path exists
    through it), ``absent``, or ``unknown`` when it cannot be examined (a loop, no permission)."""
    here, path = target, target / rel
    try:
        for part in rel.split("/")[:-1]:
            if stat.S_ISLNK((here := here / part).lstat().st_mode):
                path.lstat()  # through the link: absent, or unknown when it cannot be reached (a loop)
                return "beyond_link"
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return "absent"
    except OSError:
        return "unknown"
    return "file" if stat.S_ISREG(mode) else "dir" if stat.S_ISDIR(mode) else "link" if stat.S_ISLNK(mode) else "other"


def _git_entry_kind(target: pathlib.Path, rel: str, captured: str = "") -> str:
    """What staging REL meets now: ``absent``; an ``entry`` Git records (a file, a symlink, or a
    repository where the capture recorded a gitlink); ``other``, which no entry at REL can stand
    for (a directory, FIFO, socket or device, or whatever a symlinked parent reaches); or
    ``unknown`` when it cannot be examined."""
    kind = _entry_kind(target, rel)
    if kind in ("file", "link") or kind == "dir" and captured.startswith("160000 "):
        return "entry"
    return kind if kind in ("absent", "unknown") else "other"


def _captured_patch_states(target: pathlib.Path, patch_path: pathlib.Path, base: str, paths: List[str],
                           unhashed: Iterable[str] = ()) -> tuple[Dict[str, str], str]:
    """Each patch path's current Git representation against the captured one; bytes are not compared.

    The captured side is the recorded base with the retained patch applied, in a scratch index.
    The current side converts each file as the capture's own read did, under the target's
    attributes and config now: a path the target's index tracks is staged beside that index's
    entry and attribute files (``git diff``), any other in an empty index with physical modes
    (``git diff --no-index`` semantics), and a skip-worktree or assume-unchanged path is
    its index entry, which is all Git reads of it. A path no entry can stand for now is a type
    change; one whose hash task policy denies, or that cannot be examined or staged, is
    ``unknown`` (a batch Git refuses is halved until only such paths remain), and denied paths
    are never staged. New objects go to a scratch store that only borrows the target's, and no
    scratch index is split, so the target's files, index, HEAD and objects stay untouched.
    Returns ``(states, why_unknown)``.
    """
    if len(base) not in (40, 64) or any(ch not in "0123456789abcdef" for ch in base):
        return {}, "the capture records no Git base object, so its postimages cannot be rebuilt"
    env = {**{k: v for k, v in os.environ.items() if k not in _GIT_RETARGET_ENV},
           "GIT_CEILING_DIRECTORIES": str(target.parent)}
    wanted, scratch = set(paths), pathlib.Path(tempfile.mkdtemp(prefix="ouro_shared_findings_"))

    def git(*args: str, index: str = "", data: Optional[bytes] = None) -> bytes:
        # A split scratch index would write its shared half into the target's own .git.
        options = ["-c", "core.splitIndex=false"]
        if index == "untracked":  # no-index capture reads physical modes, not configurable index modes
            options += ["-c", "core.fileMode=true", "-c", "core.symlinks=true"]
        proc = subprocess.run(["git", *options, *args], cwd=str(target), capture_output=True,
                              env={**env, "GIT_INDEX_FILE": str(scratch / index)} if index else env, input=data)
        if proc.returncode != 0:
            raise OSError(f"{args[0]} failed: {(proc.stderr or b'').decode('utf-8', errors='replace').strip()[:300]}")
        return proc.stdout

    def entries(index: str = "", held: Optional[set[str]] = None) -> Dict[str, str]:
        """``mode blob`` of the wanted paths and attribute files in a scratch index (or the target's,
        whose skip-worktree and assume-unchanged paths go to HELD)."""
        found = {}
        for line in git("ls-files", "--stage", "-v", "-z", index=index).split(b"\0"):
            meta, _, raw = line.partition(b"\t")
            path = raw.decode("utf-8", errors="surrogateescape")
            if path in wanted or path.rpartition("/")[2] == ".gitattributes":
                tag, _, entry = meta.decode("ascii").partition(" ")
                found[path] = entry.rsplit(" ", 1)[0]
                if held is not None and (tag == "S" or tag.islower()):
                    held.add(path)
        return found

    def stage(index: str, group: List[str], failed: List[str]) -> Dict[str, str]:
        # Attribute files go last, so their own new state never changes how the others convert.
        order = sorted(group, key=lambda path: path.rpartition("/")[2] == ".gitattributes")
        try:
            git("update-index", "-z", "--add", "--remove", "--replace", "--info-only", "--stdin", index=index,
                data=b"".join(path.encode("utf-8", errors="surrogateescape") + b"\0" for path in order))
            current = entries(index)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if len(order) == 1:  # Git stages nothing of a refused batch; halves isolate what it refuses
                failed.append(str(exc))
                return {order[0]: "unknown"}
            return {**stage(index, order[:len(order) // 2], failed), **stage(index, order[len(order) // 2:], failed)}
        return {path: _drift_state(captured.get(path), current.get(path)) for path in group}

    try:
        try:
            store = str((target / git("rev-parse", "--git-path", "objects").decode().strip()).resolve())
            (scratch / "objects").mkdir()
            env.update(GIT_OBJECT_DIRECTORY=str(scratch / "objects"),
                       GIT_ALTERNATE_OBJECT_DIRECTORIES=(store if os.pathsep not in store
                                                         else json.dumps(store, ensure_ascii=False)))
            git("read-tree", base, index="captured")
            git("apply", "--cached", "--whitespace=nowarn", str(patch_path), index="captured")
            held: set[str] = set()
            captured, indexed = entries("captured"), entries(held=held)  # the target's index, as the capture read it
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return {}, f"Git could not rebuild the captured postimages or read the target's index: {exc}"
        states: Dict[str, str] = {}
        notes: List[str] = []
        denied = set(unhashed)
        kinds = {path: "denied" if path in denied else "held" if path in held else
                 _git_entry_kind(target, path, captured.get(path, "")) for path in paths}
        groups: Dict[str, set[str]] = {"tracked": set(), "untracked": set()}
        for path, kind in kinds.items():
            if kind == "entry":
                groups["tracked" if path in indexed else "untracked"].add(path)
            else:  # never staged: held by the index (sparse included, never a removal), absent, a type change
                states[path] = (_drift_state(captured.get(path), indexed.get(path) if kind == "held" else None)
                                if kind in ("held", "absent") else
                                ("recreated" if captured.get(path) is None else "changed") if kind == "other"
                                else "unknown")
        if count := sum(kind == "denied" for kind in kinds.values()):
            notes.append(f"task policy denies hashing {count} patch path(s), so none was hashed")
        if count := sum(kind == "unknown" for kind in kinds.values()):
            notes.append(f"{count} patch path(s) could not be examined")
        for name, group in groups.items():
            if not group:
                continue
            seed = ({path: entry for path, entry in indexed.items()
                     if path in group or path.rpartition("/")[2] == ".gitattributes"} if name == "tracked"
                    else {})
            failed: List[str] = []
            try:
                git("update-index", "-z", "--index-info", index=name, data="".join(
                    f"{entry}\t{path}\0" for path, entry in seed.items()).encode("utf-8", "surrogateescape"))
                states.update(stage(name, sorted(group), failed))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:  # the seed itself
                states.update(dict.fromkeys(group, "unknown"))
                failed = [str(exc)] * len(group)
            if failed:
                notes.append(f"Git could not stage {len(failed)} current {name} file(s): {failed[0]}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {path: states[path] for path in paths}, "; ".join(notes)


_FINDING_ORDER = ("changed", "removed", "recreated", "unknown", "unchanged")
_FINDING_BASIS = {"git": "patch paths in their current Git representation, under the target's attributes and "
                         "config now; bytes not compared", "exact": "file results byte-exact"}


def _findings_text(findings: List[Dict[str, str]], note: str = "") -> str:
    """Per-path states by kind for the decider, with their comparison basis; long groups name
    20 paths, the verdict all."""
    groups: Dict[str, List[str]] = {}
    for row in findings:
        groups.setdefault(row["state"], []).append(row["path"])
    parts = [f"{state} {len(paths)}: {', '.join(paths[:20])}"
             + (f" (+{len(paths) - 20} more, listed in the verdict)" if len(paths) > 20 else "")
             for state in _FINDING_ORDER if (paths := groups.get(state))]
    basis = [text for key, text in _FINDING_BASIS.items() if any(row.get("basis") == key for row in findings)]
    return (("; ".join(parts) or "no captured paths") + (f" (basis: {'; '.join(basis)})" if basis else "")
            + (f" (patch paths unknown: {note})" if note else ""))


def _accept_hint(child_task_id: str) -> str:
    return (f"\nAfter reviewing, decide: integrate_subagent_patch(task_id='{child_task_id}', decision='accept', "
            "reason='<your judgment>') records your acceptance of the result as it now stands beside this "
            "observation (never as verification); decision='reject' records a rejection.")


def _record_acceptance(ctx: ToolContext, child_task_id: str, reason: str, target: pathlib.Path,
                       manifest: Dict[str, Any], files: List[str], observed: str, conflicts: List[str],
                       observation: Dict[str, Any], lineage: Dict[str, str] | None, result_sha: str,
                       cancelled: Dict[str, Any] | None = None) -> str:
    """The parent's informed acceptance of a shared result: its judgment beside what the check saw.

    It is a decision (``accepted_shared_workspace``), never a verification: nothing is
    re-applied, transferred or reverted, and the observed outcome and findings stay recorded.
    """
    facts: List[str] = []
    verdict = _write_verdict(
        ctx, child_task_id, outcome="accepted_shared_workspace", reason=reason, files=files, manifest=manifest,
        applied=False, conflicts=conflicts, protected=[], target=str(target), lineage=lineage, child_result_sha256=result_sha,
        observation={"observed_outcome": observed, **observation}, failures=facts, cancelled=cancelled)
    warning = _record_integration_disposition(
        ctx, child_task_id, "integrated", reason, result_sha, lineage=lineage, verdict=verdict, unretained=facts)
    seen = f"{observed}: {'; '.join(conflicts)}" if conflicts else observed
    return (f"OK: accepted external_workspace child {child_task_id}'s result as it now stands in {target}, on your "
            f"judgment, not as verification. The check at this decision observed {seen}. Since capture: "
            f"{_findings_text(observation.get('path_findings') or [], observation.get('path_findings_note', ''))}. "
            "Nothing was re-applied, transferred, reverted or marked verified. "
            f"Verdict: {verdict or '(unwritten)'}.{_format_patch_exclusions(manifest)}{warning}")


def _target_is_system_repo(ctx: ToolContext) -> bool:
    """Whether an integration target is the OUROBOROS body (or a checkout of it).

    The protected-path policy (`BIBLE.md`, `.github/workflows/ci.yml`, `build.sh`,
    `ouroboros/contracts/…`) is about THIS repository's own invariants. A foreign
    project that happens to own files with those names is not covered by it, and
    gating there blocks ordinary work in an external workspace with advice about a
    runtime mode that has nothing to do with that project. The predicate mirrors
    the registry's own write gate: absent workspace means the live body; an
    isolated copy retains its admitted source identity. Legacy copies are
    interpreted as the body, as their original contract required.
    """
    from ouroboros.workspace_copies import copy_binding, source_is_system_repo
    from ouroboros.tools.tool_resolution import system_repo_dir_for

    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    if ((getattr(constraint, "surface", "") == "self_worktree"
         or getattr(ctx, "workspace_mode", "") == "self_worktree") and not copy_binding(ctx)):
        return True  # Legacy isolated records were always own-body copies.
    try:
        return source_is_system_repo(ctx.active_repo_dir(), system_repo_dir_for(ctx))
    except (AttributeError, OSError, ValueError, TypeError):
        return True


def _is_host_minted_projects_tree(path: pathlib.Path) -> bool:
    """True when ``path`` is a host-minted genesis/coop tree — i.e. inside the
    durable subagent-projects root. Owner-attached folders never live there, so this
    is the structural boundary for the coop no-op (and the checkpoint-commit).
    Child/body checkouts are not coop projects, even when an orphan result may
    return to one under the caller's existing write rights."""
    try:
        from ouroboros.config import get_subagent_projects_root
        from ouroboros.tool_access import path_is_relative_to

        return path_is_relative_to(path.resolve(), pathlib.Path(get_subagent_projects_root()).expanduser().resolve())
    except Exception:
        return False


def _registered_state(path: pathlib.Path, expected: Dict[str, Any]) -> str:
    """One registered regular file now, against its captured size and sha256; opened without following its
    name, so a link swapped in after the entry check is unknown, never read."""
    from ouroboros.artifacts import ArtifactIdentityError, stream_artifact_file

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        with os.fdopen(os.open(path, flags), "rb") as handle:
            stream_artifact_file(handle, expected=expected)
    except FileNotFoundError:
        return "removed"
    except ArtifactIdentityError:
        return "changed"
    except OSError:
        return "unknown"
    return "unchanged"


def _verify_directory_direct_result(
    ctx: ToolContext, child_task_id: str, reason: str, target: pathlib.Path,
    manifest: Dict[str, Any], artifact_dir: pathlib.Path, result_sha: str, lineage: Dict[str, str] | None = None,
    accepting: bool = False, cancelled: Dict[str, Any] | None = None,
) -> str:
    """Verify only registered postimages; direct effects are never replayed or rolled back.

    Admission (a ready capture, recorded identities, containment, current read authority)
    holds first; every registered postimage then reports its own state, so a mismatch names
    each path and an explicit acceptance records the parent's judgment beside them."""
    from ouroboros.artifacts import stream_artifact_file

    states: Dict[str, str] = {}
    try:
        if manifest.get("status") != "ready" or manifest.get("apply_state") != "already_applied":
            raise ValueError("direct folder capture is not ready")
        if pathlib.Path(str(manifest.get("workspace_root") or "")).resolve(strict=False) != target:
            raise ValueError("direct folder capture root does not match its recorded child target")
        outputs = manifest.get("registered_outputs")
        if not isinstance(outputs, list):
            raise ValueError("registered output records are unavailable")
        # A directory registers a member ledger (`<kind>_manifest`) plus its `<kind>`
        # zip package; every other record names one file. Each takes its own check.
        directories = {(str(item.get("kind") or "").removesuffix("_manifest"), str(item.get("source_path") or ""))
                       for item in outputs if str(item.get("kind") or "").endswith("_manifest")}
        for item in outputs:  # registration recorded each path resolved; now each is examined as its own entry
            rel = pathlib.Path(item["source_path"]).relative_to(target).as_posix()
            source, now = target / rel, _entry_kind(target, rel)
            artifact = pathlib.Path(item["path"]).resolve(strict=False)
            artifact.relative_to(artifact_dir.resolve(strict=False))
            if not item.get("sha256") or not isinstance(item.get("size"), int):
                raise ValueError("registered output is missing its captured identity")
            if now != "unknown" and (refusal := _shared_read_refusal(ctx, source, "read_bytes", entry=True)):
                raise ValueError(refusal)
            stream_artifact_file(artifact, expected=item)
            kind = str(item.get("kind") or "")
            directory = kind.endswith("_manifest") or (kind, str(item.get("source_path") or "")) in directories
            if now != ("dir" if directory else "file"):  # the registered kind no longer holds, or cannot be seen
                states[rel] = {"absent": "removed", "unknown": "unknown"}.get(now, "changed")
            elif kind.endswith("_manifest"):
                ledger = json.loads(artifact.read_text(encoding="utf-8"))
                if str(ledger.get("source_path") or "") != str(item["source_path"]):
                    raise ValueError("directory output ledger source does not match registration")
                for member in ledger["files"]:
                    name = pathlib.PurePosixPath(member["path"])
                    if (name.is_absolute() or ".." in name.parts or not member.get("sha256")
                            or not isinstance(member.get("size"), int)):
                        raise ValueError("directory member is missing its captured identity or leaves its folder")
                    member_rel = (pathlib.PurePosixPath(rel) / name).as_posix()
                    path, now = target / member_rel, _entry_kind(target, member_rel)
                    for operation in ("read_bytes", "hash") if now != "unknown" else ():
                        if refusal := _shared_read_refusal(ctx, path, operation, entry=True):
                            raise ValueError(refusal)
                    states[member_rel] = (_registered_state(path, member) if now == "file" else
                                          {"absent": "removed", "unknown": "unknown"}.get(now, "changed"))
            elif not directory:
                if refusal := _shared_read_refusal(ctx, source, "hash", entry=True):
                    raise ValueError(refusal)
                states[rel] = _registered_state(source, item)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        detail = f"Registered output verification failed: {exc}"
        verdict = _write_verdict(
            ctx, child_task_id, outcome="direct_output_mismatch", reason=f"{reason + '. ' if reason else ''}{detail}",
            files=sorted(states), manifest=manifest, applied=False, conflicts=[str(exc)], protected=[],
            target=str(target), lineage=lineage, child_result_sha256=result_sha, cancelled=cancelled)
        unaccepted = (" Acceptance was not recorded: these outputs cannot be observed under current integrity "
                      "and read authority." if accepting else "")
        return f"⚠️ INTEGRATE_DIRECTORY_OUTPUT_MISMATCH: {detail}. Verdict: {verdict or '(unwritten)'}.{unaccepted}"
    findings = [{"path": path, "state": state, "basis": "exact"} for path, state in sorted(states.items())]
    differs = [f"{row['path']}: {row['state']}" for row in findings if row["state"] != "unchanged"]
    outcome = ("direct_output_mismatch" if differs else
               "verified_registered_outputs" if states else "direct_result_observed")
    if accepting:
        return _record_acceptance(ctx, child_task_id, reason, target, manifest, sorted(states), outcome, differs,
                                  {"path_findings": findings}, lineage, result_sha, cancelled)
    detail = (f"Registered outputs differ from their captured postimages in {target}: {_findings_text(findings)}"
              if differs else f"Verified {len(states)} registered file postimage(s) in {target}. " if states else
              f"Recorded the parent's acceptance of the direct child result in {target}; "
              "no file postimages were verified. ")
    facts: List[str] = []
    verdict = _write_verdict(
        ctx, child_task_id, outcome=outcome, reason=f"{reason + '. ' if reason else ''}{detail}",
        files=sorted(states), manifest=manifest, applied=False, conflicts=differs, protected=[], target=str(target),
        lineage=lineage, child_result_sha256=result_sha, observation={"path_findings": findings}, failures=facts,
        cancelled=cancelled,
    )
    if differs:
        return (f"⚠️ INTEGRATE_DIRECTORY_OUTPUT_MISMATCH: {detail}. Verdict: {verdict or '(unwritten)'}. "
                f"Captured result retained; nothing was marked integrated.{_accept_hint(child_task_id)}")
    detail += "Other shell, GUI or external effects and the complete changed-file set remain unknown. No effects were re-applied, transferred to the parent, or rolled back."
    warning = _record_integration_disposition(
        ctx, child_task_id, "integrated", reason or detail, result_sha, lineage=lineage, verdict=verdict, unretained=facts)
    return f"OK: {detail} Verdict: {verdict or '(unwritten)'}.{warning}"


def _handle_external_workspace_integration(
    ctx: ToolContext,
    *,
    child_task_id: str,
    reason: str,
    requested_target: str,
    patch_path: pathlib.Path,
    manifest: Dict[str, Any],
    child_result: Dict[str, Any],
    touched: List[str],
    file_rows: List[Dict[str, Any]] = (),
    lineage: Dict[str, str] | None = None,
    accepting: bool = False,
    cancelled: Dict[str, Any] | None = None,
) -> str:
    result_sha = _child_result_sha256(child_result)
    def refused(code: str, outcome: str, detail: str, target: Any = "",
                files: List[str] = touched, conflicts: List[str] = ()) -> str:
        """An early refusal keeps its audit verdict and custody row; no disposition."""
        verdict = _write_verdict(
            ctx, child_task_id, outcome=outcome, reason=reason or detail, files=files, manifest=manifest,
            applied=False, conflicts=list(conflicts) or [detail], protected=[], target=str(target or ""),
            lineage=lineage, child_result_sha256=result_sha, cancelled=cancelled)
        from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
        text = (f"⚠️ {code}: {detail}. Verdict: {verdict or '(unwritten)'}. "
                "Captured result retained; nothing was transferred or marked integrated.")
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="INTEGRATION_BLOCKED", text=text, meta={"identifier": code}))

    def forbidden(refusal: str) -> str:
        return (f"{refusal.rstrip('. ')}. Verification reads the child's folder under your current read, Presence "
                "and task-policy authority; the assignment grants none. Inspect or reject the result")

    assigned = _child_write_root(child_result)
    try:
        target = _shared_target(child_result, manifest, requested_target)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if not assigned:  # Nothing names a target; never substitute the parent's folder.
            return refused("INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISSING", "shared_workspace_missing_target",
                           f"child {child_task_id} did not record its assigned write_root/workspace_root")
        return refused("INTEGRATE_EXTERNAL_WORKSPACE_TARGET_MISMATCH", "shared_workspace_target_mismatch",
                       f"{exc}; verification runs only at the recorded assignment {assigned}", target=assigned)
    if refusal := _shared_read_refusal(ctx, target):
        return refused("INTEGRATE_TARGET_FORBIDDEN", "shared_workspace_read_refused", forbidden(refusal), target)

    if manifest.get("capture_kind") == "directory_direct":
        return _verify_directory_direct_result(
            ctx, child_task_id, reason, target, manifest, patch_path.parent, result_sha, lineage=lineage, accepting=accepting,
            cancelled=cancelled,
        )
    # Parse all captured paths without an ancestor repo's subdirectory prefix. Git
    # ignores a ceiling equal to its cwd and never searches the ceiling itself, so the
    # artifact folder's parent stops discovery right above the artifact folder.
    from ouroboros.subagent_worktrees import isolated_git_env
    parse_env = {**isolated_git_env(), "GIT_CEILING_DIRECTORIES": str(patch_path.parent.resolve().parent)}
    patch_touched, parse_error = (_patch_touched_paths(patch_path, patch_path.parent, env=parse_env)
                                  if patch_path.is_file() and patch_path.stat().st_size else (set(), ""))
    if parse_error:
        return refused("INTEGRATE_PATCH_UNREADABLE", "shared_workspace_patch_unreadable",
                       f"cannot parse {child_task_id} workspace.patch: {parse_error[:300]}", target)
    authoritative_touched = sorted(patch_touched | {row["path"] for row in file_rows} or set(touched))
    hashed, paths = {row["path"] for row in file_rows}, {}
    for rel in authoritative_touched:  # entries: a symlink is checked as its link, never followed
        try:
            paths[rel] = entry_path(target / rel)
        except (OSError, RuntimeError):  # a looping folder above it: nothing reads it; it stays unknown
            continue
    if escaped := [rel for rel, path in paths.items() if not path.is_relative_to(target)]:
        return refused("INTEGRATE_EXTERNAL_WORKSPACE_MISSING", "shared_workspace_missing",
                       f"child {child_task_id} result names {len(escaped)} path(s) outside {target}; "
                       f"first 20: {escaped[:20]}; omitted: {max(0, len(escaped) - 20)}",
                       target, authoritative_touched, escaped)
    unhashed = set()
    for rel, path in paths.items():
        for operation in ("read_bytes", "hash"):
            if refusal := _shared_read_refusal(ctx, path, operation, entry=True):
                if operation == "hash" and rel not in hashed:
                    unhashed.add(rel)  # the reverse check still reads it; its diagnosis takes no digest
                    continue
                return refused("INTEGRATE_TARGET_FORBIDDEN", "shared_workspace_read_refused",
                               forbidden(refusal), target, authoritative_touched)
    verified, missing, detail = _verify_shared_external_workspace(
        target, patch_path, authoritative_touched, file_rows)
    # The capture is the whole shared tree's change since the child's base; the reverse
    # check decides, and each captured path's own state is what the decider reads.
    patch_paths = sorted(patch_touched)
    states, note = (_captured_patch_states(target, patch_path, str(manifest.get("base_ref") or ""), patch_paths,
                                           unhashed) if patch_paths else ({}, ""))
    exact = file_output_states(file_rows, target)
    findings = [{"path": rel, "state": exact.get(rel) or states.get(rel, "unknown"),
                 "basis": "exact" if rel in exact else "git"} for rel in authoritative_touched]
    observation = {"path_findings": findings, **({"path_findings_note": note} if note else {})}
    seen = f"Since capture: {_findings_text(findings, note)}"
    coop = _is_host_minted_projects_tree(target)
    outcome = ("coop_already_in_tree" if coop else "verified_shared_workspace") if verified else (
        "shared_workspace_missing" if missing else "shared_workspace_mismatch")
    conflicts = missing or ([detail] if detail else [])
    if accepting and not missing:
        return _record_acceptance(ctx, child_task_id, reason, target, manifest, authoritative_touched, outcome,
                                  conflicts, observation, lineage, result_sha, cancelled)
    facts: List[str] = []
    verdict = _write_verdict(
        ctx, child_task_id, outcome=outcome, reason=reason, files=authoritative_touched,
        manifest=manifest, applied=False, conflicts=conflicts, protected=[], target=str(target), lineage=lineage,
        child_result_sha256=result_sha, observation=observation, failures=facts, cancelled=cancelled)
    snapshot = (f"its captured snapshot of the shared tree ({len(authoritative_touched)} path(s): every change Git "
                "does not ignore since the child's base, which may include other writers' work; it names no author)")
    if verified:
        warning = _record_integration_disposition(
            ctx, child_task_id, "integrated",
            reason or "verified child result at its recorded shared target; no transfer to parent",
            result_sha,
            lineage=lineage, verdict=verdict, unretained=facts)
        prefix = (f"OK: cooperative no-op for child {child_task_id}: " if coop else
                  f"✅ Verified external_workspace child {child_task_id}: ")
        checkpoint = " The host attempts a best-effort checkpoint commit of this tree when the root task finalizes." if coop else ""
        # A passing reverse check does not prove whole-file equality or explain a representation change.
        tolerated = (" (the reverse check still passes despite these differences; it does not prove whole-file equality)"
                     if any(row["state"] in ("changed", "removed", "recreated") for row in findings) else "")
        return (f"{prefix}{snapshot} still verifies in {target}; only that delta was checked. {seen}{tolerated}. "
                f"No patch was re-applied or transferred to the parent's folder.{checkpoint} "
                f"Verdict: {verdict or '(unwritten)'}.{_format_patch_exclusions(manifest)}{warning}")
    if missing:
        return (f"⚠️ INTEGRATE_EXTERNAL_WORKSPACE_MISSING: child {child_task_id} result does not verify in {target}: "
                f"{missing}. Verdict: {verdict or '(unwritten)'}. Captured result retained.")
    return (f"⚠️ INTEGRATE_EXTERNAL_WORKSPACE_MISMATCH: child {child_task_id}: {snapshot} does not verify in "
            f"{target}: {detail}. {seen}. These checks establish neither authorship nor historical byte equality. "
            f"Verdict: {verdict or '(unwritten)'}. Captured result retained; nothing was applied, reverted or "
            f"marked integrated.{_format_patch_exclusions(manifest)}{_accept_hint(child_task_id)}")


# The verdict outcomes after which the ordinary path records a task-tree disposition
# (integrated or irrelevant) are decisions; conflicts, refusals and mismatches observe.
# An explicit acceptance is a decision whose verdict keeps the observation beside it.
_DECIDED_VERDICTS = frozenset({
    "applied", "rejected", "verified_shared_workspace", "coop_already_in_tree",
    "verified_registered_outputs", "direct_result_observed", "accepted_shared_workspace",
})


def _recorded_decision(drive: Any, task_ids: List[str], child_task_id: str, result_sha: str,
                       *, retained: bool = False) -> Dict[str, Any]:
    """Find a decision on this result in the chain's canonical custody marks.

    New verdicts name the consumed result; genuinely legacy hashless records cover
    every revision. Unreadable is not "none recorded"; no bounded projection is read.
    Only the Continue gate also checks retained artifacts: it can refuse a new
    decision whose old row was lost, never pretend that row reached acceptance.
    """
    from ouroboros.delegate_custody_current import marks
    from ouroboros.tools.patch_verdict import retained_verdicts

    for task_id in task_ids:
        rows = list(marks(drive, task_id).get("patch_verdicts", []))
        if retained:
            rows.extend({**v, "task_id": task_id, "disposition": v.get("outcome")}
                        for v in retained_verdicts(drive, task_id, child_task_id))
        for row in rows:
            if (row.get("child_task_id") == child_task_id and row.get("disposition") in _DECIDED_VERDICTS
                    and row.get("child_result_sha256", result_sha) == result_sha):
                return row
    return {}


def _continued_lineage(
    ctx: ToolContext, child_task_id: str, child_parent: str, current: Dict[str, Any],
) -> Tuple[Dict[str, str], str]:
    """``(verdict lineage, refusal)`` for a child this task did not start.

    Explicit owner Continue transfers a product decision, not child control: a root
    created from the child's parent (the hash-verified
    ``continued_by`` chain, ``owner_continue.continuation_chain``) may apply/verify,
    accept as it stands (shared only) or reject that parent's FINISHED native
    self_worktree/external_workspace result, once: a decision anyone on the chain
    recorded on this exact result stands.
    Room, folder, metadata and automatic retries grant nothing; ``_is_own_child`` and
    the stop/resume/discard/override rights built on it are unchanged; the ordinary
    pipeline still checks the product, its target and this task's current authority.
    """
    from ouroboros.owner_continue import continuation_chain
    from ouroboros.task_status import (
        ARTIFACT_NONTERMINAL_STATUSES, SETTLED_STATUSES,
        task_has_live_queue_ownership,
    )
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.tools.join_ledger import (
        _child_disposition_lineage, _current_child_result_disposition,
    )
    from ouroboros.workspace_copies import copy_binding

    parent_tid = str(getattr(ctx, "task_id", "") or "").strip()
    drive = canonical_data_root(ctx)
    chain = continuation_chain(drive, child_parent, parent_tid) if child_parent else []
    if not chain:
        return {}, (
            f"⚠️ INTEGRATE_LINEAGE_FORBIDDEN: {child_task_id} is not a direct child of this task "
            f"(its parent is {child_parent or '(unknown)'!r}, not {parent_tid!r}). Top-only routing: "
            "integrate only your own immediate children; descendant patches bubble up one parent at a time."
        )
    constraint = current.get("task_constraint") if isinstance(current.get("task_constraint"), dict) else {}
    surface, status = str(constraint.get("surface") or ""), str(current.get("status") or "")
    if _child_disposition_lineage(current) != (child_parent, child_parent, child_task_id):
        problem = f"it is not a direct child of the predecessor root {child_parent}"
    elif surface not in ("self_worktree", "external_workspace"):
        problem = f"its {surface or 'unrecorded'} surface is not a native self_worktree/external_workspace result"
    elif surface == "self_worktree" and not (copy_binding(current) or current.get("workspace_copy")):
        problem = "its legacy self_worktree capture records no source to return to"
    elif (status not in SETTLED_STATUSES
          or str(current.get("artifact_status") or "") in ARTIFACT_NONTERMINAL_STATUSES
          or task_has_live_queue_ownership(drive, child_task_id)):
        problem = f"it is not provably finished (status {status or 'unknown'!r}; a worker may still own it)"
    elif (decided := _current_child_result_disposition(current)) in ("integrated", "irrelevant"):
        problem = f"{child_parent} already marked this exact result {decided}; inspect the target instead"
    else:
        result_sha = _child_result_sha256(current)
        try:
            prior = _recorded_decision(drive, chain, child_task_id, result_sha, retained=True)
            covered = ("this exact result" if prior.get("child_result_sha256") else
                       "this child's result without a result hash, which covers every revision")
            problem = (f"{prior.get('task_id') or 'an earlier task'} already recorded the decision "
                       f"{prior.get('disposition')!r} for {covered}; it stands, inspect the target "
                       "instead") if prior else ""
        except Exception as exc:
            problem = (f"its decision records cannot be read ({type(exc).__name__}), so an earlier decision "
                       "cannot be ruled out")
        if not problem:
            return {"decision_authority": "owner_continue", "original_parent_task_id": child_parent,
                    "child_result_sha256": result_sha}, ""
    return {}, (
        f"⚠️ INTEGRATE_LINEAGE_FORBIDDEN: {child_task_id} is a child of {child_parent}, from which the "
        f"owner's Continue created this task, but {problem}. Nothing was applied, verified or recorded."
    )


def _integrate_subagent_patch(
    ctx: ToolContext,
    task_id: str = "",
    decision: str = "apply",
    reason: str = "",
    target_root: str = "",
) -> str:
    child_task_id = str(task_id or "").strip()
    if not child_task_id:
        return "⚠️ TOOL_ARG_ERROR (integrate_subagent_patch): task_id is required (the child whose patch to integrate)."
    decision = str(decision or "apply").strip().lower()
    if decision not in {"apply", "reject", "accept"}:
        return "⚠️ TOOL_ARG_ERROR (integrate_subagent_patch): decision must be 'apply', 'reject' or 'accept'."
    if decision == "accept" and (refusal := decision_reason_refusal(
            reason, "decision='accept' needs reason, your judgment after reading the findings")):
        return refusal

    located = _locate_child_patch(ctx, child_task_id)
    if isinstance(located, str):
        return located
    patch_path, manifest, child_result = located
    touched = [str(p) for p in (manifest.get("tracked_changed") or [])]
    touched += [str(p) for p in (manifest.get("untracked_included") or [])]

    # Only immediate children or the owner's recorded Continue relation.
    parent_tid = str(getattr(ctx, "task_id", "") or "").strip()
    child_parent = str((child_result or {}).get("parent_task_id") or "").strip()
    if not parent_tid:
        return (
            "⚠️ INTEGRATE_LINEAGE_FORBIDDEN: this task has no task_id, so child lineage cannot be "
            "verified. Integration is only allowed from the task whose task_id is the child's parent."
        )
    lineage, refusal = ({}, "") if child_parent == parent_tid else _continued_lineage(ctx, child_task_id, child_parent, child_result)
    if refusal:
        return refusal
    # A cancelled child's retained capture is decided only explicitly, as it stands; it stays cancelled.
    cancelled = _cancelled_result_facts(child_result, manifest)
    if cancelled and decision != "reject" and (refusal := _cancelled_result_refusal(
            ctx, child_task_id, child_result, reason, _child_write_root(child_result))):
        return refusal
    note = _cancelled_result_note(child_task_id, cancelled) if cancelled and decision != "reject" else ""
    if child_result.get("native_capture_pending") and decision == "apply":
        from ouroboros.headless import refresh_native_capture
        from ouroboros.tool_access_paths import canonical_data_root
        try:
            if denied := _shared_read_refusal(ctx, pathlib.Path(_child_write_root(child_result)), "read_file"):
                return f"⚠️ INTEGRATE_TARGET_FORBIDDEN: {denied}"
            note = refresh_native_capture(canonical_data_root(ctx), child_task_id)
        except Exception as exc:
            note = f"Native capture unavailable: {type(exc).__name__}: {exc}"
        return f"⚠️ INTEGRATE_NATIVE_CAPTURE: {note} No parent apply was attempted; read the current result before deciding."
    # Genesis is a standalone deliverable, never a patch to the parent's body.
    child_surface = str(((child_result or {}).get("task_constraint") or {}).get("surface") or "")
    if child_surface == "genesis" and decision != "reject":
        return (
            f"⚠️ INTEGRATE_GENESIS_FORBIDDEN: {child_task_id} is a from-scratch (genesis) project; "
            "its deliverable is the project directory itself, not a patch for this repo. Do not integrate "
            "it into the live body — use the project at its write_root directly (or decision='reject' to "
            "record a verdict)."
        )
    if decision == "accept" and child_surface != "external_workspace":
        return (f"⚠️ INTEGRATE_ACCEPT_UNSUPPORTED: {child_task_id} is a {child_surface or 'legacy'} result; "
                "decision='accept' records judgment on a shared external_workspace result only, so use apply "
                "or reject. Nothing was recorded.")

    if decision == "reject":
        facts: List[str] = []
        result_sha = _child_result_sha256(child_result)
        verdict_path = _write_verdict(
            ctx, child_task_id, outcome="rejected", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=[], protected=[], lineage=lineage, failures=facts,
            child_result_sha256=result_sha, cancelled=cancelled,
        )
        disposition_warning = _record_integration_disposition(
            ctx, child_task_id, "irrelevant", reason or "rejected the child result after review", result_sha,
            lineage=lineage, verdict=verdict_path, unretained=facts,
        )
        direct_note = (" Direct effects remain in the folder; rejecting this result does not undo them."
                       if manifest.get("capture_kind") == "directory_direct" else "")
        return (
            f"🚫 Rejected subagent patch from {child_task_id} ({len(touched)} file(s) not applied).{direct_note} "
            f"Verdict: {verdict_path or '(unwritten)'}. Reason: {reason or '(none)'}."
            f"{_format_patch_exclusions(manifest)}{disposition_warning}"
        )

    if manifest.get("capture_kind") == "directory_direct":
        if child_surface != "external_workspace":
            return "⚠️ INTEGRATE_DIRECTORY_SURFACE_MISMATCH: direct folder results require external_workspace."
        return append_published_text(ctx, _handle_external_workspace_integration(
            ctx, child_task_id=child_task_id, reason=reason, requested_target=str(target_root or "").strip(),
            patch_path=patch_path, manifest=manifest, child_result=child_result, touched=touched, lineage=lineage,
            accepting=decision == "accept", cancelled=cancelled,
        ), note)

    status = str(manifest.get("status") or "")
    if status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        return (
            f"⚠️ INTEGRATE_NO_CHANGES: child {child_task_id} workspace patch status={status!r}; "
            "nothing to apply."
            f"{_format_patch_exclusions(manifest)}"
        )
    try:
        file_rows = file_output_changes(manifest, patch_path.parent)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        return f"⚠️ INTEGRATE_FILE_OUTPUTS_UNAVAILABLE: {exc}"
    has_patch = patch_path.is_file() and patch_path.stat().st_size > 0
    if not has_patch and (not file_rows or manifest.get("patch_size")):
        return f"⚠️ INTEGRATE_PATCH_MISSING: workspace.patch for {child_task_id} not found at {patch_path}."
    expected_digest = str(manifest.get("sha256") or "")
    if has_patch and expected_digest:
        actual_digest = _sha256_file(patch_path)
        if actual_digest != expected_digest:
            return (
                f"⚠️ INTEGRATE_PATCH_CORRUPT: sha256 mismatch for {child_task_id} "
                f"(manifest {expected_digest[:12]} != file {actual_digest[:12]}); refusing to apply."
            )
    touched = sorted(set(touched) | {row["path"] for row in file_rows})

    if child_surface == "external_workspace":
        return append_published_text(ctx, _handle_external_workspace_integration(
            ctx,
            child_task_id=child_task_id,
            reason=reason,
            requested_target=str(target_root or "").strip(),
            patch_path=patch_path,
            manifest=manifest,
            child_result=child_result,
            touched=touched, file_rows=file_rows, lineage=lineage,
            accepting=decision == "accept", cancelled=cancelled,
        ), note)
    return append_published_text(ctx, _apply_to_parent_repo(
        ctx, child_task_id, reason=reason, target_root=target_root, patch_path=patch_path, manifest=manifest,
        child_result=child_result, child_surface=child_surface, file_rows=file_rows, has_patch=has_patch,
        lineage=lineage, cancelled=cancelled), note)


def _apply_to_parent_repo(ctx: ToolContext, child_task_id: str, *, reason: str, target_root: str,
                          patch_path: pathlib.Path, manifest: Dict[str, Any], child_result: Dict[str, Any],
                          child_surface: str, file_rows: List[Dict[str, Any]], has_patch: bool,
                          lineage: Dict[str, str] | None, cancelled: Dict[str, Any] | None = None) -> str:
    """An admitted isolated (self_worktree copy) or legacy result, applied and staged into its
    recorded source or the active repo: binding, capped-tree and protected-path checks, then
    the locked apply. The parent stays the sole committer. ``cancelled`` (a cancelled child's
    facts) rides every verdict of this consumed result."""
    from ouroboros.workspace_copies import copy_binding, same_directory, source_is_system_repo, copy_apply_refusal
    from ouroboros.tools.tool_resolution import system_repo_dir_for

    child_copy = copy_binding(child_result) or child_result.get("workspace_copy") or {}
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    is_acting = bool(constraint and getattr(constraint, "mode", "") == ACTING_SUBAGENT_MODE)
    try:
        active_root = pathlib.Path(ctx.active_repo_dir()).resolve(strict=False)
    except Exception as exc:
        return f"⚠️ INTEGRATE_TARGET_ERROR: could not resolve active repo: {type(exc).__name__}: {exc}."
    requested_target = str(target_root or "").strip()
    target = pathlib.Path(child_copy["source_root"]).resolve() if child_copy.get("source_root") else active_root
    if requested_target and not same_directory(requested_target, target):
        return "⚠️ INTEGRATE_TARGET_FORBIDDEN: target_root must name this copy's recorded source; legacy patches target the active root."
    if not (target / ".git").exists():
        return f"⚠️ INTEGRATE_TARGET_NOT_GIT: target {target} is not a git working tree."
    result_sha = _child_result_sha256(child_result)

    if child_surface == "self_worktree":
        if child_copy:
            if (not same_directory(child_copy.get("source_root"), target)
                    or child_copy.get("baseline_sha") != manifest.get("base_head")
                    or child_copy != manifest.get("workspace_copy")):
                return "⚠️ INTEGRATE_COPY_BINDING_MISMATCH: the recorded copy source, baseline or capture does not match this parent."
        elif ctx.is_workspace_mode() and str(getattr(ctx, "workspace_mode", "")) != "self_worktree":
            return "⚠️ INTEGRATE_SELF_WORKTREE_UNDER_WORKSPACE: legacy system-repo patch cannot be applied to an external workspace."
        if source_is_system_repo(target, system_repo_dir_for(ctx)) or (not child_copy and _target_is_system_repo(ctx)):
            capped = _capped_self_repo_refusal(ctx, child_task_id)
            if capped:
                return capped

    runtime_mode = _integration_runtime_mode(ctx)
    # Derive the changed-path set from the PATCH ITSELF (not the child-controlled
    # manifest) for the protected-path gate: a child must not be able to hide a
    # protected edit by omitting it from the manifest (sha256 verifies bytes only).
    patch_touched, parse_error = _patch_touched_paths(patch_path, target) if has_patch else (set(), "")
    if parse_error:
        return (
            f"⚠️ INTEGRATE_PATCH_UNREADABLE: cannot parse {child_task_id} workspace.patch for the "
            f"protected-path check (git apply --numstat failed): {parse_error[:300]}"
        )
    touched = sorted(patch_touched | {row["path"] for row in file_rows})
    target_is_body = source_is_system_repo(target, system_repo_dir_for(ctx)) if child_copy else _target_is_system_repo(ctx)
    protected = protected_paths_in(touched) if target_is_body else []
    if protected:
        grant_ok = (not is_acting) or bool(getattr(constraint, "protected_paths_grant", False))
        if not (mode_allows_protected_write(runtime_mode) and grant_ok):
            _write_verdict(
                ctx, child_task_id, outcome="blocked_protected", reason=reason, files=touched,
                manifest=manifest, applied=False, conflicts=[], protected=[p.path for p in protected],
                target=str(target), lineage=lineage, child_result_sha256=result_sha, cancelled=cancelled,
            )
            return protected_apply_refusal(
                path=protected[0].path,
                runtime_mode=runtime_mode,
                action=f"integrate subagent patch {child_task_id} touching", grant_ok=grant_ok,
            )

    unindexed = []
    if child_copy:
        if refusal := copy_apply_refusal(ctx, target, touched):
            return f"⚠️ INTEGRATE_TARGET_FORBIDDEN: {refusal}"
        try:
            outcome = _locked_apply(
                ctx, target, patch_path, touched, child_copy["baseline_sha"],
                file_changes=file_rows, file_baseline=child_copy.get("file_baseline", {}),
                three_way=bool(child_copy.get("source_index_clean") and child_copy.get("target_head") == child_copy["baseline_sha"]),
                admission_check=lambda: copy_apply_refusal(ctx, target, touched))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            return f"⚠️ INTEGRATE_APPLY_UNKNOWN: {type(exc).__name__}: {exc}. Inspect the target before another apply; the captured result is retained."
        if outcome.get("lock_error"):  # a busy target is no conflict (06.11): nothing applied, no verdict, result kept
            return f"⚠️ INTEGRATE_NOT_STARTED: {child_task_id} into {target}: the target is busy ({outcome['lock_error']}). Retry later."
        detail = outcome.get("admission_refusal") or outcome.get("drift_error")
        if outcome.get("drifted"):
            detail = "source files changed since copy: " + ", ".join(outcome["drifted"])
        proc, unindexed = outcome["proc"], outcome.get("unindexed") or []
        apply_attempted = proc is not None
        partial_applied = bool(outcome["staging_failure"] and not outcome["reverted"])
        if detail or outcome["staging_failure"] or proc is None:
            proc = subprocess.CompletedProcess([], 1, "", detail or outcome["staging_failure"] or "copy apply was not attempted")
    else:
        from ouroboros.tools.git import _acquire_git_lock, _release_git_lock
        try:
            _git_lock = _acquire_git_lock(ctx, repo_dir=target)
        except Exception as exc:
            return f"⚠️ INTEGRATE_NOT_STARTED: {child_task_id} into {target}: the target is busy ({type(exc).__name__}: {exc}). Retry later."
        partial_applied = False
        apply_attempted = False
        try:
            # Match --index semantics for file results too: never replace a parent's
            # staged preimage merely because its working copy matches the child base.
            index_tree = subprocess.run(
                ["git", "write-tree"], cwd=str(target), capture_output=True, text=True, check=True,
            ).stdout.strip() if file_rows else ""
            with prepare_file_outputs(file_rows, target, baseline_sha=index_tree) as prepared:
                indexed = index_entries(target, touched)  # the text patch's paths follow the same membership rule
                sparse, apply_attempted = sparse_absent(target, touched, indexed), has_patch
                proc = (subprocess.run(
                    ["git", "apply", "--3way", "--index", str(patch_path)],
                    cwd=str(target), capture_output=True, text=True,
                ) if has_patch else subprocess.CompletedProcess([], 0, "", ""))
                if proc.returncode == 0:
                    try:
                        if file_rows:
                            apply_attempted = True
                            prepared.apply()
                            if not prepared.verify_applied():
                                raise OSError("file outputs changed before staging")
                        failure, unindexed = stage_applied(target, touched, indexed)
                        if failure:
                            raise OSError(failure)
                    except Exception as exc:  # undo the whole attempt: text patch, file results, index entries
                        reverted, problems = undo_applied(
                            target, patch_path if has_patch else None, prepared, touched, indexed, sparse)
                        partial_applied = not reverted
                        detail = (f"rollback incomplete: {problems}" if problems else
                                  "the patch, file results and index entries were put back; nothing is half-applied")
                        proc = subprocess.CompletedProcess([], 1, "", f"{exc}; {detail}")
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
            proc = subprocess.CompletedProcess([], 1, "", str(exc))
        finally:
            _release_git_lock(_git_lock)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        conflicts = [ln.strip() for ln in stderr.splitlines() if "conflict" in ln.lower() or "patch failed" in ln.lower()]
        _write_verdict(
            ctx, child_task_id, outcome="partially_applied" if partial_applied else "conflict", reason=reason, files=touched,
            manifest=manifest, applied=partial_applied, conflicts=conflicts or [stderr[:500]],
            protected=[p.path for p in protected], target=str(target), lineage=lineage,
            child_result_sha256=result_sha, cancelled=cancelled,
        )
        detail = ("Integration did not finish cleanly." if apply_attempted else
                  "Result preparation failed. No file or patch apply was attempted.")
        next_step = ("Inspect with vcs_read diff and resolve, or run vcs_restore to abort, then retry or pick another child."
                     if apply_attempted else "The copy and patch are retained. Reconcile changed source files with this copy, or reject the result.")
        return (f"⚠️ INTEGRATE_CONFLICT: {child_task_id} into {target}: {detail} "
                f"Details: {stderr[:600]}\n{next_step}")

    try:
        invalidate_advisory_after_mutation(
            pathlib.Path(getattr(ctx, "drive_root", ".")),
            mutation_root=target,
            changed_paths=touched,
            source_tool="integrate_subagent_patch",
            mutating_task_id=str(getattr(ctx, "task_id", "") or ""),
        )
    except Exception:
        pass

    facts: List[str] = []
    from ouroboros.headless import record_native_received_files
    from ouroboros.tool_access_paths import canonical_data_root
    record_native_received_files(canonical_data_root(ctx), str(child_result.get("parent_task_id") or ""), target,
                                 unindexed,
                                 {"child_task_id": child_task_id, "manifest_path": str(patch_path.parent / "workspace_patch.json")})
    verdict_path = _write_verdict(
        ctx, child_task_id, outcome="applied", reason=reason, files=touched,
        manifest=manifest, applied=True, conflicts=[], protected=[p.path for p in protected],
        target=str(target), lineage=lineage, child_result_sha256=result_sha, failures=facts,
        observation={"unindexed": unindexed} if unindexed else None, cancelled=cancelled,
    )
    diffstat = str(manifest.get("diffstat") or "").strip()
    note = ""
    if protected:
        note = f" Includes {len(protected)} protected path(s) (allowed: runtime_mode={runtime_mode})."
    disposition_warning = _record_integration_disposition(
        ctx, child_task_id, "integrated", reason or "applied and staged the child result in the parent worktree",
        result_sha, lineage=lineage, verdict=verdict_path, unretained=facts,
    )
    staged, delivered, closing = staging_receipt(len(touched), unindexed)
    return (
        f"✅ Integrated subagent patch from {child_task_id} into {target} ({staged}).{note}{delivered}\n"
        f"{diffstat}{_format_patch_exclusions(manifest)}\n"
        f"Verdict: {verdict_path or '(unwritten)'}.\n"
        f"{closing} — review and run commit_reviewed yourself (you are the sole committer)."
        f"{disposition_warning}"
    )


# Per-candidate diff preview cap. Kept well under the tool's 80_000-char result
# limit (tool_capabilities.TOOL_RESULT_LIMITS) so several candidates fit side by
# side without the outer truncation hiding later candidates.
_COMPARE_PATCH_PREVIEW_CHARS = 12000


@completed_local_read
def _compare_subagent_patches(ctx: ToolContext, task_ids: Any = None) -> str:
    """Read-only best-of-N helper: show several children's returned patches side by
    side so the parent can synthesize LLM-first. Applies/commits nothing."""
    if isinstance(task_ids, str):
        ids = [task_ids.strip()] if task_ids.strip() else []
    else:
        ids = [str(t).strip() for t in (task_ids or []) if str(t).strip()]
    if not ids:
        return (
            "⚠️ TOOL_ARG_ERROR (compare_subagent_patches): task_ids must be a non-empty list of "
            "child subagent task_ids (the candidates to compare)."
        )
    parts: List[str] = [f"# Candidate comparison — {len(ids)} subagent patch(es)"]
    for cid in ids:
        located = _locate_child_patch(ctx, cid)
        if isinstance(located, str):
            parts.append(f"\n## {cid}\n{located}")
            continue
        patch_path, manifest, child_result = located
        status = str(manifest.get("status") or "")
        diffstat = str(manifest.get("diffstat") or "").strip()
        tracked = [str(p) for p in (manifest.get("tracked_changed") or [])]
        untracked = [str(p) for p in (manifest.get("untracked_included") or [])]
        result_status = str((child_result or {}).get("status") or "")
        result_summary = str((child_result or {}).get("result") or "").strip()
        if len(result_summary) > 600:
            result_summary = result_summary[:600] + " …"
        body = ""
        if status == ARTIFACT_STATUS_READY_WITH_CHANGES and patch_path.exists():
            try:
                raw = patch_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw = ""
            if len(raw) > _COMPARE_PATCH_PREVIEW_CHARS:
                body = raw[:_COMPARE_PATCH_PREVIEW_CHARS] + (
                    f"\n... [patch preview truncated; {len(raw)} bytes total — "
                    "integrate to apply/verify, or read the workspace.patch artifact for the full diff] ..."
                )
            else:
                body = raw
        parts.append(
            f"\n## {cid}\n"
            f"- patch status: {status or '(none)'} | child result status: {result_status or '(unknown)'}\n"
            f"- tracked changed: {len(tracked)} | untracked included: {len(untracked)}\n"
            f"- diffstat: {diffstat or '(none)'}{_format_patch_exclusions(manifest)}\n"
            + (f"- child summary: {result_summary}\n" if result_summary else "")
            + (f"- direct result: {manifest['note']}\n" if manifest.get("capture_kind") == "directory_direct" else "")
            + (f"- file results: {len(manifest.get('file_output_changes') or [])} captured change(s)\n"
               if manifest.get("file_output_changes") else "")
            + (f"\n```diff\n{body}\n```\n" if body else
               "- No inline patch body; inspect the recorded result and file artifacts.\n")
        )
    parts.append(
        "\nUse integrate_subagent_patch(task_id=...) to apply an isolated patch or verify shared files, "
        "or synthesize across candidates yourself (you are the sole committer). Comparison is read-only."
    )
    return "\n".join(parts)


from ouroboros.workspace_patch_rules import (  # noqa: E402 — patch-rule SSOT
    format_patch_exclusions as _format_patch_exclusions,
)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "compare_subagent_patches",
            {
                "name": "compare_subagent_patches",
                "description": (
                    "Read-only best-of-N helper: show several mutative children's returned "
                    "workspace.patch candidates side by side (status, diffstat, changed-file counts, "
                    "child summary, and a bounded diff preview) so you can pick the best one or "
                    "synthesize across them. Applies and commits NOTHING — use integrate_subagent_patch "
                    "to apply an isolated patch or verify shared files. Only sees patches reachable from your task drive roots."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Child subagent task_ids of the candidates to compare.",
                        },
                    },
                    "required": ["task_ids"],
                },
            },
            _compare_subagent_patches,
        ),
        ToolEntry(
            "integrate_subagent_patch",
            {
                "name": "integrate_subagent_patch",
                "description": (
                    "Integrate a mutative child's result or record a rejection: self_worktree uses "
                    "a source-bound apply/stage into its recorded source under your current write authority (clean copies support 3-way synthesis); native external_workspace "
                    "verifies files already in the shared tree WITHOUT reapplying. Its capture is every change in "
                    "that tree since the child's base, which may include other writers' work and names no author; "
                    "results give each captured path's state since capture, and a mismatch stays undecided until "
                    "you choose: decision='accept' with a reason records your acceptance of the result as it now "
                    "stands beside that observation (never verification), or reject. A cancelled child stays "
                    "cancelled, but its retained capture, once ready with nothing owed and no direct run of its own "
                    "still live in its shared folder, can be decided these ways with a reason; only that capture is judged. "
                    "Genesis is a standalone directory, not a repo patch. After the owner's Continue, the new root may "
                    "also decide the FINISHED native result of a direct child of its recorded predecessor the same ways: "
                    "apply/verify, accept a shared result as it stands, or reject (same checks; a "
                    "decision already recorded on that exact result stands; the verdict names the original "
                    "parent; old helpers stay outside its control). "
                    "This never commits; self-modification still requires your "
                    "commit_reviewed. For best-of-N pick a child "
                    "and integrate it, or integrate several to synthesize. Own-body protected-path changes require "
                    "pro runtime mode (and, for a nested acting parent, protected_paths_grant). Conflicts "
                    "are reported for you to resolve (vcs_read diff) or abort (vcs_restore). Writes a "
                    "subagent_patch_verdict_<task_id>.json audit artifact; an earlier one stays readable beside it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "The child subagent task_id whose workspace.patch to integrate."},
                        "decision": {"type": "string", "enum": ["apply", "reject", "accept"], "default": "apply", "description": "apply = apply/stage an isolated self_worktree patch (a new file the source's ignore rules exclude is delivered unstaged and named), or verify shared external_workspace files already written; reject = record a rejection without applying; accept = external_workspace only, after reading its findings: record that you keep the shared result as it now stands, beside what the check observed; nothing is applied, reverted or marked verified."},
                        "reason": {"type": "string", "description": "Rationale recorded in the verdict as given (why apply, accept, reject or synthesize); required for accept and for applying a cancelled child's capture: non-blank, at most 500 characters once each run of whitespace counts as one."},
                        "target_root": {"type": "string", "description": "Optional target assertion: external_workspace verifies at the child’s recorded folder under your current read authority, without transfer; an explicit target must equal that folder. Isolated copies return to their recorded source under current write authority; legacy patches target your active root."},
                    },
                    "required": ["task_id"],
                },
            },
            _integrate_subagent_patch,
        ),
        ToolEntry(
            "integrate_delegated_patch",
            {
                "name": "integrate_delegated_patch",
                "description": (
                    "EXPLICITLY apply or reject the captured result of ONE terminal delegated run. "
                    "Its starter or host-confirmed retry successor may decide it; terminal-owner "
                    "orphan recovery remains available to top-level tasks. Current target authority still applies. "
                    "Git runs edit a private execution snapshot; apply verifies its complete result "
                    "against the recorded baseline under the repository lock, applies and stages "
                    "changed files into the recorded target, and never commits. A terminal owner's orphan "
                    "may return to its original target inside the configured projects/worktrees roots, "
                    "including manually placed folders, under current write authority; no workspace is rebound. Skill-payload runs "
                    "apply LIVE apply into the non-Git payload with content-hash CAS (nothing is staged into your active root) and make its review "
                    "stale before reuse. Ordinary-folder copies use the engine's complete file "
                    "manifest and per-file baseline checks; paths may select files and remaining "
                    "results stay retained. Direct ordinary-folder work has already changed the "
                    "source: apply acknowledges those effects, reject cannot undo them, and no full "
                    "rollback is promised. Unapplied copies can be explicitly rejected; conflicts "
                    "retain their snapshot and results. A Git run whose private copy is gone with no usable capture "
                    "and no pinned or Git-held result can only be rejected: that reject records the result as lost, "
                    "keeps what a failed capture saved and applies nothing. A Git run's new .review-evidence/ files are retained "
                    "beside its capture, never applied; while its surviving copy holds named evidence that "
                    "could not be retained, apply and reject both refuse (INTEGRATE_DELEGATED_EVIDENCE_UNRETAINED) "
                    "and keep the copy. Read the captured result before deciding. A finished "
                    "run's capture still owed when you call is prepared by this call and returned for "
                    "inspection, never applied in the same call; one already ready is applied as before, "
                    "with no proof that you read it. "
                    "Undisposed snapshots or directory-copy results remain custody debt; direct runs "
                    "do not create that debt."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "The delegated run whose captured patch to integrate (from delegate_start)."},
                        "decision": {"type": "string", "enum": ["apply", "reject"], "default": "apply", "description": "apply = integrate the captured result: Git changes are STAGED into the recorded target (a new file its ignore rules exclude is delivered unstaged and named); skill changes are applied LIVE into the non-Git payload with content-hash CAS; directory copies use engine file delivery. reject = explicit discard of unapplied results, never an undo of direct effects."},
                        "paths": {"type": "array", "items": {"type": "string"}, "description": "For an engine directory copy, optionally select a nonempty list of captured file paths to apply; omit for the complete result. Remaining changes stay retained until applied or explicitly rejected. Git and skill captures are whole-result operations: omit paths or pass [] for the complete capture; nonempty selections are unsupported there."},
                        "reason": {"type": "string", "description": "Optional rationale recorded in the verdict and the durable disposition row."},
                        "acknowledge_ambiguous": {"type": "boolean", "default": False, "description": "Set true ONLY after inspecting an INTEGRATE_DELEGATED_APPLY_AMBIGUOUS state (a crashed apply left a durable unresolved intent): resolves that stale intent and re-runs the normal disposition guards, which re-verify the tree. A no-op when no ambiguity is pending."},
                    },
                    "required": ["run_id"],
                },
            },
            lambda ctx, run_id="", decision="apply", reason="", acknowledge_ambiguous=False, paths=None: _integrate_delegated_patch(
                ctx, run_id, decision, reason, acknowledge_ambiguous=bool(acknowledge_ambiguous), paths=paths),
        ),
    ]


# v7next F2 (D07): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.tools.subagent_integration_delegated import (  # noqa: E402, F401 -- intentional public re-exports
    _READY_CAPTURE_STATUSES,
    _capture_at_disposition,
    _capture_failed_refusal,
    _delegated_disposition_refusal,
    _dispose_delegated,
    _drift_refusal,
    _integrate_delegated_patch,
    _locked_apply,
    _manifest_capture_status,
    _resolve_acknowledged_intent,
    _unwritten_disposition_text,
)
