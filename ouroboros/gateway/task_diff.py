"""ONE owner-facing diff projection for every reviewable surface (no routes here).

`gateway/tasks.py` keeps the thin `GET /api/tasks/{task_id}/diff` route and
`gateway/project_threads.py` the thread-checkout one; every decision about WHAT
those routes answer is made in this module, which is pure computation over a task
result (or a thread's registered checkout) plus the repo on disk.

Three sources, ONE envelope and ONE git layer. A WORKSPACE task answers with its
durable patch bytes, a SELF-REPO task with a live projection over its attributed
paths, and a branched THREAD with everything its persistent checkout holds
against the commit it branched from (A13/X9). The third was added deliberately
here rather than beside the thread routes: a second git layer for a second
surface is how two answers to one question start disagreeing about what the owner
is looking at.

A WORKSPACE task already has a DURABLE `workspace.patch` artifact, so its diff is
those exact bytes. A SELF-REPO task has no historical patch, so its diff is an
honestly-labelled LIVE projection of the paths the mutation-attribution authority
attributed to the task window (ARCHITECTURE.md §5 "Mutation attribution"). Server-
side file stats and truncation are deliberately absent: the client parses the same
patch bytes it is shown (one snapshot = one truth), and an owner-facing diff is
never silently clipped — a patch too large to serve is REFUSED with a reason
instead.

Because the self-repo path shells out to git against the owner's real repository
while answering a browser read, the invocation discipline is part of the contract:

- every candidate path is a LITERAL (`GIT_LITERAL_PATHSPECS=1`), so a task-created
  file named `:(top)` or `:!secret` can never turn the pathspec into repo-wide
  magic that leaks the owner's own unattributed edits into the patch;
- the baseline commit is validated as a hex object name BEFORE it reaches argv, so
  a corrupted/hostile evidence record cannot smuggle an option like `--output=…`
  into the command line;
- rendering hooks stay off (`--no-ext-diff` / `--no-textconv`) so a repo-configured
  external differ cannot execute arbitrary host commands (BIBLE P3);
- git's stdout is decoded from BYTES with replacement, because a repo may legally
  contain latin-1 (or binary-ish) content that is not valid UTF-8, and a browser
  read must not 503 because of what a file happens to hold.

Its shared task-artifact resolver (declared-name lookup + directory containment)
lives here as the ONE copy; `tasks.py::api_task_artifact` imports it so the
traversal guard is written and reviewed exactly once.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import re
import subprocess
from typing import Any, Dict, List, NamedTuple, Optional

from ouroboros.config import get_task_diff_git_timeout_sec
from ouroboros.headless import (
    ARTIFACT_STATUS_FAILED,
    ARTIFACT_STATUS_FINALIZING,
    ARTIFACT_STATUS_PENDING,
    ARTIFACT_STATUS_READY_NO_CHANGES,
    task_artifacts_dir,
)
from ouroboros.task_status import FINAL_STATUSES, load_effective_task_result

DIFF_STATUS_PENDING = "pending"
DIFF_STATUS_READY = "ready"
DIFF_STATUS_EMPTY = "empty"
DIFF_STATUS_BLOCKED = "blocked"
DIFF_SOURCE_WORKSPACE = "workspace_patch"
DIFF_SOURCE_MUTATION_BASELINE = "mutation_baseline"
#: A branched THREAD's persistent checkout against the commit it branched from
#: (A13/X9) — no task, no artifact, no attributed path set: the whole tree.
DIFF_SOURCE_THREAD_CHECKOUT = "thread_checkout"

#: `git`'s own name for "the empty side" of a `--no-index` comparison. This is a GIT
#: PROTOCOL SENTINEL, not a filesystem path: git matches the exact string "/dev/null"
#: on every platform it runs on, Windows included. `os.devnull` would substitute "nul"
#: there and git would look for a FILE by that name, so the portable-looking spelling
#: is the one that actually breaks portability.
_GIT_NULL_SENTINEL = "/dev/null"
#: A baseline commit reaches git argv BEFORE `--`, so it is accepted only as a hex
#: object name. Anything else (an option-looking string, a ref expression, junk) is
#: refused as `base_commit_unknown` rather than handed to the command line.
_BASE_COMMIT_RE = re.compile(r"[0-9a-fA-F]{7,64}")
#: Owner-facing diffs are never clipped, so a patch beyond this is refused with a
#: reason. Generous on purpose: real review diffs are orders of magnitude smaller.
#: What the cap prevents is SERVING a repo-sized patch — JSON-encoding it into one
#: response body and handing the browser a string it must re-parse and render.
#: Materialization already happened in the subprocess buffer by the time we look.
DIFF_MAX_PATCH_BYTES = 8 * 1024 * 1024
#: A task that created thousands of files would otherwise mean thousands of git
#: subprocesses in ONE browser read. Past this the untracked projection is refused
#: as a whole (never truncated to a plausible-looking prefix).
DIFF_MAX_UNTRACKED_SECTIONS = 200
#: Concurrent diff reads allowed process-wide (the Changes screen and the inspector
#: can both ask at once, and each answer forks git).
DIFF_WORKER_SLOTS = 2

_diff_gate: Optional[asyncio.Semaphore] = None
_diff_gate_loop: Any = None


def diff_gate() -> asyncio.Semaphore:
    """The ONE process-wide admission gate for concurrent diff reads.

    An asyncio primitive binds to the first loop that ever WAITS on it, so a
    plain module-level ``Semaphore`` would raise "bound to a different event
    loop" in any process that runs more than one loop over its lifetime (the test
    suite, an in-process restart). The gate is therefore rebuilt when the running
    loop changes; within one gateway process that is a single construction and the
    cap is genuinely process-wide.
    """
    global _diff_gate, _diff_gate_loop
    loop = asyncio.get_running_loop()
    if _diff_gate is None or _diff_gate_loop is not loop:
        _diff_gate = asyncio.Semaphore(DIFF_WORKER_SLOTS)
        _diff_gate_loop = loop
    return _diff_gate


# --- shared task-artifact resolution ---------------------------------------

class ArtifactRefusal(NamedTuple):
    """Why ONE declared artifact could not be served, in both dialects.

    ``message``/``status`` answer an HTTP artifact request; ``reason`` is the
    machine token the diff endpoint reports as a typed blocker instead.
    """

    message: str
    status: int
    reason: str


def artifact_by_name(result: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for artifact in result.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("name") or pathlib.Path(str(artifact.get("path") or "")).name) == name:
            return artifact
    return None


def is_workspace_result(result: Dict[str, Any]) -> bool:
    return bool(str(result.get("workspace_root") or "").strip() or str(result.get("workspace_mode") or "").strip())


def resolve_task_artifact_path(
    drive_root: pathlib.Path,
    task_id: str,
    result: Dict[str, Any],
    name: str,
) -> tuple[Optional[pathlib.Path], Optional[ArtifactRefusal]]:
    """Resolve one DECLARED artifact name to a contained on-disk file.

    The ONE lookup + containment authority: an artifact is served only when the
    task result declares it, its metadata path's basename matches the requested
    name, and the resolved path stays inside this task's artifact directory.
    Shared by ``api_task_artifact`` and the diff endpoint so the traversal guard
    is written (and reviewed) exactly once.
    """
    artifact = artifact_by_name(result, name)
    if artifact is None:
        return None, ArtifactRefusal("artifact not found", 404, "artifact_not_declared")
    base = task_artifacts_dir(drive_root, task_id).resolve(strict=False)
    path = pathlib.Path(str(artifact.get("path") or "")).resolve(strict=False)
    if path.name != name:
        return None, ArtifactRefusal(
            "artifact metadata path does not match requested name", 500, "artifact_name_mismatch",
        )
    try:
        path.relative_to(base)
    except ValueError:
        return None, ArtifactRefusal(
            "artifact path is outside task artifact directory", 500, "artifact_outside_task_dir",
        )
    if not path.is_file():
        return None, ArtifactRefusal("artifact file is missing", 404, "artifact_file_missing")
    return path, None


# --- git plumbing -----------------------------------------------------------

def _git_capture(root: pathlib.Path, args: List[str]) -> tuple[int, str]:
    """Run one read-only git command under a timeout, returning (rc, stdout).

    Neither existing helper fits: ``utils.run_cmd`` RAISES on a non-zero exit
    (``git diff`` exits 1 whenever it found differences — the normal case here)
    and carries no timeout, while ``git_ops.git_capture`` is pinned to REPO_DIR
    and retries index repairs.

    Two env settings are load-bearing, not hygiene. ``GIT_LITERAL_PATHSPECS=1``
    makes every path argument a LITERAL: candidate paths come from task-attributed
    filenames, and a file named `:(top)` or `:!x` would otherwise be interpreted as
    pathspec MAGIC that silently widens the projection to the whole repo — putting
    the owner's own unattributed edits into a patch labelled as the task's work.
    ``LC_ALL/LANG=C`` keeps git's own messages parseable.

    stdout is decoded from BYTES with ``errors="replace"``: a repo may legally hold
    latin-1 or otherwise non-UTF-8 content, and an owner-facing read must degrade
    to a visible replacement character rather than fail the whole request.

    ``core.quotepath=off`` is set for EVERY invocation so one policy covers listing
    and diffing alike: git's default would render `héllo.txt` as the C-quoted
    `"h\\303\\251llo.txt"` — a path that does not exist when fed back to git, and a
    name no owner recognizes in a review surface.
    """
    env = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_LITERAL_PATHSPECS": "1",
    }
    try:
        proc = subprocess.run(
            ["git", "-c", "core.quotepath=off", *args],
            cwd=str(root),
            capture_output=True,
            env=env,
            timeout=get_task_diff_git_timeout_sec(),
        )
    except (subprocess.SubprocessError, OSError):
        return -1, ""
    return proc.returncode, (proc.stdout or b"").decode("utf-8", errors="replace")


def _git_head(root: pathlib.Path) -> str:
    rc, head = _git_capture(root, ["rev-parse", "HEAD"])
    return head.strip() if rc == 0 else ""


def _git_index_path(root: pathlib.Path) -> Optional[pathlib.Path]:
    """Where THIS worktree's index actually lives, asked of git itself.

    `<root>/.git/index` is only correct for a normal clone. In a LINKED worktree
    `.git` is a FILE pointing at `<main>/.git/worktrees/<name>`, so stat-ing
    `.git/index` raises `NotADirectoryError` and the index half of the fingerprint
    silently collapses to the same "absent" constant on every read — the guard
    looks present while catching nothing, which is exactly the stage/unstage race
    it exists to catch (BIBLE P1: a guard that cannot fire must not claim to).

    `git rev-parse --git-path index` is git's own answer and is worktree-correct;
    it answers a RELATIVE path in a plain clone and an absolute one in a linked
    worktree, so both forms are resolved against `root`. A failed resolution
    returns None and the caller keeps the honest absent marker.
    """
    rc, out = _git_capture(root, ["rev-parse", "--git-path", "index"])
    raw = out.strip()
    if rc != 0 or not raw:
        return None
    path = pathlib.Path(raw)
    return path if path.is_absolute() else (pathlib.Path(root) / path)


def _projection_fingerprint(
    root: pathlib.Path,
    candidates: List[str],
    index_path: Optional[pathlib.Path] = None,
) -> str:
    """Bind a patch read to the exact repo state it was computed from.

    HEAD plus a per-candidate stat is not enough on its own: `git add`/`git rm`
    moves work between the index and the worktree WITHOUT touching HEAD or any
    candidate's mtime, and that changes what `git diff <base> -- <path>` reports.
    This worktree's own index stat is therefore part of the fingerprint, resolved
    through `_git_index_path` so a LINKED worktree is covered too.

    `index_path` is the per-read cache: one diff answer takes up to five
    fingerprints, and the index location cannot move under them, so the caller
    resolves it once instead of forking `rev-parse` five times.
    """
    head = _git_head(root)
    rows = [head or "head_unavailable"]
    resolved = _git_index_path(root) if index_path is None else index_path
    index_row = "\x00index\x1fabsent"
    if resolved is not None:
        try:
            index_stat = resolved.stat()
            index_row = f"\x00index\x1f{index_stat.st_size}\x1f{index_stat.st_mtime_ns}"
        except OSError:
            index_row = "\x00index\x1fabsent"
    rows.append(index_row)
    for rel in candidates:
        try:
            stat = (root / rel).stat()
            rows.append(f"{rel}\x1f{stat.st_size}\x1f{stat.st_mtime_ns}")
        except OSError:
            rows.append(f"{rel}\x1fabsent")
    return hashlib.sha256("\x1e".join(rows).encode("utf-8")).hexdigest()


def _untracked_candidates(root: pathlib.Path, candidates: List[str]) -> tuple[List[str], List[str]]:
    """The attributed paths git does not track yet, plus any typed blockers.

    ``-z`` (NUL-separated, quoting suppressed) is the only listing form that
    survives a non-ASCII or space-bearing filename: the newline-separated default
    would hand back a C-quoted name that does not exist as a path, so the new file
    would silently vanish from an owner-facing diff. `_git_capture` additionally
    pins ``core.quotepath=off`` for every call.
    """
    rc, others = _git_capture(root, [
        "ls-files", "--others", "--exclude-standard", "-z", "--", *candidates,
    ])
    if rc != 0:
        return [], ["untracked_scan_failed"]
    return [item for item in others.split("\0") if item], []


def _build_projection_patch(
    root: pathlib.Path,
    base_commit: str,
    candidates: List[str],
) -> tuple[str, List[str]]:
    """`git diff <base> -- <candidates>` plus one --no-index hunk per new file."""
    blockers: List[str] = []
    rc, tracked = _git_capture(root, [
        "diff", "--no-ext-diff", "--no-textconv", "--no-color", base_commit, "--", *candidates,
    ])
    if rc not in (0, 1):
        return "", ["baseline_diff_failed"]
    sections = [tracked] if tracked else []
    # `git diff <base>` only knows paths git tracks; an attributed file created
    # during the task window is untracked and needs its own synthetic diff.
    untracked, scan_blockers = _untracked_candidates(root, candidates)
    blockers.extend(scan_blockers)
    if len(untracked) > DIFF_MAX_UNTRACKED_SECTIONS:
        # Refused as a WHOLE, not truncated: a prefix of N of M new files reads
        # exactly like a complete diff, and the owner would never know.
        return "".join(sections), blockers + ["untracked_projection_capped"]
    for rel in untracked:
        rc_new, added = _git_capture(root, [
            "diff", "--no-ext-diff", "--no-textconv", "--no-color", "--no-index",
            "--", _GIT_NULL_SENTINEL, rel,
        ])
        if added:
            sections.append(added)
        else:
            # A listed new file that produced NO diff text is an omission, never
            # "no change" — whether git failed or answered empty, say so.
            blockers.append("untracked_patch_unavailable")
    return "".join(sections), blockers


# --- envelope ---------------------------------------------------------------

def _diff_envelope(
    status: str,
    source: str,
    *,
    base_commit: str = "",
    head_advanced: bool = False,
    blockers: Optional[List[str]] = None,
    patch: str = "",
    patch_sha256: str = "",
) -> Dict[str, Any]:
    """Build the response, applying the ONE no-clipping rule for both sources.

    TWO reasons answer ``blocked`` with an EMPTY patch. They are the same rule
    seen from two sides — the owner is told a complete diff cannot be served,
    rather than shown an incomplete one they would review as complete:

    * ``patch_too_large`` — a patch over ``DIFF_MAX_PATCH_BYTES``;
    * ``untracked_projection_capped`` — more new files than
      ``DIFF_MAX_UNTRACKED_SECTIONS``, so EVERY new-file section was dropped
      rather than a prefix of them shown. That still leaves a patch behind: the
      TRACKED half, which renders exactly like a whole diff and says nothing
      about the files missing from it. Carried as a footnote beside a `ready`
      status it was a silent clip, which is the one thing this rule forbids.
    """
    body = patch
    reasons = list(blockers or [])
    digest = patch_sha256
    if "untracked_projection_capped" in reasons:
        status = DIFF_STATUS_BLOCKED
        body = ""
        digest = ""
    if len(body.encode("utf-8", errors="replace")) > DIFF_MAX_PATCH_BYTES:
        status = DIFF_STATUS_BLOCKED
        reasons.append("patch_too_large")
        body = ""
        digest = ""
    return {
        "status": status,
        "source": source,
        "base_commit": base_commit,
        "head_advanced": bool(head_advanced),
        "blockers": sorted(dict.fromkeys(str(item) for item in reasons if str(item))),
        "patch": body,
        "patch_sha256": digest,
    }


def _patch_digest(patch: str) -> str:
    """sha256 of the patch text EXACTLY as served (utf-8), on every source path.

    Digesting the on-disk bytes instead would disagree with the served string
    whenever a byte had to be replaced during decoding — an owner comparing the
    digest to what they received would be told the transfer was corrupt.
    """
    return hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else ""


# --- workspace source ------------------------------------------------------

def _workspace_diff_payload(
    drive_root: pathlib.Path,
    task_id: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Project the durable workspace patch artifact onto the diff contract."""
    manifest: Dict[str, Any] = {}
    manifest_path, _ = resolve_task_artifact_path(drive_root, task_id, result, "workspace_patch.json")
    if manifest_path is not None:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            manifest = {}
    artifact_status = str(result.get("artifact_status") or manifest.get("status") or "").lower()
    terminal = str(result.get("status") or "").lower() in FINAL_STATUSES
    base_commit = str(manifest.get("base_head") or "")
    current_head = str(manifest.get("current_head") or "")
    blockers = [
        str(row.get("type") or "workspace_artifact_error")
        for row in manifest.get("errors") or []
        if isinstance(row, dict)
    ]
    # The capture itself excluded paths (incidental lockfiles, oversized/sensitive
    # untracked files). The patch is complete for what it covers, so this is an
    # attribution note rather than a refusal — but it is never left unsaid.
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    excluded = 0
    for key in ("untracked_excluded", "tracked_excluded"):
        try:
            excluded += int(counts.get(key) or 0)
        except (TypeError, ValueError):
            continue
    if excluded > 0:
        blockers.append("workspace_paths_excluded")
    envelope = dict(
        source=DIFF_SOURCE_WORKSPACE,
        base_commit=base_commit,
        head_advanced=bool(base_commit and current_head and current_head != base_commit),
        blockers=blockers,
    )
    # A running (or finalizing) workspace task is PENDING, never empty: its patch
    # artifact does not exist yet, and answering "no changes" would be a lie.
    if artifact_status in {ARTIFACT_STATUS_PENDING, ARTIFACT_STATUS_FINALIZING}:
        return _diff_envelope(DIFF_STATUS_PENDING, **envelope)
    if not artifact_status and not terminal:
        return _diff_envelope(DIFF_STATUS_PENDING, **envelope)
    if artifact_status == ARTIFACT_STATUS_FAILED:
        envelope["blockers"] = blockers or ["workspace_artifact_failed"]
        return _diff_envelope(DIFF_STATUS_BLOCKED, **envelope)
    if artifact_status == ARTIFACT_STATUS_READY_NO_CHANGES:
        return _diff_envelope(DIFF_STATUS_EMPTY, **envelope)
    patch_path, refusal = resolve_task_artifact_path(drive_root, task_id, result, "workspace.patch")
    if patch_path is None:
        envelope["blockers"] = blockers + [refusal.reason if refusal else "workspace_patch_unavailable"]
        return _diff_envelope(DIFF_STATUS_BLOCKED, **envelope)
    try:
        raw = patch_path.read_bytes()
    except OSError:
        envelope["blockers"] = blockers + ["workspace_patch_unreadable"]
        return _diff_envelope(DIFF_STATUS_BLOCKED, **envelope)
    patch = raw.decode("utf-8", errors="replace")
    return _diff_envelope(
        DIFF_STATUS_READY if patch.strip() else DIFF_STATUS_EMPTY,
        patch=patch,
        patch_sha256=_patch_digest(patch),
        **envelope,
    )


# --- self-repo source ------------------------------------------------------

def _baseline_git_surface(evidence: Dict[str, Any], root: pathlib.Path) -> Dict[str, Any]:
    """The ONE baseline git surface whose canonical root is this repo."""
    baseline = evidence.get("baseline") if isinstance(evidence.get("baseline"), dict) else {}
    matching = [
        row for row in baseline.get("surfaces") or []
        if isinstance(row, dict)
        and str(row.get("canonical_root") or "") == str(root)
        and isinstance(row.get("git"), dict)
    ]
    return matching[0]["git"] if len(matching) == 1 else {}


def _terminal_candidate_row(evidence: Dict[str, Any], root: pathlib.Path) -> Optional[Dict[str, Any]]:
    """The persisted terminal projection row for this repo surface, if any."""
    snapshot = evidence.get("terminal_candidate_snapshot")
    if not isinstance(snapshot, dict):
        return None
    matching = [
        row for row in snapshot.get("surfaces") or []
        if isinstance(row, dict)
        and str(row.get("canonical_root") or "") == str(root)
        and str(row.get("surface_type") or "") == "system_repo"
    ]
    return matching[0] if len(matching) == 1 else None


def _self_repo_diff_payload(
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    task_id: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Project the live self-repo diff for one task's attributed paths.

    Per the owner-locked semantics: candidates come from the persisted
    ``terminal_candidate_snapshot`` for a terminal task and from the live
    ``attributed_git_candidates`` projection while the task runs — this endpoint
    never re-parses raw mutation evidence, and never claims exclusive ownership
    of a path. ``head_advanced`` discloses baseline drift as a boolean only, and it
    is computed at READ time (current HEAD vs the task baseline) because that is
    the state the patch is actually taken against — a recorded snapshot flag would
    describe a repo that has since moved on.
    """
    from ouroboros.mutation_attribution import attributed_git_candidates

    root = pathlib.Path(repo_dir).expanduser().resolve(strict=False)
    evidence = result.get("mutation_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    base_commit = str(_baseline_git_surface(evidence, root).get("base_commit") or "")
    terminal = str(result.get("status") or "").lower() in FINAL_STATUSES
    if terminal:
        row = _terminal_candidate_row(evidence, root)
        if row is None:
            blockers = ["terminal_snapshot_missing"] if evidence else ["baseline_missing"]
            return _diff_envelope(
                DIFF_STATUS_BLOCKED, DIFF_SOURCE_MUTATION_BASELINE,
                base_commit=base_commit, blockers=blockers,
            )
        candidates = [str(item) for item in row.get("candidates") or [] if str(item)]
        blockers = [str(item) for item in row.get("blockers") or [] if str(item)]
    else:
        projection = attributed_git_candidates(drive_root, task_id, root)
        candidates = [str(item) for item in projection.get("candidates") or [] if str(item)]
        blockers = [str(item) for item in projection.get("blockers") or [] if str(item)]
        base_commit = str(projection.get("base_commit") or base_commit)
    current_head = _git_head(root)
    envelope = dict(
        source=DIFF_SOURCE_MUTATION_BASELINE,
        base_commit=base_commit,
        head_advanced=bool(current_head and base_commit and current_head != base_commit),
        blockers=blockers,
    )
    if not candidates:
        # No candidates WITH blockers means the attribution authority could not
        # compute a set; without blockers it means nothing was attributed.
        return _diff_envelope(
            DIFF_STATUS_BLOCKED if blockers else DIFF_STATUS_EMPTY, **envelope,
        )
    if not _BASE_COMMIT_RE.fullmatch(base_commit):
        # A baseline that is not a hex object name never reaches argv: it would sit
        # BEFORE `--`, where git reads `--output=<path>` and friends as OPTIONS and
        # would happily write the patch over a file of the caller's choosing.
        envelope["blockers"] = blockers + ["base_commit_unknown"]
        return _diff_envelope(DIFF_STATUS_BLOCKED, **envelope)
    # Resolved ONCE for this read: the index location is a property of the
    # worktree, not of the moment, so the fingerprints below share it.
    index_path = _git_index_path(root)
    before = _projection_fingerprint(root, candidates, index_path)
    patch, patch_blockers = _build_projection_patch(root, base_commit, candidates)
    if before != _projection_fingerprint(root, candidates, index_path):
        # The repo moved under the read: retry ONCE, then refuse rather than
        # answer with a patch that does not belong to the disclosed baseline.
        before = _projection_fingerprint(root, candidates, index_path)
        patch, patch_blockers = _build_projection_patch(root, base_commit, candidates)
        if before != _projection_fingerprint(root, candidates, index_path):
            envelope["blockers"] = blockers + ["projection_changed_during_read"]
            return _diff_envelope(DIFF_STATUS_BLOCKED, **envelope)
    envelope["blockers"] = blockers + patch_blockers
    if "baseline_diff_failed" in patch_blockers:
        return _diff_envelope(DIFF_STATUS_BLOCKED, **envelope)
    return _diff_envelope(
        DIFF_STATUS_READY if patch.strip() else DIFF_STATUS_EMPTY,
        patch=patch,
        patch_sha256=_patch_digest(patch),
        **envelope,
    )


def task_diff_payload(
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    task_id: str,
) -> Optional[Dict[str, Any]]:
    """One task's diff projection, or None for an unknown task id (the only 404)."""
    result = load_effective_task_result(drive_root, task_id)
    if not result:
        return None
    if is_workspace_result(result):
        return _workspace_diff_payload(drive_root, task_id, result)
    return _self_repo_diff_payload(drive_root, repo_dir, task_id, result)


# --- thread-checkout source -------------------------------------------------

def _status_paths(status: str) -> List[str]:
    """Every path named by `git status --porcelain -z`, renames included.

    Parsed POSITIONALLY, because ``-z`` porcelain has a record structure and not
    a per-token shape. Each record is ``XY<space><path>`` NUL-terminated, and a
    rename or copy (``R``/``C`` in either status column) is TWO tokens: the record
    itself and then the ORIGIN path, bare, as its own NUL-terminated entry.

    Guessing per token — "does this look like it starts with a status code" —
    happened to work for most origin paths and mis-sliced the rest: a renamed-FROM
    path whose third character is a space (``ab cd.txt``) was read as a record and
    cut down to ``cd.txt``, so the fingerprint watched a file that does not exist
    and stopped watching one that does. Consuming the origin token as part of its
    record removes the guess entirely.
    """
    out: List[str] = []
    tokens = status.split("\0")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        if len(token) > 3 and token[2] == " ":
            out.append(token[3:])
            if token[0] in "RC" or token[1] in "RC":
                # The very next token is this record's origin path, whatever it
                # looks like. It is never a record of its own.
                while index < len(tokens) and not tokens[index]:
                    index += 1
                if index < len(tokens):
                    out.append(tokens[index])
                    index += 1
            continue
        # A token that is not a well-formed record is taken whole rather than
        # dropped: an unrecognised line must never make a path invisible.
        out.append(token)
    return out


def _checkout_fingerprint(root: pathlib.Path) -> str:
    """Bind a WHOLE-TREE read to the exact checkout state it was computed from.

    The task sources fingerprint a KNOWN list of candidate paths; a thread
    checkout has no such list, so the list is derived: git's own full status
    (untracked included) names every path that is not identical to HEAD, and each
    of those gets the same size/mtime row the task path uses. Status alone would
    not be enough — a file that stays `M` while its contents change again during
    the read keeps the same status code.

    The worktree's index stat is deliberately NOT part of this. `git status`
    REFRESHES the index, and a freshly created checkout is full of racily-clean
    files (written in the same clock tick as the index), so git rewrites the index
    on every single status call. Including its mtime made the fingerprint differ
    from itself on every read of a new checkout — a guard that fires constantly
    for no reason teaches the owner to ignore it. What the index would have caught
    — work moved between index and worktree — the status codes catch directly, and
    with the path's identity rather than a global timestamp.
    """
    rows = [_git_head(root) or "head_unavailable"]
    rc, status = _git_capture(root, ["status", "--porcelain", "-z", "--untracked-files=all"])
    if rc != 0:
        rows.append("\x00status\x1funavailable")
        return hashlib.sha256("\x1e".join(rows).encode("utf-8")).hexdigest()
    rows.append(status)
    for rel in _status_paths(status):
        try:
            stat = (root / rel).stat()
            rows.append(f"{rel}\x1f{stat.st_size}\x1f{stat.st_mtime_ns}")
        except OSError:
            rows.append(f"{rel}\x1fabsent")
    return hashlib.sha256("\x1e".join(rows).encode("utf-8")).hexdigest()


def thread_checkout_diff_payload(
    drive_root: pathlib.Path,
    project_id: str,
    thread_id: Any,
) -> Dict[str, Any]:
    """What a BRANCHED thread's own checkout currently holds (A13/X9).

    Changes is task-centric, and the per-task route cannot answer this: a thread
    worktree is a PERSISTENT checkout with no task, no artifact and no attributed
    path set, so its diff is everything that checkout holds against the commit it
    branched from — committed work, staged work, unsaved edits and new files
    alike. That is what the owner sees when they open the folder, so it is what
    the Changes screen must show for it.

    Deliberately built from the SAME hardened invocation and the SAME envelope as
    the task sources: literal pathspecs, no external differ, byte-decoded stdout,
    the no-clipping rule, and the read-repeat guard that refuses rather than serve
    a patch that does not belong to the disclosed baseline. Forking a second git
    layer for a second surface is how two answers to one question start disagreeing.

    A thread that is not branched off is NOT an error: it works in the project
    folder, so it reports the typed ``thread_not_branched`` blocker and the client
    tells the owner where its work actually lives.

    ``branch`` rides every answer, including the refusals. The header shows
    "thread · branch" and the client learns the branch HERE rather than requiring
    whoever opened the screen to already know it — which was the documented
    intent, while the payload did not actually carry the field (T3R-12).
    """
    from ouroboros.thread_worktrees import get_thread_worktree

    envelope: Dict[str, Any] = dict(source=DIFF_SOURCE_THREAD_CHECKOUT)
    branch = ""

    def _answer(status: str, **kwargs: Any) -> Dict[str, Any]:
        return {**_diff_envelope(status, **kwargs), "branch": branch}

    row = get_thread_worktree(drive_root, project_id, thread_id)
    if not row:
        return _answer(DIFF_STATUS_BLOCKED, blockers=["thread_not_branched"], **envelope)
    branch = str(row.get("branch") or "")
    root = pathlib.Path(str(row.get("path") or ""))
    base_commit = str(row.get("base_sha") or "")
    envelope["base_commit"] = base_commit
    if not root.is_dir():
        return _answer(DIFF_STATUS_BLOCKED, blockers=["checkout_missing"], **envelope)
    if not _BASE_COMMIT_RE.fullmatch(base_commit):
        # Same rule as the self-repo path: a baseline that is not a hex object
        # name never reaches argv, where git would read it as an OPTION.
        return _answer(DIFF_STATUS_BLOCKED, blockers=["base_commit_unknown"], **envelope)
    current_head = _git_head(root)
    # `head_advanced` here means the thread has committed work of its own — the
    # honest reading of "HEAD moved off the disclosed baseline" for a checkout
    # whose whole purpose is to move.
    envelope["head_advanced"] = bool(current_head and current_head != base_commit)
    before = _checkout_fingerprint(root)
    # No candidate list: a checkout's diff is the WHOLE tree against its base.
    patch, blockers = _build_projection_patch(root, base_commit, [])
    if before != _checkout_fingerprint(root):
        # The checkout moved under the read: retry ONCE, then refuse rather than
        # answer with a patch that belongs to two different states.
        before = _checkout_fingerprint(root)
        patch, blockers = _build_projection_patch(root, base_commit, [])
        if before != _checkout_fingerprint(root):
            return _answer(
                DIFF_STATUS_BLOCKED, blockers=["projection_changed_during_read"], **envelope,
            )
    envelope["blockers"] = blockers
    if "baseline_diff_failed" in blockers:
        return _answer(DIFF_STATUS_BLOCKED, **envelope)
    return _answer(
        DIFF_STATUS_READY if patch.strip() else DIFF_STATUS_EMPTY,
        patch=patch,
        patch_sha256=_patch_digest(patch),
        **envelope,
    )


__all__ = [
    "ArtifactRefusal",
    "DIFF_MAX_PATCH_BYTES",
    "DIFF_MAX_UNTRACKED_SECTIONS",
    "DIFF_SOURCE_MUTATION_BASELINE",
    "DIFF_SOURCE_THREAD_CHECKOUT",
    "DIFF_SOURCE_WORKSPACE",
    "DIFF_STATUS_BLOCKED",
    "DIFF_STATUS_EMPTY",
    "DIFF_STATUS_PENDING",
    "DIFF_STATUS_READY",
    "DIFF_WORKER_SLOTS",
    "artifact_by_name",
    "diff_gate",
    "is_workspace_result",
    "resolve_task_artifact_path",
    "task_diff_payload",
    "thread_checkout_diff_payload",
]
