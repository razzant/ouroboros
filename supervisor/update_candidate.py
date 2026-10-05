"""Managed-update CANDIDATE primitives: private-index tree serialization, the
pinned mechanical-merge baseline's merge flags, the deterministic failed-update
preservation branch, and the single-run tests-evidence contract (Q10).

Split out of ``supervisor.update_merge`` by the module-size discipline; the
import direction is one-way (``update_merge`` imports THIS module and re-exports
the names its callers and tests already use). Functions that need the update-tx
marker import ``update_merge`` lazily at call time — never at module level.
Depends on ``git_ops`` via the module object (``_g.X``) so monkeypatched
``REPO_DIR``/``DRIVE_ROOT`` are followed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from supervisor import git_ops as _g

_MERGE_NEUTRAL_FLAGS = ("-c", "rerere.enabled=false")
"""Every mechanical merge in the update flow disables rerere explicitly: the live
deployment may carry ``rerere.enabled=true`` plus a populated rr-cache in the COMMON
git dir (shared by linked worktrees), and a remembered resolution replayed into one
invocation but not another would make the plan merge, the materialized merge, and
the pinned M0 baseline silently disagree. Merge drivers from .gitattributes are
deliberately NOT neutralized — they are part of the repository's own contract and
apply identically to every invocation."""


_GIT_RUN_TIMEOUT_SEC = 300.0
"""Generous wall-clock bound for one update-plumbing git invocation. This plumbing
runs WHILE THE UPDATE LOCK IS HELD: a git process hung on a clean filter, a merge
driver, a stale repo lock or a blocked filesystem would otherwise wedge the whole
update flow with no rollback path. A timeout surfaces through the normal nonzero-rc
shape (``git_ops.FETCH_TIMEOUT_RC`` + a disclosed message) that every caller already
routes into its typed update error, after the process TREE is killed
(cross-platform: ``platform_layer.kill_process_tree``)."""


def _git_run(
    cmd: List[str], *, cwd: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    timeout: float = _GIT_RUN_TIMEOUT_SEC,
) -> Tuple[int, str, str]:
    """Run a git command with an optional cwd / extra env (e.g. GIT_INDEX_FILE), WITHOUT
    the REPO_DIR pin and index-repair retry of ``git_capture``. For merge-planning in a
    temp index / temp worktree only — never the live-repo control path. BOUNDED: reuses
    ``git_ops._run_git_process_bounded`` (timeout + process-group kill), so a hung git
    can never wedge the update lock; a timeout returns ``(FETCH_TIMEOUT_RC, "", msg)``."""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    rc, out, err = _g._run_git_process_bounded(
        cmd, timeout=timeout, cwd=cwd or _g.REPO_DIR, env=env, text=True,
    )
    return rc, str(out or "").strip(), str(err or "").strip()


def _rev_parse(ref: str) -> str:
    rc, out, _e = _g.git_capture(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"])
    return out if rc == 0 else ""


def _merge_head_sha() -> str:
    rc, out, _e = _g.git_capture(["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"])
    return out if rc == 0 else ""


def quarantine_corrupt_update_tx_marker() -> Dict[str, Any]:
    """Move an unparseable update-tx marker aside byte-intact.

    A corrupt marker used to be LEFT in place, which latched
    ``repo_writer_admission_closed`` ("managed_update_tx:corrupt") for the whole
    install with no recovery path (#447). Renaming it to ``*.corrupt-<ts>``
    keeps the corruption evidence for the owner while releasing the admission
    latch. A visible in-flight merge (MERGE_HEAD) is the one sign the marker may
    still cover a genuinely active transaction — that case stays fail-closed."""
    import time

    from supervisor import update_merge as _m

    if _merge_head_sha():
        _m._log_supervisor({"type": "managed_update_tx_corrupt_on_boot",
                            "merge_in_progress": True})
        return {"finalized": False,
                "reason": "corrupt tx marker over an in-flight merge — left for owner"}
    marker = _m._update_tx_marker_path()
    try:
        # "corrupt" conflates unreadable with unparseable (read_update_tx_strict:
        # callers MUST fail closed). Only a marker we can actually READ is proven
        # garbage; an unreadable one (permissions, transient I/O) may still be a
        # valid live transaction — leave it in place, admission stays closed.
        marker.read_bytes()
    except OSError:
        _m._log_supervisor({"type": "managed_update_tx_corrupt_on_boot",
                            "unreadable": True})
        return {"finalized": False,
                "reason": "unreadable tx marker — left for owner (fail closed)"}
    quarantine = marker.with_name(f"{marker.name}.corrupt-{int(time.time())}")
    # POSIX rename REPLACES an existing destination and the timestamp has
    # one-second resolution: two quarantines in the same second would silently
    # overwrite the first piece of evidence. Uniquify instead.
    seq = 0
    while quarantine.exists():
        seq += 1
        quarantine = marker.with_name(f"{marker.name}.corrupt-{int(time.time())}.{seq}")
    try:
        marker.rename(quarantine)
    except OSError:
        _m._log_supervisor({"type": "managed_update_tx_corrupt_on_boot"})
        return {"finalized": False,
                "reason": "corrupt tx marker — quarantine rename failed"}
    _m._log_supervisor({"type": "managed_update_tx_corrupt_quarantined",
                        "quarantine": str(quarantine)})
    return {"finalized": False, "reason": "corrupt tx marker quarantined",
            "quarantine": str(quarantine)}


def worktree_snapshot_tree(base_ref: str, *, cwd: Optional[str] = None) -> Tuple[str, str]:
    """Serialize the LIVE worktree (tracked edits, deletions, untracked files —
    conflict markers as plain file content) into a tree object via a PRIVATE
    index. The repository's real index (which may hold unmerged conflict stages
    that ``git write-tree`` refuses to serialize) and the worktree itself are
    never touched. Returns ``(tree_sha, error)``."""
    import tempfile

    fd, tmp_index_path = tempfile.mkstemp(prefix="ouro-snapshot-index-")
    os.close(fd)
    # `git read-tree` wants a NON-existent index path.
    os.unlink(tmp_index_path)
    env = {"GIT_INDEX_FILE": tmp_index_path}
    try:
        rc_r, _ro, read_error = _git_run(["git", "read-tree", base_ref], cwd=cwd, extra_env=env)
        if rc_r != 0:
            return "", read_error or f"read-tree {base_ref[:12]} failed"
        rc_a, _ao, add_error = _git_run(["git", "add", "-A"], cwd=cwd, extra_env=env)
        if rc_a != 0:
            return "", add_error or "git add -A failed"
        rc_w, tree, write_error = _git_run(["git", "write-tree"], cwd=cwd, extra_env=env)
        if rc_w != 0 or not tree:
            return "", write_error or "git write-tree failed"
        return tree, ""
    finally:
        try:
            os.unlink(tmp_index_path)
        except OSError:
            pass


def live_unmerged_paths() -> Optional[List[str]]:
    """The live repo's unmerged (conflicted) paths; ``None`` when Git itself
    failed (an unreadable inventory must never masquerade as "no conflicts" —
    callers keep their previous list or fail safe)."""
    rc_u, unmerged_out, _ue = _g._run_git_process_bounded(
        ["git", "diff", "--name-only", "--diff-filter=U", "-z"],
        timeout=_GIT_RUN_TIMEOUT_SEC, cwd=_g.REPO_DIR, text=False,
    )
    if rc_u != 0 or not isinstance(unmerged_out, bytes):
        return None
    return [os.fsdecode(path) for path in unmerged_out.split(b"\0") if path]


def _staged_blobs_batch(paths: List[bytes]) -> Tuple[Optional[Dict[bytes, bytes]], bytes]:
    """Read the staged (stage-0) blob of every path in ONE bounded ``git cat-file
    --batch`` process instead of one subprocess per path. Paths stay RAW BYTES
    end-to-end (N1): git's ``:path`` batch requests are byte-safe, while a valid
    POSIX filename with non-UTF-8 bytes fsdecodes to surrogates that a strict
    UTF-8 re-encode cannot represent. Returns ``(blobs_by_path, b"")`` or
    ``(None, failed_path)`` when any path's staged blob could not be read
    (unmerged entries resolve to ``missing``)."""
    if not paths:
        return {}, b""
    request = b"".join(b":" + path + b"\n" for path in paths)
    rc, raw, _err = _g._run_git_process_bounded(
        ["git", "cat-file", "--batch"], timeout=_GIT_RUN_TIMEOUT_SEC,
        cwd=_g.REPO_DIR, text=False, input_data=request,
    )
    if rc != 0 or not isinstance(raw, bytes):
        return None, paths[0]
    blobs: Dict[bytes, bytes] = {}
    cursor = 0
    for path in paths:
        header_end = raw.find(b"\n", cursor)
        if header_end < 0:
            return None, path
        header = raw[cursor:header_end].split(b" ")
        if len(header) == 2 and header[1] == b"missing":
            return None, path
        if len(header) != 3 or header[1] != b"blob":
            return None, path
        try:
            size = int(header[2])
        except ValueError:
            return None, path
        blob_start = header_end + 1
        blob_end = blob_start + size
        if blob_end >= len(raw) or raw[blob_end:blob_end + 1] != b"\n":
            return None, path
        blobs[path] = raw[blob_start:blob_end]
        cursor = blob_end + 1
    return blobs, b""


def managed_assisted_marker_check() -> Tuple[bool, str]:
    """Reject leftover conflict markers in the STAGED tree — the PRIMARY leakage gate: once the
    agent `git add`-s a marked file it is a 'resolved' (stage-0) entry, so `--diff-filter=U`
    no longer catches it. Scan the raw staged blob (no diff '+' prefix); flag a file only when
    BOTH a `<<<<<<<` and a `>>>>>>>` marker line are present (avoids false-positives on a lone
    markdown `=======` underline). Every git call is BOUNDED, and the blob scan is ONE
    ``git cat-file --batch`` process for the whole staged set, not one process per path."""
    import re

    start_re = re.compile(br"^<{7}", re.MULTILINE)
    end_re = re.compile(br"^>{7}", re.MULTILINE)
    rc_n, names_raw, _ne = _g._run_git_process_bounded(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRTUXB"],
        timeout=_GIT_RUN_TIMEOUT_SEC, cwd=_g.REPO_DIR, text=False,
    )
    if rc_n != 0 or not isinstance(names_raw, bytes):
        return False, "⚠️ MANAGED_UPDATE_ERROR: could not inspect staged files for conflict markers."
    # Paths stay RAW BYTES through the batch (N1); fsdecode is for DIAGNOSTICS
    # and the fallback argv only (argv round-trips via the surrogateescape
    # fsencode, so git receives the original bytes back).
    paths = [value for value in names_raw.split(b"\0") if value]
    # `cat-file --batch` requests are newline-delimited: a path containing a
    # newline cannot ride the batch and falls back to a bounded per-path read.
    batch_paths = [path for path in paths if b"\n" not in path]
    blobs, failed_path = _staged_blobs_batch(batch_paths)
    if blobs is None:
        return False, (
            "⚠️ MANAGED_UPDATE_ERROR: could not inspect staged file "
            f"{os.fsdecode(failed_path)}."
        )
    for path in (path for path in paths if b"\n" in path):
        rc_s, blob_raw, _se = _g._run_git_process_bounded(
            ["git", "show", f":{os.fsdecode(path)}"], timeout=_GIT_RUN_TIMEOUT_SEC,
            cwd=_g.REPO_DIR, text=False,
        )
        if rc_s != 0 or not isinstance(blob_raw, bytes):
            return False, (
                "⚠️ MANAGED_UPDATE_ERROR: could not inspect staged file "
                f"{os.fsdecode(path)}."
            )
        blobs[path] = blob_raw
    bad: List[str] = []
    for path in paths:
        blob = blobs.get(path, b"")
        if b"\0" in blob:
            continue
        if start_re.search(blob) and end_re.search(blob):
            bad.append(os.fsdecode(path))
    if bad:
        return False, (
            "⚠️ MANAGED_UPDATE_ERROR: unresolved conflict markers remain in: "
            + ", ".join(bad[:20])
            + " — remove every <<<<<<< / ======= / >>>>>>> before committing."
        )
    return True, ""


def existing_failed_update_ref(target_sha: str, *, not_at: str = "") -> str:
    """Return ``failed-update-<target12>`` when a prior attempt at this exact target
    left its preserved branch behind (a retry's resolver is pointed at it), else "".
    ``not_at`` filters out a branch that merely sits on the given sha (e.g. the
    pre-update base): such a ref preserves no attempt and must not be advertised."""
    if not target_sha:
        return ""
    name = f"failed-update-{target_sha[:12]}"
    resolved = _rev_parse(name)
    if not resolved or (not_at and resolved == not_at):
        return ""
    return name


def _preserve_failed_update_attempt(tx: Dict[str, Any]) -> str:
    """Publish the failed attempt on the DETERMINISTIC branch ``failed-update-<target12>``
    (keyed by the update target, so a retry of the same update overwrites one branch
    instead of scattering per-attempt refs) BEFORE the destructive reset. An UNCOMMITTED
    resolution (dirty tree and/or live MERGE_HEAD) is first serialized as a synthetic
    commit via a private index — plain ``write-tree`` is fatal on an unmerged index —
    with the natural parents ``[HEAD, MERGE_HEAD]`` (during resolution HEAD is the
    reviewed pre-update base and MERGE_HEAD is the target, preserving the [local,
    target] order). A committed candidate is preserved as-is. Best-effort by design:
    on any failure returns "" and the rescue snapshot remains the carrier of record —
    rollback itself never depends on this ref."""
    target = str(tx.get("target_sha") or "")
    rc_h, head, _he = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    if rc_h != 0 or not head:
        return ""
    name = f"failed-update-{target[:12]}" if target else f"failed-update-{head[:12]}"
    rc_s, dirty, _se = _g.git_capture(["git", "status", "--porcelain"])
    merge_head = _merge_head_sha()
    # Nothing beyond the pre-update state exists: a clean tree at the pre-update
    # sha with no live merge is NOT an attempt. Touching the branch here would
    # either mint a junk ref (rollback before any materialization) or — worse —
    # let a REPLAYED rollback (crash after the destructive reset) overwrite a
    # real preserved attempt with the bare base. Leave whatever the branch
    # already holds.
    if (
        head == str(tx.get("pre_update_sha") or "")
        and rc_s == 0 and not dirty.strip()
        and not merge_head
    ):
        return name if _rev_parse(name) else ""
    keep = head
    if (rc_s == 0 and dirty.strip()) or merge_head:
        tree, tree_error = worktree_snapshot_tree("HEAD")
        if not tree:
            _g.log.warning("failed-update snapshot tree failed: %s", tree_error)
            return ""
        parents: List[str] = ["-p", head]
        if merge_head:
            parents += ["-p", merge_head]
        rc_c, synthetic, commit_error = _git_run(
            ["git", "commit-tree", tree, *parents,
             "-m", f"ouroboros failed-update attempt (target {target[:12] or 'unknown'})"],
        )
        if rc_c != 0 or not synthetic:
            _g.log.warning("failed-update synthetic commit failed: %s", commit_error)
            return ""
        keep = synthetic
    rc_b, _bo, _be = _g.git_capture(["git", "branch", "-f", name, keep])
    if rc_b != 0 or _rev_parse(name) != keep:
        return ""
    return name


class UpdateTxCorrupt(RuntimeError):
    """The durable update-tx marker exists but is unreadable/invalid. Raised by
    ``update_tx_phase`` instead of silently REPLACING the corruption evidence
    with a caller's stale snapshot: ``read_update_tx_strict``'s contract is that
    corruption fails closed and loudly, and a merge-write that papers over it
    would launder a corrupt marker back into a "valid" one."""


def record_managed_tests_evidence(
    task_id: str, task_metadata: Optional[Dict[str, Any]] = None, *, force: bool = False,
) -> str:
    """Legacy forensic snapshot of the authorized resolver's current worktree.
    This does not establish which checkout a suite tested. New runner callers
    publish their actual tested subject through ``record_managed_tests_proof``.
    An env-disabled legacy call records nothing. ``force`` is supplied only by
    the authorized host runner after its mandatory suite actually ran green;
    it does not replace resolver authorization. Returns the tree sha, or ''.

    AUTHORITY NOTE (synthesis F2): the durable ``tests_evidence`` copy written
    here is FORENSIC/telemetry only. The tx marker is a plain writable file the
    authorized resolver's shell can reach, so the single-run gates
    (``_managed_candidate_needs_proof`` / ``_managed_post_commit_tests_gate``)
    consult ONLY the process-held record on the task ctx (see
    ``record_managed_tests_proof``), never this file."""
    if not force and os.environ.get("OUROBOROS_PRE_PUSH_TESTS", "1") != "1":
        return ""
    from supervisor import update_merge as _um

    tx = _um.authorized_assisted_task(task_id, task_metadata)
    if not tx:
        return ""
    # Fidelity guard: the hermetic runner materializes UNTRACKED files by
    # copying, which does not faithfully reproduce symlinks or other
    # non-regular entries — while `git add -A` records them as symlink blobs.
    # A tree hash covering content the suite never saw would be a forged
    # proof, so such candidates simply keep the mandatory gate run instead.
    rc_u, untracked_out, _uue = _g.git_capture(
        ["git", "ls-files", "--others", "--exclude-standard"]
    )
    if rc_u != 0:
        return ""
    for rel in untracked_out.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        full = os.path.join(str(_g.REPO_DIR), rel)
        if os.path.islink(full) or not os.path.isfile(full):
            _g.log.info("managed tests evidence: non-regular untracked entry %s — no proof recorded", rel)
            return ""
    tree, tree_error = worktree_snapshot_tree("HEAD")
    if not tree:
        _g.log.warning("managed tests evidence: snapshot failed: %s", tree_error)
        return ""
    from ouroboros.utils import utc_now_iso

    # Merge-write ONLY the evidence key onto the FRESH durable tx (W2 class):
    # a concurrent finalizer/watchdog/commit-transition write landing between
    # this function's tx read and a wholesale write-back would be rolled back
    # (phase, stash state, commit metadata) and strand recovery. On a CORRUPT
    # marker the forensic write is SKIPPED loudly (F1 semantics inside the
    # helper) — evidence is forensic-only post-F2, so the caller still gets
    # the tree for the process-held proof.
    update_tx_phase_or_keep(tx, {"tests_evidence": {"tree": tree, "at": utc_now_iso()}})
    return tree


def record_managed_tests_proof(ctx: Any, *, force: bool = False) -> str:
    """Publish forensic telemetry from the runner's process-held proof only.

    A green/None return alone and a later snapshot of the live resolver tree
    are not proof of the workload that ran. The shared runner owns authority;
    this function neither mints it nor reads it back from the transaction.
    """
    from ouroboros.commit_admission import PreflightTestProof
    from ouroboros.utils import utc_now_iso
    from supervisor import update_merge as _um

    if not force and os.environ.get("OUROBOROS_PRE_PUSH_TESTS", "1") != "1":
        return ""
    proof = getattr(ctx, "_preflight_test_proof", None)
    if not getattr(ctx, "_preflight_tests_passed", False) or not isinstance(proof, PreflightTestProof):
        return ""
    tx = _um.authorized_assisted_task(
        str(getattr(ctx, "task_id", "") or ""), getattr(ctx, "task_metadata", None),
    )
    if not tx:
        return ""
    update_tx_phase_or_keep(tx, {"tests_evidence": {"tree": proof.tree, "at": utc_now_iso()}})
    return proof.tree


def update_tx_phase(base_tx: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge-write a managed phase transition onto the FRESH durable tx.

    The commit flow holds a tx snapshot taken at attempt start, while
    in-attempt writers merge keys into the DURABLE marker mid-attempt (e.g.
    the compensating tests preflight recording ``tests_evidence``). Writing
    the stale snapshot wholesale silently drops that forensic evidence. The
    process-held runner proof separately owns reuse of an unchanged subject.
    This helper
    re-reads the durable tx and applies ONLY the caller's intended key changes
    on top, refusing to drop keys the durable record carries.

    Marker statuses are DISTINGUISHED (synthesis F1): an ABSENT marker falls
    back to the caller's snapshot (creation semantics — the snapshot is the
    only substrate left); a CORRUPT marker raises the typed
    ``UpdateTxCorrupt`` WITHOUT writing — replacing an unreadable marker with
    stale data would silently destroy the corruption evidence that
    ``read_update_tx_strict``'s fail-closed contract preserves for the owner.
    A FUTURE-schema marker (recorded by a newer release; F14) raises the same
    typed refusal — overwriting a transaction this version cannot interpret
    would corrupt the newer updater's recovery state. Returns the tx dict
    actually written."""
    from supervisor import update_merge as _um

    status, current = _um.read_update_tx_strict()
    if status == "corrupt":
        raise UpdateTxCorrupt(
            "update tx marker exists but is unreadable/invalid — refusing to "
            "overwrite the corruption evidence with a stale snapshot"
        )
    if status == "future":
        raise UpdateTxCorrupt(
            "update tx marker was recorded by a newer Ouroboros — refusing to "
            "overwrite a transaction this version cannot interpret"
        )
    merged = dict(current) if status == "valid" else dict(base_tx)
    merged.update(patch)
    _um.write_update_tx(merged)
    return merged


def update_tx_phase_or_keep(base_tx: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """``update_tx_phase`` for post-commit callers that must keep going: on a
    CORRUPT marker the phase write is SKIPPED with a loud supervisor row (the
    marker bytes stay untouched for the owner; boot recovery fails closed on
    them) and the merged dict is returned in-memory only."""
    try:
        return update_tx_phase(base_tx, patch)
    except UpdateTxCorrupt as exc:
        from supervisor.update_merge import _log_supervisor

        _g.log.error("managed update tx phase write skipped: %s", exc)
        _log_supervisor({
            "type": "managed_update_tx_phase_write_skipped_corrupt",
            "patch_keys": sorted(str(key) for key in patch),
            "error": str(exc),
        })
        merged = dict(base_tx)
        merged.update(patch)
        return merged


def managed_tests_evidence_covers(committed_tree: str) -> bool:
    """True when the durable tx carries a green-suite record for EXACTLY this
    tree. FORENSIC surface only (synthesis F2): the tx marker is
    resolver-writable, so the single-run gates never consult this — they read
    the process-held ctx record pinned by ``record_managed_tests_proof``."""
    if not committed_tree:
        return False
    from supervisor import update_merge as _um

    status, tx = _um.read_update_tx_strict()
    if status != "valid":
        return False
    evidence = tx.get("tests_evidence")
    return isinstance(evidence, dict) and str(evidence.get("tree") or "") == committed_tree


def lookup_update_stash(attempt_id: str) -> Tuple[bool, str, str]:
    """Typed stash lookup: ``(ok, sha, error)``. ``ok=False`` means the stash
    STORAGE could not be read — callers must treat that as UNKNOWN and keep
    their durable pointers, never as "nothing was stashed" (ok=True, sha="")."""
    marker = f"managed-update-{attempt_id}"
    rc_l, listing, list_error = _g.git_capture(["git", "stash", "list", "--format=%H %gs"])
    if rc_l != 0:
        return False, "", list_error or "git stash list failed"
    for line in listing.splitlines():
        sha, _sep, subject = line.strip().partition(" ")
        if marker in subject:
            return True, sha, ""
    return True, "", ""


def find_update_stash_sha(attempt_id: str) -> str:
    """Legacy shape over ``lookup_update_stash`` (error collapses to "")."""
    ok, sha, _error = lookup_update_stash(attempt_id)
    return sha if ok else ""

def stash_local_changes_for_update(attempt_id: str) -> Tuple[str, str, str]:
    """Stash tracked+untracked local work before a clean auto-update apply
    (owner decision Q1=C: dirty work rides the stash, never committed history).
    Returns ``(status, stash_sha, error)`` with status in {"ok", "push_failed",
    "lookup_unknown"}; "ok" with an empty sha means nothing to stash, and
    "lookup_unknown" means the entry EXISTS but could not be listed — callers
    must keep their durable pointers for a later retry."""
    marker = f"managed-update-{attempt_id}"
    rc, _out, error = _g.git_capture(
        ["git", "stash", "push", "--include-untracked", "-m", marker]
    )
    if rc != 0:
        return "push_failed", "", error or "git stash push failed"
    ok_l, sha, lookup_error = lookup_update_stash(attempt_id)
    if not ok_l:
        # The push succeeded but the entry cannot be listed: the entry EXISTS
        # somewhere — the caller must KEEP its durable tx pointer so boot can
        # retry the lookup, never clear it as if nothing was stashed.
        return "lookup_unknown", "", (
            "git stash list failed after a successful push; the entry named "
            f"{marker} exists — recovery retries on the next boot — "
            + (lookup_error or "unreadable stash")
        )
    if sha:
        return "ok", sha, ""
    # "No local changes to save" — the worktree raced clean; nothing to restore
    # later. The caller fail-closes if the tree still reports dirty.
    return "ok", "", ""


@dataclass(frozen=True)
class StashRestoreResult:
    """Effect, not a success-shaped note: only complete outcomes may clear a tx."""

    status: Literal["not_needed", "restored", "preserved", "incomplete"]
    note: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.status != "incomplete"


def _stash_recovery_note(stash_sha: str) -> str:
    return (f"the original local changes remain in stash {stash_sha[:12]} "
            "and current files should be inspected before "
            f"recovering with `git stash apply {stash_sha}`")


def restore_update_stash(
    stash_sha: str, context: str = "", *, before_cleanup=None,
) -> StashRestoreResult:
    """Apply the exact stash object; NEVER automatically drop an update stash.

    Transaction callers record intent before entering here and retain diagnostics
    through ``before_cleanup`` before resetting a failed partial apply. Cleanup is
    allowed only after a clean starting tree and is itself verified. A missing list
    entry says nothing about restoration: the exact object is the carrier.
    """
    if not stash_sha:
        return StashRestoreResult("not_needed")
    if _rev_parse(stash_sha) != stash_sha:
        return StashRestoreResult("incomplete", f"could not verify update stash {stash_sha}")
    recovery = _stash_recovery_note(stash_sha)
    rc_s, dirty, status_error = _g.git_capture(["git", "status", "--porcelain"])
    if rc_s != 0:
        return StashRestoreResult("incomplete", f"could not inspect current files: {status_error}; {recovery}")
    if dirty:
        return StashRestoreResult("preserved", f"local changes are present, so the stash was NOT auto-applied; {recovery}")
    rc, stdout, stderr = _g.git_capture(["git", "stash", "apply", stash_sha])
    evidence = {"apply_returncode": rc, "stdout": stdout, "stderr": stderr}
    if rc == 0:
        return StashRestoreResult("restored", "local changes restored; the update stash was retained", evidence)
    conflicts = live_unmerged_paths()
    evidence["conflict_paths"] = conflicts
    if conflicts:
        import json

        problem = "conflicting paths: " + ", ".join(json.dumps(path, ensure_ascii=False) for path in conflicts)
    elif conflicts is None:
        problem = "git stash apply failed; the conflict inventory could not be read"
    else:
        problem = "git stash apply failed without tracked conflicts (untracked collisions are not in that inventory)"
    diagnostics = "\n".join(part for part in (stdout, stderr) if part)
    note = f"{problem}; {recovery}" + (f"\nGit diagnostics:\n{diagnostics}" if diagnostics else "")
    if before_cleanup is not None:
        before_cleanup({"status": "cleanup_pending", "note": note, **evidence})
    cleanup = {}
    for name, cmd in (("reset", ["git", "reset", "--hard", "HEAD"]),
                      ("clean", ["git", "clean", "-fd"])):
        code, out, err = _g.git_capture(cmd)
        cleanup[name] = {"returncode": code, "stdout": out, "stderr": err}
        if code != 0:
            return StashRestoreResult("incomplete", f"{note}\nPartial-apply cleanup failed at {name}: {err or out}",
                                      {**evidence, "cleanup": cleanup})
    rc_s, dirty, status_error = _g.git_capture(["git", "status", "--porcelain"])
    evidence["cleanup"] = {**cleanup, "verified_clean": rc_s == 0 and not dirty}
    if rc_s != 0 or dirty:
        return StashRestoreResult("incomplete", f"{note}\nPartial-apply cleanup was not verified: {status_error or dirty}", evidence)
    return StashRestoreResult("preserved", note, evidence)


def restore_stash_with_marker(tx: Dict[str, Any], context: str) -> StashRestoreResult:
    """One recovery owner for all completion paths, with write-ahead custody.

    ``stash_restore`` is schema-2 evidence: old strict readers refuse it. Legacy
    ``stash_restored`` still means a successful recorded apply, never its intent.
    An interrupted apply/cleanup is NOT replayed: once the store works again the
    saved backup is handed back with an honest unknown-outcome note, preserving
    both the current files and any subsequent owner edits.
    """
    from supervisor import update_merge as _um

    stash_sha = str(tx.get("stash_sha") or "")
    record = tx.get("stash_restore") or {}
    if not stash_sha:
        return StashRestoreResult("not_needed")
    if not record and tx.get("stash_restored"):
        return StashRestoreResult("restored")  # existing schema-1 success evidence

    def persist(value: Dict[str, Any]) -> None:
        updated = {**tx, "stash_restore": {"context": context, "stash_sha": stash_sha, **value}}
        if value["status"] == "restored":
            updated["stash_restored"] = True
        _um.write_update_tx(updated)
        tx.update(updated)  # never publish an in-memory success before the write

    try:
        if record.get("status") in {"restored", "preserved"}:
            result = StashRestoreResult(record["status"], record.get("note", ""), record.get("evidence", {}))
        elif record:
            if _rev_parse(stash_sha) != stash_sha:
                return StashRestoreResult("incomplete", f"interrupted restore: could not verify stash {stash_sha}")
            result = StashRestoreResult(
                "preserved", "the automatic restore was interrupted; its result is unconfirmed and current files were left unchanged; "
                + _stash_recovery_note(stash_sha)
                + ("\n" + record["note"] if record.get("note") else ""),
                {"interrupted": record},
            )
            persist({"status": result.status, "note": result.note, "evidence": result.evidence})
        else:
            persist({"status": "applying"})
            result = restore_update_stash(stash_sha, context, before_cleanup=persist)
            persist({"status": result.status, "note": result.note, "evidence": result.evidence})
    except Exception as exc:
        _g.log.warning("update stash recovery could not be recorded", exc_info=True)
        return StashRestoreResult(
            "incomplete", f"automatic stash recovery remains unconfirmed ({type(exc).__name__}: {exc}); "
            "current files must not be reset or re-applied; " + _stash_recovery_note(stash_sha),
        )
    if not result.complete:
        return result
    event = {"type": "managed_update_stash_restored" if result.status == "restored" else "managed_update_stash_restore_failed",
             "context": context, "stash_sha": stash_sha, "restore_status": result.status,
             "stash_note": result.note, **result.evidence}
    # The existing event is the durable disclosure after the marker is removed.
    # A failed append retains the terminal marker; retrying it never repeats Git.
    try:
        recorded = _um._log_supervisor(event)
    except Exception:
        recorded = False
        _g.log.warning("update stash recovery event failed", exc_info=True)
    if not recorded:
        return StashRestoreResult("incomplete", result.note + "; recovery event could not be saved", result.evidence)
    return result


def destructive_apply_guard(branch: str, pre_update_sha: str) -> str:
    """Return "" when the live checkout is EXACTLY the state the apply was
    planned against, else the human-readable reason. Called under the update
    lock IMMEDIATELY before the first destructive command of an apply or a
    first materialization: the writer fence stops Ouroboros's own writers, not
    humans, and the resolver-boot wait leaves a seconds-to-minutes window where
    a late edit or commit would otherwise be reset/checked-out away without
    riding the stash or the rescue."""
    rc_b, cur, _be = _g.git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc_b != 0 or cur != branch:
        return f"on branch {cur!r}, expected {branch!r}"
    rc_h, head, _he = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    if rc_h != 0 or head != pre_update_sha:
        return f"HEAD moved to {(head or 'unknown')[:12]} (expected {pre_update_sha[:12]})"
    rc_s, dirty, _se = _g.git_capture(["git", "status", "--porcelain"])
    if rc_s != 0:
        return "could not read the worktree status"
    if dirty.strip():
        return "local changes appeared after the stash"
    if _merge_head_sha():
        return "a merge is already in progress"
    return ""


def project_version_carriers(target_sha: str, *, cwd: Optional[str] = None) -> Tuple[bool, str, str]:
    """Mandatory Q8 projection, shared by BOTH lanes: VERSION := the target's
    blob (conflicted or fork-only-drifted alike), then every purely mechanical
    carrier token is rewritten by the release-sync SSOT and staged — only for
    files that are not conflicted (adding an unmerged file would silently
    "resolve" it). Returns ``(ok, note, error)``; ok=False means the projection
    could not be completed and the caller must NOT freeze the half-projected
    tree as a merge commit or an M0 baseline (typed failure over silent drift).
    A target without a VERSION blob mandates nothing (ok=True, empty note)."""
    workdir = str(cwd or _g.REPO_DIR)
    rc_tv, target_blob, _tve = _git_run(["git", "rev-parse", f"{target_sha}:VERSION"], cwd=workdir)
    if rc_tv != 0 or not target_blob:
        return True, "", ""
    note = ""
    rc_sv, staged_blob, _sve = _git_run(["git", "rev-parse", ":VERSION"], cwd=workdir)
    if rc_sv != 0 or staged_blob != target_blob:
        rc_co, _co, co_error = _git_run(["git", "checkout", target_sha, "--", "VERSION"], cwd=workdir)
        if rc_co != 0:
            return False, "", f"VERSION checkout failed: {co_error}"
        note = ", VERSION projected to the target's version"
    rc_u, unmerged_out, u_error = _git_run(
        ["git", "diff", "--name-only", "--diff-filter=U"], cwd=workdir
    )
    if rc_u != 0:
        return False, note, f"conflict inventory unreadable: {u_error}"
    unmerged = {ln.strip() for ln in unmerged_out.splitlines() if ln.strip()}
    try:
        from ouroboros.tools.release_sync import sync_release_metadata

        changed_carriers = list(sync_release_metadata(workdir) or [])
    except Exception as exc:
        return False, note, f"carrier token sync failed: {type(exc).__name__}: {exc}"
    addable = [p for p in changed_carriers if p not in unmerged]
    if addable:
        rc_ca, _cao, ca_error = _git_run(["git", "add", "--", *addable], cwd=workdir)
        if rc_ca != 0:
            return False, note, f"carrier staging failed: {ca_error}"
        note += f", {len(addable)} carrier file(s) token-synced"
    # POSTCONDITION: the sync SSOT silently no-ops on shapes it does not
    # recognize (invalid VERSION, exotic token spellings). Verify every
    # NON-conflicted carrier actually agrees with the projected VERSION; a
    # carrier the sync cannot fix is a typed failure, never a silent success.
    try:
        from ouroboros.tools.release_sync import version_carrier_desyncs

        import pathlib as _pl

        root = _pl.Path(workdir)

        def _read(rel: str) -> str:
            path = root / rel
            return path.read_text(encoding="utf-8") if path.exists() and rel not in unmerged else ""

        version_str = (root / "VERSION").read_text(encoding="utf-8").strip()
        from ouroboros.tools.release_sync import is_release_version

        if not is_release_version(version_str):
            # Nothing enforceable for a non-release version string; the sync
            # SSOT skipped it too — that is the target's own shape, not drift.
            return True, note, ""
        readme_text = _read("README.md")
        desyncs = version_carrier_desyncs(
            version_str,
            pyproject_text=_read("pyproject.toml"),
            uv_lock_text=_read("uv.lock"),
            web_package_text=_read("web/package.json"),
            web_package_lock_text=_read("web/package-lock.json"),
            readme_text=readme_text,
            arch_text=_read("docs/ARCHITECTURE.md"),
            api_types_text=_read("web/modules/api_types.js"),
            download_readme_text=readme_text,
            site_install_text=_read("site/install/index.html"),
            docs_install_text=_read("docs/install/index.html"),
            detailed=True,
        )
    except Exception as exc:
        return False, note, f"carrier postcondition check failed: {type(exc).__name__}: {exc}"
    if desyncs:
        return False, note, "carriers still desynced after token sync: " + "; ".join(desyncs)
    return True, note, ""
