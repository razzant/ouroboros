"""Managed-update merge engine (P2): a REAL git 3-way merge in an isolated temp worktree,
the apply / rollback / smoke / finalize primitives, and a FAIL-CLOSED update lock.

Kept OUT of ``git_ops`` (module-size discipline) but depends on it for the live-repo git
helpers — referenced via the ``git_ops`` module object (``_g.X``) so a test that
monkeypatches ``git_ops.REPO_DIR`` / ``_managed_update_target`` / ``_git_dir`` /
``DRIVE_ROOT`` is followed by these primitives. Control plane: ``ouroboros.gateway.control``
orchestrates lock → kill workers → re-plan → rescue → tx marker → apply → smoke → restart.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.utils import append_jsonl, utc_now_iso
from supervisor import git_ops as _g
# Candidate primitives live in supervisor.update_candidate (module-size split);
# re-exported: every caller and test reaches them via this module (F401 intended).
from supervisor.update_candidate import (  # noqa: F401
    _MERGE_NEUTRAL_FLAGS, _git_run, _merge_head_sha, _preserve_failed_update_attempt,
    _rev_parse, existing_failed_update_ref, live_unmerged_paths,
    UpdateTxCorrupt, managed_assisted_marker_check, managed_tests_evidence_covers,
    record_managed_tests_evidence,
    record_managed_tests_proof, update_tx_phase, update_tx_phase_or_keep,
    worktree_snapshot_tree,
    find_update_stash_sha, restore_stash_with_marker, restore_update_stash,
    stash_local_changes_for_update, lookup_update_stash,
    destructive_apply_guard, project_version_carriers,
    quarantine_corrupt_update_tx_marker,
)
# The planner, the clean-plan commit builder and the live materializer — the
# carrier engine's three insertion points — live in supervisor.update_merge_plan
# (module-size split, re-cut from the redesign's two-module form); re-exported:
# every caller and test reaches them via this module (F401 intended).
from supervisor.update_merge_plan import (  # noqa: F401
    _build_clean_merge_commit, materialize_assisted_merge_live, plan_managed_update_merge,
)

UPDATE_TX_MARKER_NAME = "ouroboros-update-tx.json"

# ABI 7.0 / F14 N−1 transition shim: every tx write stamps the marker with the
# shared ``_schema_version`` key. ``finalize_managed_update_on_boot`` is the
# stable N−1→N entry point — an UNSTAMPED marker is the accepted pre-7.0 form
# (a version-N updater must finalize the transaction its N−1 predecessor
# recorded), a stamp ≤ ours is current, and an integer stamp ABOVE ours means
# a NEWER release recorded the transaction: this version cannot interpret it,
# so every strict reader fails closed (``"future"`` — never rolled back, left
# for the owner). A rolled-back N−1 binary reads a stamped marker unchanged:
# the stamp is one additive key its reader ignores.
UPDATE_TX_SCHEMA_VERSION = 1

# ``dict.get`` sentinel distinguishing a MISSING ``_schema_version`` key (the
# accepted pre-7.0 unstamped form) from an explicit stored ``null`` (corrupt).
_SCHEMA_STAMP_ABSENT = object()


def managed_update_constitution_present(ref: str = "HEAD") -> bool:
    """Whether *ref* keeps the non-empty regular BIBLE.md blob required by P4."""
    from supervisor.update_source import official_ref_has_constitution

    return official_ref_has_constitution(ref, repo_dir=_g.REPO_DIR)


def _update_tx_marker_path():
    return _g._git_dir() / UPDATE_TX_MARKER_NAME


def read_update_tx() -> Dict[str, Any]:
    import json

    path = _update_tx_marker_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def write_update_tx(payload: Dict[str, Any]) -> None:
    from ouroboros.contracts.schema_versions import with_schema_version
    from ouroboros.utils import atomic_write_json

    atomic_write_json(
        _update_tx_marker_path(),
        with_schema_version(payload, UPDATE_TX_SCHEMA_VERSION),
        trailing_newline=True,
    )


def clear_update_tx() -> bool:
    try:
        _update_tx_marker_path().unlink()
    except FileNotFoundError:
        return True
    except Exception:
        _g.log.warning("Failed to clear update tx marker", exc_info=True)
        return False
    return True


_ASSISTED_PHASES = ("materializing_assisted", "assisted_resolution", "committing_assisted")
GATE_BLOCKED_PHASE = "gate_blocked"
# The repository is ALREADY in its final good state and only the tx-marker
# unlink failed: recovery retries the cleanup and must never roll back. A
# separate phase — not a gate_blocked sub-kind — because gate_blocked's boot
# contract ("a refused merge: fresh rollback, never promotion") stays exact.
MARKER_CLEANUP_RETRY_PHASE = "marker_cleanup_retry"
_ASSISTED_AUTHORITY_FIELDS = (
    "task_id",
    "pre_update_sha",
    "pre_update_branch",
    "base_sha",
    "target_sha",
    "target_ref",
    "update_channel",
    "local_snapshot",
)
_PRE_RESTART_SMOKE_PENDING = "pending"
_PRE_RESTART_SMOKE_PASSED = "passed"


def assisted_authority_fingerprint(tx: Dict[str, Any]) -> str:
    """Bind resolver privilege to the immutable task metadata created by the host."""
    payload = {key: str(tx.get(key) or "") for key in _ASSISTED_AUTHORITY_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assisted_task_metadata_authorizes(
    tx: Dict[str, Any], task_metadata: Optional[Dict[str, Any]]
) -> bool:
    return bool(
        assisted_writer_gate_reason_from_metadata(task_metadata)
        == assisted_writer_gate_reason(tx)
    )


def assisted_writer_gate_reason_from_metadata(
    task_metadata: Optional[Dict[str, Any]],
) -> str:
    managed = (
        task_metadata.get("managed_update")
        if isinstance(task_metadata, dict)
        else None
    )
    fingerprint = (
        str(managed.get("authority_fingerprint") or "")
        if isinstance(managed, dict)
        else ""
    )
    return f"managed_update:assisted:{fingerprint}" if fingerprint else ""


def assisted_writer_gate_reason(tx: Dict[str, Any]) -> str:
    return f"managed_update:assisted:{assisted_authority_fingerprint(tx)}"


def release_assisted_writer_gate_after_task(
    task_metadata: Optional[Dict[str, Any]],
) -> bool:
    """Release the server latch after a worker cleared the durable tx on rollback."""
    reason = assisted_writer_gate_reason_from_metadata(task_metadata)
    if not reason:
        return False
    try:
        lock_fh = acquire_update_lock()
    except RuntimeError:
        return False
    try:
        if active_update_tx():
            return False
        from supervisor.workers import open_repo_writer_admission

        return open_repo_writer_admission(expected_reason=reason)
    finally:
        release_update_lock(lock_fh)


def mark_update_tx_gate_blocked(reason: str, detail: str = "", *, cleanup_only: bool = False) -> bool:
    """Re-phase a valid live tx to ``gate_blocked`` with full failure evidence.

    Boot's ``gate_blocked`` branch RETRIES the rollback (never a terminal stop),
    while the pre-gate phase comes OFF the marker: a refused merge left in
    ``committing_assisted`` reads to boot recovery as "died mid-commit" and gets
    promoted without the gate rerunning. Only a VALID live tx is re-marked
    (returns True): an absent marker would mint a permanent phantom transaction,
    and a corrupt one keeps its raw evidence for the owner (both return False)."""
    status, tx = read_update_tx_strict()
    if status != "valid":
        _log_supervisor({
            "type": "managed_update_gate_blocked_skipped",
            "reason": str(reason or "managed_update_gate_failed"),
            "marker_status": status,
        })
        return False
    tx.update({
        "gate_blocked_from_phase": str(tx.get("phase") or ""),
        "phase": MARKER_CLEANUP_RETRY_PHASE if cleanup_only else GATE_BLOCKED_PHASE,
        "gate_blocked_reason": str(reason or "managed_update_gate_failed"),
        "gate_blocked_detail": str(detail or ""),
        "gate_blocked_at": utc_now_iso(),
    })
    write_update_tx(tx)
    _log_supervisor({
        "type": "managed_update_gate_blocked",
        "reason": tx["gate_blocked_reason"],
        "detail": tx["gate_blocked_detail"],
    })
    return True


def read_update_tx_strict() -> Tuple[str, Dict[str, Any]]:
    """Strict tx read for safety-critical gates (commit authorization, tx-active rejection):
    return ``(status, tx)`` where status is ``"absent"`` / ``"valid"`` / ``"corrupt"`` /
    ``"future"``. A marker that exists but is unreadable/invalid is ``corrupt`` — callers
    MUST fail closed (block mutative update/commit ops) rather than treat it as ``absent``.

    Schema admission (F14 N−1 shim): an UNSTAMPED marker — the key ABSENT —
    is the accepted pre-7.0 form and reads ``valid``: the boot finalizer must
    drive a transaction the N−1 updater recorded. An integer stamp above
    ``UPDATE_TX_SCHEMA_VERSION`` is ``future`` (recorded by a newer release;
    the raw tx is returned as evidence but no caller may interpret or
    overwrite it); ANY present non-integer stamp — an explicit ``null``
    included — is ``corrupt`` (no writer ever stamps ``null``, so it is a
    damaged stamp, not the legacy unstamped form)."""
    import json

    path = _update_tx_marker_path()
    if not path.is_file():
        return "absent", {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "corrupt", {}
    if not isinstance(raw, dict) or not raw:
        return "corrupt", {}
    from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY

    stamp = raw.get(SCHEMA_VERSION_KEY, _SCHEMA_STAMP_ABSENT)
    if stamp is not _SCHEMA_STAMP_ABSENT:
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 1:
            return "corrupt", {}
        if stamp > UPDATE_TX_SCHEMA_VERSION:
            return "future", raw
    return "valid", raw


def active_update_tx() -> Dict[str, Any]:
    """Return the active tx dict if a (valid, corrupt or future) marker is present, else
    ``{}``. A corrupt or future marker counts as ACTIVE (fail-closed) so a second apply
    cannot proceed over it."""
    status, tx = read_update_tx_strict()
    if status == "absent":
        return {}
    return tx or {"phase": "corrupt"}


def authorized_assisted_task_strict(
    task_id: str, task_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any]]:
    """Typed variant of ``authorized_assisted_task``: ``(marker_status, tx)``.

    ``marker_status`` is the raw ``read_update_tx_strict`` status, so a caller
    that must distinguish REAL marker corruption from an honest "not the
    resolver" (the authority predicate's loud A4 channel) can read it here.
    ``tx`` is non-empty only for the authorized resolver of a valid assisted
    tx — every authorization miss, corruption included, yields ``{}``."""
    status, tx = read_update_tx_strict()
    if status != "valid":
        return status, {}
    if str(tx.get("phase") or "") not in _ASSISTED_PHASES:
        return status, {}
    if str(tx.get("task_id") or "") != str(task_id or ""):
        return status, {}
    if not assisted_task_metadata_authorizes(tx, task_metadata):
        return status, {}
    return status, tx


def authorized_assisted_task(
    task_id: str, task_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return the active assisted tx only for its host-enqueued resolver task."""
    return authorized_assisted_task_strict(task_id, task_metadata)[1]


def create_rescue_local_ref(local_snapshot: str) -> str:
    """Pin the given sha — the durable carrier of the owner's uncommitted+untracked work
    (the STASH commit since the stash-first order; the synthetic local snapshot on legacy
    transactions) — to a branch so a later rollback / git-gc can never lose it. Returns
    the branch name."""
    short = (local_snapshot or "")[:12]
    name = f"rescue-local-{short}"
    if local_snapshot:
        rc, _out, _err = _g.git_capture(["git", "branch", "-f", name, local_snapshot])
        if rc == 0 and _rev_parse(name) == local_snapshot:
            return name
    return ""


def _assisted_head_state(tx: Dict[str, Any]) -> str:
    """Classify the live HEAD vs the assisted tx for boot recovery — keyed on MERGE STATE. During
    resolution HEAD == pre_update_sha (the merge result is staged but uncommitted); the reviewed
    merge commit has pre_update_sha as its FIRST parent and target_sha as its second:
      - ``committed``  : HEAD is a 2-parent commit whose 2nd parent is target_sha (descends from
                         pre_update_sha), or tx.merge_commit is in HEAD.
      - ``in_progress``: HEAD == pre_update_sha (no commit yet — re-materialize/resume).
      - ``diverged``   : HEAD descends from pre_update_sha but is NOT the target merge (a real
                         reviewed commit landed on top — keep it, never reset over it).
      - ``unknown``    : cannot resolve (fail safe: keep)."""
    pre = str(tx.get("pre_update_sha") or "")
    target_sha = str(tx.get("target_sha") or "")
    merge_commit = str(tx.get("merge_commit") or "")
    rc_h, head, _he = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    if rc_h != 0 or not head:
        return "unknown"
    if merge_commit and (
        head == merge_commit
        or _g.git_capture(["git", "merge-base", "--is-ancestor", merge_commit, "HEAD"])[0] == 0
    ):
        return "committed"
    if pre and head == pre:
        return "in_progress"
    # A merge commit whose 2nd parent is the target and which descends from pre_update_sha.
    if pre and target_sha:
        rc_p, parents, _pe = _g.git_capture(["git", "rev-list", "--parents", "-n", "1", "HEAD"])
        descends = _g.git_capture(["git", "merge-base", "--is-ancestor", pre, "HEAD"])[0] == 0
        if rc_p == 0 and target_sha in parents.split()[1:] and descends:
            return "committed"
        if descends:
            return "diverged"
    return "unknown"


def acquire_update_lock():
    """Acquire the FAIL-CLOSED managed-update lock; return an open file handle that keeps
    the lock held. Raise RuntimeError if another update operation holds it — the update
    MUST NOT proceed unlocked (a self-mod write or owner-restart racing the reset has
    corrupted trees before). Release with ``release_update_lock(fh)``."""
    from ouroboros.platform_layer import file_lock_exclusive_nb

    lock_dir = _g.DRIVE_ROOT / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    fh = (lock_dir / "managed_update.lock").open("a+")
    try:
        file_lock_exclusive_nb(fh.fileno())  # raises OSError if already held
    except OSError as exc:
        fh.close()
        raise RuntimeError("managed_update.lock is held by another update operation") from exc
    return fh


def release_update_lock(fh) -> None:
    from ouroboros.platform_layer import file_unlock

    try:
        file_unlock(fh.fileno())
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


def apply_managed_merge_update(branch: str, merge_commit: str) -> Tuple[bool, str]:
    """Land a prepared merge commit on the LIVE repo. Caller MUST already hold the update
    lock, have stopped workers, written the rescue + tx markers, and stashed dirty local
    work (Q1=C: the stash — plus the rescue — carries it; committed history never does),
    so the worktree is reset away here. Returns (ok, message)."""
    if not merge_commit:
        return False, "no merge_commit to apply"
    rc0, _o0, e0 = _g.git_capture(["git", "reset", "--hard", "HEAD"])
    if rc0 != 0:
        return False, f"reset before apply failed: {e0}"
    rc_clean0, _co0, ce0 = _g.git_capture(["git", "clean", "-fd"])
    if rc_clean0 != 0:
        return False, f"clean before apply failed: {ce0}"
    rc1, _o1, e1 = _g.git_capture(["git", "checkout", "-B", branch, merge_commit])
    if rc1 != 0:
        return False, f"checkout -B {branch} {merge_commit[:12]} failed: {e1}"
    rc2, _o2, e2 = _g.git_capture(["git", "reset", "--hard", merge_commit])
    if rc2 != 0:
        return False, f"reset --hard {merge_commit[:12]} failed: {e2}"
    rc_clean, _co, clean_error = _g.git_capture(["git", "clean", "-fd"])
    if rc_clean != 0:
        return False, f"clean after apply failed: {clean_error}"
    return True, f"applied merge {merge_commit[:12]} to {branch}"


def rollback_managed_update(
    reason: str = "update_rollback", *, reopen_writer_admission: bool = True
) -> Tuple[bool, str]:
    """Roll a failed managed update back to the pre-update SHA in the tx marker. Preserves
    the failed candidate — an uncommitted resolution included — on the deterministic
    branch ``failed-update-<target12>`` for inspection and retry, hard-resets the branch
    to pre_update_sha, cleans, clears the update markers, and logs. Boot recovery keeps
    writer admission closed until the restored process restarts. Does NOT push (unlike
    rollback_to_version, which can push origin — wrong for an internal recovery).

    Marker admission is STRICT (F14): a FUTURE-schema marker was recorded by a newer
    release — this code cannot interpret its semantics, so the rollback refuses typed
    BEFORE any marker write or destructive reset/checkout/clean (the raw marker stays
    byte-identical, left for the owner); a corrupt marker keeps its raw evidence and
    refuses on the missing pre_update_sha exactly as before."""
    status, tx = read_update_tx_strict()
    if status == "future":
        return False, (
            "update tx marker was recorded by a newer version of Ouroboros; "
            "rollback refuses to interpret it — marker left untouched for the owner"
        )
    pre = str(tx.get("pre_update_sha") or "")
    branch = str(tx.get("pre_update_branch") or _g.BRANCH_DEV)
    if not pre:
        return False, "no pre_update_sha in update tx marker"
    tx["phase"] = "rolling_back"
    tx["rollback_reason"] = str(reason or "update_rollback")
    write_update_tx(tx)
    # Q1=C replay guard: a previous rollback attempt that already RESTORED the
    # owner's stashed work (durable stash_restored marker, written between the
    # stash apply and its drop) must not reset that work away on resume. When
    # HEAD already sits at the pre-update SHA on the right branch, the repo part
    # of the rollback is done — skip the destructive re-reset and re-restore.
    resume_with_restored_work = False
    if bool(tx.get("stash_restored")):
        rc_rh, resumed_head, _rhe = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
        rc_rb, resumed_branch, _rbe = _g.git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        resume_with_restored_work = (
            rc_rh == 0 and resumed_head == pre and rc_rb == 0 and resumed_branch == branch
        )
    # Disarm launcher replay before touching the checkout. If interrupted from
    # here onward, the durable rolling_back phase resumes this exact operation.
    if not _g._clear_update_intent():
        return False, "rollback could not clear update intent before restoring the repository"
    if resume_with_restored_work:
        return _finish_rollback(tx, pre, branch, reason, "", reopen_writer_admission)
    # Fresh rescue BEFORE the destructive reset; the persisted pointer is the replay guard.
    if not tx.get("rollback_rescue"):
        _g.rescue_into_tx(tx, key="rollback_rescue", reason=str(reason),
                          context="rollback", writer=write_update_tx)

    def _fail(message: str) -> Tuple[bool, str]:
        # Drop the marker (best-effort) so a RETRY re-rescues the tree it actually finds.
        if tx.pop("rollback_rescue", None) is not None:
            try:
                write_update_tx(tx)
            except Exception:
                _g.log.warning("could not drop the stale rollback_rescue marker", exc_info=True)
        return False, message

    failed_ref = _preserve_failed_update_attempt(tx)
    if failed_ref:
        tx["failed_update_ref"] = failed_ref
        write_update_tx(tx)
    rc0, _o0, e0 = _g.git_capture(["git", "reset", "--hard", "HEAD"])
    if rc0 != 0:
        return _fail(f"rollback reset failed before checkout: {e0}")
    rc_clean0, _co0, ce0 = _g.git_capture(["git", "clean", "-fd"])
    if rc_clean0 != 0:
        return _fail(f"rollback clean failed before checkout: {ce0}")
    rc1, _o1, e1 = _g.git_capture(["git", "checkout", "-B", branch, pre])
    if rc1 != 0:
        return _fail(f"rollback checkout -B {branch} {pre[:12]} failed: {e1}")
    rc2, _o2, e2 = _g.git_capture(["git", "reset", "--hard", pre])
    if rc2 != 0:
        return _fail(f"rollback reset --hard {pre[:12]} failed: {e2}")
    rc_clean, _co, clean_error = _g.git_capture(["git", "clean", "-fd"])
    if rc_clean != 0:
        return _fail(f"rollback clean failed: {clean_error}")
    rc_h2, restored_head, head_error = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    rc_b2, restored_branch, branch_error = _g.git_capture(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    )
    rc_s2, restored_status, status_error = _g.git_capture(["git", "status", "--porcelain"])
    if (
        rc_h2 != 0
        or restored_head != pre
        or rc_b2 != 0
        or restored_branch != branch
        or rc_s2 != 0
        or bool(restored_status.strip())
        or bool(_merge_head_sha())
        or not managed_update_constitution_present("HEAD")
    ):
        detail = head_error or branch_error or status_error or "rollback verification mismatch"
        return _fail(f"rollback could not be verified: {detail}")
    # Q1=C: the pre-update tree is exactly where the stash was taken, so this
    # restore is conflict-free; a kept stash entry is disclosed, never dropped.
    stash_note = restore_stash_with_marker(tx, "rollback")
    return _finish_rollback(tx, pre, branch, reason, stash_note, reopen_writer_admission)


def _finish_rollback(
    tx: Dict[str, Any], pre: str, branch: str, reason: str,
    stash_note: str, reopen_writer_admission: bool,
) -> Tuple[bool, str]:
    """Shared rollback tail: close admission, clear the tx, log, reopen, report."""
    gate_reason = "managed_update:rollback"
    try:
        from supervisor.workers import close_repo_writer_admission

        close_repo_writer_admission(gate_reason)
    except Exception:
        return False, "rollback restored the repository but could not close writer admission"
    if not clear_update_tx():
        return False, "rollback restored the repository but could not clear update transaction"
    rescue = tx.get("rollback_rescue") if isinstance(tx.get("rollback_rescue"), dict) else {}
    append_jsonl(
        _g.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {"ts": utc_now_iso(), "type": "managed_update_rolled_back", "reason": reason,
         "pre_update_sha": pre, "branch": branch,
         **({"failed_update_ref": tx["failed_update_ref"]} if tx.get("failed_update_ref") else {}),
         **{f"rescue_{key}": rescue[key] for key in ("path", "ref", "ts") if rescue.get(key)},
         **({"rescue_error": tx["rollback_rescue_error"]} if tx.get("rollback_rescue_error") else {})},
    )
    if reopen_writer_admission:
        try:
            from supervisor.workers import open_repo_writer_admission

            open_repo_writer_admission(expected_reason=gate_reason)
        except Exception:
            _g.log.warning("rollback restored the repository but could not reopen writer admission", exc_info=True)
    message = f"rolled back to {pre[:12]}"
    if tx.get("failed_update_ref"):
        message += f"; the attempt is preserved on branch {tx['failed_update_ref']}"
    if stash_note:
        message += f"; {stash_note}"
    return True, message


def ensure_assisted_resolver_ready(expected_sha: str, timeout_sec: float = 90.0) -> bool:
    """Prove a fresh resolver imported the exact clean tree before conflicts go live."""
    from supervisor import workers
    from supervisor.queue import _queue_lock

    with _queue_lock:
        if workers.WORKERS:
            return False  # update quiescence must leave no ambiguous prior generation
    events_cursor = workers.events_log_cursor()
    try:
        if not workers.ensure_worker_pool_started(n=1, allow_disabled_restart=True):
            return False
    except Exception:
        _g.log.warning("assisted resolver worker start failed", exc_info=True)
        return False
    deadline = time.monotonic() + max(float(timeout_sec), 0.1)
    while time.monotonic() < deadline:
        with _queue_lock:
            live_pids = {
                int(worker.proc.pid)
                for worker in workers.WORKERS.values()
                if worker.proc.pid and worker.proc.is_alive()
            }
        if not live_pids:
            return False
        boot = workers._first_worker_event_since(events_cursor, "worker_ready")
        try:
            boot_pid = int((boot or {}).get("pid") or 0)
        except (TypeError, ValueError):
            boot_pid = 0
        if boot_pid in live_pids and str((boot or {}).get("git_sha") or "") == expected_sha:
            return True
        time.sleep(0.05)
    return False


def enqueue_assisted_resolution_task(tx: Dict[str, Any]) -> str:
    """Enqueue (front) the single authorized resolution task for an assisted merge and start a
    worker for it. Used by both the apply orchestration and boot recovery so the objective +
    structured metadata stay in one place. Returns the task id."""
    from supervisor import workers
    from supervisor.queue import _queue_lock, enqueue_task
    from supervisor.update_merge_policy import assisted_objective

    task_id = str(tx.get("task_id") or "")
    prior_terminal_status = ""
    if task_id and task_id not in workers.RUNNING:
        try:
            from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

            prior = load_task_result(_g.DRIVE_ROOT, task_id) or {}
            if str(prior.get("status") or "") in _TRULY_TERMINAL_STATUSES:
                prior_terminal_status = str(prior.get("status") or "")
        except Exception:
            # Fail-open to the historical behavior: an unreadable ledger must
            # not block the resume that used to work without this check.
            prior_terminal_status = ""
    if prior_terminal_status:
        # Boot-resume after a shutdown that already settled the recorded
        # resolver terminally (e.g. SIGTERM wrote a durable "cancelled"):
        # re-enqueueing the OLD id is silently dropped pre-assignment by
        # ``_drop_cancelled_pending`` (its durable result is terminal), leaving
        # the tx wedged in assisted_resolution with no resolver task and no
        # orphan watchdog (which only arms on task_done). Mint a FRESH task id
        # and move the tx's recorded id to it. Paid-review-cycle ceiling
        # continuity: the ORIGINAL resolver id — the root the ledger already
        # charged — is pinned once and threaded into the task metadata below.
        import uuid

        fresh_id = "update_assisted_merge_" + uuid.uuid4().hex[:8]
        tx.setdefault("original_root_task_id", task_id)
        tx["task_id"] = fresh_id
        write_update_tx(tx)
        _log_supervisor({
            "type": "managed_update_assisted_reenqueued_fresh",
            "task_old": task_id,
            "task_new": fresh_id,
            "prior_status": prior_terminal_status,
        })
        task_id = fresh_id
    task = {
        "id": task_id,
        "text": assisted_objective(tx),
        "type": "task",
        "chat_id": int(tx.get("owner_chat_id") or 0),
        "metadata": {
            # Root continuity for the paid-review-cycle ceiling
            # (``commit_gate.resolve_root_task_id`` honors this key first): the
            # first enqueue pins the resolver's own id (same value the gate
            # would derive), a fresh-id re-enqueue pins the ORIGINAL id so the
            # money ceiling never resets across resume chains of one update.
            "root_task_id": str(tx.get("original_root_task_id") or task_id),
            "managed_update": {
                "target_sha": str(tx.get("target_sha") or ""),
                "conflict_paths": list(tx.get("conflict_paths") or []),
                "local_snapshot": str(tx.get("local_snapshot") or ""),
                "authority_fingerprint": assisted_authority_fingerprint(tx),
            }
        },
    }
    try:
        if not workers.ensure_worker_pool_started(allow_disabled_restart=True):
            _g.log.warning(
                "enqueue_assisted_resolution_task: worker pool remains explicitly disabled"
            )
            return ""
    except Exception:
        _g.log.warning("enqueue_assisted_resolution_task: worker pool start failed", exc_info=True)
        return ""
    with _queue_lock:
        pending = next(
            (
                candidate
                for candidate in workers.PENDING
                if str(candidate.get("id") or "") == task_id
            ),
            None,
        )
        if pending is not None:
            # Older updater versions could persist this task without the host-bound
            # authorization metadata. Refresh the durable queue row from the active
            # transaction instead of leaving boot recovery permanently gated.
            pending.update(task)
        elif task_id not in workers.RUNNING:
            enqueue_task(task, front=True)
    return task_id


def _run_update_smoke(cmd: List[str], timeout_sec: float = 120.0) -> Dict[str, Any]:
    from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs
    from ouroboros.tools.shell import _active_subprocesses, _subprocess_lock

    proc = subprocess.Popen(
        cmd,
        cwd=str(_g.REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **subprocess_new_group_kwargs(),
    )
    with _subprocess_lock:
        _active_subprocesses.add(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except Exception:
                stdout, stderr = "", ""
            return {
                "ok": False,
                "stdout": stdout or "",
                "stderr": f"update smoke exceeded {timeout_sec:.0f}s and was terminated",
                "returncode": 124,
            }
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "returncode": proc.returncode,
        }
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)


def update_restart_smoke() -> Dict[str, Any]:
    """Stronger pre-restart smoke than ``import_test`` for gating an update apply: no
    unmerged index, ``py_compile server.py``, and an import of the core boot surface.
    pytest is intentionally NOT in this blocking gate (bloat/risk in a live self-updater)."""
    if not managed_update_constitution_present("HEAD"):
        return {
            "ok": False,
            "stderr": "BIBLE.md is absent, empty, or not a regular file",
            "returncode": 1,
        }
    if getattr(sys, "frozen", False):
        return {"ok": True, "skipped": "frozen"}
    rc_u, unmerged, _ue = _g.git_capture(["git", "diff", "--name-only", "--diff-filter=U"])
    if rc_u != 0:
        return {"ok": False, "stderr": "could not inspect unmerged paths", "returncode": rc_u}
    if unmerged.strip():
        return {"ok": False, "stderr": f"unmerged paths remain: {unmerged}", "returncode": 1}
    deps_ok, deps_message = _g.sync_runtime_dependencies(reason="managed_update_pre_restart")
    if not deps_ok:
        return {"ok": False, "stderr": f"dependency sync failed: {deps_message}", "returncode": 1}
    compiled = _run_update_smoke([sys.executable, "-m", "py_compile", "server.py"])
    if not compiled["ok"]:
        return compiled
    return _run_update_smoke(
        [sys.executable, "-c",
         "import server, ouroboros.gateway.router, supervisor.queue, "
         "supervisor.events, ouroboros.tools.registry; print('smoke_ok')"]
    )


_ASSISTED_BOOT_ATTEMPT_CAP = 3


def _log_supervisor(payload: Dict[str, Any]) -> None:
    append_jsonl(_g.DRIVE_ROOT / "logs" / "supervisor.jsonl", {"ts": utc_now_iso(), **payload})


def _finalize_pending_boot_smoke(tx: Dict[str, Any], supervisor_ready: bool) -> Dict[str, Any]:
    """Health-check a committed-and-restarted update (auto_merge OR a committed assisted
    merge). Pre-restart smoke already ran inline; this is the post-boot backstop + boot-loop
    guard: clear on healthy boot, roll back to pre_update_sha on a genuine miss / brick-loop."""
    attempts = int(tx.get("boot_attempts") or 0) + 1
    merge_commit = str(tx.get("merge_commit") or "")
    rc_h, head, _he = _g.git_capture(["git", "rev-parse", "HEAD"])
    head_resolved = rc_h == 0 and bool(merge_commit)
    merge_in_head = head_resolved and (
        head == merge_commit
        or _g.git_capture(["git", "merge-base", "--is-ancestor", merge_commit, "HEAD"])[0] == 0
    )
    if (
        bool(supervisor_ready)
        and merge_in_head
        and str(tx.get("pre_restart_smoke") or "") == _PRE_RESTART_SMOKE_PENDING
    ):
        smoke = update_restart_smoke()
        if not smoke.get("ok"):
            ok, msg = rollback_managed_update(
                "recovered_pre_restart_smoke_failed", reopen_writer_admission=False
            )
            _log_supervisor({
                "type": "managed_update_recovered_smoke_failed",
                "ok": ok,
                "msg": msg,
                "smoke": smoke,
            })
            return {"finalized": False, "rolled_back": ok, "msg": msg, "smoke": smoke}
        tx["pre_restart_smoke"] = _PRE_RESTART_SMOKE_PASSED
        write_update_tx(tx)
    if bool(supervisor_ready) and merge_in_head:
        if not _g._clear_update_intent():
            mark_update_tx_gate_blocked("finalize_intent_cleanup_failed")
            return {"finalized": False, "reason": "could not clear update intent"}
        # Q1=C: the boot survived on the merged tree — bring the owner's stashed
        # local work back as uncommitted content BEFORE clearing the tx, so the
        # durable stash_sha pointer survives a crash in this window (a replayed
        # finalize sees "already consumed" and the restored tree stays). A
        # conflicting restore keeps the stash entry and the note discloses the
        # manual recovery command.
        stash_note = restore_stash_with_marker(tx, "boot_finalize")
        if not clear_update_tx():
            mark_update_tx_gate_blocked("finalize_marker_cleanup_failed", cleanup_only=True)
            return {"finalized": False, "reason": "could not clear update marker",
                    **({"stash_note": stash_note} if stash_note else {})}
        _log_supervisor({
            "type": "managed_update_finalized",
            "head": head,
            **({"stash_note": stash_note} if stash_note else {}),
        })
        result = {"finalized": True}
        if stash_note:
            result["stash_note"] = stash_note
        return result
    if (bool(supervisor_ready) and head_resolved and not merge_in_head) or attempts >= 2:
        ok, msg = rollback_managed_update(
            "post_boot_smoke_failed", reopen_writer_admission=False
        )
        _log_supervisor({"type": "managed_update_rollback_after_failed_boot",
                         "ok": ok, "msg": msg, "boot_attempts": attempts})
        return {"finalized": False, "rolled_back": ok, "msg": msg}
    tx["boot_attempts"] = attempts
    write_update_tx(tx)
    return {"finalized": False, "boot_attempts": attempts}


def _recover_assisted_on_boot(tx: Dict[str, Any], supervisor_ready: bool) -> Dict[str, Any]:
    """Recover an in-flight assisted merge after a restart/rescue — re-keyed on MERGE STATE
    (during resolution HEAD == pre_update_sha, the reviewed base) and strictly non-destructive:
    a real reviewed commit that landed on top is NEVER reset away."""
    state = _assisted_head_state(tx)
    if state == "diverged" and str(tx.get("phase") or "") == "materializing_assisted":
        rc_h, head, _error = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
        snapshot = str(tx.get("local_snapshot") or "")
        pre = str(tx.get("pre_update_sha") or "")
        if rc_h == 0 and snapshot and snapshot != pre and head == snapshot:
            ok, msg = rollback_managed_update(
                "assisted_materialization_interrupted", reopen_writer_admission=False
            )
            _log_supervisor({"type": "managed_update_assisted_materialization_interrupted",
                             "ok": ok, "msg": msg})
            return {"finalized": False, "rolled_back": ok, "msg": msg}
    if state == "committed":
        # Only pending_boot_smoke proves that exact-binding verification, post-commit
        # tests, and the pre-restart smoke all passed. A commit found under an assisted
        # phase died inside those gates, so preserve it on failed-update-* and roll back.
        ok, msg = rollback_managed_update(
            "assisted_commit_gates_interrupted", reopen_writer_admission=False
        )
        _log_supervisor({"type": "managed_update_assisted_commit_unproven",
                         "ok": ok, "msg": msg})
        return {"finalized": False, "rolled_back": ok, "msg": msg}
    if state == "diverged":
        # A real reviewed commit landed; keep it (never reset over reviewed work), abandon
        # this update — it is re-planned fresh later. The tx is the ONLY pointer that a
        # stash restore is owed: bring the owner's stashed work back (a conflicting
        # restore keeps the entry and the note discloses it) BEFORE dropping the marker.
        stash_note = restore_stash_with_marker(tx, "boot_diverged_abandon")
        if not clear_update_tx():
            mark_update_tx_gate_blocked("diverged_marker_cleanup_failed", cleanup_only=True)
            return {"finalized": False, "reason": "could not clear diverged update marker",
                    **({"stash_note": stash_note} if stash_note else {})}
        _log_supervisor({"type": "managed_update_assisted_abandoned_diverged",
                         **({"stash_note": stash_note} if stash_note else {})})
        return {"finalized": False, "abandoned": True, "reason": "head_diverged",
                **({"stash_note": stash_note} if stash_note else {})}
    if state == "in_progress":
        attempts = int(tx.get("resolution_attempts") or 0) + 1
        if attempts > _ASSISTED_BOOT_ATTEMPT_CAP:
            ok, msg = rollback_managed_update(
                "assisted_resolution_expired", reopen_writer_admission=False
            )
            _log_supervisor({"type": "managed_update_assisted_expired", "ok": ok, "msg": msg})
            return {"finalized": False, "rolled_back": ok, "msg": msg}
        # Re-establish the merge state if the restart/rescue wiped it; preserve partial
        # progress when MERGE_HEAD + a dirty tree already survived.
        rc_d, dirty, _de = _g.git_capture(["git", "status", "--porcelain"])
        has_progress = bool(_merge_head_sha()) and rc_d == 0 and bool(dirty.strip())
        if has_progress:
            if not tx.get("m0_tree"):
                if str(tx.get("phase") or "") == "materializing_assisted":
                    # The resolver never ran (it is enqueued only after the
                    # materializing phase completes): the live tree IS still the
                    # mechanical merge output. The crash may have hit BETWEEN the
                    # merge and the mandatory projection — re-run the typed
                    # projection (idempotent) before pinning, and fail closed to
                    # a rollback rather than freeze a half-projected baseline.
                    ok_bp, _bp_note, bp_error = project_version_carriers(
                        str(tx.get("target_sha") or "")
                    )
                    if not ok_bp:
                        rb_ok, rb_msg = rollback_managed_update(
                            "backfill_projection_failed", reopen_writer_admission=False
                        )
                        _log_supervisor({"type": "managed_update_backfill_projection_failed",
                                         "error": bp_error, "rollback": rb_msg})
                        return {"finalized": False, "rolled_back": rb_ok, "msg": bp_error}
                    m0_backfill, m0_err = worktree_snapshot_tree("HEAD")
                    if m0_backfill:
                        tx["m0_tree"] = m0_backfill
                    else:
                        tx["m0_tree"] = ""
                        tx["m0_missing_reason"] = f"backfill_snapshot_failed: {m0_err}"
                else:
                    # Resumed after the resolver may have edited: the mechanical
                    # baseline is unrecoverable post-edit — represent the gap
                    # (P1), never fill it in.
                    tx["m0_tree"] = ""
                    tx["m0_missing_reason"] = "resumed_with_progress_before_m0_pin"
            refreshed_conflicts = live_unmerged_paths()
            if refreshed_conflicts is not None:
                tx["conflict_paths"] = refreshed_conflicts
        rescue_info: Dict[str, Any] = {}
        if not has_progress:
            # The server's owner-control path must be resident before the
            # re-materialized conflict markers land in the live tree (#283).
            from supervisor.worker_chat_lane import preload_owner_control_path

            preload_owner_control_path()
            # Re-materialization hard-resets the tree: rescue surviving dirty work; the
            # pointer persists BEFORE materialize and reaches the objective via enqueue.
            rescue_info = _g.rescue_into_tx(
                tx, key="progress_rescue", reason="assisted_rematerialize",
                context="rematerialize", writer=write_update_tx)
            ok, msg, m0_tree = materialize_assisted_merge_live(
                str(tx.get("pre_update_branch") or _g.BRANCH_DEV),
                str(tx.get("local_snapshot") or ""),
                str(tx.get("target_sha") or ""),
                str(tx.get("pre_update_sha") or ""),
            )
            if ok:
                # Re-materialization recomputes the mechanical baseline; the refresh
                # is disclosed via the resumed-event log below (same merge inputs +
                # rerere-neutral flags make it byte-stable in practice).
                tx["m0_tree"] = m0_tree
            if not ok:
                # Could not re-stage the merge — fail closed to a clean pre-update state.
                rb_ok, rb_msg = rollback_managed_update(
                    "assisted_rematerialize_failed", reopen_writer_admission=False
                )
                _log_supervisor({"type": "managed_update_assisted_rematerialize_failed",
                                 "materialize_msg": msg, "rollback": rb_msg})
                return {"finalized": False, "rolled_back": rb_ok, "msg": msg}
        tx["phase"] = "assisted_resolution"
        tx["resolution_attempts"] = attempts
        write_update_tx(tx)
        enqueue_assisted_resolution_task(tx)
        _log_supervisor({"type": "managed_update_assisted_resumed",
                         "resolution_attempts": attempts, "preserved_progress": has_progress,
                         **({"progress_rescue_error": rescue_info["error"]} if rescue_info.get("error") else {})})
        return {"finalized": False, "resumed": True, "resolution_attempts": attempts}
    # unknown: do not touch the tree; leave the tx for the owner / a later boot.
    _log_supervisor({"type": "managed_update_assisted_unknown_state"})
    return {"finalized": False, "reason": "unknown_assisted_state"}


def managed_assisted_tx_for(
    task_id: str, task_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
    """For ``commit_reviewed``: return ``(managed_tx, block_message)``. While a managed assisted
    tx is active, ONLY its authorized resolution task may commit. A CORRUPT marker blocks too
    (fail-closed). Returns ``(tx, "")`` for the authorized task, ``({}, msg)`` to block another
    task, ``({}, "")`` when no managed tx is active. Authorization requires the durable marker
    and the matching immutable fingerprint in host-enqueued task metadata."""
    status, tx = read_update_tx_strict()
    if status == "absent":
        return {}, ""
    if status == "valid" and str(tx.get("phase") or "") in _ASSISTED_PHASES:
        if (
            str(tx.get("task_id") or "") == str(task_id or "")
            and assisted_task_metadata_authorizes(tx, task_metadata)
        ):
            return tx, ""
    elif status == "valid" and str(tx.get("phase") or "") == GATE_BLOCKED_PHASE:
        return {}, (
            "⚠️ MANAGED_UPDATE_GATE_BLOCKED: the last update could not be verified or rolled "
            "back. Commits remain blocked until recovery completes."
        )
    elif status == "valid":
        return {}, (
            "⚠️ MANAGED_UPDATE_IN_PROGRESS: the managed update is awaiting verified recovery "
            "or a healthy restart; repository mutations remain blocked."
        )
    return {}, (
        "⚠️ MANAGED_UPDATE_IN_PROGRESS: a managed update merge is being resolved by another "
        "task (or the update tx is unreadable); commits are blocked until it completes or is "
        "rolled back."
    )


def managed_assisted_precommit_verify(tx: Dict[str, Any]) -> Tuple[bool, str]:
    """Verify the live merge state matches the tx before the reviewed commit: on the expected
    branch, MERGE_HEAD == tx.target_sha, HEAD == tx.pre_update_sha (the reviewed first parent)."""
    branch = str(tx.get("pre_update_branch") or _g.BRANCH_DEV)
    target = str(tx.get("target_sha") or "")
    pre = str(tx.get("pre_update_sha") or "")
    rc_b, cur, _e = _g.git_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc_b != 0 or cur != branch:
        return False, f"⚠️ MANAGED_UPDATE_ERROR: on branch {cur!r}, expected {branch!r}"
    mh = _merge_head_sha()
    if mh != target:
        return False, f"⚠️ MANAGED_UPDATE_ERROR: MERGE_HEAD {(mh[:12] or 'absent')} != target {target[:12]}"
    rc_h, head, _he = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    if rc_h != 0 or head != pre:
        return False, f"⚠️ MANAGED_UPDATE_ERROR: HEAD {head[:12]} != reviewed base {pre[:12]}"
    if (
        "local_work_carrier" in tx
        and not str(tx.get("m0_tree") or "")
        and not str(tx.get("m0_missing_reason") or "")
    ):
        # New-format tx (stash-first) must either carry the pinned mechanical
        # baseline or explicitly represent its absence — never silently lack it.
        return False, "⚠️ MANAGED_UPDATE_ERROR: tx carries neither m0_tree nor m0_missing_reason"
    bible = _g.REPO_DIR / "BIBLE.md"
    if bible.is_symlink() or not bible.is_file() or bible.stat().st_size <= 0:
        return False, "⚠️ MANAGED_UPDATE_ERROR: resolved tree does not preserve BIBLE.md"
    return True, ""


def restore_assisted_resolution_after_commit_error(tx: Dict[str, Any]) -> bool:
    """Return a failed native commit to its retryable phase when merge state is intact."""
    ok, _message = managed_assisted_precommit_verify(tx)
    status, current = read_update_tx_strict()
    if (
        not ok
        or status != "valid"
        or str(current.get("phase") or "") != "committing_assisted"
        or str(current.get("task_id") or "") != str(tx.get("task_id") or "")
    ):
        return False
    current["phase"] = "assisted_resolution"
    write_update_tx(current)
    return True


def reestablish_merge_head(target_sha: str) -> None:
    """Re-write ``.git/MERGE_HEAD`` so a BLOCKED managed-merge review can be fixed and re-committed
    — the review's index reset (``git reset HEAD``) clears the in-progress merge state, after which
    ``managed_assisted_precommit_verify`` would fail on the agent's retry. Best-effort."""
    if not target_sha:
        return
    try:
        (_g._git_dir() / "MERGE_HEAD").write_text(target_sha + "\n", encoding="utf-8")
    except Exception:
        _g.log.warning("reestablish_merge_head failed", exc_info=True)


def managed_assisted_postcommit(tx: Dict[str, Any], commit_sha: str) -> Tuple[bool, str]:
    """After the reviewed 2-parent merge commit lands: record merge_commit + transition to
    ``pending_boot_smoke``, then run the pre-restart smoke INLINE (auto_merge parity). On smoke
    FAIL roll back to pre_update_sha (the agent's resolution survives on the deterministic
    ``failed-update-<target12>`` branch + the rescue snapshot). On PASS the agent calls
    ``request_restart`` and boot finalize verifies the healthy boot. Returns (ok, message)."""
    # Merge-write onto the FRESH durable tx (never the caller's pre-stage-cycle
    # snapshot): in-attempt keys like ``tests_evidence`` must survive this
    # phase transition — see ``update_tx_phase``. The commit already landed, so
    # a CORRUPT marker here skips the write loudly (F1) instead of aborting.
    tx = update_tx_phase_or_keep(tx, {
        "phase": "pending_boot_smoke",
        "merge_commit": commit_sha,
        "pre_restart_smoke": _PRE_RESTART_SMOKE_PENDING,
    })
    smoke = update_restart_smoke()
    if smoke.get("ok"):
        tx = update_tx_phase_or_keep(tx, {"pre_restart_smoke": _PRE_RESTART_SMOKE_PASSED})
        return True, (
            "✅ Managed update committed as a reviewed 2-parent merge and passed the pre-restart "
            "smoke. Call `request_restart` now to finish landing the update."
        )
    ok, msg = rollback_managed_update("assisted_pre_restart_smoke_failed")
    # Preserve the FULL smoke trace durably (it explains why a self-modifying update rolled
    # back — never silently sliced); the chat message shows a head with an explicit omission note.
    _log_supervisor({
        "type": "managed_update_assisted_smoke_failed", "returncode": smoke.get("returncode"),
        "stdout": str(smoke.get("stdout") or ""), "stderr": str(smoke.get("stderr") or ""),
    })
    stderr = str(smoke.get("stderr") or "")
    shown = stderr if len(stderr) <= 400 else (
        stderr[:400] + f"… (+{len(stderr) - 400} more chars — full trace in data/logs/supervisor.jsonl)"
    )
    if not ok:
        mark_update_tx_gate_blocked("assisted_pre_restart_smoke_rollback_failed", msg)
        return False, (
            "⚠️ MANAGED_UPDATE_GATE_BLOCKED: the merged code failed the pre-restart smoke "
            f"({shown}), and rollback could not be verified ({msg}). Restart/recovery is required."
        )
    return False, (
        "⚠️ MANAGED_UPDATE_SMOKE_FAILED: the merged code failed the pre-restart smoke "
        f"({shown}). Rolled back to the prior version ({msg}). "
        "The resolved merge is preserved on the failed-update-* branch named in the "
        "rollback message for inspection and retry."
    )


def abort_orphaned_assisted_tx(
    task_id: str, task_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Watchdog called when a task ENDS: if it was the authorized assisted-resolution task and
    the tx is still mid-resolution (the merge never committed — failed / cancelled / gave up),
    roll back to pre_update_sha so the live worktree AND the commit-exclusivity guard are freed
    immediately (no starvation until a restart). A task ending in the commit crash-window is
    rolled back too because its post-commit tests and smoke were never proven."""
    status, tx = read_update_tx_strict()
    orphan_phases = ("materializing_assisted", "assisted_resolution", "committing_assisted")
    if status != "valid" or str(tx.get("phase") or "") not in orphan_phases:
        return {"acted": False}
    if str(tx.get("task_id") or "") != str(task_id or ""):
        return {"acted": False}
    if not assisted_task_metadata_authorizes(tx, task_metadata):
        return {"acted": False, "reason": "resolver authority mismatch"}
    lock_fh = None
    try:
        try:
            lock_fh = acquire_update_lock()
        except RuntimeError:
            return {"acted": False, "reason": "lock held by an active apply"}
        s2, tx2 = read_update_tx_strict()  # re-read under the lock (it may have just committed)
        if s2 != "valid" or str(tx2.get("phase") or "") not in orphan_phases:
            return {"acted": False}
        if str(tx2.get("task_id") or "") != str(task_id or ""):
            return {"acted": False}
        if not assisted_task_metadata_authorizes(tx2, task_metadata):
            return {"acted": False, "reason": "resolver authority mismatch"}
        if str(tx2.get("phase") or "") == "committing_assisted":
            commit_state = _assisted_head_state(tx2)
            if commit_state not in {"in_progress", "committed"}:
                mark_update_tx_gate_blocked(
                    "assisted_commit_orphan_state_unknown", commit_state
                )
                return {"acted": True, "gate_blocked": True, "state": commit_state}
        ok, msg = rollback_managed_update("assisted_resolution_orphaned")
        _log_supervisor({"type": "managed_update_assisted_orphaned_rollback", "ok": ok, "msg": msg})
        try:
            from supervisor.workers import ensure_worker_pool_started

            if not ensure_worker_pool_started(allow_disabled_restart=True):
                _g.log.warning(
                    "abort_orphaned_assisted_tx: worker pool remains explicitly disabled"
                )
        except Exception:
            _g.log.warning("abort_orphaned_assisted_tx: worker pool start failed", exc_info=True)
        return {"acted": True, "rolled_back": ok, "msg": msg}
    finally:
        if lock_fh is not None:
            release_update_lock(lock_fh)


def _recover_replace_on_boot(tx: Dict[str, Any], supervisor_ready: bool) -> Dict[str, Any]:
    """Resume the narrow crash window between replace rescue and pending smoke."""
    pre = str(tx.get("pre_update_sha") or "")
    target = str(tx.get("target_sha") or "")
    rc_h, head, _head_error = _g.git_capture(["git", "rev-parse", "--verify", "HEAD"])
    rc_s, dirty, _status_error = _g.git_capture(["git", "status", "--porcelain"])
    if rc_h != 0 or rc_s != 0 or not pre or not target:
        ok, msg = rollback_managed_update(
            "replace_apply_state_unreadable", reopen_writer_admission=False
        )
        return {"finalized": False, "rolled_back": ok, "msg": msg}
    applied = head == target and not dirty.strip()
    if applied:
        tx["phase"] = "pending_boot_smoke"
        write_update_tx(tx)
        _log_supervisor({"type": "managed_update_replace_apply_recovered", "head": head})
        return _finalize_pending_boot_smoke(tx, supervisor_ready)
    if head == pre:
        # No target checkout is visible. Disarm replay and leave the current tree alone;
        # the rescue already exists, but a crash before reset must not erase dirty work.
        if not _g._clear_update_intent():
            mark_update_tx_gate_blocked("replace_abandon_intent_cleanup_failed")
            return {"finalized": False, "reason": "could not disarm replace replay"}
        if not clear_update_tx():
            mark_update_tx_gate_blocked("replace_abandon_tx_cleanup_failed", cleanup_only=True)
            return {"finalized": False, "reason": "could not clear replace transaction"}
        _log_supervisor({"type": "managed_update_replace_apply_abandoned", "head": head})
        return {"finalized": False, "abandoned": True, "reason": "replace_not_applied"}
    ok, msg = rollback_managed_update(
        "replace_apply_interrupted", reopen_writer_admission=False
    )
    return {"finalized": False, "rolled_back": ok, "msg": msg}


def finalize_managed_update_on_boot(supervisor_ready: bool = True) -> Dict[str, Any]:
    """Post-boot finalization of a managed update (P2). Called ONCE after the new process
    boots and the supervisor is ready. Acquires the update lock (skips if an apply holds it),
    strict-reads the tx, and dispatches by phase: ``pending_boot_smoke`` (committed +
    restarted) → health-check + boot-loop guard; an assisted phase → non-destructive
    merge-state recovery (resume / abandon-on-divergence / rollback-on-expiry). A CORRUPT
    marker is quarantined aside byte-intact (admission reopens) unless the tree shows an
    in-flight merge, which stays fail-closed; a FUTURE-schema marker recorded by a newer
    release is left untouched (F14: never rolled back by an older binary). This dispatch
    is the stable N−1→N transition contract: an UNSTAMPED (pre-7.0) tx is driven exactly
    like a stamped one. Best-effort; never raises."""
    lock_fh = None
    try:
        try:
            lock_fh = acquire_update_lock()
        except RuntimeError:
            return {"finalized": False, "reason": "update lock held by an active apply"}
        status, tx = read_update_tx_strict()
        if status == "absent":
            return {"finalized": False, "reason": "no pending update"}
        if status == "corrupt":
            # Quarantined aside byte-intact (evidence survives, the admission
            # latch does not) unless MERGE_HEAD shows a live merge (#447);
            # mechanics in update_candidate.quarantine_corrupt_update_tx_marker.
            return quarantine_corrupt_update_tx_marker()
        if status == "future":
            # Recorded by a NEWER release (this process is the rollback
            # target). This version cannot interpret the transaction; rolling
            # it back could destroy work the newer form protects — leave the
            # marker untouched for the owner. It is not corrupt, so it is not
            # quarantined either: the newer binary must still find it.
            from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY

            _log_supervisor({
                "type": "managed_update_tx_future_schema_on_boot",
                "schema_version": tx.get(SCHEMA_VERSION_KEY),
                "phase": str(tx.get("phase") or ""),
            })
            return {"finalized": False,
                    "reason": "update tx recorded by a newer version — left for owner"}
        phase = str(tx.get("phase") or "")
        if phase == "stashing_local_work":
            # Crash between the durable pre-stash marker and the merge apply:
            # nothing was applied yet. Restore whatever the attempt stashed
            # (the tx may predate the stash_sha write), then clear the marker —
            # the owner simply retries the update.
            sha = str(tx.get("stash_sha") or "")
            if not sha:
                ok_lu, sha, _lu_error = lookup_update_stash(str(tx.get("attempt_id") or ""))
                if not ok_lu:
                    # Unreadable stash storage is NOT "nothing was stashed":
                    # clearing the tx here would drop the only durable pointer.
                    _log_supervisor({"type": "managed_update_stash_recovery_unreadable"})
                    return {"finalized": False,
                            "reason": "stash storage unreadable — recovery left for a later boot"}
            stash_note = ""
            if sha:
                tx["stash_sha"] = sha
                stash_note = restore_stash_with_marker(tx, "boot_stash_recovery")
            if not clear_update_tx():
                mark_update_tx_gate_blocked("stash_recovery_marker_cleanup_failed", cleanup_only=True)
                return {"finalized": False, "reason": "could not clear update marker"}
            _log_supervisor({
                "type": "managed_update_stash_recovered_on_boot",
                **({"stash_note": stash_note} if stash_note else {}),
            })
            return {"finalized": False, "reason": "recovered pre-apply stash crash",
                    **({"stash_note": stash_note} if stash_note else {})}
        if phase == "pending_boot_smoke":
            return _finalize_pending_boot_smoke(tx, supervisor_ready)
        if phase == "applying_replace":
            return _recover_replace_on_boot(tx, supervisor_ready)
        if phase == "rolling_back":
            ok, msg = rollback_managed_update(
                str(tx.get("rollback_reason") or "boot_rollback_resume"),
                reopen_writer_admission=False,
            )
            return {"finalized": False, "rolled_back": ok, "msg": msg}
        if phase in _ASSISTED_PHASES:
            return _recover_assisted_on_boot(tx, supervisor_ready)
        if phase == MARKER_CLEANUP_RETRY_PHASE:
            # The repository already holds its final good state; ONLY the
            # marker unlink is retried — never a rollback.
            if clear_update_tx():
                _log_supervisor({"type": "managed_update_cleanup_retry_succeeded",
                                 "reason": str(tx.get("gate_blocked_reason") or "")})
                return {"finalized": True, "reason": "marker cleanup retried"}
            return {"finalized": False, "reason": "marker cleanup still failing"}
        if phase == GATE_BLOCKED_PHASE:
            ok, msg = rollback_managed_update(
                str(tx.get("gate_blocked_reason") or "boot_gate_recovery"),
                reopen_writer_admission=False,
            )
            return {"finalized": False, "rolled_back": ok, "msg": msg}
        return {"finalized": False, "reason": f"unhandled phase {phase}"}
    except Exception:
        _g.log.warning("finalize_managed_update_on_boot failed", exc_info=True)
        return {"finalized": False, "error": "exception"}
    finally:
        if lock_fh is not None:
            release_update_lock(lock_fh)
