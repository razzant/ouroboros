"""What an integration WRITES DOWN: the verdict, and the refusals it can write instead.

A subagent or delegated run ends with a durable verdict file — that is the artifact the
parent reads and the owner audits — and the interesting cases are the ones where there
is no clean verdict to write: the capture failed, the baseline drifted under the patch,
the child was acknowledged but never wrote anything, the disposition is unknown. Each
of those has a typed refusal whose text IS the record.

They live together because they are alternatives to one another. Scattered across the
integration paths, "what does the parent see when this goes wrong" had to be
reconstructed by reading every call site; here the answers sit side by side and can be
compared.
"""

from __future__ import annotations

import logging

import json
import pathlib
from typing import Any, Dict, List
from ouroboros.tools.registry import ToolContext
from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.headless import (
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
)
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)


_READY_CAPTURE_STATUSES = frozenset({
    ARTIFACT_STATUS_READY_WITH_CHANGES, ARTIFACT_STATUS_READY_NO_CHANGES})


def _sha256_file(path: pathlib.Path) -> str:
    from hashlib import sha256

    hasher = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_verdict(
    ctx: ToolContext,
    child_task_id: str,
    *,
    outcome: str,
    reason: str,
    files: List[str],
    manifest: Dict[str, Any],
    applied: bool,
    conflicts: List[str],
    protected: List[str],
    target: str = "",
) -> str:
    parent_task_id = task_id_for_artifacts(ctx)
    art_dir = task_artifact_dir_path(getattr(ctx, "drive_root", "."), parent_task_id, create=True)
    verdict = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "tool": "integrate_subagent_patch",
        "parent_task_id": parent_task_id,
        "child_task_id": child_task_id,
        "outcome": outcome,
        "applied": bool(applied),
        "reason": str(reason or ""),
        "target_root": str(target or ""),
        "files": list(files or []),
        "protected_matches": list(protected or []),
        "conflicts": list(conflicts or []),
        "patch_sha256": str((manifest or {}).get("sha256") or ""),
        "diffstat": str((manifest or {}).get("diffstat") or ""),
    }
    path = art_dir / f"subagent_patch_verdict_{child_task_id}.json"
    try:
        atomic_write_json(path, verdict, trailing_newline=True)
    except Exception:
        return ""
    return str(path)


def _drift_refusal(
    ctx: ToolContext, *, rid: str, reason: str, touched: List[str],
    manifest: Dict[str, Any], protected: List[Any], target: pathlib.Path,
    drifted: List[str], drift_error: str,
) -> str:
    """The typed, nanny-owned refusal for a target that moved (or could not be
    compared) since the run's snapshot. Nothing was applied; material persists."""
    verdict_path = _write_verdict(
        ctx, f"run_{rid}", outcome="baseline_drift" if drifted else "baseline_unverifiable",
        reason=reason, files=touched, manifest=manifest, applied=False,
        conflicts=(drifted[:50] if drifted else [drift_error]),
        protected=[p.path for p in protected], target=str(target),
    )
    if drift_error:
        return (
            f"⚠️ INTEGRATE_DELEGATED_BASELINE_UNVERIFIABLE: run {rid}'s patch was NOT "
            f"applied — its target state could not be compared against the run's "
            f"baseline ({drift_error}). Nothing was changed; the execution snapshot "
            f"and the patch are preserved. Verdict: {verdict_path or '(unwritten)'}."
        )
    return (
        f"⚠️ INTEGRATE_CONFLICT: {len(drifted)} path(s) in {target} CHANGED since run "
        f"{rid}'s snapshot was taken, so its patch was NOT applied (a plain apply "
        f"would relocate hunks and silently land them on moved content). Drifted: "
        f"{', '.join(drifted[:10])}{' …' if len(drifted) > 10 else ''}\n"
        "YOU own this conflict: the execution snapshot and the captured patch are "
        "preserved until you resolve it. Reconcile your tree with the snapshot's "
        "changes (read the patch artifact, or diff against the execution root "
        "directly), then retry the apply, or "
        "integrate_delegated_patch(decision='reject') to discard. "
        f"Verdict: {verdict_path or '(unwritten)'}."
    )


def _manifest_capture_status(manifest_path: pathlib.Path) -> str:
    """The status one capture manifest reports about ITSELF; "" when unreadable."""
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return ""
    return str(loaded.get("status") or "") if isinstance(loaded, dict) else ""


def _capture_failed_refusal(rid: str, cap_status: str, note: str) -> str:
    """The ONE typed refusal for disposing over a non-usable capture (C1-R3).

    Shared by the capture-at-disposition seam and the reject branch's own
    guard, so the two cannot drift in shape. No disposition is recorded by any
    caller of this message: the obligation stays open and the snapshot persists.
    """
    return (
        f"⚠️ INTEGRATE_DELEGATED_CAPTURE_FAILED: run {rid}'s changes were never "
        f"usably captured, and the capture-at-disposition attempt did not "
        f"produce a patch (status {cap_status or 'missing'!r}: {note[:300]}). "
        "No disposition was recorded — the obligation stays open and the "
        "execution snapshot (if present) is preserved. Inspect the snapshot "
        "directly, then retry this call."
    )


def _capture_at_disposition(
    drive: Any, entry: Any, rid: str, manifest_path: pathlib.Path,
) -> str:
    """Capture-on-demand (C1-R2) for a run that settled without terminal proof.

    A run reconciled while its daemon was absent or unreadable settled WITHOUT a
    capture — its state was unknowable then, so nothing was frozen. By disposition
    time the stale child is as settled as it will ever be, so this is the honest
    latest-possible capture point, through the SAME drive-rooted core the sweep
    and the nanny use. Returns "" when a usable capture exists (pre-existing or
    fresh); a capture that fails here is a typed refusal, because disposing over
    nothing would silently discard (reject) or skip (apply) work that still sits
    in the snapshot.

    The early return trusts ``patch_captured`` only together with the manifest's
    OWN ready status (C1-R3): a durable row minted by pre-R3 code over a failed
    manifest replays as not-captured, and the core is asked to re-capture. An
    exception ESCAPING the core (its internal try covers only the diff itself)
    is the same typed refusal, never a raw traceback out of the tool.
    """
    if entry.patch_captured and _manifest_capture_status(manifest_path) in _READY_CAPTURE_STATUSES:
        return ""
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    try:
        block = capture_terminal_patch_for_drive(drive, entry) or {}
    except Exception as exc:
        log.warning("capture-at-disposition raised for run %s", rid, exc_info=True)
        block = {"status": "", "note": f"capture raised {type(exc).__name__}: {exc}"}
    cap_status = str(block.get("status") or "")
    if cap_status in _READY_CAPTURE_STATUSES:
        return ""
    return _capture_failed_refusal(rid, cap_status, str(block.get("note") or ""))


def _delegated_disposition_refusal(status: str, entry: Any, rid: str,
                                   acknowledge_ambiguous: bool = False) -> str:
    """The early typed refusals of `integrate_delegated_patch`, one author.

    Ownership, isolation, finality, terminality, and the CR1-3 apply-intent
    ambiguity — every answer that needs no capture and mutates nothing.
    Returns "" when the disposition may proceed. ``acknowledge_ambiguous``
    (CR2-1) waives ONLY the ambiguity refusal: the caller then resolves the
    stale intent durably and runs the normal disposition guards from scratch.
    """
    from ouroboros import delegate_custody as custody

    if status != custody.OWNED or entry is None:
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_OWNED: run {rid!r} is {status} to this task. "
            "Only the task that started a delegated run may integrate its patch."
        )
    if not entry.execution_root:
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_ISOLATED: run {rid} recorded no execution "
            "snapshot (read-only, or a pre-isolation run). There is no captured patch "
            "to integrate."
        )
    if entry.patch_disposed:
        return (
            f"⚠️ INTEGRATE_DELEGATED_ALREADY_DISPOSED: run {rid}'s patch was already "
            f"{entry.patch_disposed}. A disposition is final; nothing was changed."
        )
    if not entry.settled:
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_TERMINAL: run {rid} has not settled yet. "
            "delegate_wait it to terminal first — its patch is captured there."
        )
    if entry.patch_apply_pending and not acknowledge_ambiguous:
        # CR1-3: a durable apply-intent row exists with no resolution and no
        # disposition — a previous process started the apply and died before
        # recording the outcome. The tree MAY carry the patch, so BOTH decisions
        # refuse: a reject here would record "not applied" over a possibly
        # modified, staged tree, and an apply could land the same changes twice.
        # CR2-1: the refusal names its own exit — an explicit owner
        # acknowledgment re-enters the NORMAL flow, whose guards re-verify.
        return (
            f"⚠️ INTEGRATE_DELEGATED_APPLY_AMBIGUOUS: a durable apply-intent row "
            f"exists for run {rid} but no completed disposition — a previous "
            f"process may have applied this patch into {entry.target_root} before "
            "dying. The tree MAY already carry the run's changes: inspect it "
            "(vcs_diff, compare against the patch artifact), then re-run "
            "integrate_delegated_patch with acknowledge_ambiguous=true to take "
            "the state over explicitly — that resolves the stale intent and runs "
            "the NORMAL disposition from scratch (an apply re-verifies baseline "
            "drift and honestly refuses a tree that already moved; a reject "
            "releases the snapshot while the captured patch artifact is "
            "retained). Nothing was changed now; the execution snapshot and the "
            "captured patch are preserved."
        )
    return ""


def _unwritten_disposition_text(rid: str, target_root: str, disposition: str,
                                applied: bool) -> str:
    """The typed refusal for a completed operation whose row did not land."""
    if applied:
        return (
            f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {rid}'s patch IS APPLIED and "
            f"staged in {target_root}, but the durable disposition row could not be "
            "written, so nothing on disk records that this patch was handled. Do "
            "NOT call integrate_delegated_patch again for this run — a second "
            "apply would land the same changes twice. Fix the drive/event log, "
            "then verify with vcs_diff and record the outcome; the execution "
            "snapshot is deliberately preserved."
        )
    return (
        f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {rid}'s patch was NOT applied "
        f"(disposition {disposition!r}), and the durable disposition row could not "
        "be written. Nothing in your tree changed and the execution snapshot is "
        "preserved. Fix the drive/event log; this process already holds the "
        "disposition in memory (a repeat here answers ALREADY_DISPOSED), so the "
        "run reads as undisposed again only after a restart — repeat it then."
    )


def _resolve_acknowledged_intent(drive: Any, entry: Any) -> None:
    """CR2-1: the owner explicitly took over the AMBIGUOUS crash state.

    Resolves the stale apply intent durably as owner-acknowledged; the caller
    then runs the NORMAL disposition flow from scratch — apply re-proves
    baseline drift, reject re-runs the ready-manifest guard. A failed row
    write only means a post-restart replay is ambiguous again, which is the
    fail-closed direction (see ``record_patch_apply_resolved``).
    """
    from ouroboros import delegate_custody as custody

    custody.record_patch_apply_resolved(drive, entry, reason="owner_acknowledged")


__all__ = [
    "_READY_CAPTURE_STATUSES",
    "_sha256_file",
    "_write_verdict",
    "_drift_refusal",
    "_manifest_capture_status",
    "_capture_failed_refusal",
    "_capture_at_disposition",
    "_delegated_disposition_refusal",
    "_unwritten_disposition_text",
    "_resolve_acknowledged_intent",
]
