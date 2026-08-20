"""``integrate_delegated_patch``: the C1 delegated-run disposition seam.

The typed early refusals (ownership, isolation, finality, the CR1-3 apply-intent
ambiguity), capture-at-disposition, the durable dispose/resolve writers, the
drift refusal, the locked working-tree apply and the disposition flow itself.
Extracted from ``ouroboros/tools/subagent_integration.py`` at its size gate
(v7 DEL1 split); ``tools.subagent_integration`` re-exports every name (same
objects), so sibling code — ``delegate_integration``'s payload branch included —
the tests and monkeypatch targets keep addressing them on THAT surface.
"""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
from typing import Any, Dict, List

from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.headless import (
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
)
from ouroboros.review_state import invalidate_advisory_after_mutation
from ouroboros.runtime_mode_policy import (
    mode_allows_protected_write,
    protected_paths_in,
    protected_write_block_message,
)
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE
from ouroboros.tools.registry import ToolContext

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.tools.subagent_integration")


def _si():
    """The parent integration-tool module, read at call time.

    The integration members stay monkeypatch-addressable at their historical
    ``ouroboros.tools.subagent_integration`` bindings (tests rebind them
    there), so this leaf resolves every cross-reference through the module at
    each call instead of freezing whatever object a from-import saw at import
    time.
    """
    from ouroboros.tools import subagent_integration

    return subagent_integration


# The capture statuses a disposition may proceed over (C1-R3): a usable patch
# exists (with changes), or the run provably changed nothing. Everything else —
# failed, missing, unreadable — must never be applied over or release a snapshot.
_READY_CAPTURE_STATUSES = frozenset({
    ARTIFACT_STATUS_READY_WITH_CHANGES, ARTIFACT_STATUS_READY_NO_CHANGES})


def _drift_refusal(
    ctx: ToolContext, *, rid: str, reason: str, touched: List[str],
    manifest: Dict[str, Any], protected: List[Any], target: pathlib.Path,
    drifted: List[str], drift_error: str,
) -> str:
    """The typed, nanny-owned refusal for a target that moved (or could not be
    compared) since the run's snapshot. Nothing was applied; material persists."""
    verdict_path = _si()._write_verdict(
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


def _locked_apply(
    ctx: ToolContext, target: pathlib.Path, patch_path: pathlib.Path,
    ordered_touched: List[str], baseline_sha: str,
) -> Dict[str, Any]:
    """Apply one captured patch under the repo git lock — mechanics only.

    Returns the FACTS the caller turns into a verdict: ``proc`` (None when
    nothing was attempted), ``drifted``/``drift_error`` from the pre-apply
    baseline comparison, and ``staging_failure``/``reverted`` for an apply whose
    staging failed. No verdicts, no dispositions, no messages — those belong to
    the one caller so every exit stays visible in one place.
    """
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock

    result: Dict[str, Any] = {
        "proc": None, "drifted": [], "drift_error": "",
        "staging_failure": "", "reverted": False, "lock_error": "",
    }
    try:
        _git_lock = _acquire_git_lock(ctx)
    except Exception as exc:
        result["lock_error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        # DRIFT IS PROVEN, NOT INFERRED. `git apply` relocates hunks by offset, so a
        # target that moved since the snapshot can still take the patch — at a
        # shifted position, silently. Under the same lock that serializes the
        # mutation, every touched path is compared against the run's baseline commit
        # first; ANY difference is the typed conflict the nanny owns, and nothing is
        # applied.
        try:
            result["drifted"], result["drift_error"] = _si()._baseline_drifted_paths(
                target, baseline_sha, ordered_touched)
        except Exception as exc:
            result["drifted"], result["drift_error"] = [], f"{type(exc).__name__}: {exc}"
        if result["drift_error"] or result["drifted"]:
            return result
        # WORKING-TREE apply, not --3way/--index: the baseline deliberately
        # snapshots the target's DIRTY state (that is the whole point of C1), so the
        # patch's preimage is the live working tree — while `--3way` implies index
        # binding and refuses any file whose worktree differs from the index, i.e.
        # refuses the normal shared-tree state. Touched paths are then staged
        # explicitly so the result matches integrate_subagent_patch's staged contract.
        proc = subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=str(target), capture_output=True, text=True,
        )
        result["proc"] = proc
        stageable = _si()._stageable_paths(target, ordered_touched) if proc.returncode == 0 else []
        if proc.returncode != 0 or not stageable:
            return result
        # NUL-delimited stdin pathspecs: byte-safe and immune to argv limits for a
        # patch touching thousands of files.
        add = subprocess.run(
            ["git", "add", "--pathspec-from-file=-", "--pathspec-file-nul"],
            cwd=str(target), capture_output=True,
            input=b"\0".join(
                p.encode("utf-8", errors="surrogateescape") for p in stageable) + b"\0",
        )
        if add.returncode == 0:
            return result
        # The APPLY SUCCEEDED — the tree is already mutated. Reporting this as a
        # conflict ("the tree moved") invited a retry over changed content and let a
        # later reject record "not applied" over real changes. Try to put the tree
        # back cleanly; whatever the outcome, the caller says exactly what is true.
        result["staging_failure"] = (add.stderr or add.stdout or b"").decode(
            "utf-8", errors="replace").strip()
        check = subprocess.run(
            ["git", "apply", "--check", "--reverse", str(patch_path)],
            cwd=str(target), capture_output=True, text=True,
        )
        if check.returncode == 0:
            result["reverted"] = subprocess.run(
                ["git", "apply", "--reverse", str(patch_path)],
                cwd=str(target), capture_output=True, text=True,
            ).returncode == 0
        return result
    finally:
        _release_git_lock(_git_lock)


def _manifest_capture_status(manifest_path: pathlib.Path) -> str:
    """The status one capture manifest reports about ITSELF; "" when unreadable."""
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return ""
    return str(loaded.get("status") or "") if isinstance(loaded, dict) else ""


def _capture_failed_refusal(rid: str, cap_status: str, note: str) -> str:
    """The ONE typed refusal for disposing over a non-usable capture (C1-R3).
    Shared by the capture-at-disposition seam and the reject branch's own guard
    so they cannot drift; no caller records a disposition over this message."""
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
        # CR1-3: a durable apply-intent row with no resolution/disposition — a
        # previous process started the apply and died. The target MAY carry the
        # patch, so BOTH decisions refuse (a reject would record "not applied"
        # over a possibly-mutated target; an apply could land changes twice).
        # CR2-1: the explicit owner acknowledgment re-enters the NORMAL flow.
        inspect_hint = (
            "compare the live payload content against the patch artifact"
            if getattr(entry, "authority_source", "") == "skill_payload"
            else "vcs_diff, compare against the patch artifact")
        return (
            f"⚠️ INTEGRATE_DELEGATED_APPLY_AMBIGUOUS: a durable apply-intent row "
            f"exists for run {rid} but no completed disposition — a previous "
            f"process may have applied this patch into {entry.target_root} before "
            "dying. The target MAY already carry the run's changes: inspect it "
            f"({inspect_hint}), then re-run "
            "integrate_delegated_patch with acknowledge_ambiguous=true to take "
            "the state over explicitly — that resolves the stale intent and runs "
            "the NORMAL disposition from scratch (an apply re-verifies baseline "
            "drift and honestly refuses a target that already moved; a reject "
            "releases the snapshot while the captured patch artifact is "
            "retained). Nothing was changed now; the execution snapshot and the "
            "captured patch are preserved."
        )
    return ""


def _unwritten_disposition_text(rid: str, target_root: str, disposition: str,
                                applied: bool, *, payload: bool = False) -> str:
    """The typed refusal for a completed operation whose row did not land.
    ``payload`` selects accurate apply wording (Sol P2-3): a payload apply lands
    LIVE in the non-Git payload — nothing staged, no index for vcs_diff."""
    if applied:
        landed, verify = (
            (f"applied LIVE into the non-Git payload {target_root}",
             "compare the live payload against the patch artifact") if payload
            else (f"applied and staged in {target_root}", "verify with vcs_diff"))
        return (
            f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {rid}'s patch IS {landed}, "
            "but the durable disposition row could not be "
            "written, so nothing on disk records that this patch was handled. Do "
            "NOT call integrate_delegated_patch again for this run — a second "
            "apply would land the same changes twice. Fix the drive/event log, "
            f"then {verify} and record the outcome; the execution "
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


def _dispose_delegated(drive: Any, entry: Any, snapshot_key: str, reason: str,
                       disposition: str, cleanup: bool) -> tuple[bool, str]:
    """Record a delegated disposition durably; clean up ONLY if the row landed.
    Releasing snapshot/patch on an UNWRITTEN row loses the record that the patch
    was handled (a restart could apply it twice). Shared by the Git and payload
    branches (``delegate_integration``). Returns ``(recorded, note)``."""
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_worktrees import remove_execution_snapshot

    recorded = custody.record_patch_disposed(
        drive, entry, disposition=disposition, reason=str(reason or ""))
    if not recorded:
        return False, ""
    note = ""
    if cleanup:
        try:
            removed = remove_execution_snapshot(snapshot_key)
        except Exception:
            removed = False
        note = "" if removed else " (snapshot cleanup deferred to the startup GC.)"
    return True, note


def _resolve_acknowledged_intent(drive: Any, entry: Any) -> None:
    """CR2-1: the owner explicitly took over the AMBIGUOUS crash state.
    Resolves the stale apply intent durably as owner-acknowledged; the caller
    re-runs the NORMAL disposition from scratch (apply re-proves drift, reject
    re-runs the ready-manifest guard). A failed row write only makes a
    post-restart replay ambiguous again — the fail-closed direction."""
    from ouroboros import delegate_custody as custody

    custody.record_patch_apply_resolved(drive, entry, reason="owner_acknowledged")


def _integrate_delegated_patch(
    ctx: ToolContext,
    run_id: str = "",
    decision: str = "apply",
    reason: str = "",
    acknowledge_ambiguous: bool = False,
) -> str:
    """The C1 explicit acceptance seam: apply or reject ONE delegated run's captured patch.

    A mutating delegated run executed in a PRIVATE execution snapshot; its diff was
    captured at terminal. NOTHING reaches the shared tree automatically — this tool is
    the only path in, it targets the run's recorded authority target (which must be
    this task's own active root), and under the repo git lock proves no touched path
    drifted from the run's baseline, applies the patch to the WORKING TREE (a plain
    `git apply`, deliberately NOT --3way — see `_locked_apply`: --3way implies index
    binding and refuses the normal dirty shared-tree state the baseline deliberately
    snapshots), stages the touched paths explicitly, and records the disposition
    durably. Only a recorded disposition releases the snapshot for cleanup; a
    conflict keeps the snapshot and the patch as resolution material.
    ``acknowledge_ambiguous`` (CR2-1) is the owner exit from the AMBIGUOUS
    crash state: it resolves the stale intent durably and re-runs this normal
    flow, whose own guards re-verify the tree. A no-op when nothing is pending.
    """
    from ouroboros import delegate_custody as custody

    rid = str(run_id or "").strip()
    if not rid:
        return "⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): run_id is required."
    decision = str(decision or "apply").strip().lower()
    if decision not in {"apply", "reject"}:
        return "⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): decision must be 'apply' or 'reject'."
    drive = custody.custody_root(ctx)
    status, entry = custody.lookup(drive, str(getattr(ctx, "task_id", "") or ""), rid)
    refusal = _delegated_disposition_refusal(status, entry, rid, acknowledge_ambiguous)
    if refusal:
        return refusal
    if entry.patch_apply_pending and acknowledge_ambiguous:
        _resolve_acknowledged_intent(drive, entry)
    snapshot_key = entry.snapshot_id or entry.run_id
    cap_dir = custody.delegated_capture_dir(drive, entry.task_id, snapshot_key)
    manifest_path = cap_dir / "workspace_patch.json"
    patch_path = cap_dir / "workspace.patch"
    capture_refusal = _si()._capture_at_disposition(drive, entry, rid, manifest_path)
    if capture_refusal:
        return capture_refusal
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"⚠️ INTEGRATE_MANIFEST_UNREADABLE: {manifest_path}: {type(exc).__name__}: {exc}."
    if entry.authority_source == "skill_payload" or str(manifest.get("capture_kind") or "") == "skill_payload":
        # The exact-payload branch (R1 item 3) lives in delegate_integration:
        # fresh semantic rebinding, whole-payload CAS, reserved-path whole-apply
        # refusal, index-free apply into the live NON-Git payload — no staging.
        from ouroboros.tools.delegate_integration import integrate_payload_patch

        return integrate_payload_patch(
            ctx, drive=drive, entry=entry, rid=rid, decision=decision,
            reason=reason, cap_dir=cap_dir, manifest=manifest, patch_path=patch_path)
    touched = [str(p) for p in (manifest.get("tracked_changed") or [])]
    touched += [str(p) for p in (manifest.get("untracked_included") or [])]

    def _dispose(disposition: str, cleanup: bool) -> tuple[bool, str]:
        return _dispose_delegated(drive, entry, snapshot_key, reason, disposition, cleanup)

    def _unwritten_disposition(disposition: str, applied: bool) -> str:
        return _unwritten_disposition_text(rid, str(entry.target_root), disposition, applied)

    capture_status = str(manifest.get("status") or "")
    if decision == "reject":
        # A reject RELEASES the snapshot (the child's only copy): ready-only.
        if capture_status not in _READY_CAPTURE_STATUSES:
            return _capture_failed_refusal(
                rid, capture_status, "a reject would release the snapshot over it")
        verdict_path = _si()._write_verdict(
            ctx, f"run_{rid}", outcome="rejected", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=[], protected=[],
            target=str(entry.target_root),
        )
        recorded, note = _dispose("rejected", cleanup=True)
        if not recorded:
            return _unwritten_disposition("rejected", applied=False)
        return (
            f"🚫 Rejected delegated run {rid}'s captured patch ({len(touched)} file(s) not "
            f"applied); its execution snapshot is released. Verdict: {verdict_path or '(unwritten)'}. "
            f"Reason: {reason or '(none)'}.{note}"
        )

    if capture_status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        if capture_status == ARTIFACT_STATUS_READY_NO_CHANGES:
            recorded, note = _dispose("applied", cleanup=True)
            if not recorded:
                return _unwritten_disposition("applied", applied=False)
            return (
                f"OK: delegated run {rid} changed NOTHING in its execution snapshot; "
                f"there is no patch to apply and the snapshot is released.{note}"
            )
        return (
            f"⚠️ INTEGRATE_DELEGATED_NO_CAPTURE: run {rid}'s capture status is "
            f"{capture_status or 'missing'!r} — no applicable patch. If the run just "
            "ended, delegate_wait it once more to capture; a failed capture keeps the "
            "snapshot for direct inspection."
        )
    if not patch_path.exists():
        return f"⚠️ INTEGRATE_PATCH_MISSING: captured patch not found at {patch_path}."
    expected_digest = str(manifest.get("sha256") or "")
    if expected_digest:
        actual_digest = _si()._sha256_file(patch_path)
        if actual_digest != expected_digest:
            return (
                f"⚠️ INTEGRATE_PATCH_CORRUPT: sha256 mismatch for run {rid} "
                f"(manifest {expected_digest[:12]} != file {actual_digest[:12]}); refusing to apply."
            )

    # The target is the run's RECORDED authority target, and it must be THIS task's
    # own active root — the nanny integrates into its own tree, never across trees.
    try:
        active_root = pathlib.Path(ctx.active_repo_dir()).resolve(strict=False)
    except Exception as exc:
        return f"⚠️ INTEGRATE_TARGET_ERROR: could not resolve active repo: {type(exc).__name__}: {exc}."
    target = pathlib.Path(str(entry.target_root or "")).resolve(strict=False)
    if not str(entry.target_root or "").strip() or target != active_root:
        return (
            "⚠️ INTEGRATE_DELEGATED_TARGET_MISMATCH: the run's recorded authority target "
            f"({entry.target_root or '(none)'}) is not this task's active root ({active_root}). "
            "Refusing to apply across trees."
        )
    if not (target / ".git").exists():
        return f"⚠️ INTEGRATE_TARGET_NOT_GIT: target {target} is not a git working tree."

    patch_touched, parse_error = _si()._patch_touched_paths(patch_path, target)
    if parse_error:
        return (
            f"⚠️ INTEGRATE_PATCH_UNREADABLE: cannot parse run {rid}'s captured patch "
            f"(git apply --numstat failed): {parse_error[:300]}"
        )
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    is_acting = bool(constraint and getattr(constraint, "mode", "") == ACTING_SUBAGENT_MODE)
    runtime_mode = _si().get_runtime_mode()
    # The protected-path policy is about the OUROBOROS body's own invariants, so it
    # is asked only when the target IS that body (or a self_worktree checkout of
    # it). A foreign project's `.github/workflows/ci.yml` or `build.sh` is that
    # project's file: gating it here would block the root-delegation lane (B5) with
    # advice about a runtime mode that does not govern that repository — the same
    # reason `_handle_external_workspace_integration` never applies this gate.
    protected = (
        protected_paths_in(sorted(patch_touched)) if _si()._target_is_system_repo(ctx) else []
    )
    if protected:
        grant_ok = (not is_acting) or bool(getattr(constraint, "protected_paths_grant", False))
        if not (mode_allows_protected_write(runtime_mode) and grant_ok):
            _si()._write_verdict(
                ctx, f"run_{rid}", outcome="blocked_protected", reason=reason, files=touched,
                manifest=manifest, applied=False, conflicts=[],
                protected=[p.path for p in protected], target=str(target),
            )
            return protected_write_block_message(
                path=protected[0].path,
                runtime_mode=runtime_mode,
                action=f"integrate delegated run {rid} patch touching",
            )

    ordered_touched = sorted(patch_touched)
    # CR1-3, owed-before-sent: the durable apply-intent row lands BEFORE the tree
    # can be mutated. Without it, a crash between `git apply` and the disposition
    # row replays as "never applied", and a later reject records a false rejection
    # over a modified, staged tree. An unlanded intent row refuses the mutation
    # outright (same doctrine as record_start_requested).
    if not custody.record_patch_apply_started(drive, entry, target_root=str(target)):
        return (
            f"⚠️ INTEGRATE_INTENT_UNWRITTEN: the durable apply-intent row for run "
            f"{rid} could not be written, so a crash mid-apply would leave the "
            "tree state unaccountable. Refusing to mutate; fix the drive/event "
            "log and retry. Nothing was changed."
        )
    outcome = _si()._locked_apply(ctx, target, patch_path, ordered_touched, entry.baseline_sha)
    if outcome.get("lock_error"):
        custody.record_patch_apply_resolved(drive, entry, reason="lock_error")
        return (
            "⚠️ INTEGRATE_LOCK_TIMEOUT: could not acquire the repo git lock: "
            f"{outcome['lock_error']}."
        )
    proc = outcome["proc"]
    drifted = outcome["drifted"]
    drift_error = outcome["drift_error"]
    staging_failure = outcome["staging_failure"]
    reverted = outcome["reverted"]

    if proc is None:
        # Nothing was attempted (proven drift / unverifiable baseline): the tree
        # is unmutated, so the intent resolves and the retry lane stays open.
        custody.record_patch_apply_resolved(drive, entry, reason="baseline_drift")
        return _drift_refusal(
            ctx, rid=rid, reason=reason, touched=touched, manifest=manifest,
            protected=protected, target=target, drifted=drifted, drift_error=drift_error,
        )

    if staging_failure:
        if reverted:
            # Cleanly reversed: the tree is PROVABLY back to pre-apply, so the
            # intent resolves BEFORE the verdict write — `_write_verdict` can
            # raise (artifact-dir mkdir), and a stranded pending intent over a
            # non-mutated tree would wedge the run into AMBIGUOUS (CR2-3).
            custody.record_patch_apply_resolved(drive, entry, reason="apply_reverted")
        outcome = "applied_unstaged_reverted" if reverted else "applied_unstaged"
        verdict_path = _si()._write_verdict(
            ctx, f"run_{rid}", outcome=outcome, reason=reason, files=touched,
            manifest=manifest, applied=not reverted, conflicts=[staging_failure[:500]],
            protected=[p.path for p in protected], target=str(target),
        )
        if reverted:
            return (
                f"⚠️ INTEGRATE_APPLIED_UNSTAGED: run {rid}'s patch applied cleanly into "
                f"{target} but STAGING it failed ({staging_failure[:300]}), so the apply "
                "was reversed — your tree is back to its pre-apply state and NOTHING is "
                "left half-applied. The snapshot and the patch are preserved; fix the "
                f"index problem, then call this tool again. Verdict: {verdict_path or '(unwritten)'}."
            )
        recorded, _ = _dispose("applied", cleanup=False)
        tail = "" if recorded else (
            " ⚠️ the durable disposition row could ALSO not be written — record this "
            "outcome yourself before any further integration attempt."
        )
        return (
            f"⚠️ INTEGRATE_APPLIED_UNSTAGED: run {rid}'s patch IS APPLIED in {target} "
            f"({len(touched)} file(s)) but could NOT be staged ({staging_failure[:300]}), "
            "and the apply could not be cleanly reversed. Do NOT retry this call — the "
            "changes are already in your working tree and a second apply would double "
            "them. Inspect with vcs_diff, stage what you accept yourself, and note that "
            "the run is recorded as applied. Its execution snapshot is preserved for "
            f"comparison. Verdict: {verdict_path or '(unwritten)'}.{tail}"
        )

    if proc.returncode != 0:
        # `git apply` is atomic (all-or-nothing without --reject): a non-zero exit
        # means the tree is unmutated, so the intent resolves and retry stays open.
        custody.record_patch_apply_resolved(drive, entry, reason="apply_failed")
        stderr = (proc.stderr or proc.stdout or "").strip()
        conflicts = [ln.strip() for ln in stderr.splitlines()
                     if "conflict" in ln.lower() or "patch failed" in ln.lower()]
        verdict_path = _si()._write_verdict(
            ctx, f"run_{rid}", outcome="conflict", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=conflicts or [stderr[:500]],
            protected=[p.path for p in protected], target=str(target),
        )
        return (
            f"⚠️ INTEGRATE_CONFLICT: applying run {rid}'s patch into {target} did not "
            f"apply cleanly. git said: {stderr[:600]}\n"
            "YOU own this conflict: the execution snapshot and the captured patch are "
            "preserved until you resolve it. Reconcile your tree with the snapshot's "
            "changes (read the patch artifact, or diff against the execution root "
            "directly), retry the apply, or "
            "integrate_delegated_patch(decision='reject') to discard. "
            f"Verdict: {verdict_path or '(unwritten)'}."
        )

    try:
        invalidate_advisory_after_mutation(
            pathlib.Path(getattr(ctx, "drive_root", ".")),
            mutation_root=target,
            changed_paths=touched,
            source_tool="integrate_delegated_patch",
        )
    except Exception:
        pass
    verdict_path = _si()._write_verdict(
        ctx, f"run_{rid}", outcome="applied", reason=reason, files=touched,
        manifest=manifest, applied=True, conflicts=[],
        protected=[p.path for p in protected], target=str(target),
    )
    recorded, note = _dispose("applied", cleanup=True)
    if not recorded:
        return _unwritten_disposition("applied", applied=True)
    diffstat = str(manifest.get("diffstat") or "").strip()
    prot_note = ""
    if protected:
        prot_note = f" Includes {len(protected)} protected path(s) (allowed: runtime_mode={runtime_mode})."
    return (
        f"✅ Integrated delegated run {rid}'s patch into {target} ({len(touched)} file(s), staged).{prot_note}\n"
        f"{diffstat}\n"
        f"Verdict: {verdict_path or '(unwritten)'}. Its execution snapshot is released.\n"
        "Changes are staged but NOT committed — review them yourself; you are the sole committer."
        f"{note}"
    )
