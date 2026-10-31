"""``integrate_delegated_patch``: the C1 delegated-run disposition seam.

Owns typed CR1-3 refusals, capture-at-disposition, dispose/resolve records, drift refusal
and locked apply. Extracted at the v7 DEL1 size gate: ``subagent_integration`` re-exports
the same objects, retaining sibling imports and monkeypatch targets on that surface.
"""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
from contextlib import ExitStack
from typing import Any, Callable, Dict, List

from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.headless import (
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
)
from ouroboros.review_state import invalidate_advisory_after_mutation
from ouroboros.delegate_named_evidence import (
    retain_named as _retain_named_evidence, run_named_evidence,
)
from ouroboros.runtime_mode_policy import (
    mode_allows_protected_write,
    protected_paths_in,
    protected_write_block_message,
)
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE
from ouroboros.workspace_patch_rules import format_patch_exclusions
from ouroboros.tools.registry import ToolContext

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.tools.subagent_integration")


def _si():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros.tools import subagent_integration

    return subagent_integration


_READY_CAPTURE_STATUSES = frozenset({
    ARTIFACT_STATUS_READY_WITH_CHANGES, ARTIFACT_STATUS_READY_NO_CHANGES})


def _saved_manifest(manifest_path: pathlib.Path) -> Dict[str, Any]:
    """One capture manifest as saved; {} when absent or unreadable."""
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _manifest_capture_status(manifest_path: pathlib.Path) -> str:
    """The status one capture manifest reports about ITSELF; "" when unreadable."""
    return str(_saved_manifest(manifest_path).get("status") or "")


def _drifted_no_change(manifest: Dict[str, Any]) -> bool:
    """A failed capture whose only private fact is its authority-drift record over a snapshot known to
    hold no child changes: a reject discards nothing, so a neighboring edit cannot wedge custody."""
    drift = manifest.get("authority_drift")
    return (manifest.get("status") == "failed" and isinstance(drift, dict)
            and manifest.get("authority_drift_source_status") == ARTIFACT_STATUS_READY_NO_CHANGES
            and bool(drift.get("paths") or drift.get("error")))


def _capture_failed_refusal(rid: str, cap_status: str, note: str, step: str = "") -> str:
    """The ONE typed refusal for disposing over a non-usable capture (C1-R3).
    Shared by the capture-at-disposition seam and the reject branch's own guard
    so they cannot drift; no caller records a disposition over this message. Its
    next step follows the cause (#1651): a blind repeat heals only a transient one."""
    return (
        f"⚠️ INTEGRATE_DELEGATED_CAPTURE_FAILED: run {rid}'s changes were never "
        f"usably captured, and the capture-at-disposition attempt did not "
        f"produce a patch (status {cap_status or 'missing'!r}: {note[:600]}). "
        "No disposition was recorded — the obligation stays open and the private copy (if present) is "
        "kept. " + (step or "Repeating this call fails the same way until that cause changes; once it has, call again.")
    )


def prepared_capture_inspection(rid: str, decision: str, applies_changes: bool,
                                describe: Callable[[], Dict[str, Any]], *, prepared: bool) -> str:
    """A capture this call prepared, or one that finished while it waited for the lock, was
    never available for inspection: an apply of its changes returns it instead (#1969). A
    capture ready before the call keeps the ordinary contract (its caller decides); this proves
    no reading of it. "" lets the call proceed; nothing here applies, records or releases.
    """
    if decision != "apply" or not applies_changes or not prepared:
        return ""
    return (f"⚠️ INTEGRATE_DELEGATED_INSPECT_CAPTURE: run {rid}'s captured result was NOT applied: it was "
            "prepared during this call, so it was not available for inspection. Read it (manifest_read, its "
            "file artifacts, and patch_read when present); a later apply decides this same retained capture. "
            "No target files changed; the capture stays retained.\n"
            + json.dumps(describe(), ensure_ascii=False, indent=2))


def _capture_at_disposition(
    drive: Any, entry: Any, rid: str, manifest_path: pathlib.Path, decision: str = "apply",
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

    A ``missing`` capture (the core first restores what a pin holds) is loss only on
    ``result_loss`` proof (#1651): a reject passes to its branch, which records the loss,
    and an apply is told that reject is the exit; an unproven loss names its carrier.
    """
    reject = str(decision or "").lower() == "reject"
    if (entry.patch_captured and _manifest_capture_status(manifest_path) in _READY_CAPTURE_STATUSES
            or reject and _drifted_no_change(_saved_manifest(manifest_path))):
        return ""
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive, loss_refusal, result_loss

    try:
        block = capture_terminal_patch_for_drive(drive, entry) or {}
    except Exception as exc:
        log.warning("capture-at-disposition raised for run %s", rid, exc_info=True)
        block = {"status": "", "note": f"capture raised {type(exc).__name__}: {exc}"}
    cap_status, note = str(block.get("status") or ""), str(block.get("note") or "")
    step = "A later call may capture it." if block.get("retry") == "transient" else ""
    if cap_status in _READY_CAPTURE_STATUSES:
        return ""
    if cap_status == "missing" and entry.authority_source != "skill_payload":  # a payload keeps its own reject
        lost, why = result_loss(drive, entry)
        if lost and reject:
            return ""
        note, step = loss_refusal(drive, entry, lost, why)
    return _capture_failed_refusal(rid, cap_status, note, step)


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
            "The starter or its host-confirmed retry successor may dispose a completed result; "
            "retry authority does not include live control; once the owner is terminal, a live "
            "TOP-LEVEL task may dispose the orphan. Apply requires current write authority at the "
            "recorded target; targets inside configured projects/worktrees roots need not be the active "
            "folder, regardless of creator. Payloads keep their fresh exact binding."
        )
    from ouroboros.delegate_directory import is_directory_run, is_in_place_run
    if is_in_place_run(entry):
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_ISOLATED: run {rid} worked directly in {entry.target_root} by its "
            "starter's choice (workspace='in_place'); its edits are already there. There is nothing to apply "
            "or reject: inspect them with vcs_read diff and undo through git."
        )
    if not entry.execution_root and not is_directory_run(entry):
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_ISOLATED: run {rid} recorded no execution "
            "snapshot (read-only, or a pre-isolation run). There is no captured patch "
            "to integrate."
        )
    from ouroboros.delegate_registration_policy import candidate_apply_possible
    if not candidate_apply_possible(entry):  # one predicate before every Git, payload and folder branch
        return (
            f"⚠️ INTEGRATE_DELEGATED_INVESTIGATION_EVIDENCE: run {rid} was an evidence-only investigation "
            "in a private copy. Its captured changes are retained as evidence, never integrated, and owe "
            "no disposition; nothing was changed. To act on a finding, change the candidate yourself or "
            "start a child that holds write authority."
        )
    if entry.patch_disposed == custody.NO_CHANGES:
        return (f"OK: delegated run {rid} changed nothing: its result holds exactly the files its copy started "
                "from, so it closed itself (no_changes). There is nothing to apply or reject.")
    if entry.patch_disposed:
        return (
            f"⚠️ INTEGRATE_DELEGATED_ALREADY_DISPOSED: run {rid}'s patch was already "
            f"{entry.patch_disposed}. A disposition is final; nothing was changed."
        )
    if not (entry.settled or entry.terminal_state in custody.TERMINAL_STATES):
        # Execution end, not money: a pending ledger never holds a finished run's result (#1969).
        return (
            f"⚠️ INTEGRATE_DELEGATED_NOT_TERMINAL: run {rid} has no observed terminal, so it may still be "
            "running. delegate_wait it to its terminal; this call then prepares any capture still owed."
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
            else "vcs_read diff, compare against the patch artifact")
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
    LIVE in the non-Git payload — nothing staged, no index for vcs_read diff."""
    if applied:
        landed, verify = (
            (f"applied LIVE into the non-Git payload {target_root}",
             "compare the live payload against the patch artifact") if payload
            else (f"applied and staged in {target_root}", "verify with vcs_read diff"))
        return (
            f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {rid}'s patch IS {landed}, "
            "but the durable disposition row could not be "
            "written, so nothing on disk records that this patch was handled. Do "
            "NOT call integrate_delegated_patch again for this run — a second "
            "apply would land the same changes twice. Fix the drive/event log, "
            f"then {verify} and record the outcome; the run's saved result and "
            "pins are deliberately preserved."
        )
    return (
        f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {rid}'s patch was NOT applied "
        f"(disposition {disposition!r}), and the durable disposition row could not "
        "be written. Nothing in your tree changed and any saved result of the run is "
        "preserved. Fix the drive/event log; this process already holds the "
        "disposition in memory (a repeat here answers ALREADY_DISPOSED), so the "
        "run reads as undisposed again only after a restart — repeat it then."
    )


def _dispose_delegated(drive: Any, entry: Any, snapshot_key: str, reason: str,
                       disposition: str, cleanup: bool,
                       disposed_by: str = "", **row: Any) -> tuple[bool, str]:
    """Record a delegated disposition durably; clean up ONLY if the row landed.
    Releasing snapshot/patch on an UNWRITTEN row loses the record that the patch
    was handled (a restart could apply it twice). Shared by the Git and payload
    branches (``delegate_integration``) and an investigation's evidence retention;
    ``row`` adds facts such as an engine receipt. Returns ``(recorded, note)``."""
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_worktrees import remove_execution_snapshot

    recorded = custody.record_patch_disposed(
        drive, entry, disposition=disposition, reason=str(reason or ""),
        disposed_by_task_id=str(disposed_by or ""), **row)
    if not recorded:
        return False, ""
    note = ""
    if cleanup:
        try:
            removed = remove_execution_snapshot(snapshot_key)
        except Exception:
            removed = False
        note = "" if removed else (
            " (snapshot cleanup deferred to custody GC at boot or a later custody pass, "
            "once the owning task is proven over and the owner no longer keeps it.)")
    return True, note


def retain_run_evidence(drive: Any, entry: Any, block: Dict[str, Any], gateway: Any = None, *, recapture=False, named=None) -> Dict[str, Any]:
    """Retention disposes investigations; a candidate's saved result releases its checkout and
    awaits the owner's apply/reject, and an empty one closes itself."""
    from ouroboros.delegate_registration_policy import candidate_apply_possible

    if not candidate_apply_possible(entry):
        return retain_investigation_evidence(drive, entry, block, gateway, recapture=recapture, named=named)
    if named is None:
        named = run_named_evidence(drive, entry, recapture=recapture) if block.get("status") in _READY_CAPTURE_STATUSES else {}
    return _settle_writer_capture(drive, entry, {**block, "named_evidence": named} if named else block, named)


def _settle_writer_capture(drive: Any, entry: Any, block: Dict[str, Any], named: Dict[str, Any]) -> Dict[str, Any]:
    """Once a writer's capture row and its named evidence are durably saved, the result
    lives as the patch plus pinned commits B and C, so the checkout goes; an empty result
    (C equals B, HEAD unmoved, nothing retained beside it) closes as no_changes. A pending
    continuation, a copy made before this update, a path the result could not hold (a Git
    repository or filled submodule inside the copy, a folder Git could not open) or anything still owed keeps it."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_directory import is_directory_run
    from ouroboros.delegate_registration_policy import NO_CHANGES_NOTE
    from ouroboros.subagent_worktrees import find_execution_snapshot, release_snapshot_checkout

    if (not entry.snapshot_id or entry.authority_source == "skill_payload" or is_directory_run(entry)
            or block.get("status") not in _READY_CAPTURE_STATUSES or not entry.patch_captured
            or block.get("unsaved_paths")  # what the result could not hold stays until disposition
            or entry.patch_disposed or entry.superseded_by or entry.mode == "plan"
            or not (entry.settled or entry.terminal_state in custody.TERMINAL_STATES)
            or (named and named.get("status") != "retained")
            or any(str(item.get("snapshot_id") or "") == entry.snapshot_id
                   for item in custody.pending_invocations(drive))):
        return block
    record = find_execution_snapshot(entry.snapshot_id) or {}
    if not record or "file_baseline" in record or "untracked_baseline" in record:
        return block  # a copy made before this update keeps its checkout until disposition, as before
    if (block.get("status") == ARTIFACT_STATUS_READY_NO_CHANGES and not block.get("head_moved")
            and not block.get("file_outputs") and not (named or {}).get("files")):
        recorded, note = _dispose_delegated(drive, entry, entry.snapshot_id, "the run changed nothing",
                                            custody.NO_CHANGES, True, disposed_by=entry.task_id)
        if recorded:
            return {**block, "disposition": custody.NO_CHANGES, "note": NO_CHANGES_NOTE + note}
        return block
    try:
        released = release_snapshot_checkout(entry.snapshot_id)
    except Exception:
        log.warning("Releasing the checkout of captured run %s failed", entry.run_id, exc_info=True)
        released = False
    return {**block, "checkout_released": True} if released else block


def retain_investigation_evidence(drive: Any, entry: Any, block: Dict[str, Any], gateway: Any = None, *, recapture=False, named=None) -> Dict[str, Any]:
    """A ready investigation's capture IS its disposition (#1609), never applied.
    Patch/file outputs stay in the capture directory; Git named evidence is retained first.
    Gaps keep the copy undisposed. The one dispose writer records evidence_retained before
    releasing the snapshot; an engine directory copy is discarded first, its receipt on the row.
    Non-ready or apply-capable runs stay unchanged for inspection, never invented rejections.
    Rereads keep the retained investigation's evidence-only presentation."""
    import hashlib

    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_registration_policy import INVESTIGATION_NOTE, candidate_apply_possible

    ready = {"ready", ARTIFACT_STATUS_READY_WITH_CHANGES, ARTIFACT_STATUS_READY_NO_CHANGES}
    if (candidate_apply_possible(entry) or block.get("status") not in ready
            or entry.patch_disposed not in ("", custody.EVIDENCE_RETAINED)):
        return block
    retained_note = INVESTIGATION_NOTE
    directory = str(block.get("capture_kind") or "") == "engine_directory"
    if named is None:
        named = {} if directory else _retain_named_evidence(drive, entry, reread=bool(entry.patch_disposed), recapture=recapture)
    shown = {**block, **({"named_evidence": named} if named else {})}
    if entry.patch_disposed:
        return {**shown, "disposition": custody.EVIDENCE_RETAINED, "note": retained_note}
    if named.get("status") in {"partial", "capture_failed"}:
        return {**shown, "note": "Investigation evidence: files under .review-evidence/ could not all be retained "
                                 f"({named['status']}), so the private copy {entry.execution_root} is kept and nothing "
                                 "named is lost. Nothing was applied."}
    receipt: Dict[str, Any] = {}
    if gateway is not None and directory:
        key = hashlib.sha256(f"discard:{entry.run_id}:investigation".encode()).hexdigest()
        receipt = gateway.decide_run(entry.run_id, {"action": "discard"}, idempotency_key=key) or {}
        if receipt.get("status") != "discarded":
            return {**block, "note": "Investigation evidence retained; the engine copy's discard is unconfirmed."}
    recorded, note = _dispose_delegated(
        drive, entry, entry.snapshot_id, "evidence-only investigation: retained, never applied",
        custody.EVIDENCE_RETAINED, bool(entry.snapshot_id), disposed_by=entry.task_id,
        **({"engine_receipt": receipt} if receipt else {}))
    if not recorded:
        return {**shown, "note": "Investigation evidence retained; its disposition row could not be written, "
                                 "so the private copy is kept."}
    return {**shown, "disposition": custody.EVIDENCE_RETAINED, "note": retained_note + note}


def _resolve_acknowledged_intent(drive: Any, entry: Any) -> None:
    """CR2-1: the owner explicitly took over the AMBIGUOUS crash state.
    Resolves the stale apply intent durably as owner-acknowledged; the caller
    re-runs the NORMAL disposition from scratch (apply re-proves drift, reject
    re-runs the ready-manifest guard). A failed row write only makes a
    post-restart replay ambiguous again — the fail-closed direction."""
    from ouroboros import delegate_custody as custody

    custody.record_patch_apply_resolved(drive, entry, reason="owner_acknowledged")


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
            f"baseline ({drift_error}). Nothing was changed; the saved patch is kept. "
            f"Verdict: {verdict_path or '(unwritten)'}."
        )
    return (
        f"⚠️ INTEGRATE_CONFLICT: {len(drifted)} path(s) in {target} CHANGED since run "
        f"{rid}'s snapshot was taken, so its patch was NOT applied (a plain apply "
        f"would relocate hunks and silently land them on moved content). Drifted: "
        f"{', '.join(drifted[:10])}{' …' if len(drifted) > 10 else ''}\n"
        "YOU own this conflict: the captured patch (and its pinned commits, unless the owner "
        "released them in Storage) is kept until you resolve it. Reconcile your tree with the "
        "run's changes (read the patch artifact, or vcs_read diff from the capture's baseline_id "
        "to its result_commit while they are pinned), then retry the apply, or "
        "integrate_delegated_patch(decision='reject') to discard. "
        f"Verdict: {verdict_path or '(unwritten)'}."
    )


def _locked_apply(
    ctx: ToolContext, target: pathlib.Path, patch_path: pathlib.Path,
    ordered_touched: List[str], baseline_sha: str, *, file_changes=None, file_baseline=None,
    admission_check=None, three_way: bool = False, baseline_entries=None,
) -> Dict[str, Any]:
    """Apply one captured patch under the repo git lock — mechanics only.

    Returns the FACTS the caller turns into a verdict: ``proc`` (None when
    nothing was attempted), ``drifted``/``drift_error`` from the pre-apply
    baseline comparison, and ``staging_failure``/``reverted`` for an apply whose
    staging failed. No verdicts, no dispositions, no messages — those belong to
    the one caller so every exit stays visible in one place.
    """
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock
    from ouroboros.workspace_file_outputs import prepare_file_outputs

    result: Dict[str, Any] = {
        "proc": None, "drifted": [], "drift_error": "",
        "staging_failure": "", "reverted": False, "lock_error": "",
    }
    prepared = None
    temporary = ExitStack()
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
        # first. Dirty-source copies refuse drift; proven clean task copies
        # retain Git three-way synthesis and its explicit conflicts.
        try:
            result["drifted"], result["drift_error"] = _si()._baseline_drifted_paths(
                target, baseline_sha, [path for path in ordered_touched if path not in (file_baseline or {})],
                baseline_entries)
        except Exception as exc:
            result["drifted"], result["drift_error"] = [], f"{type(exc).__name__}: {exc}"
        if result["drift_error"] or (result["drifted"] and not three_way):
            return result
        if three_way:
            result["drifted"] = []  # Git's 3-way merge reconciles clean-base contributions.
        if file_changes:
            try:
                prepared = temporary.enter_context(prepare_file_outputs(
                    file_changes, target, baseline_sha=baseline_sha, file_baseline=file_baseline))
            except Exception as exc:
                result["drift_error"] = f"file results could not be prepared: {type(exc).__name__}: {exc}"
                return result
        if admission_check and (refusal := admission_check()):
            result["admission_refusal"] = refusal
            return result
        # Dirty-source snapshots use working-tree apply: --3way implies index
        # binding and would reject the source's pre-existing unstaged changes.
        # A task copy recorded from clean HEAD/index instead keeps three-way
        # synthesis. Both paths stage touched results and never commit.
        has_patch = patch_path.is_file() and patch_path.stat().st_size > 0
        proc = subprocess.run(
            ["git", "apply", *(["--3way", "--index"] if three_way else []), str(patch_path)],
            cwd=str(target), capture_output=True, text=True,
        ) if has_patch else subprocess.CompletedProcess([], 0, "", "")
        result["proc"] = proc
        if proc.returncode != 0:
            return result
        if prepared:
            try:
                prepared.apply()
            except Exception as exc:
                result["staging_failure"] = f"file result apply failed: {type(exc).__name__}: {exc}"
        stageable = _si()._stageable_paths(target, ordered_touched) if proc.returncode == 0 else []
        if not stageable and not result["staging_failure"]:
            return result
        # NUL-delimited stdin pathspecs: byte-safe and immune to argv limits for a
        # patch touching thousands of files.
        add = subprocess.run(
            ["git", "add", "--pathspec-from-file=-", "--pathspec-file-nul"],
            cwd=str(target), capture_output=True,
            input=b"\0".join(
                p.encode("utf-8", errors="surrogateescape") for p in stageable) + b"\0",
        ) if not result["staging_failure"] else None
        if add is not None and add.returncode == 0:
            return result
        # The APPLY SUCCEEDED — the tree is already mutated. Reporting this as a
        # conflict ("the tree moved") invited a retry over changed content and let a
        # later reject record "not applied" over real changes. Try to put the tree
        # back cleanly; whatever the outcome, the caller says exactly what is true.
        if add is not None:
            result["staging_failure"] = (add.stderr or add.stdout or b"").decode(
                "utf-8", errors="replace").strip()
        files_reverted = True
        if prepared:
            try:
                prepared.rollback()
            except Exception as exc:
                files_reverted = False
                result["staging_failure"] += f"; file rollback: {type(exc).__name__}: {exc}"
        if not has_patch:
            result["reverted"] = files_reverted
            return result
        check = subprocess.run(
            ["git", "apply", "--check", "--reverse", str(patch_path)],
            cwd=str(target), capture_output=True, text=True,
        )
        if check.returncode == 0:
            result["reverted"] = subprocess.run(
                ["git", "apply", "--reverse", str(patch_path)],
                cwd=str(target), capture_output=True, text=True,
            ).returncode == 0 and files_reverted
        return result
    finally:
        temporary.close()
        _release_git_lock(_git_lock)


def _integrate_delegated_patch(
    ctx: ToolContext,
    run_id: str = "",
    decision: str = "apply",
    reason: str = "",
    acknowledge_ambiguous: bool = False,
    paths: List[str] | None = None,
) -> str:
    """Serialize a captured run's disposition across owners and retry successors.

    The canonical capture address keeps every caller on the same lock even with
    separate task drives. Admission is repeated inside it; a stale custody object
    must never authorize a second materialization or an obsolete starter.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_shared import orphan_disposition_status
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    rid = str(run_id or "").strip()
    if not rid:
        return "⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): run_id is required."
    decision = str(decision or "apply").strip().lower()
    if decision not in {"apply", "reject"}:
        return "⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): decision must be 'apply' or 'reject'."
    drive = custody.custody_root(ctx)
    status, entry, _ = orphan_disposition_status(ctx, drive, rid)
    if status != custody.OWNED or entry is None:
        return _delegated_disposition_refusal(status, entry, rid, acknowledge_ambiguous)
    captured_before = bool(entry.patch_captured)  # before any wait: a capture made meanwhile was never shown
    lock_path = custody.disposition_lock_path(drive, entry)  # one per snapshot, shared by a continuation chain
    try:
        lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=20.0, owner_aware_stale=True)
    except Exception as exc:
        return f"⚠️ INTEGRATE_LOCK_TIMEOUT: could not lock the captured result: {type(exc).__name__}: {exc}."
    if lock_fd is None:
        return "⚠️ INTEGRATE_LOCK_TIMEOUT: another disposition holds this captured result; retry after it completes."
    try:
        from ouroboros.delegate_continuation import disposition_refusal
        from ouroboros.delegate_custody_current import run

        if handover := disposition_refusal(drive, run(drive, rid)):
            return handover
        result = _integrate_delegated_patch_locked(ctx, rid, decision, reason, acknowledge_ambiguous, paths,
                                                   captured_before=captured_before)
        from ouroboros.delegate_terminal import refresh_disposed_reconciliation

        try:
            refresh_disposed_reconciliation(drive, rid, reader_task_id=str(getattr(ctx, "task_id", "") or ""))
        except Exception:
            log.warning("Disposed custody disclosure refresh failed for run %s", rid, exc_info=True)
        return result
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def _current_disposition_refusal(ctx: ToolContext, rid: str) -> str:
    """Revalidate authority after capture/preparation, allowing this call's intent."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_shared import orphan_disposition_status

    status, entry, _ = orphan_disposition_status(ctx, custody.custody_root(ctx), rid)
    return _delegated_disposition_refusal(status, entry, rid, acknowledge_ambiguous=True)


def _integrate_delegated_patch_locked(
    ctx: ToolContext,
    run_id: str = "",
    decision: str = "apply",
    reason: str = "",
    acknowledge_ambiguous: bool = False,
    paths: List[str] | None = None,
    captured_before: bool | None = None,
) -> str:
    """The C1 explicit acceptance seam: apply or reject ONE delegated run's captured patch.

    A mutating delegated run executed in a PRIVATE execution snapshot; its diff was
    captured at terminal. NOTHING reaches the shared tree automatically — this tool is
    the only path in, it targets the run's recorded authority target (which must be
    this task's active root, or an admitted orphan target), and under the repo git lock proves no touched path
    drifted from the run's baseline, applies the patch to the WORKING TREE (a plain
    `git apply`, deliberately NOT --3way — see `_locked_apply`: --3way implies index
    binding and refuses the normal dirty shared-tree state the baseline deliberately
    snapshots), stages the touched paths explicitly, and records the disposition
    durably. Only a recorded disposition releases the snapshot for cleanup; a
    conflict keeps the snapshot and the patch as resolution material.
    ``acknowledge_ambiguous`` (CR2-1) is the owner exit from the AMBIGUOUS
    crash state: it resolves the stale intent durably and re-runs this normal
    flow, whose own guards re-verify the tree. A no-op when nothing is pending.
    A capture still owed by a finished run is prepared here, under the shared lock, and
    returned for inspection instead of being applied in the same call (#1969).
    """
    from ouroboros import delegate_custody as custody, delegate_source_coverage
    from ouroboros.delegate_shared import orphan_disposition_status

    rid = run_id
    drive = custody.custody_root(ctx)
    status, entry, orphan_of = orphan_disposition_status(ctx, drive, rid)
    if captured_before is None and entry is not None:
        captured_before = bool(entry.patch_captured)
    orphan_note = f"(orphan of terminal task {orphan_of}) " if orphan_of else ""
    refusal = _delegated_disposition_refusal(status, entry, rid, acknowledge_ambiguous)
    if refusal:
        return refusal
    if decision == "apply" and entry is not None:
        if refusal := delegate_source_coverage.source_apply_refusal(entry, rid):
            return refusal
    from ouroboros.delegate_directory import is_directory_run, integrate_directory_result
    if is_directory_run(entry):
        from ouroboros.claudexor_daemon import read_owned_gateway
        try:
            with read_owned_gateway() as gateway:
                return integrate_directory_result(ctx, entry, decision, reason, gateway,
                                                  acknowledge_ambiguous=acknowledge_ambiguous,
                                                  paths=paths, orphan=bool(orphan_of),
                                                  admission_check=lambda: _current_disposition_refusal(ctx, rid),
                                                  prepared=not captured_before)
        except Exception as exc:
            return f"⚠️ INTEGRATE_DELEGATED_APPLY_UNCONFIRMED: {type(exc).__name__}: {exc}. Retain this run; no replacement was started."
    # Directory selections have their own semantics. On whole-capture paths,
    # an explicitly empty list asks for no filter, exactly as omission does.
    empty_paths = type(paths) is list and not paths
    if paths is not None and not empty_paths:
        return ("⚠️ TOOL_ARG_ERROR (integrate_delegated_patch): nonempty paths selects engine "
                "directory results only. Omit paths or use [] for the complete Git/skill capture.")
    selection_note = "\npaths=[] selects the complete captured result, as when paths is omitted." if empty_paths else ""
    if entry.patch_apply_pending and acknowledge_ambiguous:
        _resolve_acknowledged_intent(drive, entry)
    cap_dir = custody.delegated_capture_dir(drive, entry.task_id, custody.capture_key(entry))
    manifest_path = cap_dir / "workspace_patch.json"
    patch_path = cap_dir / "workspace.patch"
    prepared = not (captured_before and _manifest_capture_status(manifest_path) in _READY_CAPTURE_STATUSES)
    capture_refusal = _si()._capture_at_disposition(drive, entry, rid, manifest_path, decision)
    if capture_refusal:
        return capture_refusal
    if refusal := _current_disposition_refusal(ctx, rid):
        return refusal
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"⚠️ INTEGRATE_MANIFEST_UNREADABLE: {manifest_path}: {type(exc).__name__}: {exc}."
    from ouroboros.delegate_collection import capture_projection

    if inspection := prepared_capture_inspection(
            rid, decision, manifest.get("status") == ARTIFACT_STATUS_READY_WITH_CHANGES,
            lambda: capture_projection(drive, entry, reader_task_id=str(getattr(ctx, "task_id", "") or "")) or {},
            prepared=prepared):
        return inspection
    if entry.authority_source == "skill_payload" or str(manifest.get("capture_kind") or "") == "skill_payload":
        # The exact-payload branch (R1 item 3) lives in delegate_integration:
        # fresh semantic rebinding, whole-payload CAS, reserved-path whole-apply
        # refusal, index-free apply into the live NON-Git payload — no staging.
        from ouroboros.tools.delegate_integration import integrate_payload_patch

        return integrate_payload_patch(
            ctx, drive=drive, entry=entry, rid=rid, decision=decision,
            reason=reason, cap_dir=cap_dir, manifest=manifest, patch_path=patch_path) + (f"\n{orphan_note.rstrip()}" if orphan_note else "") + selection_note
    return _integrate_git_capture(ctx, entry, decision, reason, manifest, cap_dir, orphan_of) + selection_note


def _integrate_git_capture(ctx, entry, decision, reason, manifest, cap_dir, orphan_of):
    """Apply/dispose a Git capture after common run ownership and source admission.

    File postimages and the diagnostic patch share one target, lock and disposition.
    Payload and engine-directory products have their own materialization owners.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_shared import orphan_apply_target_ok
    from ouroboros.workspace_copies import copy_apply_refusal, source_is_system_repo, target_write_refusal
    from ouroboros.tools.tool_resolution import system_repo_dir_for

    drive, rid = custody.custody_root(ctx), entry.run_id
    snapshot_key = entry.snapshot_id or rid
    patch_path = cap_dir / "workspace.patch"
    orphan_note = f"(orphan of terminal task {orphan_of}) " if orphan_of else ""
    touched = [str(p) for p in (manifest.get("tracked_changed") or [])]
    touched += [str(p) for p in (manifest.get("untracked_included") or [])]
    from ouroboros.workspace_file_outputs import file_output_changes
    try:
        file_changes = file_output_changes(manifest, cap_dir)
    except Exception as exc:
        return _capture_failed_refusal(rid, str(manifest.get("status") or ""), str(exc))
    touched = sorted(set(touched) | {row["path"] for row in file_changes})

    named = run_named_evidence(drive, entry)  # (#2070) evidence first: a gap keeps a surviving copy undisposed
    if named.get("status") in ("partial", "capture_failed") and pathlib.Path(str(entry.execution_root)).is_dir():
        gaps = [str(gap.get("path") or "") for gap in named.get("gaps") or [] if isinstance(gap, dict)]
        return (f"⚠️ INTEGRATE_DELEGATED_EVIDENCE_UNRETAINED: run {rid}'s named evidence could not all be retained "
                f"({named['status']}: {', '.join(gaps[:5]) or named.get('error') or 'unknown'}). No disposition was "
                f"recorded and nothing changed: apply or reject would release {entry.execution_root}, the last copy "
                "of those files. Keep what you need from it (or remove what is not evidence), then repeat this call.")

    def _dispose(disposition: str, cleanup: bool, **row: Any) -> tuple[bool, str]:
        return _dispose_delegated(
            drive, entry, snapshot_key, reason, disposition, cleanup,
            disposed_by=str(getattr(ctx, "task_id", "") or ""), **row)

    def _unwritten_disposition(disposition: str, applied: bool) -> str:
        return _unwritten_disposition_text(rid, str(entry.target_root), disposition, applied)

    capture_status = str(manifest.get("status") or "")
    if decision == "reject":
        # A reject releases a ready capture, a drift-only failed no-change capture or a result proven
        # lost (#1651, re-proven under this lock): none of them discards a child byte.
        from ouroboros.tools.delegate_integration import lost_words, result_loss

        keep = capture_status in _READY_CAPTURE_STATUSES or _drifted_no_change(manifest)
        lost = {} if keep else result_loss(drive, entry)[0]
        if not (keep or lost):
            return _capture_failed_refusal(
                rid, capture_status, "a reject would release the snapshot over it")
        verdict_path = _si()._write_verdict(
            ctx, f"run_{rid}", outcome="result_lost" if lost else "rejected", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=[], protected=[],
            target=str(entry.target_root),
        )
        recorded, note = _dispose("rejected", cleanup=lost.get("copy_record") != "absent",  # nothing left to clean
                                  **({"result_lost": lost} if lost else {}))
        if not recorded:
            return _unwritten_disposition("rejected", applied=False)
        if lost:
            return (f"{orphan_note}🚫 Rejected delegated run {rid} as LOST: {lost_words(drive, entry, lost)}. This decision "
                    "changed no files; what the run did outside its copy and what it cost stay as recorded, unknown where "
                    f"unknown. Verdict: {verdict_path or '(unwritten)'}. Reason: {reason or '(none)'}.{note}")
        return (
            f"{orphan_note}🚫 Rejected delegated run {rid}'s captured patch ({len(touched)} file(s) not "
            f"applied); its private copy and pins are released. Verdict: {verdict_path or '(unwritten)'}. "
            f"Reason: {reason or '(none)'}.{format_patch_exclusions(manifest)}{note}"
        )

    if capture_status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        if capture_status == ARTIFACT_STATUS_READY_NO_CHANGES:
            recorded, note = _dispose("applied", cleanup=True)
            if not recorded:
                return _unwritten_disposition("applied", applied=False)
            return (
                f"OK: delegated run {rid} has no captured file changes to apply; "
                f"its answer and external effects remain separate evidence; its private copy is released."
                f"{format_patch_exclusions(manifest)}{note}"
            )
        return (
            f"⚠️ INTEGRATE_DELEGATED_NO_CAPTURE: run {rid}'s capture status is "
            f"{capture_status or 'missing'!r} — no applicable patch. If the run just "
            "ended, delegate_wait it once more to capture; a failed capture keeps the "
            "snapshot for direct inspection."
        )
    if not patch_path.exists() and not file_changes:
        return f"⚠️ INTEGRATE_PATCH_MISSING: captured patch not found at {patch_path}."
    expected_digest = str(manifest.get("sha256") or "")
    if expected_digest and patch_path.exists():
        actual_digest = _si()._sha256_file(patch_path)
        if actual_digest != expected_digest:
            return (
                f"⚠️ INTEGRATE_PATCH_CORRUPT: sha256 mismatch for run {rid} "
                f"(manifest {expected_digest[:12]} != file {actual_digest[:12]}); refusing to apply."
            )

    # Only the recorded target is writable: orphan recovery never rebinds the
    # caller's workspace or selects a replacement destination.
    try:
        active_root = pathlib.Path(ctx.active_repo_dir()).resolve(strict=False)
    except Exception as exc:
        return f"⚠️ INTEGRATE_TARGET_ERROR: could not resolve active repo: {type(exc).__name__}: {exc}."
    target = pathlib.Path(str(entry.target_root or "")).resolve(strict=False)
    target_ok = orphan_apply_target_ok(target, active_root) if orphan_of else target == active_root
    if not str(entry.target_root or "").strip() or not target_ok:
        return (
            "⚠️ INTEGRATE_DELEGATED_TARGET_MISMATCH: the run's recorded authority target "
            f"({entry.target_root or '(none)'}) is not this task's active root ({active_root}). "
            "Only a terminal owner's orphan targeting a folder inside the configured projects/worktrees "
            "roots may apply outside the active folder, under current write rights, regardless of creator."
        )
    if not (target / ".git").exists():
        return f"⚠️ INTEGRATE_TARGET_NOT_GIT: target {target} is not a git working tree."

    patch_touched, parse_error = (_si()._patch_touched_paths(patch_path, target)
                                  if patch_path.exists() else (set(), ""))
    patch_touched.update(row["path"] for row in file_changes)
    if parse_error:
        return (
            f"⚠️ INTEGRATE_PATCH_UNREADABLE: cannot parse run {rid}'s captured patch "
            f"(git apply --numstat failed): {parse_error[:300]}"
        )
    ordered_touched = sorted(patch_touched)
    runtime_mode = _si()._integration_runtime_mode(ctx)
    target_is_body = (source_is_system_repo(target, system_repo_dir_for(ctx))
                      if orphan_of else _si()._target_is_system_repo(ctx))
    protected = (protected_paths_in(ordered_touched)
                 if target_is_body else [])
    if protected:
        constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
        is_acting = bool(constraint and getattr(constraint, "mode", "") == ACTING_SUBAGENT_MODE)
        grant_ok = not is_acting or bool(getattr(constraint, "protected_paths_grant", False))
        if not (mode_allows_protected_write(runtime_mode) and grant_ok):
            _si()._write_verdict(
                ctx, f"run_{rid}", outcome="blocked_protected", reason=reason, files=touched,
                manifest=manifest, applied=False, conflicts=[],
                protected=[p.path for p in protected], target=str(target))
            return protected_write_block_message(
                path=protected[0].path, runtime_mode=runtime_mode,
                action=f"integrate delegated run {rid} patch touching")

    def admission_refusal():
        if refusal := _current_disposition_refusal(ctx, rid):
            return refusal
        refusal = copy_apply_refusal(ctx, target, ordered_touched)
        return target_write_refusal(target, refusal) if refusal else ""

    if refusal := admission_refusal():
        return refusal
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
    file_options = {}
    if file_changes:
        from ouroboros.subagent_worktrees import find_execution_snapshot
        snapshot = find_execution_snapshot(entry.snapshot_id) or {}
        file_options = {"file_changes": file_changes, "file_baseline": snapshot.get("file_baseline") or {}}
    outcome = _si()._locked_apply(
        ctx, target, patch_path, ordered_touched, entry.baseline_sha, baseline_entries=manifest.get("baseline_entries"),
        admission_check=admission_refusal, **file_options)
    if outcome.get("admission_refusal"):
        custody.record_patch_apply_resolved(drive, entry, reason="authority_changed")
        return outcome["admission_refusal"]
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
                f"⚠️ INTEGRATE_APPLIED_UNSTAGED: applying or staging run {rid}'s results in "
                f"{target} failed ({staging_failure[:300]}), so the apply "
                "was reversed — your tree is back to its pre-apply state and NOTHING is "
                "left half-applied. The snapshot and the patch are preserved; fix the "
                f"index problem, then call this tool again. Verdict: {verdict_path or '(unwritten)'}."
                f"{format_patch_exclusions(manifest)}"
            )
        recorded, _ = _dispose("applied", cleanup=False)
        tail = "" if recorded else (
            " ⚠️ the durable disposition row could ALSO not be written — record this "
            "outcome yourself before any further integration attempt."
        )
        return (
            f"⚠️ INTEGRATE_APPLIED_UNSTAGED: run {rid}'s result transfer in {target} "
            f"({len(touched)} file(s)) is incomplete ({staging_failure[:300]}), "
            "and the apply could not be cleanly reversed. Do NOT retry this call — the "
            "changes are already in your working tree and a second apply would double "
            "them. Inspect with vcs_read diff, stage what you accept yourself, and note that "
            "the run is recorded as applied. Its captured patch is kept for "
            f"comparison. Verdict: {verdict_path or '(unwritten)'}.{format_patch_exclusions(manifest)}{tail}"
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
            "YOU own this conflict: the captured patch (and its pinned commits, unless the owner "
            "released them in Storage) is kept until you resolve it. Reconcile your tree with the "
            "run's changes (read the patch artifact, or vcs_read diff from the capture's baseline_id "
            "to its result_commit while they are pinned), retry the apply, or "
            "integrate_delegated_patch(decision='reject') to discard. "
            f"Verdict: {verdict_path or '(unwritten)'}."
        )

    try:
        invalidate_advisory_after_mutation(
            pathlib.Path(getattr(ctx, "drive_root", ".")),
            mutation_root=target,
            changed_paths=touched,
            source_tool="integrate_delegated_patch",
            mutating_task_id=str(getattr(ctx, "task_id", "") or ""),
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
        f"{orphan_note}✅ Integrated delegated run {rid}'s patch into {target} ({len(touched)} file(s), staged).{prot_note}\n"
        f"{diffstat}{format_patch_exclusions(manifest)}\n"
        f"Verdict: {verdict_path or '(unwritten)'}. Its private copy and pins are released.\n"
        "Changes are staged but NOT committed — review them yourself; you are the sole committer."
        f"{note}"
    )
