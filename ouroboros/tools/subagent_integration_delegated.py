"""``integrate_delegated_patch`` owns typed refusals, capture-at-disposition, disposition records and locked apply.
Extracted at the v7 DEL1 size gate; ``subagent_integration`` re-exports the same objects,
retaining sibling imports and monkeypatch targets on that surface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import subprocess
from contextlib import ExitStack
from typing import Any, Callable, Dict, List

from ouroboros.artifacts import task_id_for_artifacts
from ouroboros.contracts.task_constraint import normalize_task_constraint
from ouroboros.headless import (
    ARTIFACT_STATUS_READY_NO_CHANGES,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
)
from ouroboros.review_state import invalidate_advisory_after_mutation
from ouroboros.delegate_named_evidence import (
    retain_named as _retain_named_evidence, run_named_evidence,
)
from ouroboros.runtime_mode_policy import protected_paths_in
from ouroboros.workspace_copies import protected_apply_refusal, protected_write_admitted
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE
from ouroboros.workspace_patch_rules import declared_capture_paths, format_patch_exclusions
from ouroboros.tools.patch_staging import (
    index_entries, sparse_absent, stage_applied, staged_postimage, staging_receipt, undo_applied,
)
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, append_published_text

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


def _saved_manifest(manifest_path: pathlib.Path, *, entry=None) -> Dict[str, Any]:
    """One capture manifest as saved; {} when absent or unreadable."""
    try:
        raw = manifest_path.read_bytes()
        loaded = json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    if entry is not None and loaded.get("status") in _READY_CAPTURE_STATUSES:
        from ouroboros.delegate_capture_job import manifest_recorded, ordinary_git

        if ordinary_git(entry) and not manifest_recorded(entry, raw):
            return {}  # a Git manifest its receipt does not name is not yet this run's capture
    return loaded


def _manifest_capture_status(manifest_path: pathlib.Path, *, entry=None) -> str:
    """The status one capture manifest reports about ITSELF; "" when unreadable."""
    return str(_saved_manifest(manifest_path, entry=entry).get("status") or "")


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
        f"⚠️ INTEGRATE_DELEGATED_CAPTURE_FAILED: run {rid} has no confirmed usable capture "
        f"(status {cap_status or 'missing'!r}: {note[:600]}). "
        "No disposition was recorded — the obligation stays open and the private copy (if present) is "
        "kept. " + (step or "Repeating this call fails the same way until that cause changes; once it has, call again.")
    )


def _apply_acts_on(manifest: Dict[str, Any], baseline_sha: str) -> bool:
    """A capture an apply would act on: changed files, or an empty result over a moved HEAD whose
    commits the apply's release of the copy and its pins would discard (the projection's head_moved)."""
    status, head = manifest.get("status"), str(manifest.get("current_head") or "")
    return status == ARTIFACT_STATUS_READY_WITH_CHANGES or (
        status == ARTIFACT_STATUS_READY_NO_CHANGES and bool(head and baseline_sha) and head != baseline_sha)


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
    drive: Any, entry: Any, rid: str, manifest_path: pathlib.Path, decision: str = "apply", *, custody_held=None,
) -> Dict[str, Any] | str:
    """Return this call's capture/retention facts, or a refusal. A ready (not merely
    flagged) changed manifest needs no recapture. An interrupted empty capture
    still needs settlement and uses the same projection as a fresh capture.
    Join the maintenance child's prepared result before trying to capture. Legacy
    drift-only rejects and proven loss (#1651) pass to the reject guard.
    """
    reject = decision == "reject"
    manifest = _saved_manifest(manifest_path, entry=entry)
    status = manifest.get("status")
    if (reject and _drifted_no_change(manifest) or entry.patch_captured and (
            status == ARTIFACT_STATUS_READY_WITH_CHANGES
            or status == ARTIFACT_STATUS_READY_NO_CHANGES and entry.superseded_by)):
        return {}
    from ouroboros.delegate_capture_job import accept_ready_patch
    from ouroboros.delegate_custody_current import run
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive, loss_refusal, result_loss
    from ouroboros.store import StoreError

    try:
        block = (accept_ready_patch(drive, entry, custody_held=custody_held)
                 if entry.authority_source != "skill_payload" else None)
        if block is None:
            block = capture_terminal_patch_for_drive(drive, entry, retain_investigation=False) or {}
        elif (entry := run(drive, rid)) is None:
            raise OSError("the published capture's custody is unreadable")
    except Exception as exc:
        log.warning("capture-at-disposition raised for run %s", rid, exc_info=True)
        block = {"status": "", "note": f"capture raised {type(exc).__name__}: {exc}"}
    cap_status, note = str(block.get("status") or ""), str(block.get("note") or "")
    if cap_status in _READY_CAPTURE_STATUSES:
        named = _disposition_evidence(drive, entry, custody_held)
        if isinstance(named, str):
            return named
        try:
            return retain_run_evidence(drive, entry, block, named=named, custody_held=custody_held)
        except (OSError, StoreError) as exc:
            return (f"⚠️ INTEGRATE_DELEGATED_CAPTURE_FAILED: run {rid}'s patch was captured, but checking its "
                    f"custody for settlement failed ({type(exc).__name__}: {exc}). No target files changed. "
                    "Fix the custody store and read back the disposition before retrying.")
    step = "After fixing that cause, call again." if block.get("retry") == "transient" else ""
    if cap_status == "missing" and entry.authority_source != "skill_payload":  # a payload keeps its own reject
        lost, why = result_loss(drive, entry)
        if lost and reject:
            return {}
        note, step = loss_refusal(drive, entry, lost, why)
    return _capture_failed_refusal(rid, cap_status, note, step)


_EVIDENCE_UNRETAINED = "⚠️ INTEGRATE_DELEGATED_EVIDENCE_UNRETAINED"  # reached only over a ready capture


def _disposition_evidence(drive, entry, custody_held) -> Dict[str, Any] | str:
    """Keep evidence failure distinct from a failed patch capture."""
    try:
        named = run_named_evidence(drive, entry, custody_held=custody_held)
    except OSError as exc:
        cause = str(exc)
    else:
        if named.get("status") not in ("partial", "capture_failed") or not pathlib.Path(entry.execution_root).is_dir():
            return named
        gaps = [str(gap.get("path") or "") for gap in named.get("gaps") or [] if isinstance(gap, dict)]
        cause = ", ".join(gaps[:5]) or named.get("error") or named.get("reason") or named["status"]
    return (f"{_EVIDENCE_UNRETAINED}: run {entry.run_id}'s named evidence could not all be "
            f"retained ({cause}). No disposition was recorded; no target files changed. Any private copy is kept. "
            "Fix the evidence retention problem, then call again.")


def _git_disposition_result(ctx: ToolContext, text: str, *, changed: bool = False, facts=None) -> str:
    """FACTS add this call's own typed transfer and record outcomes; without them only the change flag."""
    return _publish_tool_result(ctx, ToolResult("ok", "OK", text, {"target_files_changed": changed, **(facts or {})}))


def _delegated_disposition_refusal(status: str, entry: Any, rid: str,
                                   acknowledge_ambiguous: bool = False, *, ctx: ToolContext | None = None) -> str:
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
        return _git_disposition_result(ctx, f"OK: delegated run {rid} changed nothing: its result holds exactly the files "
                                       "its copy started from, so it closed itself (no_changes). There is nothing to apply or reject.")
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
            "but the durable disposition row could not be written, so nothing on disk records that this patch was handled. "
            "Do NOT call integrate_delegated_patch again for this run — a second apply would land the same changes twice. "
            f"Fix the drive/event log, then {verify} and record the outcome; the run's saved result and pins are deliberately preserved."
        )
    return (
        f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {rid}'s patch was NOT applied "
        f"(disposition {disposition!r}), and the durable disposition row could not "
        "be written. Nothing in your tree changed and any saved result of the run is "
        "preserved. Fix the drive/event log and read back the durable disposition before retrying."
    )


# ToolResult meta holds 8192 bytes of ASCII-escaped JSON; a transfer's typed view stays near 6 KiB of it. A root
# over its share is marked instead (the text names it whole); the unindexed names are the receipt's first 20
# within theirs, their count stays whole and the verdict keeps every name.
_ROOT_META_BYTES, _UNINDEXED_SHOWN, _UNINDEXED_META_BYTES = 2048, 20, 4096


def _transfer_facts(target: Any, touched: List[str], unindexed: List[str]) -> Dict[str, Any]:
    """The whole file effect of a Git transfer that reached TARGET, whatever was or was not recorded after it.
    TOUCHED is the operation's own path set. Text reads these whole; typed meta takes ``_typed_transfer``."""
    return {"target_files_changed": True, "target_root": str(target), "files_changed": len(touched),
            "unindexed": list(unindexed)}


def _typed_transfer(facts: Dict[str, Any], *, settled: bool = True) -> Dict[str, Any]:
    """FACTS as ToolResult meta, bounded for any supported root and output. An unsettled transfer (its undo did not
    complete) claims only its attempt: ``files_attempted``, with ``files_changed`` null as what remains is unknown."""
    root = str(facts["target_root"])
    typed: Dict[str, Any] = {"target_files_changed": True, **(
        {"target_root": root} if len(json.dumps(root, ensure_ascii=True)) <= _ROOT_META_BYTES
        else {"target_root_omitted": True})}
    if not settled:
        return {**typed, "files_attempted": facts["files_changed"], "files_changed": None}
    unindexed, shown, room = list(facts["unindexed"]), [], _UNINDEXED_META_BYTES
    for path in unindexed[:_UNINDEXED_SHOWN]:
        room -= len(json.dumps(str(path), ensure_ascii=True)) + 1
        if room < 0:
            break
        shown.append(str(path))
    return {**typed, "files_changed": facts["files_changed"], "unindexed": shown, "unindexed_count": len(unindexed)}


def _native_gap_note(native_gap: str) -> str:
    return (f" The native copy's capture could not record this apply's effect ({native_gap[:300]}): its capture "
            "stays pending with the effect unknown." if native_gap else "")


def _git_unwritten_disposition(ctx, entry, disposition: str, *, staging_proof: bool | None = None,
                               transfer: Dict[str, Any] | None = None, changed: bool = True,
                               native_gap: str = "", orphan: bool = False) -> str:
    """Report an effect, apart from the availability of proof for a receipt-only retry. ``staging_proof`` is
    None when no apply landed; ``changed`` says whether THIS call made the TRANSFER (a recovery made none).
    Only the starter or its host-confirmed retry successor may complete the record; an ORPHAN adopter's later
    call meets the ambiguity, whose provided exit (``acknowledge_ambiguous``) the report names."""
    if staging_proof is None:
        text = _unwritten_disposition_text(entry.run_id, str(entry.target_root), disposition, applied=False)
    else:
        facts = transfer or {}
        staged, delivered, _closing = staging_receipt(int(facts.get("files_changed") or 0), facts.get("unindexed") or [])
        next_step = (
            f"Once {'the custody store and the native copy accept' if native_gap else 'the custody store accepts'} "
            "writes, integrate_delegated_patch for this run from this task or its host-confirmed retry successor "
            "records this apply without changing a file, provided this attempt's verdict row, its manifest, the "
            "target folder and its .git, HEAD, the touched files and their index entries are unchanged; otherwise "
            "it refuses as ambiguous."
            if staging_proof and not orphan else
            "An orphan's adopter cannot complete this record by calling again: a later integrate_delegated_patch for "
            "this run refuses as ambiguous instead of applying again. Inspect the working tree and index against the "
            "saved patch, then call it with acknowledge_ambiguous=true to take that state over: the normal decision "
            "then runs, where an apply re-checks drift first and a reject releases the private copy and pins while "
            "the captured patch is kept."
            if orphan else
            "No staging proof was recorded, so a later integrate_delegated_patch for this run refuses as "
            "ambiguous instead of applying again. Inspect the working tree and index against the saved patch."
        )
        text = (f"⚠️ INTEGRATE_DISPOSITION_UNWRITTEN: run {entry.run_id}'s patch reached "
                f"{facts.get('target_root') or entry.target_root} ({staged}) and Git staging completed, but its "
                f"durable disposition was not recorded.{delivered} {next_step} The saved result and pins are kept."
                f"{_native_gap_note(native_gap)}")
    return _publish_tool_result(ctx, ToolResult("blocked", "INTEGRATION_BLOCKED", text,
        {"target_files_changed": False, **(_typed_transfer(transfer) if transfer and changed else {}),
         "disposition_recorded": False,
         **({"native_capture_effect": "unknown"} if native_gap else {})}))


def _native_effect(drive: Any, entry: Any, outcome: Dict[str, Any]) -> str:
    """Record an apply's effect for a native copy's capture; a failed write keeps that capture's debt open
    (its effect stays ``unknown``) and is reported, never raised over an effect already made."""
    from ouroboros.headless import record_native_capture_effect

    try:
        record_native_capture_effect(drive, entry, outcome)
    except Exception as exc:  # noqa: BLE001 - the tree may already carry the apply
        log.warning("Recording run %s's native capture effect failed", entry.run_id, exc_info=True)
        return f"{type(exc).__name__}: {exc}"
    return ""


def _dispose_delegated(drive: Any, entry: Any, snapshot_key: str, reason: str,
                       disposition: str, cleanup: bool,
                       disposed_by: str = "", defer_cleanup: bool = False, **row: Any) -> tuple[bool, str]:
    """Record a delegated disposition durably; clean up ONLY if the row landed.
    Releasing snapshot/patch on an UNWRITTEN row loses the record that the patch
    was handled (a restart could apply it twice). Shared by the Git and payload
    branches (``delegate_integration``) and an investigation's evidence retention;
    ``defer_cleanup`` leaves retirement to custody GC when native custody is held: the
    remover takes snapshot-record then task custody, never the reverse. ``row`` adds receipt facts."""
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
            removed = False if defer_cleanup else remove_execution_snapshot(snapshot_key)
        except Exception:
            removed = False
        note = "" if removed else (
            " (snapshot cleanup deferred to custody GC at boot or a later custody pass, "
            "once the owning task is proven over and the owner no longer keeps it.)")
    return True, note


def retain_run_evidence(drive: Any, entry: Any, block: Dict[str, Any], gateway: Any = None, *, recapture=False, named=None,
                        custody_held=None) -> Dict[str, Any]:
    """Retention disposes investigations; a saved candidate awaits apply/reject, an empty one
    closes itself. Native custody defers physical retirement to the ordinary GC."""
    from ouroboros.delegate_registration_policy import candidate_apply_possible

    from ouroboros.delegate_capture_job import READY, ordinary_git
    if ordinary_git(entry) and block.get("status") in READY and not entry.patch_captured:
        from ouroboros.tools.delegate_integration import capture_receipt_unwritten

        return capture_receipt_unwritten(block)
    if not candidate_apply_possible(entry):
        return retain_investigation_evidence(drive, entry, block, gateway, recapture=recapture, named=named)
    if named is None:
        named = (run_named_evidence(drive, entry, recapture=recapture, custody_held=custody_held)
                 if block.get("status") in _READY_CAPTURE_STATUSES else {})
    return _settle_writer_capture(drive, entry, {**block, "named_evidence": named} if named else block, named,
                                  defer_cleanup=custody_held is not None)


def _settle_writer_capture(drive: Any, entry: Any, block: Dict[str, Any], named: Dict[str, Any], *, defer_cleanup=False) -> Dict[str, Any]:
    """Once a writer's capture row and its named evidence are durably saved, the result
    lives as the patch plus pinned commits B and C; an eligible checkout goes unless deferred. An empty result
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
                                            custody.NO_CHANGES, True, disposed_by=entry.task_id, defer_cleanup=defer_cleanup)
        if recorded:
            return {**block, "disposition": custody.NO_CHANGES, "note": NO_CHANGES_NOTE + note}
        return {**block, "disposition_unwritten": custody.NO_CHANGES,
                "note": _unwritten_disposition_text(entry.run_id, entry.target_root, custody.NO_CHANGES, applied=False)}
    if defer_cleanup:
        return block  # preserve empty-result classification above, but never invert record/custody lock order
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
    from ouroboros.delegate_directory import is_directory_run
    from ouroboros.delegate_registration_policy import INVESTIGATION_NOTE, candidate_apply_possible

    ready = {"ready", ARTIFACT_STATUS_READY_WITH_CHANGES, ARTIFACT_STATUS_READY_NO_CHANGES}
    if (candidate_apply_possible(entry) or block.get("status") not in ready
            or entry.patch_disposed not in ("", custody.EVIDENCE_RETAINED)):
        return block
    retained_note = INVESTIGATION_NOTE
    directory = str(block.get("capture_kind") or "") == "engine_directory" or is_directory_run(entry)
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
    if directory:
        unconfirmed = "Investigation evidence retained; the engine copy's discard is unconfirmed."
        if gateway is None:
            return {**shown, "retry": "transient", "note": unconfirmed}
        key = hashlib.sha256(f"discard:{entry.run_id}:investigation".encode()).hexdigest()
        receipt = gateway.decide_run(entry.run_id, {"action": "discard"}, idempotency_key=key) or {}
        if not isinstance(receipt, dict) or receipt.get("accepted") is not True or receipt.get("status") != "discarded":
            return {**shown, "retry": "transient", "note": unconfirmed}
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


# Both conflicts (proven drift, a refused git apply) leave the same material with the same owner.
_CONFLICT_OWNED = (
    "YOU own this conflict: the captured patch (and its pinned commits, unless the owner "
    "released them in Storage) is kept until you resolve it. Reconcile your tree with the "
    "run's changes (read the patch artifact, or vcs_read diff from the capture's baseline_id "
    "to its result_commit while they are pinned), then retry the apply, or "
    "integrate_delegated_patch(decision='reject') to discard. ")


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
        f"{', '.join(drifted[:10])}{' …' if len(drifted) > 10 else ''}\n{_CONFLICT_OWNED}"
        f"Verdict: {verdict_path or '(unwritten)'}."
    )


def _locked_apply(
    ctx: ToolContext, target: pathlib.Path, patch_path: pathlib.Path,
    ordered_touched: List[str], baseline_sha: str, *, file_changes=None, file_baseline=None,
    admission_check=None, three_way: bool = False, baseline_entries=None, observe_completion=False,
) -> Dict[str, Any]:
    """Apply one captured patch under the repo git lock — mechanics only.

    Returns the FACTS the caller turns into a verdict: ``proc`` (None when
    nothing was attempted), ``drifted``/``drift_error`` from the pre-apply
    baseline comparison, ``unindexed`` (delivered new paths the target's ignore
    rules keep out of the index, `patch_staging`), and ``staging_failure``/``reverted``
    for an apply whose staging failed: ``reverted`` holds only when the tree, the
    file results and the index entries the attempt changed are all back. No
    verdicts, no dispositions, no messages — those belong to the one caller so
    every exit stays visible in one place.
    """
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock
    from ouroboros.workspace_file_outputs import prepare_file_outputs

    result: Dict[str, Any] = {
        "proc": None, "drifted": [], "drift_error": "", "unindexed": [],
        "staging_failure": "", "reverted": False, "lock_error": "",
    }
    prepared = None
    temporary = ExitStack()
    try:
        _git_lock = _acquire_git_lock(ctx, repo_dir=target)
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
        try:  # membership before the apply decides staging and what a rollback restores
            indexed = index_entries(target, ordered_touched)
        except (OSError, subprocess.SubprocessError) as exc:
            result["drift_error"] = f"the target's index could not be read: {exc}"
            return result
        sparse = sparse_absent(target, ordered_touched, indexed)
        # Dirty-source snapshots use working-tree apply: --3way implies index
        # binding and would reject the source's pre-existing unstaged changes.
        # A task copy recorded from clean HEAD/index instead keeps three-way
        # synthesis. Both paths stage touched results and never commit.
        has_patch = patch_path.is_file() and patch_path.stat().st_size > 0
        from ouroboros.tools.patch_staging import git_apply  # a work target applies it as the work domain
        proc = git_apply(target, patch_path, ["--3way", "--index"] if three_way else [],
                         text=True) if has_patch else subprocess.CompletedProcess([], 0, "", "")
        result["proc"] = proc
        if proc.returncode != 0:
            return result
        if prepared:
            try:
                prepared.apply()
            except Exception as exc:
                result["staging_failure"] = f"file result apply failed: {type(exc).__name__}: {exc}"
        if not result["staging_failure"]:
            result["staging_failure"], result["unindexed"] = stage_applied(target, ordered_touched, indexed)
            if not result["staging_failure"]:
                if observe_completion:
                    result["postimage_sha256"] = staged_postimage(target, ordered_touched)
                return result
        # The APPLY SUCCEEDED — the tree is already mutated. Reporting this as a
        # conflict ("the tree moved") invited a retry over changed content and let a
        # later reject record "not applied" over real changes. Undo the whole attempt;
        # the caller says exactly what is true.
        result["reverted"], problems = undo_applied(
            target, patch_path if has_patch else None, prepared, ordered_touched, indexed, sparse)
        result["staging_failure"] += f"; {problems}" if problems else ""
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
    """Serialize disposition at the canonical capture address across task drives and retry successors.
    Recheck admission inside the lock: stale custody authorizes neither a second apply nor an obsolete starter."""
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
        return _delegated_disposition_refusal(status, entry, rid, acknowledge_ambiguous, ctx=ctx)
    captured_before = (entry.patch_manifest_sha or True) if entry.patch_captured else False
    lock_path, held = custody.disposition_lock_path(drive, entry), {}
    try:
        lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=20.0, owner_aware_stale=True, outcome=held)
    except Exception as exc:
        return f"⚠️ INTEGRATE_LOCK_TIMEOUT: could not lock the captured result: {type(exc).__name__}: {exc}."
    if lock_fd is None:
        if held.get("reason") == "contention":
            return "⚠️ INTEGRATE_LOCK_TIMEOUT: another disposition holds this captured result; retry after it completes."
        code = held.get("errno")
        cause = f", errno {code} ({os.strerror(code)})" if isinstance(code, int) else ""
        return (f"⚠️ INTEGRATE_LOCK_TIMEOUT: the captured result's lock was not taken (platform outcome "
                f"{held.get('reason') or 'unknown'}{cause}); nothing was applied or rejected.")
    try:
        from ouroboros.delegate_continuation import disposition_refusal
        from ouroboros.delegate_custody_current import run

        if handover := disposition_refusal(drive, run(drive, rid)):
            return handover
        from ouroboros.headless import native_capture_custody, refresh_native_capture
        with native_capture_custody(drive, entry) as owner:
            result = _integrate_delegated_patch_locked(ctx, rid, decision, reason, acknowledge_ambiguous, paths,
                                                       captured_before=captured_before,
                                                       custody_held=(drive, entry.task_id) if owner else None)
            note = ""
            try:  # this decision's published result and its typed effect stand whatever the refresh meets
                if owner:
                    note = refresh_native_capture(drive, entry.task_id, locked=True,
                        read_refusal=lambda: _si()._shared_read_refusal(ctx, pathlib.Path(entry.target_root), "read_file"))
                elif (_si().load_task_result(drive, entry.task_id) or {}).get("workspace_mode") == "self_worktree":
                    note = "Native capture promotion refused: this run target does not match a registered native write root."
            except Exception as exc:  # noqa: BLE001 - reported beside the decision, never raised over it
                log.warning("Native capture refresh after run %s's decision failed", rid, exc_info=True)
                note = ((f"Native capture publication is still owed: task {entry.task_id}'s capture was not refreshed"
                         if owner else f"Whether task {entry.task_id}'s native capture needs a refresh is unknown: "
                         "its task result could not be read") + f" after this decision ({type(exc).__name__}: {exc}); "
                        "this decision's result above is unchanged.")
            result = append_published_text(ctx, result, "\n" + note if note else "")
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
    return _delegated_disposition_refusal(status, entry, rid, acknowledge_ambiguous=True, ctx=ctx)


def _integrate_delegated_patch_locked(
    ctx: ToolContext,
    run_id: str = "",
    decision: str = "apply",
    reason: str = "",
    acknowledge_ambiguous: bool = False,
    paths: List[str] | None = None,
    captured_before: bool | str | None = None,
    custody_held=None,
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
    Under the shared lock, newly prepared changes must be shown before apply (#1969).
    """
    from ouroboros import delegate_custody as custody, delegate_source_coverage
    from ouroboros.delegate_shared import orphan_disposition_status

    rid = run_id
    drive = custody.custody_root(ctx)
    status, entry, orphan_of = orphan_disposition_status(ctx, drive, rid)
    if captured_before is None and entry is not None:
        captured_before = (entry.patch_manifest_sha or True) if entry.patch_captured else False
    orphan_note = f"(orphan of terminal task {orphan_of}) " if orphan_of else ""
    from ouroboros.delegate_directory import is_directory_run, integrate_directory_result
    from ouroboros.tools.patch_verdict import git_apply_completion

    # Only the starter or its host-confirmed retry successor (never an orphan adopter) may complete a pending
    # Git apply's record, and only from the verdict row of the task its durable intent names.
    prior = (git_apply_completion(drive, entry) if status == custody.OWNED and not orphan_of and entry
             and entry.patch_apply_pending and decision == "apply" and not acknowledge_ambiguous
             and not is_directory_run(entry) and entry.authority_source != "skill_payload" else {})
    refusal = _delegated_disposition_refusal(status, entry, rid, acknowledge_ambiguous or bool(prior), ctx=ctx)
    if refusal:
        return refusal
    if decision == "apply" and entry is not None:
        if refusal := delegate_source_coverage.source_apply_refusal(entry, rid):
            return refusal
    if is_directory_run(entry):
        from ouroboros import claudexor_daemon, execution_domain  # the run's own engine applies or rejects its result
        try:
            with execution_domain.run_gateway(entry, claudexor_daemon.read_owned_gateway) as gateway:
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
    if prior:
        try:
            raw = manifest_path.read_bytes()
            manifest = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError):
            raw, manifest = b"", {}
        from ouroboros.delegate_capture_job import manifest_recorded

        if (not isinstance(manifest, dict) or manifest.get("status") != ARTIFACT_STATUS_READY_WITH_CHANGES
                or not manifest_recorded(entry, raw) or manifest.get("capture_kind") == "skill_payload"
                or hashlib.sha256(raw).hexdigest() != prior["git_apply_completion"]["manifest_sha256"]):
            return _delegated_disposition_refusal(status, entry, rid, ctx=ctx)
        # The record keeps the original apply's whole reason, never this retry's.
        result = _integrate_git_capture(ctx, entry, decision, prior["git_apply_completion"]["reason"], manifest,
                                       cap_dir, orphan_of, custody_held=custody_held, raw_manifest=raw, prior=prior)
        return append_published_text(ctx, result, selection_note)
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.delegate_custody_current import run
    from ouroboros.store import StoreError

    def fresh():  # the receipt this call published, not the row read before it; None when unreadable
        try:
            return run(drive, rid)
        except (OSError, StoreError):
            return None

    def described():  # an unreadable receipt shows the earlier row
        return capture_projection(drive, fresh() or entry, reader_task_id=str(getattr(ctx, "task_id", "") or "")) or {}

    # A different capture that arrived while the caller waited was not seen before this call.
    prepared = not (captured_before == (entry.patch_manifest_sha or True)
                    and _manifest_capture_status(manifest_path, entry=entry) in _READY_CAPTURE_STATUSES)
    capture = _si()._capture_at_disposition(drive, entry, rid, manifest_path, decision, custody_held=custody_held)
    if isinstance(capture, str):
        # A named-evidence refusal stands over a ready capture; once its custody names that capture, a capture
        # this call prepared is shown, as the next call no longer counts it as prepared. No other refusal is.
        if decision == "apply" and prepared and capture.startswith(_EVIDENCE_UNRETAINED) and (
                (row := fresh()) is not None and row.patch_captured
                and _apply_acts_on(_saved_manifest(manifest_path, entry=row), entry.baseline_sha)):
            capture += ("\nIts captured result, prepared during this call, was NOT applied; read it now (manifest_read, "
                        "its file artifacts, and patch_read when present). Once the cause above is fixed, a later "
                        "apply decides this same retained capture.\n" + json.dumps(described(), ensure_ascii=False, indent=2))
        return capture
    if capture.get("disposition_unwritten") == custody.NO_CHANGES:
        return capture["note"]  # do not turn the failed automatic disposition into an explicit one
    if capture.get("disposition") == custody.NO_CHANGES:
        text = _git_disposition_result(ctx, f"OK: delegated run {rid} closed as no_changes. {capture['note']}")
        return append_published_text(ctx, text, selection_note)  # no orphan re-admission after closure
    if refusal := _current_disposition_refusal(ctx, rid):
        return refusal
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            raw = manifest_path.read_bytes()
            loaded = json.loads(raw.decode("utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"⚠️ INTEGRATE_MANIFEST_UNREADABLE: {manifest_path}: {type(exc).__name__}: {exc}."
    if inspection := prepared_capture_inspection(
            rid, decision, _apply_acts_on(manifest, entry.baseline_sha), described, prepared=prepared):
        return inspection
    if entry.authority_source == "skill_payload" or str(manifest.get("capture_kind") or "") == "skill_payload":
        # The exact-payload branch (R1 item 3) lives in delegate_integration:
        # fresh semantic rebinding, whole-payload CAS, reserved-path whole-apply
        # refusal, index-free apply into the live NON-Git payload — no staging.
        from ouroboros.tools.delegate_integration import integrate_payload_patch

        return integrate_payload_patch(
            ctx, drive=drive, entry=entry, rid=rid, decision=decision,
            reason=reason, cap_dir=cap_dir, manifest=manifest, patch_path=patch_path) + (f"\n{orphan_note.rstrip()}" if orphan_note else "") + selection_note
    result = _integrate_git_capture(ctx, entry, decision, reason, manifest, cap_dir, orphan_of,
                                    custody_held=custody_held, named=capture.get("named_evidence"),
                                    raw_manifest=raw if manifest else b"")
    return append_published_text(ctx, result, selection_note)


def _admit_git_apply(
    ctx, entry, rid, target, ordered_touched, expected_digest, prior,
    admission_refusal, dispose,
) -> str | None:
    """Return a refusal or recover the earlier receipt; None admits a fresh apply.

    PRIOR is the verdict row its intent's author recorded for this exact attempt (``git_apply_completion``),
    read for the run's starter or host-confirmed retry successor only. Recovery changes no file: under the
    target's Git lock it re-proves the caller's admission, the attempt's target, patch and state (folder and
    .git identity, HEAD or an unborn branch, touched files, their index entries), then records only the
    native capture effect and, once it lands, the pending disposition with the original reason. A busy target
    is retried later with the intent kept; any unproven fact is the ordinary ambiguity refusal.
    """
    if refusal := admission_refusal():
        return refusal
    if not prior:
        return None

    from ouroboros import delegate_custody as custody
    from ouroboros.tools.git import _acquire_git_lock, _release_git_lock

    try:
        lock = _acquire_git_lock(ctx, repo_dir=target)
    except Exception as exc:  # noqa: BLE001 - as the apply's own lock: busy now, retried later
        return (f"⚠️ INTEGRATE_NOT_STARTED: run {rid}'s earlier apply into {target} could not be checked: the target "
                f"is busy ({type(exc).__name__}: {exc}). Retry later; its apply intent stays recorded and nothing "
                "was changed.")
    witness = prior["git_apply_completion"]
    try:
        if refusal := admission_refusal():
            return refusal
        if (prior.get("patch_sha256") != expected_digest or str(prior.get("target_root") or "") != str(target)
                or staged_postimage(target, ordered_touched) != witness["postimage_sha256"]):
            return _delegated_disposition_refusal(custody.OWNED, entry, rid, ctx=ctx)
        # The native receipt lands before the final disposition: a failed write keeps the intent for this lane.
        native_gap = _native_effect(custody.custody_root(ctx), entry, {
            "proc": subprocess.CompletedProcess([], 0), "unindexed": witness["unindexed"]})
        recorded = not native_gap and dispose("applied", cleanup=False, apply_idempotency_key=entry.patch_apply_key)[0]
    finally:
        _release_git_lock(lock)
    transfer = _transfer_facts(target, ordered_touched, witness["unindexed"])
    if not recorded:
        return _git_unwritten_disposition(ctx, entry, "applied", staging_proof=True, transfer=transfer, changed=False,
                                          native_gap=native_gap)
    return _git_disposition_result(ctx, f"OK: delegated run {rid}'s earlier Git apply is recorded with its "
                                   "original reason. This call changed no target file; cleanup stays with custody GC.",
                                   facts={"disposition_recorded": True})


def _integrate_git_capture(ctx, entry, decision, reason, manifest, cap_dir, orphan_of, *,
                           custody_held=None, named=None, raw_manifest=b"", prior=None):
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
    from ouroboros.workspace_file_outputs import file_output_changes
    try:
        file_changes = file_output_changes(manifest, cap_dir)
    except Exception as exc:
        return _capture_failed_refusal(rid, str(manifest.get("status") or ""), str(exc))
    touched = declared_capture_paths(manifest)

    if named is None and isinstance(evidence := _disposition_evidence(drive, entry, custody_held), str):
        return evidence  # otherwise evidence was retained earlier in this locked call

    def _dispose(disposition: str, cleanup: bool, **row: Any) -> tuple[bool, str]:
        return _dispose_delegated(
            drive, entry, snapshot_key, reason, disposition, cleanup,
            disposed_by=str(getattr(ctx, "task_id", "") or ""), defer_cleanup=custody_held is not None, **row)

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
            return _git_unwritten_disposition(ctx, entry, "rejected")
        if lost:
            return _git_disposition_result(ctx, f"{orphan_note}🚫 Rejected delegated run {rid} as LOST: {lost_words(drive, entry, lost)}. This decision "
                    "changed no files; what the run did outside its copy and what it cost stay as recorded, unknown where "
                    f"unknown. Verdict: {verdict_path or '(unwritten)'}. Reason: {reason or '(none)'}.{note}")
        text = (
            f"{orphan_note}🚫 Rejected delegated run {rid}'s captured patch ({len(touched)} file(s) not "
            f"applied); {'its disposition is recorded' if note else 'its private copy and pins are released'}. "
            f"Verdict: {verdict_path or '(unwritten)'}. "
            f"Reason: {reason or '(none)'}.{format_patch_exclusions(manifest, file_changes)}{note}"
        )
        return _git_disposition_result(ctx, text)

    if capture_status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        if capture_status == ARTIFACT_STATUS_READY_NO_CHANGES:
            recorded, note = _dispose("applied", cleanup=True)
            if not recorded:
                return _git_unwritten_disposition(ctx, entry, "applied")
            return _git_disposition_result(ctx,
                f"OK: delegated run {rid} has no captured file changes to apply; "
                f"its answer and external effects remain separate evidence; its disposition is recorded."
                f"{format_patch_exclusions(manifest)}{note or ' its private copy is released.'}"
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
    from ouroboros import work_domain  # a work target keeps its spelling: the container resolves it
    try:
        active_root = work_domain.settle(ctx.active_repo_dir())
    except Exception as exc:
        return f"⚠️ INTEGRATE_TARGET_ERROR: could not resolve active repo: {type(exc).__name__}: {exc}."
    target = work_domain.settle(str(entry.target_root or ""))
    target_ok = orphan_apply_target_ok(target, active_root) if orphan_of else target == active_root
    if not str(entry.target_root or "").strip() or not target_ok:
        return (
            "⚠️ INTEGRATE_DELEGATED_TARGET_MISMATCH: the run's recorded authority target "
            f"({entry.target_root or '(none)'}) is not this task's active root ({active_root}). "
            "Only a terminal owner's orphan targeting a folder inside the configured projects/worktrees "
            "roots may apply outside the active folder, under current write rights, regardless of creator."
        )
    if not work_domain.exists(target / ".git"):
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
        grant = bool(getattr(constraint, "protected_paths_grant", False))
        if not protected_write_admitted(runtime_mode, acting=is_acting, grant=grant):
            _si()._write_verdict(
                ctx, f"run_{rid}", outcome="blocked_protected", reason=reason, files=touched,
                manifest=manifest, applied=False, conflicts=[],
                protected=[p.path for p in protected], target=str(target))
            return protected_apply_refusal(
                path=protected[0].path, runtime_mode=runtime_mode, grant_ok=not is_acting or grant,
                action=f"integrate delegated run {rid} patch touching")

    def admission_refusal():
        if refusal := _current_disposition_refusal(ctx, rid):
            return refusal
        refusal = copy_apply_refusal(ctx, target, ordered_touched)
        return target_write_refusal(target, refusal) if refusal else ""

    admission_result = _admit_git_apply(ctx, entry, rid, target, ordered_touched, expected_digest, prior,
                                       admission_refusal, _dispose)
    if admission_result is not None:
        return admission_result
    # CR1-3, owed-before-sent: the durable apply-intent row lands BEFORE the tree
    # can be mutated. Without it, a crash between `git apply` and the disposition
    # row replays as "never applied", and a later reject records a false rejection
    # over a modified, staged tree. An unlanded intent row refuses the mutation
    # outright (same doctrine as record_start_requested).
    from uuid import uuid4
    # The intent names its author, the one task whose verdict row a receipt-only retry may read.
    if not custody.record_patch_apply_started(drive, entry, target_root=str(target), apply_idempotency_key=uuid4().hex,
                                              applied_by_task_id=task_id_for_artifacts(ctx)):
        return (
            f"⚠️ INTEGRATE_INTENT_UNWRITTEN: the durable apply-intent row for run "
            f"{rid} could not be written, so a crash mid-apply would leave the "
            "tree state unaccountable. Refusing to mutate; fix the drive/event "
            "log and retry. Nothing was changed."
        )
    from ouroboros.headless import begin_native_capture_update
    try:
        begin_native_capture_update(drive, entry)
    except Exception as exc:
        custody.record_patch_apply_resolved(drive, entry, reason="native_capture_unavailable")
        _native_effect(drive, entry, {"proc": None})
        return f"⚠️ INTEGRATE_NATIVE_CAPTURE_UNAVAILABLE: {exc}. No target files changed."
    file_options = {}
    if file_changes:
        from ouroboros.subagent_worktrees import find_execution_snapshot
        snapshot = find_execution_snapshot(entry.snapshot_id) or {}
        file_options = {"file_changes": file_changes, "file_baseline": snapshot.get("file_baseline") or {}}
    outcome = _si()._locked_apply(
        ctx, target, patch_path, ordered_touched, entry.baseline_sha, baseline_entries=manifest.get("baseline_entries"),
        admission_check=admission_refusal, observe_completion=True, **file_options)
    native_gap = _native_effect(drive, entry, outcome)  # its failure keeps the native debt, never the effect
    if outcome.get("admission_refusal"):
        custody.record_patch_apply_resolved(drive, entry, reason="authority_changed")
        return outcome["admission_refusal"]
    if outcome.get("lock_error"):
        custody.record_patch_apply_resolved(drive, entry, reason="lock_error")
        return (
            f"⚠️ INTEGRATE_NOT_STARTED: run {rid} into {target}: the target is busy "
            f"({outcome['lock_error']}). Retry later."
        )
    proc = outcome["proc"]
    drifted = outcome["drifted"]
    drift_error = outcome["drift_error"]
    staging_failure = outcome["staging_failure"]
    reverted = outcome["reverted"]
    unindexed = outcome.get("unindexed") or []

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
                f"{format_patch_exclusions(manifest, file_changes, 'reverted')}"
            )
        recorded, _ = _dispose("applied", cleanup=False)
        record = ("The run is recorded as applied, so a later call does not apply it again" if recorded else
                  "⚠️ Its disposition could NOT be recorded either: a later call refuses as ambiguous instead of "
                  "applying again; record this outcome yourself before any further integration attempt")
        return _publish_tool_result(ctx, ToolResult("blocked", "INTEGRATION_BLOCKED", (
            f"⚠️ INTEGRATE_APPLIED_UNSTAGED: run {rid}'s result reached {target} over "
            f"{len(ordered_touched)} path(s), but staging failed and the apply could not be cleanly reversed "
            f"({staging_failure[:300]}); which of those paths still carry the run's changes is not known. "
            "Do NOT apply it again: a second apply could double changes still in the working tree. "
            f"Inspect with vcs_read diff and stage what you accept yourself. {record}. Its captured patch is "
            f"kept for comparison. Verdict: {verdict_path or '(unwritten)'}."
            f"{format_patch_exclusions(manifest, file_changes, 'incomplete')}{_native_gap_note(native_gap)}"
        ), {**_typed_transfer(_transfer_facts(target, ordered_touched, unindexed), settled=False),
            "disposition_recorded": recorded}))

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
            f"apply cleanly. git said: {stderr[:600]}\n{_CONFLICT_OWNED}"
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
    # A receipt-only retry's proof rides the verdict's custody row, apart from the optional verdict file.
    witness = {"apply_key": entry.patch_apply_key, "invocation_id": entry.invocation_id,
               "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
               "postimage_sha256": outcome.get("postimage_sha256", ""), "reason": str(reason or ""),
               "unindexed": list(unindexed)}
    failures: List[str] = []
    verdict_path = _si()._write_verdict(
        ctx, f"run_{rid}", outcome="applied", reason=reason, files=touched,
        manifest=manifest, applied=True, conflicts=[],
        protected=[p.path for p in protected], target=str(target), failures=failures,
        observation={**({"unindexed": unindexed} if unindexed else {}),
                     **({"git_apply_completion": witness} if witness["postimage_sha256"] else {})},
    )
    staging_proof = bool(witness["postimage_sha256"]) and "custody_row_write_failed" not in failures
    # A proven apply waits for its native receipt (the receipt-only lane appends it); an unproven one is disposed.
    recorded, note = (False, "") if native_gap and staging_proof else _dispose("applied", cleanup=True)
    if not recorded:
        return _git_unwritten_disposition(ctx, entry, "applied", staging_proof=staging_proof, orphan=bool(orphan_of),
                                          transfer=_transfer_facts(target, ordered_touched, unindexed), native_gap=native_gap)
    diffstat = str(manifest.get("diffstat") or "").strip()
    prot_note = (f" Includes {len(protected)} protected path(s) (allowed: runtime_mode={runtime_mode})."
                 if protected else "")
    staged, delivered, closing = staging_receipt(len(ordered_touched), unindexed)
    return _git_disposition_result(ctx,
        f"{orphan_note}✅ Integrated delegated run {rid}'s patch into {target} ({staged}).{prot_note}{delivered}\n"
        f"{diffstat}{format_patch_exclusions(manifest, file_changes, 'applied', unindexed)}\n"
        f"Verdict: {verdict_path or '(unwritten)'}. "
        f"{'Disposition recorded.' if note else 'Its private copy and pins are released.'}\n"
        f"{closing} — review them yourself; you are the sole committer.{note}{_native_gap_note(native_gap)}",
        changed=True, facts={**_typed_transfer(_transfer_facts(target, ordered_touched, unindexed)), "disposition_recorded": True,
                             **({"native_capture_effect": "unknown"} if native_gap else {}), "verdict_file_recorded": bool(verdict_path)},
    )
