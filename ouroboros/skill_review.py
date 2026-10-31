"""Skill review — the lifecycle verb around ONE wave of the shared review mechanism.

``review_skill`` admits one skill payload: it loads and hashes the payload, runs the deterministic
preflight (a failure persists as ``pending``, non-executable outside Cyber Pro), refuses an
extension the PluginAPI admission cannot ever PASS, freezes the payload as the skill subject
(``tools/review_skill.skill_subject``) and hands it to ``review_change`` — the one wave: panel
composition by strictness, custody, the paid stamp, $0 reuse of an identical subject, a rejoin,
leaving on an addressed message, settlement and the ledger record (``surface="skill"``). The record
is projected into ``state/skills/<name>/review.json``, the gate's cache (``skill_loader``,
``skill_review_status.skill_review_gate``), and the grants follow. Tool registration lives in
``tools/skill_exec.py``; the Skills-page job and the post-review reconciliation (dependencies,
enablement, extension reload, update settlement) in ``skill_review_runner.py``.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.config import get_auto_grant_enabled
from ouroboros.skill_loader import (
    SkillReviewState,
    auto_grant_if_enabled,
    compute_content_hash,
    load_review_state,
    save_review_state,
)
from ouroboros.skill_review_output import render_skill_review_block  # noqa: F401 — the tool's rendering, imported from here
from ouroboros.skill_review_status import (  # noqa: F401 — the status vocabulary, imported from here
    STATUS_BLOCKERS,
    STATUS_CLEAN,
    STATUS_PENDING,
    STATUS_WARNINGS,
    review_status_grandfatherable,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    load_bound_skill,
)
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

SUBSTANTIVE_STATUSES = frozenset({STATUS_CLEAN, STATUS_WARNINGS, STATUS_BLOCKERS})


@dataclass
class SkillReviewOutcome:
    """Return payload from ``review_skill``: the gate's status for one content hash, the findings behind
    it, and the wave's facts (its ledger record, whether it was paid or reused, who sat)."""

    skill_name: str
    status: str  # "clean" | "warnings" | "blockers" | "pending"
    findings: List[Dict[str, Any]] = field(default_factory=list)
    reviewer_models: List[str] = field(default_factory=list)
    content_hash: str = ""
    prompt_chars: int = 0
    cost_usd: float = 0.0
    raw_result: str = ""
    raw_actor_records: List[Dict[str, Any]] = field(default_factory=list)
    advisory_result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    auto_flow: bool = False
    # Bible P3: a single configured reviewer is honored on this executable TRUST
    # gate but the lost cross-model diversity is recorded loudly and durably.
    single_reviewer_no_diversity: bool = False
    requested_keys: List[str] = field(default_factory=list)
    auto_granted_keys: List[str] = field(default_factory=list)
    requested_permissions: List[str] = field(default_factory=list)
    auto_granted_permissions: List[str] = field(default_factory=list)
    review_profile: str = ""
    # The wave's facts: ``paid`` = a panel was physically dispatched for this outcome (the shared paid
    # stamp); ``record_id`` names its ledger record; ``reused`` = a settled record of the same subject
    # answered at $0; ``pending`` = the record is still open (the author left, or a seat still answers).
    paid: bool = False
    record_id: str = ""
    reused: bool = False
    pending: bool = False
    replayed_from_ts: str = ""
    rebuttal_sha256: str = ""
    # Foreground call outcome, not the skill's pending status.
    dispatch_refusal_code: str = ""
    panel: Dict[str, Any] = field(default_factory=dict)
    seats: List[Dict[str, Any]] = field(default_factory=list)
    governance: Dict[str, Any] = field(default_factory=dict)


def _apply_auto_grant_outcome(outcome: SkillReviewOutcome, skill: Any, auto_grant: Any) -> None:
    outcome.requested_keys = list(getattr(auto_grant, "requested_keys", []) or [])
    outcome.auto_granted_keys = list(getattr(auto_grant, "granted_keys", []) or [])
    outcome.requested_permissions = list(getattr(auto_grant, "requested_permissions", []) or [])
    outcome.auto_granted_permissions = list(getattr(auto_grant, "granted_permissions", []) or [])
    # Refresh the policy even on a reused record; explicit task-authorized toggles
    # follow their own lifecycle path rather than this automatic first enable.
    outcome.auto_flow = bool(getattr(skill, "is_self_authored", False) and get_auto_grant_enabled())


# --- Before the wave: the deterministic preflight and the PluginAPI admission ------------------

def _run_deterministic_preflight(
    ctx: Any,
    drive_root: pathlib.Path,
    skill: Any,
    content_hash: str,
    *,
    persist: bool,
    binding: ResolvedResourceBinding | None = None,
) -> Optional[SkillReviewOutcome]:
    """Run deterministic checks before spending panel review tokens.

    FAIL-CLOSED (ABI-1): an exception or unparseable output from the
    preflight machinery is a structural failure of the gate itself, never a
    silent pass — in the owner-attestation path this preflight is the ENTIRE
    replacement for the LLM review, so failing open there would mint trust
    from a broken gate.
    """
    preflight_raw = ""
    try:
        from ouroboros.tools.skill_preflight import _handle_skill_preflight
        preflight_raw = _handle_skill_preflight(
            ctx, skill=skill.name, _resolved_binding=binding,
        )
        preflight = json.loads(preflight_raw)
    except Exception as exc:
        # Infrastructure failure of the gate itself: fail closed WITHOUT
        # persisting — a transient breakage must not clobber live review state
        # the way a genuine payload gate failure (below) deliberately does.
        return SkillReviewOutcome(
            skill_name=skill.name,
            status=STATUS_PENDING,
            content_hash=content_hash,
            error=(
                "deterministic preflight infrastructure failure "
                f"(fail-closed, nothing persisted): {type(exc).__name__}: {exc}"
            ),
        )
    if not isinstance(preflight, dict):
        preflight = {"ok": False, "error": "deterministic preflight returned a non-object result (fail-closed)"}
    if preflight.get("ok", True):
        return None
    findings = [{
        "item": "skill_preflight",
        "verdict": "FAIL",
        "severity": "critical",
        "reason": json.dumps(preflight, ensure_ascii=False),
        "model": "deterministic_preflight",
    }]
    # A deterministic preflight failure is a structural fact, not an LLM verdict,
    # so it is persisted as PENDING (non-executable outside Cyber Pro, in every
    # readiness/execution caller) rather than BLOCKERS (which advisory
    # enforcement would let an operator override). The skill_preflight FAIL finding
    # still records exactly why, just like the other PENDING review-time failures.
    outcome = SkillReviewOutcome(
        skill_name=skill.name,
        status=STATUS_PENDING,
        findings=findings,
        reviewer_models=["deterministic_preflight"],
        content_hash=content_hash,
        error="deterministic skill_preflight failed before LLM review; the review stays PENDING",
        raw_result=preflight_raw,
    )
    if persist:
        review_state = SkillReviewState(
            status=outcome.status,
            content_hash=content_hash,
            findings=findings,
            reviewer_models=outcome.reviewer_models,
            timestamp=utc_now_iso(),
            prompt_chars=0,
            cost_usd=0.0,
            raw_result=outcome.raw_result,
            raw_actor_records=[],
        )
        save_review_state(drive_root, skill.name, review_state)
        skill.review = review_state
        # Record what the skill requests (transparency in the review block) but
        # NEVER auto-grant a skill that FAILED the deterministic preflight gate
        # (invalid manifest, sensitive-shaped file, binary payload, path escape).
        from ouroboros.skill_loader import requested_core_setting_keys, requested_skill_permissions

        manifest = getattr(skill, "manifest", None)
        outcome.requested_keys = requested_core_setting_keys(list(getattr(manifest, "env_from_settings", []) or []))
        outcome.requested_permissions = requested_skill_permissions(
            list(getattr(manifest, "permissions", []) or []), list(getattr(manifest, "subscribe_events", []) or []))
    return outcome


def run_acceptance_preflight(ctx: Any, drive_root: pathlib.Path, skill_name: str) -> Dict[str, Any]:
    """The deterministic floor for bytes a source's trust accepts without a review.

    Installers call it after landing a trusted release, inside their lifecycle job.
    A payload failure persists the ordinary PENDING preflight record for these bytes,
    which the gate honours before the install basis; an infrastructure failure persists
    nothing and returns ``ok: False`` so the installer keeps the release off.
    """
    from ouroboros.skill_loader import find_skill

    skill = find_skill(pathlib.Path(drive_root), skill_name)
    if skill is None or skill.load_error or not skill.content_hash:
        return {"ok": False, "error": getattr(skill, "load_error", "") or f"skill {skill_name!r} not found"}
    outcome = _run_deterministic_preflight(ctx, pathlib.Path(drive_root), skill, skill.content_hash, persist=True)
    if outcome is not None:
        return {"ok": False, "error": outcome.error, "content_hash": skill.content_hash}
    return {"ok": True, "content_hash": skill.content_hash}


def plugin_api_admission_refusal_outcome(skill: Any, drive_root: pathlib.Path, *, content_hash: str,
                                         admission_error: str, persist: bool) -> SkillReviewOutcome:
    """ABI-1: typed $0 refusal at NEW-PASS issuance for an inadmissible extension.

    The predicate (``extension_new_pass_admission_error``) is common to every
    PASS-minting path and lives OUTSIDE the deterministic preflight. Clobber
    guard: when the SAME bytes already hold a live executable verdict (the
    grandfather), nothing is persisted — a repeat review must never destroy
    the hash-bound PASS the grandfather construction depends on.
    """
    findings = [{"item": "plugin_api_admission", "verdict": "FAIL", "severity": "critical",
                 "reason": admission_error, "model": "plugin_api_admission"}]
    live = getattr(skill, "review", None)
    live_pass = bool(live is not None and not live.is_stale_for(content_hash) and review_status_grandfatherable(live.status))
    outcome = SkillReviewOutcome(
        skill_name=skill.name, status=STATUS_PENDING, findings=findings, reviewer_models=["plugin_api_admission"],
        content_hash=content_hash,
        error="new review PASS refused (PluginAPI 2.0 admission): " + admission_error + (
            "; the existing hash-bound PASS for these bytes is preserved (grandfather) and nothing was persisted"
            if live_pass else ""))
    if persist and not live_pass:
        review_state = SkillReviewState(status=outcome.status, content_hash=content_hash, findings=findings,
                                        reviewer_models=outcome.reviewer_models, timestamp=utc_now_iso())
        save_review_state(drive_root, skill.name, review_state)
        skill.review = review_state
    return outcome


def _admission_gate(skill: Any, drive_root: pathlib.Path, content_hash: str, persist: bool) -> Optional[SkillReviewOutcome]:
    """ABI-1 admission at NEW-PASS issuance, checked BEFORE a paid panel that could never mint a
    PASS for these bytes ($0 typed refusal; a settled record of the same subject still answers first)."""
    from ouroboros.contracts.plugin_api import extension_new_pass_admission_error

    admission_error = extension_new_pass_admission_error(skill.manifest)
    if not admission_error:
        return None
    return plugin_api_admission_refusal_outcome(skill, drive_root, content_hash=content_hash,
                                                admission_error=admission_error, persist=persist)


# --- The wave and its projection -----------------------------------------------------------------

def _pending(skill_name: str, error: str, *, content_hash: str = "", code: str = "") -> SkillReviewOutcome:
    return SkillReviewOutcome(skill_name=skill_name, status=STATUS_PENDING, content_hash=content_hash, error=error,
                              dispatch_refusal_code=code)


def _compose(ctx: Any, reviewers: Any, reason: str) -> Tuple[Any, List[str], str]:
    """``(panel, reviewers, reason)`` of this call: the lifecycle job's panel (composed by its source, in
    force on the context); the request the job could NOT compose (``_skill_review_panel_request``), handed
    to the wave as it is — the wave collects an open paid wave of these bytes under its recorded panel or
    refuses the same way, typed and at $0; or, for a direct call, one composed here by strictness."""
    # The request is handed on as the caller wrote it: a value of the wrong type is the panel's own typed
    # REVIEW_PANEL_ARGUMENT refusal, never a string split into one-letter reviewer names.
    prior = getattr(ctx, "_skill_review_composed_panel", None)
    if prior is not None:
        return prior, _as_given(reviewers), reason
    from ouroboros.reviewer_slot_config import compose_review_panel

    request = getattr(ctx, "_skill_review_panel_request", None)
    if request:
        named, why = _as_given(request[0]), request[1]
        if not isinstance(named, list) or not isinstance(why, str):
            compose_review_panel(named, why, surface="skill")  # the panel's own typed REVIEW_PANEL_ARGUMENT
        return None, named, why

    return compose_review_panel(_as_given(reviewers), reason if reason is not None else "", surface="skill"), \
        _as_given(reviewers), reason


def _as_given(reviewers: Any) -> Any:
    """The reviewers argument as the caller gave it (a copy of a list or tuple; ``None`` → no names)."""
    if reviewers is None:
        return []
    return list(reviewers) if isinstance(reviewers, (list, tuple)) else reviewers


def _outcome_from_result(skill: Any, subject: Any, result: Dict[str, Any], *, rebuttal_sha: str) -> SkillReviewOutcome:
    """One wave result (``review_change.review_result``) → the gate's outcome (``tools/review_skill.outcome_status``)."""
    from ouroboros.tools.review_skill import legacy_actor_records, outcome_status, reviewer_models, seat_facts

    status, findings, error = outcome_status(result)
    refusal = result.get("dispatch_refusal") or {}
    seats = seat_facts(result, subject.delivery)
    packet_chars = [int(seat.get("brief_chars") or 0) for seat in seats if seat.get("delivery") == "packet"]
    cost = dict(result.get("cost") or {})
    return SkillReviewOutcome(
        skill_name=skill.name, status=status, findings=findings, reviewer_models=reviewer_models(result),
        content_hash=subject.tree_sha, prompt_chars=max(packet_chars) if packet_chars else 0,
        cost_usd=float(cost.get("usd") or 0.0), raw_actor_records=legacy_actor_records(result), error=error,
        single_reviewer_no_diversity=int((result.get("panel") or {}).get("seats") or 0) < 2,
        paid=not result.get("reused") and not refusal, record_id=str(result.get("record_id") or ""),
        reused=bool(result.get("reused")), pending=str(result.get("state") or "") == "pending",
        replayed_from_ts=str(result.get("reused_ts") or ""), rebuttal_sha256=rebuttal_sha,
        dispatch_refusal_code="LEGACY_BLOCKED" if str(refusal.get("kind") or "") == "review_cycles_exhausted" else "",
        panel=dict(result.get("panel") or {}), seats=seats,
        governance=dict((subject.delivery or {}).get("governance") or {}))


def review_skill(ctx: Any, skill_name: str, *, persist: bool = True, review_rebuttal: str = "",
                 _resolved_binding: ResolvedResourceBinding | None = None,
                 reviewers: Optional[List[str]] = None, reason: str = "") -> SkillReviewOutcome:
    """Admit one skill payload: preflight, admission, then the one review wave; persist its projection."""
    from ouroboros.body_candidate import serving_repo_dir_for
    from ouroboros.reviewer_slot_config import PanelCompositionError
    from ouroboros.skill_loader import SkillPayloadUnreadable
    from ouroboros.skill_lifecycle_queue import lend_lifecycle_lane
    from ouroboros.tools.commit_gate import compute_rebuttal_sha256
    from ouroboros.tools.review_change import ReviewChangeArgumentError, run_review_change
    from ouroboros.tools.review_skill import skill_subject
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, root="skill_payload", operation="review", path=".", skill_name=skill_name)
    except Exception as exc:
        return _pending(skill_name, str(exc))
    drive_root = binding.state_drive_root
    skill = load_bound_skill(binding)
    if skill is None:
        return _pending(skill_name, f"Skill {skill_name!r} not found")
    if skill.load_error:
        return _pending(skill_name, f"Skill manifest could not be parsed: {skill.load_error}")
    try:
        content_hash = compute_content_hash(skill.skill_dir, manifest_entry=skill.manifest.entry,
                                            manifest_scripts=skill.manifest.scripts)
    except SkillPayloadUnreadable as exc:
        return _pending(skill.name, f"Skill payload {exc.relpath!r} is unreadable ({type(exc.err).__name__}: {exc.err}). "
                                    "Review refuses to emit a PASS over a partial hash — fix file permissions or remove "
                                    "the unreadable file and re-run.")
    try:
        panel, reviewers, reason = _compose(ctx, reviewers, reason)
    except PanelCompositionError as exc:
        return _pending(skill_name, str(exc), content_hash=content_hash, code="TOOL_ARG_ERROR")
    except ValueError as exc:  # an invalid pool is the wave's own typed configuration refusal
        return _pending(skill.name, f"invalid review pool configuration blocks skill review: {exc}", content_hash=content_hash)
    preflight = _run_deterministic_preflight(ctx, drive_root, skill, content_hash, persist=persist, binding=binding)
    try:
        current_hash = compute_content_hash(skill.skill_dir, manifest_entry=skill.manifest.entry,
                                            manifest_scripts=skill.manifest.scripts)
    except SkillPayloadUnreadable as exc:
        return _pending(skill.name, str(exc), content_hash=content_hash)
    if current_hash != content_hash:
        return _pending(skill.name, "Skill payload changed during deterministic preflight; review did not dispatch.",
                        content_hash=content_hash)
    if preflight is not None:
        return preflight
    admission = _admission_gate(skill, drive_root, content_hash, persist)
    if admission is not None:
        return admission
    try:
        subject = skill_subject(ctx, skill, content_hash, drive_root=drive_root, governance_root=serving_repo_dir_for(ctx))
        # The panel reads the frozen copy bound to content_hash and mutates no skill state, so other
        # lifecycle work runs meanwhile; the lane is back before anything below persists. Every send of the
        # wave names this skill (its attempt, its llm_usage row, a session seat's start custody a late
        # collection reads), as ``_usage_rows.REVIEW_CUSTODY_KEYS`` says; the caller's own scope stays.
        scope = replace(current_usage_scope() or UsageScope(), review_skill=skill.name)
        with lend_lifecycle_lane(), usage_scope(scope):
            result = run_review_change(ctx, subject="skill", reviewers=reviewers, reason=reason if reason is not None else "",
                                       review_rebuttal=str(review_rebuttal or ""), _frozen_skill=subject, _panel=panel)
    except ReviewChangeArgumentError as exc:
        return _pending(skill.name, str(exc), content_hash=content_hash, code="TOOL_ARG_ERROR")
    outcome = _outcome_from_result(skill, subject, result, rebuttal_sha=str(compute_rebuttal_sha256(review_rebuttal) or ""))
    if not outcome.panel and panel is not None:
        outcome.panel = dict(panel.facts)
    if persist and not outcome.pending:
        _persist_projection(ctx, skill, drive_root, outcome)
    return outcome


def _persist_projection(ctx: Any, skill: Any, drive_root: pathlib.Path, outcome: SkillReviewOutcome) -> None:
    """The record's projection into ``review.json`` (the gate's cache) and the grants that follow.

    The wave ran with the lifecycle lane lent out, so the payload may have moved on (an update, an edit, an
    uninstall): the verdict lands only while the installed bytes are still the reviewed bytes, and only while
    the lifecycle job that owns this review still does. A wave without a verdict (quorum failed, not
    dispatched, not performed) never overwrites what already admits the same bytes: a substantive verdict,
    the owner's attestation, or an author acceptance bound to them."""
    from ouroboros.skill_loader import SkillPayloadUnreadable

    try:
        current_hash = compute_content_hash(skill.skill_dir, manifest_entry=skill.manifest.entry,
                                            manifest_scripts=skill.manifest.scripts)
    except (OSError, SkillPayloadUnreadable):
        current_hash = ""
    if current_hash != outcome.content_hash:
        outcome.status = STATUS_PENDING
        outcome.error = ("review outcome was not persisted because the skill payload changed while the review ran; "
                         "review the current bytes")
        return
    if getattr(ctx, "_skill_review_lifecycle_guard", False):
        from ouroboros.skill_review_runner import _can_persist_review_outcome

        if not _can_persist_review_outcome(drive_root, skill.name, outcome.content_hash,
                                           expected_job_id=str(getattr(ctx, "_skill_review_lifecycle_job_id", "") or "")):
            outcome.status = STATUS_PENDING
            outcome.error = ("review outcome was not persisted because the lifecycle job is already terminal or no "
                             "longer matches this content hash")
            return
    existing = load_review_state(drive_root, skill.name, skill_type=skill.manifest.type, skill_dir=skill.skill_dir)
    same_bytes = existing.content_hash == outcome.content_hash and existing.status in SUBSTANTIVE_STATUSES
    attested = same_bytes and existing.review_profile == "owner_attested"
    held = same_bytes and not attested
    # An author acceptance (Advisory finish) is bound to the bytes it accepted, not to the reviewer's hash.
    accepted = str((existing.author_disposition or {}).get("subject_hash") or "") == outcome.content_hash
    if outcome.status == STATUS_PENDING and (held or attested or accepted):
        # A wave that formed no verdict never erases what already admits these exact bytes: the owner's
        # Skip, the author's acceptance, or an earlier substantive verdict (opus R1-1). A substantive
        # outcome below still replaces each of them.
        stands = ("the owner's attestation" if attested else f"the earlier {existing.status} verdict" if held
                  else "the author's acceptance")
        outcome.error = (f"{outcome.error}; {stands} for these bytes stands"
                         + (f" (record {outcome.record_id} formed no verdict)" if outcome.record_id else "")).strip("; ")
        skill.review = existing
        return
    if outcome.reused and held and existing.record_id == outcome.record_id:
        skill.review = existing  # the same record is already projected: nothing to write
    else:
        state = SkillReviewState(
            status=outcome.status, content_hash=outcome.content_hash, findings=list(outcome.findings),
            reviewer_models=list(outcome.reviewer_models), timestamp=utc_now_iso(), prompt_chars=outcome.prompt_chars,
            cost_usd=outcome.cost_usd, raw_result="", raw_actor_records=list(outcome.raw_actor_records),
            review_profile=outcome.review_profile, panel=dict(outcome.panel), record_id=outcome.record_id,
            governance=dict(outcome.governance), seats=list(outcome.seats))
        try:
            save_review_state(drive_root, skill.name, state)
        except OSError as exc:  # recoverable by record identity: the next call finds the settled record and re-projects
            log.warning("review.json of %s not written for record %s: %s", skill.name, outcome.record_id, exc)
            outcome.error = f"{outcome.error}; the projection into review.json failed ({exc}); record {outcome.record_id} stands".strip("; ")
            return
        skill.review = state
    _apply_auto_grant_outcome(outcome, skill, auto_grant_if_enabled(drive_root, skill))


__all__ = [
    "SkillReviewOutcome",
    "render_skill_review_block",
    "review_skill",
    "run_acceptance_preflight",
]
