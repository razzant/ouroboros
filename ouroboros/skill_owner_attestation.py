"""Owner attestation ("Skip review" / "Turn on anyway").

The OWNER may skip the LLM skill review for the exact current bytes of a skill from any
origin (built-in, a hub source, ClawHub, a local folder, made here) while the
DETERMINISTIC preflight floor still gates it. It is a local authorization fact of this
install: a record for these bytes, never a critic PASS another install could reuse; the
critic findings of earlier runs stay in the review history. Only the shared owner action
(``skill_lifecycle_actions.run_skill_action``) reaches it, with a host-derived owner
caller. Extracted from skill_review.py to keep that module under the size gate; it reuses
skill_review's preflight + persistence machinery (referenced module-qualified so tests can
monkeypatch).
"""

from __future__ import annotations

import pathlib
from typing import Any

from ouroboros import skill_review as _sr
from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.skill_loader import (
    SKILL_OWNER_STATE_SCHEMA_VERSION,
    SkillPayloadUnreadable,
    SkillReviewState,
    compute_content_hash,
    find_skill,
    load_review_state,
    save_review_state,
    skill_state_dir,
)
from ouroboros.utils import atomic_write_json, utc_now_iso
from ouroboros.tool_access import canonical_data_root, load_bound_skill


def run_owner_attestation(ctx: Any, drive_root: pathlib.Path, skill: Any, content_hash: str):
    """Owner-attested skill review — skip the EXPENSIVE LLM review for these exact bytes.
    The DETERMINISTIC preflight floor STILL runs (an invalid manifest / sensitive-shaped /
    path-escaping payload can NEVER be attested); only the LLM phase is skipped.
    On a clean preflight, persist a durable CLEAN verdict bound to the content_hash
    (review_profile='owner_attested', reviewer_models=['owner_attestation'], one explicit
    PASS finding so the status serializes) and drop the owner-issued marker that
    load_review_state requires for the verdict to stay valid. The acting model
    interprets owner intent; the shared lifecycle action owner validates source
    provenance, caller and selected revision. Generic agent file writes cannot
    forge the owner-state marker."""
    # A FAILED attestation preflight persists as a normal review result (so the gate's
    # fresh ``preflight_failed`` fact and the Repair affordance appear) — but ONLY when
    # persisting cannot clobber a FRESH valid verdict: review.json absent, or its recorded
    # content_hash stale for the current payload bytes. A stale CLEAN is knowingly
    # superseded by a FAIL at the current hash (the fact about the CURRENT bytes wins;
    # the prior verdict stays on its review ledger record). With a fresh verdict on file the
    # failure still surfaces only through the endpoint (409) and leaves state as-is.
    persist_failed_preflight = load_review_state(
        drive_root, skill.name, skill_dir=getattr(skill, "skill_dir", None),
    ).is_stale_for(content_hash)
    preflight_outcome = _sr._run_deterministic_preflight(
        ctx, drive_root, skill, content_hash, persist=persist_failed_preflight,
        binding=getattr(ctx, "_skill_review_resolved_binding", None),
    )
    if preflight_outcome is not None:
        # Deterministic preflight FAILED -> NOT attestable (the LLM phase is skipped, this
        # floor is not). The PENDING outcome carries the skill_preflight FAIL finding.
        return preflight_outcome
    # The deterministic floor must also cover what the SKIPPED LLM manifest reviewer would
    # have caught: a parsed-but-structurally-invalid manifest surfaces only as
    # SkillManifest.validate() warnings. Owner-attestation refuses on ANY such warning.
    manifest = getattr(skill, "manifest", None)
    validate_warnings = (
        list(manifest.validate()) if manifest is not None and hasattr(manifest, "validate") else []
    )
    if validate_warnings:
        return _sr.SkillReviewOutcome(
            skill_name=skill.name, status=_sr.STATUS_PENDING, content_hash=content_hash,
            error=("owner-attestation refused: the manifest has validation issues the LLM "
                   "manifest reviewer would flag (" + "; ".join(str(w) for w in validate_warnings[:5])
                   + "); fix them or run the full skill_review"),
        )
    # ABI-1: the NEW-PASS admission predicate is common to every PASS-minting
    # path — an attestation is a PASS issuance, so a field-less extension is
    # refused here exactly like in the LLM review path. Nothing is persisted,
    # so an existing hash-bound PASS (grandfather) stays untouched.
    from ouroboros.contracts.plugin_api import extension_new_pass_admission_error

    admission_error = extension_new_pass_admission_error(manifest)
    if admission_error:
        return _sr.SkillReviewOutcome(
            skill_name=skill.name, status=_sr.STATUS_PENDING, content_hash=content_hash,
            error=f"owner-attestation refused (PluginAPI 2.0 admission): {admission_error}",
        )
    findings = [{
        "item": "owner_attestation",
        "verdict": "PASS",
        "severity": "info",
        "reason": "Owner attested this skill — the expensive LLM review was skipped; the "
                  "deterministic preflight floor still passed.",
        "model": "owner_attestation",
    }]
    # Honor the SAME lifecycle persistence guard as review_skill: when run through the
    # lifecycle, do not persist a clean verdict if the job is already terminal or the hash
    # no longer matches (a stale/superseded attestation must not overwrite live state).
    if getattr(ctx, "_skill_review_lifecycle_guard", False):
        from ouroboros.skill_review_runner import _can_persist_review_outcome
        if not _can_persist_review_outcome(
            drive_root, skill.name, content_hash,
            expected_job_id=str(getattr(ctx, "_skill_review_lifecycle_job_id", "") or ""),
        ):
            return _sr.SkillReviewOutcome(
                skill_name=skill.name, status=_sr.STATUS_PENDING, content_hash=content_hash,
                error="owner-attestation was not persisted: the lifecycle job is terminal or "
                      "no longer matches this content hash",
            )
    review_state = SkillReviewState(
        status=_sr.STATUS_CLEAN,
        content_hash=content_hash,
        findings=findings,
        reviewer_models=["owner_attestation"],
        timestamp=utc_now_iso(),
        prompt_chars=0,
        cost_usd=0.0,
        raw_result="",
        raw_actor_records=[],
        review_profile="owner_attested",
    )
    save_review_state(drive_root, skill.name, review_state)
    marker_path = skill_state_dir(drive_root, skill.name) / "owner_attestation.json"
    atomic_write_json(marker_path, with_schema_version(
        {"attested_at": utc_now_iso(), "content_hash": content_hash},
        SKILL_OWNER_STATE_SCHEMA_VERSION,
    ))
    skill.review = review_state
    outcome = _sr.SkillReviewOutcome(
        skill_name=skill.name,
        status=_sr.STATUS_CLEAN,
        findings=findings,
        reviewer_models=["owner_attestation"],
        content_hash=content_hash,
        review_profile="owner_attested",
    )
    _sr._apply_auto_grant_outcome(outcome, skill, _sr.auto_grant_if_enabled(drive_root, skill))
    return outcome


def review_skill_owner_attest(ctx: Any, skill_name: str):
    """``review_impl`` variant for the owner-attestation endpoint: load + content-hash the
    skill exactly like ``review_skill``, then run the preflight-floored owner attestation
    (no LLM phase). Reused through ``run_skill_review_lifecycle`` so it shares the standard
    job tracking / payload shape."""
    binding = getattr(ctx, "_skill_review_resolved_binding", None)
    drive_root = (
        pathlib.Path(binding.state_drive_root)
        if binding is not None
        else canonical_data_root(ctx)
    )
    skill = (
        load_bound_skill(binding)
        if binding is not None
        else find_skill(drive_root, skill_name)
    )
    if skill is None:
        return _sr.SkillReviewOutcome(
            skill_name=skill_name, status=_sr.STATUS_PENDING,
            error=f"Skill {skill_name!r} not found in the external skills checkout",
        )
    if skill.load_error:
        return _sr.SkillReviewOutcome(
            skill_name=skill_name, status=_sr.STATUS_PENDING,
            error=f"Skill manifest could not be parsed: {skill.load_error}",
        )
    # Every origin is attestable (built-in, hub, ClawHub, local, made here): the owner
    # decides for this install. The caller already proved an owner actor and the exact
    # selected revision; the deterministic floor below still refuses broken payloads.
    try:
        content_hash = compute_content_hash(
            skill.skill_dir,
            manifest_entry=skill.manifest.entry,
            manifest_scripts=skill.manifest.scripts,
        )
    except SkillPayloadUnreadable as exc:
        return _sr.SkillReviewOutcome(
            skill_name=skill.name, status=_sr.STATUS_PENDING,
            error=f"Skill payload {exc.relpath!r} is unreadable; cannot owner-attest a partial hash.",
        )
    return run_owner_attestation(ctx, drive_root, skill, content_hash)
