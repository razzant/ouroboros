"""The skill verdict's vocabulary and the run gate.

The severity policy (``skill_severity_policy``) is the checklist's own rule of which items block:
hard trust-boundary items whatever the reviewer said, severity-driven items by the reviewer's
word, advisory-only items never; it is applied to a wave's findings BEFORE the shared verdict is
reduced (``review_ledger.apply_severity_policy``) and again when a persisted verdict is read back
(``status_from_findings``), so ``skill_review`` and ``skill_loader`` compute one status. The gate
(``skill_review_gate``) turns that status, the acceptance bases and the author's finish into one
execution authority. Deliberately free of imports from the review stack: ``skill_loader`` reads it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

STATUS_CLEAN = "clean"
STATUS_WARNINGS = "warnings"
STATUS_BLOCKERS = "blockers"
STATUS_PENDING = "pending"

_LEGACY_STATUS_ALIASES = {
    "pass": STATUS_CLEAN,
    "advisory_pass": STATUS_WARNINGS,
    "advisory": STATUS_WARNINGS,
    "fail": STATUS_BLOCKERS,
    "pending": STATUS_PENDING,
    "pending_phase4": STATUS_PENDING,
}

VALID_SKILL_REVIEW_STATUSES = frozenset({
    STATUS_CLEAN,
    STATUS_WARNINGS,
    STATUS_BLOCKERS,
    STATUS_PENDING,
    *_LEGACY_STATUS_ALIASES.keys(),
})


HARD_CRITICAL_ITEMS = frozenset({
    "manifest_schema",
    # NOTE: skill_preflight is intentionally NOT here — a deterministic preflight
    # FAIL aggregates to STATUS_PENDING (handled before the severity loop below),
    # which is non-executable under every enforcement mode (stronger than blocker).
    "permissions_honesty",
    "no_repo_mutation",
    "path_confinement",
    "env_allowlist",
    "inject_chat_minimization",
    "event_subscription_minimization",
    "host_token_handling",
})

SEVERITY_DRIVEN_ITEMS = frozenset({
    "bug_hunting",
    "companion_process_safety",
    "extension_namespace_discipline",
    "widget_module_safety",
    "context_snapshot_discipline",
})
# Items only an extension carries: a FAIL on another type is advice, never a blocker.
EXTENSION_ONLY_ITEMS = frozenset({"extension_namespace_discipline", "widget_module_safety"})
# What a policy says about one item: the rule outranks the seat's word, keeps it, or applies to the rest.
SEVERITY_CRITICAL, SEVERITY_ADVISORY, SEVERITY_SEAT = "critical", "advisory", "seat"


def skill_severity_policy(skill_type: str, *, is_module_widget: bool = False, review_profile: str = "") -> Dict[str, str]:
    """The checklist's rule of which items block, as a declared per-item severity policy
    (``review_ledger.apply_severity_policy``): hard trust-boundary items are critical whatever the reviewer
    said; severity-driven items follow the reviewer's word (``seat``: advisory/warning stays advisory, anything
    else is critical), except the extension-only items on a non-extension and every severity-driven item under
    the ``official_hub`` profile (a hash-verified official payload already reviewed at submission), which are
    advisory; an item outside the checklist's table (``*``) never blocks."""
    policy: Dict[str, str] = {item: SEVERITY_CRITICAL for item in HARD_CRITICAL_ITEMS}
    for item in SEVERITY_DRIVEN_ITEMS:
        if review_profile == "official_hub" or (item in EXTENSION_ONLY_ITEMS and skill_type != "extension"):
            policy[item] = SEVERITY_ADVISORY
        else:
            policy[item] = SEVERITY_SEAT
    policy["*"] = SEVERITY_ADVISORY
    return policy


def status_from_findings(findings: List[Dict[str, Any]], skill_type: str, *, is_module_widget: bool = False,
                         review_profile: str = "") -> str:
    """The enforcement-independent verdict of persisted or collected findings: ``pending`` on a deterministic
    preflight (or PluginAPI admission) failure, ``blockers`` when any counted FAIL is critical under the policy,
    ``warnings`` when only advisory FAILs remain, else ``clean``. An adviser's finding (``additional``) is heard,
    never counted. The policy is applied by the ledger's one rule (``review_ledger.apply_severity_policy``,
    the same that decides a wave's seat verdicts; a missing or unknown severity under ``seat`` reads critical)."""
    from ouroboros.review_ledger import apply_severity_policy  # lazy: no import-time edge into the review stack

    counted = [f for f in findings if isinstance(f, dict) and not f.get("additional")]
    if preflight_failed(counted):
        return STATUS_PENDING
    policy = skill_severity_policy(skill_type, is_module_widget=is_module_widget, review_profile=review_profile)
    failed = [f for f in counted if str(f.get("verdict") or "").upper() == "FAIL"]
    judged = apply_severity_policy({"status": "", "findings": failed}, policy)["findings"]
    if any(f.get("severity") == SEVERITY_CRITICAL for f in judged):
        return STATUS_BLOCKERS
    return STATUS_WARNINGS if failed else STATUS_CLEAN


def normalize_skill_review_status(status: str) -> str:
    """Return the canonical skill review status for current code."""
    raw_status = str(status or "").strip().lower()
    return _LEGACY_STATUS_ALIASES.get(raw_status, raw_status if raw_status in {
        STATUS_CLEAN,
        STATUS_WARNINGS,
        STATUS_BLOCKERS,
        STATUS_PENDING,
    } else STATUS_PENDING)


def review_status_grandfatherable(status: str) -> bool:
    """ABI-1 grandfather predicate, shared by the PluginAPI admission refusal
    path (``skill_review.plugin_api_admission_refusal_outcome``) and the
    RC auditor: only a ``clean``/``warnings`` verdict counts as the hash-bound
    PASS that keeps a plugin_api-less extension loading. Deliberately
    ENFORCEMENT-INDEPENDENT: advisory enforcement can make a BLOCKERS verdict
    executable (``skill_review_gate``), but it never makes one a grandfather —
    the refusal path persists ``pending`` over anything else."""
    return normalize_skill_review_status(status) in (STATUS_CLEAN, STATUS_WARNINGS)


def preflight_failed(findings: Any) -> bool:
    """True when the persisted findings carry a deterministic preflight FAIL.

    The SSOT for the shape is the finding `_run_deterministic_preflight`
    persists (item=skill_preflight / model=deterministic_preflight) — plus the
    ABI-1 PluginAPI admission refusal (item/model=plugin_api_admission), the
    same structural-gate class; the same condition drives the pending
    aggregation above. UI cards use this fact to offer Repair instead of a
    Re-review that would deterministically fail again (#335).
    """
    for finding in findings or []:
        if not isinstance(finding, dict) or finding.get("additional"):
            continue
        if finding.get("verdict") == "FAIL" and (
            finding.get("item") in ("skill_preflight", "plugin_api_admission")
            or str(finding.get("model") or "") in ("deterministic_preflight", "plugin_api_admission")
        ):
            return True
    return False


# Acceptance reasons that make the current bytes executable without a critic PASS.
ACCEPTANCE_REASONS = frozenset({"trusted_source", "self_check", "owner_skip", "owner_edit", "author_accepted_advisory"})


def _acceptance_reason(acceptance: Any, current_hash: str) -> str:
    """The local acceptance basis bound to the current bytes, or "".

    ``acceptance`` carries host-owned install facts (``SkillReviewState.acceptance``):
    the install basis with its release digest, and the owner-edit digest.
    """
    if not current_hash or not isinstance(acceptance, dict):
        return ""
    basis = str(acceptance.get("basis") or "")
    if basis in {"trusted", "self_check", "owner_skip"} and acceptance.get("release_digest") == current_hash:
        return "trusted_source" if basis == "trusted" else basis
    if acceptance.get("owner_edit_digest") == current_hash:
        return "owner_edit"
    return ""


def cyber_authority() -> bool:
    """Whether Ouroboros acts on its own judgment now (Cyber Pro): reviews inform, never hold."""
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    return not review_enforcement_blocks("blocking")


def acceptance_in_force(acceptance: Any, current_hash: str, *, cyber: Optional[bool] = None) -> str:
    """The acceptance basis that admits the current bytes now, or "".

    Ouroboros's own recorded check (``self_check``) counts only while it holds Cyber Pro
    authority: outside Cyber Pro a source that reviews each skill needs its review, and only
    the owner skips one. The other bases do not depend on the mode.
    """
    reason = _acceptance_reason(acceptance, current_hash)
    if reason == "self_check" and not (cyber_authority() if cyber is None else cyber):
        return "owner_edit" if acceptance.get("owner_edit_digest") == current_hash else ""
    return reason


_ACCEPTANCE_SUMMARIES = {
    "trusted_source": "Installed unchanged from a source the owner trusts; no review is required for these exact bytes.",
    "self_check": "Ouroboros read this exact release and recorded why it may run.",
    "owner_skip": "The owner skipped the review for these exact bytes.",
    "owner_edit": "The owner, or Ouroboros on the owner's instruction, edited these exact bytes.",
}


def skill_review_gate(
    status: str, *, stale: bool = False, enforcement: Optional[str] = None,
    findings: Any = None, author_disposition: Any = None, current_hash: str = "",
    acceptance: Any = None,
) -> Dict[str, Any]:
    """Structured, agent-facing explanation of whether a review is executable.

    Advisory author acceptance binds changed bytes; Cyber retains judgment even
    without a prior verdict. Stale describes reviewer evidence, not author intent.
    Author fields are optional and appear only with a valid author disposition;
    callers without one retain the frozen gate key set.

    ``acceptance`` adds the local bases that need no critic PASS for the current
    bytes (trusted source, owner Skip, owner edit, and Ouroboros's own check only
    under Cyber Pro, ``acceptance_in_force``); a deterministic preflight failure
    recorded for the current bytes still refuses them. Under Cyber Pro the gate names
    a basis in force instead of its own authority, so grants and dependencies follow it. A skill
    installed from a source that reviews each skill (``acceptance["strict_review"]``)
    keeps blocker findings non-executable even under Advisory, until the owner skips
    the review for these bytes; an Advisory author acceptance is not one of its bases.

    Deterministic hard-gate failures (e.g. skill_preflight) are persisted as
    STATUS_PENDING by `_run_deterministic_preflight`, so they are non-executable
    here outside Cyber without needing per-caller findings — only
    LLM blocker verdicts are overridable by advisory enforcement.

    ``findings`` is optional: a caller that has the persisted findings gets a
    ``preflight_failed`` key in the gate (the typed fact behind the Repair
    affordance, #335) plus its companion ``preflight_failed_stale``. Absence
    of the keys means the caller could not know — they are never fabricated.
    A STALE review's persisted failure no longer describes the current payload
    bytes (the owner may have fixed it by hand), so ``preflight_failed`` is
    True only while the findings are fresh; the recorded-but-stale failure
    surfaces as ``preflight_failed_stale`` instead, and the card offers BOTH
    actions: the cheap Re-review (which reruns the preflight) stays primary,
    with Repair offered based on the last recorded preflight.
    """
    from ouroboros.review_records import validate_author_disposition

    author = validate_author_disposition(author_disposition)
    author_current = bool(current_hash and author and author["subject_hash"] == current_hash)
    reference = author.get("review_reference") if author else None
    reference = reference if isinstance(reference, dict) else {}
    preflight = reference.get("preflight") if isinstance(reference.get("preflight"), dict) else {}
    current_preflight = bool(author_current and preflight.get("content_hash") == current_hash)
    preflight_pass = current_preflight and preflight.get("status") == "pass"
    author_preflight_failed = current_preflight and preflight.get("status") == "failed"
    terminal_basis = (reference.get("surface") == "skill" and reference.get("basis") in {"unavailable", "partial_feedback"}
                         and reference.get("job_id") and reference.get("content_hash") and reference.get("finished_at")
                         and reference.get("status") in {"completed", "succeeded", "failed", "interrupted", "cancelled", "timeout"}
                         and reference.get("review_status") in {"pending", "failed", "interrupted", "cancelled", "timeout"})
    raw_status = normalize_skill_review_status(status)
    if enforcement is None:
        try:
            from ouroboros.config import get_review_enforcement
            enforcement = get_review_enforcement()
        except Exception:
            enforcement = "blocking"
    enforcement = str(enforcement or "blocking").lower()
    cyber = cyber_authority()
    accepted = acceptance_in_force(acceptance, current_hash, cyber=cyber)
    fresh_preflight_failure = bool(findings is not None and not stale and preflight_failed(findings))
    strict_review = bool(isinstance(acceptance, dict) and acceptance.get("strict_review"))
    if cyber and accepted and not fresh_preflight_failure:
        # The basis still says why these bytes run (keys, dependencies and the card read it).
        executable = True
        reason = accepted
        summary = _ACCEPTANCE_SUMMARIES[accepted]
    elif cyber:
        executable = True
        reason = "cyber_authority"
        summary = ("Cyber Pro permits acting on this payload by Ouroboros's judgment; "
                   "review status, staleness and failures remain independent evidence, not a PASS.")
    elif author_preflight_failed:
        executable = False
        reason = "review_pending"
        summary = "Current deterministic preflight failed; author acceptance does not make this payload executable."
    elif accepted and fresh_preflight_failure:
        executable = False
        reason = "review_pending"
        summary = "The deterministic preflight failed for these exact bytes; their acceptance basis does not make them executable."
    elif accepted:
        executable = True
        reason = accepted
        summary = _ACCEPTANCE_SUMMARIES[accepted]
    elif (not strict_review and author_current and author.get("action", "finish") == "finish"
          and author["enforcement"] == "advisory" and enforcement == "advisory"
          and (raw_status != STATUS_PENDING or (terminal_basis and preflight_pass))):
        executable = True
        reason = "author_accepted_advisory"
        summary = "The author accepted the current payload under Advisory; original critic evidence and unavailable-review facts remain unchanged."
    elif raw_status == STATUS_PENDING:
        executable = False
        reason = "review_pending"
        summary = "Review is pending or did not produce an executable verdict."
    elif stale:
        executable = False
        reason = "review_stale"
        summary = "Review is stale for the current skill content; re-run skill_review."
    elif raw_status == STATUS_CLEAN:
        executable = True
        reason = "ready"
        summary = "Review is executable: verdict clean."
    elif raw_status == STATUS_WARNINGS:
        executable = True
        reason = "warnings_do_not_block_execution"
        summary = "Review is executable: warning findings are advisory and do not block execution."
    elif raw_status == STATUS_BLOCKERS:
        if enforcement == "advisory" and strict_review:
            executable = False
            reason = "blocker_findings_need_owner"
            summary = ("Review found blockers in a skill from a source that reviews each skill; it stays off "
                       "until the owner skips the review for these bytes.")
        elif enforcement == "advisory":
            executable = True
            reason = "blockers_allowed_by_advisory_enforcement"
            summary = "Review is executable because advisory enforcement allows blocker findings by operator choice."
        else:
            executable = False
            reason = "blocker_findings_under_blocking_enforcement"
            summary = "Review is blocked: blocker findings must be fixed or review enforcement must be advisory."
    else:
        executable = False
        reason = "review_missing_or_unknown"
        summary = "Review status is missing or unknown; run skill_review."
    return {
        "status": raw_status or STATUS_PENDING,
        "stale": bool(stale),
        **({"author_accepted": reason == "author_accepted_advisory" or (cyber and author_current),
            "author_disposition": author} if author else {}),
        "executable_review": bool(executable),
        "blocking_reason": reason,
        "review_enforcement": enforcement,
        "summary": summary,
        **({
            "preflight_failed": author_preflight_failed or ((not preflight_pass) and (not stale) and preflight_failed(findings)),
            "preflight_failed_stale": bool(stale) and preflight_failed(findings),
        } if findings is not None else {}),
    }


def executable_review_requirement(gate: Dict[str, Any], *, strict_review: bool = False) -> str:
    """What a refused gate still needs: Advisory also admits a qualified author acceptance,
    except for a skill from a source that reviews each skill (``strict_review``, the
    acceptance fact beside the gate), whose blockers need the owner."""
    if strict_review or str(gate.get("blocking_reason") or "") == "blocker_findings_need_owner":
        return "the owner's Skip review for these exact bytes, or a review without blocker findings"
    if str(gate.get("review_enforcement") or "") == "advisory":
        return "a current executable reviewer verdict or a qualified Advisory author acceptance of these exact bytes"
    return "a current executable reviewer verdict"
