"""Skill review verdict aggregation.

This module is deliberately tiny so both ``skill_review`` and
``skill_loader`` can share the same live status calculation without an
import cycle.
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
})

# Backward-compatible export name for older imports.  New aggregation logic
# distinguishes hard trust-boundary blockers from severity-driven items below.
CRITICAL_ITEMS = HARD_CRITICAL_ITEMS | SEVERITY_DRIVEN_ITEMS


def _severity_blocks(finding: Dict[str, Any]) -> bool:
    severity = str(finding.get("severity") or "").strip().lower()
    if severity in {"advisory", "warning", "warn"}:
        return False
    # Missing/unknown severity stays conservative for legacy persisted findings.
    return True


def aggregate_skill_review_status(
    findings: List[Dict[str, Any]],
    skill_type: str,
    *,
    is_module_widget: bool = False,
    enforcement: Optional[str] = None,
    review_profile: str = "",
) -> str:
    """Collapse per-reviewer findings into an enforcement-independent verdict.

    The ``official_hub`` profile (hash-verified official OuroborosHub payload)
    keeps only hard trust-boundary items (``HARD_CRITICAL_ITEMS``) as blockers
    and downgrades severity-driven hygiene/bug findings to warnings, since the
    payload already passed review at submission and its bytes are verified
    against the live catalog. Deterministic preflight/sensitive/binary/path/
    dependency/grant/enablement gates remain fail-closed outside this function.
    """
    # A deterministic preflight FAIL — and the ABI-1 PluginAPI admission
    # refusal — are structural gate failures, not LLM verdicts. They persist
    # as STATUS_PENDING (non-executable under every enforcement mode) and MUST
    # reload that way — never as advisory-overridable BLOCKERS or WARNINGS —
    # so honor them before the severity-driven aggregation below.
    if preflight_failed(findings):
        return STATUS_PENDING
    is_official_hub = review_profile == "official_hub"
    has_critical_fail = False
    has_warning_fail = False
    is_extension = skill_type == "extension"
    for finding in findings:
        verdict = finding.get("verdict") == "FAIL"
        if not verdict:
            continue
        item = finding.get("item")
        item_is_critical = item in HARD_CRITICAL_ITEMS
        if item in SEVERITY_DRIVEN_ITEMS:
            if is_official_hub:
                item_is_critical = False
            elif item in {"extension_namespace_discipline", "widget_module_safety"} and not is_extension:
                item_is_critical = False
            else:
                item_is_critical = _severity_blocks(finding)
        if item_is_critical:
            has_critical_fail = True
        else:
            has_warning_fail = True
    if has_critical_fail:
        return STATUS_BLOCKERS
    if has_warning_fail:
        return STATUS_WARNINGS
    return STATUS_CLEAN


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
    path (``skill_review_cycles.plugin_api_admission_refusal_outcome``) and the
    RC auditor: only a ``clean``/``warnings`` verdict counts as the hash-bound
    PASS that keeps a plugin_api-less extension loading. Deliberately
    ENFORCEMENT-INDEPENDENT: advisory enforcement can make a BLOCKERS verdict
    executable (``skill_review_gate``), but it never makes one a grandfather —
    the refusal path persists ``pending`` over anything else."""
    return normalize_skill_review_status(status) in (STATUS_CLEAN, STATUS_WARNINGS)


WARNINGS_CONVERGENCE_ROUNDS = 3


def count_trailing_warnings_rounds(
    history: List[Dict[str, Any]],
    *,
    current_status: Optional[str] = None,
) -> int:
    """Count consecutive most-recent rounds whose verdict is advisory-only warnings.

    Structural (status-based), independent of the exact finding signature, so an
    advisory whack-a-mole (rotating advisory FAIL findings that never block) is
    detectable even when the signature changes every round. When ``current_status``
    is provided it is treated as the not-yet-persisted current round: a
    non-warnings current round breaks the streak (returns 0); a warnings current
    round is counted and the tally continues back through ``history``. Legacy
    ``advisory``/``advisory_pass`` statuses normalize to warnings.
    """
    count = 0
    if current_status is not None:
        if normalize_skill_review_status(current_status) != STATUS_WARNINGS:
            return 0
        count += 1
    for entry in reversed(list(history or [])):
        if not isinstance(entry, dict):
            break
        if normalize_skill_review_status(entry.get("status")) == STATUS_WARNINGS:
            count += 1
        else:
            break
    return count


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
        if not isinstance(finding, dict):
            continue
        if finding.get("verdict") == "FAIL" and (
            finding.get("item") in ("skill_preflight", "plugin_api_admission")
            or str(finding.get("model") or "") in ("deterministic_preflight", "plugin_api_admission")
        ):
            return True
    return False


def skill_review_gate(
    status: str, *, stale: bool = False, enforcement: Optional[str] = None,
    findings: Any = None,
) -> Dict[str, Any]:
    """Structured, agent-facing explanation of whether a review is executable.

    Deterministic hard-gate failures (e.g. skill_preflight) are persisted as
    STATUS_PENDING by `_run_deterministic_preflight`, so they are non-executable
    here under every enforcement mode without needing per-caller findings — only
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
    raw_status = normalize_skill_review_status(status)
    if enforcement is None:
        try:
            from ouroboros.config import get_review_enforcement
            enforcement = get_review_enforcement()
        except Exception:
            enforcement = "blocking"
    enforcement = str(enforcement or "blocking").lower()
    if raw_status == STATUS_PENDING:
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
        if enforcement == "advisory":
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
        "executable_review": bool(executable),
        "blocking_reason": reason,
        "review_enforcement": enforcement,
        "summary": summary,
        **({
            "preflight_failed": (not stale) and preflight_failed(findings),
            "preflight_failed_stale": bool(stale) and preflight_failed(findings),
        } if findings is not None else {}),
    }
