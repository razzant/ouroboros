"""Tri-model review for one external skill package.

Uses the Skill Review Checklist and persists verdicts in skill state, siloed
from repo commit obligations. Tool registration lives in ``tools/skill_exec.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ouroboros.config import adaptive_quorum, get_auto_grant_enabled
from ouroboros.reviewer_slot_config import reviewer_slot_config_error
from ouroboros.skill_loader import (
    SkillReviewState,
    auto_grant_if_enabled,
    compute_content_hash,
    save_review_state,
)
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    load_bound_skill,
)
from ouroboros.skill_review_history import (  # noqa: F401  (compat re-exports)
    append_history as _append_skill_review_history,
    count_attempts as _count_attempts_for_content,
    finding_signature as _finding_signature,
    load_history as _load_skill_review_history,
    review_history_path,
)
from ouroboros.skill_review_status import (  # noqa: F401  (compat re-exports)
    CRITICAL_ITEMS,
    STATUS_BLOCKERS,
    STATUS_CLEAN,
    STATUS_PENDING,
    STATUS_WARNINGS,
    WARNINGS_CONVERGENCE_ROUNDS,
    aggregate_skill_review_status,
    count_trailing_warnings_rounds,
)
from ouroboros.tools.review_helpers import (  # noqa: F401  (compat re-exports)
    REVIEW_PROMPT_TOKEN_BUDGET,
    build_anti_thrashing_rules_section,
    build_rebuttal_section,
    build_self_verification_template,
    build_skill_host_context,
    format_obligation_excerpt,
    format_prompt_code_block,
    load_checklist_section,
)
from ouroboros.triad_review import emit_review_model_error_events, extract_json_array, parse_model_review_results  # noqa: F401
from ouroboros.utils import (  # noqa: F401  (compat re-exports)
    append_jsonl,
    atomic_write_json,
    estimate_tokens,
    sanitize_tool_result_for_log,
    utc_now_iso,
)

log = logging.getLogger(__name__)

# The reviewable payload packs, the accepted-rebuttal ledger, the reviewer
# prompt and the reviewer-output rendering live in their own owners below this
# module's seam; they are re-exported here because this module is their
# historical import site, and they must never import it back.
from ouroboros.skill_review_packs import (  # noqa: F401  (compat re-exports)
    _LOADABLE_BINARY_EXTENSIONS,
    _SKILL_PACK_TOKEN_HEADROOM,
    _SkillBinaryPayload,
    _SkillFileOverBudget,
    _SkillFileUnreadable,
    _build_skill_file_packs,
    _read_skill_text,
    _skill_pack_token_budget,
)
from ouroboros.skill_review_rebuttals import (  # noqa: F401  (compat re-exports)
    _accepted_rebuttals_path,
    _build_skill_review_history_section,
    _convergence_hint,
    _fail_items_from_history_entry,
    _load_accepted_rebuttals,
    _persist_rebuttal_flips,
    _record_accepted_rebuttal,
    _render_accepted_rebuttals_section,
    _review_history_path,
)
from ouroboros.skill_review_prompt import (  # noqa: F401  (compat re-exports)
    _CRITICAL_ITEMS,
    _REPO_ROOT,
    _SKILL_CHECKLIST_SECTION,
    _SKILL_REVIEW_ITEMS,
    _build_review_prompt,
    _build_review_prompt_for_attempt,
    _emit_skill_advisory_warning,
    _load_governance_artifact,
    _review_wave_budget_block,
    _run_skill_advisory_pre_review,
)
from ouroboros.skill_review_output import (  # noqa: F401  (compat re-exports)
    _aggregate_status,
    _extract_actor_findings,
    _parse_json_array,
    render_skill_review_block,
)


def _truncate_raw_result(text: str) -> str:
    """Return full raw review text; actor records are the structured SSOT."""
    return str(text or "")

@dataclass
class SkillReviewOutcome:
    """Return payload from ``review_skill``."""

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
    convergence_hint: str = ""
    error: str = ""
    auto_flow: bool = False
    # Bible P3: a single configured reviewer is honored on this executable TRUST
    # gate but the lost cross-model diversity is recorded loudly + DURABLY (in the
    # outcome and review history), so a one-slot skill review can never later look
    # like an ordinary clean/warnings multi-reviewer review.
    single_reviewer_no_diversity: bool = False
    requested_keys: List[str] = field(default_factory=list)
    auto_granted_keys: List[str] = field(default_factory=list)
    requested_permissions: List[str] = field(default_factory=list)
    auto_granted_permissions: List[str] = field(default_factory=list)
    review_profile: str = ""


def _apply_auto_grant_outcome(outcome: SkillReviewOutcome, skill: Any, auto_grant: Any) -> None:
    outcome.requested_keys = list(getattr(auto_grant, "requested_keys", []) or [])
    outcome.auto_granted_keys = list(getattr(auto_grant, "granted_keys", []) or [])
    outcome.requested_permissions = list(getattr(auto_grant, "requested_permissions", []) or [])
    outcome.auto_granted_permissions = list(getattr(auto_grant, "granted_permissions", []) or [])
    if bool(getattr(skill, "is_self_authored", False)) and get_auto_grant_enabled():
        outcome.auto_flow = True

def _is_module_widget_skill(skill: Any) -> bool:
    return (
        skill.manifest.is_extension()
        and isinstance(skill.manifest.ui_tab, dict)
        and str(((skill.manifest.ui_tab or {}).get("render") or {}).get("kind") or "") == "module"
    )


def _run_deterministic_preflight(
    ctx: Any,
    drive_root: pathlib.Path,
    skill: Any,
    content_hash: str,
    *,
    persist: bool,
    binding: ResolvedResourceBinding | None = None,
) -> Optional[SkillReviewOutcome]:
    """Run deterministic checks before spending tri-model review tokens."""
    preflight_raw = ""
    try:
        from ouroboros.tools.skill_preflight import _handle_skill_preflight
        preflight_raw = _handle_skill_preflight(
            ctx, skill=skill.name, _resolved_binding=binding,
        )
        preflight = json.loads(preflight_raw)
    except Exception:
        preflight = {"ok": True}
    if not isinstance(preflight, dict) or preflight.get("ok", True):
        return None
    findings = [{
        "item": "skill_preflight",
        "verdict": "FAIL",
        "severity": "critical",
        "reason": _truncate_raw_result(json.dumps(preflight, ensure_ascii=False)),
        "model": "deterministic_preflight",
    }]
    # A deterministic preflight failure is a structural fact, not an LLM verdict,
    # so it is persisted as PENDING (non-executable under EVERY enforcement mode,
    # in every readiness/execution caller) rather than BLOCKERS (which advisory
    # enforcement would let an operator override). The skill_preflight FAIL finding
    # still records exactly why, just like the other PENDING review-time failures.
    outcome = SkillReviewOutcome(
        skill_name=skill.name,
        status=STATUS_PENDING,
        findings=findings,
        reviewer_models=["deterministic_preflight"],
        content_hash=content_hash,
        error="deterministic skill_preflight failed before LLM review; skill is not executable",
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
        save_review_state(
            drive_root,
            skill.name,
            review_state,
        )
        if not getattr(ctx, "_skill_review_lifecycle_guard", False):
            _append_skill_review_history(
                drive_root,
                skill.name,
                status=outcome.status,
                content_hash=content_hash,
                findings=findings,
            )
        skill.review = review_state
        # Record what the skill requests (transparency in the review block) but
        # NEVER auto-grant a skill that FAILED the deterministic preflight gate
        # (invalid manifest, sensitive-shaped file, binary payload, path escape).
        # The PENDING status already makes it non-executable everywhere; recording
        # the requested keys/permissions just keeps the review block honest about
        # what the skill wanted.
        from ouroboros.skill_loader import (
            requested_core_setting_keys,
            requested_skill_permissions,
        )
        manifest = getattr(skill, "manifest", None)
        outcome.requested_keys = requested_core_setting_keys(
            list(getattr(manifest, "env_from_settings", []) or [])
        )
        outcome.requested_permissions = requested_skill_permissions(
            list(getattr(manifest, "permissions", []) or []),
            list(getattr(manifest, "subscribe_events", []) or []),
        )
    return outcome

def _official_hub_review_profile(skill: Any) -> str:
    """Return official_hub only when local payload matches its Hub sidecar hashes."""
    if str(getattr(skill, "source", "") or "") != "ouroboroshub":
        return ""
    marker = pathlib.Path(skill.skill_dir) / ".ouroboroshub.json"
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict) or str(data.get("source") or "") != "ouroboroshub":
        return ""
    marker_name = str(data.get("sanitized_name") or data.get("slug") or "").strip()
    if marker_name and marker_name != str(getattr(skill, "name", "") or ""):
        return ""
    slug = str(data.get("slug") or marker_name or getattr(skill, "name", "") or "").strip()
    try:
        from ouroboros.marketplace import ouroboroshub

        catalog_summary = ouroboroshub.info(slug)
        catalog_files = {
            str(item.get("path") or ""): str(item.get("sha256") or "").strip().lower()
            for item in (catalog_summary.files or [])
            if isinstance(item, dict)
        }
    except Exception:
        return ""
    files = data.get("files") if isinstance(data.get("files"), list) else []
    if not files:
        return ""
    sidecar_files = {
        str(item.get("path") or ""): str(item.get("sha256") or "").strip().lower()
        for item in files
        if isinstance(item, dict)
    }
    if sidecar_files != catalog_files:
        return ""
    root = pathlib.Path(skill.skill_dir).resolve()
    for item in files:
        if not isinstance(item, dict):
            return ""
        rel = pathlib.PurePosixPath(str(item.get("path") or ""))
        expected = str(item.get("sha256") or "").strip().lower()
        if not rel.parts or rel.is_absolute() or ".." in rel.parts or not expected:
            return ""
        path = (root / pathlib.Path(*rel.parts)).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError:
            return ""
        if not path.is_file():
            return ""
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            return ""
    # Reject any EXTRA local runtime-reachable file not covered by the catalog.
    # Without this, a locally-added file (e.g. evil.py) would still earn the
    # official_hub fast-path. Provenance/control sidecars are install-time
    # artifacts, never catalog entries, so they are excluded from the compare.
    from ouroboros.skill_loader import _iter_payload_files  # pylint: disable=W0212
    from ouroboros.contracts.skill_payload_policy import SKILL_PAYLOAD_CONTROL_FILENAMES

    manifest = getattr(skill, "manifest", None)
    try:
        local_files = _iter_payload_files(
            root,
            manifest_entry=str(getattr(manifest, "entry", "") or ""),
            manifest_scripts=list(getattr(manifest, "scripts", []) or []),
        )
    except Exception:
        return ""
    local_relset = set()
    for path in local_files:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            return ""
        if rel in SKILL_PAYLOAD_CONTROL_FILENAMES:
            continue
        local_relset.add(rel)
    if local_relset != set(catalog_files.keys()):
        return ""
    return "official_hub"


def is_official_hub_payload_verified(skill: Any) -> bool:
    """Return whether a local OuroborosHub payload still matches the live catalog."""
    return _official_hub_review_profile(skill) == "official_hub"

# Public entry point


def _skill_quorum_failure_outcome(
    skill: Any,
    *,
    findings: List[Dict[str, Any]],
    models: List[str],
    content_hash: str,
    required_quorum: int,
    result_json_text: str,
    parsed_review: Any,
    advisory_evidence: Dict[str, Any],
    single_reviewer_no_diversity: bool,
    drive_root: pathlib.Path,
    persist: bool,
    lifecycle_owns_history: bool = False,
) -> SkillReviewOutcome:
    """PENDING outcome for a skill review that missed the adaptive reviewer quorum,
    preserving the single-reviewer degraded marker on the outcome AND the durable
    history (extracted to keep ``review_skill`` under the function-size gate)."""
    outcome = SkillReviewOutcome(
        skill_name=skill.name,
        status=STATUS_PENDING,
        findings=findings,
        reviewer_models=models,
        content_hash=content_hash,
        single_reviewer_no_diversity=single_reviewer_no_diversity,
        error=(
            f"Skill review quorum failure: fewer than {required_quorum} reviewers "
            "returned parseable findings. Raw result preserved."
        ),
        raw_result=_truncate_raw_result(result_json_text),
        raw_actor_records=[record.to_dict() for record in parsed_review.actor_records],
        advisory_result=advisory_evidence,
    )
    if persist and not lifecycle_owns_history:
        _append_skill_review_history(
            drive_root,
            skill.name,
            status=outcome.status,
            content_hash=content_hash,
            findings=findings,
            raw_actor_records=[record.to_dict() for record in parsed_review.actor_records],
            single_reviewer_no_diversity=single_reviewer_no_diversity,
        )
    return outcome


def review_skill(
    ctx: Any,
    skill_name: str,
    *,
    persist: bool = True,
    review_rebuttal: str = "",
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> SkillReviewOutcome:
    """Run tri-model review on one skill, optionally persisting the verdict."""
    from ouroboros.tools.review import _handle_multi_model_review
    from ouroboros.config import get_review_models
    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx, root="skill_payload", operation="review", path=".", skill_name=skill_name,
        )
    except Exception as exc:
        return SkillReviewOutcome(skill_name=skill_name, status=STATUS_PENDING, error=str(exc))
    drive_root = binding.state_drive_root
    skill = load_bound_skill(binding)
    if skill is None:
        return SkillReviewOutcome(skill_name=skill_name, status=STATUS_PENDING, error=f"Skill {skill_name!r} not found")
    if skill.load_error:
        return SkillReviewOutcome(skill_name=skill_name, status=STATUS_PENDING, error=f"Skill manifest could not be parsed: {skill.load_error}")
    from ouroboros.skill_loader import SkillPayloadUnreadable
    try:
        content_hash = compute_content_hash(
            skill.skill_dir,
            manifest_entry=skill.manifest.entry,
            manifest_scripts=skill.manifest.scripts,
        )
    except SkillPayloadUnreadable as exc:
        return SkillReviewOutcome(
            skill_name=skill.name,
            status=STATUS_PENDING,
            error=(
                f"Skill payload {exc.relpath!r} is unreadable "
                f"({type(exc.err).__name__}: {exc.err}). Review refuses "
                "to emit a PASS over a partial hash — fix file "
                "permissions or remove the unreadable file and re-run."
            ),
        )
    manifest_dump = json.dumps(
        {
            "name": skill.manifest.name,
            "description": skill.manifest.description,
            "version": skill.manifest.version,
            "type": skill.manifest.type,
            "runtime": skill.manifest.runtime,
            "timeout_sec": skill.manifest.timeout_sec,
            "permissions": list(skill.manifest.permissions),
            "conflicts": list(skill.manifest.conflicts),
            "env_from_settings": list(skill.manifest.env_from_settings),
            "requires": list(skill.manifest.requires),
            "scripts": list(skill.manifest.scripts),
            "scheduled_tasks": list(getattr(skill.manifest, "scheduled_tasks", []) or []),
            "entry": skill.manifest.entry,
        },
        ensure_ascii=False,
        indent=2,
    )
    history = _load_skill_review_history(
        drive_root,
        skill.name,
        group_id=str(getattr(ctx, "_skill_review_group_id", "") or ""),
    )
    try:
        file_packs = _build_skill_file_packs(
            skill.skill_dir,
            manifest_entry=skill.manifest.entry,
            manifest_scripts=skill.manifest.scripts,
        )
    except _SkillFileOverBudget as exc:
        return SkillReviewOutcome(
            skill_name=skill.name,
            status=STATUS_PENDING,
            content_hash=content_hash,
            error=(
                f"Skill file {exc.relpath!r} alone is ~{exc.tokens} tokens, over the "
                f"{exc.budget}-token reviewer budget. Review refuses to truncate the "
                "executable surface — shrink or split that one file so it fits a "
                "single review pass."
            ),
        )
    except _SkillBinaryPayload as exc:
        return SkillReviewOutcome(
            skill_name=skill.name,
            status=STATUS_PENDING,
            content_hash=content_hash,
            error=(
                f"Skill file {exc.relpath!r} ({exc.size_bytes} bytes) is "
                "binary / non-UTF-8. Review refuses opaque payloads in the "
                "executable skill surface — the subprocess could load them "
                "via ctypes/native addons without reviewer inspection. "
                "Remove the file from the skill or refactor the skill to "
                "store such payloads outside the hashed surface."
            ),
        )
    except _SkillFileUnreadable as exc:
        return SkillReviewOutcome(
            skill_name=skill.name,
            status=STATUS_PENDING,
            content_hash=content_hash,
            error=(
                f"Skill file {exc.relpath!r} is unreadable "
                f"({type(exc.err).__name__}: {exc.err}). Review refuses "
                "to fail open — fix the file permissions or remove the "
                "file before re-running skill_review."
            ),
        )
    preflight_outcome = _run_deterministic_preflight(
        ctx,
        drive_root,
        skill,
        content_hash,
        persist=persist,
        binding=binding,
    )
    if preflight_outcome is not None:
        return preflight_outcome
    if slot_err := reviewer_slot_config_error():  # #116: refuse loudly, never the silent default panel
        return SkillReviewOutcome(
            skill_name=skill.name, status=STATUS_PENDING, content_hash=content_hash,
            error=f"invalid reviewer-slot configuration blocks skill review: {slot_err}")
    models = list(get_review_models())
    if len(file_packs) > 1:
        log.warning(
            "Skill %s exceeds the reviewer token budget; reviewing in %d chunked passes.",
            skill.name,
            len(file_packs),
        )
    # Budget admission for the WHOLE review wave (v6.69.0): a wave that cannot
    # fit the remaining root budget is declined up front (typed event, $0 spent,
    # skill honestly pending) instead of dying mid-wave. Fail-open on unknowns.
    budget_block = _review_wave_budget_block(ctx, skill.name, file_packs, models)
    if budget_block is not None:
        return SkillReviewOutcome(skill_name=skill.name, status=STATUS_PENDING,
                                  reviewer_models=models, content_hash=content_hash,
                                  error=budget_block)
    from ouroboros.skill_review_passes import run_skill_review_passes

    prompt, advisory_evidence, result_json_text, infra_error = run_skill_review_passes(
        ctx,
        drive_root,
        skill,
        evidence={
            "manifest_dump": manifest_dump,
            "content_hash": content_hash,
            "history": history,
            "review_rebuttal": review_rebuttal,
            "required_items": _SKILL_REVIEW_ITEMS,
        },
        file_packs=file_packs,
        models=models,
        build_prompt=_build_review_prompt_for_attempt,
        run_review=_handle_multi_model_review,
    )
    if infra_error:
        log.warning("Skill review infrastructure failure for %s: %s", skill.name, infra_error)
        return SkillReviewOutcome(
            skill_name=skill.name,
            status=STATUS_PENDING,
            reviewer_models=models,
            content_hash=content_hash,
            error=f"infrastructure failure: {sanitize_tool_result_for_log(infra_error)}",
        )

    try:
        result_json = json.loads(result_json_text)
    except json.JSONDecodeError:
        return SkillReviewOutcome(
            skill_name=skill.name,
            status=STATUS_PENDING,
            reviewer_models=models,
            content_hash=content_hash,
            error="review returned non-JSON top-level response",
            raw_result=_truncate_raw_result(result_json_text),
        )

    if "error" in result_json:
        return SkillReviewOutcome(
            skill_name=skill.name,
            status=STATUS_PENDING,
            reviewer_models=models,
            content_hash=content_hash,
            error=f"review service error: {result_json['error']}",
        )

    parsed_review = parse_model_review_results(result_json, required_items=_SKILL_REVIEW_ITEMS)
    emit_review_model_error_events(ctx, parsed_review, source="skill_review", skill_name=skill.name)
    findings, responded_models = parsed_review.findings, parsed_review.responsive_models
    required_quorum = adaptive_quorum(len(models))
    single_reviewer_no_diversity = len(models) < 2
    if single_reviewer_no_diversity:
        # Skill review is an executable TRUST gate; a single configured reviewer
        # is honored but the lost diversity is recorded loudly AND durably (Bible
        # P3) — in the outcome + review history below, not just this log line.
        log.warning("Skill review (trust gate) ran with a single reviewer (single_reviewer_no_diversity).")
    if len(responded_models) < required_quorum:
        return _skill_quorum_failure_outcome(
            skill,
            findings=findings,
            models=models,
            content_hash=content_hash,
            required_quorum=required_quorum,
            result_json_text=result_json_text,
            parsed_review=parsed_review,
            advisory_evidence=advisory_evidence,
            single_reviewer_no_diversity=single_reviewer_no_diversity,
            drive_root=drive_root,
            persist=persist,
            lifecycle_owns_history=bool(getattr(ctx, "_skill_review_lifecycle_guard", False)),
        )

    review_profile = _official_hub_review_profile(skill)
    status = _aggregate_status(
        findings,
        skill_type=skill.manifest.type,
        is_module_widget=_is_module_widget_skill(skill),
        review_profile=review_profile,
    )
    outcome = SkillReviewOutcome(
        skill_name=skill.name,
        status=status,
        findings=findings,
        reviewer_models=responded_models,
        content_hash=content_hash,
        prompt_chars=len(prompt),
        single_reviewer_no_diversity=single_reviewer_no_diversity,
        raw_result=_truncate_raw_result(result_json_text),
        raw_actor_records=[record.to_dict() for record in parsed_review.actor_records],
        advisory_result=advisory_evidence,
        convergence_hint=_convergence_hint(history, findings, current_status=status),
        review_profile=review_profile,
    )

    if persist:
        if getattr(ctx, "_skill_review_lifecycle_guard", False):
            from ouroboros.skill_review_runner import _can_persist_review_outcome

            if not _can_persist_review_outcome(
                drive_root,
                skill.name,
                content_hash,
                expected_job_id=str(getattr(ctx, "_skill_review_lifecycle_job_id", "") or ""),
            ):
                outcome.status = STATUS_PENDING
                outcome.error = (
                    "review outcome was not persisted because the lifecycle job "
                    "is already terminal or no longer matches this content hash"
                )
                return outcome
        save_review_state(
            drive_root,
            skill.name,
            SkillReviewState(
                status=outcome.status,
                content_hash=content_hash,
                findings=findings,
                reviewer_models=responded_models,
                timestamp=utc_now_iso(),
                prompt_chars=outcome.prompt_chars,
                cost_usd=outcome.cost_usd,
                raw_result=outcome.raw_result,
                raw_actor_records=[record.to_dict() for record in parsed_review.actor_records],
                advisory_result=dict(advisory_evidence or {}),
                review_profile=review_profile,
            ),
        )
        if not getattr(ctx, "_skill_review_lifecycle_guard", False):
            _append_skill_review_history(
                drive_root, skill.name,
                status=outcome.status, content_hash=content_hash, findings=findings,
                single_reviewer_no_diversity=single_reviewer_no_diversity,
            )
        _persist_rebuttal_flips(
            drive_root, skill.name,
            history=history, findings=findings,
            review_rebuttal=review_rebuttal, content_hash=content_hash,
            responded_models=list(responded_models),
        )
        skill.review = SkillReviewState(
            status=outcome.status,
            content_hash=content_hash,
            findings=findings,
            reviewer_models=responded_models,
            timestamp=utc_now_iso(),
            prompt_chars=outcome.prompt_chars,
            cost_usd=outcome.cost_usd,
            raw_result=outcome.raw_result,
            raw_actor_records=[record.to_dict() for record in parsed_review.actor_records],
            advisory_result=dict(advisory_evidence or {}),
            review_profile=review_profile,
        )
        auto_grant = auto_grant_if_enabled(drive_root, skill)
        _apply_auto_grant_outcome(outcome, skill, auto_grant)

    return outcome


__all__ = [
    "SkillReviewOutcome",
    "render_skill_review_block",
    "review_skill",
]
