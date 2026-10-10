"""Multi-model review and unified pre-commit review gate."""

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.llm import LLMClient  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
from ouroboros.utils import (
    run_cmd,
    append_jsonl,
    estimate_tokens,  # noqa: F401 — patchable seam: fit_triad_prompt resolves it through THIS namespace
    truncate_review_artifact,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    utc_now_iso,
)
from ouroboros import config as _cfg
from ouroboros.owner_words import owner_words_text
from ouroboros.review_substrate import SLOT_ID_PREFIX, TYPED_FAILURE_FACT_KEYS, slot_id_for_row  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.triad_review import (
    REVIEW_JSON_ARRAY_CONTRACT,  # noqa: F401 -- facade import surface (tests and leaves read it here)
    extract_json_array,
    review_query_error_payload as _review_query_error_payload,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
)
from ouroboros.tools.review_response import (
    parse_model_response as _parse_model_response,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    review_operation_fields as _review_operation_fields,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
)

log = logging.getLogger(__name__)


# The window/limit names below stay importable and MONKEYPATCHABLE on this
# module: ``review_admission.fit_triad_prompt`` resolves them through this
# namespace at call time (tests pin that seam).
from ouroboros.reviewer_window import reviewer_context_window, window_scaled_reserves  # noqa: F401
from ouroboros.tools.review_synthesis import quorum_input_token_limit as _quorum_input_token_limit  # noqa: F401
from ouroboros.tools.review_helpers import (
    REPO_ROOT as _REPO_ROOT,
    load_checklist_section as _load_checklist_section_precise,  # noqa: F401 -- retained test/facade patch seam
    load_checklist_layers,
    load_governance_doc,  # noqa: F401 -- retained test/facade patch seam
    build_touched_file_pack,
    triad_pack_exclusions,
    build_goal_section,
    build_scope_section,
    review_drive_root,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    build_rebuttal_section,
    CRITICAL_FINDING_CALIBRATION,  # noqa: F401 -- facade import surface (devtools/measure_review_pack.py reads the packet parts here)
    anti_pattern_lock_guard,  # noqa: F401 -- facade import surface (same)
    review_preamble,  # noqa: F401 -- facade import surface (same)
    build_self_verification_template,
    build_review_history_section as _build_review_history_section,  # noqa: F401 -- retained test/facade patch seam
    calibrated_input_token_limit,  # noqa: F401 — patchable seam (see note above)
    emit_review_usage,  # noqa: F401 -- facade import surface; leaves read it through the call-time handle
    format_name_status_for_preflight,
    format_review_history_entry as _format_review_entry,
    REVIEW_PROMPT_TOKEN_BUDGET,  # noqa: F401 — patchable seam (see note above)
    REVIEW_POOL_EMPTY_REASON,
    REVIEW_POOL_EMPTY_SENTENCE,
    review_enforcement_blocks,
    single_line as _single_line,
)


# Derived alias; ``review_helpers.REPO_ROOT`` remains the repo-root SSOT.
_CHECKLISTS_PATH = _REPO_ROOT / "docs" / "CHECKLISTS.md"


def get_tools():
    return [
        ToolEntry(
            name="task_acceptance_review",
            schema={
                "name": "task_acceptance_review",
                "description": (
                    "Record a task-result claim, checklist and evidence. author_action or a recognized "
                    "agent_disposition stages completion under the current review policy. Otherwise, a "
                    "root in auto/required mode nominates its complete ready result: after this round's "
                    "tool results, the host advances final delivery's review operation; settling that "
                    "review alone does not finish the task. A child gets advisory evidence now from at "
                    "most ONE configured reviewer (reviewer_slot_id selects it); a review-off root "
                    "gets its configured panel."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "Final claim or task result the agent intends to release."},
                        "goal": {"type": "string", "description": "Original task goal."},
                        "evidence": {"type": "object", "description": "Tool trace, artifacts, tests and observations. tool_trajectory_indices selects zero-based records from the host's complete retained trajectory, materialized with corpus-SHA addresses. Bounded/missing records stay partial/unavailable; your prose stays agent-supplied evidence."},
                        "checklist": {"type": "string", "default": "", "description": "Optional acceptance checklist."},
                        "acceptance_subject": {
                            "type": "object",
                            "description": "Main's current subject decision, naming the exact owner_source_sha256 from the latest observation; optionally supply complete effective_criteria or material_tool_indices for changed requirements/evidence.",
                            "properties": {
                                "owner_source_sha256": {"type": "string"},
                                "effective_criteria": {"type": "string"},
                                "material_tool_indices": {"type": "array", "items": {"type": "integer"}},
                            },
                            "required": ["owner_source_sha256"],
                        },
                        "agent_disposition": {
                            "type": "string",
                            "enum": ["accepted", "rejected", "partial", "deferred"],
                            "default": "",
                            "description": "Completion alias: any listed stance stages finish, including before first feedback; permission to finish follows review policy. With rationale after the first host review, Advisory may finish a revised result without another panel. Later tool effects or owner/evidence supersession need a new finish stance; a stop stands as recorded. Never creates reviewer PASS.",
                        },
                        "rationale": {
                            "type": "string",
                            "default": "",
                            "description": "Required reason for explicit finish/stop. For stop, plainly name unfinished work: the owner sees this on the task row. Alone it is evidence, not completion; with author_action it records the act without inventing a stance.",
                        },
                        "author_action": {
                            "type": "string", "enum": ["finish", "stop"],
                            "description": "Finish under current review policy, or stop with unfinished work; include rationale. Stop never authorizes a blocked action. When omitted, a recognized agent_disposition still stages finish.",
                        },
                        "acceptance_retry": {
                            "type": "object",
                            "description": "ONE-USE retry of a disclosed local acceptance-preparation failure: name its incident id and basis — material_change (changed requirements/material evidence), repair_evidence (repaired cause, same material), or owner_retry (explicit owner request). Re-sending grants no further retry; re-nomination, rephrasing and status questions do not qualify.",
                            "properties": {
                                "incident_id": {"type": "string"},
                                "basis": {"type": "string", "enum": ["material_change", "repair_evidence", "owner_retry"]},
                                "rationale": {"type": "string"},
                                "owner_source_sha256": {"type": "string", "description": "Current observed owner source for owner_retry or material_change; Main judges whether it requests substantive retry, not status."},
                                "verification_receipt_index": {"type": "integer", "minimum": 0, "description": "Zero-based existing task verification receipt for repair_evidence or material_change. Host resolves its content identity. Supply this OR owner_source_sha256, and no terminal stance."},
                            },
                            "required": ["incident_id", "basis", "rationale"],
                        },
                        "reviewer_slot_id": {
                            "type": "string",
                            "description": "Child only: one review-pool row's subagent id, required when several exist; host checks membership. Choose for the claim; each row keeps its delivery.",
                        },
                        "late_review": {
                            "type": "object",
                            "description": "Owner action on one frozen delivered historical answer. Get debt_id/source_ref via get_task_result; leave claim/goal empty. Cite NEW host-resolvable owner chat/quiz/mailbox words in the caller conversation or host-bound relayed origin. Main interprets the cross-Project target, action and optional ABSOLUTE original-root USD cap; rationale explains it. Hashes/inherited origin alone grant nothing. amend_cap changes money only, even while paused: no review preparation, Resume or fence clearing. review requests the configured advisory panel after recording any supplied cap; exact delivery and current target/root/caller controls must permit dispatch. Uses the original wallet and one stable paid identity; paid/unknown work is collect-only. A separate supplement preserves the answer/decision. Read frozen sources through the supplied bounded get_task_result selector.",
                            "properties": {
                                "task_id": {"type": "string"},
                                "debt_id": {"type": "string"},
                                "owner_source": {"type": "object", "description": "chat: {kind:'chat',ref:{chat_id,client_message_id,ts,text_sha256}}; quiz: {kind:'quiz',task_id:<calling task>,quiz_id}; mailbox: {kind:'mailbox',task_id:<calling task>,msg_id}."},
                                "action": {"type": "string", "enum": ["review", "amend_cap"], "description": "Name it; omitted means amend_cap when a cap is supplied, else review."},
                                "new_original_root_cap_usd": {"type": "number", "exclusiveMinimum": 0},
                                "rationale": {"type": "string"},
                            },
                            "required": ["task_id", "debt_id", "owner_source", "rationale"],
                        },
                        "obligation_dispositions": {
                            "type": "array",
                            "default": [],
                            "description": "Optional per-obligation dispositions when the host surfaced OPEN OBLIGATIONS (blocking review policy): one entry per obligation id with disposition addressed|rejected|deferred and a short reason.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "disposition": {"type": "string", "enum": ["addressed", "rejected", "deferred"]},
                                    "reason": {"type": "string"},
                                },
                                "required": ["id", "disposition"],
                            },
                        },
                    },
                    "required": ["claim", "goal"],
                },
            },
            handler=_handle_task_acceptance_review,
            timeout_sec=int(_cfg.get_llm_transport_read_timeout_sec() + _cfg.get_finalization_grace_sec()),
        )
    ]


def _handle_task_acceptance_review(
    ctx: ToolContext,
    claim: str = "",
    goal: str = "",
    evidence: Optional[dict] = None,
    checklist: str = "",
    agent_disposition: str = "",
    rationale: str = "",
    obligation_dispositions: Optional[list] = None,
    acceptance_subject: Optional[dict] = None,
    author_action: str = "",
    acceptance_retry: Optional[dict] = None,
    reviewer_slot_id: str = "",
    late_review: Optional[dict] = None,
) -> str:
    if late_review is not None:
        from ouroboros.acceptance_late import owner_historical_acceptance
        from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
        result = owner_historical_acceptance(ctx, late_review)
        return _publish_tool_result(ctx, ToolResult(
            status="error" if result["status"] == "refused" else "ok",
            code="TOOL_ARG_ERROR" if result["status"] == "refused" else "OK",
            text=json.dumps(result, ensure_ascii=False, indent=2)))
    from ouroboros.config import get_task_review_mode
    from ouroboros.review_evidence import (
        build_task_acceptance_evidence,
        task_acceptance_evidence_revision,
    )
    from ouroboros.task_results import resolve_task_lineage

    # Child/off review-only calls build their packet; roots nominate to the host fence.
    # Explicit author actions stage completion before either dispatch path, even if
    # packet assembly is unavailable. Agent evidence never becomes host repo_diff.
    legacy_aliases = []
    if str(agent_disposition or "").strip():
        legacy_aliases.append("agent_disposition")
    if obligation_dispositions:
        legacy_aliases.append("obligation_dispositions")
    if legacy_aliases:
        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "deprecated_task_acceptance_alias",
                "task_id": str(getattr(ctx, "task_id", "") or ""),
                "aliases": legacy_aliases,
                "removal": "next_major",
            })
            log.warning(
                "Deprecated task-acceptance aliases used: %s (removal: next major)",
                ", ".join(legacy_aliases),
            )
        except Exception:
            log.warning(
                "Failed to persist deprecated task-acceptance alias event for %s",
                legacy_aliases,
                exc_info=True,
            )

    # ibl-329e1741d5f9: some providers occasionally emit `evidence` as a
    # JSON-encoded string instead of a nested object (or some other non-dict
    # type). `dict(some_str)` iterates the string char-by-char and raises
    # "dictionary update sequence element #0 has length 1; 2 is required" —
    # crashing the tool entirely instead of degrading. Coerce defensively,
    # matching the `evidence if isinstance(evidence, dict) else {}` guard
    # used everywhere else this field is consumed (e.g. review_dispatch.py,
    # subagents.py, mutation_attribution.py).
    if isinstance(evidence, str):
        try:
            parsed_evidence = json.loads(evidence)
        except (TypeError, ValueError):
            parsed_evidence = None
        evidence = parsed_evidence if isinstance(parsed_evidence, dict) else {"raw_evidence": evidence}
    # A non-dict, non-string payload (list, number, bool) is still the agent's
    # supporting evidence: wrap it the same way the string branch does rather
    # than dropping it, so the reviewer sees what was actually claimed instead
    # of an empty dict. Only a genuinely absent evidence argument yields {}.
    agent_evidence = (
        dict(evidence) if isinstance(evidence, dict)
        else ({} if evidence is None else {"raw_evidence": truncate_review_artifact(repr(evidence), limit=2000)})
    )
    # Bind claim, goal and checklist, not only the supporting references.
    agent_evidence["acceptance_request"] = {
        "claim": str(claim or ""),
        "goal": str(goal or ""),
        "checklist": str(checklist or ""),
    }
    disposition = str(agent_disposition or "").strip().lower()
    if disposition not in {"accepted", "rejected", "partial", "deferred"}:
        disposition = ""
    agent_rationale = " ".join(str(rationale or "").split()).strip()
    if author_action and (author_action not in {"finish", "stop"} or not agent_rationale):
        from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="ERROR: TOOL_ARG_ERROR: author_action requires finish|stop and a rationale."))
    # v6.54.4 obligations layer: normalized per-obligation dispositions ride the
    # same agent_decision envelope (the existing v6.54.0 mechanism, extended to
    # obligation granularity). The host loop applies them to the per-task
    # acceptance_obligations it collected under blocking enforcement.
    normalized_ob: list = []
    for entry in (obligation_dispositions or []):
        if not isinstance(entry, dict):
            continue
        oid = str(entry.get("id") or "").strip()
        odisp = str(entry.get("disposition") or "").strip().lower()
        if not oid or odisp not in {"addressed", "rejected", "deferred"}:
            continue
        normalized_ob.append({
            "id": oid[:40],
            "disposition": odisp,
            "reason": " ".join(str(entry.get("reason") or "").split())[:500],
        })
    agent_decision = {}
    if disposition or agent_rationale or normalized_ob or author_action:
        agent_decision = {
            "disposition": disposition,
            "explicit_finish": bool(disposition or author_action),
            "author_action": author_action or "finish",
            "rationale": agent_rationale[:1000],
            "source": "agent_task_acceptance_review_tool",
        }
        if normalized_ob:
            agent_decision["obligation_dispositions"] = normalized_ob
        agent_evidence["agent_decision"] = agent_decision

    # ONE-USE, source-bound retry of a disclosed local preparation failure. It is
    # a declaration about the HOST's failed assembly, not about the candidate, so
    # it is normalized here and registered before any evidence is built.
    retry_declaration: Dict[str, Any] = {}
    if isinstance(acceptance_retry, dict):
        from ouroboros.acceptance_preparation import RETRY_BASES, resolve_retry_source
        from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

        if disposition or author_action:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text="ERROR: TOOL_ARG_ERROR: acceptance_retry conflicts with a terminal author stance. "
                     "Choose retry, or finish/stop with agent_disposition/author_action, in separate decisions."))

        basis = str(acceptance_retry.get("basis") or "").strip().lower()
        incident_id = str(acceptance_retry.get("incident_id") or "").strip()
        retry_rationale = " ".join(str(acceptance_retry.get("rationale") or "").split())
        if basis not in RETRY_BASES or not incident_id or not retry_rationale:
            from ouroboros.tools.tool_result import ToolResult, _publish_tool_result
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text="ERROR: TOOL_ARG_ERROR: acceptance_retry requires the disclosed incident_id, "
                     "a basis of material_change|repair_evidence|owner_retry, and a rationale."))
        retry_declaration = {"incident_id": incident_id[:120], "basis": basis,
                             "rationale": retry_rationale[:500],
                             **{key: acceptance_retry[key] for key in (
                                 "owner_source_sha256", "verification_receipt_index") if key in acceptance_retry}}
        try:
            retry_declaration["source"] = resolve_retry_source(ctx, retry_declaration)
        except Exception as exc:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text="ERROR: TOOL_ARG_ERROR: retry source unavailable or invalid. "
                     + (str(exc) if isinstance(exc, ValueError) else "Read the current owner selector or verification receipts.")))

    metadata = (
        getattr(ctx, "task_metadata", {})
        if isinstance(getattr(ctx, "task_metadata", {}), dict)
        else {}
    )
    lineage = resolve_task_lineage(
        getattr(ctx, "task_id", ""),
        metadata=metadata,
        root_task_id=getattr(ctx, "root_task_id", None),
        parent_task_id=getattr(ctx, "parent_task_id", None),
        delegation_role=getattr(ctx, "delegation_role", None),
        original_task_id=getattr(ctx, "original_task_id", None),
        timeout_retry_from=getattr(ctx, "timeout_retry_from", None),
    )
    task_id = str(lineage["task_id"])
    is_root_task = bool(lineage["is_root_task"])
    if agent_decision.get("explicit_finish"):
        from ouroboros.tools.control_runtime import stage_completion_request
        return stage_completion_request(ctx, {
            "action": author_action or "finish", "answer": str(claim or ""),
            "rationale": agent_rationale, "acceptance_subject": acceptance_subject,
            "agent_decision": agent_decision,
        }, source="task_acceptance_review")
    if get_task_review_mode() in {"auto", "required"} and is_root_task:
        # The ROOT nomination returns BEFORE any host evidence is built: the
        # host rebuilds host-attested evidence at the authoritative fence, and
        # it cannot reconstruct the agent's claims/references from the capped
        # tool trajectory, so the redacted, bounded agent-supplied section (the
        # same normalization the host builder applies) rides this existing
        # trace record. An informed finish/stop and an explicit retry therefore
        # register whatever state the host's own builder is in (#1223). This is
        # a recorded nomination, never a reviewer verdict.
        from ouroboros.review_evidence_sections import accept_agent_supplied_section

        supplied = accept_agent_supplied_section(agent_evidence)
        deferred = {
            "status": "deferred_to_host_acceptance",
            "authoritative": False,
            # The nomination's own content stamp: the host packet's revision is
            # the host's to compute at the fence.
            "evidence_revision": task_acceptance_evidence_revision({"agent_supplied": supplied}),
            "request": {
                "surface": "task_acceptance",
                "goal": str(goal or ""),
                "subject": str(claim or ""),
                "checklist": str(checklist or ""),
                "task_id": task_id,
            },
            "agent_supplied": supplied,
            "acceptance_subject": acceptance_subject,
        }
        if agent_decision:
            deferred["agent_decision"] = agent_decision
        if retry_declaration:
            deferred["acceptance_retry"] = retry_declaration
        return json.dumps(deferred, ensure_ascii=False, indent=2, default=str)

    # Child-task and `off`-mode acceptance builds the packet here and dispatches
    # its packet rows itself; a builder failure propagates as before.
    evidence = build_task_acceptance_evidence(
        ctx,
        agent_evidence=agent_evidence,
        drive_root=pathlib.Path(ctx.drive_root) if getattr(ctx, "drive_root", None) else None,
        task_id=str(getattr(ctx, "task_id", "") or ""),
    )

    from ouroboros.review_substrate import (
        ReviewRequest,
        build_improvement_capsule,
        dissent_findings,
        run_review_request,
        triad_delivery_slots,
    )

    request = ReviewRequest(
        surface="task_acceptance",
        goal=goal,
        subject=claim,
        evidence=evidence,
        checklist=checklist,
        policy={
            "raw_output_must_be_preserved": True,
            # min_successful_slots is set below from adaptive_quorum(len(slots)) —
            # the SSOT — once the actual reviewer slot count is known.
            "fail_closed_on_errors": True,
            "classify_outcome_tier": True,
            "max_physical_attempts_per_actor": 2,
        },
        task_id=str(getattr(ctx, "task_id", "") or ""), retry_key=f"task_acceptance:{task_acceptance_evidence_revision(evidence)}",
        deadline_at=_owner_deadline_at(ctx),  # the task's own window bounds every row
    )
    # Child-task and `off`-mode acceptance is advisory evidence, never the root
    # verdict or its host acceptance (#1334, reviewer_slot_config R2): the rows
    # follow their configured delivery. A child reviews with at most ONE
    # configured row; an off-mode root's explicit call keeps the panel's
    # configured breadth. A malformed config refuses typed (R3).
    try:
        slots = triad_delivery_slots(role_hint="task acceptance")
    except ValueError as exc:
        return json.dumps({
            "status": "not_dispatched",
            "error": f"invalid review pool configuration blocks task acceptance: {exc}",
        }, ensure_ascii=False)
    if not is_root_task:
        from ouroboros.reviewer_slot_config import child_acceptance_slots

        slots, refusal = child_acceptance_slots(slots, reviewer_slot_id)
        if refusal:
            return json.dumps(refusal, ensure_ascii=False)
    if not slots:
        return json.dumps({"status": "not_dispatched", "reason": "no_review_slots",
                           "detail": "no triad reviewer row is configured; no reviewer was called"},
                          ensure_ascii=False)
    retrieving = [slot for slot in slots if slot.retrieves]
    if retrieving:
        from ouroboros.acceptance_retrieving import acceptance_retrieving_work_order
        from ouroboros.review_substrate import review_repo_dirs_for

        try:
            session_root = str(review_repo_dirs_for(ctx)[1])
        except Exception:
            session_root = ""  # the row keeps its own typed `session_root_missing`
        acceptance_retrieving_work_order(request, retrieving, session_root=session_root,
                                         data_root=pathlib.Path(ctx.drive_root))
    request.policy["min_successful_slots"] = _cfg.adaptive_quorum(len(slots))
    result = run_review_request(request, slots=slots, drive_root=pathlib.Path(ctx.drive_root), usage_ctx=ctx)
    # Agent self-call (auto): lead with the compact improvement capsule (the
    # actionable feedback) and keep the full structured result available for the
    # agent that explicitly asked for detail.
    capsule = build_improvement_capsule(result)
    payload_dict = dict(result.__dict__)
    # Dissent is recorded on the agent-called path too, so the tool-result
    # capture lands acceptance_decision.dissent_noted on EVERY path.
    payload_dict["dissent_noted"] = bool(dissent_findings(result))
    if agent_decision:
        payload_dict["agent_decision"] = agent_decision
    payload = json.dumps(payload_dict, ensure_ascii=False, indent=2, default=str)
    return f"{capsule}\n\n<full_review>\n{payload}\n</full_review>" if capsule else payload


def _owner_deadline_at(ctx: Any) -> str:
    """The task's owner deadline (ISO text) from the tool context, or ''."""
    metadata = getattr(ctx, "task_metadata", None) if ctx is not None else None
    return str(metadata.get("deadline_at") or "") if isinstance(metadata, dict) else ""


# Unified pre-commit review gate.

def _load_checklist_section(layer: str = "body") -> str:
    """Load the change-review checklist for ``layer`` (`review_helpers.
    load_checklist_layers`), fail-closed if missing/malformed.

    For the body layer the standing-disclosure archive rides along: packet-only
    (api) reviewers have no repository tools, so a bare pointer to
    docs/CHECKLISTS_ARCHIVE.md would be unresolvable for them and settled
    owner-accepted narrowings could be re-raised (#447 stage-3 wave). The
    archive is small and binding — the extraction slimmed the live checklist
    FILE, not the reviewer's contract. The core layer (a subject that is not
    the Ouroboros body) carries neither the body items nor the archive: its
    disclosures are about Ouroboros's own surfaces."""
    try:
        section = load_checklist_layers(layer)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise FileNotFoundError(
            f"docs/CHECKLISTS.md not found or malformed: {e}"
        ) from e
    if layer != "body":
        return section
    archive_path = _REPO_ROOT / "docs" / "CHECKLISTS_ARCHIVE.md"
    try:
        archive = archive_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        # Fail-closed like the checklist itself: the archive is the same
        # binding reviewer contract (FROZEN_CONTRACT_PATHS) — silently
        # reviewing without the standing disclosures would let settled
        # owner-accepted narrowings be re-litigated.
        raise FileNotFoundError(
            f"docs/CHECKLISTS_ARCHIVE.md not readable: {e}"
        ) from e
    if archive:
        section = f"{section}\n\n{archive}"
    return section


# The packet prompt's templates live with the brief builders (one brief, two
# parts); the module-level names stay importable and patchable here.
from ouroboros.tools.review_admission import (  # noqa: E402, F401 -- intentional public re-exports
    PACKET_TEMPLATE_DYNAMIC as _REVIEW_PROMPT_TEMPLATE_DYNAMIC,
    PACKET_TEMPLATE_STABLE as _REVIEW_PROMPT_TEMPLATE_STABLE,
)


def _parse_review_json(raw: str) -> Optional[list]:
    """Best-effort extraction of a JSON array from model output."""
    return extract_json_array(raw, normalize=True)


def _git_show_staged(repo_dir, path: str) -> Optional[str]:
    """Return indexed text (None only for absence); propagate failed reads."""
    from ouroboros.commit_admission import read_release_file
    return read_release_file(repo_dir, path, source="index")


def _preflight_check(commit_message: str, staged_files: str,
                     repo_dir) -> Optional[str]:
    """Fast deterministic review preflight for common incomplete staged diffs.

    Only exact mechanical checks live here — ones that compare real staged
    artifacts (version-carrier sync, README changelog row, P9 history limits,
    conftest.py hygiene). Two lexical heuristics were deliberately removed
    (issue #447): the commit-message version-reference guess (its "version"
    substring test matched "conversion") and the ".py under ouroboros/ or
    supervisor/ requires tests/ staged" predicate (it refused comment-only
    diffs and accepted tests/README.md as coverage). Both duties now live in
    the semantic checklist: docs/CHECKLISTS.md Change Review Checklist item 4
    (tests_affected) and Ouroboros Body Layer item 12 (version_bump).
    """
    import string as _string

    # Accept either name-status lines ("A  path") or plain filenames.
    raw_lines = staged_files.strip().splitlines()
    file_status: list[tuple[str, str]] = []  # (status_char, filepath)
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        # Name-status format: "X  path".
        if (len(raw) >= 4
                and raw[0] in _string.ascii_uppercase
                and raw[1:3] == "  "):
            status = raw[0].upper()
            path = raw[3:].strip()
            # Renames display as "R  old -> new".
            if " -> " in path:
                path = path.split(" -> ")[-1].strip()
            file_status.append((status, path))
        else:
            # Plain filenames are treated as modified.
            file_status.append(("M", raw))

    # active_staged excludes deletions for companion-file checks.
    staged_set = {path for _, path in file_status}
    active_staged = {path for status, path in file_status if status != "D"}
    # Added/Copied count as new modules; renames do not.
    new_files = {path for status, path in file_status if status in ("A", "C")}
    from ouroboros.commit_admission import release_metadata_diagnostics, format_release_metadata_preflight
    release_error = format_release_metadata_preflight(release_metadata_diagnostics(
        repo_dir, sorted(active_staged), source="index",
        read_text=lambda path: _git_show_staged(repo_dir, path), deleted=sorted(staged_set - active_staged),
    ))
    if release_error:
        return release_error

    # The version-reference and tests-required lexical heuristics were removed
    # here (false blocks: a "conversion" commit told to bump VERSION; a
    # comment-only .py diff refused for missing tests). See docstring —
    # CHECKLISTS.md items 4/12 own these duties semantically.

    # New logic modules require active ARCHITECTURE.md update.
    new_logic_files = [
        f for f in new_files
        if f.startswith(("ouroboros/", "supervisor/")) and f.endswith(".py")
    ]
    # The Architecture book is the obligation, not one file: a new module is
    # documented in the CHAPTER that owns its subsystem, and demanding an
    # entrypoint edit would only buy a membership-list touch that documents
    # nothing. Any staged source of the book satisfies it.
    from ouroboros.reference_books import BOOK_ENTRYPOINTS, book_entrypoint_for

    architecture_entrypoint = BOOK_ENTRYPOINTS["architecture"]
    documented = any(
        book_entrypoint_for(staged) == architecture_entrypoint for staged in active_staged
    )
    if new_logic_files and not documented:
        return (
            "⚠️ PREFLIGHT_BLOCKED: New files added in ouroboros/ or supervisor/ "
            "but no source of the Architecture book is staged.\n"
            "  New structural additions must be documented in the Architecture book "
            f"(`{architecture_entrypoint}` or a `docs/architecture/` chapter) "
            "(Bible P6: authenticity / architectural mirror).\n"
            f"  New files: {new_logic_files[:5]}\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
        )

    # conftest.py must not contain collectable module-level tests.
    conftest_files = [f for f in active_staged if pathlib.Path(f).name == "conftest.py"]
    if conftest_files:
        import ast as _ast
        for cf in conftest_files:
            try:
                cf_text = _git_show_staged(repo_dir, cf)
                if not cf_text:
                    continue
                tree = _ast.parse(cf_text, filename=cf)
                # Nested helpers inside fixtures are not pytest-collected.
                test_fns = [
                    node.name for node in tree.body
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                ]
                if test_fns:
                    shown = test_fns[:5]
                    omission = f" (⚠️ showing first 5 of {len(test_fns)})" if len(test_fns) > 5 else ""
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: {cf} contains test functions: "
                        f"{shown}{omission}.\n"
                        "  conftest.py is for fixtures/hooks only. Move test_ functions "
                        "to a test_*.py file so pytest can discover them properly.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
            except Exception:
                pass  # Non-fatal: AST parse failure or git error, skip this file

    return None


def _review_entry(
    *,
    severity: str,
    item: str,
    reason: str,
    model: str = "",
    tag: str = "triad",
    verdict: str = "FAIL",
    obligation_id: str = "",
) -> dict:
    entry = {
        "severity": severity,
        "item": item,
        "reason": reason,
        "tag": tag,
        "verdict": verdict,
    }
    if model:
        entry["model"] = model
    if obligation_id:
        entry["obligation_id"] = obligation_id
    return entry


def _append_review_warning(ctx: ToolContext, text: Any) -> None:
    if isinstance(text, dict):
        ctx._review_advisory.append(text)
        return
    warning = _single_line(str(text))
    if warning:
        ctx._review_advisory.append(warning)


def _handle_review_block_or_warning(
    ctx: ToolContext,
    blocking_review: bool,
    blocked_msg: str,
    advisory_prefix: str,
) -> Optional[str]:
    """Apply action authority while preserving the independent review signal."""
    cyber = not review_enforcement_blocks("blocking")
    if blocking_review and not cyber:
        return blocked_msg
    if cyber:
        advisory_prefix = "Cyber Pro: review does not prohibit action; original signal follows. "
    _record_advisory_override(ctx, blocked_msg)
    _append_review_warning(ctx, advisory_prefix + blocked_msg)
    ctx._review_iteration_count = 0
    ctx._review_history = []
    return None


def _record_advisory_override(ctx: ToolContext, blocked_msg: str) -> None:
    """Durable trace of a blocking signal waved through by advisory enforcement.

    Constitutional requirement (BIBLE P3 "Owner-chosen enforcement, loud
    advisory"): every decision blocking enforcement would have stopped must
    leave a durable, owner-visible trace. Persisted to events.jsonl AND to a
    persistent counter file surfaced by the review_status tool.
    """
    reason = str(getattr(ctx, "_last_review_block_reason", "") or "unknown")
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "review_advisory_override",
            "review_enforcement": _cfg.get_review_enforcement(),
            "decision_authority": "cyber_pro" if not review_enforcement_blocks("blocking") else "advisory",
            "block_reason": reason,
            "message_head": str(blocked_msg or "")[:600],
            "task_id": str(getattr(ctx, "task_id", "") or ""),
        })
    except Exception:
        log.debug("Failed to emit review_advisory_override event", exc_info=True)
    try:
        from ouroboros.utils import update_json_locked

        path = ctx.drive_root / "state" / "advisory_overrides.json"

        def _bump(current: dict) -> dict:
            recent = list(current.get("recent") or [])
            recent.append({
                "ts": utc_now_iso(),
                "block_reason": reason,
                "message_head": str(blocked_msg or "")[:300],
            })
            return {
                "count": int(current.get("count") or 0) + 1,
                "recent": recent[-10:],
            }

        update_json_locked(path, _bump)
    except Exception:
        log.warning("Failed to persist advisory override visibility", exc_info=True)


def _collect_review_findings(ctx: ToolContext, model_results: list, row_plan: Optional[dict] = None) -> tuple[list[str], list[str], list[str], list[dict]]:
    """Parse every seat's answer by the parts it was asked (contract A for a
    packet seat, contract B for a seat asked ``coupling``) and sort the FAIL
    items into critical/advisory findings. Returns ``(critical_fails,
    advisory_warns, errored_models, triad_raw_results)``."""
    from ouroboros.triad_review import parse_seat_answers

    plan = row_plan or {}
    slot_ids, parts_vec = list(plan.get("slot_ids") or []), list(plan.get("parts") or [])
    row_parts = {str(slot_ids[i]): tuple(parts_vec[i]) for i in range(min(len(slot_ids), len(parts_vec)))}
    parsed = parse_seat_answers({"results": model_results}, row_parts)
    critical_fails: List[str] = []
    advisory_warns: List[str] = []
    structured_critical: List[dict] = []
    structured_advisory: List[dict] = []
    triad_raw_results = [record.to_dict() for record in parsed.actor_records]
    errored_models = [record.model_id for record in parsed.actor_records if record.status == "error"]

    for record in parsed.actor_records:
        if record.status == "error":
            advisory_warns.append(
                f"[{record.model_id}] Model unavailable this round (transport error). "
                "Full raw response preserved in triad_raw_results (status='error')."
            )
            structured_advisory.append(_review_entry(
                severity="advisory",
                item="review_model_unavailable",
                reason=(
                    f"Model unavailable this round (transport error): {record.model_id}. "
                    "Full raw response preserved in triad_raw_results actor record."
                ),
                model=record.model_id,
            ))
            try:
                append_jsonl(ctx.drive_logs() / "events.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "review_model_error",
                    "model": record.model_id,
                    "error_note": "Full raw response preserved in triad_raw_results.",
                })
            except Exception:
                pass
            continue
        if record.status == "parse_failure":
            advisory_warns.append(
                f"[{record.model_id}] Could not parse structured review output (parse_failure). "
                "Full raw response preserved in triad_raw_results (status='parse_failure')."
            )
            structured_advisory.append(_review_entry(
                severity="advisory",
                item="review_model_parse_failure",
                reason=(
                    f"Could not parse structured review output from {record.model_id}. "
                    "Full raw response preserved in triad_raw_results actor record."
                ),
                model=record.model_id,
            ))
            # A parsed object no part of which was countable falls through to the per-part diagnostics.
        elif record.status != "responded":
            continue
        for part, answer in (record.answers or {}).items():
            if answer.get("status") != "responded":
                error = str(answer.get("error") or "")
                if error:
                    # The seat spoke but this part is not countable: the error is a
                    # typed advisory entry (it reaches the record and the author),
                    # not only a log line.
                    advisory_warns.append(f"[{record.model_id}] Part '{part}' unanswered: {error}")
                    structured_advisory.append(_review_entry(
                        severity="advisory", item=f"review_{part}_unanswered",
                        reason=f"Part '{part}' unanswered: {error}", model=record.model_id))
                # FAIL rows of a matrix the gate could not count stay visible as
                # diagnostics (their severity named in the text; nothing counted).
                for item in answer.get("discarded") or []:
                    reason = (f"not counted ({part} answer unanswered: {error or 'invalid'}); "
                              f"the seat's {str(item.get('severity') or 'advisory')} FAIL said: {item.get('reason', '')}")
                    structured_advisory.append(_review_entry(
                        severity="advisory", item=str(item.get("item", "?")), reason=reason, model=record.model_id))
                    advisory_warns.append(f"[{record.model_id}] {item.get('item', '?')}: {reason}")
                continue
            for item in answer.get("findings") or []:
                if str(item.get("verdict", "")).upper() != "FAIL":
                    continue
                desc = f"[{record.model_id}] {item.get('item', '?')}: {item.get('reason', '')}"
                target = structured_critical if item.get("severity") == "critical" else structured_advisory
                target.append(_review_entry(
                    severity="critical" if target is structured_critical else "advisory",
                    item=str(item.get("item", "?")),
                    reason=str(item.get("reason", "")),
                    model=record.model_id,
                    obligation_id=str(item.get("obligation_id", "") or ""),
                ))
                (critical_fails if target is structured_critical else advisory_warns).append(desc)

    ctx._last_review_critical_findings = structured_critical
    ctx._last_review_advisory_findings = structured_advisory
    # Withheld seats (Q28-A oversize drop) keep their typed $0 records beside
    # the dispatched panel's, so durable evidence names every configured seat.
    ctx._last_triad_raw_results = triad_raw_results + list(
        getattr(ctx, "_triad_withheld_seat_records", []) or [])
    if parsed.degraded_reasons:
        if not hasattr(ctx, "_review_degraded_reasons"):
            ctx._review_degraded_reasons = []
        ctx._review_degraded_reasons.extend(parsed.degraded_reasons)
    return critical_fails, advisory_warns, errored_models, triad_raw_results


def _build_critical_block_message(
    ctx: ToolContext,
    commit_message: str,
    critical_fails: List[str],
    advisory_warns: List[str],
    errored_note: str,
) -> str:
    critical_entries = list(getattr(ctx, "_last_review_critical_findings", []) or critical_fails)
    advisory_entries = list(getattr(ctx, "_last_review_advisory_findings", []) or advisory_warns)
    ctx._review_history.append({
        "attempt": ctx._review_iteration_count,
        "commit_message": commit_message,  # full — no [:200] truncation
        "critical": critical_entries,
        "advisory": advisory_entries,
    })

    iteration_note = f" (attempt {ctx._review_iteration_count})"

    retry_coaching = build_self_verification_template(
        critical_entries,
        attempt_idx=ctx._review_iteration_count,
        tool_name="commit_reviewed",
        context_noun="diff",
    )

    return (
        f"⚠️ REVIEW_BLOCKED{iteration_note}: Critical issues found by reviewers.\n"
        "Commit has NOT been created. Fix the issues and try again. review_rebuttal is\n"
        "legitimate when a finding is factually incorrect, when its evidence does not\n"
        "support the claimed severity, or when the requested remedy is disproportionate —\n"
        "e.g. it would remove or restrict a working capability that the accepted plan\n"
        "did not narrow. Argue for a capability-preserving remedy: change what you can\n"
        "argue for, not what you can override — a rebuttal never overrides owner-chosen\n"
        "enforcement. Repetition alone does not validate a finding: verify its evidence\n"
        "and proportionality again, retaining a justified rebuttal when appropriate.\n\n"
        + "Critical findings:\n"
        + "\n".join(f"  - {_format_review_entry(f, default_severity='critical')}" for f in critical_entries)
        + (
            "\n\nAdvisory warnings:\n"
            + "\n".join(f"  - {_format_review_entry(w)}" for w in advisory_entries)
            if advisory_entries else ""
        )
        + errored_note
        + retry_coaching
    )


def _build_preflight_staged(target_repo: str, fallback: str = "") -> str:
    """Convert git name-status to the compact preflight format."""
    try:
        name_status = run_cmd(
            ["git", "diff", "--cached", "--name-status"], cwd=target_repo
        )
        return format_name_status_for_preflight(name_status, fallback=fallback)
    except Exception:
        return fallback  # check 4 may not fire, but checks 1-3 still work


# The api pack's guaranteed-fit ladder lives with the rest of the pre-dispatch
# assembly machinery; the module-level name stays importable and patchable here.
from ouroboros.tools.review_admission import fit_triad_prompt as _fit_triad_prompt


def _triad_governance_usable_window(api_models: list, api_slots: list) -> int:
    """The usable input window the packet's governance share is taken against.

    Every api row receives the SAME stable prefix, so the share is sized against
    the QUORUM limit the fit ladder already sizes the packet with (one SSOT),
    never the narrowest row — one small slot degrades its own seat instead of
    stripping the whole panel's rules."""
    from ouroboros.reviewer_window import reviewer_window_binding

    usable: dict = {}
    for model, slot in zip(api_models, api_slots):
        window = reviewer_context_window(model, **(binding := reviewer_window_binding(slot)))
        output_reserve, tokenizer_margin = window_scaled_reserves(
            window, output_reserve=_review_output_budget(), tokenizer_margin=50_000, model_id=model, binding=binding)
        usable[slot.slot_id] = max(0, int(window) - int(output_reserve) - int(tokenizer_margin))
    return _quorum_input_token_limit(list(usable), usable) if usable else 0


def _triad_governance_context(ctx: ToolContext, touched_paths: list,
                              checklist_section: str, api_models: list, api_slots: list,
                              *, delivery: str = "packet", governance_root=None,
                              layer: str = "body", subject_root: Optional[pathlib.Path] = None):
    """The triad's shared governance tiers for either delivery class.

    ``governance_root`` is the installed body (a frozen subject names it; the
    rules are always the installed body's). Body layer: ``BIBLE.md`` is inlined
    by every api row's constitutional head and the standing disclosures ride
    the checklist section, so both are declared as already delivered: the
    manifest records them as tier-1 inline without a second copy in the
    prompt. Retrieving rows have no constitutional head, so their task
    receives BIBLE.md inline from this shared builder. Core layer (the subject
    is not the Ouroboros body): neither document is owed, so nothing is
    declared already inline; ``subject_root`` is the reviewed repository whose
    own documents the navigation names."""
    from ouroboros.tools.governance_context import GovernanceContext, governance_context

    if not api_models:
        return GovernanceContext()
    if layer == "body":
        already_inline = (("BIBLE.md", "docs/CHECKLISTS_ARCHIVE.md") if delivery == "packet"
                          else ("docs/CHECKLISTS_ARCHIVE.md",))
    else:
        already_inline = ()
    return governance_context(
        pathlib.Path(governance_root or ctx.repo_dir),
        surface="triad",
        touched_paths=touched_paths,
        usable_window_tokens=_triad_governance_usable_window(api_models, api_slots),
        delivery=delivery,
        checklist_section_text=checklist_section,
        already_inline=already_inline,
        layer=layer,
        subject_root=subject_root,
    )


def _capture_triad_staged_diff(
    ctx: ToolContext, target_repo, blocking_review: bool, frozen: Any = None,
) -> tuple[Optional[str], Optional[Any], Optional[str]]:
    """Capture the triad's review-diff evidence, or route a capture failure.

    Returns ``(diff_text, subject, None)`` on success — ``subject`` is the
    managed resolution-delta artifact, ``None`` for an ordinary commit whose
    evidence stays the byte-exact hardened staged diff — and
    ``(None, None, block_result)`` on failure: the fail-closed message in
    blocking mode, ``None`` (advisory skip) otherwise. A genuine failure fails
    closed rather than reviewing a placeholder that would yield authoritative
    findings about a diff nobody has. A ``frozen`` subject that is not the
    system repo's own index IS the evidence (its diff text and managed artifact);
    the system index keeps the gate's live capture, byte-identical to today.
    """
    from ouroboros.tools.review_binary_context import (
        StagedDiffUnavailable, capture_staged_diff)
    from ouroboros.tools.review_subject import managed_review_subject

    if frozen is not None and not frozen.is_system_index:
        return frozen.diff_text, frozen.managed, None
    try:
        subject = managed_review_subject(ctx, target_repo)
        if subject is not None:
            return subject.render_prompt_diff(), subject, None
        return capture_staged_diff(target_repo), None, None
    except StagedDiffUnavailable as exc:
        ctx._last_review_block_reason = "infra_failure"
        return None, None, _handle_review_block_or_warning(
            ctx, blocking_review,
            "⚠️ REVIEW_BLOCKED: Cannot capture the staged diff — commit cannot "
            f"proceed.\nError: {exc}\n"
            "Ensure git is available and the repository is in a valid state.",
            "Review enforcement=Advisory: staged diff capture failed; triad "
            "review skipped rather than run against a placeholder. ",
        )


def _subject_changed_paths(frozen: Any, target_repo) -> tuple[str, str]:
    """``(changed, preflight_staged)`` of the reviewed subject: the gate asks the
    staged index of the reading root; a frozen worktree/base..head subject has
    no staged index to ask and carries its own frozen path set."""
    if frozen is not None and not frozen.is_system_index:
        changed = "\n".join(path for _status, path in frozen.name_status)
        return changed, format_name_status_for_preflight(
            "\n".join(f"{status}\t{path}" for status, path in frozen.name_status), fallback=changed)
    try:
        changed = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=target_repo)
    except Exception:
        changed = ""
    return changed, _build_preflight_staged(target_repo, fallback=changed)


def _review_history_with_open_obligations(ctx: ToolContext, frozen: Any) -> str:
    """The prior-rounds section with the subject root's durable open obligations
    (``review_helpers.review_history_with_obligations``, the one owner the public
    brief builder shares)."""
    from ouroboros.tools.review_helpers import review_history_with_obligations

    return review_history_with_obligations(
        ctx._review_history, drive_root=getattr(ctx, "drive_root", None),
        repo_root=frozen.spec.root if frozen is not None else getattr(ctx, "repo_dir", None))


def _gate_governance_root(ctx: ToolContext) -> pathlib.Path:
    """The installed body governs the gate's subject: a project-aware context
    (serving/system repo, or a workspace) names it apart from the repository it
    commits — ``review_substrate.review_repo_dirs_for``'s first root — and an
    ambiguous workspace is refused there (a fail-closed assembly block); a plain
    context governs itself."""
    from ouroboros.review_substrate import review_repo_dirs_for

    project_aware = getattr(ctx, "workspace_root", None) is not None or bool(
        getattr(ctx, "serving_repo_dir", None) or getattr(ctx, "system_repo_dir", None))
    return review_repo_dirs_for(ctx)[0] if project_aware else pathlib.Path(ctx.repo_dir)


def _prepare_unified_review(ctx: ToolContext, commit_message: str,
                            review_rebuttal: str = "",
                            repo_dir=None,
                            goal: str = "",
                            scope: str = "",
                            subject: Any = None) -> tuple:
    """Assemble the triad packet WITHOUT dispatching any reviewer (Q25=A).

    Returns ``(prepared, early_result, exited)``: ``exited=True`` means the
    triad terminated during assembly and ``early_result`` (a block message, or
    ``None`` for an advisory skip / empty diff) is its final answer — nothing
    may be dispatched for it; otherwise ``prepared`` carries everything
    ``_dispatch_unified_review`` needs. A frozen ``subject`` (``FrozenSubject``)
    is read from ITS root (or isolated checkout) under the installed body's
    governance, the body's release preflight only on the body's layer; ``None``
    is the gate's path unchanged."""
    frozen = subject
    layer = str(frozen.spec.layer or "body") if frozen is not None else "body"
    target_repo = frozen.review_root if frozen is not None else (repo_dir or ctx.repo_dir)
    governance_root = pathlib.Path(frozen.spec.governance_root) if frozen is not None else _gate_governance_root(ctx)
    # The core layer indexes the SUBJECT's own documents where the reviewers read
    # them (the isolated checkout of a base..head subject); the body layer's
    # navigation is the body's own and names no subject root.
    subject_root = pathlib.Path(target_repo) if layer != "body" else None
    ctx._review_iteration_count += 1
    ctx._last_review_block_reason = ""  # reset per attempt
    ctx._last_triad_models = []  # reset forensic field so stale values never persist on early exit
    ctx._last_review_critical_findings = []  # reset to avoid stale findings from previous attempts
    ctx._last_triad_raw_results = []  # reset per-model actor records
    ctx._triad_withheld_seat_records = []  # reset Q28-dropped seat records
    ctx._review_degraded_reasons = []  # reset degraded participation markers
    review_enforcement = _cfg.get_review_enforcement()
    blocking_review = review_enforcement_blocks(review_enforcement)

    diff_text, subject, capture_block = _capture_triad_staged_diff(ctx, target_repo, blocking_review, frozen=frozen)
    if diff_text is None:  # capture failed: block (blocking) or advisory-skip (None)
        return None, capture_block, True
    if not diff_text.strip():
        return None, None, True

    changed, preflight_staged = _subject_changed_paths(frozen, target_repo)
    # Reviewers of a managed resolution read the RESOLUTION path set (delta ∪
    # conflict anchors); the preflight staged list stays on the FULL candidate
    # (I2 — full-tree invariants are never narrowed). The release preflight
    # (version carriers, Architecture rows) is the installed body's own contract:
    # a foreign root under the body's rules (layer ``core``) does not carry it.
    review_changed = "\n".join(subject.touched_paths()) if subject is not None else changed
    preflight_err = _preflight_check(commit_message, preflight_staged, target_repo) if layer == "body" else ""
    if preflight_err:
        from ouroboros.commit_admission import preflight_evidence_unavailable
        ctx._last_review_block_reason = (
            "infra_failure" if preflight_evidence_unavailable(preflight_err) else "preflight"
        )
        result = _handle_review_block_or_warning(
            ctx, blocking_review, preflight_err,
            "Review enforcement=Advisory: preflight warning did not block commit. ",
        )
        if result is not None:
            return None, result, True

    rebuttal_section = build_rebuttal_section(review_rebuttal)

    try:
        checklist_section = _load_checklist_section(layer)
    except (FileNotFoundError, ValueError) as e:
        log.error("Checklist loading failed (fail-closed): %s", e)
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Cannot load review checklist — commit cannot proceed.\n"
            f"Error: {e}\n"
            "Ensure docs/CHECKLISTS.md exists and contains the expected section headers."
        )
        return None, _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review checklist failed to load; commit proceeding anyway. ",
        ), True

    review_history_section = _review_history_with_open_obligations(ctx, frozen)

    touched_paths = [f.strip() for f in review_changed.strip().splitlines() if f.strip()]

    # Per-row identity/delivery/strength from the ONE review-pool builder
    # (``review_pool_slots``): the catalog's marked rows, which a never-configured
    # install owns as the factory rows minted at the settings read seam; there is
    # no default panel here. A malformed configuration is an infra failure, never
    # a silent api spend. Resolved BEFORE the packet's governance and file
    # evidence: only the api rows receive a packet at all, and their windows size
    # its governance share.
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.reviewer_slot_config import commit_triad_delivery, row_plan_retrieves
    from ouroboros.tools.review_admission import (
        assemble_packet_prompt, counted_retrieving_seats, prepare_retrieving_seats, seat_vectors)
    try:
        row_plan = seat_vectors(commit_triad_delivery())
    except ValueError as exc:
        ctx._last_review_block_reason = "infra_failure"
        return None, _handle_review_block_or_warning(
            ctx, blocking_review,
            f"⚠️ REVIEW_BLOCKED: invalid review pool configuration — {exc}",
            "Review enforcement=Advisory: invalid review pool configuration did not block commit. ",
        ), True
    models, row_routes = row_plan["models"], row_plan["routes"]
    if not models:
        # A configured fact (the ``## Review`` block's ``pool_empty``), stated as such
        # before anything is assembled: nothing to dispatch, no provider to blame.
        ctx._last_review_block_reason = REVIEW_POOL_EMPTY_REASON
        return None, _handle_review_block_or_warning(
            ctx, blocking_review,
            f"⚠️ REVIEW_BLOCKED: review NOT_PERFORMED — {REVIEW_POOL_EMPTY_SENTENCE}",
            "Review enforcement=Advisory: the review pool is empty, so no review ran; "
            "an explicit author decision is required. ",
        ), True
    ctx._last_triad_models = list(models)  # forensic: actual resolved model IDs
    # Packet rows only: a retrieving api row (native delivery or a configured
    # subagent) neither constrains the fit ladder nor counts as an api seat for
    # the Q28-A yield arithmetic below.
    api_indices = [i for i in range(len(models)) if not row_plan_retrieves(row_plan, i)]
    api_models = [models[i] for i in api_indices]
    from ouroboros.review_records import ReviewSlot
    api_slots = [ReviewSlot(slot_id=row_plan["slot_ids"][i], model=models[i],
                           session_profile=row_plan["session_profiles"][i], use_local=row_plan["use_local"][i])
                 for i in api_indices]

    # Which governance documents this packet carries in full, and which arrive as
    # navigation: ONE decision for every review surface (governance_context).
    # BIBLE.md rides the constitutional head of every api row and the standing
    # disclosures ride the checklist section, so both are declared as already
    # delivered rather than sent twice. An all-retrieving panel assembles no
    # packet, so it asks for none.
    governance = _triad_governance_context(
        ctx, touched_paths, checklist_section, api_models, api_slots,
        governance_root=governance_root, layer=layer, subject_root=subject_root)

    # Build touched-file pack for full current context (managed: the reviewed
    # resolution set; binary rows carry the M0 baseline identity). A plain
    # commit withholds the two disclosed pack-exclusion classes (span-only
    # release carriers, prefix-duplicated governance docs); a managed subject
    # keeps every full text — its reviewed delta is M0→staged, not HEAD→staged.
    try:
        exclude_paths, exclusion_note = (set(), "") if subject is not None else triad_pack_exclusions(
            pathlib.Path(target_repo), touched_paths,
            prefix_texts=dict(governance.inline_whole_documents),
        )
        current_files_section, _omitted = build_touched_file_pack(
            pathlib.Path(target_repo),
            touched_paths,
            represent_binary=subject is not None,
            m0_tree=getattr(subject, "m0_tree", "") or "",
            staged_tree=getattr(subject, "staged_tree", "") or "",
            exclude_paths=exclude_paths,
        )
        if _omitted:
            current_files_section += (
                f"\n\n⚠️ OMISSION NOTE: {len(_omitted)} file(s) omitted from direct context: "
                f"{', '.join(_omitted)}"
            )
        if exclusion_note:
            current_files_section += f"\n\n{exclusion_note}"
        if not current_files_section.strip():
            current_files_section = "(no touched files could be read)"
    except Exception as e:
        log.warning("Failed to build touched file pack for triad review: %s", e)
        current_files_section = f"(touched file pack unavailable: {e})"

    from ouroboros.review_evidence import commit_review_evidence_section, materialize_commit_review_session_view

    task_evidence = dict(getattr(ctx, "_commit_review_evidence", None) or {})
    if any(route is ReviewRouteKind.AGENT_SESSION for route in row_routes):
        task_evidence = materialize_commit_review_session_view(task_evidence, target_repo)
        ctx._commit_review_evidence = task_evidence
    task_evidence_compact = False
    owner_words = owner_words_text(ctx)
    goal_section = build_goal_section(goal, scope, commit_message, owner_words)
    scope_section = build_scope_section(scope)

    def _assemble_prompt(files_section: str, staged_diff: str) -> tuple:
        """Return (prompt, stable_prefix_len) — the packet seat's Part-1 prompt."""
        return assemble_packet_prompt(
            layer=layer, checklist_section=checklist_section, governance=governance,
            goal_section=goal_section, scope_section=scope_section, files_section=files_section,
            diff_text=staged_diff, changed_files=review_changed, rebuttal_section=rebuttal_section,
            review_history_section=review_history_section,
            task_evidence_section=commit_review_evidence_section(task_evidence, delivery="packet", compact=task_evidence_compact),
        )

    def _compact_task_evidence():
        nonlocal task_evidence_compact
        task_evidence_compact = True
    if task_evidence:
        _assemble_prompt.compact_optional_evidence = _compact_task_evidence

    # P3 stays one-pass. The api pack, its fit ladder and the fixed_overflow
    # gate exist ONLY for the api rows (5.2/5.7): a retrieving row reads with
    # its own tools, so it neither constrains the fit limit nor is blocked by
    # it, and a panel with no api rows skips pack assembly entirely.
    prompt, stable_prefix_len = "", 0
    if api_models:
        prompt, stable_prefix_len, fit_error = _fit_triad_prompt(
            api_models, _assemble_prompt, current_files_section, diff_text,
            review_changed, target_repo, ctx=ctx, subject=subject,
            slots=api_slots,  # a frozen non-index subject re-renders ITS pinned trees at -U0
            compact_diff=(lambda: frozen.render_prompt_diff(0)) if frozen is not None and not frozen.is_system_index else None,
        )
        for i, slot in zip(api_indices, api_slots):
            models[i], row_plan["session_profiles"][i], row_plan["use_local"][i] = slot.model, slot.session_profile, slot.use_local
        ctx._last_triad_models = list(models)
        if fit_error:
            session_count, required = counted_retrieving_seats(row_plan, api_indices)
            if session_count >= required:
                # Q28-A: packet limits gate only the api subset. Enough COUNTED
                # retrieving rows remain for the counted quorum (an added critic
                # is heard, never a vote), so the api rows are DROPPED (recorded
                # loudly, never silent) and the panel proceeds on retrieving delivery.
                from ouroboros.tools.review_admission import (
                    drop_api_rows,
                    triad_not_dispatched_records,
                )
                # Seat identity survives the drop: each yielded api seat leaves
                # a typed $0 not_dispatched actor record, merged into the
                # durable raw results after the dispatched panel reports.
                ctx._triad_withheld_seat_records = triad_not_dispatched_records(
                    row_plan,
                    "Q28-A oversize drop: this api seat could not receive the "
                    "irreducible packet; the panel's retrieving rows "
                    "satisfied the quorum without it ($0 spent).", only_api=True)
                row_plan = drop_api_rows(row_plan)
                models, row_routes = row_plan["models"], row_plan["routes"]
                ctx._last_triad_models = list(models)
                note = (
                    f"triad_api_rows_dropped_oversize_pack: {len(api_models)} api row(s) "
                    f"({', '.join(api_models)}) could not receive the irreducible packet; "
                    f"{session_count} retrieving row(s) satisfy the quorum and proceed"
                )
                ctx._review_degraded_reasons.append(note)
                log.warning("%s", note)
                api_models, prompt, stable_prefix_len = [], "", 0
            else:
                # Typed ZERO-SPEND terminal (Q28-A): quorum is unreachable
                # without the api rows, so nothing is dispatched at all. The
                # managed wording (split impossible + settings guidance) is
                # already IN the fit terminal — fit_triad_prompt replaces the
                # split clause for a managed subject, never appends below it.
                ctx._last_review_block_reason = "fixed_overflow"
                return None, fit_error, True

    # Every retrieving seat receives ITS OWN two-part brief; one seat's missing
    # context is the whole wave's typed assembly failure (one wave, $0 spent).
    row_plan, retrieving_manifests, brief_texts, brief_failure = prepare_retrieving_seats(
        ctx, row_plan, models, row_routes, target_repo=target_repo, governance_root=governance_root,
        subject=subject, frozen=frozen, diff_text=diff_text, layer=layer, checklist_section=checklist_section,
        commit_message=commit_message, goal=goal, scope=scope, review_rebuttal=review_rebuttal,
        owner_words=owner_words, task_evidence=task_evidence)
    if brief_failure is not None:
        failed_slot, exc = brief_failure
        ctx._last_review_block_reason = "infra_failure"
        return None, _handle_review_block_or_warning(
            ctx, blocking_review,
            "⚠️ REVIEW_BLOCKED: Failed to build the review brief — commit blocked.\n"
            f"Seat {failed_slot}: {exc}\n"
            "Ensure git is available and the repository is in a valid state.",
            "Review enforcement=Advisory: review brief assembly failed; an explicit author decision is required. ",
        ), True

    # The governance manifest is the packet's disclosure record: which rules were
    # inlined, which arrived as navigation and why (BIBLE P1). It rides the
    # prepared packet so the durable prompt record and the api rows' actor
    # records can carry it.
    ctx._last_triad_governance_manifest = list(governance.manifest)
    return {
        "prompt": prompt, "stable_prefix_len": stable_prefix_len,
        "models": models, "routes": row_routes, "row_plan": row_plan,
        "session_task": "", "target_repo": target_repo,
        "blocking_review": blocking_review, "task_evidence": task_evidence,
        "layer": layer,
        "governance_manifest": list(governance.manifest),
        "governance_packet_slots": [slot.slot_id for slot in api_slots],
        "retrieving_manifests": retrieving_manifests,
        "brief_texts": brief_texts,
    }, None, False


def _review_actor_label(row: dict) -> str:
    return str(row.get("model_id") or row.get("slot_id") or row.get("slot") or "reviewer")


def _uncounted_part_seat_lines(rows: list, part: str) -> List[str]:
    """One line per ledger seat row asked ``part``: how it left the question
    uncounted (the answer's own ``error``, else the seat's status) and the FAIL
    rows of a matrix the gate could not count (``discarded``, never counted)."""
    lines: List[str] = []
    for seat in rows:
        if part not in (seat.get("parts") or []):
            continue
        answer = dict((seat.get("answers") or {}).get(part) or {})
        model = str((seat.get("requested") or {}).get("model") or seat.get("observed_model") or "?")
        label = f"{seat.get('seat_id') or '?'} ({model})" + (" [additional, not counted]" if seat.get("additional") else "")
        if str(answer.get("status") or "") == "responded":
            lines.append(f"{label}: answered {str(answer.get('verdict') or '?')}")
            continue
        detail = str(answer.get("error") or "") or f"no answer (seat status: {seat.get('status') or 'unknown'})"
        discarded = [f"{i.get('item') or '?'} ({str(i.get('severity') or 'advisory')})"
                     for i in (answer.get("discarded") or []) if isinstance(i, dict)]
        if discarded:
            detail += f"; FAIL rows not counted: {', '.join(discarded)}"
        lines.append(f"{label}: {detail}")
    return lines


def _dispatch_unified_review(ctx: ToolContext, commit_message: str, prepared: dict) -> Optional[str]:
    """Dispatch the one wave and post-process the panel verdict in the §1.7
    order: NOT_DISPATCHED/pending → QUORUM_FAILED → NOT_PERFORMED (coupling)
    → FAIL → PASS, through ``review_ledger.reduce_verdict`` — the same function
    the durable record reduces with."""
    from ouroboros.review_ledger import coupling_outcome, reduce_verdict, rows_from_plan

    blocking_review = prepared["blocking_review"] and review_enforcement_blocks("blocking")
    ctx._last_review_verdict = {}
    ctx._last_coupling_result = None
    try:
        result_json = _handle_multi_model_review(
            ctx,
            content=TRIAD_USER_TURN,
            prompt=prepared["prompt"],
            models=prepared["models"],
            stable_prefix_len=prepared["stable_prefix_len"],
            routes=prepared["routes"],
            session_task=prepared.get("session_task") or "",
            session_root=str(prepared["target_repo"]),
            row_plan=prepared["row_plan"],
            retry_key=str(prepared.get("retry_key") or ""),
            task_evidence=prepared.get("task_evidence"),
            layer=str(prepared.get("layer") or "body"),
        )
        result = json.loads(result_json)
    except Exception as e:
        log.error("Unified review infrastructure failure: %s", e)
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review infrastructure failed — commit cannot proceed "
            "without a successful review.\n"
            f"Error: {e}\n"
            "Check OPENROUTER_API_KEY, network connectivity, and retry."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review infrastructure failed; an explicit author decision is required. ",
        )

    if "error" in result:
        log.error("Review returned error: %s", result["error"])
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review service returned an error — commit cannot proceed "
            "without a successful review.\n"
            f"Error: {result['error']}\n"
            "Check OPENROUTER_API_KEY, network connectivity, and retry."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review service failed; an explicit author decision is required. ",
        )

    model_results = result.get("results", [])
    if not model_results:
        ctx._last_review_block_reason = "infra_failure"
        if getattr(ctx, "_triad_withheld_seat_records", None):
            ctx._last_triad_raw_results = list(ctx._triad_withheld_seat_records)
        blocked_msg = ("⚠️ REVIEW_BLOCKED: Review returned no results from any "
                       "model — commit cannot proceed without a successful review.")
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: no model results were received; an explicit author decision is required. ")

    critical_fails, advisory_warns, errored_models, _triad_raw = _collect_review_findings(
        ctx, model_results, prepared.get("row_plan"))
    models_total = len(model_results)
    triad_raw = getattr(ctx, "_last_triad_raw_results", []) or []
    # Every delivery records which rules were actually inlined for that row.
    _governance_manifest = list(prepared.get("governance_manifest") or [])
    _packet_slots = set(prepared.get("governance_packet_slots") or [])
    _retrieving = {str(m.get("slot_id") or ""): m for m in prepared.get("retrieving_manifests") or []}
    for record in triad_raw:
        slot_id = str(record.get("slot_id") or "")
        if _governance_manifest and slot_id in _packet_slots:
            record["governance_manifest"] = _governance_manifest
        elif slot_id in _retrieving:
            record["governance_manifest"] = list(_retrieving[slot_id].get("governance_manifest") or [])
            record["brief_sha"] = (_retrieving[slot_id].get("sha") or {}).get("brief", "")
    pending_models = [_review_actor_label(r) for r in triad_raw if (
        r.get("late_result_pending") or str(r.get("operation_state") or "")
        in {"in_flight", "custody_lost"})]
    rows = rows_from_plan(prepared.get("row_plan") or {}, prepared.get("routes") or [], triad_raw)
    # The decision is over the ASSIGNED seats; a seat the author added beside the
    # pool is heard (its Part-2 findings below) but never counted in the quorum.
    verdict = reduce_verdict([seat for seat in rows if not seat.get("additional")], pending=bool(pending_models))
    ctx._last_review_verdict = verdict
    ctx._last_coupling_result = coupling_outcome(verdict, rows)
    # ``blocked`` is the gate's fact, not the verdict's: under the owner's
    # advisory enforcement a FAIL on the coupling question is recorded and
    # surfaced, and blocks nothing.
    ctx._last_coupling_result.blocked = ctx._last_coupling_result.blocked and bool(blocking_review)
    if pending_models:
        ctx._last_review_block_reason = "review_late_result_pending"
        blocked_msg = ("⚠️ REVIEW_PENDING: Physical review operation(s) remain unresolved: "
                       f"{', '.join(pending_models)}. Retry the same commit to reconcile them without a blind paid resend.")
        pending_block = _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review is pending; collect its outcome before choosing an author continuation. ",
        )
        if pending_block is not None:
            return pending_block
    failed_actors = [_review_actor_label(r) for r in triad_raw
                     if r.get("status") not in ("responded", "not_dispatched")]
    quorum = verdict["quorum"]
    if verdict["aggregate"] in ("QUORUM_FAILED", "NOT_DISPATCHED"):  # a wave that sent nothing ($0) has no quorum
        ctx._last_review_block_reason = "review_quorum"
        unavailable_str = ", ".join(failed_actors or errored_models or (sorted(  # each $0 refusal names its cause
            {str(r.get("raw_text") or "") for r in triad_raw}) if verdict["aggregate"] == "NOT_DISPATCHED" else []))
        blocked_msg = (
            f"⚠️ REVIEW_BLOCKED: Only {quorum['responded']} of {quorum['assigned']} review "
            f"models responded successfully (minimum {quorum['required']} required). "
            f"Unavailable/failed: {unavailable_str}.\n"
            "Retry the commit — transient model failures usually resolve quickly."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review quorum was not met; an explicit author decision is required. ",
        )

    if models_total < 2:
        # A single configured reviewer is honored (owner's explicit setup), but
        # the lost cross-model diversity is recorded LOUDLY (Bible P3): the immune
        # gate ran with no second opinion. Record it on the DURABLE degraded-reasons
        # channel (persisted into the commit review record by git_ops) so it
        # survives in review history/status, not just a transient log line.
        ctx._single_reviewer_no_diversity = True
        if not hasattr(ctx, "_review_degraded_reasons"):
            ctx._review_degraded_reasons = []
        if "single_reviewer_no_diversity" not in ctx._review_degraded_reasons:
            ctx._review_degraded_reasons.append("single_reviewer_no_diversity")
        log.warning("Commit review ran with a single reviewer (single_reviewer_no_diversity).")

    errored_note = ""
    all_non_responded = failed_actors or errored_models
    if all_non_responded:
        errored_note = (
            f"\n\nNote: {len(all_non_responded)} of {models_total} review models "
            f"were unavailable or failed to parse ({', '.join(all_non_responded)}). "
            f"Target is {models_total} working reviewers."
        )

    if verdict["aggregate"] == "NOT_PERFORMED":
        # Quorum stands but the gate has no answer to count; the sentence names the
        # branch that decided (``verdict["reason"]``), never a guessed one, and
        # every seat asked Part 2 says how it left the question uncounted.
        reason = str(verdict["reason"] or "review_not_performed")
        ctx._last_review_block_reason = reason
        asked = [str(s.get("seat_id") or "") for s in rows if "coupling" in (s.get("parts") or [])]
        from ouroboros.review_ledger import NOT_PERFORMED_PHRASES

        what = {
            "coupling_not_performed": (NOT_PERFORMED_PHRASES["coupling_not_performed"]
                                       + f" — asked of: {', '.join(asked) or 'no seat'}"),
            "change_unanswered": NOT_PERFORMED_PHRASES["change_unanswered"],
            "review_late_result_pending": (NOT_PERFORMED_PHRASES["review_late_result_pending"]
                                           + f" ({', '.join(pending_models) or 'custody open'}); no verdict is counted yet"),
        }.get(reason, f"the wave reduced to no countable answer ({reason})")
        part = {"coupling_not_performed": "coupling", "change_unanswered": "change"}.get(reason, "")
        seat_lines = _uncounted_part_seat_lines(rows, part) if part else []
        if reason == "coupling_not_performed" and not asked:
            # No seat read the work: only a retrieving row can answer Part 2.
            advice = ("retry the commit or configure a retrieving reviewer seat "
                      "(Settings → Agents, a Reviewer row that reads the work itself).")
        elif seat_lines:
            advice = ("retry the commit — the seats asked answered in a form the gate cannot count; "
                      "each seat's error is listed above and recorded.")
        else:
            advice = "retry the commit."
        blocked_msg = (
            f"⚠️ REVIEW_BLOCKED: review NOT_PERFORMED — {what}.\n"
            + "".join(f"  - {line}\n" for line in seat_lines)
            + f"The commit gate counts only a PASS/FAIL answer; {advice}" + errored_note
        )
        outcome = _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review was not performed; an explicit author decision is required. ",
        )
        # The wave's typed diagnostics (per-seat errors, FAIL rows the gate could
        # not count) reach the author on this early branch as on the full path.
        if outcome is None:
            for warning in getattr(ctx, "_last_review_advisory_findings", []) or []:
                _append_review_warning(ctx, warning)
        return outcome

    if critical_fails:
        # All parse issues get a parse_failure block reason.
        all_parse = all("Could not parse" in f for f in critical_fails)
        ctx._last_review_block_reason = "parse_failure" if all_parse else "critical_findings"
        if blocking_review:
            return _build_critical_block_message(
                ctx, commit_message, critical_fails, advisory_warns, errored_note,
            )

        _record_advisory_override(ctx, "; ".join(critical_fails[:5]))
        _append_review_warning(
            ctx,
            ("Cyber Pro: critical review findings do not prohibit action."
             if not review_enforcement_blocks("blocking") else
             "Review enforcement=Advisory: critical findings require an explicit author decision before committing."),
        )
        for finding in getattr(ctx, "_last_review_critical_findings", []) or []:
            _append_review_warning(ctx, finding)

    if not critical_fails:
        # All clear: reset iteration state. With critical findings present
        # (advisory enforcement), the anti-thrashing history must SURVIVE so
        # repeat findings on the next attempt are still recognized as repeats.
        ctx._review_iteration_count = 0
        ctx._review_history = []

    if errored_note or advisory_warns or getattr(ctx, "_last_review_advisory_findings", None):
        for warning in getattr(ctx, "_last_review_advisory_findings", []) or []:
            _append_review_warning(ctx, warning)
        if errored_note:
            _append_review_warning(ctx, errored_note.strip())
    return None


def _run_unified_review(ctx: ToolContext, commit_message: str,
                        review_rebuttal: str = "",
                        repo_dir=None,
                        goal: str = "",
                        scope: str = "") -> Optional[str]:
    """Run triad pre-commit review; return a block message or ``None``.

    Assembly and dispatch are two phases (Q25=A): callers that need admission
    (``run_parallel_review``) prepare BOTH gate packets before dispatching
    either; this wrapper keeps the single-call contract for everyone else."""
    prepared, early_result, exited = _prepare_unified_review(
        ctx, commit_message, review_rebuttal=review_rebuttal,
        repo_dir=repo_dir, goal=goal, scope=scope,
    )
    if exited:
        return early_result
    return _dispatch_unified_review(ctx, commit_message, prepared)


# v7next F2.3a (D06): moved spans live in their owner leaves; re-exported
# here so this facade stays the single import surface for callers and tests.
from ouroboros.tools.review_multi_model import (  # noqa: E402, F401 -- intentional public re-exports
    CONCURRENCY_LIMIT,
    MAX_MODELS,
    TRIAD_USER_TURN,
    _CONSTITUTIONAL_PREAMBLE,
    _handle_multi_model_review,
    _multi_model_review_async,
    _query_model,
    _review_output_budget,
)
