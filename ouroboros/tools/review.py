"""Multi-model review and unified pre-commit review gate."""

import os
import json
import asyncio
import logging
import pathlib
from typing import Any, List, Optional

from ouroboros.llm import LLMClient
from ouroboros.utils import (
    run_cmd,
    append_jsonl,
    estimate_tokens,  # noqa: F401 — patchable seam: fit_triad_prompt resolves it through THIS namespace
    truncate_review_artifact,
    utc_now_iso,
)
from ouroboros import config as _cfg
from ouroboros.review_substrate import SLOT_ID_PREFIX, TYPED_FAILURE_FACT_KEYS, slot_id_for_row
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.triad_review import (
    REVIEW_JSON_ARRAY_CONTRACT,
    extract_json_array,
    parse_model_review_results,
    review_query_error_payload as _review_query_error_payload,
)
from ouroboros.tools.review_response import (
    parse_model_response as _parse_model_response,
    review_operation_fields as _review_operation_fields,
)

log = logging.getLogger(__name__)

MAX_MODELS = 10
CONCURRENCY_LIMIT = 5

_CONSTITUTIONAL_PREAMBLE = """\
## CONSTITUTIONAL CONTEXT — TOP PRIORITY

You are reviewing code for Ouroboros, a self-creating AI agent. The following
is its Constitution (BIBLE.md). **All recommendations you make MUST comply
with these principles.** If a potential improvement would violate any
principle, do NOT recommend it — flag the conflict instead.

Key principles to watch for:
- **P5 (LLM-First):** All decisions through the LLM. No hardcoded logic
  for behavior selection. If behavior can be described in a prompt, it goes
  in a prompt, not in code.
- **P7 (Minimalism):** Complexity is the enemy. Every line justifies its existence.
- **P0 (Agency):** Ouroboros is not a tool but a becoming personality.
- **P1 (Continuity):** BIBLE.md and identity.md are protected.
- **P2 (Meta-over-Patch):** Fix the class, not the instance.
- **P3 (Immune Integrity):** Review surfaces, plan review, and durable memory cannot be weakened.

If you are unsure whether a recommendation complies with the Constitution,
err on the side of NOT recommending it and explain the tension.

---

"""


# The window/limit names below stay importable and MONKEYPATCHABLE on this
# module: ``review_admission.fit_triad_prompt`` resolves them through this
# namespace at call time (tests pin that seam).
from ouroboros.reviewer_window import reviewer_context_window, window_scaled_reserves  # noqa: F401
from ouroboros.tools.review_synthesis import quorum_input_token_limit as _quorum_input_token_limit  # noqa: F401
from ouroboros.tools.review_helpers import (
    REPO_ROOT as _REPO_ROOT,
    load_checklist_section as _load_checklist_section_precise,
    load_governance_doc,
    build_touched_file_pack,
    build_goal_section,
    build_scope_section,
    review_drive_root,
    build_rebuttal_section,
    CRITICAL_FINDING_CALIBRATION,
    REPO_ANTI_PATTERN_LOCK_GUARD,
    REVIEW_PREAMBLE,
    build_self_verification_template,
    build_review_history_section as _build_review_history_section,
    calibrated_input_token_limit,  # noqa: F401 — patchable seam (see note above)
    emit_review_usage,
    format_name_status_for_preflight,
    format_review_history_entry as _format_review_entry,
    REVIEW_PROMPT_TOKEN_BUDGET,  # noqa: F401 — patchable seam (see note above)
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
                    "Record a task-result claim, checklist, evidence, and optional agent disposition. "
                    "For a root task in auto/required mode this is a cheap evidence call: the host runs "
                    "the only authoritative reviewer panel after the turn becomes structurally eligible. "
                    "Child-task and off-mode behavior is unchanged."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "Final claim or task result the agent intends to release."},
                        "goal": {"type": "string", "description": "Original task goal."},
                        "evidence": {"type": "object", "description": "Relevant tool trace, artifacts, tests, and observed facts."},
                        "checklist": {"type": "string", "default": "", "description": "Optional acceptance checklist."},
                        "agent_disposition": {
                            "type": "string",
                            "enum": ["accepted", "rejected", "partial", "deferred"],
                            "default": "",
                            "description": "Optional agent-authored stance on the acceptance review: accepted, rejected, partial, or deferred. Advisory only.",
                        },
                        "rationale": {
                            "type": "string",
                            "default": "",
                            "description": "Optional concise rationale for agent_disposition, especially when rejecting, partially accepting, or deferring reviewer feedback. If rationale is provided without a disposition, the stance defaults to partial.",
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
) -> str:
    from ouroboros.config import get_task_review_mode, resolve_effort
    from ouroboros.review_evidence import (
        build_task_acceptance_evidence,
        task_acceptance_evidence_revision,
    )
    from ouroboros.task_results import resolve_task_lineage

    # v6.51.0 idea-2: build the process-aware evidence packet (full contract +
    # first-class verification_summary + host-collected redacted repo_diff + leak-safe
    # artifacts + provenance tags). The agent-tool (auto) path has no host-owned turn
    # trace, so there is no tool_trajectory and include_recent_commit stays False (it
    # cannot prove a commit happened THIS turn). The agent's own evidence is preserved
    # under `agent_supplied` (its repo_diff demoted to agent_supplied_repo_diff) — never
    # promoted to host-fact status; repo_diff is ALWAYS the HOST-collected structural fact.
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

    agent_evidence = dict(evidence or {})
    # Bind the cheap evidence revision to the agent's actual acceptance claim,
    # goal, and checklist as well as its supporting references.  Otherwise two
    # materially different claims over the same evidence dict would share a
    # misleading revision even though the host panel must treat them separately.
    agent_evidence["acceptance_request"] = {
        "claim": str(claim or ""),
        "goal": str(goal or ""),
        "checklist": str(checklist or ""),
    }
    disposition = str(agent_disposition or "").strip().lower()
    if disposition not in {"accepted", "rejected", "partial", "deferred"}:
        disposition = ""
    agent_rationale = " ".join(str(rationale or "").split()).strip()
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
    if disposition or agent_rationale or normalized_ob:
        agent_decision = {
            "disposition": disposition or "partial",
            "rationale": agent_rationale[:1000],
            "source": "agent_task_acceptance_review_tool",
        }
        if normalized_ob:
            agent_decision["obligation_dispositions"] = normalized_ob
        agent_evidence["agent_decision"] = agent_decision

    evidence = build_task_acceptance_evidence(
        ctx,
        agent_evidence=agent_evidence,
        drive_root=pathlib.Path(ctx.drive_root) if getattr(ctx, "drive_root", None) else None,
        task_id=str(getattr(ctx, "task_id", "") or ""),
    )

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
    if get_task_review_mode() in {"auto", "required"} and is_root_task:
        evidence_revision = task_acceptance_evidence_revision(evidence)
        deferred = {
            "status": "deferred_to_host_acceptance",
            "authoritative": False,
            "evidence_revision": evidence_revision,
            "request": {
                "surface": "task_acceptance",
                "goal": str(goal or ""),
                "subject": str(claim or ""),
                "checklist": str(checklist or ""),
                "task_id": task_id,
            },
            # The host rebuilds host-attested evidence at the authoritative
            # fence, but it cannot reconstruct the agent's claims/references
            # from the capped tool trajectory.  Preserve the already redacted,
            # bounded agent-supplied section in this existing trace record so
            # the one host panel sees exactly what the cheap root call recorded.
            "evidence_refs": {
                "revision": evidence_revision,
                "sections": sorted(
                    str(key) for key in evidence if str(key) != "__provenance__"
                ),
                "canonical_payload": evidence.get("canonical_payload") or {},
                "aliases": evidence.get("aliases") or {},
                "provenance": evidence.get("__provenance__") or {},
            },
            "agent_supplied": evidence.get("agent_supplied") or {},
        }
        if agent_decision:
            deferred["agent_decision"] = agent_decision
        return json.dumps(deferred, ensure_ascii=False, indent=2, default=str)

    from ouroboros.review_substrate import (
        ReviewRequest,
        build_improvement_capsule,
        dissent_findings,
        reviewer_slots,
        run_review_request,
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
    )
    # Task acceptance alone stays API-only by owner decision (D15); configured
    # rows now also route commit, scope, advisory, plan, and Skill Review.
    slots = reviewer_slots(effort=resolve_effort("review"), role_hint="task acceptance")
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


def _handle_multi_model_review(ctx: ToolContext, content: str = "",
                                prompt: str = "", models: list = None,
                                stable_prefix_len: int = 0,
                                routes: list = None,
                                session_task: str = "",
                                session_root: str = "",
                                row_plan: dict = None,
                                surface: str = "multi_model_review",
                                session_policy: dict = None,
                                usage_attribution: dict = None,
                                retry_key: str = "") -> str:
    if models is None:
        models = []
    try:
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run,
                    _multi_model_review_async(content, prompt, models, ctx, stable_prefix_len,
                                              routes, session_task, session_root, row_plan,
                                              surface, session_policy, usage_attribution,
                                              retry_key),
                ).result()
        except RuntimeError:
            result = asyncio.run(_multi_model_review_async(content, prompt, models, ctx, stable_prefix_len,
                                                           routes, session_task, session_root, row_plan,
                                                           surface, session_policy, usage_attribution,
                                                           retry_key))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Multi-model review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


def _review_output_budget() -> int:
    """Reviewer response reservation. The operator may lower it to fit a full
    input pack plus output in context; floor 8192 preserves a useful verdict and
    the knob can never raise the 65536 default."""
    try:
        raw = int(os.environ.get("OUROBOROS_REVIEW_MAX_TOKENS", "") or 65536)
    except (TypeError, ValueError):
        raw = 65536
    return max(8192, min(raw, 65536))


async def _query_model(
    llm_client: LLMClient,
    model: str,
    messages: list,
    semaphore,
    ctx: Optional[ToolContext] = None,
    slot_id: str = SLOT_ID_PREFIX,
    route: Any = None,
    session_task: str = "",
    session_root: str = "",
    effort: str = "",
    session_target: str = "",
    session_profile: str = "",
    surface: str = "multi_model_review", session_policy: dict = None, usage_attribution: dict = None,
    retry_key: str = "", subagent_id: str = "",
):
    async with semaphore:
        slot = None
        try:
            from ouroboros.review_execution import ReviewRouteKind
            from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
            slot_route = route if route is not None else ReviewRouteKind.API_CHAT
            delegated = slot_route is ReviewRouteKind.AGENT_SESSION
            # RETRIEVES class (session row OR configured-subagent api row): the
            # compact session task replaces the assembled pack for both.
            retrieves = delegated or (
                bool(subagent_id) and slot_route is ReviewRouteKind.API_CHAT
            )
            _out_budget = _review_output_budget()
            request = ReviewRequest(
                surface=surface,
                goal="Run independent multi-model review over the supplied evidence.",
                # 5.2: a retrieving slot never receives the assembled api pack.
                messages=[] if retrieves else messages,
                task_id=str(getattr(ctx, "task_id", "") or "multi_model_review") if ctx is not None else "multi_model_review",
                call_type="multi_model_review",
                max_tokens=_out_budget,
                temperature=0.2,
                no_proxy=True,
                session_task=session_task if retrieves else "",
                session_root=session_root if retrieves else "",
                policy=(session_policy or {"output_contract": REVIEW_JSON_ARRAY_CONTRACT}) if retrieves else {},
                usage_attribution=usage_attribution or {},
                task_attempt=getattr(ctx, "task_attempt", None) if ctx is not None else None,
                retry_key=str(retry_key or ""),
                reconcile_only=bool(getattr(ctx, "_review_reconcile_only", False)),
            )
            slot = ReviewSlot(
                slot_id=slot_id,
                model=model,
                effort=effort or _cfg.resolve_effort("review"),
                max_tokens=_out_budget,
                temperature=0.2,
                role_hint="multi-model review",
                use_local=_cfg.review_model_uses_local(model),
                route=slot_route,
                session_target=session_target if delegated else "",
                session_profile=session_profile if delegated else "",
                subagent_id=str(subagent_id or ""),
            )
            loop = asyncio.get_running_loop()
            run_result = await loop.run_in_executor(
                None,
                lambda: run_review_request(
                    request,
                    slots=[slot],
                    drive_root=review_drive_root(ctx),
                    llm=llm_client,
                    usage_ctx=ctx,
                ),
            )
            actor = (run_result.actors or [{}])[0]
            # Carry the substrate's real row id instead of re-deriving position.
            ran_as = str(actor.get("slot_id") or slot_id)
            typed = {key: actor.get(key) for key in TYPED_FAILURE_FACT_KEYS if actor.get(key) not in (None, "")}
            if actor.get("status") not in {"ok", "empty"}:
                return model, {
                    "error": f"Error: {actor.get('error') or actor.get('status') or 'review failed'}",
                    "usage": actor.get("usage") or {},
                    "slot_id": ran_as,
                    **_review_operation_fields(actor),
                    "prompt_ref": actor.get("prompt_ref") or {},
                    "response_ref": actor.get("response_ref") or {},
                    **typed,
                }, None
            payload = {
                "choices": [{"message": {"content": actor.get("raw_text") or ""}}],
                "usage": actor.get("usage") or {},
                "slot_id": ran_as,
                **_review_operation_fields(actor),
                "prompt_ref": actor.get("prompt_ref") or {},
                "response_ref": actor.get("response_ref") or {},
            }
            return model, payload, None
        except Exception as e:
            # Preserve full review errors; helper adds an omission note if needed.
            error_msg = truncate_review_artifact(str(e), limit=4000)
            error = f"Error: {error_msg}"
            return model, _review_query_error_payload(ctx=ctx, model=model, messages=messages, slot_id=slot_id, error=error, slot=slot), None


async def _multi_model_review_async(content: str, prompt: str,
                                     models: list, ctx: ToolContext,
                                     stable_prefix_len: int = 0,
                                     routes: list = None,
                                     session_task: str = "",
                                     session_root: str = "",
                                     row_plan: dict = None,
                                     surface: str = "multi_model_review",
                                     session_policy: dict = None,
                                     usage_attribution: dict = None,
                                     retry_key: str = ""):
    from ouroboros.review_execution import ReviewRouteKind

    row_routes = list(routes or []) + [ReviewRouteKind.API_CHAT] * max(0, len(models) - len(routes or []))
    # Per-row strength/target/identity vectors (6.1). Absent tails keep the
    # historical behavior: global effort, shared session route, positional ids.
    def _row_vector(key, filler):
        rows = list((row_plan or {}).get(key) or [])
        return rows + [filler(idx) for idx in range(len(rows), len(models))]

    row_efforts = _row_vector("efforts", lambda idx: "")
    row_targets = _row_vector("session_targets", lambda idx: "")
    row_profiles = _row_vector("session_profiles", lambda idx: "")
    row_ids = _row_vector("slot_ids", lambda idx: slot_id_for_row(idx + 1))
    row_actors = _row_vector("subagent_ids", lambda idx: "")
    # Pack assembly follows the RETRIEVES class, not the route name: an
    # api-route row bound to a configured subagent retrieves with its own
    # tools and must never trigger (or be counted into) the assembled pack.
    any_api_rows = any(
        route is ReviewRouteKind.API_CHAT and not row_actors[idx]
        for idx, route in enumerate(row_routes[:len(models)])
    )
    if not content:
        return {"error": "content is required"}
    if not prompt and any_api_rows:
        return {"error": "prompt is required"}
    if not models:
        return {"error": "models list is required"}
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        return {"error": "models must be a list of strings"}
    if len(models) > MAX_MODELS:
        return {"error": f"Too many models ({len(models)}). Maximum is {MAX_MODELS}."}

    bible_text = load_governance_doc(_REPO_ROOT, "BIBLE.md", on_missing="explicit")
    if bible_text:
        stable_head = (
            _CONSTITUTIONAL_PREAMBLE
            + "### BIBLE.md (Full Text)\n\n" + bible_text
            + "\n\n---\n\n## REVIEW INSTRUCTIONS\n\n"
        )
    else:
        log.warning("Proceeding without BIBLE.md — constitutional compliance cannot be guaranteed")
        stable_head = (
            _CONSTITUTIONAL_PREAMBLE
            + "(BIBLE.md could not be loaded)\n\n## REVIEW INSTRUCTIONS\n\n"
        )

    # System content is split at the caller-declared stable/dynamic boundary so
    # the byte-stable prefix (constitutional preamble + BIBLE + the prompt's own
    # stable governance head) carries a provider cache marker; per-round evidence
    # stays in the unmarked tail. Callers that pass no boundary still get the
    # preamble+BIBLE prefix cached. Built ONLY when an api row will send it —
    # a panel of session rows never assembles the api pack (5.2).
    if any_api_rows:
        from ouroboros.tools.review_helpers import cached_prompt_blocks

        boundary = max(0, min(int(stable_prefix_len or 0), len(prompt)))
        messages = [
            {
                "role": "system",
                "content": cached_prompt_blocks(stable_head + prompt[:boundary], prompt[boundary:]),
            },
            {"role": "user", "content": content},
        ]
    else:
        messages = []

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    llm_client = LLMClient()
    tasks = [
        _query_model(llm_client, m, messages, semaphore, ctx, slot_id=row_ids[idx],
                     route=row_routes[idx], session_task=session_task, session_root=session_root,
                     effort=row_efforts[idx], session_target=row_targets[idx],
                     session_profile=row_profiles[idx], surface=surface,
                     session_policy=session_policy, usage_attribution=usage_attribution,
                     retry_key=retry_key, subagent_id=row_actors[idx])
        for idx, m in enumerate(models)
    ]
    results = await asyncio.gather(*tasks)

    review_results = []
    for model, result, headers_dict in results:
        review_result = _parse_model_response(model, result, headers_dict)
        emit_review_usage(
            ctx,
            model=review_result.get("model", ""),
            provider=review_result.get("provider", "openrouter"),
            usage={
                "prompt_tokens": review_result.get("tokens_in", 0),
                "completion_tokens": review_result.get("tokens_out", 0),
                "cached_tokens": review_result.get("cached_tokens", 0),
                "cache_write_tokens": review_result.get("cache_write_tokens", 0),
                "prompt_cache_ttl": review_result.get("prompt_cache_ttl", ""),
                "cost": review_result.get("cost_estimate"),
            },
            source="review",
        )
        review_results.append(review_result)

    return {
        "model_count": len(models),
        "constitutional_context": bool(bible_text),
        "results": review_results,
    }


# Unified pre-commit review gate.

def _load_checklist_section() -> str:
    """Load Repo Commit Checklist, fail-closed if missing/malformed.

    The standing-disclosure archive rides along: packet-only (api) reviewers
    have no repository tools, so a bare pointer to docs/CHECKLISTS_ARCHIVE.md
    would be unresolvable for them and settled owner-accepted narrowings could
    be re-raised (#447 stage-3 wave). The archive is small and binding — the
    extraction slimmed the live checklist FILE, not the reviewer's contract."""
    try:
        section = _load_checklist_section_precise("Repo Commit Checklist")
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise FileNotFoundError(
            f"docs/CHECKLISTS.md not found or malformed: {e}"
        ) from e
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


# The triad prompt is assembled STABLE-FIRST for provider prompt caching:
# fixed instructions + checklist + governance docs form a byte-stable prefix
# reused across review rounds (marked with a cache breakpoint at dispatch),
# while goal/scope/files/diff/history are the per-commit dynamic tail.
_REVIEW_PROMPT_TEMPLATE_STABLE = """\
{preamble}

## Review instructions

Read the staged diff and the supplied post-change file context (both appear
AFTER the governance documents below). On very large changes, the fit note may
replace duplicated full-file snapshots with a path manifest; in that case the
complete added/deleted lines remain in the staged diff. Review every checklist
item, report every distinct current problem, and make every FAIL actionable
with file/symbol evidence and a concrete fix.

{critical_calibration}

{json_contract}

If an open obligation record below already names an `obligation_id` for this root cause,
reuse that exact `obligation_id`. Do NOT invent a new id when the same root cause persists.

## Anti pattern-lock guard

Run the shared semantic-breadth guard before returning:
{anti_pattern_lock_guard}

{checklist_section}

- Output ONLY a valid JSON array.  No markdown fences, no text outside the JSON.

## DEVELOPMENT.md

{dev_guide_text}

## DESIGN.md

{design_text}

## ARCHITECTURE.md

{architecture_section}
"""

_REVIEW_PROMPT_TEMPLATE_DYNAMIC = """\
{goal_section}

{scope_section}

## Current touched files (full content)

{current_files_section}

## Staged diff

{diff_text}

## Changed files

{changed_files}

{rebuttal_section}{review_history_section}
"""


def _parse_review_json(raw: str) -> Optional[list]:
    """Best-effort extraction of a JSON array from model output."""
    return extract_json_array(raw, normalize=True)


def _git_show_staged(repo_dir, path: str) -> str:
    """Return staged index content via ``git show :PATH`` or ``""``."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "show", f":{path}"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


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
    the semantic checklist: docs/CHECKLISTS.md item 6 (tests_affected) and
    item 8 (version_bump).
    """
    import re
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
    version_staged = "VERSION" in active_staged

    # VERSION staged but README missing.
    if version_staged and "README.md" not in active_staged:
        return (
            "⚠️ PREFLIGHT_BLOCKED: Staged diff is incomplete — fix before review.\n"
            "  Missing from staged: README.md (badge + changelog)\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}\n\n"
            "Stage all related files together. Use write_file for all files first,\n"
            "then commit_reviewed to stage and commit everything in one diff."
        )

    # The version-reference and tests-required lexical heuristics were removed
    # here (false blocks: a "conversion" commit told to bump VERSION; a
    # comment-only .py diff refused for missing tests). See docstring —
    # CHECKLISTS.md items 6/8 own these duties semantically.

    # New logic modules require active ARCHITECTURE.md update.
    new_logic_files = [
        f for f in new_files
        if f.startswith(("ouroboros/", "supervisor/")) and f.endswith(".py")
    ]
    if new_logic_files and "docs/ARCHITECTURE.md" not in active_staged:
        return (
            "⚠️ PREFLIGHT_BLOCKED: New files added in ouroboros/ or supervisor/ "
            "but docs/ARCHITECTURE.md is not staged.\n"
            "  New structural additions must be documented in ARCHITECTURE.md "
            "(Bible P6: authenticity / architectural mirror).\n"
            f"  New files: {new_logic_files[:5]}\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
        )

    # VERSION changes must keep staged version carriers synchronized.
    if version_staged:
        try:
            from ouroboros.tools.release_sync import (
                is_release_version,
                version_carrier_desyncs,
            )
            version_str = _git_show_staged(repo_dir, "VERSION").strip()
            if is_release_version(version_str):
                desync = version_carrier_desyncs(
                    version_str,
                    pyproject_text=_git_show_staged(repo_dir, "pyproject.toml"),
                    uv_lock_text=_git_show_staged(repo_dir, "uv.lock"),
                    web_package_text=_git_show_staged(repo_dir, "web/package.json"),
                    readme_text=_git_show_staged(repo_dir, "README.md"),
                    arch_text=_git_show_staged(repo_dir, "docs/ARCHITECTURE.md"),
                    api_types_text=_git_show_staged(repo_dir, "web/modules/api_types.js"),
                    download_readme_text=_git_show_staged(repo_dir, "README.md"),
                    site_install_text=_git_show_staged(repo_dir, "site/install/index.html"),
                    docs_install_text=_git_show_staged(repo_dir, "docs/install/index.html"),
                    detailed=True,
                )
                if desync:
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: VERSION file says {version_str} but "
                        "the following staged files have a different version value:\n"
                        + "".join(f"  - {d}\n" for d in desync)
                        + "Update all version references to match VERSION before committing.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
        except Exception:
            pass  # Non-fatal: LLM reviewers handle version sync

    # VERSION changes need a staged README changelog row, and the staged README
    # must respect P9 history limits.
    if version_staged:
        try:
            from ouroboros.tools.release_sync import is_release_version
            version_str = _git_show_staged(repo_dir, "VERSION").strip()
            if is_release_version(version_str):
                readme_text = _git_show_staged(repo_dir, "README.md")
                if readme_text and not re.search(r'\|\s*' + re.escape(version_str) + r'\s*\|', readme_text):
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: VERSION is {version_str} but README.md "
                        "changelog has no table row for this version.\n"
                        "  Add a changelog entry in the Version History table in README.md.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
        except Exception:
            pass  # Non-fatal
        try:
            readme_staged = _git_show_staged(repo_dir, "README.md")
            if readme_staged:
                from ouroboros.tools.release_sync import check_history_limit
                limit_warnings = check_history_limit(readme_staged)
                if limit_warnings:
                    return (
                        "⚠️ PREFLIGHT_BLOCKED: README.md Version History exceeds BIBLE.md P9 limits.\n"
                        + "".join(f"  - {w}\n" for w in limit_warnings)
                        + "  Trim the oldest entry in the over-limit category before committing.\n"
                        + "  Quick check: python -c \"from ouroboros.tools.release_sync import "
                        "check_history_limit; print(check_history_limit(open('README.md').read()))\"\n"
                        + f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
        except Exception:
            pass  # Non-fatal: LLM reviewers handle P9 limits as advisory fallback

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
    """Either block immediately or downgrade to advisory warning."""
    if blocking_review:
        return blocked_msg
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


def _collect_review_findings(ctx: ToolContext, model_results: list) -> tuple[list[str], list[str], list[str], list[dict]]:
    parsed = parse_model_review_results({"results": model_results})
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
            continue
        for item in record.parsed_items:
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
        "enforcement. If the same finding repeats after a rebuttal, implement the fix\n"
        "instead of re-arguing.\n\n"
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


def _triad_session_task(ctx: ToolContext, **sections) -> str:
    """Compat shim over ``review_subject.build_triad_session_task`` (5.2/5.3):
    same session task text; a managed subject inlines its authoritative delta."""
    from ouroboros.tools.review_subject import build_triad_session_task

    return build_triad_session_task(**sections)


def _capture_triad_staged_diff(
    ctx: ToolContext, target_repo, blocking_review: bool
) -> tuple[Optional[str], Optional[Any], Optional[str]]:
    """Capture the triad's review-diff evidence, or route a capture failure.

    Returns ``(diff_text, subject, None)`` on success — ``subject`` is the
    managed resolution-delta artifact, ``None`` for an ordinary commit whose
    evidence stays the byte-exact hardened staged diff — and
    ``(None, None, block_result)`` on failure: the fail-closed message in
    blocking mode, ``None`` (advisory skip) otherwise. A genuine failure fails
    closed rather than reviewing a placeholder that would yield authoritative
    findings about a diff nobody has.
    """
    from ouroboros.tools.review_binary_context import (
        StagedDiffUnavailable, capture_staged_diff)
    from ouroboros.tools.review_subject import managed_review_subject

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


def _prepare_unified_review(ctx: ToolContext, commit_message: str,
                            review_rebuttal: str = "",
                            repo_dir=None,
                            goal: str = "",
                            scope: str = "") -> tuple:
    """Assemble the triad packet WITHOUT dispatching any reviewer (Q25=A).

    Returns ``(prepared, early_result, exited)``: ``exited=True`` means the
    triad terminated during assembly and ``early_result`` (a block message, or
    ``None`` for an advisory skip / empty diff) is its final answer — nothing
    may be dispatched for it; otherwise ``prepared`` carries everything
    ``_dispatch_unified_review`` needs."""
    target_repo = repo_dir or ctx.repo_dir
    ctx._review_iteration_count += 1
    ctx._last_review_block_reason = ""  # reset per attempt
    ctx._last_triad_models = []  # reset forensic field so stale values never persist on early exit
    ctx._last_review_critical_findings = []  # reset to avoid stale findings from previous attempts
    ctx._last_triad_raw_results = []  # reset per-model actor records
    ctx._triad_withheld_seat_records = []  # reset Q28-dropped seat records
    ctx._review_degraded_reasons = []  # reset degraded participation markers
    review_enforcement = _cfg.get_review_enforcement()
    blocking_review = review_enforcement == "blocking"

    diff_text, subject, capture_block = _capture_triad_staged_diff(ctx, target_repo, blocking_review)
    if diff_text is None:  # capture failed: block (blocking) or advisory-skip (None)
        return None, capture_block, True
    if not diff_text.strip():
        return None, None, True

    try:
        changed = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=target_repo)
    except Exception:
        changed = ""
    # Reviewers of a managed resolution read the RESOLUTION path set (delta ∪
    # conflict anchors); the preflight staged list below stays on the FULL
    # candidate (I2 — full-tree invariants are never narrowed).
    review_changed = "\n".join(subject.touched_paths()) if subject is not None else changed

    preflight_staged = _build_preflight_staged(target_repo, fallback=changed)

    preflight_err = _preflight_check(commit_message, preflight_staged, target_repo)
    if preflight_err:
        ctx._last_review_block_reason = "preflight"
        result = _handle_review_block_or_warning(
            ctx, blocking_review, preflight_err,
            "Review enforcement=Advisory: preflight warning did not block commit. ",
        )
        if result is not None:
            return None, result, True

    rebuttal_section = build_rebuttal_section(review_rebuttal)

    try:
        checklist_section = _load_checklist_section()
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

    dev_guide_text = load_governance_doc(pathlib.Path(ctx.repo_dir), "docs/DEVELOPMENT.md", on_missing="explicit")
    design_text = load_governance_doc(pathlib.Path(ctx.repo_dir), "docs/DESIGN.md", on_missing="explicit")
    architecture_text = load_governance_doc(pathlib.Path(ctx.repo_dir), "docs/ARCHITECTURE.md", on_missing="explicit")

    # Durable open obligations reduce review thrashing across restarts.
    _open_obs_for_review = []
    try:
        from ouroboros.review_state import load_state, make_repo_key
        _rs = load_state(pathlib.Path(ctx.drive_root))
        _repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
        _open_obs_for_review = _rs.get_open_obligations(repo_key=_repo_key)
    except Exception:
        pass  # Non-fatal: anti-thrashing hint is best-effort
    review_history_section = _build_review_history_section(
        ctx._review_history, open_obligations=_open_obs_for_review,
    )

    # Build touched-file pack for full current context (managed: the reviewed
    # resolution set; binary rows carry the M0 baseline identity).
    try:
        touched_paths = [f.strip() for f in review_changed.strip().splitlines() if f.strip()]
        current_files_section, _omitted = build_touched_file_pack(
            pathlib.Path(target_repo),
            touched_paths,
            represent_binary=subject is not None,
            m0_tree=getattr(subject, "m0_tree", "") or "",
            staged_tree=getattr(subject, "staged_tree", "") or "",
        )
        if _omitted:
            current_files_section += (
                f"\n\n⚠️ OMISSION NOTE: {len(_omitted)} file(s) omitted from direct context: "
                f"{', '.join(_omitted)}"
            )
        if not current_files_section.strip():
            current_files_section = "(no touched files could be read)"
    except Exception as e:
        log.warning("Failed to build touched file pack for triad review: %s", e)
        current_files_section = f"(touched file pack unavailable: {e})"

    # Per-row identity/delivery/strength from the ONE reviewer-slot SSOT (6.1):
    # structured rows when configured, the migrated comma-lists otherwise. A
    # malformed configuration is an infra failure, never a silent api spend.
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.reviewer_slot_config import commit_triad_delivery

    try:
        row_plan = commit_triad_delivery()
    except ValueError as exc:
        ctx._last_review_block_reason = "infra_failure"
        return None, _handle_review_block_or_warning(
            ctx, blocking_review,
            f"⚠️ REVIEW_BLOCKED: invalid reviewer-slot configuration — {exc}",
            "Review enforcement=Advisory: invalid reviewer-slot configuration did not block commit. ",
        ), True
    models, row_routes = row_plan["models"], row_plan["routes"]
    ctx._last_triad_models = list(models)  # forensic: actual resolved model IDs
    _row_actors = list(row_plan.get("subagent_ids") or [])
    # Packet rows only: a configured-subagent api row is the RETRIEVES class —
    # it neither constrains the fit ladder nor counts as an api seat for the
    # Q28-A yield arithmetic below.
    api_models = [
        m for i, (m, r) in enumerate(zip(models, row_routes))
        if r is ReviewRouteKind.API_CHAT
        and not (i < len(_row_actors) and _row_actors[i])
    ]

    goal_section = build_goal_section(goal, scope, commit_message)
    scope_section = build_scope_section(scope)

    def _assemble_prompt(files_section: str, staged_diff: str) -> tuple:
        """Return (prompt, stable_prefix_len): the stable governance prefix is
        byte-identical across rounds and becomes the cache-marked block."""
        stable = _REVIEW_PROMPT_TEMPLATE_STABLE.format(
            preamble=REVIEW_PREAMBLE,
            critical_calibration=CRITICAL_FINDING_CALIBRATION,
            json_contract=REVIEW_JSON_ARRAY_CONTRACT,
            anti_pattern_lock_guard=REPO_ANTI_PATTERN_LOCK_GUARD,
            checklist_section=checklist_section,
            dev_guide_text=dev_guide_text or "(DEVELOPMENT.md not found)",
            design_text=design_text or "(DESIGN.md not found)",
            architecture_section=architecture_text or "(ARCHITECTURE.md not found)",
        )
        dynamic = _REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
            goal_section=goal_section,
            scope_section=scope_section,
            current_files_section=files_section,
            rebuttal_section=rebuttal_section,
            review_history_section=review_history_section,
            diff_text=staged_diff,
            changed_files=review_changed,
        )
        return stable + "\n" + dynamic, len(stable) + 1

    # P3 stays one-pass. The api pack, its fit ladder and the fixed_overflow
    # gate exist ONLY for the api rows (5.2/5.7): a session row retrieves with
    # its own tools, so it neither constrains the fit limit nor is blocked by
    # it, and a panel with no api rows skips pack assembly entirely.
    prompt, stable_prefix_len = "", 0
    if api_models:
        prompt, stable_prefix_len, fit_error = _fit_triad_prompt(
            api_models, _assemble_prompt, current_files_section, diff_text,
            review_changed, target_repo, ctx=ctx, subject=subject,
        )
        if fit_error:
            session_count = len(models) - len(api_models)
            if session_count >= _cfg.adaptive_quorum(len(models)):
                # Q28-A: packet limits gate only the api subset. Enough
                # agent-session rows remain for the quorum, so the api rows are
                # DROPPED (recorded loudly, never silent) and the panel proceeds
                # on session delivery alone.
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
                    "irreducible packet; the panel's agent-session rows "
                    "satisfied the quorum without it ($0 spent).", only_api=True)
                row_plan = drop_api_rows(row_plan)
                models, row_routes = row_plan["models"], row_plan["routes"]
                ctx._last_triad_models = list(models)
                note = (
                    f"triad_api_rows_dropped_oversize_pack: {len(api_models)} api row(s) "
                    f"({', '.join(api_models)}) could not receive the irreducible packet; "
                    f"{session_count} agent-session row(s) satisfy the quorum and proceed"
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

    session_task = ""
    if len(api_models) < len(models):
        session_task = _triad_session_task(
            ctx,
            goal_section=goal_section,
            scope_section=scope_section,
            checklist_section=checklist_section,
            rebuttal_section=rebuttal_section,
            review_history_section=review_history_section,
            dev_guide_text=dev_guide_text,
            architecture_text=architecture_text,
            subject=subject,
        )

    return {
        "prompt": prompt, "stable_prefix_len": stable_prefix_len,
        "models": models, "routes": row_routes, "row_plan": row_plan,
        "session_task": session_task, "target_repo": target_repo,
        "blocking_review": blocking_review,
    }, None, False


def _review_actor_label(row: dict) -> str:
    return str(row.get("model_id") or row.get("slot_id") or row.get("slot") or "reviewer")


def _dispatch_unified_review(ctx: ToolContext, commit_message: str, prepared: dict) -> Optional[str]:
    """Dispatch an assembled triad packet and post-process the panel verdict."""
    blocking_review = prepared["blocking_review"]
    try:
        result_json = _handle_multi_model_review(
            ctx,
            content="Review the staged diff and context provided in the instructions above.",
            prompt=prepared["prompt"],
            models=prepared["models"],
            stable_prefix_len=prepared["stable_prefix_len"],
            routes=prepared["routes"],
            session_task=prepared["session_task"],
            session_root=str(prepared["target_repo"]),
            row_plan=prepared["row_plan"],
            retry_key=str(prepared.get("retry_key") or ""),
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
            "Review enforcement=Advisory: review infrastructure failure did not block commit. ",
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
            "Review enforcement=Advisory: review service error did not block commit. ",
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
            "Review enforcement=Advisory: review returned no model results; commit proceeding anyway. ")

    critical_fails, advisory_warns, errored_models, _triad_raw = _collect_review_findings(ctx, model_results)
    models_total = len(model_results)
    triad_raw = getattr(ctx, "_last_triad_raw_results", []) or []
    pending_models = [_review_actor_label(r) for r in triad_raw if (
        r.get("late_result_pending") or str(r.get("operation_state") or "")
        in {"in_flight", "custody_lost"})]
    if pending_models:
        ctx._last_review_block_reason = "review_late_result_pending"
        blocked_msg = ("⚠️ REVIEW_PENDING: Physical review operation(s) remain unresolved: "
                       f"{', '.join(pending_models)}. Retry the same commit to reconcile them without a blind paid resend.")
        pending_block = _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: pending review work did not block commit. ",
        )
        if pending_block is not None:
            return pending_block
    successful_reviewers = sum(1 for r in triad_raw if r.get("status") == "responded")
    failed_actors = [
        _review_actor_label(r) for r in triad_raw
        if r.get("status") not in ("responded", "not_dispatched")]
    required_quorum = _cfg.adaptive_quorum(models_total)
    if successful_reviewers < required_quorum:
        ctx._last_review_block_reason = "review_quorum"
        unavailable_str = ", ".join(failed_actors) if failed_actors else ", ".join(errored_models)
        blocked_msg = (
            f"⚠️ REVIEW_BLOCKED: Only {successful_reviewers} of {models_total} review "
            f"models responded successfully (minimum {required_quorum} required). "
            f"Unavailable/failed: {unavailable_str}.\n"
            "Retry the commit — transient model failures usually resolve quickly."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review quorum failure did not block commit. ",
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
            "Review enforcement=Advisory: critical review findings did not block commit.",
        )
        for finding in getattr(ctx, "_last_review_critical_findings", []) or []:
            _append_review_warning(ctx, finding)
        for warning in getattr(ctx, "_last_review_advisory_findings", []) or []:
            _append_review_warning(ctx, warning)
        if errored_note:
            _append_review_warning(ctx, errored_note)

    if not critical_fails:
        # All clear: reset iteration state. With critical findings present
        # (advisory enforcement), the anti-thrashing history must SURVIVE so
        # repeat findings on the next attempt are still recognized as repeats.
        ctx._review_iteration_count = 0
        ctx._review_history = []

    if errored_note or advisory_warns or getattr(ctx, "_last_review_advisory_findings", None):
        ctx._review_advisory = list(getattr(ctx, "_last_review_advisory_findings", []) or [])
        if errored_note:
            ctx._review_advisory.append(errored_note.strip())
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
