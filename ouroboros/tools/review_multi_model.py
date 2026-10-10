"""Multi-model review delivery for the commit triad: the sync/async fan-out
entry, the per-row slot dispatch through the review substrate, and their
shared limits. Extracted from ouroboros/tools/review.py (v7 L-C split, re-cut
on the v7next tip where the reviewer response parser already lives in
ouroboros/tools/review_response.py); review.py re-exports every name."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from typing import Any, Optional, TYPE_CHECKING

from ouroboros.settings_integrity import live_effort_range
from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS
from ouroboros.review_substrate import SLOT_ID_PREFIX

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.llm import LLMClient
    from ouroboros.tools.registry import ToolContext

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.tools.review")


def _rev():
    """The parent review module, read at call time.

    The review module's members stay monkeypatch-addressable at their
    historical ``ouroboros.tools.review`` bindings (tests rebind them there),
    so this leaf resolves every such cross-reference through the module at
    each call instead of freezing whatever object a from-import saw at
    import time.
    """
    from ouroboros.tools import review

    return review


# The commit panel is the review pool: every marked catalog row runs, so the
# fan-out ceiling is the catalog's own ceiling (26), not a separate lane cap.
MAX_MODELS = MAX_CONFIGURED_SUBAGENTS


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

# The one user turn every api triad row sends, and the one role hint every
# triad row (packet or native episode) carries; the commit gate's wave
# admission measures the same sends the fan-out dispatches.
TRIAD_USER_TURN = "Review the staged diff and context provided in the instructions above."
TRIAD_ROLE_HINT = "multi-model review"


def _review_output_budget() -> int:
    """Reviewer response reservation. The operator may lower it to fit a full
    input pack plus output in context; floor 8192 preserves a useful verdict and
    the knob can never raise the 65536 default."""
    try:
        raw = int(os.environ.get("OUROBOROS_REVIEW_MAX_TOKENS", "") or 65536)
    except (TypeError, ValueError):
        raw = 65536
    return max(8192, min(raw, 65536))


def review_output_allowance(model: str, window: int = 0, **binding) -> int:
    """The review reservation within the exact route's known maximum response; unknown keeps it.
    A ``window`` from ``reviewer_context_window`` supplies the account it was observed on."""
    from ouroboros.response_limits import response_allowance
    from ouroboros.reviewer_window import observed_binding

    return response_allowance(str(model or ""), _review_output_budget(), **observed_binding(window, binding))


def triad_api_messages(prompt: str, stable_prefix_len: int, content: str,
                       *, layer: str = "body") -> tuple:
    """The exact api-row message pair of a triad panel, and the BIBLE text it
    carries ("" when BIBLE.md could not be loaded, or when the checklist
    ``layer`` is ``core``: a subject that is not the Ouroboros body is not
    governed by the constitution, so the head carries no constitutional
    preamble and no BIBLE — `review_body_fact.layer_for`).

    One builder for both consumers: the fan-out sends these messages, and the
    commit gate's wave admission measures them — a reservation priced on
    anything else would admit a wave the ledger then refuses seat by seat.
    """
    bible_text = ""
    if layer != "body":
        stable_head = "## REVIEW INSTRUCTIONS\n\n"
    else:
        bible_text = _rev().load_governance_doc(_rev()._REPO_ROOT, "BIBLE.md", on_missing="explicit")
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
    # preamble+BIBLE prefix cached.
    from ouroboros.tools.review_helpers import cached_prompt_blocks

    boundary = max(0, min(int(stable_prefix_len or 0), len(prompt)))
    messages = [
        {
            "role": "system",
            "content": cached_prompt_blocks(stable_head + prompt[:boundary], prompt[boundary:]),
        },
        {"role": "user", "content": content},
    ]
    return messages, bible_text


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
                                retry_key: str = "", task_evidence: dict = None,
                                layer: str = "body") -> str:
    if models is None:
        models = []
    try:
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                # copy_context (the plan_review precedent): the caller's usage
                # scope — the fence the wave was admitted with — rides into
                # the loop thread instead of a bare pool thread's empty context.
                result = pool.submit(
                    contextvars.copy_context().run,
                    asyncio.run,
                    _multi_model_review_async(content, prompt, models, ctx, stable_prefix_len,
                                              routes, session_task, session_root, row_plan,
                                              surface, session_policy, usage_attribution,
                                              retry_key, task_evidence, layer=layer),
                ).result()
        except RuntimeError:
            result = asyncio.run(_multi_model_review_async(content, prompt, models, ctx, stable_prefix_len,
                                                           routes, session_task, session_root, row_plan,
                                                           surface, session_policy, usage_attribution,
                                                           retry_key, task_evidence, layer=layer))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Multi-model review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


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
    retry_key: str = "", subagent_id: str = "", use_local: bool | None = None, task_evidence: dict = None,
    native_retrieval: bool = False, resolved_wave_id: str = "",
):
    async with semaphore:
        slot = None
        try:
            from ouroboros.review_execution import ReviewRouteKind, delivery_retrieves
            from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
            slot_route = route if route is not None else ReviewRouteKind.API_CHAT
            delegated = slot_route is ReviewRouteKind.AGENT_SESSION
            # RETRIEVES class (session row, or an api row saved as ``native``):
            # the compact session task replaces the assembled pack. The row's
            # catalog id is identity, never a delivery signal (F8).
            native_retrieval = bool(native_retrieval) and not delegated
            retrieves = delivery_retrieves(slot_route, native_retrieval)
            from ouroboros.review_evidence import commit_review_evidence_refs, commit_review_evidence_section
            evidence = task_evidence or {}
            policy = dict(session_policy or {"output_contract": _rev().REVIEW_JSON_ARRAY_CONTRACT}) if retrieves else {}
            if retrieves and evidence:
                session_task += "\n\n" + commit_review_evidence_section(evidence, delivery="session" if delegated else "native")
                if not delegated:
                    policy["native_data_root"] = evidence["data_root"]
            _out_budget = _review_output_budget()
            request = ReviewRequest(
                surface=surface,
                goal="Run independent multi-model review over the supplied evidence.",
                # 5.2: a retrieving slot never receives the assembled api pack.
                messages=[] if retrieves else messages,
                task_id=str(getattr(ctx, "task_id", "") or "multi_model_review") if ctx is not None else "multi_model_review",
                call_type="multi_model_review",
                max_tokens=_out_budget,
                default_temperature=0.2,
                no_proxy=True,
                session_task=session_task if retrieves else "",
                session_root=session_root if retrieves else "",
                policy=policy,
                evidence={"task_execution": evidence} if evidence else {},
                evidence_refs=commit_review_evidence_refs(evidence),
                usage_attribution=usage_attribution or {},
                resolved_wave_id=resolved_wave_id,
                task_attempt=getattr(ctx, "task_attempt", None) if ctx is not None else None,
                retry_key=str(retry_key or ""),
                reconcile_only=bool(getattr(ctx, "_review_reconcile_only", False)),
                # The owner deadline is a bound of every retrieving episode; it
                # reaches the row here exactly as the advisory passes it.
                deadline_at=_rev()._owner_deadline_at(ctx),
            )
            slot = ReviewSlot(
                slot_id=slot_id,
                model=model,
                effort=effort or live_effort_range()["max"],  # an absent vector tail: an Auto reviewer's level
                max_tokens=_out_budget,
                default_temperature=0.2,
                role_hint=TRIAD_ROLE_HINT,
                use_local=_rev()._cfg.review_model_uses_local(model) if use_local is None else use_local,
                route=slot_route,
                session_target=session_target if delegated else "",
                session_profile=session_profile,
                subagent_id=str(subagent_id or ""),
                # Explicit both ways for an api row: the slot's own delivery fact.
                native_retrieval_override=None if delegated else bool(native_retrieval),
            )
            loop = asyncio.get_running_loop()
            # run_in_executor copies no context: carry the usage scope (and its
            # bound fence) into the executor thread the substrate reserves on.
            run_result = await loop.run_in_executor(
                None,
                contextvars.copy_context().run,
                lambda: run_review_request(
                    request,
                    slots=[slot],
                    drive_root=_rev().review_drive_root(ctx),
                    llm=llm_client,
                    usage_ctx=ctx,
                ),
            )
            actor = (run_result.actors or [{}])[0]
            # Carry the substrate's real row id instead of re-deriving position.
            ran_as = str(actor.get("slot_id") or slot_id)
            typed = {key: actor.get(key) for key in _rev().TYPED_FAILURE_FACT_KEYS if actor.get(key) not in (None, "")}
            if actor.get("status") not in {"ok", "empty"}:
                return model, {
                    "error": f"Error: {actor.get('error') or actor.get('status') or 'review failed'}",
                    "usage": actor.get("usage") or {},
                    "slot_id": ran_as,
                    **_rev()._review_operation_fields(actor),
                    "prompt_ref": actor.get("prompt_ref") or {},
                    "response_ref": actor.get("response_ref") or {},
                    **typed,
                }, None
            payload = {
                "choices": [{"message": {"content": actor.get("raw_text") or ""}}],
                "usage": actor.get("usage") or {},
                "slot_id": ran_as,
                **_rev()._review_operation_fields(actor),
                "prompt_ref": actor.get("prompt_ref") or {},
                "response_ref": actor.get("response_ref") or {},
            }
            return model, payload, None
        except Exception as e:
            # Preserve full review errors; helper adds an omission note if needed.
            error_msg = _rev().truncate_review_artifact(str(e), limit=4000)
            error = f"Error: {error_msg}"
            return model, _rev()._review_query_error_payload(ctx=ctx, model=model, messages=messages, slot_id=slot_id, error=error, slot=slot), None


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
                                     retry_key: str = "", task_evidence: dict = None,
                                     layer: str = "body"):
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.reviewer_slot_config import row_plan_retrieves

    row_routes = list(routes or []) + [ReviewRouteKind.API_CHAT] * max(0, len(models) - len(routes or []))
    # Per-row strength/target/identity vectors (6.1). Absent tails keep the
    # historical behavior: global effort, shared session route, positional ids.
    def _row_vector(key, filler):
        rows = list((row_plan or {}).get(key) or [])
        return rows + [filler(idx) for idx in range(len(rows), len(models))]

    row_efforts = _row_vector("efforts", lambda idx: "")
    row_targets = _row_vector("session_targets", lambda idx: "")
    row_profiles = _row_vector("session_profiles", lambda idx: "")
    row_ids = _row_vector("slot_ids", lambda idx: _rev().slot_id_for_row(idx + 1))
    row_actors = _row_vector("subagent_ids", lambda idx: "")
    row_local = _row_vector("use_local", lambda idx: None)
    # One brief per seat (PR-3 B): a retrieving row carries ITS OWN two-part
    # brief and answer policy; an absent entry keeps the shared task/policy.
    row_tasks = _row_vector("session_tasks", lambda idx: "")
    row_policies = _row_vector("session_policies", lambda idx: None)
    # Pack assembly follows the RETRIEVES class, not the route name: a native
    # api row retrieves with its own tools and must never trigger (or be counted
    # into) the assembled pack. The class is the plan's explicit ``retrieves``
    # vector; an uncovered row is its route's own class (a bare api row receives
    # the packet) — never inferred from its catalog id (F8).
    plan = {**(row_plan or {}), "routes": row_routes}
    row_retrieves = [row_plan_retrieves(plan, idx) for idx in range(len(models))]
    any_api_rows = not all(row_retrieves)
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

    # Built ONLY when an api row will send it — a panel of session rows never
    # assembles the api pack (5.2); the constitutional flag below stays a fact
    # about the repository either way.
    if any_api_rows:
        messages, bible_text = triad_api_messages(prompt, stable_prefix_len, content, layer=layer)
    else:
        messages = []
        # The core layer carries no constitution on any delivery (review_body_fact.layer_for).
        bible_text = "" if layer != "body" else _rev().load_governance_doc(
            _rev()._REPO_ROOT, "BIBLE.md", on_missing="explicit")

    # One round, one wave: each row sends its own request, so a fan-out without a paid-cycle
    # key names its round here (review_records.resolve_review_wave); attribution only.
    from ouroboros.review_records import new_review_wave_id

    fan_out_wave = "" if retry_key or (usage_attribution or {}).get("review_wave_id") else new_review_wave_id()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    llm_client = _rev().LLMClient()
    tasks = [
        _query_model(llm_client, m, messages, semaphore, ctx, slot_id=row_ids[idx],
                     route=row_routes[idx], session_task=row_tasks[idx] or session_task, session_root=session_root,
                     effort=row_efforts[idx], session_target=row_targets[idx],
                     session_profile=row_profiles[idx], surface=surface,
                     session_policy=row_policies[idx] or session_policy, usage_attribution=usage_attribution,
                     retry_key=retry_key, subagent_id=row_actors[idx], use_local=row_local[idx], task_evidence=task_evidence,
                     resolved_wave_id=fan_out_wave,
                     native_retrieval=row_retrieves[idx] and row_routes[idx] is ReviewRouteKind.API_CHAT)
        for idx, m in enumerate(models)
    ]
    results = await asyncio.gather(*tasks)

    review_results = []
    for model, result, headers_dict in results:
        review_result = _rev()._parse_model_response(model, result, headers_dict)
        review_results.append(review_result)

    return {
        "model_count": len(models),
        "constitutional_context": bool(bible_text),
        "results": review_results,
    }
