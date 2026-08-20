"""Multi-model review delivery for the commit triad: the sync/async fan-out
entry, the per-row slot dispatch through the review substrate, the reviewer
response parser, and their shared limits. Extracted from
ouroboros/tools/review.py (v7 L-C split); review.py re-exports every name."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional, TYPE_CHECKING

from ouroboros import config as _cfg
from ouroboros.review_substrate import SLOT_ID_PREFIX
from ouroboros.tools.review_helpers import (
    REPO_ROOT as _REPO_ROOT,
    emit_review_usage,
)
from ouroboros.triad_review import (
    REVIEW_JSON_ARRAY_CONTRACT,
    review_query_error_payload as _review_query_error_payload,
)
from ouroboros.utils import truncate_review_artifact

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


MAX_MODELS = 10
CONCURRENCY_LIMIT = 5
DEFAULT_REVIEW_MODEL_TIMEOUT_SEC = 600.0

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


def _review_model_timeout_sec() -> float:
    raw = os.environ.get("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value > 0:
        return value
    if raw:
        log.warning(
            "Invalid or non-positive OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC=%r; using %.0fs",
            raw,
            DEFAULT_REVIEW_MODEL_TIMEOUT_SEC,
        )
    return DEFAULT_REVIEW_MODEL_TIMEOUT_SEC


def _handle_multi_model_review(ctx: ToolContext, content: str = "",
                                prompt: str = "", models: list = None,
                                stable_prefix_len: int = 0,
                                routes: list = None,
                                session_task: str = "",
                                session_root: str = "",
                                row_plan: dict = None) -> str:
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
                                              routes, session_task, session_root, row_plan),
                ).result()
        except RuntimeError:
            result = asyncio.run(_multi_model_review_async(content, prompt, models, ctx, stable_prefix_len,
                                                           routes, session_task, session_root, row_plan))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Multi-model review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


def _review_output_budget() -> int:
    """Reviewer response reservation (default 65536). `OUROBOROS_REVIEW_MAX_TOKENS`
    lets an operator LOWER it when a mega-diff's input pack plus the default
    output reservation exceeds a reviewer endpoint's context cap (input + output
    must fit; a verdict needs ~10K tokens, so shrinking the reservation preserves
    FULL review input context instead of trimming evidence). Floor 8192 so the
    knob can never squeeze a verdict into uselessness; never raises the default."""
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
):
    async with semaphore:
        timeout_sec = _review_model_timeout_sec()
        slot = None
        try:
            from ouroboros.review_execution import ReviewRouteKind
            from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

            slot_route = route if route is not None else ReviewRouteKind.API_CHAT
            delegated = slot_route is ReviewRouteKind.AGENT_SESSION
            _out_budget = _review_output_budget()
            request = ReviewRequest(
                surface="multi_model_review",
                goal="Run independent multi-model review over the supplied evidence.",
                # 5.2: a session slot never receives the assembled api pack.
                messages=[] if delegated else messages,
                task_id=str(getattr(ctx, "task_id", "") or "multi_model_review") if ctx is not None else "multi_model_review",
                call_type="multi_model_review",
                max_tokens=_out_budget,
                temperature=0.2,
                no_proxy=True,
                session_task=session_task if delegated else "",
                session_root=session_root if delegated else "",
                policy={"output_contract": REVIEW_JSON_ARRAY_CONTRACT} if delegated else {},
            )
            slot = ReviewSlot(
                slot_id=slot_id,
                model=model,
                effort=effort or _cfg.resolve_effort("review"),
                timeout_sec=timeout_sec,
                max_tokens=_out_budget,
                temperature=0.2,
                role_hint="multi-model review",
                use_local=_cfg.review_model_uses_local(model),
                route=slot_route,
                session_target=session_target if delegated else "",
                session_profile=session_profile if delegated else "",
            )
            loop = asyncio.get_running_loop()
            run_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: run_review_request(
                        request,
                        slots=[slot],
                        drive_root=_rev().review_drive_root(ctx),
                        llm=llm_client,
                        usage_ctx=ctx,
                    ),
                ),
                timeout=timeout_sec,
            )
            actor = (run_result.actors or [{}])[0]
            # The id the substrate REALLY ran under, so the durable actor record
            # downstream carries it instead of re-deriving one from position.
            ran_as = str(actor.get("slot_id") or slot_id)
            if actor.get("status") not in {"ok", "empty"}:
                return model, {
                    "error": f"Error: {actor.get('error') or actor.get('status') or 'review failed'}",
                    "usage": actor.get("usage") or {},
                    "slot_id": ran_as,
                    "prompt_ref": actor.get("prompt_ref") or {},
                    "response_ref": actor.get("response_ref") or {},
                }, None
            payload = {
                "choices": [{"message": {"content": actor.get("raw_text") or ""}}],
                "usage": actor.get("usage") or {},
                "slot_id": ran_as,
                "prompt_ref": actor.get("prompt_ref") or {},
                "response_ref": actor.get("response_ref") or {},
            }
            return model, payload, None
        except asyncio.TimeoutError:
            error = f"Error: Timeout after {timeout_sec:g}s"
            return model, _review_query_error_payload(ctx=ctx, model=model, messages=messages, slot_id=slot_id, error=error, slot=slot), None
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
                                     row_plan: dict = None):
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
    row_ids = _row_vector("slot_ids", lambda idx: _rev().slot_id_for_row(idx + 1))
    any_api_rows = any(route is ReviewRouteKind.API_CHAT for route in row_routes[:len(models)])
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

    bible_text = _rev().load_governance_doc(_REPO_ROOT, "BIBLE.md", on_missing="explicit")
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
    llm_client = _rev().LLMClient()
    tasks = [
        _query_model(llm_client, m, messages, semaphore, ctx, slot_id=row_ids[idx],
                     route=row_routes[idx], session_task=session_task, session_root=session_root,
                     effort=row_efforts[idx], session_target=row_targets[idx],
                     session_profile=row_profiles[idx])
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


def _parse_model_response(model: str, result, headers_dict) -> dict:
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    resolved_model = str(usage.get("resolved_model") or model)
    provider = str(usage.get("provider") or "openrouter")
    # Row identity travels with the envelope on EVERY branch — success, transport
    # error, malformed body — so no consumer has to guess it back from position.
    slot_id = str(result.get("slot_id") or "") if isinstance(result, dict) else ""
    if isinstance(result, str) or (isinstance(result, dict) and result.get("error")):
        return {
            "model": resolved_model, "request_model": model,
            "provider": provider, "verdict": "ERROR",
            "text": result if isinstance(result, str) else str(result.get("error") or ""),
            "tokens_in": 0, "tokens_out": 0, "cost_estimate": None,
            "slot_id": slot_id,
            "prompt_ref": result.get("prompt_ref", {}) if isinstance(result, dict) else {},
            "response_ref": result.get("response_ref", {}) if isinstance(result, dict) else {},
        }
    try:
        choices = result.get("choices", [])
        if not choices:
            # Preserve full response body; no bare hardcoded truncation.
            text = (
                "(no choices in response: "
                f"{truncate_review_artifact(json.dumps(result), limit=4000)})"
            )
            verdict = "ERROR"
        else:
            text = choices[0]["message"]["content"]
            verdict = "UNKNOWN"
            for line in text.split("\n")[:3]:
                line_upper = line.upper()
                if "PASS" in line_upper:
                    verdict = "PASS"
                    break
                elif "CONCERNS" in line_upper:
                    verdict = "CONCERNS"
                    break
                elif "FAIL" in line_upper:
                    verdict = "FAIL"
                    break
    except (KeyError, IndexError, TypeError):
        text = (
            "(unexpected response format: "
            f"{truncate_review_artifact(json.dumps(result), limit=4000)})"
        )
        verdict = "ERROR"

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    cache_write_tokens = usage.get("cache_write_tokens", 0)
    prompt_cache_ttl = str(usage.get("prompt_cache_ttl") or "")

    cost = None
    try:
        if "cost" in usage:
            cost = float(usage["cost"])
        elif "total_cost" in usage:
            cost = float(usage["total_cost"])
        elif headers_dict:
            for key, value in headers_dict.items():
                if key.lower() == "x-openrouter-cost":
                    cost = float(value)
                    break
    except (ValueError, TypeError, KeyError):
        pass

    return {
        "model": resolved_model, "request_model": model,
        "provider": provider, "verdict": verdict, "text": text,
        "tokens_in": prompt_tokens, "tokens_out": completion_tokens,
        "cached_tokens": cached_tokens, "cache_write_tokens": cache_write_tokens,
        "prompt_cache_ttl": prompt_cache_ttl,
        "cost_estimate": cost,
        "slot_id": slot_id,
        "prompt_ref": result.get("prompt_ref", {}) if isinstance(result, dict) else {},
        "response_ref": result.get("response_ref", {}) if isinstance(result, dict) else {},
    }
