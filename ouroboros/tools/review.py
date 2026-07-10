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
    truncate_review_artifact,
    utc_now_iso,
)
from ouroboros import config as _cfg
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.triad_review import extract_json_array, parse_model_review_results

log = logging.getLogger(__name__)

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
    if not raw:
        return DEFAULT_REVIEW_MODEL_TIMEOUT_SEC
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning(
            "Invalid OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC=%r; using %.0fs",
            raw,
            DEFAULT_REVIEW_MODEL_TIMEOUT_SEC,
        )
        return DEFAULT_REVIEW_MODEL_TIMEOUT_SEC
    if value <= 0:
        log.warning(
            "Non-positive OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC=%r; using %.0fs",
            raw,
            DEFAULT_REVIEW_MODEL_TIMEOUT_SEC,
        )
        return DEFAULT_REVIEW_MODEL_TIMEOUT_SEC
    return value


def _format_timeout_seconds(timeout_sec: float) -> str:
    if float(timeout_sec).is_integer():
        return str(int(timeout_sec))
    return f"{timeout_sec:g}"


from ouroboros.tools.review_helpers import (
    REPO_ROOT as _REPO_ROOT,
    load_checklist_section as _load_checklist_section_precise,
    load_governance_doc,
    build_touched_file_pack,
    build_goal_section,
    review_drive_root,
    build_rebuttal_section,
    CRITICAL_FINDING_CALIBRATION,
    REPO_ANTI_PATTERN_LOCK_GUARD,
    REVIEW_JSON_ARRAY_CONTRACT,
    REVIEW_PREAMBLE,
    build_self_verification_template,
    build_review_history_section as _build_review_history_section,
    emit_review_usage,
    format_name_status_for_preflight,
    format_review_history_entry as _format_review_entry,
    single_line as _single_line,
)


# Derived alias; ``review_helpers.REPO_ROOT`` remains the repo-root SSOT.
_CHECKLISTS_PATH = _REPO_ROOT / "docs" / "CHECKLISTS.md"


# Tool: task_acceptance_review.

def get_tools():
    return [
        ToolEntry(
            name="task_acceptance_review",
            schema={
                "name": "task_acceptance_review",
                "description": (
                    "Run independent reviewer slots over a task-result claim and evidence packet. "
                    "Verdicts are advisory; if findings are valid, continue fixing or reject them with evidence."
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
            timeout_sec=900,
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
    from ouroboros.config import resolve_effort
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.review_substrate import (
        ReviewRequest,
        build_improvement_capsule,
        dissent_findings,
        reviewer_slots,
        run_review_request,
    )

    # v6.51.0 idea-2: build the process-aware evidence packet (full contract +
    # first-class verification_summary + host-collected redacted repo_diff + leak-safe
    # artifacts + provenance tags). The agent-tool (auto) path has no host-owned turn
    # trace, so there is no tool_trajectory and include_recent_commit stays False (it
    # cannot prove a commit happened THIS turn). The agent's own evidence is preserved
    # under `agent_supplied` (its repo_diff demoted to agent_supplied_repo_diff) — never
    # promoted to host-fact status; repo_diff is ALWAYS the HOST-collected structural fact.
    agent_evidence = dict(evidence or {})
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

    request = ReviewRequest(
        surface="task_acceptance",
        goal=goal,
        subject=claim,
        evidence=evidence,
        checklist=checklist,
        # v6.60.0: dead `verdict_is_advisory` removed — enforcement is the single
        # OUROBOROS_REVIEW_ENFORCEMENT setting (advisory|blocking) via obligations.
        policy={
            "raw_output_must_be_preserved": True,
            # min_successful_slots is set below from adaptive_quorum(len(slots)) —
            # the SSOT — once the actual reviewer slot count is known.
            "fail_closed_on_errors": True,
            "classify_outcome_tier": True,
        },
        task_id=str(getattr(ctx, "task_id", "") or ""),
    )
    slots = reviewer_slots(effort=resolve_effort("review"), role_hint="task acceptance")
    request.policy["min_successful_slots"] = _cfg.adaptive_quorum(len(slots))
    result = run_review_request(request, slots=slots, drive_root=pathlib.Path(ctx.drive_root), usage_ctx=ctx)
    # Agent self-call (auto): lead with the compact improvement capsule (the
    # actionable feedback) and keep the full structured result available for the
    # agent that explicitly asked for detail.
    capsule = build_improvement_capsule(result)
    payload_dict = dict(result.__dict__)
    # v6.54.4: DISSENT is recorded on EVERY path — the agent-called flow marks it
    # in the payload so the tool-result capture lands acceptance_decision.dissent_noted
    # (review round 2: previously only the host-forced path recorded it).
    payload_dict["dissent_noted"] = bool(dissent_findings(result))
    if agent_decision:
        payload_dict["agent_decision"] = agent_decision
    payload = json.dumps(payload_dict, ensure_ascii=False, indent=2, default=str)
    return f"{capsule}\n\n<full_review>\n{payload}\n</full_review>" if capsule else payload


def _handle_multi_model_review(ctx: ToolContext, content: str = "",
                                prompt: str = "", models: list = None) -> str:
    if models is None:
        models = []
    try:
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run,
                    _multi_model_review_async(content, prompt, models, ctx),
                ).result()
        except RuntimeError:
            result = asyncio.run(_multi_model_review_async(content, prompt, models, ctx))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Multi-model review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


def _review_query_error_payload(
    *,
    ctx: Optional[ToolContext],
    model: str,
    messages: list,
    slot_id: str,
    error: str,
) -> dict:
    payload = {"error": error, "usage": {}, "prompt_ref": {}, "response_ref": {}}
    try:
        from ouroboros.observability import new_call_id, persist_call

        drive_root = review_drive_root(ctx)
        task_id = str(getattr(ctx, "task_id", "") or "multi_model_review") if ctx is not None else "multi_model_review"
        call_id = new_call_id(f"review_multi_model_review_{slot_id}_error")
        payload["prompt_ref"] = persist_call(
            drive_root,
            task_id=task_id,
            call_id=f"{call_id}_prompt",
            call_type="multi_model_review_prompt",
            payload={"messages": messages, "slot_id": slot_id, "model": model},
            manifest={"surface": "multi_model_review", "slot_id": slot_id, "model": model, "synthetic": True},
        )
        payload["response_ref"] = persist_call(
            drive_root,
            task_id=task_id,
            call_id=f"{call_id}_error",
            call_type="multi_model_review_error",
            payload={"error": error},
            manifest={"surface": "multi_model_review", "slot_id": slot_id, "model": model, "status": "error", "synthetic": True},
        )
    except Exception:
        pass
    return payload


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
    slot_id: str = "multi_model_slot",
):
    async with semaphore:
        timeout_sec = _review_model_timeout_sec()
        try:
            from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

            _out_budget = _review_output_budget()
            request = ReviewRequest(
                surface="multi_model_review",
                goal="Run independent multi-model review over the supplied evidence.",
                messages=messages,
                task_id=str(getattr(ctx, "task_id", "") or "multi_model_review") if ctx is not None else "multi_model_review",
                call_type="multi_model_review",
                max_tokens=_out_budget,
                temperature=0.2,
                no_proxy=True,
            )
            slot = ReviewSlot(
                slot_id=slot_id,
                model=model,
                effort=_cfg.resolve_effort("review"),
                timeout_sec=timeout_sec,
                max_tokens=_out_budget,
                temperature=0.2,
                role_hint="multi-model review",
            )
            loop = asyncio.get_running_loop()
            run_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: run_review_request(
                        request,
                        slots=[slot],
                        drive_root=review_drive_root(ctx),
                        llm=llm_client,
                        usage_ctx=None,
                    ),
                ),
                timeout=timeout_sec,
            )
            actor = (run_result.actors or [{}])[0]
            if actor.get("status") not in {"ok", "empty"}:
                return model, {
                    "error": f"Error: {actor.get('error') or actor.get('status') or 'review failed'}",
                    "usage": actor.get("usage") or {},
                    "prompt_ref": actor.get("prompt_ref") or {},
                    "response_ref": actor.get("response_ref") or {},
                }, None
            payload = {
                "choices": [{"message": {"content": actor.get("raw_text") or ""}}],
                "usage": actor.get("usage") or {},
                "prompt_ref": actor.get("prompt_ref") or {},
                "response_ref": actor.get("response_ref") or {},
            }
            return model, payload, None
        except asyncio.TimeoutError:
            error = f"Error: Timeout after {_format_timeout_seconds(timeout_sec)}s"
            return model, _review_query_error_payload(ctx=ctx, model=model, messages=messages, slot_id=slot_id, error=error), None
        except Exception as e:
            # Preserve full review errors; helper adds an omission note if needed.
            error_msg = truncate_review_artifact(str(e), limit=4000)
            error = f"Error: {error_msg}"
            return model, _review_query_error_payload(ctx=ctx, model=model, messages=messages, slot_id=slot_id, error=error), None


async def _multi_model_review_async(content: str, prompt: str,
                                     models: list, ctx: ToolContext):
    if not content:
        return {"error": "content is required"}
    if not prompt:
        return {"error": "prompt is required"}
    if not models:
        return {"error": "models list is required"}
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        return {"error": "models must be a list of strings"}
    if len(models) > MAX_MODELS:
        return {"error": f"Too many models ({len(models)}). Maximum is {MAX_MODELS}."}

    bible_text = load_governance_doc(_REPO_ROOT, "BIBLE.md", on_missing="explicit")
    if bible_text:
        system_content = (
            _CONSTITUTIONAL_PREAMBLE
            + "### BIBLE.md (Full Text)\n\n" + bible_text
            + "\n\n---\n\n## REVIEW INSTRUCTIONS\n\n" + prompt
        )
    else:
        log.warning("Proceeding without BIBLE.md — constitutional compliance cannot be guaranteed")
        system_content = (
            _CONSTITUTIONAL_PREAMBLE
            + "(BIBLE.md could not be loaded)\n\n## REVIEW INSTRUCTIONS\n\n" + prompt
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": content},
    ]

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    llm_client = LLMClient()
    tasks = [
        _query_model(llm_client, m, messages, semaphore, ctx, slot_id=f"multi_model_slot_{idx + 1}")
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
                "cost": review_result.get("cost_estimate", 0.0),
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
    if isinstance(result, dict) and result.get("error"):
        return {
            "model": resolved_model, "request_model": model,
            "provider": provider, "verdict": "ERROR", "text": str(result.get("error") or ""),
            "tokens_in": 0, "tokens_out": 0, "cost_estimate": 0.0,
            "prompt_ref": result.get("prompt_ref", {}),
            "response_ref": result.get("response_ref", {}),
        }
    if isinstance(result, str):
        return {
            "model": resolved_model, "request_model": model,
            "provider": provider, "verdict": "ERROR", "text": result,
            "tokens_in": 0, "tokens_out": 0, "cost_estimate": 0.0,
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
        # Preserve full response body; no bare hardcoded truncation.
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

    cost = 0.0
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
        "prompt_ref": result.get("prompt_ref", {}) if isinstance(result, dict) else {},
        "response_ref": result.get("response_ref", {}) if isinstance(result, dict) else {},
    }


# Unified pre-commit review gate.

def _load_checklist_section() -> str:
    """Load Repo Commit Checklist, fail-closed if missing/malformed."""
    try:
        return _load_checklist_section_precise("Repo Commit Checklist")
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise FileNotFoundError(
            f"docs/CHECKLISTS.md not found or malformed: {e}"
        ) from e


_REVIEW_PROMPT_TEMPLATE = """\
{preamble}

## Review instructions

Read the staged diff and full current text of every changed file. Review every
checklist item, report every distinct current problem, and make every FAIL
actionable with file/symbol evidence and a concrete fix.

{critical_calibration}

{json_contract}

If an open obligation record above already names an `obligation_id` for this root cause,
reuse that exact `obligation_id`. Do NOT invent a new id when the same root cause persists.

## Anti pattern-lock guard

If your first reading surfaces exactly one FAIL, run the shared second pass guard focused on a different concern class:
{anti_pattern_lock_guard}

{checklist_section}

- Output ONLY a valid JSON array.  No markdown fences, no text outside the JSON.

{goal_section}

## DEVELOPMENT.md

{dev_guide_text}

## ARCHITECTURE.md

{architecture_section}

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
    """Fast deterministic review preflight for common incomplete staged diffs."""
    import re

    # Accept either name-status lines ("A  path") or plain filenames.
    import string as _string
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
    msg_lower = commit_message.lower()

    has_version_ref = bool(re.search(r'v?\d+\.\d+\.\d+', commit_message)) or "version" in msg_lower
    version_staged = "VERSION" in active_staged

    missing = []

    # VERSION staged but README missing.
    if version_staged and "README.md" not in active_staged:
        missing.append("README.md (badge + changelog)")

    # Commit message references version but VERSION is not staged.
    if has_version_ref and not version_staged:
        if any(f.endswith(('.py', '.md')) and f != 'VERSION' for f in active_staged):
            missing.append("VERSION")

    if missing:
        return (
            f"⚠️ PREFLIGHT_BLOCKED: Staged diff is incomplete — fix before review.\n"
            f"  Missing from staged: {', '.join(missing)}\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}\n\n"
            "Stage all related files together. Use write_file for all files first,\n"
            "then commit_reviewed to stage and commit everything in one diff."
        )

    # Python logic touched without active tests staged.
    _LOGIC_DIRS = ("ouroboros/", "supervisor/")
    logic_changed = any(
        f.startswith(_LOGIC_DIRS) and f.endswith(".py")
        for f in staged_set  # all statuses including D
    )
    tests_staged = any(f.startswith("tests/") for f in active_staged)
    if logic_changed and not tests_staged:
        return (
            "⚠️ PREFLIGHT_BLOCKED: Python logic changed in ouroboros/ or supervisor/ "
            "but no tests/ files are staged.\n"
            "  Add or update tests to cover the changed behaviour, then re-stage.\n"
            "  If this is a docs/config-only change that triggered a false positive, "
            "check that no .py files from ouroboros/ or supervisor/ are in your staged set.\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
        )

    # New logic modules require active ARCHITECTURE.md update.
    new_logic_files = [
        f for f in new_files
        if f.startswith(_LOGIC_DIRS) and f.endswith(".py")
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
                pyproject_text = _git_show_staged(repo_dir, "pyproject.toml")
                web_package_text = _git_show_staged(repo_dir, "web/package.json")
                readme_text = _git_show_staged(repo_dir, "README.md")
                arch_text = _git_show_staged(repo_dir, "docs/ARCHITECTURE.md")
                desync = version_carrier_desyncs(
                    version_str,
                    pyproject_text=pyproject_text,
                    web_package_text=web_package_text,
                    readme_text=readme_text,
                    arch_text=arch_text,
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

    # VERSION changes need a staged README changelog row.
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

    # VERSION changes must respect P9 README history limits in staged content.
    if version_staged:
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
    ctx._last_triad_raw_results = triad_raw_results
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

    self_verify_findings = list(getattr(ctx, '_last_review_critical_findings', []) or []) or list(critical_fails)
    retry_coaching = build_self_verification_template(
        self_verify_findings,
        attempt_idx=ctx._review_iteration_count,
        tool_name="commit_reviewed",
        context_noun="diff",
    )

    return (
        f"⚠️ REVIEW_BLOCKED{iteration_note}: Critical issues found by reviewers.\n"
        "Commit has NOT been created. Fix the issues and try again. Use review_rebuttal\n"
        "ONLY if a finding is factually incorrect — not to argue against requested tests\n"
        "or artifacts. If the same finding repeats after a rebuttal, implement the fix\n"
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


def _run_unified_review(ctx: ToolContext, commit_message: str,
                        review_rebuttal: str = "",
                        repo_dir=None,
                        goal: str = "",
                        scope: str = "") -> Optional[str]:
    """Run triad pre-commit review; return a block message or ``None``."""
    target_repo = repo_dir or ctx.repo_dir
    ctx._review_iteration_count += 1
    ctx._last_review_block_reason = ""  # reset per attempt
    ctx._last_triad_models = []  # reset forensic field so stale values never persist on early exit
    ctx._last_review_critical_findings = []  # reset to avoid stale findings from previous attempts
    ctx._last_triad_raw_results = []  # reset per-model actor records
    ctx._review_degraded_reasons = []  # reset degraded participation markers
    review_enforcement = _cfg.get_review_enforcement()
    blocking_review = review_enforcement == "blocking"

    try:
        diff_text = run_cmd(["git", "diff", "--cached"], cwd=target_repo)
    except Exception:
        diff_text = "(failed to get staged diff)"

    if not diff_text.strip():
        return None

    try:
        changed = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=target_repo)
    except Exception:
        changed = ""

    preflight_staged = _build_preflight_staged(target_repo, fallback=changed)

    preflight_err = _preflight_check(commit_message, preflight_staged, target_repo)
    if preflight_err:
        ctx._last_review_block_reason = "preflight"
        result = _handle_review_block_or_warning(
            ctx, blocking_review, preflight_err,
            "Review enforcement=Advisory: preflight warning did not block commit. ",
        )
        if result is not None:
            return result

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
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review checklist failed to load; commit proceeding anyway. ",
        )

    dev_guide_text = load_governance_doc(pathlib.Path(ctx.repo_dir), "docs/DEVELOPMENT.md", on_missing="explicit")
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

    # Build touched-file pack for full current context.
    try:
        touched_paths = [f.strip() for f in changed.strip().splitlines() if f.strip()]
        current_files_section, _omitted = build_touched_file_pack(
            pathlib.Path(target_repo), touched_paths
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

    goal_section = build_goal_section(goal, scope, commit_message)

    prompt = _REVIEW_PROMPT_TEMPLATE.format(
        preamble=REVIEW_PREAMBLE,
        critical_calibration=CRITICAL_FINDING_CALIBRATION,
        json_contract=REVIEW_JSON_ARRAY_CONTRACT,
        anti_pattern_lock_guard=REPO_ANTI_PATTERN_LOCK_GUARD,
        checklist_section=checklist_section,
        goal_section=goal_section,
        dev_guide_text=dev_guide_text or "(DEVELOPMENT.md not found)",
        architecture_section=architecture_text or "(ARCHITECTURE.md not found)",
        current_files_section=current_files_section,
        rebuttal_section=rebuttal_section,
        review_history_section=review_history_section,
        diff_text=diff_text,
        changed_files=changed,
    )

    models = _cfg.get_review_models()
    ctx._last_triad_models = list(models)  # forensic: actual resolved model IDs

    try:
        result_json = _handle_multi_model_review(
            ctx,
            content="Review the staged diff and context provided in the instructions above.",
            prompt=prompt,
            models=models,
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
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review returned no results from any model — "
            "commit cannot proceed without a successful review."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review returned no model results; commit proceeding anyway. ",
        )

    critical_fails, advisory_warns, errored_models, _triad_raw = _collect_review_findings(ctx, model_results)
    models_total = len(model_results)

    # Quorum counts only parseable responded actors, not errors/parse failures.
    triad_raw = getattr(ctx, "_last_triad_raw_results", []) or []
    successful_reviewers = sum(1 for r in triad_raw if r.get("status") == "responded")
    # Non-successful actors are shown for transport/parse diagnostics.
    failed_actors = [
        r["model_id"] for r in triad_raw if r.get("status") != "responded"
    ]
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

    if errored_note:
        advisory_warns.append(errored_note.strip())
    if advisory_warns or getattr(ctx, "_last_review_advisory_findings", None):
        ctx._review_advisory = list(getattr(ctx, "_last_review_advisory_findings", []) or [])
        if errored_note:
            ctx._review_advisory.append(errored_note.strip())
    return None
