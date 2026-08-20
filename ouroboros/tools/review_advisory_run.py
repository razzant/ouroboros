"""The advisory run itself: route/slot resolution and gate availability, the
delegated Claudexor transport, the SDK budget bound, the read-only advisory
call, and the advisory output parsers with the light-model extraction
fallback. Extracted from ouroboros/tools/claude_advisory_review.py (v7 L-C
split); claude_advisory_review.py re-exports every name."""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import List, Optional

from ouroboros.skill_review_status import SEVERITY_DRIVEN_ITEMS
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_helpers import (
    emit_review_event,
    format_advisory_sdk_error as _format_advisory_error,
    get_advisory_runtime_diagnostics as _get_runtime_diagnostics,
)
from ouroboros.triad_review import (
    empty_array_is_verified_clean,
    extract_json_array,
)

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.tools.claude_advisory_review")


def _car():
    """The parent advisory module, read at call time.

    The advisory's members stay monkeypatch-addressable at their historical
    ``ouroboros.tools.claude_advisory_review`` bindings (tests rebind them
    there, including the plain ``adv_mod._ADVISORY_PROMPT_MAX_CHARS = ...``
    assignment), so this leaf resolves every such cross-reference through the
    module at each call instead of freezing whatever object a from-import
    saw at import time.
    """
    from ouroboros.tools import claude_advisory_review

    return claude_advisory_review


_ADVISORY_PROMPT_MAX_CHARS = 1_600_000  # ~400K tokens; non-blocking skip when exceeded


# The advisory's own output contract, handed to the shared extraction SSOT so one
# mechanism canonicalizes every review surface while each keeps its own contract.
_ADVISORY_EXTRACT_CONTRACT = (
    "A JSON array of checklist entries. Each element MUST have ALL of: "
    '"item" (checklist item name), "verdict" ("PASS" or "FAIL"), "severity" '
    '("critical" or "advisory" — REQUIRED even for PASS entries), "reason" (brief '
    'explanation). Optional: "obligation_id" (stable id of a previously surfaced '
    "obligation). If a FAIL entry in the source omits severity, infer it from "
    'context: "critical" for bugs, security or constitutional violations, else '
    '"advisory". If the text carries no valid checklist array, return [].'
)


def _resolve_fallback_model() -> str:
    """Resolve the configured light model for advisory extraction fallback. Uses the
    role-model accessor so an empty Light slot falls back to Main (v6.39) instead of
    yielding "" and calling the LLM with an empty model id."""
    from ouroboros.config import get_light_model
    return get_light_model()


def _llm_extract_advisory_items(raw_text: str, ctx: object) -> list:
    """Extract checklist items from narrative advisory output.

    Extraction is the SHARED SSOT (``review_execution.canonicalize_session_verdict``)
    reading the WHOLE artifact, with the advisory's own output contract. It used to
    read a 4K head + 60K tail window: a critical raised in the MIDDLE of a long
    advisory was silently dropped, and because entries may carry ``obligation_id``, a
    surviving advisory row could even close an obligation whose critical had just been
    cut away. An artifact too large for the one-send extraction rail is now the typed
    ``extraction_incomplete`` refusal — never a verdict fabricated from a visible cut.
    """
    try:
        from ouroboros.review_execution import canonicalize_session_verdict

        light_model = _resolve_fallback_model()
        content, method, fallback_usage = canonicalize_session_verdict(
            raw_text,
            # The advisory transport reports no structured-output conformance here, so
            # the trusted-schema branch is never taken on this path.
            conformance_passed=False,
            contract=_ADVISORY_EXTRACT_CONTRACT,
        )
        if method == "extraction_incomplete":
            log.warning(
                "Advisory extraction refused: artifact (%d chars) exceeds the single-send "
                "extraction bound; reporting no items rather than a windowed guess.",
                len(str(raw_text or "")),
            )
            return []

        # Track fallback LLM cost; it is real review spend.
        if fallback_usage and isinstance(ctx, ToolContext):
            fallback_raw_cost = (fallback_usage or {}).get("cost")
            fallback_cost = float(fallback_raw_cost) if fallback_raw_cost is not None else None
            from ouroboros.pricing import infer_provider_from_model as _infer_prov
            _car().emit_review_usage(
                ctx,
                model=light_model,
                cost_usd=fallback_cost,
                usage=fallback_usage,
                source="advisory_fallback",
                provider=_infer_prov(light_model),
            )

        # The SSOT already flattened provider content blocks to text; the advisory's
        # OWN contract post-processing (below) is unchanged and stays here.
        items = _parse_advisory_output(str(content or ""))
        if not _is_checklist_array(items):
            return []

        # Missing FAIL severity defaults to critical; never silently downgrade.
        normalised = []
        for it in items:
            if not isinstance(it, dict):
                continue
            verdict = str(it.get("verdict", "")).upper().strip()
            if verdict == "FAIL" and not str(it.get("severity", "")).strip():
                it = dict(it)
                it["severity"] = "critical"
            normalised.append(it)
        return normalised

    except Exception as exc:
        log.warning("Advisory LLM fallback extraction failed: %s", exc)
        return []


def _check_expected_items(items: list, expected_items: Optional[List[str]]) -> tuple[str, str]:
    """Return contract error/warning for checklist coverage mismatches."""
    if not expected_items:
        return "", ""
    expected = [str(item) for item in expected_items]
    actual = [
        str(item.get("item") or "")
        for item in items
        if isinstance(item, dict)
    ]
    # Severity-driven checklist items (bug_hunting, companion_process_safety,
    # extension_namespace_discipline, widget_module_safety) legitimately emit one
    # row per distinct issue, so collapse their repeated rows to a single
    # occurrence BEFORE the contract comparison. Single-row items keep their
    # multiplicity, so a genuine duplicate of e.g. permissions_honesty still warns.
    # Without this, a valid multi-bug advisory falsely triggered duplicates=/count=
    # contract warnings and got marked advisory_sdk_suspect_result.
    collapsed: List[str] = []
    seen_severity: set[str] = set()
    for item in actual:
        if item in SEVERITY_DRIVEN_ITEMS:
            if item in seen_severity:
                continue
            seen_severity.add(item)
        collapsed.append(item)
    actual = collapsed
    if actual == expected:
        return "", ""
    missing = [item for item in expected if item not in actual]
    extras = [item for item in actual if item not in expected]
    duplicate_count = len(actual) - len(set(actual))
    error_parts = []
    warning_parts = []
    if missing:
        error_parts.append(f"missing={missing}")
    if extras:
        error_parts.append(f"unexpected={extras}")
    if duplicate_count:
        warning_parts.append(f"duplicates={duplicate_count}")
    if len(actual) != len(expected):
        target = error_parts if (missing or extras) else warning_parts
        target.append(f"count={len(actual)} expected={len(expected)}")
    if not error_parts and not warning_parts:
        warning_parts.append("order differs from expected contract")
    prefix = "Skill advisory checklist contract mismatch: "
    return (
        (prefix + "; ".join(error_parts)) if error_parts else "",
        (prefix + "; ".join(warning_parts)) if warning_parts else "",
    )


ADVISORY_REVIEW_ROUTE_ENV = "OUROBOROS_ADVISORY_REVIEW_ROUTE"
_ADVISORY_SESSION_MAX_SECONDS = 900  # the nanny's time cap replaces the SDK budget kill


def advisory_review_route() -> str:
    """The advisory delivery route: ``api`` (Claude Agent SDK, needs the key)
    or ``agent_session`` (a delegated Claudexor run, needs no key). An unknown
    token raises — a typo must fail loudly, never silently pick a transport.

    Reads the reviewer-slot SSOT (6.1): the structured advisory row when the
    owner saved one, the legacy ``OUROBOROS_ADVISORY_REVIEW_ROUTE`` env
    otherwise (the SSOT's own migration read)."""
    from ouroboros.reviewer_slot_config import advisory_slot_config

    return "api" if advisory_slot_config().kind == "api" else "agent_session"


def advisory_slot_enabled() -> bool:
    """Whether the ONE optional advisory reviewer is enabled (D14).

    ``False`` is a standing owner decision whose constitutional consequence is
    an AUDITED BYPASS on every reviewed commit — recorded by the pre-commit
    gate, never a silent skip."""
    from ouroboros.reviewer_slot_config import advisory_slot_config

    return bool(advisory_slot_config().enabled)


def advisory_route_requires_api_key() -> bool:
    """Whether THIS advisory route needs ANTHROPIC_API_KEY (plan 5.8: the four
    key checks are route-dependent — an api route requires the key exactly as
    before; the delegated route runs without it)."""
    return advisory_review_route() == "api"


def advisory_gate_unavailability_reason() -> str | None:
    """Why the advisory cannot run, or ``None`` when it is available.

    This is the canonical diagnostic projection of the same structured facts
    used by the commit gate: owner-disabled slot, keyless ``api`` route, or an
    ``agent_session`` route with neither a parseable advisory target nor a
    shared review/subagent route (mirroring
    ``run_delegated_review_session``, which refuses that exact state with
    ``ReviewRouteUnavailable``). Reasons are stable and safe to expose. Raises
    ``ValueError`` on malformed slot/route configuration so each caller retains
    authority over its own fail direction.
    """
    if not advisory_slot_enabled():
        return "advisory_slot_disabled"
    if advisory_route_requires_api_key():
        return None if os.environ.get("ANTHROPIC_API_KEY", "") else "anthropic_api_key_missing"
    # Delegated route: mirror the runner's resolution order — the slot's own
    # target when it parses, else the shared session route; None there is a
    # typed refusal at run time, so None here is UNAVAILABLE at gate time.
    from ouroboros.review_execution import review_session_route
    from ouroboros.reviewer_slot_config import advisory_slot_config
    from ouroboros.subagents import parse_subagent_harness

    _target = str(advisory_slot_config().target_id or "")
    if _target and parse_subagent_harness(_target) is not None:
        return None
    return "agent_session_route_unavailable" if review_session_route() is None else None


def advisory_gate_unavailable() -> bool:
    """Whether the commit gate must use advisory-bypass compensation (#123).

    The boolean is intentionally only a projection of the canonical reason so
    diagnostics and gate behavior cannot drift. Malformed configuration keeps
    the reason helper's ``ValueError`` authority unchanged.
    """
    return _car().advisory_gate_unavailability_reason() is not None


def _run_advisory_delegated(prompt: str, repo_dir: pathlib.Path, ctx: ToolContext):
    """The advisory as a delegated Claudexor session, rehydrated into the same
    result structure the SDK path produces (5.8: only the transport changes).

    Runs through the ONE shared delegated-session runner (no second nanny
    loop). The SDK-side budget kill is lost by construction; the runner's time
    cap is the nanny-enforced bound. The narrative fallback is unchanged: the
    existing advisory extractor already canonicalizes non-JSON output (D19).
    Cost: the run settles through delegate_custody (the subscription-session
    ledger row); ``cost_usd`` stays 0.0 here so the SDK-path usage emit cannot
    double-count, and the disclosed spend rides ``usage`` for forensics."""
    from types import SimpleNamespace

    from ouroboros.delegate_custody import custody_root
    from ouroboros.review_execution import (
        SessionInvocation,
        review_session_output_schema,
        run_delegated_review_session,
    )

    try:
        # The advisory row's own target/effort (6.1); None keeps the shared
        # session-route fallback inside the runner.
        import dataclasses as _dc

        from ouroboros.reviewer_slot_config import advisory_slot_config
        from ouroboros.subagents import parse_subagent_harness

        _slot = advisory_slot_config()
        _session_route = parse_subagent_harness(_slot.target_id) if _slot.target_id else None
        # D1/6.3: the effort field is the ONE source; any effort embedded in the
        # target identity is dropped so it can never override the field.
        if _session_route is not None:
            _session_route = _dc.replace(_session_route, effort=str(_slot.effort or ""))
        if _session_route is not None and getattr(_slot, "profile_id", ""):
            _session_route = _dc.replace(_session_route, profile_id=_slot.profile_id)
        drive = custody_root(ctx) if getattr(ctx, "drive_root", None) else pathlib.Path(repo_dir)
        facts = run_delegated_review_session(
            prompt=prompt,
            root=str(repo_dir),
            custody_drive=drive,
            invocation=SessionInvocation(
                task_id=str(getattr(ctx, "task_id", "") or ""),
                surface="advisory_review",
                slot_id="advisory_slot_1",
                timeout_sec=_ADVISORY_SESSION_MAX_SECONDS,
                # The owner's configured advisory slot route (6.1 SSOT) rides the
                # invocation — the one identity+delivery value — not a parallel kwarg.
                session_route=_session_route,
                # The structured verdict is ASKED here exactly as the substrate's
                # session slots ask for it (D19): a review surface that never asks can
                # only reach its verdict through extraction, paying a light-model call
                # and a capability delta for what the route may support natively.
                output_schema=review_session_output_schema("advisory_review"),
            ),
        )
    except Exception as exc:
        return SimpleNamespace(
            success=False, result_text="(no output)", session_id="", cost_usd=0.0,
            usage={}, error=f"{type(exc).__name__}: {exc}", stderr_tail="",
        ), ""
    spend_final = facts["spend"] if (facts["spend"] is not None and not facts["spend_estimated"]) else None
    result_text = str(facts["text"] or "")
    if facts.get("conformance") == "passed":
        # A schema-conformant session answers with the SESSION envelope
        # ({"findings": [...]}) while every advisory consumer downstream — the
        # strict parser, the clean-verdict sentinel, the fallback gate — reads the
        # advisory's own ARRAY contract. Unwrap the trusted envelope here (D19's
        # schema-first ordering), so a clean {"findings": []} lands as the bare
        # "[]" the contract calls clean instead of as a paid extraction and a
        # parse_failure. Non-conformant output keeps its narrative path unchanged.
        from ouroboros.review_execution import _findings_array

        try:
            payload = json.loads(result_text.strip())
        except (TypeError, ValueError):
            payload = None
        findings = _findings_array(payload)
        if findings is not None:
            result_text = "[]" if not findings else json.dumps(findings, ensure_ascii=False)
    return SimpleNamespace(
        success=True,
        result_text=result_text,
        session_id=facts["run_id"],
        cost_usd=0.0,  # settled by delegate_custody; never re-emitted here
        usage={
            "delegated_run_id": facts["run_id"],
            "delegated_route": facts["route_id"],
            "cost_disclosed_usd": facts["spend"],
            "cost_estimated": facts["spend_estimated"],
            "cost_final_usd": spend_final,
            "settlement": facts["settlement"],
            # The structured-verdict facts the substrate's slots also carry: whether
            # the schema was asked at all, what the run reported, and which route(s)
            # actually served it. Conformance is TRUSTED only on "passed" — never on
            # run success (D19).
            "schema_asked": bool(facts.get("schema_asked")),
            "output_conformance": facts.get("conformance") or "",
            "conformance_trusted": (facts.get("conformance") == "passed"),
            "effective_route_ids": list(facts.get("effective_route_ids") or []),
            "capability_delta": _advisory_session_deltas(facts),
        },
        error="",
        stderr_tail="",
    ), str(facts["model"] or facts["route_id"])


def _advisory_session_deltas(facts: dict) -> List[dict]:
    """The same three landings-below-the-ask the substrate discloses (D4).

    Same vocabulary as ``AgentSessionReviewExecutor``, so one disclosure contract
    covers every delegated review surface instead of two dialects."""
    route_id = str(facts.get("route_id") or "")
    conformance = str(facts.get("conformance") or "")
    deltas: List[dict] = []
    if not facts.get("schema_asked"):
        deltas.append({
            "kind": "capability_delta",
            "requested": "outputSchema (structured verdict)",
            "effective": f"no structured output on effective route {route_id}",
            "reason": "schema_unavailable_on_effective_route",
        })
    elif conformance != "passed":
        deltas.append({
            "kind": "capability_delta",
            "requested": "outputSchema (structured verdict)",
            "effective": f"outputConformance={conformance or 'absent'}",
            "reason": "schema_not_conformed_on_effective_route",
        })
    effective = [str(r) for r in (facts.get("effective_route_ids") or [])]
    if effective and set(effective) != {route_id}:
        deltas.append({
            "kind": "capability_delta",
            "requested": f"route {route_id} (pinned pool)",
            "effective": "route(s) " + ", ".join(effective),
            "reason": "session_ran_off_pinned_route",
        })
    return deltas


def _advisory_sdk_budget(ctx: ToolContext, active_scope, drive_root, repo_dir) -> Optional[float]:
    """Remaining budget headroom for the SDK route's hard kill (api route only;
    the delegated route's bound is the nanny's time cap)."""
    from ouroboros.usage_accounting import usage_projection

    budget_root = pathlib.Path(
        drive_root
        or getattr(ctx, "budget_drive_root", "")
        or getattr(active_scope, "drive_root", "")
        or getattr(ctx, "drive_root", "") or repo_dir
    )
    root_id = str(
        (getattr(ctx, "task_metadata", {}) or {}).get("root_task_id")
        or getattr(active_scope, "root_task_id", "")
        or getattr(ctx, "task_id", "")
        or ""
    )
    caps: List[float] = []
    global_limit = getattr(active_scope, "global_limit_usd", None)
    root_limit = getattr(active_scope, "root_limit_usd", None)
    if global_limit is not None:
        global_projection = usage_projection(budget_root, global_limit_usd=float(global_limit))
        caps.append(max(0.0, float(global_limit) - float(global_projection.get("accounted_usd") or 0.0)))
    if root_id and root_limit is not None:
        root_projection = usage_projection(budget_root, root_task_id=root_id)
        caps.append(max(0.0, float(root_limit) - float(root_projection.get("accounted_usd") or 0.0)))
    return min(caps) if caps else None


def _note_meta_error(ctx: ToolContext, meta: dict, err_msg: str) -> None:
    """Record an advisory failure on the ctx meta snapshot (best-effort)."""
    try:
        meta["status"] = "error"
        meta["error"] = err_msg
        setattr(ctx, "_last_claude_advisory_meta", dict(meta))
    except Exception:
        pass


def _run_claude_advisory(
    repo_dir: pathlib.Path,
    commit_message: str,
    ctx: ToolContext,
    goal: str = "",
    scope: str = "",
    paths: Optional[List[str]] = None,
    options: Optional[dict] = None,
) -> tuple:
    """Run read-only advisory review; raw_result starts with ADVISORY_ERROR on failure."""
    try:
        delegated_route = advisory_review_route() == "agent_session"
    except ValueError as exc:
        return [], f"⚠️ ADVISORY_ERROR: {exc}", "", 0
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    # Route-dependent (plan 5.8 site 1): the api route requires the key exactly
    # as before; the delegated route runs on the subscription and needs none.
    if not api_key and not delegated_route:
        return [], "⚠️ ADVISORY_ERROR: ANTHROPIC_API_KEY not set (advisory route=api).", "", 0

    if delegated_route:
        model = ""  # the session route resolves its own model; reported after the run
        _slot = None
    else:
        from ouroboros.gateways.claude_code import resolve_claude_code_model
        from ouroboros.reviewer_slot_config import advisory_slot_config

        # The advisory row's own target applies on the api kind too (6.1): here
        # target_id is a Claude-SDK model spelling (sonnet, opus[1m], claude-…),
        # NOT an OpenRouter catalog id; '' keeps today's environment default.
        _slot = advisory_slot_config()
        model = (_slot.target_id or "").strip() or resolve_claude_code_model()
    options = dict(options or {})
    drive_root = options.get("drive_root")
    include_repo_diff = bool(options.get("include_repo_diff", True))
    review_surface = str(options.get("review_surface") or "repo")
    expected_items = options.get("expected_items")
    try:
        setattr(ctx, "_last_claude_advisory_meta", {})
    except Exception:
        pass

    try:
        if include_repo_diff:
            diff_text = _car()._get_staged_diff(repo_dir, paths=paths)
            if diff_text.startswith("⚠️ ADVISORY_ERROR:"):
                return [], diff_text, "", 0
            changed_files_text = _car()._get_changed_file_list(repo_dir, paths=paths)
            if changed_files_text.startswith("⚠️ ADVISORY_ERROR:"):
                return [], changed_files_text, "", 0
            resolved_paths, touched_pack, omitted_paths = _car().build_advisory_changed_context(
                repo_dir,
                changed_files_text=changed_files_text,
                paths=paths,
                exclude_paths={"docs/ARCHITECTURE.md"},
            )
            preflight_err = _car()._syntax_preflight_staged_py_files(repo_dir, resolved_paths)
            if preflight_err:
                log.warning("Advisory skipped — syntax preflight blocked: %s", preflight_err.splitlines()[0])
                return [], preflight_err, "", 0
        else:
            diff_text = "(not included; this advisory review is scoped to the supplied payload pack)"
            changed_files_text = "(not included; this advisory review is scoped to the supplied payload pack)"
            resolved_paths, touched_pack, omitted_paths = [], "", []

        prompt = _car()._build_advisory_prompt(
            repo_dir,
            commit_message,
            goal=goal,
            scope=scope,
            resolved_paths=resolved_paths,
            drive_root=drive_root,
            prompt_context={
                "diff": diff_text,
                "changed_files": changed_files_text,
                "touched_pack": touched_pack,
                "omitted_paths": omitted_paths,
                "review_surface": review_surface,
                "expected_items": expected_items,
            },
        )
    except RuntimeError as exc:
        return [], f"⚠️ ADVISORY_ERROR: failed to build advisory prompt: {exc}", "", 0
    except Exception as exc:
        return [], f"⚠️ ADVISORY_ERROR: unexpected error building prompt: {exc}", "", 0

    prompt_chars = len(prompt)
    diag = _get_runtime_diagnostics(model, prompt_chars, resolved_paths)

    if prompt_chars > _car()._ADVISORY_PROMPT_MAX_CHARS:
        tokens_approx = max(1, prompt_chars // 4)
        warning = (
            f"⚠️ ADVISORY_SKIPPED: advisory prompt too large "
            f"({prompt_chars:,} chars, ~{tokens_approx:,} tokens > "
            f"{_car()._ADVISORY_PROMPT_MAX_CHARS:,} char limit). "
            f"Advisory review skipped — non-blocking. Consider splitting the commit."
        )
        log.warning("Advisory skipped — prompt too large: %d chars", prompt_chars)
        return [], warning, model, prompt_chars

    log.info(
        "Advisory SDK call: model=%s prompt_chars=%d touched=%s sdk=%s cli=%s",
        diag["model"], diag["prompt_chars"], diag["touched_paths"],
        diag["sdk_version"], diag["cli_version"],
    )

    try:
        if delegated_route:
            # 5.8: only the transport changes — the delegated session runs the
            # SAME advisory prompt in the same repo root and rehydrates the same
            # result structure. The SDK budget kill is replaced by the runner's
            # nanny-enforced time cap; cost settles through delegate_custody.
            scope_effort = ""  # the session route carries its own effort
            result, model = _run_advisory_delegated(prompt, repo_dir, ctx)
        else:
            from ouroboros.gateways.claude_code import (
                DEFAULT_CLAUDE_CODE_MAX_TURNS,
                run_readonly,
            )
            from ouroboros.config import resolve_effort
            from ouroboros.usage_accounting import current_usage_scope

            # D-5b fix: the api route runs at the ADVISORY row's own effort, the
            # same field the delegated branch already honors — never the scope
            # reviewer's. The parser guarantees a non-empty effort ("low"
            # default, legacy config included), so the fallback is dead but honest.
            scope_effort = _slot.effort or resolve_effort("scope_review")
            active_scope = current_usage_scope()
            max_budget_usd = options.get("max_budget_usd")
            if max_budget_usd is None:
                max_budget_usd = _advisory_sdk_budget(ctx, active_scope, drive_root, repo_dir)
            if active_scope is not None:
                from dataclasses import replace
                from ouroboros.usage_accounting import usage_scope

                with usage_scope(replace(
                    active_scope, category="advisory_review", source="claude_advisory_review",
                )):
                    result = run_readonly(
                        prompt=prompt, cwd=str(repo_dir), model=model,
                        max_turns=DEFAULT_CLAUDE_CODE_MAX_TURNS,
                        effort=scope_effort, max_budget_usd=max_budget_usd,
                    )
            else:
                result = run_readonly(
                    prompt=prompt, cwd=str(repo_dir), model=model,
                    max_turns=DEFAULT_CLAUDE_CODE_MAX_TURNS,
                    effort=scope_effort, max_budget_usd=max_budget_usd,
                )

        meta = {
            "model": model,
            "session_id": getattr(result, "session_id", "") or "",
            "prompt_chars": prompt_chars,
            "cost_usd": float(getattr(result, "cost_usd", 0) or 0),
            "usage": getattr(result, "usage", {}) or {},
            "review_surface": review_surface,
            "effort": scope_effort,
            "status": "completed" if getattr(result, "success", False) else "error",
        }
        try:
            setattr(ctx, "_last_claude_advisory_meta", dict(meta))
        except Exception:
            pass

        if not result.success:
            err_msg = _format_advisory_error(
                prefix="SDK/CLI returned failure",
                result_error=result.error,
                stderr_tail=result.stderr_tail,
                session_id=result.session_id,
                diag=diag,
            )
            log.error("Advisory SDK failure:\n%s", err_msg)
            _note_meta_error(ctx, meta, err_msg)
            return [], err_msg, model, prompt_chars

        raw_text = str(result.result_text or "")

        if result.cost_usd > 0:
            _car().emit_review_usage(
                ctx,
                model=model,
                cost_usd=result.cost_usd,
                usage=result.usage or {},
                source="advisory_sdk",
                provider="anthropic",
                session_id=meta.get("session_id", ""),
                prompt_chars=prompt_chars,
            )

        prompt_tokens = int((result.usage or {}).get("prompt_tokens", 0) or 0)
        completion_tokens = int((result.usage or {}).get("completion_tokens", 0) or 0)
        cached_tokens = int((result.usage or {}).get("cached_tokens", 0) or 0)
        cache_write_tokens = int((result.usage or {}).get("cache_write_tokens", 0) or 0)
        if result.cost_usd > 0 and not any((
            prompt_tokens, completion_tokens, cached_tokens, cache_write_tokens,
        )):
            emit_review_event(ctx, {
                "type": "advisory_sdk_suspect_result",
                "model": model,
                "session_id": meta.get("session_id", ""),
                "prompt_chars": prompt_chars,
                "cost_usd": float(result.cost_usd or 0),
                "reason": "paid advisory SDK result had zero normalized token usage",
                "review_surface": review_surface,
            })

        if raw_text.strip() in {"", "(no output)"} and result.cost_usd > 0:
            err_msg = _format_advisory_error(
                prefix="SDK returned paid empty output",
                result_error="success=True but result_text was empty",
                stderr_tail=getattr(result, "stderr_tail", "") or "",
                session_id=meta.get("session_id", ""),
                diag=diag,
            )
            emit_review_event(ctx, {
                "type": "advisory_sdk_suspect_result",
                "model": model,
                "session_id": meta.get("session_id", ""),
                "prompt_chars": prompt_chars,
                "cost_usd": float(result.cost_usd or 0),
                "reason": "paid advisory SDK result had empty output",
                "review_surface": review_surface,
            })
            _note_meta_error(ctx, meta, err_msg)
            return [], err_msg, model, prompt_chars

        items = _parse_advisory_output(raw_text)

        if _needs_fallback_extraction(items, raw_text):
            items = _llm_extract_advisory_items(raw_text, ctx)
            if items:
                log.info("Advisory: structural parse failed, LLM fallback extracted %d items", len(items))

        contract_error, contract_warning = _check_expected_items(items, expected_items)
        if contract_error:
            err_msg = _format_advisory_error(
                prefix="SDK returned malformed checklist",
                result_error=contract_error,
                stderr_tail=getattr(result, "stderr_tail", "") or "",
                session_id=meta.get("session_id", ""),
                diag=diag,
            )
            emit_review_event(ctx, {
                "type": "advisory_sdk_suspect_result",
                "model": model,
                "session_id": meta.get("session_id", ""),
                "prompt_chars": prompt_chars,
                "cost_usd": float(result.cost_usd or 0),
                "reason": contract_error,
                "review_surface": review_surface,
            })
            _note_meta_error(ctx, meta, err_msg)
            return [], err_msg, model, prompt_chars

        if contract_warning:
            emit_review_event(ctx, {
                "type": "advisory_contract_warning",
                "model": model,
                "session_id": meta.get("session_id", ""),
                "prompt_chars": prompt_chars,
                "cost_usd": float(result.cost_usd or 0),
                "warning": contract_warning,
                "review_surface": review_surface,
            })
            try:
                meta["status"] = "completed_with_contract_warning"
                meta["contract_warning"] = contract_warning
                setattr(ctx, "_last_claude_advisory_meta", dict(meta))
            except Exception:
                pass

        return items, raw_text, model, prompt_chars

    except ImportError:
        return [], (
            "⚠️ ADVISORY_ERROR: claude-agent-sdk not installed. "
            "Install: pip install 'ouroboros[claude-sdk]'"
        ), "", 0
    except Exception as e:
        err_msg = _format_advisory_error(
            prefix=f"SDK call raised {type(e).__name__}",
            result_error=str(e),
            stderr_tail="",
            session_id="",
            diag=diag,
        )
        log.error("Advisory SDK exception:\n%s", err_msg)
        return [], err_msg, model, prompt_chars


def _is_clean_verdict(raw_text: str) -> bool:
    """Clean-verdict check on the SAME text shape ``_parse_advisory_output`` reads.

    That parser passes ``unwrap_result=True`` because the CLI may deliver the
    review inside a ``{"result": "..."}`` envelope; testing the wrapper instead
    of its payload would leave the clean verdict unrecognised exactly for the
    wrapped shape.
    """
    text = str(raw_text or "")
    try:
        envelope = json.loads(text.strip())
        if isinstance(envelope, dict) and "result" in envelope:
            text = str(envelope["result"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return empty_array_is_verified_clean(text)


def _needs_fallback_extraction(items: list, raw_text: str) -> bool:
    """True when paying the fallback extraction model can still yield items.

    A sentinel-qualified clean verdict (REVIEW_JSON_ARRAY_CONTRACT) parses to an
    empty list by design and has nothing to extract, so it must not be charged
    to the fallback model or later recorded as a parse failure.
    """
    return bool(
        not items
        and raw_text
        and not raw_text.startswith("⚠️ ADVISORY_ERROR")
        and not _is_clean_verdict(raw_text)
    )


def _parse_advisory_output(stdout: str) -> list:
    """Extract the JSON findings array from Claude CLI output."""
    return extract_json_array(
        stdout,
        unwrap_result=True,
        validate_fn=_is_checklist_array,
    ) or []


def _is_checklist_array(items: list) -> bool:
    """Return True iff items looks like a real advisory checklist array.

    Each element must be a dict containing at least 'item' and 'verdict' keys.
    An empty list is rejected (no findings = parse_failure, not a clean advisory).
    Stray arrays like [1,2,3], code snippets, or unrelated JSON lists are rejected.
    """
    if not items:
        return False
    return all(
        isinstance(el, dict) and "item" in el and "verdict" in el
        for el in items
    )
