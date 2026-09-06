"""Execution of the preflight (advisory) pre-review run and its result parsing.

Owns the run layer of the advisory organ on its native-episode form: the
route/slot vocabulary (``advisory_review_route``, ``advisory_slot_enabled``,
the gate-unavailability projection), the delegated agent-session executor on
the shared substrate seam, the dispatch orchestrator ``_run_claude_advisory``,
and the output contract — structural parse, clean-verdict recognition, the
LLM extraction fallback, and the expected-items contract check. Extracted from
ouroboros/tools/claude_advisory_review.py (v7 D06 split, re-derived on the
v7next tip: the reference leaf carried the retired Claude-SDK transport —
``advisory_route_requires_api_key`` / ``_advisory_session_deltas`` /
``_advisory_sdk_budget`` died with it and are not replayed);
claude_advisory_review.py re-exports every name. The leaf names follow the
organ's public rename (``preflight_review``, Q1). The native inspection
episode itself, the size/overflow gates, and the managed-subject diff resolver
are post-cutoff parent members: this leaf reads them through the call-time
handle, so their facade patch points keep working.
"""

from __future__ import annotations

import json
from functools import partial
import logging
import pathlib
from typing import List, Optional

from ouroboros.tools.registry import ToolContext

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.tools.claude_advisory_review")


def _car():
    """The parent claude-advisory-review module, read at call time.

    The advisory members stay monkeypatch-addressable at their historical
    ``ouroboros.tools.claude_advisory_review`` bindings (tests rebind them
    there), so this leaf resolves every such cross-reference through the
    module at each call instead of freezing whatever object a from-import saw
    at import time.
    """
    from ouroboros.tools import claude_advisory_review

    return claude_advisory_review


# EMERGENCY SANITY CEILING ONLY — never the honest fit gate. The api route's
# real admission bound is its route window from the reviewer-window SSOT
# (``reviewer_window.resolve_reviewer_window``; see ``_api_window_skip_warning``),
# and the agent_session route sends a compact pointer pack instead of inlined
# governance bodies. This constant survives purely as a backstop against a
# catastrophically mis-assembled prompt (~400K tokens).
_ADVISORY_PROMPT_MAX_CHARS = 1_600_000


# The advisory's own output contract, handed to the shared extraction SSOT so one
# mechanism canonicalizes every review surface while each keeps its own contract.
_ADVISORY_EXTRACT_CONTRACT = (
    "A JSON array of checklist entries. Each element MUST have ALL of: "
    '"item" (checklist item name), "verdict" ("PASS" or "FAIL"), "severity" '
    '("critical" or "advisory" — REQUIRED even for PASS entries), "reason" (brief '
    'explanation). Optional: "obligation_id" (stable id of a previously surfaced '
    "obligation). Preserve the reviewer's stated verdict and severity; do not "
    "infer severity from the alleged bug or change a verdict from withdrawal prose. "
    "If either cannot be faithfully recovered, return UNEXTRACTABLE."
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
            array_validator=_is_checklist_array,
            deadline_at=(getattr(ctx, "task_metadata", {}) or {}).get("deadline_at"),
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

        # Canonical enum spelling is shared with direct parsing. Missing or
        # unknown severity stays unparsed, never a host-authored critical.
        return _parse_advisory_output(str(content or ""))

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
    # contract warnings and got marked advisory_suspect_result.
    collapsed: List[str] = []
    seen_severity: set[str] = set()
    for item in actual:
        if item in _car().SEVERITY_DRIVEN_ITEMS:
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


_ADVISORY_SESSION_MAX_SECONDS = 900  # the nanny's time cap replaces the SDK budget kill


def advisory_review_route() -> str:
    """The advisory delivery kind on the shared closed vocabulary: ``api_chat``
    (the bounded NATIVE inspection episode on a routed model — the retired
    Claude-SDK transport's successor; advisory never receives an assembled
    packet) or ``agent_session`` (a delegated Claudexor run). An unknown token
    raises — a typo must fail loudly, never silently pick a transport.

    Reads the reviewer-slot SSOT (6.1): the structured advisory row when the
    owner saved one, the shipped default row otherwise (ABI-10: the legacy
    ``OUROBOROS_ADVISORY_REVIEW_ROUTE`` env is retired and ignored)."""
    from ouroboros.reviewer_slot_config import ROUTE_KIND_SESSION, advisory_slot_config

    return (
        "agent_session"
        if advisory_slot_config().kind == ROUTE_KIND_SESSION
        else "api_chat"
    )


def advisory_slot_enabled() -> bool:
    """Whether the ONE optional advisory reviewer is enabled (D14).

    ``False`` is a standing owner decision whose constitutional consequence is
    an AUDITED BYPASS on every reviewed commit — recorded by the pre-commit
    gate, never a silent skip."""
    from ouroboros.reviewer_slot_config import advisory_slot_config

    return bool(advisory_slot_config().enabled)


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
        # A migration force-disable is NOT a standing owner choice: surface
        # the parser's typed reason so the two states never conflate (a
        # legacy Claude-SDK target that could not be mapped reads as exactly
        # that, not as "the owner switched advisory off").
        from ouroboros.reviewer_slot_config import advisory_slot_config

        _reason = str(getattr(advisory_slot_config(), "disabled_reason", "") or "")
        return f"advisory_slot_disabled:{_reason}" if _reason else "advisory_slot_disabled"
    if advisory_review_route() == "api_chat":
        from ouroboros.provider_models import model_has_credentials

        return (
            None if model_has_credentials(_car()._advisory_native_model())
            else "advisory_model_credentials_missing"
        )
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


def pending_advisory_execution(ctx, commit_message, *, goal="", scope="", paths=None, review_rebuttal="",
                               skip_advisory_review=False, for_commit=False):
    """Read the existing preflight's custody before any candidate preparation.

    No prompt copy: delegate_custody.invocation_record owns its immutable
    request. Unknown/lost authority cannot authorize a replacement invocation.
    """
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.review_state import compute_snapshot_hash, make_repo_key, update_state
    from ouroboros.tools.git_review_cycle import _fingerprint_staged_diff

    repo_dir = pathlib.Path(ctx.repo_dir)
    intent = {"commit_message": commit_message, "goal": goal, "scope": scope,
              "review_rebuttal": review_rebuttal}
    def pending_records(state):
        from ouroboros.delegate_custody import custody_root, invocation_record

        records = []
        for run in state.filter_advisory_runs(repo_key=make_repo_key(repo_dir)):
            if not run.execution_pending:
                continue
            token = str(run.execution.get("pending_invocation_id") or "")
            stored = invocation_record(custody_root(ctx), token) if token else None
            if (stored and stored.get("state") == "failed_definite"
                    and stored.get("task_id") == run.task_id
                    and stored.get("surface") == "advisory_review"):
                # Only the existing definite producer discharges the checkpoint.
                # A missing row, old timestamp or missing run id proves nothing.
                run.execution.update(pending_invocation_id="", operation_state="settled")
                if run.status == "pending":
                    run.status = "error"
                continue
            if not for_commit or run.blocks_preflight:
                records.append(run)
        return records

    records = update_state(pathlib.Path(ctx.drive_root), pending_records)
    # No POST/rejoin is needed to record the explicit logical bypass. Its
    # existing writer must still persist the audit and retain these records.
    if skip_advisory_review:
        return {"intent": intent}, None
    if not records:
        return {"intent": intent}, None
    binding = _fingerprint_staged_diff(repo_dir)
    def binding_mismatches(run):
        bound_intent = run.execution.get("intent")
        return [name for name, differs in {
            "task_id": run.task_id != str(getattr(ctx, "task_id", "") or ""),
            "intent.fields": not isinstance(bound_intent, dict) or set(bound_intent) != set(intent),
            **{f"intent.{key}": not isinstance(bound_intent, dict) or bound_intent.get(key) != value
               for key, value in intent.items()},
            "snapshot_hash": compute_snapshot_hash(repo_dir, paths=run.snapshot_paths) != run.snapshot_hash,
            "staged_fingerprint": not binding.get("ok") or binding.get("fingerprint") != run.execution.get("fingerprint"),
        }.items() if differs]

    blocking = [run for run in records if run.blocks_preflight]
    # An explicit standalone call may rejoin its exact released invocation.
    # Other retained physical work is history, not admission for this request.
    records = blocking or [run for run in records
                           if run.execution.get("pending_invocation_id") and not binding_mismatches(run)]
    if not records:
        return {"intent": intent}, None
    run = records[0]
    execution = dict(run.execution)
    mismatches = binding_mismatches(run)
    if len(records) != 1:
        mismatches.append("pending_records")
    skip_hint = (" Use skip_advisory_review=true for an audited logical bypass; "
                 "existing physical work and its costs remain retained.")
    if mismatches:
        raise ReviewRouteUnavailable(
            "unresolved preflight binding differs or is unavailable: " + ", ".join(mismatches)
            + "; read review_status for the bound intent and execution. "
            "No preparation or replacement review was started." + skip_hint, code="advisory_pending_mismatch")
    if execution.get("pending_invocation_id") and advisory_review_route() != "agent_session":
        raise ReviewRouteUnavailable(
            "unresolved delegated preflight requires its agent_session delivery; the current advisory "
            "route kind differs. Restore that delivery kind to rejoin the recorded invocation; "
            "no replacement review was started." + skip_hint, code="advisory_pending_route_mismatch")
    if not execution.get("pending_invocation_id"):
        raise ReviewRouteUnavailable(
            "preflight physical outcome remains unknown without recoverable delegated custody." + skip_hint,
            code="advisory_custody_lost")
    return execution, run


def _advisory_failure(exc, executor, retry_state=None):
    """Reuse the review substrate's strongest physical-fact projection."""
    from types import SimpleNamespace
    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection

    history = _ReviewAttemptHistory()
    # This direct caller shares the main thread: an unrelated earlier Main
    # capture is not evidence about this critic. Native/delegated executors
    # supply their own failure facts; an attached capture belongs to this error.
    if getattr(exc, "physical_attempt_capture", None) is not None:
        history.observe(exc)
    usage, _, _, operation_state, code = _review_exception_projection(
        exc, executor.failure_custody(), history, retry_state,
    )
    usage["operation_state"] = operation_state
    source = getattr(executor, "_raw_transcript", None)
    return SimpleNamespace(
        source_text=source if isinstance(source, str) else "",
        success=False, result_text="(no output)",
        session_id=str(usage.get("delegated_run_id") or ""), cost_usd=0.0,
        usage=usage, failure_code=code, error=f"{type(exc).__name__}: {exc}", stderr_tail="",
    )


def _checkpoint_advisory_execution(ctx, repo_dir, commit_message, paths, options, invocation_id):
    """Bind the existing delegated record before its provider POST."""
    execution = options["execution"]
    from ouroboros.tools.git_review_cycle import _fingerprint_staged_diff
    from ouroboros.review_execution import ReviewRouteUnavailable

    binding = _fingerprint_staged_diff(repo_dir)
    if not binding.get("ok"):
        raise ReviewRouteUnavailable("preflight candidate fingerprint unavailable before dispatch", code="advisory_candidate_unavailable")
    execution.update(operation_state="in_flight", fingerprint=binding["fingerprint"])
    execution.update(invocation_id=invocation_id, pending_invocation_id=invocation_id)
    try:
        _car()._persist_preflight_record(ctx, options["snapshot_hash"], commit_message, {
            "status": "pending", "execution": execution, "paths": paths,
            "review_rebuttal": options.get("review_rebuttal", ""),
            "raw_result": "Preflight execution checkpointed before physical dispatch.",
            "strict": True,
        })
    except Exception as exc:
        raise ReviewRouteUnavailable(
            "preflight custody checkpoint could not be persisted before dispatch",
            code="review_custody_checkpoint_unwritable",
        ) from exc


def _run_advisory_delegated(prompt: str, repo_dir: pathlib.Path, ctx: ToolContext, *, execution=None, checkpoint=None):
    """The advisory as a delegated agent session on the SHARED executor seam.

    One substrate executor (``AgentSessionReviewExecutor``) owns the session:
    route resolution, the pre-POST durable invocation checkpoint and retry
    custody, D19 verdict canonicalization, and the capability-delta
    disclosure vocabulary — the advisory adds NOTHING transport-shaped of its
    own (phase C unification, owner decision 2=B, 2026-08-30). Cost: the run
    settles through delegate_custody (the subscription-session ledger row);
    ``cost_usd`` stays 0.0 here so nothing double-counts, and the disclosed
    spend rides ``usage`` for forensics."""
    from types import SimpleNamespace

    from ouroboros.delegate_custody import custody_root
    from ouroboros.llm import LLMClient
    from ouroboros.review_execution import (
        AgentSessionReviewExecutor,
        ReviewAssignment,
        ReviewRouteKind,
    )
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot
    from ouroboros.reviewer_slot_config import advisory_slot_config

    _slot = advisory_slot_config()
    _task_metadata = getattr(ctx, "task_metadata", {}) or {}
    deadline_at = (
        str(_task_metadata.get("deadline_at") or "")
        if isinstance(_task_metadata, dict) else ""
    )
    request = ReviewRequest(
        surface="advisory_review",
        goal="Advisory pre-review of the live worktree.",
        task_id=str(getattr(ctx, "task_id", "") or ""),
        session_root=str(repo_dir),
        session_task=prompt,
        policy={"output_contract": (
            "A JSON array of checklist entries: "
            '[{"item": str, "verdict": "PASS"|"FAIL", "severity": '
            '"critical"|"advisory", "reason": str, "obligation_id"?: str}]'
        )},
        no_proxy=True,
        deadline_at=deadline_at,
    )
    rslot = ReviewSlot(
        slot_id="advisory_slot_1", model=_slot.target_id or "",
        effort=str(_slot.effort or ""), role_hint="advisory pre-reviewer",
        route=ReviewRouteKind.AGENT_SESSION,
        session_target=str(_slot.target_id or ""),
        session_profile=str(getattr(_slot, "profile_id", "") or ""),
        timeout_sec=_ADVISORY_SESSION_MAX_SECONDS,
        subagent_id=str(getattr(_slot, "subagent_id", "") or ""),
    )
    drive = custody_root(ctx) if getattr(ctx, "drive_root", None) else pathlib.Path(repo_dir)
    assignment = ReviewAssignment(
        request=request, slot=rslot,
        call_id=f"advisory:{request.task_id or 'manual'}",
        custody_root=drive,
    )
    executor = AgentSessionReviewExecutor(assignment, llm=LLMClient())
    retry_state = dict(execution or {})
    executor.restore_custody(retry_state)
    executor.set_pending_invocation_checkpoint(checkpoint)
    try:
        if retry_state.get("pending_invocation_id"):
            from ouroboros.delegate_custody import invocation_record
            from ouroboros.review_execution import ReviewRouteUnavailable

            stored = invocation_record(drive, retry_state["pending_invocation_id"])
            request_body = stored.get("request") if isinstance(stored, dict) else None
            if not isinstance(request_body, dict) or not isinstance(request_body.get("prompt"), str):
                raise ReviewRouteUnavailable("canonical preflight request is unavailable", code="review_custody_lost")
            # The executor's own immutable payload cache is restored from the
            # canonical request, never rebuilt from mutable review history.
            # The shared transport still corroborates owner/root/operation.
            executor._session_prompt = request_body["prompt"]
        attempt = executor.execute()
    except Exception as exc:
        return _advisory_failure(exc, executor, retry_state), ""
    usage = dict(attempt.usage or {})
    resolved_model = str(usage.get("resolved_model") or "")
    source = attempt.message.get("session_transcript") if isinstance(attempt.message, dict) else None
    return SimpleNamespace(
        source_text=source if isinstance(source, str) else str(attempt.raw_text or ""),
        success=True,
        result_text=str(attempt.raw_text or ""),
        session_id=str(usage.get("delegated_run_id") or ""),
        cost_usd=0.0,  # settled by delegate_custody; never re-emitted here
        usage=usage,
        error="",
        stderr_tail="",
    ), resolved_model


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
    from ouroboros.reviewer_slot_config import advisory_slot_config

    _slot = advisory_slot_config()
    if delegated_route:
        model = ""  # the session route resolves its own model; reported after the run
    else:
        # The native episode runs on the row's routed catalog model (6.1);
        # '' keeps the shipped routed default; either resolves through the
        # same-model payable-spelling fallback. No provider credentials is a
        # loud typed error here — the commit gate pre-bypasses this state
        # (advisory_model_credentials_missing) before ever calling in.
        from ouroboros.provider_models import model_has_credentials

        model = _car()._advisory_native_model()
        if not model_has_credentials(model):
            return [], (
                f"⚠️ ADVISORY_ERROR: no provider credentials for advisory model "
                f"{model}; add the provider key or point the advisory row at a "
                "configured subagent / another routed model."
            ), "", 0
    options = dict(options or {})
    drive_root = options.get("drive_root")
    include_repo_diff = bool(options.get("include_repo_diff", True))
    review_surface = str(options.get("review_surface") or "repo")
    expected_items = options.get("expected_items")
    try:
        setattr(ctx, "_last_claude_advisory_meta", {})
    except Exception:
        pass

    execution = options.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    options["execution"] = execution
    def assembly_failure(message, *, phase="context"):
        execution.update(failure_phase=phase, failure_code="context_unavailable")
        _note_meta_error(ctx, {"execution": execution}, message)
        return [], message, "", 0

    resuming = bool(execution.get("pending_invocation_id"))
    if resuming:
        from ouroboros.delegate_custody import custody_root, invocation_record

        stored = invocation_record(custody_root(ctx), execution["pending_invocation_id"])
        request_body = stored.get("request") if isinstance(stored, dict) else None
        if not isinstance(request_body, dict) or not isinstance(request_body.get("prompt"), str):
            return [], "⚠️ ADVISORY_ERROR: canonical pending request is unavailable", "", 0
        prompt = request_body["prompt"]
        resolved_paths, managed_subject_diff = list(paths or []), False
    else:
        try:
            if include_repo_diff:
                diff_text, context_paths, early, managed_subject_diff = _car()._advisory_review_diff(
                    repo_dir, ctx, paths
                )
                if early is not None:
                    kind, message, early_chars = early
                    if kind == "error":
                        return assembly_failure(message, phase="authority")
                    return [], message, model, early_chars
                if diff_text.startswith("⚠️ ADVISORY_ERROR:"):
                    return assembly_failure(diff_text)
                changed_files_text = _car()._get_changed_file_list(repo_dir, paths=context_paths)
                if changed_files_text.startswith("⚠️ ADVISORY_ERROR:"):
                    return assembly_failure(changed_files_text)
                resolved_paths, touched_pack, omitted_paths = _car().build_advisory_changed_context(
                    repo_dir,
                    changed_files_text=changed_files_text,
                    paths=context_paths,
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
                managed_subject_diff = False

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
                    "review_rebuttal": str(options.get("review_rebuttal") or ""),
                    "expected_items": expected_items,
                },
                # Both deliveries RETRIEVE governance docs via mandatory-read
                # pointers (the session with its own tools, the native episode with
                # host inspection tools): the inlined multi-hundred-KB governance
                # pack died with the Claude-SDK transport.
                governance_by_retrieval=True,
            )
        except Exception as exc:
            return assembly_failure(f"⚠️ ADVISORY_ERROR: failed to build advisory prompt: {exc}")

    prompt_chars = len(prompt)
    diag = _car()._get_runtime_diagnostics(model, prompt_chars, resolved_paths)
    size_skip = None if resuming else _car()._predispatch_size_skip(ctx, delegated_route, model, prompt, managed_subject_diff)
    if size_skip is not None:
        return size_skip

    log.info(
        "Advisory dispatch: model=%s prompt_chars=%d touched=%s",
        diag["model"], diag["prompt_chars"], diag["touched_paths"],
    )

    checkpoint = partial(_checkpoint_advisory_execution, ctx, repo_dir, commit_message, paths, options)

    try:
        if delegated_route:
            # 5.8: only the transport changes — the delegated session runs the
            # SAME advisory prompt in the same repo root and rehydrates the same
            # result structure. The SDK budget kill is replaced by the runner's
            # nanny-enforced time cap; cost settles through delegate_custody.
            scope_effort = ""  # the session route carries its own effort
            custody_args = ({"execution": execution, "checkpoint": checkpoint}
                            if options.get("snapshot_hash") else {})
            result, model = _car()._run_advisory_delegated(prompt, repo_dir, ctx, **custody_args)
        else:
            # The native inspection episode (the retired Claude-SDK
            # transport's successor): same prompt, same repo root, same result
            # structure. The SDK budget kill is replaced by the episode's
            # transcript bound derived from THIS reviewer's own window
            # (``review_native_episode.review_native_transcript_bound``) — no
            # round cap; every provider call rides the ordinary usage ledger
            # under category=advisory_review.
            scope_effort = _slot.effort or "low"
            if _car().owner_deadline_exhausted_for_context(ctx, reserve_sec=_car().get_finalization_grace_sec()):
                raise TimeoutError("owner deadline leaves no dispatch window for advisory review")
            # The documents the pointer form requires read IN FULL, measured
            # from the files at prompt-build time: the episode's bound is
            # lifted to hold them when the reviewer's window allows, else the
            # prompt and the episode facts carry the typed shortfall code.
            result, model = _car()._run_advisory_native(
                prompt, repo_dir, ctx, _slot, model,
                mandatory_read_corpus_chars=_car()._mandatory_read_corpus_chars(repo_dir, review_surface),
            )

        usage = dict(getattr(result, "usage", {}) or {})
        execution.update(
            pending_invocation_id=str(usage.get("pending_invocation_id") or ""),
            operation_state=str(usage.get("operation_state") or "settled"),
            failure_code=str(getattr(result, "failure_code", "") or ""),
            failure_phase=str(usage.get("review_failure_phase") or ""),
            usage=usage,  # retain native terminal-round/receipt evidence on failures too
        )
        execution.pop("source_text", None)
        original = getattr(result, "source_text", None)
        if not isinstance(original, str):
            original = str(result.result_text or "")
        if original and original != str(result.result_text or ""):
            execution["source_text"] = original  # complete source before canonicalization
        meta = {
            "execution": execution,
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
            skip = _car()._maybe_overflow_skip(
                ctx, delegated_route, prompt_chars, model, meta,
                result.error, getattr(result, "stderr_tail", ""),
                failure_code=str(getattr(result, "failure_code", "") or ""))
            if skip is not None:
                return skip
            err_msg = _car()._format_advisory_error(
                prefix="Advisory delivery returned failure",
                result_error=result.error,
                stderr_tail=result.stderr_tail,
                session_id=result.session_id,
                diag=diag,
            )
            log.error("Advisory delivery failure:\n%s", err_msg)
            _note_meta_error(ctx, meta, err_msg)
            raw_received = str(getattr(result, "result_text", "") or "")
            return [], err_msg + "\n\nReviewer output:\n" + raw_received, model, prompt_chars

        raw_text = str(result.result_text or "")

        if raw_text.strip() in {"", "(no output)"}:
            err_msg = _car()._format_advisory_error(
                prefix="Advisory returned empty output",
                result_error="success=True but result_text was empty",
                stderr_tail=getattr(result, "stderr_tail", "") or "",
                session_id=meta.get("session_id", ""),
                diag=diag,
            )
            _car().emit_review_event(ctx, {
                "type": "advisory_suspect_result",
                "model": model,
                "session_id": meta.get("session_id", ""),
                "prompt_chars": prompt_chars,
                "cost_usd": float(result.cost_usd or 0),
                "reason": "advisory result had empty output",
                "review_surface": review_surface,
            })
            execution.update(failure_phase="format", failure_code="empty_response", source_text=original)
            _note_meta_error(ctx, meta, err_msg)
            return [], err_msg, model, prompt_chars

        items = _parse_advisory_output(raw_text)

        if _needs_fallback_extraction(items, raw_text):
            items = _car()._llm_extract_advisory_items(raw_text, ctx)
            if items:
                log.info("Advisory: structural parse failed, LLM fallback extracted %d items", len(items))

        contract_error, contract_warning = _check_expected_items(items, expected_items)
        if contract_error:
            err_msg = _car()._format_advisory_error(
                prefix="Advisory returned malformed checklist",
                result_error=contract_error,
                stderr_tail=getattr(result, "stderr_tail", "") or "",
                session_id=meta.get("session_id", ""),
                diag=diag,
            )
            _car().emit_review_event(ctx, {
                "type": "advisory_suspect_result",
                "model": model,
                "session_id": meta.get("session_id", ""),
                "prompt_chars": prompt_chars,
                "cost_usd": float(result.cost_usd or 0),
                "reason": contract_error,
                "review_surface": review_surface,
            })
            execution.update(failure_phase="format", failure_code="checklist_contract", source_text=original)
            _note_meta_error(ctx, meta, err_msg)
            return items, err_msg + "\n\nFull reviewer output:\n" + raw_text, model, prompt_chars

        if contract_warning:
            _car().emit_review_event(ctx, {
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

    except Exception as e:
        skip = _car()._maybe_overflow_skip(ctx, delegated_route, prompt_chars, model, None, str(e), verb="raised")
        if skip is not None:
            return skip
        err_msg = _car()._format_advisory_error(
            prefix=f"Advisory delivery raised {type(e).__name__}",
            result_error=str(e),
            stderr_tail="",
            session_id="",
            diag=diag,
        )
        log.error("Advisory delivery exception:\n%s", err_msg)
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
    return _car().empty_array_is_verified_clean(text)


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
    """Select the result array before validating its rows as a whole.

    Enum validation must not make the scanner skip an invalid final checklist
    and accept an earlier example. Unrelated numeric arrays remain ignorable.
    """
    items = _car().extract_json_array(
        stdout,
        unwrap_result=True,
        validate_fn=lambda rows: any(isinstance(row, dict) for row in rows),
    ) or []
    if not _is_checklist_array(items):
        return []  # the complete raw answer reaches the existing extraction rail
    return [dict(item, verdict=item["verdict"].strip().upper(), **(
        {"severity": item["severity"].strip().lower()} if "severity" in item else {}
    )) for item in items]


def _is_checklist_array(items: list) -> bool:
    """Return True iff items looks like a real advisory checklist array.

    Unknown enum values invalidate the whole array, never silently dropping a
    finding or choosing its seriousness. PASS without severity stays compatible;
    FAIL requires the reviewer's critical/advisory classification. Empty clean
    responses are recognized separately by _is_clean_verdict.
    """
    if not isinstance(items, list) or not items:
        return False
    return all(
        isinstance(el, dict)
        and isinstance(el.get("item"), str) and bool(el["item"].strip())
        and isinstance(el.get("verdict"), str) and el["verdict"].strip().upper() in {"PASS", "FAIL"}
        and (
            (el["verdict"].strip().upper() == "PASS" and "severity" not in el)
            or isinstance(el.get("severity"), str)
            and el["severity"].strip().lower() in {"critical", "advisory"}
        )
        for el in items
    )
