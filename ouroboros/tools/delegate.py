"""Nanny tools: run a subagent's cognition on an already-paid subscription session.

A delegated subagent is an ORDINARY Ouroboros subagent acting as a NANNY: it lives in
the task tree with its own deadline and authority, but instead of thinking on metered
API tokens it starts a Claudexor run, watches it, and brings the result home. Because
the nanny IS the host, verification receipts stay host-authored and the harness's
output is a claim, not proof.

Four verbs: ``delegate_start``, a time-bounded ``delegate_wait``,
``delegate_cancel``, and ``delegate_answer`` (a run's pending interactive question is
answered by its own nanny — owner decision 7=A, poltergeist phase B). There is still
no ``hurry`` — Claudexor's only control verb is ``cancel``, and cancelling a reviewer
destroys the verdict you wanted.

Read-only and mutating children share ONE nanny and ONE transport. The only difference
is the access profile the HOST derives from the calling task's authority (``readonly``
vs ``workspace_write``) and the run shape that follows from it; there is no second
pipeline and no second slot. The child gets a broker tool, never a shell, so it can ask
the host to run something but never choose with what powers.

Custody: the daemon token never leaves ``gateways.claudexor``; nothing here puts it in
a ToolContext, a child environment, or a harness sandbox. WHICH run belongs to WHICH
task is decided by ``ouroboros.delegate_custody`` against the durable event log, not by
a dict this process happens to still hold.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ouroboros import delegate_custody as custody
from ouroboros import delegate_progress as progress
from ouroboros.delegate_custody import RunCustody as _RunCustody
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.subagent_work_order import (  # noqa: F401 - compatibility re-export
    _FIELD_CHARS as _ASSIGNMENT_FIELD_CHARS,
    assignment_instructions as _assignment_instructions,
)
from ouroboros.subagent_work_order import start_binding_fingerprints
from ouroboros.delegate_supervision import delegate_wait_entry as _delegate_wait_entry
from ouroboros.subagent_runtime import (  # noqa: F401 - shared primitive re-export
    delegate_start_entry as _delegate_start_entry,
    exact_start,
)
from ouroboros.subagent_runtime import prepare_delegate_start_actor
# The staged-output + read-receipt cluster lives in its own module (size gate);
# re-exported here because sibling code, the tests and the convergence census all
# name it on THIS surface, and `_READ_COVERAGE` must stay the same object.
from ouroboros.delegate_output import (  # noqa: F401
    _ARTIFACT_SUBDIR,
    _BULK_FIELDS,
    _PAYLOAD_ENVELOPE_HEADROOM,
    _PREVIEW_PREFIX_SLACK,
    _PREVIEW_STEPS,
    _READ_COVERAGE,
    _READ_COVERAGE_MAX_KEYS,
    _STRUCTURED_FIELDS,
    _covered_whole,
    _preview_payload,
    _resolve_full_primary_output,
    _safe_run_filename,
    _stage_full_output,
    acknowledge_staged_output_read,
)
# The interactive-question cluster (waiting_on_user + delegate_answer) lives in
# its own module too (size gate); re-exported here because the wait loop, the
# tests and sibling code name it on THIS surface, and `_REPORTED_INTERACTIONS`
# must stay the same object.
from ouroboros.delegate_interactions import (  # noqa: F401
    _ANSWER_NOTES,
    _REPORTED_INTERACTIONS,
    _answer_delivery_unknown,
    _bounded_interactions,
    _delegate_answer,
    _interactions_are_news,
    _normalized_answers,
    _waiting_on_user_payload,
)
# The refusal/emit/ownership helpers live in the neutral leaf
# `ouroboros/delegate_shared.py` (moved to break the facade back-edge:
# delegate_interactions needs them, and an extracted module never imports the
# facade back); re-exported here because sibling code, the tests and
# monkeypatch targets name them on THIS surface.
from ouroboros.delegate_shared import (  # noqa: F401
    _emit,
    _fail,
    engine_supports_workspace_root,
    is_legacy_workspace_root_request,
    _owned_run,
)
from ouroboros.delegate_start_binding import (  # noqa: F401
    FreshStartBinding,
    prepare_fresh_start,
    start_request as _start_request,
)
# The C1 integration seam (mutation authority, execution snapshots, retry binding,
# terminal patch capture) lives in its own module (size gate); re-exported here
# (same objects) because sibling code and the tests address it on THIS surface.
# `_fail` is NOT re-imported from it — the one shared refusal author is
# `delegate_shared._fail`, which delegate_integration itself imports.
from ouroboros.tools.delegate_integration import (  # noqa: F401
    _CAPTURE_DELEGATED_SNAPSHOT,
    _capture_block,
    _capture_terminal_patch,
    _mutation_authority,
    _payload_mutation_authority,
    _payload_selector_refusal,
    _provision_payload_snapshot,
    _provision_snapshot,
    _resolve_retry_invocation,
    _resolved,
    _retry_binding_refusal,
    _validated_invocation,
    claimed_start_request,
    payload_host_instructions,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ouroboros.subagents import DelegatedRunShape

log = logging.getLogger(__name__)

_TERMINAL_STATES = custody.TERMINAL_STATES

# The containment verifiers moved to `ouroboros/delegate_containment.py` whole (the
# module-size gate); re-exported here because the nanny's seams and the existing
# tests address them through this module.
from ouroboros.delegate_containment import (  # noqa: E402
    _ACCESS_UNVERIFIED,  # noqa: F401  (re-export: tests address it through this module)
    _Breach,
    _home_isolation_breach,
    _widened_access,
    home_nested_under_operator_home,
)
_POLL_INTERVAL_SEC = 3.0
# Claudexor's own schema bound on maxSeconds (packages/schema/src/control.ts).
_CLAUDEXOR_MAX_SECONDS = 604_800

# The process-local memo of the durable custody rows (the authority lives in the module
# above); re-bound here because sibling code and tests name it on this surface.
_CUSTODY = custody._CUSTODY

# Layered onto every lane by Claudexor (native system-prompt channel per harness). The
# SAME prohibitions an ordinary subagent carries; a statement, not the enforcement —
# that is the access profile plus the nanny's own workspace-patch capture.
_HOST_INSTRUCTIONS = (
    "You are a delegated worker running inside another agent's working tree. Your "
    "authority is everything INSIDE this root and nothing outside it. Do not run git "
    "commit, tag, push, rebase, reset or any other history-moving command: your host "
    "takes the diff of this tree and integrates it itself, and a moved HEAD invalidates "
    "that diff and destroys your work. Do not review or accept your own change, do not "
    "touch the host's runtime controls, skills, or memory, and do not write outside "
    "this root. If your environment offers a way to ask your host a clarifying "
    "question, you may use it: your host may answer from its task context; a question "
    "that carries an engine expiry times out benignly if unanswered — continue with "
    "stated assumptions rather than blocking — while one without an expiry waits until "
    "answered. If your harness cannot ask mid-run, do NOT end the run to ask — "
    "state your assumption and continue."
)

# DESTINATION 2 of the disclosure ("Disclose instead of forbid": durable record, the
# CHILD'S PROMPT, the parent's result). The OS boundary is a REQUEST — the engine applies
# one only where it has a mechanism, and nothing at start knows which — so the child is
# told the only true thing: behave as though nothing is stopping you, and do not claim an
# isolation you cannot show. What was applied reaches the parent from the run's own
# artifacts through `_containment_evidence`.
_UNPROVEN_BOUNDARY_INSTRUCTION = (
    " An OS-enforced filesystem boundary was REQUESTED for this run but is NOT guaranteed: "
    "your engine applies one only where it has a mechanism for this host, and your host "
    "reads back from your own attempt records what was actually applied. Work as if there "
    "is no boundary — stay inside this root, do not read the operator's home directory, "
    "credential stores, or the harness runtime tree, and do NOT describe yourself in your "
    "answer as sandboxed or confined. If your own environment shows you whether a boundary "
    "was in force, say so plainly."
)


def _host_instructions(authority: "DelegatedRunShape", assignment: str = "",
                       payload_skill: str = "") -> str:
    """The system-prompt text this run's shape earns. One builder, no dialect.

    ``assignment`` is the host-authored contract block (``_assignment_instructions``);
    appended last so the prohibitions stay the opening statement. A payload run
    (``payload_skill`` non-empty) gets the truthful variant: editing the selected
    skill's user-authored files IS the assignment (gate fix 3).
    """
    text = _HOST_INSTRUCTIONS
    if payload_skill:
        text = payload_host_instructions(text, payload_skill)
    if authority.delegated:
        text += _UNPROVEN_BOUNDARY_INSTRUCTION
    if assignment:
        text += "\n\n" + assignment
    return text


def _derive_authority(ctx: ToolContext) -> "DelegatedRunShape":
    """Derive the run shape from the task's own authority — one question, asked here.

    Host-derived, never model-supplied: the child asks the host to run something, and
    the host decides with what powers. Ouroboros asks for an access PROFILE and lets
    Claudexor pick the mechanism (fs sandbox, tool allowlist, ...) — no harness branch.

    The SHAPE itself belongs to ``subagents.delegated_run_shape``, which the dispatcher
    also reads: this function only answers "does this task hold a mutating surface",
    which is the one part that needs the live ``ToolContext``. Two authorities qualify
    (B5, owner 2=A): an ACTING CHILD with a valid write surface, and the ROOT of an
    EXTERNAL-WORKSPACE task — the root already holds write+shell inside the project,
    so its delegated runs carry the same mutating shape, bounded by the same
    workspace; ``_mutation_authority`` (``tools.delegate_integration``) validates the
    concrete target either way.
    """
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tool_access import active_tool_profile

    profile = active_tool_profile(ctx)
    mutating = profile in ("acting_subagent", "external_workspace_task")
    if mutating:
        from ouroboros.presence_authority import presence_ceiling_allows_delegated_surface

        constraint = getattr(ctx, "task_constraint", None)
        surface = str(getattr(constraint, "surface", "") or "external_workspace")
        mutating = presence_ceiling_allows_delegated_surface(ctx, surface)
    return delegated_run_shape(mutating)


def _presence_delegate_read_refusal(ctx: ToolContext) -> str:
    from ouroboros.presence_authority import presence_ceiling_allows_delegated_read

    if presence_ceiling_allows_delegated_read(ctx):
        return ""
    return _fail(
        "delegate_start",
        "presence_delegate_read_root_unselected",
        "This Presence profile did not select whole-root read access for the active repository, "
        "so a delegated harness cannot honestly be started inside that broader read surface.",
    )


def _containment_breach(detail: Dict[str, Any], authority: "DelegatedRunShape") -> Optional[_Breach]:
    """Everything the ENGINE enforced, checked against what the host asked for.

    ONE reader for both halves of containment — the access profile and the harness
    HOME — because they fail identically: the request is only a request, the engine
    derives the truth, and a verification written for one half leaves the other
    trusting an echo. The HOME half is asked only of a run that carried the marker;
    a read-only child is scoped by Claudexor's ordinary envelope and asks for nothing.
    """
    widened = _widened_access(detail, authority.access)
    if widened:
        return _Breach(
            "access_profile_widened",
            f"The delegated run was enforced at access profile {widened!r} while this "
            f"task is only entitled to {authority.access!r}.",
            {"entitled_access": authority.access, "effective_access": widened},
        )
    if authority.delegated:
        return _home_isolation_breach(detail)
    return None


_NESTED_HOME_NOTE = (
    "The scoped harness HOME for this run sits INSIDE the operator's own home, which is "
    "where the engine roots its scoped homes. That is allowed and the run's work is usable, "
    "but it is not isolation from the operator's home: everything there — credential stores "
    "and the Claudexor daemon token included — stays readable at its absolute path. Do NOT "
    "describe this run as running in an isolated home"
)

_NO_BOUNDARY_NOTE = (
    "NO OS-ENFORCED BOUNDARY was applied to this run. The engine reported no confinement "
    "mechanism for it, so the only containment it had is a scoped HOME — a redirect of "
    "`~`-relative lookups, which leaves the operator's home, credential stores and the "
    "Claudexor daemon token readable at their absolute paths. The run was allowed and its "
    "work is usable; do NOT describe it as sandboxed, confined or isolated, and weigh its "
    "output as coming from an unconfined shell in this worktree"
)


def _containment_evidence(detail: Dict[str, Any]) -> Dict[str, Any]:
    """What the ARTIFACTS prove about this run's containment — never what was asked.

    DESTINATION 3 of the disclosure: this is what the nanny hands its parent.

    BOTH halves, in one reader, because a report that states only the scoped HOME is the
    defect this function was rewritten to remove: a run with a kernel-enforced boundary
    and a run with none produced BYTE-IDENTICAL evidence here, both reading
    ``verified: true`` with a note about the HOME. Claudexor's own confinement document
    says the scoped home "is not a boundary and must never be reported as one".

    The predicate is what the engine says it APPLIED (``confinement_mechanism`` plus the
    denied path it proved), never which OS this host is. Ouroboros does not know what the
    engine did — only the artifact does — and a platform test would additionally freeze
    today's answer: the day a boundary ships for another OS, this reader is already right.

    Judged by the SAME predicate that halts a breached run, not by having been reached
    after it: a report whose honesty depends on its call site is one refactor away from
    claiming a containment nobody checked.

    This is also where a MISSING fact lands, because it is a reporting question and not an
    enforcement one: an attempt that disclosed nothing proves nothing, so ``verified``
    stays false and ``disclosed`` says how much of the run is actually covered. Silence
    read as success and silence enforced as a fault are the two ways to be wrong here,
    and stating the count avoids both.
    """
    from ouroboros.gateways.claudexor import attempt_containment

    attempts = attempt_containment(str(custody.summary_of(detail).get("runDir") or ""))
    disclosed = sum(1 for attempt in attempts if attempt.home_isolated is not None)
    # An engine that reported nothing is indistinguishable from one that applied nothing,
    # and the mechanisms the ATTEMPTS name are the vocabulary — Ouroboros keeps no list of
    # its own to fall out of date. "Every attempt" and not "any": one unconfined attempt
    # is an unconfined run.
    mechanisms = sorted({attempt.boundary_mechanism for attempt in attempts})
    boundary = mechanisms[0] if attempts and len(mechanisms) == 1 and mechanisms[0] else ""
    # A3: the engine's own typed reason for a missing boundary — an AMPLIFIER of
    # the unconfined disclosure (why there is no mechanism on this host), parsed
    # from the same attempt artifact. Telemetry only, never an admission token.
    unavailable_reasons = sorted({
        attempt.confinement_unavailable_reason
        for attempt in attempts if attempt.confinement_unavailable_reason
    })
    # A3: a scoped home NESTED under the operator's own is allowed (the engine's
    # own layout — disclosed, never refused), but it is NOT "outside the
    # operator's own": the daemon token stays reachable at its absolute path.
    # Recorded on the report and honoured by every branch below, so a run that
    # ALSO carries an OS boundary can no longer be promoted to verified with a
    # note that contradicts its own artifact — and so `_record_containment` keeps
    # emitting the durable unconfined row for it.
    nested = home_nested_under_operator_home(detail)
    report = {"verified": False, "attempts": len(attempts), "disclosed": disclosed,
              "os_boundary": boundary, "nested_under_operator_home": nested}
    if unavailable_reasons:
        report["confinement_unavailable_reason"] = "; ".join(unavailable_reasons)
    breach = _home_isolation_breach(detail)
    if breach is not None:
        return {**report, "note": breach.detail}
    if not disclosed:
        return {**report, "note":
                "this run recorded no harness-HOME fact, so its confinement is UNPROVEN "
                "— do not report it as isolated"}
    if disclosed < len(attempts):
        return {**report, "note":
                "not every attempt of this run recorded a harness-HOME fact, so its "
                "confinement is UNPROVEN — do not report it as isolated"}
    if nested:
        note = _NESTED_HOME_NOTE
        if boundary:
            note += (
                f" (an {boundary} boundary WAS applied — weigh it as the real containment, "
                "but the scoped HOME is not one)"
            )
        if unavailable_reasons:
            note += " (engine-declared reason: " + "; ".join(unavailable_reasons) + ")"
        return {**report, "note": note}
    if not boundary:
        note = _NO_BOUNDARY_NOTE
        if unavailable_reasons:
            note += (
                " (engine-declared reason: " + "; ".join(unavailable_reasons) + ")"
            )
        return {**report, "note": note}
    return {**report, "verified": True, "note":
            f"every attempt recorded a scoped harness HOME outside the operator's own AND "
            f"an applied {boundary} boundary, proven against a path it denies"}


def _terminal_payload(run_id: str, detail: Dict[str, Any],
                      authority: "DelegatedRunShape") -> Dict[str, Any]:
    summary = custody.summary_of(detail)
    payload = {
        "status": "terminal",
        "run_id": run_id,
        "state": str(summary.get("state") or ""),
        # The APPLIED model, from the engine's own summary — '' when the run
        # never disclosed one (live unpinned runs really do), shown as absence
        # rather than the requested model dressed up as the applied one.
        "model": str(summary.get("model") or ""),
        "outcome_banner": detail.get("outcomeBanner"),
        "outcome_facts": summary.get("outcomeFacts"),
        "output_conformance": summary.get("outputConformance"),
        "final_summary": detail.get("finalSummary"),
        "primary_output": detail.get("primaryOutput"),
        "failure": summary.get("failure"),
        "last_seq": int(detail.get("lastSeq") or 0),
        "cost": _reported_cost(summary),
        # The ACCESS half of the same honesty, on EVERY terminal payload — see
        # `_access_evidence`. Both lanes: `readonly` staying `readonly` is the profile
        # that matters most, while `containment` is asked only of marker-carrying runs.
        "access_evidence": _access_evidence(detail, authority.access),
    }
    if authority.delegated:
        payload["containment"] = _containment_evidence(detail)
    facts = payload.get("outcome_facts")
    if isinstance(facts, dict) and str(facts.get("reason") or "") == "input_required":
        # The codex-shaped question (B4): that lane has no mid-run channel, so a
        # question arrives as this TERMINAL. There is deliberately NO rerun verb
        # here — the engine's rerun_with_feedback would start a run outside this
        # task's custody trail — so the honest answer path is a plain new start.
        payload["input_required_note"] = (
            "This run ended NEEDING INPUT (outcome_facts.reason=input_required — "
            "see outcome_facts.work_state.required_inputs). Its harness has no "
            "mid-run question channel, so the question arrives as this terminal. Answer it by "
            "starting a plain NEW delegate_start(subagent_id=..., prompt=...) whose "
            "prompt carries the original "
            "assignment plus the answers; custody of the new run stays with you. "
            "Do not look for a rerun/decision verb — none exists on this surface."
        )
    return payload


def _access_evidence(detail: Dict[str, Any], expected: str) -> Dict[str, Any]:
    """What the engine's own DERIVED profile proves about this finished run.

    ``effectiveAccess`` is the only witness: ``summary["access"]`` is computed as
    ``effectiveAccess ?? the client's own request``, so reading it compares the request
    against itself and always passes. A WIDER profile is already a breach before this
    runs; an ABSENT one cannot be enforced on a run that is over — cancelling a
    succeeded run to punish missing evidence would destroy the result the lane exists
    to fetch (the v6.87.37 lesson) — so it is named here instead.
    """
    summary = custody.summary_of(detail)
    effective = str(summary.get("effectiveAccess") or "")
    state = str(summary.get("state") or "")
    report = {"requested": expected, "effective": effective,
              "verified": bool(effective), "state": state}
    if effective:
        return report
    if state in custody.SUCCEEDED_STATES:
        return {**report, "note":
                "this run SUCCEEDED without ever disclosing an effective access "
                f"profile, so there is no evidence the engine enforced {expected!r} — "
                "do not report its containment as verified"}
    return {**report, "note":
            "no effective access profile was disclosed; a run that did not succeed may "
            "never have had one, so this is absence of evidence, not a breach"}


def _record_containment(ctx: ToolContext, entry: Optional[_RunCustody],
                        payload: Dict[str, Any]) -> None:
    """DESTINATION 1 of the disclosure: the durable record, written once per run.

    A missing boundary is not a fault and produces no refusal, which is exactly why it
    needs a durable line of its own — the run succeeds, its patch is integrated, and
    nothing else in the record would ever say the work came out of an unconfined shell.
    Emitted from what the PARENT was told, so the two cannot disagree.

    "Once per run" is now a DURABLE fact rather than a process-local one: the custody
    entry is replayed from the event log, so a restarted worker polling an already
    terminal run does not append a second identical finding.

    A NESTED scoped home is disclosed even when an OS boundary WAS recorded (A3):
    the boundary is real containment, the scoped home is not, and suppressing the
    row for that shape left the one durable line that says "this ran with the
    operator's home reachable" unwritten.
    """
    containment = payload.get("containment")
    if not isinstance(containment, dict):
        return
    if containment.get("os_boundary") and not containment.get("nested_under_operator_home"):
        return
    if entry is not None and entry.containment_disclosed:
        return
    _emit(ctx, custody.UNCONFINED, {
        "run_id": entry.run_id if entry is not None else "",
        "route": entry.route_id if entry is not None else "",
        "state": str(payload.get("state") or ""),
        "os_boundary": str(containment.get("os_boundary") or ""),
        "attempts": containment.get("attempts"),
        "home_disclosed": containment.get("disclosed"),
        "nested_under_operator_home": bool(containment.get("nested_under_operator_home")),
        "note": containment.get("note"),
        **({"confinement_unavailable_reason": containment["confinement_unavailable_reason"]}
           if containment.get("confinement_unavailable_reason") else {}),
    })
    if entry is not None:
        entry.containment_disclosed = True


def _reported_cost(summary: Dict[str, Any]) -> Dict[str, Any]:
    """What this run cost, as the AGENT will read it.

    This is the payload the nanny relays to its parent, so it must tell the same story
    the ledger does. It used to hardcode `$0.00 / final` — the exact shape the settlement
    fix was written to eliminate — so a run that really charged money settled honestly in
    the ledger and then told the reasoning path the work was free.
    """
    spend, estimated = custody.disclosed_spend(summary)
    if spend is None:
        return {
            "cost_usd": None,
            "cost_final": False,
            "note": "the harness disclosed no spend for this run; treat the cost as UNKNOWN, not zero",
        }
    if estimated:
        # The amount is the best fact anyone has, so it rides; the FINALITY does not. An
        # estimated zero is not a proven free session and an estimated charge is not a
        # closed book — both are `cost_final: False`, matching the ledger row exactly.
        return {
            "cost_usd": spend,
            "cost_final": False,
            "note": "the harness ESTIMATED this run's spend rather than settling it; treat "
                    "the amount as APPROXIMATE and the cost as NOT final",
        }
    if spend > 0:
        return {
            "cost_usd": spend,
            "cost_final": True,
            "note": "this run was BILLED — it did not ride the subscription",
        }
    return {
        "cost_usd": 0.0,
        "cost_final": True,
        "note": "subscription session — already paid; the nanny's own model calls are metered separately",
    }


# -- output delivery -----------------------------------------------------------


def _delivered_terminal_payload(ctx: ToolContext, run_id: str, detail: Dict[str, Any],
                                authority: "DelegatedRunShape",
                                entry: Optional[_RunCustody] = None,
                                gateway: Any = None) -> Dict[str, Any]:
    """The terminal payload, delivered whole or declared partial — never head-cut.

    ``final_summary``/``primary_output`` carry the run's real work product, and Claudexor
    returns a preview of up to 256 KiB. Outer truncation would head-cut that at the tool
    result limit and sever the JSON mid-string, which destroys the document rather than
    shortening it. So the payload bounds ITSELF against the same limit the truncator
    applies, and the remainder becomes a readable artifact — after the engine's bounded
    preview has been resolved to the verified full artifact, because a payload built on
    a truncated preview delivers 256 KiB wearing the whole result's name.
    """
    full = _terminal_payload(run_id, detail, authority)
    # Requested-vs-applied model, the review lane's own lexicon and rule
    # (AgentSessionReviewExecutor): compared only when BOTH are non-empty —
    # the engine writes aliases ('sonnet' beside 'claude-opus-5'), so a
    # mismatch is an advisory disclosure, never a failure of the run.
    requested_model = str(getattr(entry, "model", "") or "") if entry is not None else ""
    applied_model = str(full.get("model") or "")
    if requested_model and applied_model and requested_model != applied_model:
        full["capability_delta"] = [{
            "kind": "capability_delta",
            "requested": f"model {requested_model}",
            "effective": f"model {applied_model}",
            "reason": "session_route_resolves_its_own_model",
        }]
    primary, full_ok, full_note = _resolve_full_primary_output(
        gateway, run_id, full.get("primary_output"))
    full["primary_output"] = primary
    budget = tool_result_limit("delegate_wait")
    text = json.dumps(full, ensure_ascii=False, indent=2)
    if len(text) <= budget - _PAYLOAD_ENVELOPE_HEADROOM:
        full["output_delivery"] = {
            # An unresolved engine-side truncation makes even an inline-fitting payload
            # NOT the whole result: complete/consumed follow the verified fact.
            "complete": full_ok, "consumed": full_ok, "inline_is_preview": False,
            "total_chars": len(text), "artifact": None, "read_next": None,
            "note": ("The whole terminal payload is inline." if full_ok else
                     "INLINE BUT INCOMPLETE AT THE SOURCE: the engine reported its "
                     "primary output as a bounded preview and the full artifact could "
                     "not be matched to the size or the preview the run itself reported "
                     "(see primary_output_full). Treat this "
                     "as incomplete evidence, not as the verdict."),
        }
        if full_note is not None:
            full["output_delivery"]["primary_output_full"] = full_note
        return full
    artifact = _stage_full_output(ctx, run_id, text)
    _emit(ctx, custody.OUTPUT_SPILLED, {"run_id": run_id, "total_chars": len(text),
                                        "artifact": (artifact or {}).get("path", ""),
                                        "bytes": (artifact or {}).get("bytes"),
                                        "sha256": (artifact or {}).get("sha256", ""),
                                        "staged": artifact is not None,
                                        "full_content": bool(full_ok and artifact is not None)})
    if entry is not None and artifact is not None:
        if entry.output_consumed and entry.output_sha and artifact["sha256"] != entry.output_sha:
            # The ack named OTHER bytes: a re-stage of different content at the same
            # path owes a fresh acknowledgement — consumed never transfers by path.
            entry.output_consumed = False
        entry.output_sha = artifact["sha256"]
        entry.output_artifact = artifact["path"]
        entry.output_complete = bool(full_ok)
    return _preview_payload(full, text, artifact, budget,
                            consumed=bool(entry is not None and entry.output_consumed),
                            full_ok=full_ok, full_note=full_note)


def _delegate_start(ctx: ToolContext, prompt: str, max_seconds: Optional[int] = None,
                    retry_of: Optional[str] = None, root: Optional[str] = None,
                    bucket: Optional[str] = None, skill_name: Optional[str] = None, _resolved_binding: Any = None) -> str:
    from ouroboros.claudexor_daemon import ensure_owned_gateway
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagents import resolve_subagent_executor, route_health

    text = str(prompt or "").strip()
    if not text:
        return _fail("delegate_start", "empty_prompt", "prompt is required")
    selector_root = str(root or "").strip()
    selector_refusal = _payload_selector_refusal(selector_root, retry_of, bucket, skill_name)
    if selector_refusal:
        return selector_refusal
    if _deadline_expired(ctx):
        # An EXPIRED nanny cannot honestly bound anything; refused pre-daemon.
        return _fail(
            "delegate_start", "task_deadline_expired",
            "This task's deadline has already passed, so a delegated run started now "
            "would outlive it by design. Finalize with what you have — do not start "
            "new work a deadline has already closed.",
        )

    drive = custody.custody_root(ctx)
    owned_project_id = ""
    project_persistent = False
    invocation_id = ""
    snapshot_id = ""
    execution_root = ""
    baseline_sha = ""
    target_root = ""
    authority_source = ""
    resource_ref: Dict[str, Any] = {}
    selected_subagent_id = ""
    config_fingerprint = ""
    work_order_fingerprint, authority_fingerprint = start_binding_fingerprints(ctx, text)
    retry_token = str(retry_of or "").strip()
    recovering = bool(retry_token)
    actor, actor_refusal = prepare_delegate_start_actor(
        ctx, drive, recovering=recovering, invocation_id=retry_token,
        work_order_fingerprint=work_order_fingerprint, authority_fingerprint=authority_fingerprint,
    )
    if actor_refusal:
        return actor_refusal
    selected_subagent_id = str(actor.get("selected_subagent_id") or "")
    config_fingerprint = str(actor.get("config_fingerprint") or "")
    work_order_fingerprint = str(actor.get("work_order_fingerprint") or "")
    authority_fingerprint = str(actor.get("authority_fingerprint") or "")
    if recovering:
        binding, refusal = _resolve_retry_invocation(ctx, drive, retry_token, text)
        if refusal:
            return refusal
        (request_body, route, authority, root, execution_root, key, project_id,
         owned_project_id, project_persistent, seconds, snapshot_id, target_root,
         baseline_sha, authority_source,
         resource_ref) = binding
        invocation_id = retry_token
        payload_auth = None
    else:
        route = actor["route"]
        if selector_root:
            authority, payload_auth, payload_error = _payload_mutation_authority(
                ctx, drive, bucket, skill_name, _resolved_binding)
            if payload_error:
                return payload_error
        else:
            authority = _derive_authority(ctx)
            payload_auth = None
            if refusal := _presence_delegate_read_refusal(ctx):
                return refusal

    access = authority.access
    try:
        gateway = ensure_owned_gateway()
    except ClaudexorUnavailable as exc:
        resolution = resolve_subagent_executor("harness", route=route, unavailable_reason=exc.code)
        return _fail("delegate_start", exc.code, str(exc), executor=resolution.executor)

    try:
        workspace_root_supported = engine_supports_workspace_root(gateway)
        # Health checks the stored route/confinement shape on retries, never current
        # environment defaults; blockers stay typed instead of falling through to API spend.
        retry_execution = (request_body.get("execution") if recovering
                           and isinstance(request_body.get("execution"), dict) else {})
        unavailable, reset_at = route_health(
            gateway, route.route_id, authority, route_model=route.model,
            pinned_profile=route.profile_id,
            workspace_root_required=(
                ("workspaceRoot" in retry_execution)
                if recovering else
                (workspace_root_supported and authority.access == "workspace_write")))
        resolution = resolve_subagent_executor(
            "harness", route=route, unavailable_reason=unavailable, reset_at=reset_at,
        )
        if resolution.blocked:
            record_start_blocked(ctx, str(getattr(ctx, "task_id", "") or ""), resolution.reason)
            return _fail(
                "delegate_start", resolution.reason,
                "The delegated route cannot run now. This is a typed blocker: do NOT "
                "silently fall back onto metered API spend — decide explicitly "
                "(wait for the reset, deliver partial work, or ask the parent).",
                executor="blocked", reset_at=resolution.reset_at, route=route.route_id,
            )

        if not recovering:
            binding, refusal = prepare_fresh_start(
                ctx, drive, gateway, route, authority, actor, text, max_seconds,
                payload_auth, workspace_root_supported,
                host_instructions=_host_instructions,
                assignment_instructions=_assignment_instructions,
                bounded_max_seconds=_bounded_max_seconds,
            )
            if refusal:
                return refusal
            assert binding is not None
            (request_body, root, execution_root, invocation_id, key, project_id,
             owned_project_id, project_persistent, seconds, snapshot_id, target_root,
             baseline_sha, authority_source, resource_ref) = binding
        lineage = getattr(ctx, "task_metadata", {}) or {}
        lineage = lineage if isinstance(lineage, dict) else {}
        # Fresh payload run: busy check + durable write = ONE atomic claim (fix 5).
        requested, claim_holder = claimed_start_request(
            drive, claim_target=(target_root if (not recovering and
                                 authority_source == "skill_payload") else ""),
            run_id="", task_id=str(getattr(ctx, "task_id", "") or ""),
            idempotency_key=key, invocation_id=invocation_id,
            max_seconds=seconds, request=request_body, project_id=project_id,
            project_owned=bool(owned_project_id), project_persistent=project_persistent,
            route=route.route_id,
            # Lineage rides the request row so a run RECOVERED from a pending
            # invocation (P34R.2) can still attribute its ledger row to the tree.
            root_task_id=str(lineage.get("root_task_id") or ""),
            parent_task_id=str(lineage.get("parent_task_id") or ""),
            # The C1 isolation binding, durable BEFORE the POST: these name the
            # snapshot, baseline and authority target so a retry reproduces the
            # exact binding and the startup GC can see the pending snapshot.
            snapshot_id=snapshot_id, execution_root=execution_root,
            baseline_sha=baseline_sha, target_root=target_root,
            authority_source=authority_source, resource_ref=resource_ref,
            # Same pre-POST row as the request: crash recovery can prove the
            # exact configured actor and exact compiled brief before adopting.
            selected_subagent_id=selected_subagent_id,
            config_fingerprint=config_fingerprint,
            work_order_fingerprint=work_order_fingerprint,
            authority_fingerprint=authority_fingerprint)
        if claim_holder:
            return _fail(
                "delegate_start", "payload_delegation_busy",
                "Another delegated run claimed this exact payload first (the busy "
                "check and the start-request write are one atomic claim). Finish "
                "that run before starting another delegation against the same "
                "skill.", holder=claim_holder,
                **_retire_orphaned_registration(
                    ctx, gateway, owned_project_id,
                    project_persistent=project_persistent,
                    definite_refusal=True,
                    reason="payload_delegation_busy",
                    invocation_id=invocation_id,
                    snapshot_id=snapshot_id))
        if not requested:
            # The POST is CONDITIONAL on the durable request row: a run started
            # without it is live and unfindable if this worker dies. A fresh
            # start's registration is definitively retirable; a RETRY's project
            # belongs to the original attempt, whose POST may have bound a live
            # run — its fate stays unknown and its invocation stays pending.
            return _fail(
                "delegate_start", "start_request_row_unwritable",
                "The durable start-request row could not be written, so the run was "
                "NOT started: a run launched without its custody trail would be "
                "unfindable if this worker died. Fix the drive/event log and retry.",
                **_retire_orphaned_registration(
                    ctx, gateway, owned_project_id,
                    project_persistent=project_persistent,
                    definite_refusal=not recovering,
                    reason="start_request_row_unwritable",
                    invocation_id=invocation_id,
                    snapshot_id=("" if recovering else snapshot_id)))
        handle = gateway.start_run(request_body, idempotency_key=invocation_id)
        # A queued `jobId` is a usable GET/control handle.
        run_id = str(handle.get("runId") or handle.get("jobId") or "")
        if not run_id:
            # The POST SUCCEEDED, so a run is more likely live here than on the
            # refusal branch beside it — the registration is retained, not abandoned.
            return _fail("delegate_start", "queued_without_run_id",
                         f"Claudexor returned a queued handle without a run id: {handle!r}",
                         pending_invocation_id=invocation_id,
                         retry_hint="to retry THIS start call use "
                                    "delegate_start(prompt=..., "
                                    "retry_of=pending_invocation_id); a plain call starts a NEW run",
                         **_retire_orphaned_registration(ctx, gateway, owned_project_id,
                                                         project_persistent=project_persistent,
                                                         definite_refusal=False,
                                                         reason="queued_without_run_id",
                                                         invocation_id=invocation_id))
    except ClaudexorUnavailable as exc:
        # A registration we created BEFORE the start must not outlive a failed start.
        # It used to be left behind with nothing anywhere naming its id.
        status = int(getattr(exc, "status_code", 0) or 0)
        definite = 400 <= status < 500
        if recovering and is_legacy_workspace_root_request(request_body) \
                and exc.code == "execution_workspace_required":
            return _fail(
                "delegate_start", "legacy_workspace_root_requires_compatible_retry",
                "This legacy mutating invocation was not accepted before the engine "
                "started requiring execution.workspaceRoot. Its exact retry token is "
                "preserved; drain it on the legacy engine or use the engine's "
                "server-owned Exact Retry path before upgrading.",
                pending_invocation_id=invocation_id,
                retry_hint="retry this same invocation after the compatible engine is available",
                engine_version=str(getattr(gateway, "engine_version", "") or ""),
                project_id=project_id,
            )
        # An UNKNOWN outcome hands back the retry token: only the caller can say
        # whether the next call is a retry of this intention or a new intention, and
        # without the token every next call is a new one. A definite refusal retires
        # the id, so no token rides a refusal.
        pending = ({} if definite or not invocation_id else
                   {"pending_invocation_id": invocation_id,
                    "retry_hint": "to retry THIS start call use "
                                  "delegate_start(prompt=..., "
                                  "retry_of=pending_invocation_id); a plain call starts a NEW run"})
        return _fail("delegate_start", exc.code, str(exc), executor="blocked",
                     reset_at=getattr(exc, "reset_at", ""), **pending,
                     **_retire_orphaned_registration(ctx, gateway, owned_project_id,
                                                     project_persistent=project_persistent,
                                                     definite_refusal=definite,
                                                     reason=str(getattr(exc, "code", "")),
                                                     invocation_id=invocation_id,
                                                     snapshot_id=("" if recovering else snapshot_id)))
    except BaseException as exc:
        # EVERY pre-custody exit leaves a durable disposition, including the ones no
        # typed handler claims (a bug here, a timeout, a signal). NEVER retired: an
        # untyped exit says nothing about whether the POST reached the daemon, so a run
        # may be live against it. Named with a typed reason so the sweep's
        # pending-invocation recovery finds it, then re-raised — disclosure, not a swallow.
        _retire_orphaned_registration(ctx, gateway, owned_project_id,
                                      project_persistent=project_persistent,
                                      definite_refusal=False,
                                      reason=f"pre_custody_exit_{type(exc).__name__}",
                                      invocation_id=invocation_id)
        raise
    finally:
        gateway.close()

    metadata = getattr(ctx, "task_metadata", {}) or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    # A start whose custody row did not land does not wear the plain name: the run
    # is live and only THIS process knows it exists (the uncustodied-run leak).
    durable = custody.record_started(drive, _RunCustody(
        run_id=run_id,
        task_id=str(getattr(ctx, "task_id", "") or ""),
        route_id=route.route_id,
        model=route.model,
        profile_id=route.profile_id,  # requested account pin, beside the requested model it mirrors
        project_id=project_id,
        project_owned=bool(owned_project_id),
        project_persistent=project_persistent,
        root_task_id=str(metadata.get("root_task_id") or ""),
        parent_task_id=str(metadata.get("parent_task_id") or ""),
        # The CANONICAL (budget) root, the same one the custody rows themselves live
        # on — never ctx.drive_root, which on a split-root task is the disposable
        # child drive: a ledger row written there misses every budget fence and is
        # erased with the child drive's pruning (P34R.1).
        ledger_root=str(drive),
        idempotency_key=key,
        invocation_id=invocation_id,
        selected_subagent_id=selected_subagent_id,
        config_fingerprint=config_fingerprint,
        work_order_fingerprint=work_order_fingerprint,
        authority_fingerprint=authority_fingerprint,
        snapshot_id=snapshot_id,
        execution_root=execution_root,
        baseline_sha=baseline_sha,
        target_root=target_root,
        authority_source=authority_source,
        resource_ref=resource_ref,
        # The GRANTED shape on the custody OBJECT too, not only the row: the
        # in-process memo answers the same lookups the replay does (R1 item 2).
        access=access,
        mode=authority.mode,
        isolation=authority.isolation,
        delegated=authority.delegated,
    ), shape={
        # The shape rides on the SAME durable row as custody, so a forensic reader never
        # has to join two events to learn what authority a run was started with.
        # `max_seconds` rides too: delegate_wait reports elapsed-vs-cap from this row.
        "effort": route.effort, "access": access, "mode": authority.mode,
        "isolation": authority.isolation, "delegated": authority.delegated, "root": root,
        "execution_root": execution_root,
        "max_seconds": seconds,
        "capture_mode": (_CAPTURE_DELEGATED_SNAPSHOT if snapshot_id else ""),
    })
    return _started_payload(handle, run_id, route, access, authority, root,
                            durable=durable, recovering=recovering,
                            invocation_id=invocation_id,
                            snapshot_id=snapshot_id, execution_root=execution_root,
                            target_root=target_root,
                            baseline_sha=baseline_sha)


def _started_payload(handle: Dict[str, Any], run_id: str, route: Any, access: str,
                     authority: "DelegatedRunShape", root: str, *, durable: bool,
                     recovering: bool, invocation_id: str, snapshot_id: str,
                     execution_root: str, target_root: str, baseline_sha: str) -> str:
    """The one author of delegate_start's started result (note + payload).

    The AUTHORITY guidance and the CUSTODY warning are independent facts about the same
    start, so both are said. An undurable custody row is the louder one and goes first:
    a nanny that walks away from an uncustodied MUTATING run leaves a live shell in its
    own worktree that nothing outside this process can name.
    """
    note = "" if durable else (
        "CUSTODY IS NOT DURABLE: the run started, but its custody row could not be written, "
        "so nothing outside this worker can wait on, cancel or settle it. Do not walk away "
        "from it — finish it or delegate_cancel it in this session. ")
    note += (
        "You are the nanny and the host. Poll with delegate_wait; the run's own "
        "claims are evidence to check, not a verified result."
        + (
            " This run edits a PRIVATE SNAPSHOT of your write root, not the shared "
            "tree: at terminal its diff is captured for you, and NOTHING lands in "
            "the shared tree until you explicitly integrate_delegated_patch(run_id="
            "...) to apply or reject it — read the captured diff before you claim "
            "it, and never let the run commit. It was ASKED to run under a scoped "
            "HOME and an OS-enforced boundary; whether the engine applied either is "
            "a per-run fact that delegate_wait reads back from the run's own "
            "artifacts. A host with no boundary mechanism runs it anyway and says "
            "so there."
            if authority.isolation == "live" else
            " This run cannot write anything: it reads and answers."
        )
    )
    payload = {
        "status": "started" if durable else "started_uncustodied",
        "run_id": run_id,
        "run_dir": handle.get("runDir"),
        "route": route.route_id,
        "model": route.model,
        "effort": route.effort,
        "access": access,
        "mode": authority.mode,
        "isolation": authority.isolation or "envelope",
        "idempotent_recovery": recovering,
        # ASKED, not applied. The proof arrives with the run's own artifacts and is
        # relayed by delegate_wait; saying "isolated" here would be the exact claim
        # this whole verification exists to stop anyone from making.
        "scoped_home_requested": authority.delegated,
        "root": root,
        "custody_durable": durable,
        "invocation_id": str(invocation_id or ""),
        "note": note,
    }
    if not durable:
        payload["pending_invocation_id"] = str(invocation_id or "")
    if snapshot_id:
        # Stable project identity remains the public root; the private execution
        # workspace is disclosed separately and receives no automatic apply.
        payload["execution_root"] = execution_root
        payload["authority_target_root"] = target_root
        payload["baseline_id"] = baseline_sha
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _retire_orphaned_registration(ctx: ToolContext, gateway: Any, project_id: str, *,
                                  definite_refusal: bool, reason: str,
                                  invocation_id: str = "",
                                  snapshot_id: str = "",
                                  project_persistent: bool = False) -> Dict[str, Any]:
    """Retire a registration this start created but never bound to a run.

    Only when the daemon gave a DEFINITE negative answer (a 4xx refusal): a transport
    error, a 5xx, or a 2xx handle with no run id all mean the POST's fate is unknown, and
    a run may well be live against this very registration. An unverified outcome is never
    grounds for destroying state — the durable row names the id either way, which is what
    the old code lacked. The caller supplies the verdict, so every failing start reaches
    this one path instead of one branch retiring and its twin abandoning.

    The row also settles the INVOCATION's fate: ``definite: true`` retires the logical
    invocation id (a definitely refused invocation must not be reused — the daemon may
    hold its key against a body a reconfigured route can no longer reproduce, which
    would 409 forever), while an unknown outcome leaves it pending so a transport retry
    presents the same key and lands on whatever the daemon really has. Written even
    with no registration to retire, because the invocation's fate is its own fact.
    """
    if snapshot_id and definite_refusal:
        # The C1 execution snapshot THIS attempt provisioned. Only a definite refusal
        # proves no run can be live against it; an unknown outcome keeps it — the
        # pending invocation names it durably, and the startup GC reconciles it.
        try:
            from ouroboros.subagent_worktrees import remove_execution_snapshot

            remove_execution_snapshot(snapshot_id)
        except Exception:
            log.warning("Failed to retire delegated execution snapshot %s", snapshot_id,
                        exc_info=True)
    retired = False
    if project_persistent:
        # Stable project registrations intentionally outlive this invocation,
        # including a definite start refusal. The next attempt reuses the same
        # scope.root registration and trust/history identity.
        retired = False
    elif project_id and definite_refusal:
        try:
            gateway.remove_project(project_id)
            retired = True
        except Exception as exc:
            # A registration the daemon does not have is already retired: the same
            # absence-is-discharge fact `retire_project` settles on.
            retired = custody.daemon_says_absent(exc)
            if not retired:
                log.warning("Failed to retire orphaned delegated project %s", project_id, exc_info=True)
    if project_id or invocation_id:
        _emit(ctx, custody.START_FAILED, {"run_id": "", "project_id": project_id,
                                          "project_retired": retired, "reason": reason,
                                          "invocation_id": invocation_id,
                                          "definite": bool(definite_refusal)})
    if not project_id:
        return {"project_retired": False}
    if retired or definite_refusal:
        return {"project_retired": retired, "project_id": project_id}
    return {"project_retired": False, "project_id": project_id,
            "project_retention_reason": "start_outcome_unknown_run_may_exist"}


def _deadline_expired(ctx: ToolContext) -> bool:
    """True when the nanny HAS a deadline and it has already passed.

    The distinction the bound below could not make: ``deadline_remaining_sec`` answers
    0.0 both for "no deadline" and for "the deadline is behind us", and collapsing them
    let an EXPIRED nanny hand a fresh run the absolute task ceiling.
    """
    from ouroboros.deadline_utils import deadline_remaining_sec, parse_deadline_ts

    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    if parse_deadline_ts(meta.get("deadline_at")) is None:
        return False
    return deadline_remaining_sec(ctx) <= 0


def _bounded_max_seconds(ctx: ToolContext, requested: Optional[int]) -> int:
    """Narrow-only: the delegated run may never outlive the nanny's own deadline.

    A caller must ask ``_deadline_expired`` FIRST: an expired deadline cannot produce an
    honest bound at all, and this function's fallback is for a nanny that has NO
    deadline, never for one whose deadline is behind it.
    """
    from ouroboros.deadline_utils import deadline_remaining_sec

    remaining = int(max(0.0, deadline_remaining_sec(ctx)))
    try:
        asked = int(requested) if requested is not None else 0
    except (TypeError, ValueError):
        asked = 0
    candidates = [value for value in (asked, remaining) if value > 0]
    if candidates:
        # Clamp HERE too, not only on the fallback below: `max_seconds` is a model-supplied
        # tool argument with no maximum in its schema, so an explicit ask sailed past the
        # bound the fallback branch was careful about — the same defect, one branch over.
        return min(_CLAUDEXOR_MAX_SECONDS, min(candidates))
    # No positive bound is knowable: either the nanny has no deadline, or its deadline
    # has already passed. Omitting `maxSeconds` — the old behavior — handed the run
    # Claudexor's 7-day schema bound; the cap is damage limitation, and custody (the
    # durable start row plus reconciliation) is what actually stops an orphan.
    from ouroboros.config import get_task_abs_ceiling_sec

    # Claudexor bounds maxSeconds at 7 days (control.ts `.max(604_800)`), and the task
    # ceiling clamps only from BELOW — an owner who raises it past a week would make
    # every deadline-less start send an out-of-schema value.
    return min(_CLAUDEXOR_MAX_SECONDS, int(get_task_abs_ceiling_sec()))


def _halt_breached_run(ctx: ToolContext, gateway: Any, entry: _RunCustody,
                       breach: _Breach) -> str:
    """Stop a run the engine did not contain as asked, and say exactly what failed.

    The BREACH incident goes through ``custody.record_containment_fault``, the same
    writer an unverified cancel uses, so a breached run also surfaces as the CRITICAL
    health invariant that stays open until a terminal receipt resolves it. Emitting a
    look-alike event here instead left the breach out of the open-fault sweep.

    The stop itself goes through ``custody.cancel_and_verify`` — the ONE cancel path,
    with its four typed outcomes — and the sentence handed back to the agent is built
    from the outcome it returns. The ad-hoc cancel this replaced swallowed every
    exception into a log line and then said "The run was cancelled" unconditionally,
    which is precisely what ``record_containment_fault``'s own contract forbids: an
    incident must never surface "as a reassuring string in a tool result". An
    overpowered run that refused to stop was reported to the agent as stopped.
    """
    run_id = entry.run_id
    drive = custody.custody_root(ctx)
    try:
        cancelled = custody.cancel_and_verify(drive, gateway, entry, breach.code)
    except Exception:
        log.warning("Failed to cancel an uncontained delegated run %s", run_id, exc_info=True)
        cancelled = {"outcome": custody.CANCEL_CONTAINMENT_FAULT}
    custody.record_containment_fault(drive, entry, breach.code, breach.detail,
                                     fault=breach.code, **breach.facts)
    outcome = str(cancelled.get("outcome") or custody.CANCEL_CONTAINMENT_FAULT)
    return _fail(
        "delegate_wait", breach.code,
        f"{breach.detail} {_CANCEL_NOTES.get(outcome, '')} Do not retry it: this is a "
        "containment fault in the transport or the engine, not a task failure — report "
        "it and continue within your own authority.",
        run_id=run_id, cancel_outcome=outcome, **breach.facts,
    )


# The typed external-wait lease lives in `delegate_progress` (the wait-liveness
# module); re-bound here because the wait's own seams and the tests name it on
# this surface, exactly like the staged-output cluster above.
_external_wait_lease_until = progress.external_wait_lease_until
_emit_external_wait_lease = progress.emit_external_wait_lease


def _delegate_wait(ctx: ToolContext, run_id: str, wait_sec: Optional[int] = None,
                   since_seq: Optional[int] = None) -> str:
    """Time-bounded, progress-aware wait (docs/DEVELOPMENT.md "Timeout & Wait Control").

    HOLDS the window it was given. It returns early only on a terminal state or a
    containment fault; a journal-cursor advance past ``since_seq`` is RECORDED and
    streamed to the human live, and the model is woken once, at expiry, with the whole
    sequence in ``advances``. Returning on the first advance made the caller's window
    meaningless against a healthy run — the only path that ever consulted it was the
    SILENT one, so a streaming run cost a full-context round per event batch (measured:
    18 rounds, 861k prompt tokens, for a run that was doing fine). Progress is the
    JOURNAL cursor, so SSE ``: ping`` keepalives cannot masquerade as it.

    NARROW-ONLY, like ``_bounded_max_seconds``: the wait may not outlive the nanny's own
    deadline, minus the finalization grace it needs to answer at all. This tool is absent
    from ``_DEADLINE_CLAMPED_TOOLS`` (its ToolEntry value IS its outer bound), so nothing
    upstream cuts it — measured, a 2100s window against ten seconds of remaining deadline
    ran the full 2100s and slid the task past its deadline mid-tool, the defect that set
    built for ``web_search``. Clamping HERE keeps the graceful typed ``no_progress``
    return where the outer clamp delivers a thread-kill. Only "no deadline set" is left
    unclamped; a SPENT deadline clamps to the floor, the window is measured from before
    the connection, and every call is BOUNDED by what it has left (``progress.poll_bound``)
    so no read can outrun it as the 60s default could. Only the LAST poll of a spent
    window may go unanswered gracefully; a daemon that fails while the window still has
    time is the typed refusal it was, never a wait reported as quiet.
    """
    from ouroboros.config import get_delegate_wait_max_sec, get_delegate_wait_sec
    from ouroboros.gateways.claudexor import (
        ClaudexorGateway,
        ClaudexorUnavailable,
        pending_interactions as _cx_pending,
    )

    rid = str(run_id or "").strip()
    if not rid:
        return _fail("delegate_wait", "missing_run_id", "run_id is required")
    not_mine, entry = _owned_run(ctx, "delegate_wait", rid)
    if not_mine or entry is None:
        return not_mine or _fail("delegate_wait", "run_ownership_unknown", "custody unresolved", run_id=rid)
    ceiling = get_delegate_wait_max_sec()
    try:
        window = int(wait_sec) if wait_sec is not None else get_delegate_wait_sec()
    except (TypeError, ValueError):
        window = get_delegate_wait_sec()
    from ouroboros.deadline_utils import parse_deadline_ts, window_within_deadline

    window = window_within_deadline(ctx, max(1, min(window, ceiling)))

    # The clock starts HERE, before the connection: the window is a promise about how
    # long this CALL holds, and the opening handshake plus first poll are part of it.
    # Started after them, an unbounded connection could spend the whole deadline before
    # the window it was clamped into had begun.
    started = time.monotonic()
    deadline = started + window
    try:
        gateway = ClaudexorGateway()
        gateway.handshake(timeout_sec=progress.poll_bound(deadline - time.monotonic()))
    except ClaudexorUnavailable as exc:
        return _fail("delegate_wait", exc.code, str(exc), run_id=rid)

    # The GRANTED shape replays from the durable custody row (R1 item 2): the run
    # was admitted under host-derived authority recorded on its STARTED row, and a
    # top-level payload delegation has no acting/workspace context to re-derive
    # from — re-derivation read `readonly` and cancelled the run as widened on its
    # first wait. Ownership was already proven above (`_owned_run`), and a lost or
    # legacy custody record (no recorded access) keeps the live derivation, so a
    # missing record still cannot become a wider run.
    if entry.access:
        from ouroboros.subagents import DelegatedRunShape as _Shape

        authority = _Shape(access=entry.access, mode=entry.mode,
                           isolation=entry.isolation, delegated=entry.delegated)
    else:
        authority = _derive_authority(ctx)
    # FACTS against premature cancels: how long the run has actually been going and
    # what its cap really is, from the durable start row. A nanny that cannot see
    # these confabulates "exceeded the cap" out of its own impatience. Absent facts
    # (an old row, an unknown run) stay null — never invented.
    _started_ts, _run_max_seconds = custody.run_timing(custody.custody_root(ctx), rid)
    _started_at = parse_deadline_ts(_started_ts)
    # The idle-rail lease for THIS hold: granted before the loop, released in the
    # finally below — the supervisor's idle enforcer spares a leased task while
    # every other rail (deadline, ceiling, budget, cancel) still cuts through.
    # The grant carries a unique lease_id (F5b) and the release names the SAME
    # id, so an abandoned, executor-killed wait thread's late release can never
    # blank a newer grant made by this task's next wait.
    _lease_id = uuid.uuid4().hex
    _emit_external_wait_lease(
        ctx, rid, _external_wait_lease_until(ctx, window, _started_at, _run_max_seconds),
        lease_id=_lease_id)
    try:
        detail = progress.bounded_poll(gateway, rid, deadline - time.monotonic())
        baseline = int(since_seq) if since_seq is not None else int(detail.get("lastSeq") or 0)
        seen = progress.WindowObservations()
        seen.observe_baseline(detail, baseline)
        while True:
            summary = custody.summary_of(detail)
            state = str(summary.get("state") or "")
            last_seq = int(detail.get("lastSeq") or 0)
            breach = _containment_breach(detail, authority)
            if breach:
                return _halt_breached_run(ctx, gateway, entry, breach)
            if state in _TERMINAL_STATES:
                was_settled = bool(entry.settled)
                settlement = custody.settle_run(custody.custody_root(ctx), gateway, entry, detail)
                payload = _delivered_terminal_payload(ctx, rid, detail, authority, entry, gateway)
                payload["settlement"] = settlement
                # C1: a mutating run's changes live in its private snapshot until the
                # nanny explicitly integrates them. Captured HERE, durably, on every
                # terminal observation (idempotent), cancelled runs included — a
                # cancelled run's partial work is salvage material, not garbage.
                capture = _capture_terminal_patch(ctx, entry)
                if capture is not None:
                    payload["workspace_capture"] = capture
                # The «last delegated run» settings receipt (Subagents section):
                # requested vs applied model, written ONLY when THIS call performed
                # a SUCCESSFUL settlement — a later wait re-reading an already-settled
                # run must not re-date it (or replace a newer run as "last"), and a
                # settlement whose durable obligations failed must not mint a receipt
                # it would re-mint on every retry. The delegated REVIEW sessions never
                # pass here — they have their own receipt store
                # (reviewer_slot_last_execution.json).
                if not was_settled and bool(settlement.get("settled")):
                    from ouroboros.subagents import record_last_delegation
                    record_last_delegation(
                        route=entry.route_id, requested_model=entry.model,
                        applied_model=str(payload.get("model") or ""), run_id=rid,
                        selected_subagent_id=entry.selected_subagent_id,
                        # Applied = the settlement receipt's authRoute fact (never invented); requested replays off STARTED.
                        requested_profile=entry.profile_id,
                        applied_profile=str((summary.get("authRoute") or {}).get("profileId") or ""))
                # D7 made load-bearing: settlement is where "paid for and never read"
                # becomes permanent, so the parent is told in WORDS here — not left to
                # infer it from `output_delivery.consumed`. Re-settling an already
                # settled run reports the CURRENT durable fact, so the line disappears
                # once the read has happened rather than echoing a stale omission.
                if custody.record_settled_unread(custody.custody_root(ctx), entry):
                    payload["result_not_collected"] = (
                        "THIS RESULT IS NOT COLLECTED YET: the run is settled and its "
                        "full output is staged, but nothing has read it to EOF. Read the "
                        "artifact named in output_delivery with read_file "
                        "root='task_drive' until it is covered end to end — a result you "
                        "have not read is not a result you may report."
                    )
                # The containment disclosure is read off what the PARENT was told, so the
                # durable line and the relayed payload cannot disagree. It runs on the
                # PREVIEW path too: a payload big enough to spill is exactly the one whose
                # containment block a reader is least likely to reach.
                _record_containment(ctx, entry, payload)
                from ouroboros.tools.control import cache_horizon_note
                _horizon = cache_horizon_note(ctx, time.monotonic() - started)
                if _horizon:
                    payload["cache_horizon_note"] = _horizon
                return json.dumps(payload, ensure_ascii=False, indent=2)
            if last_seq > baseline:
                # The STREAM is not collapsed — the TIMER is. Every advance reaches the
                # live progress surface the instant this loop sees it, so the human's
                # view stays as rich; what stops is waking the MODEL per event batch.
                # The emit is also the frame the supervisor's idle enforcer reads, which
                # a silently blocking wait would starve.
                progress.emit(ctx, rid, seen.record(detail, last_seq, int(time.monotonic() - started)))
                baseline = last_seq          # so the NEXT advance is counted once
            pending = _cx_pending(detail)
            if pending and _interactions_are_news(rid, pending):
                # A NEW question returns IMMEDIATELY: the old wait kept only the
                # waitingOnUser boolean and showed it at window expiry, so a paused
                # run burned the rest of the window (up to the engine's whole
                # answer timeout) in dead metered polling. A question the model
                # ALREADY saw does not re-trigger — a nanny that escalated to its
                # human keeps holding windows instead of busy-looping, and the
                # engine timeout stays the backstop.
                return _waiting_on_user_payload(ctx, rid, state, last_seq, pending,
                                                seen=seen)
            def _expired() -> str:
                rendered = progress.rendered_window(
                    run_id=rid, state=state, last_seq=last_seq, window=window,
                    elapsed_seconds=(None if _started_at is None else max(0, int(
                        (_dt.datetime.now(tz=_dt.timezone.utc) - _started_at).total_seconds()))),
                    max_seconds=_run_max_seconds or None,
                    waiting_on_user=bool(summary.get("waitingOnUser")) or bool(pending),
                    pending_interactions=_bounded_interactions(pending) if pending else None,
                    detail=detail, seen=seen,
                    budget=tool_result_limit("delegate_wait"))
                from ouroboros.tools.control import cache_horizon_note
                _horizon = cache_horizon_note(ctx, time.monotonic() - started)
                return f"{rendered}\n\n{_horizon}" if _horizon else rendered

            if time.monotonic() >= deadline:
                return _expired()
            time.sleep(min(_POLL_INTERVAL_SEC, max(0.0, deadline - time.monotonic())))
            # BOUNDED whether or not the window is spent: a poll STARTED a moment before
            # expiry still carries the client's 60s read default, so the clamp bounded the
            # sleeping and not the waiting. What an UNANSWERED one MEANS is what differs.
            # The last poll of a spent window is bounded and never skipped — terminal
            # state and breach are judged on fresh data or not at all — and a daemon too
            # slow to answer THAT one is this window's expiry. Earlier, the window still
            # has time and there is no expiry to report: the typed refusal propagates to
            # the handler below, because a daemon that died mid-window relayed as a quiet
            # completed wait is a fabricated duration on top of a run nobody is watching.
            left = deadline - time.monotonic()
            fresh = (progress.expiring_poll(gateway, rid) if left <= 0
                     else progress.bounded_poll(gateway, rid, left))
            if fresh is None:
                return _expired()   # unanswered AT expiry: expire on what is already held
            detail = fresh
    except ClaudexorUnavailable as exc:
        return _fail("delegate_wait", exc.code, str(exc), run_id=rid)
    finally:
        _emit_external_wait_lease(ctx, rid, 0.0, lease_id=_lease_id)
        gateway.close()


_CANCEL_NOTES = {
    custody.CANCEL_CONFIRMED: "VERIFIED terminal: the run has stopped. Partial artifacts are "
                              "preserved by Claudexor; a cancelled run has no verdict.",
    custody.CANCEL_REQUESTED: "The daemon ACCEPTED the cancel but the run is not terminal yet. "
                              "It is still running. Call delegate_wait to confirm it stops.",
    custody.CANCEL_FAILED: "The daemon REFUSED the cancel and the run is still live and still "
                           "mutating. Escalate — this is not a stopped run.",
    custody.CANCEL_CONTAINMENT_FAULT: "CONTAINMENT FAULT: the cancel could not be verified, so an "
                                      "overpowered mutating run MAY STILL BE LIVE. A durable "
                                      "incident was recorded and is surfaced as a critical health "
                                      "invariant until a terminal receipt clears it.",
}


def _delegate_cancel(ctx: ToolContext, run_id: str, reason: str = "") -> str:
    """Stop a delegated run. Destructive by nature — a cancelled reviewer has no verdict.

    Reports only what a terminal receipt proves. Saying "cancelled" over an unverified
    control is worse than saying nothing: it retires the operator's attention from a run
    that is still writing to a workspace.
    """
    from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable

    rid = str(run_id or "").strip()
    if not rid:
        return _fail("delegate_cancel", "missing_run_id", "run_id is required")
    not_mine, entry = _owned_run(ctx, "delegate_cancel", rid)
    if not_mine or entry is None:
        return not_mine or _fail("delegate_cancel", "run_ownership_unknown", "custody unresolved", run_id=rid)
    try:
        gateway = ClaudexorGateway()
        gateway.handshake()
    except ClaudexorUnavailable as exc:
        return _fail("delegate_cancel", exc.code, str(exc), run_id=rid)
    try:
        result = custody.cancel_and_verify(custody.custody_root(ctx), gateway, entry, reason)
    finally:
        gateway.close()
    return json.dumps({
        "status": result["outcome"],
        "run_id": rid,
        "run_may_still_be_live": result["outcome"] != custody.CANCEL_CONFIRMED,
        "accepted": result["accepted"],
        "control_status": result["control_status"],
        "state": result["state"],
        "fault_reason": result["fault_reason"],
        "detail": result["detail"],
        "note": _CANCEL_NOTES.get(result["outcome"], ""),
    }, ensure_ascii=False, indent=2)


def get_tools() -> List[ToolEntry]:
    from ouroboros.config import get_task_abs_ceiling_sec

    return [
        ToolEntry("delegate_start", {
            "name": "delegate_start",
            "description": (
                "Start a delegated run on the owner's configured subscription harness and "
                "become its NANNY. Subscription execution is REQUESTED, so the usual case "
                "is no metered API money — but the actual spend is a fact of the finished "
                "run, not a promise of this call: it may come back zero, billed, "
                "estimated, or undisclosed (an expired session, a route that bills by "
                "construction, or an auth fallback all charge real money). Read the "
                "terminal `cost` block from delegate_wait before you treat this as free; "
                "it also costs time, quota and a worker slot. Your working root, "
                "access profile and route come from YOUR task authority; you cannot widen "
                "them, and there is no argument here that would let you try. If you hold "
                "a MUTATING shape the run executes in a PRIVATE SNAPSHOT of your write "
                "root — it never edits the shared tree in place. Its diff is captured at "
                "terminal (delegate_wait's workspace_capture block) and reaches your tree "
                "ONLY when you explicitly call integrate_delegated_patch(run_id=..., "
                "decision='apply'|'reject'); read the captured diff before applying, and "
                "never let the run commit inside its snapshot. If you are read-only it "
                "can only read and answer. "
                "A TOP-LEVEL task may instead select ONE exact installed user-managed "
                "skill payload with root='skill_payload' + bucket + skill_name: the "
                "selector chooses authority you already hold (it grants nothing), the "
                "run edits a private standalone snapshot of that payload, the LIVE "
                "payload stays byte-identical until you explicitly "
                "integrate_delegated_patch, and after an apply the skill's prior "
                "review is stale — run skill_preflight and skill_review as usual. "
                "The payload must already exist (create a NEW skill's manifest first). "
                "Seeded native stays system-repo territory; markerless native is logical external. "
                "Returns a run_id: watch it with delegate_wait, stop it with "
                "delegate_cancel. The run's output is a CLAIM you must check — you are the "
                "host, so verification receipts are still yours to write. If no route is "
                "configured or it is unavailable you get a typed refusal: choose an "
                "explicit configured alternative, wait, narrow, or report blocked. Fresh "
                "starts require subagent_id; recovery retries use retry_of without a new "
                "subagent selector."
            ),
            "parameters": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                "prompt": {"type": "string", "description": "The complete task for the delegated session."},
                "subagent_id": {"type": "string", "description":
                    "Required for a fresh start: exact agent_session actor id from Available "
                    "subagents. Omit when replaying retry_of. API actor ids are refused here "
                    "and must be scheduled as recursive children."},
                "root": {"type": "string", "enum": ["skill_payload"], "description":
                    "Optional exact-resource selector: 'skill_payload' delegates ONE "
                    "installed user-managed skill payload you can already write. Omit "
                    "for ordinary workspace delegation."},
                "bucket": {"type": "string", "description":
                    "With root='skill_payload': the payload location "
                    "(external|clawhub|ouroboroshub|user_repo)."},
                "skill_name": {"type": "string", "description":
                    "With root='skill_payload': the exact skill name."},
                "max_seconds": {"type": "integer", "description":
                    "Wall-clock cap for the run; narrowed to your own remaining deadline. "
                    "Harness runs routinely need 3-5+ minutes end to end, so do not set a "
                    "tight cap for what feels like a quick edit. While delegate_wait shows "
                    "an advancing cursor the run is WORKING, and it enforces this cap "
                    "itself — cancelling a progressing run discards the whole run's spend."},
                "retry_of": {"type": "string", "description":
                    "EXPLICIT retry token: the pending_invocation_id from a start whose "
                    "outcome was unknown (transport failure, lost response). Replays THAT "
                    "invocation byte-identically under its original key, so the engine "
                    "returns the run it already accepted instead of starting a second one. "
                    "Omit subagent_id on this recovery path; supplying both selectors is a "
                    "typed conflict. "
                    "Never set it for an intended new run — a plain call always starts a "
                    "NEW invocation, even with an identical prompt."},
                },
            },
        }, _delegate_start_entry,
           timeout_sec=120),
        ToolEntry("delegate_wait", {
            "name": "delegate_wait",
            "description": (
                "Sleep on a delegated run until a meaningful event. Quiet transport windows "
                "are renewed by the host with zero model calls; journal progress still streams "
                "to the human but does not wake you. Terminal settlement, a new interaction, "
                "fault, addressed owner/task message, cancel/deadline control, or an explicit "
                "one-shot checkpoint wakes exactly once. A run that asks its "
                "user a question returns IMMEDIATELY as status='waiting_on_user' with "
                "the full question set (interaction/question ids ride WHOLE, never "
                "truncated): answer it with delegate_answer, or escalate to your "
                "human and keep waiting (a question with a timeout_at benign-declines "
                "at the engine timeout; timeout_at=null waits until answered). A "
                "large terminal result is delivered as a bounded preview plus an "
                "artifact: read output_delivery and finish reading the artifact before "
                "you rely on it."
            ),
            "parameters": {"type": "object", "required": ["run_id"], "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "since_seq": {"type": "integer", "description": "Event cursor: advances past it are recorded as progress."},
                "checkpoint_after_sec": {"type": "integer", "description":
                    "Optional one-shot future inspection time. Requires checkpoint_reason; "
                    "a real earlier wake consumes it."},
                "checkpoint_reason": {"type": "string", "description":
                    "Why one proactive inspection is worth a model call. No repeating cadence."},
            }},
        }, _delegate_wait_entry, timeout_sec=get_task_abs_ceiling_sec() + 120),
        ToolEntry("delegate_cancel", {
            "name": "delegate_cancel",
            "description": (
                "Cancel a delegated run. Claudexor keeps partial artifacts, but a cancelled "
                "session has no verdict and no finished work product — cancel a stuck or "
                "misdirected run, never one you merely want to hurry. The result is typed: "
                "only `confirmed` means a verified terminal receipt; `requested`, `failed` "
                "and `containment_fault_run_may_still_be_live` all mean it may still be running."
            ),
            "parameters": {"type": "object", "required": ["run_id"], "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "reason": {"type": "string", "description": "Why you are stopping it."},
            }},
        }, lambda ctx, run_id, reason="": _delegate_cancel(ctx, run_id, reason), timeout_sec=120),
        ToolEntry("delegate_answer", {
            "name": "delegate_answer",
            "description": (
                "Answer a delegated run's pending interactive question — the "
                "status='waiting_on_user' payload from delegate_wait names the "
                "interaction_id and its questions. Only the task that started the run "
                "may answer. Policy: answer from the task context you already hold; a "
                "question ABOVE your authority (spending money, changing scope, "
                "external actions) is not yours to guess — surface it to your human "
                "via progress and keep waiting; an unanswered question with a "
                "timeout_at benign-declines at the engine timeout (the run continues "
                "on stated assumptions), while timeout_at=null waits until answered. "
                "Typed outcomes: delivered; already_resolved (the run moved on — do "
                "not re-post); not_found; rejected (a definite engine refusal of "
                "these rows — HTTP 400/409/413/422 only — fix them); "
                "subscription_window_exhausted (a distinct outcome carrying reset_at "
                "— the answer did NOT land; retry the SAME answers after reset_at); "
                "delivery_unknown "
                "(transport died mid-answer — re-check with delegate_wait and NEVER "
                "post a different answer for the same interaction). Codex-lane runs "
                "have no mid-run questions: a run that ENDS needing input "
                "(outcome_facts.reason=input_required) is answered with a plain NEW "
                "delegate_start(subagent_id=..., prompt=...) whose prompt carries the "
                "assignment plus the answers "
                "— there is no rerun/decision verb, and custody stays with you."
            ),
            "parameters": {"type": "object",
                           "required": ["run_id", "interaction_id", "answers"],
                           "properties": {
                "run_id": {"type": "string", "description": "Run id from delegate_start."},
                "interaction_id": {"type": "string", "description":
                    "The interaction being answered, from the waiting_on_user payload."},
                "answers": {"type": "array", "items": {"type": "object", "properties": {
                    "question_id": {"type": "string", "description":
                        "The question's id from the waiting_on_user payload."},
                    "selected_labels": {"type": "array", "items": {"type": "string"},
                                        "description": "Labels of the chosen option(s)."},
                    "free_text": {"type": "string", "description":
                        "Free-text answer; omit when options were selected."},
                }, "required": ["question_id"]}, "description":
                    "One row per question you are answering."},
            }},
        }, lambda ctx, run_id, interaction_id, answers: _delegate_answer(
            ctx, run_id, interaction_id, answers), timeout_sec=120),
    ]


__all__ = ["get_tools"]
