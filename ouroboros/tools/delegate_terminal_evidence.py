"""The terminal story of ONE delegated run, as the parent reads it.

Containment breach detection and evidence, the terminal payload with its access
evidence and reported cost, and whole-or-declared-partial delivery. Extracted
from ``ouroboros/tools/delegate.py`` at its size gate (v7 DEL1 split);
``tools.delegate`` re-exports every name (same objects), so the wait loop, the
tests and monkeypatch targets keep addressing them on THAT surface.

The v7 ledger named this leaf ``tools/delegate_terminal.py``; it lands as
``delegate_terminal_evidence.py`` because upstream already owns
``ouroboros/delegate_terminal.py`` (the terminal reconciliation boundary) and
two neighbouring modules answering to one name would be a permanent grep trap
(owner fork F-2=A; the rename is recorded in the carried ledger).

Every parent-scope name the moved bodies read at call time is DECLARED and read
through ``_delegate()`` — the parent monolith keeps the one rebindable binding
per member, so patches on the historical surface keep their teeth.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - annotation-only names, lazy under future annotations
    from ouroboros.delegate_containment import _Breach
    from ouroboros.delegate_custody import RunCustody as _RunCustody
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.registry import ToolContext


def _delegate():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros.tools import delegate

    return delegate


def _containment_breach(detail: Dict[str, Any], authority: "DelegatedRunShape") -> Optional[_Breach]:
    """Everything the ENGINE enforced, checked against what the host asked for.

    ONE reader for both halves of containment — the access profile and the harness
    HOME — because they fail identically: the request is only a request, the engine
    derives the truth, and a verification written for one half leaves the other
    trusting an echo. The HOME half is asked only of a run that carried the marker;
    a read-only child is scoped by Claudexor's ordinary envelope and asks for nothing.
    """
    widened = _delegate()._widened_access(detail, authority.access)
    if widened:
        return _delegate()._Breach(
            "access_profile_widened",
            f"The delegated run was enforced at access profile {widened!r} while this "
            f"task is only entitled to {authority.access!r}.",
            {"entitled_access": authority.access, "effective_access": widened},
        )
    if authority.delegated:
        return _delegate()._home_isolation_breach(detail)
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

    attempts = attempt_containment(str(_delegate().custody.summary_of(detail).get("runDir") or ""))
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
    nested = _delegate().home_nested_under_operator_home(detail)
    report = {"verified": False, "attempts": len(attempts), "disclosed": disclosed,
              "os_boundary": boundary, "nested_under_operator_home": nested}
    if unavailable_reasons:
        report["confinement_unavailable_reason"] = "; ".join(unavailable_reasons)
    breach = _delegate()._home_isolation_breach(detail)
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
    from ouroboros.gateways.claudexor import final_attempt_facts

    summary = _delegate().custody.summary_of(detail)
    observed = final_attempt_facts(detail, run_id)
    payload = {
        "status": "terminal",
        "run_id": run_id,
        "state": str(summary.get("state") or ""),
        # Model, harness and profile come from the SAME final attempt. The
        # summary's requested model and cross-attempt route are not evidence.
        "model": observed.get("model", ""),
        "observed_attempt": observed,
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
    summary = _delegate().custody.summary_of(detail)
    effective = str(summary.get("effectiveAccess") or "")
    state = str(summary.get("state") or "")
    report = {"requested": expected, "effective": effective,
              "verified": bool(effective), "state": state}
    if effective:
        return report
    if state in _delegate().custody.SUCCEEDED_STATES:
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
    _delegate()._emit(ctx, _delegate().custody.UNCONFINED, {
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
    spend, estimated = _delegate().custody.disclosed_spend(summary)
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
    if entry is not None:
        _delegate().add_terminal_source_verification(full, entry)
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
    primary, full_ok, full_note = _delegate()._resolve_full_primary_output(
        gateway, run_id, full.get("primary_output"))
    full["primary_output"] = primary
    budget = _delegate().tool_result_limit("delegate_wait")
    text = json.dumps(full, ensure_ascii=False, indent=2)
    if len(text) <= budget - _delegate()._PAYLOAD_ENVELOPE_HEADROOM:
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
    artifact = _delegate()._stage_full_output(ctx, run_id, text)
    _delegate()._emit(ctx, _delegate().custody.OUTPUT_SPILLED, {"run_id": run_id, "total_chars": len(text),
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
    return _delegate()._preview_payload(full, text, artifact, budget,
                            consumed=bool(entry is not None and entry.output_consumed),
                            full_ok=full_ok, full_note=full_note)
