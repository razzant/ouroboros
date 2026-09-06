"""Rendering for the plan-review engine: the wave view, the next-step guidance and the
one host-owned control line. Split from ``plan_review.py`` so the engine stays under the
size target; no behaviour lives here that the engine does not dictate."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ouroboros.tools.review_synthesis import PLAN_REVIEW_CONTROL_PREFIX
from ouroboros.tools.plan_spec import MAX_FINDINGS_PER_SLOT


# B2 (honest DEGRADED): every aggregate reaches the control line as itself — the
# old DEGRADED→REVIEW_REQUIRED laundering hid the no-quorum fact from the agent.
_CONTROL_OUTCOME = {
    "GREEN": "GREEN", "REVIEW_REQUIRED": "REVIEW_REQUIRED",
    "REVISE_PLAN": "REVISE_PLAN", "DEGRADED": "DEGRADED",
}


# The closed control vocabulary of the ONE host-owned footer line.
# B2 (honest DEGRADED): the no-quorum aggregate is a legal, always-OPEN control
# outcome — the render layer no longer launders it into REVIEW_REQUIRED.
_PLAN_REVIEW_OUTCOMES = frozenset({"GREEN", "REVIEW_REQUIRED", "REVISE_PLAN", "DEGRADED"})


def wave_control_state(wave: dict) -> tuple[str, bool]:
    """The host-owned control projection of one recorded wave.

    The rendered ``PLAN_REVIEW_CONTROL_JSON`` line and the native ToolResult
    metadata (D02) both read THIS pair, so the text a human sees and the
    structured control the loop trusts can never diverge."""
    return (
        _CONTROL_OUTCOME.get(str(wave.get("aggregate") or ""), "REVIEW_REQUIRED"),
        bool(wave.get("closed")),
    )


def _parse_plan_review_control(text: str) -> tuple[str, bool] | None:
    """Parse one exact host-owned plan-review control marker fail-closed."""
    markers = [
        line[len(PLAN_REVIEW_CONTROL_PREFIX):]
        for line in str(text or "").splitlines()
        if line.startswith(PLAN_REVIEW_CONTROL_PREFIX)
    ]
    if len(markers) != 1:
        return None

    def _unique_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(markers[0], object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"outcome", "closed"}:
        return None
    outcome = str(payload.get("outcome") or "")
    closed = payload.get("closed")
    if outcome not in _PLAN_REVIEW_OUTCOMES or type(closed) is not bool:
        return None
    if (outcome == "GREEN" and not closed) or (outcome in {"REVISE_PLAN", "DEGRADED"} and closed):
        return None
    return outcome, closed


def _quote_control_lines(text: str) -> str:
    """Keep reviewer/agent text visible without impersonating the host control footer."""
    return "".join(
        "> " + line if line.startswith(PLAN_REVIEW_CONTROL_PREFIX) else line
        for line in str(text or "").splitlines(keepends=True)
    )



def _actor_outcome(actor: dict) -> str:
    """``ok``, or a FAILED tail led by the typed facts when the record carries them
    (B1: ``FAILED[code] (resets …): prose``). Without a typed code the prose renders
    exactly as before — rows from pre-typed engines lose nothing."""
    if actor.get("ok"):
        return "ok"
    code = str(actor.get("failure_code") or "")
    if not code:
        return "FAILED: " + str(actor.get("error"))
    reset = str(actor.get("reset_at") or "")
    return (f"FAILED[{code}]" + (f" (resets {reset})" if reset else "")
            + ": " + str(actor.get("error")))


def _degraded_replay_note(wave: dict) -> str:
    """The honest replay mechanics of one recorded DEGRADED wave (aligned with the
    engine's `plan_wave_replay_decision`): a wave with structural snapshot evidence
    replays while its epoch and the reviewer roster stand; one without (its slots
    died at dispatch time, invisible to the pre-fan-out snapshot) never replays —
    a transient death is never cached as structural."""
    if wave.get("health_epoch"):
        return (
            "an identical envelope replays this recorded result at no further cost while "
            "the recorded lane-health epoch and the reviewer roster stand (a healed or "
            "newly dead lane, or a changed roster, re-dispatches)"
        )
    return (
        "no structural lane evidence was recorded for this wave (its slots failed at "
        "dispatch time, invisible to the pre-fan-out health snapshot), so an identical "
        "envelope re-dispatches a fresh panel — a transient death is never cached as structural"
    )


def _next_step(wave: dict, *, enforcement: str, cap: Optional[int], cycles_paid: int) -> str:
    aggregate = str(wave.get("aggregate") or "")
    fp = str(wave.get("request_fingerprint") or "")
    at_cap = cap is not None and cycles_paid >= cap
    if bool(wave.get("closed")):
        return "Closed: proceed with the reviewed spec."
    if bool(wave.get("custody_pending")):
        return (
            "Open: one or more paid reviewer operations are still in flight. "
            "The responses received so far are not final authority; wait for "
            "custody reconciliation before treating this wave as closed."
        )
    if aggregate == "DEGRADED":
        # B2: facts, not a retry coach (BIBLE P5 — the host never dictates the next tool
        # call). Quorum arithmetic, per-slot typed states above, and the replay mechanics;
        # the decision (revise the spec, wait, escalate, proceed if permitted) is the LLM's.
        counts = wave.get("counts") if isinstance(wave.get("counts"), dict) else {}
        text = (
            f"DEGRADED: parseable reviewer verdicts {counts.get('parseable', 0)} of "
            f"{counts.get('configured', 0)} configured slot(s) — below the review quorum "
            f"({counts.get('quorum', '?')}). Per-slot typed states (code and reset time, when "
            "known) are listed under Reviewer slots above. This wave is recorded and OPEN; "
            f"{_degraded_replay_note(wave)}; a changed spec starts the next paid cycle. "
        )
        if wave.get("quorum_unreachable"):
            # Naming asymmetry, on purpose: the wave fact is the bare
            # `quorum_unreachable` (scoped by the record it sits on); the task-level
            # typed reason is outcomes.REASON_REVIEW_QUORUM_UNREACHABLE
            # ("plan_review_quorum_unreachable") — surface-prefixed because it
            # travels task-wide. Do not "align" one to the other.
            dead = ", ".join(str(s) for s in wave.get("structurally_dead_slots") or [])
            reset = str(wave.get("earliest_reset") or "")
            text += (
                f"Quorum is STRUCTURALLY unreachable for this wave: slot(s) {dead} are "
                "window-spent, leaving fewer live slots than the quorum"
                + (f"; earliest recorded reset {reset}" if reset else "")
                + ". "
            )
    elif aggregate == "REVIEW_REQUIRED":
        blocking = [f for f in wave.get("findings") or [] if f.get("class") == "blocking"]
        text = (
            "Disposition every finding id (accept | reject | defer, each with a rationale) in ONE "
            f"call: plan_task(review_disposition={{review_fingerprint: '{fp}', items: [...]}}) — no "
            "reviewer call, no cycle. "
        )
        if blocking:
            ids = ", ".join(str(f.get("finding_id") or f.get("id")) for f in blocking[:4])
            text += (
                f"NOTE: {len(blocking)} BLOCKING finding(s) below quorum ({ids}) stay OPEN whatever "
                "you disposition — a blocking finding closes only through a changed spec "
                "(new fingerprint, next paid cycle) or a reject that the next paid delta cycle judges. "
            )
    else:
        text = (
            "Blocking findings: accept ⇒ change the spec and re-call plan_task (new fingerprint, "
            f"{'the cap is reached — no further paid cycle' if at_cap else 'next paid cycle ' + str(cycles_paid + 1) + ('' if cap is None else f' of {cap}')}); "
            "reject ⇒ record reject + rationale via review_disposition naming this fingerprint — it "
            "rides into the next paid delta cycle where reviewers mark it resolved or still-open. "
            "A disposition never closes REVISE_PLAN. "
        )
    if enforcement == "blocking":
        text += (
            "Blocking enforcement: the review must close before the work starts"
            + (" — the cycle cap is reached: exits are owner unstick (Swarm/hurry) or finalizing "
               "with outcome_tier=blocked_with_evidence." if at_cap else ".")
        )
        if wave.get("quorum_unreachable") and not bool(wave.get("closed")):
            # B2b facts, never imperatives: the honest exits that exist alongside
            # each other while the quorum stays structurally unreachable.
            text += (
                " With the quorum structurally unreachable, finalization is RELEASED: "
                "finalizing now records outcome_tier=blocked_with_evidence with the review "
                "left OPEN and implementation still held. Waiting is also open — a one-shot "
                "deferred follow-up can be registered through schedule_followup for the "
                "earliest reset — as is asking the owner."
            )
    else:
        text += (
            "Advisory enforcement: you may proceed with the review OPEN; the host discloses "
            "that loudly in the task result."
        )
    return text



def _render_wave(
    wave: dict, *, cap: Optional[int], cycles_paid: int, enforcement: str,
    cached: bool = False, notes: Optional[List[str]] = None, reminder: str = "",
) -> str:
    aggregate = str(wave.get("aggregate") or "")
    closed = bool(wave.get("closed"))
    counts = wave.get("counts") if isinstance(wave.get("counts"), dict) else {}
    manifest = wave.get("evidence_manifest") if isinstance(wave.get("evidence_manifest"), dict) else {}
    lines = [
        f"## Plan Review — cycle {wave.get('cycle_index')} · paid cycles {cycles_paid}"
        + ("" if cap is None else f"/{cap}") + f" · enforcement {enforcement}",
        "",
        f"**Plan fingerprint:** `{wave.get('request_fingerprint') or ''}`"
        + ("  (cached exact review — no reviewer was called)" if cached else ""),
        f"**Constitutional:** {'yes' if wave.get('constitutional') else 'no'} — {wave.get('constitutional_note') or ''}",
        f"**Evidence:** {len(manifest.get('attached') or [])} attached; omissions: "
        + (", ".join(f"{o.get('locator')}: {o.get('reason')}" for o in manifest.get("omissions") or []) or "none"),
    ]
    if wave.get("compact"):
        ref = wave.get("wave_artifact") if isinstance(wave.get("wave_artifact"), dict) else {}
        artifact_path = str(ref.get("path") or "")
        detail = (
            "its exact findings remain in the immutable task artifact. "
            f"Exact wave: read_file({artifact_path})."
            if artifact_path else
            "legacy exact bytes are unavailable; this summary is not disposition authority."
        )
        lines += ["", f"(bounded hot history: this wave is a compact summary; {detail})"]
    if reminder:
        lines += ["", "⚠️ " + reminder]
    if wave.get("custody_pending"):
        lines += [
            "", "⚠️ REVIEW CUSTODY PENDING: the received quorum is provisional; "
            "a paid reviewer operation is still in flight and this wave remains open."
        ]
    elif aggregate == "DEGRADED":
        # Banner aligned with _next_step: the replay promise depends on whether the
        # wave carries structural snapshot evidence (see _degraded_replay_note).
        lines += ["", "⚠️ DEGRADED: no parseable reviewer quorum — recorded as an OPEN wave; "
                  + _degraded_replay_note(wave) + "."]
    actor_lines = [
        f"- {a.get('slot_id')} · {a.get('model')} · {a.get('route')} · host_file_read: "
        f"{a.get('host_file_read_attestation')} · {_actor_outcome(a)}"
        + (f" · disclosures: {', '.join(a['disclosures'])}" if a.get("disclosures") else "")
        for a in wave.get("actors") or []
    ] or ["(no actor records)"]
    findings = list(wave.get("findings") or [])
    findings_total = int(wave.get("findings_total") or len(findings))
    finding_page = findings[:MAX_FINDINGS_PER_SLOT]
    lines += [
        "", "### Reviewer slots", "", *actor_lines,
        "", "### Findings (per slot; finding_id = slot:id)", "", "```json",
        json.dumps(finding_page, ensure_ascii=False, indent=2, default=str), "```",
    ]
    if findings_total > len(finding_page):
        ref = wave.get("wave_artifact") if isinstance(wave.get("wave_artifact"), dict) else {}
        lines += [
            "",
            f"Rendered finding page: 1-{len(finding_page)} of {findings_total}. "
            f"Exact immutable wave: read_file(root='{ref.get('root') or 'artifact_store'}', "
            f"path='{ref.get('path') or ''}').",
        ]
    previews = [a for a in wave.get("actors") or [] if a.get("raw_text_preview")]
    if previews:
        lines += ["", "### Unparseable reviewer output (bounded preview)", ""]
        for actor in previews:
            lines += [f"#### {actor.get('slot_id')}", _quote_control_lines(str(actor.get("raw_text_preview"))), ""]
    lines += [
        "", f"### Aggregate: {aggregate}" + (" (closed)" if closed else " (open)"),
        "", "Reasons: " + (", ".join(str(r) for r in wave.get("reasons") or []) or "none")
        + f". Counts: {json.dumps(counts, sort_keys=True)}",
    ]
    if wave.get("dispositions"):
        lines += ["", "### Dispositions", "", "```json",
                  json.dumps(wave.get("dispositions"), ensure_ascii=False, indent=2), "```"]
    if wave.get("closure_notes") or notes:
        lines += ["", "Closure notes: " + "; ".join([*(wave.get("closure_notes") or []), *(notes or [])])]
    outcome, closed = wave_control_state(wave)
    lines += [
        "", "## Plan Review Contract", "",
        _next_step(wave, enforcement=enforcement, cap=cap, cycles_paid=cycles_paid), "",
        PLAN_REVIEW_CONTROL_PREFIX + json.dumps({"outcome": outcome, "closed": closed}, separators=(",", ":")),
    ]
    return "\n".join(lines)
