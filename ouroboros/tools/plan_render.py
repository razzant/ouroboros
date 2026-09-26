"""Rendering for the plan-review engine: the wave view, the next-step guidance and the
one host-owned control line. Split from ``plan_review.py`` so the engine stays under the
size target; no behaviour lives here that the engine does not dictate."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ouroboros.task_results import plan_review_notes_are_annotatable
from ouroboros.tools.review_synthesis import PLAN_REVIEW_CONTROL_PREFIX
from ouroboros.tools.plan_spec import MAX_FINDINGS_PER_SLOT
from ouroboros.tools.review_helpers import review_enforcement_blocks


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



def _actor_outcome(actor: dict, slot_class: str = "") -> str:
    """``ok``, or a FAILED tail led by the typed facts when the record carries them
    (B1: ``FAILED[code] (resets …): prose``). Without a typed code the prose renders
    exactly as before — rows from pre-typed engines lose nothing. ``slot_class`` is the
    row's ``plan_wave_slot_census`` class: a slot with no collected answer is a gap, so
    it never wears the FAILED form (a planned wait omits its window prose, which reads
    like a timeout; an unresolved custody state keeps its typed state and prose)."""
    if slot_class == "awaiting":
        return f"NO ANSWER YET ({actor.get('operation_state')})"
    if slot_class == "settled_late":
        return "SETTLED LATE — its answer is under Historical feedback below"
    if slot_class == "uncollected":
        return "SETTLED — not collected yet"
    if slot_class == "unresolved":
        return f"NO ANSWER — {actor.get('operation_state')}: {actor.get('error')}"
    if actor.get("ok"):
        return "ok"
    code = str(actor.get("failure_code") or "")
    reset = str(actor.get("reset_at") or "")
    cause = str(actor.get("reported_cause") or "")
    if cause:
        # The engine's reported sentence, quoted, in place of its retry coach: the mind
        # reads typed facts (code, reset, model, the words), never ``nextActions``.
        return (f"FAILED[{code or 'none'}]" + (f" (resets {reset})" if reset else "")
                + f' — model={actor.get("model")}; reported cause: "{cause}"')
    if not code:
        return "FAILED: " + str(actor.get("error"))
    return (f"FAILED[{code}]" + (f" (resets {reset})" if reset else "")
            + ": " + str(actor.get("error")))


def _degraded_replay_note(wave: dict, *, paid_available: bool = True) -> str:
    """The honest replay mechanics of one recorded DEGRADED wave (aligned with the
    engine's `plan_wave_replay_decision`): a wave with structural snapshot evidence
    replays while its epoch and the reviewer roster stand; one without (its slots
    died at dispatch time, invisible to the pre-fan-out snapshot) never replays —
    a transient death is never cached as structural. Every re-dispatch asks every callable
    slot as the next paid cycle: the engine has no failed-slots-only path."""
    whole = ("every callable slot is asked again as the next paid cycle, slots that already answered "
             "included (health-skipped lanes stay $0 rows); no failed-slots-only path exists")
    if not paid_available:
        replay = ("an identical envelope can replay this result for free while its health epoch and roster stand; "
                  if wave.get("health_epoch") else "no structural lane evidence was recorded; ")
        return replay + "the cycle cap is reached, so neither an identical nor revised request can start another paid panel, even after lane recovery"
    if wave.get("health_epoch"):
        return (
            "an identical envelope replays this recorded result at no further cost while "
            "the recorded lane-health epoch and the reviewer roster stand (a healed or "
            f"newly dead lane, or a changed roster, re-dispatches: {whole})"
        )
    return (
        "no structural lane evidence was recorded for this wave (its slots failed at "
        "dispatch time, invisible to the pre-fan-out health snapshot), so an identical "
        f"envelope re-dispatches a fresh panel ({whole}) — a transient death is never cached as structural"
    )


# The enforcement facts every OPEN state ends with; one wording for the settled
# and the custody-pending contract.
_BLOCKING_HOLDS = "Blocking enforcement: the review must close before the work starts"
_ADVISORY_PROCEEDS = (
    "Advisory enforcement: you may proceed with the review OPEN; the open review "
    "stays typed in this task's state and result either way, and your own final "
    "answer is where it is said in words."
)


def _quorum_unreachable_fact(wave: dict) -> str:
    # Naming asymmetry, on purpose: the wave fact is the bare
    # `quorum_unreachable` (scoped by the record it sits on); the task-level
    # typed reason is outcomes.REASON_REVIEW_QUORUM_UNREACHABLE
    # ("plan_review_quorum_unreachable") — surface-prefixed because it
    # travels task-wide. Do not "align" one to the other.
    dead = ", ".join(str(s) for s in wave.get("structurally_dead_slots") or [])
    reset = str(wave.get("earliest_reset") or "")
    return (
        f"Quorum is STRUCTURALLY unreachable for this wave: slot(s) {dead} are "
        "window-spent, leaving fewer live slots than the quorum"
        + (f"; earliest recorded reset {reset}" if reset else "")
        + ". "
    )


def _next_step(wave: dict, *, enforcement: str, cap: Optional[int], cycles_paid: int) -> str:
    aggregate = str(wave.get("aggregate") or "")
    fp = str(wave.get("request_fingerprint") or "")
    at_cap = cap is not None and cycles_paid >= cap
    author = wave.get("author_disposition")
    author_note = ""
    if isinstance(author, dict) and author.get("disposition") and author.get("rationale"):
        author_note = (
            f"Author finish recorded as {author.get('disposition')} against this exact "
            "review fingerprint; raw reviewer findings remain evidence. "
        )
        if review_enforcement_blocks(enforcement):
            author_note += "Blocking enforcement still holds the open plan gate. "
        elif not review_enforcement_blocks("blocking"):
            author_note += "Cyber Pro preserves final judgment with Ouroboros. "
        else:
            author_note += "Advisory enforcement permits proceeding with the review open. "
    cyber = not review_enforcement_blocks("blocking")
    if cyber and not wave.get("custody_pending"):
        return (
            author_note + "Cyber Pro: Ouroboros decides whether and how to continue. "
            "The recorded verdict, open findings and any unresolved physical reviewers remain "
            "independent facts; continuation does not close the wave or create a PASS. "
            f"The existing $0 plan_task(review_disposition={{review_fingerprint: '{fp}', items: [...]}}) "
            "can collect results or record a disposition without a new panel."
        )
    if bool(wave.get("closed")):
        if plan_review_notes_are_annotatable(wave):
            return (
                "Closed: proceed with the reviewed spec. Notes are optional; you may record "
                "accept | reject | defer with a rationale through "
                f"plan_task(review_disposition={{review_fingerprint: '{fp}', items: [...]}}). "
                "This neither reopens the review nor calls reviewers or consumes a cycle."
            )
        return "Closed: proceed with the reviewed spec."
    if bool(wave.get("custody_pending")):
        # Facts about the route that exists (B2), not an instruction to take it: the
        # settlement frame is a mailbox message, so the ordinary in-task wait returns
        # on it. A followup would mint a NEW root task, which cannot collect this wave.
        # Custody-first on purpose: this branch never reaches the settled-wave advice
        # below, so a wave that merely awaits is told neither a replay recipe nor an exit.
        from ouroboros.tools.plan_review_runtime import plan_wave_slot_census

        census = plan_wave_slot_census(wave)
        lead = (
            "one or more reviewer operations have no recorded answer and their custody is "
            "unresolved (typed state under Reviewer slots above)"
            if census["unresolved"] and not census["awaiting"] else
            "one or more reviewer operations are still in flight"
        )
        text = (
            f"Open: {lead}. No reviewer verdict exists yet: "
            "the responses received so far are not final authority, and this wave is not closed "
            "before custody reconciliation completes. The host writes ONE "
            "message into this task's mailbox when every released slot settles: wait_task on "
            "this task's own id (wait_tasks while children run) returns on it, and the $0 "
            f"plan_task(review_disposition={{review_fingerprint: '{fp}', items: []}}) then "
            "collects this wave without a second panel. The same $0 call at any earlier moment "
            "is legitimate: it returns what has settled, never waits and never re-dispatches. "
            "Answers that settle after this task ends are kept in the durable record and do not "
            "close this wave: only a collecting call reads them into it. "
        )
        answered, total = len(census["answered"]), census["configured"]
        if total and answered < total:  # ONE typed count line from the census; omitted once every slot answered
            dead = [f"{len(rows)} {label}" for label, rows in
                    (("did not answer", census["failed"]), ("not sent", census["skipped"])) if rows]
            text += f"Reviewers: {answered} of {total} answered{'; ' + ', '.join(dead) if dead else ''}. "
        unreachable = bool(wave.get("quorum_unreachable"))  # typed window-spent lanes: no awaited answer restores the quorum
        if unreachable:
            text += _quorum_unreachable_fact(wave)
        if cyber:
            return text + ("Cyber Pro: Ouroboros decides whether and how to continue; "
                           "continuation does not close the wave or create a PASS.")
        if enforcement != "blocking":
            return text + _ADVISORY_PROCEEDS
        # The gate's own release fact, stated without a route: finalization stops being
        # refused for this wave whether or not a slot is still awaited.
        return text + _BLOCKING_HOLDS + "." + (
            " With the quorum structurally unreachable, finalization is RELEASED even while a slot is "
            "awaited: finalizing now records outcome_tier=blocked_with_evidence with the review left "
            "OPEN and implementation still held." if unreachable else "")
    if aggregate == "DEGRADED":
        # B2: facts, not a retry coach (BIBLE P5 — the host never dictates the next tool
        # call). Quorum arithmetic, per-slot typed states above, and the replay mechanics;
        # the decision (revise the spec, wait, escalate, proceed if permitted) is the LLM's.
        counts = wave.get("counts") if isinstance(wave.get("counts"), dict) else {}
        text = author_note + (
            f"DEGRADED: parseable reviewer verdicts {counts.get('parseable', 0)} of "
            f"{counts.get('configured', 0)} configured slot(s) — below the review quorum "
            f"({counts.get('quorum', '?')}). Per-slot typed states (code and reset time, when "
            "known) are listed under Reviewer slots above. This wave is recorded and OPEN; "
            f"{_degraded_replay_note(wave, paid_available=not at_cap)}. "
            + ("A changed spec may start another paid cycle. " if not at_cap else "")
        )
        if wave.get("quorum_unreachable"):
            text += _quorum_unreachable_fact(wave)
    elif aggregate == "REVIEW_REQUIRED":
        blocking = [f for f in wave.get("findings") or [] if f.get("class") == "blocking"]
        text = author_note + (
            "Notes are optional. Open need_evidence requests (an evidence locator or "
            "a question addressed to you by spec id) close with ONE $0 call: "
            f"plan_task(review_disposition={{review_fingerprint: '{fp}', items: [...]}}) — accept = "
            "answered (your rationale is the answer; "
            + ("the cap leaves no further paid cycle to deliver it to reviewers), " if at_cap else
               "it reaches reviewers on the next paid cycle), ")
            + "reject, or defer = deferred openly; no reviewer call, no cycle. A revised envelope "
            "supersedes this wave and its open requests can no longer be dispositioned. "
        )
        if blocking:
            ids = ", ".join(str(f.get("finding_id") or f.get("id")) for f in blocking[:4])
            if enforcement == "advisory":  # the closure table's per-finding rule, stated as a fact
                text += (
                    f"NOTE: {len(blocking)} BLOCKING finding(s) below quorum ({ids}): a reject with its "
                    "rationale closes each one; accept or defer keeps it open until a changed spec is reviewed. "
                    + ("" if not at_cap else "The cycle cap is reached; no further paid panel is available. ")
                )
            else:
                text += (
                    f"NOTE: {len(blocking)} BLOCKING finding(s) below quorum ({ids}) stay OPEN whatever "
                    "you disposition: a changed spec, or a justified rejection judged in another paid "
                    "cycle, closes them. "
                    + ("" if not at_cap else "The cycle cap is reached; no further paid panel is available. ")
                )
    else:
        text = author_note + (
            "Blocking findings: accept ⇒ change the spec and re-call plan_task (new fingerprint, "
            f"{'the cap is reached — no further paid cycle' if at_cap else 'next paid cycle ' + str(cycles_paid + 1) + ('' if cap is None else f' of {cap}')}); "
            "reject ⇒ record reject + rationale via review_disposition naming this fingerprint — "
            + ("it is recorded as evidence; the cap leaves no further paid delta cycle to judge it. "
               if at_cap else
               "it rides into the next paid delta cycle where reviewers mark it resolved or still-open. ")
            + "A disposition never closes REVISE_PLAN. "
        )
    if enforcement == "blocking":
        text += (
            _BLOCKING_HOLDS
            + (" — the cycle cap is reached: exits are owner unstick (Swarm/hurry), a revised spec "
               "once the owner raises OUROBOROS_REVIEW_MAX_CYCLES, or finalizing with "
               "outcome_tier=blocked_with_evidence." if at_cap else ".")
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
        text += _ADVISORY_PROCEEDS
    return text



def _closure_note_view(note: str) -> str:
    """Legacy host notes describe state; the current renderer owns available steps."""
    prefix = str(note).partition(":")[0]
    meaning = {
        "closed_by_disposition": "the open set emptied; the wave is recorded GREEN",
        "blocking_finding_below_quorum_stays_open": "blocking findings remain open after disposition",
        "revise_plan_not_closable_by_disposition": "disposition does not close blocking findings",
        "degraded_not_closable_by_disposition": "no parseable reviewer quorum; disposition does not close the wave",
    }.get(prefix)
    return f"{prefix}: {meaning}" if meaning else str(note)


def _dialogue_source_view(wave: dict, *, cached: bool) -> list[str]:
    """Expose recorded context, without re-reading or judging later messages."""
    own = (wave.get("evidence_manifest_full") or {}).get("own_dialogue") or {}
    source = wave.get("dialogue_source_ref") or own.get("source_ref") or {}
    if not own and not source:
        return []
    if own.get("gap"):
        return [f"**Own-room dialogue:** unavailable ({own['gap']})."]
    coverage = {key: {field: len(value) if field == "generations" else value
                      for field, value in section.items()} if isinstance(section, dict) else section
                for key, section in (own.get("coverage") or {}).items()}
    rows = [f"**Dialogue snapshot:** `{source.get('sha256') or own.get('sha256') or 'unavailable'}` "
            f"captured {own.get('captured_at') or 'time unavailable'}; {own.get('bytes', source.get('size', '?'))} bytes.",
            f"Source: read_file(root='artifact_store', path='{source.get('path') or ''}').",
            "Snapshot coverage: " + json.dumps(coverage, ensure_ascii=False, default=str)]
    for sid, facts in (wave.get("dialogue_delivery") or {}).items():
        rows.append(f"- {sid} prepared dialogue coverage (physical/read status below): " + json.dumps(facts, ensure_ascii=False, default=str))
    if not wave.get("dialogue_delivery"):
        rows.append("Per-slot dialogue coverage was not recorded in this historical wave.")
    rows.append(("Cached review" if cached else "This review") + " covers this recorded snapshot only. "
                "Later messages are not claimed reviewed; their implications remain your judgment. "
                "A changed plan/evidence request captures current discussion.")
    return rows


def _render_wave(
    wave: dict, *, cap: Optional[int], cycles_paid: int, enforcement: str,
    cached: bool = False, notes: Optional[List[str]] = None, reminder: str = "",
    historical_feedback: Optional[list[dict]] = None,
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
        f"**Declared/requested evidence:** {len(manifest.get('attached') or [])} attached; omissions: "
        + (", ".join(f"{o.get('locator')}: {o.get('reason')}" for o in manifest.get("omissions") or []) or "none"),
    ]
    lines.extend(_dialogue_source_view(wave, cached=cached))
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
    # The typed slot census is read before any slot or aggregate is worded: an answer
    # that has not arrived is a gap, never a failed slot and never a verdict.
    from ouroboros.tools.plan_review_runtime import plan_wave_slot_census

    census = plan_wave_slot_census(wave)
    late_word = "settled_late" if historical_feedback is not None else "uncollected"  # a free historical read collects nothing
    slot_class = {id(row): (late_word if name == "uncollected" else name)
                  for name in ("awaiting", "unresolved", "uncollected") for row in census[name]}
    if wave.get("custody_pending"):
        lines += [
            "", f"⚠️ REVIEW CUSTODY PENDING: {len(census['answered'])} of {census['configured']} reviewer(s) have answered"
            + (f", {len(census['uncollected'])} settled but not collected yet" if census["uncollected"] else "")
            + "; no reviewer verdict exists yet — the "
            f"{aggregate or 'recorded'} aggregate below is a placeholder that keeps this wave open."
        ]
    elif aggregate == "DEGRADED" and historical_feedback is None:
        # Banner aligned with _next_step: the replay promise depends on whether the
        # wave carries structural snapshot evidence (see _degraded_replay_note).
        lines += ["", "⚠️ DEGRADED: no parseable reviewer quorum — recorded as an OPEN wave; "
                  + _degraded_replay_note(wave, paid_available=cap is None or cycles_paid < cap) + "."]
    actor_lines = [
        f"- {a.get('slot_id')} · {a.get('model')} · {a.get('route')}"
        + (f" · effort {a['effort']}{' (ordered)' if a.get('declared_effort') else ''}" if a.get("effort") else "")
        + f" · host_file_read: {a.get('host_file_read_attestation')}"
        + (f" · room snapshot read {a['room_read_coverage'].get('covered_chars')}/{a['room_read_coverage'].get('complete_chars')} "
           f"chars ({a['room_read_coverage'].get('provenance')})" if isinstance(a.get("room_read_coverage"), dict) else "")
        + f" · {_actor_outcome(a, slot_class.get(id(a), ''))}"
        + ((" · not sent" if a.get("operation_state") == "not_dispatched" else " · did not answer")
           + "; its earlier finding is still listed" if a.get("carried_findings") else "")
        + (f" · disclosures: {', '.join(a['disclosures'])}" if a.get("disclosures") else "")
        for a in wave.get("actors") or []
    ] or ["(no actor records)"]
    findings = list(wave.get("findings") or [])
    findings_total = int(wave.get("findings_total") or len(findings))
    finding_page = findings[:MAX_FINDINGS_PER_SLOT]
    if wave.get("reviewer_effort"):
        actor_lines.append(f"- reviewer effort ordered for this envelope: {wave['reviewer_effort']}")
    if isinstance(wave.get("ordered_weaker"), dict) and wave["ordered_weaker"]:
        actor_lines.append("- ORDERED WEAKER THAN THE OWNER SETTING on " + ", ".join(
            f"{sid} ({row.get('effort')} < {row.get('owner_effort')})"
            for sid, row in wave["ordered_weaker"].items() if isinstance(row, dict)))
    lines += [
        "", "### Reviewer slots" + (" (original recorded state)" if historical_feedback is not None else ""), "", *actor_lines,
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
    if wave.get("historical_supplements"):
        lines += ["", "### Historical feedback", "",
                  "Late responses are retained separately. The original actors, aggregate and closure below "
                  "have not been recomputed; these sources do not create a new PASS."]
        resolved = {row["operation_id"]: row for row in historical_feedback or []}
        for row in wave["historical_supplements"]:
            ref = row.get("source_ref") or {}
            lines += [f"#### {row.get('slot_id')} · cycle {row.get('cycle_index')} · {row.get('operation_state')}",
                      f"Operation: {row.get('operation_id')}. Source SHA256: {ref.get('sha256') or 'unavailable'}.",
                      f"Source: read_file(root='artifact_store', path='{ref.get('path') or ''}')."]
            result = (resolved.get(row.get("operation_id")) or {}).get("result")
            if result is not None:
                if result.get("error"):
                    lines += ["Recorded error: " + _quote_control_lines(str(result["error"]))]
                lines += [_quote_control_lines(str(result.get("text") or "(no reviewer text)"))]
            else:
                lines += ["Full source is retained at the reference above; its body was not read for this view."]
    # DISPLAY only — the stored ``reasons`` stay whole (the quorum arithmetic wrote them).
    # The arithmetic's ``slot_unparseable:<slot>:`` entry of a slot with no collected
    # answer is left out by that slot's typed census class and the slot is named as what
    # it is, because later reviewer waves read this text as dialogue evidence.
    gaps = {"awaiting": census["awaiting"],
            ("settled late" if historical_feedback is not None else "not collected yet"): census["uncollected"]}
    hidden = tuple(f"slot_unparseable:{row.get('slot_id')}:" for rows in gaps.values() for row in rows)
    # A failed slot with a reported cause shows the engine's sentence in place of its error
    # prose (a display substitution keyed on slot_id; the stored reason stays whole).
    causes = {f"slot_unparseable:{a.get('slot_id')}:": str(a.get("reported_cause"))
              for a in wave.get("actors") or [] if a.get("reported_cause") and not a.get("ok")}
    reasons = [next((key + cause for key, cause in causes.items() if str(r).startswith(key)), str(r))
               for r in wave.get("reasons") or [] if not str(r).startswith(hidden)]
    reasons += [f"{label}: " + ", ".join(str(row.get("slot_id")) for row in rows) for label, rows in gaps.items() if rows]
    lines += [
        "", "### Aggregate: " + (f"no verdict — held open as {aggregate}" if wave.get("custody_pending") else aggregate)
        + (" (closed)" if closed else " (open)"),
        "", "Reasons: " + (", ".join(reasons) or "none")
        + f". Counts: {json.dumps(counts, sort_keys=True)}",
    ]
    if wave.get("dispositions"):
        lines += ["", "### Dispositions", "", "```json",
                  json.dumps(wave.get("dispositions"), ensure_ascii=False, indent=2), "```"]
    if isinstance(wave.get("author_disposition"), dict):
        lines += ["", "### Author finish", "", "```json",
                  json.dumps(wave.get("author_disposition"), ensure_ascii=False, indent=2), "```"]
    if wave.get("closure_notes") or notes:
        lines += ["", "Closure notes: " + "; ".join(_closure_note_view(note) for note in [*(wave.get("closure_notes") or []), *(notes or [])])]
    outcome, closed = wave_control_state(wave)
    lines += [
        "", "## Plan Review Contract", "",
        ("This is a free read of the completed historical responses. No reviewer was called and no cycle "
         "was consumed. Consider the feedback alongside the current task; the original review decision "
         "remains unchanged." if historical_feedback is not None else
         _next_step(wave, enforcement=enforcement, cap=cap, cycles_paid=cycles_paid)), "",
        PLAN_REVIEW_CONTROL_PREFIX + json.dumps({"outcome": outcome, "closed": closed}, separators=(",", ":")),
    ]
    return "\n".join(lines)
