"""Delivery builder for ``plan_task`` asks — pure string builders, no I/O.

What one participant receives for one ask, in this order (DESIGN §3):

1. the STABLE part (``build_stable_prompt``, the system prompt): its role in the
   council, the Plan Review Checklist section verbatim (the one home of what an
   objection is and what may hold), the reply contract, and — for a body plan — the
   constitutional pack (BIBLE.md and ARCHITECTURE.md inline for a packet seat,
   mandatory full reads for a retrieving seat) plus the DESIGN navigation map. A
   non-body plan carries no Ouroboros governance and no navigation maps (owner 4.3=A);
2. the PARTICIPANT part (``build_participant_content``, the user content): the task
   objective with its author (``owner_words.objective_author_line``), the owner's
   verbatim words, then the own room's conversation the delivery layer fits per seat —
   the cache-stable prefix (``plan_user_stable_len``) — then the plan:
   whole the first time this participant sees it, afterwards the delta since the
   revision it last answered about plus the current text, and before any revision no
   plan at all (a reading of the request); this ask's room snapshot facts and related
   rooms; everything said in the council since the ask this participant last answered
   in (peers' answers attributed by seat and model, the mind's messages), its OWN open
   objections to adjudicate first, the mind's message, and the attached evidence, each
   labelled by kind, with every omission named.

The reply contract (``PLAN_REPLY_CONTRACT`` / ``PLAN_REPLY_SESSION_OUTPUT_SCHEMA``) is
one JSON object ``{prose, objections[{id,text}], withdrawn[{id,why}]}``; the host never
aggregates replies and never votes. The old packet's divergent rubric copies are gone:
the checklist section is delivered, not paraphrased.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence


class PlanPacketError(ValueError):
    """Typed assembly failure: a body plan's pack without its governance documents."""


PLAN_REPLY_CONTRACT = """\
Return ONE JSON object, optionally inside one code fence, and nothing after it:
{"prose": "<your reading of the plan, or of the request before any plan: questions, suggestions, agreement or disagreement with peers — free text>",
 "objections": [{"id": "<your own short stable id, e.g. o1>", "text": "<what in the plan causes which failure>"}],
 "withdrawn": [{"id": "<an id YOU raised earlier>", "why": "<why it no longer stands>"}]}
An objection is yours alone: it stands until YOU list it under "withdrawn"; an open objection of
yours that you do not mention stays open and is flagged to the author as not adjudicated. Restate
one you keep (same id, the residual in "text"). Suggestions, questions and preferences are prose and
never hold anything. You cannot withdraw another participant's objection — agree or disagree with it
in prose. A reply the host cannot read as this object changes nothing and is reported as unreadable.
Only replies to a stored plan revision update standing objections. A reading before any plan neither
raises nor withdraws them: put that reading in the JSON object's prose field and leave objections and
withdrawn empty; its reply remains visible as advice.
"""

# Strict structured-output providers (OpenAI strict mode) require EVERY property and reject
# an omitted optional key, so the two optional lists are required-but-nullable: a participant
# with nothing to object or withdraw sends ``null`` or ``[]`` and ``parse_reply`` reads both as
# empty. Nested objects are closed too, as strict mode demands.
PLAN_REPLY_SESSION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prose", "objections", "withdrawn"],
    "properties": {
        "prose": {"type": "string"},
        "objections": {
            "type": ["array", "null"],
            "items": {"type": "object", "required": ["id", "text"], "additionalProperties": False,
                      "properties": {"id": {"type": "string"}, "text": {"type": "string"}}},
        },
        "withdrawn": {
            "type": ["array", "null"],
            "items": {"type": "object", "required": ["id", "why"], "additionalProperties": False,
                      "properties": {"id": {"type": "string"}, "why": {"type": "string"}}},
        },
    },
}

_ROLE = (
    "You are one independent participant of the council that weighs an INTENTION before the work "
    "starts — a plan for code, research, a deliverable, a computer-use flow or an action in the "
    "world. Before any plan exists, Ouroboros may ask how you read the request. "
    "Ouroboros authors the plan and asks you; it keeps its own judgment and answers you on the "
    "record. Judge the plan against ITS OWN domain with the evidence in front of you; say when the "
    "direction itself is wrong and what the simpler path to the owner-chosen outcome is. You speak "
    "for yourself: the host never aggregates the council's replies into a verdict and nobody votes. "
    "Other participants' answers reach you on later asks; the first ask is blind by construction."
)

# The user content's cache-stable prefix ends where the plan begins: the objective, the
# owner's words and the room's conversation rows are byte-identical across the asks of
# one task while those rows stay the same; each capture's snapshot facts (locator, counts,
# capture time, related rooms), the plan and the council's talk follow it.
_PLAN_MARKER = "## THE PLAN"


def build_stable_prompt(
    *,
    checklist_section: str,
    body: bool,
    enforcement: str,
    bible_text: Optional[str] = None,
    architecture_text: Optional[str] = None,
    bible_locator: str = "BIBLE.md",
    architecture_locator: str = "docs/ARCHITECTURE.md",
    governance_by_retrieval: bool = False,
    design_nav_map: Optional[str] = None,
) -> str:
    """The byte-stable part of every delivery of one plan (role, checklist, reply
    contract, the body pack). A body plan without BIBLE.md or ARCHITECTURE.md text is a
    typed ``PlanPacketError``: the pack is never silently incomplete."""
    parts = [
        _ROLE + "\n\n"
        f"Review enforcement for this task: {str(enforcement or 'unknown').strip()} (host information; under "
        "Blocking an open objection of an owner-marked reviewer to a plan revision holds the author's final answer, "
        "under Advisory and in Cyber Pro it is disclosed; it does not change what you should say).\n\n"
        f"## Reply contract\n\n{PLAN_REPLY_CONTRACT}\n---\n",
    ]
    if checklist_section:
        parts.append(f"## Plan Review Checklist (verbatim from docs/CHECKLISTS.md)\n\n{checklist_section}\n\n---\n")
    else:
        parts.append("⚠️ OMISSION NOTE: Plan Review Checklist section not supplied by the host.\n\n---\n")
    if body:
        if not bible_text:
            raise PlanPacketError("a body plan's pack requires BIBLE.md; none supplied")
        if not architecture_text:
            raise PlanPacketError("a body plan's pack requires ARCHITECTURE.md; none supplied")
        if governance_by_retrieval:
            parts.append(
                "## Governance pack (this plan touches Ouroboros's own body) — MANDATORY FULL READS\n\n"
                "You are a retrieving participant: before judging, read BOTH documents in full with your own "
                f"tools — the constitution `{str(bible_locator or 'BIBLE.md').strip()}` and the architecture "
                f"reference `{str(architecture_locator or 'docs/ARCHITECTURE.md').strip()}`. A packet participant "
                "receives them inline; you receive them by retrieval (same pack, same authority).\n\n---\n"
            )
        else:
            parts.append(f"## BIBLE.md (constitution — this plan touches Ouroboros's own body)\n\n{bible_text}\n\n---\n")
            parts.append("## ARCHITECTURE.md (architecture and data flow — inline, in full, for a plan that changes "
                         f"Ouroboros's own body)\n\n{architecture_text}\n\n---\n")
        if design_nav_map:
            parts.append("\n" + design_nav_map + "\n")
            parts.append(
                "Read relevant original DESIGN sections with your own tools.\n" if governance_by_retrieval else
                "This participant has no repository tools: relevant DESIGN text must be attached as evidence; "
                "a heading map is not those excerpts.\n"
            )
    return "\n".join(parts)


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _evidence_author(item: Mapping[str, Any]) -> str:
    """Whose words an attachment carries, by the kind its resolver recorded (``plan_evidence``)."""
    kind = str(item.get("kind") or "")
    if kind == "task":
        task_id = str(item.get("locator") or "").split(":", 1)[-1].split("::", 1)[0].strip()
        return f"the result of Ouroboros's task {task_id} with host facts about it"
    if kind == "chat":
        return "room rows, each signed with its author"
    return "file content; its author is not attested by the host"


def render_evidence(manifest: Mapping[str, Any]) -> str:
    """The attached evidence, each item labelled with whose words it carries, every absence named."""
    declared = list(manifest.get("declared") or [])
    if not declared and not manifest.get("omissions"):
        return "(nothing attached)\n"
    lines: list[str] = []
    for item in manifest.get("attached") or []:
        redacted = ", secrets_redacted=true" if item.get("secrets_redacted") else ""
        lines.append(
            f"### {item.get('locator')} (kind={item.get('kind')}, sha256={item.get('sha256')}, "
            f"bytes={item.get('bytes')}, attached_bytes={item.get('attached_bytes')}{redacted})\n"
            f"Carries: {_evidence_author(item)}.\n"
            f"----- BEGIN {item.get('locator')} -----\n{item.get('text') or ''}\n"
            f"----- END {item.get('locator')} -----\n"
        )
    omissions = list(manifest.get("omissions") or [])
    lines.append("### OMISSIONS (every absence named)\n")
    if omissions:
        lines.append("| locator | reason |\n|---|---|")
        lines.extend(f"| {_cell(o.get('locator'))} | {_cell(o.get('reason'))} |" for o in omissions)
        lines.append("")
    else:
        lines.append("none\n")
    return "\n".join(lines)


def _json_block(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n```"


def render_revision(revision: Mapping[str, Any]) -> str:
    """One revision, whole: goal, done_when, affected paths, body flag, prose."""
    done_when = [{k: v for k, v in row.items()} for row in revision.get("done_when") or []]
    body = "yes" if revision.get("body") else "no"
    return (
        f"Revision {revision.get('n')} (recorded {revision.get('ts') or '?'}; changes Ouroboros's own body: {body}"
        + (f" — {revision.get('body_note')}" if revision.get("body_note") else "") + ")\n\n"
        f"### Goal\n\n{str(revision.get('goal') or '').strip() or '(none stated)'}\n\n"
        "### done_when (what acceptance will judge; id · claim · how it will be shown)\n\n"
        + (_json_block(done_when) if done_when else "(none stated yet)") + "\n\n"
        "### affected_paths (files the work will change; [] = none)\n\n"
        + _json_block(list(revision.get("affected_paths") or [])) + "\n\n"
        f"### Plan\n\n{str(revision.get('plan') or '').strip() or '(no prose yet)'}\n"
    )


def _render_delta(delta: Mapping[str, Any]) -> str:
    lines = [f"Changes since revision {delta.get('from')} (the revision you last answered about):"]
    if delta.get("goal_changed"):
        lines.append("- the goal changed")
    for row in delta.get("done_when_added") or []:
        lines.append(f"- done_when added: {row.get('id')} — {row.get('claim')}" + (f" (shown by: {row.get('support')})" if row.get("support") else ""))
    for row in delta.get("done_when_removed") or []:
        lines.append(f"- done_when removed: {row.get('id')} — {row.get('claim')}")
    for row in delta.get("done_when_reworded") or []:
        before, after = row.get("before") or {}, row.get("after") or {}
        lines.append(f"- done_when reworded: {row.get('id')}: «{before.get('claim')}» → «{after.get('claim')}»"
                     + (f"; shown by «{before.get('support')}» → «{after.get('support')}»" if before.get("support") != after.get("support") else ""))
    for path in delta.get("affected_paths_added") or []:
        lines.append(f"- affected path added: {path}")
    for path in delta.get("affected_paths_removed") or []:
        lines.append(f"- affected path removed: {path}")
    if delta.get("body_changed"):
        lines.append("- whether the plan changes Ouroboros's own body changed")
    if delta.get("plan_previous_unavailable"):
        lines.append("- the prose you saw is no longer inline and its retained source could not be read "
                     f"({delta['plan_previous_unavailable']}); it was not diffed")
    elif delta.get("plan_changed"):
        lines.append("- the plan prose changed:\n```diff\n" + "\n".join(delta.get("plan_diff") or []) + "\n```")
    if len(lines) == 1:
        lines.append("- nothing in the recorded fields changed")
    return "\n".join(lines) + "\n"


def _render_reply(entry: Mapping[str, Any]) -> str:
    who = f"{entry.get('seat_id')}" + (f" ({entry.get('model')})" if entry.get("model") else "")
    status = str(entry.get("status") or "")
    if status != "answered":
        detail = str(entry.get("error") or entry.get("parse_error") or entry.get("not_dispatched_reason") or "")
        return f"- {who}: {status}" + (f" — {detail}" if detail else "")
    reply = entry.get("reply") or {}
    lines = [f"- {who} answered:"]
    prose = str(reply.get("prose") or "").strip()
    omitted = reply.get("prose_omitted")
    if prose:
        lines.append("  " + prose.replace("\n", "\n  "))
    elif isinstance(omitted, dict):  # the index moved the prose out; the retained source did not fit or could not be read
        why = {"too_large": "larger than one attachment", "unreadable": "its retained source could not be read"}.get(
            str(omitted.get("reason") or ""), "no retained source")
        lines.append(f"  (prose not delivered: {omitted.get('chars') or '?'} chars, {why}"
                     + (f"; retained at {omitted['address']}" if omitted.get("address") else "") + ")")
    elif not reply.get("prose_inline", True):
        lines.append("  (prose not inline: moved to its retained source, not delivered here)")
    for item in reply.get("objections") or []:
        lines.append(f"  objection {item.get('id')}: {item.get('text')}")
    for item in reply.get("withdrawn") or []:
        lines.append(f"  withdrew {item.get('id')}: {item.get('why')}")
    for item in reply.get("ignored") or []:  # a malformed item the host skipped, named so nobody counts it as said
        lines.append(f"  (ignored malformed {item.get('key')}[{item.get('index')}]"
                     + (f" id {item.get('id')!r}" if item.get("id") else "") + f": {item.get('reason')})")
    if len(lines) == 1:
        lines.append("  (no prose, no objection, no withdrawal)")
    return "\n".join(lines)


def failure_tag(participant: Mapping[str, Any]) -> str:
    """A failed seat by its TYPED facts — the code, the reset instant, the engine's reported
    sentence — never the raw error prose with its retry coach (BIBLE P5: the host puts no
    engine template in the mind's or the owner's mouth); the one owner of that line for the
    read view and the progress line alike. A row that reported nothing typed keeps a bounded
    first line of the error."""
    from ouroboros.utils import truncate_review_artifact

    code = str(participant.get("failure_code") or "")
    cause = str(participant.get("reported_cause") or "")
    reset = str(participant.get("reset_at") or "")
    tag = "failed" + (f"[{code}]" if code else "")
    if cause:
        tag += f' — reported cause: "{cause}"'
    elif participant.get("error"):
        first_line = str(participant["error"]).strip().splitlines()[0] if str(participant["error"]).strip() else ""
        tag += " — " + truncate_review_artifact(first_line, limit=160)
    if reset:
        tag += f" (reset at {reset})"
    return tag


def render_cross_talk(asks: Sequence[Mapping[str, Any]], *, seat_id: str) -> str:
    """Everything said in the council in ``asks`` (the ones since this participant last
    answered, preceded by earlier asks' replies that settled after its last delivery,
    marked ``late_replies``), attributed by seat and model; the participant's own replies
    are not repeated to it."""
    if not asks:
        return ""
    from ouroboros.plan_record import ask_subject

    sections = []
    for ask in asks:
        head = f"### Ask {ask.get('ask_id')} (about {ask_subject(ask.get('revision'))}, {ask.get('ts') or '?'})"
        if ask.get("late_replies"):
            head += " — answers that arrived after your last delivery"
        message = str(ask.get("message") or "").strip()
        lines = [head, f"Ouroboros asked: {message}" if message else "Ouroboros asked for a review."]
        for participant in ask.get("participants") or []:
            if str(participant.get("seat_id") or "") == seat_id:
                continue
            lines.append(_render_reply(participant))
        sections.append("\n".join(lines) + "\n")
    return "\n".join(sections)


def render_own_objections(objections: Sequence[Mapping[str, Any]], *, reading: bool = False) -> str:
    """This participant's open objections; a reading (before any plan) cannot withdraw or restate them."""
    from ouroboros.plan_record import ask_subject

    if not objections:
        return "You have no open objection.\n" if reading else "You have no open objection on this plan.\n"
    lines = [("They stand as they are: a reading before any plan neither withdraws nor restates them; a reply to a plan "
              "revision does.\n") if reading else
             ("Adjudicate these FIRST: list each under \"withdrawn\" with why, or restate it with what is still missing. "
              "One you do not mention stays open and is flagged as not adjudicated.\n")]
    for item in objections:
        flag = " (flagged: not adjudicated in your last reply)" if item.get("not_adjudicated") else ""
        where = ("your blocking finding in this task's previous plan review, imported; it holds the task until you withdraw it"
                 if item.get("legacy") else f"raised in ask {item.get('raised_in')} about {ask_subject(item.get('revision'))}")
        lines.append(f"- {item.get('id')} ({where}){flag}: {item.get('text')}")
    return "\n".join(lines) + "\n"


def build_participant_content(
    *,
    objective: str,
    owner_words: str,
    dialogue_section: str,
    revision: Optional[Mapping[str, Any]],
    delta: Optional[Mapping[str, Any]],
    cross_talk: str,
    own_objections: Sequence[Mapping[str, Any]],
    message: str,
    evidence_section: str,
    first_time: bool,
    objective_author: str = "",
    snapshot_section: str = "",
) -> str:
    """One participant's user content for one ask (see the module docstring for the order).
    ``objective_author`` is static per task, so it rides the cache-stable prefix; a reading
    (``revision=None``) says so after the plan marker, where everything changes per ask, and
    so does ``snapshot_section`` (``plan_dialogue.render_dialogue``: what each capture changes)."""
    if revision is None:
        plan = "No plan yet: Ouroboros asks how you read the request before it writes one.\n"
    else:
        plan = ("You see this plan for the first time; it follows whole.\n\n" if first_time or not delta or delta.get("first")
                else _render_delta(delta) + "\nThe current text follows whole.\n\n") + render_revision(revision)
    sections = [
        "## TASK OBJECTIVE\n\n" + (objective or "(none declared)")
        + (f"\n\nThe objective above: {objective_author}." if objective_author else "") + "\n",
        *([owner_words + "\n"] if str(owner_words or "").strip() else []),
        dialogue_section,
        _PLAN_MARKER + "\n\n" + plan,
        *([snapshot_section] if snapshot_section else []),
        "## THE COUNCIL SO FAR (since the ask you last answered in)\n\n"
        + (cross_talk if cross_talk else (f"Nothing yet: this is the first ask{' on this plan' if revision is not None else ''} "
                                          "and every first answer is blind.\n" if first_time else "Nothing was said since your last answer.\n")),
        "## YOUR OPEN OBJECTIONS\n\n" + render_own_objections(own_objections, reading=revision is None),
        "## WHAT OUROBOROS ASKS YOU NOW\n\n"
        + (str(message or "").strip() or f"(no message beyond the {'plan' if revision is not None else 'request'} itself)") + "\n",
        "## ATTACHED EVIDENCE\n\n" + evidence_section,
    ]
    return "\n".join(sections)


def plan_user_stable_len(user_content: str) -> int:
    """Byte offset where the cache-stable prefix ends: everything before the plan
    (objective, owner words, the room's conversation rows) is byte-identical while that
    conversation is unchanged; the snapshot facts, the plan and the council's talk change
    between asks."""
    index = str(user_content or "").find(_PLAN_MARKER)
    return index if index > 0 else 0


def settled_frame(ask_id: str, participants: Sequence[Mapping[str, Any]], objections: Sequence[Mapping[str, Any]],
                  custody: Optional[Mapping[str, str]] = None) -> str:
    """The mailbox frame when the counted council finishes: who answered,
    who could not be read, who failed — and a seat whose outcome is UNKNOWN (custody lost, or
    the dispatch lost after the hand-off) is named as unresolved (neither a failure nor a
    reply), never counted as settled; a new ask to it may pay twice. ``custody`` is the wave
    as custody settled it: a seat it settled that the record has not taken is named so — the
    answer exists and plan_task() collects it at $0 — never reported as still owed."""
    from ouroboros.review_custody import RELEASED_UNRESOLVED

    recorded = {str(p.get("seat_id") or "?"): p for p in participants}
    answered = sum(1 for p in participants if p.get("status") == "answered")
    unread = sum(1 for p in participants if p.get("status") == "unparseable")
    failed = sum(1 for p in participants if p.get("status") == "failed")
    unresolved = [str(p.get("seat_id") or "?") for p in participants if p.get("status") == "pending" and p.get("unresolved")]
    held = [sid for sid, state in (custody or {}).items() if state and state != RELEASED_UNRESOLVED
            and (sid not in recorded or (recorded[sid].get("status") == "pending" and not recorded[sid].get("unresolved")))]
    waiting = [str(p.get("seat_id") or "?") for p in participants
               if p.get("status") == "pending" and not p.get("unresolved") and str(p.get("seat_id") or "?") not in held]
    advisers = [sid for sid in waiting if not recorded[sid].get("owner_marked")]
    waiting = [sid for sid in waiting if recorded[sid].get("owner_marked")]
    own = [o for o in objections if o.get("ask_id") == ask_id]
    head = (f"Plan ask {ask_id}: counted participants settled" if advisers and not waiting
            else f"Plan ask {ask_id}: every participant settled" if not unresolved and not waiting and not advisers and not held
            else f"Plan ask {ask_id}: custody has nothing more to deliver")
    return (f"{head} — {answered} answered" + (f", {unread} unreadable" if unread else "") + (f", {failed} failed" if failed else "")
            + (f", {len(unresolved)} unresolved ({', '.join(unresolved)}: outcome unknown — neither a failure nor a reply; nothing waits on it, "
               "a new ask may pay twice)" if unresolved else "")
            + (f", {len(held)} settled in custody but not yet in the plan record ({', '.join(held)}: plan_task() collects them at $0)"
               if held else "")
            + (f", {len(waiting)} still owed ({', '.join(waiting)})" if waiting else "")
            + (f"; advisers still working: {', '.join(advisers)} (not counted; replies land in the plan record; "
               "plan_task() reads them; no further notice)" if advisers else "")
            + (f"; {len(own)} open objection{'' if len(own) == 1 else 's'} from this ask" if own else "; no open objection from this ask")
            + ". plan_task() reads the answers.")
