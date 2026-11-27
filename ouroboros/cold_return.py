"""A returned work's first request that cannot fit parks the same work instead of ending it.

A start that restores a continuation (``owner_wait.governing_continuation``: a cold owner wait, an exact pause or
saved work) meets the memory and rules as they stand now beside what it saved then: its transcript, the overview and
index rows it held, today's tool frame. Fresh memory is admitted against a fresh Main (``nano_publication``), never
against every saved work, so that request can be larger than the window the route serves.

When the serving instance's own count refuses it before dispatch, the size recovery takes every form it may (the
refusal passes, the configured routes, the view's renewal). If the last of them is refused the same way, nothing of
this work was sent since it returned: ending it would lose saved work to a window, not to a decision about the work.
The loop parks it instead (``park_refused``): the ordinary exact pause (``budget_pause.request_pause``) on the
technical rail ``RAIL_CONTEXT_WINDOW`` with the typed cause ``CAUSE`` and the refusal's own numbers on the durable row
(``FACTS_KEY``). The same id keeps its saved state under a new pause, as every rail's park does; only an explicit
Resume returns it, because no readiness knows this cause (``sleep_wake.authority_readable``), and each Resume is the
real attempt again, with no forecast in front of it. A park that cannot be saved leaves the ordinary size terminal.

The saved view is the one the size recovery left: replaced rows stay exact behind their addresses. The park needs
an exact reply-floor refusal in owner-selected Nano on a queued root using the built-in 80k Main model, including
a queued Project root on that route. This is not the fresh Main memory budget or work-room guarantee. An exact
input count alone is insufficient: another reserve can refuse an input that leaves room for the minimum reply.
Other profiles, windows and refusals keep ordinary recovery and its terminal.

The proof is the return's, not one round's: while nothing of the work was sent since it returned, a round after
the entry one keeps the same witness (``unsent_return``). A view the size recovery rebuilt and never sent
(``context_renewal``) therefore parks on its next refusal as the entry round would; the entry round's count of the
Nano floor (``nano_minimum``) is not repeated for it.

A window dialog binds all three explicit choices (``door_choice``) to its current stop identity, pause and generation,
and the grant checks the same stop projection. A stale or unreadable offer grants nothing. An ordinary Resume that
was not a choice in this dialog retains its own contract.

The grant is the single-use right to enter again; the choice it carries becomes the work's own saved state before
that grant is spent (``carry_choice``, where the loop takes the pause over), so the checkpoint an ordinary restart
restores holds it too (``arm``). The choice is made once, about one attempt, and it ends with the owner's next
decision about this work: the chosen end itself, a reply the work reads, the next window park (which takes the
chosen overview or finds nothing to take, and whose own Resume carries only its own choice), or the owner's own
Pause, whose Resume returns the work as any Pause's does (``carry_choice`` for a pause that restores, ``_choice``
for one that kept this stack warm). Nothing else ends it. A hold that was requested and never saved, a pause row
that could not be read and a technical wait that returns by itself or after its own cause is repaired decide
nothing about the window: the work comes back from them still carrying the choice, so such a wait never turns a
chosen end into an ordinary request. The loop's order stands as it was: Stop, Pause, the deadline and money decide
before the chosen end, and none of them is answered with it.

``end_attempt``: before the park such a return ended on the size terminal, which the owner could Continue as new
work. The choice restores the saved work and ends this attempt on that same terminal, with its recorded cause and
nothing sent, where the loop would otherwise call the model (``end_chosen``): a Stop, a Pause, the deadline, money
or an answer the work had ready decide first, as for any round. The existing Continue eligibility then applies.

``overview_revision``: a work that holds the overview at an older revision carries that text whole, so room made by
shortening the overview never reaches it. A park counts its refused request once more with the current overview in
the held one's place (``_overview_forms``): every section the overview reaches as its own producer renders it now,
in the form this request shows it, the outlined index beside the address its exchanged text would be kept at. The
count is taken at the reply floor on the same serving binding, clock line, tools and rows; when it fits, the row
shows both revisions and both counts (``OVERVIEW_KEY``). Choosing the revision changes nothing by itself: its Resume
is the real attempt on the held overview again. Only if that is refused again and the same count still fits with
the very revision chosen does the park that follows save the work holding it, so the next Resume continues there.
That is decided where the saved state is built, not where the count was taken: the save takes the overview writers'
shelf lock and reads that one note against the counted revision (``held_notes.Taking``), so a revision written
meanwhile leaves the held pins and transcript, and the stale offer is not kept on the row (``settled_facts``); a
check that cannot be made does the same without the row's offer changing. Counts and the wait for live producers
stay outside that lock. If the held one fits now the work simply continues on it; if the shelf moved or the count
no longer fits, the held one stays and the new park shows what it measured. The replaced revision stays readable
at its versioned address.

No readiness, restart or timer makes either choice.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

RAIL_CONTEXT_WINDOW = "context_window"  # ``budget_pause.RAILS``: the technical park this refusal takes
CAUSE = "context_nonfit"  # its typed cause on the pause row and the queue marker
FACTS_KEY = "context_nonfit"  # the refused request's numbers on that row
OVERVIEW_KEY = "overview"  # within them: the current overview this park counted in the held one's place
REASON_TEXT = "its saved work did not fit the model's window"
# The work row's reason line (``work_stop.stop_presentation``): what holds it and the owner's next step.
WAIT_TEXT = "Its saved work does not fit the model's window. Make room, then press Resume."
OFFER_TEXT = "The current overview fit at the last check. Press Resume to choose it."
READY_TEXT = ("Your chosen overview is in place. The request fit at the last check. "
              "Press Resume.")
CHOICE_KEY = "context_choice"  # the Resume body's field, and the same choice on the single-use grant
# In a task's usage, so in every checkpoint of its return: the choice its spent grant carried (``carry_choice``).
CARRIED_KEY = "_window_park_choice"
# On a task's tool context: what this process had recorded when it restored the work (``arm``).
_ARMED = "_cold_return"
_PAUSED_UNDER = "_cold_return_fence"  # beside it: the owner's Pause fence this stack had last waited under then
_CHOICE: ContextVar[Optional[Dict[str, Any]]] = ContextVar("window_park_choice", default=None)
_NUMBERS = ("input_tokens", "window_tokens", "reply_floor_tokens", "over_tokens")


def arm(tool_ctx: Any, usage: Dict[str, Any], continuation: Any) -> None:
    """A start restored ``continuation``: keep, by identity, the last physical send and size refusal known then, and
    the owner's choice this return still carries, with the numbers of the park it was made about.

    The first two are the previous process's or an earlier task's; only what this start records after them is its own.
    The choice is read from the grant of the pause this start restores (``carry_choice``), else from the work's own
    saved state (``_carried``): a restart after that grant was spent restores a checkpoint, not the pause.
    A start that restored nothing disarms the context: its refusals keep the ordinary terminal."""
    armed = None
    if continuation:
        from ouroboros.context_view_facts import SIZE_REFUSAL_KEY
        from ouroboros.usage_accounting import last_physical_attempt_capture

        row = continuation.get("_pause_row")
        choice = carry_choice(usage, row) if isinstance(row, dict) else _carried(usage)
        armed = (last_physical_attempt_capture(), usage.get(SIZE_REFUSAL_KEY), choice)
        setattr(tool_ctx, _PAUSED_UNDER, str(getattr(tool_ctx, "_owner_pause_fence_id", "") or ""))
    setattr(tool_ctx, _ARMED, armed)


def carry_choice(usage: Dict[str, Any], row: Any) -> Dict[str, Any]:
    """Keep the choice pause ``row``'s grant carries in the work's own saved state, and return what this return
    carries now.

    Called with the saved usage before that grant is spent, so the checkpoint written for its consumption holds the
    choice (``loop._resume_continuation``). A grant with a choice replaces what the usage held. Without one, the
    pause decides: a window park and the owner's own Pause are the owner's next decision about this work, and
    their plain Resume carries nothing an earlier hold's grant did, whether that Pause is its own row or lies over
    a technical or money stop (the grant names the owner's fence); any other pause (a technical wait, money)
    decides nothing about the window and returns the work with what it carried. ``answers_read`` is the count of
    replies this work had read then (``transcript_prefix``): the choice is about the attempt before the next one."""
    from ouroboros.owner_pause import RAIL_OWNER_PAUSE
    from ouroboros.transcript_prefix import _answer

    choice = _granted(row)
    if choice:
        usage[CARRIED_KEY] = {**choice, "facts": row.get(FACTS_KEY), "answers_read": _answer(usage)[0]}
    elif isinstance(row, dict) and (row.get("rail") in (RAIL_CONTEXT_WINDOW, RAIL_OWNER_PAUSE)
                                    or (row.get("grant") or {}).get("owner_pause_fence_id")):
        # The owner's Pause may lie over a saved technical or money stop and leave that row's rail as it was: the
        # grant of the Resume that releases it names the closed fence it was granted under (``budget_resume``).
        usage.pop(CARRIED_KEY, None)
    return _carried(usage)


def _choice(tool_ctx: Any, usage: Dict[str, Any]) -> Dict[str, Any]:
    """The choice this return still carries in this process (``arm``), read where it is applied.

    The owner's own Pause ends it. One that parked this work cold restores through its pause row (``carry_choice``).
    One that kept this very stack warm shows here: ``budget_pause.enter_owner_pause`` names the closed fence a
    stack waits under, and a fence named since the arming is a Pause that held this work. From then on nothing is
    carried, in this process or in what it saves next. A fence that could not be read names nothing and ends
    nothing."""
    armed = getattr(tool_ctx, _ARMED, None)
    if armed is None or not armed[2]:
        return {}
    fence = str(getattr(tool_ctx, "_owner_pause_fence_id", "") or "")
    if fence and fence != getattr(tool_ctx, _PAUSED_UNDER, ""):
        usage.pop(CARRIED_KEY, None)
        setattr(tool_ctx, _ARMED, (*armed[:2], {}))
        return {}
    return armed[2]


def _carried(usage: Dict[str, Any]) -> Dict[str, Any]:
    """The choice a restored work still carries (``carry_choice``), with its park's numbers; ``{}`` and nothing kept
    once it is not a whole choice or the work has read a reply since."""
    from ouroboros.transcript_prefix import _answer

    held = usage.get(CARRIED_KEY)
    named = ({key: value for key, value in held.items() if key not in ("facts", "answers_read")}
             if isinstance(held, dict) else None)
    choice = door_choice({CHOICE_KEY: named})[0] if named is not None else None
    if choice is None or type(held.get("answers_read")) is not int or held["answers_read"] != _answer(usage)[0]:
        usage.pop(CARRIED_KEY, None)
        return {}
    return {**choice, "facts": held.get("facts")}


def unsent_return(call: Any, plan: Any) -> bool:
    """Whether this round still belongs to a returned work nothing was sent for: an ordinary queued root on the
    built-in Main lane in Nano (the root ``nano_minimum.measurement_episode`` observes at its entry round), armed
    in this process, with no physical send recorded since.

    The witness of ``nano_reply_floor_proof`` for a round after the entry one. It asks for no count: the entry
    round's observation of the Nano floor belongs to that round alone."""
    from ouroboros.usage_accounting import last_physical_attempt_capture

    slot = call.tools._ctx
    armed = getattr(slot, _ARMED, None)
    return bool(armed is not None and last_physical_attempt_capture() is armed[0]
                and getattr(plan, "preferred_mode", "") == "nano" and getattr(call, "active_use_local", False)
                and getattr(plan, "model_role", "") == "main" and getattr(plan, "view_kind", "") == "owner_turn"
                and getattr(plan, "core", None) and call.task_type == "task"
                and not (getattr(slot, "is_direct_chat", False)
                         or (getattr(slot, "task_metadata", None) or {}).get("_is_direct_chat")))


def nano_reply_floor_proof(request: Any, target: dict, candidate: dict, measured: dict, floor: int) -> dict:
    """Only the exact Nano reply limiter can attest this deficit, on the prepared input of a queued root."""
    from ouroboros.llm_local import _endpoint_port
    from ouroboros.nano_minimum import _sample, _verified_binding
    from ouroboros.nano_publication import QUALIFIED_WINDOW
    from ouroboros.send_clock import split_clock_note

    physical = request.physical_context
    # The entry witness is built only for ordinary queued roots on the local Main lane.
    # It can name a cold_start on a proven-unsent pause; arm() decides whether this is a return.
    # A later round of a return nothing was sent for carries ``nano_floor_return`` instead (``unsent_return``).
    if not (physical and (physical.nano_floor_plan or physical.nano_floor_return) and physical.view_kind == "owner_turn"
            and physical.profile == "owner_nano" and physical.preferred_mode == physical.rendered_mode == "nano"
            and request.provider == "local" and request.context_window_tokens_exact == QUALIFIED_WINDOW
            and target.get("context_window_confirmed") is True
            and target.get("context_window_tokens") == QUALIFIED_WINDOW):
        return {}
    port = _endpoint_port(target.get("local_dispatch_base_url"))
    sample = _sample(candidate, measured)
    binding, gap = _verified_binding(measured, expected_window=QUALIFIED_WINDOW)
    if not (port and sample and binding and not gap
            and str(binding["process_id"]) == target.get("window_instance_id")):
        return {}
    # A fitting floor proves a sendable comparison only when the elastic reply
    # uses the whole input and admission does not charge a separate larger reserve.
    floor_sufficient = not physical.required_book_tokens and max(
        physical.response_reserve_tokens, request.working_reserve_tokens or 0) <= floor
    return {"reply_floor_tokens": floor, "port": port, "sample": sample, "binding": binding,
            "clock_note": split_clock_note(candidate)[0], "floor_sufficient": floor_sufficient}


def refusal_facts(tool_ctx: Any, usage: Dict[str, Any], round_idx: int) -> Dict[str, Any]:
    """The numbers of a returned work's refused request, ``{}`` unless this round proved every part of it.

    A restore in this process; no physical send recorded since (so no reply was read either); this round's latest
    size refusal recorded by this process at the scoped Nano reply floor; and no attempt left unresolved or owed
    a repeat. ``over_tokens`` is strictly positive."""
    from ouroboros.context_view_facts import size_refusal
    from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY, TRANSPORT_DEATHS_KEY
    from ouroboros.usage_accounting import last_physical_attempt_capture

    armed = getattr(tool_ctx, _ARMED, None)
    slot = size_refusal(usage, round_idx)
    if (armed is None or last_physical_attempt_capture() is not armed[0] or slot is None or slot is armed[1]
            or usage.get("_last_llm_error_kind") != "context_overflow"
            or slot.get("basis") != "exact_local_count" or isinstance(usage.get(TRANSPORT_DEATHS_KEY), dict)
            or usage.get("_pending_transport_outcome") or TIMEOUT_REPEAT_KEY in usage):
        return {}
    proof = slot.get("nano_reply_floor")
    needed, window = slot.get("tokens"), slot.get("local_window")
    reply = proof.get("reply_floor_tokens") if isinstance(proof, dict) else None
    if not (all(type(value) is int and value > 0 for value in (needed, window, reply)) and needed + reply > window):
        return {}
    return {"input_tokens": needed, "window_tokens": window, "reply_floor_tokens": reply,
            "over_tokens": needed + reply - window, "model": str(slot.get("model") or ""), "basis": slot["basis"]}


def park_refused(limit_ctx: Any) -> None:
    """Park a returned work whose request the serving instance refused by exact count with nothing sent.

    Raises ``budget_pause.BudgetPauseRequested`` once the park is durable, as every rail does. Returns when the
    round proved no such refusal, or when no exact pause could be saved (``exact_pause_unavailable`` names why):
    the caller's ordinary size terminal follows. The park records the current overview's count where it fits in
    the held one's place. It saves the work holding that overview only if this start's Resume chose that very
    revision and the shelf still holds it where the save's state is built (``held_notes.Taking``)."""
    tool_ctx = limit_ctx.tools._ctx
    facts = refusal_facts(tool_ctx, limit_ctx.accumulated_usage, limit_ctx.round_idx)
    if not facts:
        return
    from ouroboros.budget_pause import request_pause
    from ouroboros.held_notes import Taking, holding

    forms = None
    try:
        forms = _overview_forms(limit_ctx, facts)
    except Exception:  # a count that could not be taken offers nothing; the park itself needs none
        log.warning("The current overview could not be counted for a window park", exc_info=True)
    chosen = bool(forms) and _choice(tool_ctx, limit_ctx.accumulated_usage).get("overview_revision") == forms["current"]
    if forms:
        facts[OVERVIEW_KEY] = {"held_revision": forms["held"], "current_revision": forms["current"],
                               "input_tokens": forms["input_tokens"]}
    taking = Taking(tool_ctx, forms, limit_ctx.messages, _overview_notice(forms)) if chosen else None
    with holding(taking):
        request_pause(limit_ctx, rail=RAIL_CONTEXT_WINDOW, scope="task", reason_text=REASON_TEXT, cause=CAUSE,
                      facts={FACTS_KEY: facts})


def settled_facts(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """A window park's typed-cause fields as the save just built left them (``budget_pause.request_pause``, right
    after the build of its state): the park's own, until a save that carries the owner's choice has decided.

    The seed never claims the choice. A save that took the chosen revision says so on its row (``taken``). One that
    saw the shelf hold another revision offers the stale one no longer: its row keeps the numbers and nothing of
    the overview, and the next park counts the shelf as it is. A check that could not be made changes nothing: the
    plain offer stays, as of the last check, and nothing is taken. The pins, rows and transcript of those saves are
    the held ones; the work stays an ordinary window park that Resume returns."""
    from ouroboros.held_notes import current_taking

    taking, facts = current_taking(), envelope.get(FACTS_KEY)
    offer = facts.get(OVERVIEW_KEY) if isinstance(facts, dict) else None
    if taking is None or not isinstance(offer, dict) or taking.seen not in ("accepted", "moved"):
        return envelope
    kept = ({**facts, OVERVIEW_KEY: {**offer, "taken": True}} if taking.seen == "accepted"
            else {key: value for key, value in facts.items() if key != OVERVIEW_KEY})
    return {**envelope, FACTS_KEY: kept}


def _overview_notice(forms: dict) -> dict:
    return {"role": "user", "content": "[SYSTEM NOTICE]\nThe owner chose the current overview for this work. "
            "The earlier overview remains readable: "
            f"knowledge_read(topic='overview', scope='global', version='{forms['held']}')."}


def _overview_forms(limit_ctx: Any, facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``held_notes.current_overview``'s forms with the count they give, when this work holds the overview at an
    older revision than the shelf's and its refused request fits with the current one in the held one's place;
    else ``None``.

    The comparison is one request against itself. The serving instance counts the refused input again under the
    refusal's own binding, clock line, tools and reply floor (``llm_local.recount_with``), then the same rows with
    only the overview's sections exchanged (``_exchanged``): the exchanged system row states the current revision
    where the overview's own sections render it; only those sections are exchanged, so an independent reference to
    the held revision elsewhere in the request survives untouched. A refusal a larger reserve than the reply floor made
    (``floor_sufficient``) is not comparable, and rows that no longer make the refused input offer nothing.
    The exchanged form includes the host's one new notice of the owner's choice, so the counted candidate also pays
    for that notice at the place where the next attempt would read it, immediately before its clock line."""
    from ouroboros.context_view_facts import size_refusal
    from ouroboros.held_notes import current_overview

    tool_ctx, recount = limit_ctx.tools._ctx, getattr(limit_ctx.llm, "recount_with", None)
    refused = size_refusal(limit_ctx.accumulated_usage, limit_ctx.round_idx) or {}
    proof = refused.get("nano_reply_floor") or {}
    forms = current_overview(tool_ctx) if callable(recount) and proof.get("floor_sufficient") else None
    if not forms:
        return None

    def exchanged(rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        other = _exchanged(rows, forms["swaps"])
        if other is None or forms["current"] not in other[0]["content"]:
            return None
        notice = _overview_notice(forms)
        if clock := proof.get("clock_note"):
            if other[-1] != {"role": "user", "content": clock}:
                return None  # no proof of the position of the new host row on this formatter
            return [*other[:-1], notice, other[-1]]
        return [*other, notice]

    counted = recount(limit_ctx.messages, limit_ctx.tool_schemas, exchanged, refusal=proof)
    if (not counted or counted[0] != facts["input_tokens"] or counted[1] is None
            or counted[1] + facts["reply_floor_tokens"] > facts["window_tokens"]):
        return None
    return {**forms, "input_tokens": counted[1]}


def _exchanged(rows: List[Dict[str, Any]], swaps: Any) -> Optional[List[Dict[str, Any]]]:
    """``rows`` with each held text of ``swaps`` exchanged in the system row; every other row is the same object.
    ``None`` unless every held text stands there exactly once: a request that does not carry the held forms whole,
    as their sections render them, has nothing comparable to count."""
    system = rows[0] if rows and rows[0].get("role") == "system" and isinstance(rows[0].get("content"), str) else None
    if system is None:
        return None
    text = system["content"]
    for old, new in swaps:
        if text.count(old) != 1:
            return None
        text = text.replace(old, new)
    return [{**system, "content": text}, *rows[1:]]


def _measured_overview(facts: Any) -> dict:
    offer = facts.get(OVERVIEW_KEY) if isinstance(facts, dict) else None
    return offer if isinstance(offer, dict) and type(offer.get("input_tokens")) is int else {}


def park_wait_text(facts: Any) -> str:
    """The card's short next step from this pause's own measured overview."""
    offer = _measured_overview(facts)
    if not offer:
        return WAIT_TEXT
    return READY_TEXT if offer.get("taken") is True else OFFER_TEXT


def park_notice(facts: Any) -> str:
    """The chat notice describes the check that parked this attempt, not a forecast of its next request."""
    step = "Its work is saved: make room in always-loaded memory or raise the window, then press Resume."
    if not isinstance(facts, dict) or not all(type(facts.get(key)) is int for key in _NUMBERS):
        return f"The task's saved work did not fit the model's window. {step}"
    floor, window = facts["reply_floor_tokens"], facts["window_tokens"]
    offer = _measured_overview(facts)
    taken = offer.get("taken") is True
    earlier = " with the earlier overview" if taken else ""
    said = (f"At the last check, the request{earlier} needed {facts['input_tokens'] + floor:,} tokens with room for "
            f"a reply, {facts['over_tokens']:,} more than the model's {window:,}-token window.")
    if not offer:
        return f"{said} {step}"
    fits = (f"With the {'chosen' if taken else 'current'} overview, it fit at "
            f"{offer['input_tokens'] + floor:,} tokens with room for a reply.")
    if taken:
        return (f"As you chose, the current overview is in place. {said} {fits} "
                "The earlier overview stays readable. Press Resume.")
    return f"{said} {fits} The earlier overview is still in use. Press Resume to see your options."


def window_facts(row: Any) -> Dict[str, Any]:
    """What a client shows and names for a waiting window park (``work_stop.task_current_stop``): the hold's
    identity, its numbers, and the current overview it may choose. ``{}`` for any other pause row."""
    from ouroboros.budget_pause import STATE_PAUSED

    facts = row.get(FACTS_KEY) if isinstance(row, dict) else None
    if not (isinstance(facts, dict) and row.get("rail") == RAIL_CONTEXT_WINDOW and row.get("cause") == CAUSE
            and row.get("state") == STATE_PAUSED):
        return {}
    offer = facts.get(OVERVIEW_KEY) if isinstance(facts.get(OVERVIEW_KEY), dict) else None
    return {"pause_id": row.get("pause_id"), "pause_generation": row.get("pause_generation"),
            **{key: facts[key] for key in _NUMBERS if type(facts.get(key)) is int},
            **({OVERVIEW_KEY: dict(offer)} if offer else {})}


def door_choice(body: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(choice, error)`` of a Resume body's ``context_choice``: absent is no choice. Present, it names the hold it
    was shown for and one action. The HTTP door also requires the current stop identity; older saved grants lacking
    it remain readable as already-made choices, not fresh requests."""
    if not isinstance(body, dict) or CHOICE_KEY not in body:
        return None, ""
    choice = body[CHOICE_KEY]
    named = (isinstance(choice, dict) and isinstance(choice.get("pause_id"), str) and bool(choice["pause_id"])
             and type(choice.get("pause_generation")) is int)
    action = set(choice) - {"pause_id", "pause_generation", "stop_identity"} if named else set()
    if (action in ({"end_attempt"}, {"try_again"}) and choice[next(iter(action))] is True) or (
            action == {"overview_revision"} and isinstance(choice["overview_revision"], str) and choice["overview_revision"]):
        return dict(choice), ""
    return None, ("context_choice names its hold (pause_id, pause_generation) and exactly one of "
                  "try_again: true, end_attempt: true or overview_revision.")


@contextmanager
def choosing(choice: Optional[Dict[str, Any]]):
    """Bind the owner's explicit choice about a window park for this one Resume call (its door only)."""
    token = _CHOICE.set(dict(choice) if choice else None)
    try:
        yield
    finally:
        _CHOICE.reset(token)


_NOT_OFFERED = {"ok": False, "error": "context_choice_not_offered",
                "message": "This task's current hold does not offer that choice; nothing was changed."}


def _offered(row: Any, choice: Dict[str, Any]) -> bool:
    """Whether pause ``row`` is the window park ``choice`` names and it offers that action: the same pause by id
    and generation, held for the recorded typed cause, and for an overview the one revision it counted and has
    not taken yet."""
    if not (isinstance(row, dict) and row.get("rail") == RAIL_CONTEXT_WINDOW and row.get("cause") == CAUSE
            and isinstance(row.get(FACTS_KEY), dict) and row.get("pause_id") == choice.get("pause_id")
            and row.get("pause_generation") == choice.get("pause_generation")):
        return False
    if "overview_revision" in choice:
        offer = row[FACTS_KEY].get(OVERVIEW_KEY)
        return (isinstance(offer, dict) and not offer.get("taken")
                and offer.get("current_revision") == choice["overview_revision"])
    return choice.get("end_attempt") is True or choice.get("try_again") is True


def _granted(row: Any) -> Dict[str, Any]:
    """The choice a restored pause row's grant carries, only where that row is the hold it names."""
    choice = (row.get("grant") or {}).get(CHOICE_KEY) if isinstance(row, dict) else None
    return dict(choice) if isinstance(choice, dict) and _offered(row, choice) else {}


def bound_choice() -> bool:
    """This request is about the window dialog, not another Resume door."""
    return _CHOICE.get() is not None


def granted_choice(row: Dict[str, Any], stop_identity: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``(grant fields, refusal)`` at the exact grant of pause ``row``: the bound choice rides the single-use grant
    only when that row is the hold it names and offers it; otherwise it refuses, and nothing is granted."""
    choice = _CHOICE.get()
    if not choice:
        return {}, None
    if stop_identity and stop_identity == choice.get("stop_identity") and _offered(row, choice):
        return {CHOICE_KEY: dict(choice)}, None
    return {}, dict(_NOT_OFFERED)


def end_chosen(limit_ctx: Any) -> Any:
    """The owner chose to end this window park's attempt: the loop's result, or ``None`` without that choice.

    Called where the round would otherwise call the model, so every control and a ready answer of the restored
    work have decided before it (``loop.run_llm_loop``). The ending is the size terminal the return met before it
    could park (``loop._handle_provider_unavailable``), with the park's own numbers as the round's refusal: the
    result names them, nothing is sent, and the owner's Continue reads the cause it always read."""
    chosen = _choice(limit_ctx.tools._ctx, limit_ctx.accumulated_usage)
    if not chosen.get("end_attempt"):
        return None
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY
    from ouroboros.loop import _handle_provider_unavailable, _merge_finalization_trace

    usage, facts = limit_ctx.accumulated_usage, chosen.get("facts")
    usage.update(_last_llm_error_kind="context_overflow", reason_code="context_overflow",
                 _last_llm_error="The owner ended this attempt: its saved work did not fit the model's window.")
    if isinstance(facts, dict) and type(facts.get("input_tokens")) is int and type(facts.get("window_tokens")) is int:
        usage[SIZE_REFUSAL_KEY] = {
            "round": limit_ctx.round_idx, "model": str(facts.get("model") or ""), "key": "", "route_fp": "",
            "kind": "input", "tokens": facts["input_tokens"], "basis": "exact_local_count", "reserve": 0,
            "local_window": facts["window_tokens"], "reply_limit": int(facts.get("reply_floor_tokens") or 0)}
    text, usage, trace = _handle_provider_unavailable(limit_ctx, error_kind="context_overflow")
    return text, usage, _merge_finalization_trace(limit_ctx.llm_trace, trace)
