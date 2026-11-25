"""``delegate_to_ouroboros``: a letter to my human's linked Ouroboros on the shared delegate custody (engine kind
``peer``, ``gateways/peer.py``).

The intention row lands before the POST (``record_start_requested``; the invocation id is the letter id). A
receiver refusal is definite and typed with its ``refusal_code``; any other outcome leaves acceptance unknown
(``unconfirmed``) and nothing is sent again on its own: ``retry_of`` sends the recorded letter again, the same
bytes, which the receiver answers with the same record while it keeps that letter's work. The letter's ceiling never
exceeds this work tree's remaining known budget, and a known empty one sends nothing. A reading that does not let a
letter go keeps the exact ceiling it read for the identical request, so that call is the same operation for Safety's
exits; once that ceiling no longer fits, the request is refused with what remains (``sender_budget_lowered``), never
read again with another sum: a smaller one is the mind's explicit new operation. The chain of asking works
grows by this one. The Safety Supervisor reads the letter as it is sent, once, right before each POST
(``safety.check_peer_assignment``); the raw call is left unread and a lookup reads nothing. ``delegate_wait``
reads the letter, ``delegate_cancel`` stops it, a ``needs_input`` question returns as ``waiting_on_user`` and
``delegate_answer`` sends the answer letter; there is no patch to integrate.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from ouroboros.tools.tool_catalog import ToolEntry

log = logging.getLogger(__name__)

TOOL_NAME = "delegate_to_ouroboros"
_STARTED_NOTE = (
    "Read it with delegate_wait(run_id=...), stop it with delegate_cancel; a question from the linked "
    "Ouroboros returns as waiting_on_user and delegate_answer sends the answer letter."
)
_UNCUSTODIED_NOTE = (
    "The receiving Ouroboros accepted this letter, but durable local custody (STARTED) did not land. "
    "The run is held in this process only; do NOT start a duplicate letter. Use delegate_wait or "
    "delegate_cancel with run_id now, or retry_of=pending_invocation_id to rebind custody."
)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _remaining_known_usd(ctx: Any, drive: Any) -> Optional[float]:
    """This work tree's remaining known budget (settled basis): the least room of its ledger caps and of its own
    in-task stop (``task_pacing.CostCeiling``: a letter root's, a place's or an explicit profile's ceiling, which
    stops the tree when its known spend reaches it), or ``None`` when no limit binds or the ledger cannot be read:
    unknown is never a number."""
    from ouroboros.task_pacing import COST_CEILING_ACTIVE
    from ouroboros.usage_admission import task_money_snapshot

    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        snapshot = task_money_snapshot(drive, {"id": task_id}, str(metadata.get("root_task_id") or task_id))
    except Exception:  # noqa: BLE001 - an unreadable ledger is unknown, not a limit
        log.debug("sender tree money snapshot unavailable", exc_info=True)
        return None
    if not isinstance(snapshot, dict):
        return None
    rooms = [_number(snapshot.get("remaining_known_usd"))]
    stop, spent = getattr(ctx, "_cost_ceiling", None), _number(snapshot.get("settled_usd"))
    ceiling = _number(getattr(stop, "ceiling_usd", None)) if getattr(stop, "state", "") == COST_CEILING_ACTIVE else None
    if ceiling is not None and spent is not None:
        rooms.append(ceiling - spent)
    known = [room for room in rooms if room is not None]
    return min(known) if known else None


def _chain(ctx: Any, own_install_id: str, task_id: str) -> List[Dict[str, str]]:
    """The asking works, oldest first: the chain this work's own letter carried (when it is one), then this work."""
    metadata = getattr(ctx, "task_metadata", None)
    letter = metadata.get("peer_letter") if isinstance(metadata, dict) else None
    carried = letter.get("chain") if isinstance(letter, dict) else None
    chain = [dict(row) for row in carried or [] if isinstance(row, dict)]
    return chain + [{"install_id": str(own_install_id or ""), "task_id": str(task_id or "")}]


def _started_result(run_id: str, place: str, link_id: str, letter: Dict[str, Any], *,
                    durable: bool, state: Any = None, phase: Any = None,
                    replayed: bool = False, recovered_via_lookup: bool = False) -> Any:
    from ouroboros.delegate_shared import delegate_result

    payload: Dict[str, Any] = {
        "status": "started" if durable else "started_uncustodied",
        "run_id": run_id, "letter_id": run_id, "place_id": place, "link_id": link_id,
        "state": state, "phase": phase, "cost_ceiling_usd": letter.get("cost_ceiling_usd"),
        "custody_durable": durable,
        **({"replayed": True} if replayed else {}),
        **({"recovered_via_lookup": True} if recovered_via_lookup else {}),
        **({"pending_invocation_id": run_id} if not durable else {}),
        "note": _STARTED_NOTE if durable else _UNCUSTODIED_NOTE,
    }
    return delegate_result(payload)


def _unconfirmed_result(place: str, invocation_id: str, detail: str) -> Any:
    from ouroboros.delegate_shared import delegate_result

    return delegate_result({
        "status": "unconfirmed", "place_id": place, "pending_invocation_id": invocation_id,
        "detail": detail, "note": "Whether the linked Ouroboros accepted this letter is unknown; nothing "
        "was sent again. delegate_to_ouroboros(place_id=..., retry_of=<pending_invocation_id>) checks "
        "whether it arrived and sends the same letter, the same bytes, only if not found."})


def _retry_refusal(ctx: Any, drive: Any, place: str, letter: Dict[str, Any]) -> Optional[tuple]:
    """Why a retry whose letter the receiver did not find may not create it now, ``(code, detail)``, else ``None``:
    what a first send meets before its intention row (``_delegate_to_ouroboros``), held to the recorded bytes a
    retry cannot change, so a recorded ceiling above this tree's remaining known budget no longer goes."""
    from ouroboros import peer_links
    from ouroboros.gateways.peer import peer_outgoing_speech_refusal

    link = peer_links.outgoing_link(place, drive_root=drive)
    if link is None:
        return ("peer_link_unavailable", "no usable link to that place here (unknown, revoked, paused, its code "
                "unconfirmed or not marked mine); nothing was sent")
    if link.remote_unknown_at:
        return ("link_unknown", f"the linked Ouroboros has not known this link since {link.remote_unknown_at}; it "
                "needs a new introduction; nothing was sent")
    refusal = peer_outgoing_speech_refusal(link.drive_root or drive, place)
    if refusal is not None:
        return refusal
    remaining, ceiling = _remaining_known_usd(ctx, drive), _number(letter.get("cost_ceiling_usd"))
    if remaining is not None and remaining <= 0:
        return "sender_budget_exhausted", "this work tree's known budget is spent; nothing was sent"
    if remaining is not None and (ceiling is None or ceiling > remaining):
        recorded = _usd(ceiling) if ceiling else "none"
        return ("sender_budget_exceeded", f"the recorded letter's cost ceiling ({recorded}) "
                f"exceeds this work tree's remaining known budget ({_usd(remaining)}), and retry_of cannot change its "
                "recorded bytes; the lookup did not find it there. A smaller ceiling is a new letter (another id), "
                "issued with an explicit cost_ceiling_usd below what remains and read once; nothing was sent")
    return None


def _kept_ceilings(ctx: Any) -> Dict[str, float]:
    """This task's ceilings that a reading did not let go, by request (``_intent``): ``ctx``'s own, like the words
    ``peer_letters.admit_words`` admitted, so the identical request carries the exact ceiling that was read."""
    kept = getattr(ctx, "_peer_assignment_ceilings", None)
    if not isinstance(kept, dict):
        kept = {}
        ctx._peer_assignment_ceilings = kept
    return kept


def _intent(place: str, text: str, title: str, requested: Optional[float]) -> str:
    """One request of the mind, apart from this tree's budget as it stands: its place, words, title and the ceiling
    it asked for (``None`` when it asked for none)."""
    import hashlib
    import json

    raw = json.dumps([place, text, title, requested], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _usd(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _no_longer_fits(ctx: Any, drive: Any, letter: Dict[str, Any]) -> Optional[tuple]:
    """``(ceiling, remaining)`` when this work tree's remaining known budget, read now, no longer holds the letter's
    exact ceiling (``None`` for a letter without one, which then binds no more); ``None`` when it does or none binds."""
    remaining, ceiling = _remaining_known_usd(ctx, drive), _number(letter.get("cost_ceiling_usd"))
    if remaining is None or (ceiling is not None and ceiling <= remaining):
        return None
    return ceiling, remaining


def _lowered(ceiling: Optional[float], remaining: float) -> str:
    """Why the identical request no longer goes, and the one next step that does."""
    ceiling_text = _usd(ceiling) if ceiling is not None else "none"
    return (f"this letter was read with cost_ceiling_usd {ceiling_text}, and this work tree's remaining known "
            f"budget is now {_usd(remaining)}: the same letter no longer fits. A smaller ceiling is another operation, "
            "which no earlier reading, and no answer of my human's to one, covers: issue it with an explicit "
            "cost_ceiling_usd below what remains, leaving room for your own next steps; it is read once. Nothing was "
            "sent")


_DERIVED = ("\nHost fact: this letter's cost_ceiling_usd ({ceiling}) was not your explicit choice but this work "
            "tree's remaining known budget at this call, which your next steps lower. The identical call carries "
            "that ceiling again only while it still fits, else it is refused with what remains; a letter with an "
            "explicit cost_ceiling_usd you can keep below that is another operation, read once.")


def _send(ctx: Any, drive: Any, task_id: str, ids: Dict[str, str], engine_ref: Dict[str, Any],
          letter: Dict[str, Any], place: str, *, retry: bool = False,
          record: Optional[Dict[str, Any]] = None, keep: Optional[tuple] = None) -> Any:
    """POST one recorded letter through its record's engine under Pause/Stop exclusion and settle the outcome.

    The letter meets its one Safety reading (``safety.check_peer_assignment``; the registry left the raw call unread)
    right before its POST, as the exact dict that is sent: a first send once its intention row landed, a retry only
    when its lookup found nothing there and the recorded letter may still go (``_retry_refusal``). A reading that
    does not let it through sends nothing: a first send's intention row closes unsent, a retry stays pending, and a
    first send keeps the exact ceiling it was read with for the identical request (``keep``: ``_kept_ceilings``, its
    request, whether this tree's budget rather than the mind chose that ceiling). This tree's money is read once more
    after the reading, before the handoff: a ceiling that no longer fits does not go (``sender_budget_lowered``) and
    stays kept; a reading that lets it go and still fits drops it. The note a reading lets it through with trails the
    result, as the registry's would. On ``retry=True``: first
    performs a non-creating lookup (``GET /api/letters/{id}`` via ``delegate_accepted``), which reads and sends
    nothing more, so an already-accepted letter binds cleanly even when the receiver paused the link or Resume
    marked the invocation lookup-only; and NO refusal on a retry is ever marked ``definitely_unrun`` /
    ``failed_definite``.
    """
    from ouroboros import delegate_accepted, delegate_custody as custody, delegate_resume, owner_pause
    from ouroboros.delegate_shared import SUBSTRATE_REFUSAL_CODE, _fail
    from ouroboros.execution_domain import bound_gateway
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.gateways.peer import NOTHING_SENT, UNCONFIRMED
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.safety import check_peer_assignment
    from ouroboros.tools.tool_result import ToolResult, _compose_execute_result_result, safety_refusal

    invocation_id = str(letter["letter_id"])
    shape = {"access": "readonly", "delegated": True}
    entry = custody.RunCustody(
        run_id=invocation_id, task_id=task_id, invocation_id=invocation_id,
        engine_ref=dict(engine_ref), route_id="peer", category="peer", source="peer_letter",
        root_task_id=ids["root"], parent_task_id=ids["parent"], ledger_root=str(drive),
    )

    def _close_unsent(code: str, detail: str) -> bool:
        closed = delegate_resume.close_unsent_start(drive, task_id, invocation_id, code)
        custody.emit(drive, custody.START_FAILED, {
            "run_id": "", "task_id": task_id, "invocation_id": invocation_id,
            "reason": code, "detail": detail[:500], "definite": bool(closed),
        })
        return bool(closed)

    def _pre_post_refusal(code: str, detail: str, **facts: Any) -> Any:
        if retry:
            return _fail(TOOL_NAME, code, f"{detail}; the recorded letter's acceptance stays unknown and "
                         "retry_of may check or send it again once allowed", place_id=place,
                         retry_of=invocation_id, pending_invocation_id=invocation_id,
                         host_code=SUBSTRATE_REFUSAL_CODE, **facts)
        closed = _close_unsent(code, detail)
        return _fail(TOOL_NAME, code, detail, place_id=place, host_code=SUBSTRATE_REFUSAL_CODE,
                     pending_invocation_id=invocation_id, **({"definitely_unrun": True} if closed else {}), **facts)

    def _refused(exc: Any) -> Any:
        """A POST that raised: nothing left, acceptance unknown, or the receiver's typed refusal."""
        if not entered_post or exc.code in NOTHING_SENT:
            return _pre_post_refusal(exc.code, str(exc))
        refusal = str(getattr(exc, "refusal_code", "") or "")
        if exc.code == UNCONFIRMED or not refusal:
            return _unconfirmed_result(place, invocation_id, str(exc))
        if retry:
            # A retry refusal (e.g. link_paused after a lost first reply) never proves the first POST unrun.
            if int(getattr(exc, "status_code", 0) or 0):
                held, read = delegate_accepted.accepted_after_refusal(drive, gateway, invocation_id, exc)
                if held is not None:
                    durable = read.get("reason") != "start_state_not_recorded"
                    if not durable:
                        custody.record_started(drive, entry, shape, publish=False)
                    return _started_result(held.run_id, place, engine_ref["link_id"], letter,
                                           durable=durable, state=read.get("state") or "running",
                                           replayed=True, recovered_via_lookup=True)
            custody.emit(drive, custody.START_FAILED, {
                "run_id": "", "task_id": task_id, "invocation_id": invocation_id,
                "reason": refusal, "detail": str(exc)[:500], "definite": False,
            })
            return _fail(TOOL_NAME, exc.code, f"{exc}; the earlier send's acceptance stays unknown and "
                         "is not marked definitely_unrun", place_id=place, refusal_code=refusal,
                         retry_of=invocation_id, pending_invocation_id=invocation_id)
        definite = refusal != "letter_id_reused"
        custody.emit(drive, custody.START_FAILED, {
            "run_id": "", "task_id": task_id, "invocation_id": invocation_id, "reason": refusal,
            "detail": str(exc)[:500], "definite": definite})
        return _fail(TOOL_NAME, exc.code, str(exc), place_id=place, refusal_code=refusal,
                     pending_invocation_id=invocation_id, **({"definitely_unrun": True} if definite else {}))

    try:
        gateway = bound_gateway(engine_ref)
    except ClaudexorUnavailable as exc:
        if retry:
            return _fail(TOOL_NAME, exc.code, f"{exc}; the recorded letter's acceptance stays unknown and "
                         "retry_of may check or send it again once the link is usable", place_id=place,
                         retry_of=invocation_id, pending_invocation_id=invocation_id,
                         host_code=SUBSTRATE_REFUSAL_CODE)
        return _pre_post_refusal(exc.code, str(exc))

    entered_post, handle, result, note = False, None, None, ""
    try:
        if retry and record is not None:
            held, looked = delegate_accepted.recover(drive, gateway, {**record, "invocation_id": invocation_id})
            if held is not None:
                durable = looked.get("reason") != "start_state_not_recorded"
                if not durable:
                    custody.record_started(drive, entry, shape, publish=False)
                return _started_result(held.run_id, place, engine_ref["link_id"], letter,
                                       durable=durable, state=looked.get("state") or "running",
                                       replayed=True, recovered_via_lookup=True)
            lookup_outcome = str(looked.get("lookup") or "")
            if lookup_outcome == "found":
                # Accepted on receiver, but local STARTED publication failed: never re-POST.
                custody.record_started(drive, entry, shape, publish=False)
                return _started_result(invocation_id, place, engine_ref["link_id"], letter,
                                       durable=False, replayed=True, recovered_via_lookup=True)
            if lookup_outcome == "unavailable":
                return _unconfirmed_result(place, invocation_id, str(looked.get("lookup_detail") or "lookup unavailable"))
            if lookup_outcome == "refused":
                refusal = str(looked.get("refusal_code") or "peer_lookup_refused")
                custody.emit(drive, custody.START_FAILED, {
                    "run_id": "", "task_id": task_id, "invocation_id": invocation_id,
                    "reason": refusal, "detail": str(looked.get("lookup_detail") or "")[:500], "definite": False,
                })
                return _fail(TOOL_NAME, refusal, str(looked.get("lookup_detail") or refusal),
                             place_id=place, refusal_code=refusal, retry_of=invocation_id,
                             pending_invocation_id=invocation_id, host_code=SUBSTRATE_REFUSAL_CODE)
            refused = _retry_refusal(ctx, drive, place, letter)
            if refused is not None:
                return _pre_post_refusal(*refused)
        allowed, note = check_peer_assignment(ctx, place_id=place, link_id=str(engine_ref["link_id"]), letter=letter)
        # The money boundary as it stands after the reading (which may itself have spent), read once before the
        # handoff: an exact ceiling that no longer fits does not go, and is never read again with another sum.
        lowered = _no_longer_fits(ctx, drive, letter) if allowed else None
        if keep is not None:
            kept, intent = keep[0], keep[1]
            if (allowed and lowered is None) or letter.get("cost_ceiling_usd") is None:
                kept.pop(intent, None)
            else:
                kept[intent] = float(letter["cost_ceiling_usd"])
        if not allowed:
            status, code = safety_refusal(note)
            if not retry:  # nothing was sent: the first send's intention row closes
                _close_unsent(code.lower(), note)
            if keep is not None and keep[2]:
                note += _DERIVED.format(ceiling=_usd(float(letter["cost_ceiling_usd"])))
            return ToolResult(status, code, text=note)
        if lowered is not None:
            return _pre_post_refusal("sender_budget_lowered", _lowered(*lowered), read_ceiling_usd=lowered[0],
                                     remaining_known_usd=lowered[1])

        def submit() -> Dict[str, Any]:
            nonlocal entered_post
            entered_post = True
            return gateway.start_run(letter, idempotency_key=invocation_id)

        try:
            handle = owner_pause.run_operation(
                ctx, delegate_resume.checked_start(drive, task_id, invocation_id, submit)
            )
        except delegate_resume.StartHeld as exc:
            result = _pre_post_refusal(exc.reason, str(exc))
        except (owner_pause.OwnerPauseRefused, PhysicalDispatchInterrupted) as exc:
            result = _pre_post_refusal(getattr(exc, "code", "") or str(exc) or "owner_paused", str(exc))
        except ClaudexorUnavailable as exc:
            result = _refused(exc)
    finally:
        gateway.close()

    if handle is not None:
        # The letter's work runs under the receiver's own authority; here it may change nothing of this workspace.
        bound = custody.bind_started(drive, entry, shape=shape, observed_state=str(handle.get("state") or ""))
        run_id = bound.custody.run_id if bound.custody is not None else invocation_id
        result = _started_result(
            run_id, place, engine_ref["link_id"], letter,
            durable=bool(bound.durable), state=handle.get("state"), phase=handle.get("phase"),
            replayed=bool(retry or bound.adopted),
        )
    return _compose_execute_result_result(TOOL_NAME, result, "", note) if note else result


def _delegate_to_ouroboros(ctx: Any, place_id: str = "", text: str = "", title: str = "",
                           cost_ceiling_usd: Optional[float] = None, retry_of: str = "") -> Any:
    from ouroboros import delegate_custody as custody, owner_pause, peer_links
    from ouroboros.delegate_shared import AGENT_FAULT_CODE, SUBSTRATE_REFUSAL_CODE, _fail
    from ouroboros.delegate_start_claims import claim_refusal_fields, claimed_start_request
    from ouroboros.gateway.peer_contracts import MAX_CHAIN, TEXT_MAX_CHARS, TITLE_MAX_CHARS
    from ouroboros.gateways.peer import peer_outgoing_speech_refusal

    place, drive = str(place_id or "").strip(), custody.custody_root(ctx)
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    ids = {"root": str(metadata.get("root_task_id") or task_id), "parent": str(metadata.get("parent_task_id") or "")}
    if not place:
        return _fail(TOOL_NAME, "missing_place_id", "place_id names the linked Ouroboros's place",
                     host_code=AGENT_FAULT_CODE)
    if retry_of:
        return _retry(ctx, drive, task_id, str(retry_of).strip(), place,
                      text=text, title=title, cost_ceiling_usd=cost_ceiling_usd)
    if not str(text or "").strip():
        return _fail(TOOL_NAME, "empty_prompt", "text is the letter: the assignment for the linked Ouroboros",
                     host_code=AGENT_FAULT_CODE)
    if len(str(text)) > TEXT_MAX_CHARS:
        return _fail(TOOL_NAME, "invalid_letter",
                     f"text must be at most {TEXT_MAX_CHARS} characters; nothing was sent",
                     place_id=place, host_code=AGENT_FAULT_CODE, definitely_unrun=True)
    if str(title or "").strip() and len(str(title)) > TITLE_MAX_CHARS:
        return _fail(TOOL_NAME, "invalid_letter",
                     f"title must be at most {TITLE_MAX_CHARS} characters; nothing was sent",
                     place_id=place, host_code=AGENT_FAULT_CODE, definitely_unrun=True)
    link = peer_links.outgoing_link(place, drive_root=drive)
    if link is None:
        return _fail(TOOL_NAME, "peer_link_unavailable", "no usable link to that place here (unknown, revoked, "
                     "paused, its code unconfirmed or not marked mine); nothing was sent", place_id=place,
                     host_code=SUBSTRATE_REFUSAL_CODE, definitely_unrun=True)
    if link.remote_unknown_at:
        return _fail(TOOL_NAME, "link_unknown", f"the linked Ouroboros has not known this link since "
                     f"{link.remote_unknown_at}; it needs a new introduction; nothing was sent", place_id=place,
                     refusal_code="link_unknown", host_code=SUBSTRATE_REFUSAL_CODE, definitely_unrun=True)
    speech_refusal = peer_outgoing_speech_refusal(getattr(link, "drive_root", "") or drive, place)
    if speech_refusal is not None:
        return _fail(TOOL_NAME, speech_refusal[0], speech_refusal[1], place_id=place,
                     host_code=SUBSTRATE_REFUSAL_CODE, definitely_unrun=True)
    requested: Optional[float] = None
    if cost_ceiling_usd is not None:
        if isinstance(cost_ceiling_usd, bool) or not isinstance(cost_ceiling_usd, (int, float)) \
                or not math.isfinite(float(cost_ceiling_usd)) or cost_ceiling_usd <= 0:
            return _fail(TOOL_NAME, "cost_ceiling_invalid", "cost_ceiling_usd must be a positive number",
                         host_code=AGENT_FAULT_CODE, definitely_unrun=True)
        requested = float(cost_ceiling_usd)
    remaining = _remaining_known_usd(ctx, drive)
    if remaining is not None and remaining <= 0:
        return _fail(TOOL_NAME, "sender_budget_exhausted", "this work tree's known budget is spent; nothing was sent",
                     host_code=SUBSTRATE_REFUSAL_CODE, definitely_unrun=True)
    kept = _kept_ceilings(ctx)
    intent = _intent(place, str(text), str(title or "") if str(title or "").strip() else "", requested)
    ceiling = kept.get(intent)  # the exact ceiling a reading of this same request did not let go
    if ceiling is not None and remaining is not None and ceiling > remaining:
        return _fail(TOOL_NAME, "sender_budget_lowered", _lowered(ceiling, remaining), place_id=place,
                     host_code=SUBSTRATE_REFUSAL_CODE, definitely_unrun=True,
                     read_ceiling_usd=ceiling, remaining_known_usd=remaining)
    if ceiling is None and remaining is not None:
        ceiling = remaining if requested is None else min(requested, remaining)
    elif ceiling is None:
        ceiling = requested
    chain = _chain(ctx, link.own_install_id, task_id)
    if len(chain) > MAX_CHAIN:  # the receiver refuses it too: a cut chain would lose its facts and the hop bound
        return _fail(TOOL_NAME, "chain_too_long", f"the chain of asking works would hold {len(chain)} entries; a "
                     f"letter carries at most {MAX_CHAIN}; nothing was sent", place_id=place,
                     host_code=SUBSTRATE_REFUSAL_CODE, definitely_unrun=True)
    invocation_id = custody.new_invocation_id()  # the letter id: [A-Za-z0-9_.-]{1,64}
    letter: Dict[str, Any] = {"letter_id": invocation_id, "text": str(text), "protocol": peer_links.PROTOCOL,
                              "chain": chain,
                              **({"title": str(title)} if str(title or "").strip() else {}),
                              **({"cost_ceiling_usd": ceiling} if ceiling is not None else {})}
    engine_ref = {"kind": "peer", "place_id": place, "link_id": link.link_id}
    try:
        landed, refusal = claimed_start_request(
            drive, actor_ctx=ctx, run_id="", task_id=task_id, idempotency_key=invocation_id,
            invocation_id=invocation_id, request=letter, project_id="", route="peer",
            source="peer_letter", category="peer",
            root_task_id=ids["root"], parent_task_id=ids["parent"], engine_ref=engine_ref,
        )
    except owner_pause.OwnerPauseRefused as exc:
        return _fail(TOOL_NAME, str(exc) or "owner_paused", str(exc), place_id=place,
                     host_code=SUBSTRATE_REFUSAL_CODE, definitely_unrun=True)
    if not landed:
        if not refusal:
            return _fail(TOOL_NAME, "start_custody_unavailable", "the intention row did not land; nothing was sent",
                         place_id=place, host_code=SUBSTRATE_REFUSAL_CODE, definitely_unrun=True)
        reason, detail, facts = claim_refusal_fields(refusal, "")
        return _fail(TOOL_NAME, reason, detail, place_id=place, host_code=SUBSTRATE_REFUSAL_CODE,
                     definitely_unrun=True, **facts)
    return _send(ctx, drive, task_id, ids, engine_ref, letter, place,
                 keep=(kept, intent, ceiling is not None and ceiling != requested))


def _retry(ctx: Any, drive: Any, task_id: str, token: str, place: str, *,
           text: str = "", title: str = "", cost_ceiling_usd: Optional[float] = None) -> Any:
    """The recorded letter of an invocation whose acceptance is unknown, checked (lookup-first) and sent again only
    for that exact recorded send under its recorded ownership and bytes."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_shared import AGENT_FAULT_CODE, _fail
    from ouroboros.execution_domain import is_peer_ref

    record = custody.invocation_record(drive, token)
    if record is None:
        return _fail(TOOL_NAME, "unknown_invocation", "no letter to that place was recorded with that id; a call "
                     "without retry_of writes a new letter", retry_of=token, host_code=AGENT_FAULT_CODE,
                     definitely_unrun=True)
    state = str(record.get("state") or "")
    if state == "legacy_gap":
        return _fail(TOOL_NAME, "invocation_request_unrecorded",
                     "that invocation's original letter request was not recorded; nothing was sent",
                     retry_of=token, host_code=AGENT_FAULT_CODE)
    recorded_task = str(record.get("task_id") or "")
    if not task_id or recorded_task != task_id:
        return _fail(TOOL_NAME, "invocation_not_owned",
                     "that pending letter belongs to another task; only the task that requested it may retry it",
                     retry_of=token, host_code=AGENT_FAULT_CODE)
    ref = record.get("engine_ref") if isinstance(record.get("engine_ref"), dict) else {}
    if not is_peer_ref(ref) or str(ref.get("place_id") or "") != place:
        return _fail(TOOL_NAME, "unknown_invocation", "no letter to that place was recorded with that id; a call "
                     "without retry_of writes a new letter", retry_of=token, host_code=AGENT_FAULT_CODE,
                     definitely_unrun=True)
    if state != "pending" or not isinstance(record.get("request"), dict):
        reason = {"started": "invocation_already_started", "failed_definite": "invocation_definitely_refused"}.get(
            state, "invocation_not_pending")
        return _fail(TOOL_NAME, reason, f"that letter's start is {state or 'unknown'}, not unknown; it is not sent "
                     "again", retry_of=token, run_id=str(record.get("run_id") or ""), host_code=AGENT_FAULT_CODE)
    recorded_request = dict(record["request"])
    if str(text or "").strip() and str(text) != str(recorded_request.get("text") or ""):
        return _fail(TOOL_NAME, "retry_text_mismatch",
                     "retry_of must replay the original letter verbatim (omit text or pass the exact recorded text)",
                     retry_of=token, host_code=AGENT_FAULT_CODE)
    if str(title or "").strip() and str(title) != str(recorded_request.get("title") or ""):
        return _fail(TOOL_NAME, "retry_title_mismatch",
                     "retry_of keeps the recorded title (omit title or pass the exact recorded title)",
                     retry_of=token, host_code=AGENT_FAULT_CODE)
    if cost_ceiling_usd is not None:
        given_ceiling = _number(cost_ceiling_usd)
        if given_ceiling is None or given_ceiling != _number(recorded_request.get("cost_ceiling_usd")):
            return _fail(TOOL_NAME, "retry_ceiling_mismatch",
                         "retry_of keeps the recorded cost_ceiling_usd; omit it or pass the exact recorded value",
                         retry_of=token, host_code=AGENT_FAULT_CODE)
    recorded_ids = {
        "root": str(record.get("root_task_id") or recorded_task),
        "parent": str(record.get("parent_task_id") or ""),
    }
    return _send(ctx, drive, recorded_task, recorded_ids, dict(ref), recorded_request, place,
                 retry=True, record=record)


MESSAGE_TOOL_NAME = "message_ouroboros"


def _reply_target(ctx: Any, place: str) -> str:
    """The linked Ouroboros's message a focus turn's tool send answers: the last input its model covered
    (``presence_continuation.reply_anchor``, the input its automatic reply names too), before version 2 its waking
    message; ``""`` (a standalone message) outside a focus a message of this place woke."""
    from ouroboros.presence_continuation import reply_anchor

    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
    if metadata.get("wake_cause") != "focus" or presence.get("place_id") != place:
        return ""
    event = presence.get("event") if isinstance(presence.get("event"), dict) else {}
    anchor = reply_anchor(ctx, metadata) or {"message": event.get("message")}
    message = anchor.get("message") if isinstance(anchor.get("message"), dict) else {}
    return str(message.get("message_id") or "") if message.get("peer_message") is True else ""


def _message_ouroboros(ctx: Any, place_id: str = "", text: str = "", retry_of: str = "") -> Any:
    """An ordinary message to my human's linked Ouroboros, never a work assignment. A new message (without
    ``retry_of``) takes the current reply address (``_reply_target``) and its deterministic letter id; an explicit
    retry (``retry_of=<letter_id>``) binds to the exact recorded task, place, recipient, verbatim text and original
    ``reply_to`` (ignoring any newer incoming message that arrived meanwhile). Each physical attempt passes this
    task's short launch gate (``owner_pause.run_operation``). It reads as received only beside its delivered receipt
    in this room."""
    from ouroboros import delegate_custody as custody, owner_pause, peer_links
    from ouroboros.delegate_shared import AGENT_FAULT_CODE, SUBSTRATE_REFUSAL_CODE, _fail, delegate_result
    from ouroboros.gateway.peer_contracts import LETTER_ID_PATTERN, TEXT_MAX_CHARS
    from ouroboros.peer_replies import (
        AttemptHeld, message_letter_id, resolve_message_retry, send_peer_message,
    )
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted

    place, drive = str(place_id or "").strip(), custody.custody_root(ctx)
    body_text, retry_id = str(text or ""), str(retry_of or "").strip()
    if not place:
        return _fail(MESSAGE_TOOL_NAME, "missing_place_id", "place_id names the linked Ouroboros's place",
                     host_code=AGENT_FAULT_CODE)
    if not body_text.strip() or len(body_text) > TEXT_MAX_CHARS:
        return _fail(MESSAGE_TOOL_NAME, "invalid_message",
                     f"text must be a non-empty string of at most {TEXT_MAX_CHARS} characters",
                     host_code=AGENT_FAULT_CODE)
    if retry_id and not LETTER_ID_PATTERN.fullmatch(retry_id):
        return _fail(MESSAGE_TOOL_NAME, "invalid_retry_of",
                     "retry_of must be the letter_id returned by an earlier message_ouroboros call",
                     host_code=AGENT_FAULT_CODE)
    try:
        found = peer_links.link_of_place(drive, place)
    except peer_links.PeerLinkError as exc:
        if retry_id:
            return _fail(MESSAGE_TOOL_NAME, "peer_links_unreadable",
                         f"the link registry cannot be read now ({exc}); nothing was sent; "
                         f"keep retry_of={retry_id!r} and the exact same text once storage is readable",
                         place_id=place, letter_id=retry_id, retry_of=retry_id, host_code=SUBSTRATE_REFUSAL_CODE)
        return _fail(MESSAGE_TOOL_NAME, "peer_links_unreadable", f"the link registry cannot be read now ({exc}); "
                     "nothing was sent; call again once storage is readable",
                     place_id=place, host_code=SUBSTRATE_REFUSAL_CODE)
    if found is None:
        return _fail(MESSAGE_TOOL_NAME, "peer_link_unavailable", "no link to that place here; nothing was sent",
                     place_id=place, **({"letter_id": retry_id, "retry_of": retry_id} if retry_id else {}),
                     host_code=SUBSTRATE_REFUSAL_CODE)
    link_id, account_id = found[0], str(found[1].get("peer_install_id") or "")
    task_id = str(getattr(ctx, "task_id", "") or "")

    if retry_id:
        recorded, read_err = resolve_message_retry(drive, link_id, account_id, retry_id)
        if read_err:
            return _fail(MESSAGE_TOOL_NAME, read_err,
                         "earlier message records cannot be read now; nothing was sent; "
                         f"keep retry_of={retry_id!r} and the exact same text once storage is readable",
                         place_id=place, letter_id=retry_id, retry_of=retry_id, link_id=link_id,
                         host_code=SUBSTRATE_REFUSAL_CODE)
        if recorded is None:
            return _fail(MESSAGE_TOOL_NAME, "unknown_message_retry",
                         f"no recorded message_ouroboros letter {retry_id!r} for this place/link",
                         place_id=place, retry_of=retry_id, host_code=AGENT_FAULT_CODE)
        rec_origin = recorded.get("origin") if isinstance(recorded.get("origin"), dict) else {}
        if rec_origin.get("kind") != "tool" or not task_id or str(rec_origin.get("task_id") or "") != task_id:
            return _fail(MESSAGE_TOOL_NAME, "retry_not_owned",
                         "retry_of may retry only a message_ouroboros letter created by this task",
                         place_id=place, retry_of=retry_id, host_code=AGENT_FAULT_CODE)
        rec_place = str(recorded.get("place_id") or "")
        if rec_place and rec_place != place:
            return _fail(MESSAGE_TOOL_NAME, "retry_place_mismatch",
                         "retry_of keeps the original place_id",
                         place_id=place, retry_of=retry_id, host_code=AGENT_FAULT_CODE)
        rec_account = str(recorded.get("account_id") or "")
        if not account_id or rec_account != account_id:
            return _fail(MESSAGE_TOOL_NAME, "retry_recipient_mismatch",
                         "retry_of keeps the original recipient installation",
                         place_id=place, retry_of=retry_id, host_code=AGENT_FAULT_CODE)
        if body_text != str(recorded.get("text") or ""):
            return _fail(MESSAGE_TOOL_NAME, "retry_text_mismatch",
                         "retry_of requires the exact verbatim text of the original message",
                         place_id=place, retry_of=retry_id, host_code=AGENT_FAULT_CODE)
        letter_id = retry_id
        reply_to = str(recorded["reply_to"]) if ("reply_to" in recorded and recorded["reply_to"] is not None) else ""
    else:
        reply_to = _reply_target(ctx, place)
        letter_id = message_letter_id(link_id, task_id, reply_to, body_text)

    def gate(attempt: Any) -> Any:  # a Pause, Stop or deadline before this attempt keeps it from leaving
        try:
            return owner_pause.run_operation(ctx, attempt)
        except (owner_pause.OwnerPauseRefused, PhysicalDispatchInterrupted) as exc:
            code = getattr(exc, "code", "") or str(exc) or "owner_paused"
            raise AttemptHeld(code, f"{exc or code}; nothing was sent") from exc

    sent = send_peer_message(drive, place, letter_id=letter_id, text=body_text, reply_to=reply_to,
                             origin={"kind": "tool", "task_id": task_id}, gate=gate,
                             link_id=link_id, account_id=account_id)
    facts = {"place_id": place, "letter_id": letter_id, "link_id": link_id}
    if sent["state"] == "delivered" and sent["recorded"]:
        again = " It was received earlier: nothing was sent again." if sent.get("earlier") else ""
        return delegate_result({"status": "received", **facts, "note": (
            "The linked Ouroboros saved it in its conversation room, and its delivered receipt is in this room. Saved "
            "is not read: it says nothing of a reading, a wake or work there." + again)})
    if sent["state"] == "held":
        retry_fields = {"letter_id": retry_id, "retry_of": retry_id, "link_id": link_id} if retry_id else {}
        return _fail(MESSAGE_TOOL_NAME, sent["code"], sent["message"], place_id=place, **retry_fields,
                     **({"refusal_code": "link_unknown"} if sent["code"] == "link_unknown" else {}),
                     host_code=SUBSTRATE_REFUSAL_CODE)
    if sent["state"] == "unknown":
        if retry_id:
            return _fail(MESSAGE_TOOL_NAME, sent["code"],
                         f"{sent['message']}; keep retry_of={retry_id!r} and the exact same text once storage is readable",
                         **facts, retry_of=retry_id, host_code=SUBSTRATE_REFUSAL_CODE)
        return _fail(MESSAGE_TOOL_NAME, sent["code"],
                     f"{sent['message']}; call again once storage is readable",
                     place_id=place, link_id=link_id, host_code=SUBSTRATE_REFUSAL_CODE)
    if sent["state"] == "busy":
        return _fail(MESSAGE_TOOL_NAME, sent["code"], sent["message"] + "; its outcome is that call's", **facts)
    if sent["state"] == "failed":
        if sent.get("earlier"):
            return _fail(MESSAGE_TOOL_NAME, sent["code"], sent["message"] + "; other words are another message",
                         **facts)
        return _fail(MESSAGE_TOOL_NAME, sent["code"], sent["message"], refusal_code=sent["code"], **facts)
    if sent["state"] == "retry":
        return _fail(MESSAGE_TOOL_NAME, sent["code"], "the linked Ouroboros could not take it now; call again later "
                     f"with retry_of={letter_id!r} and the same text and the same letter goes again", **facts,
                     **({"retry_after": sent["retry_after"]} if sent.get("retry_after") is not None else {}))
    if sent["state"] == "uncertain" and sent.get("code") == "attempts_spent":
        return _fail(MESSAGE_TOOL_NAME, "attempts_spent", sent["message"], **facts)
    unconfirmed = sent["state"] == "uncertain"
    return _fail(MESSAGE_TOOL_NAME, "unconfirmed" if unconfirmed else "receipt_unrecorded",
                 (f"whether the linked Ouroboros saved it is unknown ({sent.get('message') or 'unconfirmed'})"
                  if unconfirmed else "the linked Ouroboros confirmed it, but this room's receipt could not be written") +
                 f"; call again with retry_of={letter_id!r} and the same text: the same letter goes again, never a second one",
                 **facts)


def get_tools() -> List[ToolEntry]:
    return [ToolEntry("delegate_to_ouroboros", {  # a literal: tests/test_safety_policy.py scans ToolEntry names
        "name": "delegate_to_ouroboros",
        "description": (
            "Write a letter to my human's own linked Ouroboros (a place with connector 'ouroboros' marked mine): "
            "it becomes ordinary work there, starts at once even with its consciousness off, spends that "
            "installation's budget under cost_ceiling_usd (never above this work tree's remaining known budget) "
            "and carries no owner authority; the receiver chooses its own model, effort, access and workspace. "
            "Read it with delegate_wait, stop it with delegate_cancel; its question returns as waiting_on_user and "
            "delegate_answer sends the answer letter; its result arrives in the terminal payload. A refusal names "
            "its refusal_code (link_paused, peer_may_not_start, no_budget, ...). If acceptance is unknown "
            "(unconfirmed), nothing is resent automatically: retry_of is lookup-first (binds an already-accepted "
            "letter without re-POSTing) and otherwise sends only that exact recorded letter (same bytes)."),
        "parameters": {"type": "object", "properties": {
            "place_id": {"type": "string", "description": "The linked Ouroboros's place id (configure_presence list)."},
            "text": {"type": "string", "description": "The letter: the whole assignment, self-contained."},
            "title": {"type": "string", "description": "Optional short title."},
            "cost_ceiling_usd": {"type": "number", "description": "Optional ceiling in USD for the receiver's work; "
                                 "omitted, the receiver's own per-work limit applies."},
            "retry_of": {"type": "string", "description": "pending_invocation_id of an unconfirmed letter to the "
                         "same place: lookup-first (binds without re-sending if accepted) and sends only that exact "
                         "recorded letter (same bytes) if not found; omit text/title/cost_ceiling_usd or pass exact "
                         "recorded values."},
        }, "required": ["place_id"]},
    }, _delegate_to_ouroboros, timeout_sec=120, safety_policy="check"),
        ToolEntry("message_ouroboros", {  # a literal: tests/test_safety_policy.py scans ToolEntry names
            "name": "message_ouroboros",
            "description": (
                "Send a standalone message or note to my human's linked Ouroboros (a place with connector 'ouroboros' "
                "marked mine) without starting a work assignment. Works even when may_start is false on the receiver. "
                "The message is saved in the receiver's conversation room for this link; if the receiver's "
                "Consciousness is Off it stays visible without waking a model turn or task, and if On it follows the "
                "place's attention and permission rules. received means saved there, never read. After an unconfirmed, "
                "retry or receipt_unrecorded outcome (which returns a letter_id), call again with retry_of=<letter_id> "
                "and the exact same text so the original letter and reply address go again, never a second message; if "
                "retrying an existing letter_id hits unreadable storage, keep the same retry_of once storage is "
                "readable; omit retry_of to send a new message. In a focus turn woken by a linked Ouroboros, your "
                "non-silent finish is already sent back as your reply to that link."),
            "parameters": {"type": "object", "properties": {
                "place_id": {"type": "string", "description": "The linked Ouroboros's place id (configure_presence list)."},
                "text": {"type": "string", "description": "The message or note to send (verbatim original text when retry_of is set)."},
                "retry_of": {"type": "string", "description": "Optional letter_id from an earlier message_ouroboros call "
                             "in this task to retry that exact letter under its original reply address."},
            }, "required": ["place_id", "text"]},
        }, _message_ouroboros, timeout_sec=120, safety_policy="check")]
