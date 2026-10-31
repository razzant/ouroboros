"""The paid act of ``plan_task``: one ask to the council over the plan record.

Composition (DESIGN §3): the default council is the review pool, which already
falls back to Main for a valid empty pool; Blocking widens a body plan's first ask
to every marked reviewer, while Advisory requires a counted seat and a reason to narrow;
a mind-named enabled catalog row outside the pool is a participant whose
objections are answered but never hold; children the mind schedules are not
catalog rows and cannot be named. Each participant receives its own host-built
delivery (``plan_packet``; what it is told is selected by ``plan_delivery``): the
stable part, the objective with its author, the owner's verbatim words and room
snapshot, the plan whole or as the delta since the revision it last answered about
(before any revision no plan: a reading of the request, whose replies move no
objection), everything said in the council since the ask it last answered in plus the
peer replies to earlier asks that settled after its last delivery, its own open
objections, the mind's message, the attachments. Continuation is per participant: a
packet seat continues its own exact transcript, a session seat its own thread; a
changed roster elsewhere resets nobody.

Dispatch goes through the existing review substrate (``run_plan_review_slots``)
with release at the dispatch barrier; replies settle into the plan record from the
custody hook (``settle_released_reply``), the exact texts and costs into one
review-ledger record per ask (``plan_intention_ledger``: ``surface="intention"``, a
narrow adapter that never asks the change/coupling reducer for a verdict), and the
last settlement wakes the task mailbox. A seat whose process died before settling is
recovered later from its retained operation (``collect_pending``), never resent. Every
refusal before the dispatch writes nothing and engages no gate, except an ask that NO
named route could take: it leaves a typed attempt fact (nobody called, nothing held) so
the record reads "asked, but no route" instead of "never asked".
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import pathlib
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any, Dict, List, Mapping, Optional

from ouroboros import plan_record
from ouroboros.plan_intention_ledger import settle_ledger_seat, write_intention_record
from ouroboros.tools import plan_delivery, plan_packet
from ouroboros.tools.plan_review_runtime import (
    PLAN_REVIEW_MAX_TOKENS,
    PLAN_REVIEW_SLOT_TIMEOUT_SEC,
    _governance_text,
    _plan_row_from_actor,
    _session_task_text,
    effective_plan_slots,
    plan_deadline_skip,
    plan_fanout_inputs,
    plan_payload_roots,
    plan_review_slots,
    plan_row_typed_facts,
    run_plan_review_slots,
    slot_is_session,
    slot_retrieves,
)
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_PARTICIPANT_FACT_KEYS = ("recovery_binding", "operation_id", "operation_state", "late_result_pending",
                          "awaiting_since", "pending_invocation_id", "delegated_run_id",
                          # B1 typed failure facts and the substrate's disclosures travel with the seat
                          "failure_code", "reset_at", "http_status", "transport_status", "reported_cause", "capability_delta")
# Receipts the substrate produced for this seat (absent when it produced none): the applied
# account, the engine's resolved model, the room-read coverage and the host's read attestation.
_PARTICIPANT_RECEIPT_KEYS = ("resolved_model", "applied_profile", "profile_continuity_receipt", "auth_route_receipt",
                             "room_read_coverage", "host_file_read_attestation")


class PlanAskRefused(Exception):
    """A typed $0 refusal; ``code`` is a tool-result code, ``attempt`` the one fact (asked, no route) the handler records."""

    def __init__(self, code: str, text: str, attempt: Optional[Dict[str, Any]] = None):
        super().__init__(text)
        self.attempt = attempt
        self.code, self.text = code, text


@dataclass
class Participant:
    slot: Any
    owner_marked: bool
    host_added: bool = False

    @property
    def seat_id(self) -> str:
        return str(self.slot.slot_id)


# ------------------------------------------------------------------ roots and body

def planning_roots(ctx: Any) -> tuple[pathlib.Path, pathlib.Path]:
    """``(system_root, active_root)``; a ValueError is the caller's typed refusal."""
    from ouroboros.review_substrate import review_repo_dirs_for

    return review_repo_dirs_for(ctx)


def state_location(ctx: Any) -> tuple[pathlib.Path, str]:
    root = pathlib.Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        raise ValueError("PLAN_TASK_ID_REQUIRED: a plan record belongs to a real task")
    return root, task_id


def plan_is_body(ctx: Any, affected_paths: List[str], *, system_root: pathlib.Path, active_root: pathlib.Path) -> tuple[bool, str]:
    """Body plan = the declared paths change the installed body (``resolve_constitutional``),
    or a recognized body copy at the active root (``review_body_fact``), or own-body research
    the task declared (``context_requires_self_body_docs`` with a recognized body source)."""
    from ouroboros.review_body_fact import BODY_TRUE, body_fact
    from ouroboros.tools import plan_spec

    payload_roots = plan_payload_roots(ctx, list(affected_paths))
    constitutional, note = plan_spec.resolve_constitutional(
        active_root=active_root, system_repo_root=system_root, affected_paths=affected_paths, evidence=(),
        payload_roots=payload_roots)
    if constitutional:
        return True, note
    if affected_paths and pathlib.Path(active_root) != pathlib.Path(system_root):
        under_root, _ = plan_spec.resolve_constitutional(
            active_root=active_root, system_repo_root=active_root, affected_paths=affected_paths, evidence=(),
            payload_roots=payload_roots)
        if under_root and body_fact(active_root, system_repo=system_root, data_dir=ctx.drive_root).body == BODY_TRUE:
            return True, f"body: the declared paths change a recognized copy of Ouroboros's own body ({active_root})"
    if not affected_paths:
        from ouroboros.context_input_selection import _explicit_self_body_docs_flag
        from ouroboros.context_layout import task_design_source

        metadata = getattr(ctx, "task_metadata", {})
        task = dict(metadata) if isinstance(metadata, Mapping) else {}
        task["task_contract"] = getattr(ctx, "task_contract", {})
        if _explicit_self_body_docs_flag(task) is True and task_design_source(ctx, task) is not None:
            return True, "body: own-body research the task declared (context_requires_self_body_docs) with a recognized body source"
    return False, note


# ------------------------------------------------------------------ participants

def _pool_slots(reviewer_effort: str) -> List[Any]:
    try:
        return list(plan_review_slots(default_effort=str(reviewer_effort or "")))
    except ValueError as exc:
        raise PlanAskRefused("CAPABILITY_UNAVAILABLE",
                             f"ERROR: PLAN_ASK_REVIEW_POOL_INVALID: the review pool cannot be read — {exc}. "
                             "Fix the Reviewer rows on the Agents tab in Settings. Nothing was written.") from exc


def resolve_participants(ask: Any, *, first_body_ask: bool, reviewer_effort: str = "",
                         reason: str = "") -> tuple[List[Participant], Dict[str, Any]]:
    """Use the shared composer; only Blocking tops up the first own-body ask."""
    from ouroboros.reviewer_slot_config import PanelCompositionError, compose_review_panel

    try:
        panel = compose_review_panel(() if ask == "council" else ask, reason,
            surface="plan_first" if first_body_ask else "plan", pool=_pool_slots(reviewer_effort),
            reviewer_effort=reviewer_effort)
    except PanelCompositionError as exc:
        detail = str(exc).replace("REVIEWER_UNKNOWN:", "PARTICIPANT_UNKNOWN:", 1)
        raise PlanAskRefused("TOOL_ARG_ERROR", f"ERROR: PLAN_{detail} Nothing was written.") from exc
    requested = set(panel.facts["reviewers_requested"])
    widened = bool(panel.facts["widened_from"])
    # Resolve handles before deciding which counted participants the host added.
    from ouroboros.reviewer_slot_config import catalog_review_row
    for name in tuple(requested):
        try:
            requested.add(catalog_review_row(None, name).slot_id)
        except ValueError:
            pass  # Main and test-provided pool seats already use their stored id.
    seats = sorted(panel.seats, key=lambda seat: seat.slot.slot_id not in requested) if widened else panel.seats
    return ([Participant(replace(seat.slot, role_hint="plan reviewer", timeout_sec=PLAN_REVIEW_SLOT_TIMEOUT_SEC,
                                max_tokens=PLAN_REVIEW_MAX_TOKENS, default_temperature=0.2), owner_marked=not seat.additional,
                         host_added=widened and not seat.additional and seat.slot.slot_id not in requested)
             for seat in seats], panel.facts)


def first_body_ask(record: Mapping[str, Any], revision: Optional[Mapping[str, Any]], *, body: bool = False) -> bool:
    """The first ask about a body intention, counted apart for readings and revisions: the
    first ask about a body revision, and the first reading of declared body work (``revision=None``
    with the task's own ``body`` fact). A reading never spends the revision's first ask; a later
    reading names its participants like any later ask."""
    if not (revision.get("body") if revision is not None else body):
        return False
    for ask in record.get("asks") or []:
        if revision is None:  # an earlier first reading of the body spent the reading's own floor
            if ask.get("revision") is None and ask.get("body_first_ask"):
                return False
            continue
        earlier = plan_record.revision_by_number(record, ask.get("revision"))
        if earlier is not None and earlier.get("body"):
            return False
    return True


# ------------------------------------------------------------------ delivery

def _evidence_deny_paths(ctx: Any) -> list[str]:
    """Paths evidence may never attach: the runtime data plane and the live settings file."""
    from ouroboros import config as _config

    out: list[str] = []
    for value in (getattr(_config, "SETTINGS_PATH", ""), getattr(_config, "DATA_DIR", "")):
        if str(value or "").strip():
            out.append(str(value).strip())
    try:
        from ouroboros.tool_access import canonical_data_root

        drive = canonical_data_root(ctx)
        if drive:
            out.append(str(drive))
    except Exception:
        pass
    return out


def _readable_roots(ctx: Any, system_root: pathlib.Path, active_root: pathlib.Path) -> List[pathlib.Path]:
    roots = [pathlib.Path(active_root), pathlib.Path(system_root)]
    try:
        from ouroboros.tool_access import profile_readable_root_paths

        roots.extend(path for _label, path in profile_readable_root_paths(ctx, operation="read"))
    except Exception:
        log.debug("readable roots unavailable; attachments resolve against the active and system roots", exc_info=True)
    return list(dict.fromkeys(roots))


def attachment_manifest(ctx: Any, attach: List[str], *, state_root: pathlib.Path, task_id: str,
                        system_root: pathlib.Path, active_root: pathlib.Path) -> Dict[str, Any]:
    """The attached evidence, bounded, every absence named (``plan_evidence``), resolved
    against the task's readable roots."""
    from ouroboros.tools import plan_evidence
    from ouroboros.tools.plan_dialogue import plan_chat_reader

    manifest = plan_evidence.resolve_evidence(
        list(attach or []), active_root=active_root, allowed_roots=_readable_roots(ctx, system_root, active_root),
        resolve_task=plan_evidence.task_evidence_reader(state_root), deny_paths=_evidence_deny_paths(ctx),
        resolve_chat=plan_chat_reader(state_root, task_id))
    manifest["declared"] = list(attach or [])
    return manifest


def _task_objective(ctx: Any) -> str:
    contract = getattr(ctx, "task_contract", {})
    metadata = getattr(ctx, "task_metadata", {})
    if not isinstance(contract, dict) or not contract:
        contract = metadata.get("task_contract") if isinstance(metadata, dict) else {}
    contract = contract if isinstance(contract, dict) else {}
    objective = str(contract.get("objective") or contract.get("description") or "")
    context = str(contract.get("context") or "").strip()
    if context:
        objective += f"\n\nContract context: {context}"
    return objective


def stable_prompt(ctx: Any, *, body: bool, system_root: pathlib.Path, enforcement: str, by_retrieval: bool) -> str:
    from ouroboros.context_layout import design_navigation
    from ouroboros.tools.review_checklist import load_checklist_section

    try:
        checklist = load_checklist_section("Plan Review Checklist")
    except Exception as exc:
        log.warning("Could not load the Plan Review Checklist section: %s", exc)
        checklist = ""
    bible = _governance_text(system_root, "BIBLE.md") if body else ""
    architecture = _governance_text(system_root, "docs/ARCHITECTURE.md") if body else ""
    design_map = (design_navigation(_governance_text(system_root, "docs/DESIGN.md") or None,
                                    source=str(pathlib.Path(system_root) / "docs/DESIGN.md")) if body else None)
    return plan_packet.build_stable_prompt(
        checklist_section=checklist, body=body, enforcement=enforcement,
        bible_text=bible or None, architecture_text=architecture or None,
        bible_locator=str(pathlib.Path(system_root) / "BIBLE.md"),
        architecture_locator=str(pathlib.Path(system_root) / "docs" / "ARCHITECTURE.md"),
        governance_by_retrieval=by_retrieval, design_nav_map=design_map)


def _prior_transcript(state_root: pathlib.Path, task_id: str, participant_row: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """A packet seat's exact prior request messages plus its answer, from the sources the
    ask's ledger record retained; ``None`` when they cannot be read (a fresh send, disclosed)."""
    from ouroboros.review_ledger import read_source

    try:
        messages = json.loads(read_source(state_root, task_id, participant_row["delivery_ref"]).decode("utf-8"))
        answer = read_source(state_root, task_id, participant_row["reply_ref"]).decode("utf-8")
    except (KeyError, TypeError, ValueError, OSError):
        return None
    if not isinstance(messages, list) or not messages:
        return None
    return [*messages, {"role": "assistant", "content": answer}]


def participant_content(ctx: Any, record: Mapping[str, Any], revision: Optional[Mapping[str, Any]], participant: Participant, *,
                        message: str, manifest: Mapping[str, Any], dialogue_section: str,
                        legacy: Optional[Mapping[str, Any]] = None, state_root: Any = None, task_id: str = "",
                        objective_author: str = "", snapshot_section: str = "") -> str:
    """One participant's user content: whole plan the first time, the delta since the revision
    it last answered about afterwards (none before any revision: a reading of the request); the
    council since its last answer; its own open objections (an imported legacy one included: its
    raiser must see what holds the task); message; attachments."""
    seat_id = participant.seat_id
    last_ask, _row = plan_delivery.last_answer_row(record, seat_id)
    first_time = last_ask is None
    hydrate = state_root is not None and bool(task_id)
    delta = None
    if not first_time and revision is not None:
        seen = plan_record.revision_by_number(record, last_ask.get("revision"))
        if seen is not None and hydrate:
            seen = plan_delivery.delivered_revision(state_root, task_id, seen)
        delta = plan_record.revision_delta(seen, revision) if seen is not None else None
        if delta is not None and seen.get("plan_unavailable"):
            delta["plan_previous_unavailable"] = seen["plan_unavailable"]
    own = [o for o in plan_record.open_objections(record, legacy) if o.get("seat_id") == seat_id]
    # The council since the ask it last answered in, preceded by the peer replies to earlier asks
    # that settled after that delivery was built (selected by answered_at, never lost to the index).
    council = [*plan_delivery.late_peer_replies(record, seat_id, last_ask),
               *plan_delivery.asks_since(record, str((last_ask or {}).get("ask_id") or ""))]
    if hydrate:
        council = plan_delivery.delivered_asks(state_root, task_id, council)
    cross_talk = plan_packet.render_cross_talk(council, seat_id=seat_id)
    return plan_packet.build_participant_content(
        objective=_task_objective(ctx), owner_words=str(manifest.get("owner_words") or ""),
        dialogue_section=dialogue_section, revision=revision, delta=delta, cross_talk=cross_talk,
        own_objections=own, message=message, evidence_section=plan_packet.render_evidence(manifest), first_time=first_time,
        objective_author=objective_author, snapshot_section=snapshot_section)


# ------------------------------------------------------------------ prepare

@dataclass
class PreparedAsk:
    ask_id: str
    retry_key: str
    participants: List[Participant]
    callable: List[Any]
    not_sendable: Dict[str, str]
    system_prompt: str
    session_prompt: str
    contents: Dict[str, str]
    session_tasks: Dict[str, str]
    delivery: Dict[str, Any]
    session_threads: Dict[str, str]
    continued: Dict[str, bool]
    manifest: Dict[str, Any]
    message: str
    attach: List[str]
    reviewer_effort: str
    body: bool
    body_first_ask: bool
    enforcement: str
    system_root: pathlib.Path
    active_root: pathlib.Path
    panel: Dict[str, Any] = field(default_factory=dict)


def prepare_ask(ctx: Any, *, record: Mapping[str, Any], revision: Optional[Mapping[str, Any]], ask: Any, message: str,
                attach: List[str], reviewer_effort: str, state_root: pathlib.Path, task_id: str,
                legacy: Optional[Mapping[str, Any]] = None, reason: str = "") -> PreparedAsk:
    """Everything that can refuse, before anything is written: participants, the deadline rail, each
    participant's delivery, fit and lane health, the budget admission (``legacy``: the imported v2 state).
    ``revision=None`` is a reading of the request before any plan: no paths are declared, so its body
    fact is the task's own declaration (``plan_is_body`` with no paths)."""
    from ouroboros.config import get_review_enforcement
    from ouroboros.owner_words import objective_author_text
    from ouroboros.tools.plan_dialogue import attach_own_dialogue, dialogue_slot_inputs, render_dialogue, session_input_limits
    from ouroboros.tools.review_helpers import review_wave_binding_fence, review_wave_budget_gate

    system_root, active_root = planning_roots(ctx)
    body = (bool(revision.get("body")) if revision is not None
            else plan_is_body(ctx, [], system_root=system_root, active_root=active_root)[0])
    body_first = first_body_ask(record, revision, body=body)
    participants, panel = resolve_participants(ask, first_body_ask=body_first, reviewer_effort=reviewer_effort, reason=reason)
    if not participants:
        raise PlanAskRefused("CAPABILITY_UNAVAILABLE", "ERROR: PLAN_ASK_EMPTY: no participant could be named. Nothing was written.")
    deadline = plan_deadline_skip(ctx)
    if deadline:
        plan_deadline_skip(ctx, emit=True)
        raise PlanAskRefused("CAPABILITY_UNAVAILABLE", "ERROR: PLAN_ASK_SKIPPED_DEADLINE: " + deadline + " Nothing was written.")
    ask_id = plan_record.next_ask_id(record)
    retry_key = f"plan_ask:{task_id}:{ask_id}"
    enforcement = str(get_review_enforcement() or "")
    slots = effective_plan_slots([p.slot for p in participants])
    by_seat = {str(s.slot_id): s for s in slots}
    for participant in participants:
        participant.slot = by_seat.get(participant.seat_id, participant.slot)
    manifest = attachment_manifest(ctx, attach, state_root=state_root, task_id=task_id, system_root=system_root, active_root=active_root)
    manifest = attach_own_dialogue(ctx, state_root, manifest, f"plan-ask-{task_id}-{ask_id}", persist=True)
    dialogue_section, snapshot_section = render_dialogue(manifest)  # rows before the plan, capture facts after
    try:
        system_prompt = stable_prompt(ctx, body=body, system_root=system_root, enforcement=enforcement, by_retrieval=False)
        session_prompt = stable_prompt(ctx, body=body, system_root=system_root, enforcement=enforcement, by_retrieval=True)
    except plan_packet.PlanPacketError as exc:
        raise PlanAskRefused("TOOL_ERROR", f"ERROR: PLAN_ASK_PACK_INCOMPLETE: {exc}. Nothing was written.") from exc
    contents, session_tasks, slot_messages, session_threads, continued = {}, {}, {}, {}, {}
    author = objective_author_text(ctx)  # one typed fact per task: byte-stable across its asks (the cached prefix)
    for participant in participants:
        sid = participant.seat_id
        contents[sid] = participant_content(ctx, record, revision, participant, message=message, manifest=manifest,
                                            dialogue_section=dialogue_section, legacy=legacy, state_root=state_root, task_id=task_id,
                                            objective_author=author, snapshot_section=snapshot_section)
        session_tasks[sid] = _session_task_text(session_prompt, contents[sid], str(active_root))
        _ask, prior = plan_delivery.last_answer_row(record, sid)
        if prior is None:
            continue
        if slot_is_session(participant.slot):
            thread = str(prior.get("review_thread_id") or "")
            if thread:
                session_threads[sid] = thread
                continued[sid] = True
        elif not slot_retrieves(participant.slot):
            transcript = _prior_transcript(state_root, task_id, prior)
            if transcript is not None:
                # The stable prompt is the host's standing frame, not the seat's words: a seat
                # continued from an earlier ask carries the CURRENT one, so the governance pack
                # reaches it the moment the plan becomes a body plan (DESIGN §3, Q4=A).
                if transcript and str(transcript[0].get("role") or "") == "system":
                    transcript = [{"role": "system", "content": system_prompt}, *transcript[1:]]
                slot_messages[sid] = [*transcript, {"role": "user", "content": contents[sid]}]
                continued[sid] = True
    merged: Dict[str, Any] = {"slot_messages": {}, "slot_session_tasks": {}, "slot_prompt_chars": {}, "dialogue_delivery": {},
                              "continuation_restarted": {}, "request_policy": {}}
    limits = session_input_limits(slots)
    for participant in participants:
        sid = participant.seat_id
        one = dialogue_slot_inputs(
            [participant.slot], system_prompt=system_prompt, user_content=contents[sid], session_task=session_tasks[sid],
            manifest=manifest, slot_messages={sid: slot_messages[sid]} if sid in slot_messages else {},
            native_mandatory_chars=len(system_prompt) + len(contents[sid]), data_root=state_root,
            session_root=str(active_root), task_id=task_id, session_limits={sid: limits[sid]} if sid in limits else {},
            output_contract=plan_packet.PLAN_REPLY_CONTRACT)
        for key in ("slot_messages", "slot_session_tasks", "slot_prompt_chars", "dialogue_delivery", "continuation_restarted"):
            merged[key].update(one.get(key) or {})
        merged["request_policy"] = one.get("request_policy") or merged["request_policy"]
    for sid in merged["continuation_restarted"]:
        continued.pop(sid, None)
    fanout = plan_fanout_inputs(slots, prompt_chars=max([len(system_prompt) + len(c) for c in contents.values()] or [0]),
                                slot_prompt_chars=merged["slot_prompt_chars"])
    callable_slots = list(fanout.get("callable_slots") or [])
    not_sendable = {str(row.get("slot_id") or ""): str(row.get("error") or "not sendable")
                    for row in [*fanout.get("health_skip_rows", []), *fanout.get("oversize_rows", [])]}
    if not callable_slots:
        raise PlanAskRefused("CAPABILITY_UNAVAILABLE",
                             "ERROR: PLAN_ASK_NOTHING_SENDABLE: no named participant can take this plan — "
                             + "; ".join(f"{sid}: {why}" for sid, why in not_sendable.items()) + ". Nobody was called.",
                             attempt={"kind": "no_route", "ask": copy.deepcopy(ask), "participants": [p.seat_id for p in participants],
                                      "not_sendable": dict(not_sendable)})
    admission = review_wave_budget_gate(
        ctx, surface="plan_review", models=[str(s.model) for s in callable_slots],
        prompt_chars=[int(merged["slot_prompt_chars"].get(str(s.slot_id), len(system_prompt))) for s in callable_slots],
        max_completion_tokens=PLAN_REVIEW_MAX_TOKENS)
    if admission is not None:
        fence, remedy = review_wave_binding_fence(admission)
        raise PlanAskRefused("CAPABILITY_UNAVAILABLE",
                             f"ERROR: PLAN_ASK_SKIPPED_BUDGET: the ask was declined before dispatch — known spend has reached "
                             f"the {fence} (remaining ${admission.get('remaining_usd')}; the ask was estimated at "
                             f"~${admission.get('estimated_wave_usd')}). To continue, {remedy}. Nothing was written.")
    return PreparedAsk(
        ask_id=ask_id, retry_key=retry_key, participants=participants, callable=callable_slots, not_sendable=not_sendable,
        system_prompt=system_prompt, session_prompt=session_prompt, contents=contents, session_tasks=session_tasks,
        delivery=merged, session_threads=session_threads, continued=continued, manifest=manifest, message=message,
        attach=list(attach or []), reviewer_effort=str(reviewer_effort or ""), body=body, body_first_ask=body_first,
        enforcement=enforcement, system_root=system_root, active_root=active_root, panel=panel)


# ------------------------------------------------------------------ dispatch

def run_plan_coroutine(coro: Any) -> Any:
    """Run one coroutine to completion from the synchronous tool handler (the ToolEntry
    envelope is the outer bound; ``copy_context`` carries the registry's result sidecar)."""
    import concurrent.futures
    import contextvars

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(contextvars.copy_context().run, asyncio.run, coro).result()


def frozen_seat(slot: Any) -> Dict[str, Any]:
    route = getattr(slot, "route", "api_chat")
    return {
        "slot_id": str(getattr(slot, "slot_id", "") or ""), "model": str(getattr(slot, "model", "") or ""),
        "effort": str(getattr(slot, "effort", "") or ""), "route": str(getattr(route, "value", route) or "api_chat"),
        "session_target": str(getattr(slot, "session_target", "") or ""),
        "session_profile": str(getattr(slot, "session_profile", "") or ""),
        "delivery": "native" if getattr(slot, "native_retrieval", False) else (
            "session" if str(getattr(route, "value", route)) == "agent_session" else "packet"),
        "subagent_id": str(getattr(slot, "subagent_id", "") or ""), "use_local": bool(getattr(slot, "use_local", False)),
        "processing_preference": str(getattr(slot, "processing_preference", "") or ""),
        "session_access": str(getattr(slot, "session_access", "") or ""),
    }


def reply_facts_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """One substrate row → the participant facts the record keeps (status, reply or its
    typed absence, cost, custody facts)."""
    facts: Dict[str, Any] = {
        "model": str(row.get("model") or ""), "cost": row.get("cost"),
        "error": str(row.get("error") or ""), "failure_code": str(row.get("failure_code") or ""),
        "reset_at": str(row.get("reset_at") or ""), "review_thread_id": str(row.get("review_thread_id") or ""),
        "prompt_ref": row.get("prompt_ref") or {}, "response_ref": row.get("response_ref") or {},
        **{k: v for k, v in plan_row_typed_facts(dict(row)).items() if k in _PARTICIPANT_FACT_KEYS},
        **{k: row[k] for k in _PARTICIPANT_RECEIPT_KEYS if row.get(k)},
        **({"review_controller": dict(row["review_controller"])} if isinstance(row.get("review_controller"), dict) and row["review_controller"] else {}),
    }
    state = str(row.get("operation_state") or "settled")
    facts["unresolved"] = False  # a settled row resolves an earlier unknown outcome; the pending branch below may re-raise it
    if state == "not_dispatched":
        # A typed $0 refusal before the paid boundary (a spent window, a dead route, the deadline before
        # the first send): the seat never reached a reviewer, so the dispatch flag the barrier wrote is
        # withdrawn here — the one conversion both the barrier rows and a late settlement pass through.
        facts.update(status="not_dispatched", not_dispatched_reason=str(row.get("error") or "not dispatched"), dispatched=False)
    elif row.get("late_result_pending") or state in {"pending_dispatch", "in_flight", "custody_lost"}:
        facts.update(status="pending", unresolved=state == "custody_lost")
    elif row.get("error"):
        facts.update(status="failed")
    else:
        reply, parse_error = plan_record.parse_reply(str(row.get("text") or ""))
        if reply is None:
            facts.update(status="unparseable", parse_error=parse_error)
        else:
            facts.update(status="answered", reply={**reply, "prose_inline": True}, answered_at=utc_now_iso())
    return facts


def _reference(ctx: Any, task_id: str, record: Mapping[str, Any], state_root: pathlib.Path) -> None:
    from ouroboros.tools.plan_review_references import _emit_review_reference

    try:
        if str(getattr(ctx, "task_id", "") or "") == str(task_id):
            from ouroboros.review_history_view import queue_review_history_context

            queue_review_history_context(ctx, family="plan")
        _emit_review_reference(ctx, task_id, record, surface="plan_review", state_root=state_root)
    except Exception:
        log.debug("plan record reference emission failed", exc_info=True)


def _progress(ctx: Any, text: str) -> None:
    try:
        ctx.emit_progress_fn(text)
    except Exception:
        log.debug("plan progress line failed", exc_info=True)


def dispatch_ask(ctx: Any, prepared: PreparedAsk, *, revision: Optional[Mapping[str, Any]], state_root: pathlib.Path,
                 task_id: str) -> Dict[str, Any]:
    """Record the ask, send it, record the dispatch barrier. Returns the plan record. ``revision=None``
    is a reading of the request: the ask row and its ledger record name no revision."""
    from ouroboros import review_ledger as ledger
    from ouroboros.tools.plan_review_runtime import plan_reviewer_config_fingerprint

    number = (revision or {}).get("n")
    record_id = ledger.new_record_id()
    base_facts = {str(s.slot_id): {"status": "pending"} for s in prepared.callable}
    for sid, why in prepared.not_sendable.items():
        base_facts[sid] = {"status": "not_dispatched", "not_dispatched_reason": why, "operation_state": "not_dispatched"}
    delivery_refs = write_intention_record(ctx, prepared, base_facts, record_id=record_id, state_root=state_root, task_id=task_id,
                                           revision=revision)
    participants = []
    for participant in prepared.participants:
        sid = participant.seat_id
        participants.append({
            "seat_id": sid, "model": str(participant.slot.model), "route": frozen_seat(participant.slot)["route"],
            "retrieves": bool(slot_retrieves(participant.slot)), "owner_marked": bool(participant.owner_marked),
            "host_added": bool(participant.host_added), "dispatched": sid in base_facts and base_facts[sid]["status"] == "pending",
            "continued": bool(prepared.continued.get(sid)), "slot": frozen_seat(participant.slot),
            "delivery_ref": delivery_refs.get(sid), **base_facts[sid],
        })
    record = plan_record.append_ask(state_root, task_id, {
        "ask_id": prepared.ask_id, "ts": utc_now_iso(), "revision": number, "message": prepared.message,
        "attach": list(prepared.attach), "attached": [{k: a.get(k) for k in ("locator", "sha256", "bytes")} for a in prepared.manifest.get("attached") or []],
        "omissions": list(prepared.manifest.get("omissions") or []), "reviewer_effort": prepared.reviewer_effort,
        "body_first_ask": bool(prepared.body_first_ask), "panel": prepared.panel, "reason": prepared.panel.get("reason", ""),
        **_dialogue_carrier(prepared.manifest),
        "host_added": [p.seat_id for p in prepared.participants if p.host_added],
        "participants": participants, "record_id": record_id, "retry_key": prepared.retry_key,
    })
    _reference(ctx, task_id, record, state_root)
    sent = [str(s.slot_id) for s in prepared.callable]
    _progress(ctx, f"📐 Plan: ask {prepared.ask_id} on {plan_record.ask_subject(number)} — sending to {len(sent)} participant"
                   f"{'' if len(sent) == 1 else 's'}" + (f", {len(prepared.not_sendable)} not sent" if prepared.not_sendable else "")
                   + ((", body plan" if revision is not None else ", Ouroboros's own body") if prepared.body else "") + "…")
    first = str(prepared.callable[0].slot_id)
    try:
        rows = run_plan_coroutine(run_plan_review_slots(
            ctx, list(prepared.callable), system_prompt=prepared.system_prompt, user_content=prepared.contents.get(first, ""),
            session_task=prepared.session_tasks.get(first, ""), session_root=str(prepared.active_root),
            output_contract=plan_packet.PLAN_REPLY_CONTRACT, slot_messages=prepared.delivery["slot_messages"],
            slot_session_tasks=prepared.delivery["slot_session_tasks"], request_policy=prepared.delivery["request_policy"],
            session_threads=prepared.session_threads, retry_key=prepared.retry_key, release_at_dispatch=True,
            reconciliation_identity={"subject_hash": prepared.retry_key, "roster_hash": plan_reviewer_config_fingerprint(prepared.callable),
                                     "epoch": prepared.retry_key, "health_epoch": [], "plan_ask_id": prepared.ask_id,
                                     "plan_revision": number}))
    except Exception as exc:
        log.error("plan ask %s dispatch failed: %s", prepared.ask_id, exc, exc_info=True)
        # The substrate already held every sent seat: an exception after the hand-off is UNKNOWN —
        # never proof that no send happened or that nothing was paid. The seats stay pending and
        # unresolved (a reply that still arrives settles them through the custody hook), the barrier's
        # dispatched fact stands, and the mind is told a new ask may pay twice. Missing state is never
        # cleaned into "nobody reached".
        unknown = {"status": "pending", "unresolved": True, "error": f"dispatch outcome unknown: {exc}"}
        for sid in sent:
            record = plan_record.update_participant(state_root, task_id, ask_id=prepared.ask_id, seat_id=sid, facts=unknown)
            settle_ledger_seat(state_root, task_id, record_id, sid, unknown)
        _progress(ctx, f"📐 Plan: ask {prepared.ask_id} — dispatch outcome unknown ({type(exc).__name__}); the {len(sent)} sent "
                       f"participant{'' if len(sent) == 1 else 's'} stay pending, a reply may still arrive; a new ask may pay twice.")
        _reference(ctx, task_id, record, state_root)
        return record
    for row in rows:
        sid = str(row.get("slot_id") or "")
        facts = reply_facts_from_row(row)
        current = next((p for a in record.get("asks") or [] if a.get("ask_id") == prepared.ask_id for p in a.get("participants") or []
                        if str(p.get("seat_id") or "") == sid), None)
        # a seat the custody hook settled meanwhile keeps its answer: ``update_participant`` refuses the pending echo
        record = plan_record.update_participant(state_root, task_id, ask_id=prepared.ask_id, seat_id=sid, facts=facts)
        if facts.get("status") != "pending":
            reply_ref = settle_ledger_seat(state_root, task_id, record_id, sid, facts, raw_text=str(row.get("text") or ""))
            if reply_ref:
                record = plan_record.update_participant(state_root, task_id, ask_id=prepared.ask_id, seat_id=sid, facts={"reply_ref": reply_ref})
            _announce_reply(ctx, current or {"seat_id": sid}, facts)
    _reference(ctx, task_id, record, state_root)
    return record


# ------------------------------------------------------------------ settlement

def _dialogue_carrier(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The ask row's carrier of the own-room snapshot every participant saw: the exact source
    the ``chat:<id>@<sha256>`` locator in their delivery resolves to (``plan_chat_reader``) and
    the source closure keeps through promotion and cleanup."""
    own = manifest.get("own_dialogue") if isinstance(manifest.get("own_dialogue"), Mapping) else {}
    ref = own.get("source_ref") if isinstance(own.get("source_ref"), Mapping) else None
    if not ref or not own.get("sha256"):
        return {}
    return {"dialogue_ref": {"chat_id": own.get("chat_id"), "sha256": str(own.get("sha256")), "ref": dict(ref)}}


def _actor_row(actor: Any, slot: Any) -> Dict[str, Any]:
    raw = asdict(actor) if is_dataclass(actor) else dict(actor if isinstance(actor, Mapping) else getattr(actor, "__dict__", {}))
    return _plan_row_from_actor(raw, slot)


def _ask_and_participant(record: Mapping[str, Any], ask_id: str, seat_id: str) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    ask = next((a for a in record.get("asks") or [] if str(a.get("ask_id") or "") == ask_id), None)
    if ask is None:
        return None, None
    return ask, next((p for p in ask.get("participants") or [] if str(p.get("seat_id") or "") == seat_id), None)


def _announce_reply(ctx: Any, participant: Mapping[str, Any], facts: Mapping[str, Any]) -> None:
    who = str(facts.get("model") or participant.get("model") or participant.get("seat_id") or "a participant")
    status = str(facts.get("status") or "")
    if status == "answered":
        reply = facts.get("reply") or {}
        n_obj, n_with, n_bad = len(reply.get("objections") or []), len(reply.get("withdrawn") or []), len(reply.get("ignored") or [])
        _progress(ctx, f"📐 Plan: {who} answered — {n_obj} objection{'' if n_obj == 1 else 's'}"
                       + (f", {n_with} withdrawn" if n_with else "")
                       + (f", {n_bad} malformed item{'' if n_bad == 1 else 's'} ignored" if n_bad else "") + ".")
    elif status == "unparseable":
        _progress(ctx, f"📐 Plan: {who} answered, but the reply could not be read as the reply object; nothing changed.")
    elif status == "failed":  # typed facts, never the raw provider error with its retry coach
        _progress(ctx, f"📐 Plan: {who} didn't answer: {plan_packet.failure_tag({**participant, **facts})}.")


def record_reply(ctx: Any, *, state_root: pathlib.Path, task_id: str, ask_id: str, seat_id: str, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Settle one seat's row into the record and the ledger; ``None`` when the ask or seat is unknown."""
    record = plan_record.load_plan_record(state_root, task_id)
    ask, participant = _ask_and_participant(record, ask_id, seat_id)
    if ask is None or participant is None:
        return None
    facts = reply_facts_from_row(row)
    if facts.get("status") == "pending" and participant.get("status") != "pending":
        return record  # a settled seat is never re-opened by a late pending echo
    if facts.get("status") == "pending" and all(participant.get(key) == value for key, value in facts.items()):
        return record  # unchanged custody loss is an observation, not another task write
    record = plan_record.update_participant(state_root, task_id, ask_id=ask_id, seat_id=seat_id, facts=facts)
    if facts.get("status") != "pending" and ask.get("record_id"):
        reply_ref = settle_ledger_seat(state_root, task_id, str(ask["record_id"]), seat_id, facts, raw_text=str(row.get("text") or ""))
        if reply_ref:
            record = plan_record.update_participant(state_root, task_id, ask_id=ask_id, seat_id=seat_id, facts={"reply_ref": reply_ref})
            if participant.get("status") == "pending":  # it settles now
                _keep_late_reply_in_view(state_root, task_id, ask, participant, facts, reply_ref)
    if ctx is not None:
        _announce_reply(ctx, participant, facts)
        _reference(ctx, task_id, record, state_root)
    return record


def _keep_late_reply_in_view(state_root: pathlib.Path, task_id: str, ask: Mapping[str, Any], participant: Mapping[str, Any],
                             facts: Mapping[str, Any], reply_ref: Mapping[str, Any]) -> None:
    """A reply that settles after its task ended stays in view in the task's room until the integrating mind
    releases it (#1738, ``late_review_marks``): advice for whoever continues the work, never a verdict, a hold
    or a wake. The mark points at the ledger's retained reply and copies nothing else. A reply that lands
    while its author runs stays with the gate and ``plan_task()``."""
    from ouroboros import late_review_marks

    source, ended = reply_ref.get("ref"), late_review_marks.ended_task_row(state_root, task_id)
    if not isinstance(source, Mapping) or ended is None:
        return
    who, seat = str(facts.get("model") or participant.get("model") or "a participant"), str(participant.get("seat_id") or "")
    reply = facts.get("reply") or {}
    lines = [f"Plan council reply from {who} (seat {seat}) to ask {ask.get('ask_id')} on "
             f"{plan_record.ask_subject(ask.get('revision'), plan=True)}, "
             "recorded after its task ended: advice for whoever continues this work, not a verdict; it holds nothing."]
    if facts.get("status") != "answered":
        lines.append("Its reply could not be read as the reply object; the full text is in its retained source.")
    lines += [f"Prose: {' '.join(str(reply['prose']).split())}"] if str(reply.get("prose") or "").strip() else []
    lines += [f"- objection {o.get('id')}: {' '.join(str(o.get('text') or '').split())}" for o in reply.get("objections") or []]
    late_review_marks.publish_late_review(state_root, {
        "surface": "plan_review", "task_id": task_id, "publication_id": f"plan-late:{ask.get('ask_id')}:{seat}",
        "chat_id": ended.get("chat_id"), "lines": lines,
        "settled_at": str(facts.get("answered_at") or ""), "source_ref": {**source, "task_id": str(reply_ref.get("task_id") or task_id)},
        "subject": {"ask_id": str(ask.get("ask_id") or ""), "seat_id": seat, "plan_revision": ask.get("revision"),
                    "record_id": str(ask.get("record_id") or "")}})


def settle_released_reply(usage_ctx: Any, *, request: Any, task_id: str, actor: Any, settled_wave: Dict[str, str],
                          roster_size: int = 0, slot: Any = None, settled_actors: Optional[Mapping[str, Any]] = None) -> None:
    """The custody hook for a plan seat released at the dispatch barrier: settle it into
    the record, and when its counted participants finish wake the task mailbox once. The hook that
    composes the frame first absorbs, from custody's own copies (``settled_actors``), every
    seat whose record write has not landed — a sibling's hook still running, or a write lost
    to a store failure — at $0, never a send; what it still cannot record the frame names as
    settled in custody, never as still owed. A request without a plan ask id is an
    earlier-schema (v2) wave: its result stays in the operation's custody as history."""
    if str(getattr(request, "surface", "") or "") != "plan_review":
        return
    identity = dict(getattr(request, "reconciliation_identity", {}) or {})
    ask_id = str(identity.get("plan_ask_id") or "")
    if not ask_id:
        return
    if usage_ctx is None or not getattr(usage_ctx, "drive_root", None):
        return
    seat_id = str(getattr(actor, "slot_id", "") or "")
    try:
        state_root, _ = state_location(usage_ctx)
    except Exception:
        log.warning("plan ask %s: no state root to settle %s into", ask_id, seat_id, exc_info=True)
        return
    record = None
    try:
        record = record_reply(usage_ctx, state_root=state_root, task_id=task_id, ask_id=ask_id, seat_id=seat_id,
                              row=_actor_row(actor, slot))
    except Exception as exc:  # the reply is safe in custody; the owner hears it, nothing is lost silently
        log.warning("plan ask %s: reply of %s could not be settled", ask_id, seat_id, exc_info=True)
        _progress(usage_ctx, f"📐 Plan: {seat_id}'s reply reached custody, but the plan record could not take it "
                             f"({type(exc).__name__}); plan_task() collects it at $0.")
    if slot is not None and record is not None:
        try:  # D22: the seat's «runs as» row, recorded when it settles (the barrier skipped it as pending)
            from ouroboros.reviewer_slot_config import record_reviewer_slot_executions

            record_reviewer_slot_executions("plan_review", [actor], {seat_id: slot}, keep_on=usage_ctx)
        except Exception:
            log.debug("plan ask %s: last-execution row for %s not recorded", ask_id, seat_id, exc_info=True)
    participants: List[Mapping[str, Any]] = []
    objections: List[Mapping[str, Any]] = []
    completed = None
    try:
        record = plan_record.load_plan_record(state_root, task_id)
        ask, _ = _ask_and_participant(record, ask_id, "")
        for participant in list((ask or {}).get("participants") or []):
            sid = str(participant.get("seat_id") or "")
            held = (settled_actors or {}).get(sid)
            if participant.get("status") != "pending" or participant.get("unresolved") or not isinstance(held, Mapping):
                continue
            try:  # $0: custody's copy of a seat the record has not taken yet (its own hook announces it when it lands)
                settled = record_reply(None, state_root=state_root, task_id=task_id, ask_id=ask_id, seat_id=sid,
                                       row=_actor_row(held.get("actor"), held.get("slot")))
                if settled is not None:
                    record = settled
            except Exception:
                log.warning("plan ask %s: custody's copy of %s could not be recorded", ask_id, sid, exc_info=True)
        completed = plan_record.claim_ask_completion(state_root, task_id, ask_id, settled_wave)
        if completed is None:
            return
        ask, _ = _ask_and_participant(completed, ask_id, "")
        participants = (ask or {}).get("participants") or []
        objections = plan_record.open_objections(completed)
    except Exception:
        log.warning("plan ask %s: the record could not be read for the settled frame", ask_id, exc_info=True)
        if not settled_wave or not all(settled_wave.values()):
            return  # without the recorded standing, only physical completion is known
    from ouroboros.owner_mailbox import write_task_message

    delivered = False
    try:
        delivered = write_task_message(pathlib.Path(str(usage_ctx.drive_root)), plan_packet.settled_frame(ask_id, participants, objections, custody=settled_wave),
                                       task_id, source_task_id=task_id, provenance="system", msg_id=f"plan-complete:{task_id}:{ask_id}")
    except Exception:
        log.warning("plan ask %s settled-wave frame failed for %s", ask_id, task_id, exc_info=True)
    finally:
        if not delivered and completed is not None:
            try:
                plan_record.release_ask_completion(state_root, task_id, ask_id)
            except Exception:
                log.warning("plan ask %s: undelivered completion claim could not be released", ask_id, exc_info=True)


def collect_pending(ctx: Any, record: Mapping[str, Any], *, state_root: pathlib.Path, task_id: str) -> Dict[str, Any]:
    """Settle, at $0, every pending seat whose answer already exists — through the shared
    recorded-actor collector (``review_operation.collect_recorded_review_actor``, the skill
    review's precedent): the live join with the worker that still owns the seat, the settled
    cache, the retained producer outcome, an attach-only read of a delegated run whose worker
    is gone, and the controller check (a live controller in another process is deferred to).
    Never a new send. An outcome still in flight or deferred stays pending; a seat whose
    custody is lost stays pending and unresolved — never cleaned, never re-sent (03, D6)."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_operation import collect_recorded_review_actor

    current = dict(record)
    for ask, participant in list(plan_record.pending_participants(record)):
        operation = str(participant.get("operation_id") or "")
        if not operation:
            continue
        seat = str(participant.get("seat_id") or "")
        slot = SimpleNamespace(slot_id=seat, model=str(participant.get("model") or ""),
                               route=ReviewRouteKind(str(participant.get("route") or "api_chat")))
        row = {**participant, "slot_id": seat,
               "usage": {key: participant[key] for key in ("review_controller", "pending_invocation_id", "delegated_run_id")
                         if participant.get(key)}}
        actor = collect_recorded_review_actor(state_root, task_id=task_id, surface="plan_review", row=row,
                                              retry_key=str(ask.get("retry_key") or ""), usage_ctx=ctx)
        state = str(actor.get("operation_state") or "")
        if state in {"in_flight", "pending_dispatch"} or (actor.get("late_result_pending") and state != "custody_lost"):
            continue  # still running, or deferred to the worker / live controller that owns it
        settled = record_reply(ctx, state_root=state_root, task_id=task_id, ask_id=str(ask.get("ask_id") or ""),
                               seat_id=seat, row=_actor_row(actor, slot))
        if settled is not None:
            current = settled
    return current


def attach_late_reply(state_root: Any, event: Mapping[str, Any]) -> int:
    """The supervisor's late-result hook: settle the one operation the event names into
    its plan record; an earlier-schema (v2) operation stays history in its custody."""
    if str(event.get("surface") or "") != "plan_review":
        return 0
    from ouroboros.observability import read_call_payload

    task_id, operation = str(event.get("task_id") or ""), str(event.get("operation_id") or "")
    if not task_id or not operation:
        return 0
    root = pathlib.Path(str(state_root))
    try:
        _manifest, prompt, _refs = read_call_payload(root, task_id=task_id, call_id=f"{operation}_prompt")
    except (OSError, ValueError, KeyError, TypeError):
        return 0
    identity = (prompt.get("request") or {}).get("reconciliation_identity") or {}
    if not identity.get("plan_ask_id"):
        return 0
    record = plan_record.load_plan_record(root, task_id)
    before = {(a.get("ask_id"), p.get("seat_id")): p.get("status") for a in record.get("asks") or [] for p in a.get("participants") or []}
    after = collect_pending(None, record, state_root=root, task_id=task_id)
    return sum(1 for a in after.get("asks") or [] for p in a.get("participants") or []
               if before.get((a.get("ask_id"), p.get("seat_id"))) == "pending" and p.get("status") != "pending")
