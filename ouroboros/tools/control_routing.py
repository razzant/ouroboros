"""Start separate work, place the current work, and deliver messages to live work.

Tools state an intent; the supervisor owns admission and its durable receipt.
"""

from __future__ import annotations

import logging
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ouroboros.dialogue_provenance import presence_root_carrier, presence_root_relation
from ouroboros.tools.control_events import (
    _emit_and_wait_for_routing,
    _promotion_pool_disabled_from_snapshot,
    _routing_status_root,
)
from ouroboros.tool_access_paths import canonical_data_root
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, completed_local_read, publish_no_effect
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)




def _root_effort_arg(value: Any, tool_name: str) -> tuple[Dict[str, str], str, str]:
    """``(event fields, refusal, note)`` for the new root's explicit starting effort.

    Omitted (``None``) leaves the field absent, so the root starts at the owner's
    recommended level; a supplied value must name a tier, before any effect. Outside
    Cyber Pro a root Ouroboros creates itself is not pinned by that value either: the
    field stays absent and the note says so (owner ingress — the API, the CLI, owner
    schedules, Continue — keeps pinning); in Cyber Pro it is applied.
    """
    if value is None or value == "":
        return {}, "", ""
    from ouroboros.runtime_mode_policy import effort_range_binds
    from ouroboros.settings_scales import effort_range, requested_effort
    from ouroboros.tools.arg_feedback import ignored_argument_note

    try:
        tier = requested_effort(value)
    except ValueError as exc:
        return {}, f"⚠️ TOOL_ARG_ERROR ({tool_name}): {exc}", ""
    if effort_range_binds():
        return {}, "", ignored_argument_note(
            "reasoning_effort", tier,
            f"outside Cyber Pro a root I create starts at my human's recommended level "
            f"({effort_range()['recommended']}); deeper thinking is delegated with schedule_subagent(effort=...)")
    return {"reasoning_effort": tier}, "", ""


def _attach_origin_from_metadata(ctx: ToolContext, evt: Dict[str, Any]) -> None:
    """Copy the ingress-captured owner-message origin (ref + full text) onto a
    promote-shaped event BY VALUE. The host built the ref at chat admission;
    producers never re-derive identity from content (DEVELOPMENT.md
    anti-pattern: content-derived identity for host-minted records)."""
    metadata = getattr(ctx, "task_metadata", None)
    if not isinstance(metadata, dict):
        return
    ref = metadata.get("origin_message_ref")
    if isinstance(ref, dict) and ref:
        evt["source_ref"] = dict(ref)
        snapshot = metadata.get("origin_message_snapshot") or {}
        if snapshot:
            evt["origin_message_snapshot"] = snapshot
        text = snapshot.get("placeholder_text") or metadata.get("origin_message_text")
        if isinstance(text, str) and text:
            evt["source_text"] = text
    elif metadata.get("origin_suppressed"):
        evt["origin_suppressed"] = True


def _attach_drafted_objective(ctx: ToolContext, evt: Dict[str, Any], source: Optional[Dict[str, Any]] = None,
                              *, owner_basis: Optional[Dict[str, Any]] = None) -> None:
    """Keep the routed objective's author separate from the ingress owner corpus.

    A routing run with neither owner-door stamp opens the new root with this drafted
    objective and no owner message. Only this producer knows that, so it records
    ``origin_message_absent``: a later reader cannot tell an absent stamp from a lost
    one, and a suppressed stamp is owner input that the door did not log.

    The corpus is persisted with the start (its receipt, the queue snapshot, the running
    record): the one owner-corpus projection (``owner_words.carried_owner_corpus``);
    ``source``: the held row ``owner_source`` selected to open the start.
    """
    from ouroboros.owner_words import carried_owner_corpus, merged_corpus

    evt["objective_author"] = {"kind": "task", "task_id": str(getattr(ctx, "task_id", "") or "")}
    if not (evt.get("source_ref") or evt.get("origin_suppressed")):
        evt["origin_message_absent"] = True
    words = "" if (evt.get("origin_message_snapshot") or {}).get("placeholder_text") else evt.get("source_text")
    evt["owner_corpus"] = carried_owner_corpus(ctx, words, source=source)
    if owner_basis:  # the cited words ride into the new root alone, never back into this run's own corpus
        if owner_basis["source"] in ("owner_press", "owner_quiz_answer") or (owner_basis.get("reply_to") or {}).get("card_id"):
            evt["owner_corpus"] = merged_corpus(evt.get("owner_corpus"), [owner_basis])
        else:
            evt["owner_corpus"] = [*evt.get("owner_corpus", []), owner_basis]


def _selected_source(ctx: ToolContext, handle: str, presence_carrier: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """``(row, refusal)``: the held owner row ``owner_source`` names to open the new work, read before any effect.

    Only a root the owner door stamped (``run_origin``'s ``owner_ingress``) and that acts for no place chooses; the
    row must be one this run holds and still quotable now (``owner_relay.quote_rows``: a revoked account's or a
    forwarded row is not). A quote rebuilt from the corpus keeps its words without becoming a source row.
    Choosing names which of the owner's messages the work answers; it grants nothing new.
    """
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.owner_relay import quote_rows
    from ouroboros.task_results import resolve_ctx_lineage
    from ouroboros.tools.join_ledger import _is_delegated_task

    metadata = getattr(ctx, "task_metadata", None)
    if presence_carrier or _is_delegated_task(ctx) or not resolve_ctx_lineage(ctx)["is_root_task"] or not run_origin(
            {"metadata": metadata if isinstance(metadata, dict) else {}})["owner_ingress"]:
        return {}, ("owner_source belongs to a root your human's message started; this work keeps the source it "
                    "inherits (a place's work names its destination, a private start its owner_quote).")
    quotes, error = quote_rows(ctx, [handle])
    if error:
        return {}, f"owner_source names no held owner message now ({error})."
    from ouroboros.dialogue_store import ROW_SHA_KEY
    from ouroboros.owner_relay import held_owner_messages, owner_row_source
    from ouroboros.project_dialogue import owner_message_ref_is_valid, resolve_owner_message_source
    from ouroboros.store import StoreError

    row = quotes[0]["owner_row"]
    if row.get("_owner_corpus_rebuilt") or row.get("corpus_answer"):
        return {}, "owner_source is a corpus quote. Choose the original message to open new work from it."
    ref, _words = owner_row_source(row)
    if not owner_message_ref_is_valid(ref):
        return {}, "owner_source has no complete host row identity."
    if isinstance(held_owner_messages(ctx)[handle].get("carried"), dict) and not row.get(ROW_SHA_KEY):
        # Older automatic carries did not distinguish a reconstructed quote. Require the exact
        # existing row, not a weaker search by the quote's display fields.
        try:
            original = resolve_owner_message_source(row.get("_owner_source_root") or canonical_data_root(ctx), ref)
        except StoreError:
            original = None
        if original is None:
            return {}, "owner_source is a carried quote whose original message could not be confirmed."
    return row, ""


def _selected_source_fields(ctx: ToolContext, row: Dict[str, Any]) -> Dict[str, Any]:
    """The start's source fields from the selected row, by value: its identity and the words its checksum covers
    (``owner_relay.owner_row_source``), its client id, its files from the root it was read from, its sending
    surface, each or its absence. The caller's chat, project and workspace stay the call's addressing. The files
    are a held row's (``inherited``): staging names one it cannot copy and refuses nothing, unlike a turn's own."""
    from ouroboros.owner_relay import origin_snapshot, owner_row_source, upload_specs
    from ouroboros.tool_access import canonical_data_root

    ref, words = owner_row_source(row)
    surface = row.get("client_surface")
    snapshot = origin_snapshot(row, canonical_data_root(ctx), include_reply=True)
    return {"source_ref": ref, "source_text": words, "origin_message_snapshot": snapshot,
            "client_message_id": ref["client_message_id"],
            "attachment_uploads": [{**spec, "inherited": True} for spec in upload_specs(
                row, row.get("_owner_source_root") or canonical_data_root(ctx))],
            **({"client_surface": dict(surface)} if isinstance(surface, dict) and surface else {})}


def _is_turn_message(ctx: ToolContext, own: Any, ref: Dict[str, Any]) -> bool:
    """Whether the selected row (``ref``) IS the message this turn answers (``own``: its origin ref), by physical row.
    Only the row key the owner door minted at ingress decides: an older stamp without one proves nothing (the one
    held row its fields identify is not that key), and equal room, client id, time and words prove nothing either —
    two rows can share them and carry different files."""
    key = ref.get("row_sha256")
    return bool(isinstance(own, dict) and key and own.get("row_sha256") == key)


def _origin_file_refs(ctx: ToolContext, ref: Dict[str, Any]) -> list | None:
    """The file refs of the message this run answers, by value; ``None`` where they cannot be known.

    A snapshot the owner door recorded answers for its row (one without ``attachments`` records no files). Where none
    was recorded (an older stamp; no writer keeps an empty one) the list is recovered from the one row the origin ref
    identifies: the row this run holds, else the canonical row that ref picks out uniquely and quotably, never one
    matched by equal words. With neither, the files are a gap, not an empty list."""
    metadata = getattr(ctx, "task_metadata", None)
    snapshot = metadata.get("origin_message_snapshot") if isinstance(metadata, dict) else None
    if isinstance(snapshot, dict) and snapshot:
        return snapshot.get("attachments") or []
    from ouroboros.owner_relay import origin_snapshot, quotable_owner_row
    from ouroboros.owner_words import _held_origin
    from ouroboros.project_dialogue import resolve_owner_message_source

    try:
        held = getattr(ctx, "_held_owner_messages", None)  # built every round: no second read of what it holds
        row = _held_origin(ref, held) if isinstance(held, dict) and ref.get("row_sha256") else None
        if row is None:
            found = resolve_owner_message_source(canonical_data_root(ctx), ref)
            row = found if found and quotable_owner_row(found, canonical_data_root(ctx)) else None
        return None if row is None else origin_snapshot(row, canonical_data_root(ctx)).get("attachments") or []
    except Exception:
        log.debug("start_task: the files of the origin row could not be resolved", exc_info=True)
        return None


def _source_files(ctx: ToolContext, evt: Dict[str, Any], *, selected: bool) -> tuple[list, bool]:
    """Settle the start's files before it is sent: ``(this turn's own files the choice displaced, whether its source's
    file list is unknown)``. ``evt["attachment_uploads"]`` is what the start asks the host to stage.

    Without a selection, separate work answers the message this run answers and receives that message's files: a
    direct turn's own fresh uploads (``chat_attachment_uploads``, already on the event), else the files its origin
    row records (``_origin_file_refs``), staged without refusal. A selection replaces the turn's own uploads with
    the selected row's own complete ordered specs (labels, duplicates and availability included), which are named
    as not given; selecting the turn's own message keeps its fresh uploads only when the ingress origin ref and
    the selection carry the same physical row key (``_is_turn_message``). Admission, never this settling, says
    what was staged."""
    from ouroboros.owner_relay import source_upload_specs

    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    fresh = [spec for spec in metadata.get("chat_attachment_uploads") or [] if isinstance(spec, dict)]
    unknown = False
    if not selected and not fresh and evt.get("source_ref"):
        refs = _origin_file_refs(ctx, evt["source_ref"])
        unknown = refs is None
        evt["attachment_uploads"] = source_upload_specs(ctx, refs or [])
    if selected and fresh and _is_turn_message(ctx, metadata.get("origin_message_ref"), evt["source_ref"]):
        evt["attachment_uploads"] = fresh
        return [], unknown
    return ([str(spec.get("label") or "attachment") for spec in fresh] if selected else []), unknown


def _files_note(ctx: ToolContext, evt: Dict[str, Any], receipt: Dict[str, Any], files: tuple | None) -> str:
    """The sentences of the start's result about its files (``files``: what ``_source_files`` returned).

    Each file of its source reads as the admission recorded it: the token-matched receipt of a scheduled start
    (``routing_wait.admitted_files``). With no admission yet nothing is known of staging: a file with a path is
    pending and one without names its known reason. An unknown file list and the turn's own files a selection
    displaced each get a sentence of their own; labels are never rewritten."""
    if files is None:
        return ""
    from ouroboros.owner_relay import NOT_GIVEN

    displaced, unknown = files
    specs = [spec for spec in evt.get("attachment_uploads") or [] if isinstance(spec, dict)]
    more = 0
    if specs and str(receipt.get("status") or "") == "scheduled":
        from ouroboros.routing_wait import complete_admitted_files

        rows = complete_admitted_files(_routing_status_root(ctx), str(evt.get("task_id") or ""), receipt)
        declared = receipt.get("attachment_declared")  # reporting runs after the effect: it may not raise
        more = max(0, declared - len(rows)) if type(declared) is int else 0
        named = [f"{row.get('label') or 'attachment'} (" + ("staged at start" if row.get("status") == "staged" else
                 f"not staged: {row.get('reason') or 'staging_failed'}") + ")" for row in rows] or [
                 f"{spec.get('label') or 'attachment'} (no admission outcome recorded)" for spec in specs]
    else:
        named = [f"{spec.get('label') or 'attachment'} (" + ("pending with its admission" if spec.get("path") else
                 f"not staged: {spec.get('unavailable') or 'invalid_path'}") + ")" for spec in specs]
    return ((f" Its source's files: {'; '.join(named)}" + (f"; {more} more are in the new work's own input manifest"
                                                           if more else "") + "." if named else "")
            + (" Its source's file list is unknown (no snapshot of it was recorded and its row cannot be read): no files "
               "were given." if unknown else "")
            + (f" This turn's own files are {NOT_GIVEN} ({'; '.join(displaced)}): its source is the selected message."
               if displaced else ""))


def _source_note(ctx: ToolContext, evt: Dict[str, Any], selected: str) -> str:
    """One sentence for the start's result: which owner message the new work answers (its ``source_ref``), by the
    handle this run holds for it, else its room and client id, and a short snippet."""
    from ouroboros.answer_facts import OWNER_CARD_INPUT, OWNER_PRESS
    from ouroboros.owner_relay import owner_text
    from ouroboros.owner_words import _held_origin
    from ouroboros.project_dialogue import opening_input

    ref = evt["source_ref"]
    held = getattr(ctx, "_held_owner_messages", None) or {}
    snapshot = evt.get("origin_message_snapshot") or {}
    address = selected or (snapshot.get("reply") or {}).get("address")
    if not address:
        known = _held_origin(ref, held)
        address = next((handle for handle, row in held.items() if row is known), "")
    words = "" if snapshot.get("placeholder_text") else evt.get("source_text")
    row = held.get(address) or _held_origin(ref, held)
    item = opening_input(row, canonical_data_root(ctx), retained=snapshot.get("input"))
    pressed = item.get("source") == OWNER_PRESS
    if item:
        words = "" if item.get("source") == OWNER_CARD_INPUT else item.get("content")
    elif row:
        words = owner_text(row)
    words = " ".join(str(words or "").split())
    shown = (f"«{words if len(words) <= 80 else words[:79].rstrip() + '…'}»" if words else
             "(no words of theirs)" if snapshot.get("placeholder_text") else "(owner words unavailable)")
    where = address or (
        f"chat {ref.get('chat_id')} / {ref.get('client_message_id') or ref.get('ts')}")
    what = "the owner's card press" if pressed else "the owner message"
    return f" It answers {what} {where} ({'selected' if selected else 'inherited'}): {shown}."


def _inherited_project_scope(ctx: ToolContext) -> str:
    """Only the caller's own room supplies a default for separate new work."""
    from ouroboros.projects_registry import project_id_for_task

    return str(project_id_for_task(canonical_data_root(ctx), str(getattr(ctx, "task_id", "") or ""), strict=True)
               or getattr(ctx, "project_id", "") or "")


def _attach_client_surface(ctx: ToolContext, evt: Dict[str, Any]) -> None:
    """Copy the routing turn's per-message client-surface fact onto a
    promote/route/steer event BY VALUE (the origin_message_ref rail's sibling:
    the fact was captured at ingress; producers never re-derive it)."""
    metadata = getattr(ctx, "task_metadata", None)
    if not isinstance(metadata, dict):
        return
    fact = metadata.get("client_surface")
    if isinstance(fact, dict) and fact:
        evt["client_surface"] = dict(fact)


ISSUER_OWNER_TURN = "owner_turn"
ISSUER_TASK = "task"


def _routing_issuer(ctx: ToolContext) -> Dict[str, Any]:
    """WHO speaks through this routing act -- minted by value where the host knows.

    An OWNER TURN is the direct turn the owner door stamped: ``is_direct_chat`` AND
    ``run_origin``'s ``owner_ingress`` (``origin_message_ref`` / ``origin_suppressed``,
    written only by owner routing). Every other context speaks as a TASK, including
    a pooled root relaying an owner message it just drained. ``last_owner_delivery``
    keys the receipt, never the issuer. The 14.09 incident decided this five times
    from proxies (a routing contract a Swarm root never has, an empty client id read
    as "agent-issued", a room veto keyed on the chat): the host now states it once,
    and the model has no argument to claim otherwise.

    Neither the lane nor a client id makes an owner turn: a consciousness wake-up,
    a Presence event (which carries the provider's event id as its client id) and
    the auto-resume template all run on the direct lane, and nobody typed them; a
    promoted root inherits the owner's stamp as ancestry but is not a direct turn.
    All of them speak as a task.
    """
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    from ouroboros.dialogue_provenance import run_origin

    if bool(getattr(ctx, "is_direct_chat", False)) and run_origin({"metadata": metadata})["owner_ingress"]:
        return {"kind": ISSUER_OWNER_TURN}
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    return {
        "kind": ISSUER_TASK,
        "task_id": task_id,
        "root_task_id": str(metadata.get("root_task_id") or task_id),
    }


def _finish_swarm_handoff(
    ctx: ToolContext,
    evt: Dict[str, Any],
    response: str,
    *,
    status: str,
    reason: str = "",
) -> str:
    """Preserve the first Presence admission receipt for its terminal consumers.

    The owner's private start takes no part: it is no handoff of this conversation, so
    whatever its outcome, the slot stays free for an outward start or with the one that holds it.
    """
    from ouroboros.presence_owner_contact import DESTINATION_OWNER

    metadata = getattr(ctx, "task_metadata", {})
    presence_turn = isinstance(metadata, dict) and bool(metadata.get("presence"))
    if presence_turn and evt.get("destination") != DESTINATION_OWNER and not isinstance(
        getattr(ctx, "_swarm_handoff_attempt", None), dict
    ):
        ctx._swarm_handoff_attempt = {
            "task_id": str(evt.get("task_id") or ""),
            "routing_token": str(evt.get("routing_token") or ""),
            "status": status,
            "reason": reason,
            "response": response,
        }
    return response


def _effective_scope_note(ctx: ToolContext, project_id: str) -> str:
    """Report the separate work's recorded destination, not its requested room."""
    pid = str(project_id or "").strip()
    if not pid:
        return ""
    name = ""
    try:
        from ouroboros.projects_registry import get_project

        name = str((get_project(canonical_data_root(ctx), pid) or {}).get("name") or "").strip()
    except Exception:
        log.debug("promote: effective project name lookup failed", exc_info=True)
    return f" in project '{name}' ({pid})" if name and name != pid else f" in project '{pid}'"


def _requested_scope_label(display_name: str, project_id: str) -> str:
    """What the caller ASKED for, named only where the outcome is unknown."""
    if display_name:
        return f"new project '{display_name}'"
    return f"project '{project_id}'" if project_id else "the main chat"


def _task_card_title_arg(title: str, verb: str) -> tuple[str, str]:
    """Model-created roots need a name that survives admission's lexical cleaner."""
    from ouroboros.project_naming import admission_names

    card_title = str(title or "").strip()[:80]
    if admission_names({"title": card_title}, "")[0]:
        return card_title, ""
    return "", (
        f"⚠️ TOOL_ARG_ERROR ({verb}): title is required: the short name "
        "the owner sees on the new task's card (<=80 chars). Supply title separately "
        "from the executor's work order. Nothing was scheduled."
    )


def _start_destination(ctx: ToolContext, carrier: Dict[str, Any], destination: Any,
                       owner_quote: Any) -> tuple[Dict[str, Any], str]:
    """``(private start's event fields, refusal)`` of a start's named destination, read before any effect.

    A caller holding Presence authority (``carrier``) names where each new work belongs; omitted, unknown
    or contradictory choices refuse and the mind chooses again. Work without one has nothing to choose.
    """
    from ouroboros.presence_owner_contact import (
        DESTINATION_OWNER,
        DESTINATION_PLACE,
        ContactRequestInvalid,
        private_start_fields,
    )

    refused = "⚠️ TOOL_ARG_ERROR (start_task): {} Nothing was scheduled."
    if not carrier:
        return {}, (refused.format("destination and owner_quote apply only to a start from a Presence "
                                   "conversation or work acting for a place.")
                    if destination not in (None, "") or owner_quote not in (None, "") else "")
    if destination not in (DESTINATION_PLACE, DESTINATION_OWNER):
        return {}, refused.format(
            f"destination must be {DESTINATION_PLACE!r} (this place's work) or {DESTINATION_OWNER!r} (the owner's "
            "private work, with owner_quote naming their held answer to a private question).")
    if destination == DESTINATION_PLACE:
        return {}, refused.format("owner_quote belongs to destination='owner'.") if owner_quote not in (None, "") else ""
    try:
        return private_start_fields(ctx, owner_quote), ""
    except ContactRequestInvalid as exc:
        return {}, refused.format(str(exc))


def _start_task(
    ctx: ToolContext,
    objective: str,
    expected_output: str = "",
    project_id: str = "",
    workspace_root: str = "",
    title: str = "",
    project_name: str = "",
    workspace: str = "",
    source: str = "",
    context_requires_self_body_docs: bool = False,
    reasoning_effort: Any = None,
    model: str = "",
    model_as_typed: bool = False,
    context_requires_development: Optional[bool] = None,
    destination: str = "",
    owner_quote: str = "",
    owner_words: str = "",
    review_answer: bool = False,
    owner_source: str = "",
) -> str:
    """Start separate new work; the caller keeps its identity and obligations."""
    from ouroboros.owner_words import cited_owner_basis

    effort_fields, effort_error, effort_note = _root_effort_arg(reasoning_effort, "start_task")
    if effort_error:
        return effort_error
    goal = str(objective or "").strip()
    if not goal:
        return "⚠️ TOOL_ARG_ERROR (start_task): objective is required"
    model_basis, cite_error = cited_owner_basis(ctx, owner_words, call_speech=bool(model))
    owner_basis = model_basis if model_basis and model_basis["source"] != "live_call" else None
    if cite_error and model:
        cite_error = "The model needs verified owner_words. Cite your human's exact words or the complete label they pressed."
    if cite_error:  # the same typed argument refusal the schedule doors publish: nothing was started
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                                                 text=f"⚠️ OWNER_WORDS_UNVERIFIED (start_task): {cite_error}"))
    if owner_basis and destination == "owner":  # a private owner start names its held answer (#2795), not a place cite
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "⚠️ TOOL_ARG_ERROR (start_task): owner_words cites your human's words for this place's work "
            "(destination='place'); a private owner start names their held answer in owner_quote. Nothing was scheduled.")))
    card_title, title_error = _task_card_title_arg(title, "start_task")
    if title_error:
        return title_error
    metadata = getattr(ctx, "task_metadata", {})
    presence_carrier = presence_root_carrier(metadata, task_contract=getattr(ctx, "task_contract", None))
    private, destination_error = _start_destination(ctx, presence_carrier, destination, owner_quote)
    if destination_error:
        return destination_error
    chosen: Dict[str, Any] = {"destination": destination} if presence_carrier else {}
    if private:
        chosen["owner_quote"] = owner_quote
    selected = owner_source.strip() if isinstance(owner_source, str) else owner_source
    source_row: Dict[str, Any] = {}
    if selected not in (None, ""):
        source_row, source_error = (({}, "owner_source must be one held owner_quote_handle string.")
                                    if not isinstance(selected, str) else _selected_source(ctx, selected, presence_carrier))
        if source_error:
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
                f"⚠️ TOOL_ARG_ERROR (start_task): {source_error} Nothing was scheduled.")))
        chosen["owner_source"] = selected  # the selector rides the durable intent: an explicit retry names it again
    from ouroboros.project_facts import explicit_project_id_ok, project_id_from_display_name, sanitize_project_id

    display_name = str(project_name or "").strip()
    pid = ""
    if str(project_id or "").strip():
        if not explicit_project_id_ok(project_id):
            return (
                f"⚠️ TOOL_ARG_ERROR (start_task): project_id {project_id!r} is not "
                "filesystem-clean; use lowercase alphanumeric/_/-/. (<=64 chars)"
            )
        pid = sanitize_project_id(project_id)
    elif display_name:
        # LLM-first "create a NAMED project and work there": derive a filesystem
        # id from the display name. A non-ASCII name (e.g. a Cyrillic "динозавры")
        # falls back to a deterministic hash id so the project is still created —
        # the human-readable name rides project_name on the registry.
        pid = project_id_from_display_name(display_name)
    else:
        # The caller's own room supplies the default, never a sibling's origin.
        pid = sanitize_project_id(_inherited_project_scope(ctx))
    if private and not str(project_id or "").strip() and not display_name:
        # A private owner start must not silently inherit the Presence project's room.
        # Without an explicit owner project, ordinary owner work starts in Main;
        # a selected project is still honored above.
        pid = ""
    from ouroboros.tools.control_models import start_choice, refuse

    try:
        model_metadata, model_note = start_choice(ctx, model, model_basis, pid, model_as_typed=model_as_typed)
    except ValueError as exc:
        return refuse(ctx, str(exc))
    try:
        current_chat_id = int(getattr(ctx, "current_chat_id", None) or 0)
    except (TypeError, ValueError):
        current_chat_id = 0
    requested_root = str(workspace_root or "").strip()
    workspace_sentinel = str(workspace or "").strip().lower()
    repo_root_note = ""
    intent_system_repo = False
    if requested_root:
        # Q4=A: naming the Ouroboros repository ITSELF names the documented
        # default (no separate workspace — the ordinary self-modification task),
        # so the EXACT root maps onto the existing "none" sentinel, in Main and
        # in every project room alike. A subfolder, the data drive and every
        # other path pass through unchanged; admission refuses them with the
        # repair hint.
        from ouroboros.tool_access import paths_overlap_casefold
        from ouroboros.workspace_admission import WORKSPACE_NONE

        system_repo = getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir", None)
        try:
            requested_path = Path(requested_root).expanduser().resolve(strict=False)
            system_repo_path = Path(str(system_repo)).resolve(strict=False) if system_repo else None
            same_root = (
                system_repo_path is not None
                and len(requested_path.parts) == len(system_repo_path.parts)
                and paths_overlap_casefold(requested_path, system_repo_path)
            )
        except (OSError, ValueError, RuntimeError):
            same_root = False
        if same_root:
            requested_root, workspace_sentinel = "", WORKSPACE_NONE
            intent_system_repo = True  # the CHOICE survives the sentinel (#1315)
            repo_root_note = (
                " (workspace_root named the Ouroboros repository itself; started as an "
                "ordinary task over it — no separate workspace)"
            )
    tid = uuid.uuid4().hex[:16]
    routing_token = uuid.uuid4().hex
    disabled_reason = _promotion_pool_disabled_from_snapshot(ctx)
    if disabled_reason:
        response = (
            f"⚠️ PROMOTE_REJECTED: task {tid} was not scheduled "
            f"(worker_pool_unavailable: {disabled_reason}){_cause_words('worker_pool_unavailable')}. "
            "No project/workspace admission side effects were started."
        )
        return _finish_swarm_handoff(
            ctx,
            {"task_id": tid, "routing_token": routing_token, **chosen},
            response,
            status="rejected",
            reason=f"worker_pool_unavailable:{disabled_reason}",
        )
    evt: Dict[str, Any] = {
        "type": "promote_chat_to_task",
        "task_id": tid,
        "routing_token": routing_token,
        "objective": goal,
        **({"metadata": model_metadata} if model_metadata else {}),
        "expected_output": str(expected_output or "").strip(),
        "project_id": pid,
        "project_name": display_name,
        "title": card_title,
        "workspace_root": requested_root,
        # Source admission is intentionally supervisor-side, after the
        # authoritative worker-pool and duplicate-id gates.
        "source": str(source or "").strip(),
        # v6.58.0: "none" opts a project-room task OUT of the room's working_dir
        # default (a folder-less task in a folder-ful project stays possible).
        "workspace": workspace_sentinel,
        "context_requires_self_body_docs": context_requires_self_body_docs,
        # Absent is not false: only an explicit answer overrides the subject-of-work rule (BIBLE P1).
        **({"context_requires_development": bool(context_requires_development)}
           if isinstance(context_requires_development, bool) else {}),
        **({"review_answer": True} if review_answer is True else {}),  # task_contract.review_answer
        **effort_fields,
        "chat_id": current_chat_id,
        **({"resource_intent": {"kind": "system_repo"}} if intent_system_repo else {}),
        "client_message_id": str(
            ((getattr(ctx, "task_metadata", {}) or {}).get("client_message_id") or "")
            if isinstance(getattr(ctx, "task_metadata", {}), dict) else ""
        ),
        "attachment_uploads": list(
            ((getattr(ctx, "task_metadata", {}) or {}).get("chat_attachment_uploads") or [])
            if isinstance(getattr(ctx, "task_metadata", {}), dict) else []
        ),
        "ts": utc_now_iso(),
        **chosen,
    }
    if private:
        # The owner's work this answer asks for: the same ordinary start and relation as a private
        # late answer's (``presence_owner_contact.private_start_fields``), nothing of this caller's.
        evt.update(private)
    elif presence_carrier:
        contract = dict(getattr(ctx, "task_contract", {}) or {})
        evt.update({**presence_carrier, "task_contract": contract})
        if "capability_ceiling" in contract:
            # Only retained legacy work has an immutable scope ceiling. A new
            # focus keeps ordinary workspace admission and its return carrier.
            evt.update({key: "" for key in ("project_id", "project_name", "workspace_root", "workspace", "source")})
            repo_root_note = ""
        if "presence" in presence_carrier:
            from ouroboros.presence_context import PRESENCE_FINAL_ROUTE_KEY, handoff_fact

            # ``_finish_swarm_handoff`` holds the first attempt whatever its status; start_task runs
            # serially. A candidate written before submission: only what the source publishes decides.
            evt[PRESENCE_FINAL_ROUTE_KEY] = handoff_fact(getattr(ctx, "task_id", ""), tid, held=not isinstance(
                getattr(ctx, "_swarm_handoff_attempt", None), dict))
    else:
        # The owner's own work related to a place (#2482): an ordinary owner start
        # with its own contract; only the relation the place reads it by rides along.
        evt.update(presence_root_relation(metadata))
    # A promote from a consciousness turn/tree mints a consciousness root: the
    # origin label, ledger category and level ride the event by value; the
    # supervisor stamps them on the new root (worker_promotion) — no presence-style
    # stripping, the wake chooses project/workspace like any Main turn (В9').
    from ouroboros.consciousness_authority import consciousness_origin_metadata, is_focus_wake

    if not private:  # a private owner start carries its own owner contract (#2795); a place start, its writer's kind
        evt.update(consciousness_origin_metadata({**metadata, "task_id": str(getattr(ctx, "task_id", "") or "")},
                                                 started_root=True, owner_basis=owner_basis))
        if source_row:  # the selected row opens the start in place of the run's own: its identity, files and surface
            evt.update(_selected_source_fields(ctx, source_row))
        else:
            _attach_origin_from_metadata(ctx, evt)
            _attach_client_surface(ctx, evt)
        if (pid and not presence_carrier and not is_focus_wake(metadata) and getattr(ctx, "is_direct_chat", False)
                and (not source_row or _is_turn_message(ctx, metadata.get("origin_message_ref"), evt.get("source_ref") or {}))):
            from ouroboros.chat_references import TURN_ADDRESS, reply_transport_from_metadata

            if transported := reply_transport_from_metadata(canonical_data_root(ctx), metadata):
                evt[TURN_ADDRESS] = transported
    # An ordinary owner-word start settles its source's files; a private start and a place's work keep their own doors.
    files = None if private or presence_carrier else _source_files(ctx, evt, selected=bool(source_row))
    _attach_drafted_objective(ctx, evt, source_row or None, owner_basis=owner_basis)
    kind_note = " " + model_note if model_note else ""
    source_note = (_source_note(ctx, evt, selected if source_row else "") if isinstance(evt.get("source_ref"), dict)
                   and evt["source_ref"] and not private and not presence_carrier else "")
    kind_note += source_note
    if is_focus_wake(metadata) and not private:  # a place turn's start: whose work it is, in one plain sentence
        kind_note += (" Started as your human's work on their pressed choice." if owner_basis and owner_basis["source"] == "owner_press"
                      else " Started as your human's work on their cited words." if owner_basis else
                     " Started as your own work (consciousness kind): it draws on the consciousness allowance, and "
                     "the Off switch stops it.")
    mode, confirmation = _emit_and_wait_for_routing(ctx, evt)
    confirmation_status = str(confirmation.get("status") or "unconfirmed")
    reason = str(confirmation.get("reason") or "")
    detail = str(confirmation.get("detail") or "")
    disabled_reason = str(confirmation.get("worker_pool_disabled_reason") or "")
    if confirmation_status == "scheduled":
        source_confirmation = f" [{detail}]" if detail else ""
        effective_pid = str(confirmation.get("effective_project_id") or "")
        scope_note = _effective_scope_note(ctx, effective_pid)
        response = (
            f"OK: task {tid}{scope_note} accepted and durably scheduled ({mode}){repo_root_note}."
            f"{source_confirmation}{kind_note}{_files_note(ctx, evt, confirmation, files)} "
            "The task now runs independently, and follow-up chat can steer it. "
            "Use wait_task/get_task_result if its result "
            "is needed in this conversation."
            + (" It is the owner's private work: nothing it does reaches this conversation by itself, "
               "and it is not this turn's deferred work." if private else "")
            + (f"\n{effort_note}" if effort_note else "")
        )
        return _finish_swarm_handoff(ctx, evt, response, status="scheduled")
    if confirmation_status in {"rejected", "needs_manual_target", "not_started"}:
        shown_reason = (
            f"{reason}: {disabled_reason}" if disabled_reason else reason
        )
        if detail:
            shown_reason = f"{shown_reason}: {detail}" if shown_reason else detail
        response = (
            f"⚠️ PROMOTE_REJECTED: task {tid} was not scheduled"
            f"{f' ({shown_reason})' if shown_reason else ''}{_cause_words(reason)}. "
            "Do not report this task as created."
        )
        return _finish_swarm_handoff(ctx, evt, response, status="rejected", reason=shown_reason or "admission_rejected")
    root = Path(str(getattr(ctx, "budget_drive_root", "") or ctx.drive_root))
    try:
        append_jsonl(
            root / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "promote_chat_to_task_unconfirmed",
                "task_id": tid,
                "transport_mode": mode,
                "reason": reason or "confirmation_timeout",
                "routing_token": routing_token,
            },
        )
    except Exception:
        log.debug("Failed to record unconfirmed promote", exc_info=True)
    # A live wait that timed out says what it measured and the body as recorded, each with its own clock;
    # a deferred emission was never waited for (its admission comes after this turn).
    waited, observed = confirmation.get("waited_sec"), ""
    if mode == "live" and isinstance(waited, (int, float)):
        from ouroboros.context_runtime_facts import admission_observation

        observed = (f" No confirmation came in {float(waited):.1f} s of waiting. "
                    + admission_observation(root, observed_at=utc_now_iso()))
    response = (
        f"Not confirmed yet: separate task {tid} is waiting to start. Its complete intent is durable; "
        f"admission to {_requested_scope_label(display_name, pid)} is pending ({mode}).{observed}{' ' + model_note if model_note else ''}{source_note}{_files_note(ctx, evt, confirmation, files)} "
        "A [ROUTING_RECEIPTS] row reports its recorded outcome. "
        f"Use get_task_result({tid}) to reconcile it before an explicit retry."
    )
    return _finish_swarm_handoff(
        ctx, evt, response, status="unconfirmed", reason=reason or "confirmation_timeout",
    )


@completed_local_read
def _list_projects(ctx: ToolContext, limit: int = 50, closed: Optional[bool] = None,
                   oldest_first: bool = False, start: int = 0) -> str:
    """Enumerate the owner's projects (id, name, recency) so the one mind can
    decide whether a main-chat message belongs to an existing project. The registry
    lives on the CANONICAL data root: a forked execution drive never carries
    ``state/projects.json``, so reading the task's own drive answered "no projects".
    The list requires readability (``GET /api/projects``' own check): a registry that
    cannot be read, or that is lost beside its commit witness, is a typed
    PROJECTS_ERROR; filter/offset misses in a partial read never prove absence."""
    if type(start) is not int or start < 0:
        return "⚠️ TOOL_ARG_ERROR (list_projects): start must be a nonnegative integer."
    try:
        from ouroboros.projects_registry import list_sidebar_projects, projects_summary
        limit = max(1, min(int(limit or 50), 200))
        root = canonical_data_root(ctx)
        sidebar_rows, complete = list_sidebar_projects(root, require_readable=True, with_coverage=True)
        rows = projects_summary(root, limit=limit + 1, start=start, closed=closed, oldest_first=oldest_first,
                                sidebar_rows=sidebar_rows)
    except Exception as exc:
        return f"⚠️ PROJECTS_ERROR: {type(exc).__name__}: {exc}"
    if not rows and complete:
        return "No matching projects. move_to_project places this work in a named room."
    lines = [] if rows else ["No readable projects match on this page."]
    for p in rows[:limit]:
        pid = str(p.get("id") or "")
        name = str(p.get("name") or pid)
        last = str(p.get("last_active_at") or p.get("created_at") or "")
        closure = json.dumps(p.get("closed", False), ensure_ascii=False)
        # No liveness word: the registry's activity flag records that a room ever held work, not that work runs now.
        lines.append(f"- {pid} — {name} · last_active_at={last} · closed={closure}")
    if not complete:
        lines.append("Project registry read is incomplete. Only readable matches are shown.")
    if len(rows) > limit:
        args = [f"start={start + limit}", f"limit={limit}", f"oldest_first={oldest_first!r}"]
        if closed is not None:
            args.append(f"closed={closed!r}")
        lines.append("Next: list_projects(" + ", ".join(args) + ")")
    return "Projects (move_to_project places this work in a room):\n" + "\n".join(lines)


def _close_project(ctx: ToolContext, project_id: str, note: str, reopen: bool = False) -> str:
    """Record a project decision and owe its owner one System row in that room, whose delivery the closed record names,
    as does the closing page written before it when anything is open there."""
    meta = getattr(ctx, "task_metadata", {}) or {}
    lineage = (getattr(ctx, "task_contract", {}) or {}).get("lineage") or {}
    if str(meta.get("delegation_role") or lineage.get("delegation_role") or "") == "subagent":
        return "⚠️ TOOL_ERROR (close_project): subagents inherit the parent's project scope."
    from ouroboros.project_facts import explicit_project_id_ok
    from ouroboros.projects_registry import update_project
    from supervisor import terminal_delivery as outbox

    if not explicit_project_id_ok(project_id) or not isinstance(note, str) or not note.strip() or type(reopen) is not bool:
        return "⚠️ TOOL_ARG_ERROR (close_project): supply a filesystem-clean project_id, a nonempty note and boolean reopen."
    task_id = str(getattr(ctx, "task_id", "") or "")
    if not task_id:
        return "⚠️ TOOL_ERROR (close_project): this work has no task id."
    root, at = canonical_data_root(ctx), utc_now_iso()
    by = f"ouroboros:{task_id}"
    notice = f"project-closure:{uuid.uuid4().hex}"  # the record and the page name it first; it is posted after the page
    try:
        closed = None if reopen else {"at": at, "by": by, "note": note.strip(), "delivery": notice}
        project, changed = update_project(root, project_id, return_changed=True, closed=closed) or (None, False)
    except Exception as exc:
        return f"⚠️ PROJECTS_ERROR: {type(exc).__name__}: {exc}"
    if project is None:
        return "⚠️ PROJECTS_ERROR: project is missing or no longer active."
    state = "open" if reopen else "closed"
    if not changed:
        return f"OK: project {project['id']} is already {state}."
    action = "reopened" if reopen else "closed"
    from ouroboros.host_notice import phrase

    # The mind's own note is put back as written, never translated; which work did it is the notice's detail.
    text, said = phrase(f"project_{action}", f"Ouroboros {action} this project." + (" {note}" if note.strip() else ""),
                        step="" if reopen else "reopen_project", details=f"by task {task_id}",
                        **({"note": note.strip()} if note.strip() else {}))
    page = "" if reopen else _closing_page(ctx, root, project, note.strip(), notice)
    try:
        delivery = outbox.enqueue_terminal_delivery_outcome(root, {
            "type": "send_message", "chat_id": project["chat_id"],
            "role": "system", "system_type": "project_closure", "text": text, "delivery_id": notice,
            "progress_meta": said,
        }, event_queue=getattr(ctx, "event_queue", None))
    except Exception:
        log.warning("Project closure committed but room notice failed", exc_info=True)
        delivery = outbox.ENQUEUE_UNAVAILABLE
    suffix = ("Room notice queued." if delivery in {outbox.ENQUEUE_QUEUED, outbox.ENQUEUE_ALREADY_DELIVERED}
              else f"Room notice delivery is {delivery}; the project change is saved.")
    return f"OK: project {project['id']} is {state}. {suffix}{page}"


def _closing_page(ctx: ToolContext, root: Any, project: Dict[str, Any], note: str, notice: str) -> str:
    """S2: closing a project writes its room's last page from the closing note, over the room's open rows, as
    ``chronicle_write(kind='page', covers={'to': …})`` writes one (``tools.chronicle.publish_covered_page``), naming
    the delivery of the closure's System notice (``notice``, posted after it). Nothing open writes none; a page the
    store refuses, or memory not yet active, leaves the closure standing and says why."""
    from ouroboros import chat_chain, memory_inventory
    from ouroboros.tools.chronicle import _activated, _NotActivated, _signed_author, publish_covered_page

    room = str(project["chat_id"])
    try:
        entries = memory_inventory.open_room_rows(root, room)
        if not entries:
            return " Nothing is open in its room: no page."
        author, role = _signed_author(ctx, "page")
        if author is None:
            return f" No page: a {role} drafts no project's last page."
        newest = chat_chain.format_address(max(entries, key=lambda entry: entry[2])[0])
        result, covers, _facts, _stamp = publish_covered_page(root, _activated(root), author, room, note, {"to": newest},
                                                              closes_project=str(project["id"]), closure_delivery=notice)
    except _NotActivated:
        return " No page: my memory is not active yet; the closure stands."
    except Exception as exc:  # a store refusal path or an unreadable room never undoes the closure
        log.warning("Project closure page not written", exc_info=True)
        return f" Its page was not written ({type(exc).__name__}: {exc}); the closure stands."
    if not result.ok:
        return f" Its page was not written ({result.reason}); the closure stands."
    return f" The closing note is its room's page {result.record['id']}, sealing {covers['count']} open rows."


def _manage_project_folder(ctx: ToolContext, project_id: str, action: str = "view", path: str = "",
                           expected: str = "", reason: str = "") -> str:
    """The mind's hand on a Project's working folder: the view and the one change the owner's menu calls
    (``project_folder``), on the canonical data root. A change is recorded with this work as its actor."""
    from ouroboros import project_folder
    from ouroboros.body_candidate import serving_repo_dir_for
    from ouroboros.project_facts import explicit_project_id_ok

    meta = getattr(ctx, "task_metadata", {}) or {}
    lineage = (getattr(ctx, "task_contract", {}) or {}).get("lineage") or {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    changing = action in (project_folder.CLEAR, project_folder.CHOOSE)
    if changing and str(meta.get("delegation_role") or lineage.get("delegation_role") or "") == "subagent":
        return "⚠️ TOOL_ERROR (manage_project_folder): subagents inherit the parent's project scope; they may view."
    if not explicit_project_id_ok(project_id) or not (changing or action == "view"):
        return "⚠️ TOOL_ARG_ERROR (manage_project_folder): supply a filesystem-clean project_id and action view, clear or choose."
    if changing and not (task_id and isinstance(reason, str) and reason.strip() and isinstance(expected, str) and expected):
        return ("⚠️ TOOL_ARG_ERROR (manage_project_folder): a change needs the view's expected and a reason; "
                "read the folder first with action='view'.")
    root, repo = canonical_data_root(ctx), serving_repo_dir_for(ctx)
    try:
        if not changing:
            outcome = project_folder.view_outcome(root, project_id, repo_dir=repo)
        else:
            from ouroboros.tools.project_journal import _project_scope, _scope_authority

            # A place's work changes only its own room's folder, as it writes only that room's journal; it may view any.
            refusal = _project_scope(ctx, project_id, write=True)[1] if _scope_authority(ctx)[0] == "presence" else ""
            if refusal:
                return f"{refusal} (manage_project_folder)"
            outcome = project_folder.change_folder(root, project_id, action=action, expected=expected, repo_dir=repo,
                                                   actor=f"ouroboros:{task_id}", path=str(path or ""), note=reason.strip())
    except Exception as exc:
        return f"⚠️ PROJECTS_ERROR: {type(exc).__name__}: {exc}"
    view = outcome.get("view") or {}
    if outcome["status"] == "changed":
        folder = view.get("folder") or outcome["receipt"]["new"]["working_dir"]  # the receipt's, when no view was read
        outcome["effects"] = project_folder.effects(view, action, folder)
        outcome["note"] = ("Recorded as project_folder_changed with you as the actor. The host tells nobody: "
                           "say what you changed and why where your human reads.") + (
            " The change stands, but the folder could not be read after it (view_error): do not repeat it; read it "
            "again with action='view'." if outcome.get("view_error") else "")
    elif view:
        outcome["effects"] = {offered: project_folder.effects(view, offered, str(path or "")) for offered in view["actions"]}
        outcome["note"] = ("Nothing changed. A change sends view.expected back with a reason." if view["actions"]
                           else "Nothing changed. This folder is in place; it is not moved from here.")
    return json.dumps({"ok": outcome["status"] in ("changed", "view"), **outcome}, ensure_ascii=False, default=str)


def _cause_words(code: str) -> str:
    """``": <words>"`` for a refusal code the host cause table explains, else "".

    The machine prefix and the code stay exactly where readers parse them; the
    words come from the ONE table the owner's receipts read
    (``project_dialogue.ROUTING_REFUSAL_CAUSES``), so the mind reads the same
    fact the owner does. A code without a row stays bare -- never a near-miss.
    """
    from ouroboros.project_dialogue import ROUTING_REFUSAL_CAUSES

    phrase = ROUTING_REFUSAL_CAUSES.get(str(code or "").strip(), "")
    return f": {phrase}" if phrase else ""


