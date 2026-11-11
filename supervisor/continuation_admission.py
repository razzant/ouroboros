"""The owner's Continue, queue side: one action identity, admitted once (Batch4 1A/3A).

``admit_continuation`` is the one transaction behind ``POST
/api/tasks/{id}/continue``; its order is the protocol (data half:
``ouroboros/owner_continue.py``):

1. REPLAY FIRST. The successor id is derived from ``(predecessor, nonce)``;
   an existing successor row carrying the SAME full binding is returned as
   the recorded admission at any status (still queued, held, running, ended).
   A row under that id with another binding refuses. A queue row that exists
   without its result row (a stop between the snapshot and the result write)
   is recovered as the same admission, never admitted twice.
2. CLAIM on the predecessor before any effect (``claim_on_predecessor``): eligibility,
   complete owner sources, the author-stop money pre-check and binding (room from
   the predecessor's canonical Project binding, ``continuation_room``) precede the claim.
   A failed write refuses with nothing done. The same nonce
   resumes its own claim and frozen room; another nonce is answered with the
   accepted successor.
3. ADMISSION through the existing reservation and durable enqueue (queue row
   under the queue lock, its snapshot published outside it, then the result row
   under the queue lock again, the row revalidated still queued). A
   predecessor whose writers are not proven settled — a RUNNING or
   dispatchable member of its tree, a delegated run its custody still holds
   or cannot read — admits the SAME task under a typed non-dispatch hold
   (``continuation_writer_unsettled``): no diagnostic successor, no second
   writer, no timeout release. The owner's existing conversation and controls
   diagnose; the hold releases only when a fresh check finds the writers
   settled (an explicit Resume, a reconciliation at a tree member's terminal, or
   the body cadence's re-check while the hold stands) and no newer
   Stop/Pause/Restart hold has replaced it. An explicit Resume of the Continue a
   later stop holds returns it to this hold while all its writers observably run;
   an unknown outcome still refuses (``events_budget.resume_into_writer_wait``).
   A damaged old record no link ties to the tree is no writer: the admission and its answer name it
   (``membership_gaps``).

A request that times out on the client is never "not admitted": the client
retries with the same nonce and step 1 answers it.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.project_naming import CONTINUE_TITLE_PREFIX
from ouroboros.task_store import patch_document
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

from supervisor.events_budget import HOLD_CONTINUATION_WRITER  # noqa: E402 -- record writer
from supervisor.tree_census import (  # noqa: E402, F401 -- the census's public names, read through this module too
    CONTINUE_PURPOSE, conflicting_writers, model_handoffs, only_running_writers,
    trees_model_handoffs, writer_facts, writers_unsettled_refusal,
)
from ouroboros.work_stop import writer_wait
from ouroboros.store import StoreError


def _recorded_admission(result: Dict[str, Any], predecessor: str, nonce: str) -> Optional[Dict[str, Any]]:
    from ouroboros.owner_continue import binding_sha

    admission = result.get("continuation_admission") if isinstance(result.get("continuation_admission"), dict) else {}
    binding = admission.get("binding") if isinstance(admission.get("binding"), dict) else {}
    if not binding:
        return None
    if (str(binding.get("predecessor_task_id") or "") != predecessor
            or str(binding.get("action_nonce") or "") != nonce
            or str(admission.get("binding_sha256") or "") != binding_sha(binding)):
        return {"mismatch": True}
    return admission


def _replay(q: Any, predecessor: str, nonce: str, successor: str) -> Optional[Dict[str, Any]]:
    """The recorded admission of the same nonce, at any status (module docstring, step 1); None when there is none.

    The reads and the decision run under Q; a row still carrying ``_continuation_prepared`` (its result row was
    written but its final publication never confirmed) is published outside Q, and the marker returns to the row
    when that barrier fails.
    """
    from ouroboros.work_stop import hold_fact as budget_hold_fact
    from ouroboros.owner_continue import binding_sha, carried_execution_fields, mark_claim_admitted
    from ouroboros.task_results import load_task_result, write_task_result

    try:
        prepared = None
        with q._queue_lock:
            stored = load_task_result(q.DRIVE_ROOT, successor, strict=True) or {}
            row = next((item for item in q.PENDING if str(item.get("id") or "") == successor), None)
            if row is None:
                row = (q.RUNNING.get(successor) or {}).get("task")
            if not stored and row is None:
                return None
            claim = (load_task_result(q.DRIVE_ROOT, predecessor, strict=True) or {}).get("continued_by") or {}
            admission = _recorded_admission(stored, predecessor, nonce)
            recovered = admission is None
            if admission is None and row is not None:
                admission = ((row.get("metadata") or {}).get("continuation") or {}).get("admission")
            if admission is None:
                admission = ((stored.get("metadata") or {}).get("continuation") or {}).get("admission")
            binding = (admission or {}).get("binding") or {}
            if (not binding or _recorded_admission({"continuation_admission": admission}, predecessor, nonce)
                    in (None, {"mismatch": True}) or binding != claim.get("binding")
                    or claim.get("binding_sha256") != binding_sha(binding)
                    or binding.get("successor_task_id") != successor):
                return {"ok": False, "error": "continuation_identity_conflict", "successor_task_id": successor}
            if recovered:
                write_task_result(q.DRIVE_ROOT, successor, stored.get("status") or "scheduled",
                                  continuation_admission=admission, metadata=(row or stored).get("metadata"),
                                  root_task_id=successor, chat_id=binding.get("chat_id"),
                                  project_id=binding.get("project_id") or "",
                                  **{k: v for k, v in (row or {}).items() if k in _SUCCESSOR_ROW_KEYS and v},
                                  **carried_execution_fields(binding))
            if row is not None and row.get("_continuation_prepared"):
                prepared = row.pop("_continuation_prepared")
        if prepared is not None and q.publish_queue_snapshot(reason="owner_continue_binding_recovered") is not True:
            with q._queue_lock:  # outside Q the barrier failed: the marker returns to the row it still names
                if row is not None and not row.get("_continuation_prepared"):
                    row["_continuation_prepared"] = prepared
            return {"ok": False, "error": "queue_snapshot_persist_failed",
                    "successor_task_id": successor, "unconfirmed": True}
        mark_claim_admitted(q.DRIVE_ROOT, predecessor, nonce)
        with q._queue_lock:  # admission is historical; the live row may have released its hold
            hold = budget_hold_fact(row) if row is not None else None
        return {"ok": True, "replay": True, "recovered": recovered, "task_id": successor,
                "successor_task_id": successor, "predecessor_task_id": predecessor,
                "status": str(stored.get("status") or "scheduled"),
                "held": bool(hold) if row is not None else bool(admission.get("held")),
                **({"blockers": list(hold["blockers"])} if hold and "blockers" in hold else {}),
                **({"membership_gaps": admission["membership_gaps"]} if admission.get("membership_gaps") else {})}
    except Exception as exc:
        return {"ok": False, "error": "continuation_publication_unconfirmed",
                "successor_task_id": successor, "unconfirmed": True, "detail": str(exc)[:200]}


def action_writers(q: Any, predecessor: str, *, drive_root: Any = None, owner_pause_fence_id: str = "",
                   purpose: str = "", gaps: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    """An ADDRESSED action's census (Continue, Resume, held selection) of ``predecessor``.

    It first retires the positively ended local owners of this tree only
    (``local_custody_repair``: positive proof of the owner's end) and applies
    the caller's named decision to go on past held calls of this tree whose
    outcome is unknown (``control_actor.accepting_unknown``), then observes
    exactly as ``conflicting_writers`` — which stays the passive, write-free
    reader every census and GET uses (#1554). ``purpose`` and ``gaps`` are
    passed through unchanged: only the owner's Continue and its held selection
    name ``CONTINUE_PURPOSE``; Pause and Resume keep the default.
    """
    from ouroboros.control_actor import accepted_unknown, control_actor
    from ouroboros.local_custody_repair import accept_unknown_custody, repair_ended_local_custody
    from ouroboros.utils import utc_now_iso

    root = pathlib.Path(drive_root or q.DRIVE_ROOT)
    events = pathlib.Path(q.DRIVE_ROOT) / "logs" / "events.jsonl"
    members = None
    try:
        facts = writer_facts(root, predecessor)
        if facts.census_error:
            raise ValueError(facts.census_error)
        members = set(facts.rows) | {predecessor}
        repaired = repair_ended_local_custody(root, members, root_task_id=predecessor)
        if repaired:
            q.append_jsonl(events, {"type": "local_custody_retired", "root_task_id": predecessor, "retired": repaired})
    except Exception:
        log.warning("Ended local custody of %s was not repaired; it stays held", predecessor, exc_info=True)
    decision = accepted_unknown()  # the caller's named decision for this call, on this tree only
    if decision and members:
        by = {"actor": "owner", "text": decision["text"], "at": utc_now_iso(), **control_actor()}
        try:
            outcome = accept_unknown_custody(root, members, decision["ids"], root_task_id=predecessor, by=by)
            if outcome["accepted"] or outcome["refused"]:  # a later census of the same call finds nothing new
                q.append_jsonl(events, {"type": "local_custody_unknown_accepted", "root_task_id": predecessor,
                                        "by": by, **outcome})
        except Exception:
            log.warning("The named decision on %s was not recorded; its calls stay held", predecessor,
                        exc_info=True)
    return conflicting_writers(q, predecessor, drive_root=drive_root, owner_pause_fence_id=owner_pause_fence_id,
                               purpose=purpose, gaps=gaps)


# Integrity bound only: a recorded chain is a handful of Continues; a corrupted one
# must never make one admission walk its edges without limit.
_WORK_CHAIN_MAX_HOPS = 64


def _work_name(record: Dict[str, Any]) -> str:
    """An episode's name as its card shows it (coined, else authored), without Continue prefixes."""
    from ouroboros.project_naming import work_shown_name

    return work_shown_name(record)


def _work_identity(q: Any, predecessor: str, result: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    """``(work_id, work_name, first root's record or {})`` of the work a Continue of ``predecessor`` belongs to.

    A stamped episode names its work: the first root's current name, else the
    stamped copy. An older one is walked back over its
    recorded ``predecessor_task_id`` edges to the first root, whose id and card
    name are the work's. As in the work projection's walk over the same edges,
    an edge to a record that cannot be read ends at the last episode read, and a
    cycle (or the integrity bound) at the predecessor itself, so a stamped
    successor and its unstamped episodes resolve to one work. Reads only stored
    results, outside the queue lock.
    """
    from ouroboros.task_results import load_task_result

    current, record, seen = predecessor, result, {predecessor}
    for _hop in range(_WORK_CHAIN_MAX_HOPS):
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        continuation = metadata.get("continuation") if isinstance(metadata.get("continuation"), dict) else {}
        if continuation.get("work_id"):
            work_id = str(continuation["work_id"])
            try:
                origin = (load_task_result(q.DRIVE_ROOT, work_id) if q is not None else None) or {}
            except Exception:
                origin = {}
            return work_id, _work_name(origin) or str(continuation.get("work_name") or "") or _work_name(record), origin
        earlier = str(continuation.get("predecessor_task_id") or "")
        if not earlier:
            return current, _work_name(record), record
        if earlier in seen:
            break
        try:
            earlier_record = load_task_result(q.DRIVE_ROOT, earlier) if q is not None else None
        except Exception:
            earlier_record = None
        if not isinstance(earlier_record, dict) or not earlier_record:
            return current, _work_name(record), {}
        current, record = earlier, earlier_record
        seen.add(earlier)
    return predecessor, _work_name(result), {}


def _recorded_work_limits(result: Dict[str, Any], metadata: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, Any]:
    """The same work's own limits, grants and acceptance conditions, as its record carries them.

    Each field is read the way the predecessor's registry and acceptance readers bound it: withheld tools from every
    carrier (``task_disabled_tools``), resources with the contract winning per key over the row's own map, the policy,
    the ingress claims, the answer protocol and answer-review requirement, the delegation rights and the budget profile (its cost stop) an API caller
    declared from the contract (canonical over stored) else the row, as ``build_task_contract`` reads them, and its own
    direct handbook answer, false too, from the contract alone. The successor holds them on its stored contract carrier,
    which its admission builds into its own contract (``enqueue_task``) and its scheduled result keeps for a successor
    that never starts; its workspace executor rides where the API door stores it. Nothing is added or widened, and only
    these fields move: the whole contract would also hand on the predecessor's identity (lineage, objective, source).
    """
    from ouroboros.consciousness_authority import task_disabled_tools
    from ouroboros.contracts.task_contract import (
        build_task_contract, normalize_allowed_resources, normalize_answer_protocol, normalize_disabled_tools,
    )

    resources: Dict[str, Any] = {}
    for source in (result, metadata, contract):
        resources.update(normalize_allowed_resources(source.get("allowed_resources")))
    built = build_task_contract(result)  # merges ``contract`` the same way: its policy and claims, read once
    limits = {
        "allowed_resources": resources,
        "resource_policy": built["resource_policy"],
        "disabled_tools": normalize_disabled_tools(sorted(task_disabled_tools({**result, "task_contract": contract}))),
        "acceptance_claims": built["acceptance_claims"],
        "answer_protocol": normalize_answer_protocol(contract.get("answer_protocol") or metadata.get("answer_protocol")
                                                     or result.get("answer_protocol")),
        **{key: value for key, value in built.items()
           if key in {"delegation_budget", "budget_profile", "context_requires_development", "review_answer"}},
    }
    fields = {key: value for key, value in limits.items() if value or value is False}  # a direct handbook "no" moves too
    carried: Dict[str, Any] = {"task_contract": fields} if fields else {}
    executor = metadata.get("executor_ref") or result.get("executor_ref")
    if isinstance(executor, dict) and executor:
        carried["executor_ref"] = dict(executor)
    teardown = str(metadata.get("service_teardown") or "").strip().lower()
    if teardown in {"keep", "stop"}:  # the services' fate at the task's end (``tools/services.task_service_teardown``)
        carried["service_teardown"] = teardown
    return carried


# The successor row's own copy of what its Continue, restore and place read/stop read back (the queue row too).
_SUCCESSOR_ROW_KEYS = ("deadline_at", "reasoning_effort", "workspace_root", "workspace_mode", "delegation_role",
                       "memory_mode", "drive_root", "child_drive_root", "budget_drive_root", "depth")
_GROUP_FIELDS = ("billing_group_id", "billing_group_limit_usd", "billing_group_limit_source",
                 "billing_group_limit_revision")  # the whole-work group a chain keeps (``_billing_group``)
# A recorded root drive mode -> the mode its successor's own drive is prepared with: every root drive
# producer forks unless it recorded "empty"; the API door's project-scoped "shared" runs forked.
_DRIVE_MODES = {"empty": "empty", "forked": "forked", "shared": "forked"}


def _successor_drive_mode(q: Any, predecessor: str, result: Dict[str, Any]) -> str:
    """The mode of the drive the predecessor's own producer prepared; "" when it ran on the canonical drive. Evidence
    is a producer's fact only: the row's recorded child drive at ``prepare_task_drive``'s path for this task (API door,
    promotion, schedule occurrence, this admission), or that producer's record naming this task; a ``task_drives/<id>``
    scratch folder or any other path proves nothing. The mode is the recorded one (``_DRIVE_MODES``), else the record's;
    an unsupported value, or one the record contradicts, refuses as unknown: never invented, never silently canonical."""
    from ouroboros.confined_files import root_spellings
    from ouroboros.headless import prepared_drive_mode, task_state_dir
    from ouroboros.task_custody import recorded_execution_drive

    drive = task_state_dir(q.DRIVE_ROOT, predecessor) / "data"
    state = prepared_drive_mode(drive, predecessor) or ""
    if not (recorded_execution_drive(result, drive, root_spellings(q.DRIVE_ROOT)) or state):
        return ""
    raw = str(result.get("memory_mode") or "").strip().lower()
    mode = _DRIVE_MODES.get(raw, "") if raw else state
    if not mode or (state and state != mode):
        raise ValueError("continuation_drive_mode_unknown")
    return mode


def _successor_drive_fields(q: Any, binding: Dict[str, Any]) -> Dict[str, Any]:
    """The successor's own drive fields, at the one producer's path for its id ({} on the canonical drive)."""
    if not binding.get("memory_mode"):
        return {}
    from ouroboros.headless import task_state_dir

    path = str(task_state_dir(q.DRIVE_ROOT, binding["successor_task_id"]) / "data")
    return {"memory_mode": binding["memory_mode"], "drive_root": path, "child_drive_root": path,
            "budget_drive_root": str(q.DRIVE_ROOT)}


def _prepare_successor_drive(q: Any, successor: str, mode: str, project_id: str) -> None:
    """The successor's OWN drive, prepared once by ``headless.prepare_task_drive`` from the current canonical root,
    never from the predecessor's. A drive its producer's record (written last) names for this successor and mode is
    complete and reused untouched; one naming anything else is a conflict; one without a record never finished and is
    prepared again. A failed preparation admitted nothing, so its partial drive is freed."""
    from ouroboros.headless import prepare_task_drive, prepared_drive_mode, task_state_dir

    state = prepared_drive_mode(task_state_dir(q.DRIVE_ROOT, successor) / "data", successor)
    if state == mode:
        return
    if state is not None:
        raise ValueError("continuation_drive_conflict")
    try:
        prepare_task_drive(q.DRIVE_ROOT, successor, mode, project_id=project_id)
    except Exception as exc:
        _release_unadmitted_drive(q, successor)
        raise ValueError("continuation_drive_unavailable") from exc


def _release_unadmitted_drive(q: Any, successor: str, staged: Optional[List[Dict[str, Any]]] = None) -> None:
    """Free what this attempt prepared once the successor's admission is PROVEN absent (a refused enqueue, a rollback
    its snapshot recorded, a failed preparation or input copy): its own drive through the one deletion owner, else the
    input copies it made (``staged``) while no reservation or live row holds the id; never after a durable or unknown
    publication, whose accepted work keeps them."""
    if staged is not None:
        from ouroboros.artifacts import remove_staged_attachments

        with q._queue_lock:  # a retry reserves under this lock before it copies or reuses the same files
            if successor not in q.ADMISSION_RESERVATIONS and q.task_settlement_liveness(successor) is False:
                remove_staged_attachments(staged)
        return
    from ouroboros.headless import remove_subagent_task_drive

    try:
        remove_subagent_task_drive(q.DRIVE_ROOT, successor, live=q.task_settlement_liveness,
                                   guard=q.task_settlement_interlock, admission_rollback=True)
    except Exception:
        log.warning("Continue %s: its unadmitted drive stays for the drive prune", successor, exc_info=True)


def _with_document_receipts(q: Any, predecessor: str, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Carry observed document imports (``engine_imports``) from the predecessor's canonical custody marks.

    The successor preserves ready, uncertain and terminal states with their exact key and request; it cannot mint
    over a gap. Release stays with the originating task, whose whole work includes this successor. Unreadable
    marks refuse ``continuation_inputs_unavailable`` before a copy can lose that identity.
    """
    from ouroboros.delegate_documents import DocumentSelectionError, carry_owner_receipts

    try:
        carry_owner_receipts(q.DRIVE_ROOT, predecessor, inputs)
    except DocumentSelectionError as exc:
        raise ValueError("continuation_inputs_unavailable") from exc
    return inputs


def _carry_inputs(q: Any, task: Dict[str, Any], predecessor: str, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy the work's declared inputs into the successor's OWN store and declare them on its row as the API door does
    (manifest, ``attachments``, images, a read line each); returns the copy. A failed copy keeps none of its files and,
    having admitted nothing, frees the successor's own drive too (``continuation_inputs_unavailable``)."""
    from ouroboros.artifacts import attachment_manifest_projection, materialize_inherited_attachment_manifest
    from ouroboros.artifacts import remove_staged_attachments
    from ouroboros.owner_continue import inputs_text

    drive = pathlib.Path(task.get("drive_root") or q.DRIVE_ROOT)
    copied, error = materialize_inherited_attachment_manifest(inputs, drive, task["id"])
    try:
        if error:
            raise OSError(error)
        authority = attachment_manifest_projection(drive, task["id"], copied) if copied else {}
    except (OSError, ValueError, TypeError) as exc:
        remove_staged_attachments(copied)
        if task.get("child_drive_root"):
            _release_unadmitted_drive(q, task["id"])
        raise ValueError("continuation_inputs_unavailable") from exc
    if authority:
        task.update(authority, attachments=authority["attachment_manifest"],
                    attachment_images=[row for row in copied if row.get("status") == "staged" and row.get("is_image")])
        task["text"] = task["objective"] = f"{task['text']}\n\n{inputs_text(predecessor, copied, authority)}"
    return copied


def _author_stop_note(result: Dict[str, Any]) -> str:
    """The stop's own recorded reason, as its account, and the retention its offer does not add (``still_continuable``)."""
    from ouroboros.review_records import recorded_stop_rationale

    rationale = recorded_stop_rationale(result)
    return ((f"The stopping author's recorded reason (its own account, not an owner instruction; verbatim): {rationale}\n"
             if rationale else "The stop recorded no reason.\n")
            + "Being offered Continue added no retention: the work's ordinary retention, Keep and custody rules still "
              "applied, so some of that task's native session material (helper sessions, snapshots, review threads) may "
              "already be released, and a continue_from that can no longer bind is refused. Its stored result and the "
              "owner's words remain the sources.")


def _tree_note(q: Any, predecessor: str) -> str:
    """The predecessor's helper tree as one host fact: each member, its status and parent, and the reader of
    its work order and final; its counters are not copied. A tree that cannot be read is named, never empty."""
    from ouroboros.task_store import tree_facts

    try:
        census = tree_facts(q.DRIVE_ROOT, predecessor)
    except Exception as exc:
        log.warning("Predecessor tree unreadable for %s", predecessor, exc_info=True)
        return f"Its helper tree could not be read when this Continue was admitted ({type(exc).__name__})."
    named = [f"{tid} ({row.get('status') or 'status unknown'}"
             + (f", under {row['parent_task_id']})" if row.get("parent_task_id") not in (None, "", predecessor) else ")")
             for tid, row in sorted(census.members.items()) if tid != predecessor]
    named += [f"{tid} (record unreadable)" for tid, gap in sorted(census.unresolved.items())
              if gap.get("linked") and tid not in census.members]
    return ("Its helper tree when this Continue was admitted (host facts, not owner instructions): "
            f"{', '.join(named)}. get_task_result(task_id=<id>) returns each one's final, and with "
            "include_work_order_source=True its work order.") if named else ""


def _successor_task(q: Any, predecessor: str, result: Dict[str, Any], binding: Dict[str, Any],
                    verdict: Dict[str, Any], sources: Dict[str, Any], admission: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.consciousness_authority import consciousness_origin_metadata
    from ouroboros.dialogue_provenance import presence_binding_authority_metadata, presence_relation_metadata
    from ouroboros.owner_continue import AUTHOR_STOP_CAUSE, owner_corpus_rows, recorded_system_checkout, work_order_text
    from ouroboros.project_naming import work_name_open
    from ouroboros.task_pacing import recorded_producer_ceiling

    authority_source: Dict[str, Any] = {}
    try:
        from ouroboros.agent_startup_checks import valid_task_result_authority_source
        from ouroboros.server_routing_context import _task_result_ground_truth

        source = _task_result_ground_truth({**result, "task_id": predecessor}).get("authority_source")
        if valid_task_result_authority_source(source, predecessor):
            authority_source = dict(source)
    except Exception:
        log.debug("Predecessor authority pointer unavailable for %s", predecessor, exc_info=True)
    deadline_at = str(binding.get("deadline_at") or "")
    drive = _successor_drive_fields(q, binding)
    # Every episode carries its work: the first root's id and name. The title is
    # that name under ONE prefix, however long the chain.
    work_id, work_name, origin = _work_identity(q, predecessor, result)
    text = work_order_text(predecessor, verdict["cause"], sources, deadline_at=deadline_at,
                           authority_source=authority_source, checkout=recorded_system_checkout(result),
                           successor_task_id=str(binding.get("successor_task_id") or ""),
                           drive_mode=drive.get("memory_mode", ""),
                           stop_note=_author_stop_note(result) if verdict["cause"] == AUTHOR_STOP_CAUSE else "",
                           tree_note=_tree_note(q, predecessor), work_name_open=bool(origin) and work_name_open(origin))
    work_name = (work_name or _work_name(result)
                 or (_work_name({"title": result.get("objective")}) or predecessor)[:80])
    title = work_name[:80]
    original = sources.get("original") or {}
    origin_ref = original.get("origin_message_ref")
    from ouroboros.settings_scales import EFFORT_SCALE

    # The same work keeps the effort and the model ROLE it was explicitly started on. A
    # root's stored values are only those explicit choices (dispatch stamps children alone;
    # a switch_model is not stored; ``metadata.model_role`` names a slot such as a
    # consciousness wake's, never the route it resolved to), read from the result row its
    # admission wrote even if no worker ever ran. A task on the install default stores
    # neither and so follows the current default.
    effort = str(result.get("reasoning_effort") or "")
    # Admission/replay persist under metadata until a worker publishes the canonical
    # contract. Merge stored carriers per field, canonical winning; don't normalize
    # an absent handbook choice into false before deciding whether to copy it.
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    model_role = str(metadata.get("model_role") or "").strip()
    # The same work keeps its primary, a recorded follow included (route_choice),
    # never the room's newer choice.
    from ouroboros.route_choice import LEAD_KEY, MODEL_ROLE_SETTINGS, lead_snapshot

    model_role = model_role if model_role in MODEL_ROLE_SETTINGS else ""
    lead = lead_snapshot(metadata.get(LEAD_KEY))
    stored = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict) else {}
    canonical = result.get("task_contract") if isinstance(result.get("task_contract"), dict) else {}
    contract = {**stored, **canonical}
    self_body_docs = contract.get("context_requires_self_body_docs")
    # The recorded resource choice rides with the folder (#1315), so placement keeps an explicit
    # folder or opt-out instead of the destination room's default (``project_placement``).
    intent = metadata.get("resource_intent")
    task: Dict[str, Any] = {
        **({"reasoning_effort": effort} if effort in EFFORT_SCALE else {}),
        **({"context_requires_self_body_docs": self_body_docs} if isinstance(self_body_docs, bool) else {}),
        "id": binding["successor_task_id"], "type": "task", "chat_id": binding.get("chat_id"),
        "project_id": str(binding.get("project_id") or ""), "text": text, "objective": text,
        "title": f"{CONTINUE_TITLE_PREFIX}{title}", "suggested_name": f"{CONTINUE_TITLE_PREFIX}{title}",
        "root_task_id": binding["successor_task_id"], "depth": result.get("depth", 0), "delegation_role": "root",
        **({"deadline_at": deadline_at} if deadline_at else {}),
        **{key: binding[key] for key in ("workspace_root", "workspace_mode") if binding.get(key)},
        # The work's explicit execution budget (#939): the same total, and the closed execution
        # handed on once (``settle_execution_carry`` at assignment), never a fresh allowance.
        **({"execution_timeout_sec": binding["execution_timeout_sec"],
            "_execution_handoff_sec": binding["execution_consumed_sec"]} if "execution_timeout_sec" in binding else {}),
        # The work's recorded isolation: its OWN drive of the same mode, prepared at admission (``drive``).
        **drive,
        "metadata": {
            # Legacy results may keep their verified owner reference only on
            # the source row. Explicit kind/origin stamps still take priority;
            # a wake's own row names itself as the successor's origin wake.
            **consciousness_origin_metadata({**metadata, "task_id": predecessor, **(
                {"origin_message_ref": origin_ref} if isinstance(origin_ref, dict) and origin_ref else {})}),
            # A place-bound predecessor's binding, or related owner work's relation; never its speaker, turn or lease.
            **presence_binding_authority_metadata(metadata, task_contract=contract), **presence_relation_metadata(metadata),
            **({"project_id": binding["project_id"]} if binding.get("project_id") else {}),
            **({"model_role": model_role} if model_role else {}),
            **({LEAD_KEY: lead} if lead else {}),
            **({"resource_intent": dict(intent)} if isinstance(intent, dict) and intent.get("kind") else {}),
            # The same work's withheld tools, protected artifacts, resources, criteria, executor and teardown.
            **_recorded_work_limits(result, metadata, contract),
            # A producer's money TOTAL for the whole work (a wake's allowance), unchanged: its group spend meets it.
            **({"root_cost_ceiling_usd": total} if (total := recorded_producer_ceiling(result)) else {}),
            **{key: drive[key] for key in ("child_drive_root", "budget_drive_root") if key in drive},
            # Keep verified owner-door provenance on its canonical carrier;
            # the new objective below is still the host's continuation text.
            **({"origin_message_ref": dict(origin_ref), "origin_message_text": original["content"]}
               if isinstance(origin_ref, dict) and origin_ref else {}),
            # The objective is host-composed facts; the owner corpus is seeded
            # with the owner's exact words only (loop_messages seeding).
            "objective_author": {"kind": "continuation", "predecessor_task_id": predecessor},
            "owner_corpus": owner_corpus_rows(sources),
            "continuation": {
                "predecessor_task_id": predecessor, "work_id": work_id, "work_name": work_name,
                "cause": verdict["cause"],
                "owner_sources": {"original": sources.get("original"), "schedule": sources.get("schedule"),
                                  **({"start": sources["start"]} if sources.get("start") else {}),
                                  **({"wake": sources["wake"]} if sources.get("wake") else {}),
                                  **({"api": sources["api"]} if sources.get("api") else {}),
                                  "later": sources.get("later") or [],
                                  **({"confirmed_place": sources["confirmed_place"]} if sources.get("confirmed_place") else {}),
                                  "held_owner_messages": sources.get("held_owner_messages") or {},
                                  "open_relays": sources.get("open_relays") or {}},
                "peer_context": sources.get("peer_context") or [],
                **{key: binding.get(key) for key in _GROUP_FIELDS},
                "admission": admission,
            },
        },
    }
    if authority_source:
        task["predecessor_task_id"] = predecessor
        task["predecessor_authority_source"] = authority_source
    # A follow-up accepted before the relation existed: what already ran keeps what it ran with, and this NEW work
    # reads the origin its receipt names (carried by value across Continues), as a new occurrence does.
    from supervisor.schedule_occurrence import _adopt_relation, _origin_relation

    _adopt_relation(task, _origin_relation({"metadata": metadata, "task_contract": contract},
                                           {"followup_origin": (sources.get("schedule") or {}).get("origin")}))
    return task


def _billing_group(q: Any, predecessor: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """The whole-work group the successor spends from, and the cap it started under.

    A chain keeps the FIRST root's group and cap; otherwise the predecessor's
    own earliest live ledger row names the group it spent in and the cap it
    started under (an older block's own literal included, ``legacy_live``).
    With neither a durable initial binding nor a ledger cap, the predecessor
    stays its own group, its spend preserved, under the configured cap — the
    choice is pinned on the predecessor's result (``legacy_default``) so its
    own later work and every successor share one ceiling.
    """
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    carried = result.get("billing_group") or metadata.get("billing_group") or metadata.get("continuation") or {}
    if carried.get("billing_group_id") and "billing_group_limit_usd" in carried:
        return {key: carried.get(key) for key in _GROUP_FIELDS}
    from ouroboros.usage_admission import ledger_billing_binding, original_group_limit

    group = str(result.get("root_task_id") or predecessor)
    recorded = ledger_billing_binding(q.DRIVE_ROOT, group)  # the root's own first row: its group, not only itself
    if recorded:
        return {key: recorded.get(key) for key in ("billing_group_id", "billing_group_limit_usd", "billing_group_limit_source")}
    found = original_group_limit(q.DRIVE_ROOT, group)
    if found["source"] != "no_attempt_recorded":
        return {"billing_group_id": group, "billing_group_limit_usd": found["limit_usd"],
                "billing_group_limit_source": found["source"]}
    from ouroboros.config import runtime_setting
    from ouroboros.task_results import stamp_task_result_schema

    limit = float(runtime_setting("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
    binding = {"billing_group_id": group, "billing_group_limit_usd": limit if limit > 0 else None,
               "billing_group_limit_source": "legacy_default", "billing_group_limit_revision": utc_now_iso()}

    def pin(current):
        nonlocal binding
        if current.get("billing_group"):  # already chosen (by an earlier Continue or admission): share it
            binding = current["billing_group"]
            return None
        return stamp_task_result_schema({**current, "billing_group": binding})

    if result.get("status"):  # a recorded predecessor: the choice becomes its durable binding, or nothing is admitted
        try:
            patch_document(q.DRIVE_ROOT, group, pin)
        except (OSError, ValueError, TimeoutError, StoreError) as exc:
            # An unpinned choice would let the predecessor's own later work follow a changed
            # setting instead of the successor's ceiling: refuse now; the same nonce retries.
            log.warning("Could not pin the default billing group on %s", group, exc_info=True)
            raise ValueError("billing_authority_unavailable") from exc
    return {key: binding.get(key) for key in ("billing_group_id", "billing_group_limit_usd", "billing_group_limit_source")}


def _group_money_spent(q: Any, billing: Dict[str, Any]) -> bool:
    """Whether the whole work's known spend reached its cap. An author's stop is ended by no money rail, so its row
    cannot say; the offer stays a row read, and an unreadable ledger raises into the caller's billing refusal."""
    from ouroboros.usage_accounting import usage_projection

    limit = billing.get("billing_group_limit_usd")
    return limit is not None and usage_projection(
        q.DRIVE_ROOT, billing_group_id=billing["billing_group_id"])["settled_usd"] >= float(limit)


def _prepare_project_room(q: Any, predecessor: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """The successor's prepared Project basis; bound work also binds its successor.

    A fresh admission carries the basis (the producer seam schedule_task uses); a
    basis-less row would be held as legacy forever. When the predecessor is bound
    to the claim's room, the successor joins it through the same writer a timeout
    retry uses, so late frames, helpers and history resolve the room by lineage,
    with the original owner message as its origin. The successor id belongs to
    this claim alone: the bind is idempotent, fenced by the basis and made before
    the row exists. Refusals raise; the queue lock is not held here.
    """
    from ouroboros.projects_registry import bind_task_to_project, project_admission_view, project_binding_for_task

    bound = project_binding_for_task(q.DRIVE_ROOT, predecessor, strict=True) or {}
    basis = project_admission_view(q.DRIVE_ROOT, task["project_id"], frozen=True,
                                   allow_unregistered=bound.get("project_id") != task["project_id"])
    if (bound.get("project_id"), bound.get("project_chat_id")) == (task["project_id"], task["chat_id"]):
        ref = bound.get("source_ref")
        origin = ({"ref": ref, **({"text": bound["source_text"]} if "source_text" in bound else {})}
                  if isinstance(ref, dict) else {"absent": bound.get("origin_absent") or "post_hoc_unresolved"})
        bind_task_to_project(q.DRIVE_ROOT, task["id"], task["project_id"], task["chat_id"],
                             origin=origin, admission_basis=basis)
    return basis


def admit_continuation(predecessor_task_id: str, *, action_nonce: str, declared_by: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Admit (or replay) Continue of one eligible root (module docstring). ``declared_by`` is an HTTP caller's own
    statement of who pressed, recorded as ``admission.by`` and never a permission; a mind's door binds ``control_actor``."""
    from ouroboros.owner_continue import (
        AUTHOR_STOP_CAUSE, VERB, BINDING_VERSION, binding_sha, carried_execution_fields, claim_on_predecessor, continuation_eligibility,
        continuation_room, declared_inputs, execution_carry, owner_sources, successor_id, valid_nonce,
    )
    from ouroboros.task_results import STATUS_SCHEDULED, load_task_result, validate_task_id, write_task_result

    from supervisor import queue as q
    predecessor = validate_task_id(predecessor_task_id)
    nonce = valid_nonce(action_nonce)
    successor = successor_id(predecessor, nonce)
    replay = _replay(q, predecessor, nonce, successor)
    if replay is not None:
        return replay
    from supervisor.queue_snapshot import saved_queue_unowned

    if saved_queue_unowned():  # a successor published now would replace the saved queue (#1952)
        return {"ok": False, "error": "queue_not_restored"}
    try:
        result = load_task_result(q.DRIVE_ROOT, predecessor, strict=True) or {}
    except Exception:
        return {"ok": False, "error": "predecessor_record_unreadable"}
    claim = result.get("continued_by") if isinstance(result.get("continued_by"), dict) else {}
    if claim and str(claim.get("action_nonce") or "") != nonce:
        return {"ok": False, "error": "already_continued",
                "successor_task_id": str(claim.get("successor_task_id") or ""),
                **({"state": "bound", "action_nonce": claim.get("action_nonce")}
                   if claim.get("state") == "bound" else {})}
    verdict = continuation_eligibility(result, predecessor) if not claim else {
        "eligible": True, "cause": str((claim.get("binding") or {}).get("cause") or ""), "refusal": ""}
    if not verdict["eligible"]:
        return {"ok": False, "error": verdict["refusal"], "cause": verdict["cause"]}
    sources = owner_sources(q.DRIVE_ROOT, result, predecessor)
    if sources["gaps"]:
        return {"ok": False, "error": "owner_source_missing", "gaps": list(sources["gaps"])}
    try:  # before anything is written: an unknown mode or input set leaves the predecessor untouched
        drive_mode = "" if claim else _successor_drive_mode(q, predecessor, result)
        inputs = _with_document_receipts(q, predecessor, declared_inputs(predecessor, result, sources))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        billing = {} if claim else _billing_group(q, predecessor, result)
        if billing and verdict["cause"] == AUTHOR_STOP_CAUSE and _group_money_spent(q, billing):
            return {"ok": False, "error": "hard_limit_reached", "cause": "budget_exhausted"}
    except Exception:
        return {"ok": False, "error": "billing_authority_unavailable"}
    try:
        # A recorded claim keeps the room it froze, even if the work was converted since.
        room = {} if claim else continuation_room(q.DRIVE_ROOT, predecessor, result)
    except (OSError, ValueError):
        return {"ok": False, "error": "project_routing_fence_lookup_failed"}
    binding = dict(claim.get("binding") or {}) if claim else {
        "verb": VERB, "binding_version": BINDING_VERSION, "predecessor_task_id": predecessor,
        "action_nonce": nonce, "successor_task_id": successor, "cause": verdict["cause"], **room,
        "workspace_root": str(result.get("workspace_root") or ""),
        "workspace_mode": str(result.get("workspace_mode") or ""),
        "deadline_at": str(result.get("deadline_at") or (result.get("metadata") or {}).get("deadline_at")
                           or (result.get("task_contract") or {}).get("deadline_at") or ""), **billing,
        # Eligibility refused an unknown or spent budget: this is the known total and closed execution.
        **(execution_carry(result) or {}),
        **({"memory_mode": drive_mode} if drive_mode else {}),  # its own drive's mode; none on the canonical drive
    }
    binding["admission_token"] = f"continue:{binding_sha({k: v for k, v in binding.items() if k != 'admission_token'})[:40]}"
    if not claim:
        try:
            claim, _created = claim_on_predecessor(q.DRIVE_ROOT, predecessor, nonce, binding, declared_by=declared_by)
        except ValueError as exc:
            text = str(exc)
            if text.startswith("already_continued:"):
                try:
                    other = (load_task_result(q.DRIVE_ROOT, predecessor, strict=True) or {}).get("continued_by") or {}
                    return {"ok": False, "error": "already_continued", "successor_task_id": other["successor_task_id"],
                            **({"state": "bound", "action_nonce": other["action_nonce"]}
                               if other.get("state") == "bound" else {})}
                except Exception:
                    return {"ok": False, "error": "continuation_unconfirmed"}
            return {"ok": False, "error": text or "continuation_refused"}
        except Exception as exc:
            return {"ok": False, "error": "continuation_claim_unwritable", "detail": str(exc)[:200]}
        binding = dict(claim["binding"])
    from ouroboros.pause_notices import gap_fact

    gaps: List[Dict[str, str]] = []
    blockers = action_writers(q, predecessor, purpose=CONTINUE_PURPOSE, gaps=gaps)
    admission = {"by": claim.get("by") or {},
                 "binding": binding, "binding_sha256": binding_sha(binding), "admitted_at": utc_now_iso(),
                 "held": bool(blockers), **({"membership_gaps": gap_fact(gaps)} if gaps else {})}
    task = _successor_task(q, predecessor, result, binding, verdict, sources, admission)
    if blockers:
        from supervisor.events_budget import hold_continuation_writers

        hold_continuation_writers(task, predecessor, blockers)
    project_basis = None
    if task.get("project_id"):
        try:
            project_basis = _prepare_project_room(q, predecessor, task)
        except (OSError, ValueError, RuntimeError) as exc:
            # The claim stays: the same nonce retries once the authority is readable.
            return {"ok": False, "error": getattr(exc, "reason", "project_routing_fence_lookup_failed"),
                    "successor_task_id": successor}
    token = binding["admission_token"]
    reservation = q.reserve_task_admission(successor, token)
    if reservation.get("status") == "existing_same_token":
        return _replay(q, predecessor, nonce, successor) or {"ok": False, "error": "continuation_unconfirmed"}
    if reservation.get("status") not in {"reserved", "already_reserved"}:
        return {"ok": False, "error": str(reservation.get("reason") or "admission_blocked"),
                "successor_task_id": successor, "unconfirmed": True}
    task["_admission_token"] = token
    task["_continuation_prepared"] = admission["binding_sha256"]
    try:  # file copies never run under the queue lock
        if binding.get("memory_mode"):
            _prepare_successor_drive(q, successor, binding["memory_mode"], str(task.get("project_id") or ""))
        staged = _carry_inputs(q, task, predecessor, inputs)
    except ValueError as exc:
        q.release_task_admission(successor, token)
        return {"ok": False, "error": str(exc), "successor_task_id": successor}
    refused, unadmitted = None, False  # a proven absence frees the successor's drive or input copies after the lock
    # Q spans memory only: the row is enqueued under Q, published outside it, its result row written under Q again
    # with the row revalidated still queued (a ``_continuation_prepared`` row is never dispatched; only a Stop can
    # take it meanwhile), and its final publication runs outside Q once more.
    with q.prepared_root_billing(task), q._queue_lock:  # the ledger read happens before the lock
        admitted = q.enqueue_task(task, project_admission=project_basis, continuation=True)  # no consciousness slot
        blocked = isinstance(admitted, dict) and admitted.get("_admission_blocked")
        if blocked:
            q.release_task_admission(successor, token)
    if blocked:
        refused, unadmitted = {"ok": False, "error": str(admitted["_admission_blocked"]),
                               "successor_task_id": successor}, True
    elif q.publish_queue_snapshot(reason="owner_continue_admitted") is not True:  # durable before ok
        with q._queue_lock:
            q.PENDING[:] = [row for row in q.PENDING if str(row.get("id") or "") != successor]
        unadmitted = q.publish_queue_snapshot(reason="owner_continue_rollback") is True
        q.release_task_admission(successor, token)
        # The claim stays: the same nonce retries this admission, a new one is refused.
        refused = {"ok": False, "error": "queue_snapshot_persist_failed", "successor_task_id": successor}
    else:
        with q._queue_lock:
            if not any(row is admitted for row in q.PENDING):
                # A Stop's custody took the row between the holds and settles it; the claim stays with the
                # predecessor and the same nonce reads the recorded outcome (``_replay``).
                q.release_task_admission(successor, token)
                return {"ok": False, "error": "continuation_unconfirmed", "successor_task_id": successor,
                        "unconfirmed": True, "detail": "the admitted row left the queue before its record was written"}
            try:
                write_task_result(
                    q.DRIVE_ROOT, successor, STATUS_SCHEDULED, continuation_admission=admission,
                    chat_id=task.get("chat_id"), project_id=task.get("project_id") or "", title=task["title"],
                    suggested_name=task["suggested_name"], root_task_id=successor, metadata=task["metadata"],
                    # The binding a successor that never starts must still hand on to its own Continue.
                    **{key: task[key] for key in _SUCCESSOR_ROW_KEYS if task.get(key)},
                    **carried_execution_fields(binding),
                    **({"reason_code": HOLD_CONTINUATION_WRITER,
                        "resource_limit": {"status": "budget_hold", "auto_resume": False, "exact_continuation": False,
                                           **task["_budget_pause_hold"]}} if blockers else {}),
                    result=("Continue accepted; held until the previous attempt's own work is settled."
                            if blockers else "Continue accepted and durably scheduled."))
            except Exception as exc:  # the queue row and its snapshot are durable: the drive stays with them
                return {"ok": False, "error": "continuation_publication_unconfirmed",
                        "successor_task_id": successor, "unconfirmed": True, "detail": str(exc)[:200]}
            prepared = admitted.pop("_continuation_prepared")
        if q.publish_queue_snapshot(reason="owner_continue_committed") is not True:
            with q._queue_lock:
                admitted["_continuation_prepared"] = prepared
            return {"ok": False, "error": "queue_snapshot_persist_failed",
                    "successor_task_id": successor, "unconfirmed": True}
        q.release_task_admission(successor, token)
    if refused is not None:
        if unadmitted and (binding.get("memory_mode") or staged):
            _release_unadmitted_drive(q, successor, None if binding.get("memory_mode") else staged)
        return refused
    try:
        from ouroboros.owner_continue import mark_claim_admitted

        mark_claim_admitted(q.DRIVE_ROOT, predecessor, nonce)
    except Exception:
        # Only the predecessor's mailbox release waits on this mark; a replay rewrites it.
        log.warning("Continue of %s admitted but its claim mark was not written", predecessor, exc_info=True)
    append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                 {"ts": utc_now_iso(), "type": "owner_continue_admitted", "task_id": successor,
                  "predecessor_task_id": predecessor, "cause": verdict["cause"], "held": bool(blockers),
                  "blockers": blockers, "owner_visible": True,
                  "toast_once": f"{successor}:continue-admitted"})
    return {"ok": True, "replay": False, "task_id": successor, "successor_task_id": successor,
            "predecessor_task_id": predecessor, "status": STATUS_SCHEDULED, "held": bool(blockers),
            **({"blockers": blockers} if blockers else {}),
            **({"membership_gaps": admission["membership_gaps"]} if gaps else {})}


def release_settled_continuations(tree_root: str) -> List[str]:
    """Release Continues whose predecessor ``tree_root`` has no unsettled writer left.

    The reconciliation half of the writer hold, called when a member of that
    tree reaches its terminal and by ``recheck_held_continuations``. Only a row
    still under ITS OWN continuation hold is released (a newer
    Restart/Pause/Stop hold is never overridden), and the release is the
    ordinary selection, which re-checks the writers itself. Once its writers
    are observed settled, the selection's answer is the row's latest Resume
    attempt (``work_stop.resume_asked``): a refusal for any other reason is
    said in its own words in place of the wait's, dated, while the wait and
    this re-check go on; the release that follows replaces it. This is the
    host's own act wherever it runs, a door's Resume of the predecessor
    included (``budget_resume.resume_late_phase``): that door asked about
    another task, so nothing here is its author, lifts an allowance for it or
    goes on past a held call it named (``control_actor.host_control``).
    """
    from ouroboros.control_actor import host_control
    from ouroboros.work_stop import resume_asked, resume_outcome
    from supervisor.events_budget import budget_hold_fact, observe_held_budget_selection, select_held_budget_row

    from supervisor import queue as q
    released: List[str] = []
    with q._queue_lock:
        task_ids = [str(task.get("id") or "") for task in q.PENDING
                    if held_continuation_predecessors([task]) == [tree_root]]
    with host_control():
        for task_id in task_ids:
            observation = observe_held_budget_selection(q, task_id)
            settled = not observation.get("error") and not observation.get("blockers")
            answered = resume_asked(q, task_id, selected_by="reconciliation") if settled else None
            with q._queue_lock:
                task = next((row for row in q.PENDING if str(row.get("id") or "") == task_id), None)
                hold = budget_hold_fact(task)
                if task is None or not writer_wait(task):
                    continue
                outcome = select_held_budget_row(q, task, hold, selected_by="reconciliation", observation=observation)
            if outcome.get("ok"):
                released.append(task_id)
            if answered is not None:  # its writers settled: why it starts, or why it does not
                answered(resume_outcome(outcome))
    return released


def held_continuation_predecessors(pending: Any) -> List[str]:
    """Each predecessor whose Continue still waits under its own writer hold, once."""
    return sorted({str(((task.get("metadata") or {}).get("continuation") or {}).get("predecessor_task_id") or "")
                   for task in list(pending or ()) if isinstance(task, dict)
                   and writer_wait(task)} - {""})


def recheck_held_continuations(*, stop_requested: Any = None) -> List[str]:
    """The body cadence's re-check of every writer-held Continue (``server_maintenance``).

    A writer can settle while no member reaches a terminal: a delegated run's
    custody closes, a model or tool handoff resolves, an owned process exits,
    a late phase ends. A writer hold installs no root fence, so the held rows
    themselves schedule this pass, and each distinct predecessor gets one
    ``release_settled_continuations``: its selection re-checks the writers and
    selects a row once. It runs off the loop and outside every custody and
    store transaction, never from a repair (whose census it would re-enter).
    """
    from supervisor import queue as q

    with q._queue_lock:
        predecessors = held_continuation_predecessors(q.PENDING)
    released: List[str] = []
    for predecessor in predecessors:
        if stop_requested is not None and stop_requested():
            break
        released.extend(release_settled_continuations(predecessor))
    return released
