"""The mind's own context view: inspect the observed send, author a working note, restore sources."""

from __future__ import annotations

import logging
import copy
import hashlib
import json
import threading
from collections import Counter
from pathlib import Path
from typing import List

from ouroboros.tools.registry import ToolEntry

log = logging.getLogger(__name__)
_CANONICAL_OBSERVATION = object()
_OBSERVATION_LOCK = threading.Lock()  # Only exchanges observation and map references.
_NOTE_REQUIRED = ("authored_note_required: inspect the working view, then supply your own working_note "
                  "(keep_unit_ids or keep_last_n select what stays raw, fold_last_n what folds); "
                  "no helper writes it for you.")
# What a fold's own call carries once its note is written (``point_call_at_note``); ``_points_at`` recognises it.
NOTE_POINTER = "[in note {unit_id} above]"


def _own_call_arguments(messages: list, index: int):
    """The arguments of the assistant row at ``index`` when its only call is ``compact_context`` and the next row is
    that call's result, else None."""
    row = messages[index] if 0 <= index < len(messages) else {}
    calls = row.get("tool_calls") if row.get("role") == "assistant" else None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return None
    function, result = calls[0].get("function"), messages[index + 1] if index + 1 < len(messages) else {}
    if (not isinstance(function, dict) or function.get("name") != "compact_context"
            or result.get("role") != "tool" or result.get("tool_call_id") != calls[0].get("id")):
        return None
    try:
        arguments = json.loads(function.get("arguments") or "")
    except (TypeError, ValueError, RecursionError):
        return None
    return arguments if isinstance(arguments, dict) else None


def _points_at(messages: list, index: int, note_id) -> bool:
    arguments = _own_call_arguments(messages, index)
    return bool(note_id) and arguments is not None and arguments.get("working_note") == NOTE_POINTER.format(
        unit_id=note_id)


def point_call_at_note(messages: list, receipt: dict, working_note) -> list:
    """After an applied authored fold, the view with the fold's own call, when it directly follows the note this
    fold wrote and repeats ``working_note``, carrying ``NOTE_POINTER`` in that argument instead; else unchanged."""
    from ouroboros.context_compaction import _capsule_metadata

    if receipt.get("status") != "applied" or not receipt.get("selection_fingerprint"):
        return messages
    note_id = "view:" + receipt["selection_fingerprint"]
    at = next((i for i, row in enumerate(messages) if (_capsule_metadata(row)[1] or {}).get("unit_id") == note_id), None)
    arguments = _own_call_arguments(messages, at + 1) if at is not None else None
    if arguments is None or arguments.get("working_note") != working_note:
        return messages
    row = copy.deepcopy(messages[at + 1])
    arguments["working_note"] = NOTE_POINTER.format(unit_id=note_id)
    row["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return [*messages[:at + 1], row, *messages[at + 2:]]


def _note_meta(row) -> dict | None:
    """A capsule row's own metadata, read without validating or hashing it ({} when unreadable); None otherwise."""
    content = row.get("content") if row.get("role") in {"assistant", "user"} else None
    blocks = [block for block in content if isinstance(block, dict) and "_context_capsule" in block] if isinstance(
        content, list) else []
    return (blocks[0]["_context_capsule"] if isinstance(blocks[0]["_context_capsule"], dict) else {}) if blocks else None


def standing_rows(messages: list, released=()) -> dict:
    """Positions that stand through an ordinary fold, each with the position of the note whose rows they are (None
    for the start). One walk over the rows; nothing is hashed (it runs whenever a round's facts row is written).

    The start: every row between the assignment and the first assistant answer (a note is none), typed or not. Then,
    in order, a note (a capsule that is not a host source view) or a row no fold selects (owner words, a row that is no
    unit) continues the walk; a note directly followed by its own pointing call (``NOTE_POINTER``) stands with every
    row from that call to the next assistant row, the whole input of the request sent after that fold, unless its
    position is in ``released``; anything a full fold would take (a tool unit, prose, a host source view) ends it."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_ROW_KINDS
    from ouroboros.context_compaction import _is_dialogue_prose_row

    first = next((i for i, row in enumerate(messages) if row.get("role") == "user"), None)
    if first is None:
        return {}
    end = next((i for i in range(first + 1, len(messages)) if messages[i].get("role") == "assistant"
                and _note_meta(messages[i]) is None), len(messages))
    standing = {i: None for i in range(first + 1, end) if _note_meta(messages[i]) is None}
    i = first + 1
    while i < len(messages):
        row, meta = messages[i], _note_meta(messages[i])
        if meta is not None and meta.get("retention") != "source_view":
            if i not in released and _points_at(messages, i + 1, meta.get("unit_id")):
                stop = next((j for j in range(i + 2, len(messages)) if messages[j].get("role") == "assistant"),
                            len(messages))
                standing.update(dict.fromkeys(range(i + 1, stop), i))
                i = stop
                continue
        elif not (i in standing or row.get(HOST_CONTEXT_KIND_KEY) in OWNER_ROW_KINDS or meta is None and not (
                row.get("role") == "assistant" and row.get("tool_calls") or _is_dialogue_prose_row(row))):
            break  # what a full fold takes: no later anchor stands
        i += 1
    return standing


def _standing_unit_ids(units, standing: dict, named=()) -> list:
    return [unit.unit_id for unit in units if unit.unit_id not in named
            and all(i in standing for i in range(unit.start, unit.end + 1))]


def record_context_view(ctx, messages, tool_schemas, *, physical_capture=_CANONICAL_OBSERVATION,
                        fit_measurement=None) -> None:
    """Capture a usable turn's canonical source; prospective pricing never calls this.

    Nothing is added to the prompt or cache identity. Inspect pins this one
    observation for a later authored request, so its own tool pair and newer
    owner messages cannot make that request stale by construction.
    Physical vision/provider projections remain in the existing request artifacts.
    """
    from ouroboros.context_compaction import context_reclaim_transcript_sha256

    observed = copy.deepcopy(messages)
    plan, mode = getattr(ctx, "context_fit_plan", None), getattr(ctx, "active_context_mode", "") or "max"
    memory = bool(getattr(getattr(plan, "core", None), "memory_view_json", ""))
    observation = {
        "revision": context_reclaim_transcript_sha256(observed),
        "messages": observed, "tool_schemas": copy.deepcopy(tool_schemas),
        # Continuations retain values, never the live plan. The same core can
        # render this captured selection even after another form was chosen.
        "memory_core_sha256": plan.core_sha256 if memory else "",
        "memory_mode": mode if memory else None,
        "memory_facts": copy.deepcopy(plan.projection(mode).memory_facts) if memory else {},
        "fit_measurement": copy.deepcopy(fit_measurement),
    }
    publish_context_observation(ctx, observation)
    if physical_capture is _CANONICAL_OBSERVATION:
        return  # Existing non-Main actors retain their own observation boundary.
    ref = getattr(physical_capture, "candidate_manifest_ref", None)
    observation["chronicle_lead_rule"] = ref.get("chronicle_lead_rule") if isinstance(ref, dict) else None
    observation.update(exposed_units=[], physical_source_status="unavailable")
    if physical_capture is None or not physical_capture.candidate_manifest_ref:
        return
    from ouroboros.context_compaction import exposed_context_units, exposed_context_units_from_atoms, physical_message_rows
    from ouroboros.observability import read_blob_ref, read_call_manifest_ref

    for root in (getattr(ctx, "budget_drive_root", None), getattr(ctx, "drive_root", None)):
        if root is None:
            continue
        try:
            manifest = read_call_manifest_ref(Path(root), physical_capture.candidate_manifest_ref, task_id=ctx.task_id)
            # The send path records the content identities of the exact prepared bytes before
            # observability redaction (``model_send_seal.persist_physical_candidate``): a unit
            # whose result carries a redacted secret or native content is still recognized.
            # Older manifests fall back to the redacted custody projection they stored. The
            # physical source status names the same attempt manifest either way (the
            # historical-input reader copies its redacted projection); ``exposure_basis``
            # tells which identities decided the exposure.
            atoms = manifest.get("exposure_atoms")
            if isinstance(atoms, dict) and atoms:
                exposed = exposed_context_units_from_atoms(observed, atoms)
                basis = str(manifest.get("exposure_atoms_basis") or "prepared_atoms")
            else:
                payload = read_blob_ref(Path(root), manifest["full_payload_ref"], task_id=ctx.task_id)
                physical_messages = physical_message_rows(payload)
                if physical_messages is None:
                    continue
                exposed, basis = exposed_context_units(observed, physical_messages), "redacted_projection"
            observation.update(
                exposed_units=list(exposed), exposure_basis=basis,
                physical_source_status="observed_projection", physical_attempt_id=physical_capture.attempt_id,
                physical_source_ref=physical_capture.candidate_manifest_ref,
            )
            return
        except (OSError, ValueError, KeyError, TypeError):
            log.debug("Physical context source unavailable", exc_info=True)


def _producing_view(ctx):
    """The send that produced this call: a live call's queued request can outlive its connection."""
    from ouroboros.tool_call_log import current_tool_call_id

    issued = (getattr(ctx, "_issued_views", None) or {}).get(current_tool_call_id())
    return issued if issued is not None else getattr(ctx, "_last_context_observation", None)


def residue_by_kind(messages: list) -> dict:
    """What a view keeps in dialogue units, by kind, protected dialogue apart (host estimates)."""
    from ouroboros.context_compaction import context_units, owner_protected_unit_ids, unit_kind

    units = context_units(messages, scope="dialogue")
    protected, residue = owner_protected_unit_ids(messages, units), {}
    for unit in units:
        row = residue.setdefault("protected" if unit.unit_id in protected else unit_kind(messages, unit), [0, 0])
        row[0], row[1] = row[0] + 1, row[1] + unit.context_size_tokens
    return {kind: {"units": units_, "tokens": tokens} for kind, (units_, tokens) in sorted(residue.items())}


def view_notes(messages: list, *, own: bool = False) -> list:
    """The working-view notes in ``messages``, in transcript order: ``{id, written, first_line}`` for every capsule
    except a host source view, under the unit id inspect shows and ``replace_note_ids`` takes; ``own``: only the notes
    the mind authored (``authorship`` actor, as the round facts count its own notes), never a helper's or one whose
    author is not recorded. ``written`` is ``context_compaction.note_written``: the note's own write time, or "not
    recorded" for a note older than that. The one reader of notes for a fold receipt and an arrival line."""
    from ouroboros.context_compaction import _capsule_metadata, _first_line, _unit_from_slice, note_written

    notes = []
    for index, message in enumerate(messages):
        meta = _capsule_metadata(message)[1]
        if meta is None or meta.get("retention") == "source_view" or own and meta.get("authorship") != "actor":
            continue
        unit = _unit_from_slice(messages, index, index, trace_refs_by_tool_call_id={}, capsule_meta=meta)
        if unit is not None:
            notes.append({"id": unit.unit_id, "written": note_written(meta), "first_line": _first_line(message)})
    return notes


def _body_source(messages: list, unit) -> str:
    """What a removed unit was, as inspect names its kind; a tool unit adds each call's name and argument head."""
    from ouroboros.context_compaction import _BODY_ARGUMENT_CHARS, unit_kind

    kind = unit_kind(messages, unit)
    if kind != "tool":
        return kind
    calls = []
    for call in messages[unit.start].get("tool_calls") or ():
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        arguments = str(function.get("arguments") or "")
        calls.append(f"{function.get('name') or '?'} " + (arguments if len(arguments) <= _BODY_ARGUMENT_CHARS
                                                           else arguments[:_BODY_ARGUMENT_CHARS] + "…"))
    return "tool " + "; ".join(call.rstrip() for call in calls)


# What a fold receipt tells its author, beside its counts: every key a fact, none an instruction.
FOLD_FACT_KEYS = ("replaced_notes", "notes_kept", "restore_checkpoint", "bodies_removed")


def fold_receipt_facts(observed: list, candidate: list, receipt: dict) -> dict:
    """An applied view's facts for the receipt its author reads, and no instruction: the notes in view after it
    (``notes_kept``, ``view_notes``) and each unit of the observed view whose body left it (``bodies_removed``:
    ``{source, address}``), restorable from ``restore_checkpoint`` by the address. Notes the author replaced stay
    named by ``replaced_notes``; a body whose unit no longer matches its id and hash is not named. An unapplied
    receipt changed nothing and has none."""
    from ouroboros.context_compaction import _unit_from_slice

    if receipt.get("status") != "applied":
        return {}
    facts = {"notes_kept": view_notes(candidate)} if candidate else {}
    replaced = {note.get("unit_id") for note in (receipt.get("replaced_notes") or {}).get("notes") or ()}
    bodies = []
    for unit_id in receipt.get("selected_unit_ids") or ():
        try:
            _, start, end, _ = str(unit_id).split(":")
            unit = _unit_from_slice(observed, int(start), int(end), trace_refs_by_tool_call_id={})
        except (ValueError, IndexError):
            continue
        if unit is not None and unit.unit_id == unit_id and unit_id not in replaced:
            bodies.append({"source": _body_source(observed, unit),
                           "address": {"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256}})
    checkpoint = receipt.get("checkpoint_ref")
    if bodies and isinstance(checkpoint, dict):
        facts.update(restore_checkpoint={key: checkpoint[key] for key in ("kind", "root", "path", "size", "sha256")
                                         if key in checkpoint}, bodies_removed=bodies)
    return {key: value for key, value in facts.items() if value}


def _ordinary_frame(observed):
    facts = observed.get("memory_facts") or {}
    room = json.dumps(facts.get("max_room"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"core_sha256": observed.get("memory_core_sha256"), "forms_version": (facts.get("floor") or {}).get("forms_version"),
            "max_room_sha256": hashlib.sha256(room.encode("utf-8")).hexdigest()}


def _matches_frame(saved, frame):
    return saved and frame["core_sha256"] and all(saved["identity"].get(k) == v for k, v in frame.items())


def _last_map(observed):
    return observed.get("ordinary_memory") or observed.get("previous_ordinary_memory")


def publish_context_observation(ctx, observation, connection_ctx=None):
    """Install a completed frame, taking the last inspected map at publication time."""
    with _OBSERVATION_LOCK:
        saved = _last_map(getattr(ctx, "_last_context_observation", None) or {})
        if saved:
            key = "ordinary_memory" if _matches_frame(saved, _ordinary_frame(observation)) else "previous_ordinary_memory"
            observation[key] = saved
        ctx._last_context_observation = observation
        if connection_ctx is not None:
            connection_ctx._last_context_observation = observation


def _ordinary_memory(ctx, observed, *, continuation_page):
    """Retain one on-demand metadata source, independent of transcript pagination and later tool pairs."""
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
    from ouroboros.memory_view import snapshot_from_json
    from ouroboros.memory_form_facts import ordinary_composition
    from ouroboros.memory_floor import FORMS_VERSION

    plan = getattr(ctx, "context_fit_plan", None)
    raw = getattr(getattr(plan, "core", None), "memory_view_json", "")
    frame = _ordinary_frame(observed)
    with _OBSERVATION_LOCK:
        current = getattr(ctx, "_last_context_observation", None) or {}
        latest_map = _last_map(current)
        saved = latest_map if _matches_frame(latest_map, frame) else _last_map(observed)

    def publish(result):
        with _OBSERVATION_LOCK:
            latest = getattr(ctx, "_last_context_observation", None)
            observed.pop("previous_ordinary_memory", None)
            observed["ordinary_memory"] = result
            if latest is not None and latest is not observed:
                if _matches_frame(result, _ordinary_frame(latest)):
                    latest.pop("previous_ordinary_memory", None)
                    latest["ordinary_memory"] = result
                elif not latest.get("ordinary_memory") and _last_map(latest) == saved:
                    latest["previous_ordinary_memory"] = result

    def unavailable(reason):
        reply = {"status": "unavailable", "reason": reason, "observed_frame": frame}
        if saved:
            which = "retained" if _matches_frame(saved, frame) else "previous"
            reply.update({which + "_source": saved["source"], which + "_identity": saved["identity"]})
        return reply
    if not raw or not observed.get("memory_core_sha256"):
        return unavailable("retained_snapshot_missing")
    if observed["memory_core_sha256"] != plan.core_sha256:
        return unavailable("different_core")
    if ((observed.get("memory_facts") or {}).get("floor") or {}).get("forms_version") != FORMS_VERSION:
        return unavailable("different_forms")
    if not getattr(ctx, "task_id", "") or not getattr(ctx, "drive_root", None):
        return unavailable("actor_source_store_missing")
    max_room = (observed.get("memory_facts") or {}).get("max_room")
    snapshot_sha = (saved["identity"]["snapshot_sha256"] if saved and
                    saved["identity"]["core_sha256"] == plan.core_sha256 else
                    hashlib.sha256(raw.encode("utf-8")).hexdigest())
    identity = {**frame, "snapshot_sha256": snapshot_sha, "map_version": 2}
    if saved and all(saved["identity"].get(key) == value for key, value in identity.items()):
        if not continuation_page:
            try:
                read_actor_source_bytes(Path(ctx.drive_root), ctx.task_id, saved["source"])
            except (OSError, ValueError):
                return unavailable("retained_metadata_source_unreadable")
        publish(saved)
        return {"status": "retained_unverified" if continuation_page else "available", **saved}
    try:
        snapshot = snapshot_from_json(raw)
        composition = ordinary_composition(snapshot)
        totals = {name: {"json_chars": block["json_chars"], "estimated_tokens": block["estimated_tokens"],
                         "parts": len(block["parts"])} for name, block in composition["blocks"].items()}
        largest = sorted(({"block": name, "part": index, "kind": part["kind"], "form": part["form"],
                           "json_chars": part["json_chars"]}
                          for name, block in composition["blocks"].items() for index, part in enumerate(block["parts"])),
                         key=lambda part: -part["json_chars"])[:3]
        measured = {"story": totals["story"]["estimated_tokens"], "room": totals["room"]["estimated_tokens"],
                    "room_id": snapshot.spec.room_id, "levels": snapshot.spec.work_room_levels}
        recorded = {key: max_room.get(key) for key in measured} if isinstance(max_room, dict) else None
        comparison = {"status": "same_raw_measure" if recorded == measured else
                      "different_raw_measure" if recorded is not None else "unavailable",
                      "recorded": recorded, "measured": measured,
                      "allowance": max_room.get("allowance") if isinstance(max_room, dict) else None,
                      "basis": "Raw story/room and room identity only; calibrated need is not compared."}
        identity.update(role=snapshot.spec.role, room_id=snapshot.spec.room_id)
        document = {"schema_version": identity["map_version"], "identity": identity, "captured_max_room": max_room,
                    "max_room_comparison": comparison, **composition}
        data = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        roots = dict.fromkeys((Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root).resolve(),
                               Path(ctx.drive_root).resolve()))
        for root in roots:
            source = store_actor_source_bytes(root, ctx.task_id, category="context_checkpoints",
                                             source_id="ordinary-memory", data=data, extension="json")
        read_actor_source_bytes(Path(ctx.drive_root), ctx.task_id, source)
        result = {"identity": identity, "blocks": totals, "largest": largest, "estimated_tokens": composition["estimated_tokens"],
                  "max_room_comparison": comparison, "source": source,
                  "basis": "ordinary; additive escaped chars, two block roundings, no floor note or savings estimate"}
        publish(result)
        return {"status": "different_source" if saved else "available", **result,
                **({"previous_source_sha256": saved["source"]["sha256"], "previous_source": saved["source"],
                    "previous_identity": saved["identity"]} if saved else {})}
    except (OSError, ValueError, KeyError, TypeError):
        log.debug("Ordinary memory metadata could not be retained", exc_info=True)
        return unavailable("metadata_source_unavailable")


def _inspect(ctx, observed: dict, *, offset: int) -> str:
    """A bounded inventory of the observed view: counts and sizes by kind, then one page of
    units from ``offset`` within the tool's producer page size (``tool_result_limit``), naming
    ``next_offset`` while units remain. The reader rebuilds a restore reference from the one
    ``restore_checkpoint`` and a unit's id and hash instead of receiving it repeated per unit.
    The fields every page would share come once, on page one (``offset`` 0); a later page carries
    its units, the paging, the view revision, the ``restore_checkpoint`` of the observation it lists (a newer
    observation has its own, so a listed unit always restores from its page) and the retained ordinary-memory
    map (``_ordinary_memory``'s continuation read). A page holds at least one unit, so paging always advances:
    only page one's shared fields can leave no room for it."""
    from ouroboros.context_compaction import (_capsule_metadata, _first_line, context_units, note_written,
                                              owner_protected_unit_ids, unit_kind)
    from ouroboros.review_history_view import attachment_transfer_options, review_note_options
    from ouroboros.tool_capabilities import tool_result_limit

    messages = observed["messages"]
    units = context_units(messages, scope="dialogue")
    protected = owner_protected_unit_ids(messages, units)
    checkpoint = observed.get("checkpoint_ref")
    if not checkpoint and getattr(ctx, "task_id", "") and getattr(ctx, "drive_root", None):
        from ouroboros.artifacts import store_actor_source_bytes

        try:
            roots = dict.fromkeys((Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root).resolve(),
                                   Path(ctx.drive_root).resolve()))
            raw = json.dumps({"messages": messages,
                "observed_view_revision": observed["revision"], "selection_fingerprint": observed["revision"],
                "selected_unit_ids": [unit.unit_id for unit in units]}, ensure_ascii=False).encode("utf-8")
            for root in roots:
                checkpoint = store_actor_source_bytes(root, ctx.task_id, category="context_checkpoints",
                    source_id="view", data=raw, extension="json")
            observed["checkpoint_ref"] = checkpoint
        except (OSError, ValueError):
            checkpoint = None
            log.debug("Inspected context source could not be retained", exc_info=True)
    # Every restore ref remains self-contained. The reader needs the stable
    # content-addressed path/size/hash, not a repeated read_file invocation.
    # Keep the full reader hint on the pinned observation, outside this reply.
    restore_checkpoint = ({key: checkpoint[key] for key in ("kind", "root", "path", "size", "sha256")}
                          if checkpoint else None)
    transfers = attachment_transfer_options(ctx) if getattr(ctx, "task_id", "") else {}
    decisions = review_note_options(ctx) if getattr(ctx, "task_id", "") else {}
    observed["review_decisions"] = copy.deepcopy(decisions)
    exposed = ({(ref.get("unit_id"), ref.get("raw_sha256")) for ref in observed["exposed_units"]}
               if "exposed_units" in observed else None)
    rows, counts, tokens = [], Counter(), Counter()
    standing, unit_at = standing_rows(messages), {unit.start: unit.unit_id for unit in units}
    for unit in units:
        kind = unit_kind(messages, unit)
        counts[kind] += 1
        tokens[kind] += unit.context_size_tokens
        stands = unit.start in standing and all(i in standing for i in range(unit.start, unit.end + 1))
        row = {"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256, "kind": kind,
               "tokens": unit.context_size_tokens, "eligible": unit.unit_id not in protected and not stands,
               **({"stands": "start of the request" if standing[unit.start] is None
                   else f"own rows of note {unit_at.get(standing[unit.start])}"} if stands else {})}
        if kind == "capsule":
            meta = _capsule_metadata(messages[unit.start])[1]
            row.update(authorship=meta.get("authorship") or "unrecorded", first_line=_first_line(messages[unit.start]),
                       **({} if meta.get("retention") == "source_view" else {"written": note_written(meta)}))
        if exposed is not None:
            row["exposed"] = (unit.unit_id, unit.raw_sha256) in exposed
        rows.append(row)
    from ouroboros.memory_floor import view_inventory
    plan = getattr(ctx, "context_fit_plan", None)
    memory_available = bool(observed.get("memory_core_sha256")) and (
        observed["memory_core_sha256"] == getattr(plan, "core_sha256", None))
    memory_rows = view_inventory(plan if memory_available else None, getattr(ctx, "active_context_mode", "") or "max",
                                 memory_facts=observed.get("memory_facts"))
    rows.extend(memory_rows)
    counts['memory'] += 0
    tokens['memory'] += 0
    for row in memory_rows:
        counts[row['kind']] += 1
        tokens[row['kind']] += row['tokens']
    start = max(0, int(offset or 0))
    measurement = observed.get('fit_measurement')
    if measurement is None and not start:
        from ouroboros.loop_model_call import _measure_main_context_view
        measurement = _measure_main_context_view(plan, messages, observed['tool_schemas'],
            getattr(ctx, 'active_context_mode', '') or 'max', getattr(ctx, 'active_effort', ''), 'inspect')
    body = {"view_revision": observed["revision"], "restore_checkpoint": restore_checkpoint,
            "ordinary_memory": _ordinary_memory(ctx, observed, continuation_page=True)} if start else {
        "fit_measurement": measurement,
        "work_room": observed.get('work_room'),
        "resident_memory": (observed.get('memory_facts') or {}).get('blocks', {}),
        "selected_memory_mode": observed.get("memory_mode") if observed.get("memory_core_sha256") else None,
        "ordinary_memory": _ordinary_memory(ctx, observed, continuation_page=start > 0),
        **({"review_attachment_transfer": transfers} if transfers else {}),
        **({"review_decisions": decisions} if decisions else {}),
        "view_revision": observed["revision"],
        "restore_checkpoint": restore_checkpoint,
        "memory_view_status": "observed_snapshot" if memory_available else "unavailable",
        "memory_re_read_price": "Changing a memory form rewrites the system prefix; later cached input may need re-reading.",
        "inventory": {"units": len(rows), "eligible": sum(row["eligible"] for row in rows),
                      "tokens": sum(tokens.values()),
                      "by_kind": {kind: {"units": counts[kind], "tokens": tokens[kind]} for kind in sorted(counts)}},
        "schema_names": [s["function"]["name"] for s in observed["tool_schemas"]],
        "rule": ("This revision names the observed messages; schemas are listed separately. Units are complete tool "
                 "units, earlier records, your own completed replies (kind assistant) and completed prose rows (kind user; "
                 "eligibility distinguishes governing owner words); a unit with eligible=false carries protected dialogue "
                 "and stays whole, or stands (stands says why) until an explicit keep_unit_ids list omits it or its note is replaced. units is one page in transcript order (offset, total; next_offset while more remain: "
                 "call inspect again with that offset; a later page repeats only view_revision, restore_checkpoint and "
                 "ordinary_memory beside its units and the paging). Select complete unit IDs to keep and write a local working_note; "
                 "a restore_unit_refs entry is {checkpoint_ref: restore_checkpoint, unit_id, raw_sha256}. Earlier notes "
                 "stay unless replace_note_ids names them (capsule rows show authorship and first_line, a note's also "
                 "when it was written). Restored "
                 "sources and host emergency records follow keep_unit_ids; keep_last_n keeps all unnamed capsules. "
                 "view_address selects listed memory forms from this snapshot; a note's restore_view_address restores it whole. Core, governing words, the assignment "
                 "and newer owner/tool messages stay whole."),
    }

    def render(page: list, next_offset) -> str:
        return json.dumps({**body, "offset": start, "total": len(rows), "units": page,
                           **({"next_offset": next_offset} if next_offset is not None else {})},
                          ensure_ascii=False, separators=(",", ":"))

    limit, page = tool_result_limit("compact_context"), []
    for index in range(start, len(rows)):
        candidate = [*page, rows[index]]
        if page and len(render(candidate, index + 1)) > limit:
            break
        page = candidate
    end = start + len(page)
    return render(page, end if end < len(rows) else None)


def _compact_context(ctx, keep_last_n: int | None = None, *, inspect: bool = False, offset: int = 0,
                     expected_view_revision: str = "", working_note: str | None = None,
                     keep_unit_ids: List[str] | None = None, replace_note_ids: List[str] | None = None,
                     restore_unit_refs: List[dict] | None = None,
                     schema_names: List[str] | None = None, review_transfers: List[dict] | None = None,
                     review_notes: List[dict] | None = None, view_address: List[str] | None = None,
                     fold_last_n: int | None = None, **kwargs) -> str:
    """Inspect or queue a replacement through the existing context materializer.

    The authored view (inspect / working_note / restore) addresses the dialogue scope:
    complete tool units, capsules, the actor's own completed replies and host prose rows,
    all under the same positional, hash-bound unit ids. Rows carrying the owner's typed
    words are listed but never eligible. Earlier notes leave only when ``replace_note_ids``
    names them; host source views and emergency records follow ``keep_unit_ids``.
    ``keep_last_n`` keeps every unnamed capsule; ``fold_last_n`` folds from its N-th most recent
    tool unit onward. Without an explicit non-empty ``keep_unit_ids`` list the rows that stand
    (``standing_rows``) stay; a replaced note releases its own. Standalone restoration
    keeps current units/schemas and authors no note. A count-only request (no note, no
    restore) is refused: no helper writes the note.
    """
    from ouroboros.context_compaction import context_units, unit_kind
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    if inspect:
        observed = _producing_view(ctx)
        if not isinstance(observed, dict):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
                text="Context view unavailable: this actor has no recorded model-send observation yet."))
        ctx._inspected_context_view = observed
        return _inspect(ctx, observed, offset=offset)
    if working_note is None and restore_unit_refs:
        if (any(value is not None for value in (keep_last_n, keep_unit_ids, replace_note_ids, schema_names, fold_last_n))
                or review_transfers or review_notes):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text="Standalone restore_unit_refs keeps current units, schemas and review accounts. "
                     "Omit other selectors, or supply working_note to combine restoration with an authored view."))
        working_note = ""
    if working_note is None and view_address:
        working_note = ""
    if working_note is None:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=_NOTE_REQUIRED))
    # This invocation is a response to the actor's recorded physical send.
    # The host already owns that causal binding; echoing its hash is only
    # needed when the actor explicitly selects an earlier inspected view.
    observed = getattr(ctx, "_inspected_context_view", None) if expected_view_revision else _producing_view(ctx)
    if expected_view_revision and (not isinstance(observed, dict) or observed.get("revision") != expected_view_revision):
        last = getattr(ctx, "_last_context_observation", None)
        if isinstance(last, dict) and last.get("revision") == expected_view_revision:
            observed = last
    if not isinstance(observed, dict) or expected_view_revision and observed.get("revision") != expected_view_revision:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="Context view mismatch: call compact_context(inspect=true) and use its view_revision. Current context is unchanged."))
    if (not isinstance(working_note, str)
            or any(values is not None and (not isinstance(values, list)
                   or not all(isinstance(v, str) and v for v in values))
                   for values in (keep_unit_ids, replace_note_ids, schema_names, view_address))
            or review_notes is not None and not isinstance(review_notes, list)
            or review_transfers is not None and not isinstance(review_transfers, list)
            or restore_unit_refs is not None and (not isinstance(restore_unit_refs, list)
                or not all(isinstance(ref, dict) for ref in restore_unit_refs))):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="Invalid context view request: use a prose working_note, arrays of exact unit/schema names and checkpoint reference objects."))
    from ouroboros.review_history_view import resolve_observed_review_notes
    review_notes = resolve_observed_review_notes(review_notes or [], observed)
    if view_address:
        from ouroboros.memory_floor import select_view
        try:
            plan = getattr(ctx, "context_fit_plan", None)
            if plan is None or observed.get("memory_core_sha256") != plan.core_sha256:
                raise ValueError("memory snapshot changed; inspect the current view")
            select_view(plan, getattr(ctx, "active_context_mode", "") or "max", view_address,
                        owner_note=bool(working_note.strip()))
        except ValueError as exc:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=str(exc)))
    named = list(replace_note_ids or ())
    if named:
        notes = {unit.unit_id for unit in context_units(observed["messages"], scope="dialogue")
                 if unit_kind(observed["messages"], unit) == "capsule"}
        conflict = sorted(set(named) & set(keep_unit_ids or ()))
        unknown = [unit_id for unit_id in named if unit_id not in notes]
        if conflict or unknown:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
                f"keep_unit_ids and replace_note_ids both name {conflict}: a note is either kept or replaced."
                if conflict else f"replace_note_ids names earlier notes of the observed view (inspect kind capsule); "
                f"these are not: {unknown}.") + " Current context is unchanged."))
    if fold_last_n is not None and (isinstance(fold_last_n, bool) or not isinstance(fold_last_n, int)
                                    or fold_last_n < 1 or keep_unit_ids is not None or keep_last_n is not None):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "fold_last_n is a whole number of at least 1." if keep_unit_ids is None and keep_last_n is None else
            f"fold_last_n and {'keep_unit_ids' if keep_unit_ids is not None else 'keep_last_n'} both select: "
            "fold_last_n folds the newest tool units, the others name what stays raw; use one.")
            + " Current context is unchanged."))
    rows, literal = observed["messages"], bool(keep_unit_ids)  # an explicit non-empty list is the mind's whole choice
    units = context_units(rows, scope="dialogue")
    standing = standing_rows(rows, released={unit.start for unit in units if unit.unit_id in named})
    stand = _standing_unit_ids(units, standing, named)
    if fold_last_n is not None:
        # The newest N completed tool units that do not stand fold, with everything from the first of them
        # onward (prose and host rows between and after them; owner words and notes stay as always).
        tool_units = [unit for unit in units if unit_kind(rows, unit) == "tool" and unit.unit_id not in stand]
        if not tool_units:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
                "fold_last_n found no completed tool unit to fold in the observed view. Current context is unchanged.")))
        anchor = tool_units[-min(fold_last_n, len(tool_units))].start
        keep_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in named and unit.start < anchor]
    elif keep_unit_ids is None and keep_last_n is not None:
        # keep_last_n counts completed tool units; everything from the N-th most recent
        # tool unit onward (prose rows between and after them included) stays raw. Earlier
        # capsules (notes and host sources) stay wherever they sit unless named.
        count = max(2, min(keep_last_n, 20))
        tool_units = [unit for unit in units if unit_kind(rows, unit) == "tool"]
        anchor = tool_units[-count].start if len(tool_units) >= count else 0
        keep_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in named
                         and (unit.start >= anchor or unit_kind(rows, unit) == "capsule")]
    elif keep_unit_ids is None and named:
        # Omitted selectors keep everything, except the own rows of a note this one replaces.
        released = set(standing_rows(rows)) - set(standing)
        keep_unit_ids = [unit.unit_id for unit in units if unit.unit_id not in named
                         and not any(i in released for i in range(unit.start, unit.end + 1))]
        keep_unit_ids = None if len(keep_unit_ids) + len(named) == len(units) else keep_unit_ids
    if keep_unit_ids is not None and not literal:
        keep_unit_ids = [*keep_unit_ids, *(unit_id for unit_id in stand if unit_id not in keep_unit_ids)]
    from ouroboros.utils import utc_now_iso

    ctx._pending_compaction = {
        "observed": observed, "working_note": working_note, "review_transfers": list(review_transfers or []),
        "review_notes": list(review_notes or []),
        "view_address": tuple(view_address or ()),
        "expected_view_revision": observed["revision"],
        "keep_unit_ids": None if keep_unit_ids is None else tuple(keep_unit_ids),
        "replace_note_ids": tuple(named),
        "restore_unit_refs": tuple(restore_unit_refs or ()),
        "schema_names": None if schema_names is None else tuple(schema_names),
        # When the mind wrote its note: the boundary that applies it may come later (a pause keeps it queued).
        "written_at": utc_now_iso(timespec="seconds"),
    }
    return "Working view requested. The next complete tool boundary preserves exact sources and checks the full candidate before applying it; the resulting receipt reports actual changes."


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="compact_context",
            schema={
                "name": "compact_context",
                "description": (
                    "Fold context under your working_note; no helper summarizes for you. Only view_address or "
                    "restore_unit_refs may be used without a note. Core, assignment, owner words, typed owner-directed "
                    "dialogue and newer tail stay whole. Each older tool call and contiguous results is atomic; exact "
                    "bytes are checkpointed with CAS provenance before replacement, so raw evidence remains retrievable. "
                    "Notes stay unless replace_note_ids names them; restored sources and emergency records follow "
                    "the keep selection; replacement occupies the first replaced unit. "
                    "Batch usefully, shorter than sources: rewriting loses signed reasoning from the earliest change. "
                    "How much is re-read after a fold depends on the nearest cache boundary before its first change. "
                    "Round facts report the measured re-read. The request's start rows and a full fold's own rows "
                    "stay until its note is replaced. "
                    "Native review ignores schema_names, review_notes and review_transfers; its inspection reports "
                    "window evidence, serialized characters and previous input, never current free tokens."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "view_address": {"type": "array", "items": {"type": "string"}, "description": "Exact inspected memory forms; restore_view_address restores a note whole. Applied at the next tool boundary from the same snapshot, preserving other choices and arrivals; receipt measures the complete result and prefix re-reading."},
                        "inspect": {"type": "boolean", "description": "Pin view_revision; return counts/tokens by kind, one transcript-ordered page of unit IDs/eligibility, restore_checkpoint and schema names; no mutation."},
                        "offset": {"type": "integer", "minimum": 0, "description": "Inspection page's zero-based start; next_offset names remaining units."},
                        "expected_view_revision": {"type": "string", "description": "Exact inspected revision; omitted binds the model-send view producing this call, so inspection is optional."},
                        "working_note": {"type": "string", "description": "For a future reader without sources: keep situation/purpose, who decided what, their status within those sources, scope, changes, corrections and unresolved work; label inference and gaps, invent no detail. Later grants/cancellations supersede only their scope, not unaffected branches; proposals are not approvals, and superseded restrictions do not survive scoped approval. Concise, no fixed template; required for folds."},
                        "keep_unit_ids": {"type": "array", "items": {"type": "string"}, "description": "Exact complete units to keep raw, overriding keep_last_n; empty keeps only rows that stand, omitting all selectors selects all."},
                        "replace_note_ids": {"type": "array", "items": {"type": "string"}, "description": "Capsule IDs this working_note replaces. The new note keeps their source addresses, the receipt names their first lines; an ID also in keep_unit_ids is refused."},
                        "restore_unit_refs": {"type": "array", "items": {"type": "object", "properties": {
                            "checkpoint_ref": {"type": "object"}, "unit_id": {"type": "string"}, "raw_sha256": {"type": "string"}},
                            "required": ["checkpoint_ref", "unit_id", "raw_sha256"]},
                            "description": "Restore sources, never protocol replay; checkpoint_ref is restore_checkpoint (emergency records list members separately). Main allows this alone; native review requires working_note, possibly empty; combining selectors requires a note."},
                        "schema_names": {"type": "array", "items": {"type": "string"}, "description": "Nano schema residency, never permissions; Low/Max keep the full permitted envelope."},
                        "review_notes": {"type": "array", "items": {"type": "object", "properties": {
                            "bound_decision": {"anyOf": [{"type": "object"}, {"type": "string"}]},
                            "bound_decisions": {"type": "array", "items": {"anyOf": [{"type": "object"}, {"type": "string"}]}},
                            "remark": {"type": "string"}, "reason": {"type": "string"}},
                            "required": ["remark", "reason"], "additionalProperties": False},
                            "description": "With working_note, exact observed bindings/decision_ref (not positional IDs) annotate one decision or group closed history, preserving verdicts, current contracts, open/deferred/pending facts and gaps; stale/missing bindings stay full with reasons."},
                        "review_transfers": {"type": "array", "items": {"type": "object", "properties": {
                            "source": {"type": "object"}, "operative_spec_sha256": {"type": "string"},
                            "decision_ids": {"type": "array", "items": {"type": "string"}}},
                            "required": ["source", "operative_spec_sha256", "decision_ids"]},
                            "description": "With working_note, declare attachment decisions/reasons transferred to named current plan decisions using inspected source/spec identities; no completeness is inferred, and absent/stale transfer keeps the attachment whole."},
                        "keep_last_n": {
                            "type": "integer",
                            "description": "With working_note and no keep_unit_ids: recent completed tool units to keep raw (2-20); earlier notes stay.",
                        },
                        "fold_last_n": {"type": "integer", "minimum": 1, "description": "With working_note, alone: fold the newest N completed tool units and everything after the first of them."},
                    },
                    "required": [],
                },
            },
            handler=_compact_context,
            timeout_sec=15,
        ),
    ]
