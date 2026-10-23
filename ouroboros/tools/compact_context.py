"""The mind's own context view: inspect the observed send, author a working note, restore sources."""

from __future__ import annotations

import logging
import copy
import json
from collections import Counter
from pathlib import Path
from typing import List

from ouroboros.tools.registry import ToolEntry

log = logging.getLogger(__name__)
_CANONICAL_OBSERVATION = object()
_NOTE_REQUIRED = ("authored_note_required: inspect the working view, then supply your own working_note "
                  "(keep_unit_ids or keep_last_n select what stays raw); no helper writes it for you.")


def record_context_view(ctx, messages, tool_schemas, *, physical_capture=_CANONICAL_OBSERVATION) -> None:
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
    ctx._last_context_observation = {
        "revision": context_reclaim_transcript_sha256(observed),
        "messages": observed, "tool_schemas": copy.deepcopy(tool_schemas),
        # Continuations retain values, never the live plan. The same core can
        # render this captured selection even after another form was chosen.
        "memory_core_sha256": plan.core_sha256 if memory else "",
        "memory_facts": {"floor": copy.deepcopy(plan.projection(mode).memory_facts.get("floor") or {})} if memory else {},
    }
    if physical_capture is _CANONICAL_OBSERVATION:
        return  # Existing non-Main actors retain their own observation boundary.
    observation = ctx._last_context_observation
    observation.update(exposed_units=[], physical_source_status="unavailable")
    if physical_capture is None or not physical_capture.candidate_manifest_ref:
        return
    from ouroboros.context_compaction import exposed_context_units, exposed_context_units_from_atoms
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
                payload = read_blob_ref(Path(root), manifest["full_payload_ref"])
                physical_messages = payload.get("messages")
                if not isinstance(physical_messages, list):
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


def _inspect(ctx, observed: dict, *, offset: int) -> str:
    """A bounded inventory of the observed view: counts and sizes by kind, then one page of
    units from ``offset`` within the tool's producer page size (``tool_result_limit``), naming
    ``next_offset`` while units remain. The reader rebuilds a restore reference from the one
    ``restore_checkpoint`` and a unit's id and hash instead of receiving it repeated per unit."""
    from ouroboros.context_compaction import context_units, owner_protected_unit_ids, unit_kind
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
    for unit in units:
        kind = unit_kind(messages, unit)
        counts[kind] += 1
        tokens[kind] += unit.context_size_tokens
        row = {"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256, "kind": kind,
               "tokens": unit.context_size_tokens, "eligible": unit.unit_id not in protected}
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
    counts["memory"] = len(memory_rows)
    tokens["memory"] = sum(row["tokens"] for row in memory_rows)
    body = {
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
                 "and stays whole. units is one page in transcript order (offset, total; next_offset while more remain: "
                 "call inspect again with that offset). Select complete unit IDs to keep and write a local working_note; "
                 "a restore_unit_refs entry is {checkpoint_ref: restore_checkpoint, unit_id, raw_sha256}. Earlier notes "
                 "stay where they are unless selected for replacement; select several to merge their account and sources. "
                 "view_address selects listed memory forms from this snapshot; a note's restore_view_address restores it whole. Core, governing words, the assignment "
                 "and newer owner/tool messages stay whole."),
    }
    start = max(0, int(offset or 0))

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
                     keep_unit_ids: List[str] | None = None, restore_unit_refs: List[dict] | None = None,
                     schema_names: List[str] | None = None, review_transfers: List[dict] | None = None,
                     review_notes: List[dict] | None = None, view_address: List[str] | None = None, **kwargs) -> str:
    """Inspect or queue a replacement through the existing context materializer.

    The authored view (inspect / working_note / restore) addresses the dialogue scope:
    complete tool units, capsules, the actor's own completed replies and host prose rows,
    all under the same positional, hash-bound unit ids. Rows carrying the owner's typed
    words are listed but never eligible. Standalone restoration keeps current units/schemas
    and authors no note. A count-only request (no note, no restore) is refused: no helper
    writes the note.
    """
    from ouroboros.context_compaction import context_units, unit_kind
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    if inspect:
        observed = getattr(ctx, "_last_context_observation", None)
        if not isinstance(observed, dict):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
                text="Context view unavailable: this actor has no recorded model-send observation yet."))
        ctx._inspected_context_view = observed
        return _inspect(ctx, observed, offset=offset)
    if working_note is None and restore_unit_refs:
        if (any(value is not None for value in (keep_last_n, keep_unit_ids, schema_names))
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
    observed = (getattr(ctx, "_inspected_context_view", None) if expected_view_revision
                else getattr(ctx, "_last_context_observation", None))
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
                   for values in (keep_unit_ids, schema_names, view_address))
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
    if keep_unit_ids is None and keep_last_n is not None:
        # keep_last_n counts completed tool units; everything from the N-th most recent
        # tool unit onward (prose rows between and after them included) stays raw.
        count = max(2, min(keep_last_n, 20))
        units = context_units(observed["messages"], scope="dialogue")
        tool_units = [unit for unit in units if unit_kind(observed["messages"], unit) == "tool"]
        anchor = tool_units[-count].start if len(tool_units) >= count else 0
        keep_unit_ids = [unit.unit_id for unit in units if unit.start >= anchor
                         or unit_kind(observed["messages"], unit) == "capsule"]
    ctx._pending_compaction = {
        "observed": observed, "working_note": working_note, "review_transfers": list(review_transfers or []),
        "review_notes": list(review_notes or []),
        "view_address": tuple(view_address or ()),
        "expected_view_revision": observed["revision"],
        "keep_unit_ids": None if keep_unit_ids is None else tuple(keep_unit_ids),
        "restore_unit_refs": tuple(restore_unit_refs or ()),
        "schema_names": None if schema_names is None else tuple(schema_names),
    }
    return "Working view requested. The next complete tool boundary preserves exact sources and checks the full candidate before applying it; the resulting receipt reports actual changes."


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="compact_context",
            schema={
                "name": "compact_context",
                "description": (
                    "Fold complete-input context under your own authored working_note (inspect optional), or "
                    "restore exact sources with restore_unit_refs alone. No helper summarizes for you: a request "
                    "without working_note is refused unless it only selects view_address memory forms or restores sources. Native review does not apply "
                    "schema_names, review_notes or review_transfers; its inspect response reports window evidence, "
                    "serialized characters and previous-request input, not current free tokens. "
                    "inspect answers a bounded inventory (counts and tokens by kind) plus one page of units in "
                    "transcript order (offset / next_offset), each with kind and eligibility; the owner's words and "
                    "your typed owner-directed dialogue, the assignment and the core stay whole. "
                    "Earlier notes stay in place unless selected for replacement; select several to merge them. "
                    "The note occupies the first replaced unit. Rewriting can lose prefix-cache "
                    "reuse and provider-signed reasoning from the earliest change: fold for space/clarity in useful batches, with a note shorter than its sources. "
                    "A selected older assistant tool call and all of its "
                    "contiguous matching results stay atomic; Ouroboros checkpoints their exact "
                    "actor-visible bytes before replacement, with checkpoint/CAS provenance. Raw evidence remains retrievable (restore_unit_refs; an emergency "
                    "record lists its checkpoint once and its member units separately)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "view_address": {"type": "array", "items": {"type": "string"}, "description": "Exact memory form addresses from inspect; a note's restore_view_address restores it whole. Apply at the next tool boundary from the same snapshot, retaining earlier choices and new arrivals. May accompany a working_note fold. Core and governing words cannot be selected. The receipt prices prefix re-reading and measures the complete result."},
                        "inspect": {"type": "boolean", "description": "Return and pin the last observed view revision: an inventory by kind and size, one page of complete unit IDs, the restore checkpoint and current schema names, without changing context."},
                        "offset": {"type": "integer", "minimum": 0, "description": "With inspect: zero-based position of the first unit of the page; the reply names next_offset while units remain."},
                        "expected_view_revision": {"type": "string", "description": "Optional view_revision from inspect, checked exactly. Omitted binds the actual model-send view that produced this call; inspect is not required to replace all completed units."},
                        "working_note": {"type": "string", "description": "For a future reader without selected sources in view: preserve the source-supported situation and purpose behind conclusions, who decided what, its status within those sources, scope, changes, corrections and unresolved work. Distinguish inference and gaps; invent no detail. Later grants/cancellations supersede only their scope, not unaffected branches. Proposals are not approvals; do not retain obsolete restrictions after a later scoped approval. Be concise, no fixed template. Required for every fold except a standalone restore."},
                        "keep_unit_ids": {"type": "array", "items": {"type": "string"}, "description": "Exact inspected complete units to retain raw; takes precedence over keep_last_n, empty keeps none eligible. If both selectors are omitted, an authored note keeps all. Owner words, the assignment, the system view and the newer tail are preserved whatever the selection."},
                        "restore_unit_refs": {"type": "array", "items": {"type": "object", "properties": {
                            "checkpoint_ref": {"type": "object"}, "unit_id": {"type": "string"}, "raw_sha256": {"type": "string"}},
                            "required": ["checkpoint_ref", "unit_id", "raw_sha256"]},
                            "description": "Read exact checkpoint-local units back as labelled sources, never live tool protocol replay (checkpoint_ref is inspect's restore_checkpoint). In Main a nonempty list may be used alone; native review also requires working_note (empty is valid). Supply a note to combine restoration with other view selectors."},
                        "schema_names": {"type": "array", "items": {"type": "string"}, "description": "Nano only: desired canonical schemas. This selects residency, never execution permissions. Low/Max retain their full permitted envelope."},
                        "review_notes": {"type": "array", "items": {"type": "object", "properties": {
                            "bound_decision": {"anyOf": [{"type": "object"}, {"type": "string"}]},
                            "bound_decisions": {"type": "array", "items": {"anyOf": [{"type": "object"}, {"type": "string"}]}},
                            "remark": {"type": "string"}, "reason": {"type": "string"}},
                            "required": ["remark", "reason"], "additionalProperties": False},
                            "description": "With working_note: bound_decision keeps one decision's status with your short remark/reason. Alternatively bound_decisions explicitly replaces a group of closed historical decisions with your account. Use exact bindings or decision_ref values from this observed view; positional IDs alone do not bind a version. Current contracts/statuses, open/deferred/pending facts and gaps stay visible. Missing/stale entries stay full with reasons; no new verdict, host clipping or helper summary."},
                        "review_transfers": {"type": "array", "items": {"type": "object", "properties": {
                            "source": {"type": "object"}, "operative_spec_sha256": {"type": "string"},
                            "decision_ids": {"type": "array", "items": {"type": "string"}}},
                            "required": ["source", "operative_spec_sha256", "decision_ids"]},
                            "description": "With working_note only: explicitly declare that an attachment's decisions and reasons have been transferred into the named current plan spec decisions. Inspect supplies exact source/spec identities. No semantic completeness is inferred; absent or stale transfer keeps the attachment full."},
                        "keep_last_n": {
                            "type": "integer",
                            "description": "With working_note and no keep_unit_ids: number of recent completed atomic tool units to keep raw (range 2-20). Without a note the request is refused.",
                        },
                    },
                    "required": [],
                },
            },
            handler=_compact_context,
            timeout_sec=15,
        ),
    ]
