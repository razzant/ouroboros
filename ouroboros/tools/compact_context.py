"""LLM-requested tool-history compaction trigger."""

from __future__ import annotations

import logging
import copy
import json
from pathlib import Path
from typing import List

from ouroboros.tools.registry import ToolEntry

log = logging.getLogger(__name__)
_CANONICAL_OBSERVATION = object()


def record_context_view(ctx, messages, tool_schemas, *, physical_capture=_CANONICAL_OBSERVATION) -> None:
    """Capture a usable turn's canonical source; prospective pricing never calls this.

    Nothing is added to the prompt or cache identity. Inspect pins this one
    observation for a later authored request, so its own tool pair and newer
    owner messages cannot make that request stale by construction.
    Physical vision/provider projections remain in the existing request artifacts.
    """
    from ouroboros.context_compaction import context_reclaim_transcript_sha256

    observed = copy.deepcopy(messages)
    ctx._last_context_observation = {
        "revision": context_reclaim_transcript_sha256(observed),
        "messages": observed, "tool_schemas": copy.deepcopy(tool_schemas),
    }
    if physical_capture is _CANONICAL_OBSERVATION:
        return  # Existing non-Main actors retain their own observation boundary.
    observation = ctx._last_context_observation
    observation.update(exposed_units=[], physical_source_status="unavailable")
    if physical_capture is None or not physical_capture.candidate_manifest_ref:
        return
    from ouroboros.context_compaction import exposed_context_units
    from ouroboros.observability import read_blob_ref, read_call_manifest_ref

    for root in (getattr(ctx, "budget_drive_root", None), getattr(ctx, "drive_root", None)):
        if root is None:
            continue
        try:
            manifest = read_call_manifest_ref(Path(root), physical_capture.candidate_manifest_ref, task_id=ctx.task_id)
            payload = read_blob_ref(Path(root), manifest["full_payload_ref"])
            physical_messages = payload.get("messages")
            if not isinstance(physical_messages, list):
                continue
            observation.update(
                exposed_units=list(exposed_context_units(observed, physical_messages)),
                physical_source_status="observed_projection", physical_attempt_id=physical_capture.attempt_id,
                physical_source_ref=physical_capture.candidate_manifest_ref,
            )
            return
        except (OSError, ValueError, KeyError, TypeError):
            log.debug("Physical context source unavailable", exc_info=True)


def _compact_context(ctx, keep_last_n: int | None = None, *, inspect: bool = False,
                     expected_view_revision: str = "", working_note: str | None = None,
                     keep_unit_ids: List[str] | None = None, restore_unit_refs: List[dict] | None = None,
                     schema_names: List[str] | None = None, **kwargs) -> str:
    """Inspect or queue a replacement through the existing context materializer."""
    from ouroboros.context_compaction import _atomic_units
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    if inspect:
        observed = getattr(ctx, "_last_context_observation", None)
        if not isinstance(observed, dict):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
                text="Context view unavailable: this actor has no recorded model-send observation yet."))
        ctx._inspected_context_view = observed
        units = _atomic_units(observed["messages"])
        checkpoint = observed.get("checkpoint_ref")
        if not checkpoint and getattr(ctx, "task_id", "") and getattr(ctx, "drive_root", None):
            from ouroboros.artifacts import store_actor_source_bytes

            try:
                roots = dict.fromkeys((Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root).resolve(),
                                       Path(ctx.drive_root).resolve()))
                raw = json.dumps({"messages": observed["messages"],
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
        # Keep the full reader hint on the pinned observation, outside this O(N) reply.
        restore_checkpoint = ({key: checkpoint[key] for key in ("kind", "root", "path", "size", "sha256")}
                              if checkpoint else None)
        return json.dumps({
            "view_revision": observed["revision"],
            "units": [{"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256,
                       "estimated_tokens": unit.context_size_tokens, "source_refs": unit.source_refs,
                       "restore_ref": {"checkpoint_ref": restore_checkpoint, "unit_id": unit.unit_id,
                                       "raw_sha256": unit.raw_sha256} if checkpoint else None,
                       "physically_exposed": ({"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256}
                                              in observed["exposed_units"]) if "exposed_units" in observed else None}
                      for unit in units],
            "schema_names": [s["function"]["name"] for s in observed["tool_schemas"]],
            "rule": "This revision names the observed messages; schemas are listed separately. Select complete unit IDs to keep, write one working_note, and preserve original sources. Newer owner/tool messages remain untouched.",
        }, ensure_ascii=False, separators=(",", ":"))
    if working_note is not None:
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
                       for values in (keep_unit_ids, schema_names))
                or restore_unit_refs is not None and (not isinstance(restore_unit_refs, list)
                    or not all(isinstance(ref, dict) for ref in restore_unit_refs))):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text="Invalid context view request: use a prose working_note, arrays of exact unit/schema names and checkpoint reference objects."))
        if keep_unit_ids is None and keep_last_n is not None:
            count = max(2, min(keep_last_n, 20))
            keep_unit_ids = [unit.unit_id for unit in _atomic_units(observed["messages"])[-count:]]
        ctx._pending_compaction = {
            "observed": observed, "working_note": working_note,
            "expected_view_revision": observed["revision"],
            "keep_unit_ids": None if keep_unit_ids is None else tuple(keep_unit_ids),
            "restore_unit_refs": tuple(restore_unit_refs or ()),
            "schema_names": None if schema_names is None else tuple(schema_names),
        }
        return "Working view requested. The next complete tool boundary preserves exact sources and checks the full candidate before applying it; the resulting receipt reports actual changes."

    keep_last_n = max(2, min(6 if keep_last_n is None else keep_last_n, 20))

    ctx._pending_compaction = keep_last_n

    return (
        f"✅ Context reclaim scheduled: keeping the last {keep_last_n} completed tool units raw. "
        "After a useful selection is known, Ouroboros checkpoints the exact actor-visible "
        "transcript and replaces only fully covered older atomic units with active-context "
        "summaries carrying checkpoint/CAS provenance. The raw evidence remains retrievable "
        "from those references. This takes effect on the next round."
    )


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="compact_context",
            schema={
                "name": "compact_context",
                "description": (
                    "Request complete-input context reclaim for old completed tool units. "
                    "Supply your working_note to author the replacement, or omit it for helper summarization. "
                    "Select exact keep_unit_ids or explicitly keep_last_n recent units raw; "
                    "an authored note without either keeps all raw units. A selected older assistant tool call and all of its "
                    "contiguous matching results stay atomic; Ouroboros checkpoints their exact "
                    "actor-visible bytes before replacing them with summaries whose metadata points "
                    "to the checkpoint/CAS evidence. Active context becomes summarized; raw evidence "
                    "remains retrievable through the recorded provenance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "inspect": {"type": "boolean", "description": "Return and pin the last observed view revision, complete unit IDs, source references and current schema names without changing context."},
                        "expected_view_revision": {"type": "string", "description": "Optional view_revision from inspect, checked exactly. Omitted binds the actual model-send view that produced this call; inspect is not required to replace all completed units."},
                        "working_note": {"type": "string", "description": "One coherent account of current understanding, corrections and unresolved work. Supplying it selects your authored view; omission keeps legacy helper compaction."},
                        "keep_unit_ids": {"type": "array", "items": {"type": "string"}, "description": "Exact inspected complete units to retain raw; takes precedence over keep_last_n, empty keeps none. If both selectors are omitted, an authored note keeps all. Owner/system messages and newer tail are preserved."},
                        "restore_unit_refs": {"type": "array", "items": {"type": "object", "properties": {
                            "checkpoint_ref": {"type": "object"}, "unit_id": {"type": "string"}, "raw_sha256": {"type": "string"}},
                            "required": ["checkpoint_ref", "unit_id", "raw_sha256"]},
                            "description": "Read exact checkpoint-local units back as labelled sources, never live tool protocol replay."},
                        "schema_names": {"type": "array", "items": {"type": "string"}, "description": "Nano only: desired canonical schemas. This selects residency, never execution permissions. Low/Max retain their full permitted envelope."},
                        "keep_last_n": {
                            "type": "integer",
                            "description": "Number of recent completed atomic tool units to keep raw (range 2-20). With working_note, used only when explicitly supplied and keep_unit_ids is omitted. Without working_note, defaults to 6 for helper summarization.",
                        },
                    },
                    "required": [],
                },
            },
            handler=_compact_context,
            timeout_sec=15,
        ),
    ]
