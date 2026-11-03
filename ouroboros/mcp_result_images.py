"""MCP protocol images: exact response custody, a concise projection and same-round attachment.

A confirmed MCP response holding a protocol image (``ImageContent`` data, or an
``EmbeddedResource`` blob whose mimeType is image/*) leaves the transport as
``ReturnedResult``, which also carries its SDK value (never in meta, text or repr); every
other response stays the plain adapted result. ``project_returned`` engages only for such a
value and a calling task. It keeps the complete response once as an exact task source, writes
each decodable image as its own task file, and shows text parts, structured content and
non-binary fields with each binary field read as a marker (an image's names its file for
``view_image``), plus one host note naming the source. Any other value keeps the ordinary
projection byte for byte. Configured secrets are masked in data, never inside base64 (a short
secret can occur there, and masking it would corrupt the image) and never in the protocol's own
vocabulary: field names, the ``type`` discriminator and the media type that opens a ``mimeType``
stay exact, so a secret that spells one cannot reclassify the payload. Fields a server adds, maps
such as ``_meta``, MIME parameters and all other text are data and are masked. Each image file
keeps its exact decoded bytes; the shared image loader alone decides, when it is viewed, whether
and how they are shown. When the source cannot be kept, the sole payload stays inline with the
gap named and no image is offered. Projection never changes the call's status, and a fault in it
is a named gap, so a completed remote call never reads as a retryable failure.

These sources live in the task's store on the canonical data root from the start, like
delegated outputs in the same plane: write-once by digest, they outlive a child's execution
drive, and the retention walk that transfers a child's sources does not follow them. Printed
readers are absolute, so one address holds before and after retirement for the task, a later
attempt and its parent; dispatch and the common image loader still decide who may open it.

Only host metadata offers attachment: ``mcp_response_source`` with ``mcp_image_owner``, the
calling context's [task, attempt], minted on a successful call. Remote text, ``_meta``,
URIs, paths and tool-name spellings grant nothing and are never followed.
``loop_tool_execution.process_tool_results`` calls ``attach_returned_images`` on its staged
batch history, through the reader ``view_image`` uses, for at most the owner mode's image
floor per result (``context_images.image_floor``); the other images stay files
(``source_only``). ``settle_returned_images`` then records each attached block's final state
in that history (a later image of the batch may have evicted it) and adds one host note.

Adapted from razzant/ouroboros#1401 ``mcp_result_images.py`` (Andrei Kaznacheev, 55ab6a7b;
Ouroboros, 4384d057): its task-owned Chrome projection, generalized to every configured
server. A Chrome adapter keeps its lease, session and post-state checks and composes
``project_returned`` on its returned value; a refused post-state becomes an error result,
which keeps the source and attaches nothing.
"""
from __future__ import annotations

import base64
import copy
import dataclasses
import json
import logging
import pathlib
import re
from types import SimpleNamespace
from typing import Any, Optional

from ouroboros.tools.tool_result import ToolResult

log = logging.getLogger(__name__)

_RETAINED = "kept exactly in the retained response"
# A media type's own RFC 6838 tokens; parameters after it are free text.
_MEDIA_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}")


@dataclasses.dataclass(frozen=True)
class ReturnedResult(ToolResult):
    """An adapted transport result and the image-bearing SDK value it came from, for projection only."""

    returned: Any = dataclasses.field(default=None, repr=False, compare=False)


def with_returned(result: ToolResult, raw: Any) -> ToolResult:
    """``result`` itself, unless ``raw`` holds a protocol image: then its carrier for projection."""
    try:
        if not any(_is_image(_part(item)) for item in getattr(raw, "content", None) or []):
            return result
    except Exception:  # noqa: BLE001 - the remote call is complete; the ordinary result still holds its answer
        log.warning("MCP response could not be inspected for images; the ordinary projection is used", exc_info=True)
        return result
    return ReturnedResult(**{item.name: getattr(result, item.name) for item in dataclasses.fields(ToolResult)},
                          returned=raw)


def _owner(ctx: Any) -> Optional[list]:
    """The calling context's [task, attempt]; None for a context that owns no task."""
    from ouroboros.task_results import validate_task_id

    try:
        task = validate_task_id(getattr(ctx, "task_id", None))
    except ValueError:
        return None
    attempt = getattr(ctx, "task_attempt", None)
    return [task, str(1 if attempt in (None, "") else attempt)]


def _binary_field(part: Any) -> Optional[tuple]:
    """(container, field) of a serialized content part's protocol binary field."""
    if not isinstance(part, dict):
        return None
    if part.get("type") in {"image", "audio"} and isinstance(part.get("data"), str):
        return part, "data"
    resource = part.get("resource")
    if part.get("type") == "resource" and isinstance(resource, dict) and isinstance(resource.get("blob"), str):
        return resource, "blob"
    return None


def _is_image(part: Any) -> bool:
    found = _binary_field(part)
    return found is not None and (part["type"] == "image"
                                  or str(found[0].get("mimeType") or "").lower().startswith("image/"))


def _masked(value: Any, cfg: Any) -> Any:
    """Configured secrets masked in every string and key, the MCP result policy."""
    from ouroboros.mcp_schema import _redact_error_text

    if isinstance(value, str):
        return _redact_error_text(value, cfg)
    if isinstance(value, list):
        return [_masked(item, cfg) for item in value]
    if isinstance(value, dict):
        return {_masked(key, cfg): _masked(item, cfg) for key, item in value.items()}
    return value


def _protocol_view(value: Any, model: Any, cfg: Any, binary: Any = None) -> Any:
    """``value``, the JSON of SDK object ``model``, with configured secrets masked in its data.

    A model's field names, its ``type`` and the media type opening its ``mimeType`` are protocol
    vocabulary and stay exact; its protocol binary field stays exact, or becomes ``binary``. Fields
    outside the model's schema, plain maps, MIME parameters and all other strings are masked."""
    found = _binary_field(value)

    def view(node: Any, source: Any) -> Any:
        if not isinstance(node, dict) or isinstance(source, dict):
            return _masked(node, cfg)
        fields = getattr(type(source), "model_fields", None)
        schema = {field.alias or name: name for name, field in fields.items()} if fields else {key: key for key in node}
        out = {}
        for key, item in node.items():
            name = schema.get(key)
            if found is not None and node is found[0] and key == found[1]:
                out[key] = item if binary is None else binary
            elif name is None:
                out[_masked(key, cfg)] = _masked(item, cfg)
            elif name == "type" and isinstance(item, str):
                out[key] = item
            elif name == "mimeType" and isinstance(item, str) and (head := _MEDIA_TYPE.match(item)):
                out[key] = head.group(0) + _masked(item[head.end():], cfg)
            else:
                out[key] = view(item, getattr(source, name, None))
        return out

    return view(value, model)


def _part(item: Any) -> dict:
    """The ordinary JSON view of one content part, with SDK annotations made JSON-safe."""
    from ouroboros.mcp_schema import _serialize_content_part

    part = _serialize_content_part(item)
    dump = getattr(part.get("annotations"), "model_dump", None)
    if callable(dump):
        part["annotations"] = dump(mode="json", by_alias=True, exclude_none=True)
    return part


def _response_bytes(raw: Any, parts: list, cfg: Any) -> bytes:
    """The complete returned value as canonical JSON, through ``_protocol_view``: data masked,
    protocol vocabulary and binary fields exact."""
    payload = None
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            payload = dump(mode="json", by_alias=True)
        except (TypeError, ValueError):
            payload = None
    if not isinstance(payload, dict):
        payload = {"content": parts, "isError": bool(getattr(raw, "isError", False) or getattr(raw, "is_error", False)),
                   "structuredContent": getattr(raw, "structuredContent", None)}
    items = list(getattr(raw, "content", None) or [])
    content = [_protocol_view(part, items[index] if index < len(items) else None, cfg)
               for index, part in enumerate(payload.get("content") or [])]
    rest = _protocol_view({key: value for key, value in payload.items() if key != "content"}, raw, cfg)
    return json.dumps({**rest, "content": content}, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _saved_image(drive_root: pathlib.Path, task_id: str, source: dict, index: int, part: dict) -> dict:
    """Decode one protocol image into its own task file: ``{path, size}`` or ``{gap}``. The exact
    bytes are kept whatever their format; the shared image loader decides when they are viewed."""
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path
    from ouroboros.image_preparation import image_mime

    container, field = _binary_field(part)
    try:
        data = base64.b64decode("".join(container[field].split()), validate=True)
    except ValueError as exc:
        return {"gap": f"image data is not valid base64 ({exc})"}
    mime = image_mime(data)  # names the file only; an unknown signature keeps a generic suffix
    try:
        ref = store_actor_source_bytes(drive_root, task_id, category="tool_results",
                                       source_id=f"mcp-{source['sha256'][:16]}.content-{index}",
                                       data=data, extension=mime.split("/")[-1] if mime else "image")
    except Exception as exc:  # noqa: BLE001 - the exact bytes stay in the retained response
        return {"gap": f"image file not saved ({type(exc).__name__})"}
    return {"path": task_artifact_dir_path(drive_root, task_id) / ref["path"], "size": len(data)}


def project_returned(adapted: ToolResult, raw: Any, ctx: Any, cfg: Any, tool_name: str) -> Optional[ToolResult]:
    """``adapted`` retained and projected for ``ctx``'s task when ``raw`` holds a protocol
    image; ``None`` keeps the ordinary projection."""
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path
    from ouroboros.mcp_schema import _model_facing_result, _redact_error_text
    from ouroboros.tool_access import canonical_data_root

    owner = _owner(ctx)
    try:
        items = list(getattr(raw, "content", None) or []) if raw is not None else []
        parts = [_part(item) for item in items]
        if owner is None or not any(_is_image(part) for part in parts):
            return None
        root = canonical_data_root(ctx)
    except Exception:  # noqa: BLE001 - the ordinary projection still carries the complete response
        log.warning("MCP image projection unavailable; the ordinary projection is used", exc_info=True)
        return None
    source, meta = None, {}
    try:
        source = store_actor_source_bytes(root, owner[0], category="tool_results",
                                          source_id=f"mcp_{cfg.id}__{tool_name}.response",
                                          data=_response_bytes(raw, parts, cfg), extension="json")
        note = (f"MCP response retained exactly: read_file(path={str(task_artifact_dir_path(root, owner[0]) / source['path'])!r}); "
                f"{source['size']} bytes, sha256 {source['sha256']}.")
        meta["mcp_response_source"] = source
    except Exception as exc:  # noqa: BLE001 - the sole payload then stays inline
        note = (f"MCP response retention failed ({type(exc).__name__}): its original content stays inline above "
                "and no image is attached automatically.")
        meta["mcp_response_retention_gap"] = type(exc).__name__
    try:
        producer = _model_facing_result(cfg, tool_name, _projected_body(adapted, raw, items, parts, source, owner, root, cfg))
        if source is not None and adapted.status == "ok":
            meta["mcp_image_owner"] = owner
    except Exception as exc:  # noqa: BLE001 - the remote call is complete: a named gap, never a failure or a resend
        log.warning("MCP image projection failed after the remote call", exc_info=True)
        producer = _model_facing_result(cfg, tool_name, _redact_error_text(adapted.text, cfg))
        note += (f" Its projection failed ({type(exc).__name__}): the ordinary result stays above "
                 "and no image is attached automatically.")
        meta["mcp_response_projection_gap"] = type(exc).__name__
    return ToolResult(status=adapted.status, code=adapted.code, text=f"{producer}\n\n{note}",
                      meta={**adapted.meta, **meta}, producer_text=producer, host_annotations=(note,))


def _projected_body(adapted: ToolResult, raw: Any, items: list, parts: list, source: Optional[dict],
                    owner: list, root: pathlib.Path, cfg: Any) -> str:
    """The ordinary rendering with each binary field a marker; exact, unmasked, without a source."""
    from ouroboros.mcp_schema import _content_part_text, _redact_error_text, _stringify_call_result

    shown = []
    for index, (item, part) in enumerate(zip(items, parts)):
        found = _binary_field(part)
        if found is None:
            shown.append(SimpleNamespace(text=_redact_error_text(_content_part_text(item), cfg)))
            continue
        if source is None:
            binary = found[0][found[1]]
        elif _is_image(part):
            saved = _saved_image(root, owner[0], source, index, part)
            binary = (f"[image data, {saved['size']} bytes, retained as {saved['path']}]"
                      if "path" in saved else f"[{saved['gap']}; {_RETAINED}]")
        else:
            binary = f"[binary data {_RETAINED}]"
        shown.append(SimpleNamespace(text=json.dumps(_protocol_view(part, item, cfg, binary), ensure_ascii=False)))
    return _stringify_call_result(SimpleNamespace(
        content=shown, isError=adapted.status == "error",
        structuredContent=_masked(getattr(raw, "structuredContent", None), cfg)))


def attach_returned_images(ctx: Any, result: ToolResult, staged_messages: Optional[list]) -> dict:
    """Attach a successful result's retained images to the staged history, up to the image floor.

    ``ctx`` is the original task context, never a staging copy: it must be the stamped
    task attempt and it receives the eviction stamp. An unreadable or unverified source is
    a gap; nothing attaches and nothing runs again."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.context_images import image_floor
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.vision import attach_local_image_to_context

    observation: dict = {"status": "unavailable", "items": [], "_blocks": []}
    owner = _owner(ctx)
    if owner is None or list(result.meta.get("mcp_image_owner") or ()) != owner:
        observation["note"] = "the retained images belong to another task attempt"
        return observation
    source = dict(result.meta.get("mcp_response_source") or {})
    try:
        root = canonical_data_root(ctx)
        payload = json.loads(read_actor_source_bytes(root, owner[0], source))
        staged = ctx
        if staged_messages is not None:
            staged = copy.copy(ctx)
            staged.messages = staged_messages  # never rebind the shared live context
        keep = image_floor(getattr(getattr(ctx, "context_fit_plan", None), "preferred_mode", "max"))[0]
        for index, part in enumerate(payload.get("content") or []):
            if not _is_image(part):
                continue
            item = {"content_index": index, "status": "unavailable"}
            observation["items"].append(item)
            saved = _saved_image(root, owner[0], source, index, part)
            if "gap" in saved:
                item["note"] = saved["gap"]
            elif len(observation["_blocks"]) >= keep:
                item.update(status="source_only", path=str(saved["path"]))
            else:
                ok, message = attach_local_image_to_context(staged, str(saved["path"]), stamp=ctx, observation=item)
                if ok:
                    item["status"] = "attached"
                    observation["_blocks"].append((item, staged.messages[-1]["content"][-1]))
                else:
                    item["note"] = message
    except Exception as exc:  # noqa: BLE001 - a gap, never a failed call or a re-run
        observation["note"] = f"retained response unavailable ({type(exc).__name__}: {exc})"
    return observation


def settle_returned_images(rows: list, observations: list, history: list) -> None:
    """Record each attached block's final state in the staged ``history`` and note it on its row."""
    live = {id(block) for row in history if isinstance(row.get("content"), list) for block in row["content"]}
    for row, observation in zip(rows, observations):
        blocks = observation.pop("_blocks", None) if isinstance(observation, dict) else None
        if blocks is None:
            continue
        for item, block in blocks:
            if id(block) not in live:
                item.update(status="evicted", note="a later image of this batch replaced it; its placeholder names the file")
        states = [item["status"] for item in observation["items"]]
        observation["status"] = ("attached" if states and set(states) == {"attached"}
                                 else "partial" if "attached" in states else "unavailable")
        summary: dict = {"status": observation["status"]}
        for item in observation["items"]:
            entry = {str(item["content_index"]): item["note"]} if item.get("note") else item["content_index"]
            summary.setdefault(item["status"], []).append(entry)
        if observation.get("note"):
            summary["note"] = observation["note"]
        typed = row["tool_result"]
        note = "Host image attachment: " + json.dumps(summary, ensure_ascii=False)
        text = f"{typed.text}\n\n{note}"
        row.update(result=text, tool_result=dataclasses.replace(
            typed, text=text, producer_text=typed.producer_text if typed.producer_text is not None else typed.text,
            host_annotations=typed.host_annotations + (note,)))
