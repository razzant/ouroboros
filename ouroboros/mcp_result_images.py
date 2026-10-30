"""Chrome result projection and same-round images from host-retained MCP bytes.

Only the validated task-owned Chrome call mints ``mcp_browser_image_owner``.
Neither a tool-name prefix nor remote text/_meta grants attachment authority.
The raw response remains the source; no remote path or URL is followed.
"""
from __future__ import annotations

import base64
import copy
import dataclasses
import json
import pathlib
from types import SimpleNamespace
from typing import Any

from ouroboros.tools.tool_result import ToolResult, _replace_tool_result


def _binary(part: dict) -> tuple[dict, str] | None:
    """Protocol binary fields only; text/structured JSON is not an image instruction."""
    if part.get("type") in {"image", "audio"} and isinstance(part.get("data"), str):
        return part, "data"
    resource = part.get("resource")
    if part.get("type") == "resource" and isinstance(resource, dict) and isinstance(resource.get("blob"), str):
        return resource, "blob"
    return None


def project_browser_result(raw: Any, source: dict | None, retention_gap: str,
                           *, source_root: pathlib.Path | None, owner_task_id: str) -> ToolResult:
    """Use the ordinary MCP adapter, replacing only retained binary fields in its view."""
    from ouroboros.mcp_client import _serialize_content_part, _tool_result_from_call_result

    content, binary_count = [], 0
    for index, item in enumerate(getattr(raw, "content", None) or []):
        part = _serialize_content_part(item)
        binary = _binary(part)
        if binary is None:
            content.append(item)
            continue
        binary_count += 1
        if source is None:
            content.append(item)  # No loss of the sole copy when retention failed.
            continue
        # The generic adapter keeps SDK objects such as Annotations intact. Only
        # this binary projection needs a JSON-safe SDK dump before json.dumps.
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            part = dump(mode="json", by_alias=True)
            binary = _binary(part)
            if binary is None:
                content.append(item)
                continue
        container, field = binary
        container[field] = f"[binary retained in response source: content[{index}] {field}]"
        content.append(SimpleNamespace(type="text", text=json.dumps(part, ensure_ascii=False)))
    projected = SimpleNamespace(
        content=content, structuredContent=getattr(raw, "structuredContent", None),
        isError=bool(getattr(raw, "isError", False) or getattr(raw, "is_error", False)),
    )
    result = _tool_result_from_call_result(projected)
    if not binary_count:
        return result
    if source is None:
        note = (f"Binary response retention failed ({retention_gap or 'unknown'}); "
                "original content kept inline; automatic image attachment unavailable.")
        updated = _replace_tool_result(result, meta_updates={"mcp_response_retention_gap": retention_gap or "unknown"})
        return dataclasses.replace(updated, text=result.text + "\n\n" + note,
                                   producer_text=result.text, host_annotations=(note,))
    from ouroboros.artifacts import task_artifact_dir_path

    if source_root is None:
        raise ValueError("retained MCP response has no readable source root")
    pointer_path = str(task_artifact_dir_path(source_root, owner_task_id, create=False) / source["path"])
    pointer = (f"MCP binary response source: read_file(root='artifact_store', path='{pointer_path}'); "
               f"size={source['size']} bytes; sha256={source['sha256']}. "
               "Binary payload remains in the source; image attachment is reported separately when offered.")
    updated = _replace_tool_result(result, meta_updates={"mcp_response_source": source})
    return dataclasses.replace(updated, text=result.text + "\n\n" + pointer,
                               producer_text=result.text, host_annotations=(pointer,))


def attach_browser_images(ctx: Any, result: ToolResult, *, staged_messages: list | None = None) -> dict:
    """Consume only host-minted owner/source metadata through the existing vision path."""
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes, task_artifact_dir_path
    from ouroboros.mcp_task_sessions import _data_root, _owner
    from ouroboros.tools.vision import _VLM_MAX_FILE_BYTES, attach_local_image_to_context

    observation: dict[str, Any] = {"status": "unavailable"}
    try:
        if result.status != "ok" or list(_owner(ctx)) != result.meta.get("mcp_browser_image_owner"):
            return observation
        raw = json.loads(read_actor_source_bytes(_data_root(ctx), _owner(ctx)[0],
                                                 result.meta["mcp_response_source"]))
        if staged_messages is not None:
            ctx = copy.copy(ctx)
            ctx.messages = staged_messages
        items = []
        for index, part in enumerate(raw.get("content") or []):
            binary = _binary(part)
            if binary is None:
                continue
            container, field = binary
            if part.get("type") != "image" and not str(container.get("mimeType", "")).startswith("image/"):
                continue
            item = {"content_index": index, "status": "unavailable"}
            try:
                encoded = container[field]
                if len(encoded) > 4 * ((_VLM_MAX_FILE_BYTES + 2) // 3):
                    raise ValueError("image exceeds the local image byte limit")
                data = base64.b64decode(encoded, validate=True)
                image = store_actor_source_bytes(ctx.drive_root, _owner(ctx)[0], category="tool_results",
                                                 source_id=f"mcp-image-{index}", data=data, extension="bin")
                path = task_artifact_dir_path(ctx.drive_root, _owner(ctx)[0], create=False) / image["path"]
                ok, note = attach_local_image_to_context(ctx, str(path))
                item.update(status="attached" if ok else "unavailable", note=note)
            except Exception as exc:
                item["note"] = f"Image attachment unavailable ({type(exc).__name__})."
            items.append(item)
        observation["items"] = items
        if items and all(item["status"] == "attached" for item in items):
            observation["status"] = "attached"
        elif any(item["status"] == "attached" for item in items):
            observation["status"] = "partial"
        elif not items:
            observation["status"] = "not_offered"
    except Exception as exc:
        observation["note"] = f"Retained image response unavailable ({type(exc).__name__})."
    return observation
