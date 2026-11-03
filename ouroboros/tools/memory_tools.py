"""Memory registry tools for tracking data sources, gaps, and trust."""

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read

import re
import logging
from pathlib import Path
from typing import List
from hashlib import sha256

from ouroboros.tools.registry import ToolEntry, ToolContext

log = logging.getLogger(__name__)

REGISTRY_PATH = "memory/registry.md"


def _registry_file(ctx: ToolContext) -> Path:
    from ouroboros.tool_access import canonical_data_root

    return canonical_data_root(ctx) / REGISTRY_PATH


@completed_local_read
def _memory_map(ctx: ToolContext) -> str:
    """Read the memory registry — a map of all data sources."""
    path = _registry_file(ctx)
    if not path.exists():
        return (
            "Memory registry not found at memory/registry.md.\n"
            "revision=\"\" (create-only). Use memory_update_registry to create entries."
        )
    raw = path.read_bytes()
    return f"revision={sha256(raw).hexdigest()}\n\n" + raw.decode("utf-8")


def _memory_update_registry(
    ctx: ToolContext, source_id: str, updates: str, expected_revision: str | None = None
) -> str:
    """Update or create an entry in the memory registry."""
    if not source_id or not isinstance(source_id, str):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("⚠️ source_id must be a non-empty string.")))
    source_id = source_id.strip()
    if "/" in source_id or "\\" in source_id or ".." in source_id:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("⚠️ Invalid characters in source_id.")))

    from ouroboros.knowledge import focus_signature
    from ouroboros.memory_publication import publish_memory_file, publication_refusal

    path = _registry_file(ctx)
    result = publish_memory_file(path, expected_revision, lambda old: _registry_entry(old, source_id, updates),
                                 journal=path.with_name("registry_history.jsonl"),
                                 author=focus_signature(ctx), source_type="memory_update_registry")
    if not result.ok:
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
            text=publication_refusal(result, REGISTRY_PATH)))
    return f"✅ Registry entry '{source_id}' updated. revision={result.revision}"


def _registry_entry(content: str, source_id: str, updates: str) -> str:
    content = content or "# Memory Registry\n\nMetacognitive map: what I know, what I don't, and where to look.\n\n"
    pattern = rf'^### {re.escape(source_id)}\s*$'
    lines = content.split("\n")
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if re.match(pattern, line):
            start_idx = i
        elif start_idx is not None and line.startswith("### "):
            end_idx = i
            break

    new_section = f"### {source_id}\n{updates.strip()}\n"

    if start_idx is not None:
        if end_idx is None:
            end_idx = len(lines)
        while end_idx > start_idx and not lines[end_idx - 1].strip():
            end_idx -= 1
        lines[start_idx:end_idx] = [new_section]
        content = "\n".join(lines)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += "\n" + new_section

    return content

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("memory_map", {
            "name": "memory_map",
            "description": (
                (
                'Read memory/registry.md, the optional hand-written source registry: recorded coverage, gaps and trust. '
                'This is not a discovered integration or file inventory; absence means nothing was recorded. Check '
                'source availability before relying on cached impressions.'
            )
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            },
        }, _memory_map),
        ToolEntry("memory_update_registry", {
            "name": "memory_update_registry",
            "description": (
                (
                "Replace one '### <source_id>' entry in the hand-written memory/registry.md source registry with "
                'updates, its full new text.'
            )
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expected_revision": {
                        "type": "string",
                        "description": "Revision returned by memory_map; required when the registry exists. Empty means create-only."
                    },
                    "source_id": {
                        "type": "string",
                        "description": "Source identifier (e.g. 'user-context', 'project-notes')"
                    },
                    "updates": {
                        "type": "string",
                        "description": "Full entry content in markdown (- **Path:** ... \\n- **Type:** ... etc)"
                    }
                },
                "required": ["source_id", "updates"]
            },
        }, _memory_update_registry),
    ]
