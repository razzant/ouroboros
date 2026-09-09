"""ACP session facts projected onto the existing task log vocabulary.

Adapted from Q00/ouroboros (Copyright 2025 Q00, MIT); the complete license
notice is in ``ouroboros/gateways/copilot_acp.py``. No SDK event names or
model-output prose are interpreted as ACP protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from ouroboros.gateways.copilot_acp import ACPError

_SHELL_EXIT = re.compile(r"(?:^|\n)<shellId: [A-Za-z0-9_-]+ completed with exit code (-?\d{1,9})>\s*\Z")


def content_text(value: Any) -> str:
    """Extract complete text/diff content, never stringify opaque image resources."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (content_text(item) for item in value)))
    if isinstance(value, dict):
        if value.get("type") == "diff":
            return f"{value.get('path', '')}\n--- before\n{value.get('oldText') or ''}\n+++ after\n{value.get('newText') or ''}"
        for key in ("text", "content"):
            if key in value:
                return content_text(value[key])
    return ""


def _correlation(update: dict) -> dict:
    meta = update.get("_meta")
    sources = [update, meta] if isinstance(meta, dict) else [update]
    return {
        target: source[key]
        for source in sources
        for key, target in (
            ("parentToolCallId", "parent_tool_call_id"),
            ("agentId", "agent_id"), ("toolName", "tool_name"),
        )
        if isinstance(source.get(key), str)
    }


@dataclass
class ACPEventTranslator:
    """Per-session tool correlation and the root's post-tool answer candidate."""

    session_id: str
    tools: dict[str, dict] = field(default_factory=dict)
    answer: list[str] = field(default_factory=list)
    answer_bytes: int = 0
    seen_message_chunk: bool = False
    # A refused oversized answer is incomplete, never silently truncated success.
    max_answer_bytes: int = 8 * 1024 * 1024

    def translate(self, frame: dict) -> list[dict]:
        params = frame.get("params", {})
        if not isinstance(params, dict) or params.get("sessionId") != self.session_id:
            return []
        if frame.get("method") != "session/update":
            return []
        update = params.get("update")
        if not isinstance(update, dict) or not isinstance(update.get("sessionUpdate"), str):
            raise ACPError("Malformed ACP session/update.")
        kind = update["sessionUpdate"]
        base = {"acp_session_id": self.session_id, "acp_update_type": kind, **_correlation(update)}
        if kind in {"agent_message_chunk", "agent_thought_chunk"}:
            content = update.get("content", {})
            if not isinstance(content, dict):
                raise ACPError("Malformed ACP message content.")
            if content.get("type") != "text":
                return [{"type": "task_runtime_update", **base, "text": "Copilot sent non-text content; inspect its retained protocol record."}]
            text = content.get("text")
            if not isinstance(text, str):
                raise ACPError("Malformed ACP text chunk.")
            if kind == "agent_message_chunk" and not base.get("parent_tool_call_id"):
                first = not self.seen_message_chunk
                self.seen_message_chunk = True
                # Copilot's ACP preview sends its tool-filter notice as the first
                # message chunk. Preserve it as CLI telemetry, not model output.
                if first and text.startswith("Info: Disabled tools: ") and "\n" not in text:
                    return [{"type": "task_runtime_update", **base, "acp_update_type": "configuration_notice", "text": text}]
                self.answer_bytes += len(text.encode("utf-8"))
                if self.answer_bytes > self.max_answer_bytes:
                    raise ACPError("Copilot answer exceeds the in-memory limit; complete frames remain in observability.", "acp_answer_too_large")
                self.answer.append(text)
            return [{"type": "task_runtime_update", **base, "text": text}] if text else []
        if kind == "plan":
            entries = update.get("entries")
            if not isinstance(entries, list) or any(
                not isinstance(row, dict) or not isinstance(row.get("content"), str)
                for row in entries
            ):
                raise ACPError("Malformed ACP plan.")
            return [{
                "type": "task_runtime_update", **base, "plan": entries,
                "text": "\n".join(f"{row.get('status', 'pending')}: {row['content']}" for row in entries),
            }]
        if kind in {"tool_call", "tool_call_update"}:
            return self._tool_update(update, base)
        # A preview protocol can grow: unknown events stay in the exact transcript,
        # without being mistaken for a final answer, tool result, or permission.
        return []

    def _tool_update(self, update: dict, base: dict) -> list[dict]:
        tool_id = update.get("toolCallId")
        if not isinstance(tool_id, str) or not tool_id:
            raise ACPError("ACP tool update omitted toolCallId.")
        first = tool_id not in self.tools
        if first and len(self.tools) >= 4096:
            raise ACPError("ACP tool correlation limit exceeded.", "acp_tool_limit")
        previous = self.tools.get(tool_id, {})
        state = {**previous, **update}
        self.tools[tool_id] = state
        correlation = {**_correlation(previous), **base}
        if first and not correlation.get("parent_tool_call_id"):
            self.answer.clear()
            self.answer_bytes = 0
        title = state.get("title")
        tool = str(state.get("kind") or "tool")
        fields = {
            **correlation, "tool_call_id": tool_id,
            "tool": f"Copilot {tool}", "title": title if isinstance(title, str) else tool,
            "args": state.get("rawInput", {}), "locations": state.get("locations", []),
        }
        events = [{"type": "tool_call_started", **fields}] if first else []
        status = state.get("status")
        if status is not None and not isinstance(status, str):
            raise ACPError("Malformed ACP tool status.")
        if status in {"completed", "failed"}:
            if previous.get("status") not in {"completed", "failed"}:
                raw = state.get("rawOutput", {})
                exit_code = next(
                    (raw[key] for key in ("exitCode", "exit_code", "returncode") if type(raw.get(key)) is int),
                    None,
                ) if isinstance(raw, dict) else None
                if exit_code is None and tool == "execute":
                    footer = _SHELL_EXIT.search(content_text(raw))
                    if footer:
                        exit_code = int(footer.group(1))
                is_error = status == "failed" or (
                    isinstance(raw, dict) and (raw.get("isError") is True or raw.get("success") is False)
                ) or (exit_code is not None and exit_code != 0)
                events.append({
                    "type": "tool_call_finished", **fields, "status": status,
                    "is_error": bool(is_error), "exit_code": exit_code,
                    "result_preview": content_text(state.get("rawOutput")) or content_text(state.get("content")),
                })
            # Retain identity, not potentially megabyte-sized terminal tool bodies.
            self.tools[tool_id] = {"status": status, **_correlation(state)}
        elif not first:
            events.append({
                "type": "task_runtime_update", **base, "tool_call_id": tool_id,
                "text": content_text(update.get("content")) or str(title or "Copilot tool running"),
            })
        for item in update.get("content", []) if isinstance(update.get("content"), list) else []:
            if isinstance(item, dict) and item.get("type") == "diff":
                events.append({"type": "task_runtime_update", **base, "acp_update_type": "diff", "diff": item, "text": content_text(item)})
        return events

    def finish(self, result: dict) -> str:
        if result.get("stopReason") != "end_turn":
            raise ACPError(f"Copilot ACP stopped: {result.get('stopReason', 'unknown')}.", "acp_turn_incomplete")
        if any(row.get("status") not in {"completed", "failed"} for row in self.tools.values()):
            raise ACPError("Copilot ACP ended with unfinished tools.", "acp_turn_incomplete")
        answer = "".join(self.answer)
        if not answer.strip():
            raise ACPError("Copilot ACP ended without a final answer.", "acp_empty_answer")
        return answer
