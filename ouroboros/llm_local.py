"""The local llama.cpp lane and its context budget.

A local model has no vision, a small window, and no cost. Fitting a transcript
into that window is a policy decision — compact the sections a local run can
lose, keep the ones it cannot, and refuse rather than silently truncate — so the
compaction rules live beside the send that depends on them.
"""


from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ouroboros.context_budget import context_overflow_message
from ouroboros.llm_attempt import (
    _attempt_request,
    _candidate_before_dispatch,
    _execute_candidate,
    _is_structured_context_overflow_exception,
    _physical_candidate,
)
from ouroboros.usage_accounting import PhysicalAttemptCapture, UsageAccountingError


# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")


class LocalContextTooLargeError(RuntimeError):
    """Raised when a local model cannot fit context without silent truncation."""


# Lives beside its proxy constant; the historical private name stays importable.
from ouroboros.context_budget import estimate_message_chars as _estimate_message_chars


def _split_markdown_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    lines = str(text or "").splitlines()
    preamble: List[str] = []
    sections: List[Tuple[str, str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_title is None:
                preamble = current_lines[:]
            else:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_title is None:
        return "\n".join(lines).strip(), []

    sections.append((current_title, "\n".join(current_lines).strip()))
    return "\n".join(preamble).strip(), sections


def _compact_markdown_sections(
    text: str,
    preserve_titles: Set[str],
    reason: str,
) -> str:
    preamble, sections = _split_markdown_sections(text)
    if not sections:
        return text

    parts: List[str] = []
    if preamble:
        parts.append(preamble)

    for title, section in sections:
        if title in preserve_titles:
            parts.append(section)
            continue
        omitted_chars = max(0, len(section))
        parts.append(
            f"## {title}\n\n"
            f"[Compacted for local-model context: omitted {omitted_chars} chars. {reason}]"
        )

    return "\n\n".join(p for p in parts if p).strip()


_LOCAL_COMPACTION_MODES = {
    "static": (
        {"BIBLE.md"},
        "Use a larger-context model or read the source file directly if this section becomes necessary.",
    ),
    "semi_stable": (
        {"Identity"},
        "Identity was preserved; non-core stable memory sections were compacted for local execution.",
    ),
    "dynamic": (
        {
            "Scratchpad",
            "Dialogue History",
            "Dialogue Summary",
            "Memory Registry (what I know / don't know)",
            "Drive state",
            "Runtime context",
            "Health Invariants",
        },
        "Working-memory and runtime sections were preserved; non-core recent/history sections were compacted for local execution.",
    ),
    "system": (
        {
            "BIBLE.md",
            "Scratchpad",
            "Identity",
            "Drive state",
            "Runtime context",
            "Health Invariants",
            "Recent observations",
            "Background consciousness info",
        },
        "Non-core sections were compacted for local execution.",
    ),
}


def _compact_local_text(text: str, mode: str) -> str:
    preserve_titles, reason = _LOCAL_COMPACTION_MODES[mode]
    return _compact_markdown_sections(text, preserve_titles=preserve_titles, reason=reason)


class _LocalLaneMixin:
    """Local-context compaction and the local chat request."""

    def _prepare_messages_for_local_context(
        self,
        messages: List[Dict[str, Any]],
        ctx_len: int,
        max_tokens: int,
    ) -> List[Dict[str, Any]]:
        available_tokens = max(256, ctx_len - max_tokens - 64)
        target_chars = available_tokens * 3
        total_chars = _estimate_message_chars(messages)
        if total_chars <= target_chars:
            return messages

        compacted = copy.deepcopy(messages)
        for msg in compacted:
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for idx, block in enumerate(content):
                    if not isinstance(block, dict) or block.get("type") != "text":
                        continue
                    block_text = str(block.get("text", ""))
                    if idx == 0:
                        block["text"] = _compact_local_text(block_text, "static")
                    elif idx == 1:
                        block["text"] = _compact_local_text(block_text, "semi_stable")
                    else:
                        block["text"] = _compact_local_text(block_text, "dynamic")
            elif isinstance(content, str):
                msg["content"] = _compact_local_text(content, "system")
            break

        compacted_chars = _estimate_message_chars(compacted)
        if compacted_chars <= target_chars:
            return compacted

        raise LocalContextTooLargeError(
            f"Local model context too large after safe compaction "
            f"({compacted_chars} chars > target {target_chars})."
        )

    def _chat_local(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        tool_choice: str,
        timeout: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send a chat request to the local llama-cpp-python server."""
        client = self._get_local_client()

        messages = self._normalize_system_message_placement(messages)
        clean_messages = self._strip_openrouter_roundtrip_metadata(
            self._copy_messages_with_cache_policy(
                messages,
                allow_message_cache_control=False,
                flatten_tool_content_blocks=True,
            )
        )
        # Local llama.cpp has no vision; avoid flattening base64 into the prompt.
        for msg in clean_messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for idx, block in enumerate(content):
                if isinstance(block, dict) and str(block.get("type") or "") in ("image_url", "image"):
                    content[idx] = {"type": "text", "text": "[image omitted: model has no vision]"}
        local_max = min(max_tokens, 2048)
        ctx_len = 0
        try:
            from ouroboros.local_model import get_manager
            ctx_len = get_manager().get_context_length()
            if ctx_len > 0:
                local_max = min(max_tokens, max(256, ctx_len // 4))
        except Exception:
            pass

        if ctx_len > 0:
            clean_messages = self._prepare_messages_for_local_context(clean_messages, ctx_len, local_max)

        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = "\n\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )

        clean_tools = None
        if tools:
            clean_tools = [
                {k: v for k, v in t.items() if k != "cache_control"}
                for t in tools
            ]

        kwargs: Dict[str, Any] = {
            "model": "local-model",
            "messages": clean_messages,
            "max_tokens": local_max,
        }
        if clean_tools:
            kwargs["tools"] = clean_tools
            kwargs["tool_choice"] = tool_choice
        if timeout and timeout > 0:
            kwargs["timeout"] = float(timeout)

        candidate = _physical_candidate(kwargs)
        local_target = {"provider": "local", "usage_model": "local-model"}
        # ONE physical attempt per call. Re-sending here spent the caller's
        # physical-attempt budget without the caller authorising it, so a
        # transient local failure now surfaces to the single retry policy that
        # owns the decision (``loop_llm_call.call_llm_with_retry``), which counts
        # the attempts it authorises.
        try:
            request = _attempt_request(local_target, candidate, source="llm.local")
            resp = _execute_candidate(
                request,
                lambda: client.chat.completions.create(**candidate),
                _candidate_before_dispatch(candidate, request),
            )
        except UsageAccountingError:
            raise
        except Exception as exc:
            err = str(exc)
            if (_is_structured_context_overflow_exception(exc)
                    or context_overflow_message(err)):
                raise LocalContextTooLargeError(err) from exc
            # Exception-owned capture proves this attempt; prior ContextVar may be unrelated.
            capture = getattr(exc, "physical_attempt_capture", None)
            if isinstance(capture, PhysicalAttemptCapture) and capture.state in {"dispatched", "unresolved"}:
                raise  # Outer custody owns an unknown physical outcome.
            log.warning("Local model request failed: %s", exc)
            raise

        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        if not msg.get("tool_calls") and msg.get("content") and clean_tools:
            allowed_tool_names = {
                str(t.get("function", {}).get("name", "")).strip()
                for t in clean_tools
                if isinstance(t, dict)
            }
            msg = self._parse_tool_calls_from_content(msg, allowed_tool_names)

        usage["cost"] = 0.0
        usage["cost_final"] = True
        # CPL6-F1: the same usage provenance every remote lane stamps — the
        # ledger row already carried provider=local, but a consumer reading the
        # returned usage alone could not attribute the call.
        usage["provider"] = "local"
        usage["resolved_model"] = "local-model"
        return msg, usage
