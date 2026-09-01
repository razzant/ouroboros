"""Local tool-call wire markup: DeepSeek DSML and legacy ``<tool_call>`` XML.

Some OpenAI-compatible routes (notably Relace) echo DeepSeek DSML tags in
``message.content`` while leaving native ``tool_calls`` empty, and local models
emit ``<tool_call>`` XML blocks.  This module is the single wire-format seam
for both: detection (``content_has_tool_markup``), reasoning-wrapper stripping,
and fail-closed promotion to native ``tool_calls`` (only well-formed markup is
upgraded; malformed markup stays plain content so the caller can classify it
as a protocol failure instead of a final answer).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

TOOL_MARKUP_PROTOCOL_FAIL_TEXT = (
    "⚠️ PROTOCOL_FAIL: the model returned tool-call markup that was not converted into native tool_calls."
)
TOOL_MARKUP_PROTOCOL_FAIL_NOTE = "Unparsed tool-call markup is a protocol failure, not a final answer."

# DeepSeek DSML tool-call wire tags.
_DSML_MARK = "\uff5cDSML\uff5c"
_DSML_TOOL_CALLS_OPEN = f"<{_DSML_MARK}tool_calls>"
_DSML_INVOKE_OPEN = f"<{_DSML_MARK}invoke"
_PLAIN_DSML_TOOL_CALLS_OPEN = "<tool_calls>"
_PLAIN_DSML_INVOKE_OPEN = "<invoke"
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call\b", re.IGNORECASE)
_DSML_INVOKE_RE = re.compile(
    rf"<(?P<invoke_mark>{_DSML_MARK})?invoke\s+name=\"([^\"]+)\"\s*>"
    rf"(.*?)</(?(invoke_mark){_DSML_MARK})invoke>",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    rf"<(?P<parameter_mark>{_DSML_MARK})?parameter\s+name=\"([^\"]+)\"([^>]*)>"
    rf"(.*?)</(?(parameter_mark){_DSML_MARK})parameter>",
    re.DOTALL,
)
_DSML_STRING_ATTR_RE = re.compile(r'\bstring\s*=\s*"(true|false)"', re.IGNORECASE)
_DSML_WRAPPER_RE = re.compile(rf"</?(?:{_DSML_MARK})?tool_calls>", re.IGNORECASE)
_DSML_WRAPPER_OPEN_RE = re.compile(rf"^\s*<(?P<wrapper_mark>{_DSML_MARK})?tool_calls>", re.IGNORECASE)
_DSML_TAG_RE = re.compile(
    rf"</?(?:{_DSML_MARK})?(?:tool_calls|invoke|parameter)\b",
    re.IGNORECASE,
)
_LOCAL_TOOL_CALL_FULL_RE = re.compile(
    r"^(?:\s*<tool_call>\s*\{.*?\}\s*</tool_call>\s*)+$",
    re.DOTALL,
)
_LOCAL_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def message_content_text(content: Any) -> str:
    """Flatten assistant ``content`` to the text a wire-format parser can read."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str) and block:
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def content_has_tool_markup(content: Any) -> bool:
    """True when content begins with a concrete tool-call wire envelope.

    Leading ``think``/``reasoning`` wrappers may precede the envelope.
    Ordinary prose or code examples that merely quote a tag remain content.
    """
    text = message_content_text(content)
    if not text:
        return False
    cut = tool_markup_start(text)
    if cut < 0:
        return False
    prefix = text[:cut].strip()
    for tag in ("think", "reasoning"):
        prefix = re.sub(
            rf"<{tag}>.*?</{tag}>",
            "",
            prefix,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()
    return not prefix


def tool_markup_start(text: str) -> int:
    """Return the earliest tool-markup offset in ``text``, or -1."""
    cuts: List[int] = []
    tool_call_start = _TOOL_CALL_TAG_RE.search(text)
    if tool_call_start:
        cuts.append(tool_call_start.start())
    dsml_idx = text.find(_DSML_TOOL_CALLS_OPEN)
    if dsml_idx < 0:
        dsml_idx = text.find(_DSML_INVOKE_OPEN)
    if dsml_idx < 0:
        dsml_idx = text.find(_PLAIN_DSML_TOOL_CALLS_OPEN)
    if dsml_idx < 0:
        dsml_idx = text.find(_PLAIN_DSML_INVOKE_OPEN)
    if dsml_idx >= 0:
        cuts.append(dsml_idx)
    return min(cuts) if cuts else -1


def strip_reasoning_wrappers(text: str) -> Tuple[str, str]:
    """Strip leading think/reasoning wrappers before the first tool markup."""
    # Split at first wire tool block so we never touch JSON inside payloads.
    cut = tool_markup_start(text)
    if cut >= 0:
        prefix = text[:cut]
        suffix = text[cut:]
    else:
        prefix = text
        suffix = ""

    reasoning_parts: list = []

    def _extract(tag: str, s: str) -> str:
        pattern = re.compile(
            r"<" + re.escape(tag) + r">(.*?)</" + re.escape(tag) + r">",
            re.DOTALL | re.IGNORECASE,
        )
        inner_texts = pattern.findall(s)
        reasoning_parts.extend(p.strip() for p in inner_texts if p.strip())
        return pattern.sub("", s)

    cleaned_prefix = _extract("think", prefix)
    cleaned_prefix = _extract("reasoning", cleaned_prefix)

    combined = (cleaned_prefix.strip() + ("\n" if cleaned_prefix.strip() and suffix else "") + suffix).strip()
    return combined, "\n\n".join(reasoning_parts)


def _parse_dsml_parameter_value(raw: str, attrs: str) -> Any:
    text = raw.strip()
    string_attr = _DSML_STRING_ATTR_RE.search(attrs or "")
    if string_attr and string_attr.group(1).lower() == "true":
        return text
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if string_attr and string_attr.group(1).lower() == "false":
            raise
        return text


def parse_dsml_tool_calls(
    text: str,
    allowed_tool_names: Optional[Set[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Return OpenAI-shaped tool_calls for well-formed DSML, else None."""
    if not text or not content_has_tool_markup(text):
        return None
    wrapper = _DSML_WRAPPER_OPEN_RE.match(text)
    if wrapper:
        expected_close = f"</{wrapper.group('wrapper_mark') or ''}tool_calls>"
        if not text.rstrip().lower().endswith(expected_close.lower()):
            return None
    invokes = list(_DSML_INVOKE_RE.finditer(text))
    if not invokes:
        return None
    allowed = {name for name in (allowed_tool_names or set()) if name}
    tool_calls: List[Dict[str, Any]] = []
    for index, match in enumerate(invokes):
        name = str(match.group(2) or "").strip()
        body = match.group(3) or ""
        if not name:
            return None
        if allowed and name not in allowed:
            return None
        arguments: Dict[str, Any] = {}
        for param in _DSML_PARAM_RE.finditer(body):
            key = str(param.group(2) or "").strip()
            if not key:
                return None
            try:
                arguments[key] = _parse_dsml_parameter_value(
                    param.group(4) or "",
                    param.group(3) or "",
                )
            except (json.JSONDecodeError, ValueError):
                return None
        leftover = _DSML_PARAM_RE.sub("", body).strip()
        if leftover:
            return None
        tool_calls.append({
            "id": f"call_dsml_{index}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        })
    remainder = _DSML_WRAPPER_RE.sub("", _DSML_INVOKE_RE.sub("", text)).strip()
    if _DSML_TAG_RE.search(remainder):
        return None
    return tool_calls or None


def parse_tool_calls_from_content(
    msg: Dict[str, Any],
    allowed_tool_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Parse local <tool_call> XML or well-formed DeepSeek DSML content."""
    content = message_content_text(msg.get("content"))
    stripped_raw = content.strip()
    if not stripped_raw:
        return msg

    # Only explicit reasoning wrappers are removed; arbitrary prose is left.
    stripped, reasoning = strip_reasoning_wrappers(stripped_raw)
    if not stripped:
        return msg

    # Upgrade only pure XML tool-call output; mixed prose stays plain text
    # unless the remainder is well-formed DSML (Relace often prefixes prose).
    if not _LOCAL_TOOL_CALL_FULL_RE.fullmatch(stripped):
        dsml_calls = parse_dsml_tool_calls(stripped, allowed_tool_names)
        if not dsml_calls:
            return msg
        msg = dict(msg)
        msg["tool_calls"] = dsml_calls
        remainder = _DSML_WRAPPER_RE.sub("", _DSML_INVOKE_RE.sub("", stripped)).strip()
        msg["content"] = remainder or reasoning or None
        log.info("Parsed %d DSML tool call(s) from content", len(dsml_calls))
        return msg

    matches = _LOCAL_TOOL_CALL_BLOCK_RE.findall(stripped)
    if not matches:
        return msg

    allowed = {name for name in (allowed_tool_names or set()) if name}
    tool_calls = []
    for i, raw in enumerate(matches):
        try:
            raw_stripped = raw.strip()
            try:
                obj = json.loads(raw_stripped)
            except json.JSONDecodeError:
                if raw_stripped.startswith("{{") and raw_stripped.endswith("}}"):
                    obj = json.loads(raw_stripped[1:-1])
                else:
                    raise
            if not isinstance(obj, dict):
                raise ValueError("tool_call payload must be an object")
            name = str(obj.get("name", "")).strip()
            args = obj.get("arguments", {})
            if not name:
                raise ValueError("tool_call missing function name")
            if allowed and name not in allowed:
                raise ValueError(f"unknown tool '{name}'")
            if not isinstance(args, dict):
                raise ValueError("tool_call arguments must be an object")
            tool_calls.append({
                "id": f"call_local_{i}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            })
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Rejected local <tool_call> block: %s (%s)", raw[:200], exc)
            return msg

    if not tool_calls:
        return msg

    msg = dict(msg)
    msg["tool_calls"] = tool_calls
    # Preserve reasoning text for loop progress; None/empty remains falsy.
    msg["content"] = reasoning or None
    log.info("Parsed %d local tool call(s) from text output", len(tool_calls))
    return msg


def promote_tool_markup(
    msg: Dict[str, Any],
    allowed_tool_names: Optional[Set[str]] = None,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Second-chance promotion of leftover markup to native ``tool_calls``.

    Returns ``(msg, tool_calls)`` with the parsed calls and the remainder
    content when the content is well-formed markup, or ``None`` when the
    markup is malformed — a protocol failure, never a final answer.
    """
    parsed = parse_tool_calls_from_content(
        {"content": msg.get("content"), "tool_calls": []},
        allowed_tool_names,
    )
    tool_calls = parsed.get("tool_calls") or []
    if not tool_calls:
        return None
    return dict(msg, tool_calls=tool_calls, content=parsed.get("content")), tool_calls


def tool_markup_protocol_fail(
    accumulated_usage: Dict[str, Any], llm_trace: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Stamp and return the typed protocol-failure finalization for malformed markup."""
    from ouroboros.outcomes import RESULT_INFRA_FAILED

    accumulated_usage.update(execution_status=RESULT_INFRA_FAILED, reason_code="protocol_fail")
    llm_trace["reasoning_notes"].append(TOOL_MARKUP_PROTOCOL_FAIL_NOTE)
    return TOOL_MARKUP_PROTOCOL_FAIL_TEXT, accumulated_usage, llm_trace


def resolve_tool_markup(
    msg: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    content: Any,
    accumulated_usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    tool_schemas: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    Any,
    Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]],
]:
    """Promote leftover markup or return its typed protocol failure."""
    if tool_calls or not content_has_tool_markup(content):
        return msg, tool_calls, content, None
    allowed: Set[str] = set()
    for schema in tool_schemas:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = (
            function.get("name")
            if isinstance(function, dict)
            else schema.get("name") if isinstance(schema, dict) else None
        )
        if isinstance(name, str) and name:
            allowed.add(name)
    promoted = promote_tool_markup(msg, allowed)
    if promoted is None:
        return msg, tool_calls, content, tool_markup_protocol_fail(
            accumulated_usage, llm_trace,
        )
    resolved_msg, resolved_calls = promoted
    return resolved_msg, resolved_calls, resolved_msg.get("content"), None
