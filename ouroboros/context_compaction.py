"""Checkpoint-bound, complete-input context-reclaim materialization.

Eligible units are completed assistant-call plus contiguous matching-result slices.
Owner turns and malformed, interrupted or visually opaque slices stay verbatim, and an
unfinished Anthropic native unit is ineligible. Selection stops once the predicted
reclaim reaches the goal. A non-empty selection first writes an exact private checkpoint,
then summarizes complete, gap-free hashed map/fold input into a labelled third-person
host record (user role), each original restorable from the checkpoint. The automatic
Main pass uses raw units first; typed provider refusal alone permits capsule re-folding.
Eligible units must appear completely in the persisted physical projection
(``exposed_context_units``); unknown exposure keeps a unit raw. ``goal_reached`` in the receipt reports whether the measured reclaim met the goal.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import pathlib
from dataclasses import asdict, replace
from collections import Counter
from typing import Any, Callable, Dict, List, Literal, Mapping, MutableSet, Optional, Sequence, Tuple

from ouroboros.context_budget import (
    HOST_CONTEXT_KIND_KEY,
    CONTEXT_OVERFLOW_CODES as _TYPED_CONTEXT_OVERFLOW_CODES,
    context_overflow_message as _context_overflow_message,
    ContextReclaimReceipt,
    ContextReclaimRequest,
    ReclaimStatus,
    SummarizerContextOverflow,
    _AtomicUnit,
    _Part,
    _SelectedUnit,
    _Selection,
    _UnitSummaryFailure,
    _UnsafeVisual,
)
from ouroboros.anthropic_native_custody import anthropic_tool_unit_active, custody_private_key
from ouroboros.config import runtime_setting
from ouroboros.tool_result_record import read_tool_result_record
from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY

log = logging.getLogger(__name__)

_CAPSULE_VERSION = 1
_SUMMARY_CONTRACT_VERSION = 1
_SUMMARY_OUTPUT_TOKENS = 32_768
_BLOCKS_PER_BATCH = 8
_MIN_CAPSULE_BYTES = 512
_SHA256_HEX = frozenset("0123456789abcdef")
_SUMMARY_GUIDANCE = (
    "Summarize each supplied context source without dropping late facts. Preserve "
    "the actor's hypotheses, exact consequential errors, tool inputs when they "
    "affect meaning, results, decisions, owner-visible constraints, and unresolved "
    "next steps. Every source is complete input: do not assume that a head excerpt "
    "represents its tail. Honor each source's summary_budget_tokens. Write a third-person "
    "host record, attributing the actor's judgments to Ouroboros; you are the summarization helper."
)

_CONTEXT_SUMMARIES_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_context_summaries",
        "description": "Emit exactly one non-empty summary for every supplied source_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "summaries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": ["source_id", "summary"],
                    },
                }
            },
            "required": ["summaries"],
        },
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _reclaim_tokens_for_byte_delta(byte_delta: int, measurement_density: float) -> int:
    density = float(measurement_density)
    if not math.isfinite(density) or density <= 0:
        raise ValueError("context reclaim measurement_density must be finite and positive")
    return max(0, math.ceil((max(0, int(byte_delta)) / 4) * density))


def _context_tokens_for_messages(
    messages: Sequence[Mapping[str, Any]], measurement_density: float,
) -> int:
    from ouroboros.context_fit import estimate_context_prompt_tokens
    _reclaim_tokens_for_byte_delta(0, measurement_density)
    estimate = estimate_context_prompt_tokens(list(messages))
    return max(0, math.ceil(estimate * float(measurement_density)))


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def context_reclaim_transcript_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(_canonical_bytes(list(messages)))


def _unique_strings(values: Sequence[Any]) -> Tuple[str, ...]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _unique_refs(values: Sequence[Any]) -> Tuple[Dict[str, Any], ...]:
    seen: set[str] = set()
    result: List[Dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping) or not value:
            continue
        normalized = dict(value)
        identity = _sha256(_canonical_bytes(normalized))
        if identity not in seen:
            seen.add(identity)
            result.append(normalized)
    return tuple(result)


def _summary_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        kind = str(value.get("type") or "").strip().lower()
        if kind in {"image", "image_url"}:
            descriptor = str(
                value.get("_caption")
                or value.get("caption")
                or value.get("alt")
                or value.get("text")
                or ""
            ).strip()
            if not descriptor:
                raise _UnsafeVisual("image/base64 content has no safe textual descriptor")
            return {"type": "image_descriptor", "text": descriptor}
        projected: Dict[str, Any] = {}
        for key, item in value.items():
            if str(key) == "_context_capsule" or custody_private_key(key):
                continue
            projected[str(key)] = _summary_projection(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [_summary_projection(item) for item in value]
    if isinstance(value, bytes):
        raise _UnsafeVisual("opaque byte content has no safe textual descriptor")
    if isinstance(value, str) and "data:image/" in value.lower():
        raise _UnsafeVisual("embedded image data has no safe textual descriptor")
    return value


def _capsule_metadata(message: Mapping[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    content = message.get("content")
    if str(message.get("role") or "") not in {"assistant", "user"} or not isinstance(content, list):
        return False, None
    blocks = [block for block in content
              if isinstance(block, Mapping) and "_context_capsule" in block]
    if not blocks:
        return False, None
    if len(content) != 1 or len(blocks) != 1 or message.get("tool_calls"):
        return True, None
    block = blocks[0]
    meta = block.get("_context_capsule")
    text = str(block.get("text") or "")
    if not isinstance(meta, Mapping) or not text.strip():
        return True, None
    try:
        generation = int(meta.get("generation") or 0)
        version = int(meta.get("version") or 0)
        source_length = int(meta.get("source_length_chars"))
        part_count = int(meta.get("part_count"))
        summary_contract_version = int(meta.get("summary_contract_version"))
    except (TypeError, ValueError):
        return True, None
    source_hashes, source_refs = meta.get("source_hashes"), meta.get("source_refs")
    parts, checkpoint_ref = meta.get("parts"), meta.get("checkpoint_ref")
    source_unit_sha256 = str(meta.get("source_unit_sha256") or "")
    summary_contract_digest = str(meta.get("summary_contract_digest") or "")
    valid = (
        str(block.get("type") or "") == "text"
        and version == _CAPSULE_VERSION
        and generation >= 1
        and str(meta.get("retention") or "") in {"summarized", "source_view"}
        and bool(str(meta.get("unit_id") or "").strip())
        and str(meta.get("visible_sha256") or "") == _sha256(text)
        and len(source_unit_sha256) == 64 and set(source_unit_sha256) <= _SHA256_HEX
        and isinstance(source_hashes, list) and bool(source_hashes)
        and source_unit_sha256 in source_hashes
        and all(isinstance(item, str) and len(item) == 64 and set(item) <= _SHA256_HEX
                for item in source_hashes)
        and isinstance(source_refs, list)
        and all(isinstance(item, Mapping) and item for item in source_refs)
        and isinstance(parts, list) and bool(parts)
        and part_count == len(parts)
        and source_length > 0
        and isinstance(checkpoint_ref, Mapping) and bool(checkpoint_ref)
        and any(dict(item) == dict(checkpoint_ref) for item in source_refs)
        and str(meta.get("measurement_basis") or "")
        in {"fresh_route_usage", "fresh_model_usage", "cold_estimate"}
        and summary_contract_version == _SUMMARY_CONTRACT_VERSION
        and summary_contract_digest in {_SUMMARY_CONTRACT_DIGEST, _LEGACY_SUMMARY_CONTRACT_DIGEST}
    )
    if not valid:
        return True, None
    expected_start = 0
    seen_source_ids: set[str] = set()
    for part in parts:
        if not isinstance(part, Mapping):
            return True, None
        try:
            start = int(part.get("start_char"))
            end = int(part.get("end_char"))
        except (TypeError, ValueError):
            return True, None
        source_id = str(part.get("source_id") or "")
        digest = str(part.get("sha256") or "")
        if (not source_id or source_id in seen_source_ids or start != expected_start
                or end <= start or len(digest) != 64 or not set(digest) <= _SHA256_HEX
                or digest not in source_hashes):
            return True, None
        seen_source_ids.add(source_id)
        expected_start = end
    return (True, dict(meta)) if expected_start == source_length else (True, None)


def _unit_from_slice(
    messages: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
    *,
    trace_refs_by_tool_call_id: Mapping[str, Any],
    capsule_meta: Optional[Mapping[str, Any]] = None,
    measurement_density: float = 1.0,
) -> Optional[_AtomicUnit]:
    raw_messages = list(messages[start:end + 1])
    raw_bytes = _canonical_bytes(raw_messages)
    try:
        source_text = _canonical_json(_summary_projection(raw_messages))
    except _UnsafeVisual:
        return None
    raw_sha = _sha256(raw_bytes)
    source_sha = _sha256(source_text)
    generation = int((capsule_meta or {}).get("generation") or 0)
    lineage_hashes: List[str] = list((capsule_meta or {}).get("source_hashes") or [])
    refs: List[Any] = list((capsule_meta or {}).get("source_refs") or [])
    if capsule_meta:
        refs.append(capsule_meta.get("checkpoint_ref"))
    for message in raw_messages:
        if message.get("role") == "tool":
            record = read_tool_result_record(message)
            refs.extend((record.get("trace_ref"), record.get("source_ref")))
        if review_ref := _review_source_ref(message):
            refs.append(review_ref)
    lineage_hashes.extend((raw_sha, source_sha))
    context_size_tokens = _context_tokens_for_messages(raw_messages, measurement_density)
    capsule_floor = _reclaim_tokens_for_byte_delta(_MIN_CAPSULE_BYTES, measurement_density)
    predicted = max(0, context_size_tokens - capsule_floor)
    return _AtomicUnit(
        unit_id=f"unit:{start}:{end}:{raw_sha[:16]}",
        start=start,
        end=end,
        raw_sha256=raw_sha,
        raw_size_bytes=len(raw_bytes),
        context_size_tokens=context_size_tokens,
        source_text=source_text,
        source_sha256=source_sha,
        predicted_reclaim_tokens=predicted,
        generation=generation,
        lineage_hashes=_unique_strings(lineage_hashes),
        source_refs=_unique_refs(refs),
    )


def _atomic_units(
    messages: Sequence[Mapping[str, Any]],
    *,
    trace_refs_by_tool_call_id: Optional[Mapping[str, Any]] = None,
    measurement_density: float = 1.0,
) -> Tuple[_AtomicUnit, ...]:
    """Find completed tool units and capsules; leave malformed units raw."""
    trace_refs = trace_refs_by_tool_call_id or {}
    units: List[_AtomicUnit] = []
    idx = 0
    while idx < len(messages):
        message = messages[idx]
        capsule_present, capsule_meta = _capsule_metadata(message)
        if capsule_present:
            if capsule_meta is not None:
                unit = _unit_from_slice(
                    messages, idx, idx,
                    trace_refs_by_tool_call_id=trace_refs,
                    capsule_meta=capsule_meta,
                    measurement_density=measurement_density,
                )
                if unit is not None:
                    units.append(unit)
            idx += 1
            continue

        calls = message.get("tool_calls") or []
        if (str(message.get("role") or "") != "assistant"
                or not isinstance(calls, list) or not calls
                or not all(isinstance(call, Mapping)
                           and isinstance(call.get("function"), Mapping)
                           and str(call["function"].get("name") or "").strip()
                           for call in calls)):
            idx += 1
            continue
        call_ids = [str((call or {}).get("id") or "") for call in calls]
        if not all(call_ids) or len(call_ids) != len(set(call_ids)):
            idx += 1
            continue
        end = idx
        results: List[Mapping[str, Any]] = []
        cursor = idx + 1
        while cursor < len(messages) and str(messages[cursor].get("role") or "") == "tool":
            results.append(messages[cursor])
            end = cursor
            cursor += 1
        result_ids = [str(result.get("tool_call_id") or "") for result in results]
        complete = (
            len(result_ids) == len(call_ids)
            and all(result_ids)
            and len(result_ids) == len(set(result_ids))
            and set(result_ids) == set(call_ids)
        )
        if complete and not anthropic_tool_unit_active(messages, idx, end):
            unit = _unit_from_slice(
                messages, idx, end,
                trace_refs_by_tool_call_id=trace_refs,
                measurement_density=measurement_density,
            )
            if unit is not None:
                units.append(unit)
        idx = max(idx + 1, end + 1)
    return tuple(units)


# Keys a plain dialogue row may carry and still be a complete prose unit. Any other
# key marks a host-typed row (acceptance observation, review feedback, native
# continuation, custody) that stays raw: typed meaning never gets flattened to prose.
_PLAIN_ROW_KEYS = frozenset({"role", "content", "name", "cache_control", HOST_CONTEXT_KIND_KEY})
_ASSISTANT_PROSE_KEYS = _PLAIN_ROW_KEYS | {"reasoning_content", "reasoning_details", "refusal", "annotations"}
UnitScope = Literal["tool", "dialogue"]
UnitKind = Literal["capsule", "tool", "assistant", "user"]


def _plain_text(content: Any) -> Optional[str]:
    """Text of a text-only content; ``None`` for native, image, capsule or opaque blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(
            isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str)
            and "_context_capsule" not in block and not any(custody_private_key(key) for key in block)
            for block in content):
        return "".join(block["text"] for block in content)
    return None


def _is_dialogue_prose_row(message: Any) -> bool:
    """A completed single prose row: the assistant's own reply or a host prose row. Never a
    row with tool protocol, native or non-text blocks, or host-typed keys. ``role=user`` is
    not authorship: owner words are excluded by callers through typed provenance."""
    if not isinstance(message, Mapping):
        return False
    role = str(message.get("role") or "")
    allowed = _ASSISTANT_PROSE_KEYS if role == "assistant" else _PLAIN_ROW_KEYS if role == "user" else frozenset()
    if _review_source_ref(message):
        allowed = allowed | {REVIEW_HISTORY_MESSAGE_KEY}
    if any(key not in allowed for key in message) or message.get("tool_calls") or message.get("function_call"):
        return False
    text = _plain_text(message.get("content"))
    return text is not None and bool(text.strip())


def _review_source_ref(message: Mapping[str, Any]) -> Optional[dict]:
    """A typed captured source, bound to the exact visible text of this row."""
    from ouroboros.review_history_view import _binding

    meta, text = message.get(REVIEW_HISTORY_MESSAGE_KEY), message.get("content")
    if (message.get("role") != "user" or not isinstance(meta, Mapping) or meta.get("version") != 1
            or not isinstance(text, str) or not meta.get("task_id") or meta.get("visible_sha256") != _sha256(text)):
        return None
    try:
        return _binding(meta.get("binding"))
    except (TypeError, ValueError):
        return None


def context_units(
    messages: Sequence[Mapping[str, Any]],
    *,
    scope: UnitScope = "tool",
    trace_refs_by_tool_call_id: Optional[Mapping[str, Any]] = None,
    measurement_density: float = 1.0,
) -> Tuple[_AtomicUnit, ...]:
    """Units a view may address, in transcript order: ``tool`` is the automatic and
    count-only reader (complete tool units and capsules); ``dialogue`` adds completed
    assistant and host prose rows for the explicit view, inspection, exposure and
    restore. Ids are positional and hash-bound, identical in both scopes."""
    trace_refs = trace_refs_by_tool_call_id or {}
    units = list(_atomic_units(messages, trace_refs_by_tool_call_id=trace_refs,
                               measurement_density=measurement_density))
    if scope == "dialogue":
        covered = {index for unit in units for index in range(unit.start, unit.end + 1)}
        # The system row and the assignment (first user row) are never units.
        covered.add(next((i for i, m in enumerate(messages) if str(m.get("role") or "") == "user"), -1))
        units.extend(unit for unit in (
            _unit_from_slice(messages, idx, idx, trace_refs_by_tool_call_id=trace_refs,
                             measurement_density=measurement_density)
            for idx, message in enumerate(messages) if idx not in covered and _is_dialogue_prose_row(message))
            if unit is not None)
        units.sort(key=lambda unit: unit.start)
    return tuple(units)


def unit_kind(messages: Sequence[Mapping[str, Any]], unit: _AtomicUnit) -> UnitKind:
    message = messages[unit.start] if 0 <= unit.start < len(messages) else {}
    if _capsule_metadata(message)[0]:
        return "capsule"
    if message.get("tool_calls"):
        return "tool"
    return "assistant" if str(message.get("role") or "") == "assistant" else "user"


def owner_protected_unit_ids(
    messages: Sequence[Mapping[str, Any]], units: Sequence[_AtomicUnit], protected_texts: Sequence[str],
) -> frozenset:
    """Keep governing words and typed outward dialogue whole, without promoting
    the actor's speech to owner authority. A mixed owner row stays whole."""
    texts = [text for text in protected_texts if text]
    return frozenset(
        unit.unit_id for unit in units
        if messages[unit.start].get(HOST_CONTEXT_KIND_KEY) == "owner_dialogue"
        or (texts and unit_kind(messages, unit) == "user"
            and any(text in (_plain_text(messages[unit.start].get("content")) or "") for text in texts)))


def _source_atoms(messages: Sequence[Mapping[str, Any]]) -> Counter:
    """Content identities across function/custom and native tool-result syntax.

    The transport may change JSON syntax and coalesce text blocks, but a call's
    name/arguments and its whole result still have to be present. IDs alone are
    never exposure evidence; counts also distinguish repeated identical parts.
    """
    atoms: Counter = Counter()
    calls_by_id: Dict[str, tuple] = {}
    calls_by_name: Dict[str, tuple] = {}
    last_call: tuple = ("", None)

    def add(kind: str, *values: Any) -> None:
        atoms[_sha256(_canonical_bytes((kind, *values)))] += 1

    def arguments(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                pass
        return value

    def content(value: Any) -> Any:
        if isinstance(value, list) and all(isinstance(p, Mapping) and p.get("type") == "text" for p in value):
            return "".join(str(p.get("text") or "") for p in value)
        return value

    def result_content(value: Any) -> Any:
        value = content(value) or "(no tool output)"
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return {"result": value}  # native function-result JSON envelope
        return value

    for message in messages:
        role, body = message.get("role"), message.get("content")
        if role in {"tool", "function"}:
            call = calls_by_id.get(str(message.get("tool_call_id") or "")) or calls_by_name.get(str(message.get("name") or "")) or last_call
            add("result", *call, result_content(body))
            continue
        blocks = body if isinstance(body, list) else [{"type": "text", "text": body}]
        text = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, Mapping))
        if role == "assistant" and text:
            add("text", text)
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            if role == "user" and block.get("type") == "text" and block.get("text"):
                add("host_text", str(block["text"]))
            if block.get("type") == "tool_use":
                last_call = (str(block.get("name") or ""), arguments(block.get("input")))
                calls_by_id[str(block.get("id") or "")] = last_call
                calls_by_name[last_call[0]] = last_call
                add("call", *last_call)
            elif block.get("type") == "tool_result":
                call = calls_by_id.get(str(block.get("tool_use_id") or "")) or last_call
                add("result", *call, result_content(block.get("content")))
        if role == "assistant" and message.get("reasoning_content"):
            add("reasoning", message["reasoning_content"])
        calls = message.get("tool_calls") or []
        if message.get("function_call"):
            calls = [{"function": message["function_call"]}]
        for call in calls:
            function = call.get("function") or call.get("custom") or {}
            last_call = (str(function.get("name") or ""), arguments(function.get("arguments", function.get("input"))))
            calls_by_id[str(call.get("id") or "")] = last_call
            calls_by_name[last_call[0]] = last_call
            add("call", *last_call)
    return atoms


def exposed_context_units(messages: list, physical_messages: list) -> Tuple[Dict[str, str], ...]:
    """Exact canonical source identities whose complete contents reached the wire."""
    return exposed_context_units_from_atoms(messages, _source_atoms(physical_messages))


def exposed_context_units_from_atoms(
    messages: Sequence[Mapping[str, Any]], atoms: Mapping[str, int],
) -> Tuple[Dict[str, str], ...]:
    """Exposure against the prepared candidate's content identities (``_source_atoms``),
    recorded by the send path BEFORE observability redaction (hashes and counts only), so a
    unit whose result carries a redacted secret is still recognized. Dialogue scope."""
    remaining: Counter = Counter({str(key): int(count) for key, count in dict(atoms).items()})
    exposed = []
    for unit in context_units(messages, scope="dialogue"):
        unit_atoms = _source_atoms(messages[unit.start:unit.end + 1])
        if unit_atoms and all(remaining[key] >= count for key, count in unit_atoms.items()):
            remaining.subtract(unit_atoms)
            exposed.append({"unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256})
    return tuple(exposed)


def _typed_context_overflow(exc: BaseException) -> bool:
    try:
        from ouroboros.llm import LocalContextTooLargeError

        if isinstance(exc, LocalContextTooLargeError):
            return True
    except Exception:
        pass
    candidates = [getattr(exc, key, None) for key in ("kind", "code", "type")]
    body = getattr(exc, "body", None)
    payloads = [body] if isinstance(body, Mapping) else []
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            response_body = response.json()
        except Exception:
            response_body = None
        if isinstance(response_body, Mapping):
            payloads.append(response_body)
    for payload in payloads:
        candidates.extend(payload.get(key) for key in ("kind", "code", "type"))
        nested = payload.get("error")
        if isinstance(nested, Mapping):
            candidates.extend(nested.get(key) for key in ("kind", "code", "type"))
    capture = getattr(exc, "physical_attempt_capture", None)
    candidates.extend((getattr(capture, "provider_code", None),
                       getattr(capture, "provider_error_type", None)))
    if any(str(value or "").strip().lower() in _TYPED_CONTEXT_OVERFLOW_CODES for value in candidates):
        return True
    return _context_overflow_message(str(exc))


def _record_usage(total: Dict[str, Any], usage: Mapping[str, Any]) -> None:
    from ouroboros.llm import add_usage

    add_usage(total, dict(usage))
    for key in ("provider", "model", "prompt_cache_ttl"):
        if usage.get(key) is not None:
            total[key] = usage[key]


def _summarizer_spec() -> Dict[str, Any]:
    from ouroboros.config import get_light_model

    model = str(get_light_model() or "")
    use_local = runtime_setting("USE_LOCAL_LIGHT", "").strip().lower() in {"1", "true", "yes", "on"}
    route: Dict[str, Any] = {"model": model, "use_local": use_local}
    if use_local:
        route.update({"provider": "local", "resolved_model": model})
    else:
        try:
            from ouroboros.llm import LLMClient

            target = LLMClient()._resolve_remote_target(model)
            route.update({"provider": str(target.get("provider") or ""),
                          "resolved_model": str(target.get("usage_model") or target.get("resolved_model") or model),
                          "base_url": str(target.get("base_url") or "")})
        except Exception:
            route.update({"provider": "unknown", "resolved_model": model, "base_url": ""})
    route["route_fp"] = _sha256(_canonical_bytes(route))
    route["effort"] = "low"
    from ouroboros.response_limits import response_allowance
    route["output_budget"] = response_allowance(model, _SUMMARY_OUTPUT_TOKENS, use_local=use_local,
                                                model_role="light", allow_fetch=True)
    return route


def _output_budget(spec: Dict[str, Any]) -> int:
    """One summarizer send's allowance; a spec without a known maximum keeps the shipped budget."""
    return int(spec.get("output_budget") or _SUMMARY_OUTPUT_TOKENS)


_SUMMARY_CONTRACT_DIGEST = _sha256(_canonical_bytes({
    "version": _SUMMARY_CONTRACT_VERSION, "guidance": _SUMMARY_GUIDANCE,
    "tool": _CONTEXT_SUMMARIES_TOOL,
    "input_fields": ("source_id", "start_char", "end_char", "sha256",
                     "summary_budget_tokens", "content"),
}))
# The previous serialized capsule contract remains readable after attribution changes.
_LEGACY_SUMMARY_CONTRACT_DIGEST = "6bd892aa2479a9c8b65036b0c18e53ccc076cb5bf5a0d87e351ea38182fa3402"


def _summary_map_from_entries(entries: Any) -> Dict[str, str]:
    if not isinstance(entries, list):
        return {}
    result: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            return {}
        source_id = str(entry.get("source_id") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        if not source_id or not summary or source_id in result:
            return {}
        result[source_id] = summary
    return result


def _parse_structured_summaries(message: Mapping[str, Any]) -> Dict[str, str]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
        return {}
    function = calls[0].get("function") or {}
    if not isinstance(function, Mapping) or str(function.get("name") or "") != "emit_context_summaries":
        return {}
    try:
        payload = json.loads(function.get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return _summary_map_from_entries(payload.get("summaries"))


def _parse_json_summaries(content: Any) -> Dict[str, str]:
    try:
        payload = json.loads(str(content or ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return _summary_map_from_entries(payload.get("summaries") if isinstance(payload, Mapping) else None)


def _call_summarizer(
    parts: Sequence[_Part],
    *,
    drive_root: pathlib.Path,
    task_id: str,
    phase: Literal["map", "fold"],
    spec: Mapping[str, Any],
    summary_budgets: Mapping[str, int],
    usage_total: Dict[str, Any],
) -> Dict[str, str]:
    """Summarize complete parts. Typed overflow is never retried unchanged."""
    from ouroboros.llm import LLMClient
    from ouroboros.llm_observability import chat_observed
    payload = [
        {
            "source_id": part.source_id,
            "start_char": part.start_char,
            "end_char": part.end_char,
            "sha256": part.sha256,
            "summary_budget_tokens": int(summary_budgets[part.root_id]),
            "content": part.text,
        }
        for part in parts
    ]
    source_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    client = LLMClient()
    use_local = bool(spec.get("use_local"))
    common = {
        "drive_root": drive_root,
        "task_id": task_id,
        "call_type": f"context_compaction_{phase}",
        "model_role": "light",
        "model": str(spec.get("model") or ""),
        "reasoning_effort": str(spec.get("effort") or "low"),
        "max_tokens": _output_budget(spec),
        "use_local": use_local,
        # One sticky session per data root: without it a system-less prompt gets
        # no OpenRouter session_id/OpenAI key, so sibling map/fold calls may land
        # on different upstream caches.
        "cache_affinity": "" if use_local else f"context_compaction:{drive_root}",
    }

    if not use_local:
        prompt = (_SUMMARY_GUIDANCE
                  + "\nCall emit_context_summaries exactly once, with one entry for every source_id.\n"
                  + source_json)
        try:
            from ouroboros.openai_chat_dispatch import call_with_custom_validation_continuation
            message, observed_usage, executable = call_with_custom_validation_continuation(
                lambda request_messages: chat_observed(
                    client, messages=request_messages,
                    tools=[_CONTEXT_SUMMARIES_TOOL], tool_choice="required", **common,
                ),
                [{"role": "user", "content": prompt}],
            )
            for _usage in observed_usage:
                _record_usage(usage_total, _usage)
            from ouroboros._usage_response import OUTPUT_LIMIT_FINISH_REASONS, response_finish_reason
            if observed_usage and response_finish_reason(observed_usage[-1], message)[1] in OUTPUT_LIMIT_FINISH_REASONS:
                return {}
            if executable:
                parsed = _parse_structured_summaries(message)
                if parsed:
                    return parsed
        except Exception as exc:
            if _typed_context_overflow(exc):
                raise SummarizerContextOverflow(str(exc)) from exc
            from ouroboros.llm_claudexor import propagate_model_error
            propagate_model_error(exc)
            log.warning("Structured context summary failed; trying JSON response", exc_info=True)

    prompt = (_SUMMARY_GUIDANCE
              + "\nReturn only a JSON object {\"summaries\":[{\"source_id\":...,\"summary\":...}]}, "
                "with one entry for every source_id.\n"
              + source_json)
    try:
        message, _usage = chat_observed(
            client,
            messages=[{"role": "user", "content": prompt}],
            **common,
        )
        _record_usage(usage_total, _usage)
    except Exception as exc:
        if _typed_context_overflow(exc):
            raise SummarizerContextOverflow(str(exc)) from exc
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(exc)
        raise _UnitSummaryFailure(
            f"summary call failed: {type(exc).__name__}: {exc}",
        ) from exc
    from ouroboros._usage_response import OUTPUT_LIMIT_FINISH_REASONS, response_finish_reason
    if response_finish_reason(_usage, message)[1] in OUTPUT_LIMIT_FINISH_REASONS:
        return {}
    return _parse_json_summaries(message.get("content"))


def _part(source_id: str, text: str, start: int = 0) -> _Part:
    end = start + len(text)
    return _Part(root_id=source_id,
                 source_id=f"{source_id}:{start}:{end}:{_sha256(text)[:12]}",
                 start_char=start, end_char=end, text=text, sha256=_sha256(text))


def _split_part(part: _Part) -> Optional[Tuple[_Part, _Part]]:
    if len(part.text) < 2:
        return None
    midpoint = len(part.text) // 2
    # Newline boundary only NEAR the midpoint: a lone newline near an edge
    # used to produce degenerate 1%/99% splits that re-overflowed unchanged.
    window = max(1, len(part.text) // 4)
    before = part.text.rfind("\n", 1, midpoint + 1)
    after = part.text.find("\n", midpoint, len(part.text) - 1)
    candidates = [pos + 1 for pos in (before, after)
                  if pos > 0 and abs((pos + 1) - midpoint) <= window]
    split_at = min(candidates, key=lambda pos: abs(pos - midpoint)) if candidates else midpoint
    if split_at <= 0 or split_at >= len(part.text):
        split_at = midpoint
    if split_at <= 0 or split_at >= len(part.text):
        return None
    left_text = part.text[:split_at]
    right_text = part.text[split_at:]
    return (_part(part.root_id, left_text, part.start_char),
            _part(part.root_id, right_text, part.start_char + split_at))


def _map_complete_parts(
    parts: Sequence[_Part],
    *,
    drive_root: pathlib.Path,
    task_id: str,
    spec: Mapping[str, Any],
    summary_budgets: Mapping[str, int],
    usage_total: Dict[str, Any],
) -> Tuple[Tuple[_Part, ...], Dict[str, str], set[str]]:
    try:
        summaries = _call_summarizer(
            parts,
            drive_root=drive_root,
            task_id=task_id,
            phase="map",
            spec=spec,
            summary_budgets=summary_budgets,
            usage_total=usage_total,
        )
    except SummarizerContextOverflow:
        if len(parts) > 1:
            middle = len(parts) // 2
            subsets: Sequence[Sequence[_Part]] = (parts[:middle], parts[middle:])
        else:
            split = _split_part(parts[0])
            if split is None:
                return (), {}, {parts[0].root_id}
            # One call PER half: both halves together equal the failed payload.
            subsets = ((split[0],), (split[1],))
        leaves: List[_Part] = []
        merged: Dict[str, str] = {}
        failed: set[str] = set()
        for subset in subsets:
            child_leaves, child_map, child_failed = _map_complete_parts(
                subset,
                drive_root=drive_root,
                task_id=task_id,
                spec=spec,
                summary_budgets=summary_budgets,
                usage_total=usage_total,
            )
            leaves.extend(child_leaves)
            merged.update(child_map)
            failed.update(child_failed)
        return tuple(leaves), merged, failed
    except _UnitSummaryFailure:
        return (), {}, {part.root_id for part in parts}
    expected = {part.source_id: part.root_id for part in parts}
    if set(summaries) - set(expected):
        return tuple(parts), {}, set(expected.values())
    missing = set(expected) - set(summaries)
    failed = {expected[source_id] for source_id in missing}
    return tuple(parts), summaries, failed


def _fold_summaries(
    leaves: Sequence[_Part],
    summaries: Mapping[str, str],
    *,
    drive_root: pathlib.Path,
    task_id: str,
    spec: Mapping[str, Any],
    summary_budget_tokens: int,
    usage_total: Dict[str, Any],
) -> str:
    nodes = [(leaf.source_id, summaries[leaf.source_id]) for leaf in leaves]
    while len(nodes) > 1:
        next_nodes: List[Tuple[str, str]] = []
        for offset in range(0, len(nodes), _BLOCKS_PER_BATCH):
            group = nodes[offset:offset + _BLOCKS_PER_BATCH]
            if len(group) == 1:
                next_nodes.append(group[0])
                continue
            labelled = "\n\n".join(f"[{source_id}]\n{summary}" for source_id, summary in group)
            fold_id = "fold:" + _sha256(labelled)[:16]
            fold_part = _part(fold_id, labelled)
            try:
                folded = _call_summarizer(
                    [fold_part],
                    drive_root=drive_root,
                    task_id=task_id,
                    phase="fold",
                    spec=spec,
                    summary_budgets={fold_part.root_id: summary_budget_tokens},
                    usage_total=usage_total,
                )
                text = str(folded.get(fold_part.source_id) or "").strip() \
                    if set(folded) == {fold_part.source_id} else ""
            except (SummarizerContextOverflow, _UnitSummaryFailure):
                text = ""
            next_nodes.append((fold_id, text or labelled))
        nodes = next_nodes
    return nodes[0][1]


def _negative_memo_key(
    unit: _AtomicUnit,
    *,
    summary_budget_tokens: int,
    spec: Mapping[str, Any],
) -> str:
    return _sha256(_canonical_bytes({
        "unit_raw_sha256": unit.raw_sha256,
        "unit_source_sha256": unit.source_sha256,
        "lineage_hashes": unit.lineage_hashes,
        "source_ref_hashes": [_sha256(_canonical_bytes(ref)) for ref in unit.source_refs],
        "requested_unit_summary_budget": summary_budget_tokens,
        "summarizer_route_fp": str(spec.get("route_fp") or ""),
        "summarizer_model": str(spec.get("resolved_model") or spec.get("model") or ""),
        "summarizer_effort": str(spec.get("effort") or ""),
        "summarizer_output_budget": int(spec.get("output_budget") or 0),
        "summary_contract_digest": _SUMMARY_CONTRACT_DIGEST,
        "summary_contract_version": _SUMMARY_CONTRACT_VERSION,
    }))


def _select_units(
    messages: Sequence[Mapping[str, Any]],
    request: ContextReclaimRequest,
    *,
    keep_recent: int,
    trace_refs_by_tool_call_id: Mapping[str, Any],
    negative_memo: MutableSet[str],
    spec: Mapping[str, Any],
    exposed_units: Optional[Sequence[Mapping[str, Any]]] = None,
    automatic: bool = False,
    provider_refused: bool = False,
) -> Tuple[Optional[_Selection], ReclaimStatus]:
    units = list(_atomic_units(
        messages, trace_refs_by_tool_call_id=trace_refs_by_tool_call_id,
        measurement_density=request.measurement_density))
    # Earlier records are residue, not fresh sources for another helper retelling. Only the
    # provider's own refusal of this request lets an automatic pass re-fold them, as the last
    # resort after every raw source. Keep the general unit reader broad for explicit
    # authored views and restore.
    if automatic:
        units = ([unit for unit in units if unit.generation == 0]
                 + ([unit for unit in units if unit.generation] if provider_refused else []))
    if exposed_units is not None or automatic:
        exposed = {(ref.get("unit_id"), ref.get("raw_sha256")) for ref in (exposed_units or ())}
        units = [unit for unit in units if (unit.unit_id, unit.raw_sha256) in exposed]
    if keep_recent > 0:
        units = units[:-keep_recent] if len(units) > keep_recent else []
    if not units:
        return None, "no_eligible"
    if int(request.reclaim_goal_tokens) <= 0:
        return None, "no_positive_reclaim"
    selected: List[_SelectedUnit] = []
    predicted = 0
    for unit in units:
        if unit.predicted_reclaim_tokens <= 0:
            continue
        source_tokens = max(1, len(unit.source_text.encode("utf-8")) // 4)
        summary_budget = min(max(1, _output_budget(spec) - 128), max(
            512, min(source_tokens // 2, max(512, int(request.reclaim_goal_tokens))),
        ))
        memo_key = _negative_memo_key(unit, summary_budget_tokens=summary_budget, spec=spec)
        if memo_key in negative_memo:
            continue
        selected.append(_SelectedUnit(unit, summary_budget, memo_key))
        predicted += unit.predicted_reclaim_tokens
        if predicted >= int(request.reclaim_goal_tokens):
            break
    if not selected or (not request.allow_partial_shrink and predicted < int(request.reclaim_goal_tokens)):
        return None, "no_positive_reclaim"
    fingerprint = _sha256(_canonical_bytes({
        "request": {
            "route_fp": request.route_fp,
            "round_id": request.round_id,
            "transcript_sha256": request.transcript_sha256,
            "measurement_basis": request.measurement_basis,
            "measurement_density": float(request.measurement_density),
            "reclaim_goal_tokens": int(request.reclaim_goal_tokens),
            "allow_partial_shrink": bool(request.allow_partial_shrink),
        },
        "units": [
            {
                "unit_id": item.unit.unit_id,
                "raw_sha256": item.unit.raw_sha256,
                "memo_key": item.negative_memo_key,
            }
            for item in selected
        ],
    }))
    return _Selection(tuple(selected), fingerprint, predicted), "applied"


def _persist_reclaim_checkpoint(
    messages: Sequence[Mapping[str, Any]],
    request: ContextReclaimRequest,
    selection: _Selection,
    *,
    drive_root: pathlib.Path,
    task_id: str,
) -> Optional[Dict[str, Any]]:
    from ouroboros.observability import new_call_id, persist_call
    payload = {"messages": list(messages), "request": asdict(request),
        "observed_view_revision": context_reclaim_transcript_sha256(messages),
        "selection_fingerprint": selection.fingerprint, "selected_unit_ids": [item.unit.unit_id for item in selection.units],
    }
    try:
        persisted = persist_call(
            pathlib.Path(drive_root).resolve(strict=False),
            task_id=str(task_id or "context_compaction"),
            call_id=new_call_id("context_reclaim_checkpoint"),
            call_type="context_reclaim_checkpoint",
            payload=payload,
            manifest={
                "route_fp": request.route_fp, "round_id": request.round_id,
                "selection_fingerprint": selection.fingerprint, "selected_unit_ids": [item.unit.unit_id for item in selection.units],
            },
            keep_raw=True,
        )
    except Exception:
        log.debug("Failed to persist selected context-reclaim checkpoint", exc_info=True)
        return None
    if not isinstance(persisted, Mapping) or not persisted.get("manifest_ref"): return None
    try:
        from ouroboros.artifacts import store_actor_source_bytes
        return store_actor_source_bytes(
            pathlib.Path(drive_root), str(task_id or "context_compaction"),
            category="context_checkpoints", source_id=selection.fingerprint,
            data=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            extension="json",
        )
    except Exception:
        log.debug("Failed to persist actor-readable context-reclaim checkpoint", exc_info=True)
        return None


def _capsule_message(
    selected: _SelectedUnit,
    summary: str,
    parts: Sequence[_Part],
    checkpoint_ref: Mapping[str, Any],
    request: ContextReclaimRequest,
    *, retention: str = "summarized", label: Optional[str] = None,
    address: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    unit = selected.unit
    generation = unit.generation + 1
    if retention == "source_view":
        label = label or ("Historical source: read-only projection, not live assistant/tool turns; "
                          "private transport data omitted; exact original retained by checkpoint")
        address = dict(address) if address is not None else {
            "checkpoint_ref": dict(checkpoint_ref), "unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256}
        text = f"[{label}]\nSource reference: {_canonical_json(address)}\n" + str(summary or "").strip()
    else:
        label = label or f"Host memory record from a summarization helper, generation {generation}; exact source retained by checkpoint"
        text = f"[{label}]\n" + str(summary or "").strip()
    source_hashes = _unique_strings([
        *unit.lineage_hashes, unit.raw_sha256, unit.source_sha256,
        *(part.sha256 for part in parts),
    ])
    source_refs = _unique_refs([*unit.source_refs, dict(checkpoint_ref)])
    metadata: Dict[str, Any] = {
        "version": _CAPSULE_VERSION,
        "generation": generation,
        "retention": retention,
        "unit_id": unit.unit_id,
        "source_unit_sha256": unit.raw_sha256,
        "source_length_chars": len(unit.source_text),
        "source_hashes": list(source_hashes),
        "source_refs": list(source_refs),
        "checkpoint_ref": dict(checkpoint_ref),
        "parts": [{
            "source_id": part.source_id, "start_char": part.start_char,
            "end_char": part.end_char, "sha256": part.sha256,
        } for part in parts],
        "part_count": len(parts),
        "route_fp": request.route_fp,
        "round_id": request.round_id,
        "measurement_basis": request.measurement_basis,
        "summary_contract_version": _SUMMARY_CONTRACT_VERSION,
        "summary_contract_digest": _SUMMARY_CONTRACT_DIGEST,
        "visible_sha256": _sha256(text),
        "authorship": "host" if retention == "source_view" else "helper",
    }
    message = {"role": "user", "content": [{
        "type": "text", "text": text, "_context_capsule": metadata,
    }]}
    capsule_ref = {
        "unit_id": unit.unit_id,
        "generation": generation,
        "checkpoint_ref": dict(checkpoint_ref),
        "source_hashes": list(source_hashes),
        "source_refs": list(source_refs),
    }
    return message, capsule_ref


def _receipt(
    status: ReclaimStatus,
    *,
    before_sha: str,
    after_sha: Optional[str] = None,
    selection: Optional[_Selection] = None,
    reclaimed_tokens: int = 0,
    goal_reached: bool = False,
    checkpoint_ref: Optional[Dict[str, Any]] = None,
    capsule_refs: Sequence[Dict[str, Any]] = (),
    **view_facts: Any,
) -> ContextReclaimReceipt:
    return ContextReclaimReceipt(
        status=status,
        before_transcript_sha256=before_sha,
        after_transcript_sha256=after_sha or before_sha,
        selection_fingerprint=selection.fingerprint if selection else "",
        selected_unit_ids=tuple(item.unit.unit_id for item in selection.units) if selection else (),
        reclaimed_tokens=max(0, int(reclaimed_tokens)),
        goal_reached=bool(goal_reached),
        checkpoint_ref=checkpoint_ref,
        capsule_refs=tuple(dict(item) for item in capsule_refs),
        **view_facts,
    )


def _view_source_messages(messages: Sequence[Mapping[str, Any]]) -> list:
    """Compare source while ignoring only the host's cache presentation."""
    normalized = copy.deepcopy(list(messages))
    first_user = next((i for i, m in enumerate(normalized) if m.get("role") == "user"), None)
    for idx, message in enumerate(normalized):
        message.pop("cache_control", None)
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
            if ((message.get("role") == "tool" or idx == first_user) and len(content) == 1
                    and isinstance(content[0], dict) and set(content[0]) == {"type", "text"}
                    and content[0]["type"] == "text" and isinstance(content[0]["text"], str)):
                message["content"] = content[0]["text"]
    return normalized


def _materialize_replacements(messages: list, replacements: Mapping[int, tuple]) -> tuple[list, list]:
    """Publish complete units at their existing positions; preserve every other turn."""
    rebuilt, capsule_refs = [], []
    pending = dict(replacements)
    idx = 0
    while idx <= len(messages):
        replacement = pending.pop(idx, None)
        if replacement is not None:
            end, added, refs = replacement
            rebuilt.extend(added)
            capsule_refs.extend(refs)
            idx = end + 1
        elif idx < len(messages):
            rebuilt.append(messages[idx])
            idx += 1
        else:
            break
    return rebuilt, capsule_refs


def _restored_source_views(refs: Sequence[Mapping[str, Any]], *, drive_root: pathlib.Path,
                           task_id: str, request: ContextReclaimRequest) -> tuple[list, list]:
    """Retrieve exact checkpoint-local units as source text, never live protocol turns."""
    from ouroboros.artifacts import read_actor_source_bytes

    messages, capsule_refs = [], []
    for ref in _unique_refs(refs):
        checkpoint = ref.get("checkpoint_ref")
        payload = json.loads(read_actor_source_bytes(drive_root, task_id, checkpoint))
        original = payload.get("messages") if isinstance(payload, Mapping) else None
        if not isinstance(original, list) or not all(isinstance(m, Mapping) for m in original):
            raise ValueError("checkpoint has no complete message source")
        unit = next((u for u in context_units(original, scope="dialogue")
                     if u.unit_id == ref.get("unit_id") and u.raw_sha256 == ref.get("raw_sha256")), None)
        if unit is None:
            raise ValueError("checkpoint unit identity or full raw hash does not match")
        unit = replace(unit, source_refs=_unique_refs([*unit.source_refs, ref]))
        message, capsule_ref = _capsule_message(
            _SelectedUnit(unit, 0, ""), unit.source_text, [_part(unit.unit_id, unit.source_text)],
            checkpoint, request, retention="source_view")
        messages.append(message)
        capsule_refs.append(capsule_ref)
    return messages, capsule_refs


def _authored_view(
    messages: list, request: ContextReclaimRequest, *, observed_messages: Optional[Sequence[Mapping[str, Any]]],
    observed_tool_schemas: Optional[Sequence[Mapping[str, Any]]],
    tool_schemas: Sequence[Mapping[str, Any]], fit_candidate: Optional[Callable[[list, list], Mapping[str, Any]]],
    drive_root: pathlib.Path, task_id: str, trace_refs: Mapping[str, Any],
    exposed_units: Optional[Sequence[Mapping[str, Any]]] = None,
    protected_texts: Sequence[str] = (),
) -> Tuple[list, ContextReclaimReceipt, None]:
    """Materialize one actor's selection through the existing unit/checkpoint/capsule engine
    over the dialogue scope (tool units, capsules, the actor's own completed replies, host
    prose). Rows carrying the owner's typed words (``protected_texts``) stay whole."""
    restore_refs = []
    for ref in request.restore_unit_refs:
        checkpoint = ref.get("checkpoint_ref") if isinstance(ref, Mapping) else None
        if (isinstance(checkpoint, Mapping) and set(checkpoint) == {"kind", "root", "path", "size", "sha256"}
                and checkpoint.get("kind") == "task_source" and checkpoint.get("root") == "artifact_store"):
            # Inspection omits the repeated reader hint. Restore still verifies
            # exact source bytes below; published custody keeps the full contract.
            checkpoint = {**checkpoint, "read": {"tool": "read_file", "arguments": {
                "root": checkpoint["root"], "path": checkpoint["path"]}}}
            ref = {**ref, "checkpoint_ref": checkpoint}
        restore_refs.append(ref)
    request = replace(request, restore_unit_refs=tuple(restore_refs))
    before_sha = context_reclaim_transcript_sha256(messages)
    observed = copy.deepcopy(list(observed_messages)) if observed_messages is not None else []
    observed_sha = context_reclaim_transcript_sha256(observed)
    facts = {"observed_view_revision": observed_sha, "view_revision": before_sha,
             "schema_names": request.schema_names, "restored_unit_refs": _unique_refs(request.restore_unit_refs)}
    if (not isinstance(request.working_note, str) or observed_messages is None or request.expected_view_revision != observed_sha
            or _view_source_messages(messages[:len(observed)]) != _view_source_messages(observed)):
        return messages, _receipt("binding_mismatch", before_sha=before_sha, **facts), None
    if any(not isinstance(ref, Mapping) or not ref for ref in request.restore_unit_refs):
        return messages, _receipt("source_unavailable", before_sha=before_sha, **facts), None
    if not callable(fit_candidate):
        return messages, _receipt("fit_rejected", before_sha=before_sha,
                                  fit={"accepted": False, "reason": "fit_callback_missing"}, **facts), None
    units = context_units(observed, scope="dialogue", trace_refs_by_tool_call_id=trace_refs,
                          measurement_density=request.measurement_density)
    keep = set(request.keep_unit_ids) if request.keep_unit_ids is not None else {u.unit_id for u in units}
    if not keep <= {u.unit_id for u in units}:
        return messages, _receipt("binding_mismatch", before_sha=before_sha, **facts), None
    protected = owner_protected_unit_ids(observed, units, protected_texts)
    authored = [u for u in units if (_capsule_metadata(observed[u.start])[1] or {}).get("authorship") == "actor"]
    removed = [u for u in units if u.unit_id not in keep and u.unit_id not in protected]
    if exposed_units is not None:
        exposed = {(ref.get("unit_id"), ref.get("raw_sha256")) for ref in exposed_units}
        removed = [u for u in removed if (u.unit_id, u.raw_sha256) in exposed]
    facts["retained_unit_ids"] = tuple(u.unit_id for u in units if u not in removed)
    visible_sources = [meta for u in units if u not in removed
                       if (meta := _capsule_metadata(observed[u.start])[1]) and meta.get("retention") == "source_view"]
    restore = [ref for ref in facts["restored_unit_refs"] if not any(
        meta["unit_id"] == ref.get("unit_id") and meta["source_unit_sha256"] == ref.get("raw_sha256")
        and meta["checkpoint_ref"] == ref.get("checkpoint_ref") for meta in visible_sources)]
    same_note = next((u for u in reversed(authored)
        if (not removed or removed == [u])
        and observed[u.start]["content"][0]["text"].partition("\n")[2] == request.working_note.strip()), None)
    messages_unchanged = not restore and (same_note is not None or not removed and not request.working_note.strip())
    schemas_unchanged = (list(observed_tool_schemas) == list(tool_schemas) if observed_tool_schemas is not None
                         else request.schema_names is None)
    no_op = messages_unchanged and schemas_unchanged
    if messages_unchanged:
        facts["retained_unit_ids"] = tuple(u.unit_id for u in units)
    checkpoint_ref, selection, capsule_refs = None, None, []
    if same_note is not None and messages_unchanged:
        # A transfer-only request names this exact observed account, never the
        # first of several notes. No new source or actor text is manufactured.
        meta = _capsule_metadata(observed[same_note.start])[1]
        selection = _Selection((), meta["unit_id"].removeprefix("view:"), 0)
    candidate = messages
    if not messages_unchanged:
        try:
            restored, restored_capsules = _restored_source_views(
                restore, drive_root=drive_root, task_id=task_id, request=request)
        except (OSError, ValueError, TypeError, KeyError):
            return messages, _receipt("source_unavailable", before_sha=before_sha, **facts), None
        if not removed and not request.working_note.strip():
            # Append after the complete current turn, including arrivals after an
            # older inspected view. No prior prefix, account or checkpoint changes.
            at = len(messages)
            candidate, capsule_refs = _materialize_replacements(messages, {at: (at - 1, restored, restored_capsules)})
            facts["source_refs"] = facts["restored_unit_refs"]
        else:
            fingerprint = _sha256(_canonical_bytes({"observed": observed_sha, "removed": [u.unit_id for u in removed],
                                                   "working_note": request.working_note, "restore": restore}))
            selection = _Selection(tuple(_SelectedUnit(u, 0, "") for u in removed), fingerprint, 0)
            checkpoint_ref = _persist_reclaim_checkpoint(observed, request, selection,
                                                         drive_root=drive_root, task_id=task_id)
            if checkpoint_ref is None:
                return messages, _receipt("checkpoint_failed", before_sha=before_sha, selection=selection, **facts), None
            source_refs = _unique_refs([ref for u in removed for ref in (*u.source_refs, {
                "checkpoint_ref": checkpoint_ref, "unit_id": u.unit_id, "raw_sha256": u.raw_sha256})])
            source = [m for u in removed for m in observed[u.start:u.end + 1]]
            combined = _unit_from_slice(source, 0, len(source) - 1, trace_refs_by_tool_call_id=trace_refs,
                                        measurement_density=request.measurement_density)
            combined = replace(combined, unit_id=f"view:{fingerprint}", generation=max((u.generation for u in removed), default=0),
                               lineage_hashes=_unique_strings([h for u in removed for h in u.lineage_hashes]),
                               source_refs=source_refs)
            note, note_ref = _capsule_message(_SelectedUnit(combined, 0, ""), request.working_note,
                                             [_part(combined.unit_id, combined.source_text)], checkpoint_ref, request)
            note["content"][0]["_context_capsule"]["authorship"] = "actor"
            note["role"] = "assistant"
            note["content"][0]["text"] = "[Actor-authored working view; exact source retained by checkpoint]\n" + request.working_note.strip()
            note["content"][0]["_context_capsule"]["visible_sha256"] = _sha256(note["content"][0]["text"])
            replacements = {u.start: (u.end, [], []) for u in removed}
            at = removed[0].start if removed else len(observed)
            replacements[at] = (removed[0].end if removed else at - 1,
                                [*restored, note], [*restored_capsules, note_ref])
            candidate, capsule_refs = _materialize_replacements(messages, replacements)
            facts["source_refs"] = _unique_refs([*source_refs, checkpoint_ref, *facts["restored_unit_refs"]])
    try:
        fit = dict(fit_candidate(copy.deepcopy(candidate), copy.deepcopy(list(tool_schemas))))
    except Exception as exc:
        fit = {"accepted": False, "reason": f"fit_callback_failed:{type(exc).__name__}"}
    facts["fit"] = fit
    if context_reclaim_transcript_sha256(messages) != before_sha:
        return messages, _receipt("binding_mismatch", before_sha=before_sha, selection=selection,
                                  checkpoint_ref=checkpoint_ref, **facts), None
    if fit.get("accepted") is not True:
        return messages, _receipt("fit_rejected", before_sha=before_sha, selection=selection,
                                  checkpoint_ref=checkpoint_ref, **facts), None
    after_sha = context_reclaim_transcript_sha256(candidate)
    facts["view_revision"] = after_sha
    reclaimed = (_context_tokens_for_messages(messages, request.measurement_density)
                 - _context_tokens_for_messages(candidate, request.measurement_density))
    return candidate, _receipt("no_op" if no_op else "applied", before_sha=before_sha, after_sha=after_sha,
                               selection=selection, checkpoint_ref=checkpoint_ref, capsule_refs=capsule_refs,
                               reclaimed_tokens=reclaimed, goal_reached=reclaimed >= request.reclaim_goal_tokens,
                               **facts), None


# Emergency representation shares this materializer but never writes a summary.
from ouroboros.context_source_view import emergency_address_view  # noqa: E402,F401


def compact_tool_history_llm(
    messages: list,
    keep_recent: int = 0,
    *,
    request: Optional[ContextReclaimRequest] = None,
    drive_root: Optional[pathlib.Path] = None,
    task_id: str = "context_compaction",
    trace_refs_by_tool_call_id: Optional[Mapping[str, Any]] = None,
    negative_memo: Optional[MutableSet[str]] = None,
    observed_messages: Optional[Sequence[Mapping[str, Any]]] = None,
    observed_tool_schemas: Optional[Sequence[Mapping[str, Any]]] = None,
    tool_schemas: Sequence[Mapping[str, Any]] = (),
    fit_candidate: Optional[Callable[[list, list], Mapping[str, Any]]] = None,
    exposed_units: Optional[Sequence[Mapping[str, Any]]] = None,
    automatic_deficit_tokens: Optional[int] = None,
    provider_refused: bool = False,
    protected_texts: Sequence[str] = (),
) -> Tuple[list, ContextReclaimReceipt, Optional[Dict[str, Any]]]:
    """Return a candidate and receipt; the caller owns atomic view publication.

    Legacy ``keep_recent`` requests use Light. An explicit ``working_note``
    uses the actor's observed snapshot/selection, requires a non-generating prospective
    ``fit_candidate(messages, tools)`` returning ``accepted`` plus fit facts,
    and never calls Light. Supply observed schemas to prove a whole-view no-op
    when selecting schemas. Missing fit evidence leaves current messages intact.
    Main supplies ``automatic_deficit_tokens`` without optional headroom: only
    newly exposed raw units are eligible, and an unreachable measured deficit
    returns facts before any checkpoint or paid helper call. Zero means that
    a real overflow has not supplied a measurable deficit, not that no shrink helps.
    ``provider_refused`` (the provider's typed refusal of this very request) also
    admits earlier capsules, after every raw unit, as the last resort; a re-folded
    capsule is one generation up and keeps its lineage. Adjacent replaced units of an
    automatic pass merge into one record without another paid fold, each original
    still restorable. ``protected_texts`` keeps governing source rows whole in an
    explicit authored view; automatic and count-only passes never touch dialogue rows.
    """

    before_sha = context_reclaim_transcript_sha256(messages)
    effective_request = request or ContextReclaimRequest(
        route_fp="manual",
        round_id=str(task_id or "context_compaction"),
        transcript_sha256=before_sha,
        measurement_basis="cold_estimate",
        measurement_density=1.0,
        reclaim_goal_tokens=2**31 - 1,
        allow_partial_shrink=True,
    )
    if str(effective_request.transcript_sha256 or "") != before_sha:
        return messages, _receipt("binding_mismatch", before_sha=before_sha), None
    _reclaim_tokens_for_byte_delta(0, effective_request.measurement_density)
    root = pathlib.Path(drive_root) if drive_root is not None else pathlib.Path("../data").resolve(strict=False)
    if effective_request.working_note is not None:
        return _authored_view(messages, effective_request, observed_messages=observed_messages,
                              observed_tool_schemas=observed_tool_schemas,
                              tool_schemas=tool_schemas, fit_candidate=fit_candidate, drive_root=root,
                              task_id=str(task_id or "context_compaction"), trace_refs=trace_refs_by_tool_call_id or {},
                              exposed_units=exposed_units, protected_texts=protected_texts)

    memo = negative_memo if negative_memo is not None else set()
    trace_refs = trace_refs_by_tool_call_id or {}
    spec = _summarizer_spec()
    selection, empty_status = _select_units(
        messages,
        effective_request,
        keep_recent=max(0, int(keep_recent)),
        trace_refs_by_tool_call_id=trace_refs,
        negative_memo=memo,
        spec=spec,
        exposed_units=exposed_units,
        automatic=automatic_deficit_tokens is not None,
        provider_refused=provider_refused,
    )
    reclaim_fit = None
    if automatic_deficit_tokens is not None:
        # Optimistic upper bound: even deleting every eligible source must be
        # able to reach the triggering boundary. Do not subtract a guessed
        # capsule size or require the optional low-water margin to be reachable.
        removed = {index for item in (selection.units if selection else ())
                   for index in range(item.unit.start, item.unit.end + 1)}
        residue = [message for index, message in enumerate(messages) if index not in removed]
        maximum = max(0, _context_tokens_for_messages(messages, effective_request.measurement_density)
                      - _context_tokens_for_messages(residue, effective_request.measurement_density))
        required = max(0, int(automatic_deficit_tokens))
        reclaim_fit = {"required_reclaim_tokens": required, "maximum_reclaim_tokens": maximum,
                       "measurement_basis": effective_request.measurement_basis,
                       "measurement_density": effective_request.measurement_density}
        if maximum < required:
            reclaim_fit["reason"] = "automatic_reclaim_unreachable"
            return messages, _receipt(
                "no_positive_reclaim" if selection else empty_status,
                before_sha=before_sha, selection=selection, fit=reclaim_fit,
            ), None
    if selection is None:
        return messages, _receipt(empty_status, before_sha=before_sha, fit=reclaim_fit), None

    checkpoint_ref = _persist_reclaim_checkpoint(
        messages, effective_request, selection, drive_root=root, task_id=str(task_id or "context_compaction"),
    )
    if checkpoint_ref is None:
        return messages, _receipt(
            "checkpoint_failed", before_sha=before_sha, selection=selection,
        ), None

    replacements: Dict[int, tuple] = {}
    memo_candidates: List[str] = []
    usage_total: Dict[str, Any] = {}
    summary_failures = 0
    initial_parts = [_part(item.unit.unit_id, item.unit.source_text) for item in selection.units]
    summary_budgets = {part.root_id: item.summary_budget_tokens for part, item in zip(initial_parts, selection.units)}
    leaves_by_root: Dict[str, List[_Part]] = {part.root_id: [] for part in initial_parts}
    summaries: Dict[str, str] = {}
    failed_roots: set[str] = set()
    batches, batch, allowance = [], [], 0
    for part in initial_parts:
        cost = summary_budgets[part.root_id] + 128
        if batch and (len(batch) >= _BLOCKS_PER_BATCH or allowance + cost > _output_budget(spec)):
            batches.append(batch)
            batch, allowance = [], 0
        batch.append(part)
        allowance += cost
    if batch:
        batches.append(batch)
    for batch in batches:
        leaves, batch_map, failed = _map_complete_parts(
            batch, drive_root=root,
            task_id=str(task_id or "context_compaction"), spec=spec,
            summary_budgets=summary_budgets, usage_total=usage_total)
        for leaf in leaves:
            leaves_by_root[leaf.root_id].append(leaf)
        summaries.update(batch_map)
        failed_roots.update(failed)

    for selected in selection.units:
        leaves = tuple(leaves_by_root[selected.unit.unit_id])
        if (selected.unit.unit_id in failed_roots or not leaves
                or any(leaf.source_id not in summaries for leaf in leaves)):
            summary_failures += 1
            continue
        summary = _fold_summaries(
            leaves, summaries, drive_root=root,
            task_id=str(task_id or "context_compaction"), spec=spec,
            summary_budget_tokens=selected.summary_budget_tokens,
            usage_total=usage_total,
        )
        if not str(summary or "").strip():
            summary_failures += 1
            continue
        replacement, capsule_ref = _capsule_message(
            selected, summary, leaves, checkpoint_ref, effective_request,
        )
        from ouroboros.knowledge import observed_route_stamp
        replacement["content"][0]["_context_capsule"]["author"] = {
            "kind": "summarization_helper", "route": observed_route_stamp(usage_total),
        }
        if _context_tokens_for_messages(
            [replacement], effective_request.measurement_density,
        ) >= selected.unit.context_size_tokens:
            memo_candidates.append(selected.negative_memo_key)
            continue
        replacements[selected.unit.start] = (selected.unit.end, [replacement], [capsule_ref])

    if context_reclaim_transcript_sha256(messages) != before_sha:
        receipt = _receipt("binding_mismatch", before_sha=before_sha, selection=selection,
                           checkpoint_ref=checkpoint_ref)
        return messages, receipt, usage_total or None

    memo.update(memo_candidates)

    if not replacements:
        status: ReclaimStatus = "summarizer_failed" if summary_failures else "no_measurable_shrink"
        receipt = _receipt(status, before_sha=before_sha, selection=selection, checkpoint_ref=checkpoint_ref)
        return messages, receipt, usage_total or None

    if automatic_deficit_tokens is not None:
        # One host record per uninterrupted replaced range. Reuse the complete
        # unit summaries without another paid fold, and keep each original unit
        # individually restorable. An owner/control turn or retained unit breaks
        # adjacency, so grouping cannot move information across those boundaries.
        # Ranges follow the transcript, not the selection's raw-first order.
        groups: list[list[_SelectedUnit]] = []
        for item in sorted(selection.units, key=lambda item: item.unit.start):
            if item.unit.start not in replacements:
                continue
            if groups and groups[-1][-1].unit.end + 1 == item.unit.start:
                groups[-1].append(item)
            else:
                groups.append([item])
        grouped = {}
        for group in groups:
            start, end = group[0].unit.start, group[-1].unit.end
            unit = _unit_from_slice(messages, start, end, trace_refs_by_tool_call_id=trace_refs,
                                    measurement_density=effective_request.measurement_density)
            # A re-folded earlier capsule (after a provider refusal) keeps counting its generations and
            # keeps the original provenance union: every member's lineage hashes and references.
            unit = replace(unit, generation=max(item.unit.generation for item in group),
                           lineage_hashes=_unique_strings([*unit.lineage_hashes, *(
                               digest for item in group for digest in item.unit.lineage_hashes)]),
                           source_refs=_unique_refs([*unit.source_refs, *(
                               ref for item in group for ref in item.unit.source_refs), *(
                               {"checkpoint_ref": checkpoint_ref, "unit_id": item.unit.unit_id,
                                "raw_sha256": item.unit.raw_sha256} for item in group)]))
            text = "\n\n".join(
                f"Source unit {item.unit.unit_id}:\n"
                + replacements[item.unit.start][1][0]["content"][0]["text"].partition("\n")[2]
                for item in group)
            record, ref = _capsule_message(_SelectedUnit(unit, 0, ""), text,
                [_part(unit.unit_id, unit.source_text)], checkpoint_ref, effective_request)
            record["content"][0]["_context_capsule"]["author"] = {
                "kind": "summarization_helper", "route": observed_route_stamp(usage_total),
            }
            grouped[start] = (end, [record], [ref])
        replacements = grouped
    rebuilt, capsule_refs = _materialize_replacements(messages, replacements)

    before_tokens = _context_tokens_for_messages(messages, effective_request.measurement_density)
    after_tokens = _context_tokens_for_messages(rebuilt, effective_request.measurement_density)
    if after_tokens >= before_tokens:
        receipt = _receipt("no_measurable_shrink", before_sha=before_sha, selection=selection,
                           checkpoint_ref=checkpoint_ref)
        return messages, receipt, usage_total or None
    reclaimed_tokens = before_tokens - after_tokens
    after_sha = context_reclaim_transcript_sha256(rebuilt)
    return rebuilt, _receipt(
        "applied",
        before_sha=before_sha,
        after_sha=after_sha,
        selection=selection,
        reclaimed_tokens=reclaimed_tokens,
        goal_reached=reclaimed_tokens >= int(effective_request.reclaim_goal_tokens),
        checkpoint_ref=checkpoint_ref,
        capsule_refs=capsule_refs,
        fit=reclaim_fit,
    ), usage_total or None
