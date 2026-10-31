"""Checkpoint-bound materialization of the mind's working view and of host source views.

Units are completed assistant-call plus contiguous matching-result slices and, in the
dialogue scope, completed prose rows. Owner turns and malformed, interrupted or visually
opaque slices stay verbatim, and an unfinished Anthropic native unit is ineligible. A
selection first writes an exact private checkpoint, then replaces the chosen units by one
capsule whose text the mind wrote (``_authored_view``) or the host derived without a model
(``context_source_view``); each original stays restorable from the checkpoint. No host
summarizer exists: a request without ``working_note`` is ``authored_note_required``.
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
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from ouroboros.context_budget import (
    HOST_CONTEXT_KIND_KEY,
    HOST_TEXT_SHA_KEY,
    ContextReclaimReceipt,
    ContextReclaimRequest,
    ReclaimStatus,
    _AtomicUnit,
    _Part,
    _Selection,
    _UnsafeVisual,
)
from ouroboros.anthropic_native_custody import anthropic_tool_unit_active, custody_private_key
from ouroboros.tool_result_record import read_tool_result_record
from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY

log = logging.getLogger(__name__)

_CAPSULE_VERSION = 1
# Every capsule stamps the contract its metadata follows (``summary_contract_version`` /
# ``summary_contract_digest``). The digests were derived from the retired helper summarizer's
# contract and are frozen as literals, so capsules written before and after its removal (actor
# notes, host source views, legacy helper records in live transcripts) validate identically.
_CAPSULE_CONTRACT_VERSION = 1
_CAPSULE_CONTRACT_DIGEST = "98d43b93d717fc52dbc84e872923b2469de2b54ead052718916039fa93d2c2ea"
_LEGACY_CAPSULE_CONTRACT_DIGEST = "6bd892aa2479a9c8b65036b0c18e53ccc076cb5bf5a0d87e351ea38182fa3402"
_MIN_CAPSULE_BYTES = 512
_SHA256_HEX = frozenset("0123456789abcdef")


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
        and summary_contract_version == _CAPSULE_CONTRACT_VERSION
        and summary_contract_digest in {_CAPSULE_CONTRACT_DIGEST, _LEGACY_CAPSULE_CONTRACT_DIGEST}
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
_PLAIN_ROW_KEYS = frozenset({"role", "content", "name", "cache_control", HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY})
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


def owner_protected_unit_ids(messages: Sequence[Mapping[str, Any]], units: Sequence[_AtomicUnit]) -> frozenset:
    """Keep the owner's and principals' inbound words and the outward dialogue whole.

    Provenance is the row's kind, typed at delivery (``context_budget.OWNER_ROW_KINDS``),
    never a text match: a short owner answer is a substring of most host rows. A row
    delivered before typing is ordinary prose the mind may fold by its own choice.
    """
    from ouroboros.context_budget import OWNER_ROW_KINDS

    return frozenset(unit.unit_id for unit in units if messages[unit.start].get(HOST_CONTEXT_KIND_KEY) in OWNER_ROW_KINDS)


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


def _part(source_id: str, text: str, start: int = 0) -> _Part:
    """The one hashed part a capsule records for its source text."""
    end = start + len(text)
    return _Part(root_id=source_id,
                 source_id=f"{source_id}:{start}:{end}:{_sha256(text)[:12]}",
                 start_char=start, end_char=end, text=text, sha256=_sha256(text))


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
        "selection_fingerprint": selection.fingerprint, "selected_unit_ids": [unit.unit_id for unit in selection.units],
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
                "selection_fingerprint": selection.fingerprint, "selected_unit_ids": [unit.unit_id for unit in selection.units],
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
    unit: _AtomicUnit,
    body: str,
    parts: Sequence[_Part],
    checkpoint_ref: Mapping[str, Any],
    request: ContextReclaimRequest,
    *, retention: str = "summarized", label: Optional[str] = None,
    address: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One capsule row standing for ``unit``: the mind's own account (``summarized``, authored by
    the actor) or a host source view (``source_view``, nothing retold); every original stays
    restorable through ``checkpoint_ref``."""
    generation = unit.generation + 1
    if retention == "source_view":
        label = label or ("Historical source: read-only projection, not live assistant/tool turns; "
                          "private transport data omitted; exact original retained by checkpoint")
        address = dict(address) if address is not None else {
            "checkpoint_ref": dict(checkpoint_ref), "unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256}
        text = f"[{label}]\nSource reference: {_canonical_json(address)}\n" + str(body or "").strip()
    else:
        label = label or "Actor-authored working view; exact source retained by checkpoint"
        text = f"[{label}]\n" + str(body or "").strip()
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
        "summary_contract_version": _CAPSULE_CONTRACT_VERSION,
        "summary_contract_digest": _CAPSULE_CONTRACT_DIGEST,
        "visible_sha256": _sha256(text),
        "authorship": "host" if retention == "source_view" else "actor",
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
        selected_unit_ids=tuple(unit.unit_id for unit in selection.units) if selection else (),
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
            unit, unit.source_text, [_part(unit.unit_id, unit.source_text)],
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
) -> Tuple[list, ContextReclaimReceipt]:
    """Materialize one actor's selection through the existing unit/checkpoint/capsule engine
    over the dialogue scope (tool units, capsules, the actor's own completed replies, host
    prose). Rows typed as people's words (``owner_protected_unit_ids``) stay whole."""
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
        return messages, _receipt("binding_mismatch", before_sha=before_sha, **facts)
    if any(not isinstance(ref, Mapping) or not ref for ref in request.restore_unit_refs):
        return messages, _receipt("source_unavailable", before_sha=before_sha, **facts)
    if not callable(fit_candidate):
        return messages, _receipt("fit_rejected", before_sha=before_sha,
                                  fit={"accepted": False, "reason": "fit_callback_missing"}, **facts)
    units = context_units(observed, scope="dialogue", trace_refs_by_tool_call_id=trace_refs,
                          measurement_density=request.measurement_density)
    keep = set(request.keep_unit_ids) if request.keep_unit_ids is not None else {u.unit_id for u in units}
    if not keep <= {u.unit_id for u in units}:
        return messages, _receipt("binding_mismatch", before_sha=before_sha, **facts)
    protected = owner_protected_unit_ids(observed, units)
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
            return messages, _receipt("source_unavailable", before_sha=before_sha, **facts)
        if not removed and not request.working_note.strip():
            # Append after the complete current turn, including arrivals after an
            # older inspected view. No prior prefix, account or checkpoint changes.
            at = len(messages)
            candidate, capsule_refs = _materialize_replacements(messages, {at: (at - 1, restored, restored_capsules)})
            facts["source_refs"] = facts["restored_unit_refs"]
        else:
            fingerprint = _sha256(_canonical_bytes({"observed": observed_sha, "removed": [u.unit_id for u in removed],
                                                   "working_note": request.working_note, "restore": restore}))
            selection = _Selection(tuple(removed), fingerprint, 0)
            checkpoint_ref = _persist_reclaim_checkpoint(observed, request, selection,
                                                         drive_root=drive_root, task_id=task_id)
            if checkpoint_ref is None:
                return messages, _receipt("checkpoint_failed", before_sha=before_sha, selection=selection, **facts)
            source_refs = _unique_refs([ref for u in removed for ref in (*u.source_refs, {
                "checkpoint_ref": checkpoint_ref, "unit_id": u.unit_id, "raw_sha256": u.raw_sha256})])
            source = [m for u in removed for m in observed[u.start:u.end + 1]]
            combined = _unit_from_slice(source, 0, len(source) - 1, trace_refs_by_tool_call_id=trace_refs,
                                        measurement_density=request.measurement_density)
            combined = replace(combined, unit_id=f"view:{fingerprint}", generation=max((u.generation for u in removed), default=0),
                               lineage_hashes=_unique_strings([h for u in removed for h in u.lineage_hashes]),
                               source_refs=source_refs)
            note, note_ref = _capsule_message(combined, request.working_note,
                                             [_part(combined.unit_id, combined.source_text)], checkpoint_ref, request)
            note["role"] = "assistant"
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
                                  checkpoint_ref=checkpoint_ref, **facts)
    if fit.get("accepted") is not True:
        return messages, _receipt("fit_rejected", before_sha=before_sha, selection=selection,
                                  checkpoint_ref=checkpoint_ref, **facts)
    after_sha = context_reclaim_transcript_sha256(candidate)
    facts["view_revision"] = after_sha
    reclaimed = (_context_tokens_for_messages(messages, request.measurement_density)
                 - _context_tokens_for_messages(candidate, request.measurement_density))
    return candidate, _receipt("no_op" if no_op else "applied", before_sha=before_sha, after_sha=after_sha,
                               selection=selection, checkpoint_ref=checkpoint_ref, capsule_refs=capsule_refs,
                               reclaimed_tokens=reclaimed, goal_reached=reclaimed >= request.reclaim_goal_tokens,
                               **facts)


# Emergency representation shares this materializer but never writes a summary.
from ouroboros.context_source_view import emergency_address_view  # noqa: E402,F401


def compact_tool_history_llm(
    messages: list,
    *,
    request: ContextReclaimRequest,
    drive_root: Optional[pathlib.Path] = None,
    task_id: str = "context_compaction",
    trace_refs_by_tool_call_id: Optional[Mapping[str, Any]] = None,
    observed_messages: Optional[Sequence[Mapping[str, Any]]] = None,
    observed_tool_schemas: Optional[Sequence[Mapping[str, Any]]] = None,
    tool_schemas: Sequence[Mapping[str, Any]] = (),
    fit_candidate: Optional[Callable[[list, list], Mapping[str, Any]]] = None,
    exposed_units: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[list, ContextReclaimReceipt]:
    """Materialize the mind's authored view: the candidate and its receipt; the caller owns
    atomic publication.

    ``request.working_note`` is the mind's own account of what it folds. No helper writes it:
    a request without one is refused as ``authored_note_required`` and nothing changes. The
    observed snapshot/selection binds the view, the non-generating ``fit_candidate(messages,
    tools)`` must accept the complete candidate, and missing fit evidence leaves current
    messages intact. Supply observed schemas to prove a whole-view no-op when selecting
    schemas. Rows typed as people's words stay whole.
    """
    before_sha = context_reclaim_transcript_sha256(messages)
    if str(request.transcript_sha256 or "") != before_sha:
        return messages, _receipt("binding_mismatch", before_sha=before_sha)
    _reclaim_tokens_for_byte_delta(0, request.measurement_density)
    if request.working_note is None:
        return messages, _receipt("authored_note_required", before_sha=before_sha)
    root = pathlib.Path(drive_root) if drive_root is not None else pathlib.Path("../data").resolve(strict=False)
    return _authored_view(messages, request, observed_messages=observed_messages,
                          observed_tool_schemas=observed_tool_schemas,
                          tool_schemas=tool_schemas, fit_candidate=fit_candidate, drive_root=root,
                          task_id=str(task_id or "context_compaction"), trace_refs=trace_refs_by_tool_call_id or {},
                          exposed_units=exposed_units)
