import hashlib
import json
import logging
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.utils import (
    append_jsonl,
    atomic_write_json,
    read_json_dict,
    replace_atomic,
    utc_now_iso,
    read_text,
    write_text,
)

from ouroboros.platform_layer import (
    file_lock_exclusive as _lock_ex,
    file_lock_exclusive_nb as _lock_nb,
    file_unlock as _unlock,
)

log = logging.getLogger(__name__)

BLOCK_SIZE = 100                          # Messages per consolidation block
MAX_SUMMARY_BLOCKS = 10                   # Compress into era when exceeded
ERA_COMPRESS_COUNT = 4                    # Oldest blocks to compress per era
BLOCK_SUMMARY_MAX_CHARS = 24_000         # ibl-local consolidator: per-block byte cap (BIBLE P1)
_ENTRY_TEXT_PRETRUNCATE_LIMIT = 2_000     # ibl-local consolidator: producer-side cap on per-entry text passed to the LLM (drops bulk tool output / dumps before the prompt can reproduce them)
ERA_SUMMARY_MAX_CHARS = 96_000           # ibl-local consolidator: per-era byte cap (4× block cap)
PENDING_BYTES_TRIGGER = 50_000           # ibl-local consolidator: byte-aware consolidation trigger


def _consolidation_route() -> Tuple[str, bool]:
    """Resolve summaries through the configured Light lane.

    Reuse the lane resolver so an empty Light slot inherits both Main's model
    and its local-routing flag. Remote routes retain the provider-independence
    fallback; explicitly local routes must never be rewritten to a remote
    credentialed model.
    """
    from ouroboros.provider_models import resolve_credentialed_model
    from ouroboros.subagents import resolve_subagent_lane

    lane = resolve_subagent_lane("light")
    if lane.use_local_model:
        return lane.model, True
    return resolve_credentialed_model(lane.model), False


CONSOLIDATION_REASONING_EFFORT = "medium"


def _ordered_chat_generation_paths(source_path: pathlib.Path) -> List[pathlib.Path]:
    """Return the consolidator-owned physical chat chain, oldest to live."""
    archive_dir = source_path.parent.parent / "archive"
    try:
        archives = sorted(archive_dir.glob("chat_*.jsonl"), key=lambda p: p.name)
    except OSError:
        archives = []
    return [*archives, source_path]


def _resolve_generation_segments(
    meta: Dict[str, Any], source_path: pathlib.Path,
) -> Tuple[List[pathlib.Path], int, bool]:
    """Generation-aware consolidation cursor (v6.73.0).

    The cursor (``last_consolidated_offset`` + ``chat_log_signature``) points into
    ONE log generation. Rotation moves that generation to ``archive/chat_<ts>.jsonl``
    verbatim, so the stored first-line hash locates it in the ordered archive chain
    and consolidation continues over ``archives[i:] + live`` — the pre-rotation
    tail (and any number of intervening rotations) is consolidated, never dropped.
    Returns ``(ordered segments, offset into their concatenation, gap_detected)``;
    ``gap_detected`` is True only when the stored generation no longer exists
    anywhere (manual deletion/corruption — archives are never auto-pruned).
    """
    last_offset = int(meta.get("last_consolidated_offset", 0) or 0)
    stored_sig = meta.get("chat_log_signature") or {}
    stored_first = str(stored_sig.get("first_line_sha256") or "") if isinstance(stored_sig, dict) else ""
    live_sig = _chat_log_signature(source_path)
    archives = _ordered_chat_generation_paths(source_path)[:-1]
    if not stored_first:
        # Uninitialized cursor. Any archives that already exist rotated BEFORE
        # the first consolidation ever ran — they are unconsolidated by
        # definition, so the whole ordered chain is the window (offset 0).
        # A nonzero offset WITHOUT a signature is an ambiguous pre-signature
        # legacy shape: keep the historical live-only behavior for it.
        if last_offset == 0 and archives:
            return [*archives, source_path], 0, False
        return [source_path], last_offset, False
    if stored_first == str(live_sig.get("first_line_sha256") or ""):
        return [source_path], last_offset, False
    for index, archive_path in enumerate(archives):
        sig = _chat_log_signature(archive_path)
        if str(sig.get("first_line_sha256") or "") == stored_first:
            return [*archives[index:], source_path], last_offset, False
    return [source_path], 0, True


def _pending_bytes(segments: List[pathlib.Path], last_offset: int) -> int:
    """Approximate pending bytes across the resolved segments chain starting at last_offset (chain-position).

    Walks segments in order. While ``last_offset`` exceeds a segment's line
    count, the segment is fully consumed and we advance past it. The segment
    in which the cursor lands contributes a proportional share of its bytes
    (uniform-line-size approximation). All segments AFTER the cursor segment
    are entirely pending. Returns 0 for empty / missing segments, or when the
    cursor is at or past the chain tail.

    Closes ibl-f60344038572: the trigger must look at PENDING bytes, not the
    cumulative st_size of every segment (which would falsely trip once the
    archived chain alone exceeds PENDING_BYTES_TRIGGER and recompute O(history)
    work every tick). Mirrors the existing line-count semantics:
        pending_lines = total_lines - last_offset
    """
    if not segments:
        return 0
    pending = 0
    remaining = last_offset
    for path in segments:
        if not path.exists():
            continue
        size = path.stat().st_size
        seg_lines = _count_lines(path)
        if seg_lines <= 0:
            continue
        if remaining >= seg_lines:
            remaining -= seg_lines
            continue
        # Cursor lands within this segment OR at/before its start.
        if remaining <= 0:
            pending += size
        else:
            proportion = (seg_lines - remaining) / seg_lines
            pending += int(size * proportion)
        remaining = 0  # subsequent segments are fully pending
    return pending


def should_consolidate(
    meta_path: pathlib.Path,
    chat_path: pathlib.Path,
) -> bool:
    if not chat_path.exists():
        return False
    meta = _load_meta(meta_path)
    segments, last_offset, gap_detected = _resolve_generation_segments(meta, chat_path)
    if gap_detected:
        # A detected discontinuity must be RECORDED (BIBLE P1), so it schedules a
        # run regardless of pending volume: the run appends the one durable gap
        # block and rebases the cursor even below BLOCK_SIZE.
        return True
    total_lines = sum(_count_lines(path) for path in segments if path.exists())
    if last_offset > total_lines:
        # Cursor past total — fall back to live-only heuristic. Line count is
        # the primary trigger; the byte threshold (PENDING_BYTES_TRIGGER) is a
        # SECONDARY safety net for pathological long-message cases where message
        # count stays under BLOCK_SIZE but raw bytes are already bloating.
        live_lines = _count_lines(chat_path)
        if live_lines >= BLOCK_SIZE:
            return True
        return _pending_bytes(segments, last_offset) >= PENDING_BYTES_TRIGGER
    pending_lines = total_lines - last_offset
    if pending_lines >= BLOCK_SIZE:
        return True
    # Byte-aware secondary trigger (ibl-local consolidator, ibl-f60344038572):
    # a single pathological long-message run can accumulate 50K chars across
    # well under 100 messages. The trigger computes pending bytes from
    # segments starting at last_offset — NOT the cumulative st_size of every
    # segment (which would falsely trip once the archived chain alone
    # exceeded PENDING_BYTES_TRIGGER).
    return _pending_bytes(segments, last_offset) >= PENDING_BYTES_TRIGGER


def consolidate(
    chat_path: pathlib.Path,
    blocks_path: pathlib.Path,
    meta_path: pathlib.Path,
    llm_client: Any,
    identity_text: str = "",
) -> Optional[Dict[str, Any]]:
    lock_path = meta_path.parent / ".consolidation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            _lock_nb(lock_fd)
        except (OSError, BlockingIOError):
            log.info("Chat block consolidation already running, skipping")
            return None

        return _run_block_consolidation(
            source_path=chat_path,
            blocks_path=blocks_path,
            meta_path=meta_path,
            llm_client=llm_client,
            identity_text=identity_text,
        )
    finally:
        if lock_fd is not None:
            try:
                _unlock(lock_fd)
                os.close(lock_fd)
            except OSError:
                pass
def _capture_generation_window(
    meta_path: pathlib.Path,
    source_path: pathlib.Path,
    segments: List[pathlib.Path],
    last_offset: int,
) -> Optional[Tuple[List[pathlib.Path], List[Dict[str, Any]], List[List[Dict[str, Any]]], List[Dict[str, Any]], int]]:
    """Verified capture of the resolved generation window (v6.73.0).

    Captures each segment's signature WITH its entries so the cursor commits
    against read-time identities; anchors the first segment to the cursor
    generation in meta; verifies the mutable live segment sig->read->sig; any
    detected change re-resolves and re-captures (bounded), a mid-loop gap or a
    rotation storm defers coherently. Returns None on deferral, else
    ``(segments, segment_sigs, segment_entries, all_entries, last_offset)``."""
    all_entries: List[Dict[str, Any]] = []
    for _capture_attempt in range(3):
        segment_sigs = [_chat_log_signature(path) for path in segments]
        segment_entries = [_read_chat_entries(path) for path in segments]
        live_sig_after = _chat_log_signature(source_path)
        # The FIRST captured segment must still be the generation the stored
        # cursor points at — a rotation between the initial resolve and this
        # capture would otherwise let the old offset be applied to the NEW live
        # generation (skipping its prefix and dropping the archived tail).
        cursor_first = ""
        meta_now = _load_meta(meta_path)
        cursor_sig = meta_now.get("chat_log_signature") or {}
        if isinstance(cursor_sig, dict):
            cursor_first = str(cursor_sig.get("first_line_sha256") or "")
        if cursor_first and str(segment_sigs[0].get("first_line_sha256") or "") != cursor_first:
            segments, last_offset, _mid_gap = _resolve_generation_segments(meta_now, source_path)
            if _mid_gap:
                log.warning("Cursor generation vanished mid-consolidation; deferring to the loud gap path")
                return None
            continue
        if str(live_sig_after.get("first_line_sha256") or "") != str(
            segment_sigs[-1].get("first_line_sha256") or ""
        ):
            segments, last_offset, _mid_gap = _resolve_generation_segments(
                _load_meta(meta_path), source_path,
            )
            if _mid_gap:
                log.warning("Cursor generation vanished mid-consolidation; deferring to the loud gap path")
                return None
            continue
        all_entries = [entry for segment in segment_entries for entry in segment]
        if last_offset > len(all_entries):
            refreshed, refreshed_offset, _refreshed_gap = _resolve_generation_segments(
                _load_meta(meta_path), source_path,
            )
            if _refreshed_gap:
                log.warning("Cursor generation vanished mid-consolidation; deferring to the loud gap path")
                return None
            if [str(p) for p in refreshed] != [str(p) for p in segments] or (
                refreshed_offset != last_offset
            ):
                # The resolution CHANGED (a rotation landed mid-call): adopt it
                # and re-capture verified on the next iteration.
                segments, last_offset = refreshed, refreshed_offset
                continue
            log.warning("Chat consolidation offset beyond generation entries, resetting offset")
            last_offset = 0
        break
    else:
        log.warning("Chat log rotated during every capture attempt; deferring consolidation")
        return None

    return segments, segment_sigs, segment_entries, all_entries, last_offset


def _run_block_consolidation(
    source_path: pathlib.Path,
    blocks_path: pathlib.Path,
    meta_path: pathlib.Path,
    llm_client: Any,
    identity_text: str,
) -> Optional[Dict[str, Any]]:
    meta = _load_meta(meta_path)
    segments, last_offset, gap_detected = _resolve_generation_segments(meta, source_path)
    if gap_detected:
        # The stored generation is gone (manual deletion/corruption). The lost
        # span is represented as ONE EXPLICIT durable gap block (BIBLE P1: a gap
        # is a fact in memory, not a silent absence); the cursor rebases ONLY
        # once the marker is durably present (idempotent by lost-cursor id), so
        # a failed block write keeps the old cursor for retry and an interrupted
        # attempt never duplicates the marker.
        lost_sig = meta.get("chat_log_signature") or {}
        lost_marker = (
            f"{str(lost_sig.get('first_line_sha256') or 'unknown')[:16]}"
            f":{int(meta.get('last_consolidated_offset', 0) or 0)}"
        )
        log.warning(
            "Chat consolidation cursor generation not found in archive chain; "
            "appending explicit gap block (last_offset=%d, live_entries=%d)",
            int(meta.get("last_consolidated_offset", 0) or 0), _count_lines(source_path),
        )
        if not _append_gap_block(blocks_path, lost_marker):
            return None
        meta["last_consolidated_offset"] = 0
        meta["chat_log_signature"] = _chat_log_signature(source_path)
        atomic_write_json(meta_path, meta)
        last_offset = 0

    captured = _capture_generation_window(meta_path, source_path, segments, last_offset)
    if captured is None:
        return None
    segments, segment_sigs, segment_entries, all_entries, last_offset = captured
    new_entries = all_entries[last_offset:]
    if len(new_entries) < BLOCK_SIZE:
        return None

    total_usage: Dict[str, Any] = {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0,
    }
    new_blocks: List[Dict[str, Any]] = []
    chunks_to_process = len(new_entries) // BLOCK_SIZE
    processed = 0

    for i in range(chunks_to_process):
        chunk = new_entries[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
        formatted = _format_entries_for_block(chunk)
        first_ts = str(chunk[0].get("ts", "unknown"))
        last_ts = str(chunk[-1].get("ts", "unknown"))

        content, usage = _create_block_summary(
            llm_client=llm_client,
            messages_text=formatted,
            first_ts=first_ts,
            last_ts=last_ts,
            identity_text=identity_text,
            message_count=len(chunk),
        )

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[key] += usage.get(key, 0)
        if total_usage["cost"] is not None:
            if usage.get("cost") is None:
                total_usage["cost"] = None
            else:
                total_usage["cost"] += float(usage["cost"])

        if content and content.strip():
            first_date, last_date = first_ts[:10], last_ts[:10]
            first_time, last_time = first_ts[11:16], last_ts[11:16]
            if first_date == last_date:
                range_str = f"{first_date} {first_time} - {last_time}"
            else:
                range_str = f"{first_date} {first_time} - {last_date} {last_time}"

            new_blocks.append({
                "ts": utc_now_iso(),
                "type": "summary",
                "range": range_str,
                "message_count": len(chunk),
                "content": content.strip(),
            })
            processed += len(chunk)
        else:
            log.warning("Block summary empty for chunk %d, will retry next cycle", i)
            break

    if not new_blocks:
        _advance_cursor(meta, segments, segment_sigs, segment_entries, last_offset + processed)
        atomic_write_json(meta_path, meta)
        return total_usage if total_usage["prompt_tokens"] or total_usage["completion_tokens"] else None

    existing_blocks = _load_blocks(blocks_path)
    all_blocks = existing_blocks + new_blocks

    if len(all_blocks) > MAX_SUMMARY_BLOCKS:
        compress_count = min(ERA_COMPRESS_COUNT, len(all_blocks) - 1)
        old_blocks = all_blocks[:compress_count]
        remaining = all_blocks[compress_count:]
        # Gap markers are DURABLE discontinuity facts (BIBLE P1): they keep
        # their exact chronological positions, and an era may only compress ONE
        # CONTIGUOUS run of ordinary summary blocks — never a span that bridges
        # a known discontinuity.
        def _is_gap(block: Any) -> bool:
            return isinstance(block, dict) and bool(block.get("gap_id"))

        run_start = next((i for i, b in enumerate(old_blocks) if not _is_gap(b)), None)
        era = None
        if run_start is not None:
            run_end = run_start
            while run_end < len(old_blocks) and not _is_gap(old_blocks[run_end]):
                run_end += 1
            era, era_usage = _compress_blocks_to_era(
                old_blocks[run_start:run_end], llm_client, identity_text,
            )
        if era is not None:
            all_blocks = [
                *old_blocks[:run_start], era, *old_blocks[run_end:], *remaining,
            ]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[key] += era_usage.get(key, 0)
            if total_usage["cost"] is not None:
                if era_usage.get("cost") is None:
                    total_usage["cost"] = None
                else:
                    total_usage["cost"] += float(era_usage["cost"])

    _write_locked_json(blocks_path, all_blocks)

    _advance_cursor(meta, segments, segment_sigs, segment_entries, last_offset + processed)
    meta["last_consolidated_at"] = utc_now_iso()
    atomic_write_json(meta_path, meta)

    log.info("Block consolidation: %d messages -> %d new blocks (total %d)",
             processed, len(new_blocks), len(all_blocks))
    return total_usage


def _strip_think_blocks(raw: str) -> str:
    """Strip ``...`` blocks from LLM output before storing.

    Models like minimax/M2.7-highspeed emit `` reasoning regardless of
    `reasoning_effort="low"`. Storing those sections inflates dialogue_blocks.json
    (45.8% of current content per cycle-54 measurement, observed range
    0–95% per block) and bloats downstream scratchpad rendering. We strip them
    AFTER the call — the model still thinks, the stored memory only carries
    the answer. Conservative: only strips WELL-FORMED closing tags; an unclosed
    `` tag fails safe (no strip) so partial responses are preserved
    verbatim and surfaced to the cursor advance / retry path."""
    if not raw:
        return ""
    stripped = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
    return stripped.strip()


def _truncate_with_marker(content: str, max_chars: int, span_label: str) -> Tuple[str, bool]:
    """Bound stored content at ``max_chars`` and append an honest truncation marker.

    Returns ``(content, was_truncated)``. When content fits within the cap
    the original is returned unchanged with ``False``; when content exceeds
    the cap the bounded prefix plus marker is returned with ``True``. The
    typed ``was_truncated`` lets callers record the cut as a durable fact
    rather than infer it from content shape.

    BIBLE P1: a content cap that silently loses the tail is NOT acceptable —
    the marker is a durable, visible fact that the stored span is bounded.
    Cursor and span_label are recorded so a reader can correlate the truncation
    back to the generation that produced it.
    """
    if len(content) <= max_chars:
        return content, False
    kept = content[:max_chars].rstrip()
    marker = (
        f"\n\n[...truncated at {max_chars} chars (original ~{len(content)} chars); "
        f"span: {span_label} — single-block byte cap prevents dialogue_blocks.json bloat]..."
    )
    return kept + marker, True


def _call_consolidation_llm(
    llm_client: Any,
    prompt: str,
    label: str,
    *,
    max_chars: int = BLOCK_SUMMARY_MAX_CHARS,
    span_label: str = "consolidation",
) -> Tuple[str, Dict[str, Any]]:
    try:
        model, use_local = _consolidation_route()
        msg, usage = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            model_role="light",
            tools=None,
            reasoning_effort="low",
            max_tokens=16384,
            use_local=use_local,
        )
        raw = msg.get("content", "")
        stripped = _strip_think_blocks(raw)
        bounded, was_truncated = _truncate_with_marker(stripped, max_chars, span_label)
        if was_truncated and len(bounded) > max_chars + 200:  # marker overhead capped ~200 chars
            log.warning(
                "Consolidation output for %s exceeded byte cap %d (stripped %d chars, truncated to %d)",
                span_label, max_chars, len(stripped), max_chars,
            )
        return bounded, usage
    except Exception as e:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(e)
        log.error("%s failed: %s", label, e, exc_info=True)
        return "", {"cost": 0}


def _create_block_summary(
    llm_client: Any,
    messages_text: str,
    first_ts: str,
    last_ts: str,
    identity_text: str,
    message_count: int,
) -> Tuple[str, Dict[str, Any]]:
    first_date = first_ts[:10]
    first_time = first_ts[11:16]
    last_time = last_ts[11:16]

    identity_section = ""
    if identity_text:
        identity_section = f"\n## Identity context\n{identity_text}\n"

    prompt = f"""You are a memory consolidator for Ouroboros, a self-modifying AI agent.
Create a CONCISE episodic memory entry from these {message_count} messages.

## Header
### Block: {first_date} {first_time} - {last_time}

## Hard target
- Output must stay below ~2,500 characters total (about 500 words).
- When the input is large, the OUTPUT must be DRAMATICALLY shorter than the input. Compress hard.
- Do NOT reproduce verbatim source text (raw tool output, JSON dumps, repeated markdown headers, code blocks).

## PRESERVE — non-negotiable
- Decisions and agreements ("we decided X")
- State changes (code changed, env changed, identity/state changed)
- Lessons learned (what worked, what failed)
- Unresolved questions (open threads, pending items)
- Task ids when the message references a specific task

## DROP — noise for an episodic memory block
- Routine tool calls that succeed without commentary
- Tool-result JSON payloads (already in chat.jsonl, durable)
- Repeated stack traces and debug output
- Verbatim dialogue longer than 2 lines from a single exchange
- Source markdown tables — paraphrase the cell content instead of copying
- Source code blocks — describe the change instead of copying

## Hard prohibitions
- Do NOT preserve markdown headers from the source verbatim (#, ##, ### at column 0)
- Do NOT include model banners like "Sure, here is..." or "I will..."
- Do NOT pad the entry — say what happened, stop

## Voice
First person as Ouroboros: "I did...", "I learned...", "the user asked..."
{identity_section}
## Messages to summarize
{messages_text}
"""

    return _call_consolidation_llm(
        llm_client,
        prompt,
        "Block summary LLM call",
        max_chars=BLOCK_SUMMARY_MAX_CHARS,
        span_label=f"block {first_date} {first_time}-{last_time}",
    )


def _compress_blocks_to_era(
    blocks: List[Dict[str, Any]],
    llm_client: Any,
    identity_text: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    start_date = blocks[0].get("range", "unknown")[:10]
    last_range = blocks[-1].get("range", "unknown")
    if " to " in last_range:
        end_date = last_range.split(" to ")[-1].strip()[:10]
    else:
        end_date = last_range[:10]

    combined = "\n\n---\n\n".join(
        f"### {b.get('range', 'unknown')}\n{b.get('content', '')}"
        for b in blocks
    )

    prompt = f"""Compress these older memory blocks into a single era summary.
Preserve: key decisions, personality discoveries, relationship moments, technical milestones.
Drop: debugging details, routine operations, redundant info.
Header: ### Era: {start_date} to {end_date}
Write as Ouroboros (first person). Aim for 30-40% of original length.

## Blocks to compress

{combined}
"""

    content, usage = _call_consolidation_llm(
        llm_client,
        prompt,
        "Era compression",
        max_chars=ERA_SUMMARY_MAX_CHARS,
        span_label=f"era {start_date}-{end_date}",
    )
    if not content or not content.strip():
        log.warning("Era compression returned empty — keeping original blocks (Bible P1)")
        return None, usage
    era = {
        "ts": utc_now_iso(),
        "type": "era",
        "range": f"{start_date} to {end_date}",
        "message_count": sum(b.get("message_count", 0) for b in blocks),
        "content": content.strip(),
    }
    return era, usage

def _format_entries_for_block(entries: List[Dict[str, Any]]) -> str:
    lines = []
    for e in entries:
        ts_raw = str(e.get("ts", ""))
        ts = ts_raw[:10] + " " + ts_raw[11:16] if len(ts_raw) >= 16 else ts_raw
        dir_raw = str(e.get("direction", "")).lower()
        if dir_raw in ("out", "outgoing"):
            direction_prefix = "-> "
            author = "Ouroboros"
        elif dir_raw == "system":
            direction_prefix = "[system] "
            author = "Ouroboros"
        else:
            direction_prefix = ""
            from ouroboros.dialogue_provenance import dialogue_author

            author = dialogue_author(e)
        text = str(e.get("text", ""))
        if len(text) > _ENTRY_TEXT_PRETRUNCATE_LIMIT:
            # Producer-side pre-truncation. The LLM cannot reproduce verbatim
            # content that no longer exists in its input. Reasoning / decisions
            # / outcomes remain untouched (they are not in tool output JSON);
            # chat.jsonl is the durable source so the cut is reversible for
            # forensics (BIBLE P1 boundary on dialogue_blocks.json is preserved).
            text = f"[TEXT TRUNCATED: original ~{len(text)} chars, see chat.jsonl]"
        lines.append(f"[{ts}] {direction_prefix}{author}: {text}")
    return "\n\n".join(lines)


def _load_blocks(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(read_text(path))
        if not isinstance(data, list):
            raise ValueError(f"dialogue blocks store is {type(data).__name__}, not a list")
        return data
    except (json.JSONDecodeError, ValueError):
        # Memory loss must never be silent (P1): quarantine the corrupt store
        # for forensic recovery instead of overwriting it on the next write.
        quarantine = path.with_name(f"{path.name}.corrupt-{utc_now_iso().replace(':', '')}.bak")
        try:
            replace_atomic(path, quarantine)
            log.error("Corrupt blocks file %s — quarantined to %s, starting fresh", path, quarantine)
        except OSError:
            log.error("Corrupt blocks file %s — quarantine failed, starting fresh", path, exc_info=True)
        try:
            from ouroboros.utils import append_jsonl

            append_jsonl(path.parent.parent / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "memory_store_corrupt",
                "path": str(path),
                "quarantine": str(quarantine),
            })
        except Exception:
            log.debug("Failed to emit memory_store_corrupt event", exc_info=True)
        return []


def _write_locked_json(path: pathlib.Path, payload: Any) -> None:
    """Write JSON under the cross-process write lock, atomically.

    The lock serializes concurrent consolidators; the write itself goes
    through a temp file + rename so a crash mid-write can never leave a
    truncated dialogue_blocks.json (the long-term memory store).
    """
    _mutate_locked_json_list(path, lambda _current: payload)


def _mutate_locked_json_list(path: pathlib.Path, mutator: Any) -> Any:
    """Locked read-modify-write for a JSON list store (atomic replace).

    ``mutator(current_list) -> new_list`` runs while the sidecar lock is held,
    so concurrent appenders cannot be lost between the re-read and the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        _lock_ex(fd)
        current: Any = []
        if path.exists():
            try:
                current = json.loads(read_text(path))
            except (json.JSONDecodeError, ValueError):
                current = []
        updated = mutator(current if isinstance(current, list) else [])
        atomic_write_json(path, updated)
        return updated
    finally:
        if fd is not None:
            try:
                _unlock(fd)
                os.close(fd)
            except OSError:
                pass

def _append_gap_block(blocks_path: pathlib.Path, lost_marker: str) -> bool:
    """Durable explicit discontinuity marker; idempotent by the lost-cursor id.

    Returns True only when the marker is durably present (freshly appended or
    already there from an interrupted earlier attempt) — the caller advances the
    cursor ONLY on True, so a failed write never erases the old cursor without
    its promised gap record, and a crash between block and meta writes cannot
    duplicate the marker on retry."""
    gap_id = f"gap:{lost_marker}"
    gap_block = {
        "ts": utc_now_iso(),
        "type": "summary",
        "range": "unknown",
        "message_count": 0,
        "gap_id": gap_id,
        "content": (
            "[MEMORY GAP] The chat-log generation holding the consolidation cursor "
            "could not be located in the archive chain; an un-consolidated span of "
            "dialogue precedes this point and is not summarized here."
        ),
    }

    def _add_once(blocks):
        if any(block.get("gap_id") == gap_id for block in blocks if isinstance(block, dict)):
            return blocks
        return [*blocks, gap_block]

    try:
        # A corrupt existing store must be QUARANTINED (same P1 discipline as
        # _load_blocks), never silently reset to [] by the locked mutator —
        # this write path would otherwise destroy the forensic copy.
        _load_blocks(blocks_path)
        updated = _mutate_locked_json_list(blocks_path, _add_once)
        return any(
            block.get("gap_id") == gap_id for block in updated if isinstance(block, dict)
        )
    except Exception:
        log.warning("Failed to append consolidation gap block", exc_info=True)
        return False


def _advance_cursor(
    meta: Dict[str, Any],
    segments: List[pathlib.Path],
    segment_sigs: List[Dict[str, Any]],
    segment_entries: List[List[Dict[str, Any]]],
    position: int,
) -> None:
    """Stamp offset + the CAPTURED signature of the segment the position falls in.

    While consumption still ends inside an archived segment, that segment's
    signature is kept so the next run resumes exactly there; only when the
    cursor crosses into the live file does the signature advance to it. The
    signature is the one captured at READ time — if the live file rotated during
    summarization, the captured identity now names an archived generation and
    the next run's chain walk continues from it without loss."""
    segment_start = 0
    for path, sig, entries in zip(segments, segment_sigs, segment_entries):
        if position < segment_start + len(entries) or path is segments[-1]:
            meta["last_consolidated_offset"] = position - segment_start
            meta["chat_log_signature"] = sig
            return
        segment_start += len(entries)


def _load_meta(path: pathlib.Path) -> Dict[str, Any]:
    return read_json_dict(path) or {}


from ouroboros.utils import jsonl_generation_signature as _chat_log_signature


def _count_lines(path: pathlib.Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _read_chat_entries(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    # Full project awareness (v6.32.0): the one identity's consolidated dialogue
    # (dialogue_blocks.json) is its WHOLE conversation — main + project threads —
    # because Ouroboros is one awareness/biography across direct chat, project
    # rooms, and background consciousness (BIBLE P1). Only A2A virtual-transport
    # ids are excluded (machine-to-machine traffic, not the human dialogue). This
    # MUST match memory.read_jsonl_tail_after_offset so the shared consolidation
    # offset indexes the same stream.
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_a2a_chat_id(entry.get("chat_id", 1)):
                entries.append(entry)
    return entries

def _rebuild_knowledge_index(knowledge_dir: pathlib.Path, *, _locked: bool = False) -> None:
    if not _locked:
        from ouroboros.tools.knowledge import _knowledge_write_lock

        with _knowledge_write_lock(knowledge_dir):
            _rebuild_knowledge_index(knowledge_dir, _locked=True)
        return
    try:
        if not knowledge_dir.exists():
            return
        entries = []
        for md_file in sorted(knowledge_dir.glob("*.md")):
            if md_file.name.startswith("_") or md_file.name == "index-full.md":
                continue
            topic = md_file.stem
            first_line = ""
            try:
                first_line = next(
                    (line.strip()[:120] for line in md_file.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.strip().startswith("#")),
                    "",
                )
            except Exception:
                pass
            entries.append(f"- **{topic}**: {first_line}" if first_line else f"- **{topic}**")
        write_text(knowledge_dir / "index-full.md", "# Knowledge Base Index\n\n" + "\n".join(entries) + "\n")
    except Exception:
        log.warning("Failed to rebuild knowledge index", exc_info=True)

from ouroboros.context_budget import (
    SCRATCHPAD_CONSOLIDATION_THRESHOLD_CHARS as SCRATCHPAD_CONSOLIDATION_THRESHOLD,
)


def should_consolidate_scratchpad(memory: Any) -> bool:
    try:
        blocks = memory.load_scratchpad_blocks()
        return len(blocks) >= 3 and sum(len(b.get("content", "")) for b in blocks) > SCRATCHPAD_CONSOLIDATION_THRESHOLD
    except Exception:
        return False


def consolidate_scratchpad(
    memory: Any,
    knowledge_dir: pathlib.Path,
    llm_client: Any,
    identity_text: str = "",
) -> Optional[Dict[str, Any]]:
    blocks = memory.load_scratchpad_blocks()

    if len(blocks) < 3:
        return None
    return _consolidate_scratchpad_blocks(memory, blocks, knowledge_dir, llm_client, identity_text)


def _consolidate_scratchpad_blocks(
    memory: Any,
    blocks: List[Dict[str, Any]],
    knowledge_dir: pathlib.Path,
    llm_client: Any,
    identity_text: str,
) -> Optional[Dict[str, Any]]:
    total_chars = sum(len(b.get("content", "")) for b in blocks)
    if total_chars <= SCRATCHPAD_CONSOLIDATION_THRESHOLD:
        return None

    compress_count = max(2, len(blocks) // 2)
    old_blocks = blocks[:compress_count]

    old_content = "\n\n---\n\n".join(
        f"[{b.get('ts', '?')[:16]} \u2014 {b.get('source', '?')}]\n{b.get('content', '')}"
        for b in old_blocks
    )

    prompt = f"""You are a memory consolidator for Ouroboros, a self-modifying AI agent.

The scratchpad working memory has {len(blocks)} blocks totaling {total_chars} chars.
The oldest {compress_count} blocks need compression.

Rules:
1. Identify insights, patterns, lessons, and architectural decisions worth
   preserving long-term. Output them as knowledge_entries with topic + content.
   Each "topic" must be a short kebab-case slug (lowercase letters/digits/hyphens,
   e.g. "api-gotchas"), not a sentence — a topic with spaces or punctuation is
   rejected and the entry is dropped.
2. Compress the old blocks into a SINGLE shorter summary block. Keep active
   tasks, unresolved questions, admin instructions still in force. Remove
   stale/completed items and routine status updates.
3. Write as Ouroboros (first person). Don't lose signal — keep uncertain items
   rather than dropping them.

Identity context: {identity_text if identity_text else "(not available)"}

## Old blocks to compress

{old_content}

Respond with JSON only (no fences):
{{"knowledge_entries": [{{"topic": "kebab-case-slug", "content": "text"}}], "compressed_block": "single compressed block text"}}
"""

    try:
        model, use_local = _consolidation_route()
        msg, usage = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            model_role="light",
            reasoning_effort="low",
            max_tokens=16384,
            use_local=use_local,
        )
        raw = (msg.get("content") or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)

        compressed_text = result.get("compressed_block", "")
        if not compressed_text or not compressed_text.strip():
            log.warning("Scratchpad block consolidation returned empty, skipping")
            return usage

        from ouroboros.tools.knowledge import _knowledge_write_lock

        with _knowledge_write_lock(knowledge_dir):
            _write_knowledge_entries(
                knowledge_dir, result.get("knowledge_entries", []), _locked=True,
            )
            _rebuild_knowledge_index(knowledge_dir, _locked=True)

        compressed_block = {
            "ts": utc_now_iso(),
            "source": "consolidation",
            "content": compressed_text.strip(),
        }

        source_bytes = json.dumps(
            old_blocks, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        source_entry_id = "scratchpad-consolidation:" + hashlib.sha256(source_bytes).hexdigest()
        source_ref = memory.scratchpad_journal_source_ref(source_entry_id)
        if not append_jsonl(memory.journal_path(), {
            "ts": utc_now_iso(),
            "type": "blocks_consolidated",
            "entry_id": source_entry_id,
            "source_blocks": old_blocks,
            "source_ref": source_ref,
        }):
            log.error("Scratchpad consolidation source journal write failed; preserving blocks")
            return usage
        compressed_block["metadata"] = {"source_ref": source_ref}

        # Merge-aware replace UNDER the write lock: blocks appended DURING the
        # slow LLM call live only on disk — building the new list from the
        # pre-call snapshot would silently drop them. Re-read inside the lock
        # and keep every block outside the compressed window (ts+source key).
        compressed_keys = {
            (str(b.get("ts") or ""), str(b.get("source") or "")) for b in old_blocks
        }

        def _merge_survivors(live_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            survivors = [
                b for b in live_blocks
                if (str(b.get("ts") or ""), str(b.get("source") or "")) not in compressed_keys
            ]
            return [compressed_block] + survivors

        new_blocks = memory.mutate_scratchpad_blocks(_merge_survivors)

        log.info("Scratchpad blocks consolidated: %d blocks (%d chars) -> %d blocks (%d chars)",
                 len(blocks), total_chars,
                 len(new_blocks), sum(len(b.get("content", "")) for b in new_blocks))
        return usage

    except Exception as e:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(e)
        log.error("Scratchpad block consolidation failed: %s", e, exc_info=True)
        return None


def _write_knowledge_entries(
    knowledge_dir: pathlib.Path,
    entries: List[Dict[str, Any]],
    *,
    _locked: bool = False,
) -> None:
    # Validate topics through the ONE knowledge-topic validator (P7/C9.4) instead of
    # a private char-filter that silently munged names into a different file than
    # the knowledge tool would. An invalid topic is skipped + logged, never coerced.
    from ouroboros.tools.knowledge import _sanitize_topic

    if not _locked:
        from ouroboros.tools.knowledge import _knowledge_write_lock

        with _knowledge_write_lock(knowledge_dir):
            _write_knowledge_entries(knowledge_dir, entries, _locked=True)
            _rebuild_knowledge_index(knowledge_dir, _locked=True)
        return

    knowledge_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        topic = entry.get("topic", "").strip()
        kb_content = entry.get("content", "").strip()
        if not topic or not kb_content:
            continue
        try:
            safe_topic = _sanitize_topic(topic)
        except ValueError:
            log.debug("consolidator: skipping invalid knowledge topic %r", topic)
            continue
        kb_path = knowledge_dir / f"{safe_topic}.md"
        existing = read_text(kb_path) if kb_path.exists() else ""
        write_text(kb_path, existing.rstrip() + "\n\n" + kb_content if existing else f"# {topic}\n\n{kb_content}\n")
