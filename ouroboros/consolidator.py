import hashlib
import json
import logging
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    total = sum(_count_lines(path) for path in segments if path.exists())
    if last_offset > total:
        return _count_lines(chat_path) >= BLOCK_SIZE
    return (total - last_offset) >= BLOCK_SIZE


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
        source_hash = hashlib.sha256(json.dumps([identity_text, formatted], ensure_ascii=False).encode("utf-8")).hexdigest()
        retry = meta.get("consolidation_retry") or {}

        def remember_refusal(input_limit: Dict[str, Any]) -> None:
            # The caller still holds .consolidation.lock. Persist before another
            # part can enter a quota wait or propagate an owner/deadline stop.
            meta["consolidation_retry"] = {"source_sha256": source_hash, "input_limit": input_limit}
            atomic_write_json(meta_path, meta)

        content, usage = _create_block_summary(
            llm_client=llm_client,
            messages_text=formatted,
            first_ts=first_ts,
            last_ts=last_ts,
            identity_text=identity_text,
            message_count=len(chunk),
            _retry=retry.get("input_limit") if retry.get("source_sha256") == source_hash else None,
            _on_refusal=remember_refusal,
        )

        total_usage = _merge_consolidation_usage(total_usage, usage)
        if (meta.get("consolidation_retry") or {}).get("source_sha256") == source_hash:
            meta.pop("consolidation_retry", None)
        if not content and usage.get("_consolidation_retry"):
            meta["consolidation_retry"] = {"source_sha256": source_hash, "input_limit": usage["_consolidation_retry"]}
        if usage.get("_consolidation_errors"):
            meta["last_consolidation_error"] = dict(
                usage["_consolidation_errors"][-1], cursor_offset=last_offset + processed,
                chat_log_signature=segment_sigs[0], message_count=len(chunk),
            )

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
        atomic_write_json(meta_path, meta)
        return total_usage

    existing_blocks = _load_blocks(blocks_path)
    all_blocks = existing_blocks + new_blocks

    if len(all_blocks) > MAX_SUMMARY_BLOCKS and content.strip():
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
            total_usage = _merge_consolidation_usage(total_usage, era_usage)
        if era is not None:
            all_blocks = [
                *old_blocks[:run_start], era, *old_blocks[run_end:], *remaining,
            ]

    _write_locked_json(blocks_path, all_blocks)

    _advance_cursor(meta, segments, segment_sigs, segment_entries, last_offset + processed)
    meta["last_consolidated_at"] = utc_now_iso()
    atomic_write_json(meta_path, meta)

    log.info("Block consolidation: %d messages -> %d new blocks (total %d)",
             processed, len(new_blocks), len(all_blocks))
    return total_usage


def _merge_consolidation_usage(*usages: Dict[str, Any]) -> Dict[str, Any]:
    """Combine helper usage without turning absent spend/counters into zero."""
    merged: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [usage.get(key) for usage in usages]
        merged[key] = None if None in values else sum(values)
    for key in ("ledger_attempt_ids", "_consolidation_errors"):
        merged[key] = [value for usage in usages for value in usage.get(key, [])]
    return merged


def _call_consolidation_llm(
    llm_client: Any, prompt: str, label: str, *, fixed_prompt: str = "",
    input_limit: Optional[Dict[str, Any]] = None,
    model_route: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    from contextlib import nullcontext
    from math import ceil
    from ouroboros.capability_evidence import is_known
    from ouroboros.context_budget import SummarizerContextOverflow
    from ouroboros.context_fit import (
        _failed_route_evidence, _route_calibration_ratio, estimate_context_prompt_tokens, resolve_context_fit_route,
    )
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
    from ouroboros.model_wait import current_model_wait
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model

    facts: Dict[str, Any] = {}
    prepared_values: Dict[str, Any] = {}
    model_route = model_route if model_route is not None else {}
    invoked = False
    waiter = current_model_wait()

    def prepare(values: Dict[str, Any], *, check_fit: bool = True) -> Dict[str, Any]:
        # Use the same role, captured pin (including Auto), local flag and
        # observed account as dispatch. Revalidate after an owner route switch.
        observed = values.pop("_model_observed_route", None)
        prepared_values.clear()
        prepared_values.update(values)
        task = {
            "model": values["model"], "use_local_model": values["use_local"],
            "model_role": values["model_role"],
            "credential_profile_id": values["model_account_override"],
            "model_route": observed,
        }
        try:
            # Local health and subscription catalogs establish identity without
            # a generation. Other providers keep their cache-only preparation.
            route, evidence = resolve_context_fit_route(
                task, allow_fetch=values["use_local"] or provider_for_model(values["model"]) == "claudexor",
            )
        except Exception:
            log.debug("Consolidation capacity unavailable; retaining unknown capacity", exc_info=True)
            try:
                route, evidence = _failed_route_evidence(task)
            except Exception:
                # The fallback shares the same settings reader. Its failure
                # cannot invalidate the Light request already captured above.
                facts.clear()
                model_route.clear()
                log.warning("Consolidation route metadata unavailable; capacity remains unknown", exc_info=True)
                return values
        options = route.get("options") or {}
        model_route.clear()
        if route["provider"] == "claudexor":
            source, native_model = parse_claudexor_model(route["model"])
            model_route.update(
                source=source, model=native_model,
                credentialProfileId=getattr(evidence, "credential_profile_id", "") or options.get("credential_profile_id", ""),
                accountFingerprint=getattr(evidence, "account_fingerprint", "") or options.get("account_fingerprint", ""),
            )
        density = _route_calibration_ratio(None, evidence.route_fp, route["model"])
        def measure(text: str) -> int:
            return ceil(estimate_context_prompt_tokens(
                [{"role": "user", "content": text}], values["tools"],
                provider=route["provider"], reasoning_effort=values["reasoning_effort"],
            ) * density)
        window = int(evidence.window_tokens) if is_known(evidence, require_fresh=True) else None
        output_reserve = values["max_tokens"]
        if values["use_local"]:
            from ouroboros.llm_local import local_context_limits
            _local_window, output_reserve = local_context_limits(output_reserve)
        limit = window - output_reserve if window is not None else None
        binding = dict(route_fp=evidence.route_fp, capacity_tokens=window, output_reserve_tokens=output_reserve)
        byte_limit = (input_limit["input_bytes"] if input_limit
                      and all(input_limit.get(key) == value for key, value in binding.items()) else None)
        facts.update(binding, input_tokens=measure(prompt), fixed_tokens=measure(fixed_prompt),
                     measurement_density=density, input_limit=limit, byte_limit=byte_limit,
                     input_bytes=len(prompt.encode("utf-8")), fixed_bytes=len(fixed_prompt.encode("utf-8")))
        if check_fit and ((limit is not None and facts["input_tokens"] > limit)
                          or (byte_limit is not None and facts["input_bytes"] > byte_limit)):
            raise SummarizerContextOverflow("Complete consolidation request exceeds the route input capacity")
        return values

    try:
        model, use_local = _consolidation_route()
        values = dict(messages=[{"role": "user", "content": prompt}], model=model,
                      model_role="light", tools=None, reasoning_effort="low", max_tokens=16384,
                      use_local=use_local,
                      model_account_override=model_role_option(MODEL_ACCOUNTS_KEY, "light"))
        if waiter:
            values.update(waiter.overrides.get("light", {}))
        # Carry part-to-part evidence only on initial preparation. A wait's
        # reprepare without an observed receipt rediscovers Auto after rotation.
        values = prepare({**values, "_model_observed_route": dict(model_route)})
        with waiter.register_reprepare("light", prepare) if waiter else nullcontext():
            invoked = True
            msg, usage = llm_client.chat(**values)
        if isinstance(usage.get("claudexor"), dict):
            model_route.clear()
            model_route.update(usage["claudexor"].get("route") or {})
        content = msg.get("content") or ""
        if content.strip():
            return content, usage
        kind, message, preflight = "empty_summary", "Consolidation returned no summary", False
    except Exception as error:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(error)
        from ouroboros.loop_llm_call import classify_llm_exception
        from ouroboros.transport_custody import _capture_on_chain

        capture = _capture_on_chain(error)
        if getattr(error, "route", None):
            # A refusal belongs to the actual account, which can differ from
            # catalog discovery. Rebind its facts without masking the refusal
            # with a second preflight exception or sending another request.
            prepare({**prepared_values, "_model_observed_route": error.route}, check_fit=False)
        preflight = isinstance(error, SummarizerContextOverflow) or not invoked
        kind = ("context_overflow" if isinstance(error, SummarizerContextOverflow)
                else "provider_outcome_unknown" if getattr(capture, "state", "") in {"dispatched", "unresolved"}
                else classify_llm_exception(error).kind)
        message = str(error)
        usage = dict(getattr(error, "usage", None) or {})
        usage.setdefault("cost", None if invoked else 0.0)
        usage["ledger_attempt_ids"] = list(getattr(error, "ledger_attempt_ids", []))
    from ouroboros.utils import sanitize_tool_result_for_log

    fact = dict(facts, kind=kind, message=sanitize_tool_result_for_log(message), preflight_only=preflight)
    log.warning("%s failed (%s): %s", label, kind, fact["message"])
    return "", {**usage, "_consolidation_errors": [fact]}


def _block_prompt(
    messages_text: str,
    first_ts: str,
    last_ts: str,
    identity_text: str,
    message_count: int,
) -> str:
    first_date = first_ts[:10]
    first_time = first_ts[11:16]
    last_time = last_ts[11:16]
    identity_section = f"\n## Identity context\n{identity_text}\n" if identity_text else ""
    return f"""You are a memory consolidator for Ouroboros, a self-modifying AI agent.
Create a detailed episodic memory entry from the supplied source of {message_count} messages.
The source may be one contiguous part of the block; summarize only the supplied part.

## Rules
1. Header: ### Block: {first_date} {first_time} - {last_time}
2. Preserve: decisions, agreements, technical discoveries, emotional moments, task outcomes, what worked/failed
3. Compress: routine tool calls, repetitive back-and-forth
4. Quote key phrases directly when important
5. First person as Ouroboros: "I did...", "the user asked..."
6. Length: 200-500 words depending on content density
7. Include task_ids when referencing specific tasks
{identity_section}
## Messages to summarize
{messages_text}
"""


def _split_consolidation_text(text: str) -> Optional[Tuple[str, str]]:
    """Split a source payload near its midpoint without dropping any bytes."""
    if len(text) < 2:
        return None
    midpoint = len(text) // 2
    radius = max(1, len(text) // 4)
    before = text.rfind("\n", 1, midpoint + 1)
    after = text.find("\n", midpoint, len(text) - 1)
    candidates = [p + 1 for p in (before, after) if p > 0 and abs((p + 1) - midpoint) <= radius]
    split_at = min(candidates, key=lambda p: abs(p - midpoint)) if candidates else midpoint
    if not 0 < split_at < len(text):
        return None
    return text[:split_at], text[split_at:]


def _create_block_summary(
    llm_client: Any, messages_text: str, first_ts: str, last_ts: str,
    identity_text: str, message_count: int,
    _retry: Optional[Dict[str, Any]] = None,
    _on_refusal: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Summarize a complete logical block, splitting only to fit its Light route.

    Parts cover the source in order without clipping. Any failed/empty part
    withholds the entire block and its cursor; unknown/control failures never
    authorize another part. A real refusal lowers the same route's byte limit
    for remaining parts and the next cycle, independent of density calibration.
    """
    pending, summaries, usages = [messages_text], [], []
    model_route: Dict[str, Any] = {}
    input_limit = _retry
    def result(content: str) -> Tuple[str, Dict[str, Any]]:
        return content, {**_merge_consolidation_usage(*usages), "_consolidation_retry": input_limit}

    fixed = _block_prompt("", first_ts, last_ts, identity_text, message_count)
    while pending:
        part = pending.pop()
        prompt = _block_prompt(part, first_ts, last_ts, identity_text, message_count)
        content, usage = _call_consolidation_llm(
            llm_client, prompt, "Block summary LLM call", fixed_prompt=fixed, input_limit=input_limit,
            model_route=model_route,
        )
        usages.append(usage)
        if content.strip():
            summaries.append(content.strip())
            continue
        failure = usage["_consolidation_errors"][-1]
        if failure["kind"] != "context_overflow":
            return result("")
        if not failure["preflight_only"] and "input_bytes" in failure:
            input_limit = {key: failure[key] for key in (
                "route_fp", "capacity_tokens", "output_reserve_tokens",
            )}
            input_limit["input_bytes"] = failure["input_bytes"] - 1
            failure["byte_limit"] = input_limit["input_bytes"]
            if _on_refusal is not None:
                _on_refusal(input_limit)
        split = _split_consolidation_text(part)
        if split is None or any(failure.get(limit) is not None and failure[fixed] > failure[limit]
                                for fixed, limit in (("fixed_tokens", "input_limit"), ("fixed_bytes", "byte_limit"))):
            return result("")
        pending.extend(reversed(split))
    return result("\n\n".join(summaries))


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

    content, usage = _call_consolidation_llm(llm_client, prompt, "Era compression")
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


def _try_extract_embedded_json(raw: str) -> Optional[Dict[str, Any]]:
    """One bounded recovery: a model reply that wraps JSON inside prose or fences.

    Finds the FIRST ``{`` and the matching LAST ``}`` (counting nested braces so
    prose after the JSON does not truncate it), then parses the slice. Returns
    ``None`` when no balanced ``{...}`` slice parses as a dict — caller defers.
    """
    if not raw:
        return None
    first = raw.find("{")
    if first < 0:
        return None
    last = raw.rfind("}")
    if last <= first:
        return None
    depth = 0
    end = -1
    in_string = False
    escape = False
    for idx in range(first, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end < 0:
        return None
    candidate = raw[first : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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

        # BIBLE P1 — gap-not-lost: an empty/non-JSON model reply is a recoverable
        # transient (e.g. 429-rate-limited fallback returning ""), NOT a fatal
        # consolidation error. Defer cleanly with usage so the next pass retries;
        # the broad outer except stays for genuinely unexpected failures (lock /
        # IO), but a transient bad model reply must NEVER reach it.
        if not raw:
            log.warning("Scratchpad block consolidation: model returned empty response, deferring")
            return usage
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # One bounded recovery: fenced/embedded JSON inside prose (find first
            # `{` and the matching last `}`). If even that fails, defer cleanly.
            recovered = _try_extract_embedded_json(raw)
            if recovered is None:
                log.warning(
                    "Scratchpad block consolidation: non-JSON model response, deferring: %s",
                    raw[:120],
                )
                return usage
            result = recovered

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
