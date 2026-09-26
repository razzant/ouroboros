import hashlib
import json
import logging
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros import room_consolidation
from ouroboros.utils import append_jsonl, atomic_write_json, replace_atomic, utc_now_iso, read_text

from ouroboros.platform_layer import (
    file_lock_exclusive as _lock_ex, file_lock_exclusive_nb as _lock_nb, file_unlock as _unlock,
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


def _light_dispatch_binding() -> Dict[str, Any]:
    """The EFFECTIVE Light binding a call dispatches on NOW, in dispatch's own field names.

    The configured lane, then the Light account pin, then the live model-wait
    override for the role — exactly what ``_call_consolidation_llm`` sends, so a
    key derived here changes whenever the physical dispatch would."""
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
    from ouroboros.model_wait import current_model_wait

    model, use_local = _consolidation_route()
    binding: Dict[str, Any] = {"model": model, "use_local": use_local,
                               "model_account_override": model_role_option(MODEL_ACCOUNTS_KEY, "light")}
    waiter = current_model_wait()
    if waiter:
        binding.update(waiter.overrides.get("light", {}))
    return binding


def _light_route() -> Any:
    """The ``era_retry`` dispatch key: the effective binding of ``_light_dispatch_binding``.

    A dispatch key, never a provenance stamp: what actually answered is
    ``knowledge.observed_route_stamp`` over the returned usage. An account pin or
    a model-wait override changes this key (the paid retry is allowed) exactly
    because it changes the dispatch; an empty (Auto) account is omitted so a
    record keyed before the account joined still holds on an unchanged route."""
    try:
        binding = _light_dispatch_binding()
    except Exception:
        return "unknown"
    account = binding.get("model_account_override") or ""
    return {"model": binding["model"], "use_local": binding["use_local"],
            **({"model_account_override": account} if account else {})}


def _route_stamp(usage: Any) -> Any:
    """The route the usage says answered, for a history stamp; unknown without a physical fact."""
    from ouroboros.knowledge import observed_route_stamp

    return observed_route_stamp(usage)


def _emit_event(logs_dir: pathlib.Path, kind: str, **fields: Any) -> None:
    """A memory-maintenance outcome is a typed fact beside the chat it concerns, never silence (I4)."""
    try:
        append_jsonl(logs_dir / "events.jsonl", {"ts": utc_now_iso(), "type": kind, **fields})
    except Exception:
        log.debug("Failed to emit %s event", kind, exc_info=True)


def retain_memory_source(context: Any, source_id: str, data: bytes, extension: str = "md") -> Dict[str, Any]:
    """Use existing immutable source storage with a reader valid after this task."""
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path
    root, task_id = pathlib.Path(context.drive_root).resolve(), str(context.task_id or "consolidation")
    ref = store_actor_source_bytes(root, task_id, category="context_checkpoints",
                                  source_id=source_id, data=data, extension=extension)
    path = task_artifact_dir_path(root, task_id, create=False) / ref["path"]
    return {**ref, "task_id": task_id, "canonical_root": str(root), "read": {"tool": "read_file",
            "arguments": {"root": "runtime_data", "path": path.relative_to(root).as_posix(), "start_line": 1}}}


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
    chat_path: pathlib.Path, blocks_path: pathlib.Path, meta_path: pathlib.Path, llm_client: Any,
    identity_text: str = "", *, knowledge_context: Any = None, force_tail: bool = False,
    compact_chronicle: bool = False, pressure_fits: Optional[Callable[[], bool]] = None,
    room_registry_root: Any = None,
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
            _emit_event(chat_path.parent, "consolidation_skipped_locked", lock_path=str(lock_path),
                        task_id=str(getattr(knowledge_context, "task_id", "") or ""))
            return None

        usage = _run_block_consolidation(
            source_path=chat_path, blocks_path=blocks_path, meta_path=meta_path, llm_client=llm_client,
            identity_text=identity_text, knowledge_context=knowledge_context, force_tail=force_tail,
            room_registry_root=room_registry_root)
        if (compact_chronicle and not (usage or {}).get("_consolidation_errors")
                and not (pressure_fits is not None and pressure_fits())):
            reduced = _compact_chronicle(blocks_path, llm_client, identity_text, knowledge_context,
                                         meta_path=meta_path)
            merged = _merge_consolidation_usage(*([usage] if usage else []), reduced)
            # A fixed-key merge would drop this receipt; a chronicle-only pass wrote no block.
            merged["_blocks_written"] = (usage or {}).get("_blocks_written", 0)
            usage = merged
        return usage
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
    knowledge_context: Any = None,
    force_tail: bool = False,
    room_registry_root: Any = None,
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
    if not new_entries or (len(new_entries) < BLOCK_SIZE and not force_tail):
        return None

    total_usage: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
    # A failed chunk withholds only itself (its earlier sibling chunks are
    # complete units); the stale-error clear below must know whether THIS run
    # recorded a failure it would otherwise erase.
    run_failed = False
    new_blocks: List[Dict[str, Any]] = []
    from ouroboros.dialogue_provenance import RoomLabelResolver

    # The chat log may live in a forked/task drive while projects.json remains
    # on the canonical data root.  Provenance must follow the registry root,
    # never the incidental location of the source bytes.
    registry_root = room_registry_root
    if registry_root is None and knowledge_context is not None:
        registry_root = (getattr(knowledge_context, "budget_drive_root", None)
                         or getattr(knowledge_context, "drive_root", None))
    room_resolver = RoomLabelResolver(registry_root or source_path.parent.parent)
    chunks_to_process = (len(new_entries) + BLOCK_SIZE - 1) // BLOCK_SIZE if force_tail else len(new_entries) // BLOCK_SIZE
    processed = 0
    knowledge_instruction = (KNOWLEDGE_MAINTENANCE_PROMPT + "\nAfter the episodic summary, optionally add "
                             'a final line KNOWLEDGE_ENTRIES_JSON: [{"topic":"...","scope":"global","edits":[...]}] ("content" for a new topic).\n'
                             if knowledge_context is not None else "")
    block = None

    for i in range(chunks_to_process):
        chunk = new_entries[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
        # The host partitions by the actual chat id BEFORE any model call: each
        # room's episodic summary uses its own chronological bytes; cumulative
        # knowledge can span rooms. Each section's identity is a host fact.
        rooms = room_consolidation.partition_entries(chunk, room_resolver)
        first_ts = str(chunk[0].get("ts", "unknown"))
        last_ts = str(chunk[-1].get("ts", "unknown"))
        source_hash = hashlib.sha256(json.dumps(
            [identity_text, [room.text for room in rooms]], ensure_ascii=False).encode("utf-8")).hexdigest()
        retry = meta.get("consolidation_retry") or {}

        def remember_refusal(input_limit: Dict[str, Any]) -> None:
            # The caller still holds .consolidation.lock. Persist before another
            # part can enter a quota wait or propagate an owner/deadline stop.
            meta["consolidation_retry"] = {"source_sha256": source_hash, "input_limit": input_limit}
            atomic_write_json(meta_path, meta)

        block, usage = room_consolidation.summarize_block(
            _light_call(llm_client, knowledge_context, {}), rooms, first_ts=first_ts, last_ts=last_ts,
            identity_text=identity_text, knowledge_instruction=knowledge_instruction,
            input_limit=retry.get("input_limit") if retry.get("source_sha256") == source_hash else None,
            on_refusal=remember_refusal,
        )

        total_usage = _merge_consolidation_usage(total_usage, usage)
        if (meta.get("consolidation_retry") or {}).get("source_sha256") == source_hash:
            meta.pop("consolidation_retry", None)
        if block is None and usage.get("_consolidation_retry"):
            meta["consolidation_retry"] = {"source_sha256": source_hash, "input_limit": usage["_consolidation_retry"]}
        # A refused part that was split and then fully summarized still carries its
        # attempt errors in usage; only a chunk that produced NO block failed.
        if usage.get("_consolidation_errors") and block is None:
            run_failed = True
            meta["last_consolidation_error"] = dict(
                usage["_consolidation_errors"][-1], cursor_offset=last_offset + processed,
                chat_log_signature=segment_sigs[0], message_count=len(chunk))

        if block is None:
            log.warning("Block summary withheld for chunk %d, will retry next cycle", i)
            break
        new_blocks.append({
            "ts": utc_now_iso(), "type": "summary", "message_count": len(chunk), **block,
            **({"knowledge_entries": usage["_knowledge_entries"], "_nomination_route": _route_stamp(usage)}
               if usage.get("_knowledge_entries") else {})})
        processed += len(chunk)

    # Set after the last merge of this stretch: _merge_consolidation_usage forwards
    # fixed keys only, so an earlier assignment would be dropped by the next merge.
    # The transaction boundary is the logical chunk: every room section and its
    # correction succeed before that chunk's block exists at all.  Earlier
    # chunks of the same run are complete units and stay published, so a
    # transient failure on chunk N never discards N-1 finished chunks (that
    # would let a flaky route starve the cursor forever); the failed chunk is
    # retried from its own offset next cycle.
    total_usage["_blocks_written"] = len(new_blocks)
    if not new_blocks:
        atomic_write_json(meta_path, meta)
        return total_usage

    # The route that nominated a block's entries is a history stamp for the
    # writes below, never a persisted block field; receipts keep their pairs.
    nomination_routes = {id(block): block.pop("_nomination_route", "unknown") for block in new_blocks}
    pending_knowledge = [(block, block.pop("knowledge_entries")) for block in new_blocks
                         if block.get("knowledge_entries")]
    if pending_knowledge:
        source = {"path": str(source_path), "generations": segment_sigs,
                  "start_offset": last_offset, "end_offset": last_offset + processed,
                  "nominations": [{"range": block["range"], "entries": entries} for block, entries in pending_knowledge]}
        source_id = hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        ref = {"read": {"tool": "read_file", "arguments": {
            "root": "runtime_data", "path": "memory/knowledge_history.jsonl"}},
            "canonical_root": str(knowledge_context.drive_root), "entry_id": source_id}
        if not append_jsonl(pathlib.Path(knowledge_context.drive_root) / "memory" / "knowledge_history.jsonl", {
            "ts": utc_now_iso(), "type": "dialogue_knowledge_nominations", "entry_id": source_id,
            "source_ref": ref, **source,
        }, ensure_record_boundary=True, require_lock=True):
            log.warning("Dialogue knowledge nominations could not be retained; preserving original blocks/cursor")
            total_usage["_blocks_written"] = 0
            return total_usage
        for nominated_block, _entries in pending_knowledge:
            nominated_block["knowledge_source_ref"] = ref
        if knowledge_context is not None:
            from ouroboros.memory_nomination_receipts import prepare
            pending_ids = prepare(meta, source_id, pending_knowledge)
            atomic_write_json(meta_path, meta)  # Debt precedes block and note publication.

    existing_blocks = _load_blocks(blocks_path)
    all_blocks = existing_blocks + new_blocks

    if len(all_blocks) > MAX_SUMMARY_BLOCKS and block is not None:
        # Gap markers are DURABLE discontinuity facts (BIBLE P1) that keep their
        # chronological positions, and an earlier era is a boundary too: an era
        # compresses ONE CONTIGUOUS run of ordinary summary blocks, never a span
        # bridging a discontinuity and never a summary of its own summary. The
        # run is the OLDEST run of up to ERA_COMPRESS_COUNT summary blocks anywhere
        # before the newest block — eras and gaps ahead of it are skipped, not a
        # reason to stop compressing (a window of the first four blocks went blind
        # once those four were eras).
        run_start = next((i for i, b in enumerate(all_blocks[:-1]) if not _is_run_boundary(b)), None)
        era = None
        if run_start is not None:
            run_end = run_start
            while (run_end < len(all_blocks) - 1 and run_end - run_start < ERA_COMPRESS_COUNT
                   and not _is_run_boundary(all_blocks[run_end])):
                run_end += 1
            era, era_usage = _era_for_run(all_blocks[run_start:run_end], meta, source_path.parent,
                                          llm_client, identity_text, knowledge_context)
            if era_usage is not None:
                total_usage = _merge_consolidation_usage(total_usage, era_usage)
        if era is not None:
            all_blocks = [*all_blocks[:run_start], era, *all_blocks[run_end:]]

    _write_locked_json(blocks_path, all_blocks)

    if knowledge_context is not None:
        published: List[Dict[str, Any]] = []
        for block, entries in pending_knowledge:
            block["knowledge_writes"] = _write_knowledge_entries(
                pathlib.Path(knowledge_context.drive_root) / "memory" / "knowledge",
                entries, context=knowledge_context, stamp={
                    "writer": "consolidation", "route": nomination_routes.get(id(block), "unknown"),
                    "writer_input_ref": block["knowledge_source_ref"]})
            published.extend(block["knowledge_writes"])
            if any(not outcome["ok"] for outcome in block["knowledge_writes"]):
                append_jsonl(pathlib.Path(knowledge_context.drive_root) / "memory" / "knowledge_history.jsonl", {
                    "ts": utc_now_iso(), "type": "dialogue_knowledge_writes_incomplete",
                    "source_ref": block["knowledge_source_ref"], "outcomes": block["knowledge_writes"],
                })
        if pending_knowledge:
            _write_locked_json(blocks_path, all_blocks)
            from ouroboros.memory_nomination_receipts import settle
            settle(meta, pending_ids, published)
            # Legacy batch-only receipts remain open: no positional evidence can
            # prove which old entry a later successful nomination resolved.

    _advance_cursor(meta, segments, segment_sigs, segment_entries, last_offset + processed)
    if not run_failed:  # An advance by a run that recorded no failure retires a stale error.
        meta.pop("last_consolidation_error", None)
    meta["last_consolidated_at"] = utc_now_iso()
    atomic_write_json(meta_path, meta)

    log.info("Block consolidation: %d messages -> %d new blocks (total %d)",
             processed, len(new_blocks), len(all_blocks))
    total_usage["_blocks_written"] = len(new_blocks)
    return total_usage


def _light_call(llm_client: Any, knowledge_context: Any, model_route: Dict[str, Any]) -> room_consolidation.LightCall:
    """Bind the Light transport for one logical unit: same route evidence, fresh read context per call."""
    def call(prompt: str, label: str, *, fixed_prompt: str = "", input_limit: Optional[Dict[str, Any]] = None,
             call_type: str = "memory_consolidation") -> Tuple[str, Dict[str, Any], Any]:
        knowledge = KnowledgeReadContext(knowledge_context, call_type) if knowledge_context is not None else None
        content, usage = _call_consolidation_llm(
            llm_client, prompt, label, fixed_prompt=fixed_prompt, input_limit=input_limit,
            model_route=model_route, knowledge=knowledge)
        return content, usage, knowledge
    return call


def _merge_consolidation_usage(*usages: Dict[str, Any]) -> Dict[str, Any]:
    """Combine helper usage without turning absent spend/counters into zero."""
    from ouroboros.knowledge import observed_route_stamp

    merged: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [usage.get(key) for usage in usages]
        merged[key] = None if None in values else sum(values)
    for key in ("ledger_attempt_ids", "_consolidation_errors"):
        merged[key] = [value for usage in usages for value in usage.get(key, [])]
    # The route that answered the LAST send of this unit, and only that one: a
    # physical usage carries provider/resolved_model, a merged one its forwarded
    # stamp, and a final call without a physical fact reads unknown — an earlier
    # call's stamp never masquerades as the final call's.
    if usages:
        last = observed_route_stamp(usages[-1])
        if isinstance(last, dict):
            merged["_observed_route"] = last
    return merged


class KnowledgeReadContext:
    """Reads made by this existing Light operation, before its nominations."""

    def __init__(self, context: Any, call_type: str = "memory_consolidation"):
        from ouroboros.tools.knowledge import get_tools
        from ouroboros.tools.compact_context import get_tools as context_tools
        from ouroboros.tools.core import get_tools as core_tools

        self.context = context
        self.call_type = call_type
        self.reads: Dict[Tuple[str, str], str] = {}
        self.read_ranges: Dict[Tuple[str, str, str], Tuple[int, List[Tuple[int, int]]]] = {}
        self.pending_delivery: List[Dict[str, Any]] = []
        self.required_source: Optional[Dict[str, Any]] = None
        self.tools = [{"type": "function", "function": tool.schema} for tool in get_tools()
                      if tool.name in {"knowledge_read", "knowledge_list"}]
        self.tools.extend({"type": "function", "function": tool.schema} for tool in context_tools())
        self.tools.extend({"type": "function", "function": tool.schema} for tool in core_tools() if tool.name == "read_file")

    def read_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        from ouroboros.tools.knowledge import _knowledge_list, _knowledge_read
        from ouroboros.tools.tool_result import (
            _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
        )

        function = call.get("function") or {}
        name = function.get("name")
        meta, status = {}, "error"
        try:
            arguments = function.get("arguments") or "{}"
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if name in {"knowledge_read", "read_file"}:
                sentinel = object()
                token = _install_tool_result_sidecar(self.context, sentinel)
                try:
                    if name == "read_file":
                        from ouroboros.tools.core_file_tools import _read_file
                        text = _read_file(self.context, **args)
                    else:
                        text = _knowledge_read(self.context, **args)
                    result = _published_tool_result(self.context, sentinel)
                    meta = dict(getattr(result, "meta", {}))
                    status = getattr(result, "status", "")
                    if name == "read_file" and self.context.last_read_view:
                        meta["read_file_source"] = dict(self.context.last_read_view)
                        status = "ok"
                finally:
                    _restore_tool_result_sidecar(token)
            elif name == "knowledge_list":
                text = _knowledge_list(self.context, **args)
            elif name == "compact_context":
                from ouroboros.tools.compact_context import _compact_context
                text = _compact_context(self.context, **args)
            else:
                text = "This memory operation supports knowledge_read, knowledge_list, read_file and compact_context."
        except (ValueError, KeyError, TypeError, OSError) as exc:
            text = f"Knowledge read unavailable: {type(exc).__name__}: {exc}"
        return {"tool_call_id": str(call.get("id") or ""), "fn_name": name,
                "result": text, "result_meta": meta, "status": status}

    def accept_delivery(self) -> None:
        """Credit only source characters in the request the model answered."""
        for row in self.pending_delivery:
            if row.get("status") != "ok":
                continue
            meta = row.get("result_meta") or {}
            source = meta.get("knowledge_source") or {}
            try:
                file = meta.get("read_file_source")
                if file:
                    if file.get("source_masked"):
                        continue
                    scope, topic, revision = file["opened_root"], file["opened_path"], file["source_revision"]
                    total, start, end = file["complete_chars"], file["source_start_char"], file["source_end_char"]
                    header, body = file["body_start"], file["body_chars"]
                else:
                    args = source["read"]["arguments"]
                    scope, topic, revision = args["scope"], args["topic"], source["revision"]
                    total, start, end = source["complete_chars"], source["start_char"], source["end_char"]
                    header, body = meta["knowledge_body_start"], meta["knowledge_body_chars"]
                if (not all(type(n) is int for n in (total, start, end, header, body))
                        or not 0 <= start <= end <= total or body != end - start or header < 0):
                    continue
                shown = (row["result_source_view"]["delivered_range"][1]
                         if row.get("result_partial") else len(row["result"]))
                if shown < header:
                    continue
                delivered_end = start + min(body, shown - header)
                key = (scope, topic, revision)
                old_total, ranges = self.read_ranges.get(key, (total, []))
                self.reads.pop((scope, topic), None)
                if old_total != total:
                    continue
                merged: List[Tuple[int, int]] = []
                for lo, hi in sorted([*ranges, (start, delivered_end)]):
                    if merged and lo <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                    else:
                        merged.append((lo, hi))
                self.read_ranges[key] = (total, merged)
                if merged == [(0, total)]:
                    self.reads[(scope, topic)] = revision
            except (KeyError, TypeError, ValueError):
                continue
        self.pending_delivery = []

    def source_complete(self) -> bool:
        ref = self.required_source
        args = ref["read"]["arguments"] if ref else {}
        return ref is None or self.reads.get((args["root"], args["path"])) == ref["sha256"]

    def next_messages(self, values: dict, message: dict, *, fit_candidate: Callable,
                      facts: dict, round_id: str) -> list:
        """Complete one tool batch, apply the actor's view, then project its results."""
        from dataclasses import asdict
        from ouroboros.context_budget import ContextReclaimRequest, SummarizerContextOverflow
        from ouroboros.context_compaction import compact_tool_history_llm, context_reclaim_transcript_sha256
        from ouroboros.context_fit import project_tool_result_batch

        rows = [self.read_call(call) for call in message["tool_calls"]]
        before = [*values["messages"], {**message, "role": "assistant"}]

        def tool_messages(results):
            return [{"role": "tool", "tool_call_id": row["tool_call_id"], "content": row["result"]}
                    for row in results]

        pending = getattr(self.context, "_pending_compaction", None)
        if pending is not None:
            self.context._pending_compaction = None
            receipt = {"status": "authored_note_required",
                       "reason": "This Light operation uses your authored view. Call inspect=true, then supply working_note and keep_unit_ids; no helper was called."}
            if isinstance(pending, dict):
                observed = pending["observed"]
                complete = [*before, *tool_messages(rows)]
                request = ContextReclaimRequest(
                    route_fp=str(facts.get("route_fp") or "unknown"), round_id=round_id,
                    transcript_sha256=context_reclaim_transcript_sha256(complete),
                    measurement_basis="cold_estimate", measurement_density=facts.get("measurement_density", 1.0),
                    reclaim_goal_tokens=0,
                    **{key: pending[key] for key in ("working_note", "expected_view_revision", "keep_unit_ids", "restore_unit_refs", "schema_names")})
                candidate, applied, _usage = compact_tool_history_llm(
                    complete, request=request, observed_messages=observed["messages"],
                    observed_tool_schemas=observed["tool_schemas"], tool_schemas=values["tools"],
                    fit_candidate=fit_candidate, drive_root=self.context.drive_root,
                    task_id=str(self.context.task_id or "consolidation"))
                facts_receipt = asdict(applied)
                # Full provenance already lives in the checkpoint/capsule.
                # Replaying that growing lineage in every tool reply can fill
                # the very window the actor just reclaimed.
                receipt = {key: facts_receipt[key] for key in (
                    "status", "checkpoint_ref", "view_revision", "reclaimed_tokens", "fit") if key in facts_receipt}
                receipt["receipt_ref"] = retain_memory_source(self.context, "memory_context_view",
                    json.dumps(facts_receipt, ensure_ascii=False).encode("utf-8"), "json")
                if applied.status in {"applied", "no_op"}:
                    before = candidate[:-len(rows)]  # the new completed batch is preserved verbatim
            for row in rows:
                if row["fn_name"] == "compact_context":
                    row["result"] = json.dumps({"context_view": receipt}, ensure_ascii=False)
        projected, projection = project_tool_result_batch(
            rows, before, values["tools"], drive_root=self.context.drive_root,
            task_id=str(self.context.task_id or "consolidation"), fit_candidate=fit_candidate)
        self.pending_delivery = projected
        if projection["status"] == "minimum_view_unfit":
            raise SummarizerContextOverflow("Memory tool-result source locators exceed the current working window")
        return [*before, *tool_messages(projected)]

    def bind_entries(self, entries: Any) -> List[Dict[str, Any]]:
        from ouroboros.tools.knowledge import _address

        bound = []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            try:
                address = _address(self.context, entry.get("topic"), entry.get("scope", ""))
            except (ValueError, TypeError):
                continue
            # The host records what THIS operation actually read. Model-supplied
            # revision text cannot attest an unread note; absent read permits
            # creation only, because the common writer requires existing CAS.
            # Underscore keys are host facts (``_nomination_route`` and any later
            # one): a model-supplied value is dropped here, so a forged route can
            # never reach history — the host stamps only after this binding.
            bound.append({**{key: value for key, value in entry.items() if not str(key).startswith("_")},
                          "scope": address.scope,
                          "expected_revision": self.reads.get((address.scope, address.topic)),
                          "canonical_root": str(address.canonical_root),
                          "task_id": str(getattr(self.context, "task_id", "") or "")})
        return bound


KNOWLEDGE_MAINTENANCE_PROMPT = """
You may use knowledge_list and knowledge_read to understand existing notes before
nominating a durable revision. An episodic summary describes only its supplied source;
a knowledge note is cumulative understanding, grounded in the complete CURRENT note
you read in this operation together with the new episode. Absence from this episode
does not refute prior knowledge; an earlier episode cutoff does not undo later known
events. Preserve useful established facts, sources, uncertainty, unknown metadata and
links. Correct, remove or reorganize stale or unsupported understanding when the
evidence warrants it; memory is revisable, not append-only. Read the whole current
note before changing it, rather than merely repeating fragments. An existing note changes by
"edits": [{"old_text": a passage occurring exactly once in its body, "new_text": its replacement,
empty to remove it, "basis": source and reason}]; spans never overlap and unmentioned text stays,
so a broader rewrite or reorganization quotes the whole span it replaces. Edits reach only the body;
revise the summary with "summary": "new text" beside them (other metadata survives). A new topic
takes only "content" with complete Markdown and needs no prior read.
Understanding of the people involved — preferences, recurring reactions, shared history,
tentative interpretations with their source — is ordinary knowledge to nominate in global scope;
a pattern across several moments is worth more than one; revise the existing note rather than minting a rule,
and an explicit standing request stays explicit. Author a YAML summary for a new or meaningfully revised note —
the summary is what stays resident in the index — and revise it when the note's meaning changes.
The note overview (scope global) is the shared orientation loaded into every future context; keep it
current, and when none exists and this episode gives real understanding, create it after reading the index.
Scope is a separate field, never a topic prefix.
Do not treat the generated index or earlier previews as authored truth. Patterns and
improvement-backlog retain their dedicated semantic maintainers; nominate ordinary
knowledge here. If no memory change is useful, nominate none. This is the same memory
operation, not another mandatory analysis or review.
Read large notes using explicit start_char/end_char ranges. Read coverage belongs
to one exact revision; repeat or overlapping reads do not fill unread gaps.
Use compact_context(inspect=true) before the view fills, then give your own
working_note and selected complete unit IDs to retain. Source checkpoints preserve
the original reads; keep your current conclusions while reading the next range.
This Light operation supports authored views; keep_last_n alone returns guidance
without calling another helper. All tools remain available in this operation.
"""


def _call_consolidation_llm(
    llm_client: Any, prompt: str, label: str, *, fixed_prompt: str = "",
    input_limit: Optional[Dict[str, Any]] = None,
    model_route: Optional[Dict[str, Any]] = None,
    knowledge: Optional[KnowledgeReadContext] = None,
    source_ref: Optional[Dict[str, Any]] = None,
    reasoning_effort: str = "low",
) -> Tuple[str, Dict[str, Any]]:
    from contextlib import nullcontext
    from math import ceil
    from ouroboros.capability_evidence import is_known
    from ouroboros.context_budget import SummarizerContextOverflow
    from ouroboros.context_fit import (
        _failed_route_evidence, _route_calibration_ratio, estimate_context_prompt_tokens, resolve_context_fit_route,
    )
    from ouroboros.tools.compact_context import record_context_view
    from ouroboros.model_wait import current_model_wait
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model

    facts: Dict[str, Any] = {}
    prepared_values: Dict[str, Any] = {}
    model_route = model_route if model_route is not None else {}
    invoked = False
    usages: List[Dict[str, Any]] = []
    waiter = current_model_wait()
    if knowledge:
        from ouroboros.llm_claudexor import ModelTurnState
        turn_state = ModelTurnState()

    def prepare(values: Dict[str, Any], *, check_fit: bool = True) -> Dict[str, Any]:
        # Use the same role, captured pin (including Auto), local flag and
        # observed account as dispatch. Revalidate after an owner route switch.
        observed = values.pop("_model_observed_route", None)
        if knowledge:
            from ouroboros.llm_claudexor import turn_state_for_route
            values["model_turn_state"] = turn_state_for_route(
                turn_state, "local" if values["use_local"] else provider_for_model(values["model"]))
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
        def measure(messages: List[Dict[str, Any]]) -> int:
            return ceil(estimate_context_prompt_tokens(
                messages, values["tools"],
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
        request_text = (values["messages"][0]["content"] if len(values["messages"]) == 1 else
                        json.dumps(values["messages"], ensure_ascii=False, separators=(",", ":")))
        facts.update(binding, input_tokens=measure(values["messages"]), provider=route["provider"],
                     fixed_tokens=measure([{"role": "user", "content": fixed_prompt}]),
                     measurement_density=density, input_limit=limit, byte_limit=byte_limit,
                     input_bytes=len(request_text.encode("utf-8")), fixed_bytes=len(fixed_prompt.encode("utf-8")))
        if check_fit and ((limit is not None and facts["input_tokens"] > limit)
                          or (byte_limit is not None and facts["input_bytes"] > byte_limit)):
            raise SummarizerContextOverflow("Complete consolidation request exceeds the route input capacity")
        return values

    def fit_candidate(messages: list, tools: list) -> Dict[str, Any]:
        # Reuse the captured preparation facts; binary-searching a view performs
        # no catalog/network reads and never changes the operation's model route.
        tokens = ceil(estimate_context_prompt_tokens(
            messages, tools, provider=facts.get("provider", ""),
            reasoning_effort=prepared_values.get("reasoning_effort")) * facts.get("measurement_density", 1.0))
        text = messages[0]["content"] if len(messages) == 1 else json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        size = len(text.encode("utf-8"))
        accepted = ((facts.get("input_limit") is None or tokens <= facts["input_limit"])
                    and (facts.get("byte_limit") is None or size <= facts["byte_limit"]))
        return {"accepted": accepted, "input_tokens": tokens, "input_bytes": size,
                "input_limit": facts.get("input_limit"), "output_reserve_tokens": facts.get("output_reserve_tokens"),
                "measurement_basis": "canonical_visible_estimate", "strict_bound_proven": False}

    try:
        # The same effective binding keys era_retry (``_light_route``): a refusal
        # recorded under one binding never suppresses the retry under another.
        values = dict(messages=[{"role": "user", "content": prompt}],
                      model_role="light", tools=knowledge.tools if knowledge else None,
                      reasoning_effort=reasoning_effort, max_tokens=16384,
                      **_light_dispatch_binding())
        # Carry part-to-part evidence only on initial preparation. A wait's
        # reprepare without an observed receipt rediscovers Auto after rotation.
        try:
            values = prepare({**values, "_model_observed_route": dict(model_route)})
        except SummarizerContextOverflow:
            if knowledge is None:
                raise
            source_ref = source_ref or retain_memory_source(knowledge.context, label, prompt.encode("utf-8"))
            knowledge.required_source = source_ref
            pointer = (f"Complete source and instructions for {label} are retained here. "
                       "Read the whole source through read_file in ranges before your final response. "
                       "Use compact_context with your authored working_note while progressing through ranges; "
                       "preserve the whole temporal horizon, uncertainty and source references. "
                       "This locator is not a summary. Knowledge reads and all current tools remain available.\n"
                       + json.dumps(source_ref, ensure_ascii=False))
            values = prepare({**prepared_values, "messages": [{"role": "user", "content": pointer}],
                              "_model_observed_route": dict(model_route)})
        while True:
            def reprepare_with_route(next_values: Dict[str, Any]) -> Dict[str, Any]:
                observed = next_values.get("_model_observed_route")
                if isinstance(observed, dict):
                    model_route.clear()
                    model_route.update(observed)
                return prepare(next_values)

            with waiter.register_reprepare("light", reprepare_with_route) if waiter else nullcontext():
                invoked = True
                if knowledge:
                    from ouroboros.llm_observability import chat_observed
                    record_context_view(knowledge.context, values["messages"], values["tools"])
                    msg, usage = chat_observed(
                        llm_client, drive_root=knowledge.context.drive_root,
                        task_id=str(knowledge.context.task_id or "consolidation"),
                        call_type=knowledge.call_type, **values)
                else:
                    msg, usage = llm_client.chat(**values)
            usages.append(usage)
            if knowledge:
                # A wait may have re-prepared this same call with another route;
                # pin its final canonical view before executing the returned tools.
                record_context_view(knowledge.context, prepared_values["messages"], prepared_values["tools"])
                knowledge.accept_delivery()
            if isinstance(usage.get("claudexor"), dict):
                model_route.clear()
                model_route.update(usage["claudexor"].get("route") or {})
            calls = msg.get("tool_calls") or []
            if knowledge is not None and calls:
                invoked = False
                messages = knowledge.next_messages(prepared_values, msg, fit_candidate=fit_candidate,
                                                    facts=facts, round_id=str(len(usages)))
                values = prepare({**prepared_values, "messages": messages, "_model_observed_route": dict(model_route)})
                continue
            content = msg.get("content") or ""
            if content.strip():
                if knowledge and not knowledge.source_complete():
                    response_ref = retain_memory_source(knowledge.context, "incomplete_memory_response", content.encode("utf-8"))
                    return "", {**_merge_consolidation_usage(*usages), "_consolidation_errors": [{
                        "kind": "source_incomplete", "label": label,
                        "message": "The complete retained source was not delivered; originals are preserved.",
                        "source_ref": knowledge.required_source, "response_ref": response_ref}]}
                # OpenAI-family lanes report the cut in usage.response_finish_reason;
                # the native Anthropic lane puts stop_reason on the message itself.
                cut_markers = {str(usage.get("response_finish_reason") or "").lower(),
                               str(msg.get("stop_reason") or "").lower()}
                if cut_markers & {"length", "max_tokens"}:
                    # A summary cut at the output ceiling is silent truncation
                    # (BIBLE P1): keep the originals rather than a clipped memory.
                    usages.pop()
                    kind, message, preflight = "output_truncated", "Consolidation output was cut at the output ceiling", False
                    break
                return content, _merge_consolidation_usage(*usages)
            usages.pop()  # the empty response is added once as the failed result below
            kind, message, preflight = "empty_summary", "Consolidation returned no summary", False
            break
    except Exception as error:
        from ouroboros.llm_claudexor import propagate_model_error
        from ouroboros.loop_llm_call import classify_llm_exception
        from ouroboros.transport_custody import outcome_unknown_on_chain
        from ouroboros.usage_accounting import BudgetExceeded
        propagate_model_error(error)
        if getattr(error, "route", None):
            # A refusal belongs to the actual account, which can differ from
            # catalog discovery. Rebind its facts without masking the refusal
            # with a second preflight exception or sending another request.
            prepare({**prepared_values, "_model_observed_route": error.route}, check_fit=False)
        preflight = isinstance(error, SummarizerContextOverflow) or not invoked
        kind = ("budget_exhausted" if isinstance(error, BudgetExceeded)
                else "context_overflow" if isinstance(error, SummarizerContextOverflow)
                else "provider_outcome_unknown" if outcome_unknown_on_chain(error)
                else classify_llm_exception(error).kind)
        message = str(error)
        usage = dict(getattr(error, "usage", None) or {})
        usage.setdefault("cost", None if invoked else 0.0)
        if preflight:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage.setdefault(key, 0)
        usage["ledger_attempt_ids"] = list(getattr(error, "ledger_attempt_ids", []))
    from ouroboros.utils import sanitize_tool_result_for_log

    if knowledge is not None and usages and kind == "context_overflow":
        kind = "knowledge_source_unfit"  # splitting the original episode cannot shrink a requested note
    fact = dict(facts, kind=kind, label=label, message=sanitize_tool_result_for_log(message), preflight_only=preflight)
    log.warning("%s failed (%s): %s", label, kind, fact["message"])
    return "", {**_merge_consolidation_usage(*usages, usage), "_consolidation_errors": [fact]}


def _compress_blocks_to_era(
    blocks: List[Dict[str, Any]],
    llm_client: Any,
    identity_text: str,
    knowledge_context: Any = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Retain the exact source run, then compress it room by room; failure keeps the originals."""
    source_ref = (retain_memory_source(knowledge_context, "chronicle_blocks",
                  json.dumps(blocks, ensure_ascii=False).encode("utf-8"), "json") if knowledge_context else None)
    era, usage = room_consolidation.compress_blocks_to_era(
        _light_call(llm_client, knowledge_context, {}), blocks,
        identity_text if knowledge_context is not None else "")
    if era is None:
        log.warning("Era compression returned empty — keeping original blocks (Bible P1)")
        return None, usage
    return {"ts": utc_now_iso(), "type": "era", **era, **({"source_ref": source_ref} if source_ref else {})}, usage


def _is_gap_block(block: Any) -> bool:
    """The writer's gap ID and the legacy marker still read by Memory."""
    return isinstance(block, dict) and bool(block.get("gap_id") or "[MEMORY GAP]" in str(block.get("content") or ""))


def _is_run_boundary(block: Any) -> bool:
    """Gaps and earlier eras bound a run: an era is built from summary blocks, never from an era."""
    return _is_gap_block(block) or (isinstance(block, dict) and block.get("type") == "era")


ERA_RETRY_MAX_RUNS = 16  # refusals remembered per meta; the oldest run's record ages out first


def _era_retry_runs(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``era_retry`` as {source_sha256: {route, observed_route}}; a legacy single record reads as one entry."""
    retry = meta.get("era_retry")
    if not isinstance(retry, dict):
        return {}
    if "source_sha256" in retry:  # legacy single-record shape
        return {str(retry["source_sha256"]): {key: value for key, value in retry.items() if key != "source_sha256"}}
    return {str(key): dict(value) for key, value in retry.items() if isinstance(value, dict)}


def _era_for_run(run: List[Dict[str, Any]], meta: Dict[str, Any], logs_dir: pathlib.Path, llm_client: Any,
                 identity_text: str, context: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """The era of one run when it is a COMPRESSION; otherwise a recorded, visible refusal.

    Per-room sections and the length-adaptive correction can make an era longer than
    its blocks; keeping the blocks loses nothing. The refusal is ``era_retry`` in meta
    (keyed by source hash, PER RUN — a chronicle holds several runs between gaps and
    eras, and one run's refusal or success must not erase another's; each record keeps the
    dispatch key, read AFTER the call because an owner switch during a wait inside it rebinds
    the role before the paid send, and the route that ANSWERED): the same source is not paid for
    again while the binding a call would dispatch on now is the one that refused, and every
    refusal, attempted or not, is an ``era_not_shorter`` event. Returns ``(era or None, usage or None without a call)``."""
    fact = {"source_sha256": hashlib.sha256(json.dumps(run, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "route": _light_route(), "blocks": len(run), "source_chars": sum(len(b.get("content", "")) for b in run)}
    runs = _era_retry_runs(meta)
    prior = runs.get(fact["source_sha256"])
    if prior is not None and prior.get("route") == fact["route"]:
        _emit_event(logs_dir, "era_not_shorter", attempted=False, **fact)
        return None, None
    era, usage = _compress_blocks_to_era(run, llm_client, identity_text,
                                         **({"knowledge_context": context} if context is not None else {}))
    if era is not None and len(era.get("content", "")) >= fact["source_chars"]:
        # The dispatch key the next attempt compares against, read AFTER the call: a switch inside it
        # rebound the role, so a pre-call key would suppress what never answered and repay what did.
        fact["route"] = _light_route()
        runs.pop(fact["source_sha256"], None)
        runs[fact["source_sha256"]] = {"route": fact["route"], "observed_route": _route_stamp(usage)}
        while len(runs) > ERA_RETRY_MAX_RUNS:
            runs.pop(next(iter(runs)))
        meta["era_retry"] = runs
        _emit_event(logs_dir, "era_not_shorter", attempted=True, era_chars=len(era["content"]),
                    observed_route=_route_stamp(usage), **fact)
        return None, usage
    if era is not None and runs.pop(fact["source_sha256"], None) is not None:
        if runs:
            meta["era_retry"] = runs
        else:
            meta.pop("era_retry", None)
    return era, usage


def _compact_chronicle(blocks_path: pathlib.Path, llm_client: Any,
                       identity_text: str, context: Any, *, meta_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Reduce every contiguous run of summary blocks, preserving gaps, earlier eras and exact sources.

    The pressure pass consults and records the SAME ``era_retry`` as the ordinary
    run (``meta_path``): a run that was not shorter on this route is not paid for
    again by the next pressure pass on unchanged input. Without a meta path (a caller
    that has none) it reads and writes no durable retry metadata: every run is still paid for."""
    blocks = _load_blocks(blocks_path)
    meta: Dict[str, Any] = {}
    if meta_path is not None:
        try:
            meta = _load_meta(meta_path)
        except Exception:
            # An unreadable meta is the caller's typed maintenance gap; the
            # chronicle pass must not write a rebuilt meta over it.
            log.warning("Chronicle pass cannot read dialogue meta; era_retry not consulted", exc_info=True)
            meta_path = None
    retry_before = json.dumps(meta.get("era_retry"), sort_keys=True)
    reduced, usages, start = [], [], 0
    while start < len(blocks):
        if _is_run_boundary(blocks[start]):
            reduced.append(blocks[start])
            start += 1
            continue
        end = start + 1
        while end < len(blocks) and not _is_run_boundary(blocks[end]):
            end += 1
        run = blocks[start:end]
        era, usage = _era_for_run(run, meta, blocks_path.parent.parent / "logs", llm_client, identity_text, context)
        if usage is not None:  # a recorded refusal makes no call and has no usage
            usages.append(usage)
        reduced.extend([era] if era is not None else run)
        if (usage or {}).get("_consolidation_errors"):
            reduced.extend(blocks[end:])
            break
        start = end
    if reduced != blocks:
        _mutate_locked_json_list(blocks_path, lambda live:
            reduced + live[len(blocks):] if live[:len(blocks)] == blocks else live)
    if meta_path is not None and json.dumps(meta.get("era_retry"), sort_keys=True) != retry_before:
        atomic_write_json(meta_path, meta)
    return _merge_consolidation_usage(*usages)


def maintain_memory_pressure(memory: Any, llm_client: Any, context: Any, *,
                             fits: Callable[[], bool], current_topic: str = "") -> Dict[str, Any]:
    """One existing maintenance batch, called only after measured core pressure.

    The caller's callback only rebuilds/measures. It owns the normal send and
    decides whether remaining immutable context fits; this helper has no cadence.
    """
    root = pathlib.Path(memory.drive_root)
    chat, blocks, meta = root / "logs/chat.jsonl", root / "memory/dialogue_blocks.json", root / "memory/dialogue_meta.json"
    shelf = root / "memory/knowledge"
    tracked = [blocks, meta, memory.scratchpad_blocks_path(), memory.scratchpad_path(),
               shelf / "overview.md", shelf / "index-full.md", memory.identity_path()]
    def snapshot() -> Dict[str, Any]:
        result = {}
        for path in tracked:
            raw = path.read_bytes() if path.exists() else None
            result[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                                 "bytes": len(raw) if raw is not None else 0}
        return result
    before, actions, usages = snapshot(), [], []
    def result() -> Dict[str, Any]:
        after = snapshot()
        changed = [{"path": path, "before": before[path], "after": after[path]}
                   for path in before if before[path] != after[path]]
        return {"status": "fitting" if fits() else "progress" if changed else "no_progress",
                "actions": actions, "changed_sources": changed, "usage": _merge_consolidation_usage(*usages)}
    if fits():
        return result()
    identity = "## Current task\n" + current_topic if current_topic else ""
    if memory.identity_path().exists():
        identity_ref = retain_memory_source(context, "maintenance_identity", memory.identity_path().read_bytes())
        identity += "\nExact identity source, available through read_file; no identity rewrite is authorized here:\n" + json.dumps(identity_ref)
    if chat.exists() or blocks.exists():
        from ouroboros.memory_nomination_receipts import DialogueMetaUnreadable

        try:
            usage = consolidate(chat, blocks, meta, llm_client, identity, knowledge_context=context,
                                force_tail=True, compact_chronicle=True, pressure_fits=fits)
        except DialogueMetaUnreadable as exc:
            # A damaged existing cursor is neither empty nor permission to rewrite
            # memory. Keep the original context available to Main, with a typed
            # maintenance gap instead of aborting its first round.
            usage = {"_consolidation_errors": [{"kind": "dialogue_meta_unreadable",
                                                "message": str(exc)}]}
        if usage is not None:
            usages.append(usage)
        actions.append({"owner": "dialogue_consolidation", "usage": usage})
        if fits() or (usage or {}).get("_consolidation_errors"):
            return result()
    if memory.load_scratchpad_blocks():
        usage = consolidate_scratchpad(memory, shelf, llm_client, identity,
                                        pressure=True, knowledge_context=context)
        if usage is not None:
            usages.append(usage)
        actions.append({"owner": "scratchpad_consolidation", "usage": usage})
        if fits() or (usage or {}).get("_consolidation_errors"):
            return result()
    if (shelf / "overview.md").exists() or (shelf / "index-full.md").exists():
        knowledge = KnowledgeReadContext(context, "knowledge_maintenance")
        prompt = KNOWLEDGE_MAINTENANCE_PROMPT + (
            "\nThe shared memory projection exceeds the current task's measured working window. "
            "Read the complete global overview with knowledge_read, then nominate edits making the authored "
            "overview shorter while preserving the whole scope of current understanding and source-relative links to details. "
            "Do not remove useful uncertainty or evidence merely to save space. Use ordinary knowledge notes "
            "for detail when useful. Return JSON: {\"knowledge_entries\": [...]}.\n" + identity)
        if not (shelf / "overview.md").exists():
            prompt += "\nNo authored overview exists. This is the complete legacy inventory/context source, " \
                      "not an authored summary; create an honest overview after reading it:\n" + read_text(shelf / "index-full.md")
        source_ref = retain_memory_source(context, "knowledge_maintenance", prompt.encode("utf-8"))
        raw, usage = _call_consolidation_llm(llm_client, prompt, "Knowledge maintenance", knowledge=knowledge, source_ref=source_ref)
        usages.append(usage)
        action = {"owner": "knowledge_maintenance", "source_ref": source_ref, "usage": usage}
        if raw.strip():
            try:
                entries = knowledge.bind_entries(json.loads(raw).get("knowledge_entries"))
                action["writes"] = _write_knowledge_entries(shelf, entries, context=context, stamp={
                    "writer": "knowledge_maintenance", "route": _route_stamp(usage), "writer_input_ref": source_ref})
            except (ValueError, TypeError, AttributeError) as exc:
                action["error"] = str(exc)
        actions.append(action)
    return result()

def _format_entries_for_block(
    entries: List[Dict[str, Any]], *, include_room_labels: bool = False,
    room_resolver: Any = None, drive_root: Any = None,
    source_spans: Optional[List[Tuple[int, int, str]]] = None,
) -> str:
    from ouroboros.dialogue_provenance import dialogue_author, dialogue_provenance, dialogue_text

    if include_room_labels and room_resolver is None:
        from ouroboros.dialogue_provenance import RoomLabelResolver

        room_resolver = RoomLabelResolver(drive_root)

    lines, offset = [], 0
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
            author = dialogue_author(e)
        if dir_raw in ("out", "outgoing", "system") and e.get("transport"):
            provenance = dialogue_provenance(e)
            if provenance:
                author = f"{author} [{provenance}]"
        text = dialogue_text(e)
        room_prefix = (
            f"[room={room_resolver.label(e)}] "
            if include_room_labels and room_resolver is not None else ""
        )
        header = f"[{ts}] {room_prefix}{direction_prefix}{author}: "
        line = header + text
        if source_spans is not None:
            source_spans.append((offset, offset + len(line), header))
        lines.append(line)
        offset += len(line) + 2  # The original separator belongs to the source.
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
        _emit_event(path.parent.parent / "logs", "memory_store_corrupt", path=str(path), quarantine=str(quarantine))
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
    from ouroboros.memory_nomination_receipts import load_meta

    return load_meta(path)


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
    """Compatibility entrypoint; the common knowledge owner renders every index."""
    from contextlib import nullcontext
    from ouroboros.knowledge import KnowledgeAddress, knowledge_write_lock, rebuild_knowledge_index

    address = KnowledgeAddress(knowledge_dir.parent.parent, knowledge_dir, "overview")
    with nullcontext() if _locked else knowledge_write_lock(knowledge_dir):
        rebuild_knowledge_index(address)


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
    memory: Any, knowledge_dir: pathlib.Path, llm_client: Any, identity_text: str = "",
    *, pressure: bool = False, knowledge_context: Any = None,
) -> Optional[Dict[str, Any]]:
    blocks = memory.load_scratchpad_blocks()
    total_chars = sum(len(b.get("content", "")) for b in blocks)
    if not blocks or not pressure and (len(blocks) < 3 or total_chars <= SCRATCHPAD_CONSOLIDATION_THRESHOLD):
        return None

    compress_count = len(blocks) if pressure else max(2, len(blocks) // 2)
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
   preserving long-term. Output them as knowledge_entries with topic + content
   for a new note. Topics are source-relative Markdown paths; preserve their exact identities.
   For an existing topic, read its complete current source using knowledge_read,
   then propose its anchored edits, not a blind append of the new fragment.
2. Compress the old blocks into a SINGLE shorter summary block. Keep active
   tasks, unresolved questions, admin instructions still in force. Remove
   stale/completed items and routine status updates.
3. Write as Ouroboros (first person). Don't lose signal — keep uncertain items
   rather than dropping them.

Identity context: {identity_text if identity_text else "(not available)"}

## Old blocks to compress

{old_content}

Respond with JSON only (no fences), after any useful knowledge reads:
{{"knowledge_entries": [{{"topic": "topic/path", "scope": "global", "edits": [{{"old_text": "exact passage", "new_text": "revision", "basis": "source and reason"}}]}}], "compressed_block": "single compressed block text"}}
"""

    usage: Dict[str, Any] = {}
    outcome, source_entry_id, writes, new_blocks = "failed", "", [], blocks
    try:
        from ouroboros.tools.registry import ToolContext

        context = knowledge_context or ToolContext(repo_dir=getattr(memory, "repo_dir", None) or memory.drive_root,
                              drive_root=memory.drive_root)
        knowledge = KnowledgeReadContext(context, "scratchpad_consolidation")
        raw, usage = _call_consolidation_llm(
            llm_client, KNOWLEDGE_MAINTENANCE_PROMPT + prompt, "Scratchpad consolidation",
            knowledge=knowledge)
        raw = raw.strip()
        if not raw:
            outcome = "call_failed" if usage.get("_consolidation_errors") else "empty_response"
            return usage
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)

        compressed_text = result.get("compressed_block", "")
        if not compressed_text or not compressed_text.strip():
            log.warning("Scratchpad block consolidation returned empty, skipping")
            outcome = "empty_block"
            return usage
        if pressure and len(compressed_text) >= sum(len(b.get("content", "")) for b in old_blocks):
            outcome = "not_shorter"
            return usage  # an authored expansion is not pressure relief

        entries = knowledge.bind_entries(result.get("knowledge_entries"))
        compressed_block = {"ts": utc_now_iso(), "source": "consolidation", "content": compressed_text.strip()}
        source_bytes = json.dumps(old_blocks, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        source_entry_id = "scratchpad-consolidation:" + hashlib.sha256(source_bytes).hexdigest()
        source_ref = memory.scratchpad_journal_source_ref(source_entry_id)
        if not append_jsonl(memory.journal_path(), {
                "ts": utc_now_iso(), "type": "blocks_consolidated", "entry_id": source_entry_id,
                "source_blocks": old_blocks, "source_ref": source_ref, "knowledge_entries": entries}):
            log.error("Scratchpad consolidation source journal write failed; preserving blocks")
            outcome = "journal_unavailable"
            return usage
        compressed_block["metadata"] = {"source_ref": source_ref}
        writes = _write_knowledge_entries(knowledge_dir, entries, context=context, stamp={
            "writer": "scratchpad_consolidation", "route": _route_stamp(usage), "writer_input_ref": source_ref})
        if writes:
            compressed_block["metadata"]["knowledge_writes"] = writes
            if any(not row["ok"] for row in writes):
                compressed_block["content"] += (
                    "\n\nSome nominated knowledge updates were not published; their complete "
                    "proposals and original episode remain in the source journal referenced by this block.")
                append_jsonl(memory.journal_path(), {"ts": utc_now_iso(), "type": "knowledge_writes_incomplete",
                                                     "source_ref": source_ref, "knowledge_writes": writes})

        # Merge-aware replace UNDER the write lock: blocks appended DURING the
        # slow LLM call live only on disk — building the new list from the
        # pre-call snapshot would silently drop them. Re-read inside the lock
        # and keep every block outside the exact compressed source window.
        def _merge_survivors(live_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if live_blocks[:len(old_blocks)] != old_blocks:
                return live_blocks  # source changed; ts/source keys alone cannot authorize replacement
            return [compressed_block] + live_blocks[len(old_blocks):]

        new_blocks = memory.mutate_scratchpad_blocks(_merge_survivors)
        outcome = "replaced" if new_blocks[:1] == [compressed_block] else "source_changed"

        log.info("Scratchpad blocks consolidated: %d blocks (%d chars) -> %d blocks (%d chars)",
                 len(blocks), total_chars,
                 len(new_blocks), sum(len(b.get("content", "")) for b in new_blocks))
        return usage

    except Exception as e:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(e)
        log.error("Scratchpad block consolidation failed: %s", e, exc_info=True)
        usage = {**usage, "_consolidation_errors": [*usage.get("_consolidation_errors", []), {
            "kind": "scratchpad_consolidation_failed", "message": f"{type(e).__name__}: {e}"}]}
        return usage
    finally:
        # Every exit above names its outcome; the chat_block_consolidation row is the model.
        errors = usage.get("_consolidation_errors") or []
        _emit_event(pathlib.Path(memory.drive_root) / "logs", "scratchpad_consolidation", outcome=outcome,
                    pressure=pressure, blocks_before=len(blocks), chars_before=total_chars,
                    compressed_blocks=len(old_blocks), blocks_after=len(new_blocks),
                    chars_after=sum(len(b.get("content", "")) for b in new_blocks), source_entry_id=source_entry_id,
                    knowledge_writes={"ok": sum(w["ok"] for w in writes), "failed": sum(not w["ok"] for w in writes)},
                    last_error_kind=(errors[-1] or {}).get("kind") if errors else None,
                    accounted_upper_bound_usd=round(float(usage["cost"]), 6) if usage.get("cost") is not None else None)


def _write_knowledge_entries(
    knowledge_dir: pathlib.Path, entries: List[Dict[str, Any]], *, context: Any = None,
    stamp: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Publish only source-aware nominations through the common note writer.

    ``stamp`` is the caller's ``writer``/``route``/``writer_input_ref`` history
    stamp (see ``write_knowledge_note``); an unnamed caller leaves ``unknown``. An
    entry carrying its own ``_nomination_route`` outranks the caller's block-level
    ``route``: provenance is per nomination. That field is HOST-authored only —
    ``KnowledgeReadContext.bind_entries`` strips every underscore key a model
    supplied, and the room seam stamps it after binding from the correction
    call's own usage — so the writer never trusts model output for it. A note this operation
    read (``expected_revision``) takes anchored ``edits`` (+ ``summary``), never whole ``content``;
    an unread topic is create-only (``nomination_write_form`` owns the shape)."""
    from ouroboros.knowledge import KnowledgeAddress, nomination_write_form, sanitize_topic, write_knowledge_note
    from ouroboros.tools.knowledge import _address, _record_backlog_history

    outcomes = []
    for entry in entries:
        if not isinstance(entry, dict):
            outcomes.append({"topic": "", "ok": False, "reason": "malformed_nomination"})
            continue
        topic, content, revision = entry.get("topic"), entry.get("content"), entry.get("expected_revision")
        try:
            form = nomination_write_form(entry)  # every refusal still leaves this entry's one outcome
            topic = sanitize_topic(topic)
            address = (_address(context, topic, str(entry.get("scope") or "")) if context is not None
                       else KnowledgeAddress(knowledge_dir.parent.parent, knowledge_dir, topic))
            if topic == "improvement-backlog":
                from ouroboros.improvement_backlog import backlog_path, merge_backlog_text
                merged = merge_backlog_text(address.canonical_root, content)
                if merged >= 0:
                    _record_backlog_history(backlog_path(address.canonical_root), topic, "overwrite",
                                            str(entry.get("task_id") or ""))
                outcomes.append({"topic": topic, "scope": "global", "ok": merged >= 0,
                                 "reason": "backlog_merge" if merged >= 0 else "unparseable_backlog"})
                continue
            entry_stamp = dict(stamp or {})
            if entry.get("_nomination_route") is not None:
                entry_stamp["route"] = entry["_nomination_route"]
            if form["mode"] == "overwrite" and revision is not None:  # a read existing note: never a whole replacement
                outcomes.append({"topic": topic, "scope": address.scope, "ok": False, "reason": "existing_note_requires_edits"})
                continue
            result = write_knowledge_note(address, expected_revision=revision, task_id=str(entry.get("task_id") or ""),
                                          **form, **entry_stamp)
            outcomes.append({"topic": topic, "scope": address.scope, "ok": result.ok, "reason": result.reason,
                             "source_ref": result.current.source_ref() if result.current else None})
        except (ValueError, OSError) as exc:
            outcomes.append({"topic": topic, "ok": False, "reason": str(exc)})
    return outcomes
