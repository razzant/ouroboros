"""History/cost endpoints extracted from server.py."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from typing import Any, Callable, Dict, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.gateway._helpers import _TAIL_WINDOW_START_BYTES, read_rotated_jsonl_entries
from ouroboros.task_results import TASK_COST_META_FIELDS as _TASK_COST_META_FIELDS
from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

# Hard cap on binding-backed origin rows synthesized per /api/chat/history
# response (realistically 1-3 per project; the cap keeps the endpoint's
# bounded-row contract honest even against pathological binding state).
_ORIGIN_SYNTH_CAP = 10

# Default per-type quotas for the /api/chat/history window (perf2 P3,
# owner-approved 150/60/300). The web client's default request sends NO quota
# params, so these constants ARE the UI's effective first-load window. The
# explicit-quota caps (1500/600) are unchanged: a "Load older" escalation asks
# for a bigger window with explicit n_human/n_progress.
_DEFAULT_N_HUMAN = 150
_DEFAULT_N_PROGRESS = 60
_MAX_N_HUMAN = 1500
_MAX_N_PROGRESS = 600
# Bound subagent lineage so a huge swarm fan-out can't balloon the response.
_LINEAGE_CAP = 300
# Mirror of read_rotated_jsonl_entries' max_archives default: the rotated
# backfill never consults more than this many newest archive segments, so a
# quota the newest segments cannot satisfy is an "archive_floor" truncation.
_ARCHIVE_BACKFILL_CAP = 3

_ACCOUNTING_SUMMARY_FIELDS = (
    "settled_usd",
    "confirmed_usd",
    "estimated_usd",
    "reserved_usd",
    "unresolved_upper_bound_usd",
    "accounted_usd",
    "unknown_unmetered",
    "cost_final",
    # `cost_final`'s DISCLOSED CAUSE travels with the flag it explains — without it
    # the client's "Pending (N open)" text could never render (costs.js reads
    # `accounting.non_final_rows`), so the reason for a non-final cost never
    # reached the owner at all.
    "non_final_rows",
    "attempt_counts",
)

_PROGRESS_META_FIELDS = (
    "ephemeral_decision",
    "subagent_event",
    "subagent_task_id",
    "root_task_id",
    "parent_task_id",
    "delegation_role",
    "subagent_role",
    "accepted",
    "active_subagent_count",
    "max_active_subagents",
    "queued_behind_active_cap",
    "required_capabilities",
    "write_surface",
    "status",
    "cancelable",
    *_TASK_COST_META_FIELDS,
    "result",
    "result_truncated",
    "trace_summary",
    "trace_summary_truncated",
    "error",
    "artifact_status",
    "outcome_axes",
    "reason_code",
    "review_projection",
    "worker_saturation_warning",
    "model_lane",
    "requested_model_lane",
    "effective_model_lane",
    "model",
    # Phase 6: the resolved delegated route (a harness id), so a replayed
    # bubble keeps its executor chip instead of losing it on reload.
    "executor_route",
    # The completion-seam evidence block (delegated runs started/settled,
    # subscription spend, harness models) — the chip's layered truth on replay.
    "execution_evidence",
    "task_group_id",
)


def _compat_cost_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cost": round(float(bucket.get("settled_usd") or 0.0), 6),
        "calls": int(bucket.get("physical_calls") or 0),
        "prompt_tokens": int(bucket.get("prompt_tokens") or 0),
        "completion_tokens": int(bucket.get("completion_tokens") or 0),
        "cached_tokens": int(bucket.get("cached_tokens") or 0),
        "cache_write_tokens": int(bucket.get("cache_write_tokens") or 0),
        "prompt_cache_ttls": dict(bucket.get("prompt_cache_ttls") or {}),
    }


def _compat_cost_groups(
    groups: Dict[str, Dict[str, Any]],
    unattributed: Dict[str, Any],
    *,
    group_key: Optional[Callable[[str], str]] = None,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name, raw_bucket in groups.items():
        if not (
            int(raw_bucket.get("physical_calls") or 0)
            or int(raw_bucket.get("unknown_unmetered") or 0)
            or float(raw_bucket.get("accounted_usd") or 0.0)
        ):
            continue
        key = group_key(str(name)) if group_key else str(name)
        source = _compat_cost_bucket(raw_bucket)
        if key not in result:
            result[key] = source
            continue
        target = result[key]
        for field in (
            "cost", "calls", "prompt_tokens", "completion_tokens",
            "cached_tokens", "cache_write_tokens",
        ):
            target[field] += source[field]
        for ttl, count in source["prompt_cache_ttls"].items():
            target["prompt_cache_ttls"][ttl] = int(target["prompt_cache_ttls"].get(ttl, 0)) + int(count)
    if (
        int(unattributed.get("physical_calls") or 0)
        or int(unattributed.get("unknown_unmetered") or 0)
        or float(unattributed.get("accounted_usd") or 0.0)
    ):
        result["unattributed"] = _compat_cost_bucket(unattributed)
    for bucket in result.values():
        bucket["cost"] = round(float(bucket["cost"]), 6)
    return dict(sorted(result.items(), key=lambda item: item[1]["cost"], reverse=True))


def _project_history_context(
    data_dir: pathlib.Path,
    thread_id: int,
) -> tuple[set[int], Any, Dict[str, Any], Dict[str, int]]:
    """Load the read-only Project history lenses (synchronous).

    Runs inside the endpoint's single ``asyncio.to_thread`` assembly call
    (perf2 P3), so the loads stay off the event loop without per-load thread
    hops. The task->project-chat bindings map is preloaded ONCE per request
    (v6.90.x P2): `_bound_project_chat` previously re-read
    state/project_task_bindings.json for every uncached (task, parent, root)
    lineage key — up to three file reads per history row.

    The per-thread lens is the SHARED ``thread_ancestry_lens`` — the same object
    ``ouroboros/context.py`` builds for the agent — so a forked thread's history
    is identical on both surfaces, and its ancestors' source refs travel with
    it (a fork of a CONVERTED project's thread would otherwise lose the Main
    origin row the parent projects)."""
    try:
        from ouroboros.projects_registry import reserved_project_chat_ids

        project_chat_ids = reserved_project_chat_ids(data_dir)
    except Exception:
        project_chat_ids = set()
    try:
        from ouroboros.thread_history import thread_ancestry_lens

        lens = thread_ancestry_lens(data_dir, thread_id)
    except Exception:
        log.warning("Failed to build the thread ancestry lens", exc_info=True)
        from ouroboros.thread_history import ThreadLens

        # DISCLOSED, never substituted. This fallback used to be an own-thread lens
        # indistinguishable from a genuine non-fork thread, so a fork that had lost
        # its whole ancestry was still reported as a COMPLETE window.
        lens = ThreadLens(
            chat_id=thread_id, cutoffs={thread_id: ""}, order=[thread_id],
            truncated=True, lens_unavailable=True,
        )
    try:
        from ouroboros.project_dialogue import latest_chat_annotations

        annotations = latest_chat_annotations(data_dir)
    except Exception:
        annotations = {}
    try:
        from ouroboros.projects_registry import all_task_bindings

        bindings_by_task = all_task_bindings(data_dir)
    except Exception:
        bindings_by_task = {}
    return project_chat_ids, lens, annotations, bindings_by_task


def _user_annotation(
    role: str,
    client_message_id: str,
    annotations: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    annotation = annotations.get(client_message_id)
    if role != "user" or not isinstance(annotation, dict):
        return None
    return {key: annotation.get(key) for key in ("action", "target", "status")}


def make_cost_breakdown_endpoint(data_dir: pathlib.Path):
    async def api_cost_breakdown(_request: Request) -> JSONResponse:
        """Return ledger-derived cost and physical-attempt breakdowns."""
        try:
            from ouroboros.pricing import infer_model_category
            from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

            ensure_legacy_imported(data_dir)
            breakdown = usage_breakdown(data_dir)
            unattributed = dict(breakdown.get("unattributed") or {})
            by_model_raw = dict(breakdown.get("by_model") or {})
            try:
                from supervisor.state import TOTAL_BUDGET_LIMIT

                limit = float(TOTAL_BUDGET_LIMIT or 0.0)
            except (ImportError, TypeError, ValueError):
                limit = 0.0
            if limit <= 0 and "TOTAL_BUDGET" in os.environ:
                try:
                    limit = max(0.0, float(os.environ.get("TOTAL_BUDGET") or 0.0))
                except (TypeError, ValueError):
                    limit = 0.0
            accounting = {field: breakdown.get(field) for field in _ACCOUNTING_SUMMARY_FIELDS}
            accounting.update({
                "available": True,
                "authority": "physical_attempt_ledger",
                "limit_usd": round(limit, 6),
                "remaining_known_usd": (
                    round(max(0.0, limit - float(breakdown.get("accounted_usd") or 0.0)), 6)
                    if limit > 0
                    else None
                ),
            })
            return JSONResponse({
                # Compatibility fields now project the physical-attempt ledger;
                # events.jsonl is import evidence, never a second cost authority.
                "total_cost": round(float(breakdown.get("settled_usd") or 0.0), 6),
                "total_calls": int(breakdown.get("physical_calls") or 0),
                "total_prompt_tokens": int(breakdown.get("prompt_tokens") or 0),
                "total_completion_tokens": int(breakdown.get("completion_tokens") or 0),
                "total_cached_tokens": int(breakdown.get("cached_tokens") or 0),
                "total_cache_write_tokens": int(breakdown.get("cache_write_tokens") or 0),
                "prompt_cache_ttls": dict(breakdown.get("prompt_cache_ttls") or {}),
                "by_model": _compat_cost_groups(by_model_raw, dict(unattributed.get("model") or {})),
                "by_api_key": _compat_cost_groups(
                    dict(breakdown.get("by_provider") or {}),
                    dict(unattributed.get("provider") or {}),
                ),
                "by_model_category": _compat_cost_groups(
                    by_model_raw,
                    dict(unattributed.get("model") or {}),
                    group_key=infer_model_category,
                ),
                "by_task_category": _compat_cost_groups(
                    dict(breakdown.get("by_category") or {}),
                    dict(unattributed.get("category") or {}),
                ),
                "accounting": accounting,
                "unattributed": unattributed,
            })
        except Exception:
            log.exception("Physical-attempt accounting unavailable")
            return JSONResponse({
                "error": "Physical-attempt accounting unavailable",
                "accounting": {
                    "available": False,
                    "authority": "physical_attempt_ledger",
                    "cost_final": False,
                    "error_code": "ledger_unavailable",
                },
            }, status_code=503)

    return api_cost_breakdown


def _origin_fallback_rows(data_dir, lens: Any, human_tail: list) -> list:
    """Binding-backed origin rows for a Project thread (v6.73.0 lens fallback).

    Synthesizes a start-message row from the binding's own ``source_text`` for
    every cross-thread origin whose canonical row is NOT among the rows actually
    emitted to the client — identity-deduped (client_message_id, else ts), hard-
    capped at ``_ORIGIN_SYNTH_CAP`` with a DISCLOSED omission note naming the
    omitted count and the durable full-copy source (BIBLE P1: no silent cut).

    Ancestors are included with their own cutoffs (X4): a fork of a CONVERTED
    project's thread must still show the Main-chat message that started the
    project, which is exactly what the parent's binding holds. Every ancestor's
    rows come from ONE bucketed bindings read (``origin_rows_by_chat``), deduped
    on the same identity tuple across the whole chain — asking per ancestor
    re-read the bindings file once per link and could synthesize one owner
    message twice."""
    from ouroboros.project_dialogue import origin_rows_by_chat

    order = list(getattr(lens, "order", []) or [])
    cutoffs = getattr(lens, "cutoffs", {}) or {}
    by_chat = origin_rows_by_chat(data_dir, order)
    origin_rows: list = []
    for owner_chat in order:
        cutoff = cutoffs.get(owner_chat, "")
        for row in by_chat.get(owner_chat, ()):
            if cutoff and str((row.get("ref") or {}).get("ts") or "") > cutoff:
                continue
            origin_rows.append(row)
    if not origin_rows:
        return []
    emitted_ids = {
        str(m.get("client_message_id") or "")
        for m in human_tail
        if m.get("role") == "user" and m.get("client_message_id")
    }
    emitted_ts = {str(m.get("ts") or "") for m in human_tail if m.get("role") == "user"}
    synthesized: list = []
    for index, row in enumerate(origin_rows):
        ref = row.get("ref") or {}
        cmid = str(ref.get("client_message_id") or "")
        if (cmid and cmid in emitted_ids) or (
            not cmid and str(ref.get("ts") or "") in emitted_ts
        ):
            continue
        synthesized.append({
            "text": str(row.get("text") or ""),
            "role": "user",
            "ts": str(ref.get("ts") or ""),
            "is_progress": False,
            "system_type": "",
            "markdown": False,
            "source": "",
            "sender_label": "",
            "sender_session_id": "",
            "client_message_id": cmid,
            "task_id": "",
            "telegram_chat_id": 0,
            "origin_projected": True,
        })
        if len(synthesized) >= _ORIGIN_SYNTH_CAP:
            omitted = sum(
                1 for later in origin_rows[index + 1:]
                if str(((later.get("ref") or {}).get("client_message_id")) or "")
                not in emitted_ids
            )
            if omitted:
                synthesized.append({
                    "text": (
                        f"⚠️ OMISSION NOTE: {omitted} more archived project origin "
                        "message(s) not rendered; full copies live in "
                        "state/project_task_bindings.json (source_text)."
                    ),
                    "role": "system", "ts": str(ref.get("ts") or ""),
                    "is_progress": False, "system_type": "origin_omission",
                    "markdown": False, "source": "", "sender_label": "",
                    "sender_session_id": "", "client_message_id": "",
                    "task_id": "", "telegram_chat_id": 0,
                })
            break
    return synthesized


def _read_chat_history_entries(live, adir, want, row_matches_thread):
    """Read a bounded live chat.jsonl tail plus a newest-first archive backfill.

    The live chat.jsonl is rotated to ``archive/chat_<ts>.jsonl`` once it crosses
    ~800KB. Reading only the live file would erase the visible conversation right
    after a rotation (and any file bubble delivered before it). Backfill from the
    most recent archives — newest first, until we have enough human rows to satisfy
    ``want``, bounded to a few files — then reassemble chronologically (oldest
    archive -> live). ``row_matches_thread`` is the endpoint's A2A + chat_id/
    project-thread filter, threaded in so a human row counts toward the backfill
    quota only if it would survive the same filter applied in the render loop —
    otherwise a project-thread request whose live file already holds ``want``
    unrelated main-chat rows would skip the archives and still lose the rotated
    project messages/documents this backfill exists to recover. The live read is
    a window-doubled byte tail (``read_rotated_jsonl_entries``), so the endpoint
    is O(window), not O(whole live file).
    """
    return read_rotated_jsonl_entries(
        live, adir, "chat", want, _chat_quota_predicate(row_matches_thread)
    )


def _chat_quota_predicate(row_matches_thread):
    """Human-row quota predicate shared by the rotated chat read and the
    window-truncation accounting (must equal the render loop's filter)."""

    def _counts_toward_thread(e):
        if not isinstance(e, dict):
            return False
        if str(e.get("direction", "")).lower() not in ("in", "out"):
            return False
        if is_a2a_chat_id(e.get("chat_id", 1)):
            return False
        try:
            ec = int(e.get("chat_id", 1) or 1)
        except (TypeError, ValueError):
            ec = 1
        return row_matches_thread(ec, e)

    return _counts_toward_thread


def _read_progress_history_entries(live, adir, want, counts_toward_quota):
    """Bounded, rotation-aware read of logs/progress.jsonl (mirror of
    ``_read_chat_history_entries``): a window-doubled live byte tail plus a
    newest-first ``archive/progress_*.jsonl`` backfill (capped like chat's 3).

    ``counts_toward_quota`` is the endpoint's filter for rows that satisfy the
    ``n_progress`` slice: thread-matching, non-A2A, non-empty, NON-LINEAGE rows.
    Subagent-lineage rows ride on top of the quota and are safe under this stop
    condition by construction: the lineage recency floor is the ts of the oldest
    retained non-lineage row, which the >=want stop guarantees to be inside the
    returned window, and every lineage row at-or-after it is newer, hence also
    in-window. That guarantee assumes progress rows are appended with
    non-decreasing ts (the writers share one host clock); a backdated
    out-of-order row older than the floor can fall outside the lineage window."""
    return read_rotated_jsonl_entries(live, adir, "progress", want, counts_toward_quota)


def _copy_task_summary_metadata(rec: Dict[str, Any], entry: Dict[str, Any]) -> None:
    """Copy the bounded task-summary fields replayed by the Chat surface."""
    if entry.get("type") != "task_summary":
        return
    for key in ("tool_calls", "rounds"):
        if key in entry:
            rec[key] = int(entry[key])
    rec["outcome_axes"] = normalize_outcome_axes(entry)
    if "reason_code" in entry:
        rec["reason_code"] = str(entry.get("reason_code") or "")
    if isinstance(entry.get("review_projection"), dict):
        rec["review_projection"] = dict(entry.get("review_projection") or {})
    # v6.82 P1: the summary row now carries the flat task-scope cost snapshot
    # written by agent_task_pipeline; replay it so a reload still shows cost.
    # _annotate_terminal_task_truth later OVERRIDES these with the persisted
    # task_results values when the result file survives (row = fallback only).
    for key in _TASK_COST_META_FIELDS:
        if key in entry:
            rec[key] = entry[key]


def _load_terminal_result(
    data_dir: pathlib.Path,
    task_id: str,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Effective task result for history projection, cached per request.

    Status/cost projection only — a history GET must never copy artifacts or
    claim disposition hashes (materialize contract). The cache is shared
    between the pre-floor lineage terminal-truth pass (perf2 P3 variant A) and
    ``_annotate_terminal_task_truth``, so each task_results file is read at
    most once per request."""
    if task_id in cache:
        return cache[task_id]
    try:
        from ouroboros.task_status import load_effective_task_result

        result = load_effective_task_result(data_dir, task_id, materialize_artifacts=False)
    except Exception:
        result = None
    cache[task_id] = result or {}
    return cache[task_id]


def _annotate_terminal_task_truth(
    combined: list[Dict[str, Any]],
    data_dir: pathlib.Path,
    result_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Project bounded terminal truth onto the card rows that survive history replay.

    Runs AFTER quota slicing (v6.90.x P2) on exactly the rows the response emits,
    so the endpoint pays for the task ids of the WINDOW, not of the whole parsed
    history. Intended behavior change: terminal truth (cost/axes/review) lands on
    the latest IN-WINDOW progress row / the in-window summary row — previously it
    was anchored to the globally-latest row and to summary rows that the quota may
    then have evicted, leaving the surviving in-window card without the truth.

    ``result_cache`` (task_id -> effective result) lets the endpoint share the
    task_results reads already performed by the pre-floor lineage pass."""

    try:
        from ouroboros.task_status import FINAL_STATUSES

        cache = result_cache if result_cache is not None else {}
        progress_task_ids = {
            str(message.get("task_id") or "")
            for message in combined
            if message.get("is_progress") and message.get("task_id")
        }
        summary_task_ids = {
            str(message.get("task_id") or "")
            for message in combined
            if str(message.get("system_type") or "") == "task_summary"
            and message.get("task_id")
        }
        terminal_status_by_task: Dict[str, str] = {}
        terminal_truth_by_task: Dict[str, Dict[str, Any]] = {}
        suggested_name_by_task: Dict[str, str] = {}
        for task_id in progress_task_ids | summary_task_ids:
            result = _load_terminal_result(data_dir, task_id, cache)
            status = str(result.get("status") or "")
            if status in FINAL_STATUSES:
                terminal_status_by_task[task_id] = status
                terminal_truth: Dict[str, Any] = {
                    "outcome_axes": normalize_outcome_axes(result),
                }
                if result.get("reason_code"):
                    terminal_truth["reason_code"] = str(result.get("reason_code") or "")
                review_projection = result.get("review_projection")
                if isinstance(review_projection, dict):
                    terminal_truth["review_projection"] = dict(review_projection)
                # v6.82 P1: attach the persisted terminal cost truth. Applied via
                # message.update() below, so it OVERRIDES any row-embedded
                # task_summary snapshot values (the result file is authoritative;
                # the row snapshot is the pruned-result fallback).
                for key in _TASK_COST_META_FIELDS:
                    if key in result:
                        terminal_truth[key] = result[key]
                terminal_truth_by_task[task_id] = terminal_truth
            suggested_name = str(result.get("suggested_name") or "").strip()
            if suggested_name:
                suggested_name_by_task[task_id] = suggested_name

        latest_progress_by_task: Dict[str, Dict[str, Any]] = {}
        for message in combined:
            task_id = str(message.get("task_id") or "")
            if not task_id or not message.get("is_progress"):
                continue
            previous = latest_progress_by_task.get(task_id)
            if previous is None or str(message.get("ts") or "") >= str(previous.get("ts") or ""):
                latest_progress_by_task[task_id] = message

        for message in combined:
            task_id = str(message.get("task_id") or "")
            if not task_id:
                continue
            if message.get("is_progress") and task_id in terminal_status_by_task:
                message["task_terminal_status"] = terminal_status_by_task[task_id]
            is_summary = str(message.get("system_type") or "") == "task_summary"
            if is_summary or (
                task_id not in summary_task_ids
                and latest_progress_by_task.get(task_id) is message
            ):
                message.update(terminal_truth_by_task.get(task_id) or {})
            if (message.get("is_progress") or is_summary) and task_id in suggested_name_by_task:
                message["suggested_name"] = suggested_name_by_task[task_id]
    except Exception as exc:
        log.debug("Failed to annotate terminal task status in history: %s", exc)


def _stream_truncation_cause(
    filtered_rows: int,
    quota: int,
    live_size: int,
    archives_total: int,
) -> Optional[str]:
    """Truncation cause for one log stream's window, or ``None`` when complete.

    ``filtered_rows`` counts the rows returned by the rotated reader that
    satisfy the stream's quota predicate. The reader stops as soon as the
    quota is met, so:

    - more filtered rows than the quota -> the tail slice dropped rows
      ("quota");
    - exactly quota rows -> complete only when the whole stream provably fit
      in the first live byte window with no archives; otherwise older unread
      rows may exist behind the reader's quota stop ("quota");
    - fewer rows than the quota -> the live file was fully parsed and up to
      ``_ARCHIVE_BACKFILL_CAP`` newest archives were consulted, so archive
      segments beyond that cap are the only possible loss ("archive_floor").
    """
    if filtered_rows > quota:
        return "quota"
    if filtered_rows == quota and filtered_rows > 0:
        if archives_total == 0 and live_size <= _TAIL_WINDOW_START_BYTES:
            return None
        return "quota"
    if archives_total > _ARCHIVE_BACKFILL_CAP:
        return "archive_floor"
    return None


def _live_log_size(path: pathlib.Path) -> int:
    try:
        return pathlib.Path(path).stat().st_size
    except OSError:
        return 0


def _archive_segment_count(archive_dir: pathlib.Path, prefix: str) -> int:
    try:
        return sum(1 for _ in pathlib.Path(archive_dir).glob(f"{prefix}_*.jsonl"))
    except Exception:
        return 0


def _make_thread_filter(
    thread_id: int,
    project_chat_ids: set,
    lens: Any,
    bindings_by_task: Dict[str, int],
):
    """Build the per-request thread-filter closure (perf2 P3 decomposition).

    Returns the ``row_matches_thread`` predicate shared by both stream readers
    and both transform loops. ``lens`` is the shared thread-ancestry lens: for
    an ordinary thread it admits exactly its own chat, and for a FORK it also
    admits each ancestor chat up to that ancestor's effective (intersected,
    inclusive) cutoff.

    The project-thread half of the predicate is the SHARED
    ``thread_history.admits_row`` / ``bound_chat_for_row`` pair that
    ``ouroboros/context.py`` uses for the agent's focused view — "does this row
    belong to the thread" has ONE implementation, not one per surface. The
    lineage binding walk (own -> parent -> root, so a subagent's rows classify
    into its ROOT's project thread) lives there too, served from the ONE
    bindings map preloaded per request."""
    from ouroboros.thread_history import admits_row, bound_chat_for_row

    def _row_matches_thread(entry_chat: int, entry: Optional[dict] = None) -> bool:
        # A post-hoc bound task keeps its original (main) chat_id on its rows
        # but belongs to a project — classify by the durable LINEAGE binding too.
        bound_chat = bound_chat_for_row(entry, bindings_by_task)
        if thread_id in project_chat_ids:
            if isinstance(entry, dict):
                return admits_row(lens, entry, bound_chat)
            return lens.admits(entry_chat, "")
        # Main / non-project view: everything that is NOT another project. A
        # bound task's rows are project-owned, so mirror only its sanitized
        # progress/task_summary and exclude its raw chat (same as a native
        # project row), never leak raw project chat into the штаб.
        if entry_chat in project_chat_ids or bound_chat > 0:
            if not isinstance(entry, dict):
                return False
            return bool(entry.get("is_progress")) or str(entry.get("type") or "") == "task_summary"
        return entry_chat not in project_chat_ids

    return _row_matches_thread


def _collect_chat_rows(
    chat_path: pathlib.Path,
    archive_dir: pathlib.Path,
    n_human: int,
    row_matches_thread,
    chat_annotations: Dict[str, Any],
) -> tuple[list, int]:
    """Read + transform the chat stream.

    Returns ``(rows, quota_row_count)`` — the transformed history records and
    how many read entries satisfied the reader's quota predicate (feeds the
    window-truncation metadata)."""
    combined: list = []
    chat_quota_rows = 0
    _chat_counts_toward_quota = _chat_quota_predicate(row_matches_thread)
    try:
        # Rotation-aware archive backfill lives in the module-level
        # _read_chat_history_entries helper (endpoint's thread filter threaded in).
        _chat_entries = _read_chat_history_entries(
            chat_path, archive_dir, n_human, row_matches_thread
        )
        # Window accounting for the response's truncation metadata: how many
        # read rows satisfy the SAME quota predicate the reader stopped on.
        chat_quota_rows = sum(
            1 for entry in _chat_entries if _chat_counts_toward_quota(entry)
        )
        for entry in _chat_entries:
            # Skip A2A virtual chat_ids so A2A task traffic does not appear in human chat history.
            if is_a2a_chat_id(entry.get("chat_id", 1)):
                continue
            try:
                entry_chat = int(entry.get("chat_id", 1) or 1)
            except (TypeError, ValueError):
                entry_chat = 1
            if not row_matches_thread(entry_chat, entry):
                continue
            direction = str(entry.get("direction", "")).lower()
            role = {"in": "user", "out": "assistant", "system": "system"}.get(direction)
            if role is None:
                continue
            rec = {
                "text": str(entry.get("text", "")),
                "role": role,
                "ts": str(entry.get("ts", "")),
                "is_progress": False,
                "system_type": str(entry.get("type", "")),
                "markdown": str(entry.get("format", "")).lower() == "markdown",
                "source": str(entry.get("source", "")),
                "sender_label": str(entry.get("sender_label", "")),
                "sender_session_id": str(entry.get("sender_session_id", "")),
                "client_message_id": str(entry.get("client_message_id", "")),
                "task_id": str(entry.get("task_id", "")),
                "telegram_chat_id": int(entry.get("telegram_chat_id") or 0),
            }
            annotation = _user_annotation(role, rec["client_message_id"], chat_annotations)
            if annotation is not None:
                rec["chat_annotation"] = annotation
            # Delivered document rows carry lightweight media metadata (no
            # base64); surface a msg_type + download_url so the frontend
            # rebuilds the file bubble on reload instead of a bare text line.
            if entry.get("type") == "document":
                rec["msg_type"] = "document"
                rec["filename"] = str(entry.get("filename") or "file")
                rec["mime"] = str(entry.get("mime") or "application/octet-stream")
                rec["download_url"] = str(entry.get("download_url") or "")
                rec["caption"] = str(entry.get("caption") or "")
            _copy_task_summary_metadata(rec, entry)
            combined.append(rec)
    except Exception as exc:
        log.warning("Failed to read chat history: %s", exc)
    return combined, chat_quota_rows


def _collect_progress_rows(
    progress_path: pathlib.Path,
    archive_dir: pathlib.Path,
    n_progress: int,
    row_matches_thread,
) -> tuple[list, int]:
    """Read + transform the progress stream.

    Returns ``(rows, quota_row_count)`` (mirror of ``_collect_chat_rows``)."""

    def _progress_counts_toward_quota(entry) -> bool:
        # A row satisfies the n_progress quota only if it survives the render
        # filter below (A2A + thread + non-empty text) AND is NOT subagent
        # lineage — lineage rows ride on top of the quota, so counting them
        # here would let a swarm's lifecycle burst stop the tail read before
        # the window holds n_progress ordinary telemetry rows.
        if not isinstance(entry, dict):
            return False
        if is_a2a_chat_id(entry.get("chat_id", 1)):
            return False
        try:
            entry_chat = int(entry.get("chat_id", 1) or 1)
        except (TypeError, ValueError):
            entry_chat = 1
        if not row_matches_thread(entry_chat, {"is_progress": True, **entry}):
            return False
        if not str(entry.get("content", entry.get("text", ""))):
            return False
        if str(entry.get("delegation_role") or "").lower() == "subagent" or entry.get("subagent_event"):
            return False
        return True

    combined: list = []
    progress_quota_rows = 0
    try:
        _progress_entries = _read_progress_history_entries(
            progress_path,
            archive_dir,
            n_progress,
            _progress_counts_toward_quota,
        )
        progress_quota_rows = sum(
            1 for entry in _progress_entries if _progress_counts_toward_quota(entry)
        )
        for entry in _progress_entries:
            # Skip A2A virtual chat_ids.
            if is_a2a_chat_id(entry.get("chat_id", 1)):
                continue
            try:
                entry_chat = int(entry.get("chat_id", 1) or 1)
            except (TypeError, ValueError):
                entry_chat = 1
            if not row_matches_thread(entry_chat, {"is_progress": True, **entry}):
                continue
            text = str(entry.get("content", entry.get("text", "")))
            if not text:
                continue
            rec = {
                "text": text,
                "role": "assistant",
                "ts": str(entry.get("ts", "")),
                "is_progress": True,
                "markdown": str(entry.get("format", "")).lower() == "markdown",
                "task_id": str(entry.get("task_id", "")),
            }
            if isinstance(entry.get("lifecycle"), dict):
                rec["lifecycle"] = dict(entry.get("lifecycle") or {})
            for field in _PROGRESS_META_FIELDS:
                if field in entry:
                    rec[field] = entry[field]
            combined.append(rec)
    except Exception as exc:
        log.warning("Failed to read progress log: %s", exc)
    return combined, progress_quota_rows


def _active_lifecycle_row() -> Optional[Dict[str, Any]]:
    """Synthesize the virtual progress row for an in-flight skill-lifecycle
    operation (or ``None`` when nothing is running)."""
    try:
        from ouroboros.skill_lifecycle_queue import queue_snapshot

        active = queue_snapshot().get("active")
        if isinstance(active, dict) and active.get("status") == "running":
            label = "stale" if active.get("stale") else "running"
            detail = active.get("error") or active.get("message") or active.get("status") or ""
            text = (
                f"Skill {active.get('kind') or 'operation'}: `{active.get('target') or 'skill'}`"
                f" — {label}{f' — {detail}' if detail else ''}"
            )
            lifecycle = dict(active)
            lifecycle["phase"] = label
            return {
                "text": text,
                "role": "assistant",
                "ts": utc_now_iso(),
                "is_progress": True,
                "markdown": False,
                "task_id": str(active.get("chat_task_id") or ""),
                "lifecycle": lifecycle,
                "lifecycle_virtual": True,
            }
    except Exception as exc:
        log.debug("Failed to synthesize active lifecycle history: %s", exc)
    return None


def _apply_window_quotas(
    data_dir: pathlib.Path,
    thread_id: int,
    project_chat_ids: set,
    lens: Any,
    combined: list,
    n_human: int,
    n_progress: int,
) -> tuple[list, Dict[str, Dict[str, Any]], bool, bool]:
    """Quota slicing, origin fallback, and the lineage floor/cap (perf2 P3).

    Returns ``(messages, result_cache, human_rows_dropped, lineage_truncated)``
    for the annotation pass and the window metadata.
    """
    # Tail human conversation and progress telemetry with SEPARATE quotas so a
    # burst of progress messages can never push the user's real conversation out
    # (the previous single combined[-limit:] tail). Subagent lineage is kept on
    # top of the progress quota so a flood can't evict a RECENT child's lifecycle
    # events (the client rebuilds child-card lineage from them) — but only WITHIN
    # the recent telemetry window: resurrecting an old finished swarm's child
    # events would recreate an orphaned "Working" parent card whose own terminal
    # row has already aged out of the window.
    def _is_subagent_lineage(m: dict) -> bool:
        # Only true SUBAGENT lifecycle (delegation_role 'subagent' or any
        # subagent_event) is lineage-critical. delegation_role can also be
        # 'root', which must NOT bypass the progress quota.
        return str(m.get("delegation_role") or "").lower() == "subagent" or bool(m.get("subagent_event"))

    # NOTE: guard 0 explicitly — Python's list[-0:] is list[0:] (the WHOLE list),
    # so a `[-quota:]` slice with quota==0 would leak everything, not nothing.
    human = sorted((m for m in combined if not m.get("is_progress")), key=lambda m: m.get("ts", ""))
    progress = sorted((m for m in combined if m.get("is_progress")), key=lambda m: m.get("ts", ""))
    human_tail = human[-n_human:] if n_human > 0 else []
    # MAJOR review fix: the n_human slice also drops direction:"system" rows
    # (e.g. the per-task task_summary), which the reader's in/out quota
    # predicate does NOT count — so the stream cause alone could report a
    # "complete" window while the slice silently cut system rows. Any actual
    # drop by this slice is a "quota" truncation, independent of direction.
    human_rows_dropped = len(human) > len(human_tail)
    # v6.73.0 retention-proof origin projection: a Project's start message is
    # synthesized from the binding's own source_text when its canonical row is
    # not among the rows ACTUALLY EMITTED (rotated past the archive window OR
    # pruned by the n_human tail). Post-quota, identity-deduped, hard-capped
    # with a disclosed omission note (helper below the endpoint factory).
    if thread_id in project_chat_ids and n_human > 0:
        try:
            synthesized = _origin_fallback_rows(data_dir, lens, list(human_tail))
            if synthesized:
                human_tail = sorted(
                    human_tail + synthesized, key=lambda m: m.get("ts", "")
                )
        except Exception:
            log.debug("Project origin fallback synthesis failed", exc_info=True)
    other = [m for m in progress if not _is_subagent_lineage(m)]
    other_tail = other[-n_progress:] if n_progress > 0 else []
    # Recency floor = oldest retained telemetry row. Drop lineage older than it so
    # long-finished swarms don't re-materialise as stuck "Working" parent cards.
    floor = str(other_tail[0].get("ts") or "") if other_tail else ""
    lineage_rows = [m for m in progress if _is_subagent_lineage(m)]
    # perf2 P3 variant A (owner decision 2026-08-09): a QUIET but still-ACTIVE
    # child must survive the recency floor — its card is reproducible only from
    # these lineage rows. Terminal truth for the lineage task ids of the READ
    # window is resolved BEFORE the floor/cap slice; the same cache then feeds
    # _annotate_terminal_task_truth after the slice, so each task_results file
    # is read at most once per request. The floor keeps dropping rows of
    # terminal/unknown children (anti-zombie preserved), and the effective-
    # status orphan guard resolves a long-dead raw "running" child as failed,
    # i.e. terminal, so it cannot pin its lineage forever.
    result_cache: Dict[str, Dict[str, Any]] = {}
    active_children: set = set()
    if floor and lineage_rows:
        try:
            from ouroboros.task_status import FINAL_STATUSES

            for task_id in {str(m.get("task_id") or "") for m in lineage_rows if m.get("task_id")}:
                status = str(
                    _load_terminal_result(data_dir, task_id, result_cache).get("status") or ""
                )
                if status and status not in FINAL_STATUSES:
                    active_children.add(task_id)
        except Exception as exc:
            log.debug("Failed to resolve pre-floor lineage terminal truth: %s", exc)
    lineage = [
        m for m in lineage_rows
        if not floor
        or str(m.get("ts") or "") >= floor
        or str(m.get("task_id") or "") in active_children
    ]
    # The cap stays a suffix slice applied AFTER the floor: an active child with
    # more rows than the cap keeps its newest rows, so its card stays alive.
    lineage_truncated = len(lineage) > _LINEAGE_CAP
    if lineage_truncated:
        lineage = lineage[-_LINEAGE_CAP:]  # keep the most recent lineage events
    progress_tail = lineage + other_tail
    messages = sorted(human_tail + progress_tail, key=lambda m: m.get("ts", ""))
    return messages, result_cache, human_rows_dropped, lineage_truncated


def _window_metadata(
    chat_quota_rows: int,
    progress_quota_rows: int,
    n_human: int,
    n_progress: int,
    chat_path: pathlib.Path,
    progress_path: pathlib.Path,
    archive_dir: pathlib.Path,
    human_rows_dropped: bool,
    lineage_truncated: bool,
    lens: Any = None,
) -> Dict[str, Any]:
    """Additive window metadata (perf2 P3; frozen contract extended explicitly).

    The reader learns WHETHER this window is the complete reachable history
    and WHAT bounded it — the quota tail slice ("quota"), the bounded archive
    backfill ("archive_floor"), the lineage cap ("lineage_cap"), or a bounded
    fork ANCESTRY ("ancestry_depth"). The client gates its "Load older"
    affordance on this instead of guessing; no existing field changes meaning.

    ``ancestry_depth`` is the thread-ancestry lens's own ``truncated`` flag:
    the chain hit ``MAX_ANCESTRY_DEPTH``, closed in a cycle, or named an
    ancestor with no project binding, so part of the shared past this thread
    claims was NOT read. The lens set that flag from the start; not threading
    it here meant the shared past could be cut while the response still called
    itself complete — a silent gap ARCHITECTURE already promised was
    disclosed.

    ``lens_unavailable`` is the narrower, worse case: the lens could not be BUILT
    at all (the registry was unreadable), so whether this thread even HAS ancestors
    is unknown. It rides its own cause rather than only ``ancestry_depth`` because
    the two need different sentences — one says part of a known past was not read,
    the other says the past could not be looked up. Both are also ``truncated``, so
    a consumer that only knows the older cause still stops calling the window
    complete."""
    truncated_by: list[str] = []
    for cause in (
        "quota" if human_rows_dropped else None,
        _stream_truncation_cause(
            chat_quota_rows, n_human, _live_log_size(chat_path),
            _archive_segment_count(archive_dir, "chat"),
        ),
        _stream_truncation_cause(
            progress_quota_rows, n_progress, _live_log_size(progress_path),
            _archive_segment_count(archive_dir, "progress"),
        ),
        "lineage_cap" if lineage_truncated else None,
        "ancestry_depth" if bool(getattr(lens, "truncated", False)) else None,
        "lens_unavailable" if bool(getattr(lens, "lens_unavailable", False)) else None,
    ):
        if cause and cause not in truncated_by:
            truncated_by.append(cause)
    return {"complete": not truncated_by, "truncated_by": truncated_by}


def _assemble_history_response(
    data_dir: pathlib.Path,
    thread_id: int,
    n_human: int,
    n_progress: int,
) -> bytes:
    """Assemble the complete /api/chat/history payload as serialized JSON bytes.

    perf2 P3: the WHOLE pipeline — project-context loads, both rotation-aware
    log reads, both transform loops, quota slicing, the lineage floor/cap,
    terminal-truth annotation, origin fallback, and the JSON encode — runs
    synchronously inside the endpoint's single ``asyncio.to_thread`` call, so
    none of it executes on the event loop. (Decomposed into the private
    single-purpose helpers above; behavior is identical.)
    """
    project_chat_ids, lens, chat_annotations, bindings_by_task = (
        _project_history_context(data_dir, thread_id)
    )
    row_matches_thread = _make_thread_filter(
        thread_id, project_chat_ids, lens, bindings_by_task
    )
    chat_path = data_dir / "logs" / "chat.jsonl"
    progress_path = data_dir / "logs" / "progress.jsonl"
    archive_dir = data_dir / "archive"
    combined, chat_quota_rows = _collect_chat_rows(
        chat_path, archive_dir, n_human, row_matches_thread, chat_annotations
    )
    progress_rows, progress_quota_rows = _collect_progress_rows(
        progress_path, archive_dir, n_progress, row_matches_thread
    )
    combined.extend(progress_rows)
    lifecycle_row = _active_lifecycle_row()
    if lifecycle_row is not None:
        combined.append(lifecycle_row)
    messages, result_cache, human_rows_dropped, lineage_truncated = (
        _apply_window_quotas(
            data_dir, thread_id, project_chat_ids, lens, combined, n_human, n_progress
        )
    )

    # Annotate progress messages whose task already reached a terminal (or
    # cancel-intent) status on disk. Tasks torn down by crash storm, hard
    # timeout, or cancellation emit a live task_done but never write a
    # task_summary, so on reload/reconnect the client would otherwise replay
    # their progress and re-inflate a "Working" spinner that never resolves.
    # Runs AFTER the quota slice — on the rows actually emitted — so the
    # response pays only for in-window task ids and the truth always lands on
    # a row the client will see (see _annotate_terminal_task_truth).
    _annotate_terminal_task_truth(messages, data_dir, result_cache=result_cache)

    # Background consciousness writes no task_result, so its progress would
    # otherwise replay as a perpetual "thinking" card after reload. Mark its
    # most recent IN-WINDOW progress entry terminal; a fresh live event
    # re-activates the card if a new cycle starts. (Structured signal,
    # consumed by log_events.js.)
    try:
        bg_msgs = [
            m for m in messages
            if m.get("is_progress") and str(m.get("task_id") or "") == "bg-consciousness"
        ]
        if bg_msgs:
            latest = max(bg_msgs, key=lambda m: str(m.get("ts") or ""))
            latest["task_terminal_status"] = "done"
    except Exception as exc:
        log.debug("Failed to annotate bg-consciousness terminal status: %s", exc)

    payload = {
        "messages": messages,
        "window": _window_metadata(
            chat_quota_rows, progress_quota_rows, n_human, n_progress,
            chat_path, progress_path, archive_dir,
            human_rows_dropped, lineage_truncated, lens,
        ),
    }
    # Same rendering options as starlette's JSONResponse — serialized here so
    # the encode of a large payload also happens off the event loop.
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def make_chat_history_endpoint(data_dir: pathlib.Path):
    async def api_chat_history(request: Request) -> Response:
        """Return recent chat, system, and progress messages merged chronologically."""
        def _int_param(name: str, default: int, cap: int) -> int:
            try:
                return max(0, min(int(request.query_params.get(name, default)), cap))
            except (ValueError, TypeError):
                return default

        # Separate per-type quotas so a burst of progress/telemetry can never evict
        # the user's real conversation from a single combined tail. Defaults are
        # the module-level window constants — the web client's default request
        # sends no quota params. (`limit` is still accepted for backward-compat
        # but no longer governs the slice.)
        n_human = _int_param("n_human", _DEFAULT_N_HUMAN, _MAX_N_HUMAN)
        n_progress = _int_param("n_progress", _DEFAULT_N_PROGRESS, _MAX_N_PROGRESS)
        # Multi-project thread filter (v6.32.0): each chat fetches its own
        # history. Default 1 = main chat (legacy rows without chat_id are main).
        # The filter only PARTITIONS when the requested thread is a registered
        # project chat; for the main chat (and any non-project chat_id, e.g. an
        # external-transport mirror) it keeps the historic behavior of showing
        # every non-project, non-A2A row so transport conversations stay visible.
        thread_id = _int_param("chat_id", 1, 2**31 - 1) or 1
        # ONE thread hop for the whole assembly (perf2 P3): reads, transforms,
        # slicing, annotation, and the JSON encode all run off the event loop.
        body = await asyncio.to_thread(
            _assemble_history_response, data_dir, thread_id, n_human, n_progress
        )
        return Response(content=body, media_type="application/json")

    return api_chat_history
