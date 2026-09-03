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

from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, is_a2a_chat_id
from ouroboros.gateway._helpers import (
    _TAIL_WINDOW_START_BYTES,
    coerce_int,
    read_rotated_jsonl_entries,
)
from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
from ouroboros.subagent_messages import SUBAGENT_MESSAGE_FIELDS, subagent_message_meta
from ouroboros.task_results import TASK_COST_META_FIELDS as _TASK_COST_META_FIELDS
from ouroboros.utils import strip_markdown, utc_now_iso

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
    # The substrate FACT derived from that evidence
    # (harness_used/harness_attempted/native_only).
    "actual_substrate",
    "task_group_id",
    # A duplicate lifecycle call is a typed pointer/ack, not a task. Preserve
    # the pointer on reload while its outer task_id stays empty.
    "lifecycle_pointer",
)

_SKILL_REVIEW_STRING_FIELDS = (
    "skill", "status", "content_hash", "job_id", "group_id",
    "presentation_owner_task_id", "origin_task_id", "origin_root_task_id",
    "root_task_id", "source", "job_status", "terminal_reason",
    "replayed_from_ts",
)
_SKILL_REVIEW_INT_FIELDS = ("review_round", "snapshot_attempt")
_SKILL_REVIEW_BOOL_FIELDS = ("snapshot_revised",)


def _review_executions(value: Any) -> list[Dict[str, str]]:
    from ouroboros.review_execution_projection import normalize_review_executions

    return normalize_review_executions(value)


def _stored_chat_id(value: Any, default: int = 1) -> int:
    """Coerce a stored route while preserving explicit panel chat zero."""
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _compat_cost_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cost": round(float(bucket.get("settled_usd") or 0.0), 6),
        "calls": int(bucket.get("physical_calls") or 0),
        # Keep the compatibility tables honest about rows whose settled dollar
        # amount is zero but whose accounting is still open or undisclosed.
        "unknown_unmetered": int(bucket.get("unknown_unmetered") or 0),
        "non_final_rows": int(bucket.get("non_final_rows") or 0),
        "cost_final": bool(bucket.get("cost_final")),
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
            "cost", "calls", "unknown_unmetered", "non_final_rows",
            "prompt_tokens", "completion_tokens",
            "cached_tokens", "cache_write_tokens",
        ):
            target[field] += source[field]
        target["cost_final"] = target["cost_final"] and source["cost_final"]
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
) -> tuple[set[int], list[dict], Dict[str, Any], Dict[str, int]]:
    """Load the read-only Project history lenses (synchronous).

    Runs inside the endpoint's single ``asyncio.to_thread`` assembly call
    (perf2 P3), so the loads stay off the event loop without per-load thread
    hops. The task->project-chat bindings map is preloaded ONCE per request
    (v6.90.x P2): `_bound_project_chat` previously re-read
    state/project_task_bindings.json for every uncached (task, parent, root)
    lineage key — up to three file reads per history row."""
    try:
        from ouroboros.projects_registry import reserved_project_chat_ids

        project_chat_ids = reserved_project_chat_ids(data_dir)
    except Exception:
        project_chat_ids = set()
    source_refs: list[dict] = []
    if thread_id in project_chat_ids:
        try:
            from ouroboros.project_dialogue import source_refs_for_project

            source_refs = source_refs_for_project(data_dir, thread_id)
        except Exception:
            log.debug("Failed to load canonical Project source refs", exc_info=True)
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
    return project_chat_ids, source_refs, annotations, bindings_by_task


def _matches_project_source(entry: Dict[str, Any], source_refs: list[dict]) -> bool:
    if not source_refs:
        return False
    try:
        from ouroboros.project_dialogue import entry_matches_source_ref

        return entry_matches_source_ref(entry, source_refs)
    except Exception:
        log.debug("Project source-ref classification failed", exc_info=True)
        return False


def _user_annotation(
    role: str,
    client_message_id: str,
    annotations: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    annotation = annotations.get(client_message_id)
    if role != "user" or not isinstance(annotation, dict):
        return None
    return {
        key: annotation.get(key)
        for key in (
            "action", "target", "target_label", "status", "detail", "options",
            "attachment_manifest", "routing_token",
        )
        if key in annotation
    }


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


def _origin_fallback_rows(data_dir, thread_id: int, human_tail: list) -> list:
    """Binding-backed origin rows for a Project thread (v6.73.0 lens fallback).

    Synthesizes a start-message row from the binding's own ``source_text`` for
    every cross-thread origin whose canonical row is NOT among the rows actually
    emitted to the client — identity-deduped (client_message_id, else ts), hard-
    capped at ``_ORIGIN_SYNTH_CAP`` with a DISCLOSED omission note naming the
    omitted count and the durable full-copy source (BIBLE P1: no silent cut)."""
    from ouroboros.project_dialogue import project_origin_rows

    origin_rows = project_origin_rows(data_dir, thread_id)
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


def _read_chat_history_entries(live, adir, want, row_matches_thread, *, include_gaps=False):
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
        live,
        adir,
        "chat",
        want,
        _chat_quota_predicate(row_matches_thread),
        include_gaps=include_gaps,
    )


def _chat_quota_predicate(row_matches_thread):
    """Human-row quota predicate shared by the rotated chat read and the
    window-truncation accounting (must equal the render loop's filter)."""

    def _counts_toward_thread(e):
        if not isinstance(e, dict):
            return False
        if str(e.get("direction", "")).lower() not in ("in", "out"):
            return False
        if e.get("type") == "routing_options":
            # LLM-grounding rows (#198) are skipped by the render loop, so
            # they must not consume the human-row quota either — a run of
            # picker refusals would silently evict real messages.
            return False
        if is_a2a_chat_id(e.get("chat_id", 1)):
            return False
        ec = _stored_chat_id(e.get("chat_id"), 1)
        return row_matches_thread(ec, e)

    return _counts_toward_thread


def _read_progress_history_entries(live, adir, want, counts_toward_quota, *, include_gaps=False):
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
    return read_rotated_jsonl_entries(
        live, adir, "progress", want, counts_toward_quota, include_gaps=include_gaps
    )


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

    Status/cost and compact child-identity projection only — a history GET must never copy artifacts or
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
    floor: str = "",
    anchored_children: Optional[set] = None,
) -> None:
    """Project bounded terminal truth and legacy child identity onto replay rows.

    Runs AFTER quota slicing (v6.90.x P2) on exactly the rows the response emits,
    so the endpoint pays for the task ids of the WINDOW, not of the whole parsed
    history. Intended behavior change: terminal truth (cost/axes/review) lands on
    the latest IN-WINDOW progress row / the in-window summary row — previously it
    was anchored to the globally-latest row and to summary rows that the quota may
    then have evicted, leaving the surviving in-window card without the truth.

    ``result_cache`` (task_id -> effective result) lets the endpoint share the
    task_results reads already performed by the pre-floor lineage pass.

    ``floor``/``anchored_children`` (already computed by ``_apply_window_quotas``)
    extend the progress-stream lineage anchor to chat FINALS: a non-progress
    subagent row older than the floor whose child is NOT anchored (see #496 —
    the child is alive, or its parent is represented by this response, or the
    parent is alive, transitively) loses its lineage identity (raw fields
    stripped, legacy injection undone), so an ABSENT swarm's final cannot
    re-mint an orphaned "Working" parent card on reload. Uses ONLY the floor and
    the pre-computed set — zero extra reads."""

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
        legacy_final_task_ids = {
            str(message.get("task_id") or "")
            for message in combined
            if message.get("task_id")
            and not message.get("is_progress")
            and str(message.get("role") or "") in {"assistant", "system"}
            and str(message.get("task_id") or "") not in progress_task_ids
        }
        terminal_status_by_task: Dict[str, str] = {}
        terminal_truth_by_task: Dict[str, Dict[str, Any]] = {}
        terminal_receipt_by_task: Dict[str, Dict[str, Any]] = {}
        legacy_child_meta_by_task: Dict[str, Dict[str, Any]] = {}
        suggested_name_by_task: Dict[str, str] = {}
        finalizing_tasks: set = set()
        for task_id in progress_task_ids | summary_task_ids | legacy_final_task_ids:
            result = _load_terminal_result(data_dir, task_id, cache)
            child_meta = subagent_message_meta(result, task_id=task_id)
            if child_meta:
                legacy_child_meta_by_task[task_id] = child_meta
            status = str(result.get("status") or "")
            checkpoint = result.get("root_phase_checkpoint")
            synthesis = (
                str(checkpoint.get("post_task_synthesis") or "")
                if isinstance(checkpoint, dict) else ""
            )
            # An OPEN post-task checkpoint means the final answer is stored
            # but synthesis (and the settled task_done) has not landed: the
            # task is FINALIZING, not terminal. This covers the plain root
            # (status already "completed") AND the split-drive project root,
            # whose canonical status stays scheduled/running until copy-back.
            # A failed/cancelled record stays terminal immediately, and a
            # record without a checkpoint keeps the legacy terminal semantics.
            checkpoint_open = post_task_synthesis_is_open(synthesis)
            if checkpoint_open and (status == "completed" or status not in FINAL_STATUSES):
                finalizing_tasks.add(task_id)
            elif status in FINAL_STATUSES:
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
                envelope = result.get("subagent_envelope")
                evidence = (
                    envelope.get("execution_evidence")
                    if isinstance(envelope, dict)
                    else None
                )
                if isinstance(evidence, dict) and evidence:
                    receipt: Dict[str, Any] = {
                        "execution_evidence": dict(evidence),
                    }
                    actual_substrate = str(
                        envelope.get("actual_substrate")
                        or result.get("actual_substrate")
                        or ""
                    ).strip()
                    if actual_substrate:
                        receipt["actual_substrate"] = actual_substrate
                    terminal_receipt_by_task[task_id] = receipt
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

        anchored = anchored_children or set()
        for message in combined:
            task_id = str(message.get("task_id") or "")
            if not task_id:
                continue
            if str(message.get("system_type") or "") != "skill_review":
                for key, value in legacy_child_meta_by_task.get(task_id, {}).items():
                    message.setdefault(key, value)
            # Every row of a finalizing task carries the typed phase so replay
            # (progress cards AND the early final answer row) holds the card
            # on "Finalizing…" instead of resolving it as done.
            if task_id in finalizing_tasks:
                message["task_phase"] = "finalizing"
            if message.get("is_progress") and task_id in terminal_status_by_task:
                message["task_terminal_status"] = terminal_status_by_task[task_id]
                if latest_progress_by_task.get(task_id) is message:
                    message.update(terminal_receipt_by_task.get(task_id) or {})
            elif task_id in anchored and task_id not in latest_progress_by_task:
                # #496: a still-anchored child whose progress rows fell out of the
                # read tail keeps its executor receipt on the one final row the
                # window still holds, so its harness chip needs no "Load older".
                message.update(terminal_receipt_by_task.get(task_id) or {})
            is_summary = str(message.get("system_type") or "") == "task_summary"
            if is_summary or (
                task_id not in summary_task_ids
                and latest_progress_by_task.get(task_id) is message
            ):
                message.update(terminal_truth_by_task.get(task_id) or {})
            if (message.get("is_progress") or is_summary) and task_id in suggested_name_by_task:
                message["suggested_name"] = suggested_name_by_task[task_id]
            # Floor-symmetric closed lineage window for chat FINALS: strip runs
            # AFTER the legacy setdefault injection above as the LAST writer —
            # a strip before the legacy_final_task_ids scan would re-qualify
            # the de-roled row for revival. (The anchored_children clause is
            # load-bearing for mid-run child SPEECH rows (every child chat row
            # carries lineage, not only finals), symmetric with the progress
            # anchor, and since #496 it is live for terminal finals too: a
            # finished child under a represented or living parent keeps its
            # identity. Do not "simplify" it away.)
            stale = (
                not message.get("is_progress")
                and str(message.get("delegation_role") or "").lower() == "subagent"
                and bool(floor)
                and str(message.get("ts") or "") < floor
                and task_id not in anchored
            )
            if stale:
                for key in SUBAGENT_MESSAGE_FIELDS:
                    message.pop(key, None)
                log.debug("Stripped stale subagent lineage from chat final %s", task_id)
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
    project_source_refs: list,
    bindings_by_task: Dict[str, int],
):
    """Build the per-request thread-filter closure (perf2 P3 decomposition).

    Returns the one thread predicate shared by both durable stream readers."""

    def _bound_project_chat(task_id: str, parent_task_id: str = "", root_task_id: str = "") -> int:
        # Resolve by LINEAGE (own binding -> parent -> root) so a subagent's rows
        # classify into its root's project thread (only the root is bound).
        # Same semantics as projects_registry.project_chat_for_task_tree, served
        # from the ONE bindings map preloaded per request (no per-row file reads).
        for candidate in (task_id, parent_task_id, root_task_id):
            tid = str(candidate or "").strip()
            if tid and bindings_by_task.get(tid):
                return int(bindings_by_task[tid])
        return 0

    def _row_matches_thread(entry_chat: int, entry: Optional[dict] = None) -> bool:
        # A post-hoc bound task keeps its original (main) chat_id on its rows
        # but belongs to a project — classify by the durable LINEAGE binding too.
        bound_chat = (
            _bound_project_chat(
                str(entry.get("task_id") or ""),
                str(entry.get("parent_task_id") or ""),
                str(entry.get("root_task_id") or ""),
            ) if isinstance(entry, dict) else 0
        )
        is_project_lifecycle_row = bool(
            isinstance(entry, dict)
            and str(entry.get("type") or "")
            in {"project_started", "project_completion_summary"}
        )
        is_cognitive_projection = bool(
            isinstance(entry, dict)
            and str(entry.get("summary_kind") or "")
            in {"terminal_result_projection", "terminal_root_projection"}
        )
        if thread_id in project_chat_ids:
            # The compact host-stamped lifecycle rows (started + terminal
            # completion) belong only to Main; the Project thread already owns
            # the complete task timeline/result.
            if is_project_lifecycle_row:
                return False
            if bound_chat == thread_id:
                return True
            if isinstance(entry, dict) and _matches_project_source(entry, project_source_refs):
                return True
            return entry_chat == thread_id
        # The hidden partition (Skill Review, and every headless run admitted
        # without a registered project). Main is 1; explicit partition rows never
        # become ordinary conversation history. Keep this after the Project
        # branch so durable task binding stays unchanged.
        if entry_chat == HIDDEN_CHAT_ID:
            return False
        # Main / non-project view: exactly the two host-stamped Project-root
        # lifecycle rows (started + terminal completion) are admitted from the
        # canonical Main chat. Project progress, logs, child traffic, ordinary
        # summaries and raw dialogue stay in Project.
        if is_project_lifecycle_row:
            return entry_chat not in project_chat_ids
        if is_cognitive_projection:
            return False
        if entry_chat in project_chat_ids or bound_chat > 0:
            return False
        return entry_chat not in project_chat_ids

    return _row_matches_thread


def _collect_chat_rows(
    chat_path: pathlib.Path,
    archive_dir: pathlib.Path,
    n_human: int,
    row_matches_thread,
    chat_annotations: Dict[str, Any],
    *,
    include_gaps: bool = False,
) -> tuple[list, int] | tuple[list, int, set[str]]:
    """Read + transform the chat stream.

    Returns ``(rows, quota_row_count)`` — the transformed history records and
    how many read entries satisfied the reader's quota predicate (feeds the
    window-truncation metadata)."""
    # Quiz lifecycle merge (#Q-2b): the chat row froze the card at ask time
    # ("open"); the durable truth lives in the owner_quiz task-result
    # projection. One projection read per distinct asking task, cached for
    # this call.
    quiz_projection_cache: Dict[str, Dict[str, Any]] = {}
    combined: list = []
    chat_quota_rows = 0
    stream_gaps: set[str] = set()
    _chat_counts_toward_quota = _chat_quota_predicate(row_matches_thread)
    try:
        # Rotation-aware archive backfill lives in the module-level
        # _read_chat_history_entries helper (endpoint's thread filter threaded in).
        if include_gaps:
            _chat_entries = _read_chat_history_entries(
                chat_path,
                archive_dir,
                n_human,
                row_matches_thread,
                include_gaps=True,
            )
            _chat_entries, stream_gaps = _chat_entries
        else:
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
            entry_chat = _stored_chat_id(entry.get("chat_id"), 1)
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
            if rec["system_type"] in {"project_started", "project_completion_summary"}:
                # Read-side plain normalization for lifecycle rows persisted
                # before the producer stripped markdown; a no-op on new rows.
                # The durable chat.jsonl is never rewritten.
                rec["text"] = strip_markdown(rec["text"])
                for key in ("project_id", "project_name", "target_label", "status"):
                    if key in entry:
                        rec[key] = str(entry.get(key) or "")
            annotation = _user_annotation(role, rec["client_message_id"], chat_annotations)
            if annotation is not None:
                rec["chat_annotation"] = annotation
            # Skill-review rows already carry the exact-job reference the
            # producer writes (v6.66.0 a776639f); pass it through so the Chat
            # card can lazily fetch the full rendered review. Rows without a
            # job_id (legacy full-text rows) keep today's behavior.
            if rec["system_type"] == "skill_review":
                for key in _SKILL_REVIEW_STRING_FIELDS:
                    rec[key] = str(entry.get(key, "") or "")
                for key in _SKILL_REVIEW_INT_FIELDS:
                    rec[key] = coerce_int(entry.get(key), 0)
                for key in _SKILL_REVIEW_BOOL_FIELDS:
                    rec[key] = bool(entry.get(key))
                rec["executions"] = _review_executions(entry.get("executions"))
            # Delivered document rows carry lightweight media metadata (no
            # base64); surface a msg_type + download_url so the frontend
            # rebuilds the file bubble on reload instead of a bare text line.
            if entry.get("type") == "routing_options":
                # LLM-context grounding row (#198, decision 4=A): the web
                # surface renders the richer picker card from the annotation,
                # so the plain-text list never double-renders here.
                continue
            if entry.get("type") == "document":
                rec["msg_type"] = "document"
                rec["filename"] = str(entry.get("filename") or "file")
                rec["mime"] = str(entry.get("mime") or "application/octet-stream")
                rec["download_url"] = str(entry.get("download_url") or "")
                rec["caption"] = str(entry.get("caption") or "")
                if "size_bytes" in entry:
                    rec["size_bytes"] = coerce_int(entry.get("size_bytes"), 0)
            elif entry.get("type") in {"photo", "video"} and entry.get("download_url"):
                rec.update(msg_type=str(entry["type"]), mime=str(entry.get("mime") or ""),
                           download_url=str(entry["download_url"]),
                           download_url_compat=str(entry.get("download_url_compat") or ""),
                           caption=str(entry.get("caption") or ""))
            elif entry.get("type") == "links":
                rec.update(msg_type="links", actions=list(entry.get("actions") or []), title=str(entry.get("title") or ""))
            elif entry.get("type") == "quiz" and isinstance(entry.get("quiz"), dict):
                quiz = dict(entry["quiz"])
                _qtid = str(entry.get("task_id") or "")
                if _qtid:
                    if _qtid not in quiz_projection_cache:
                        from ouroboros.owner_quiz import quiz_states

                        quiz_projection_cache[_qtid] = quiz_states(chat_path.parent.parent, _qtid)
                    _live = quiz_projection_cache[_qtid].get(str(quiz.get("quiz_id") or ""))
                    if isinstance(_live, dict):
                        quiz["state"] = str(_live.get("state") or quiz.get("state") or "open")
                        for key in ("answered_index", "comment"):  # the recorded answer itself
                            if key in _live:
                                quiz[key] = _live[key]
                rec.update(msg_type="quiz", quiz=quiz)
            if "task_terminal_status" in entry:
                rec["task_terminal_status"] = str(entry.get("task_terminal_status") or "")
            _copy_task_summary_metadata(rec, entry)
            for field in SUBAGENT_MESSAGE_FIELDS:
                if field in entry:
                    rec[field] = entry[field]
            combined.append(rec)
    except Exception as exc:
        log.warning("Failed to read chat history: %s", exc)
    return (combined, chat_quota_rows, stream_gaps) if include_gaps else (combined, chat_quota_rows)


def _collect_progress_rows(
    progress_path: pathlib.Path,
    archive_dir: pathlib.Path,
    n_progress: int,
    row_matches_thread,
    *,
    include_gaps: bool = False,
) -> tuple[list, int] | tuple[list, int, set[str]]:
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
        if str(entry.get("type") or "") == "review_reference":
            return False
        if is_a2a_chat_id(entry.get("chat_id", 1)):
            return False
        entry_chat = _stored_chat_id(entry.get("chat_id"), 1)
        if not row_matches_thread(entry_chat, {"is_progress": True, **entry}):
            return False
        if not str(entry.get("content", entry.get("text", ""))):
            return False
        if str(entry.get("delegation_role") or "").lower() == "subagent" or entry.get("subagent_event"):
            return False
        return True

    combined: list = []
    progress_quota_rows = 0
    stream_gaps: set[str] = set()
    try:
        if include_gaps:
            _progress_entries = _read_progress_history_entries(
                progress_path,
                archive_dir,
                n_progress,
                _progress_counts_toward_quota,
                include_gaps=True,
            )
            _progress_entries, stream_gaps = _progress_entries
        else:
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
            entry_chat = _stored_chat_id(entry.get("chat_id"), 1)
            if not row_matches_thread(entry_chat, {"is_progress": True, **entry}):
                continue
            text = str(entry.get("content", entry.get("text", "")))
            is_review_reference = str(entry.get("type") or "") == "review_reference"
            if not text and not is_review_reference:
                continue
            rec = {
                "text": text,
                "role": "assistant",
                "ts": str(entry.get("ts", "")),
                "is_progress": True,
                "markdown": str(entry.get("format", "")).lower() == "markdown",
                "task_id": str(entry.get("task_id", "")),
            }
            if is_review_reference:
                rec["system_type"] = "review_reference"
                for field in (
                    "surface", "presentation_owner_task_id",
                    "review_fingerprint", "state_revision",
                ):
                    if field in entry:
                        rec[field] = str(entry.get(field) or "")
            if isinstance(entry.get("lifecycle"), dict):
                rec["lifecycle"] = dict(entry.get("lifecycle") or {})
            for field in _PROGRESS_META_FIELDS:
                if field in entry:
                    rec[field] = entry[field]
            combined.append(rec)
    except Exception as exc:
        log.warning("Failed to read progress log: %s", exc)
    return (
        (combined, progress_quota_rows, stream_gaps)
        if include_gaps
        else (combined, progress_quota_rows)
    )


def _fold_task_bound_skill_reviews(combined: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Fold already-read terminal Skill refs by their logical task-bound group.

    This is deliberately a bounded read projection over ``combined``. It never
    replays Skill history and never claims an authoritative total beyond the
    references present in this Chat window.
    """
    groups: Dict[tuple[str, str, str], list[tuple[int, Dict[str, Any]]]] = {}
    for index, row in enumerate(combined):
        if row.get("is_progress") or str(row.get("system_type") or "") != "skill_review":
            continue
        skill = str(row.get("skill") or "")
        group_id = str(row.get("group_id") or "")
        owner = str(row.get("presentation_owner_task_id") or "")
        # Legacy compatibility is intentionally narrow: only the producer's
        # historical root+skill identity is safe to reconstruct. An initiator
        # task id alone is never treated as presentation ownership.
        if not group_id and not owner:
            root_task_id = str(row.get("root_task_id") or "")
            if root_task_id and skill:
                group_id = f"task:{root_task_id}:{skill}"
                owner = root_task_id
        if not group_id or not owner or not skill:
            continue
        groups.setdefault((group_id, owner, skill), []).append((index, row))
    if not groups:
        return combined

    replacements: Dict[int, Dict[str, Any]] = {}
    removed: set[int] = set()
    for (group_id, owner, skill), indexed_rows in groups.items():
        indexed_rows.sort(key=lambda item: (str(item[1].get("ts") or ""), item[0]))
        latest_job_position: Dict[str, int] = {}
        for position, (_index, row) in enumerate(indexed_rows):
            job_id = str(row.get("job_id") or "")
            if job_id:
                latest_job_position[job_id] = position
        surviving_rows = [
            item for position, item in enumerate(indexed_rows)
            if not str(item[1].get("job_id") or "")
            or latest_job_position[str(item[1].get("job_id") or "")] == position
        ]
        attempts: list[Dict[str, Any]] = []
        for position, (_index, row) in enumerate(surviving_rows):
            attempt = {
                "ts": str(row.get("ts") or ""),
                "task_id": str(row.get("task_id") or ""),
                # Legacy jobless rows have no exact-detail ref. Their compact
                # terminal text is therefore the only useful attempt body.
                "text": str(row.get("text") or ""),
                "superseded": position < len(surviving_rows) - 1,
                "executions": _review_executions(row.get("executions")),
            }
            for key in _SKILL_REVIEW_STRING_FIELDS:
                if key in row:
                    attempt[key] = str(row.get(key) or "")
            for key in _SKILL_REVIEW_INT_FIELDS:
                if key in row:
                    attempt[key] = coerce_int(row.get(key), 0)
            for key in _SKILL_REVIEW_BOOL_FIELDS:
                if key in row:
                    attempt[key] = bool(row.get(key))
            attempt.update(
                skill=skill,
                group_id=group_id,
                presentation_owner_task_id=owner,
            )
            attempts.append(attempt)
        latest_index, latest_source = surviving_rows[-1]
        latest = dict(latest_source)
        latest["group_id"] = group_id
        latest["presentation_owner_task_id"] = owner
        latest["review_group"] = {
            "surface": "skill",
            "id": group_id,
            "skill": skill,
            "presentation_owner_task_id": owner,
            "projected_attempt_count": len(attempts),
            "count_is_authoritative": False,
            "attempts": attempts,
        }
        replacements[latest_index] = latest
        removed.update(index for index, _row in indexed_rows if index != latest_index)
    return [
        replacements.get(index, row)
        for index, row in enumerate(combined)
        if index not in removed
    ]


def _active_lifecycle_row(row_matches_thread) -> Optional[Dict[str, Any]]:
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
            from supervisor.message_bus import notification_chat_route

            route = notification_chat_route(active.get("chat_id"), 0)
            chat_id = int(route if route is not None else 0)
            row = {
                "text": text,
                "role": "assistant",
                "ts": utc_now_iso(),
                "is_progress": True,
                "markdown": False,
                "task_id": str(active.get("chat_task_id") or ""),
                "root_task_id": str(active.get("root_task_id") or ""),
                "group_id": str(active.get("group_id") or ""),
                "presentation_owner_task_id": str(
                    active.get("presentation_owner_task_id") or ""
                ),
                "origin_task_id": str(active.get("origin_task_id") or ""),
                "origin_root_task_id": str(active.get("origin_root_task_id") or ""),
                "chat_id": chat_id,
                "lifecycle": lifecycle,
                "lifecycle_virtual": True,
            }
            if row_matches_thread(chat_id, row):
                return row
    except Exception as exc:
        log.debug("Failed to synthesize active lifecycle history: %s", exc)
    return None


def _apply_window_quotas(
    data_dir: pathlib.Path,
    thread_id: int,
    project_chat_ids: set,
    combined: list,
    n_human: int,
    n_progress: int,
) -> tuple[list, Dict[str, Dict[str, Any]], bool, bool, bool, str, set]:
    """Quota slicing, origin fallback, and the lineage floor/cap (perf2 P3).

    Returns ``(messages, result_cache, human_rows_dropped, lineage_truncated,
    review_overlays_truncated, floor, anchored_children)`` for annotation and
    window metadata (the last two feed the chat-final lineage strip in
    ``_annotate_terminal_task_truth``).
    """
    # Tail human conversation and progress telemetry with SEPARATE quotas so a
    # burst of progress messages can never push the user's real conversation out
    # (the previous single combined[-limit:] tail). Subagent lineage is kept on
    # top of the progress quota so a flood can't evict a RECENT child's lifecycle
    # events (the client rebuilds child-card lineage from them). Older lineage
    # survives when the window still describes its topology (#496, below);
    # lineage whose parent is neither represented here nor alive is dropped, so
    # a finished swarm cannot recreate an orphaned "Working" parent card whose
    # own terminal row has already aged out of the window.
    def _is_subagent_lineage(m: dict) -> bool:
        # Only true SUBAGENT lifecycle (delegation_role 'subagent' or any
        # subagent_event) is lineage-critical. delegation_role can also be
        # 'root', which must NOT bypass the progress quota.
        return str(m.get("delegation_role") or "").lower() == "subagent" or bool(m.get("subagent_event"))

    # NOTE: guard 0 explicitly — Python's list[-0:] is list[0:] (the WHOLE list),
    # so a `[-quota:]` slice with quota==0 would leak everything, not nothing.
    def _is_folded_review(m: dict) -> bool:
        return (
            not m.get("is_progress")
            and str(m.get("system_type") or "") == "skill_review"
            and isinstance(m.get("review_group"), dict)
        )

    folded_reviews = [m for m in combined if _is_folded_review(m)]
    # Folded Skill groups are a task-detail hydration overlay, just like Plan
    # references below: neither consumes human/telemetry rows, but distinct
    # owners still fan out to one lazy task-detail read each. Bound owners to
    # the requested progress window while retaining every group/attempt for a
    # selected owner. Load older already expands this one shared window.
    folded_review_owner_latest: Dict[str, str] = {}
    for message in folded_reviews:
        owner = str(message.get("presentation_owner_task_id") or "")
        if not owner:
            continue
        latest = str(message.get("ts") or "")
        if latest >= folded_review_owner_latest.get(owner, ""):
            folded_review_owner_latest[owner] = latest
    folded_review_owners = sorted(
        folded_review_owner_latest,
        key=lambda owner: (folded_review_owner_latest[owner], owner),
    )
    folded_reviews_truncated = len(folded_review_owners) > n_progress
    selected_folded_review_owners = set(
        folded_review_owners[-n_progress:] if n_progress > 0 else []
    )
    folded_reviews = [
        message for message in folded_reviews
        if str(message.get("presentation_owner_task_id") or "")
        in selected_folded_review_owners
    ]
    human = sorted(
        (
            m for m in combined
            if not m.get("is_progress") and not _is_folded_review(m)
        ),
        key=lambda m: m.get("ts", ""),
    )
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
            synthesized = _origin_fallback_rows(data_dir, thread_id, list(human_tail))
            if synthesized:
                human_tail = sorted(
                    human_tail + synthesized, key=lambda m: m.get("ts", "")
                )
        except Exception:
            log.debug("Project origin fallback synthesis failed", exc_info=True)
    # Plan references are durable invalidations on the existing progress rail.
    # They do not consume visible telemetry quota. Task detail remains
    # authority, so keep only the latest ref per owner, then independently cap
    # the overlay to the same requested progress window.
    review_references_by_owner: Dict[tuple[str, str], Dict[str, Any]] = {}
    for message in progress:
        if str(message.get("system_type") or "") != "review_reference":
            continue
        key = (
            str(message.get("surface") or ""),
            str(message.get("presentation_owner_task_id") or message.get("task_id") or ""),
        )
        review_references_by_owner[key] = message
    review_references = sorted(
        review_references_by_owner.values(), key=lambda m: m.get("ts", ""),
    )
    review_references_truncated = len(review_references) > n_progress
    review_overlays_truncated = (
        folded_reviews_truncated or review_references_truncated
    )
    review_references = review_references[-n_progress:] if n_progress > 0 else []
    other = [
        m for m in progress
        if not _is_subagent_lineage(m)
        and str(m.get("system_type") or "") != "review_reference"
    ]
    other_tail = other[-n_progress:] if n_progress > 0 else []
    # Recency floor = oldest retained telemetry row. Lineage older than it is
    # dropped unless anchored below, so an ABSENT finished swarm cannot
    # re-materialise as a stuck "Working" parent card.
    floor = str(other_tail[0].get("ts") or "") if other_tail else ""
    lineage_rows = [m for m in progress if _is_subagent_lineage(m)]
    # perf2 P3 variant A (owner decision 2026-08-09): a QUIET but still-ACTIVE
    # child must survive the recency floor — its card is reproducible only from
    # these lineage rows. Terminal truth for the lineage task ids of the READ
    # window is resolved BEFORE the floor/cap slice; the same cache then feeds
    # _annotate_terminal_task_truth after the slice, so each task_results file
    # is read at most once per request. The effective-status orphan guard
    # resolves a long-dead raw "running" child as failed, i.e. terminal, so a
    # dead child cannot pin its own lineage forever.
    # #496: recency is the wrong PROXY for "does this child belong to a topology
    # the window still describes". The honest predicate (owner liveness doctrine
    # 2026-08-23) anchors a child when the child is itself alive, OR its parent is
    # REPRESENTED by this response, OR the parent is alive — and a child so
    # anchored represents ITS OWN children, so a swarm is kept or dropped whole.
    # "Represented" is narrower than "some emitted row carries this task id": a
    # delivery (photo, document, quiz) carries one mid-run and proves nothing
    # about the task's card — the client refuses role+task_id as a conclusion for
    # the same reason — so counting those would let a parent with no closable fact
    # re-anchor a finished swarm, the zombie the floor existed to prevent.
    result_cache: Dict[str, Dict[str, Any]] = {}
    anchored_children: set = set()
    child_rows = [
        m for m in (*lineage_rows, *human_tail, *other_tail)
        if _is_subagent_lineage(m) and m.get("task_id")
    ]
    if floor and child_rows:
        def _represents(m: dict) -> str:
            """The task this row proves is present in the window, or ""."""
            owner = str(m.get("presentation_owner_task_id") or "")
            if owner:
                return owner  # a folded review / plan reference names its owner
            task_id = str(m.get("task_id") or "")
            if not task_id or m.get("is_progress"):
                return task_id  # telemetry IS the task's own narration
            # Its own message or summary closes a task; a typed delivery or a
            # host project lifecycle row does not.
            return task_id if str(m.get("system_type") or "") in ("", "task_summary") else ""

        # Seeded outside the lineage still being decided: a PRE-FLOOR child's own
        # final cannot represent it to its children before the window has decided
        # whether that child belongs at all. At or after the floor a lineage row
        # is never stripped, so it is closable and counts like any other.
        seed = (*other_tail, *folded_reviews, *review_references,
                *(m for m in human_tail
                  if not _is_subagent_lineage(m) or str(m.get("ts") or "") >= floor))
        represented = {owner for owner in map(_represents, seed) if owner}
        try:
            from ouroboros.task_status import FINAL_STATUSES

            def _alive(task_id: str) -> bool:
                # Shared with the terminal-truth annotation above: a stored
                # "completed" whose post-task synthesis is still OPEN is
                # FINALIZING, not terminal, so its children must not be dropped.
                result = _load_terminal_result(data_dir, task_id, result_cache)
                status = str(result.get("status") or "")
                checkpoint = result.get("root_phase_checkpoint")
                synthesis = (
                    str(checkpoint.get("post_task_synthesis") or "")
                    if isinstance(checkpoint, dict) else ""
                )
                if post_task_synthesis_is_open(synthesis) and status not in {"failed", "cancelled"}:
                    return True
                return bool(status) and status not in FINAL_STATUSES

            # Fixed point: an anchored child is itself represented for its own
            # children. Seeded only from facts above, so a cycle or a
            # self-parenting row can never bootstrap itself into the window.
            pending = list(child_rows)
            while True:
                grew = False
                for m in pending:
                    task_id = str(m.get("task_id") or "")
                    if task_id in anchored_children:
                        continue
                    # The IMMEDIATE parent only: anchoring to a represented tree
                    # ROOT would keep a leaf whose own parent is absent and dead,
                    # and the client mints that missing parent's card from the
                    # leaf's lineage — a card this response cannot close. The
                    # fixed point carries a live root down one real link at a time.
                    parent = str(m.get("parent_task_id") or "")
                    if (
                        (parent and (parent in represented or parent in anchored_children))
                        or _alive(task_id)
                        or (parent and parent != task_id and _alive(parent))
                    ):
                        anchored_children.add(task_id)
                        grew = True
                if not grew:
                    break
        except Exception as exc:
            log.debug("Failed to resolve pre-floor lineage terminal truth: %s", exc)
    lineage = [
        m for m in lineage_rows
        if not floor
        or str(m.get("ts") or "") >= floor
        or str(m.get("task_id") or "") in anchored_children
    ]
    # The cap stays a suffix slice applied AFTER the floor: an active child with
    # more rows than the cap keeps its newest rows, so its card stays alive.
    lineage_truncated = len(lineage) > _LINEAGE_CAP
    if lineage_truncated:
        lineage = lineage[-_LINEAGE_CAP:]  # keep the most recent lineage events
    progress_tail = lineage + other_tail + review_references
    messages = sorted(
        human_tail + folded_reviews + progress_tail,
        key=lambda m: m.get("ts", ""),
    )
    return (
        messages, result_cache, human_rows_dropped, lineage_truncated,
        review_overlays_truncated, floor, anchored_children,
    )


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
    review_overlays_truncated: bool,
    stream_gaps: Optional[Dict[str, set[str]]] = None,
) -> Dict[str, Any]:
    """Additive window metadata (perf2 P3; frozen contract extended explicitly).

    The reader learns WHETHER this window is the complete reachable history
    and WHAT bounded it — the quota tail slice ("quota"), the bounded archive
    backfill ("archive_floor"), or the lineage cap ("lineage_cap"). The client
    gates its "Load older" affordance on this instead of guessing; no existing
    field changes meaning."""
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
        "quota" if review_overlays_truncated else None,
        "lineage_cap" if lineage_truncated else None,
    ):
        if cause and cause not in truncated_by:
            truncated_by.append(cause)
    for stream in ("chat", "progress"):
        for gap in sorted((stream_gaps or {}).get(stream, set())):
            cause = f"{stream}_{gap}"
            if cause not in truncated_by:
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
    project_chat_ids, project_source_refs, chat_annotations, bindings_by_task = (
        _project_history_context(data_dir, thread_id)
    )
    row_matches_thread = _make_thread_filter(
        thread_id, project_chat_ids, project_source_refs, bindings_by_task
    )
    chat_path = data_dir / "logs" / "chat.jsonl"
    progress_path = data_dir / "logs" / "progress.jsonl"
    archive_dir = data_dir / "archive"
    combined, chat_quota_rows, chat_gaps = _collect_chat_rows(
        chat_path, archive_dir, n_human, row_matches_thread, chat_annotations,
        include_gaps=True,
    )
    progress_rows, progress_quota_rows, progress_gaps = _collect_progress_rows(
        progress_path, archive_dir, n_progress, row_matches_thread,
        include_gaps=True,
    )
    combined.extend(progress_rows)
    combined = _fold_task_bound_skill_reviews(combined)
    lifecycle_row = _active_lifecycle_row(row_matches_thread)
    if lifecycle_row is not None:
        combined.append(lifecycle_row)
    (
        messages, result_cache, human_rows_dropped, lineage_truncated,
        review_overlays_truncated, floor, anchored_children,
    ) = _apply_window_quotas(
        data_dir, thread_id, project_chat_ids, combined, n_human, n_progress
    )

    # Annotate progress messages whose task already reached a terminal (or
    # cancel-intent) status on disk. Tasks torn down by crash storm, hard
    # timeout, or cancellation emit a live task_done but never write a
    # task_summary, so on reload/reconnect the client would otherwise replay
    # their progress and re-inflate a "Working" spinner that never resolves.
    # Runs AFTER the quota slice — on the rows actually emitted — so the
    # response pays only for in-window task ids and the truth always lands on
    # a row the client will see (see _annotate_terminal_task_truth).
    _annotate_terminal_task_truth(
        messages, data_dir, result_cache=result_cache,
        floor=floor, anchored_children=anchored_children,
    )

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
            human_rows_dropped, lineage_truncated, review_overlays_truncated,
            {"chat": chat_gaps, "progress": progress_gaps},
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
        # sends no quota params. The legacy `limit` parameter (shipped CLIs sent
        # it while the server ignored it) is honored as the n_human default, so
        # an old client that asks for N human rows finally gets N, within cap; a
        # garbled or non-positive `limit` means "absent"; an explicit n_human wins.
        n_human = _int_param("n_human", _int_param("limit", 0, _MAX_N_HUMAN) or _DEFAULT_N_HUMAN, _MAX_N_HUMAN)
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
