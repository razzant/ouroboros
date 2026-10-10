"""The record passport: which local records an external observer may rely on.

Ouroboros keeps its history on the machine as JSONL ledgers. A deployment that
wants monitoring connects its own tools to them (a file-tailing collector, an
error tracker inside the process, a companion that joins rows); nothing leaves
the machine by default. This module names the supported subset of those
records: the anchor rows, the keys every one of them carries, which may be
null, which carry user or model content, how rows correlate, where they live
and for how long, how they rotate, which are written twice, how older rows
differ, and where money must be read instead. It describes what the writers
already write and changes none of them; the observer owns interpretation and
delivery.

Compatibility: a guaranteed field is a key every row of its type written by the
current writers carries (null or "" are values, not absence). The guaranteed
sets are deliberately narrow and fixed within ``RECORD_CONTRACT_VERSION``; a
new field arrives as optional, and an optional field is present today with no
promise. Promoting, removing or renaming a guaranteed field, or changing its
meaning, needs a successor version and a migration note
(docs/architecture/11-frozen-contracts-v1.md §11.3). A field this module does
not list may change without notice. tests/test_record_contract.py checks every
guaranteed field against its writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

RECORD_CONTRACT_VERSION = 1


def _fields(names: str) -> frozenset[str]:
    return frozenset(names.split())


@dataclass(frozen=True)
class AnchorRow:
    """One supported row type. ``log`` is the file stem and ``plane`` where rows of
    this type are written (``PLANES``). Field names are top-level keys; a field in
    ``content`` can hold user, model or exception text anywhere inside it.
    ``natural_key`` is a correlation and replay hint, not a uniqueness promise."""

    type: str
    log: str
    plane: str
    meaning: str
    guaranteed: frozenset[str]
    nullable: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()
    content: frozenset[str] = frozenset()
    natural_key: tuple[str, ...] = ()

    @property
    def metadata(self) -> frozenset[str]:
        """Fields that carry no content by name: listed, and not content."""
        return (self.guaranteed | self.optional) - self.content


PLANES: Mapping[str, str] = {
    "canonical": (
        "<data>/logs/<log>.jsonl, rotated into <data>/archive/<log>_<UTC YYYYMMDDTHHMMSS>[_<n>].jsonl; "
        "the archive chain is never deleted."
    ),
    "drive_logs": (
        "The task's own log directory: the canonical file for Main turns and pooled roots; for every subagent "
        "and forked-memory task its child drive <data>/state/headless_tasks/<task_id>/data/logs/<log>.jsonl, "
        "found through task_results/<task_id>.json child_drive_root. A child drive is never rotated and is "
        "deleted once its task is terminal and older than OUROBOROS_GC_RETENTION_DAYS, or at once for a "
        "cancelled subagent."
    ),
    "both": "The task's log directory and the canonical file receive the same row (see REPLICA_RULE).",
}

_TOOL = "ts type task_id tool invocation_id"
_TOOL_OPTIONAL = (
    "root_task_id parent_task_id delegation_role task_depth task_attempt execution_id round_id llm_call_id "
    "tool_call_id args args_source_ref args_source_status terminal_wait routing_action "
    "completion_control start_log"
)
_CONTEXT_FIT = (
    "context_route_fp estimated_prompt_tokens context_fit_mode context_profile context_measurement_basis "
    "context_measurement_density context_raw_input_tokens context_reply_allowance_tokens "
    "context_target_total_tokens context_capacity_total_tokens context_target_deficit_tokens "
    "context_capacity_deficit_tokens context_reclaim_goal_tokens context_target_miss "
    "context_automatic_pass_used context_predicted_capacity_miss"
)
_COST_META = (
    "accounted_upper_bound_usd accounted_upper_bound_usd_with_children cost_presentation cost_accounting_status "
    "cost_accounting_error cost_final cost_with_children_partial unknown_unmetered non_final_rows reserved_usd "
    "unresolved_upper_bound_usd ledger_integrity_degraded"
)

ANCHOR_ROWS: tuple[AnchorRow, ...] = (
    AnchorRow(
        "task_received", "events", "drive_logs",
        "A worker accepted a task. `task` is the sanitized task: `task.id` joins it, and the rest (text up to "
        "4000 chars, contract, metadata) is content.",
        _fields("ts type task"), optional=_fields("activity_emitted_at"), content=_fields("task"),
        natural_key=("type", "ts"),
    ),
    AnchorRow(
        "task_done", "events", "canonical",
        "The supervisor accepted a task's terminal; task_results/<task_id>.json stays the lifecycle authority.",
        _fields("ts type task_id task_type chat_id status reason_code"), nullable=_fields("reason_code"),
        optional=_fields(_COST_META + " outcome_axes _is_direct_chat root_phase_checkpoint artifact_status "
                                      "total_rounds prompt_tokens completion_tokens review_status review_projection "
                                      "model_execution artifact_bundle cancel_origin typed_routing_action"),
        content=_fields("outcome_axes review_status review_projection artifact_bundle cancel_origin"),
        natural_key=("task_id", "type", "ts"),
    ),
    AnchorRow(
        "task_error", "events", "drive_logs",
        "A task ended on an unexpected exception, or deep self-review failed (that row carries reason_code).",
        _fields("ts type task_id error"), optional=_fields("traceback reason_code"),
        content=_fields("error traceback"), natural_key=("task_id", "type", "ts"),
    ),
    AnchorRow(
        "task_metrics_event", "supervisor", "canonical",
        "Duration and tool counts of a task that reached the pipeline's own finalization; the counts are null "
        "(unknown, never 0) for a task that ended without loop evidence.",
        _fields("ts type task_id task_type duration_sec tool_calls tool_errors reason_code"),
        nullable=_fields("tool_calls tool_errors"),
        optional=_fields("outcome_axes routing_tool_calls completion_tool_calls tool_call_counts chat_id "
                         "parent_task_id root_task_id"),
        content=_fields("outcome_axes"), natural_key=("task_id", "type", "ts"),
    ),
    AnchorRow(
        "task_cost_finalized", "events", "canonical",
        "The task's cost facts as the ledger stood when its accounting closed; supersedes task_done's.",
        _fields("ts type task_id root_task_id"),
        optional=_fields(_COST_META + " post_task_status post_task_stop_reason total_rounds prompt_tokens "
                                      "completion_tokens"),
        natural_key=("task_id", "type", "ts"),
    ),
    AnchorRow(
        "llm_round", "events", "drive_logs",
        "One settled logical model round of a task's main loop. duration_ms spans it on the monotonic clock "
        "from its first started frame to settlement: retries, fallbacks and in-round waits count however long "
        "they take, the context fit before it and the tools after it do not, time the machine slept may not, "
        "and a budget pause restarts it. A forced final answer after a round that never settled reuses its "
        "round_id and includes those failed attempts. ledger_attempt_ids are the settling call's attempts "
        "(physical_attempt_id the last); earlier failed attempts are llm_api_error rows or non-anchor rows "
        "(COVERAGE_GAPS) with the same round_id. first_request_at and first_answer_at are the execution's "
        "first request and first answer, repeated on later rounds. A Claudexor operation id rides inside "
        "`claudexor`.",
        _fields("ts type task_id execution_id round_id llm_call_id round model provider prompt_tokens "
                "completion_tokens cached_tokens cache_write_tokens duration_ms ledger_attempt_ids "
                "physical_attempt_id"),
        nullable=_fields("cached_tokens duration_ms physical_attempt_id"),
        optional=_fields("reasoning_effort source model_category prompt_cache_ttl cache_hit_rate cache_cold_restart "
                         "gap_since_prev_round_sec cost_usd request_ref response_ref effort effort_resolution "
                         "request_wire claudexor first_request_at first_answer_at " + _CONTEXT_FIT),
        content=_fields("claudexor"), natural_key=("llm_call_id",),
    ),
    AnchorRow(
        "llm_usage", "events", "canonical",
        "A compatibility projection of a model call's usage as the supervisor received it: main-loop, review "
        "and tool-model calls (COVERAGE_GAPS names calls that write none); accounting_authority names the "
        "ledger. Never a money source (ACCOUNTING_RULE).",
        _fields("ts type task_id root_task_id parent_task_id delegation_role category model provider "
                "prompt_tokens completion_tokens cached_tokens cache_write_tokens accounting_authority "
                "ledger_attempt_ids"),
        optional=_fields(
            "task_group_id requested_model_lane effective_model_lane api_key_type model_category source "
            "cost_estimated cost cost_known prompt_cache_ttl projection_update_status llm_call_id execution_id "
            "round_id round chat_id effort request_wire effort_resolution claudexor reasoning_pin "
            "reasoning_effort_clamped web_search_sources review_skill review_wave_id review_slot_id"),
        content=_fields("web_search_sources claudexor"), natural_key=("ledger_attempt_ids",),
    ),
    AnchorRow(
        "llm_api_error", "events", "drive_logs",
        "One failed model attempt inside a round, with its classification and the attempts it reserved.",
        _fields("ts type task_id execution_id round_id llm_call_id round attempt model error error_kind "
                "status_code ledger_attempt_ids"),
        nullable=_fields("status_code"),
        optional=_fields("retry_same_request provider_code provider_message requested_profile observed_route "
                         "request_ref operation_id physical_attempt_id attempt_custody_state provider_error_type "
                         "transport_cause_type " + _CONTEXT_FIT),
        content=_fields("error provider_message"), natural_key=("llm_call_id",),
    ),
    AnchorRow(
        "tool_call_started", "tools", "both",
        "The host began one tool invocation; written before the handler runs.",
        _fields(_TOOL), optional=_fields(_TOOL_OPTIONAL + " timeout_sec"), content=_fields("args"),
        natural_key=("invocation_id", "type"),
    ),
    AnchorRow(
        "tool_call", "tools", "both",
        "The invocation's settlement, one per invocation.",
        _fields(_TOOL + " elapsed_ms is_error status"),
        optional=_fields(_TOOL_OPTIONAL + " result_preview args_ref result_ref tool_result_meta exit_code signal "
                                          "duration_ms timed_out killed_by_host pre_exec_failure resolved_runtime "
                                          "runtime_provenance ws_relay_failures"),
        content=_fields("args result_preview tool_result_meta"), natural_key=("invocation_id", "type"),
    ),
    AnchorRow(
        "tool_call_timeout", "tools", "both",
        "The loop stopped waiting for the invocation; its settlement may still follow as a tool_call row.",
        _fields(_TOOL + " timeout_sec waited_ms"),
        optional=_fields(_TOOL_OPTIONAL + " result_preview result_ref"),
        content=_fields("args result_preview"), natural_key=("invocation_id", "type"),
    ),
    AnchorRow(
        "worker_crash", "supervisor", "canonical",
        "A worker caught an exception in one of its phases; also a logging record with its stack.",
        _fields("ts type worker_id pid phase error"), optional=_fields("traceback"),
        content=_fields("error traceback"), natural_key=("pid", "phase", "ts"),
    ),
    AnchorRow(
        "worker_dead_detected", "supervisor", "canonical",
        "The supervisor found a worker process gone (a hard kill or crash the worker could not record).",
        _fields("ts type worker_id exitcode busy_task_id"), nullable=_fields("busy_task_id"),
        optional=_fields("task_type task_description uptime_sec attempt signal"),
        content=_fields("task_description"), natural_key=("worker_id", "ts"),
    ),
    AnchorRow(
        "supervisor_loop_stall", "supervisor", "canonical",
        "The supervisor loop stopped publishing its liveness stamp (new-message intake starved); the loop's "
        "last published facts (phase, CPU) ride along when it published any.",
        _fields("ts type stalled_sec"),
        optional=_fields("phase loop_thread_cpu_sec cpu_interval_sec loop_thread_cpu_total_sec daemon_pin_matched "
                         "max_event_lag_sec stack loop_stack_truncated"),
        content=_fields("stack"), natural_key=("type", "ts"),
    ),
    AnchorRow(
        "supervisor_loop_stall_end", "supervisor", "canonical",
        "The stalled supervisor loop ticked again; where its thread spent the stall. phase is null when the "
        "loop had published no facts.",
        _fields("ts type stalled_sec phase"), nullable=_fields("phase"),
        optional=_fields("loop_thread_cpu_sec cpu_interval_sec samples top_frames last_stack"),
        content=_fields("top_frames last_stack"), natural_key=("type", "ts"),
    ),
)

ANCHOR_BY_TYPE: Mapping[str, AnchorRow] = {row.type: row for row in ANCHOR_ROWS}

CORRELATION_IDS: Mapping[str, str] = {
    "task_id": "The physical task.",
    "root_task_id": "The logical subtree and budget authority; survives a retry that issues a fresh task_id.",
    "parent_task_id": "The task that started this one.",
    "delegation_role": "Lineage role; `root` for a root task.",
    "task_depth": "Depth below the root task.",
    "task_attempt": "The worker attempt of the task.",
    "chat_id": "The audience: 1 Main, 0 the hidden partition, >= 1000 a project chat, negative agent-to-agent.",
    "execution_id": "One task execution.",
    "round_id": "<execution_id>:round:<n>, one logical model round.",
    "llm_call_id": "One model call attempt inside a round.",
    "invocation_id": "One host tool invocation.",
    "tool_call_id": "The provider's tool-call id; providers reuse it, so it is not an identity.",
    "ledger_attempt_ids": "The usage-ledger physical attempts behind a row.",
    "physical_attempt_id": "The physical attempt that carried the response or failed.",
    "operation_id": "A domain-local operation (a Claudexor model operation, a review); not a cross-log id.",
    "worker_id": "A pool worker slot; pid is the process.",
}

ENVELOPE_RULE = (
    "One JSON object per line; a line can exceed 300 KB, and a reader skips a line that does not parse. `ts` "
    "is the producer's ISO-8601 UTC stamp with an offset; rows relayed through the supervisor keep it, so "
    "file order is not time order: sort by ts. `type` names every row of events, tools, supervisor and "
    "progress; chat.jsonl rows carry `direction` (in, out, system) and `type` only when typed. There is no "
    "event id: join and replay by the natural key."
)

ROTATION_RULE = (
    "Each canonical JSONL log is renamed atomically into the archive once it passes the rotation size (about "
    "800 KB today; the size is not part of this contract), checked on every supervisor loop turn under the "
    "writers' append lock, and a fresh live file is created; a busy lock postpones the rotation. Read the "
    "live file first, then <log>_*.jsonl in the archive sorted by name (never parse the stamp), and drop the "
    "just-rotated generation by inode; a tailer that was down must read the archive generations it missed. "
    "Child-drive logs are never rotated."
)

REPLICA_RULE = (
    "Only tools rows are written twice: when the task's log directory is not the canonical one, it and the "
    "canonical tools.jsonl receive the same row. Read tools rows from the canonical file, or drop duplicates "
    "by (invocation_id, type); a legacy row without invocation_id is a duplicate only when the whole row is "
    "equal. No other anchor row is mirrored."
)

LEGACY_RULE = (
    "Rows carry no version stamp: a field missing from an older row is unknown, never a default. Known "
    "transitions: tools rows without invocation_id predate 2026-09-27 and stand alone (a timeout was then a "
    "tool_call row without is_error); direct-root tools rows lacked lineage before 2026-09-21; cost_usd and "
    "cost_usd_with_children on task rows before 7.0.0 mean accounted_upper_bound_usd[_with_children]; "
    "outcome_axes members may be bare strings; llm_round duration_ms, ledger_attempt_ids and "
    "physical_attempt_id, and llm_api_error ledger_attempt_ids, start with RECORD_CONTRACT_VERSION 1."
)

ACCOUNTING_RULE = (
    "Money comes only from the validated accounting views, named by API rather than storage. Totals: GET "
    "/api/state `accounting` and GET /api/cost-breakdown `accounting`; respect available (both) and "
    "integrity_degraded (/api/state), and unavailable means unknown, never $0. Per task: GET "
    "/api/tasks/{task_id} `cost_breakdown` for roots and the cost fields of task_done, task_cost_finalized "
    "and the task result; respect cost_accounting_status, ledger_integrity_degraded, cost_final and "
    "non_final_rows. Those are snapshots and upper bounds as of their row, not additive transactions: "
    "task_cost_finalized supersedes task_done for the same task, and a parent's _with_children figure already "
    "contains its children. Never sum llm_usage (cost is often null and the row is a projection) or llm_round "
    "cost_usd, and never read or tail the money store, whose file is an implementation detail. The views "
    "carry no as-of time: an answer reflects the store when it was read, and a read that cannot reach the "
    "store within a short wait reports unavailable rather than an older figure."
)

COVERAGE_GAPS = (
    "No canonical row marks a task start: task_results/<task_id>.json with status running is the start fact.",
    "Worker rows (llm_round, llm_api_error, task_received, task_error) of a task with a child drive vanish "
    "with the drive after retention; canonical rows and the archive stay. Ship child drives before then.",
    "llm_round covers a task's main loop only: review, search, vision and memory calls appear only as "
    "llm_usage, without a duration. A round that never settles has no llm_round: its failed attempts are "
    "llm_api_error rows or the non-anchor rows named below.",
    "Skill reviews started from the Skills page write no llm_usage row; their usage is in the accounting "
    "views only.",
    "A terminal whose supervisor handler failed has no task_done: supervisor.jsonl worker_event_handler_error "
    "records it, and task_results stays the authority.",
    "tools rows carry no chat_id; task_done carries no root_task_id (join task_results or task_cost_finalized).",
    "Neighbours of llm_api_error (local_context_overflow, remote_context_overflow, "
    "llm_non_retryable_same_request, provider_incomplete_response, llm_empty_response, provider_body_error, "
    "llm_retry_deadline_exhausted, llm_not_dispatched), worker_event_handler_error and worker_crash_task_dump "
    "are recorded but are not anchors: their fields carry no guarantee.",
    "Uncaught thread and main-thread exceptions are stdlib logging records, not JSONL rows.",
    "Text logs (server.log, launcher.log, agent_stdout.log) and the Claudexor daemon's journals are outside "
    "this contract.",
)

__all__ = [
    "ACCOUNTING_RULE",
    "ANCHOR_BY_TYPE",
    "ANCHOR_ROWS",
    "AnchorRow",
    "CORRELATION_IDS",
    "COVERAGE_GAPS",
    "ENVELOPE_RULE",
    "LEGACY_RULE",
    "PLANES",
    "RECORD_CONTRACT_VERSION",
    "REPLICA_RULE",
    "ROTATION_RULE",
]
