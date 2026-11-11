"""Provenance contracts for benchmark artefacts (v6.81.0).

Two claims a benchmark artefact must never make falsely:

* FIX A — a container carries only the provider credentials the run DECLARED, and the
  manifest discloses which ones it got (by fingerprint, never by value);
* FIX B — a task the RUNTIME stopped for a reason other than finishing (the per-task USD
  reservation rail, the round cap, the loop-local deadline) says so in the artefact, instead
  of being indistinguishable from an honest failure. The vocabulary for that is DERIVED from
  `ouroboros.outcomes.BEST_EFFORT_REASON_CODES`, never restated: the first cut hand-copied it
  and got it wrong in both directions at once (see
  `test_truncation_vocabulary_is_derived_from_the_runtime_not_restated`).
"""

from __future__ import annotations

import ast
import pathlib
import re


from devtools.benchmarks.common.result_index import (
    RUNTIME_TRUNCATION_REASON_CODES,
)
from ouroboros.provider_models import (
    provider_credential_plan,
)
from ouroboros.request_wire_contract import WIRE_REASON_CODES
from tests._source_scan import parse


# --------------------------------------------------------------------------- FIX A


def test_a_settings_mapping_with_no_slots_fails_OPEN_and_discloses_it(monkeypatch):
    """No resolvable slot at all must not mean "no credentials" — that kills a run outright.

    Ambiguity resolves toward carrying a spare, never toward removing one, and the escape is
    taken OPENLY: `fail_open` rides in the record so an auditor is not left reading a full
    credential list as if the slots had asked for it.
    """
    import ouroboros.provider_models as pm

    # Realistic case: SETTINGS_DEFAULTS fill the empty slots, so this still resolves narrowly.
    plan = provider_credential_plan({"OUROBOROS_MODEL": "", "CLAUDE_CODE_MODEL": ""})
    assert plan["fail_open"] is False and plan["planned_keys"]

    # Degenerate case: nothing resolvable at all (a ddgs web search needs no credential).
    monkeypatch.setattr(pm, "declared_model_settings", lambda _settings: {})
    degenerate = provider_credential_plan({"OUROBOROS_WEBSEARCH_BACKEND": "ddgs"})
    assert degenerate["fail_open"] is True
    assert degenerate["planned_keys"] == sorted(pm.ALL_PROVIDER_CREDENTIAL_KEYS)


# --------------------------------------------------------------------------- FIX B


# ------------------------------------------------------- FIX B: the vocabulary is DERIVED

# Every literal `reason_code` the runtime writes, and the DECISION for each: does an auditor
# risk mistaking it for a capability result? This table is the record the drift guard below
# enforces — a new runtime code with no row here fails the suite, which is the only thing
# that stops the vocabulary from being hand-copied beside the check again.
_TRUNCATION_DECISIONS: dict[str, tuple[bool, str]] = {
    "task_model_invalid": (False, "gateway/task_model.py: malformed model act refused before admission; no execution is truncated"),
    "presence_model_unsupported": (False, "owner_model.py: this control is unavailable for a Presence turn; the turn continues"),
    "not_a_root_task": (False, "owner_model.py: a delegated actor cannot move a root primary; no execution ends"),
    "model_control_unavailable": (False, "owner_model.py: no durable record for a primary act; execution is unchanged"),
    "cancel_pending": (False, "owner_model.py: a pending cancellation excludes this model act; the cancellation owns any terminal"),
    "model_action_state": (False, "owner_model.py: Return, Wait and Keep require a running fallback; refusing the act ends no work"),
    "model_recents_pending": (False, "owner_model.py: the primary was saved but its suggestion write needs retry; no work is truncated"),
    "model_queue_publish_failed": (False, "owner_model.py: primary saved, queue publication unconfirmed; not a task terminal"),
    "model_wake_failed": (False, "owner_model.py: primary saved, wake unconfirmed; the same request can repair delivery"),
    # Logs reads refuse selectors, sources or byte ranges; no task is stopped.
    "invalid_view": (False, "gateway/logs.py unknown view (400); not a task terminal"),
    "invalid_log_source": (False, "gateway/logs.py scope/projection refusal (400); not execution truncation"),
    "log_source_unavailable": (False, "gateway/logs.py unreadable or unadmitted source (404/503); not a task terminal"),
    "invalid_raw_request": (False, "gateway/logs.py invalid source selector/page bounds (400); not a task terminal"),
    "log_source_changed": (False, "gateway/logs.py exact stored-row identity conflict (409); not execution truncation"),
    "raw_range_invalid": (False, "gateway/logs.py out-of-range or split UTF-8 offset (416); not a task terminal"),
    "unknown_log": (False, "gateway/logs.py unsupported Raw stream (404); not a task terminal"),
    "request_invalid": (False, "gateway/skill_publish.py: unreadable publish-preflight body; HTTP 400 before any action, not a task terminal"),
    "secret_request_form_only": (False, "protected secret-form policy refuses an ordinary quiz answer; no execution or source is truncated"),
    "project_routing_fence_lookup_failed": (False, "gateway/tasks.py: Project authority unreadable before admission; HTTP 409, no running attempt truncated"),
    "input_source_selection_unsupported": (False, "HTTP 400 before root-task admission; no execution was truncated"),
    "task_source_invalid": (False, "gateway/task_archive.py: source selector basename differs from the requested artifact; HTTP 400, not a task terminal or trial truncation"),
    "artifact_archive_empty": (False, "gateway/task_archive.py: no eligible recorded directory member; HTTP refusal, not a task terminal"),
    "artifact_archive_invalid": (False, "gateway/task_archive.py: invalid selector; HTTP refusal, not a task terminal"),
    "artifact_archive_unavailable": (False, "gateway/task_archive.py: confined read or spool unavailable; HTTP refusal, not a task terminal"),
    "artifact_archive_unverified": (False, "gateway/task_archive.py: member drift or capture verification failure; HTTP refusal, not a task terminal"),
    "artifact_identity_changed": (False, "gateway/task_archive.py: a mutable file's bytes no longer match its recorded identity; HTTP 409, not a task terminal"),
    "artifact_name_ambiguous": (False, "gateway/tasks.py: ambiguous nested basename; HTTP refusal, not a task terminal"),
    "artifact_relpath_invalid": (False, "gateway/tasks.py: invalid exact artifact selector; HTTP refusal, not a task terminal"),
    "invalid_action_nonce": (False, "task_cancel rejects malformed action identity; no task transition"),
    "stop_action_conflict": (False, "task_cancel refuses conflicting action replay; no task transition"),
    "teardown_unconfirmed": (False, "gateway/tasks.py: an HTTP 202 Stop acknowledgement; the recorded intent is still being finished, never a task outcome"),
    "artifact_unavailable": (False, "gateway/task_archive.py: confined single-file read unavailable; HTTP refusal, not a task terminal"),
    "artifact_unverified": (False, "gateway/task_archive.py: single-file drift or capture verification failure; HTTP refusal, not a task terminal"),
    "wrap_up_document_invalid": (False, "gateway/task_archive.py: the Wrap up document selector combined with relpath or source; HTTP 400, not a task terminal"),
    "wrap_up_document_unavailable": (False, "gateway/task_archive.py: no Wrap up receipt by that name and Stop, or its on-read document could not be written; HTTP 404/503, not a task terminal"),
    "upload_unavailable": (False, "gateway/files.py: a chat attachment is missing or not a regular file; HTTP 404, not a task terminal"),
    "upload_unreadable": (False, "gateway/files.py: a chat attachment's confined read failed; HTTP 503, not a task terminal"),
    "history_source_unavailable": (False, "gateway/history_paging.py: readable recent projection with explicit source gap; no task attempt was truncated"),
    "review_history_unavailable": (False, "gateway/task_reviews.py: a task's saved review history could not be read; HTTP 503 on a read-only route, not a task terminal"),
    "review_source_invalid": (False, "gateway/task_reviews.py: the exact saved-review selector is malformed; HTTP 400, not a task terminal"),
    "review_source_changed": (False, "gateway/task_reviews.py: the exact selected saved review source is no longer named by its owning record; HTTP 409, not a task terminal"),
    "review_source_unavailable": (False, "gateway/task_reviews.py: saved review text failed its integrity check or cannot be read; HTTP 404, not a task terminal"),
    "quiz_delivery_failed": (False, "gateway/task_decision.py: the answer was recorded and its delivery stays owed (503, retry); never a task terminal or trial truncation"),
    "quiz_answer_unconfirmed": (False, "503 before any answer was recorded; not a task terminal or truncation"),
    "awaiting_recipient_read": (False, "question_answers.py: a recorded answer is in its living recipient's mailbox (paused or held too), read pending; a delivery fact on the question, never a task terminal"),
    "asker_not_reading": (False, "question_answers.py: no living asker or successor holds the answer, so it goes to named ingress in its room (the asker's live late-answer head, else a fresh head); a delivery fact, never a task terminal"),
    "promotion_not_started": (False, "promotion_intent.py: boot reconciliation of a start_task intent proven never admitted; that separate work never ran (explicit retry), so nothing was truncated and it is never a trial root's code"),
    "migrating": (False, "gateway/_helpers.py: a write door (create, Cancel, Pause, Presence) answered 503 inside the boot import window of state/current.sqlite (retry); no task attempt was truncated"),
    "boot_changed": (False, "gateway/control.py: a command whose if_boot names an earlier boot.since reached a restarted server; HTTP 409 with accepted=false before anything was queued, never a task terminal"),
    "budget_pausing_no_extraction": (False, "review_verdict_extraction.py: Light verdict extraction refused while the task's exact budget pause is closing dispatch (#1196); the review row stays undispatched and the attempt is paused, not truncated"),
    "owner_pause": (False, "tools/skill_publish.py: publication tool returns its completed stages after owner Pause interrupts the formatter; the task is retained for Resume, not terminalized or scored"),
    "action_not_found": (False, "gateway/card_decision.py: a press names a control the card does not have now; HTTP 404 on the card ingress, never a task terminal"),
    "card_action_once": (False, "gateway/card_decision.py: a once control already counted another press; HTTP 409 on the card ingress, never a task terminal"),
    "card_app_not_found": (False, "gateway/cards.py: a card app's stored bytes are missing; HTTP 404 on a read, never a task terminal"),
    "card_not_found": (False, "gateway/card_decision.py: a press or read names no stored card; HTTP 404, never a task terminal"),
    "card_stale": (False, "gateway/card_decision.py: the card changed since the press was drawn; HTTP 409, nothing recorded, never a task terminal"),
    "card_write_failed": (False, "gateway/card_decision.py: the press could not be recorded; HTTP 503 (retry), never a task terminal"),
    "chat_unavailable": (False, "gateway/ui_holds.py: no chat writer serves this data root, so a page's hold report named nothing; HTTP 503 (the next page reports again), never a task terminal"),
    "in_flight": (False, "gateway/ui_holds.py: another report is naming the same hold now; HTTP 503 (the next page reports again), never a task terminal"),
    "not_confirmed": (False, "gateway/ui_holds.py: a hold row's write was not confirmed; HTTP 503 (the next page reports again), never a task terminal"),
    "pin_write_failed": (False, "gateway/cards.py: Keep on Widgets could not be saved; HTTP 503, never a task terminal"),
    "revision_required": (False, "gateway/card_decision.py: a press without the card revision it was drawn from; HTTP 400, never a task terminal"),
    "values_invalid": (False, "gateway/card_decision.py: form values the card's form does not accept; HTTP 400, never a task terminal"),
    "github_actor_unavailable": (False, "skill_publish_github.py: the GitHub account behind the token cannot be read before any publication mutation; a typed publication refusal, not a task terminal"),
    "fork_prepare_failed": (False, "skill_publish_github.py: the account's fork could not be prepared; the publication result is partial and retryable, not a task terminal"),
    "fork_sync_failed": (False, "skill_publish_github.py: a 422-only fork synchronization failed; the publication result is partial and retryable, not a task terminal"),
    # -- truncating: the rail stopped the attempt, so reward 0 is not a capability fact ----
    "budget_exhausted": (True, "loop.py:287 per-task USD reservation rail"),
    "round_limit": (True, "loop.py:3128 _handle_round_limit, the round cap"),
    "finalization_grace": (True, "loop.py:3146 supervisor finalize_now grace"),
    "deadline_local": (True, "loop.py:3220 loop-local deadline"),
    "provider_unavailable": (True, "loop.py:3185 reroute + fallback exhausted"),
    "children_unabsorbed": (True, "historical forced terminal; retained for old benchmark records, no longer emitted"),
    "llm_api_error": (True, "loop_llm_call.py:630 transport death; never a fair shot"),
    "context_overflow": (True, "loop_model_call.py: known-window admission exhausted size recovery; host-stopped, not a capability result"),
    "context_capacity": (True, "agent_dispatch.py pooled start / context_renewal.py growth: the known window cannot hold the "
                               "task's context; host-stopped before or during execution, not a capability result"),
    "output_recovery_exhausted": (True, "loop_output_exhaustion.py: empty-answer recovery exhausted; incomplete provider outcome, not task completion"),
    # S3 owner graceful stop ("Wrap up"): the owner ended the attempt, so
    # reward 0 is an owner decision, never a fair-shot capability fact (CF-02:
    # reusing finalization_grace would persist the deadline's false reason).
    "owner_requested_finalization": (True, "loop.py _handle_owner_stop_finalization; owner-requested stop"),
    # -- not truncating: a real terminal the agent reached, or a rejected tool call --------
    # An explicit `executor=harness` request that could not be honored ends the CHILD
    # unrun rather than silently spending metered money on the native path. It is a
    # SUBAGENT dispatch terminal: a benchmark trial's reason code is the ROOT task's, and
    # a blocked child surfaces to its parent as an unabsorbed/failed child, under the
    # root's own code. Marking it truncating would require adding it to
    # RUNTIME_TRUNCATION_REASON_CODES and to the PUBLISHED CL-Bench operator patch that
    # mirrors that tuple — changing the provenance of an already-reported result to
    # describe a code that cannot appear in it.
    "subagent_executor_unavailable": (False, "agent.py executor_blocked_outcome; a subagent terminal, never a trial's"),
    # Q1A preflight (2026-08-10 amendments): an explicit harness pin whose child toolset
    # hides the delegate verbs ends the CHILD unrun before any LLM spend — the same
    # subagent-terminal class as subagent_executor_unavailable, never a trial's code.
    "delegate_tools_invisible": (
        False,
        "agent.py executor_blocked_outcome / preflight_delegate_visibility; a subagent terminal, never a trial's",
    ),
    "delegate_visibility_unverified": (
        False,
        "agent.py preflight_delegate_visibility broken-introspection path; a subagent terminal, never a trial's",
    ),
    "delegated_custody_unreconciled": (
        False,
        "stored reason of rows written before #1583 (the removed terminal custody overlay); new rows keep their own reason beside the delegated_runs_unreconciled debt list; never a truncation",
    ),
    "provider_outcome_unknown": (
        False,
        "tools/search.py recoverable ambiguous paid-call result returned to the LLM; the root task remains live",
    ),
    # tell_task: both are one-call tool results for the model's next turn, never the root task's terminal reason.
    "argument_invalid": (
        False,
        "tools/control_relay.py tell_task argument refusal (TOOL_ARG_ERROR, published as completed_no_effect before "
        "any write) returned to the LLM; the root task remains live",
    ),
    "arrival_effect_unknown": (
        False,
        "tools/control_relay.py tell_task arrival hand-over whose write may or may not have landed (TELL_UNCONFIRMED: "
        "retry the same arrival; place_intake's `arrival_effect_unknown:<task>` refusal is the same fact) returned to "
        "the LLM; the root task remains live",
    ),
    "finalize_control_pending": (
        False,
        "loop_transport.wait_transport_repeat llm_not_dispatched event; the never-sent repeat was withdrawn, while the later task terminal carries owner_requested_finalization or its actual rail reason",
    ),
    "host_cancelled": (
        False,
        "llm_claudexor.py check_control sends a model-operation cancel request; the POST proves neither operation terminality nor the root task's outcome",
    ),
    "model_wait_decision_failed": (
        False,
        "gateway/task_model_wait.py owner-action ingress refusal (503); it does not terminalize the waiting task",
    ),
    "deadline_exhausted": (
        False,
        "loop_llm_call.py owner deadline admission rail; no provider call was dispatched",
    ),
    "task_exception": (False, "agent.py:777 the attempt ran and crashed; an honest failure"),
    "capability_profile_mismatch": (False, "control_delegation.py:81 rejected delegate call"),
    "option_index_required": (False, "gateway/task_decision.py quiz-answer HTTP 400 validation; an owner-UI refusal, never a trial rail"),
    "action_invalid": (False, "gateway/task_decision.py rejects an unknown owner decision action (HTTP 400); no task transition"),
    "close_payload_invalid": (False, "gateway/task_decision.py rejects answer fields or a non-quiz target for closure (HTTP 400); no task transition"),
    "close_reason_invalid": (False, "gateway/task_decision.py refuses a closure reason that is not text or exceeds 2000 characters (HTTP 400); a blank one records the owner's standard sentence; the question remains unanswered, no task transition"),
    "quiz_close_failed": (False, "gateway/task_decision.py closure ingress failed (HTTP 503); question delivery may remain owed, but no task attempt was truncated"),
    "dismiss_payload_invalid": (False, "gateway/task_decision.py rejects answer fields or a non-quiz target for the owner's removal of an unknown delivery (HTTP 400); no task transition"),
    "quiz_dismiss_failed": (False, "gateway/task_decision.py removal ingress failed (HTTP 503); the answer and its unknown delivery stay as recorded, no task attempt was truncated"),
    "delegation_rights_may_delegate": (False, "control_delegation.py explicit parent recursion-right refusal"),
    "delegation_rights_may_fan_out": (False, "control_delegation.py explicit parent fan-out refusal"),
    "delegation_rights_depth_exhausted": (False, "control_delegation.py exhausted typed depth budget refusal"),
    "delegation_rights_max_children": (False, "control_delegation.py explicit direct-child budget cap refusal"),
    "delegation_rights_child_count_unknown": (False, "control_delegation.py unavailable direct-child authority scan"),
    "delegation_constraint_block_surface": (False, "control_delegation.py:116 rejected call"),
    "delegation_constraint_child_cap": (False, "control_delegation.py:149 rejected call"),
    "delegation_constraint_halt_fanout": (False, "control_delegation.py:103 rejected call"),
    "delegation_constraint_require_lane": (False, "control_delegation.py:126 rejected call"),
    "deep_self_review_unavailable": (False, "deep_self_review.py typed unavailable deep_review row/route on a review task; never a truncation"),
    "deep_self_review_error": (False, "deep_self_review.py review-stage error (and agent.py's exception branch) on a review task; never a truncation"),
    "worker_pool_unavailable": (False, "gateway/tasks.py managed-task admission refusal"),
    "worker_pool_state_unavailable": (False, "gateway/tasks.py fail-closed admission inspection"),
    "attachment_admission_rejected": (
        False,
        "gateway/_helpers.py pre-task attachment admission refusal; never a task terminal",
    ),
    "authority_source_unavailable": (
        False,
        "agent.py pre-context canonical-authority refusal; no substantive task attempt",
    ),
    # The same terminal under its own cause: a handbook the task's own contract requires (start_task alone authors
    # it) could not be read, so no model round ran and no execution was cut short.
    "required_reference_unavailable": (False, "context_input_selection.py pre-model source refusal; no task attempt ran"),
    # Phase-A AR2-1: the HTTP cancel ingress refuses (503) when the durable
    # cancel-intent write fails — an ingress refusal about a CANCEL request; the
    # task itself keeps running untouched, so no trial ever terminalizes with it.
    "cancel_intent_write_failed": (False, "gateway/tasks.py fail-closed cancel ingress refusal; never a task terminal"),
    # GR4-8: the corrupt-projection flavor of the same ingress refusal — still a
    # refusal about a CANCEL request (503), never a task terminal.
    "cancel_intent_projection_corrupt": (
        False,
        "gateway/tasks.py fail-closed cancel ingress refusal (corrupt projection); never a task terminal",
    ),
    # S3 hurry ingress (POST /api/tasks/<id>/hurry): all four are refusals about
    # a HURRY request — the task itself keeps running untouched, so no trial
    # ever terminalizes with any of them.
    "request_id_required": (False, "gateway/task_hurry.py hurry ingress refusal (400); never a task terminal"),
    "invalid_request_body": (
        False,
        "gateway/task_hurry.py ABI-3 derived-schema ingress refusal (400); never a task terminal",
    ),
    "unexpected_fields": (
        False,
        "gateway/task_hurry.py hurry ingress refusal (400, text-free contract); never a task terminal",
    ),
    "task_not_live": (False, "gateway/task_hurry.py hurry ingress refusal (404); never a task terminal"),
    "pause_reason_invalid": (
        False,
        "gateway/task_pause.py refuses a declared Pause reason that is not text or exceeds 400 characters (HTTP 400); "
        "nothing is written and no task transition happens",
    ),
    # #Q-2b decision-ingress refusal codes — HTTP replies, never task terminals.
    "unknown_decision_family": (False, "gateway/task_decision.py ingress refusal (400)"),
    "decision_family_not_served": (False, "gateway/task_decision.py ingress refusal (501)"),
    "malformed_decision_id": (False, "gateway/task_decision.py ingress refusal (400)"),
    "comment_invalid": (False, "gateway/task_decision.py ingress refusal (400)"),
    "comment_too_long": (
        False,
        "gateway/task_decision.py verbatim-comment refusal (400) — refuses instead of truncating",
    ),
    "option_index_invalid": (False, "gateway/task_decision.py ingress refusal (400)"),
    "quiz_history_write_failed": (
        False,
        "gateway/task_decision.py retryable owner-answer history append refusal (503); never a task terminal or trial truncation",
    ),
    "mailbox_write_failed": (
        False,
        "gateway/task_hurry.py fail-closed hurry ingress refusal (503); never a task terminal",
    ),
    # Request-wire reasons describe a physical candidate adjustment inside one
    # LLM call. They are emitted in usage.request_wire, not as the root task's
    # terminal reason_code, so none changes benchmark truncation provenance.
    "provider_metadata_constraint": (False, "request-wire candidate adjustment; never a task terminal"),
    "provider_prescribed_value": (False, "request-wire exact-value repair; never a task terminal"),
    "provider_recovery_succeeded": (False, "request-wire successful repair; never a task terminal"),
    "provider_refused_hint_value": (False, "request-wire hint withdrawal; never a task terminal"),
    "provider_rejected_tool_dialect": (False, "request-wire dialect fallback; never a task terminal"),
    "provider_required_reasoning": (False, "request-wire mandatory-reasoning floor; never a task terminal"),
    "provider_unsupported_field": (False, "request-wire unsupported-field repair; never a task terminal"),
    "requested_wire_form": (False, "request-wire initial physical form; never a task terminal"),
    "task_local_availability_fallback": (False, "request-wire task-local candidate; never a task terminal"),
    # Issue #265 selected publish preflight and publish tool: gateway values are
    # HTTP/domain facts produced before a managed task exists; shared tool values
    # are recoverable one-call failures returned to the LLM. None truncates a trial.
    "skill_invalid": (
        False,
        "gateway preflight refusal or recoverable tools/skill_publish.py call; never a task terminal",
    ),
    "github_token_missing": (
        False,
        "gateway preflight refusal or recoverable tools/skill_publish.py call; never a task terminal",
    ),
    "skill_not_found": (
        False,
        "gateway preflight refusal or recoverable tools/skill_publish.py call; never a task terminal",
    ),
    "skill_identity_ambiguous": (
        False,
        "gateway/skill_publish.py selected-preflight identity refusal; no task exists",
    ),
    "skill_source_unsupported": (
        False,
        "gateway preflight refusal or recoverable tools/skill_publish.py call; never a task terminal",
    ),
    "warnings_present": (
        False,
        "gateway/skill_publish.py successful read-only preflight fact; never a task terminal",
    ),
    "runtime_missing": (False, "Betterleaks runtime availability; no task finalization"),
    "scanner_report_invalid": (False, "Publication scanner repair evidence; no task finalization"),
    "scanner_input_invalid": (False, "skill_publish_scanner.py rejects invalid payload names, scope or timeout before scanning; recoverable publication refusal, no task finalization"),
    "task_admission_unavailable": (False, "Publication task was not admitted; no running task truncated"),
    # Issue #265: these are structured failures of one recoverable publish-tool
    # call. They return to the next LLM turn with a repair hint; none is the
    # managed task's terminal reason or evidence that a benchmark trial was cut
    # off before the agent could act.
    "upstream_read_failed": (
        False,
        "skill_publish_github.py recoverable read-only tool failure; never a task terminal",
    ),
    "branch_create_failed": (False, "skill_publish_github.py recoverable branch tool failure; never a task terminal"),
    "commit_create_failed": (False, "skill_publish_github.py recoverable commit tool failure; never a task terminal"),
    "pr_open_indeterminate": (
        False,
        "skill_publish_github.py typed ambiguous remote effect; task receipt veto remains separate",
    ),
    "unexpected_publish_error": (False, "tools/skill_publish.py recoverable typed tool failure; never a task terminal"),
}

_REASON_CODE_LITERAL = re.compile(r"""reason_code(?:=|["']\s*:\s*)\s*["']([a-z0-9_]+)["']""")


def _forwarded_reason_arguments(tree: ast.Module) -> dict[str, dict[str, int]]:
    """Local wrappers may pass a named parameter straight into reason_code."""
    forwarding = {}
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = function.args.posonlyargs + function.args.args
        forwarded = {
            node.value.id for node in ast.walk(function)
            if isinstance(node, ast.keyword) and node.arg == "reason_code"
            and isinstance(node.value, ast.Name)
        }
        arguments = {parameter.arg: index for index, parameter in enumerate(parameters)
                     if parameter.arg in forwarded}
        if arguments:
            forwarding[function.name] = arguments
    return forwarding


def _runtime_reason_code_literals() -> dict[str, str]:
    """Every literal reason code the runtime source assigns, with its first emitting line."""
    root = pathlib.Path(__file__).resolve().parents[1] / "ouroboros"
    found: dict[str, str] = {}

    def values(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.IfExp):
            yield from values(node.body)
            yield from values(node.orelse)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                yield from values(value)

    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(source.splitlines(), 1):
            for match in _REASON_CODE_LITERAL.finditer(line):
                found.setdefault(match.group(1), f"{path.relative_to(root.parent)}:{lineno}")
        # Conditional/fallback reason values remain producers; their spelling
        # need not be adjacent to the keyword on one source line.
        tree = parse(source)
        if path.name == "owner_model.py":
            # The endpoint forwards this typed exception's code, including the
            # eligibility function's returned refusal; those are wire reasons too.
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "ModelActRefused" and node.args):
                    for code in values(node.args[0]):
                        found.setdefault(code, f"{path.relative_to(root.parent)}:{node.lineno}")
            eligibility = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "eligibility")
            for node in ast.walk(eligibility):
                if isinstance(node, ast.Return):
                    for code in values(node.value):
                        if code:
                            found.setdefault(code, f"{path.relative_to(root.parent)}:{node.lineno}")
        forwarded = _forwarded_reason_arguments(tree)
        constants = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)}
        for function in tree.body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            forwarded_names = {node.value.id for node in ast.walk(function)
                               if isinstance(node, ast.keyword) and node.arg == "reason_code" and isinstance(node.value, ast.Name)}
            for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
                if argument.arg in forwarded_names:
                    value = constants.get(default.id) if isinstance(default, ast.Name) else default
                    for code in values(value):
                        if code:
                            found.setdefault(code, f"{path.relative_to(root.parent)}:{function.lineno}")
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "reason_code":
                for code in values(node.value):
                    if code:
                        found.setdefault(code, f"{path.relative_to(root.parent)}:{node.lineno}")
            # _answer_pending still emits an HTTP reason through _refused;
            # extracting that helper must not make its live codes look retired.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                for parameter, index in forwarded.get(node.func.id, {}).items():
                    argument = (node.args[index] if index < len(node.args) else
                                next((kw.value for kw in node.keywords if kw.arg == parameter), None))
                    for code in values(argument):
                        if code:
                            found.setdefault(code, f"{path.relative_to(root.parent)}:{node.lineno}")
    for code in WIRE_REASON_CODES:
        found.setdefault(code, "ouroboros/request_wire_contract.py:WIRE_REASON_CODES")
    return found


# Historical reason codes keep their recorded decision, truncating or not, after their producing rail is retired:
# old records still carry them.
_RETIRED_REASON_CODES = {"children_unabsorbed", "delegated_custody_unreconciled"}


def test_every_runtime_reason_code_has_a_recorded_truncation_decision():
    """Drift guard. A reason code added to the runtime tomorrow silently defaults to
    `truncated: false` — an affirmative claim — unless somebody DECIDED it should. This
    fails until the decision is written down, so the omission cannot pass as a judgement."""
    emitted = _runtime_reason_code_literals()
    undecided = sorted(set(emitted) - set(_TRUNCATION_DECISIONS))
    assert not undecided, "new runtime reason code(s) with no recorded decision in _TRUNCATION_DECISIONS: " + ", ".join(
        f"{code} ({emitted[code]})" for code in undecided
    )
    stale = sorted(set(_TRUNCATION_DECISIONS) - set(emitted) - _RETIRED_REASON_CODES)
    assert not stale, f"decision recorded for code(s) the runtime no longer emits: {stale}"
    forgotten = sorted(_RETIRED_REASON_CODES - set(_TRUNCATION_DECISIONS))
    assert not forgotten, f"retired code(s) whose old records lost their decision: {forgotten}"

    decided_truncating = {c for c, (yes, _why) in _TRUNCATION_DECISIONS.items() if yes}
    assert decided_truncating == set(RUNTIME_TRUNCATION_REASON_CODES)


def test_the_two_out_of_tree_mirrors_stay_pinned_to_the_derived_vocabulary():
    """Three hand-written copies of one vocabulary is how all three came to be wrong.

    The Harbor runner is a source template executed inside the task container, so it now
    INTERPOLATES the SSOT instead of restating it. The CL-Bench bridge genuinely cannot
    import it (that module lives in an external clone reached only through a call-time
    sys.path insert), so its mirror stays — but pinned here, because the comment that asked
    for it to be kept in sync demonstrably did not hold.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]

    harbor = (repo / "devtools/benchmarks/terminal_bench/harbor_installed_agent.py").read_text(encoding="utf-8")
    assert "truncated = reason_code in {truncation_codes_literal}" in harbor
    assert "truncation_codes_literal = repr(tuple(sorted(RUNTIME_TRUNCATION_REASON_CODES)))" in harbor
    assert "max_rounds_exceeded" not in harbor

    patch = (
        repo / "devtools/benchmarks/continual_learning/operator_patches/clb_multi_instance_outcomes.v6746.patch"
    ).read_text(encoding="utf-8")
    mirrored = re.search(r'"truncated": reason_code in \((.*?)\),', patch, re.S)
    assert mirrored, "the CL-Bench mirror's truncation expression is no longer parseable"
    assert set(re.findall(r'"([a-z0-9_]+)"', mirrored.group(1))) == set(RUNTIME_TRUNCATION_REASON_CODES), (
        "the CL-Bench operator patch has drifted from RUNTIME_TRUNCATION_REASON_CODES"
    )
