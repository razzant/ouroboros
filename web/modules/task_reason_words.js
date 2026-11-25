/** Task-cause words and exact-code disclosures; no outcome or lifecycle inference. */
import { fmt, tr } from './i18n.js';

// One sentence per typed cause; records keep exact codes. Raw lookup never guesses; details-bearing views pair an
// unknown code with a neutral sentence. Python twin project_dialogue.TASK_CAUSE_PHRASES; outcome_phase_parity pins both.
export const TASK_CAUSE_PHRASES = {
    previous_revision_accepted: "The reviewers approved an earlier version of this answer; the current version was not re-reviewed.",
    admission_close_unconfirmed: "Reviewers approved this answer; the supervisor did not confirm that task admission was closed.",
    author_stop: "Ouroboros stopped with unfinished work.",
    review_outcome_received: "Ouroboros received the reviewers' outcome and finished on that.",
    author_finish: "Ouroboros delivered this answer without the reviewers' sign-off.",
    author_finish_pending: "Ouroboros delivered this answer before the reviewers answered; their verdict is kept when it arrives.",
    review_degraded: "The reviewers did not reach a verdict on this answer.",
    infra_failure: "The review could not run because of an infrastructure failure, so there is no verdict.",
    acceptance_preparation_failed: "Ouroboros could not assemble the evidence for this answer's review, so this preparation attempt dispatched no new reviewers; the work itself is kept.",
    dialogue_terminal: "The reviewers and Ouroboros could not agree, and both positions were kept.",
    improvement_capsule: "The reviewers asked for one more pass and Ouroboros was given their notes.",
    fence_reopen_failed: "The requested extra pass could not be started, so the answer stands as it was.",
    review_cycles_exhausted: "The task used up its review rounds before the answer was signed off.",
    open_obligations: "The answer was delivered with reviewer requests still open.",
    improvement_window_closed: "There was no room left for another pass, so the answer stands as it was.",
    capsule_spent: "The one allowed improvement pass was already used.",
    reviewer_fail_no_capsule: "A reviewer rejected the answer and suggested nothing to change.",
    no_actionable_changes: "The re-review was not clean and suggested nothing to change.",
    identical_acceptance_refused: "Nothing had changed since the last review, so the recorded verdict stands.",
    review_skipped_deadline_reserve: "There was not enough time left to review the answer.",
    delivery_binding_superseded: "The answer or its evidence changed, so the earlier review no longer covered it.",
    owner_followup: "A new message from you arrived, so the review was set aside for it.",
    evidence_refresh: "The work changed after the review was frozen, so it no longer covered the answer.",
    revision_unavailable_on_forced_rail: "The task had to stop, so the requested rework never happened.",
    owner_hurry: "You asked me to hurry, so no further review was started.",
    unspecified: "The answer was not signed off, and no cause was recorded.",
    acceptance_bypassed_budget_exhausted: "The task ran out of budget before the answer could be reviewed.",
    acceptance_bypassed_round_limit: "The task hit its round limit before the answer could be reviewed.",
    acceptance_bypassed_deadline: "The task ran out of time before the answer could be reviewed.",
    acceptance_bypassed_provider_unavailable: "The model provider was unavailable, so the answer was never reviewed.",
    acceptance_bypassed_context_overflow: "The task outgrew its context before the answer could be reviewed.",
    acceptance_bypassed_children_unabsorbed: "Some sub-tasks had not been folded in, so the answer was never reviewed.",
    plan_review_advisory: "The plan review was never closed; the work went on with what the reviewers said.",
    plan_review_awaiting: "Not every plan reviewer had answered when the task ended.",
    // The plan review's outcome CLASS at delivery (outcome_axes.execution.plan_review) picks one of these.
    plan_review_unanswered: "Only some of the plan reviewers answered; the work went on with their notes.",
    plan_review_none_answered: "None of the plan reviewers answered; the work went on without their notes.",
    plan_review_answered_open: "The plan reviewers answered, but the review was never closed; the work went on with their notes.",
    plan_review_quorum_unreachable: "Too few plan reviewers could answer, so the work was held.",
    plan_review_objections_open: "Objections to the plan were still open when this answer was delivered; the work went on with them disclosed.",
    plan_review_unreadable: "The plan record could not be read, so the state of the plan review is unknown.",
    host_child_status_suffix: "A child task had not settled when the answer was delivered",
    invalid_delivery_control_after_repair: "Ouroboros's final delivery instruction could not be read even after repair, so the answer stands as delivered.",
    budget_exhausted: "The task ran out of budget before it could finish cleanly",
    billing_authority_unreadable: "Ouroboros could not read this task's billing authority, so it admitted no further paid call; budget exhaustion was not established",
    billing_authority_refused: "This task's billing lineage was missing or not its own, so it admitted no further paid call; budget exhaustion was not established",
    control_authority_unreadable: "Ouroboros could not read a control this task needs before a call, so it admitted none; budget exhaustion was not established",
    // The other forced-finalization rails (outcomes.BEST_EFFORT_REASON_CODES / ACCEPTANCE_BYPASS_REASON_BY_RAIL keys).
    round_limit: "The task hit its round limit before it could finish cleanly",
    finalization_grace: "The task hit a time limit and had to wrap up before it could finish cleanly",
    deadline_local: "The task reached its deadline before it could finish cleanly",
    context_overflow: "The task outgrew its context before it could finish cleanly",
    context_capacity: "This model's window is too small for this work",
    required_reference_unavailable: "The handbook this task requires could not be read",
    output_recovery_exhausted: "Incomplete model replies exhausted recovery; saved work remains available",
    children_unabsorbed: "Some sub-task results were never folded in, so the task had to wrap up",
    // The supervisor's timeout rails (queue_timeouts.TIMEOUT_TERMINAL_REASONS): the reaper's task_done reason_code.
    absolute_ceiling: "The task reached its maximum running time",
    deadline: "The task reached its deadline",
    idle_timeout: "The task made no progress for too long",
    // The reason codes outcomes.derive_loop_outcome stamps from typed terminal facts.
    provider_failure: "The model provider failed to answer, so the task could not finish",
    empty_final_text: "The task ended without a final answer",
    deep_self_review_unavailable: "The deep self-review could not run",
    deep_self_review_error: "The deep self-review stopped on an error",
    // A helper the host could not start (agent_dispatch.executor_blocked_outcome): it never ran and spent nothing.
    subagent_executor_unavailable: "This subagent could not start because its route was unavailable; it spent nothing",
    delegate_tools_invisible: "This subagent could not start because the tools that run its session are hidden from it; it spent nothing",
    delegate_visibility_unverified: "This subagent could not start because its tools could not be checked; it spent nothing",
    // #869: the provider-death rail's terminal words (twin of project_dialogue.TASK_CAUSE_PHRASES).
    provider_unavailable: "The task could not get a usable model answer, so it could not finish",
    accounting_wait_expired: "Accounting access did not recover within this turn’s wait window",
    delivery_control_degraded: "Ouroboros's final delivery instruction could not be applied, so the answer stands as delivered.",
    authoring_handover_incomplete: "The saved result records an incomplete model handover.",
    // A helper set to run as a coding session: keys picked by configuredActorCauseKey, not reason codes.
    configured_actor_first_undetermined: "First coding attempt: outcome unknown.",
    configured_actor_first_failed: "First coding attempt failed.",
    configured_actor_first_cancelled: "First coding attempt cancelled.",
    configured_actor_first_interrupted: "First coding attempt interrupted.",
    configured_actor_first_succeeded: "First coding session finished.",
    configured_actor_first_succeeded_source_unverified: "First coding session ended; brief coverage unverified.",
    configured_actor_first_refused: "First coding attempt: start refused.",
    configured_actor_not_started: "First coding attempt did not start.",
    delegated_custody_unreconciled: "Some delegated work was never reconciled.",
    // The one non-reason-code key: the task-result FIELD terminal_plan_review_open, a standing limitation.
    terminal_plan_review_open: "The plan review was still open when this answer was delivered",
    child_results_deferred: "Some sub-task results were deferred instead of being folded into this answer",
    tool_failure: "A tool this task used failed and nothing recovered it",
    receipt_absent: "No recorded check or result review examined this result after its last change; this does not establish whether other checks ran.",
    verification_red_unreconciled: "A check did not pass and no later run recorded as the same check passed; this record does not say whether other checks passed after it.",
    children_undecided: "Some finished sub-task results carry no recorded decision on how this answer used them.",
    expected_output_ungrounded: "A retired check recorded no tool calls or structured answer; it does not assess the written result.",
    residual_tool_errors_without_review: "Tool errors remain in the record.",
    artifacts_unavailable: "Some result files are missing or could not be preserved.",
    partial_result: "The result is partial; remaining work is described in task details.",
    task_exception: "The task stopped on an internal error",
    // The rest of owner_continue.TECHNICAL_REASON_CODES: worker, reaper and recovery endings and a failed model call.
    llm_api_error: "A model call ended in an error, so the task could not finish",
    provider_rejected_tool_dialect: "The model provider rejected the tool format of this task's requests, so the task could not finish",
    execution_deadline: "The task's execution time limit ran out",
    reaper_wedged_worker_alive: "The task timed out and its worker could not be confirmed stopped",
    workers_unavailable: "The worker pool was unavailable, so the task was not scheduled",
    worker_pool_unavailable: "No worker was available to run the task",
    evolution_disabled_in_light_mode: "Evolution is disabled in light runtime mode",
    worker_crash_signal: "The worker process running the task crashed",
    worker_crash_retry_exhausted: "The worker process running the task kept crashing until its retries ran out",
    worker_crash_owner_wait: "The worker process died while the task was waiting; completed actions were not retried",
    worker_crash_budget_pausing: "The worker process died while a continuation was being saved; no exact continuation was available",
    orphaned_running_after_worker_restart: "The workers restarted while the task was running, and it recorded no ending",
    interrupted_retry_lost: "The task was interrupted, and its retry was lost when the workers restarted",
    // Recovery finalized a dead attempt from its registered answer; the Done card's delivery
    // disposition keys (the row's typed terminal_delivery), rendered by terminalDeliveryClause.
    finalization_incomplete: "The answer was authored, but the task's finalization did not complete, so its review evidence, artifacts and cost accounting are incomplete.",
    finalization_incomplete_delivered: "The answer was delivered. The task stopped before its review evidence, artifacts and cost accounting were stored.",
    terminal_delivery_exhausted: "Sending the final answer was not confirmed after the delivery attempts ran out",
    terminal_delivery_unknown: "Delivery of the final answer could not be confirmed because the delivery registry could not be read; the result is in task details.",
    // Presentation only; no producer stamps this as a reason code.
    reason_undescribed: "The recorded reason has no description yet.",
};

// The codes above of a helper the host could not start: its only result is the host's notice to its parent (codes,
// alternatives), so its card's collapsed line states the cause and the notice stays in Details.
// Warnings that are details for the owner and never raise the headline (9 Nov 2026): nothing checked the result,
// a red check no later run passed, finished children without a decision. Twin: outcomes.DETAIL_ONLY_WARNINGS.
export const DETAIL_ONLY_WARNINGS = ['receipt_absent', 'verification_red_unreconciled', 'children_undecided'];
export const UNSTARTED_REASONS = new Set(['subagent_executor_unavailable', 'delegate_tools_invisible', 'delegate_visibility_unverified']);

// Twin of working_checkpoint.delivered_answer_cause: the model's own final answer, proven delivered before its
// task's finalization was cut, reads Done with this cause; '' for every other record.
export function deliveredAnswerCause(record, reason = 'finalization_incomplete') {
    return reason === 'finalization_incomplete' && record?.reason_code === reason && record.terminal_origin === 'model_final'
        && record.finalization_recovery?.answer === 'delivered' ? 'finalization_incomplete_delivered' : '';
}

// Twin of tool_error_words.first_attempt_cause_key: the sentence key the stored actor_first_terminal PROVES about the
// FIRST coding attempt, else ''. The typed witness is the single source — no producer status×reason matrix is
// re-derived here (a duplicated list reads every new producer fact as unknown). Closed vocabularies match with
// includes(), never a property lookup, so an inherited key ('constructor') is no state; strict types, never coerced.
// Witnesses: an initial_execution.state out of FIRST_ATTEMPT_STATES with its own run id (a refused start:
// 'failed_definite' without one; a settled success whose work-order coverage was never confirmed carries
// 'succeeded_source_unverified' and proves only the ending), or a typed zero-run receipt — the decision-as-status spelling,
// or a validated historical 'complete' receipt that still proves no physical start while the fact stays unknown.
const FIRST_ATTEMPT_STATES = ['failed', 'cancelled', 'interrupted', 'succeeded', 'succeeded_source_unverified'];
export function firstAttemptCauseKey(execution) {
    const fact = execution?.actor_first_terminal;
    if (!fact || typeof fact !== 'object' || !['incomplete', 'unknown'].includes(fact.status) || typeof fact.reason !== 'string') return '';
    if (fact.zero_run === true) {
        if (fact.zero_run_decision === fact.status && fact.reason === `configured_actor_zero_run_${fact.status}`) return 'configured_actor_not_started';
        return fact.status === 'unknown' && fact.zero_run_decision === 'complete' && fact.reason === 'historical_zero_run_complete'
            ? 'configured_actor_not_started' : '';
    }
    const initial = fact.initial_execution;
    if (!initial || typeof initial !== 'object') return '';
    const { state, run_id: runId } = initial;
    if (FIRST_ATTEMPT_STATES.includes(state)) {
        return typeof runId === 'string' && runId.trim() !== '' ? `configured_actor_first_${state}` : '';
    }
    const noRun = runId === undefined || runId === null || (typeof runId === 'string' && runId.trim() === '');
    return state === 'failed_definite' && noRun ? 'configured_actor_first_refused' : '';
}

// Twin of tool_error_words.configured_actor_cause_key: the reason codes stay as recorded; the words come from the
// proven first-attempt fact only when the fact's own status matches the code, else the conservative sentence.
export function configuredActorCauseKey(reason, execution) {
    if (typeof reason !== 'string') return '';
    if (reason !== 'configured_actor_incomplete' && reason !== 'configured_actor_unknown') return reason;
    const fact = execution?.actor_first_terminal;
    const undetermined = 'configured_actor_first_undetermined';
    if (!fact || typeof fact !== 'object' || fact.status !== reason.slice('configured_actor_'.length)) return undetermined;
    return firstAttemptCauseKey(execution) || undetermined;
}

// Twin of tool_error_words.first_attempt_clause_key: the proven key to state BESIDE a different primary reason.
export function firstAttemptClauseKey(reason, execution) {
    return typeof reason !== 'string' || reason === 'configured_actor_incomplete' || reason === 'configured_actor_unknown'
        ? '' : firstAttemptCauseKey(execution);
}

export function taskReasonPhrase(code) {
    const raw = String(code || '');
    // English is the source; an install language reads the sentence by its stable code (i18n.js).
    return TASK_CAUSE_PHRASES[raw] ? tr(`task.cause.${raw}`, TASK_CAUSE_PHRASES[raw]) : raw;
}

// The steps that most often stay failed at an ending; any other reads "A tool step". Python twin:
// ouroboros/tool_error_words.py (outcome_phase_parity pins both).
const TOOL_NOUNS = {
    run_command: 'A command',
    run_script: 'A script',
    read_file: 'A file read',
    list_files: 'A folder listing',
    query_data: 'A data query',
    github_read: 'A GitHub read',
    query_code: 'A code search',
    search_code: 'A code search',
    pr_merge: 'A pull request merge',
    review_change: 'A review request',
    finish_task: 'A finish request',
    // The effects the main view names once command failures stay in the details (owner 09.11, A).
    write_file: 'A file write',
    edit_text: 'A file edit',
    edit_batch: 'A file edit',
    apply_patch: 'A file edit',
    github_write: 'A GitHub write',
    send_user_message: 'A message send',
    place_reply: 'A reply',
    schedule_subagent: 'A subagent start',
    commit_reviewed: 'A commit',
    vcs_commit_reviewed: 'A commit',
    skill_exec: 'A skill run',
};
const LISTED = 20;  // outcomes keeps at most this many errors per list
const HELD_STATUSES = ['safety_returned', 'safety_unavailable'];  // twin: _outcome_tool_errors.SAFETY_HELD_STATUSES
const toolNoun = (tool) => tr(`task.tool_noun.${Object.hasOwn(TOOL_NOUNS, tool) ? tool : 'other'}`, TOOL_NOUNS[tool] || 'A tool step');

/** What happened to a call Safety returned or could not read, in the words of its step row and of the open Done card
 *  alike, or '' for any other status. */
export function heldCallPhrase(tool, status) {
    if (!HELD_STATUSES.includes(status)) return '';
    const noun = toolNoun(tool);
    return status === 'safety_returned' ? fmt('{noun} was returned by the Safety Supervisor', { noun })
        : fmt('{noun} was not run: the Safety Supervisor could not read it', { noun });
}

// What ended a call before it returned, by the word its row names (tools.jsonl ``control_reason``, written by
// ``tool_call_log.host_settlement``); any other word reads "Interrupted".
const INTERRUPTED_BY = {
    pause: 'Interrupted by Pause', stop: 'Interrupted by Stop', wrap_up: 'Interrupted by Wrap up',
    time_limit: 'Interrupted: time limit', spend_limit: 'Interrupted: spend limit',
    task_ended: 'Interrupted: the task ended', call_end: 'Interrupted: the call ended', other: 'Interrupted',
};
const CONTROL_KIND = {
    owner_pause: 'pause', pausing: 'pause', owner_pause_before_review_launch: 'pause', cancelled: 'stop',
    owner_stopped_direct_chat_turn: 'stop', finalize_requested: 'wrap_up', deadline: 'time_limit',
    execution_deadline: 'time_limit', absolute_ceiling: 'time_limit', spend_limit_global: 'spend_limit',
    spend_limit_root: 'spend_limit', task_ended: 'task_ended', "the call's end": 'call_end',  // live_tool_lane.CALL_END
};
const interruptedBy = (reason) => {
    const kind = Object.hasOwn(CONTROL_KIND, reason) ? CONTROL_KIND[reason] : 'other';
    return tr(`task.interrupted.${kind}`, INTERRUPTED_BY[kind]);
};

// A call the host refused a further step (``extension_dispatch``: ``tool_result_meta.refused_step``), by the reason the
// host gave (``extension_plugin_api._STOP_WORDS``). The host attests its refusal, never that the extension started
// nothing more; what ran before stays the extension's report.
const REFUSED_STEP = {
    cancelled: 'A further step was refused by Stop',
    task_cancelled: 'A further step was refused: the task had already ended as cancelled',
    operation_already_returned: 'A further step was refused: the call had already returned',
    stop_unreadable: "A further step was refused: the task's Stop could not be read", other: 'A further step was refused',
};

/** The headline of a call the host refused a further step, or '' for any other call. */
export function refusedStepPhrase(evt = {}) {
    if (evt.tool_result_meta?.refused_step !== true) return '';
    const reason = String(evt.tool_result_meta.control_reason || '');
    const kind = Object.hasOwn(REFUSED_STEP, reason) ? reason : 'other';
    return `${tr(`task.refused_step.${kind}`, REFUSED_STEP[kind])}${evt.tool ? ` · ${evt.tool}` : ''}`;
}

/** A non-error completion refused further work by Stop reads neutral (DESIGN §4 "Status marks").
 *  A failed call keeps its error; an unread Stop or other refusal alone stays a warning. */
export function refusedByStop(evt = {}) {
    return evt.status === 'blocked' && evt.tool_result_meta?.refused_step === true
        && ['cancelled', 'task_cancelled'].includes(evt.tool_result_meta.control_reason);
}

/** These two producer buckets are handled refusals; neither increments the call-error count. */
export function handledRefusalHeadline(evt) {
    if (evt.is_error) return '';
    if (evt.status === 'git_error') return tr('task.tools.git_refused', 'Git refused the operation');
    if (evt.status === 'review_blocked') return tr('task.tools.review_blocked', 'Review blocked the operation');
    return '';
}

/** The headline of a finished step that did not succeed. */
export function failedStepHeadline(evt = {}) {
    const tool = evt.tool ? ` · ${evt.tool}` : '';
    if (evt.status === 'interrupted') return `${interruptedBy(evt.control_reason)}${tool}`;
    return (evt.status === 'blocked' && refusedStepPhrase(evt)) || handledRefusalHeadline(evt) || heldCallPhrase(evt.tool, evt.status) || (evt.status === 'non_zero_exit'
        ? `A command returned ${Number.isInteger(evt.exit_code) ? `exit code ${evt.exit_code}` : 'a non-zero exit code'}`
        : `One of the steps failed${tool}`);
}

// The first error of both lists a view names (unresolved first), how many more, and whether a capped list is named
// through and through: the loud ones, or with `held` the calls Safety held; never a failed read or details-only leftover.
function namedToolErrors(execution, held) {
    const list = (key) => (Array.isArray(execution?.[key]) ? execution[key].filter((item) => item && typeof item === 'object') : []);
    const listed = [list('unresolved_tool_errors'), list('cosmetic_tool_errors')];
    const named = listed.map((errors) => errors.filter((item) => !(item.read_only || item.details_only)
        && HELD_STATUSES.includes(item.status) === held));
    // "More" only when a capped list is named through and through; otherwise the retained ones are counted exactly.
    const capped = listed.some((errors, index) => errors.length >= LISTED && named[index].length === errors.length);
    return { first: (named[0].length ? named[0] : named[1])[0], more: named[0].length + named[1].length - 1, capped };
}

/** The residual-warning clause naming the first recorded LOUD tool error in plain words (never a tool name, its
 *  result text or a path), or '' when the record names none. A failed read (`read_only`, owner 06.11 A) and a
 *  details-only leftover (`details_only`, owner 09.11 A: a command that ran and failed, a call refused before running)
 *  stay in the details, a held call in the open card (`heldCallsLine`, owner 11.11 A). Twin of
 *  tool_error_words.residual_tool_errors_phrase / _outcome_tool_errors.warns_without_review. */
export function residualToolErrorsPhrase(execution = {}) {
    const { first, more, capped } = namedToolErrors(execution, false);
    if (!first) return '';
    const noun = toolNoun(first.tool);
    const status = String(first.status || ''), code = Number.isInteger(first.exit_code) ? first.exit_code : null;
    const signal = first.signal ?? '';
    const subject = status === 'unknown_tool' ? tr('task.tool_error.unknown_tool', 'A call named a tool that does not exist')
        : status === 'non_zero_exit' && signal !== '' ? fmt('{noun} was stopped by signal {signal}', { noun, signal })
            : status === 'non_zero_exit' ? (code !== null ? fmt('{noun} returned exit code {code}', { noun, code })
                : fmt('{noun} returned a non-zero exit code', { noun }))
                : status.endsWith('timeout') ? fmt('{noun} timed out', { noun })
                    : status === 'argument_error' ? fmt('{noun} was called with invalid arguments', { noun })
                        : fmt('{noun} failed', { noun });
    const tail = capped ? tr('task.tool_error.more_capped', ', plus more tool errors')
        : more === 1 ? tr('task.tool_error.more_one', ', plus 1 more tool error')
            : more ? fmt(', plus {n} more tool errors', { n: more }) : '';
    return fmt('{subject}{tail}.', { subject, tail });
}

/** The open Done card's own line for the calls Safety returned or could not read that nothing replaced, counted as
 *  the loud clause counts, or ''. No collapsed line, notification or host sentence names them (owner 11.11, A). */
export function heldCallsLine(execution = {}) {
    const { first, more, capped } = namedToolErrors(execution, true);
    if (!first) return '';
    const tail = capped ? tr('task.tool_error.held_more_capped', ', plus more held by Safety')
        : more === 1 ? tr('task.tool_error.held_more_one', ', plus 1 more held by Safety')
            : more ? fmt(', plus {n} more held by Safety', { n: more }) : '';
    return fmt('{subject}{tail}.', { subject: heldCallPhrase(first.tool, first.status), tail });
}

// Details retain unknown codes once, in order; their visible cause never guesses. Twin: project_dialogue._cause_phrase.
export function describedCause(code, undescribed) {
    const raw = String(code || '');
    if (!raw || Object.hasOwn(TASK_CAUSE_PHRASES, raw)) return taskReasonPhrase(raw);
    if (!undescribed.includes(raw)) undescribed.push(raw);
    return taskReasonPhrase('reason_undescribed');
}

/** Exact undescribed codes, for the cause line's accessible details. */
export function reasonCodesNote(codes = []) {
    return codes.length ? fmt(codes.length === 1 ? 'Recorded reason: {code}' : 'Recorded reasons: {code}', { code: codes.join(', ') }) : '';
}

