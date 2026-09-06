import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    OWNER_STOP_DETAIL_MARKER,
    summarizeChatLiveEvent,
    summarizeLogEvent,
    taskDoneIsTerminal,
    taskPresentation,
    taskTerminalPhase,
} from '../modules/log_events.js';
import {
    captureLiveCardPhaseState,
    desiredLiveCardPhase,
    replayTerminalPhase,
    restoreLiveCardPhaseState,
    setLiveCardPhase,
} from '../modules/task_phase_chip.js';

const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
const activitySource = readFileSync(new URL('../modules/chat_activity.js', import.meta.url), 'utf8');
const logEventsSource = readFileSync(new URL('../modules/log_events.js', import.meta.url), 'utf8');
const chatMediaSource = readFileSync(new URL('../modules/chat_media.js', import.meta.url), 'utf8');

const terminalCases = [
    ['clean Done', { status: 'completed' }, { phase: 'done', headline: 'Done' }],
    ['Done with warnings', {
        status: 'completed', outcome_axes: { execution: { status: 'degraded' } },
    }, { phase: 'warn', headline: 'Done with warnings' }],
    ['Failed', {
        status: 'failed', reason_code: 'delegated_custody_unreconciled',
    }, { phase: 'error', headline: 'Failed' }],
    ['Cancelled', { status: 'cancelled' }, { phase: 'cancelled', headline: 'Cancelled' }],
];

test('factual task presentation uses the approved five-word family', () => {
    assert.deepEqual(taskPresentation('working'), {
        phase: 'working', headline: 'Working',
    });
    for (const [name, payload, expected] of terminalCases) {
        assert.deepEqual(taskPresentation(taskTerminalPhase(payload)), expected, name);
    }
    // Incidental outcome/review/artifact facts cannot manufacture terminality;
    // callers must first supply an explicit phase from the lifecycle authority.
    assert.deepEqual(taskPresentation({
        status: 'running',
        outcome_axes: { review: { status: 'fail' } },
        artifact_status: 'missing',
    }), { phase: 'working', headline: 'Working' });
});

test('live task_done and replay/log task truth have phase and headline parity', () => {
    for (const [name, payload, expected] of terminalCases) {
        const evt = { type: 'task_done', ...payload };
        const live = summarizeChatLiveEvent(evt);
        const replay = summarizeLogEvent(evt);
        assert.deepEqual({ phase: live.phase, headline: live.headline }, expected, `${name}/live`);
        assert.deepEqual({ phase: replay.phase, headline: replay.headline }, expected, `${name}/replay`);
        assert.doesNotMatch(`${live.headline} ${replay.headline}`, /Issue|Notice|delegated_custody_unreconciled/);
        if (payload.reason_code) {
            assert.match(live.body, /Reason: delegated_custody_unreconciled/);
            assert.ok(replay.meta.includes('delegated_custody_unreconciled'));
        }
    }
    assert.match(
        chatSource,
        /const presentation = taskPresentation\(finalizing \? 'working' : taskTerminalPhase\(msg \|\| \{\}\)\);/,
    );
});

test('typed terminal status drives an error phase on live and replay cards', () => {
    const failed = { task_terminal_status: 'failed' };
    assert.equal(taskTerminalPhase(failed), 'error');
    assert.equal(taskDoneIsTerminal(failed), true);
    assert.match(
        chatSource,
        /finishLiveCard\(taskId, msg\.task_terminal_status \? taskTerminalPhase\(msg\) : replayTerminalPhase\(taskState, record\)\);/,
    );
    assert.match(chatSource, /finishLiveCard\(explicitTaskId, taskTerminalPhase\(msg\)\) \|\| changed;/);
});

test('interrupted task_done remains retryable and cannot finish a root card', () => {
    const evt = {
        type: 'task_done', status: 'interrupted',
        reason_code: 'worker_restart_interrupted',
        outcome_axes: { lifecycle: { status: 'interrupted' } },
    };
    const live = summarizeChatLiveEvent(evt);
    const replay = summarizeLogEvent(evt);
    assert.equal(taskDoneIsTerminal(evt), false);
    assert.deepEqual(
        { phase: live.phase, headline: live.headline, terminal: live.terminal },
        { phase: 'working', headline: 'Working', terminal: false },
    );
    assert.deepEqual({ phase: replay.phase, headline: replay.headline }, {
        phase: 'working', headline: 'Working',
    });
    assert.match(chatSource, /if \(eventType === 'task_done' && summary\.terminal\)/);
    assert.match(chatSource, /if \(!taskDoneIsTerminal\(terminalRecord\)\) continue;/);
    assert.equal(taskDoneIsTerminal({
        status: 'completed',
        root_phase_checkpoint: { post_task_synthesis: 'running' },
    }), false);
});

test('review lifecycle timeout and error are terminal lifecycle errors', () => {
    for (const status of ['timeout', 'error']) {
        const view = summarizeChatLiveEvent({
            type: 'send_message',
            is_progress: true,
            task_id: 'review-task',
            lifecycle: { kind: 'review', status, target: 'alpha', error: 'transport failed' },
        });
        assert.equal(view.phase, 'lifecycle_error', status);
        assert.equal(view.terminal, true, status);
    }
});

test('owner soft-stop is factual Done and keeps its marker in details', () => {
    const evt = {
        type: 'task_done', status: 'done', reason_code: 'owner_requested_finalization',
        outcome_axes: { execution: { status: 'best_effort' } },
    };
    const live = summarizeChatLiveEvent(evt);
    const replay = summarizeLogEvent(evt);
    assert.deepEqual({ phase: live.phase, headline: live.headline }, { phase: 'done', headline: 'Done' });
    assert.deepEqual({ phase: replay.phase, headline: replay.headline }, { phase: 'done', headline: 'Done' });
    assert.ok(live.meta.includes(OWNER_STOP_DETAIL_MARKER));
    assert.ok(replay.meta.includes(OWNER_STOP_DETAIL_MARKER));
    assert.doesNotMatch(live.headline, /owner_requested_finalization/);
});

test('failed child remains a compact local fact without owner-alarm semantics', () => {
    const child = summarizeChatLiveEvent({
        type: 'send_message', is_progress: true, delegation_role: 'subagent',
        parent_task_id: 'root-working', subagent_task_id: 'child-failed',
        subagent_role: 'researcher', subagent_event: 'failed', status: 'failed',
        error: 'daemon unreachable', reason_code: 'delegated_custody_unreconciled',
    });
    assert.equal(child.phase, 'error');
    assert.equal(child.terminal, true);
    // Identity only; the chip carries `Failed` (DESIGN.md §4), the headline never does.
    assert.equal(child.headline, 'researcher');
    assert.doesNotMatch(child.body, /delegated_custody_unreconciled/);
    assert.match(child.fullBody, /Reason: delegated_custody_unreconciled/);
    assert.equal('ownerAlarm' in child, false);
    assert.equal('notification' in child, false);
    const adapter = chatSource.slice(
        chatSource.indexOf('function routeSubagentTerminalToCard'),
        chatSource.indexOf('function updateLiveCardFromLogEvent'),
    );
    assert.match(adapter, /reason_code: evt\.reason_code \|\| ''/);
});

test('interrupted child stays retryable with a Working chip and inspectable detail', () => {
    const child = summarizeChatLiveEvent({
        type: 'send_message', is_progress: true, delegation_role: 'subagent',
        parent_task_id: 'root-working', subagent_task_id: 'child-interrupted',
        subagent_role: 'researcher', subagent_event: 'interrupted', status: 'interrupted',
        error: 'transport interrupted; retry remains available',
    });
    assert.equal(child.phase, 'warn');
    assert.equal(child.terminal, false);
    assert.equal(child.visible, true);
    assert.equal(child.headline, 'researcher');
    assert.match(child.fullBody, /transport interrupted; retry remains available/);
    assert.equal(taskPresentation(child.terminal ? child.phase : 'working').headline, 'Working');
    const applyState = chatSource.slice(
        chatSource.indexOf('function applyLiveCardStateMutation'),
        chatSource.indexOf('function finishLiveCard'),
    );
    assert.match(applyState, /const desiredPhase = desiredLiveCardPhase\(record, activePhase\);/);
    assert.match(applyState, /setLiveCardPhase\(record, desiredPhase\.phase, desiredPhase\.text, desiredPhase\.className\);/);
    assert.equal([...applyState.matchAll(/setLiveCardPhase\(/g)].length, 1);
});

test('desired phase-chip precedence keeps stop/finalizing state sticky', () => {
    assert.deepEqual(desiredLiveCardPhase({ finished: true }, 'error'), {
        phase: 'error', text: 'Failed', className: 'chat-live-phase error',
    });
    assert.deepEqual(desiredLiveCardPhase({
        finished: false, cancelPendingPolicy: 'finalize', finalizingHold: true,
    }, 'warn'), {
        phase: 'working', text: 'Finalizing…', className: 'chat-live-phase working cancelling',
    });
    assert.deepEqual(desiredLiveCardPhase({
        finished: false, cancelPendingPolicy: 'immediate', finalizingHold: true,
    }, 'warn'), {
        phase: 'working', text: 'Cancelling…', className: 'chat-live-phase working cancelling',
    });
    assert.deepEqual(desiredLiveCardPhase({ finished: false, finalizingHold: true }, 'warn'), {
        phase: 'working', text: 'Finalizing…', className: 'chat-live-phase working finalizing',
    });
    assert.deepEqual(desiredLiveCardPhase({ finished: false }, 'warn'), {
        phase: 'working', text: 'Working', className: 'chat-live-phase working',
    });
});

test('failed optimistic stop restores the finalizing fact, not only its DOM text', () => {
    const record = {
        finished: false,
        cancelPendingPolicy: '',
        finalizingHold: true,
        phaseEl: { dataset: { phase: 'working' } },
    };
    const snapshot = captureLiveCardPhaseState(record);
    record.cancelPendingPolicy = 'immediate';
    record.finalizingHold = false;
    const restored = restoreLiveCardPhaseState(record, snapshot);
    assert.equal(record.cancelPendingPolicy, '');
    assert.equal(record.finalizingHold, true);
    assert.deepEqual(restored, {
        phase: 'working', text: 'Finalizing…', className: 'chat-live-phase working finalizing',
    });
    // A previous cancel-pending visual is not resurrected after durable detail
    // proves that no cancel intent remains.
    const staleCancel = {
        finished: false,
        cancelPendingPolicy: 'finalize',
        finalizingHold: false,
        phaseEl: { dataset: { phase: 'working' } },
    };
    const staleSnapshot = captureLiveCardPhaseState(staleCancel);
    staleCancel.cancelPendingPolicy = 'immediate';
    assert.deepEqual(restoreLiveCardPhaseState(staleCancel, staleSnapshot), {
        phase: 'working', text: 'Working', className: 'chat-live-phase working',
    });
});

test('task-detail healing reuses the full terminal-summary projection', () => {
    const cancelHeal = chatSource.slice(
        chatSource.indexOf('function reconcileCancelCardFromDetail'),
        chatSource.indexOf('async function cancelRunFromCard'),
    );
    assert.match(cancelHeal, /taskDoneIsTerminal\(stored\)/);
    assert.match(cancelHeal, /appendTaskSummaryToLiveCard\(\{ \.\.\.stored, task_id: taskId \}\)/);
    assert.doesNotMatch(cancelHeal, /finishLiveCard\(/);

    const missingHeal = chatSource.slice(
        chatSource.indexOf('async function reconcileMissingManagedTask'),
        chatSource.indexOf('function observeMissingManagedTask'),
    );
    assert.match(missingHeal, /isTerminalTaskDetail\(detail\)/);
    assert.match(missingHeal, /appendTaskSummaryToLiveCard\(\{ \.\.\.detail, task_id: taskId \}\)/);
    assert.doesNotMatch(missingHeal, /finishLiveCard\(/);
});

test('history replay keeps open summaries live and terminal fallbacks factual', () => {
    const summary = chatSource.slice(
        chatSource.indexOf('function appendTaskSummaryToLiveCard'),
        chatSource.indexOf('// child task_id'),
    );
    assert.match(summary, /const finalizing = msg\?\.task_phase === 'finalizing' \|\| msg\?\.outcome_final === false;/);
    assert.match(summary, /terminal: !finalizing/);
    assert.match(summary, /record\.finalizingHold = true/);
    assert.match(summary, /if \(finalizing\) return changed;\s*changed = finishLiveCard/);

    assert.equal(replayTerminalPhase({}, { finished: false, phaseEl: {
        dataset: { phase: 'working' },
    } }), 'done');
    assert.equal(replayTerminalPhase({}, { finished: true, phaseEl: {
        dataset: { phase: 'error' },
    } }), 'error');
    assert.equal(replayTerminalPhase({ completedPhase: 'warn' }, {}), 'warn');
    assert.equal(
        [...chatSource.matchAll(/finishLiveCard\(taskId, msg\.task_terminal_status \? taskTerminalPhase\(msg\) : replayTerminalPhase\(taskState, record\)\);/g)].length,
        2,
    );
    assert.doesNotMatch(
        chatSource,
        /taskState\?\.completedPhase \|\| record\?\.phaseEl\?\.dataset\?\.phase \|\| 'done'/,
    );

    const wsSummary = chatSource.slice(
        chatSource.indexOf("if (msg.system_type === 'task_summary')"),
        chatSource.indexOf("if (explicitTaskId && subagentChildParents.has", chatSource.indexOf("if (msg.system_type === 'task_summary')")),
    );
    assert.match(wsSummary, /if \(!finalizing\) markAssistantReply\(explicitTaskId\);/);
});

test('phase chips are contextual polite status regions without repeat announcements', () => {
    const attrs = new Map();
    const attrWrites = new Map();
    let textContent = '';
    let textWrites = 0;
    const phaseEl = {
        dataset: {},
        className: '',
        get textContent() { return textContent; },
        set textContent(value) { textContent = value; textWrites += 1; },
        getAttribute(name) { return attrs.get(name) ?? null; },
        setAttribute(name, value) {
            attrs.set(name, value);
            attrWrites.set(name, (attrWrites.get(name) || 0) + 1);
        },
    };
    const record = { phaseEl, isSubagent: true };
    assert.equal(setLiveCardPhase(record, 'working'), true);
    assert.deepEqual({ ...phaseEl.dataset }, { phase: 'working' });
    assert.equal(phaseEl.className, 'chat-live-phase working');
    assert.equal(textContent, 'Working');
    assert.equal(attrs.get('role'), 'status');
    assert.equal(attrs.get('aria-live'), 'polite');
    assert.equal(attrs.get('aria-atomic'), 'true');
    assert.equal(attrs.get('aria-label'), 'Subagent status: Working');
    const stableTextWrites = textWrites;
    const stableLabelWrites = attrWrites.get('aria-label');
    assert.equal(setLiveCardPhase(record, 'working'), false);
    assert.equal(textWrites, stableTextWrites);
    assert.equal(attrWrites.get('aria-label'), stableLabelWrites);
    assert.equal(setLiveCardPhase(record, 'error', 'Failed'), true);
    assert.equal(textContent, 'Failed');
    assert.equal(attrs.get('aria-label'), 'Subagent status: Failed');
    assert.equal(textWrites, stableTextWrites + 1);
    assert.equal(attrWrites.get('aria-label'), stableLabelWrites + 1);
    const sticky = desiredLiveCardPhase({ cancelPendingPolicy: 'finalize' }, 'warn');
    assert.equal(setLiveCardPhase(record, sticky.phase, sticky.text, sticky.className), true);
    const stickyTextWrites = textWrites;
    const stickyLabelWrites = attrWrites.get('aria-label');
    assert.equal(setLiveCardPhase(record, sticky.phase, sticky.text, sticky.className), false);
    assert.equal(textContent, 'Finalizing…');
    assert.equal(phaseEl.className, 'chat-live-phase working cancelling');
    assert.equal(textWrites, stickyTextWrites);
    assert.equal(attrWrites.get('aria-label'), stickyLabelWrites);
    assert.match(chatSource, /aria-label="\$\{options\.isSubagent \? 'Subagent' : 'Task'\} status: Working"/);
    assert.doesNotMatch(chatSource, /record\.phaseEl\.(?:dataset\.phase|textContent|className)\s*=/);
});

test('nonterminal diagnostics stay visible facts but never promote the task', () => {
    const diagnostics = [
        summarizeChatLiveEvent({ type: 'llm_round_error', error: 'temporary provider error' }),
        summarizeChatLiveEvent({ type: 'tool_timeout', tool: 'delegate_wait' }),
        summarizeChatLiveEvent({ type: 'tool_call_finished', tool: 'run_command', is_error: true }),
        summarizeChatLiveEvent({ type: 'task_checkpoint', checkpoint_kind: 'context_fit_low_retry' }),
    ];
    for (const diagnostic of diagnostics) {
        assert.equal(diagnostic.visible, true);
        assert.equal(diagnostic.promote, false);
        assert.equal(diagnostic.terminal, false);
    }
    assert.doesNotMatch(chatSource, /showContextFitToast|context-fit:/);
    // The dedupe-keyed incident toast lives in chat_media.js (byte-ratchet
    // extraction) while chat.js's progress fan-out still invokes it.
    assert.match(chatMediaSource, /export function showTaskIncidentToast\(msg\)/);
    assert.match(chatSource, /showTaskIncidentToast\(msg\);/);
    const success = summarizeChatLiveEvent({ type: 'task_done', status: 'completed' });
    assert.deepEqual({ phase: success.phase, headline: success.headline }, { phase: 'done', headline: 'Done' });
    assert.match(chatSource, /const shouldPromote = Boolean\(summary\.promote\) \|\| record\.finished;/);
    assert.match(chatSource, /record\.updates > 1 \? record\.titleEl\.textContent : ''/);
    assert.match(chatSource, /\|\| 'Working\.\.\.'/);
    assert.doesNotMatch(chatSource, /record\.lastHumanHeadline \|\| headline/);
});

test('unknown keyword-shaped Chat event does not synthesize an alarm', () => {
    const unknown = summarizeChatLiveEvent({
        type: 'future_worker_crash_recovered', error: 'diagnostic payload',
    });
    assert.equal(unknown.visible, false);
    assert.equal(unknown.promote, false);
    assert.equal(unknown.terminal, false);
    assert.doesNotMatch(unknown.headline, /Issue|Attention/);
    const chatSummarizer = logEventsSource.slice(
        logEventsSource.indexOf('export function summarizeChatLiveEvent'),
        logEventsSource.indexOf('export function duplicateLogEventKey'),
    );
    assert.doesNotMatch(chatSummarizer, /t\.includes\('error'\)|t\.includes\('crash'\)|t\.includes\('fail'\)/);
});

test('header status has no terminal-attention state or writer', () => {
    assert.doesNotMatch(chatSource, /lastTerminalAttention/);
    assert.doesNotMatch(activitySource, /lastTerminalAttention|text: 'Attention'/);
});

test('a review-caused warning names the acceptance decision on the card and in Logs', () => {
    // The execution reason beside it ('final_message') names the delivery step,
    // not the cause; the card body and the Logs meta now say what happened.
    const evt = {
        type: 'task_done', status: 'completed', reason_code: 'final_message',
        outcome_axes: {
            execution: { status: 'ok' },
            review: {
                status: 'degraded',
                acceptance_decision: {
                    status: 'finalized_unaccepted',
                    rationale: 'Acceptance reviewers did not reach a valid quorum.',
                },
            },
        },
    };
    const live = summarizeChatLiveEvent(evt);
    const replay = summarizeLogEvent(evt);
    assert.deepEqual({ phase: live.phase, headline: live.headline }, { phase: 'warn', headline: 'Done with warnings' });
    assert.match(live.body, /Acceptance: finalized_unaccepted — Acceptance reviewers did not reach a valid quorum\./);
    assert.doesNotMatch(live.body, /final_message/);
    assert.ok(replay.meta.includes('review degraded'));
    assert.ok(replay.meta.includes('acceptance finalized_unaccepted'));
});
