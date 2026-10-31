import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { clearStickyCardState } from '../modules/chat_activity.js';

import {
    OWNER_STOP_DETAIL_MARKER,
    summarizeChatLiveEvent,
    summarizeLogEvent,
    taskDoneIsTerminal,
    taskPresentation,
    taskTerminalPhase,
    taskTerminalSummary,
} from '../modules/log_events.js';
import {
    captureLiveCardPhaseState,
    desiredLiveCardPhase,
    replayTerminalPhase,
    restoreLiveCardPhaseState,
    setInertCardPresentation,
    setLiveCardPhase,
    syncParkedPhase,
    taskActivityPresentation,
} from '../modules/task_phase_chip.js';
import { ElementStub, TextStub } from './chat_dom_fixture.js';

// A chip as the page has it: the shared status mark first, its words one text node beside it.
const chipDoc = { createElement: (tag) => new ElementStub(tag, chipDoc), createTextNode: (data) => new TextStub(data) };
const chipElement = () => new ElementStub('span', chipDoc);
const markOf = (chip) => {
    const mark = chip.firstElementChild;
    return mark && !mark.hidden ? `${mark.dataset.shape}/${mark.dataset.hue}` : 'none';
};
// The look each chip view carries (docs/DESIGN.md "Status marks"): the live word, or an outcome in its tone.
const LIVE = { mark: 'moving', tone: 'live', owner: false };
const OUTCOME = (outcome, tone) => ({ mark: 'none', tone, owner: false, outcome });

const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
const activitySource = readFileSync(new URL('../modules/chat_activity.js', import.meta.url), 'utf8');
const logEventsSource = readFileSync(new URL('../modules/log_events.js', import.meta.url), 'utf8');
const chatMediaSource = readFileSync(new URL('../modules/chat_media.js', import.meta.url), 'utf8');

const terminalCases = [
    ['clean Done', { status: 'completed' }, { phase: 'done', headline: 'Done' }],
    ['Done with warnings', {
        status: 'completed', outcome_axes: { execution: { status: 'degraded' } },
    }, { phase: 'warn', headline: 'Done with warnings' }],
    // The debt list rides along: the host twin states this cause only while the
    // row still owes it, so a fixture without it would describe a record the
    // durable writers no longer render this way.
    ['Failed', {
        status: 'failed', reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: ['run-a1'],
    }, { phase: 'error', headline: 'Failed' }],
    ['Stopped', { status: 'cancelled' }, { phase: 'cancelled', headline: 'Stopped' }],
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
            // The card body says the cause in words; the raw code stays in the
            // record half (Logs meta), which is where a machine code belongs.
            assert.match(live.body, /Some delegated work was never reconciled\./);
            assert.doesNotMatch(live.body, /delegated_custody_unreconciled/);
            assert.ok(replay.meta.includes('delegated_custody_unreconciled'));
        }
    }
    assert.match(chatSource, /const summary = taskTerminalSummary\(\{ \.\.\.msg, task_id: taskId \}\)/);
    const finalizing = taskTerminalSummary({ status: 'completed', task_phase: 'finalizing' });
    assert.deepEqual({ phase: finalizing.phase, terminal: finalizing.terminal }, { phase: 'working', terminal: false });
});

test('typed terminal status drives an error phase on live and replay cards', () => {
    const failed = { task_terminal_status: 'failed' };
    assert.equal(taskTerminalPhase(failed), 'error');
    assert.equal(taskDoneIsTerminal(failed), true);
    assert.match(
        chatSource,
        /finishLiveCard\(taskId, msg\.task_terminal_status \? taskTerminalPhase\(msg\) : replayTerminalPhase\(liveCardRecords\.get\(taskId\)\)\);/,
    );
    assert.match(chatSource, /appendTaskSummaryToLiveCard\(msg\) \|\| changed;/);
});

test('canonical replay status wins over a stale row without erasing open post-work', () => {
    const record = { status: 'running', task_terminal_status: 'failed' };
    assert.equal(taskDoneIsTerminal(record), true);
    assert.equal(taskTerminalPhase(record), 'error');
    const waiting = { ...record, root_phase_checkpoint: { post_task_synthesis: 'running' } };
    assert.equal(taskDoneIsTerminal(waiting), false);
    assert.equal(taskTerminalPhase(waiting), 'error');
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
    assert.match(chatSource, /if \(!taskDoneIsTerminal\(terminalRecord\)\) return;/);
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
    assert.ok(live.body.includes(OWNER_STOP_DETAIL_MARKER));
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
    assert.match(child.fullBody, /Some delegated work was never reconciled\./);
    assert.doesNotMatch(child.fullBody, /Reason: /, 'no machine label in front of the owner sentence');
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
    // #1110: an unfinished card's chip comes from the record's own state, so the
    // terminal phase is passed only when the card is finished.
    assert.match(applyState, /desiredLiveCardPhase\(record, closedPhase \|\| \(record\.finished \? summary\.phase \|\| 'done' : ''\)\)/);
    assert.match(applyState, /setLiveCardPhase\(record, desiredPhase\.phase, desiredPhase\.text, desiredPhase\.className,\s*desiredPhase\.secondary\)/);
    assert.equal([...applyState.matchAll(/setLiveCardPhase\(/g)].length, 1);
    // The Failed-into-the-title workaround is gone: the name stays stable.
    assert.doesNotMatch(chatSource, /failedHeadline/);
});

test('desired phase-chip precedence keeps stop/finalizing state sticky', () => {
    assert.deepEqual(desiredLiveCardPhase({ finished: true }, 'error'), {
        phase: 'error', text: 'Failed', className: 'chat-live-phase error', ...OUTCOME('error', 'failed'),
    });
    assert.deepEqual(desiredLiveCardPhase({
        finished: false, cancelPendingPolicy: 'finalize', finalizingHold: true,
    }, 'warn'), {
        phase: 'working', text: 'Finalizing…', className: 'chat-live-phase working cancelling', ...LIVE,
    });
    assert.deepEqual(desiredLiveCardPhase({
        finished: false, cancelPendingPolicy: 'immediate', finalizingHold: true,
    }, 'warn'), {
        phase: 'working', text: 'Cancelling…', className: 'chat-live-phase working cancelling', ...LIVE,
    });
    assert.deepEqual(desiredLiveCardPhase({ finished: false, finalizingHold: true }, 'warn'), {
        phase: 'working', text: 'Finalizing…', className: 'chat-live-phase working finalizing', ...LIVE,
    });
    assert.deepEqual(desiredLiveCardPhase({ finished: false }, 'warn'), {
        phase: 'working', text: 'Working', className: 'chat-live-phase working', ...LIVE,
    });
    // Stopped is someone's decision and Done with warnings succeeds: neither is amber.
    assert.deepEqual(desiredLiveCardPhase({ finished: true }, 'cancelled'), {
        phase: 'cancelled', text: 'Stopped', className: 'chat-live-phase cancelled', ...OUTCOME('cancelled', 'neutral'),
    });
    assert.deepEqual(desiredLiveCardPhase({ finished: true }, 'warn'), {
        phase: 'warn', text: 'Done with warnings', className: 'chat-live-phase warn', ...OUTCOME('warn', 'ok'),
    });
});

test('#1110 an observed outcome owns the chip while Finalizing states itself beside it', () => {
    // The outcome is known (the lifecycle status settled) while post-task work
    // still runs: the card says Failed and adds the hold as a SECOND fact, so
    // the title never has to carry the failure.
    assert.deepEqual(desiredLiveCardPhase({
        finished: false, finalizingHold: true, observedOutcome: 'error',
    }, 'working'), {
        phase: 'error', text: 'Failed', className: 'chat-live-phase error', ...OUTCOME('error', 'failed'),
        secondary: 'Finalizing…',
    });
    // An owner stop outranks the hold entirely — no secondary, no outcome chip.
    assert.deepEqual(desiredLiveCardPhase({
        finished: false, cancelPendingPolicy: 'immediate', finalizingHold: true,
        observedOutcome: 'error',
    }, 'working'), {
        phase: 'working', text: 'Cancelling…', className: 'chat-live-phase working cancelling', ...LIVE,
    });
    // A hold with no observed outcome keeps the plain Finalizing chip.
    assert.equal('secondary' in desiredLiveCardPhase({
        finished: false, finalizingHold: true,
    }, 'working'), false);

    // The observed outcome comes from a task-scope frame whose own lifecycle
    // status settled — NEVER from a failed tool call, which is diagnostics.
    assert.equal(taskTerminalSummary({
        status: 'failed', task_phase: 'finalizing',
    }).observedOutcome, 'error');
    assert.equal(taskTerminalSummary({ status: 'failed', task_phase: 'finalizing' }).terminal, false);
    // R1 (#1110): a finalizing REPLAY row often carries no stamp — history merges
    // the result's canonical axes, not `task_terminal_status`. The settled
    // lifecycle inside those axes is the same knowledge: the Failed is painted,
    // the hold stays secondary. A still-running lifecycle stays unobserved.
    const unstampedFailed = taskTerminalSummary({
        task_phase: 'finalizing', outcome_final: false, outcome_phase: 'error',
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed', reason_code: 'provider_unavailable' } },
    });
    assert.equal(unstampedFailed.observedOutcome, 'error');
    assert.equal(unstampedFailed.terminal, false);
    assert.equal(taskTerminalSummary({
        task_phase: 'finalizing', outcome_axes: { lifecycle: { status: 'running' } },
    }).observedOutcome, undefined);
    // While unfinished the frame itself paints Working: the chip is the record's.
    assert.equal(taskTerminalSummary({ status: 'failed', task_phase: 'finalizing' }).phase, 'working');
    assert.equal(taskTerminalSummary({ status: 'running' }).observedOutcome, undefined);
    assert.equal(summarizeChatLiveEvent({
        type: 'tool_call_finished', task_id: 't1', is_error: true, tool: 'bash',
    }).observedOutcome, undefined);

    // Recycling a card slot drops the outcome with the rest of the cycle state.
    const recycled = { observedOutcome: 'error', finalizingHold: true, censusPhase: 'thinking' };
    clearStickyCardState(recycled);
    assert.equal(recycled.observedOutcome, '');
    assert.equal(recycled.censusPhase, '');
    assert.equal(desiredLiveCardPhase(recycled).text, 'Working');
    assert.match(chatSource, /record\.observedOutcome = summary\.observedOutcome;/);
});

test('#1110 the secondary chip is a separate element with its own accessible name', () => {
    const record = {
        isSubagent: false,
        phaseEl: chipElement(),
        phaseSecondaryEl: { textContent: '', hidden: true, isConnected: true },
    };
    setLiveCardPhase(record, 'error', 'Failed', 'chat-live-phase error', 'Finalizing…');
    assert.equal(record.phaseEl.textContent, 'Failed');
    assert.equal(record.phaseSecondaryEl.textContent, 'Finalizing…');
    assert.equal(record.phaseSecondaryEl.hidden, false);
    assert.equal(record.phaseEl.getAttribute('aria-label'), 'Task status: Failed, Finalizing…');
    setLiveCardPhase(record, 'error', 'Failed', 'chat-live-phase error');
    assert.equal(record.phaseSecondaryEl.textContent, '');
    assert.equal(record.phaseSecondaryEl.hidden, true);
    assert.equal(record.phaseEl.getAttribute('aria-label'), 'Task status: Failed');
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
        phase: 'working', text: 'Finalizing…', className: 'chat-live-phase working finalizing', ...LIVE,
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
        phase: 'working', text: 'Working', className: 'chat-live-phase working', ...LIVE,
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
    // A retained typed historical lifecycle may finish the card only in the
    // proven-absent result branch after complete fresh activity excluded it.
    assert.match(missingHeal, /if \(!vouched && detail === null\)/);
    assert.match(missingHeal, /return settleMissingTaskRecord\(currentRecord\)/);
    const fallback = chatSource.slice(chatSource.indexOf('function settleMissingTaskRecord'),
        chatSource.indexOf('function markTaskCancelable'));
    assert.match(fallback, /applyHistoricalModelExecution\(record, historical\)/);
    assert.match(fallback, /return finishLiveCard\(record\.groupId, historical\.phase\)/);
    assert.match(fallback, /setHistoricalUnavailable\(record, true\)/);
});

test('history replay keeps open summaries live and terminal fallbacks factual', () => {
    const summary = chatSource.slice(
        chatSource.indexOf('function appendTaskSummaryToLiveCard'),
        chatSource.indexOf('// child task_id'),
    );
    assert.match(summary, /const finalizing = msg\?\.task_phase === 'finalizing' \|\| msg\?\.outcome_final === false;/);
    assert.match(summary, /taskTerminalSummary\(\{ \.\.\.msg, task_id: taskId \}\)/);
    for (const frame of [
        { status: 'completed', task_phase: 'finalizing' },
        { system_type: 'task_summary', outcome_final: false },
    ]) assert.equal(taskTerminalSummary(frame).terminal, false);
    assert.equal(taskTerminalSummary({ system_type: 'task_summary', outcome_final: true }).terminal, true);
    assert.match(summary, /record\.finalizingHold = true/);
    assert.match(summary, /if \(finalizing\) return changed;\s*changed = finishLiveCard/);

    assert.equal(replayTerminalPhase({ finished: false, phaseEl: {
        dataset: { phase: 'working' },
    } }), 'done');
    assert.equal(replayTerminalPhase({ finished: true, phaseEl: {
        dataset: { phase: 'error' },
    } }), 'error');
    assert.equal(replayTerminalPhase({}), 'done');
    // One replay-conclusion site finishes a child's and a root's card alike (A5 fold).
    assert.equal(
        [...chatSource.matchAll(/finishLiveCard\(taskId, msg\.task_terminal_status \? taskTerminalPhase\(msg\) : replayTerminalPhase\(liveCardRecords\.get\(taskId\)\)\);/g)].length,
        1,
    );
    assert.doesNotMatch(
        chatSource,
        /taskState\?\.completedPhase \|\| record\?\.phaseEl\?\.dataset\?\.phase \|\| 'done'/,
    );

    const wsSummary = chatSource.slice(
        chatSource.indexOf("if (msg.system_type === 'task_summary')"),
        chatSource.indexOf("if (explicitTaskId && subagentChildParents.has", chatSource.indexOf("if (msg.system_type === 'task_summary')")),
    );
    assert.match(wsSummary, /const changed = appendTaskSummaryToLiveCard\(msg\);/);
});

test('phase chips are contextual polite status regions without repeat announcements', () => {
    const attrs = new Map();
    const attrWrites = new Map();
    let textWrites = 0;
    const phaseEl = chipElement();
    phaseEl.getAttribute = (name) => attrs.get(name) ?? null;
    phaseEl.setAttribute = (name, value) => {
        attrs.set(name, value);
        attrWrites.set(name, (attrWrites.get(name) || 0) + 1);
    };
    // The words are the chip's one text node: count every write it takes.
    chipDoc.createTextNode = (data) => {
        const node = new TextStub(data);
        let value = node.data;
        Object.defineProperty(node, 'data', { get: () => value, set: (next) => { value = next; textWrites += 1; } });
        return node;
    };
    const record = { phaseEl, isSubagent: true };
    assert.equal(setLiveCardPhase(record, 'working'), true);
    chipDoc.createTextNode = (data) => new TextStub(data);
    assert.deepEqual({ ...phaseEl.dataset }, { phase: 'working', chipTone: 'live' });
    assert.equal(phaseEl.className, 'chat-live-phase working');
    assert.equal(markOf(phaseEl), 'moving/ink');
    const textContent = () => phaseEl.textContent;
    assert.equal(textContent(), 'Working');
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
    assert.equal(textContent(), 'Failed');
    assert.deepEqual([markOf(phaseEl), phaseEl.dataset.chipTone], ['dot/failed', 'failed']);
    assert.equal(attrs.get('aria-label'), 'Subagent status: Failed');
    assert.equal(textWrites, stableTextWrites + 1);
    assert.equal(attrWrites.get('aria-label'), stableLabelWrites + 1);
    const sticky = desiredLiveCardPhase({ cancelPendingPolicy: 'finalize' }, 'warn');
    assert.equal(setLiveCardPhase(record, sticky.phase, sticky.text, sticky.className), true);
    const stickyTextWrites = textWrites;
    const stickyLabelWrites = attrWrites.get('aria-label');
    assert.equal(setLiveCardPhase(record, sticky.phase, sticky.text, sticky.className), false);
    assert.equal(textContent(), 'Finalizing…');
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
    // Retired with #1992: the `shouldPromote` switch and the `updates > 1` title carry-over
    // are gone, because only an explicit name, a child's identity or authored narration
    // titles a card (chat.js cardTitle). chat_headline_reconciliation.test.js pins the
    // behaviour they guarded: these diagnostics stay rows and never retitle the card.
    // #1369: the status chip already says Working; no title or headline fallback repeats it.
    assert.doesNotMatch(chatSource, /\|\| 'Working\.\.\.'/);
    assert.doesNotMatch(logEventsSource, /headline = 'Working\.\.\.'|\|\| 'Working\.\.\.'/);
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
                    reason: 'review_degraded',
                    rationale: 'Acceptance reviewers did not reach a valid quorum.',
                },
            },
        },
    };
    const live = summarizeChatLiveEvent(evt);
    const replay = summarizeLogEvent(evt);
    assert.deepEqual({ phase: live.phase, headline: live.headline }, { phase: 'warn', headline: 'Done with warnings' });
    assert.match(live.body, /The reviewers did not reach a verdict on this answer\./);
    assert.doesNotMatch(live.body, /final_message/);
    // The raw code lives on in the record half, never in the card body.
    assert.doesNotMatch(live.body, /finalized_unaccepted/);
    assert.ok(replay.meta.includes('review degraded'));
    assert.ok(replay.meta.includes('acceptance finalized_unaccepted'));
});

test('#1110 inert history cannot retain a floating Finalizing chip', () => {
    const record = {
        finalizingHold: true, observedOutcome: 'error', finished: false,
        phaseEl: chipElement(),
        phaseSecondaryEl: { hidden: true, textContent: '', isConnected: true },
    };
    setLiveCardPhase(record, 'error', 'Failed', '', 'Finalizing…');
    assert.equal(record.phaseSecondaryEl.hidden, false);
    setInertCardPresentation(record, true);
    setLiveCardPhase(record, 'error', 'Failed', '', 'Finalizing…');
    assert.equal(record.phaseEl.hidden, true);
    assert.equal(record.phaseSecondaryEl.hidden, true);
    setInertCardPresentation(record, false);
    assert.equal(record.phaseSecondaryEl.hidden, false);
});


test('ordinary early-final handler consumes producer status without ending lifecycle', () => {
    const body = chatSource.slice(chatSource.indexOf('    function markLiveCardFinalizing('),
        chatSource.indexOf('    // Durable cancel state wins', chatSource.indexOf('    function markLiveCardFinalizing(')));
    for (const [fact, expected] of [
        [{ task_terminal_status: 'failed' }, 'error'],
        [{ task_terminal_status: 'completed', outcome_axes: { execution: { status: 'degraded' } } }, 'warn'],
        [{}, 'working'],
        [{ type: 'tool_error', status: 'running' }, 'working'],
    ]) {
        const record = { phaseEl: {}, title: 'Stable task name', finished: false };
        const handler = new Function('liveCardRecords', 'taskKey', 'withStableViewport',
            'markReviewAnchor', 'taskTerminalSummary', 'desiredLiveCardPhase', 'setLiveCardPhase',
            `${body}; return markLiveCardFinalizing;`)(new Map([['t', record]]), x => x, fn => fn(),
            () => false, taskTerminalSummary, desiredLiveCardPhase, () => true);
        handler('t', { ...fact, task_phase: 'finalizing' });
        assert.equal(desiredLiveCardPhase(record).phase, expected);
        assert.equal(record.finalizingHold, true);
        assert.equal(record.finished, false);
        assert.equal(record.title, 'Stable task name');
    }
});

function chipRecord(extra = {}) {
    const phaseEl = chipElement();
    const attrs = { get: (key) => phaseEl.getAttribute(key) };
    return { phaseEl, attrs, finished: false, ...extra };
}

test('Batch4: a paused or pausing card never says Working; Stop still outranks the pause', () => {
    // A settled Pause stands until Resume (its bars); one still settling moves; neither is amber.
    assert.deepEqual(desiredLiveCardPhase({ finished: false, parkedPhase: 'budget_paused' }), {
        phase: 'paused', text: 'Paused', className: 'chat-live-phase warn', mark: 'pause', tone: 'neutral', owner: false,
    });
    assert.deepEqual(desiredLiveCardPhase({ finished: false, parkedPhase: 'budget_pausing' }), {
        phase: 'working', text: 'Pausing…', className: 'chat-live-phase working waiting', mark: 'moving', tone: 'neutral',
        owner: false,
    });
    assert.equal(desiredLiveCardPhase({ finished: false, parkedPhase: 'budget_paused', cancelPendingPolicy: 'immediate' }).text,
        'Cancelling…');
    assert.equal(desiredLiveCardPhase({ finished: true, parkedPhase: 'budget_paused' }, 'done').text, 'Done');
});

test('Batch4: the census phase parks and releases a live card chip', () => {
    const record = chipRecord();
    assert.equal(syncParkedPhase(record, 'budget_pausing'), true);
    assert.equal(record.phaseEl.textContent, 'Pausing…');
    assert.equal(record.attrs.get('aria-label'), 'Task status: Pausing…');
    assert.equal(markOf(record.phaseEl), 'moving/ink', 'a Pause still settling keeps its moving dots, in neutral ink');
    assert.equal(record.phaseEl.dataset.chipTone, 'neutral');
    assert.equal(syncParkedPhase(record, 'budget_paused'), true);
    assert.equal(record.phaseEl.textContent, 'Paused');
    assert.equal(markOf(record.phaseEl), 'pause/ink', 'a paused card shows its bars, nothing that moves');
    assert.equal(syncParkedPhase(record, 'budget_paused'), false, 'an unchanged phase writes nothing');
    assert.equal(syncParkedPhase(record, 'unknown'), true);
    assert.equal(record.phaseEl.textContent, 'Activity unconfirmed');
    assert.equal(markOf(record.phaseEl), 'ring/ink');
    assert.equal(syncParkedPhase(record, 'budget_paused'), true);
    assert.equal(record.phaseEl.textContent, 'Paused');
    assert.equal(syncParkedPhase(record, 'working'), true, 'Resume releases it');
    assert.equal(record.phaseEl.textContent, 'Working');
    assert.deepEqual([markOf(record.phaseEl), record.phaseEl.dataset.chipTone], ['moving/ink', 'live']);
    const finished = chipRecord({ finished: true });
    assert.equal(syncParkedPhase(finished, 'budget_paused'), false, 'a settled card keeps its outcome');
});

test('Batch4: chat feeds every census phase to its card', () => {
    const hydrate = chatSource.slice(chatSource.indexOf('function hydrateDirectActivities'),
        chatSource.indexOf('const isKnownProjectFrame'));
    assert.match(hydrate, /syncParkedPhase\(record, v\.phase, v\);/);
});

test('an owner Pause says the review work it lets finish beside its word, and the line clears when it ends', () => {
    const record = chipRecord({ phaseSecondaryEl: { textContent: '', hidden: true, isConnected: true } });
    const line = () => [record.phaseSecondaryEl.textContent, record.phaseSecondaryEl.hidden];
    // The census `owner` cause also covers Ouroboros's own Pause: who paused is the work projection's to say.
    assert.equal(syncParkedPhase(record, 'budget_paused', { pause_cause: 'owner', finishing_reviews: true }), true);
    assert.deepEqual([record.phaseEl.textContent, ...line()], ['Paused', 'Review work finishing', false]);
    assert.equal(record.attrs.get('aria-label'), 'Task status: Paused, Review work finishing', 'one status, said whole');
    assert.equal(syncParkedPhase(record, 'budget_paused', { pause_cause: 'owner', finishing_reviews: true }), false,
        'an unchanged fact writes nothing');
    assert.doesNotMatch(record.phaseSecondaryEl.textContent, /\d/, 'the census names tasks and sends alike: never a count');
    assert.equal(syncParkedPhase(record, 'budget_paused', { pause_cause: 'owner' }), true);
    assert.deepEqual([record.phaseEl.textContent, ...line()], ['Paused', '', true]);
    syncParkedPhase(record, 'budget_pausing', { pause_cause: 'owner', finishing_reviews: true });
    assert.deepEqual([record.phaseEl.textContent, ...line()], ['Pausing…', '', true],
        'only a settled Pause names finishing review work');
});

test('a census pause says its word alone whatever its cause, and follows current typed facts without a phase transition', () => {
    const record = chipRecord();
    for (const [pause_cause, label] of [['budget', 'Paused'], ['owner', 'Paused'], ['restart', 'Paused'],
        ['authority', 'Paused'], ['future-cause', 'Paused'], ['sleep', 'Waiting']]) {
        syncParkedPhase(record, 'budget_paused', { pause_cause });
        assert.equal(record.phaseEl.textContent, label, pause_cause);
    }
    // A cold sleep waits (still dots); every other cause holds its bars.
    assert.equal(markOf(record.phaseEl), 'still/ink');
    assert.equal(syncParkedPhase(record, 'budget_paused', { pause_cause: 'budget' }), true, 'the cause changed the look');
    assert.equal(markOf(record.phaseEl), 'pause/ink');
    syncParkedPhase(record, 'budget_pausing', { pause_cause: 'budget' });
    assert.equal(record.phaseEl.textContent, 'Pausing…');
    assert.equal(record.phaseEl.dataset.chipTone, 'neutral', 'submitted work settles without the live word');
    const late = desiredLiveCardPhase({ finalizingHold: true, observedOutcome: 'error',
        parkedPhase: 'budget_paused', pauseCause: 'restart', finishingReviews: true });
    assert.deepEqual([late.text, late.secondary], ['Failed', 'Paused · Held after a restart · Review work finishing'],
        'the outcome owns the chip; the pause, what holds it and what it lets finish stand beside it');
});

test('the card of a work\'s current episode says the work\'s word and line; Stop and a closed socket still win', () => {
    const record = chipRecord();
    record.workView = { phase: 'paused', text: 'Paused', secondary: 'Ouroboros: pause before the update',
        classes: 'paused', tone: 'neutral', waiting: true };
    record.workLook = { mark: 'pause', owner: false };
    assert.equal(syncParkedPhase(record, 'budget_paused', { pause_cause: 'owner' }), true);
    assert.equal(record.phaseEl.textContent, 'Paused');
    assert.equal(desiredLiveCardPhase(record).secondary, 'Ouroboros: pause before the update');
    assert.equal(record.phaseEl.className, 'chat-live-phase paused');
    assert.deepEqual([markOf(record.phaseEl), record.phaseEl.dataset.chipTone], ['pause/ink', 'neutral'],
        'Ouroboros\'s Pause is not amber');
    assert.equal(desiredLiveCardPhase({ ...record, cancelPendingPolicy: 'immediate' }).text, 'Cancelling…');
    syncParkedPhase(record, 'unknown', { pause_cause: 'owner' });
    assert.equal(record.phaseEl.textContent, 'Activity unconfirmed');
    record.workView = null;
    syncParkedPhase(record, 'budget_paused', { pause_cause: 'owner' });
    assert.equal(record.phaseEl.textContent, 'Paused', 'without its work the card keeps its own census words');
});

test('a census pause says its word alone and what holds it on its line, never who', () => {
    const view = (pauseCause, phase = 'budget_paused') => taskActivityPresentation({ phase, pauseCause });
    for (const [cause, word, line] of [
        ['budget', 'Paused', 'Spending limit reached'],
        ['restart', 'Paused', 'Held after a restart'],
        ['authority', 'Paused', 'After a technical problem'],
        ['sleep', 'Waiting', 'Sleeping until it is woken'],
    ]) {
        assert.deepEqual([view(cause).text, view(cause).secondary], [word, line], cause);
    }
    // The census's owner cause cannot tell the owner's Pause from Ouroboros's: the word alone, no line.
    assert.deepEqual([view('owner').text, view('owner').secondary], ['Paused', undefined]);
    assert.equal(view('owner', 'budget_pausing').text, 'Pausing…');
    for (const cause of ['budget', 'restart', 'authority', 'sleep', 'owner']) {
        assert.doesNotMatch(view(cause).text, / · /, `${cause}: the chip never carries a reason`);
    }
});
