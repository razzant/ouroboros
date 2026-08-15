import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    summarizeChatLiveEvent,
    summarizeLogEvent,
    taskCancelPending,
    taskOutcomeSeverity,
    taskTerminalPhase,
} from '../modules/log_events.js';
import { cancelRunEligibility, isTerminalTaskPhase } from '../modules/chat.js';

// --- cancelled severity reducer (added ONCE, consumed everywhere) ---

test('taskOutcomeSeverity classifies cancelled lifecycle as its own severity', () => {
    assert.equal(taskOutcomeSeverity({ status: 'cancelled' }), 'cancelled');
    // 'cancel_requested' as a STATUS is legacy replay only (phase A moved intent
    // to the typed cancel_state projection) — old frames still resolve honestly.
    assert.equal(taskOutcomeSeverity({ status: 'cancel_requested' }), 'cancelled');
    assert.equal(taskOutcomeSeverity({
        outcome_axes: { lifecycle: { status: 'cancelled' }, execution: { status: 'cancelled' } },
    }), 'cancelled');
});

// --- phase A: typed pending-cancel projection + interim card state ---

test('taskCancelPending reads the typed projection, never the status', () => {
    assert.equal(taskCancelPending({ status: 'running', cancel_state: 'pending' }), true);
    assert.equal(taskCancelPending({ status: 'scheduled', cancel_state: 'pending' }), true);
    // A settled record is settled — the projection cannot resurrect it.
    assert.equal(taskCancelPending({ status: 'cancelled', cancel_state: 'pending' }), false);
    assert.equal(taskCancelPending({ status: 'completed', cancel_state: 'pending' }), false);
    // No projection = no pending cancel.
    assert.equal(taskCancelPending({ status: 'running' }), false);
});

test('the cancel click shows the honest interim, not an instant Cancelled', () => {
    // Pinned at source: the click handler marks the card "Cancelling…" while the
    // durable intent settles, and the stored-record reconcile branches keep the
    // interim for a nonterminal record with cancel_state=pending instead of
    // finishing the card — through the SHARED taskCancelPending helper (AR2-8:
    // one consumer path for the typed projection, never an inline status peek).
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    assert.match(chat, /function markLiveCardCancelPending\(/);
    assert.match(chat, /markLiveCardCancelPending\(taskId\);\n[\s\S]{0,400}await cancelTask\(/);
    assert.match(chat, /taskCancelPending\(stored\)[\s\S]{0,400}markLiveCardCancelPending\(taskId\)/);
    assert.doesNotMatch(chat, /cancel_state === 'pending'/);
    assert.match(chat, /Cancelling…/);
});

test('cancellation wins over failure-shaped teardown side facts', () => {
    // A cancelled workspace task legitimately has artifacts=missing — that must
    // not relabel an owner-requested cancellation as "Failed".
    assert.equal(taskOutcomeSeverity({
        status: 'cancelled',
        artifact_status: 'missing',
        outcome_axes: { lifecycle: { status: 'cancelled' }, artifacts: { status: 'missing' } },
    }), 'cancelled');
});

test('non-cancelled severities are unchanged', () => {
    assert.equal(taskOutcomeSeverity({ status: 'done' }), 'done');
    assert.equal(taskOutcomeSeverity({ outcome_axes: { lifecycle: { status: 'failed' } } }), 'error');
    assert.equal(taskOutcomeSeverity({ outcome_axes: { execution: { status: 'degraded' } } }), 'warn');
});

test('taskTerminalPhase maps every severity to its card phase', () => {
    assert.equal(taskTerminalPhase({ status: 'cancelled' }), 'cancelled');
    assert.equal(taskTerminalPhase({ outcome_axes: { lifecycle: { status: 'failed' } } }), 'error');
    assert.equal(taskTerminalPhase({ outcome_axes: { execution: { status: 'degraded' } } }), 'warn');
    assert.equal(taskTerminalPhase({ status: 'done' }), 'done');
});

// --- both task_done summarizers ---

test('chat live task_done summarizer renders an honest Cancelled state', () => {
    const view = summarizeChatLiveEvent({ type: 'task_done', status: 'cancelled' });
    assert.equal(view.phase, 'cancelled');
    assert.equal(view.headline, 'Cancelled');
    assert.equal(view.terminal, true);
});

test('chat live task_done summarizer keeps the clean Done contract', () => {
    const view = summarizeChatLiveEvent({ type: 'task_done', status: 'done' });
    assert.equal(view.phase, 'done');
    assert.equal(view.headline, 'Done');
});

test('logs task_done summarizer labels cancellation as Cancelled', () => {
    const view = summarizeLogEvent({ type: 'task_done', status: 'cancelled' });
    assert.equal(view.phase, 'cancelled');
    assert.equal(view.headline, 'Cancelled');
});

// --- terminal phase + history replay fallback ---

test('cancelled is a terminal card phase (card resolves, never re-inflates)', () => {
    assert.equal(isTerminalTaskPhase('cancelled'), true);
    assert.equal(isTerminalTaskPhase('done'), true);
    assert.equal(isTerminalTaskPhase('working'), false);
});

test('history replay of a cancelled root resolves to Cancelled, not Done', () => {
    // The reload fallback builds {...row, status: task_terminal_status} and asks
    // taskTerminalPhase for the finishLiveCard phase (chat.js terminal fallback).
    const terminalRecord = { task_id: 'root1', status: 'cancelled' };
    assert.equal(taskTerminalPhase(terminalRecord), 'cancelled');
    assert.notEqual(taskTerminalPhase(terminalRecord), 'done');
});

// --- Cancel run eligibility (host-attested marker + structural gates) ---

test('Cancel run offered only on live, marker-attested root cards', () => {
    const eligible = {
        groupId: 'abc12345', isSubagent: false, finished: false, cancelable: true, converted: false,
    };
    assert.equal(cancelRunEligibility(eligible), true);
    // Subagent cards never offer it (the root cascade covers them).
    assert.equal(cancelRunEligibility({ ...eligible, isSubagent: true }), false);
    // Reusable slots (background consciousness / legacy active) never offer it.
    assert.equal(cancelRunEligibility({ ...eligible, groupId: 'bg-consciousness' }), false);
    assert.equal(cancelRunEligibility({ ...eligible, groupId: 'active' }), false);
    // Finished and converted cards have nothing live to cancel.
    assert.equal(cancelRunEligibility({ ...eligible, finished: true }), false);
    assert.equal(cancelRunEligibility({ ...eligible, converted: true }), false);
    // Without the host-attested marker (e.g. a direct-chat turn's card, which has
    // the same shape but no queue entry) the button must not appear.
    assert.equal(cancelRunEligibility({ ...eligible, cancelable: false }), false);
    assert.equal(cancelRunEligibility({ ...eligible, groupId: '' }), false);
});

test('both cancel surfaces report a refused cancellation', () => {
    // The endpoint answers only after the teardown, so success needs no extra
    // reporting — but a refusal must never read as a silent no-op click.
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const activity = readFileSync(new URL('../modules/activity.js', import.meta.url), 'utf8');
    for (const source of [chat, activity]) {
        assert.match(source, /await cancelTask\(/);
    }
    // ...and Activity no longer swallows a refused cancel (503) as a no-op click,
    // while keeping the documented 404 completion race graceful.
    assert.match(activity, /exc\?\.status !== 404/);
    assert.match(activity, /catch \(exc\)[\s\S]{0,400}showToast\(`Action failed/);
});

test('a timeout-retry root gains Cancel run: the host marker is the truth', () => {
    // A retry root's frame carries root_task_id naming the ORIGINAL task, so any
    // structural frameRoot===taskId gate would reject exactly the marker the
    // supervisor attested. Pinned at source: the handler trusts the marker alone.
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    assert.match(chat, /msg\?\.cancelable === true && msg\?\.task_id\) markTaskCancelable/);
    assert.doesNotMatch(chat, /frameRoot === taskId\) *&&[\s\S]{0,80}markTaskCancelable/);
    // ...and the eligibility reducer still refuses subagent/finished/reusable cards,
    // so trusting the marker does not widen the button beyond live pooled roots.
    assert.equal(cancelRunEligibility({
        groupId: 'retry-1', isSubagent: false, finished: false, cancelable: true,
    }), true);
    assert.equal(cancelRunEligibility({
        groupId: 'child-1', isSubagent: true, finished: false, cancelable: true,
    }), false);
});

test('a 404 cancel reconciles the card from the durable record', () => {
    // 404 says "not live"; if the terminal frame was lost the card would sit
    // "Working" forever. The branch must fetch the durable record and resolve the
    // card through the SAME terminal seam replay uses — not merely hide a button.
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const branch = chat.slice(chat.indexOf('cancelableTaskIds.delete(taskId)'));
    // …and NOT from cache: the whole point is to see the fresh terminal status, and a
    // cached pre-cancel 200 leaves the card "Working" behind a dead disabled button.
    assert.match(
        branch.slice(0, 1600),
        /apiFetch\(`\/api\/tasks\/\$\{encodeURIComponent\(taskId\)\}`, \{ cache: 'no-store' \}\)/,
    );
    // The terminal resolution itself now lives in the shared reconcile helper
    // (finishLiveCard is called there), so the branch is pinned on the helper.
    assert.match(branch.slice(0, 1600), /reconcileCancelCardFromDetail\(record, taskId, stored\)/);
});

test('a successful cancel also reconciles when task_done publication is lost', () => {
    // Durable cancellation precedes fail-soft publication. A 200 with no WS frame
    // must therefore read the stored result before leaving the button disabled.
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const success = chat.slice(chat.indexOf('await cancelTask(taskId, { cascade: true })'));
    const beforeCatch = success.slice(0, success.indexOf('} catch (exc)'));
    // Fresh, not cached: a cached pre-cancel 200 would leave the card "Working"
    // behind a dead disabled button — the exact failure this reconcile prevents.
    assert.match(
        beforeCatch,
        /apiFetch\(`\/api\/tasks\/\$\{encodeURIComponent\(taskId\)\}`, \{ cache: 'no-store' \}\)/,
    );
    assert.match(beforeCatch, /reconcileCancelCardFromDetail\(record, taskId, stored\)/);
});

// --- GR2-8: cancel-state honesty (failure restore + pending-before-terminal) ---

test('task-detail reconciliation consults taskCancelPending BEFORE the legacy terminal fallback', () => {
    // GR2-8b: a live task wedged in the legacy `cancel_requested` STATUS latch is
    // INTENT, not outcome — it must show as cancel-pending, never resolve as a
    // terminal "Cancelled" while the supervisor is still tearing it down. Pinned
    // at source: the shared reconcile helper checks the typed projection first
    // and only falls through to the terminal list afterwards.
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const helper = chat.slice(chat.indexOf('function reconcileCancelCardFromDetail'));
    const pendingAt = helper.indexOf('taskCancelPending(stored)');
    const terminalAt = helper.indexOf("'cancel_requested'");
    assert.ok(pendingAt > 0 && terminalAt > 0, 'both branches exist in the helper');
    assert.ok(pendingAt < terminalAt, 'the typed pending check runs before the terminal fallback');
    // ALL reconcile call sites (success, 404, and the GR3-10 non-404 failure)
    // route through the ONE helper (no inline order drift).
    assert.equal(chat.match(/reconcileCancelCardFromDetail\(record, taskId, stored\);/g).length, 3);
});

test('a failed cancel reconciles through the shared helper before touching the button', () => {
    // GR3-10 (supersedes the GR2-8a inline check): the non-404 failure path
    // must reconcile the fetched durable detail through the SHARED
    // reconcileCancelCardFromDetail helper — keep the button disabled while
    // cancel_state=pending, finish the card for a terminal record — and only
    // a genuinely-live, non-pending task gets its prior phase restored and
    // the button re-enabled.
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    assert.match(chat, /const priorPhase = captureLiveCardPhase\(record\);\n\s*markLiveCardCancelPending\(taskId\);/);
    const failure = chat.slice(chat.indexOf('showToast(`Cancel failed:'));
    const branch = failure.slice(0, 2200);
    // The shared seam runs BEFORE any button re-enable.
    const reconcileAt = branch.indexOf('reconcileCancelCardFromDetail(record, taskId, stored)');
    const reenableAt = branch.indexOf('btn.disabled = false');
    assert.ok(reconcileAt > 0 && reenableAt > 0, 'both the reconcile and the re-enable exist');
    assert.ok(reconcileAt < reenableAt, 'reconciliation happens before the button is re-enabled');
    // Pending or terminal ⇒ return WITHOUT re-enabling or restoring the phase.
    assert.match(branch, /if \(record\.finished \|\| stillPending\) return;/);
    assert.match(branch, /taskCancelPending\(stored\)/);
    assert.match(branch, /restoreLiveCardPhase\(record, priorPhase\)/);
});

test('every cancel button on a task cascades, and none of them can forget to', () => {
    // The class: a NEW call to an existing API that omits a parameter every
    // neighbouring call passes. `cancelRemoteTask` called the duplicate
    // `apiClient.taskCancel` spelling — which took no options at all — so cancelling
    // a remote orchestrator from its own card left its subagents running on the
    // target, while the two other buttons for the same action cascaded.
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const activity = readFileSync(new URL('../modules/activity.js', import.meta.url), 'utf8');
    const client = readFileSync(new URL('../modules/api_client.js', import.meta.url), 'utf8');
    // Every cancel POST goes through the ONE helper, and every one of them cascades.
    for (const source of [chat, activity]) {
        const calls = source.match(/cancelTask\([^)]*\)/g) || [];
        assert.ok(calls.length > 0);
        for (const call of calls) assert.match(call, /cascade: true/);
        // The duplicate spelling that could not express cascade is gone from the
        // client, so no caller can reach for it again.
        assert.doesNotMatch(source, /taskCancel\(/);
    }
    assert.doesNotMatch(client, /taskCancel:/);
    // The chat prompt names the consequence, like Activity's does for the same task.
    assert.match(chat, /Cancel this task and all its subagents\?/);
    // And the optimistic status is one the state vocabulary actually knows: the old
    // `cancel_requested` was in neither CANCELLABLE nor TERMINAL, so the card sat in
    // a limbo that offered no Cancel and never read as finished.
    const remote = chat.slice(chat.indexOf('async function cancelRemoteTask'));
    const body = remote.slice(0, remote.indexOf('async function reconnectRemoteTask'));
    assert.match(body, /taskStatus: 'cancelled'/);
    // Matched on the ASSIGNMENT, not the word: the comment above it explains what the
    // old status was, and a prose ban would forbid saying so.
    assert.doesNotMatch(body, /taskStatus: 'cancel_requested'/);
    assert.doesNotMatch(body, /completion: 'cancel_requested'/);
});
