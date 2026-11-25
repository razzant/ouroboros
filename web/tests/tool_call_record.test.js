import assert from 'node:assert/strict';
import test from 'node:test';

import { noteToolCall, toolEvidenceView } from '../modules/chat_activity.js';
import { categorizeLogEvent, summarizeChatLiveEvent, summarizeLogEvent } from '../modules/log_events.js';
import { refusedStepView } from '../modules/verification_presentation.js';

// #1316: a durable tool_call_started row (a Logs backfill) records that host
// processing began; replayed alone it must not read as a live "Running" call.
test('a tool start reads as started, its wait end as a timeout, its result as a result', () => {
    const base = { task_id: 't', tool: 'run_command', invocation_id: 'i1', args: { cmd: 'make' } };
    const started = summarizeLogEvent({ ...base, type: 'tool_call_started', timeout_sec: 60 });
    assert.equal(started.headline, 'Run command started');
    assert.doesNotMatch(started.headline, /Running/);
    assert.equal(summarizeLogEvent({ ...base, type: 'tool_call_timeout', timeout_sec: 60 }).headline,
        'Run command timed out');
    assert.equal(summarizeLogEvent({ ...base, type: 'tool_call', result_preview: 'ok' }).headline,
        'Run command result');
});

// #1316: providers reuse their call ids across rounds (GigaChat always says "call_0"),
// so the live row key is the host's invocation id; a legacy frame keeps the provider id.
test('two invocations sharing a provider id stay two rows and never inherit a status', () => {
    const frame = (type, row) => summarizeChatLiveEvent({ type, task_id: 't', tool: 'read_file', tool_call_id: 'call_0', ...row });
    const first = [frame('tool_call_started', { invocation_id: 'i1' }),
        frame('tool_call_finished', { invocation_id: 'i1', is_error: true, error: 'boom' })];
    const second = [frame('tool_call_started', { invocation_id: 'i2' }), frame('tool_call_finished', { invocation_id: 'i2' })];
    assert.notEqual(first[0].toolCall.key, second[0].toolCall.key);
    assert.equal(first[1].dedupeKey, 'tool:t:i1', 'the failure row is keyed by its own invocation');
    const record = {};
    for (const view of [...first, ...second]) noteToolCall(record, view.toolCall);
    assert.deepEqual([...record.toolFold.calls.values()].map((call) => call.status), ['error', 'ok']);
    const row = toolEvidenceView(record.toolFold);
    assert.deepEqual([row.calls, row.errors], [2, 1]);
    const legacy = frame('tool_call_finished', { is_error: true, error: 'boom' });
    assert.notEqual(legacy.dedupeKey, frame('tool_call_finished', { is_error: true }).dedupeKey,
        'legacy observations cannot be joined by reused provider ids');
});


test('independent wait end and durable settlement converge in both orders', () => {
    const row = type => summarizeChatLiveEvent({ type, task_id: 't', tool: 'read_file', invocation_id: 'i' });
    for (const order of [['tool_call_timeout', 'tool_call'], ['tool_call', 'tool_call_timeout']]) {
        const record = {};
        for (const type of order) noteToolCall(record, row(type).toolCall);
        record.toolFold.host = { calls: 1, errors: 1 };
        const view = toolEvidenceView(record.toolFold);
        assert.equal(view.errors, 0);
        assert.match(view.headline, /wait ended/);
    }
});

test('historical starts are unknown and event aliases retain their actual phase', () => {
    const record = {};
    noteToolCall(record, summarizeChatLiveEvent({ event: 'tool_call_started', task_id: 't', invocation_id: 'i' }).toolCall);
    assert.match(toolEvidenceView(record.toolFold).headline, /outcome unknown/);
    noteToolCall(record, summarizeChatLiveEvent({ type: 'tool_call', task_id: 't', invocation_id: 'i' }).toolCall);
    assert.doesNotMatch(toolEvidenceView(record.toolFold).headline, /unknown|error/);
});

test('a call Safety held reads as held outside the error phase, and the block counts no error for it', () => {
    const frame = (status, invocation_id) => summarizeChatLiveEvent({ type: 'tool_call_finished', task_id: 't',
        tool: 'run_command', invocation_id, is_error: true, status });
    const views = [frame('safety_returned', 'i1'), frame('safety_unavailable', 'i2'), frame('error', 'i3')];
    assert.deepEqual(views.map((view) => [view.phase, view.headline]), [
        ['warn', 'A command was returned by the Safety Supervisor'],
        ['warn', 'A command was not run: the Safety Supervisor could not read it'],
        ['error', 'One of the steps failed · run_command']]);
    const record = {};
    for (const view of views) noteToolCall(record, view.toolCall);
    assert.deepEqual([...record.toolFold.calls.values()].map((call) => call.status), ['held', 'held', 'error']);
    assert.deepEqual([toolEvidenceView(record.toolFold).calls, toolEvidenceView(record.toolFold).errors], [3, 1]);
    const logged = summarizeLogEvent({ type: 'tool_call', tool: 'run_command', is_error: true, status: 'safety_unavailable' });
    assert.deepEqual([logged.phase, logged.headline], ['warn', 'A command was not run: the Safety Supervisor could not read it']);
});

test('a call a control ended reads by its control outside the error phase, and the block counts no error for it', () => {
    const frame = (control_reason, invocation_id, extra = {}) => summarizeChatLiveEvent({ type: 'tool_call', task_id: 't',
        tool: 'run_command', invocation_id, is_error: true, status: 'interrupted', control_reason, ...extra });
    const views = [frame('owner_pause', 'i1'), frame('cancelled', 'i2'), frame('spend_limit_root', 'i3'),
        frame('model_wait_reprepare_required', 'i4'),
        summarizeChatLiveEvent({ type: 'tool_call', task_id: 't', tool: 'run_command', invocation_id: 'i5', is_error: true, status: 'host_error' })];
    assert.deepEqual(views.map((view) => [view.phase, view.headline]), [
        ['info', 'Interrupted by Pause · run_command'], ['info', 'Interrupted by Stop · run_command'],
        ['info', 'Interrupted: spend limit · run_command'], ['info', 'Interrupted · run_command'],
        ['error', 'One of the steps failed · run_command']]);
    // A Pause fence's own reason and a live call's end name their control; a word that merely shares a property
    // name with every object names none.
    assert.deepEqual(['owner_pause_before_review_launch', "the call's end", 'toString'].map((word) => frame(word, word).headline),
        ['Interrupted by Pause · run_command', 'Interrupted: the call ended · run_command', 'Interrupted · run_command']);
    const record = {};
    for (const view of views) noteToolCall(record, view.toolCall);
    assert.deepEqual([...record.toolFold.calls.values()].map((call) => call.status),
        ['interrupted', 'interrupted', 'interrupted', 'interrupted', 'error']);
    assert.deepEqual([toolEvidenceView(record.toolFold).calls, toolEvidenceView(record.toolFold).errors], [5, 1]);
    // A row the host settled itself is provisional: the handler's own later settlement of the same call replaces it.
    noteToolCall(record, summarizeChatLiveEvent({ type: 'tool_call', task_id: 't', tool: 'run_command', invocation_id: 'i1',
        status: 'ok' }).toolCall);
    assert.equal(record.toolFold.calls.get([...record.toolFold.calls.keys()][0]).status, 'ok');
    const logged = summarizeLogEvent({ type: 'tool_call', tool: 'run_command', is_error: true, status: 'interrupted',
        control_reason: 'owner_pause' });
    assert.deepEqual([logged.phase, logged.headline], ['info', 'Run command interrupted (owner_pause)']);
});

// A non-error completion refused further work reads neutral for Stop/already cancelled, warning for unread Stop.
// The same code without the host's mark (a Pause refusal) and a neighbouring failure keep their failure headline.
test('a call the host refused a further step reads as that refusal outside the error phase, live and settled alike', () => {
    const refused = { refused_step: true, control_reason: 'cancelled' };
    const ended = { refused_step: true, control_reason: 'task_cancelled' };
    const frame = (type, invocation_id, extra = {}) => summarizeChatLiveEvent({ type, task_id: 't', tool: 'ext_stopper_send',
        invocation_id, is_error: true, status: 'blocked', ...extra });
    const views = [frame('tool_call', 'i1', { tool_result_code: 'OWNER_OPERATION_REFUSED', tool_result_meta: refused }),
        frame('tool_call_finished', 'i1', { tool_result_meta: refused }),
        frame('tool_call_finished', 'i2', { tool_result_meta: { refused_step: true, control_reason: 'stop_unreadable' } }),
        frame('tool_call_finished', 'i3', { tool_result_meta: { refused_step: true, control_reason: 'toString' } }),
        frame('tool_call_finished', 'i4', { tool_result_code: 'OWNER_OPERATION_REFUSED' }),
        summarizeChatLiveEvent({ type: 'tool_call_finished', task_id: 't', tool: 'run_command', invocation_id: 'i5',
            is_error: true, status: 'error' }),
        frame('tool_call', 'i6', { tool_result_code: 'OWNER_OPERATION_REFUSED', tool_result_meta: ended }),
        frame('tool_call_finished', 'i6', { tool_result_meta: ended })];
    const ending = 'A further step was refused: the task had already ended as cancelled · ext_stopper_send';
    assert.deepEqual(views.map((view) => [view.phase, view.headline]), [
        ['info', 'A further step was refused by Stop · ext_stopper_send'],
        ['info', 'A further step was refused by Stop · ext_stopper_send'],
        ['warn', "A further step was refused: the task's Stop could not be read · ext_stopper_send"],
        ['warn', 'A further step was refused · ext_stopper_send'],
        ['error', 'One of the steps failed · ext_stopper_send'],
        ['error', 'One of the steps failed · run_command'],
        ['info', ending], ['info', ending]]);
    assert.deepEqual(views.map((view) => Boolean(view.hostRefusedStep)),
        [true, true, true, true, false, false, true, true]);
    const record = {};
    for (const view of views) noteToolCall(record, view.toolCall);
    assert.deepEqual([...record.toolFold.calls.values()].map((call) => call.status),
        ['held', 'held', 'held', 'error', 'error', 'held']);
    assert.deepEqual([toolEvidenceView(record.toolFold).calls, toolEvidenceView(record.toolFold).errors], [6, 2]);
    const logged = (meta) => summarizeLogEvent({ type: 'tool_call', tool: 'ext_stopper_send', is_error: true,
        status: 'blocked', tool_result_meta: meta });
    assert.deepEqual([refused, ended, { refused_step: true, control_reason: 'stop_unreadable' }].map((meta) => logged(meta))
        .map((view) => [view.phase, view.headline]), [
        ['info', 'A further step was refused by Stop · ext_stopper_send'], ['info', ending],
        ['warn', "A further step was refused: the task's Stop could not be read · ext_stopper_send"]]);
});


test('a failed call and its refusal stay separate, red and counted once in live and replay views', () => {
    for (const status of ['extension_error', 'tool_reported_failure', 'timeout']) {
        for (const control_reason of ['cancelled', 'task_cancelled', 'stop_unreadable']) {
            const record = {};
            for (const type of ['tool_call_finished', 'tool_call']) {
                const evt = { type, task_id: 't', tool: 'ext_stopper_send', invocation_id: 'i', status, is_error: true,
                    tool_result_meta: { refused_step: true, control_reason }, result_preview: 'Original failure' };
                const chat = summarizeChatLiveEvent(evt);
                const log = summarizeLogEvent(evt);
                assert.deepEqual([chat.phase, chat.headline], ['error', 'One of the steps failed · ext_stopper_send']);
                assert.match(chat.body, /A further step was refused/);
                assert.match(chat.fullBody, /Original failure/);
                assert.equal(log.phase, 'error');
                assert.match(log.headline, /failed$/);
                assert.match(log.body, /A further step was refused/);
                assert.match(log.body, /Original failure/);
                assert.equal(categorizeLogEvent(evt, log), 'errors');
                assert.equal(chat.hostRefusedStep, true);
                noteToolCall(record, chat.toolCall);
            }
            assert.deepEqual([toolEvidenceView(record.toolFold).calls, toolEvidenceView(record.toolFold).errors], [1, 1]);
            assert.equal([...record.toolFold.calls.values()][0].status, 'error');
        }
    }
    const producerOnly = summarizeChatLiveEvent({ type: 'tool_call', task_id: 't', tool: 'ext_stopper_send',
        invocation_id: 'producer', is_error: true, status: 'tool_reported_failure',
        result_preview: '{"ok":false,"refused_step":true,"control_reason":"cancelled"}' });
    assert.equal(producerOnly.phase, 'error');
    assert.doesNotMatch(producerOnly.headline, /refused/);
    assert.equal(producerOnly.hostRefusedStep, undefined);
});

// The reload side of the refusal contract: the host's replay projection (tool_call_log.
// _refusal_evidence) redraws exactly the row the live frame drew, under the same invocation
// key — an ordinary Stop reads neutral with the producer's bounded preview, an independent
// failure stays red with the refusal in its body. Both views are built from the same typed
// facts; this pins them together so live and history cannot drift.
test('a refused step replays the row its live frame drew, under the same key', () => {
    const preview = 'part one sent.\n\n⚠️ OWNER_OPERATION_REFUSED: the host refused this call a further action: stopping (cancelled).';
    const refused = { refused_step: true, control_reason: 'cancelled' };
    const cases = [
        [refused, 'blocked', 'held'],
        [{ refused_step: true, control_reason: 'task_cancelled' }, 'blocked', 'held'],
        [{ refused_step: true, control_reason: 'stop_unreadable' }, 'blocked', 'held'],
        [{ refused_step: true, control_reason: 'toString' }, 'blocked', 'held'],
        [refused, 'extension_error', 'error'],
        [refused, 'timeout', 'error'],
    ];
    for (const [meta, liveStatus, bucket] of cases) {
        const live = summarizeChatLiveEvent({ type: 'tool_call_finished', task_id: 't', tool: 'ext_stopper_send',
            invocation_id: 'i', is_error: true, status: liveStatus, tool_result_meta: meta, result_preview: preview });
        const seen = { key: 'tool:t:i', tool: 'ext_stopper_send', receipt: false, live: false, fact: 'settled',
            recorded_status: liveStatus, status: bucket, hostError: false, ts: '2026-11-17T01:00:20Z', tool_result_meta: meta, result_preview: preview };
        const view = refusedStepView(seen, seen.key, summarizeChatLiveEvent);
        assert.deepEqual([view.phase, view.headline, view.body, view.fullBody, view.dedupeKey],
            [live.phase, live.headline, live.body, live.fullBody, live.dedupeKey],
            `live and replay agree for ${liveStatus}/${meta.control_reason}`);
        assert.equal(view.visible, true);
        assert.equal(view.sourceTs, seen.ts);
    }
    // No host mark, no row; a settled `ok` folds even with the mark; no key, no row.
    assert.equal(refusedStepView({ key: 'tool:t:i', status: 'error', result_preview: preview }, 'tool:t:i'), null);
    assert.equal(refusedStepView({ key: 'tool:t:i', status: 'ok', tool_result_meta: refused }, 'tool:t:i'), null,
        'a settled ok folds; only a failed or refused call grew a live row');
    assert.equal(refusedStepView({ key: 'tool:t:i', status: 'held', tool_result_meta: refused }, ''), null);
});
