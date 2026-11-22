import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { cardMetaKeys } from '../modules/chat_activity.js';
import { summarizeChatLiveEvent, taskReasonDetail, taskTerminalSummary } from '../modules/log_events.js';

test('recorded cancellation: Python/browser parity including absent cause and punctuation', () => {
    const cases = JSON.parse(readFileSync(new URL('./fixtures/cancel_cause_parity.json', import.meta.url)));
    for (const { record, text } of cases) {
        assert.equal(taskReasonDetail(record), text, JSON.stringify(record));
        assert.equal(taskTerminalSummary({ ...record, type: 'task_done' }).body, text);
        assert.equal(taskTerminalSummary({ ...record, system_type: 'task_summary' }).body, text);
    }
    assert.equal(taskReasonDetail({ status: 'cancelled', cancel_origin: { reason: '🙂'.repeat(200) } }),
        '🙂'.repeat(159) + '… (preview; the full reason is kept with the task)');
    assert.equal(taskReasonDetail({ status: 'cancelled', cancel_origin: { reason: '🙂'.repeat(160) } }),
        '🙂'.repeat(160) + '.');
});

test('child live and replay use genuine lineage and keep saved work inspectable', () => {
    const origin = { source: 'cascade_descendant', requested_by: 'root' };
    const carried = cardMetaKeys({ cancel_origin: origin });
    assert.deepEqual(carried.cancel_origin, origin);
    for (const frame of [
        { subagent_event: 'cancelled' },
        { subagent_event: 'completed', outcome_axes: { lifecycle: { status: 'cancelled' } } },
    ]) {
        const view = summarizeChatLiveEvent({
            type: 'send_message', is_progress: true, delegation_role: 'subagent',
            parent_task_id: 'root', subagent_task_id: 'child', result: 'Saved partial work',
            ...carried, ...frame,
            outcome_axes: { ...frame.outcome_axes, execution: { earlier_stop: 'provider_failure' } },
        });
        assert.equal(view.phase, 'cancelled');
        assert.equal(view.body, 'Stopped with the task tree it belongs to · Stopped with its parent task · Earlier recorded stop: The model provider failed to answer, so the task could not finish.');
        assert.equal(view.activityPreview, view.body);
        assert.match(view.fullBody, /Saved partial work/);
        assert.doesNotMatch(view.body, /initiator:|root/);
    }
});

test('retained origin does not replace a non-cancelled child frame', () => {
    for (const subagent_event of ['running', 'failed']) {
        const view = summarizeChatLiveEvent({
            type: 'send_message', is_progress: true, delegation_role: 'subagent',
            parent_task_id: 'root', subagent_task_id: 'child', subagent_event,
            result: 'Saved partial work', error: subagent_event === 'failed' ? 'Worker failed' : '',
            status: 'cancelled', cancel_origin: { source: 'http_single' },
        });
        assert.doesNotMatch(view.body, /Stopped from/);
        assert.equal(view.phase, subagent_event === 'failed' ? 'error' : 'working');
    }
});

test('a saved end row says when the task ended when its line was added later', () => {
    const at = value => new Date(value).toLocaleString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    const row = (terminal_time, ts = '2026-09-26T19:21:43+00:00') => ({ task_id: 'late', system_type: 'task_summary',
        status: 'failed', ts, text: '', ...(terminal_time === undefined ? {} : { terminal_time }) });
    const known = { v: 1, source: 'executor_terminal', occurred_at: '2026-09-24T18:18:27+00:00', attempt: {} };
    assert.match(taskTerminalSummary(row(known)).body,
        new RegExp(`Task ended ${at(known.occurred_at)} · Notification added ${at('2026-09-26T19:21:43+00:00')}`));
    assert.match(taskTerminalSummary(row({ ...known, source: 'unknown', occurred_at: null })).body,
        /Task end time not recorded · Notification added /);
    for (const prompt of [row(known, '2026-09-24T18:18:40+00:00'), row(undefined), row(null)]) {
        assert.doesNotMatch(taskTerminalSummary(prompt).body, /Task end|Notification added/,
            'a prompt or legacy row keeps its ordinary terminal line');
    }
    assert.doesNotMatch(taskTerminalSummary({ ...row(known), system_type: '' }).body, /Notification added/,
        'a live task_done frame is not a saved notification');
});

test('an end row published an hour later across a DST fallback still says when the task ended', t => {
    const zone = process.env.TZ;
    t.after(() => { if (zone === undefined) delete process.env.TZ; else process.env.TZ = zone; });
    process.env.TZ = 'America/New_York';
    const at = (value, timeZoneName) => new Date(value).toLocaleString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName });
    const [ended, added] = ['2026-11-01T05:30:10Z', '2026-11-01T06:30:40Z'];
    assert.equal(at(ended), at(added), 'both instants read 01:30 on the local clock');
    const row = ts => ({ task_id: 'late', system_type: 'task_summary', status: 'failed', ts, text: '',
        terminal_time: { v: 1, source: 'executor_terminal', occurred_at: ended, attempt: {} } });
    const body = taskTerminalSummary(row(added)).body;
    assert.ok(body.includes(`Task ended ${at(ended, 'short')} · Notification added ${at(added, 'short')}`), body);
    assert.notEqual(at(ended, 'short'), at(added, 'short'), 'their zone names tell them apart');
    assert.doesNotMatch(taskTerminalSummary(row('2026-11-01T05:30:55Z')).body, /Task end|Notification added/,
        'the same minute still keeps the ordinary line');
});
