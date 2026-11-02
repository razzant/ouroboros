// A completed child's result is its report, written for its parent. Where the owner reads it the card
// says so: "Report to parent" before the collapsed preview and in place of the [RESULT] marker over the
// full text. The words, the model, the expansion and a copy of the preview stay what they were; another
// ending's result keeps the neutral marker (it may be host text, not the child's report).
import assert from 'node:assert/strict';
import test from 'node:test';

import { buildTimelineItemHtml, clearStickyCardState, renderCollapsedActivity } from '../modules/chat_activity.js';
import { createChatInstance } from '../modules/chat.js';
import { childReportLabel, summarizeChatLiveEvent } from '../modules/log_events.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const REPORT = '## summary\n\nThe shared class is a false word about work that already ended, not a missing status machine.';
const child = (overrides = {}) => ({
    type: 'send_message', is_progress: true, chat_id: 2, task_id: 'child-1', delegation_role: 'subagent',
    parent_task_id: 'root-1', root_task_id: 'root-1', subagent_task_id: 'child-1', subagent_role: '',
    subagent_event: 'completed', status: 'completed', model: 'x-ai/grok-4.7', result: REPORT,
    ts: '2026-10-25T14:12:58Z', ...overrides,
});

test('a completed child labels its report before the preview and over the full text', () => {
    const view = summarizeChatLiveEvent(child());
    assert.equal(view.activityLead, 'Report to parent');
    assert.ok(view.fullBody.startsWith(`## Report to parent\n${REPORT}`), view.fullBody);
    assert.doesNotMatch(view.fullBody, /\[RESULT\]/);
    // The preview, the timeline row and the model keep their words.
    assert.equal(view.activityPreview, view.body);
    assert.doesNotMatch(view.activityPreview, /Report to parent/);
    assert.equal(view.model, 'x-ai/grok-4.7');
});

test('the collapsed preview of a report is plain text: list and table markers go, the full report keeps them', () => {
    const report = '## summary\n\n- first finding\n- second finding\n\n| check | result |\n|---|---|\n| tests | passed |';
    const view = summarizeChatLiveEvent(child({ result: report }));
    assert.equal(view.activityPreview, 'summary — first finding second finding check result tests passed');
    assert.ok(view.fullBody.endsWith(report), view.fullBody);
    // A fenced list is code and keeps its marker; the words of the report stay the face.
    const fenced = summarizeChatLiveEvent(child({ result: 'Ran:\n```sh\n- not a bullet\n```\n- done' }));
    assert.match(fenced.activityPreview, /- not a bullet/);
    assert.match(fenced.activityPreview, /``` done$/);
});

test('another ending, or a child whose line states a stop cause, keeps the neutral marker and no label', () => {
    const failed = summarizeChatLiveEvent(child({ subagent_event: 'failed', status: 'failed',
        result: '⚠️ The task could not finish.', error: 'executor unavailable' }));
    assert.equal(failed.activityLead, undefined);
    assert.match(failed.fullBody, /\[RESULT\]\n⚠️ The task could not finish\./);
    const stopped = summarizeChatLiveEvent(child({ reason_code: 'owner_requested_finalization',
        cancel_origin: { source: 'http_single', scope: 'single', stop_policy: 'immediate' } }));
    assert.equal(stopped.activityLead, undefined, 'the stop cause leads that line, not the label');
    assert.deepEqual(['completed', 'failed', 'cancelled', ''].map(childReportLabel), ['Report to parent', '', '', '']);
});

test('the expanded line leads with the label on its own line, as a timeline label (semibold, then a copyable break)', () => {
    const view = summarizeChatLiveEvent(child());
    const priorDocument = globalThis.document;
    globalThis.document = { createElement: () => ({ textContent: '', get innerHTML() { return this.textContent; } }) };
    try {
        const item = { ...view, lineKey: 'k1', fullBody: view.fullBody };
        const html = buildTimelineItemHtml(item, { expandedLineKeys: new Set(['k1']), groupId: 'child-1' });
        assert.match(html, /chat-live-line-body[^>]*><strong class="md-h2">Report to parent<\/strong><br>/);
        assert.doesNotMatch(html, /\[RESULT\]/);
        const closed = buildTimelineItemHtml(item, { expandedLineKeys: new Set(), groupId: 'child-1' });
        assert.doesNotMatch(closed, /Report to parent/, 'the collapsed row keeps its preview words');
    } finally {
        globalThis.document = priorDocument;
    }
});

test('the collapsed line carries the label as data-lead, so its fit counts it and a copy keeps only the words', () => {
    const el = { dataset: {}, textContent: '', isConnected: false, hasAttribute: () => false, removeAttribute() {} };
    const record = { activityEl: el, isSubagent: true, root: { dataset: { subagent: '1' } }, collapsedActivityLead: 'Report to parent' };
    renderCollapsedActivity(record, 'summary — The shared class is a false word');
    assert.equal(el.dataset.lead, 'Report to parent');
    assert.equal(el.textContent, 'summary — The shared class is a false word');
    record.collapsedActivityLead = '';
    renderCollapsedActivity(record, 'summary — The shared class is a false word');
    assert.equal(Object.hasOwn(el.dataset, 'lead'), false, 'a later line without the label drops it');
    record.collapsedActivityLead = 'Report to parent';
    renderCollapsedActivity(record, '');
    assert.equal(Object.hasOwn(el.dataset, 'lead'), false, 'an empty line draws no label');
    renderCollapsedActivity(record, 'again');
    clearStickyCardState(record);
    assert.equal(Object.hasOwn(el.dataset, 'lead'), false);
    assert.equal(record.collapsedActivityLead, '');
});

test('a replayed child card shows the label on its collapsed line', async () => {
    const rows = [
        { chat_id: 2, role: 'system', is_progress: true, content: 'Reading the brief.', narration: true,
          ts: '2026-10-25T13:35:40Z', task_id: 'root-1' },
        { ...child(), role: 'system', content: '', text: '', narration: false },
    ];
    const { prior, mount } = installDom(async (url) => (String(url).startsWith('/api/chat/history')
        ? { ok: true, json: async () => ({ messages: rows, window: { complete: true } }) }
        : { ok: true, json: async () => ({ active_direct_turns: [] }) }));
    const ws = { on() { return () => {}; }, isConnected: () => true, send() {} };
    let generation = 0;
    const stateSnapshots = { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} };
    const instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true });
    try {
        await instance.refreshHistory({ revision: 1 });
        await new Promise((resolve) => setTimeout(resolve, 25));
        const walk = (node) => node.dataset?.taskId === 'child-1' ? node
            : (node.children || []).reduce((hit, next) => hit || walk(next), null);
        const card = walk(globalThis.document.byId.get('chat-messages'));
        assert.ok(card, 'the child card is minted inside its root');
        const line = card.querySelector('[data-live-activity]');
        assert.equal(line?.dataset.lead, 'Report to parent');
        assert.doesNotMatch(line?.textContent || '', /Report to parent/);
        assert.match(line?.textContent || '', /The shared class is a false word/);
    } finally {
        instance.destroy();
        restoreDom(prior);
    }
});

for (const [name, fields, report] of [
    ['completed', {}, true],
    ['completed with warnings', { outcome_axes: { execution: { status: 'best_effort' } } }, true],
    ['failed execution behind completed', { outcome_axes: { execution: { status: 'failed' } } }, false],
    ['failed objective behind completed', { outcome_axes: { objective: { status: 'fail' } } }, false],
    ['cancelled', { status: 'cancelled', subagent_event: 'cancelled' }, false],
    ['Stop now behind completed', { reason_code: 'owner_requested_finalization',
        cancel_origin: { source: 'http_graceful', stop_policy: 'immediate' } }, false],
]) {
    test(`Show full keeps the preview's heading for ${name}`, async () => {
        const result = 'REPORT_HEAD\n' + 'The retained answer has a complete paragraph.\n'.repeat(120) + 'REPORT_TAIL';
        const trace = 'TRACE_HEAD\n' + 'Read the recorded evidence.\n'.repeat(160) + 'TRACE_TAIL';
        assert.ok(result.length > 4000 && trace.length > 4000);
        const detail = { ...child(fields), result, trace_summary: trace };
        const frame = { ...detail, role: 'system', content: '', text: '',
            result: result.slice(0, 4000), result_truncated: true,
            trace_summary: trace.slice(0, 4000), trace_summary_truncated: true };
        const preview = summarizeChatLiveEvent(frame);
        assert.equal(preview.fullBody.startsWith('## Report to parent'), report);
        const rows = [{ chat_id: 2, role: 'system', is_progress: true, content: 'Reading the brief.',
            narration: true, ts: '2026-10-25T13:35:40Z', task_id: 'root-1' }, frame];
        let detailReads = 0;
        const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () => {
            if (String(url).startsWith('/api/chat/history')) return { messages: rows, page_cursor: 'recent',
                has_more: false, next_cursor: null, window: { complete: true } };
            if (String(url) === '/api/tasks/child-1') { detailReads++; return detail; }
            return { active_direct_turns: [] };
        } }));
        // This stub keeps parsed nodes flat. Observe the real renderer's markup;
        // native DOM layout/patching is covered by the WebKit flow.
        let rendered = '';
        const html = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
        Object.defineProperty(ElementStub.prototype, 'innerHTML', { ...html, set(value) {
            if (/^\s*<div\s+class="chat-live-line\s/.test(String(value)) && /REPORT_HEAD|Show full/.test(String(value))) {
                rendered = String(value);
            }
            html.set.call(this, value);
        } });
        const ws = { on() { return () => {}; }, isConnected: () => true, send() {} };
        let generation = 0;
        const stateSnapshots = { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} };
        const instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true });
        try {
            await instance.refreshHistory({ revision: 1 });
            const card = walkCard(globalThis.document.byId.get('chat-messages'), 'child-1');
            assert.ok(card);
            card.querySelector('[data-live-disclosure]').click();
            await new Promise(resolve => setTimeout(resolve, 0)); // finish the synchronous fixture's review-detail promises
            const timeline = card.querySelector('[data-live-timeline]');
            assert.match(rendered, /Show full/);
            const toggle = () => {
                const line = timeline.querySelector('.chat-live-line.expandable');
                // The DOM fixture does not implement general closest/bubbling; dispatch the real listener.
                const target = { closest: selector => selector === '.chat-live-line.expandable' ? line : null };
                for (const fn of timeline.listeners.get('click') || []) fn({ target, stopPropagation() {} });
            };
            const priorReads = detailReads;
            toggle();
            for (let tick = 0; tick < 30 && !rendered.includes('REPORT_TAIL'); tick++) {
                await new Promise(resolve => setTimeout(resolve, 5));
            }
            assert.equal(detailReads, priorReads + 1, 'Show full fetches the already-addressed task detail');
            assert.match(rendered, /REPORT_TAIL/);
            assert.match(rendered, /TRACE_TAIL/);
            assert.equal(rendered.includes('Report to parent'), report);
            if (!report) assert.match(rendered, /\[RESULT\]/);
            const fetched = rendered;
            toggle();
            toggle();
            assert.equal(detailReads, priorReads + 1, 'reopening uses the fetched body');
            assert.equal(rendered, fetched);
        } finally {
            instance.destroy();
            restoreDom(prior);
            Object.defineProperty(ElementStub.prototype, 'innerHTML', html);
        }
    });
}
