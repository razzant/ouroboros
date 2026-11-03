import assert from 'node:assert/strict';
import test from 'node:test';
import { taskTerminalSummary, taskTerminalPhase } from '../modules/log_events.js';
import { buildTimelineItemHtml, candidateRemainderBody, syncCandidateRemainderItem } from '../modules/chat_activity.js';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const id = 'candidate-task';
const fact = (state = 'known', count = 2) => ({ state, count, surfaces: [{ root: '/saved/candidate', commit: 'abc123', paths: count ? ['src/one.py', 'src/two.py'] : [] }] });
const detail = (remainder) => ({ task_id: id, status: 'completed', outcome_final: true, outcome_axes: {
    lifecycle: { status: 'completed' }, execution: { status: 'ok', mutation_attribution: { uncommitted_remainder: remainder } },
    objective: { status: 'pass' }, review: { status: 'not_requested' },
} });

test('one short fact, zero omitted, unknown not zero, no outcome mutation', () => {
    for (const [value, expected] of [[fact(), /2 uncommitted files remain/], [fact('known', 0), /^$/],
        [fact('unknown', null), /could not be checked/]]) {
        const record = detail(value);
        const before = JSON.stringify(record);
        const summary = taskTerminalSummary(record);
        assert.match(summary.activityPreview || '', expected);
        assert.equal(taskTerminalPhase(record), 'done');
        assert.equal(JSON.stringify(record), before);
    }
});

test('paths and HEAD use the existing keyed Details item, clearing known zero', () => {
    const record = { groupId: id, items: [] };
    assert.equal(syncCandidateRemainderItem(record, detail(fact())), true);
    assert.equal(syncCandidateRemainderItem(record, detail(fact())), false);
    assert.match(record.items[0].body, /HEAD: abc123\n\nFiles:\nsrc\/one.py\nsrc\/two.py/);
    assert.equal(syncCandidateRemainderItem(record, {}), false, 'old sparse event does not erase a fact');
    assert.equal(syncCandidateRemainderItem(record, detail(fact('known', 0))), true);
    assert.equal(record.items.length, 0);
});

test('Details keeps file names literal and separate after a record round trip', () => {
    const record = { groupId: id, items: [], expandedLineKeys: new Set() };
    const value = fact();
    value.surfaces[0].paths = ['two  spaces.py', '`code`/**name**.py', '<img src=x>.py', '[link](https://example.com).py'];
    syncCandidateRemainderItem(record, detail(value));
    const item = JSON.parse(JSON.stringify(record.items[0]));
    const { prior } = installDom();
    try {
        const html = buildTimelineItemHtml(item, record);
        assert.match(html, /data-body-format="text"/);
        assert.ok(html.includes(value.surfaces[0].paths.slice(0, 2).join('\n')));
        assert.ok(html.includes('&lt;img src=x&gt;.py'));
        assert.doesNotMatch(html, /<code|<strong|<img|<a /);
    } finally { restoreDom(prior); }
});

test('bounded terminal fact loads all exact paths from the existing task detail, never a newer snapshot', () => {
    const paths = Array.from({ length: 5000 }, (_, i) => `generated/${i}/<literal> **file**.py`);
    const compact = { state: 'known', count: 5000, paths_omitted: 5000, task_id: 'owning-root',
        snapshot_at: '2026-10-25T12:00:00Z', baseline_hash: 'baseline' };
    const source = { mutation_evidence: { terminal_candidate_snapshot: { captured_at: compact.snapshot_at,
        baseline_hash: compact.baseline_hash, surfaces: [{ canonical_root: '/saved/candidate', base_commit: 'abc123', uncommitted_paths: paths }] } } };
    const record = { groupId: id, items: [] };
    syncCandidateRemainderItem(record, detail(compact));
    assert.equal(record.items[0].fullRef, 'owning-root');
    assert.equal(record.items[0].truncated, true);
    assert.match(record.items[0].body, /5000 recorded file paths are not loaded/);
    assert.equal(candidateRemainderBody({}, compact), null, 'no invented exact data on an unavailable read');
    const text = candidateRemainderBody(source, compact);
    assert.equal(text, `Candidate: /saved/candidate\nHEAD: abc123\n\nFiles:\n${paths.join('\n')}`);
    syncCandidateRemainderItem(record, { ...detail(compact), ...source });
    assert.equal(record.items[0].body, text);
    assert.equal(record.items[0].truncated, false);
    syncCandidateRemainderItem(record, detail(compact));
    assert.equal(record.items[0].body, text, 'a later compact frame does not erase loaded details');
    source.mutation_evidence.terminal_candidate_snapshot.captured_at = '2026-10-25T13:00:00Z';
    assert.equal(candidateRemainderBody(source, compact), null, 'a different snapshot is not the summary source');
});

test('the remainder sentence is terminated before the existing task-time note', () => {
    const summary = taskTerminalSummary({ ...detail(fact()), system_type: 'task_summary',
        terminal_time: { source: 'unknown' }, ts: '2026-10-25T12:00:00Z' });
    assert.match(summary.body, /2 uncommitted files remain in the saved candidate\.\nTask end time not recorded/);
});

for (const replay of [false, true]) test(`real chat card ${replay ? 'cold replay' : 'live terminal'} keeps Done and a visible remainder`, async () => {
    const rows = [
        { task_id: id, chat_id: 1, role: 'system', is_progress: true, content: 'Saving the candidate', ts: '2026-10-25T12:00:00Z' },
        { ...detail(fact()), chat_id: 1, role: 'system', system_type: 'task_summary', content: 'Finished.',
            task_terminal_status: 'completed', ts: '2026-10-25T12:01:00Z' },
    ];
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/tasks/') ? detail(fact())
            : String(url).startsWith('/api/chat/history') ? { messages: replay ? rows : [] } : { active_direct_turns: [] } }));
    const handlers = new Map();
    let instance;
    try {
        instance = createChatInstance({ ws: { on(type, fn) { handlers.set(type, fn); return () => {}; }, isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
            chatId: 1, idPrefix: 'chat', mountEl: mount });
        if (replay) await instance.refreshHistory({ revision: 1 });
        else { await new Promise(r => setTimeout(r, 0)); rows.forEach(row => handlers.get('chat')(row)); }
        const card = walkCard(globalThis.document.byId.get('chat-messages'), id);
        assert.equal(card.dataset.finished, '1');
        assert.equal(card.querySelector('[data-live-phase]').textContent, 'Done');
        const activity = card.querySelector('[data-live-activity]');
        assert.match(activity.textContent, /2 uncommitted files remain in the saved candidate/);
        const timeline = card.querySelector('[data-live-timeline]');
        assert.equal(timeline.children.filter(node => node.dataset.liveLineKey === `candidate-remainder-${id}`).length, 1);
    } finally { instance?.destroy(); restoreDom(prior); }
});
