// The real Chat consumer for #1350: a delegated run's typed observation lands
// as its own row in the owning card. A child's own narration row survives it,
// the child's title and collapsed line stay the child's, a silent stretch is
// one row, and a root card keeps its title placeholder.
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const originalQuery = ElementStub.prototype.querySelector;
after(() => { ElementStub.prototype.querySelector = originalQuery; });
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

const TS = '2026-09-27T12:00:00Z';
const ROOT = 'root-1350';
const KID = 'kid-1350';
const lineage = { subagent_task_id: KID, parent_task_id: ROOT, root_task_id: ROOT,
    delegation_role: 'subagent', subagent_role: 'researcher' };

function fixture() {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: [], window: { complete: true } } : { active_direct_turns: [] } }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const nodes = (node) => [node, ...(node?.children || []).flatMap(nodes)];
    return {
        card: (id) => walkCard(messages, id),
        rows: (id) => nodes(walkCard(messages, id)).filter((n) => n.classList?.contains('chat-live-line')
            && !Object.hasOwn(n.dataset, 'delegatedFolded')),
        title: (id) => walkCard(messages, id)?.querySelector('[data-live-title]')?.textContent ?? null,
        activity: (id) => walkCard(messages, id)?.querySelector('[data-live-activity]')?.textContent ?? null,
        open: (id) => walkCard(messages, id).querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 }),
        emit: (row, seconds = 0) => handlers.get('chat')({ chat_id: 1, role: 'system', is_progress: true,
            ts: `2026-09-27T12:00:${String(seconds).padStart(2, '0')}Z`, ...row }),
        census: (rows) => instance.hydrateStateSnapshot({ active_chat_activities: rows,
            active_chat_activities_complete: true, supervisor_ready: true }, Infinity, ++generation),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const record = (task, after, through, parts, extra = {}) => ({ v: 1, task_id: task, run_id: 'run-9', after_seq: after,
    through_seq: through, source: { kind: 'run_events', read_through: through, events: through - after }, parts, ...extra });
const silent = (task, after, through) => record(task, after, through, [], {
    technical: { count: through - after, labels: [['Read', through - after]], seqs: [[after + 1, through]] } });

test('a child keeps its own narration row, title and collapsed line beside the executor rows', () => {
    const f = fixture();
    try {
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child queued', subagent_event: 'scheduled', ...lineage });
        f.emit({ role: 'assistant', task_id: KID, content: 'Watching the delegated run.', narration: true,
            subagent_event: 'progress', ...lineage }, 1);
        f.open(KID);
        const before = f.rows(KID).length;
        const title = f.title(KID);
        const said = record(KID, 0, 10, [{ kind: 'message', actor: 'claude/a01', text: 'Tracing the Telegram consumer.', seq: 4 },
            { kind: 'thinking', actor: 'claude/a01', text: 'maybe in plugin.py', seq: 5 }],
        { technical: { count: 8, labels: [['Read', 5], ['Bash', 3]], seqs: [[1, 3], [6, 10]] } });
        f.emit({ task_id: KID, content: '💬 🛰 delegated run run-9 @seq 10\n[claude/a01] Bash · tool_result', narration: false,
            subagent_event: 'progress', delegated_activity: said, ...lineage }, 2);
        assert.equal(f.rows(KID).length, before + 1, 'the executor observation is its own row, not the child narration row');
        for (const [after, through, second] of [[10, 12, 3], [12, 15, 4]]) {
            f.emit({ task_id: KID, content: `💬 🛰 delegated run run-9 @seq ${through}`, narration: false,
                subagent_event: 'progress', delegated_activity: silent(KID, after, through), ...lineage }, second);
        }
        assert.equal(f.rows(KID).length, before + 2, 'a silent stretch reads as one row');
        // A restarted observer re-reads (0, 15]: nothing it carries is shown twice.
        f.emit({ task_id: KID, content: '💬 🛰 delegated run run-9 @seq 15', narration: false, subagent_event: 'progress',
            delegated_activity: record(KID, 0, 15, said.parts, { technical: { count: 13, labels: [['Read', 13]],
                seqs: [[1, 3], [6, 15]] } }), ...lineage }, 5);
        assert.equal(f.rows(KID).length, before + 2, 'a re-read range adds no visible row');
        assert.equal(f.activity(KID), 'Watching the delegated run.', 'executor rows never take the collapsed line');
        assert.equal(f.title(KID), title, 'nor the child title');
    } finally { f.close(); }
});

test('a root card shows the executor row without letting a problem finish the task or claim the title', () => {
    const f = fixture();
    try {
        f.census([{ activity_id: ROOT, chat_id: 1, kind: 'managed_task', phase: 'working' }]);
        f.emit({ task_id: ROOT, content: 'plain host note', narration: false });
        const title = f.title(ROOT);
        f.emit({ task_id: ROOT, content: '💬 🛰 delegated run run-9 @seq 3', narration: false,
            delegated_activity: record(ROOT, 0, 3, [{ kind: 'problem', actor: 'claude/a01', label: 'Bash', text: 'exit 1', seq: 3 }]) }, 1);
        f.emit({ task_id: ROOT, content: '💬 🛰 delegated run run-9 @seq 5', narration: false,
            delegated_activity: record(ROOT, 3, 5, [{ kind: 'message', actor: 'claude/a01', text: 'exit 1', seq: 5 }]) }, 2);
        assert.equal(f.rows(ROOT).length, 3, 'each observation keeps its own row even when words repeat');
        assert.notEqual(f.card(ROOT).dataset.finished, '1', 'a failed step is not a finished or failed task');
        assert.equal(f.title(ROOT), title);
    } finally { f.close(); }
});
