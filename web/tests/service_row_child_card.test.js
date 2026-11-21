// #1868 R6: a service-finalization checkpoint (`task_checkpoint` carrying the host's own
// `card_row_id`) registers ONCE on the card it belongs to — a helper's card included — whatever
// the order of the helper's terminal row and the service event, live and on reload. Its key is
// the host's own row id (`cardrow|service:…`): it never revives a finished card, never repaints
// its phase, title or meta, and never adds a second row on redelivery. A launch's later file
// result is the same row's next `card_row_revision`: it revises the row where it stands, on a
// running card and on a finished one, and a late copy of an earlier revision never takes it back.
// Everything below drives the real Chat module with the producer's own envelopes. A helper's
// timeline is drawn when its Details opens, so each check opens it before reading the rows.
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, parseRowMarkup, restoreDom, walkCard } from './chat_dom_fixture.js';

const ROOT = 'root-turn';
const KID = 'kid-turn';
const LINE = 'Background service stopped: paper (abc1234567) · at task end · log saved';
const LATE_WORDS = 'late words after the child ended';
const T = (n) => `2026-11-20T10:00:0${n}Z`;
// The producer's row (`service_finalization._checkpoint`): one per (task, launch), revision 1 at its confirmed end ...
const checkpoint = (text, { kind = 'services_stopped', row = 'service:launch-abc1234567', revision = 1 } = {}) => ({
    checkpoint_kind: kind, role: 'system', system_type: 'task_checkpoint', card_row_id: row, card_row_revision: revision,
    detail: text, content: text, text });
// ... on its live frame (`loop_messages._emit_checkpoint_event`): a `task_checkpoint` log event with the lineage the
// child's own event endpoint stamps (`log_addressing.TASK_LINEAGE_FIELDS`) ...
const LINEAGE = { root_task_id: ROOT, parent_task_id: ROOT, delegation_role: 'subagent' };
const serviceRow = (patch = {}) => ({ type: 'task_checkpoint', ts: T(3), task_id: KID, ...LINEAGE, ...checkpoint(LINE), ...patch });
// ... and as the room's history returns any progress row (`gateway.history`): no event type, its own place.
const stored = ({ type: _event, ...row }, offset) => ({ ...row, chat_id: 1, is_progress: true,
    history_id: `progress:${offset}`, history_position: { source: 'progress', offset } });
const CHILD = { delegation_role: 'subagent', subagent_task_id: KID, parent_task_id: ROOT, subagent_role: 'scout' };
const childFrame = (patch = {}) => ({ role: 'assistant', is_progress: true, task_id: KID, ts: T(1), ...CHILD, ...patch });
const rootFrame = (patch = {}) => ({ task_id: ROOT, role: 'assistant', is_progress: true, content: 'Root working', ts: T(0), ...patch });
// One launch's stop whose declared file was missing, the same launch's retried capture (its row's revision 2), and a
// second launch of the same name (`service_finalization._launch_line`).
const STOPPED = 'Background service stopped: paper (abc1234567) · at task end · log saved · output 1 absent';
const UPDATED = 'Background service outputs updated: paper (abc1234567) · ended · log saved · output 1 saved: preview.txt';
const SECOND = 'Background service stopped: paper (def7654321) · requested stop · log saved';
const first = serviceRow({ ts: T(3), ...checkpoint(STOPPED) });
const second = serviceRow({ ts: T(4), ...checkpoint(SECOND, { row: 'service:launch-def7654321' }) });
const retried = serviceRow({ ts: T(5), ...checkpoint(UPDATED, { kind: 'service_outputs_updated', revision: 2 }) });

// The flat fixture's querySelector does not descend; the card's chip, title and meta do.
const originalQuery = ElementStub.prototype.querySelector;
after(() => { ElementStub.prototype.querySelector = originalQuery; });
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};
// A rendered row's text is assertable: each parsed child keeps its own markup.
parseRowMarkup();

function fixture(history = []) {
    let historyReads = 0;
    const env = installDom(async (url) => {
        const isHistory = String(url).startsWith('/api/chat/history');
        if (isHistory) historyReads++;
        return { ok: true, json: async () => (isHistory
            ? { messages: history, window: { complete: true } } : { active_direct_turns: [] }) };
    });
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const card = (id) => walkCard(messages, id);
    // A card's own rows: a nested child card's lines belong to the child, never to its parent.
    const owned = (node, root) => (node?.children || []).flatMap((child) => (child !== root && child.classList?.contains('chat-live-card')
        ? [] : [...(child.classList?.contains('chat-live-line') ? [child] : []), ...owned(child, root)]));
    const rows = (id) => owned(card(id), card(id));
    const nodes = (node) => [node, ...(node?.children || []).flatMap(nodes)];
    return {
        instance, messages, card, rows,
        // Every card of a task in the feed: a frame that outran its lineage would mint a root-shaped twin.
        cards: (id) => nodes(messages).filter((node) => node.dataset?.taskId === id && node.classList?.contains('chat-live-card')),
        // The card's note count ('' below two notes): how many rows its record holds, drawn or not.
        notes: (id) => card(id)?.querySelector('[data-live-count]')?.textContent || '',
        activity: (id) => card(id)?.querySelector('[data-live-activity]')?.textContent || '',
        historyReads: () => historyReads,
        said: (id, text = LINE) => rows(id).filter((row) => row.innerHTML.includes(text)),
        // The launch sentences the card's rows say, in the card's order.
        told: (id) => rows(id).flatMap((row) => [STOPPED, UPDATED, SECOND].filter((text) => row.innerHTML.includes(text))),
        // Details, the card's one disclosure, as its button is pressed; a helper draws its timeline then.
        open: (id) => { if (card(id).dataset.expanded !== '1') card(id).querySelector('[data-live-disclosure]').click(); },
        phase: (id) => card(id)?.querySelector('[data-live-phase]')?.textContent || '',
        title: (id) => card(id)?.querySelector('[data-live-title]')?.textContent || '',
        meta: (id) => card(id)?.querySelector('[data-live-meta]')?.innerHTML || '',
        chat: (row) => handlers.get('chat')({ chat_id: 1, ...row }),
        fire: (type, row) => handlers.get(type)({ chat_id: 1, ...row }),
        log: (row) => handlers.get('log')({ chat_id: 1, data: row }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const runningChild = () => childFrame({ content: 'Reading the spec.', narration: true });
const terminalChild = () => childFrame({ ts: T(2), subagent_event: 'completed', status: 'completed', result: 'Child result' });

// The live order that loses the fact: the helper's terminal row lands first.
test('a service row landing after the helper\'s terminal row registers once on its card', () => {
    const f = fixture();
    try {
        f.chat(rootFrame());
        f.chat(runningChild());
        f.chat(terminalChild());
        const card = f.card(KID);
        assert.ok(card, 'the child frames mint the child card');
        assert.equal(card.dataset.finished, '1', 'the terminal frame finishes the child card');
        f.open(KID);
        const [phase, title, meta] = [f.phase(KID), f.title(KID), f.meta(KID)];
        f.log(serviceRow());
        assert.equal(f.said(KID).length, 1, 'the service fact is one row of the child card');
        f.log(serviceRow());
        assert.equal(f.said(KID).length, 1, 'redelivery adds nothing');
        assert.equal(f.card(KID).dataset.finished, '1', 'the card stays finished');
        assert.deepEqual([f.phase(KID), f.title(KID), f.meta(KID)], [phase, title, meta],
            'the row states no phase, title or meta of its own');
        assert.equal(f.said(ROOT).length, 0, 'the fact stays on the card its task names');
    } finally { f.close(); }
});

// The other order must land the same ONE row.
test('the service row arriving before the terminal frame is the same one row', () => {
    const f = fixture();
    try {
        f.chat(rootFrame());
        f.chat(runningChild());
        f.open(KID);
        f.log(serviceRow());
        assert.equal(f.said(KID).length, 1, 'a running card takes the row');
        f.chat(terminalChild());
        assert.equal(f.said(KID).length, 1, 'the terminal frame neither duplicates nor drops it');
        assert.equal(f.card(KID).dataset.finished, '1');
    } finally { f.close(); }
});

// A reopened room reads the same ONE row, whichever order the stored rows carry.
for (const order of ['terminal-first', 'service-first']) {
    test(`a reloaded helper card reads one service row (${order})`, async () => {
        const rows = {
            'terminal-first': [runningChild(), terminalChild(), serviceRow()],
            'service-first': [runningChild(), serviceRow(), terminalChild()],
        }[order];
        const f = fixture([stored(rootFrame(), 0), ...rows.map((row, index) => stored(row, index + 1))]);
        try {
            await f.instance.refreshHistory({ revision: 1 });
            assert.equal(f.card(KID)?.dataset.finished, '1', 'the stored terminal settles the card');
            f.open(KID);
            assert.equal(f.said(KID).length, 1, 'one row after the reload');
            f.fire('open', { previouslyConnected: true });
            await new Promise(setImmediate);
            await new Promise(setImmediate);
            assert.ok(f.historyReads() >= 2, 'the reconnect refetched history');
            assert.equal(f.said(KID).length, 1, 'the refetch keeps one row');
        } finally { f.close(); }
    });
}

// A late row on a finished ROOT card (the R5 card) registers by the same rule, live.
test('a late service row registers once on a finished root card too', () => {
    const f = fixture();
    try {
        f.chat(rootFrame());
        f.log({ task_id: ROOT, ts: T(1), type: 'task_done', status: 'completed' });
        assert.equal(f.card(ROOT).dataset.finished, '1');
        const row = { type: 'task_checkpoint', ts: T(3), task_id: ROOT, root_task_id: ROOT, ...checkpoint(LINE) };
        f.log(row);
        assert.equal(f.said(ROOT).length, 1, 'the host-keyed row still registers on the closed card');
        assert.equal(f.phase(ROOT), 'Done', 'and the card keeps its own phase');
        f.log(row);
        assert.equal(f.said(ROOT).length, 1, 'redelivery adds nothing');
    } finally { f.close(); }
});

// The preserved rule: an ordinary late frame still never revives a terminal child.
test('an ordinary late checkpoint or progress frame does not revive the finished helper card', () => {
    const f = fixture();
    try {
        f.chat(rootFrame());
        f.chat(runningChild());
        f.chat(terminalChild());
        f.open(KID);
        const before = f.rows(KID).length;
        assert.ok(before > 0, 'the open helper shows its own rows');
        f.log({ task_id: KID, ts: T(4), ...CHILD, type: 'task_checkpoint', checkpoint_kind: 'context_view', status: 'no_op' });
        f.log({ task_id: KID, ts: T(5), ...CHILD, type: 'send_message', is_progress: true, content: LATE_WORDS });
        assert.equal(f.rows(KID).length, before, 'no row is added');
        assert.equal(f.said(KID, 'Context inspected').length, 0);
        assert.equal(f.said(KID, LATE_WORDS).length, 0);
        assert.equal(f.card(KID).dataset.finished, '1');
    } finally { f.close(); }
});

// The stub cannot repaint a row patched in place, but a helper draws its timeline when Details opens: the frames all
// land first, so the rows read the card's final state. The first launch's row stands where its stop landed (before the
// second launch) and says the retried file; the late copy of its first revision neither repaints it nor adds a row.
test('a launch\'s retried file result revises its row in place on a running helper card; a late first revision does not', () => {
    const f = fixture();
    try {
        f.chat(rootFrame());
        f.chat(runningChild());
        [first, second, retried, first].forEach((row) => f.log(row));
        f.open(KID);
        assert.notEqual(f.card(KID).dataset.finished, '1', 'the helper is still running');
        assert.deepEqual(f.told(KID), [UPDATED, SECOND]);
    } finally { f.close(); }
});

// The same row across the card's two states: its stop lands while the helper runs, its retried file after it ended.
test('a finished helper card takes a launch\'s retried file result into the row its stop opened; a late first revision does not', () => {
    const f = fixture();
    try {
        f.chat(rootFrame());
        f.chat(runningChild());
        [first, second].forEach((row) => f.log(row));
        f.chat(terminalChild());
        assert.equal(f.card(KID).dataset.finished, '1');
        [retried, first].forEach((row) => f.log(row));
        f.open(KID);
        assert.deepEqual(f.told(KID), [UPDATED, SECOND]);
        assert.equal(f.card(KID).dataset.finished, '1', 'the card stays finished');
    } finally { f.close(); }
});

// A reopened room reads each launch's newest revision once, even when a copy of the first is stored after it.
test('a reloaded helper card reads each launch\'s newest revision once', async () => {
    const rows = [runningChild(), terminalChild(), first, second, retried, first];
    const f = fixture([stored(rootFrame(), 0), ...rows.map((row, index) => stored(row, index + 1))]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        f.open(KID);
        assert.deepEqual(f.told(KID), [UPDATED, SECOND]);
        f.fire('open', { previouslyConnected: true });
        await new Promise(setImmediate);
        await new Promise(setImmediate);
        assert.ok(f.historyReads() >= 2, 'the reconnect refetched history');
        assert.deepEqual(f.told(KID), [UPDATED, SECOND], 'and the refetch keeps them');
    } finally { f.close(); }
});

// A helper's service row can be the first frame of that helper this page sees. The lineage it carries places it on the
// helper's own card, registering the helper when nothing else did: there it stays its launch's one row, whichever
// frame of the helper came first, and it is never the helper's words, a root card's row, or the card's title or meta.
for (const order of ['checkpoint first', 'helper first']) {
    test(`a helper's service row is its launch's one row on that helper's card (${order})`, () => {
        const f = fixture();
        try {
            f.chat(rootFrame());
            if (order === 'checkpoint first') {
                f.log(first);
                assert.equal(f.card(KID)?.dataset.parentTaskId, ROOT, 'the row alone registers the helper under its parent');
                assert.ok(walkCard(f.card(ROOT), KID), 'inside the parent\'s card');
                assert.equal(f.notes(KID), '', 'one note: the row, not the row and the same words as the helper\'s');
            }
            f.chat(runningChild());
            if (order === 'helper first') f.log(first);
            assert.equal(f.notes(KID), '2 notes', 'once the helper is known: its own words and the row');
            [retried, first].forEach((row) => f.log(row));
            assert.equal(f.notes(KID), '2 notes', 'a revision and a late copy add no row');
            f.open(KID);
            assert.deepEqual(f.told(KID), [UPDATED], 'the launch is one row, saying its retried file');
            assert.equal(f.said(KID, 'Reading the spec.').length, 1);
            assert.equal(f.rows(KID).length, 2, 'and no row keeps the first sentence');
            assert.equal(f.cards(KID).length, 1, 'one card for the helper');
            assert.deepEqual(f.told(ROOT), [], 'no row of the root says it');
            for (const [part, shown] of [['title', f.title(KID)], ['meta', f.meta(KID)], ['line', f.activity(KID)]]) {
                assert.ok(![STOPPED, UPDATED].some((text) => shown.includes(text)), `the card's ${part} does not say it`);
            }
        } finally { f.close(); }
    });
}
