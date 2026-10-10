// The host's card placement (`card_row`) puts a task-keyed System row inside
// that task's card, live and on replay, as ONE content-only timeline item keyed
// by the host's row identity. A row the page cannot place — no placement fact,
// or no card record for its task — keeps the standalone System row it has
// always had, because hiding it would lose the fact.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const CUSTODY_ROW = {
    chat_id: 2, role: 'system', system_type: 'custody_notice',
    card_row: 'timeline', card_row_id: 'final:t1:abc:custody_notice', task_id: 't1',
    content: 'Open delegated execution: run-x.\nPending delegated invocations: none observed.',
    ts: '2026-09-16T00:05:00Z',
};

const systemBubbles = (messages) => messages.children.filter((node) => node.classList.contains('chat-bubble')
    && node.classList.contains('system') && !node.classList.contains('typing-bubble'));

const walkNodes = (node) => [node, ...(node?.children || []).flatMap(walkNodes)];
const timelineLines = (card) => walkNodes(card).filter((node) => node.classList.contains('chat-live-line'));
const phasedLines = (card, phase) => timelineLines(card).filter((node) => node.classList.contains(phase));

// The fixture parses innerHTML flat, so a rendered timeline line keeps its
// classes but not its text; read the markup the renderer built for it instead.
function captureRenderedLines() {
    const doc = globalThis.document;
    const create = doc.createElement.bind(doc);
    const built = [];
    doc.createElement = (tag) => { const node = create(tag); built.push(node); return node; };
    return () => built.map((node) => node.innerHTML).filter((html) => html.includes('class="chat-live-line '));
}

function openCardChat(fetchImpl) {
    const { prior, mount } = installDom(fetchImpl);
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    const instance = createChatInstance({
        ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
        chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
    });
    const working = (taskId) => handlers.get('chat')({
        chat_id: 2, role: 'system', is_progress: true, task_id: taskId,
        content: 'reading the ledger', ts: '2026-09-16T00:00:00Z',
    });
    return { prior, instance, handlers, working,
        messages: () => globalThis.document.byId.get('chat-messages') };
}

test('a placement-stamped System row becomes one timeline item of its task card', () => {
    const { prior, instance, handlers, working, messages } = openCardChat();
    try {
        working('t1');
        const card = walkCard(messages(), 't1');
        assert.ok(card, 'the progress frame created the task card');
        const before = {
            phase: card.querySelector('[data-live-phase]')?.textContent,
            finished: card.dataset.finished,
            expanded: card.dataset.expanded,
            lines: timelineLines(card).length,
            bubbles: systemBubbles(messages()).length,
        };
        const renderedLines = captureRenderedLines();
        handlers.get('chat')(CUSTODY_ROW);
        assert.equal(systemBubbles(messages()).length, before.bubbles,
            'the stamped row raised no standalone System bubble');
        assert.equal(timelineLines(card).length, before.lines + 1,
            'the card gained exactly one timeline item');
        assert.equal(phasedLines(card, 'warn').length, 1, 'a custody fact reads as a warning line');
        const markup = renderedLines().join('');
        assert.match(markup, /chat-live-line-title[^>]*>Open delegated execution: run-x\.</,
            'the first line of the row is the item headline');
        assert.match(markup, /Pending delegated invocations: none observed\./,
            'the rest of the row is the item body');
        assert.equal(card.querySelector('[data-live-phase]')?.textContent, before.phase);
        assert.equal(card.dataset.finished, before.finished);
        assert.equal(card.dataset.expanded, before.expanded);
        // The host's row identity keys the item, so redelivery adds nothing.
        handlers.get('chat')(CUSTODY_ROW);
        assert.equal(timelineLines(card).length, before.lines + 1);
        assert.equal(systemBubbles(messages()).length, before.bubbles);
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('a placement-stamped row whose task has no card keeps its standalone System row', () => {
    const { prior, instance, handlers, messages } = openCardChat();
    try {
        handlers.get('chat')({ ...CUSTODY_ROW, task_id: 't9', card_row_id: 'final:t9:abc:custody_notice' });
        const bubbles = systemBubbles(messages());
        assert.equal(bubbles.length, 1, 'the fact the page cannot place is still shown');
        assert.match(bubbles[0].innerHTML, /Open delegated execution: run-x\./);
        assert.equal(walkCard(messages(), 't9'), null, 'and it minted no card of its own');
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('a reviews-placed row attaches to the card and asks the Reviews group to re-read once', async () => {
    const calls = [];
    const { prior, instance, handlers, working, messages } = openCardChat(async (url) => {
        calls.push(String(url));
        return { ok: true, json: async () => (String(url).startsWith('/api/chat/history')
            ? { messages: [] } : { id: 't1', active_direct_turns: [] }) };
    });
    const settlement = {
        chat_id: 2, role: 'system', system_type: 'acceptance_late_settlement',
        card_row: 'reviews', card_row_id: 'acceptance-late:k', task_id: 't1',
        content: 'Reviewers later passed this answer.\n- triad_one: PASS — Budget section is complete.',
        ts: '2026-09-16T00:06:00Z',
    };
    const detailReads = () => calls.filter((url) => url.startsWith('/api/tasks/t1')).length;
    const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
    try {
        working('t1');
        const card = walkCard(messages(), 't1');
        const before = { lines: timelineLines(card).length, reads: detailReads(),
            bubbles: systemBubbles(messages()).length };
        handlers.get('chat')(settlement);
        await settle();
        assert.equal(systemBubbles(messages()).length, before.bubbles,
            'the stamped row raised no standalone System bubble');
        assert.equal(timelineLines(card).length, before.lines + 1);
        assert.equal(phasedLines(card, 'result').length, 1, 'a settled review reads as a result line');
        assert.equal(detailReads(), before.reads + 1, 'the Reviews group was asked to re-read once');
        handlers.get('chat')(settlement);
        await settle();
        assert.equal(timelineLines(card).length, before.lines + 1);
        assert.equal(detailReads(), before.reads + 1, 'a repeated row asks for no second read');
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('history replay places stamped rows on their cards and keeps unplaceable ones visible', async () => {
    const rows = [
        { task_id: 'r1', is_progress: true, text: '💬 reading the ledger', ts: '2026-09-16T00:00:00Z' },
        { task_id: 'r1', role: 'system', system_type: 'custody_notice', card_row: 'timeline',
          card_row_id: 'final:r1:abc:custody_notice', ts: '2026-09-16T00:01:00Z',
          history_id: 'h-r1-custody', history_position: { source: 'chat', offset: 2 },
          text: 'Open delegated execution: run-x.\nPending delegated invocations: none observed.' },
        { task_id: 'r1', role: 'assistant', text: 'The report is ready.', ts: '2026-09-16T00:02:00Z' },
        { task_id: 'r9', role: 'system', system_type: 'custody_notice', card_row: 'timeline',
          card_row_id: 'final:r9:zzz:custody_notice', ts: '2026-09-16T00:03:00Z',
          text: 'Open delegated execution: run-y.' },
    ];
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: rows } : { active_direct_turns: [] } }));
    const ws = { on() { return () => {}; }, isConnected: () => true, send() {} };
    let instance;
    try {
        instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
                isCurrent: () => true, apply() {} }, chatId: 1, idPrefix: 'chat', mountEl: mount });
        await instance.refreshHistory({ revision: 1 });
        const messages = globalThis.document.byId.get('chat-messages');
        const card = walkCard(messages, 'r1');
        assert.ok(card, 'the replayed task kept its card');
        assert.equal(card.dataset.finished, '1', 'the untyped final still concludes the card');
        const bubbles = systemBubbles(messages);
        assert.equal(bubbles.length, 1, 'only the row with no card in this page stayed a System row');
        assert.match(bubbles[0].innerHTML, /Open delegated execution: run-y\./);
        if (card.dataset.expanded !== '1') {
            card.querySelector('[data-live-summary-button]').listeners.get('click')[0]({ detail: 0 });
        }
        assert.equal(phasedLines(card, 'warn').length, 1,
            'the replayed custody row is a timeline item of its card');
        // The replayed item keeps its history identity, so it sorts by its source
        // position and leaves the card when its page is released.
        assert.equal(phasedLines(card, 'warn')[0].dataset.liveLineKey, 'history-h-r1-custody');
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('an older page places its content-only row inside the recent owner card', async () => {
    const recent = { task_id: 'r1', is_progress: true, text: 'Recent owner progress',
        ts: '2026-09-16T01:00:00Z', history_id: 'progress:90' };
    const receipt = { task_id: 'r1', role: 'system', system_type: 'host_progress',
        is_progress: true, narration: false, card_row: 'reviews', card_row_id: 'merge-receipt:source',
        card_row_revision: 3, text: 'PR #1347 merge: merged', ts: '2026-09-16T00:01:00Z',
        history_id: 'progress:0', history_position: { source: 'progress', offset: 0 } };
    const calls = [];
    const { prior, mount } = installDom(async url => {
        if (!String(url).startsWith('/api/chat/history')) return { ok: true, json: async () => ({ active_direct_turns: [] }) };
        const cursor = new URL(url, 'http://local').searchParams.get('cursor');
        calls.push(cursor);
        return { ok: true, json: async () => ({ messages: cursor ? [receipt] : [recent],
            page_cursor: cursor || 'page:0', next_cursor: cursor ? null : 'page:1', has_more: !cursor }) };
    });
    let instance;
    try {
        instance = createChatInstance({
            ws: { on() { return () => {}; }, isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            stateSnapshots: { begin: () => ({ generation: 1 }), gate() { return Promise.resolve(this.begin()); },
                isCurrent: () => true, apply() {} }, chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
        });
        await instance.refreshHistory({ revision: 1 });
        const messages = globalThis.document.byId.get('chat-messages');
        const button = messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
        for (const handler of button.listeners.get('click')) await handler({ target: button });
        const card = walkCard(messages, 'r1');
        assert.deepEqual(calls, [null, 'page:1']);
        assert.ok(card);
        assert.equal(phasedLines(card, 'result').filter(line => line.dataset.liveLineKey === 'history-progress-0').length, 1,
            'the exact supplying page must place its receipt inside the recent owner');
        assert.equal(systemBubbles(messages).length, 0, 'the receipt never falls back to an orphan bubble');
    } finally { instance?.destroy(); restoreDom(prior); }
});

// A subagent that ended through host salvage: the host stamps the receipt as a
// row of THAT child's card. Production-shaped rows — the gateway emits `text`,
// the live frame `content`, and every child row carries its lineage.
const CHILD = { delegation_role: 'subagent', subagent_task_id: 'kid-1', parent_task_id: 'root-1',
    root_task_id: 'root-1', subagent_role: 'researcher' };
const RECEIPT = '⚠️ A model-provider outage stopped this task before Ouroboros produced a complete answer. '
    + 'The full intermediate output and technical details are preserved in the task details.';
const CHILD_INCIDENT = { role: 'system', system_type: 'terminal_incident', task_id: 'kid-1', ...CHILD,
    card_row: 'timeline', card_row_id: 'final:kid-1:0123456789abcdef:terminal_incident' };
const lifecycle = (event, ts, chatId) => ({ chat_id: chatId, role: 'assistant', is_progress: true, task_id: 'root-1',
    text: event, content: event, subagent_event: event, ts, ...CHILD });
const expand = (card) => {
    if (card.dataset.expanded !== '1') {
        card.querySelector('[data-live-summary-button]').listeners.get('click')[0]({ detail: 0 });
    }
};

async function replayChat(rows) {
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: rows } : { active_direct_turns: [] } }));
    const ws = { on() { return () => {}; }, isConnected: () => true, send() {} };
    const instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} }, chatId: 1, idPrefix: 'chat', mountEl: mount });
    return { prior, instance, messages: () => globalThis.document.byId.get('chat-messages') };
}

test('a live child terminal incident is a row of the child card, not a System bubble', () => {
    const { prior, instance, handlers, working, messages } = openCardChat();
    try {
        working('root-1');
        handlers.get('chat')(lifecycle('running', '2026-09-16T00:00:01Z', 2));
        const kid = walkCard(messages(), 'kid-1');
        assert.ok(kid, 'the lifecycle frame minted the child card');
        expand(kid);
        const before = { lines: timelineLines(kid).length, finished: kid.dataset.finished,
            phase: kid.querySelector('[data-live-phase]')?.textContent };
        handlers.get('chat')({ ...CHILD_INCIDENT, chat_id: 2, content: RECEIPT, ts: '2026-09-16T00:02:00Z' });
        assert.equal(systemBubbles(messages()).length, 0, 'no standalone System bubble in the feed');
        assert.equal(timelineLines(kid).length, before.lines + 1, 'the child card gained exactly one row');
        assert.equal(phasedLines(kid, 'warn').length, 1);
        // The receipt is a fact beside the child's work: it concludes and relabels nothing.
        assert.equal(kid.dataset.finished, before.finished);
        assert.equal(kid.querySelector('[data-live-phase]')?.textContent, before.phase);
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('a live child terminal incident with no card record stays a System row and mints no card', () => {
    const { prior, instance, handlers, messages } = openCardChat();
    try {
        handlers.get('chat')({ ...CHILD_INCIDENT, chat_id: 2, content: RECEIPT, ts: '2026-09-16T00:02:00Z' });
        const bubbles = systemBubbles(messages());
        assert.equal(bubbles.length, 1, 'the fact the page cannot place is still shown');
        assert.match(bubbles[0].innerHTML, /model-provider outage/);
        assert.equal(walkCard(messages(), 'kid-1'), null, 'the receipt is not rendered as the child\'s answer');
        assert.equal(walkCard(messages(), 'root-1'), null, 'and it minted no parent shell');
    } finally { instance?.destroy(); restoreDom(prior); }
});

for (const [label, withScheduledRow] of [['after its card record', true], ['before its card record', false]]) {
    test(`history replay places a child terminal incident that arrives ${label}`, async () => {
        // The child's own rows can be outside the page while the parent's later
        // `failed` lifecycle row still mints the child's record in the same pass.
        const rows = [
            { task_id: 'root-1', role: 'assistant', is_progress: true, text: 'Planning the swarm.', ts: '2026-09-16T00:00:00Z' },
            ...(withScheduledRow ? [lifecycle('scheduled', '2026-09-16T00:00:01Z', 1)] : []),
            { ...CHILD_INCIDENT, text: RECEIPT, ts: '2026-09-16T00:02:00Z',
              history_id: 'h-kid-incident', history_position: { source: 'chat', offset: 3 } },
            lifecycle('failed', '2026-09-16T00:02:01Z', 1),
        ];
        const { prior, instance, messages } = await replayChat(rows);
        try {
            await instance.refreshHistory({ revision: 1 });
            assert.equal(systemBubbles(messages()).length, 0, 'the receipt is not a bubble beside the card');
            const kid = walkCard(messages(), 'kid-1');
            assert.ok(kid, 'the replayed child kept its card');
            expand(kid);
            assert.equal(phasedLines(kid, 'warn').filter((line) =>
                line.dataset.liveLineKey === 'history-h-kid-incident').length, 1);
        } finally { instance?.destroy(); restoreDom(prior); }
    });
}

test('a second history sync redraws a child System row whose old bubble it released', async () => {
    // No card record in the page: the escape row. The re-sync releases the old
    // bubble of every child row so a formerly unbound final can move into its
    // card; a row that stays a bubble must come back, not vanish until reload.
    const rows = [{ ...CHILD_INCIDENT, text: RECEIPT, ts: '2026-09-16T00:02:00Z',
        history_id: 'h-kid-incident', history_position: { source: 'chat', offset: 3 } }];
    const { prior, instance, messages } = await replayChat(rows);
    try {
        await instance.refreshHistory({ revision: 1 });
        const [first] = systemBubbles(messages());
        assert.ok(first, 'the first sync drew the standalone System row');
        // Scrolled out of the viewport: a bubble being read is kept as it is.
        first.getBoundingClientRect = () => ({ top: 900, bottom: 920, left: 0, right: 100, width: 100, height: 20 });
        await instance.refreshHistory({ revision: 2 });
        const bubbles = systemBubbles(messages());
        assert.equal(bubbles.length, 1, 'the row is still in the feed after the second sync');
        assert.notEqual(bubbles[0], first, 'as a redrawn node: the old one was released');
        assert.match(bubbles[0].innerHTML, /model-provider outage/);
    } finally { instance?.destroy(); restoreDom(prior); }
});

// Optional stdin fixture comes from the real Python outbox/dedup/history test.
// The standalone node lane uses the same adversarial revision/timestamp shape.
test('canonical receipt revisions keep one merged row through live and history replay', async () => {
    const supplied = process.env.MERGE_RECEIPT_PROJECTION
        ? JSON.parse((await import('node:fs')).readFileSync(process.env.MERGE_RECEIPT_PROJECTION, 'utf8')) : null;
    const base = { chat_id: 7, task_id: 'merge-task', role: 'system', system_type: 'host_progress',
        is_progress: true, narration: false, card_row: 'reviews', card_row_id: 'merge-receipt:r' };
    const merged = { ...base, card_row_revision: 3, content: 'PR #7 merge: merged', ts: '2026-09-16T00:01:00Z' };
    const queued = { ...base, card_row_revision: 2, content: 'PR #7 merge: queued', ts: '2026-09-16T00:02:00Z' };
    const live = supplied?.live || [merged, queued, merged];
    const history = supplied?.history || [merged, queued].map((row, i) => ({ ...row, text: row.content,
        history_id: `h-receipt-${i}`, history_position: { source: 'progress', offset: i } }));
    const opened = openCardChat();
    try {
        opened.working('merge-task');
        const markup = captureRenderedLines();
        for (const row of live) opened.handlers.get('chat')({ ...row, chat_id: 2 });
        // Older unversioned rows (e.g. an old outbox) cannot overwrite a revision either.
        opened.handlers.get('chat')({ ...live[0], card_row_revision: undefined, chat_id: 2,
            content: 'PR #7 merge: queued', ts: '2099-01-01T00:00:00Z' });
        assert.match(markup().at(-1), /merge: merged/);
        assert.equal(phasedLines(walkCard(opened.messages(), 'merge-task'), 'result').length, 1);
    } finally { opened.instance.destroy(); restoreDom(opened.prior); }
    for (const rows of [history, [...history].reverse()]) {
        const replay = await replayChat([{ task_id: 'merge-task', is_progress: true,
            text: 'Reading the PR.', ts: '2026-09-16T00:00:00Z' }, ...rows.map(row => ({ ...row, chat_id: 1 }))]);
        try {
            const markup = captureRenderedLines();
            await replay.instance.refreshHistory({ revision: 1 });
            assert.match(markup().at(-1), /merge: merged/);
            assert.equal(phasedLines(walkCard(replay.messages(), 'merge-task'), 'result').length, 1);
        } finally { replay.instance.destroy(); restoreDom(replay.prior); }
    }
});

// This test needs the real endpoint's bounded selection, not hand-reordered frames.
test('cold bounded receipt history renders canonical truth in a new Chat', {
    skip: !process.env.COLD_RECEIPT_PROJECTION,
}, async () => {
    const { payload, selected } = JSON.parse((await import('node:fs'))
        .readFileSync(process.env.COLD_RECEIPT_PROJECTION, 'utf8'));
    assert.equal(selected.length, 60);
    assert.deepEqual(selected.filter(row => row.card_row_id).map(row => row.card_row_revision), [2]);
    const replay = await replayChat(payload.messages);
    try {
        const markup = captureRenderedLines();
        await replay.instance.refreshHistory({ revision: 1 });
        assert.match(markup().join('\n'), /merge: merged/);
        assert.doesNotMatch(markup().join('\n'), /merge: queued/);
        assert.equal(phasedLines(walkCard(replay.messages(), 'merge-task'), 'result').length, 1);
    } finally { replay.instance.destroy(); restoreDom(replay.prior); }
});
