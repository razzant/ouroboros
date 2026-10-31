// Owner-facing host notices (#1369): a late-review row placed in a task card
// offers the exact applied review record it names through the existing artifact
// route and nothing for a pointer of another shape; a running card's status chip
// says Working while the title waits for a coined name or the turn's own
// narration, so no placeholder repeats the chip and no empty note poses as
// narration. Authored titles keep their words even when the word is Working.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { taskSourceDownloadUrl } from '../modules/api_client.js';
import { evidenceLinkHtml } from '../modules/chat_activity.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const HASH = 'a'.repeat(64);
const SOURCE_REF = { kind: 'task_source', root: 'artifact_store', sha256: HASH, size: 75000,
    path: `source_handles/context_checkpoints/acceptance-${HASH}.json` };
const NOTE = 'On the delivered version of this answer, reviewers later passed it.\n'
    + '- openai/gpt-5.5: passed it — The budget section is complete.';
const lateRow = (taskId, evidence, extra = {}) => ({
    chat_id: 2, role: 'system', system_type: 'acceptance_late_settlement', task_id: taskId,
    card_row: 'reviews', card_row_id: `acceptance-late:${taskId}`, content: NOTE,
    ts: '2026-09-28T00:06:00Z', ...(evidence ? { late_evidence: evidence } : {}), ...extra,
});
const evidenceFor = (taskId, ref = SOURCE_REF) => ({ task_id: taskId, panel_id: 'panel_1',
    settled_at: '2026-09-28T00:06:00Z', reviewed_revision: 'delivered', reviewed_is_emitted: true, source_ref: ref });

const walkNodes = (node) => [node, ...(node?.children || []).flatMap(walkNodes)];
const timelineLines = (card) => walkNodes(card).filter((node) => node.classList.contains('chat-live-line'));

// The flat fixture parses innerHTML without text; read the markup the renderer built.
function captureRenderedLines() {
    const doc = globalThis.document;
    const create = doc.createElement.bind(doc);
    const built = [];
    doc.createElement = (tag) => { const node = create(tag); built.push(node); return node; };
    return () => built.map((node) => node.innerHTML).filter((html) => html.includes('class="chat-live-line '));
}

function openChat(fetchImpl) {
    const { prior, mount } = installDom(fetchImpl);
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} };
    const instance = createChatInstance({
        ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
        chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
    });
    const emit = (row) => handlers.get('chat')({ chat_id: 2, ts: '2026-09-28T00:00:00Z', ...row });
    const hostNote = (taskId) => emit({ role: 'system', is_progress: true, task_id: taskId, content: 'Checkpoint saved.', narration: false });
    return { prior, instance, handlers, emit, hostNote, messages: () => globalThis.document.byId.get('chat-messages') };
}

test('a late-review row offers its exact applied review record through the artifact route', () => {
    const { prior, instance, emit, hostNote, messages } = openChat();
    try {
        hostNote('t1');
        const card = walkCard(messages(), 't1');
        assert.ok(card, 'the host note minted the task card');
        const rendered = captureRenderedLines();
        emit(lateRow('t1', evidenceFor('t1')));
        const markup = rendered().join('');
        assert.match(markup, /chat-live-line-title[^>]*>On the delivered version of this answer, reviewers later passed it\./);
        assert.match(markup, /openai\/gpt-5\.5: passed it/);
        const href = taskSourceDownloadUrl('t1', SOURCE_REF);
        assert.equal(href, `/api/tasks/t1/artifacts/acceptance-${HASH}.json?source=${encodeURIComponent(SOURCE_REF.path)}`);
        // The chat link ink (`md-link`), never the browser's default link colour.
        assert.match(markup, new RegExp(`<a class="md-link" href="${href.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/&/g, '&amp;')}" download data-live-line-evidence>Download the review record</a>`),
            'the row links the exact applied review record, as a download');
        assert.doesNotMatch(markup, /OMISSION NOTE|DEGRADED|triad_/);
        assert.equal(timelineLines(card).length, 2, 'the note row and the late row');
        // Redelivery keys on the host's row identity: no second row, no second link.
        const again = captureRenderedLines();
        emit(lateRow('t1', evidenceFor('t1')));
        assert.equal(timelineLines(card).length, 2, 'the redelivered row added no timeline item');
        assert.ok((again().join('').match(/data-live-line-evidence/g)?.length ?? 0) <= 1, 'at most the one link per render');
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('a pointer of another shape offers no link: forged, legacy and absent evidence stay text-only', () => {
    const cases = {
        absent: null,
        legacy: { task_id: 't2', panel_id: 'p', settled_at: '', reviewed_revision: 'unknown' },
        wrongRoot: evidenceFor('t2', { ...SOURCE_REF, root: 'runtime_data' }),
        traversal: evidenceFor('t2', { ...SOURCE_REF, path: '../private.json' }),
        badDigest: evidenceFor('t2', { ...SOURCE_REF, sha256: 'not-a-digest' }),
        noSize: evidenceFor('t2', { ...SOURCE_REF, size: '75000' }),
        stringRef: { ...evidenceFor('t2'), source_ref: 'source_handles/x.json' },
    };
    for (const [name, evidence] of Object.entries(cases)) {
        const { prior, instance, emit, hostNote } = openChat();
        try {
            hostNote('t2');
            const rendered = captureRenderedLines();
            emit(lateRow('t2', evidence));
            const markup = rendered().join('');
            assert.match(markup, /reviewers later passed it\./, `${name}: the words still reach the card`);
            assert.doesNotMatch(markup, /data-live-line-evidence|<a /, `${name}: no guessed link`);
        } finally { instance?.destroy(); restoreDom(prior); }
    }
});

test('history replay keeps the late row on its card with its record link', async () => {
    const rows = [
        { task_id: 'r1', role: 'system', is_progress: true, narration: false, text: 'Checkpoint saved.', ts: '2026-09-28T00:00:00Z' },
        { task_id: 'r1', role: 'assistant', text: 'The report is ready.', ts: '2026-09-28T00:01:00Z' },
        { ...lateRow('r1', evidenceFor('r1')), chat_id: 1, text: NOTE, history_id: 'h-r1-late',
          history_position: { source: 'chat', offset: 3 } },
    ];
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: rows } : { active_direct_turns: [] } }));
    const ws = { on() { return () => {}; }, isConnected: () => true, send() {} };
    let instance;
    try {
        instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
                isCurrent: () => true, apply() {} }, chatId: 1, idPrefix: 'chat', mountEl: mount });
        const rendered = captureRenderedLines();
        await instance.refreshHistory({ revision: 1 });
        const messages = globalThis.document.byId.get('chat-messages');
        const card = walkCard(messages, 'r1');
        assert.ok(card, 'the replayed task kept its card');
        if (card.dataset.expanded !== '1') {
            card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        }
        const markup = rendered().join('');
        assert.match(markup, /reviewers later passed it\./);
        assert.match(markup, /data-live-line-evidence>Download the review record<\/a>/, 'the stored pointer replays as the same link');
        assert.equal(timelineLines(card).filter((line) => line.dataset.liveLineKey === 'history-h-r1-late').length, 1,
            'the replayed item keeps its history identity');
    } finally { instance?.destroy(); restoreDom(prior); }
});

// A shortened note points at the review record in words; with no card in this chat
// the System row it becomes must still offer that record.
const SHORTENED = `${NOTE} Every claim is backed… (shortened; the complete text is in the review record)`;
const bubblesFor = (root, taskId) => walkNodes(root).filter((node) =>
    node.classList.contains('chat-bubble') && node.dataset.taskId === taskId);
const escapedHref = (taskId) => taskSourceDownloadUrl(taskId, SOURCE_REF).replace(/&/g, '&amp;');

test('a late-review row with no card in this chat stays a System row that keeps its record link', () => {
    for (const [name, evidence, linked] of [['stored', evidenceFor('n1'), true],
        ['forged', evidenceFor('n1', { ...SOURCE_REF, path: '../private.json' }), false]]) {
        const { prior, instance, emit, messages } = openChat();
        try {
            emit(lateRow('n1', evidence, { content: SHORTENED }));
            assert.equal(walkCard(messages(), 'n1'), null, `${name}: no card was minted for the row`);
            const [bubble, ...extra] = bubblesFor(messages(), 'n1');
            assert.ok(bubble && !extra.length, `${name}: the row is one System bubble`);
            assert.match(bubble.innerHTML, /shortened; the complete text is in the review record/, name);
            const link = `<a class="md-link" href="${escapedHref('n1')}" download data-live-line-evidence>Download the review record</a>`;
            assert.equal(bubble.innerHTML.includes(link), linked, `${name}: ${linked ? 'the exact record link' : 'no guessed link'}`);
            if (!linked) assert.doesNotMatch(bubble.innerHTML, /data-live-line-evidence|<a /, name);
        } finally { instance?.destroy(); restoreDom(prior); }
    }
});

test('history replay of a card-less late row keeps its record link on the System row', async () => {
    const rows = [{ ...lateRow('h1', evidenceFor('h1')), chat_id: 1, text: SHORTENED, history_id: 'h-h1-late',
        history_position: { source: 'chat', offset: 1 } }];
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
        assert.equal(walkCard(messages, 'h1'), null);
        const [bubble] = bubblesFor(messages, 'h1');
        assert.ok(bubble, 'the replayed row is a System bubble');
        assert.equal(bubble.dataset.historyId, 'h-h1-late');
        assert.ok(bubble.innerHTML.includes(`<a class="md-link" href="${escapedHref('h1')}" download data-live-line-evidence>`),
            'the stored pointer replays as the same link');
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('a record link is only ever the task artifact route it was minted on', () => {
    const { prior } = installDom();
    try {
        const href = taskSourceDownloadUrl('t1', SOURCE_REF);
        assert.match(evidenceLinkHtml({ href, label: 'Download the review record' }), /data-live-line-evidence/);
        for (const forged of ['javascript:alert(1)', 'https://example.com/x', '//example.com/x', '', null]) {
            assert.equal(evidenceLinkHtml({ href: forged }), '', String(forged));
        }
        assert.equal(evidenceLinkHtml(null), '');
    } finally { restoreDom(prior); }
});

test('the status chip says Working; the title waits for a name or narration and keeps authored words', () => {
    const { prior, instance, emit, hostNote, handlers, messages } = openChat();
    try {
        hostNote('w1');
        const card = walkCard(messages(), 'w1');
        assert.equal(card.querySelector('[data-live-phase]').textContent, 'Working', 'the chip states the running state');
        assert.equal(card.querySelector('[data-live-title]').textContent, '', 'no placeholder repeats the chip');
        // An empty narration frame names nothing and stays no narration.
        emit({ role: 'assistant', is_progress: true, task_id: 'w1', content: '', narration: true });
        assert.equal(card.querySelector('[data-live-title]').textContent, '', 'an empty note does not become a title');
        // The model's own words are the title, even when the word is Working.
        emit({ role: 'assistant', is_progress: true, task_id: 'w1', content: 'Working on the ledger', narration: true });
        assert.equal(card.querySelector('[data-live-title]').textContent, 'Working on the ledger');
        handlers.get('task_named')({ task_id: 'w1', suggested_name: 'Working' });
        assert.equal(card.querySelector('[data-live-title]').textContent, 'Working', 'a coined name is kept verbatim');
        assert.equal(card.querySelector('[data-live-phase]').textContent, 'Working');
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('log_events mints no placeholder headline', () => {
    const empty = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't', content: '', narration: true });
    assert.deepEqual([empty.headline, empty.visible], ['', false]);
    const usage = summarizeChatLiveEvent({ type: 'llm_usage', task_id: 't', model: 'm', round: 1 });
    assert.equal(usage.headline, '', 'a frame that names nothing has no headline');
    const spoken = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't', content: 'Working on it', narration: true });
    assert.deepEqual([spoken.headline, spoken.promote, spoken.human], ['Working on it', true, true], 'authored words are kept');
});
