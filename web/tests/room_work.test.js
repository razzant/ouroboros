import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { cardRowView, createRoomWork, latestRepresentedRoot, workCard } from '../modules/room_work.js';
import { markView } from '../modules/status_mark.js';
import { desiredLiveCardPhase } from '../modules/task_phase_chip.js';
import { emptyWorkState, reconcileWorkState, roomStatusView } from '../modules/work_status.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { SERVER_CASES, projection, room, work } from './work_fixture.js';
const work_ = work;  // a test-local name where `work` is a mounted list

// The instance's work state for one room, from the works a test states (the server's shapes).
function workState(works = [], activeRoom = null, chatId = 1) {
    return reconcileWorkState(emptyWorkState(), { supervisor_ready: true, ...projection(chatId, works, activeRoom) }, chatId);
}
const running = (id, extra = {}) => work({ activity_id: id, chat_id: 1, kind: 'managed_task' }, {
    controls: [{ action: 'wrap_up', target: id }, { action: 'hurry', target: id }, { action: 'pause', target: id },
        { action: 'stop', target: id, scope: 'cascade' }], ...extra });
const asking = (id, quizId) => work({ activity_id: id, chat_id: 1, kind: 'managed_task' }, { activity: 'waiting',
    release: 'owner', because: [{ code: 'owner_answer', text: 'Waiting for your answer' }],
    turn: [{ kind: 'question', task_id: id, quiz_id: quizId }], controls: [{ action: 'stop', target: id }] });
// A row's or the control's one mark, as painted: its shape and its hue.
const markOf = (node) => { const mark = node.querySelector('.status-mark'); return [mark.dataset.shape, mark.dataset.hue]; };

test('a Project keeps its latest represented root: an unfinished one first, never a child or a removed card', () => {
    const done = { root: { isConnected: true }, finished: true };
    const active = { root: { isConnected: true }, finished: false };
    const child = { root: { isConnected: true }, isSubagent: true };
    const removed = { root: { isConnected: false } };
    assert.equal(latestRepresentedRoot([done, active, child, removed]), active);
    active.finished = true;
    assert.equal(latestRepresentedRoot([done, active, child]), active);
    assert.equal(latestRepresentedRoot([child, removed]), null);
});

test('an older prepended card at the same timestamp does not become the latest root', () => {
    const card = (offset) => ({ root: { isConnected: true, dataset: { ts: '1000' } },
        historyPosition: { source: 'progress', offset }, finished: true });
    const newer = card(500), older = card(100);
    assert.equal(latestRepresentedRoot([newer, older]), newer);
});

test('a card\'s own words as a row keep the outcome\'s tone and dot; an unfinished card that leads nowhere is a ring', () => {
    const settled = (phase) => {
        const view = cardRowView(desiredLiveCardPhase({ finished: true }, phase), { settled: true });
        return [view.word, view.tone, markView(view)];
    };
    assert.deepEqual(settled('error'), ['Failed', 'failed', { shape: 'dot', hue: 'failed' }]);
    assert.deepEqual(settled('done'), ['Done', 'done', { shape: 'dot', hue: 'ok' }]);
    // Done with warnings succeeded (its word says the warnings); Stopped was someone's decision, neither good nor bad.
    assert.deepEqual(settled('warn'), ['Done with warnings', 'done', { shape: 'dot', hue: 'ok' }]);
    assert.deepEqual(settled('cancelled'), ['Stopped', 'stopped', { shape: 'dot', hue: 'neutral' }]);
    const ended = cardRowView({ phase: 'ended', text: 'No longer listed as active' }, { settled: true });
    assert.deepEqual([ended.tone, markView(ended)], ['ended', { shape: 'ring', hue: 'ink' }]);
    const unknown = cardRowView(desiredLiveCardPhase({}), {});
    assert.deepEqual([unknown.tone, markView(unknown)], ['unknown', { shape: 'ring', hue: 'ink' }]);
    assert.deepEqual(cardRowView({ phase: 'error', text: 'Failed', secondary: 'Finalizing…' }),
        { word: 'Failed', secondary: 'Finalizing…', tone: 'failed', mark: 'none', owner: false, outcome: 'error', focus: '' });
});

test('a work\'s row goes to its current episode\'s card, else another shown episode card; never a helper\'s', () => {
    const held = SERVER_CASES.held_continue.works[0];
    const card = (extra = {}) => ({ root: { isConnected: true, hidden: false }, ...extra });
    const current = card(), first = card(), helper = card({ isSubagent: true });
    assert.equal(workCard(held, new Map([['c', current], ['p', first]])), current);
    assert.equal(workCard(held, new Map([['p', first], ['h0', helper]])), first, 'the work\'s first card stands for it');
    assert.equal(workCard(held, new Map([['h0', helper]])), null, 'a helper card is never the work\'s place');
    assert.equal(workCard(held, new Map([['c', card({ root: { isConnected: true, hidden: true } })]])), null,
        'an earlier episode hidden under the chain\'s card is not shown');
});

test('the mounted list holds its order while open, settles ended work and only navigates to a question', async () => {
    const calls = [];
    const { prior, mount } = installDom(async (url, init) => {
        calls.push([String(url), init?.method || 'GET']);
        return { ok: true, json: async () => ({}) };
    });
    let shade = null;
    try {
        const doc = globalThis.document;
        const node = (className, parent) => { const el = doc.createElement('div'); el.className = className; parent.appendChild(el); return el; };
        const page = node('page', mount);
        const input = doc.createElement('textarea');
        input.id = 'chat-input'; // Main has the id; only namespaced Project inputs gain its mirror class.
        page.appendChild(input);
        const header = node('chat-page-header', page);
        // The product's bar: the title row (where the status goes) and the trailing commands (where it was).
        const titleRow = node('app-page-title-row', node('app-page-title-block', header));
        const group = node('app-page-actions ui-chrome-group', header);
        const badge = doc.createElement('span');
        badge.className = 'status-badge thinking';
        badge.textContent = 'Working';
        Object.assign(badge.dataset, { tone: 'live', mark: 'moving', owner: '', amber: '' });  // what setStatus writes
        group.appendChild(badge);
        const messages = node('chat-messages', page);
        let works = [running('t-run')];
        const reveals = [];
        shade = createRoomWork({
            page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { stick: true, cancel() {}, scroll() {} },
            works: () => workState(works, room('working')), liveCardRecords: new Map(), pendingSubmissions: new Map(),
            revealQuestion: async (...args) => { reveals.push(args); return true; },
        });
        // The status follows the room's title (1A) and its Shade hangs from the bar's bottom edge (one Shade).
        const trigger = titleRow.children.find((child) => child.classList.contains('rw-trigger'));
        assert.equal(group.children.some((child) => child.classList.contains('rw-trigger')), false);
        const anchor = header.children.find((child) => child.classList.contains('rw-shade-anchor'));
        const surface = anchor.children.find((child) => child.classList.contains('rw-surface'));
        assert.ok(surface.classList.contains('rw-shade'));
        const list = surface.children[0];
        const ids = () => list.children.map((row) => row.dataset.id);
        assert.ok(trigger.contains(badge), 'the real status badge is the control’s word');
        assert.equal(trigger.dataset.tone, 'live');
        assert.equal(trigger.disabled, false);

        trigger.click();
        assert.equal(shade.isOpen(), true);
        assert.deepEqual(ids(), ['t-run']);
        // A question arriving while open joins at the end: order holds under the reader.
        works = [running('t-run'), asking('t-ask', 'q-ask')];
        shade.update();
        assert.deepEqual(ids(), ['t-run', 't-ask']);
        // Each row's one mark: moving work in its word's ink, the owner's amber dot where the question waits.
        assert.deepEqual([markOf(list.children[0]), markOf(list.children[1])], [['moving', 'ink'], ['dot', 'owner']]);
        // Work that ends while open keeps its row, says so, and offers nothing.
        works = [asking('t-ask', 'q-ask')];
        shade.update();
        const ended = list.children[0];
        assert.equal(ended.dataset.settled, '1');
        assert.equal(ended.dataset.tone, 'ended');
        assert.deepEqual(markOf(ended), ['ring', 'ink'], 'its outcome is not known here: a ring');
        assert.equal(ended.querySelectorAll('.status-mark').length, 1, 'one mark node, repainted in place');
        assert.equal(ended.children.find((child) => child.classList.contains('rw-more-slot')).children.length, 0);
        // View question goes to the exact waited question, closes the list and posts nothing.
        const verb = list.children[1].children.find((child) => child.classList.contains('rw-verb'));
        verb.children[0].click();
        await Promise.resolve();
        assert.deepEqual(reveals, [['t-ask', 'q-ask', { focus: 'question' }]]);
        assert.equal(shade.isOpen(), false);
        assert.ok(calls.every(([url, method]) => method === 'GET' && !url.includes('/api/decisions')), JSON.stringify(calls));
        // Closed, the list sorts again: the owner's turn first.
        works = [running('t-run'), asking('t-ask', 'q-ask')];
        trigger.click();
        assert.deepEqual(ids(), ['t-ask', 't-run']);
        // Reconcile ended-only Main membership before returning focus: a now-disabled
        // status trigger cannot be the keyboard owner's next place.
        works = [];
        shade.update();
        assert.equal(shade.isOpen(), true);
        trigger.click();
        assert.equal(shade.isOpen(), false);
        assert.equal(trigger.disabled, true);
        assert.ok(doc.activeElement === input, 'closing an ended-only list returns focus to the composer');
    } finally {
        shade?.destroy();
        restoreDom(prior);
    }
});

// A Continue or a helper asks: View question goes to the task that asked, never the work's first root.
test('View question reaches the asking task of a work, not its first root', async () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let shade = null;
    try {
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const group = node('div', 'app-page-actions ui-chrome-group', header);
        const badge = node('span', 'status-badge online', group);
        Object.assign(badge.dataset, { tone: 'wait', mark: 'still', owner: '1', amber: '' });
        const messages = node('div', 'chat-messages', page);
        const reveals = [];
        const asked = (taskId) => ({ ...asking('w-first', 'q-next'), episode: { task_id: 'c-next', live: true },
            turn: [{ kind: 'question', task_id: taskId, quiz_id: 'q-next' }] });
        let works = [asked('c-next')];
        shade = createRoomWork({
            page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { stick: true, cancel() {}, scroll() {} },
            works: () => workState(works, room('waiting')), liveCardRecords: new Map(), pendingSubmissions: new Map(),
            revealQuestion: async (...args) => { reveals.push(args); return true; },
        });
        const trigger = titleRow.children.find((child) => child.classList.contains('rw-trigger'));
        const list = header.children.find((child) => child.classList.contains('rw-shade-anchor')).children
            .find((child) => child.classList.contains('rw-surface')).children[0];
        const ask = async () => {
            trigger.click();
            list.children[0].children.find((child) => child.classList.contains('rw-verb')).children[0].click();
            await Promise.resolve();
        };
        await ask();
        works = [asked('h-helper')];
        await ask();
        works = [asked('w-first')];
        await ask();
        assert.deepEqual(reveals.map((args) => args.slice(0, 2)),
            [['c-next', 'q-next'], ['h-helper', 'q-next'], ['w-first', 'q-next']]);
    } finally {
        shade?.destroy();
        restoreDom(prior);
    }
});

// A progress card can be painted before the slow projection lists the work that answers the same send.
test('a pending send keeps its slot until its work is listed; the work takes the same node, never a ghost', () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work;
    try {
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const group = node('div', 'app-page-actions ui-chrome-group', header);
        const badge = node('span', 'status-badge thinking', group);
        const messages = node('div', 'chat-messages', page);
        let works = [];
        const records = new Map();
        const pending = new Map([['send-1', { timestamp: 1000 }]]);
        work = createRoomWork({ page, statusBadge: badge, messagesDiv: messages,
            ws: { isConnected: () => true }, reading: { cancel() {}, scroll() {} },
            works: () => workState(works), liveCardRecords: records, pendingSubmissions: pending });
        const trigger = titleRow.children.find(el => el.classList.contains('rw-trigger'));
        const list = header.children.find(el => el.classList.contains('rw-shade-anchor')).children[0].children[0];
        const turns = () => list.children.filter(el => el.dataset.id === 'turn-1');
        trigger.click();
        const card = node('div', 'chat-live-block', messages);
        records.set('turn-1', { root: card, groupId: 'turn-1', phase: 'working' });
        // The send's own row takes the slot until the work that answers it is listed.
        work.update();
        const sendRow = list.children.find(el => el.dataset.id === 'pending:send-1');
        assert.ok(sendRow, 'the owner\'s own unconfirmed send has its row');
        works = [work_({ activity_id: 'turn-1', kind: 'direct_chat', chat_id: 1, client_message_id: 'send-1' })];
        pending.clear();
        work.update();
        const existing = turns()[0];
        assert.equal(turns().length, 1, 'the send and its work converge to one tracked view');
        assert.equal(existing, sendRow, 'the work takes the send\'s own node and slot');
        assert.equal(existing.dataset.settled, '0');
        assert.equal(existing.querySelector('.rw-main').dataset.target, 'card', 'the listed work goes to its card');
        trigger.click(); trigger.click();
        assert.equal(turns().length, 1, 'reopening leaves no orphan Working row');
        assert.equal(turns()[0].dataset.settled, '0');
    } finally { work?.destroy(); restoreDom(prior); }
});

test('a work the server lists but cannot confirm keeps its Stop; a work kept from an incomplete body offers nothing', () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let shade = null;
    try {
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const badge = node('span', 'status-badge', node('div', 'app-page-actions ui-chrome-group', header));
        const messages = node('div', 'chat-messages', page);
        // The server's own shape for an unconfirmed member: unknown, unconfirmed, and still its Stop.
        const unknown = work_({ activity_id: 't-unknown', chat_id: 1, kind: 'managed_task' }, { activity: 'unknown',
            confirmed: false, controls: [{ action: 'stop', target: 't-unknown', scope: 'cascade' }] });
        let state = workState([unknown], room('unknown'));
        shade = createRoomWork({ page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { cancel() {}, scroll() {} }, works: () => state, liveCardRecords: new Map(), pendingSubmissions: new Map() });
        titleRow.children.find((el) => el.classList.contains('rw-trigger')).click();
        const list = header.children.find((el) => el.classList.contains('rw-shade-anchor')).children[0].children[0];
        const row = () => list.children.find((el) => el.dataset.id === 't-unknown');
        const more = () => row().children.find((el) => el.classList.contains('rw-more-slot')).children.length;
        assert.equal(row().dataset.tone, 'unknown');
        assert.equal(more(), 1, 'the server offers Stop for work it lists, confirmed or not');
        // The next body is incomplete and omits it: the row stays unconfirmed, with nothing to press.
        state = reconcileWorkState(state, { supervisor_ready: true, ...projection(1, [], null, false) }, 1);
        shade.update();
        assert.equal(row().dataset.tone, 'unknown');
        assert.equal(more(), 0, 'a work no current body lists has no controls');
    } finally { shade?.destroy(); restoreDom(prior); }
});

test('a row names its card\'s coined name first, then its work\'s own name; a host not ready leaves every row unknown', () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let shade = null;
    try {
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const badge = node('span', 'status-badge', node('div', 'app-page-actions ui-chrome-group', header));
        const messages = node('div', 'chat-messages', page);
        const card = node('div', 'chat-live-block', messages);
        const record = { root: card, groupId: 't-run', suggestedName: 'Compare the cited pages' };
        let ready = true;
        shade = createRoomWork({ page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { cancel() {}, scroll() {} }, works: () => workState([running('t-run', { name: 'Source comparison' })],
                room('working')), liveCardRecords: new Map([['t-run', record]]), pendingSubmissions: new Map(),
            hostReady: () => ready });
        titleRow.children.find((el) => el.classList.contains('rw-trigger')).click();
        const list = header.children.find((el) => el.classList.contains('rw-shade-anchor')).children[0].children[0];
        const row = () => list.children.find((el) => el.dataset.id === 't-run');
        const name = () => row().querySelector('.rw-name').textContent;
        const more = () => row().children.find((el) => el.classList.contains('rw-more-slot')).children.length;
        // The card's coined name is the newest: the server's work name may still lag behind it.
        assert.equal(name(), 'Compare the cited pages');
        record.suggestedName = '';
        shade.update();
        assert.equal(name(), 'Source comparison', 'without a coined name, the work\'s own name');
        assert.deepEqual([row().dataset.tone, more()], ['live', 1]);
        // A host that is not ready proves nothing current: the row stays, unknown and with nothing to press.
        ready = false;
        shade.update();
        assert.deepEqual([row().dataset.tone, more()], ['unknown', 0]);
        ready = true;
        shade.update();
        assert.deepEqual([row().dataset.tone, more()], ['live', 1]);
    } finally { shade?.destroy(); restoreDom(prior); }
});

test('the one mark after the word is the room\'s shape, the owner\'s amber dot in its place, Online\'s own dot at rest', () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work;
    try {
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const group = node('div', 'app-page-actions ui-chrome-group', header);
        const badge = node('span', 'status-badge thinking', group);
        const messages = node('div', 'chat-messages', page);
        const works = [work_({ activity_id: 't-q', kind: 'managed_task', chat_id: 1 }, { activity: 'queued' }), asking('t-ask', 'q-1')];
        work = createRoomWork({ page, statusBadge: badge, messagesDiv: messages,
            ws: { isConnected: () => true }, reading: { cancel() {}, scroll() {} },
            works: () => workState(works, room('queued')), liveCardRecords: new Map(), pendingSubmissions: new Map() });
        const trigger = titleRow.children.find(el => el.classList.contains('rw-trigger'));
        // The word leads; its one mark follows it, then the caret, and nothing else is a mark.
        const order = trigger.children.map(el => el.className.split(' ').find(c => /^(status-badge|status-mark|rw-caret)$/.test(c)))
            .filter(Boolean);
        assert.deepEqual(order, ['status-badge', 'status-mark', 'rw-caret']);
        const mark = trigger.children.find(el => el.classList.contains('status-mark'));
        assert.equal(mark.getAttribute('aria-hidden'), 'true', 'the mark is drawn; the word and the note say it');
        const waitNote = trigger.children.find(el => el.classList.contains('rw-sr'));
        // The producer (work_status.roomStatusView) hands its view through setStatus: word, tone, mark and the
        // owner's move together, as chat.js writes them.
        const paint = (view) => {
            badge.textContent = view.text;
            Object.assign(badge.dataset, { tone: view.tone, mark: view.mark, owner: view.owner ? '1' : '', amber: view.amber,
                step: view.step || '' });
            work.update();
            return [trigger.dataset.tone, mark.hidden ? 'hidden' : mark.dataset.shape, mark.dataset.hue,
                waitNote.hidden ? '' : waitNote.textContent];
        };
        const inRoom = (activity, extra = {}) => ({ ...emptyWorkState(), room: room(activity, extra) });
        const question = { turn: [{ kind: 'question', task_id: 't-ask', quiz_id: 'q-1' }] };
        const action = { turn: [{ kind: 'action', task_id: 't-q' }] };
        // Before the producer's first answer (no tone on the badge) the control never claims Online.
        work.update();
        assert.deepEqual([trigger.dataset.tone, mark.hidden], ['offline', true]);
        // The work's shape in the word's ink: it moves, holds still, stands paused, is unconfirmed.
        assert.deepEqual(paint(roomStatusView(inRoom('working'))), ['live', 'moving', 'ink', '']);
        assert.deepEqual(paint(roomStatusView(inRoom('queued'))), ['queued', 'still', 'ink', '']);
        assert.deepEqual(paint(roomStatusView(inRoom('waiting'))), ['held', 'still', 'ink', '']);
        assert.deepEqual(paint(roomStatusView(inRoom('paused'))), ['held', 'pause', 'ink', ''], 'a Pause is not the owner\'s move');
        assert.deepEqual(paint(roomStatusView(inRoom('unknown'))), ['unknown', 'ring', 'ink', '']);
        // Something waits for the owner: the amber dot takes the mark's place beside any word, said in words too.
        assert.deepEqual(paint(roomStatusView(inRoom('working', question))),
            ['live', 'dot', 'owner', 'A question waits for your answer'], 'Working keeps its word and ink');
        assert.deepEqual(paint(roomStatusView(inRoom('queued', question))), ['queued', 'dot', 'owner', 'A question waits for your answer']);
        assert.deepEqual(paint(roomStatusView(inRoom('paused', action))), ['held', 'dot', 'owner', 'Something here waits for you']);
        // The word already says it: the amber dot beside the amber word, and no note repeats it.
        assert.deepEqual(paint(roomStatusView(inRoom('waiting', question))), ['wait', 'dot', 'owner', '']);
        // At rest the room says Online beside its green dot; the client's own facts lay their marks over the room.
        assert.deepEqual(paint(roomStatusView(emptyWorkState())), ['online', 'dot', 'ok', '']);
        assert.deepEqual(paint(roomStatusView(inRoom('working', question), { connected: false })), ['offline', 'ring', 'ink', '']);
        assert.deepEqual(paint(roomStatusView(emptyWorkState(), { ready: false })), ['starting', 'hidden', 'ink', '']);
        assert.deepEqual(paint(roomStatusView(emptyWorkState(), { sending: true })), ['live', 'moving', 'ink', '']);
        assert.deepEqual(paint(roomStatusView(inRoom('working'), { stopping: 'finalize' })), ['live', 'moving', 'ink', '']);
        // One mark node throughout, repainted in place.
        assert.equal(trigger.children.filter(el => el.classList.contains('status-mark')).length, 1);
        assert.equal(trigger.children.find(el => el.classList.contains('status-mark')), mark);
    } finally { work?.destroy(); restoreDom(prior); }
});

// DESIGN "Shell" on a phone (375 px), where the bar leaves the status about 90 px: a preparing room's start step goes
// first, then the word whole; the mark stays and the control's name keeps the whole status (touch has no hover).
test('on a phone-width bar a preparing room drops its start step before its word, and the name keeps both', () => {
    const css = readFileSync(new URL('../room_work.css', import.meta.url), 'utf8');
    for (const rule of ['.status-badge[data-step]:not([data-step=""])::after { content: " · " attr(data-step); }',
        '.rw-trigger[data-step="off"] .status-badge::after { content: none; }',
        '.rw-trigger[data-word="off"] .status-badge { display: none; }']) assert.ok(css.includes(rule), rule);
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work, resized = null;
    try {
        globalThis.ResizeObserver = class { constructor(fn) { resized = fn; } observe() {} disconnect() {} };
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const badge = node('span', 'status-badge thinking', node('div', 'app-page-actions ui-chrome-group', header));
        // Those rules at 7 px a character: the step shows unless the control drops it, the word unless the control
        // drops it whole, and the badge clips at the width the bar leaves it.
        let space = 600;
        const shown = () => {
            const off = badge.parentNode.dataset;
            if (off.word === 'off') return '';
            return badge.textContent + (badge.dataset.step && off.step !== 'off' ? ` · ${badge.dataset.step}` : '');
        };
        Object.defineProperty(badge, 'scrollWidth', { get: () => 7 * shown().length });
        Object.defineProperty(badge, 'clientWidth', { get: () => Math.min(7 * shown().length, space) });
        Object.defineProperty(titleRow, 'clientWidth', { get: () => space });
        work = createRoomWork({ page, statusBadge: badge, messagesDiv: node('div', 'chat-messages', page),
            ws: { isConnected: () => true }, reading: { cancel() {}, scroll() {} },
            works: () => workState(), liveCardRecords: new Map(), pendingSubmissions: new Map() });
        const trigger = titleRow.children.find(el => el.classList.contains('rw-trigger'));
        const paint = (state) => {
            const view = roomStatusView(state);
            badge.textContent = view.text;
            Object.assign(badge.dataset, { tone: view.tone, mark: view.mark, owner: view.owner ? '1' : '', amber: view.amber,
                step: view.step || '' });
            work.update();
            return [shown(), trigger.title];
        };
        const resize = (width) => { space = width; resized([{ target: header }]); return [shown(), trigger.title]; };
        const whole = 'Preparing · Building context';
        const preparing = reconcileWorkState(emptyWorkState(),
            { supervisor_ready: true, ...structuredClone(SERVER_CASES.direct_preparing) }, 1);
        assert.deepEqual(paint(preparing), [whole, whole], 'a wide bar says both');
        assert.deepEqual(resize(90), ['Preparing', whole], 'room for the word, not for its step');
        assert.deepEqual(resize(40), ['', whole], 'narrower still: the word goes whole, the mark stays');
        assert.deepEqual(resize(90), ['Preparing', whole], 'a wider bar brings the word back first');
        assert.deepEqual(resize(600), [whole, whole]);
        // A room without a step is fitted by its word alone.
        assert.deepEqual(paint(workState([], room('working'))), ['Working', 'Working']);
        assert.deepEqual(resize(40), ['', 'Working']);
    } finally { work?.destroy(); restoreDom(prior); }
});

// DESIGN "Shell": a Project's name wins its bar. A name its two lines would still cut makes the consciousness chip show
// its brain alone (the chip's name keeps the whole state); a name that fits leaves the chip whole.
test('a Project bar gives its name the chip\'s words only while the name would be cut, and gives them back', () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work, resized = null;
    try {
        globalThis.ResizeObserver = class { constructor(fn) { resized = fn; } observe() {} disconnect() {} };
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const panel = node('div', 'project-panel', mount);
        const bar = node('div', 'project-panel-bar', panel);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', bar));
        const name = node('h2', 'app-page-title', titleRow);
        const chip = node('button', 'cx-chip', bar);
        const page = node('div', 'chat-instance-panel', node('div', 'project-panel-body', panel));
        page.closest = (selector) => (selector === '.project-panel' ? panel : null);
        const badge = node('span', 'status-badge online', node('div', 'chat-panel-statusbar', page));
        // The name's two lines hold 30 characters beside the whole chip, 40 beside its brain alone.
        let length = 20, width = 200;
        Object.defineProperty(name, 'clientHeight', { get: () => 40 });
        Object.defineProperty(name, 'scrollHeight', { get: () => (length > (chip.dataset.fit === 'mark' ? 40 : 30) ? 60 : 40) });
        Object.defineProperty(badge, 'scrollWidth', { get: () => 0 });
        Object.defineProperty(badge, 'clientWidth', { get: () => 0 });
        Object.defineProperty(titleRow, 'clientWidth', { get: () => width });
        work = createRoomWork({ page, asPanel: true, statusBadge: badge, messagesDiv: node('div', 'chat-messages', page),
            ws: { isConnected: () => true }, reading: { cancel() {}, scroll() {} },
            works: () => workState(), liveCardRecords: new Map(), pendingSubmissions: new Map() });
        const named = (chars) => { length = chars; width += 1; resized([{ target: bar }]); return chip.dataset.fit || 'whole'; };
        assert.equal(named(20), 'whole', 'a name that fits leaves the chip whole');
        assert.equal(named(36), 'mark', 'a name two lines would cut: the chip shows its brain alone');
        assert.equal(named(60), 'mark', 'still cut there: the name ends its second line with an ellipsis, the chip stays small');
        assert.equal(named(12), 'whole', 'a short name gives the chip its words back');
    } finally { work?.destroy(); restoreDom(prior); }
});

// DESIGN "Room work status": a row goes to its message wherever it lies in the room's history.
test('an unloaded row reads the page around its task and lands on its card; one with no message says so', async () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work;
    try {
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const group = node('div', 'app-page-actions ui-chrome-group', header);
        const badge = node('span', 'status-badge thinking', group);
        const messages = node('div', 'chat-messages', page);
        const records = new Map(), asked = [];
        const turn = (id, extra = {}, row = {}) => work_({ activity_id: id, kind: 'direct_chat', chat_id: 1, ...row },
            { controls: [{ action: 'stop', target: id }], ...extra });
        const paused = { activity: 'paused', because: [{ code: 'paused_by_owner', text: 'You paused it' }] };
        const works = new Map([['t-old', turn('t-old', paused)], ['t-wake', turn('t-wake')],
            ['t-own', turn('t-own', {}, { client_message_id: 'msg-1' })]]);
        const historyPager = {
            pageOf: () => 'replay:recent', revisit: async () => ({ status: 'applied' }),
            around: async (address, wanted) => {
                asked.push([address, wanted()]);
                if (address !== 'task:t-old') return { status: 'anchor', anchor: { state: 'absent', task_id: address.slice(5) } };
                const card = node('div', 'chat-live-card', messages);  // the page around it mounts its card
                records.set('t-old', { root: card, groupId: 't-old' });
                return { status: 'applied', anchor: { state: 'found', task_id: 't-old', history_id: 'chat:100' } };
            },
        };
        work = createRoomWork({ page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { stick: false, cancel() {}, scroll() {}, claim: () => () => true }, historyPager,
            captureVisibleTimelineAnchor: () => ({ historyId: 'chat:900' }),
            works: () => workState([...works.values()]), liveCardRecords: records, pendingSubmissions: new Map() });
        const trigger = titleRow.children.find((el) => el.classList.contains('rw-trigger'));
        const anchor = header.children.find((el) => el.classList.contains('rw-shade-anchor'));
        const list = anchor.children.find((el) => el.classList.contains('rw-surface')).children[0];
        const row = (id) => list.children.find((el) => el.dataset.id === id);
        const main = (id) => row(id).querySelector('.rw-main');
        const text = (el) => [el.textContent, ...el.children.map(text)].join(' ');
        const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
        trigger.click();
        // Before it asks, a row claims nothing about where its message is.
        assert.match(main('t-old').getAttribute('aria-label'), /^Open: /);
        assert.doesNotMatch(main('t-old').getAttribute('aria-label'), /loaded/);
        assert.equal(main('t-old').dataset.target, 'history');
        assert.doesNotMatch(text(list), /Not in loaded messages|cannot take you to it/);

        main('t-old').click();
        await settle();
        assert.deepEqual(asked, [['task:t-old', true]]);
        assert.equal(work.isOpen(), false, 'landed: the reader came to read it');
        assert.equal(page.children.find((el) => el.classList.contains('rw-back')).hidden, false, 'Back returns to the passage left');

        trigger.click();
        assert.match(main('t-old').getAttribute('aria-label'), /^Go to task: /, 'its card is loaded now');
        main('t-wake').click();
        await settle();
        assert.equal(work.isOpen(), true, 'nothing to land on: the list stays and says so');
        assert.match(text(row('t-wake')), /No message yet\./);
        assert.match(main('t-wake').getAttribute('aria-label'), /^Details: /);
        main('t-wake').click();  // its details close; it does not ask again
        await settle();
        assert.equal(asked.length, 2);
        // Work whose facts change may have a message now: its row claims nothing again and a press asks.
        works.set('t-wake', turn('t-wake', paused));
        work.update();
        await new Promise((resolve) => setTimeout(resolve, 20));
        assert.match(main('t-wake').getAttribute('aria-label'), /^Open: /);
        main('t-wake').click();
        await settle();
        assert.equal(asked.length, 3);
        assert.match(text(row('t-wake')), /No message yet\./, 'the press answered again');
        // Its details are open now: another change keeps them open, without the stale answer.
        works.set('t-wake', turn('t-wake'));
        work.update();
        await new Promise((resolve) => setTimeout(resolve, 20));
        assert.equal(main('t-wake').getAttribute('aria-expanded'), 'true', 'opened details survive');
        assert.match(main('t-wake').getAttribute('aria-label'), /^Open: /, 'without the stale answer: a press asks again');
        main('t-own').click();
        await settle();
        assert.match(text(row('t-own')), /Its message could not be checked\./, 'an owner’s message is never “no message”');
        assert.doesNotMatch(text(list), /Not in loaded messages|cannot take you to it/);
        trigger.click();
        trigger.click();
        assert.match(main('t-wake').getAttribute('aria-label'), /^Open: /, 'a later open asks again');
        // A list dismissed while a row looks drops that look: nothing moves when it answers later.
        let answer, wantedThen = null;
        historyPager.around = (address, wanted) => new Promise((resolve) => {
            answer = () => { wantedThen = wanted(); resolve({ status: 'superseded' }); };
        });
        main('t-wake').click();
        await settle();
        trigger.click();  // dismissed
        answer();
        await settle();
        assert.equal(wantedThen, false, 'the dismissed list no longer wants that page');
        // Pressed again after the list reopened, then dismissed again: the first look's late answer never undoes
        // the second look's cancellation.
        const answers = [], wanted = [];
        historyPager.around = (address, isWanted) => new Promise((resolve) => {
            answers.push(() => { wanted.push(isWanted()); resolve({ status: 'superseded' }); });
        });
        trigger.click();
        main('t-wake').click();
        await settle();
        trigger.click();  // dismissed
        trigger.click();
        main('t-wake').click();  // the same row again
        await settle();
        answers[0]();
        await settle();
        trigger.click();  // dismissed again
        answers[1]();
        await settle();
        assert.deepEqual(wanted, [false, false], 'neither look applies once its list closed');
    } finally { work?.destroy(); restoreDom(prior); }
});

test('Back reads the page it left again, stays after a failed read and takes a follower back to the present', async () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work;
    try {
        const node = (tag, cls, parent) => {
            const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el;
        };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const badge = node('span', 'status-badge thinking', node('div', 'app-page-actions ui-chrome-group', header));
        const messages = node('div', 'chat-messages', page);
        const records = new Map(), revisits = [], presents = [];
        let mounted = false, fails = 1, gone = false;
        const historyPager = {
            pageOf: () => 'replay:left',
            revisit: async (cursor, wanted) => {
                revisits.push([cursor, wanted()]);
                if (gone) return { status: 'error', error: { status: 409 } };
                if (fails-- > 0) return { status: 'error' };
                mounted = true;  // the page the reader left is drawn again
                return { status: 'applied' };
            },
            around: async () => {
                records.set('t-old', { root: node('div', 'chat-live-card', messages), groupId: 't-old' });
                return { status: 'applied', anchor: { state: 'found', task_id: 't-old', history_id: 'chat:100' } };
            },
        };
        const reading = { stick: false, nearBottom: () => false, cancel() {}, scroll() {}, claim: () => () => true };
        work = createRoomWork({ page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true }, reading,
            historyPager, showLatest: async () => presents.push('newest'),
            captureVisibleTimelineAnchor: () => ({ historyId: 'chat:900' }),
            restoreVisibleTimelineAnchor: (_anchor, { exact } = {}) => Boolean(exact && mounted),
            works: () => workState([work_({ activity_id: 't-old', kind: 'direct_chat', chat_id: 1 })]),
            liveCardRecords: records, pendingSubmissions: new Map() });
        const trigger = titleRow.children.find((el) => el.classList.contains('rw-trigger'));
        const list = header.children.find((el) => el.classList.contains('rw-shade-anchor')).children[0].children[0];
        const main = () => list.children.find((el) => el.dataset.id === 't-old').querySelector('.rw-main');
        const back = page.children.find((el) => el.classList.contains('rw-back'));
        const live = page.children.find((el) => el.getAttribute('role') === 'status');
        const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
        trigger.click();
        main().click();
        await settle();
        back.click();
        await settle();
        assert.deepEqual(revisits, [['replay:left', true]], 'the passage was let go: its page is read again, once');
        assert.equal(back.hidden, false, 'a failed read keeps Back for another try');
        assert.match(live.textContent, /could not be read/);
        back.click();
        await settle();
        assert.equal(revisits.length, 2);
        assert.equal(back.hidden, true);
        assert.equal(live.textContent, 'Returned to where you were reading.');
        assert.deepEqual(presents, []);
        // A page that cannot be read again (the room's view changed) ends Back instead of asking for a retry.
        mounted = false;
        gone = true;
        trigger.click();
        main().click();
        await settle();
        back.click();
        await settle();
        assert.equal(revisits.length, 3);
        assert.equal(live.textContent, 'The earlier reading place is no longer loaded.');
        gone = false;
        // A reader following the present goes back to it and reads no left page.
        Object.assign(reading, { stick: true, nearBottom: () => true });
        trigger.click();
        main().click();
        await settle();
        back.click();
        await settle();
        assert.deepEqual(presents, ['newest']);
        assert.equal(revisits.length, 3);
    } finally { work?.destroy(); restoreDom(prior); }
});
