// Project unread (DESIGN "Unread"): a Project revision is read only
// when a history read that STARTED after the revision was observed has been
// painted in a visible room whose reader is at the newest messages. These run
// the real createChatInstance consumer against a scripted history server.
import assert from 'node:assert/strict';
import test from 'node:test';
import { historyNodeIsProtected } from '../modules/chat_history_replay.js';
import { createProjectReadReceipt, isAtNewestMessage } from '../modules/project_read_state.js';
import { ElementStub } from './chat_dom_fixture.js';
import { TS, row, late, room, placeAtTop, until, quiet } from './project_read_state_fixture.js';

test('a history read already in flight cannot acknowledge a newer revision', async (t) => {
    const r = room(t);
    r.server.rows = [row(1)];
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).painted, true);

    // Revision 2 is observed and its read starts; revision 3 lands while it is in flight.
    r.server.rows = [row(1), row(2)];
    const release = r.gate();
    const second = r.instance.refreshHistory({ revision: 2 });
    r.server.rows = [row(1), row(2), row(3)];
    const third = r.instance.refreshHistory({ revision: 3 });
    release();
    const [superseded, receipt] = await Promise.all([second, third]);

    assert.equal(superseded.painted, false, 'a superseded paint acknowledges nothing');
    assert.ok(!receipt.painted || r.shown().includes('chat:3'),
        'no paint receipt for revision 3 while its row is absent (false ACK)');
    assert.deepEqual(r.reads.at(-1), ['chat:1', 'chat:2', 'chat:3'],
        'revision 3 is covered by a read that started after it was observed');
    assert.equal(receipt.painted, true);
    assert.ok(r.shown().includes('chat:3'), 'the acknowledged revision is on the page');
});

test('a room read further up is painted but not read until the reader reaches the newest messages', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    r.server.rows = [row(1), row(2), row(3)];
    const first = await r.instance.refreshHistory({ revision: 1 });
    assert.deepEqual([first.painted, first.read], [true, true], 'a first open lands on the newest messages');

    r.readUp();
    r.server.rows = [row(1), row(2), row(3), row(4)];
    const upper = await r.instance.refreshHistory({ revision: 2 });
    assert.deepEqual([upper.painted, upper.read], [true, false], 'new content above the fold is not read');
    assert.ok(r.shown().includes('chat:4'), 'the content itself is on the page');
    assert.equal(r.messages.scrollTop, 0, 'the reader keeps their place');
    assert.equal(arrivals, 0);

    const readsBefore = r.reads.length;
    r.readLatest();
    r.readLatest();
    assert.equal(arrivals, 1, 'one arrival at the newest messages, not one per scroll event');
    const retried = await r.instance.refreshHistory({ revision: 2 });
    assert.deepEqual([retried.painted, retried.read], [true, true]);
    assert.equal(r.reads.length, readsBefore, 'a covered revision is not read again');

    r.readUp();
    r.readLatest();
    assert.equal(arrivals, 2, 'leaving and returning is a new arrival');
});

test('a hidden document paints but does not read; showing it again reports the reader once', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    r.server.rows = [row(1)];
    await r.instance.refreshHistory({ revision: 1 });
    r.setHidden(true);
    r.server.rows = [row(1), row(2)];
    const hidden = await r.instance.refreshHistory({ revision: 2 });
    assert.equal(hidden.read, false, 'nobody can see a hidden window');
    r.scroll(r.messages.scrollHeight);
    assert.equal(arrivals, 0, 'scrolling a hidden window is not arriving');
    r.setHidden(false);
    assert.equal(arrivals, 1, 'the window shown at the newest messages retries the acknowledgement');
    assert.equal((await r.instance.refreshHistory({ revision: 2 })).read, true);
});

test('a recent read that could not open the chat source reads nothing; an older gap does not block', async (t) => {
    const r = room(t);
    r.server.rows = [row(1)];
    // The server names no arrival it cannot read (test_an_unreadable_chat_source_leaves_the_arrival_unknown).
    r.server.window = { complete: false, truncated_by: ['chat_source_unavailable'], latest_message: null };
    const gap = await r.instance.refreshHistory({ revision: 1 });
    assert.deepEqual([gap.painted, Boolean(gap.read)], [false, false]);
    r.server.window = { complete: false, truncated_by: ['archive_floor', 'progress_unreadable_source'] };
    const reads = r.reads.length;
    const older = await r.instance.refreshHistory({ revision: 1 });
    assert.equal(r.reads.length, reads + 1, 'the uncovered revision is read again, not assumed');
    assert.deepEqual([older.painted, older.read], [true, true],
        'an incomplete older archive is no reason to withhold the newest messages');
});

test('a room closed while its read is in flight reads nothing and reports no arrival', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    r.server.rows = [row(1)];
    const release = r.gate();
    const pending = r.instance.refreshHistory({ revision: 1 });
    r.instance.destroy();
    release();
    assert.deepEqual(await pending, { painted: false, revision: 1 });
    r.readUp();
    r.readLatest();
    r.setHidden(false);
    assert.equal(arrivals, 0, 'a destroyed room never asks for an acknowledgement');
});

// The newest message is the one that ARRIVED last (window.latest_message). A late
// answer keeps the time it was written at, so it sorts above messages that arrived
// before it — or above the whole recent window, where only an older page shows it.
test('a late answer below the recent window is read only once an older page shows it on screen', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [row(1), row(2), row(3)];
    r.server.older = [late(9)];
    r.server.window = { complete: false, truncated_by: ['quota'],
        latest_message: { history_id: 'chat:9', out_of_order: true } };
    const landed = await r.instance.refreshHistory({ revision: 4 });
    assert.deepEqual([landed.painted, landed.read], [true, false],
        'the bottom is painted, but the message that arrived last is not on the page');
    assert.ok(!r.shown().includes('chat:9'));
    assert.equal(arrivals, 0);

    r.readUp();
    await until(() => r.shown().includes('chat:9'));
    assert.equal(r.shown()[0], 'chat:9', 'the older page places it by the time it was written');
    r.scroll(0);
    assert.equal(arrivals, 1, 'the late answer on screen is an arrival at the newest message');
    const read = await r.instance.refreshHistory({ revision: 4 });
    assert.deepEqual([read.painted, read.read], [true, true]);

    r.readLatest();
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, false,
        'the bottom of the conversation is not where the late answer is');
});

test('a late answer sorted above earlier arrivals is read on screen, not at the bottom', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [late(9), row(1), row(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    const landed = await r.instance.refreshHistory({ revision: 3 });
    assert.deepEqual(r.shown(), ['chat:9', 'chat:1', 'chat:2'], 'it keeps its chronological place');
    assert.deepEqual([landed.painted, landed.read], [true, false]);
    r.readUp();
    assert.equal(arrivals, 1);
    assert.equal((await r.instance.refreshHistory({ revision: 3 })).read, true);
});

// Opening an unread room lands on the message that arrived last (owner decision
// 2026-10-11): once, for that opening, and never over the reader's own move.
const wheel = (r, deltaY) => {
    for (const handler of r.messages.listeners.get('wheel') || []) handler({ type: 'wheel', deltaY, target: r.messages, timeStamp: 0 });
};

test('an unread room opened lands on the message that arrived last above later rows, and reads it', async (t) => {
    const r = room(t);
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [late(9), row(1), row(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    void r.instance.showLatest({ arrive: true });
    r.instance.cancelHistoryPaint();  // as app.js's acknowledgement does before its read
    const landed = await r.instance.refreshHistory({ revision: 3 });
    assert.deepEqual([landed.painted, landed.read], [true, true], 'the opening lands on it and reads it');
    assert.equal(r.messages.scrollTop, 0, 'it is at the top of the read band, what came after it below');

    r.server.rows = [late(9), row(1), row(2), row(3)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:3', out_of_order: false } };
    r.readLatest();
    const top = r.messages.scrollTop;
    const later = await r.instance.refreshHistory({ revision: 4 });
    assert.equal(r.messages.scrollTop, top, 'a later read moves no one: the landing belongs to the opening');
    assert.equal(later.read, true);
});

test('a reader who moves before the opening read keeps their place', async (t) => {
    const r = room(t);
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [late(9), row(1), row(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    void r.instance.showLatest({ arrive: true });
    r.scroll(40);
    wheel(r, 1);
    const moved = await r.instance.refreshHistory({ revision: 3 });
    assert.equal(r.messages.scrollTop, 40, 'the reader\'s own gesture superseded the landing');
    assert.equal(moved.read, false);
});

test('an opening whose newest message is unknown lands nowhere', async (t) => {
    const r = room(t);
    r.server.rows = [row(1)];
    r.server.window = { complete: false, truncated_by: ['chat_malformed_jsonl'], latest_message: null };
    void r.instance.showLatest({ arrive: true });
    const unknown = await r.instance.refreshHistory({ revision: 1 });
    assert.deepEqual([unknown.painted, Boolean(unknown.read)], [false, false], 'an honest gap, as before');
});

test('an unknown opening cannot later jump to a newly arrived answer', async (t) => {
    const r = room(t);
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [row(1), row(2)];
    r.server.window = { complete: false, truncated_by: ['chat_malformed_jsonl'], latest_message: null };
    await r.instance.showLatest({ arrive: true });
    await r.instance.refreshHistory({ revision: 2 });
    assert.ok(r.messages.scrollTop > 0, 'the unknown opening stayed at the bottom');
    r.server.rows = [late(9), row(1), row(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    const later = await r.instance.refreshHistory({ revision: 3 });
    assert.equal(r.messages.scrollTop, r.messages.scrollHeight,
        'ordinary bottom-follow may account for added height, but cannot jump to the late answer');
    assert.equal(later.read, false, 'the newly arrived answer remains unread above the viewport');
});

test('a room reopened without unread keeps its bottom', async (t) => {
    const r = room(t);
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [late(9), row(1), row(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    void r.instance.showLatest();
    const plain = await r.instance.refreshHistory({ revision: 3 });
    assert.equal(plain.read, false, 'reopened at its bottom, as before');
    assert.notEqual(r.messages.scrollTop, 0);
});

test('an unread opening whose newest message is on screen at the bottom moves nothing', async (t) => {
    const r = room(t);
    r.server.rows = [row(1), row(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:2', out_of_order: false } };
    void r.instance.showLatest({ arrive: true });
    const bottom = await r.instance.refreshHistory({ revision: 2 });
    assert.deepEqual([bottom.read, r.messages.scrollTop], [true, r.messages.scrollHeight]);
});

test('a read that cannot tell which message arrived last reads nothing', async (t) => {
    const r = room(t);
    r.server.rows = [row(1)];
    r.server.window = { complete: false, truncated_by: ['chat_malformed_jsonl'], latest_message: null };
    const unknown = await r.instance.refreshHistory({ revision: 1 });
    assert.deepEqual([unknown.painted, Boolean(unknown.read)], [false, false]);
    r.server.window = { complete: false, truncated_by: ['chat_malformed_jsonl'],
        latest_message: { history_id: 'chat:1', out_of_order: false } };
    const named = await r.instance.refreshHistory({ revision: 1 });
    assert.deepEqual([named.painted, named.read], [true, true],
        'a named newest message in its ordinary place is read at the bottom, as before');
});

// A question is a standalone message: when it arrived last, its card is the
// message the reader must see, whether the card was built from history or
// live first and then named by a later read.
const question = (n) => ({
    role: 'assistant', msg_type: 'quiz', task_id: 'root-1', text: `Merge now? ${n}`, ts: TS,
    history_id: `chat:${n}`, history_position: { source: 'chat', offset: n },
    quiz: { quiz_id: `q${n}`, options: [{ label: 'Yes' }, { label: 'No' }], state: 'open' },
});
// The feed node that shows question `id` (the card rides in a message frame).
const cardOf = (r, id) => r.messages.children.find((node) => (node.children || [])
    .some((child) => child.classList?.contains('chat-quiz-card') && child.dataset.quizId === id));

test('a question that arrived last is read where its card is on screen', async (t) => {
    const r = room(t);
    r.server.rows = [row(1), question(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:2', out_of_order: false } };
    const opened = await r.instance.refreshHistory({ revision: 2 });
    assert.equal(cardOf(r, 'q2')?.dataset.historyId, 'chat:2', 'the card carries the question row it shows');
    assert.deepEqual([opened.painted, opened.read], [true, true], 'the question on screen is read');
});

test('a question first shown live is read once a history read names it', async (t) => {
    const r = room(t);
    r.server.rows = [row(1)];
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).read, true);
    const { history_id: _id, history_position: _position, ...frame } = question(2);
    r.emit('quiz', { ...frame, type: 'quiz', chat_id: 2 });
    assert.ok(cardOf(r, 'q2'), 'the live question is on the page');
    r.server.rows = [row(1), question(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:2', out_of_order: false } };
    const live = cardOf(r, 'q2');
    const named = await r.instance.refreshHistory({ revision: 2 });
    assert.equal(cardOf(r, 'q2'), live, 'the named row is the live card, not a second one');
    assert.equal(live.dataset.historyId, 'chat:2', 'the live card now carries the row it shows');
    assert.deepEqual([named.painted, named.read], [true, true]);
});

// The bottom is not enough even for a message in its ordinary place: later card
// rows and the owner's own messages can push it above the fold.
test('a newest message pushed above the fold by later rows is not read at the bottom', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    placeAtTop(t, r, 'chat:1');
    const owner = (n) => ({ ...row(n), role: 'user', text: `owner note ${n}` });
    r.server.rows = [row(1), owner(2), owner(3)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:1', out_of_order: false } };
    const landed = await r.instance.refreshHistory({ revision: 1 });
    assert.ok(r.messages.scrollTop >= 20, 'the reader is at the bottom, past the answer');
    assert.deepEqual([landed.painted, landed.read], [true, false], 'painted, but the answer is above the fold');
    assert.equal(arrivals, 0);

    r.scroll(0);
    assert.equal(arrivals, 1, 'the answer on screen is an arrival at the newest message');
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).read, true);
});

test('a named newest message is read where it is on screen, and only there', () => {
    const el = (top, bottom) => ({ isConnected: true, getClientRects: () => [{}],
        getBoundingClientRect: () => ({ top, bottom }), closest: () => null });
    const at = (top, atBottom) => isAtNewestMessage({ history_id: 'chat:9', out_of_order: false }, {
        delivered: () => true, nodes: () => [el(top, top + 20)], viewport: el(0, 300), atBottom: () => atBottom });
    assert.equal(at(-40, true), false, 'above the fold at the bottom of the conversation');
    assert.equal(at(100, true), true);
    assert.equal(at(100, false), true, 'on screen with taller rows below it: nowhere else can it be read');
});

// A page loaded while the reader stays put (a short room, `Load older`) can bring
// the newest arrival on screen without any scroll to report it.
test('an older page that shows the late answer on screen is an arrival without a scroll', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    r.server.rows = [row(1), row(2), row(3)];
    r.server.older = [late(9)];
    r.server.window = { complete: false, truncated_by: ['quota'],
        latest_message: { history_id: 'chat:9', out_of_order: true } };
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, false);
    r.readUp();
    assert.equal(arrivals, 0, 'the scroll that asks for the page finds nothing to read yet');
    await until(() => r.shown().includes('chat:9'));
    assert.equal(arrivals, 1, 'the applied page reports the reader at the newest message');
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, true);
});

// The page is applied before the frame that draws it; the controls and the
// viewport settle in that frame, without a scroll to report the reader.
test('an older page takes the edge again once drawn', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    let top = -40;
    const rect = ElementStub.prototype.getBoundingClientRect;
    ElementStub.prototype.getBoundingClientRect = function () {
        return this.dataset?.historyId === 'chat:9' ? r.box(top, top + 20) : rect.call(this);
    };
    t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; });
    r.server.rows = [row(1), row(2), row(3)];
    r.server.older = [late(9)];
    r.server.window = { complete: false, truncated_by: ['quota'],
        latest_message: { history_id: 'chat:9', out_of_order: true } };
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, false);
    const frame = globalThis.requestAnimationFrame;
    const frames = [];
    globalThis.requestAnimationFrame = (fn) => { frames.push(fn); return frames.length; };
    try {
        r.readUp();
        for (const fn of frames.splice(0)) fn(); // the gesture's own frame asks for the page
        await until(() => r.shown().includes('chat:9'));
        assert.equal(arrivals, 0, 'as applied, the late answer is still above the fold');
        top = 0;
        for (const fn of frames.splice(0)) fn();
        assert.equal(arrivals, 1, 'drawn and settled on screen, it is the arrival');
    } finally {
        globalThis.requestAnimationFrame = frame;
    }
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, true);
});

// Once a read names the newest message, only that message on screen reads the
// room: the bottom of the conversation never stands in for it, in its ordinary
// place or not, wherever the page happens to draw it.
test('a named newest message is never read at the bottom in its place', () => {
    const el = (top, bottom, { drawn = true, parent = null, card = false } = {}) => ({
        isConnected: true, parentElement: parent,
        getClientRects: () => (drawn ? [{}] : []), getBoundingClientRect: () => ({ top, bottom }),
        closest(selector) { return selector === '.chat-live-card' && card ? this : parent?.closest(selector) ?? null; },
    });
    const viewport = el(0, 300);
    for (const out_of_order of [false, true]) {
        const at = (nodes, atBottom = true, facts = {}) => isAtNewestMessage({ history_id: 'chat:9', out_of_order }, {
            delivered: () => true, nodes: () => nodes, viewport, atBottom: () => atBottom, ...facts });
        assert.equal(at([]), false, 'delivered but drawn nowhere is not on screen, even at the bottom');
        const card = el(-400, 280, { card: true });
        assert.equal(at([el(-200, -180, { parent: card })]), false,
            'inside a task card scrolled above the fold, at the bottom, is not read');
        assert.equal(at([el(100, 120, { parent: card })], false), true, 'inside a card, on screen');
        assert.equal(at([el(0, 0, { drawn: false, parent: el(100, 140, { card: true }) })], false), true,
            'a collapsed card on screen shows what it holds');
        assert.equal(at([el(0, 0, { drawn: false, parent: el(-90, -40, { card: true }) })]), false,
            'a collapsed card above the fold does not');
        assert.equal(at([el(0, 0, { drawn: false, parent: viewport })]), false, 'a node with no boxes outside a card');
        assert.equal(at([el(100, 120)], true, { delivered: () => false }), false, 'not on the page yet');
    }
    const facts = { delivered: () => true, nodes: () => [], viewport, atBottom: () => true };
    assert.equal(isAtNewestMessage(null, facts), false, 'an unknown arrival is never read');
    assert.equal(isAtNewestMessage(undefined, facts), true, 'a read that names nothing keeps the bottom rule');
    assert.equal(isAtNewestMessage(undefined, { ...facts, atBottom: () => false }), false);
});

// The ordinary case of the same rule through the real room: a task card that
// grew after the answer (its progress and a host row placed in it) plus the
// owner's own reply fill the bottom; none of them is a conversation message.
test('an in-place newest reply pushed above the fold by a later task card is not read at the bottom', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    placeAtTop(t, r, 'chat:1');
    const later = { task_id: 'root-2', ts: '2026-09-28T12:05:00.000Z' };
    r.server.rows = [row(1),
        { ...later, role: 'assistant', text: 'Working on it', is_progress: true, history_id: 'progress:2',
            history_position: { source: 'progress', offset: 2 } },
        { ...later, role: 'system', text: 'Custody settled', system_type: 'custody_notice', card_row: 'timeline',
            card_row_id: 'final:root-2:custody', history_id: 'chat:3', history_position: { source: 'chat', offset: 3 } },
        { ...row(4), role: 'user', text: 'owner reply', ts: '2026-09-28T12:06:00.000Z' }];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:1', out_of_order: false } };
    const landed = await r.instance.refreshHistory({ revision: 1 });
    assert.ok(r.shown().includes('chat:1') && r.messages.scrollTop >= 20, 'the reader is at the bottom, past the reply');
    assert.deepEqual([landed.painted, landed.read], [true, false], 'the bottom is not the reply');
    r.readLatest();
    assert.equal(arrivals, 0, 'scrolling at the bottom is still not reaching it');

    r.scroll(0);
    assert.equal(arrivals, 1, 'the reply on screen is the arrival');
    assert.equal((await r.instance.refreshHistory({ revision: 1 })).read, true);
});

// The composer is drawn over the bottom of the feed (and Main's header over its
// top): a late answer beneath it intersects the feed's viewport but cannot be
// read there. Eviction still treats it as on screen.
test('a late answer beneath the composer is not on screen until it clears it; eviction still keeps it', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    r.chrome({ viewport: [0, 300], below: [220, 300] });
    let at = [250, 270];
    const rect = ElementStub.prototype.getBoundingClientRect;
    ElementStub.prototype.getBoundingClientRect = function () {
        return this.dataset?.historyId === 'chat:9' ? r.box(...at) : rect.call(this);
    };
    t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; });
    r.server.rows = [late(9), row(1), row(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    const covered = await r.instance.refreshHistory({ revision: 3 });
    assert.deepEqual([covered.painted, covered.read], [true, false], 'under the composer is not read');
    const node = r.messages.children.find((child) => child.dataset?.historyId === 'chat:9');
    assert.equal(historyNodeIsProtected(node, r.messages), true, 'eviction geometry is the whole viewport, unchanged');
    r.scroll(1);
    assert.equal(arrivals, 0);

    at = [200, 220];
    r.scroll(2);
    assert.equal(arrivals, 1, 'clear of the composer is an arrival');
    assert.equal((await r.instance.refreshHistory({ revision: 3 })).read, true);
});

test('the read band is the viewport less the header above and the composer below', () => {
    const el = (top, bottom, rendered = true) => ({
        isConnected: true, getClientRects: () => (rendered ? [{}] : []),
        getBoundingClientRect: () => ({ top, bottom }), closest: () => null,
    });
    const at = (top, facts) => isAtNewestMessage({ history_id: 'chat:9', out_of_order: true }, {
        delivered: () => true, nodes: () => [el(top, top + 20)], viewport: el(0, 300), atBottom: () => true, ...facts });
    const chrome = { header: el(0, 40), composer: el(220, 300) };
    assert.equal(at(10, chrome), false, 'under the header');
    assert.equal(at(250, chrome), false, 'under the composer');
    assert.equal(at(100, chrome), true);
    assert.equal(at(250, { header: el(0, 40), composer: el(220, 300, false) }), true, 'a hidden composer covers nothing');
    assert.equal(at(250, {}), true, 'a feed without chrome is its viewport');
});

// Chrome can cover the whole feed (a tall draft in a short window): nothing is
// visible, even where the node overlaps the inverted band's two edges.
test('a feed the header and composer cover entirely shows no message', () => {
    const el = (top, bottom) => ({ isConnected: true, getClientRects: () => [{}],
        getBoundingClientRect: () => ({ top, bottom }), closest: () => null });
    const at = (node, viewport, chrome) => isAtNewestMessage({ history_id: 'chat:9', out_of_order: false }, {
        delivered: () => true, nodes: () => [el(...node)], viewport: el(...viewport), atBottom: () => true, ...chrome });
    assert.equal(at([40, 120], [80, 200], { composer: el(60, 300) }), false, 'the composer rises above the feed');
    assert.equal(at([90, 110], [80, 200], { composer: el(100, 300) }), true, 'the strip it leaves is still read');
    assert.equal(at([100, 140], [80, 200], { header: el(0, 120), composer: el(120, 300) }), false, 'no strip is left');
    assert.equal(at([90, 160], [80, 200], { header: el(0, 150), composer: el(100, 300) }), false, 'they overlap');
});

// A read that names another newest message moves where the reader must be, so
// the arrival edge is taken again then, not only when the reader scrolls.
test('a refresh naming a new newest message takes the edge again, so reaching it retries', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [row(1), row(2), row(3)];
    await r.instance.refreshHistory({ revision: 1 });
    r.readUp();
    r.readLatest();
    assert.equal(arrivals, 1, 'the reader is at the newest messages');

    r.server.rows = [late(9), row(1), row(2), row(3)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    const withheld = await r.instance.refreshHistory({ revision: 2 });
    assert.deepEqual([withheld.painted, withheld.read], [true, false], 'the late answer is off screen at the bottom');
    r.scroll(0);
    assert.equal(arrivals, 2, 'reaching it is an arrival, though the reader never left the bottom before it was named');
    assert.equal((await r.instance.refreshHistory({ revision: 2 })).read, true);
});

test('a reconnect read naming a newest message already on screen retries without a scroll', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [late(9), row(1), row(2)];
    r.server.window = { complete: false, truncated_by: ['chat_malformed_jsonl'], latest_message: null };
    assert.equal((await r.instance.refreshHistory({ revision: 3 })).painted, false, 'an unknown arrival reads nothing');
    r.readUp();
    assert.equal(arrivals, 0);
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    const reads = r.reads.length;
    r.reconnect();
    await until(() => r.reads.length > reads);
    await until(() => arrivals === 1);
});

test('a covered revision whose newest message a later read cannot name is read again once the source heals', async (t) => {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [late(9), row(1), row(2)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    const landed = await r.instance.refreshHistory({ revision: 3 });
    assert.deepEqual([landed.painted, landed.read], [true, false], 'known, and off screen');
    // A reconnect read meets a line still being written (the owner's own message: no new revision).
    r.server.window = { complete: false, truncated_by: ['chat_incomplete_live_line'], latest_message: null };
    const reads = r.reads.length;
    r.reconnect();
    await until(() => r.reads.length > reads);
    r.readUp();
    assert.equal(arrivals, 0, 'unknown: being at the late answer is not reading it');
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:9', out_of_order: true } };
    const healed = await r.instance.refreshHistory({ revision: 3 });
    assert.equal(r.reads.length, reads + 2, 'the same revision is read again, not answered from the unknown read');
    assert.deepEqual([healed.painted, healed.read], [true, true], 'the healed read names it, on screen');
});

test('the read receipt takes its edge again only when a read names another newest message', () => {
    let reading = true, arrivals = 0;
    const receipt = createProjectReadReceipt({ read: async () => true, isShown: () => true,
        isReadingLatest: () => reading, onReadingLatest: () => { arrivals += 1; } });
    const settle = (latest) => { receipt.recent({ window: { latest_message: latest } }); receipt.settle(); };
    settle(undefined);
    assert.equal(arrivals, 0, 'the first read only records what it named');
    settle(undefined);
    assert.equal(arrivals, 0, 'the same newest message is no change');
    settle({ history_id: 'chat:9', out_of_order: true });
    assert.equal(arrivals, 1, 'another newest message, with the reader at it');
    reading = false;
    settle(null);
    assert.equal(arrivals, 1);
    reading = true;
    receipt.note();
    assert.equal(arrivals, 2, 'the edge was taken again, so the next arrival reports');
});

// The recent read's bounded search can run out before the newest message (card rows
// and the owner's own messages after it): unknown, but searched down to
// `latest_before`. The older pages of its chain carry the search on; the one that
// names the newest message makes it readable, on screen, while every later recent
// read still finds nothing newer down to that chain's boundary.
const span = (from, to, chain = 'c1') => ({ from, to, chain, gaps: [] });
const coverage = (upper, from, chain = 'c1') => ({ v: 1, view: 'v', upper: { chat: upper, progress: 0 },
    spans: { chat: span(from, upper, chain), progress: span(0, 0, 'empty') } });
function searchedRoom(t) {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; } });
    placeAtTop(t, r, 'chat:9');
    r.server.rows = [row(20), row(21)].map((item) => ({ ...item, role: 'user', text: `owner note ${item.history_position.offset}` }));
    r.server.older = [row(9), row(12)].map((item, index) => (index ? { ...item, role: 'user' } : late(9)));
    r.server.window = { complete: false, truncated_by: ['quota'], latest_message: null, latest_before: 15 };
    r.server.coverage = coverage(100, 20);
    r.server.olderWindow = { latest_message: { history_id: 'chat:9', out_of_order: true } };
    r.server.olderCoverage = coverage(100, 5);
    return { r, arrivals: () => arrivals };
}

test('an older page that carries the search on to the newest message makes it readable on screen', async (t) => {
    const { r, arrivals } = searchedRoom(t);
    const landed = await r.instance.refreshHistory({ revision: 4 });
    assert.deepEqual([landed.painted, Boolean(landed.read)], [false, false], 'unknown: nothing is read');
    r.readUp();
    await until(() => r.shown().includes('chat:9'));
    await until(() => arrivals() === 1);
    const reads = r.reads.length;
    const read = await r.instance.refreshHistory({ revision: 4 });
    assert.equal(r.reads.length, reads + 1, 'the uncovered revision is read again; that read still finds nothing newer');
    assert.deepEqual([read.painted, read.read], [true, true], 'the named message on screen is read');
    r.readLatest();
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, false, 'the bottom is not where it is');
});

for (const [name, change] of [
    ['a newer arrival the next read searched past', (r) => { r.server.window = { ...r.server.window, latest_before: 140 };
        r.server.coverage = coverage(160, 150); }],
    ['a newer arrival the next read names', (r, t) => { r.server.window = { complete: true, truncated_by: [],
        latest_message: { history_id: 'chat:30', out_of_order: false } };
        // Named below the fold, where only it can be read.
        const rect = ElementStub.prototype.getBoundingClientRect;
        ElementStub.prototype.getBoundingClientRect = function () {
            return this.dataset?.historyId === 'chat:30' ? r.box(100, 120) : rect.call(this);
        };
        t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; }); }],
    ['a source gap', (r) => { r.server.window = { complete: false, truncated_by: ['chat_malformed_jsonl'], latest_message: null }; }],
    ['another chain', (r) => { r.server.coverage = coverage(100, 20, 'c2'); }],
    // Another task became the room's member: the same bytes, read through another lens.
    ['another view on the same chain', (r) => { r.server.coverage = { ...coverage(100, 20), view: 'v2' }; }],
]) test(`the message an older page named is not read after ${name}`, async (t) => {
    const { r, arrivals } = searchedRoom(t);
    await r.instance.refreshHistory({ revision: 4 });
    r.readUp();
    await until(() => r.shown().includes('chat:9'));
    await until(() => arrivals() === 1);
    change(r, t);
    r.server.rows = [...r.server.rows, { ...row(30), text: 'newer answer' }];
    const after = await r.instance.refreshHistory({ revision: 5 });
    assert.ok(r.shown().includes('chat:9'), 'the named message is still on screen');
    assert.equal(Boolean(after.read), false, 'it is no longer known to be the newest message');
});

test('an older page that reaches the start of the chat with no message there lets the bottom decide', async (t) => {
    const { r, arrivals } = searchedRoom(t);
    // Only the owner's own notes and a child's words: an old revision counted the child's words.
    r.server.older = [row(9), row(12)].map((item) => ({ ...item, role: 'user', text: `owner note ${item.history_position.offset}` }));
    r.server.olderWindow = { latest_absent: true };
    r.server.olderCoverage = coverage(100, 0);
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).painted, false, 'unknown until the start is reached');
    r.readUp();
    await until(() => r.shown().includes('chat:9'));
    r.readLatest();
    await until(() => arrivals() === 1);
    const read = await r.instance.refreshHistory({ revision: 4 });
    assert.deepEqual([read.painted, read.read], [true, true], 'no standalone message: the bottom is read');
    r.server.coverage = { ...coverage(100, 20), view: 'v2' };
    assert.equal((await r.instance.refreshHistory({ revision: 5 })).painted, false, 'bound to the view it was proven in');
});

// Opening an unread room shows the message that arrived last however far below the
// opening read's page it lies (owner decision 2026-10-11): that opening alone reads
// the room's older pages until a drawn one holds it, then lands once.
const chain = (count, last = [late(9)]) => Array.from({ length: count }, (_, index) => ({
    rows: index === count - 1 ? last : [row(200 - index)], window: {} }));
const named = { complete: false, truncated_by: ['quota'], latest_message: { history_id: 'chat:9', out_of_order: true } };
function deepRoom(t, pages, { window = named, projectId, walked, top = 'chat:9' } = {}) {
    let arrivals = 0;
    const r = room(t, { onReadingLatest: () => { arrivals += 1; }, projectId });
    const rect = ElementStub.prototype.getBoundingClientRect;
    placeAtTop(t, r, top);
    const placed = ElementStub.prototype.getBoundingClientRect;  // `walked` rows lie above the viewport, unprotected
    ElementStub.prototype.getBoundingClientRect = function () {
        return walked?.test(this.dataset?.historyId || '') ? r.box(-500, -480) : placed.call(this);
    };
    t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; });  // after placeAtTop's own
    Object.assign(r.server, { rows: [row(300), row(301)], window, pages });
    const hold = (k) => { let open; r.server.pageGates[k] = new Promise((resolve) => { open = resolve; }); return open; };
    const open = () => {
        void r.instance.showLatest({ arrive: true });
        r.instance.cancelHistoryPaint();  // as app.js's acknowledgement does before its read
        return r.instance.refreshHistory({ revision: 4 });
    };
    return { r, open, hold, arrivals: () => arrivals };
}

test('an unread opening reads older pages to its newest message, past the pages kept loaded, and lands once', async (t) => {
    const { r, open, arrivals } = deepRoom(t, chain(5), { walked: /^chat:19\d|^chat:200$/ });
    assert.equal(Boolean((await open()).read), false, 'the opening read itself cannot show it');
    await until(() => r.shown().includes('chat:9') && r.messages.scrollTop === 0);
    assert.deepEqual(r.pageReads, [1, 2, 3, 4, 5], 'one older page at a time, with no click');
    assert.ok(!r.shown().includes('chat:200'), 'the pager keeps its pages loaded: those walked past went');
    assert.equal(arrivals(), 1, 'the landing reports the reader at the newest message');
    const read = await r.instance.refreshHistory({ revision: 4 });
    assert.deepEqual([read.painted, read.read], [true, true], 'and the next read acknowledges it');
    r.readLatest();
    r.server.window = { ...named, latest_message: { history_id: 'chat:7', out_of_order: true } };
    await r.instance.refreshHistory({ revision: 5 });
    assert.deepEqual(r.pageReads, [1, 2, 3, 4, 5], 'a later arrival never pages: the reach was the opening\'s');
});

test('a gesture during the reach, even inside a run of empty pages, ends it: no further page, no landing', async (t) => {
    const empty = { rows: [], window: {} };
    const { r, open, hold, arrivals } = deepRoom(t, [empty, empty, empty, { rows: [late(9)], window: {} }]);
    const release = hold(2);
    await open();
    await until(() => r.pageReads.length === 2);
    wheel(r, 1);
    const kept = r.messages.scrollTop;
    release();
    await quiet();
    assert.deepEqual(r.pageReads, [1, 2], 'the page already asked for is drawn; nothing more is asked');
    assert.equal(r.messages.scrollTop, kept, 'nothing lands over the reader\'s own move');
    assert.equal(arrivals(), 0);
});

test('a question opened during the reach wins: its page is drawn, nothing lands', async (t) => {
    const { r, open, hold } = deepRoom(t, chain(2), { projectId: 'deep' });
    const release = hold(2);
    await open();
    await until(() => r.pageReads.length === 2);
    const kept = r.messages.scrollTop;
    globalThis.document.body = globalThis.document.createElement('body');  // where "Question unavailable." shows
    void r.instance.revealQuestion('task-q', 'quiz-q');
    release();
    await until(() => r.shown().includes('chat:9'));
    await quiet();
    assert.deepEqual(r.pageReads, [1, 2]);
    assert.equal(r.messages.scrollTop, kept, 'the question owns the room: the newest message is not landed on');
});

for (const [name, close] of [['closed', (r) => r.instance.destroy()], ['hidden', (r) => { r.instance.page.hidden = true; }]]) {
    test(`a room ${name} during the reach asks for no further page`, async (t) => {
        const { r, open, hold, arrivals } = deepRoom(t, chain(3));
        const release = hold(2);
        await open();
        await until(() => r.pageReads.length === 2);
        close(r);
        release();
        await quiet();
        assert.deepEqual(r.pageReads, [1, 2]);
        assert.equal(arrivals(), 0);
    });
}

test('a window hidden during the reach ends it as a hidden room does: shown again, no page, jump or read', async (t) => {
    const { r, open, hold, arrivals } = deepRoom(t, chain(3));
    const release = hold(2);
    await open();
    await until(() => r.pageReads.length === 2);
    r.setHidden(true);
    release();
    await quiet();
    assert.deepEqual(r.pageReads, [1, 2], 'the page already asked for is drawn; nothing more is asked');
    r.setHidden(false);
    const later = await r.instance.refreshHistory({ revision: 4 });
    await quiet();
    assert.deepEqual([r.pageReads, r.messages.scrollTop, later.read, arrivals()], [[1, 2], r.messages.scrollHeight, false, 0],
        'the spent opening neither pages nor lands; the dot stays');
});

test('a failed page ends the reach at the bottom; its repair later moves no one', async (t) => {
    const { r, open, arrivals } = deepRoom(t, chain(3));
    r.server.pageFails = 2;
    await open();
    await quiet();
    assert.deepEqual(r.pageReads, [1, 2], 'no automatic retry');
    assert.equal(r.messages.scrollTop, r.messages.scrollHeight, 'the opening stayed at the bottom, the dot with it');
    assert.equal(arrivals(), 0);
    const later = await r.instance.refreshHistory({ revision: 4 });
    await quiet();
    assert.deepEqual([r.pageReads, r.messages.scrollTop, later.read], [[1, 2], r.messages.scrollHeight, false],
        'a later read finds no opening owed: no page, no jump');
});

test('a reopening owns its landing: the superseded reach ends without spending it', async (t) => {
    const { r, open, hold } = deepRoom(t, chain(2));
    const release = hold(2);
    await open();
    await until(() => r.pageReads.length === 2);
    r.server.pageFails = 2;  // the first opening's page fails after the room is opened again
    const again = open();
    release();
    await again;
    await until(() => r.shown().includes('chat:9') && r.messages.scrollTop === 0);
    assert.deepEqual(r.pageReads, [1, 2, 2], 'the newer opening goes on from the same page, one request at a time');
});

test('a newer arrival during the reach is where the one landing goes', async (t) => {
    const { r, open, hold } = deepRoom(t, chain(3));
    const release = hold(2);
    await open();
    await until(() => r.pageReads.length === 2);
    r.server.rows = [row(300), row(301), row(302)];
    r.server.window = { complete: true, truncated_by: [], latest_message: { history_id: 'chat:302', out_of_order: false } };
    assert.equal((await r.instance.refreshHistory({ revision: 5 })).read, true, 'the newer message is on screen');
    release();
    await quiet();
    assert.deepEqual(r.pageReads, [1, 2], 'the newest message is drawn: the reach stops there');
    assert.equal(r.messages.scrollTop, r.messages.scrollHeight);
});

// A searched opening: the recent read's bounded search ran out (`latest_before`).
const search = { complete: false, truncated_by: ['quota'], latest_message: null, latest_before: 15 };
const notes = (n) => ({ ...row(n), role: 'user', text: `owner note ${n}` });
function searchedOpening(t, last, upper = 5) {
    const opening = deepRoom(t, [{ rows: [notes(14)], window: {}, coverage: coverage(100, 12) },
        { rows: [notes(11)], window: {}, coverage: coverage(100, 10) }, { ...last, coverage: coverage(100, upper) }], { window: search });
    opening.r.server.coverage = coverage(100, 20);
    return opening;
}

test('a searched opening reads its quiet chain until a page names the newest message, then lands', async (t) => {
    const { r, open, arrivals } = searchedOpening(t, { rows: [late(9), notes(6)],
        window: { latest_message: { history_id: 'chat:9', out_of_order: true } } });
    assert.equal((await open()).painted, false, 'unknown to the opening read');
    await until(() => r.messages.scrollTop === 0 && arrivals() === 1);
    assert.deepEqual(r.pageReads, [1, 2, 3]);
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, true);
});

test('a searched opening whose chain reaches the start with no message lets the bottom decide', async (t) => {
    const { r, open } = searchedOpening(t, { rows: [notes(3)], window: { latest_absent: true } }, 0);
    await open();
    await quiet();
    assert.deepEqual(r.pageReads, [1, 2, 3], 'absence is proven only by the page reaching the start');
    assert.equal(r.messages.scrollTop, r.messages.scrollHeight, 'the bottom decides');
});

// Past the pages kept the newest page went, so the bottom drawn is not the room's (`canNewer`): the opening
// returns there once, as ↓ does, and reaches what that fresh read names (a late answer meanwhile) the same way.
for (const after of ['bottom', 'kept', 'moved', 'failed', 'named', 'named, moved']) {
    test(`a clean start proven at the chat's start returns the opening to the newest page (${after})`, async (t) => {
        const quietPage = (n, from) => ({ rows: [notes(n)], window: {}, coverage: coverage(100, from) });
        const { r, open, hold, arrivals } = deepRoom(t, [quietPage(14, 12), quietPage(11, 10), quietPage(8, 7),
            { ...quietPage(3, 0), window: { latest_absent: true } }],
        { window: search, top: 'chat:109', walked: after === 'kept' ? null : /^chat:(14|11|8|3)$/ });
        r.server.coverage = coverage(100, 20);
        const start = hold(4), arrived = after.startsWith('named'), moved = after.endsWith('moved');
        await open();
        await until(() => r.pageReads.length === 4);
        const reads = r.reads.length, present = after === 'kept' ? () => {} : r.gate();
        if (arrived) Object.assign(r.server, { rows: [notes(180), notes(190)], coverage: coverage(200, 150),
            window: { ...named, latest_message: { history_id: 'chat:109', out_of_order: true } } });
        start();
        await until(() => after === 'kept' || r.reads.length === reads + 1);
        if (arrived) r.server.pages = [{ rows: [notes(160)], window: {} }, { rows: [late(109), notes(150)], window: {} }];
        let kept, chase = arrived && moved ? hold(1) : () => {};
        if (moved && !arrived) { r.readUp(); kept = r.messages.scrollTop; }
        present(after === 'failed' ? 'fail' : null);
        await until(() => r.pageReads.length >= (arrived ? 5 : 4));
        if (moved && arrived) { wheel(r, 1); kept = r.messages.scrollTop; }  // not up, where the answer is drawn
        chase();
        await quiet();
        assert.deepEqual([r.pageReads, r.reads.length - reads, arrivals(), moved && r.messages.scrollTop === kept],
            [[1, 2, 3, 4, ...{ named: [1, 2], 'named, moved': [1] }[after] || []], after === 'kept' ? 0 : 1, ['bottom', 'kept', 'named'].includes(after) ? 1 : 0,
                moved], 'one walk, one read of the present past the pages kept, then only the page with what it names; a move ends it');
        if (moved) return;
        const retry = r.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn').textContent;
        assert.deepEqual([r.messages.scrollTop, retry, r.shown().some((id) => /^chat:(14|11|8|3)$/.test(id))],
            [arrived ? 0 : r.messages.scrollHeight, after === 'failed' ? 'Retry loading messages' : 'Load more history',
                after === 'kept' || after === 'failed'], 'landed once; a failed present keeps the bottom, its pages and Retry');
        const top = r.messages.scrollTop, later = await r.instance.refreshHistory({ revision: 4 });
        await quiet();
        assert.deepEqual([later.read, r.pageReads.length, r.reads.length - reads, r.messages.scrollTop],
            [after !== 'failed', arrived ? 6 : 4, after === 'kept' ? 1 : 2, top], 'a later read reads it, or pages and moves nothing');
    });
}

// A room that grew past its pager's chain while kept (a horizon gap) reopens: ↓'s read
// of the present is in flight when the opening read lands, so the reach waits for it.
for (const hidden of [true, false]) {
    test(`a reach waiting on a failed read of the present ${hidden ? 'in a room hidden meanwhile asks for nothing'
        : 're-anchors the chain itself and goes on'}`, async (t) => {
        const { r, open } = deepRoom(t, chain(3));
        for (const [revision, read] of [[1, coverage(100, 20)], [2, coverage(300, 200)]]) {
            r.server.coverage = read;
            await r.instance.refreshHistory({ revision });
        }
        const reads = r.reads.length;
        r.server.gates.push(null);  // the opening read
        const present = r.gate();
        void open();
        await until(() => r.reads.length === reads + 2);
        await quiet();
        if (hidden) r.instance.page.hidden = true;
        present('fail');
        await quiet();
        assert.deepEqual([r.reads.length - reads, r.pageReads], hidden ? [2, []] : [3, [1, 2, 3]],
            hidden ? 'no rebase at the present, no older page' : 'the reach rebases at the present first, then reads on');
    });
}

// Its newest message on the opening read's own page, an unread reopening has nothing to
// reach: the read re-anchors the outgrown chain at itself (no second read of the same
// page) as a plain reopening's does. One older than that page is reached on the
// re-anchored chain, with no second return to the present.
for (const reopened of ['read', 'unread', 'unread, newest older']) {
    test(`a kept room grown past its chain is re-anchored when reopened ${reopened}`, async (t) => {
        const deep = reopened.endsWith('older');
        const r = room(t, { projectId: 'kept' });
        placeAtTop(t, r, 'chat:9');
        const newest = (n) => ({ ...named, latest_message: { history_id: `chat:${n}`, out_of_order: false } });
        Object.assign(r.server, { rows: [row(300), row(301)], coverage: coverage(100, 20), window: newest(301),
            pages: [{ rows: [row(200)], window: {}, coverage: coverage(100, 10) }] });
        await r.instance.refreshHistory({ revision: 1 });
        await quiet();
        const reads = r.reads.length;
        Object.assign(r.server, { rows: [row(400), row(401)], coverage: coverage(300, 200), window: newest(deep ? 9 : 401) });
        if (deep) r.server.pages = [{ rows: [row(350)], window: {} }, { rows: [late(9)], window: {} }];
        void r.instance.showLatest({ arrive: reopened !== 'read' });
        r.instance.cancelHistoryPaint();
        const paint = await r.instance.refreshHistory({ revision: 2 });
        await quiet();
        assert.deepEqual([r.reads.length - reads, r.pageReads, paint.read, r.messages.scrollTop === 0], [1, deep ? [1, 2] : [], !deep, deep],
            'one read, the chain re-anchored at it; only a newest message older than its page is walked to and landed on');
    });
}

test('a broken quiet chain ends the search: no guess at what lies below', async (t) => {
    const { r, open } = searchedOpening(t, { rows: [late(9)], window: { latest_message: { history_id: 'chat:9' } } });
    r.server.pages[0].window = { latest_message: null };  // unreadable: this page cannot carry the search on
    await open();
    await quiet();
    assert.deepEqual(r.pageReads, [1]);
    assert.equal(r.messages.scrollTop, r.messages.scrollHeight);
});

test('only an older page can unsay what the quiet chain found, not the recent page read again', async (t) => {
    const raf = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (fn) => setImmediate(fn);
    t.after(() => { globalThis.requestAnimationFrame = raf; });
    const receipt = createProjectReadReceipt({ read: async () => true, isShown: () => true,
        isReadingLatest: (latest) => latest?.history_id === 'chat:9' });
    receipt.recent({ window: search, coverage: coverage(100, 20) });
    receipt.page({ index: 1, window: { latest_message: { history_id: 'chat:9', out_of_order: true } }, coverage: coverage(100, 5) });
    assert.equal((await receipt.refresh({ revision: 4 })).read, true);
    receipt.page({ index: 0, window: { latest_message: null }, coverage: coverage(100, 20) });
    assert.equal((await receipt.refresh({ revision: 4 })).read, true, 'frozen at its boundary, it cannot tell');
    receipt.page({ index: 1, window: { latest_message: null }, coverage: coverage(100, 5) });
    assert.equal((await receipt.refresh({ revision: 4 })).painted, false, 'an unreadable older page ends what it carried');
});

// A reader paging older past the pages kept, then down, reads the recent page again
// (`newer` to index 0). history_paging.latest_arrival names nothing there.
test('the recent page read again keeps the newest message an older page found readable on screen', async (t) => {
    const pages = [
        { messages: [notes(900)], coverage: coverage(1000, 800), window: { ...search, latest_before: 500 } },
        { messages: [late(100), notes(110)], coverage: coverage(1000, 100),
            window: { latest_message: { history_id: 'chat:100', out_of_order: true } } },
        { messages: [notes(90)], coverage: coverage(1000, 90), window: {} },
        { messages: [notes(80)], coverage: coverage(1000, 0), window: {} },
    ];
    const cursors = [];
    let landed = false;
    const r = room(t, { fetchHistory: async (url) => {
        const cursor = new URL(url, 'http://local').searchParams.get('cursor'), k = cursor ? Number(cursor.slice(1)) : 0;
        cursors.push(cursor ?? 'recent');
        return { ok: true, json: async () => ({ ...pages[k], window: cursor === 'p0' ? { latest_message: null } : pages[k].window,
            page_cursor: `p${k}`, next_cursor: k < 3 ? `p${k + 1}` : null, has_more: k < 3 }) };
    } });
    const rect = ElementStub.prototype.getBoundingClientRect;
    ElementStub.prototype.getBoundingClientRect = function () {
        const id = this.dataset?.historyId;
        if (id === 'chat:100') return landed ? r.box(0, 20) : r.box(-100, -80);
        if (id === 'chat:110') return r.box(0, 20);  // on screen: its page stays
        return id ? r.box(-500, -480) : rect.call(this);
    };
    t.after(() => { ElementStub.prototype.getBoundingClientRect = rect; });
    void r.instance.showLatest();
    await r.instance.refreshHistory({ revision: 4 });
    const older = r.messages.querySelector('.chat-load-older').querySelector('.chat-load-older-btn');
    for (let i = 0; i < 3; i += 1) for (const click of older.listeners.get('click') || []) await click({ target: older });
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, false, 'found, but not on screen');
    wheel(r, 1);
    await quiet();
    assert.deepEqual(cursors, ['recent', 'p1', 'p2', 'p3', 'recent', 'p0'], 'page zero went and was read again');
    landed = true;
    r.scroll(0);
    assert.equal((await r.instance.refreshHistory({ revision: 4 })).read, true);
});

test('an unknown newest message is never searched', async (t) => {
    const { r, open } = deepRoom(t, chain(2), { window: { complete: false, truncated_by: ['chat_malformed_jsonl'], latest_message: null } });
    await open();
    await quiet();
    assert.deepEqual(r.pageReads, [], 'no guess: the honest gap stays');
});

test('a read room and a room opened without an unread revision read no older page', async (t) => {
    const { r } = deepRoom(t, chain(2));
    void r.instance.showLatest();
    const plain = await r.instance.refreshHistory({ revision: 4 });
    await quiet();
    assert.deepEqual([r.pageReads, plain.read], [[], false], 'it opens at the bottom, as before');
});

test('the page holding the named message ends the reach even where a task card shows it', async (t) => {
    const placed = { role: 'system', system_type: 'custody_notice', card_row: 'timeline', card_row_id: 'final:old-task:custody',
        task_id: 'old-task', text: 'Custody settled', ts: TS, history_id: 'chat:9', history_position: { source: 'chat', offset: 9 } };
    const { r, open } = deepRoom(t, chain(3, [placed]).slice(1).concat(chain(1)));
    await open();
    await quiet();
    assert.deepEqual(r.pageReads, [1, 2], 'its page is drawn: no chase through the rest of the room');
});
