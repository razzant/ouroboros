// Replies of one chat (chat_reply.js) over a DOM double: a row's quote and Reply control, the relay panel (shape F:
// one Ouroboros bubble, his words above the note), the host-derived quote that repeats nothing directly under its
// message, the composer strip and what the frame carries, the owner's echo adopting its address, the question
// card's own answer door, and a missing fact leaving today's bubble.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatReplies } from '../modules/chat_reply.js';
import { excerptBody } from '../modules/chat_reply_excerpt.js';
import { createMessageActions } from '../modules/message_actions.js';
import { createChatReadingPosition } from '../modules/chat_reading_position.js';
import { installDomDouble } from './dom_double.js';
import { applyPayload } from '../modules/i18n.js';

function fixture({ actions = undefined } = {}) {
    const env = installDomDouble();
    const { doc, el } = env;
    const feed = el('div', { className: 'chat-messages' });
    const input = el('textarea', { attrs: { id: 'chat-input' } });
    const textRow = el('div', { className: 'chat-text-row' }, input);
    // The composer's markup: the field holds the text row, and a staged strip becomes its first line.
    const field = el('div', { className: 'chat-field' }, textRow);
    const wrap = el('div', { className: 'chat-input-wrap' }, field);
    doc.body.append(feed, wrap);
    const positions = [];
    const replies = createChatReplies({ chatId: 1, feed, input, key: 'ouro_chat_reply', actions,
        reading: () => ({ cancel() { positions.push('cancel'); }, scroll() { positions.push('scroll'); }, remember() {}, stick: true }) });
    const bubble = (role, text, { address = '', taskId = '' } = {}) => {
        const node = el('div', { className: `chat-bubble ${role}`, dataset: { ...(taskId ? { taskId } : {}) } },
            el('div', { className: 'sender', text: role === 'user' ? 'You' : 'Ouroboros' }),
            el('div', { className: 'message', text }),
            el('div', { className: 'msg-time', text: '09:40' }));
        if (address) node.dataset.ts = String(Date.parse('2026-10-16T09:40:00Z'));
        return node;
    };
    const quote = (node) => node.children.find((child) => child.classList.contains('chat-quote')) || null;
    // The strip the owner sees; one that is closing (shut) has already left and goes once its motion has run.
    const strips = () => field.children.filter((child) => child.classList.contains('chat-quote--composer'));
    const strip = () => strips().find((child) => child.dataset.motion !== 'shut') || null;
    const leaving = () => strips().filter((child) => child.dataset.motion === 'shut');
    return { ...env, feed, input, textRow, field, wrap, replies, bubble, quote, strip, leaving, positions,
        close() { replies.destroy(); env.restore(); } };
}

const owner = (address, text = 'please check the logs') => ({ row_address: address, role: 'user', text });

test('an explicit quote landing replaces the old-width anchor before reflow; later reading owns its place', async t => {
    const env = installDomDouble(), frames = [], prior = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = callback => frames.push(callback);
    t.after(() => { globalThis.requestAnimationFrame = prior; env.restore(); });
    const feed = env.el('div'), input = env.el('textarea');
    const row = env.el('div', { dataset: { rowAddress: 'row:1@then#123456789abc' } });
    env.doc.body.append(feed, input); feed.append(row);
    Object.assign(feed, { clientWidth: 0, clientHeight: 300, scrollHeight: 1200, scrollTop: 0 });
    feed.getBoundingClientRect = () => ({ top: 0 });
    row.getBoundingClientRect = () => ({ top: 600 - feed.scrollTop });
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
        anchors: { capture: () => ({ top: feed.scrollTop }), restore: anchor => { feed.scrollTop = anchor.top; return true; } },
        changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    const unbind = reading.bindGestures(() => {});
    const replies = createChatReplies({ feed, input, reading: () => reading,
        history: () => ({ exact: () => { feed.clientWidth = 389; return row; } }) });
    t.after(() => { replies.destroy(); unbind(); });
    assert.deepEqual(await replies.reveal(row.dataset.rowAddress), { state: 'found', target: row }, 'a landing names its row');
    assert.equal(env.doc.activeElement, row);
    assert.equal(feed.scrollTop, 528);
    reading.reflow();
    assert.equal(feed.scrollTop, 528, 'the queued resize must keep the measured landing, not restore the opening at 0');
    feed.emit('wheel', { deltaY: -120 });
    feed.scrollTop = 408; reading.scroll();
    frames.splice(0).forEach(callback => callback());
    feed.clientWidth = 360; reading.reflow();
    assert.equal(feed.scrollTop, 408, 'a later resize preserves the reader’s gesture, not the quote landing');
});

test('a quote landing clears the bar and its frosted edge; without an edge it clears the bar alone', async t => {
    const land = async (edge) => {
        const env = installDomDouble(), view = env.doc.defaultView, prior = view.getComputedStyle;
        view.getComputedStyle = (node) => ({ ...prior(node), getPropertyValue: (name) => (name === '--edge-reach'
            && String(node.className || '').split(/\s+/).includes('chat-page-header') ? edge : '') });
        const page = env.el('div', { className: 'page' }), header = env.el('div', { className: 'chat-page-header' });
        const feed = env.el('div'), input = env.el('textarea');
        const row = env.el('div', { dataset: { rowAddress: 'row:1@then#123456789abc' } });
        env.doc.body.append(page, input); page.append(header, feed); feed.append(row);
        Object.assign(feed, { clientWidth: 0, clientHeight: 300, scrollHeight: 1200, scrollTop: 0 });
        feed.getBoundingClientRect = () => ({ top: 0 });
        header.getBoundingClientRect = () => ({ top: 0, bottom: 80 });
        row.getBoundingClientRect = () => ({ top: 600 - feed.scrollTop });
        const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
            anchors: { capture: () => ({ top: feed.scrollTop }), restore: anchor => { feed.scrollTop = anchor.top; return true; } },
            changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
        const replies = createChatReplies({ feed, input, reading: () => reading, history: () => ({ exact: () => row }) });
        try {
            assert.deepEqual(await replies.reveal(row.dataset.rowAddress), { state: 'found', target: row });
            return feed.scrollTop;
        } finally {
            replies.destroy();
            view.getComputedStyle = prior;
            env.restore();
        }
    };
    // The bar ends at 80 px: a row lands 12 px below its 32 px edge (style.css "A room's edges"), or below the bar alone.
    assert.equal(await land('32px'), 600 - (80 + 32 + 12));
    assert.equal(await land('0px'), 600 - (80 + 12));
});

test('a drawn row is a destination app needs no read for; a portal or a row not drawn is not', async t => {
    const env = installDomDouble();
    t.after(() => env.restore());
    const feed = env.el('div'), input = env.el('textarea');
    env.doc.body.append(feed, input);
    const card = env.el('div', { dataset: { rowAddress: 'task:c0ffee42' } });
    const portal = env.el('div', { dataset: { rowAddress: 'task:d3adb33f', portal: '1' } });
    const drawn = new Map([[card.dataset.rowAddress, card], [portal.dataset.rowAddress, portal]]);
    const pinned = [];
    const replies = createChatReplies({ feed, input, reading: () => ({ cancel() {}, scroll() {}, remember() {}, stick: true }),
        history: () => ({ exact: (address) => drawn.get(address) || null, pin: (address, node) => pinned.push([address, node]) }) });
    t.after(() => replies.destroy());
    assert.equal(replies.draws('task:c0ffee42'), true);
    assert.equal(replies.draws('task:d3adb33f'), false, 'its work lives in another room, which the read names');
    assert.equal(replies.draws('task:0badf00d'), false);
    assert.deepEqual(pinned, [], 'draws() inspection never sets an explicit pin');
    assert.deepEqual(await replies.reveal('task:c0ffee42', () => false), null);
    assert.deepEqual(pinned, [], 'a superseded reveal never sets an explicit pin');
    assert.deepEqual(await replies.reveal('task:c0ffee42'), { state: 'found', target: card });
    assert.deepEqual(pinned, [['task:c0ffee42', card]], 'only a successful mounted fastpath landing notifies history.pin');
});

test('a reveal that moves nothing hands a following reader the live edge back; a gesture, newer reveal or exit decides instead', async t => {
    const env = installDomDouble(), frames = [], prior = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = callback => frames.push(callback);
    t.after(() => { globalThis.requestAnimationFrame = prior; env.restore(); });
    const feed = env.el('div'), input = env.el('textarea');
    env.doc.body.append(feed, input);
    Object.assign(feed, { clientWidth: 400, clientHeight: 300, scrollHeight: 1200, scrollTop: 900 });
    feed.getBoundingClientRect = () => ({ top: 0 });
    const reading = createChatReadingPosition({ feed, visible: () => true, alive: () => true, ready: () => true,
        anchors: { capture: () => ({ top: feed.scrollTop }), restore: anchor => { feed.scrollTop = anchor.top; return true; } },
        changed() {}, afterWrite() {}, activity() {}, updateButton() {} });
    const unbind = reading.bindGestures(() => {});
    const answers = [];
    const replies = createChatReplies({ feed, input, reading: () => reading,
        history: () => ({ exact: () => null, reference: () => new Promise(resolve => answers.push(resolve)) }) });
    t.after(() => { replies.destroy(); unbind(); });
    const settle = () => { while (frames.length) frames.splice(0).forEach(callback => callback()); };
    const arrive = () => reading.mutate(() => { feed.scrollHeight += 400; }, { remoteContent: true });
    const address = 'row:1@then#aaaaaaaaaaaa';

    let shown = replies.reveal(address);
    assert.equal(reading.stick, false, 'an arrival does not pull the view while the read is pending');
    arrive();
    assert.equal(reading.nearBottom(), false);
    answers.shift()({ state: 'unknown' });
    assert.deepEqual(await shown, { state: 'unknown' });
    settle();
    assert.equal(reading.stick, true);
    assert.equal(reading.nearBottom(), true, 'what arrived while it loaded is shown');
    arrive();
    assert.equal(reading.nearBottom(), true, 'and the next reply is followed, with no ↓');

    shown = replies.reveal(address);
    feed.emit('wheel', { deltaY: -120 });  // the reader turns back while it loads
    settle();
    answers.shift()({ state: 'unknown' });
    assert.equal(await shown, null);
    settle();
    assert.equal(reading.stick, false, 'the gesture decided; the failed read does not override it');

    reading.stick = true; reading.followAfterLayout(); settle();  // ↓
    const older = replies.reveal(address), newer = replies.reveal('row:1@then#bbbbbbbbbbbb');
    answers.shift()({ state: 'unknown' });
    assert.equal(await older, null, 'the newer reveal superseded it');
    assert.equal(reading.stick, false);
    answers.shift()({ state: 'absent' });
    assert.deepEqual(await newer, { state: 'absent' });
    settle();
    assert.equal(reading.stick, true, 'the newer failure keeps the intent the reader had before both');

    shown = replies.reveal(address);
    reading.cancel();  // the room left the screen (closeTransient)
    answers.shift()({ state: 'unknown' });
    assert.equal(await shown, null);
    assert.equal(reading.stick, false);
    reading.stick = true;
    shown = replies.reveal(address);
    answers.shift()({ state: 'unavailable', left: true });  // a page was drawn around it, without the exact row
    assert.deepEqual(await shown, { state: 'unavailable', left: true });
    settle();
    assert.equal(reading.stick, false, 'the drawn page is where the reader now is');
});

test('a quote that cannot land says why; only a read that may answer differently asks to try again', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const env = installDomDouble();
    t.after(() => env.restore());
    env.doc.getElementById = id => env.doc.body.descendants().find(node => node.id === id) || null;
    const feed = env.el('div'), input = env.el('textarea');
    const bubble = env.el('div', { className: 'chat-bubble user' }, env.el('div', { className: 'sender', text: 'You' }),
        env.el('div', { className: 'message', text: 'and this one?' }));
    env.doc.body.append(feed, input); feed.append(bubble);
    const answers = [];
    const replies = createChatReplies({ feed, input, reading: () => ({ cancel() {}, scroll() {}, remember() {}, stick: false }),
        history: () => ({ exact: () => null, async reference() {
            const next = answers.shift();
            if (next instanceof Error) throw next;
            return next;
        } }) });
    t.after(() => replies.destroy());
    replies.decorate(bubble, { role: 'user', reply_to: { address: 'task:work-1', role: 'assistant', kind: 'work_card', snippet: 'Work' } });
    const typed = (status, reasonCode) => Object.assign(new Error(`HTTP ${status}`), { status, body: { reason_code: reasonCode } });
    const said = [];
    for (const answer of [{ state: 'moved', task_id: 'work-1' }, { state: 'unknown', error: typed(503, 'history_source_unavailable') },
        { state: 'unknown', error: typed(503, 'migrating') }, { state: 'unknown' }, new TypeError('offline')]) {
        answers.push(answer);
        bubble.querySelector('.chat-quote-main').click();
        await new Promise(resolve => setImmediate(resolve));
        said.push(env.doc.body.querySelectorAll('.toast').at(-1)?.textContent);
    }
    assert.deepEqual(said, [
        'This work is in another room.',
        'Saved history that may hold the quoted message could not be read.',
        'Saved history is being upgraded. Try again shortly.',
        'Could not load the quoted message. Try again.',
        'Could not load the quoted message. Try again.',
    ]);
});

test('a derived question quote that names another room is read at its address, never landed on the copy drawn under it', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const env = installDomDouble();
    t.after(() => env.restore());
    const feed = env.el('div'), input = env.el('textarea');
    // Main draws a Project's question as a pointer copy under the card's own address.
    const copy = env.el('div', { className: 'chat-bubble assistant project-question', dataset: { rowAddress: 'row:1@q#1' } });
    env.doc.body.append(feed, input); feed.append(copy);
    const reads = [], moves = [];
    let answer = { state: 'moved', chat_id: 7 };
    const replies = createChatReplies({ feed, input, reading: () => ({ cancel() {}, scroll() {}, remember() {}, stick: false }),
        history: () => ({ exact: (address) => feed.children.find((node) => node.dataset.rowAddress === address) || null,
            async reference(address) { reads.push(address); return answer; } }),
        navigate: async (address, chatId) => { moves.push([address, chatId]); return { state: 'found' }; } });
    t.after(() => replies.destroy());
    const question = { address: 'row:1@q#1', role: 'assistant', kind: 'question', author_label: 'Ouroboros', snippet: 'Which format?',
        ts: '2026-11-23T10:00:00Z' };
    const bubbles = new Map(Object.entries({
        away: { ...question, derived_question: true, room_label: 'Weekly report', ts: question.ts },
        home: { ...question, derived_question: true },  // the host names no other room: the card is here
        ordinary: { ...question },  // an owner's own Reply to what he saw, the copy included
    }).map(([name, reply_to]) => {
        const bubble = env.el('div', { className: 'chat-bubble user' }, env.el('div', { className: 'sender', text: 'You' }),
            env.el('div', { className: 'message', text: name }));
        feed.append(bubble);
        replies.decorate(bubble, { role: 'user', row_address: `row:1@a#${name}`, reply_to });
        return [name, bubble];
    }));
    const press = async (name) => {
        bubbles.get(name).querySelector('.chat-quote-main').click();
        await new Promise((resolve) => setImmediate(resolve));
        return [reads.length, moves.length, env.doc.body.querySelectorAll('.toast').at(-1)?.textContent];
    };
    const ringed = () => feed.children.filter((node) => node.classList.contains('is-quote-target'));

    // Another room is the card's home: the read of its address names it, the copy is neither ringed nor focused.
    assert.deepEqual(await press('away'), [1, 1, undefined]);
    assert.deepEqual([reads, moves], [['row:1@q#1'], [['row:1@q#1', 7]]]);
    assert.deepEqual(ringed(), []);
    assert.notEqual(env.doc.activeElement, copy);
    // Its original cannot be found: the view stays and says so; the copy stands in for nothing.
    answer = { state: 'absent' };
    assert.deepEqual(await press('away'), [2, 1, 'The quoted message was not found in saved history.']);
    assert.deepEqual(ringed(), []);
    // The card is here, or the quote is the owner's own Reply: what is drawn under the address lands, with no read.
    assert.deepEqual(await press('home'), [2, 1, 'The quoted message was not found in saved history.']);
    assert.deepEqual(ringed(), [copy]);
    assert.equal(env.doc.activeElement, copy);
    copy.classList.remove('is-quote-target');
    assert.deepEqual(await press('ordinary'), [2, 1, 'The quoted message was not found in saved history.']);
    assert.deepEqual(ringed(), [copy]);
});

test('a saved row gets Reply beside Copy; a row without an address (an unsaved echo, an older host) stays as today', () => {
    const f = fixture();
    try {
        const saved = f.bubble('user', 'please check the logs');
        const copy = f.el('button', { className: 'chat-message-copy' });
        saved.append(copy);
        f.feed.append(saved);
        f.replies.decorate(saved, owner('row:1@a#1'));
        assert.equal(saved.dataset.rowAddress, 'row:1@a#1');
        const actions = saved.children.find((child) => child.classList.contains('chat-message-actions'));
        assert.deepEqual(actions.children.map((child) => child.className.split(' ')[0]), ['chat-message-reply', 'chat-message-copy'],
            'one control row: Reply, then Copy');
        assert.ok(actions.children[0].classList.contains('ui-hit-area'), 'Reply gains the shared finger hit area');
        assert.equal(actions.children[0].getAttribute('aria-label'), 'Reply', 'an icon-only control has a name');
        assert.ok(saved.classList.contains('has-reply'));
        const legacy = f.bubble('assistant', 'an answer');
        f.feed.append(legacy);
        f.replies.decorate(legacy, { role: 'assistant', text: 'an answer' });
        assert.equal(legacy.dataset.rowAddress, undefined);
        assert.equal(legacy.children.some((child) => child.classList.contains('chat-message-actions')), false);
        assert.equal(f.quote(legacy), null);
    } finally { f.close(); }
});

test('an owner press is still a whole-row Reply target, not an authored passage', t => {
    const f = fixture();
    t.after(() => f.close());
    const press = f.bubble('user', '1. Take the ridge');
    const comment = f.bubble('user', 'Follow the creek at dawn.');
    f.feed.append(press, comment);
    for (const row of [press, comment]) row.querySelector('.message').dataset.replyBody = '1';
    f.replies.decorate(press, { ...owner('row:1@press#1'), reply_excerpt_disabled: true });
    f.replies.decorate(comment, owner('row:1@comment#2'));
    assert.equal(excerptBody(press), null);
    assert.equal(excerptBody(comment), comment.querySelector('.message'));
    assert.equal(f.replies.bubbleRef(press).role, 'user', 'it is still the owner\'s bubble: only its passages are off');
    press.querySelector('.chat-message-reply').click();
    assert.deepEqual(f.replies.frameField(), { reply_to: 'row:1@press#1' });
});

test('a Reply to a System row signs System, whatever sender line it wears or label an older frame stamped', t => {
    const f = fixture();
    t.after(() => f.close());
    const opening = f.bubble('system', 'Call opened: say hello.');  // this double's sender line says Ouroboros
    const summary = f.bubble('system', 'Task finished.');
    summary.querySelector('.sender').textContent = 'Task Summary';   // a sender line System rows wear
    f.feed.append(opening, summary);
    f.replies.decorate(opening, { row_address: 'row:1@call#1', role: 'system', text: 'Call opened', reply_author_label: 'You' });
    f.replies.decorate(summary, { row_address: 'row:1@sum#2', role: 'system', text: 'Task finished.' });
    for (const row of [opening, summary]) assert.equal(f.replies.bubbleRef(row).author_label, 'System');
    opening.querySelector('.chat-message-reply').click();
    const staged = JSON.parse(f.storage.get('ouro_chat_reply'));
    assert.deepEqual([staged.role, staged.author_label], ['system', 'System']);
    assert.equal(f.strip().querySelector('.chat-quote-author').textContent, 'System');
    const mine = f.bubble('user', 'a note');  // the owner's own bubble keeps the host's label for its author
    f.feed.append(mine);
    f.replies.decorate(mine, { ...owner('row:1@mine#3'), reply_author_label: 'Mira' });
    assert.equal(f.replies.bubbleRef(mine).author_label, 'Mira');
});

test('Reply stages the strip and the next frame carries only the address; Esc or × cancels; a send clears it', async () => {
    const f = fixture();
    try {
        const saved = f.bubble('assistant', 'The audit found three stale projects.');
        f.feed.append(saved);
        f.replies.decorate(saved, { row_address: 'row:1@b#2', role: 'assistant' });
        saved.querySelector('.chat-message-reply').click();
        const strip = f.strip();
        assert.ok(strip, 'the strip is staged');
        assert.equal(f.field.children.indexOf(strip), 0, 'the field\'s first line, inside the field');
        assert.equal(f.field.children.indexOf(f.textRow), 1, 'right before its text row');
        assert.equal(f.doc.activeElement, f.input, 'the field takes focus');
        assert.match(f.input.getAttribute('aria-describedby'), /reply$/);
        assert.deepEqual(f.replies.frameField(), { reply_to: 'row:1@b#2' }, 'the host makes the snippet');
        assert.equal(f.replies.echoRow().reply_to.address, 'row:1@b#2', 'the local echo draws the same quote');
        assert.equal(JSON.parse(f.storage.get('ouro_chat_reply')).address, 'row:1@b#2', 'kept for a reload');
        assert.equal(strip.dataset.motion, 'open', 'it opens from no height (style.css runs it like the fold)');
        f.textRow.emit('keydown', { key: 'Escape' });
        assert.equal(f.strip(), null, 'Esc cancels');
        assert.deepEqual(f.leaving().map((node) => node.inert), [true], 'it closes before it goes, taking no press');
        assert.deepEqual(f.replies.frameField(), {});
        assert.equal(f.storage.has('ouro_chat_reply'), false);
        saved.querySelector('.chat-message-reply').click();
        f.strip().querySelector('.chat-quote-cancel').click();
        assert.equal(f.strip(), null, '× cancels');
        saved.querySelector('.chat-message-reply').click();
        f.replies.clear();
        assert.equal(f.strip(), null, 'a send clears it');
        assert.equal(f.replies.echoRow(), null);
        await new Promise((resolve) => setTimeout(resolve, 40));
        assert.equal(f.leaving().length, 0, 'a closed strip goes once its motion has run');
    } finally { f.close(); }
});

test('a staged strip is held shut through the first frame after staging and opens from the next', async () => {
    const frames = [], prior = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = (callback) => frames.push(callback);
    const f = fixture();
    try {
        const saved = f.bubble('assistant', 'The audit found three stale projects.');
        f.feed.append(saved);
        f.replies.decorate(saved, { row_address: 'row:1@b#2', role: 'assistant' });
        saved.querySelector('.chat-message-reply').click();
        const strip = f.strip();
        assert.deepEqual([strip.dataset.motion, 'hold' in strip.dataset], ['open', true],
            'staging focuses the field and lays it out: the strip waits that frame out shut');
        frames.splice(0).forEach((callback) => callback());
        await Promise.resolve();
        assert.deepEqual([strip.dataset.motion, 'hold' in strip.dataset], ['open', true],
            'the first frame lays the focused field out (the long one): still held');
        frames.splice(0).forEach((callback) => callback());
        await Promise.resolve();  // the frame's callback settles before that frame's style is worked out
        assert.deepEqual([strip.dataset.motion, 'hold' in strip.dataset], ['open', false], 'then it opens');
    } finally {
        f.close();
        globalThis.requestAnimationFrame = prior;
    }
});

test('a reply to a still-open question card answers in the card\'s own field, never a second path', () => {
    const f = fixture();
    try {
        const field = f.el('textarea', { className: 'chat-quiz-comment' });
        const card = f.el('div', { className: 'chat-bubble assistant', dataset: { rowAddress: 'row:1@q#3' } },
            f.el('div', { className: 'chat-quiz-card' }, f.el('div', { className: 'chat-quiz-question', text: 'Which project?' }), field));
        f.feed.append(card);
        f.replies.start(f.replies.bubbleRef(card));
        assert.equal(f.doc.activeElement, field);
        assert.equal(f.strip(), null);
        assert.equal(f.replies.bubbleRef(card).kind, 'question');
        field.disabled = true;  // settled: the card takes no answer any more
        f.replies.start(f.replies.bubbleRef(card));
        assert.ok(f.strip(), 'a settled question is quoted like any row');
    } finally { f.close(); }
});

test('the owner\'s reply shows his quote; an unresolved address says so; a quote jumps to its loaded row', () => {
    const f = fixture();
    try {
        const target = f.bubble('assistant', 'earlier answer');
        target.dataset.rowAddress = 'row:1@c#4';
        const reply = f.bubble('user', 'what about this?');
        f.feed.append(target, reply);
        f.replies.decorate(reply, { role: 'user', row_address: 'row:1@c#5',
            reply_to: { address: 'row:1@c#4', role: 'assistant', author_label: 'Ouroboros', kind: 'message', snippet: 'earlier answer' } });
        const panel = f.quote(reply);
        assert.equal(panel.dataset.author, 'assistant', 'the quoted author\'s ink');
        assert.equal(panel.dataset.derived, undefined, 'the owner\'s own reference is always shown');
        panel.querySelector('.chat-quote-main').click();
        assert.equal(f.doc.activeElement, target, 'the jump lands on the quoted row');
        assert.ok(target.classList.contains('is-quote-target'));
        const gone = f.bubble('user', 'and this?');
        f.feed.append(gone);
        f.replies.decorate(gone, { role: 'user', reply_to: { address: 'row:9@x#0', unresolved: true } });
        assert.match(f.quote(gone).querySelector('.chat-quote-text').textContent, /Open quoted message/);
        assert.equal(f.quote(gone).dataset.kind, 'message');
    } finally { f.close(); }
});

test('a host-derived quote on an answer repeats nothing directly under its message and shows once something came between', () => {
    const f = fixture();
    try {
        const ask = f.bubble('user', 'how many projects?');
        ask.dataset.rowAddress = 'row:1@d#1';
        const answer = f.bubble('assistant', 'Seven.', { taskId: 'turn-1' });
        const derived = { row_address: 'row:1@d#2', role: 'assistant',
            reply_to: { address: 'row:1@d#1', role: 'user', author_label: 'You', snippet: 'how many projects?' } };
        f.feed.append(ask, answer);
        f.replies.decorate(answer, derived);
        f.replies.syncDerivedQuotes(answer);
        assert.equal(f.quote(answer).dataset.derived, '1');
        assert.equal(f.quote(answer).hidden, true, 'directly under the message it answers');
        // The answer's own work card is part of the answer, not something between.
        const ownCard = f.el('div', { className: 'chat-live-card', dataset: { taskId: 'turn-1' } });
        f.feed.insertBefore(ownCard, answer);
        f.replies.syncDerivedQuotes(ownCard);
        assert.equal(f.quote(answer).hidden, true);
        // Another owner message (or another work's card) came between: the quote says what is answered.
        const other = f.bubble('user', 'and the logs?');
        f.feed.insertBefore(other, ownCard);
        f.replies.syncDerivedQuotes(other);
        assert.equal(f.quote(answer).hidden, false);
        other.remove();
        f.replies.syncDerivedQuotes();
        assert.equal(f.quote(answer).hidden, true, 'the full pass re-reads every derived quote');
    } finally { f.close(); }
});

test('relay shape F: one Ouroboros bubble, his words in a relay panel above the note, files named', () => {
    const f = fixture();
    try {
        const entry = f.bubble('assistant', 'Anton writes from Main: check the stalled projects.');
        f.feed.append(entry);
        f.replies.decorate(entry, { row_address: 'row:5@e#1', role: 'assistant', type: 'relay_entry', relay: {
            relay_id: 'r-1', text: 'проверь прерванные проекты\nи верни их к работе',
            attachments: [{ filename: 'shot.png', display_name: 'shot.png' }, { name: 'plan.md' }],
            source: { chat_id: 1, label: 'Main', ts: '2026-10-16T09:37:00Z', address: 'row:1@m#7' } },
            reply_to: { address: 'row:1@m#7', role: 'user', snippet: 'ignored: the relay panel already quotes him' } });
        const panels = entry.children.filter((child) => child.classList.contains('chat-quote'));
        assert.equal(panels.length, 1, 'one panel: the relay, never a second quote beside it');
        const [panel] = panels;
        assert.equal(panel.dataset.mode, 'relay');
        assert.equal(panel.dataset.relayId, 'r-1');
        assert.equal(entry.children.indexOf(panel) < entry.children.indexOf(entry.querySelector('.message')), true,
            'his words stand above the note');
        assert.equal(panel.querySelector('.chat-quote-text').textContent, 'проверь прерванные проекты\nи верни их к работе',
            'his complete words, verbatim');
        assert.equal(panel.querySelector('.chat-quote-files').textContent, 'Attached: shot.png, plan.md');
        assert.equal(panel.querySelector('.chat-quote-author').textContent, 'You');
        assert.match(panel.querySelector('.chat-quote-when').textContent, /Main/);
        assert.equal(entry.classList.contains('user'), false, 'voices never mixed: the bubble stays Ouroboros\'s');
        const owners = f.bubble('user', 'mine');
        f.feed.append(owners);
        f.replies.decorate(owners, { role: 'user', relay: { text: 'not a relay row' } });
        assert.equal(f.quote(owners), null, 'only an Ouroboros row carries a relay');
    } finally { f.close(); }
});

test('a canonical row read into a room adopts the retained copy it verified by origin, an id-less one included', () => {
    const f = fixture();
    try {
        const copy = f.bubble('user', 'Original request');
        Object.assign(copy.dataset, { originId: 'origin-a', originProjected: '1' });
        f.feed.append(copy);
        f.replies.decorate(copy, { role: 'user', origin_projected: true });
        assert.equal(copy.querySelector('.chat-message-reply'), null, 'a copy whose original is unread has no Reply');
        const canonical = { role: 'user', client_message_id: '', row_address: 'row:1@then#origin' };
        assert.equal(f.replies.adopt(canonical), false, 'an id-less row finds no bubble by itself');
        assert.equal(f.replies.adopt({ ...canonical, origin_projected: true }, copy), false, 'a copy verifies nothing');
        assert.equal(copy.dataset.rowAddress, undefined);
        assert.equal(f.replies.adopt(canonical, copy), true);
        assert.equal(copy.dataset.rowAddress, canonical.row_address);
        assert.ok(copy.querySelector('.chat-message-reply'));
        assert.equal(f.replies.adopt(canonical, copy), false, 'once');
    } finally { f.close(); }
});

test('the owner\'s echo adopts the saved row\'s address and with it Reply; Send as new prepares the composer', () => {
    const f = fixture();
    try {
        const echo = f.bubble('user', 'hello');
        echo.dataset.clientMessageId = 'cm-1';
        f.feed.append(echo);
        f.replies.decorate(echo, null);
        assert.equal(echo.querySelector('.chat-message-reply'), null, 'an unsaved echo has no address yet');
        assert.equal(f.replies.adopt({ role: 'user', client_message_id: 'cm-1', row_address: 'row:2@f#2' }), false);
        assert.equal(f.replies.adopt({ role: 'user', client_message_id: 'cm-1', row_address: 'row:1@f#1' }), true);
        assert.equal(echo.dataset.rowAddress, 'row:1@f#1');
        assert.ok(echo.querySelector('.chat-message-reply'));
        assert.equal(f.replies.adopt({ role: 'user', client_message_id: 'cm-1', row_address: 'row:2@f#2' }), false);
        assert.equal(echo.dataset.rowAddress, 'row:1@f#1', 'the same client id in another room is a different row');
        assert.equal(f.replies.adopt({ role: 'user', client_message_id: 'cm-1', row_address: 'row:1@f#1' }), false, 'once');
        assert.equal(f.replies.adopt({ role: 'assistant', client_message_id: 'cm-1', row_address: 'x' }), false);
    } finally { f.close(); }
});

test('a work card draws no Reply of its own; its row menu quotes the work by its task address', () => {
    // Owner 17.10 (Q9 B): Reply on a work card lives only in its row menu; the action row holds commands.
    const f = fixture();
    try {
        const actions = f.el('div', { className: 'chat-live-actions' });
        const root = f.el('div', { className: 'chat-live-card', dataset: { taskId: 't-9' } }, actions);
        f.feed.append(root);
        const record = { groupId: 't-9', root, footEl: null, footEndEl: null, titleEl: f.el('div', { text: 'Audit' }),
            activityEl: f.el('div', { text: 'Reading the third project' }), phaseEl: f.el('div', { text: 'Working' }) };
        const stop = f.el('button', { text: 'Stop…' });
        actions.append(stop);
        assert.equal(f.replies.cardRef(record), null, 'nothing to quote before the card is known as a root work card');
        f.replies.syncCard(record, true);
        assert.deepEqual(actions.children.map((child) => child.textContent), ['Stop…'], 'the action row keeps only commands');
        assert.equal(root.querySelector('.btn'), null, 'no Reply capsule anywhere on the card');
        assert.equal(root.dataset.rowAddress, 't-9'.replace(/^/, 'task:'), 'the card is a quote\'s jump target');
        f.replies.start(f.replies.cardRef(record));
        assert.deepEqual(f.replies.frameField(), { reply_to: 'task:t-9' });
        const staged = JSON.parse(f.storage.get('ouro_chat_reply'));
        assert.equal(staged.kind, 'work_card');
        assert.equal(staged.author_label, 'Audit');
        assert.equal(staged.snippet, 'Reading the third project');
        f.replies.syncCard(record, false);
        assert.equal(f.replies.cardRef(record), null, 'not a root work card any more: no Reply');
        assert.equal(root.dataset.rowAddress, 'task:t-9', 'a quote of it still finds it');
    } finally { f.close(); }
});

test('the registered message actions stand between Reply and Copy on the rows they apply to, until the room goes', () => {
    const actions = createMessageActions();
    actions.register({ key: 'read_aloud', label: 'Read aloud', icon: 'volume', applies: (row) => row.role === 'assistant',
        run() {} });
    const f = fixture({ actions });
    try {
        const answer = f.bubble('assistant', 'The audit found three stale projects.');
        answer.append(f.el('button', { className: 'chat-message-copy' }));
        const mine = f.bubble('user', 'please check the logs');
        mine.append(f.el('button', { className: 'chat-message-copy' }));
        f.feed.append(answer, mine);
        f.replies.decorate(answer, { row_address: 'row:1@b#2', role: 'assistant' });
        f.replies.decorate(mine, owner('row:1@a#1'));
        const row = (node) => node.children.find((child) => child.classList.contains('chat-message-actions'))
            .children.map((child) => child.className.split(' ')[0]);
        assert.deepEqual(row(answer), ['chat-message-reply', 'chat-message-action', 'chat-message-copy'],
            'Copy keeps the corner; the registered action stands beside Reply');
        assert.deepEqual(row(mine), ['chat-message-reply', 'chat-message-copy'], 'an action draws only where it applies');
        const off = actions.register({ key: 'save', label: 'Save', applies: () => true, run() {} });
        assert.deepEqual(row(mine), ['chat-message-reply', 'chat-message-action', 'chat-message-copy'],
            'a later registration reaches the rows already drawn');
        off();
        f.replies.destroy();
        actions.register({ key: 'late', label: 'Late', applies: () => true, run() {} });
        assert.deepEqual(row(mine), ['chat-message-reply', 'chat-message-copy'], 'a destroyed room draws nothing more');
    } finally { f.restore(); }
});


test('a quote to an earlier work reply lands visibly in the feed reading band, not just focused behind its clip', async () => {
    const f = fixture();
    try {
        const target = f.bubble('user', 'Earlier task reply');
        target.dataset.rowAddress = 'row:1@then#123456789abc';
        // The work reply is the feed's own chronological bubble (owner 23.11), never a card's Details copy.
        f.feed.append(target);
        Object.assign(f.feed, { clientHeight: 400, scrollHeight: 1200, scrollTop: 0 });
        f.feed.getBoundingClientRect = () => ({ top: 0, bottom: 400 });
        target.getBoundingClientRect = () => ({ top: 520 - f.feed.scrollTop, bottom: 540 - f.feed.scrollTop });
        const result = await f.replies.reveal(target.dataset.rowAddress);
        assert.equal(result.target, target);
        assert.equal(f.doc.activeElement, target);
        const line = target.getBoundingClientRect();
        assert.ok(line.top >= 72 && line.bottom <= 400,
            'the target is visible in the room reading band, below the header and above the composer');
    } finally { f.close(); }
});

test('System voice survives from the exact addressed timeline line to its staged and saved quote', () => {
    const f = fixture();
    try {
        const address = 'row:1@then#123456789abc';
        const systemRow = f.el('div', { className: 'chat-live-line', dataset: { workReplyRow: '1', rowAddress: address } },
            f.el('span', { className: 'chat-live-line-title', text: 'Paused by the host' }));
        f.feed.append(systemRow);
        f.replies.start(f.replies.bubbleRef(systemRow));
        assert.equal(f.strip().dataset.author, 'system');
        assert.equal(f.strip().querySelector('.chat-quote-author').textContent, 'System');
        const reloaded = f.bubble('user', 'Please leave it paused');
        f.replies.decorate(reloaded, { reply_to: { address, kind: 'system', role: 'system', snippet: 'Paused by the host' } });
        assert.equal(f.quote(reloaded).dataset.author, 'system');
        assert.equal(f.quote(reloaded).querySelector('.chat-quote-author').textContent, 'System');
    } finally { f.close(); }
});

test('child narration and expanded final report lines quote their own words and address, never the role label', () => {
    const f = fixture();
    try {
        const child = f.el('div', { className: 'chat-live-card subagent', dataset: { taskId: 'child-7' } });
        const narration = f.el('div', { className: 'chat-live-line', dataset: {
            workReplyRow: 'assistant', rowAddress: 'row:1@2026-10-16T09:41:00Z#narr00000001',
        } },
            f.el('span', { className: 'chat-live-line-title', text: 'worker' }),
            f.el('div', { className: 'chat-live-line-body', text: 'Inspecting the archive index.' }));
        const report = f.el('div', { className: 'chat-live-line expandable', dataset: {
            workReplyRow: 'assistant', rowAddress: 'row:1@2026-10-16T09:42:00Z#rprt00000002',
            rowWords: 'All three shards matched the checksum.',
        } },
            f.el('span', { className: 'chat-live-line-title', text: 'worker' }),
            f.el('div', { className: 'chat-live-line-body', text: 'Report to parent\nAll three shards matched the checksum.' }));
        child.append(narration, report);
        f.feed.append(child);

        const narrRef = f.replies.bubbleRef(narration);
        assert.deepEqual(
            [narrRef.address, narrRef.role, narrRef.kind, narrRef.author_label, narrRef.snippet, narrRef.ts, narrRef.task_id],
            ['row:1@2026-10-16T09:41:00Z#narr00000001', 'assistant', 'message', 'Ouroboros', 'Inspecting the archive index.', '2026-10-16T09:41:00Z', 'child-7'],
        );
        f.replies.start(narrRef);
        assert.deepEqual(f.replies.frameField(), { reply_to: 'row:1@2026-10-16T09:41:00Z#narr00000001' });
        assert.equal(f.strip().querySelector('.chat-quote-author').textContent, 'Ouroboros');
        assert.equal(f.strip().querySelector('.chat-quote-text').textContent, 'Inspecting the archive index.');

        const reportRef = f.replies.bubbleRef(report);
        assert.deepEqual(
            [reportRef.address, reportRef.snippet, reportRef.ts],
            ['row:1@2026-10-16T09:42:00Z#rprt00000002', 'All three shards matched the checksum.', '2026-10-16T09:42:00Z'],
        );
        f.replies.start(reportRef);
        assert.equal(f.strip().querySelector('.chat-quote-text').textContent, 'All three shards matched the checksum.');
    } finally { f.close(); }
});

test('syncDerivedQuestion adds, updates, no-ops and removes only derived question quotes while keeping focus and explicit Reply', () => {
    const f = fixture();
    try {
        const question = f.bubble('assistant', 'Which format should I ship?', { address: 'row:1@q#1' });
        f.feed.append(question);
        f.replies.decorate(question, { row_address: 'row:1@q#1', role: 'assistant' });

        const answer = f.bubble('user', 'PDF, please');
        f.feed.append(answer);
        f.replies.decorate(answer, { row_address: 'row:1@a#1', role: 'user' });
        assert.equal(f.quote(answer), null);

        const qRef = {
            address: 'row:1@q#1', role: 'assistant', kind: 'question', author_label: 'Ouroboros',
            snippet: 'Which format should I ship?', room_label: 'Main', ts: '2026-11-23T10:00:00Z',
            derived_question: true,
        };
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user', reply_to: qRef }, { authoritative: true }), true);
        const firstQuote = f.quote(answer);
        assert.ok(firstQuote);
        assert.equal(firstQuote.dataset.derivedQuestion, '1');
        assert.equal(firstQuote.hidden, false);
        assert.equal(f.replies.syncDerivedQuotes(answer), false, 'user derived question quote is not auto-hidden under the question');
        assert.equal(firstQuote.hidden, false);

        // Unchanged history refresh is a strict no-op: exact same DOM node and focus remain intact.
        const firstButton = firstQuote.querySelector('.chat-quote-main');
        firstButton.focus();
        assert.equal(f.doc.activeElement, firstButton);
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user', reply_to: { ...qRef } }, { authoritative: true }), false);
        assert.equal(f.quote(answer), firstQuote);
        assert.equal(f.doc.activeElement, firstButton);

        // Updated derived quote (e.g. room renamed) replaces the node and keeps focus on the quote button.
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user', reply_to: { ...qRef, room_label: 'Renamed' } },
            { authoritative: true }), true);
        const updatedQuote = f.quote(answer);
        assert.notEqual(updatedQuote, firstQuote);
        assert.match(updatedQuote.querySelector('.chat-quote-room').textContent, /Renamed/);
        assert.equal(f.doc.activeElement, updatedQuote.querySelector('.chat-quote-main'));

        // Incomplete live frames (non-authoritative or shared also_rooms) never remove an already-shown derived quote.
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user' }, { authoritative: false }), false);
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user', also_rooms: [2] }, { authoritative: true }), false);
        assert.equal(f.quote(answer), updatedQuote);

        // Authoritative history without reply_to removes the stale derived quote and moves focus to the bubble.
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user' }, { authoritative: true }), true);
        assert.equal(f.quote(answer), null);
        assert.equal(f.doc.activeElement, answer);
        assert.equal(answer.tabIndex, -1);

        // When focus is elsewhere (e.g. composer), adding/updating/removing never steals focus.
        f.input.focus();
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user', reply_to: qRef }, { authoritative: true }), true);
        assert.equal(f.doc.activeElement, f.input);
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user' }, { authoritative: true }), true);
        assert.equal(f.doc.activeElement, f.input);

        // An explicit owner Reply (without derived_question) is never overwritten or removed.
        const explicit = f.bubble('user', 'Explicit answer');
        f.feed.append(explicit);
        f.replies.decorate(explicit, {
            row_address: 'row:1@a#2', role: 'user',
            reply_to: { address: 'row:1@q#1', role: 'assistant', snippet: 'Owner picked this' },
        });
        const explicitQuote = f.quote(explicit);
        assert.ok(explicitQuote);
        assert.equal(explicitQuote.dataset.derivedQuestion, undefined);
        assert.equal(f.replies.syncDerivedQuestion(explicit, { role: 'user' }, { authoritative: true }), false);
        assert.equal(f.replies.syncDerivedQuestion(explicit, { role: 'user', reply_to: qRef }, { authoritative: true }), false);
        assert.equal(f.quote(explicit), explicitQuote);
    } finally { f.close(); }
});

test('retranslateDerivedQuotes follows a dictionary change in place: chrome and aria re-render, data and focus stay', () => {
    const EN = { language: '', english: true, revision: 0, entries: {} };
    const f = fixture();
    // The DOM double has no <html> element; i18n's payload application writes the document's language onto it.
    Object.defineProperty(f.doc, 'documentElement', { value: f.el('html'), configurable: true });
    // and no tree walker: the translator's restore pass needs one (it finds no text nodes here, which is the point).
    f.doc.createTreeWalker = (root) => { const texts = []; (function collect(node) {
        for (const child of node.childNodes || []) { if (child.nodeType === 3) texts.push(child); else collect(child); }
    })(root); let at = 0; return { nextNode: () => (at < texts.length ? texts[at++] : null) }; };
    try {
        applyPayload(EN);
        const answer = f.bubble('user', 'PDF, please');
        f.feed.append(answer);
        f.replies.decorate(answer, { row_address: 'row:1@a#1', role: 'user' });
        const qRef = {
            address: 'row:1@q#1', role: 'assistant', kind: 'question', author_label: 'Ouroboros',
            snippet: 'Which format should I ship?', room_label_kind: 'project', ts: '2026-11-23T10:00:00Z',
            derived_question: true,
        };
        assert.equal(f.replies.syncDerivedQuestion(answer, { role: 'user', reply_to: qRef }, { authoritative: true }), true);
        const mounted = f.quote(answer);
        assert.ok(mounted);
        assert.equal(mounted.querySelector('.chat-quote-room').textContent, '\u00b7 Project');
        assert.match(mounted.querySelector('.chat-quote-main').getAttribute('aria-label'), /^Reply to /);

        // The dictionary changes without any reconnect: the mounted quote re-renders its chrome and aria in place.
        applyPayload({ language: 'ru', english: false, revision: 2, entries: {
            ['code:chat.reply.reply_to']: { text: '\u041e\u0442\u0432\u0435\u0442 \u043d\u0430' },
            ['code:project.reference.generic']: { text: '\u041f\u0440\u043e\u0435\u043a\u0442' },
        } });
        assert.equal(f.replies.retranslateDerivedQuotes(), true);
        const translated = f.quote(answer);
        assert.notEqual(translated, mounted);
        assert.equal(translated.querySelector('.chat-quote-room').textContent, '\u00b7 \u041f\u0440\u043e\u0435\u043a\u0442');
        assert.match(translated.querySelector('.chat-quote-main').getAttribute('aria-label'), /^\u041e\u0442\u0432\u0435\u0442 \u043d\u0430 /);
        // The host's data never translates: the author's label and the quoted words stay verbatim.
        assert.equal(translated.querySelector('.chat-quote-author').textContent, 'Ouroboros');
        assert.equal(translated.querySelector('.chat-quote-text').textContent, 'Which format should I ship?');

        // Focus inside the quote survives the re-render onto the new jump control.
        translated.querySelector('.chat-quote-main').focus();
        assert.equal(f.doc.activeElement, translated.querySelector('.chat-quote-main'));
        applyPayload({ language: 'ru', english: false, revision: 3, entries: {
            ['code:chat.reply.reply_to']: { text: '\u041e\u0442\u0432\u0435\u0442 \u043d\u0430' },
            ['code:project.reference.generic']: { text: '\u041f\u0440\u043e\u0435\u043a\u0442' },
        } });
        assert.equal(f.replies.retranslateDerivedQuotes(), true);
        const refocused = f.quote(answer);
        assert.notEqual(refocused, translated);
        assert.equal(f.doc.activeElement, refocused.querySelector('.chat-quote-main'));

        // An unchanged dictionary writes nothing: the same node and focus remain.
        assert.equal(f.replies.retranslateDerivedQuotes(), false);
        assert.equal(f.quote(answer), refocused);
        assert.equal(f.doc.activeElement, refocused.querySelector('.chat-quote-main'));
    } finally {
        applyPayload(EN);
        f.close();
    }
});

test('adopt mounts a derived_question quote on an owner echo without disturbing explicit replies or removing derived quotes on sparse live frames', () => {
    const f = fixture();
    try {
        const echo = f.bubble('user', 'Ship PDF');
        echo.dataset.clientMessageId = 'cm-derived';
        f.feed.append(echo);
        f.replies.decorate(echo, null);
        const derivedRef = {
            address: 'row:1@q#9', role: 'assistant', kind: 'question', author_label: 'Ouroboros',
            snippet: 'Which format?', ts: '2026-11-23T10:00:00Z', derived_question: true,
        };
        assert.equal(f.replies.adopt({
            role: 'user', client_message_id: 'cm-derived', row_address: 'row:1@a#9', reply_to: derivedRef,
        }), true);
        const mounted = f.quote(echo);
        assert.ok(mounted);
        assert.equal(mounted.dataset.derivedQuestion, '1');

        // A verified retained copy that already shows a derived quote does not lose it if adopt receives a sparse row.
        const copy = f.bubble('user', 'Retained late answer');
        copy.dataset.originProjected = '1';
        f.feed.append(copy);
        f.replies.decorate(copy, { role: 'user', origin_projected: true, reply_to: derivedRef });
        const copyQuote = f.quote(copy);
        assert.equal(copyQuote.dataset.derivedQuestion, '1');
        assert.equal(f.replies.adopt({ role: 'user', row_address: 'row:1@a#10' }, copy), true);
        assert.equal(f.quote(copy), copyQuote, 'adopt alone never removes a derived quote; only authoritative history sync does');
    } finally { f.close(); }
});
