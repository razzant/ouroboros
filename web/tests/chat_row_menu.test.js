// The row menu (chat_row_menu.js): one menu per bubble or root work card, reached by a right-click, Shift+F10 or the
// ContextMenu key on the focused row, or a long press on touch; its items are the row's own doors; Esc and an outside
// press close it and focus returns to the row. Native hit testing and real gestures are in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatRowMenu } from '../modules/chat_row_menu.js';
import { buildTimelineItemHtml } from '../modules/chat_activity.js';
import { createChatMedia } from '../modules/chat_media.js';
import { createChatReplies } from '../modules/chat_reply.js';
import { createTextBubbles } from '../modules/chat_text_bubble.js';
import { createMessageActions } from '../modules/message_actions.js';
import { cardControls, emptyWorkState } from '../modules/work_status.js';
import { installDomDouble } from './dom_double.js';

const menuRows = (menu) => menu?.querySelector(':scope > .ui-popup-body')?.children || [];

function fixture({ actions = undefined, selectText, phone = false } = {}) {
    const env = installDomDouble();
    const { doc, el } = env;
    const feed = el('div', { className: 'chat-messages' });
    doc.body.append(feed);
    if (phone) {
        window.ouroLayout = { layout: 'stack' };
        window.matchMedia = () => ({ matches: true });
        feed.rect = { left: 0, top: 50, right: 390, bottom: 670, width: 390, height: 620 };
    }
    const started = [];
    const replies = {
        bubbleRef: (row) => (row.dataset.rowAddress ? { address: row.dataset.rowAddress } : null),
        cardRef: (record) => (record.replyable ? { address: `task:${record.groupId}` } : null),
        start: (ref) => started.push(ref.address),
        canSelect: row => Boolean(selectText && row.dataset.rowAddress), selectText,
    };
    const copied = [];
    const copy = el('button', { className: 'chat-message-copy' });
    copy.addEventListener('click', () => copied.push('bubble'));
    const link = el('a', { attrs: { href: 'https://example.com' }, text: 'a link' });
    const words = el('div', { className: 'message' }, link);
    const reference = el('button', { className: 'chat-quiz-project', attrs: { 'data-intent': 'open-project' } });
    const opened = [];
    reference.addEventListener('click', () => opened.push('project'));
    const bubble = el('div', { className: 'chat-bubble assistant', dataset: { rowAddress: 'row:1@t#a' } },
        el('div', { className: 'sender', text: 'Ouroboros' }), words,
        el('div', { className: 'system-message-actions' }, reference),
        el('div', { className: 'chat-message-actions' }, copy));
    const plain = el('div', { className: 'chat-bubble user' }, el('div', { className: 'message', text: 'unsaved echo' }));
    feed.append(bubble, plain);
    const activity = { phase: 'working' };
    const works = emptyWorkState();
    const cards = new Map();
    const episode = { token: {} };
    // chat.js's own wiring: the card's Stop… items come from its work, else from the card's own task controls.
    const menu = createChatRowMenu({ feed, replies, cardOf: (id) => cards.get(id),
        readControls: (record) => ({ ...cardControls(works, record.groupId), activity, legacy: true,
            episode: episode.token, resume: Boolean(record.resumeRunBtn) }), actions });
    const openMenu = () => doc.body.children.find((node) => node.classList.contains('chat-row-menu')) || null;
    const items = () => menuRows(openMenu()).map((node) => node.dataset.rowAction);
    const touch = (node, type, extra = {}) => node.emit(type, { pointerType: 'touch', isPrimary: true, pointerId: 1,
        clientX: 30, clientY: 60, ...extra });
    return { ...env, feed, cards, bubble, plain, copy, link, words, menu, started, copied, opened, activity, works, episode, openMenu, items, touch,
        close() { menu.destroy(); env.restore(); } };
}

function workCard(f, { resume = false, stop = true, cancelPending = false } = {}) {
    const { el } = f;
    const title = el('div', { className: 'chat-live-title', text: 'Audit the projects' });
    const line = el('div', { className: 'chat-live-activity', text: 'Checking the third project' });
    const summary = el('div', { className: 'chat-live-summary', attrs: { role: 'button', tabindex: '0' } }, title, line);
    const actions = el('div', { className: 'chat-live-actions' });
    const card = el('div', { className: 'chat-live-card', dataset: { taskId: 't-1', portal: '1' } }, summary, actions);
    const pressed = [];
    const button = (label, key) => {
        const node = el('button', { className: 'btn btn-sm btn-default', text: label });
        node.addEventListener('click', () => pressed.push(key));
        actions.append(node);
        return node;
    };
    // A work card draws no Reply of its own (owner 17.10, Q9 B): the menu's Reply calls the reply path directly.
    const record = { groupId: 't-1', root: card, titleEl: title, activityEl: line, replyable: true,
        resumeRunBtn: resume ? button('Resume', 'resume') : null,
        cancelRunBtn: stop ? button('Stop…', 'stop') : null, cancelPendingPolicy: cancelPending ? 'finalize' : '' };
    const footer = el('button', { className: 'chat-live-bound-pointer', attrs: { 'data-intent': 'open-project' } });
    footer.addEventListener('click', () => pressed.push('project'));
    card.append(footer);
    const child = el('div', { className: 'chat-live-card subagent', dataset: { taskId: 'child' } },
        el('button', { className: 'chat-quiz-project', attrs: { 'data-intent': 'open-project' } }));
    card.insertBefore(child, footer);
    f.feed.append(card);
    f.cards.set('t-1', record);
    return { card, record, pressed, summary, child };
}

test('a right-click on a bubble opens its menu at the pointer: Reply, Copy, Open project, each the row\'s own door', () => {
    const f = fixture();
    try {
        const event = f.words.emit('contextmenu', { clientX: 40, clientY: 50 });
        assert.equal(event.defaultPrevented, true, 'the browser menu yields to the row menu');
        assert.deepEqual(f.items(), ['reply', 'copy', 'project']);
        const menu = f.openMenu();
        assert.equal(menu.getAttribute('role'), 'menu');
        assert.equal(menu.getAttribute('aria-label'), 'Message actions');
        assert.equal(menuRows(menu).every((item) => item.getAttribute('role') === 'menuitem'), true);
        assert.equal(menu.css.get('--ui-popup-left'), '40px', 'placed at the pointer');
        assert.equal(f.doc.activeElement, menuRows(menu)[0], 'keyboard focus enters the menu');
        menuRows(menu)[0].click();
        assert.deepEqual(f.started, ['row:1@t#a'], 'Reply quotes this row');
        assert.equal(f.openMenu(), null, 'an item closes the menu');
        f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
        menuRows(f.openMenu())[1].click();
        assert.deepEqual(f.copied, ['bubble'], 'Copy is the bubble\'s own Copy');
        f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
        menuRows(f.openMenu())[2].click();
        assert.deepEqual(f.opened, ['project'], 'Open project presses the row\'s Project reference');
    } finally { f.close(); }
});

// Spoken lines as a room draws and revises them: the real text bubble, its Copy binding and the real reply path.
function spokenRoom(t, { actions = createMessageActions() } = {}) {
    const env = installDomDouble();
    const { doc, el } = env;
    doc.parseHtml = true;
    const prior = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    const written = [];
    Object.defineProperty(globalThis, 'navigator', { configurable: true,
        value: { clipboard: { writeText: async (text) => { written.push(text); } } } });
    const feed = el('div', { className: 'chat-messages' });
    const input = el('textarea', { attrs: { id: 'chat-input' } });
    doc.body.append(feed, el('div', {}, el('div', { className: 'chat-text-row' }, input)));
    const replies = createChatReplies({ chatId: 1, feed, input, actions: createMessageActions(),
        reading: () => ({ cancel() {}, scroll() {}, remember() {}, stick: true }) });
    const media = createChatMedia({ replies });
    const draw = createTextBubbles({ chatMedia: media, replies, feed });
    const menu = createChatRowMenu({ feed, replies, cardOf: () => null, readControls: () => ({}),
        actions });
    const frame = (item, version) => ({
        meta: { source: 'live_call', projectId: '' },
        opts: { reply: { role: 'assistant', source: 'live_call', spoken: { call_id: 'call-1', item_id: item },
            row_address: `row:1@2026-11-27T10:00:00Z#${item}-v${version}` } },
    });
    const say = (item, words) => {
        const { bubble } = draw(words, 'assistant', frame(item, 1));
        feed.append(bubble);
        return bubble;
    };
    const revise = (bubble, item, words) => assert.equal(draw.revise(bubble, words, 'assistant', frame(item, 2)), true);
    const openOn = (bubble) => {
        bubble.querySelector('.message').emit('contextmenu', { clientX: 1, clientY: 1 });
        const open = doc.body.children.find((node) => node.classList.contains('chat-row-menu'));
        return (kind) => menuRows(open).find((node) => node.dataset.rowAction === kind);
    };
    t.after(async () => {
        await new Promise((resolve) => setImmediate(resolve));  // Copy's own mark is timed after the write settles
        menu.destroy(); replies.destroy(); media.destroy();
        if (prior) Object.defineProperty(globalThis, 'navigator', prior); else delete globalThis.navigator;
        env.restore();
    });
    return { replies, written, say, revise, openOn };
}

test('Copy in a menu opened before a spoken line was revised copies the words the same row shows at the press', (t) => {
    const room = spokenRoom(t);
    room.say('other', 'A neighbouring line.');
    const line = room.say('mine', 'We meet at ten.');
    const item = room.openOn(line);
    const opened = line.querySelector('.chat-message-copy');
    room.revise(line, 'mine', 'We meet on Friday at ten.');
    assert.notEqual(line.querySelector('.chat-message-copy'), opened, 'the revision gave the row a new Copy');
    item('copy').click();
    assert.deepEqual(room.written, ['We meet on Friday at ten.'], 'one write: this row\'s present words');
});

test('Reply in a menu opened before a spoken line was revised quotes the same row\'s present words and address', (t) => {
    const room = spokenRoom(t);
    room.say('other', 'A neighbouring line.');
    const line = room.say('mine', 'We meet at ten.');
    const item = room.openOn(line);
    room.revise(line, 'mine', 'We meet on Friday at ten.');
    item('reply').click();
    const staged = room.replies.echoRow()?.reply_to;
    assert.deepEqual([staged?.address, staged?.snippet],
        ['row:1@2026-11-27T10:00:00Z#mine-v2', 'We meet on Friday at ten.']);
});

test('a registered action in a menu opened before a spoken line was revised acts on the row as it is at the press', (t) => {
    const actions = createMessageActions();
    const ran = [];
    actions.register({ key: 'share', label: 'Share', applies: (row) => row.role === 'assistant',
        run: (row) => ran.push([row.address, row.text, row.answerWords]) });
    const room = spokenRoom(t, { actions });
    room.say('other', 'A neighbouring line.');
    const line = room.say('mine', 'We meet at ten.');
    const item = room.openOn(line);
    room.revise(line, 'mine', 'We meet on Friday at ten.');
    item('share').click();
    assert.deepEqual(ran, [['row:1@2026-11-27T10:00:00Z#mine-v2', 'We meet on Friday at ten.',
        'We meet on Friday at ten.']], 'the action reads the present words and address, not those of the opening');
});

test('a link, a field or selected words keep the browser menu; a row with nothing to offer opens none', () => {
    const f = fixture();
    try {
        assert.equal(f.link.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, undefined);
        assert.equal(f.openMenu(), null);
        f.doc.selection = { isCollapsed: false, anchorNode: f.words };
        assert.equal(f.words.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, undefined, 'selected text stays copyable');
        f.doc.selection = { isCollapsed: true, anchorNode: f.words };
        f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.ok(f.openMenu(), 'a collapsed caret is no selection');
        menuRows(f.openMenu())[0].click();
        // A right button selects the word under it (macOS) before contextmenu: that selection is the gesture's own.
        f.doc.selection = { isCollapsed: true, anchorNode: f.words };
        f.words.emit('pointerdown', { button: 2, pointerType: 'mouse', isPrimary: true, pointerId: 9 });
        let cleared = false;
        f.doc.selection = { isCollapsed: false, anchorNode: f.words, removeAllRanges() { cleared = true; } };
        assert.equal(f.words.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, true);
        assert.ok(f.openMenu(), 'the row menu opens over the word the right button selected');
        assert.equal(cleared, true, 'and that gesture-made selection goes');
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        // The owner's own selection, there before the right button went down, keeps the browser's menu.
        f.words.emit('pointerdown', { button: 2, pointerType: 'mouse', isPrimary: true, pointerId: 10 });
        assert.equal(f.words.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, undefined);
        assert.equal(f.openMenu(), null);
        f.doc.selection = null;
        const event = f.plain.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.equal(event.defaultPrevented, undefined, 'an unsaved echo has no address and no Copy: nothing to offer');
        assert.equal(f.openMenu(), null);
    } finally { f.close(); }
});

test('Shift+F10 and the ContextMenu key open the focused row\'s menu; Esc closes it and focus returns', () => {
    const f = fixture();
    try {
        f.copy.focus();
        const key = f.copy.emit('keydown', { key: 'F10', shiftKey: true });
        assert.equal(key.defaultPrevented, true);
        const menu = f.openMenu();
        assert.ok(menu);
        assert.equal(f.doc.activeElement, menuRows(menu)[0]);
        menu.emit('keydown', { key: 'Escape', target: menuRows(menu)[0] });
        assert.equal(f.openMenu(), null);
        assert.equal(f.doc.activeElement, f.copy, 'focus returns to where it was in the row');
        f.copy.emit('keydown', { key: 'ContextMenu' });
        assert.ok(f.openMenu(), 'the ContextMenu key too');
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        assert.equal(f.copy.emit('keydown', { key: 'F10' }).defaultPrevented, undefined, 'F10 alone is not the menu key');
        assert.equal(f.openMenu(), null);
    } finally { f.close(); }
});

test('a pointer-opened menu returns focus to the row itself on Esc; an outside press closes it', () => {
    const f = fixture();
    try {
        f.words.emit('contextmenu', { clientX: 5, clientY: 5 });
        assert.equal(f.bubble.getAttribute('tabindex'), '-1', 'the bubble can hold focus to be returned to');
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        assert.equal(f.doc.activeElement, f.bubble);
        f.words.emit('contextmenu', { clientX: 5, clientY: 5 });
        f.feed.parent.emit('pointerdown', { target: f.doc.body });
        assert.equal(f.openMenu(), null, 'an outside press closes it');
    } finally { f.close(); }
});

test('a long touch press opens the menu where the finger is; the tap that ends it presses nothing', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 1, clientX: 30, clientY: 60 });
        t.mock.timers.tick(449);
        assert.equal(f.openMenu(), null, 'not before the press is long');
        t.mock.timers.tick(1);
        const menu = f.openMenu();
        assert.ok(menu, 'a long press opens it');
        assert.equal(menu.css.get('--ui-popup-top'), '60px');
        assert.equal(f.words.emit('contextmenu', { clientX: 30, clientY: 60 }).defaultPrevented, true,
            'the contextmenu Android also fires for a long press is answered already');
        f.words.emit('pointerup', { pointerType: 'touch', pointerId: 1 });
        const tap = menuRows(menu)[0].emit('click', { target: menuRows(menu)[0] });
        assert.equal(tap.defaultPrevented, true, 'the lifting finger does not press the item under it');
        assert.deepEqual(f.started, []);
        t.mock.timers.tick(1000);
        menuRows(menu)[0].click();
        assert.deepEqual(f.started, ['row:1@t#a'], 'a deliberate tap afterwards works');
    } finally { f.close(); }
});

test('Select text releases the touched body to native handles without staging a reply', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const calls = [];
    const f = fixture({ selectText(row, point) { calls.push(point); row.dataset.selectText = '1'; } });
    const hold = () => f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 1, clientX: 20, clientY: 40 });
    try {
        hold();
        t.mock.timers.tick(500);
        const item = menuRows(f.openMenu()).find(button => button.textContent === 'Select text');
        f.words.emit('pointerup', { pointerType: 'touch', pointerId: 1 });
        item.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 2 });
        item.click();
        assert.deepEqual(calls, [{ x: 20, y: 40 }]);
        hold();
        t.mock.timers.tick(800);
        assert.equal(f.openMenu(), null);
        assert.equal(f.words.emit('contextmenu', { clientX: 20, clientY: 40 }).defaultPrevented, undefined);
        assert.deepEqual(f.started, []);
    } finally { f.close(); }
});

test('feed scroll cancels a pending hold, but not the release guard of an open menu', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const f = fixture();
    try {
        f.touch(f.words, 'pointerdown');
        t.mock.timers.tick(200);
        assert.equal(f.bubble.dataset.rowPress, 'hold');
        f.feed.emit('scroll');
        assert.equal(f.bubble.dataset.rowPress, undefined, 'a waiting row stops sinking');
        t.mock.timers.tick(250);
        assert.equal(f.openMenu(), null, 'a pending press cannot open after a scroll');

        f.touch(f.words, 'pointerup');
        f.touch(f.words, 'pointerdown');
        t.mock.timers.tick(450);
        const menu = f.openMenu();
        assert.ok(menu);
        f.feed.emit('scroll'); // a reflow can scroll the feed beneath the lifted row without closing its menu
        f.touch(f.words, 'pointerup');
        const reply = menuRows(menu)[0];
        const click = reply.emit('click', { isTrusted: true, detail: 1 });
        assert.deepEqual(f.started, [], 'reflow did not turn the original release into Reply');
        assert.equal(click.defaultPrevented, true);
        assert.equal(f.openMenu(), menu);
    } finally { f.close(); }
});

test('an unrelated pointer end does not end a pending or fired hold', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const f = fixture();
    try {
        f.touch(f.words, 'pointerdown');
        f.touch(f.words, 'pointerup', { pointerId: 2 });
        t.mock.timers.tick(450);
        const menu = f.openMenu();
        assert.ok(menu, 'the waiting finger is still down');
        f.touch(f.words, 'pointercancel', { pointerId: 2 });
        f.touch(f.words, 'pointerup');
        assert.equal(menuRows(menu)[0].emit('click', { isTrusted: true, detail: 1 }).defaultPrevented, true,
            'only the original pointer ends the fired hold');
        assert.deepEqual(f.started, []);
    } finally { f.close(); }
});

for (const end of ['pointerup', 'pointercancel']) {
    test(`${end} outside the feed still ends a hold whose captured row was detached`, (t) => {
        t.mock.timers.enable({ apis: ['setTimeout'] });
        const f = fixture();
        try {
            f.touch(f.words, 'pointerdown');
            t.mock.timers.tick(450);
            const reply = menuRows(f.openMenu())[0];
            f.bubble.remove();
            f.touch(reply, end); // the portalled target is in the document, not in the chat feed
            assert.equal(reply.emit('click', { isTrusted: true, detail: 1 }).defaultPrevented, true);
            assert.deepEqual(f.started, []);
        } finally { f.close(); }
    });
}

test('movement outside the feed past the tolerance ends a pending hold after capture changed', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const f = fixture();
    try {
        f.touch(f.words, 'pointerdown');
        t.mock.timers.tick(200);
        assert.equal(f.bubble.dataset.rowPress, 'hold');
        f.touch(f.doc.body, 'pointermove', { clientY: 71 });
        assert.equal(f.bubble.dataset.rowPress, undefined);
        t.mock.timers.tick(250);
        assert.equal(f.openMenu(), null);
    } finally { f.close(); }
});

test('a hold also swallows its compatibility mousedown, but not the next independent mouse press', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const f = fixture();
    try {
        f.touch(f.words, 'pointerdown');
        t.mock.timers.tick(450);
        const [reply, copy] = menuRows(f.openMenu());
        assert.equal(f.doc.activeElement, reply);
        f.touch(f.words, 'pointerup');
        const mouse = { isTrusted: true, detail: 1 };
        const compatDown = copy.emit('mousedown', { ...mouse });
        assert.equal(compatDown.defaultPrevented && compatDown.stopped, true, 'the focus default is suppressed');
        // No compatibility click arrives. A real mouse button down disarms the old tail.
        reply.emit('pointerdown', { pointerType: 'mouse', pointerId: 2 });
        assert.equal(reply.emit('mousedown', { ...mouse }).defaultPrevented, undefined);
        assert.equal(reply.emit('click', { ...mouse }).defaultPrevented, undefined);
        assert.deepEqual(f.started, ['row:1@t#a']);
    } finally { f.close(); }
});

test("Android's contextmenu before the long press's time opens the menu once, and the lift presses nothing", (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 1, clientX: 30, clientY: 60 });
        t.mock.timers.tick(300);
        assert.equal(f.words.emit('contextmenu', { clientX: 30, clientY: 60 }).defaultPrevented, true);
        const menu = f.openMenu();
        assert.ok(menu, 'the early contextmenu opens it');
        assert.equal(f.bubble.dataset.rowPress, 'lift', 'the row lifts as a long press would');
        t.mock.timers.tick(500);
        assert.equal(f.openMenu(), menu, 'the long press time opens no second menu');
        f.words.emit('pointerup', { pointerType: 'touch', pointerId: 1 });
        assert.equal(menuRows(menu)[0].emit('click', { target: menuRows(menu)[0] }).defaultPrevented, true,
            'the lifting finger does not press the item under it');
        assert.deepEqual(f.started, []);
    } finally { f.close(); }
});

test('a tap 250 ms after a long press presses the item it meets, once; the lift\'s own click presses nothing', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    const hold = (id) => {
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: id, clientX: 30, clientY: 60 });
        t.mock.timers.tick(450);
        f.words.emit('pointerup', { pointerType: 'touch', pointerId: id });
        return f.openMenu();
    };
    const tap = (node, id) => {
        node.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: id, clientX: 30, clientY: 60 });
        node.emit('pointerup', { pointerType: 'touch', pointerId: id });
        return node.emit('click', { target: node, detail: 1 });
    };
    try {
        assert.equal(f.doc.captures.get('click')?.size || 0, 0, 'nothing listens for clicks on the document before a press');
        const menu = hold(1);  // this host sends no click for the hold
        t.mock.timers.tick(250);
        assert.equal(tap(menuRows(menu)[0], 2).defaultPrevented, undefined, 'the next finger presses Reply');
        assert.deepEqual(f.started, ['row:1@t#a'], 'once');
        assert.equal(f.openMenu(), null);
        // A host that does send the hold's click: that one presses nothing, the next finger's does.
        const again = hold(3);
        assert.equal(menuRows(again)[1].emit('click', { target: menuRows(again)[1], detail: 1 }).defaultPrevented, true);
        assert.deepEqual(f.copied, []);
        t.mock.timers.tick(250);
        tap(menuRows(again)[1], 4);
        assert.deepEqual(f.copied, ['bubble'], 'Copy, once');
        assert.equal(f.doc.captures.get('click')?.size || 0, 0, 'and nothing waits any more');
    } finally { f.close(); }
});

test('after the browser took the hold, or by the menu\'s keys, the next action acts at once; a right-click opens anew', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    const hold = (id) => {
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: id, clientX: 30, clientY: 60 });
        t.mock.timers.tick(450);
        return f.openMenu();
    };
    try {
        const menu = hold(1);
        f.words.emit('pointercancel', { pointerType: 'touch', pointerId: 1 });
        t.mock.timers.tick(250);
        menuRows(menu)[0].emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 2 });
        assert.equal(menuRows(menu)[0].emit('click', { target: menuRows(menu)[0], detail: 1 }).defaultPrevented, undefined,
            'pointercancel, then a new finger');
        assert.deepEqual(f.started, ['row:1@t#a']);
        const keyed = hold(3);
        f.words.emit('pointerup', { pointerType: 'touch', pointerId: 3 });
        assert.equal(f.doc.activeElement, menuRows(keyed)[0], 'the keyboard focus is in the menu the press opened');
        menuRows(keyed)[1].focus();
        menuRows(keyed)[1].emit('keydown', { key: 'Enter' });
        menuRows(keyed)[1].emit('click', { target: menuRows(keyed)[1], detail: 0 });
        assert.deepEqual(f.copied, ['bubble'], 'Enter presses Copy at once');
        // The contextmenu Android may still fire for a lifted hold opens nothing; a right-click after it does.
        hold(5);
        f.words.emit('pointerup', { pointerType: 'touch', pointerId: 5 });
        f.doc.defaultView.emit('blur');  // the window lost the focus: the menu went
        assert.equal(f.openMenu(), null);
        assert.equal(f.words.emit('contextmenu', { clientX: 30, clientY: 60 }).defaultPrevented, true);
        assert.equal(f.openMenu(), null, 'the same hold is no second opening');
        f.words.emit('pointerdown', { button: 2, pointerType: 'mouse', isPrimary: true, pointerId: 6 });
        assert.equal(f.words.emit('contextmenu', { clientX: 40, clientY: 50 }).defaultPrevented, true);
        assert.ok(f.openMenu(), 'a right-click right after it opens the menu');
    } finally { f.close(); }
});

test('movement, a mouse, a second finger or a lift before the time is no long press', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    try {
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 1, clientX: 0, clientY: 0 });
        f.words.emit('pointermove', { pointerId: 1, clientX: 0, clientY: 11 });
        t.mock.timers.tick(500);
        assert.equal(f.openMenu(), null, 'a scroll gesture moved the finger');
        f.words.emit('pointerdown', { pointerType: 'mouse', isPrimary: true, pointerId: 2, clientX: 0, clientY: 0 });
        t.mock.timers.tick(500);
        assert.equal(f.openMenu(), null, 'a held mouse button is not a long press');
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: false, pointerId: 3, clientX: 0, clientY: 0 });
        t.mock.timers.tick(500);
        assert.equal(f.openMenu(), null);
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 4, clientX: 0, clientY: 0 });
        f.words.emit('pointerup', { pointerId: 4 });
        t.mock.timers.tick(500);
        assert.equal(f.openMenu(), null, 'a short tap');
        f.link.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 5, clientX: 0, clientY: 0 });
        t.mock.timers.tick(500);
        assert.equal(f.openMenu(), null, 'a link keeps its own long press');
    } finally { f.close(); }
});

test('a work card offers Reply, Copy, Open project, Resume, Pause and Stop… exactly when the card does', () => {
    const f = fixture();
    try {
        const { card, record, pressed, summary, child } = workCard(f, { resume: true });
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.equal(f.openMenu().getAttribute('aria-label'), 'Work actions');
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'resume', 'pause', 'stop']);
        assert.equal(menuRows(f.openMenu()).at(-1).textContent, 'Stop…', 'the card\'s own words');
        const press = (kind) => {
            summary.emit('contextmenu', { clientX: 1, clientY: 1 });
            menuRows(f.openMenu()).find((node) => node.dataset.rowAction === kind).click();
        };
        press('reply'); press('project'); press('resume'); press('stop');
        assert.deepEqual(pressed.splice(0), ['project', 'resume', 'stop'], 'each item presses the card\'s own control');
        assert.deepEqual(f.started.splice(0), ['task:t-1'], 'Reply quotes the work by its task address');
        // A child without its own loaded record cannot borrow its parent's Reply or work controls.
        child.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.equal(f.openMenu(), null);
        // What the card does not offer, the menu does not offer.
        record.resumeRunBtn.remove();
        f.activity.phase = 'budget_paused';
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'stop'], 'a paused work: no Pause, Resume only when shown');
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        f.activity.phase = 'working';
        record.cancelPendingPolicy = 'finalize';
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'stop'], 'a pending stop offers only its escalation');
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        record.cancelRunBtn.remove();
        record.replyable = false;
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['copy', 'project'], 'a card that offers no Reply, settled without Stop…');
        assert.equal(card.contains(f.openMenu()), false, 'the menu is portalled outside the card');
    } finally { f.close(); }
});

test('a work card\'s Pause follows its work\'s controls, never the census phase', () => {
    const f = fixture();
    try {
        const { summary } = workCard(f);
        const work = (controls) => f.works.episodes.set('t-1', { work_id: 't-1', confirmed: true,
            episode: { task_id: 't-1', live: true }, controls });
        work([{ action: 'stop', target: 't-1' }]);
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'stop'], 'the census says working; the work offers no Pause');
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        work([{ action: 'pause', target: 't-1' }, { action: 'stop', target: 't-1' }]);
        f.activity.phase = 'budget_paused';
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'pause', 'stop'], 'the work\'s Pause, whatever the census says');
    } finally { f.close(); }
});

test('a row menu opened before control withdrawal cannot stop, pause or resume, but keeps its reading actions', async (t) => {
    const f = fixture();
    const posts = [];
    t.mock.method(globalThis, 'fetch', async (url, init) => {
        if (init?.method === 'POST') posts.push(String(url));
        return { ok: true, json: async () => ({ ok: true, state: 'paused' }) };
    });
    const { card, record, pressed, summary } = workCard(f, { resume: true });
    const open = () => { f.menu.open(card, { anchor: summary }); return f.openMenu(); };
    const item = (menu, kind) => menu.querySelector(`[data-row-action="${kind}"]`);
    const stop = record.cancelRunBtn;
    try {
        summary.focus();
        const pause = item(open(), 'pause');
        f.activity.phase = 'budget_paused';  // Pause withdrawn, Stop still exists.
        pause.click();
        assert.deepEqual(posts, []);
        assert.equal(f.doc.activeElement, summary, 'a withdrawn press returns focus to its own row');

        f.activity.phase = 'working';
        const oldAttempt = item(open(), 'pause');
        f.episode.token = {};
        oldAttempt.click();
        assert.deepEqual(posts, [], 'the same visible button does not authorize a different attempt');
        const openedStop = item(open(), 'stop');
        stop.remove(); record.cancelRunBtn = null;
        assert.doesNotThrow(() => openedStop.click());
        assert.equal(f.doc.activeElement, summary, 'no successor control took focus');

        const openedResume = item(open(), 'resume');
        record.resumeRunBtn.remove(); record.resumeRunBtn = null;
        assert.doesNotThrow(() => openedResume.click());
        assert.deepEqual(pressed, []);
        assert.equal(f.doc.activeElement, summary);
        const reading = open();
        assert.deepEqual(f.items(), ['reply', 'copy', 'project']);
        item(reading, 'reply').click();
        assert.deepEqual(f.started, ['task:t-1']);
        assert.ok(item(open(), 'copy'), 'Copy remains a door on the same row');
        item(f.openMenu(), 'project').click();
        card.append(stop); record.cancelRunBtn = stop;
        item(open(), 'pause').click();
        await new Promise(resolve => setImmediate(resolve));
        assert.deepEqual(posts, ['/api/tasks/t-1/pause'], 'a current Pause uses the shared request once');
        assert.deepEqual(pressed, ['project']);
    } finally { f.close(); }
});

test('destroy closes an open menu and leaves no listener on the feed or the document', () => {
    const f = fixture();
    f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
    assert.ok(f.openMenu());
    f.menu.destroy();
    assert.equal(f.openMenu(), null);
    assert.equal(f.feed.listenerCount(), 0);
    assert.equal(f.doc.captures.get('click')?.size || 0, 0);
    assert.equal(f.doc.captures.get('pointerdown')?.size || 0, 0);
    f.restore();
});

test('destroy disposes both document tracking during a held press and its guard after release', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    for (const release of [false, true]) {
        const f = fixture();
        try {
            f.touch(f.words, 'pointerdown');
            t.mock.timers.tick(release ? 450 : 200);
            if (release) f.touch(f.words, 'pointerup');
            assert.ok(f.doc.captures.get(release ? 'mousedown' : 'pointerup')?.size);
            f.menu.destroy();
            t.mock.timers.tick(500);
            assert.equal(f.openMenu(), null);
            assert.equal(f.bubble.dataset.rowPress, undefined);
            assert.ok([...f.doc.captures.values()].every((listeners) => listeners.size === 0));
        } finally { f.close(); }
    }
});

test('a bubble\'s menu lists the registered actions after its own doors, in their present words; a work card none', () => {
    const actions = createMessageActions();
    const ran = [];
    let speaking = false;
    actions.register({ key: 'read_aloud', label: 'Read aloud', applies: (row) => row.role === 'assistant',
        state: () => (speaking ? { label: 'Stop' } : null), run: (row, ctx) => ran.push([row.address, ctx.anchor]) });
    actions.register({ key: 'mine', label: 'Mine only', applies: (row) => row.role === 'user', run() {} });
    const f = fixture({ actions });
    try {
        f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'read_aloud']);
        assert.equal(menuRows(f.openMenu())[3].textContent, 'Read aloud');
        menuRows(f.openMenu())[3].click();
        assert.deepEqual(ran, [['row:1@t#a', f.bubble]], 'the action runs on this row');
        assert.equal(f.openMenu(), null);
        speaking = true;
        f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.equal(menuRows(f.openMenu())[3].textContent, 'Stop', 'the words follow the action\'s own state');
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        const { summary } = workCard(f);
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.equal(f.items().includes('read_aloud'), false, 'a work card is not a reply row');
    } finally { f.close(); }
});

test('on touch the row sinks while the finger holds, lifts with a haptic while its menu is open, and settles back', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture();
    const played = [];
    globalThis.window.Telegram = { WebApp: { initData: 'q', isVersionAtLeast: () => true,
        HapticFeedback: { impactOccurred: (style) => played.push(style) } } };
    try {
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 1, clientX: 30, clientY: 60 });
        t.mock.timers.tick(149);
        assert.equal(f.bubble.dataset.rowPress, undefined, 'a tap or the start of a scroll never shows it');
        t.mock.timers.tick(1);
        assert.equal(f.bubble.dataset.rowPress, 'hold', 'a still finger: the row sinks');
        t.mock.timers.tick(300);
        assert.ok(f.openMenu());
        assert.equal(f.bubble.dataset.rowPress, 'lift', 'the menu opens over the lifted row');
        assert.deepEqual(played, ['medium'], 'one haptic as the press is recognised');
        f.words.emit('pointerup', { pointerType: 'touch', pointerId: 1 });
        assert.equal(f.bubble.dataset.rowPress, 'lift', 'it stays lifted while the menu is open');
        f.feed.parent.emit('pointerdown', { target: f.doc.body });
        assert.equal(f.openMenu(), null);
        assert.equal(f.bubble.dataset.rowPress, 'release', 'on its way back');
        t.mock.timers.tick(260);
        assert.equal(f.bubble.dataset.rowPress, undefined, 'at rest');
        // A finger that moves (a scroll) puts the sinking row back at once, and nothing opens.
        f.words.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 2, clientX: 30, clientY: 60 });
        t.mock.timers.tick(200);
        assert.equal(f.bubble.dataset.rowPress, 'hold');
        f.words.emit('pointermove', { pointerId: 2, clientX: 30, clientY: 80 });
        assert.equal(f.bubble.dataset.rowPress, undefined);
        t.mock.timers.tick(500);
        assert.equal(f.openMenu(), null);
        // A row with nothing to offer never sinks; a right-click opens the menu with no lift and no haptic.
        f.plain.emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 3, clientX: 30, clientY: 60 });
        t.mock.timers.tick(200);
        assert.equal(f.plain.dataset.rowPress, undefined);
        f.plain.emit('pointerup', { pointerType: 'touch', pointerId: 3 });
        t.mock.timers.tick(1000);
        f.words.emit('contextmenu', { clientX: 30, clientY: 60 });
        assert.ok(f.openMenu());
        assert.equal(f.bubble.dataset.rowPress, undefined);
        assert.deepEqual(played, ['medium']);
    } finally { delete globalThis.window.Telegram; f.close(); }
});


test('a phone swipe and the actual row menu quote the same exact System row or child card, never the enclosing work', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture({ phone: true });
    try {
        const { record, child, summary } = workCard(f);
        const childTitle = f.el('div', { text: 'Read the other room' });
        const systemTitle = f.el('span', { className: 'chat-live-line-title', text: 'Work paused' });
        const systemRow = f.el('div', { className: 'chat-live-line', dataset: {
            workReplyRow: '1', rowAddress: 'row:1@t#system',
        } }, systemTitle);
        child.append(childTitle, systemRow);
        f.cards.set('child', { groupId: 'child', root: child, titleEl: childTitle, replyable: true });

        for (const [target, row, address] of [[systemTitle, systemRow, 'row:1@t#system'],
            [childTitle, child, 'task:child'], [summary, record.root, 'task:t-1']]) {
            assert.equal(f.menu.open(row), true);
            const action = menuRows(f.openMenu()).find(node => node.dataset.rowAction === 'reply');
            action.emit('pointerdown', { pointerType: 'touch', isPrimary: true });
            action.click();
            assert.deepEqual(f.started.splice(0), [address]);
            const fingers = (x) => [{ identifier: 1, clientX: x, clientY: 220 }];
            target.emit('touchstart', { touches: fingers(300) });
            for (const x of [280, 205]) target.emit('touchmove', { touches: fingers(x), cancelable: true });
            assert.equal(row.dataset.rowSwipe, 'drag');
            target.emit('touchend', { touches: [], changedTouches: fingers(205) });
            assert.deepEqual(f.started.splice(0), [address]);
            t.mock.timers.tick(500);
        }
    } finally { f.close(); }
});

test('child narration and final report quote their own addresses; unstamped timeline rows open their card\'s doors, never its Reply', () => {
    const actions = createMessageActions();
    actions.register({ key: 'any_bubble', label: 'Bubble action', applies: () => true, run() {} });
    const f = fixture({ actions, phone: true });
    f.doc.parseHtml = true;
    try {
        const { child } = workCard(f);
        const childTitle = f.el('div', { className: 'chat-live-title', text: 'worker' });
        const timeline = f.el('div', { className: 'chat-live-timeline', dataset: { liveTimeline: '' } });
        child.append(childTitle, timeline);
        f.cards.set('child', { groupId: 'child', root: child, titleEl: childTitle, replyable: true });

        const items = [
            { lineKey: 'narr-short', phase: 'working', headline: 'worker', body: 'Scanning logs.', fullBody: 'Scanning logs.',
                replyAddress: 'row:1@t1#narr-short', replyRole: 'assistant', dedupeKey: 'history:progress:1' },
            { lineKey: 'narr-long', phase: 'working', headline: 'worker', body: 'Long preview', fullBody: 'Long preview with full details',
                replyAddress: 'row:1@t2#narr-long', replyRole: 'assistant', dedupeKey: 'history:progress:2' },
            { lineKey: 'report', phase: 'done', headline: 'worker', body: 'Audit complete.',
                fullBody: '## Report to parent\nAudit complete.\nExtra details.', outputPreview: 'Audit complete.',
                replyAddress: 'row:1@t3#report', replyRole: 'assistant', dedupeKey: 'subagent-result:child' },
            { lineKey: 'tool', phase: 'result', headline: '2 tool calls', body: '', fullBody: 'read_file · grep',
                dedupeKey: 'tools|child' },
            { lineKey: 'lifecycle', phase: 'start', headline: 'worker · Started', body: '', fullBody: '',
                dedupeKey: 'subagent-lifecycle:child' },
        ];
        const record = { groupId: 'child', items, expandedLineKeys: new Set(['report']), timelineEl: timeline };
        timeline.innerHTML = items.map((item) => buildTimelineItemHtml(item, record)).join('');

        const line = (key) => timeline.querySelector(`[data-live-line-key="${key}"]`);
        const narrShort = line('narr-short'), narrLong = line('narr-long'), report = line('report');
        const tool = line('tool'), lifecycle = line('lifecycle');
        const narrToggle = narrLong.querySelector('[data-live-line-toggle]');
        const reportToggle = report.querySelector('[data-live-line-toggle]');
        const toolToggle = tool.querySelector('[data-live-line-toggle]');

        // Tab stops and DOM snippet bounds: 1 stop on short addressable row, toggle-only stop on expandable rows, 0 on unstamped non-expandable.
        assert.equal(narrShort.getAttribute('tabindex'), '0');
        assert.equal(narrShort.dataset.rowWords, undefined);
        assert.equal(narrLong.getAttribute('tabindex'), null, 'expandable row adds no second stop on its wrapper');
        assert.equal(narrToggle.getAttribute('tabindex'), '0');
        assert.equal(narrLong.dataset.rowWords, undefined);
        assert.equal(report.getAttribute('tabindex'), null);
        assert.equal(reportToggle.getAttribute('tabindex'), '0');
        assert.equal(report.dataset.rowWords, 'Audit complete.');
        assert.equal(tool.getAttribute('tabindex'), null);
        assert.equal(toolToggle.getAttribute('tabindex'), '0');
        assert.equal(lifecycle.getAttribute('tabindex'), null);

        // Right-click on narration and report quotes each exact address without leaking bubble-only actions.
        for (const [target, address] of [[narrShort, 'row:1@t1#narr-short'], [report, 'row:1@t3#report']]) {
            assert.equal(target.emit('contextmenu', { clientX: 12, clientY: 16 }).defaultPrevented, true);
            assert.deepEqual(f.items(), ['reply'], 'timeline rows expose Reply without leaking bubble-only actions');
            menuRows(f.openMenu())[0].click();
            assert.deepEqual(f.started.splice(0), [address]);
        }

        // Shift+F10 and ContextMenu work from both a focused expandable toggle and a focused short row, restoring focus on Esc.
        narrToggle.focus();
        assert.equal(narrToggle.emit('keydown', { key: 'F10', shiftKey: true }).defaultPrevented, true);
        assert.deepEqual(f.items(), ['reply']);
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        assert.equal(f.doc.activeElement, narrToggle, 'Esc returns focus to the focused toggle');
        assert.equal(narrToggle.emit('keydown', { key: 'ContextMenu' }).defaultPrevented, true);
        menuRows(f.openMenu())[0].click();
        assert.deepEqual(f.started.splice(0), ['row:1@t2#narr-long']);

        narrShort.focus();
        assert.equal(narrShort.emit('keydown', { key: 'F10', shiftKey: true }).defaultPrevented, true);
        menuRows(f.openMenu())[0].click();
        assert.deepEqual(f.started.splice(0), ['row:1@t1#narr-short']);

        // Unstamped UI-only lines never quote `task:child` on right-click, keyboard, or swipe: the child's own doors
        // answer (its Copy, the only one this record has), and a finger swipe leaves the line alone.
        for (const unstamped of [tool, toolToggle, lifecycle]) {
            assert.equal(unstamped.emit('contextmenu', { clientX: 12, clientY: 16 }).defaultPrevented, true);
            assert.equal(f.openMenu().getAttribute('aria-label'), 'Work actions');
            assert.deepEqual(f.items(), ['copy'], 'the card\'s doors, without the card\'s Reply');
            f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        }
        toolToggle.focus();
        assert.equal(toolToggle.emit('keydown', { key: 'F10', shiftKey: true }).defaultPrevented, true);
        assert.deepEqual(f.items(), ['copy']);
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        assert.equal(f.doc.activeElement, toolToggle, 'Esc returns focus to the disclosure the menu key was pressed on');
        const fingers = (x) => [{ identifier: 1, clientX: x, clientY: 220 }];
        lifecycle.emit('touchstart', { touches: fingers(300) });
        for (const x of [280, 205]) lifecycle.emit('touchmove', { touches: fingers(x), cancelable: true });
        lifecycle.emit('touchend', { touches: [], changedTouches: fingers(205) });
        assert.deepEqual(f.started, [], 'swiping an unstamped line does not silently quote the card');
        assert.equal(lifecycle.dataset.rowSwipe, undefined, 'nor does the line follow the finger');

        // Pressing the child card's own header still quotes `task:child`.
        childTitle.emit('contextmenu', { clientX: 12, clientY: 16 });
        assert.ok(f.items().includes('reply'));
        menuRows(f.openMenu())[0].click();
        assert.deepEqual(f.started.splice(0), ['task:child']);
    } finally { f.close(); }
});

// The timeline of a work card as the product builds it: an unstamped tool-evidence line and lifecycle line (no row
// address, so nothing to quote) around one addressed narration line.
function cardTimeline(f, card, record) {
    f.doc.parseHtml = true;
    const timeline = f.el('div', { className: 'chat-live-timeline', dataset: { liveTimeline: '' } });
    const items = [
        { lineKey: 'tool', phase: 'result', headline: '2 tool calls', body: '', fullBody: 'read_file · grep', dedupeKey: `tools|${record.groupId}` },
        { lineKey: 'narr', phase: 'working', headline: 'worker', body: 'Scanning logs.', fullBody: 'Scanning logs.',
            replyAddress: `row:1@t1#${record.groupId}-narr`, replyRole: 'assistant', dedupeKey: 'history:progress:1' },
        { lineKey: 'life', phase: 'start', headline: 'worker · Started', body: '', fullBody: '', dedupeKey: `subagent-lifecycle:${record.groupId}` },
    ];
    timeline.innerHTML = items.map((item) => buildTimelineItemHtml(item, { groupId: record.groupId, items,
        expandedLineKeys: new Set(), timelineEl: timeline })).join('');
    card.insertBefore(timeline, card.children.at(-1));
    const line = (key) => timeline.querySelector(`[data-live-line-key="${key}"]`);
    return { timeline, tool: line('tool'), narr: line('narr'), life: line('life') };
}

test('a line with no address opens its card\'s doors without the card\'s Reply; the card, not the line, takes the press', (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'] });
    const f = fixture({ phone: true });
    try {
        const { card, record, pressed } = workCard(f, { resume: true });
        const { tool, narr, life } = cardTimeline(f, card, record);
        const open = (node) => node.emit('contextmenu', { clientX: 12, clientY: 16 });
        const choose = (node, kind) => { open(node); menuRows(f.openMenu()).find((item) => item.dataset.rowAction === kind).click(); };

        // Pointer: the card's own doors in the card's own order, Reply alone gone (its title is not this line).
        assert.equal(open(tool).defaultPrevented, true);
        assert.equal(f.openMenu().getAttribute('aria-label'), 'Work actions');
        assert.deepEqual(f.items(), ['copy', 'project', 'resume', 'pause', 'stop']);
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        for (const kind of ['project', 'resume', 'stop']) choose(life, kind);
        assert.deepEqual(pressed.splice(0), ['project', 'resume', 'stop'], 'each item presses the card\'s own control');
        assert.deepEqual(f.started, [], 'no item quotes the card');

        // The same offer the card itself makes: a paused work shows no Pause and the line follows it.
        f.activity.phase = 'budget_paused';
        open(tool);
        assert.deepEqual(f.items(), ['copy', 'project', 'resume', 'stop']);
        f.openMenu().emit('keydown', { key: 'Escape', target: menuRows(f.openMenu())[0] });
        f.activity.phase = 'working';

        // Touch: the card sinks while the finger holds and lifts while its menu is open; the line never does.
        f.touch(life, 'pointerdown');
        t.mock.timers.tick(150);
        assert.equal(card.dataset.rowPress, 'hold');
        assert.equal(life.dataset.rowPress, undefined);
        t.mock.timers.tick(300);
        assert.equal(card.dataset.rowPress, 'lift');
        assert.equal(life.dataset.rowPress, undefined);
        assert.deepEqual(f.items(), ['copy', 'project', 'resume', 'pause', 'stop']);
        f.touch(life, 'pointerup');
        f.feed.parent.emit('pointerdown', { target: f.doc.body });
        assert.equal(card.dataset.rowPress, 'release', 'on its way back');
        t.mock.timers.tick(260);
        assert.equal(card.dataset.rowPress, undefined, 'at rest');

        // A finger that moves on the line (a scroll) puts the sinking card back and opens nothing.
        f.touch(tool, 'pointerdown');
        t.mock.timers.tick(200);
        assert.equal(card.dataset.rowPress, 'hold');
        f.touch(tool, 'pointermove', { clientY: 80 });
        assert.equal(card.dataset.rowPress, undefined);
        t.mock.timers.tick(500);
        assert.equal(f.openMenu(), null);

        // A swipe on the line replies to nothing and moves nothing; on the addressed line beside it, it replies to that line.
        const fingers = (x) => [{ identifier: 1, clientX: x, clientY: 220 }];
        const swipe = (node) => {
            node.emit('touchstart', { touches: fingers(300) });
            for (const x of [280, 205]) node.emit('touchmove', { touches: fingers(x), cancelable: true });
            const dragged = node.dataset.rowSwipe || card.dataset.rowSwipe || '';
            node.emit('touchend', { touches: [], changedTouches: fingers(205) });
            t.mock.timers.tick(500);
            return dragged;
        };
        assert.equal(swipe(tool), '');
        assert.equal(swipe(life), '');
        assert.deepEqual(f.started, [], 'a swipe on an unaddressed line executes nothing of its card');
        assert.equal(swipe(narr), 'drag');
        assert.deepEqual(f.started.splice(0), [`row:1@t1#t-1-narr`], 'the addressed line replies as its own menu does');

        // An addressed line takes its own press and offers its own Reply alone, never its card's doors.
        f.touch(narr, 'pointerdown');
        t.mock.timers.tick(150);
        assert.equal(narr.dataset.rowPress, 'hold');
        t.mock.timers.tick(300);
        assert.equal(narr.dataset.rowPress, 'lift');
        assert.equal(card.dataset.rowPress, undefined);
        assert.deepEqual(f.items(), ['reply']);
    } finally { f.close(); }
});

test('a child\'s unaddressed line offers that child\'s own doors, never its root\'s; a child with no record of its own offers none', () => {
    const f = fixture();
    try {
        const { record, pressed, child } = workCard(f, { resume: true });
        const title = f.el('div', { className: 'chat-live-title', text: 'worker' });
        const stop = f.el('button', { className: 'btn btn-sm btn-default', text: 'Stop…' });
        stop.addEventListener('click', () => pressed.push('child-stop'));
        child.append(title, stop);
        const own = { groupId: 'child', root: child, titleEl: title, replyable: true, cancelRunBtn: stop };
        f.cards.set('child', own);
        const { tool } = cardTimeline(f, child, own);
        const open = () => tool.emit('contextmenu', { clientX: 12, clientY: 16 });

        assert.equal(open().defaultPrevented, true);
        assert.deepEqual(f.items(), ['copy', 'pause', 'stop'], 'the child\'s own: no Resume of the root, no Reply');
        menuRows(f.openMenu()).find((item) => item.dataset.rowAction === 'stop').click();
        assert.deepEqual(pressed.splice(0), ['child-stop'], 'the child\'s Stop, not the root\'s');
        assert.equal(record.cancelRunBtn.isConnected, true);

        f.cards.delete('child');  // no record of its own: it cannot borrow its root's Reply or controls
        assert.equal(open().defaultPrevented, undefined);
        assert.equal(f.openMenu(), null);
    } finally { f.close(); }
});
