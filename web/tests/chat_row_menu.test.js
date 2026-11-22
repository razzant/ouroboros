// The row menu (chat_row_menu.js): one menu per bubble or root work card, reached by a right-click, Shift+F10 or the
// ContextMenu key on the focused row, or a long press on touch; its items are the row's own doors; Esc and an outside
// press close it and focus returns to the row. Native hit testing and real gestures are in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatRowMenu } from '../modules/chat_row_menu.js';
import { createMessageActions } from '../modules/message_actions.js';
import { cardMenuActions, emptyWorkState } from '../modules/work_status.js';
import { installDomDouble } from './dom_double.js';

function fixture({ actions = undefined, selectText } = {}) {
    const env = installDomDouble();
    const { doc, el } = env;
    const feed = el('div', { className: 'chat-messages' });
    doc.body.append(feed);
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
    // chat.js's own wiring: the card's Stop… items come from its work, else from the card's own task controls.
    const menu = createChatRowMenu({ feed, replies, cardOf: (id) => cards.get(id),
        actionsOf: (record) => cardMenuActions(works, record, activity), actions });
    const openMenu = () => doc.body.children.find((node) => node.classList.contains('chat-row-menu')) || null;
    const items = () => (openMenu()?.children || []).map((node) => node.dataset.rowAction);
    const touch = (node, type, extra = {}) => node.emit(type, { pointerType: 'touch', isPrimary: true, pointerId: 1,
        clientX: 30, clientY: 60, ...extra });
    return { ...env, feed, cards, bubble, plain, copy, link, words, menu, started, copied, opened, activity, works, openMenu, items, touch,
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
        assert.equal(menu.children.every((item) => item.getAttribute('role') === 'menuitem'), true);
        assert.equal(menu.css.get('--ui-popup-left'), '40px', 'placed at the pointer');
        assert.equal(f.doc.activeElement, menu.children[0], 'keyboard focus enters the menu');
        menu.children[0].click();
        assert.deepEqual(f.started, ['row:1@t#a'], 'Reply quotes this row');
        assert.equal(f.openMenu(), null, 'an item closes the menu');
        f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
        f.openMenu().children[1].click();
        assert.deepEqual(f.copied, ['bubble'], 'Copy is the bubble\'s own Copy');
        f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
        f.openMenu().children[2].click();
        assert.deepEqual(f.opened, ['project'], 'Open project presses the row\'s Project reference');
    } finally { f.close(); }
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
        f.openMenu().children[0].click();
        // A right button selects the word under it (macOS) before contextmenu: that selection is the gesture's own.
        f.doc.selection = { isCollapsed: true, anchorNode: f.words };
        f.words.emit('pointerdown', { button: 2, pointerType: 'mouse', isPrimary: true, pointerId: 9 });
        let cleared = false;
        f.doc.selection = { isCollapsed: false, anchorNode: f.words, removeAllRanges() { cleared = true; } };
        assert.equal(f.words.emit('contextmenu', { clientX: 1, clientY: 1 }).defaultPrevented, true);
        assert.ok(f.openMenu(), 'the row menu opens over the word the right button selected');
        assert.equal(cleared, true, 'and that gesture-made selection goes');
        f.openMenu().emit('keydown', { key: 'Escape', target: f.openMenu().children[0] });
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
        assert.equal(f.doc.activeElement, menu.children[0]);
        menu.emit('keydown', { key: 'Escape', target: menu.children[0] });
        assert.equal(f.openMenu(), null);
        assert.equal(f.doc.activeElement, f.copy, 'focus returns to where it was in the row');
        f.copy.emit('keydown', { key: 'ContextMenu' });
        assert.ok(f.openMenu(), 'the ContextMenu key too');
        f.openMenu().emit('keydown', { key: 'Escape', target: f.openMenu().children[0] });
        assert.equal(f.copy.emit('keydown', { key: 'F10' }).defaultPrevented, undefined, 'F10 alone is not the menu key');
        assert.equal(f.openMenu(), null);
    } finally { f.close(); }
});

test('a pointer-opened menu returns focus to the row itself on Esc; an outside press closes it', () => {
    const f = fixture();
    try {
        f.words.emit('contextmenu', { clientX: 5, clientY: 5 });
        assert.equal(f.bubble.getAttribute('tabindex'), '-1', 'the bubble can hold focus to be returned to');
        f.openMenu().emit('keydown', { key: 'Escape', target: f.openMenu().children[0] });
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
        const tap = menu.children[0].emit('click', { target: menu.children[0] });
        assert.equal(tap.defaultPrevented, true, 'the lifting finger does not press the item under it');
        assert.deepEqual(f.started, []);
        t.mock.timers.tick(1000);
        menu.children[0].click();
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
        const item = f.openMenu().children.find(button => button.textContent === 'Select text');
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
        const reply = menu.children[0];
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
        assert.equal(menu.children[0].emit('click', { isTrusted: true, detail: 1 }).defaultPrevented, true,
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
            const reply = f.openMenu().children[0];
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
        const [reply, copy] = f.openMenu().children;
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
        assert.equal(menu.children[0].emit('click', { target: menu.children[0] }).defaultPrevented, true,
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
        assert.equal(tap(menu.children[0], 2).defaultPrevented, undefined, 'the next finger presses Reply');
        assert.deepEqual(f.started, ['row:1@t#a'], 'once');
        assert.equal(f.openMenu(), null);
        // A host that does send the hold's click: that one presses nothing, the next finger's does.
        const again = hold(3);
        assert.equal(again.children[1].emit('click', { target: again.children[1], detail: 1 }).defaultPrevented, true);
        assert.deepEqual(f.copied, []);
        t.mock.timers.tick(250);
        tap(again.children[1], 4);
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
        menu.children[0].emit('pointerdown', { pointerType: 'touch', isPrimary: true, pointerId: 2 });
        assert.equal(menu.children[0].emit('click', { target: menu.children[0], detail: 1 }).defaultPrevented, undefined,
            'pointercancel, then a new finger');
        assert.deepEqual(f.started, ['row:1@t#a']);
        const keyed = hold(3);
        f.words.emit('pointerup', { pointerType: 'touch', pointerId: 3 });
        assert.equal(f.doc.activeElement, keyed.children[0], 'the keyboard focus is in the menu the press opened');
        keyed.children[1].focus();
        keyed.children[1].emit('keydown', { key: 'Enter' });
        keyed.children[1].emit('click', { target: keyed.children[1], detail: 0 });
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
        assert.equal(f.openMenu().children.at(-1).textContent, 'Stop…', 'the card\'s own words');
        const press = (kind) => {
            summary.emit('contextmenu', { clientX: 1, clientY: 1 });
            f.openMenu().children.find((node) => node.dataset.rowAction === kind).click();
        };
        press('reply'); press('project'); press('resume'); press('stop');
        assert.deepEqual(pressed.splice(0), ['project', 'resume', 'stop'], 'each item presses the card\'s own control');
        assert.deepEqual(f.started.splice(0), ['task:t-1'], 'Reply quotes the work by its task address');
        // A nested child is part of its root's work: its press opens the root's menu, and its own reference is not the root's.
        child.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'resume', 'pause', 'stop']);
        f.openMenu().emit('keydown', { key: 'Escape', target: f.openMenu().children[0] });
        // What the card does not offer, the menu does not offer.
        record.resumeRunBtn.remove();
        f.activity.phase = 'budget_paused';
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'stop'], 'a paused work: no Pause, Resume only when shown');
        f.openMenu().emit('keydown', { key: 'Escape', target: f.openMenu().children[0] });
        f.activity.phase = 'working';
        record.cancelPendingPolicy = 'finalize';
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'stop'], 'a pending stop offers only its escalation');
        f.openMenu().emit('keydown', { key: 'Escape', target: f.openMenu().children[0] });
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
        f.openMenu().emit('keydown', { key: 'Escape', target: f.openMenu().children[0] });
        work([{ action: 'pause', target: 't-1' }, { action: 'stop', target: 't-1' }]);
        f.activity.phase = 'budget_paused';
        summary.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.deepEqual(f.items(), ['reply', 'copy', 'project', 'pause', 'stop'], 'the work\'s Pause, whatever the census says');
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
        assert.equal(f.openMenu().children[3].textContent, 'Read aloud');
        f.openMenu().children[3].click();
        assert.deepEqual(ran, [['row:1@t#a', f.bubble]], 'the action runs on this row');
        assert.equal(f.openMenu(), null);
        speaking = true;
        f.words.emit('contextmenu', { clientX: 1, clientY: 1 });
        assert.equal(f.openMenu().children[3].textContent, 'Stop', 'the words follow the action\'s own state');
        f.openMenu().emit('keydown', { key: 'Escape', target: f.openMenu().children[0] });
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
