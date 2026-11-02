// The Main and Project composer (composer.js), mounted by every chat room: the same chat-* ids (namespaced in a
// Project panel), the room's draft kept across the room's teardown and cleared by Send, the draft API that later
// controls use, the input recall seeded once from the room's own history, and the "+" menu with its mount points.
// Enter and Swarm are in chat_composer_enter.test.js; the real engines are in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { composerMarkup, createPlusMenu } from '../modules/composer.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { installDomDouble } from './dom_double.js';

const tick = () => new Promise((resolve) => setImmediate(resolve));
const SUFFIXES = ['input-area', 'attachment-preview', 'subagents-mode', 'context-mode', 'plus', 'file-input',
    'input', 'send'];

function fetchStub(history = []) {
    return async (url) => ({ ok: true, json: async () => (String(url).startsWith('/api/chat/history')
        ? { messages: history, window: { complete: true } }
        : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true }) });
}

function mount(env, { chatId = 1, projectId = '', idPrefix = 'chat', frames = [] } = {}) {
    let generation = 0;
    return createChatInstance({
        ws: { on: () => () => {}, isConnected: () => true,
            send: (frame) => { frames.push(frame); return { status: 'sent', clientMessageId: `cm-${frames.length}` }; } },
        state: { activePage: 'chat', projectChatIds: new Set(chatId === 1 ? [] : [chatId]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, projectId, idPrefix, mountEl: env.mount, asPanel: chatId !== 1,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
}

const PROJECT = { chatId: 7, projectId: 'p7', idPrefix: 'pchat-p7' };
const node = (instance, id) => instance.page.querySelector(`[id="${id}"]`);

test('every room mounts the composer with the chat-* ids, namespaced in a Project panel with the mirror classes', () => {
    for (const options of [{}, PROJECT]) {
        const prefix = options.idPrefix || 'chat';
        const env = installDom(fetchStub());
        const instance = mount(env, options);
        try {
            for (const suffix of SUFFIXES) {
                const el = node(instance, `${prefix}-${suffix}`);
                assert.ok(el, `${prefix}-${suffix}`);
                if (prefix !== 'chat') {
                    assert.equal(el.classList.contains(`chat-${suffix}`), true, `${suffix}: the mirror class keeps the styling`);
                    assert.equal(node(instance, `chat-${suffix}`), null, `${suffix}: no Main id inside a Project panel`);
                }
            }
            assert.equal(instance.composer.input, node(instance, `${prefix}-input`), 'the composer owns the room\'s field');
            assert.equal(Object.hasOwn(node(instance, `${prefix}-plus`).dataset, 'composerPlus'), true);
            assert.equal(typeof instance.composer.plusMenu.setSections, 'function');
            assert.equal(Object.hasOwn(instance.composer.voiceSlot.dataset, 'composerVoice'), true);
            assert.equal(Object.hasOwn(instance.composer.chipsSlot.dataset, 'composerChips'), true);
        } finally { instance.destroy(); restoreDom(env.prior); }
    }
});

test('"+" stands where the paperclip stood; the voice and chips slots are empty, hidden and in their places', () => {
    const markup = composerMarkup();
    const tag = (pattern) => markup.match(pattern)?.[0] || '';
    const plus = tag(/<button class="chat-plus-btn[ "][^>]*>/);
    for (const attr of ['ui-hit-area', 'id="chat-plus"', 'type="button"', 'data-composer-plus', 'aria-label="More actions"',
        'aria-haspopup="menu"', 'aria-expanded="false"']) assert.ok(plus.includes(attr), attr);
    assert.doesNotMatch(markup, /chat-attach-btn|id="chat-attach"/, 'no paperclip beside it');
    assert.match(markup, /<span class="chat-voice-slot" data-composer-voice hidden><\/span>/);
    assert.match(markup, /<span class="chat-composer-chips" data-composer-chips hidden><\/span>/);
    // Text row: "+", the file input, the field, the voice slot, Send. Capsule row: Subagents, Context, then the chips
    // slot (Effort mounts between them; Main's questions pill joins after). No one-shot Swarm pill anywhere.
    assert.doesNotMatch(markup, /chat-swarm|chat-composer-pills|data-armed/);
    const order = (names) => names.map((name) => markup.indexOf(name));
    const rising = (list) => list.every((value, i) => value >= 0 && (i === 0 || value > list[i - 1]));
    assert.ok(rising(order(['chat-plus-btn', 'chat-file-input', 'id="chat-input"', 'data-composer-voice', 'chat-send-group'])));
    assert.ok(rising(order(['chat-subagents-mode', 'chat-context-mode', 'data-composer-chips', 'chat-text-row'])));
});

test('a room keeps its own draft: setDraft and insertAtCursor keep it, a reopened room restores it, Send clears it', async () => {
    const env = installDom(fetchStub());
    const frames = [];
    let instance = mount(env, { ...PROJECT, frames });
    try {
        const { composer } = instance;
        composer.setDraft('hello world');
        assert.equal(composer.getDraft(), 'hello world');
        assert.equal(sessionStorage.getItem('ouro_chat_draft:7'), 'hello world', 'kept under the room\'s own key');
        assert.equal(sessionStorage.getItem('ouro_chat_draft'), null, 'never under Main\'s');
        // A selection is replaced by the inserted words; the caret follows them in the focused field.
        const field = composer.input;
        let caret = null;
        field.setSelectionRange = (start, end) => { caret = [start, end]; };
        field.selectionStart = 6;
        field.selectionEnd = 11;
        composer.insertAtCursor('there');
        assert.equal(composer.getDraft(), 'hello there');
        assert.deepEqual(caret, [11, 11]);
        assert.equal(document.activeElement, field, 'the field has focus for the next word');
        assert.equal(sessionStorage.getItem('ouro_chat_draft:7'), 'hello there');
        // A collapsed caret inserts in place, nothing replaced.
        field.selectionStart = 5;
        field.selectionEnd = 5;
        composer.insertAtCursor(',');
        assert.equal(composer.getDraft(), 'hello, there');
        assert.deepEqual(caret, [6, 6]);

        instance.destroy();
        instance = mount(env, { ...PROJECT, frames });
        assert.equal(instance.composer.getDraft(), 'hello, there', 'a reopened room restores its draft');
        node(instance, 'pchat-p7-send').click();
        await tick();
        assert.equal(frames.length, 1);
        assert.deepEqual([frames[0].content, frames[0].chat_id, frames[0].project_id], ['hello, there', 7, 'p7']);
        assert.equal(instance.composer.getDraft(), '');
        assert.equal(sessionStorage.getItem('ouro_chat_draft:7'), null, 'Send clears the kept draft');
        instance.composer.setDraft('');
        assert.equal(sessionStorage.getItem('ouro_chat_draft:7'), null, 'an empty draft is not kept');
    } finally { instance.destroy(); restoreDom(env.prior); }
});

test('the input recall is seeded once from the room\'s own sent words, newest nearest', async () => {
    const row = (id, role, text) => ({ role, text, ts: `2026-10-17T10:0${id}:00.000Z`, history_id: `chat:${id}`,
        history_position: { source: 'chat', offset: id } });
    const env = installDom(fetchStub([row(1, 'user', 'first ask'), row(2, 'assistant', 'an answer'), row(3, 'user', 'second ask')]));
    const instance = mount(env, PROJECT);
    try {
        for (let i = 0; i < 10; i += 1) await tick();
        const field = instance.composer.input;
        field.setSelectionRange = (start, end) => { field.selectionStart = start; field.selectionEnd = end; };
        const press = (key) => { for (const listener of field.listeners.get('keydown')) listener({ key, target: field, preventDefault() {} }); };
        field.selectionStart = 0;
        field.selectionEnd = 0;
        press('ArrowUp');
        assert.equal(field.value, 'second ask');
        field.setSelectionRange(0, 0);
        press('ArrowUp');
        assert.equal(field.value, 'first ask', 'an assistant row is never recalled');
        press('ArrowDown');
        assert.equal(field.value, 'second ask');
        press('ArrowDown');
        assert.equal(field.value, '', 'past the newest: the words being written');
        assert.deepEqual(JSON.parse(sessionStorage.getItem('ouro_chat_input_history')), ['first ask', 'second ask']);
    } finally { instance.destroy(); restoreDom(env.prior); }
});

function plusFixture() {
    const env = installDomDouble();
    const { doc, el } = env;
    const field = el('textarea');
    const button = el('button', { attrs: { 'aria-label': 'More actions', 'aria-expanded': 'false' } });
    doc.body.append(el('div', { className: 'chat-text-row' }, button, field));
    const chosen = [];
    const menu = createPlusMenu({ button, hostItems: [
        { key: 'attach', label: 'Attach file', icon: 'attach', onSelect: () => chosen.push('attach') },
    ] });
    const sheet = () => doc.body.children.find((child) => child.classList.contains('chat-plus-menu')) || null;
    const items = () => sheet()?.querySelectorAll('[role="menuitem"]') || [];
    return { ...env, field, button, menu, chosen, sheet, items, keys: () => items().map((item) => item.dataset.plusItem),
        key: (target, key) => target.emit('keydown', { key }),
        close() { menu.destroy(); env.restore(); } };
}

test('"+" by keyboard opens the menu with focus on Attach file; arrows move, Esc closes and returns focus', () => {
    const f = plusFixture();
    try {
        // "+" at the composer's left end, at the bottom of the window.
        f.button.rect = { left: 40, right: 76, top: 700, bottom: 736, width: 36, height: 36 };
        f.button.click();  // Enter or Space: a click with no pointer count
        const menu = f.sheet();
        assert.ok(menu, 'the menu is mounted outside the composer');
        assert.deepEqual([menu.css.get('--ui-popup-left'), menu.css.get('--ui-popup-top')], ['40px', '666px'],
            'it opens above "+", lined up with its start');
        assert.equal(menu.getAttribute('role'), 'menu');
        assert.equal(menu.getAttribute('aria-label'), 'More actions');
        assert.equal(menu.classList.contains('ui-popup'), true, 'the shared menu sheet and its viewport placement');
        assert.equal(f.button.getAttribute('aria-expanded'), 'true');
        assert.deepEqual(f.keys(), ['attach']);
        const [attach] = f.items();
        assert.equal(attach.textContent, 'Attach file');
        assert.equal(attach.children[0].getAttribute('aria-hidden'), 'true', 'the icon is decoration');
        assert.equal(f.doc.activeElement, attach);
        f.key(attach, 'Escape');
        assert.equal(f.sheet(), null);
        assert.equal(f.button.getAttribute('aria-expanded'), 'false');
        assert.equal(f.doc.activeElement, f.button, 'focus returns to "+"');
        f.button.click();
        f.items()[0].click();  // Enter or Space on the focused item
        assert.deepEqual(f.chosen, ['attach'], 'Attach file does what the paperclip did');
        assert.equal(f.sheet(), null, 'choosing closes the menu');
    } finally { f.close(); }
});

test('a pointer opens and chooses from "+" without taking the field\'s focus; Esc and an outside press close it', () => {
    const f = plusFixture();
    try {
        f.field.focus();
        assert.equal(f.button.emit('mousedown').defaultPrevented, true, 'pressing "+" keeps the field focused');
        f.button.emit('click', { detail: 1 });
        assert.ok(f.sheet());
        assert.equal(f.doc.activeElement, f.field, 'a phone keeps its keyboard');
        const escape = f.key(f.field, 'Escape');
        assert.equal(escape.stopped, true, 'Esc closes only the menu, not the field\'s own Esc');
        assert.equal(f.sheet(), null);
        assert.equal(f.doc.activeElement, f.field);
        f.button.emit('click', { detail: 1 });
        assert.equal(f.items()[0].emit('mousedown').defaultPrevented, true);
        f.items()[0].emit('click', { detail: 1 });
        assert.deepEqual([f.chosen, f.sheet(), f.doc.activeElement], [['attach'], null, f.field]);
        f.button.emit('click', { detail: 1 });
        f.doc.body.emit('pointerdown');
        assert.equal(f.sheet(), null, 'an outside press closes it');
        f.button.focus();
        f.button.emit('click', { detail: 1 });
        assert.equal(f.doc.activeElement, f.items()[0], 'a press on a focused "+" (a screen reader) moves focus in');
        f.key(f.items()[0], 'Escape');
        f.field.focus();
        f.button.emit('click', { detail: 1 });
        f.button.emit('click', { detail: 1 });
        assert.equal(f.sheet(), null, 'a second press on "+" closes it');
        // With the menu closed, Esc in the field is the field's again.
        assert.notEqual(f.key(f.field, 'Escape').stopped, true);
    } finally { f.close(); }
});

test('later controls add labelled sections after Attach file; a disabled item is skipped and an empty section omitted', () => {
    const f = plusFixture();
    try {
        const picked = [];
        f.menu.setSections([
            { key: 'templates', label: 'Templates', items: [
                { key: 'templates:simpler', label: 'Explain simpler', title: 'Ask for a simpler answer', onSelect: () => picked.push('simpler') },
                { key: 'templates:off', label: 'Unavailable here', disabled: true, onSelect: () => picked.push('off') },
            ] },
            { key: 'empty', label: 'Nothing yet', items: [] },
        ]);
        f.button.click();
        assert.deepEqual(f.keys(), ['attach', 'templates:simpler', 'templates:off']);
        const group = f.sheet().children[1];
        assert.deepEqual([group.getAttribute('role'), group.getAttribute('aria-label'), group.dataset.plusSection],
            ['group', 'Templates', 'templates']);
        assert.equal(group.children[0].textContent, 'Templates');
        assert.equal(f.sheet().children.length, 2, 'the empty section is not drawn');
        const [attach, simpler, off] = f.items();
        assert.equal(simpler.title, 'Ask for a simpler answer');
        assert.equal(off.disabled, true);
        f.key(attach, 'ArrowDown');
        assert.equal(f.doc.activeElement, simpler);
        f.key(simpler, 'ArrowDown');
        assert.equal(f.doc.activeElement, attach, 'the disabled item is skipped');
        f.key(attach, 'End');
        assert.equal(f.doc.activeElement, simpler, 'End reaches the last item that can be chosen');
        simpler.click();
        assert.deepEqual(picked, ['simpler']);
        f.button.click();
        f.menu.setSections([]);
        assert.deepEqual(f.keys(), ['attach', 'templates:simpler', 'templates:off'], 'an open menu keeps its list');
        f.button.click();
        f.button.click();
        assert.deepEqual(f.keys(), ['attach'], 'the next opening shows the new one');
        f.menu.destroy();
        assert.equal(f.sheet(), null, 'destroy removes an open menu');
    } finally { f.close(); }
});
