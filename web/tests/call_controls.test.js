import test from 'node:test';
import assert from 'node:assert/strict';
import { bindConversationControls, projectRoomOptions, startConversations } from '../modules/conversation_controls.js';

test('conversation mounts receive the room label, reveal, visibility and settled callback from navigation', () => {
    let options, shown = false, active = true, pending = false, destroyed = 0, revealed = 0, released = 0;
    const entry = { impl: { mountRoom(value) { options = value; return {
        retains: () => active, destroy: () => destroyed++,
    }; } } };
    const runtime = { onProvidersChange() {}, clientProviders: () => [entry] };
    startConversations({ hosts: runtime, openSettings() {}, roomOptions: ({ chatId }) => ({
        label: () => `Project ${chatId}`, onScreen: () => shown,
        reveal() { shown = true; revealed++; }, onSettled() { released++; },
    }) });
    const composer = { hasPendingWork: () => pending };
    const dispose = bindConversationControls({ composer, page: { getClientRects: () => [1] }, chatId: 71 });
    try {
        assert.equal(options.label?.(), 'Project 71');
        assert.equal(options.onScreen(), false);
        options.reveal();
        assert.equal(revealed, 1);
        assert.equal(options.onScreen(), true);
        assert.equal(composer.hasPendingWork(), true);
        active = false;
        assert.equal(composer.hasPendingWork(), false);
        pending = true;
        assert.equal(composer.hasPendingWork(), true);
        options.onSettled();
        assert.equal(released, 1);
    } finally { dispose(); }
    assert.equal(destroyed, 1);
    assert.equal(composer.hasPendingWork(), true, 'original pending-work owner restored');
});

test('project room options name Main and a Project, reveal the right page and release a project left behind', async () => {
    const nav = { activeProjectId: 'p7' }, state = { activePage: 'chat' };
    const rows = [{ id: 'p7', chat_id: 7, name: 'Seven' }];
    const opened = [], shown = [], destroyed = [];
    const options = projectRoomOptions({
        rows: () => rows, nav, state, title: { textContent: 'Panel title' },
        freshProjectRow: (project) => rows.find(row => row.id === project.id) || project,
        openProjectPanel: (project) => opened.push(project.id), showPage: (name) => shown.push(name),
        putAway: (id) => destroyed.push(id),
    });
    const main = options({ page: { hidden: false }, chatId: 1 });
    assert.equal(main.label(), 'Main');
    assert.equal(main.onScreen(), false);  // a project is active, so Main is not on screen
    main.reveal();
    assert.deepEqual(shown, ['chat']);
    const seven = options({ page: { hidden: false }, chatId: 7 });
    assert.equal(seven.label(), 'Seven');
    assert.equal(seven.onScreen(), true);
    seven.reveal();
    assert.deepEqual(opened, ['p7']);
    nav.activeProjectId = 'p9';  // the owner opened an unlisted Project directly: its panel is the active one
    const unlisted = options({ page: { hidden: true }, chatId: 9 });  // not in the census
    assert.equal(unlisted.label(), 'Panel title');
    assert.equal(unlisted.onScreen(), false);
    seven.onSettled();
    await new Promise(resolve => queueMicrotask(resolve));
    assert.deepEqual(destroyed, ['p7']);  // the project the owner left is released once the call settled
    main.onSettled();
    await new Promise(resolve => queueMicrotask(resolve));
    assert.deepEqual(destroyed, ['p7']);  // Main never destroys anything
});

// The send button's face from the first frame (U018): a conversation feature that offers Talk paints it seconds after
// a load, so the room paints the face the feature last asked this device for at once and hands it over at mount.
const FACE = { html: '<span class="call-entry-mark" aria-hidden="true"><svg width="20" height="14"></svg></span><span class="call-entry-word">Talk</span>',
    title: 'Talk with Ouroboros' };
const FILLED_BADGE = { className: 'attach-badge' };

function fakeButton() {
    const attrs = new Map();
    let html = 'Send';
    return {
        dataset: {}, title: 'Send message',
        get innerHTML() { return html; }, set innerHTML(value) { html = String(value); },
        get textContent() { return html.replace(/<[^>]*>/g, ''); }, set textContent(value) { html = String(value); },
        setAttribute: (key, value) => attrs.set(key, String(value)), removeAttribute: (key) => attrs.delete(key),
        getAttribute: (key) => attrs.get(key) ?? null,
    };
}

function entryRoom({ remembered = null, value = '', store = null } = {}) {
    const storage = new Map(remembered ? [['ouro:conversation-entry', typeof remembered === 'string' ? remembered : JSON.stringify(remembered)]] : []);
    globalThis.localStorage = { getItem: (k) => storage.get(k) ?? null, setItem: (k, v) => storage.set(k, String(v)), removeItem: (k) => storage.delete(k) };
    const observers = [];
    globalThis.MutationObserver = class {
        constructor(fn) { this.fn = fn; observers.push(this); }
        observe(target) { this.target = target; }
        disconnect() { this.target = null; }
    };
    const listeners = new Map();
    const send = fakeButton();
    const field = { value, addEventListener: (type, fn) => listeners.set(type, fn), removeEventListener: (type) => listeners.delete(type) };
    const preview = { badge: null, querySelector: (sel) => (sel === '.attach-badge' ? preview.badge : null) };
    const page = { getClientRects: () => [1], querySelector: (sel) => ({ '.chat-send-inline': send, textarea: field, '.chat-attachment-preview': preview })[sel] || null };
    const contributions = store || { status: () => ({ loaded: false }), providers: () => [], fault: () => null, onChange: () => () => {} };
    const changed = (target) => observers.filter((o) => o.target === target).forEach((o) => o.fn([]));
    return { storage, send, field, preview, listeners, page, contributions, changed };
}

// The room binds before the feature's module has loaded; `mount()` loads it. A feature of this version takes the face over
// while it mounts (`claims`); an older one does not know the hand-over.
function bindRoom(room, { claims = true } = {}) {
    let options = null, claimed = null, available = [];
    const changed = [];
    const provider = { impl: { mountRoom(value) { options = value; claimed = claims ? value.claimEntry() : null; return { destroy() {} }; } } };
    startConversations({ hosts: { onProvidersChange: (fn) => changed.push(fn), clientProviders: () => available }, openSettings() {} });
    const dispose = bindConversationControls({ composer: { hasPendingWork: () => false }, page: room.page, chatId: 1, contributions: room.contributions });
    const mount = () => { available = [provider]; changed.forEach((fn) => fn()); return options; };
    return { mount, claimed: () => claimed, dispose };
}

test('a room paints the face its conversation feature last asked for from the first frame, and Send without one', () => {
    try {
        const talk = entryRoom({ remembered: FACE });
        const bound = bindRoom(talk);
        try {
            assert.equal(talk.send.innerHTML, FACE.html, 'the remembered face is painted before the feature mounts');
            assert.equal(talk.send.title, FACE.title);
            assert.equal(talk.send.getAttribute('aria-disabled'), 'true', 'a press cannot start a call before the mount');
            const options = bound.mount();
            assert.equal(bound.claimed(), true, 'the feature takes the shown face over');
            assert.equal(talk.send.innerHTML, FACE.html, 'handing over changes no pixel');
            assert.equal(talk.send.getAttribute('aria-disabled'), null);
            assert.equal(talk.send.dataset.entryReserved, undefined);
            assert.equal(options.claimEntry(), false, 'one hand-over');
            options.rememberEntry(null);
            assert.equal(talk.send.innerHTML, FACE.html, 'after the hand-over the face is the feature\'s to change');
            assert.equal(talk.storage.has('ouro:conversation-entry'), false, 'the next load starts with Send');
            options.rememberEntry(FACE);
            assert.deepEqual(JSON.parse(talk.storage.get('ouro:conversation-entry')), FACE, 'the next load paints Talk again');
        } finally { bound.dispose(); }
        for (const [remembered, value, why] of [[null, '', 'nothing remembered'], [FACE, 'hello', 'a restored draft'],
            ['talk', '', 'an unreadable record'], ['{"html":', '', 'a broken record']]) {
            const room = entryRoom({ remembered, value });
            const plain = bindRoom(room);
            try {
                assert.equal(room.send.innerHTML, 'Send', `${why}: the first frame is Send`);
                assert.equal(room.send.getAttribute('aria-disabled'), null, why);
                plain.mount();
                assert.equal(plain.claimed(), false, `${why}: nothing to hand over`);
            } finally { plain.dispose(); }
        }
    } finally { delete globalThis.localStorage; delete globalThis.MutationObserver; }
});

test('a reserved face gives way to Send when the composer fills or no feature will take it, and to any other paint', () => {
    try {
        const filled = [['input', (room) => { room.field.value = 'hello'; }], ['keydown', (room) => { room.field.value = 'a recalled message'; }],
            ['change', (room) => { room.field.value = 'dictated'; }], ['staged', (room) => { room.preview.badge = FILLED_BADGE; }]];
        for (const [how, fill] of filled) {
            const room = entryRoom({ remembered: FACE });
            const bound = bindRoom(room);
            try {
                if (how === 'staged') { fill(room); room.changed(room.preview); }
                else { room.listeners.get(how)(); assert.equal(room.send.innerHTML, FACE.html, `${how} on an empty field keeps the face`); fill(room); room.listeners.get(how)(); }
                assert.equal(room.send.textContent, 'Send', `${how}: Send`);
                assert.equal(room.send.title, 'Send message');
                assert.equal(room.send.getAttribute('aria-disabled'), null);
                bound.mount();
                assert.equal(bound.claimed(), false, 'the feature starts from its own read');
                assert.equal(room.storage.has('ouro:conversation-entry'), true, 'the owner filling the composer is not the feature\'s answer');
            } finally { bound.dispose(); }
        }
        const loaded = (providers) => ({ status: () => ({ loaded: true }), providers: () => providers, fault: () => null, onChange: () => () => {} });
        const live = { key: 'live_conversation:call', runtime: 'client', provides: 'conversation' };
        for (const [providers, why] of [[[], 'no live conversation feature'], [[live, { ...live, key: 'other:call' }], 'two features, none mounts']]) {
            const gone = entryRoom({ remembered: FACE, store: loaded(providers) });
            const bound = bindRoom(gone);
            try {
                assert.equal(gone.send.textContent, 'Send', `${why}: Send`);
                assert.equal(gone.storage.has('ouro:conversation-entry'), false, why);
            } finally { bound.dispose(); }
        }
        const kept = entryRoom({ remembered: FACE, store: loaded([live]) });
        const boundKept = bindRoom(kept);
        try { assert.equal(kept.send.innerHTML, FACE.html, 'a declared live feature keeps the face until it mounts'); } finally { boundKept.dispose(); }
        const repainted = entryRoom({ remembered: FACE });
        const boundRepainted = bindRoom(repainted);
        try {
            repainted.send.textContent = 'Saving';  // the chat's own label (a pending room setting before a send)
            repainted.changed(repainted.send);
            assert.equal(repainted.send.textContent, 'Saving', 'another paint is never overwritten');
            assert.equal(repainted.send.getAttribute('aria-disabled'), null);
            boundRepainted.mount();
            assert.equal(boundRepainted.claimed(), false);
        } finally { boundRepainted.dispose(); }
        const older = entryRoom({ remembered: FACE });
        const boundOlder = bindRoom(older, { claims: false });
        try {
            boundOlder.mount();
            assert.equal(older.send.textContent, 'Send', 'a feature that does not take the face over paints its own, from Send');
            assert.equal(older.send.getAttribute('aria-disabled'), null);
            assert.equal(older.storage.has('ouro:conversation-entry'), true, 'its version is not the feature\'s answer');
        } finally { boundOlder.dispose(); }
    } finally { delete globalThis.localStorage; delete globalThis.MutationObserver; }
});
