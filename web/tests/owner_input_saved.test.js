// В9: the host stamps a typed `ingress_accepted: true` on the web owner echo
// only after the durable chat write (supervisor/message_bus.py
// handle_web_message: log_chat(require_write=True), then the echo). The owner
// bubble with that client_message_id may then say `Input saved` — a fact about
// the saved row, never that work started. An echo without the typed flag is
// unknown and adds nothing, and the flag never retires `Sending...` (2A).
import assert from 'node:assert/strict';
import test from 'node:test';
import { markIngressSaved } from '../modules/chat_activity.js';
import { createUnconfirmedSends, placeIngressNote } from '../modules/chat_attachments.js';
import { createChatInstance } from '../modules/chat.js';
import { ordinaryUnavailability, previewEntryKey, PREVIEW_FAMILY, PREVIEW_KEY } from '../modules/chat_preview.js';
import { installDom, PREVIEW_SCOPE, restoreDom, roomKey } from './chat_dom_fixture.js';

const ME = 'session-me';
const TS = '2026-09-25T09:00:00Z';

function fixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { profile_scope: PREVIEW_SCOPE, messages: history, window: { complete: true } }
            : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } }));
    globalThis.sessionStorage.setItem('ouro_chat_session_id', ME);
    const handlers = new Map();
    let generation = 0;
    let sent = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send: () => ({ status: 'sent', clientMessageId: `cm-${++sent}` }) },
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const messages = document.byId.get('chat-messages');
    const bubble = (cmid) => messages.children.find((node) => node.dataset.clientMessageId === cmid);
    return {
        instance, messages, bubble,
        // The saved note wherever it stands: the quiet tick in the time line, or a row above it.
        notes: (cmid) => bubble(cmid)?.querySelectorAll('[data-ingress-saved]') || [],
        status: () => document.byId.get('chat-status')?.textContent,
        async send(text) {
            document.byId.get('chat-input').value = text;
            for (const fn of document.byId.get('chat-send').listeners.get('click')) fn();
            await settle();
        },
        echo: (row) => handlers.get('chat')({ type: 'chat', role: 'user', chat_id: 1, ts: TS, source: 'web', ...row }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}
const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };

test('the typed flag on my own echo says Input saved once, and Sending... stays', async () => {
    const f = fixture();
    try {
        await f.send('please look at the logs');
        assert.equal(f.status(), 'Sending');
        // A live echo of the running host always says what it did with the row (here: dispatched).
        f.echo({ content: 'please look at the logs', sender_session_id: ME, client_message_id: 'cm-1', ingress_accepted: true,
            ingress_dispatched: true });
        const [note] = f.notes('cm-1');
        assert.equal(note?.textContent, 'Input saved');
        assert.equal(note.className, 'msg-pending', 'the quiet delivery-note style, no new chrome');
        const time = f.bubble('cm-1').querySelector('.msg-time');
        assert.equal(note.parentNode, time, 'a quiet tick in the time line, before its words: no row, no second tick');
        assert.equal(note.title, 'Input saved', 'its words stay its text and a pointer\'s title');
        assert.equal(f.status(), 'Sending', 'a saved row is not a started turn');
        f.echo({ content: 'please look at the logs', sender_session_id: ME, client_message_id: 'cm-1', ingress_accepted: true,
            ingress_dispatched: true });
        assert.equal(f.notes('cm-1').length, 1, 'a repeated echo keeps one note');
        assert.equal(f.messages.children.filter((node) => node.classList.contains('user')).length, 1);
    } finally { f.close(); }
});

// Only the plain saved states draw the quiet tick. Any other state keeps its words in a row right above the time: the
// delivery doubt, a legacy `undispatched` ("Saved, not delivered."), and a note that was never saved at all.
test('only a plain saved note becomes the tick; every other state keeps its words above the time', () => {
    const env = installDom();
    try {
        for (const [state, where] of [['', 'time'], ['pending', 'time'], ['unconfirmed', 'row'], ['undispatched', 'row'], [undefined, 'row']]) {
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble user';
            const time = document.createElement('div');
            time.className = 'msg-time';
            bubble.appendChild(time);
            const note = Object.assign(document.createElement('div'), { className: 'msg-pending', textContent: 'words' });
            if (state !== undefined) note.dataset.ingressSaved = state;
            placeIngressNote(bubble, note);
            assert.equal(note.parentNode === time ? 'time' : 'row', where, String(state));
            if (where === 'row') assert.equal(bubble.children.indexOf(note) + 1, bubble.children.indexOf(time), `${state}: right above the time`);
        }
    } finally { restoreDom(env.prior); }
});

test('no typed flag, no note: absent, untyped, unkeyed, foreign or not an owner row', async () => {
    const f = fixture();
    try {
        await f.send('one');
        await f.send('two');
        for (const row of [
            { client_message_id: 'cm-1' },
            { client_message_id: 'cm-1', ingress_accepted: 'true' },
            { client_message_id: 'cm-1', ingress_accepted: 1 },
            { client_message_id: 'cm-1', ingress_accepted: false },
            { client_message_id: '', ingress_accepted: true },
            { client_message_id: 'cm-1', ingress_accepted: true, chat_id: 7 },
        ]) {
            f.echo({ content: 'one', sender_session_id: ME, ...row });
            assert.equal(f.notes('cm-1').length, 0, JSON.stringify(row));
        }
        f.echo({ content: 'two', sender_session_id: ME, client_message_id: 'cm-2', ingress_accepted: true });
        assert.equal(f.notes('cm-2').length, 1, 'keyed to its own client_message_id');
        assert.equal(f.notes('cm-1').length, 0, 'the other bubble stays unknown');
        assert.equal(markIngressSaved(f.messages, { role: 'assistant', client_message_id: 'cm-1', ingress_accepted: true }), false);
        assert.equal(f.notes('cm-1').length, 0);
    } finally { f.close(); }
});

test('an echo from another tab mounts the owner row with its note; without the flag it is plain', () => {
    const f = fixture();
    try {
        f.echo({ content: 'from my phone', sender_session_id: 'session-other', client_message_id: 'cm-other', ingress_accepted: true });
        assert.equal(f.notes('cm-other')[0]?.textContent, 'Input saved');
        f.echo({ content: 'telegram mirror', sender_session_id: 'session-other', client_message_id: 'cm-legacy' });
        assert.ok(f.bubble('cm-legacy'), 'the row still mounts');
        assert.equal(f.notes('cm-legacy').length, 0, 'an unflagged echo proves nothing about saving');
    } finally { f.close(); }
});

test('replay: history confirming a noted row keeps the note; a replayed row alone never gets one', async () => {
    const row = (cmid, text) => ({ role: 'user', text, ts: TS, chat_id: 1, source: 'web', sender_session_id: ME,
        client_message_id: cmid, history_id: `h-${cmid}` });
    const f = fixture([row('cm-1', 'saved live'), row('cm-old', 'earlier')]);
    try {
        await f.send('saved live');
        f.echo({ content: 'saved live', sender_session_id: ME, client_message_id: 'cm-1', ingress_accepted: true });
        await f.instance.refreshHistory({ revision: 1 });
        assert.equal(f.bubble('cm-1')?.dataset.historyId, 'h-cm-1', 'history re-stamped the same bubble');
        assert.equal(f.notes('cm-1').length, 1, 'the live fact survives the confirmation');
        assert.ok(f.bubble('cm-old'), 'the older row replays');
        assert.equal(f.notes('cm-old').length, 0, 'history rows carry no ingress flag: unknown, no note');
    } finally { f.close(); }
});

// A local send() of an attachment message is not acceptance either: until the saved echo
// arrives, its frame is kept under its own client_message_id. A socket close (or the host's
// refusal notice) before that says "Not confirmed as saved" with "Send again", which resends
// the SAME frame and id — the host rejoins a message it saved after all, never a second one.
test('a sent attachment message stays recoverable until the host says it is saved', async () => {
    const upload = `${'a'.repeat(32)}_plan.pdf`;
    const view = { name: 'plan.pdf', kind: 'file', mime: 'application/pdf', size: 4, available: true,
        url: `/api/files/download?upload=${upload}` };
    let history = [];
    const env = installDom(async (url) => ({ ok: true, status: 200, json: async () =>
        String(url).startsWith('/api/chat/upload') ? { ok: true, filename: upload, display_name: 'plan.pdf',
            mime: 'application/pdf', view }
            : String(url).startsWith('/api/chat/history') ? { profile_scope: PREVIEW_SCOPE, messages: history, window: { complete: true } }
                : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } }));
    globalThis.sessionStorage.setItem('ouro_chat_session_id', ME);
    const handlers = new Map();
    const fire = (type, payload) => [...(handlers.get(type) || [])].forEach((fn) => fn(payload));
    const frames = [];
    const socket = { readyState: WebSocket.OPEN };
    let generation = 0;
    const instance = createChatInstance({
        ws: {
            ws: socket,
            on(type, fn) { if (!handlers.has(type)) handlers.set(type, new Set()); handlers.get(type).add(fn); return () => handlers.get(type).delete(fn); },
            isConnected: () => socket.readyState === WebSocket.OPEN,
            send(frame, options) {
                const id = frame.client_message_id || `cm-${frames.length + 1}`;
                frames.push({ frame: { ...frame, client_message_id: id }, options });
                return { status: 'sent', clientMessageId: id };
            },
        },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const messages = document.byId.get('chat-messages');
    const bubble = (cmid) => messages.children.find((node) => node.dataset.clientMessageId === cmid);
    const doubt = (cmid) => (bubble(cmid)?.children || []).find((node) => Object.hasOwn(node.dataset, 'ingressUnconfirmed'));
    const sendWithFile = async (text) => {
        const fileInput = document.byId.get('chat-file-input');
        fileInput.files = [new File(['%PDF'], 'plan.pdf', { type: 'application/pdf' })];
        for (const fn of fileInput.listeners.get('change')) fn();
        document.byId.get('chat-input').value = text;
        for (const fn of document.byId.get('chat-send').listeners.get('click')) fn();
        await settle();
    };
    try {
        await sendWithFile('look');
        assert.equal(frames.length, 1);
        assert.deepEqual(frames[0].options, { queue: false });
        assert.deepEqual(frames[0].frame.attachments, [{ filename: upload, display_name: 'plan.pdf', mime: 'application/pdf' }]);
        assert.ok(bubble('cm-1') && !doubt('cm-1'), 'sent: a plain bubble, no claim either way');
        assert.equal(instance.hasPendingWork(), true, 'an unsaved message keeps its room alive (hidden, not destroyed)');

        fire('close');
        const note = doubt('cm-1');
        assert.equal(note?.className, 'msg-pending');
        assert.ok(note.textContent.startsWith('Not confirmed as saved.'));
        const button = note.children.find((node) => node.tagName === 'BUTTON');
        assert.equal(button?.textContent, 'Send again');
        fire('close');
        assert.equal(bubble('cm-1').children.filter((node) => Object.hasOwn(node.dataset, 'ingressUnconfirmed')).length, 1);

        button.click();
        assert.equal(frames.length, 2);
        assert.deepEqual(frames[1].frame, frames[0].frame, 'the SAME frame and client_message_id: never a second message');
        assert.deepEqual(frames[1].options, { queue: false });
        assert.equal(doubt('cm-1'), undefined, 'resent: the doubt waits for the next fact');

        fire('close');
        assert.ok(doubt('cm-1'), 'still unconfirmed after another close');
        fire('chat', { type: 'chat', role: 'user', chat_id: 1, ts: TS, source: 'web', content: 'look',
            sender_session_id: ME, client_message_id: 'cm-1', ingress_accepted: true, ingress_dispatched: true, attachments: [view] });
        assert.equal(doubt('cm-1'), undefined, 'the saved echo settles the doubt');
        assert.equal(instance.hasPendingWork(), false, 'the saved echo settles the frame itself');
        assert.equal(bubble('cm-1').querySelectorAll('[data-ingress-saved]').length, 1);
        fire('close');
        assert.equal(doubt('cm-1'), undefined, 'a saved message is never doubted again');

        await sendWithFile('second');
        fire('chat', { type: 'chat', role: 'system', system_type: 'initialization_notice', content: '⚠️ try again', ts: TS });
        assert.ok(doubt('cm-3'), "the host's refusal notice also offers Send again");
        assert.equal(doubt('cm-1'), undefined);

        // Its echo was lost, but history shows the host saved it: the history row settles it too.
        history = [{ role: 'user', text: 'second\n\n[Attached file: plan.pdf]', ts: TS, chat_id: 1, source: 'web',
            sender_session_id: ME, client_message_id: 'cm-3', history_id: 'h-cm-3', ingress_accepted: true, ingress_dispatched: true,
            attachments: [view] }];
        await instance.refreshHistory({ revision: 1 });
        assert.equal(doubt('cm-3'), undefined, 'the saved history row ends the doubt');
        assert.equal(instance.hasPendingWork(), false, 'and drops the frame');
    } finally {
        instance.destroy();
        restoreDom(env.prior);
    }
});

test('Send again while offline keeps the doubt and says why', () => {
    const env = installDom();
    try {
        const root = document.createElement('div');
        root.isConnected = true;
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble user';
        bubble.dataset.clientMessageId = 'cm-9';
        root.appendChild(bubble);
        bubble.isConnected = true;
        const toasts = [];
        const sends = [];
        let status = 'failed';
        const unconfirmed = createUnconfirmedSends({ send: (frame, options) => { sends.push([frame, options]); return { status }; },
            root: () => root, onDomWrite: (fn) => fn(), showToast: (text, tone, options) => toasts.push([text, tone, options?.haptic]) });
        unconfirmed.track({ type: 'chat', content: 'x', client_message_id: 'cm-9' });
        unconfirmed.unsettle();
        const note = bubble.children.find((node) => Object.hasOwn(node.dataset, 'ingressUnconfirmed'));
        note.children.find((node) => node.tagName === 'BUTTON').click();
        assert.deepEqual(toasts, [['Still offline. Reconnect and send again.', 'error', undefined]],
            'the owner\'s Send again failed: its toast keeps the error haptic');
        assert.ok(bubble.children.includes(note), 'the doubt and its button stay');
        status = 'sent';
        note.children.find((node) => node.tagName === 'BUTTON').click();
        assert.equal(sends.length, 2);
        assert.ok(!bubble.children.includes(note));

        // The feed releasing or rebuilding the bubble proves nothing about saving: the frame stays.
        bubble.remove();
        unconfirmed.unsettle();
        assert.equal(unconfirmed.count, 1, 'a bubble that left the feed keeps its unsaved frame');
        root.appendChild(bubble);
        unconfirmed.unsettle();
        const again = bubble.children.find((node) => Object.hasOwn(node.dataset, 'ingressUnconfirmed'));
        again.children.find((node) => node.tagName === 'BUTTON').click();
        assert.equal(sends.length, 3, 'back in the feed, its doubt and Send again return');
        assert.deepEqual(sends[2][0], sends[0][0], 'still the same frame and id');

        // Only the saved row settles it: an unflagged row, another id or an assistant row do not.
        for (const row of [{ role: 'user', client_message_id: 'cm-9' }, { role: 'assistant', client_message_id: 'cm-9', ingress_accepted: true },
            { role: 'user', client_message_id: 'cm-8', ingress_accepted: true }]) unconfirmed.settle(row);
        assert.equal(unconfirmed.count, 1);
        unconfirmed.settle({ role: 'user', client_message_id: 'cm-9', ingress_accepted: true, ingress_dispatched: true });
        assert.equal(unconfirmed.count, 0, 'the saved row ends the doubt');
        unconfirmed.unsettle();
        assert.equal(bubble.children.filter((node) => Object.hasOwn(node.dataset, 'ingressUnconfirmed')).length, 0);

        // Nothing is evicted by count: many unsaved sends all stay recoverable.
        for (let index = 0; index < 120; index += 1) unconfirmed.track({ type: 'chat', client_message_id: `bulk-${index}` });
        assert.equal(unconfirmed.count, 120);
    } finally {
        restoreDom(env.prior);
    }
});

// This tab keeps a sent attachment message (sessionStorage: words, id, routing, upload refs and
// the bubble's views — never bytes or the send-time surface) until the host says it is saved, so a
// reload or a room's teardown cannot lose its manual retry. The next instance reconciles the kept
// copy with its FIRST history read: confirmed dispatch ends it, pending dispatch keeps it,
// and anything else — no row, or a read that fails while the core says the room is still one of the
// page's profile's ordinary rooms — is in
// doubt. Nothing resends by itself; "Discard" forgets only this tab's copy, never an upload.
function keptTab({ chatId = 7 } = {}) {
    const upload = `${'b'.repeat(32)}_scan.pdf`;
    const view = { name: 'scan.pdf', kind: 'file', mime: 'application/pdf', size: 4, available: true,
        url: `/api/files/download?upload=${upload}` };
    // `socket`: the page's one socket (null while it is down); `profile`: whether it proved the page's data profile
    // (ws.js `connection_profile`); `ordinary`: whether the core still counts the room among the profile's ordinary rooms;
    // `gate`: a promise the history door's answers wait for, when a case holds them; `status`: another answer it gives
    // instead (a refusal, say); `previews`: whether the socket lets this browser's saved text paint (ws.js).
    const host = { history: [], fails: false, calls: [], socket: { readyState: 1 }, profile: true, ordinary: true, gate: null,
        status: 0, previews: false };
    const env = installDom(async (url, init = {}) => {
        host.calls.push([String(url), String(init.method || 'GET')]);
        const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
        if (String(url).startsWith('/api/chat/upload')) {
            return reply({ ok: true, filename: upload, display_name: 'scan.pdf', mime: 'application/pdf', view });
        }
        if (String(url).startsWith('/api/chat/history')) {
            if (host.gate) await host.gate;
            if (host.status) return reply({ error: 'Refused', reason_code: 'refused' }, host.status);
            // The core's answer while the room's source cannot be read (gateway/history.py): a 503 naming its reason and,
            // while the room is still one of the profile's ordinary rooms, that room under the profile's scope.
            return host.fails ? reply({ messages: [], error: 'history_source_unavailable', reason_code: 'history_source_unavailable',
                has_more: true, ...(host.ordinary ? { ordinary_room: chatId, profile_scope: PREVIEW_SCOPE } : {}) }, 503)
                : reply({ profile_scope: PREVIEW_SCOPE, messages: host.history, window: { complete: true } });
        }
        return reply({ active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true });
    });
    globalThis.sessionStorage.setItem('ouro_chat_session_id', ME);
    const priorSocket = globalThis.WebSocket;
    globalThis.WebSocket = { OPEN: 1 };
    let coined = 0;
    const rooms = [];
    // A saved row is the live host process's (its dispatch entered) unless dispatch is pending, or `restarted`:
    // taken by a host process that has since ended, which no longer knows what became of it.
    const live = ({ restarted, ...extra } = {}) => ({ ...(extra.ingress_pending || restarted ? {} : { ingress_dispatched: true }), ...extra });
    // One room instance (a Project room, or Main for `chatId: 1`), as a reload or a reopened room creates it on this tab's storage.
    function open() {
        const handlers = new Map();
        const sent = [];
        let generation = 0;
        const instance = createChatInstance({
            ws: { get ws() { return host.socket; },
                on(type, fn) { if (!handlers.has(type)) handlers.set(type, new Set()); handlers.get(type).add(fn); return () => handlers.get(type).delete(fn); },
                isConnected: () => host.socket !== null,
                // The profile the current socket's server named at accept (ws.js `connection_profile`), once it has.
                servingProfileScope: () => (host.socket && host.profile ? PREVIEW_SCOPE : ''),
                canShowSavedProfile: () => host.previews,
                send(frame, options) {
                    const id = frame.client_message_id || `cm-${++coined}`;
                    sent.push({ frame: { ...frame, client_message_id: id }, options });
                    return { status: 'sent', clientMessageId: id };
                } },
            state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
            updateUnreadBadge() {}, chatId, idPrefix: 'chat', mountEl: env.mount, asPanel: chatId !== 1,
            stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
        });
        const messages = document.byId.get('chat-messages');
        const bubble = (cmid) => messages.children.find((node) => node.dataset.clientMessageId === cmid);
        const room = {
            instance, sent, bubble,
            bubbles: (cmid) => messages.children.filter((node) => node.dataset.clientMessageId === cmid).length,
            preview: () => messages.children.find((node) => node.dataset.savedPreview === 'true'),
            fire: (type, payload) => [...(handlers.get(type) || [])].forEach((fn) => fn(payload)),
            notes: (cmid) => bubble(cmid)?.querySelectorAll('.msg-pending') || [],
            actions: (cmid) => room.notes(cmid).flatMap((note) => note.children.filter((node) => node.tagName === 'BUTTON')),
            // Send again and Discard rejoin or forget the kept frame; `Send as a new message` is the doubt's own action (it only
            // prepares the composer). Tests of the first kind must not read the second as a retry.
            retries: (cmid) => room.actions(cmid).filter((node) => node.dataset.unconfirmedAction !== 'send_as_new'),
            asNew: (cmid) => room.actions(cmid).filter((node) => node.dataset.unconfirmedAction === 'send_as_new'),
            press: (cmid, label) => room.actions(cmid).find((node) => node.textContent === label).click(),
            async send(text) {
                const fileInput = document.byId.get('chat-file-input');
                fileInput.files = [new File(['%PDF'], 'scan.pdf', { type: 'application/pdf' })];
                for (const fn of fileInput.listeners.get('change')) fn();
                document.byId.get('chat-input').value = text;
                for (const fn of document.byId.get('chat-send').listeners.get('click')) fn();
                await settle();
            },
            echo: (cmid, content, extra = {}) => room.fire('chat', { type: 'chat', role: 'user', chat_id: chatId, ts: TS, source: 'web',
                content, sender_session_id: ME, client_message_id: cmid, ingress_accepted: true, attachments: [view], ...live(extra) }),
        };
        rooms.push(room);
        return room;
    }
    const kept = () => JSON.parse(globalThis.sessionStorage.getItem(roomKey('ouro_chat_unconfirmed', chatId)) || '[]');
    const row = (cmid, text, extra = {}) => ({ role: 'user', text, ts: TS, chat_id: chatId, source: 'web', sender_session_id: ME,
        client_message_id: cmid, history_id: `h-${cmid}`, ingress_accepted: true, attachments: [view], ...live(extra) });
    return {
        host, view, open, kept, row,
        close() { for (const room of rooms) room.instance.destroy(); restoreDom(env.prior); globalThis.WebSocket = priorSocket; },
    };
}
const until = async (check, why) => {
    for (let i = 0; i < 200 && !check(); i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
    assert.ok(check(), why);
};

test('an initialization refusal names one kept send in its own room; a nameless refusal names the room', async () => {
    const tab = keptTab();
    try {
        const room = tab.open();
        await room.send('first');
        await room.send('second');
        const counts = () => ['cm-1', 'cm-2'].map((id) => room.actions(id).length);
        const notice = (chat_id, client_message_id = '') => room.fire('chat', {
            type: 'chat', role: 'system', system_type: 'initialization_notice', chat_id, client_message_id,
            content: 'Still starting. Send again when ready.',
        });
        assert.deepEqual(counts(), [0, 0]);
        notice(1, 'cm-1');
        notice(1);
        notice(7, 'other-message');
        assert.deepEqual(counts(), [0, 0], 'a foreign room or unknown id changes neither send');
        notice(7, 'cm-1');
        assert.deepEqual(counts(), [2, 0], 'the chat consumer scopes doubt to the named submission');
        assert.deepEqual(room.actions('cm-1').map((node) => node.textContent), ['Send again', 'Discard']);
        room.press('cm-1', 'Send again');
        assert.deepEqual(room.sent[2], room.sent[0], 'retry keeps the original frame and id');
        assert.deepEqual(counts(), [0, 0]);
        notice(7);
        assert.deepEqual(counts(), [2, 2], 'the legacy nameless refusal still doubts the whole room');
        assert.equal(room.sent.length, 3, 'a notice never resends by itself');
    } finally { tab.close(); }
});

test('a sent attachment message outlives a reload and a room teardown: Send again with the same id', async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('look');
        const [{ frame }] = first.sent;
        assert.deepEqual(Object.keys(tab.kept()[0].frame).sort(),
            ['attachments', 'chat_id', 'client_message_id', 'content', 'sender_session_id', 'type']);
        assert.equal(tab.kept()[0].frame.content, 'look\n\n[Attached file: scan.pdf]');
        assert.deepEqual(tab.kept()[0].views, [tab.view]);
        first.instance.destroy();
        assert.equal(tab.kept().length, 1, "a room's teardown keeps the tab's copy");

        const reloaded = tab.open();  // the host never saw it: its history has no row
        await until(() => reloaded.actions('cm-1').length === 2, 'the kept message is back, in doubt, after the first read');
        assert.ok(reloaded.notes('cm-1')[0].textContent.startsWith('Not confirmed as saved.'));
        assert.deepEqual(reloaded.actions('cm-1').map((node) => node.textContent), ['Send again', 'Discard']);
        assert.equal(reloaded.sent.length, 0, 'nothing resends by itself');
        assert.ok(reloaded.bubble('cm-1').children.some((node) => node.className === 'chat-attachments'), 'with its attachments');
        assert.equal(reloaded.instance.hasPendingWork(), true);

        reloaded.press('cm-1', 'Send again');
        assert.deepEqual(reloaded.sent.map((item) => [item.frame, item.options]), [[frame, { queue: false }]],
            'the same words, id, routing and upload refs');
        assert.equal(reloaded.notes('cm-1').length, 0, 'resent: the doubt waits for the next fact');
        reloaded.echo('cm-1', frame.content);
        assert.deepEqual(reloaded.notes('cm-1').map((node) => node.textContent), ['Input saved']);
        assert.equal(globalThis.sessionStorage.getItem(roomKey('ouro_chat_unconfirmed', 7)), null, 'the saved row ends the kept copy');
        assert.equal(reloaded.bubbles('cm-1'), 1, 'the saved echo settles the restored bubble, never a second one');
        assert.equal(reloaded.instance.hasPendingWork(), false);
    } finally { tab.close(); }
});

test('the first read settles dispatch and keeps a pending row until its confirmation', async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('saved');
        await first.send('pending');
        const content = first.sent[1].frame.content;
        first.instance.destroy();
        tab.host.history = [tab.row('cm-1', first.sent[0].frame.content),
            tab.row('cm-2', content, { ingress_pending: true })];

        const reloaded = tab.open();
        await until(() => reloaded.notes('cm-2').length === 1, 'the pending saved row replays');
        for (const id of ['cm-1', 'cm-2']) {
            assert.deepEqual(reloaded.notes(id).map((node) => node.textContent), ['Input saved']);
            assert.deepEqual(reloaded.actions(id), []);
        }
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-2']);
        assert.deepEqual(reloaded.sent, [], 'pending dispatch does not resend');
        reloaded.echo('cm-2', content);
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), ['Input saved']);
        assert.deepEqual(tab.kept(), []);
        assert.deepEqual([reloaded.bubbles('cm-1'), reloaded.bubbles('cm-2')], [1, 1], 'history rows, no restored duplicates');
        reloaded.fire('chat', { type: 'chat', role: 'user', chat_id: 7, client_message_id: 'cm-2', ingress_accepted: true,
            ingress_pending: true, content, ts: TS, source: 'web', sender_session_id: ME });
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), ['Input saved'], 'Input saved is final');
    } finally { tab.close(); }
});

// A reload whose first read fails shows the kept copy in doubt when the core says, on the socket serving this page's
// profile, that the room is still one of that profile's ordinary rooms. Discard forgets the tab's copy alone. Send again,
// the owner's own press, asks the core anew: a read begun by the press, no frame before its answer, then the SAME frame
// and id once; a second click meanwhile asks nothing more. Nothing goes by itself.
test('in a Project room the core still counts, unreadable history is no proof either way: Discard forgets only the kept copy, Send again asks anew and keeps its id', async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('maybe');
        await first.send('again');
        const resent = first.sent[1].frame;
        first.instance.destroy();
        tab.host.fails = true;
        const reloaded = tab.open();
        await until(() => reloaded.actions('cm-2').length === 2, 'the kept bubbles are in doubt, not declared unsaved');
        for (const id of ['cm-1', 'cm-2']) {
            assert.ok(reloaded.notes(id)[0].textContent.startsWith('Not confirmed as saved.'), 'never "not saved"');
            assert.deepEqual(reloaded.actions(id).map((node) => node.textContent), ['Send again', 'Discard']);
        }
        assert.equal(reloaded.sent.length, 0, 'nothing resends by itself');
        reloaded.press('cm-1', 'Discard');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-2'], 'only that copy is forgotten');
        assert.equal(reloaded.sent.length, 0);
        assert.deepEqual(tab.host.calls.filter(([, method]) => method === 'DELETE'), [], 'no upload is deleted');
        assert.equal(reloaded.actions('cm-1').length, 0);
        assert.equal(reloaded.notes('cm-1').length, 1, 'the bubble still says what is known');
        const reads = () => tab.host.calls.filter(([url]) => url.startsWith('/api/chat/history')).length;
        const before = reads();
        let answer;
        tab.host.gate = new Promise((resolve) => { answer = resolve; });
        reloaded.press('cm-2', 'Send again');
        reloaded.press('cm-2', 'Send again');
        await until(() => reads() > before, 'the press began a read of its own');
        await settle();
        assert.equal(reads(), before + 1, 'one ask, whatever the clicks');
        assert.deepEqual(reloaded.sent, [], 'no frame before the core answers');
        answer();
        await until(() => reloaded.sent.length > 0, 'the fresh word still counts the room: the frame goes');
        await settle();
        assert.deepEqual(reloaded.sent.map((item) => [item.frame, item.options]), [[{ ...resent, ordinary_room: 7 }, { queue: false }]],
            'once, the same words, id, routing and upload refs, naming the room the word stood on for the core to ask again');
        assert.ok(tab.kept().every((entry) => !('ordinary_room' in entry.frame)), 'the kept copy stays as it was');
        assert.equal(reloaded.notes('cm-2').length, 0, 'resent: the doubt waits for the next fact');
        reloaded.echo('cm-2', resent.content);
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), ['Input saved']);
        assert.deepEqual(tab.kept(), [], 'the saved row ends the kept copy');
        assert.equal(reloaded.instance.hasPendingWork(), false);
        reloaded.fire('close');
        assert.equal(reloaded.actions('cm-1').length, 0, 'a discarded message is not offered again');
    } finally { tab.close(); }
});

// Main the same way: the core counts it among the profile's ordinary rooms while the registry tells every room apart.
test('in Main the core still counts, an unreadable first read shows the kept copy in doubt', async () => {
    const tab = keptTab({ chatId: 1 });
    try {
        const first = tab.open();
        await first.send('main words');
        first.instance.destroy();
        tab.host.fails = true;
        const reloaded = tab.open();
        await until(() => reloaded.actions('cm-1').length === 2, 'Main\'s kept copy is in doubt');
        assert.ok(reloaded.notes('cm-1')[0].textContent.startsWith('Not confirmed as saved.'));
        assert.deepEqual(reloaded.actions('cm-1').map((node) => node.textContent), ['Send again', 'Discard']);
        assert.equal(reloaded.sent.length, 0, 'nothing resends by itself');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1']);
    } finally { tab.close(); }
});

// Shown, the copy stands only while the core's word does. The socket that word was read on closes: the copy leaves the
// feed, kept. The next socket's own read brings the word and the copy back. The registry's word that the room changed
// lapses it again, and a room the core no longer counts stays unshown. Nothing is sent by itself throughout.
test('a copy shown on the core\'s word leaves the feed, kept, when that word lapses, and returns with a new one', async () => {
    const tab = keptTab();
    const kept = () => tab.kept().map((entry) => entry.frame.client_message_id);
    try {
        const first = tab.open();
        await first.send('later');
        first.instance.destroy();
        tab.host.fails = true;
        const room = tab.open();
        await until(() => room.actions('cm-1').length === 2, 'the core\'s word: the kept copy is in doubt');

        tab.host.socket = null;  // the socket the word was read on closes
        room.fire('close');
        assert.equal(room.bubble('cm-1'), undefined, 'the word lapsed with its socket: the copy left the feed');
        assert.deepEqual(kept(), ['cm-1'], 'and stays kept');

        tab.host.socket = { readyState: 1 };  // a new socket of the same profile, and its own read
        room.fire('open', { previouslyConnected: true });
        await until(() => room.actions('cm-1').length === 2, 'the new word brings the copy back');
        assert.equal(room.bubbles('cm-1'), 1, 'one bubble, drawn again');

        tab.host.ordinary = false;  // the registry let the room go
        room.fire('projects_changed', { type: 'projects_changed', project_id: 'p7', chat_id: 7 });
        assert.equal(room.bubble('cm-1'), undefined, 'the registry\'s word on the room lapses what was shown');
        await settle();
        assert.equal(room.bubble('cm-1'), undefined, 'the room the core no longer counts stays unshown');
        assert.deepEqual(kept(), ['cm-1']);
        assert.deepEqual(room.sent, [], 'nothing was sent by itself');
    } finally { tab.close(); }
});

// A reload's first read can fail before its socket named the profile it serves: the core's word then waits, the copy
// kept and unshown, and the socket's handshake asks for it again on that socket.
test('a word read before this socket proved the profile is asked again on its handshake', async () => {
    const tab = keptTab();
    const reads = () => tab.host.calls.filter(([url]) => url.startsWith('/api/chat/history')).length;
    try {
        const first = tab.open();
        await first.send('early');
        first.instance.destroy();
        Object.assign(tab.host, { fails: true, profile: false });
        const before = reads();
        const room = tab.open();
        await until(() => reads() > before, 'the first read');
        await settle();
        assert.equal(room.bubble('cm-1'), undefined, 'no socket has proved the profile: the copy stays unshown');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1'], 'and kept');
        tab.host.profile = true;
        room.fire('connection_profile', { type: 'connection_profile', profile_scope: PREVIEW_SCOPE });
        await until(() => room.actions('cm-1').length === 2, 'asked again on the socket that proved it: shown');
        assert.ok(reads() > before + 1, 'by a read of its own');
        assert.deepEqual(room.sent, [], 'nothing was sent by itself');
    } finally { tab.close(); }
});

// The core's state can change with nothing broadcast. A press asks it afresh and sends nothing unless that answer still
// counts the room: here the core stops counting it, and then, the room reopened, its history reads again.
test('a press sends nothing when the fresh answer no longer counts the room, or when history reads again', async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('maybe');
        first.instance.destroy();
        tab.host.fails = true;
        let room = tab.open();
        await until(() => room.actions('cm-1').length === 2, 'shown on the core\'s word');
        tab.host.ordinary = false;  // the core stops counting the room; nothing is broadcast
        room.press('cm-1', 'Send again');
        await until(() => room.bubble('cm-1') === undefined, 'the fresh answer ends the word: the copy leaves the feed');
        assert.deepEqual(room.sent, [], 'and nothing was sent');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1'], 'the copy stays kept');
        room.instance.destroy();

        tab.host.ordinary = true;
        room = tab.open();
        await until(() => room.actions('cm-1').length === 2, 'shown again on a new word');
        tab.host.fails = false;  // history reads again; nothing is broadcast
        room.press('cm-1', 'Send again');
        await until(() => room.instance.hasPaintedHistory(), 'the ordinary read takes over');
        await settle();
        assert.deepEqual(room.sent, [], 'the press itself sent nothing');
    } finally { tab.close(); }
});

// Leaving the room lapses the core's word: what it showed leaves the feed, kept, and no answer asked before the leave
// brings it back. The return shows it again only on the word its own read brings.
test('leaving the room takes a copy shown on the core\'s word off the feed; the return shows it only on a fresh word', async () => {
    const tab = keptTab();
    const reads = () => tab.host.calls.filter(([url]) => url.startsWith('/api/chat/history')).length;
    try {
        const first = tab.open();
        await first.send('later');
        first.instance.destroy();
        tab.host.fails = true;
        const room = tab.open();
        await until(() => room.actions('cm-1').length === 2, 'shown on the core\'s word');
        room.instance.leave();
        assert.equal(room.bubble('cm-1'), undefined, 'leaving lapses the word: the copy left the feed');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1'], 'and stays kept');
        let answer;
        tab.host.gate = new Promise((resolve) => { answer = resolve; });
        const before = reads();
        room.instance.stay();
        room.fire('open', { previouslyConnected: true, resumed: true });
        await until(() => reads() > before, 'the return reads anew');
        await settle();
        assert.equal(room.bubble('cm-1'), undefined, 'nothing shows before that read answers');
        answer();
        await until(() => room.actions('cm-1').length === 2, 'the fresh word shows it again');
        assert.equal(room.bubbles('cm-1'), 1, 'one bubble, drawn again');
        assert.deepEqual(room.sent, [], 'nothing was sent by itself');
    } finally { tab.close(); }
});

// Astra 8: Send again's own read is one of the room's reads. A current refusal for authentication takes this browser's
// saved text and every copy painted from it, as the room's other reads' does; any other failure, or an answer that
// comes after the room left, leaves both.
for (const [status, leaves, takes] of [[401, false, true], [403, false, true], [500, false, false], [401, true, false]]) {
    test(`Send again's read answered ${status}${leaves ? ' after the room left' : ''} ${takes ? 'takes' : 'leaves'} the saved text`, async () => {
        const tab = keptTab();
        const family = () => [...localStorage.values.keys()].filter((key) => key.startsWith(PREVIEW_FAMILY));
        try {
            const first = tab.open();
            await first.send('maybe');
            first.instance.destroy();
            localStorage.setItem(PREVIEW_KEY, '{"v":3}');
            localStorage.setItem(previewEntryKey(PREVIEW_SCOPE, [7, '']), JSON.stringify({ read: 1, rows: [
                { id: 'chat:1', role: 'assistant', text: 'Saved words', speaker: 'Ouroboros' }] }));
            Object.assign(tab.host, { fails: true, previews: true });
            const room = tab.open();
            await until(() => room.actions('cm-1').length === 2, 'shown on the core\'s word');
            assert.equal(room.preview()?.children[0].textContent, 'Saved text preview · not current', 'painted, read failing');
            const stored = family();
            let answer;
            tab.host.gate = new Promise((resolve) => { answer = resolve; });
            tab.host.status = status;
            room.press('cm-1', 'Send again');
            await settle();
            if (leaves) room.instance.leave();
            answer();
            await settle();
            assert.deepEqual(family(), takes ? [] : stored);
            assert.equal(Boolean(room.preview()), !takes, takes ? 'the painted copy goes at once' : 'the copy stays');
            assert.deepEqual(room.sent, [], 'nothing was sent');
        } finally { tab.close(); }
    });
}

// Astra 4: Send again's read belongs to the visit and the socket it was asked on. Its refusal, landing once the room has
// left and come back (its return read the room and saved its newest text), or once that socket gave way, takes nothing.
for (const after of ['return', 'socket']) {
    test(`Send again's read refused after ${after === 'return' ? 'the room left and came back' : 'its socket gave way'} takes no saved text`, async () => {
        const tab = keptTab();
        const family = () => [...localStorage.values].filter(([key]) => key.startsWith(PREVIEW_FAMILY));
        try {
            const first = tab.open();
            await first.send('maybe');
            first.instance.destroy();
            localStorage.setItem(PREVIEW_KEY, '{"v":3}');
            localStorage.setItem(previewEntryKey(PREVIEW_SCOPE, [7, '']), JSON.stringify({ read: 1, rows: [
                { id: 'chat:1', role: 'assistant', text: 'Saved words', speaker: 'Ouroboros' }] }));
            Object.assign(tab.host, { fails: true, previews: true });
            const room = tab.open();
            await until(() => room.actions('cm-1').length === 2, 'shown on the core\'s word');
            // The press's own read is held, and refused when released; every other read answers as the host does.
            let refuse = null;
            const host = globalThis.fetch;
            globalThis.fetch = (url, init) => (!refuse && String(url).startsWith('/api/chat/history') && String(url).includes('reason=retry')
                ? new Promise((resolve) => { refuse = () => resolve({ ok: false, status: 401, json: async () => ({ error: 'Refused' }) }); })
                : host(url, init));
            room.press('cm-1', 'Send again');
            await until(() => refuse, 'the press asks the core');
            if (after === 'return') {
                room.instance.leave();
                Object.assign(tab.host, { fails: false, history: [{ role: 'assistant', text: 'Saved on the return', ts: TS,
                    chat_id: 7, history_id: 'chat:2' }] });
                room.instance.stay();
                room.fire('open', { previouslyConnected: true, resumed: true });
                await until(() => room.instance.hasPaintedHistory(), 'the return reads the room');
                assert.match(JSON.stringify(family()), /Saved on the return/, 'and saves its newest text');
            } else tab.host.socket = { readyState: 1 };  // the socket the press asked on gives way to another
            const saved = family();
            refuse();
            await settle();
            assert.deepEqual(family(), saved, 'the late refusal takes nothing');
            assert.deepEqual(room.sent, [], 'nothing was sent');
        } finally { tab.close(); }
    });
}

// Each read a kept send makes says why (history_attribution.py): Send again's and its hand-over to the ordinary read
// are a retry, as is a failed newest read asked again on the socket that serves the profile now (its proof, or a
// socket that changed under it); the registry's word on the room is a refresh.
test('the reads the kept sends make name their action', async () => {
    const tab = keptTab();
    let since = 0;  // the reads of the room under test, after the earlier tab's
    const reads = () => tab.host.calls.filter(([url]) => url.startsWith('/api/chat/history')).slice(since)
        .map(([url]) => new URL(url, 'http://local').searchParams.get('reason'));
    try {
        const first = tab.open();
        await first.send('maybe');
        first.instance.destroy();
        Object.assign(tab.host, { fails: true, profile: false });
        since = reads().length;
        const room = tab.open();
        await until(() => reads().length === 1, 'the opening read');
        await settle();
        tab.host.profile = true;
        room.fire('connection_profile', { type: 'connection_profile', profile_scope: PREVIEW_SCOPE });
        await until(() => room.actions('cm-1').length === 2, 'asked again on the proving socket: shown');
        assert.deepEqual(reads(), ['open', 'retry']);
        room.fire('projects_changed', { type: 'projects_changed', chat_id: 7 });
        await until(() => room.actions('cm-1').length === 2 && reads().length === 3, 'the registry\'s word, asked again');
        assert.equal(reads()[2], 'refresh');
        let answer;
        tab.host.gate = new Promise((resolve) => { answer = resolve; });
        room.fire('projects_changed', { type: 'projects_changed', chat_id: 7 });
        await until(() => reads().length === 4, 'asked again');
        tab.host.socket = { readyState: 1 };  // the socket changes under the read: its answer is asked again on the new one
        tab.host.gate = null;
        answer();
        await until(() => reads().length === 5 && room.actions('cm-1').length === 2, 'asked again on this socket: shown');
        assert.equal(reads()[4], 'retry');
        tab.host.fails = false;  // history reads again
        room.press('cm-1', 'Send again');
        await until(() => room.instance.hasPaintedHistory(), 'the ordinary read takes over');
        assert.deepEqual(reads().slice(5), ['retry', 'retry'], 'Send again\'s read and its hand-over to the ordinary read');
    } finally { tab.close(); }
});

// ouro:panic reaches the page's own work through its global event bus (chat_activity.js raises it once its Panic frame
// is sent); the DOM fixture has none, so a case installs one. Nothing here sends a Panic.
function panicBus() {
    const keys = ['addEventListener', 'removeEventListener', 'dispatchEvent'];
    const prior = keys.map((key) => Object.getOwnPropertyDescriptor(globalThis, key));
    const bus = new EventTarget();
    for (const key of keys) globalThis[key] = bus[key].bind(bus);
    return () => keys.forEach((key, i) => (prior[i] ? Object.defineProperty(globalThis, key, prior[i]) : delete globalThis[key]));
}

// A Panic ends a Send again still asking the core, as it ends the composer's send (composer.js): whatever the core then
// answers, nothing goes, and only a press made after it asks anew.
test('a Panic ends a Send again still asking the core; a press after it is a new one', async () => {
    const restoreBus = panicBus();
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('maybe');
        const resent = first.sent[0].frame;
        first.instance.destroy();
        tab.host.fails = true;
        const room = tab.open();
        await until(() => room.actions('cm-1').length === 2, 'shown on the core\'s word');
        let answer;
        tab.host.gate = new Promise((resolve) => { answer = resolve; });
        room.press('cm-1', 'Send again');
        globalThis.dispatchEvent(new CustomEvent('ouro:panic'));
        answer();
        await settle();
        assert.deepEqual(room.sent, [], 'the press made before the Panic sends nothing, though the core still counts the room');
        assert.deepEqual(room.actions('cm-1').map((node) => node.textContent), ['Send again', 'Discard'], 'the copy still waits');
        tab.host.gate = null;
        room.press('cm-1', 'Send again');
        await until(() => room.sent.length > 0, 'a press after the Panic asks anew and sends');
        assert.deepEqual(room.sent.map((item) => [item.frame, item.options]), [[{ ...resent, ordinary_room: 7 }, { queue: false }]]);
    } finally { tab.close(); restoreBus(); }
});

// The core's acceptance refuses a Send again whose room it no longer confirms by a typed cause (gateway/ws.py): that
// copy loses its word and leaves the feed, kept, until a fresh word shows it again. The same notice without that cause
// leaves it offered as before.
test('a refusal typed as the room\'s lapsed word takes that copy off the feed, kept, until a fresh word', async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('later');
        first.instance.destroy();
        tab.host.fails = true;
        const room = tab.open();
        await until(() => room.actions('cm-1').length === 2, 'shown on the core\'s word');
        const refuse = (extra = {}) => room.fire('chat', { type: 'chat', role: 'system', system_type: 'initialization_notice',
            chat_id: 7, client_message_id: 'cm-1', content: 'Not saved.', ...extra });
        refuse();
        assert.deepEqual(room.actions('cm-1').map((node) => node.textContent), ['Send again', 'Discard'], 'another refusal keeps it offered');
        refuse({ reason_code: 'ordinary_room_unconfirmed' });
        assert.equal(room.bubble('cm-1'), undefined, 'its word lapsed: the copy left the feed');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1'], 'and stays kept');
        room.fire('open', { previouslyConnected: true });
        await until(() => room.actions('cm-1').length === 2, 'a fresh word shows it again');
        assert.deepEqual(room.sent, [], 'nothing was sent by itself');
    } finally { tab.close(); }
});

test('only the core\'s word that this room is still one of the page profile\'s ordinary rooms is an ordinary unavailability', () => {
    const answer = (body, status = 503) => Object.assign(new Error(`HTTP ${status}`), { status, body });
    const word = { reason_code: 'history_source_unavailable', ordinary_room: 7, profile_scope: PREVIEW_SCOPE };
    assert.equal(ordinaryUnavailability(answer(word), 7, PREVIEW_SCOPE), true, 'the source cannot be read now');
    assert.equal(ordinaryUnavailability(answer({ ...word, reason_code: 'migrating' }), 7, PREVIEW_SCOPE), true, 'still importing');
    for (const [why, error, chatId = 7, scope = PREVIEW_SCOPE] of [
        ['no word on the room', answer({ reason_code: 'history_source_unavailable' })],
        ['a word on another room', answer(word), 8],
        ['Main\'s archive offer', answer({ reason_code: 'history_source_unavailable', project_registry: 'unavailable' }), 1],
        ['no answer at all', new TypeError('Failed to fetch')],
        ['a page that proves no profile', answer(word), 7, ''],
    ]) assert.equal(ordinaryUnavailability(error, chatId, scope), false, why);
});

// A message sent as new work on a model choice that changed elsewhere was not saved (`model_choice_changed`). Its copy
// stops naming that choice, so no Send again repeats the refused intent; plain words return to a free composer.
test('a copy refused because the model changed elsewhere never names that choice again', () => {
    const env = installDom();
    try {
        const root = document.createElement('div');
        root.isConnected = true;
        const copy = (id) => {
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble user';
            bubble.dataset.clientMessageId = id;
            root.appendChild(bubble);
            return bubble;
        };
        const kept = new Map();
        const sends = [];
        const composed = [];
        const free = { value: true };
        const unconfirmed = createUnconfirmedSends({ send: (frame) => { sends.push(frame); return { status: 'sent' }; },
            root: () => root, onDomWrite: (fn) => fn(), showToast: () => {},
            compose: (words, quote, options) => { composed.push([words, quote, options]); return free.value; },
            storage: { getItem: (key) => kept.get(key) ?? null, setItem: (key, value) => kept.set(key, value), removeItem: (key) => kept.delete(key) },
            storageKey: 'k' });
        const refuse = (id, extra = { reason_code: 'model_choice_changed' }) => unconfirmed.refused({ client_message_id: id, ...extra });
        const keptFrames = () => JSON.parse(kept.get('k') || '[]').map((entry) => entry.frame);
        const actions = (bubble) => (bubble.children[0]?.children || []).map((node) => node.textContent).filter(Boolean);

        // Plain words go back to a free composer, unasked and quietly, and their copy leaves the feed and the tab.
        const plain = copy('cm-1');
        unconfirmed.track({ type: 'chat', content: 'compare the drafts', client_message_id: 'cm-1', chat_id: 7, new_work: 'token-sol' });
        assert.equal(keptFrames()[0].new_work, 'token-sol', 'a reload keeps the choice an unanswered frame named');
        refuse('cm-1');
        assert.deepEqual(composed, [['compare the drafts', null, { quiet: true }]]);
        assert.deepEqual([root.children.includes(plain), unconfirmed.count, keptFrames()], [false, 0, []]);

        // Files stay with their copy: it is offered as before, and its Send again is an ordinary message of the same id.
        const files = copy('cm-2');
        unconfirmed.track({ type: 'chat', content: 'read this', client_message_id: 'cm-2', chat_id: 7, new_work: 'token-sol',
            attachments: [{ filename: 'f', display_name: 'n', mime: 'm' }] });
        refuse('cm-2');
        assert.deepEqual([composed.length, root.children.includes(files), actions(files)], [1, true, ['Send again', 'Discard']]);
        assert.equal('new_work' in keptFrames()[0], false, 'the kept copy no longer names the choice either');
        files.children[0].children.find((node) => node.textContent === 'Send again').click();
        assert.deepEqual(sends, [{ type: 'chat', content: 'read this', client_message_id: 'cm-2', chat_id: 7,
            attachments: [{ filename: 'f', display_name: 'n', mime: 'm' }] }]);

        // A composer that holds another draft takes nothing: the copy stays, without the choice.
        free.value = false;
        const waiting = copy('cm-3');
        unconfirmed.track({ type: 'chat', content: 'later words', client_message_id: 'cm-3', chat_id: 7, new_work: 'token-sol' });
        refuse('cm-3');
        assert.deepEqual([composed.length, root.children.includes(waiting), actions(waiting)], [2, true, ['Send again', 'Discard']]);
        waiting.children[0].children.find((node) => node.textContent === 'Send again').click();
        assert.deepEqual(sends.at(-1), { type: 'chat', content: 'later words', client_message_id: 'cm-3', chat_id: 7 });

        // Any other refusal leaves the frame as it was sent: the host may still judge that same choice.
        free.value = true;
        const busy = copy('cm-4');
        unconfirmed.track({ type: 'chat', content: 'again', client_message_id: 'cm-4', chat_id: 7, new_work: 'token-sol' });
        refuse('cm-4', {});
        assert.deepEqual([composed.length, root.children.includes(busy)], [2, true]);
        assert.equal(keptFrames().find((frame) => frame.client_message_id === 'cm-4').new_work, 'token-sol');
        unconfirmed.release();
    } finally {
        restoreDom(env.prior);
    }
});

test('a tab that cannot keep the copy says so, and Send again still works until a reload', () => {
    const env = installDom();
    try {
        const root = document.createElement('div');
        root.isConnected = true;
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble user';
        bubble.dataset.clientMessageId = 'cm-5';
        root.appendChild(bubble);
        const toasts = [];
        const sends = [];
        const storage = { getItem: () => 'not json', setItem() { throw new Error('QuotaExceededError'); }, removeItem() {} };
        const unconfirmed = createUnconfirmedSends({ send: (frame) => { sends.push(frame); return { status: 'sent' }; },
            root: () => root, onDomWrite: (fn) => fn(), showToast: (text, tone, options) => toasts.push([text, tone, options?.haptic]),
            storage, storageKey: 'k' });
        assert.equal(unconfirmed.count, 0, 'an unreadable copy restores nothing');
        assert.deepEqual(toasts, [[UNREAD, 'error', false]], 'and says so, without the error haptic: no action of the owner failed');
        unconfirmed.track({ type: 'chat', content: 'x', client_message_id: 'cm-5' });
        assert.deepEqual(toasts.map(([, tone, haptic]) => [tone, haptic]), [['error', false], ['error', false]]);
        assert.ok(toasts[1][0].includes('could not keep'));
        unconfirmed.unsettle();
        bubble.children[0].children.find((node) => node.textContent === 'Send again').click();
        assert.deepEqual(sends, [{ type: 'chat', content: 'x', client_message_id: 'cm-5' }]);
        unconfirmed.release();
        assert.equal(unconfirmed.count, 0);

        // What a working tab keeps: words, id, routing and upload refs — no surface, no bytes, no retired one-shot field.
        const kept = new Map();
        const keeping = createUnconfirmedSends({ send: () => ({ status: 'sent' }), root: () => root, onDomWrite: (fn) => fn(),
            showToast: () => {}, storage: { getItem: (key) => kept.get(key) ?? null, setItem: (key, value) => kept.set(key, value),
                removeItem: (key) => kept.delete(key) }, storageKey: 'k' });
        keeping.track({ type: 'chat', content: 'y', client_message_id: 'cm-6', chat_id: 7, force_plan: false,
            client_surface: { ua: 'agent' }, image_base64: 'AAAA', attachments: [{ filename: 'f', display_name: 'n', mime: 'm', path: '/x' }] });
        assert.deepEqual(JSON.parse(kept.get('k'))[0].frame, { type: 'chat', content: 'y', client_message_id: 'cm-6', chat_id: 7,
            attachments: [{ filename: 'f', display_name: 'n', mime: 'm' }] });
        keeping.release();
        assert.equal(kept.size, 1, 'teardown keeps the copy for the next instance');
    } finally {
        restoreDom(env.prior);
    }
});

// The first read after a reload answers for what this tab kept, not for a message this page sent
// while that read was still pending (on a slow timer the read lands after the send): the new one
// waits for its echo, a close or a refusal, like any other sent message.
test('the first read doubts the kept copy, never a message sent while it was pending', () => {
    const env = installDom();
    try {
        const root = document.createElement('div');
        const bubbles = Object.fromEntries(['cm-kept', 'cm-new'].map((id) => {
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble user';
            bubble.dataset.clientMessageId = id;
            root.appendChild(bubble);
            return [id, bubble];
        }));
        const doubts = (id) => bubbles[id].children.filter((node) => Object.hasOwn(node.dataset, 'ingressUnconfirmed'));
        const kept = JSON.stringify([{ frame: { type: 'chat', content: 'kept', client_message_id: 'cm-kept' }, views: [], ts: TS }]);
        const unconfirmed = createUnconfirmedSends({ send: () => ({ status: 'sent' }), root: () => root, onDomWrite: (fn) => fn(),
            showToast: () => {}, storage: { getItem: () => kept, setItem() {}, removeItem() {} }, storageKey: 'k' });
        unconfirmed.track({ type: 'chat', content: 'new', client_message_id: 'cm-new' });
        const shown = [];
        unconfirmed.reconcile((entry) => shown.push(entry));
        assert.deepEqual(shown, [], 'both bubbles are in the feed');
        assert.equal(doubts('cm-kept').length, 1, 'the kept copy the read did not settle is in doubt');
        assert.equal(doubts('cm-new').length, 0, 'the message sent during the read is not');
        assert.equal(unconfirmed.count, 2, 'and both frames stay until their saved rows');
        unconfirmed.unsettle();  // the socket closes
        assert.equal(doubts('cm-new').length, 1, 'a close still doubts the new one');
        assert.equal(doubts('cm-kept').length, 1, 'the kept one keeps its one doubt');
    } finally {
        restoreDom(env.prior);
    }
});

const UNREAD = 'This tab could not read an unsaved message it kept for a reload, so it cannot be offered again.';

test('a kept copy this tab cannot read is said, whichever way the read fails; a readable one is restored', () => {
    const env = installDom();
    try {
        const frame = { type: 'chat', content: 'kept', client_message_id: 'cm-k', attachments: [] };
        const open = (getItem) => {
            const toasts = [];
            const sends = createUnconfirmedSends({ send: () => ({ status: 'sent' }), root: () => null, onDomWrite: (fn) => fn(),
                showToast: (text, tone) => toasts.push([text, tone]), storage: { getItem, setItem() {}, removeItem() {} }, storageKey: 'k' });
            return { sends, toasts };
        };
        for (const [why, getItem] of [
            ['storage refused the read', () => { throw new Error('SecurityError'); }],
            ['not JSON', () => '{"frame":'],
            ['not a list', () => '{}'],
            ['no frame in it', () => JSON.stringify([{ views: [] }])],
        ]) {
            const { sends, toasts } = open(getItem);
            assert.equal(sends.count, 0, why);
            assert.deepEqual(toasts, [[UNREAD, 'error']], why);
        }
        const partly = open(() => JSON.stringify([{ frame, views: [], ts: TS }, { frame: { type: 'chat' } }]));
        assert.equal(partly.sends.count, 1, 'the readable copy is restored');
        assert.deepEqual(partly.toasts, [[UNREAD, 'error']], 'the unreadable one beside it is still said');
        const none = open(() => null);
        assert.deepEqual([none.sends.count, none.toasts], [0, []], 'nothing kept is nothing to say');
        const whole = open(() => JSON.stringify([{ frame, views: [], ts: TS }]));
        assert.deepEqual([whole.sends.count, whole.toasts], [1, []]);
    } finally {
        restoreDom(env.prior);
    }
});

// A host restart ends the process that knew what became of a saved row (its `ingress_dispatched`). A kept send that meets such a row says it is saved with its delivery
// unconfirmed and keeps no Send again — the host would only rejoin it — and nothing is resent. Rows nobody kept are
// history and keep their plain `Input saved`.
test('after a host restart a kept send meets its saved row: delivery not confirmed, nothing replays', async () => {
    const tab = keptTab();
    const DOUBT = 'Saved; delivery not confirmed.';
    try {
        const first = tab.open();
        await first.send('reload');
        await first.send('live');
        await first.send('pending');
        const [reloadWords, liveWords, pendingWords] = first.sent.map((item) => item.frame.content);
        first.instance.destroy();
        // The third saved row still awaits dispatch; a later restart will remove the live fact.
        tab.host.history = [tab.row('cm-1', reloadWords, { restarted: true }), tab.row('cm-3', pendingWords, { ingress_pending: true })];

        const reloaded = tab.open();
        await until(() => reloaded.notes('cm-1')[0]?.textContent === DOUBT, 'the first read meets a row of the ended process');
        assert.deepEqual(reloaded.retries('cm-1'), [], 'no Send again: it could only rejoin');
        assert.deepEqual(reloaded.asNew('cm-1').map((node) => node.textContent), ['Send as new'], 'the one honest next step is a new message');
        assert.deepEqual(reloaded.asNew('cm-1').map((node) => node.getAttribute('aria-label')), ['Send as a new message'],
            'short words on a phone row, the whole name for a screen reader');
        assert.equal(reloaded.notes('cm-1').length, 1, 'one note: the doubt replaced Input saved');
        assert.deepEqual(reloaded.notes('cm-3')[0].textContent, 'Input saved');
        assert.deepEqual(reloaded.actions('cm-3'), []);
        assert.equal(reloaded.actions('cm-2').length, 2, 'the one the host never saw is still in doubt with its actions');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1', 'cm-2', 'cm-3']);
        assert.equal(tab.kept()[0].deliveryUnknown, true);

        // The pending row after a restart: the host no longer knows; the doubt replaces Input saved.
        tab.host.history = [tab.row('cm-1', reloadWords, { restarted: true }), tab.row('cm-3', pendingWords, { restarted: true })];
        await reloaded.instance.refreshHistory({ revision: 2 });
        assert.deepEqual(reloaded.notes('cm-3').map((node) => node.textContent), [DOUBT]);
        assert.deepEqual(reloaded.retries('cm-3'), []);
        assert.equal(reloaded.asNew('cm-3').length, 1, 'the doubt keeps its one action');
        assert.deepEqual(reloaded.notes('cm-1').map((node) => node.textContent), [DOUBT], 'the doubt is final for that read');

        // Live: Send again reaches the restarted host, which rejoins the row an ended process saved.
        reloaded.press('cm-2', 'Send again');
        reloaded.echo('cm-2', liveWords, { restarted: true });
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), [DOUBT]);
        assert.deepEqual(reloaded.retries('cm-2'), []);
        reloaded.fire('close');
        assert.deepEqual([reloaded.retries('cm-1'), reloaded.retries('cm-2'), reloaded.retries('cm-3')], [[], [], []],
            'a socket close offers none of them again');
        assert.deepEqual(reloaded.sent.map((item) => item.frame.client_message_id), ['cm-2'], 'only the one manual Send again');
        // The doubt's own action prepares the composer and sends nothing: the owner reads, edits and sends it, or does not.
        reloaded.press('cm-1', 'Send as new');
        assert.equal(document.activeElement, document.byId.get('chat-input'), 'the composer is focused');
        assert.deepEqual(reloaded.sent.map((item) => item.frame.client_message_id), ['cm-2'], 'preparing the composer sends nothing');
        assert.ok(tab.kept().every(entry => entry.deliveryUnknown), 'the tab keeps all saved delivery doubts');
        assert.equal(reloaded.instance.hasPendingWork(), false);
        reloaded.instance.destroy();

        // The next read keeps the warning and recovery action, including for rows absent from this history page.
        tab.host.history = [tab.row('cm-1', reloadWords, { restarted: true })];
        const later = tab.open();
        await until(() => later.asNew('cm-1').length === 1, 'history replays the kept doubt');
        assert.deepEqual(later.notes('cm-1').map((node) => node.textContent), [DOUBT]);
        assert.deepEqual(later.retries('cm-1'), []);
        later.echo('cm-1', reloadWords);
        assert.deepEqual(tab.kept().map(entry => entry.frame.client_message_id), ['cm-2', 'cm-3']);
        assert.deepEqual(later.asNew('cm-1'), [], 'Send as new leaves the bubble with the warning, not only the storage');
        assert.deepEqual(later.notes('cm-1').map((node) => node.textContent), ['Input saved'], 'the saved tick stands in its place');
    } finally { tab.close(); }
});

// The running host can be read between its append and its dispatch (supervisor/message_bus.py writes the row,
// then enters dispatch; a deferred dispatch is echoed first): that row says `ingress_pending`. It is no delivery
// doubt: the kept frame waits for confirmation of dispatch. A row of a process that has since ended
// (neither dispatch fact) ends it in the doubt.
test('a saved row the running host has not yet dispatched keeps its frame for the next fact', async () => {
    const tab = keptTab();
    const pending = { restarted: true, ingress_pending: true };
    try {
        const room = tab.open();
        await room.send('during the append');
        const first = room.sent[0].frame.content;
        tab.host.history = [tab.row('cm-1', first, pending)];
        await room.instance.refreshHistory({ revision: 1 });
        assert.deepEqual(room.notes('cm-1').map((node) => node.textContent), ['Input saved'], 'saved, and nothing more is claimed');
        assert.deepEqual(room.actions('cm-1'), [], 'no doubt and no Send again while the host has not said');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1'], 'the frame waits');
        room.echo('cm-1', first);  // the same process entered its dispatch
        assert.deepEqual(room.notes('cm-1').map((node) => node.textContent), ['Input saved']);
        assert.deepEqual(tab.kept(), [], 'positive dispatch evidence settles it');

        await room.send('then dispatch waits');
        const second = room.sent[1].frame.content;
        tab.host.history = [tab.row('cm-1', first), tab.row('cm-2', second, pending)];
        await room.instance.refreshHistory({ revision: 2 });
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-2']);
        tab.host.history = [tab.row('cm-1', first), tab.row('cm-2', second, { ingress_pending: true })];
        await room.instance.refreshHistory({ revision: 3 });
        assert.deepEqual(room.notes('cm-2').map((node) => node.textContent), ['Input saved']);
        assert.deepEqual(room.actions('cm-2'), [], 'another pending read keeps waiting');
        assert.equal(room.instance.hasPendingWork(), true);
        tab.host.history = [tab.row('cm-1', first), tab.row('cm-2', second)];
        await room.instance.refreshHistory({ revision: 4 });
        assert.deepEqual(tab.kept(), [], 'history confirmation settles the pending frame');
        assert.equal(room.sent.length, 2, 'only the two original sends');
        assert.equal(room.instance.hasPendingWork(), false);
    } finally { tab.close(); }
});

// A reload's first read that shows a kept send's row `ingress_pending` has answered for it: saved, the running host
// not yet said. The kept frame waits for that word like a live one — no doubt, no actions, nothing resent.
test('a kept send the first read shows pending waits for the next fact, with no doubt from the reload', async () => {
    const tab = keptTab();
    const pending = { restarted: true, ingress_pending: true };
    try {
        const first = tab.open();
        await first.send('handed over later');
        await first.send('dispatched later');
        await first.send('never seen');
        const [handover, dispatched] = first.sent.map((item) => item.frame.content);
        first.instance.destroy();
        tab.host.history = [tab.row('cm-1', handover, pending), tab.row('cm-2', dispatched, pending)];

        const reloaded = tab.open();
        await until(() => reloaded.actions('cm-3').length === 2, 'the first read is reconciled: the unseen one is in doubt');
        for (const cmid of ['cm-1', 'cm-2']) {
            assert.deepEqual(reloaded.notes(cmid).map((node) => node.textContent), ['Input saved'], 'saved, and nothing more is claimed');
            assert.deepEqual(reloaded.actions(cmid).map((node) => node.textContent), [], 'no Send again or Discard while the running host has not said');
            assert.equal(reloaded.bubbles(cmid), 1, 'the history row, no restored duplicate');
        }
        assert.deepEqual(reloaded.sent, [], 'nothing resends by itself');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1', 'cm-2', 'cm-3'], 'the frames wait');

        tab.host.history = [tab.row('cm-1', handover, { ingress_pending: true }), tab.row('cm-2', dispatched)];
        await reloaded.instance.refreshHistory({ revision: 2 });
        assert.equal(reloaded.notes('cm-1')[0].textContent, 'Input saved');
        assert.deepEqual(reloaded.actions('cm-1'), [], 'still pending, without retry actions');
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), ['Input saved']);
        assert.deepEqual(reloaded.actions('cm-2').map((node) => node.textContent), []);
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1', 'cm-3'], 'dispatched ends its kept copy');
        reloaded.echo('cm-1', handover);
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-3']);
        assert.deepEqual(reloaded.sent, [], 'nothing resent');
    } finally { tab.close(); }
});

test('an echo sent before a deferred dispatch waits; a close offers Send again, a restart ends it in the doubt', async () => {
    const tab = keptTab();
    const DOUBT = 'Saved; delivery not confirmed.';
    try {
        const room = tab.open();
        await room.send('first');
        await room.send('second');
        const [first, second] = room.sent.map((item) => item.frame.content);
        room.echo('cm-1', first, { restarted: true, ingress_pending: true });
        room.echo('cm-2', second, { restarted: true, ingress_pending: true });
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1', 'cm-2']);
        const time = room.bubble('cm-1').querySelector('.msg-time');
        assert.equal(room.notes('cm-1')[0].parentNode, time, 'saved and still pending: the quiet tick');
        room.fire('close');
        assert.deepEqual(room.actions('cm-1').map((node) => node.textContent), ['Send again', 'Discard'],
            'the socket closed before the host said: the owner may ask again');
        const [row] = room.notes('cm-1');
        assert.deepEqual([row.parentNode, row.textContent], [room.bubble('cm-1'), 'Input saved'], 'with actions it is a row of words again');
        assert.equal(room.bubble('cm-1').children.indexOf(row) + 1, room.bubble('cm-1').children.indexOf(time), 'right above the time');
        room.press('cm-1', 'Send again');
        assert.deepEqual(room.actions('cm-1'), [], 'resent: the doubt waits for the next fact');
        assert.equal(room.notes('cm-1')[0].parentNode, time, 'and without its actions the tick again');
        room.echo('cm-1', first);  // the rejoin says the same process dispatched it
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-2']);
        // cm-2's host ended before it said: the next read shows its row from an ended process.
        tab.host.history = [tab.row('cm-1', first), tab.row('cm-2', second, { restarted: true })];
        await room.instance.refreshHistory({ revision: 1 });
        assert.deepEqual(room.notes('cm-2').map((node) => node.textContent), [DOUBT]);
        assert.deepEqual(room.retries('cm-2'), []);
        assert.equal(room.asNew('cm-2').length, 1, 'the doubt keeps its one action');
        assert.deepEqual(tab.kept().map(entry => [entry.frame.client_message_id, entry.deliveryUnknown]), [['cm-2', true]]);
        assert.equal(room.instance.hasPendingWork(), false);
    } finally { tab.close(); }
});

// Only the web composer writes the `[Attached file: …]` tail; skills/telegram/plugin.py passes the owner's caption
// unchanged, so a Telegram row keeps every word of it — live and on replay — while a web row hides its own tail.
test('a Telegram caption that ends like a tail is kept word for word, live and on replay', async () => {
    const upload = `${'c'.repeat(32)}_plan.pdf`;
    const view = { name: 'plan.pdf', kind: 'file', mime: 'application/pdf', size: 4, available: true,
        url: `/api/files/download?upload=${upload}` };
    const caption = 'Keep this note\n\n[Attached file: plan.pdf]';
    const telegram = { role: 'user', chat_id: 1, ts: TS, source: 'skill:telegram', sender_label: 'Telegram (Anton)',
        sender_session_id: '', attachments: [view] };
    const message = (node) => node.innerHTML.match(/<div class="message[^"]*" data-reply-body>([\s\S]*?)<\/div>/)[1];
    const live = fixture();
    try {
        live.echo({ ...telegram, content: caption, client_message_id: 'tg:1' });
        live.echo({ role: 'user', content: 'Look\n\n[Attached file: plan.pdf]', sender_session_id: 'session-other',
            client_message_id: 'web-1', attachments: [view] });
        assert.equal(message(live.bubble('tg:1')), caption, 'live: the Telegram caption, whole');
        assert.equal(message(live.bubble('web-1')), 'Look', "live: the composer's own tail is hidden");
    } finally { live.close(); }
    const replay = fixture([{ ...telegram, text: caption, client_message_id: 'tg:1', history_id: 'h-tg-1' },
        { role: 'user', text: 'Look\n\n[Attached file: plan.pdf]', ts: TS, chat_id: 1, source: 'web', sender_session_id: ME,
            client_message_id: 'web-1', history_id: 'h-web-1', attachments: [view] }]);
    try {
        await replay.instance.refreshHistory({ revision: 1 });
        assert.equal(message(replay.bubble('tg:1')), caption, 'replay: the Telegram caption, whole');
        assert.equal(message(replay.bubble('web-1')), 'Look');
    } finally { replay.close(); }
});

// ws.py and server_control.dispatch_accepted_restart recognize only the exact `/restart`; a staged file must not
// turn it into `/restart\n\n[Attached file: …]` (an ordinary message to the startup door). Prose stays a message.
test('Restart typed with a staged file stays the exact command; words around it stay a message', async () => {
    const tab = keptTab();
    try {
        const room = tab.open();
        await room.send('/restart');
        await room.send('/restart please');
        const [command, prose] = room.sent.map((item) => item.frame);
        assert.equal(command.content, '/restart', 'exact, as the host gate matches it');
        assert.equal(command.attachments.length, 1, 'its file rides the same row');
        assert.equal(prose.content, '/restart please\n\n[Attached file: scan.pdf]', 'prose keeps the tail: never the command');
        assert.ok(room.bubble('cm-1').innerHTML.includes('<div class="message" data-reply-body>/restart</div>'));
    } finally { tab.close(); }
});

const DOUBT = 'Saved; delivery not confirmed.';

// The receipt "saved, delivery unknown" belongs to the kept frame, never to the DOM: a bubble the feed released, a
// tab nobody is looking at and a reload in between change nothing about it. The restored bubble shows the doubt and
// its one action; the running host's own fact on the row takes both off the bubble itself, hidden tab or not.
test('a saved row of an ended host met without its bubble, in an inactive tab, stays a doubt that a restored bubble shows and the host\'s fact removes', async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('reload');
        const { client_message_id: cmid, content: words } = first.sent[0].frame;
        first.bubble(cmid).remove();                       // the feed released the bubble
        document.hidden = true;                            // and the tab is inactive
        first.echo(cmid, words, { restarted: true });      // the saved row of a host process that has since ended
        assert.equal(tab.kept()[0].deliveryUnknown, true, 'the receipt is kept with the frame, not with the bubble');
        assert.equal(first.instance.hasPendingWork(), false, 'a saved delivery doubt is not pending work');
        first.instance.destroy();
        document.hidden = false;

        tab.host.history = [];                             // the next first read has no row to settle the kept copy
        const reloaded = tab.open();
        await until(() => reloaded.asNew(cmid).length === 1, 'the restored bubble shows the doubt and its one action');
        assert.deepEqual(reloaded.notes(cmid).map((node) => node.textContent), [DOUBT]);
        assert.deepEqual(reloaded.retries(cmid), [], 'nothing replays: the host would only rejoin it');
        document.hidden = true;                            // inactive again when the host later confirms its dispatch
        reloaded.echo(cmid, words);
        assert.deepEqual(reloaded.asNew(cmid), [], 'Send as new leaves the bubble with the warning');
        assert.deepEqual(reloaded.notes(cmid).map((node) => node.textContent), ['Input saved']);
        assert.deepEqual(tab.kept(), [], 'and the kept copy with them');
    } finally { document.hidden = false; tab.close(); }
});

test('the running host\'s pending fact ends a delivery doubt on the bubble and in the kept copy alike', async () => {
    const tab = keptTab();
    try {
        const room = tab.open();
        await room.send('pending later');
        const { client_message_id: cmid, content: words } = room.sent[0].frame;
        room.echo(cmid, words, { restarted: true });
        assert.deepEqual(room.notes(cmid).map((node) => node.textContent), [DOUBT]);
        assert.equal(tab.kept()[0].deliveryUnknown, true);
        room.echo(cmid, words, { ingress_pending: true });  // the live host took the row and still owes its dispatch
        assert.deepEqual(room.notes(cmid).map((node) => node.textContent), ['Input saved'], 'no warning of an ended host remains');
        assert.deepEqual(room.asNew(cmid), []);
        assert.equal(tab.kept()[0].deliveryUnknown, undefined, 'the kept copy no longer doubts it either');
        room.echo(cmid, words);                             // then its dispatch
        assert.deepEqual(tab.kept(), []);
    } finally { tab.close(); }
});

function bubbleIn(root, id) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble user';
    bubble.dataset.clientMessageId = id;
    const time = document.createElement('div');
    time.className = 'msg-time';
    bubble.appendChild(time);
    root.appendChild(bubble);
    bubble.isConnected = true;
    return bubble;
}

test('the host\'s dispatch fact takes a delivery doubt, a Not confirmed note and their actions off the bubble by itself', () => {
    const env = installDom();
    try {
        const root = document.createElement('div');
        root.isConnected = true;
        const bubble = bubbleIn(root, 'cm-7');
        const unconfirmed = createUnconfirmedSends({ send: () => ({ status: 'sent' }), root: () => root, onDomWrite: (fn) => fn(),
            showToast() {}, compose() {} });
        const buttons = () => bubble.querySelectorAll('[data-unconfirmed-action]').map((node) => node.dataset.unconfirmedAction);
        const notes = () => [...bubble.querySelectorAll('[data-ingress-saved]'), ...bubble.querySelectorAll('[data-ingress-unconfirmed]')];
        const saved = { role: 'user', client_message_id: 'cm-7', ingress_accepted: true };

        unconfirmed.track({ type: 'chat', content: 'words', client_message_id: 'cm-7' });
        unconfirmed.settle(saved);                                    // a row of an ended host
        assert.deepEqual(buttons(), ['send_as_new']);
        unconfirmed.settle({ ...saved, ingress_dispatched: true });   // no saved tick is drawn by anyone here: the fact alone clears
        assert.deepEqual([buttons(), notes().length, unconfirmed.count], [[], 0, 0]);

        unconfirmed.track({ type: 'chat', content: 'again', client_message_id: 'cm-7' });
        unconfirmed.unsettle();                                       // the socket closed first: Not confirmed as saved
        assert.deepEqual(buttons(), ['retry', 'discard']);
        unconfirmed.settle({ ...saved, ingress_dispatched: true });
        assert.deepEqual([buttons(), notes().length, unconfirmed.count], [[], 0, 0]);

        unconfirmed.track({ type: 'chat', content: 'third', client_message_id: 'cm-7' });
        markIngressSaved(root, { ...saved, ingress_pending: true });  // saved by the live host, dispatch still owed
        unconfirmed.unsettle();                                       // a close then offers Send again on that note
        assert.deepEqual(buttons(), ['retry', 'discard']);
        unconfirmed.settle({ ...saved, ingress_dispatched: true });
        assert.deepEqual(buttons(), [], 'a pending note keeps its tick and loses its actions');
        assert.equal(notes().length, 1);
    } finally {
        restoreDom(env.prior);
    }
});

test('a copy given up names its message to the room: Discard, and Send as new once the composer took its words', () => {
    const env = installDom();
    try {
        const root = document.createElement('div');
        root.isConnected = true;
        const doubted = bubbleIn(root, 'cm-7'), unsaved = bubbleIn(root, 'cm-8');
        const given = [];
        let takes = false;
        const unconfirmed = createUnconfirmedSends({ send: () => ({ status: 'sent' }), root: () => root, onDomWrite: (fn) => fn(),
            showToast() {}, compose: () => takes, onGivenUp: (id) => given.push(id) });
        const press = (bubble, kind) => bubble.querySelectorAll('[data-unconfirmed-action]').find((node) => node.dataset.unconfirmedAction === kind).click();
        unconfirmed.track({ type: 'chat', content: 'words', client_message_id: 'cm-7', new_work: 'token' });
        unconfirmed.settle({ role: 'user', client_message_id: 'cm-7', ingress_accepted: true });  // a row of an ended host
        press(doubted, 'send_as_new');
        assert.deepEqual(given, [], 'a composer busy with another draft took nothing: the copy is still the message');
        takes = true;
        press(doubted, 'send_as_new');
        assert.deepEqual(given, ['cm-7'], 'its words are the owner\'s next message now');
        unconfirmed.track({ type: 'chat', content: 'again', client_message_id: 'cm-8', new_work: 'token' });
        unconfirmed.unsettle(['cm-8']);  // the socket closed before its row
        press(unsaved, 'retry');
        assert.deepEqual(given, ['cm-7'], 'Send again gives nothing up: the same message goes once more');
        unconfirmed.unsettle(['cm-8']);
        press(unsaved, 'discard');
        assert.deepEqual(given, ['cm-7', 'cm-8']);
    } finally {
        restoreDom(env.prior);
    }
});

test('a saved row of an ended host met before the kept bubble is shown is kept as unknown, and reconcile shows that bubble with the doubt', () => {
    const env = installDom();
    try {
        const root = document.createElement('div');
        root.isConnected = true;
        const store = new Map([['k', JSON.stringify([{ frame: { type: 'chat', content: 'words', client_message_id: 'cm-9' }, views: [], ts: TS }])]]);
        const unconfirmed = createUnconfirmedSends({ send: () => ({ status: 'sent' }), root: () => root, onDomWrite: (fn) => fn(),
            showToast() {}, compose() {}, storage: { getItem: (key) => store.get(key) ?? null,
                setItem: (key, value) => store.set(key, value), removeItem: (key) => store.delete(key) }, storageKey: 'k' });
        unconfirmed.settle({ role: 'user', client_message_id: 'cm-9', ingress_accepted: true });  // no bubble in the feed yet
        assert.equal(JSON.parse(store.get('k'))[0].deliveryUnknown, true, 'the receipt does not wait for a bubble');
        const shown = [];
        unconfirmed.reconcile((entry) => { shown.push(entry.frame.client_message_id); bubbleIn(root, entry.frame.client_message_id); });
        assert.deepEqual(shown, ['cm-9'], 'the first read left it out, so the kept words bring the bubble back');
        const note = root.querySelectorAll('[data-ingress-saved]')[0];
        assert.equal(note?.textContent.startsWith(DOUBT), true, 'with the doubt on it');
        assert.deepEqual(root.querySelectorAll('[data-unconfirmed-action]').map((node) => node.dataset.unconfirmedAction), ['send_as_new']);
        assert.equal(unconfirmed.count, 0);
    } finally {
        restoreDom(env.prior);
    }
});

