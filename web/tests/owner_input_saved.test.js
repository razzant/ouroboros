// В9: the host stamps a typed `ingress_accepted: true` on the web owner echo
// only after the durable chat write (supervisor/message_bus.py
// handle_web_message: log_chat(require_write=True), then the echo). The owner
// bubble with that client_message_id may then say `Input saved` — a fact about
// the saved row, never that work started. An echo without the typed flag is
// unknown and adds nothing, and the flag never retires `Sending...` (2A).
import assert from 'node:assert/strict';
import test from 'node:test';
import { markIngressSaved } from '../modules/chat_activity.js';
import { createUnconfirmedSends } from '../modules/chat_attachments.js';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

const ME = 'session-me';
const TS = '2026-09-25T09:00:00Z';

function fixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: history, window: { complete: true } }
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
        notes: (cmid) => (bubble(cmid)?.children || []).filter((node) => Object.hasOwn(node.dataset, 'ingressSaved')),
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
        assert.equal(f.status(), 'Sending...');
        f.echo({ content: 'please look at the logs', sender_session_id: ME, client_message_id: 'cm-1', ingress_accepted: true });
        const [note] = f.notes('cm-1');
        assert.equal(note?.textContent, 'Input saved');
        assert.equal(note.className, 'msg-pending', 'the quiet delivery-note style, no new chrome');
        const kids = f.bubble('cm-1').children;
        assert.equal(kids.indexOf(note), kids.findIndex((node) => node.classList.contains('msg-time')) - 1, 'above the time');
        assert.equal(f.status(), 'Sending...', 'a saved row is not a started turn');
        f.echo({ content: 'please look at the logs', sender_session_id: ME, client_message_id: 'cm-1', ingress_accepted: true });
        assert.equal(f.notes('cm-1').length, 1, 'a repeated echo keeps one note');
        assert.equal(f.messages.children.filter((node) => node.classList.contains('user')).length, 1);
    } finally { f.close(); }
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
            : String(url).startsWith('/api/chat/history') ? { messages: history, window: { complete: true } }
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
        assert.equal(bubble('cm-1').children.filter((node) => Object.hasOwn(node.dataset, 'ingressSaved')).length, 1);
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
            root: () => root, onDomWrite: (fn) => fn(), showToast: (text, tone) => toasts.push([text, tone]) });
        unconfirmed.track({ type: 'chat', content: 'x', client_message_id: 'cm-9' });
        unconfirmed.unsettle();
        const note = bubble.children.find((node) => Object.hasOwn(node.dataset, 'ingressUnconfirmed'));
        note.children.find((node) => node.tagName === 'BUTTON').click();
        assert.deepEqual(toasts, [['Still offline. Reconnect and send again.', 'error']]);
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
// copy with its FIRST history read: a saved row ends it, a row the host proved never dispatched
// (`ingress_undispatched`) keeps it, and anything else — no row, or no readable history — is in
// doubt. Nothing resends by itself; "Discard" forgets only this tab's copy, never an upload.
function keptTab() {
    const upload = `${'b'.repeat(32)}_scan.pdf`;
    const view = { name: 'scan.pdf', kind: 'file', mime: 'application/pdf', size: 4, available: true,
        url: `/api/files/download?upload=${upload}` };
    const host = { history: [], fails: false, calls: [] };
    const env = installDom(async (url, init = {}) => {
        host.calls.push([String(url), String(init.method || 'GET')]);
        const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
        if (String(url).startsWith('/api/chat/upload')) {
            return reply({ ok: true, filename: upload, display_name: 'scan.pdf', mime: 'application/pdf', view });
        }
        if (String(url).startsWith('/api/chat/history')) {
            return host.fails ? reply({ error: 'HTTP 503' }, 503) : reply({ messages: host.history, window: { complete: true } });
        }
        return reply({ active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true });
    });
    globalThis.sessionStorage.setItem('ouro_chat_session_id', ME);
    const priorSocket = globalThis.WebSocket;
    globalThis.WebSocket = { OPEN: 1 };
    let coined = 0;
    const rooms = [];
    // A saved row is the live host process's (its dispatch entered) unless it is proven undispatched, or `restarted`:
    // taken by a host process that has since ended, which no longer knows what became of it.
    const live = ({ restarted, ...extra } = {}) => ({ ...(extra.ingress_undispatched || restarted ? {} : { ingress_dispatched: true }), ...extra });
    // One Project room instance, as a reload or a reopened room creates it on this tab's storage.
    function open() {
        const handlers = new Map();
        const sent = [];
        let generation = 0;
        const instance = createChatInstance({
            ws: { ws: { readyState: 1 },
                on(type, fn) { if (!handlers.has(type)) handlers.set(type, new Set()); handlers.get(type).add(fn); return () => handlers.get(type).delete(fn); },
                isConnected: () => true,
                send(frame, options) {
                    const id = frame.client_message_id || `cm-${++coined}`;
                    sent.push({ frame: { ...frame, client_message_id: id }, options });
                    return { status: 'sent', clientMessageId: id };
                } },
            state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
            updateUnreadBadge() {}, chatId: 7, idPrefix: 'chat', mountEl: env.mount, asPanel: true,
            stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
        });
        const messages = document.byId.get('chat-messages');
        const bubble = (cmid) => messages.children.find((node) => node.dataset.clientMessageId === cmid);
        const room = {
            instance, sent, bubble,
            bubbles: (cmid) => messages.children.filter((node) => node.dataset.clientMessageId === cmid).length,
            fire: (type, payload) => [...(handlers.get(type) || [])].forEach((fn) => fn(payload)),
            notes: (cmid) => (bubble(cmid)?.children || []).filter((node) => node.className === 'msg-pending'),
            actions: (cmid) => room.notes(cmid).flatMap((note) => note.children.filter((node) => node.tagName === 'BUTTON')),
            press: (cmid, label) => room.actions(cmid).find((node) => node.textContent === label).click(),
            async send(text) {
                const fileInput = document.byId.get('chat-file-input');
                fileInput.files = [new File(['%PDF'], 'scan.pdf', { type: 'application/pdf' })];
                for (const fn of fileInput.listeners.get('change')) fn();
                document.byId.get('chat-input').value = text;
                for (const fn of document.byId.get('chat-send').listeners.get('click')) fn();
                await settle();
            },
            echo: (cmid, content, extra = {}) => room.fire('chat', { type: 'chat', role: 'user', chat_id: 7, ts: TS, source: 'web',
                content, sender_session_id: ME, client_message_id: cmid, ingress_accepted: true, attachments: [view], ...live(extra) }),
        };
        rooms.push(room);
        return room;
    }
    const kept = () => JSON.parse(globalThis.sessionStorage.getItem('ouro_chat_unconfirmed:7') || '[]');
    const row = (cmid, text, extra = {}) => ({ role: 'user', text, ts: TS, chat_id: 7, source: 'web', sender_session_id: ME,
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

test('a sent attachment message outlives a reload and a room teardown: Send again with the same id', async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('look');
        const [{ frame }] = first.sent;
        assert.deepEqual(Object.keys(tab.kept()[0].frame).sort(),
            ['attachments', 'chat_id', 'client_message_id', 'content', 'force_plan', 'sender_session_id', 'type']);
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
        assert.equal(globalThis.sessionStorage.getItem('ouro_chat_unconfirmed:7'), null, 'the saved row ends the kept copy');
        assert.equal(reloaded.bubbles('cm-1'), 1, 'the saved echo settles the restored bubble, never a second one');
        assert.equal(reloaded.instance.hasPendingWork(), false);
    } finally { tab.close(); }
});

test("the first read settles a saved row and keeps the one the host proved never dispatched", async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('saved');
        await first.send('undispatched');
        const content = first.sent[1].frame.content;
        first.instance.destroy();
        tab.host.history = [tab.row('cm-1', first.sent[0].frame.content),
            tab.row('cm-2', content, { ingress_undispatched: true })];

        const reloaded = tab.open();
        await until(() => reloaded.actions('cm-2').length === 2, 'the undispatched row offers its one handover');
        assert.deepEqual(reloaded.notes('cm-1').map((node) => node.textContent), ['Input saved']);
        assert.equal(reloaded.notes('cm-2')[0].textContent.startsWith('Saved, not delivered.'), true);
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-2'], 'saved ends one, proof keeps the other');

        reloaded.press('cm-2', 'Send again');
        assert.equal(reloaded.sent.length, 1);
        assert.equal(reloaded.sent[0].frame.client_message_id, 'cm-2');
        assert.equal(reloaded.actions('cm-2').length, 0);
        assert.equal(reloaded.notes('cm-2')[0].textContent, 'Saved, not delivered.', 'still saved, not yet delivered');
        reloaded.fire('close');
        assert.equal(reloaded.actions('cm-2').length, 2, 'a close before the answer offers it again');
        reloaded.echo('cm-2', content);  // the host handed it over: a plain saved echo
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), ['Input saved']);
        assert.deepEqual(tab.kept(), []);
        assert.deepEqual([reloaded.bubbles('cm-1'), reloaded.bubbles('cm-2')], [1, 1], 'history rows, no restored duplicates');
        reloaded.fire('chat', { type: 'chat', role: 'user', chat_id: 7, client_message_id: 'cm-2', ingress_accepted: true,
            ingress_undispatched: true, content, ts: TS, source: 'web', sender_session_id: ME });
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), ['Input saved'], 'Input saved is final');
    } finally { tab.close(); }
});

test('Discard forgets only the kept copy, and unreadable history is no proof either way', async () => {
    const tab = keptTab();
    try {
        const first = tab.open();
        await first.send('maybe');
        first.instance.destroy();
        tab.host.fails = true;
        const reloaded = tab.open();
        await until(() => reloaded.actions('cm-1').length === 2, 'the preview bubble is in doubt, not declared unsaved');
        assert.ok(reloaded.notes('cm-1')[0].textContent.startsWith('Not confirmed as saved.'), 'never "not saved"');
        reloaded.press('cm-1', 'Discard');
        assert.deepEqual(tab.kept(), []);
        assert.equal(reloaded.sent.length, 0);
        assert.deepEqual(tab.host.calls.filter(([, method]) => method === 'DELETE'), [], 'no upload is deleted');
        assert.equal(reloaded.actions('cm-1').length, 0);
        assert.equal(reloaded.notes('cm-1').length, 1, 'the bubble still says what is known');
        assert.equal(reloaded.instance.hasPendingWork(), false);
        reloaded.fire('close');
        assert.equal(reloaded.actions('cm-1').length, 0, 'a discarded message is not offered again');
    } finally { tab.close(); }
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
            root: () => root, onDomWrite: (fn) => fn(), showToast: (text, tone) => toasts.push([text, tone]), storage, storageKey: 'k' });
        assert.equal(unconfirmed.count, 0, 'an unreadable copy restores nothing');
        assert.deepEqual(toasts, [[UNREAD, 'error']], 'and says so');
        unconfirmed.track({ type: 'chat', content: 'x', client_message_id: 'cm-5' });
        assert.deepEqual(toasts.map(([, tone]) => tone), ['error', 'error']);
        assert.ok(toasts[1][0].includes('could not keep'));
        unconfirmed.unsettle();
        bubble.children[0].children.find((node) => node.textContent === 'Send again').click();
        assert.deepEqual(sends, [{ type: 'chat', content: 'x', client_message_id: 'cm-5' }]);
        unconfirmed.release();
        assert.equal(unconfirmed.count, 0);

        // What a working tab keeps: words, id, routing and upload refs — no surface, no bytes.
        const kept = new Map();
        const keeping = createUnconfirmedSends({ send: () => ({ status: 'sent' }), root: () => root, onDomWrite: (fn) => fn(),
            showToast: () => {}, storage: { getItem: (key) => kept.get(key) ?? null, setItem: (key, value) => kept.set(key, value),
                removeItem: (key) => kept.delete(key) }, storageKey: 'k' });
        keeping.track({ type: 'chat', content: 'y', client_message_id: 'cm-6', chat_id: 7, force_plan: false,
            client_surface: { ua: 'agent' }, image_base64: 'AAAA', attachments: [{ filename: 'f', display_name: 'n', mime: 'm', path: '/x' }] });
        assert.deepEqual(JSON.parse(kept.get('k'))[0].frame, { type: 'chat', content: 'y', client_message_id: 'cm-6', chat_id: 7,
            force_plan: false, attachments: [{ filename: 'f', display_name: 'n', mime: 'm' }] });
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

// A host restart ends the process that knew what became of a saved row (its `ingress_dispatched`), and with it any
// proof that the row never reached dispatch. A kept send that meets such a row says it is saved with its delivery
// unconfirmed and keeps no Send again — the host would only rejoin it — and nothing is resent. Rows nobody kept are
// history and keep their plain `Input saved`.
test('after a host restart a kept send meets its saved row: delivery not confirmed, nothing replays', async () => {
    const tab = keptTab();
    const DOUBT = 'Saved; delivery not confirmed.';
    try {
        const first = tab.open();
        await first.send('reload');
        await first.send('live');
        await first.send('proven');
        const [reloadWords, liveWords, provenWords] = first.sent.map((item) => item.frame.content);
        first.instance.destroy();
        // Before the restart the host had proved cm-3 never dispatched; the restart took that proof.
        tab.host.history = [tab.row('cm-1', reloadWords, { restarted: true }), tab.row('cm-3', provenWords, { ingress_undispatched: true })];

        const reloaded = tab.open();
        await until(() => reloaded.notes('cm-1')[0]?.textContent === DOUBT, 'the first read meets a row of the ended process');
        assert.deepEqual(reloaded.actions('cm-1'), [], 'no Send again: it could only rejoin');
        assert.equal(reloaded.notes('cm-1').length, 1, 'one note: the doubt replaced Input saved');
        assert.deepEqual(reloaded.notes('cm-3')[0].textContent, 'Saved, not delivered.');
        assert.equal(reloaded.actions('cm-2').length, 2, 'the one the host never saw is still in doubt with its actions');
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-2', 'cm-3']);

        // The proven row after a restart: the host no longer knows; the doubt replaces Saved, not delivered.
        tab.host.history = [tab.row('cm-1', reloadWords, { restarted: true }), tab.row('cm-3', provenWords, { restarted: true })];
        await reloaded.instance.refreshHistory({ revision: 2 });
        assert.deepEqual(reloaded.notes('cm-3').map((node) => node.textContent), [DOUBT]);
        assert.deepEqual(reloaded.actions('cm-3'), []);
        assert.deepEqual(reloaded.notes('cm-1').map((node) => node.textContent), [DOUBT], 'the doubt is final for that read');

        // Live: Send again reaches the restarted host, which rejoins the row an ended process saved.
        reloaded.press('cm-2', 'Send again');
        reloaded.echo('cm-2', liveWords, { restarted: true });
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), [DOUBT]);
        assert.deepEqual(reloaded.actions('cm-2'), []);
        reloaded.fire('close');
        assert.deepEqual([reloaded.actions('cm-1'), reloaded.actions('cm-2'), reloaded.actions('cm-3')], [[], [], []],
            'a socket close offers none of them again');
        assert.deepEqual(reloaded.sent.map((item) => item.frame.client_message_id), ['cm-2'], 'only the one manual Send again');
        assert.deepEqual(tab.kept(), []);
        assert.equal(reloaded.instance.hasPendingWork(), false);
        reloaded.instance.destroy();

        // Nothing is kept any more: the next read shows these rows as history, like every other saved row.
        tab.host.history = [tab.row('cm-1', reloadWords, { restarted: true })];
        const later = tab.open();
        await until(() => later.notes('cm-1').length === 1, 'history replays');
        assert.deepEqual(later.notes('cm-1').map((node) => node.textContent), ['Input saved']);
    } finally { tab.close(); }
});

// The running host can be read between its append and its dispatch (supervisor/message_bus.py writes the row,
// then enters dispatch; a deferred dispatch is echoed first): that row says `ingress_pending`. It is no delivery
// doubt: the kept frame waits for the next fact — dispatched settles it, a proven-undispatched row offers its one
// handover, and only a row of a process that has since ended (none of the three) ends it in the doubt.
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

        await room.send('then its write failed');
        const second = room.sent[1].frame.content;
        tab.host.history = [tab.row('cm-1', first), tab.row('cm-2', second, pending)];
        await room.instance.refreshHistory({ revision: 2 });
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-2']);
        tab.host.history = [tab.row('cm-1', first), tab.row('cm-2', second, { ingress_undispatched: true })];
        await room.instance.refreshHistory({ revision: 3 });
        assert.deepEqual(room.notes('cm-2').map((node) => node.textContent.split(' ')[0]), ['Saved,']);
        assert.ok(room.notes('cm-2')[0].textContent.startsWith('Saved, not delivered.'), 'the later fact replaces the pending note');
        assert.deepEqual(room.actions('cm-2').map((node) => node.textContent), ['Send again', 'Discard'], 'its one handover');
        room.press('cm-2', 'Send again');
        assert.deepEqual(room.sent[2].frame, room.sent[1].frame, 'the same frame and id');
        assert.equal(room.instance.hasPendingWork(), true);
    } finally { tab.close(); }
});

// A reload's first read that shows a kept send's row `ingress_pending` has answered for it: saved, the running host
// not yet said. The kept frame waits for that word like a live one — no doubt, no actions, nothing resent — so a
// Discard cannot drop the one handover a later proven-undispatched row offers.
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

        tab.host.history = [tab.row('cm-1', handover, { ingress_undispatched: true }), tab.row('cm-2', dispatched)];
        await reloaded.instance.refreshHistory({ revision: 2 });
        assert.ok(reloaded.notes('cm-1')[0].textContent.startsWith('Saved, not delivered.'));
        assert.deepEqual(reloaded.actions('cm-1').map((node) => node.textContent), ['Send again', 'Discard'], 'its one handover');
        assert.deepEqual(reloaded.notes('cm-2').map((node) => node.textContent), ['Input saved']);
        assert.deepEqual(reloaded.actions('cm-2').map((node) => node.textContent), []);
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-1', 'cm-3'], 'dispatched ends its kept copy');
        reloaded.press('cm-1', 'Send again');
        assert.deepEqual(reloaded.sent.map((item) => item.frame), [first.sent[0].frame], 'the same frame and id');
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
        room.fire('close');
        assert.deepEqual(room.actions('cm-1').map((node) => node.textContent), ['Send again', 'Discard'],
            'the socket closed before the host said: the owner may ask again');
        room.press('cm-1', 'Send again');
        assert.deepEqual(room.actions('cm-1'), [], 'resent: the doubt waits for the next fact');
        room.echo('cm-1', first);  // the rejoin says the same process dispatched it
        assert.deepEqual(tab.kept().map((entry) => entry.frame.client_message_id), ['cm-2']);
        // cm-2's host ended before it said: the next read shows its row from an ended process.
        tab.host.history = [tab.row('cm-1', first), tab.row('cm-2', second, { restarted: true })];
        await room.instance.refreshHistory({ revision: 1 });
        assert.deepEqual(room.notes('cm-2').map((node) => node.textContent), [DOUBT]);
        assert.deepEqual(room.actions('cm-2'), []);
        assert.deepEqual(tab.kept(), []);
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
    const message = (node) => node.innerHTML.match(/<div class="message[^"]*">([\s\S]*?)<\/div>/)[1];
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
        assert.ok(room.bubble('cm-1').innerHTML.includes('<div class="message">/restart</div>'));
    } finally { tab.close(); }
});
