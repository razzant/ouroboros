// В9: the host stamps a typed `ingress_accepted: true` on the web owner echo
// only after the durable chat write (supervisor/message_bus.py
// handle_web_message: log_chat(require_write=True), then the echo). The owner
// bubble with that client_message_id may then say `Input saved` — a fact about
// the saved row, never that work started. An echo without the typed flag is
// unknown and adds nothing, and the flag never retires `Sending...` (2A).
import assert from 'node:assert/strict';
import test from 'node:test';
import { markIngressSaved } from '../modules/chat_activity.js';
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
