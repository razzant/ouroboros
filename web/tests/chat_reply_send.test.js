// The send side of replies and the delivery doubt through the real chat consumer (CONTRACTS K5): a reply sends only
// the quoted row's address on the chat frame, the kept frame keeps it for Send again (same id, same frame), and every
// owner send — words or files, not a bare slash command — waits for its saved row; a frame still in the socket's
// offline queue is that queue's until it is written.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

const ME = 'session-me';
const TS = '2026-10-16T09:40:00Z';
const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };

function fixture({ chatId = 1, history = () => [], savedReply, keptSends, onSend } = {}) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: history(), window: { complete: true } }
            : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } }));
    globalThis.sessionStorage.setItem('ouro_chat_session_id', ME);
    if (savedReply) globalThis.sessionStorage.setItem('ouro_chat_reply', JSON.stringify(savedReply));
    if (keptSends) globalThis.sessionStorage.setItem('ouro_chat_unconfirmed', JSON.stringify(keptSends));
    const handlers = new Map();
    const sent = [];
    let coined = 0;
    let status = 'sent';
    let generation = 0;
    const instance = createChatInstance({
        ws: { ws: { readyState: 1 },
            on(type, fn) { if (!handlers.has(type)) handlers.set(type, new Set()); handlers.get(type).add(fn); return () => handlers.get(type).delete(fn); },
            isConnected: () => true,
            send(frame, options) {
                const id = frame.client_message_id || `cm-${++coined}`;
                sent.push({ frame: { ...frame, client_message_id: id }, options });
                onSend?.(frame);
                return { status, clientMessageId: id };
            } },
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, idPrefix: 'chat', mountEl: env.mount, asPanel: chatId !== 1,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const messages = document.byId.get('chat-messages');
    const fire = (type, payload) => [...(handlers.get(type) || [])].forEach((fn) => fn(payload));
    const bubbleOf = (pred) => messages.children.find((node) => node.classList.contains('chat-bubble') && pred(node));
    // Delivery notes wherever they stand: the saved tick in the time line, or a row of words above it.
    const notes = (node) => node?.querySelectorAll('.msg-pending') || [];
    const actions = (node) => notes(node).flatMap((note) => note.children.filter((child) => child.tagName === 'BUTTON'))
        .map((button) => button.textContent);
    return {
        instance, messages, sent, fire, bubbleOf, notes, actions,
        set status(value) { status = value; },
        async send(text) {
            document.byId.get('chat-input').value = text;
            for (const fn of document.byId.get('chat-send').listeners.get('click')) fn();
            await settle();
        },
        echo: (cmid, content, extra = {}) => fire('chat', { type: 'chat', role: 'user', chat_id: 1, ts: TS, source: 'web', content,
            sender_session_id: ME, client_message_id: cmid, ingress_accepted: true, ingress_dispatched: true, ...extra }),
        kept: () => JSON.parse(globalThis.sessionStorage.getItem('ouro_chat_unconfirmed') || '[]'),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const replyButton = (bubble) => bubble.children.find((child) => child.className === 'chat-message-actions')
    ?.children.find((child) => child.classList.contains('chat-message-reply'));

test('Reply on an answer sends the next message with that row\'s address; the echo quotes it; the strip leaves', async () => {
    const f = fixture();
    try {
        f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'Seven projects are stale.',
            row_address: 'row:1@a#1', task_id: 'turn-1' });
        const answer = f.bubbleOf((node) => node.dataset.rowAddress === 'row:1@a#1');
        assert.ok(answer, 'the answer carries its address');
        replyButton(answer).click();
        await f.send('which ones?');
        const { frame } = f.sent.at(-1);
        assert.equal(frame.reply_to, 'row:1@a#1', 'only the address travels; the host makes the snippet');
        assert.equal(frame.content, 'which ones?');
        const echo = f.bubbleOf((node) => node.dataset.clientMessageId === frame.client_message_id);
        assert.ok(echo.children.some((child) => child.classList.contains('chat-quote')), 'the local echo quotes the answer');
        assert.equal(f.kept().at(-1).frame.reply_to, 'row:1@a#1', 'the kept frame keeps the reference');
        await f.send('and the rest?');
        assert.equal(f.sent.at(-1).frame.reply_to, undefined, 'a send answers once; the next message answers nothing');
        // The host's saved row gives the echo its own address and, with it, its Reply.
        f.echo(frame.client_message_id, 'which ones?', { row_address: 'row:1@a#2' });
        assert.equal(echo.dataset.rowAddress, 'row:1@a#2');
        assert.ok(replyButton(echo));
    } finally { f.close(); }
});

test('a new reply staged as the socket sends belongs to the next message', async () => {
    let next;
    const f = fixture({ onSend: () => next.click() });
    try {
        const stage = (address) => {
            f.fire('chat', { role: 'assistant', chat_id: 1, ts: TS, content: address, row_address: address });
            const bubble = f.bubbleOf(node => node.dataset.rowAddress === address);
            bubble.dataset.rowWords = address;
            return replyButton(bubble);
        };
        const sent = 'row:1@a#1', later = 'row:1@b#2';
        stage(sent).click();
        next = stage(later);
        await f.send('Check this one.');
        const { frame } = f.sent.at(-1);
        const echo = f.bubbleOf(node => node.dataset.clientMessageId === frame.client_message_id);
        assert.equal(frame.reply_to, sent);
        assert.equal(echo.querySelector('.chat-quote-text').textContent, sent);
        assert.equal(f.kept().at(-1).reply.reply_to.address, sent);
        assert.equal(JSON.parse(sessionStorage.getItem('ouro_chat_reply')).address, later);
    } finally { f.close(); }
});

test('a Project\'s copy of an id-less original takes that row\'s Reply once the row is read, and sends its exact address', async () => {
    // The copy and the row are one message by the binding's exact origin, never by its words or time.
    let rows = [{ role: 'user', text: 'Original request', ts: TS, client_message_id: '', origin_projected: true,
        origin_id: 'binding-origin', chat_id: 1 }];
    const f = fixture({ chatId: 42, history: () => rows });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        const kept = f.bubbleOf((node) => node.dataset.originId === 'binding-origin');
        assert.ok(kept && !replyButton(kept), 'the copy alone, unverified, offers no Reply');
        const address = 'row:1@then#origin';
        rows = [{ ...rows[0], origin_projected: false, history_id: 'chat:origin', row_address: address, source: 'web' }];
        await f.instance.refreshHistory({ revision: 2 });
        assert.equal(f.bubbleOf((node) => node.dataset.originId === 'binding-origin'), kept, 'adopted in place');
        replyButton(kept).click();
        await f.send('continue from here');
        assert.equal(f.sent.at(-1).frame.reply_to, address, 'the composer sends the adopted row\'s own address');
    } finally { f.close(); }
});

test('every owner message waits for its saved row: a closed socket offers Send again with the same frame', async () => {
    const f = fixture();
    try {
        await f.send('please look at the logs');
        const { frame } = f.sent.at(-1);
        const mine = f.bubbleOf((node) => node.dataset.clientMessageId === frame.client_message_id);
        f.fire('close');
        assert.deepEqual(f.actions(mine), ['Send again', 'Discard'], 'words, not only files, get the doubt');
        f.notes(mine)[0].children.find((child) => child.textContent === 'Send again').click();
        assert.deepEqual(f.sent.at(-1).frame, frame, 'the SAME frame and id: the host rejoins a row it saved after all');
        f.echo(frame.client_message_id, 'please look at the logs');
        assert.deepEqual(f.notes(mine).map((note) => note.textContent), ['Input saved']);
        assert.deepEqual(f.kept(), [], 'the saved row ends the doubt');
        assert.equal(f.instance.hasPendingWork(), false);
    } finally { f.close(); }
});

test('a saved row an ended host took says delivery not confirmed, and Send as new prepares the composer', async () => {
    const f = fixture();
    try {
        await f.send('restart the audit');
        const { frame } = f.sent.at(-1);
        const mine = f.bubbleOf((node) => node.dataset.clientMessageId === frame.client_message_id);
        f.echo(frame.client_message_id, 'restart the audit', { ingress_dispatched: undefined });
        assert.deepEqual(f.notes(mine).map((note) => note.textContent), ['Saved; delivery not confirmed.']);
        assert.deepEqual(f.actions(mine), ['Send as new'], 'nothing replays; the one honest step is a new message');
        assert.equal(f.kept()[0].deliveryUnknown, true);
        assert.equal(f.instance.hasPendingWork(), false, 'a saved delivery doubt is not pending work');
        const sentBefore = f.sent.length;
        mine.querySelector('.message').textContent = 'restart the audit';  // the stub DOM keeps no bubble words
        document.byId.get('chat-input').value = 'my plain draft';
        f.notes(mine)[0].children.find((child) => child.textContent === 'Send as new').click();
        assert.equal(document.activeElement, document.byId.get('chat-input'), 'the composer takes focus');
        assert.equal(document.activeElement.value, 'my plain draft\n\nrestart the audit', 'ordinary text-only recovery joins ordinary words');
        assert.equal(f.sent.length, sentBefore, 'preparing sends nothing');
    } finally { f.close(); }
});

test('a selected reply keeps its delivery doubt and recovery through two reloads, even without history', async () => {
    const savedReply = { address: 'row:1@source#0123456789ab', role: 'user', author_label: 'Mira',
        excerpt: { text: 'north gate', start_utf16: 12 } };
    let f = fixture({ savedReply });
    let kept;
    try {
        await f.send('Which gate?');
        const frame = f.sent.at(-1).frame;
        f.echo(frame.client_message_id, frame.content, { ingress_dispatched: undefined, reply_to: savedReply });
        kept = f.kept();
    } finally { f.close(); }
    for (let reload = 0; reload < 2; reload += 1) {
        f = fixture({ keptSends: kept });
        try {
            for (let n = 0; n < 100 && !f.bubbleOf(node => node.dataset.clientMessageId === kept[0].frame.client_message_id); n += 1) {
                await new Promise(resolve => setTimeout(resolve, 5));
            }
            const bubble = f.bubbleOf(node => node.dataset.clientMessageId === kept[0].frame.client_message_id);
            assert.ok(bubble);
            assert.deepEqual(f.actions(bubble), ['Send as new']);
            assert.ok(f.notes(bubble)[0].textContent.startsWith('Saved; delivery not confirmed.'));
            assert.equal(f.sent.length, 0);
            assert.equal(f.instance.hasPendingWork(), false);
            if (reload === 1) {
                bubble.querySelector('.message').textContent = 'Which gate?';
                bubble.querySelector('[data-unconfirmed-action="send_as_new"]').click();
                assert.equal(document.byId.get('chat-input').value, 'Which gate?');
                assert.deepEqual(JSON.parse(sessionStorage.getItem('ouro_chat_reply')).excerpt, savedReply.excerpt);
                f.echo(kept[0].frame.client_message_id, 'Which gate?');
                assert.deepEqual(f.kept(), [], 'a confirmed dispatch settles the stored delivery doubt');
                assert.deepEqual(f.actions(bubble), [], 'and Send as new leaves the bubble with its warning, not only the storage');
                assert.deepEqual(f.notes(bubble).map((note) => note.textContent), ['Input saved'], 'the saved tick stands where it was');
            } else kept = f.kept();
        } finally { f.close(); }
    }
});

test('a selected reply\'s delivery doubt met without its bubble in an inactive tab returns with its quote and Send as new, and the host\'s fact removes both', async () => {
    const savedReply = { address: 'row:1@source#0123456789ab', role: 'user', author_label: 'Mira',
        excerpt: { text: 'north gate', start_utf16: 12 } };
    let f = fixture({ savedReply });
    let kept, cmid;
    try {
        await f.send('Which gate?');
        const frame = f.sent.at(-1).frame;
        cmid = frame.client_message_id;
        f.bubbleOf((node) => node.dataset.clientMessageId === cmid).remove();  // the feed released the bubble
        document.hidden = true;                                                 // and the tab is inactive
        f.echo(cmid, frame.content, { ingress_dispatched: undefined, reply_to: savedReply });  // a row of an ended host
        kept = f.kept();
        assert.equal(kept[0].deliveryUnknown, true, 'the receipt does not wait for a bubble');
        assert.deepEqual(kept[0].reply.reply_to.excerpt, savedReply.excerpt, 'and keeps the quote it must give back');
    } finally { f.close(); }
    f = fixture({ keptSends: kept });
    try {
        for (let n = 0; n < 100 && !f.bubbleOf(node => node.dataset.clientMessageId === cmid); n += 1) {
            await new Promise(resolve => setTimeout(resolve, 5));
        }
        const bubble = f.bubbleOf(node => node.dataset.clientMessageId === cmid);
        assert.ok(bubble, 'the kept words bring the bubble back');
        assert.deepEqual(f.actions(bubble), ['Send as new']);
        assert.ok(f.notes(bubble)[0].textContent.startsWith('Saved; delivery not confirmed.'));
        document.hidden = true;                                                 // inactive again at the confirmation
        f.echo(cmid, 'Which gate?');
        assert.deepEqual(f.actions(bubble), [], 'Send as new is gone with the warning');
        assert.deepEqual(f.notes(bubble).map((note) => note.textContent), ['Input saved']);
        assert.deepEqual(f.kept(), []);
    } finally { f.close(); }
});

test('a reply stages the speaker projected by the host rather than its generic displayed sender', async () => {
    const f = fixture();
    try {
        f.fire('chat', { role: 'user', chat_id: 1, ts: TS, content: 'Meet at noon.',
            row_address: 'row:1@mira#0123456789ab', reply_author_label: 'Mira' });
        const source = f.bubbleOf(node => node.dataset.rowAddress === 'row:1@mira#0123456789ab');
        replyButton(source).click();
        assert.equal(JSON.parse(sessionStorage.getItem('ouro_chat_reply')).author_label, 'Mira');
    } finally { f.close(); }
});

test('a queued frame belongs to the socket queue until it is written; a bare slash command is no message', async () => {
    const f = fixture();
    try {
        f.status = 'queued';
        await f.send('sent while offline');
        const queued = f.sent.at(-1).frame.client_message_id;
        const bubble = f.bubbleOf((node) => node.dataset.clientMessageId === queued);
        f.fire('close');
        assert.deepEqual(f.actions(bubble), [], 'the queue will write it: nothing is offered twice');
        assert.equal(f.kept().at(-1).frame.client_message_id, queued, 'a reload still finds it kept');
        f.fire('outbound_sent', { clientMessageId: queued, queued: true });
        f.fire('close');
        assert.deepEqual(f.actions(bubble), ['Send again', 'Discard'], 'written but not saved: now it is in doubt');
        f.status = 'sent';
        await f.send('/status');
        const command = f.sent.at(-1).frame.client_message_id;
        f.fire('close');
        assert.deepEqual(f.actions(f.bubbleOf((node) => node.dataset.clientMessageId === command)), [],
            'a command is never offered again');
        assert.equal(f.kept().some((entry) => entry.frame.client_message_id === command), false);
    } finally { f.close(); }
});

test('a refused selected Reply waits in its bubble with the complete selection; Send again resends it, the field stays the owner\'s', async () => {
    const address = 'row:1@2026-10-16T09:40:00Z#0123456789ab';
    const excerpt = { text: 'Blue lanterns. '.repeat(20) + '\n  END', start_utf16: 18 };
    const savedReply = { address, chat_id: 1, author_label: 'Ouroboros', role: 'assistant', snippet: 'Opening.', excerpt };
    const f = fixture({ savedReply });
    try {
        await f.send('Which lantern?');
        const frame = f.sent.at(-1).frame;
        assert.deepEqual(frame.reply_excerpt, excerpt);
        assert.equal(frame.reply_to, address);
        assert.equal(document.byId.get('chat-input').value, '', 'the field is free for the next message');
        assert.equal(sessionStorage.getItem('ouro_chat_reply'), null);
        const bubble = f.bubbleOf(node => node.dataset.clientMessageId === frame.client_message_id);
        assert.ok(bubble.children.some(child => child.dataset.excerpt === '1'));
        document.byId.get('chat-input').value = 'A newer draft.';
        f.fire('chat', { role: 'system', system_type: 'initialization_notice', chat_id: 1,
            content: 'The quoted message is unavailable. Retry or clear the quote.', client_message_id: frame.client_message_id });
        assert.equal(document.byId.get('chat-input').value, 'A newer draft.', 'a late refusal does not erase the newer draft');
        assert.deepEqual(f.actions(bubble), ['Send again', 'Send as new', 'Discard']);
        assert.deepEqual(f.kept().at(-1).frame.reply_excerpt, excerpt);
        bubble.querySelector('[data-unconfirmed-action="retry"]').click();
        assert.deepEqual(f.sent.at(-1).frame, frame, 'the SAME frame: id, words and the complete selection');
        f.echo(frame.client_message_id, frame.content, { row_address: 'row:1@new#987654321abc', reply_to: savedReply });
        assert.deepEqual(f.kept(), []);
        assert.equal(document.byId.get('chat-input').value, 'A newer draft.');
    } finally { f.close(); }
});

test('a late acceptance leaves the newer draft alone and the sent quote does not return to it', async () => {
    const address = 'row:1@2026-10-16T09:40:00Z#0123456789ab';
    const savedReply = { address, role: 'assistant', excerpt: { start_utf16: 3, text: 'star atlas' } };
    const f = fixture({ savedReply });
    try {
        await f.send('First note.');
        const frame = f.sent.at(-1).frame;
        assert.equal(document.byId.get('chat-input').value, '');
        document.byId.get('chat-input').value = 'Another observation.';
        f.echo(frame.client_message_id, frame.content, { reply_to: savedReply });
        assert.equal(document.byId.get('chat-input').value, 'Another observation.');
        assert.equal(sessionStorage.getItem('ouro_chat_reply'), null, 'the quote left with the message it was sent in');
        assert.deepEqual(f.kept(), []);
        await f.send('Another observation.');
        assert.equal(f.sent.at(-1).frame.reply_to, undefined, 'the next message answers nothing');
    } finally { f.close(); }
});
