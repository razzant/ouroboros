// The send side of replies and the delivery doubt through the real chat consumer (CONTRACTS K5): a reply sends only
// the quoted row's address on the chat frame, the kept frame keeps it for Send again (same id, same frame), and every
// owner send — words or files, not a bare slash command — waits for its saved row; a frame still in the socket's
// offline queue is that queue's until it is written.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { uploadView } from './helpers/attachment_views.js';

const ME = 'session-me';
const TS = '2026-10-16T09:40:00Z';
const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };
const roomFetch = async (url) => ({ ok: true, json: async () =>
    String(url).startsWith('/api/chat/history') ? { messages: [], window: { complete: true } }
        : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } });

function fixture(fetch = roomFetch) {
    const env = installDom(fetch);
    globalThis.sessionStorage.setItem('ouro_chat_session_id', ME);
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
                return { status, clientMessageId: id };
            } },
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
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

test('a path typed in chat waits for its saved row and Send again keeps the same frame', async () => {
    const f = fixture();
    try {
        const path = '/tmp/project logs';
        await f.send(path);
        const { frame } = f.sent.at(-1);
        assert.equal(f.kept().at(-1).frame.content, path);
        const mine = f.bubbleOf((node) => node.dataset.clientMessageId === frame.client_message_id);
        f.fire('close');
        assert.equal(f.sent.length, 1);
        assert.deepEqual(f.actions(mine), ['Send again', 'Discard'], 'words, not only files, get the doubt');
        f.notes(mine)[0].children.find((child) => child.textContent === 'Send again').click();
        assert.deepEqual(f.sent.at(-1).frame, frame, 'the SAME frame and id: the host rejoins a row it saved after all');
        f.echo(frame.client_message_id, path);
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
        const sentBefore = f.sent.length;
        f.notes(mine)[0].children.find((child) => child.textContent === 'Send as new').click();
        // (The stub DOM keeps no bubble words; chat_reply.test.js pins the words joining the draft.)
        assert.equal(document.activeElement, document.byId.get('chat-input'), 'the composer takes focus');
        assert.equal(f.sent.length, sentBefore, 'preparing sends nothing');
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

test('one click owns its own words and reply; edits made while the room saves a setting stay for the next message', async () => {
    const saves = [];
    const f = fixture(async (url, init = {}) => {
        if (String(url) === '/api/owner/context-mode') return new Promise((resolve) => saves.push({ resolve }));
        return roomFetch(url, init);
    });
    try {
        f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'First answer.', row_address: 'row:1@a#1' });
        f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'Second answer.', row_address: 'row:2@a#2' });
        f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'Third answer.', row_address: 'row:3@a#3' });
        const of = (address) => f.bubbleOf((node) => node.dataset.rowAddress === address);
        replyButton(of('row:1@a#1')).click();          // this send answers the first row
        const input = document.byId.get('chat-input');
        const capsule = document.byId.get('chat-context-mode');
        const pick = (mode) => {
            const seg = capsule.querySelector(`.chat-seg[data-mode="${mode}"]`);
            for (const listener of capsule.listeners.get('click')) listener({ target: { closest: (selector) => (selector === '.chat-seg' ? seg : null) } });
        };
        const press = () => { for (const listener of document.byId.get('chat-send').listeners.get('click')) listener(); };
        const type = (words) => { input.value = words; for (const listener of input.listeners.get('input')) listener({ target: input }); };
        pick('low');                                    // a context choice is saving when Send is pressed
        type('clicked words');
        press();
        replyButton(of('row:2@a#2')).click();          // while it saves, the owner answers someone else…
        type('typed while saving');                     // …and types new words
        saves[0].resolve({ ok: true, json: async () => ({ context_mode: 'low' }) });
        await settle();
        const { frame } = f.sent.at(-1);
        assert.equal(frame.content, 'clicked words', 'the click\'s own words went out');
        assert.equal(frame.reply_to, 'row:1@a#1', 'the click\'s own reply rides the frame');
        const echo = f.bubbleOf((node) => node.dataset.clientMessageId === frame.client_message_id);
        assert.ok(echo.children.some((child) => child.classList.contains('chat-quote')), 'the echo quotes the click\'s reply');
        assert.equal(input.value, 'typed while saving', 'the words typed while it saved stay');
        assert.equal(globalThis.sessionStorage.getItem('ouro_chat_draft'), 'typed while saving', 'and the room keeps them');
        await f.send('follow-up');
        assert.equal(f.sent.at(-1).frame.reply_to, 'row:2@a#2', 'the reply staged while it saved rides the next message');
        assert.equal(f.sent.at(-1).frame.content, 'follow-up');

        // A send clicked with no reply answers nothing; one staged while it saves stays its own.
        pick('max');                                    // the second context choice is saving
        type('plain send');
        press();
        replyButton(of('row:3@a#3')).click();
        saves[1].resolve({ ok: true, json: async () => ({ context_mode: 'max' }) });
        await settle();
        assert.equal(f.sent.at(-1).frame.content, 'plain send');
        assert.equal(f.sent.at(-1).frame.reply_to, undefined, 'a click with no reply sends none');
        await f.send('after all');
        assert.equal(f.sent.at(-1).frame.reply_to, 'row:3@a#3', 'and the reply staged meanwhile stays for its own message');
    } finally { f.close(); }
});

// One click owns what it was pressed with (composer.js, chat_attachments.js): its words, its reply and the files staged
// then. What the owner stages, types or removes while the room saves a setting or the files upload is the next
// message's, and the room's kept draft is cleared only while it still is this field's own copy.
function clickFixture() {
    const priorSocket = globalThis.WebSocket;
    globalThis.WebSocket = { OPEN: 1, CONNECTING: 0 };
    const saves = [], uploads = [], removed = [], held = [], refuse = new Set();
    const f = fixture(async (url, init = {}) => {
        if (String(url) === '/api/owner/context-mode') return new Promise((resolve, reject) => saves.push({ resolve, reject }));
        if (String(url) === '/api/chat/upload') {
            if (init.method === 'DELETE') { removed.push(JSON.parse(init.body).filename); return { ok: true, status: 200, json: async () => ({ ok: true }) }; }
            const name = init.body.get('file').name;
            uploads.push(name);
            const answer = refuse.has(name) ? { ok: false, status: 500, json: async () => ({ ok: false, error: 'disk full' }) }
                : { ok: true, status: 200, json: async () => ({ ok: true, filename: `stored-${name}`, display_name: name,
                    mime: 'text/plain', view: uploadView(name, 'file') }) };
            return held.length ? new Promise((resolve) => held[0].gates.push(() => resolve(answer))) : answer;
        }
        return roomFetch(url, init);
    });
    const input = document.byId.get('chat-input');
    input.setSelectionRange = (start, end) => { input.selectionStart = start; input.selectionEnd = end; };
    const capsule = document.byId.get('chat-context-mode');
    const preview = document.byId.get('chat-attachment-preview');
    const files = document.byId.get('chat-file-input');
    let mode = 'max';
    const api = {
        f, saves, uploads, removed, held, refuse, input,
        // Choosing the other context mode starts a save the fake server holds: a Send pressed now waits for it.
        hold() {
            mode = mode === 'low' ? 'max' : 'low';
            const seg = capsule.querySelector(`.chat-seg[data-mode="${mode}"]`);
            for (const listener of capsule.listeners.get('click')) listener({ target: { closest: (selector) => (selector === '.chat-seg' ? seg : null) } });
        },
        land: (index = 0) => saves[index].resolve({ ok: true, json: async () => ({ context_mode: mode }) }),
        type(words) {
            input.value = words;
            for (const listener of input.listeners.get('input')) listener({ target: input });
            input.selectionStart = input.selectionEnd = words.length;   // the caret follows what was typed
        },
        // The owner's own key in the field: ArrowUp at the very start recalls the previous message.
        arrow(key) { for (const listener of input.listeners.get('keydown')) listener({ key, target: input, preventDefault() {} }); },
        // What the browser reports before a typed character: the selection the edit will replace, or the caret it begins at.
        beforeInput(start, end = start) {
            input.setSelectionRange(start, end);
            for (const listener of input.listeners.get('beforeinput') || []) listener({ target: input });
        },
        composer: f.instance.composer,
        // The selection a browser reports to `beforeinput`: every word selected, so the next edit begins at the start.
        selectAll() {
            input.selectionStart = 0;
            input.selectionEnd = input.value.length;
            for (const listener of input.listeners.get('beforeinput') || []) listener({ target: input });
        },
        press() { for (const listener of document.byId.get('chat-send').listeners.get('click')) listener(); },
        attach(...names) {
            files.files = names.map((name) => new File(['x'], name, { type: 'text/plain' }));
            for (const listener of files.listeners.get('change')) listener({ target: files });
        },
        staged: () => [...preview.innerHTML.matchAll(/class="attach-name" title="([^"]*)"/g)].map((match) => match[1]),
        takeOut: (index) => preview.querySelectorAll('[data-attachment-remove]')[index].click(),
        kept: () => globalThis.sessionStorage.getItem('ouro_chat_draft'),
        close() { f.close(); if (priorSocket === undefined) delete globalThis.WebSocket; else globalThis.WebSocket = priorSocket; },
    };
    return api;
}

test('a click sends the files staged when it was pressed; a file staged while the room saves is the next message\'s', async () => {
    const c = clickFixture();
    try {
        c.attach('first.txt');
        c.hold();
        c.type('with the first file');
        c.press();
        c.attach('second.txt');           // staged after the click, while the setting saves
        c.land();
        await settle();
        assert.deepEqual(c.uploads, ['first.txt'], 'only the click\'s own file uploaded');
        const { frame } = c.f.sent.at(-1);
        assert.deepEqual(frame.attachments.map((item) => item.display_name), ['first.txt'], 'the frame names what went out');
        assert.match(frame.content, /first\.txt/);
        assert.doesNotMatch(frame.content, /second\.txt/);
        assert.deepEqual(c.staged(), ['second.txt'], 'the later file stays staged');
        assert.equal(c.f.instance.hasPendingWork(), true);
        await c.f.send('the second one');
        assert.deepEqual(c.uploads, ['first.txt', 'second.txt']);
        assert.deepEqual(c.f.sent.at(-1).frame.attachments.map((item) => item.display_name), ['second.txt']);
        assert.deepEqual(c.staged(), []);
    } finally { c.close(); }
});

test('a file the owner takes out while the room saves is not sent, and the rest of the click goes', async () => {
    const c = clickFixture();
    try {
        c.attach('keep.txt', 'drop.txt');
        c.hold();
        c.type('two files');
        c.press();
        c.takeOut(1);                      // the owner changes their mind before anything uploads
        c.land();
        await settle();
        assert.deepEqual(c.uploads, ['keep.txt']);
        assert.deepEqual(c.f.sent.at(-1).frame.attachments.map((item) => item.display_name), ['keep.txt']);
        assert.deepEqual(c.staged(), []);
        c.attach('only.txt');
        c.hold();
        c.press();                         // no words, one file: taken out, nothing is left to send
        c.takeOut(0);
        c.land(1);
        await settle();
        assert.equal(c.f.sent.length, 1, 'a click whose every file left and has no words sends nothing');
    } finally { c.close(); }
});

test('while files upload the owner cannot stage more; what they type and reply to meanwhile stays for the next message', async () => {
    const c = clickFixture();
    try {
        c.f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'First answer.', row_address: 'row:1@a#1' });
        c.f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'Second answer.', row_address: 'row:2@a#2' });
        const of = (address) => c.f.bubbleOf((node) => node.dataset.rowAddress === address);
        replyButton(of('row:1@a#1')).click();
        c.attach('big.txt');
        const upload = { gates: [] };
        // Hold the upload by routing the next fetch through a gate the test opens.
        globalThis.fetch = ((inner) => async (url, init) => {
            if (String(url) === '/api/chat/upload' && init?.method !== 'DELETE') {
                const answer = await inner(url, init);
                return new Promise((resolve) => upload.gates.push(() => resolve(answer)));
            }
            return inner(url, init);
        })(globalThis.fetch);
        c.type('upload me');
        c.press();
        await settle();
        assert.equal(upload.gates.length, 1, 'the upload is under way');
        c.attach('late.txt');               // refused: staging waits for the upload
        replyButton(of('row:2@a#2')).click();
        c.type('upload me, then more');
        upload.gates[0]();
        await settle();
        const { frame } = c.f.sent.at(-1);
        assert.equal(frame.reply_to, 'row:1@a#1', 'the click\'s own reply rode the frame');
        assert.deepEqual(frame.attachments.map((item) => item.display_name), ['big.txt']);
        assert.deepEqual(c.staged(), [], 'the refused file never staged and the sent one left');
        assert.equal(c.input.value, ', then more', 'only words added after the click stay');
        await c.f.send('next');
        assert.equal(c.f.sent.at(-1).frame.reply_to, 'row:2@a#2', 'the reply chosen while it uploaded answers the next message');
    } finally { c.close(); }
});

test('a failed upload cleans what went up and keeps every staged file, the later ones too', async () => {
    const c = clickFixture();
    try {
        c.attach('a.txt', 'b.txt');
        c.refuse.add('b.txt');
        c.hold();
        c.type('retry me');
        c.press();
        c.attach('c.txt');
        c.land();
        await settle();
        assert.deepEqual(c.uploads, ['a.txt', 'b.txt']);
        assert.deepEqual(c.removed, ['stored-a.txt'], 'the file that did go up is removed again');
        assert.equal(c.f.sent.length, 0, 'nothing was sent');
        assert.deepEqual(c.staged(), ['a.txt', 'b.txt', 'c.txt']);
        assert.equal(c.input.value, 'retry me');
        c.refuse.clear();
        c.press();
        await settle();
        assert.deepEqual(c.f.sent.at(-1).frame.attachments.map((item) => item.display_name), ['a.txt', 'b.txt', 'c.txt'],
            'the retry is a new click and carries what is staged now');
        assert.deepEqual(c.staged(), []);
    } finally { c.close(); }
});

test('words added after the click stay; a rewrite stays whole even when it shares the click\'s beginning or its words', async () => {
    const c = clickFixture();
    try {
        const sentContent = () => c.f.sent.at(-1).frame.content;
        c.type('hello');
        c.hold();
        c.press();
        c.type('hello world');              // words added at the end
        c.land(0);
        await settle();
        assert.equal(sentContent(), 'hello');
        assert.deepEqual([c.input.value, c.kept()], [' world', ' world'], 'only the added words stay');

        c.type('hello');
        c.hold();
        c.press();
        c.type('');                          // select all, type again: shares the beginning, is not an append
        c.type('hello there');
        c.land(1);
        await settle();
        assert.equal(sentContent(), 'hello');
        assert.deepEqual([c.input.value, c.kept()], ['hello there', 'hello there'], 'the rewrite stays whole');

        c.type('hello');
        c.hold();
        c.press();
        c.type('hell');                      // deleted and typed back to the very same words
        c.type('hello');
        c.land(2);
        await settle();
        assert.deepEqual([c.input.value, c.kept()], ['hello', 'hello'], 'a replaced draft with the same words is not consumed');

        c.type('plain');
        c.hold();
        c.press();
        c.land(3);
        await settle();
        assert.deepEqual([c.input.value, c.kept()], ['', null], 'an untouched field is consumed with its kept draft');

        c.type('hello');
        c.hold();
        c.press();
        for (const words of ['hello wor', 'hello wo', 'hello wo', 'hello wo and more']) c.type(words);  // typing and correcting after it
        c.land(4);
        await settle();
        assert.equal(sentContent(), 'hello');
        assert.deepEqual([c.input.value, c.kept()], [' wo and more', ' wo and more'], 'a correction made after the click\'s words is still an append');

        c.type('hello');
        c.hold();
        c.press();
        c.selectAll();                       // select all and paste over it, in one step: the beginning is the same
        c.type('hello there, rewritten in one step');
        c.land(5);
        await settle();
        assert.deepEqual([c.input.value, c.kept()], ['hello there, rewritten in one step', 'hello there, rewritten in one step'],
            'an edit that began inside the click\'s words is a rewrite, whatever it ended with');
    } finally { c.close(); }
});

test('the room\'s kept draft is cleared only while it is still this field\'s own copy', async () => {
    const c = clickFixture();
    try {
        c.type('clicked words');
        c.hold();
        c.press();
        globalThis.sessionStorage.setItem('ouro_chat_draft', 'a newer copy kept elsewhere');
        c.land();
        await settle();
        assert.equal(c.f.sent.at(-1).frame.content, 'clicked words');
        assert.equal(c.input.value, '', 'the field consumed what the click sent');
        assert.equal(c.kept(), 'a newer copy kept elsewhere', 'a newer kept version is neither cleared nor overwritten');

        c.type('more words');
        c.hold();
        c.press();
        c.f.instance.destroy();             // the room closed while the setting saved
        globalThis.sessionStorage.setItem('ouro_chat_draft', 'the reopened room\'s draft');
        c.land(1);
        await settle();
        assert.equal(c.kept(), 'the reopened room\'s draft', 'a send that outlived its room leaves the next one\'s draft alone');
    } finally { c.close(); }
});

for (const [failure, name] of [['refused', 'a refused'], ['rejected', 'a network-failed']]) {
    test(`${name} settings save sends nothing and leaves the click's words, files and reply as they were`, async () => {
        const c = clickFixture();
        try {
            c.f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'An answer.', row_address: 'row:1@a#1' });
            replyButton(c.f.bubbleOf((node) => node.dataset.rowAddress === 'row:1@a#1')).click();
            c.attach('kept.txt');
            c.hold();
            c.type('still mine');
            c.press();
            if (failure === 'refused') c.saves[0].resolve({ ok: false, status: 400, json: async () => ({ error: 'Not available', saved: false }) });
            else c.saves[0].reject(new Error('Failed to fetch'));
            await settle();
            assert.equal(c.f.sent.length, 0);
            assert.equal(c.input.value, 'still mine');
            assert.deepEqual([c.staged(), c.uploads], [['kept.txt'], []], 'nothing uploaded, nothing unstaged');
            assert.equal(document.byId.get('chat-send').disabled, false, 'Send can be pressed again');
            await c.f.send('again');
            assert.equal(c.f.sent.at(-1).frame.reply_to, 'row:1@a#1', 'the reply is still staged');
            assert.deepEqual(c.f.sent.at(-1).frame.attachments.map((item) => item.display_name), ['kept.txt']);
        } finally { c.close(); }
    });
}

// The strip leaves with the message that carried its reply. The send holds the staged reference it captured when pressed, so
// a reply chosen again for the very same row while it saves or uploads is a new staging: the next message's, not consumed.
test('a reply chosen again for the same row while the room saves stays for the next message; the click\'s own frame and echo are exact', async () => {
    const c = clickFixture();
    try {
        c.f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'An answer.', row_address: 'row:1@a#1' });
        const answer = c.f.bubbleOf((node) => node.dataset.rowAddress === 'row:1@a#1');
        replyButton(answer).click();
        c.hold();
        c.type('first words');
        c.press();
        replyButton(answer).click();       // the same row again while the setting saves
        c.land();
        await settle();
        const { frame } = c.f.sent.at(-1);
        assert.deepEqual([frame.content, frame.reply_to], ['first words', 'row:1@a#1'], 'the click\'s frame carries its address, as ever');
        const echo = c.f.bubbleOf((node) => node.dataset.clientMessageId === frame.client_message_id);
        assert.ok(echo.children.some((child) => child.classList.contains('chat-quote')), 'and its echo quotes it');
        await c.f.send('second words');
        assert.equal(c.f.sent.at(-1).frame.reply_to, 'row:1@a#1', 'the reply chosen again answers the next message');
        await c.f.send('third words');
        assert.equal(c.f.sent.at(-1).frame.reply_to, undefined, 'and the send that carried it consumes it');
    } finally { c.close(); }
});

test('a reply chosen again for the same row while the files upload stays for the next message', async () => {
    const c = clickFixture();
    try {
        c.f.fire('chat', { type: 'chat', role: 'assistant', chat_id: 1, ts: TS, content: 'An answer.', row_address: 'row:1@a#1' });
        const answer = c.f.bubbleOf((node) => node.dataset.rowAddress === 'row:1@a#1');
        replyButton(answer).click();
        c.attach('big.txt');
        const upload = { gates: [] };
        c.held.push(upload);               // the upload answers when the test opens its gate
        c.type('with a file');
        c.press();
        await settle();
        assert.equal(upload.gates.length, 1, 'the upload is under way');
        replyButton(answer).click();
        upload.gates[0]();
        await settle();
        const { frame } = c.f.sent.at(-1);
        assert.deepEqual([frame.reply_to, frame.attachments.map((item) => item.display_name)], ['row:1@a#1', ['big.txt']]);
        await c.f.send('next');
        assert.equal(c.f.sent.at(-1).frame.reply_to, 'row:1@a#1');
        await c.f.send('after');
        assert.equal(c.f.sent.at(-1).frame.reply_to, undefined);
    } finally { c.close(); }
});

// Lineage, not prefix: words a Send click read are rewritten by any edit that began inside them or replaced the whole
// field, and a field that merely begins with them is the owner's new draft then. Typing after them is an append.
test('a whole-field history recall while Send waits stays whole even when it begins with the words being sent', async () => {
    const c = clickFixture();
    try {
        await c.f.send('hello there');                 // an earlier message the recall can bring back
        c.type('hello');
        c.hold();
        c.press();                                     // the click reads "hello" and waits for the setting
        c.input.setSelectionRange(0, 0);
        c.arrow('ArrowUp');                            // "hello there" begins with the words being sent
        assert.equal(c.input.value, 'hello there');
        c.land();
        await settle();
        assert.equal(c.f.sent.at(-1).frame.content, 'hello');
        assert.deepEqual([c.input.value, c.kept()], ['hello there', 'hello there'], 'the recalled draft is the next message, whole');

        c.type('hello');                               // the words being sent come back by recall: still a replacement
        c.hold();
        c.press();
        c.input.setSelectionRange(0, 0);
        c.arrow('ArrowUp');
        c.arrow('ArrowDown');                          // past the newest entry: the draft the recall began from
        assert.equal(c.input.value, 'hello');
        c.land(1);
        await settle();
        assert.deepEqual([c.input.value, c.kept()], ['hello', 'hello'], 'a replaced draft with the same words is not consumed');
    } finally { c.close(); }
});

// A key that moves the recall nowhere changed nothing: the field is still the words the click read, not a new draft.
test('ArrowDown on the newest draft recalls nothing: the waiting Send consumes its words, and words added after them stay', async () => {
    const c = clickFixture();
    try {
        await c.f.send('earlier');                     // the recall holds an entry, so the key is weighed at all
        c.type('hello');
        c.hold();
        c.press();                                     // the click reads "hello" and waits for the setting, held here
        c.arrow('ArrowDown');                          // the caret is at the end of the newest draft: no entry to move to
        assert.deepEqual([c.input.value, c.f.sent.length], ['hello', 1], 'the field is untouched and nothing went out yet');
        c.land();
        await settle();
        assert.equal(c.f.sent.at(-1).frame.content, 'hello');
        assert.deepEqual([c.input.value, c.kept()], ['', null], 'the sent words are consumed, never left as a second copy');

        c.type('hello');
        c.hold();
        c.press();
        c.arrow('ArrowDown');
        c.beforeInput(5);                              // the owner goes on typing after the words being sent
        c.type('hello again');
        c.land(1);
        await settle();
        assert.equal(c.f.sent.at(-1).frame.content, 'hello');
        assert.deepEqual([c.input.value, c.kept()], [' again', ' again'], 'only what came after the sent words stays');

        c.type('hello');                               // a real recall on the way is still a replacement
        c.hold();
        c.press();
        c.input.setSelectionRange(0, 0);
        c.arrow('ArrowUp');
        c.arrow('ArrowDown');                          // back to the draft the recall began from: an entry changed
        c.arrow('ArrowDown');                          // and this one moves nowhere
        c.land(2);
        await settle();
        assert.deepEqual([c.input.value, c.kept()], ['hello', 'hello'], 'the earlier replacement keeps the field whole');
    } finally { c.close(); }
});

test('an insertion that begins inside the words being sent rewrites them; one after them, and typing there, only append', async () => {
    const c = clickFixture();
    try {
        const insert = (words, from, to = from) => { c.input.setSelectionRange(from, to); c.composer.insertAtCursor(words); };
        c.type('hello');
        c.hold();
        c.press();
        insert('ello there', 1, 5);                    // dictation typed over "ello": the result begins with the sent words
        c.land(0);
        await settle();
        assert.deepEqual([c.input.value, c.kept()], ['hello there', 'hello there']);

        c.type('hello');
        c.hold();
        c.press();
        c.beforeInput(5);                              // the owner types after the words…
        c.type('hello t');
        insert('here', 7, 7);                          // …and a toolbar item or dictation adds at the caret there
        c.land(1);
        await settle();
        assert.deepEqual([c.input.value, c.kept()], [' there', ' there'], 'only what came after the sent words stays');

        c.type('hello');
        c.hold();
        c.press();
        c.composer.setDraft('hello world');            // a draft replaced by another that begins with the same words
        c.land(2);
        await settle();
        assert.deepEqual([c.input.value, c.kept()], ['hello world', 'hello world']);
    } finally { c.close(); }
});
