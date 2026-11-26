// A room away (room_retire.js) is hidden and quiet, yet nothing it owes is lost on its return. Through the real chat
// consumer: a disconnect while it is away still leaves its unsaved sends in doubt (Send again once it is shown, never
// drawn into the hidden room, and only for what was written before that disconnect), and a review detail read that
// answers while it is away is drawn on its return, never into the hidden room.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createChatInstance } from '../modules/chat.js';
import { createEarlyTextBubbles, createTextBubbles } from '../modules/chat_text_bubble.js';
import { enhanceChatMarkdown } from '../modules/chat_markdown.js';
import { isWindowText } from '../modules/chat_row_placement.js';
import { createRoomRetirement, createWindowReturn, roomSocket } from '../modules/room_retire.js';
import { installDesktopShellLinkInterceptor } from '../modules/ui_helpers.js';
import { installDom, PREVIEW_SCOPE, restoreDom, roomKey, walkCard } from './chat_dom_fixture.js';
import { installCardDom } from './helpers/card_dom.js';

const ME = 'session-me';
const TS = '2026-11-04T09:40:00Z';
const settle = async () => { for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };

// The page's one socket (ws.js), as a room's view of it sees it: subscriptions, connection and sends.
function pageSocket() {
    const handlers = new Map();
    let connected = true, coined = 0, status = 'sent', scope = PREVIEW_SCOPE;
    const sent = [];
    return {
        sent,
        set status(value) { status = value; },
        set connected(value) { connected = value; },
        set scope(value) { scope = value; },
        servingProfileScope: () => scope,
        listening: (type) => handlers.get(type)?.size || 0,
        ws: { readyState: 1 },
        on(type, fn) {
            if (!handlers.has(type)) handlers.set(type, new Set());
            handlers.get(type).add(fn);
            return () => handlers.get(type).delete(fn);
        },
        isConnected: () => connected,
        send(frame, options) {
            const id = frame.client_message_id || `cm-${++coined}`;
            sent.push({ frame: { ...frame, client_message_id: id }, options });
            return { status, clientMessageId: id };
        },
        fire: (type, payload) => [...(handlers.get(type) || [])].forEach((fn) => fn(payload)),
    };
}

function fixture({ detail = null, chatId = 7, observeDocument = false } = {}) {
    const reads = [];
    let history = { messages: [], window: { complete: true } };
    const env = installDom(async (url) => {
        const path = String(url);
        if (path.startsWith('/api/tasks/') && detail) return new Promise((resolve) => reads.push(resolve));
        return { ok: true, json: async () => (path.startsWith('/api/chat/history')
            ? { profile_scope: PREVIEW_SCOPE, ...await history }
            : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true }) };
    });
    if (observeDocument) {
        const events = new Map();
        document.addEventListener = (type, fn) => {
            if (!events.has(type)) events.set(type, new Set());
            events.get(type).add(fn);
        };
        document.removeEventListener = (type, fn) => events.get(type)?.delete(fn);
        document.dispatchEvent = event => [...(events.get(event.type) || [])].forEach(fn => fn(event));
    }
    globalThis.sessionStorage.setItem('ouro_chat_session_id', ME);
    const socket = pageSocket();
    let generation = 0;
    const instance = createChatInstance({
        ws: roomSocket(socket),
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, projectId: chatId === 1 ? '' : 'p1', idPrefix: 'chat', mountEl: env.mount, asPanel: chatId !== 1,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const instances = new Map([['p1', instance]]);
    const rooms = createRoomRetirement({ instances });
    const messages = document.byId.get('chat-messages');
    const bubbleOf = (cmid) => messages.children.find((node) => node.dataset.clientMessageId === cmid);
    const actions = (node) => (node?.querySelectorAll('.msg-pending') || [])
        .flatMap((note) => note.children.filter((child) => child.tagName === 'BUTTON')).map((button) => button.textContent);
    return {
        instance, socket, rooms, messages, bubbleOf, actions, reads,
        hide() { document.hidden = true; rooms.suspend(instance, true); },
        show() { document.hidden = false; rooms.restore(instance); },
        delayHistory() {
            let deliver;
            history = new Promise(resolve => { deliver = (messages, facts) => resolve({ messages, window: { complete: true }, ...facts }); });
            return deliver;
        },
        async send(text) {
            document.byId.get('chat-input').value = text;
            for (const fn of document.byId.get('chat-send').listeners.get('click')) fn();
            await settle();
            return socket.sent.at(-1).frame.client_message_id;
        },
        away() { rooms.putAway('p1'); },
        async back() { instance.page.hidden = false; rooms.show('p1'); await settle(); },
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

test('a disconnect while the room is away leaves its unsaved send in doubt on its return, not in the hidden room', async () => {
    const f = fixture();
    try {
        const cmid = await f.send('look at the logs');
        f.away();
        f.socket.connected = false;
        f.socket.fire('close');
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), [], 'nothing is drawn into the hidden room');
        f.socket.connected = true;  // the socket came back while the room was away; the host never saved the row
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), ['Send again', 'Discard'],
            'back, after its history caught up, the room says the send was not confirmed');
        assert.equal(f.instance.hasPendingWork(), true);
    } finally { f.close(); }
});

test('back while still disconnected, the room shows the outage and the doubt at once', async () => {
    const f = fixture();
    try {
        const cmid = await f.send('still offline?');
        f.away();
        f.socket.connected = false;
        f.socket.fire('close');
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), ['Send again', 'Discard']);
    } finally { f.close(); }
});

test('a send saved while the room was away is not put in doubt by an earlier disconnect', async () => {
    const f = fixture();
    try {
        const cmid = await f.send('saved after all');
        f.away();
        f.socket.connected = false;
        f.socket.fire('close');
        f.socket.connected = true;
        // The custody listener hears the host's saved row while the room is away.
        f.socket.fire('chat', { type: 'chat', role: 'user', chat_id: 7, ts: TS, source: 'web', content: 'saved after all',
            sender_session_id: ME, client_message_id: cmid, ingress_accepted: true, ingress_dispatched: true });
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), []);
        assert.equal(f.instance.hasPendingWork(), false);
    } finally { f.close(); }
});

test('a saved row of an ended host reaching the room while it is away is a kept doubt that its return shows and the host\'s fact removes', async () => {
    const f = fixture();
    const kept = () => JSON.parse(globalThis.sessionStorage.getItem(roomKey('ouro_chat_unconfirmed', 7)) || '[]');
    const echo = (cmid, extra = {}) => f.socket.fire('chat', { type: 'chat', role: 'user', chat_id: 7, ts: TS, source: 'web',
        content: 'look at the logs', sender_session_id: ME, client_message_id: cmid, ingress_accepted: true, ...extra });
    try {
        const cmid = await f.send('look at the logs');
        f.away();
        echo(cmid);  // no dispatch fact: the host process that saved the row has since ended
        assert.equal(kept()[0]?.deliveryUnknown, true, 'the receipt is kept while the room is away');
        assert.equal(f.instance.hasPendingWork(), false, 'a saved delivery doubt is not pending work');
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), ['Send as new'], 'back, the bubble wears the doubt and its one action');
        echo(cmid, { ingress_dispatched: true });  // later, the host confirms its dispatch
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), [], 'Send as new is gone with the warning');
        assert.deepEqual(kept(), []);
    } finally { f.close(); }
});

test('a frame the socket queue wrote after the disconnect is not in doubt for that disconnect', async () => {
    const f = fixture();
    try {
        f.socket.status = 'queued';
        f.socket.connected = false;
        const cmid = await f.send('sent while offline');
        f.away();
        f.socket.fire('close');                       // a failed reconnect attempt: the frame is still the queue's
        f.socket.connected = true;
        f.socket.fire('outbound_sent', { clientMessageId: cmid, queued: true });  // written once the socket came back
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), [], 'written after that disconnect: it waits for its saved row');
        f.socket.fire('close');
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), ['Send again', 'Discard'], 'a disconnect after it was written does');
    } finally { f.close(); }
});

test('a review detail that answers while the room is away is drawn on its return, not into the hidden room', async () => {
    const taskId = 'review-task';
    const detail = { task_id: taskId, status: 'running', review_projection: { panels: [], acceptance_incident: {
        incident_id: 'acceptance-preparation:1', status: 'failed', stage: 'preparation', attempts: 1, source_known: true,
        failure_kind: 'UnicodeDecodeError', failure_detail: 'bad byte', feedback_delivered: true } } };
    const f = fixture({ detail });
    try {
        f.socket.fire('chat', { chat_id: 7, role: 'system', is_progress: true, task_id: taskId, content: 'assembling',
            ts: '2026-11-04T09:00:00Z' });
        f.socket.fire('chat', { task_id: taskId, chat_id: 7, presentation_owner_task_id: taskId, role: 'system',
            is_progress: true, system_type: 'review_reference', surface: 'task_acceptance', state_revision: 'a'.repeat(64),
            ts: '2026-11-04T09:00:01Z', content: '' });
        await settle();
        assert.equal(f.reads.length, 1, 'the card reads its review detail');
        const host = walkCard(f.messages, taskId).querySelector('[data-live-reviews-host]');
        const before = host.innerHTML;
        f.away();
        f.reads.shift()({ ok: true, json: async () => detail });
        await settle();
        assert.equal(host.innerHTML, before, 'the answer waits: nothing is drawn into the hidden room');
        await f.back();
        assert.match(host.innerHTML, /acceptance evidence · host attempt 1/, 'back, the room draws what its read returned');
    } finally { f.close(); }
});

test('a room away draws no status when another room\'s send is written; back, it shows its own again', async () => {
    const f = fixture();
    try {
        await settle();
        const status = () => document.byId.get('chat-status')?.textContent;
        const shown = status();
        assert.notEqual(shown, 'Reconnecting…');
        f.away();
        // Every room's custody listener hears every written frame; a paused socket view is not an outage.
        f.socket.fire('outbound_sent', { clientMessageId: 'another-room-send' });
        assert.equal(status(), shown, 'nothing is drawn into the hidden room, least of all a false outage');
        await f.back();
        assert.equal(status(), shown);
    } finally { f.close(); }
});

const speech = (chatId, n, content = 'old answer', extra = {}) => ({ role: 'assistant', chat_id: chatId,
    content, ts: TS, row_address: `row:${chatId}@${TS}#${String(n).padStart(12, '0')}`, ...extra });

test('a delivered direct answer opens its ordinary link on return and keeps equal words during adoption', (t) => {
    const { doc, win, restore } = installCardDom();
    let disposeMarkdown;
    t.after(() => { try { disposeMarkdown?.(); } finally { restore(); } });
    const prior = { marked: globalThis.marked, DOMPurify: globalThis.DOMPurify, hljs: globalThis.hljs };
    t.after(() => Object.assign(globalThis, prior));
    const parser = {};
    vm.runInNewContext(readFileSync(new URL('../marked.min.js', import.meta.url), 'utf8'), parser);
    globalThis.marked = parser.marked;
    // The home double records entry through the sanitizer, not the sanitizer's own behavior.
    const sanitized = t.mock.fn(html => html);
    globalThis.DOMPurify = { sanitize: sanitized };
    // The card double implements compound selectors only. Supply the three code selectors used by Markdown.
    const element = Object.getPrototypeOf(doc.body);
    const select = element.querySelectorAll;
    t.mock.method(element, 'querySelectorAll', function (selector) {
        const insidePre = code => code.parentElement?.tagName === 'PRE';
        if (selector === 'pre > code') return select.call(this, 'code').filter(insidePre);
        if (selector === 'code:not(pre code)') return select.call(this, 'code').filter(code => !code.closest('pre'));
        if (selector === '.md-code-block pre > code') {
            return select.call(this, 'code').filter(code => insidePre(code) && code.closest('.md-code-block'));
        }
        return select.call(this, selector);
    });
    const create = doc.createElement;
    doc.createElement = (tag) => {
        const node = create(tag);
        if (tag === 'template') Object.defineProperty(node, 'innerHTML', {
            get: () => node.content.innerHTML, set: value => { node.content.innerHTML = value; },
        });
        if (tag === 'a') for (const attr of ['target', 'rel']) Object.defineProperty(node, attr, {
            set: value => node.setAttribute(attr, value),
        });
        return node;
    };
    const opened = [];
    win.pywebview = { api: { async open_external_url(url) { opened.push(url); return { ok: true }; } } };
    doc.addEventListener = (...args) => doc.body.addEventListener(...args);
    installDesktopShellLinkInterceptor({ win, doc });

    const feed = doc.createElement('main');
    doc.body.append(feed);
    const decorate = t.mock.fn();
    const draw = createTextBubbles({ chatMedia: { attachCopyControl() {} }, replies: { decorate }, feed });
    const early = createEarlyTextBubbles(feed, draw, node => feed.append(node), node => node.remove());
    const target = 'https://docs.example.test/guide';
    const frame = speech(1, 1, [`Read **these words** in the [guide](${target}).`, '',
        '```js', 'answer()', '```', '', '```js', 'control()', '```'].join('\n'), { markdown: true, _is_direct_chat: true });
    early.show([{ address: frame.row_address, text: frame.content, ts: frame.ts, markdown: frame.markdown }]);
    const bubble = early.get(frame.row_address);
    const body = bubble.querySelector('.message');
    const link = body.querySelector('a[href]');
    const selectedWord = body.querySelector('strong').firstChild;
    const [selectedCode, otherCode] = body.querySelectorAll('code');
    const selectedText = selectedCode.firstChild;
    assert.ok(body.querySelector('.md-code-copy'));
    assert.equal(bubble.dataset.windowWords, '', 'the provisional code Copy is hidden by the window-words rule');
    assert.equal(link.getAttribute('href'), target);
    assert.ok(sanitized.mock.calls.some(call => call.arguments[0].includes(target)));
    assert.equal(link.click().defaultPrevented, true, 'the ordinary shell interceptor owns this link');
    assert.deepEqual(opened, [target]);
    assert.equal(body.hasAttribute('data-reply-body'), false);
    assert.equal(bubble.dataset.historyId, undefined);
    assert.equal(decorate.mock.callCount(), 0);

    const saved = { ts: frame.ts, markdown: true, meta: {}, opts: { historyId: 'saved-direct', reply: frame,
        evidenceRef: { href: '/api/tasks/test/evidence', label: 'Evidence' } } };
    body.querySelector('strong').setAttribute('tabindex', '0'); // stand-in for the scroller's live enhancement
    const adopted = draw(frame.content, 'assistant', { ...saved, existing: bubble }).bubble;
    assert.equal(adopted.querySelector('.message'), body);
    assert.equal(body.querySelector('a[href]'), link);
    assert.equal(body.querySelector('strong').firstChild, selectedWord);
    assert.equal(body.hasAttribute('data-reply-body'), true);
    assert.equal(adopted.dataset.historyId, 'saved-direct');
    doc.getSelection = () => ({ isCollapsed: false, rangeCount: 1, getRangeAt: () => ({ intersectsNode: node => node === selectedCode }) });
    const highlight = source => ({ value: `<span>${source}</span>` });
    globalThis.hljs = { getLanguage: () => true, highlight, highlightAuto: highlight };
    disposeMarkdown = enhanceChatMarkdown(adopted);
    assert.equal(selectedCode.firstChild, selectedText, 'canonical enhancement must not detach the selected text');
    assert.equal(otherCode.classList.contains('hljs'), true, 'an unselected block is still highlighted');
    assert.equal(adopted.dataset.windowWords, undefined, 'normal Copy is offered after adoption');
    assert.equal(decorate.mock.callCount(), 1);
    assert.equal(link.click().defaultPrevented, true);
    assert.deepEqual(opened, [target, target]);

    // A second confirmed row corrects its early destination; a normal new bubble paints the same canonical route.
    const next = { address: speech(1, 2).row_address, text: frame.content, ts: frame.ts, markdown: true };
    early.delete(frame.row_address);
    early.show([next]);
    const canonical = frame.content.replace('/guide)', '/reference)');
    const corrected = draw(canonical, 'assistant', { ...saved, existing: early.get(next.address) }).bubble;
    const ordinary = draw(canonical, 'assistant', saved).bubble;
    const canonicalHref = 'https://docs.example.test/reference';
    for (const row of [corrected, ordinary]) {
        assert.equal(row.querySelector('.message').querySelector('a[href]').getAttribute('href'), canonicalHref);
    }
    assert.ok(sanitized.mock.calls.some(call => call.arguments[0].includes(canonicalHref)));
});

for (const chatId of [1, 7]) test(`hidden follower ${chatId}: show words before HTTP, not a live conclusion or read`, async () => {
    const f = fixture({ chatId });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        const deliver = f.delayHistory();
        const legacy = speech(chatId, 1);
        const sameMs = speech(chatId, 2);  // different saved row, identical words and milliseconds
        const root = speech(chatId, 3, 'root answer', { task_id: 'root', root_task_id: 'root',
            task_phase: 'finalizing', task_terminal_status: 'completed', delivery_id: 'final:root' });
        const live = speech(chatId, 4, 'live speech from work', { task_id: 'root', root_task_id: 'root', system_type: 'proactive_message' });
        const child = speech(chatId, 5, 'child belongs to a card', { task_id: 'child', root_task_id: 'root' });
        const ambiguous = speech(chatId, 6, 'task without proved lineage', { task_id: 'unproved' });
        const retry = speech(chatId, 7, 'physical retry answer', { task_id: 'retry', root_task_id: 'root',
            delegation_role: 'root', task_phase: 'finalizing', task_terminal_status: 'completed', delivery_id: 'final:retry' });
        const unprovedRetry = speech(chatId, 10, 'retry role unproved', { task_id: 'unproved-retry', root_task_id: 'root' });
        const conflictingRetry = speech(chatId, 11, 'retry has a parent', { ...retry, content: 'retry has a parent',
            task_id: 'nested-retry', parent_task_id: 'root', row_address: speech(chatId, 11).row_address });
        const other = speech(19, 1, 'another room', { also_rooms: [chatId] });
        const nodes = text => f.messages.children.filter(node => node.innerHTML.includes(text));

        f.hide();
        for (const frame of [sameMs, legacy, root, live, child, ambiguous, retry, unprovedRetry, conflictingRetry, other, legacy]) f.socket.fire('chat', frame);
        assert.equal(nodes('old answer').length, 0, 'words never repaint the hidden feed');
        f.show();
        const early = nodes('old answer');
        assert.equal(early.length, 2, 'different physical rows are not deduped by text or time');
        assert.equal(nodes('root answer').length, 1);
        assert.equal(nodes('live speech from work').length, 1);
        assert.equal(nodes('child belongs to a card').length + nodes('another room').length, 0);
        assert.equal(nodes('task without proved lineage').length, 0);
        assert.equal(nodes(retry.content).length, 1, 'a host-marked root retry is standalone speech');
        assert.equal(nodes(unprovedRetry.content).length + nodes(conflictingRetry.content).length, 0);
        assert.equal(nodes(retry.content)[0].dataset.rowAddress, undefined, 'a retry gains no early Reply target');
        assert.equal(early[0].dataset.historyId, undefined);
        assert.equal(early[0].dataset.rowAddress, undefined, 'no Reply target until the row is confirmed');
        assert.equal(early[0].querySelector('[data-reply-body]'), null);
        assert.ok(!walkCard(f.messages, 'root'), 'early rendering does not give finalizing authority');
        assert.ok(!walkCard(f.messages, 'retry'));

        f.hide();
        f.show();
        assert.deepEqual(nodes('old answer'), early, 'an empty repeat return keeps the actual nodes');
        f.hide();
        const twin = { ...live, row_address: speech(chatId, 8).row_address, delivery_id: 'proactive:second' };
        f.socket.fire('chat', twin);
        const firstProactive = nodes(live.content)[0];
        assert.equal(nodes(live.content).length, 1, 'no hidden paint');
        f.show();
        assert.deepEqual(nodes('old answer'), early, 'a return with new words keeps the old nodes too');
        assert.equal(nodes(live.content)[0], firstProactive);
        const secondProactive = nodes(live.content)[1];
        assert.ok(secondProactive);
        const visibleTwin = { ...live, row_address: speech(chatId, 9).row_address, delivery_id: 'proactive:third' };
        f.socket.fire('chat', visibleTwin);
        const visibleProactive = nodes(live.content).find(node => node.dataset.rowAddress === visibleTwin.row_address);
        assert.ok(visibleProactive, 'a physically different equal live reply is not deduplicated');

        const cmid = await f.send('a new question');
        const sending = () => document.byId.get('chat-status').textContent;
        assert.match(sending(), /Sending/);
        f.socket.fire('chat', legacy);
        assert.equal(nodes('old answer').length, 2);
        assert.match(sending(), /Sending/, 'a delayed legacy repeat cannot settle a different send');

        deliver([legacy, sameMs, root, live, twin, visibleTwin, retry].map((frame, index) => ({ ...frame, text: frame.content,
            history_id: `row-${index}`, history_position: { source: 'chat', offset: index } })));
        await settle();
        assert.deepEqual(nodes('old answer').map(node => node.dataset.rowAddress), [legacy.row_address, sameMs.row_address]);
        assert.deepEqual(nodes('old answer').map(node => node.dataset.historyId), ['row-0', 'row-1']);
        assert.equal(nodes('old answer')[0], early[1], 'the stamped saved row moves the same early node into its canonical slot');
        for (const [node, frame, index] of [[firstProactive, live, 3], [secondProactive, twin, 4], [visibleProactive, visibleTwin, 5]]) {
            assert.equal(node.dataset.rowAddress, frame.row_address, 'each Reply retains the exact physical row');
            assert.equal(node.dataset.historyId, `row-${index}`);
        }
        assert.equal(nodes(retry.content)[0].dataset.rowAddress, retry.row_address);
        f.socket.fire('chat', legacy);
        assert.match(sending(), /Sending/, 'an exact late repeat after canonical adoption is also not a live conclusion');
        assert.equal(nodes('old answer').length, 2);
        assert.ok(f.bubbleOf(cmid));
    } finally { f.close(); }
});

test('window return declines overflow and loses its lease on a scope change or socket close', () => {
    const socket = pageSocket();
    const doc = { hidden: true, querySelector: () => ({ content: PREVIEW_SCOPE }) };
    const lease = createWindowReturn({ socket, doc, chatId: 7, projectId: 'p1', accepts: () => true, knownChild: () => false });
    for (const invalidate of [
        () => { for (let n = 0; n < 13; n++) socket.fire('chat', speech(7, n)); },
        () => socket.fire('chat', speech(7, 7, 'x'.repeat(32768))),
        () => { socket.scope = '1111111111111111'; },
        () => socket.fire('close'),
        () => socket.fire('projects_changed', {}),
    ]) {
        doc.hidden = true;
        socket.scope = PREVIEW_SCOPE;
        lease.hold();
        socket.fire('chat', speech(7, 88));
        assert.equal(socket.listening('log'), 0);
        invalidate();
        doc.hidden = false;
        assert.equal(lease.take(), null);
        for (const event of ['chat', 'close', 'projects_changed']) assert.equal(socket.listening(event), 0, event);
    }
});

test('window return leaves an older reading position alone and waits for canonical history', async () => {
    const f = fixture();
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        const feed = f.messages;
        feed.scrollHeight = 1200;
        feed.scrollTop = 220;
        for (const wheel of feed.listeners.get('wheel')) wheel({ type: 'wheel', target: feed, deltaY: -100, timeStamp: 1000 });
        for (const scroll of feed.listeners.get('scroll')) scroll({ target: feed });
        await settle();
        const deliver = f.delayHistory();
        const frame = speech(7, 1, 'new while reading earlier');
        f.hide();
        f.socket.fire('chat', frame);
        f.show();
        assert.equal(feed.scrollTop, 220);
        assert.ok(feed.children.every(node => !node.innerHTML.includes(frame.content)));
        deliver([{ ...frame, text: frame.content, history_id: 'reading-control-latest', history_position: { source: 'chat', offset: 1 } }]);
        await settle();
        assert.ok(feed.children.some(node => node.dataset.rowAddress === frame.row_address));
        assert.equal(feed.scrollTop, 220, 'canonical arrival does not jump to the latest row');
    } finally { f.close(); }
});

test('leaving the live edge disables early collection until this window sequence settles', () => {
    const socket = pageSocket();
    const doc = { hidden: true, querySelector: () => ({ content: PREVIEW_SCOPE }) };
    const lease = createWindowReturn({ socket, doc, chatId: 7, projectId: 'p1', accepts: () => true, knownChild: () => false });
    const hide = following => { doc.hidden = true; lease.hold(following); };
    const show = (following = true) => { doc.hidden = false; return lease.take(following).map(row => row.text); };
    hide(true); socket.fire('chat', speech(7, 1, 'first')); assert.deepEqual(show(), ['first']);
    hide(false); socket.fire('chat', speech(7, 2, 'while reading')); assert.deepEqual(show(false), []);
    hide(true); socket.fire('chat', speech(7, 3, 'after the gap')); assert.deepEqual(show(), []);
    lease.drop();
    hide(true); socket.fire('chat', speech(7, 4, 'fresh sequence')); assert.deepEqual(show(), ['fresh sequence']);
    lease.drop();
});

const savedSpeech = (frame, offset) => ({ ...frame, text: frame.content, history_id: `saved-${offset}`,
    history_position: { source: 'chat', offset } });

for (const chatId of [1, 7]) test(`saved-only, early and live replies in one millisecond follow time before offsets (${chatId})`, async () => {
    const f = fixture({ chatId });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        const deliver = f.delayHistory();
        const frames = [0, 1, 2].map(n => speech(chatId, n, `ordered words ${n}`, { ts: `2026-11-04T09:40:00.00000${n}Z` }));
        const bubbles = () => f.messages.children.filter(node => node.innerHTML.includes('ordered words'));
        f.hide(); f.socket.fire('chat', frames[1]); f.show();
        f.socket.fire('chat', frames[2]);
        const [early, live] = bubbles();
        assert.ok(early && live && early !== live);
        deliver(frames.map((frame, index) => savedSpeech(frame, [3, 1, 2][index])));
        await settle();
        const rows = bubbles();
        assert.deepEqual(rows.map(node => node.dataset.historyOffset), ['3', '1', '2']);
        assert.deepEqual(rows.map(node => node.dataset.rowAddress), frames.map(frame => frame.row_address));
        assert.equal(rows[1], early);
        assert.equal(rows[2], live);
    } finally { f.close(); }
});

test('a redelivered early legacy row after project invalidation cannot settle a later send', async () => {
    const f = fixture();
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        const deliver = f.delayHistory();
        const frame = speech(7, 1, 'legacy words');
        const rows = () => f.messages.children.filter(node => node.innerHTML.includes(frame.content));
        f.hide(); f.socket.fire('chat', frame); f.show();
        assert.equal(rows().length, 1);
        f.socket.fire('projects_changed', { chat_id: 7 });
        assert.equal(rows().length, 0);
        const cmid = await f.send('newer pending question');
        const status = () => document.byId.get('chat-status').textContent;
        assert.match(status(), /Sending/);
        f.socket.fire('chat', frame);
        assert.match(status(), /Sending/);
        assert.ok(f.bubbleOf(cmid));
        assert.equal(rows().length, 1, 'the fresh ordinary visible delivery can still draw the words');
        assert.equal(rows()[0].dataset.historyId, undefined);
        deliver([savedSpeech(frame, 0)]);
        await settle();
        assert.equal(rows().length, 1);
        assert.equal(rows()[0].dataset.historyId, 'saved-0');
    } finally { f.close(); }
});

test('a second hide while reading back keeps the shown early words but no longer collects', async () => {
    const f = fixture();
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        const deliver = f.delayHistory();
        const early = speech(7, 1, 'the first return');
        const later = speech(7, 2, 'the non-following hide');
        const bubble = frame => f.messages.children.find(node => node.innerHTML.includes(frame.content));
        f.hide(); f.socket.fire('chat', early); f.show();
        const first = bubble(early);
        assert.ok(first);
        const feed = f.messages;
        feed.scrollHeight = 1200;
        feed.scrollTop = 220;
        for (const wheel of feed.listeners.get('wheel')) wheel({ type: 'wheel', target: feed, deltaY: -100, timeStamp: 1000 });
        for (const scroll of feed.listeners.get('scroll')) scroll({ target: feed });
        await settle();
        f.hide();
        // This exceeds the batch limit only if the non-following hide kept collecting.
        for (let n = 10; n < 23; n++) f.socket.fire('chat', speech(7, n));
        f.socket.fire('chat', later);
        f.show();
        assert.equal(bubble(early), first, 'unrelated arrivals cannot evict already shown words');
        assert.equal(bubble(later), undefined);
        assert.equal(feed.scrollTop, 220);
        deliver([savedSpeech(early, 0), savedSpeech(later, 1)]);
        await settle();
        assert.equal(bubble(early), first);
        assert.equal(bubble(later).dataset.historyId, 'saved-1');
    } finally { f.close(); }
});

// Only Main's actual shell resumes reading after it restores the room.
function mainVisibility(f, unread) {
    const source = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
    const functions = ['mainOnScreen', 'enterMain', 'syncMainLifetime'].map(name =>
        source.match(new RegExp(`^function ${name}\\([^\\n]*\\{[\\s\\S]*?^}`, 'm'))[0]);
    const listener = source.match(/^document\.addEventListener\('visibilitychange', \(\) => \{[\s\S]*?^\}\);/m)[0];
    let change;
    vm.runInNewContext([...functions, listener].join('\n'), {
        document: { get hidden() { return document.hidden; }, addEventListener(_type, fn) { change = fn; } },
        state: { activePage: 'chat' }, navState: { activeProjectId: null }, rooms: f.rooms, mainChat: f.instance,
        projectInstances: new Map(), mainUnreadCount: () => unread,
        acknowledgeMainAfterPaint: () => assert.fail('the window return leaves read credit to canonical history'),
    });
    return hidden => { document.hidden = hidden; change(); };
}

for (const [unread, back] of [[0, false], [2, false], [2, true]]) {
    test(`Main's visibility change keeps collecting while resume waits (${unread} unread, reading back: ${back})`, async () => {
        const f = fixture({ chatId: 1 });
        try {
            await f.instance.refreshHistory({ revision: 1 }); await settle();
            const change = mainVisibility(f, unread), deliver = f.delayHistory();
            const first = speech(1, 1, 'first hidden reply'), second = speech(1, 2, 'second hidden reply');
            const bubble = frame => f.messages.children.find(node => node.innerHTML.includes(frame.content));
            change(true); f.socket.fire('chat', first); change(false);
            const kept = bubble(first);
            assert.ok(kept);
            await settle();  // Main's automatic resume now waits for HTTP
            if (back) {
                const feed = f.messages;
                feed.scrollHeight = 1200; feed.scrollTop = 220;
                for (const wheel of feed.listeners.get('wheel')) wheel({ type: 'wheel', target: feed, deltaY: -100, timeStamp: 1000 });
                for (const scroll of feed.listeners.get('scroll')) scroll({ target: feed });
            }
            change(true); f.socket.fire('chat', second); change(false);
            assert.equal(bubble(first), kept);
            assert.equal(Boolean(bubble(second)), !back, 'only the human ends early collection');
            if (back) assert.equal(f.messages.scrollTop, 220);
            deliver([savedSpeech(first, 0), savedSpeech(second, 1)]); await settle();
            assert.equal(bubble(second).dataset.rowAddress, second.row_address);
        } finally { f.close(); }
    });
}

for (const outcome of ['selection ends', 'focus ends', 'older adoption', 'registry revokes', 'complete miss', 'placement mismatch']) {
    test(`retiring early words after history: ${outcome}`, async () => {
        const f = fixture({ observeDocument: true });
        try {
            await f.instance.refreshHistory({ revision: 1 }); await settle();
            const deliver = f.delayHistory(), early = speech(7, 1, 'the provisional words');
            const bubble = () => f.messages.children.find(node => node.innerHTML.includes(early.content));
            f.hide(); f.socket.fire('chat', early); f.show();
            const row = bubble(), body = row.querySelector('.message');
            const text = document.createTextNode(early.content);
            body.append(text);  // the fixture's HTML parser doesn't create text nodes
            const range = { startContainer: text, endContainer: text, startOffset: 0, endOffset: 3,
                intersectsNode: node => node.contains(text) };
            if (outcome === 'focus ends') document.activeElement = body;
            else document.getSelection = () => ({ rangeCount: 1, isCollapsed: false, getRangeAt: () => range });

            const owners = Array.from({ length: 150 }, (_, i) => savedSpeech(speech(7, i + 5, `owner text ${i}`,
                { role: 'user', ts: '2026-11-04T09:41:00Z' }), i + 5));
            const bounded = { has_more: true, page_cursor: 'present', next_cursor: 'older',
                window: { complete: false, truncated_by: ['quota'] } };
            const mismatch = outcome === 'placement mismatch', absent = outcome === 'complete miss';
            deliver(mismatch ? [savedSpeech({ ...early, role: 'system', content: 'a canonical System row' }, 0), ...owners] : owners,
                absent ? { has_more: false, page_cursor: 'present' } : bounded);
            await settle();
            if (mismatch || absent) {
                assert.equal(bubble(), undefined, 'canonical disproof outranks the interaction');
                if (mismatch) assert.ok(f.messages.children.some(node => node.dataset.rowAddress === early.row_address
                    && node.classList.contains('system')));
                return;
            }
            assert.equal(bubble(), row, 'bounded omission is not disproof');
            assert.equal(text.parentNode, body);
            assert.equal(row.dataset.historyId, undefined);
            assert.equal(row.dataset.rowAddress, undefined);
            assert.equal(body.hasAttribute('data-reply-body'), false);
            if (outcome === 'registry revokes') f.socket.fire('projects_changed', { chat_id: 7 });
            else if (outcome === 'older adoption') {
                const older = f.delayHistory(), button = f.instance.page.querySelector('.chat-load-older-btn');
                for (const click of button.listeners.get('click')) click();
                older([savedSpeech(early, 0)], { has_more: false, page_cursor: 'older' });
                await settle();
                assert.equal(bubble(), row, 'an older exact row adopts, never duplicates');
                assert.equal(row.dataset.rowAddress, early.row_address);
                assert.equal(row.dataset.historyId, 'saved-0');
                return;
            } else {
                document.getSelection = () => null; document.activeElement = null;
                if (outcome === 'focus ends') {
                    for (const release of f.messages.listeners.get('focusout') || []) release({ target: body });
                } else document.dispatchEvent(new Event('selectionchange'));
            }
            await settle();
            assert.equal(bubble(), undefined);
        } finally { f.close(); }
    });
}

test('only standalone local assistant words are eligible for window return', () => {
    const known = id => id === 'known-child';
    const root = { task_id: 'root', root_task_id: 'root' };
    for (const [note, extra, eligible] of [
        ['ordinary reply', {}, true], ['root speech', { ...root, system_type: 'proactive_message' }, true],
        ['typed root final', { ...root, task_terminal_status: 'completed' }, true],
        ['physical root retry', { ...root, task_id: 'retry', delegation_role: 'root' }, true],
        ['owner', { role: 'user' }, false], ['System', { role: 'system' }, false],
        ['progress', { is_progress: true }, false], ['narration', { narration: true }, false],
        ['placed review', { ...root, card_row: 'reviews' }, false], ['task summary', { ...root, system_type: 'task_summary' }, false],
        ['relay', { relay: {} }, false], ['origin projection', { origin_projected: true }, false],
        ['media', { msg_type: 'photo' }, false], ['attachment', { attachments: [{}] }, false],
        ['attachment placeholder', { text_placeholder: true }, false], ['connector attribution', { presence_contact: {} }, false],
        ['child speech', { ...root, task_id: 'child', delegation_role: 'subagent', system_type: 'proactive_message' }, false],
        ['known child', { task_id: 'known-child', root_task_id: 'known-child' }, false],
        ['unproved root retry', { ...root, task_id: 'retry' }, false],
    ]) assert.equal(isWindowText(speech(7, 1, 'some words', extra), known), eligible, note);
});
