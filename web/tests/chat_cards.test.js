// Living cards in the feed (chat_cards.js): one row live and on replay, never an empty box (the fallback text shows
// while the body loads and when it cannot be read), Keep on Widgets with Undo, Make it a skill and an app's submit
// into the owner's draft (never a send), card_state, at most three live frames, and the start ledger: a frame's start
// stands until its load and a second, a card held after a start the page did not outlive rests as its words with the
// reason and Open (which starts it without its saved state), Close keeps a card as its face, an app's errors show.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatCards, createFrameBudget, readCardBody } from '../modules/chat_cards.js';
import { createHolds, HELD_LINES, HOLDS_KEY } from '../modules/holds.js';
import { createChatMedia } from '../modules/chat_media.js';
import { messageRow } from '../modules/message_actions.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { installCardDom } from './helpers/card_dom.js';
import { joinFrame } from './helpers/fake_channel.js';

const flush = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const json = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body, text: async () => JSON.stringify(body) });
const page = (html) => ({ ok: true, status: 200, json: async () => { throw new Error('text'); }, text: async () => html });

function serve(routes) {
    const calls = [];
    const prior = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        const key = `${String(init.method || 'GET').toUpperCase()} ${url}`;
        calls.push(key);
        const route = routes[key];
        return (typeof route === 'function' ? route(init) : route) || json(404, { error: 'card not found', reason_code: 'card_not_found' });
    };
    return { calls, restore: () => { globalThis.fetch = prior; } };
}

function feed({ composer = null, holds = undefined, settleMs = 5 } = {}) {
    const frameNode = (msg, node) => {
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble assistant';
        bubble.append(node);
        return bubble;
    };
    return createChatCards({ frameNode, composer: () => composer, ...(holds ? { holds } : {}), settleMs });
}

// A ledger over one storage map, as the next page opening on this device would read it.
function ledgerStore() {
    const map = new Map();
    return { map, getItem: (key) => (map.has(key) ? map.get(key) : null), setItem: (key, value) => map.set(key, String(value)),
        removeItem: (key) => map.delete(key) };
}
const ledgerOn = (store) => createHolds({ win: { location: { href: 'http://local/' }, localStorage: store } });
const ledgerEntries = (store) => JSON.parse(store.map.get(HOLDS_KEY) || 'null')?.entries || {};

const identity = (id, kind, extra = {}) => ({ type: 'card', role: 'assistant', card: { card_id: id, kind, title: 'Plans', revision: 1 },
    fallback_text: 'Basic 9 · Pro 19', ts: '2026-10-17T10:00:00Z', ...extra });
const componentsBody = (id, extra = {}) => ({ card_id: id, kind: 'components', title: 'Plans', fallback_text: 'Basic 9 · Pro 19', revision: 1,
    pinned: false, components: [{ type: 'action', label: 'Choose Pro', submit: { id: 'pro' } }], submissions_summary: [], ...extra });

test('at most three frames live: near ones mount while there is room, far ones unmount, Open makes room', () => {
    const log = [];
    const budget = createFrameBudget({ max: 3, mount: (r) => log.push(`+${r}`), unmount: (r) => log.push(`-${r}`) });
    for (const record of ['a', 'b', 'c', 'd']) budget.near(record, true);
    assert.deepEqual(budget.live(), ['a', 'b', 'c']);
    budget.near('a', false);
    assert.deepEqual(budget.live(), ['b', 'c', 'd'], 'the far one unmounted, the waiting near one took its place');
    budget.open('e');
    assert.equal(budget.live().length, 3);
    assert.ok(budget.live().includes('e'));
    budget.forget('e');
    assert.ok(!budget.live().includes('e'));
    assert.deepEqual(log.slice(0, 5), ['+a', '+b', '+c', '-a', '+d']);
});

test('a near frame the ledger holds waits for Open; Open still mounts it', () => {
    const log = [];
    const held = new Set(['b']);
    const budget = createFrameBudget({ max: 3, mount: (r) => log.push(`+${r}`), unmount: (r) => log.push(`-${r}`),
        canStart: (record) => !held.has(record) });
    for (const record of ['a', 'b', 'c']) budget.near(record, true);
    assert.deepEqual(budget.live(), ['a', 'c'], 'the held one does not start by itself');
    budget.open('b');
    assert.deepEqual(budget.live(), ['a', 'c', 'b'], 'the owner\'s Open starts it');
});

test('a card body is read once per revision; a failed read is tried again', async () => {
    let reads = 0;
    const read = async () => { reads += 1; if (reads === 3) throw new Error('down'); return { revision: reads === 1 ? 1 : 2 }; };
    await readCardBody('card_cache1', 1, read);
    await readCardBody('card_cache1', 1, read);
    assert.equal(reads, 1);
    await readCardBody('card_cache1', 2, read);
    assert.equal(reads, 2, 'a newer revision is read again');
    await assert.rejects(readCardBody('card_cache1', 3, read));
    await readCardBody('card_cache1', 3, read);
    assert.equal(reads, 4, 'a failed read is not cached');
});

test('the fallback text shows while the body loads; a component card then draws with Keep on Widgets, then Undo', async () => {
    const dom = installCardDom();
    let pinned = false;
    const server = serve({
        'GET /api/cards/card_k1': () => json(200, componentsBody('card_k1', { pinned })),
        'POST /api/cards/card_k1/pin': () => { pinned = true; return json(200, { ok: true, card_id: 'card_k1', pinned: true }); },
        'DELETE /api/cards/card_k1/pin': () => { pinned = false; return json(200, { ok: true, card_id: 'card_k1', pinned: false }); },
    });
    try {
        const cards = feed();
        const bubble = cards.buildCard(identity('card_k1', 'components'));
        assert.ok(bubble.classList.contains('chat-card-bubble'));
        assert.equal(bubble.dataset.cardId, 'card_k1');
        assert.equal(bubble.querySelector('.chat-card-fallback').textContent, 'Basic 9 · Pro 19', 'never an empty box');
        assert.equal(bubble.querySelector('.chat-card-status').textContent, 'Loading the card…');
        assert.equal(bubble.querySelector('.chat-card-made').textContent, 'Ouroboros made this');
        await flush();
        assert.equal(bubble.querySelector('.chat-card-fallback'), null);
        assert.equal(bubble.querySelector('[data-widget-action]').textContent, 'Choose Pro');
        assert.equal(bubble.querySelector('.chat-card-status').hidden, true);
        bubble.querySelector('[data-card-pin]').click();
        await flush();
        assert.deepEqual(server.calls.filter((c) => c.includes('/pin')), ['POST /api/cards/card_k1/pin']);
        assert.match(bubble.querySelector('[data-card-actions]').textContent, /Kept on Widgets/);
        assert.equal(bubble.querySelector('[data-card-unpin]').textContent, 'Undo');
        bubble.querySelector('[data-card-unpin]').click();
        await flush();
        assert.equal(server.calls.at(-1), 'DELETE /api/cards/card_k1/pin');
        assert.equal(bubble.querySelector('[data-card-pin]').textContent, 'Keep on Widgets');
        // The kept state from another window arrives as extension_lifecycle for skill "_cards".
        cards.lifecycle({ skill: '_cards', action: 'card_pinned', data: { card_id: 'card_k1' } });
        assert.equal(bubble.querySelector('[data-card-unpin]').textContent, 'Remove');
        cards.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('a card that cannot be read keeps its words: Retry after a failure, a plain note when it is gone', async () => {
    const dom = installCardDom();
    let failing = true;
    const server = serve({
        'GET /api/cards/card_f1': () => (failing ? json(503, { error: 'store migrating' }) : json(200, componentsBody('card_f1'))),
    });
    try {
        const cards = feed();
        const broken = cards.buildCard(identity('card_f1', 'components'));
        await flush();
        assert.equal(broken.querySelector('.chat-card-fallback').textContent, 'Basic 9 · Pro 19');
        assert.match(broken.querySelector('.chat-card-status').textContent, /^The card could not be loaded: store migrating/);
        assert.equal(broken.querySelector('.chat-card-status').dataset.tone, 'danger');
        failing = false;
        broken.querySelector('[data-card-retry]').click();
        await flush();
        assert.equal(broken.querySelector('[data-widget-action]').textContent, 'Choose Pro');

        const gone = cards.buildCard(identity('card_g1', 'components'));
        await flush();
        assert.equal(gone.querySelector('.chat-card-fallback').textContent, 'Basic 9 · Pro 19');
        assert.equal(gone.querySelector('.chat-card-status').textContent, 'This card is no longer available.');
        assert.equal(gone.querySelector('[data-card-retry]'), null);
        cards.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('an app shows its face with Open; open, its submit and Make it a skill fill the draft on their own line', async () => {
    const dom = installCardDom();
    const server = serve({
        'GET /api/cards/card_a1': json(200, { card_id: 'card_a1', kind: 'app', title: 'Loan', fallback_text: 'A loan calculator', revision: 1,
            app: { sha256: 'f00d', size: 12, url: '/api/cards/card_a1/app' }, submissions_summary: [] }),
        'GET /api/cards/card_a1/app': page('<p>calc</p>'),
    });
    try {
        const inserted = [];
        const composer = { input: { value: 'Hello', selectionStart: 5 }, insertAtCursor: (text) => inserted.push(text) };
        const cards = feed({ composer });
        const bubble = cards.buildCard(identity('card_a1', 'app', { card: { card_id: 'card_a1', kind: 'app', title: 'Loan', revision: 1 },
            fallback_text: 'A loan calculator' }));
        await flush();
        assert.equal(bubble.querySelector('.chat-card-face').querySelector('.chat-card-fallback').textContent, 'A loan calculator');
        assert.equal(bubble.querySelector('iframe'), null, 'no frame before Open (no viewport observer here)');
        bubble.querySelector('[data-card-open]').click();
        await flush();
        const frame = bubble.querySelector('iframe');
        assert.equal(frame.getAttribute('sandbox'), 'allow-scripts');
        assert.ok(server.calls.includes('GET /api/cards/card_a1/app'));
        frame.contentWindow = {};
        const nonce = frame.srcdoc.match(/const nonce = "([^"]+)"/)[1];
        window.emit('message', { source: frame.contentWindow, data: { type: 'ouro-card-submit', value: { monthly: 1520 }, summary: 'Monthly', nonce } });
        assert.deepEqual(inserted, ['\nFrom «Loan»: Monthly {"monthly":1520}']);
        bubble.querySelector('[data-card-make-skill]').click();
        assert.equal(inserted[1], '\nTurn the card «Loan» into a skill so I can keep it on Widgets.');
        composer.input = { value: '', selectionStart: 0 };
        bubble.querySelector('[data-card-make-skill]').click();
        assert.equal(inserted[2], 'Turn the card «Loan» into a skill so I can keep it on Widgets.', 'an empty draft takes the line as is');
        cards.release(bubble);
        assert.equal(bubble.querySelector('iframe'), null, 'released: the frame is gone');
        assert.equal(window.listeners('message'), 0);
    } finally {
        server.restore();
        dom.restore();
    }
});

test('card_state: recorded presses update the drawn card in place; a newer revision is read again', async () => {
    const dom = installCardDom();
    let revision = 1;
    const server = serve({ 'GET /api/cards/card_s1': () => json(200, componentsBody('card_s1', {
        revision, components: [{ type: 'action', label: revision === 1 ? 'Choose Pro' : 'Choose Max', submit: { id: 'pro', once: true } }],
    })) });
    try {
        const cards = feed();
        const bubble = cards.buildCard(identity('card_s1', 'components'));
        await flush();
        const reads = () => server.calls.filter((c) => c === 'GET /api/cards/card_s1').length;
        assert.equal(reads(), 1);
        cards.applyCardState({ type: 'card_state', card: { card_id: 'card_s1', kind: 'components', title: 'Plans', revision: 1 },
            submissions_summary: [{ action_id: 'pro', label: 'Choose Pro', at: 't', delivery: 'delivered' }], submissions_dropped: 0 });
        assert.equal(reads(), 1, 'a press needs no read');
        assert.equal(bubble.querySelector('[data-widget-action]').disabled, true, 'its once control counted');
        revision = 2;
        assert.equal(cards.applyCardState({ type: 'card_state', card: { card_id: 'card_s1', revision: 2 }, submissions_summary: [] }), true);
        await flush();
        assert.equal(reads(), 2);
        assert.equal(bubble.querySelector('[data-widget-action]').textContent, 'Choose Max');
        assert.equal(cards.applyCardState({ card: { card_id: 'card_unknown', revision: 9 } }), false, 'a card this room does not show');
        // A change missed while the socket was down is read when it opens again; the same body stays drawn.
        const action = bubble.querySelector('[data-widget-action]');
        cards.reconnect();
        await flush();
        assert.equal(reads(), 3);
        assert.equal(bubble.querySelector('[data-widget-action]'), action, 'the same body is not drawn again');
        revision = 3;
        cards.reconnect();
        await flush();
        assert.equal(reads(), 4);
        assert.notEqual(bubble.querySelector('[data-widget-action]'), action, 'a newer body is drawn');
        cards.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('overlapping reads: an older answer that settles after a newer one never draws the older card back', async () => {
    const dom = installCardDom();
    const held = [];
    let calls = 0;
    const body = (revision) => json(200, componentsBody('card_o1', { revision,
        components: [{ type: 'action', label: revision === 1 ? 'Choose Pro' : 'Choose Max', submit: { id: 'pro' } }] }));
    const server = serve({ 'GET /api/cards/card_o1': () => {
        calls += 1;
        return calls === 1 ? new Promise((resolve) => held.push(() => resolve(body(1)))) : body(2);
    } });
    try {
        const cards = feed();
        const bubble = cards.buildCard(identity('card_o1', 'components'));
        await flush();
        cards.applyCardState({ type: 'card_state', card: { card_id: 'card_o1', revision: 2 }, submissions_summary: [] });
        await flush();
        assert.equal(bubble.querySelector('[data-widget-action]').textContent, 'Choose Max', 'the newer card is drawn');
        held.shift()();
        await flush();
        assert.equal(bubble.querySelector('[data-widget-action]').textContent, 'Choose Max', 'the late older answer is dropped');
        assert.equal(calls, 2);
        cards.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('a widget card whose skill is off keeps its words and says so', async () => {
    const dom = installCardDom();
    const server = serve({
        'GET /api/cards/card_w1': json(200, { card_id: 'card_w1', kind: 'widget', title: 'Weather', fallback_text: 'Sunny', revision: 1,
            widget: { skill: 'weather', tab_id: 'now' }, submissions_summary: [] }),
        'GET /api/widgets': json(200, { ui_tabs: [{ key: 'other:tab', skill: 'other', tab_id: 'tab', render: { kind: 'declarative', components: [] } }] }),
    });
    try {
        const cards = feed();
        const bubble = cards.buildCard(identity('card_w1', 'widget', { card: { card_id: 'card_w1', kind: 'widget', title: 'Weather', revision: 1 },
            fallback_text: 'Sunny' }));
        await flush();
        assert.equal(bubble.querySelector('.chat-card-fallback').textContent, 'Sunny');
        assert.equal(bubble.querySelector('.chat-card-status').textContent, 'This skill is off.');
        assert.equal(bubble.querySelector('[data-card-actions]').textContent, '', 'a skill\'s widget has nothing to keep');
        cards.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('a classic widget frame survives a same-bytes republication; an ESM card follows its publication', async () => {
    const dom = installCardDom();
    let tab = { key: 'clock:face', skill: 'clock', tab_id: 'face', revision: '1', generation: 'g1',
        render: { kind: 'module', entry: 'widget.js' } };
    const server = serve({
        'GET /api/cards/card_p1': json(200, { card_id: 'card_p1', kind: 'widget', title: 'Clock', fallback_text: '10:00', revision: 1,
            widget: { skill: 'clock', tab_id: 'face' }, submissions_summary: [] }),
        'GET /api/widgets': () => json(200, { ui_tabs: [tab] }),
        'GET /api/extensions/clock/module/widget.js': page('// reviewed module'),
        // A classic frame's elements load its routes through the mount's route lease.
        'POST /api/widgets/asset-leases': json(200, { token: 'route-lease', base_url: '/api/widget-resources/route-lease/', expires_in: 120, renew_after: 40 }),
    });
    try {
        const cards = feed();
        const republish = async (change) => {
            tab = { ...tab, ...change };
            cards.lifecycle({ skill: 'clock', action: 'enabled' });  // the shared widget list is read fresh
            await flush();
        };
        cards.lifecycle({ skill: 'clock', action: 'enabled' });
        const bubble = cards.buildCard(identity('card_p1', 'widget', { card: { card_id: 'card_p1', kind: 'widget', title: 'Clock', revision: 1 }, fallback_text: '10:00' }));
        await flush();
        bubble.querySelector('[data-card-open]').click();
        await flush();
        const frame = bubble.querySelector('iframe');
        assert.ok(frame, 'the classic frame runs');
        await republish({ generation: 'g2' });
        assert.equal(bubble.querySelector('iframe'), frame, 'same revision, new publication: the classic frame keeps running');
        await republish({ render: { kind: 'module', entry_type: 'module', entry: 'app/main.js' } });
        const face = bubble.querySelector('.chat-card-face');
        assert.ok(face && !bubble.querySelector('iframe'), 'a changed declaration draws the card again');
        await republish({ generation: 'g3' });
        assert.notEqual(bubble.querySelector('.chat-card-face'), face, 'an ESM card is bound to its publication');
        cards.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('the live card frame and its history row are one row; replay draws it before final handling', async () => {
    const dom = installCardDom();
    const server = serve({ 'GET /api/cards/card_r1': json(200, componentsBody('card_r1')) });
    try {
        const root = document.createElement('div');
        const media = createChatMedia({
            chatSessionId: 's', durableChatMediaUrl: () => '', formatMsgTime: () => null, senderLabel: () => 'Owner',
            insertMessageNode: (node) => root.append(node), stampNodeTimestamp,
        });
        const handlers = {};
        const seen = new Set();
        const delivery = media.wireDeliveries({
            onWs: (type, fn) => { handlers[type] = fn; }, isMyThread: () => true, hideTypingIndicatorOnly() {}, syncChatStatus() {},
            seenMessageKeys: seen, rememberMessageKey: (key) => seen.add(key),
            chatMediaMessageKey: () => '', documentMessageKey: () => '', messagesRoot: () => root,
        });
        assert.equal(typeof handlers.card, 'function');
        assert.equal(typeof handlers.card_state, 'function');
        handlers.card(identity('card_r1', 'components', { row_address: 'chat:1:abc' }));
        assert.equal(root.children.length, 1);
        // A new card is a message the owner has not read: the server counts it (card_delivery.py feeds its room).
        assert.equal(root.children[0].dataset.rowAddress, 'chat:1:abc');
        assert.equal(messageRow(root.children[0]).text, 'Basic 9 · Pro 19', 'a message action reads the card\'s words');
        const row = { ...identity('card_r1', 'components'), type: undefined, msg_type: 'card', system_type: 'card', text: 'Basic 9 · Pro 19',
            history_id: 'chat:7', history_position: { source: 'chat', offset: 700 } };
        assert.equal(delivery.appendDelivery(row), true, 'a card row is a delivery');
        assert.equal(root.children.length, 1, 'the same card: one row');
        assert.equal(root.children[0].dataset.historyId, 'chat:7');
        assert.equal(delivery.appendDelivery({ msg_type: undefined, text: 'hi' }), false, 'an ordinary row is not');
        await flush();
        media.release(root.children[0]);
        handlers.card_state({ card: { card_id: 'card_r1', revision: 1 }, submissions_summary: [] });
        media.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('a replaced card shows its new title and words: heading, label, Copy, a message action, its face, Make it a skill', async () => {
    const dom = installCardDom();
    let revision = 1;
    const server = serve({ 'GET /api/cards/card_n1': () => json(200, revision === 1 ? componentsBody('card_n1')
        : { card_id: 'card_n1', kind: 'app', title: 'Prices', fallback_text: 'Basic 10 · Pro 20', revision: 2,
            app: { sha256: 'beef', size: 9, url: '/api/cards/card_n1/app' }, submissions_summary: [] }) });
    const copied = [];
    const priorNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { clipboard: { writeText: async (text) => { copied.push(text); } } } });
    try {
        const inserted = [];
        const composer = { input: { value: '', selectionStart: 0 }, insertAtCursor: (text) => inserted.push(text) };
        const root = document.createElement('div');
        const media = createChatMedia({
            chatSessionId: 's', durableChatMediaUrl: () => '', formatMsgTime: () => null, senderLabel: () => 'Owner',
            insertMessageNode: (node) => root.append(node), stampNodeTimestamp, composer: () => composer,
        });
        const handlers = {};
        media.wireDeliveries({
            onWs: (type, fn) => { handlers[type] = fn; }, isMyThread: () => true, hideTypingIndicatorOnly() {}, syncChatStatus() {},
            seenMessageKeys: new Set(), rememberMessageKey() {},
            chatMediaMessageKey: () => '', documentMessageKey: () => '', messagesRoot: () => root,
        });
        handlers.card(identity('card_n1', 'components', { row_address: 'chat:1:n1' }));
        await flush();
        const bubble = root.children[0];
        assert.equal(bubble.querySelector('.chat-card-title').textContent, 'Plans');
        revision = 2;
        handlers.card_state({ type: 'card_state', card: { card_id: 'card_n1', kind: 'app', title: 'Prices', revision: 2 }, submissions_summary: [] });
        await flush();
        assert.equal(bubble.querySelector('.chat-card-title').textContent, 'Prices', 'the new title, not only the new body');
        assert.equal(bubble.querySelector('.chat-card').getAttribute('aria-label'), 'Prices');
        assert.equal(bubble.querySelector('.chat-card-face').querySelector('.chat-card-fallback').textContent, 'Basic 10 · Pro 20');
        assert.equal(messageRow(bubble).text, 'Basic 10 · Pro 20', 'Reply and message actions read the new words');
        bubble.querySelector('.chat-message-copy').click();
        await flush();
        assert.deepEqual(copied, ['Basic 10 · Pro 20'], 'Copy takes the new words');
        bubble.querySelector('[data-card-make-skill]').click();
        assert.deepEqual(inserted, ['Turn the card «Prices» into a skill so I can keep it on Widgets.']);
        media.destroy();
    } finally {
        if (priorNavigator) Object.defineProperty(globalThis, 'navigator', priorNavigator); else delete globalThis.navigator;
        server.restore();
        dom.restore();
    }
});

test('a stopped widget frame cannot size or fault the frame that replaced it; the live frame still sizes itself', async () => {
    const dom = installCardDom();
    let revision = 1;
    const server = serve({
        'GET /api/cards/card_m1': () => json(200, { card_id: 'card_m1', kind: 'widget', title: 'Clock', fallback_text: '10:00', revision,
            widget: { skill: 'clock', tab_id: 'face' }, submissions_summary: [] }),
        'GET /api/widgets': json(200, { ui_tabs: [{ key: 'clock:face', skill: 'clock', tab_id: 'face', revision: '1',
            render: { kind: 'module', entry: 'widget.js' } }] }),
        'GET /api/extensions/clock/module/widget.js': page('// reviewed module'),
        // A classic frame's elements load its routes through the mount's route lease.
        'POST /api/widgets/asset-leases': json(200, { token: 'route-lease', base_url: '/api/widget-resources/route-lease/', expires_in: 120, renew_after: 40 }),
    });
    try {
        const cards = feed();
        cards.lifecycle({ skill: 'clock', action: 'enabled' });  // the shared widget list is read fresh
        const bubble = cards.buildCard(identity('card_m1', 'widget', { card: { card_id: 'card_m1', kind: 'widget', title: 'Clock', revision: 1 }, fallback_text: '10:00' }));
        await flush();
        const open = async () => {
            bubble.querySelector('[data-card-open]').click();
            await flush();
            const frame = bubble.querySelector('iframe');
            frame.contentWindow = { postMessage() {} };  // a live child: its stop waits for the acknowledgement
            const { send } = joinFrame(frame, (event) => window.emit('message', event));
            return { host: bubble.querySelector('.chat-card-frame-host'), send };
        };
        const height = (host) => host.style.getPropertyValue('--widget-frame-height');
        const first = await open();
        first.send({ type: 'ouro-widget-resize', height: 300 });
        assert.equal(height(first.host), '302px', 'a live frame sizes itself');
        revision = 2;
        cards.applyCardState({ type: 'card_state', card: { card_id: 'card_m1', kind: 'widget', title: 'Clock', revision: 2 }, submissions_summary: [] });
        await flush();
        const second = await open();
        assert.notEqual(second.host, first.host);
        assert.equal(height(second.host), '120px');
        // The first frame is still in its ordered stop and keeps answering its child.
        first.send({ type: 'ouro-widget-resize', height: 900 });
        first.send({ type: 'ouro-widget-error', kind: 'script', message: 'boom' });
        assert.equal(height(second.host), '120px', 'the stopped frame did not size the new one');
        assert.equal(bubble.querySelector('.chat-card-expand').hidden, true, 'nor show the shared Expand');
        assert.equal(bubble.querySelector('.chat-card-status').hidden, true, 'nor paint its fault on the new frame');
        second.send({ type: 'ouro-widget-resize', height: 300 });
        assert.equal(height(second.host), '302px', 'the live frame still sizes itself');
        first.send({ type: 'ouro-widget-disposed' });
        cards.destroy();
        second.send({ type: 'ouro-widget-disposed' });
        assert.equal(window.listeners('message'), 0);
    } finally {
        server.restore();
        dom.restore();
    }
});

test('overlapping draws: an earlier draw that resolves late mounts nothing, and only the latest widget stays mounted', async () => {
    const dom = installCardDom();
    const revisions = { card_d1: 1, card_d2: 1 };
    const widgetBody = (id) => ({ card_id: id, kind: 'widget', title: 'Weather', fallback_text: 'Sunny', revision: revisions[id],
        widget: { skill: 'weather', tab_id: 'now' }, submissions_summary: [] });
    let answerWidgets = null;
    const server = serve({
        'GET /api/cards/card_d1': () => json(200, revisions.card_d1 === 1 ? widgetBody('card_d1') : componentsBody('card_d1', { revision: 2 })),
        'GET /api/cards/card_d2': () => json(200, widgetBody('card_d2')),
        'GET /api/widgets': () => new Promise((resolve) => { answerWidgets = () => resolve(json(200, { ui_tabs: [{ key: 'weather:now',
            skill: 'weather', tab_id: 'now', revision: '1', render: { kind: 'declarative', components: [{ type: 'callout', text: 'Sunny' }] } }] })); }),
    });
    try {
        const cards = feed();
        cards.lifecycle({ skill: 'weather', action: 'enabled' });  // the shared widget list is read fresh (and slowly)
        const replaced = cards.buildCard(identity('card_d1', 'widget'));
        const redrawn = cards.buildCard(identity('card_d2', 'widget'));
        await flush();  // both first draws wait for the widget list
        revisions.card_d1 = 2;
        revisions.card_d2 = 2;
        cards.applyCardState({ type: 'card_state', card: { card_id: 'card_d1', kind: 'components', title: 'Plans', revision: 2 }, submissions_summary: [] });
        cards.applyCardState({ type: 'card_state', card: { card_id: 'card_d2', kind: 'widget', title: 'Weather', revision: 2 }, submissions_summary: [] });
        await flush();
        assert.equal(replaced.querySelector('[data-widget-action]').textContent, 'Choose Pro', 'the replacement drew at once');
        answerWidgets();
        await flush();
        assert.equal(replaced.querySelector('.chat-card-widget'), null, 'the earlier widget draw resolved late and drew nothing');
        assert.equal(replaced.querySelector('[data-widget-action]').textContent, 'Choose Pro', 'the later draw won');
        assert.equal(redrawn.querySelector('.chat-card-widget').textContent, 'Sunny');
        assert.equal(window.listeners('resize'), 1, 'one widget mounted, under one size policy');
        assert.equal(redrawn.querySelector('.chat-card-expand').listenerCount(), 1);
        cards.destroy();
        assert.equal(window.listeners('resize'), 0, 'nothing left mounted');
    } finally {
        server.restore();
        dom.restore();
    }
});

const appBody = (id, sha = 'f00d') => json(200, { card_id: id, kind: 'app', title: 'Cats', fallback_text: 'A cat calculator', revision: 1,
    app: { sha256: sha, size: 12, url: `/api/cards/${id}/app` }, submissions_summary: [] });
const appIdentity = (id) => identity(id, 'app', { card: { card_id: id, kind: 'app', title: 'Cats', revision: 1 }, fallback_text: 'A cat calculator' });
const loadFrame = (frame) => frame.listeners?.get?.('load')?.forEach((handler) => handler({ type: 'load' }));

test('an app\'s start stands in the ledger until its frame loaded and a second passed; Close keeps it as its face', async () => {
    const dom = installCardDom();
    const server = serve({ 'GET /api/cards/card_l1': appBody('card_l1'), 'GET /api/cards/card_l1/app': page('<p>cats</p>') });
    try {
        const store = ledgerStore();
        const cards = feed({ holds: ledgerOn(store) });
        const bubble = cards.buildCard(appIdentity('card_l1'));
        await flush();
        assert.equal(bubble.querySelector('[data-card-close]'), null, 'no Close before a frame runs');
        bubble.querySelector('[data-card-open]').click();
        await flush();
        assert.deepEqual(ledgerEntries(store)['app:card_l1'], { reason: 'start', rev: 'f00d', at: ledgerEntries(store)['app:card_l1'].at, n: 1 },
            'written before the app\'s first byte runs');
        assert.equal(bubble.querySelector('[data-card-close]').textContent, 'Close');
        loadFrame(bubble.querySelector('iframe'));
        await new Promise((resolve) => setTimeout(resolve, 15));
        assert.equal(ledgerEntries(store)['app:card_l1'], undefined, 'the page outlived the start');
        bubble.querySelector('[data-card-close]').click();
        await flush();
        assert.equal(bubble.querySelector('iframe'), null, 'Close stops the frame');
        assert.ok(bubble.querySelector('[data-card-open]'));
        assert.equal(bubble.querySelector('[data-card-close]'), null);
        assert.equal(bubble.querySelector('.chat-card-status').hidden, true, 'a closed card needs no reason line');
        assert.equal(ledgerEntries(store)['card:card_l1'].reason, 'owner', 'kept on this device');
        cards.destroy();
        const reopened = feed({ holds: ledgerOn(store) });
        const again = reopened.buildCard(appIdentity('card_l1'));
        await flush();
        assert.ok(again.querySelector('.chat-card-face'), 'the next page opens it as its face');
        again.querySelector('[data-card-open]').click();
        await flush();
        assert.ok(again.querySelector('iframe'), 'Open starts it again');
        assert.equal(ledgerEntries(store)['card:card_l1'], undefined, 'and ends the owner\'s Close');
        reopened.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('an app the page did not outlive rests as its words with the reason; Open starts it without its saved state', async () => {
    const dom = installCardDom();
    const server = serve({ 'GET /api/cards/card_h1': appBody('card_h1', 'beef'), 'GET /api/cards/card_h1/app': page('<p>cats</p>') });
    try {
        const store = ledgerStore();
        store.setItem(HOLDS_KEY, JSON.stringify({ v: 1, entries: { 'app:card_h1': { reason: 'start', rev: 'beef', at: 1 } } }));
        dom.storage.setItem('ouro_card_state:card_h1:beef', '{"display":"7"}');
        const cards = feed({ holds: ledgerOn(store) });
        const bubble = cards.buildCard(appIdentity('card_h1'));
        await flush();
        assert.equal(bubble.querySelector('.chat-card-face').querySelector('.chat-card-fallback').textContent, 'A cat calculator');
        assert.equal(bubble.querySelector('.chat-card-status').textContent, HELD_LINES.start);
        assert.equal(bubble.querySelector('.chat-card-status').hidden, false);
        bubble.querySelector('[data-card-open]').click();
        await flush();
        const frame = bubble.querySelector('iframe');
        assert.match(frame.srcdoc, /let saved = null;/, 'the state it saved may be what stopped the page');
        assert.equal(dom.storage.getItem('ouro_card_state:card_h1:beef'), null, 'and is forgotten: no later mount hands it back');
        assert.equal(bubble.querySelector('.chat-card-status').hidden, true);
        cards.destroy();
        dom.storage.setItem('ouro_card_state:card_h1:beef', '{"display":"7"}');
        const unheld = feed({ holds: ledgerOn(ledgerStore()) });
        const plain = unheld.buildCard(appIdentity('card_h1'));
        await flush();
        assert.equal(plain.querySelector('.chat-card-status').hidden, true, 'nothing held: no line');
        plain.querySelector('[data-card-open]').click();
        await flush();
        assert.match(plain.querySelector('iframe').srcdoc, /let saved = "\{\\"display\\":\\"7\\"\}";/, 'an ordinary open hands its state back');
        unheld.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});

test('an app\'s script error shows on its card, once per message', async () => {
    const dom = installCardDom();
    const server = serve({ 'GET /api/cards/card_e1': appBody('card_e1'), 'GET /api/cards/card_e1/app': page('<p>cats</p>') });
    try {
        const cards = feed({ holds: ledgerOn(ledgerStore()) });
        const bubble = cards.buildCard(appIdentity('card_e1'));
        await flush();
        bubble.querySelector('[data-card-open]').click();
        await flush();
        const frame = bubble.querySelector('iframe');
        assert.match(frame.srcdoc, /addEventListener\('unhandledrejection'/, 'the bridge listens for the app\'s errors');
        frame.contentWindow = {};
        const nonce = frame.srcdoc.match(/const nonce = "([^"]+)"/)[1];
        window.emit('message', { source: frame.contentWindow, data: { type: 'ouro-card-error', kind: 'error', message: 'x is not defined', nonce } });
        const status = bubble.querySelector('.chat-card-status');
        assert.equal(status.textContent, 'App error: x is not defined');
        assert.equal(status.dataset.tone, 'danger');
        window.emit('message', { source: {}, data: { type: 'ouro-card-error', kind: 'error', message: 'forged', nonce } });
        assert.equal(status.textContent, 'App error: x is not defined', 'only this frame\'s window speaks for it');
        cards.destroy();
    } finally {
        server.restore();
        dom.restore();
    }
});
