import './fixtures/call_modules.js';
import test from 'node:test';
import assert from 'node:assert/strict';
const { createCallCards } = await import('../../skills/live_conversation/ui/call_card.js');
const { CallSession } = await import('../../skills/live_conversation/ui/call.js');
class FakeNode {
    constructor(tag, doc) {
        Object.assign(this, { tagName: tag.toUpperCase(), ownerDocument: doc, children: [], parentNode: null,
            dataset: {}, className: '', ownText: '', attributes: {}, listeners: {}, offsetTop: 0, offsetHeight: 0 });
    }
    get classList() {
        const names = () => this.className.split(/\s+/).filter(Boolean);
        return {
            contains: name => names().includes(name),
            add: name => { if (!names().includes(name)) this.className = [...names(), name].join(' '); },
            remove: name => { this.className = names().filter(other => other !== name).join(' '); },
        };
    }
    get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
    set textContent(value) { this.children = []; this.ownText = String(value); }
    get isConnected() { let node = this; while (node.parentNode) node = node.parentNode; return node === this.ownerDocument.feed; }
    get lastChild() { return this.children.at(-1) || null; }
    get previousElementSibling() { const kids = this.parentNode?.children || []; return kids[kids.indexOf(this) - 1] || null; }
    append(...nodes) { for (const node of nodes) { node.remove(); node.parentNode = this; this.children.push(node); } }
    replaceChildren(...nodes) { for (const child of [...this.children]) child.remove(); this.append(...nodes); }
    insertBefore(node, before) { node.remove(); node.parentNode = this; this.children.splice(this.children.indexOf(before), 0, node); }
    remove() { if (!this.parentNode) return; this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; }
    querySelector(selector) {
        const name = selector.replace(/^\./, '');
        for (const child of this.children) {
            if (child.classList.contains(name)) return child;
            const deeper = child.querySelector(selector);
            if (deeper) return deeper;
        }
        return null;
    }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    addEventListener(type, fn) { this.listeners[type] = fn; }
    click() { this.listeners.click?.(); }
}

function feedFixture({ live = () => 'c1' } = {}) {
    const doc = { createElement: tag => new FakeNode(tag, doc) };
    const feed = new FakeNode('div', doc);
    doc.feed = feed;
    Object.assign(feed, { scrollTop: 0, scrollHeight: 1000, clientHeight: 400 });
    const listeners = {};
    const ws = { on(type, fn) { (listeners[type] ||= new Set()).add(fn); return () => listeners[type].delete(fn); },
        emit(type, data) { for (const fn of listeners[type] || []) fn(data); }, listeners };
    let liveCall = live;
    const cards = createCallCards({ feed, ws, doc, live: () => liveCall() });
    const card = () => feed.children.find(node => node.className === 'call-card');
    const lines = () => card()?.querySelector('.call-card-lines').children.map(node => [node.className, node.textContent]) || [];
    // A room row as the chat mounts it: a bubble with its identity stamped.
    const row = (role, text, data = {}, ts = 0) => {
        const node = new FakeNode('div', doc);
        node.className = `chat-bubble ${role}`;
        const sender = new FakeNode('div', doc);
        sender.className = 'sender';
        sender.textContent = role === 'user' ? 'voice' : 'Ouroboros';
        const message = new FakeNode('div', doc);
        message.className = 'message';
        message.textContent = text;
        node.append(sender, message);
        Object.assign(node.dataset, data, ts ? { ts: String(ts) } : {});
        feed.append(node);
        return node;
    };
    return { doc, feed, ws, cards, card, lines, row, setLive: fn => { liveCall = fn; } };
}


test('R15: stopping capture preserves queued video accounting until the socket drains', () => {
    const session = new CallSession({ env: { clearInterval() {} } });
    session.state = 'live';
    session.socket = { readyState: 1, bufferedAmount: 300 * 1024, send() {} };
    session.vision.kind = 'screen';
    session.vision.pendingBytes = 300 * 1024;
    const failures = [];
    session.fail = code => failures.push(code);
    const frame = { type: 'capture', generation: session.captureGeneration, rate: 24000, pcm: new Int16Array(480) };
    session.onAudio(frame);
    assert.deepEqual(failures, [], 'an in-flight screenshot is not stalled audio');
    session.vision.stop();
    session.onAudio(frame);
    assert.deepEqual(failures, [], 'stopping capture does not remove bytes already queued in the same socket');
});

test('R15: meeting speech is a transcript, not a call nobody joined', () => {
    const { cards, card, row, doc } = feedFixture({ live: () => '' });
    row('system', 'The owner opened this call. Engine: Grok; no screen vision.',
        { clientMessageId: 'call:c1:open', taskId: 't1', systemType: 'call_open', rowAddress: 'row:1@t#1' });
    row('system', 'Heard in the call: The first participant speaks.', { taskId: 't1', systemType: 'call_heard' });
    row('system', 'Heard in the call: The second participant speaks.', { taskId: 't1', systemType: 'call_heard' });
    const ending = row('system', 'Call ended: no credits. Its transcript is in this room.', { taskId: 't1' });
    const quote = new FakeNode('div', doc);
    quote.className = 'chat-quote'; quote.dataset.address = 'row:1@t#1'; ending.append(quote);
    cards.scan();
    const title = card().querySelector('.call-card-title').textContent;
    assert.notEqual(card().dataset.callCard, 'silent', title);
    card().querySelector('.call-card-head').click();
    assert.equal(card().dataset.callOpen, '1');
    assert.match(card().querySelector('.call-card-lines').textContent, /second participant/);
    cards.destroy();
});
