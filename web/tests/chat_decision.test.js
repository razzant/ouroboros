import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatDecision } from '../modules/chat_decision.js';

class Classes {
    constructor() { this.values = new Set(); }
    set(value) { this.values = new Set(String(value || '').split(/\s+/).filter(Boolean)); }
    add(...values) { values.forEach((value) => this.values.add(value)); }
    contains(value) { return this.values.has(value); }
    toggle(value, force) {
        const enabled = force === undefined ? !this.contains(value) : Boolean(force);
        if (enabled) this.add(value); else this.values.delete(value);
        return enabled;
    }
}

class NodeStub {
    constructor(tag = 'div') {
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.dataset = {};
        this.classList = new Classes();
        this.disabled = false;
        this.listeners = new Map();
        this.type = '';
        this._text = '';
    }
    set className(value) { this.classList.set(value); }
    get className() { return [...this.classList.values].join(' '); }
    set textContent(value) { this._text = String(value ?? ''); }
    get textContent() { return this._text; }
    append(...nodes) { nodes.forEach((node) => { node.parentNode = this; this.children.push(node); }); }
    remove() {
        const parent = this.parentNode;
        if (!parent) return;
        const i = parent.children.indexOf(this);
        if (i >= 0) parent.children.splice(i, 1);
        this.parentNode = null;
    }
    before(node) {
        const parent = this.parentNode;
        if (!parent) return;
        node.parentNode = parent;
        parent.children.splice(parent.children.indexOf(this), 0, node);
    }
    addEventListener(type, handler) { this.listeners.set(type, handler); }
    click() { const handler = this.listeners.get('click'); if (handler) handler(); }
    matchesClass(name) { return this.classList.contains(name); }
    collect(name, out = []) {
        if (this.matchesClass(name)) out.push(this);
        this.children.forEach((child) => child.collect(name, out));
        return out;
    }
    querySelector(selector) { return this.collect(selector.replace(/^\./, ''))[0] || null; }
    querySelectorAll(selector) { return this.collect(selector.replace(/^\./, '')); }
}

function countPropertyWrites(target, key) {
    let value = target[key];
    let writes = 0;
    Object.defineProperty(target, key, {
        configurable: true,
        get: () => value,
        set: (next) => { writes += 1; value = next; },
    });
    return () => writes;
}

function fixture({ fetchImpl, renderMarkdown, onDomWrite } = {}) {
    const prior = { document: globalThis.document, crypto: globalThis.crypto };
    globalThis.document = { createElement: (tag) => new NodeStub(tag) };
    if (!globalThis.crypto || !globalThis.crypto.randomUUID) {
        Object.defineProperty(globalThis, 'crypto', {
            configurable: true, value: { randomUUID: () => 'fixed-request-id' },
        });
    }
    const toasts = [];
    const calls = [];
    const decision = createChatDecision({
        apiFetch: async (url, init) => {
            calls.push({ url, init });
            if (fetchImpl) return fetchImpl(url, init);
            return { ok: true, status: 200 };
        },
        frameNode: (_msg, node) => node,
        renderMarkdown,
        enhanceMarkdown: renderMarkdown ? () => {} : null,
        showToast: (text, tone) => toasts.push({ text, tone }),
        onDomWrite,
    });
    return { decision, toasts, calls, restore: () => {
        globalThis.document = prior.document;
        Object.defineProperty(globalThis, 'crypto', { configurable: true, value: prior.crypto });
    } };
}

const WS_MSG = {
    type: 'quiz', role: 'assistant', quiz_id: 'qz-1', task_id: 't-1',
    question: 'Merge now?', stake: 'release timing',
    assumption: 'continuing with the merge', state: 'open',
    options: [{ label: 'Yes' }, { label: 'No', detail: 'wait for CI' }],
    ts: '2026-08-31T10:00:00Z',
};

test('quiz card renders full anatomy from a WS frame', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        assert.ok(card);
        assert.equal(card.dataset.state, 'open');
        assert.equal(card.querySelector('.chat-quiz-question').textContent, 'Merge now?');
        assert.match(card.querySelector('.chat-quiz-stake').textContent, /At stake: release timing/);
        assert.match(card.querySelector('.chat-quiz-assumption').textContent, /Continuing meanwhile: continuing with the merge/);
        assert.equal(card.querySelector('.chat-quiz-status-text').textContent, 'Awaiting answer');
        const buttons = card.querySelectorAll('.chat-quiz-option');
        assert.equal(buttons.length, 2);
        assert.equal(buttons[1].querySelector('.chat-quiz-option-detail').textContent, 'wait for CI');
        assert.ok(buttons.every((btn) => !btn.disabled));
    } finally { fx.restore(); }
});

test('quiz card renders the replay shape and settled states disable buttons', () => {
    const fx = fixture();
    try {
        const replayMsg = {
            msg_type: 'quiz', role: 'assistant', task_id: 't-1',
            text: 'Merge now?', ts: 'x',
            quiz: {
                quiz_id: 'qz-2', state: 'expired_terminal',
                options: [{ label: 'Yes' }, { label: 'No' }],
                stake: '', assumption: 'merging meanwhile',
            },
        };
        const card = fx.decision.buildQuizCard(replayMsg);
        assert.ok(card);
        assert.equal(card.dataset.state, 'expired_terminal');
        assert.match(card.querySelector('.chat-quiz-status-text').textContent, /question expired/);
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => btn.disabled));
        // The assumption line survives settlement: it is the record of the path taken.
        assert.match(card.querySelector('.chat-quiz-assumption').textContent, /merging meanwhile/);
    } finally { fx.restore(); }
});

test('an accepted answer marks the chosen option; degenerate cards refuse to render', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
        assert.equal(fx.calls[0].url, '/api/decisions');
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal(body.decision_id, 'quiz:t-1:qz-1');
        assert.equal(body.option_index, 1);
        assert.ok(body.request_id);
        assert.equal(card.dataset.state, 'answered');
        const buttons = card.querySelectorAll('.chat-quiz-option');
        assert.ok(buttons[1].classList.contains('chosen'));
        assert.ok(buttons.every((btn) => btn.disabled));

        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, options: [{ label: 'only' }] }), null);
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: '' }), null);
        // An anonymous quiz has no answer address: refuse to render buttons.
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, task_id: '' }), null);
    } finally { fx.restore(); }
});

test('one corrupt option refuses that card only, preserving index integrity', () => {
    const fx = fixture();
    try {
        // Filtering would shift option_index against the producer's original
        // list and submit a silently WRONG answer once the ingress exists.
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, options: [null, null] }), null);
        assert.equal(fx.decision.buildQuizCard({
            ...WS_MSG, options: [null, 'Plain', { label: 'Real' }, { detail: 'no label' }],
        }), null);
        // String options remain a legal producer shorthand.
        const card = fx.decision.buildQuizCard({ ...WS_MSG, options: ['Plain', { label: 'Real' }] });
        assert.ok(card);
        assert.equal(card.querySelectorAll('.chat-quiz-option').length, 2);
    } finally { fx.restore(); }
});

test('question and stake go through the injected markdown pipeline', () => {
    const fx = fixture({ renderMarkdown: (text) => `<md>${text}</md>` });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        assert.equal(card.querySelector('.chat-quiz-question').innerHTML, '<md>Merge now?</md>');
        assert.equal(card.querySelector('.chat-quiz-stake').innerHTML, '<md>At stake: release timing</md>');
    } finally { fx.restore(); }
});

test('a second click while the first answer is in flight is ignored', async () => {
    let resolveFetch;
    const fx = fixture({ fetchImpl: () => new Promise((resolve) => { resolveFetch = resolve; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const buttons = card.querySelectorAll('.chat-quiz-option');
        buttons[0].click();
        buttons[1].click();
        resolveFetch({ ok: true, status: 200 });
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
        assert.equal(card.dataset.state, 'answered');
    } finally { fx.restore(); }
});

test('a non-409 failure keeps the card open with an honest toast', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: false, status: 404 }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'open');
        assert.match(fx.toasts[0].text, /Could not record the answer \(404\)/);
        // The pending latch is released: a later click retries.
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 2);
    } finally { fx.restore(); }
});

test('a 409 settles the card by the BODY state — a lost race reads answered, not expired', async () => {
    // The refusal body carries the true lifecycle state: the loser of a
    // first-wins race must see the winning answer, never a false expiry.
    const fx = fixture({ fetchImpl: async () => ({
        ok: false, status: 409,
        json: async () => ({ ok: false, error: 'quiz_closed', state: 'answered', answered_index: 1 }),
    }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.ok(card.querySelectorAll('.chat-quiz-option')[1].classList.contains('chosen'));
        assert.match(fx.toasts[0].text, /Already answered/);
    } finally { fx.restore(); }
});

test('a bodyless 409 still settles the card as expired', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: false, status: 409 }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'expired_terminal');
        assert.match(fx.toasts[0].text, /no longer open/);
    } finally { fx.restore(); }
});

test('a retry reuses the SAME request_id (stable idempotency key)', async () => {
    const seen = [];
    let failFirst = true;
    const fx = fixture({ fetchImpl: async (url, init) => {
        seen.push(JSON.parse(init.body).request_id);
        if (failFirst) { failFirst = false; return { ok: false, status: 503, json: async () => ({}) }; }
        return { ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 0 }) };
    } });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const btn = card.querySelectorAll('.chat-quiz-option')[0];
        btn.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'open'); // 503 leaves the card open for retry
        btn.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.equal(seen.length, 2);
        assert.equal(seen[0], seen[1]);
    } finally { fx.restore(); }
});

test('applyQuizStateFrame settles an existing card and ignores unknown ids', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: true, status: 200 }) });
    if (!globalThis.CSS) globalThis.CSS = { escape: (v) => String(v) };
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const inner = card.children[0] || card; // frameNode wraps the card
        const quizCard = inner.matchesClass && inner.matchesClass('chat-quiz-card') ? inner : card;
        const root = {
            querySelector: (sel) => (sel.includes('qz-1') ? quizCard : null),
        };
        // The live lifecycle frame (WS quiz_state) settles the card in place.
        const applied = fx.decision.applyQuizStateFrame(root, {
            quiz_id: 'qz-1', task_id: 't-1', state: 'answered', answered_index: 0,
        });
        assert.equal(applied, true);
        assert.equal(quizCard.dataset.state, 'answered');
        const buttons = quizCard.querySelectorAll('chat-quiz-option');
        assert.ok(buttons.length >= 2);
        assert.ok(buttons.every((btn) => btn.disabled));
        assert.ok(buttons[0].classList.contains('chosen'));
        const stateWrites = countPropertyWrites(quizCard.dataset, 'state');
        const disabledWrites = buttons.map((btn) => countPropertyWrites(btn, 'disabled'));
        let chosenWrites = 0;
        for (const btn of buttons) {
            const toggle = btn.classList.toggle.bind(btn.classList);
            btn.classList.toggle = (...args) => { chosenWrites += 1; return toggle(...args); };
        }
        assert.equal(fx.decision.applyQuizStateFrame(root, {
            quiz_id: 'qz-1', task_id: 't-1', state: 'answered', answered_index: 0,
        }), false, 'an identical acknowledgement performs no DOM mutation');
        assert.equal(stateWrites(), 0);
        assert.ok(disabledWrites.every((writes) => writes() === 0));
        assert.equal(chosenWrites, 0);

        // Unknown id: no card found, nothing thrown, honest false.
        assert.equal(fx.decision.applyQuizStateFrame(root, { quiz_id: 'other', state: 'answered' }), false);
    } finally {
        fx.restore();
    }
});

test('a live quiz_state frame carries the owner comment onto the card like replay does (#471)', () => {
    const fx = fixture();
    if (!globalThis.CSS) globalThis.CSS = { escape: (v) => String(v) };
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const inner = card.children[0] || card;
        const quizCard = inner.matchesClass && inner.matchesClass('chat-quiz-card') ? inner : card;
        const root = { querySelector: (sel) => (sel.includes('qz-1') ? quizCard : null) };
        // The owner rejected every option and answered in their own words: the
        // frame carries no answered_index and the recorded comment.
        assert.equal(fx.decision.applyQuizStateFrame(root, {
            quiz_id: 'qz-1', task_id: 't-1', state: 'answered', comment: 'neither — use duckdb',
        }), true);
        assert.equal(quizCard.dataset.state, 'answered');
        assert.equal(quizCard.dataset.ownerComment, 'neither — use duckdb');
        assert.equal(quizCard.querySelector('.chat-quiz-answer').textContent,
            "Owner's answer: neither — use duckdb");
        assert.ok(quizCard.querySelectorAll('.chat-quiz-option').every((btn) => !btn.classList.contains('chosen')));
        // A later lifecycle frame without a comment never wipes the recorded one.
        fx.decision.applyQuizStateFrame(root, { quiz_id: 'qz-1', task_id: 't-1', state: 'superseded' });
        assert.equal(quizCard.dataset.ownerComment, 'neither — use duckdb');
        assert.equal(quizCard.dataset.state, 'superseded');
    } finally {
        fx.restore();
    }
});

// ---- free answer (the owner's own option) ----

function commentParts(card) {
    return {
        field: card.querySelector('.chat-quiz-comment'),
        send: card.querySelector('.chat-quiz-send'),
    };
}

test('a typed remark rides WITH an option click as one answer', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        assert.ok(field && send);
        field.value = 'yes, but only after CI';
        field.listeners.get('input')();
        assert.equal(send.disabled, false);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal(body.option_index, 0);
        assert.equal(body.comment, 'yes, but only after CI');
        // Settled: the draft field is gone and the words are the record.
        assert.equal(card.querySelector('.chat-quiz-comment-box'), null);
        assert.equal(card.querySelector('.chat-quiz-answer').textContent,
            "Owner's answer: yes, but only after CI");
    } finally { fx.restore(); }
});

test('the send button answers WITHOUT an option_index', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        field.value = '  neither — do C  ';
        field.listeners.get('input')();
        send.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal('option_index' in body, false, 'no option was taken — the key must be absent');
        // VERBATIM: the owner's exact characters, edges included.
        assert.equal(body.comment, '  neither — do C  ');
        assert.equal(card.dataset.state, 'answered');
        // No option is highlighted: the owner chose none of them.
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => !btn.classList.contains('chosen')));
    } finally { fx.restore(); }
});

test('an empty field leaves the send button disabled and sends nothing', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        assert.equal(send.disabled, true);
        field.value = '   ';
        field.listeners.get('input')();
        assert.equal(send.disabled, true);
        send.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 0);
        assert.equal(card.dataset.state, 'open');
    } finally { fx.restore(); }
});

test('an over-long answer is refused client-side, not truncated', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        field.value = 'x'.repeat(2001);
        field.listeners.get('input')();
        assert.equal(send.disabled, true, 'the cap is the ingress limit, mirrored');
        send.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 0);
        assert.match(fx.toasts[0].text, /under 2000 characters/);
        assert.equal(card.dataset.state, 'open');
    } finally { fx.restore(); }
});

test('a settled replayed card shows the owner answer and offers no field', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard({
            msg_type: 'quiz', role: 'assistant', task_id: 't-1', text: 'Merge now?', ts: 'x',
            quiz: {
                quiz_id: 'qz-3', state: 'answered', stake: '', assumption: 'merging meanwhile',
                options: [{ label: 'Yes' }, { label: 'No' }],
                comment: 'neither — revert first',
            },
        });
        assert.equal(card.querySelector('.chat-quiz-comment-box'), null);
        assert.equal(card.querySelector('.chat-quiz-answer').textContent,
            "Owner's answer: neither — revert first");
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => btn.disabled));
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => !btn.classList.contains('chosen')));
    } finally { fx.restore(); }
});

test('a live state frame for an open card never wipes the typed draft', () => {
    const fx = fixture();
    if (!globalThis.CSS) globalThis.CSS = { escape: (v) => String(v) };
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field } = commentParts(card);
        field.value = 'half-written thought';
        field.listeners.get('input')();
        const root = { querySelector: () => card };
        assert.equal(fx.decision.applyQuizStateFrame(root, {
            quiz_id: 'qz-1', task_id: 't-1', state: 'open',
        }), false, 'an open→open frame is a no-op');
        assert.equal(card.querySelector('.chat-quiz-comment').value, 'half-written thought');
        assert.ok(card.querySelector('.chat-quiz-comment-box'));
    } finally { fx.restore(); }
});

// ---- routing picker (#198) ----

function routingBubble(cmid = 'cm-1') {
    const bubble = new NodeStub('div');
    bubble.dataset.clientMessageId = cmid;
    return bubble;
}

const ROUTING_ANNOTATION = {
    status: 'needs_manual_target', routing_token: 'tok-1',
    options: [
        { action: 'steer_task', task_id: 't1', title: 'Fix CI' },
        { action: 'new_task_in_project', project_id: 'p1', project_name: 'Web' },
    ],
};

test('an actionable refusal renders the picker card; other statuses fall back to text', () => {
    const fx = fixture();
    try {
        const bubble = routingBubble();
        assert.equal(fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION), true);
        assert.equal(fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION), false);
        const card = bubble.querySelector('.chat-routing-card');
        assert.ok(card);
        assert.equal(card.dataset.state, 'open');
        const buttons = card.querySelectorAll('.chat-quiz-option');
        assert.equal(buttons.length, 2);
        assert.equal(buttons[0].querySelector('.chat-quiz-option-label').textContent, 'Fix CI');
        assert.equal(buttons[1].querySelector('.chat-quiz-option-label').textContent, 'New task in Web');
        // A later settled annotation (the dispatch ack) REPLACES the card
        // with the plain text line — the card never lingers past its attempt.
        const settled = {
            status: 'delivered', action: 'steer_task', target: 't1', target_label: 'Fix CI',
        };
        assert.equal(fx.decision.renderRoutingDecision(bubble, settled), true);
        const note = bubble.querySelector('.msg-routing-annotation');
        const textWrites = countPropertyWrites(note, 'textContent');
        const noteStatusWrites = countPropertyWrites(note.dataset, 'annotationStatus');
        const bubbleStatusWrites = countPropertyWrites(bubble.dataset, 'chatAnnotationStatus');
        assert.equal(fx.decision.renderRoutingDecision(bubble, settled), false);
        assert.deepEqual(
            [textWrites(), noteStatusWrites(), bubbleStatusWrites()], [0, 0, 0],
        );
        assert.equal(bubble.querySelector('.chat-routing-card'), null);
        assert.match(bubble.querySelector('.msg-routing-annotation').textContent, /Steered task/);
        assert.equal(fx.decision.renderRoutingDecision(bubble, null), true);
        assert.equal(bubble.querySelector('.msg-routing-annotation'), null);
        assert.equal(bubble.dataset.chatAnnotationStatus, undefined);
        assert.equal(fx.decision.renderRoutingDecision(bubble, null), false);
    } finally { fx.restore(); }
});

test('a routing click posts the routing decision id with a STABLE request id', async () => {
    const fx = fixture();
    try {
        const bubble = routingBubble('cm-2');
        fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION);
        const card = bubble.querySelector('.chat-routing-card');
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal(body.decision_id, 'routing:cm-2:tok-1');
        assert.equal(body.option_index, 1);
        assert.ok(body.request_id);
        assert.equal(card.dataset.state, 'answered');
        assert.ok(card.querySelectorAll('.chat-quiz-option')[1].classList.contains('chosen'));
    } finally { fx.restore(); }
});

test('a routing 409 settles the card from the body state, never a false expiry', async () => {
    const fx = fixture({
        fetchImpl: async () => ({
            ok: false, status: 409,
            json: async () => ({ state: 'answered', answered_index: 0 }),
        }),
    });
    try {
        const bubble = routingBubble('cm-3');
        fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION);
        const card = bubble.querySelector('.chat-routing-card');
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.ok(card.querySelectorAll('.chat-quiz-option')[0].classList.contains('chosen'));
        assert.equal(fx.toasts.length, 1);
    } finally { fx.restore(); }
});

test('more than eight options hide behind a show-all control', () => {
    let writes = 0;
    const fx = fixture({ onDomWrite: (mutate) => { writes += 1; return mutate(); } });
    try {
        const bubble = routingBubble('cm-4');
        const wide = {
            ...ROUTING_ANNOTATION,
            options: Array.from({ length: 11 }, (_, i) => (
                { action: 'steer_task', task_id: `t${i}`, title: `Task ${i}` })),
        };
        fx.decision.renderRoutingDecision(bubble, wide);
        const card = bubble.querySelector('.chat-routing-card');
        const buttons = card.querySelectorAll('.chat-quiz-option');
        assert.equal(buttons.length, 11);
        assert.equal(buttons.filter((btn) => btn.hidden).length, 3);
        const more = card.querySelector('.chat-quiz-more');
        assert.match(more.textContent, /11/);
        const beforeClick = writes;
        more.click();
        assert.equal(writes, beforeClick + 1);
        assert.equal(buttons.filter((btn) => btn.hidden).length, 0);
        assert.equal(card.querySelector('.chat-quiz-more'), null);
    } finally { fx.restore(); }
});

test('a duplicate confirmation renders the recorded free answer, not the retried click', async () => {
    const fx = fixture({ fetchImpl: async () => ({
        ok: true, status: 200,
        json: async () => ({ ok: true, state: 'answered', duplicate: true, comment: 'my own plan' }),
    }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const area = card.querySelector('.chat-quiz-comment');
        area.value = 'rewritten retry';
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.dataset.ownerComment, 'my own plan');
        // No option is fabricated for a recorded free answer.
        assert.ok(!card.querySelectorAll('.chat-quiz-option')[1].classList.contains('chosen'));
    } finally { fx.restore(); }
});

test('a 409 loser adopts the winning comment over its local draft', async () => {
    const fx = fixture({ fetchImpl: async () => ({
        ok: false, status: 409,
        json: async () => ({ ok: false, error: 'quiz_closed', state: 'answered', answered_index: 0, comment: 'winning note' }),
    }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const area = card.querySelector('.chat-quiz-comment');
        area.value = 'losing draft';
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.dataset.ownerComment, 'winning note');
    } finally { fx.restore(); }
});
