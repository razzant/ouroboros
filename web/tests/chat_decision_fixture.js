// Shared by chat_decision.test.js and question_mirrors.test.js: a small DOM stub (descendant lookup
// by class), the chat-decision fixture and one quiz frame.
import { createChatDecision } from '../modules/chat_decision.js';

class Classes {
    constructor() { this.values = new Set(); }
    set(value) { this.values = new Set(String(value || '').split(/\s+/).filter(Boolean)); }
    add(...values) { values.forEach((value) => this.values.add(value)); }
    remove(...values) { values.forEach((value) => this.values.delete(value)); }
    contains(value) { return this.values.has(value); }
    toggle(value, force = !this.contains(value)) {
        if (force) this.add(value); else this.remove(value);
        return Boolean(force);
    }
}

export class NodeStub {
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
    replaceWith(node) {
        const parent = this.parentNode;
        if (!parent) return;
        node.parentNode = parent;
        parent.children.splice(parent.children.indexOf(this), 1, node);
        this.parentNode = null;
    }
    get nextElementSibling() {
        const siblings = this.parentNode?.children || [];
        return siblings[siblings.indexOf(this) + 1] || null;
    }
    // Like a browser, :focus-visible follows the last input modality, not the element.
    matches(selector) {
        return selector === ':focus-visible' && globalThis.document.activeElement === this && !globalThis.document.pointerModality;
    }
    addEventListener(type, handler) { this.listeners.set(type, handler); }
    click(event = {}) {
        const handler = this.listeners.get('click');
        if (handler) handler({ target: this, detail: 0, stopPropagation() {}, preventDefault() {}, ...event });
    }
    setAttribute(name, value) { (this.attributes ||= {})[name] = String(value); }
    getAttribute(name) { return this.attributes?.[name] ?? null; }
    removeAttribute(name) { delete this.attributes?.[name]; }
    contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
    closest() {
        for (let node = this; node; node = node.parentNode)
            if (node.tagName === 'BUTTON' || node.attributes?.role === 'button') return node;
        return null;
    }
    focus() { globalThis.document.activeElement = this; }
    matchesClass(name) { return this.classList.contains(name); }
    collect(name, out = []) {
        if (this.matchesClass(name)) out.push(this);
        this.children.forEach((child) => child.collect(name, out));
        return out;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    querySelectorAll(selector) { return this.collect(selector.replace(/^\./, '')); }
}

export function countPropertyWrites(target, key) {
    let value = target[key];
    let writes = 0;
    Object.defineProperty(target, key, {
        configurable: true,
        get: () => value,
        set: (next) => { writes += 1; value = next; },
    });
    return () => writes;
}

export function fixture({ fetchImpl, renderMarkdown, enhanceMarkdown, onDomWrite, fetchDetail, isMain = false,
    frameNode = (_msg, node) => node, insertMessageNode = null, removeMessageNode = null, focusAfterRemoval = null } = {}) {
    const prior = { document: globalThis.document, crypto: globalThis.crypto, window: globalThis.window };
    globalThis.document = { createElement: (tag) => new NodeStub(tag) };
    const opened = [];
    globalThis.window = { dispatchEvent: (event) => opened.push(event.detail) };
    if (!globalThis.crypto?.randomUUID) Object.defineProperty(globalThis, 'crypto', {
        configurable: true, value: { randomUUID: () => 'fixed-request-id' },
    });
    const toasts = [];
    const calls = [];
    const decision = createChatDecision({
        apiFetch: async (url, init) => {
            calls.push({ url, init });
            if (fetchImpl) return fetchImpl(url, init);
            const sent = JSON.parse(init.body);
            return { ok: true, status: 200, json: async () => ({ ok: true, state: 'answered',
                ...(Number.isInteger(sent.option_index) ? { answered_index: sent.option_index } : {}),
                ...(sent.comment ? { comment: sent.comment } : {}) }) };
        },
        frameNode,
        renderMarkdown,
        enhanceMarkdown: enhanceMarkdown || (renderMarkdown ? () => {} : null),
        showToast: (text, tone) => toasts.push({ text, tone }),
        onDomWrite,
        fetchDetail,
        isMain,
        insertMessageNode,
        removeMessageNode,
        focusAfterRemoval,
    });
    return { decision, toasts, calls, opened, restore: () => {
        globalThis.document = prior.document;
        globalThis.window = prior.window;
        Object.defineProperty(globalThis, 'crypto', { configurable: true, value: prior.crypto });
    } };
}

export const WS_MSG = {
    type: 'quiz', role: 'assistant', quiz_id: 'qz-1', task_id: 't-1',
    question: 'Merge now?', stake: 'release timing',
    assumption: 'continuing with the merge', state: 'open',
    options: [{ label: 'Yes' }, { label: 'No', detail: 'wait for CI' }],
    ts: '2026-08-31T10:00:00Z',
};
export const turn = () => new Promise((resolve) => setImmediate(resolve));
