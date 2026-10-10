// The DOM stub and media-controller fixture the chat media tests share (moved from
// chat_media.test.js as is): nested innerHTML, class/data/tag selectors, counted listeners.
import { createChatMedia } from '../../modules/chat_media.js';
import { stampNodeTimestamp } from '../../modules/chat_activity.js';

class Classes {
    constructor(node) { this.node = node; this.values = new Set(); }
    set(value) { this.values = new Set(String(value || '').split(/\s+/).filter(Boolean)); }
    add(...values) { values.forEach((value) => this.values.add(value)); }
    contains(value) { return this.values.has(value); }
    toggle(value, force) {
        const enabled = force === undefined ? !this.contains(value) : Boolean(force);
        if (enabled) this.add(value); else this.values.delete(value);
        return enabled;
    }
}

export class NodeStub {
    constructor(tag = 'div', tracker = null) {
        this.tagName = tag.toUpperCase();
        this.tracker = tracker;
        this.children = [];
        this.parentNode = null;
        this.dataset = {};
        this.attributes = new Map();
        this.classList = new Classes(this);
        this.style = { setProperty() {} };
        this.value = '';
        this.disabled = false;
        this.paused = true;
        this.currentTime = 0;
        this.duration = 0;
        this.playbackRate = 1;
        this.loop = false;
        this.muted = false;
        this.pauseCalls = 0;
        this.selectCalls = 0;
        this.listeners = new Map();
    }
    set className(value) { this.classList.set(value); }
    get className() { return [...this.classList.values].join(' '); }
    set textContent(value) {
        this._text = String(value ?? '');
        this._html = this._text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    get textContent() { return this._text || ''; }
    set innerHTML(html) {
        this._html = String(html || '');
        this.children = [];
        const stack = [this];
        for (const token of String(html || '').matchAll(/<\/?[a-z0-9-]+(?:\s[^>]*)?>/gi)) {
            const source = token[0];
            if (source.startsWith('</')) {
                if (stack.length > 1) stack.pop();
                continue;
            }
            const tag = source.match(/^<([a-z0-9-]+)/i)?.[1] || 'div';
            const node = new NodeStub(tag, this.tracker);
            const classes = source.match(/\sclass="([^"]*)"/i)?.[1];
            if (classes) node.className = classes;
            for (const data of source.matchAll(/\sdata-([a-z0-9-]+)(?:="([^"]*)")?/gi)) {
                const key = data[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                node.dataset[key] = data[2] ?? '';
            }
            stack.at(-1).appendChild(node);
            if (!/\/$/.test(source) && !['IMG', 'INPUT', 'SOURCE'].includes(node.tagName)) stack.push(node);
        }
    }
    get innerHTML() { return this._html || ''; }
    appendChild(node) {
        node.parentNode?.removeChild(node);
        this.children.push(node);
        node.parentNode = this;
        return node;
    }
    removeChild(node) {
        const index = this.children.indexOf(node);
        if (index >= 0) this.children.splice(index, 1);
        node.parentNode = null;
    }
    remove() { this.parentNode?.removeChild(this); }
    contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
    matches(selector) { return selector.split(',').some((part) => part.trim().startsWith('.')
        && this.classList.contains(part.trim().slice(1))); }
    before(node) {
        if (!this.parentNode) return;
        const index = this.parentNode.children.indexOf(this);
        this.parentNode.children.splice(index, 0, node);
        node.parentNode = this.parentNode;
    }
    append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) || ''; }
    removeAttribute(name) { this.attributes.delete(name); }
    addEventListener(type, fn) {
        if (!this.listeners.has(type)) this.listeners.set(type, new Set());
        this.listeners.get(type).add(fn);
        this.tracker.adds += 1;
    }
    removeEventListener(type, fn) {
        this.listeners.get(type)?.delete(fn);
        this.tracker.removes += 1;
    }
    async click() {
        for (const listener of this.listeners.get('click') || []) await listener({ currentTarget: this });
    }
    querySelector(selector) {
        return this.querySelectorAll(selector)[0] || null;
    }
    querySelectorAll(selector) {
        const matches = (node) => {
            if (selector.startsWith('.')) return node.classList.contains(selector.slice(1));
            const data = selector.match(/^\[data-([a-z0-9-]+)(?:="([^"]*)")?\]$/i);
            if (data) {
                const key = data[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                return Object.hasOwn(node.dataset, key) && (data[2] === undefined || node.dataset[key] === data[2]);
            }
            return node.tagName === selector.toUpperCase();
        };
        const found = [];
        const visit = (node) => node.children.forEach((child) => {
            if (matches(child)) found.push(child);
            visit(child);
        });
        visit(this);
        return found;
    }
    async play() { this.paused = false; }
    pause() { this.paused = true; this.pauseCalls += 1; }
    select() { this.selectCalls += 1; }
    load() {}
}

export function fixture({ insertNode = null } = {}) {
    const tracker = { adds: 0, removes: 0, created: [] };
    const body = new NodeStub('body', tracker);
    const inserted = [];
    const prior = { document: globalThis.document, window: globalThis.window, navigator: globalThis.navigator };
    globalThis.document = {
        body,
        createElement: (tag) => {
            const node = new NodeStub(tag, tracker);
            tracker.created.push(node);
            return node;
        },
        createDocumentFragment: () => new NodeStub('#fragment', tracker),
        execCommand: () => true,
    };
    globalThis.window = { open() {} };
    Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: { clipboard: { writeText: async () => {} } },
    });
    const controller = createChatMedia({
        chatSessionId: 'session',
        durableChatMediaUrl: (value) => String(value || ''),
        formatMsgTime: () => null,
        insertMessageNode(node) {
            if (insertNode) insertNode(node);
            else body.appendChild(node);
            inserted.push(node);
        },
        senderLabel: () => 'Owner',
        stampNodeTimestamp,
    });
    return { controller, inserted, tracker, restore: () => {
        globalThis.document = prior.document;
        globalThis.window = prior.window;
        Object.defineProperty(globalThis, 'navigator', { configurable: true, value: prior.navigator });
    } };
}
