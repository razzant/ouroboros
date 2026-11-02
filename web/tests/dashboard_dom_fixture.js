import { TextStub } from './chat_dom_fixture.js';

// DOM bookkeeping only: the production initializers, HTTP client, status helper,
// stream subscription and log deduplication all run unchanged. Layout/AT behavior
// remains a browser check; this fixture does not pretend to measure either.
export class NodeStub {
    constructor(tag = 'div') {
        this.tagName = tag;
        this.children = [];
        this.attrs = {};
        this.dataset = {};
        this.events = new Map();
        this.scrollHeight = this.scrollTop = this.clientHeight = 0;
        this.hidden = this.disabled = false;
        this._text = '';
        this.classList = {
            contains: (name) => this.className.split(/\s+/).includes(name),
            toggle: (name, force) => {
                const names = new Set(this.className.split(/\s+/).filter(Boolean));
                const enabled = force ?? !names.has(name);
                if (enabled) names.add(name); else names.delete(name);
                this.className = [...names].join(' ');
                return enabled;
            },
            add: (name) => this.classList.toggle(name, true),
        };
    }
    get id() { return this.attrs.id || ''; }
    set id(value) { this.attrs.id = value; }
    get ownerDocument() { return globalThis.document; }
    get isConnected() { return true; }
    get tabIndex() { return this.attrs.tabindex === undefined ? 0 : Number(this.attrs.tabindex); }
    set tabIndex(value) { this.attrs.tabindex = String(value); }
    getClientRects() { return [{ width: 1, height: 1 }]; }
    get className() { return this.attrs.class || ''; }
    set className(value) { this.attrs.class = value; }
    setAttribute(key, value) {
        this.attrs[key] = String(value);
        if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(value);
        if (key === 'disabled' || key === 'hidden') this[key] = true;
    }
    getAttribute(key) { return this.attrs[key] ?? null; }
    hasAttribute(key) { return key in this.attrs; }
    removeAttribute(key) { delete this.attrs[key]; }
    get firstElementChild() { return this.children.find((child) => child.nodeType !== 3); }
    get firstChild() { return this.children[0] || null; }
    get lastChild() { return this.children.at(-1) || null; }
    get nextElementSibling() { return this.parentElement?.children[this.parentElement.children.indexOf(this) + 1]; }
    get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
    set textContent(text) { this._text = String(text); this.children = []; }
    // escapeHtmlText's round trip (text written as textContent reads back escaped); markup
    // children are not serialized back.
    get innerHTML() { return this.textContent.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
    set innerHTML(html) {
        this.textContent = '';
        const stack = [this];
        for (const token of String(html).match(/<[^>]+>|[^<]+/g) || []) {
            if (token.startsWith('</')) { stack.pop(); continue; }
            if (!token.startsWith('<')) { stack.at(-1)._text += token; continue; }
            const tag = token.match(/^<([\w-]+)/)?.[1];
            if (!tag) continue;
            const node = new NodeStub(tag);
            for (const attr of token.slice(tag.length + 1, -1).matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
                node.setAttribute(attr[1], attr[2] ?? '');
            }
            stack.at(-1).appendChild(node);
            if (!['input', 'br', 'hr', 'img', 'meta', 'link'].includes(tag)) stack.push(node);
        }
    }
    appendChild(node) { node.remove(); node.parentElement = this; this.children.push(node); return node; }
    insertBefore(node, before) {
        node.remove();
        node.parentElement = this;
        this.children.splice(before ? this.children.indexOf(before) : this.children.length, 0, node);
        return node;
    }
    append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
    remove() {
        if (this.parentElement) this.parentElement.children.splice(this.parentElement.children.indexOf(this), 1);
        this.parentElement = null;
    }
    contains(node) { return this === node || this.children.some((child) => child.contains(node)); }
    matches(selector) {
        if (selector.includes(',')) return selector.split(',').some((part) => this.matches(part.trim()));
        if (selector.includes('][')) return (selector.match(/\[[^\]]+\]/g) || []).every((part) => this.matches(part));
        if (selector.startsWith('#')) return this.id === selector.slice(1);
        if (selector.startsWith('.')) return this.classList.contains(selector.slice(1));
        const attr = selector.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);
        return attr ? this.hasAttribute(attr[1]) && (attr[2] === undefined || this.getAttribute(attr[1]) === attr[2]) : this.tagName === selector;
    }
    querySelectorAll(selector) {
        return this.children.flatMap((child) => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
    focus() { globalThis.document.activeElement = this; }
    addEventListener(type, fn) {
        if (!this.events.has(type)) this.events.set(type, []);
        this.events.get(type).push(fn);
    }
    async fire(type, event = {}) { for (const fn of this.events.get(type) || []) await fn({ currentTarget: this, ...event }); }
    dispatchEvent(event) { for (const fn of this.events.get(event.type) || []) fn(event); }
    removeEventListener(type, fn) { this.events.set(type, (this.events.get(type) || []).filter((listener) => listener !== fn)); }
}

export function setup(t) {
    const mount = new NodeStub();
    const body = new NodeStub('body');
    const documentEvents = new NodeStub('#document');
    const document = {
        body,
        hidden: false,
        activeElement: null,
        defaultView: { getComputedStyle: () => ({ visibility: 'visible' }) },
        createElement: (tag) => new NodeStub(tag),
        createTextNode: (data) => new TextStub(data),
        getElementById: (id) => mount.querySelector(`#${id}`) || body.querySelector(`#${id}`),
        addEventListener: (type, fn) => documentEvents.addEventListener(type, fn),
        removeEventListener: (type, fn) => documentEvents.removeEventListener(type, fn),
        dispatchEvent: (event) => documentEvents.dispatchEvent(event),
    };
    const window = new NodeStub();
    const routes = new Map();
    const calls = [];
    const wsHandlers = new Map();
    const ws = {
        on: (type, fn) => { wsHandlers.set(type, fn); },
        send() {},
        emit: (type, data = {}) => wsHandlers.get(type)?.(data),
    };
    const previous = Object.fromEntries(['document', 'window', 'fetch', 'requestAnimationFrame', 'setTimeout'].map((key) => [key, globalThis[key]]));
    // Timers still fire during the test; the handles are tracked so a toast's
    // dismissal timeout cannot keep the runner's event loop alive after it.
    const timers = new Set();
    const realSetTimeout = globalThis.setTimeout;
    Object.assign(globalThis, {
        document, window, requestAnimationFrame: (fn) => fn(),
        setTimeout: (fn, ms, ...rest) => {
            const handle = realSetTimeout(fn, ms, ...rest);
            timers.add(handle);
            return handle;
        },
        fetch: async (url, init) => {
            calls.push(url);
            if (!routes.has(url)) throw new Error(`Unexpected request ${url}`);
            const value = routes.get(url);
            if (typeof value === 'function') return value(url, init);
            if (value instanceof Error) throw value;
            return value;
        },
    });
    t.after(() => {
        timers.forEach((handle) => clearTimeout(handle));
        Object.assign(globalThis, previous);
    });
    const toasts = () => body.querySelectorAll('.toast').map((node) => node.textContent);
    return { mount, window, document, ws, routes, calls, toasts };
}

