// A small DOM double for row-level chat modules (chat_reply.js, chat_row_menu.js): bubbling events, focus,
// geometry, and the selector grammar those modules use — tags, classes, attributes ([a], [a="v"], [a^="v"]),
// :not(…), :disabled, :scope, the descendant and child combinators and selector lists. Real engines, hit
// testing and layout are checked in the browser lane.
// A passive listener cannot cancel its event, as in a browser: its preventDefault() does nothing.
function call(listener, event, passive) {
    if (!passive) return listener(event);
    const prevent = event.preventDefault;
    event.preventDefault = () => {};
    try { return listener(event); } finally { event.preventDefault = prevent; }
}

class Events {
    listeners = new Map();
    passive = new Map();  // type → the listeners added with { passive: true }
    addEventListener(type, listener, options) {
        if (!this.listeners.has(type)) this.listeners.set(type, new Set());
        this.listeners.get(type).add(listener);
        if (!this.passive.has(type)) this.passive.set(type, new Set());
        if (options?.passive) this.passive.get(type).add(listener); else this.passive.get(type).delete(listener);
    }
    removeEventListener(type, listener) { this.listeners.get(type)?.delete(listener); this.passive.get(type)?.delete(listener); }
    listenerCount() { return [...this.listeners.values()].reduce((n, set) => n + set.size, 0); }
    emit(type, event = {}) {
        event.type = type;
        event.target ??= this;
        event.preventDefault ??= () => { event.defaultPrevented = true; };
        event.stopPropagation ??= () => { event.stopped = true; };
        // Capture on the document first (the modules' capture listeners live there), then bubble.
        const doc = this.ownerDocument;
        if (doc && this !== doc) for (const listener of [...(doc.captures.get(type) || [])]) listener(event);
        if (event.stopped) return event;
        for (let node = this; node && !event.stopped; node = node.parent || (node !== doc ? doc : null)) {
            const passive = node.passive.get(type);
            for (const listener of [...(node.listeners.get(type) || [])]) call(listener, event, passive?.has(listener));
            if (node === doc) break;
        }
        return event;
    }
}

function compound(part) {
    const not = [];
    const text = part.replace(/:not\(([^)]*)\)/g, (_all, inner) => { not.push(inner); return ''; });
    return {
        tag: text.match(/^[a-z][a-z0-9]*/i)?.[0]?.toLowerCase() || '',
        classes: [...text.matchAll(/\.([\w-]+)/g)].map((m) => m[1]),
        attrs: [...text.matchAll(/\[([^=\]^]+)(\^?=)?(?:"([^"]*)")?\]/g)].map(([, key, op, value]) => ({ key, op, value })),
        disabled: text.includes(':disabled'),
        not,
    };
}

class Node extends Events {
    constructor(doc, tag = 'div', attrs = {}) {
        super();
        this.nodeType = 1;
        this.ownerDocument = doc;
        this.tagName = tag.toUpperCase();
        this.attrs = new Map(Object.entries(attrs));
        this.children = [];
        this.parent = null;
        this.disabled = false;
        this.hidden = false;
        this._text = '';
        this.dataset = {};
        this.classes = new Set();
        this.classList = {
            add: (...names) => names.forEach((n) => this.classes.add(n)),
            remove: (...names) => names.forEach((n) => this.classes.delete(n)),
            contains: (n) => this.classes.has(n),
            toggle: (n, on = !this.classes.has(n)) => (on ? this.classes.add(n) : this.classes.delete(n), on),
        };
        this.css = new Map();
        this.style = { setProperty: (key, value) => this.css.set(key, value), removeProperty: (key) => this.css.delete(key),
            getPropertyValue: (key) => this.css.get(key) ?? '' };
        this.rect = { left: 0, top: 0, right: 100, bottom: 30, width: 100, height: 30 };
    }
    get parentElement() { return this.parent; }
    get isConnected() { return this.parent ? this.parent.isConnected : this === this.ownerDocument.body; }
    get className() { return [...this.classes].join(' '); }
    set className(value) { this.classes = new Set(String(value || '').split(/\s+/).filter(Boolean)); }
    get id() { return this.attrs.get('id') || ''; }
    set id(value) { this.attrs.set('id', String(value)); }
    get tabIndex() {
        return this.attrs.has('tabindex') ? Number(this.attrs.get('tabindex'))
            : ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A'].includes(this.tagName) ? 0 : -1;
    }
    set tabIndex(value) { this.attrs.set('tabindex', String(value)); }
    get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
    set textContent(value) { this.children = []; this._text = String(value ?? ''); }
    set innerHTML(value) { this.children = []; this._text = ''; this.html = String(value); }
    get innerHTML() { return this.html || ''; }
    setAttribute(key, value) { this.attrs.set(key, String(value)); }
    getAttribute(key) {
        if (key === 'hidden' || key === 'disabled') return this[key] ? '' : null;  // reflected like the real properties
        if (key.startsWith('data-')) {
            const name = key.slice(5).replace(/-([a-z])/g, (_all, c) => c.toUpperCase());
            if (Object.hasOwn(this.dataset, name)) return this.dataset[name];
        }
        return this.attrs.get(key) ?? null;
    }
    hasAttribute(key) { return this.getAttribute(key) !== null; }
    removeAttribute(key) { this.attrs.delete(key); }
    matchesCompound(c) {
        if (c.tag && c.tag !== this.tagName.toLowerCase()) return false;
        if (c.disabled && !this.disabled) return false;
        if (!c.classes.every((n) => this.classes.has(n))) return false;
        if (!c.attrs.every(({ key, op, value }) => {
            const actual = this.getAttribute(key);
            return op === '^=' ? Boolean(actual?.startsWith(value)) : op === '=' ? actual === value : actual !== null;
        })) return false;
        return !c.not.some((inner) => this.matches(inner));
    }
    matchChain(steps, scope) {
        const [last, ...rest] = [...steps].reverse();
        if (last.scope ? this !== scope : !this.matchesCompound(last.compound)) return false;
        let node = this;
        for (let i = 0; i < rest.length; i += 1) {
            const step = rest[i];
            const combinator = (i === 0 ? last : rest[i - 1]).combinator;
            const ok = (candidate) => (step.scope ? candidate === scope : candidate.matchesCompound(step.compound));
            if (combinator === '>') { node = node.parent; if (!node || !ok(node)) return false; }
            else { do node = node.parent; while (node && !ok(node)); if (!node) return false; }
        }
        return true;
    }
    matches(selector, scope = null) {
        return selector.split(',').some((part) => {
            const tokens = part.trim().replace(/\s*>\s*/g, ' > ').split(/\s+/);
            const steps = [];
            let combinator = ' ';
            for (const token of tokens) {
                if (token === '>') { combinator = '>'; continue; }
                steps.push({ scope: token === ':scope', compound: compound(token), combinator });
                combinator = ' ';
            }
            return this.matchChain(steps, scope);
        });
    }
    closest(selector) { return this.matches(selector) ? this : this.parent?.closest?.(selector) || null; }
    contains(node) { return this === node || this.children.some((child) => child.contains(node)); }
    descendants() { return this.children.flatMap((child) => [child, ...child.descendants()]); }
    querySelectorAll(selector) { return this.descendants().filter((node) => node.matches(selector, this)); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    append(...nodes) { for (const node of nodes) this.insertBefore(node, null); }
    appendChild(node) { return this.insertBefore(node, null); }
    prepend(...nodes) { const first = this.children[0] || null; for (const node of nodes) this.insertBefore(node, first); }
    insertBefore(node, before) {
        node.parent?.removeChild(node);
        const index = before ? this.children.indexOf(before) : -1;
        if (index >= 0) this.children.splice(index, 0, node); else this.children.push(node);
        node.parent = this;
        return node;
    }
    removeChild(node) {
        const index = this.children.indexOf(node);
        if (index >= 0) this.children.splice(index, 1);
        node.parent = null;
    }
    remove() { this.parent?.removeChild(this); }
    before(node) { this.parent?.insertBefore(node, this); }
    after(node) { const list = this.parent?.children || []; this.parent?.insertBefore(node, list[list.indexOf(this) + 1] || null); }
    replaceWith(node) { this.parent?.insertBefore(node, this); this.remove(); }
    replaceChildren(...nodes) { for (const child of [...this.children]) this.removeChild(child); this._text = ''; this.append(...nodes); }
    get previousElementSibling() { const list = this.parent?.children || []; return list[list.indexOf(this) - 1] || null; }
    get nextElementSibling() { const list = this.parent?.children || []; return list[list.indexOf(this) + 1] || null; }
    get firstElementChild() { return this.children[0] || null; }
    getClientRects() { return this.isConnected && !this.closest('[hidden]') && !this.hidden ? [this.rect] : []; }
    getBoundingClientRect() { return this.rect; }
    get offsetWidth() { return this.rect?.width ?? 0; }
    get offsetHeight() { return this.rect?.height ?? 0; }
    focus() { if (this.isConnected) this.ownerDocument.activeElement = this; }
    click() { if (!this.disabled) this.emit('click', { target: this }); }
    dispatchEvent(event) { return !this.emit(event.type, { native: event }).defaultPrevented; }
}

/** A document with a body, focus, selection, capture listeners and a window for popup geometry. */
export function domDouble() {
    const doc = new Events();
    doc.captures = new Map();
    const bubble = doc.addEventListener.bind(doc);
    doc.addEventListener = (type, listener, options) => {
        const capture = typeof options === 'object' ? Boolean(options?.capture) : Boolean(options);
        if (!capture) return bubble(type, listener, options);
        if (!doc.captures.has(type)) doc.captures.set(type, new Set());
        doc.captures.get(type).add(listener);
        return undefined;
    };
    const unbubble = doc.removeEventListener.bind(doc);
    doc.removeEventListener = (type, listener, options) => ((typeof options === 'object' ? options?.capture : options)
        ? doc.captures.get(type)?.delete(listener) : unbubble(type, listener));
    doc.ownerDocument = doc;
    doc.defaultView = Object.assign(new Events(), {
        innerWidth: 1024, innerHeight: 768,
        getComputedStyle: (node) => ({ visibility: node.visibility || 'visible', whiteSpace: 'normal', lineHeight: '20px' }),
    });
    doc.selection = null;
    doc.getSelection = () => doc.selection;
    doc.createElement = (tag) => new Node(doc, tag);
    doc.body = new Node(doc, 'body');
    doc.getElementById = (id) => doc.body.descendants().find((node) => node.id === id) || null;
    doc.activeElement = doc.body;
    return { doc, el: (tag = 'div', { className = '', attrs = {}, dataset = {}, text = '' } = {}, ...children) => {
        const node = new Node(doc, tag, attrs);
        node.className = className;
        Object.assign(node.dataset, dataset);
        if (text) node.textContent = text;
        node.append(...children);
        return node;
    } };
}

/** Install the double as the global document/window for modules that read them at call time. */
export function installDomDouble() {
    const prior = { document: globalThis.document, window: globalThis.window, sessionStorage: globalThis.sessionStorage };
    const env = domDouble();
    const storage = new Map();
    globalThis.document = env.doc;
    globalThis.window = Object.assign(env.doc.defaultView, { document: env.doc, dispatchEvent() {} });
    globalThis.sessionStorage = { getItem: (k) => storage.get(k) ?? null, setItem: (k, v) => storage.set(k, String(v)),
        removeItem: (k) => storage.delete(k) };
    return { ...env, storage, restore() { Object.assign(globalThis, prior); } };
}
