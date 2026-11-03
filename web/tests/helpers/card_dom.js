// A small DOM double for the living-card modules (card_components.js, card_frames.js, chat_cards.js): it parses the
// markup they and widget_render.js write (tags, attributes, text, entities), keeps live dataset/classList/hidden/
// disabled views of the attributes, supports the keyed in-place patch (childNodes, cloneNode, insertBefore, a
// template's content), compound selectors (tag, .class, [attr], [attr="v"], lists) and bubbling events with an
// AbortSignal. Real engines, layout and frames are checked in the browser lane.
const VOID = new Set(['input', 'img', 'br', 'meta', 'source', 'hr', 'link']);
const decode = (text) => text.replace(/&(amp|lt|gt|quot|#39);/g, (_all, name) => ({ amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'" })[name]);
const encode = (text) => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const camel = (name) => name.replace(/-([a-z])/g, (_all, c) => c.toUpperCase());
const kebab = (name) => name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);

function parseSelector(selector) {
    return selector.split(',').map((part) => {
        const text = part.trim();
        return {
            tag: text.match(/^[a-z][a-z0-9-]*/i)?.[0]?.toLowerCase() || '',
            classes: [...text.matchAll(/\.([\w-]+)/g)].map((m) => m[1]),
            attrs: [...text.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)].map((m) => ({ name: m[1], value: m[2] })),
        };
    });
}

export class TextNode {
    constructor(text, doc) { this.nodeType = 3; this.nodeName = '#text'; this.nodeValue = text; this.ownerDocument = doc; this.parentNode = null; }
    get textContent() { return this.nodeValue; }
    get nextSibling() { const list = this.parentNode?.childNodes || []; return list[list.indexOf(this) + 1] || null; }
    cloneNode() { return new TextNode(this.nodeValue, this.ownerDocument); }
    remove() { this.parentNode?.removeChild(this); }
    serialize() { return encode(this.nodeValue); }
}

export class Element {
    constructor(tag, doc) {
        this.nodeType = 1;
        this.tagName = tag.toUpperCase();
        this.nodeName = this.tagName;
        this.ownerDocument = doc;
        this.childNodes = [];
        this.parentNode = null;
        this.attrs = new Map();
        this.listeners = new Map();
        this.styles = new Map();
        this.value = '';
        this.checked = false;
        this.style = {
            setProperty: (name, value) => this.styles.set(name, String(value)),
            removeProperty: (name) => this.styles.delete(name),
            getPropertyValue: (name) => this.styles.get(name) || '',
        };
        const self = this;
        this.dataset = new Proxy({}, {
            get: (_t, key) => (typeof key === 'string' && self.attrs.has(`data-${kebab(key)}`) ? self.attrs.get(`data-${kebab(key)}`) : undefined),
            set: (_t, key, value) => { self.attrs.set(`data-${kebab(key)}`, String(value)); return true; },
            deleteProperty: (_t, key) => { self.attrs.delete(`data-${kebab(key)}`); return true; },
            has: (_t, key) => self.attrs.has(`data-${kebab(key)}`),
            ownKeys: () => [...self.attrs.keys()].filter((name) => name.startsWith('data-')).map((name) => camel(name.slice(5))),
            getOwnPropertyDescriptor: (_t, key) => (self.attrs.has(`data-${kebab(key)}`)
                ? { enumerable: true, configurable: true, value: self.attrs.get(`data-${kebab(key)}`) } : undefined),
        });
        this.classList = {
            add: (...names) => this.setClasses([...new Set([...this.classes(), ...names])]),
            remove: (...names) => this.setClasses(this.classes().filter((name) => !names.includes(name))),
            contains: (name) => this.classes().includes(name),
            toggle: (name, force = !this.classes().includes(name)) => { (force ? this.classList.add : this.classList.remove)(name); return force; },
        };
        if (this.tagName === 'TEMPLATE') this.content = new Element('#fragment', doc);
    }
    classes() { return String(this.attrs.get('class') || '').split(/\s+/).filter(Boolean); }
    setClasses(names) { this.attrs.set('class', names.join(' ')); }
    get className() { return this.attrs.get('class') || ''; }
    set className(value) { this.attrs.set('class', String(value)); }
    get id() { return this.attrs.get('id') || ''; }
    get hidden() { return this.attrs.has('hidden'); }
    set hidden(value) { if (value) this.attrs.set('hidden', ''); else this.attrs.delete('hidden'); }
    get disabled() { return this.attrs.has('disabled'); }
    set disabled(value) { if (value) this.attrs.set('disabled', ''); else this.attrs.delete('disabled'); }
    get attributes() { return [...this.attrs].map(([name, value]) => ({ name, value })); }
    getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
    setAttribute(name, value) { this.attrs.set(name, String(value)); }
    hasAttribute(name) { return this.attrs.has(name); }
    removeAttribute(name) { this.attrs.delete(name); }
    toggleAttribute(name, force = !this.attrs.has(name)) { if (force) this.attrs.set(name, ''); else this.attrs.delete(name); return force; }
    get children() { return this.childNodes.filter((node) => node.nodeType === 1); }
    get firstChild() { return this.childNodes[0] || null; }
    get firstElementChild() { return this.children[0] || null; }
    get nextSibling() { const list = this.parentNode?.childNodes || []; return list[list.indexOf(this) + 1] || null; }
    get parentElement() { return this.parentNode?.nodeType === 1 ? this.parentNode : null; }
    get textContent() { return this.childNodes.map((node) => node.textContent).join(''); }
    set textContent(value) { this.replaceChildren(); if (String(value ?? '')) this.append(new TextNode(String(value), this.ownerDocument)); }
    get innerHTML() { return this.childNodes.map((node) => node.serialize()).join(''); }
    set innerHTML(html) { (this.tagName === 'TEMPLATE' ? this.content : this).replaceChildren(...parse(String(html ?? ''), this.ownerDocument)); }
    serialize() {
        const attrs = [...this.attrs].map(([name, value]) => (value === '' ? ` ${name}` : ` ${name}="${encode(value).replace(/"/g, '&quot;')}"`)).join('');
        const tag = this.tagName.toLowerCase();
        return VOID.has(tag) ? `<${tag}${attrs}>` : `<${tag}${attrs}>${this.innerHTML}</${tag}>`;
    }
    insertAdjacentHTML(where, html) {
        const nodes = parse(html, this.ownerDocument);
        if (where === 'afterbegin') nodes.reverse().forEach((node) => this.insertBefore(node, this.firstChild));
        else this.append(...nodes);
    }
    appendChild(node) { return this.insertBefore(node, null); }
    append(...nodes) { nodes.forEach((node) => this.insertBefore(typeof node === 'string' ? new TextNode(node, this.ownerDocument) : node, null)); }
    insertBefore(node, ref) {
        if (node.tagName === '#FRAGMENT') { [...node.childNodes].forEach((child) => this.insertBefore(child, ref)); return node; }
        node.parentNode?.removeChild(node);
        const index = ref ? this.childNodes.indexOf(ref) : -1;
        if (index < 0) this.childNodes.push(node); else this.childNodes.splice(index, 0, node);
        node.parentNode = this;
        return node;
    }
    removeChild(node) {
        const index = this.childNodes.indexOf(node);
        if (index >= 0) this.childNodes.splice(index, 1);
        node.parentNode = null;
        return node;
    }
    remove() { this.parentNode?.removeChild(this); }
    replaceChildren(...nodes) { [...this.childNodes].forEach((node) => this.removeChild(node)); this.append(...nodes); }
    replaceWith(node) { const parent = this.parentNode; if (parent) { parent.insertBefore(node, this); parent.removeChild(this); } }
    contains(node) { for (let at = node; at; at = at.parentNode) if (at === this) return true; return false; }
    cloneNode(deep = false) {
        const copy = this.ownerDocument.createElement(this.tagName.toLowerCase());
        this.attrs.forEach((value, name) => copy.attrs.set(name, value));
        if (deep) this.childNodes.forEach((node) => copy.append(node.cloneNode(true)));
        return copy;
    }
    matches(selector) {
        return parseSelector(selector).some(({ tag, classes, attrs }) => (!tag || this.tagName.toLowerCase() === tag)
            && classes.every((name) => this.classList.contains(name))
            && attrs.every(({ name, value }) => this.attrs.has(name) && (value === undefined || this.attrs.get(name) === value)));
    }
    closest(selector) { for (let at = this; at?.nodeType === 1; at = at.parentNode) if (at.matches(selector)) return at; return null; }
    querySelectorAll(selector) {
        const found = [];
        const visit = (node) => node.children.forEach((child) => { if (child.matches(selector)) found.push(child); visit(child); });
        visit(this);
        return found;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    addEventListener(type, handler, options = {}) {
        if (!this.listeners.has(type)) this.listeners.set(type, new Set());
        this.listeners.get(type).add(handler);
        options?.signal?.addEventListener?.('abort', () => this.removeEventListener(type, handler));
    }
    removeEventListener(type, handler) { this.listeners.get(type)?.delete(handler); }
    listenerCount() { return [...this.listeners.values()].reduce((sum, set) => sum + set.size, 0); }
    dispatch(type, extra = {}) {
        const event = { type, target: this, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...extra };
        for (let at = this; at; at = at.parentNode) for (const handler of [...(at.listeners?.get(type) || [])]) handler(event);
        return event;
    }
    click() { return this.dispatch('click'); }
    focus() { this.ownerDocument.activeElement = this; }
}

function parse(html, doc) {
    const root = new Element('#fragment', doc);
    const stack = [root];
    for (const token of html.matchAll(/<\/([a-z0-9-]+)\s*>|<([a-z][a-z0-9-]*)((?:\s+[\w-]+(?:="[^"]*")?)*)\s*\/?>|([^<]+)/gi)) {
        if (token[1]) { while (stack.length > 1 && stack.pop().tagName !== token[1].toUpperCase()); continue; }
        if (token[4] !== undefined) { if (token[4]) stack.at(-1).append(new TextNode(decode(token[4]), doc)); continue; }
        const node = doc.createElement(token[2]);
        for (const attr of (token[3] || '').matchAll(/([\w-]+)(?:="([^"]*)")?/g)) node.attrs.set(attr[1], decode(attr[2] ?? ''));
        stack.at(-1).append(node);
        if (!VOID.has(node.tagName.toLowerCase())) stack.push(node);
    }
    return [...root.childNodes];
}

/** Install the double as `document` / `window` (+ sessionStorage); `restore()` puts the old globals back. */
export function installCardDom({ innerHeight = 800 } = {}) {
    const prior = Object.fromEntries(['document', 'window', 'sessionStorage', 'innerHeight', 'getComputedStyle', 'addEventListener', 'removeEventListener']
        .map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    const doc = { activeElement: null };
    doc.createElement = (tag) => new Element(tag, doc);
    doc.body = doc.createElement('body');
    doc.documentElement = doc.createElement('html');
    const windowListeners = new Map();
    const win = {
        location: { origin: 'http://local' },
        innerHeight,
        addEventListener(type, handler) { if (!windowListeners.has(type)) windowListeners.set(type, new Set()); windowListeners.get(type).add(handler); },
        removeEventListener(type, handler) { windowListeners.get(type)?.delete(handler); },
        emit(type, event = {}) { for (const handler of [...(windowListeners.get(type) || [])]) handler({ type, ...event }); },
        listeners: (type) => windowListeners.get(type)?.size || 0,
    };
    const store = new Map();
    const storage = { getItem: (key) => (store.has(key) ? store.get(key) : null), setItem: (key, value) => store.set(key, String(value)),
        removeItem: (key) => store.delete(key), store };
    const define = (key, value) => Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
    define('document', doc);
    define('window', win);
    define('sessionStorage', storage);
    define('innerHeight', innerHeight);
    define('getComputedStyle', () => ({ getPropertyValue: () => '' }));
    define('addEventListener', win.addEventListener);
    define('removeEventListener', win.removeEventListener);
    return {
        doc, win, storage,
        restore() {
            for (const [key, descriptor] of Object.entries(prior)) {
                if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
            }
        },
    };
}
