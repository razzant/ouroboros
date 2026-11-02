export class ClassList {
    constructor(node) { this.node = node; this.names = new Set(); }
    add(...names) { names.forEach((name) => this.names.add(name)); this.sync(); }
    remove(...names) { names.forEach((name) => this.names.delete(name)); this.sync(); }
    contains(name) { return this.names.has(name); }
    toggle(name, force) {
        const enabled = force === undefined ? !this.names.has(name) : Boolean(force);
        if (enabled) this.names.add(name); else this.names.delete(name);
        this.sync();
        return enabled;
    }
    sync() { this.node._className = [...this.names].join(' '); }
    from(value) { this.names = new Set(String(value || '').split(/\s+/).filter(Boolean)); this.sync(); }
}
// A text node: a task chip keeps its words in one beside its status mark (task_phase_chip.paintTaskPhase).
export class TextStub {
    constructor(data = '') { this.nodeType = 3; this.data = String(data); this.children = []; this.dataset = {}; }
    get textContent() { return this.data; }
    set textContent(value) { this.data = String(value ?? ''); }
    get classList() { return { contains: () => false }; }
    contains(node) { return node === this; }
    querySelector() { return null; }
    querySelectorAll() { return []; }
    matches() { return false; }
    getAttribute() { return null; }
    hasAttribute() { return false; }
    remove() { this.parentNode?.removeChild?.(this); }
}
export class ElementStub {
    constructor(tag = 'div', doc = null) {
        this.tagName = tag.toUpperCase();
        this.ownerDocument = doc;
        this.dataset = {};
        const styleValues = new Map();
        this.style = { setProperty: (name, value) => styleValues.set(name, String(value)),
            getPropertyValue: (name) => styleValues.get(name) || '' };
        this.attributes = new Map();
        this.children = [];
        this.listeners = new Map();
        this.classList = new ClassList(this);
        this._className = '';
        this._innerHTML = '';
        this._textContent = '';
        this.value = '';
        this.hidden = false;
        this.disabled = false;
        // Detached until mounted under the connected mount (insertBefore /
        // innerHTML propagate): a stub that reports every fresh node as
        // connected hides the history-rebuild card path, whose pass 2 mounts a
        // replayed root card only when `!rec.root.isConnected`.
        this.isConnected = false;
        this.offsetParent = {};
        this.offsetHeight = 0;
        this.scrollTop = 0;
        this.scrollHeight = 0;
        this.clientHeight = 400;
    }
    set className(value) { this.classList.from(value); }
    get className() { return this._className; }
    set textContent(value) {
        this._textContent = String(value ?? '');
        this._innerHTML = this._textContent
            .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }
    get textContent() { return this._textContent + this.children.filter((child) => child.nodeType === 3).map((child) => child.data).join(''); }
    set innerHTML(value) {
        this._innerHTML = String(value || '');
        if (!this.ownerDocument) return;
        this.children = [];
        const stack = [this];
        for (const match of this._innerHTML.matchAll(/<(\/?)([a-z0-9-]+)([^>]*)>/gi)) {
            if (match[1]) {
                const index = stack.findLastIndex((node) => node.tagName === match[2].toUpperCase());
                if (index > 0) stack.length = index;
                continue;
            }
            const node = new ElementStub(match[2], this.ownerDocument);
            const attrs = match[3];
            const idMatch = attrs.match(/\sid="([^"]+)"/i);
            if (idMatch) node.id = idMatch[1];
            const classMatch = match[0].match(/\sclass="([^"]*)"/i);
            if (classMatch) node.className = classMatch[1];
            for (const data of attrs.matchAll(/\sdata-([a-z0-9-]+)(?:="([^"]*)")?/gi)) {
                const key = data[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                node.dataset[key] = data[2] ?? '';
            }
            stack.at(-1).appendChild(node);
            if (node.id) this.ownerDocument.byId.set(node.id, node);
            // Card templates stay flat in this stub. A media figure needs its
            // real descendants: its actions are bound to that item, not the bubble.
            if ((stack.length > 1 || node.tagName === 'FIGURE') && !/\/\s*$/.test(attrs)
                && !['AREA', 'BASE', 'BR', 'COL', 'EMBED', 'HR', 'IMG', 'INPUT', 'LINK', 'META',
                    'PARAM', 'SOURCE', 'TRACK', 'WBR'].includes(node.tagName)) stack.push(node);
        }
    }
    get innerHTML() { return this._innerHTML; }
    addEventListener(type, fn) {
        if (!this.listeners.has(type)) this.listeners.set(type, []);
        this.listeners.get(type).push(fn);
    }
    removeEventListener() {}
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) || ''; }
    removeAttribute(name) { this.attributes.delete(name); }
    hasAttribute(name) { return this.attributes.has(name); }
    appendChild(node) { return this.insertBefore(node, null); }
    append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
    prepend(...nodes) { const before = this.children[0] || null; nodes.forEach((node) => this.insertBefore(node, before)); }
    before(node) { return this.parentNode?.insertBefore(node, this); }
    after(node) { const list = this.parentNode?.children || []; return this.parentNode?.insertBefore(node, list[list.indexOf(this) + 1] || null); }
    insertAdjacentElement(_position, node) { const list = this.parentNode?.children || []; return this.parentNode?.insertBefore(node, list[list.indexOf(this) + 1] || null); }
    insertBefore(node, before) {
        if (node === before) return node;  // as in a DOM: a node put before itself stays where it is
        if (node?.isDocumentFragment) {
            for (const child of [...node.children]) this.insertBefore(child, before);
            return node;
        }
        node.parentNode?.removeChild?.(node);
        const index = before ? this.children.indexOf(before) : -1;
        if (index >= 0) this.children.splice(index, 0, node); else this.children.push(node);
        node.parentNode = this;
        node.parentElement = this;
        const connect = (el, value) => { el.isConnected = value; for (const child of el.children || []) connect(child, value); };
        connect(node, this.isConnected);
        this.scrollHeight = this.children.length * 20;
        return node;
    }
    removeChild(node) {
        const index = this.children.indexOf(node);
        if (index >= 0) this.children.splice(index, 1);
        node.parentNode = null;
        node.parentElement = null;
    }
    remove() { this.parentNode?.removeChild?.(this); this.isConnected = false; }
    get previousElementSibling() { return this.#sibling(-1); }
    get nextElementSibling() { return this.#sibling(1); }
    #sibling(step) {
        const list = (this.parentNode?.children || []).filter((child) => child.nodeType !== 3);
        const index = list.indexOf(this);
        return index < 0 ? null : list[index + step] || null;
    }
    get firstElementChild() { return this.children.find((child) => child.nodeType !== 3) || null; }
    get lastElementChild() { return this.children.findLast((child) => child.nodeType !== 3) || null; }
    get firstChild() { return this.children[0] || null; }
    get lastChild() { return this.children.at(-1) || null; }
    replaceChild(next, current) { this.insertBefore(next, current); this.removeChild(current); return current; }
    replaceChildren(...nodes) { this.children = []; nodes.forEach((node) => this.appendChild(node)); }
    contains(node) {
        if (node === this) return true;
        return this.children.some((child) => child.contains(node));
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    querySelectorAll(selector) {
        // Document order, including a figure's descendants. Attribute filters
        // matter too: clearing pending receipts must not clear settled ones.
        const descendants = this.children.filter((child) => child.nodeType !== 3)
            .flatMap((child) => [child, ...child.querySelectorAll('*')]);
        if (selector === '*') return descendants;
        if (selector === '[id]') return descendants.filter((child) => child.id);
        const id = selector.match(/^\[id="([^"]+)"\]$/)?.[1] || selector.match(/^#([\w-]+)$/)?.[1];
        if (id) return descendants.filter((child) => child.id === id);
        const data = selector.match(/\[data-([a-z0-9-]+)(?:="([^"]*)")?\]/i);
        const classes = selector.startsWith('.') ? selector.slice(1).split(/[ :>\[]/)[0].split('.') : [];
        const tag = /^[a-z][a-z0-9-]*$/i.test(selector) ? selector.toUpperCase() : '';
        return descendants.filter((child) => {
            if (classes.some((name) => !child.classList.contains(name))) return false;
            if (data) {
                const key = data[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                if (!Object.hasOwn(child.dataset, key) || (data[2] !== undefined && child.dataset[key] !== data[2])) return false;
            }
            return classes.length > 0 || Boolean(data) || (tag && child.tagName === tag);
        });
    }
    matches(selector) { return selector.split(',').some((part) => part.trim().startsWith('.')
        && this.classList.contains(part.trim().slice(1))); }
    closest(selector) {
        if (selector === '.page.active' && this.classList.contains('page') && this.classList.contains('active')) return this;
        return this.parentElement?.closest?.(selector) || null;
    }
    getBoundingClientRect() { return { top: 0, bottom: 20, left: 0, right: 100, width: 100, height: 20 }; }
    getClientRects() { return [this.getBoundingClientRect()]; }
    focus() { if (this.ownerDocument) this.ownerDocument.activeElement = this; }
    // Like a browser, a disabled button dispatches no click.
    click() { if (!this.disabled) for (const fn of this.listeners.get('click') || []) fn({ target: this }); }
}
export function installDom(fetchImpl = async () => ({ ok: true, json: async () => ({ active_direct_turns: [] }) })) {
    const prior = {
        document: globalThis.document, window: globalThis.window,
        sessionStorage: globalThis.sessionStorage, fetch: globalThis.fetch,
        ResizeObserver: globalThis.ResizeObserver,
        requestAnimationFrame: globalThis.requestAnimationFrame,
        getComputedStyle: globalThis.getComputedStyle,
    };
    const document = {
        byId: new Map(), hidden: false, activeElement: null,
        createElement(tag) { return new ElementStub(tag, document); },
        createTextNode(data) { return new TextStub(data); },
        createTreeWalker(root) {
            const texts = [];
            const collect = node => { for (const child of node.children || []) {
                if (child.nodeType === 3) texts.push(child); else collect(child);
            } };
            collect(root);
            let index = 0;
            return { nextNode: () => texts[index++] || null };
        },
        createDocumentFragment() {
            const fragment = new ElementStub('#document-fragment', document);
            fragment.isDocumentFragment = true;
            return fragment;
        },
        getElementById(id) { return document.byId.get(id) || null; },
        addEventListener() {}, removeEventListener() {},
    };
    // A mounted chat has a body (room visibility/toasts) and a document root
    // (language changes). The overlay walks text nodes, not the HTML strings.
    document.documentElement = new ElementStub('html', document);
    document.documentElement.isConnected = true;
    document.body = new ElementStub('body', document);
    document.documentElement.appendChild(document.body);
    const mount = new ElementStub('div', document);
    document.body.appendChild(mount);
    document.byId.set('content', mount);
    const storage = new Map();
    globalThis.getComputedStyle = () => ({ lineHeight: '24px', height: '168px', getPropertyValue: () => '7' });
    globalThis.document = document;
    globalThis.window = {
        document, location: { href: 'http://local/' }, history: { replaceState() {} },
        addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
        getSelection: () => null, innerHeight: 800, CSS: { escape: (value) => value },
    };
    globalThis.sessionStorage = {
        getItem: (key) => storage.get(key) || null,
        setItem: (key, value) => storage.set(key, String(value)),
        removeItem: (key) => storage.delete(key),
    };
    globalThis.fetch = fetchImpl;
    globalThis.ResizeObserver = class { observe() {} disconnect() {} };
    globalThis.requestAnimationFrame = (fn) => { fn(); return 1; };
    return { prior, mount };
}
export function restoreDom(prior) {
    Object.assign(globalThis, prior);
}

export function walkCard(node, taskId) {
    if (node?.dataset?.taskId === taskId && node.classList?.contains('chat-live-card')) return node;
    for (const child of node?.children || []) {
        const hit = walkCard(child, taskId);
        if (hit) return hit;
    }
    return null;
}
