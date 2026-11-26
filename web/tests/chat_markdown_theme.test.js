import test from 'node:test';
import assert from 'node:assert/strict';

/* A mounted message is the hard case for appearance: by the time the owner
   switches palettes, the mermaid fence has already been REPLACED by its SVG and
   the chart fence by a canvas, so neither node still carries what it draws. This
   exercises the real consumer against a fake DOM to prove the refresh redraws
   from what was parked at mount and never re-reads the markdown or rebuilds the
   chart — a rebuild would drop live data. */

const TOKENS = {
    dark: { '--diagram-primary': '#25222c', '--chart-text': '#a8b0bd', '--chart-grid': '#222222' },
    light: { '--diagram-primary': '#f3e8ec', '--chart-text': '#4a515d', '--chart-grid': '#dddddd' },
};

const dashed = (key) => `data-${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`;

class El {
    constructor(tag = 'div') {
        this.tagName = tag.toUpperCase();
        this.attrs = {};
        // A real dataset write also creates the attribute, and the refresh path
        // selects on `[data-mermaid-source]` — so the fake must mirror too.
        this.dataset = new Proxy({}, {
            set: (store, key, value) => {
                store[key] = String(value);
                this.attrs[dashed(key)] = String(value);
                return true;
            },
            deleteProperty: (store, key) => {
                delete store[key];
                delete this.attrs[dashed(key)];
                return true;
            },
        });
        this.childNodes = [];
        this.parentNode = null;
        this.isConnected = true;
        this._text = '';
        this._classes = new Set();
        this.style = { setProperty() {} };
        this.listeners = new Map();
    }

    get className() { return [...this._classes].join(' '); }
    set className(value) { this._classes = new Set(String(value).split(/\s+/).filter(Boolean)); }
    get classList() {
        return { add: (...n) => n.forEach((x) => this._classes.add(x)), remove: (...n) => n.forEach((x) => this._classes.delete(x)) };
    }

    get textContent() {
        return this.childNodes.length ? this.childNodes.map((c) => c.textContent).join('') : this._text;
    }
    set textContent(value) { this._text = String(value); this.childNodes.forEach((c) => { c.parentNode = null; c.isConnected = false; }); this.childNodes = []; }

    setAttribute(key, value) { this.attrs[key] = String(value); }
    getAttribute(key) { return this.attrs[key] ?? null; }
    hasAttribute(key) { return key in this.attrs; }
    removeAttribute(key) {
        delete this.attrs[key];
        if (key.startsWith('data-')) delete this.dataset[key.slice(5).replace(/-(.)/g, (_, c) => c.toUpperCase())];
    }
    addEventListener(type, fn) { this.listeners.set(type, fn); }
    removeEventListener(type) { this.listeners.delete(type); }
    getBoundingClientRect() { return { width: 600 }; }
    closest() { return null; }

    append(...nodes) { nodes.forEach((n) => { n.parentNode = this; n.isConnected = this.isConnected; this.childNodes.push(n); }); }
    appendChild(node) { this.append(node); return node; }
    prepend(node) { node.parentNode = this; node.isConnected = this.isConnected; this.childNodes.unshift(node); }
    replaceChildren(...nodes) { this.textContent = ''; this.append(...nodes); }
    remove() {
        if (!this.parentNode) return;
        this.parentNode.childNodes = this.parentNode.childNodes.filter((c) => c !== this);
        this.parentNode = null; this.isConnected = false;
    }
    replaceWith(node) {
        const parent = this.parentNode;
        if (!parent) return;
        parent.childNodes = parent.childNodes.map((c) => (c === this ? node : c));
        node.parentNode = parent; node.isConnected = true;
        this.parentNode = null; this.isConnected = false;
    }
    cloneNode() {
        const copy = new El(this.tagName);
        copy.className = this.className;
        copy.attrs = { ...this.attrs };
        for (const [key, value] of Object.entries(this.dataset)) copy.dataset[key] = value;
        copy._text = this._text;
        return copy;
    }

    /* Only the selector shapes chat_markdown actually uses. */
    matches(selector) {
        const [, cls, attr] = selector.match(/^(?:\.([\w-]+))?(?:\[([\w-]+)\])?$/) || [];
        if (cls && !this._classes.has(cls)) return false;
        if (attr && !(attr in this.attrs)) return false;
        return Boolean(cls || attr);
    }
    querySelectorAll(selector) {
        const out = [];
        const walk = (node) => node.childNodes.forEach((child) => {
            if (selector === 'a' ? child.tagName === 'A' : child.matches(selector)) out.push(child);
            walk(child);
        });
        walk(this);
        out.forEach = Array.prototype.forEach.bind(out);
        return out;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

function installDom({ theme = 'dark' } = {}) {
    const priors = {
        document: globalThis.document, window: globalThis.window,
        getComputedStyle: globalThis.getComputedStyle, Chart: globalThis.Chart, mermaid: globalThis.mermaid,
    };
    const body = new El('body');
    const documentElement = new El('html');
    documentElement.dataset.theme = theme;
    globalThis.document = {
        body, head: new El('head'), documentElement,
        createElement: (tag) => new El(tag),
        getElementById: () => null,
    };
    globalThis.getComputedStyle = () => ({
        getPropertyValue: (key) => TOKENS[globalThis.document.documentElement.dataset.theme][key] ?? '',
    });
    const listeners = new Set();
    globalThis.window = {
        addEventListener: (type, fn) => { if (type === 'ouro:theme-changed') listeners.add(fn); },
        removeEventListener: (type, fn) => listeners.delete(fn),
    };
    return {
        body,
        restore: () => Object.assign(globalThis, priors),
        setTheme: (next) => { globalThis.document.documentElement.dataset.theme = next; },
        announce: () => [...listeners].forEach((fn) => fn()),
        subscribers: () => listeners.size,
    };
}

function installMermaid({ gate = null } = {}) {
    const api = { inits: [], runs: [] };
    api.initialize = (config) => api.inits.push(config.themeVariables.primaryColor);
    api.run = async ({ nodes }) => {
        api.runs.push(nodes[0].textContent);
        // The palette is fixed when the render STARTS (mermaid was initialized
        // with those themeVariables); a switch during the await cannot change
        // the ink of the SVG already being drawn. That is the whole race.
        const painted = globalThis.document.documentElement.dataset.theme;
        if (gate) await gate.promise;
        nodes.forEach((node) => {
            node.dataset.paintedWith = painted;
            node.textContent = '<svg/>';
        });
    };
    globalThis.mermaid = api;
    return api;
}

function fakeChartLib() {
    const built = [];
    class FakeChart {
        constructor(canvas, config) {
            this.canvas = canvas;
            this.data = config.data;
            this.options = config.options || {};
            this.scales = { y: { options: {} } };
            this.updates = [];
            this.destroyed = false;
            built.push(this);
        }
        update(mode) { this.updates.push(mode); }
        destroy() { this.destroyed = true; }
    }
    globalThis.Chart = FakeChart;
    return built;
}

const flush = async (rounds = 12) => {
    for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

function mountRoot(dom, ...children) {
    const root = new El('div');
    children.forEach((child) => root.append(child));
    dom.body.append(root);
    return root;
}

function mermaidFence(source) {
    const node = new El('div');
    node.className = 'md-mermaid';
    node.textContent = source;
    return node;
}

function chartFence(config) {
    const node = new El('div');
    node.className = 'md-chart';
    node.textContent = JSON.stringify(config);
    return node;
}

test('a mounted diagram redraws from its parked source, in the new ink', async () => {
    const dom = installDom();
    const mermaid = installMermaid();
    // Fresh module per test: mermaid's per-theme init is module state.
    const { enhanceChatMarkdown } = await import(`../modules/chat_markdown.js?diagram=${Date.now()}`);
    try {
        const root = mountRoot(dom, mermaidFence('graph TD; A-->B'));
        const dispose = enhanceChatMarkdown(root);
        await flush();
        const mounted = root.childNodes[0];
        assert.equal(mounted.textContent, '<svg/>');
        assert.equal(mounted.dataset.paintedWith, 'dark');
        assert.equal(mounted.dataset.mermaidSource, 'graph TD; A-->B');
        assert.deepEqual(mermaid.inits, ['#25222c']);

        dom.setTheme('light');
        dom.announce();
        await flush();
        const repainted = root.childNodes[0];
        assert.equal(repainted.dataset.paintedWith, 'light');
        // Redrawn from the parked source, not from anything left in the document:
        // the node it replaced held an SVG, not the graph.
        assert.deepEqual(mermaid.runs, ['graph TD; A-->B', 'graph TD; A-->B']);
        assert.equal(repainted.dataset.mermaidSource, 'graph TD; A-->B');
        assert.deepEqual(mermaid.inits, ['#25222c', '#f3e8ec']);
        dispose();
    } finally { dom.restore(); }
});

test('a mounted chart is re-tinted in place and keeps its data', async () => {
    const dom = installDom();
    const built = fakeChartLib();
    const { enhanceChatMarkdown } = await import(`../modules/chat_markdown.js?chart=${Date.now()}`);
    try {
        const root = mountRoot(dom, chartFence({
            type: 'line', data: { labels: ['a', 'b'], datasets: [{ label: 's', data: [1, 2] }] },
        }));
        const dispose = enhanceChatMarkdown(root);
        await flush();
        assert.equal(built.length, 1);
        assert.equal(built[0].options.color, '#a8b0bd');

        dom.setTheme('light');
        dom.announce();
        await flush();
        assert.equal(built.length, 1, 'a repaint must never construct a second chart');
        assert.equal(built[0].options.color, '#4a515d');
        assert.equal(built[0].scales.y.options.grid.color, '#dddddd');
        assert.deepEqual(built[0].data.datasets[0].data, [1, 2], 'the data must survive');
        assert.deepEqual(built[0].updates, ['none', 'none']);
        dispose();
    } finally { dom.restore(); }
});

test('a palette switch mid-render keeps selected Mermaid source until its replacement can mount', async () => {
    const dom = installDom();
    let release;
    const gate = { promise: new Promise((resolve) => { release = resolve; }) };
    const mermaid = installMermaid({ gate });
    const { enhanceChatMarkdown } = await import(`../modules/chat_markdown.js?race=${Date.now()}`);
    try {
        const raw = mermaidFence('');
        const selected = new El(); selected.textContent = 'graph TD; A-->B'; raw.append(selected);
        const doc = raw.ownerDocument = new EventTarget();
        let held = true;
        doc.getSelection = () => ({ isCollapsed: !held, rangeCount: 1,
            getRangeAt: () => ({ intersectsNode: node => node === raw }) });
        const root = mountRoot(dom, raw);
        const dispose = enhanceChatMarkdown(root);
        await flush(2);
        assert.equal(mermaid.runs.length, 1, 'the first render is still in flight');
        dom.setTheme('light');
        dom.announce();
        await flush(2);
        assert.equal(raw.childNodes[0], selected, 'theme must not reset textContent on the mounted source');
        // The switch started a second pass; now let the FIRST,
        // dark-palette render finish. Its result must be thrown away.
        gate.promise = Promise.resolve();
        release();
        await flush();
        assert.equal(root.childNodes[0], raw, 'rendered SVG waits for the selection to finish');
        held = false;
        doc.dispatchEvent(new Event('selectionchange'));
        const mounted = root.childNodes[0];
        assert.equal(mounted.dataset.paintedWith, 'light',
            'a stale render must not install an old-palette diagram');
        dispose();
    } finally { dom.restore(); }
});

test('a disposed message stops following the theme and releases its charts', async () => {
    const dom = installDom();
    const built = fakeChartLib();
    const mermaid = installMermaid();
    const { enhanceChatMarkdown } = await import(`../modules/chat_markdown.js?dispose=${Date.now()}`);
    try {
        const root = mountRoot(dom, mermaidFence('graph TD; A-->B'), chartFence({
            type: 'bar', data: { labels: ['a'], datasets: [{ data: [1] }] },
        }));
        const dispose = enhanceChatMarkdown(root);
        await flush();
        assert.equal(dom.subscribers(), 1);
        dispose();
        assert.equal(dom.subscribers(), 0, 'a removed bubble must not be retained by the theme bus');
        assert.equal(built[0].destroyed, true);
        const renders = mermaid.runs.length;
        const updates = built[0].updates.length;
        dom.setTheme('light');
        dom.announce();
        await flush();
        assert.equal(mermaid.runs.length, renders);
        assert.equal(built[0].updates.length, updates);
    } finally { dom.restore(); }
});

test('a theme switch during the initial library download still mounts the diagram', async () => {
    const dom = installDom();
    delete globalThis.mermaid;
    const { enhanceChatMarkdown } = await import(`../modules/chat_markdown.js?late-load=${Date.now()}`);
    try {
        const root = mountRoot(dom, mermaidFence('graph TD; A-->B'));
        const dispose = enhanceChatMarkdown(root);
        const script = document.head.childNodes[0];
        assert.ok(script, 'the library download has started');
        dom.setTheme('light');
        dom.announce();
        const mermaid = installMermaid();
        script.listeners.get('load')();
        await flush();
        assert.equal(root.childNodes[0].dataset.paintedWith, 'light');
        assert.equal(root.childNodes[0].dataset.mermaidSource, 'graph TD; A-->B');
        assert.deepEqual(mermaid.runs, ['graph TD; A-->B'], 'only the current epoch renders');
        dispose();
    } finally { dom.restore(); }
});

test('local theme redraws do not reuse the incoming-content writer', async () => {
    const dom = installDom();
    installMermaid();
    const { enhanceChatMarkdown } = await import(`../modules/chat_markdown.js?local=${Date.now()}`);
    try {
        let remoteWrites = 0; let localWrites = 0;
        const root = mountRoot(dom, mermaidFence('graph TD; A-->B'));
        const dispose = enhanceChatMarkdown(root, {
            onDomWrite: (write) => { remoteWrites += 1; return write(); },
            onThemeDomWrite: (write) => { localWrites += 1; return write(); },
        });
        await flush();
        const initialWrites = remoteWrites;
        assert.ok(initialWrites > 0);
        dom.setTheme('light'); dom.announce(); await flush();
        assert.equal(remoteWrites, initialWrites);
        assert.ok(localWrites > 0);
        assert.equal(root.childNodes[0].dataset.paintedWith, 'light');
        dispose();
    } finally { dom.restore(); }
});
