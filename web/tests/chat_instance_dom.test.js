import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
class ClassList {
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
class ElementStub {
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
    get textContent() { return this._textContent; }
    set innerHTML(value) {
        this._innerHTML = String(value || '');
        if (!this.ownerDocument) return;
        this.children = [];
        for (const match of this._innerHTML.matchAll(/<([a-z0-9-]+)([^>]*)>/gi)) {
            const node = new ElementStub(match[1], this.ownerDocument);
            const attrs = match[2];
            const idMatch = attrs.match(/\sid="([^"]+)"/i);
            if (idMatch) node.id = idMatch[1];
            const classMatch = match[0].match(/\sclass="([^"]*)"/i);
            if (classMatch) node.className = classMatch[1];
            for (const data of attrs.matchAll(/\sdata-([a-z0-9-]+)(?:="([^"]*)")?/gi)) {
                const key = data[1].replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                node.dataset[key] = data[2] ?? '';
            }
            node.parentNode = this;
            node.parentElement = this;
            node.isConnected = this.isConnected;
            this.children.push(node);
            if (node.id) this.ownerDocument.byId.set(node.id, node);
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
    appendChild(node) { return this.insertBefore(node, null); }
    append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
    prepend(node) { return this.insertBefore(node, this.children[0] || null); }
    insertAdjacentElement(_position, node) { const list = this.parentNode?.children || []; return this.parentNode?.insertBefore(node, list[list.indexOf(this) + 1] || null); }
    insertBefore(node, before) {
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
    replaceChildren(...nodes) { this.children = []; nodes.forEach((node) => this.appendChild(node)); }
    contains(node) {
        if (node === this) return true;
        return this.children.some((child) => child.contains(node));
    }
    querySelector(selector) {
        const id = selector.match(/^\[id="([^"]+)"\]$/)?.[1];
        if (id) return this.ownerDocument?.byId.get(id) || null;
        const data = selector.match(/^\[data-([a-z0-9-]+)\]$/i)?.[1];
        if (data) {
            const key = data.replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
            return this.children.find((child) => Object.hasOwn(child.dataset, key)) || null;
        }
        if (selector === '.typing-bubble') return this.children.find((child) => child.classList.contains('typing-bubble')) || null;
        if (selector.startsWith('.')) {
            const className = selector.slice(1).split(/[ :>\[]/)[0];
            return this.children.find((child) => child.classList.contains(className)) || null;
        }
        return null;
    }
    querySelectorAll(selector) {
        if (selector === '[id]') return this.children.filter((child) => child.id);
        const data = selector.match(/^\[data-([a-z0-9-]+)\]$/i)?.[1];
        if (data) {
            const key = data.replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
            return this.children.filter((child) => Object.hasOwn(child.dataset, key));
        }
        if (selector.startsWith('.')) {
            const className = selector.slice(1).split(/[ :>\[]/)[0];
            return this.children.filter((child) => child.classList.contains(className));
        }
        return [];
    }
    closest(selector) {
        if (selector === '.page.active' && this.classList.contains('page') && this.classList.contains('active')) return this;
        return this.parentElement?.closest?.(selector) || null;
    }
    getBoundingClientRect() { return { top: 0, bottom: 20, left: 0, right: 100, width: 100, height: 20 }; }
    getClientRects() { return [this.getBoundingClientRect()]; }
    focus() { if (this.ownerDocument) this.ownerDocument.activeElement = this; } click() {}
}
function installDom(fetchImpl = async () => ({ ok: true, json: async () => ({ active_direct_turns: [] }) })) {
    const prior = {
        document: globalThis.document, window: globalThis.window,
        sessionStorage: globalThis.sessionStorage, fetch: globalThis.fetch,
        ResizeObserver: globalThis.ResizeObserver,
        requestAnimationFrame: globalThis.requestAnimationFrame,
    };
    const document = {
        byId: new Map(), hidden: false, activeElement: null,
        createElement(tag) { return new ElementStub(tag, document); },
        createDocumentFragment() {
            const fragment = new ElementStub('#document-fragment', document);
            fragment.isDocumentFragment = true;
            return fragment;
        },
        getElementById(id) { return document.byId.get(id) || null; },
        addEventListener() {}, removeEventListener() {},
    };
    const mount = new ElementStub('div', document);
    mount.isConnected = true;
    document.byId.set('content', mount);
    const storage = new Map();
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
function restoreDom(prior) {
    Object.assign(globalThis, prior);
}
test('createChatInstance renders a real assistant bubble without senderLabel shadowing', () => {
    const { prior, mount } = installDom();
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true,
        send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true,
        apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws,
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {},
            stateSnapshots,
            chatId: 2,
            idPrefix: 'chat',
            mountEl: mount,
            asPanel: true,
        });
        assert.doesNotThrow(() => handlers.get('chat')({
            chat_id: 2,
            role: 'assistant',
            content: 'hello from Ouroboros',
            ts: '2026-08-24T00:00:00Z',
        }));
        const messages = globalThis.document.byId.get('chat-messages');
        const bubble = messages.children.find((node) => node.classList.contains('chat-bubble')
            && node.classList.contains('assistant') && !node.classList.contains('typing-bubble'));
        assert.ok(bubble, 'the actual createChatInstance message path appended a bubble');
        assert.match(bubble.innerHTML, /<div class="sender">Ouroboros<\/div>/);
        assert.match(bubble.innerHTML, /hello from Ouroboros/);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('first task-bound review hydrates a progress-created owner once and reconciles task truth', async () => {
    const calls = [];
    let resolveDeferredDetail = null, resolveNoopDetail = null;
    let reconnectRows = [];
    const { prior, mount } = installDom(async (url) => {
        calls.push(String(url));
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: reconnectRows, window: {
                complete: false, truncated_by: ['quota'] } }) };
        }
        if (String(url).startsWith('/api/tasks/root-review')) {
            return { ok: true, json: async () => ({
                task_id: 'root-review', status: 'running', cancel_state: 'pending',
                stop_policy: 'finalize_then_cancel',
                plan_review_state: {
                    current_attempt: { fingerprint: 'plan-root-review', status: 'closed' },
                    waves_omitted: 0,
                    waves: [{
                        request_fingerprint: 'plan-root-review',
                        aggregate: 'GREEN',
                        closed: true,
                    }],
                },
            }) };
        }
        if (String(url).startsWith('/api/tasks/root-synthesis')) {
            return { ok: true, json: async () => ({
                task_id: 'root-synthesis', status: 'completed',
                root_phase_checkpoint: { post_task_synthesis: 'running' },
            }) };
        }
        if (String(url).startsWith('/api/tasks/root-deferred')) {
            return new Promise((resolve) => {
                resolveDeferredDetail = () => resolve({ ok: true, json: async () => ({
                    task_id: 'root-deferred', status: 'running', cancel_state: 'pending',
                    stop_policy: 'finalize_then_cancel',
                }) });
            });
        }
        if (String(url).startsWith('/api/tasks/root-noop')) {
            return new Promise((resolve) => {
                resolveNoopDetail = () => resolve({
                    ok: true, json: async () => ({ task_id: 'root-noop' }),
                });
            });
        }
        if (String(url).startsWith('/api/tasks/root-terminal-active')) {
            return { ok: true, json: async () => ({
                task_id: 'root-terminal-active', status: 'completed',
            }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true,
        send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true,
        apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws,
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        handlers.get('typing')({
            chat_id: 2, activity_id: 'root-review', task_id: 'root-review',
            kind: 'managed_task', phase: 'working',
        });
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'root-review', content: 'Owner work is already visible',
            ts: '2026-08-24T00:00:00Z',
        });
        const messages = globalThis.document.byId.get('chat-messages');
        const jump = globalThis.document.byId.get('chat-scroll-bottom');
        const progressCard = messages.children.find((node) => node.dataset.taskId === 'root-review');
        assert.ok(progressCard, 'progress created the owner card before its first review');
        handlers.get('chat')({
            chat_id: 2,
            role: 'system',
            system_type: 'skill_review',
            task_id: 'initiator-child',
            ts: '2026-08-24T00:00:01Z',
            review_group: {
                surface: 'skill', id: 'task:root-review:alpha',
                presentation_owner_task_id: 'root-review', skill: 'alpha', status: 'clean',
                attempts: [{ job_id: 'job-1', skill: 'alpha', status: 'clean' }],
            },
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(calls.filter((url) => url.startsWith('/api/tasks/root-review')).length, 1,
            'the first review on an existing owner card used one bounded detail read');
        const pendingCard = messages.children.find((node) => node.dataset.taskId === 'root-review');
        assert.equal(pendingCard, progressCard, 'the review reused the progress-created card');
        assert.ok(pendingCard, 'the review was attached to its explicit owner card');
        assert.equal(pendingCard.querySelector('[data-live-review-summary]')?.textContent, 'Reviews 2',
            'the same task-detail read hydrated the resident Plan and live Skill groups');
        assert.equal(pendingCard.dataset.finished, '0');
        assert.equal(pendingCard.querySelector('.chat-live-phase')?.textContent, 'Finalizing…');
        assert.equal(messages.children.some((node) => node.classList.contains('system')), false,
            'the task-bound review did not fall through to a standalone system bubble');
        handlers.get('chat')({
            chat_id: 2,
            role: 'system',
            system_type: 'skill_review',
            task_id: 'initiator-child',
            ts: '2026-08-24T00:00:01.500Z',
            review_group: {
                surface: 'skill', id: 'task:root-review:alpha',
                presentation_owner_task_id: 'root-review', skill: 'alpha', status: 'clean',
                attempts: [
                    { job_id: 'job-1', skill: 'alpha', status: 'clean' },
                    { job_id: 'job-1b', skill: 'alpha', status: 'clean' },
                ],
            },
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(calls.filter((url) => url.startsWith('/api/tasks/root-review')).length, 1,
            'later attempts in the same card generation add no task-detail GET');
        messages.scrollTop = 600;
        messages.listeners.get('scroll')[0]();
        handlers.get('chat')({
            chat_id: 2,
            role: 'system',
            system_type: 'skill_review',
            task_id: 'initiator-child',
            ts: '2026-08-24T00:00:02Z',
            review_group: {
                surface: 'skill', id: 'task:root-synthesis:alpha',
                presentation_owner_task_id: 'root-synthesis', skill: 'alpha', status: 'clean',
                attempts: [{ job_id: 'job-2', skill: 'alpha', status: 'clean' }],
            },
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(calls.filter((url) => url.startsWith('/api/tasks/root-synthesis')).length, 1,
            'open post-task synthesis reused the same single detail read');
        const synthesisCard = messages.children.find((node) => node.dataset.taskId === 'root-synthesis');
        assert.ok(synthesisCard);
        assert.equal(synthesisCard.dataset.finished, '0',
            'completed task detail stays live while post-task synthesis is open');
        handlers.get('chat')({
            chat_id: 2,
            role: 'system',
            system_type: 'skill_review',
            task_id: 'initiator-child',
            ts: '2026-08-24T00:00:03Z',
            review_group: {
                surface: 'skill', id: 'task:root-deferred:alpha',
                presentation_owner_task_id: 'root-deferred', skill: 'alpha', status: 'clean',
                attempts: [{ job_id: 'job-3', skill: 'alpha', status: 'clean' }],
            },
        });
        assert.equal(typeof resolveDeferredDetail, 'function', 'the one detail GET is in flight');
        const oldDeferredCard = messages.children.find((node) => node.dataset.taskId === 'root-deferred');
        reconnectRows = [{
            chat_id: 2,
            role: 'system',
            system_type: 'skill_review',
            task_id: 'initiator-child',
            ts: '2026-08-24T00:00:03Z',
            review_group: {
                surface: 'skill', id: 'task:root-deferred:alpha',
                presentation_owner_task_id: 'root-deferred', skill: 'alpha', status: 'clean',
                attempts: [{ job_id: 'job-3', skill: 'alpha', status: 'clean' }],
            },
        }];
        handlers.get('open')({ previouslyConnected: true });
        await new Promise((resolve) => setTimeout(resolve, 25));
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'root-deferred', content: 'Fresh owner activity after reconnect',
            ts: '2026-08-24T00:00:03.500Z',
        });
        const rebuiltDeferredCard = messages.children.find(
            (node) => node.dataset.taskId === 'root-deferred',
        );
        assert.ok(rebuiltDeferredCard, 'reconnect rebuilt the durable review owner');
        assert.notEqual(rebuiltDeferredCard, oldDeferredCard, 'the old card generation was replaced');
        handlers.get('typing')({
            chat_id: 2, activity_id: 'root-deferred', task_id: 'root-deferred',
            kind: 'managed_task', phase: 'working',
        });
        messages.scrollHeight = 1000; messages.clientHeight = 400; messages.scrollTop = 500;
        messages.listeners.get('scroll')[0]();
        resolveDeferredDetail();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(calls.filter((url) => url.startsWith('/api/tasks/root-deferred')).length, 1,
            'typing during the detail read did not trigger a second GET');
        const deferredCard = messages.children.find((node) => node.dataset.taskId === 'root-deferred');
        assert.equal(deferredCard?.dataset.finished, '0');
        assert.equal(deferredCard?.querySelector('.chat-live-phase')?.textContent, 'Finalizing…',
            'the delayed read reconciled the current card and pending cancel outranked fresh activity');
        assert.equal(messages.scrollTop, 500, 'the delayed remote detail preserves the reader');
        assert.equal(jump.getAttribute('aria-label'), 'New activity — scroll to latest message');
        messages.scrollTop = 600;
        messages.listeners.get('scroll')[0]();
        handlers.get('typing')({
            chat_id: 2, activity_id: 'root-terminal-active', task_id: 'root-terminal-active',
            kind: 'managed_task', phase: 'working',
        });
        handlers.get('chat')({
            chat_id: 2,
            role: 'system',
            system_type: 'skill_review',
            task_id: 'initiator-child',
            ts: '2026-08-24T00:00:04Z',
            review_group: {
                surface: 'skill', id: 'task:root-terminal-active:alpha',
                presentation_owner_task_id: 'root-terminal-active', skill: 'alpha', status: 'clean',
                attempts: [{ job_id: 'job-4', skill: 'alpha', status: 'clean' }],
            },
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(calls.filter((url) => url.startsWith('/api/tasks/root-terminal-active')).length, 1);
        const terminalCard = messages.children.find(
            (node) => node.dataset.taskId === 'root-terminal-active',
        );
        assert.equal(terminalCard?.dataset.finished, '0',
            'fresh managed activity prevents stale terminal detail from closing the card');
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'root-noop', content: 'No-op detail target',
        });
        messages.scrollHeight = 1000;
        messages.clientHeight = 400;
        messages.scrollTop = 600;
        messages.listeners.get('scroll')[0]();
        handlers.get('chat')({
            chat_id: 2, role: 'system', system_type: 'skill_review',
            task_id: 'initiator-child',
            review_group: {
                surface: 'skill', id: 'task:root-noop:alpha',
                presentation_owner_task_id: 'root-noop', skill: 'alpha', status: 'clean',
                attempts: [{ skill: 'alpha', status: 'clean' }],
            },
        });
        assert.equal(typeof resolveNoopDetail, 'function');
        messages.scrollTop = 560; messages.listeners.get('scroll')[0]();
        resolveNoopDetail(); await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(messages.scrollTop, 560, 'a late task-detail no-op cannot follow a reader from the 40px zone');
        const routineRows = [
            { chat_id: 2, role: 'system', is_progress: true, task_id: 'routine-root', content: 'First visible progress', ts: '2026-08-24T00:01:00Z' },
            { chat_id: 2, role: 'system', is_progress: true, task_id: 'routine-root', content: 'Second visible progress', ts: '2026-08-24T00:01:01Z' },
            { chat_id: 2, role: 'system', is_progress: true, task_id: 'routine-child', delegation_role: 'subagent', subagent_event: 'running', subagent_task_id: 'routine-child', parent_task_id: 'routine-root', subagent_role: 'reader', content: 'Child progress', ts: '2026-08-24T00:01:02Z' },
        ];
        // #691: an ephemeral decision row renders like any other progress row, so
        // the no-op replay is exactly the rows already on screen.
        for (const row of routineRows) handlers.get('chat')(row); reconnectRows = [...routineRows];
        messages.scrollHeight = 1000; messages.clientHeight = 400; messages.scrollTop = 600; messages.listeners.get('scroll')[0](); messages.scrollTop = 560; messages.listeners.get('scroll')[0]();
        await instance.refreshHistory({ revision: 1 });
        assert.equal(messages.scrollTop, 560, 'ordinary and subagent replay no-ops cannot consume the 40px follow zone');
        assert.equal(jump.getAttribute('aria-label'), 'Scroll to latest message');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('review-only reconnect anchors stay inert until task truth arrives', async () => {
    let historyRows = [];
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: historyRows }) };
        }
        if (String(url).startsWith('/api/tasks/review-404')) {
            return { ok: false, status: 404, text: async () => 'not found' };
        }
        if (String(url).startsWith('/api/tasks/review-error')) throw new Error('offline');
        if (String(url).startsWith('/api/tasks/review-running')) {
            return { ok: true, json: async () => ({ task_id: 'review-running', status: 'running' }) };
        }
        if (String(url) === '/api/tasks/review-terminal') {
            return { ok: true, json: async () => ({ task_id: 'review-terminal', status: 'completed' }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true,
        send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true,
        apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws,
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        historyRows = [
            'review-root', 'review-404', 'review-error', 'review-running', 'review-terminal',
            'review-snapshot', 'review-finalizing-ws', 'review-terminal-ws',
        ]
            .map((owner, index) => ({
                chat_id: 2,
                role: 'system',
                system_type: 'skill_review',
                task_id: `review-child-${index}`,
                ts: `2026-08-24T00:00:0${index + 1}Z`,
                review_group: {
                    surface: 'skill', id: `task:${owner}:alpha`,
                    presentation_owner_task_id: owner, skill: 'alpha', status: 'clean',
                    attempts: [{ job_id: `review-job-${index}`, skill: 'alpha', status: 'clean' }],
                },
            }));
        handlers.get('open')({ previouslyConnected: true });
        await new Promise((resolve) => setTimeout(resolve, 25));
        const messages = globalThis.document.byId.get('chat-messages');
        const card = (owner) => messages.children.find((node) => node.dataset.taskId === owner);
        for (const owner of [
            'review-root', 'review-404', 'review-error', 'review-snapshot',
            'review-finalizing-ws', 'review-terminal-ws',
        ]) {
            const ownerCard = card(owner);
            assert.ok(ownerCard, `${owner} keeps the explicit review owner card`);
            assert.equal(ownerCard.querySelector('[data-live-review-summary]')?.textContent, 'Reviews 1');
            assert.equal(ownerCard.querySelector('[data-live-phase]')?.hidden, true, owner);
            assert.equal(ownerCard.querySelector('[data-live-title]')?.textContent, 'Reviews');
            assert.equal(ownerCard.querySelector('[data-live-typing]')?.style.display, 'none');
        }
        assert.equal(card('review-running')?.querySelector('[data-live-phase]')?.hidden, false);
        assert.equal(card('review-running')?.querySelector('[data-live-phase]')?.textContent, 'Working');
        assert.equal(card('review-terminal')?.dataset.finished, '1');
        const snapshotCard = card('review-snapshot');
        instance.hydrateStateSnapshot({
            active_direct_turns: [{
                activity_id: 'review-snapshot', chat_id: 2,
                kind: 'managed_task', phase: 'working',
            }],
        });
        assert.equal(card('review-snapshot'), snapshotCard, 'late task truth promotes the same card');
        assert.equal(snapshotCard.querySelector('[data-live-phase]')?.hidden, false);
        assert.equal(snapshotCard.querySelector('[data-live-phase]')?.textContent, 'Working');
        const finalizingCard = card('review-finalizing-ws');
        const finalizingFrame = { chat_id: 2, role: 'assistant', task_id: 'review-finalizing-ws',
            content: 'Answer delivered while synthesis runs', task_phase: 'finalizing',
            ts: '2026-08-24T00:01:00Z' };
        handlers.get('chat')(finalizingFrame);
        assert.equal(card('review-finalizing-ws'), finalizingCard);
        assert.equal(finalizingCard.querySelector('[data-live-phase]')?.hidden, false);
        assert.equal(finalizingCard.querySelector('[data-live-phase]')?.textContent, 'Finalizing…');
        messages.scrollHeight = 1000; messages.clientHeight = 400; messages.scrollTop = 600;
        messages.listeners.get('scroll')[0]();
        messages.scrollTop = 500;
        messages.listeners.get('scroll')[0]();
        handlers.get('chat')(finalizingFrame);
        assert.equal(messages.scrollTop, 500, 'an identical finalizing frame is not a scroll author');
        assert.equal(globalThis.document.byId.get('chat-scroll-bottom').getAttribute('aria-label'), 'Scroll to latest message');
        const terminalWsCard = card('review-terminal-ws');
        handlers.get('chat')({
            chat_id: 2, role: 'assistant', task_id: 'review-terminal-ws',
            content: 'Task completed', task_terminal_status: 'completed',
        });
        assert.equal(card('review-terminal-ws'), terminalWsCard);
        assert.equal(terminalWsCard.dataset.finished, '1');
        assert.equal(terminalWsCard.querySelector('[data-live-phase]')?.hidden, false);
        assert.equal(terminalWsCard.querySelector('[data-live-phase]')?.textContent, 'Done');
        handlers.get('log')({
            chat_id: 2,
            data: {
                type: 'task_cost_finalized', task_id: 'review-terminal-ws',
                post_task_status: 'completed', cost_accounting_status: 'available',
                accounted_upper_bound_usd: 0.42, cost_final: true,
            },
        });
        assert.equal(terminalWsCard.dataset.finished, '1');
        assert.equal(terminalWsCard.querySelector('[data-live-phase]')?.textContent, 'Done');
        const anchoredCard = card('review-root');
        handlers.get('typing')({
            chat_id: 2, activity_id: 'review-root', task_id: 'review-root',
            kind: 'managed_task', phase: 'working',
        });
        assert.equal(card('review-root'), anchoredCard, 'task activity promotes the same card');
        assert.equal(anchoredCard.querySelector('[data-live-phase]')?.hidden, false);
        assert.equal(anchoredCard.querySelector('[data-live-phase]')?.textContent, 'Working');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('Plan invalidation applies terminal task detail to its review-created owner card', async () => {
    const revision = 'a'.repeat(64);
    const detailCalls = [];
    let historyRows = [];
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: historyRows }) };
        }
        if (String(url).startsWith('/api/tasks/root-terminal')) {
            detailCalls.push(String(url));
            return { ok: true, json: async () => ({
                task_id: 'root-terminal', status: 'completed',
                plan_review_state: {
                    current_attempt: { fingerprint: revision, status: 'closed' },
                    waves_omitted: 0,
                    waves: [{ request_fingerprint: revision, aggregate: 'GREEN', closed: true }],
                },
            }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        handlers.get('chat')({
            chat_id: 2, role: 'system', system_type: 'review_reference',
            surface: 'plan_review', task_id: 'root-terminal', state_revision: revision,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        const messages = globalThis.document.byId.get('chat-messages');
        const card = messages.children.find((node) => node.dataset.taskId === 'root-terminal');
        assert.ok(card, 'the plan projection attached to its explicit owner card');
        assert.equal(card.dataset.finished, '1', 'canonical terminal detail settled the owner card');
        assert.equal(card.querySelector('.chat-live-phase')?.textContent, 'Done');
        assert.equal(detailCalls.length, 1);
        historyRows = [{
            chat_id: 2, role: 'system', is_progress: true, task_id: 'plan-review-rail',
            progress_meta: { review_reference: {
                surface: 'plan_review', presentation_owner_task_id: 'root-terminal',
                state_revision: revision,
            } },
        }];
        handlers.get('open')({ previouslyConnected: true });
        await new Promise((resolve) => setTimeout(resolve, 0));
        await new Promise((resolve) => setTimeout(resolve, 0));
        const rebuiltCard = messages.children.find((node) => node.dataset.taskId === 'root-terminal');
        assert.ok(rebuiltCard, 'the same durable revision reattached after a full reconnect rebuild');
        assert.notEqual(rebuiltCard, card);
        assert.equal(rebuiltCard.querySelector('[data-live-review-summary]')?.textContent, 'Reviews 1');
        assert.equal(detailCalls.length, 2,
            'the applied-revision receipt reset for the new card generation');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('source-incomplete typed review lifecycle is consumed in history, live chat, and logs', async () => {
    const lifecycle = { kind: 'review', status: 'running', target: 'manual-skill', job_id: 'manual-job' };
    const historyRow = {
        chat_id: 2, role: 'system', is_progress: true, task_id: 'history-review-lifecycle',
        progress_meta: { lifecycle },
    };
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: [historyRow] }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true, task_id: 'live-review-lifecycle', lifecycle,
        });
        handlers.get('log')({
            chat_id: 2,
            data: { chat_id: 2, type: 'task_progress', is_progress: true, task_id: 'log-review-lifecycle', lifecycle },
        });
        const messages = globalThis.document.byId.get('chat-messages');
        for (const taskId of ['history-review-lifecycle', 'live-review-lifecycle', 'log-review-lifecycle']) {
            assert.equal(messages.children.some((node) => node.dataset.taskId === taskId), false, taskId);
        }
        assert.equal(messages.children.some((node) => node.classList.contains('system')), false,
            'manual/legacy review lifecycle did not fall through to a chat bubble');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('terminal task-bound review lifecycle resyncs canonical verdict without reconnect', async () => {
    const calls = [];
    let historyRows = [];
    const { prior, mount } = installDom(async (url) => {
        const value = String(url);
        calls.push(value);
        if (value.startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: historyRows }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true,
        send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true,
        apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws,
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        const groupId = 'task:live-review:alpha';
        const lifecycleBase = {
            kind: 'review', target: 'alpha', job_id: 'job-terminal',
            group_id: groupId, presentation_owner_task_id: 'live-review',
        };
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'skill_lifecycle_review_alpha_job-terminal',
            progress_meta: { lifecycle: { ...lifecycleBase, status: 'running' } },
        });
        const messages = globalThis.document.byId.get('chat-messages');
        const owner = messages.children.find((node) => node.dataset.taskId === 'live-review');
        assert.ok(owner, 'source-complete lifecycle attaches to its owner card');
        const reviewHost = owner.querySelector('[data-live-reviews-host]');
        assert.match(reviewHost?.innerHTML || '', /running/);
        const initialHistoryCalls = calls.filter((url) => url.startsWith('/api/chat/history')).length;
        historyRows = [{
            chat_id: 2, role: 'system', system_type: 'skill_review',
            task_id: 'review-child', ts: '2026-08-28T00:00:02Z',
            review_group: {
                surface: 'skill', id: groupId,
                presentation_owner_task_id: 'live-review', skill: 'alpha', status: 'clean',
                attempts: [{ job_id: 'job-terminal', skill: 'alpha', status: 'clean' }],
            },
        }];
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'skill_lifecycle_review_alpha_job-terminal',
            progress_meta: {
                lifecycle: { ...lifecycleBase, status: 'succeeded', phase: 'completed' },
            },
        });
        await new Promise((resolve) => setTimeout(resolve, 760));
        assert.equal(calls.filter((url) => url.startsWith('/api/chat/history')).length, initialHistoryCalls + 1,
            'terminal lifecycle schedules one debounced canonical history fetch');
        const refreshedOwner = messages.children.find((node) => node.dataset.taskId === 'live-review');
        const refreshedReviewHost = refreshedOwner?.querySelector('[data-live-reviews-host]');
        assert.doesNotMatch(refreshedReviewHost?.innerHTML || '', /Review verdict unavailable/);
        assert.match(refreshedReviewHost?.innerHTML || '', /clean/i);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('duplicate lifecycle pointer never mints a task and enriches only an existing exact owner', async () => {
    const { prior, mount } = installDom();
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    const pointer = {
        kind: 'review', job_id: 'job-pointer', status: 'running', target: 'alpha',
        group_id: 'task:root-pointer:alpha', presentation_owner_task_id: 'root-pointer',
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        const messages = globalThis.document.byId.get('chat-messages');
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'skill_lifecycle_review_alpha_job-pointer',
            content: 'Skill review alpha is already running in its original chat.',
            progress_meta: { lifecycle_pointer: pointer },
        });
        assert.equal(messages.children.some((node) => node.dataset.taskId === 'root-pointer'), false);
        assert.equal(messages.children.some(
            (node) => node.dataset.taskId === 'skill_lifecycle_review_alpha_job-pointer',
        ), false);
        const acknowledgements = messages.children.filter(
            (node) => node.classList.contains('chat-bubble')
                && node.classList.contains('assistant')
                && node.classList.contains('progress'),
        );
        assert.equal(acknowledgements.length, 1);
        assert.equal(acknowledgements[0].classList.contains('system'), false);
        assert.equal(acknowledgements[0].dataset.taskId, undefined);
        assert.match(acknowledgements[0].innerHTML, /already running in its original chat/);
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'root-pointer', content: 'Owner task is already visible',
        });
        const owner = messages.children.find((node) => node.dataset.taskId === 'root-pointer');
        assert.ok(owner);
        const jump = globalThis.document.byId.get('chat-scroll-bottom');
        messages.scrollHeight = 1000;
        messages.clientHeight = 400;
        messages.scrollTop = 600;
        messages.listeners.get('scroll')[0]();
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'skill_lifecycle_review_alpha_job-pointer',
            content: 'Skill review alpha is already running in its original chat.',
            progress_meta: { lifecycle_pointer: pointer },
        });
        assert.equal(messages.children.find((node) => node.dataset.taskId === 'root-pointer'), owner);
        assert.equal(owner.querySelector('[data-live-review-summary]')?.textContent, 'Reviews 1 · 1 active');
        assert.equal(messages.children.some(
            (node) => node.dataset.taskId === 'skill_lifecycle_review_alpha_job-pointer',
        ), false);
        assert.equal(messages.children.filter(
            (node) => node.classList.contains('chat-bubble')
                && node.classList.contains('assistant')
                && node.classList.contains('progress'),
        ).length, 1, 'an existing exact owner consumes the pointer without another acknowledgement');
        messages.scrollTop = 0;
        messages.listeners.get('scroll')[0]();
        handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true,
            task_id: 'skill_lifecycle_review_alpha_job-pointer',
            content: 'Skill review alpha is already running in its original chat.',
            progress_meta: { lifecycle_pointer: pointer },
        });
        assert.equal(jump.getAttribute('aria-label'), 'Scroll to latest message',
            'a consumed duplicate pointer is not visible remote activity');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('history replay keeps one duplicate lifecycle acknowledgement without a task card', async () => {
    const pointerRow = {
        chat_id: 2, role: 'assistant', is_progress: true, task_id: '',
        text: 'Skill review alpha is already running in its original chat.',
        ts: '2026-08-25T00:00:00Z',
        lifecycle_pointer: {
            kind: 'review', job_id: 'job-pointer', status: 'running', target: 'alpha',
            group_id: 'task:root-pointer:alpha', presentation_owner_task_id: 'root-pointer',
        },
    };
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: [pointerRow] }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await instance.refreshHistory({ revision: 1 });
        const messages = globalThis.document.byId.get('chat-messages');
        const acknowledgements = messages.children.filter(
            (node) => node.classList.contains('chat-bubble')
                && node.classList.contains('assistant')
                && node.classList.contains('progress'),
        );
        assert.equal(acknowledgements.length, 1);
        assert.match(acknowledgements[0].innerHTML, /already running in its original chat/);
        assert.equal(messages.children.some((node) => node.dataset.taskId === 'root-pointer'), false);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('history replay keeps an ownerless duplicate acknowledgement without a task card', async () => {
    const pointerRow = {
        chat_id: 2, role: 'assistant', is_progress: true, task_id: '',
        text: 'Skill review alpha is already running in its original chat.',
        ts: '2026-08-25T00:00:00Z',
        lifecycle_pointer: {
            kind: 'review', job_id: 'manual-job', status: 'running', target: 'alpha',
        },
    };
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: [pointerRow] }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await instance.refreshHistory({ revision: 1 });
        const messages = globalThis.document.byId.get('chat-messages');
        const acknowledgements = messages.children.filter(
            (node) => node.classList.contains('chat-bubble')
                && node.classList.contains('assistant')
                && node.classList.contains('progress'),
        );
        assert.equal(acknowledgements.length, 1);
        assert.match(acknowledgements[0].innerHTML, /already running in its original chat/);
        assert.equal(messages.children.some((node) => node.dataset.taskId), false);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('an alias-free subagent terminal keeps the honest amount, live and on reload', async () => {
    // Stage-2 regression: since ABI-3 the write seam strips the retired
    // `cost_usd*` aliases (cost_projection.with_cost_aliases), so a terminal
    // frame carries ONLY the honest names — while the card's cost whitelist
    // materializes the retired names as own properties valued `undefined`.
    // The shared pair resolver read that as "the deprecated name is present"
    // and answered null, so a child card that already showed `up to $0.99`
    // froze on `cost pending` (sticky, and the terminal reading is final).
    let historyRows = [];
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: historyRows, window: { complete: true } }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true,
        send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true,
        apply() {},
    };
    // Subagent cards live inside their parent card, so walk the tree.
    const findCard = (taskId) => {
        const walk = (node) => {
            if (node.dataset?.taskId === taskId) return node;
            for (const child of node.children || []) {
                const hit = walk(child);
                if (hit) return hit;
            }
            return null;
        };
        return walk(globalThis.document.byId.get('chat-messages'));
    };
    const costText = (taskId) => findCard(taskId)?.querySelector('[data-live-meta]')?.innerHTML || '';
    let instance;
    try {
        instance = createChatInstance({
            ws,
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        handlers.get('chat')({
            chat_id: 2, role: 'assistant', is_progress: true, content: 'child working',
            ts: '2026-09-02T00:00:00Z', task_id: 'child-1', parent_task_id: 'root-1',
            subagent_task_id: 'child-1', delegation_role: 'subagent',
            subagent_event: 'running', subagent_role: 'coder', model: 'm',
            accounted_upper_bound_usd_with_children: 0.99,
            cost_accounting_status: 'available', cost_final: false,
        });
        assert.match(costText('child-1'), /up to \$0\.99/, 'honest progress frame shows the ceiling');
        // Same class, other reader: a terminal frame with NO accounting evidence
        // must not touch the card's cost at all. The whitelist materializes all
        // twelve names, so the evidence test has to mean a VALUE too — otherwise
        // a costless task_done outranked the live ceiling on recency alone.
        handlers.get('log')({ chat_id: 2, data: {
            type: 'task_done', task_id: 'child-1', status: 'completed',
            ts: '2026-09-02T00:00:05Z',
        } });
        assert.match(costText('child-1'), /up to \$0\.99/, 'a costless terminal keeps the ceiling');
        handlers.get('log')({ chat_id: 2, data: {
            type: 'task_done', task_id: 'child-1', status: 'completed',
            ts: '2026-09-02T00:00:10Z',
            accounted_upper_bound_usd_with_children: 0.99,
            cost_accounting_status: 'available', cost_final: true,
        } });
        assert.match(costText('child-1'), /\$0\.99/, 'the alias-free terminal settles the amount');
        assert.doesNotMatch(costText('child-1'), /cost pending/);
        // An upstream producer that still MIRRORS the retired alias keeps
        // winning its pair (deprecated-wins is the one precedence rule).
        handlers.get('log')({ chat_id: 2, data: {
            type: 'task_done', task_id: 'child-1', status: 'completed',
            ts: '2026-09-02T00:00:20Z', cost_usd_with_children: 1.25,
            cost_accounting_status: 'available', cost_final: true,
        } });
        assert.match(costText('child-1'), /\$1\.25/, 'a mirrored legacy alias still reads');
        // Reload replays the same terminal routing for a fresh child card.
        historyRows = [
            {
                chat_id: 2, role: 'assistant', is_progress: true, content: 'child working',
                ts: '2026-09-02T01:00:00Z', task_id: 'child-2', parent_task_id: 'root-2',
                subagent_task_id: 'child-2', delegation_role: 'subagent',
                subagent_event: 'running', subagent_role: 'coder', model: 'm',
            },
            {
                chat_id: 2, role: 'assistant', text: 'child done', content: 'child done',
                ts: '2026-09-02T01:00:10Z', task_id: 'child-2',
                task_terminal_status: 'completed',
                accounted_upper_bound_usd_with_children: 0.5,
                cost_accounting_status: 'available', cost_final: true,
            },
        ];
        handlers.get('open')({ previouslyConnected: true });
        await new Promise((resolve) => setTimeout(resolve, 25));
        assert.match(costText('child-2'), /\$0\.50/, 'reload keeps the replayed terminal amount');
        assert.doesNotMatch(costText('child-2'), /cost pending/);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
test('a stopped direct turn replays its persisted terminal word, never a blanket Done', async () => {
    // The pipeline's closed-phase outbox stamp is persisted on the chat row
    // (message_bus copies task_terminal_status) and replay reads it as the
    // card's phase: a "Stop now" turn settles `failed` under
    // owner_requested_finalization, so its final row must not read Done.
    const replay = async (terminalStatus) => {
        const rows = [
            {
                // The latest progress row replays the durable row's truth (gateway/history
                // _annotate_terminal_task_truth): the stopped turn settled `failed`.
                chat_id: 2, role: 'system', is_progress: true, content: 'listing the root',
                ts: '2026-09-05T00:00:00Z', task_id: `stopped-${terminalStatus}`,
                task_terminal_status: 'failed', reason_code: 'owner_requested_finalization',
                outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'failed' } },
                tool_calls: 3, rounds: 4,
            },
            {
                // The final row carries the pipeline's outbox stamp verbatim.
                chat_id: 2, role: 'assistant', text: 'Stopped: listed the root so far.',
                content: 'Stopped: listed the root so far.', ts: '2026-09-05T00:00:10Z',
                task_id: `stopped-${terminalStatus}`, task_terminal_status: terminalStatus,
            },
        ];
        const { prior, mount } = installDom(async (url) => {
            if (String(url).startsWith('/api/chat/history')) {
                return { ok: true, json: async () => ({ messages: rows }) };
            }
            return { ok: true, json: async () => ({ active_direct_turns: [] }) };
        });
        const ws = { on() { return () => {}; }, isConnected: () => true, send() {} };
        let generation = 0;
        const stateSnapshots = {
            begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            isCurrent: () => true, apply() {},
        };
        let instance;
        try {
            instance = createChatInstance({
                ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
                updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
                asPanel: true,
            });
            await instance.refreshHistory({ revision: 1 });
            await new Promise((resolve) => setTimeout(resolve, 25));
            const messages = globalThis.document.byId.get('chat-messages');
            const card = messages.children.find((node) => node.classList.contains('chat-live-card')
                && node.dataset.taskId === `stopped-${terminalStatus}`);
            assert.ok(card, 'the replayed final row minted its task card');
            assert.equal(card.dataset.finished, '1');
            return card.querySelector('[data-live-phase]')?.textContent;
        } finally {
            instance?.destroy();
            restoreDom(prior);
        }
    };
    assert.equal(await replay('failed'), 'Failed', 'the stopped turn keeps its durable terminal word');
    assert.equal(await replay('completed'), 'Done', 'the old blanket stamp is exactly what flipped the card');
});

// ---------------------------------------------------------------------------
// #636: a full history rebuild keeps nested branches inside their parents.
// Production-shaped window: the parent subagent is represented ONLY by its
// final row (no progress/lifecycle rows, no task_terminal_status on subagent
// finals), its grandchildren by task_summary rows.
// ---------------------------------------------------------------------------
function walkCard(node, taskId) {
    if (node?.dataset?.taskId === taskId && node.classList?.contains('chat-live-card')) return node;
    for (const child of node?.children || []) {
        const hit = walkCard(child, taskId);
        if (hit) return hit;
    }
    return null;
}

test('history rebuild keeps a lineage-known branch nested, never appended top-level (#636)', async () => {
    const rows = [
        { chat_id: 2, role: 'user', content: 'run the tree', text: 'run the tree', ts: '2026-09-06T14:00:00Z' },
        // previous root: final answer only (its progress rows fell out of the window)
        { chat_id: 2, role: 'assistant', content: 'root done', text: 'root done',
          ts: '2026-09-06T15:09:43Z', task_id: 'prev-root', task_terminal_status: 'completed' },
        // code-lens: a child known ONLY by its final row + lineage
        { chat_id: 2, role: 'assistant', content: 'code lens findings', text: 'code lens findings',
          ts: '2026-09-06T14:54:00Z', task_id: 'code-lens', delegation_role: 'subagent',
          parent_task_id: 'prev-root', root_task_id: 'prev-root', subagent_task_id: 'code-lens',
          subagent_role: 'code-lens', model: 'm' },
        // grandchild-infra-lens: child of code-lens, final row only
        { chat_id: 2, role: 'assistant', content: 'infra lens findings', text: 'infra lens findings',
          ts: '2026-09-06T14:54:31Z', task_id: 'grandchild-infra-lens', delegation_role: 'subagent',
          parent_task_id: 'code-lens', root_task_id: 'prev-root', subagent_task_id: 'grandchild-infra-lens',
          subagent_role: 'infra-lens', model: 'm' },
        // its own cancelled descendant: task_summary row
        { chat_id: 2, role: 'system', system_type: 'task_summary', content: 'Cancelled before start.',
          text: 'Cancelled before start.', ts: '2026-09-06T14:53:04Z', task_id: 'scout',
          status: 'cancelled', delegation_role: 'subagent', parent_task_id: 'grandchild-infra-lens',
          root_task_id: 'prev-root', subagent_task_id: 'scout', subagent_role: 'scout', model: 'm' },
        // the continuation root, 90 minutes later
        { chat_id: 2, role: 'user', content: 'continue', text: 'continue', ts: '2026-09-06T16:22:27Z' },
        { chat_id: 2, role: 'assistant', content: 'continuation acknowledged', text: 'continuation acknowledged',
          ts: '2026-09-06T16:24:52Z', task_id: 'cont-root', task_terminal_status: 'completed' },
    ];
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: rows, window: { complete: true } }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await instance.refreshHistory({ revision: 1 });
        const messages = globalThis.document.byId.get('chat-messages');
        const topCards = () => messages.children
            .filter((n) => n.classList.contains('chat-live-card')).map((n) => n.dataset.taskId);
        // The continuation root has only its answer row in this window, so it
        // is a plain assistant bubble, not a card; the previous root is a card
        // because its branch is nested under it.
        assert.deepEqual(topCards(), ['prev-root'], 'only roots are top-level cards');
        const continuation = messages.children.find((n) => /continuation acknowledged/.test(n.innerHTML));
        assert.ok(continuation && !continuation.classList.contains('chat-live-card'));
        const root = walkCard(messages, 'prev-root');
        const codeLens = walkCard(messages, 'code-lens');
        const infra = walkCard(messages, 'grandchild-infra-lens');
        const scout = walkCard(messages, 'scout');
        assert.ok(root && codeLens && infra && scout, 'every node of the tree is rendered');
        assert.equal(codeLens.classList.contains('subagent'), true);
        assert.equal(codeLens.parentNode?.dataset?.subagentsFor, 'prev-root', 'code-lens sits in its root');
        assert.equal(infra.parentNode?.dataset?.subagentsFor, 'code-lens', 'grandchild sits in code-lens');
        assert.equal(scout.parentNode?.dataset?.subagentsFor, 'grandchild-infra-lens');
        assert.equal(codeLens.dataset.parentTaskId, 'prev-root');
        assert.equal(infra.dataset.parentTaskId, 'code-lens');
        // A reconnect replays the same window again: still nested, no duplicates.
        handlers.get('open')({ previouslyConnected: true });
        await new Promise((resolve) => setTimeout(resolve, 25));
        assert.deepEqual(topCards(), ['prev-root']);
        assert.equal(walkCard(messages, 'grandchild-infra-lens').parentNode?.dataset?.subagentsFor, 'code-lens');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

// ---------------------------------------------------------------------------
// #691: a decision (ephemeral) turn's tool work shows on the ordinary live card,
// concludes truthfully, merges its accounting, and claims no task authority.
// ---------------------------------------------------------------------------
test('an ephemeral decision turn renders its tool work on a card without task authority (#691)', async () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({ active_direct_turns: [] }) }));
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        // Main chat: the only surface that offers "Turn into project".
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 1, idPrefix: 'chat', mountEl: mount,
        });
        const messages = globalThis.document.byId.get('chat-messages');
        handlers.get('log')({ chat_id: 1, data: {
            type: 'task_started', task_id: 'eph-1', ephemeral_decision: true, ts: '2026-09-05T10:00:00Z',
        } });
        assert.equal(walkCard(messages, 'eph-1'), null, 'a plain start mints no card');
        handlers.get('log')({ chat_id: 1, data: {
            type: 'tool_call_started', task_id: 'eph-1', tool: 'read_file', ts: '2026-09-05T10:00:01Z',
        } });
        const card = walkCard(messages, 'eph-1');
        assert.ok(card, 'real tool work reveals the ordinary live card');
        assert.equal(card.querySelector('[data-turn-into-project]'), null, 'no task claim: no Convert');
        assert.equal(card.querySelector('[data-cancel-run]'), null, 'no host cancelable marker: no Cancel');
        handlers.get('chat')({
            chat_id: 1, role: 'assistant', is_progress: true, ephemeral_decision: true,
            content: 'Comparing the reset windows…', ts: '2026-09-05T10:00:02Z', task_id: 'eph-1',
        });
        assert.equal(card.dataset.finished, '0');
        assert.ok(card.isConnected, 'progress keeps the card, it is not torn down');
        // The typed conclusion: the final answer frame finishes the card and
        // is the ONE inline receipt.
        handlers.get('chat')({
            chat_id: 1, role: 'assistant', content: 'The earliest window resets on Monday.',
            ts: '2026-09-05T10:22:00Z', task_id: 'eph-1', task_terminal_status: 'completed',
        });
        assert.equal(card.dataset.finished, '1');
        const receipts = messages.children.filter((n) => n.classList.contains('chat-bubble')
            && n.classList.contains('assistant') && !n.classList.contains('progress')
            && /resets on Monday/.test(n.innerHTML));
        assert.equal(receipts.length, 1);
        // The blank-status ephemeral task_done carries the accounting facts; it
        // must merge them without reopening the finished card.
        handlers.get('log')({ chat_id: 1, data: {
            type: 'task_done', task_id: 'eph-1', status: '', ephemeral_decision: true,
            ts: '2026-09-05T10:22:01Z', outcome_axes: { execution: { status: 'degraded' } },
            reason_code: 'tool_failure', accounted_upper_bound_usd: 2.700732,
            cost_accounting_status: 'available', cost_final: true,
        } });
        assert.equal(card.dataset.finished, '1');
        assert.match(card.querySelector('[data-live-meta]').innerHTML, /\$2\.70/);
        assert.equal(card.querySelector('[data-turn-into-project]'), null);
        // An ephemeral turn WITHOUT tool work or progress stays a plain answer.
        handlers.get('log')({ chat_id: 1, data: {
            type: 'task_started', task_id: 'eph-2', ephemeral_decision: true, ts: '2026-09-05T11:00:00Z',
        } });
        handlers.get('chat')({
            chat_id: 1, role: 'assistant', content: 'Just a short answer.',
            ts: '2026-09-05T11:00:01Z', task_id: 'eph-2', task_terminal_status: 'completed',
        });
        assert.equal(walkCard(messages, 'eph-2'), null);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

test('history replay of an ephemeral turn shows its progress and its typed conclusion (#691)', async () => {
    const rows = [
        { chat_id: 1, role: 'user', content: 'compare the reset windows', text: 'compare the reset windows',
          ts: '2026-09-05T10:00:00Z' },
        { chat_id: 1, role: 'assistant', is_progress: true, ephemeral_decision: true,
          content: 'Reading the account snapshots…', ts: '2026-09-05T10:00:02Z', task_id: 'eph-h' },
        { chat_id: 1, role: 'assistant', is_progress: true, ephemeral_decision: true,
          content: 'Comparing the reset windows…', ts: '2026-09-05T10:05:00Z', task_id: 'eph-h' },
        { chat_id: 1, role: 'assistant', content: 'The earliest window resets on Monday.',
          text: 'The earliest window resets on Monday.', ts: '2026-09-05T10:22:00Z', task_id: 'eph-h',
          task_terminal_status: 'completed' },
    ];
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: rows, window: { complete: true } }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 1, idPrefix: 'chat', mountEl: mount,
        });
        await instance.refreshHistory({ revision: 1 });
        const messages = globalThis.document.byId.get('chat-messages');
        const card = walkCard(messages, 'eph-h');
        assert.ok(card, 'replayed progress mints the card');
        assert.equal(card.dataset.finished, '1', 'the persisted typed conclusion finishes it');
        assert.equal(card.querySelector('[data-turn-into-project]'), null);
        assert.equal(messages.children.filter((n) => /resets on Monday/.test(n.innerHTML)).length, 1);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});

// ---------------------------------------------------------------------------
// #300: a root proven terminal settles a child card whose terminal frame was
// lost, from the child's own durable result — one read, no cascade.
// ---------------------------------------------------------------------------
test('a terminal root settles its still-open child card from the child result (#300)', async () => {
    const detailReads = [];
    const { prior, mount } = installDom(async (url) => {
        const u = String(url);
        if (u.startsWith('/api/tasks/')) {
            detailReads.push(u);
            if (u.includes('child-lost')) {
                return { ok: true, json: async () => ({
                    task_id: 'child-lost', status: 'completed', result: 'child finished quietly',
                    outcome_axes: { execution: { status: 'ok' } },
                    accounted_upper_bound_usd: 0.3, cost_accounting_status: 'available', cost_final: true,
                }) };
            }
            if (u.includes('child-open')) {
                return { ok: true, json: async () => ({ task_id: 'child-open', status: 'running' }) };
            }
            return { ok: false, status: 404, json: async () => ({}) };
        }
        if (u.startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: [], window: { complete: true } }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        // Let the instance's initial history hydration settle first: the live
        // frames below are the whole story, the durable window is empty.
        await instance.refreshHistory({ revision: 1 });
        const messages = globalThis.document.byId.get('chat-messages');
        const child = (id, role) => ({
            chat_id: 2, role: 'assistant', is_progress: true, content: `${role} working`,
            ts: '2026-09-06T12:00:01Z', task_id: id, parent_task_id: 'root-x', root_task_id: 'root-x',
            subagent_task_id: id, delegation_role: 'subagent', subagent_event: 'running',
            subagent_role: role, model: 'm',
        });
        handlers.get('chat')({ chat_id: 2, role: 'assistant', is_progress: true, content: 'root working',
            ts: '2026-09-06T12:00:00Z', task_id: 'root-x', cancelable: true });
        handlers.get('chat')(child('child-lost', 'scout'));
        handlers.get('chat')(child('child-open', 'writer'));
        const lost = walkCard(messages, 'child-lost');
        const open = walkCard(messages, 'child-open');
        assert.equal(lost.dataset.finished, '0');
        // The root's terminal task_done arrives; the child's own terminal was lost.
        handlers.get('log')({ chat_id: 2, data: {
            type: 'task_done', task_id: 'root-x', status: 'completed', ts: '2026-09-06T12:30:00Z',
        } });
        await new Promise((resolve) => setTimeout(resolve, 25));
        assert.equal(walkCard(messages, 'root-x').dataset.finished, '1');
        assert.equal(lost.dataset.finished, '1', 'the lost child closes from its own durable result');
        assert.match(lost.querySelector('[data-live-meta]').innerHTML, /\$0\.30/);
        assert.equal(open.dataset.finished, '0', 'a child with no proven terminal keeps its state');
        assert.deepEqual(detailReads.map((u) => u.replace('/api/tasks/', '')).sort(), ['child-lost', 'child-open']);
        // A second root terminal (replayed task_done) does not re-read settled children.
        handlers.get('log')({ chat_id: 2, data: {
            type: 'task_done', task_id: 'root-x', status: 'completed', ts: '2026-09-06T12:30:01Z',
        } });
        await new Promise((resolve) => setTimeout(resolve, 25));
        assert.equal(detailReads.filter((u) => u.includes('child-lost')).length, 1);
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
