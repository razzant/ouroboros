import assert from 'node:assert/strict';
import test from 'node:test';

import { escapeHtmlAttr } from '../modules/utils.js';
import { FRAME_SANDBOX, OVERLAY_ID, READINESS_RETRY_MS, initOnboardingOverlay } from '../modules/onboarding_overlay.js';

// --- Minimal DOM ----------------------------------------------------------
// Enough of the shape the overlay actually uses: element creation, attributes,
// child lists, removal, and attribute/class selectors. Nothing is parsed from
// innerHTML, because the paths under test build their nodes with createElement.

function makeElement(tag) {
    const el = {
        tagName: String(tag).toUpperCase(),
        children: [],
        attributes: {},
        parentElement: null,
        listeners: {},
        className: '',
        id: '',
        innerHTML: '',
        textContent: '',
        setAttribute(name, value) { this.attributes[name] = String(value); },
        getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; },
        appendChild(child) {
            child.parentElement = this;
            this.children.push(child);
            return child;
        },
        remove() {
            const parent = this.parentElement;
            if (!parent) return;
            parent.children = parent.children.filter((node) => node !== this);
            this.parentElement = null;
        },
        addEventListener(name, handler) {
            (this.listeners[name] = this.listeners[name] || []).push(handler);
        },
        matches(selector) {
            if (selector.startsWith('[') && selector.endsWith(']')) {
                return selector.slice(1, -1) in this.attributes;
            }
            if (selector.startsWith('.')) return this.className.split(/\s+/).includes(selector.slice(1));
            return this.tagName === selector.toUpperCase();
        },
        querySelector(selector) {
            for (const child of this.children) {
                if (child.matches(selector)) return child;
                const found = child.querySelector(selector);
                if (found) return found;
            }
            return null;
        },
        click() { (this.listeners.click || []).forEach((handler) => handler()); },
    };
    return el;
}

function installDom({ coreOwned = false } = {}) {
    const body = makeElement('body');
    const previous = { document: globalThis.document, window: globalThis.window, fetch: globalThis.fetch };
    globalThis.document = {
        body,
        createElement: makeElement,
        querySelector(selector) {
            if (selector === 'meta[name="ouroboros-ui-owner"]') {
                return coreOwned ? { getAttribute: (name) => name === 'content' ? 'core' : null } : null;
            }
            return body.querySelector(selector);
        },
        getElementById(id) {
            const walk = (node) => {
                for (const child of node.children) {
                    if (child.id === id) return child;
                    const found = walk(child);
                    if (found) return found;
                }
                return null;
            };
            return walk(body);
        },
    };
    const replaced = [];
    const listeners = new Set();
    globalThis.window = {
        location: {
            origin: 'http://127.0.0.1:8765',
            href: 'http://127.0.0.1:8765/?_ouro_refresh=1&_ouro_reason=sha-change#settings',
            reload() {},
            replace(url) { replaced.push(String(url)); },
        },
        addEventListener(name, handler) { listeners.add(handler); },
        removeEventListener(name, handler) { listeners.delete(handler); },
    };
    // Every overlay this DOM mounts is disposed with it, so a failed assertion
    // never leaves a readiness retry running past its test.
    const disposers = [];
    return {
        body,
        replaced,
        listeners,
        async init() {
            const dispose = await initOnboardingOverlay();
            disposers.push(dispose);
            return dispose;
        },
        restore() {
            disposers.splice(0).forEach((dispose) => dispose());
            globalThis.document = previous.document;
            globalThis.window = previous.window;
            globalThis.fetch = previous.fetch;
        },
    };
}

function overlayIn(body) {
    return body.children.find((node) => node.id === OVERLAY_ID) || null;
}

// --- The blocking surface -------------------------------------------------

test('a readiness transport failure leaves a blocking overlay with a retry, not a bare console error', async () => {
    const dom = installDom();
    const errors = [];
    const previousError = console.error;
    console.error = (...args) => errors.push(args);
    try {
        globalThis.fetch = async () => { throw new Error('Failed to fetch'); };

        const dispose = await dom.init();
        dispose();

        const overlay = overlayIn(dom.body);
        // The reported defect: overlay_elements_created === 0, so after a
        // cancelled desktop wizard (or a headless first run) the ordinary UI was
        // exposed with no onboarding and no way back to it.
        assert.ok(overlay, 'the blocking overlay must still be mounted');
        assert.equal(overlay.getAttribute('aria-modal'), 'true');
        const retry = overlay.querySelector('[data-onboarding-retry]');
        assert.ok(retry, 'a failed readiness check must offer a retry');
        assert.ok(overlay.querySelector('.onboarding-overlay-backdrop'), 'the scrim comes with the answer');
        assert.equal(overlay.querySelector('.onboarding-frame'), null);
        assert.equal(errors.length, 1, 'the failure is still logged');
    } finally {
        console.error = previousError;
        dom.restore();
    }
});

test('retrying after a failure reaches the wizard once the probe answers', async () => {
    const dom = installDom();
    const previousError = console.error;
    console.error = () => {};
    try {
        let attempt = 0;
        globalThis.fetch = async () => {
            attempt += 1;
            if (attempt === 1) throw new Error('Failed to fetch');
            return { status: 200, ok: true };
        };

        const dispose = await dom.init();
        const overlay = overlayIn(dom.body);
        overlay.querySelector('[data-onboarding-retry]').click();
        await new Promise((resolve) => setImmediate(resolve));
        dispose();

        const frame = overlay.querySelector('.onboarding-frame');
        assert.ok(frame, 'a successful retry mounts the wizard frame');
        assert.equal(frame.src, '/onboarding');
        assert.equal(overlay.querySelector('[data-onboarding-retry]'), null);
    } finally {
        console.error = previousError;
        dom.restore();
    }
});

test('only a verified 204 takes the blocking overlay down, and it never dims a configured app', async () => {
    const dom = installDom();
    let scrimSeen = false;
    try {
        globalThis.fetch = async () => {
            // Observed mid-probe: the shell is already up (nothing can slip past
            // it) but the dark scrim has not been painted, so a configured
            // install does not flash a dimmed screen on every load.
            const overlay = overlayIn(dom.body);
            assert.ok(overlay, 'the shell exists before the answer');
            scrimSeen = Boolean(overlay.querySelector('.onboarding-overlay-backdrop'));
            return { status: 204, ok: true };
        };

        await dom.init();

        assert.equal(scrimSeen, false);
        assert.equal(overlayIn(dom.body), null, '204 means configured: no overlay is due');
    } finally {
        dom.restore();
    }
});

test('a non-2xx readiness answer blocks too, instead of falling through to the app', async () => {
    const dom = installDom();
    const previousError = console.error;
    console.error = () => {};
    try {
        globalThis.fetch = async () => ({ status: 503, ok: false });

        (await dom.init())();

        const overlay = overlayIn(dom.body);
        assert.ok(overlay);
        assert.ok(overlay.querySelector('[data-onboarding-retry]'));
    } finally {
        console.error = previousError;
        dom.restore();
    }
});

test('the core-owned client frames builtin setup without a redundant recovery footer', async () => {
    const dom = installDom({ coreOwned: true });
    try {
        globalThis.fetch = async () => ({ status: 200, ok: true });

        await dom.init();

        const overlay = overlayIn(dom.body);
        const frame = overlay.querySelector('.onboarding-frame');
        assert.equal(frame.src, '/onboarding');
        assert.equal(overlay.querySelector('.onboarding-recovery'), null);
        assert.equal(overlay.querySelector('[data-onboarding-recovery]'), null);
        assert.deepEqual(frame.parentElement.children, [frame], 'no footer takes space from the builtin wizard');
        // The policy itself is owned by FRAME_SANDBOX (the login step's
        // popup capability is decided there, on the completion branch); this
        // pins only that the frame is actually sandboxed by it.
        assert.equal(frame.getAttribute('sandbox'), FRAME_SANDBOX);
        assert.ok(FRAME_SANDBOX.includes('allow-scripts'));
    } finally {
        dom.restore();
    }
});

test('the copied client can recover a broken selected setup without its iframe JavaScript', async () => {
    const dom = installDom({ coreOwned: false });
    let requests = 0;
    let reloads = 0;
    try {
        globalThis.fetch = async () => { requests += 1; return { status: 200, ok: true }; };
        globalThis.window.location.reload = () => { reloads += 1; };
        await dom.init();
        const overlay = overlayIn(dom.body);
        const frame = overlay.querySelector('.onboarding-frame');
        const recovery = overlay.querySelector('[data-onboarding-recovery]');
        assert.equal(frame.src, '/onboarding', 'the selected view is the normal first choice');
        assert.equal(recovery.textContent, 'Open built-in setup');
        assert.equal(recovery.type, 'button');
        assert.equal(frame.querySelector('[data-onboarding-recovery]'), null, 'recovery belongs to the parent');
        assert.equal(recovery.parentElement.parentElement, frame.parentElement);
        recovery.click();
        assert.equal(frame.src, '/onboarding?builtin=1');
        assert.equal(overlay.querySelector('.onboarding-frame'), frame, 'navigate the same frame');
        assert.equal(requests, 1, 'recovery saves no settings and starts no second readiness loop');
        assert.equal(overlayIn(dom.body), overlay, 'opening builtin setup is not completion');

        for (const handler of dom.listeners) handler({
            origin: window.location.origin, data: { type: 'ouroboros:onboarding-complete', restart_required: false },
        });
        assert.equal(overlayIn(dom.body), null, 'the existing completion path still works after recovery');
        assert.equal(reloads, 1);
    } finally {
        dom.restore();
    }
});

// --- Signed out, and a core that comes back -------------------------------
// A separately hosted page reaches the app without the gate's login page, and an
// open page outlives its session: the gate's 401 hands over to its own sign-in.

const tick = () => new Promise((resolve) => setImmediate(resolve));

test('a 401 readiness answer opens the gate sign-in returning here, never the dead-end card', async () => {
    const dom = installDom();
    try {
        let calls = 0;
        globalThis.fetch = async () => { calls += 1; return { status: 401, ok: false }; };

        const dispose = await dom.init();

        assert.deepEqual(dom.replaced, ['/auth/login?next=%2F%23settings'],
            'the return keeps path, query and hash and drops only the reload bookkeeping');
        const overlay = overlayIn(dom.body);
        assert.ok(overlay, 'the app stays blocked while the browser leaves');
        assert.equal(overlay.querySelector('[data-onboarding-retry]'), null);
        assert.equal(overlay.querySelector('.onboarding-frame'), null);
        await tick();
        assert.equal(calls, 1, 'a refusal is not retried: signing in is the answer');
        dispose();
    } finally {
        dom.restore();
    }
});

test('only 401 means signed out: a 403 or 404 keeps the failure card and its retry', async () => {
    for (const status of [403, 404, 502]) {
        const dom = installDom();
        const previousError = console.error;
        console.error = () => {};
        try {
            globalThis.fetch = async () => ({ status, ok: false });
            const dispose = await dom.init();
            dispose();
            assert.deepEqual(dom.replaced, [], `HTTP ${status} is not a sign-in`);
            assert.ok(overlayIn(dom.body).querySelector('[data-onboarding-retry]'));
        } finally {
            console.error = previousError;
            dom.restore();
        }
    }
});

test('a core down at load is checked again on a growing pause and unblocks when it answers', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const dom = installDom();
    const previousError = console.error;
    console.error = () => {};
    try {
        const answers = [() => { throw new Error('Failed to fetch'); }, () => ({ status: 502, ok: false }),
            () => ({ status: 204, ok: true })];
        let calls = 0;
        globalThis.fetch = async () => answers[Math.min(calls++, answers.length - 1)]();

        const dispose = await dom.init();
        const card = overlayIn(dom.body).querySelector('[data-onboarding-failure]');
        assert.ok(card, 'the outage is named while it lasts');
        assert.equal(calls, 1);

        t.mock.timers.tick(READINESS_RETRY_MS - 1);
        await tick();
        assert.equal(calls, 1, 'nothing before the pause');
        t.mock.timers.tick(1);
        await tick();
        assert.equal(calls, 2, 'the first automatic check');
        assert.equal(overlayIn(dom.body).querySelector('[data-onboarding-failure]'), card,
            'a repeated failure updates the same card, so a focused Try again keeps focus');
        assert.match(card.querySelector('p').textContent, /HTTP 502/);

        t.mock.timers.tick(READINESS_RETRY_MS);
        await tick();
        assert.equal(calls, 2, 'the second pause is longer than the first');
        t.mock.timers.tick(READINESS_RETRY_MS);
        await tick();
        assert.equal(calls, 3);
        assert.equal(overlayIn(dom.body), null, 'the answer unblocks the app without a click');

        t.mock.timers.tick(60000);
        await tick();
        assert.equal(calls, 3, 'an answered check arms nothing further');
        dispose();
    } finally {
        console.error = previousError;
        dom.restore();
    }
});

test('Try again during a pending check joins it rather than starting a second chain', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const dom = installDom();
    const previousError = console.error;
    console.error = () => {};
    try {
        let calls = 0;
        let release;
        globalThis.fetch = async () => {
            calls += 1;
            if (calls === 1) throw new Error('Failed to fetch');
            return new Promise((resolve) => { release = () => resolve({ status: 503, ok: false }); });
        };
        const dispose = await dom.init();
        const retry = overlayIn(dom.body).querySelector('[data-onboarding-retry]');
        retry.click();
        await tick();
        retry.click();
        await tick();
        assert.equal(calls, 2, 'the second press joined the pending check');
        release();
        await tick();
        t.mock.timers.tick(READINESS_RETRY_MS * 3);
        await tick();
        assert.equal(calls, 3, 'one chain: one automatic check per pause');
        dispose();
    } finally {
        console.error = previousError;
        dom.restore();
    }
});

test('disposal stops the automatic checks and releases the completion listener', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const dom = installDom();
    const previousError = console.error;
    console.error = () => {};
    try {
        let calls = 0;
        globalThis.fetch = async () => { calls += 1; throw new Error('Failed to fetch'); };
        const dispose = await dom.init();
        assert.equal(dom.listeners.size, 1);
        dispose();
        assert.equal(dom.listeners.size, 0);
        t.mock.timers.tick(120000);
        await tick();
        assert.equal(calls, 1, 'no check outlives its owner');
    } finally {
        console.error = previousError;
        dom.restore();
    }
});

test('a check still pending at disposal arms nothing when it fails afterwards', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const dom = installDom();
    const previousError = console.error;
    console.error = () => {};
    try {
        let calls = 0;
        let release;
        globalThis.fetch = async () => {
            calls += 1;
            if (calls === 1) throw new Error('Failed to fetch');
            return new Promise((resolve) => { release = () => resolve({ status: 502, ok: false }); });
        };
        const dispose = await dom.init();
        overlayIn(dom.body).querySelector('[data-onboarding-retry]').click();
        await tick();
        dispose();
        release();
        await tick();
        t.mock.timers.tick(120000);
        await tick();
        assert.equal(calls, 2, 'the late failure of a disposed check starts no chain');
    } finally {
        console.error = previousError;
        dom.restore();
    }
});

// --- The shared escape helper the onboarding modules now import ------------

test('the shared escape helper both onboarding modules import neutralizes every HTML meta', () => {
    assert.equal(
        escapeHtmlAttr(`&<>"'\``),
        '&amp;&lt;&gt;&quot;&#39;&#96;',
    );
    // Ampersand first: any other order double-encodes the entity numbers.
    assert.equal(escapeHtmlAttr('&lt;'), '&amp;lt;');
    assert.equal(escapeHtmlAttr(null), '');
});
