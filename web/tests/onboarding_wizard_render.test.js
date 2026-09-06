// The onboarding wizard boots and renders under a bare DOM stand-in.
//
// The wizard is an IIFE: importing the module runs `render()` once, and the
// save path runs it again before the completion POST. The dead call that
// stranded every fresh desktop install on "Saving..." (issues #557/#607)
// lived at the END of `render()` — the first paint succeeded because the DOM
// was already written, so nothing short of executing the function saw the
// ReferenceError. This test executes it: a Proxy stands in for `document` and
// `window` (every element exists, every method is a no-op, every value is
// inert), so the only way the import can throw is a real defect in the
// module's own code — an undeclared name, a bad destructure, a null
// dereference on state the wizard itself owns.
//
// This is the runtime half of the class gate; `no_undef.test.js` is the static
// half (it sees every module and every path, this sees the wizard's boot path
// with real control flow).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// The SAME bootstrap the server injects into the page (`build_setup_bootstrap`
// on an empty settings document); tests/test_onboarding_wizard.py pins the
// fixture byte-for-byte against the live Python contract, so the wizard here
// walks the real step order with the real field lists.
const BOOTSTRAP = JSON.parse(readFileSync(
    new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8',
));

function inertElement() {
    const listeners = new Map();
    const target = {
        innerHTML: '', textContent: '', value: '', hidden: false, disabled: false, checked: false,
        dataset: {}, style: {}, classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
        children: [], childNodes: [], attributes: [],
        addEventListener(type, fn) { listeners.set(type, fn); },
        removeEventListener() {},
        dispatchEvent() { return true; },
        // Test-side: run the LAST listener bound for `type` (the wizard rebinds
        // on every render), the way a click would.
        fire(type, event = {}) { const fn = listeners.get(type); if (fn) fn(event); },
        setAttribute() {}, removeAttribute() {}, getAttribute: () => null, hasAttribute: () => false,
        appendChild: (c) => c, removeChild: (c) => c, replaceChildren() {}, insertBefore: (c) => c,
        querySelector: () => inertElement(), querySelectorAll: () => [], closest: () => null,
        contains: () => false, focus() {}, blur() {}, click() {}, scrollIntoView() {},
        getBoundingClientRect: () => ({ top: 0, left: 0, width: 0, height: 0, right: 0, bottom: 0 }),
        matches: () => false, remove() {},
    };
    return new Proxy(target, {
        get(obj, prop) {
            if (prop in obj) return obj[prop];
            if (typeof prop === 'symbol') return undefined;
            // Unknown property: a callable that also behaves like an inert element.
            return Object.assign(() => inertElement(), { then: undefined });
        },
        set(obj, prop, value) { obj[prop] = value; return true; },
    });
}

function inertDocument() {
    const doc = inertElement();
    const byId = new Map();
    doc.body = inertElement();
    doc.documentElement = inertElement();
    doc.head = inertElement();
    // One element per id, so a listener the wizard binds is the one a test fires.
    doc.getElementById = (id) => {
        if (!byId.has(id)) byId.set(id, inertElement());
        return byId.get(id);
    };
    doc.createElement = () => inertElement();
    doc.createTextNode = (text) => ({ textContent: String(text) });
    doc.createDocumentFragment = () => inertElement();
    doc.activeElement = null;
    doc.readyState = 'complete';
    doc.cookie = '';
    doc.title = '';
    return doc;
}

// Import the wizard once under the stand-ins (cache-busting query on the
// import URL — an ES module is evaluated once per URL) and keep them installed
// while `body` drives it; the exact prior descriptors are restored afterwards.
async function withWizard(bootstrap, query, body, { fetch, location } = {}) {
    const doc = inertDocument();
    const win = new Proxy({
        document: doc,
        location: location || { origin: 'http://127.0.0.1:8765', href: 'http://127.0.0.1:8765/onboarding', search: '', hash: '', pathname: '/onboarding' },
        navigator: { userAgent: 'node', platform: 'node', clipboard: { writeText: async () => {} } },
        localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
        __OURO_ONBOARDING_BOOTSTRAP__: bootstrap,
        addEventListener() {}, removeEventListener() {},
        setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
        requestAnimationFrame: (fn) => setTimeout(fn, 0), getComputedStyle: () => ({}),
        matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
        fetch: fetch || (async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => '' })),
        open() {}, scrollTo() {}, parent: null, pywebview: undefined,
    }, {
        get(obj, prop) { return prop in obj ? obj[prop] : undefined; },
        set(obj, prop, value) { obj[prop] = value; return true; },
    });
    // Node 22+ exposes some Web IDL globals (`navigator`) as getter-only
    // properties: a plain assignment throws before the wizard is imported.
    // Install every stand-in through defineProperty and restore the exact
    // prior descriptor afterwards, so the smoke runs the same on every Node
    // CI pins.
    const installed = {};
    const install = (name, value) => {
        installed[name] = Object.getOwnPropertyDescriptor(globalThis, name);
        Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
    };
    install('document', doc);
    install('window', win);
    install('navigator', win.navigator);
    install('localStorage', win.localStorage);
    install('location', win.location);
    install('fetch', win.fetch);
    install('requestAnimationFrame', win.requestAnimationFrame);
    install('setTimeout', win.setTimeout);
    install('setInterval', win.setInterval);
    install('clearTimeout', win.clearTimeout);
    install('clearInterval', win.clearInterval);
    try {
        // A ReferenceError here is the exact failure that shipped in 6.113.3–6.114.0.
        await import(`../modules/onboarding_wizard.js?${query}`);
        await body({ doc, win });
    } finally {
        for (const [name, descriptor] of Object.entries(installed)) {
            if (descriptor) Object.defineProperty(globalThis, name, descriptor);
            else delete globalThis[name];
        }
    }
}

// One import per step: the wizard renders `stepOrder[0]` at boot, so the
// bootstrap is rotated to put each step first and every step's renderer
// executes, the summary/save re-render included — the surface #557/#607
// actually broke on.
for (const [index, step] of BOOTSTRAP.stepOrder.entries()) {
test(`importing the onboarding wizard renders the '${step}' step without throwing`, async () => {
    const rotated = { ...BOOTSTRAP, stepOrder: [...BOOTSTRAP.stepOrder.slice(index), ...BOOTSTRAP.stepOrder.slice(0, index)] };
    await withWizard(rotated, `step=${index}`, () => {});
    assert.ok(true);
});
}

test('wizard renders and submits the edited owner draft on Finish', { timeout: 3000 }, async () => {
    // Keep the real contract and input handlers: only start at the model step,
    // after provider access, so this regression needs no subscription service.
    const modelIndex = BOOTSTRAP.stepOrder.indexOf('models');
    const bootstrap = {
        ...BOOTSTRAP,
        stepOrder: [...BOOTSTRAP.stepOrder.slice(modelIndex), ...BOOTSTRAP.stepOrder.slice(0, modelIndex)],
        initialState: {
            ...BOOTSTRAP.initialState,
            openrouterKey: 'test-key-not-a-secret',
            reviewEnforcement: 'blocking',
        },
    };
    const modelInputs = {
        'main-model': 'test/owner-main',
        'light-model': 'test/owner-light',
        'vision-model': 'test/owner-vision',
        'consciousness-model': 'test/owner-consciousness',
        'fallback-model': 'test/owner-fallback',
    };
    const requests = [];
    const replaced = [];
    let recordSubmission;
    const submitted = new Promise((resolve) => { recordSubmission = resolve; });
    let returnReceipt;
    const response = new Promise((resolve) => { returnReceipt = resolve; });
    let recordCompletion;
    const completed = new Promise((resolve) => { recordCompletion = resolve; });
    const fetch = (url, init) => {
        requests.push({ url, method: init.method, body: JSON.parse(init.body) });
        recordSubmission();
        return response;
    };
    const location = {
        origin: 'http://127.0.0.1:8765',
        replace(href) { replaced.push(href); recordCompletion(); },
    };

    await withWizard(bootstrap, 'save=owner-draft', async ({ doc }) => {
        for (const [id, value] of Object.entries(modelInputs)) {
            const input = doc.getElementById(id);
            input.value = value;
            input.fire('input');
        }
        doc.getElementById('next-btn').fire('click');   // Models → review mode.
        doc.getElementById('next-btn').fire('click');   // Review mode → budget.
        for (const [id, value] of [['total-budget', '15.5'], ['per-task-budget', '5.25']]) {
            const input = doc.getElementById(id);
            input.value = value;
            input.fire('input');
        }
        doc.getElementById('next-btn').fire('click');   // Budget → summary.
        assert.match(doc.getElementById('root').innerHTML, /Start Ouroboros/);
        doc.getElementById('next-btn').fire('click');
        await submitted;
        assert.equal(requests.length, 1);
        assert.equal(requests[0].url, '/api/onboarding/complete');
        assert.equal(requests[0].method, 'POST');
        const expected = {
            OPENROUTER_API_KEY: 'test-key-not-a-secret',
            OUROBOROS_MODEL: 'test/owner-main',
            OUROBOROS_MODEL_LIGHT: 'test/owner-light',
            OUROBOROS_MODEL_VISION: 'test/owner-vision',
            OUROBOROS_MODEL_CONSCIOUSNESS: 'test/owner-consciousness',
            OUROBOROS_MODEL_FALLBACKS: 'test/owner-fallback',
            TOTAL_BUDGET: 15.5,
            OUROBOROS_PER_TASK_COST_USD: 5.25,
            OUROBOROS_REVIEW_ENFORCEMENT: 'blocking',
            OUROBOROS_RUNTIME_MODE: 'advanced',
            subscriptionsConnected: false,
            skipSubscriptionPresets: false,
        };
        for (const [key, value] of Object.entries(expected)) assert.equal(requests[0].body[key], value, key);
        assert.deepEqual(replaced, [], 'a pending save is not a completion');
        assert.equal(doc.getElementById('next-btn').disabled, true);
        returnReceipt({
            ok: true, status: 200,
            json: async () => ({ ok: true, runtime_mode: 'advanced', restart_required: false }),
        });
        await completed;
        assert.deepEqual(replaced, ['/']);
        assert.equal(requests.length, 1, 'completion must not start a second settings write');
    }, { fetch, location });
});

test('a 503 settings_save_timeout keeps the wizard open with "Check status", which proceeds once the probe says the save landed', async () => {
    // The completion POST runs through the shared bounded settings writer:
    // past twice the document-lock bound it answers 503 `settings_save_timeout`
    // with `saved: null` — the save is STILL RUNNING in the server, so neither
    // "saved" nor "nothing saved" is true. The wizard must not offer a blind
    // retry (a second write over an unknown first); it re-reads the readiness
    // probe on request and proceeds exactly as a receipt would once it passes.
    const summaryFirst = {
        ...BOOTSTRAP,
        stepOrder: ['summary', ...BOOTSTRAP.stepOrder.filter((step) => step !== 'summary')],
        initialState: { ...BOOTSTRAP.initialState, openrouterKey: 'sk-or-v1-abcdefghijklmnop' },
    };
    const calls = [];
    const fetch = async (url, init = {}) => {
        calls.push(`${init.method || 'GET'} ${String(url)}`);
        if (String(url) === '/api/onboarding/complete') {
            return {
                ok: false, status: 503, text: async () => '',
                json: async () => ({
                    error: 'the settings save is still running in the server after 60s and was left to finish on its own; reload Settings to see what landed',
                    code: 'settings_save_timeout', saved: null,
                }),
            };
        }
        if (String(url) === '/api/onboarding') {
            return { ok: true, status: 204, text: async () => '', json: async () => { throw new Error('no body'); } };
        }
        return { ok: true, status: 200, json: async () => ({}), text: async () => '' };
    };
    const replaced = [];
    const location = {
        origin: 'http://127.0.0.1:8765', href: 'http://127.0.0.1:8765/onboarding', search: '', hash: '', pathname: '/onboarding',
        replace: (href) => replaced.push(href),
    };
    const settle = async () => { for (let i = 0; i < 20; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

    await withWizard(summaryFirst, 'save=timeout', async ({ doc }) => {
        doc.getElementById('next-btn').fire('click');   // "Start Ouroboros"
        await settle();
        assert.deepEqual(calls, ['POST /api/onboarding/complete']);
        const html = doc.getElementById('root').innerHTML;
        assert.match(html, /is unknown — the save is still running/);
        assert.match(html, /id="check-save-btn"[^>]*>Check status</);
        assert.doesNotMatch(html, /Saving\.\.\./, 'the wizard is not stuck on Saving...');
        assert.equal(replaced.length, 0, 'an unknown save is not announced as a completion');
        // "Check status" is the ONE primary action while the outcome is
        // unknown; the re-submit stays offered, but as an explicit secondary
        // "Retry save" — the default "Start Ouroboros" beside it re-POSTed a
        // second write over the first (rc.15 review MINOR 3).
        assert.match(html, /class="btn btn-primary" id="check-save-btn"/, 'Check status is the primary action');
        assert.match(html, /class="btn btn-secondary" id="next-btn"[^>]*>Retry save</, 'the retry is explicit and secondary');
        assert.equal((html.match(/btn-primary/g) || []).length, 1, 'exactly one primary action');

        doc.getElementById('next-btn').fire('click');   // the explicit retry path stays reachable
        await settle();
        assert.deepEqual(calls, ['POST /api/onboarding/complete', 'POST /api/onboarding/complete']);
        assert.match(doc.getElementById('root').innerHTML, /class="btn btn-primary" id="check-save-btn"/);

        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.deepEqual(calls, ['POST /api/onboarding/complete', 'POST /api/onboarding/complete', 'GET /api/onboarding']);
        // 204 = the readiness gate passes: the transaction landed. The plain
        // browser shell proceeds as it does on a receipt (no restart needed —
        // the runtime mode the wizard holds is the one the page loaded with).
        assert.deepEqual(replaced, ['/']);
    }, { fetch, location });
});
