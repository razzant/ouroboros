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
        children: [], childNodes: [], attributes: [], firstElementChild: null,
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
    const localModes = BOOTSTRAP.contract.localRoutingModes.map(({ value }) => {
        const button = inertElement();
        button.getAttribute = (key) => key === 'data-local-mode' ? value : null;
        return button;
    });
    doc.getElementById('root').querySelectorAll = (selector) => selector === '[data-local-mode]' ? localModes : [];
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
    const listeners = new Map();
    const win = new Proxy({
        document: doc,
        location: location || { origin: 'http://127.0.0.1:8765', href: 'http://127.0.0.1:8765/onboarding', search: '', hash: '', pathname: '/onboarding' },
        navigator: { userAgent: 'node', platform: 'node', clipboard: { writeText: async () => {} } },
        localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
        __OURO_ONBOARDING_BOOTSTRAP__: bootstrap,
        addEventListener(type, listener) { listeners.set(listener, type); },
        removeEventListener(type, listener) { listeners.delete(listener); },
        setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
        requestAnimationFrame: (fn) => setTimeout(fn, 0), getComputedStyle: () => ({}),
        matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
        fetch: fetch || (async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => '' })),
        open() {}, scrollTo() {}, parent: null, pywebview: undefined,
    }, {
        get(obj, prop) { return prop in obj ? obj[prop] : undefined; },
        set(obj, prop, value) { obj[prop] = value; return true; },
    });
    doc.defaultView = win;
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
        for (const [listener, type] of listeners) {
            if (type === 'pagehide') listener({ persisted: false });
        }
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

test('the setup contract renders Cyber Pro beside the independent Blocking choice', async () => {
    const reviewIndex = BOOTSTRAP.stepOrder.indexOf('review_mode');
    const bootstrap = {
        ...BOOTSTRAP,
        stepOrder: [...BOOTSTRAP.stepOrder.slice(reviewIndex), ...BOOTSTRAP.stepOrder.slice(0, reviewIndex)],
    };
    await withWizard(bootstrap, 'cyber-pro-grid', async ({ doc }) => {
        const html = doc.getElementById('root').innerHTML;
        assert.match(html, /wizard-choice-grid four/);
        assert.match(html, /data-runtime-mode="cyber_pro"[\s\S]*Cyber Pro/);
        assert.match(html, /data-review-mode="blocking"[\s\S]*Blocking/);
    });
});

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
    const requests = [];
    // The summary step's Language control reads the translation memory when it mounts (once per
    // page; settings_language.test.js pins that). This test is about the completion write, so the
    // memory read is left out of the request list it asserts.
    const calls = { filter: (fn) => requests.filter(fn) };
    Object.defineProperty(calls, 'list', { get: () => requests.filter((call) => call !== 'GET /api/ui/i18n') });
    const fetch = async (url, init = {}) => {
        requests.push(`${init.method || 'GET'} ${String(url)}`);
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
        assert.deepEqual(calls.list, ['POST /api/onboarding/complete']);
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
        assert.deepEqual(calls.list, ['POST /api/onboarding/complete', 'POST /api/onboarding/complete']);
        assert.match(doc.getElementById('root').innerHTML, /class="btn btn-primary" id="check-save-btn"/);

        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.deepEqual(calls.list, ['POST /api/onboarding/complete', 'POST /api/onboarding/complete', 'GET /api/onboarding']);
        // 204 = the readiness gate passes: the transaction landed. The plain
        // browser shell proceeds as it does on a receipt (no restart needed —
        // the runtime mode the wizard holds is the one the page loaded with).
        assert.deepEqual(replaced, ['/']);
    }, { fetch, location });
});


test('the summary step stages the language and applies it only after the completion transaction', async () => {
    // The harness renders inert elements, so the control itself is exercised in settings_language.test.js
    // (`stage` hands the choice over instead of writing). The wizard's part is the order: bind in
    // staging mode, then the one language writer after `completeOnboardingAtomically` and before the
    // completion is announced — a write before it would create settings.json ahead of the transaction.
    const fs = await import('node:fs/promises');
    const source = await fs.readFile(new URL('../modules/onboarding_wizard.js', import.meta.url), 'utf8');
    assert.match(source, /stage: \(value\) => \{ state\.languageChoice = value; \} \}\)/);
    const complete = source.indexOf('const result = await completeOnboardingAtomically(payload);');
    const apply = source.indexOf('await applyStagedLanguage();', complete);
    const announce = source.indexOf('announceCompletion(result);', complete);
    assert.ok(complete > 0 && apply > complete && announce > apply, 'complete → apply the staged language → announce');
    assert.match(source, /await saveLanguageChoice\(choice\)/, 'the shared writer, not a second path');
    // The other verified-success path — a save that timed out and was then confirmed by the
    // readiness probe — applies the staged language too, and the re-mounted control gets the draft back.
    const recovered = source.indexOf('if (status === 204) {');
    assert.ok(recovered > 0 && source.indexOf('if (!(await applyStagedLanguage())) {', recovered) < source.indexOf('announceCompletion(', recovered));
    assert.match(source, /bindLanguageSettings\(root, \{ staged: state\.languageChoice, stage:/);
    // A completion whose settings landed but whose later step failed (`saved: true`) applies it as well;
    // a refusal that wrote nothing (`saved: false`) must not — that would be the write before completion again.
    assert.match(source, /if \(notice\.saved\) await applyStagedLanguage\(\);/);
});

test('a staged language whose writer is still busy keeps the wizard open; the next check applies it and proceeds', async () => {
    // The completion timed out (503, `saved: null`) and the readiness probe then says it landed, while the
    // finishing save still holds the settings document: the language writer answers 503. The staged choice
    // must not be dropped with the page — the wizard stays, and the next "Check status" applies it.
    const staged = {
        ...BOOTSTRAP,
        stepOrder: ['summary', ...BOOTSTRAP.stepOrder.filter((step) => step !== 'summary')],
        initialState: { ...BOOTSTRAP.initialState, openrouterKey: 'sk-or-v1-abcdefghijklmnop', languageChoice: 'en' },
    };
    const requests = [];
    const languageAnswers = [
        { ok: false, status: 503, body: { ok: false, error: 'another settings save is still running', code: 'settings_busy' } },
        { ok: true, status: 200, body: { language: '', english: true, chosen: true, entries: {}, revision: 0, languages: [] } },
    ];
    const fetch = async (url, init = {}) => {
        const call = `${init.method || 'GET'} ${String(url)}`;
        if (call !== 'GET /api/ui/i18n') requests.push(call);
        if (String(url) === '/api/onboarding/complete') {
            return { ok: false, status: 503, text: async () => '', json: async () => ({ error: 'still running', code: 'settings_save_timeout', saved: null }) };
        }
        if (String(url) === '/api/onboarding') return { ok: true, status: 204, text: async () => '', json: async () => { throw new Error('no body'); } };
        if (String(url) === '/api/ui/i18n/language') {
            const answer = languageAnswers.shift();
            return { ok: answer.ok, status: answer.status, text: async () => '', json: async () => answer.body };
        }
        return { ok: true, status: 200, json: async () => ({}), text: async () => '' };
    };
    const replaced = [];
    const location = {
        origin: 'http://127.0.0.1:8765', href: 'http://127.0.0.1:8765/onboarding', search: '', hash: '', pathname: '/onboarding',
        replace: (href) => replaced.push(href),
    };
    const settle = async () => { for (let i = 0; i < 20; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
    const warn = console.warn;
    console.warn = () => {};
    try {
        await withWizard(staged, 'save=timeout-language-busy', async ({ doc }) => {
            doc.getElementById('next-btn').fire('click');   // "Start Ouroboros" → 503, outcome unknown
            await settle();
            assert.deepEqual(requests, ['POST /api/onboarding/complete'], 'nothing is written for the language before the save is known to have landed');
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.deepEqual(requests.slice(1), ['GET /api/onboarding', 'POST /api/ui/i18n/language']);
            assert.equal(replaced.length, 0, 'the wizard does not leave with the choice unapplied');
            const html = doc.getElementById('root').innerHTML;
            assert.match(html, /Setup is saved\. The interface language is still being applied/);
            assert.match(html, /id="check-save-btn"/, 'the same action retries');
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.deepEqual(requests.slice(3), ['GET /api/onboarding', 'POST /api/ui/i18n/language']);
            assert.deepEqual(replaced, ['/'], 'applied: the wizard proceeds as it does on a receipt');
        }, { fetch, location });
    } finally {
        console.warn = warn;
    }
});

test('a staged language the gateway refuses for good does not hold the finished setup', async () => {
    // A failure a retry cannot fix (here: the gateway cannot work out the typed name) is logged and the
    // completed setup proceeds; the owner picks the language in Settings → Appearance, which says why.
    const staged = {
        ...BOOTSTRAP,
        stepOrder: ['summary', ...BOOTSTRAP.stepOrder.filter((step) => step !== 'summary')],
        initialState: { ...BOOTSTRAP.initialState, openrouterKey: 'sk-or-v1-abcdefghijklmnop', languageChoice: 'Quenya' },
    };
    const requests = [];
    const fetch = async (url, init = {}) => {
        const call = `${init.method || 'GET'} ${String(url)}`;
        if (call !== 'GET /api/ui/i18n') requests.push(call);
        if (String(url) === '/api/onboarding/complete') {
            return { ok: false, status: 503, text: async () => '', json: async () => ({ error: 'still running', code: 'settings_save_timeout', saved: null }) };
        }
        if (String(url) === '/api/onboarding') return { ok: true, status: 204, text: async () => '', json: async () => { throw new Error('no body'); } };
        if (String(url) === '/api/ui/i18n/language') {
            return { ok: false, status: 400, text: async () => '', json: async () => ({ ok: false, saved: false, code: 'language_needs_model', error: 'no credentialed model' }) };
        }
        return { ok: true, status: 200, json: async () => ({}), text: async () => '' };
    };
    const replaced = [];
    const location = {
        origin: 'http://127.0.0.1:8765', href: 'http://127.0.0.1:8765/onboarding', search: '', hash: '', pathname: '/onboarding',
        replace: (href) => replaced.push(href),
    };
    const settle = async () => { for (let i = 0; i < 20; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
    const warn = console.warn;
    const warned = [];
    console.warn = (...args) => { warned.push(String(args[0])); };
    try {
        await withWizard(staged, 'save=timeout-language-refused', async ({ doc }) => {
            doc.getElementById('next-btn').fire('click');
            await settle();
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.deepEqual(requests, ['POST /api/onboarding/complete', 'GET /api/onboarding', 'POST /api/ui/i18n/language']);
            assert.deepEqual(replaced, ['/']);
            assert.ok(warned.some((line) => line.includes('language not applied yet')));
        }, { fetch, location });
    } finally {
        console.warn = warn;
    }
});

test('the summary warns, and still offers Start, when Blocking review has no reviewer that reads the work', async () => {
    // Completion closes the wizard, so a save warning would never be read: the summary says it before the save.
    const reviewer = (id, target, delivery) => ({
        subagent_id: id, recommended_use: `use ${id}`, review_eligible: true,
        route: { kind: 'api_model', target_id: target }, ...(delivery ? { delivery } : {}),
    });
    const packetOnly = { enabled: true, items: [reviewer('packet-a', 'openai/gpt-5.6-sol', 'packet'), reviewer('packet-b', 'anthropic/claude-fable-5', 'packet')] };
    const withReader = { ...packetOnly, items: [...packetOnly.items, reviewer('reader', 'x-ai/grok-4.6')] };
    const summaryHtml = async (query, initial) => {
        let html = '';
        const bootstrap = {
            ...BOOTSTRAP,
            stepOrder: ['summary', ...BOOTSTRAP.stepOrder.filter((step) => step !== 'summary')],
            initialState: { ...BOOTSTRAP.initialState, ...initial },
        };
        await withWizard(bootstrap, query, ({ doc }) => { html = doc.getElementById('root').innerHTML; });
        return html;
    };
    const warning = /data-review-pool-warning>Every reviewer is a Packet row, so none reads the repository/;

    const warned = await summaryHtml('pool=packet-blocking', { reviewEnforcement: 'blocking', availableSubagents: packetOnly });
    assert.match(warned, warning);
    assert.match(warned, /Start Ouroboros/, 'a warning, never a refusal');
    for (const [query, initial] of [
        ['pool=packet-advisory', { reviewEnforcement: 'advisory', availableSubagents: packetOnly }],
        ['pool=reader-blocking', { reviewEnforcement: 'blocking', availableSubagents: withReader }],
        ['pool=packet-cyber-pro', { reviewEnforcement: 'blocking', runtimeMode: 'cyber_pro', availableSubagents: packetOnly }],
    ]) {
        assert.doesNotMatch(await summaryHtml(query, initial), /data-review-pool-warning/, query);
    }
});

// PR #1560: walk the real wizard from Accounts to the completion POST. A fresh Z.ai-only
// setup saves the provider's image-capable Vision; an owner's edit and an existing
// install's saved value, blank included, are what gets saved instead.
async function finishFromAccounts(bootstrap, query, { inputs = {}, models = {}, revisit = false,
    localMode = '', revisitInputs = {}, revisitLocalMode = '', shortcut = false, previews = [] } = {}) {
    const posted = [];
    const subscription = JSON.parse(readFileSync(new URL('./fixtures/subscription_setup.json', import.meta.url), 'utf8'));
    let connected = false;
    const fetch = async (url, init = {}) => {
        let body = {};
        if (String(url) === '/api/onboarding/complete') {
            posted.push(JSON.parse(init.body));
            body = { ok: true, runtime_mode: 'advanced', restart_required: false };
        } else if (String(url) === '/api/onboarding/subagents/preview') {
            const payload = JSON.parse(init.body);
            previews.push(payload);
            // An API-only draft: the server proposes no model settings, only Main's reviewers.
            const route = { kind: 'api_chat', target_id: JSON.parse(init.body).OUROBOROS_MODEL };
            body = { ok: true, model_settings: {}, available_subagents: { enabled: true, items: [] },
                reviewer_slots: JSON.stringify({ triad: [{ slot_id: 'triad_1', route }], scope: [{ slot_id: 'scope_1', route }], advisory: { enabled: true, route } }) };
            body.model_settings = Object.fromEntries(Object.keys(subscription.preview.model_settings).map((key) => [key, payload[key]]));
        } else if (String(url) === '/api/claudexor/status') {
            body = { ...subscription.status, profiles: { ...subscription.status.profiles,
                profiles: connected ? subscription.status.profiles.profiles : [] } };
        } else if (String(url) === '/api/model-catalog') {
            body = subscription.catalog;
        }
        return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) };
    };
    const location = { origin: 'http://127.0.0.1:8765', href: 'http://127.0.0.1:8765/onboarding', search: '', hash: '', pathname: '/onboarding', replace() {} };
    const settle = async () => { for (let i = 0; i < 20; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
    await withWizard(bootstrap, query, async ({ doc }) => {
        const type = (values) => Object.entries(values).forEach(([id, value]) => {
            const input = doc.getElementById(id);
            input.value = value;
            input.fire('input');
        });
        const click = async (id) => { doc.getElementById(id).fire('click'); await settle(); };
        const routeLocal = (mode) => {
            if (mode) doc.getElementById('root').querySelectorAll('[data-local-mode]')
                .find((button) => button.getAttribute('data-local-mode') === mode).fire('click');
        };
        type(inputs);
        routeLocal(localMode);
        await click('next-btn');   // Accounts → Models.
        type(models);
        if (revisit) {
            await click('back-btn');
            if (shortcut) {
                connected = true;
                await (await import('../modules/claudexor_status_store.js')).claudexorStatus.refresh();
                await settle();
                assert.equal(doc.getElementById('quick-start-btn').hidden, false);
            }
            type(revisitInputs);
            routeLocal(revisitLocalMode);
            await click(shortcut ? 'quick-start-btn' : 'next-btn');
        }
        for (let step = 0; step < (shortcut ? 1 : 4); step += 1) await click('next-btn');   // … → Start Ouroboros.
    }, { fetch, location });
    assert.equal(posted.length, 1, 'the walk reached the completion POST');
    return posted[0];
}

const ZAI_KEY = 'zai-test-key-not-a-secret';
const FRESH = { ...BOOTSTRAP, freshInstall: true };

for (const plan of ['', 'payg', 'coding']) {
    test(`a fresh Z.ai-only setup (plan '${plan}') saves the image-capable Vision default`, async () => {
        const body = await finishFromAccounts(FRESH, `zai-fresh-${plan}`, { inputs: { 'zai-key': ZAI_KEY, 'zai-plan': plan } });
        assert.deepEqual([body.ZAI_API_KEY, body.ZAI_PLAN, body.OUROBOROS_MODEL, body.OUROBOROS_MODEL_VISION],
            [ZAI_KEY, plan, 'zai::glm-5.3', BOOTSTRAP.modelDefaults.zai.vision]);
        assert.equal(BOOTSTRAP.modelDefaults.zai.vision, 'zai::glm-5.3-flash');
    });
}

test('an owner who clears or replaces the recommended Vision keeps that choice across Back/Next', async () => {
    for (const value of ['', 'zai::glm-ocr']) {
        const body = await finishFromAccounts(FRESH, `zai-owner-${value || 'cleared'}`,
            { inputs: { 'zai-key': ZAI_KEY }, models: { 'vision-model': value }, revisit: true });
        assert.equal(body.OUROBOROS_MODEL_VISION, value);
    }
});

test('a reopened wizard saves an existing install\'s Vision as it was, blank or custom', async () => {
    for (const saved of ['', 'zai::glm-ocr']) {
        const initialState = { ...BOOTSTRAP.initialState, zaiKey: BOOTSTRAP.secretPlaceholder, mainModel: 'zai::glm-5.3',
            lightModel: 'zai::glm-5.3-flash', fallbackModel: 'zai::glm-5.3-flash', visionModel: saved };
        const body = await finishFromAccounts({ ...BOOTSTRAP, freshInstall: false, initialState }, `zai-existing-${saved || 'blank'}`);
        assert.equal(body.OUROBOROS_MODEL_VISION, saved);
    }
});

test('the Z.ai Vision default stays inside a Z.ai-only setup', async () => {
    for (const [query, inputs, models] of [
        ['openrouter', { 'zai-key': ZAI_KEY, 'openrouter-key': 'sk-or-v1-test-not-a-secret' }, {}],
        ['direct-multi', { 'zai-key': ZAI_KEY, 'openai-key': 'sk-openai-test-not-a-secret' }, { 'main-model': 'zai::glm-5.3' }],
        ['local-only', { 'local-source': '/models/local-test.gguf' }, {}],
    ]) {
        const body = await finishFromAccounts(FRESH, `zai-boundary-${query}`, { inputs, models });
        assert.equal(body.OUROBOROS_MODEL_VISION, '', query);
    }
});

test('fresh local Main with a Z.ai key has no remote Vision suggestion in either input order', async () => {
    for (const inputs of [
        { 'zai-key': ZAI_KEY, 'local-source': '/models/local-test.gguf' },
        { 'local-source': '/models/local-test.gguf', 'zai-key': ZAI_KEY },
    ]) {
        const body = await finishFromAccounts(FRESH, `zai-local-${Object.keys(inputs)[0]}`, { inputs, localMode: 'all' });
        assert.equal(body.LOCAL_ROUTING_MODE, 'all');
        assert.equal(body.OUROBOROS_MODEL_VISION, '');
    }
});

test('moving Main local withdraws only the generated Vision, including after a Main edit', async () => {
    for (const [name, models, expected] of [
        ['untouched', {}, ''],
        ['main-edit', { 'main-model': 'zai::owner-main' }, ''],
        ['clear', { 'vision-model': '' }, ''],
        ['custom', { 'vision-model': 'zai::glm-ocr' }, 'zai::glm-ocr'],
        ['explicit-default', { 'vision-model': 'zai::glm-5.3-flash' }, 'zai::glm-5.3-flash'],
    ]) {
        const body = await finishFromAccounts(FRESH, `zai-remote-local-${name}`, {
            inputs: { 'zai-key': ZAI_KEY }, models, revisit: true,
            revisitInputs: { 'local-source': '/models/local-test.gguf' }, revisitLocalMode: 'all',
        });
        assert.equal(body.LOCAL_ROUTING_MODE, 'all');
        assert.equal(body.OUROBOROS_MODEL_VISION, expected, name);
        if (name === 'main-edit') assert.equal(body.OUROBOROS_MODEL, 'zai::owner-main');
    }
});

test('local fallback still permits the remote Main Vision suggestion', async () => {
    const body = await finishFromAccounts(FRESH, 'zai-local-fallback', {
        inputs: { 'zai-key': ZAI_KEY, 'local-source': '/models/local-test.gguf' }, localMode: 'fallback',
    });
    assert.equal(body.LOCAL_ROUTING_MODE, 'fallback');
    assert.equal(body.OUROBOROS_MODEL_VISION, 'zai::glm-5.3-flash');
});

for (const [name, models, expected] of [
    ['generated', {}, ''],
    ['main-edited', { 'main-model': 'zai::owner-main' }, ''],
    ['deliberate-Flash', { 'vision-model': 'zai::glm-5.3-flash' }, 'zai::glm-5.3-flash'],
]) {
    test(`Review & start reconciles ${name} Vision before preview when Main moves local`, async () => {
        const previews = [];
        const body = await finishFromAccounts(FRESH, `zai-shortcut-${name}`, {
            inputs: { 'zai-key': ZAI_KEY }, models, revisit: true, shortcut: true, previews,
            revisitInputs: { 'local-source': '/models/local-test.gguf' }, revisitLocalMode: 'all',
        });
        const localPreviews = previews.filter((draft) => draft.LOCAL_ROUTING_MODE === 'all');
        assert.ok(localPreviews.length > 0, 'the shortcut requested a preview of the local draft');
        assert.ok(localPreviews.every((draft) => draft.OUROBOROS_MODEL_VISION === expected));
        assert.equal(body.subscriptionsConnected, true);
        assert.equal(body.LOCAL_ROUTING_MODE, 'all');
        assert.equal(body.OUROBOROS_MODEL_VISION, expected);
        if (name === 'main-edited') assert.equal(body.OUROBOROS_MODEL, 'zai::owner-main');
    });
}
