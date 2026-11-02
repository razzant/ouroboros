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
// Reads that write nothing: the language list and the Summary's starter skills list.
const PASSIVE_READS = new Set(['GET /api/ui/i18n', 'GET /api/onboarding/starter-skills']);

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
async function withWizard(bootstrap, query, body, { fetch, location, prepare } = {}) {
    const doc = inertDocument();
    prepare?.(doc);  // stand-ins a step binds at import time (its choice buttons, a held note)
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

// IMPL_SPEC §5, verbatim: what a Reviewer tick means per strictness (Cyber Pro outranks enforcement).
const TICK = {
    blocking: 'Counts toward review: every ticked reviewer checks each change Ouroboros makes to itself, each skill and each required result; a critical finding stops it.',
    advisory: 'Counts when chosen: for each check Ouroboros asks one, several or all ticked reviewers and writes why; it never skips a check that is owed.',
    cyber_pro: 'Trusted to judge: Ouroboros decides whom to ask, if anyone; reviews inform and never block.',
};

test('the Review step says what a Reviewer tick means under the chosen mode, repainted in place', async () => {
    const reviewIndex = BOOTSTRAP.stepOrder.indexOf('review_mode');
    const bootstrap = {
        ...BOOTSTRAP,
        stepOrder: [...BOOTSTRAP.stepOrder.slice(reviewIndex), ...BOOTSTRAP.stepOrder.slice(0, reviewIndex)],
        initialState: {
            ...BOOTSTRAP.initialState, reviewEnforcement: 'advisory', runtimeMode: 'advanced',
            availableSubagents: { enabled: true, items: [{ subagent_id: 'reviewer_one', review_eligible: true,
                recommended_use: 'Reviews changes.', route: { kind: 'api_model', target_id: 'openai/gpt-5.6-luna' } }] },
        },
    };
    const note = { textContent: '' };  // the held node: a choice never rebuilds the step
    const choices = (attribute, values) => values.map((value) => {
        const button = inertElement();
        button.getAttribute = (key) => (key === attribute ? value : null);
        return button;
    });
    const [advisory, blocking] = choices('data-review-mode', ['advisory', 'blocking']);
    const access = choices('data-runtime-mode', ['light', 'advanced', 'pro', 'cyber_pro']);
    const lead = 'Reviewers: 1 — the rows marked Reviewer under Models → Available subagents. ';
    const prepare = (doc) => {
        const root = doc.getElementById('root');
        const others = root.querySelectorAll;
        root.querySelector = (selector) => (selector === '[data-reviewers-note]' ? note : inertElement());
        root.querySelectorAll = (selector) => ({ '[data-review-mode]': [advisory, blocking], '[data-runtime-mode]': access }[selector]
            || others(selector));
    };
    await withWizard(bootstrap, 'reviewers-note', async ({ doc }) => {
        const html = doc.getElementById('root').innerHTML;
        assert.ok(html.includes(`data-reviewers-note>${lead}${TICK.advisory}</div>`), html.match(/data-reviewers-note>[^<]*/)?.[0]);
        assert.doesNotMatch(html, /each of them reviews every change|composes the panel from them/);
        blocking.fire('click');
        assert.equal(note.textContent, lead + TICK.blocking);
        access[3].fire('click');  // Cyber Pro Access: findings never block, whatever the enforcement kept
        assert.equal(note.textContent, lead + TICK.cyber_pro);
        advisory.fire('click');
        assert.equal(note.textContent, lead + TICK.cyber_pro);
        access[1].fire('click');
        assert.equal(note.textContent, lead + TICK.advisory);
    }, { prepare });
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

// The completion endpoint as an unknown save meets it. Every POST records the attempt id it carried and
// answers `posted` (a function of the POST's number; an Error is a reply that never arrived); `GET ?attempt=<id>`
// answers `attempts[id]`, by
// default "not confirmed". The readiness probe always answers 204, as it does on any install whose provider
// was saved before: exactly the answer that used to confirm a save it knew nothing about. A call for which
// `server.hold(call)` is true waits for `server.release(prefix)`, and is answered as things stand then.
const SAVE_TIMEOUT = { status: 503, body: { code: 'settings_save_timeout', saved: null,
    error: 'the settings save is still running in the server after 60s and was left to finish on its own; reload Settings to see what landed' } };
const receipt = (runtimeMode, restartRequired) => ({ ok: true, status: 'saved', runtime_mode: runtimeMode,
    restart_required: restartRequired, preset: { applied: false, reason: 'not_requested', harnesses: [], receipt: {} } });
const SUMMARY_FIRST = {
    ...BOOTSTRAP,
    stepOrder: ['summary', ...BOOTSTRAP.stepOrder.filter((step) => step !== 'summary')],
    initialState: { ...BOOTSTRAP.initialState, openrouterKey: 'sk-or-v1-abcdefghijklmnop' },
};
const settle = async () => { for (let i = 0; i < 20; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

function completionServer({ posted = SAVE_TIMEOUT, attempts = {}, language = null, starter = {} } = {}) {
    const server = { requests: [], sent: [], bodies: [], attempts, replaced: [], held: [], hold: () => false };
    const answer = (status, body) => ({ ok: status >= 200 && status < 300, status, text: async () => '', json: async () => body });
    server.fetch = async (url, init = {}) => {
        const call = `${init.method || 'GET'} ${String(url)}`;
        if (!PASSIVE_READS.has(call)) server.requests.push(call);
        if (call === 'POST /api/onboarding/complete') server.bodies.push(JSON.parse(init.body));
        if (call === 'POST /api/onboarding/complete') server.sent.push(server.bodies.at(-1).attempt_id);
        if (server.hold(call)) await new Promise((go) => server.held.push({ call, go }));
        if (call === 'POST /api/onboarding/complete') {
            const reply = typeof posted === 'function' ? posted(server.sent.length) : posted;
            if (reply instanceof Error) throw reply;
            return answer(reply.status, reply.body);
        }
        const asked = /^\/api\/onboarding\/complete\?attempt=(.+)$/.exec(String(url));
        if (asked) return answer(200, server.attempts[decodeURIComponent(asked[1])] || { saved: null });
        if (String(url) === '/api/onboarding') return { ok: true, status: 204, text: async () => '', json: async () => { throw new Error('no body'); } };
        if (String(url) === '/api/ui/i18n/language' && language) return answer(...language.shift());
        if (String(url) === '/api/onboarding/starter-skills') return answer(200, starter);
        return answer(200, {});
    };
    server.location = {
        origin: 'http://127.0.0.1:8765', href: 'http://127.0.0.1:8765/onboarding', search: '', hash: '', pathname: '/onboarding',
        replace: (href) => server.replaced.push(href),
    };
    server.readBack = (index = -1) => `GET /api/onboarding/complete?attempt=${server.sent.at(index)}`;
    server.release = (prefix) => server.held.splice(server.held.findIndex(({ call }) => call.startsWith(prefix)), 1)[0].go();
    return server;
}

test('an unknown save keeps the draft; Check status reads back THIS attempt, which a ready older provider never confirms', async () => {
    // The completion POST answered 503 `settings_save_timeout` (`saved: null`): whether the bytes landed is
    // unknown, so a retry would be a second write. "Check status" used to accept the readiness probe's 204,
    // which every reopened install already passes with its OLD provider, and left the wizard on a save
    // nobody had seen land. It now asks about the attempt id this save carried.
    const server = completionServer();
    await withWizard(SUMMARY_FIRST, 'save=unknown-not-confirmed', async ({ doc }) => {
        doc.getElementById('next-btn').fire('click');   // "Start Ouroboros"
        await settle();
        assert.deepEqual(server.requests, ['POST /api/onboarding/complete']);
        assert.match(server.sent[0], /^[0-9a-f]{32}$/, 'the attempt id is made before sending');
        let html = doc.getElementById('root').innerHTML;
        assert.match(html, /The save result is unknown\. Your draft has not been discarded\. Use &quot;Check status&quot; to read the result of this save\./);
        assert.match(html, /class="footer-copy">The last save did not return a confirmed result\.</);
        assert.doesNotMatch(html, /Saving\.\.\./, 'the wizard is not stuck on Saving...');
        // "Check status" is the ONE primary action while the outcome is unknown; the re-submit stays offered,
        // as an explicit secondary "Retry save" (rc.15 review MINOR 3).
        assert.match(html, /class="btn btn-primary" id="check-save-btn"[^>]*>Check status</);
        assert.match(html, /class="btn btn-secondary" id="next-btn"[^>]*>Retry save</);
        assert.equal((html.match(/btn-primary/g) || []).length, 1, 'exactly one primary action');

        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.deepEqual(server.requests, ['POST /api/onboarding/complete', server.readBack(0)]);
        assert.deepEqual(server.replaced, [], 'an unconfirmed save is never announced as a completion');
        html = doc.getElementById('root').innerHTML;
        assert.match(html, /This save is not confirmed\. Your draft remains here\. You can check again or choose Retry save\./);
        assert.match(html, /class="btn btn-primary" id="check-save-btn"/, 'the same check stays offered');
    }, { fetch: server.fetch, location: server.location });
});

test('Retry save is a new attempt: the earlier one landing late never confirms it, stored is not done', async () => {
    const server = completionServer();
    await withWizard(SUMMARY_FIRST, 'save=retry-new-attempt', async ({ doc }) => {
        doc.getElementById('next-btn').fire('click');   // attempt A: unknown
        await settle();
        doc.getElementById('next-btn').fire('click');   // "Retry save": attempt B, also unknown
        await settle();
        assert.equal(server.sent.length, 2);
        assert.notEqual(server.sent[0], server.sent[1], 'each explicit save carries its own id');
        server.attempts[server.sent[0]] = { saved: true, receipt: receipt('advanced', false) };   // A landed after all
        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.equal(server.requests.at(-1), server.readBack(1), 'the check asks about B, the save the draft was sent with');
        assert.deepEqual(server.replaced, [], 'A finishing does not confirm B');
        assert.match(doc.getElementById('root').innerHTML, /This save is not confirmed\./);

        server.attempts[server.sent[1]] = { saved: true };   // B's bytes are on disk; its post-commit steps still run
        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.deepEqual(server.replaced, []);
        assert.match(doc.getElementById('root').innerHTML,
            /This save is stored\. Setup completion and restart requirements are not yet confirmed\./);

        server.attempts[server.sent[1]] = { saved: true, receipt: receipt('advanced', false) };
        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.deepEqual(server.replaced, ['/'], 'B\'s own receipt completes exactly as the reply would have');
        assert.equal(server.requests.filter((call) => call === 'POST /api/onboarding/complete').length, 2, 'checking never re-sends');
        assert.ok(!server.requests.includes('GET /api/onboarding'), 'readiness is never asked');
    }, { fetch: server.fetch, location: server.location });
});

test('a confirmed restart comes from the attempt\'s own receipt, never from a guess about the draft', async () => {
    // The draft keeps the runtime mode the page loaded with, which the old check read as "no restart needed".
    const server = completionServer();
    await withWizard(SUMMARY_FIRST, 'save=confirmed-restart', async ({ doc }) => {
        doc.getElementById('next-btn').fire('click');
        await settle();
        server.attempts[server.sent[0]] = { saved: true, receipt: receipt('pro', true) };
        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.deepEqual(server.replaced, [], 'a plain tab does not navigate past a pending restart');
        const html = doc.getElementById('root').innerHTML;
        assert.match(html, /Setup saved — restart to apply it/);
        assert.match(html, /Ouroboros saved <code>pro<\/code> as the runtime mode for the next boot/);
    }, { fetch: server.fetch, location: server.location });
});

test('an answer that never arrives is unknown, and a failure recorded for the attempt reads like its reply', async () => {
    const lost = completionServer({ posted: new TypeError('Failed to fetch') });
    await withWizard(SUMMARY_FIRST, 'save=reply-lost', async ({ doc }) => {
        doc.getElementById('next-btn').fire('click');
        await settle();
        const html = doc.getElementById('root').innerHTML;
        assert.match(html, /The answer to this save did not arrive \(Failed to fetch\)\. — The save result is unknown\./);
        assert.match(html, /id="check-save-btn"/, 'a lost answer is checked, not presumed failed');
    }, { fetch: lost.fetch, location: lost.location });

    const refused = completionServer();
    await withWizard(SUMMARY_FIRST, 'save=recorded-refusal', async ({ doc }) => {
        doc.getElementById('next-btn').fire('click');
        await settle();
        refused.attempts[refused.sent[0]] = { saved: false, code: 'onboarding_state_changed',
            error: 'The settings file changed while onboarding was being saved, so this save would have overwritten it; nothing was written. Try finishing again.' };
        doc.getElementById('check-save-btn').fire('click');
        await settle();
        const html = doc.getElementById('root').innerHTML;
        assert.match(html, /nothing was written\. Try finishing again\./);
        assert.doesNotMatch(html, /id="check-save-btn"/, 'a known refusal leaves nothing to check');
        assert.match(html, /class="btn btn-primary" id="next-btn"[^>]*>Start Ouroboros</);
        assert.deepEqual(refused.replaced, []);
    }, { fetch: refused.fetch, location: refused.location });
});


test('a late answer about the attempt Retry replaced never completes, fails or repaints the retry', async () => {
    // "Check status" reads A; while that read is out the owner presses Retry save, so B is the save in
    // progress. Whatever A's late answer says, it must not announce, apply the staged language, show its
    // text, release Start or clear B: B's own answer decides, and the next check asks about B.
    const lateAnswers = [
        ['receipt', { saved: true, receipt: receipt('advanced', false) }],
        ['refusal', { saved: false, code: 'onboarding_state_changed',
            error: 'The settings file changed while onboarding was being saved, so this save would have overwritten it; nothing was written. Try finishing again.' }],
        ['stored', { saved: true }],
    ];
    for (const [label, late] of lateAnswers) {
        const server = completionServer();
        const staged = { ...SUMMARY_FIRST, initialState: { ...SUMMARY_FIRST.initialState, languageChoice: 'en' } };
        await withWizard(staged, `save=late-${label}`, async ({ doc }) => {
            doc.getElementById('next-btn').fire('click');   // attempt A: unknown
            await settle();
            const [first] = server.sent;
            server.hold = (call) => call === server.readBack(0) || call === 'POST /api/onboarding/complete';
            doc.getElementById('check-save-btn').fire('click');   // the read of A is out
            await settle();
            doc.getElementById('next-btn').fire('click');   // "Retry save" meanwhile: attempt B, its answer out too
            await settle();
            assert.equal(server.sent.length, 2, label);
            server.attempts[first] = late;
            server.release(server.readBack(0));
            await settle();
            const html = doc.getElementById('root').innerHTML;
            assert.deepEqual(server.replaced, [], `${label}: A never announces while B is out`);
            assert.ok(!server.requests.includes('POST /api/ui/i18n/language'), `${label}: nor applies the language for A`);
            assert.match(html, /id="next-btn" type="button" disabled>Saving\.\.\.</, `${label}: B is still the save in progress`);
            assert.doesNotMatch(html, /nothing was written|This save is (?:stored|not confirmed)/, label);

            server.hold = () => false;
            server.release('POST /api/onboarding/complete');   // B's answer: unknown as well
            await settle();
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.equal(server.requests.at(-1), server.readBack(1), `${label}: the tracked attempt is B`);
            assert.match(doc.getElementById('root').innerHTML, /This save is not confirmed\./, label);
            assert.deepEqual(server.replaced, [], label);
        }, { fetch: server.fetch, location: server.location });
    }
});

test('while a confirmed answer is applied, Retry and Check are held, and completion is announced once', async () => {
    // Between the receipt and the announcement the wizard awaits the staged language write and the agents
    // step's release. As in the save path, Start/Retry stay held over both, so no new attempt can start
    // under a completion that is already being announced.
    const staged = { ...SUMMARY_FIRST, initialState: { ...SUMMARY_FIRST.initialState, languageChoice: 'en' } };
    const server = completionServer({ language: [[200, { language: '', english: true, chosen: true, entries: {}, revision: 0, languages: [] }]] });
    await withWizard(staged, 'save=held-while-applied', async ({ doc }) => {
        doc.getElementById('next-btn').fire('click');   // attempt A: unknown
        await settle();
        server.attempts[server.sent[0]] = { saved: true, receipt: receipt('advanced', false) };
        server.hold = (call) => call === 'POST /api/ui/i18n/language';
        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.match(doc.getElementById('root').innerHTML, /id="next-btn" type="button" disabled>Saving\.\.\.</);
        doc.getElementById('next-btn').fire('click');   // Retry while the language write is out
        doc.getElementById('check-save-btn').fire('click');   // and another check
        await settle();
        assert.deepEqual(server.requests, ['POST /api/onboarding/complete', server.readBack(0), 'POST /api/ui/i18n/language']);
        server.release('POST /api/ui/i18n/language');
        doc.getElementById('next-btn').fire('click');   // and again as that write answers
        await settle();
        assert.deepEqual(server.replaced, ['/'], 'announced exactly once');
        assert.equal(server.requests.length, 3, 'no second save and no second read');
    }, { fetch: server.fetch, location: server.location });
});

test('a newer Check status retires an older one\'s answer, so a receipt is acted on once', async () => {
    const server = completionServer();
    await withWizard(SUMMARY_FIRST, 'save=two-checks', async ({ doc }) => {
        doc.getElementById('next-btn').fire('click');
        await settle();
        server.hold = (call) => call === server.readBack(0);
        doc.getElementById('check-save-btn').fire('click');
        doc.getElementById('check-save-btn').fire('click');
        await settle();
        assert.equal(server.held.length, 2);
        server.attempts[server.sent[0]] = { saved: true, receipt: receipt('advanced', false) };
        server.release(server.readBack(0));   // the older read answers first: retired
        await settle();
        assert.deepEqual(server.replaced, []);
        server.release(server.readBack(0));
        await settle();
        assert.deepEqual(server.replaced, ['/']);
    }, { fetch: server.fetch, location: server.location });
});

test('a retry refused while the earlier save still finishes keeps that save checkable; only its own values complete', async () => {
    // A is stored (its bytes are on disk, its post-commit steps still run), so Retry B is refused with nothing
    // written. The wizard says so and keeps A to check. A's receipt completes only the values A carried: once the
    // owner edited the draft, it is reported as confirmed without completing what is shown now.
    const settingsBusy = { status: 503, body: { code: 'settings_busy', saved: false,
        error: 'another settings write is still in progress after 30s; nothing was written — try again' } };
    const budgetFirst = { ...SUMMARY_FIRST, stepOrder: ['budget', 'summary',
        ...BOOTSTRAP.stepOrder.filter((step) => !['budget', 'summary'].includes(step))] };
    for (const edited of [false, true]) {
        const server = completionServer({ posted: (count) => (count === 1 ? SAVE_TIMEOUT : settingsBusy) });
        await withWizard(budgetFirst, `save=busy-retry-${edited}`, async ({ doc }) => {
            doc.getElementById('next-btn').fire('click');   // Budget → Summary
            doc.getElementById('next-btn').fire('click');   // "Start Ouroboros": attempt A, unknown
            await settle();
            const [first] = server.sent;
            server.attempts[first] = { saved: true };
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.match(doc.getElementById('root').innerHTML, /This save is stored\./);
            if (edited) {
                doc.getElementById('back-btn').fire('click');   // Summary → Budget
                const total = doc.getElementById('total-budget');
                total.value = '42';
                total.fire('input');
                doc.getElementById('next-btn').fire('click');   // Budget → Summary
            }
            doc.getElementById('next-btn').fire('click');   // "Retry save": attempt B, refused before writing
            await settle();
            assert.equal(server.sent.length, 2);
            let html = doc.getElementById('root').innerHTML;
            assert.match(html, /nothing was written — try again — This retry was not saved\. The earlier save may still be finishing; Check status reads that save\./);
            assert.match(html, /class="btn btn-primary" id="check-save-btn"/, 'the earlier save stays checkable');
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.equal(server.requests.at(-1), server.readBack(0), 'what is checked is A, the save that may still finish');
            server.attempts[first] = { saved: true, receipt: receipt('advanced', false) };
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            html = doc.getElementById('root').innerHTML;
            if (!edited) {
                assert.deepEqual(server.replaced, ['/'], 'A carried exactly the values shown, so its receipt completes them');
                return;
            }
            assert.deepEqual(server.replaced, [], 'A\'s receipt never completes values it did not carry');
            assert.match(html, /The earlier save is confirmed, but it does not contain the values shown now\. Start Ouroboros to save them\./);
            assert.match(html, /class="btn btn-primary" id="next-btn"[^>]*>Start Ouroboros</);
            assert.doesNotMatch(html, /id="check-save-btn"/);
        }, { fetch: server.fetch, location: server.location });
    }
});

test('starter skills or hub trust changed after a stored save are values its receipt does not complete', async () => {
    // The Summary's starter ticks and hub trust ride in the same POST as the settings. A is stored, the owner
    // changes one of them, Retry B is refused before writing: A's receipt is reported without completing the
    // draft, because A never carried the choice now shown.
    const settingsBusy = { status: 503, body: { code: 'settings_busy', saved: false,
        error: 'another settings write is still in progress after 30s; nothing was written — try again' } };
    const starter = { groups: [{ category: 'research', items: [{ id: 'hub/notes', name: 'Notes', recommended: true },
        { id: 'hub/papers', name: 'Papers' }] }], hub_rows: [{ id: 'ouroboros-hub', name: 'Ouroboros hub', part: 'official', trust: 'trusted' }] };
    const changes = {
        trust: { type: 'click', target: { closest: (selector) => (selector === '[data-starter-trust]'
            ? { dataset: { starterTrust: 'ouroboros-hub', trust: 'review' } } : null) } },
        tick: { type: 'change', target: { closest: (selector) => (selector === '[data-starter-item]'
            ? { checked: false, dataset: { starterItem: 'hub/notes' }, closest: () => null } : null) } },
    };
    for (const [label, change] of Object.entries(changes)) {
        const server = completionServer({ posted: (count) => (count === 1 ? SAVE_TIMEOUT : settingsBusy), starter });
        const host = inertElement();   // the Summary's starter row mounts here on every render
        const prepare = (doc) => {
            const root = doc.getElementById('root');
            root.querySelector = (selector) => (selector === '[data-starter-skills]' ? host : inertElement());
        };
        await withWizard(SUMMARY_FIRST, `save=starter-${label}`, async ({ doc }) => {
            await settle();   // the starter list is read
            doc.getElementById('next-btn').fire('click');   // attempt A: unknown
            await settle();
            const [first] = server.sent;
            assert.deepEqual(server.bodies[0].skill_source_trust, { 'ouroboros-hub': 'trusted' }, label);
            assert.deepEqual(server.bodies[0].starter_skills, ['hub/notes'], label);
            server.attempts[first] = { saved: true };
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            host.fire(change.type, change);   // the owner changes a starter choice
            doc.getElementById('next-btn').fire('click');   // "Retry save": B, refused before writing
            await settle();
            assert.notDeepEqual([server.bodies[1].skill_source_trust, server.bodies[1].starter_skills],
                [server.bodies[0].skill_source_trust, server.bodies[0].starter_skills], `${label}: B sent the changed choice`);
            assert.match(doc.getElementById('root').innerHTML, /This retry was not saved\. The earlier save may still be finishing/, label);
            server.attempts[first] = { saved: true, receipt: receipt('advanced', false) };
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.deepEqual(server.replaced, [], `${label}: A's receipt does not complete a choice A never sent`);
            assert.match(doc.getElementById('root').innerHTML,
                /The earlier save is confirmed, but it does not contain the values shown now\. Start Ouroboros to save them\./, label);
        }, { fetch: server.fetch, location: server.location, prepare });
    }
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
    // The other verified-success path — an unknown save whose own receipt the attempt read-back returned —
    // applies the staged language too, and the re-mounted control gets the draft back.
    const recovered = source.indexOf('const outcome = await readCompletionAttempt(attempt);');
    assert.ok(recovered > 0 && source.indexOf('if (!(await applyStagedLanguage())) {', recovered) < source.indexOf('announceCompletion(', recovered));
    assert.match(source, /bindLanguageSettings\(root, \{ staged: state\.languageChoice, stage:/);
    // A completion whose settings landed but whose later step failed (`saved: true`) applies it as well;
    // a refusal that wrote nothing (`saved: false`) must not — that would be the write before completion again.
    assert.match(source, /if \(notice\.saved\) await applyStagedLanguage\(\);/);
});

test('a staged language whose writer is still busy keeps the wizard open; the next check applies it and proceeds', async () => {
    // The completion timed out (503, `saved: null`) and its read-back then returns the attempt's receipt, while
    // the finishing save still holds the settings document: the language writer answers 503. The staged choice
    // must not be dropped with the page — the wizard stays, and the next "Check status" applies it.
    const staged = { ...SUMMARY_FIRST, initialState: { ...SUMMARY_FIRST.initialState, languageChoice: 'en' } };
    const server = completionServer({ language: [
        [503, { ok: false, error: 'another settings save is still running', code: 'settings_busy' }],
        [200, { language: '', english: true, chosen: true, entries: {}, revision: 0, languages: [] }],
    ] });
    const warn = console.warn;
    console.warn = () => {};
    try {
        await withWizard(staged, 'save=timeout-language-busy', async ({ doc }) => {
            doc.getElementById('next-btn').fire('click');   // "Start Ouroboros" → 503, outcome unknown
            await settle();
            assert.deepEqual(server.requests, ['POST /api/onboarding/complete'], 'nothing is written for the language before the save is known to have landed');
            server.attempts[server.sent[0]] = { saved: true, receipt: receipt('advanced', false) };
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.deepEqual(server.requests.slice(1), [server.readBack(0), 'POST /api/ui/i18n/language']);
            assert.equal(server.replaced.length, 0, 'the wizard does not leave with the choice unapplied');
            const html = doc.getElementById('root').innerHTML;
            assert.match(html, /Setup is saved\. The interface language is still being applied/);
            assert.match(html, /id="check-save-btn"/, 'the same action retries');
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.deepEqual(server.requests.slice(3), [server.readBack(0), 'POST /api/ui/i18n/language']);
            assert.deepEqual(server.replaced, ['/'], 'applied: the wizard proceeds as it does on a receipt');
        }, { fetch: server.fetch, location: server.location });
    } finally {
        console.warn = warn;
    }
});

test('a staged language the gateway refuses for good does not hold the finished setup', async () => {
    // A failure a retry cannot fix (here: the gateway cannot work out the typed name) is logged and the
    // completed setup proceeds; the owner picks the language in Settings → Appearance, which says why.
    const staged = { ...SUMMARY_FIRST, initialState: { ...SUMMARY_FIRST.initialState, languageChoice: 'Quenya' } };
    const server = completionServer({ language: [
        [400, { ok: false, saved: false, code: 'language_needs_model', error: 'no credentialed model' }],
    ] });
    const warn = console.warn;
    const warned = [];
    console.warn = (...args) => { warned.push(String(args[0])); };
    try {
        await withWizard(staged, 'save=timeout-language-refused', async ({ doc }) => {
            doc.getElementById('next-btn').fire('click');
            await settle();
            server.attempts[server.sent[0]] = { saved: true, receipt: receipt('advanced', false) };
            doc.getElementById('check-save-btn').fire('click');
            await settle();
            assert.deepEqual(server.requests, ['POST /api/onboarding/complete', server.readBack(0), 'POST /api/ui/i18n/language']);
            assert.deepEqual(server.replaced, ['/']);
            assert.ok(warned.some((line) => line.includes('language not applied yet')));
        }, { fetch: server.fetch, location: server.location });
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

// Owner 4A: a fresh local-only setup proposes no unreachable remote Light or Fallbacks — blank
// Light follows the local Main and blank Fallbacks is no chain (the server's own proposal). An
// owner's edit is saved as typed; with a router key beside the local runtime, the shipped
// router Light and Fallbacks are reachable and stay the proposal.
test('a fresh local-only setup saves Light and Fallbacks that follow the local Main', async () => {
    const local = { 'local-source': '/models/local-test.gguf' };
    const routerDefaults = BOOTSTRAP.modelDefaults.openrouter;
    const body = await finishFromAccounts(FRESH, 'local-only-slots', { inputs: local, localMode: 'all' });
    assert.deepEqual([body.LOCAL_ROUTING_MODE, body.OUROBOROS_MODEL, body.OUROBOROS_MODEL_LIGHT, body.OUROBOROS_MODEL_FALLBACKS],
        ['all', routerDefaults.main, '', '']);
    const edited = await finishFromAccounts(FRESH, 'local-only-edited', { inputs: local, localMode: 'all',
        models: { 'light-model': 'owner-light' } });
    assert.deepEqual([edited.OUROBOROS_MODEL_LIGHT, edited.OUROBOROS_MODEL_FALLBACKS], ['owner-light', '']);
    const initialState = { ...BOOTSTRAP.initialState, localSource: '/models/local-test.gguf', localRoutingMode: 'all',
        mainModel: 'owner-main', lightModel: 'owner-light', fallbackModel: 'owner-fallback' };
    const reopened = await finishFromAccounts({ ...BOOTSTRAP, freshInstall: false, initialState }, 'local-only-saved');
    assert.deepEqual([reopened.OUROBOROS_MODEL, reopened.OUROBOROS_MODEL_LIGHT, reopened.OUROBOROS_MODEL_FALLBACKS],
        ['owner-main', 'owner-light', 'owner-fallback']);
    const mixed = await finishFromAccounts(FRESH, 'local-fallback-router', { localMode: 'fallback',
        inputs: { ...local, 'openrouter-key': 'sk-or-v1-test-not-a-secret' } });
    assert.deepEqual([mixed.OUROBOROS_MODEL_LIGHT, mixed.OUROBOROS_MODEL_FALLBACKS], [routerDefaults.light, routerDefaults.fallback]);
});
