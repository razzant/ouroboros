// The Skills page reads the full installed list once per owner action, never to
// paint a spinner, and its sub-tab searches work on what is already loaded. These
// drive the production controllers against their network boundaries, the way
// skills_read_state.test.js does, and count every list read.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { setImmediate as nextTurn } from 'node:timers/promises';

import { renderInstalledSkillCard, renderSkillHubBadges } from '../modules/skill_card_renderer.js';
import { hubFactsPending, hubListingRowFor, hubSyncVerdict } from '../modules/hub_sync.js';
import { escapeHtmlAttr, renderHubCard, renderSubmissionHistory, reviewTone } from '../modules/utils.js';

function source(file, from, until) {
    const text = readFileSync(new URL(`../modules/${file}.js`, import.meta.url), 'utf8');
    const start = text.indexOf(from);
    const end = until ? text.indexOf(until, start) : text.length;
    assert.ok(start >= 0 && end > start, `${file}: source boundaries`);
    return text.slice(start, end).replace(/^export /gm, '');
}

function node() {
    return {
        innerHTML: '', textContent: '', hidden: true, isConnected: true,
        className: '', dataset: {}, handlers: {},
        classList: { add() {}, remove() {} },
        addEventListener(name, handler) { this.handlers[name] = handler; },
        removeEventListener(name, handler) { if (this.handlers[name] === handler) delete this.handlers[name]; },
        querySelectorAll: () => [],
        setAttribute() {},
    };
}

function deferred() {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

const HASH = 'a'.repeat(64);
const demo = {
    name: 'demo', source: 'external', payload_root: 'skills/external/demo', content_hash: HASH,
    version: '1.0.0', review_status: 'pending', review_gate: { executable_review: false },
    permissions: [], grants: { all_granted: true },
    review_findings: [
        { item: 'exec', verdict: 'warn', reason: 'Spawns a shell' },
        { item: 'skill_preflight', verdict: 'fail', reason: JSON.stringify({ files: [{ path: 'main.py', ok: false, stderr: 'SyntaxError: bad token\nmore' }] }) },
    ],
};

// The production action handlers with every network boundary counted.
function actionHandlers({ post = async () => ({ ok: true, status: 'clean' }), skills = [demo] } = {}) {
    const container = node();
    const counts = { listReads: 0, renders: 0, repaints: 0, posts: [] };
    const reviewing = new Set(), repairing = new Set();
    const context = vm.createContext({
        fetchSkills: async () => { counts.listReads += 1; return { skills }; },
        openConfirmDialog: async () => true,
        postWithFeedback: async (url, body) => { counts.posts.push(url); return post(url, body); },
        apiClient: {}, emitSkillLifecycle() {}, showToast() {}, reviewTone,
        buildHealPrompt: () => 'repair prompt',
    });
    vm.runInContext(source('skills', 'function attachActionHandlers(', '\nfunction activateTab'), context);
    context.attachActionHandlers(container, () => { counts.renders += 1; }, reviewing, repairing, { showPage() {} }, () => { counts.repaints += 1; });
    const click = (button, selector) => container.handlers.click({ target: {
        closest: wanted => (wanted === selector ? button : null),
    } });
    return { counts, reviewing, repairing, click, container, context };
}

const primary = (action, skill = 'demo') => ({ dataset: { skill, skillAction: action, keys: '' }, classList: { contains: () => false } });
const menuItem = (cls, skill = 'demo') => ({ dataset: { skill }, classList: { contains: name => name === cls } });

test('Review reads the list once after the action, never before it and never to paint the spinner', async () => {
    for (const [label, button, selector] of [
        ['primary Review', primary('review'), '[data-skill-action]'],
        ['primary Re-review', primary('rereview'), '[data-skill-action]'],
        ['menu Review', menuItem('skills-review'), 'button[data-skill]'],
    ]) {
        const view = actionHandlers();
        await view.click(button, selector);
        assert.equal(view.counts.posts.length, 1, label);
        assert.equal(view.counts.listReads, 0, `${label}: the server re-checks existence; no pre-read`);
        assert.equal(view.counts.repaints, 1, `${label}: the spinner is a local repaint`);
        assert.equal(view.counts.renders, 1, `${label}: one list read after the action`);
        assert.equal(view.reviewing.size, 0);
    }
    // A failed review POST still ends with the one render and a cleared spinner.
    const failing = actionHandlers({ post: async () => { throw new Error('HTTP 500'); } });
    await failing.click(primary('review'), '[data-skill-action]');
    assert.equal(failing.counts.renders, 1);
    assert.equal(failing.counts.repaints, 1);
    assert.equal(failing.reviewing.size, 0);
});

test('Repair and Skip review keep their row pre-read (prompt findings, content hash) and read the list once after', async () => {
    const repair = actionHandlers();
    await repair.click(primary('repair'), '[data-skill-action]');
    assert.deepEqual(repair.counts.posts, ['/api/command']);
    assert.equal(repair.counts.listReads, 1, 'the repair prompt is built from the row');
    assert.equal(repair.counts.repaints, 1);
    assert.equal(repair.counts.renders, 1);
    assert.equal(repair.repairing.size, 0);

    const attest = actionHandlers();
    await attest.click(menuItem('skills-attest-review'), 'button[data-skill]');
    assert.deepEqual(attest.counts.posts, ['/api/owner/skills/demo/attest-review']);
    assert.equal(attest.counts.listReads, 1, 'attestation sends the row content hash');
    assert.equal(attest.counts.repaints, 1);
    assert.equal(attest.counts.renders, 1);
    assert.equal(attest.reviewing.size, 0);
});

test('overlapping actions repaint locally while their requests run and read the list once each afterwards', async () => {
    const held = [];
    const view = actionHandlers({
        post: () => { const gate = deferred(); held.push(gate); return gate.promise; },
        skills: [demo, { ...demo, name: 'other' }],
    });
    const review = view.click(primary('review'), '[data-skill-action]');
    await nextTurn();
    const attest = view.click(menuItem('skills-attest-review', 'other'), 'button[data-skill]');
    await nextTurn();
    assert.equal(held.length, 2);
    // The same skill is one action at a time: a second Review/Skip on it is refused, not queued.
    await view.click(menuItem('skills-review'), 'button[data-skill]');
    assert.equal(held.length, 2);
    assert.equal(view.counts.repaints, 2, 'both spinners painted from memory');
    assert.equal(view.counts.renders, 0, 'no list read while the actions run');
    assert.equal(view.counts.listReads, 1, 'only the attestation pre-read');
    held[1].resolve({ ok: true });
    await attest;
    assert.equal(view.counts.renders, 1);
    held[0].resolve({ ok: true, status: 'clean' });
    await review;
    assert.equal(view.counts.renders, 2);
    assert.equal(view.counts.listReads, 1);
});

// The production list reader plus the local repaint, against a fake list node.
function skillsReader(overrides = {}) {
    const status = node(), empty = node(), list = node();
    let html = '', cards = [];
    const patches = [];
    const cardFor = value => ({ ...node(), dataset: { skill: /data-skill="([^"]+)"/.exec(value)[1] } });
    list.querySelectorAll = selector => selector.includes('.skills-card') ? cards : [];
    list.insertBefore = card => { cards.unshift(card); };
    Object.defineProperty(list, 'firstElementChild', { get: () => cards[0] || null });
    Object.defineProperty(list, 'innerHTML', {
        get: () => html,
        set: value => { html = value; cards = [...value.matchAll(/<article[^>]+>/g)].map(row => cardFor(row[0])); },
    });
    const apiClient = {
        state: async () => ({ github_token_configured: true }),
        extensions: async () => ({ skills: [demo], live: {} }),
        skillLifecycleQueue: async () => ({ events: [] }),
        ...overrides,
    };
    const context = vm.createContext({
        apiClient, renderInstalledSkillCard, renderSkillHubBadges,
        patchInstalledSkillEnrichment: (card, skill, reviewing) => { patches.push({ skill: skill.name, reviewing: [...reviewing] }); },
        LIFECYCLE_VISIBLE_STATUSES: new Set(['queued', 'running', 'failed']),
        document: {
            getElementById: id => id === 'skills-status' ? status : null,
            createElement: () => ({ content: {}, set innerHTML(value) { this.content.firstElementChild = cardFor(value); } }),
        },
        hubCatalog: { byName: new Map(), available: false }, hubFactsPending,
        loadHubCatalog: () => Promise.resolve(), sortSkillsForDisplay: rows => rows,
    });
    vm.runInContext(`let skillsRenderGeneration = 0; let skillsSnapshot = null;
        ${source('skills', 'async function fetchSkills(', '\nfunction updateQueueBadges')}
        function updateQueueBadges() {}
        ${source('skills', 'async function renderSkillsList(', '\nasync function postWithFeedback')}`, context);
    return { context, apiClient, status, list, patches,
        render: () => context.renderSkillsList(list, empty, new Set(), new Set(), {}),
        repaint: reviewing => context.repaintSkillsList(list, reviewing, new Set(), {}) };
}

test('the local repaint patches cards from the list in memory and leaves an in-flight read untouched', async () => {
    const view = skillsReader();
    assert.equal(view.repaint(new Set(['demo'])), undefined, 'nothing to repaint before the first read');
    assert.equal(view.patches.length, 0);
    await view.render();
    const held = deferred();
    view.apiClient.extensions = () => held.promise;
    const rendering = view.render();
    await nextTurn();
    view.repaint(new Set(['demo']));
    assert.deepEqual(view.patches, [{ skill: 'demo', reviewing: ['demo'] }], 'the card is patched, not re-read');
    held.resolve({ skills: [{ ...demo, name: 'current' }], live: {} });
    await rendering;
    assert.match(view.list.innerHTML, /data-skill="current"/, 'the held read still paints as the newest answer');
    assert.equal(view.patches.length, 1);
});

// The production Hub tab controller against stubbed HTTP answers.
function hubController({ catalog, listing, immediateTimers = false }) {
    const nodes = Object.fromEntries(['#oh-query', '#oh-results', '#oh-status', '[data-oh-search]'].map(selector => [selector, node()]));
    const pane = { ...node(), querySelector: selector => nodes[selector] };
    const reads = [];
    const context = vm.createContext({
        setTimeout: immediateTimers ? (callback) => { callback(); return 0; } : setTimeout, clearTimeout,
        hubFactsPending, hubListingRowFor, hubSyncVerdict,
        escapeHtml: escapeHtmlAttr, renderHubCard, renderSubmissionHistory,
        template: () => '', getPending: () => undefined, setPending() {}, clearPending() {},
        startLifecyclePoller: () => () => {}, emitSkillLifecycle() {}, openConfirmDialog: async () => false,
        fetchJson: async (path) => {
            reads.push(path);
            if (path.startsWith('/api/extensions')) return { skills: listing };
            return { results: typeof catalog === 'function' ? catalog() : catalog };
        },
    });
    vm.runInContext(source('ouroboroshub', 'const HUB_REPLACEMENT_KEEPS', '\nfunction controlsTemplate')
        + source('ouroboroshub', 'export function initOuroborosHub('), context);
    return { context, pane, nodes, reads, type: query => nodes['#oh-query'].handlers.input({ target: { value: query } }),
        html: () => nodes['#oh-results'].innerHTML, status: () => nodes['#oh-status'].textContent };
}

test('typing in the OuroborosHub search filters the loaded catalog locally; Search re-reads it', async () => {
    const hub = hubController({
        catalog: [
            { slug: 'weather', sanitized_name: 'weather', display_name: 'Weather', description: 'Local forecast' },
            { slug: 'notes', sanitized_name: 'notes', display_name: 'Notes', description: 'Keep notes' },
        ],
        listing: [{ name: 'weather', source: 'ouroboroshub', location: 'ouroboroshub', version: '1.0', content_hash: HASH }],
        immediateTimers: true,
    });
    await hub.context.initOuroborosHub(hub.pane);
    const initial = hub.reads.length;
    assert.match(hub.html(), /data-slug="weather"[\s\S]*data-slug="notes"/);
    hub.type('fore');
    assert.equal(hub.reads.length, initial, 'typing reads nothing');
    assert.match(hub.html(), /data-slug="weather"/);
    assert.doesNotMatch(hub.html(), /data-slug="notes"/);
    assert.equal(hub.status(), '1 official skill');
    hub.type('');
    assert.equal(hub.reads.length, initial);
    assert.match(hub.html(), /data-slug="notes"/);
    assert.equal(hub.status(), '2 official skills');
    hub.nodes['[data-oh-search]'].handlers.click();
    await nextTurn(); await nextTurn();
    assert.ok(hub.reads.length > initial, 'Search is the network refresh');
});

test('filtering after a failed catalog refresh keeps the unavailable notice; a fresh catalog shows plain counts', async () => {
    let failing = false;
    const rows = [
        { slug: 'weather', sanitized_name: 'weather', display_name: 'Weather', description: 'Local forecast' },
        { slug: 'notes', sanitized_name: 'notes', display_name: 'Notes', description: 'Keep notes' },
    ];
    const hub = hubController({
        catalog: () => { if (failing) throw new Error('hub offline'); return rows; },
        listing: [], immediateTimers: true,
    });
    await hub.context.initOuroborosHub(hub.pane);
    hub.type('fore');
    assert.equal(hub.status(), '1 official skill', 'a fresh catalog reports plain counts');
    failing = true;
    hub.nodes['[data-oh-search]'].handlers.click();
    await nextTurn(); await nextTurn();
    assert.match(hub.status(), /^Hub catalog unavailable: hub offline\. Showing previous results\. Refresh to retry\.$/);
    hub.type('');
    assert.match(hub.html(), /data-slug="notes"/, 'the previous rows are still filtered locally');
    assert.equal(hub.status(), 'Hub catalog unavailable. Showing previous results (2 official skills). Refresh to retry.');
});

test('the ClawHub installed read outlives the 3 s bound of its optional enrichment', async () => {
    const context = vm.createContext({ AbortController, setTimeout: (callback) => { callback(); return 0; }, clearTimeout, console: { warn() {} } });
    vm.runInContext(source('marketplace', 'async function loadInstalled(', '\nasync function runSearch'), context);
    const seen = [];
    context.fetchJson = async (path, init) => {
        seen.push([path, Boolean(init?.signal?.aborted)]);
        if (init?.signal?.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
        return { skills: [{ ...demo, provenance: { slug: 'demo' } }] };
    };
    const result = await context.loadInstalled();
    assert.equal(result.available, true, 'the primary read is not cancelled by the enrichment bound');
    assert.equal(result.enrichmentAvailable, false);
    assert.equal(result.map.get('demo').name, 'demo');
    assert.deepEqual(seen, [['/api/marketplace/clawhub/installed', false], ['/api/extensions', true]]);
    // A newer refresh still cancels both.
    const external = new AbortController();
    external.abort();
    const cancelled = await context.loadInstalled({ signal: external.signal });
    assert.equal(cancelled.available, false);
});

test('typing in the ClawHub search re-reads the registry, not the installed list it already knows', async () => {
    const nodes = Object.fromEntries(['#mp-query', '#mp-only-official', '[data-mp-search]', '#mp-results', '#mp-pagination', '#mp-status']
        .map(selector => [selector, node()]));
    const pane = { ...node(), querySelector: selector => nodes[selector] };
    const counts = { search: 0, installed: 0 };
    let installedAvailable = true;
    const pending = [];
    const context = vm.createContext({
        AbortController, clearTimeout, URLSearchParams,
        setTimeout: (callback) => { pending.push(callback()); return 0; },
        paneTemplate: () => '', getPendingBySlug: () => new Map(), getPending: () => undefined, setPending() {},
        document: { getElementById: id => nodes[`#${id}`] },
        startLifecyclePoller: () => () => {},
        runSearch: async () => { counts.search += 1; return { results: [{ slug: 'demo' }] }; },
        loadInstalled: async () => { counts.installed += 1; return { available: installedAvailable, map: new Map(), enrichmentAvailable: true }; },
        renderResults() {}, renderPagination() {}, isRateLimitError: () => false,
    });
    vm.runInContext(source('marketplace', 'function installErrorCopy(', '\nconst safeExternalUrl')
        + source('marketplace', 'function showStatus(', '\nasync function loadInstalled')
        + source('marketplace', 'export function initMarketplace('), context);
    await context.initMarketplace(pane);
    assert.deepEqual(counts, { search: 1, installed: 1 });
    const type = async (value) => {
        nodes['#mp-query'].handlers.input({ target: { value } });
        await Promise.all(pending.splice(0));
    };
    await type('de');
    await type('demo');
    assert.deepEqual(counts, { search: 3, installed: 1 }, 'keystrokes search without re-reading installed state');
    nodes['[data-mp-search]'].handlers.click();
    await Promise.all(pending.splice(0));
    assert.deepEqual(counts, { search: 4, installed: 2 }, 'the Search button refreshes both');
    installedAvailable = false;
    await pane._marketplaceRefresh();
    assert.deepEqual(counts, { search: 5, installed: 3 });
    await type('demo!');
    assert.deepEqual(counts, { search: 6, installed: 4 }, 'an unavailable installed state is read again with the search');
    assert.match(nodes['#mp-status'].textContent, /could not be read/);
});

test('a post-action installed read that a ClawHub keystroke supersedes is still made', async () => {
    const nodes = Object.fromEntries(['#mp-query', '#mp-only-official', '[data-mp-search]', '#mp-results', '#mp-pagination', '#mp-status']
        .map(selector => [selector, node()]));
    const pane = { ...node(), querySelector: selector => nodes[selector] };
    const counts = { search: 0, installed: 0 };
    const timers = new Map();
    let nextTimer = 0;
    const flush = async () => { const due = [...timers.values()]; timers.clear(); await Promise.all(due.map(callback => callback())); };
    const context = vm.createContext({
        AbortController, URLSearchParams,
        setTimeout: (callback) => { timers.set(++nextTimer, callback); return nextTimer; },
        clearTimeout: (id) => { timers.delete(id); },
        paneTemplate: () => '', getPendingBySlug: () => new Map(), getPending: () => undefined, setPending() {},
        document: { getElementById: id => nodes[`#${id}`] },
        startLifecyclePoller: () => () => {},
        runSearch: async () => { counts.search += 1; return { results: [{ slug: 'demo' }] }; },
        loadInstalled: async () => { counts.installed += 1; return { available: true, map: new Map(), enrichmentAvailable: true }; },
        renderResults() {}, renderPagination() {}, isRateLimitError: () => false,
    });
    vm.runInContext(source('marketplace', 'function installErrorCopy(', '\nconst safeExternalUrl')
        + source('marketplace', 'function showStatus(', '\nasync function loadInstalled')
        + source('marketplace', 'export function initMarketplace('), context);
    await context.initMarketplace(pane);
    assert.deepEqual(counts, { search: 1, installed: 1 });
    // Search (like a confirmed action) schedules a refresh that reads the installed list;
    // a keystroke replaces that timer before it fires.
    nodes['[data-mp-search]'].handlers.click();
    nodes['#mp-query'].handlers.input({ target: { value: 'de' } });
    await flush();
    assert.deepEqual(counts, { search: 2, installed: 2 }, 'the owed installed read rides the keystroke refresh');
    nodes['#mp-query'].handlers.input({ target: { value: 'demo' } });
    await flush();
    assert.deepEqual(counts, { search: 3, installed: 2 }, 'once read, keystrokes search only again');
});

test('an older ClawHub refresh landing after an action cannot settle the read the action owes', async () => {
    const nodes = Object.fromEntries(['#mp-query', '#mp-only-official', '[data-mp-search]', '#mp-results', '#mp-pagination', '#mp-status']
        .map(selector => [selector, node()]));
    const pane = { ...node(), querySelector: selector => nodes[selector] };
    const counts = { search: 0, installed: 0 };
    const timers = new Map();
    let nextTimer = 0;
    let heldSearch = null;
    const flush = async () => { const due = [...timers.values()]; timers.clear(); await Promise.all(due.map(callback => callback())); };
    const context = vm.createContext({
        AbortController, URLSearchParams,
        setTimeout: (callback) => { timers.set(++nextTimer, callback); return nextTimer; },
        clearTimeout: (id) => { timers.delete(id); },
        paneTemplate: () => '', getPendingBySlug: () => new Map(), getPending: () => undefined, setPending() {},
        document: { getElementById: id => nodes[`#${id}`] },
        startLifecyclePoller: () => () => {},
        runSearch: async () => { counts.search += 1; if (heldSearch) await heldSearch.promise; return { results: [{ slug: 'demo' }] }; },
        loadInstalled: async () => { counts.installed += 1; return { available: true, map: new Map(), enrichmentAvailable: true }; },
        renderResults() {}, renderPagination() {}, isRateLimitError: () => false,
    });
    vm.runInContext(source('marketplace', 'function installErrorCopy(', '\nconst safeExternalUrl')
        + source('marketplace', 'function showStatus(', '\nasync function loadInstalled')
        + source('marketplace', 'export function initMarketplace('), context);
    await context.initMarketplace(pane);
    heldSearch = deferred();
    const older = pane._marketplaceRefresh();      // refresh A reads the pre-action installed list, its search is held
    await nextTurn();
    assert.deepEqual(counts, { search: 2, installed: 2 });
    nodes['[data-mp-search]'].handlers.click();     // an action-like refresh owes a newer installed read ...
    nodes['#mp-query'].handlers.input({ target: { value: 'de' } });  // ... and a keystroke replaces its timer
    heldSearch.resolve(); heldSearch = null;
    await older;                                    // A lands with a matching token
    await flush();
    assert.deepEqual(counts, { search: 3, installed: 3 }, 'the keystroke refresh still makes the owed read');
});
