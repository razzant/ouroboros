import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { setImmediate as nextTurn } from 'node:timers/promises';

import { renderInstalledSkillCard, renderSkillHubBadges } from '../modules/skill_card_renderer.js';
import { lifecycleFor } from '../modules/marketplace.js';
import { lifecycleCardClassFor, lifecycleSpinnerFor } from '../modules/lifecycle_card.js';
import { hubFactsPending, hubListingRowFor, hubSyncVerdict } from '../modules/hub_sync.js';
import { hubVersionText } from '../modules/ouroboroshub.js';
import { escapeHtmlAttr, isRateLimitError, renderHubCard, renderSubmissionHistory } from '../modules/utils.js';

// Run the production controllers against their network/DOM boundaries. No
// copied renderer, browser globals or process-wide fetch mutation is needed.
function source(file, from, until) {
    const text = readFileSync(new URL(`../modules/${file}.js`, import.meta.url), 'utf8');
    const start = text.indexOf(from);
    const end = until ? text.indexOf(until, start) : text.length;
    assert.ok(start >= 0 && end > start, `${file}: source boundaries`);
    return text.slice(start, end).replace(/^export /, '');
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

const demo = {
    name: 'demo', source: 'external', payload_root: 'skills/external/demo',
    version: '1.0.0', review_status: 'clean', review_gate: { executable_review: true },
    permissions: [], grants: { all_granted: true },
};

function skillsReader(overrides = {}, contextOverrides = {}) {
    const status = node(), empty = node(), list = node(), badge = node();
    badge.dataset.skillHubBadges = 'demo';
    let writes = 0, html = '', cards = [];
    const patches = [];
    const cardFor = value => ({ ...node(), dataset: { skill: /data-skill="([^"]+)"/.exec(value)[1] } });
    list.querySelectorAll = selector => selector.includes('.skills-card') ? cards : [badge];
    list.insertBefore = card => { cards.unshift(card); };
    Object.defineProperty(list, 'firstElementChild', { get: () => cards[0] || null });
    Object.defineProperty(list, 'innerHTML', {
        get: () => html,
        set: value => { writes += 1; html = value; cards = [...value.matchAll(/<article[^>]+>/g)].map(row => cardFor(row[0])); },
    });
    const apiClient = {
        state: async () => ({ github_token_configured: true }),
        extensions: async () => ({ skills: [demo], live: {} }),
        skillLifecycleQueue: async () => ({ events: [] }),
        ...overrides,
    };
    const context = vm.createContext({
        apiClient, renderInstalledSkillCard, renderSkillHubBadges,
        patchInstalledSkillEnrichment: (card, skill) => { patches.push({ card, skill }); },
        LIFECYCLE_VISIBLE_STATUSES: new Set(['queued', 'running', 'failed']),
        document: {
            getElementById: id => id === 'skills-status' ? status : null,
            createElement: () => ({ content: {}, set innerHTML(value) { this.content.firstElementChild = cardFor(value); } }),
        },
        hubCatalog: { byName: new Map(), available: false }, hubFactsPending,
        loadHubCatalog: () => Promise.resolve(), sortSkillsForDisplay: rows => rows,
        ...contextOverrides,
    });
    vm.runInContext(`let skillsRenderGeneration = 0; let skillsSnapshot = null;
        ${source('skills', 'async function fetchSkills(', '\nfunction updateQueueBadges')}
        function updateQueueBadges() {}
        ${source('skills', 'async function renderSkillsList(', '\nasync function postWithFeedback')}`, context);
    return { context, apiClient, status, empty, list, badge, patches, writes: () => writes,
        render: interactions => context.renderSkillsList(list, empty, new Set(), new Set(), interactions) };
}

test('the list never waits for the hub: pending hub facts arrive with one re-read after the catalog', async () => {
    const hub = facts => ({ ...demo, name: 'hub', source: 'ouroboroshub', payload_root: 'skills/ouroboroshub/hub', ...facts });
    const catalog = deferred();
    const responses = [hub({ official_hub_verified: null, owner_attestable: null }),
        hub({ official_hub_verified: true, owner_attestable: true })];
    let reads = 0;
    const hubCatalog = { byName: new Map(), available: false };
    const view = skillsReader(
        { extensions: async () => ({ skills: [responses[Math.min(reads++, 1)]], live: {} }) },
        { hubCatalog, loadHubCatalog: () => catalog.promise },
    );
    const rendered = view.render();
    await rendered;
    assert.equal(reads, 1, 'first paint comes from the local listing alone');
    assert.equal(view.writes(), 1);
    hubCatalog.available = true;
    catalog.resolve();
    await nextTurn();
    assert.equal(reads, 2, 'one re-read once the catalog read has landed');
    assert.equal(view.writes(), 1, 'cards are patched in place, never recreated');
    assert.equal(view.patches.at(-1).skill.official_hub_verified, true);

    // No pending fact, or no catalog: nothing to re-read.
    for (const [facts, available] of [[{ official_hub_verified: false }, true], [{ official_hub_verified: null }, false]]) {
        let count = 0;
        const quiet = skillsReader(
            { extensions: async () => { count += 1; return { skills: [hub(facts)], live: {} }; } },
            { hubCatalog: { byName: new Map(), available }, loadHubCatalog: () => Promise.resolve() },
        );
        await quiet.render();
        await nextTurn();
        assert.equal(count, 1);
    }
});

test('a render that reused an older catalog snapshot refreshes it before its one re-read', async () => {
    // Any in-page remount after the server's 120 s display memo expired paints
    // hub facts as null while the page's own catalog promise is long resolved.
    // A second listing read on the same unknown answer would change nothing; the
    // render forces exactly one catalog read first, then re-reads once.
    const hub = facts => ({ ...demo, name: 'hub', source: 'ouroboroshub', payload_root: 'skills/ouroboroshub/hub', ...facts });
    const responses = [hub({ official_hub_verified: null, owner_attestable: null }),
        hub({ official_hub_verified: true, owner_attestable: true })];
    let reads = 0;
    const forced = [];
    const hubCatalog = { byName: new Map(), available: true, settled: true, promise: Promise.resolve() };
    const view = skillsReader(
        { extensions: async () => ({ skills: [responses[Math.min(reads++, 1)]], live: {} }) },
        { hubCatalog, loadHubCatalog: force => { forced.push(Boolean(force)); return hubCatalog.promise; } },
    );
    await view.render();
    await nextTurn();
    await nextTurn();
    assert.deepEqual(forced, [false, true], 'the reused snapshot is refreshed exactly once');
    assert.equal(reads, 2, 'then one listing re-read');
    assert.equal(view.patches.at(-1).skill.official_hub_verified, true);

    // The refreshed catalog is unavailable: unknown stays unknown, no blind re-read.
    let count = 0;
    const offline = { byName: new Map(), available: true, settled: true, promise: Promise.resolve() };
    const quiet = skillsReader(
        { extensions: async () => { count += 1; return { skills: [hub({ official_hub_verified: null })], live: {} }; } },
        { hubCatalog: offline, loadHubCatalog: force => { if (force) offline.available = false; return offline.promise; } },
    );
    await quiet.render();
    await nextTurn();
    await nextTurn();
    assert.equal(count, 1);
});

test('primary failure preserves previous cards and is never a successful empty list', async () => {
    const view = skillsReader({ extensions: async () => { throw new Error('HTTP 503'); } });
    await assert.rejects(view.context.fetchSkills(), /HTTP 503/);
    await view.render();
    assert.equal(view.empty.hidden, true);
    assert.equal(view.list.innerHTML, '');
    assert.match(view.status.textContent, /Could not load installed skills.*503/);

    view.apiClient.extensions = async () => ({ skills: [demo], live: {} });
    await view.render();
    const previous = view.list.innerHTML;
    view.apiClient.extensions = async () => { throw new Error('offline'); };
    await view.render();
    assert.equal(view.list.innerHTML, previous);
    assert.match(view.status.textContent, /Showing the previous list/);

    view.apiClient.extensions = async () => ({ skills: [], live: {} });
    await view.render();
    assert.equal(view.list.innerHTML, '');
    assert.equal(view.empty.hidden, false);
    assert.equal(view.status.textContent, '');
    view.apiClient.extensions = async () => { throw new Error('offline again'); };
    await view.render();
    assert.equal(view.empty.hidden, true, 'an old empty response is not the current availability claim');
});

test('malformed primary data fails, optional state/queue failures retain known facts', async () => {
    const view = skillsReader({ skillLifecycleQueue: async () => ({
        active: { id: 'disable-1', target: 'demo', kind: 'disable', status: 'running' }, events: [],
    }) });
    await view.render();
    view.apiClient.state = async () => { throw new Error('settings offline'); };
    view.apiClient.skillLifecycleQueue = async () => { throw new Error('queue offline'); };
    await view.render();
    assert.match(view.list.innerHTML, /Disabling/);
    assert.match(view.list.innerHTML, /data-submit-disabled="false"/);
    assert.match(view.status.textContent, /Skill settings could not be refreshed/);
    assert.match(view.status.textContent, /Lifecycle progress could not be refreshed/);
    view.apiClient.extensions = async () => ({ error: 'invalid response' });
    await assert.rejects(view.context.fetchSkills(), /Installed skills response is unavailable/);
});

test('state, queue and both pending never delay the primary render or its completion', async () => {
    for (const held of [['state'], ['queue'], ['state', 'queue']]) {
        const state = deferred(), queue = deferred();
        const view = skillsReader({
            state: () => held.includes('state') ? state.promise : Promise.resolve({ github_token_configured: true }),
            skillLifecycleQueue: () => held.includes('queue') ? queue.promise : Promise.resolve({ events: [] }),
        });
        let completed = false;
        const rendering = view.render().then(() => { completed = true; });
        await nextTurn();
        try {
            assert.equal(completed, true, `${held}: primary render must complete independently`);
            assert.match(view.list.innerHTML, /data-skill="demo"/);
            assert.match(view.status.textContent, /loading/);
            assert.doesNotMatch(view.list.innerHTML, /Configure GITHUB_TOKEN/, 'pending is not known absence');
        } finally {
            state.resolve({ github_token_configured: true }); queue.resolve({ events: [] });
            await rendering; await nextTurn();
        }
        assert.equal(view.status.textContent, '');
        assert.equal(view.writes(), 1, 'late facts never repaint the whole list');
    }
});

test('late queue settlement merges raw primary rows, clearing prior running annotations', async () => {
    const queued = { events: [{ id: 'disable', target: 'demo', kind: 'disable', status: 'running' }] };
    const view = skillsReader({ skillLifecycleQueue: async () => queued });
    await view.render();
    const queue = deferred();
    view.apiClient.skillLifecycleQueue = () => queue.promise;
    await view.render();
    assert.match(view.list.innerHTML, /Disabling/);
    queue.resolve({ events: [{ ...queued.events[0], status: 'done' }] });
    await nextTurn();
    const updated = view.patches.at(-1).skill;
    assert.equal(updated.name, 'demo');
    assert.equal(Object.hasOwn(updated, 'lifecycle_pending'), false);
    assert.equal(Object.hasOwn(demo, 'lifecycle_pending'), false, 'raw extensions stay unannotated');
    assert.equal(view.writes(), 2);
});

test('late optional failures are named and older generation responses are ignored', async () => {
    const state = deferred(), queue = deferred();
    const view = skillsReader({ state: () => state.promise, skillLifecycleQueue: () => queue.promise });
    await view.render();
    state.reject(new Error('settings offline')); queue.reject(new Error('queue offline'));
    await nextTurn();
    assert.match(view.status.textContent, /Skill settings could not be refreshed/);
    assert.match(view.status.textContent, /Lifecycle progress could not be refreshed/);
    assert.equal(view.writes(), 1);
    const oldState = deferred(), oldQueue = deferred();
    view.apiClient.state = () => oldState.promise; view.apiClient.skillLifecycleQueue = () => oldQueue.promise;
    await view.render();
    view.apiClient.state = async () => ({ github_token_configured: true });
    view.apiClient.skillLifecycleQueue = async () => ({ events: [] });
    await view.render();
    const count = view.patches.length;
    oldState.resolve({ github_token_configured: false }); oldQueue.reject(new Error('old failed read'));
    await nextTurn();
    assert.equal(view.patches.length, count);
    assert.equal(view.status.textContent, '');
});

test('primary replacement closes menus after the read, whereas a failed read leaves them attached', async () => {
    const view = skillsReader();
    await view.render();
    for (const failed of [false, true]) {
        const primary = deferred();
        view.apiClient.extensions = () => primary.promise;
        let closes = 0;
        const rendering = view.render({ beforeReplace: () => { closes += 1; } });
        await nextTurn();
        assert.equal(closes, 0, 'a menu opened during this request still has its owner');
        if (failed) primary.reject(new Error('offline'));
        else primary.resolve({ skills: [demo], live: {} });
        await rendering;
        assert.equal(closes, failed ? 0 : 1);
    }
});

test('legacy publish keeps unknown, confirmed absent and present token facts distinct', () => {
    for (const value of [undefined, false, true]) {
        const html = renderInstalledSkillCard(demo, new Set(), new Set(), {}, { githubTokenConfigured: value });
        assert.equal(html.includes('Configure GITHUB_TOKEN'), value === false);
        assert.equal(html.includes('GitHub token status unavailable'), value === undefined);
        assert.equal(html.includes('data-submit-disabled="false"'), value === true);
    }
    const html = renderInstalledSkillCard({ ...demo, submit_hub: { visible: true, task_start_allowed: true } });
    assert.match(html, /data-submit-disabled="false"/, 'authoritative backend admission outranks optional state');
});

test('late older success and failure cannot replace the current list or its status', async () => {
    for (const fails of [false, true]) {
        const old = deferred(), fresh = deferred();
        const reads = [old, fresh];
        const view = skillsReader({ extensions: () => reads.shift().promise });
        const first = view.render(), second = view.render();
        fresh.resolve({ skills: [{ ...demo, name: 'current' }], live: {} });
        await second;
        if (fails) old.reject(new Error('old failure'));
        else old.resolve({ skills: [{ ...demo, name: 'obsolete' }], live: {} });
        await first;
        assert.match(view.list.innerHTML, /current/);
        assert.doesNotMatch(view.list.innerHTML, /obsolete/);
        assert.equal(view.status.textContent, '');
    }
});

test('late optional Hub badges update without recreating the skill card', async () => {
    const catalog = deferred();
    const view = skillsReader({ extensions: async () => ({ skills: [{
        ...demo, source: 'ouroboroshub', payload_root: 'skills/ouroboroshub/demo',
    }], live: {} }) });
    view.context.loadHubCatalog = () => catalog.promise;
    await view.render();
    const card = view.list.innerHTML;
    view.context.hubCatalog.available = true;
    view.context.hubCatalog.byName.set('demo', { sanitized_name: 'demo', slug: 'demo', latest_version: '2.0.0' });
    catalog.resolve();
    await nextTurn();
    assert.match(view.badge.innerHTML, /Update available/);
    assert.equal(view.list.innerHTML, card);
    assert.equal(view.writes(), 1, 'only badge contents may change after catalog enrichment');
});

test('one identical lifecycle/load failure renders once; distinct causes remain visible', () => {
    const common = { ...demo, lifecycle_status: 'failed', lifecycle_error: 'same failure', load_error: 'same failure' };
    assert.equal(renderInstalledSkillCard(common).split('same failure').length - 1, 1);
    assert.equal(renderInstalledSkillCard({ ...common, lifecycle_virtual: true, description: 'same failure' })
        .split('same failure').length - 1, 1);
    const html = renderInstalledSkillCard({ ...common, load_error: 'different failure' });
    assert.match(html, /same failure/);
    assert.match(html, /different failure/);
});

test('a primary read failure during owner attestation produces feedback without submitting', async () => {
    const container = node(), messages = [];
    const context = vm.createContext({
        document: { addEventListener() {} }, window: { addEventListener() {} },
        fetchSkills: async () => { throw new Error('inventory offline'); },
        showToast: message => { messages.push(message); },
    });
    vm.runInContext(source('skills', 'function attachActionHandlers(', '\nfunction activateTab'), context);
    context.attachActionHandlers(container, () => {}, new Set(), new Set());
    const button = { dataset: { skill: 'demo' }, classList: { contains: name => name === 'skills-attest-review' } };
    await container.handlers.click({ target: {
        closest: selector => selector === 'button[data-skill]' ? button : null,
    } });
    assert.deepEqual(messages, ['demo: inventory offline']);
});

test('cancelled skill actions retain their card; confirmed effects still refresh it', async () => {
    for (const kind of ['skills-delete-local', 'skills-uninstall', 'skills-submit-hub', 'skills-grant', 'review', 'repair']) {
        const container = node();
        let renders = 0, writes = 0;
        const context = vm.createContext({
            fetchSkills: async () => ({ skills: [{ ...demo, content_hash: 'a'.repeat(64) }] }),
            openConfirmDialog: async () => false,
            runSkillPublishFlow: async () => ({ started: false }),
            apiClient: { deleteSkill: async () => { writes += 1; return { ok: true }; } },
            postWithFeedback: async () => { writes += 1; return { ok: true }; },
            emitSkillLifecycle() {}, showToast() {},
        });
        vm.runInContext(source('skills', 'function attachActionHandlers(', '\nfunction activateTab'), context);
        context.attachActionHandlers(container, () => { renders += 1; }, new Set(), new Set());
        const primary = ['review', 'repair'].includes(kind);
        const button = { dataset: { skill: 'demo', skillAction: kind, keys: 'KEY' },
            classList: { contains: name => name === kind } };
        const event = { target: { closest: selector => (
            primary ? selector === '[data-skill-action]' : selector === 'button[data-skill]'
        ) ? button : null } };
        await container.handlers.click(event);
        assert.equal(writes, 0, `${kind}: cancellation has no side effect`);
        assert.equal(renders, 0, `${kind}: cancellation does not replace the focus return target`);
        assert.equal(button.disabled, false);

        if (kind === 'skills-delete-local' || kind === 'skills-uninstall') {
            context.openConfirmDialog = async () => true;
            await container.handlers.click(event);
            assert.equal(writes, 1, `${kind}: confirmation preserves the existing operation`);
            assert.equal(renders, 1);
        }
    }
});

test('failed lifecycle helpers do not claim computation and unknown installed state offers no Install', () => {
    assert.equal(lifecycleSpinnerFor({ failed: true }), '');
    assert.equal(lifecycleCardClassFor({ failed: true }), 'marketplace-card');
    assert.match(lifecycleSpinnerFor({ failed: false }), /spinner/);
    assert.equal(lifecycleFor({}, null, null).action, 'install', 'a confirmed absent skill is installable');
    const unknown = lifecycleFor({}, null, { failed: true, retry_action: 'install' }, { installedUnavailable: true });
    assert.equal(unknown.action, '');
    assert.equal(unknown.disabled, true);
    assert.doesNotMatch(unknown.label, /Not installed/);
});

test('ClawHub primary and optional installed reads have independent outcomes', async () => {
    const context = vm.createContext({ AbortController, setTimeout, clearTimeout, console: { warn() {} } });
    vm.runInContext(source('marketplace', 'async function loadInstalled(', '\nasync function runSearch'), context);
    context.fetchJson = async path => {
        if (path.endsWith('/installed')) throw new Error('primary failed');
        return { skills: [] };
    };
    const unavailable = await context.loadInstalled();
    assert.equal(unavailable.available, false);
    assert.equal(unavailable.map, null);
    context.fetchJson = async path => {
        if (path.endsWith('/extensions')) throw new Error('optional failed');
        return { skills: [{ ...demo, provenance: { slug: 'demo' } }] };
    };
    const partial = await context.loadInstalled();
    assert.equal(partial.available, true);
    assert.equal(partial.map.get('demo').name, 'demo');
    assert.equal(partial.enrichmentAvailable, false);
});

// The production Hub tab controller (helpers + initOuroborosHub) as one script.
function hubSource() {
    return (source('ouroboroshub', 'const HUB_REPLACEMENT_KEEPS', '\nfunction controlsTemplate')
        + source('ouroboroshub', 'export function initOuroborosHub(')).replace(/^export /gm, '');
}

function catalogPane(kind) {
    const selectors = kind === 'marketplace'
        ? ['#mp-query', '#mp-only-official', '[data-mp-search]', '#mp-results', '#mp-pagination', '#mp-status']
        : ['#oh-query', '#oh-results', '#oh-status', '[data-oh-search]'];
    const nodes = Object.fromEntries(selectors.map(selector => [selector, node()]));
    const pane = { ...node(), querySelector: selector => nodes[selector] };
    return { pane, nodes };
}

test('ClawHub retains catalog results on failure, updates installed availability, and recovers through Refresh', async () => {
    const { pane, nodes } = catalogPane('marketplace');
    let failed = false, filtered = false, onPending;
    const pending = new Map();
    const paints = [];
    const context = vm.createContext({
        AbortController, setTimeout, clearTimeout, URLSearchParams, isRateLimitError,
        paneTemplate: () => '', getPendingBySlug: () => pending,
        getPending: slug => pending.get(slug),
        setPending: (slug, value) => { pending.set(slug, value); onPending(); },
        document: { getElementById: id => nodes[`#${id}`] },
        startLifecyclePoller: callback => { onPending = callback; return () => {}; },
        runSearch: async () => { if (failed) throw new Error('catalog offline'); return { results: filtered ? [] : [{ slug: 'demo' }] }; },
        loadInstalled: async () => ({ available: !failed, map: new Map(), enrichmentAvailable: true }),
        renderResults: (host, rows, installed, count, facts) => {
            host.innerHTML = rows.map(row => row.slug + (pending.get(row.slug)?.message || '')).join(',');
            host.querySelectorAll = () => rows.map(row => ({ dataset: { slug: row.slug } }));
            paints.push({ count, unavailable: facts.installedUnavailable });
        },
        renderPagination() {},
    });
    vm.runInContext(source('marketplace', 'function installErrorCopy(', '\nconst safeExternalUrl')
        + source('marketplace', 'function showStatus(', '\nasync function loadInstalled')
        + source('marketplace', 'export function initMarketplace('), context);
    await context.initMarketplace(pane);
    assert.equal(nodes['#mp-results'].innerHTML, 'demo');
    failed = true;
    await pane._marketplaceRefresh();
    assert.equal(nodes['#mp-results'].innerHTML, 'demo');
    assert.equal(paints.at(-1).unavailable, true);
    assert.match(nodes['#mp-status'].textContent, /catalog offline.*Showing previous results/);
    onPending();
    assert.match(nodes['#mp-status'].textContent, /catalog offline/);
    failed = false;
    await pane._marketplaceRefresh();
    assert.equal(paints.at(-1).unavailable, false);
    assert.doesNotMatch(nodes['#mp-status'].textContent, /offline/);

    // The request may outlive its visible row when the owner searches again.
    for (const rowGone of [true, false]) {
        filtered = false;
        await pane._marketplaceRefresh();
        const install = deferred();
        context.jsonPost = () => install.promise;
        const button = { dataset: { slug: 'demo', mpAction: 'install' } };
        const action = nodes['#mp-results'].handlers.click({ target: {
            closest: selector => selector === '[data-mp-action]' ? button : null,
        } });
        filtered = rowGone;
        await pane._marketplaceRefresh();
        install.reject(new Error('install failed after filtering'));
        await action;
        const status = nodes['#mp-status'].textContent;
        const cards = nodes['#mp-results'].innerHTML;
        assert.equal((status + cards).split('install failed after filtering').length - 1, 1);
        assert.equal(status.includes('install failed after filtering'), rowGone);
    }
});

test('Hub failed listing never becomes Install, and failed catalog keeps useful rows without duplicate errors', async () => {
    const { pane, nodes } = catalogPane('hub');
    let catalogFailed = true, listingFailed = true, filtered = false, onPending, install;
    const pending = new Map();
    nodes['#oh-results'].querySelectorAll = () => nodes['#oh-results'].innerHTML.includes('data-slug="demo"')
        ? [{ dataset: { slug: 'demo' } }] : [];
    const context = vm.createContext({
        URLSearchParams, setTimeout, clearTimeout, hubFactsPending, hubListingRowFor, hubSyncVerdict,
        escapeHtml: escapeHtmlAttr, renderHubCard, renderSubmissionHistory,
        template: () => '', getPending: slug => pending.get(slug),
        setPending: (slug, value) => { pending.set(slug, value); onPending(); },
        startLifecyclePoller: callback => { onPending = callback; return () => {}; },
        fetchJson: async (path, options) => {
            if (options?.method === 'POST') return install.promise;
            if (path.startsWith('/api/extensions')) {
                if (listingFailed) throw new Error('listing offline');
                return { skills: [] };
            }
            if (catalogFailed) throw new Error('catalog offline');
            return { results: filtered ? [] : [{ slug: 'demo', sanitized_name: 'demo', latest_version: '1.0.0' }] };
        },
    });
    vm.runInContext(hubSource(), context);
    await context.initOuroborosHub(pane);
    onPending();
    assert.equal(nodes['#oh-results'].innerHTML, '', 'first failure is not an empty result');
    catalogFailed = false;
    await pane._ouroboroshubRefresh();
    assert.match(nodes['#oh-results'].innerHTML, /Hub facts unavailable/);
    assert.doesNotMatch(nodes['#oh-results'].innerHTML, /data-oh-action="install"/);
    listingFailed = false;
    await pane._ouroboroshubRefresh();
    assert.match(nodes['#oh-results'].innerHTML, /data-oh-action="install"/);
    catalogFailed = true;
    await pane._ouroboroshubRefresh();
    assert.match(nodes['#oh-results'].innerHTML, /demo/);
    assert.doesNotMatch(nodes['#oh-results'].innerHTML, /catalog offline|data-oh-action="install"/);
    assert.match(nodes['#oh-status'].textContent, /catalog offline.*Showing previous results/);
    catalogFailed = false;
    for (const rowGone of [true, false]) {
        filtered = false;
        await pane._ouroboroshubRefresh();
        install = deferred();
        const button = { dataset: { ohSlug: 'demo', ohAction: 'install' } };
        const action = nodes['#oh-results'].handlers.click({ target: {
            closest: selector => selector === '[data-oh-action]' ? button : null,
        } });
        filtered = rowGone;
        await pane._ouroboroshubRefresh();
        install.reject(new Error('hub install failed after filtering'));
        await action;
        const status = nodes['#oh-status'].textContent;
        const cards = nodes['#oh-results'].innerHTML;
        assert.equal((status + cards).split('hub install failed after filtering').length - 1, 1);
        assert.equal(status.includes('hub install failed after filtering'), rowGone);
    }
});

const HASH_H = 'c'.repeat(64);
const receiptFor = (name, version, extra = {}) => ({
    slug: name, version, content_hash: HASH_H, repository: 'razzant/OuroborosHub', pr_number: 60,
    pr_url: 'https://github.com/razzant/OuroborosHub/pull/60', published_at: '2026-09-14T00:00:00Z', ...extra,
});

// Real Hub tab controller against stubbed HTTP answers; every POST is recorded.
function hubController({ catalog, listing, confirm = async () => false }) {
    const { pane, nodes } = catalogPane('hub');
    const pending = new Map();
    const posts = [], dialogs = [];
    let onPending = () => {};
    const state = { catalog, listing, catalogError: null, postError: null };
    const context = vm.createContext({
        setTimeout, clearTimeout, hubFactsPending, hubListingRowFor, hubSyncVerdict,
        escapeHtml: escapeHtmlAttr, renderHubCard, renderSubmissionHistory,
        template: () => '', getPending: slug => pending.get(slug),
        setPending: (slug, value) => { pending.set(slug, value); onPending(); },
        clearPending: slug => { pending.delete(slug); onPending(); },
        startLifecyclePoller: callback => { onPending = callback; return () => {}; },
        emitSkillLifecycle() {},
        openConfirmDialog: async (options) => { dialogs.push(options); return confirm(options); },
        fetchJson: async (path, options) => {
            if (options?.method === 'POST') {
                posts.push({ path, body: JSON.parse(options.body || '{}') });
                if (state.postError) throw new Error(state.postError);
                return { ok: true, sanitized_name: 'x' };
            }
            if (path.startsWith('/api/extensions')) return { skills: state.listing };
            if (state.catalogError) throw new Error(state.catalogError);
            return { results: state.catalog };
        },
    });
    vm.runInContext(hubSource(), context);
    const click = (action, slug) => nodes['#oh-results'].handlers.click({ target: {
        closest: selector => selector === '[data-oh-action]' ? { dataset: { ohSlug: slug, ohAction: action } } : null,
    } });
    const search = async (query) => {
        nodes['#oh-query'].handlers.input({ target: { value: query } });
        clearTimeout(pane._ohTimer);
        await pane._ouroboroshubRefresh();
    };
    return { context, pane, nodes, posts, dialogs, state, click, search,
        html: () => nodes['#oh-results'].innerHTML, status: () => nodes['#oh-status'].textContent };
}

test('Hub Update and Use Hub version confirm before any mutation; Cancel posts nothing', async () => {
    let answer = false;
    const hub = hubController({
        catalog: [
            { slug: 'hubbed', sanitized_name: 'hubbed', latest_version: '0.3.0' },
            { slug: 'context_lens', sanitized_name: 'context_lens', latest_version: '1.1.3' },
            { slug: 'pending_update', sanitized_name: 'pending_update', latest_version: '0.3.0' },
        ],
        listing: [
            { name: 'hubbed', source: 'ouroboroshub', location: 'ouroboroshub', version: '0.2.0', content_hash: 'd'.repeat(64) },
            { name: 'context_lens', source: 'self_authored', location: 'external', version: '1.1.2', content_hash: HASH_H,
                published: receiptFor('context_lens', '1.1.2') },
            { name: 'pending_update', source: 'self_authored', location: 'external', version: '0.4.0', content_hash: HASH_H,
                published: receiptFor('pending_update', '0.4.0') },
        ],
        confirm: async () => answer,
    });
    await hub.context.initOuroborosHub(hub.pane);
    const html = hub.html();
    // Both #1314 directions: the catalog moved past the submission, and the
    // submission is not served yet — each offers the catalog copy.
    assert.match(html, /data-oh-action="adopt" data-oh-slug="context_lens">Use Hub version v1\.1\.3</);
    assert.match(html, /data-oh-action="adopt" data-oh-slug="pending_update">Use Hub version v0\.3\.0</);
    assert.match(html, /data-oh-action="update" data-oh-slug="hubbed">Update v0\.3\.0</);
    assert.doesNotMatch(html, /Submitted PR|Waiting for the hub|disabled>Submitted/);
    assert.match(html, /<summary>Submission history<\/summary>\s*<div class="skills-detail-row">Submitted v1\.1\.2 · <a href="https:\/\/github\.com\/razzant\/OuroborosHub\/pull\/60"/);
    assert.match(html, /Submitted v0\.4\.0 · <a /);

    await hub.click('update', 'hubbed');
    await hub.click('adopt', 'context_lens');
    assert.equal(hub.posts.length, 0, 'Cancel never posts');
    const [update, adopt] = hub.dialogs;
    assert.equal(update.title, 'Update hubbed');
    assert.equal(update.confirmLabel, 'Update');
    assert.match(update.body, /^Replace the local files of hubbed, including any local edits, with the current OuroborosHub copy\? Its saved data, enablement and review history stay; the new files are reviewed again and may need access granted again\.$/);
    assert.equal(JSON.stringify(update.details.rows),
        JSON.stringify([{ label: 'Installed version', value: 'v0.2.0' }, { label: 'Last seen in Hub', value: 'v0.3.0' }]));
    assert.equal(adopt.title, 'Use Hub version of context_lens');
    assert.equal(adopt.confirmLabel, 'Use Hub version');
    assert.match(adopt.body, /^Replace the local copy \(external, v1\.1\.2\), including any local edits, with the current OuroborosHub copy\? Its saved data/);
    assert.doesNotMatch(adopt.body, /grants .*kept|belong to someone else/);
    assert.equal(adopt.details.rows.find(row => row.label === 'Submitted').value, 'v1.1.2 · PR #60');
    assert.equal(adopt.details.rows.find(row => row.label === 'Last seen in Hub').value, 'v1.1.3');

    answer = true;
    await hub.click('update', 'hubbed');
    assert.equal(JSON.stringify(hub.posts.map(post => post.path)), JSON.stringify(['/api/marketplace/ouroboroshub/update/hubbed']));
    await hub.click('adopt', 'context_lens');
    assert.equal(hub.posts.length, 2, 'one confirmation, one POST');
    assert.equal(JSON.stringify(hub.posts[1]), JSON.stringify({ path: '/api/marketplace/ouroboroshub/install', body: {
        slug: 'context_lens', adopt: true, expected_content_hash: HASH_H, auto_review: true } }));
});

test('an unknown Hub or local version reads "unknown version", never a bare v, and changes no action', async () => {
    const hub = hubController({
        catalog: [
            { slug: 'hubbed', sanitized_name: 'hubbed', latest_version: '' },
            { slug: 'lens', sanitized_name: 'lens', latest_version: '' },
            { slug: 'noversion', sanitized_name: 'noversion', latest_version: '0.3.0' },
            { slug: 'blank', sanitized_name: 'blank', latest_version: '' },
        ],
        listing: [
            { name: 'hubbed', source: 'ouroboroshub', location: 'ouroboroshub', version: '0.2.0', content_hash: 'd'.repeat(64) },
            { name: 'lens', source: 'self_authored', location: 'external', version: '1.1.2', content_hash: HASH_H,
                published: receiptFor('lens', '') },
            { name: 'noversion', source: 'ouroboroshub', location: 'ouroboroshub', version: '', content_hash: 'd'.repeat(64) },
            { name: 'blank', source: 'ouroboroshub', location: 'ouroboroshub', version: '', content_hash: 'd'.repeat(64) },
        ],
    });
    await hub.context.initOuroborosHub(hub.pane);
    const html = hub.html();
    // The card header's Installed chip names the LOCAL version only; an
    // unknown local version never borrows the catalog's.
    const installedChip = slug => html.split('<article').find(card => card.includes(`data-slug="${slug}"`))
        .match(/skills-status-chip skills-status-ok">([^<]*)</)?.[1];
    assert.equal(installedChip('hubbed'), 'Installed v0.2.0');
    assert.equal(installedChip('noversion'), 'Installed');
    assert.equal(installedChip('blank'), 'Installed');
    // Eligibility is unchanged: an empty catalog string still differs from the local one.
    assert.match(html, /data-oh-action="update" data-oh-slug="hubbed">Update</);
    assert.match(html, /data-oh-action="adopt" data-oh-slug="lens">Use Hub version</);
    assert.match(html, /data-oh-action="update" data-oh-slug="noversion">Update v0\.3\.0</);
    assert.match(html, /Hub version unknown\./);
    assert.match(html, /<strong>Installed<\/strong>/, 'an unknown local version drops the suffix');
    // No label, badge, hint or button ends in a bare "v".
    assert.doesNotMatch(html, /\bv(?=[<.)\s])/);

    await hub.click('update', 'hubbed');
    await hub.click('adopt', 'lens');
    await hub.click('update', 'noversion');
    assert.equal(hub.posts.length, 0, 'Cancel never posts');
    const [update, adopt, local] = hub.dialogs;
    assert.equal(JSON.stringify(update.details.rows), JSON.stringify([
        { label: 'Installed version', value: 'v0.2.0' }, { label: 'Last seen in Hub', value: 'unknown version' }]));
    assert.match(adopt.body, /^Replace the local copy \(external, v1\.1\.2\), including any local edits/);
    assert.equal(adopt.details.rows.find(row => row.label === 'Last seen in Hub').value, 'unknown version');
    assert.equal(adopt.details.rows.find(row => row.label === 'Submitted').value, 'unknown version · PR #60');
    assert.equal(JSON.stringify(local.details.rows), JSON.stringify([
        { label: 'Installed version', value: 'unknown version' }, { label: 'Last seen in Hub', value: 'v0.3.0' }]));
});

test('a failed Hub Update offers Retry, and Retry confirms again before posting', async () => {
    let answer = true;
    const hub = hubController({
        catalog: [{ slug: 'hubbed', sanitized_name: 'hubbed', latest_version: '0.3.0' }],
        listing: [{ name: 'hubbed', source: 'ouroboroshub', location: 'ouroboroshub', version: '0.2.0', content_hash: 'd'.repeat(64) }],
        confirm: async () => answer,
    });
    await hub.context.initOuroborosHub(hub.pane);
    hub.state.postError = 'dependency install failed; previous copy restored';
    await hub.click('update', 'hubbed');
    assert.equal(hub.posts.length, 1);
    assert.match(hub.html(), /Failed[\s\S]*dependency install failed; previous copy restored/);
    assert.match(hub.html(), /data-oh-action="update" data-oh-slug="hubbed">Retry</);
    answer = false;
    await hub.click('update', 'hubbed');
    assert.equal(hub.dialogs.length, 2, 'Retry re-opens the confirmation');
    assert.equal(hub.posts.length, 1, 'a cancelled Retry posts nothing');
});

test('a catalog identity conflict keeps the local history visible but offers no action or Clear', async () => {
    const hub = hubController({
        catalog: [
            { slug: 'dup', sanitized_name: 'dup', latest_version: '1.0.0', identity_conflict: true },
            { slug: 'Dup!', sanitized_name: 'dup', latest_version: '2.0.0', identity_conflict: true },
        ],
        listing: [{ name: 'dup', source: 'self_authored', location: 'external', version: '1.0.0', content_hash: HASH_H,
            published: receiptFor('dup', '1.0.0') }],
    });
    await hub.context.initOuroborosHub(hub.pane);
    assert.match(hub.html(), /Catalog entry conflict/);
    assert.doesNotMatch(hub.html(), /data-oh-action=|data-oh-clear-publication=/);
    assert.match(hub.html(), /Submitted v1\.0\.0 · <a /);
});

test('listing-only submissions are not official or counted, and absence is judged against the whole catalog', async () => {
    const hub = hubController({
        catalog: [{ slug: 'My Skill', sanitized_name: 'My_Skill', display_name: 'My Skill', latest_version: '2.0.0',
            description: 'Served by the Hub' }],
        listing: [
            { name: 'My_Skill', source: 'self_authored', location: 'external', version: '1.0.0', content_hash: HASH_H,
                published: receiptFor('My_Skill', '1.0.0') },
            { name: 'fresh_submission', source: 'self_authored', location: 'external', version: '0.1.0', content_hash: HASH_H,
                description: 'First submission', published: receiptFor('fresh_submission', '0.1.0', { pr_url: 'javascript:alert(1)' }) },
        ],
    });
    await hub.context.initOuroborosHub(hub.pane);
    const officialBadges = () => (hub.html().match(/skills-badge-ok">official</g) || []).length;
    assert.equal(officialBadges(), 1, 'only the catalog row is official');
    assert.equal(hub.status(), '1 official skill · 1 local submission not in the catalog');
    const fresh = hub.html().slice(hub.html().indexOf('data-slug="fresh_submission"'));
    assert.match(fresh, /Not in the Hub catalog/);
    assert.match(fresh, /Submitted v0\.1\.0 · PR #60</, 'an unsafe URL keeps the other facts');
    // The unsafe URL survives only inside the escaped Clear CAS echo, never as a link.
    assert.doesNotMatch(fresh, /href="javascript|<a |data-oh-action=/);

    // The canonical name matches this query but the catalog row's searchable
    // fields do not: the name is still IN the catalog, so no absence is claimed.
    await hub.search('my_skill');
    assert.doesNotMatch(hub.html(), /Not in the Hub catalog|data-slug="My_Skill"/);
    assert.equal(hub.status(), '0 official skills');
    await hub.search('served by');
    assert.match(hub.html(), /data-slug="My Skill"/);
    assert.equal(hub.status(), '1 official skill');
    await hub.search('first submission');
    assert.match(hub.html(), /data-slug="fresh_submission"/);
    assert.equal(officialBadges(), 0);
    assert.equal(hub.status(), '0 official skills · 1 local submission not in the catalog');

    // An outage never turns into "not in the catalog".
    hub.state.catalogError = 'catalog offline';
    await hub.pane._ouroboroshubRefresh();
    assert.match(hub.html(), /Catalog unavailable/);
    assert.doesNotMatch(hub.html(), /Not in the Hub catalog/);
    assert.match(hub.html(), /Submitted v0\.1\.0/, 'history stays a local fact during the outage');
});

test('My skills Update confirms for OuroborosHub before posting; ClawHub keeps its direct Update', async () => {
    const hubSkill = { ...demo, name: 'hubbed', source: 'ouroboroshub', payload_root: 'skills/ouroboroshub/hubbed', version: '0.2.0' };
    for (const [catalog, expected] of [
        [{ available: true, settled: true, byName: new Map([['hubbed', { latest_version: '0.2.0' }]]) }, 'v0.2.0'],
        [{ available: true, settled: true, byName: new Map([['hubbed', { latest_version: '' }]]) }, 'unknown version'],
        [{ available: true, settled: true, byName: new Map() }, 'not in the catalog'],
        [{ available: false, settled: true, byName: new Map() }, 'catalog unavailable'],
        [{ available: false, settled: false, byName: new Map() }, 'not checked yet'],
    ]) {
        const container = node(), posts = [], asked = [];
        let answer = false, renders = 0;
        const context = vm.createContext({
            hubCatalog: catalog, skillsSnapshot: { rawSkills: [hubSkill] }, hubVersionText,
            confirmHubUpdate: async (name, facts) => { asked.push({ name, ...facts }); return answer; },
            postWithFeedback: async (url) => { posts.push(url); return { ok: true }; },
            closeSkillMenus() {}, emitSkillLifecycle() {}, showToast() {},
        });
        vm.runInContext(source('skills', 'function hubUpdateFacts(', '\nasync function fetchSkills')
            + source('skills', 'function attachActionHandlers(', '\nfunction activateTab'), context);
        context.attachActionHandlers(container, () => { renders += 1; }, new Set(), new Set());
        const clickUpdate = (name, sourceTag) => container.handlers.click({ target: { closest: selector => (
            selector === 'button[data-skill]'
                ? { dataset: { skill: name, source: sourceTag }, classList: { contains: cls => cls === 'skills-update' } }
                : null) } });
        await clickUpdate('hubbed', 'ouroboroshub');
        assert.equal(posts.length, 0, 'Cancel posts nothing');
        assert.equal(renders, 0);
        // Same-version manual Update stays available; the dialog names what it saw.
        assert.equal(JSON.stringify(asked), JSON.stringify([{ name: 'hubbed', localVersion: '0.2.0', hubVersion: expected }]));
        answer = true;
        await clickUpdate('hubbed', 'ouroboroshub');
        assert.equal(JSON.stringify(posts), JSON.stringify(['/api/marketplace/ouroboroshub/update/hubbed']));
        await clickUpdate('clawed', 'clawhub');
        assert.equal(asked.length, 2, 'ClawHub Update is not widened into the Hub confirmation');
        assert.equal(JSON.stringify(posts.slice(1)), JSON.stringify(['/api/marketplace/clawhub/update/clawed']));
    }
});

test('Skills header Refresh and page revisit call the currently selected catalog', async () => {
    const ids = ['content', 'skills-list', 'skills-empty', 'skills-refresh'];
    const nodes = Object.fromEntries(ids.map(id => [id, node()]));
    nodes.content.appendChild = () => {};
    const tabs = ['installed', 'marketplace', 'ouroboroshub'].map(tab => ({ ...node(), dataset: { tab } }));
    const calls = [], listeners = {};
    const context = vm.createContext({
        document: {
            createElement: () => ({ firstElementChild: {} }),
            getElementById: id => nodes[id], querySelector: () => node(),
        },
        window: {
            addEventListener: (event, callback) => { listeners[event] = callback; },
            removeEventListener: event => { delete listeners[event]; },
        },
        skillsPageTemplate: () => '', activateTab() {},
        loadHubCatalog(force) { calls.push(`catalog:${force}`); },
        attachActionHandlers: () => ({ closeMenus() {}, destroy() {} }),
        bindTabStrip: (strip, { onChange }) => {
            tabs.forEach(tab => { tab.handlers.click = () => onChange(tab.dataset.tab, tab); });
            return { select() {}, destroy() {} };
        },
        renderSkillsList: async () => { calls.push('installed'); },
        renderMarketplacePane: async () => { calls.push('marketplace'); },
        renderOuroborosHubPane: async () => { calls.push('ouroboroshub'); },
        setTimeout: callback => callback(), console,
        showToast: message => { throw new Error(message); },
    });
    vm.runInContext(source('skills', 'export function initSkills('), context);
    context.initSkills({});
    for (const tab of tabs) {
        tab.handlers.click();
        await nextTurn();
        calls.length = 0;
        await nodes['skills-refresh'].handlers.click();
        assert.deepEqual(calls, [tab.dataset.tab]);
        calls.length = 0;
        listeners['ouro:page-shown']({ detail: { page: 'skills' } });
        await nextTurn();
        // The OuroborosHub pane reads the catalog itself: no second forced read on page show.
        assert.deepEqual(calls, tab.dataset.tab === 'ouroboroshub' ? ['ouroboroshub'] : ['catalog:true', tab.dataset.tab]);
    }
    const older = deferred(), current = deferred();
    context.renderMarketplacePane = () => older.promise;
    context.renderOuroborosHubPane = () => current.promise;
    tabs[1].handlers.click();
    tabs[2].handlers.click();
    older.resolve();
    await nextTurn();
    assert.equal(nodes['skills-refresh'].disabled, true, 'older tab request does not finish the current refresh');
    current.resolve();
    await nextTurn();
    assert.equal(nodes['skills-refresh'].disabled, false);
});
