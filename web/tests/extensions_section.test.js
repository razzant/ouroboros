// Extensions (docs/DESIGN.md §10 "A section with pages"):
// Skills, MCP servers, Browse and Sources are the rows of one list, opened by the levels Settings opens its pages with
// (section_levels.js). The old Settings addresses of these pages lead here, with Back to the Settings page they were
// pressed on; a page's late read asks whether its own path is the one on screen; an explicit request overrides the page
// this device last left the section on.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createExtensions, extensionsPageOf, installedAnswer, mcpServersAnswer } from '../modules/extensions.js';
import { refreshStatus } from '../modules/mcp_settings.js';
import { initSettings } from '../modules/settings.js';
import { homeAnswers, searchEntries, searchSettings } from '../modules/settings_home.js';
import { registerSettingsPage, settingsPageFor } from '../modules/settings_pages.js';
import { HOME_ROWS, rowById, rowsOf } from '../modules/settings_rows.js';
import { capturePagePlace, createPageNavigation } from '../modules/page_navigation.js';
import { settle, world } from './helpers/settings_dom.js';

function memoryStorage(t) {
    const saved = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    const store = new Map();
    Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
        getItem: (key) => (store.has(key) ? store.get(key) : null), setItem: (key, value) => store.set(key, String(value)) } });
    t.after(() => { if (saved) Object.defineProperty(globalThis, 'localStorage', saved); else delete globalThis.localStorage; });
    return store;
}

test('the addresses of Extensions\' pages, and the one door Settings keeps to it', () => {
    assert.deepEqual(['installed', 'tools', 'tools/slack', 'tools/new', 'skill', 'skill/telegram', 'skill-sources', 'browse', 'sources']
        .map(extensionsPageOf), ['installed', 'tools', 'tools', 'tools', 'installed', 'installed', 'sources', 'browse', 'sources']);
    for (const settings of ['', 'models', 'accounts', 'computer/advanced', 'toolsx', 'skills']) assert.equal(extensionsPageOf(settings), '', settings);
    // One door in Connections, found by the names its pages and the old Skills and Tools rows had.
    const door = HOME_ROWS.find((row) => row.to === 'page:skills');
    assert.equal(door.name, 'Extensions');
    assert.equal(HOME_ROWS.some((row) => row.to === 'tools'), false, 'Settings has no Tools page row');
    for (const words of ['Skills', 'Tools', 'MCP servers']) {
        assert.ok(searchSettings(words).some((hit) => hit.id === 'home:skills'), words);
    }
    // The global tool timeout is one row of This computer › Advanced; the MCP wait stays with the servers.
    assert.equal(rowById('OUROBOROS_TOOL_TIMEOUT_SEC').page, 'computer/advanced');
    assert.equal(rowById('MCP_TOOL_TIMEOUT_SEC').page, 'tools');
    assert.equal(rowsOf('tools').some((row) => row.id === 'tools:door:skills'), false, 'Skills is the page beside it');
    // A row of a page of Extensions says where it lives now and is a door there.
    const entry = searchEntries({ forms: [] }).find((each) => each.id === 'MCP_ENABLED');
    assert.deepEqual([entry.to, entry.where], ['skills:tools', 'Extensions › MCP servers']);
    const folder = searchEntries({ forms: [] }).find((each) => each.id === 'OUROBOROS_SKILLS_REPO_PATH');
    assert.equal(folder.to, 'skills:skill-sources');
    const timeout = searchEntries({ forms: [] }).find((each) => each.id === 'OUROBOROS_TOOL_TIMEOUT_SEC');
    assert.equal(timeout.to, 'computer/advanced');
});

test('the rows\' answers say both kinds by their own names, and the owner\'s move', () => {
    assert.equal(installedAnswer({ installed: null }), null);
    assert.deepEqual(installedAnswer({ installed: 9, needs: 2 }), { text: '9 installed · 2 need you', owner: true });
    assert.deepEqual(installedAnswer({ installed: 4, needs: 0 }), { text: '4 installed', owner: false });
    const servers = [{ enabled: true, tool_count: 3 }, { enabled: true, tool_count: 0, sign_in: { state: 'needed' } }, { enabled: false }];
    assert.deepEqual(mcpServersAnswer({ enabled: true, servers }), { text: '2 of 3 on · 1 needs sign-in', owner: true });
    servers.push({ enabled: true, tool_count: 0 }, { enabled: true, tool_count: 0, code: 'MCP_CONFIG_ERROR' },
        { enabled: true, tool_count: 0, code: 'MCP_ID_AMBIGUOUS', configured_count: 2 });
    const tools = { enabled: true, servers };
    assert.deepEqual(mcpServersAnswer(tools), { text: '3 of 7 on · 1 needs sign-in', owner: true });
    assert.deepEqual(homeAnswers({ tools })['page:skills'], { text: '3 of 7 MCP servers on', owner: true });
    assert.deepEqual(mcpServersAnswer({ enabled: false, servers }), { text: 'Off', owner: false });
    assert.deepEqual(mcpServersAnswer(null), { text: 'Not checked', owner: false });
});

// Real page-navigation and section owners; only the DOM, geometry and page data are fixtures. Browser tests
// exercise the same paths in app.js with real history, focus and the 40rem container query.
function sectionWorld(t, { wide = false, draft = false, realTools = false } = {}) {
    const store = memoryStorage(t), history = [];
    const base = world(t, { history: { pushState: () => history.push('push'), back: () => history.push('back') } });
    const { env, state, layers } = base;
    const previousObserver = Object.getOwnPropertyDescriptor(globalThis, 'ResizeObserver'), observers = [];
    globalThis.ResizeObserver = class {
        constructor(callback) { this.callback = callback; this.targets = new Set(); observers.push(this); }
        observe(target) { this.targets.add(target); }
        disconnect() { this.targets.clear(); }
    };
    t.after(() => { if (previousObserver) Object.defineProperty(globalThis, 'ResizeObserver', previousObserver); else delete globalThis.ResizeObserver; });
    const originals = new Map(['tools', 'skill'].map(path => [path, settingsPageFor(path)]));
    t.after(() => originals.forEach((value, path) => registerSettingsPage(path, value)));
    const bodies = new Map(), contexts = new Map(), mounted = [], gone = [], views = [];
    const fake = title => ({ title, mount(body, { path, ctx }) {
        mounted.push(path); bodies.set(path, body); contexts.set(path, ctx);
        body.innerHTML = '<input data-draft><button data-door>Accounts</button>';
        return { unsaved: () => draft && Boolean(body.querySelector('[data-draft]').value), unmount: () => gone.push(path) };
    } });
    if (!realTools) registerSettingsPage('tools', fake(path => path === 'tools' ? 'MCP servers' : path.split('/').at(-1)));
    registerSettingsPage('skill', fake('Skill form'));
    registerSettingsPage('test-accounts', fake('Accounts'));
    registerSettingsPage('test-voice', fake('Voice'));
    let settings, section;
    // commitPage is synchronous in app.js. Restore does not go through navigation or recapture an origin.
    const commitPage = (name, options = {}) => {
        if (state.activePage !== name) {
            state.activePage = name;
            env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: name, reveal: Boolean(options.reveal) } }));
        }
        options.enter?.();
    };
    const navigation = createPageNavigation({ layers, doc: env.document, canLeave: () => true,
        capture: opener => capturePagePlace({ state, navState: {}, settingsControls: settings, skillsControls: section,
            doc: env.document, commitPage, restorePlaces: { capture: (_root, target) => () => target?.focus?.() } }, opener) });
    const showPage = (name, options = {}) => navigation.go(options.key || name, () => commitPage(name, options), options);
    // As with the real socket, both sections hear the same open event object, in registration order.
    const opens = new Set();
    const ws = { ...env.ws, on(type, fn) {
        if (type !== 'open') return env.ws.on(type, fn);
        opens.add(fn);
        return () => opens.delete(fn);
    } };
    const socketOpen = (previouslyConnected) => {
        const event = { previouslyConnected };
        opens.forEach((fn) => fn(event));
    };
    settings = initSettings({ state, layers, showPage, ws, openExtensions: (...args) => section.open(...args) });
    section = createExtensions({ ctx: { state, layers, showPage, ws, openSettings: settings.open },
        viewsHtml: '<div class="skills-scroll"><button data-view-link>Other view</button></div>',
        views: { show: view => views.push(view), hide: () => views.push('hidden') },
        facts: () => ({ installed: 3, needs: 1, sources: 2 }) });
    for (const page of [settings.page, section.page]) {
        page.getBoundingClientRect = () => ({ width: wide ? 1000 : 390 });
        page.querySelector('.section-rail').getClientRects = () => wide || page.dataset.sectionView === 'home' ? [{}] : [];
    }
    const resize = async value => { wide = value; observers.forEach(observer => observer.callback()); await settle(); };
    const back = () => (state.activePage === 'skills' ? section.page : settings.page).querySelector('[data-shell-back]');
    const close = async method => {
        if (method === 'button') await back().fire('click', { detail: 1 });
        else if (method === 'history') env.window.dispatchEvent({ type: 'popstate' });
        else layers.back('escape');
        await settle();
    };
    const current = () => section.page.querySelectorAll('[data-extensions-open]')
        .filter(row => row.getAttribute('aria-current') === 'page').map(row => row.dataset.extensionsOpen);
    // destroy() takes the section's listeners off the globals it mounted with; the fixture's after hooks
    // run in registration order, so those globals may already be restored by then. Put the world's own
    // back for the teardown, as world() does for its pagehide (helpers/settings_dom.js).
    t.after(() => {
        navigation.destroy();
        const kept = { window: globalThis.window, document: globalThis.document };
        Object.assign(globalThis, { window: env.window, document: env.document });
        try { section.destroy(); } finally { Object.assign(globalThis, kept); }
    });
    return { ...base, store, history, settings, section, navigation, showPage, resize, close, back, current, bodies, contexts, mounted, gone, views, socketOpen };
}

test('old addresses and Settings search doors forward into Extensions without callback navigation', async t => {
    const w = sectionWorld(t);
    await w.settings.open('test-voice', { map: true }); await settle();
    for (const path of ['tools/slack', 'skill/example', 'skill-sources']) {
        assert.equal(await w.settings.open(path), true); await settle();
        assert.equal(w.state.activePage, 'skills');
        assert.equal(w.back().getAttribute('aria-label'), 'Back to Settings › Voice');
        await w.close('button');
        assert.equal(w.state.settingsPath, 'test-voice');
    }
    await w.settings.open('page:skills'); await settle();
    assert.equal(w.state.activePage, 'skills');
    assert.equal(w.back().getAttribute('aria-label'), 'Back to Settings › Voice');
});

test('return announces only the restored Settings level, including a return within Settings', async t => {
    const w = sectionWorld(t);
    await w.settings.open('test-voice', { map: true });
    await w.settings.open('tools/new');
    await w.settings.open('test-accounts');
    const paths = [];
    w.env.window.addEventListener('ouro:settings-page-shown', (e) => { if (e.detail.place === 'settings') paths.push(e.detail.path); });
    await w.close('escape');
    await w.close('escape');
    assert.deepEqual(paths, ['test-voice']);
    await w.settings.open('test-accounts');
    paths.length = 0;
    await w.close('escape');
    assert.deepEqual(paths, ['test-voice']);
});

test('local pages share levels, visibility and reading place; a sibling replaces the previous clean editor', async t => {
    const w = sectionWorld(t, { wide: true });
    await w.section.open('', { map: true }); await settle();
    assert.deepEqual(w.current(), ['installed']);
    assert.equal(w.back(), null);
    w.section.go('tools/slack'); await settle();
    assert.equal(w.contexts.get('tools/slack').onScreen('tools/slack'), true);
    assert.equal(w.contexts.get('tools').onScreen('tools'), false);
    assert.equal(w.back().getAttribute('aria-label'), 'Back to MCP servers');
    assert.equal(w.views.at(-1), 'hidden');
    w.section.go('tools/files'); await settle();
    assert.deepEqual(w.gone, ['tools/slack']);
    await w.close('button');
    assert.deepEqual(w.gone, ['tools/slack', 'tools/files']);
    assert.equal(w.contexts.get('tools').onScreen('tools'), true);
    assert.equal(w.back(), null);
    assert.equal(w.store.get('ouroboros.extensions.last-page'), 'tools');
    await w.section.open('browse', { map: true }); await settle();
    assert.deepEqual(w.current(), ['browse']);
    assert.equal(w.views.at(-1), 'browse');
    w.section.paintAnswers();
    const answer = id => w.section.page.querySelector(`[data-extensions-open="${id}"]`).querySelector('[data-extensions-answer]').textContent;
    assert.deepEqual([answer('installed'), answer('sources')], ['3 installed · 1 needs you', '2 connected']);
    await w.resize(false);
    w.section.pressedAgain();
    assert.equal(w.section.page.dataset.sectionView, 'home');
    assert.equal(w.back(), null);
});

for (const wide of [false, true]) for (const method of ['button', 'escape', 'history']) {
    test(`a live return survives ${wide ? 'wide to narrow' : 'narrow to wide'} landing (${method})`, async t => {
        const w = sectionWorld(t, { wide });
        await w.settings.open('test-voice', { map: true }); await settle();
        // An explicit blank landing chooses Skills even when it needs an asynchronous resize frame.
        await w.section.open(''); await settle();
        const jump = w.layers.returnPage().id;
        const changes = [];
        const off = w.layers.onChange(snapshot => changes.push(snapshot.ids));
        const pushes = w.history.filter(item => item === 'push').length;
        await w.resize(!wide);
        assert.equal(w.layers.page().id, jump);
        assert.equal(w.back().getAttribute('aria-label'), 'Back to Settings › Voice');
        assert.ok(changes.every(ids => ids.includes(jump)), 'resize cannot expose an empty stack');
        assert.equal(w.history.filter(item => item === 'push').length, pushes);
        off();
        await w.close(method);
        assert.equal(w.state.activePage, 'settings');
        assert.equal(w.state.settingsPath, 'test-voice');
        assert.equal(w.layers.returnPage(), null);
    });
}

test('a cold wide landing keeps its return; a later child closes before it and list choices keep it', async t => {
    const w = sectionWorld(t, { wide: true });
    await w.settings.open('test-voice', { map: true }); await settle();
    await w.section.open('tools'); await settle();
    const jump = w.layers.returnPage().id;
    assert.equal(w.layers.page().id, jump);
    w.section.go('tools/slack'); await settle();
    assert.notEqual(w.layers.page().id, jump, 'a child opened after landing goes above the return');
    await w.close('escape');
    assert.equal(w.state.activePage, 'skills');
    assert.equal(w.layers.page().id, jump);
    await w.section.page.querySelector('[data-extensions-rail]').fire('click', {
        target: w.section.page.querySelector('[data-extensions-open="browse"]') });
    assert.equal(w.layers.returnPage().id, jump, 'a section choice is not a map choice');
    await w.close('button');
    assert.equal(w.state.activePage, 'settings');
});

test('a remembered landing and an explicit page request have separate meanings', async t => {
    const w = sectionWorld(t, { wide: true });
    await w.section.open('tools', { map: true }); await settle();
    assert.equal(w.store.get('ouroboros.extensions.last-page'), 'tools');
    await w.showPage('chat', { map: true });
    await w.section.open('installed'); await settle();
    assert.deepEqual(w.current(), ['installed']);
    assert.equal(w.views.at(-1), 'installed');
    assert.equal(w.back().getAttribute('aria-label'), 'Back to Main');
});

test('a failed passive MCP read clears the Extensions count and dot; only the newest request paints', async t => {
    const w = sectionWorld(t, { wide: true });
    const needed = { enabled: true, servers: [{ id: 'docs', enabled: true, tool_count: 0, sign_in: { state: 'needed' } }] };
    const reply = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });
    w.env.routes.set('/api/mcp/status', () => reply(needed));
    await w.section.open('', { map: true }); await settle();
    const row = w.section.page.querySelector('[data-extensions-open="tools"]');
    const mark = row.querySelector('.extensions-mark');
    const view = () => [row.querySelector('[data-extensions-answer]').textContent, mark.dataset.hue, mark.hidden];
    assert.deepEqual(view(), ['1 of 1 on · 1 needs sign-in', 'owner', false]);

    let finishOld;
    const holdOld = () => {
        w.env.routes.set('/api/mcp/status', () => new Promise(resolve => { finishOld = resolve; }));
        return refreshStatus();
    };
    const before = w.env.calls.length;
    const oldSuccess = holdOld();
    w.env.routes.set('/api/mcp/status', () => reply({}, 503));
    await refreshStatus();
    assert.deepEqual(view(), ['Not checked', 'ink', true]);
    finishOld(reply(needed));
    await oldSuccess;
    assert.deepEqual(view(), ['Not checked', 'ink', true]);

    const oldFailure = holdOld();
    w.env.routes.set('/api/mcp/status', () => reply(needed));
    await refreshStatus();
    finishOld(reply({}, 503));
    await oldFailure;
    assert.deepEqual(view(), ['1 of 1 on · 1 needs sign-in', 'owner', false]);
    assert.deepEqual(w.env.calls.slice(before), Array(4).fill('/api/mcp/status'));
    assert.deepEqual(w.posts, []);
});

test('passive MCP mounts and reconnects share a fresh status request, never a pre-reconnect flight', async t => {
    const w = sectionWorld(t, { wide: true, realTools: true });
    const requests = [];
    const reply = (body) => ({ ok: true, status: 200, json: async () => body });
    const status = (needsSignIn) => ({ enabled: true, sdk_available: true, servers: [{ id: 'fixture', enabled: true, tool_count: 0,
        ...(needsSignIn ? { sign_in: { state: 'sign_in_required' } } : {}) }] });
    w.env.routes.set('/api/settings', () => reply({ MCP_ENABLED: true,
        MCP_SERVERS: [{ id: 'fixture', name: 'Fixture', enabled: true, url: 'https://example.test/mcp' }] }));
    w.env.routes.set('/api/mcp/status', (_url, init) => new Promise(resolve => requests.push({ resolve, signal: init.signal })));
    const row = w.section.page.querySelector('[data-extensions-open="tools"]');
    const answer = () => row.querySelector('[data-extensions-answer]').textContent;

    await w.section.open('tools/fixture', { map: true });
    await settle();
    assert.equal(requests.length, 1, 'Extensions, MCP list, editor and shown signal coalesce');
    w.socketOpen(false);
    assert.equal(requests.length, 1, 'the first open is not a reconnect');
    w.socketOpen(true);
    await settle();
    assert.equal(requests.length, 2, 'one post-reconnect GET while the earlier one is still unanswered');
    assert.equal(requests[0].signal.aborted, true);
    requests[1].resolve(reply(status(false)));
    await settle();
    assert.equal(answer(), '1 of 1 on');
    requests[0].resolve(reply(status(true)));  // simulate an earlier transport whose reply arrives despite abort
    await settle();
    assert.equal(answer(), '1 of 1 on');

    await w.settings.open('', { map: true });
    await settle();
    assert.equal(requests.length, 3);
    w.socketOpen(true);
    await settle();
    assert.equal(requests.length, 4, 'the Settings list also needs a post-reconnect read, and the hidden section adds none');
    assert.equal(requests[2].signal.aborted, true);
    requests[3].resolve(reply(status(true)));
    requests[2].resolve(reply(status(false)));
    await settle();
    assert.ok(w.settings.page.querySelector('[data-needs="tool-sign-in"]'));

    await w.section.open('tools', { map: true });
    await settle();
    assert.equal(requests.length, 5);
    await w.showPage('chat', { map: true });
    w.socketOpen(true);
    assert.equal(requests.length, 5, 'no GET starts with both sections away');
    assert.equal(requests[4].signal.aborted, true);
    await w.section.open('tools', { map: true });
    await settle();
    assert.equal(requests.length, 6, 'return does not join a request predating an off-screen reconnect');
    requests[5].resolve(reply(status(false)));
    requests[4].resolve(reply(status(true)));
    await settle();
    assert.equal(answer(), '1 of 1 on');
    assert.deepEqual(w.posts, []);
});

for (const draft of [false, true]) for (const wide of [false, true]) {
    test(`S → Accounts → T restores the same ${draft ? 'draft' : 'clean'} S across 40rem (${wide})`, async t => {
        const w = sectionWorld(t, { wide, draft });
        await w.section.open('tools', { map: true }); await settle();
        const localOpener = w.bodies.get('tools').querySelector('[data-door]');
        w.section.go('tools/S', { opener: localOpener }); await settle();
        const original = w.bodies.get('tools/S'), opener = original.querySelector('[data-door]');
        original.querySelector('[data-draft]').value = draft ? 'kept words' : '';
        opener.focus();
        await w.settings.open('test-accounts'); await settle();
        const originJump = w.layers.returnPage().id;
        await w.settings.open('tools/T'); await settle();
        assert.equal(w.gone.includes('tools/S'), false, 'the origin owns its clean nodes while T occupies the same section');
        await w.resize(!wide);
        await w.close('escape');
        assert.equal(w.state.activePage, 'settings');
        assert.equal(w.layers.page().id, originJump);
        await w.close('button');
        assert.equal(w.state.activePage, 'skills');
        assert.equal(w.bodies.get('tools/S'), original, 'restoration does not mount a replacement form');
        assert.equal(w.mounted.filter(path => path === 'tools/S').length, 1);
        assert.equal(original.querySelector('[data-draft]').value, draft ? 'kept words' : '');
        assert.equal(w.env.document.activeElement, opener);
        assert.equal(w.layers.returnPage(), null);
        assert.equal(w.back().getAttribute('aria-label'), 'Back to MCP servers');
        await w.close('button');
        assert.equal(w.env.document.activeElement, localOpener, 'the restored level still owes its original local opener');
    });
    test(`map choice releases the held ${draft ? 'draft only after its save' : 'clean form'} (${wide})`, async t => {
        const w = sectionWorld(t, { wide, draft });
        await w.section.open('tools/S', { map: true }); await settle();
        const original = w.bodies.get('tools/S');
        original.querySelector('[data-draft]').value = draft ? 'kept words' : '';
        await w.settings.open('test-accounts'); await settle();
        await w.settings.open('tools/T'); await settle();
        await w.showPage('chat', { map: true }); await settle();
        assert.equal(w.layers.returnPage(), null);
        assert.equal(w.gone.includes('tools/S'), !draft, 'map releases a clean held form; unsaved words keep their existing policy');
        await w.section.open('tools/S', { map: true }); await settle();
        assert.equal(w.bodies.get('tools/S') === original, draft);
        if (draft) {
            original.querySelector('[data-draft]').value = ''; // the registered page reports its save
            w.section.go('tools/T'); await settle();
            assert.equal(w.gone.includes('tools/S'), true);
        }
    });
}

test('a pointer origin is captured before the asynchronous leave check, including WebKit event dispatch', async t => {
    const w = sectionWorld(t);
    await w.settings.open('test-voice', { map: true }); await settle();
    const button = w.bodies.get('test-voice').querySelector('[data-door]');
    const previousFocus = w.bodies.get('test-voice').querySelector('[data-draft]');
    previousFocus.focus();
    const click = { type: 'click', eventPhase: 1, detail: 1, target: button };
    w.env.document.dispatchEvent(click);
    const opened = w.settings.open('skill/example');
    click.eventPhase = 0;
    await opened; await settle();
    await w.close('button');
    assert.equal(w.env.document.activeElement, button);
    assert.equal(w.state.settingsPath, 'test-voice');
});

test('a link between Extensions views owes its own return; a rail row just replaces the page', async t => {
    const w = sectionWorld(t, { wide: true });
    await w.section.open('installed', { map: true }); await settle();
    const link = w.section.page.querySelector('[data-view-link]');
    await w.section.open('browse', { opener: link });
    assert.deepEqual(w.current(), ['browse']);
    assert.equal(w.back().getAttribute('aria-label'), 'Back to Extensions › Skills');
    await w.close('button');
    assert.deepEqual(w.current(), ['installed']);
    assert.equal(w.env.document.activeElement, link);
    await w.section.page.querySelector('[data-extensions-rail]').fire('click', {
        target: w.section.page.querySelector('[data-extensions-open="browse"]'),
    });
    assert.deepEqual(w.current(), ['browse']);
    assert.equal(w.layers.returnPage(), null);
});
