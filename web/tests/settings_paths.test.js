// Every address of a setting that the code names is a living one (docs/DESIGN.md "Settings"): each literal
// `openSettings(` path in web/ and skills/, each `{ path: … }` a skill hands to it, and each `row:` it asks to be
// focused resolves to a page a home row opens or a module registered, or to a described row. The tabs Settings had
// (Providers, Secrets, Behavior, Advanced, About) are no addresses. A skill installed before the rework may still
// name one: the opener then shows the home with that word searched, never an empty page.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

import { initSettings } from '../modules/settings.js';
import { isDoorId } from '../modules/settings_home.js';
import { registerSettingsPage, settingsPageFor, settingsPagePrefixes } from '../modules/settings_pages.js';
import { HAND_WRITTEN, HOME_ROWS, ROWS, SEARCH_DOORS } from '../modules/settings_rows.js';
import { Stub, settle, world } from './helpers/settings_dom.js';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const DOORS = new Set([...HOME_ROWS.map((row) => row.to).filter(isDoorId), 'main:context']);
const HOME_PAGES = HOME_ROWS.map((row) => row.to).filter((to) => !isDoorId(to));

// Pages another lane registers and rows another lane describes: named here until that lane lands, then removed. A
// name left here after its page is registered fails below, so both lists are empty on the integrated tree.
const PENDING_PAGES = {};
const PENDING_ROWS = {};

/** Why `path` is no address ('' when it is one): the home, a door, or a page that a home row opens or a module serves. */
function deadPath(path, { registered = (target) => Boolean(settingsPageFor(target)) } = {}) {
    if (path === '') return '';
    if (isDoorId(path)) return DOORS.has(path) ? '' : `no door "${path}"`;
    const first = path.split('/')[0];
    return registered(path) || HOME_PAGES.includes(first) ? '' : `no page serves "${path}"`;
}
const deadRow = (id) => ([...ROWS, ...HAND_WRITTEN].some((row) => row.id === id) || Object.hasOwn(PENDING_ROWS, id) ? '' : `no row "${id}"`);

/** The addresses a source names: `openSettings('path'…)` (a template counts up to its first `${`), the `row:` beside
 *  it, and in a skill's interface the `{ path: '…' }` objects it passes on. */
function addresses(source, { skill = false } = {}) {
    const found = [];
    for (const call of source.matchAll(/openSettings\??\.?\(\s*(['"`])([^'"`$]*)(\1|\$\{)([^)]*)/g)) {
        found.push({ kind: 'path', value: call[3] === '${' ? call[2].replace(/\/$/, '') : call[2] });
        const row = /\brow:\s*'([^']+)'/.exec(call[4]);
        if (row) found.push({ kind: 'row', value: row[1] });
    }
    if (skill) {
        for (const item of source.matchAll(/\{\s*path:\s*'([^']*)'(?:,\s*row:\s*'([^']+)')?\s*\}/g)) {
            found.push({ kind: 'path', value: item[1] });
            if (item[2]) found.push({ kind: 'row', value: item[2] });
        }
    }
    return found;
}

function sources(dir, keep) {
    return readdirSync(dir).flatMap((name) => {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) return name === 'node_modules' || name === 'tests' || name === 'vendor' ? [] : sources(path, keep);
        return keep(path) ? [path] : [];
    });
}

test('the guard fires on a dead tab name and is quiet on a living path', () => {
    for (const dead of ['providers', 'secrets', 'behavior', 'advanced', 'about', 'dashboard:nowhere', 'skillz/x']) {
        assert.notEqual(deadPath(dead), '', dead);
    }
    for (const alive of ['', 'agents', 'models/local', 'accounts/codex', 'computer/about', 'skill/telegram', 'skill-sources',
        'dashboard:costs', 'page:skills', 'main:context']) assert.equal(deadPath(alive), '', alive);
    assert.deepEqual(addresses("ctx.openSettings?.('behavior', { back })"), [{ kind: 'path', value: 'behavior' }]);
    assert.deepEqual(addresses("host.openSettings('permissions', { row: 'SOME_ROW' })"),
        [{ kind: 'path', value: 'permissions' }, { kind: 'row', value: 'SOME_ROW' }]);
    assert.deepEqual(addresses('await openSettings(`skill/${connector.connector}`)'), [{ kind: 'path', value: 'skill' }]);
    assert.deepEqual(addresses("const KEY = { path: 'secrets' };", { skill: true }), [{ kind: 'path', value: 'secrets' }]);
    assert.notEqual(deadRow('s-total-budget'), '');
    assert.equal(deadRow('TOTAL_BUDGET'), '');
});

test('every address in web/ and skills/ is a living one', () => {
    const files = [
        ...sources(join(ROOT, 'web/modules'), (path) => path.endsWith('.js')), join(ROOT, 'web/app.js'),
        ...sources(join(ROOT, 'skills'), (path) => path.endsWith('.js')),
    ];
    const dead = [];
    let seen = 0;
    for (const file of files) {
        for (const address of addresses(readFileSync(file, 'utf8'), { skill: file.startsWith(join(ROOT, 'skills')) })) {
            seen += 1;
            const why = address.kind === 'row' ? deadRow(address.value) : deadPath(address.value);
            if (why) dead.push(`${file.slice(ROOT.length)}: ${why}`);
        }
    }
    assert.ok(seen >= 8, `the scan reads the callers (${seen} addresses)`);
    assert.deepEqual(dead, []);
});

test('every page the home opens is served, or is named as pending with its lane; no pending name outlives its page', () => {
    const prefixes = settingsPagePrefixes();
    for (const page of HOME_PAGES) {
        assert.ok(prefixes.includes(page) || Object.hasOwn(PENDING_PAGES, page), `the home opens "${page}" and nothing serves it`);
    }
    for (const [page, lane] of Object.entries(PENDING_PAGES)) {
        assert.ok(!prefixes.includes(page), `"${page}" is registered: take it off the pending list (${lane})`);
    }
    for (const [id, lane] of Object.entries(PENDING_ROWS)) {
        assert.ok(![...ROWS, ...HAND_WRITTEN].some((row) => row.id === id), `row "${id}" is described: take it off the pending list (${lane})`);
    }
    for (const prefix of ['skill', 'skill-sources']) assert.ok(prefixes.includes(prefix), prefix);
    for (const door of SEARCH_DOORS) assert.equal(deadPath(door.to), '', `search offers a door to "${door.to}"`);
});

test('an address no page registers lands on the home with its last part searched; a registered one opens its page', async (t) => {
    const { content, state, showPage } = world(t);
    registerSettingsPage('test-known', { title: 'Known', mount(body) { body.innerHTML = '<p data-known>here</p>'; } });
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const search = () => content.querySelector('#settings-search').value || '';
    const results = () => content.querySelectorAll('[data-settings-result]').map((row) => row.dataset.settingsResult);
    const pages = () => content.querySelectorAll('.settings-page').map((level) => level.dataset.settingsPage);

    // A registered path opens its page, and the search field is left alone.
    assert.equal(await controls.open('test-known'), true);
    await settle();
    assert.deepEqual([pages(), state.settingsPath, search()], [['test-known'], 'test-known', '']);
    assert.ok(content.querySelector('[data-known]'));

    // A tab Settings had, as a skill written before the rework names it: the home, the address's last part searched
    // whole with its hyphens and underscores read as spaces (never its last word alone: `local-model` is the Local
    // model, not every model), the page it became among the results, and no level with nothing inside. The page
    // that was open is closed.
    for (const [address, word, found] of [['secrets', 'secrets', 'home:accounts'], ['behavior', 'behavior', 'home:permissions'],
        ['old-tabs/advanced', 'advanced', 'door:skill'], ['Providers', 'Providers', 'home:accounts'],
        ['old-tabs/local-model', 'local model', 'models:local'], ['local_model', 'local model', 'models:local']]) {
        assert.equal(await controls.open(address), true, address);
        await settle();
        assert.deepEqual([pages(), state.settingsPath, search()], [[], '', word], address);
        assert.ok(results().includes(found), `${address}: ${results()}`);
        assert.ok(content.querySelector('[data-settings-ask]'), 'the Ask line stands under the results');
    }
    // The whole part is searched, so `local-model` does not list what only `model` would find.
    assert.ok(!results().includes('models:main'), `local model: ${results()}`);
    // Words no setting has still show the home, never an empty page: the list says so and offers the Ask line.
    await controls.open('zzz-nowhere');
    await settle();
    assert.deepEqual([pages(), search(), results()], [[], 'zzz nowhere', []]);
    // From a page inside Settings the same holds, and a known path still opens over it.
    await controls.open('test-known');
    await settle();
    assert.deepEqual(pages(), ['test-known']);
});

// The fixture's element, also reading a selector of two parts (`A B`, a descendant) as a browser does: Back now stands in
// the page's header (79fac1cf4), found there as `.app-page-header [data-settings-back]` (web/modules/settings.js).
class DescendantStub extends Stub {
    querySelectorAll(selector) {
        const parts = selector.trim().split(/\s+/);
        if (parts.length !== 2 || selector.includes(',')) return super.querySelectorAll(selector);
        const found = new Set(super.querySelectorAll(parts[0]).flatMap((outer) => outer.querySelectorAll(parts[1])));
        return super.querySelectorAll(parts[1]).filter((node) => found.has(node));
    }
}

test('a page left with unsaved words is put aside and comes back as it was left; without them it is let go and drawn anew', async (t) => {
    const { env, content, state, showPage } = world(t);
    env.document.createElement = (tag) => new DescendantStub(tag);
    const drawn = [];
    const letGo = [];
    let unsaved = false;
    registerSettingsPage('test-editor', { title: 'Editor', mount(body) {
        drawn.push(body);
        body.innerHTML = '<input data-words value="stored">';
        return { unsaved: () => unsaved, unmount() { letGo.push(body); } };
    } });
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const pages = () => content.querySelectorAll('.settings-page').map((level) => level.dataset.settingsPage);
    const words = () => content.querySelector('[data-words]');
    await controls.open('test-editor');
    await settle();

    // The owner types and presses Back without Save: the page leaves the screen and is not let go.
    words().setAttribute('value', 'typed');
    unsaved = true;
    await content.querySelector('[data-settings-back]').fire('click');
    await settle();
    assert.deepEqual([pages(), state.settingsPath, letGo.length], [[], '', 0]);
    // Its path opens the same page again, as it was left, with Back named anew.
    await controls.open('test-editor');
    await settle();
    assert.deepEqual([pages(), drawn.length, words().getAttribute('value')], [['test-editor'], 1, 'typed']);
    assert.equal(content.querySelector('[data-settings-back]').getAttribute('aria-label'), 'Back to Settings');

    // Saved or discarded, the page is let go when it is left and drawn anew the next time.
    unsaved = false;
    await controls.open('');
    await settle();
    assert.deepEqual([pages(), letGo.length], [[], 1]);
    await controls.open('test-editor');
    await settle();
    assert.deepEqual([drawn.length, words().getAttribute('value')], [2, 'stored']);
});

test('words the shell searched are spent by the first press on their results; words the owner typed stay', async (t) => {
    const { content, state, showPage } = world(t);
    const dashboard = [];
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab: (...args) => dashboard.push(args), setBeforePageLeave() {} });
    const field = () => content.querySelector('#settings-search');
    const results = () => content.querySelectorAll('[data-settings-result]').map((row) => row.dataset.settingsResult);
    const press = async (id) => {
        const home = content.querySelector('[data-settings-home]');
        await home.fire('click', { target: content.querySelectorAll('[data-settings-result]').find((row) => row.dataset.settingsResult === id) });
        await settle();
    };

    // An old address: its word is searched for the owner, who presses the row it became.
    await controls.open('old-tabs/budget');
    await settle();
    assert.equal(field().value, 'budget');
    assert.ok(results().includes('home:costs'), String(results()));
    await press('home:costs');
    assert.deepEqual(dashboard.at(-1), ['costs', { row: '', back: '' }], 'the door was pressed inside Settings: the tab leads back to the home');
    // The search did its work: the field is empty and the home shows its rows again, so a return finds no stale results.
    assert.deepEqual([field().value, results()], ['', []]);
    assert.ok(content.querySelector('[data-settings-open="models"]'), 'the home\'s own rows are back');

    // Words the owner typed are the owner's: they stay after a press, as they would on any return to the home.
    field().value = 'budget';
    await field().fire('input', {});
    await press('home:costs');
    assert.equal(field().value, 'budget');
    assert.ok(results().includes('home:costs'));

    // Searched by the shell, then changed by the owner: the owner's now.
    await controls.open('old-tabs/budget');
    await settle();
    field().value = 'budget limits';
    await field().fire('input', {});
    field().value = 'budget';
    await field().fire('input', {});
    await press('home:costs');
    assert.equal(field().value, 'budget');
});

// Two panes (from 40rem) or one column, as settings.js measures them: the page box's width, and whether the list is on
// screen beside a level. Every box stands in the list's view; the card moves between the list and its pane.
function layout(t, controls, content, wide) {
    const proto = Object.getPrototypeOf(Stub.prototype);
    const before = { box: proto.getBoundingClientRect, prepend: proto.prepend };
    proto.getBoundingClientRect = function box() { return { top: 0, bottom: 10, width: 0 }; };
    proto.prepend = function prepend(node) { this.insertBefore(node, this.children[0] || null); };
    t.after(() => Object.assign(proto, { getBoundingClientRect: before.box, prepend: before.prepend }));
    controls.page.getBoundingClientRect = () => ({ top: 0, bottom: 10, width: wide.on ? 1000 : 300 });
    content.querySelector('[data-settings-home]').getClientRects = () => (wide.on ? [{ width: 1, height: 1 }] : []);
}

test('the list\'s chosen row returns to its page from a deeper level, and keeps it as it stands when it is on top', async (t) => {
    const { content, state, showPage } = world(t);
    const drawn = { first: 0, deeper: 0 };
    let firstCtx = null;
    registerSettingsPage('test-first', { title: 'First', mount(body, { ctx }) { drawn.first += 1; firstCtx = ctx; } });
    registerSettingsPage('test-deeper', { title: 'Deeper', mount() { drawn.deeper += 1; } });
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    layout(t, controls, content, { on: true });
    const pages = () => content.querySelectorAll('.settings-page').map((el) => el.dataset.settingsPage).filter(Boolean);
    const level = (path) => content.querySelectorAll('.settings-page').find((el) => el.dataset.settingsPage === path);
    const pressRow = async () => {
        const home = content.querySelector('[data-settings-home]');
        const row = new Stub('button');
        row.setAttribute('data-settings-open', 'test-first');
        home.querySelector('[data-settings-list]').appendChild(row);
        await home.fire('click', { target: row });
        await settle();
    };
    await controls.open('test-first');
    await settle();
    const firstEl = level('test-first');

    // Its page on top: the press keeps it as it stands, never drawn anew.
    await pressRow();
    assert.deepEqual([pages(), state.settingsPath, drawn.first], [['test-first'], 'test-first', 1]);

    // A deeper level over it: the same press returns to the first-level page, which is not drawn anew.
    firstCtx.openSettings('test-deeper');
    await settle();
    assert.deepEqual([pages(), state.settingsPath], [['test-first', 'test-deeper'], 'test-deeper']);
    await pressRow();
    assert.deepEqual([pages(), state.settingsPath, drawn.first], [['test-first'], 'test-first', 1]);
    assert.equal(level('test-first'), firstEl);
    assert.equal(firstEl.hidden, false);
});

test('a door from one page of the list to another chooses the other\'s row and hands it the door\'s view', async (t) => {
    const { content, state, showPage } = world(t);
    const views = [];
    let doorCtx = null;
    registerSettingsPage('test-from', { title: 'From', mount(body, { ctx, view }) { views.push(['test-from', view ?? '']); doorCtx = ctx; } });
    registerSettingsPage('test-to', { title: 'To', mount(body, { view }) { views.push(['test-to', view ?? '']); } });
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    layout(t, controls, content, { on: true });
    const home = content.querySelector('[data-settings-home]');
    const rows = Object.fromEntries(['test-from', 'test-to'].map((path) => {
        const row = new Stub('button');
        row.setAttribute('data-settings-open', path);
        home.querySelector('[data-settings-list]').appendChild(row);
        return [path, row];
    }));
    const chosen = () => Object.keys(rows).filter((path) => rows[path].getAttribute('aria-current') === 'page');
    const pages = () => content.querySelectorAll('.settings-page').map((el) => el.dataset.settingsPage).filter(Boolean);
    await controls.open('test-from');
    await settle();
    assert.deepEqual([chosen(), views], [['test-from'], [['test-from', '']]]);
    // The door opens its page over the one it was pressed on (Back returns there), and the list marks the page that is open.
    doorCtx.openSettings('test-to', { view: 'reviewers' });
    await settle();
    assert.deepEqual([pages(), chosen()], [['test-from', 'test-to'], ['test-to']]);
    assert.deepEqual(views.at(-1), ['test-to', 'reviewers']);
    // The chosen row pressed again keeps its page as it stands; the row under it returns to its own page.
    await home.fire('click', { target: rows['test-to'] });
    await settle();
    assert.deepEqual([pages(), chosen(), views.length], [['test-from', 'test-to'], ['test-to'], 2]);
    await home.fire('click', { target: rows['test-from'] });
    await settle();
    assert.deepEqual([pages(), chosen(), views.length], [['test-from'], ['test-from'], 2]);
});

test('a row chosen under the rail\'s bottom fade comes just clear of it; a row clear of both fades never moves', async (t) => {
    const { content, state, showPage } = world(t);
    registerSettingsPage('test-low', { title: 'Low', mount() {} });
    registerSettingsPage('test-mid', { title: 'Mid', mount() {} });
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    layout(t, controls, content, { on: true });
    const home = content.querySelector('[data-settings-home]');
    // The owner's window: a rail 678 px tall from y 66, its list 1168 px, scrolled to 457 (33 px more wait below).
    Object.assign(home, { scrollHeight: 1168, clientHeight: 678, scrollTop: 457 });
    home.getBoundingClientRect = () => ({ top: 66, bottom: 744, width: 300 });
    await content.querySelector('.settings-shell').fire('scroll', { target: home });  // the shell keeps where the rail was left
    const row = (path, at) => {
        const node = new Stub('button');
        node.setAttribute('data-settings-open', path);
        node.getBoundingClientRect = () => ({ top: 66 + at - home.scrollTop, bottom: 66 + at + 46 - home.scrollTop });
        home.querySelector('[data-settings-list]').appendChild(node);
        return node;
    };
    // Its bottom 4 px inside the rail, under the 32 px fade: pressed, it comes 24 px clear of the edge (FADE_CLEARANCE).
    const low = row('test-low', 1085);
    await home.fire('click', { target: low });
    await settle();
    assert.equal(home.scrollTop, 477);
    assert.equal(low.getBoundingClientRect().bottom, 744 - 24);
    const mid = row('test-mid', 700);
    await home.fire('click', { target: mid });
    await settle();
    assert.equal(home.scrollTop, 477, 'a row in the clear stays where it is');
});

test('Settings pressed again in the sidebar: two panes keep a first-level page and return from a deeper one; one column shows the list', async (t) => {
    const { env, content, state, showPage } = world(t);
    const drawn = { first: 0, deeper: 0 };
    let letGo = 0;
    let firstCtx = null;
    registerSettingsPage('test-first', { title: 'First', mount(body, { ctx }) {
        drawn.first += 1;
        firstCtx = ctx;
        return { unmount() { letGo += 1; } };
    } });
    registerSettingsPage('test-deeper', { title: 'Deeper', mount() { drawn.deeper += 1; } });
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const wide = { on: true };
    layout(t, controls, content, wide);
    const pages = () => content.querySelectorAll('.settings-page').map((el) => el.dataset.settingsPage).filter(Boolean);
    const level = (path) => content.querySelectorAll('.settings-page').find((el) => el.dataset.settingsPage === path);
    let announced = 0;
    env.window.addEventListener('ouro:settings-page-shown', () => { announced += 1; });
    await controls.open('test-first');
    await settle();
    const firstEl = level('test-first');
    announced = 0;

    // Beside the list, a first-level page: nothing closes, nothing is drawn or announced again.
    assert.equal(controls.pressedAgain(), true);
    await settle();
    assert.deepEqual([pages(), state.settingsPath, drawn.first, letGo, announced], [['test-first'], 'test-first', 1, 0, 0]);
    assert.equal(level('test-first'), firstEl);

    // A deeper level: back to its section's first-level page, as it was left.
    firstCtx.openSettings('test-deeper');
    await settle();
    controls.pressedAgain();
    await settle();
    assert.deepEqual([pages(), state.settingsPath, drawn.first, letGo], [['test-first'], 'test-first', 1, 0]);
    assert.equal(firstEl.hidden, false);

    // One column: the list, whatever level was open.
    wide.on = false;
    firstCtx.openSettings('test-deeper');
    await settle();
    controls.pressedAgain();
    await settle();
    assert.deepEqual([pages(), state.settingsPath], [[], '']);

    // Settings not shown: the press is the sidebar's ordinary showing, not this one.
    state.activePage = 'chat';
    assert.equal(controls.pressedAgain(), false);
});

// The sidebar's press (app.js `pressNavPage`): Settings pressed while shown goes to `pressedAgain`, never to an opening
// anew, and the open drawer it was pressed in closes as it does for every other place; another page is shown.
test('app.js: Settings pressed again goes to pressedAgain and closes the open drawer it was pressed in', () => {
    const source = readFileSync(join(ROOT, 'web/app.js'), 'utf8').replace(/\r\n?/g, '\n');
    const from = source.search(/\nfunction pressNavPage\(/);
    const until = source.indexOf('\n}\n', from + 1);
    assert.ok(from >= 0 && until > from, 'pressNavPage is a top-level function in app.js');
    const calls = [];
    const context = vm.createContext({
        state: { activePage: 'settings' }, navState: { mobileDrawerOpen: true },
        settingsControls: { pressedAgain: () => calls.push('again'), open: (path) => calls.push(`open:${path}`) },
        setMobileDrawerOpen: (open) => { calls.push(`drawer:${open}`); context.navState.mobileDrawerOpen = open; },
        showPage: (name) => calls.push(`show:${name}`),
    });
    vm.runInContext(source.slice(from + 1, until + 2), context);
    context.pressNavPage('settings');
    assert.deepEqual(calls, ['again', 'drawer:false']);
    // No drawer open (two panes, or the drawer already closed): nothing else happens.
    calls.length = 0;
    context.pressNavPage('settings');
    assert.deepEqual(calls, ['again']);
    // Away from Settings, or another page: the ordinary showing, which closes the drawer itself.
    calls.length = 0;
    context.state.activePage = 'chat';
    context.pressNavPage('settings');
    context.pressNavPage('dashboard');
    assert.deepEqual(calls, ['show:settings', 'show:dashboard']);
});

// The first-level page two panes land on is the one last opened on this device (docs/DESIGN.md "Settings"), kept across
// loads; a stored page no longer registered falls back to Models.
test('two panes land on the first-level page last opened on this device, or on Models when it is gone', async (t) => {
    const saved = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    const store = new Map();
    Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
        getItem: (key) => (store.has(key) ? store.get(key) : null), setItem: (key, value) => store.set(key, String(value)) } });
    t.after(() => { if (saved) Object.defineProperty(globalThis, 'localStorage', saved); else delete globalThis.localStorage; });
    registerSettingsPage('test-kept', { title: 'Kept', mount() {} });
    const land = async (sub) => {
        const { env, content, state, showPage } = world(sub);
        const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
        layout(sub, controls, content, { on: true });
        state.activePage = 'settings';
        env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'settings' } }));
        await settle();
        return { controls, state };
    };
    // A page opened in one load is stored, and the next load lands on it.
    await t.test('stored', async (sub) => {
        const { controls, state } = await land(sub);
        assert.equal(state.settingsPath, 'models', 'a first visit lands on Models');
        await controls.open('test-kept');
        await settle();
        assert.equal(store.get('ouroboros.settings.last-page'), 'test-kept');
        const next = await land(sub);
        assert.equal(next.state.settingsPath, 'test-kept');
    });
    // A stored page nobody registers any more: Models, never the home searched for it.
    await t.test('gone', async (sub) => {
        store.set('ouroboros.settings.last-page', 'no-such-page');
        const { state } = await land(sub);
        assert.equal(state.settingsPath, 'models');
    });
});

// The header's Back (settings.js paintLead): beside the list a first level opened inside Settings needs none, since
// the list it would lead to is on screen; in one column it leads to the list; a deeper level's Back is named after the
// level under it.
test('the header Back is hidden beside the list, leads to the list in one column, and names the level under a deeper one', async (t) => {
    const { env, content, state, showPage } = world(t);
    env.document.createElement = (tag) => new DescendantStub(tag);
    let deeper = null;
    registerSettingsPage('test-outer', { title: 'Outer', mount(body, { ctx }) {
        body.innerHTML = '<p>outer</p>';
        deeper = () => ctx.openSettings('test-inner');
    } });
    registerSettingsPage('test-inner', { title: 'Inner', mount(body) { body.innerHTML = '<p>inner</p>'; } });
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const back = () => content.querySelector('.app-page-header [data-settings-back]');
    const said = () => [back().hidden, back().getAttribute('aria-label')];
    const pages = () => content.querySelectorAll('.settings-page').map((level) => level.dataset.settingsPage);
    await controls.open('');
    await settle();
    assert.equal(back().hidden, true, 'the list alone: no Back');
    // Beside the list (the fixture lays every element out, so the list is on screen).
    await controls.open('test-outer');
    await settle();
    assert.deepEqual([pages(), said()], [['test-outer'], [true, 'Back to Settings']], 'a first level beside the list: the list is on screen');
    deeper();
    await settle();
    assert.deepEqual(pages(), ['test-outer', 'test-inner']);
    assert.deepEqual(said(), [false, 'Back to Outer'], 'a deeper level: Back leads to the level under it, by its name');
    await back().fire('click');
    await settle();
    assert.deepEqual([pages(), said()], [['test-outer'], [true, 'Back to Settings']]);
    // One column: the list is not on screen, so a first level shows Back to it.
    content.querySelector('[data-settings-home]').getClientRects = () => [];
    await controls.open('test-outer');
    await settle();
    assert.deepEqual(said(), [false, 'Back to Settings']);
    await back().fire('click');
    await settle();
    assert.deepEqual([pages(), back().hidden], [[], true]);
});
