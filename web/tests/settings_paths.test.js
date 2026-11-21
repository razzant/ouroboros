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

import { EXTENSIONS_PAGES } from '../modules/extensions.js';
import { initSettings } from '../modules/settings.js';
import { isDoorId } from '../modules/settings_home.js';
import { extensionsPageOf, registerSettingsPage, settingsPageFor, settingsPagePrefixes } from '../modules/settings_pages.js';
import { HAND_WRITTEN, HOME_ROWS, ROWS, SEARCH_DOORS } from '../modules/settings_rows.js';
import { Stub, settle, world } from './helpers/settings_dom.js';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const DOORS = new Set([...HOME_ROWS.map((row) => row.to).filter(isDoorId), 'main:context']);
const HOME_PAGES = HOME_ROWS.map((row) => row.to).filter((to) => !isDoorId(to));

// Pages another lane registers and rows another lane describes: named here until that lane lands, then removed. A
// name left here after its page is registered fails below, so both lists are empty on the integrated tree.
const PENDING_PAGES = {};
const PENDING_ROWS = {};

test('an empty Settings page band passes over its scroller; a named level retains its header row', () => {
    const css = readFileSync(new URL('../settings.css', import.meta.url), 'utf8');
    const wide = css.slice(css.indexOf('@container sectionplace (min-width: 40rem)'), css.indexOf('/* A phone keeps'));
    assert.match(wide, /#page-settings:is\(\[data-settings-depth="0"\], \[data-settings-depth="1"\]\) > \.app-page-header:not\(\[data-return\]\) \{[^}]*position: absolute/);
    assert.match(wide, /\.settings-home \{ margin-top: var\(--page-header-row-h\);/,
        'the list keeps its band in a native window and in a browser without a chrome token');
    assert.match(wide, /\.settings-page \{ padding-top: calc\(var\(--page-header-row-h\) \+ var\(--space-2\)\);/,
        'both hosts start the page below the band, before content scrolls beneath it');
    assert.match(css, /\.settings-page \{[^}]*padding: var\(--space-2\) var\(--page-header-padding-x\) 56px/,
        'a named band and one-column pages keep their ordinary inset');
    assert.doesNotMatch(css, /#page-settings > \.app-page-header\[data-lead\] \{ grid-template-columns/,
        'a band without actions keeps no end column through the section places\' one rule (page_header.test.js)');
});

/** Why `path` is no address ('' when it is one): the home, a door, or a page that a home row opens or a module serves. */
function deadPath(path, { registered = (target) => Boolean(settingsPageFor(target)) } = {}) {
    if (path === '') return '';
    if (path.startsWith('skills:')) {
        const target = path.slice('skills:'.length);
        // A search door can open a section view or a registered editor belonging to Extensions.
        const served = EXTENSIONS_PAGES.some((page) => page.id === target) || (extensionsPageOf(target) && registered(target));
        return served ? '' : `no Extensions page serves "${target}"`;
    }
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
    for (const dead of ['providers', 'secrets', 'behavior', 'advanced', 'about', 'dashboard:nowhere', 'skillz/x',
        'skills:nowhere', 'skills:models', 'skills:installed/missing']) {
        assert.notEqual(deadPath(dead), '', dead);
    }
    for (const alive of ['', 'agents', 'models/local', 'accounts/codex', 'computer/about', 'skill/telegram', 'skill-sources',
        'dashboard:costs', 'page:skills', 'main:context', 'skills:skill', 'skills:skill/telegram', 'skills:tools/new',
        'skills:skill-sources', 'skills:installed', 'skills:browse', 'skills:sources']) assert.equal(deadPath(alive), '', alive);
    assert.notEqual(deadPath('skills:skill', { registered: () => false }), '', 'an unregistered editor is still a dead door');
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
    const { content, state, showPage, layers } = world(t);
    registerSettingsPage('test-known', { title: 'Known', mount(body) { body.innerHTML = '<p data-known>here</p>'; } });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
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
// the page's header (79fac1cf4), found there as `.app-page-header [data-shell-back]` (web/modules/settings.js).
class DescendantStub extends Stub {
    querySelectorAll(selector) {
        const parts = selector.trim().split(/\s+/);
        if (parts.length !== 2 || selector.includes(',')) return super.querySelectorAll(selector);
        const found = new Set(super.querySelectorAll(parts[0]).flatMap((outer) => outer.querySelectorAll(parts[1])));
        return super.querySelectorAll(parts[1]).filter((node) => found.has(node));
    }
}

test('a page left with unsaved words is put aside and comes back as it was left; without them it is let go and drawn anew', async (t) => {
    const { env, content, state, showPage, layers } = world(t);
    env.document.createElement = (tag) => new DescendantStub(tag);
    const drawn = [];
    const letGo = [];
    let unsaved = false;
    registerSettingsPage('test-editor', { title: 'Editor', mount(body) {
        drawn.push(body);
        body.innerHTML = '<input data-words value="stored">';
        return { unsaved: () => unsaved, unmount() { letGo.push(body); } };
    } });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const pages = () => content.querySelectorAll('.settings-page').map((level) => level.dataset.settingsPage);
    const words = () => content.querySelector('[data-words]');
    content.querySelector('[data-settings-home]').getClientRects = () => [];
    await controls.open('test-editor');
    await settle();

    // The owner types and presses Back without Save: the page leaves the screen and is not let go. (One column: the
    // list is not beside the page, so the page shows its Back.)
    words().setAttribute('value', 'typed');
    unsaved = true;
    await content.querySelector('.app-page-header [data-shell-back]').fire('click');
    await settle();
    assert.deepEqual([pages(), state.settingsPath, letGo.length], [[], '', 0]);
    // Its path opens the same page again, as it was left, with Back named anew.
    await controls.open('test-editor');
    await settle();
    assert.deepEqual([pages(), drawn.length, words().getAttribute('value')], [['test-editor'], 1, 'typed']);
    assert.equal(content.querySelector('.app-page-header [data-shell-back]').getAttribute('aria-label'), 'Back to Settings');

    // Saved or discarded, the page is let go when it is left and drawn anew the next time.
    unsaved = false;
    await controls.open('');
    await settle();
    assert.deepEqual([pages(), letGo.length], [[], 1]);
    await controls.open('test-editor');
    await settle();
    assert.deepEqual([drawn.length, words().getAttribute('value')], [2, 'stored']);
});

test('a Settings jump retains its clean origin until Back, while a map choice releases an abandoned origin', async (t) => {
    const { createPageNavigation } = await import('../modules/page_navigation.js');
    const { env, content, state, layers } = world(t);
    const drawn = [], released = [];
    for (const path of ['test-return-clean', 'test-return-next']) registerSettingsPage(path, {
        title: path, mount(body) { drawn.push(body); return { unmount() { released.push(body); } }; },
    });
    let controls;
    const shown = name => {
        state.activePage = name;
        env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: name } }));
    };
    const nav = createPageNavigation({ layers, doc: env.document, canLeave: async () => true, capture: () => {
        const saved = controls.capture();
        return { key: `settings:${state.settingsPath}`, label: 'Settings', retains: saved.retains, release: saved.release,
            restore() { shown('settings'); saved.restore(); } };
    } });
    t.after(() => nav.destroy());
    const showPage = (name, options = {}) => nav.go(options.key || name, () => { shown(name); options.enter?.(); }, options);
    controls = initSettings({ state, ws: env.ws, layers, showPage });
    assert.equal(controls.capture().label, 'Settings', 'the home returns to the section');
    await controls.open('test-return-clean', { map: true }); await settle();
    assert.equal(controls.capture().label, 'test-return-clean', 'a deeper origin names the level it returns to');
    assert.equal(controls.capture().title, 'Settings › test-return-clean');
    const fromHome = controls.capture(content.querySelector('[data-settings-home]'));
    assert.equal(fromHome.label, 'Settings', 'a door in the list returns to the home even while a pane is open');
    assert.equal(fromHome.title, 'Settings');
    const clean = () => content.querySelectorAll('.settings-page').find(level => level.dataset.settingsPage === 'test-return-clean');
    const origin = clean();
    const body = origin.querySelector('.settings-page-body');
    await controls.open('test-return-next'); await settle();
    assert.equal(content.contains(origin), false);
    assert.equal(released.includes(body), false, 'the jump owns its detached origin');
    layers.back(); await settle();
    assert.equal(clean(), origin);
    assert.equal(drawn.filter(item => item === body).length, 1, 'Back restores the same node and bindings');
    await controls.open('test-return-next'); await settle();
    await showPage('chat', { map: true }); await settle();
    assert.equal(released.filter(item => item === body).length, 1, 'abandoning the return releases its clean origin once');
    assert.equal(layers.returnPage(), null);
});

test('words the shell searched are spent by the first press on their results; words the owner typed stay', async (t) => {
    const { content, state, showPage, layers } = world(t);
    const dashboard = [];
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab: (...args) => dashboard.push(args), setBeforePageLeave() {} });
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
    assert.deepEqual(dashboard.at(-1), ['costs', { row: '' }], 'the door was pressed inside Settings: the tab leads back to the home');
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
    const { content, state, showPage, layers } = world(t);
    const drawn = { first: 0, deeper: 0 };
    let firstCtx = null;
    registerSettingsPage('test-first', { title: 'First', mount(body, { ctx }) { drawn.first += 1; firstCtx = ctx; } });
    registerSettingsPage('test-deeper', { title: 'Deeper', mount() { drawn.deeper += 1; } });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
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
    const { content, state, showPage, layers } = world(t);
    const views = [];
    let doorCtx = null;
    registerSettingsPage('test-from', { title: 'From', mount(body, { ctx, view }) { views.push(['test-from', view ?? '']); doorCtx = ctx; } });
    registerSettingsPage('test-to', { title: 'To', mount(body, { view }) { views.push(['test-to', view ?? '']); } });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
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
    const { content, state, showPage, layers } = world(t);
    registerSettingsPage('test-low', { title: 'Low', mount() {} });
    registerSettingsPage('test-mid', { title: 'Mid', mount() {} });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    layout(t, controls, content, { on: true });
    const home = content.querySelector('[data-settings-home]');
    home.ownerDocument.defaultView.getComputedStyle = () => ({ getPropertyValue: () => '32px' });
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
    // Its bottom 4 px inside the rail, under the 32 px fade: pressed, it comes 32 px clear of the edge (the sheet’s own token).
    const low = row('test-low', 1085);
    await home.fire('click', { target: low });
    await settle();
    assert.equal(home.scrollTop, 485);
    assert.equal(low.getBoundingClientRect().bottom, 744 - 32);
    const mid = row('test-mid', 700);
    await home.fire('click', { target: mid });
    await settle();
    assert.equal(home.scrollTop, 485, 'a row in the clear stays where it is');
});

test('returning to the overview keeps the rail passage; explicitly choosing its row still reveals it', async (t) => {
    const { env, content, state, showPage } = world(t);
    globalThis.requestAnimationFrame = fn => setImmediate(fn);
    const observers = [];
    const previous = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class { constructor(fn) { observers.push(this); this.fn = fn; this.seen = []; } observe(box) { this.seen.push(box); } disconnect() {} };
    t.after(() => { globalThis.ResizeObserver = previous; });
    const controls = initSettings({ state, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const wide = { on: true };
    layout(t, controls, content, wide);
    await controls.open('');
    await settle();
    const home = content.querySelector('[data-settings-home]');
    const shell = content.querySelector('.settings-shell');
    const row = home.querySelector('[data-settings-overview-row]');
    Object.assign(home, { scrollHeight: 1200, clientHeight: 600, scrollTop: 0 });
    home.getBoundingClientRect = () => ({ top: 0, bottom: 600, width: 240 });
    row.getBoundingClientRect = () => ({ top: 40 - home.scrollTop, bottom: 86 - home.scrollTop });
    await home.fire('click', { target: row });
    await settle();
    assert.equal(state.settingsPath, '');
    home.scrollTop = 320;
    await shell.fire('scroll', { target: home });
    const resize = observers.find(observer => observer.seen.includes(shell));
    assert.ok(resize);
    // A hidden section loses its box, then gains its two panes on return: these are quiet layout changes.
    for (const name of ['dashboard', 'settings']) {
        state.activePage = name;
        wide.on = name === 'settings';
        env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: name } }));
        resize.fn([]);
        await settle();
    }
    assert.equal(home.scrollTop, 320, 'showing the section again is not another choice of its overview row');
    await home.fire('click', { target: row });
    await settle();
    assert.ok(home.scrollTop < 320 && row.getBoundingClientRect().top >= 0, 'an explicit choice still brings the row into view');
});

test('Settings pressed again in the sidebar: two panes keep a first-level page and return from a deeper one; one column shows the list', async (t) => {
    const { env, content, state, showPage, layers } = world(t);
    const drawn = { first: 0, deeper: 0 };
    let letGo = 0;
    let firstCtx = null;
    registerSettingsPage('test-first', { title: 'First', mount(body, { ctx }) {
        drawn.first += 1;
        firstCtx = ctx;
        return { unmount() { letGo += 1; } };
    } });
    registerSettingsPage('test-deeper', { title: 'Deeper', mount() { drawn.deeper += 1; } });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
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
test('app.js: Settings and Extensions pressed again go to pressedAgain and close the open drawer', () => {
    const source = readFileSync(join(ROOT, 'web/app.js'), 'utf8').replace(/\r\n?/g, '\n');
    const from = source.search(/\nfunction pressNavPage\(/);
    const until = source.indexOf('\n}\n', from + 1);
    assert.ok(from >= 0 && until > from, 'pressNavPage is a top-level function in app.js');
    const calls = [];
    const context = vm.createContext({
        state: { activePage: 'settings' }, navState: { mobileDrawerOpen: true },
        settingsControls: { pressedAgain: () => calls.push('again'), open: (path) => calls.push(`open:${path}`) },
        skillsControls: { pressedAgain: () => calls.push('extensions-again') },
        setMobileDrawerOpen: (open) => { calls.push(`drawer:${open}`); context.navState.mobileDrawerOpen = open; },
        dashboardControls: { pressedAgain: () => calls.push('dashboard-again') },
        showPage: (name, options) => {
            assert.equal(options.map, true);
            calls.push(`show:${name}`);
            options.enter?.();
            if (context.navState.mobileDrawerOpen) context.setMobileDrawerOpen(false);
        },
    });
    vm.runInContext(source.slice(from + 1, until + 2), context);
    // Execute the actual registration too: an unused correct helper is no protection against an old callback.
    const binding = source.match(/document.querySelectorAll\('\[data-nav-page\]'\)\.forEach\(btn => \{[\s\S]*?\n\}\);/)[0];
    const buttons = ['settings', 'dashboard', 'skills'].map((name) => ({ dataset: { navPage: name },
        addEventListener(type, callback) { assert.equal(type, 'click'); this.click = callback; } }));
    context.document = { querySelectorAll: () => buttons };
    vm.runInContext(binding, context);
    buttons[0].click();
    assert.deepEqual(calls, ['show:settings', 'again', 'drawer:false']);
    // No drawer open (two panes, or the drawer already closed): nothing else happens.
    calls.length = 0;
    buttons[0].click();
    assert.deepEqual(calls, ['show:settings', 'again']);
    // Away from Settings, or another page: the ordinary showing, which closes the drawer itself.
    calls.length = 0;
    context.state.activePage = 'chat';
    buttons[0].click();
    buttons[1].click();
    assert.deepEqual(calls, ['show:settings', 'show:dashboard', 'dashboard-again']);
    calls.length = 0;
    context.state.activePage = 'skills';
    buttons[2].click();
    assert.deepEqual(calls, ['show:skills', 'extensions-again']);
});

// The first-level page two panes land on is the one last opened on this device (docs/DESIGN.md "Settings"), kept across
// loads; a stored page no longer registered falls back to Models.
test('passive landing keeps a restored home row focused; an explicitly opened page still takes focus', async (t) => {
    const { env, content, state, showPage, layers } = world(t);
    registerSettingsPage('test-focus', { title: 'Focus', mount() {} });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage });
    layout(t, controls, content, { on: true });
    const row = content.querySelector('[data-settings-open="voice"]');
    row.focus();
    state.activePage = 'settings';
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'settings' } }));
    await settle();
    assert.equal(env.document.activeElement, row);
    await controls.open('test-focus'); await settle();
    assert.equal(env.document.activeElement.dataset.settingsPage, 'test-focus');
});

test('Settings chosen from the sidebar lands its page with the focus, as a place that takes it', async (t) => {
    const { env, content, state, showPage, layers } = world(t);
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage });
    layout(t, controls, content, { on: true });
    const row = new Stub('button');  // the sidebar's Settings row, outside the page: a map choice keeps its focus
    env.document.body.appendChild(row);
    row.focus();
    state.activePage = 'settings';
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'settings' } }));
    await settle();
    assert.equal(state.settingsPath, 'models');
    assert.equal(env.document.activeElement.dataset.settingsPage, 'models', 'a keyboard reaches the landed page at once');
});

test('two panes land on the first-level page last opened on this device, or on Models when it is gone', async (t) => {
    const saved = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    const store = new Map();
    Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
        getItem: (key) => (store.has(key) ? store.get(key) : null), setItem: (key, value) => store.set(key, String(value)) } });
    t.after(() => { if (saved) Object.defineProperty(globalThis, 'localStorage', saved); else delete globalThis.localStorage; });
    registerSettingsPage('test-kept', { title: 'Kept', mount() {} });
    const land = async (sub, extensions) => {
        const { env, content, state, showPage, layers } = world(sub);
        if (extensions) env.routes.set('/api/extensions', async () => ({ ok: true, status: 200, json: async () => extensions }));
        const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
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
        assert.equal(store.get('ouroboros.settings.last-page'), 'no-such-page', 'landing does not replace the owner’s choice');
        await controlsChoice(sub);
    });
    async function controlsChoice(sub) {
        const { controls } = await land(sub);
        await controls.open('test-kept');
        assert.equal(store.get('ouroboros.settings.last-page'), 'test-kept');
    }
    await t.test('remembered agent child without an existence check', async (sub) => {
        const agents = settingsPageFor('agents');
        const mounted = [];
        registerSettingsPage('agents', { ...agents, mount(_body, { path }) { mounted.push(path); } });
        sub.after(() => registerSettingsPage('agents', agents));
        store.set('ouroboros.settings.last-page', 'agents/reviewer');
        const { state } = await land(sub);
        assert.equal(state.settingsPath, 'agents/reviewer');
        assert.deepEqual(mounted, ['agents', 'agents/reviewer'], 'the remembered child keeps its named list page underneath');
        assert.equal(store.get('ouroboros.settings.last-page'), 'agents/reviewer');
    });
    await t.test('a remembered page that moved to Extensions (a skill\'s form, Tools) is not landed on here', async (sub) => {
        for (const moved of ['skill/one', 'tools', 'skill-sources']) {
            store.set('ouroboros.settings.last-page', moved);
            const installed = await land(sub, { skills: [{ name: 'one' }], live: { settings_sections: [
                { skill: 'one', title: 'One', render: { components: [{ type: 'markdown', text: 'Installed form' }] } },
            ] } });
            assert.equal(installed.state.settingsPath, 'models', moved);
            assert.equal(store.get('ouroboros.settings.last-page'), moved, 'landing never overwrites the device\'s choice');
        }
    });
    await t.test('unreadable skill registry keeps the stored choice', async (sub) => {
        store.set('ouroboros.settings.last-page', 'skill/one');
        const { state } = await land(sub, { error: 'unavailable' });
        assert.equal(state.settingsPath, 'models');
        assert.equal(store.get('ouroboros.settings.last-page'), 'skill/one');
    });
});

// The header's Back (settings.js paintLead): beside the list a first level opened inside Settings needs none, since
// the list it would lead to is on screen; in one column it leads to the list; a deeper level's Back is named after the
// level under it.
test('the header Back is hidden beside the list, leads to the list in one column, and names the level under a deeper one', async (t) => {
    const { env, content, state, showPage, layers } = world(t);
    env.document.createElement = (tag) => new DescendantStub(tag);
    let deeper = null;
    registerSettingsPage('test-outer', { title: 'Outer', mount(body, { ctx }) {
        body.innerHTML = '<p>outer</p>';
        deeper = () => ctx.openSettings('test-inner');
    } });
    registerSettingsPage('test-inner', { title: 'Inner', mount(body) { body.innerHTML = '<p>inner</p>'; } });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const back = () => content.querySelector('.app-page-header [data-shell-back]');
    const said = () => back()?.getAttribute('aria-label') || null;
    const pages = () => content.querySelectorAll('.settings-page').map((level) => level.dataset.settingsPage);
    await controls.open('');
    await settle();
    assert.equal(back(), null, 'the list alone: no Back');
    // Beside the list (the fixture lays every element out, so the list is on screen).
    await controls.open('test-outer');
    await settle();
    assert.deepEqual([pages(), said()], [['test-outer'], null], 'a first level beside the list: the list is on screen');
    deeper();
    await settle();
    assert.deepEqual(pages(), ['test-outer', 'test-inner']);
    assert.deepEqual(said(), 'Back to Outer', 'a deeper level: Back leads to the level under it, by its name');
    await back().fire('click');
    await settle();
    assert.deepEqual([pages(), said()], [['test-outer'], null]);
    // One column: the list is not on screen, so a first level shows Back to it.
    content.querySelector('[data-settings-home]').getClientRects = () => [];
    await controls.open('test-outer');
    await settle();
    assert.deepEqual(said(), 'Back to Settings');
    await back().fire('click');
    await settle();
    assert.deepEqual([pages(), back()], [[], null]);
});

test('after a jump into Settings a level opened there peels before the jump; the landing stays under its return', async (t) => {
    const { createPageNavigation } = await import('../modules/page_navigation.js');
    const { env, content, state, layers } = world(t);
    env.document.createElement = (tag) => new DescendantStub(tag);
    let deeper = null;
    registerSettingsPage('test-landing', { title: 'Landing', mount(body, { ctx }) {
        body.innerHTML = '<p>landing</p>';
        deeper = () => ctx.openSettings('test-after');
    } });
    registerSettingsPage('test-after', { title: 'After', mount(body) { body.innerHTML = '<p>after</p>'; } });
    const shown = (name) => {
        state.activePage = name;
        env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: name } }));
    };
    const nav = createPageNavigation({ layers, doc: env.document, canLeave: async () => true,
        capture: () => ({ key: state.activePage, label: 'Main', restore() { shown('chat'); } }) });
    t.after(() => nav.destroy());
    const showPage = (name, options = {}) => nav.go(options.key || name, () => { shown(name); options.enter?.(); }, options);
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    content.querySelector('[data-settings-home]').getClientRects = () => [];  // one column: a level hides the list
    const said = () => content.querySelector('.app-page-header [data-shell-back]')?.getAttribute('aria-label') || null;
    const pages = () => content.querySelectorAll('.settings-page').map((level) => level.dataset.settingsPage);
    const kinds = () => layers.ids().map((id) => (layers.get(id).returnTo ? 'return' : layers.get(id).back.label));
    // Main → Settings › Landing (the jump): its landing is the return itself.
    await controls.open('test-landing'); await settle();
    assert.deepEqual([pages(), kinds(), said()], [['test-landing'], ['return'], 'Back to Main']);
    // A level opened there afterwards goes above the return and is the first to go.
    deeper(); await settle();
    assert.deepEqual([pages(), kinds(), said()], [['test-landing', 'test-after'], ['return', 'Landing'], 'Back to Landing']);
    layers.back('escape'); await settle();
    assert.deepEqual([pages(), kinds(), said()], [['test-landing'], ['return'], 'Back to Main']);
    layers.back('escape'); await settle();
    assert.deepEqual([state.activePage, kinds()], ['chat', []]);
    // A landing's own parent stays under the jump: landing on Landing › After puts After's Back to Landing beneath the
    // return, and it is there once the jump is spent.
    await controls.open('test-landing', { map: true }); await settle();
    deeper(); await settle();
    await showPage('chat', { map: true }); await settle();
    await controls.open('test-after'); await settle();
    assert.deepEqual([pages(), kinds(), said()], [['test-landing', 'test-after'], ['Landing', 'return'], 'Back to Main']);
    layers.back('escape'); await settle();
    await showPage('settings', { map: true }); await settle();
    assert.deepEqual([pages(), kinds(), said()], [['test-landing', 'test-after'], ['Settings', 'Landing'], 'Back to Landing']);
});

test('a direct child address keeps its named parent and row; a first-level address has no extra level', async (t) => {
    const { env, content, state, showPage, layers, shown } = world(t);
    env.document.createElement = (tag) => new DescendantStub(tag);
    const models = settingsPageFor('models');
    t.after(() => registerSettingsPage('models', models));
    registerSettingsPage('models', { title: path => path.includes('/') ? 'Consciousness' : 'Models', mount() {} });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, setBeforePageLeave() {} });
    await controls.open('models/role/consciousness', { row: 'model-source' }); await settle();
    assert.equal(shown.at(-1)[1].row, 'model-source', 'the navigation owner must not reset a specifically revealed row');
    assert.equal(controls.page.dataset.settingsDepth, '2');
    assert.equal(content.querySelector('[data-settings-lead-name]').textContent, 'Consciousness');
    assert.equal(content.querySelector('[data-shell-back]').getAttribute('aria-label'), 'Back to Models');
    layers.back(); await settle();
    assert.equal(state.settingsPath, 'models');
    assert.equal(controls.page.dataset.settingsDepth, '1');
    assert.equal(content.querySelector('[data-shell-back]'), null);
    await controls.open('models'); await settle();
    assert.equal(controls.page.dataset.settingsDepth, '1', 'a root never gains a synthetic parent');
    assert.equal(shown.at(-1)[1].row, '', 'an unnamed forward landing still starts at the top');
});

test('a root font that moves the two panes\' 40rem repaints Settings: the pane and the list are watched beside the shell', async (t) => {
    const { env, content, state, showPage, layers } = world(t);
    env.document.createElement = (tag) => new DescendantStub(tag);
    const observers = [];
    const before = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class { constructor(fn) { Object.assign(this, { fn, seen: [] }); observers.push(this); } observe(box) { this.seen.push(box); } disconnect() {} };
    t.after(() => { globalThis.ResizeObserver = before; });
    let font = 16;
    env.window.getComputedStyle = () => ({ fontSize: `${font}px` });
    registerSettingsPage('test-font', { title: 'Font', mount() {} });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    layout(t, controls, content, { on: true });
    controls.page.getBoundingClientRect = () => ({ top: 0, bottom: 10, width: 735 });  // the owner's window
    await controls.open('test-font');
    await settle();
    const back = () => content.querySelector('.app-page-header [data-shell-back]');
    assert.equal(back(), null, 'beside the list: no Back');
    // The browser's font grows to 20 px: 40rem is 800 px, the sheets draw one column and hide the list, the shell keeps
    // its width. The pane, which loses the list beside it, says so, and the page leads back to the list.
    font = 20;
    content.querySelector('[data-settings-home]').getClientRects = () => [];
    const heard = (box) => observers.find((observer) => observer.seen.includes(box));
    const pane = content.querySelector('.settings-levels');
    assert.ok(heard(pane), 'the pane is watched');
    assert.ok(heard(content.querySelector('[data-settings-list]')), 'the list is watched');
    heard(pane).fn([]);
    assert.equal(back().getAttribute('aria-label'), 'Back to Settings');
});

test('Settings search belongs to the phone list and returns to the desktop header as the same working field', async (t) => {
    const { env, content, state, showPage, layers } = world(t);
    globalThis.requestAnimationFrame = fn => setImmediate(fn);
    env.document.createElement = tag => new DescendantStub(tag);
    env.window.ouroLayout = { layout: 'stack' };
    const observers = [];
    const previous = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class { constructor(fn) { observers.push(fn); } observe() {} disconnect() {} };
    t.after(() => { globalThis.ResizeObserver = previous; });
    const controls = initSettings({ state, layers, ws: { on() {} }, showPage, openDashboardTab() {}, setBeforePageLeave() {} });
    const wide = { on: false };
    layout(t, controls, content, wide);
    await controls.open('');
    await settle();
    const field = content.querySelector('#settings-search');
    const search = field.parentElement;
    const home = content.querySelector('.settings-home-column');
    const slot = content.querySelector('[data-settings-search-slot]');
    const header = content.querySelector('.app-page-header');
    assert.equal(home.firstChild, search, 'one phone band: the search starts the scrolling list');
    assert.equal(header.hasAttribute('data-search-in-home'), true);
    field.value = 'context';
    await field.fire('input');
    assert.ok(content.querySelectorAll('[data-settings-open]').some(row => row.dataset.settingsOpen === 'models'));
    // Moving a focused input's parent drops native focus; model that browser seam in this small DOM.
    const remove = search.remove.bind(search);
    search.remove = () => { if (search.contains(env.document.activeElement)) env.document.activeElement = env.document.body; remove(); };
    field.focus();
    // One pane on both sides of the shell switch: no automatic page opening has its own focus to take.
    for (const layoutName of ['docked', 'stack']) {
        env.window.ouroLayout.layout = layoutName;
        const before = search.parentElement;
        observers.forEach(fn => fn([{ target: content.querySelector('.settings-shell') }]));
        assert.equal(search.parentElement, before, 'a resize notification never reparents the field inside the observer');
        await settle();
        assert.equal(env.document.activeElement, field, 'a resize keeps the keyboard in Search');
    }
    for (const [layoutName, two, parent] of [['docked', false, slot], ['docked', true, slot], ['stack', true, home], ['stack', false, home]]) {
        wide.on = two;
        env.window.ouroLayout.layout = layoutName;
        observers.forEach(fn => fn([{ target: content.querySelector('.settings-shell') }]));
        await settle();
        assert.equal(search.parentElement, parent, `${layoutName}/${two}: search follows the list`);
        assert.equal(header.hasAttribute('data-search-in-home'), parent === home, 'the header state follows Search both ways');
        assert.equal(controls.page.dataset.searchPlace, parent === home ? 'list' : 'band', 'the writer supplies late placement to WebKit');
        assert.equal(content.querySelector('#settings-search'), field);
        assert.equal(field.value, 'context', 'moving the field preserves the query and its listener');
    }
    assert.equal(home.firstChild, search);
});
