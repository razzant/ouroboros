// Every address of a setting that the code names is a living one (docs/DESIGN.md "Settings"): each literal
// `openSettings(` path in web/ and skills/, each `{ path: … }` a skill hands to it, and each `row:` it asks to be
// focused resolves to a page a home row opens or a module registered, or to a described row. The tabs Settings had
// (Providers, Secrets, Behavior, Advanced, About) are no addresses. A skill installed before the rework may still
// name one: the opener then shows the home with that word searched, never an empty page.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';

import { initSettings } from '../modules/settings.js';
import { isDoorId } from '../modules/settings_home.js';
import { registerSettingsPage, settingsPageFor, settingsPagePrefixes } from '../modules/settings_pages.js';
import { HAND_WRITTEN, HOME_ROWS, ROWS, SEARCH_DOORS } from '../modules/settings_rows.js';
import { settle, world } from './helpers/settings_dom.js';

const ROOT = new URL('../../', import.meta.url).pathname;
const DOORS = new Set([...HOME_ROWS.map((row) => row.to).filter(isDoorId), 'main:context']);
const HOME_PAGES = HOME_ROWS.map((row) => row.to).filter((to) => !isDoorId(to));

// Pages another lane registers and rows another lane describes: named here until that lane lands, then removed. A
// name left here after its page is registered fails below, so both lists are empty on the integrated tree.
const PENDING_PAGES = { models: 'lane E2', agents: 'lane E1', tools: 'lane E2' };
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

test('an address no page registers lands on the home with its last word searched; a registered one opens its page', async (t) => {
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

    // A tab Settings had, as a skill written before the rework names it: the home, the word searched, the page it
    // became among the results, and no level with nothing inside. The page that was open is closed.
    for (const [address, word, found] of [['secrets', 'secrets', 'home:accounts'], ['behavior', 'behavior', 'home:permissions'],
        ['old-tabs/advanced', 'advanced', 'door:skill'], ['Providers', 'Providers', 'home:accounts']]) {
        assert.equal(await controls.open(address), true, address);
        await settle();
        assert.deepEqual([pages(), state.settingsPath, search()], [[], '', word], address);
        assert.ok(results().includes(found), `${address}: ${results()}`);
        assert.ok(content.querySelector('[data-settings-ask]'), 'the Ask line stands under the results');
    }
    assert.ok(results().includes('home:accounts'));
    // A word no setting has still shows the home, never an empty page: the list says so and offers the Ask line.
    await controls.open('zzz-nowhere');
    await settle();
    assert.deepEqual([pages(), search(), results()], [[], 'nowhere', []]);
    // From a page inside Settings the same holds, and a known path still opens over it.
    await controls.open('test-known');
    await settle();
    assert.deepEqual(pages(), ['test-known']);
});
