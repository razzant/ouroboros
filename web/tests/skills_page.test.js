// The Skills page controller: lifecycle merge, the Installed filter strip,
// «Update all», publish destinations and the repair prompt.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createMySkillsDestination } from '../modules/skill_dialogs.js';
import { publicationDisclosure } from '../modules/skill_presentation.js';
import { browseCardState } from '../modules/skill_browse.js';
import { installedSkillModel } from '../modules/skill_card_renderer.js';
import { settle, Stub, world } from './helpers/settings_dom.js';

import {
    buildHealPrompt, destinationFromChoice, inFlightCount, initSkills, installedFilterCounts, lifecycleEventsFromQueue,
    matchesInstalledFilter, mergeLifecycleEvents, publishDestinations, updateAllCandidates, skillsViewsTemplate,
    SKILLS_ACTIONS,
} from '../modules/skills.js';

const LISTING = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/extensions_v4.json', import.meta.url), 'utf8'));
const SOURCES = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/skill_sources.json', import.meta.url), 'utf8')).sources;
const rows = () => structuredClone(LISTING.skills);
const byName = (name) => rows().find((skill) => skill.name === name);

test('search and chips stay above the one scroller; the views\' actions belong to the band', () => {
    const page = new Stub('div');
    page.innerHTML = skillsViewsTemplate();
    const scroll = page.querySelector('.skills-scroll');
    for (const selector of ['#skills-pane-installed', '#skills-pane-browse', '#skills-pane-sources']) {
        assert.ok(scroll.contains(page.querySelector(selector)), selector);
    }
    // Each view's search row stands over the scroller, never inside it (skills.js: "each view's search
    // row, then their one scroller").
    for (const selector of ['#skills-installed-search', '#skills-installed-filters', '#skills-browse-chrome']) {
        const chrome = page.querySelector(selector);
        assert.ok(chrome, selector);
        assert.equal(scroll.contains(chrome), false, `${selector} stays above the scroller`);
    }
    // Update all and Refresh are the band's end (extensions.js draws them there from SKILLS_ACTIONS);
    // the views block keeps no action of its own, and no tab strip of the retired page survives.
    assert.match(SKILLS_ACTIONS, /id="skills-update-all"/);
    assert.match(SKILLS_ACTIONS, /id="skills-refresh"/);
    for (const selector of ['#skills-update-all', '#skills-refresh', '.skills-tabs']) {
        assert.equal(page.querySelector(selector), null, `${selector} is not part of the views`);
    }
});

test('the chip rails start at the column edge, and on a phone run to the page\'s edges as Logs\' rail does', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.doesNotMatch(css, /\.skills-(filter|category)-strip \{[^}]*margin(-left)?: [^;}]*calc\(-1 \* var\(--space-2\)\)/,
        'no chip rail starts 8px left of the column');
    assert.match(css, /\.skills-filter-strip \{ flex-basis: 100%; \}/);
    assert.match(css, /html\[data-layout="stack"\] \.skills-filter-strip \{ flex-basis: calc\(100% \+ 2 \* var\(--page-header-padding-x\)\); \}/);
    assert.match(css, /html\[data-layout="stack"\] :is\(\.skills-filter-strip, \.skills-category-strip\) \{[^}]*margin-inline: calc\(-1 \* var\(--page-header-padding-x\)\);\s*padding-inline: var\(--page-header-padding-x\);\s*scroll-padding-inline: var\(--page-header-padding-x\);/);
});

test('Update counts trusted candidates briefly in the band and keeps the full scope in its accessible name', () => {
    const source = readFileSync(new URL('../modules/skills.js', import.meta.url), 'utf8');
    const button = { setAttribute(key, value) { this[key] = value; } };
    const context = vm.createContext({ snapshot: { rawSkills: true }, projected: () => [], paintFilters() {},
        updateAllCandidates: () => Array.from({ length: context.count }), cache: {}, activeTab: 'installed',
        updateAllBtn: button, document: { getElementById: () => null }, lifecycleEventsFromQueue: () => [], inFlightCount: () => 0,
        section: { paintAnswers() {} } });
    vm.runInContext(source.match(/    function paintChrome\(\) \{[\s\S]+?\n    \}/)[0], context);
    for (const count of [4, 0]) {
        context.count = count; context.paintChrome();
        assert.equal(button.textContent, `Update ${count}`);
        assert.equal(button['aria-label'], `Update ${count} trusted skills`);
        assert.equal(button.hidden, count < 2);
    }
});

test('a skill opens Widgets as a jump; another view is the section\'s page, never a private tab', async () => {
    const source = readFileSync(new URL('../modules/skills.js', import.meta.url), 'utf8');
    const shown = [], gone = [];
    const opener = {};
    const context = vm.createContext({ ctx: { showPage: (...args) => shown.push(args) }, activeTab: 'installed', visible: true,
        section: { async open(view, options) { gone.push([view, options.opener]); context.activeTab = view; return true; } },
        browse: { searchFor: (name) => gone.push(['search', name]) } });
    vm.runInContext(source.match(/    async function runInstalledAction\([\s\S]+?\n    \}/)[0], context);
    await context.runInstalledAction('example', 'open_widgets');
    assert.deepEqual(shown, [['widgets']], 'the action does not press a primary map row');
    await context.runInstalledAction('example', 'open_widgets', { opener });
    assert.equal(shown[1][0], 'widgets');
    assert.equal(shown[1][1].opener, opener, 'the sheet hands the same jump its underlying Browse control');
    await context.runInstalledAction('example', 'find_in_browse', { node: opener });
    assert.deepEqual(gone, [['browse', opener], ['search', 'example']], 'Find in Browse jumps from its pressed action');
});

test('the active lifecycle job joins the event stream once', () => {
    const active = { id: 'j2', target: 'b', status: 'running', kind: 'install' };
    assert.deepEqual(lifecycleEventsFromQueue({ events: [{ id: 'j1', target: 'a', status: 'failed' }, active], active }),
        [{ id: 'j1', target: 'a', status: 'failed' }, active]);
    assert.deepEqual(lifecycleEventsFromQueue(null), []);
});

test('an in-flight job annotates the existing card; an unknown target becomes a virtual row; the newest event wins', () => {
    const merged = mergeLifecycleEvents([{ name: 'a', enabled: true }], [
        { target: 'a', status: 'running', kind: 'disable' },
        { target: 'n', status: 'queued', kind: 'install', message: 'Installing n' },
    ]);
    assert.equal(merged.length, 2);
    const a = merged.find((skill) => skill.name === 'a');
    assert.equal(a.lifecycle_pending, true);
    assert.equal(a.lifecycle_kind, 'disable');
    const n = merged.find((skill) => skill.name === 'n');
    assert.equal(n.lifecycle_virtual, true);
    assert.equal(n.description, 'Installing n');
    const settled = mergeLifecycleEvents([{ name: 'a' }], [
        { target: 'a', status: 'running', kind: 'enable' },
        { target: 'a', status: 'succeeded', kind: 'enable' },
    ]);
    assert.equal(settled[0].lifecycle_pending, undefined, 'a newer terminal event clears the older pending one');
    assert.equal(inFlightCount([{ target: 'a', status: 'running' }, { target: 'a', status: 'succeeded' }, { target: 'b', status: 'queued' }]), 1);
});

test('the filter strip counts facts of the loaded list', () => {
    const counts = installedFilterCounts(rows());
    assert.deepEqual(counts, { all: 9, needs: 4, updates: 3, off: 5, made: 2 });
    assert.equal(matchesInstalledFilter(byName('zoom_meeting'), 'needs'), true);
    assert.equal(matchesInstalledFilter(byName('telegram'), 'needs'), false);
    assert.equal(matchesInstalledFilter(byName('weather'), 'updates'), true);
    assert.equal(matchesInstalledFilter(byName('old_tool'), 'updates'), true, 'withdrawn counts as an update to act on');
    assert.equal(matchesInstalledFilter(byName('minecraft-widget'), 'made'), true);
    assert.equal(matchesInstalledFilter({ name: 'v', lifecycle_virtual: true, enabled: false }, 'off'), false);
});

test('Sources Publish one filters the retained Installed cards even while the refresh is pending and if it fails', async (t) => {
    const { env, state, layers, showPage } = world(t);
    const reply = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });
    const initial = reply({ skills: [
        { name: 'local-example', source: 'external' },
        { name: 'downloaded-example', source: 'external', install: { source: { id: 'catalog-example' } } },
    ] });
    let finish;
    env.routes.set('/api/extensions', () => new Promise((resolve) => { finish = resolve; }));
    env.routes.set('/api/state', reply({ github_token_configured: false }));
    env.routes.set('/api/skills/lifecycle-queue', reply({ events: [] }));
    env.routes.set('/api/skill-sources', reply({ sources: [] }));
    env.routes.set('/api/skill-catalog', reply({ items: [], sources: [] }));
    const controls = initSkills({ state, layers, showPage });
    try {
        const { document } = env;
        document.getElementById('skills-installed-search').value = '';
        const list = document.getElementById('skills-list');
        const query = list.querySelectorAll.bind(list);
        // Supply this one direct-child selector to the Node DOM fixture.
        list.querySelectorAll = (selector) => selector === ':scope > .skills-card'
            ? list.children.filter((node) => node.matches('.skills-card')) : query(selector);
        await controls.open('installed');
        await settle(); // Let optional facts arrive before the first list; this fixture does not model template enrichment.
        finish(initial);
        await settle();
        const answer = document.getElementById('page-skills').querySelector('[data-extensions-open="installed"]')
            .querySelector('[data-extensions-answer]');
        const good = answer.textContent;
        assert.match(good, /^2 installed/);
        const local = list.querySelector('[data-skill="local-example"]');
        const downloaded = list.querySelector('[data-skill="downloaded-example"]');
        assert.equal(local.hidden, false);
        assert.equal(downloaded.hidden, false);

        await controls.open('sources');
        await settle();
        finish = undefined;
        let finishState;
        env.routes.set('/api/state', () => new Promise(resolve => { finishState = resolve; }));
        const pane = document.getElementById('skills-pane-sources');
        await pane.fire('click', { target: pane.querySelector('[data-source-action="made_here"]') });
        assert.equal(typeof finish, 'function', 'the fresh listing has started but has not answered');
        assert.equal(document.getElementById('skills-pane-installed').hidden, false);
        assert.equal(document.getElementById('skills-installed-filters').querySelector('[data-installed-filter="made"]')
            .getAttribute('aria-pressed'), 'true');
        assert.equal(local.hidden, false);
        assert.equal(downloaded.hidden, true, 'the retained cards already match Made here');

        finish(reply({ error: 'fixture unavailable' }, 503));
        await settle();
        finishState(reply({ github_token_configured: false }));
        await settle();
        assert.match(document.getElementById('skills-status').textContent, /Showing the previous list/);
        assert.equal(answer.textContent, `Last known: ${good}`);
        assert.equal(list.querySelector('[data-skill="local-example"]'), local);
        assert.equal(list.querySelector('[data-skill="downloaded-example"]'), downloaded);
        assert.equal(local.hidden, false);
        assert.equal(downloaded.hidden, true);
        const retry = document.getElementById('skills-refresh').fire('click');
        finishState(reply({ github_token_configured: false }));
        finish(reply({ skills: [] }));
        await retry;
        await settle();
        assert.equal(answer.textContent, 'None yet');
    } finally {
        controls.destroy();
    }
});

test('Sources arrival and Refresh use one authoritative list for both the rail and page', async (t) => {
    const { env, state, layers, showPage } = world(t);
    const reply = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });
    const reads = [];
    env.routes.set('/api/extensions', reply({ skills: [] }));
    env.routes.set('/api/state', reply({ github_token_configured: false }));
    env.routes.set('/api/skills/lifecycle-queue', reply({ events: [] }));
    env.routes.set('/api/skill-sources', () => new Promise(resolve => reads.push(resolve)));
    const catalog = reply({ items: [], sources: [] });
    env.routes.set('/api/skill-catalog', catalog);
    let finishCatalog;
    env.routes.set('/api/skill-catalog?refresh=1', () => new Promise(resolve => { finishCatalog = resolve; }));
    const controls = initSkills({ state, layers, showPage });
    try {
        env.document.getElementById('skills-installed-search').value = '';
        const page = env.document.getElementById('page-skills');
        const answer = page.querySelector('[data-extensions-open="sources"]').querySelector('[data-extensions-answer]');
        const pane = env.document.getElementById('skills-pane-sources');
        const refresh = env.document.getElementById('skills-refresh');
        const status = () => pane.querySelector('.skills-sources-id')?.textContent || '';
        const settleRead = async (finish, result) => { finish(result); await settle(); };
        const fail = reply({ error: 'fixture unavailable' }, 503);
        await controls.open('sources');
        await settle();
        assert.equal(reads.length, 1, 'the arriving page and place share one request');
        assert.equal(answer.textContent, '');
        assert.doesNotMatch(status(), /0 sources/);
        await settleRead(reads[0], fail);
        assert.equal(answer.textContent, 'Could not read');
        assert.match(status(), /couldn’t be read:/);
        assert.equal(reads.length, 1, 'a first-start check does not retry a failed arrival');

        await showPage('chat');
        await controls.open('sources');
        await settle();
        assert.equal(reads.length, 2, 'the return starts one fresh read');
        const pending = refresh.fire('click');
        await settle();
        assert.equal(reads.length, 3, 'an explicit Refresh supersedes the arrival read');
        await settleRead(reads[2], reply({ sources: [{ id: 'one', kind: 'git', name: 'Fresh list' }] }));
        assert.equal(answer.textContent, '1 connected');
        assert.match(status(), /1 source connected/);
        assert.match(pane.querySelector('[data-source-row="one"]').textContent, /Fresh list/);
        await settleRead(reads[1], fail);
        assert.equal(answer.textContent, '1 connected');
        assert.match(status(), /1 source connected/, 'an obsolete failure cannot repaint the view, even with catalog health pending');
        finishCatalog(catalog);
        await pending;

        const failed = refresh.fire('click');
        await settle();
        assert.equal(reads.length, 4);
        await settleRead(reads[3], fail);
        assert.equal(answer.textContent, 'Last known: 1 connected');
        assert.match(status(), /couldn’t be read:/);
        assert.match(pane.querySelector('[data-source-row="one"]').textContent, /Fresh list/);
        finishCatalog(catalog);
        await failed;
        assert.equal(reads.length, 4);
    } finally {
        controls.destroy();
    }
});

// Browse through the real page: the place's shared reads answered by the test, the board's layout inert (its
// observers and inline styles are absent from this fixture and nothing here measures).
function browsePage(t) {
    const page = world(t);
    class Observer { observe() {} disconnect() {} }
    const absent = Object.entries({ ResizeObserver: Observer, MutationObserver: Observer, CSS: { escape: String } })
        .filter(([name]) => globalThis[name] === undefined);
    for (const [name, value] of absent) globalThis[name] = value;
    const inert = { removeProperty() {}, setProperty() {}, getPropertyValue: () => '' };
    const styled = 'style' in Stub.prototype;
    if (!styled) Object.defineProperty(Stub.prototype, 'style', { configurable: true, get: () => inert });
    t.after(() => {
        for (const [name] of absent) delete globalThis[name];
        if (!styled) delete Stub.prototype.style;
    });
    const { document } = page.env;
    const reply = (body, status = 200) => ({ ok: status === 200, status, json: async () => body });
    page.env.routes.set('/api/state', reply({ github_token_configured: false }));
    page.env.routes.set('/api/skills/lifecycle-queue', reply({ events: [] }));
    const pane = () => document.getElementById('skills-pane-browse');
    const card = (id) => pane().querySelectorAll('.marketplace-card').find((node) => node.dataset.item === id);
    return { ...page, document, reply, card,
        says: (id) => card(id).querySelector('.skills-status-text').textContent,
        // A press as the page hears it: every listener from the pressed node up, the document's last.
        async press(node) {
            for (let at = node; at; at = at.parentElement) await at.fire('click', { target: node, preventDefault() {} });
            await settle();
        },
        // An entry as the host's catalog gives it: its source's id, name, address and trust ride on the entry itself.
        entry: (source, name, extra = {}) => ({ id: `${source.id}/${name}`, name, description: `About ${name}.`, version: '1.0.0',
            source_id: source.id, source_name: source.name, trust: source.trust, repo_url: source.url, ...extra }) };
}

test('Browse says how an entry installs only while its source’s setting holds: a later source list changes no card or open detail, a saved choice re-reads the catalog', async (t) => {
    const { env, state, layers, showPage, document, reply, card, says, press, entry } = browsePage(t);
    // A new install: the hub's settings are suggested, not confirmed. The host installs by them all the same.
    const hub = { id: 'hub', name: 'Example hub', kind: 'hub', part: 'official', url: 'https://example.test/hub', trust: 'trusted', confirmed: false };
    const shared = { id: 'shared', name: 'Example shared', kind: 'git', url: 'https://example.test/shared', trust: 'review', confirmed: true };
    const catalogOf = (...sources) => ({ categories: [], sources: sources.map((source) => ({ ...source, state: 'ready' })),
        items: [entry(sources[0], 'tiles', { official: true }), entry(sources[1], 'labels')] });
    let catalog = catalogOf(hub, shared);
    const sourceReads = [];
    const posts = [];
    env.routes.set('/api/extensions', reply({ skills: [] }));
    env.routes.set('/api/skill-sources', () => new Promise((resolve) => sourceReads.push(resolve)));
    env.routes.set('/api/skill-catalog', async () => reply(catalog));
    env.routes.set('/api/skill-catalog/item/shared/labels', () => new Promise(() => {}));
    env.routes.set('/api/skill-sources/hub', async (_url, init) => { posts.push(JSON.parse(init.body)); return reply({ ok: true }); });
    const controls = initSkills({ state, layers, showPage });
    const settleSources = async (...rows) => { sourceReads.at(-1)(reply({ sources: rows })); await settle(); };
    const firstStart = () => document.body.querySelector('.skills-dialog');
    const reviewOf = (sheet) => sheet.querySelectorAll('.skills-sheet-section')
        .find((section) => section.querySelector('h3')?.textContent === 'Review').querySelector('p');
    try {
        document.getElementById('skills-installed-search').value = '';
        await controls.open('browse');
        await settle();
        await settleSources(hub, shared);
        assert.equal(says('hub/tiles'), 'Installs without review while that is the suggested setting for Example hub',
            'the effective default, labelled as suggested');
        assert.equal(says('shared/labels'), 'Reviewed before it runs while that is the setting for Example shared');
        // Arriving on a new install asks once how the hub installs. The owner chooses to review its skills: the
        // write is followed by one read of the sources and one of the catalog under Browse, whose cards then say so.
        assert.ok(firstStart(), 'the first-start question opened over Browse');
        const chosen = { ...hub, trust: 'review', confirmed: true };
        catalog = catalogOf(chosen, shared);
        const row = firstStart().querySelectorAll('[data-row]').find((node) => node.dataset.row === 'hub');
        await press(row.querySelectorAll('[data-trust]').find((node) => node.dataset.trust === 'review'));
        await press(firstStart().querySelector('[data-first-ok]'));
        assert.deepEqual(posts, [{ trust: 'review', confirmed: true }]);
        assert.equal(sourceReads.length, 2, 'the saved choice read the sources again');
        await settleSources(chosen, shared);
        assert.equal(firstStart(), null);
        assert.equal(says('hub/tiles'), 'Reviewed before it runs while that is the setting for Example hub');

        // The place is shown again: its fresh source list is still on its way when a detail is opened.
        await showPage('chat');
        await controls.open('browse');
        await settle();
        assert.equal(sourceReads.length, 3, 'the return started one fresh read of the sources');
        const tiles = card('hub/tiles');
        await press(card('shared/labels').querySelector('.skills-card-name'));
        const sheet = document.body.querySelector('.skills-detail-sheet');
        const review = reviewOf(sheet);
        const shown = 'Ouroboros reviews it on this computer before it runs while that is the setting for the Example shared source. You can skip the review.';
        assert.equal(review.textContent, shown);
        const install = sheet.querySelector('[data-sheet-action="install"]');
        install.focus();
        // That list answers: Example shared was set to install without review somewhere else, same id and address.
        // An Install pressed now would go without review, since the host reads the source at the press. Nothing on
        // screen says otherwise: each sentence holds only while the setting it names does, and none is repainted
        // from a list that is no fact of the shown entries.
        await settleSources(chosen, { ...shared, trust: 'trusted' });
        assert.equal(card('hub/tiles'), tiles, 'no card was rebuilt');
        assert.equal(says('shared/labels'), 'Reviewed before it runs while that is the setting for Example shared');
        assert.equal(reviewOf(sheet), review);
        assert.equal(review.textContent, shown);
        assert.equal(sheet.querySelector('[data-sheet-action="install"]'), install, 'the open detail keeps its action');
        assert.equal(document.activeElement, install, 'and the keyboard its place');
        // The Official mark of a card is explained from that entry too, under the same condition.
        await press(tiles.querySelector('[data-official-mark]'));
        const popup = document.body.querySelector('.skills-mark-popup');
        assert.equal(popup.querySelector('p').textContent, 'Official — the maintainer of Ouroboros vouches for this exact version. '
            + 'It comes from the Example hub source (example.test/hub). A new install is still reviewed while that is the setting for the Example hub source.');
        await press(popup.querySelector('[data-popup-close]'));
        await press(sheet.querySelector('[data-sheet-close]'));
        assert.deepEqual(posts, [{ trust: 'review', confirmed: true }], 'nothing else was written');
    } finally {
        controls.destroy();
    }
});

test('Compare… promises no review and no absence of one: the replacement installs as its source is set at that moment', async (t) => {
    const { env, state, layers, showPage, document, reply, card, says, press, entry } = browsePage(t);
    const shared = { id: 'shared', name: 'Example shared', kind: 'git', url: 'https://example.test/shared', trust: 'review', confirmed: true };
    // The name is taken by a copy from a source that is not connected now: the listing names no id for it.
    const mine = { name: 'notes', version: '0.9.0', enabled: true, description: 'Mine.', install: { source: { name: 'Elsewhere', kind: 'git' } } };
    let finishListing, finishSources;
    env.routes.set('/api/extensions', () => new Promise((resolve) => { finishListing = resolve; }));
    env.routes.set('/api/skill-sources', () => new Promise((resolve) => { finishSources = resolve; }));
    env.routes.set('/api/skill-catalog', reply({ categories: [], sources: [{ ...shared, state: 'ready' }], items: [entry(shared, 'notes')] }));
    const controls = initSkills({ state, layers, showPage });
    try {
        document.getElementById('skills-installed-search').value = '';
        await controls.open('browse');
        await settle();  // the optional facts first: this fixture draws the installed list once and patches no card
        finishListing(reply({ skills: [mine] }));
        await settle();
        finishSources(reply({ sources: [shared] }));  // then the board is drawn, with the installed listing known
        await settle();
        assert.equal(says('shared/notes'), 'You have another “notes” (from Elsewhere)');
        await press(card('shared/notes').querySelector('[data-browse-action="compare"]'));
        const dialog = document.body.querySelector('.skills-dialog');
        assert.match(dialog.querySelector('h3').textContent, /Two skills are called “notes”/);
        const note = dialog.querySelector('.skills-dialog-note').textContent;
        assert.equal(note, 'Keeping yours changes nothing. Using the other one replaces yours and keeps a backup of it; '
            + 'its source’s setting in Sources decides whether it is reviewed first.');
        assert.doesNotMatch(dialog.textContent, /reviewed before it runs|without review/, 'the entry’s setting as read is no promise of the install');
        await press(dialog.querySelector('[data-dialog-cancel]'));
        assert.equal(document.body.querySelector('.skills-dialog'), null);
    } finally {
        controls.destroy();
    }
});

test('Needs you counts an enabled, permitted skill whose declared secret is not set, and nothing else for that secret', () => {
    const secret = (state) => ({ secret_config: { fields: [{ id: 'botToken', key: 'TELEGRAM_BOT_TOKEN', state }] } });
    const withTelegram = (overrides) => rows().map((skill) => (skill.name === 'telegram' ? { ...skill, ...overrides } : skill));
    const base = installedFilterCounts(rows());
    const missing = withTelegram(secret('missing'));
    assert.deepEqual(installedFilterCounts(missing), { ...base, needs: base.needs + 1 });
    assert.equal(matchesInstalledFilter(missing.find((skill) => skill.name === 'telegram'), 'needs'), true);
    for (const state of ['unknown', 'configured']) assert.deepEqual(installedFilterCounts(withTelegram(secret(state))), base, state);
    assert.deepEqual(installedFilterCounts(withTelegram({ ...secret('missing'), enabled: false })), { ...base, off: base.off + 1 },
        'an off skill is under Off; its secret asks nothing');
    const ungranted = { ...secret('missing'), grants: { all_granted: false, missing_keys: ['TELEGRAM_BOT_TOKEN'] } };
    assert.deepEqual(installedFilterCounts(withTelegram(ungranted)), { ...base, needs: base.needs + 1 }, 'Needs access, counted once');
});

test('page-module faults participate in Installed filters and Browse state', () => {
    const telegram = byName('telegram');
    const pageModules = { faults: [{ skill: telegram.name, current: true, reason: 'fixture failed' }], held: [] };
    assert.equal(installedSkillModel(telegram, { pageModules }).needs, true);
    assert.equal(matchesInstalledFilter(telegram, 'needs', installedSkillModel(telegram, { pageModules })), true);
    const counts = installedFilterCounts(rows(), pageModules);
    assert.equal(counts.needs, 5, 'a page-local fault must be reflected in Needs you');
    const browse = browseCardState(
        { id: 'telegram-row', name: telegram.name, source_id: telegram.install?.source?.id, source_name: 'built-in', installed: true },
        { installed: telegram, listed: true, pageModules },
    );
    assert.match(browse.status, /needs you/);
});

test('a held page module after Panic explains reload instead of offering a dead Start it action', () => {
    const telegram = byName('telegram');
    const model = installedSkillModel(telegram, {
        pageModules: { faults: [], held: [{ key: 'module:telegram', skill: telegram.name, reason: 'start' }], stopped: true },
    });
    const html = model.trays.join('');
    assert.equal(model.needs, true);
    assert.match(html, /Reload this page to restore it/);
    assert.doesNotMatch(html, /start_host_module/);
});

test('«Update all» covers only unmodified updates installed through trust from a row still trusted', () => {
    const list = rows();
    assert.deepEqual(updateAllCandidates(list, SOURCES).map((skill) => skill.name), ['weather']);
    const second = structuredClone(byName('weather'));
    second.name = 'weather2';
    assert.deepEqual(updateAllCandidates([...list, second], SOURCES).map((skill) => skill.name), ['weather', 'weather2']);
    const reviewNow = SOURCES.map((source) => (source.id === 'ouroboros' ? { ...source, trust: 'review' } : source));
    assert.deepEqual(updateAllCandidates([...list, second], reviewNow), [], 'a row switched to review drops out');
    assert.deepEqual(updateAllCandidates([...list, second], null).length, 2, 'without the sources read, the install basis decides');
    const unknownBase = structuredClone(second);
    unknownBase.install.base_known = false;
    assert.deepEqual(updateAllCandidates([unknownBase], SOURCES), [], 'an install whose original is unknown asks what to keep');
    for (const fixed of [{ deployment: { name: 'weather' } }, { trust: { state: 'trusted', administered: true } },
        { trust: { state: 'untrusted', administered: false } }]) {
        const managed = { ...byName('weather'), ...fixed };
        assert.equal(installedSkillModel(managed).update, false);
        assert.deepEqual(updateAllCandidates([managed], SOURCES), [], 'bulk update follows the card’s organization policy');
    }
    const permitted = { ...byName('weather'), trust: { state: 'trusted', administered: false } };
    assert.equal(installedSkillModel(permitted).update, true);
    assert.deepEqual(updateAllCandidates([permitted], SOURCES), [permitted]);
});

test('publish destinations: back to the origin, the owner’s git source, the community catalog', () => {
    const withGit = [...SOURCES, { id: 'mine', name: 'My skills repo', kind: 'git', trust: 'trusted' }];
    const madeHere = publishDestinations(byName('minecraft-widget'), withGit);
    assert.deepEqual(madeHere.map((choice) => choice.value), ['source:mine', 'source:ouroboros-community']);
    assert.match(madeHere[0].label, /^My skills repo — your own source$/);
    assert.match(madeHere[1].detail, /New skills and new authors are read by the maintainer first\./);
    // The hub's two rows are one repository: a skill from it goes back as its origin, or beside it as a new skill.
    const fromCommunity = publishDestinations(byName('anime_studio'), withGit);
    assert.deepEqual(fromCommunity.map((choice) => choice.value), ['origin:ouroboros-community', 'source:mine', 'new:ouroboros-community']);
    assert.equal(fromCommunity[0].label, 'Back to Ouroboros Community — where it came from');
    assert.equal(fromCommunity[2].label, 'Ouroboros Community — as a new skill');
    const fromOfficial = publishDestinations(byName('weather'), SOURCES);
    assert.deepEqual(fromOfficial.map((choice) => choice.value), ['origin:ouroboros', 'new:ouroboros-community'],
        'from either half of the hub: never «share with everyone» into the same repository');
    const pointer = byName('anime_studio');
    pointer.install.release.repo = 'https://github.com/CrossRead/anime';
    assert.equal(publishDestinations(pointer, withGit)[0].label, 'Back to CrossRead/anime — its author’s repository',
        'a hub entry that points at its author’s repository goes back there');
    // The destination the host takes ({source_id, mode}: origin | entry | new_entry).
    assert.deepEqual(destinationFromChoice('origin:ouroboros-community'), { source_id: 'ouroboros-community', mode: 'origin' });
    assert.deepEqual(destinationFromChoice('new:ouroboros-community'), { source_id: 'ouroboros-community', mode: 'new_entry' });
    assert.deepEqual(destinationFromChoice('source:mine'), { source_id: 'mine', mode: 'entry' });
    assert.equal(destinationFromChoice(''), null);
    assert.equal(destinationFromChoice('default'), null, 'no connected destination keeps the host default, never an invented source id');
    // A hub read from a folder of its repository takes no entries; its pointer entries still go back to their authors.
    const folderHub = { id: 'team-hub', name: 'Team hub', kind: 'hub', url: 'https://github.com/team/tools', path: 'catalog' };
    const fromFolderHub = byName('anime_studio');
    fromFolderHub.install.source = { id: 'team-hub', kind: 'hub', name: 'Team hub' };
    assert.deepEqual(publishDestinations(fromFolderHub, [...withGit, folderHub]).map((choice) => choice.value),
        ['source:mine', 'source:ouroboros-community']);
    fromFolderHub.install.release.repo = 'https://github.com/CrossRead/anime';
    assert.equal(publishDestinations(fromFolderHub, [...withGit, folderHub])[0].value, 'origin:team-hub');
    // A view pinned to a commit never reads a publication: not a destination of its own.
    const pinned = { id: 'pinned', name: 'Pinned', kind: 'git', url: 'https://github.com/alice/skills', commit: 'a'.repeat(40) };
    assert.deepEqual(publishDestinations(byName('minecraft-widget'), [...withGit, pinned]).map((choice) => choice.value),
        ['source:mine', 'source:ouroboros-community']);
    // Neither does a view of a tag named in full; a branch named in full is an ordinary destination.
    const tagged = { ...pinned, id: 'tagged', commit: undefined, ref: 'refs/tags/v1.0.0' };
    const branch = { ...pinned, id: 'branch', commit: undefined, ref: 'refs/heads/main' };
    assert.deepEqual(publishDestinations(byName('minecraft-widget'), [...withGit, tagged, branch]).map((choice) => choice.value),
        ['source:mine', 'source:branch', 'source:ouroboros-community']);
    assert.deepEqual(publishDestinations(byName('telegram'), SOURCES).map((choice) => choice.value), ['source:ouroboros-community']);
    assert.deepEqual(publishDestinations(byName('telegram'), SOURCES.filter((row) => row.kind !== 'hub')), [],
        'nothing to choose: the host publishes to its default');
});

test('a copy whose view is connected again under another id follows the source the host matches now', () => {
    // Team now reads another view of the repository; Shared reads the view the copy came from. The host names
    // Shared in the listing, under the name the copy was installed with.
    const team = { id: 'team', name: 'Team', kind: 'git', url: 'https://github.com/example/skills', ref: 'next', path: 'pack', trust: 'trusted' };
    const shared = { ...team, id: 'shared', name: 'Shared', ref: 'stable', trust: 'review' };
    const alpha = byName('weather');
    alpha.install.source = { id: 'shared', name: 'Team', kind: 'git' };
    assert.deepEqual(publishDestinations(alpha, [team, shared]).map((choice) => choice.value), ['origin:shared', 'source:team']);
    assert.deepEqual(updateAllCandidates([alpha], [team, shared]), [], 'its updates come from Shared, a review row');
    assert.deepEqual(updateAllCandidates([alpha], [team, { ...shared, trust: 'trusted' }]).map((skill) => skill.name), ['weather']);
});

test('the repair prompt carries bounded diagnostics as data', () => {
    const prompt = buildHealPrompt({ ...byName('minecraft-widget'), load_error: 'x'.repeat(5000) });
    assert.match(prompt, /Repair and run the installed Ouroboros skill selected in the Skills UI\./);
    assert.match(prompt, /"name": "minecraft-widget"/);
    assert.match(prompt, /widget_safety/);
    assert.ok(prompt.length < 9000, 'the load error is bounded');
});

test('saved My skills and git origins keep unknown visibility after a private connection', async () => {
    const answer = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/my_skills.json', import.meta.url), 'utf8')).private;
    const connected = await createMySkillsDestination(async () => answer);
    assert.match(publicationDisclosure(connected.private, connected.repository), /your private repository/);
    const saved = publishDestinations(byName('minecraft-widget'), [answer.source])[0];
    assert.equal(saved.repository, answer.repository);
    assert.equal(saved.private, undefined, 'the saved source contains no visibility fact');
    assert.equal(publicationDisclosure(saved.private, saved.repository),
        "Everything in the skill's folder goes to owner/ouroboros-skills; whoever can read that repository can see it.");
    for (const kind of ['git', 'hub']) {
        const source = { ...answer.source, kind };
        const origin = publishDestinations({ install: { source } }, [source])[0];
        assert.equal(origin.repository, answer.repository);
        assert.match(publicationDisclosure(origin.private, origin.repository), /whoever can read that repository/);
    }
    const pointer = byName('anime_studio');
    pointer.install.release.repo = 'https://github.com/author/private-skill';
    const choice = publishDestinations(pointer, SOURCES)[0];
    assert.equal(choice.repository, 'author/private-skill');
    assert.match(publicationDisclosure(choice.private, choice.repository), /goes to author\/private-skill; whoever can read/);
});
