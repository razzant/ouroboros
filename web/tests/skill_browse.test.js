// Extensions → Browse: sort, filter, category counts, the recommended shelf and the
// consequence row each card shows before its one action is pressed.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
    CONNECTORS, browseCardState, categoryCounts, createSkillBrowse, installPolicy, isConnector, matchesQuery, recommendedMissing, renderBrowseCard,
    shelfItems, sortCatalogItems,
} from '../modules/skill_browse.js';
import { domDouble } from './dom_double.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/skills_v4/${name}`, import.meta.url), 'utf8'));
const CATALOG = fixture('skill_catalog.json');
const LISTING = fixture('extensions_v4.json').skills;
const item = (id) => structuredClone(CATALOG.items.find((entry) => entry.id === id));
const installed = (name) => structuredClone(LISTING.find((skill) => skill.name === name));
const text = (html) => html.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
const tone = (html) => html.match(/^<span class="ui-status skills-status" data-tone="([^"]+)"/)?.[1];
// Work that runs now: the shared moving mark in the sentence's ink (web/modules/status_mark.js).
const MOVING = /data-work><span class="status-mark" aria-hidden="true" data-shape="moving" data-hue="ink">/;

test('Popular sorts by votes with unknown counts last; Name and Recently updated are stable', () => {
    const items = CATALOG.items;
    const popular = sortCatalogItems(items, 'popular').map((entry) => entry.name);
    assert.deepEqual(popular.slice(0, 4), ['duckduckgo', 'pdf_tools', 'weather', 'whoop-health']);
    assert.ok(popular.indexOf('happy_farm') < popular.indexOf('telegram'), 'a counted zero outranks an unknown count');
    assert.deepEqual(sortCatalogItems(items, 'name').slice(0, 2).map((entry) => entry.name), ['chess-3d', 'duckduckgo']);
    // «Recently updated» orders by the host's dates; an item without one (built in, a folder, a release
    // older than the fetched history) sorts last, by name.
    const updated = sortCatalogItems(items, 'updated').map((entry) => entry.name);
    assert.deepEqual(updated.slice(0, 3), ['slack-bridge', 'weather', 'whoop-health']);
    assert.deepEqual(updated.slice(-4), ['chess-3d', 'happy_farm', 'old_tool', 'telegram'], 'no date sorts last, by name');
    assert.deepEqual(sortCatalogItems(items, 'best', 'weather').slice(0, 1).map((entry) => entry.name), ['weather']);
});

test('the local filter reads names, descriptions, authors and category labels', () => {
    const labels = new Map(CATALOG.categories.map((cat) => [cat.id, cat.label]));
    assert.equal(matchesQuery(item('ouroboros/duckduckgo'), 'web search', labels), true);
    assert.equal(matchesQuery(item('ouroboros/pdf_tools'), 'kazzand', labels), true);
    assert.equal(matchesQuery(item('ouroboros/pdf_tools'), 'documents', labels), true);
    assert.equal(matchesQuery(item('ouroboros/pdf_tools'), 'telegram', labels), false);
    assert.equal(matchesQuery(item('ouroboros/pdf_tools'), '', labels), true);
});

// A hub title unrelated to the id, before and after a rename; the id typed as words; ordinary neighbours.
const titled = (title, extra = {}) => ({ ...item('ouroboros/weather'), id: 'ouroboros/claudexor_quotas', name: 'claudexor_quotas', title,
    description: 'Shows what is left of each subscription.', category: [], owner: 'razzant', signals: { votes: 0 }, ...extra });
// Three names that share no word: the id, the catalog's title and the title its widget declares.
const disjoint = () => ({ ...item('ouroboros/weather'), id: 'ouroboros/relay_post', name: 'relay_post', title: 'Messages for others',
    widget_title: 'Courier Desk', description: 'Hands notes to people who are not here.', category: [], owner: '', signals: { votes: 0 } });

test('a card is found by the title it prints, its widget\'s title and its id typed with any of the three separators', () => {
    const labels = new Map(CATALOG.categories.map((cat) => [cat.id, cat.label]));
    for (const title of ['Account limits', 'Claudexor Quotas']) {
        const entry = titled(title);
        assert.match(renderBrowseCard(entry, { state: { status: '', action: '' } }), new RegExp(`>${title}<`), 'the card prints the title');
        for (const query of [title, title.toUpperCase(), ` ${title.toLowerCase()} `, 'claudexor_quotas', 'Claudexor Quotas',
            'claudexor-quotas', 'claudexor', 'razzant', 'subscription']) {
            assert.equal(matchesQuery(entry, query, labels), true, `${title}: ${query}`);
        }
        assert.equal(matchesQuery(entry, 'quotas claudexor', labels), false, 'a name is matched as typed, in order');
        assert.equal(matchesQuery(entry, 'limits weather', labels), false);
        assert.equal(matchesQuery(entry, 'telegram', labels), false);
    }
    const relay = disjoint();
    const card = renderBrowseCard(relay, { state: { status: '', action: '' } });
    assert.match(card, />Messages for others</);
    assert.doesNotMatch(card, /Courier Desk/, 'the widget\'s title finds the card without being printed on it');
    for (const query of ['Messages for others', 'messages FOR', 'Courier Desk', 'courier_desk', 'courier-desk', 'relay post', 'relay-post', 'relay_post']) {
        assert.equal(matchesQuery(relay, query, labels), true, query);
    }
    assert.equal(matchesQuery(relay, 'Desk Courier', labels), false);
    // Other cards keep their own matches and gain none from these names.
    for (const id of ['ouroboros/pdf_tools', 'ouroboros/duckduckgo', 'builtin/telegram']) {
        for (const query of ['Account limits', 'Courier Desk', 'Messages for others']) assert.equal(matchesQuery(item(id), query, labels), false, `${id}: ${query}`);
        assert.equal(matchesQuery(item(id), item(id).name, labels), true, id);
    }
    assert.equal(matchesQuery(item('ouroboros/pdf_tools'), 'pdf tools', labels), true, 'an id with _ is found as words');
});

test('Best match puts the card whose printed title, widget title or id is the query first, ahead of more popular partial matches', () => {
    const others = CATALOG.items.map((entry) => ({ ...entry, description: `${entry.description} Account limits, courier desk.` }));
    for (const [entry, query] of [[titled('Account limits'), 'Account limits'], [titled('Claudexor Quotas'), 'claudexor quotas'],
        [titled('Account limits'), 'claudexor quotas'], [disjoint(), 'Courier Desk']]) {
        const list = [...others, entry];
        assert.equal(sortCatalogItems(list, 'best', query)[0].name, entry.name, `${entry.title}: ${query}`);
        assert.equal(sortCatalogItems(list, 'best', query.split(' ')[0])[0].name, entry.name, `${entry.title}: a prefix`);
    }
    assert.equal(sortCatalogItems(CATALOG.items, 'best', 'pdf tools')[0].name, 'pdf_tools');
});

test('ordinary searches find and order the cards they found before', () => {
    // The board as Browse builds it (filter, then Best match), recorded on the catalog fixture before titles were searched.
    const labels = new Map(CATALOG.categories.map((cat) => [cat.id, cat.label]));
    const board = (query) => sortCatalogItems(CATALOG.items.filter((entry) => matchesQuery(entry, query, labels)), 'best', query)
        .map((entry) => entry.name);
    const before = {
        weather: ['weather'], pdf: ['pdf_tools'], slack: ['slack-bridge'], he: ['weather', 'whoop-health', 'scholion', 'chess-3d', 'telegram'],
        telegram: ['telegram'], health: ['whoop-health', 'scholion'], messaging: ['slack-bridge', 'telegram'], 'web search': ['duckduckgo'],
        pdf_tools: ['pdf_tools'], 'chess-3d': ['chess-3d'], happy: ['happy_farm'],
    };
    for (const [query, names] of Object.entries(before)) assert.deepEqual(board(query), names, query);
});

test('category counts are computed over the loaded catalog', () => {
    const counts = categoryCounts(CATALOG.items, CATALOG.categories);
    assert.equal(counts.get('messaging'), 2);
    assert.equal(counts.get('health'), 2);
    assert.equal(counts.get('planning'), 0);
});

test('the shelf offers only recommended skills the owner does not have', () => {
    const have = new Map(LISTING.map((skill) => [skill.name, skill]));
    assert.deepEqual(recommendedMissing(CATALOG.items, have).map((entry) => entry.name), ['duckduckgo', 'pdf_tools']);
    assert.deepEqual(recommendedMissing(CATALOG.items, new Map()).map((entry) => entry.name), ['telegram', 'duckduckgo', 'pdf_tools']);
    have.set('duckduckgo', { name: 'duckduckgo', lifecycle_virtual: true });
    have.set('local_pdf', { name: 'local_pdf', catalog_id: 'ouroboros/pdf_tools' });
    assert.deepEqual(recommendedMissing(CATALOG.items, have), []);
});

test('the shelf is for an install that predates the starter skills, on the unfiltered board only', () => {
    const have = new Map(LISTING.map((skill) => [skill.name, skill]));
    assert.equal(CATALOG.starter.state, 'upgrade');
    assert.deepEqual(shelfItems(CATALOG, have).map((entry) => entry.name), ['duckduckgo', 'pdf_tools']);
    for (const state of ['done', 'pending', 'running', 'fresh', '']) {
        assert.deepEqual(shelfItems({ ...CATALOG, starter: { state } }, have), [], `a ${state || 'missing'} marker shows no shelf`);
    }
    assert.deepEqual(shelfItems({ ...CATALOG, starter: {} }, have), [], 'a new install chose its ticks in the wizard');
    assert.deepEqual(shelfItems(CATALOG, null), [], 'unknown installs: nothing is offered yet');
    for (const filters of [{ query: 'pdf' }, { source: 'ouroboros' }, { category: 'search' }, { officialOnly: true }]) {
        assert.deepEqual(shelfItems(CATALOG, have, filters), [], JSON.stringify(filters));
    }
});

test('not installed: how the entry\'s own source was set is the consequence sentence, said only while that setting holds', () => {
    // The entry carries its source's trust as its catalog read it; the host reads the source again when Install is
    // pressed. The same catalog's row of that source says whether the owner confirmed the setting.
    const says = (id, sources) => text(browseCardState(item(id), { policy: installPolicy(item(id), sources) }).status);
    const confirmed = CATALOG.sources.map((row) => ({ ...row, confirmed: true }));
    assert.equal(says('ouroboros/duckduckgo', confirmed), 'Installs without review while you trust Ouroboros');
    assert.equal(says('ouroboros-community/scholion', confirmed), 'Reviewed before it runs while that is the setting for Ouroboros Community');
    // A new install has not confirmed the hub: the setting the host installs by is said, as the suggested one.
    assert.equal(CATALOG.sources.find((row) => row.id === 'ouroboros').confirmed, false);
    assert.equal(says('ouroboros/duckduckgo', CATALOG.sources), 'Installs without review while that is the suggested setting for Ouroboros');
    assert.equal(says('ouroboros-community/scholion', CATALOG.sources),
        'Reviewed before it runs while that is the suggested setting for Ouroboros Community');
    // Built in is where a skill comes from (its chip and who line); the sentence is its source's setting like any other.
    const builtin = { ...item('builtin/telegram'), trust: 'review' };
    assert.equal(text(browseCardState(builtin, { policy: installPolicy(builtin, CATALOG.sources.map((row) => ({ ...row, trust: row.id === 'builtin' ? 'review' : row.trust }))) }).status),
        'Reviewed before it runs while that is the setting for Built in');
    // Another source's row is never borrowed: without the entry's own row nothing is said of the owner's choice,
    // and without the entry's own trust nothing is promised.
    assert.equal(says('ouroboros/duckduckgo', confirmed.filter((row) => row.id !== 'ouroboros')), 'Installs without review while that is the setting for Ouroboros');
    const bare = { ...item('ouroboros/duckduckgo'), trust: undefined };
    assert.equal(text(browseCardState(bare, { policy: installPolicy(bare, confirmed) }).status), 'Its source’s setting decides the review');
    assert.match(browseCardState(item('ouroboros/duckduckgo')).action, /class="btn btn-primary btn-sm" data-browse-action="install"[^>]*>Install</);
});

test('a registry hit is another answer than the catalog: only a row that is the same source with the same setting says the owner confirmed it', () => {
    const claw = fixture('skill_catalog_search.json').items.find((entry) => entry.id === 'clawhub/steipete/weather');
    const says = (entry, sources) => text(browseCardState(entry, { policy: installPolicy(entry, sources) }).status);
    assert.deepEqual(installPolicy(claw, CATALOG.sources), { trust: 'review', name: 'ClawHub', confirmed: true });
    // The search answered after the registry was set otherwise: the catalog's row is the earlier setting, so its
    // «confirmed» is no fact of this one and «you trust» is not said.
    const since = { ...claw, trust: 'trusted' };
    assert.deepEqual(installPolicy(since, CATALOG.sources), { trust: 'trusted', name: 'ClawHub', confirmed: undefined });
    assert.equal(says(since, CATALOG.sources), 'Installs without review while that is the setting for ClawHub');
    // The id names another source in the catalog: its «suggested» is not this registry's either.
    const another = CATALOG.sources.map((row) => (row.id === 'clawhub' ? { ...row, name: 'Another registry', confirmed: false } : row));
    assert.equal(installPolicy(claw, another).confirmed, undefined);
    assert.equal(says(claw, another), 'Reviewed before it runs while that is the setting for ClawHub');
});

test('install in flight: Installing… (trusted) or Reviewing with Skip review, never a timer or percent', () => {
    const installing = browseCardState(item('ouroboros/duckduckgo'), { pending: { phase: 'installing', startedAt: Date.now() } });
    assert.match(installing.status, MOVING);
    assert.equal(tone(installing.status), 'muted', 'an install goes on by itself: a neutral sentence');
    assert.equal(text(installing.status), 'Installing…');
    assert.match(installing.action, /<button disabled/);
    const reviewing = browseCardState(item('ouroboros-community/whoop-health'), { pending: { phase: 'reviewing', startedAt: Date.now() } });
    assert.equal(text(reviewing.status), 'Reviewing · started just now');
    assert.deepEqual([tone(reviewing.status), MOVING.test(reviewing.status)], ['muted', true]);
    assert.match(reviewing.action, /data-browse-action="skip"[^>]*>.*Skip review/s);
    assert.match(reviewing.note, /you can leave this page/);
});

test('a failed install says nothing was installed and offers Try again', () => {
    const failed = browseCardState(item('ouroboros/duckduckgo'), { failure: 'GitHub didn’t answer.' });
    assert.equal(text(failed.status), 'Couldn’t install duckduckgo from Ouroboros');
    assert.match(failed.note, /GitHub didn’t answer\. Nothing was installed\./);
    assert.match(failed.action, />Try again</);
});

test('installed from this source: On, Off · Turn on, an update, needs you, modified', () => {
    const on = browseCardState(item('builtin/telegram'), { installed: installed('telegram') });
    assert.equal(text(on.status), 'On · v1.3.0');
    const update = browseCardState(item('ouroboros/weather'), { installed: installed('weather') });
    assert.equal(text(update.status), 'Installed v0.3.1 · v0.3.2 available');
    assert.match(update.action, /data-browse-action="update"/);
    const off = installed('weather');
    off.enabled = false;
    off.install.update_available = null;
    assert.equal(text(browseCardState(item('ouroboros/weather'), { installed: off }).status), 'Off · v0.3.1');
    const needs = browseCardState(item('ouroboros-community/whoop-health'), { installed: { ...installed('whoop-health'), skill_review: {} } });
    assert.equal(text(needs.status), 'Installed v0.3.1 · needs you');
    assert.equal(tone(needs.status), 'warn', 'the owner\'s move stays amber');
    const reviewing = browseCardState(item('ouroboros-community/whoop-health'), { installed: installed('whoop-health') });
    assert.equal(text(reviewing.status), 'Reviewing · you can skip');
    assert.deepEqual([tone(reviewing.status), MOVING.test(reviewing.status)], ['muted', true]);
    assert.equal(tone(on.status), 'ok');
    assert.doesNotMatch(on.status, /status-mark/, 'at rest the sentence keeps its own dot');
    const modified = browseCardState({ ...item('ouroboros/weather'), name: 'anime_studio', source_id: 'ouroboros-community' }, { installed: installed('anime_studio') });
    assert.equal(text(modified.status), 'Installed · modified here');
    const modifiedNeeds = browseCardState({ ...item('ouroboros/weather'), name: 'anime_studio', source_id: 'ouroboros-community' }, {
        installed: { ...installed('anime_studio'), skill_review: {} },
        pageModules: { faults: [{ skill: 'anime_studio', current: true, reason: 'fixture failed' }], held: [] },
    });
    assert.equal(text(modifiedNeeds.status), 'Installed v3.3.0 · needs you');
    assert.match(modifiedNeeds.action, /data-browse-action="installed"/);
});

test('one name, two Git sources: Compare…; an unrelated registry slug still offers Install', () => {
    const search = fixture('skill_catalog_search.json');
    const claw = search.items.find((entry) => entry.id === 'clawhub/steipete/weather');
    const state = browseCardState(claw, { installed: installed('weather') });
    assert.match(state.action, /data-browse-action="install"[^>]*>Install</);
    // The listing names no id for a copy no connected source matches (its id now reads another address or view).
    const elsewhere = installed('weather');
    delete elsewhere.install.source.id;
    const moved = browseCardState(item('ouroboros/weather'), { installed: elsewhere });
    assert.equal(text(moved.status), 'You have another “weather” (from Ouroboros)');
    assert.match(moved.action, /data-browse-action="compare"/);
    // Its view connected again under another id: the host names that source, so its card offers the update.
    const reconnected = installed('weather');
    reconnected.install.source = { id: 'shared', name: 'Ouroboros', kind: 'hub' };
    const shared = browseCardState({ ...item('ouroboros/weather'), id: 'shared/weather', source_id: 'shared' }, { installed: reconnected });
    assert.equal(text(shared.status), 'Installed v0.3.1 · v0.3.2 available');
    assert.match(shared.action, /data-browse-action="update"/);
    assert.match(browseCardState(item('ouroboros/weather'), { installed: reconnected }).action, /data-browse-action="compare"/);
    for (const catalog_id of [null, 'mirror/weather']) {
        assert.match(browseCardState(item('ouroboros/weather'), { installed: { ...installed('weather'), catalog_id } }).action,
            /data-browse-action="compare"/, 'a present field cannot fall back to the old source and name');
    }
    const confirmed = { ...installed('weather'), catalog_id: 'shared/weather' };
    assert.match(browseCardState({ ...item('ouroboros/weather'), id: 'shared/weather', source_id: 'shared' }, { installed: confirmed }).action,
        /data-browse-action="update"/, 'the confirmed entry can move without changing the recorded install source');
    const unrecorded = browseCardState({ ...item('ouroboros/weather'), name: 'zoom_meeting' }, { installed: installed('zoom_meeting') });
    assert.doesNotMatch(text(unrecorded.status), /You have another/, 'a copy without a source record is not another');
    assert.match(text(browseCardState(item('ouroboros-community/old_tool'), {}).status), /^Withdrawn by Ouroboros Community$/);
});

test('an omitted source key is legacy; a local alias needs both the exact source key and qualified catalog address', () => {
    const entry = { ...item('ouroboros/weather'), source_key: 'git-a' };
    const row = { ...installed('weather'), catalog_id: entry.id };
    for (const source_key of [null, '', 'git-b', undefined]) {
        assert.doesNotMatch(browseCardState(entry, { installed: { ...row, source_key } }).action, /data-browse-action="update"/);
    }
    assert.doesNotMatch(browseCardState(entry, { installed: row }).action, /data-browse-action="update"/);
    assert.doesNotMatch(browseCardState(entry, { installed: { ...row, source_key: 'git-a', name: 'other' } }).action, /data-browse-action="update"/);
    assert.match(browseCardState(entry, { installed: { ...row, source_key: 'git-a', catalog_id: null } }).action, /data-browse-action="update"/);
    const local = { ...row, name: 'folder_weather', source: 'user_repo', source_key: 'local-a' };
    const offer = { ...entry, source_key: 'local-a' };
    assert.match(browseCardState(offer, { installed: local }).action, /data-browse-action="update"/);
    assert.doesNotMatch(browseCardState(offer, { installed: { ...local, catalog_id: 'other/weather' } }).action, /data-browse-action="update"/);
    assert.doesNotMatch(browseCardState(offer, { installed: { ...local, source_key: 'local-b' } }).action, /data-browse-action="update"/);
});

test('Browse offers the current update after rejection, including a newer release and a modified restored copy', () => {
    for (const row of fixture('extensions_fix_r2.json').skills) {
        for (const modified of [false, true]) {
            row.install.modified = modified;
            const entry = { ...item('ouroboros/weather'), id: `${row.install.source.id}/${row.name}`, name: row.name,
                source_id: row.install.source.id, digest: row.install.update_available?.digest || '' };
            const state = browseCardState(entry, { installed: row });
            if (row.install.update_available) {
                assert.match(state.action, /data-browse-action="update"/);
                assert.doesNotMatch(state.action, /Open in Installed/);
                assert.ok(text(state.status).includes(`v${row.install.update_available.version} available`));
            } else {
                assert.match(state.action, modified && !/needs you/.test(text(state.status)) ? />Details</ : /Open in Installed/);
                assert.doesNotMatch(state.action, /data-browse-action="update"/);
            }
        }
    }
    const row = fixture('extensions_fix_r2.json').skills[0];
    row.install.update_available = null;
    assert.match(browseCardState({ ...item('ouroboros/weather'), id: `ouroboros/${row.name}`, name: row.name }, { installed: row }).action, /Open in Installed/);
});

test('before the installed listing is read, the catalog\'s own join says installed or taken, with no action', () => {
    const early = browseCardState(item('ouroboros/weather'), { listed: false });
    assert.equal(text(early.status), 'Installed · v0.3.1');
    assert.equal(early.action, '', 'every action needs the listing row');
    const taken = { ...item('ouroboros/duckduckgo'), installed: { state: 'name_taken', source_id: 'my-folder', source_name: 'My skills' } };
    assert.equal(text(browseCardState(taken, { listed: false }).status), 'You have another “duckduckgo” (from My skills)');
    assert.match(browseCardState(item('ouroboros/duckduckgo'), { listed: false }).action, />Install</);
    assert.equal(text(browseCardState(item('ouroboros/weather'), { listed: true }).status), 'Installs without review while that is the setting for Ouroboros',
        'once read, the listing decides (here: not installed)');
});

test('a card: name opens the sheet, Official only on endorsed entries, unknown popularity is said, categories under All', () => {
    const labels = new Map(CATALOG.categories.map((cat) => [cat.id, cat.label]));
    const weather = renderBrowseCard(item('ouroboros/weather'), { state: { status: 'S', action: 'A' }, categoryLabels: labels });
    assert.match(weather, /data-browse-action="open" data-item="ouroboros\/weather"><span data-i18n-authored>weather/);
    assert.match(weather, /data-official-mark/);
    assert.doesNotMatch(weather, /data-official-source/, 'the mark is explained from this entry, never a source looked up by id');
    assert.match(weather, />▲ 25 votes</);
    assert.match(weather, /<span class="ui-chip skills-chip" data-tone="neutral">City &amp; services<\/span>/);
    const hidden = renderBrowseCard(item('ouroboros/weather'), { state: { status: '', action: '' }, categoryLabels: labels, showCategory: false });
    assert.doesNotMatch(hidden, /City &amp; services/, 'inside a category the chip would repeat the filter');
    const community = renderBrowseCard(item('ouroboros-community/whoop-health'), { state: { status: '', action: '' } });
    assert.doesNotMatch(community, /data-official-mark/);
    assert.match(community, /▲ 20 votes · needs 2 keys/);
    assert.match(community, /Ouroboros Community · by kuznetsoffandrey · updated [^<·]+</);
    assert.doesNotMatch(renderBrowseCard(item('ouroboros-community/happy_farm'), { state: { status: '', action: '' } }), /updated/,
        'an entry without a date says none');
    assert.match(renderBrowseCard(item('ouroboros-community/old_tool'), { state: { status: '', action: '' } }), />No data</);
    assert.match(renderBrowseCard(item('ouroboros-community/happy_farm'), { state: { status: '', action: '' } }), />No votes yet</);
    const builtin = renderBrowseCard(item('builtin/telegram'), { state: { status: '', action: '' } });
    assert.match(builtin, /Built in<\/span>/);
    assert.equal((builtin.match(/data-official-mark/g) || []).length, 1);
    assert.match(builtin, /Comes with the app · updates with Ouroboros/);
    const description = `${'A practical description for rainy days. '.repeat(17)}END`;
    const long = renderBrowseCard({ ...item('ouroboros/weather'), description }, { state: { status: '', action: '' } });
    assert.match(long, /data-browse-purpose-preview/);
    assert.doesNotMatch(long, /data-fold-clipped/, 'layout, not text length, decides whether content is clipped');
    assert.ok(long.includes(`${description}</p>`), 'the card markup keeps the catalog text');
    assert.match(long, />Read more ›</);
    assert.doesNotMatch(weather, /data-browse-purpose-preview|data-fold-clipped|Read more/);
});

test('a connector is known by its place declaration, never a typed tag: icon, «Connects …», the Connectors filter', () => {
    const slack = { ...item('ouroboros/duckduckgo'), connects: ['Slack'], icon: { name: 'chat' } };
    assert.equal(isConnector(slack), true);
    assert.equal(isConnector(item('ouroboros/duckduckgo')), false);
    assert.equal(isConnector({ category: [CONNECTORS] }), true, 'a catalog category of the same id joins it');
    const card = renderBrowseCard(slack, { state: browseCardState(slack) });
    assert.match(card, /<div class="skills-head-title"><span class="skills-card-icon" aria-hidden="true"><svg/);
    assert.match(card, /<div class="skills-card-cats"><span class="ui-chip skills-chip" data-tone="neutral">Connects Slack<\/span>/);
    const plain = renderBrowseCard(item('ouroboros/duckduckgo'), { state: browseCardState(item('ouroboros/duckduckgo')) });
    assert.doesNotMatch(plain, /skills-card-icon|Connects /);
});

test('a skill that writes into your chat without a place says its messages count as yours', () => {
    const bridge = { ...item('ouroboros/duckduckgo'), permissions: ['net', 'inject_chat'] };
    assert.match(renderBrowseCard(bridge, { state: browseCardState(bridge) }), /<div class="skills-card-facts">Messages from here count as yours<\/div>/);
    const place = { ...bridge, connects: ['A2A'] };
    assert.doesNotMatch(renderBrowseCard(place, { state: browseCardState(place) }), /count as yours/);
});

test('Browse uses the optional title or connector label while retaining its canonical id', () => {
    const base = item('ouroboros/weather');
    const connector = { connects: ['Mail'], contributes: [{ provides: 'place', label: 'Mailbox' }] };
    for (const [extra, title] of [[{ ...connector, title: 'Email tools' }, 'Email tools'], [connector, 'Mailbox'], [{ connects: ['Mail'] }, 'weather'], [{}, 'weather']]) {
        const candidate = { ...base, ...extra };
        const html = renderBrowseCard(candidate, { state: browseCardState(candidate) });
        assert.ok(html.includes(`<span data-i18n-authored>${title}</span>`));
        assert.match(html, /skills-card-who"><span data-i18n-authored>weather<\/span>/);
        assert.match(html, /data-item="ouroboros\/weather"/);
    }
});

function browseDom() {
    const { doc, el } = domDouble({ parseHtml: true });
    const pane = el(), chrome = el(), scroller = el();
    scroller.className = 'skills-scroll';
    scroller.append(pane);
    doc.body.append(chrome, scroller);
    class Observer { observe() {} disconnect() {} }
    const globals = { document: doc, CSS: { escape: String }, ResizeObserver: Observer, MutationObserver: Observer,
        requestAnimationFrame: (fn) => fn(), localStorage: { getItem: () => null } };
    const previous = Object.fromEntries(Object.keys(globals).map((key) => [key, globalThis[key]]));
    Object.assign(globalThis, globals);
    return { doc, pane, chrome, scroller, restore: () => Object.assign(globalThis, previous) };
}

test('Compare → Replace confirms the same installed payload across fresh views', async (t) => {
    const { pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const source = { id: 'my-git', name: 'My git', kind: 'git', trust: 'review' };
    const offer = { ...item('ouroboros/weather'), id: 'my-git/weather', source_id: source.id, source_name: source.name,
        source_key: 'git-offer', digest: 'd'.repeat(64), version: '0.1.0' };
    const legacy = installed('weather'); // the legacy listing omits source_key but names the real folder and landing
    const modern = { ...legacy, source_key: 'hub-installed' };
    const manual = { ...legacy, source: 'external', payload_root: 'skills/external/weather', catalog_id: null,
        source_key: null, install: null, removal: 'delete' };
    const selfAuthored = { ...manual, source: 'self_authored' };
    let row, shownOffer = offer, confirm;
    const installs = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => new Map([[row.name, structuredClone(row)]]), sourceSnapshot: () => ({ rows: [source] }),
        sources: async () => [source], catalog: async () => ({ items: [shownOffer], sources: [source] }),
        compare: () => new Promise((resolve) => { confirm = resolve; }), refreshInstalled: async () => {}, toast: () => {},
    }, api: { install: async (...args) => { installs.push(args); return { already_installed: true }; } } });
    const begin = async (initial) => {
        row = initial;
        confirm = null;
        await browse.show();
        pane.querySelector('[data-browse-action="compare"]').click();
        assert.equal(typeof confirm, 'function', 'opened Compare for the selected offer');
    };
    const replace = async () => {
        confirm('replace');
        await new Promise((resolve) => setImmediate(resolve));
    };
    try {
        for (const initial of [legacy, modern, manual, selfAuthored]) {
            const before = installs.length;
            await begin(initial);
            row = { ...initial, enabled: false, lifecycle_status: 'failed', catalog_title: 'A fresh title',
                install: initial.install && { ...initial.install, update_available: null } };
            await replace();
            assert.deepEqual(installs.slice(before), [[offer.id, { turnOnWhenReady: true, replace: true,
                expectedSourceKey: offer.source_key, expectedDigest: offer.digest }]]);
        }
        const confirmedCount = installs.length;
        for (const [initial, drift] of [[modern, { content_hash: 'e'.repeat(64) }],
            [modern, { payload_root: 'skills/external/weather' }], [modern, { source_key: 'another-hub' }],
            [modern, { identity_collision: true }], [manual, { content_hash: 'e'.repeat(64) }],
            [manual, { payload_root: 'skills/external/moved-weather' }]]) {
            await begin(initial);
            row = { ...initial, ...drift }; // same name, a fresh reading rather than a mutation of the shown object
            await replace();
            assert.equal(installs.length, confirmedCount, `must not replace ${initial.source} after drift in ${Object.keys(drift).join()}`);
        }
        for (const unconfirmed of [{ ...manual, install: legacy.install, removal: 'uninstall' }, { ...manual, removal: '' },
            { ...manual, source: 'user_repo', payload_root: 'skills/user_repo/weather', removal: '' }]) {
            await begin(unconfirmed);
            await replace();
            assert.equal(installs.length, confirmedCount, 'a null source key alone does not confirm a manual external folder');
        }
        await begin(modern);
        shownOffer = { ...offer, digest: 'f'.repeat(64) };
        await browse.show();
        await replace();
        assert.equal(installs.length, confirmedCount, 'the confirmed offer did not survive the catalog refresh');
    } finally { browse.destroy(); restore(); }
});

test('a registry hit drawn from its search answer says that answer\'s setting, and nothing of a confirmation the catalog recorded for another', async (t) => {
    const { pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    // The catalog was read while the registry reviewed each skill; the search answers after it was set otherwise.
    const registry = { id: 'clawhub', name: 'ClawHub', kind: 'registry', trust: 'review', confirmed: true };
    const hit = { id: 'clawhub/acme/notes', name: 'notes', registry_slug: 'acme/notes', description: 'Notes.',
        source_id: registry.id, source_name: registry.name, trust: 'trusted' };
    const asked = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => new Map(), sources: async () => {}, sourceSnapshot: () => ({ rows: [registry] }),
        catalog: async () => ({ items: [], sources: [registry] }),
    }, api: { search: async (query) => { asked.push(query); return { items: [hit] }; } } });
    try {
        await browse.show();
        await browse.searchFor('notes');
        await new Promise((resolve) => setImmediate(resolve));
        assert.deepEqual(asked, [{ q: 'notes', source: '' }]);
        assert.equal(pane.querySelector('.marketplace-card .skills-status-text').textContent,
            'Installs without review while that is the setting for ClawHub', 'its own setting; «you trust» is not said of it');
    } finally { browse.destroy(); restore(); }
});

test('qualified Browse names and registry actions use the exact slug, known origin and actual local name', async (t) => {
    const { doc, pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const source = { id: 'clawhub', name: 'ClawHub', kind: 'registry', url: 'https://registry.test' };
    const hit = { id: 'clawhub/acme/notes', name: 'notes', registry_slug: 'acme/notes', source_id: source.id, source_key: 'registry-a', version: '1.0.0' };
    const other = { ...hit, id: 'clawhub/other/notes', registry_slug: 'other/notes', registry_stats: { stars: 100 } };
    for (const query of [hit.id, hit.registry_slug]) {
        assert.equal(matchesQuery(hit, query), true);
        assert.equal(matchesQuery(other, query), false);
        assert.equal(sortCatalogItems([other, hit], 'best', query)[0], hit);
    }
    const own = { name: 'acme__notes', version: '1.0.0', review_status: 'clean', source_key: 'registry-a',
        install: { source }, registry_slug: 'acme/notes' };
    const occupant = { ...own, name: 'notes', registry_slug: 'someone/notes', provenance: { slug: 'acme/notes' } };
    const elsewhere = { ...own, name: 'imported_notes', source_key: 'registry-b' };
    const listing = new Map([[occupant.name, occupant], [elsewhere.name, elsewhere], [own.name, own]]);
    let answer = hit;
    const calls = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => listing, sources: async () => {}, sourceSnapshot: () => ({ rows: [source] }),
        refreshInstalled: async () => {},
        catalog: async () => ({ items: [], sources: [source] }),
        runInstalledAction: (name, action) => calls.push([name, action]),
        compare: (_item, row) => { calls.push(['compare', row.name]); return 'keep'; },
    }, api: { search: async () => ({ items: [answer] }), install: async (...args) => { calls.push(['install', ...args]); return { basis: 'review' }; } } });
    try {
        await browse.show();
        await browse.searchFor(hit.id);
        pane.querySelector('[data-browse-action="turn_on"]').click();
        pane.querySelector('.skills-card-name').click();
        doc.body.querySelector('[data-sheet-action="turn_on"]').click();
        assert.deepEqual(calls, [['acme__notes', 'turn_on'], ['acme__notes', 'turn_on']]);

        own.source_key = 'registry-b';
        doc.body.querySelector('[data-sheet-action="turn_on"]').click();
        assert.equal(calls.length, 2, 'a stale control cannot act on a copy from another origin');
        await browse.show();
        await browse.searchFor(hit.id);
        assert.equal(pane.querySelector('[data-browse-action="compare"]'), null);
        pane.querySelector('[data-browse-action="install"]').click();
        await new Promise((resolve) => setImmediate(resolve));
        assert.deepEqual(calls.at(-1), ['install', hit.id, { turnOnWhenReady: true, replace: false, expectedSourceKey: 'registry-a', version: '1.0.0' }]);
        answer = { ...hit, source_key: 'registry-b' };
        listing.delete(own.name);
        await browse.searchFor(hit.id);
        assert.equal(doc.body.querySelector('[data-sheet-action]'), null, 'the open sheet cannot offer actions for the repointed source');
    } finally { browse.destroy(); restore(); }
});

test('cards and the shelf say how each entry installs while its source\'s setting holds; the list of sources changes none of them', async (t) => {
    const { pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const source = { id: 'ouroboros', name: 'Ouroboros', kind: 'hub', part: 'official', trust: 'review', confirmed: false };
    const entry = (name, recommended = false) => ({ id: `ouroboros/${name}`, name, source_id: source.id, source_name: source.name,
        trust: source.trust, recommended });
    const catalog = { starter: { state: 'upgrade' }, sources: [source], items: [entry('tiles', true), entry('weather')] };
    // The list of sources names the filters. Here it says something else under the same id, then cannot be read.
    let facts = { rows: [{ ...source, trust: 'trusted', confirmed: true }], error: null };
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => new Map(), sources: async () => {}, sourceSnapshot: () => facts, catalog: async () => catalog,
    } });
    const says = () => pane.querySelectorAll('.marketplace-card .skills-status-text').map((node) => node.textContent);
    try {
        await browse.show();
        const suggested = 'Reviewed before it runs while that is the suggested setting for Ouroboros';
        assert.deepEqual(says(), [suggested, suggested], 'one recommended entry and one on the board');
        assert.equal(pane.querySelector('.skills-group-title p').textContent, `Picked by Ouroboros · ${suggested}`);
        assert.equal(chrome.querySelector('[data-browse-source]').textContent, 'All sourcesOuroboros (2)');
        assert.equal(browse.repaintSourceFacts, undefined, 'a source read has nothing of Browse to repaint');
        assert.deepEqual(browse.officialSource('ouroboros/tiles'), { trust: 'review', name: 'Ouroboros', confirmed: false, url: undefined });
        assert.equal(browse.officialSource('removed/tiles'), null, 'an entry that is not shown says nothing of a source');
        const cards = pane.querySelectorAll('.marketplace-card');
        facts = { ...facts, error: new Error('display fixture: read unavailable') };
        browse.repaintStates();  // the installed listing moved; the failed list of sources is no fact of a card
        assert.deepEqual(says(), [suggested, suggested]);
        assert.deepEqual(pane.querySelectorAll('.marketplace-card'), cards);
    } finally { browse.destroy(); restore(); }
});

test('a confirmed catalog change keeps the old sheet’s facts and withdraws its action, while the fresh card acts on the fresh offer', async (t) => {
    const { doc, pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const source = { id: 'ouroboros', name: 'Ouroboros', kind: 'hub', part: 'official', trust: 'review', confirmed: true };
    const entry = (version, trust) => ({ ...item('ouroboros/weather'), version, source_id: source.id, source_name: source.name, trust });
    let catalog = { sources: [source], items: [entry('1.0.0', 'review')] };
    // The name is taken by a copy from a source that is not connected now: the card and the sheet offer Compare….
    const here = installed('weather');
    const elsewhere = { ...here, install: { ...here.install, source: { name: 'Elsewhere', kind: 'git' } } };
    let listing = new Map([[elsewhere.name, elsewhere]]);
    const compared = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => listing, sources: async () => {}, sourceSnapshot: () => ({ rows: [source] }), catalog: async () => catalog,
        compare: async (shown) => { compared.push(shown.version); return 'keep'; },
    }, api: { item: () => new Promise(() => {}) } });
    try {
        await browse.show();
        assert.equal(pane.querySelector('.marketplace-card .skills-status-text').textContent, 'You have another “weather” (from Elsewhere)');
        pane.querySelector('.skills-card-name').click();
        const sheet = doc.body.querySelector('.skills-detail-sheet');
        const review = () => sheet.querySelectorAll('.skills-sheet-section')
            .find((section) => section.querySelector('h3')?.textContent === 'Review').querySelector('p').textContent;
        const opened = 'Ouroboros reviews it on this computer before it runs while that is the setting for the Ouroboros source. You can skip the review.';
        assert.equal(review(), opened, 'this entry, never the review of the other installed copy');
        const compare = sheet.querySelector('[data-sheet-action="compare"]');
        compare.focus();
        browse.repaintStates();  // the installed listing was read again and says the same
        assert.equal(sheet.querySelector('[data-sheet-action="compare"]'), compare, 'an unchanged action keeps its button');
        assert.equal(doc.activeElement, compare, 'and the keyboard its place');

        // The catalog is read again: the same id now carries another version, from a source set otherwise.
        catalog = { sources: [{ ...source, trust: 'trusted' }], items: [entry('2.0.0', 'trusted')] };
        await browse.show();
        browse.repaintStates();
        assert.match(sheet.querySelector('.document-reader-meta').textContent, /v1\.0\.0/);
        assert.equal(review(), opened, 'an id brings the open sheet no other entry\'s setting');
        assert.equal(sheet.querySelector('[data-sheet-action="compare"]'), null);
        assert.equal(doc.activeElement, sheet.querySelector('[data-sheet-body]'));
        assert.match(sheet.querySelector('[role="status"]').textContent, /offer changed.+refresh Browse/);
        compare.click();
        pane.querySelector('[data-browse-action="compare"]').click();
        await Promise.resolve();
        assert.deepEqual(compared, ['2.0.0'], 'the stale action is gone; only the fresh card offers a comparison');

        // Even a later installed-list read cannot re-enable the stale sheet.
        listing = new Map([[here.name, here]]);
        browse.repaintStates();
        assert.equal(sheet.querySelector('[data-sheet-action]'), null);
        assert.equal(review(), opened);
    } finally { browse.destroy(); restore(); }
});

test('Browse binds a completed detail read and an Install click to the drawn offer', async (t) => {
    const { doc, pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const source = { id: 'hub', name: 'Example hub', kind: 'hub', trust: 'review' };
    const offer = { ...item('ouroboros/weather'), id: 'hub/weather', source_id: 'hub', source_name: source.name,
        title: 'Weather for you', description: 'Description on the card.', source_key: 'origin-a', version: '1.0.0', digest: 'a'.repeat(64) };
    let resolveDetail;
    const installs = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => new Map(), sources: async () => [source], sourceSnapshot: () => ({ rows: [source] }),
        catalog: async () => ({ items: [offer], sources: [source] }), refreshInstalled: async () => {}, toast: () => {},
    }, api: {
        item: () => new Promise((resolve) => { resolveDetail = resolve; }),
        install: async (...args) => { installs.push(args); return { already_installed: true }; },
    } });
    try {
        await browse.show();
        const freshSheet = () => {
            pane.querySelector('.skills-card-name').click();
            return doc.body.querySelector('.skills-detail-sheet');
        };
        let sheet = freshSheet();
        assert.match(sheet.querySelector('.document-reader-title').textContent, /Weather for you/);
        const button = sheet.querySelector('[data-sheet-action="install"]');
        resolveDetail({ item: offer, versions: [], about: 'Full description of this release.' });
        await new Promise((resolve) => setImmediate(resolve));
        assert.match(sheet.querySelector('[data-sheet-content]').textContent, /Full description of this release/);
        button.click();
        await new Promise((resolve) => setImmediate(resolve));
        assert.deepEqual(installs, [[offer.id, { turnOnWhenReady: true, replace: false, expectedSourceKey: 'origin-a', expectedDigest: offer.digest }]]);
        sheet.querySelector('[data-sheet-close]').click();
        for (const mismatch of [{ id: 'hub/other' }, { source_key: 'origin-b' }, { version: '2.0.0' }, { digest: 'b'.repeat(64) }]) {
            sheet = freshSheet();
            resolveDetail({ item: { ...offer, ...mismatch }, about: 'ABOUT FROM A DIFFERENT OFFER',
                uses: [{ kind: 'key', id: 'DIFFERENT_KEY' }], versions: [{ version: '2.0.0', current: true }] });
            await new Promise((resolve) => setImmediate(resolve));
            assert.match(sheet.querySelector('[data-sheet-content]').textContent, /offer changed.+refresh Browse/);
            assert.match(sheet.querySelector('[data-sheet-content]').textContent, /Description on the card/);
            assert.doesNotMatch(sheet.textContent, /DIFFERENT|v2\.0\.0/);
            assert.match(sheet.querySelector('.document-reader-meta').textContent, /v1\.0\.0/);
            browse.repaintStates();
            assert.equal(sheet.querySelector('[data-sheet-action]'), null);
            sheet.querySelector('[data-sheet-close]').click();
        }
        assert.equal(installs.length, 1);
    } finally { browse.destroy(); restore(); }
});

test('only a successful source-list read can withdraw a shown offer; identical files of unknown origin only open Installed', async (t) => {
    const { doc, pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const source = { id: 'hub', name: 'Example hub', kind: 'hub' };
    const offer = { ...item('ouroboros/weather'), id: 'hub/weather', source_id: 'hub', source_key: 'origin-a', digest: 'a'.repeat(64) };
    let sources = { rows: [source] };
    let listing = new Map();
    const jumps = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => listing, sources: async () => sources.rows, sourceSnapshot: () => sources,
        catalog: async () => ({ items: [offer], sources: [source] }), showInstalled: (name) => jumps.push(name),
    }, api: { item: () => new Promise(() => {}) } });
    try {
        await browse.show();
        pane.querySelector('.skills-card-name').click();
        const sheet = doc.body.querySelector('.skills-detail-sheet');
        const button = sheet.querySelector('[data-sheet-action="install"]');
        assert.ok(button);
        sources = { rows: [], error: new Error('not a successful read') };
        browse.repaintSheet();
        assert.equal(sheet.querySelector('[data-sheet-action="install"]'), button);
        sources = { rows: [] };
        browse.repaintSheet();
        assert.match(sheet.textContent, /offer changed.+refresh Browse/);
        assert.equal(sheet.querySelector('[data-sheet-action]'), null);
        sheet.querySelector('[data-sheet-close]').click();
        const installed = { name: 'weather', source_key: null, content_hash: offer.digest };
        listing = new Map([[installed.name, installed]]);
        browse.repaintStates();
        assert.equal(pane.querySelector('[data-browse-action="compare"]'), null);
        pane.querySelector('[data-browse-action="installed"]').click();
        assert.deepEqual(jumps, ['weather']);
    } finally { browse.destroy(); restore(); }
});

test('an older cached copy and a newer search answer of one id: a card opens and reports the row it shows', async (t) => {
    const { doc, pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const hub = { id: 'hub', name: 'Example hub', kind: 'hub', trust: 'review', confirmed: true };
    const registry = { id: 'clawhub', name: 'ClawHub', kind: 'registry', trust: 'review', confirmed: true };
    const copy = (title, version, repo_url) => ({ id: 'hub/notes', name: 'notes', title, version, repo_url, description: 'Keeps things.',
        source_id: hub.id, source_name: hub.name, trust: hub.trust });
    // The cached catalog still has the old title; the search read the refreshed source.
    const cached = copy('Old notes', '1.0.0', 'https://example.test/old');
    const fresh = copy('Fresh letters', '2.0.0', 'https://example.test/new');
    const read = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => new Map(), sources: async () => {}, sourceSnapshot: () => ({ rows: [hub, registry] }),
        catalog: async () => ({ items: [cached], sources: [hub, registry] }),
    }, api: { search: async () => ({ items: [fresh] }), item: (source, name) => { read.push(`${source}/${name}`); return new Promise(() => {}); } } });
    const shown = async (query) => {
        await browse.searchFor(query);
        await new Promise((resolve) => setImmediate(resolve));
        const cards = pane.querySelectorAll('.marketplace-card');
        assert.equal(cards.length, 1, query);
        browse.repaintStates();
        pane.querySelector('.skills-card-name').click();
        const sheet = doc.body.querySelector('.skills-detail-sheet');
        const facts = [pane.querySelector('.skills-card-name').textContent.replace('›', ''),
            sheet.querySelector('.document-reader-title').textContent,
            /v\d\.0\.0/.exec(sheet.querySelector('.document-reader-meta').textContent)?.[0],
            browse.officialSource('hub/notes').url];
        sheet.querySelector('[data-sheet-close]').click();
        return facts;
    };
    try {
        await browse.show();
        assert.deepEqual(await shown('Fresh letters'), ['Fresh letters', 'Fresh letters', 'v2.0.0', 'https://example.test/new'],
            'only the search answer matches: its row is drawn and opened');
        assert.deepEqual(await shown('notes'), ['Old notes', 'Old notes', 'v1.0.0', 'https://example.test/old'],
            'both match: the cached row is drawn and opened');
        assert.deepEqual(read, ['hub/notes', 'hub/notes']);
    } finally { browse.destroy(); restore(); }
});

test('Browse redraw keeps each detail opener; Widgets leaves a sheet through the same action with its opener', async () => {
    const { doc, pane, chrome, restore } = browseDom();
    const source = { id: 'fixture', kind: 'registry' };
    const entry = { id: 'fixture/widget', name: 'widget', source_id: source.id, description: 'Long description. '.repeat(40) };
    const ready = { name: entry.name, version: '1.0.0', type: 'extension', enabled: true, live_loaded: true,
        review_status: 'clean' };
    let live;
    const jumps = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => new Map([[ready.name, ready]]), live: () => live,
        sources: async () => [source], sourceSnapshot: () => ({ rows: [source] }), catalog: async () => ({ items: [entry] }),
        runInstalledAction(name, action, options) {
            jumps.push({ name, action, options, modal: doc.body.querySelector('.skills-detail-sheet') });
        },
    } });
    try {
        await browse.show();
        for (const selector of ['.skills-card-name', '.skills-read-more', '.skills-card-action-row [data-browse-action="open"]']) {
            const before = pane.querySelector(selector);
            assert.ok(before, selector);
            before.focus();
            await browse.show();
            assert.equal(before.isConnected, false, 'a real board rebuild took place');
            assert.equal(doc.activeElement, pane.querySelector(selector), selector);
        }

        live = { ui_tabs: [{ skill: ready.name }] };
        await browse.show();
        pane.querySelector('[data-browse-action="widgets"]').click();
        assert.deepEqual(jumps[0], { name: 'widget', action: 'open_widgets', options: undefined, modal: null });

        const opener = pane.querySelector('.skills-read-more');
        opener.focus();
        opener.click();
        await browse.show();  // the modal remains up, but its opening button was replaced
        assert.equal(opener.isConnected, false);
        const current = pane.querySelector('.skills-read-more');
        doc.body.querySelector('[data-sheet-action="widgets"]').click();
        assert.deepEqual(jumps[1], { name: 'widget', action: 'open_widgets', options: { opener: current }, modal: null });
        assert.equal(doc.activeElement, current);
    } finally {
        browse.destroy();
        restore();
    }
});

test('a refresh drops the registry answer it no longer describes, and one still on its way; the query stays', async (t) => {
    const { pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const registry = { id: 'clawhub', name: 'ClawHub', kind: 'registry', trust: 'review', confirmed: true };
    const hit = (slug) => ({ id: `clawhub/${slug}`, name: slug, registry_slug: slug, source_id: 'clawhub', source_name: 'ClawHub',
        trust: 'review', description: 'Notes from the registry.', digest: '', version: '1.0.0' });
    const answers = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => new Map(), sources: async () => {}, sourceSnapshot: () => ({ rows: [registry] }),
        catalog: async () => ({ items: [], sources: [registry] }),
    }, api: { search: () => new Promise((resolve) => answers.push(resolve)), item: () => new Promise(() => {}) } });
    const settle = () => new Promise((resolve) => setImmediate(resolve));
    try {
        await browse.show();
        await browse.searchFor('notes');
        answers.shift()({ items: [hit('notes')] });
        await settle();
        assert.equal(pane.querySelectorAll('.marketplace-card').length, 1, 'the registry answered');
        browse.searchFor('notes');  // asked again; this answer is still on its way when the refresh starts
        await settle();
        await browse.refresh();
        answers.shift()?.({ items: [hit('stale')] });
        await settle();
        assert.equal(pane.querySelectorAll('.marketplace-card').length, 0, 'neither the old answer nor the late one stays');
        assert.equal(chrome.querySelector('[data-browse-search]').value, 'notes');
        assert.match(pane.querySelector('[data-browse-status]').textContent, /press Enter to search ClawHub too/);
    } finally { browse.destroy(); restore(); }
});

test('a legacy registry copy of unknown origin tied by its source row and full slug only opens Installed', async (t) => {
    const { pane, chrome, restore } = browseDom();
    t.mock.method(globalThis, 'fetch', () => assert.fail('This display test has no network'));
    const registry = { id: 'clawhub', name: 'ClawHub', kind: 'registry', trust: 'review', confirmed: true };
    const hit = { id: 'clawhub/acme/notes', name: 'notes', registry_slug: 'acme/notes', source_id: 'clawhub', source_name: 'ClawHub',
        source_key: 'registry:k', trust: 'review', description: 'Notes from the registry.', digest: '', version: '2.0.0' };
    const legacy = { name: 'acme__notes', source_key: null, registry_slug: 'acme/notes', install: { source: { id: 'clawhub', kind: 'registry' } } };
    let listing = new Map([[legacy.name, legacy]]);
    const jumps = [];
    const installs = [];
    const browse = createSkillBrowse({ pane, chrome, host: {
        installedByName: () => listing, sources: async () => {}, sourceSnapshot: () => ({ rows: [registry] }),
        catalog: async () => ({ items: [], sources: [registry] }), showInstalled: (name) => jumps.push(name),
    }, api: { search: async () => ({ items: [hit] }), install: async (id) => { installs.push(id); return {}; }, item: () => new Promise(() => {}) } });
    try {
        await browse.show();
        await browse.searchFor('acme/notes');
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(pane.querySelector('[data-browse-action="install"]'), null);
        assert.equal(pane.querySelector('[data-browse-action="compare"]'), null);
        assert.match(pane.querySelector('.marketplace-card').textContent, /Installed · source unknown/);
        pane.querySelector('[data-browse-action="installed"]').click();
        assert.deepEqual(jumps, ['acme__notes']);
        assert.deepEqual(installs, []);
        listing = new Map([[legacy.name, { ...legacy, install: { source: { id: 'other-registry', kind: 'registry' } } }]]);
        browse.repaintStates();
        assert.equal(pane.querySelector('[data-browse-action="installed"]'), null, 'another source row is no tie');
    } finally { browse.destroy(); restore(); }
});
