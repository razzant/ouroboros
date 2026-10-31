// Skills → Browse: sort, filter, category counts, the recommended shelf and the
// consequence row each card shows before its one action is pressed.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
    CONNECTORS, browseCardState, categoryCounts, isConnector, matchesQuery, recommendedMissing, renderBrowseCard, shelfItems,
    sortCatalogItems,
} from '../modules/skill_browse.js';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/skills_v4/${name}`, import.meta.url), 'utf8'));
const CATALOG = fixture('skill_catalog.json');
const SOURCES = new Map(fixture('skill_sources.json').sources.map((source) => [source.id, source]));
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

test('not installed: the trust of the source is the consequence sentence', () => {
    const trusted = browseCardState(item('ouroboros/duckduckgo'), { source: SOURCES.get('ouroboros') });
    assert.equal(text(trusted.status), 'Installs without review — you trust Ouroboros');
    assert.match(trusted.action, /class="btn btn-primary btn-sm" data-browse-action="install"[^>]*>Install</);
    const reviewed = browseCardState(item('ouroboros-community/scholion'), { source: SOURCES.get('ouroboros-community') });
    assert.equal(text(reviewed.status), 'Ouroboros reviews it before it runs');
});

test('install in flight: Installing… (trusted) or Reviewing with Skip review, never a timer or percent', () => {
    const installing = browseCardState(item('ouroboros/duckduckgo'), { source: SOURCES.get('ouroboros'), pending: { phase: 'installing', startedAt: Date.now() } });
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
});

test('one name, two sources: the card offers Compare… instead of a silent replace', () => {
    const search = fixture('skill_catalog_search.json');
    const claw = search.items.find((entry) => entry.id === 'clawhub/steipete/weather');
    const state = browseCardState(claw, { installed: installed('weather') });
    assert.equal(text(state.status), 'You have another “weather” (from Ouroboros)');
    assert.match(state.action, /data-browse-action="compare"[^>]*>Compare…</);
    assert.match(text(browseCardState(item('ouroboros-community/old_tool'), {}).status), /^Withdrawn by Ouroboros Community$/);
});

test('Browse offers the current update after rejection, including a newer release and a modified restored copy', () => {
    for (const row of fixture('extensions_fix_r2.json').skills) {
        for (const modified of [false, true]) {
            row.install.modified = modified;
            const entry = { ...item('ouroboros/weather'), name: row.name, source_id: row.install.source.id };
            const state = browseCardState(entry, { installed: row });
            if (row.install.update_available) {
                assert.match(state.action, /data-browse-action="update"/);
                assert.doesNotMatch(state.action, /Open in Installed/);
                assert.ok(text(state.status).includes(`v${row.install.update_available.version} available`));
            } else {
                assert.match(state.action, modified ? />Details</ : /Open in Installed/);
                assert.doesNotMatch(state.action, /data-browse-action="update"/);
            }
        }
    }
    const row = fixture('extensions_fix_r2.json').skills[0];
    row.install.update_available = null;
    assert.match(browseCardState(item('ouroboros/weather'), { installed: row }).action, /Open in Installed/);
});

test('before the installed listing is read, the catalog\'s own join says installed or taken, with no action', () => {
    const early = browseCardState(item('ouroboros/weather'), { listed: false });
    assert.equal(text(early.status), 'Installed · v0.3.1');
    assert.equal(early.action, '', 'every action needs the listing row');
    const taken = { ...item('ouroboros/duckduckgo'), installed: { state: 'name_taken', source_id: 'my-folder', source_name: 'My skills' } };
    assert.equal(text(browseCardState(taken, { listed: false }).status), 'You have another “duckduckgo” (from My skills)');
    assert.match(browseCardState(item('ouroboros/duckduckgo'), { listed: false, source: SOURCES.get('ouroboros') }).action, />Install</);
    assert.equal(text(browseCardState(item('ouroboros/weather'), { listed: true }).status), 'Ouroboros reviews it before it runs',
        'once read, the listing decides (here: not installed)');
});

test('a card: name opens the sheet, Official only on endorsed entries, unknown popularity is said, categories under All', () => {
    const labels = new Map(CATALOG.categories.map((cat) => [cat.id, cat.label]));
    const weather = renderBrowseCard(item('ouroboros/weather'), { state: { status: 'S', action: 'A' }, categoryLabels: labels });
    assert.match(weather, /data-browse-action="open" data-item="ouroboros\/weather"><span data-i18n-authored>weather/);
    assert.match(weather, /data-official-mark/);
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
    const long = renderBrowseCard({ ...item('ouroboros/weather'), description: 'x '.repeat(400) }, { state: { status: '', action: '' } });
    assert.match(long, /data-clamped/);
    assert.match(long, />Read more ›</);
});

test('a connector is known by its place declaration, never a typed tag: icon, «Connects …», the Connectors filter', () => {
    const slack = { ...item('ouroboros/duckduckgo'), connects: ['Slack'], icon: { name: 'chat' } };
    assert.equal(isConnector(slack), true);
    assert.equal(isConnector(item('ouroboros/duckduckgo')), false);
    assert.equal(isConnector({ category: [CONNECTORS] }), true, 'a catalog category of the same id joins it');
    const card = renderBrowseCard(slack, { source: SOURCES.get(slack.source_id), state: browseCardState(slack) });
    assert.match(card, /<div class="skills-head-title"><span class="skills-card-icon" aria-hidden="true"><svg/);
    assert.match(card, /<div class="skills-card-cats"><span class="ui-chip skills-chip" data-tone="neutral">Connects Slack<\/span>/);
    const plain = renderBrowseCard(item('ouroboros/duckduckgo'), { source: SOURCES.get('ouroboros'), state: browseCardState(item('ouroboros/duckduckgo')) });
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
        const html = renderBrowseCard(candidate, { source: SOURCES.get(candidate.source_id), state: browseCardState(candidate) });
        assert.ok(html.includes(`<span data-i18n-authored>${title}</span>`));
        assert.match(html, /skills-card-who"><span data-i18n-authored>weather<\/span>/);
        assert.match(html, /data-item="ouroboros\/weather"/);
    }
});
