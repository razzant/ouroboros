// The wizard's Starter skills row (board B13): ticks = the maintainer's recommended
// marks, built-ins lead, one scrollable categorized list, the two hub trust choices,
// and one payload the atomic Finish carries — nothing when the list could not be read.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { createStarterSkillsRow, normalizeStarterGroups } from '../modules/onboarding_starter_skills.js';

const STARTER = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/onboarding_starter_skills.json', import.meta.url), 'utf8'));

function fakeHost() {
    const handlers = {};
    const summary = { innerHTML: '' };
    return {
        isConnected: true, innerHTML: '', handlers,
        addEventListener(type, fn) { handlers[type] = fn; },
        removeEventListener(type) { delete handlers[type]; },
        querySelector(selector) { return selector === '.skills-starter-summary' ? summary : null; },
    };
}
const target = (attrs) => ({ closest: (selector) => {
    const key = Object.keys(attrs).find((name) => selector.includes(name));
    return key ? attrs[key] : null;
} });
const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

test('built-ins lead as their own group; the host’s category order stays', () => {
    const lifted = normalizeStarterGroups([
        { category: 'messaging', label: 'Messaging', items: [{ id: 'builtin/telegram', name: 'telegram', builtin: true }, { id: 'h/x', name: 'x' }] },
        { category: 'computer', label: 'Computer control', items: [{ id: 'builtin/ucu', name: 'unix_computer_use', builtin: true }] },
    ]);
    assert.deepEqual(lifted.map((group) => [group.label, group.items.map((item) => item.name)]), [
        ['Built in — comes with the app', ['telegram', 'unix_computer_use']],
        ['Messaging', ['x']],
    ]);
    // The host's answer groups built-ins inside their categories; the row lifts them into one group first.
    assert.deepEqual(normalizeStarterGroups(STARTER.groups).map((group) => group.label), [
        'Built in — comes with the app', 'Messaging', 'Documents', 'Search & research', 'About Ouroboros', 'Writing & review']);
    assert.deepEqual(normalizeStarterGroups(null), []);
});

test('collapsed: «N · M ticked · Change ›» with the ticked names; the payload carries ticks and both trust choices', async () => {
    const row = createStarterSkillsRow({ read: async () => STARTER });
    await row.loading;
    const host = fakeHost();
    row.mount(host);
    assert.match(text(host.innerHTML), /^Starter skills 10 · 6 ticked · Change › Ticked: telegram, pdf_tools, word_tools, duckduckgo, memory-atlas, anti-slop\. 4 more you can tick\.$/);
    assert.deepEqual(row.payload(), {
        starter_skills: ['builtin/telegram', 'ouroboros/pdf_tools', 'ouroboros/word_tools', 'ouroboros/duckduckgo', 'ouroboros/memory-atlas', 'ouroboros/anti-slop'],
        skill_source_trust: { ouroboros: 'trusted', 'ouroboros-community': 'review' },
    });
    assert.doesNotMatch(host.innerHTML, /skills-starter-panel/, 'the list opens only on Change');
});

test('expanded: one scroll of categorized cards, Tick all / Untick all, and the hub trust pills', async () => {
    const row = createStarterSkillsRow({ read: async () => STARTER });
    await row.loading;
    const host = fakeHost();
    row.mount(host);
    host.handlers.click({ target: target({ '[data-starter-toggle]': {} }) });
    const html = host.innerHTML;
    assert.match(html, /data-starter-toggle aria-expanded="true">Hide ▾</);
    assert.match(html, /class="skills-starter-scroll" tabindex="0" aria-label="Starter skills"/);
    assert.match(html, /<h4>Built in — comes with the app<span>2<\/span><\/h4>/);
    assert.match(html, /<h4>Search &amp; research<span>3<\/span><\/h4>/);
    assert.match(html, /class="wizard-choice skills-starter-card active">[\s\S]*?data-starter-item="ouroboros\/duckduckgo" checked/);
    assert.match(html, /data-starter-item="builtin\/unix_computer_use"><strong>unix_computer_use<\/strong><span class="skills-starter-marks"><span class="ui-chip skills-chip" data-tone="neutral">Built in<\/span>/);
    assert.match(html, /Needs: KEENABLE_API_KEY/);
    assert.match(html, /They install without review — you trust Ouroboros\. Skills that need a key wait for it\./);
    assert.match(html, /data-starter-trust="ouroboros-community" data-trust="review">Review each</);

    host.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '0' } } }) });
    assert.deepEqual(row.payload().starter_skills, []);
    host.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '1' } } }) });
    assert.equal(row.payload().starter_skills.length, 10);
    host.handlers.click({ target: target({ '[data-starter-trust]': { dataset: { starterTrust: 'ouroboros', trust: 'review' } } }) });
    assert.equal(row.payload().skill_source_trust.ouroboros, 'review');
    assert.match(host.innerHTML, /Ouroboros reviews each one before it runs\./);
});

test('a box changed by the owner updates the ticks and the summary in place', async () => {
    const row = createStarterSkillsRow({ read: async () => STARTER });
    await row.loading;
    const host = fakeHost();
    row.mount(host);
    const card = { classList: { toggle() {} } };
    const box = { checked: true, dataset: { starterItem: 'ouroboros/keenable' }, closest: (sel) => (sel === '.skills-starter-card' ? card : null) };
    host.handlers.change({ target: { closest: (sel) => (sel === '[data-starter-item]' ? box : null) } });
    assert.ok(row.payload().starter_skills.includes('ouroboros/keenable'));
    assert.match(host.querySelector('.skills-starter-summary').innerHTML, /10 · 7 ticked/);
});

test('an unreadable list leaves the host defaults and says so; the Finish payload adds nothing', async () => {
    const row = createStarterSkillsRow({ read: async () => { throw new Error('offline'); } });
    await row.loading;
    const host = fakeHost();
    row.mount(host);
    assert.match(text(host.innerHTML), /The default set The list couldn’t be read \(offline\)\. Ouroboros installs the recommended skills after it starts/);
    assert.deepEqual(row.payload(), {});
    const empty = createStarterSkillsRow({ read: async () => ({}) });
    await empty.loading;
    assert.deepEqual(empty.payload(), {});
});

function pollingRow(read) {
    let time = 0;
    let id = 0;
    const timers = new Map();
    const row = createStarterSkillsRow({ read, now: () => time,
        setTimer(fn, ms) { timers.set(++id, { fn, ms }); return id; }, clearTimer(key) { timers.delete(key); } });
    return { row, timers, advance(ms) { time += ms; }, async tick() {
        const [key, timer] = [...timers][0];
        timers.delete(key);
        assert.equal(timer.ms, 3000);
        time += timer.ms;
        await timer.fn();
    } };
}
const freshOffer = () => ({ ...structuredClone(STARTER),
    groups: STARTER.groups.map((group) => ({ ...group, items: group.items.filter((item) => item.builtin) })).filter((group) => group.items.length),
    sources: STARTER.sources.map((source) => ({ ...source, state: source.kind === 'hub' ? 'reading' : source.state })),
});

function changeBox(host, id, checked) {
    const box = { checked, dataset: { starterItem: id }, closest: () => ({ classList: { toggle() {} } }) };
    host.handlers.change({ target: { closest: () => box } });
}

test('fresh-root polling merges recommended additions, preserves edits and trust, and stops once ready', async () => {
    let calls = 0;
    const poll = pollingRow(async () => ++calls === 1 ? freshOffer() : STARTER);
    await poll.row.loading;
    const host = fakeHost();
    poll.row.mount(host);
    assert.match(text(host.innerHTML), /Reading the Ouroboros hub…/);
    assert.match(text(host.innerHTML), /installs the default starter list once it can read the hub/);
    assert.equal('starter_skills' in poll.row.payload(), false, 'untouched incomplete offer leaves defaults to the host');
    changeBox(host, 'builtin/telegram', false);
    changeBox(host, 'builtin/unix_computer_use', true);
    host.handlers.click({ target: target({ '[data-starter-trust]': { dataset: { starterTrust: 'ouroboros', trust: 'review' } } }) });
    assert.deepEqual(poll.row.payload().starter_skills, ['builtin/unix_computer_use']);
    await poll.tick();
    assert.equal(calls, 2);
    assert.equal(poll.timers.size, 0);
    assert.deepEqual(poll.row.payload().starter_skills, ['builtin/unix_computer_use', 'ouroboros/pdf_tools', 'ouroboros/word_tools', 'ouroboros/duckduckgo', 'ouroboros/memory-atlas', 'ouroboros/anti-slop']);
    assert.equal(poll.row.payload().skill_source_trust.ouroboros, 'review');
    assert.doesNotMatch(text(host.innerHTML), /Reading the Ouroboros hub|default starter list/);
});

test('Untick all also leaves newly discovered recommended cards unticked', async () => {
    let calls = 0;
    const poll = pollingRow(async () => ++calls === 1 ? freshOffer() : STARTER);
    await poll.row.loading;
    const host = fakeHost();
    poll.row.mount(host);
    host.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '0' } } }) });
    assert.deepEqual(poll.row.payload().starter_skills, []);
    await poll.tick();
    assert.deepEqual(poll.row.payload().starter_skills, []);
});

test('polling is bounded to three minutes and an untouched failed hub still leaves the default payload', async () => {
    let calls = 0;
    const poll = pollingRow(async () => { calls += 1; return freshOffer(); });
    await poll.row.loading;
    poll.row.mount(fakeHost());
    while (poll.timers.size) await poll.tick();
    assert.equal(calls, 61, 'one initial read and at most sixty three-second polls');
    assert.equal('starter_skills' in poll.row.payload(), false);
    const failed = freshOffer();
    failed.sources.forEach((source) => { if (source.kind === 'hub') source.state = 'error'; });
    const offline = pollingRow(async () => failed);
    await offline.row.loading;
    assert.equal(offline.timers.size, 0);
    assert.equal('starter_skills' in offline.row.payload(), false);
});

test('destroy cancels polling and ignores an in-flight answer', async () => {
    let resolve;
    let calls = 0;
    const poll = pollingRow(() => ++calls === 1 ? freshOffer() : new Promise((done) => { resolve = done; }));
    await poll.row.loading;
    poll.row.mount(fakeHost());
    const pending = poll.tick();
    poll.row.destroy();
    resolve(STARTER);
    await pending;
    assert.equal(poll.row.snapshot().groups.flatMap((group) => group.items).length, 2);
    assert.equal(poll.timers.size, 0);
    const mounted = pollingRow(async () => freshOffer());
    await mounted.row.loading;
    mounted.row.mount(fakeHost());
    mounted.row.destroy();
    assert.equal(mounted.timers.size, 0);
});

test('the polling window starts on Summary mount after a long setup and does not reset on remount', async () => {
    let calls = 0;
    const poll = pollingRow(async () => { calls += 1; return freshOffer(); });
    await poll.row.loading;
    assert.equal(calls, 1, 'the initial offer can be prefetched');
    assert.equal(poll.timers.size, 0, 'earlier wizard steps do not spend the polling window');
    poll.advance(300_000);
    poll.row.mount(fakeHost());
    assert.equal(poll.timers.size, 1, 'Summary starts polling even after five minutes in setup');
    await poll.tick();
    assert.equal(calls, 2);
    poll.row.mount(fakeHost());
    assert.equal(poll.timers.size, 1, 'rerendering Summary does not add a poller');
    while (poll.timers.size) await poll.tick();
    assert.equal(calls, 61, 'three minutes are measured from first mount');
    poll.row.mount(fakeHost());
    assert.equal(poll.timers.size, 0, 'remounting does not renew the bounded window');
    poll.row.destroy();
});

test('built-in starter cards carry one Official mark next to Built in and disclose suggested trust', async () => {
    const row = createStarterSkillsRow({ read: async () => STARTER });
    await row.loading;
    const host = fakeHost();
    row.mount(host);
    host.handlers.click({ target: target({ '[data-starter-toggle]': {} }) });
    assert.match(host.innerHTML, /Built in<\/span><span class="skills-mark" data-mark="official">/);
    assert.match(text(host.innerHTML), /Suggested choices apply until you choose\./);
    assert.equal((host.innerHTML.match(/data-mark="official"/g) || []).length, 10);
});
