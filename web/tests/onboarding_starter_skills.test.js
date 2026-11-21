// The wizard's Skills step (owner 2026-11-21): ticks = the maintainer's recommended marks, built-ins lead, one
// categorized list of cards with the catalog's words, the cards arriving once per entry, one quiet line about how
// the hub's skills install, Summary's one row back to the step, and one payload the atomic Finish carries —
// nothing when the list could not be read or the step is not offered.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { createStarterSkillsStep, needsText, normalizeStarterGroups } from '../modules/onboarding_starter_skills.js';

const STARTER = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/onboarding_starter_skills.json', import.meta.url), 'utf8'));

function fakeHost(view = 'step') {
    const handlers = {};
    const count = { textContent: '' };
    return {
        isConnected: true, innerHTML: '', handlers, count, dataset: { starterSkills: view },
        addEventListener(type, fn) { handlers[type] = fn; },
        removeEventListener(type) { delete handlers[type]; },
        querySelector(selector) { return selector === '.skills-starter-count' ? count : null; },
    };
}
const target = (attrs) => ({ closest: (selector) => {
    const key = Object.keys(attrs).find((name) => selector.includes(name));
    return key ? attrs[key] : null;
} });
const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const cardOf = (html, id) => html.split('<label ').find((part) => part.includes(`data-starter-card="${id}"`)) || '';

test('built-ins lead as their own group; the host’s category order stays', () => {
    const lifted = normalizeStarterGroups([
        { category: 'messaging', label: 'Messaging', items: [{ id: 'builtin/telegram', name: 'telegram', builtin: true }, { id: 'h/x', name: 'x' }] },
        { category: 'computer', label: 'Computer control', items: [{ id: 'builtin/ucu', name: 'unix_computer_use', builtin: true }] },
    ]);
    assert.deepEqual(lifted.map((group) => [group.label, group.items.map((item) => item.name)]), [
        ['Built in — comes with the app', ['telegram', 'unix_computer_use']],
        ['Messaging', ['x']],
    ]);
    // The host's answer groups built-ins inside their categories; the step lifts them into one group first.
    assert.deepEqual(normalizeStarterGroups(STARTER.groups).map((group) => group.label), [
        'Built in — comes with the app', 'Messaging', 'Documents', 'Search & research', 'About Ouroboros', 'Writing & review']);
    assert.deepEqual(normalizeStarterGroups(null), []);
});

test('the prerequisite is the catalog’s own words; without them, the settings the skill reads in words', () => {
    assert.equal(needsText({ needs: 'Needs a Telegram bot token', needs_keys: ['TELEGRAM_BOT_TOKEN'] }), 'Needs a Telegram bot token');
    assert.equal(needsText({ needs: '', needs_keys: ['KEENABLE_API_KEY'] }), '', 'a blank one says the key is optional');
    assert.equal(needsText({ needs: null, needs_keys: ['KEENABLE_API_KEY'] }), 'Uses your Keenable API key');
    assert.equal(needsText({ needs_keys: ['OPENROUTER_API_KEY'] }, { OPENROUTER_API_KEY: 'OpenRouter API Key' }), 'Uses your OpenRouter API Key');
    assert.equal(needsText({ needs_keys: ['SLACK_BOT_TOKEN', 'SLACK_APP_TOKEN'] }), 'Uses your Slack bot token and 1 more setting');
    assert.equal(needsText({ needs_keys: ['A_B', 'C', 'D'] }), 'Uses your A b and 2 more settings');
    assert.equal(needsText({ needs_keys: [] }), '');
});

test('the step: the contract’s header over one list of cards with titles, pitches and their fallbacks', async () => {
    const step = createStarterSkillsStep({ read: async () => STARTER });
    assert.match(step.html({ title: 'Skills', copy: 'Ouroboros starts with the ticked skills; clear any you do not want.' }),
        /<h2 class="step-title">Skills<\/h2>[\s\S]*<p class="step-copy">Ouroboros starts with the ticked skills; clear any you do not want\.<\/p>[\s\S]*data-starter-skills="step"/);
    await step.loading;
    const host = fakeHost();
    step.mount(host);
    const html = host.innerHTML;
    assert.match(text(html), /^6 of 10 ticked Tick all · Untick all Built in — comes with the app 2/);
    assert.match(html, /<h3 style="--arrive: 0">Built in — comes with the app<span>2<\/span><\/h3>/);
    // The catalog's words translate; a slug and an author's description keep their own words.
    assert.match(cardOf(html, 'builtin/telegram'), /<strong>Telegram<\/strong>[\s\S]*Built in[\s\S]*<span class="skills-starter-pitch">Chat with Ouroboros from your phone in Telegram<\/span>[\s\S]*<span class="skills-starter-needs">Needs a Telegram bot token<\/span>/);
    assert.match(cardOf(html, 'builtin/unix_computer_use'), /<strong data-i18n-authored>unix_computer_use<\/strong>[\s\S]*<span class="skills-starter-pitch" data-i18n-authored>Desktop control: screenshots/);
    assert.doesNotMatch(cardOf(html, 'ouroboros/keenable'), /skills-starter-needs/, 'its key is optional: no line');
    assert.match(cardOf(html, 'ouroboros/slack-bridge'), /<span class="skills-starter-needs" data-i18n-authored>Uses your Slack bot token and 1 more setting<\/span>/);
    assert.match(cardOf(html, 'ouroboros/duckduckgo'), /class="wizard-choice skills-starter-card active"[\s\S]*data-starter-item="ouroboros\/duckduckgo" checked/);
    // One mark per card says where it comes from: Built in for the app's own, Official for the hub's vouched ones.
    assert.equal((html.match(/>Built in<\/span>/g) || []).length, 2);
    assert.equal((html.match(/data-mark="official"/g) || []).length, 8);
    assert.doesNotMatch(cardOf(html, 'builtin/telegram'), /data-mark="official"/);
    // One quiet line says how the hub's skills install; its choices open in place.
    assert.match(text(html), /They install without review — you trust Ouroboros\. Skills that need a key wait for it\. Change$/);
    assert.doesNotMatch(html, /data-starter-trust="/);
    host.handlers.click({ target: target({ '[data-starter-trust-toggle]': {} }) });
    assert.match(host.innerHTML, /data-starter-trust-toggle aria-expanded="true">Hide</);
    assert.match(host.innerHTML, /data-starter-trust="ouroboros-community" data-trust="review">Review each</);
    assert.match(text(host.innerHTML), /Suggested choices apply until you choose\.$/);
    assert.deepEqual(step.payload(), {
        starter_skills: ['builtin/telegram', 'ouroboros/pdf_tools', 'ouroboros/word_tools', 'ouroboros/duckduckgo', 'ouroboros/memory-atlas', 'ouroboros/anti-slop'],
        skill_source_trust: { ouroboros: 'trusted', 'ouroboros-community': 'review' },
    });
});

test('Tick all, Untick all (an explicit empty choice), one box and the hub trust', async () => {
    const step = createStarterSkillsStep({ read: async () => STARTER });
    await step.loading;
    const host = fakeHost();
    step.mount(host);
    host.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '0' } } }) });
    assert.deepEqual(step.payload().starter_skills, []);
    assert.match(text(host.innerHTML), /^0 of 10 ticked/);
    host.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '1' } } }) });
    assert.equal(step.payload().starter_skills.length, 10);
    const box = { checked: false, dataset: { starterItem: 'ouroboros/keenable' }, closest: (sel) => (sel === '.skills-starter-card' ? { classList: { toggle() {} } } : null) };
    host.handlers.change({ target: { closest: (sel) => (sel === '[data-starter-item]' ? box : null) } });
    assert.ok(!step.payload().starter_skills.includes('ouroboros/keenable'));
    assert.equal(host.count.textContent, '9 of 10 ticked', 'the count changes in place');
    host.handlers.click({ target: target({ '[data-starter-trust-toggle]': {} }) });
    host.handlers.click({ target: target({ '[data-starter-trust]': { dataset: { starterTrust: 'ouroboros', trust: 'review' } } }) });
    assert.equal(step.payload().skill_source_trust.ouroboros, 'review');
    assert.match(host.innerHTML, /Ouroboros reviews each one before it runs\./);
});

test('Summary’s one row: «N will install · Change», and Change returns to the step', async () => {
    let opened = 0;
    const step = createStarterSkillsStep({ read: async () => STARTER, open: () => { opened += 1; } });
    await step.loading;
    const row = fakeHost('row');
    step.mount(row);
    assert.equal(text(row.innerHTML), 'Skills 6 will install · Change');
    assert.doesNotMatch(row.innerHTML, /data-starter-card/);
    row.handlers.click({ target: target({ '[data-starter-change]': {} }) });
    assert.equal(opened, 1);
    const host = fakeHost();
    step.mount(host);
    host.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '0' } } }) });
    step.mount(fakeHost('row'));
    assert.deepEqual(step.payload().starter_skills, [], 'the cleared ticks survive the walk to Summary');
});

test('the cards arrive once per entry: never on a re-render or a refresh, again after leaving and coming back', async () => {
    const step = createStarterSkillsStep({ read: async () => STARTER });
    await step.loading;
    const first = fakeHost();
    step.mount(first);
    assert.match(first.innerHTML, /<div class="skills-starter-list" data-arrive>/);
    assert.match(cardOf(first.innerHTML, 'ouroboros/anti-slop'), /style="--arrive: 9"/);
    const rerendered = fakeHost();  // the wizard re-renders the step it is on: a new node, the same entry
    step.mount(rerendered);
    assert.match(rerendered.innerHTML, /<div class="skills-starter-list">/);
    rerendered.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '1' } } }) });
    assert.doesNotMatch(rerendered.innerHTML, /data-arrive/, 'a repaint never replays it');
    step.mount(fakeHost('row'));  // Summary
    const again = fakeHost();
    step.mount(again);
    assert.match(again.innerHTML, /data-arrive/, 'entering the step again plays it again');
    const capped = createStarterSkillsStep({ read: async () => ({ ...STARTER, groups: [{ category: 'x', label: 'X',
        items: Array.from({ length: 30 }, (_, index) => ({ id: `h/s${index}`, name: `s${index}`, recommended: true })) }] }) });
    await capped.loading;
    const many = fakeHost();
    capped.mount(many);
    assert.equal(Math.max(...[...many.innerHTML.matchAll(/--arrive: (\d+)/g)].map((match) => Number(match[1]))), 12,
        'from the twelfth card on they arrive together');
});

test('an unreadable list leaves the host defaults and says so; the Finish payload adds nothing', async () => {
    const step = createStarterSkillsStep({ read: async () => { throw new Error('offline'); } });
    await step.loading;
    const host = fakeHost();
    step.mount(host);
    assert.match(text(host.innerHTML), /^The list couldn’t be read \(offline\)\. Ouroboros installs the recommended skills after it starts; you can change them on the Skills page\.$/);
    assert.doesNotMatch(host.innerHTML, /data-arrive/);
    const row = fakeHost('row');
    step.mount(row);
    assert.match(text(row.innerHTML), /^Skills The recommended set · Change The list couldn’t be read \(offline\)/);
    assert.deepEqual(step.payload(), {});
    const empty = createStarterSkillsStep({ read: async () => ({}) });
    await empty.loading;
    assert.deepEqual(empty.payload(), {});
});

test('a step the server does not offer reads nothing, paints nothing and sends nothing', async () => {
    let reads = 0;
    const step = createStarterSkillsStep({ offered: false, read: async () => { reads += 1; return STARTER; } });
    await step.loading;
    const row = fakeHost('row');
    step.mount(row);
    assert.equal(reads, 0);
    assert.equal(row.innerHTML, '');
    assert.equal(step.offered(), false);
    assert.deepEqual(step.payload(), {});
});

function pollingStep(read) {
    let time = 0;
    let id = 0;
    const timers = new Map();
    const step = createStarterSkillsStep({ read, now: () => time,
        setTimer(fn, ms) { timers.set(++id, { fn, ms }); return id; }, clearTimer(key) { timers.delete(key); } });
    return { step, timers, advance(ms) { time += ms; }, async tick() {
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
    const poll = pollingStep(async () => ++calls === 1 ? freshOffer() : STARTER);
    await poll.step.loading;
    const host = fakeHost();
    poll.step.mount(host);
    assert.match(text(host.innerHTML), /Reading the Ouroboros hub…/);
    assert.match(text(host.innerHTML), /installs the default starter list once it can read the hub/);
    assert.equal('starter_skills' in poll.step.payload(), false, 'untouched incomplete offer leaves defaults to the host');
    changeBox(host, 'builtin/telegram', false);
    changeBox(host, 'builtin/unix_computer_use', true);
    host.handlers.click({ target: target({ '[data-starter-trust]': { dataset: { starterTrust: 'ouroboros', trust: 'review' } } }) });
    assert.deepEqual(poll.step.payload().starter_skills, ['builtin/unix_computer_use']);
    await poll.tick();
    assert.equal(calls, 2);
    assert.equal(poll.timers.size, 0);
    assert.deepEqual(poll.step.payload().starter_skills, ['builtin/unix_computer_use', 'ouroboros/pdf_tools', 'ouroboros/word_tools', 'ouroboros/duckduckgo', 'ouroboros/memory-atlas', 'ouroboros/anti-slop']);
    assert.equal(poll.step.payload().skill_source_trust.ouroboros, 'review');
    assert.doesNotMatch(text(host.innerHTML), /Reading the Ouroboros hub|default starter list/);
    assert.doesNotMatch(host.innerHTML, /data-arrive/, 'the cards the hub brings join without a second entrance');
});

test('a read that changes nothing rewrites nothing', async () => {
    const poll = pollingStep(async () => freshOffer());
    await poll.step.loading;
    const host = fakeHost();
    poll.step.mount(host);
    const painted = host.innerHTML;
    host.innerHTML = 'kept';  // a stand-in for the live nodes a rewrite would replace
    await poll.tick();
    assert.equal(host.innerHTML, 'kept');
    assert.match(painted, /data-arrive/);
});

test('Untick all also leaves newly discovered recommended cards unticked', async () => {
    let calls = 0;
    const poll = pollingStep(async () => ++calls === 1 ? freshOffer() : STARTER);
    await poll.step.loading;
    const host = fakeHost();
    poll.step.mount(host);
    host.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '0' } } }) });
    assert.deepEqual(poll.step.payload().starter_skills, []);
    await poll.tick();
    assert.deepEqual(poll.step.payload().starter_skills, []);
});

test('polling is bounded to three minutes and an untouched failed hub still leaves the default payload', async () => {
    let calls = 0;
    const poll = pollingStep(async () => { calls += 1; return freshOffer(); });
    await poll.step.loading;
    poll.step.mount(fakeHost());
    while (poll.timers.size) await poll.tick();
    assert.equal(calls, 61, 'one initial read and at most sixty three-second polls');
    assert.equal('starter_skills' in poll.step.payload(), false);
    const failed = freshOffer();
    failed.sources.forEach((source) => { if (source.kind === 'hub') source.state = 'error'; });
    const offline = pollingStep(async () => failed);
    await offline.step.loading;
    assert.equal(offline.timers.size, 0);
    assert.equal('starter_skills' in offline.step.payload(), false);
});

test('destroy cancels polling and ignores an in-flight answer', async () => {
    let resolve;
    let calls = 0;
    const poll = pollingStep(() => ++calls === 1 ? freshOffer() : new Promise((done) => { resolve = done; }));
    await poll.step.loading;
    poll.step.mount(fakeHost());
    const pending = poll.tick();
    poll.step.destroy();
    resolve(STARTER);
    await pending;
    assert.equal(poll.step.snapshot().groups.flatMap((group) => group.items).length, 2);
    assert.equal(poll.timers.size, 0);
    const mounted = pollingStep(async () => freshOffer());
    await mounted.step.loading;
    mounted.step.mount(fakeHost());
    mounted.step.destroy();
    assert.equal(mounted.timers.size, 0);
});

test('the polling window starts when the step (or Summary) first shows after a long setup and does not reset', async () => {
    let calls = 0;
    const poll = pollingStep(async () => { calls += 1; return freshOffer(); });
    await poll.step.loading;
    assert.equal(calls, 1, 'the initial offer can be prefetched');
    assert.equal(poll.timers.size, 0, 'earlier wizard steps do not spend the polling window');
    poll.advance(300_000);
    poll.step.mount(fakeHost());
    assert.equal(poll.timers.size, 1, 'the step starts polling even after five minutes in setup');
    await poll.tick();
    assert.equal(calls, 2);
    poll.step.mount(fakeHost('row'));
    assert.equal(poll.timers.size, 1, 'Summary does not add a poller');
    while (poll.timers.size) await poll.tick();
    assert.equal(calls, 61, 'three minutes are measured from first mount');
    poll.step.mount(fakeHost());
    assert.equal(poll.timers.size, 0, 'remounting does not renew the bounded window');
    poll.step.destroy();
});
