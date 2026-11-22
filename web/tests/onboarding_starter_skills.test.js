// The wizard's Skills step (owner 2026-11-21): ticks = the maintainer's recommended marks, built-ins lead, one
// categorized list of rows (not boxes) with the catalog's words, the rows arriving once per entry, one small sentence
// under the list about where the hub's skills come from and how they install, Summary's one row back to the step, and
// one payload the atomic Finish carries — nothing when the list could not be read or the step is not offered.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

import { createStarterSkillsStep, needsText, normalizeStarterGroups } from '../modules/onboarding_starter_skills.js';
import { checkIdentity, normalizeConnection, pairingCode } from '../modules/onboarding_connection_card.js';
import { qrSvg } from '../modules/qr_svg.js';
import { applyPayload, setMissTransport } from '../modules/i18n.js';

// The vendored encoder as a page loads it: a classic script that defines the global `qrcodegen`.
vm.runInThisContext(readFileSync(new URL('../vendor/qrcodegen/qrcodegen.js', import.meta.url), 'utf8'));

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
const rowOf = (html, id) => (html.split('<label ').find((part) => part.includes(`data-starter-skill="${id}"`)) || '').split('</label>')[0];

test('built-ins lead as their own group; the host’s category order stays', () => {
    const lifted = normalizeStarterGroups([
        { category: 'messaging', label: 'Messaging', items: [{ id: 'builtin/telegram', name: 'telegram', builtin: true }, { id: 'h/x', name: 'x' }] },
        { category: 'computer', label: 'Computer control', items: [{ id: 'builtin/ucu', name: 'unix_computer_use', builtin: true }] },
    ]);
    assert.deepEqual(lifted.map((group) => [group.label, group.items.map((item) => item.name)]), [
        ['Built in', ['telegram', 'unix_computer_use']],
        ['Messaging', ['x']],
    ]);
    // The host's answer groups built-ins inside their categories; the step lifts them into one group first.
    assert.deepEqual(normalizeStarterGroups(STARTER.groups).map((group) => group.label), [
        'Built in', 'Messaging', 'Documents', 'Search & research', 'About Ouroboros', 'Writing & review']);
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

test('the step: the contract’s header over one list of rows with titles, pitches and their fallbacks', async () => {
    const step = createStarterSkillsStep({ read: async () => STARTER });
    assert.match(step.html({ title: 'Skills', copy: 'Ouroboros starts with the ticked skills; clear any you do not want.' }),
        /<h2 class="step-title">Skills<\/h2>[\s\S]*<p class="step-copy">Ouroboros starts with the ticked skills; clear any you do not want\.<\/p>[\s\S]*data-starter-skills="step"/);
    await step.loading;
    const host = fakeHost();
    step.mount(host);
    const html = host.innerHTML;
    assert.match(text(html), /^6 of 10 ticked Tick all · Untick all Built in unix_computer_use /);
    // A group's head is its name, no count; the categories of one skill (Messaging, About Ouroboros, Writing & review
    // here) close the list together as «More».
    assert.deepEqual((html.match(/<h3[^>]*>[^<]*<\/h3>/g) || []).map((head) => text(head)), ['Built in', 'Documents', 'Search &amp; research', 'More']);
    // One row: the box, the title, then what it gives and, in a few words, what it lacks; no box of its own. The
    // whole sentence is read out in place of the mark. The catalog's words translate; a slug and an author's
    // description keep their own words.
    assert.match(rowOf(html, 'builtin/telegram'), /<strong>Telegram<\/strong>\s*<span class="skills-starter-pitch">Chat with Ouroboros from your phone<\/span>\s*<span class="skills-starter-needs" aria-hidden="true">· bot token<\/span><span class="skills-starter-sr">Needs a Telegram bot token<\/span><\/span>/);
    assert.match(rowOf(html, 'builtin/unix_computer_use'), /<strong data-i18n-authored>unix_computer_use<\/strong>[\s\S]*<span class="skills-starter-pitch" data-i18n-authored>Desktop control: screenshots/);
    assert.doesNotMatch(rowOf(html, 'ouroboros/keenable'), /skills-starter-needs/, 'its key is optional: no line');
    // Without the catalog's mark the row says «needs setup» and reads out what the skill reads.
    assert.match(rowOf(html, 'ouroboros/slack-bridge'), /<span class="skills-starter-needs" aria-hidden="true">· needs setup<\/span><span class="skills-starter-sr" data-i18n-authored>Uses your Slack bot token and 1 more setting<\/span>/);
    // A ticked row is the box alone: no card, no chosen fill.
    assert.match(rowOf(html, 'ouroboros/duckduckgo'), /^class="skills-starter-skill"[\s\S]*data-starter-item="ouroboros\/duckduckgo" checked/);
    assert.doesNotMatch(html, /wizard-choice|skills-starter-skill active/);
    // Where they come from is said once, never per row: the built-in group's head, the one sentence under the list.
    assert.doesNotMatch(html, /data-mark="official"|skills-chip|skills-starter-marks|maintainer vouches/);
    assert.equal((html.match(/Official skills from the Ouroboros hub/g) || []).length, 1);
    assert.match(text(html), /Built in .* More Better writing /);
    // Inside a group what works at once leads (recommended first) and what needs something comes last.
    const order = [...html.matchAll(/data-starter-skill="([^"]+)"/g)].map((match) => match[1]);
    assert.deepEqual(order, ['builtin/unix_computer_use', 'builtin/telegram', 'ouroboros/pdf_tools', 'ouroboros/word_tools',
        'ouroboros/duckduckgo', 'ouroboros/keenable', 'ouroboros/perplexity', 'ouroboros/anti-slop', 'ouroboros/memory-atlas', 'ouroboros/slack-bridge']);
    // Where the hub's skills come from and how they install is one sentence under the list, in plain words; its
    // choices open in place, in Settings' words.
    assert.match(text(html), /Official skills from the Ouroboros hub install without a review; skills from the community are reviewed first\. Change$/);
    assert.doesNotMatch(html, /data-starter-trust="|Without review|Review each/);
    host.handlers.click({ target: target({ '[data-starter-trust-toggle]': {} }) });
    assert.match(host.innerHTML, /data-starter-trust-toggle aria-expanded="true">Hide</);
    assert.match(host.innerHTML, /<strong>Community skills<\/strong>[\s\S]*data-starter-trust="ouroboros-community" data-trust="review">Review each skill</);
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
    const box = { checked: false, dataset: { starterItem: 'ouroboros/keenable' }, closest: () => null };
    host.handlers.change({ target: { closest: (sel) => (sel === '[data-starter-item]' ? box : null) } });
    assert.ok(!step.payload().starter_skills.includes('ouroboros/keenable'));
    assert.equal(host.count.textContent, '9 of 10 ticked', 'the count changes in place');
    host.handlers.click({ target: target({ '[data-starter-trust-toggle]': {} }) });
    host.handlers.click({ target: target({ '[data-starter-trust]': { dataset: { starterTrust: 'ouroboros', trust: 'review' } } }) });
    assert.equal(step.payload().skill_source_trust.ouroboros, 'review');
    assert.match(text(host.innerHTML), /Official skills from the Ouroboros hub and skills from the community are reviewed first\. Hide/);
    host.handlers.click({ target: target({ '[data-starter-trust]': { dataset: { starterTrust: 'ouroboros-community', trust: 'trusted' } } }) });
    assert.match(text(host.innerHTML), /Skills from the community install without a review; Official skills from the Ouroboros hub are reviewed first\./);
});

test('a skill that is not Official gets no mark of its own either: the sentence speaks of the hub rows only', async () => {
    const mixed = structuredClone(STARTER);
    mixed.groups[0].items[1].official = false;  // a build's own starter id from another source
    const step = createStarterSkillsStep({ read: async () => mixed });
    await step.loading;
    const host = fakeHost();
    step.mount(host);
    assert.doesNotMatch(rowOf(host.innerHTML, 'ouroboros/slack-bridge'), /Official|data-mark/);
    assert.equal((host.innerHTML.match(/Official skills from the Ouroboros hub/g) || []).length, 1);
    const unread = createStarterSkillsStep({ read: async () => ({ ...mixed, hub_rows: [] }) });
    await unread.loading;
    const bare = fakeHost();
    unread.mount(bare);
    assert.doesNotMatch(bare.innerHTML, /Official skills|skills-starter-trust/, 'no hub row, no sentence');
});

test('Summary’s one row: «N will install · Change», and Change returns to the step', async () => {
    let opened = 0;
    const step = createStarterSkillsStep({ read: async () => STARTER, open: () => { opened += 1; } });
    await step.loading;
    const row = fakeHost('row');
    step.mount(row);
    assert.equal(text(row.innerHTML), 'Skills 6 will install · Change');
    assert.doesNotMatch(row.innerHTML, /data-starter-skill/);
    row.handlers.click({ target: target({ '[data-starter-change]': {} }) });
    assert.equal(opened, 1);
    const host = fakeHost();
    step.mount(host);
    host.handlers.click({ target: target({ '[data-starter-all]': { dataset: { starterAll: '0' } } }) });
    step.mount(fakeHost('row'));
    assert.deepEqual(step.payload().starter_skills, [], 'the cleared ticks survive the walk to Summary');
});

test('the rows arrive once per entry: never on a re-render or a refresh, again after leaving and coming back', async () => {
    const step = createStarterSkillsStep({ read: async () => STARTER });
    await step.loading;
    const first = fakeHost();
    step.mount(first);
    assert.match(first.innerHTML, /<div class="skills-starter-list" data-arrive>/);
    assert.match(rowOf(first.innerHTML, 'ouroboros/slack-bridge'), /style="--arrive: 9"/);
    assert.match(first.innerHTML, /<h3 style="--arrive: 2">Documents/, 'a group head arrives before its rows');
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
        'from the twelfth row on they arrive together');
});

test('an unreadable list leaves the host defaults and says so; the Finish payload adds nothing', async () => {
    const step = createStarterSkillsStep({ read: async () => { throw new Error('offline'); } });
    await step.loading;
    const host = fakeHost();
    step.mount(host);
    assert.match(text(host.innerHTML), /^The list couldn’t be read \(offline\)\. Ouroboros installs the recommended skills after it starts\. Manage them in Extensions › Skills\.$/);
    assert.doesNotMatch(host.innerHTML, /data-arrive/);
    const row = fakeHost('row');
    step.mount(row);
    assert.match(text(row.innerHTML), /^Skills The recommended set · Change The list couldn’t be read \(offline\)/);
    assert.deepEqual(step.payload(), {});
    const empty = createStarterSkillsStep({ read: async () => ({}) });
    await empty.loading;
    assert.deepEqual(empty.payload(), {});
});

test('a step the server does not offer shows no ticks; Summary keeps where skills change and the hub trust choices', async () => {
    const step = createStarterSkillsStep({ offered: false, read: async () => STARTER });
    await step.loading;
    assert.equal(step.offered(), false);
    const stepHost = fakeHost('step');
    step.mount(stepHost);
    assert.equal(stepHost.innerHTML, '', 'no step, so no tick Finish would ignore');
    const row = fakeHost('row');
    step.mount(row);
    assert.doesNotMatch(row.innerHTML, /data-starter-item|data-starter-change|will install/);
    assert.match(text(row.innerHTML), /^Skills Add or remove them in Extensions › Skills Official skills from the Ouroboros hub install without a review; skills from the community are reviewed first\. Change$/);
    // The trust choices stay editable and ride Finish alone: returning installs still honor them.
    row.handlers.click({ target: target({ '[data-starter-trust-toggle]': {} }) });
    row.handlers.click({ target: target({ '[data-starter-trust]': { dataset: { starterTrust: 'ouroboros-community', trust: 'trusted' } } }) });
    assert.deepEqual(step.payload(), { skill_source_trust: { ouroboros: 'trusted', 'ouroboros-community': 'trusted' } });
    // Offered, the row is the step's own again.
    const offered = createStarterSkillsStep({ read: async () => STARTER });
    await offered.loading;
    const back = fakeHost('row');
    offered.mount(back);
    assert.match(text(back.innerHTML), /^Skills \d+ will install · Change/);
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
    assert.doesNotMatch(host.innerHTML, /data-arrive/, 'the rows the hub brings join without a second entrance');
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

test('Untick all also leaves newly discovered recommended rows unticked', async () => {
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

// A skill's declared connection card (the agreed manifest shape, with the skill's own words) on its starter item.
const CARD = {
    skill: 'telegram', token: 'botToken',
    pairing: { link: 'https://t.me/{username}?start={code}', hint: 'Scan it with your phone’s camera, then press Start in Telegram.',
        note: 'You can finish setup now; the bot answers once Ouroboros starts.' },
    title: 'Phone via Telegram', pitch: 'Write to Ouroboros and get its results on your phone',
    steps: [{ text: 'Open @BotFather in Telegram', link: 'https://t.me/BotFather', action: 'Open @BotFather' }, { text: 'Send /newbot' },
        { text: 'Choose a name, then a username ending in “bot”' }, { text: 'BotFather replies with a token: paste it here and press Check' }],
    options: [{ key: 'TELEGRAM_NOTIFY_TASKS', label: 'Send results and errors of work here', on: 'on', off: 'off', new_connection: true,
        store: 'skill_settings', route: 'settings/save' }],
};
const TOKEN = '123456789:AAF-secret-token-value';
function withCard(declaration = CARD) {
    const answer = structuredClone(STARTER);
    answer.groups.flatMap((group) => group.items).find((item) => item.id === 'builtin/telegram').connection = declaration;
    return answer;
}
const settle = () => new Promise((done) => setTimeout(done, 0));
async function cardStep(answer, check) {
    const checks = [];
    const step = createStarterSkillsStep({ read: async () => answer, card: { random: (bytes) => bytes.fill(7), encoder: async () => globalThis.qrcodegen,
        check: async (skill, token) => { checks.push([skill, token]); return check(token); } } });
    await step.loading;
    const host = fakeHost();
    step.mount(host);
    return { step, host, checks };
}
const paste = (host, value) => host.handlers.input({ target: target({ '[data-connection-token]': { value } }), inputType: 'insertFromPaste' });
const cardOf = (html) => html.split('<div class="skills-starter-bar">')[0];

test('a declared connection card leads the step as one row with Connect; its skill leaves its group, not the count', async () => {
    const { host } = await cardStep(withCard(), () => ({ status: 'ok', username: 'my_ouro_bot' }));
    const card = cardOf(host.innerHTML);
    assert.match(text(card), /^Phone via Telegram Write to Ouroboros and get its results on your phone Connect$/);
    assert.match(card, /class="skills-connection" data-connection="telegram" style="--arrive: 0"/);
    assert.match(card, /data-starter-item="builtin\/telegram" checked/, 'its box is the skill’s starter tick');
    // Its built-in neighbour is left alone in its group, so it joins «More» at the end.
    assert.match(text(host.innerHTML), /6 of 10 ticked Tick all · Untick all Documents .* More .*unix_computer_use/);
    assert.equal((host.innerHTML.match(/data-starter-skill="builtin\/telegram"/g) || []).length, 1);
    // Without a declaration, or with one the card cannot show, the skill stays a row of its group.
    for (const broken of [undefined, { ...CARD, pairing: { ...CARD.pairing, link: 'https://t.me/bot' } }, { ...CARD, token: '' }]) {
        const plain = await cardStep(broken ? withCard(broken) : STARTER, () => ({ status: 'ok', username: 'x' }));
        assert.doesNotMatch(plain.host.innerHTML, /skills-connection/);
        assert.match(text(plain.host.innerHTML), /Tick all · Untick all Built in unix_computer_use .* Telegram /);
    }
    assert.equal(normalizeConnection({ ...CARD, pairing: { ...CARD.pairing, link: 'http://t.me/{username}?start={code}' } }), null, 'https only');
});

test('Connect opens the skill’s steps and its token field; a checked token shows the QR of the link with a code made here', async () => {
    const { step, host, checks } = await cardStep(withCard(), () => ({ status: 'ok', username: 'my_ouro_bot' }));
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    const code = step.snapshot().connection.code;
    assert.equal(code, pairingCode((bytes) => bytes.fill(7)));
    assert.match(code, /^[A-Za-z0-9_-]{32}$/, 'as the skill mints one: 24 random bytes');
    let card = cardOf(host.innerHTML);
    assert.match(text(card), /Phone via Telegram .* Hide Open @BotFather in Telegram Open @BotFather Send \/newbot Choose a name, then a username ending in “bot” BotFather replies with a token: paste it here and press Check Bot token Check$/);
    // The field stands under the skill's last step, which says what to paste; its own name is read out, not shown.
    assert.match(card, /<li class="skills-connection-token"><span>BotFather replies with a token: paste it here and press Check<\/span><label for="connection-telegram-token" class="skills-starter-sr">Bot token<\/label>/);
    assert.match(card, /href="https:\/\/t\.me\/BotFather" target="_blank" rel="noopener noreferrer"/);
    assert.match(card, /type="password" autocomplete="off" spellcheck="false" data-connection-token/);
    assert.doesNotMatch(card, /<svg|data-connection-account/, 'no QR before a token is checked');
    paste(host, TOKEN);
    await settle();
    assert.deepEqual(checks, [['telegram', TOKEN]]);
    card = cardOf(host.innerHTML);
    const link = `https://t.me/my_ouro_bot?start=${code}`;
    assert.ok(card.includes(qrSvg(link, { label: 'QR code for @my_ouro_bot' })), 'the QR is the link with this card’s code');
    assert.match(text(card), /@my_ouro_bot Scan it with your phone’s camera, then press Start in Telegram\. You can finish setup now; the bot answers once Ouroboros starts\. On this device\? Open @my_ouro_bot Send results and errors of work here$/);
    assert.match(card, new RegExp(`href="${link.replace(/[.?]/g, '\\$&')}" target="_blank" rel="noopener noreferrer" data-connection-link`));
    assert.match(card, /data-connection-option="TELEGRAM_NOTIFY_TASKS" checked/, 'on for a new connection');
    assert.equal(card.split(TOKEN).length - 1, 1, 'the token is the field’s own value only: never in the QR or a link');
    // The code is made once: hiding and opening again, or checking again, keeps it.
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    assert.match(text(cardOf(host.innerHTML)), /^Phone via Telegram @my_ouro_bot · connects when you finish setup Change$/);
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    host.handlers.click({ target: target({ '[data-connection-check]': {} }) });
    await settle();
    assert.equal(step.snapshot().connection.code, code);
    assert.equal(checks.length, 1, 'a settled answer for this token is not asked again');
    // Its values ride the one completion apart from the settings, the option as the box says.
    host.handlers.change({ target: target({ '[data-connection-option]': { checked: false, dataset: { connectionOption: 'TELEGRAM_NOTIFY_TASKS' } } }) });
    const payload = step.payload();
    assert.deepEqual(payload.connections, { telegram: { fields: { botToken: TOKEN }, code, options: { TELEGRAM_NOTIFY_TASKS: 'off' } } });
    assert.ok(!Object.keys(payload).some((key) => key.startsWith('TELEGRAM_')), 'nothing of the card in the settings part');
    const row = fakeHost('row');
    step.mount(row);
    assert.match(text(row.innerHTML), /^Skills 6 will install · Change Phone via Telegram @my_ouro_bot · Change$/);
    // Unticked, the skill is not installed and its card sends nothing.
    step.mount(host);
    host.handlers.change({ target: { closest: (selector) => (selector === '[data-starter-item]' ? { checked: false, dataset: { starterItem: 'builtin/telegram' } } : null) } });
    assert.equal(step.payload().connections, undefined);
});

test('each failure says only its own cause and never blocks; a token the service refused is not sent', async () => {
    const words = {};
    for (const answer of [{ status: 'refused' }, { status: 'unreachable' }, { status: 'unavailable', reason: 'HTTP 404' }]) {
        const { step, host } = await cardStep(withCard(), () => answer);
        host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
        paste(host, TOKEN);
        await settle();
        words[answer.status] = text(cardOf(host.innerHTML)).replace(/^.* Check /, '');
        assert.doesNotMatch(host.innerHTML, /<svg|data-connection-option/, 'no QR, and no options without it');
        // No QR was drawn, so no code is armed: the token alone rides the completion (the service's refusal: nothing).
        assert.deepEqual(step.payload().connections, answer.status === 'refused' ? undefined : { telegram: { fields: { botToken: TOKEN } } });
    }
    assert.match(words.refused, /^Telegram did not accept this token; it is not saved\.$/);
    assert.match(words.unreachable, /^Could not reach Telegram from here, so there is no QR code\. The token is still saved when you finish\.$/);
    assert.match(words.unavailable, /^The token could not be checked \(HTTP 404\), so there is no QR code\. The token is still saved when you finish\.$/);
    assert.doesNotMatch(words.refused, /reach|checked/);
    assert.doesNotMatch(words.unreachable, /accept/);
});

test('a code is armed only with the QR that shows it: none while the check runs, never ran or failed', async () => {
    let answer = null;
    const { step, host, checks } = await cardStep(withCard(), () => new Promise((done) => { answer = done; }));
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    host.handlers.input({ target: target({ '[data-connection-token]': { value: TOKEN } }), inputType: 'insertText' });
    assert.deepEqual(step.payload().connections, { telegram: { fields: { botToken: TOKEN } } }, 'typed, never checked: the token alone');
    host.handlers.click({ target: target({ '[data-connection-check]': {} }) });
    assert.equal(checks.length, 1);
    assert.deepEqual(step.payload().connections, { telegram: { fields: { botToken: TOKEN } } }, 'still checking: the token alone');
    answer({ status: 'ok', username: 'my_ouro_bot' });
    await settle();
    assert.match(host.innerHTML, /<svg/);
    const { code } = step.snapshot().connection;
    assert.deepEqual(step.payload().connections, { telegram: { fields: { botToken: TOKEN }, code, options: { TELEGRAM_NOTIFY_TASKS: 'on' } } },
        'the QR drawn: its code and the options ride with the token');
});

test('a code rides only with a QR the page painted: a check answering once the card is hidden or the step left arms none', async () => {
    for (const away of ['hide', 'leave']) {
        let answer = null;
        const { step, host } = await cardStep(withCard(), () => new Promise((done) => { answer = done; }));
        const row = fakeHost('row');
        host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
        paste(host, TOKEN);  // the check starts at once
        if (away === 'hide') host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
        else step.mount(row);  // Continue, before the answer
        answer({ status: 'ok', username: 'my_ouro_bot' });
        await settle();
        assert.doesNotMatch(host.innerHTML + row.innerHTML, /<svg|data-connection-link/, `${away}: nothing was painted`);
        assert.deepEqual(step.payload().connections, { telegram: { fields: { botToken: TOKEN } } }, `${away}: the token alone`);
        if (away === 'hide') assert.match(text(cardOf(host.innerHTML)), /^Phone via Telegram Token checked · QR not shown yet Change$/);
        step.mount(row);
        assert.match(text(row.innerHTML), /Phone via Telegram Token checked · QR not shown yet · Change$/, `${away}: no account named`);
        // Back on the step, opening the card paints the QR: its code rides now, and leaving the step again keeps it.
        step.mount(host);
        if (away === 'hide' || !host.innerHTML.includes('data-connection-link')) host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
        assert.match(host.innerHTML, /data-connection-link/, `${away}: the QR painted`);
        const { code } = step.snapshot().connection;
        step.mount(row);
        assert.deepEqual(step.payload().connections, { telegram: { fields: { botToken: TOKEN }, code, options: { TELEGRAM_NOTIFY_TASKS: 'on' } } });
        assert.match(text(row.innerHTML), /Phone via Telegram @my_ouro_bot · Change$/);
    }
});

test('a changed draft takes the shown check of the one before off the page at once, the field kept; going back shows it again', async () => {
    const { step, host } = await cardStep(withCard(), () => ({ status: 'ok', username: 'former_bot' }));
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    paste(host, TOKEN);
    await settle();
    assert.match(host.innerHTML, /data-connection-link/);
    let html = host.innerHTML;
    let writes = 0;
    Object.defineProperty(host, 'innerHTML', { get: () => html, set(value) { writes += 1; html = value; } });
    const removed = [];
    const typed = (value) => {
        const node = { querySelectorAll: (selector) => selector.split(', ').map((one) => ({ remove: () => removed.push(one) })) };
        const field = { value, closest: (selector) => ({ '[data-connection-token]': field, '[data-connection]': node })[selector] || null };
        host.handlers.input({ target: field, inputType: 'insertText' });
    };
    typed(TOKEN);  // the checked token itself: nothing leaves
    assert.deepEqual([removed, writes], [[], 0]);
    typed(TOKEN + '8');
    assert.deepEqual(removed, ['.skills-connection-pair', '[data-connection-status]'], 'the old QR, link and words leave at once');
    assert.equal(writes, 0, 'nothing rewritten: the field being typed into stays, caret and all');
    assert.deepEqual(step.payload().connections, { telegram: { fields: { botToken: `${TOKEN}8` } } }, 'an unchecked draft: no code');
    typed(TOKEN + '89');
    assert.deepEqual([removed.length, writes], [2, 0], 'nothing of the old check is left to take');
    // Back to the token it checked: its QR shows again (the step writes anew), and its code rides.
    typed(TOKEN);
    assert.equal(writes, 1);
    assert.match(html, /data-connection-link/);
    const { code } = step.snapshot().connection;
    assert.deepEqual(step.payload().connections, { telegram: { fields: { botToken: TOKEN }, code, options: { TELEGRAM_NOTIFY_TASKS: 'on' } } });
});

test('a draft that is no token is said here, never checked and never sent; the service’s words stay its own', async () => {
    for (const [draft, words] of [[`${TOKEN} more`, /^A token has no spaces or line breaks inside, so this one is not saved\.$/],
        [`${TOKEN}\nsecond line`, /^A token has no spaces or line breaks inside, so this one is not saved\.$/],
        ['x'.repeat(513), /^A token is at most 512 characters, so this one is not saved\.$/]]) {
        const { step, host, checks } = await cardStep(withCard(), () => ({ status: 'ok', username: 'my_ouro_bot' }));
        host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
        paste(host, draft);
        await settle();
        assert.deepEqual(checks, [], 'nothing went to the check');
        assert.match(text(cardOf(host.innerHTML)).replace(/^.* Check /, ''), words);
        assert.doesNotMatch(host.innerHTML, /did not accept/);
        assert.equal(step.payload().connections, undefined, 'a completion never carries it');
        host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
        assert.match(text(cardOf(host.innerHTML)), /^Phone via Telegram Not a token · not saved Change$/);
    }
    // At the bound and without inner spaces, a draft is checked as usual (the ends are trimmed).
    const { checks, host } = await cardStep(withCard(), () => ({ status: 'refused' }));
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    paste(host, ` ${'x'.repeat(512)} `);
    await settle();
    assert.deepEqual(checks, [['telegram', 'x'.repeat(512)]]);
});

test('a repaint never rewrites the step inside another: an edit the removed field reports is checked and shown after it', async () => {
    // Chromium reports a focused field's unreported edit (`change`) from inside the rewrite that removes it, and a
    // rewrite from that handler makes the outer one throw («…moved in a 'blur' event handler?»). This host does the
    // same: its focused token field's pending edit is reported from inside its rewrite; a rewrite inside it is recorded.
    let loaded = null;
    const checks = [];
    const step = createStarterSkillsStep({ read: async () => withCard(), card: { random: (bytes) => bytes.fill(7),
        encoder: () => new Promise((done) => { loaded = done; }),  // the QR encoder still loading
        check: async (skill, token) => { checks.push(token); return token === TOKEN ? new Promise(() => {}) : { status: 'refused' }; } } });
    await step.loading;
    const host = fakeHost();
    const nested = [];
    let html = '';
    let writing = false;
    host.edit = null;  // what the person typed into the focused token field, not yet reported by the browser
    Object.defineProperty(host, 'innerHTML', { get: () => html, set(value) {
        if (writing) { nested.push(value); return; }
        writing = true;
        try {
            const edit = host.edit;
            host.edit = null;
            const field = { value: edit, closest: (selector) => (selector === '[data-connection-token]' ? field : null) };
            if (edit !== null) host.handlers.change({ target: field });
            html = value;
        } finally { writing = false; }
    } });
    step.mount(host);
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    host.handlers.input({ target: target({ '[data-connection-token]': { value: '123:wrong' } }), inputType: 'insertText' });
    host.handlers.click({ target: target({ '[data-connection-check]': {} }) });
    assert.match(text(cardOf(host.innerHTML)), /Checking…$/, 'a repaint of its own writes at once');
    await settle();
    assert.match(text(cardOf(host.innerHTML)), /Telegram did not accept this token; it is not saved\.$/);
    // A new token typed into the still focused field; then the encoder finishes loading and repaints the step.
    host.handlers.input({ target: target({ '[data-connection-token]': { value: TOKEN } }), inputType: 'insertText' });
    host.edit = TOKEN;
    loaded(globalThis.qrcodegen);
    await settle();
    assert.deepEqual(nested, [], 'no rewrite inside a rewrite');
    assert.deepEqual(checks, ['123:wrong', TOKEN], 'the edit the removed field reported is checked');
    assert.match(text(cardOf(host.innerHTML)), /Checking…$/, 'and shown, right after the rewrite that removed the field');
    step.destroy();
});

test('the card’s declared words translate as the catalog’s do; an account’s @name stays; composed words go through fmt', async () => {
    const { host } = await cardStep(withCard(), () => ({ status: 'ok', username: 'my_ouro_bot' }));
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    paste(host, TOKEN);
    await settle();
    const card = cardOf(host.innerHTML);
    // Title, pitch, steps, the step's button, hint, note and option label carry no mark that keeps them English.
    assert.deepEqual(card.match(/<[^>]*data-i18n-authored[^>]*>/g), ['<div class="skills-connection-qr" data-i18n-authored>',
        '<strong data-connection-account data-i18n-authored>'], 'only the QR (its label names the account) and the @name');
    assert.match(card, /data-i18n-fmt="Open @\{account\}">Open @my_ouro_bot<\/a>/, 'the link’s words are composed, not keyed with a name');
    assert.match(card, /<span>Send results and errors of work here<\/span>/);
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    assert.match(cardOf(host.innerHTML), /<span class="skills-starter-pitch" data-i18n-fmt="@\{account\} · connects when you finish setup">@my_ouro_bot · connects when you finish setup<\/span>/);
    // A language with a memory: a failure's sentence names the service through the composed seam, never keyed with it.
    setMissTransport(() => {});
    applyPayload({ language: 'ru', english: false, entries: {
        '{service} did not accept this token; it is not saved.': { text: '{service} не принял этот токен, он не сохранён.' } } });
    try {
        const refused = await cardStep(withCard(), () => ({ status: 'refused' }));
        refused.host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
        paste(refused.host, TOKEN);
        await settle();
        assert.match(cardOf(refused.host.innerHTML), /data-connection-status="refused" data-i18n-fmt="\{service\} did not accept this token; it is not saved\.">Telegram не принял этот токен, он не сохранён\.<\/p>/);
    } finally {
        applyPayload({ language: 'en', english: true });
    }
});

test('where the install grants skills nothing on its own, the card says the skill starts once allowed, never «connects»', async () => {
    const { host } = await cardStep(withCard({ ...CARD, needs_allow: true }), () => ({ status: 'ok', username: 'my_ouro_bot' }));
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    paste(host, TOKEN);
    await settle();
    const card = cardOf(host.innerHTML);
    assert.match(text(card), /@my_ouro_bot Scan it with your phone’s camera, then press Start in Telegram\. Telegram starts after you allow it in Extensions\. On this device\? Open @my_ouro_bot /);
    assert.doesNotMatch(text(card), /the bot answers once Ouroboros starts|connects when you finish setup/, 'the skill’s note would promise what this install does not do');
    host.handlers.click({ target: target({ '[data-connection-toggle]': {} }) });
    assert.match(text(cardOf(host.innerHTML)), /^Phone via Telegram @my_ouro_bot · starts after you allow it in Extensions Change$/);
    assert.equal(normalizeConnection(CARD).allow, false);
    assert.equal(normalizeConnection({ ...CARD, needs_allow: 'true' }).allow, false, 'only the host’s own true');
});

test('the host’s identity check: its two causes, its answer and anything else as a check that did not run', async () => {
    const answer = (status, body) => async (url, init) => {
        assert.equal(url, '/api/onboarding/connection/identity');
        assert.deepEqual(JSON.parse(init.body), { skill: 'telegram', token: TOKEN });
        return { ok: status < 400, status, json: async () => body };
    };
    assert.deepEqual(await checkIdentity('telegram', TOKEN, answer(200, { ok: true, username: 'my_ouro_bot' })), { status: 'ok', username: 'my_ouro_bot' });
    assert.deepEqual(await checkIdentity('telegram', TOKEN, answer(200, { ok: false, cause: 'refused' })), { status: 'refused' });
    assert.deepEqual(await checkIdentity('telegram', TOKEN, answer(200, { ok: false, cause: 'unreachable' })), { status: 'unreachable' });
    assert.deepEqual(await checkIdentity('telegram', TOKEN, answer(404, { error: 'Not Found' })), { status: 'unavailable', reason: 'HTTP 404' });
    assert.deepEqual(await checkIdentity('telegram', TOKEN, answer(200, { ok: true, username: 'not an account' })),
        { status: 'unavailable', reason: 'an answer it could not read' });
    const failed = await checkIdentity('telegram', TOKEN, async () => { throw new Error(`fetch ${TOKEN}`); });
    assert.deepEqual(failed, { status: 'unavailable', reason: 'the request did not complete' }, 'never the token or the error text');
    assert.notEqual(pairingCode(), pairingCode());
});

test('one muted line under the list counts the catalog’s other skills; none while the catalog is unread', async () => {
    const lines = [];
    for (const more of [24, 1, 0, null, undefined]) {
        const answer = { ...structuredClone(STARTER), ...(more === undefined ? {} : { catalog_more: more }) };
        const step = createStarterSkillsStep({ read: async () => answer });
        await step.loading;
        const host = fakeHost();
        step.mount(host);
        lines.push((host.innerHTML.match(/<p class="skills-starter-sub" data-starter-more[^>]*>([^<]*)<\/p>/) || [])[1] || '');
        if (more === 24) assert.match(host.innerHTML, /data-i18n-fmt="\{n\} more skills in the catalog, after start\."/);
    }
    assert.deepEqual(lines, ['24 more skills in the catalog, after start.', '1 more skill in the catalog, after start.', '', '', '']);
});

test('categories of one skill close the list together as «More»; a list without them has none', async () => {
    const pairs = structuredClone(STARTER);
    pairs.groups = [{ category: 'a', label: 'A', items: [{ id: 'h/a1', name: 'a1' }, { id: 'h/a2', name: 'a2' }] },
        { category: 'b', label: 'B', items: [{ id: 'h/b1', name: 'b1' }, { id: 'h/b2', name: 'b2' }] }];
    const even = createStarterSkillsStep({ read: async () => pairs });
    await even.loading;
    const host = fakeHost();
    even.mount(host);
    assert.deepEqual((host.innerHTML.match(/<h3[^>]*>[^<]*<\/h3>/g) || []).map((head) => text(head)), ['A', 'B']);
    pairs.groups.push({ category: 'c', label: 'C', items: [{ id: 'h/c1', name: 'c1' }] });
    const lone = createStarterSkillsStep({ read: async () => pairs });
    await lone.loading;
    const other = fakeHost();
    lone.mount(other);
    assert.deepEqual((other.innerHTML.match(/<h3[^>]*>[^<]*<\/h3>/g) || []).map((head) => text(head)), ['A', 'B', 'More']);
    assert.doesNotMatch(other.innerHTML, /<h3[^>]*>C</);
});
