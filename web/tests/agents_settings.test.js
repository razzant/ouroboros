// Settings › Agents (subagents_settings.js): the list of heads and one agent's page, against a fake settings server.
// Every change is one write of the roster through writeEntry and the checker's endpoint; the server's half answers
// here as PINS section 6 pins it (`_meta.tokens`, `if_unchanged`, a receipt in `changes`, its refusals).
import assert from 'node:assert/strict';
import test from 'node:test';

import { APP_SETTINGS_ORIGIN } from '../modules/api_client.js';
import { forgetSettingRows, readSettings } from '../modules/settings_controls.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { HAND_WRITTEN, ROWS, rowsOf } from '../modules/settings_rows.js';
import { agentName, rosterResult } from '../modules/subagents_settings.js';
import { recheckDraft, recheckSaved } from '../modules/safety_recheck.js';
import { SAFETY_MODEL_KEYS } from '../modules/consciousness_api.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

const KEY = 'OUROBOROS_SUBAGENTS';
const CHECKER = '/api/owner/safety-model';
const EMPTY_POOL = 'Every row marked Reviewer is switched off, so the main model would review alone.';
const TWINS = 'Subagent 3 runs the same engine as Subagent 2 — change its model, effort, access, account or processing.';
// The fixture's nodes keep attributes and properties apart: a drawn switch is on by its attribute, a pressed or
// painted one by its property.
NodeStub.prototype.toggleAttribute ??= function toggleAttribute(name, force) {
    if (force ?? !this.hasAttribute(name)) this.setAttribute(name, ''); else this.removeAttribute(name);
};
Object.defineProperty(NodeStub.prototype, 'attributes', { configurable: true,
    get() { return Object.entries(this.attrs).map(([name, value]) => ({ name, value })); } });
// A field's text: what was typed into it, else what it was drawn with.
const words = (field) => field.value ?? field.textContent;
const isOn = (input) => input.checked ?? input.hasAttribute('checked');
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = async (rounds = 12) => { for (let i = 0; i < rounds; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

const session = (id, target, extra = {}) => ({ subagent_id: id, recommended_use: `${id} notes`, route: { kind: 'agent_session', target_id: target }, ...extra });
const api = (id, target, extra = {}) => ({ subagent_id: id, recommended_use: `${id} notes`, route: { kind: 'api_model', target_id: target }, ...extra });
const ROSTER = () => [
    session('reviewer', 'codex=gpt-test', { review_eligible: true }),
    api('recheck', 'deepseek/deepseek-v5', { delivery: 'packet', rechecks_flags: true }),
    api('scout', 'openai::gpt-6-luna'),
];

// The settings server's roster half: the stored list, its token, and one queue of scripted answers.
function fakeServer(env, { items = ROSTER(), saved = true } = {}) {
    const server = { roster: { enabled: true, items }, saved, version: 0, posts: [], generic: [], answers: [], reads: 0,
        plain: { OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: '', OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT: 6, OUROBOROS_MAX_SUBAGENT_DEPTH: 3,
            OUROBOROS_SUBAGENT_WORKTREE_ROOT: '', OUROBOROS_SUBAGENT_PROJECTS_ROOT: '' } };
    server.token = () => `roster@${server.version}`;
    server.agent = (id) => server.roster.items.find((item) => item.subagent_id === id);
    server.document = () => ({
        [KEY]: server.saved ? JSON.stringify(server.roster) : '', ...server.plain,
        OUROBOROS_RUNTIME_MODE: 'pro',
        _meta: { tokens: { [KEY]: server.token() }, effects: { [KEY]: 'next_task', OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: 'next_task' },
            available_subagents: { source: server.saved ? 'configured' : 'undecided', candidate: server.saved ? null : server.roster } },
    });
    server.accept = (body) => {
        const roster = body.settings[KEY];
        const pool = roster.items.filter((item) => item.review_eligible === true && item.enabled !== false);
        if (roster.items.length && !pool.length && body.allow_empty_review_pool !== true) {
            return response({ saved: false, code: 'empty_review_pool', error: EMPTY_POOL }, 400);
        }
        Object.assign(server, { roster: structuredClone(roster), saved: true, version: server.version + 1 });
        return response({ ok: true, changes: { [KEY]: { withheld: 'structure', token: server.token() } } });
    };
    env.routes.set('/api/settings', async (_url, init = {}) => {
        if (init.method !== 'POST') { server.reads += 1; return response(server.document()); }
        const body = JSON.parse(init.body);
        server.generic.push(body);
        const changes = Object.fromEntries(Object.keys(body).filter((key) => key in server.plain)
            .map((key) => [key, { old: server.plain[key], new: body[key], token: `${key}@${server.generic.length}` }]));
        Object.assign(server.plain, Object.fromEntries(Object.entries(changes).map(([key, change]) => [key, change.new])));
        return response({ status: 'saved', changes });
    });
    env.routes.set(CHECKER, async (_url, init) => {
        const body = JSON.parse(init.body);
        server.posts.push({ body, headers: init.headers });
        if (body.if_unchanged?.[KEY] !== server.token()) {
            return response({ saved: false, code: 'settings_changed_elsewhere', error: 'The agents list changed elsewhere.' }, 409);
        }
        return (server.answers.shift() || server.accept)(body);
    });
    env.routes.set('/api/review-pool', response({ pool: [], last_executions: {}, row_costs: {} }));
    return server;
}

// The app's shell in one function: mount the registered page for a path into a body, as settings.js does.
async function open(t, path = 'agents', options = {}) {
    forgetSettingRows();
    const env = options.env || setup(t);
    const server = options.server || fakeServer(env, options);
    if (!options.env) {
        // The shared status store and the model catalogue read by themselves; neither is this page's subject.
        const fetch = globalThis.fetch;
        globalThis.fetch = async (url, init) => (/^\/api\/(claudexor|model-catalog)/.test(url) ? response({}, 503) : fetch(url, init));
        const createElement = env.document.createElement;
        env.document.createElement = (tag) => {
            const node = createElement(tag);
            if (tag === 'template') node.content = node;  // a template's content is where its markup is parsed
            return node;
        };
    }
    const container = new NodeStub();
    env.mount.appendChild(container);
    const asked = [];
    const answers = options.answers || [];
    const opened = [];
    const ctx = { readSettings, openSettings: (to) => { opened.push(to); return true; },
        ask: async (question) => { asked.push(question); return answers.length ? answers.shift() : true; } };
    const handle = settingsPageFor(path).mount(container, { path, row: '', ctx });
    await settle();
    const head = (id) => container.querySelector(`[data-entry-id="${id}"]`);
    return { env, server, container, asked, opened, handle, head,
        heads: () => container.querySelectorAll('[data-entry-id]').map((node) => node.dataset.entryId),
        toggle: (id) => head(id).querySelector('[data-entry-switch]'),
        result: () => (container.querySelector('[data-entry-result]') || container.querySelector('[data-editor-result]')).textContent,
        async press(input, checked) {
            input.checked = checked;
            await input.fire('change', { target: input });
            await settle();
        },
        async shown() {
            await env.window.fire('ouro:settings-page-shown', { detail: { path } });
            await settle();
        } };
}

test('the Agents page is one registered family: the list, an agent\'s page, a new agent', () => {
    const page = settingsPageFor('agents');
    assert.equal(page, settingsPageFor('agents/reviewer'));
    assert.equal(page, settingsPageFor('agents/new'));
    assert.equal(page.title('agents'), 'Agents');
    assert.equal(page.title('agents/new'), 'New agent');
    // The roster is one of the checker's keys: its row names the checker's writer, and its page draws it by hand.
    const roster = ROWS.find((row) => row.id === KEY);
    assert.ok(SAFETY_MODEL_KEYS.includes(KEY));
    assert.deepEqual([roster.writer, roster.control.kind, roster.page, roster.fold], ['checker', 'custom', 'agents', false]);
    assert.ok(roster.aliases.includes('Delegation'));
    assert.deepEqual(rowsOf('agents', { fold: true }).map((row) => row.keys[0]), ['OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS',
        'OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT', 'OUROBOROS_MAX_SUBAGENT_DEPTH', 'OUROBOROS_SUBAGENT_WORKTREE_ROOT',
        'OUROBOROS_SUBAGENT_PROJECTS_ROOT']);
    assert.equal(HAND_WRITTEN.some((row) => row.keys.includes(KEY)), false, 'one key, one control: the roster is described once');
});

test('an agent is named by its model in short and the facets its handle spells', () => {
    assert.equal(agentName(api('a', 'deepseek/deepseek-v5')), 'deepseek-v5');
    assert.equal(agentName(api('a', 'openai::gpt-6-luna', { effort: 'high' })), 'gpt-6-luna/high');
    assert.equal(agentName(session('a', 'codex=gpt-test', { access: 'readonly', route: { kind: 'agent_session', target_id: 'codex=gpt-test', credential_profile_id: 'work' } })),
        'gpt-test/readonly/@work');
    assert.equal(agentName(session('a', 'codex')), 'codex', 'a session on its engine\'s default model is named by the engine');
    assert.equal(agentName(api('a', '')), 'New agent');
});

test('the list draws one head per agent with its switch, and the roster\'s own switch above it', async (t) => {
    const page = await open(t);
    assert.deepEqual(page.heads(), ['reviewer', 'recheck', 'scout']);
    assert.deepEqual(page.heads().map((id) => isOn(page.toggle(id))), [true, true, true]);
    assert.match(page.head('reviewer').querySelector('.ui-row-title').textContent, /^gpt-testReviewer/);
    assert.match(page.head('scout').querySelector('.ui-row-meta').textContent, /scout notes$/);
    assert.equal(page.head('scout').querySelectorAll('textarea').length, 0, 'a head holds no form');
    assert.equal(page.container.querySelectorAll('[data-subagent-field="recommended_use"]').length, 0);
    const delegation = page.container.querySelector(`[data-setting-row="${KEY}"]`).querySelector('[data-setting-control]');
    assert.equal(isOn(delegation), true);
    assert.equal(page.container.querySelector('.agents-card-title').textContent, '3 agents');
    assert.equal(page.server.posts.length, 0, 'showing the list writes nothing');
    // A head opens its agent's page; Add opens an unsaved one.
    await page.head('scout').querySelector('[data-subagent-field="open"]').fire('click', {});
    await page.container.querySelector('[data-subagent-add]').fire('click', {});
    assert.deepEqual(page.opened, ['agents/scout', 'agents/new']);
});

test('a plain row\'s switch writes that one entry at once through the checker\'s endpoint, with the read\'s token', async (t) => {
    const page = await open(t);
    await page.press(page.toggle('scout'), false);
    assert.equal(page.server.posts.length, 1);
    const { body, headers } = page.server.posts[0];
    assert.deepEqual(Object.keys(body.settings), [KEY]);
    assert.deepEqual(body.if_unchanged, { [KEY]: 'roster@0' });
    assert.equal(headers['X-Ouroboros-Surface'], APP_SETTINGS_ORIGIN['X-Ouroboros-Surface']);
    assert.deepEqual(body.settings[KEY].items, [...ROSTER().slice(0, 2), { ...ROSTER()[2], enabled: false }], 'only its entry changed');
    assert.equal(page.server.generic.length, 0, 'never the generic save: outside Cyber Pro it refuses a change of the checker');
    assert.equal(isOn(page.toggle('scout')), false);
    assert.equal(page.result(), 'Saved · applies to new tasks');
    assert.equal(page.asked.length, 0);
    // On again: the stored entry loses the flag it only carries while off.
    await page.press(page.toggle('scout'), true);
    assert.equal(page.server.posts[1].body.settings[KEY].items[2].enabled, true);
    assert.equal(isOn(page.toggle('scout')), true);
});

test('a row with the re-check mark and the roster\'s own switch take the same owner transaction', async (t) => {
    const page = await open(t);
    await page.press(page.toggle('recheck'), false);
    assert.equal(page.server.posts.length, 1);
    assert.equal(page.server.generic.length, 0);
    assert.deepEqual(page.server.agent('recheck'), { ...ROSTER()[1], enabled: false }, 'the mark stays with its row');
    const delegation = page.container.querySelector(`[data-setting-row="${KEY}"]`).querySelector('[data-setting-control]');
    await page.press(delegation, false);
    assert.equal(page.server.posts.length, 2);
    assert.equal(page.server.roster.enabled, false);
    assert.equal(page.server.roster.items.length, 3, 'the roster\'s own switch changes no entry');
    assert.equal(page.server.generic.length, 0);
    assert.equal(isOn(delegation), false);
    assert.equal(page.container.querySelector(`[data-setting-row="${KEY}"]`).querySelector('[data-setting-result]').textContent,
        'Saved · applies to new tasks');
    assert.equal(page.container.querySelector('[data-review-pool-stays]').textContent, 'Helpers are off; agents marked Reviewer still review.');
});

test('the agent page owns the second-check binding and releases it when left', async (t) => {
    const page = await open(t, 'agents/recheck');
    const mark = page.container.querySelector('[data-recheck-mark="recheck"]');
    assert.ok(mark, 'the saved second-check switch participates in the shared binding');
    assert.equal(recheckSaved(), 'recheck');
    mark.checked = false;
    // The fixture dispatches capture and target separately, as the browser does in that order.
    await page.container.fire('change', { target: mark });
    assert.equal(recheckDraft(), '', 'the page itself binds the shared one-mark rule');
    await page.press(mark, false);
    assert.equal(page.server.agent('recheck').rechecks_flags, false);
    assert.equal(recheckSaved(), '', 'the completed read resets the mark from the stored roster');
    assert.equal(recheckDraft(), null);
    page.handle.unmount();
    mark.checked = true;
    await page.container.fire('change', { target: mark });
    assert.equal(recheckDraft(), null, 'a detached page cannot change the shared mark');
});

test('a toggle on a roster that already holds a twin pair is sent, and the server\'s answer is shown as it is', async (t) => {
    const items = [...ROSTER(), api('scout-twin', 'openai::gpt-6-luna')];
    const page = await open(t, 'agents', { items });
    // The server judges twins only when THIS save changes the roster: here it refuses for a pair the press never touched.
    page.server.answers.push(() => response({ saved: false, error: TWINS }, 400));
    await page.press(page.toggle('reviewer'), false);
    assert.equal(page.server.posts.length, 1, 'sent once, never retried');
    assert.equal(page.result(), `Not saved. ${TWINS}`);
    assert.equal(isOn(page.toggle('reviewer')), true, 'the switch shows what is stored');
    assert.equal(page.asked.length, 0);
    // Accepted, it is saved like any other.
    await page.press(page.toggle('scout'), false);
    assert.equal(page.server.agent('scout').enabled, false);
    assert.equal(page.result(), 'Saved · applies to new tasks');
});

test('switching the last reviewer off asks, then sends the SAME body again with the flag', async (t) => {
    const page = await open(t);
    await page.press(page.toggle('reviewer'), false);
    assert.equal(page.asked.length, 1);
    assert.equal(page.asked[0].body, 'No agent marked Reviewer would be on, so the main model reviews alone: one model, no second opinion.');
    assert.equal(page.asked[0].title, 'Save without reviewers?');
    assert.equal(page.server.posts.length, 2);
    assert.deepEqual(page.server.posts[1].body, { ...page.server.posts[0].body, allow_empty_review_pool: true });
    assert.equal(page.server.agent('reviewer').enabled, false);
    assert.equal(isOn(page.toggle('reviewer')), false);

    // No reviewer is on now: the next changed roster is judged again and asks for ITSELF; a No writes nothing more.
    const declined = await open(t, 'agents', { env: page.env, server: page.server, answers: [false] });
    await declined.press(declined.toggle('scout'), false);
    assert.equal(declined.asked.length, 1);
    assert.equal(page.server.posts.length, 3, 'the refused body, and no second one');
    assert.equal(page.server.posts[2].body.allow_empty_review_pool, undefined, 'a confirmation is never carried to another roster');
    assert.equal(page.server.agent('scout').enabled, undefined);
    assert.equal(isOn(declined.toggle('scout')), true, 'the switch returns to what is stored');
    assert.equal(declined.result(), '');
});

test('a roster that was never saved is written whole on the first press', async (t) => {
    const page = await open(t, 'agents', { saved: false });
    assert.deepEqual(page.heads(), ['reviewer', 'recheck', 'scout'], 'the list shows the candidate the server proposes');
    await page.press(page.toggle('scout'), false);
    assert.equal(page.server.posts.length, 1);
    // The list as shown, in its canonical spelling (a session's access is written out), with the one entry changed.
    assert.deepEqual(page.server.posts[0].body.settings[KEY], { enabled: true,
        items: [{ ...ROSTER()[0], access: 'full' }, ROSTER()[1], { ...ROSTER()[2], enabled: false }] });
    assert.deepEqual(page.server.posts[0].body.if_unchanged, { [KEY]: 'roster@0' }, 'guarded by the token of the unsaved value');
    assert.equal(page.server.saved, true);
    assert.deepEqual(page.heads().map((id) => isOn(page.toggle(id))), [true, true, false]);
});

test('a list that changed elsewhere is refused, repainted, and nothing is retried', async (t) => {
    const page = await open(t);
    // Another device removes an agent and switches one off between this page's read and the write's own read...
    page.server.answers.push((body) => {
        Object.assign(page.server, { roster: { enabled: true, items: [{ ...ROSTER()[0] }, { ...ROSTER()[2], enabled: false }] }, version: 7 });
        return response({ saved: false, code: 'settings_changed_elsewhere', error: 'The agents list changed elsewhere.', sent: body.if_unchanged }, 409);
    });
    await page.press(page.toggle('recheck'), false);
    assert.equal(page.server.posts.length, 1, 'one write, no silent retry');
    assert.equal(page.result(), 'The list changed elsewhere. This is how it stands now.');
    assert.deepEqual(page.heads(), ['reviewer', 'scout'], 'the page shows the list the server holds');
    assert.equal(isOn(page.toggle('scout')), false);
    // ...and a press for an agent that is gone from the fresh read sends nothing at all.
    const stale = await open(t, 'agents', { env: page.env, server: page.server });
    page.server.roster.items.pop();
    page.server.version += 1;
    await stale.press(stale.toggle('scout'), true);
    assert.equal(page.server.posts.length, 1);
    assert.equal(stale.result(), 'The list changed elsewhere. This is how it stands now.');
    assert.deepEqual(stale.heads(), ['reviewer']);
});

test('an agent\'s page: a switch never publishes its unsaved form, and Save writes the form into its own entry only', async (t) => {
    const list = await open(t);
    const page = await open(t, 'agents/scout', { env: list.env, server: list.server });
    const field = (name) => page.container.querySelector(`[data-subagent-field="${name}"]`);
    assert.deepEqual(page.container.querySelectorAll('.agents-card-title').map((node) => node.textContent), ['In use', 'What it is good at', 'Runs on']);
    assert.equal(field('enabled').getAttribute('role'), 'switch', 'a saved agent\'s on/off acts at once');
    assert.equal(field('rechecks_flags'), null, 'only an API row with Packet delivery, or the marked one, may take the re-check');
    // The Reviewer row says what the mark means under the saved Access.
    assert.match(page.container.querySelector('.agent-page').textContent, /Counts when chosen: for each check Ouroboros asks one, several or all reviewers/);
    field('recommended_use').value = 'UNSAVED WORDS';
    await field('recommended_use').fire('input', { target: field('recommended_use') });

    // The list's switch on another agent, then this page's own Reviewer mark: neither carries the unsaved words.
    await list.press(list.toggle('recheck'), false);
    await page.press(field('review_eligible'), true);
    assert.equal(list.server.posts.length, 2);
    assert.equal(JSON.stringify(list.server.posts).includes('UNSAVED WORDS'), false);
    assert.equal(list.server.agent('scout').review_eligible, true);
    assert.equal(list.server.agent('scout').recommended_use, 'scout notes');
    assert.equal(words(field('recommended_use')), 'UNSAVED WORDS', 'the form keeps its text through the repaint');

    // Elsewhere the agent is switched off; this page's Save then keeps the stored on/off and marks, and its own text.
    Object.assign(list.server.agent('scout'), { enabled: false });
    list.server.version += 1;
    await page.container.querySelector('[data-editor-save]').fire('click', { target: page.container.querySelector('[data-editor-save]') });
    await settle();
    assert.equal(list.server.posts.length, 3);
    assert.deepEqual(list.server.agent('scout'), { ...ROSTER()[2], recommended_use: 'UNSAVED WORDS', review_eligible: true, enabled: false });
    assert.deepEqual(list.server.agent('recheck'), { ...ROSTER()[1], enabled: false }, 'no other entry is rewritten');
    assert.equal(page.result(), 'Saved · applies to new tasks');
    assert.equal(isOn(field('enabled')), false, 'the page shows the stored value after its Save');
});

test('a page that is left keeps its unsaved form for the next opening; Remove asks first and leaves the page', async (t) => {
    const first = await open(t, 'agents/scout');
    const field = (page, name) => page.container.querySelector(`[data-subagent-field="${name}"]`);
    field(first, 'recommended_use').value = 'KEPT WORDS';
    await field(first, 'recommended_use').fire('input', { target: field(first, 'recommended_use') });
    first.handle.unmount();
    const again = await open(t, 'agents/scout', { env: first.env, server: first.server, answers: [false, true] });
    assert.equal(words(field(again, 'recommended_use')), 'KEPT WORDS');
    assert.equal(first.server.posts.length, 0);

    const remove = again.container.querySelector('[data-editor-remove]');
    await remove.fire('click', { target: remove });
    await settle();
    assert.equal(again.asked[0].title, 'Remove gpt-6-luna?');
    assert.equal(first.server.posts.length, 0, 'No removes nothing');
    await remove.fire('click', { target: remove });
    await settle();
    assert.deepEqual(first.server.roster.items.map((item) => item.subagent_id), ['reviewer', 'recheck']);
    assert.deepEqual(again.opened, ['agents']);
    again.handle.unmount();
    const fresh = await open(t, 'agents', { env: first.env, server: first.server });
    assert.deepEqual(fresh.heads(), ['reviewer', 'recheck']);
});

test('a new agent is an unsaved form whose on/off is a field of it; its Save adds it and returns to the list', async (t) => {
    const page = await open(t, 'agents/new');
    const field = (name) => page.container.querySelector(`[data-subagent-field="${name}"]`);
    assert.equal(field('enabled').getAttribute('role'), null, 'a checkbox of the form, not a switch that writes');
    assert.equal(field('enabled').classList.contains('ui-checkbox'), true);
    await page.press(field('enabled'), false);
    assert.equal(page.server.posts.length, 0, 'the on/off of an unsaved agent writes nothing');
    const save = page.container.querySelector('[data-editor-save]');
    await save.fire('click', { target: save });
    await settle();
    assert.equal(page.server.posts.length, 0, 'an agent without a model is not sent');
    assert.match(page.container.querySelector('[data-subagents-validation]').textContent, /New agent needs a model or agent-session route\./);
    // The form's own listener takes the typed model (the chooser's suggestion list is a browser matter).
    field('model').value = 'qwen/qwen3.8-max';
    field('model').events.get('input')[0]({ target: field('model') });
    await save.fire('click', { target: save });
    await settle();
    assert.equal(page.server.posts.length, 1);
    const added = page.server.roster.items.at(-1);
    assert.deepEqual({ ...added, subagent_id: '' }, { subagent_id: '', recommended_use: '', route: { kind: 'api_model', target_id: 'qwen/qwen3.8-max' }, enabled: false });
    assert.equal(page.server.roster.items.length, 4);
    assert.deepEqual(page.opened, ['agents']);
});

test('Helpers may change files is a strip over the stored spelling: Auto is the unset value, and a receipt reads back in its words', async (t) => {
    const page = await open(t);
    const fold = page.container.querySelector('[data-agents-fold]');
    const row = fold.querySelector('[data-setting-row="OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS"]');
    const option = (label) => row.querySelectorAll('[data-setting-value]').find((button) => button.textContent === label);
    const pressed = () => row.querySelectorAll('[data-setting-value]').filter((button) => button.getAttribute('aria-pressed') === 'true')
        .map((button) => button.textContent);
    assert.deepEqual(row.querySelectorAll('[data-setting-value]').map((button) => button.textContent), ['Off', 'Auto', 'On']);
    assert.deepEqual(pressed(), ['Auto']);
    await fold.fire('click', { target: option('On') });
    await settle();
    assert.deepEqual(page.server.generic, [{ OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: 'true' }], 'its one key, by the generic save');
    assert.deepEqual(pressed(), ['On']);
    assert.equal(row.querySelector('[data-setting-result]').textContent, 'Saved · applies to new tasks · was Auto · Undo');
    await fold.fire('click', { target: option('Auto') });
    await settle();
    assert.deepEqual(page.server.generic[1], { OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: '' }, 'Auto is stored as nothing');
    assert.equal(row.querySelector('[data-setting-result]').textContent, 'Saved · applies to new tasks · was On · Undo');
    // A value written long ago in another spelling reads as the same choice.
    page.server.plain.OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS = 'off';
    await page.shown();
    assert.deepEqual(pressed(), ['Off']);
    assert.equal(page.server.posts.length, 0, 'the roster is no part of these rows');
});

test('a roster write says Saved with when it applies, a refusal with the server\'s sentence, and nothing for no change', () => {
    assert.deepEqual(rosterResult({ state: 'saved', effect: 'next_task', withheld: 'structure', token: 't', message: '' }),
        { text: 'Saved · applies to new tasks', tone: 'ok', details: '' });
    assert.equal(rosterResult({ state: 'saved', effect: 'next_task', withheld: '', message: '' }).text, 'Saved', 'no receipt: no claim about when');
    assert.equal(rosterResult({ state: 'refused', effect: 'next_task', message: TWINS }).text, `Not saved. ${TWINS}`);
    assert.equal(rosterResult({ state: 'unknown', effect: 'next_task', message: 'timeout' }).text, 'Not confirmed yet');
    assert.deepEqual(rosterResult({ state: 'unchanged' }), {});
});
