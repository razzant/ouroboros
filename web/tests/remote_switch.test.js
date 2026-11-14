// Remote in a room's bar (remote_switch.js, DESIGN "Page header"): the capsule says the saved preference and, beside
// it, what it moves now (the server's `remote_execution` projection, the resolver dispatch uses); a press with no engine
// opens the Remote engines setup, a configured target toggles at once, an unmade choice asks and never takes the first;
// a write is the ordinary settings save, and the capsule shows only what the server confirms.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { createRemoteSwitch, chooserNotes, REMOTE_ROW, remoteValue, remoteView } from '../modules/remote_switch.js';
import { createStateSnapshotSequencer, readStateSnapshot } from '../modules/state_snapshots.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { flush } from './state_read_fixture.js';

const engine = (name, label = name, enabled = true) => ({ name, label, enabled });
const state = ({ on = false, target = '', engines = [engine('vds', 'VDSina')], scope = 'follows',
    model = 'claudexor::codex=gpt-6', effective = null } = {}) => ({
    on, engine: target, engines,
    main: { model, use_local: false, scope, effective: effective || { model: on && scope === 'follows' ? `claudexor@${target}::codex=gpt-6` : model, use_local: false } },
});

test('the stored value is the server\'s canonical JSON text', () => {
    assert.equal(remoteValue(true, 'vds'), '{"engine":"vds","on":true}');
    assert.equal(remoteValue(false, 'vds'), '{"engine":"vds","on":false}');
    assert.deepEqual(REMOTE_ROW.keys, ['OUROBOROS_REMOTE_EXECUTION']);
});

test('no engine saved: the capsule is Off and its press opens the setup', () => {
    const view = remoteView(state({ engines: [] }));
    assert.deepEqual([view.configured, view.on, view.word, view.press], [false, false, 'Off', 'setup']);
    assert.match(view.said, /^Remote: no remote engine is set up\./);
    assert.match(view.said, /Press to set one up in Settings$/);
    assert.equal(remoteView(null), null, 'an absent projection hides the capsule, never a guessed Off');
    assert.equal(remoteView({ on: true }), null);
});

test('one engine: Off turns On for it in one press, On turns Off; Main moves only when it follows', () => {
    const off = remoteView(state());
    assert.deepEqual([off.press, off.target.name, off.word, off.mixed], ['on', 'vds', 'Off', false]);
    assert.match(off.said, /Main and new agents run on this computer\./);
    assert.match(off.said, /Press to run Main and new agents on VDSina$/);
    const on = remoteView(state({ on: true, target: 'vds' }));
    assert.deepEqual([on.press, on.word, on.mixed, on.mainMoves, on.mark.shape], ['off', 'VDSina', false, true, 'none']);
    assert.match(on.said, /^Remote: on, VDSina\. Main and new agents run on VDSina\./);
    assert.match(on.said, /that engine's own accounts \(Auto\)/);
    assert.match(on.said, /Press to run Main and new agents on this computer$/);
});

test('On that leaves Main where it is says so: API key, local model, a Main naming its own engine', () => {
    const api = remoteView(state({ on: true, target: 'vds', scope: 'api', model: 'openrouter/x' }));
    assert.deepEqual([api.mixed, api.mainMoves], [true, false]);
    assert.match(api.said, /New agents run on VDSina; Main uses an API key and stays as it is\./);
    assert.match(api.said, /Press to run new agents on this computer$/, 'a press moves only what it moves');
    const local = remoteView(state({ on: true, target: 'vds', scope: 'local', model: 'qwen (local)' }));
    assert.match(local.said, /Main uses the local model and stays on this computer/);
    const named = remoteView(state({ on: true, target: 'vds', scope: 'engine', model: 'claudexor@lab::codex=gpt-6',
        engines: [engine('vds', 'VDSina'), engine('lab', 'Lab box')] }));
    assert.match(named.said, /New agents run on VDSina; Main runs on Lab box, as set in Models\./);
    // Off can leave an explicit remote Main where it is: the capsule never claims this computer for it.
    const offNamed = remoteView(state({ scope: 'engine', model: 'claudexor@lab::codex=gpt-6',
        engines: [engine('vds', 'VDSina'), engine('lab', 'Lab box')] }));
    assert.match(offNamed.said, /New agents run on this computer; Main runs on Lab box, as set in Models\./);
});

test('several engines: an unmade choice asks, a remembered usable one toggles at once, a stale one asks again', () => {
    const engines = [engine('vds', 'VDSina'), engine('lab', 'Lab box')];
    assert.equal(remoteView(state({ engines })).press, 'choose', 'never the first by default');
    const remembered = remoteView(state({ engines, target: 'lab' }));
    assert.deepEqual([remembered.press, remembered.target.name], ['on', 'lab']);
    const switchedOff = remoteView(state({ engines: [engine('vds', 'VDSina'), engine('lab', 'Lab box', false)], target: 'lab' }));
    assert.equal(switchedOff.press, 'choose', 'a remembered engine switched off is not replaced silently');
    const gone = remoteView(state({ engines, target: 'old' }));
    assert.equal(gone.press, 'choose');
    assert.equal(remoteView(state({ engines: [engine('vds', 'VDSina', false)] })).press, 'choose', 'all off: the chooser says why');
});

test('On whose target takes no new work is the owner\'s move: the amber dot and the reason', () => {
    const off = remoteView(state({ on: true, target: 'vds', engines: [engine('vds', 'VDSina', false)] }));
    assert.deepEqual([off.targetProblem, off.mark.shape, off.mark.hue, off.press], ['off', 'dot', 'owner', 'off']);
    assert.match(off.said, /switched off in Settings: new work for it is refused/);
    const removed = remoteView(state({ on: true, target: 'gone', engines: [engine('vds', 'VDSina')] }));
    assert.deepEqual([removed.targetProblem, removed.word], ['removed', 'gone']);
    assert.match(removed.said, /gone is no longer saved: new work for it is refused until you choose again/);
});

test('On whose every engine was removed still turns Off in one press, never behind the setup', () => {
    const gone = remoteView(state({ on: true, target: 'gone', engines: [] }));
    assert.deepEqual([gone.on, gone.press, gone.chooser, gone.configured, gone.word], [true, 'off', true, false, 'gone']);
    assert.deepEqual([gone.mark.shape, gone.mark.hue], ['dot', 'owner']);
    assert.match(gone.said, /^Remote: on, gone\./);
    assert.match(gone.said, /gone is no longer saved: new work for it is refused until you choose again\. No remote engine is saved now\./);
    assert.match(gone.said, /Press to run Main and new agents on this computer$/);
    assert.equal(remoteView(state({ engines: [] })).press, 'setup', 'only Off with nothing saved opens the setup');
});

test('the chooser says what Emergency Stop and Off do and what a remote engine refuses', () => {
    const notes = chooserNotes(remoteView(state({ on: true, target: 'vds' }))).join(' ');
    assert.match(notes, /Emergency Stop closes the tunnel at once; work already on VDSina may go on there/);
    assert.match(notes, /plan turns, in-place edits and folders that are not Git projects are refused/);
    assert.match(notes, /A press moves Main at its next call and the agents started after it; agents already running keep their engine/);
    assert.doesNotMatch(notes, /only new work/, 'Main-following work moves at its next call, not only new work');
    assert.doesNotMatch(chooserNotes(remoteView(state())).join(' '), /Emergency Stop/);
});

function mount({ commit, openSettings = () => {} } = {}) {
    const env = installDom();
    const bar = document.createElement('div');
    const chip = document.createElement('button');
    chip.className = 'cx-chip';
    bar.append(chip);
    const toasts = [];
    let refreshed = 0;
    const remote = createRemoteSwitch({ bar, openSettings, refreshState: () => { refreshed += 1; },
        commit, notify: (text, tone) => toasts.push([text, tone]) });
    return { env, bar, remote, toasts, refreshed: () => refreshed };
}

test('mounted after the consciousness chip, hidden until the projection arrives; setup opens Remote engines', () => {
    const opened = [];
    const { env, bar, remote } = mount({ openSettings: (path, options) => opened.push([path, options]),
        commit: async () => assert.fail('no write without a target') });
    try {
        assert.equal(bar.children.indexOf(remote.el), 1, 'right after the chip');
        assert.equal(remote.el.hidden, true);
        remote.update({ accounting: {} });
        assert.equal(remote.el.hidden, true, 'an answer without the projection says nothing');
        remote.update({ remote_execution: state({ engines: [] }) });
        assert.equal(remote.el.hidden, false);
        const toggle = remote.el.querySelector('.rx-toggle');
        assert.equal(toggle.getAttribute('aria-pressed'), 'false');
        assert.equal(remote.el.querySelector('.rx-more').hidden, true, 'no chooser without an engine');
        toggle.click();
        assert.deepEqual(opened, [['accounts', { row: 'OUROBOROS_ENGINE_CONNECTIONS' }]]);
    } finally { remote.destroy(); restoreDom(env); }
});

test('one press writes the toggle; the capsule says Saving until a read after the write, never a false On', async () => {
    const writes = [];
    let answer = { state: 'saved' };
    const { env, remote, refreshed } = mount({ commit: async (row, value, options) => { writes.push([row.id, value, options]); return answer; } });
    try {
        remote.update({ remote_execution: state() });
        const toggle = remote.el.querySelector('.rx-toggle');
        const word = remote.el.querySelector('.rx-word');
        toggle.click();
        assert.equal(word.textContent, 'Saving…');
        assert.equal(toggle.getAttribute('aria-busy'), 'true');
        toggle.click();  // a press while the write is on its way sends nothing more
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(writes, [['OUROBOROS_REMOTE_EXECUTION', '{"engine":"vds","on":true}', { confirmed: true }]]);
        assert.equal(refreshed(), 1, 'the next state answer says what the server shows');
        assert.equal(remote.el.dataset.on, 'false', 'not On before the server shows it');
        remote.update({ remote_execution: state({ on: true, target: 'vds' }) });
        assert.deepEqual([word.textContent, remote.el.dataset.on, toggle.getAttribute('aria-pressed')], ['VDSina', 'true', 'true']);
        toggle.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(writes[1][1], '{"engine":"vds","on":false}', 'Off keeps the remembered engine');
        remote.update({ remote_execution: state({ target: 'vds' }) });
        assert.equal(word.textContent, 'Off');
        answer = { state: 'refused', message: 'The remote engine VDSina is switched off in Settings…' };
        toggle.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(word.textContent, 'Off', 'a refusal leaves the confirmed state');
        assert.equal(toggle.getAttribute('aria-busy'), 'false');
    } finally { remote.destroy(); restoreDom(env); }
});

test('the removed-target falsifier in the DOM: the press writes Off, the chevron stays, no setup opens', async () => {
    const writes = [];
    const opened = [];
    const { env, remote } = mount({ commit: async (row, value) => { writes.push(value); return { state: 'saved' }; },
        openSettings: (...args) => opened.push(args) });
    try {
        remote.update({ remote_execution: state({ on: true, target: 'gone', engines: [] }) });
        assert.equal(remote.el.querySelector('.rx-more').hidden, false, 'the chooser stays reachable while On');
        assert.equal(remote.el.dataset.chooser, 'true');
        remote.el.querySelector('.rx-toggle').click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual([writes, opened], [['{"engine":"gone","on":false}'], []]);
    } finally { remote.destroy(); restoreDom(env); }
});

test('a refused or unconfirmed write is said, and the capsule keeps what the server last showed', async () => {
    const outcomes = [{ state: 'refused', message: 'No remote engine named \'vds\' is saved…' }, { state: 'unknown', message: 'timeout' }];
    const { env, remote, toasts } = mount({ commit: async () => outcomes.shift() });
    try {
        remote.update({ remote_execution: state() });
        const toggle = remote.el.querySelector('.rx-toggle');
        toggle.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        toggle.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(toasts, [['No remote engine named \'vds\' is saved…', 'danger'], ['Remote change not confirmed: timeout', 'warn']]);
        assert.equal(remote.el.dataset.on, 'false');
    } finally { remote.destroy(); restoreDom(env); }
});

test('the mixed scope is visible in the capsule, and the target problem is its one mark', () => {
    const { env, remote } = mount({ commit: async () => ({ state: 'saved' }) });
    try {
        remote.update({ remote_execution: state({ on: true, target: 'vds', scope: 'api', model: 'openrouter/x' }) });
        assert.equal(remote.el.dataset.mixed, 'true');
        assert.equal(remote.el.querySelector('.rx-scope').hidden, false);
        assert.equal(remote.el.querySelector('.rx-scope').textContent, '· agents');
        remote.update({ remote_execution: state({ on: true, target: 'vds', engines: [engine('vds', 'VDSina', false)] }) });
        assert.deepEqual([remote.el.querySelector('.rx-mark').dataset.shape, remote.el.querySelector('.rx-mark').dataset.hue], ['dot', 'owner']);
    } finally { remote.destroy(); restoreDom(env); }
});

// The page's own state reads (state_snapshots.js: the one sequencer and its reader) over a transport that answers
// when told, so a read begun before a save can answer after the save was confirmed (Astra review, 14.11.2026).
function mountOnReads(t) {
    const calls = [];
    const open = (call) => !call.answered && !call.signal?.aborted;
    const env = installDom((_url, init = {}) => new Promise((resolve) => {
        const call = { signal: init.signal, answered: false };
        call.answer = (body) => { call.answered = true; resolve({ ok: true, status: 200, json: async () => body }); };
        calls.push(call);
    }));
    const events = new EventTarget();  // the window the switch listens on for the page's saved settings
    globalThis.document.defaultView = events;
    const bar = document.createElement('div');
    const chip = document.createElement('button');
    chip.className = 'cx-chip';
    bar.append(chip);
    let remote = null;
    const snapshots = createStateSnapshotSequencer((data) => remote?.update(data));
    const read = async (force = false) => {
        const request = await snapshots.gate(force);
        if (request) await readStateSnapshot(snapshots, request);
    };
    const page = { calls, read, writes: [], retired: 0, outcome: { state: 'saved' } };
    // The settings writer's order (settings_controls.js drain): the outcome, then the page-wide event of a landed save.
    const commit = async (row, value) => {
        page.writes.push(value);
        if (page.outcome.state === 'saved') events.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { keys: row.keys } }));
        return page.outcome;
    };
    remote = createRemoteSwitch({ bar, refreshState: read, commit, notify: () => {},
        retireState: () => { page.retired += 1; snapshots.retire(); } });
    page.remote = remote;
    page.open = () => calls.filter(open);
    page.answerOpen = async (body) => {  // every read now open, and the coalesced reads they release
        for (let reads = page.open(); reads.length; reads = page.open()) {
            reads.forEach((call) => call.answer(body));
            await flush();
        }
    };
    t.after(async () => { remote.destroy(); await page.answerOpen({}); restoreDom(env.prior); });
    return page;
}

for (const start of [false, true]) {
    test(`a read begun before a confirmed save never confirms it: ${start ? 'On to Off' : 'Off to On'}, then the owner's reversal`, async (t) => {
        const page = mountOnReads(t);
        const shown = (on) => ({ remote_execution: state(on ? { on: true, target: 'vds' } : { target: 'vds' }) });
        const toggle = page.remote.el.querySelector('.rx-toggle');
        const word = page.remote.el.querySelector('.rx-word');
        const says = () => [word.textContent, page.remote.el.dataset.on, toggle.getAttribute('aria-busy')];
        void page.read();
        await flush();
        await page.answerOpen(shown(start));
        assert.deepEqual(says(), [start ? 'VDSina' : 'Off', String(start), 'false']);
        for (const [press, on] of [[1, !start], [2, start]]) {  // the press, then the owner's reversal
            void page.read();  // the page's poll, begun before this save: it answers what was saved before it
            await flush();
            const [stale] = page.open();
            toggle.click();
            await flush();
            assert.deepEqual(page.writes, [remoteValue(!start, 'vds'), remoteValue(start, 'vds')].slice(0, press));
            stale.answer(shown(!on));
            await flush();
            assert.deepEqual(says(), ['Saving…', String(!on), 'true'], 'a read begun before the save never confirms it');
            toggle.click();
            await flush();
            assert.equal(page.writes.length, press, 'and no second press is taken meanwhile');
            assert.equal(page.retired, press, 'each confirmed save retires the reads begun before it');
            await page.answerOpen(shown(on));  // the read begun after the save
            assert.deepEqual(says(), [on ? 'VDSina' : 'Off', String(on), 'false']);
        }
        // A save that did not land changes nothing: the read in flight stands and answers.
        page.outcome = { state: 'refused', message: 'No.' };
        void page.read();
        await flush();
        toggle.click();
        await flush();
        assert.equal(page.retired, 2, 'a refusal retires nothing');
        assert.equal(page.open().length, 1, 'the read in flight goes on');
        await page.answerOpen(shown(start));
        assert.deepEqual(says(), [start ? 'VDSina' : 'Off', String(start), 'false']);
    });
}

test('Main\'s bar: the real settings save landing retires the page\'s read begun before it (chat.js)', async (t) => {
    const reads = [];
    const posts = [];
    const reply = (body) => ({ ok: true, status: 200, json: async () => body });
    const env = installDom(async (url, init = {}) => {
        url = String(url);
        if (url.startsWith('/api/state')) {
            return new Promise((resolve) => {
                const call = { signal: init.signal, answered: false };
                call.answer = (body) => { call.answered = true; resolve(reply(body)); };
                reads.push(call);
            });
        }
        if (url === '/api/settings' && init.method === 'POST') { posts.push(JSON.parse(init.body)); return reply({}); }
        if (url === '/api/settings') return reply({ _meta: { tokens: {}, effects: {} } });
        return reply(url.startsWith('/api/chat/history') ? { messages: [], window: { complete: true } } : { active_direct_turns: [] });
    });
    const socket = { connect() {}, on: () => () => {}, isConnected: () => true, send: () => ({ status: 'sent' }) };
    let chat = null;
    const stateSnapshots = createStateSnapshotSequencer((data, at) => chat?.hydrateStateSnapshot(data, at));
    chat = createChatInstance({ ws: socket, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots, chatId: 1, idPrefix: 'chat', mountEl: env.mount });
    const answerOpen = async (body) => {
        for (let open = reads.filter((call) => !call.answered && !call.signal?.aborted); open.length;
            open = reads.filter((call) => !call.answered && !call.signal?.aborted)) {
            open.forEach((call) => call.answer(body));
            await flush();
        }
    };
    t.after(async () => { chat.destroy(); await answerOpen({}); restoreDom(env.prior); });
    const shown = (on) => ({ accounting: { available: true },
        remote_execution: state(on ? { on: true, target: 'vds' } : { target: 'vds' }) });
    await flush();
    await answerOpen(shown(false));  // Main's own first read at mount
    const capsule = env.mount.querySelector('.rx-capsule');
    const word = capsule.querySelector('.rx-word');
    assert.equal(word.textContent, 'Off');
    // The page's poll (app.js refreshProjectsNav(false)): begun before the save, it answers Off after it.
    void stateSnapshots.gate(false).then((request) => request && readStateSnapshot(stateSnapshots, request));
    await flush();
    const stale = reads.at(-1);
    capsule.querySelector('.rx-toggle').click();
    await flush();
    assert.deepEqual(posts, [{ OUROBOROS_REMOTE_EXECUTION: '{"engine":"vds","on":true}' }], 'the ordinary settings save');
    stale.answer(shown(false));
    await flush();
    assert.equal(word.textContent, 'Saving…', 'the read begun before the save never confirms it');
    await answerOpen(shown(true));
    assert.deepEqual([word.textContent, capsule.dataset.on], ['VDSina', 'true']);
});
