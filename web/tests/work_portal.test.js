// One work, one card (work_portal.js) through the real Main chat consumer: a card whose work moved into a Project
// room stays the same node and becomes its portal — from /api/state task_bindings, from a census row naming the
// Project, and from a project_started / project_handoff row (no second entry) — reads its present from the
// all-chat census (CONTRACTS K8: last_line, last_line_at, tool_calls, detached), keeps its commands (… [Stop…]), and
// settles from the completion row's typed outcome. A census without the new fields leaves today's card.
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { createWorkPortals } from '../modules/work_portal.js';
import { clock } from '../modules/work_status.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
import { projection, room as serverRoom, work } from './work_fixture.js';

const originalQuery = ElementStub.prototype.querySelector;
after(() => { ElementStub.prototype.querySelector = originalQuery; });
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

// A bubbling event, as the browser delivers the Continue control's announcement to its chat's feed.
const originalDispatch = ElementStub.prototype.dispatchEvent;
after(() => { ElementStub.prototype.dispatchEvent = originalDispatch; });
ElementStub.prototype.dispatchEvent = function (event) {
    for (let node = this; node; node = node.parentNode) {
        for (const fn of node.listeners?.get(event.type) || []) fn(event);
        if (!event.bubbles) break;
    }
    return true;
};

const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };

function fixture({ history = [], details = {}, continues = {}, chatId = 1 } = {}) {
    const reads = [];
    const env = installDom(async (url, init = {}) => {
        const path = String(url);
        if (init.method === 'POST' && path.endsWith('/continue')) {
            return { ok: true, json: async () => continues[path.split('/').at(-2)] || { ok: false, error: 'not admitted' } };
        }
        if (path.startsWith('/api/tasks/')) reads.push(path.split('/').pop().split('?')[0]);
        // `null` is a record the host does not have: its real answer, a 404.
        if (path.startsWith('/api/tasks/') && details[path.split('/').pop().split('?')[0]] === null) {
            return { ok: false, status: 404, json: async () => ({ error: 'task not found' }) };
        }
        return { ok: true, json: async () => (path.startsWith('/api/chat/history') ? { messages: history, window: { complete: true } }
            : path.startsWith('/api/tasks/') ? details[path.split('/').pop().split('?')[0]] || { status: 'running' }
                : { active_direct_turns: [] }) };
    });
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, asPanel: chatId !== 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const card = (id) => walkCard(messages, id);
    return {
        instance, messages, card, reads,
        part: (id, attr) => card(id)?.querySelector(`[data-${attr}]`),
        narrate: (taskId, content, seconds = 0, extra = {}) => handlers.get('chat')({ chat_id: chatId, role: 'assistant',
            is_progress: true, narration: true, task_id: taskId, content, ts: `2026-10-10T12:00:${String(seconds).padStart(2, '0')}Z`,
            ...extra }),
        frame: (row) => handlers.get('chat')({ chat_id: chatId, ...row }),
        emit: (type, row) => handlers.get(type)?.(row),
        state: (data) => instance.hydrateStateSnapshot({ supervisor_ready: true, active_chat_activities_complete: true,
            active_chat_activities: [], ...data }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const room = (extra = {}) => ({ activity_id: 'w-1', chat_id: 7, kind: 'managed_task', phase: 'working',
    started_at: Date.parse('2026-10-10T12:00:00Z') / 1000, ...extra });
const controls = (f, id) => (f.card(id)?.querySelector('.chat-live-actions')?.children || []).map((node) => node.textContent);
// The Project reference leads the card's own action row, under its meta line (owner 26.10, B).
const reference = (node) => node?.querySelector('[data-live-foot-end]')?.firstElementChild || null;
const shownName = (node) => reference(node)?.querySelector('.chat-live-project-name')?.textContent;

test('a bound card becomes its work\'s portal in place: Project reference, the room\'s present, Stop…', async () => {
    const f = fixture();
    try {
        f.narrate('w-1', 'Reading the logs before the audit.');
        const node = f.card('w-1');
        assert.ok(node, 'the work card is in Main');
        f.state({ task_bindings: { 'w-1': { project_id: 'p-1', chat_id: 7 } }, projects: [{ id: 'p-1', name: 'Audit' }],
            active_chat_activities: [room({ last_line: 'Checking the third project.', last_line_at: '2026-10-10T12:05:00Z',
                tool_calls: 41 })] });
        await settle();
        assert.equal(f.card('w-1'), node, 'the SAME node, never a second card or an entry');
        assert.equal(node.dataset.portal, '1');
        assert.equal(node.dataset.projectId, 'p-1');
        const door = reference(node);
        assert.ok(door?.classList.contains('chat-live-bound-pointer'), 'the Project reference leads the card\'s action row');
        assert.equal(door.getAttribute('aria-label'), 'Open project Audit');
        assert.equal(shownName(node), 'Audit', 'a Project named apart from its work shows its name');
        assert.equal(node.querySelectorAll('[data-intent="open-project"]').length, 1, 'one door, never a second footer');
        assert.equal(f.part('w-1', 'live-activity').textContent, 'Checking the third project.', 'the room\'s latest line');
        assert.match(f.part('w-1', 'live-meta').innerHTML, /41 tool calls/);
        assert.match(f.part('w-1', 'live-meta').innerHTML, /updated /);
        assert.deepEqual(controls(f, 'w-1'), ['Stop…'], 'one action row of commands: Stop… while the room runs it (Reply is the row menu\'s)');
        assert.equal(node.querySelector('[data-turn-into-project]'), null, 'a moved work no longer converts');
        assert.equal(messages(f).filter((child) => child.classList.contains('project-handoff')).length, 0);
    } finally { f.close(); }
});

const messages = (f) => f.messages.children;

test('an older census without the new facts leaves today\'s card: its own line, a start time, no invented count', async () => {
    const f = fixture();
    try {
        f.narrate('w-1', 'Reading the logs before the audit.');
        f.narrate('w-1', 'Comparing the two runs.', 1);
        const before = { title: f.part('w-1', 'live-title').textContent, line: f.part('w-1', 'live-activity').textContent };
        f.state({ task_bindings: { 'w-1': { project_id: 'p-1' } }, active_chat_activities: [room()] });
        await settle();
        const node = f.card('w-1');
        assert.equal(node.dataset.portal, '1');
        assert.deepEqual({ title: f.part('w-1', 'live-title').textContent, line: f.part('w-1', 'live-activity').textContent },
            before, 'its own title and last line stay as they were');
        const meta = f.part('w-1', 'live-meta').innerHTML;
        assert.doesNotMatch(meta, /tool call/, 'a count the host did not send is never shown');
        assert.match(meta, /(updated|started) /, 'an unfinished card always carries a time');
        assert.equal(reference(node).getAttribute('aria-label'), 'Open project Project', 'an unnamed Project reads Project');
    } finally { f.close(); }
});

test('a census row naming the work\'s Project binds it too; detached work says in background', async () => {
    const f = fixture();
    try {
        f.narrate('w-1', 'Started the audit.');
        f.state({ active_chat_activities: [room({ project_id: 'p-2', project_name: 'Second room', detached: true,
            last_line: 'Queued behind the build.' })] });
        await settle();
        const node = f.card('w-1');
        assert.equal(node.dataset.portal, '1');
        assert.equal(reference(node).getAttribute('aria-label'), 'Open project Second room');
        assert.equal(f.part('w-1', 'live-phase-secondary')?.textContent, 'in background');
    } finally { f.close(); }
});

test('beside a title that already says its name the reference reads Project; its accessible name stays whole (owner 26.10, B)', async () => {
    const f = fixture();
    try {
        for (const id of ['w-1', 'w-2']) f.narrate(id, 'Reading the logs before the audit.');
        f.emit('task_named', { task_id: 'w-1', suggested_name: 'Audit the projects' });
        f.emit('task_named', { task_id: 'w-2', suggested_name: 'Audit the projects' });
        f.state({ task_bindings: { 'w-1': { project_id: 'p-1', chat_id: 7 }, 'w-2': { project_id: 'p-2', chat_id: 8 } },
            projects: [{ id: 'p-1', name: 'audit  the Projects' }, { id: 'p-2', name: 'Audit room' }],
            active_chat_activities: [room(), room({ activity_id: 'w-2', chat_id: 8 })] });
        await settle();
        assert.equal(f.part('w-1', 'live-title').textContent, 'Audit the projects');
        assert.equal(shownName(f.card('w-1')), 'Project', 'the same words (case and spacing aside) are not said twice');
        assert.equal(reference(f.card('w-1')).getAttribute('aria-label'), 'Open project audit  the Projects');
        assert.equal(reference(f.card('w-1')).title, 'Open project audit  the Projects', 'the tooltip names it too');
        assert.equal(shownName(f.card('w-2')), 'Audit room', 'a differently named Project keeps its name on the capsule');
        assert.equal(reference(f.card('w-2')).getAttribute('aria-label'), 'Open project Audit room');
    } finally { f.close(); }
});

test('an unnamed portal relabels on narration and completion before the next census', async () => {
    for (const [before, after] of [['Comparing the two runs.', 'Audit the projects'],
        ['Audit the projects', 'Comparing the two runs.']]) {
        const f = fixture();
        try {
            f.narrate('w-1', before);
            const binding = { task_bindings: { 'w-1': { project_id: 'p-1', chat_id: 7 } },
                projects: [{ id: 'p-1', name: 'Audit the projects' }] };
            f.state({ ...binding, active_chat_activities: [room()] });
            await settle();
            f.narrate('w-1', after, 1);
            const expected = after === 'Audit the projects' ? 'Project' : 'Audit the projects';
            assert.equal(f.part('w-1', 'live-title').textContent, after);
            assert.equal(shownName(f.card('w-1')), expected, 'narration changes both title and capsule immediately');
            f.frame({ role: 'system', system_type: 'task_summary', task_id: 'w-1', status: 'completed',
                outcome_axes: { lifecycle: { status: 'completed' } }, content: 'Finished.', ts: '2026-10-10T12:00:02Z' });
            f.state({ ...binding, active_chat_activities: [] });
            await settle();
            assert.equal(shownName(f.card('w-1')), expected, 'settlement and a later census do not leave a stale label');
            assert.equal(reference(f.card('w-1')).getAttribute('aria-label'), 'Open project Audit the projects');
        } finally { f.close(); }
    }
});

test('a project_started row binds its work\'s card at the request\'s place instead of drawing an entry', async () => {
    const history = [
        { role: 'user', text: 'make the audit a project', ts: '2026-10-10T11:59:00Z', chat_id: 1, source: 'web',
            history_id: 'h-1', row_address: 'row:1@h#1' },
        { role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-3', project_id: 'p-3',
            project_name: 'Audit room', task_name: 'Audit the projects', text: 'Audit room › Audit the projects · Started',
            ts: '2026-10-10T12:00:00Z', history_id: 'h-2' },
    ];
    const f = fixture({ history });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        const node = f.card('w-3');
        assert.ok(node, 'the work has its card');
        assert.equal(node.dataset.portal, '1');
        assert.equal(f.part('w-3', 'live-title').textContent, 'Audit the projects', 'named by the row\'s work name');
        assert.equal(messages(f).filter((child) => child.dataset?.systemType === 'project_started').length, 0,
            'the row is data: no second entry');
    } finally { f.close(); }
});

test('the author\'s final settles the portal from its typed outcome; without one, the durable detail decides', async () => {
    const f = fixture({ details: { 'w-5': { task_id: 'w-5', status: 'failed', outcome_axes: { lifecycle: { status: 'failed' } } } } });
    try {
        for (const id of ['w-4', 'w-5']) f.narrate(id, 'Working on it.');
        f.state({ task_bindings: { 'w-4': { project_id: 'p-1' }, 'w-5': { project_id: 'p-1' } },
            active_chat_activities: [room({ activity_id: 'w-4' }), room({ activity_id: 'w-5' })] });
        await settle();
        f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: 'w-4', project_id: 'p-1',
            project_name: 'Audit', content: 'All seven projects run again.', completion_answer: 'All seven projects run again.',
            outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } }, status: 'completed',
            ts: '2026-10-10T13:00:00Z' });
        await settle();
        assert.equal(f.part('w-4', 'live-phase').textContent, 'Done', 'the typed outcome speaks');
        f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: 'w-5', project_id: 'p-1',
            project_name: 'Audit', content: 'Stopped early.', status: 'completed', ts: '2026-10-10T13:01:00Z' });
        await settle();
        assert.ok(f.reads.includes('w-5'), 'the bare lifecycle word never decides: the durable detail is read');
    } finally { f.close(); }
});

test('Main only: a Project room\'s portals bind nothing and read nothing', () => {
    const records = new Map([['w-1', { groupId: 'w-1', root: { dataset: {}, isConnected: true } }]]);
    const calls = [];
    const portals = createWorkPortals({ isMain: false, feed: null, records, onDomWrite: (write) => (calls.push('write'), write()),
        card: new Proxy({}, { get: (_target, name) => () => calls.push(name) }) });
    assert.equal(portals.bind(records.get('w-1'), { id: 'p-1' }), false);
    portals.sync({ task_bindings: { 'w-1': { project_id: 'p-1' } }, active_chat_activities: [room()] });
    assert.equal(portals.bindRow({ task_id: 'w-1', project_id: 'p-1' }), false);
    assert.deepEqual(calls, [], 'the room shows the full card');
    assert.equal(records.get('w-1').root.dataset.portal, undefined);
});

// A Continue made in the Project room (owner QB = A, 16.10 Q1 = A): Main sees none of the successor's frames, so the
// predecessor's admitted link (its durable detail) and the successor's census row are what its ONE card follows.
const bindings = { 'w-1': { project_id: 'p-1', chat_id: 7 } };
const failedDetail = (extra = {}) => ({ task_id: 'w-1', status: 'failed', reason_code: 'model_provider_failed',
    terminal_time: { occurred_at: '2026-10-10T12:20:00Z', attempt: { started_at: '2026-10-10T12:00:00Z' } }, ...extra });
const continuedBy = (state = 'admitted') => ({
    continued_by: { state, successor_task_id: 'w-1-c1', action_nonce: 'n-1', binding: { project_id: 'p-1', chat_id: 7 } },
    continuation_offer: state === 'bound'
        ? { eligible: false, refusal: 'continuation_unconfirmed', state, successor_task_id: 'w-1-c1', action_nonce: 'n-1' }
        : { eligible: false, refusal: 'already_continued', state, successor_task_id: 'w-1-c1' } });
const successorRow = (extra = {}) => room({ activity_id: 'w-1-c1', project_id: 'p-1', project_name: 'Audit',
    started_at: Date.parse('2026-10-10T12:30:00Z') / 1000, last_line: 'Continuing the audit.', tool_calls: 3,
    cancelable: true, ...extra });

async function runningPortal(f) {
    f.narrate('w-1', 'Reading the logs before the audit.');
    f.emit('task_named', { task_id: 'w-1', suggested_name: 'Audit the projects' });
    f.state({ task_bindings: bindings, projects: [{ id: 'p-1', name: 'Audit' }],
        active_chat_activities: [room({ project_id: 'p-1', project_name: 'Audit' })] });
    await settle();
    assert.equal(f.card('w-1').dataset.portal, '1');
    return f.card('w-1');
}

function assertFollowed(f, node) {
    const next = f.card('w-1-c1');
    assert.ok(next, 'the successor has the work\'s card in Main');
    assert.equal(node.hidden, true, 'the ended episode folds into it: never a second card');
    assert.equal(next.hidden, false);
    const feed = f.messages.children;
    assert.equal(feed.indexOf(next), feed.indexOf(node) + 1, 'it stands where the work began');
    assert.equal(feed.filter((child) => child.classList.contains('chat-live-card') && !child.hidden).length, 1,
        'one visible card for the work');
    assert.equal(next.dataset.portal, '1');
    assert.equal(reference(next).getAttribute('aria-label'), 'Open project Audit', 'the Project reference leads its action row');
    assert.equal(f.part('w-1-c1', 'live-title').textContent, 'Audit the projects', 'the work keeps its first name');
    assert.equal(f.part('w-1-c1', 'live-phase').textContent, 'Working', 'the current episode owns the chip');
    assert.equal(f.part('w-1-c1', 'live-activity').textContent, 'Continuing the audit.');
    assert.match(f.part('w-1-c1', 'live-meta').innerHTML, /3 tool calls/);
    assert.deepEqual(controls(f, 'w-1-c1'), ['Stop…'], 'its own controls: Stop… while the room runs it');
    const episodes = f.part('w-1-c1', 'live-episodes');
    assert.equal(episodes.hidden, false);
    assert.match(episodes.innerHTML, /Episode 1<\/span> · <span class="chat-live-episode-name">Audit the projects<\/span> · <span class="chat-live-episode-span">[^<]+<\/span> · <span class="chat-live-episode-outcome" data-tone="failed">Failed<\/span>/,
        'the ended episode is one meta line on the same card');
    assert.match(episodes.innerHTML, /data-episode="2" data-current="1"/);
}

// The capsule's words follow the card's title through the title's naming owner (task_episodes.js `renamed`), with no
// watch of its own on the title (DEVELOPMENT "UI resources carry a disposer") and no census in between.
test('the title\'s naming owner names the Project reference again: a rename and a Continue\'s first name (owner 26.10, B)', async () => {
    const details = {};
    const f = fixture({ details });
    try {
        f.narrate('w-1', 'Reading the logs before the audit.');
        f.emit('task_named', { task_id: 'w-1', suggested_name: 'Audit the projects' });
        f.state({ task_bindings: bindings, projects: [{ id: 'p-1', name: 'Audit the projects' }],
            active_chat_activities: [room({ project_id: 'p-1', project_name: 'Audit the projects' })] });
        await settle();
        assert.equal(shownName(f.card('w-1')), 'Project');
        f.emit('task_named', { task_id: 'w-1', suggested_name: 'Audit, second pass' });
        assert.equal(f.part('w-1', 'live-title').textContent, 'Audit, second pass');
        assert.equal(shownName(f.card('w-1')), 'Audit the projects', 'renamed apart: the capsule names its Project at once');
        f.emit('task_named', { task_id: 'w-1', suggested_name: 'Audit the projects' });
        assert.equal(shownName(f.card('w-1')), 'Project');
        // Continued in its room: the successor's card is minted, then the chain gives it the work's first name.
        details['w-1'] = failedDetail(continuedBy());
        f.state({ task_bindings: bindings, active_chat_activities: [] });
        await settle();
        f.state({ task_bindings: bindings, active_chat_activities: [successorRow({ project_name: 'Audit the projects' })] });
        await settle();
        const next = f.card('w-1-c1');
        assert.equal(f.part('w-1-c1', 'live-title').textContent, 'Audit the projects', 'the work keeps its first name');
        assert.equal(shownName(next), 'Project', 'in the same census that minted it');
        assert.equal(reference(next).getAttribute('aria-label'), 'Open project Audit the projects');
    } finally { f.close(); }
});

test('a Continue made in the room: the successor\'s census row takes the work\'s card in Main, live; no row, no change', async () => {
    const details = {};
    const f = fixture({ details });
    try {
        const node = await runningPortal(f);
        // Interrupted and continued in its room: Main reads the link, but the successor's row is not here yet.
        details['w-1'] = failedDetail(continuedBy());
        f.state({ task_bindings: bindings, active_chat_activities: [] });
        await settle();
        assert.equal(f.card('w-1-c1'), null, 'without the successor\'s own facts Main mints nothing');
        assert.equal(node.hidden, false, 'and the card stays as it was');
        assert.equal(f.part('w-1', 'live-phase').textContent, 'Failed');
        // The census that lists it arrives while Main is open.
        f.state({ task_bindings: bindings, active_chat_activities: [successorRow()] });
        await settle();
        assertFollowed(f, node);
    } finally { f.close(); }
});

test('continued later in the room: a new root in its Project makes the settled portal read its own record once', async () => {
    const details = {};
    const f = fixture({ details });
    try {
        const node = await runningPortal(f);
        details['w-1'] = failedDetail({ continuation_offer: { eligible: true, cause: 'model_provider_failed' } });
        f.state({ task_bindings: bindings, active_chat_activities: [] });
        await settle();
        assert.deepEqual(controls(f, 'w-1'), ['Continue'], 'the interrupted work offers Continue');
        const reads = () => f.reads.filter((id) => id === 'w-1').length;
        const read = reads();
        // Another work starts in the same room: the portal's record names no successor, so nothing changes.
        const other = room({ activity_id: 'w-9', project_id: 'p-1', project_name: 'Audit' });
        f.state({ task_bindings: bindings, active_chat_activities: [other] });
        await settle();
        assert.equal(reads(), read + 1, 'a new root in its Project: the settled portal reads its own record');
        assert.equal(f.card('w-9'), null, 'a root its record does not name is not its Continue');
        assert.equal(node.hidden, false);
        f.state({ task_bindings: bindings, active_chat_activities: [other] });
        await settle();
        assert.equal(reads(), read + 1, 'once per root, never a poll');
        // The owner (or the mind) presses Continue in the room: the record now names its admitted successor.
        details['w-1'] = failedDetail(continuedBy());
        f.state({ task_bindings: bindings, active_chat_activities: [other, successorRow()] });
        await settle();
        assertFollowed(f, node);
    } finally { f.close(); }
});

test('a queued successor whose start the census does not know still stands at the work\'s place', async () => {
    const details = {};
    const f = fixture({ details });
    try {
        const node = await runningPortal(f);
        details['w-1'] = failedDetail(continuedBy());
        f.state({ task_bindings: bindings, active_chat_activities: [successorRow({ started_at: 0 })] });
        await settle();
        assertFollowed(f, node);
    } finally { f.close(); }
});

test('after a Main reload the start row binds the work and its record names the successor: the same one card', async () => {
    const history = [
        { role: 'user', text: 'audit the projects in their own room', ts: '2026-10-10T11:59:00Z', chat_id: 1, source: 'web',
            history_id: 'h-1', row_address: 'row:1@h#1' },
        { role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-1', project_id: 'p-1', project_name: 'Audit',
            task_name: 'Audit the projects', text: 'Audit › Audit the projects · Started', ts: '2026-10-10T12:00:00Z',
            history_id: 'h-2' },
    ];
    const f = fixture({ history, details: { 'w-1': failedDetail(continuedBy()) } });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        f.state({ task_bindings: bindings, projects: [{ id: 'p-1', name: 'Audit' }], active_chat_activities: [successorRow()] });
        await settle();
        assertFollowed(f, f.card('w-1'));
        assert.equal(f.messages.children.filter((child) => child.dataset?.systemType === 'project_started').length, 0);
    } finally { f.close(); }
});

test('a bound claim stays pending: Retry Continue on the work\'s own card, nothing folds until the host admits it', async () => {
    const details = {};
    const f = fixture({ details });
    try {
        const node = await runningPortal(f);
        details['w-1'] = failedDetail(continuedBy('bound'));
        for (let pass = 0; pass < 2; pass += 1) {
            f.state({ task_bindings: bindings, active_chat_activities: [successorRow()] });
            await settle();
        }
        assert.equal(f.card('w-1-c1'), null, 'a bound claim mints no successor card');
        assert.equal(node.hidden, false, 'nor hides or replaces the work\'s card');
        assert.deepEqual(controls(f, 'w-1'), ['Retry Continue'], 'the claim shows as pending');
        details['w-1'] = failedDetail(continuedBy());
        f.state({ task_bindings: bindings, active_chat_activities: [successorRow()] });
        await settle();
        assertFollowed(f, node);
    } finally { f.close(); }
});

test('an admitted successor paused in the room carries Resume as its one primary, from its work\'s own control', async () => {
    const details = {};
    const f = fixture({ details });
    try {
        const node = await runningPortal(f);
        details['w-1'] = failedDetail(continuedBy());
        const paused = successorRow({ phase: 'budget_paused', pause_cause: 'owner', can_resume: true, last_line: '' });
        // The work runs in its Project room (chat 7): the portal's card in Main still reads that work's controls.
        const pressed = '2026-09-15T10:00:00Z';
        const pausedWork = (controls) => projection(7, [work(paused, { activity: 'paused', because: [{ code: 'paused_by_owner',
            text: 'You paused it', since: pressed, by: { actor: 'owner', text: '', at: pressed } }], controls })],
        serverRoom('paused'));
        f.state({ task_bindings: bindings, active_chat_activities: [paused],
            ...pausedWork([{ action: 'stop', target: 'w-1-c1' }]) });
        await settle();
        assert.equal(node.hidden, true);
        assert.deepEqual(controls(f, 'w-1-c1'), ['Stop…'], 'a paused phase alone offers no Resume');
        f.state({ task_bindings: bindings, active_chat_activities: [paused],
            ...pausedWork([{ action: 'resume', target: 'w-1-c1' }, { action: 'stop', target: 'w-1-c1' }]) });
        await settle();
        assert.deepEqual(controls(f, 'w-1-c1'), ['Resume', 'Stop…'], 'paused: Resume beside Stop…');
        assert.deepEqual([f.part('w-1-c1', 'live-phase').textContent, f.part('w-1-c1', 'live-phase-secondary').textContent],
            ['Paused', `You paused it · ${clock(pressed)}`], 'the portal says the work\'s word, its line beside it');
        const primary = f.card('w-1-c1').querySelector('.chat-live-actions').children.filter((b) => b.classList.contains('btn-primary'));
        assert.deepEqual(primary.map((b) => b.textContent), ['Resume']);
        // Released in the room: the next body says working, and the card's controls follow its work.
        const working = successorRow();
        f.state({ task_bindings: bindings, active_chat_activities: [working], ...projection(7, [work(working, {
            controls: [{ action: 'wrap_up', target: 'w-1-c1' }, { action: 'stop', target: 'w-1-c1' }] })],
        serverRoom('working')) });
        await settle();
        assert.deepEqual(controls(f, 'w-1-c1'), ['Stop…']);
        assert.equal(f.part('w-1-c1', 'live-activity').textContent, 'Continuing the audit.');
    } finally { f.close(); }
});

// A work's own card where its frames land (owner QB = A on every surface): the Project room, or Main for a work that
// never moved. A Continue pressed there, or made anywhere else, folds the ended episode into the successor's card, live
// and not only after a reload. `here` builds census rows of that chat.
async function failedInRoom(f, details, here = room) {
    f.narrate('w-1', 'Reading the logs before the audit.', 0);
    f.emit('task_named', { task_id: 'w-1', suggested_name: 'Audit the projects' });
    f.state({ active_chat_activities: [here({ cancelable: true })] });
    await settle();
    details['w-1'] = failedDetail({ continuation_offer: { eligible: true, cause: 'model_provider_failed' } });
    f.frame({ role: 'system', system_type: 'task_summary', task_id: 'w-1', status: 'failed', reason_code: 'model_provider_failed',
        outcome_axes: { lifecycle: { status: 'failed' } }, content: 'The model provider failed.', ts: '2026-10-10T12:00:20Z' });
    f.state({ active_chat_activities: [] });
    await settle();
    assert.equal(f.part('w-1', 'live-phase').textContent, 'Failed');
    assert.ok(controls(f, 'w-1').includes('Continue'), 'the interrupted work offers Continue on its own card');
    return f.card('w-1');
}

// The real Continue control answers from the fixture's POST /continue; its toast gets a body and its dismiss timer
// may not hold the run.
function quietToasts() {
    document.body ||= new ElementStub('body', document);
    const real = globalThis.setTimeout;
    globalThis.setTimeout = (fn, ms, ...args) => { const timer = real(fn, ms, ...args); if (ms >= 1000) timer?.unref?.(); return timer; };
    return () => { globalThis.setTimeout = real; };
}

const pressContinue = (f, id) => f.card(id).querySelector('[data-continue-task]').onclick({ stopPropagation() {} });
const successorFrame = (f) => f.narrate('w-1-c1', 'Continuing the audit.', 40, { cancelable: true });

function assertRoomFollowed(f, node) {
    const next = f.card('w-1-c1');
    assert.ok(next, 'the successor\'s card is in the room');
    assert.equal(node.hidden, true, 'the ended episode folds into it: never a second card');
    assert.equal(next.hidden, false);
    const feed = f.messages.children;
    assert.equal(feed.indexOf(next), feed.indexOf(node) + 1, 'it stands where the work began');
    assert.equal(f.part('w-1-c1', 'live-title').textContent, 'Audit the projects', 'the work keeps its first name');
    assert.equal(f.part('w-1-c1', 'live-phase').textContent, 'Working', 'the current episode owns the chip');
    assert.equal(f.part('w-1-c1', 'live-activity').textContent, 'Continuing the audit.', 'and its own latest line');
    assert.ok(controls(f, 'w-1-c1').includes('Stop…'), 'and its controls');
    assert.ok(!controls(f, 'w-1-c1').includes('Continued'));
    const episodes = f.part('w-1-c1', 'live-episodes');
    assert.equal(episodes.hidden, false);
    assert.match(episodes.innerHTML, /Episode 1<\/span> · <span class="chat-live-episode-name">Audit the projects<\/span> · <span class="chat-live-episode-span">[^<]+<\/span> · <span class="chat-live-episode-outcome" data-tone="failed">Failed<\/span>/,
        'the ended episode is one meta line on the same card');
    assert.match(episodes.innerHTML, /data-episode="2" data-current="1"/);
}

const chats = [[7, room], [1, (extra = {}) => room({ chat_id: 1, ...extra })]];  // the room; Main's own work

test('a Continue pressed on the work\'s own card is the same card there, live, whether the successor speaks before or after the answer', async () => {
    for (const [chatId, here] of chats) for (const successorFirst of [false, true]) {
        const details = {};
        const f = fixture({ details, chatId, continues: { 'w-1': { ok: true, successor_task_id: 'w-1-c1' } } });
        const restore = quietToasts();
        try {
            const node = await failedInRoom(f, details, here);
            details['w-1'] = failedDetail(continuedBy());  // the record the host has written once the press is admitted
            const pressed = pressContinue(f, 'w-1');
            if (successorFirst) successorFrame(f);
            await pressed;
            await settle();
            if (!successorFirst) {
                assert.equal(f.card('w-1-c1'), null);
                assert.equal(node.hidden, false, 'nothing folds before the successor has a card');
                successorFrame(f);
            }
            await settle();
            assertRoomFollowed(f, node);
        } finally { restore(); f.close(); }
    }
});

test('a Continue made elsewhere (another chat or tab, the mind): the card\'s chat reads its record when the census lists the new root', async () => {
    for (const [chatId, here] of chats) {
    const details = {};
    const f = fixture({ details, chatId });
    try {
        const node = await failedInRoom(f, details, here);
        const reads = () => f.reads.filter((id) => id === 'w-1').length;
        const read = reads();
        // Another work starts in the room: the settled card's record names no successor, so nothing changes.
        f.narrate('w-9', 'Another audit.', 30);
        f.state({ active_chat_activities: [here({ activity_id: 'w-9' })] });
        await settle();
        assert.equal(reads(), read + 1, 'a new root in this chat: the settled card reads its own record');
        assert.equal(node.hidden, false);
        f.state({ active_chat_activities: [here({ activity_id: 'w-9' })] });
        await settle();
        assert.equal(reads(), read + 1, 'once per root, never a poll');
        // Continued elsewhere: the successor speaks in the room before its link is read, and nothing is guessed.
        details['w-1'] = failedDetail(continuedBy());
        successorFrame(f);
        await settle();
        assert.equal(node.hidden, false, 'no census, no read: two cards stay as they are');
        f.state({ active_chat_activities: [here({ activity_id: 'w-9' }), here({ activity_id: 'w-1-c1', cancelable: true })] });
        await settle();
        assertRoomFollowed(f, node);
        assert.equal(f.card('w-9').hidden, false, 'the unrelated work keeps its own card');
    } finally { f.close(); }
    }
});

test('after a room reload its history folds the same chain: one card for the work', async () => {
    const offer = { eligible: false, refusal: 'already_continued', state: 'admitted', successor_task_id: 'w-1-c1' };
    const history = [
        { role: 'user', text: 'audit the projects', chat_id: 7, source: 'web', ts: '2026-10-10T11:59:00Z', history_id: 'h-1',
            row_address: 'row:7@h#1' },
        { role: 'assistant', is_progress: true, narration: true, chat_id: 7, task_id: 'w-1', text: 'Reading the logs before the audit.',
            ts: '2026-10-10T12:00:00Z', history_id: 'h-2', suggested_name: 'Audit the projects', task_terminal_status: 'failed',
            reason_code: 'model_provider_failed', outcome_axes: { lifecycle: { status: 'failed' } }, outcome_final: true,
            continuation_offer: offer },
        { role: 'assistant', is_progress: true, narration: true, chat_id: 7, task_id: 'w-1-c1', text: 'Continuing the audit.',
            ts: '2026-10-10T12:00:40Z', history_id: 'h-3', cancelable: true },
    ];
    const f = fixture({ history, chatId: 7, details: { 'w-1': failedDetail(continuedBy()) } });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        f.state({ active_chat_activities: [room({ activity_id: 'w-1-c1', cancelable: true })] });
        await settle();
        assertRoomFollowed(f, f.card('w-1'));
    } finally { f.close(); }
});

// A successor that ended before Main's census ever listed it (astra W2 R2): the predecessor's admitted link names it and
// its own durable record, read through the existing task read, is its whole present. The work's ONE card follows it and
// settles from that outcome; a bound claim or an absent record changes nothing.
const finishedSuccessor = (id = 'w-1-c1', extra = {}) => ({ task_id: id, status: 'completed', chat_id: 7, project_id: 'p-1',
    outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } },
    terminal_time: { occurred_at: '2026-10-10T12:42:00Z', attempt: { started_at: '2026-10-10T12:30:00Z' } }, ...extra });

function assertSettledFollow(f, folded, { id = 'w-1-c1', episodes = 2 } = {}) {
    const next = f.card(id);
    assert.ok(next, 'the finished successor has the work\'s card in Main');
    for (const node of folded) assert.equal(node.hidden, true, 'every ended episode folds into it: never a second card');
    assert.equal(next.hidden, false);
    const feed = f.messages.children;
    assert.equal(feed.indexOf(next), feed.indexOf(folded[0]) + folded.length, 'it stands where the work began');
    assert.equal(feed.filter((child) => child.classList.contains('chat-live-card') && !child.hidden).length, 1,
        'one visible card for the work');
    assert.equal(next.dataset.portal, '1');
    assert.equal(reference(next).getAttribute('aria-label'), 'Open project Audit', 'the Project reference leads its action row');
    assert.equal(f.part(id, 'live-title').textContent, 'Audit the projects', 'the work keeps its first name');
    assert.equal(f.part(id, 'live-phase').textContent, 'Done', 'the current episode settles from its durable outcome');
    assert.ok(!controls(f, id).includes('Stop…') && !controls(f, id).includes('Continued'), 'a settled work offers no Stop');
    const lines = f.part(id, 'live-episodes').innerHTML;
    assert.equal((lines.match(/data-episode="/g) || []).length, episodes);
    assert.match(lines, /Episode 1<\/span> · <span class="chat-live-episode-name">Audit the projects<\/span> · <span class="chat-live-episode-span">[^<]+<\/span> · <span class="chat-live-episode-outcome" data-tone="failed">Failed<\/span>/);
    assert.match(lines, new RegExp(`data-episode="${episodes}" data-current="1"><span class="chat-live-episode-label">Episode ${episodes}</span> · (?:<span class="chat-live-episode-name">[^<]+</span> · )?<span class="chat-live-episode-span">[^<]+–[^<]+</span></div>`),
        'the finished current episode carries its own span; the chip is its outcome');
}

test('a successor that finished before Main\'s census is the work\'s current episode, live; a bound claim or an absent record changes nothing', async () => {
    for (const [claim, successor] of [['bound', finishedSuccessor()], ['admitted', null], ['admitted', finishedSuccessor()]]) {
        const details = {};
        const f = fixture({ details });
        try {
            const node = await runningPortal(f);
            details['w-1'] = failedDetail(continuedBy(claim));
            details['w-1-c1'] = successor;
            // Failed, continued in its room, and the successor already finished: the census lists neither.
            f.state({ task_bindings: { ...bindings, 'w-1-c1': { project_id: 'p-1', chat_id: 7 } }, active_chat_activities: [] });
            await settle();
            f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: 'w-1-c1', project_id: 'p-1',
                project_name: 'Audit', content: 'The audit is complete.', ts: '2026-10-10T12:42:30Z' });
            await settle();
            if (claim === 'bound' || successor === null) {
                assert.equal(f.card('w-1-c1'), null, `${claim}/${successor ? 'record' : 'no record'}: no successor card`);
                assert.equal(node.hidden, false, 'the old card stays as it was');
                assert.equal(f.part('w-1', 'live-phase').textContent, 'Failed');
                assert.equal(f.reads.includes('w-1-c1'), claim !== 'bound', `${claim}: only an admitted claim reads its successor`);
                continue;
            }
            assertSettledFollow(f, [node]);
        } finally { f.close(); }
    }
});

test('after a Main reload a finished successor named by the record still settles the work\'s one card', async () => {
    const history = [
        { role: 'user', text: 'audit the projects in their own room', ts: '2026-10-10T11:59:00Z', chat_id: 1, source: 'web',
            history_id: 'h-1', row_address: 'row:1@h#1' },
        { role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-1', project_id: 'p-1', project_name: 'Audit',
            task_name: 'Audit the projects', text: 'Audit › Audit the projects · Started', ts: '2026-10-10T12:00:00Z',
            history_id: 'h-2' },
    ];
    const f = fixture({ history, details: { 'w-1': failedDetail(continuedBy()), 'w-1-c1': finishedSuccessor() } });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        f.state({ task_bindings: bindings, projects: [{ id: 'p-1', name: 'Audit' }], active_chat_activities: [] });
        await settle();
        assertSettledFollow(f, [f.card('w-1')]);
    } finally { f.close(); }
});

test('a chain of successors that each ended unseen follows its proven links to one card', async () => {
    const details = {};
    const f = fixture({ details });
    try {
        const node = await runningPortal(f);
        details['w-1'] = failedDetail(continuedBy());
        // The first successor failed too and was continued; the second finished. Each record names the next.
        details['w-1-c1'] = finishedSuccessor('w-1-c1', { status: 'failed', reason_code: 'model_provider_failed',
            outcome_axes: { lifecycle: { status: 'failed' } },
            terminal_time: { occurred_at: '2026-10-10T12:35:00Z', attempt: { started_at: '2026-10-10T12:30:00Z' } },
            continued_by: { state: 'admitted', successor_task_id: 'w-1-c2', binding: { project_id: 'p-1', chat_id: 7 } } });
        details['w-1-c2'] = finishedSuccessor('w-1-c2');
        f.state({ task_bindings: bindings, active_chat_activities: [] });
        await settle();
        assertSettledFollow(f, [node, f.card('w-1-c1')], { id: 'w-1-c2', episodes: 3 });
    } finally { f.close(); }
});

// Main already read the moved work's Failed record (no link yet) and offers Continue; the owner continues in the room and
// the successor finishes before any census lists it (astra W3). What Main still receives about it: its new Project
// binding and its mirrored final. Either is a reason for the settled portals of that Project to read their own record
// once; only the record's admitted link joins the cards. Project membership alone never does.
async function failedWithContinue(f, details) {
    const node = await runningPortal(f);
    details['w-1'] = failedDetail({ continuation_offer: { eligible: true, cause: 'model_provider_failed' } });
    f.state({ task_bindings: bindings, active_chat_activities: [] });
    await settle();
    assert.deepEqual(controls(f, 'w-1'), ['Continue'], 'Main shows the interruption and offers Continue');
    return node;
}
const lateFinal = (id = 'w-1-c1') => ({ role: 'system', system_type: 'project_completion_summary', task_id: id, project_id: 'p-1',
    project_name: 'Audit', content: 'The audit is complete.', ts: '2026-10-10T12:42:30Z' });

test('W3: a portal already showing Failed follows a later Continue that finished between censuses', async () => {
    for (const via of ['binding and final', 'binding', 'final']) {
        const details = {};
        const f = fixture({ details });
        try {
            const node = await failedWithContinue(f, details);
            details['w-1'] = failedDetail(continuedBy());
            details['w-1-c1'] = finishedSuccessor();
            const after = { task_bindings: { ...bindings, ...(via === 'final' ? {} : { 'w-1-c1': { project_id: 'p-1', chat_id: 7 } }) },
                active_chat_activities: [] };
            if (via !== 'binding') f.frame(lateFinal());
            for (let pass = 0; pass < 3; pass += 1) { f.state(after); await settle(); }
            assert.ok(f.card('w-1-c1'), `${via}: still the old episode, reads=${JSON.stringify(f.reads)}`);
            assertSettledFollow(f, [node]);
        } finally { f.close(); }
    }
});

test('W3: an unrelated work in the same Project, or a claim still bound, leaves the Failed card as it was', async () => {
    for (const [claim, other] of [['none', 'w-7'], ['bound', 'w-1-c1']]) {
        const details = {};
        const f = fixture({ details });
        try {
            const node = await failedWithContinue(f, details);
            const reads = () => f.reads.filter((id) => id === 'w-1').length;
            const read = reads();
            if (claim === 'bound') details['w-1'] = failedDetail(continuedBy('bound'));
            details[other] = finishedSuccessor(other);
            const after = { task_bindings: { ...bindings, [other]: { project_id: 'p-1', chat_id: 7 } }, active_chat_activities: [] };
            f.frame(lateFinal(other));
            for (let pass = 0; pass < 3; pass += 1) { f.state(after); await settle(); }
            assert.equal(f.card(other), null, `${claim}: no second card for ${other}`);
            assert.equal(node.hidden, false, 'the old card stays');
            assert.equal(f.part('w-1', 'live-phase').textContent, 'Failed');
            assert.ok(reads() > read, 'the observation made the settled portal read its own record');
            assert.ok(reads() <= read + 2, 'once per observation, never a poll');
            assert.deepEqual(controls(f, 'w-1'), [claim === 'bound' ? 'Retry Continue' : 'Continue']);
        } finally { f.close(); }
    }
});
