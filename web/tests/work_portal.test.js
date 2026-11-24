// One work, one card (work_portal.js) through the real Main chat consumer: a card whose work moved into a Project
// room stays the same node and becomes its portal — from /api/state task_bindings, from a census row naming the
// Project, and from a project_started / project_handoff row (no second entry) — reads its present from the
// all-chat census (CONTRACTS K8: last_line, last_line_at, tool_calls, detached), keeps its commands (… [Stop…]), and
// settles from the completion row's typed outcome. A census without the new fields leaves today's card.
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { createWorkPortals } from '../modules/work_portal.js';
import { closeTaskControlMenu } from '../modules/task_control_menu.js';
import { clock } from '../modules/work_status.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
import { projection, room as serverRoom, SERVER_CASES, work as fixtureWork } from './work_fixture.js';

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

function fixture({ history = [], details = {}, continues = {}, chatId = 1, hold = {}, fail = {}, historyGate = null,
    detailReply, stopFailure = false, conversion = null } = {}) {
    const reads = [];
    const posts = [];
    const env = installDom(async (url, init = {}) => {
        const path = String(url);
        if (init.method === 'POST' && path.endsWith('/projects/from-task') && conversion) return conversion(init);
        if (init.method === 'POST' && path.endsWith('/continue')) {
            return { ok: true, json: async () => continues[path.split('/').at(-2)] || { ok: false, error: 'not admitted' } };
        }
        if (init.method === 'POST') {
            posts.push(path);
            if (stopFailure && path.endsWith('/cancel')) return { ok: false, status: 409, json: async () => ({ error: 'Synthetic refusal' }) };
            return { ok: true, json: async () => ({ ok: true, applied: true }) };
        }
        if (path.startsWith('/api/tasks/')) {
            const id = path.split('/').pop().split('?')[0];
            reads.push(id);
            if (detailReply) return detailReply(id, reads.filter(key => key === id).length);
        }
        // A history read the test gates answers only when the gate opens: the instance's own first read (after its
        // hydration window, a timer) and the test's own both wait, whatever the host's timer granularity.
        if (path.startsWith('/api/chat/history') && historyGate) await historyGate;
        // A read the test holds open answers only when the test releases it.
        if (path.startsWith('/api/tasks/') && hold[path.split('/').pop().split('?')[0]]) await hold[path.split('/').pop().split('?')[0]];
        // A read the test fails answers 500, which the strict read raises: a failed read, never an absent record.
        if (path.startsWith('/api/tasks/') && fail[path.split('/').pop().split('?')[0]]) {
            return { ok: false, status: 500, json: async () => ({ error: 'unavailable' }) };
        }
        // `null` is a record the host does not have: its real answer, a 404.
        if (path.startsWith('/api/tasks/') && details[path.split('/').pop().split('?')[0]] === null) {
            return { ok: false, status: 404, json: async () => ({ error: 'task not found' }) };
        }
        return { ok: true, json: async () => (path.startsWith('/api/chat/history') ? { messages: history, window: { complete: true } }
            : path.startsWith('/api/tasks/') ? details[path.split('/').pop().split('?')[0]] || { status: 'running' }
                : { active_direct_turns: [] }) };
    });
    const handlers = new Map();
    let generation = 0, connected = true;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => connected, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, asPanel: chatId !== 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const card = (id) => walkCard(messages, id);
    return {
        instance, messages, card, reads, posts,
        connection: (on) => { connected = on; if (!on) handlers.get('close')?.(); },
        part: (id, attr) => card(id)?.querySelector(`[data-${attr}]`),
        narrate: (taskId, content, seconds = 0, extra = {}) => handlers.get('chat')({ chat_id: chatId, role: 'assistant',
            is_progress: true, narration: true, task_id: taskId, content, ts: `2026-10-10T12:00:${String(seconds).padStart(2, '0')}Z`,
            ...extra }),
        frame: (row) => handlers.get('chat')({ chat_id: chatId, ...row }),
        emit: (type, row) => handlers.get(type)?.(row),
        state: (data) => instance.hydrateStateSnapshot({ supervisor_ready: true, active_chat_activities_complete: true,
            active_chat_activities: [], projects: [{ id: 'p-1', name: 'Audit', chat_id: 7 },
                { id: 'p-2', name: 'Release', chat_id: 8 }], ...data }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const room = (extra = {}) => ({ activity_id: 'w-1', chat_id: 7, kind: 'managed_task', phase: 'working', cancelable: true,
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
        assert.ok(f.card('w-1') === node, 'the SAME node, never a second card or an entry');
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
        assert.ok(!node.querySelector('[data-turn-into-project]'), 'a moved work no longer converts');
        assert.equal(messages(f).filter((child) => child.classList.contains('project-handoff')).length, 0);
    } finally { f.close(); }
});

const messages = (f) => f.messages.children;

const status = (f, id = 'w-1') => {
    const chip = f.part(id, 'live-phase');
    const line = f.part(id, 'live-phase-secondary');
    return [chip.hidden ? null : chip.textContent, chip.firstElementChild.dataset.shape, line.hidden ? '' : line.textContent,
        controls(f, id)];
};
const audit = room({ project_id: 'p-1', project_name: 'Audit' });
const startedAudit = { chat_id: 1, task_id: 'w-1', role: 'system', system_type: 'project_started',
    project_id: 'p-1', project_name: 'Audit', task_name: 'Compare notes', ts: '2026-10-10T12:00:00Z', history_id: 'audit-start' };
const stop = { action: 'stop', target: 'w-1' };
const pause = { action: 'pause', target: 'w-1' };
const resume = { action: 'resume', target: 'w-1' };
// Also exercise projections without a Project id: the positively bound room is sufficient.
const work = (...args) => { const value = fixtureWork(...args); delete value.project_id; return value; };
const partial = (works, activities = []) => ({ active_chat_activities_complete: false,
    active_chat_activities: activities, ...projection(7, works, null, false) });
const unconfirmed = ['Activity unconfirmed', 'ring', '', []];

test('conversion leaves Main controls available and a concurrent binding survives its late refusal', async () => {
    let resolve;
    let sends = 0;
    let sent;
    const reply = new Promise(done => { resolve = done; });
    const f = fixture({ conversion: (init) => { sends += 1; sent = JSON.parse(init.body); return reply; } });
    const held = { activity: 'paused', controls: [stop, resume] };
    try {
        f.narrate('w-1', 'Reading the harbor map.');
        const node = f.card('w-1');
        const convert = node.querySelector('[data-turn-into-project]');
        clickControl(convert);
        clickControl(convert);
        assert.equal(sends, 1);
        assert.deepEqual(sent, { task_id: 'w-1' }, 'the server owns the destination id and the name');
        assert.equal(node.dataset.projectCreating, '1');
        assert.equal(convert.disabled, true);

        const main = { ...audit, chat_id: 1 };
        f.state({ ...partial([work(main, held)], [main]), ...projection(1, [work(main, held)], null, false) });
        assert.ok(f.part('w-1', 'resume-run'));
        assert.ok(f.part('w-1', 'cancel-run'));
        f.state({ ...partial([work(audit, held)], [audit]),
            task_bindings: { 'w-1': { project_id: 'p-1', chat_id: 7 } } });
        assert.equal(shownName(node), 'Audit');
        assert.equal(node.dataset.projectCreating, undefined);
        assert.deepEqual(status(f), ['Paused', 'pause', '', ['Resume', 'Stop…']]);

        resolve({ ok: false, status: 409, json: async () => ({ reason_code: 'project_name_unavailable' }) });
        await settle();
        assert.equal(f.card('w-1'), node);
        assert.ok(!node.querySelector('[data-turn-into-project]'));
        assert.deepEqual(status(f), ['Paused', 'pause', '', ['Resume', 'Stop…']]);
    } finally { f.close(); }
});

test('a naming refusal shows the server\'s one sentence and leaves the card and its button for a retry', async () => {
    const sentences = {
        project_name_unavailable: 'The naming model gave no title, so the work stayed in Main. Press Turn into project to try again.',
        project_name_source_unavailable: 'This work couldn\'t be named automatically, so it stayed in Main.',
    };
    const toasts = (text) => {
        const found = [];
        const walk = (node) => { for (const child of node.children || []) { if (child.textContent === text) found.push(child); walk(child); } };
        walk(document.body);
        return found;
    };
    for (const [reason, sentence] of Object.entries(sentences)) {
        const f = fixture({ conversion: async () => ({ ok: false, status: 409, json: async () => ({ error: sentence, reason_code: reason }) }) });
        const restore = quietToasts();
        try {
            f.narrate('w-1', 'Reading the harbor map.');
            const node = f.card('w-1');
            clickControl(node.querySelector('[data-turn-into-project]'));
            await settle();
            assert.equal(toasts(sentence).length, 1, `${reason}: the sentence the server wrote, as written`);
            assert.equal(f.card('w-1'), node, 'the card stays in Main');
            const again = node.querySelector('[data-turn-into-project]');
            assert.ok(again && !again.disabled, 'the existing button is available again');
            assert.equal(again.textContent, 'Turn into project');
            assert.equal(node.dataset.projectCreating, undefined);
        } finally { restore(); f.close(); }
    }
});

test('R2: a projected finalizing outcome never inherits a running row’s axes', () => {
    const f = fixture();
    const running = { ...audit, task_attempt: 1, status: 'running', outcome_axes: { lifecycle: { status: 'running' } } };
    const finishing = work(audit, { transition: 'finalizing', outcome: 'cancelled' });
    try {
        f.frame(startedAudit);
        f.state(partial([finishing], [running]));
        assert.deepEqual(status(f), ['Stopped', 'dot', 'Finalizing…', []]);
        assert.equal(f.card('w-1').dataset.finished, '0');
    } finally { f.close(); }
});

for (const [axis, observed, heading] of [['execution', 'degraded', 'Done with warnings'],
    ['review', 'fail', 'Failed']]) test(`bare completed cannot erase a known ${observed} outcome`, () => {
    const f = fixture();
    const row = { ...audit, task_attempt: 1, phase: 'finalizing', status: 'completed' };
    const finishing = work(audit, { transition: 'finalizing', outcome: 'completed' });
    const axes = { lifecycle: { status: 'completed' }, [axis]: { status: observed } };
    try {
        f.frame(startedAudit);
        f.state(partial([finishing], [{ ...row, outcome_axes: axes }]));
        const chip = [heading, 'dot', 'Finalizing…'];
        assert.deepEqual(status(f).slice(0, 3), chip);
        f.state(partial([finishing], [{ ...row, outcome_axes: { lifecycle: axes.lifecycle } }]));
        assert.deepEqual(status(f).slice(0, 3), chip);
        f.state(partial([finishing]));
        assert.deepEqual(status(f).slice(0, 3), chip, 'the work-only word cannot replace an outcome with axes');
        f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: 'w-1', project_id: 'p-1',
            task_attempt: 1, status: 'completed', outcome_axes: { lifecycle: axes.lifecycle, [axis]: { status: 'ok' } } });
        assert.deepEqual(status(f), ['Done', 'dot', '', []]);
    } finally { f.close(); }
});

const skillReview = (owner = 'w-1') => ({ role: 'system', system_type: 'skill_review', task_id: owner,
    review_group: { surface: 'skill', id: `task:${owner}:audit-skill`, presentation_owner_task_id: owner,
        skill: 'audit-skill', status: 'clean', attempts: [{ job_id: 'audit-job', skill: 'audit-skill', status: 'clean' }] } });

test('R2: an inert Skill review owner becomes a visible portal even while both live and detail reads have gaps', async () => {
    const f = fixture({ fail: { 'w-1': true } });
    try {
        f.frame(skillReview());
        const keeper = f.card('w-1');
        assert.equal(f.part('w-1', 'live-phase').hidden, true, 'review alone is an inert anchor');
        await settle();
        f.state(partial([]));
        f.frame(startedAudit);
        assert.equal(f.card('w-1'), keeper);
        assert.deepEqual(status(f), unconfirmed);
        assert.equal(reference(keeper).getAttribute('aria-label'), 'Open project Audit');
        f.state(partial([work(audit, { controls: [stop] })]));
        assert.deepEqual(status(f), ['Working', 'moving', '', ['Stop…']]);
    } finally { f.close(); }
});

test('R2: a replayed pre-conversion progress row preserves the portal’s legacy Stop trigger', async () => {
    const history = [{ ...startedAudit, role: 'assistant', system_type: '', is_progress: true, narration: true,
        cancelable: true, history_id: 'pre-move', text: 'Starting to compare the notes in Main.' }];
    const f = fixture({ history });
    try {
        f.frame(startedAudit);
        f.state(partial([], [audit]));
        const trigger = f.part('w-1', 'cancel-run');
        assert.ok(trigger);
        trigger.focus();
        await f.instance.refreshHistory({ revision: 1 });
        assert.equal(f.part('w-1', 'cancel-run'), trigger);
        assert.equal(document.activeElement, trigger);
    } finally { f.close(); }
});

test('R2: a portal uses a matching project or its positively known address, not an origin queue row', () => {
    const f = fixture();
    const origin = { ...audit, chat_id: 8, project_id: '' };
    const unlabelled = { ...audit, project_id: '' };
    const snapshot = (bound, row) => f.state({ ...partial([], [row]), task_bindings: bound, projects: [] });
    try {
        f.frame(startedAudit);
        snapshot({}, origin);
        assert.deepEqual(status(f), unconfirmed);
        snapshot(undefined, audit); // positively binds its room even without a registry read
        assert.deepEqual(status(f), ['Working', 'moving', '', ['Stop…']]);
        snapshot({}, unlabelled);
        assert.deepEqual(status(f), ['Working', 'moving', '', ['Stop…']], 'the same portal keeps its last positive bound chat');
        snapshot({}, origin);
        assert.deepEqual(status(f), unconfirmed);
        snapshot({ 'w-1': { project_id: 'p-1', chat_id: 9 } }, unlabelled);
        assert.deepEqual(status(f), unconfirmed, 'an authoritative address change supersedes the remembered chat');
    } finally { f.close(); }
});

test('R2: complete activity census does not settle a portal during incomplete upkeep projection', async () => {
    const f = fixture({ details: { 'w-1': { task_id: 'w-1', status: 'completed' } } });
    try {
        f.frame(startedAudit);
        f.state({ ...projection(7, [], null, false), active_chat_activities: [] });
        await settle();
        assert.equal(f.reads.length, 0, 'no negative inference on an incomplete work read');
        f.state(partial([work(audit, { transition: 'finalizing', outcome: 'completed' })]));
        assert.deepEqual(status(f), ['Done', 'dot', 'Finalizing…', []]);
        assert.equal(f.card('w-1').dataset.finished, '0');
        f.state({ active_chat_activities: [] }); // legacy host, no works/works_complete
        await settle();
        assert.ok(f.reads.includes('w-1'));
        assert.deepEqual(status(f), ['Done', 'dot', '', []]);
    } finally { f.close(); }
});

test('R2: delayed detail can file reviews without concluding an again-live Main portal', async () => {
    let release;
    const f = fixture({ hold: { 'w-1': new Promise(resolve => { release = resolve; }) }, details: {
        'w-1': { task_id: 'w-1', status: 'failed', plan_review_state: { waves_omitted: 0,
            waves: [{ request_fingerprint: 'plan-after-start', aggregate: 'GREEN', closed: true }] } },
    } });
    try {
        f.frame(startedAudit);
        f.state({ active_chat_activities: [] });
        assert.ok(f.reads.includes('w-1'));
        f.state(partial([work(audit, { transition: 'finalizing', outcome: 'completed' })]));
        release();
        await settle();
        assert.deepEqual(status(f), ['Done', 'dot', 'Finalizing…', []]);
        assert.equal(f.card('w-1').dataset.finished, '0');
        assert.match(f.part('w-1', 'live-review-summary')?.textContent || '', /Reviews 1/);
    } finally { release(); f.close(); }
});

for (const [reader, missing, rootOnly] of [['census', false, true], ['census', true, false],
    ['review reference', false, false]]) test(`late ${reader} detail (${missing ? '404' : 'Failed'}) cannot settle after a positive return`, async () => {
    let release;
    const id = 'w-1';
    const details = { [id]: missing ? null : { task_id: id, status: 'failed', plan_review_state: { waves_omitted: 0,
        waves: [{ request_fingerprint: 'plan-after-start', aggregate: 'GREEN', closed: true }] } } };
    const f = fixture({ hold: { [id]: new Promise(resolve => { release = resolve; }) }, details });
    const finishing = ['Done', 'dot', 'Finalizing…', []];
    try {
        f.frame(startedAudit);
        if (reader === 'census') f.state({ active_chat_activities: [] });
        else {
            f.frame({ role: 'system', is_progress: true, system_type: 'review_reference', task_id: id,
                surface: 'plan_review', state_revision: 'a'.repeat(64) });
            await settle();
        }
        assert.deepEqual(f.reads, [id], 'the actual reader starts a held GET');
        const member = { activity: 'working', confirmed: true, transition: 'finalizing', mark: 'moving', live: true };
        const inProgress = rootOnly ? { confirmed: false, members: { ids: [id, 'child'], states: {
            [id]: member, child: { activity: 'unknown', confirmed: false, transition: 'none', mark: 'ring', live: false },
        } } } : {};
        f.state(partial([work(audit, { transition: 'finalizing', outcome: 'completed', ...inProgress })]));
        assert.deepEqual(status(f), finishing);
        f.state(partial([]));
        release();
        await settle();
        if (!missing) assert.match(f.part(id, 'live-review-summary')?.textContent || '', /Reviews 1/);
        assert.equal(f.card(id).dataset.finished, '0', 'the old response cannot close the episode');
        assert.deepEqual(status(f), ['Done', 'dot', '', []], 'the gap keeps the outcome, not the live hold');
        assert.deepEqual(f.reads, [id], 'partial absence cannot start another read');
        details[id] = { task_id: id, status: 'completed' };
        f.state({ active_chat_activities: [] });
        await settle();
        assert.deepEqual(f.reads, [id, id], 'another complete absence permits a fresh read');
        assert.deepEqual(status(f), ['Done', 'dot', '', []]);
        assert.equal(f.card(id).dataset.finished, '1');
        if (!missing) assert.match(f.part(id, 'live-review-summary')?.textContent || '', /Reviews 1/);
        assert.deepEqual(f.posts, []);
    } finally { release(); f.close(); }
});

test('fresh detail with failed open synthesis says Failed; fresh absence cannot hide the outcome', async () => {
    const id = 'w-1';
    const details = { [id]: null };
    const f = fixture({ details });
    try {
        // Finish Main's initial history replay before counting reads from the test's snapshots.
        await f.instance.refreshHistory();
        assert.ok(f.instance.hasPaintedHistory(), 'Main history was applied');
        f.frame(startedAudit);
        f.state({ active_chat_activities: [] });
        await settle();
        assert.equal(status(f)[0], null, 'an absent record hides the chip only while no outcome is known');
        details[id] = { task_id: id, status: 'failed', root_phase_checkpoint: { post_task_synthesis: 'running' } };
        f.state({ active_chat_activities: [] });
        await settle();
        assert.equal(status(f)[0], 'Failed');
        assert.equal(f.card(id).dataset.finished, '0');
        f.state(partial([]));
        assert.deepEqual(status(f), ['Failed', 'dot', '', []]);
        details[id] = null;
        f.state({ active_chat_activities: [] });
        await settle();
        assert.deepEqual(f.reads, [id, id, id]);
        assert.deepEqual(status(f), ['Failed', 'dot', '', []]);
        assert.equal(f.card(id).dataset.finished, '0');
        details[id] = { task_id: id, status: 'failed' };
        f.state({ active_chat_activities: [] });
        await settle();
        assert.equal(f.card(id).dataset.finished, '1');
    } finally { f.close(); }
});

for (const historyFirst of [false, true]) test(`history and work-only state join a visible portal (${historyFirst ? 'history' : 'state'} first)`, async () => {
    let release;
    const historyGate = new Promise(resolve => { release = resolve; });
    const history = [startedAudit, { task_id: 'w-1', chat_id: 1, system_type: 'task_evidence', role: 'system',
        ts: '2026-10-10T12:01:00Z', history_id: 'audit-evidence' }];
    const f = fixture({ history, historyGate });
    const listed = partial([work(audit, { call: 'sending', controls: [stop] })]);
    try {
        if (!historyFirst) f.state(listed);
        release();
        await f.instance.refreshHistory({ revision: 1 });
        if (historyFirst) {
            assert.deepEqual(status(f), unconfirmed, 'a bound card has a visible ring before any live proof');
            f.state(listed);
        }
        assert.deepEqual(status(f), ['Sending', 'moving', '', ['Stop…']]);
        assert.equal(f.reads.length, 0, 'a partial omission does not imply durable detail');
    } finally { f.close(); }
});

test('an existing Main card takes the work-only call and tail immediately when history binds it', () => {
    const f = fixture();
    try {
        f.narrate('w-1', 'Inspecting notes in Main.', 0, { cancelable: true });
        const node = f.card('w-1');
        f.state(partial([work(audit, { call: 'sending', controls: [stop] })]));
        f.frame(startedAudit);
        assert.equal(f.card('w-1'), node);
        assert.deepEqual(status(f), ['Sending', 'moving', '', ['Stop…']], 'the same body is used when the card becomes a portal');
    } finally { f.close(); }
});

test('a failed poll and an offline or projection-less body cannot keep old portal offers or tails', () => {
    const f = fixture();
    const running = partial([work(audit, { call: 'sending', controls: [stop],
        tails: [{ code: 'helpers_running', text: '1 subagent still working', count: 1 }] })]);
    const held = partial([work(audit, { activity: 'paused', controls: [stop, resume] })]);
    const main = room({ ...audit, chat_id: 1 });  // a matching Project id alone is only a lease
    const staleMain = () => f.state(partial([work(main, { controls: [stop, resume] })], [main]));
    try {
        f.narrate('w-1', 'Inspecting notes in Main.', 0, { cancelable: true });
        f.frame(startedAudit);
        f.state(running);
        assert.deepEqual(status(f), ['Sending', 'moving', '1 subagent still working', ['Stop…']]);
        f.instance.hydrateStateSnapshot(null);
        assert.deepEqual(status(f), unconfirmed, 'the REST failure retires an earlier working body');
        f.state(held);
        assert.deepEqual(status(f), ['Paused', 'pause', '', ['Resume', 'Stop…']]);
        f.state({ active_chat_activities_complete: false });
        assert.deepEqual(status(f), unconfirmed, 'an absent works field cannot offer Resume from the previous body');
        f.state(held);
        f.connection(false);
        assert.deepEqual(status(f), unconfirmed, 'the socket close itself removes the commands and live phase');
        f.state(held);
        staleMain();
        assert.deepEqual(status(f), unconfirmed, 'an offline response and the old Main marker grant no action');
        f.connection(true);  // a new connection by itself says nothing about work
        assert.deepEqual(status(f), unconfirmed);
        staleMain();
        assert.deepEqual(status(f), unconfirmed, 'a connected Main-addressed row cannot stand for the bound room');
        f.state(running);
        assert.deepEqual(status(f), ['Sending', 'moving', '1 subagent still working', ['Stop…']]);
    } finally { f.close(); }
});

test('a new finalizing result replaces a bare hold; only a known outcome survives the missing live fact', () => {
    const f = fixture();
    const open = (outcome) => partial([work(audit, { transition: 'finalizing', outcome })]);
    try {
        f.frame(startedAudit);
        f.state(open(undefined));
        assert.deepEqual(status(f), ['Finalizing…', 'moving', '', []]);
        f.state(partial([]));
        assert.deepEqual(status(f), unconfirmed, 'a bare prior hold cannot attest current activity');
        f.state(open(undefined));
        f.state(open('failed'));
        assert.deepEqual(status(f), ['Failed', 'dot', 'Finalizing…', []], 'fresh answer through the existing hold');
        f.state(partial([]));
        assert.deepEqual(status(f), ['Failed', 'dot', '', []], 'observed outcome stays without current Finalizing evidence');
        assert.equal(f.card('w-1').dataset.finished, '0');
        f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: 'w-1', project_id: 'p-1',
            outcome_axes: { lifecycle: { status: 'failed' } }, status: 'failed', content: 'Export failed.' });
        assert.deepEqual(status(f), ['Failed', 'dot', '', []]);
    } finally { f.close(); }
});

for (const [mode, expected] of [
    ['single-target projection replaces census', ['w-1']], ['appearance of an attempt number', ['w-1']],
    ['legacy WS to room poll', ['w-1']], ['ended author with surviving helper', ['w-1']],
    ['census to ended author with surviving helper', ['p']],
    ['WS attempt before poll', []], ['WS attempt followed by an older poll', []],
    ['positive attempt replacement then sparse', []], ['first known attempt then another then sparse', []],
    ['additional stop target', []], ['same group', ['w-1', 'w-2']], ['different bound chat', []],
]) test(`R2: an open Stop action keeps its episode and complete targets (${mode})`, async () => {
    const native = mode === 'legacy WS to room poll' || mode.startsWith('WS attempt');
    const survivor = mode === 'census to ended author with surviving helper';
    const f = fixture({ chatId: native ? 7 : 1 });
    const observer = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() {} };
    const click = (node) => { assert.ok(node); node.dispatchEvent({ type: 'click', target: node, stopPropagation() {} }); };
    const secondStop = { action: 'stop', target: 'w-2' };
    const snapshot = (attempt, { controls = [stop], census = false, projectionOnly = false, ended = false, roomId = 7 } = {}) => {
        const row = { ...audit, task_attempt: attempt, chat_id: roomId };
        f.state({ ...partial(census ? [] : [work(row, { controls, ...(ended ? { episode: { task_id: 'w-1', live: false } } : {}) })],
            projectionOnly ? [] : [row]), task_bindings: { 'w-1': { project_id: 'p-1', chat_id: roomId } } });
    };
    try {
        const id = survivor ? 'p' : 'w-1';
        if (survivor) {
            f.frame({ ...startedAudit, task_id: id });
            f.state(partial([], [{ ...audit, activity_id: id, task_attempt: 1 }]));
        } else if (native) f.narrate('w-1', 'Starting the archive comparison.', 0, { cancelable: true });
        else f.frame(startedAudit);
        if (mode.startsWith('WS attempt')) f.state(partial([], [{ ...audit, task_attempt: 1 }]));
        else if (mode === 'single-target projection replaces census') snapshot(1, { census: true });
        else if (mode === 'same group') snapshot(1, { controls: [stop, secondStop] });
        else if (mode === 'appearance of an attempt number' || mode === 'first known attempt then another then sparse') snapshot(undefined, { projectionOnly: true });
        else if (!native && !survivor) snapshot(1);
        const trigger = f.part(id, 'cancel-run');
        click(trigger);
        const stopNow = document.body.querySelector('[data-task-control="stop_now"]');
        if (mode === 'positive attempt replacement then sparse' || mode === 'first known attempt then another then sparse') {
            if (mode.startsWith('first')) snapshot(1);
            snapshot(2);
            snapshot(undefined, { projectionOnly: true });
        } else if (mode === 'additional stop target' || mode === 'same group') snapshot(1, { controls: [stop, secondStop] });
        else if (mode === 'different bound chat') snapshot(1, { roomId: 9 });
        else if (mode === 'ended author with surviving helper') {
            f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: 'w-1', project_id: 'p-1',
                status: 'completed', outcome_axes: { lifecycle: { status: 'completed' } } });
            snapshot(undefined, { projectionOnly: true, ended: true });
        } else if (survivor) {
            f.state(partial([{ ...SERVER_CASES.helper_after_parent.works[0], room: 7 }]));
            f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: id, project_id: 'p-1',
                task_attempt: 1, status: 'completed', outcome_axes: { lifecycle: { status: 'completed' } } });
        } else if (mode.startsWith('WS attempt')) {
            f.emit('log', { chat_id: 7, data: { type: 'task_started', task_id: id, task_attempt: 2, cancelable: true } });
            if (mode.endsWith('older poll')) f.state(partial([], [{ ...audit, task_attempt: 1 }]));
        } else snapshot(1, { census: native });
        if (!mode.startsWith('WS attempt')) assert.equal(f.part(id, 'cancel-run'), trigger,
            'a current offer keeps the same trigger even if the old press is stale');
        click(stopNow);
        await settle();
        assert.deepEqual(f.posts, expected.map((id) => `/api/tasks/${id}/cancel`));
    } finally { closeTaskControlMenu(); globalThis.IntersectionObserver = observer; f.close(); }
});

test('an addressed ended episode paints its outcome before the completion frame and only offers its surviving Stop', () => {
    const f = fixture();
    try {
        f.frame({ ...startedAudit, task_id: 'p' });
        f.state(partial([{ ...SERVER_CASES.helper_after_parent.works[0], room: 7 }]));
        assert.deepEqual(status(f, 'p'), ['Done', 'dot', '', ['Stop…']]);
        assert.equal(f.card('p').dataset.finished, '1');
        f.state(partial([]));
        assert.deepEqual(status(f, 'p'), ['Done', 'dot', '', []]);
    } finally { f.close(); }
});

const clickControl = node => { assert.ok(node); node.dispatchEvent({ type: 'click', target: node, stopPropagation() {} }); };
const menuItem = action => document.body.querySelector(`[data-task-control="${action}"]`);
const statusLog = (f, type, attempt, extra = {}) => f.emit('log', { chat_id: 7,
    data: { type, task_id: 'w-1', task_attempt: attempt, ...extra } });
function inControlMenu(f, run) {
    const previous = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() {} };
    return Promise.resolve().then(run).finally(() => { closeTaskControlMenu(); globalThis.IntersectionObserver = previous; f.close(); });
}

test('R4: the first in-room physical ending at 0 is not older than queue ordinal 1', () => {
    const f = fixture({ chatId: 7 });
    try {
        f.narrate('w-1', 'First direct turn.', 0, { cancelable: true, task_attempt: 0 });
        f.state(partial([], [{ ...audit, kind: 'direct_chat', task_attempt: 1 }]));
        assert.ok(f.part('w-1', 'cancel-run'));
        statusLog(f, 'task_done', 0, { status: 'completed', duration_sec: 9 });
        assert.equal(f.card('w-1').dataset.finished, '1');
        assert.equal(status(f)[0], 'Done');
        assert.ok(!f.part('w-1', 'cancel-run'));
    } finally { f.close(); }
});

for (const ended of [false, true]) test(`R4: a physical retry has its own Stop after ${ended ? 'census settlement' : 'an old poll'}`, () => {
    const f = fixture({ chatId: 7 });
    return inControlMenu(f, async () => {
        const first = { ...audit, task_attempt: 1 };
        f.narrate('w-1', 'First attempt.', 0, { cancelable: true, task_attempt: 1 });
        f.state(partial([], [first]));
        clickControl(f.part('w-1', 'cancel-run'));
        const old = menuItem('stop_now');
        if (ended) {
            f.state(partial([], [{ ...first, task_attempt: 2, status: 'completed' }]));
            assert.equal(f.card('w-1').dataset.finished, '1');
        }
        statusLog(f, 'task_started', 2);
        statusLog(f, 'tool_call_started', 2, { tool: 'read_file', cancelable: true });
        clickControl(old);
        assert.deepEqual(f.posts, []);
        assert.equal(f.card('w-1').dataset.finished, '0');
        clickControl(f.part('w-1', 'cancel-run'));
        clickControl(menuItem('stop_now'));
        await settle();
        assert.deepEqual(f.posts, ['/api/tasks/w-1/cancel']);
    });
});

const bareFinal = (id = 'w-1') => ({ role: 'system', system_type: 'project_completion_summary', task_id: id,
    project_id: 'p-1', status: 'completed' });

for (const [axis, outcome, headline] of [['review', 'fail', 'Failed'], ['execution', 'degraded', 'Done with warnings']])
    test(`R4: a finished parent with a live helper refines ${outcome}, keeps Stop and publishes its closing facts`, () => {
        let release;
        const details = { p: { task_id: 'p', status: 'completed', task_attempt: 0,
            outcome_axes: { lifecycle: { status: 'completed' }, [axis]: { status: outcome } },
            continuation_offer: { eligible: true }, wake_summary: 'A durable closing sentence.' } };
        const f = fixture({ hold: { p: new Promise(resolve => { release = resolve; }) }, details });
        return inControlMenu(f, async () => {
            try {
                f.frame({ ...startedAudit, task_id: 'p' });
                const surviving = { ...SERVER_CASES.helper_after_parent.works[0], room: 7,
                    members: { ids: ['p', 'child'], states: { p: { queue_attempt: 1 } } } };
                f.state(partial([surviving]));
                const node = f.card('p');
                assert.equal(node.dataset.finished, '1');
                clickControl(f.part('p', 'cancel-run'));
                const stopNow = menuItem('stop_now');
                f.frame(bareFinal('p'));
                f.state(partial([surviving]));
                assert.deepEqual(f.reads, ['p'], 'live-tail projection and bare ending must share one GET');
                release();
                await settle();
                f.state(partial([surviving]));
                assert.deepEqual(f.reads, ['p']);
                assert.equal(f.card('p'), node);
                assert.equal(node.dataset.finished, '1');
                assert.equal(status(f, 'p')[0], headline);
                assert.ok(controls(f, 'p').includes('Continue') && controls(f, 'p').includes('Stop…'));
                assert.equal(f.part('p', 'live-activity').textContent, 'A durable closing sentence.');
                clickControl(stopNow);
                await settle();
                assert.deepEqual(f.posts, ['/api/tasks/p/cancel']);
            } finally { release(); }
        });
    });

const detailResponse = detail => ({ ok: detail !== null, status: detail === null ? 404 : 200, json: async () => detail });

for (const [signal, first] of [['complete census', null], ['bare final', { status: 'failed' }]])
    test(`R4: ${signal} uses one newer detail instead of joining the pre-proof GET`, async () => {
        let release;
        const held = new Promise(resolve => { release = resolve; });
        const f = fixture({ detailReply: async (_id, number) => {
            if (number === 1) await held;
            return detailResponse(number === 1 ? first : { status: 'completed' });
        } });
        try {
            f.frame(startedAudit);
            f.frame({ role: 'system', is_progress: true, system_type: 'review_reference', task_id: 'w-1',
                surface: 'plan_review', state_revision: 'a'.repeat(64) });
            await settle();
            assert.deepEqual(f.reads, ['w-1']);
            f.state(partial([work(audit, { controls: [stop] })], [audit]));
            if (signal === 'complete census') f.state({ works: [], works_complete: true });
            else f.frame(bareFinal());
            await settle();
            assert.deepEqual(f.reads, ['w-1', 'w-1']);
            release();
            await settle();
            assert.equal(f.card('w-1').dataset.finished, '1');
            assert.equal(status(f)[0], 'Done');
            assert.deepEqual(f.posts, []);
        } finally { release(); f.close(); }
    });

for (const refused of [false, true]) test(`R4: late detail after ${refused ? 'a refused Stop' : 'Wrap up'} cannot replace a newer phase`, () => {
    let release;
    const held = new Promise(resolve => { release = resolve; });
    const f = fixture({ stopFailure: refused, detailReply: async () => {
        await held;
        return detailResponse(refused ? { status: 'running' } : { status: 'failed' });
    } });
    return inControlMenu(f, async () => {
        try {
            f.frame(startedAudit);
            f.state(partial([], [{ ...audit, phase: 'thinking' }]));
            clickControl(f.part('w-1', 'cancel-run'));
            clickControl(menuItem(refused ? 'stop_now' : 'finalize'));
            await settle();
            assert.deepEqual(f.posts, ['/api/tasks/w-1/cancel']);
            assert.deepEqual(f.reads, ['w-1']);
            f.state(partial([work(audit, { transition: 'finalizing', outcome: 'completed', controls: [stop] })]));
            const current = status(f).slice(0, 3);
            release();
            await settle();
            assert.equal(f.card('w-1').dataset.finished, '0');
            assert.deepEqual(status(f).slice(0, 3), current);
        } finally { release(); }
    });
});

test('R4: a replayed finalizing bare completion cannot replace a typed observed outcome', async () => {
    const history = [{ ...startedAudit, role: 'system', system_type: 'task_evidence', is_progress: true,
        history_id: 'old-finalizing', task_phase: 'finalizing', status: 'completed', text: 'Earlier interim final.' }];
    const f = fixture({ history });
    try {
        f.frame(startedAudit);
        const row = { ...audit, phase: 'finalizing', status: 'completed',
            outcome_axes: { lifecycle: { status: 'completed' }, review: { status: 'fail' } } };
        f.state(partial([work(audit, { transition: 'finalizing', outcome: 'completed' })], [row]));
        assert.equal(status(f)[0], 'Failed');
        f.state(partial([]));
        await f.instance.refreshHistory({ revision: 1 });
        f.state(partial([]));
        assert.deepEqual(status(f), ['Failed', 'dot', '', []]);
    } finally { f.close(); }
});

test('R4: late review detail does not remount a card after its history page is released', async () => {
    let release;
    const id = 'w-1', history = [
        { ...startedAudit, history_id: 'old-progress', role: 'assistant', system_type: '', is_progress: true,
            narration: true, text: 'Read the archived notes.' },
        { ...startedAudit, history_id: 'old-ending', system_type: 'task_summary', status: 'completed',
            outcome_final: true, outcome_phase: 'done', outcome_axes: { lifecycle: { status: 'completed' } } },
    ];
    const details = { [id]: { status: 'completed', plan_review_state: { waves_omitted: 0,
        waves: [{ request_fingerprint: 'plan-after-history', aggregate: 'GREEN', closed: true }] } } };
    const f = fixture({ history, details, hold: { [id]: new Promise(resolve => { release = resolve; }) } });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        assert.equal(f.card(id)?.dataset.finished, '1');
        f.frame({ role: 'system', is_progress: true, system_type: 'review_reference', task_id: id,
            surface: 'plan_review', state_revision: 'a'.repeat(64) });
        await settle();
        assert.deepEqual(f.reads, [id]);
        // This entire old card has scrolled away; the next full history omits it.
        const offscreen = node => {
            node.getBoundingClientRect = () => ({ top: -40, bottom: -20, left: 0, right: 100, width: 100, height: 20 });
            for (const child of node.children) offscreen(child);
        };
        offscreen(f.card(id));
        history.length = 0;
        await f.instance.refreshHistory({ revision: 2 });
        assert.ok(!f.card(id), 'the old card actually leaves before the response');
        release();
        await settle();
        assert.ok(!f.card(id), 'an old review read is not a new anchor');
        assert.deepEqual(f.reads, [id]);
    } finally { release(); f.close(); }
});

test('R4: a direct turn on a projection gap needs addressed positive Stop evidence', () => {
    const f = fixture({ chatId: 7 });
    const direct = { ...audit, kind: 'direct_chat', cancelable: undefined, task_attempt: 1 };
    try {
        f.narrate('w-1', 'An ordinary direct turn.');
        f.state(partial([], [direct]));
        assert.ok(!f.part('w-1', 'cancel-run'));
        f.narrate('w-1', 'A direct turn whose current frame offers Stop.', 1, { cancelable: true, task_attempt: 0 });
        f.state(partial([], [direct]));
        assert.deepEqual(controls(f, 'w-1'), ['Stop…']);
    } finally { f.close(); }
});

for (const current of [true, false]) test(`R4: census model waiting affects controls only on its current attempt (${current})`, () => {
    const f = fixture();
    return inControlMenu(f, () => {
        f.frame(startedAudit);
        f.state(partial([], [{ ...audit, task_attempt: 2, model_waits: { wait: { state: 'waiting', task_attempt: current ? 2 : 1 } } }]));
        clickControl(f.part('w-1', 'cancel-run'));
        assert.ok(menuItem('pause') && menuItem('stop_now'));
        assert.equal(Boolean(menuItem('hurry') && menuItem('finalize')), !current);
    });
});

test('R4: a withdrawn then restored Pause cannot use the old menu; continuously offered Stop can', () => {
    const f = fixture();
    return inControlMenu(f, async () => {
        const state = controls => f.state(partial([work(audit, { controls })]));
        f.frame(startedAudit);
        state([pause, stop]);
        clickControl(f.part('w-1', 'cancel-run'));
        const oldPause = menuItem('pause'), continuousStop = menuItem('stop_now');
        state([stop]);
        state([pause, stop]);
        clickControl(oldPause);
        clickControl(continuousStop);
        await settle();
        assert.deepEqual(f.posts, ['/api/tasks/w-1/cancel']);
    });
});

for (const phase of ['queued', 'budget_pausing']) test(`legacy ${phase} keeps Pause and Stop, not Hurry or Wrap up`, () => {
    const f = fixture();
    const observer = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    try {
        f.frame(startedAudit);
        f.state(partial([], [{ ...audit, phase }]));
        f.part('w-1', 'cancel-run').dispatchEvent({ type: 'click', stopPropagation() {} });
        assert.deepEqual(document.body.querySelectorAll('[data-task-control]').map(el => el.dataset.taskControl), ['pause', 'stop_now']);
    } finally { closeTaskControlMenu(); globalThis.IntersectionObserver = observer; f.close(); }
});

test('R2: Presence census alone offers only Stop unless moved text can Pause; explicit work actions still decide', async () => {
    const f = fixture();
    const observer = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() {} };
    const click = (node) => { assert.ok(node); node.dispatchEvent({ type: 'click', target: node, stopPropagation() {} }); };
    const open = () => click(f.part('w-1', 'cancel-run'));
    const item = (action) => document.body.querySelector(`[data-task-control="${action}"]`);
    const presence = { ...audit, kind: 'presence', task_attempt: 1 };
    try {
        f.frame(startedAudit);
        f.state(partial([], [audit]));
        open();
        assert.ok(item('hurry') && item('finalize'), 'legacy managed controls survive');
        f.state(partial([], [presence]));
        open();
        assert.ok(!item('hurry') && !item('finalize') && item('stop_now'));
        f.state(partial([], [{ ...presence, presence_pause: true }]));
        open();
        assert.ok(item('pause') && item('stop_now') && !item('hurry') && !item('finalize'));
        const addressedGap = work(presence, { confirmed: false,
            gaps: [{ reason: 'episode_record_unavailable', task_id: 'w-1' }], controls: [stop] });
        f.state(partial([addressedGap], [presence]));
        open();
        assert.ok(item('stop_now') && !item('hurry'), 'a listed guarded Stop survives an unreadable episode result');
        f.state(partial([{ ...addressedGap, controls: [] }], [presence]));
        assert.deepEqual(controls(f, 'w-1'), [], 'an addressed work with no offer is not a license for legacy controls');
        f.state(partial([work(presence, { place_turn: true, controls: [stop, pause] })], [presence]));
        open();
        assert.ok(item('pause'), 'the work names the sanctioned Pause');
        click(item('stop_now'));
        await settle();
        assert.deepEqual(f.posts, ['/api/tasks/w-1/cancel']);
    } finally { closeTaskControlMenu(); globalThis.IntersectionObserver = observer; f.close(); }
});

test('an open portal Stop menu checks Pause and Stop at dispatch, without disabling a newly proved action', async () => {
    const f = fixture();
    const observer = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    const click = (node) => { assert.ok(node); node.dispatchEvent({ type: 'click', target: node, stopPropagation() {} }); };
    const menu = (action) => {
        click(f.part('w-1', 'cancel-run'));
        return document.body.querySelector(`[data-task-control="${action}"]`);
    };
    try {
        f.frame(startedAudit);
        f.state(partial([work(audit, { controls: [pause, stop] })]));
        const stalePause = menu('pause');
        f.instance.hydrateStateSnapshot(null);
        click(stalePause);
        f.state(partial([work(audit, { controls: [pause, stop] })]));
        const staleStop = menu('stop_now');
        f.state(partial([]));
        click(staleStop);
        await settle();
        assert.deepEqual(f.posts, [], 'neither stale menu item submitted a request');

        const running = () => f.state(partial([work(audit, { controls: [pause, stop] })]));
        running();
        const currentPause = menu('pause');
        running();  // a new body carrying the same attempt and target does not revoke the choice
        click(currentPause);
        await settle();
        assert.deepEqual(f.posts, ['/api/tasks/w-1/pause'], 'same action on the current work posts once');
        f.state(partial([work(audit, { activity: 'paused', controls: [resume] })]));
        const staleResume = f.part('w-1', 'resume-run');
        f.instance.hydrateStateSnapshot(null);
        click(staleResume);
        assert.deepEqual(f.posts, ['/api/tasks/w-1/pause']);
        f.state(partial([work(audit, { activity: 'paused', controls: [resume] })]));
        click(f.part('w-1', 'resume-run'));
        await settle();
        assert.deepEqual(f.posts, ['/api/tasks/w-1/pause', '/api/tasks/w-1/resume']);
    } finally { closeTaskControlMenu(); globalThis.IntersectionObserver = observer; f.close(); }
});

test('an addressed, cancelable Presence turn keeps a hard Stop even when its phase is unknown', async () => {
    const f = fixture();
    const observer = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    const click = (node) => { assert.ok(node); node.dispatchEvent({ type: 'click', target: node, stopPropagation() {} }); };
    try {
        f.frame(startedAudit);
        f.state(partial([], [{ ...audit, kind: 'presence', presence_pause: true, phase: 'unknown' }]));
        assert.deepEqual(status(f), ['Activity unconfirmed', 'ring', '', ['Stop…']]);
        click(f.part('w-1', 'cancel-run'));
        assert.ok(!document.body.querySelector('[data-task-control="pause"]'));
        click(document.body.querySelector('[data-task-control="stop_now"]'));
        await settle();
        assert.deepEqual(f.posts, ['/api/tasks/w-1/cancel']);
    } finally { closeTaskControlMenu(); globalThis.IntersectionObserver = observer; f.close(); }
});

test('a partial census unconfirms only the omitted portal until the next positive read', async () => {
    const f = fixture();
    const a = room({ project_id: 'p-1', project_name: 'Audit', last_line: 'Comparing the source notes.' });
    const b = room({ activity_id: 'w-2', chat_id: 8, project_id: 'p-2', project_name: 'Release' });
    const stop = ({ activity_id: target }) => [{ action: 'stop', target }];
    const working = work(a, { controls: stop(a) });
    const unrelated = work(b, { controls: stop(b) });
    const sending = { ...a, model_activity: { phase: 'awaiting_response' } };
    const sendingWork = work(a, { call: 'sending', controls: stop(a),
        tails: [{ code: 'helpers_running', text: '1 subagent still working', count: 1 }] });
    const snapshot = (complete, activities, works) => f.state({ active_chat_activities_complete: complete,
        active_chat_activities: activities, ...projection(7, works, null, complete) });
    const status = (id) => {
        const chip = f.part(id, 'live-phase');
        const secondary = f.part(id, 'live-phase-secondary');
        return [chip.textContent, chip.firstElementChild.dataset.shape, secondary.hidden ? '' : secondary.textContent,
            controls(f, id)];
    };
    try {
        for (const row of [a, b]) f.frame({ role: 'system', system_type: 'project_started', task_id: row.activity_id,
            project_id: row.project_id, project_name: row.project_name, task_name: `${row.project_name} work` });
        snapshot(false, [a, b], [working, unrelated]);
        await settle();
        const first = f.card('w-1'), second = f.card('w-2'), door = reference(first);
        const running = ['Working', 'moving', '', ['Stop…']];
        assert.deepEqual(status('w-1'), running, 'a listed portal takes its own phase even in a partial read');
        snapshot(true, [sending, b], [sendingWork, unrelated]);
        await settle();
        assert.deepEqual(status('w-1'), ['Sending', 'moving', '1 subagent still working', ['Stop…']]);

        snapshot(false, [b], [unrelated]);
        await settle();
        assert.deepEqual(status('w-1'), ['Activity unconfirmed', 'ring', '', []], 'omission is not current activity');
        assert.ok(f.card('w-1') === first && reference(first) === door && !first.hidden && first.dataset.finished === '0');
        assert.ok(!f.reads.includes('w-1'), 'partial absence does not send the portal down the detail path');
        assert.ok(f.card('w-2') === second);
        assert.deepEqual(status('w-2'), running, 'the other work is still listed');

        snapshot(true, [a, b], [working, unrelated]);
        await settle();
        assert.ok(f.card('w-1') === first && reference(first) === door);
        assert.deepEqual(status('w-1'), running);
    } finally { f.close(); }
});

test('a moved card follows fresh work projection, not a retained Main row or control marker', async () => {
    const f = fixture();
    const moved = room({ project_id: 'p-1', project_name: 'Audit' });
    const offered = work(moved, { call: 'sending', controls: [{ action: 'stop', target: 'w-1' }] });
    const partial = (activities, works = []) => f.state({ active_chat_activities_complete: false,
        active_chat_activities: activities, ...projection(7, works, null, false) });
    const chip = () => f.part('w-1', 'live-phase');
    try {
        f.narrate('w-1', 'Checking the notes.', 0, { cancelable: true });
        f.state({ active_chat_activities: [room({ chat_id: 1, cancelable: true })] });
        assert.ok(controls(f, 'w-1').includes('Stop…'));
        partial([moved], [work(moved)]);
        assert.equal(f.card('w-1').dataset.portal, '1');
        assert.equal(chip().textContent, 'Working', 'the new room row outranks Main’s retained, unconfirmed row');
        partial([]);
        assert.equal(chip().textContent, 'Activity unconfirmed');
        assert.deepEqual(controls(f, 'w-1'), [], 'the old Main marker is not current portal control authority');
        partial([], [work(moved)]);
        assert.equal(chip().textContent, 'Working', 'a positive work read need not include the activity row');
        assert.deepEqual(controls(f, 'w-1'), [], 'a work that offers nothing does not inherit Main’s old Stop');
        partial([], [offered]);
        f.frame({ role: 'system', system_type: 'task_evidence', task_id: 'w-1' });
        assert.equal(chip().textContent, 'Sending', 'a listed, confirmed current episode speaks without a census row');
        assert.equal(chip().firstElementChild.dataset.shape, 'moving');
        assert.deepEqual(controls(f, 'w-1'), ['Stop…'], 'the work itself lists the control');
    } finally { f.close(); }
});

test('a partial omission does not replace a portal’s observed finalizing outcome', async () => {
    const f = fixture();
    try {
        f.narrate('w-1', 'Checking the notes.');
        f.state({ active_chat_activities: [room({ chat_id: 1 })] });
        f.state({ active_chat_activities_complete: false,
            active_chat_activities: [room({ project_id: 'p-1', phase: 'finalizing', status: 'completed' })] });
        const phase = () => [f.part('w-1', 'live-phase').textContent, f.part('w-1', 'live-phase-secondary').textContent];
        assert.deepEqual(phase(), ['Done', 'Finalizing…']);
        f.state({ active_chat_activities_complete: false });
        f.frame({ role: 'system', system_type: 'task_evidence', task_id: 'w-1' });
        assert.deepEqual(phase(), ['Done', '']);
        assert.deepEqual(controls(f, 'w-1'), []);
        assert.equal(f.card('w-1').dataset.finished, '0');
        f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: 'w-1', project_id: 'p-1',
            outcome_axes: { lifecycle: { status: 'completed' } }, status: 'completed', content: 'Finished.' });
        assert.deepEqual(phase(), ['Done', '']);
        assert.equal(f.card('w-1').dataset.finished, '1');
    } finally { f.close(); }
});

test('a portal whose attempt is selected without a scalar keeps the census count on its line', async () => {
    const f = fixture();
    try {
        f.narrate('w-1', 'Reading the logs before the audit.');
        // A journal-only metrics frame for a numbered attempt selects the attempt (its scalar
        // store empties) without stating this attempt's counters: the census count is the work's
        // whole present and must keep the line, not the empty attempt store.
        f.emit('log', { chat_id: 1, data: { task_id: 'w-1', type: 'task_metrics_event', task_attempt: 1,
            tool_evidence: { observations: [{ key: 'one', tool: 'read_file', fact: 'settled', status: 'ok' }] } } });
        await settle();
        f.state({ task_bindings: { 'w-1': { project_id: 'p-1', chat_id: 7 } }, projects: [{ id: 'p-1', name: 'Audit' }],
            active_chat_activities: [room({ last_line: 'Checking the third project.', last_line_at: '2026-10-10T12:05:00Z',
                tool_calls: 41 })] });
        await settle();
        const node = f.card('w-1');
        assert.ok(node, 'the work card is in Main');
        assert.equal(node.dataset.portal, '1');
        assert.match(f.part('w-1', 'live-meta').innerHTML, /41 tool calls/);
    } finally { f.close(); }
});

test('an older census without the new facts leaves today\'s card: its own line, a start time, no invented count', async () => {
    const f = fixture();
    try {
        f.narrate('w-1', 'Reading the logs before the audit.');
        f.narrate('w-1', 'Comparing the two runs.', 1);
        const before = { title: f.part('w-1', 'live-title').textContent, line: f.part('w-1', 'live-activity').textContent };
        f.state({ projects: [], task_bindings: { 'w-1': { project_id: 'p-1' } }, active_chat_activities: [room()] });
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
        f.state({ projects: [], active_chat_activities: [room({ project_id: 'p-2', project_name: 'Second room', detached: true,
            last_line: 'Queued behind the build.' })] });
        await settle();
        const node = f.card('w-1');
        assert.equal(node.dataset.portal, '1');
        assert.equal(reference(node).getAttribute('aria-label'), 'Open project Second room');
        assert.equal(f.part('w-1', 'live-phase-secondary')?.textContent, 'in background');
    } finally { f.close(); }
});

test('a Main-addressed row\'s Project ID is a folder lease, not a room: the card stays Main\'s own', async () => {
    // The scope/census twin of the server rule (gateway/main_decisions.py): a Main start in a registered
    // room's folder carries that room's own ID as its lease scope while its census row keeps Main's
    // address. The ID alone binds nothing; a real move re-homes the row's address with its binding.
    const f = fixture();
    try {
        f.narrate('w-1', 'Reading the logs before the audit.');
        const node = f.card('w-1');
        f.state({ projects: [{ id: 'p-2', name: 'Second room' }],
            active_chat_activities: [room({ chat_id: 1, project_id: 'p-2', project_name: 'Second room' })] });
        await settle();
        assert.ok(f.card('w-1') === node, 'the SAME card stays mounted in Main');
        assert.ok(node.dataset.portal !== '1', 'a lease is no move: the card is no portal');
        assert.equal(node.querySelectorAll('[data-intent="open-project"]').length, 0,
            'the scope alone draws no Project reference');
        // The host re-homes the row's address with its durable binding: now the same ID is the room.
        f.state({ task_bindings: { 'w-1': { project_id: 'p-2', chat_id: 7 } },
            projects: [{ id: 'p-2', name: 'Second room' }],
            active_chat_activities: [room({ chat_id: 7, project_id: 'p-2', project_name: 'Second room' })] });
        await settle();
        assert.ok(f.card('w-1') === node, 'a real move keeps the work\'s one card');
        assert.equal(node.dataset.portal, '1');
        assert.equal(reference(node).getAttribute('aria-label'), 'Open project Second room');
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

test('the completion row carries the root\'s closing sentence to its portal; without one the old line goes', async () => {
    const f = fixture();
    try {
        for (const id of ['w-6', 'w-7']) f.narrate(id, 'Copying the export presets into the v3 store.');
        f.state({ task_bindings: { 'w-6': { project_id: 'p-1', chat_id: 7 }, 'w-7': { project_id: 'p-1', chat_id: 7 } },
            active_chat_activities: [room({ activity_id: 'w-6' }), room({ activity_id: 'w-7' })] });
        await settle();
        const done = { role: 'system', system_type: 'project_completion_summary', project_id: 'p-1', project_name: 'Audit',
            outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } }, status: 'completed' };
        f.frame({ ...done, task_id: 'w-6', content: 'Audit › Export · Done', ts: '2026-10-10T13:00:00Z',
            wake_summary: 'The export settings moved to v3; nothing needed you.' });
        f.frame({ ...done, task_id: 'w-7', content: 'Audit › Export · Done', ts: '2026-10-10T13:00:01Z' });
        await settle();
        assert.equal(f.part('w-6', 'live-phase').textContent, 'Done');
        assert.equal(f.part('w-6', 'live-activity').textContent, 'The export settings moved to v3; nothing needed you.',
            'the row\'s own closing sentence replaces the census line');
        assert.equal(f.part('w-7', 'live-activity').textContent, '', 'no recorded sentence: nothing, never the old census line');
        assert.ok(!f.reads.includes('w-6'), 'the row carries the sentence: no record read');
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
// A task detail's end as the host records it (terminal_time.py): the executor observed it.
const endedAt = (occurred_at, started_at) => ({ v: 1, occurred_at, source: 'executor_terminal', attempt: { started_at } });
const failedDetail = (extra = {}) => ({ task_id: 'w-1', status: 'failed', reason_code: 'model_provider_failed',
    terminal_time: endedAt('2026-10-10T12:20:00Z', '2026-10-10T12:00:00Z'), ...extra });
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
        assert.ok(!f.card('w-1-c1'), 'without the successor\'s own facts Main mints nothing');
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
        assert.ok(!f.card('w-9'), 'a root its record does not name is not its Continue');
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

test('after a Main reload whose census arrives before the history draws the card, that census still decides it', async () => {
    // The page's state read can land before its history replay mounts the start row's card; the next census is a
    // poll away, so the card drawn later meets the census already read (both directions: before and after the replay).
    const history = [
        { role: 'user', text: 'audit the projects in their own room', ts: '2026-10-10T11:59:00Z', chat_id: 1, source: 'web',
            history_id: 'h-1', row_address: 'row:1@h#1' },
        { role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-1', project_id: 'p-1', project_name: 'Audit',
            task_name: 'Audit the projects', text: 'Audit › Audit the projects · Started', ts: '2026-10-10T12:00:00Z',
            history_id: 'h-2' },
    ];
    // The census is applied at once; the instance also reads its history by itself after its hydration window (a timer).
    // On a host with a coarse timer (a Windows runner) that read landed inside `settle()`, before the check below and
    // the test's own read. The gate holds every history read until the test opens it.
    let open;
    const historyGate = new Promise((resolve) => { open = resolve; });
    const f = fixture({ history, historyGate, details: { 'w-1': failedDetail(continuedBy()) } });
    try {
        f.state({ task_bindings: bindings, projects: [{ id: 'p-1', name: 'Audit' }], active_chat_activities: [successorRow()] });
        await settle();
        assert.ok(!f.card('w-1'), 'no card before the history is read');
        const read = f.instance.refreshHistory({ revision: 1 });
        open();
        await read;
        await settle();
        assertFollowed(f, f.card('w-1'));
    } finally { open(); f.close(); }
});

test('a record read the reload started and Main left before its answer changes nothing away; the return reconciles', async () => {
    // Quiet rooms (PR #2951): Main left while the replay's read of the moved work's record is in flight neither draws
    // the answer nor reads again while away; shown again, it takes the state and follows the admitted link.
    const history = [
        { role: 'user', text: 'audit the projects in their own room', ts: '2026-10-10T11:59:00Z', chat_id: 1, source: 'web',
            history_id: 'h-1', row_address: 'row:1@h#1' },
        { role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-1', project_id: 'p-1', project_name: 'Audit',
            task_name: 'Audit the projects', text: 'Audit › Audit the projects · Started', ts: '2026-10-10T12:00:00Z',
            history_id: 'h-2' },
    ];
    let release;
    const hold = { 'w-1': new Promise((resolve) => { release = resolve; }) };
    const f = fixture({ history, hold, details: { 'w-1': failedDetail(continuedBy()) } });
    const census = { task_bindings: bindings, projects: [{ id: 'p-1', name: 'Audit' }], active_chat_activities: [successorRow()] };
    try {
        f.state(census);
        await settle();
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        assert.ok(f.reads.includes('w-1'), 'the replayed card reads its record');
        const asked = f.reads.length;
        f.instance.leave();
        delete hold['w-1'];
        release();
        await settle();
        assert.ok(!f.card('w-1-c1'), 'nothing is drawn while Main is away');
        assert.equal(f.card('w-1').hidden, false, 'the old card stays as it was left');
        assert.equal(f.reads.length, asked, 'no new read while away');
        f.instance.stay();
        f.state(census);
        await settle();
        assertFollowed(f, f.card('w-1'));
    } finally { f.close(); }
});

test('a census read before a disconnect never decides a later replay: the card stays unconfirmed until the next census', async () => {
    // A reconnect reads the history again, and its answer can land before the post-open state read (which can also
    // fail): the replay meets only a census read after the outage, so the card the outage parked stays unconfirmed.
    const history = [
        { role: 'user', text: 'audit the projects in their own room', ts: '2026-10-10T11:59:00Z', chat_id: 1, source: 'web',
            history_id: 'h-1', row_address: 'row:1@h#1' },
        { role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-1', project_id: 'p-1', project_name: 'Audit',
            task_name: 'Audit the projects', text: 'Audit › Audit the projects · Started', ts: '2026-10-10T12:00:00Z',
            history_id: 'h-2' },
    ];
    const f = fixture({ history });
    const census = { task_bindings: { 'w-1': { project_id: 'p-1', chat_id: 7 } }, projects: [{ id: 'p-1', name: 'Audit' }],
        active_chat_activities: [room()] };
    const phase = () => f.part('w-1', 'live-phase').dataset.phase;
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        f.state(census);
        await settle();
        assert.equal(phase(), 'working', 'the census before the outage shows the work running');
        f.emit('close');
        await settle();
        assert.equal(phase(), 'unknown', 'the outage parks the card as unconfirmed');
        await f.instance.refreshHistory({ revision: 2 });
        await settle();
        assert.equal(phase(), 'unknown', 'the replay after the outage does not bring the old census back');
        f.state(census);
        await settle();
        assert.equal(phase(), 'working', 'the census read after the outage decides');
    } finally { f.close(); }
});

test('a card first bound after a disconnect takes nothing from the census read before it', async () => {
    // The census before the outage lists a work Main holds no card for; the replay after the reconnect draws its start
    // row. Until a census read after the outage, the new card shows neither that room's line nor its Stop.
    const history = [];
    const started = { role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-2', project_id: 'p-1',
        project_name: 'Audit', task_name: 'Audit the projects', text: 'Audit › Audit the projects · Started',
        ts: '2026-10-10T12:00:00Z', history_id: 'h-2' };
    const f = fixture({ history });
    const census = { task_bindings: { 'w-2': { project_id: 'p-1', chat_id: 7 } }, projects: [{ id: 'p-1', name: 'Audit' }],
        active_chat_activities: [room({ activity_id: 'w-2', last_line: 'Checking the third project.', cancelable: true })] };
    try {
        f.state(census);
        await settle();
        assert.ok(!f.card('w-2'), 'no card before its start row is read');
        f.emit('close');
        await settle();
        history.push(started);
        await f.instance.refreshHistory({ revision: 2 });
        await settle();
        assert.equal(f.card('w-2')?.dataset.portal, '1', 'the replay binds the work\'s card');
        assert.notEqual(f.part('w-2', 'live-activity').textContent, 'Checking the third project.',
            'the room\'s line from before the outage is not drawn');
        assert.ok(!controls(f, 'w-2').includes('Stop…'), 'no Stop from the census read before the outage');
        f.state(census);
        await settle();
        assert.equal(f.part('w-2', 'live-activity').textContent, 'Checking the third project.', 'the next census decides');
        assert.ok(controls(f, 'w-2').includes('Stop…'));
    } finally { f.close(); }
});

test('a complete census read before a disconnect proves no work gone after it', async () => {
    // A work started during the outage is absent from the complete census read before it. A replay after the close
    // binds its card: the old census does not drop its line or send it to the missing-work read.
    const history = [{ role: 'user', text: 'audit the projects', ts: '2026-10-10T11:59:00Z', chat_id: 1, source: 'web',
        history_id: 'h-1', row_address: 'row:1@h#1' }];
    const f = fixture({ history });
    const reads = () => f.reads.filter((id) => id === 'w-2').length;
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        f.state({ active_chat_activities: [] });  // complete, and w-2 has not started yet
        await settle();
        f.emit('close');
        await settle();
        f.narrate('w-2', 'Started during the outage.');
        await settle();
        const before = reads();
        history.push({ role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-2', project_id: 'p-1',
            project_name: 'Audit', task_name: 'Audit the projects', text: 'Audit › Audit the projects · Started',
            ts: '2026-10-10T12:00:00Z', history_id: 'h-2' });
        await f.instance.refreshHistory({ revision: 2 });
        await settle();
        assert.equal(f.card('w-2')?.dataset.portal, '1', 'the replay binds the work\'s card');
        assert.equal(reads(), before, 'no missing-work read: the old census proves no absence');
    } finally { f.close(); }
});

test('a history replay retries no failed successor read; the next census does', async () => {
    // A followed successor whose record read failed is read again at the next census (sync), never by a history read,
    // whose answer may land while Main is away.
    const history = [
        { role: 'user', text: 'audit the projects in their own room', ts: '2026-10-10T11:59:00Z', chat_id: 1, source: 'web',
            history_id: 'h-1', row_address: 'row:1@h#1' },
        { role: 'system', system_type: 'project_started', chat_id: 1, task_id: 'w-1', project_id: 'p-1', project_name: 'Audit',
            task_name: 'Audit the projects', text: 'Audit › Audit the projects · Started', ts: '2026-10-10T12:00:00Z',
            history_id: 'h-2' },
    ];
    const f = fixture({ history, fail: { 'w-1-c1': true },
        details: { 'w-1': failedDetail({ ...continuedBy(), wake_summary: 'The first attempt stopped here.' }) } });
    const census = { task_bindings: bindings, projects: [{ id: 'p-1', name: 'Audit' }], active_chat_activities: [] };
    const successorReads = () => f.reads.filter((id) => id === 'w-1-c1').length;
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        f.state(census);
        await settle();
        assert.equal(successorReads(), 1, 'the admitted successor\'s record is read once, and that read fails');
        await f.instance.refreshHistory({ revision: 2 });
        await settle();
        assert.equal(successorReads(), 1, 'a history read does not read it again');
        f.state(census);
        await settle();
        assert.equal(successorReads(), 2, 'the next census does');
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
        assert.ok(!f.card('w-1-c1'), 'a bound claim mints no successor card');
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
                assert.ok(!f.card('w-1-c1'), 'no successor card before the successor speaks');
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
    terminal_time: endedAt('2026-10-10T12:42:00Z', '2026-10-10T12:30:00Z'), ...extra });

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
            details['w-1'] = failedDetail({ ...continuedBy(claim), wake_summary: 'The first attempt stopped here.' });
            details['w-1-c1'] = successor;
            // Failed, continued in its room, and the successor already finished: the census lists neither.
            f.state({ task_bindings: { ...bindings, 'w-1-c1': { project_id: 'p-1', chat_id: 7 } }, active_chat_activities: [] });
            await settle();
            f.frame({ role: 'system', system_type: 'project_completion_summary', task_id: 'w-1-c1', project_id: 'p-1',
                project_name: 'Audit', content: 'The audit is complete.', ts: '2026-10-10T12:42:30Z' });
            await settle();
            if (claim === 'bound' || successor === null) {
                assert.ok(!f.card('w-1-c1'), `${claim}/${successor ? 'record' : 'no record'}: no successor card`);
                assert.equal(node.hidden, false, 'the old card stays as it was');
                assert.equal(f.part('w-1', 'live-phase').textContent, 'Failed');
                assert.equal(f.reads.includes('w-1-c1'), claim !== 'bound', `${claim}: only an admitted claim reads its successor`);
                continue;
            }
            assertSettledFollow(f, [node]);
            assert.equal(f.part('w-1-c1', 'live-activity').textContent, '', 'Continue without a sentence leaves no old closing line');
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
    const f = fixture({ history, details: { 'w-1': failedDetail({ ...continuedBy(), wake_summary: 'The first attempt stopped here.' }), 'w-1-c1': finishedSuccessor() } });
    try {
        await f.instance.refreshHistory({ revision: 1 });
        await settle();
        f.state({ task_bindings: bindings, projects: [{ id: 'p-1', name: 'Audit' }], active_chat_activities: [] });
        await settle();
        assertSettledFollow(f, [f.card('w-1')]);
        assert.equal(f.part('w-1-c1', 'live-activity').textContent, '', 'reload keeps the second episode free of the first sentence');
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
            terminal_time: endedAt('2026-10-10T12:35:00Z', '2026-10-10T12:30:00Z'),
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
            assert.ok(!f.card(other), `${claim}: no second card for ${other}`);
            assert.equal(node.hidden, false, 'the old card stays');
            assert.equal(f.part('w-1', 'live-phase').textContent, 'Failed');
            assert.ok(reads() > read, 'the observation made the settled portal read its own record');
            assert.ok(reads() <= read + 2, 'once per observation, never a poll');
            assert.deepEqual(controls(f, 'w-1'), [claim === 'bound' ? 'Retry Continue' : 'Continue']);
        } finally { f.close(); }
    }
});

for (const [words, expected] of [
    [{ wakeSummary: 'Export fixed.' }, 'Export fixed.'],
    [{ wakeSummary: 'Export fixed.', suggestedName: 'Export fixed.' }, ''],
    [{ wakeSummary: 'Export fixed.', causeHeld: true, collapsedActivity: 'The upload failed.' }, 'The upload failed.'],
    [{ wakeSummary: 'Export fixed.', finished: false }, ''],
    [{}, ''],
]) test(`a portal leaving the census keeps its closing words (${expected})`, () => {
    const { prior, mount } = installDom();
    try {
        const root = document.createElement('div'); mount.appendChild(root); root.dataset.portal = '1';
        const activityEl = document.createElement('div'); root.appendChild(activityEl);
        const record = { root, groupId: 'w-1', activityEl, finished: true, collapsedActivity: 'Old activity.', ...words };
        const portals = createWorkPortals({ isMain: true, feed: mount, records: new Map([['w-1', record]]),
            onDomWrite: write => write(), card: new Proxy({}, { get: () => () => false }) });
        portals.sync({ supervisor_ready: true, active_chat_activities_complete: true, active_chat_activities: [] });
        assert.equal(record.collapsedActivity, expected);
        assert.equal(activityEl.textContent, expected);
    } finally { restoreDom(prior); }
});
