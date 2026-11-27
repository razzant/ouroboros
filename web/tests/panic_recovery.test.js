// After Panic the owner chooses which saved works come back (docs/DESIGN.md "Project question mirror"): one System
// message in Main names them, opening and reconnecting send nothing, one press sends exactly the chosen rows each with
// the stop it was chosen under, and each row then says what its door answered.
import assert from 'node:assert/strict';
import test from 'node:test';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { createMainWaiting } from '../modules/main_waiting.js';
import { createPanicRecovery, panicCards, panicRow, SHOWN_ROWS } from '../modules/panic_recovery.js';
import { holdResumes, resumePending, taskControlBusy } from '../modules/task_control_menu.js';

const turn = () => new Promise(resolve => setTimeout(resolve, 0));
const settle = async () => { for (let i = 0; i < 30; i += 1) await turn(); };
const all = (node, predicate) => [...(predicate(node) ? [node] : []), ...node.children.flatMap(child => all(child, predicate))];
const one = (node, predicate) => all(node, predicate)[0] || null;
const hasData = key => node => Object.hasOwn(node.dataset || {}, key);
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const refusal = (status, body) => Object.assign(new Error(body?.error || `HTTP ${status}`), { status, body });

const ROWS = {
    digest: { task_id: 'task-digest', task_name: 'Weekly digest', panic_hold_id: 'hold-digest', project_id: 'proj-notes', chat_id: 21 },
    sweep: { task_id: 'task-sweep', task_name: 'Model sweep', panic_hold_id: 'hold-sweep', chat_id: 1 },
    audit: { task_id: 'task-audit', task_name: 'Disk audit', panic_hold_id: 'hold-audit', project_id: 'proj-ops', chat_id: 22 },
};
const census = (rows, complete = true, extra = {}) => ({ panic_recovery: rows, panic_recovery_complete: complete,
    projects: [{ id: 'proj-notes', name: 'Notes', chat_id: 21 }, { id: 'proj-ops', name: 'Ops', chat_id: 22 }], ...extra });
// The same body's owner-action card of a Panic-held work (main_decisions.owner_actions): its Resume, and its Stop.
const card = (id, patch = {}) => ({ task_id: id, activity_id: id, phase: 'budget_paused', can_resume: true,
    cancelable: true, attention_reason: 'resume', model_waits: {}, ...patch });
const GAP = 'Some saved work could not be read, so this list may be incomplete.';

// The message alone, over the DOM stub: `answers` answers each task's Resume (a value, an error to throw, or a function
// of its body and transport options). A real door (`realResume`) leaves the shared client and `fetch` to answer.
function rig(t, { answers = {}, lifetimeMs, realResume = false, notify, extra = {} } = {}) {
    const env = installDom();
    document.body = env.mount;  // the shared toasts land here
    t.after(() => restoreDom(env.prior));
    const feed = document.createElement('div');
    env.mount.append(feed);
    const state = { clock: 1000, posts: [], signals: [], refreshes: 0, onRefresh: null, changes: 0 };
    // The card's `Stop…` as main_waiting.js hands it over: a button, its work's card and its current authority.
    const stops = new Map();
    const panic = createPanicRecovery({
        insertMessageNode: node => { feed.append(node); return true; },
        stampNodeTimestamp,
        refresh: async () => { state.refreshes += 1; await state.onRefresh?.(); },
        onChange: () => { state.changes += 1; },
        ...(realResume ? {} : { resume: async (taskId, body, init) => {
            state.posts.push([taskId, body]);
            state.signals.push(init?.signal);
            const answer = answers[taskId];
            const value = typeof answer === 'function' ? await answer(body, init) : answer;
            if (value instanceof Error) throw value;
            return value ?? { ok: true, task_id: taskId };
        } }),
        stopControl: (taskId, { row, allowed }) => {
            const button = document.createElement('button');
            button.textContent = 'Stop…';
            button.dataset.stopFor = taskId;
            stops.set(taskId, { row, allowed });
            return button;
        },
        now: () => state.clock,
        ...(lifetimeMs ? { lifetimeMs } : {}),
        ...(notify ? { notify } : {}),
        ...extra,
    });
    t.after(() => panic.destroy());
    const message = () => one(feed, node => node.classList?.contains('panic-recovery'));
    const row = id => one(message(), node => node.classList?.contains('panic-recovery-row') && node.dataset.taskId === id);
    const box = id => one(row(id), node => node.tagName === 'INPUT');
    const toggle = (id, on = !box(id).checked) => {
        const input = box(id);
        input.checked = on;
        for (const fn of input.listeners.get('change') || []) fn({ target: input });
    };
    const part = key => one(message(), hasData(key));
    const outcome = id => one(row(id), node => node.classList?.contains('panic-recovery-outcome'));
    const meta = id => one(row(id), node => node.classList?.contains('panic-recovery-meta'));
    const shown = () => all(message(), node => node.classList?.contains('panic-recovery-row'))
        .filter(node => !node.hidden).map(node => node.dataset.taskId);
    const stop = id => one(row(id), hasData('stopFor'));
    const count = () => one(message(), node => node.classList?.contains('panic-recovery-count')).textContent;
    return { panic, feed, state, stops, message, row, box, toggle, part, outcome, meta, shown, stop, count };
}

test('a census row needs its task and the stop it names; anything less is not a row', () => {
    assert.equal(panicRow({ task_id: 't', task_name: 'T' }), null);
    assert.equal(panicRow({ panic_hold_id: 'h' }), null);
    assert.deepEqual(panicRow({ task_id: ' t ', panic_hold_id: 'h', chat_id: 3, extra: 1 }),
        { task_id: 't', task_name: '', panic_hold_id: 'h', project_id: '', chat_id: 3, root_task_id: '' });
});

test('opening Main names every stopped work and sends nothing; a reconnect or a repaint sends nothing either', async (t) => {
    const r = rig(t);
    r.panic.snapshot(null);
    assert.equal(r.message(), null, 'no census, no message');
    r.panic.snapshot(census([]), 1);
    assert.equal(r.message(), null, 'nothing stopped: no message');
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep, ROWS.audit]), 2);
    const message = r.message();
    assert.ok(message, 'one System message');
    assert.equal(one(message, node => node.classList?.contains('sender')).textContent, 'System');
    assert.equal(one(message, node => node.tagName === 'STRONG').textContent, 'Stopped by Panic');
    assert.deepEqual(r.shown(), ['task-digest', 'task-sweep', 'task-audit']);
    assert.equal(one(r.row('task-digest'), node => node.classList?.contains('panic-recovery-name')).textContent, 'Weekly digest');
    assert.equal(r.meta('task-digest').textContent, 'Notes', 'its Project');
    assert.equal(r.meta('task-sweep').textContent, 'Main', 'Main\'s own work');
    assert.equal(r.message().querySelector('.owner-action-reason').textContent, 'Choose which to resume. The others stay stopped.');
    assert.equal(r.box('task-digest').checked, false, 'nothing is chosen for the owner');
    assert.equal(r.part('panicResume').disabled, true);
    assert.equal(r.part('panicResume').textContent, 'Resume selected');
    assert.equal(r.part('panicAll').hidden, false, 'a complete list offers all');
    assert.equal(r.part('panicAll').textContent, 'Choose all 3');
    assert.equal(one(message, node => node.classList?.contains('panic-recovery-count')).textContent, '0 of 3 chosen');
    r.panic.setConnected(false);
    r.panic.setConnected(true);
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep, ROWS.audit]), 3);
    await settle();
    assert.deepEqual(r.state.posts, [], 'no Resume without the owner\'s press');
    assert.equal(r.feed.children.length, 1, 'repaints keep one message');
});

test('a chosen subset sends exactly those rows, each with only the stop it was chosen under', async (t) => {
    const r = rig(t);
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep, ROWS.audit]), 10);
    r.toggle('task-sweep', true);
    assert.equal(r.part('panicResume').textContent, 'Resume 1 selected');
    r.state.onRefresh = () => r.panic.snapshot(census([ROWS.digest, ROWS.audit]), r.state.clock + 1);
    r.state.clock = 20;
    r.part('panicResume').click();
    await settle();
    assert.deepEqual(r.state.posts, [['task-sweep', { expected_panic_hold_id: 'hold-sweep' }]]);
    assert.equal(r.outcome('task-sweep').textContent, 'Resuming: the task returns to the queue.');
    assert.equal(r.outcome('task-sweep').dataset.tone, 'ok');
    assert.equal(r.box('task-sweep').hidden, true, 'what came back keeps no box');
    assert.equal(r.box('task-digest').disabled, false, 'the others stay choosable after the reconciling read');
    assert.equal(r.box('task-digest').checked, false);
    assert.equal(r.part('panicSummary').textContent, '1 accepted · 2 still stopped');
    assert.equal(r.state.refreshes, 1, 'one reconciling state read');
});

test('Choose all sends one request per row in order; while it is out nothing claims success', async (t) => {
    const gates = { 'task-digest': deferred(), 'task-sweep': deferred(), 'task-audit': deferred() };
    const r = rig(t, { answers: Object.fromEntries(Object.entries(gates).map(([id, gate]) => [id, () => gate.promise])) });
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep, ROWS.audit]), 10);
    r.part('panicAll').click();
    assert.equal(r.part('panicResume').textContent, 'Resume 3 selected');
    r.part('panicResume').click();
    await settle();
    assert.deepEqual(r.state.posts.map(([id]) => id), ['task-digest'], 'one at a time');
    assert.equal(r.part('panicResume').textContent, 'Sending…');
    assert.equal(r.part('panicResume').disabled, true);
    assert.equal(r.part('panicResume').getAttribute('aria-busy'), 'true');
    for (const id of ['task-digest', 'task-sweep', 'task-audit']) {
        assert.equal(r.outcome(id).textContent, 'Sending…');
        assert.equal(r.outcome(id).dataset.tone, undefined, 'pending has no tone, never ok');
        assert.equal(resumePending(id), true, 'every chosen task is pending to the rooms\' own Resume doors');
    }
    assert.equal(r.part('panicSummary').hidden, true, 'no count before an answer');
    const lead = () => r.message().querySelector('.owner-action-reason');
    assert.equal(lead().hidden, true, 'while answers are out it neither asks to choose nor says nothing waits');
    gates['task-digest'].resolve({ ok: true, task_id: 'task-digest' });
    await settle();
    assert.equal(r.outcome('task-digest').textContent, 'Resuming: the task returns to the queue.');
    assert.equal(r.outcome('task-sweep').textContent, 'Sending…');
    gates['task-sweep'].resolve({ ok: true, task_id: 'task-sweep', held: true, blockers: [{ kind: 'running_member', task_id: 'kid' }] });
    gates['task-audit'].resolve({ ok: true, task_id: 'task-audit', tree_only: true });
    await settle();
    assert.deepEqual(r.state.posts.map(([id, body]) => [id, Object.keys(body)]), [
        ['task-digest', ['expected_panic_hold_id']], ['task-sweep', ['expected_panic_hold_id']], ['task-audit', ['expected_panic_hold_id']]]);
    assert.equal(r.outcome('task-sweep').textContent,
        'Resume accepted; it starts by itself when the earlier attempt\'s work finishes: task kid is still running.');
    assert.equal(r.outcome('task-audit').textContent, 'Resumed. Ouroboros or you choose which subagents continue.');
    for (const id of ['task-digest', 'task-sweep', 'task-audit']) assert.equal(resumePending(id), false, 'released after the read');
    assert.equal(r.part('panicSummary').textContent, '3 accepted', 'accepted is a receipt: each line says what it means');
    assert.equal(r.part('panicResume').hidden, true, 'nothing left to resume');
    assert.equal(lead().hidden, true, 'no census read after the answers: nothing is claimed');
    r.panic.snapshot(census([]), r.state.clock + 1);
    assert.equal(lead().textContent, 'No saved work stopped by Panic waits for Resume now.', 'a complete read after them');
});

test('a press sends in the latest census order, not the order its rows were first seen', async (t) => {
    const r = rig(t);
    const kid = { task_id: 'task-kid', task_name: 'Collect figures', panic_hold_id: 'hold-kid', root_task_id: 'task-root' };
    const root = { task_id: 'task-root', task_name: 'Quarterly report', panic_hold_id: 'hold-root', root_task_id: 'task-root' };
    r.panic.snapshot(census([kid], false), 10);  // an incomplete read names the descendant alone
    r.panic.snapshot(census([root, kid]), 11);   // the complete read lists its root before it
    r.part('panicAll').click();
    r.part('panicResume').click();
    await settle();
    assert.deepEqual(r.state.posts.map(([id]) => id), ['task-root', 'task-kid']);
});

test('Choose all and every whole-list claim need today\'s complete census of a ready core', async (t) => {
    const r = rig(t);
    const shown = key => !r.part(key).hidden && Boolean(r.part(key).textContent);
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 10);
    assert.equal(shown('panicAll'), true);
    r.panic.snapshot(null, 11);  // a failed read
    assert.equal(shown('panicAll'), false, 'no remembered completeness');
    assert.equal(shown('panicGap'), false, 'a missing read claims nothing either way');
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 12);
    assert.equal(shown('panicAll'), true);
    r.panic.setConnected(false);
    assert.equal(shown('panicAll'), false, 'a closed socket withdraws it');
    r.panic.setConnected(true);
    assert.equal(shown('panicAll'), false, 'reopening alone restores nothing');
    r.panic.snapshot({ ...census([ROWS.digest, ROWS.sweep]), supervisor_ready: false }, 13);
    assert.equal(shown('panicAll'), false, 'a core not ready proves no whole list');
    assert.equal(shown('panicGap'), true);
    r.toggle('task-digest', true);  // a named row of a partial list may still be chosen on purpose
    r.state.onRefresh = () => r.panic.snapshot(null, 30);
    r.state.clock = 20;
    r.part('panicResume').click();
    await settle();
    assert.deepEqual(r.state.posts, [['task-digest', { expected_panic_hold_id: 'hold-digest' }]]);
    assert.equal(r.part('panicSummary').textContent, '1 accepted', 'the unread list is not called still stopped');
    assert.equal(r.message().querySelector('.owner-action-reason').textContent,
        'Choose which to resume. The others stay stopped.', 'never that nothing waits after a failed read');
});

test('an answer belongs to the stop it was sent for: under a newer stop it reads as the earlier stop\'s, and the newer one is not chosen', async (t) => {
    const gate = deferred();
    const r = rig(t, { answers: { 'task-digest': () => gate.promise } });
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 10);
    r.toggle('task-digest', true);
    r.part('panicResume').click();
    await settle();
    // While its Resume is out, a read names the same work under a newer stop.
    r.panic.snapshot(census([{ ...ROWS.digest, panic_hold_id: 'hold-digest-2' }, ROWS.sweep]), 20);
    const said = () => [r.outcome('task-digest').textContent, r.outcome('task-digest').dataset.tone];
    assert.deepEqual(said(), ['Earlier stop: Sending…', undefined], 'the press still out is the earlier stop\'s');
    assert.equal(r.box('task-digest').disabled, true, 'nothing is chosen while the press is out');
    r.state.clock = 30;
    gate.resolve({ ok: true, task_id: 'task-digest' });
    await settle();
    assert.deepEqual(r.state.posts, [['task-digest', { expected_panic_hold_id: 'hold-digest' }]]);
    assert.deepEqual(said(), ['Earlier stop: Resuming: the task returns to the queue.', undefined],
        'the press\'s result stays in view as the earlier stop\'s: a record, without a tone');
    assert.equal(r.box('task-digest').hidden, false);
    assert.deepEqual([r.box('task-digest').checked, r.box('task-digest').disabled], [false, false],
        'the newer stop is never chosen for the owner, and can be chosen');
    assert.equal(r.part('panicResume').disabled, true);
    assert.equal(r.part('panicSummary').textContent, '2 still stopped', 'no acceptance is counted for the newer stop');
    assert.ok(r.panic.target(), 'it still waits on the owner');
    // Chosen again on purpose, it is sent with the newer stop alone.
    r.toggle('task-digest', true);
    r.part('panicResume').click();
    await settle();
    assert.deepEqual(r.state.posts.at(-1), ['task-digest', { expected_panic_hold_id: 'hold-digest-2' }]);
    assert.deepEqual(said(), ['Resuming: the task returns to the queue.', 'ok'], 'now the answer about its stop');
});

test('a stop that changed after its press reads as the earlier stop\'s, whatever came back for it', async (t) => {
    const r = rig(t, { lifetimeMs: 20, answers: {
        'task-digest': refusal(409, { error: 'selection_owner_held', message: 'Resume refused: it is paused separately.' }),
        'task-sweep': () => new Promise(() => {}),
        'task-audit': refusal(409, { error: 'panic_selection_changed' }),
    } });
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep, ROWS.audit]), 10);
    r.part('panicAll').click();
    r.state.clock = 100;
    // The reconciling read names all three under newer stops.
    const newer = [ROWS.digest, ROWS.sweep, ROWS.audit].map(row => ({ ...row, panic_hold_id: `${row.panic_hold_id}-2` }));
    r.state.onRefresh = () => r.panic.snapshot(census(newer), 200);
    r.part('panicResume').click();
    for (let i = 0; i < 20 && r.part('panicResume').disabled; i += 1) await new Promise(resolve => setTimeout(resolve, 10));
    await settle();
    const said = id => [r.outcome(id).textContent, r.outcome(id).dataset.tone];
    assert.deepEqual(said('task-digest'), ['Earlier stop: Resume refused: it is paused separately.', undefined]);
    assert.deepEqual(said('task-sweep'), ['Earlier stop: No answer came back; it may already be back in the queue.', undefined]);
    assert.deepEqual(said('task-audit'), ['Not resumed: its stop changed after you chose it.', 'warn'], 'a changed stop says so itself');
    for (const id of ['task-digest', 'task-sweep', 'task-audit']) {
        assert.deepEqual([r.box(id).hidden, r.box(id).checked, r.box(id).disabled], [false, false, false], id);
    }
    assert.equal(r.part('panicSummary').textContent, '3 still stopped');
    assert.equal(r.state.posts.length, 3, 'none is sent again by itself');
});

test('an incomplete list never offers all: it says so, and only the rows a read names can be chosen', async (t) => {
    const r = rig(t);
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep], false), 10);
    assert.equal(r.part('panicAll').hidden, true, 'no Choose all');
    assert.equal(r.part('panicGap').textContent, 'Some saved work could not be read, so this list may be incomplete.');
    assert.equal(r.part('panicGap').dataset.tone, 'warn');
    assert.equal(r.box('task-digest').disabled, false, 'a named row can still be chosen on its own');
    r.toggle('task-digest', true);
    r.panic.snapshot(census([ROWS.sweep], false), 11);
    assert.equal(r.box('task-digest').disabled, true, 'a row the latest read did not name is withheld');
    assert.equal(r.meta('task-digest').textContent, 'Notes · not confirmed by the latest read');
    assert.equal(r.part('panicResume').disabled, true, 'its choice cannot be sent');
    // A body without the completeness fact is not complete either.
    r.panic.snapshot({ panic_recovery: [ROWS.digest, ROWS.sweep] }, 12);
    assert.equal(r.part('panicAll').hidden, true);
    assert.equal(r.box('task-digest').checked, true, 'named again, under the same stop, its choice stands');
    // A row nothing can act on makes the set incomplete, whatever the flag says.
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep, { task_id: 'task-x' }]), 13);
    assert.equal(r.part('panicAll').hidden, true);
    assert.equal(r.row('task-x'), null);
    await settle();
    assert.deepEqual(r.state.posts, []);
});

test('each row says its own answer; none is sent again by itself', async (t) => {
    const hang = new Promise(() => {});
    const rows = [ROWS.digest, ROWS.sweep, ROWS.audit,
        { task_id: 'task-gone', task_name: 'Old report', panic_hold_id: 'hold-gone' },
        { task_id: 'task-lost', task_name: 'Lost answer', panic_hold_id: 'hold-lost' },
        { task_id: 'task-slow', task_name: 'Slow answer', panic_hold_id: 'hold-slow' }];
    const r = rig(t, { lifetimeMs: 20, answers: {
        'task-digest': { ok: true, task_id: 'task-digest' },
        'task-sweep': refusal(409, { error: 'panic_selection_changed', task_id: 'task-sweep' }),
        'task-audit': refusal(409, { error: 'billing_authority_unavailable', task_id: 'task-audit',
            message: 'This task\'s billing conditions still could not be read.' }),
        'task-gone': refusal(404, { error: 'task_not_pending', task_id: 'task-gone' }),
        'task-lost': new TypeError('Failed to fetch'),
        'task-slow': () => hang,
    } });
    r.panic.snapshot(census(rows), 10);
    r.part('panicAll').click();
    r.state.clock = 100;
    // The reconciling read was asked for before the answers came back: it names the lost one under the same stop.
    r.state.onRefresh = () => r.panic.snapshot(census(rows.filter(row => row.task_id !== 'task-digest')), 50);
    r.part('panicResume').click();
    for (let i = 0; i < 20 && r.part('panicResume').disabled; i += 1) await new Promise(resolve => setTimeout(resolve, 10));
    await settle();
    assert.deepEqual(r.state.posts.map(([id]) => id), rows.map(row => row.task_id), 'each chosen row once');
    const said = id => [r.outcome(id).textContent, r.outcome(id).dataset.tone];
    assert.deepEqual(said('task-digest'), ['Resuming: the task returns to the queue.', 'ok']);
    assert.deepEqual(said('task-sweep'), ['Not resumed: its stop changed after you chose it.', 'warn']);
    assert.deepEqual(said('task-audit'), ['This task\'s billing conditions still could not be read.', 'error']);
    assert.deepEqual(said('task-gone'), ['Not resumed: it no longer waits for Resume.', 'neutral']);
    assert.deepEqual(said('task-lost'), ['No answer came back; it may already be back in the queue.', 'neutral']);
    assert.deepEqual(said('task-slow'), ['No answer came back; it may already be back in the queue.', 'neutral']);
    assert.equal(r.box('task-lost').disabled, true, 'a read older than its answer cannot make it choosable again');
    assert.equal(r.part('panicSummary').textContent, '1 accepted · 3 still stopped · 2 without an answer',
        'every row once: a refused or changed row is still stopped, its own line says why');
    // A read asked for after the answers still names it under the same stop: the owner may choose it again.
    r.panic.snapshot(census(rows.filter(row => row.task_id !== 'task-digest')), 200);
    assert.equal(r.box('task-lost').disabled, false);
    assert.equal(r.outcome('task-lost').textContent, 'No answer came back; it may already be back in the queue.', 'its answer stays');
    // A partial read that does not name the refused row: it is no longer counted as still stopped.
    r.panic.snapshot(census(rows.filter(row => !['task-digest', 'task-audit'].includes(row.task_id)), false), 300);
    assert.equal(r.meta('task-audit').textContent, 'Ops · not confirmed by the latest read');
    assert.equal(r.part('panicSummary').textContent, '1 accepted · 2 still stopped · 2 without an answer');
    await settle();
    assert.equal(r.state.posts.length, rows.length, 'nothing re-sent');
});

test('a complete read that drops answered work keeps its answer as the record, and drops unanswered work', async (t) => {
    const r = rig(t, { answers: { 'task-sweep': new TypeError('Failed to fetch'),
        'task-audit': refusal(409, { error: 'update_in_progress', message: 'An update is in progress; press Resume after the restart.' }) } });
    const idle = { task_id: 'task-idle', task_name: 'Not chosen', panic_hold_id: 'hold-idle' };
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep, ROWS.audit, idle]), 10);
    r.toggle('task-digest', true);
    r.toggle('task-sweep', true);
    r.toggle('task-audit', true);
    r.state.clock = 20;
    r.state.onRefresh = () => r.panic.snapshot(census([]), 30);
    r.part('panicResume').click();
    await settle();
    assert.equal(r.outcome('task-sweep').textContent, 'No answer came back; the latest list no longer holds it.');
    assert.equal(r.row('task-idle'), null, 'work nobody answered for leaves with the list');
    assert.equal(r.part('panicSummary').textContent, '1 accepted · 1 without an answer · 1 not resumed');
    assert.equal(r.meta('task-audit').textContent, 'Ops · no longer waits for Resume');
    for (const id of ['task-digest', 'task-sweep', 'task-audit']) assert.equal(r.box(id).hidden, true);
    assert.equal(r.message().querySelector('.owner-action-reason').textContent,
        'No saved work stopped by Panic waits for Resume now.');
    assert.equal(r.part('panicResume').hidden, true);
    assert.equal(r.part('panicResume').parentNode.parentNode.querySelector('.panic-recovery-picks').hidden, true);
    assert.equal(r.panic.target(), null, 'nothing waits on the owner any more');
});

test('a choice belongs to the stop it was made under: a new stop of the same work is chosen again', (t) => {
    const r = rig(t);
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 10);
    r.toggle('task-digest', true);
    assert.equal(r.part('panicResume').textContent, 'Resume 1 selected');
    r.panic.snapshot(census([{ ...ROWS.digest, panic_hold_id: 'hold-digest-2' }, ROWS.sweep]), 11);
    assert.equal(r.box('task-digest').checked, false);
    assert.equal(r.part('panicResume').disabled, true);
    assert.equal(r.part('panicResume').textContent, 'Resume selected');
});

test('a closed socket withdraws every box and ignores late bodies; only the reopen\'s census restores them', async (t) => {
    const r = rig(t);
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 10);
    r.toggle('task-digest', true);
    r.panic.setConnected(false);
    assert.equal(r.box('task-digest').disabled, true);
    assert.equal(r.box('task-digest').hidden, false, 'a box withdrawn is still there: nothing moves');
    assert.equal(r.part('panicResume').disabled, true);
    assert.equal(r.meta('task-digest').textContent, 'Notes', 'the outage says itself in the room, not on each row');
    assert.ok(r.panic.target(), 'the owner\'s move is still counted');
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 11);
    assert.equal(r.box('task-digest').disabled, true, 'a body read while the socket was down restores nothing');
    r.panic.setConnected(true);
    assert.equal(r.box('task-digest').disabled, true, 'reopening alone restores nothing');
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 12);
    assert.equal(r.box('task-digest').disabled, false);
    assert.equal(r.box('task-digest').checked, true, 'the same stop keeps its choice');
    r.part('panicResume').click();
    await settle();
    assert.deepEqual(r.state.posts, [['task-digest', { expected_panic_hold_id: 'hold-digest' }]]);
});

test('a long list shows its first rows and one Show all; Choose all counts every row', (t) => {
    const r = rig(t);
    const rows = Array.from({ length: SHOWN_ROWS + 3 }, (_, index) => ({
        task_id: `task-${index}`, task_name: index === 4 ? 'Twin' : index === 5 ? 'Twin' : `Work ${index}`, panic_hold_id: `hold-${index}` }));
    r.panic.snapshot(census(rows), 10);
    assert.equal(r.shown().length, SHOWN_ROWS);
    assert.equal(r.part('panicMore').hidden, false);
    assert.equal(r.part('panicMore').textContent, `Show all (${rows.length})`);
    assert.equal(r.meta('task-4').textContent, 'task-4', 'a twin name says its short id');
    r.part('panicMore').click();
    assert.equal(r.shown().length, rows.length);
    assert.equal(r.part('panicMore').hidden, true);
    assert.equal(document.activeElement, r.box(`task-${SHOWN_ROWS}`), 'focus goes to the first row it revealed');
    r.part('panicAll').click();
    assert.equal(r.part('panicResume').textContent, `Resume ${rows.length} selected`);
    r.part('panicNone').click();
    assert.equal(r.part('panicResume').textContent, 'Resume selected');
});

test('Main\'s pill counts the Panic choice as one action and lands on its first box, never on Resume', async (t) => {
    const env = installDom();
    document.body = env.mount;
    t.after(() => restoreDom(env.prior));
    const box = (top, height) => ({ top, bottom: top + height, height, left: 0, right: 600, width: 600 });
    const page = document.createElement('div');
    const header = document.createElement('div');
    header.className = 'chat-page-header';
    const feed = document.createElement('div');
    const toolbar = document.createElement('div');
    toolbar.className = 'chat-toolbar-row';
    page.append(header, feed, toolbar);
    env.mount.append(page);
    header.getBoundingClientRect = () => box(0, 60);
    feed.getBoundingClientRect = () => box(0, 600);
    toolbar.getBoundingClientRect = () => box(560, 40);
    Object.defineProperty(feed, 'scrollHeight', { configurable: true, get: () => feed.children.length * 1000, set() {} });
    const insert = node => {
        feed.append(node);
        node.getBoundingClientRect = () => box(feed.children.indexOf(node) * 1000 - feed.scrollTop, 200);
        return true;
    };
    const posts = [];
    globalThis.fetch = async (url, init) => { posts.push([url, init?.method]); return { ok: true, json: async () => ({}) }; };
    const waiting = createMainWaiting({ page, messagesDiv: feed, toolbar, insertMessageNode: insert, stampNodeTimestamp,
        decision: () => ({ syncCurrentQuestions() {}, setConnected() {}, observedQuestion: () => null }),
        reading: () => ({ claim() {}, stick: true }) });
    t.after(() => waiting.destroy());
    const pill = toolbar.children.at(-1);
    // An owner-required row the Panic list does not name keeps its own Resume beside the choice; a work the
    // message itself offers carries no second card of its own (main_waiting.js), so the pill counts the two
    // cards it sees, never the same work twice.
    waiting.snapshot({ current_questions: [], current_questions_complete: true,
        main_attention: [{ task_id: 'task-restart', activity_id: 'task-restart', task_name: 'Restart the model server', phase: 'budget_paused',
            pause_cause: 'restart', can_resume: true, cancelable: false, attention_reason: 'restart', model_waits: {} },
            { ...ROWS.sweep, activity_id: 'task-sweep', phase: 'budget_paused', pause_cause: 'panic', can_resume: true,
                cancelable: false, attention_reason: 'resume', model_waits: {} }],
        main_attention_complete: true, ...census([ROWS.digest, ROWS.sweep]) });
    feed.scrollTop = 3000;  // both messages are above the reading area
    waiting.refresh();
    assert.equal(pill.hidden, false);
    assert.equal(pill.querySelector('.chat-question-nav-count').textContent, '2');
    assert.equal(pill.querySelector('.chat-question-nav-word').textContent, ' actions', 'the choice is one action beside the row');
    pill.click();
    await settle();
    pill.click();
    await settle();
    const panicMessage = one(feed, node => node.classList?.contains('panic-recovery'));
    const first = one(panicMessage, node => node.tagName === 'INPUT');
    assert.equal(document.activeElement, first, 'the landing focuses the first box: choosing is the next step');
    assert.deepEqual(posts, [], 'landing sends nothing');
});

// The bare line leaves the feed. A host not ready yet is said by the room bar (Starting…) and the setup wizard covers
// everything; a ready host's own `false` with nothing read is told once per episode by the page's notice, a reload of the
// same tab included, until a whole read ends it; a person's Retry that still does not read whole is answered.
function noticeTrack() {
    const told = [];
    const notify = (text, options) => {
        const notice = { text, options, disposed: false, isConnected: true, dispose() { this.disposed = true; this.isConnected = false; } };
        told.push(notice);
        return notice;
    };
    const store = new Map();
    const storage = { getItem: k => store.get(k) ?? null, setItem: (k, v) => store.set(k, String(v)), removeItem: k => store.delete(k) };
    return { told, notify, storage };
}

test('a list not whole with no row read is no message: a ready host\'s word is one notice per episode, a starting one none', async (t) => {
    const n = noticeTrack();
    let covered = false;
    const r = rig(t, { notify: n.notify, extra: { storage: n.storage, isVisible: () => !covered } });
    const told = n.told;
    const READY = { supervisor_ready: true, boot: { tasks_ready: true, since: '2026-11-27T10:00:00Z' } };
    for (const starting of [{ supervisor_ready: false }, { supervisor_ready: true, boot: { tasks_ready: false } }, {}]) {
        r.panic.snapshot(census([], false, starting), 1);
        assert.deepEqual([r.message(), r.feed.children.length, told.length], [null, 0, 0],
            `a host not ready, or one that does not say, is told by nothing here (${JSON.stringify(starting)})`);
    }
    covered = true;  // the setup wizard is up: nothing about rooms or recovery is announced over it
    r.panic.snapshot(census([], false, READY), 2);
    assert.equal(told.length, 0, 'not over the setup wizard');
    covered = false;
    r.panic.snapshot(census([], false, READY), 2);
    assert.deepEqual([r.message(), r.feed.children.length], [null, 0], 'never a System message or an empty card');
    assert.deepEqual(told.map(x => x.text), ['Saved work could not be checked for anything waiting to resume.'],
        'told once the wizard is gone');
    assert.equal(r.panic.target(), null, 'the pill counts nothing');
    r.panic.snapshot(census([], false, READY), 3);
    r.panic.snapshot(null, 4);
    r.panic.setConnected(false);
    r.panic.setConnected(true);
    r.panic.snapshot(census([], false, READY), 5);
    assert.equal(told.length, 1, 'once per episode: a re-read, a failed read or a reconnect adds nothing');
    // A person's Retry that still reads not whole is answered; it resumes nothing.
    r.state.onRefresh = () => r.panic.snapshot(census([], false, READY), 6);
    await told[0].options.action.run();
    assert.deepEqual([r.state.refreshes, told.length], [1, 2], 'its Retry reads again and says it still could not');
    r.state.onRefresh = () => { throw new Error('offline'); };
    await told[1].options.action.run();
    assert.equal(told.length, 3, 'a Retry whose read fails is answered too');
    r.panic.snapshot(census([], true, READY), 7);
    assert.deepEqual([told[2].disposed, r.message()], [true, null], 'a whole read ends the episode and its notice');
    r.panic.snapshot(census([], false, READY), 8);
    assert.equal(told.length, 4, 'a new episode is told again');
    // The working case: the same partial word with a row read is the message with its amber gap, and no notice.
    r.panic.snapshot(census([ROWS.digest], false, READY), 9);
    assert.equal(told[3].disposed, true, 'the message now says it');
    assert.deepEqual([r.part('panicGap').textContent, r.part('panicGap').dataset.tone], [GAP, 'warn']);
    const heading = one(r.message(), node => node.classList?.contains('model-wait-heading'));
    assert.deepEqual([heading.hidden, heading.firstChild?.textContent], [false, 'Stopped by Panic']);
    await settle();
    assert.deepEqual(r.state.posts, []);
});

test('the Panic notice is Main\'s: withdrawn when Main is left or covered, told again on return only after a fresh read', async (t) => {
    const n = noticeTrack();
    let shown = true;
    const r = rig(t, { notify: n.notify, extra: { storage: n.storage, isVisible: () => shown } });
    const READY = { supervisor_ready: true, boot: { tasks_ready: true, since: 'run-a' } };
    r.panic.snapshot(census([], false, READY), 1);
    assert.equal(n.told.length, 1);
    // Retry pressed with Main shown reads the state.
    r.state.onRefresh = () => r.panic.snapshot(census([], false, READY), 2);
    await n.told[0].options.action.run();
    assert.deepEqual([r.state.refreshes, n.told.length], [1, 2]);
    // Main left (Settings, a Project on a phone) or covered: the notice leaves with it.
    shown = false;
    r.panic.notice.hide();
    assert.equal(n.told[1].disposed, true, 'withdrawn');
    assert.equal(r.panic.notice.owed, false, 'not told again before a fresh read on return');
    shown = true;
    r.panic.snapshot(census([], false, READY), 3);
    assert.equal(n.told.length, 3, 'back and still not whole: told again');
    // Closed by the person, it is not told again in this episode.
    n.told[2].isConnected = false;
    r.panic.notice.hide();
    r.panic.snapshot(census([], false, READY), 4);
    assert.equal(n.told.length, 3, 'closed by the person: not again');
    // A whole read while away means no notice on return.
    const m = noticeTrack();
    let seen = true;
    const s2 = rig(t, { notify: m.notify, extra: { storage: m.storage, isVisible: () => seen } });
    s2.panic.snapshot(census([], false, { ...READY, boot: { tasks_ready: true, since: 'run-b' } }), 1);
    seen = false;
    s2.panic.notice.hide();
    s2.panic.snapshot(census([], true, READY), 2);
    seen = true;
    s2.panic.snapshot(census([], true, READY), 3);
    assert.equal(m.told.length, 1, 'the list read whole meanwhile: nothing on return');
    // A hidden Main owes the first telling until it is shown.
    const h = noticeTrack();
    let visible = false;
    const s3 = rig(t, { notify: h.notify, extra: { storage: h.storage, isVisible: () => visible } });
    s3.panic.snapshot(census([], false, { ...READY, boot: { tasks_ready: true, since: 'run-c' } }), 1);
    assert.deepEqual([h.told.length, s3.panic.notice.owed], [0, true]);
    visible = true;
    s3.panic.notice.announce();
    assert.equal(h.told.length, 1, 'told when Main is shown');
});

test('a reload of the same tab is not a new episode; a host restart inside the tab is', (t) => {
    const n = noticeTrack();
    const READY = (since) => ({ supervisor_ready: true, boot: { tasks_ready: true, since } });
    const first = rig(t, { notify: n.notify, extra: { storage: n.storage, isVisible: () => true } });
    first.panic.snapshot(census([], false, READY('run-1')), 1);
    assert.equal(n.told.length, 1);
    first.panic.destroy();
    const reloaded = rig(t, { notify: n.notify, extra: { storage: n.storage, isVisible: () => true } });
    reloaded.panic.snapshot(census([], false, READY('run-1')), 1);
    assert.equal(n.told.length, 1, 'the same host run in the same tab: already told');
    reloaded.panic.snapshot(census([], false, READY('run-2')), 2);
    assert.equal(n.told.length, 2, 'the host restarted inside this tab: a new run is a new episode, told at once');
    n.told[1].dispose();
    reloaded.panic.notice.hide();
    reloaded.panic.snapshot(census([], false, READY('run-2')), 3);
    assert.equal(n.told.length, 2, 'closed: not again on this run');
    reloaded.panic.destroy();
    const again = rig(t, { notify: n.notify, extra: { storage: n.storage, isVisible: () => true } });
    again.panic.snapshot(census([], false, READY('run-2')), 1);
    assert.equal(n.told.length, 2, 'a reload keeps the decision taken on this run');
    again.panic.snapshot(census([], true, READY('run-2')), 2);
    assert.equal(n.storage.getItem('ouro.panic_list_unread_told'), null, 'a whole read clears the tab\'s record');
    again.panic.destroy();
});

test('an older host that names no completeness is taken as neither whole, incomplete nor stopped', async (t) => {
    const r = rig(t);
    const old = rows => ({ panic_recovery: rows, projects: census([]).projects });
    r.panic.snapshot({ projects: census([]).projects }, 1);
    assert.equal(r.message(), null, 'a body without the list names nothing');
    r.panic.snapshot(old([]), 2);
    assert.equal(r.message(), null, 'an empty list it does not call incomplete is no fact to show');
    r.panic.snapshot(census([{ task_id: 'task-x', task_name: 'No stop named' }]), 3);
    assert.equal(r.message(), null, 'nor is a row without its stop: absence proves nothing');
    // Rows it names are its works, each its own choice: no `Choose all`, no gap claimed, never that nothing waits.
    r.panic.snapshot(old([ROWS.digest, ROWS.sweep]), 4);
    assert.deepEqual([r.part('panicAll').hidden, r.part('panicGap').textContent], [true, '']);
    r.toggle('task-sweep', true);
    r.state.onRefresh = () => r.panic.snapshot(old([ROWS.digest]), r.state.clock + 1);
    r.state.clock = 10;
    r.part('panicResume').click();
    await settle();
    r.toggle('task-digest', true);
    r.state.onRefresh = () => r.panic.snapshot(old([]), r.state.clock + 1);
    r.state.clock = 20;
    r.part('panicResume').click();
    await settle();
    assert.deepEqual(r.state.posts.map(([id]) => id), ['task-sweep', 'task-digest']);
    assert.equal(r.part('panicGap').textContent, '');
    assert.equal(r.message().querySelector('.owner-action-reason').hidden, true, 'never that nothing waits');
});

test('an incomplete list keeps saying so after the answer to the last row it knew', async (t) => {
    const r = rig(t);
    r.panic.snapshot(census([ROWS.digest], false), 1);
    assert.deepEqual([r.part('panicGap').textContent, r.part('panicGap').dataset.tone], [GAP, 'warn']);
    r.toggle('task-digest', true);
    r.state.onRefresh = () => r.panic.snapshot(census([], false), r.state.clock + 1);
    r.state.clock = 10;
    r.part('panicResume').click();
    await settle();
    assert.equal(r.outcome('task-digest').textContent, 'Resuming: the task returns to the queue.');
    assert.equal(r.part('panicGap').textContent, GAP, 'after the answer to the last row it knew');
    assert.equal(r.message().querySelector('.owner-action-reason').hidden, true, 'and never that nothing waits');
    r.panic.snapshot(null, 30);
    assert.equal(r.part('panicGap').textContent, GAP, 'a missing read proves no whole list');
    r.panic.snapshot(census([]), 40);
    assert.equal(r.part('panicGap').textContent, '', 'a whole list ends it');
    assert.equal(r.message().querySelector('.owner-action-reason').textContent, 'No saved work stopped by Panic waits for Resume now.');
    assert.deepEqual(r.state.posts, [['task-digest', { expected_panic_hold_id: 'hold-digest' }]]);
});

test('Choose all N names exactly the rows a read offers now; with fewer than two there is none', async (t) => {
    const gate = deferred();
    const r = rig(t, { answers: { 'task-sweep': () => gate.promise } });
    const rows = [ROWS.digest, ROWS.sweep, ROWS.audit, { task_id: 'task-tidy', task_name: 'Tidy', panic_hold_id: 'hold-tidy' }];
    r.panic.snapshot(census(rows), 10);
    assert.deepEqual([r.part('panicAll').textContent, r.count()], ['Choose all 4', '0 of 4 chosen']);
    // A room's own Resume of one of them is out: that row cannot be chosen here, and all names the other three.
    let release;
    const roomPress = holdResumes(['task-audit'], () => new Promise(resolve => { release = resolve; }));
    r.panic.snapshot(census(rows), 11);
    assert.deepEqual([r.part('panicAll').textContent, r.count()], ['Choose all 3', '0 of 3 chosen']);
    r.part('panicAll').click();
    assert.equal(r.part('panicResume').textContent, 'Resume 3 selected', 'it chose exactly the three it named');
    assert.equal(r.box('task-audit').checked, false);
    assert.equal(r.part('panicAll').disabled, true, 'nothing more to choose');
    r.part('panicNone').click();
    // While a press is out every box is held; all names what stays to choose, and is not offered.
    r.toggle('task-sweep', true);
    r.part('panicResume').click();
    await settle();
    assert.deepEqual([r.part('panicAll').textContent, r.part('panicAll').disabled, r.count()], ['Choose all 2', true, '0 of 2 chosen']);
    gate.resolve({ ok: true, task_id: 'task-sweep' });
    release();
    await roomPress;
    await settle();
    r.panic.snapshot(census(rows.filter(row => row.task_id !== 'task-sweep')), r.state.clock + 1);
    assert.deepEqual([r.part('panicAll').textContent, r.part('panicAll').disabled], ['Choose all 3', false]);
    // One row left to choose: no `Choose all 1`.
    r.panic.snapshot(census([ROWS.digest, ROWS.audit]), r.state.clock + 2);
    let free;
    const held = holdResumes(['task-audit'], () => new Promise(resolve => { free = resolve; }));
    r.panic.snapshot(census([ROWS.digest, ROWS.audit]), r.state.clock + 3);
    assert.deepEqual([r.part('panicAll').hidden, r.count()], [true, '0 of 1 chosen']);
    free();
    await held;
});

test('each work keeps the Stop… its card offered; it waits only for that work\'s own answer', async (t) => {
    const gates = { 'task-digest': deferred(), 'task-sweep': deferred() };
    const r = rig(t, { answers: { 'task-digest': () => gates['task-digest'].promise, 'task-sweep': () => gates['task-sweep'].promise } });
    const body = census([ROWS.digest, ROWS.sweep, ROWS.audit], true, { main_attention_complete: true,
        main_attention: [card('task-digest'), card('task-sweep'), card('task-audit', { attention_reason: 'budget' })] });
    assert.deepEqual([...panicCards(body).keys()], ['task-digest', 'task-sweep'], 'a spending-limit card stays its own');
    r.panic.snapshot(body, 10);
    assert.deepEqual([Boolean(r.stop('task-digest')), Boolean(r.stop('task-sweep')), r.stop('task-audit')], [true, true, null]);
    assert.equal(r.stop('task-digest').disabled, false);
    assert.equal(r.stop('task-digest').getAttribute('aria-describedby'), r.row('task-digest').children[0].children[1].id,
        'it names its work');
    assert.equal(r.stops.get('task-digest').row().phase, 'budget_paused', 'its menu reads the card it stands for');
    r.toggle('task-digest', true);
    r.toggle('task-sweep', true);
    r.part('panicResume').click();
    await settle();
    assert.deepEqual([r.stop('task-digest').disabled, r.stop('task-sweep').disabled], [true, true], 'each Resume is out');
    gates['task-digest'].reject(refusal(409, { error: 'selection_owner_held', message: 'Resume refused: it is paused separately.' }));
    await settle();
    assert.deepEqual([taskControlBusy('task-digest'), resumePending('task-digest')], [false, true],
        'its own request is answered; every Resume door still waits for the read');
    assert.equal(r.stop('task-digest').disabled, false, 'its Stop no longer waits for another row\'s slow answer');
    assert.equal(r.stops.get('task-digest').allowed(), true);
    assert.equal(r.stop('task-sweep').disabled, true);
    r.state.onRefresh = () => r.panic.snapshot(census([ROWS.digest, ROWS.audit], true, { main_attention_complete: true,
        main_attention: [card('task-digest'), card('task-audit', { attention_reason: 'budget' })] }), r.state.clock + 1);
    r.state.clock = 50;
    gates['task-sweep'].resolve({ ok: true, task_id: 'task-sweep' });
    await settle();
    assert.equal(r.stop('task-sweep').hidden, true, 'what came back keeps no Stop here: its room has its controls');
    assert.equal(r.stop('task-digest').disabled, false);
    // A read that cannot vouch for the card withdraws its Stop; a closed socket too.
    r.panic.snapshot(census([ROWS.digest, ROWS.audit], true, { main_attention_complete: false, main_attention: [] }), 60);
    assert.deepEqual([r.stop('task-digest').hidden, r.stop('task-digest').disabled], [false, true]);
    r.panic.snapshot(census([ROWS.digest, ROWS.audit], true, { main_attention_complete: true, main_attention: [card('task-digest')] }), 70);
    assert.equal(r.stop('task-digest').disabled, false);
    r.panic.setConnected(false);
    assert.equal(r.stop('task-digest').disabled, true);
});

test('a work whose Resume was accepted keeps its card\'s Stop while that card grants it, whatever another row awaits', async (t) => {
    const gate = deferred();
    const r = rig(t, { answers: { 'task-sweep': () => gate.promise } });  // the digest answers at once: accepted
    const body = (extra = {}) => census([ROWS.digest, ROWS.sweep], true, { main_attention_complete: true,
        main_attention: [card('task-digest'), card('task-sweep')], ...extra });
    r.panic.snapshot(body(), 10);
    r.part('panicAll').click();
    r.part('panicResume').click();
    await settle();
    assert.equal(r.outcome('task-digest').textContent, 'Resuming: the task returns to the queue.');
    assert.equal(r.box('task-digest').hidden, true, 'it came back: no box');
    assert.deepEqual([r.stop('task-digest').hidden, r.stop('task-digest').disabled], [false, false],
        'its card still grants it Stop, whatever its Resume answered');
    assert.deepEqual([taskControlBusy('task-digest'), resumePending('task-digest')], [false, true],
        'its right is its own, never the press still out for another row; no Resume door offers it again');
    assert.equal(r.stops.get('task-digest').allowed(), true);
    assert.equal(r.stop('task-sweep').disabled, true, 'that row\'s own request is out');
    // A fresh read that cannot vouch for its card withdraws its Stop; one that no longer grants it takes it away.
    r.panic.snapshot(body({ main_attention_complete: false, main_attention: [card('task-sweep')] }), 20);
    assert.deepEqual([r.stop('task-digest').hidden, r.stop('task-digest').disabled], [false, true]);
    assert.equal(r.stops.get('task-digest').allowed(), false, 'a press of its open menu does nothing then');
    r.panic.snapshot(body({ main_attention: [card('task-sweep')] }), 30);
    assert.equal(r.stop('task-digest').hidden, true);
    gate.resolve({ ok: true, task_id: 'task-sweep' });
    await settle();
    assert.equal(r.state.posts.length, 2, 'one Resume each, never again');
});

test('a refusal is said in the host\'s words; only a changed stop says its stop changed, and no answer is no refusal', async (t) => {
    const rows = [ROWS.digest, ROWS.sweep, ROWS.audit,
        { task_id: 'task-moved', task_name: 'Moved', panic_hold_id: 'hold-moved' },
        { task_id: 'task-down', task_name: 'Server failed', panic_hold_id: 'hold-down' }];
    const r = rig(t, { answers: {
        'task-digest': refusal(409, { error: 'installation_panic_pending', message: 'Resume refused: the installation is still stopped by Panic.' }),
        'task-sweep': refusal(409, { error: 'selection_authority_unavailable', message: 'Resume refused: its state could not be read.' }),
        'task-audit': refusal(409, { error: 'selection_owner_held', message: 'Resume refused: it is paused separately.' }),
        'task-moved': refusal(409, { error: 'panic_selection_changed', message: 'Resume refused: The selected Panic hold changed.' }),
        'task-down': refusal(503, { error: 'store unavailable' }),
    } });
    r.panic.snapshot(census(rows), 10);
    r.part('panicAll').click();
    r.part('panicResume').click();
    await settle();
    const said = id => [r.outcome(id).textContent, r.outcome(id).dataset.tone];
    assert.deepEqual(said('task-digest'), ['Resume refused: the installation is still stopped by Panic.', 'error']);
    assert.deepEqual(said('task-sweep'), ['Resume refused: its state could not be read.', 'error']);
    assert.deepEqual(said('task-audit'), ['Resume refused: it is paused separately.', 'error']);
    assert.deepEqual(said('task-moved'), ['Not resumed: its stop changed after you chose it.', 'warn']);
    assert.deepEqual(said('task-down'), ['No answer came back; it may already be back in the queue.', 'neutral']);
    await settle();
    assert.equal(r.state.posts.length, rows.length, 'nothing is sent again by itself');
});

test('the lifetime\'s signal reaches the request: an ended lifetime cancels it and is no answer, never a retry', async (t) => {
    const r = rig(t, { lifetimeMs: 150, answers: { 'task-digest': () => new Promise(() => {}) } });
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 10);
    r.toggle('task-digest', true);
    r.part('panicResume').click();
    await turn();
    assert.equal(r.state.signals[0]?.aborted, false, 'the request carries a live signal');
    for (let i = 0; i < 50 && r.part('panicResume').disabled; i += 1) await new Promise(resolve => setTimeout(resolve, 10));
    assert.equal(r.state.signals[0].aborted, true, 'the ended lifetime cancelled it');
    assert.equal(r.outcome('task-digest').textContent, 'No answer came back; it may already be back in the queue.');
    await new Promise(resolve => setTimeout(resolve, 60));
    assert.equal(r.state.posts.length, 1);
});

test('the shared Resume door hands that signal to fetch with only the chosen stop', async (t) => {
    const r = rig(t, { lifetimeMs: 20, realResume: true });
    const calls = [];
    globalThis.fetch = async (url, init) => { calls.push([url, init]); return new Promise(() => {}); };
    r.panic.snapshot(census([ROWS.digest, ROWS.sweep]), 10);
    r.toggle('task-sweep', true);
    r.part('panicResume').click();
    for (let i = 0; i < 20 && r.part('panicResume').disabled; i += 1) await new Promise(resolve => setTimeout(resolve, 10));
    assert.equal(calls.length, 1);
    const [url, init] = calls[0];
    assert.deepEqual([url, init.method, JSON.parse(init.body)], ['/api/tasks/task-sweep/resume', 'POST', { expected_panic_hold_id: 'hold-sweep' }]);
    assert.equal(init.signal.aborted, true, 'fetch itself was cancelled when the lifetime ended');
    assert.equal(r.outcome('task-sweep').textContent, 'No answer came back; it may already be back in the queue.');
});

// Main itself over the DOM stub: its owner-action cards, the Panic message and the shared menu, every request
// answered by `answer(url)` (the path's own answer, `{ok: true}` unless it says otherwise) and recorded.
function mainRig(t, answer = async () => ({ ok: true })) {
    const env = installDom();
    document.body = env.mount;
    t.after(() => restoreDom(env.prior));
    const priorObserver = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    t.after(() => { globalThis.IntersectionObserver = priorObserver; });
    const page = document.createElement('div');
    const feed = document.createElement('div');
    const toolbar = document.createElement('div');
    toolbar.className = 'chat-toolbar-row';
    page.append(feed, toolbar);
    env.mount.append(page);
    const posts = [];
    globalThis.fetch = async (url, init) => {
        posts.push([url, init?.method]);
        const body = await answer(url);
        return { ok: true, json: async () => body };
    };
    const waiting = createMainWaiting({ page, messagesDiv: feed, toolbar, insertMessageNode: node => { feed.append(node); return true; },
        stampNodeTimestamp, decision: () => ({ syncCurrentQuestions() {}, setConnected() {}, observedQuestion: () => null }) });
    t.after(() => waiting.destroy());
    const message = () => one(feed, node => node.classList?.contains('panic-recovery'));
    const stopOf = id => one(one(message(), node => node.dataset?.taskId === id && node.tagName === 'LI'),
        node => node.tagName === 'BUTTON' && node.textContent === 'Stop…');
    // Opens a Stop… menu and returns its offered items.
    const open = button => {
        for (const fn of button.listeners.get('click') || []) fn({ target: button, stopPropagation() {}, preventDefault() {} });
        return all(env.mount, node => Boolean(node.dataset?.taskControl));
    };
    const tasks = () => posts.filter(([url]) => url.startsWith('/api/tasks/'));
    return { env, feed, waiting, message, stopOf, open, tasks };
}

test('in Main a named work\'s Stop… opens its stop alone, and a card for anything else stays its own', async (t) => {
    const { feed, waiting, message, stopOf, open, tasks } = mainRig(t);
    waiting.snapshot({ current_questions: [], current_questions_complete: true, main_attention_complete: true,
        main_attention: [{ ...card('task-digest'), task_name: 'Weekly digest' },
            { ...card('task-audit', { attention_reason: 'budget', pause_cause: 'budget' }), task_name: 'Disk audit' }],
        ...census([ROWS.digest, ROWS.audit]) });
    const cardOf = id => one(feed, node => node.dataset?.ownerAction === id);
    assert.equal(cardOf('task-digest'), null, 'the message is its card');
    assert.ok(cardOf('task-audit'), 'the spending-limit card stays its own, with its own controls');
    const stop = stopOf('task-digest');
    assert.ok(stop, 'the Stop… its card had stands in its row');
    assert.equal(stopOf('task-audit'), null, 'never a second Stop… beside a card that keeps its own');
    assert.ok(message());
    const items = open(stop);
    assert.deepEqual(items.map(node => node.dataset.taskControl), ['stop_now'], 'its stop alone: the box is its Resume');
    items[0].listeners.get('click')[0]({ target: items[0], stopPropagation() {} });
    await settle();
    assert.deepEqual(tasks(), [['/api/tasks/task-digest/cancel', 'POST']]);
});

test('in Main a work whose Resume was accepted stops by its one Stop while another row\'s answer is still out', async (t) => {
    const gate = deferred();
    const { waiting, message, stopOf, open, tasks } = mainRig(t, async url => {
        if (url === '/api/tasks/task-sweep/resume') await gate.promise;
        return { ok: true };
    });
    waiting.snapshot({ current_questions: [], current_questions_complete: true, main_attention_complete: true,
        main_attention: [card('task-digest'), card('task-sweep')], ...census([ROWS.digest, ROWS.sweep]) });
    one(message(), hasData('panicAll')).click();
    one(message(), hasData('panicResume')).click();
    await settle();
    assert.deepEqual(tasks(), [['/api/tasks/task-digest/resume', 'POST'], ['/api/tasks/task-sweep/resume', 'POST']]);
    const stop = stopOf('task-digest');
    assert.equal(stop.disabled, false, 'its card still grants it; the slow answer is another row\'s');
    assert.equal(stopOf('task-sweep').disabled, true, 'that row\'s own request is out');
    const items = open(stop);
    assert.deepEqual(items.map(node => node.dataset.taskControl), ['stop_now'], 'one Stop, never a second Resume');
    items[0].listeners.get('click')[0]({ target: items[0], stopPropagation() {} });
    await settle();
    assert.deepEqual(tasks().at(-1), ['/api/tasks/task-digest/cancel', 'POST']);
    gate.resolve();
    await settle();
    assert.equal(tasks().filter(([url]) => url.endsWith('/resume')).length, 2, 'each Resume sent once');
});
