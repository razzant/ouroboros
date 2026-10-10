import test from 'node:test';
import assert from 'node:assert/strict';
import { handoffPhase, receiptNotice } from '../modules/project_handoff.js';

test('binding alone and offline activity never imply Working', () => {
    assert.equal(handoffPhase(null, null).text, 'Activity unconfirmed');
    assert.equal(handoffPhase({ phase: 'working' }, null, false).text, 'Activity unconfirmed');
    assert.equal(handoffPhase({ phase: 'working' }, null).text, 'Working');
});
test('terminal truth wins over stale census and survives offline', () => {
    for (const online of [true, false]) {
        assert.equal(handoffPhase({ phase: 'working' }, { status: 'completed' }, online).text, 'Done');
        assert.equal(handoffPhase(null, { status: 'failed' }, online).text, 'Failed');
    }
});
test('waiting, queued, paused and finalizing stay distinct', () => {
    assert.equal(handoffPhase({ phase: 'queued' }, null).text, 'Queued');
    assert.equal(handoffPhase({ phase: 'budget_paused' }, null).text, 'Paused');
    assert.equal(handoffPhase({ phase: 'budget_pausing' }, null).text, 'Pausing…');
    assert.equal(handoffPhase({ phase: 'budget_pausing' }, null).motion, false);
    assert.equal(handoffPhase({ phase: 'finalizing' }, null).text, 'Finalizing…');
    assert.equal(handoffPhase({ phase: 'working', required_question: {} }, null).text, 'Activity unconfirmed');
    assert.equal(handoffPhase({ phase: 'working', owner_wait: { owner_wait_state: 'waiting' } }, null).text, 'Waiting for your answer');
    assert.equal(handoffPhase(null, { status: 'interrupted' }).text, 'Activity unconfirmed');
});
test('receipt words: durable and already delivered say nothing, every other word names its gap', () => {
    assert.equal(receiptNotice('durable'), '');
    assert.equal(receiptNotice('already_delivered'), '');
    assert.match(receiptNotice('unregistered'), /not protected against a restart/);
    assert.match(receiptNotice('unavailable'), /could not be sent/);
    assert.match(receiptNotice('origin_unproven'), /no recorded Main origin/);
    assert.match(receiptNotice(undefined), /unconfirmed/);
});

// Exercise the actual controller, not a second phase state machine.
import { createProjectHandoffs } from '../modules/project_handoff.js';
class Element {
    constructor() { this.children = []; this.dataset = {}; this.className = ''; this.hidden = false;
        this.classList = { add: value => { this.className += ` ${value}`; },
            remove: value => { this.className = this.className.split(' ').filter(c => c && c !== value).join(' '); } }; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    setAttribute(key, value) { (this.attrs ||= {})[key] = value; }
    getAttribute(key) { return this.attrs?.[key] ?? null; }
    addEventListener() {}
    querySelector(selector) {
        for (const child of this.children) {
            if (child.className.split(' ').some(name => `.${name}` === selector)) return child;
            const found = child.querySelector(selector); if (found) return found;
        }
        return null;
    }
}
const phaseNode = node => node.querySelector('.chat-live-phase');
const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
function setup(fetchDetail) {
    const saved = globalThis.document;
    globalThis.document = { createElement: () => new Element() };
    const nodes = new Set(), starts = [], annotations = [];
    let scans = 0;
    const feed = { contains: node => nodes.has(node), querySelectorAll: sel => {
        scans++; return sel === '.msg-routing-annotation' ? annotations : starts; } };
    const controller = createProjectHandoffs({ feed, fetchDetail, mutate: fn => fn() });
    function mount(taskId = 't', handoffId = 'h', extra = {}) {
        const node = new Element(); nodes.add(node);
        const anchor = controller.mount(node, { taskId, projectId: 'p', projectName: 'Room', title: 'Work', handoffId, ...extra });
        return { node, anchor, status: phaseNode(anchor) };
    }
    return { controller, nodes, starts, annotations, mount, scans: () => scans,
        done() { controller.destroy(); globalThis.document = saved; } };
}
const census = (activities = [], complete = true) => ({ active_chat_activities: activities,
    active_chat_activities_complete: complete, supervisor_ready: true });

test('canonical handoff identity deduplicates retries, not independent requests', () => {
    const h = setup(async () => null);
    try {
        const first = h.mount('old', 'origin-1');
        assert.equal(h.mount('retry', 'origin-1').anchor, first.node);
        assert.notEqual(h.mount('old', 'origin-2').anchor, first.node);
        const started = new Element(); started.dataset = { taskId: 'retry', projectId: 'p' };
        h.starts.push(started); h.controller.reconcile(); assert.equal(started.hidden, true);
        // The evicted anchor hands over to the shadowed duplicate; the Started
        // row reappears only once no node carries the transfer.
        const retryNode = [...h.nodes][1];
        h.nodes.delete(first.node); h.controller.reconcile();
        assert.equal(started.hidden, true); assert.equal(retryNode.hidden, false);
        h.nodes.delete(retryNode); h.controller.reconcile(); assert.equal(started.hidden, false);
    } finally { h.done(); }
});
test('the live card outranks a receipt row in either arrival order; nothing live is ever hidden', () => {
    const h = setup(async () => null);
    try {
        // Receipt first (WS frame beats the conversion HTTP answer), card second.
        const receipt = h.mount('t', 'h', { kind: 'receipt' });
        const card = h.mount('t', 'h', { kind: 'card' });
        assert.equal(card.anchor, card.node);
        assert.equal(card.node.hidden, false);
        assert.equal(receipt.node.hidden, true);
        // Card first, receipt second: the receipt is the shadow.
        const card2 = h.mount('u', 'h2', { kind: 'card' });
        const receipt2 = h.mount('u', 'h2', { kind: 'receipt' });
        assert.equal(receipt2.anchor, card2.node);
        assert.equal(receipt2.node.hidden, true);
        assert.equal(card2.node.hidden, false);
    } finally { h.done(); }
});
test('evicting the anchor restores the shadowed receipt instead of losing the transfer', () => {
    const h = setup(async () => null);
    try {
        const card = h.mount('t', 'h', { kind: 'card' });
        const receipt = h.mount('t', 'h', { kind: 'receipt' });
        const started = new Element(); started.dataset = { taskId: 't', projectId: 'p' };
        h.starts.push(started); h.controller.reconcile(); assert.equal(started.hidden, true);
        h.nodes.delete(card.node);
        h.controller.snapshot(census([{ activity_id: 't', phase: 'working' }]));
        assert.equal(receipt.node.hidden, false);
        assert.equal(phaseNode(receipt.node).textContent, 'Working');
        assert.equal(started.hidden, true, 'the restored receipt still represents the Started row');
        h.nodes.delete(receipt.node); h.controller.snapshot(census());
        assert.equal(started.hidden, false);
    } finally { h.done(); }
});
test('an ordinary message reconciles only itself; the feed is scanned when an anchor changes', () => {
    const h = setup(async () => null);
    try {
        h.mount('t', 'h', { kind: 'receipt' });
        const before = h.scans();
        const plain = new Element(); plain.dataset = { systemType: 'task_summary' };
        for (let i = 0; i < 50; i++) h.controller.reconcile(plain);
        assert.equal(h.scans(), before, 'no feed-wide query per replayed message');
        const started = new Element(); started.dataset = { systemType: 'project_started', taskId: 't', projectId: 'p' };
        h.controller.reconcile(started);
        assert.equal(started.hidden, true);
        assert.equal(h.scans(), before, 'a Started row is projected against the rows map, not the feed');
        for (let i = 0; i < 5; i++) h.controller.snapshot(census());
        assert.equal(h.scans(), before, 'a steady census tick scans nothing');
    } finally { h.done(); }
});
test('two converted cards of one owner message share an identity and both stay visible', () => {
    // A direct turn and the root it promoted share the origin-based handoff id;
    // converting both must never hide the card the owner just clicked.
    const h = setup(async () => null);
    try {
        const first = h.mount('t1direct', 'origin', { kind: 'card' });
        const second = h.mount('t2promoted', 'origin', { kind: 'card' });
        assert.equal(second.anchor, second.node);
        assert.equal(first.node.hidden, false); assert.equal(second.node.hidden, false);
        const receipt = h.mount('t1direct', 'origin', { kind: 'receipt' });
        assert.equal(receipt.node.hidden, true, 'the durable receipt folds under a visible card');
        h.controller.snapshot(census([{ activity_id: 't2promoted', phase: 'working' }]));
        assert.equal(phaseNode(second.node).textContent, 'Working');
        assert.equal(phaseNode(first.node).textContent, 'Activity unconfirmed', 'each card paints its own subject');
        const started = new Element(); started.dataset = { systemType: 'project_started', taskId: 't1direct', projectId: 'p' };
        h.nodes.delete(first.node); h.controller.snapshot(census());
        assert.equal(receipt.node.hidden, true, 'a surviving card keeps representing the transfer');
        h.controller.reconcile(started);
        assert.equal(started.hidden, true, 'the survivor inherits the evicted card\'s own subjects');
        h.nodes.delete(second.node); h.controller.snapshot(census());
        assert.equal(receipt.node.hidden, false, 'the receipt takes over only when no card remains');
    } finally { h.done(); }
});
test('every folded receipt survives two evictions: the shadow chain is inherited, not cut', () => {
    const h = setup(async () => null);
    try {
        const r1 = h.mount('t', 'h', { kind: 'receipt' });
        const r2 = h.mount('t', 'h', { kind: 'receipt' });
        const card = h.mount('t', 'h', { kind: 'card' });
        assert.equal(r1.node.hidden, true); assert.equal(r2.node.hidden, true);
        h.nodes.delete(card.node); h.controller.reconcile();
        assert.equal(r1.node.hidden, false); assert.equal(r2.node.hidden, true);
        h.nodes.delete(r1.node); h.controller.reconcile();
        assert.equal(r2.node.hidden, false, 'the second duplicate receipt still carries the transfer');
    } finally { h.done(); }
});
test('a promoted shadow keeps the followed retry as its subject instead of cycling', async () => {
    const h = setup(async id => id === 't' ? { status: 'interrupted', superseded_by: 'r' } : { status: 'running' });
    try {
        const card = h.mount('t', 'h', { kind: 'card' });
        const receipt = h.mount('t', 'h', { kind: 'receipt' });
        h.controller.snapshot(census()); await flush(); await flush();
        h.nodes.delete(card.node);
        h.controller.snapshot(census([{ activity_id: 'r', phase: 'working' }]));
        assert.equal(phaseNode(receipt.node).textContent, 'Working');
    } finally { h.done(); }
});
test('a durable receipt arriving later clears the card\'s not-saved mark', () => {
    const h = setup(async () => null);
    try {
        const card = h.mount('t', 'h', { kind: 'card', receipt: 'unavailable' });
        assert.equal(card.node.dataset.receipt, 'unavailable');
        h.mount('t', 'h', { kind: 'receipt' });
        assert.equal(card.node.dataset.receipt, undefined);
        assert.doesNotMatch(card.node.className, /project-handoff--unsaved/);
    } finally { h.done(); }
});
test('a converted card with a non-durable receipt is marked, a durable one is not', () => {
    const h = setup(async () => null);
    try {
        const saved = h.mount('a', 'h1', { kind: 'card', receipt: 'durable' });
        assert.equal(saved.node.dataset.receipt, undefined);
        const unsaved = h.mount('b', 'h2', { kind: 'card', receipt: 'origin_unproven' });
        assert.equal(unsaved.node.dataset.receipt, 'origin_unproven');
        assert.match(unsaved.node.className, /project-handoff--unsaved/);
    } finally { h.done(); }
});
test('slow terminal response survives unrelated census ticks; no repeated detail polling', async () => {
    let settle, calls = 0;
    const h = setup(() => { calls++; return new Promise(resolve => { settle = resolve; }); });
    try {
        const { status } = h.mount();
        h.controller.snapshot(census()); await flush();
        for (let i = 0; i < 10; i++) h.controller.snapshot(census());
        assert.equal(calls, 1);
        settle({ status: 'completed' }); await flush();
        assert.equal(status.textContent, 'Done');
        h.controller.snapshot(census()); await flush(); assert.equal(calls, 1);
    } finally { h.done(); }
});
test('failed detail stays unknown until reconnect, and new positive evidence invalidates an old read', async () => {
    let calls = 0, settle;
    const h = setup(() => { calls++; if (calls === 1) throw new Error('unavailable');
        return new Promise(resolve => { settle = resolve; }); });
    try {
        const { status } = h.mount();
        h.controller.snapshot(census()); await flush();
        h.controller.snapshot(census()); await flush(); assert.equal(calls, 1);
        h.controller.setConnected(false); h.controller.setConnected(true);
        h.controller.snapshot(census()); await flush(); assert.equal(calls, 2);
        h.controller.snapshot(census([{ activity_id: 't', phase: 'working' }]));
        settle({ status: 'completed' }); await flush(); assert.equal(status.textContent, 'Working');
        h.controller.snapshot(census([], false)); assert.equal(status.textContent, 'Activity unconfirmed');
    } finally { h.done(); }
});
test('explicit retry linkage selects successor without a second anchor; disposal ignores late result', async () => {
    const ids = []; let settle;
    const h = setup(id => { ids.push(id); return id === 't'
        ? { status: 'interrupted', superseded_by: 'r' }
        : new Promise(resolve => { settle = resolve; }); });
    try {
        const { status } = h.mount(); h.controller.snapshot(census()); await flush();
        assert.deepEqual(ids, ['t', 'r']);
        h.controller.destroy(); settle({ status: 'completed' }); await flush();
        assert.equal(status.textContent, 'Activity unconfirmed');
    } finally { h.done(); }
});
test('model waits use the current attempt and ignore resolved older waits', () => {
    const model_waits = { old: { wait_id: 'old', revision: 1, reason: 'quota', state: 'waiting', task_attempt: 1 },
        now: { wait_id: 'now', revision: 1, reason: 'quota', state: 'resolved', task_attempt: 2 } };
    assert.equal(handoffPhase({ phase: 'working', model_waits, task_attempt: 2 }).text, 'Working');
    model_waits.now.state = 'waiting';
    assert.equal(handoffPhase({ phase: 'working', model_waits, task_attempt: 2 }).text, 'Waiting for access');
});

test('real direct-turn census thinking is positive activity', () => {
    assert.equal(handoffPhase({ activity_id: 'direct', kind: 'direct_chat', phase: 'thinking' }).text, 'Thinking');
});
test('effective retry detail selects its projected task_id, not the requested ancestor', async () => {
    const ids = [];
    const h = setup(id => { ids.push(id); return id === 't'
        ? { task_id: 'r', original_task_id: 't', retry_lineage: [{ task_id: 't', retry_task_id: 'r' }], status: 'running' }
        : { task_id: 'r', status: 'completed' }; });
    try {
        const { status } = h.mount(); h.controller.snapshot(census()); await flush(); await flush();
        assert.deepEqual(ids, ['t', 'r']); assert.equal(status.textContent, 'Done');
    } finally { h.done(); }
});
test('a painted Main card shows a budget pause and the Project wait together, then each clears alone', () => {
    const h = setup(async () => null);
    try {
        const card = h.mount('t', 'h', { kind: 'card' });
        const hold = { label: 'Waiting for Project verification', reason: 'project_routing_fence_lookup_failed' };
        h.controller.snapshot(census([{ activity_id: 't', phase: 'budget_paused', project_admission_hold: hold }]));
        assert.equal(card.status.textContent, `Paused · ${hold.label}`);
        assert.equal(card.status.className, 'chat-live-phase warn');
        h.controller.snapshot(census([{ activity_id: 't', phase: 'queued', project_admission_hold: hold }]));
        assert.equal(card.status.textContent, hold.label);
        h.controller.snapshot(census([{ activity_id: 't', phase: 'budget_paused' }]));
        assert.equal(card.status.textContent, 'Paused');
    } finally { h.done(); }
});

test('creation has its own live presentation and type, folding only under matching transferred work', () => {
    const h = setup(async () => null);
    try {
        const start = h.mount('t', '', { kind: 'started' }); h.starts.push(start.node);
        h.controller.reconcile(start.node);
        h.controller.snapshot(census([{ activity_id: 't', phase: 'working' }]));
        assert.equal(start.node.dataset.systemType, 'project_started');
        assert.equal(start.node.dataset.handoffId, undefined);
        assert.equal(start.node.hidden, false, 'a creation never folds itself');
        assert.equal(start.status.textContent, 'Working');
        assert.equal(start.status.dataset.motion, '1');
        const unrelated = h.mount('other', 'unrelated');
        h.controller.reconcile(unrelated.node);
        assert.equal(start.node.hidden, false, 'sharing a Project does not identify work');
        const receipt = h.mount('t', 'transfer'); h.controller.reconcile(receipt.node);
        assert.equal(start.node.hidden, true);
        h.nodes.delete(receipt.node);
        h.controller.snapshot(census([{ activity_id: 't', phase: 'queued' }]));
        assert.equal(start.node.hidden, false);
        assert.equal(start.status.textContent, 'Queued');
        assert.equal(start.status.dataset.motion, '0');
    } finally { h.done(); }
});

test('receipt grouping never lends a completed task outcome to another converted root', async () => {
    const h = setup(async id => ({ task_id: id, status: 'completed' }));
    try {
        const receipt = h.mount('finished', 'same-origin');
        h.controller.snapshot(census()); await flush();
        assert.equal(receipt.status.textContent, 'Done');
        const card = h.mount('working', 'same-origin', { kind: 'card' });
        h.controller.snapshot(census([{ activity_id: 'working', phase: 'working' }]));
        assert.equal(card.status.textContent, 'Working');
        assert.equal(card.status.dataset.motion, '1');
        h.nodes.delete(card.node); h.controller.snapshot(census()); await flush();
        assert.equal(receipt.node.hidden, false);
        assert.equal(receipt.status.textContent, 'Done', 'the restored receipt retains its own subject');
    } finally { h.done(); }
});

test('an in-flight detail read before retry publication is revisited by normal census refresh', async () => {
    let linked = false; const calls = [];
    const h = setup(async id => { calls.push(id); return linked
        ? { task_id: 'successor', original_task_id: 't', status: 'completed' }
        : { task_id: 't', status: 'running' }; });
    try {
        const { status } = h.mount(); h.controller.snapshot(census()); await flush();
        assert.equal(status.textContent, 'Activity unconfirmed');
        linked = true; h.controller.snapshot(census()); await flush(); await flush();
        assert.equal(status.textContent, 'Done');
        assert.deepEqual(calls, ['t', 't', 'successor']);
        h.controller.snapshot(census()); await flush(); assert.equal(calls.length, 3);
    } finally { h.done(); }
});

test('current retry evidence heals a failed detail read; owner Continue stays a separate root', async () => {
    const h = setup(async () => { throw new Error('missing detail'); });
    try {
        const { status } = h.mount(); h.controller.snapshot(census()); await flush();
        h.controller.snapshot(census([{ activity_id: 'continued', predecessor_task_id: 't', phase: 'working' }]));
        assert.equal(status.textContent, 'Activity unconfirmed');
        h.controller.snapshot(census([{ activity_id: 'retry', timeout_retry_from: 't', phase: 'working' }]));
        assert.equal(status.textContent, 'Working');
        assert.equal(status.dataset.motion, '1');
        h.controller.setConnected(false); h.controller.setConnected(true);
        assert.equal(status.textContent, 'Activity unconfirmed', 'reconnect alone cannot revive cached activity');
        h.controller.snapshot(census([{ activity_id: 'retry', phase: 'working' }]));
        assert.equal(status.textContent, 'Working');
    } finally { h.done(); }
});

test('census-only outcomes survive disconnection and partial omission without reviving motion', () => {
    const h = setup(async () => null);
    try {
        const card = h.mount();
        const secondary = card.node.querySelector('.chat-live-phase-secondary');
        const activity = { activity_id: 't', phase: 'finalizing', status: 'failed',
            root_phase_checkpoint: { post_task_synthesis: 'running' } };
        h.controller.snapshot(census([activity]));
        assert.equal(card.status.textContent, 'Failed');
        assert.equal(secondary.textContent, 'Finalizing…');
        assert.equal(secondary.dataset.motion, '1');
        h.controller.snapshot(census([], false));
        assert.equal(card.status.textContent, 'Failed');
        assert.equal(secondary.textContent, 'Activity unconfirmed');
        assert.equal(secondary.dataset.motion, '0');
        h.controller.snapshot(census([activity]));
        h.controller.setConnected(false);
        h.controller.setConnected(true);
        assert.equal(card.status.textContent, 'Failed');
        assert.equal(secondary.textContent, 'Activity unconfirmed');
        assert.equal(secondary.dataset.motion, '0');
        h.controller.snapshot(census([activity]));
        assert.equal(secondary.textContent, 'Finalizing…');
        assert.equal(secondary.dataset.motion, '1');
        h.controller.snapshot(census([{ activity_id: 'retry', timeout_retry_from: 't', phase: 'working' }]));
        assert.equal(card.status.textContent, 'Working', 'a successor cannot borrow the predecessor result');
        assert.equal(secondary.hidden, true);
    } finally { h.done(); }
});

test('fresh outcome facts replace cached open finalization, including offline', () => {
    const h = setup(async () => null);
    try {
        const card = h.mount();
        const secondary = card.node.querySelector('.chat-live-phase-secondary');
        h.controller.snapshot(census([{ activity_id: 't', phase: 'finalizing', status: 'failed',
            root_phase_checkpoint: { post_task_synthesis: 'running' } }]));
        h.controller.snapshot(census([{ activity_id: 't', phase: 'finalizing', status: 'completed',
            root_phase_checkpoint: { post_task_synthesis: 'completed' } }]));
        h.controller.setConnected(false);
        assert.equal(card.status.textContent, 'Done');
        assert.equal(secondary.hidden, true, 'a completed checkpoint replaces the cached running checkpoint');
        assert.equal(card.status.dataset.motion, '0');
    } finally { h.done(); }
});
