import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
import { projection, room, work } from './work_fixture.js';
// Preserve the production test fixture, adding actual descendant lookup for
// lazily inserted controls, which the original flat fixture does not model.
const originalQuery = ElementStub.prototype.querySelector;
ElementStub.prototype.querySelector = function(selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};
function makeInstance(rows, details = {}) {
    const env = installDom(async url => String(url).startsWith('/api/tasks/')
        ? (details[String(url).split('/').at(-1)] ? { ok: true, json: async () => details[String(url).split('/').at(-1)] }
            : { ok: false, status: 404, json: async () => ({error: 'missing'}) })
        : { ok: true, json: async () => String(url).startsWith('/api/chat/history')
            ? {messages: rows} : {active_direct_turns: []} });
    const handlers = new Map();
    const instance = createChatInstance({
        ws: {on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {}},
        state: {activePage: 'chat', projectChatIds: new Set(), unreadCount: 0}, updateUnreadBadge() {},
        stateSnapshots: {begin: () => ({generation: 1, requestedAt: Date.now()}), gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}},
        chatId: 1, idPrefix: 'chat', mountEl: env.mount,
    });
    return {...env, handlers, instance, card: id => walkCard(globalThis.document.byId.get('chat-messages'), id)};
}
test(' current activity restores previously attested Stop after cold unconfirmed replay', async () => {
    const fx = makeInstance([{task_id: 'live', is_progress: true, text: 'Working on the report',
        cancelable: true, ts: '2026-09-09T09:00:00Z'}]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        const card = fx.card('live');
        assert.equal(card.querySelector('[data-live-phase]').hidden, true);
        assert.equal(card.querySelector('[data-cancel-run]'), null);
        fx.instance.hydrateStateSnapshot({active_chat_activities: [{activity_id: 'live', chat_id: 1, kind: 'managed_task', phase: 'working'}],
            active_chat_activities_complete: true, supervisor_ready: true}, Infinity, 2);
        assert.equal(card.querySelector('[data-live-phase]').hidden, false);
        assert.ok(card.querySelector('[data-cancel-run]'), 'positive current activity restores existing control authority');
        fx.handlers.get('chat')({chat_id: 1, task_id: 'live', role: 'assistant', is_progress: true,
            cancelable: true, content: 'Continued real work'});
        assert.ok(card.querySelector('[data-cancel-run]'), 'fresh positive evidence must restore Stop');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});
test(' retained cancelled fact outranks untyped historical answer after result quarantine', async () => {
    const historical = {status: 'cancelled', phase: 'cancelled', ts: '2026-09-09T09:02:00Z', provenance: 'canonical_task_result_after_finalization'};
    const fx = makeInstance([
        {task_id: 'past', is_progress: true, text: 'Work before cancellation', ts: '2026-09-09T09:00:00Z', historical_terminal: historical},
        {task_id: 'past', role: 'assistant', text: 'Preserved partial answer', ts: '2026-09-09T09:02:00Z', historical_terminal: historical},
    ]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.instance.hydrateStateSnapshot({active_chat_activities: [], active_chat_activities_complete: false, supervisor_ready: true}, Infinity, 1);
        assert.equal(fx.card('past').dataset.finished, '0');
        fx.instance.hydrateStateSnapshot({active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true}, Infinity, 2);
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(fx.card('past').querySelector('[data-live-phase]').textContent, 'Stopped');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

test('terminal-root history projection settles a card after a later finalizing summary', async () => {
    const terminal = {status: 'completed', phase: 'done', ts: '2026-09-09T09:02:00Z',
        provenance: 'canonical_task_result_after_finalization'};
    const fx = makeInstance([
        {task_id: 'replayed', is_progress: true, text: 'Working', ts: '2026-09-09T09:00:00Z'},
        {task_id: 'replayed', summary_kind: 'terminal_root_projection', historical_terminal: terminal,
            role: 'system', system_type: 'task_summary', outcome_final: true, ts: '2026-09-09T09:02:00Z'},
        {task_id: 'replayed', summary_kind: 'authored_root_summary', role: 'system',
            system_type: 'task_summary', task_phase: 'finalizing', outcome_final: false,
            status: 'completed', ts: '2026-09-09T09:03:00Z'},
    ]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        assert.equal(fx.card('replayed').dataset.finished, '1');
        assert.equal(fx.card('replayed').querySelector('[data-live-phase]').textContent, 'Done');
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});
for (const via of ['log', 'detail']) test(`child terminal ${via} retains the producer model observation`, async () => {
    const observation = {source: 'usable_solve_response', used_model: 'fallback', requested_model: 'initial',
        used_local: false, requested_use_local: false, llm_call_id: 'call', provider: 'openrouter'};
    const executor = {task_id: 'child', task_attempt: '1', run_id: 'run', attempt_id: 'a01',
        harness_id: 'cursor', phase: 'harness.event', revision: 1, model: 'cursor-model', model_source: 'requested'};
    const fx = makeInstance([], {child: {task_id: 'child', status: 'completed', model_execution: observation}});
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, content: 'child working',
            task_id: 'child', subagent_task_id: 'child', parent_task_id: 'root', delegation_role: 'subagent',
            subagent_role: 'reader', subagent_event: 'running', model: 'initial', executor_observation: executor});
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: via === 'log' ? 'child' : 'root', status: 'completed',
            ...(via === 'log' ? {model_execution: observation} : {})}});
        await new Promise(resolve => setImmediate(resolve));
        const meta = fx.card('child').querySelector('[data-live-meta]').innerHTML;
        const facts = fx.card('child').querySelector('[data-live-facts]').innerHTML;
        // The compact line names the executor and its own (requested) model; the coordinator's model and its
        // solve evidence are labelled facts of the details, never the executor's.
        assert.match(meta, /Cursor/);
        assert.match(meta, /requested cursor-model/);
        assert.doesNotMatch(meta, /fallback|initial/);
        assert.match(facts, /<dt>Last solve response<\/dt><dd>fallback/);
        assert.match(facts, /<dt>Coordinator<\/dt><dd>initial<\/dd>/);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

for (const fresh of [false, true]) test(`pruned model history preserves current evidence: ${fresh}`, async () => {
    const historical = {status: 'completed', phase: 'done', ts: '2026-09-09T09:02:00Z',
        provenance: 'canonical_task_result_after_finalization',
        model_execution: {source: 'usable_solve_response', used_model: 'old-fallback', used_local: false}};
    const rows = [{task_id: 'past', is_progress: true, text: 'Historical work',
        ts: '2026-09-09T09:00:00Z', historical_terminal: historical}];
    const fx = makeInstance(rows);
    try {
        if (fresh) {
            rows[0] = {...rows[0], historical_terminal: undefined,
                model_execution: {source: 'usable_solve_response', used_model: 'current-model', used_local: false}};
            await fx.instance.refreshHistory({revision: 1});
            rows[0] = {...rows[0], model_execution: undefined, historical_terminal: historical};
        }
        await fx.instance.refreshHistory({revision: 2});
        const meta = () => fx.card('past').querySelector('[data-live-meta]').innerHTML;
        const facts = () => fx.card('past').querySelector('[data-live-facts]').innerHTML;
        assert.doesNotMatch(facts(), /<dt>Last solve response<\/dt><dd>old-fallback/);
        if (fresh) assert.match(facts(), /<dt>Last solve response<\/dt><dd>current-model/);
        fx.instance.hydrateStateSnapshot({active_chat_activities: [],
            active_chat_activities_complete: true, supervisor_ready: true}, Infinity, 2);
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(fx.card('past').querySelector('[data-live-phase]').textContent, 'Done');
        const kept = fresh ? 'current-model' : 'old-fallback';
        assert.match(facts(), new RegExp(`<dt>Last solve response</dt><dd>${kept}`));
        // The observed model is the compact line's model fact.
        assert.match(meta(), new RegExp(`>${kept}<`));
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

for (const external of [false, true]) test(`terminal child retains late lineage model independently of solve evidence: ${external}`, async () => {
    const fx = makeInstance([]);
    const child = {chat_id: 1, role: 'assistant', is_progress: true, task_id: 'planner',
        subagent_task_id: 'planner', parent_task_id: 'root', delegation_role: 'subagent',
        subagent_role: 'planner', subagent_event: 'scheduled', content: 'queued', model: '',
        ...(external ? {executor_observation: {task_id: 'planner', task_attempt: '1', run_id: 'run',
            attempt_id: 'a01', harness_id: 'cursor', phase: 'harness.event', revision: 1}} : {})};
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(child);
        fx.handlers.get('chat')({chat_id: 1, role: 'system', system_type: 'task_summary',
            task_id: 'planner', subagent_task_id: 'planner', parent_task_id: 'root',
            delegation_role: 'subagent', subagent_role: 'planner', content: 'Done',
            task_terminal_status: 'completed',
            model_execution: {source: 'usable_solve_response', used_model: 'actual-solver', used_local: false}});
        fx.handlers.get('chat')({...child, subagent_event: 'running', content: 'planning', model: 'google/gemini-3.6-flash'});
        const card = fx.card('planner');
        const meta = card.querySelector('[data-live-meta]').innerHTML;
        const facts = card.querySelector('[data-live-facts]').innerHTML;
        assert.match(facts, external ? /<dt>Coordinator<\/dt><dd>gemini-3\.6-flash<\/dd>/
            : /<dt>Agent model<\/dt><dd>gemini-3\.6-flash<\/dd>/);
        assert.match(facts, /<dt>Last solve response<\/dt><dd>actual-solver/);
        // A delegated child's compact line never borrows the coordinator's model; a native one shows the solver.
        if (external) assert.doesNotMatch(meta, /gemini|actual-solver/);
        else assert.match(meta, />actual-solver</);
        assert.equal(card.dataset.finished, '1', 'late model evidence must not revive the terminal child');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// Root-only census and complete work membership exercise the real card consumer.
const tick = () => new Promise(resolve => setImmediate(resolve));
function childFixture(details) {
    const fx = {details, reads: [], gate: null, members: ['root', 'child'], connected: true};
    const activity = {activity_id: 'root', chat_id: 1, kind: 'managed_task', phase: 'working'};
    const census = () => ({supervisor_ready: true, active_chat_activities_complete: true, active_chat_activities: [activity],
        ...projection(1, [work(activity, {members: {ids: fx.members}})], room('working'))});
    const env = installDom(async raw => {
        const url = String(raw);
        if (url.startsWith('/api/tasks/')) {
            const id = url.slice('/api/tasks/'.length);
            fx.reads.push(id);
            const detail = fx.details[id] || 'missing';
            if (fx.gate) await fx.gate;
            if (detail === 'network') throw new Error('transport unavailable');
            return ['missing', 'unavailable'].includes(detail)
                ? {ok: false, status: detail === 'missing' ? 404 : 503, json: async () => ({})}
                : {ok: true, json: async () => ({task_id: id, ...(typeof detail === 'string' ? {status: detail} : detail)})};
        }
        if (url.startsWith('/api/tasks?')) return {ok: fx.queue !== 'unavailable', status: 503,
            json: async () => ({queue: fx.queue ?? {pending: [], running: []}})};
        return {ok: true, json: async () => (url.startsWith('/api/chat/history') ? {messages: fx.history || []} : census())};
    });
    const handlers = new Map();
    const instance = createChatInstance({
        ws: {on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => fx.connected, send() {}},
        state: {activePage: 'chat', projectChatIds: new Set(), unreadCount: 0}, updateUnreadBadge() {},
        stateSnapshots: {begin: () => ({generation: 1, requestedAt: Date.now()}), gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}},
        chatId: 1, idPrefix: 'chat', mountEl: env.mount,
    });
    const card = id => walkCard(globalThis.document.byId.get('chat-messages'), id);
    return Object.assign(fx, {
        instance, handlers, card, census,
        chip: id => [card(id).dataset.finished, card(id).querySelector('[data-live-phase]').textContent],
        readsOf: id => fx.reads.filter(read => read === id).length,
        child: (id, parent = 'root') => handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true,
            content: `${id} works`, task_id: id, subagent_task_id: id, parent_task_id: parent, root_task_id: 'root',
            delegation_role: 'subagent', subagent_role: 'reader', subagent_event: 'running'}),
        async hydrate(requestedAt = Infinity, override = {}) { instance.hydrateStateSnapshot({...census(), ...override}, requestedAt); await tick(); await tick(); },
        async start() {
            await instance.refreshHistory({revision: 1});
            handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works'});
            fx.child('child');
            await fx.hydrate();
        },
        close() { instance.destroy(); restoreDom(env.prior); },
    });
}

test('a lost child terminal settles from its own result while its root still works', async () => {
    const fx = childFixture({child: 'completed'});
    try {
        await fx.start();
        assert.equal(fx.readsOf('child'), 0, 'a listed child is not read');
        const root = fx.chip('root');
        fx.members = ['root'];
        await fx.hydrate();
        assert.deepEqual(fx.chip('child'), ['1', 'Done']);
        assert.deepEqual(fx.chip('root'), root);
        await fx.hydrate();
        assert.equal(fx.readsOf('child'), 1, 'a settled child is not read again');
    } finally { fx.close(); }
});

test('a nested child settles alone; its open parent child keeps working', async () => {
    const fx = childFixture({child: 'completed', grandchild: 'failed'});
    try {
        await fx.start();
        fx.child('grandchild', 'child');
        await fx.hydrate();
        assert.deepEqual([fx.chip('grandchild'), fx.chip('child')], [['1', 'Failed'], ['0', 'Working']]);
        assert.deepEqual([fx.readsOf('grandchild'), fx.readsOf('child')], [1, 0]);
    } finally { fx.close(); }
});

for (const first of ['running', 'interrupted', 'cancel_requested', 'unavailable', 'network',
    {status:'running', cancel_state:'pending'}, {status:'completed', root_phase_checkpoint:{post_task_synthesis:'running'}}]) {
    test(`a ${JSON.stringify(first)} first child read retries with the next complete body`, async () => {
        const fx = childFixture({child: first});
        try {
            await fx.start();
            fx.members = ['root'];
            await fx.hydrate();
            assert.equal(fx.chip('child')[0], '0', 'the response is not a settled terminal fact');
            fx.details.child = 'cancelled';
            await fx.hydrate();
            assert.deepEqual([fx.readsOf('child'), fx.chip('child')], [2, ['1', 'Stopped']]);
        } finally { fx.close(); }
    });
}

test('a 404 child waits for re-listing or fresh activity, not another poll', async () => {
    const fx = childFixture({});
    const priorNow = Date.now;
    try {
        Date.now = () => 100000;
        await fx.start();
        fx.members = ['root'];
        for (let pass = 0; pass < 3; pass++) await fx.hydrate();
        assert.deepEqual([fx.readsOf('child'), fx.chip('child')[0]], [1, '0']);
        assert.equal(fx.card('child').dataset.inert, '1', 'proven missing outcome is inert, not ongoing work');
        assert.match(fx.card('child').querySelector('[data-live-meta]').innerHTML, /Outcome unavailable/);
        fx.members = ['root', 'child'];
        await fx.hydrate();
        assert.equal(fx.card('child').dataset.inert, '0', 'positive membership revives an unavailable card');
        assert.doesNotMatch(fx.card('child').querySelector('[data-live-meta]').innerHTML, /Outcome unavailable/);
        fx.members = ['root'];
        await fx.hydrate();
        assert.equal(fx.readsOf('child'), 2);
        fx.child('child'); // deliberately the SAME clock tick
        fx.details.child = 'completed';
        await fx.hydrate();
        assert.deepEqual([fx.readsOf('child'), fx.chip('child')], [3, ['1', 'Done']]);
    } finally { Date.now = priorNow; fx.close(); }
});

test('a body requested before fresh child activity cannot admit absence', async () => {
    const fx = childFixture({child: 'completed'});
    try {
        await fx.start(); fx.members = ['root'];
        await fx.hydrate(Date.now() - 1000);
        assert.deepEqual([fx.readsOf('child'), fx.chip('child')], [0, ['0', 'Working']]);
        await fx.hydrate();
        assert.deepEqual(fx.chip('child'), ['1', 'Done']);
    } finally { fx.close(); }
});

for (const patch of [{works_complete:false}, {supervisor_ready:false}, {works:undefined}, {works:null}]) {
    test(`an incomplete child membership observation is not absence: ${JSON.stringify(patch)}`, async () => {
        const fx = childFixture({child:'completed'});
        try {
            await fx.start(); fx.members=['root'];
            await fx.hydrate(Infinity,patch);
            assert.deepEqual([fx.readsOf('child'),fx.chip('child')[0]],[0,'0']);
            await fx.hydrate(); assert.equal(fx.chip('child')[0],'1');
        } finally { fx.close(); }
    });
}

for (const newer of ['activity', 'listed', 'finished', 'destroyed', 'disconnected']) {
    test(`a late child detail respects newer ${newer}`, async () => {
        const fx=childFixture({child:'completed'});
        const priorNow=Date.now;
        let release;
        try {
            Date.now=()=>100000;
            await fx.start(); fx.members=['root'];
            fx.gate=new Promise(resolve=>{release=resolve;});
            await fx.hydrate(); await fx.hydrate();
            assert.equal(fx.readsOf('child'),1,'single flight across snapshots');
            const child=fx.card('child');
            if(newer==='activity') fx.child('child');
            if(newer==='listed') {fx.members=['root','child'];await fx.hydrate();}
            if(newer==='finished') fx.handlers.get('log')({chat_id:1,data:{type:'task_done',task_id:'child',status:'failed'}});
            if(newer==='destroyed') fx.instance.destroy();
            if(newer==='disconnected') {fx.connected=false;fx.handlers.get('close')();}
            fx.gate=null; release(); await tick(); await tick();
            assert.equal(child.dataset.finished,newer==='finished'?'1':'0');
            if(newer==='finished') assert.equal(fx.chip('child')[1],'Failed');
            if(newer==='activity') {await fx.hydrate();assert.equal(fx.chip('child')[0],'1');}
        } finally {Date.now=priorNow; release?.();fx.close();}
    });
}

test('reconnect permits one new read after a confirmed 404', async () => {
    const fx=childFixture({});
    try {
        await fx.start();fx.members=['root'];await fx.hydrate();await fx.hydrate();
        assert.equal(fx.readsOf('child'),1);
        fx.connected=false;fx.handlers.get('close')();await fx.hydrate();
        assert.equal(fx.readsOf('child'),1,'disconnected snapshots cannot admit a read');
        fx.details.child='completed';fx.connected=true;
        await fx.hydrate(); assert.deepEqual([fx.readsOf('child'),fx.chip('child')],[2,['1','Done']]);
    } finally {fx.close();}
});

test('settled child status wins a stale pending-cancel field', async () => {
    const fx=childFixture({child:{status:'completed',cancel_state:'pending'}});
    try {
        await fx.start();fx.members=['root'];await fx.hydrate();
        assert.deepEqual(fx.chip('child'),['1','Done']);
    } finally {fx.close();}
});

test('child terminal detail keeps its own model and cost beside an unfinished root', async () => {
    const fx=childFixture({child:{status:'completed', model_execution:{source:'usable_solve_response',used_model:'child-solver',used_local:false},
        cost_presentation:{scope:'own',tracked_amount:3.5,has_unpriced:false,tracked_final:true,accounting_open:false,has_rows:true}}});
    try {
        await fx.start();fx.members=['root'];await fx.hydrate();
        assert.deepEqual(fx.chip('child'),['1','Done']);
        assert.match(fx.card('child').querySelector('[data-live-meta]').innerHTML,/child-solver/);
        assert.match(fx.card('child').querySelector('[data-live-meta]').innerHTML,/3\.50/);
        assert.equal(fx.chip('root')[0],'0');
        assert.doesNotMatch(fx.card('root').querySelector('[data-live-meta]').innerHTML,/child-solver/);
    } finally {fx.close();}
});

for (const queue of ['unavailable', {}]) test(`complete membership heals a held child despite queue failure: ${JSON.stringify(queue)}`, async () => {
    const fx = childFixture({child: 'running'});
    const hold = {label: 'Waiting for Project verification', detail: 'Authority is unreadable.'};
    try {
        fx.queue = {pending: [{id: 'child', task: {project_admission_hold: hold}}], running: []};
        await fx.start();
        assert.equal(fx.readsOf('child'), 0, 'a listed held child needs no detail read');
        assert.equal(fx.chip('child')[1], hold.label);
        fx.members = ['root']; fx.queue = queue;
        await fx.hydrate();
        assert.equal(fx.readsOf('child'), 1, 'complete absence has an independent strict result reader');
        assert.deepEqual(fx.chip('child'), ['0', hold.label], 'absence alone does not release the hold');
        fx.details.child = 'completed';
        await fx.hydrate(); await fx.hydrate();
        assert.deepEqual([fx.readsOf('child'), fx.chip('child'), fx.chip('root')[0]], [2, ['1', 'Done'], '0']);
    } finally {fx.close();}
});

for (const held of [false, true]) for (const [status, phase, word] of [['completed', 'done', 'Done'], ['cancelled', 'cancelled', 'Stopped']]) {
    test(`child history remains terminal evidence after a confirmed missing row: ${status}, held=${held}`, async () => {
        const fx = childFixture({child: 'unavailable'});
        try {
            fx.history = [{task_id: 'child', subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root',
                delegation_role: 'subagent', subagent_role: 'reader', subagent_event: 'running', is_progress: true,
                text: 'Saved child work', ts: '2026-10-01T01:00:00Z', historical_terminal: {
                    status, phase, ts: '2026-10-01T01:01:00Z', provenance: 'canonical_task_result_after_finalization',
                    model_execution: {source: 'usable_solve_response', used_model: 'retained-child-model', used_local: false}}}];
            if (held) fx.queue = {pending: [{id: 'child', task: {project_admission_hold: {label: 'Waiting for Project verification'}}}], running: []};
            await fx.start(); fx.members = ['root']; fx.queue = 'unavailable';
            await fx.hydrate();
            assert.equal(fx.chip('child')[0], '0', '503 is not evidence that the current row is absent');
            fx.details.child = 'missing'; await fx.hydrate();
            assert.deepEqual(fx.chip('child'), ['1', word]);
            assert.equal(fx.chip('root')[0], '0');
            assert.match(fx.card('child').querySelector('[data-live-meta]').innerHTML, /retained-child-model/);
        } finally {fx.close();}
    });
}

for (const transition of ['reconnect', 'relisted', 'revisit']) test(`an old 404 cannot relatch after ${transition}`, async () => {
    const fx = childFixture({});
    let release;
    try {
        await fx.start(); fx.members = ['root'];
        fx.gate = new Promise(resolve => {release = resolve;});
        await fx.hydrate();
        if (transition === 'reconnect') {
            fx.connected = false; fx.handlers.get('close')(); fx.connected = true;
        } else if (transition === 'relisted') {
            fx.members = ['root', 'child']; await fx.hydrate(); fx.members = ['root'];
        } else await fx.instance.showLatest();
        await fx.hydrate();
        assert.equal(fx.readsOf('child'), 1, 'invalidating the observation retains in-flight custody');
        fx.details.child = 'completed'; fx.gate = null; release(); await tick(); await tick();
        await fx.hydrate();
        assert.deepEqual([fx.readsOf('child'), fx.chip('child')], [2, ['1', 'Done']]);
    } finally {release?.(); fx.close();}
});
