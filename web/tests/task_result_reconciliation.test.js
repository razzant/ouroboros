import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { formatMsgTime } from '../modules/chat_activity.js';
import { ElementStub, PREVIEW_SCOPE, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
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
function makeInstance(rows, details = {}, onDom = () => {}) {
    const env = installDom(async url => String(url).startsWith('/api/tasks/')
        ? (details[String(url).split('/').at(-1)] ? { ok: true, json: async () => details[String(url).split('/').at(-1)] }
            : { ok: false, status: 404, json: async () => ({error: 'missing'}) })
        : { ok: true, json: async () => String(url).startsWith('/api/chat/history')
            ? {profile_scope: PREVIEW_SCOPE, messages: rows} : {active_direct_turns: []} });
    onDom(env);
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
        assert.match(facts, /<dt>Last coordinator solve response<\/dt><dd>fallback/);
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
        assert.match(facts, external ? /<dt>Last coordinator solve response<\/dt><dd>actual-solver/
            : /<dt>Last solve response<\/dt><dd>actual-solver/);
        // A delegated child's compact line never borrows the coordinator's model; a native one shows the solver.
        if (external) assert.doesNotMatch(meta, /gemini|actual-solver/);
        else assert.match(meta, />actual-solver</);
        assert.equal(card.dataset.finished, '1', 'late model evidence must not revive the terminal child');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// A session child's Coordinator row (DESIGN §5: coordinator, provenance and billing stay in details) is recorded by its
// scheduling and running frames; a later chip that states none (a sparse observation, a receipt, a missed terminal's
// stored detail, a cancel before dispatch) replaces the executor's sentence, never those recorded facts.
const COORD = {selection: {kind: 'agent_session', target_id: 'codex=gpt-5.5'},
    snapshot: {model: 'openai/gpt-6-astra', use_local_model: false, chosen_by: 'delegator', selected_supervisor_snapshot: {selected_subagent_id: 'nanny-row'}},
    lineage: {chat_id: 1, role: 'system', is_progress: true, task_id: 'kid', subagent_task_id: 'kid', parent_task_id: 'root',
        root_task_id: 'root', delegation_role: 'subagent', subagent_role: 'reviewer'},
    evidence: {delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_failed: 0, harness_models: ['gpt-5.5-codex']}};
const coordinatorRow = (card) => card.querySelector('[data-live-facts]').innerHTML.match(/<dt>Coordinator<\/dt><dd>([^<]*)<\/dd>/)?.[1];
const scheduledFrame = () => ({...COORD.lineage, content: 'Scheduled', subagent_event: 'scheduled', requested_executor_route: COORD.selection,
    parent_cognitive_route: COORD.snapshot, ts: '2026-11-22T10:00:00Z'});
const runningFrame = () => ({...COORD.lineage, content: 'Running', subagent_event: 'running', requested_executor_route: COORD.selection,
    ts: '2026-11-22T10:00:01Z',
    model: 'openai/gpt-6-astra', executor_route: 'codex=gpt-5.5', parent_cognitive_route: COORD.snapshot,
    coordinator_route: 'openai/gpt-6-astra', coordinator_billing: 'measured', use_local_model: false});
for (const via of ['log', 'detail', 'history']) test(`a child's recorded coordinator outlives its observation and terminal chips: ${via}`, async () => {
    const recorded = 'gpt-6-astra (chosen by the delegator (nanny-row)) · bills as measured';
    const observation = {...COORD.lineage, role: 'assistant', content: 'Reading the renderer', ts: '2026-11-22T10:00:02Z',
        executor_observation: {task_id: 'kid', task_attempt: '0', run_id: 'run', attempt_id: 'a01', harness_id: 'codex',
            phase: 'harness.event', revision: 1}};
    // A stored row's `ts` is when it was first written (its REQUESTED receipt, before any live frame); its last write
    // is the receipt this page missed, so that one, not the creation, competes with the live chips.
    const detail = {task_id: 'kid', status: 'completed', delegation_role: 'subagent', parent_task_id: 'root', model: 'openai/gpt-6-astra',
        ts: '2026-11-22T09:59:00Z', updated_at: '2026-11-22T10:05:00Z',
        parent_cognitive_route: COORD.snapshot, requested_executor_route: COORD.selection,
        subagent_envelope: {execution_evidence: COORD.evidence}};
    const terminal = {...COORD.lineage, content: 'Subagent completed (reviewer).', subagent_event: 'completed', status: 'completed', ts: '2026-11-22T10:04:00Z',
        task_terminal_status: 'completed', executor_route: 'codex=gpt-5.5', requested_executor_route: COORD.selection, execution_evidence: COORD.evidence};
    const fx = makeInstance(via === 'history' ? [scheduledFrame(), runningFrame(), observation, terminal] : [], {kid: detail});
    try {
        await fx.instance.refreshHistory({revision: 1});
        if (via !== 'history') {
            fx.handlers.get('chat')(scheduledFrame());
            // Queued: the snapshot names the coordinator and its provenance; its unstated billing reads unknown.
            assert.equal(coordinatorRow(fx.card('kid')), 'gpt-6-astra (chosen by the delegator (nanny-row)) · bills as unknown');
            fx.handlers.get('chat')(runningFrame());
            assert.equal(coordinatorRow(fx.card('kid')), recorded);
            fx.handlers.get('chat')(observation);
            assert.match(fx.card('kid').querySelector('[data-live-meta]').innerHTML, /last update/);
            assert.equal(coordinatorRow(fx.card('kid')), recorded, 'a sparse observation keeps the recorded coordinator');
            fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: via === 'log' ? 'kid' : 'root', status: 'completed',
                ...(via === 'log' ? {executor_route: 'codex=gpt-5.5', requested_executor_route: COORD.selection, execution_evidence: COORD.evidence} : {})}});
            await new Promise(resolve => setImmediate(resolve));
        }
        const card = fx.card('kid');
        assert.equal(card.dataset.finished, '1');
        assert.match(card.querySelector('[data-live-meta]').innerHTML, /gpt-5\.5-codex/);
        assert.equal(coordinatorRow(card), recorded, 'the terminal receipt keeps the recorded coordinator');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

test('a child cancelled before dispatch keeps its selection and the coordinator its scheduling recorded', async () => {
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(scheduledFrame());
        // The real pending-cancel frame: no dispatch route, no evidence, only the stored selection (events_task_done).
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'kid', status: 'cancelled', requested_executor_route: COORD.selection}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('kid');
        const facts = card.querySelector('[data-live-facts]').innerHTML;
        assert.equal(card.dataset.finished, '1');
        assert.match(facts, /<dt>Executor<\/dt><dd>Selected Codex, model gpt-5\.5 \(requested\); no dispatch was recorded before it ended<\/dd>/);
        assert.doesNotMatch(facts, /run evidence arrives|run outcome unknown/);
        // Its queued chip's facts stay recorded: the same coordinator and the `unknown` basis it read, as on replay.
        assert.equal(coordinatorRow(card), 'gpt-6-astra (chosen by the delegator (nanny-row)) · bills as unknown');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// The run's own progress report (delegate_progress.executor_observation): what the card's line shows while the run is in
// flight, never after its terminal record.
const reported = (ts, extra = {}) => ({...COORD.lineage, role: 'assistant', content: 'Reading the renderer', ts,
    executor_observation: {task_id: 'kid', task_attempt: '0', run_id: 'run', attempt_id: 'a01', harness_id: 'codex',
        phase: 'harness.event', revision: 3, model: 'gpt-5.5-2026-11', model_source: 'observed', ...extra}});
const meta = (card) => card.querySelector('[data-live-meta]').innerHTML;
const lineModel = (card) => meta(card).match(/<span class="chat-live-meta-text">([^<]+)</)?.[1];  // a live line's dot rides inside it
const chipLabel = (card) => meta(card).match(/harness-identity-label">([^<]+)</)?.[1];
const factRow = (card, label) => card.querySelector('[data-live-facts]').innerHTML.match(new RegExp(`<dt>${label}</dt><dd>([^<]*)</dd>`))?.[1];
const stoppedRow = () => ({...COORD.lineage, content: 'Subagent cancelled (reviewer).', subagent_event: 'cancelled', status: 'cancelled',
    task_terminal_status: 'cancelled', ts: '2026-11-22T10:04:00Z', requested_executor_route: COORD.selection, executor_route: 'codex=gpt-5.5'});
// Stop without settlement: the terminal record replaces the progress pin. The line names the scheduled request, the
// Executor details keep the run's last report, and a start the card saw is never told as "no dispatch".
for (const via of ['live', 'live, no route recorded', 'history']) test(`a child stopped without settlement names its scheduled request: ${via}`, async () => {
    const unrouted = via === 'live, no route recorded';
    const running = unrouted ? {...runningFrame(), executor_route: undefined} : runningFrame();
    const fx = makeInstance(via === 'history' ? [scheduledFrame(), running, reported('2026-11-22T10:00:02Z'), stoppedRow()] : []);
    try {
        await fx.instance.refreshHistory({revision: 1});
        if (via !== 'history') {
            fx.handlers.get('chat')(scheduledFrame());
            fx.handlers.get('chat')(running);
            fx.handlers.get('chat')(reported('2026-11-22T10:00:02Z'));
            // In flight, the run's own report is the line's model.
            assert.equal(chipLabel(fx.card('kid')), 'Codex · last update');
            assert.equal(lineModel(fx.card('kid')), 'gpt-5.5-2026-11');
            fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'kid', status: 'cancelled', requested_executor_route: COORD.selection,
                ...(unrouted ? {} : {executor_route: 'codex=gpt-5.5'})}});
            await new Promise(resolve => setImmediate(resolve));
        }
        const card = fx.card('kid');
        assert.equal(card.dataset.finished, '1');
        assert.equal(chipLabel(card), 'Codex · run outcome unknown');
        assert.equal(lineModel(card), 'requested gpt-5.5');
        assert.equal(factRow(card, 'Executor'), 'The Codex route was assigned, model gpt-5.5 (requested), but this terminal record has no '
            + 'delegated-run evidence; the run outcome is unknown. Latest observed activity: Codex, model gpt-5.5-2026-11 (observed), '
            + 'run run, attempt a01, revision 3. This is progress, not a terminal execution receipt.');
        assert.equal(factRow(card, 'Requested'), undefined, 'the line already names the request');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

test('a child stopped after two attempts keeps the request on its line and its latest attempt as history', async () => {
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(scheduledFrame());
        fx.handlers.get('chat')(runningFrame());
        fx.handlers.get('chat')(reported('2026-11-22T10:00:02Z', {model: 'first-model'}));
        fx.handlers.get('chat')(reported('2026-11-22T10:02:00Z', {task_attempt: '1', run_id: 'run-2', attempt_id: 'a02', revision: 1, model: 'second-model'}));
        assert.equal(lineModel(fx.card('kid')), 'second-model', 'the newer attempt reports, live');
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'kid', status: 'cancelled', executor_route: 'codex=gpt-5.5',
            requested_executor_route: COORD.selection}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('kid');
        assert.equal(lineModel(card), 'requested gpt-5.5');
        assert.match(factRow(card, 'Executor'), /Latest observed activity: Codex, model second-model \(observed\), run run-2, attempt a02, revision 1\./);
        assert.doesNotMatch(meta(card).replace(/title="[^"]*"/g, ''), /first-model|second-model/, 'neither attempt claims the stopped line');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// An API-model child runs on no harness. A harness it calls through delegate_start reports progress under the child's
// task id; that report is its tool's, so the card names the child's own model and, once recorded, its own solve response.
test('an API child that calls a harness is named by its own model and solve response, never that harness', async () => {
    const api = {kind: 'api_model', target_id: 'openai::gpt-5.5-pro'}, lineage = {...COORD.lineage, task_id: 'api', subagent_task_id: 'api'};
    const nested = (extra) => ({...lineage, role: 'system', content: 'Codex: reading the renderer', ts: '2026-11-22T10:00:03Z', ...extra,
        executor_observation: {task_id: 'api', task_attempt: '0', run_id: 'nested', attempt_id: 'n01', harness_id: 'codex',
            phase: 'harness.event', revision: 2, model: 'gpt-5.5-codex', model_source: 'requested'}});
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({...lineage, content: 'Scheduled', subagent_event: 'scheduled', requested_executor_route: api, ts: '2026-11-22T10:00:00Z'});
        fx.handlers.get('chat')({...lineage, content: 'Running', subagent_event: 'running', requested_executor_route: api,
            model: 'openai::gpt-5.5-pro', executor_route: 'openai::gpt-5.5-pro', ts: '2026-11-22T10:00:01Z'});
        fx.handlers.get('chat')(nested({requested_executor_route: api}));  // a frame stating the selection
        fx.handlers.get('chat')(nested({ts: '2026-11-22T10:00:04Z'}));  // and a sparse one: the card knows its selection
        assert.doesNotMatch(meta(fx.card('api')), /harness-chip|Codex/);
        assert.equal(lineModel(fx.card('api')), 'gpt-5.5-pro');
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'api', status: 'completed', requested_executor_route: api,
            executor_route: 'openai::gpt-5.5-pro', execution_evidence: {delegated_runs_started: 1, delegated_runs_settled: 1,
                delegated_runs_failed: 0, harness_models: ['gpt-5.5-codex']},
            model_execution: {source: 'usable_solve_response', requested_model: 'openai::gpt-5.5-pro', requested_use_local: false,
                used_model: 'openai::gpt-5.5-pro', reported_model: 'gpt-5.5-pro-2026-11', used_local: false, provider: 'openai', llm_call_id: 'call'}}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('api');
        assert.equal(card.dataset.finished, '1');
        assert.doesNotMatch(meta(card), /harness-chip|Codex/);
        assert.equal(lineModel(card), 'gpt-5.5-pro-2026-11');
        assert.equal(factRow(card, 'Last solve response'), 'gpt-5.5-pro-2026-11 · openai · route: openai::gpt-5.5-pro');
        assert.equal(factRow(card, 'Agent model'), 'gpt-5.5-pro');
        assert.equal(factRow(card, 'Requested'), 'gpt-5.5-pro', 'the provider named a dated model; the request stays in Details');
        assert.equal(factRow(card, 'Executor'), undefined);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// A card keeps its frozen selection and reads a sparse terminal record (a log or stored detail naming neither route nor
// selection) with it: the request returns to the line, the run's last report stays in Details, and a start the card saw
// is never told "no dispatch".
for (const via of ['log', 'detail']) test(`a sparse terminal record ends a selected child's progress pin: ${via}`, async () => {
    const fx = makeInstance([], {kid: {task_id: 'kid', status: 'cancelled', delegation_role: 'subagent', parent_task_id: 'root'}});
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(scheduledFrame());
        fx.handlers.get('chat')(reported('2026-11-22T10:00:02Z'));
        assert.equal(lineModel(fx.card('kid')), 'gpt-5.5-2026-11');
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: via === 'log' ? 'kid' : 'root', status: 'cancelled'}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('kid');
        assert.equal(card.dataset.finished, '1');
        assert.equal(chipLabel(card), 'Codex · run outcome unknown');
        assert.equal(lineModel(card), 'requested gpt-5.5');
        assert.match(factRow(card, 'Executor'), /run outcome is unknown\. Latest observed activity: Codex, model gpt-5\.5-2026-11 \(observed\), run run/);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// Its running notice alone is a recorded start; a child stopped while queued keeps "no dispatch".
for (const started of [true, false]) test(`a sparse terminal record after ${started ? 'a running notice' : 'queueing only'}`, async () => {
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(scheduledFrame());
        if (started) fx.handlers.get('chat')({...runningFrame(), executor_route: undefined, requested_executor_route: undefined});
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'kid', status: 'cancelled'}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('kid');
        assert.equal(card.dataset.finished, '1');
        assert.equal(lineModel(card), 'requested gpt-5.5');
        assert.equal(factRow(card, 'Executor'), started
            ? 'The Codex route was assigned, model gpt-5.5 (requested), but this terminal record has no delegated-run evidence; the run outcome is unknown'
            : 'Selected Codex, model gpt-5.5 (requested); no dispatch was recorded before it ended');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

test('a terminal row carrying its run\'s last report keeps it as the start and as history: history', async () => {
    const last = {...stoppedRow(), executor_route: undefined, requested_executor_route: undefined, executor_observation: {task_id: 'kid',
        task_attempt: '0', run_id: 'run', attempt_id: 'a01', harness_id: 'codex', phase: 'harness.event', revision: 5,
        model: 'gpt-5.5-2026-11', model_source: 'observed'}};
    const fx = makeInstance([scheduledFrame(), last]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        const card = fx.card('kid');
        assert.equal(card.dataset.finished, '1');
        assert.equal(chipLabel(card), 'Codex · run outcome unknown');
        assert.equal(lineModel(card), 'requested gpt-5.5');
        assert.ok(factRow(card, 'Executor').endsWith('unknown. Latest observed activity: Codex, model gpt-5.5-2026-11 (observed), run run, '
            + 'attempt a01, revision 5. This is progress, not a terminal execution receipt.'), factRow(card, 'Executor'));
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// Stop after a typed update, its run started and never settled: the receipt owns the line, the update stays in Details.
test('a child stopped after a typed update keeps that update\'s run, attempt and model source under its receipt', async () => {
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(scheduledFrame());
        fx.handlers.get('chat')(runningFrame());
        fx.handlers.get('chat')(reported('2026-11-22T10:00:02Z'));
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'kid', status: 'cancelled', executor_route: 'codex=gpt-5.5',
            requested_executor_route: COORD.selection, execution_evidence: {delegated_runs_started: 1, delegated_runs_settled: 0}}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('kid');
        assert.equal(card.dataset.finished, '1');
        assert.equal(chipLabel(card), 'Codex · not settled');
        assert.equal(lineModel(card), 'requested gpt-5.5');
        assert.equal(factRow(card, 'Executor'), 'Delegated to your Codex account — 1 run(s) started, none settled. Latest observed activity: '
            + 'Codex, model gpt-5.5-2026-11 (observed), run run, attempt a01, revision 3. This is progress, not a terminal execution receipt.');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// A retried child's receipt keeps only its own attempt's report: the interrupted attempt's run is no history of the next,
// whether the end names its attempt, names none (the card's retry marks date it) or is replayed; the same attempt keeps it.
for (const via of ['live', 'live, end names no attempt', 'history', 'same attempt']) test(`a retried child's receipt keeps only its own attempt's report: ${via}`, async () => {
    const same = via === 'same attempt';
    const first = reported('2026-11-22T10:00:02Z', {task_attempt: '1', run_id: 'old-run'});
    const interrupted = {...COORD.lineage, content: 'Subagent interrupted (reviewer).', subagent_event: 'interrupted', status: 'interrupted',
        task_attempt: 1, ts: '2026-11-22T10:01:00Z'};
    const end = {...stoppedRow(), ...(via === 'live, end names no attempt' ? {} : {task_attempt: same ? 1 : 2}),
        execution_evidence: {delegated_runs_started: 1, delegated_runs_settled: 0}};
    const frames = [scheduledFrame(), {...runningFrame(), task_attempt: 1}, first,
        ...(same ? [] : [interrupted, {...runningFrame(), task_attempt: 2, ts: '2026-11-22T10:02:00Z'}]), end];
    const fx = makeInstance(via === 'history' ? frames : []);
    try {
        await fx.instance.refreshHistory({revision: 1});
        if (via !== 'history') for (const frame of frames) fx.handlers.get('chat')(frame);
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('kid');
        assert.equal(card.dataset.finished, '1');
        assert.equal(chipLabel(card), 'Codex · not settled');
        assert.equal(lineModel(card), 'requested gpt-5.5');
        assert.equal(factRow(card, 'Executor'), 'Delegated to your Codex account — 1 run(s) started, none settled' + (same
            ? '. Latest observed activity: Codex, model gpt-5.5-2026-11 (observed), run old-run, attempt a01, revision 3. This is progress, not a terminal execution receipt.' : ''));
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// A reconnect that missed the retry meets attempt 2's end only: by the log's task_done (its attempt rides `execution_owner`),
// by the stored detail a terminal root's read settles the child with, or by the companion notice naming the attempt. None
// of them may close with attempt 1's run, revision or model as its history.
for (const via of ['log', 'detail', 'notice']) test(`a stop of attempt 2 met after a missed retry keeps none of attempt 1's run: ${via}`, async () => {
    const end = {status: 'cancelled', execution_owner: {task_id: 'kid', task_attempt: 2}, requested_executor_route: COORD.selection,
        executor_route: 'codex=gpt-5.5', execution_evidence: {delegated_runs_started: 1, delegated_runs_settled: 0}};
    const fx = makeInstance([], {kid: {task_id: 'kid', delegation_role: 'subagent', parent_task_id: 'root', ...end}});
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(scheduledFrame());
        fx.handlers.get('chat')({...runningFrame(), task_attempt: 1});
        fx.handlers.get('chat')(reported('2026-11-22T10:00:02Z', {task_attempt: '1', run_id: 'old-run'}));
        assert.equal(lineModel(fx.card('kid')), 'gpt-5.5-2026-11');
        if (via === 'notice') fx.handlers.get('chat')({...stoppedRow(), task_attempt: 2, execution_evidence: end.execution_evidence});
        else fx.handlers.get('log')({chat_id: 1, data: via === 'log' ? {type: 'task_done', task_id: 'kid', ...end}
            : {type: 'task_done', task_id: 'root', status: 'cancelled'}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('kid');
        assert.equal(card.dataset.finished, '1');
        assert.equal(chipLabel(card), 'Codex · not settled');
        assert.equal(lineModel(card), 'requested gpt-5.5');
        assert.equal(factRow(card, 'Executor'), 'Delegated to your Codex account — 1 run(s) started, none settled');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// A parent-addressed refusal names a child id it never admitted (gateway/history.py keeps it a separate request): it is its
// parent's row naming its own refused selection, and an admitted child under that id keeps working on its own executor.
const cardText = (node) => [node?.textContent || '', node?.innerHTML || '', ...(node?.children || []).map(cardText)].join(' ');
for (const via of ['live', 'history']) test(`a refusal is its parent's row and leaves an admitted child of the same id intact: ${via}`, async () => {
    const refusal = {chat_id: 1, role: 'system', system_type: 'subagent_admission_notice', is_progress: true, task_id: 'root',
        subagent_task_id: 'kid', parent_task_id: 'root', root_task_id: 'root', delegation_role: 'subagent', subagent_role: 'reviewer',
        subagent_event: 'rejected', accepted: false, status: 'failed', content: '⚠️ Refused: too many helpers.', ts: '2026-11-22T10:00:03Z',
        requested_executor_route: {kind: 'agent_session', target_id: 'claude=claude-opus-5-5'}};
    const root = {chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works', ts: '2026-11-22T09:59:00Z'};
    const frames = [root, scheduledFrame(), runningFrame(), reported('2026-11-22T10:00:02Z'), refusal];
    const fx = makeInstance(via === 'history' ? frames : []);
    try {
        await fx.instance.refreshHistory({revision: 1});
        if (via === 'live') for (const frame of frames) fx.handlers.get('chat')(frame);
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('kid');
        assert.notEqual(card.dataset.finished, '1', 'the admitted child is not ended by another request\'s refusal');
        assert.match(chipLabel(card), /^Codex/);
        assert.doesNotMatch(meta(card), /Claude Code|claude-opus-5-5/);
        assert.match(cardText(fx.card('root')), /Refused: too many helpers\. · requested Claude Code · claude-opus-5-5/);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// An old record with no selection: its explicit requested text stays (the producer-model test above), but a finished card
// never answers with its run's observed progress model; that report stays in Details.
test('a finished card without a selection never answers with its run\'s observed progress model', async () => {
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, content: 'child working', task_id: 'child',
            subagent_task_id: 'child', parent_task_id: 'root', delegation_role: 'subagent', subagent_role: 'reader', subagent_event: 'running',
            model: 'initial', executor_observation: {task_id: 'child', task_attempt: '1', run_id: 'run', attempt_id: 'a01', harness_id: 'cursor',
                phase: 'harness.event', revision: 1, model: 'cursor-observed', model_source: 'observed'}});
        assert.equal(lineModel(fx.card('child')), 'cursor-observed');
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'child', status: 'completed'}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('child');
        assert.equal(card.dataset.finished, '1');
        assert.doesNotMatch(meta(card).replace(/title="[^"]*"/g, ''), /cursor-observed/);
        assert.match(factRow(card, 'Executor'), /Latest observed activity: Cursor, model cursor-observed \(observed\)/);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// A recorded lane names the model on a child card: a remote API model named `local-…`, before any solve, and a harness's
// progress model gain no local mark from their words.
test('a remote model whose name says `local` gains no local mark: an API child before its solve, a harness report', async () => {
    const api = {kind: 'api_model', target_id: 'local-model'}, lineage = {...COORD.lineage, task_id: 'api', subagent_task_id: 'api'};
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({...lineage, content: 'Running', subagent_event: 'running', requested_executor_route: api, model: 'local-model',
            use_local_model: false, ts: '2026-11-22T10:00:01Z'});
        assert.equal(lineModel(fx.card('api')), 'local-model');
        assert.equal(factRow(fx.card('api'), 'Agent model'), 'local-model');
        fx.handlers.get('chat')(scheduledFrame());
        fx.handlers.get('chat')(reported('2026-11-22T10:00:02Z', {model: 'local-coder'}));
        assert.equal(lineModel(fx.card('kid')), 'local-coder');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// An API child running another model than it selected (a fallback) names it in the lane its frame recorded
// (`use_local_model`); with none recorded, the model's words make no lane.
for (const local of [false, true, undefined]) test(`an API child on another model than selected takes its recorded lane: ${local}`, async () => {
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({...COORD.lineage, task_id: 'api', subagent_task_id: 'api', content: 'Running', subagent_event: 'running',
            ts: '2026-11-22T10:00:01Z', requested_executor_route: {kind: 'api_model', target_id: 'openai::original'}, model: 'openai::local-model',
            ...(local === undefined ? {} : {use_local_model: local})});
        const named = local ? 'local-model (local)' : 'local-model';
        assert.equal(lineModel(fx.card('api')), named);
        assert.equal(factRow(fx.card('api'), 'Agent model'), named);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// DESIGN §5: a root (and an API child) labels its own solve `Last solve response`; only a session child has a coordinator.
test('a root that started a delegated run keeps its own solve and model words', async () => {
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        // The root's own delegated run reports through its progress: the root card wears that harness's chip.
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'lead', content: 'Codex is reading the renderer',
            executor_route: 'codex', model: 'openai/gpt-6-astra', ts: '2026-11-22T10:00:00Z'});
        assert.match(meta(fx.card('lead')), /harness-chip/);
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'lead', status: 'completed', model_execution: {
            source: 'usable_solve_response', requested_model: 'openai/gpt-6-astra', requested_use_local: false,
            used_model: 'openai/gpt-6-astra', used_local: false, provider: 'openrouter', llm_call_id: 'call'}}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('lead');
        assert.equal(card.dataset.finished, '1');
        assert.equal(factRow(card, 'Last solve response'), 'gpt-6-astra · openrouter');
        assert.equal(factRow(card, 'Agent model'), 'gpt-6-astra');
        assert.doesNotMatch(card.querySelector('[data-live-facts]').innerHTML, /[Cc]oordinator|bills as/);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// A helper's card names no local API connection: neither its key's label on the line nor the named route in Details. The
// root's own card keeps both, and the records (dispatch, billing) keep the route as it ran.
test('a helper solving through a named API connection shows its model without the connection; a root keeps it', async () => {
    const route = 'openai@team::example-model';
    const solve = {source: 'usable_solve_response', requested_model: route, requested_use_local: false, used_model: route,
        reported_model: 'example-model-2026', used_local: false, provider: 'openai', llm_call_id: 'call'};
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        const lineage = {...COORD.lineage, task_id: 'api', subagent_task_id: 'api'}, selection = {kind: 'api_model', target_id: route};
        fx.handlers.get('chat')({...lineage, content: 'Running', subagent_event: 'running', requested_executor_route: selection,
            model: route, ts: '2026-11-22T10:00:01Z'});
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'api', status: 'completed', requested_executor_route: selection,
            model_execution: solve}});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'solo', content: 'Working', model: route,
            ts: '2026-11-22T10:00:02Z'});
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'solo', status: 'completed', model_execution: solve}});
        await new Promise(resolve => setImmediate(resolve));
        const helper = fx.card('api'), root = fx.card('solo');
        assert.equal(helper.dataset.finished, '1');
        assert.equal(lineModel(helper), 'example-model-2026');
        assert.equal(factRow(helper, 'Last solve response'), 'example-model-2026 · openai · route: openai::example-model');
        assert.doesNotMatch(meta(helper) + helper.querySelector('[data-live-facts]').innerHTML, /@team|\bteam\b/);
        assert.equal(lineModel(root), 'example-model-2026 · team');
        assert.equal(factRow(root, 'Last solve response'), `example-model-2026 · openai · route: ${route}`);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// The chosen route keeps the dispatch grammar: a bare id is OpenRouter's, so a solve another provider served keeps the
// request; that provider's own reported id of the very route chosen adds none.
for (const target of ['Qwen/Qwen3.5-9B', 'openai-compatible::Qwen/Qwen3.5-9B']) test(`an API helper's request beside its solve's provider: ${target}`, async () => {
    const used = 'openai-compatible::Qwen/Qwen3.5-9B', selection = {kind: 'api_model', target_id: target};
    const lineage = {...COORD.lineage, task_id: 'api', subagent_task_id: 'api'};
    const fx = makeInstance([]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({...lineage, content: 'Running', subagent_event: 'running', requested_executor_route: selection, model: target,
            ts: '2026-11-22T10:00:01Z'});
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'api', status: 'completed', requested_executor_route: selection,
            model_execution: {source: 'usable_solve_response', requested_model: target, requested_use_local: false, used_model: used,
                reported_model: 'Qwen/Qwen3.5-9B', used_local: false, provider: 'openai-compatible', llm_call_id: 'call'}}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('api');
        assert.equal(card.dataset.finished, '1');
        assert.equal(lineModel(card), 'Qwen3.5-9B');
        assert.equal(factRow(card, 'Requested'), target === used ? undefined : 'Qwen/Qwen3.5-9B');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

test('reopening a child card distinguishes one successful executor model from models across runs', async () => {
    const selected = {kind: 'agent_session', target_id: 'codex=gpt-6.2-sol'};
    const cases = [
        ['single', {delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_succeeded: 1,
            harness_models: ['gpt-6.1-sol']}, 'gpt-6.1-sol'],
        // The successful retry settled without a model; the only name on the receipt is from its failed predecessor.
        ['retry', {delegated_runs_started: 2, delegated_runs_settled: 2, delegated_runs_failed: 1,
            harness_models: ['first-attempt-A']}, 'requested gpt-6.2-sol'],
        ['legacy', {delegated_runs_started: 1, delegated_runs_settled: 1,
            harness_models: ['legacy-model']}, 'requested gpt-6.2-sol'],
        ['contradictory', {delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_failed: 0,
            delegated_runs_succeeded: 0, harness_models: ['contradictory-model']}, 'requested gpt-6.2-sol'],
    ];
    const fx = makeInstance(cases.map(([id, evidence]) => ({...COORD.lineage,
        task_id: id, subagent_task_id: id, content: 'Subagent completed (reviewer).', subagent_event: 'completed',
        status: 'completed', task_terminal_status: 'completed', ts: '2026-11-22T10:04:00Z',
        executor_route: 'codex', requested_executor_route: selected, execution_evidence: evidence})));
    try {
        await fx.instance.refreshHistory({revision: 1});
        for (const [id, evidence, expected] of cases) {
            const card = fx.card(id);
            assert.equal(card.dataset.finished, '1', id);
            const meta = card.querySelector('[data-live-meta]').innerHTML;
            const facts = card.querySelector('[data-live-facts]').innerHTML;
            assert.equal(meta.match(/<span class="chat-live-meta-text">([^<]+)<\/span>/)?.[1], expected, id);
            if (id === 'single') {
                assert.match(facts, /<dt>Requested<\/dt><dd>Codex · gpt-6\.2-sol<\/dd>/);
                assert.match(facts, /successful run model: gpt-6\.1-sol/);
            } else assert.ok(facts.includes(`models observed across settled runs: ${evidence.harness_models[0]}`), id);
            if (id === 'retry') assert.match(facts, /2 runs settled \(1 ok, 1 failed\)/);
        }
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

test('same-name API fallback keeps a different frozen request, but does not duplicate the same identity', async () => {
    const row = (id, requestedLocal, usedLocal, requested = 'qwen.gguf', used = requested) => ({...COORD.lineage,
        task_id: id, subagent_task_id: id, subagent_event: 'completed', status: 'completed', task_terminal_status: 'completed',
        content: 'Subagent completed (reviewer).', ts: '2026-11-22T10:04:00Z',
        requested_executor_route: {kind: 'api_model', target_id: `${requested}${requestedLocal ? ' (local)' : ''}`},
        model_execution: {source: 'usable_solve_response', used_model: used, used_local: usedLocal,
            requested_model: requested, requested_use_local: requestedLocal},
    });
    const fx = makeInstance([row('to-remote', true, false), row('to-local', false, true), row('unchanged-local', true, true),
        row('same-vendor', false, false, 'vendor-a/sol'), row('other-vendor', false, false, 'vendor-a/sol', 'vendor-b/sol'),
        row('third-vendor', false, false, 'vendor-a/sol', 'vendor-c/sol'), row('named', false, false, 'vendor-a/sol', 'openai@team::sol')]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        const facts = id => fx.card(id).querySelector('[data-live-facts]').innerHTML;
        assert.match(facts('to-remote'), /<dt>Requested<\/dt><dd>qwen\.gguf \(local\)<\/dd>/);
        assert.match(facts('to-local'), /<dt>Requested<\/dt><dd>qwen\.gguf<\/dd>/);
        assert.doesNotMatch(facts('unchanged-local'), /<dt>Requested<\/dt>/);
        assert.doesNotMatch(facts('same-vendor'), /<dt>Requested<\/dt>/);
        assert.match(facts('other-vendor'), /<dt>Requested<\/dt><dd>vendor-a\/sol<\/dd>/);
        // Two answers under the request's short name keep their own qualified ids in Details; the line keeps the short name.
        const solve = id => factRow(fx.card(id), 'Last solve response');
        assert.equal(solve('other-vendor'), 'vendor-b/sol (initial request: openrouter::vendor-a/sol)');
        assert.equal(solve('third-vendor'), 'vendor-c/sol (initial request: openrouter::vendor-a/sol)');
        assert.equal(solve('named'), 'openai::sol (initial request: openrouter::vendor-a/sol)', 'a connection name stays masked');
        assert.doesNotMatch(facts('named') + meta(fx.card('named')), /team/);
        for (const id of ['other-vendor', 'third-vendor', 'named']) assert.equal(lineModel(fx.card(id)), 'sol', id);
        assert.equal(solve('same-vendor'), 'sol');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// Host notes (`narration: false`) are not the child's words. Its scheduling note and a plain `running` frame date no
// start, so its clock reads its first row, the queue entry, as `since`. The running agent's own typed note (agent.py
// `subagent_started`, at its host time) is the helper task's start: `started` at that time, live and on replay.
const QUEUED_AT = '2026-11-22T10:00:00Z', PLAIN_RUNNING_AT = '2026-11-22T10:03:00Z', STARTED_AT = '2026-11-22T10:07:00Z';
const hostNote = (frame, ts, extra) => ({...frame, narration: false, ts, ...extra});
const queuedNote = () => hostNote(scheduledFrame(), QUEUED_AT, {system_type: 'task_scheduled', content: '🗓️ Scheduled subagent kid (reviewer): check the card'});
const plainRunning = () => hostNote(runningFrame(), PLAIN_RUNNING_AT, {content: 'Running'});
const startedNote = () => hostNote(runningFrame(), STARTED_AT, {system_type: 'subagent_started', content: 'Subagent running (reviewer).'});
for (const via of ['live', 'history']) test(`a child's clock says started only at its typed start; queued and a plain running say since: ${via}`, async () => {
    const fx = makeInstance(via === 'history' ? [queuedNote(), plainRunning(), startedNote()] : []);
    const line = () => fx.card('kid').querySelector('[data-live-meta]').innerHTML;
    const at = (iso) => formatMsgTime(iso).short;
    try {
        await fx.instance.refreshHistory({revision: 1});
        if (via === 'live') {
            const queued = () => {
                assert.ok(line().includes(`since ${at(QUEUED_AT)}`), line());
                assert.doesNotMatch(line(), /started/);
            };
            fx.handlers.get('chat')(queuedNote());
            queued();  // its queue entry is never called a start
            fx.handlers.get('chat')(plainRunning());
            queued();  // nor is a plain running frame's time
            fx.handlers.get('chat')({...startedNote(), ts: undefined});
            queued();  // a typed start without its host time dates nothing
            fx.handlers.get('chat')(startedNote());
        }
        assert.ok(line().includes(`started ${at(STARTED_AT)}`), line());
        assert.doesNotMatch(line(), /since/);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

// Root-only census and complete work membership exercise the real card consumer.
const tick = () => new Promise(resolve => setImmediate(resolve));
function childFixture(details) {
    const fx = {details, reads: [], gate: null, members: ['root', 'child'], connected: true};
    const activity = {activity_id: 'root', chat_id: 1, kind: 'managed_task', phase: 'working'};
    const census = () => ({supervisor_ready: true, active_chat_activities_complete: true, active_chat_activities: [activity],
        ...projection(1, [work(activity, {members: {ids: fx.members, ...(fx.states && {states: fx.states})}})], room('working'))});
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
        return {ok: true, json: async () => (url.startsWith('/api/chat/history') ? {profile_scope: PREVIEW_SCOPE, messages: fx.history || []} : census())};
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

// The real consumer compares the member's `queue_attempt` (0 valid) and the frame's own `task_attempt`.
function childClockHarness() {
    const fx = childFixture({});
    const member = (activity, extra = {}) => ({activity, transition: 'none', call: '', release: 'self', live: activity === 'working',
        confirmed: true, turn: [], because: [], tails: [], mark: activity === 'working' ? 'moving' : 'still', ...extra});
    const frame = (extra) => fx.handlers.get('chat')({chat_id: 1, role: 'system', is_progress: true, narration: false,
        task_id: 'child', subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root',
        delegation_role: 'subagent', subagent_role: 'reader', ...extra});
    const scheduled = () => frame({subagent_event: 'scheduled', accepted: true, system_type: 'task_scheduled',
        content: 'Scheduled subagent child (reader)', ts: '2026-11-22T10:00:00Z'});
    const started = (ts, extra = {}) => frame({subagent_event: 'running', system_type: 'subagent_started',
        content: 'Subagent running (reader).', ts, ...extra});
    const interrupted = (ts, extra = {}) => frame({system_type: 'subagent_terminal_notice',
        subagent_event: 'interrupted', status: 'interrupted', content: 'Subagent interrupted (reader).', ts, ...extra});
    const at = (iso) => formatMsgTime(iso).short;
    const line = () => fx.card('child').querySelector('[data-live-meta]').innerHTML;
    return {fx, member, scheduled, started, interrupted, at, line};
}

test('a requeued child entering Working before the retry\'s typed start says since, never the old attempt\'s started', async () => {
    const {fx, member, scheduled, started, at, line} = childClockHarness();
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works'});
        scheduled();
        started('2026-11-22T10:07:00Z', {task_attempt: 1});
        fx.states = {child: member('working', {queue_attempt: 1})};
        await fx.hydrate();
        assert.equal(fx.chip('child')[1], 'Working');
        assert.ok(line().includes(`started ${at('2026-11-22T10:07:00Z')}`), line());
        fx.states = {child: member('queued', {queue_attempt: 2})};  // interrupted and queued for retry
        await fx.hydrate();
        assert.equal(fx.chip('child')[1], 'Queued');
        assert.doesNotMatch(line(), /started/, line());
        assert.ok(line().includes(`since ${at('2026-11-22T10:00:00Z')}`), line());
        fx.states = {child: member('working', {queue_attempt: 2})};  // census sees Working before the retry's subagent_started arrives
        await fx.hydrate();
        assert.equal(fx.chip('child')[1], 'Working');
        assert.doesNotMatch(line(), /started/, line());
        assert.ok(line().includes(`since ${at('2026-11-22T10:00:00Z')}`), line());
        started('2026-11-22T10:20:00Z', {task_attempt: 2});
        assert.ok(line().includes(`started ${at('2026-11-22T10:20:00Z')}`), line());
    } finally { fx.close(); }
});

test('a requeued child\'s clock holds no earlier start while Queued and preserves an early retry start across both arrival orders', async () => {
    const {fx, member, scheduled, started, at, line} = childClockHarness();
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works'});
        scheduled();
        started('2026-11-22T10:07:00Z', {task_attempt: 0});
        fx.states = {child: member('queued', {queue_attempt: 0})};  // a stale same-attempt Queued poll hides started without wiping it
        await fx.hydrate();
        assert.equal(fx.chip('child')[1], 'Queued');
        assert.doesNotMatch(line(), /started/, line());
        fx.states = {child: member('working', {queue_attempt: 0})};
        await fx.hydrate();
        assert.equal(fx.chip('child')[1], 'Working');
        assert.ok(line().includes(`started ${at('2026-11-22T10:07:00Z')}`), line());
        fx.states = {child: member('queued', {queue_attempt: 1})};
        await fx.hydrate();
        assert.equal(fx.chip('child')[1], 'Queued');
        assert.doesNotMatch(line(), /started/, line());
        assert.ok(line().includes(`since ${at('2026-11-22T10:00:00Z')}`), line());
        started('2026-11-22T10:20:00Z', {task_attempt: 1});  // arrives before the Working census
        assert.equal(fx.chip('child')[1], 'Queued');
        assert.doesNotMatch(line(), /started/, line());
        assert.ok(line().includes(`since ${at('2026-11-22T10:00:00Z')}`), line());
        fx.states = {child: member('working', {queue_attempt: 1})};
        await fx.hydrate();
        assert.equal(fx.chip('child')[1], 'Working');
        assert.ok(line().includes(`started ${at('2026-11-22T10:20:00Z')}`), line());
        scheduled();  // a stale scheduling note
        assert.equal(fx.chip('child')[1], 'Working');
        assert.ok(line().includes(`started ${at('2026-11-22T10:20:00Z')}`), line());
    } finally { fx.close(); }
});

test('Working 0->1->2 across missed Queued polls drops the old start in DOM and rejects late old or unbound start frames on live and reload', async () => {
    const {fx, member, scheduled, started, at, line} = childClockHarness();
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works'});
        scheduled();
        fx.states = {child: member('working', {queue_attempt: 0})};
        await fx.hydrate();
        started('2026-11-22T10:07:00Z', {task_attempt: 0});
        assert.equal(fx.chip('child')[1], 'Working');
        assert.ok(line().includes(`started ${at('2026-11-22T10:07:00Z')}`), line());
        // Retry 0 -> 1 missed Queued between polls: chip stays Working, DOM drops old started for neutral since.
        fx.states = {child: member('working', {queue_attempt: 1})};
        await fx.hydrate();
        assert.equal(fx.chip('child')[1], 'Working');
        assert.doesNotMatch(line(), /started/, line());
        assert.ok(line().includes(`since ${at('2026-11-22T10:00:00Z')}`), line());
        // First-time delayed old start (attempt 0) and an unbound frame (no task_attempt) after the boundary both stay neutral.
        started('2026-11-22T10:08:00Z', {task_attempt: 0});
        started('2026-11-22T10:15:00Z');
        assert.doesNotMatch(line(), /started/, line());
        assert.ok(line().includes(`since ${at('2026-11-22T10:00:00Z')}`), line());
        started('2026-11-22T10:16:00Z', {task_attempt: 1});
        assert.ok(line().includes(`started ${at('2026-11-22T10:16:00Z')}`), line());
        // Retry 1 -> 2 also missed Queued: drops attempt 1's started immediately, then takes attempt 2's start and later updated narration.
        fx.states = {child: member('working', {queue_attempt: 2})};
        await fx.hydrate();
        assert.doesNotMatch(line(), /started/, line());
        assert.ok(line().includes(`since ${at('2026-11-22T10:00:00Z')}`), line());
        started('2026-11-22T10:20:00Z', {task_attempt: 2});
        assert.ok(line().includes(`started ${at('2026-11-22T10:20:00Z')}`), line());
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, narration: true,
            task_id: 'child', subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root',
            delegation_role: 'subagent', subagent_role: 'reader', subagent_event: 'progress',
            content: 'Checking retry output', ts: '2026-11-22T10:22:00Z'});
        assert.ok(line().includes(`updated ${at('2026-11-22T10:22:00Z')}`), line());
    } finally { fx.close(); }

    // History/reload branch: an earlier attempt's replayed start (numbered or legacy unnumbered) stays neutral on a higher queue_attempt.
    for (const oldStart of [{task_attempt: 1}, {}]) {
        const replay = childClockHarness();
        replay.fx.history = [
            {task_id: 'root', role: 'assistant', is_progress: true, text: 'Root still works', ts: '2026-11-22T09:59:00Z'},
            {...queuedNote(), task_id: 'child', subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root'},
            {...startedNote(), ...oldStart, task_id: 'child', subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root'},
        ];
        try {
            replay.fx.states = {child: replay.member('working', {queue_attempt: 2})};
            await replay.fx.instance.refreshHistory({revision: 1});
            await replay.fx.hydrate();
            assert.equal(replay.fx.chip('child')[1], 'Working');
            assert.doesNotMatch(replay.line(), /started/, replay.line());
            assert.ok(replay.line().includes(`since ${replay.at(QUEUED_AT)}`), replay.line());
            replay.started('2026-11-22T10:20:00Z', {task_attempt: 2});
            assert.ok(replay.line().includes(`started ${replay.at('2026-11-22T10:20:00Z')}`), replay.line());
        } finally { replay.fx.close(); }
    }
});

test('interrupted clears started in DOM, blocks delayed revival before the next poll, and keeps a newer attempt\'s start', async () => {
    const {fx, member, scheduled, started, interrupted, at, line} = childClockHarness();
    const neutral = () => {
        assert.doesNotMatch(line(), /started/, line());
        assert.ok(line().includes(`since ${at('2026-11-22T10:00:00Z')}`), line());
    };
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works'});
        scheduled();
        fx.states = {child: member('working', {queue_attempt: 1})};
        await fx.hydrate();
        started('2026-11-22T10:07:00Z', {task_attempt: 1});
        assert.ok(line().includes(`started ${at('2026-11-22T10:07:00Z')}`), line());
        interrupted('2026-11-22T10:12:00Z');  // unnumbered dated interruption before the next census poll
        neutral();
        started('2026-11-22T10:07:00Z', {task_attempt: 1});  // delayed repeat of the interrupted attempt's start
        started('2026-11-22T10:12:00Z', {task_attempt: 1});  // equal timestamp also stays behind the interruption boundary
        neutral();
        started('2026-11-22T10:20:00Z', {task_attempt: 2});
        assert.ok(line().includes(`started ${at('2026-11-22T10:20:00Z')}`), line());
        interrupted('2026-11-22T10:12:00Z');  // delayed old unnumbered interruption never erases the newer start
        assert.ok(line().includes(`started ${at('2026-11-22T10:20:00Z')}`), line());
        interrupted('2026-11-22T10:25:00Z', {task_attempt: 2});  // numbered interruption retires its attempt
        neutral();
        started('2026-11-22T10:26:00Z', {task_attempt: 2});  // cannot revive the same numbered attempt even at a later ts
        neutral();
        started('2026-11-22T10:24:00Z', {task_attempt: 3});  // strictly higher attempt remains positive
        assert.ok(line().includes(`started ${at('2026-11-22T10:24:00Z')}`), line());
        interrupted('2026-11-22T10:30:00Z', {task_attempt: 2});  // delayed older numbered interruption does not erase it
        assert.ok(line().includes(`started ${at('2026-11-22T10:24:00Z')}`), line());
    } finally { fx.close(); }
});

test('mounted and later nested cards paint their own member state while the root works', async () => {
    const fx = childFixture({});
    try {
        await fx.start();
        fx.members = ['root', 'child', 'sibling'];
        const queued = {activity: 'queued', transition: 'none', call: '', release: 'self', live: false,
            confirmed: true, turn: [], because: [], tails: [], mark: 'still'};
        fx.states = {child: {...queued, activity: 'waiting',
            because: [{code: 'sleeping', text: 'Sleeping until it is woken'}],
            tails: [{kind: 'run', state: 'open', code: 'external_runs_open', count: 1, ids: ['run']}]}, sibling: queued};
        fx.queue = {pending: [{id: 'child', task: {project_admission_hold: {}}}], running: []};
        await fx.hydrate();
        fx.child('sibling');
        const child = fx.card('child');
        const word = child.querySelector('[data-live-phase]');
        const reason = child.querySelector('[data-live-phase-secondary]');
        assert.deepEqual([fx.chip('root')[1], word.textContent, fx.chip('sibling')[1]], ['Working', 'Waiting', 'Queued']);
        assert.equal(reason.textContent, 'Sleeping until it is woken · 1 external run is open');
        assert.equal(reason.parentNode, child.querySelector('[data-live-summary-button]'));
        assert.equal(word.getAttribute('aria-label'), `Subagent status: Waiting, ${reason.textContent}`);
        assert.equal(fx.card('sibling').querySelector('[data-live-phase-secondary]').textContent, '');
        fx.states.child = { ...fx.states.child, confirmed: false, activity: 'unknown', because: [], mark: 'ring',
            tails: [{ kind: 'run', state: 'live', code: 'delegated_running', count: 1, ids: ['run'] }] };
        await fx.hydrate();
        const external = { task_id: 'child', run_id: 'run', after_seq: 0, through_seq: 1,
            parts: [{ kind: 'message', actor: 'executor', text: 'Reading', seq: 1 }], source: { kind: 'run_events', read_through: 1 } };
        fx.handlers.get('chat')({ chat_id: 1, role: 'assistant', is_progress: true, task_id: 'child',
            parent_task_id: 'root', subagent_task_id: 'child', delegated_activity: external, content: 'Reading' });
        assert.deepEqual([word.textContent, reason.textContent], ['Activity unconfirmed', '1 external run still working']);
        fx.states = {sibling: queued};
        await fx.hydrate(Infinity, {works_complete: false});
        assert.deepEqual([word.textContent, reason.textContent, word.firstElementChild.dataset.shape],
            ['Activity unconfirmed', '', 'ring']);
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'child', status: 'cancelled'}});
        fx.states.child = queued;
        await fx.hydrate();
        assert.deepEqual(fx.chip('child'), ['1', 'Stopped']);
    } finally { fx.close(); }
});

// Cold open: a nested child's own phase is unconfirmed while its own external run is independently observed.
// Whichever the room reads first, its history page or the first state body, the evidenced external activity
// keeps the child's chip visible with its line, and a member nothing current attests stays a quiet
// historical row (DESIGN "Room work status": a nested card shows its own member and external work).
const coldRows = [
    {task_id: 'root', is_progress: true, text: 'Root works', ts: '2026-11-19T09:00:00Z'},
    {task_id: 'child', subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root',
        delegation_role: 'subagent', subagent_role: 'reader', subagent_event: 'running',
        is_progress: true, text: 'Saved child work', ts: '2026-11-19T09:00:01Z'},
];
const unknownMember = (tails) => ({activity: 'unknown', transition: 'none', call: '', release: 'self',
    live: false, confirmed: false, turn: [], because: [], mark: 'ring', tails});
const liveRun = [{kind: 'run', state: 'live', code: 'delegated_running', count: 1, ids: ['run']}];

for (const first of ['history', 'state']) {
    test(`cold open (${first} first) shows a child its observed external run evidences`, async () => {
        const fx = childFixture({});
        try {
            fx.history = coldRows;
            fx.states = {child: unknownMember(liveRun)};
            if (first === 'state') await fx.hydrate();
            await fx.instance.refreshHistory({revision: 1});
            if (first === 'history') {
                const quiet = fx.card('child');
                assert.equal(quiet.querySelector('[data-live-phase]').hidden, true,
                    'before any state body the replayed row is quiet');
                assert.match(quiet.querySelector('[data-live-meta]').innerHTML, /Activity unconfirmed/);
                await fx.hydrate();
            }
            const card = fx.card('child');
            const chip = card.querySelector('[data-live-phase]');
            assert.equal(chip.hidden, false, 'the observed external run is current evidence in either order');
            assert.equal(chip.textContent, 'Activity unconfirmed');
            assert.equal(card.querySelector('[data-live-phase-secondary]').textContent,
                '1 external run still working');
            assert.notEqual(card.dataset.inert, '1');
            assert.doesNotMatch(card.querySelector('[data-live-meta]').innerHTML, /Activity unconfirmed/);
        } finally { fx.close(); }
    });

    test(`cold open (${first} first) keeps a member nothing observes quiet`, async () => {
        const fx = childFixture({});
        try {
            fx.history = coldRows;
            fx.states = {child: unknownMember([])};
            if (first === 'state') await fx.hydrate();
            await fx.instance.refreshHistory({revision: 1});
            if (first === 'history') await fx.hydrate();
            const card = fx.card('child');
            assert.equal(card.querySelector('[data-live-phase]').hidden, true,
                'an unknown member without observed external work stays a quiet historical row');
            assert.equal(card.dataset.inert, '1');
            assert.match(card.querySelector('[data-live-meta]').innerHTML, /Activity unconfirmed/);
        } finally { fx.close(); }
    });

    test(`cold open (${first} first) keeps a child a complete body lists nowhere quiet`, async () => {
        const fx = childFixture({});
        try {
            fx.history = coldRows;
            fx.members = ['root'];
            if (first === 'state') await fx.hydrate();
            await fx.instance.refreshHistory({revision: 1});
            await fx.hydrate();
            const card = fx.card('child');
            assert.equal(card.querySelector('[data-live-phase]').hidden, true,
                'a member the complete body lists nowhere is no current evidence');
            assert.equal(card.dataset.inert, '1');
        } finally { fx.close(); }
    });
}

// A read started before a child's live frame and answered after it: each frame below lands after `started`.
test('a read older than a live child frame still applies its explicit state and its failure', async () => {
    const fx = childFixture({});
    const priorNow = Date.now;
    let now = 100000;
    const member = (activity, extra = {}) => ({activity, transition: 'none', call: '', release: 'self',
        live: activity === 'working', confirmed: true, turn: [], because: [], tails: [],
        mark: activity === 'working' ? 'moving' : 'still', ...extra});
    const unconfirmed = member('unknown', {confirmed: false, mark: 'ring'});
    const later = () => { const started = now; now += 10; return started; };
    try {
        Date.now = () => now;
        await fx.start();
        fx.states = {child: member('queued')};
        await fx.hydrate(now);
        let started = later();
        fx.child('child');
        assert.equal(fx.chip('child')[1], 'Queued', 'a recorded hold keeps its word against a live frame');
        fx.states = {child: member('working')};
        await fx.hydrate(started);
        assert.equal(fx.chip('child')[1], 'Working', 'the frame the hold ignored discards no explicit state');
        fx.states = {child: unconfirmed};
        await fx.hydrate(now);
        started = later();
        fx.child('child');
        assert.equal(fx.chip('child')[1], 'Working', 'an unconfirmed child takes its frame\'s phase');
        fx.states = {};
        await fx.hydrate(started);
        assert.equal(fx.chip('child')[1], 'Working', 'a read begun before that frame says nothing against it');
        await fx.hydrate(now + 1);
        assert.equal(fx.chip('child')[1], 'Activity unconfirmed', 'a read begun after the frame speaks for the child');
        // A failed read is never outranked by a frame: after a body that listed no member (an empty map) or the child.
        for (const states of [{}, {child: unconfirmed}]) {
            fx.states = states;
            now += 10;
            await fx.hydrate(now);
            started = later();
            fx.child('child');
            assert.equal(fx.chip('child')[1], 'Working');
            fx.instance.hydrateStateSnapshot(null, started);
            await tick(); await tick();
            assert.equal(fx.chip('child')[1], 'Activity unconfirmed', `a failed read after ${JSON.stringify(states)}`);
        }
    } finally { Date.now = priorNow; fx.close(); }
});

// A child's own frames may reach the room before its lineage: the root-shaped card they made becomes the child's card.
test('a shell promoted to a child keeps its newer live phase only against silence and takes a recorded hold', async () => {
    const fx = childFixture({});
    const priorNow = Date.now;
    let now = 200000;
    const queued = {activity: 'queued', transition: 'none', call: '', release: 'self', live: false,
        confirmed: true, turn: [], because: [], tails: [], mark: 'still'};
    const shell = (id) => fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: id,
        content: `${id} works`});
    // A lifecycle frame that paints no phase of its own: what the card says after it is what promotion left.
    const lineage = (id) => fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: id,
        content: `${id} scheduled`, subagent_task_id: id, parent_task_id: 'root', root_task_id: 'root',
        delegation_role: 'subagent', subagent_role: 'reader', subagent_event: 'scheduled'});
    try {
        Date.now = () => now;
        await fx.start();
        fx.members = ['root', 'child', 'held', 'quiet'];
        fx.states = {held: queued};
        await fx.hydrate(now);
        now += 10;
        for (const id of ['held', 'quiet']) shell(id);
        assert.deepEqual([fx.chip('held')[1], fx.chip('quiet')[1]], ['Working', 'Working'], 'root-shaped shells');
        for (const id of ['held', 'quiet']) lineage(id);
        assert.equal(fx.chip('held')[1], 'Queued', 'the recorded hold applies when the shell becomes the child');
        assert.equal(fx.chip('quiet')[1], 'Working', 'an older read silent about it keeps its live phase');
        await fx.hydrate(now + 1);
        assert.equal(fx.chip('quiet')[1], 'Activity unconfirmed', 'a read begun after the frame speaks for the child');
    } finally { Date.now = priorNow; fx.close(); }
});

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

// A retry's executor line (the review's control): attempt 1's run report stops naming the model at the attempt boundary,
// whichever arrives first, the census's `queue_attempt` or the new attempt's own start; the frozen request shows until the
// new attempt reports, and a late report of the old attempt never reclaims the line.
for (const first of ['census', 'frame']) test(`a retried child's line drops the previous attempt's observed model: ${first} first`, async () => {
    const {fx, member, started} = childClockHarness();
    const selection = {requested_executor_route: {kind: 'agent_session', target_id: 'codex=gpt-5.5'}, executor_route: 'codex=gpt-5.5'};
    const report = (attempt, model, ts) => fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'child',
        subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root', delegation_role: 'subagent', subagent_role: 'reader',
        content: 'Reading the renderer', ts, task_attempt: attempt, executor_observation: {task_id: 'child', task_attempt: String(attempt),
            run_id: `run-${attempt}`, attempt_id: 'a01', harness_id: 'codex', phase: 'harness.event', revision: 1, model, model_source: 'observed'}});
    const line = () => fx.card('child').querySelector('[data-live-meta]').innerHTML;
    const model = () => line().match(/<span class="chat-live-meta-text">([^<]+)</)?.[1];
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works'});
        started('2026-11-22T10:07:00Z', {task_attempt: 1, ...selection});
        fx.states = {child: member('working', {queue_attempt: 1})};
        await fx.hydrate();
        report(1, 'first-observed', '2026-11-22T10:08:00Z');
        assert.equal(model(), 'first-observed');
        if (first === 'census') {
            fx.states = {child: member('queued', {queue_attempt: 2})};
            await fx.hydrate();
            assert.equal(fx.chip('child')[1], 'Queued');
            assert.equal(model(), 'requested gpt-5.5', 'queued for the retry, the line names the request, not attempt 1');
            started('2026-11-22T10:20:00Z', {task_attempt: 2, ...selection});
        } else {
            started('2026-11-22T10:20:00Z', {task_attempt: 2, ...selection});
            assert.equal(model(), 'requested gpt-5.5', 'the new attempt\'s start before the census');
            fx.states = {child: member('working', {queue_attempt: 2})};
            await fx.hydrate();
        }
        assert.equal(model(), 'requested gpt-5.5', 'started, before its first report');
        assert.doesNotMatch(line().replace(/title="[^"]*"/g, ''), /first-observed/);
        report(2, 'second-observed', '2026-11-22T10:21:00Z');
        assert.equal(model(), 'second-observed');
        report(1, 'first-observed', '2026-11-22T10:22:00Z');
        assert.equal(model(), 'second-observed', 'a late report of the old attempt never reclaims the line');
    } finally { fx.close(); }
});

// Progress and census out of order: a report names the model only for its own attempt against the latest known one. The
// first numbered census of a later attempt retires an earlier report; a newer report that outruns its census survives it.
for (const order of ['report-then-census', 'census-then-report']) test(`a child's line reconciles a report's own attempt with the census: ${order}`, async () => {
    const {fx, member, scheduled, started} = childClockHarness();
    const selection = {requested_executor_route: {kind: 'agent_session', target_id: 'codex=gpt-5.5'}, executor_route: 'codex=gpt-5.5'};
    const report = (attempt, model, ts) => fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'child',
        subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root', delegation_role: 'subagent', subagent_role: 'reader',
        content: 'Reading the renderer', ts, task_attempt: attempt, ...selection, executor_observation: {task_id: 'child',
            task_attempt: String(attempt), run_id: `run-${attempt}`, attempt_id: 'a01', harness_id: 'codex', phase: 'harness.event',
            revision: 1, model, model_source: 'observed'}});
    const model = () => fx.card('child').querySelector('[data-live-meta]').innerHTML.match(/<span class="chat-live-meta-text">([^<]+)</)?.[1];
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works'});
        scheduled();
        if (order === 'report-then-census') {
            report(1, 'first-observed', '2026-11-22T10:08:00Z');
            assert.equal(model(), 'first-observed');
            fx.states = {child: member('queued', {queue_attempt: 2})};  // the first numbered census is already attempt 2
            await fx.hydrate();
            assert.equal(fx.chip('child')[1], 'Queued');
            assert.equal(model(), 'requested gpt-5.5', 'queued for attempt 2, the line no longer names attempt 1\'s model');
            report(2, 'second-observed', '2026-11-22T10:21:00Z');  // control: the latest attempt's own report
            assert.equal(model(), 'second-observed');
        } else {
            started('2026-11-22T10:07:00Z', {task_attempt: 1, ...selection});
            fx.states = {child: member('queued', {queue_attempt: 2})};
            await fx.hydrate();
            report(3, 'third-observed', '2026-11-22T10:30:00Z');  // arrives before the census names attempt 3
            assert.equal(model(), 'third-observed', 'a newer report is not discarded by a lagging census');
            fx.states = {child: member('working', {queue_attempt: 3})};
            await fx.hydrate();
            assert.equal(model(), 'third-observed', 'the census confirming attempt 3 keeps its report');
            report(2, 'second-observed', '2026-11-22T10:31:00Z');
            assert.equal(model(), 'third-observed', 'an older attempt\'s report never reclaims the line');
        }
    } finally { fx.close(); }
});

// A stop that names no attempt after the first numbered census already showed a later one: the retired report of an earlier
// attempt stays out of the Stopped details whichever of report and census arrived first; the latest attempt's own report stays.
for (const order of ['report-then-census', 'census-then-report', 'same attempt']) test(`an unnumbered stop keeps only the latest attempt's report: ${order}`, async () => {
    const {fx, member, scheduled} = childClockHarness();
    const selection = {requested_executor_route: {kind: 'agent_session', target_id: 'codex=gpt-5.5'}, executor_route: 'codex=gpt-5.5'};
    const report = (attempt, model, ts) => fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'child',
        subagent_task_id: 'child', parent_task_id: 'root', root_task_id: 'root', delegation_role: 'subagent', subagent_role: 'reader',
        content: 'Reading the renderer', ts, task_attempt: attempt, ...selection, executor_observation: {task_id: 'child',
            task_attempt: String(attempt), run_id: `run-${attempt}`, attempt_id: 'a01', harness_id: 'codex', phase: 'harness.event',
            revision: 1, model, model_source: 'observed'}});
    const census = () => { fx.states = {child: member('queued', {queue_attempt: 2})}; return fx.hydrate(); };
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, task_id: 'root', content: 'Root still works'});
        scheduled();
        if (order === 'report-then-census') { report(1, 'first-observed', '2026-11-22T10:08:00Z'); await census(); }
        else { await census(); report(order === 'same attempt' ? 2 : 1, 'reported-model', '2026-11-22T10:08:00Z'); }
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: 'child', status: 'cancelled', ...selection}});
        await new Promise(resolve => setImmediate(resolve));
        const card = fx.card('child');
        assert.equal(card.dataset.finished, '1');
        const executor = factRow(card, 'Executor') || '';
        if (order === 'same attempt') assert.match(executor, /model reported-model \(observed\), run run-2/, executor);
        else assert.doesNotMatch(executor, /observed|run-1/, executor);
    } finally { fx.close(); }
});

// A card is no overlay root (i18n.SKIP_ROOTS): its line's own qualifiers are read through templates at paint and repainted
// when the language changes; the model identifier and time stay verbatim.
test('a child\'s line qualifiers follow the reader\'s language and repaint on a change', async () => {
    const {applyPayload, currentPayload, setMissTransport} = await import('../modules/i18n.js');
    setMissTransport(() => Promise.resolve(null));
    const prior = currentPayload();
    const listeners = {};
    const fx = makeInstance([], {}, () => { globalThis.window.addEventListener = (type, fn) => { (listeners[type] ||= []).push(fn); }; });
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(scheduledFrame());
        assert.equal(lineModel(fx.card('kid')), 'requested gpt-5.5');
        assert.match(meta(fx.card('kid')), /updated /);
        applyPayload({language: 'ru', english: false, entries: {'requested {model}': {text: 'запрошена {model}'}, 'updated {time}': {text: 'обновлено {time}'}}});
        for (const fn of listeners['ouro:language-changed'] || []) fn();
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(lineModel(fx.card('kid')), 'запрошена gpt-5.5');
        assert.match(meta(fx.card('kid')), /обновлено /);
        assert.doesNotMatch(meta(fx.card('kid')), /updated /);
    } finally {applyPayload(prior || {language: '', english: true, entries: {}}); fx.instance.destroy(); restoreDom(fx.prior);}
});
