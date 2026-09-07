// Logs diagnostic classification (#323) and the fan-out label (#318): the
// category a row files under and the phase pill it shows come from ONE typed
// projection (summarizeLogEvent), so live frames and replayed rows agree, a
// recovered/cosmetic event with a failure-shaped name stays inspectable
// without being filed as an unresolved error, and the name-substring test is
// pinned as the non-expanding REMAINDER for unknown names carrying no typed
// fact — not a taxonomy.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import { categorizeLogEvent, summarizeLogEvent } from '../modules/log_events.js';

const logEventsSource = readFileSync(new URL('../modules/log_events.js', import.meta.url), 'utf8');

// [name, event, expected phase, expected category]
const TABLE = [
    // genuine failures with typed facts
    ['failed task_done', { type: 'task_done', task_id: 't', status: 'failed', reason_code: 'tool_failure' }, 'error', 'errors'],
    ['live tool failure', { type: 'tool_call_finished', tool: 'run_command', is_error: true }, 'error', 'errors'],
    ['tool timeout', { type: 'tool_call_timeout', tool: 'delegate_wait', timeout_sec: 30 }, 'timeout', 'errors'],
    ['LLM call failure (live)', { type: 'llm_round_error', error: 'boom' }, 'error', 'errors'],
    ['LLM call failure (durable)', { type: 'llm_api_error', error: 'boom', model: 'm' }, 'error', 'errors'],
    ['extension frame with level=error', { type: 'extension_ws', level: 'error', message: 'handler raised' }, 'error', 'errors'],
    ['typed ok=false', { type: 'managed_update_rollback_after_failed_boot', ok: false, msg: 'rollback failed' }, 'error', 'errors'],
    // genuine failures whose ONLY typed fact is the name (the pinned remainder)
    ['crash storm (counters only)', { type: 'crash_storm_detected', crash_count: 4, worker_count: 2 }, 'error', 'errors'],
    ['cascade scope record failed (task id only)', { type: 'cascade_scope_record_failed', task_id: 'x' }, 'error', 'errors'],
    // recovered / warned: typed facts outrank the failure-shaped name
    ['rollback after failed boot succeeded', { type: 'managed_update_rollback_after_failed_boot', ok: true, msg: 'rolled back' }, 'ok', 'system'],
    ['settings reload failed at task start', { type: 'task_start_settings_reload_failed', task_id: 't', error: 'bad json' }, 'warn', 'tasks'],
    ['extension frame with level=warning', { type: 'extension_ws', level: 'warning', message: 'handler not live' }, 'warn', 'system'],
    // partial / degraded outcome is a warning, not an unresolved error
    ['degraded task_done', { type: 'task_done', task_id: 't', status: 'completed', outcome_axes: { execution: { status: 'degraded' } } }, 'warn', 'tasks'],
    ['cancelled task_done', { type: 'task_done', task_id: 't', status: 'cancelled' }, 'cancelled', 'tasks'],
    // ordinary rows keep their domain family
    ['clean tool result', { type: 'tool_call_finished', tool: 'read_file', is_error: false }, 'done', 'tools'],
    ['LLM usage', { type: 'llm_usage', model: 'm' }, 'usage', 'llm'],
    ['unknown quiet event', { type: 'future_scheduler_tick' }, 'info', 'system'],
];

test('Logs category is derived from the typed phase of the same projection', () => {
    for (const [name, evt, phase, category] of TABLE) {
        const view = summarizeLogEvent(evt);
        assert.equal(view.phase, phase, `${name}: phase`);
        assert.equal(categorizeLogEvent(evt, view), category, `${name}: category (with view)`);
        assert.equal(categorizeLogEvent(evt), category, `${name}: category (self-summarized)`);
    }
});

test('a replayed tools.jsonl row reads the same failure facts as the live frame', () => {
    const live = summarizeLogEvent({ type: 'tool_call_finished', tool: 'run_command', is_error: true, duration_sec: 2 });
    const replayed = summarizeLogEvent({ type: 'tool_call', tool: 'run_command', is_error: true, duration_sec: 2, result_preview: 'exit 1' });
    assert.equal(replayed.phase, live.phase);
    assert.equal(replayed.phase, 'error');
    assert.equal(replayed.headline, 'run_command failed');
    assert.equal(categorizeLogEvent({ type: 'tool_call', tool: 'run_command', is_error: true }), 'errors');
    const killed = summarizeLogEvent({ type: 'tool_call', tool: 'run_command', signal: 'SIGKILL' });
    assert.equal(killed.phase, 'error');
    assert.equal(killed.headline, 'run_command killed (SIGKILL)');
    const clean = summarizeLogEvent({ type: 'tool_call', tool: 'read_file', is_error: false, result_preview: 'ok' });
    assert.equal(clean.phase, 'result');
    assert.equal(clean.headline, 'read_file result');
    assert.equal(categorizeLogEvent({ type: 'tool_call', tool: 'read_file' }), 'tools');
});

test('the name-substring remainder is disclosed: unknown names with no typed fact only', () => {
    // An unknown name carrying no `ok`/`level` keeps the remainder verdict so a
    // producer-side failure with only a name stays visible under Errors ...
    const unknown = summarizeLogEvent({ type: 'future_worker_crash_recovered' });
    assert.equal(unknown.phase, 'error');
    assert.equal(categorizeLogEvent({ type: 'future_worker_crash_recovered' }), 'errors');
    // ... while the SAME name with a typed recovery fact is not an error.
    assert.equal(summarizeLogEvent({ type: 'future_worker_crash_recovered', ok: true }).phase, 'ok');
    assert.equal(categorizeLogEvent({ type: 'future_worker_crash_recovered', ok: true }), 'system');
    // Exactly one name-substring severity test remains in the module — the
    // Logs fallback remainder — and categorizeLogEvent itself has none.
    const categorizer = logEventsSource.slice(
        logEventsSource.indexOf('export function categorizeLogEvent'),
        logEventsSource.indexOf('export function normalizeLogTs'),
    );
    assert.doesNotMatch(categorizer, /includes\('error'\)|includes\('crash'\)|includes\('fail'\)/);
    assert.equal(logEventsSource.match(/t\.includes\('error'\) \|\| t\.includes\('crash'\) \|\| t\.includes\('fail'\)/g).length, 1);
    // The extension-frame `message` body is rendered, not dropped.
    assert.equal(summarizeLogEvent({ type: 'extension_ws', level: 'error', message: 'handler raised' }).body, 'handler raised');
});

test('a delegated harness run is not labelled as a subagent in the fan-out row (#318)', () => {
    const delegated = summarizeLogEvent({
        type: 'swarm_fanout', role: 'delegated_run', requested_count: 1, task_group_id: 'g1',
    });
    assert.equal(delegated.headline, 'swarm fan-out: delegated run requested');
    assert.ok(delegated.meta.includes('role=delegated_run'));
    const subagents = summarizeLogEvent({ type: 'swarm_fanout', role: 'researcher', requested_count: 3 });
    assert.equal(subagents.headline, 'swarm fan-out: 3 subagent(s) requested');
    assert.equal(summarizeLogEvent({ type: 'swarm_fanout', task_ids: ['a', 'b'] }).headline,
        'swarm fan-out: 2 subagent(s) requested');
});
