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
import { machineWords } from '../modules/utils.js';

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
    ['companion exhausted its restarts', { type: 'extension_companion_exhausted', returncode: 7 }, 'error', 'errors'],
    ['companion restart failed', { type: 'extension_companion_restart_failed', returncode: 7 }, 'error', 'errors'],
    ['companion replacement failed', { type: 'extension_companion_refresh_failed' }, 'error', 'errors'],
    ['companion recovered', { type: 'extension_companion_recovered', returncode: 7 }, 'info', 'system'],
    ['superseded companion exited', { type: 'extension_companion_exited', returncode: 7 }, 'info', 'system'],
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
    assert.equal(replayed.headline, 'Run command failed');
    assert.equal(categorizeLogEvent({ type: 'tool_call', tool: 'run_command', is_error: true }), 'errors');
    const killed = summarizeLogEvent({ type: 'tool_call', tool: 'run_command', signal: 'SIGKILL' });
    assert.equal(killed.phase, 'error');
    assert.equal(killed.headline, 'Run command killed (SIGKILL)');
    const clean = summarizeLogEvent({ type: 'tool_call', tool: 'read_file', is_error: false, result_preview: 'ok' });
    assert.equal(clean.phase, 'result');
    assert.equal(clean.headline, 'Read file result');
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
    assert.ok(delegated.meta.includes('role delegated_run'));
    const subagents = summarizeLogEvent({ type: 'swarm_fanout', role: 'researcher', requested_count: 3 });
    assert.equal(subagents.headline, 'swarm fan-out: 3 subagent(s) requested');
    assert.equal(summarizeLogEvent({ type: 'swarm_fanout', task_ids: ['a', 'b'] }).headline,
        'swarm fan-out: 2 subagent(s) requested');
});

test('a provider content refusal reads as the filter stopping the request, with its model and the provider\'s words', () => {
    // Before: no branch, so the row showed the raw type name. The words are about the wire, never a verdict.
    const refusal = summarizeLogEvent({
        type: 'provider_content_refusal', task_id: 't', model: 'anthropic/claude-opus-5.5', round: 7,
        response_provider: 'Amazon Bedrock', provider_message: 'This request was blocked.',
    });
    assert.equal(refusal.phase, 'empty');
    assert.equal(refusal.headline, "The provider's content filter stopped this request · claude-opus-5.5", 'its model is its subject');
    assert.equal(refusal.body, 'The provider said: "This request was blocked."');
    // Its facts: the round, the provider. The task's id is a machine reference: its record has it.
    assert.deepEqual(refusal.meta, ['r7', 'Amazon Bedrock']);
    assert.equal(summarizeLogEvent({ type: 'provider_content_refusal', model: 'm' }).body, '');  // no words: none claimed
    // A model call's own outcome: the LLM filter shows it beside the empty response (before: filed under System).
    assert.equal(categorizeLogEvent({ type: 'provider_content_refusal', model: 'm', round: 7 }), 'llm');
    const empty = summarizeLogEvent({ type: 'llm_empty_response', model: 'm', round: 2, provider_message: 'x' });
    assert.equal(empty.headline, 'Model returned empty response · m');
    assert.equal(empty.body, '');
});

// The owner saw a delegated run's progress as its machine text in a bold headline: an emoji, the run's id and seq, a
// `source:` trailer with a storage path and a 64-character hash. Its typed observation (delegated_activity.js) holds
// the words: what the executor said is the event's words (authored prose, its Markdown read as text in logs.js), and
// who said it (an attempt's id), the run and its source stay in the event's record. A host note keeps its text; only
// the opening its producer marks it with goes (host_notice_text.js).
test('a delegated run\'s progress reads from its typed words; its run, path and hash stay in the record', () => {
    const hash = 'f8d21e8a1ba14dd716312f5f107b074cc293f75f146e5ebe5f512290f8772aba';
    const path = `source_handles/delegated_activity/run-3771bc6bc76d-194-194-${hash}.jsonl`;
    const text = `💬 🛰 delegated run run-3771bc6bc76d @seq 194\nclaude · a01: The execution root matches.\nsource: artifact_store:${path}`;
    const activity = { v: 1, task_id: '87b03e1732aa4687', run_id: 'run-3771bc6bc76d', after_seq: 193, through_seq: 194,
        source: { kind: 'run_events', read_through: 194, events: 1, ref: { kind: 'task_source', root: 'artifact_store', path, size: 900, sha256: hash } },
        parts: [{ kind: 'message', actor: 'claude/a01', seq: 194, text: 'The execution root matches.' }] };
    const evt = { type: 'send_message', task_id: '87b03e1732aa4687', is_progress: true, role: 'system', system_type: 'host_progress',
        narration: false, text, content: text, delegated_activity: activity };
    const view = summarizeLogEvent(evt);
    assert.equal(view.phase, 'progress');
    assert.equal(view.headline, '', 'its words name it: no name repeats them');
    assert.equal(view.body, 'The execution root matches.');
    assert.equal(view.markdown, true);
    assert.deepEqual(view.meta, [], 'no attempt id and no category word among its facts');
    assert.doesNotMatch(JSON.stringify(view), /run-3771bc6bc76d|artifact_store|source:|@seq|🛰|💬|f8d21e8a/);
    assert.equal(categorizeLogEvent(evt, view), 'tasks');
    // Without the typed field the text is the host's, shown as it is; a host note loses only its marked opening.
    const untyped = summarizeLogEvent({ ...evt, delegated_activity: undefined });
    assert.match(untyped.body, /^🛰 delegated run run-3771bc6bc76d/);
    const note = summarizeLogEvent({ type: 'send_message', is_progress: true, role: 'system', system_type: 'host_progress',
        narration: false, text: '💬 🌐 Connection restored after 3 s.' });
    assert.equal(note.body, 'Connection restored after 3 s.');
});

// A row of another type that names a tool keeps its own words: before, a safety check's wait read "run_command result".
// An event the module has no words for is named by its type in words (utils.js machineWords), with the tool it is
// about: before, it read as its machine name alone ("safety_verdict"), its subject gone.
test('a checkpoint or a safety row that names a tool is not read as that tool\'s result', () => {
    const wait = summarizeLogEvent({ type: 'task_checkpoint', checkpoint_kind: 'safety_check_wait', tool: 'run_command', phase: 'ended',
        task_id: 't', detail: 'Safety check ended', content: 'Safety check ended', text: 'Safety check ended' });
    assert.doesNotMatch(wait.headline, /result/);
    const verdict = summarizeLogEvent({ type: 'safety_verdict', tool: 'mcp_superset__call_tool', task_id: 't' });
    assert.equal(verdict.headline, 'Safety verdict · Mcp superset call tool');
    assert.equal(summarizeLogEvent({ type: 'delegate_run_start_requested', task_id: 't' }).headline, 'Delegate run start requested');
    assert.equal(summarizeLogEvent({ type: 'future_worker_crash_recovered', ok: false }).headline, 'Future worker crash recovered');
    // The one shared helper (utils.js): underscores spaced, a capital first letter, nothing invented for an empty name.
    assert.deepEqual(['supervisor_load', 'mcp__call', ' worker_ready ', '', null].map(machineWords),
        ['Supervisor load', 'Mcp call', 'Worker ready', '', '']);
    assert.equal(categorizeLogEvent({ type: 'safety_verdict', tool: 'mcp_superset__call_tool' }), 'tools');
    // An untyped durable tools.jsonl row still reads as its call.
    assert.equal(summarizeLogEvent({ tool: 'read_file', is_error: false, result_preview: 'ok' }).headline, 'Read file result');
});

// The owner's sheets (14.11): "delegate_run_started" carried "gemini-4-carbon-high" on its line; its words ("Delegate run
// started") had filed the model as a fact behind a press. An event about a model or a tool names it as its subject on
// its head, the tool first; the model a fact only beside a tool.
test('an event about a model or a tool names it on its head as its subject, not among its facts', () => {
    const started = summarizeLogEvent({ type: 'delegate_run_started', task_id: 'presence-4c11a22b', model: 'gemini-4-carbon-high',
        route: 'agy', run_id: 'run-747107dcc4de' });
    assert.equal(started.headline, 'Delegate run started · gemini-4-carbon-high');
    assert.ok(!started.meta.some((fact) => fact.includes('gemini')), JSON.stringify(started.meta));
    for (const type of ['llm_usage', 'llm_round', 'llm_round_error']) {
        const view = summarizeLogEvent({ type, model: 'anthropic/claude-opus-5.5', round: 2 });
        assert.match(view.headline, / · claude-opus-5\.5$/, type);
        assert.ok(!view.meta.some((fact) => fact.includes('claude')), type);
    }
    assert.equal(summarizeLogEvent({ type: 'llm_round_started', model: 'anthropic/claude-opus-5.5' }).headline, 'Calling claude-opus-5.5');
    const both = summarizeLogEvent({ type: 'safety_verdict', tool: 'run_command', model: 'openai/gpt-6' });
    assert.match(both.headline, /^Safety verdict · /);
    assert.deepEqual(both.meta, ['gpt-6'], 'beside a tool, the model stays a fact');
    assert.equal(summarizeLogEvent({ type: 'maintenance', phase: 'started' }).headline, 'Maintenance', 'nothing to name: its name alone');
});

// The owner's sheets (14.11): "await_messages failed ⚠ TOOL_ARG_ERROR (await_messages): …", "Safety verdict ·
// run_command", "role=delegated_run". A tool's name in words is said as its event's is (machineWords); a failed call's
// error loses the host's warning mark, which its chip says (host_notice_text.js), and keeps its words; a fact is words.
test('a tool\'s name, a failed call\'s error and an event\'s facts read as words', () => {
    const failed = summarizeLogEvent({ type: 'tool_call', tool: 'await_messages', is_error: true, status: 'argument_error',
        result_preview: '⚠️ TOOL_ARG_ERROR (await_messages): this task has no continuation owner to sleep under' });
    assert.deepEqual([failed.headline, failed.body],
        ['Await messages failed', 'TOOL_ARG_ERROR (await_messages): this task has no continuation owner to sleep under']);
    // A result that did not fail is its tool's own output, kept as written, its mark too.
    assert.equal(summarizeLogEvent({ type: 'tool_call', tool: 'read_file', result_preview: '⚠️ TODO in file' }).body, '⚠️ TODO in file');
    assert.equal(summarizeLogEvent({ type: 'safety_route_fallback', tool: 'run_command' }).headline, 'Safety route fallback · Run command');
    const facts = [
        summarizeLogEvent({ type: 'swarm_fanout', role: 'researcher', requested_model_lane: 'fast', depth: 1, requested_count: 2 }),
        summarizeLogEvent({ type: 'task_message_injected', task_id: 'r', source_task_id: 's', provenance: 'ancestor' }),
        summarizeLogEvent({ type: 'task_cost_finalized', accounted_upper_bound_usd: 1, accounted_upper_bound_usd_with_children: 2 }),
        summarizeLogEvent({ type: 'task_done', status: 'completed', accounted_upper_bound_usd: 1, accounted_upper_bound_usd_with_children: 2 }),
        summarizeLogEvent({ type: 'send_message', is_progress: true, delegation_role: 'subagent', subagent_task_id: 'c',
            subagent_role: 'scout', model: 'm', write_surface: 'workspace', text: 'x' }),
    ].flatMap((view) => view.meta);
    assert.ok(facts.length >= 7, JSON.stringify(facts));
    assert.ok(facts.every((fact) => !/=/.test(fact)), JSON.stringify(facts));
    for (const fact of ['role researcher', 'lane fast', 'depth 1', 'provenance ancestor', 'with children $2.00', 'write surface workspace']) {
        assert.ok(facts.includes(fact), fact);
    }
});
