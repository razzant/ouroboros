import { accountedUpperBound, accountedUpperBoundWithChildren, formatUsd4, joinMarkdownHeadings } from './utils.js';
import { harnessPresentation } from './harness_presentation.js';
import {
    classifyReviewLifecycle,
    classifyReviewLifecyclePointer,
    formatReviewProjection,
} from './review_presentation.js';

const REVIEW_LIFECYCLE_ERROR_STATUSES = new Set([
    'failed', 'interrupted', 'timeout', 'error',
]);

export { formatReviewProjection } from './review_presentation.js';

export const LOG_CATEGORIES = {
    tools: { label: 'Tools', color: 'var(--blue)' },
    llm: { label: 'LLM', color: 'var(--accent)' },
    errors: { label: 'Errors', color: 'var(--red)' },
    tasks: { label: 'Tasks', color: 'var(--amber)' },
    system: { label: 'System', color: 'var(--text-muted)' },
    consciousness: { label: 'Consciousness', color: 'var(--accent)' },
};

export function categorizeLogEvent(evt) {
    const t = evt.type || evt.event || '';
    if (evt.is_progress) {
        return evt.task_id === 'bg-consciousness' ? 'consciousness' : 'tasks';
    }
    if (t.includes('error') || t.includes('crash') || t.includes('fail')) return 'errors';
    if (t.includes('llm') || t.includes('model')) return 'llm';
    if (t.includes('tool') || evt.tool) return 'tools';
    if (t.includes('task') || t.includes('evolution') || t.includes('review')) return 'tasks';
    if (t.includes('consciousness') || t.includes('bg_')) return 'consciousness';
    return 'system';
}

export function normalizeLogTs(isoStr, now = new Date()) {
    if (!isoStr) return '';
    try {
        const d = new Date(isoStr);
        if (Number.isNaN(d.getTime())) return '';
        const time = d.toLocaleTimeString([], { hour12: false });
        if (d.toDateString() === now.toDateString()) return time;
        const date = d.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
        return `${date}, ${time}`;
    } catch {
        return '';
    }
}

function shortText(text, maxLen = 180) {
    const s = String(text || '').replace(/\s+/g, ' ').trim();
    if (!s) return '';
    return s.length > maxLen ? s.slice(0, maxLen - 3) + '...' : s;
}

// For markdown narration, headings are projected (markers off, ` — ` before the
// text under them) BEFORE the newlines collapse into the one-line preview:
// afterwards no line-anchored rule could tell a heading from prose. Typed text
// (shell commands, errors, traces) is never markdown: a `# comment` stays one.
// `full` stays the source text either way.
function describeText(text, maxLen = 180, { markdown = false } = {}) {
    const full = String(text || '').trim();
    if (!full) return { preview: '', full: '' };
    const previewSource = (markdown ? joinMarkdownHeadings(full) : full).replace(/\s+/g, ' ');
    return {
        preview: previewSource.length > maxLen ? previewSource.slice(0, maxLen - 3) + '...' : previewSource,
        full,
    };
}

function subagentId(evt) {
    return String(evt.subagent_task_id || evt.task_id || '').trim();
}

function isSubagentEvent(evt) {
    return String(evt.delegation_role || '').toLowerCase() === 'subagent' || Boolean(evt.subagent_task_id);
}

// E2 (v6.39 UI): compact model name for the subagent label — drop the provider prefix
// ("anthropic/claude-sonnet-4.6" -> "claude-sonnet-4.6") and mark a local route. Shared SSOT
// reused by the chat live-card headline (web/modules/chat.js).
export function compactModel(model = '') {
    const m = String(model || '').trim();
    if (!m) return '';
    // Provider-prefixed IDs use either "provider/model" (OpenRouter) or "provider::model"
    // (direct providers, e.g. openai::gpt-5.5, cloudru::…); show just the model part. Take
    // whatever follows the LAST '/' or '::' separator.
    const slash = m.lastIndexOf('/');
    const dcolon = m.lastIndexOf('::');
    const cut = Math.max(slash >= 0 ? slash + 1 : 0, dcolon >= 0 ? dcolon + 2 : 0);
    const short = m.slice(cut);
    return /local/i.test(m) ? `${short} (local)` : short;
}

// Phase 6, owner directive #1: «бейдж точно нужен, но не рекламный … что ТУТ
// бабл \ субагент на codex». A small chip carrying the harness route this bubble
// or subagent was DISPATCHED to — icon plus the short harness name, in the style
// of the agent account rows on the Agents tab, never a promotional badge.
// Dispatch, not receipt:
// `executor_route` is resolved once when the work is sent, so the chip says where
// it was sent, and a landing below that ask is disclosed on `capability_delta`.
//
// Only a DELEGATED route is a fact worth a chip: the native API path is the
// ordinary case and prints nothing, so the lane never fills with "api" noise on
// every ordinary bubble. Absent fact -> null -> no chip element at all.
// The completion seam's typed substrate claim (subagents.actual_substrate),
// carried on the terminal frame beside the counts. Surfaced as a tooltip
// clause — the counts own the label; the enum never travels bare.
const SUBSTRATE_NOTE = {
    harness_used: 'custody evidence confirms a harness run',
    harness_attempted: 'harness attempted, no delegated run succeeded',
    native_only: 'no harness run recorded',
};

export function executorChip(evt) {
    const route = String(evt?.executor_route || '').trim();
    if (!route) return null;
    // The route id is OPAQUE (`harness` or `harness=model`): print the harness
    // part only, never interpreted beyond splitting the spelling Claudexor uses.
    const harness = route.split('=')[0].trim().toLowerCase();
    if (!harness) return null;
    const name = harnessPresentation(harness).label;
    const base = { harness, label: name };
    // LAYERED TRUTH, label-level. Identity (mark + product name) comes from the
    // harness_presentation SSOT; the run STATE stays on this label. The route is
    // a DISPATCH decision; whether a delegated run actually happened is
    // EVIDENCE, reconciled once at the completion seam
    // (subagents.envelope_from_task -> execution_evidence) and carried on the
    // terminal frame. The chip label always states the run FACT beside the
    // harness name (`{harness} · {state}`) — a bare product name reads as
    // "ran on codex", a receipt nothing may have issued, and the hover-only
    // tooltip is invisible on touch, to AT, and in copies.
    const evidence = (evt && typeof evt.execution_evidence === 'object' && evt.execution_evidence)
        ? evt.execution_evidence : null;
    // The substrate clause is a completion-seam claim coupled to evidence:
    // never attach it to an evidence-less frame (a bare enum beside "dispatched"
    // could contradict the label if a producer ever decoupled them).
    const substrateNote = evidence ? (SUBSTRATE_NOTE[String(evt?.actual_substrate || '')] || '') : '';
    const withSubstrate = (title) => (substrateNote ? `${title} — ${substrateNote}` : title);
    if (!evidence) {
        // Evidence rides TERMINAL frames only, so a live frame proves nothing
        // either way — and under the pre-start charter the leaf usually IS
        // running by now. "dispatched" states the dispatch-plan fact the frame
        // actually carries; an evidence-grade negative ("no run yet") here
        // would be false for most of the live phase.
        return {
            ...base,
            hasEvidence: false,
            label: `${name} · dispatched`,
            title: `Dispatched to ${name} — run evidence arrives with the terminal receipt; this subagent itself runs on the API`,
        };
    }
    const started = Number(evidence.delegated_runs_started || 0);
    const settled = Number(evidence.delegated_runs_settled || 0);
    // Historical frames (v6.94–v6.99) carry delegated_runs_succeeded without
    // delegated_runs_failed: reconstruct the exact complement rather than
    // rendering a clean receipt over an all-failed delegation. Frames with
    // neither counter stay plain, exactly as wide as what they disclosed.
    let failed = Number(evidence.delegated_runs_failed ?? NaN);
    if (!Number.isFinite(failed)) {
        const succeeded = Number(evidence.delegated_runs_succeeded ?? NaN);
        failed = Number.isFinite(succeeded) ? Math.max(0, settled - succeeded) : 0;
    }
    failed = Math.max(0, failed);
    if (evidence.evidence_read_failed) {
        // The custody log EXISTS but could not be (fully) read: the counts are
        // UNKNOWN, not an established fact (sol finding, b49f8192 wave). This
        // holds past recorded starts too — the partial work-order replay sets
        // the flag with started>0, and a confident settled/spend receipt over
        // admittedly incomplete evidence would be a lie. No substrate clause:
        // the seam never claims a substrate over unreadable evidence.
        return {
            ...base,
            hasEvidence: true,
            label: `${name} · evidence unavailable`,
            title: started
                ? `The ${name} route was assigned and at least ${started} delegated run(s) started, but the evidence could not be fully read — final counts are unknown`
                : `The ${name} route was assigned, but the delegated-run evidence could not be read — whether a run happened is unknown, not "none"`,
        };
    }
    if (!started) {
        return {
            ...base,
            hasEvidence: true,
            label: `${name} · no run yet`,
            title: withSubstrate(`The ${name} route was assigned, but there is no durable record of a delegated run for this subagent`),
        };
    }
    if (!settled) {
        // Evidence is terminal-frame material: started-but-unsettled here means
        // the run(s) never settled (orphaned or lost), not "still executing" —
        // a present-tense "running" on a finished card would be a lie.
        return {
            ...base,
            hasEvidence: true,
            label: `${name} · ${started} started, none settled`,
            title: withSubstrate(`Delegated to your ${name} account — ${started} run(s) started, none settled`),
        };
    }
    const cost = evidence.subscription_cost_usd;
    const approx = evidence.subscription_cost_estimated ? '~' : '';
    const costPart = (cost === null || cost === undefined)
        ? 'subscription spend undisclosed'
        : `${approx}$${Number(cost).toFixed(2)} subscription`;
    const runsPart = `${settled} run${settled === 1 ? '' : 's'}`;
    // The owner dictionary is "N ok, M failed" (plan D9): ok = settled − failed
    // when either counter is disclosed; a frame with neither counter renders
    // plain "N runs", exactly as wide as what it disclosed. All-failed runs
    // must never read as a clean receipt.
    const counted = Number.isFinite(Number(evidence.delegated_runs_failed ?? NaN))
        || Number.isFinite(Number(evidence.delegated_runs_succeeded ?? NaN));
    const ok = Math.max(0, settled - failed);
    const okPart = `${ok} ok${failed ? `, ${failed} failed` : ''}`;
    // Unverified work-order coverage is an honesty fact, not a style note:
    // those runs were discounted from the ok-count, and the label must say
    // why a "successful" delegation reads short. Access rides the tooltip —
    // requested surface from the frame, applied access from the settlement
    // receipts when the engine disclosed it (empty = predates the receipt).
    const unresolved = Number(evidence.delegated_runs_source_unresolved || 0);
    const unresolvedPart = unresolved
        ? `, ${unresolved} unverified`
        : '';
    const accessClauses = [];
    const surface = String(evt?.write_surface || '').trim();
    if (surface) accessClauses.push(`write surface ${surface}`);
    const applied = Array.isArray(evidence.applied_access_profiles)
        ? evidence.applied_access_profiles.filter(Boolean) : [];
    if (applied.length) accessClauses.push(`access applied: ${applied.join(', ')}`);
    const accessPart = accessClauses.length ? ` — ${accessClauses.join('; ')}` : '';
    const unresolvedTitle = unresolved
        ? ` — ${unresolved} run(s) settled with unverified work-order coverage (not counted ok)`
        : '';
    return {
        ...base,
        hasEvidence: true,
        label: counted ? `${name} · ${okPart}${unresolvedPart}` : `${name} · ${runsPart}${unresolvedPart}`,
        title: withSubstrate(`Delegated to your ${name} account — ${runsPart} settled${counted ? ` (${okPart})` : ''}, ${costPart}`) + unresolvedTitle + accessPart,
    };
}

// The child card's headline is its identity, not its status: `role · model`
// (or `Subagent · model` when the role is unknown). The status lives in the
// card's chip, so no ` — Done` suffix, and the short task id is never part of
// the compact form — chat.js appends it for twins at render time. Logs keep
// the full diagnostic form (`role · model (id) — status`).
function subagentHeadline(sid = '', role = '', label = '', model = '', { full = false } = {}) {
    const shortId = String(sid || '').slice(0, 8);
    const cleanRole = String(role || '').trim() || 'Subagent';
    const suffix = full && label ? ` — ${label}` : '';
    // Show the resolved model compactly NEXT TO the role (e.g. "planning-scout · gemini-3.5-flash").
    const modelPart = compactModel(model) ? ` · ${compactModel(model)}` : '';
    return `${cleanRole}${modelPart}${shortId && full ? ` (${shortId})` : ''}${suffix}`;
}

const SUBAGENT_CARD_LABEL = {
    scheduled: 'Working',
    running: 'Working',
    interrupted: 'Working',
};

export function formatLogMoney(value) {
    return formatUsd4(value);
}

export function formatLogDuration(sec) {
    const num = Number(sec);
    if (!Number.isFinite(num) || num < 0) return '';
    if (num >= 60) {
        const mins = Math.floor(num / 60);
        const rem = Math.round(num % 60);
        return `${mins}m ${rem}s`;
    }
    return `${num < 10 ? num.toFixed(1) : Math.round(num)}s`;
}

function formatLogTokens(evt) {
    const prompt = Number(evt.prompt_tokens || 0);
    const completion = Number(evt.completion_tokens || 0);
    if (!prompt && !completion) return '';
    return `${prompt}\u2192${completion} tok`;
}

function compactJson(value, maxLen = 220) {
    if (value == null) return '';
    let txt = '';
    try {
        txt = JSON.stringify(value);
    } catch {
        txt = String(value);
    }
    return shortText(txt, maxLen);
}

function extractCommandText(args) {
    if (!args || typeof args !== 'object') return '';
    const cmd = args.cmd;
    if (Array.isArray(cmd)) {
        return cmd.map((part) => String(part || '').trim()).filter(Boolean).join(' ');
    }
    if (typeof cmd === 'string') return cmd;
    return '';
}

function describeStartupChecks(checks) {
    if (!checks || typeof checks !== 'object') return '';
    const parts = [];
    for (const [key, value] of Object.entries(checks)) {
        if (value && typeof value === 'object' && value.status) {
            parts.push(`${key}:${value.status}`);
        }
    }
    return shortText(parts.join(' | '), 240);
}

// Typed pending-cancel projection (phase A cancel redesign): a durable cancel
// intent is open and the supervisor teardown has not settled yet. This is NOT a
// terminal severity — the status stays running/scheduled and the record carries
// cancel_state="pending"; the card shows an interim "Cancelling…" and resolves
// on the settled task_done (Cancelled, or Completed when the run finished first).
export function taskCancelPending(record) {
    const status = String(record?.status || '').toLowerCase();
    const settled = ['completed', 'failed', 'cancelled', 'rejected_duplicate'].includes(status);
    return !settled && String(record?.cancel_state || '') === 'pending';
}

// S3 (Q1/Q2): the pending soft stop — same typed pending projection, but the
// durable intent's policy is finalize_then_cancel, so the card honestly shows
// "Finalizing…" (a bounded final turn is running) instead of "Cancelling…".
export function taskSoftStopPending(record) {
    return taskCancelPending(record) && String(record?.stop_policy || '') === 'finalize_then_cancel';
}

// S3 (owner decision №8/Q3): an owner-requested finalization is a SUCCESSFUL
// soft stop, not a warning — the owner asked for the summary and received the
// best available result. The factual task headline remains "Done" while the
// task details carry this owner-stop marker (spec §17).
export const OWNER_STOP_DETAIL_MARKER = "summary at the owner's request — best available result";

export function taskStoppedWithSummary(evt) {
    return String(evt?.reason_code || '') === 'owner_requested_finalization';
}

// The typed degradation causes a card can state in the owner's words. The record
// keeps the machine code (Logs, task detail, benchmark ledgers); only the card
// speaks. An UNKNOWN code stays raw on purpose: a reason we have no sentence for
// must read as itself rather than as a wrong sentence.
const TASK_REASON_PHRASES = {
    plan_review_advisory: 'plan review never closed; the work continued under advisory enforcement',
    host_child_status_suffix: 'a child task had not settled when the answer was delivered',
    invalid_delivery_control_after_repair: 'the delivery control object was still malformed after repair',
    budget_exhausted: 'the task ran out of budget before it could finish cleanly',
    delivery_control_degraded: 'delivery finished in a degraded control state',
};

export function taskReasonPhrase(code) {
    const raw = String(code || '');
    return TASK_REASON_PHRASES[raw] || raw;
}

export function taskReasonDetail(evt) {
    // An owner-requested stop is a success and carries its own marker instead.
    if (taskStoppedWithSummary(evt) || !evt?.reason_code) return '';
    return `Reason: ${taskReasonPhrase(evt.reason_code)}`;
}

// S3 (HQ1): the ONE shared projection of a typed owner_hurry event for the
// task-detail/card surfaces. Never a chat message: chat.js renders only a
// compact task-card status from this, and the timeline summarizer hides the
// family (visible=false).
export function ownerHurryProjection(evt) {
    const phase = String(evt?.phase || '');
    return {
        taskId: String(evt?.task_id || ''),
        phase,
        applied: phase === 'applied',
        label: phase === 'applied' ? 'Owner hurry applied'
            : phase === 'requested' ? 'Owner hurry requested'
                : `Owner hurry ${phase || 'event'}`,
    };
}

function normalizeTaskTerminalRecord(evt) {
    if (!evt || typeof evt !== 'object') return evt || {};
    const terminalStatus = String(evt.task_terminal_status || '').trim();
    return terminalStatus ? { ...evt, status: terminalStatus } : evt;
}

export function taskOutcomeSeverity(evt) {
    const record = normalizeTaskTerminalRecord(evt);
    const lifecycle = String(record.outcome_axes?.lifecycle?.status || record.status || '').toLowerCase();
    // v6.82 (P5): a cancelled task is neither Done nor Failed — it is honestly
    // Cancelled. Checked first: forced teardown routinely leaves failure-shaped
    // side facts (e.g. artifacts missing on a cancelled workspace task) that must
    // not relabel an owner-requested cancellation as a failure.
    // 'cancel_requested' as a STATUS is legacy replay only (phase A moved cancel
    // intent to the durable cancel_state projection); old task_done frames and
    // pre-redesign history rows keep resolving as Cancelled.
    if (lifecycle === 'cancelled' || lifecycle === 'cancel_requested') {
        return 'cancelled';
    }
    const execution = String(record.outcome_axes?.execution?.status || '').toLowerCase();
    const objective = String(record.outcome_axes?.objective?.status || '').toLowerCase();
    const review = String(record.outcome_axes?.review?.status || record.review_status?.status || '').toLowerCase();
    const artifacts = String(record.outcome_axes?.artifacts?.status || record.artifact_bundle?.status || record.artifact_status || '').toLowerCase();
    const artifactStatus = String(record.artifact_bundle?.status || record.artifact_status || '').toLowerCase();
    if (
        lifecycle === 'failed'
        || ['failed', 'infra_failed'].includes(execution)
        || objective === 'fail'
        || review === 'fail'
        || ['failed', 'missing'].includes(artifacts)
        || artifactStatus === 'failed'
    ) {
        return 'error';
    }
    // Owner-requested finalization is a best_effort SUCCESS (№8/Q3): the owner
    // asked for the stop, so it must not read as "Finished with warnings".
    if (taskStoppedWithSummary(evt)) {
        return 'done';
    }
    if (
        lifecycle === 'rejected_duplicate'
        || ['degraded', 'best_effort'].includes(execution)
        || ['degraded', 'best_effort'].includes(objective)
        || review === 'degraded'
        || Boolean(record.outcome_axes?.objective?.warning)
    ) {
        return 'warn';
    }
    return 'done';
}

// v6.82 (P5): one shared severity→card-phase mapping for terminal task frames,
// so live task_done, history task_summary rows, and the terminal-status replay
// fallback all resolve a cancelled root to the same honest 'cancelled' phase.
export function taskTerminalPhase(evt) {
    const severity = taskOutcomeSeverity(evt);
    if (severity === 'cancelled') return 'cancelled';
    if (severity === 'error') return 'error';
    if (severity === 'warn') return 'warn';
    return 'done';
}

// Durable task detail is allowed to finish a card only at one of the task
// result store's genuinely-settled statuses. In particular, interrupted and
// the legacy cancel_requested latch remain retryable rather than becoming a
// fabricated Done/Cancelled projection.
const TERMINAL_TASK_DETAIL_STATUSES = new Set([
    'completed', 'failed', 'cancelled', 'rejected_duplicate',
]);
const OPEN_POST_TASK_SYNTHESIS_STATUSES = new Set(['pending_once', 'running']);

export function isTerminalTaskDetail(record) {
    const status = String(record?.status || '').toLowerCase();
    const synthesis = String(record?.root_phase_checkpoint?.post_task_synthesis || '').toLowerCase();
    return TERMINAL_TASK_DETAIL_STATUSES.has(status)
        && !(status === 'completed' && OPEN_POST_TASK_SYNTHESIS_STATUSES.has(synthesis));
}

// A task_done normally mirrors durable task detail. Keep the detail predicate
// as the terminality authority; the two aliases below exist only on old event
// frames, not in durable detail (`done`, and the pre-cancel-redesign settled
// `cancel_requested` event spelling).
export function taskDoneIsTerminal(evt) {
    const record = normalizeTaskTerminalRecord(evt);
    const status = String(record?.status || '').toLowerCase();
    return isTerminalTaskDetail(record) || status === 'done' || status === 'cancel_requested';
}

// One factual phase -> presentation vocabulary for task chips and terminal
// headlines. Technical outcome and terminality truth stay in their existing
// reducers; this translator never inspects event payloads or infers completion.
export function taskPresentation(phase = 'working') {
    const normalizedPhase = typeof phase === 'string' && phase.trim() ? phase.trim() : 'working';
    const headline = normalizedPhase === 'done' ? 'Done'
        : normalizedPhase === 'warn' ? 'Done with warnings'
            : normalizedPhase === 'cancelled' ? 'Cancelled'
                : ['error', 'timeout', 'lifecycle_error'].includes(normalizedPhase) ? 'Failed'
                    : 'Working';
    return { phase: normalizedPhase, headline };
}

function taskOutcomeMeta(evt) {
    const axes = evt.outcome_axes || {};
    return [
        axes.lifecycle?.status ? `lifecycle ${axes.lifecycle.status}` : '',
        axes.execution?.status ? `execution ${axes.execution.status}` : '',
        axes.objective?.status ? `objective ${axes.objective.status}` : '',
    ].filter(Boolean);
}

export function summarizeLogEvent(evt) {
    const t = evt.type || evt.event || 'unknown';
    const view = (phase, headline, { body = '', meta = [], typeLabel = t } = {}) => ({
        typeLabel,
        phase,
        headline,
        body,
        meta: meta.filter(Boolean),
    });
    const taskMeta = (...items) => [evt.task_id ? `task=${evt.task_id}` : '', ...items];

    if (evt.is_progress || t === 'send_message') {
        if (isSubagentEvent(evt)) {
            const sid = subagentId(evt);
            const event = String(evt.subagent_event || 'update').toLowerCase();
            const role = String(evt.subagent_role || '').trim();
            return view(event === 'completed' ? 'done' : event === 'failed' || event === 'rejected' ? 'warn' : 'progress', subagentHeadline(sid, role, event, evt.model, { full: true }), {
                body: shortText(String(evt.content || evt.text || '').replace(/^💬\s*/, ''), 240),
                meta: [
                    sid ? `task=${sid}` : '',
                    role ? `role=${role}` : '',
                    evt.model ? `model=${evt.model}` : '',
                    evt.write_surface ? `write=${evt.write_surface}` : '',
                    evt.parent_task_id ? `parent=${evt.parent_task_id}` : '',
                    evt.root_task_id ? `root=${evt.root_task_id}` : '',
                ],
            });
        }
        return view(
            evt.task_id === 'bg-consciousness' ? 'thought' : 'progress',
            shortText(String(evt.content || evt.text || '').replace(/^💬\s*/, ''), 240) || 'Progress update',
            { meta: [evt.task_id === 'bg-consciousness' ? 'background' : 'task'] },
        );
    }

    if (t === 'task_started') {
        return view('start', `Started ${evt.task_type || 'task'}`, {
            body: shortText(evt.task_text, 220),
            meta: taskMeta(evt.direct_chat ? 'chat' : 'queued'),
        });
    }

    if (t === 'task_received') {
        const task = evt.task || {};
        return view('queued', `Received ${task.type || 'task'}`, {
            body: shortText(task.text, 220),
            meta: [task.id ? `task=${task.id}` : '', task.text_len ? `${task.text_len} chars` : ''],
        });
    }

    if (t === 'context_building_started') {
        return view('context', 'Building context', { meta: taskMeta(evt.task_type || '') });
    }

    if (t === 'context_building_finished') {
        return view('ready', 'Context ready', {
            meta: taskMeta(
                evt.message_count != null ? `${evt.message_count} msgs` : '',
                Number.isFinite(Number(evt.budget_remaining_usd)) ? `$${Number(evt.budget_remaining_usd).toFixed(2)} left` : '',
            ),
        });
    }

    if (t === 'task_heartbeat') {
        return view(evt.phase || 'alive', 'Still working', {
            meta: taskMeta(evt.task_type || '', formatLogDuration(evt.runtime_sec)),
        });
    }

    if (t === 'llm_round_started') {
        return view('calling', `Calling ${evt.model || 'model'}`, {
            meta: taskMeta(
                evt.round ? `r${evt.round}` : '',
                evt.attempt ? `try ${evt.attempt}` : '',
                evt.reasoning_effort || '',
                evt.use_local ? 'local' : '',
            ),
        });
    }

    if (t === 'llm_round_finished' || t === 'llm_round') {
        return view('done', `LLM round ${evt.round || ''} finished`.trim(), {
            meta: taskMeta(
                evt.model || '',
                formatLogTokens(evt),
                formatLogMoney(evt.cost_usd ?? evt.cost),
                evt.response_kind === 'tool_calls' ? `${evt.tool_call_count || 0} tool calls` : evt.response_kind || '',
            ),
        });
    }

    if (t === 'llm_round_empty' || t === 'llm_empty_response') {
        return view('empty', 'Model returned empty response', {
            meta: taskMeta(evt.model || '', evt.round ? `r${evt.round}` : ''),
        });
    }

    if (t === 'llm_round_error' || t === 'llm_api_error') {
        return view('error', 'LLM call failed', {
            body: shortText(evt.error, 260),
            meta: taskMeta(evt.model || '', evt.round ? `r${evt.round}` : ''),
        });
    }

    if (t === 'llm_usage') {
        return view('usage', 'LLM usage recorded', {
            meta: taskMeta(
                evt.model || '',
                formatLogTokens(evt),
                formatLogMoney(evt.cost_usd ?? evt.cost),
                evt.category || '',
            ),
        });
    }

    if (t === 'tool_call_started') {
        return view('start', `Running ${evt.tool || 'tool'}`, {
            body: compactJson(evt.args, 260),
            meta: taskMeta(evt.timeout_sec ? `timeout ${evt.timeout_sec}s` : ''),
        });
    }

    if (t === 'tool_call_finished') {
        // A child killed by a signal (typed signal name / negative exit code)
        // is a failure even when the handler rendered a normal result (T11).
        const signalDeath = Boolean(evt.signal) || (typeof evt.exit_code === 'number' && evt.exit_code < 0);
        const isError = Boolean(evt.is_error) || signalDeath;
        const label = signalDeath ? `killed (${evt.signal || evt.exit_code})`
            : evt.is_error ? 'failed'
            : 'finished';
        return view(isError ? 'error' : 'done', `${evt.tool || 'tool'} ${label}`, {
            body: shortText(evt.result_preview, 260),
            meta: taskMeta(formatLogDuration(evt.duration_sec)),
        });
    }

    if (t === 'tool_call_timeout' || t === 'tool_timeout') {
        return view('timeout', `${evt.tool || 'tool'} timed out`, {
            body: compactJson(evt.args, 220),
            meta: taskMeta(evt.timeout_sec ? `limit ${evt.timeout_sec}s` : '', formatLogDuration(evt.duration_sec)),
        });
    }

    if (t === 'tool_call' || evt.tool) {
        return view('result', `${evt.tool || 'tool'} result`, {
            body: shortText(evt.result_preview || compactJson(evt.args, 220), 260),
            meta: taskMeta(),
        });
    }

    if (t === 'owner_hurry') {
        // S3 (HQ1): the typed non-chat control family. The LOGS tab is a
        // diagnostic surface, so the row renders here; chat stays silent (see
        // the explicit visible=false branch in summarizeChatLiveEvent).
        const proj = ownerHurryProjection(evt);
        return view('info', proj.label, {
            body: shortText(evt.detail, 220),
            meta: taskMeta(
                evt.request_id ? `request=${evt.request_id}` : '',
                evt.attempt_key != null ? `attempt=${evt.attempt_key}` : '',
                evt.effect ? `effect=${evt.effect}` : '',
                evt.status ? `status=${evt.status}` : '',
            ),
        });
    }

    if (t === 'task_metrics_event' || t === 'task_eval') {
        return view('metrics', 'Task metrics', {
            meta: taskMeta(
                evt.task_type || '',
                ...taskOutcomeMeta(evt),
                evt.reason_code || '',
                formatLogDuration(evt.duration_sec),
                evt.tool_calls != null ? `${evt.tool_calls} tools` : '',
                evt.tool_errors ? `${evt.tool_errors} errors` : '',
                evt.response_len ? `${evt.response_len} chars` : '',
            ),
        });
    }

    if (t === 'task_done') {
        const terminal = taskDoneIsTerminal(evt);
        const presentation = taskPresentation(terminal ? taskTerminalPhase(evt) : 'working');
        const reasonCode = evt.reason_code ? String(evt.reason_code) : '';
        const artifactStatus = evt.artifact_bundle?.status || evt.artifact_status || '';
        const reviewDetails = formatReviewProjection(evt.review_projection);
        const unavailable = evt.cost_accounting_status === 'unavailable';
        // C13: the SHARED accessor and its null policy — same alias precedence as
        // chat.js and the Python seams, and a REAL $0 prints instead of vanishing.
        const ownValue = accountedUpperBound(evt) ?? (evt.cost ?? null);
        const ownCost = unavailable
            ? 'cost unavailable'
            : (ownValue != null ? `${formatLogMoney(ownValue)}${evt.cost_final === false ? ' (pending)' : ''}` : '');
        return view(presentation.phase, presentation.headline, {
            body: reviewDetails,
            meta: taskMeta(
                ...taskOutcomeMeta(evt),
                // №8/Q3: the owner-requested soft stop shows the honest marker
                // instead of the raw machine reason code.
                taskStoppedWithSummary(evt) ? OWNER_STOP_DETAIL_MARKER : reasonCode,
                artifactStatus ? `artifacts ${artifactStatus}` : '',
                ownCost,
                // v6.57.0 (P6b): show the recursive cost incl. children when it adds up to
                // more than this task's own spend, so a parent isn't under-reported.
                (accountedUpperBoundWithChildren(evt) ?? -1) > (ownValue ?? 0)
                    ? `+children=${formatLogMoney(accountedUpperBoundWithChildren(evt))}${evt.cost_with_children_partial ? ' (partial)' : ''}`
                    : '',
                evt.total_rounds ? `${evt.total_rounds} rounds` : '',
                formatLogTokens(evt),
            ),
        });
    }

    if (t === 'task_cost_finalized') {
        const unavailable = evt.cost_accounting_status === 'unavailable';
        const ownCost = unavailable ? 'cost unavailable' : formatLogMoney(accountedUpperBound(evt));
        const subtreeCost = unavailable ? '' : formatLogMoney(accountedUpperBoundWithChildren(evt));
        return view(unavailable ? 'warn' : 'metrics', 'Task cost finalized', {
            meta: taskMeta(ownCost, subtreeCost ? `subtree=${subtreeCost}` : '', evt.post_task_status || ''),
        });
    }

    if (t === 'startup_verification') {
        return view(Number(evt.issues_count || 0) > 0 ? 'warn' : 'ok', 'Startup verification', {
            body: describeStartupChecks(evt.checks),
            meta: [evt.git_sha ? String(evt.git_sha).slice(0, 8) : '', `${evt.issues_count || 0} issues`],
        });
    }

    if (t === 'worker_spawn_start') {
        return view('start', `Spawning ${evt.count || '?'} workers`, { meta: [evt.start_method || ''] });
    }

    if (t === 'worker_sha_verify') {
        return view(evt.ok ? 'ok' : 'warn', evt.ok ? 'Worker SHA verified' : 'Worker SHA mismatch', {
            meta: [
                evt.expected_sha ? `exp ${String(evt.expected_sha).slice(0, 8)}` : '',
                evt.observed_sha ? `got ${String(evt.observed_sha).slice(0, 8)}` : '',
                evt.worker_pid ? `pid ${evt.worker_pid}` : '',
            ],
        });
    }

    if (t === 'worker_boot') {
        return view('boot', 'Worker booted', {
            meta: [evt.pid ? `pid ${evt.pid}` : '', evt.git_sha ? String(evt.git_sha).slice(0, 8) : ''],
        });
    }

    if (t === 'deps_sync_ok') {
        return view('ok', 'Dependencies in sync', { meta: [evt.reason || '', shortText(evt.source, 60)] });
    }

    if (t === 'reset_unsynced_rescued_then_reset') {
        return view('warn', 'Recovered dirty worktree before restart', {
            meta: [
                evt.reason || '',
                evt.dirty_count != null ? `${evt.dirty_count} dirty` : '',
                evt.unpushed_count != null ? `${evt.unpushed_count} unpushed` : '',
            ],
        });
    }

    if (t === 'task_checkpoint') {
        if (evt.checkpoint_kind === 'context_fit_low_retry') {
            return view('warn', 'Context rebuilt in Low mode', {
                meta: taskMeta(
                    evt.model ? compactModel(evt.model) : '',
                    evt.round ? `r${evt.round}` : '',
                    'same-model retry',
                ),
            });
        }
        const cpNum = evt.checkpoint_number || Math.floor((evt.round || 0) / 15);
        return view('thinking', `Checkpoint ${cpNum}`, {
            meta: taskMeta(
                evt.round ? `r${evt.round}` : '',
                evt.context_tokens ? `~${evt.context_tokens} tok` : '',
                formatLogMoney(evt.task_cost),
            ),
        });
    }

    if (t.includes('error') || t.includes('crash') || t.includes('fail')) {
        return view('error', t, {
            body: shortText(evt.error || evt.result_preview || evt.text || '', 260),
            meta: taskMeta(evt.tool ? `tool=${evt.tool}` : ''),
        });
    }

    if (t === 'swarm_fanout') {
        const n = (evt.requested_count != null)
            ? evt.requested_count
            : (Array.isArray(evt.task_ids) ? evt.task_ids.length : 0);
        return view('info', `swarm fan-out: ${n} subagent(s) requested`, {
            meta: [
                evt.task_group_id ? `group=${evt.task_group_id}` : '',
                evt.role ? `role=${evt.role}` : '',
                evt.requested_model_lane ? `lane=${evt.requested_model_lane}` : '',
                evt.depth != null ? `depth=${evt.depth}` : '',
                evt.inter_wave_latency_sec != null ? `Δ=${evt.inter_wave_latency_sec}s` : '',
            ],
        });
    }

    return view('info', shortText(t, 120), {
        body: shortText(evt.text || evt.error || evt.result_preview || compactJson(evt.args || evt.task || evt.checks, 260), 260),
        meta: taskMeta(evt.model || '', formatLogMoney(evt.cost_usd ?? evt.cost)),
    });
}

function chatView({
    phase = 'working',
    headline = 'Working...',
    body = '',
    fullBody = '',
    fullHeadline = '',
    activityPreview,
    visible = false,
    promote = false,
    terminal = false,
    human = false,
    dedupeKey = '',
    meta = [],
    fullRef = '',
    truncated = false,
    chip = null,
} = {}) {
    const out = {
        phase,
        headline,
        body,
        visible,
        promote,
        terminal,
        human,
        dedupeKey,
    };
    if (fullBody) out.fullBody = fullBody;
    if (fullHeadline) out.fullHeadline = fullHeadline;
    // Explicit emptiness is part of the presentation contract: a review-only
    // frame has no activity and must not fall back to its disclosure body.
    if (activityPreview !== undefined) out.activityPreview = String(activityPreview || '');
    if (Array.isArray(meta) && meta.length) out.meta = meta.filter(Boolean);
    // P3 uniform contract: when the WS body was truncated server-side, carry a
    // fetch ref (a task id -> GET /api/tasks/{id}) so the bubble can load the
    // genuinely-full output on demand instead of showing only the capped preview.
    if (fullRef) out.fullRef = String(fullRef);
    if (truncated) out.truncated = true;
    // Phase 6: the executor chip rides the projection so live and replay routes
    // paint the same fact; absent stays absent (no placeholder chip).
    if (chip) out.executorChip = chip;
    return out;
}

export function summarizeChatLiveEvent(evt) {
    const t = evt.type || evt.event || 'unknown';
    const groupId = getLogTaskGroupId(evt);
    const progressText = describeText(String(evt.content || evt.text || '').replace(/^💬\s*/, ''), 240, { markdown: true });
    const key = (...parts) => [t, groupId, ...parts].join(':');

    if (t === 'owner_hurry') {
        // S3 (HQ1) EXPLICIT hide branch: the typed hurry control family never
        // renders a chat timeline row or bubble — chat.js paints only a compact
        // card status from ownerHurryProjection, and the durable facts live in
        // the task detail. Explicit (not the fallthrough) so a future default
        // change cannot silently surface the family in chat.
        return chatView({ visible: false, dedupeKey: key(evt.phase || '', evt.request_id || '') });
    }

    if (evt.lifecycle && typeof evt.lifecycle === 'object') {
        const lifecycle = evt.lifecycle;
        const status = String(lifecycle.status || '').toLowerCase();
        const stale = Boolean(lifecycle.stale);
        const phase = status === 'succeeded' ? 'done'
            : status === 'cancelled' ? 'cancelled'
                : REVIEW_LIFECYCLE_ERROR_STATUSES.has(status) ? 'lifecycle_error'
                    : stale ? 'warn'
                        : 'working';
        const label = lifecycle.phase || status || 'working';
        const target = lifecycle.target ? `\`${lifecycle.target}\`` : 'skill';
        const headline = progressText.preview || `Skill ${lifecycle.kind || 'operation'}: ${target} — ${label}`;
        const body = stale
            ? (lifecycle.recovery_hint || 'Lifecycle work is still running; restart may be required.')
            : (lifecycle.error || lifecycle.message || '');
        return chatView({
            phase,
            headline,
            body: shortText(body, 220),
            fullHeadline: progressText.full || headline,
            fullBody: body,
            activityPreview: progressText.preview || shortText(headline, 240),
            visible: true,
            promote: true,
            terminal: ['done', 'lifecycle_error', 'cancelled'].includes(phase),
            human: true,
            dedupeKey: lifecycle.id ? `lifecycle:${lifecycle.id}:${status}:${label}:${stale ? 'stale' : 'fresh'}` : key(status, label),
        });
    }

    if ((evt.is_progress || t === 'send_message') && isSubagentEvent(evt)) {
        const sid = subagentId(evt);
        const rawEvent = String(evt.subagent_event || '').toLowerCase();
        const role = String(evt.subagent_role || '').trim();
        const status = String(evt.status || '').trim();
        const resultText = describeText(evt.result || '', 320, { markdown: true });
        const traceText = describeText(evt.trace_summary || '', 320);
        const errorText = describeText(evt.error || '', 220);
        const reasonDetail = evt.reason_code ? `Reason: ${taskReasonPhrase(evt.reason_code)}` : '';
        const detailParts = [
            progressText.full,
            resultText.full ? `[RESULT]\n${resultText.full}` : '',
            traceText.full ? `[TRACE]\n${traceText.full}` : '',
            errorText.full ? `[ERROR]\n${errorText.full}` : '',
            reasonDetail,
        ].filter(Boolean);
        // A generic "completed" event still carries authoritative outcome axes.
        // Normalize it once here so every live/replay route gets the same label,
        // phase and terminal truth from the canonical projector.
        const completionSeverity = rawEvent === 'completed' ? taskOutcomeSeverity(evt) : 'done';
        const event = rawEvent === 'completed'
            ? (completionSeverity === 'cancelled' ? 'cancelled'
                : completionSeverity === 'error' ? 'failed'
                    : completionSeverity === 'warn' ? 'completed_warn'
                        : 'completed')
            : rawEvent;
        // Cancelled is distinct; rejected/interrupted remain notices rather than
        // red failures. Interrupted is retryable and therefore non-terminal.
        const phase = event === 'completed' ? 'done'
            : event === 'completed_warn' ? 'warn'
                : event === 'cancelled' ? 'cancelled'
                    : event === 'failed' ? 'error'
                        : ['rejected', 'interrupted'].includes(event) ? 'warn'
                            : event === 'scheduled' ? 'start'
                                : 'working';
        const terminal = ['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(event);
        const label = terminal
            ? taskPresentation(phase).headline
            : (SUBAGENT_CARD_LABEL[event] || 'Working');
        // The compact activity line describes the child's work/result; review
        // evidence is rendered separately by the owning card's Reviews section.
        const activity = terminal
            ? (phase === 'error' && errorText.full ? errorText
                : resultText.full ? resultText
                    : errorText.full ? errorText
                        : traceText.full ? traceText
                            : progressText)
            : (progressText.full ? progressText
                : resultText.full ? resultText
                    : errorText.full ? errorText
                        : traceText);
        return chatView({
            phase,
            headline: subagentHeadline(sid, role, label, evt.model),
            body: activity.preview || '',
            fullBody: detailParts.join('\n\n'),
            activityPreview: activity.preview || '',
            visible: true,
            promote: true,
            human: true,
            terminal,
            // P3: the WS result/trace were capped at 4000 server-side; expose the
            // subagent task id so "show full" can fetch the genuinely-full output.
            fullRef: sid,
            truncated: Boolean(evt.result_truncated || evt.trace_summary_truncated),
            meta: [
                evt.write_surface ? `write=${evt.write_surface}` : '',
                status ? `status=${status}` : '',
            ],
            // «ТУТ … субагент на codex» — the child's own executor chip.
            chip: executorChip(evt),
            dedupeKey: `subagent:${sid}:${label}:${status}:${progressText.full || resultText.full || errorText.full || ''}`,
        });
    }

    if (evt.is_progress || t === 'send_message') {
        const lifecycleTerminal = String(evt.task_id || '').startsWith('skill_lifecycle_')
            && /\s—\s(completed|failed)\b/i.test(progressText.full);
        // Background consciousness has no task_result; the backend signals end-of-cycle
        // with a structured `consciousness_state` marker (and history replay annotates
        // the latest entry with `task_terminal_status`). Both are structured, not text.
        const bgConsciousness = evt.task_id === 'bg-consciousness';
        const bgState = String(evt.consciousness_state || '');
        const bgErrored = bgState === 'error_backoff' || bgState === 'error';
        const bgTerminal = bgConsciousness
            && (Boolean(bgState) || Boolean(evt.task_terminal_status));
        const bgPhase = bgTerminal ? (bgErrored ? 'lifecycle_error' : 'done') : 'thinking';
        return chatView({
            phase: bgConsciousness
                ? bgPhase
                : (lifecycleTerminal ? (/failed\b/i.test(progressText.full) ? 'lifecycle_error' : 'done') : 'working'),
            // The bg end-of-cycle marker carries no text; pass an empty headline so
            // the card keeps its last thought as the title instead of "Working...".
            headline: (bgTerminal && !progressText.preview) ? '' : (progressText.preview || 'Working...'),
            fullHeadline: progressText.full || '',
            activityPreview: progressText.preview || '',
            visible: Boolean(progressText.preview),
            promote: true,
            human: true,
            // «ТУТ бабл … на codex» — an ordinary progress bubble carries the chip
            // too whenever the frame disclosed a delegated executor.
            chip: executorChip(evt),
            dedupeKey: progressText.full ? `progress:${progressText.full}` : `progress:${evt.task_id || ''}`,
        });
    }

    if (t === 'task_started' || t === 'task_received') {
        return chatView({ headline: 'Working on it', promote: true, dedupeKey: key() });
    }

    if (t === 'context_building_started') {
        return chatView({ headline: 'Getting ready', promote: true, dedupeKey: key() });
    }

    if (t === 'context_building_finished') {
        return chatView({ headline: 'Looking through the context', dedupeKey: key() });
    }

    if (t === 'task_heartbeat') {
        return chatView({ headline: 'Still working', dedupeKey: key(evt.phase || '') });
    }

    if (t === 'llm_round_started') {
        return chatView({ phase: 'thinking', headline: 'Thinking', dedupeKey: key(evt.round || '', evt.attempt || '') });
    }

    if (t === 'tool_call_started') {
        return chatView({ headline: 'Working through the next step', dedupeKey: key(evt.tool || '') });
    }

    if (t === 'task_checkpoint') {
        if (evt.checkpoint_kind === 'context_fit_low_retry') {
            return chatView({
                phase: 'warn',
                headline: 'Context rebuilt in Low mode — retrying the same model once',
                visible: true,
                dedupeKey: key(evt.checkpoint_kind, evt.round || ''),
            });
        }
        // Not visible in chat live card — the emit_progress message is the visible source
        // for the chat timeline (avoids duplicate timeline entries). This event remains
        // visible in the Logs tab via summarizeLogEvent.
        const cpNum = evt.checkpoint_number || Math.floor((evt.round || 0) / 15);
        return chatView({
            phase: 'thinking',
            headline: `Checkpoint ${cpNum} — periodic self-check`,
            dedupeKey: key(cpNum),
        });
    }

    if (t === 'llm_round_error' || t === 'llm_api_error') {
        const errorText = describeText(evt.error, 220);
        return chatView({
            phase: 'error',
            headline: 'Thinking step failed',
            body: errorText.preview,
            fullBody: errorText.full,
            visible: true,
            dedupeKey: key(evt.round || ''),
        });
    }

    if (t === 'task_start_settings_reload_failed') {
        // #285 loud disclosure: the task runs on the previously applied
        // configuration — the owner must see that in the chat timeline, not
        // only on the Logs tab.
        const errorText = describeText(evt.error, 220);
        return chatView({
            phase: 'warn',
            headline: 'Settings reload failed at task start',
            body: 'This task runs on the previously applied configuration.'
                + (errorText.preview ? ` (${errorText.preview})` : ''),
            fullBody: errorText.full,
            visible: true,
            dedupeKey: key(),
        });
    }

    if (t === 'tool_call_timeout' || t === 'tool_timeout') {
        return chatView({
            phase: 'error',
            headline: 'One of the steps took too long',
            visible: true,
            dedupeKey: key(evt.tool || ''),
        });
    }

    if (t === 'tool_call_finished' && evt.is_error) {
        const commandText = describeText(extractCommandText(evt.args), 120);
        const errorResult = describeText(evt.result_preview || evt.error, 220);
        const bodyParts = [];
        const fullBodyParts = [];
        if (commandText.preview) bodyParts.push(`Command: ${commandText.preview}`);
        if (errorResult.preview) bodyParts.push(errorResult.preview);
        if (commandText.full) fullBodyParts.push(`Command: ${commandText.full}`);
        if (errorResult.full) fullBodyParts.push(errorResult.full);
        if (evt.status === 'non_zero_exit') {
            const exitCode = Number(evt.exit_code);
            return chatView({
                phase: 'warn',
                headline: `A command returned ${Number.isFinite(exitCode) ? `exit code ${exitCode}` : 'a non-zero exit code'}`,
                body: shortText(bodyParts.join(' '), 220),
                fullBody: fullBodyParts.join('\n\n'),
                visible: true,
                dedupeKey: key(evt.tool || '', evt.status || '', evt.exit_code || '', commandText.full || errorResult.full),
            });
        }
        return chatView({
            phase: 'error',
            headline: 'One of the steps failed',
            body: shortText(bodyParts.join(' '), 220),
            fullBody: fullBodyParts.join('\n\n'),
            visible: true,
            dedupeKey: key(evt.tool || '', evt.status || '', evt.exit_code || '', commandText.full || errorResult.full),
        });
    }

    if (t === 'task_done') {
        const terminal = taskDoneIsTerminal(evt);
        const presentation = taskPresentation(terminal ? taskTerminalPhase(evt) : 'working');
        const unavailable = evt.cost_accounting_status === 'unavailable';
        // C13: the SHARED accessor and its null policy — same alias precedence as
        // chat.js and the Python seams, and a REAL $0 prints instead of vanishing.
        const ownValue = accountedUpperBound(evt) ?? (evt.cost ?? null);
        const ownCost = unavailable
            ? 'cost unavailable'
            : (ownValue != null ? `${formatLogMoney(ownValue)}${evt.cost_final === false ? ' (pending)' : ''}` : '');
        const childrenCost = (accountedUpperBoundWithChildren(evt) ?? -1) > (ownValue ?? 0)
            ? `+children=${formatLogMoney(accountedUpperBoundWithChildren(evt))}${evt.cost_with_children_partial ? ' (partial)' : ''}`
            : '';
        // №8/Q3: an owner-requested soft stop keeps 'done' severity but carries
        // its own headline and the owner-request marker in the details meta.
        const softStopped = taskStoppedWithSummary(evt);
        const reasonDetail = taskReasonDetail(evt);
        return chatView({
            phase: presentation.phase,
            headline: presentation.headline,
            body: reasonDetail,
            visible: true,
            promote: true,
            terminal,
            meta: [softStopped ? OWNER_STOP_DETAIL_MARKER : '', ownCost, childrenCost].filter(Boolean),
            dedupeKey: key(
                JSON.stringify(evt.outcome_axes || {}),
                JSON.stringify(evt.review_projection || {}),
                evt.status || '',
                evt.reason_code || '',
            ),
        });
    }


    if (t === 'task_cost_finalized') {
        const unavailable = evt.cost_accounting_status === 'unavailable';
        const ownCost = unavailable ? 'cost unavailable' : formatLogMoney(accountedUpperBound(evt));
        const subtreeCost = unavailable ? '' : formatLogMoney(accountedUpperBoundWithChildren(evt));
        // A cost checkpoint is bookkeeping, never the task's conclusion: only
        // the settled task_done resolves the card. On the blocking lane this
        // frame precedes task_done; treating it as terminal closed the card
        // early, and a live card mid-"Finalizing…" must absorb it quietly.
        return chatView({
            phase: unavailable ? 'warn' : 'usage',
            headline: unavailable ? 'Cost accounting unavailable' : 'Cost finalized',
            visible: false,
            terminal: false,
            meta: [ownCost, subtreeCost ? `subtree=${subtreeCost}` : ''].filter(Boolean),
            dedupeKey: key('task-cost-finalized', evt.post_task_status || ''),
        });
    }

    return chatView({ dedupeKey: key() });
}

export function duplicateLogEventKey(evt) {
    const t = evt.type || evt.event || '';
    if (t === 'startup_verification') return `${t}:${evt.git_sha || ''}:${evt.issues_count || 0}`;
    if (t === 'worker_sha_verify') return `${t}:${evt.expected_sha || ''}:${evt.observed_sha || ''}:${evt.ok ? 1 : 0}`;
    if (t === 'deps_sync_ok') return `${t}:${evt.reason || ''}:${evt.source || ''}`;
    return '';
}

export function prettyLogEvent(evt) {
    try {
        return JSON.stringify(evt, null, 2);
    } catch {
        return String(evt);
    }
}

export function getLogTaskGroupId(evt) {
    const pointer = classifyReviewLifecyclePointer(evt);
    // A duplicate lifecycle pointer is an acknowledgement for an existing
    // owner card, never task lineage. Logs may show it as a compact standalone
    // row, but must not create a synthetic task group from its outer task_id.
    if (pointer.classification !== 'not_pointer') return '';
    const review = classifyReviewLifecycle(evt);
    if (review.classification === 'source_complete') {
        return String(review.group.presentationOwnerTaskId || '');
    }
    if (review.classification === 'source_incomplete') return '';
    if (evt.subagent_task_id) return String(evt.subagent_task_id);
    if (evt.task_id) return String(evt.task_id);
    const task = evt.task;
    if (task && typeof task === 'object' && task.id) return String(task.id);
    return '';
}

export function isGroupedTaskEvent(evt) {
    const groupId = getLogTaskGroupId(evt);
    if (!groupId) return false;
    const t = evt.type || evt.event || '';
    return (
        evt.is_progress
        || t.startsWith('task_')
        || t.startsWith('llm_')
        || t.startsWith('tool_')
        || t === 'context_building_started'
        || t === 'context_building_finished'
        || t === 'send_message'
    );
}

// Sticky-card precedence (adversarial wave B-ADV-2): an evidence-bearing
// (receipt) chip is never downgraded by a later evidence-less (dispatch)
// frame — the history sync after justFinished anchors on a mid-run row.
export function keepStickyExecutorChip(prior, next) {
    return !!(prior && prior.hasEvidence && next && !next.hasEvidence);
}
