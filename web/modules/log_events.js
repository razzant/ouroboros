import { accountedUpperBound, accountedUpperBoundWithChildren, formatUsd4, plainCauseText, projectMarkdownLines, savedTerminalRowNote } from './utils.js';
import { cancelCauseClauses, ownerStopEnding } from './cancel_presentation.js';
import { TASK_CAUSE_PHRASES, taskReasonPhrase, describedCause, reasonCodesNote, residualToolErrorsPhrase } from './task_reason_words.js';
export { taskReasonPhrase, reasonCodesNote } from './task_reason_words.js';
import { taskCheckpointLabel, taskCheckpointIdentity, checkpointHasProgressRow, waitEpisode } from './task_checkpoints.js';
export { taskCheckpointLabel } from './task_checkpoints.js';
import { coordinatorRouteClause, formatCoordinatorProvenance, harnessPresentation } from './harness_presentation.js';
import { acceptanceIncidentClauses } from './acceptance_incident_presentation.js';
import { fmt, tr } from './i18n.js';
import { historyRetentionView } from './history_retention.js';
import { checkFailureView } from './verification_presentation.js';
import { hostRowDisplayText } from './host_notice_text.js';
import { effortEvidenceText } from './effort_evidence.js';
import { delegatedActivityView } from './delegated_activity.js';
import { workerShaLogView, workerShaRelation } from './worker_sha_presentation.js';
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

// The Errors filter follows the same typed failure phases on live and replay rows.
const ERROR_LOG_PHASES = new Set(['error', 'timeout', 'lifecycle_error']);

export function categorizeLogEvent(evt, view = summarizeLogEvent(evt)) {
    const t = evt.type || evt.event || '';
    // A wake-up's rows carry the turn's origin label (`initiator`).
    const wake = evt.initiator === 'consciousness';
    if (evt.is_progress) {
        return wake ? 'consciousness' : 'tasks';
    }
    // Severity comes from the typed projection, never from the event name; the
    // name substrings below only pick the domain family of a non-error row.
    if (ERROR_LOG_PHASES.has(String(view?.phase || ''))) return 'errors';
    if (wake) return 'consciousness';
    if (t.includes('llm') || t.includes('model')) return 'llm';
    if (t.includes('tool') || evt.tool) return 'tools';
    if (t === 'history_retention' || t.includes('task') || t.includes('evolution') || t.includes('review')) return 'tasks';
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

// For markdown narration, the line-anchored markers (headings with ` — ` before
// the text under them, bullets, table rows) are projected BEFORE the newlines
// collapse into the one-line preview: afterwards no line-anchored rule could tell
// a heading or a list item from prose. Typed text (shell commands, errors, traces)
// is never markdown: a `# comment` stays one. Marker-only text keeps its source,
// and `full` stays the source text either way.
function describeText(text, maxLen = 180, { markdown = false } = {}) {
    const full = String(text || '').trim();
    if (!full) return { preview: '', full: '' };
    const lined = markdown ? projectMarkdownLines(full) : full;
    const previewSource = (lined.trim() ? lined : full).replace(/\s+/g, ' ');
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
// бабл \ субагент на codex». A small chip naming the harness this bubble or
// subagent was DISPATCHED to — icon plus the short harness name, in the style of
// the agent account rows on the Agents tab, never a promotional badge.
// Dispatch, not receipt: `executor_route` is resolved once when the work is sent,
// so the chip says where it was sent; a landing below that ask is disclosed on
// `capability_delta`. Only a DELEGATED route is worth a chip: the native API path
// is the ordinary case and prints nothing (absent fact -> null -> no element).
// The completion seam's typed substrate claim (subagents.actual_substrate) rides
// the terminal frame beside the counts and joins the detail sentence; the enum
// never travels bare.
const SUBSTRATE_NOTE = {
    harness_used: 'custody evidence confirms a harness run',
    harness_attempted: 'harness attempted, no delegated run succeeded',
    native_only: 'no harness run recorded',
};

// Quiet label (owner choice #360 A). `label` is the helper's name, with one short
// ` · phrase` only where the bare name would read as a receipt nothing issued: a
// progress actor (`last update`), a refused pin (`blocked`, frozen), unreadable
// evidence, no run, runs not settled, failed or unverified runs, an undisclosed
// outcome split. The ordinary case — an assigned route while the card's own phase
// chip says Working, or every settled run ok — is the bare name: it names the
// selected helper and never proves its session computes now. `title` is the
// detail sentence (counts, model and its source, spend, substrate, access);
// the expanded card prints it in its Executor details row, because a hover
// title alone is invisible on touch, to AT and in copies. `hasEvidence`,
// `observation`, `sourceTs` and the revision feed keepStickyExecutorChip.
export function executorChip(evt) {
    const coordinatorProvenance = formatCoordinatorProvenance(evt);
    const raw = evt?.executor_observation;
    const observation = raw && raw.task_id === String(evt?.subagent_task_id || evt?.task_id || '')
        && raw.run_id && raw.attempt_id && raw.harness_id && Number.isInteger(raw.revision)
        ? raw : null;
    if (observation && !evt?.execution_evidence) {
        // A well-formed observation without execution evidence is the run's own
        // progress report: it keeps `last update` (an observation, not a receipt)
        // and its requested/observed model source, or `model unconfirmed`, in the detail.
        const name = harnessPresentation(observation.harness_id).label;
        const model = observation.model_source === 'observed' || observation.model_source === 'requested'
            ? compactModel(observation.model) : '';
        return {
            harness: observation.harness_id,
            hasEvidence: false,
            observation,
            sourceTs: Date.parse(evt?.ts || evt?.timestamp || '') || 0,
            label: `${name} · last update`,
            title: `Latest observed activity: ${name}, ${model ? `model ${model} (${observation.model_source})` : 'model unconfirmed'}, run ${observation.run_id}, attempt ${observation.attempt_id}, revision ${observation.revision}. This is progress, not a terminal execution receipt.`,
            ...(coordinatorProvenance ? { coordinatorProvenance } : {}),
        };
    }
    const route = String(evt?.executor_route || '').trim();
    if (!route) return null;
    // The route id is OPAQUE (`harness` or `harness=model`): print the harness
    // part only, never interpreted beyond splitting the spelling Claudexor uses.
    const harness = route.split('=')[0].trim().toLowerCase();
    if (!harness) return null;
    const name = harnessPresentation(harness).label;
    const base = {
        harness, label: name, sourceTs: Date.parse(evt?.ts || evt?.timestamp || '') || 0,
        ...(coordinatorProvenance ? { coordinatorProvenance } : {}),
    };
    const chip = (hasEvidence, phrase, title) => ({ ...base, hasEvidence, label: phrase ? `${name} · ${phrase}` : name, title });
    // LAYERED TRUTH: identity (mark + product name) comes from the
    // harness_presentation SSOT; the route is a DISPATCH decision; whether a
    // delegated run happened is EVIDENCE, reconciled once at the completion seam
    // (subagents.envelope_from_task -> execution_evidence) on the terminal frame.
    const evidence = (evt && typeof evt.execution_evidence === 'object' && evt.execution_evidence)
        ? evt.execution_evidence : null;
    // The substrate clause is a completion-seam claim coupled to evidence: never
    // attach it to an evidence-less frame, where it could contradict the dispatch.
    const substrateNote = evidence ? (SUBSTRATE_NOTE[String(evt?.actual_substrate || '')] || '') : '';
    const withSubstrate = (title) => (substrateNote ? `${title} — ${substrateNote}` : title);
    if (String(evt?.reason_code || '') === 'subagent_executor_unavailable') {
        // The typed $0 terminal of a harness pin the resolution refused (#363):
        // the route names WHO refused, the reason code says the child never ran.
        // Checked before the evidence branches — an empty custody read would
        // otherwise print "no run yet" over a child never dispatched at all.
        // Evidence-grade so a later dispatch-shaped frame cannot downgrade it.
        return chip(true, 'blocked', `Pinned to ${name}, but the route could not run — the task was NOT run (no metered API spend, no delegated run)`);
    }
    if (!evidence) {
        const record = normalizeTaskTerminalRecord(evt);
        const status = String(record.outcome_axes?.lifecycle?.status || record.status || evt?.subagent_event || '').toLowerCase();
        if (OBSERVED_OUTCOME_STATUSES.has(status)) {
            return chip(false, 'run outcome unknown', `The ${name} route was assigned, but this terminal record has no delegated-run evidence; the run outcome is unknown`);
        }
        // Evidence rides TERMINAL frames only, so a live frame proves nothing
        // either way, and under the pre-start charter the leaf usually IS running
        // by now: an evidence-grade negative ("no run yet") would be false for most
        // of the live phase. The bare name states the assignment the frame carries.
        return chip(false, '', `Dispatched to ${name} — run evidence arrives with the terminal receipt; ${coordinatorRouteClause(evt)}`);
    }
    const started = Number(evidence.delegated_runs_started || 0);
    const settled = Number(evidence.delegated_runs_settled || 0);
    // Historical frames (v6.94–v6.99) carry delegated_runs_succeeded without
    // delegated_runs_failed: reconstruct the exact complement rather than
    // rendering a clean receipt over an all-failed delegation. A frame with
    // neither counter disclosed no outcome split, so it never reads clean.
    let failed = Number(evidence.delegated_runs_failed ?? NaN);
    if (!Number.isFinite(failed)) {
        const succeeded = Number(evidence.delegated_runs_succeeded ?? NaN);
        failed = Number.isFinite(succeeded) ? Math.max(0, settled - succeeded) : 0;
    }
    failed = Math.max(0, failed);
    if (evidence.evidence_read_failed) {
        // The custody log EXISTS but could not be (fully) read: the counts are
        // UNKNOWN, not an established fact (sol finding, b49f8192 wave) — past
        // recorded starts too, where the partial work-order replay sets the flag.
        // No substrate clause: the seam never claims one over unreadable evidence.
        return chip(true, 'evidence unavailable', started
            ? `The ${name} route was assigned and at least ${started} delegated run(s) started, but the evidence could not be fully read — final counts are unknown`
            : `The ${name} route was assigned, but the delegated-run evidence could not be read — whether a run happened is unknown, not "none"`);
    }
    if (!started) {
        return chip(true, 'no run yet', withSubstrate(`The ${name} route was assigned, but there is no durable record of a delegated run for this subagent`));
    }
    if (!settled) {
        // Terminal-frame material: started-but-unsettled means the run(s) never
        // settled (orphaned or lost), not "still executing" — a present-tense
        // "running" on a finished card would be a lie.
        return chip(true, 'not settled', withSubstrate(`Delegated to your ${name} account — ${started} run(s) started, none settled`));
    }
    const cost = evidence.subscription_cost_usd;
    const approx = evidence.subscription_cost_estimated ? '~' : '';
    const costPart = (cost === null || cost === undefined)
        ? 'subscription spend undisclosed'
        : evidence.subscription_cost_partial
            ? `Tracked: ${approx ? 'up to ' : ''}$${Number(cost).toFixed(2)} (total unknown)`
            : `${approx}$${Number(cost).toFixed(2)} subscription`;
    const runsPart = `${settled} run${settled === 1 ? '' : 's'}`;
    // Unverified work-order coverage is an honesty fact: those runs sit inside the
    // aggregate failed count, discounted from ok, so label and detail name them
    // instead of claiming a physical failure, and a mix says "failed or unverified".
    // Nothing orders failures and recoveries in time: the counts carry no chronology.
    // The detail keeps the owner dictionary "N ok, M failed" (plan D9) when either
    // counter is disclosed. Access rides the detail — requested surface from the
    // frame, applied access from the settlement receipts when the engine disclosed
    // it (empty = predates the receipt).
    const counted = Number.isFinite(Number(evidence.delegated_runs_failed ?? NaN))
        || Number.isFinite(Number(evidence.delegated_runs_succeeded ?? NaN));
    const unresolved = Number(evidence.delegated_runs_source_unresolved || 0);
    const notOk = Math.max(failed, unresolved);
    const ok = Math.max(0, settled - notOk);
    // Coverage can be incomplete on a non-successful run; equal aggregate counts
    // cannot erase that terminal fact. Legacy frames without states stay count-based.
    const terminalFailure = Array.isArray(evidence.delegated_run_failure_states)
        && evidence.delegated_run_failure_states.some(state => typeof state === 'string' && state && state !== 'source_incomplete');
    const notOkWord = unresolved && (terminalFailure || unresolved < notOk) ? 'failed or unverified'
        : unresolved ? 'unverified' : 'failed';
    const okPart = `${ok} ok${notOk ? `, ${notOk} ${notOkWord}` : ''}`;
    const unsettled = Math.max(0, started - settled);
    const phrase = notOk ? (settled === 1 ? `run ${notOkWord}` : `${notOk} of ${settled} runs ${notOkWord}`)
        : unsettled ? `${unsettled} of ${started} runs not settled` : (counted ? '' : 'run outcome unknown');
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
    const unsettledTitle = unsettled ? ` — ${unsettled} more run(s) started, not settled` : '';
    return {
        ...chip(true, phrase, withSubstrate(`Delegated to your ${name} account — ${runsPart} settled${counted ? ` (${okPart})` : ''}, ${costPart}`)
            + unresolvedTitle + unsettledTitle + accessPart),
        observedModels: Array.isArray(evidence.harness_models)
            ? [...new Set(evidence.harness_models.filter((model) => typeof model === 'string' && model))] : [],
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
    const modelPart = full && compactModel(model) ? ` · agent model ${compactModel(model)}` : '';
    return `${cleanRole}${modelPart}${shortId && full ? ` (${shortId})` : ''}${suffix}`;
}

const SUBAGENT_CARD_LABEL = { scheduled: 'Working', running: 'Working', interrupted: 'Working' };

// A completed child's result is its report, written for its parent: the label says so before the
// card's preview and over the full text. Another ending's result may be host text: [RESULT] stays.
export function childReportLabel(childStatus) {
    return ['completed', 'completed_warn'].includes(String(childStatus || '').toLowerCase())
        ? tr('subagent.report_to_parent', 'Report to parent') : '';
}

export function formatLogMoney(value) {
    return formatUsd4(value);
}

export function formatLogDuration(sec) {
    const num = Number(sec);
    if (!Number.isFinite(num) || num < 0) return '';
    if (num >= 60) return `${Math.floor(num / 60)}m ${Math.round(num % 60)}s`;
    return `${num < 10 ? num.toFixed(1) : Math.round(num)}s`;
}

// task_done ledger totals: a sum no row reported (null) is '?', never 0.
function formatLogTokens(evt) {
    const count = (n) => (n === null ? '?' : Number(n || 0));
    const prompt = count(evt.prompt_tokens), completion = count(evt.completion_tokens);
    if (![prompt, completion].some((n) => n !== '?' && n)) return '';
    return `${prompt}\u2192${completion} tok`;
}

// Usage rows only (#1578): v2 null = not reported, 0 = a reported zero; an unstamped/v1 zero stays ambiguous (LEGACY_RULE).
// A session's input_token_usage total is its whole input, as a direct call's prompt_tokens; its harness's own
// prompt_tokens may count only the uncached rest, so a split without a total shows the input as unknown.
function formatUsageTokens(evt) {
    const count = (n) => (!Number.isSafeInteger(n) || n < 0 ? '?' : n === 0 && evt.record_contract_version !== 2 ? '0?' : String(n));
    const split = evt.input_token_usage && typeof evt.input_token_usage === 'object' ? evt.input_token_usage : null;
    const prompt = count(split ? split.total_tokens : evt.prompt_tokens), completion = count(evt.completion_tokens);
    if (prompt === '?' && completion === '?') return 'tokens not reported';
    return `${prompt}\u2192${completion} tok${[prompt, completion].includes('0?') ? ' (legacy 0 may be unreported)' : ''}`;
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

// Legacy observations lack host identity: retain them separately rather than
// guessing a call from a reused provider id, tool name or target.
const legacyToolObservations = new WeakMap();
let nextLegacyToolObservation = 0;
function toolCallKey(evt, groupId) {
    if (evt.invocation_id) return `tool:${groupId}:${evt.invocation_id}`;
    if (evt.history_id) return `tool:${groupId}:history:${evt.history_id}`;
    if (evt.event_id) return `tool:${groupId}:event:${evt.event_id}`;
    if (!legacyToolObservations.has(evt)) legacyToolObservations.set(evt, ++nextLegacyToolObservation);
    return `tool:${groupId}:observation:${legacyToolObservations.get(evt)}`;
}

const toolObservation = (evt, groupId, status) => ({
    key: toolCallKey(evt, groupId), status, receipt: Boolean(evt.routing_action) || evt.completion_control === true, tool: evt.tool || '',
    fact: (evt.type || evt.event) === 'tool_call_started' ? 'started'
        : ['tool_call_timeout', 'tool_timeout'].includes(evt.type || evt.event) ? 'wait_ended' : 'settled',
    hostError: evt.status === 'host_error', live: evt._live_tool_frame === true,
});

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

// A Wrap up's answer is a SUCCESSFUL soft stop, not a warning: the headline stays "Done"
// while the details carry this marker; so does an older record, which names no button.
export const OWNER_STOP_DETAIL_MARKER = "summary at the owner's request — best available result";

export function taskStoppedWithSummary(evt) {
    return ownerStopEnding(normalizeTaskTerminalRecord(evt))[1] === 'summary';
}

// Twin of outcomes._advisory_procedural_refusal: an old task-acceptance FAIL that was only the
// Advisory host declining a second panel over a verdict that was not FAIL reads best_effort.
function advisoryProceduralRefusal(record) {
    const { objective, review, execution } = record?.outcome_axes || {};
    const decision = review?.acceptance_decision;
    const signals = Array.isArray(review?.aggregate_signals) ? review.aggregate_signals.map((s) => String(s).toUpperCase()) : [];
    const ready = Object.hasOwn(record, 'result') ? typeof record.result === 'string' && Boolean(record.result)
        : record._result_text_present === true;
    return ready
        && ['', 'model_final'].includes(record.terminal_origin ?? '')
        && execution?.status === 'ok' && objective?.status === 'fail' && objective.source === 'task_acceptance_review'
        && objective.reason === 'identical_acceptance_refused' && decision?.reason === objective.reason
        && decision.status === 'finalized_unaccepted' && decision.enforcement === 'advisory'
        && review.status === 'degraded' && !signals.includes('FAIL');
}

// The card's one sentence about a delivery the host could not confirm: an exhausted delivery names
// the saved copy when one exists and says so when it does not. Twin: project_dialogue.terminal_delivery_clause.
export function terminalDeliveryClause(delivery) {
    const state = String(delivery?.state || '');
    if (!TASK_CAUSE_PHRASES[`terminal_delivery_${state}`]) return '';
    const path = String(delivery?.preserved_path || '');
    return taskReasonPhrase(`terminal_delivery_${state}`) + (state !== 'exhausted' ? ''
        : path ? `; a copy of the answer is saved at ${path}.` : '; no copy of the answer could be saved.');
}

// The cause lookup the shared acceptance presenter receives: no second path reads the table.
const causePhraseLookup = (key) => (TASK_CAUSE_PHRASES[String(key || '')] ? taskReasonPhrase(key) : '');

// The debt is the record's current delegated_runs_unreconciled list, never its reason (#1583): only a nonempty
// list states debt beside the cause; absence proves no reconciliation. Live task_done carries it too
// (_custody_debt_event_fields); an older row's custody stamp is read beneath, from its typed execution reason.
// Twin: _custody_debt_reason; other codes pass unchanged.
const CUSTODY_DEBT_REASON = 'delegated_custody_unreconciled';

function custodyDebtReason(record) {
    const raw = String(record?.reason_code || '');
    const typed = String(record?.outcome_axes?.execution?.reason_code || '');  // a stamp read back names no cause
    const debt = record?.delegated_runs_unreconciled;
    return [
        raw === CUSTODY_DEBT_REASON ? (typed === CUSTODY_DEBT_REASON ? '' : typed) : raw,
        Array.isArray(debt) && debt.length ? CUSTODY_DEBT_REASON : '',
    ];
}

// The headline's objective warning: a nonempty current debt list warns, and an older row's custody warning
// stops counting once that list is known empty; other warnings, and one beside an absent list, keep the word.
// Twin: outcomes.objective_warned.
function objectiveWarned(record) {
    const objective = record?.outcome_axes?.objective || {};
    const debt = record?.delegated_runs_unreconciled;
    if (Array.isArray(debt) && debt.length) return true;
    if (!Array.isArray(debt) || objective.warning !== CUSTODY_DEBT_REASON) return Boolean(objective.warning);
    return (Array.isArray(objective.warnings) ? objective.warnings : []).some((code) => code && code !== CUSTODY_DEBT_REASON);
}

// outcome_axes.execution.plan_review picks its sentence by lookup, never by its spelling; absent classes keep fallback.
function planReviewKey(record, fallback) {
    const key = `plan_review_${String(record?.outcome_axes?.execution?.plan_review || '')}`;
    return Object.hasOwn(TASK_CAUSE_PHRASES, key) ? key : fallback;
}

// The browser twin of project_dialogue._join_cause_clauses: one line, one
// clause per distinct fact; a clause that is not last drops its own full stop.
function joinCauseClauses(clauses) {
    const kept = [];
    for (const clause of clauses) if (clause && !kept.includes(clause)) kept.push(clause);
    return kept
        .map((clause, index) => (index < kept.length - 1 && clause.endsWith('.')
            ? clause.slice(0, -1) : clause))
        .join(' · ');
}

// The explicit author stop (TZ-2 C4): the typed author_stop, or the structured stop recorded under its TRUE
// cause when the review rounds ran out. The twin of review_records.recorded_author_stop.
const explicitAuthorStop = (d) => d?.reason === 'author_stop' || (d?.reason === 'review_cycles_exhausted'
    && d.author_action === 'stop' && d.author_disposition?.action === 'stop');
// Its AUTHOR's reason beside the typed sentence (never a finish's or reviewer prose). Twin: project_dialogue._author_stop_rationale.
const authorStopRationale = (d) => (explicitAuthorStop(d) ? String(d.author_disposition?.rationale || '').split(/\s+/).filter(Boolean).join(' ') : '');

// Recorded standing limitations: deferred children and terminal_plan_review_open, worded by its class when present.
// Twin: project_dialogue._terminal_limitations. Open classes exclude the merely awaited case.
const PLAN_REVIEW_OPEN_CLASSES = new Set(['plan_review_unanswered', 'plan_review_none_answered', 'plan_review_answered_open', 'plan_review_objections_open']);

function terminalLimitations(record, reason, held = false) {
    const deferred = Number(record?.outcome_axes?.objective?.deferred_count || 0) > 0;
    const planKey = reason === 'plan_review_advisory' ? reason : 'terminal_plan_review_open';
    // The class rides the live event and the replayed row where the result-only flag does not;
    // a HELD task states the hold as its primary cause and never a limitation of work that went on.
    const open = !held && (record?.terminal_plan_review_open === true
        || PLAN_REVIEW_OPEN_CLASSES.has(`plan_review_${String(record?.outcome_axes?.execution?.plan_review || '')}`));
    return [deferred ? taskReasonPhrase('child_results_deferred') : '', open ? taskReasonPhrase(planReviewKey(record, planKey)) : ''];
}

// Independent functional causes precede review provenance, never replacing it.
// Twin of project_dialogue._completion_verdict; neither renderer mutates health.
export function taskReasonCodes(evt) {
    const codes = [];
    taskReasonDetail(evt, codes);
    return codes;
}

export function taskReasonDetail(evt, codes = []) {
    const phrase = (code) => describedCause(code, codes);
    const record = normalizeTaskTerminalRecord(evt);
    const decision = record.outcome_axes?.review?.acceptance_decision
        ?? record.review_status?.acceptance_decision;
    const severity = taskOutcomeSeverity(evt);
    const decisionCause = String(decision?.reason || '');
    const objective = record.outcome_axes?.objective;
    const completion = record.task_completion ?? record.outcome_axes?.execution?.task_completion;
    const [storedReason, custody] = custodyDebtReason(record);
    const typedReason = String(record.outcome_axes?.execution?.reason_code || '');
    const reason = ['', 'final_message'].includes(storedReason) && typedReason !== CUSTODY_DEBT_REASON
        ? (typedReason || storedReason) : storedReason;
    const origin = record.cancel_origin;
    const held = severity === 'error' && String(objective?.source || '').startsWith('plan_review_');
    const reviewCause = !taskStoppedWithSummary(evt) && decision?.status
        && (decision.status !== 'accepted' || Object.hasOwn(TASK_CAUSE_PHRASES, decisionCause))
        && (['done', 'warn'].includes(severity) || explicitAuthorStop(decision)
            || (severity === 'error' && decision.status !== 'accepted' && !custody
                && ['', 'final_message'].includes(reason))) ? decisionCause : '';
    const bypass = { budget_exhausted: 'acceptance_bypassed_budget_exhausted', round_limit: 'acceptance_bypassed_round_limit',
        finalization_grace: 'acceptance_bypassed_deadline', deadline_local: 'acceptance_bypassed_deadline',
        provider_unavailable: 'acceptance_bypassed_provider_unavailable', context_overflow: 'acceptance_bypassed_context_overflow',
        children_unabsorbed: 'acceptance_bypassed_children_unabsorbed', owner_requested_finalization: 'acceptance_bypassed_owner_requested_finalization' };
    let clause = '';
    if (reason === 'foreign_install_root' && record.reason_detail) clause = String(record.reason_detail);
    else if (taskStoppedWithSummary(evt)) clause = '';
    else if ((severity === "cancelled" || ownerStopEnding(record)[1] === 'stop') && origin && typeof origin === "object"
        && !Array.isArray(origin) && Object.keys(origin).length) {
        clause = joinCauseClauses(cancelCauseClauses(origin, record));
    } else if (held) {
        clause = phrase(String(objective.reason || ''));
    } else if (!reason || reason === 'final_message') {
        clause = completion?.action === 'stop' && severity !== 'cancelled' ? taskReasonPhrase('author_stop')
            : severity === 'done' && record.outcome_axes?.execution?.plan_review === 'awaiting' ? taskReasonPhrase('plan_review_awaiting') : '';
    } else {
        const receiptVeto = objective?.receipt_veto;
        clause = receiptVeto?.reason === reason && receiptVeto.detail
            ? String(receiptVeto.detail).split(/\s+/).filter(Boolean).join(' ')
            : phrase(reason === 'plan_review_advisory' ? planReviewKey(record, reason) : reason);
    }
    if (reviewCause && (reason === reviewCause || bypass[reason] === reviewCause)) clause = '';
    const excluded = ['', 'delegated_custody_unreconciled', 'terminal_delivery_unknown', 'terminal_delivery_exhausted'];
    const warnings = [...(Array.isArray(objective?.warnings) ? objective.warnings : []), objective?.warning || ''];
    const named = residualToolErrorsPhrase(record.outcome_axes?.execution);  // the failure the record holds, in owner words
    const warningClauses = warnings.filter(w => !excluded.includes(w))
        .map(w => (w === 'residual_tool_errors_without_review' && named) || taskReasonPhrase(w));
    const artifacts = record.artifact_bundle?.status || record.artifact_status || record.outcome_axes?.artifacts?.status;
    if (['failed', 'missing'].includes(artifacts)) warningClauses.push(taskReasonPhrase('artifacts_unavailable'));
    if (decision?.open_obligations?.length) warningClauses.push(taskReasonPhrase('open_obligations'));
    if (objective?.status === 'best_effort' && !clause && !objective.deferred_count && !taskStoppedWithSummary(evt)
        && ['', 'previous_revision_accepted'].includes(reviewCause)) warningClauses.push(taskReasonPhrase('partial_result'));
    const rationale = completion?.action === 'stop' ? plainCauseText(completion.rationale, 0) : authorStopRationale(decision);
    const remainder = record.outcome_axes?.execution?.mutation_attribution?.uncommitted_remainder;
    const count = remainder?.count;
    const remainderClause = remainder?.state === 'unknown' ? 'Uncommitted files in the saved candidate could not be checked'
        : remainder?.state === 'known' && Number.isInteger(count) && count > 0
            ? `${count} uncommitted file${count === 1 ? '' : 's'} remain${count === 1 ? 's' : ''} in the saved candidate` : '';
    const line = joinCauseClauses([clause, ...warningClauses, phrase(reviewCause), rationale, remainderClause,
        ...acceptanceIncidentClauses(decision, reason, causePhraseLookup),
        ...terminalLimitations(record, reason, held), custody ? taskReasonPhrase(custody) : '',
        terminalDeliveryClause(record.terminal_delivery)]);  // the host twin appends the delivery clause last too
    return (severity === 'cancelled' || remainderClause) && line && !/[.!?…)\u2026]$/.test(line) ? `${line}.` : line;
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
    // Stopped. Checked first: forced teardown routinely leaves failure-shaped
    // side facts (e.g. artifacts missing on a cancelled workspace task) that must
    // not relabel an owner-requested cancellation as a failure.
    // 'cancel_requested' as a STATUS is legacy replay only (phase A moved cancel
    // intent to the durable cancel_state projection); old task_done frames and
    // pre-redesign history rows keep resolving as cancelled.
    if (lifecycle === 'cancelled' || lifecycle === 'cancel_requested') {
        return 'cancelled';
    }
    const [command, ending] = ownerStopEnding(record);  // Stop now is Stopped whatever it kept
    if (command === 'stop_now') return 'cancelled';
    const unanswered = ending === 'stop';  // a Wrap up without an answer: its own failure is the stop
    const execution = String(record.outcome_axes?.execution?.status || '').toLowerCase();
    const objective = advisoryProceduralRefusal(record) ? 'best_effort'
        : String(record.outcome_axes?.objective?.status || '').toLowerCase();
    const review = String(record.outcome_axes?.review?.status || record.review_status?.status || '').toLowerCase();
    const authorFinished = objective === 'pass' && record.outcome_axes?.objective?.source === 'author_acceptance';
    const artifacts = String(record.outcome_axes?.artifacts?.status || record.artifact_bundle?.status || record.artifact_status || '').toLowerCase();
    const artifactStatus = String(record.artifact_bundle?.status || record.artifact_status || '').toLowerCase();
    if (
        ((lifecycle === 'failed' || execution === 'failed') && !unanswered)
        || execution === 'infra_failed'
        || objective === 'fail'
        || (review === 'fail' && !authorFinished)
        || ['failed', 'missing'].includes(artifacts)
        || artifactStatus === 'failed'
    ) {
        return 'error';
    }
    if (unanswered) return 'cancelled';
    // A Wrap up's answer is a best_effort SUCCESS, not "Done with warnings".
    if (taskStoppedWithSummary(evt)) {
        return 'done';
    }
    if (
        lifecycle === 'rejected_duplicate'
        || ['degraded', 'best_effort'].includes(execution)
        || ['degraded', 'best_effort'].includes(objective)
        || (review === 'degraded' && !authorFinished)
        || objectiveWarned(record)
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
// fabricated Done/Stopped projection.
const TERMINAL_TASK_DETAIL_STATUSES = new Set([
    'completed', 'failed', 'cancelled', 'rejected_duplicate',
]);
// Statuses that make the OUTCOME known, whether or not the card may close yet.
const OBSERVED_OUTCOME_STATUSES = new Set([...TERMINAL_TASK_DETAIL_STATUSES, 'done', 'cancel_requested']);
const OPEN_POST_TASK_SYNTHESIS_STATUSES = new Set(['pending_once', 'running', 'paused']);

export function isTerminalTaskDetail(record) {
    const status = String(record?.status || '').toLowerCase();
    const synthesis = String(record?.root_phase_checkpoint?.post_task_synthesis || '').toLowerCase();
    return TERMINAL_TASK_DETAIL_STATUSES.has(status)
        && !(['completed', 'failed'].includes(status) && OPEN_POST_TASK_SYNTHESIS_STATUSES.has(synthesis));
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
export function taskPresentation(phase = 'working', quietEffective = false) {
    const normalizedPhase = typeof phase === 'string' && phase.trim() ? phase.trim() : 'working';
    const word = normalizedPhase === 'done' ? 'done'
        : normalizedPhase === 'warn' ? 'warn'
            : normalizedPhase === 'cancelled' ? 'cancelled'
                : ['error', 'timeout', 'lifecycle_error'].includes(normalizedPhase) ? 'error'
                    : 'working';
    const english = { done: 'Done', warn: 'Done with warnings', cancelled: 'Stopped', error: 'Failed', working: 'Working' }[word];
    // The English twin of project_dialogue.OUTCOME_PHASE_HEADLINE; an install language reads it by code.
    return { phase: normalizedPhase, headline: quietEffective === true
        ? tr('task.headline.quiet', 'Done · no message') : tr(`task.headline.${word}`, english) };
}

function taskOutcomeMeta(evt) {
    const axes = evt.outcome_axes || {};
    return [
        axes.lifecycle?.status ? `lifecycle ${axes.lifecycle.status}` : '',
        axes.execution?.status ? `execution ${axes.execution.status}` : '',
        axes.objective?.status ? `objective ${axes.objective.status}` : '',
        axes.review?.status ? `review ${axes.review.status}` : '',
        axes.review?.acceptance_decision?.status ? `acceptance ${axes.review.acceptance_decision.status}` : '',
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
        const narration = describeText(String(evt.content || evt.text || '').replace(/^💬\s*/, ''), 240, { markdown: true });
        if (isSubagentEvent(evt)) {
            const sid = subagentId(evt);
            const event = String(evt.subagent_event || 'update').toLowerCase();
            const role = String(evt.subagent_role || '').trim();
            return view(event === 'completed' ? 'done' : event === 'failed' || event === 'rejected' ? 'warn' : 'progress', subagentHeadline(sid, role, event, evt.model, { full: true }), {
                body: narration.preview,
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
        return view('progress', narration.preview || 'Progress update', { meta: ['task'] });
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
            body: effortEvidenceText(evt),
            meta: taskMeta(
                evt.model || '',
                formatUsageTokens(evt),
                // Backfill uses accounted upper bounds; live frames may carry cost.
                formatLogMoney(accountedUpperBound(evt) ?? evt.cost),
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
            body: effortEvidenceText(evt),
            meta: taskMeta(
                evt.model || '',
                formatUsageTokens(evt),
                formatLogMoney(accountedUpperBound(evt) ?? evt.cost),
                evt.category || '',
            ),
        });
    }

    if (t === 'tool_call_started') {  // processing began; a replayed start alone is no liveness claim
        return view('start', `Started ${evt.tool || 'tool'}`, {
            body: compactJson(evt.args, 260),
            meta: taskMeta(evt.timeout_sec ? `timeout ${evt.timeout_sec}s` : ''),
        });
    }

    if (t === 'tool_call_timeout' || t === 'tool_timeout') {
        return view('timeout', `${evt.tool || 'tool'} timed out`, {
            body: compactJson(evt.args, 220),
            meta: taskMeta(evt.timeout_sec ? `limit ${evt.timeout_sec}s` : '', formatLogDuration(evt.duration_sec)),
        });
    }

    if (t === 'tool_call' || t === 'tool_call_finished' || evt.tool) {
        // The durable tools.jsonl row (replay/backfill) carries the same typed
        // failure facts as the live tool_call_finished frame — is_error plus the
        // signal/exit facts — so a failed call reads the same after a reload.
        const signalDeath = Boolean(evt.signal) || (typeof evt.exit_code === 'number' && evt.exit_code < 0);
        const failed = Boolean(evt.is_error) || signalDeath;
        const finished = t === 'tool_call_finished';
        const label = signalDeath ? `killed (${evt.signal || evt.exit_code})`
            : failed ? 'failed' : finished ? 'finished' : 'result';
        return view(failed ? 'error' : finished ? 'done' : 'result', `${evt.tool || 'tool'} ${label}`, {
            body: shortText(evt.result_preview || (finished ? '' : compactJson(evt.args, 220)), 260),
            meta: taskMeta(formatLogDuration(evt.duration_sec)),
        });
    }

    if (t === 'task_start_settings_reload_failed') {
        // #285 disclosure, same valence as the Chat card: the task runs on the
        // previously applied configuration — a warning, not an unresolved error.
        return view('warn', 'Settings reload failed at task start', {
            body: shortText(evt.error, 260),
            meta: taskMeta('runs on the previously applied configuration'),
        });
    }

    if (t === 'history_retention') {
        const retention = historyRetentionView(evt);
        if (retention) return view(retention.phase, retention.headline, { body: retention.body, meta: taskMeta(...retention.meta) });
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

    if (t === 'task_message_injected') {
        // A message from another task landed in THIS task's transcript (its
        // timeline groups on task_id). The sender is named by value; the
        // provenance says how it was framed (ancestor / relayed peer /
        // independent task / system / escalation).
        const source = evt.source_task_id ? String(evt.source_task_id) : 'another task';
        return view('info', `Message from task ${source}`, {
            body: shortText(evt.text_preview, 200),
            meta: taskMeta(
                evt.provenance ? `provenance=${evt.provenance}` : '',
                evt.relayed_from_task_id ? `relayed=${evt.relayed_from_task_id}` : '',
            ),
        });
    }

    if (t === 'task_message_routed') {
        // The SENDER's row for a task-authored message (task_id is the author):
        // written to the target's mailbox, or refused with the host's reason.
        const target = evt.target_task_id ? String(evt.target_task_id) : 'task';
        const written = String(evt.status || '') === 'written';
        return view(written ? 'info' : 'warn', written ? `Message sent to task ${target}` : `Message to task ${target} refused`, {
            body: written ? '' : shortText(evt.reason, 160),
            meta: taskMeta(`target=${target}`, evt.status ? String(evt.status) : ''),
        });
    }

    if (t === 'task_metrics_event' || t === 'task_eval') {
        return view('metrics', 'Task metrics', {
            meta: taskMeta(
                evt.task_type || '',
                ...taskOutcomeMeta(evt),
                modelExecutionLabel(evt.model_execution),
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
        const outcome = taskTerminalPhase(evt);
        const presentation = taskPresentation(terminal || outcome === 'error' ? outcome : 'working', terminal && evt.quiet_effective === true);
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
                modelExecutionLabel(evt.model_execution),
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

    // One row per worker-lifecycle frame, naming the fields it carries: the two
    // readiness frames fell through to the unknown-name remainder and rendered as a
    // bare event name with no fields, so a stalled spawn wave was unreadable (#1073).
    if (t === 'worker_spawn_start') return view('start', `Spawning ${evt.count || '?'} workers`, { meta: [evt.start_method || ''] });
    if (t === 'worker_starting') return view('start', 'Worker starting', { meta: [
        evt.worker_id != null ? `worker ${evt.worker_id}` : '', evt.pid ? `pid ${evt.pid}` : '', evt.phase ? `phase ${evt.phase}` : ''] });
    if (t === 'worker_ready_window_extended') return view('warn', 'Worker readiness window extended', { meta: [
        Array.isArray(evt.worker_ids) && evt.worker_ids.length ? `workers ${evt.worker_ids.join(', ')}` : '',
        evt.attempt != null ? `attempt ${evt.attempt}` : '', evt.window_sec != null ? `window ${evt.window_sec}s` : '',
        evt.ceiling_sec != null ? `ceiling ${evt.ceiling_sec}s` : ''] });

    if (t === 'worker_sha_verify') {
        const result = workerShaLogView(evt);
        return view(result.phase, result.headline, result);
    }

    if (t === 'worker_boot') return view('boot', 'Worker booted', {
        meta: [evt.pid ? `pid ${evt.pid}` : '', evt.git_sha ? String(evt.git_sha).slice(0, 8) : ''] });

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

    if (t === 'task_checkpoint' || t === 'context_reclaim') {
        return view(['context_fit_low_retry', 'service_finalization_error'].includes(evt.checkpoint_kind) ? 'warn' : 'thinking', taskCheckpointLabel(evt), {
            meta: taskMeta(evt.model ? compactModel(evt.model) : '', evt.round ? `r${evt.round}` : '',
                evt.context_tokens ? `~${evt.context_tokens} tok` : '',
                formatLogMoney(evt.task_cost)),
        });
    }

    if (t === 'swarm_fanout') {
        const n = (evt.requested_count != null)
            ? evt.requested_count
            : (Array.isArray(evt.task_ids) ? evt.task_ids.length : 0);
        // #318: a delegated harness run rides the same telemetry with the host
        // constant role="delegated_run"; it is not a subagent.
        const headline = evt.role === 'delegated_run'
            ? 'swarm fan-out: delegated run requested'
            : `swarm fan-out: ${n} subagent(s) requested`;
        return view('info', headline, {
            meta: [
                evt.task_group_id ? `group=${evt.task_group_id}` : '',
                evt.role ? `role=${evt.role}` : '',
                evt.requested_model_lane ? `lane=${evt.requested_model_lane}` : '',
                evt.depth != null ? `depth=${evt.depth}` : '',
                ('fanout_interval_sec' in evt ? evt.fanout_interval_sec : evt.inter_wave_latency_sec) != null
                    ? `since previous fan-out ${'fanout_interval_sec' in evt ? evt.fanout_interval_sec : evt.inter_wave_latency_sec}s` : '',
            ],
        });
    }

    // Typed severity carried by host/extension frames (`ok`, logging `level`) outranks the
    // event name. The name-substring test that follows is the NON-EXPANDING remainder for an
    // unknown name that carries no typed fact: it keeps a genuine producer-side failure with
    // only a name visible under Errors, and it is pinned as a remainder, not a taxonomy.
    const level = String(evt.level || '').toLowerCase();
    const body = shortText(
        evt.error || evt.message || evt.text || evt.result_preview || effortEvidenceText(evt)
            || compactJson(evt.args || evt.task || evt.checks, 260), 260,
    );
    if (evt.ok === true) {
        return view('ok', shortText(t, 120), { body, meta: taskMeta() });
    }
    if (evt.ok === false || level === 'error' || level === 'critical' || level === 'fatal') {
        return view('error', shortText(t, 120), { body, meta: taskMeta(evt.tool ? `tool=${evt.tool}` : '') });
    }
    if (level === 'warning' || level === 'warn') {
        return view('warn', shortText(t, 120), { body, meta: taskMeta() });
    }
    if (t.includes('error') || t.includes('crash') || t.includes('fail')) {
        return view('error', t, { body, meta: taskMeta(evt.tool ? `tool=${evt.tool}` : '') });
    }

    return view('info', shortText(t, 120), {
        body,
        meta: taskMeta(evt.model || '', formatLogMoney(accountedUpperBound(evt) ?? evt.cost)),
    });
}

function chatView({
    phase = 'working',
    headline = '',  // no placeholder: a frame naming nothing has none; the chip says Working (#1369)
    body = '',
    fullBody = '',
    fullHeadline = '',
    activityPreview,
    activityLead = '',  // an inline label the collapsed activity line draws before its text
    visible = false,
    promote = false,
    terminal = false,
    human = false,
    dedupeKey = '',
    meta = [],
    fullRef = '',
    truncated = false,
    chip = null,
    model = '',
    receipt = false,
    toolCall = null,
    waitEpisode = null,
} = {}) {
    const out = { phase, headline, body, visible, promote, terminal, human, dedupeKey };
    // Receipt rows render inside existing blocks but cannot create one.
    if (receipt) out.receipt = true;
    if (toolCall) out.toolCall = toolCall;  // folded into the block's one evidence row
    if (waitEpisode) out.waitEpisode = waitEpisode;  // folded into the block's one row per wait kind
    if (fullBody) out.fullBody = fullBody;
    if (fullHeadline) out.fullHeadline = fullHeadline;
    // Explicit emptiness is part of the presentation contract: a review-only
    // frame has no activity and must not fall back to its disclosure body.
    if (activityPreview !== undefined) out.activityPreview = String(activityPreview || '');
    if (activityLead) out.activityLead = activityLead;
    if (Array.isArray(meta) && meta.length) out.meta = meta.filter(Boolean);
    // P3 uniform contract: when the WS body was truncated server-side, carry a
    // fetch ref (a task id -> GET /api/tasks/{id}) so the bubble can load the
    // genuinely-full output on demand instead of showing only the capped preview.
    if (fullRef) out.fullRef = String(fullRef);
    if (truncated) out.truncated = true;
    if (model) out.model = model;
    // Phase 6: the executor chip rides the projection so live and replay routes
    // paint the same fact; absent stays absent (no placeholder chip).
    if (chip) out.executorChip = chip;
    return out;
}

// Final chat, task_done and replay share one logical completion note.
export function taskTerminalSummary(evt = {}) {
    const terminal = evt.task_phase !== 'finalizing' && evt.outcome_final !== false
        && (evt.outcome_final === true || evt.system_type === 'task_summary'
            || taskDoneIsTerminal(evt));
    const outcome = taskTerminalPhase(evt);
    // Terminality and outcome knowledge are two facts (#1110): the lifecycle status
    // can settle while post-task finalization still runs. Painting the whole frame
    // 'error' put Failed where only a terminal frame may, so the title carried it.
    // The stamp (`task_terminal_status`/`status`) is the producer's word; a row
    // that carries only the canonical axes still knows its lifecycle, so a
    // settled `outcome_axes.lifecycle.status` counts too (as taskOutcomeSeverity
    // already reads it) — an unstamped finalizing replay of a Failed root must
    // not hide the Failed behind "Finalizing…".
    const record = normalizeTaskTerminalRecord(evt);
    const knownStatus = String(record?.status || record?.outcome_axes?.lifecycle?.status || '').toLowerCase();
    const observedOutcome = OBSERVED_OUTCOME_STATUSES.has(knownStatus) ? outcome : '';
    const quiet = terminal && evt.quiet_effective === true;
    const presentation = taskPresentation(terminal ? outcome : 'working', quiet);
    const body = quiet ? '' : [taskStoppedWithSummary(evt) ? OWNER_STOP_DETAIL_MARKER : '', taskReasonDetail(evt),
        reasonCodesNote(taskReasonCodes(evt)), terminal ? savedTerminalRowNote(evt) : ''].filter(Boolean).join('\n');
    const cause = terminal && !quiet ? taskReasonDetail(evt) : '';
    return {
        ...chatView({
            phase: presentation.phase, headline: presentation.headline, body,
            visible: true, promote: true, terminal,
            dedupeKey: `task_done|${evt.task_id || ''}`,
            // A task that ends with a cause states it on the collapsed line (DESIGN, activity block).
            ...(cause ? { activityPreview: cause } : {}),
        }),
        ...(observedOutcome ? { observedOutcome } : {}),
        ...(typeof evt.quiet_effective === 'boolean' ? { quietEffective: evt.quiet_effective } : {}),
        ...(evt.model_execution && typeof evt.model_execution === 'object'
            ? { modelExecution: evt.model_execution } : {}),
        ...(Number.isInteger(evt.tool_calls) ? { toolCalls: evt.tool_calls } : {}),
        ...(evt.initiator ? { initiator: String(evt.initiator) } : {}),
    };
}

// Requested route, usable solve route and provider-reported name are distinct.
// `model` is set only when a usable solve response observed it; an unobserved
// execution keeps its sentence and names no model as the one that ran.
export function modelExecutionView(fact) {
    if (!fact || typeof fact !== 'object') return null;
    const requested = compactModel(fact.requested_model || '');
    if (fact.source !== 'usable_solve_response') {
        return { observed: false, model: '', label: '',
            value: requested ? `Requested ${requested} · execution not observed` : 'Execution not observed' };
    }
    const used = String(fact.used_model || '');
    const reported = String(fact.reported_model || '');
    const requestDiffers = String(fact.requested_model || '') !== used
        || fact.requested_use_local !== fact.used_local;
    const initial = requested && requestDiffers
        ? ` (initial request: ${requested}${fact.requested_use_local ? ' · local' : ''})` : '';
    const model = compactModel(reported || used);
    return { observed: true, model, label: 'Last solve response',
        value: [`${model}${initial}`, fact.used_local === true ? 'local' : '', fact.provider || '',
            reported && reported !== used ? `route: ${used}` : ''].filter(Boolean).join(' · ') };
}

export function modelExecutionLabel(fact) {
    const view = modelExecutionView(fact);
    return !view ? '' : view.label ? `${view.label}: ${view.value}` : view.value;
}

// The turn's origin label rides every projected frame of the turn (progress,
// tool, heartbeat, terminal) so the block's meta line can name it whichever
// frame minted the card; the projection branches below stay label-blind.
export function summarizeChatLiveEvent(evt) {
    const view = summarizeChatLiveEventView(evt);
    if (view && evt?.initiator) view.initiator = String(evt.initiator);
    return view;
}

function summarizeChatLiveEventView(evt) {
    const t = evt.type || evt.event || 'unknown';
    const groupId = getLogTaskGroupId(evt);
    const progressText = describeText(String(evt.content || evt.text || '').replace(/^💬\s*/, ''), 240, { markdown: true });
    const key = (...parts) => [t, groupId, ...parts].join(':');

    if (t === 'history_retention') return chatView({ visible: false, dedupeKey: key(evt.status || '') });

    if (t === 'owner_hurry') {
        // S3 (HQ1) EXPLICIT hide branch: the typed hurry control family never renders a chat
        // timeline row or bubble — chat.js paints only a compact card status from
        // ownerHurryProjection, and the durable facts live in the task detail. Explicit (not
        // the fallthrough) so a future default change cannot silently surface it in chat.
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

    const delegated = delegatedActivityView(evt); // a delegated run's typed observation, root or child
    if (delegated) return Object.assign(chatView({ ...delegated, chip: executorChip(evt), model: evt.model }), { activity: delegated.activity });
    if ((evt.is_progress || t === 'send_message') && isSubagentEvent(evt)) {
        const note = describeText(hostRowDisplayText({ ...evt, text: progressText.full }), 240, { markdown: true });
        const sid = subagentId(evt);
        const rawEvent = String(evt.subagent_event || '').toLowerCase();
        const role = String(evt.subagent_role || '').trim();
        const status = String(evt.status || '').trim();
        const resultText = describeText(evt.result || '', 320, { markdown: true });
        const traceText = describeText(evt.trace_summary || '', 320);
        const errorText = describeText(evt.error || '', 220);
        const cancelDetail = evt.cancel_origin && (rawEvent === 'cancelled'
            || (rawEvent === 'completed' && taskOutcomeSeverity(evt) === 'cancelled'))
            // The child's OWN lineage rides along, so the cancel clause can prove the
            // parent relation instead of naming the swept root as an initiator (#1061).
            ? taskReasonDetail({ status: 'cancelled', cancel_origin: evt.cancel_origin,
                task_id: evt.subagent_task_id || evt.task_id,
                parent_task_id: evt.parent_task_id, root_task_id: evt.root_task_id }) : '';
        const stopDetail = (evt.task_completion ?? evt.outcome_axes?.execution?.task_completion)?.action === 'stop' ? taskReasonDetail(evt) : '';
        const reasonDetail = cancelDetail || stopDetail || (evt.reason_code ? taskReasonPhrase(evt.reason_code) : '');
        // Completed frames take label and phase from their authoritative outcome axes.
        const completionSeverity = rawEvent === 'completed' ? taskOutcomeSeverity(evt) : 'done';
        const event = rawEvent === 'completed'
            ? (completionSeverity === 'cancelled' ? 'cancelled'
                : completionSeverity === 'error' ? 'failed'
                    : completionSeverity === 'warn' ? 'completed_warn'
                        : 'completed')
            : rawEvent;
        const report = resultText.full ? childReportLabel(event) : '';
        const detailParts = [
            note.full,
            resultText.full ? `${report ? `## ${report}` : '[RESULT]'}\n${resultText.full}` : '',  // a timeline label line
            traceText.full ? `[TRACE]\n${traceText.full}` : '',
            errorText.full ? `[ERROR]\n${errorText.full}` : '',
            reasonDetail,
            !cancelDetail && stopDetail ? reasonCodesNote(taskReasonCodes(evt)) : '',
        ].filter(Boolean);
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
        const terminalCause = terminal ? (cancelDetail || stopDetail) : '';
        // A child's own note carries the same voice fact (the progress branch below): a host
        // note inside the child's turn is a visible row that never claims the card's collapsed
        // line. The lifecycle, result and error frames state no voice, so they keep leading.
        const promoted = terminal || evt.narration === true || evt.narration === undefined;
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
                            : note)
            : (note.full ? note
                : resultText.full ? resultText
                    : errorText.full ? errorText
                        : traceText);
        return chatView({
            phase,
            headline: subagentHeadline(sid, role, label, evt.model),
            body: terminalCause || activity.preview || '',
            fullBody: detailParts.join('\n\n'),
            activityPreview: terminalCause || activity.preview || '',
            activityLead: !terminalCause && activity === resultText ? report : '',
            visible: true,
            promote: promoted,
            human: promoted,
            terminal,
            // P3: the WS result/trace were capped at 4000 server-side; expose the
            // subagent task id so "show full" can fetch the genuinely-full output.
            fullRef: sid,
            truncated: Boolean(evt.result_truncated || evt.trace_summary_truncated),
            // «ТУТ … субагент на codex» — the child's own executor chip.
            chip: executorChip(evt),
            model: evt.model,
            dedupeKey: `subagent:${sid}:${label}:${status}:${progressText.full || resultText.full || errorText.full || ''}`,
        });
    }

    if (evt.is_progress || t === 'send_message') {
        const lifecycleTerminal = String(evt.task_id || '').startsWith('skill_lifecycle_')
            && /\s—\s(completed|failed)\b/i.test(progressText.full);
        // Voice, not wording (P5): the worker stamps `narration` on every note; a
        // host note is a typed fact, never a text match. Both stay visible rows; only
        // narration is promoted (title, collapsed line). ABSENT = predates the fact.
        const narration = evt.narration === true || evt.narration === undefined;
        const note = describeText(hostRowDisplayText({ ...evt, text: progressText.full }), 240, { markdown: true });
        return chatView({
            phase: lifecycleTerminal ? (/failed\b/i.test(progressText.full) ? 'lifecycle_error' : 'done') : 'working',
            // An empty note is no narration: a placeholder would become the card title as if said (#1369).
            headline: note.preview || '',
            fullHeadline: note.full || '',
            activityPreview: note.preview || '',
            visible: Boolean(note.preview),
            promote: narration,
            human: narration,
            // «ТУТ бабл … на codex» — an ordinary progress bubble carries the chip
            // too whenever the frame disclosed a delegated executor.
            chip: executorChip(evt),
            model: evt.model,
            dedupeKey: progressText.full ? `progress:${progressText.full}` : `progress:${evt.task_id || ''}`,
        });
    }

    if (t === 'llm_usage') {
        // A helper call can share the task id. Only the task's own loop
        // supplies its coordinating model.
        const ownLoop = Number.isInteger(evt.round);
        return chatView({ model: ownLoop ? evt.model : '', visible: false, dedupeKey: key(evt.round || '') });
    }

    if (t === 'task_started' || t === 'task_received') {
        return chatView({ headline: tr('task.progress.working_on_it', 'Working on it'), promote: true, dedupeKey: key() });
    }

    if (t === 'context_building_started') {
        return chatView({ headline: tr('task.progress.getting_ready', 'Getting ready'), promote: true, dedupeKey: key() });
    }

    if (t === 'context_building_finished') {
        return chatView({ headline: tr('task.progress.looking_through_context', 'Looking through the context'), dedupeKey: key() });
    }

    if (t === 'task_heartbeat') {
        return chatView({ headline: tr('task.progress.still_working', 'Still working'), dedupeKey: key(evt.phase || '') });
    }

    if (t === 'llm_round_started') {
        return chatView({ phase: 'thinking', headline: tr('task.progress.thinking', 'Thinking'), dedupeKey: key(evt.round || '', evt.attempt || '') });
    }

    if (t === 'task_message_injected') {
        // A message from another task landed in this task's transcript: a
        // visible row in the receiver's block (owner 5=A), named by value.
        const source = evt.source_task_id ? String(evt.source_task_id) : tr('task.progress.another_task', 'another task');
        const preview = String(evt.text_preview || '');
        return chatView({
            phase: 'info',
            headline: fmt('Message from task {source}', { source }),
            body: shortText(preview, 200),
            fullBody: preview,
            visible: true,
            dedupeKey: key(source, evt.ts || ''),
        });
    }

    if (t === 'tool_call_started' || (['tool_call_finished', 'tool_call'].includes(t) && !evt.is_error)) {
        // A successful call is execution evidence, not narration: start and finish feed the
        // block's ONE folded row (counts; tools behind Expand), a receipt while every counted
        // call is a host-stamped routing or completion act. Failures keep their own row; so does a
        // completed check whose criterion failed, while its call still counts once as `ok`. `done`
        // is the TASK's phase; a finished CALL is `ok`.
        const status = t === 'tool_call_started' ? (evt._live_tool_frame ? 'calling' : 'unknown') : 'ok';
        return chatView({
            phase: status, headline: '', visible: true, dedupeKey: `tools|${groupId}`,
            receipt: Boolean(evt.routing_action) || evt.completion_control === true,
            ...checkFailureView(evt, toolCallKey(evt, groupId)),
            toolCall: toolObservation(evt, groupId, status),
        });
    }

    if (t === 'task_checkpoint' || t === 'context_reclaim') {
        // Visible in the expanded timeline, with no narration/title promotion; a host wait's
        // episodes fold into one receipt row per kind (task_checkpoints.foldWaitRow).
        const lowRetry = evt.checkpoint_kind === 'context_fit_low_retry';
        const wait = waitEpisode(evt);
        return chatView({
            phase: lowRetry || evt.checkpoint_kind === 'service_finalization_error' ? 'warn' : 'thinking',
            headline: taskCheckpointLabel(evt),
            visible: !checkpointHasProgressRow(evt) && !(evt.checkpoint_kind === 'prompt_prefix_break' && evt.reasoning_retired),
            dedupeKey: wait ? key('wait', wait.kind) : key(evt.checkpoint_kind || '', taskCheckpointIdentity(evt)),
            receipt: Boolean(wait), waitEpisode: wait,
        });
    }

    if (t === 'llm_round_error' || t === 'llm_api_error') {
        const errorText = describeText(evt.error, 220);
        return chatView({
            phase: 'error',
            headline: tr('task.fault.thinking_step_failed', 'Thinking step failed'),
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
            headline: tr('task.fault.settings_reload_failed', 'Settings reload failed at task start'),
            body: tr('task.fault.settings_reload_failed_body', 'This task runs on the previously applied configuration.')
                + (errorText.preview ? ` (${errorText.preview})` : ''),
            fullBody: errorText.full,
            visible: true,
            dedupeKey: key(),
        });
    }

    if (t === 'tool_call_timeout' || t === 'tool_timeout') {
        return chatView({
            phase: 'warn',
            headline: `Tool wait ended; operation may still settle${evt.tool ? ` · ${evt.tool}` : ''}`,
            visible: true,
            dedupeKey: toolCallKey(evt, groupId),
            toolCall: toolObservation(evt, groupId, 'wait_ended'),
        });
    }

    if (['tool_call_finished', 'tool_call'].includes(t) && evt.is_error) {
        const failed = toolObservation(evt, groupId, 'error');
        const commandText = describeText(extractCommandText(evt.args), 120);
        const errorResult = describeText(evt.result_preview || evt.error, 220);
        const bodyParts = [];
        const fullBodyParts = [];
        if (commandText.preview) bodyParts.push(`Command: ${commandText.preview}`);
        if (errorResult.preview) bodyParts.push(errorResult.preview);
        if (commandText.full) fullBodyParts.push(`Command: ${commandText.full}`);
        if (errorResult.full) fullBodyParts.push(errorResult.full);
        const commandFailed = evt.status === 'non_zero_exit';
        const exitCode = commandFailed ? Number(evt.exit_code) : NaN;
        return chatView({
            phase: commandFailed ? 'warn' : 'error',
            headline: commandFailed
                ? `A command returned ${Number.isFinite(exitCode) ? `exit code ${exitCode}` : 'a non-zero exit code'}`
                : `One of the steps failed${evt.tool ? ` · ${evt.tool}` : ''}`,
            body: shortText(bodyParts.join(' '), 220),
            fullBody: fullBodyParts.join('\n\n'),
            visible: true,
            dedupeKey: toolCallKey(evt, groupId),
            toolCall: failed,
        });
    }

    if (t === 'task_done') return taskTerminalSummary(evt);

    if (t === 'task_cost_finalized') {
        const unavailable = evt.cost_accounting_status === 'unavailable';
        const ownCost = unavailable ? 'cost unavailable' : formatLogMoney(accountedUpperBound(evt));
        const subtreeCost = unavailable ? '' : formatLogMoney(accountedUpperBoundWithChildren(evt));
        // A cost checkpoint is bookkeeping, never the task's conclusion: only the settled
        // task_done resolves the card. On the blocking lane this frame precedes task_done;
        // treating it as terminal closed the card early — a live card mid-"Finalizing…" absorbs it.
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
    if (t === 'worker_sha_verify') return `${t}:${evt.expected_sha || ''}:${evt.observed_sha || ''}:${workerShaRelation(evt)}`;
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
    if (!prior || !next) return false;
    if (prior.sourceTs && next.sourceTs && next.sourceTs < prior.sourceTs) return true;
    const before = prior.observation, after = next.observation;
    if (before && after) {
        return before.task_id === after.task_id && before.task_attempt === after.task_attempt
            && before.run_id === after.run_id && after.revision < before.revision;
    }
    // A typed live observation outranks any later evidence-less dispatch label,
    // even when the route name changes while the same card is reconciled.
    if (before && !after && !next.hasEvidence) return true;
    return Boolean(prior.hasEvidence && !next.hasEvidence);
}
