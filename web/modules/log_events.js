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

// Display-only gate for the agent's reasoning rows (UI preference
// `show_reasoning`, default off). The backend keeps emitting and storing the
// stamped frames either way, so turning it on reveals them on replay too.
// Both reasoning branches below read it and answer with the file's existing
// "not visible" contract (`visible: false`) instead of a new sentinel.
// localStorage mirrors the server preference (as theme.js does) for pre-fetch renders.
const REASONING_STORAGE_KEY = 'ouro.show_reasoning';
function storedReasoningVisible() {
    try { return localStorage.getItem(REASONING_STORAGE_KEY) === '1'; } catch { return false; }
}
let reasoningVisible = storedReasoningVisible();

export const REASONING_VISIBILITY_EVENT = 'ouro:reasoning-visibility';

export function isReasoningVisible() {
    return reasoningVisible;
}

/** Single writer of the flag. It notifies like theme.js/applyTheme does, so a
    control bound before the preference arrives (and the Logs filter chips) can
    resync; outside a DOM (node tests) it is a plain assignment. */
export function setReasoningVisible(value) {
    reasoningVisible = value === true;
    try { localStorage.setItem(REASONING_STORAGE_KEY, reasoningVisible ? '1' : '0'); } catch { /* storage blocked: server value still wins on boot */ }
    if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function'
        && typeof CustomEvent === 'function') {
        window.dispatchEvent(new CustomEvent(REASONING_VISIBILITY_EVENT, {
            detail: { visible: reasoningVisible },
        }));
    }
    return reasoningVisible;
}

export const LOG_CATEGORIES = {
    tools: { label: 'Tools', color: 'var(--blue)' },
    llm: { label: 'LLM', color: 'var(--accent)' },
    errors: { label: 'Errors', color: 'var(--red)' },
    tasks: { label: 'Tasks', color: 'var(--amber)' },
    reasoning: { label: 'Reasoning', color: 'var(--accent)' },
    system: { label: 'System', color: 'var(--text-muted)' },
    consciousness: { label: 'Consciousness', color: 'var(--accent)' },
};

// Logs phases that file a row under the Errors filter (#323). They are the
// typed failure outcomes summarizeLogEvent already derives — a failed task_done,
// a tool that errored/was killed/timed out, an LLM call failure, a review
// lifecycle error — so the category and the phase pill of one row can never
// disagree, live or on replay.
const ERROR_LOG_PHASES = new Set(['error', 'timeout', 'lifecycle_error']);

export function categorizeLogEvent(evt, view = summarizeLogEvent(evt)) {
    const t = evt.type || evt.event || '';
    // A wake-up's rows carry the turn's origin label (`initiator`).
    const wake = evt.initiator === 'consciousness';
    if (evt.is_progress) {
        // A reasoning-stamped row files under its own chip, read off the same
        // projection that paints its `thinking` phase pill.
        if (String(view?.phase || '') === 'thinking') return 'reasoning';
        return wake ? 'consciousness' : 'tasks';
    }
    // Severity comes from the typed projection, never from the event name; the
    // name substrings below only pick the domain family of a non-error row.
    if (ERROR_LOG_PHASES.has(String(view?.phase || ''))) return 'errors';
    if (wake) return 'consciousness';
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
    const raw = evt?.executor_observation;
    const observation = raw && raw.task_id === String(evt?.subagent_task_id || evt?.task_id || '')
        && raw.run_id && raw.attempt_id && raw.harness_id && Number.isInteger(raw.revision)
        ? raw : null;
    if (observation && !evt?.execution_evidence) {
        const name = harnessPresentation(observation.harness_id).label;
        const model = observation.model_source === 'observed' || observation.model_source === 'requested'
            ? compactModel(observation.model) : '';
        return {
            harness: observation.harness_id,
            hasEvidence: false,
            observation,
            sourceTs: Date.parse(evt?.ts || evt?.timestamp || '') || 0,
            label: `${name}${model ? ` · ${model} (${observation.model_source})` : ' · model unconfirmed'} · last update`,
            title: `Latest observed activity: ${name}, run ${observation.run_id}, attempt ${observation.attempt_id}, revision ${observation.revision}. This is progress, not a terminal execution receipt.`,
        };
    }
    const route = String(evt?.executor_route || '').trim();
    if (!route) return null;
    // The route id is OPAQUE (`harness` or `harness=model`): print the harness
    // part only, never interpreted beyond splitting the spelling Claudexor uses.
    const harness = route.split('=')[0].trim().toLowerCase();
    if (!harness) return null;
    const name = harnessPresentation(harness).label;
    const base = { harness, label: name, sourceTs: Date.parse(evt?.ts || evt?.timestamp || '') || 0 };
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
    if (String(evt?.reason_code || '') === 'subagent_executor_unavailable') {
        // The typed $0 terminal of a harness pin the resolution refused (#363):
        // the route names WHO refused, the reason code says the child never
        // ran. Checked before the evidence branches — an empty custody read
        // would otherwise print "no run yet" over a child that was never
        // dispatched at all. Evidence-grade so a later dispatch-shaped frame
        // cannot downgrade it to "dispatched".
        return {
            ...base,
            hasEvidence: true,
            label: `${name} · blocked`,
            title: `Pinned to ${name}, but the route could not run — the task was NOT run (no metered API spend, no delegated run)`,
        };
    }
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
        observedModels: Array.isArray(evidence.harness_models)
            ? [...new Set(evidence.harness_models.filter((model) => typeof model === 'string' && model))] : [],
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
    const modelPart = full && compactModel(model) ? ` · agent model ${compactModel(model)}` : '';
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

// The compact row for one tool call: the command, else the first string
// argument (a path, a query, a url — whatever the tool names first), lexical
// only. The complete arguments stay behind the row's expand.
function toolCallTarget(args) {
    const cmd = extractCommandText(args);
    if (cmd) return cmd;
    for (const value of Object.values(args && typeof args === 'object' ? args : {})) {
        if (typeof value === 'string' && value.trim()) return value;
    }
    return '';
}

// Start, finish, failure and timeout of one call share a row: the call id when
// the producer stamped one, else the tool with its target.
function toolCallKey(evt, groupId) {
    return `tool:${groupId}:${evt.tool_call_id || `${evt.tool || ''}|${toolCallTarget(evt.args)}`}`;
}

const toolObservation = (evt, groupId, status) => ({  // one frame's fact about one invocation
    key: toolCallKey(evt, groupId), status, receipt: Boolean(evt.routing_action), tool: evt.tool || '' });

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

// The typed causes a card can state in the owner's words, keyed on the CODE
// alone. The record keeps the machine code (Logs, task detail, benchmark
// ledgers); only the card speaks. An UNKNOWN code stays raw on purpose: a
// reason we have no sentence for must read as itself rather than as a wrong
// sentence. The byte-identical twin of project_dialogue.TASK_CAUSE_PHRASES;
// web/tests/fixtures/outcome_phase_parity.json pins both.
const TASK_CAUSE_PHRASES = {
    previous_revision_accepted: "The reviewers approved the earlier version of this answer; it changed before they finished.",
    author_finish: "The answer was delivered on Main's own judgement; the reviewers had not signed it off.",
    review_degraded: "No reviewer verdict was established for this answer.",
    infra_failure: "A review infrastructure failure prevented a settled verdict.",
    dialogue_terminal: "The reviewers and Main could not agree, and both positions were kept.",
    improvement_capsule: "The reviewers asked for one more pass and Main was given their notes.",
    fence_reopen_failed: "The requested extra pass could not be started, so the answer stands as it was.",
    review_cycles_exhausted: "The task used up its review rounds before the answer was signed off.",
    open_obligations: "The answer was delivered with reviewer requests still open.",
    improvement_window_closed: "There was no room left for another pass, so the answer stands as it was.",
    capsule_spent: "The one allowed improvement pass was already used.",
    reviewer_fail_no_capsule: "A reviewer rejected the answer and suggested nothing to change.",
    no_actionable_changes: "The re-review was not clean and suggested nothing to change.",
    identical_acceptance_refused: "Nothing had changed since the last review, so the recorded verdict stands.",
    review_skipped_deadline_reserve: "There was not enough time left to review the answer.",
    delivery_binding_superseded: "The answer or its evidence changed, so the earlier review no longer covered it.",
    owner_followup: "A new message from you arrived, so the review was set aside for it.",
    evidence_refresh: "The work changed after the review was frozen, so it no longer covered the answer.",
    revision_unavailable_on_forced_rail: "The task had to stop, so the requested rework never happened.",
    owner_hurry: "You asked me to hurry, so no further review was started.",
    unspecified: "The answer was not signed off, and no cause was recorded.",
    acceptance_bypassed_budget_exhausted: "The task ran out of budget before the answer could be reviewed.",
    acceptance_bypassed_round_limit: "The task hit its round limit before the answer could be reviewed.",
    acceptance_bypassed_deadline: "The task ran out of time before the answer could be reviewed.",
    acceptance_bypassed_provider_unavailable: "The model provider was unavailable, so the answer was never reviewed.",
    acceptance_bypassed_context_overflow: "The task outgrew its context before the answer could be reviewed.",
    acceptance_bypassed_children_unabsorbed: "Some sub-tasks had not been folded in, so the answer was never reviewed.",
    plan_review_advisory: "Plan review never closed; the work continued under advisory enforcement",
    host_child_status_suffix: "A child task had not settled when the answer was delivered",
    invalid_delivery_control_after_repair: "The delivery control object was still malformed after repair",
    budget_exhausted: "The task ran out of budget before it could finish cleanly",
    delivery_control_degraded: "Delivery finished in a degraded control state",
    delegated_custody_unreconciled: "Some delegated work was never reconciled.",
};

export function taskReasonPhrase(code) {
    const raw = String(code || '');
    return TASK_CAUSE_PHRASES[raw] || raw;
}

// The custody overlay stamps this code as the row's reason_code while a
// delegated run is still unreconciled, and the debt then heals from the WRITE
// side while the stored code may not be rewritten. So the code outlives the
// fact, and the debt list the record CARRIES is the only fresh truth. ONE rule
// from ONE source on every surface: a non-empty list names the debt beside the
// execution reason, while an empty list or none at all leaves the execution
// reason standing alone. Nothing is inferred from absence, because the live
// task_done event carries the stored list too (agent_task_pipeline
// ._custody_debt_event_fields), so a record without one is a record that states
// nothing about the debt. The browser twin of
// project_dialogue._custody_debt_reason: the debt is a warning BESIDE the rail
// cause, never a replacement, and any other code passes through untouched.
const CUSTODY_DEBT_REASON = 'delegated_custody_unreconciled';

function custodyDebtReason(record) {
    const raw = String(record?.reason_code || '');
    if (raw !== CUSTODY_DEBT_REASON) return [raw, ''];
    const debt = record?.delegated_runs_unreconciled;
    return [
        String(record?.outcome_axes?.execution?.reason_code || ''),
        Array.isArray(debt) && debt.length ? CUSTODY_DEBT_REASON : '',
    ];
}

// Transport is recorded fact, never proof that an HTTP caller was the owner.
const CANCEL_SOURCE_PHRASES = {
    http_single: 'Stopped from the app (Stop now)',
    http_cascade: 'Stopped from the app (Stop now)',
    http_graceful: 'Stopped from the app (Wrap up)',
};

export function taskReasonDetail(evt) {
    // An owner-requested stop is a success and carries its own marker instead.
    if (taskStoppedWithSummary(evt)) return '';
    // A warning caused by REVIEW must not be explained by the execution reason
    // that happens to sit beside it: the host's acceptance decision is the
    // cause, and it speaks in its own stored words. A hard failure or a
    // cancellation keeps explaining itself by its execution reason.
    const record = normalizeTaskTerminalRecord(evt);
    const decision = record.outcome_axes?.review?.acceptance_decision
        ?? record.review_status?.acceptance_decision;
    const severity = taskOutcomeSeverity(evt);
    const decisionCause = String(decision?.reason || '');
    if (severity !== 'error' && severity !== 'cancelled' && decision?.status
        && (decision.status !== 'accepted' || Object.hasOwn(TASK_CAUSE_PHRASES, decisionCause))) {
        // The decision's own typed reason speaks (an accepted decision only when
        // it has a sentence); the stored reviewer rationale stays in the card
        // body, the task result and Logs.
        return taskReasonPhrase(decisionCause);
    }
    const origin = record.cancel_origin;
    if (severity === 'cancelled' && origin && typeof origin === 'object' && !Array.isArray(origin)) {
        const source = String(origin.source || '');
        const actor = origin.request_origin?.kind === 'agent_task' ? origin.request_origin.task_id : '';
        return [
            Object.hasOwn(CANCEL_SOURCE_PHRASES, source) ? CANCEL_SOURCE_PHRASES[source] : source,
            origin.scope === 'cascade' ? 'this task and its sub-tasks' : '',
            `initiator: ${String(actor || origin.requested_by || '') || 'not recorded'}`,
        ].filter(Boolean).join(' · ');
    }
    if (!evt?.reason_code || evt.reason_code === 'final_message') return '';
    // A healed debt is never restored: naming it again would state a debt the
    // same record shows as empty. The current execution reason speaks when
    // there is one, otherwise the row states no cause and leaves the headline
    // to the frozen outcome axis that owns it.
    const [reason, custody] = custodyDebtReason(record);
    if (!reason) return taskReasonPhrase(custody);
    const receiptVeto = record.outcome_axes?.objective?.receipt_veto;
    const cause = receiptVeto?.reason === reason && receiptVeto.detail
        ? String(receiptVeto.detail).split(/\s+/).filter(Boolean).join(' ')
        : taskReasonPhrase(reason);
    return `${cause}${custody ? ` (${taskReasonPhrase(custody)})` : ''}`;
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
        // Display off + a reasoning-only round's frame (`narration: true`): ordinary row.
        if (evt.reasoning === true && (reasoningVisible || evt.narration !== true)
            && (!isSubagentEvent(evt) || !reasoningVisible)) {
            const thinking = view('thinking', 'Thinking', {
                body: shortText(String(evt.content || evt.text || '').replace(/^💬\s*/, ''), 240),
                meta: taskMeta(),
            });
            // Hidden by default: the durable row stays in the log, it just
            // renders no entry until the owner turns the display on. A subagent's
            // reasoning frame carries the lineage stamps as well, so while the
            // display is off it is intercepted here too — otherwise the subagent
            // branch below would render it as an ordinary row and escape the
            // preference. With the display on it keeps falling through to that
            // branch, where the child's work reads as one lineage-labelled row.
            return reasoningVisible ? thinking : { ...thinking, visible: false };
        }
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
            meta: taskMeta(
                evt.model || '',
                formatLogTokens(evt),
                // ABI-3: /api/logs backfill rows carry the honest name; live
                // frames still say cost_usd/cost — resolve the pair via the
                // SSOT helper, then the live-frame `cost` spelling.
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
            meta: taskMeta(
                evt.model || '',
                formatLogTokens(evt),
                formatLogMoney(accountedUpperBound(evt) ?? evt.cost),
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
        // The durable tools.jsonl row (replay/backfill) carries the same typed
        // failure facts as the live tool_call_finished frame — is_error plus the
        // signal/exit facts — so a failed call reads the same after a reload.
        const signalDeath = Boolean(evt.signal) || (typeof evt.exit_code === 'number' && evt.exit_code < 0);
        const failed = Boolean(evt.is_error) || signalDeath;
        const label = signalDeath ? `killed (${evt.signal || evt.exit_code})` : failed ? 'failed' : 'result';
        return view(failed ? 'error' : 'result', `${evt.tool || 'tool'} ${label}`, {
            body: shortText(evt.result_preview || compactJson(evt.args, 220), 260),
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
        const presentation = taskPresentation(terminal || outcome === 'error' ? outcome : 'working');
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
        evt.error || evt.message || evt.text || evt.result_preview
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
    model = '',
    receipt = false,
    toolCall = null,
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
    // A receipt row renders inside a block but is not content the block can
    // stand on: the fact it reports lives elsewhere (the owner message's
    // routing annotation for an addressing call).
    if (receipt) out.receipt = true;
    if (toolCall) out.toolCall = toolCall;  // folded into the block's one evidence row
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
    const presentation = taskPresentation(terminal || outcome === 'error' ? outcome : 'working');
    const body = [taskStoppedWithSummary(evt) ? OWNER_STOP_DETAIL_MARKER : '', taskReasonDetail(evt)]
        .filter(Boolean).join('\n');
    return {
        ...chatView({
            phase: presentation.phase, headline: presentation.headline, body,
            visible: true, promote: true, terminal,
            dedupeKey: `task_done|${evt.task_id || ''}`,
        }),
        ...(evt.model_execution && typeof evt.model_execution === 'object'
            ? { modelExecution: evt.model_execution } : {}),
        ...(Number.isInteger(evt.tool_calls) ? { toolCalls: evt.tool_calls } : {}),
        ...(evt.initiator ? { initiator: String(evt.initiator) } : {}),
    };
}

// Requested route, usable solve route and provider-reported name are distinct.
export function modelExecutionLabel(fact) {
    if (!fact || typeof fact !== 'object') return '';
    const requested = compactModel(fact.requested_model || '');
    if (fact.source !== 'usable_solve_response') {
        return requested ? `Requested ${requested} · execution not observed` : 'Execution not observed';
    }
    const used = String(fact.used_model || '');
    const reported = String(fact.reported_model || '');
    const requestDiffers = String(fact.requested_model || '') !== used
        || fact.requested_use_local !== fact.used_local;
    const initial = requested && requestDiffers
        ? ` (initial request: ${requested}${fact.requested_use_local ? ' · local' : ''})` : '';
    return [`Last solve response: ${compactModel(reported || used)}${initial}`,
        fact.used_local === true ? 'local' : '', fact.provider || '',
        reported && reported !== used ? `route: ${used}` : ''].filter(Boolean).join(' · ');

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

    if ((evt.is_progress || t === 'send_message') && evt.reasoning === true
        && (reasoningVisible || evt.narration !== true)
        && (!reasoningVisible || !isSubagentEvent(evt))) {
        // The agent's own reasoning: a collapsed "Thinking" timeline line (body =
        // preview, fullBody = the whole text for the existing Expand toggle). It is
        // neither human narration nor promoted, so the card headline, phase and the
        // collapsed activity summary keep showing the last action.
        // A subagent's frame carries the reasoning stamp AND the lineage stamps, so
        // this branch must run before the subagent branch below: while the display
        // is off it claims the frame and renders nothing, and with the display on it
        // hands the frame over, keeping the child's progress collapsed into its card
        // line instead of a second Thinking row. `summarizeLogEvent` splits on the
        // same condition.
        return chatView({
            phase: 'thinking',
            headline: 'Thinking',
            body: progressText.preview,
            fullBody: progressText.full,
            activityPreview: '',
            // Hidden by default; the frame keeps its body so turning the
            // display on renders the same line, live and on history replay.
            visible: reasoningVisible,
            dedupeKey: `reasoning:${evt.ts || ''}:${progressText.full}`,
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
        const cancelDetail = evt.cancel_origin && (rawEvent === 'cancelled'
            || (rawEvent === 'completed' && taskOutcomeSeverity(evt) === 'cancelled'))
            ? taskReasonDetail({ status: 'cancelled', cancel_origin: evt.cancel_origin }) : '';
        const reasonDetail = cancelDetail || (evt.reason_code ? taskReasonPhrase(evt.reason_code) : '');
        const detailParts = [
            progressText.full,
            resultText.full ? `[RESULT]\n${resultText.full}` : '',
            traceText.full ? `[TRACE]\n${traceText.full}` : '',
            errorText.full ? `[ERROR]\n${errorText.full}` : '',
            reasonDetail,
        ].filter(Boolean);
        // A generic "completed" event still carries authoritative outcome axes: normalize it
        // once here so every live/replay route takes label, phase and terminal truth from the
        // canonical projector.
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
                            : progressText)
            : (progressText.full ? progressText
                : resultText.full ? resultText
                    : errorText.full ? errorText
                        : traceText);
        return chatView({
            phase,
            headline: subagentHeadline(sid, role, label, evt.model),
            body: cancelDetail || activity.preview || '',
            fullBody: detailParts.join('\n\n'),
            activityPreview: cancelDetail || activity.preview || '',
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
        return chatView({
            phase: lifecycleTerminal ? (/failed\b/i.test(progressText.full) ? 'lifecycle_error' : 'done') : 'working',
            headline: progressText.preview || 'Working...',
            fullHeadline: progressText.full || '',
            activityPreview: progressText.preview || '',
            visible: Boolean(progressText.preview),
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

    if (t === 'task_message_injected') {
        // A message from another task landed in this task's transcript: a
        // visible row in the receiver's block (owner 5=A), named by value.
        const source = evt.source_task_id ? String(evt.source_task_id) : 'another task';
        const preview = String(evt.text_preview || '');
        return chatView({
            phase: 'info',
            headline: `Message from task ${source}`,
            body: shortText(preview, 200),
            fullBody: preview,
            visible: true,
            dedupeKey: key(source, evt.ts || ''),
        });
    }

    if (t === 'tool_call_started' || (t === 'tool_call_finished' && !evt.is_error)) {
        // A successful call is execution evidence, not narration: start and finish feed the
        // block's ONE folded row (counts; tools behind Expand), a receipt while every counted
        // call is a host-stamped addressing act (`routing_action`, reported by the owner
        // message's annotation). A failure keeps its own error row and still counts. `done` is
        // the TASK's phase; a finished CALL is `ok`.
        const status = t === 'tool_call_finished' ? 'ok' : 'calling';
        return chatView({
            phase: status,
            headline: '',
            visible: true,
            receipt: Boolean(evt.routing_action),
            dedupeKey: `tools|${groupId}`,
            toolCall: toolObservation(evt, groupId, status),
        });
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
            headline: `One of the steps took too long${evt.tool ? ` · ${evt.tool}` : ''}`,
            visible: true,
            dedupeKey: toolCallKey(evt, groupId),
            toolCall: toolObservation(evt, groupId, 'error'),
        });
    }

    if (t === 'tool_call_finished' && evt.is_error) {
        const failed = toolObservation(evt, groupId, 'error');
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
                dedupeKey: toolCallKey(evt, groupId),
                toolCall: failed,
            });
        }
        return chatView({
            phase: 'error',
            headline: `One of the steps failed${evt.tool ? ` · ${evt.tool}` : ''}`,
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
