// Activity dashboard subtab (P4): a single observability + minimal-control view for
// cron/scheduled tasks, what is running/queued now, and background consciousness.
// Management is DIRECT mechanical control via existing APIs (cancel a task, enable/
// disable/delete/restore schedules, start/stop background consciousness). Every
// lifecycle button NAMES its action to the one audited server seam, so the owner's
// buttons and the agent's tool mean exactly the same thing.

import { fetchJson, isStoreMigrating, readStateJson } from './api_client.js';
import { setInlineStatus } from './ui_helpers.js';
import { fmt, tr } from './i18n.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { taskCancelPending } from './log_events.js';
import { waitingModel } from './project_activity.js';
import { memberState, openWorkControls, resumeTargets, workLine, workOfEpisode, workRowView } from './work_status.js';
import { censusTaskFacts, censusTaskPhase, paintTaskPhase } from './task_phase_chip.js';
import {
    ACTION_HURRY,
    ACTION_RESUME,
    TASK_CONTROL_TRIGGER_LABEL,
    heldWorkFacts,
    hurryTaskAction,
    isRootTaskRow,
    openTaskControlMenu,
    requestStop,
    resumeTaskAction,
    taskControlBusy,
} from './task_control_menu.js';
import { showToast } from './toast.js';
import { allowanceLabel } from './utils.js';

function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, (c) => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

const getJson = (url) => fetchJson(url, { cache: 'no-store' });

/** How long a task has run, in words (DESIGN §6: quantities in human words): «45s» below a minute, then «9 min»,
 * «1 h 30 min»; never «1700s». A duration the row does not carry is '' — unknown is never «0s». */
export function elapsedWords(seconds) {
    const value = Number(seconds);
    if (seconds === null || seconds === '' || !Number.isFinite(value)) return '';
    const total = Math.max(0, Math.round(value));
    if (total < 60) return `${total}s`;
    const minutes = Math.round(total / 60);
    if (minutes < 60) return fmt('{n} min', { n: minutes });
    const hours = Math.floor(minutes / 60);
    const rest = minutes % 60;
    return rest ? fmt('{hours} h {minutes} min', { hours, minutes: rest }) : fmt('{n} h', { n: hours });
}

// Census phases that only say the task is still active.
const ACTIVE_PHASES = new Set(['working', 'thinking', 'finalizing']);

// A row's OWN census facts (its activity_id, never its root's: a child must not borrow a
// parent's result) as the shared chip reads them. Activity shows no model-call telemetry, so
// `model_activity` never reaches it.
function ownCensusView(activity, withHold = true) {
    const { model_activity: _telemetry, ...own } = activity;
    if (!withHold) delete own.project_admission_hold;  // the queue line already names it
    return { own, view: censusTaskPhase(own, null, true, waitingModel(own)) };
}
// Those words in one line: the chip's word, then the quieter fact it says beside it.
function censusWords(activity) {
    const { view } = ownCensusView(activity);
    return [view.text, view.secondary].filter(Boolean).join(' · ');
}

/** The shared chip for a row whose own census facts know its outcome; null keeps the row's own
 * word. The outcome is the status and an unfinished fact its quieter second one (DESIGN "Status
 * and chips"). A queue row passes its own wait or hold word (`queueFact`, '' for plain running),
 * which outranks a merely active census phase; membership and controls stay with the queue
 * and census. */
export function ownOutcomePhase(activity, queueFact = null) {
    const facts = activity ? censusTaskFacts(activity) : null;
    if (!facts?.outcome || !(facts.ended || facts.finalizing)) return null;
    const { own, view } = ownCensusView(activity, queueFact === null);
    return queueFact && (!view.secondary || (ACTIVE_PHASES.has(own.phase) && !view.waiting))
        ? { ...view, secondary: queueFact } : view;
}

/** An answered root whose late work remains (its upkeep, or a review its Pause deferred) names
 * its own work, when it answered, and that only the work after the answer waits — never a generic turn. */
function lateWorkHtml(a, late) {
    const at = new Date(String(late.answered_at || ''));
    const iso = Number.isNaN(at.getTime()) ? '' : at.toISOString();
    const when = iso ? ` <time datetime="${esc(iso)}" title="${esc(iso)}">${esc(at.toLocaleString([], {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }))}</time>` : '';
    const state = { budget_paused: 'paused', budget_pausing: 'pausing', unknown: 'status unknown' }[a.phase] || 'running';
    // The state leads: a narrow row's ellipsis may cut the time, never what waits.
    return { name: esc(late.title || 'Answered task'), sub: `Remaining work ${state} · answer delivered${when}` };
}

/** A census-only live root (a direct turn, or an answered root whose late work remains):
 * its state in the chip's own words (never the census code), and the shared control —
 * Resume only where its work's Resume control targets it (`resume`, work_status.resumeTargets).
 * `slot` marks where the caller paints a known outcome (`ownOutcomePhase`). */
export function liveActivityRowHtml(a, nowMs = Date.now(), slot = null, resume = false) {
    const names = { direct_chat: 'Direct turn', managed_task: 'Managed task' };
    const started = Number(a.started_at) || 0;
    const spent = started > 0 ? elapsedWords(nowMs / 1000 - started) : '';
    const elapsed = spent ? ` · ${spent}` : '';
    const late = a.late_work && typeof a.late_work === 'object' ? lateWorkHtml(a, a.late_work) : null;
    const outcome = !late && slot && ownOutcomePhase(a);
    return `<div class="activity-row">
                <div class="activity-row-main">
                    <span class="activity-name">${late ? late.name : esc(names[a.kind] || 'Live turn')}</span>
                    <span class="activity-sub">${late ? late.sub : `${outcome ? slot(outcome)
                        : `<span data-i18n-skip>${esc(censusWords(a))}</span>`}${elapsed}`}</span>
                </div>
                <div class="activity-row-actions">
                    <button type="button" class="btn btn-xs btn-default" data-act="task-control" data-id="${esc(a.activity_id || '')}" data-root="1"${a.phase === 'budget_paused' ? ' data-budget-paused="1"' : ''}${resume ? ' data-resume="1"' : ''}>${esc(TASK_CONTROL_TRIGGER_LABEL)}</button>
                </div>
            </div>`;
}

/** A stored UTC schedule instant for the owner: this viewer's local time, the exact
 * UTC instant beside it (and in `datetime`/`title`). Records stay UTC; an unparseable
 * value is shown raw rather than guessed. `timeZone` exists for tests only. */
export function scheduleInstantHtml(value, { timeZone, includeYear = false } = {}) {
    const raw = String(value || '');
    const parsed = new Date(raw);
    if (!raw || Number.isNaN(parsed.getTime())) return esc(raw);
    const fields = { ...(includeYear ? { year: 'numeric' } : {}), month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    const local = parsed.toLocaleString([], { ...fields, timeZone, timeZoneName: 'short' });
    const utc = parsed.toLocaleString([], { ...fields, timeZone: 'UTC' });
    const exact = parsed.toISOString().replace('.000Z', 'Z');
    return `<time datetime="${esc(exact)}" title="${esc(exact)}">${esc(local)} (${esc(utc)} UTC)</time>`;
}

// The server names the lifecycle word; these fall back for an older payload
// so a row is never silently promoted to "active" by a missing field.
export function scheduleStatus(s) {
    const once = String((s.trigger || {}).type || 'cron') === 'once';
    if (s.status) return String(s.status);
    if (s.delete_requested_at) return 'delete_pending';
    if (once && s.completed_at) return 'consumed';
    return s.enabled === false ? 'disabled' : 'active';
}

// Known work in flight keeps a withdrawn row in the standing list. Other
// withdrawn rows and fired one-shots remain reachable in history.
export function isRetainedSchedule(s) {
    const status = scheduleStatus(s);
    return status === 'consumed'
        || (status === 'delete_pending' && Boolean(s.withdrawal) && s.withdrawal.work_in_flight !== true);
}

function scheduleTextHtml(s, note, expanded) {
    const text = String((note ? s.notification?.text : s.task?.text || s.task?.objective)
        || s.description || s.objective_preview || '').trim();
    if (!text) return '';
    const preview = text.split(/\r?\n/)[0].slice(0, 96);
    if (preview === text) return `<span class="activity-sub" data-i18n-authored>${esc(text)}</span>`;
    return `<details class="project-ending-details activity-sub" data-schedule-text="${esc(s.id || '')}"${expanded ? ' open' : ''}>
        <summary data-act="schedule-text" data-id="${esc(s.id || '')}" data-i18n-authored>${esc(preview)}…</summary>
        <div data-i18n-authored>${esc(text).replace(/\r?\n/g, '<br>')}</div>
    </details>`;
}

function scheduleRunHtml(s) {
    const outcome = s.outcome;
    const parts = [];
    if (outcome?.state === 'pending_observation') {
        parts.push('last run: outcome not observed yet');
    } else if (outcome?.state) {
        const quiet = outcome.quiet === true && outcome.notice !== true;
        const state = quiet && outcome.state === 'completed' ? tr('schedule.outcome.quiet', 'ran · no message') : outcome.state;
        // The line is the host's notice, else the run's own words (its sentence, or the end of its answer).
        const line = outcome.line ? ` · ${outcome.notice ? `notice: ${esc(outcome.line)}` : `<span data-i18n-authored>${esc(outcome.line)}</span>`}` : '';
        parts.push(`last run: ${esc(state)}${outcome.ended_at ? ` · ${scheduleInstantHtml(outcome.ended_at, { includeYear: true })}` : ''}${line}`);
    }
    if (s.previous_run_open) {
        parts.push(`previous run still open${s.previous_run_open.task_id ? `: ${esc(s.previous_run_open.task_id)}` : ''}`);
    }
    return parts.map((line) => `<span class="activity-sub">${line}</span>`).join('');
}

export function scheduleRowHtml(s, expanded = false) {
    const trigger = s.trigger || {};
    const once = String(trigger.type || 'cron') === 'once';
    // One-shot rows have no cron: show the fire instant + a "one-shot" tag. A cron
    // expression runs in its record's zone, or the server's when none is stored.
    const timing = once
        ? `one-shot · at/after ${scheduleInstantHtml(trigger.run_at)}`
        : `${esc(trigger.expr || s.cron || '')} (${s.timezone ? esc(s.timezone) : 'server time zone'})`;
    const note = ['note', 'notify'].includes(s.kind);
    const next = s.next_run_at ? scheduleInstantHtml(s.next_run_at) : '';
    const status = scheduleStatus(s);
    const enabled = status === 'active';
    const consumed = status === 'consumed';
    const pendingDelete = status === 'delete_pending';
    const id = esc(s.id || '');
    // A due occurrence that waits (capacity, a missing folder, an unknown fact) says why.
    const wait = s.waiting_detail ?? s.hold ?? s.waiting;
    const waitText = typeof wait === 'string' ? wait : [wait?.reason, wait?.detail].filter(Boolean).join(': ');
    const waiting = waitText ? ` · <span class="activity-tag">waiting: ${esc(waitText)}</span>` : '';
    // Why the row did not do its job, as stored (a note whose chat write was not
    // confirmed is consumed and never retried, so this is the only place it says so).
    const failed = s.last_error ? ` · <span class="activity-tag">last error: ${esc(s.last_error)}</span>` : '';
    const relation = String(s.relation || 'unknown');
    const hold = s.followup_hold || {};
    const work = s.billing_group || {};
    const binding = work.billing_group_id ? ` · work ${esc(work.billing_group_id)} · cap ${work.billing_group_limit_usd == null ? 'unbounded' : '$' + esc(work.billing_group_limit_usd)}` : '';
    const explanations = {
        origin_stopped: 'Original work stopped or restarted', relationship_unknown: 'Relationship needs a decision',
        origin_owner_paused: 'Original work is paused', owner_restart_in_progress: 'Restart in progress',
        followup_authority_unavailable: 'Original work details unavailable', work_deadline_passed: 'Original hard deadline passed',
        pending_binding_unavailable: 'Fired task awaits verified work binding',
    };
    const followupWait = s.waiting_detail?.followup_wait || s.followup_wait;
    const withdrawal = s.withdrawal;
    const statusLabel = !pendingDelete ? status : withdrawal?.work_in_flight === true
        ? 'deletion pending · work in flight' : withdrawal?.restorable === true
            ? 'withdrawn · restorable' : withdrawal ? 'withdrawn' : 'deletion pending';
    const destination = s.destination || s.task || {};
    const address = `${destination.project_id ? ` · project ${esc(destination.project_id)}` : ''}${destination.chat_id != null ? ` · chat ${esc(destination.chat_id)}` : ''}`;
    const sub = `${note ? 'reminder · ' : s.kind === 'task' ? 'task · ' : ''}${timing}${next && !consumed && !pendingDelete ? ` · next ${next}` : ''} · ${esc(statusLabel)} · ${esc(relation)}${binding}${address}${s.deadline_at ? ` · deadline ${scheduleInstantHtml(s.deadline_at, { includeYear: true })}` : ''}${hold.reason ? ` · ${esc(explanations[hold.reason] || hold.reason)}${hold.detail ? `: ${esc(hold.detail)}` : ''}` : ''}${followupWait ? ` · ${esc(explanations[followupWait] || followupWait)}` : ''}${s.completed_at ? ' · already fired' : ''}${s.skill ? ` · from skill ${esc(s.skill)}` : ''}${waiting}${failed}`;
    const withdrawalDetail = withdrawal ? [
        withdrawal.at ? `withdrawn ${scheduleInstantHtml(withdrawal.at, { includeYear: true })}` : '',
        withdrawal.by ? `by ${esc(withdrawal.by)}` : '',
        withdrawal.reason ? esc(withdrawal.reason) : '',
    ].filter(Boolean).join(' · ') : '';
    // The exact hold release stays available wherever a hold exists, including on a
    // deleted row whose accepted task still needs it; it never re-enables the row.
    const holdControl = !hold.hold_id ? ''
        : relation === 'unknown'
            ? '<span class="activity-tag">resolve relationship in conversation</span>'
            : !s.hold_persisted
                ? '<span class="activity-tag">hold persistence pending</span>'
                : `<button type="button" class="btn btn-xs btn-default" data-act="schedule-toggle" data-id="${id}" data-action="restore" data-hold-id="${esc(hold.hold_id || '')}">Restore hold</button>`;
    // Restore cancels a withdrawal the server calls restorable (not fired, not held);
    // exact hold release is a separate action.
    const canRestore = !hold.hold_id && (withdrawal ? withdrawal.restorable === true : !(once && s.completed_at));
    const restoreWithdrawal = canRestore
        ? `<button type="button" class="btn btn-xs btn-default" data-act="schedule-toggle" data-id="${id}" data-action="restore">Restore</button>` : '';
    const lifecycle = pendingDelete
        ? `${holdControl}${restoreWithdrawal}`
        : status === 'waiting'
            ? '<span class="activity-tag">waiting for existing control</span>'
            : hold.hold_id
                ? holdControl
                : consumed
                    ? '<span class="activity-tag">consumed once · history</span>'
                    : `<button type="button" class="btn btn-xs btn-default" data-act="schedule-toggle" data-id="${id}" data-action="${enabled ? 'disable' : 'restore'}">${enabled ? 'Disable' : 'Enable'}</button>`;
    return `<div class="activity-row activity-schedule">
        <div class="activity-row-main">
            <span class="activity-name" data-i18n-authored>${esc(s.display_name || s.name || s.id || 'schedule')}</span>
            ${scheduleTextHtml(s, note, expanded)}
            <span class="activity-sub">${sub}</span>
            ${scheduleRunHtml(s)}
            ${withdrawalDetail ? `<span class="activity-sub">${withdrawalDetail}</span>` : ''}
        </div>
        <div class="activity-row-actions">${lifecycle}
           <button type="button" class="btn btn-xs btn-danger" data-act="schedule-delete" data-id="${id}" data-note="${note ? '1' : ''}">Delete</button></div>
    </div>`;
}

// What one lifecycle response means for the owner: ``[text, level]`` or null.
// The response says two separable things: whether the schedule CHANGED, and
// whether both audit facts landed. A change whose outcome record was lost is
// neither a clean success nor a failure, and saying either would be wrong.
export function scheduleOutcomeToast(action, outcome) {
    if (!outcome || typeof outcome !== 'object' || typeof outcome.changed !== 'boolean') {
        return [`Schedule ${action}: the server did not say what happened.`, 'error'];
    }
    const detail = String(outcome.detail || outcome.status || 'no status reported');
    if (outcome.changed === false) return [`Schedule ${action} did not change anything: ${detail}`, 'error'];
    // Restriction refresh can change bookkeeping while REFUSING release.
    // Only typed applied outcomes may produce an applied/success message.
    if (!['updated', 'withdrawn', 'deleted', 'delete_deferred',
        'hold_released', 'changed_audit_incomplete'].includes(outcome.status)) {
        return [`Schedule ${action} refused: ${detail}`, 'error'];
    }
    if (outcome.audit !== 'recorded') return [`Schedule ${action} applied, but its audit record is incomplete: ${detail}`, 'warn'];
    if (outcome.status === 'delete_deferred') return [`Schedule deletion is pending: ${detail}`, 'info'];
    if (outcome.ok !== true) return [`Schedule ${action} changed, but did not finish successfully: ${detail}`, 'warn'];
    const done = outcome.status === 'withdrawn' ? 'withdrawn' : `${action}d`;
    // Lifecycle actions govern FUTURE dispatch; a run already admitted keeps
    // going. Silence here would let the owner read the button as a stop.
    if (outcome.running_or_queued === true) {
        return [`Schedule ${done}. A task it already admitted is queued or running and was not cancelled.`, 'info'];
    }
    if (outcome.running_or_queued === null || outcome.running_or_queued === undefined) {
        return [`Schedule ${done}. Whether a task it already started is still running is unknown.`, 'info'];
    }
    return outcome.status === 'withdrawn' ? [`Schedule ${done}.`, 'info'] : null;
}

export function initActivity({ mount, ws, openDashboardTab = null } = {}) {
    if (!mount) return { refresh: () => {} };
    let busy = false;
    let refreshRevision = 0;
    let migrationTimer = null, destroyed = false;
    // Remembered across re-renders: the owner's retained-history disclosure state.
    let historyOpen = false;
    let expandedSchedules = new Set();
    mount.innerHTML = `<div class="activity-scroll">
        <div class="activity-section" data-activity-section="queue"><h3 class="activity-h">Running &amp; queued</h3></div>
        <div class="activity-section" data-activity-section="background"><h3 class="activity-h">Background</h3></div>
        <div class="activity-section" data-activity-section="schedules"><h3 class="activity-h">Scheduled</h3></div>
    </div>`;
    const sections = ['queue', 'background', 'schedules'].map((name) => {
        const root = mount.querySelector(`[data-activity-section="${name}"]`);
        const status = document.createElement('div');
        status.className = 'ui-status activity-read-status';
        status.setAttribute('role', 'status');
        const content = document.createElement('div');
        content.className = 'activity-section-content';
        root.append(status, content);
        return { root, status, content, loaded: false };
    });
    // Known-outcome chips of the last queue render; paintTaskPhase writes them into
    // their slots once the markup is mounted. The words come from the translation seam.
    let phaseViews = [];
    // The body the queue was last drawn from: a work's controls come from it, at their exact targets.
    let lastCensus = null;
    const phaseSlot = (view) => `<span class="activity-phase" data-i18n-skip><span data-activity-phase="${
        phaseViews.push(view) - 1}"></span><span class="chat-live-phase-secondary" hidden></span></span>`;

    function renderQueue(queue, census) {
        if (!Array.isArray(queue?.running) || !Array.isArray(queue?.pending)) throw new Error('Queue unavailable');
        lastCensus = census;
        const { running, pending } = queue;
        const activities = Array.isArray(census?.active_chat_activities) ? census.active_chat_activities : [];
        const phases = new Map(activities.map((a) => [String(a?.activity_id || ''), a?.phase]));
        // Resume is offered only on a task a work's Resume control names (the server's door decision).
        const resumable = resumeTargets(census);
        phaseViews = [];
        const row = (q, kind) => {
            const t = (q && q.task) || {};
            const id = esc(q.id || t.id || '');
            const label = esc(t.title || t.objective || t.text || q.type || id || 'task');
            const spent = kind === 'running' ? elapsedWords(q.runtime_sec) : '';
            const rt = spent ? ` · ${spent}` : '';
            const ownId = String(q.id || t.id || '');
            const projected = workOfEpisode(census, ownId) || memberState(census, ownId);
            const view = workRowView(projected || { activity: 'unknown', confirmed: false });
            const parked = projected?.activity === 'paused';
            const kindLabel = view.word;
            const outcome = ownId && ownOutcomePhase(activities.find((entry) => entry?.activity_id === ownId),
                view.outcome || (projected?.activity === 'working' && projected.transition === 'none') ? '' : kindLabel);
            // The chip already says Finalizing beside its outcome; retain only the work's other facts.
            const secondary = outcome && view.outcome ? workLine(projected, kindLabel) : view.secondary;
            // The projection owns the state; the current wait's recorded blockers
            // keep its specific cause readable after the Continue toast expires.
            const waiting = t._budget_pause_hold;
            const heldFacts = projected?.confirmed !== false && projected?.because?.[0]?.code === 'earlier_work_unconfirmed'
                && waiting?.reason === 'continuation_writer_unsettled' && !waiting.selected ? heldWorkFacts(waiting.blockers) : '';
            const meta = `${outcome ? phaseSlot(outcome) : esc(kindLabel)}${secondary ? ` · ${esc(secondary)}` : ''}${heldFacts ? ` · ${esc(heldFacts)}` : ''}${q.type ? ` · ${esc(q.type)}` : ''}${rt}`;
            const hold = t.project_admission_hold;
            const projectWait = hold?.label ? ` · ${esc(hold.label)}: ${esc(hold.detail || hold.reason)}` : '';
            // The held cause is what the owner reads before Resume: it wraps, never ellipsized.
            return `<div class="activity-row${projected?.activity === 'waiting' ? ' activity-held' : ''}">
                <div class="activity-row-main">
                    <span class="activity-name">${label}</span>
                    <span class="activity-sub">${meta}${projectWait}</span>
                </div>
                <div class="activity-row-actions">
                    <button type="button" class="btn btn-xs btn-default" data-act="task-control" data-id="${id}"${parked || resumable.has(ownId) ? ' data-budget-paused="1"' : ''}${resumable.has(ownId) ? ' data-resume="1"' : ''}${phases.has(ownId) || isRootTaskRow(t, ownId) ? ' data-root="1"' : ''}>${esc(TASK_CONTROL_TRIGGER_LABEL)}</button>
                </div>
            </div>`;
        };
        // Queue facts keep their runtime/budget controls. The census adds only
        // missing identities; queued direct turns can appear in both sources.
        const known = new Set([...running, ...pending].map((q) => String(q.id || q.task?.id || '')));
        const live = activities.filter((a) => a && !known.has(String(a.activity_id || '')));
        const parts = [...running.map((q) => row(q, 'running')), ...pending.map((q) => row(q, 'pending')),
            ...live.map((a) => liveActivityRowHtml(a, Date.now(), phaseSlot, resumable.has(String(a.activity_id || ''))))];
        if (parts.length) return parts.join('');
        return census?.active_chat_activities_complete === true
            ? '<div class="activity-empty">Nothing running or queued.</div>'
            : '<div class="activity-empty">Queue empty; live turns unknown.</div>';
    }

    function formatWhen(value) {
        if (!value) return '';
        const parsed = new Date(value);
        return Number.isNaN(parsed.getTime()) ? '' : parsed.toLocaleString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' });
    }

    function renderBg(stateData) {
        if (typeof stateData?.bg_consciousness_enabled !== 'boolean') throw new Error('Background state unavailable');
        const enabled = stateData.bg_consciousness_enabled;
        const bg = (stateData && stateData.bg_consciousness_state) || {};
        const detail = esc(bg.detail || (enabled ? 'enabled' : 'disabled'));
        const facts = [
            bg.level ? `autonomy ${esc(bg.level)}` : '',
            bg.last_wake_at ? `last wake ${esc(formatWhen(bg.last_wake_at))}${bg.last_wake_outcome ? ` (${esc(bg.last_wake_outcome)})` : ''}` : '',
            allowanceLabel(bg.spent_24h_usd, bg.daily_usd, bg.unknown_unmetered, bg.integrity_degraded)
                ? `allowance ${esc(allowanceLabel(bg.spent_24h_usd, bg.daily_usd, bg.unknown_unmetered, bg.integrity_degraded))} (24 h)` : '',
            Number.isFinite(Number(bg.max_tasks)) ? esc(fmt('own tasks {running}/{max}', { running: Number(bg.tasks_running || 0), max: Number(bg.max_tasks) })) : '',
        ].filter(Boolean).join(' · ');
        return `<div class="activity-row">
            <div class="activity-row-main">
                <span class="activity-name">Consciousness</span>
                <span class="activity-sub">${enabled ? 'enabled' : 'disabled'}${detail ? ` · ${detail}` : ''}${facts ? ` · ${facts}` : ''}</span>
            </div>
            <div class="activity-row-actions">
                <button type="button" class="btn btn-xs btn-default" data-act="bg-toggle" data-enabled="${enabled ? '1' : '0'}"${ws ? '' : ' disabled'}>${enabled ? 'Stop' : 'Start'}</button>
                <button type="button" class="btn btn-xs btn-ghost" data-act="bg-open" aria-label="Open Dashboard › Consciousness"${openDashboardTab ? '' : ' hidden'}>Details</button>
            </div>
        </div>`;
    }

    function renderSchedules(data) {
        if (!Array.isArray(data?.tasks)) throw new Error('Schedules unavailable');
        const tasks = data.tasks;
        if (!tasks.length) return '<div class="activity-empty">No scheduled tasks.</div>';
        // History stays reachable without padding the standing schedules.
        const standing = [];
        const retained = [];
        for (const s of tasks) (isRetainedSchedule(s) ? retained : standing).push(s);
        const parts = standing.length
            ? standing.map((s) => scheduleRowHtml(s, expandedSchedules.has(String(s.id || ''))))
            : ['<div class="activity-empty">No active or disabled schedules.</div>'];
        if (retained.length) {
            // A refresh rebuilds this markup, so the disclosure carries the state
            // the owner LEFT it in: a poll that silently re-collapses the history
            // they just opened (and drops the focus they had inside it) reads as
            // the app undoing their action.
            parts.push(`<details class="activity-history" data-activity-history${historyOpen ? ' open' : ''}>
                <summary>History &amp; withdrawn (${retained.length})</summary>
                ${retained.map((s) => scheduleRowHtml(s, expandedSchedules.has(String(s.id || '')))).join('')}
            </details>`);
        }
        return parts.join('');
    }

    function rememberDisclosure() {
        const details = mount.querySelector('[data-activity-history]');
        if (details) historyOpen = Boolean(details.open ?? details.hasAttribute('open'));
        expandedSchedules = new Set([...mount.querySelectorAll('[data-schedule-text]')]
            .filter((node) => node.open ?? node.hasAttribute('open')).map((node) => node.dataset.scheduleText));
    }

    function focusedControl() {
        const active = document.activeElement;
        return active && mount.contains?.(active) ? active.closest('[data-act]') : null;
    }

    function restoreFocus(previous) {
        if (!previous) return;
        const target = [...mount.querySelectorAll('[data-act]')].find((el) => (
            el.dataset.act === previous.act && (el.dataset.id || '') === previous.id
            && (el.dataset.holdId || '') === previous.holdId));
        if (!target) return;
        // Focus lives inside the disclosure for a retained row: open it rather
        // than focusing a control the owner cannot see.
        const details = target.closest?.('[data-activity-history]');
        if (details && !details.open) details.open = true;
        target.focus?.();
    }

    async function refresh() {
        if (destroyed) return;
        clearTimeout(migrationTimer); migrationTimer = null;
        const revision = ++refreshRevision;
        rememberDisclosure();
        const focused = focusedControl();
        const focusedKey = focused && { act: focused.dataset.act, id: focused.dataset.id || '', holdId: focused.dataset.holdId || '' };
        sections.forEach(({ root, status, loaded }) => {
            root.setAttribute('aria-busy', 'true');
            setInlineStatus(status, loaded ? 'Refreshing… Previously loaded values shown.' : 'Loading…');
        });
        const results = await Promise.allSettled([
            // This view renders only the queue; queue_only skips the whole
            // task-results scan server-side (v6.9x P2).
            getJson('/api/tasks?queue_only=1'),
            readStateJson(),
            getJson('/api/schedules'),
        ]);
        if (destroyed || revision !== refreshRevision) return;
        // A request can take long enough for the owner to move focus elsewhere.
        // Only restore the original control when it is still the active owner;
        // if this render detaches it, activeElement is still the original node
        // immediately before replacement.  A different active element means
        // the owner made a newer choice while the request was in flight.
        const mayRestoreFocus = !focusedKey || document.activeElement === focused;
        const census = results[1].status === 'fulfilled' ? results[1].value : null;
        const renderers = [(data) => renderQueue(data?.queue, census), renderBg, renderSchedules];
        sections.forEach((section, index) => {
            const { root, status, content } = section;
            root.removeAttribute('aria-busy');
            try {
                const result = results[index];
                if (result.status === 'rejected') throw result.reason;
                // Disclosure is an owner choice too.  Capture it at the render
                // boundary, after the async reads, so opening/closing history
                // while a poll is pending is not undone by a stale snapshot.
                if (index === 2) rememberDisclosure();
                content.innerHTML = renderers[index](result.value);
                content.querySelectorAll('[data-activity-phase]').forEach((el) => paintTaskPhase(
                    el, phaseViews[Number(el.dataset.activityPhase)], el.nextElementSibling));
                section.loaded = true;
                setInlineStatus(status, '');
            } catch (error) {
                if (isStoreMigrating(error)) {
                    setInlineStatus(status, 'Saved tasks are being upgraded…');
                    migrationTimer ??= setTimeout(refresh, 700);
                    return;
                }
                setInlineStatus(status, section.loaded
                    ? 'Could not refresh. Previously loaded values shown; current state is unknown. Reopen Activity to try again.'
                    : 'Could not load. Current state is unknown. Reopen Activity to try again.', 'error');
            }
        });
        if (mayRestoreFocus) restoreFocus(focusedKey);
    }

    // One seam for every lifecycle button: the owner NAMES the action and why.
    // The server applies and audits it; nothing here infers a command from text.
    // Delete goes through here too — reading the outcome is not optional for one
    // button and skipped for another, or a refused delete reads as a silent no-op.
    async function scheduleAction(id, action, reason, expectedHoldId = '') {
        const outcome = await fetchJson(`/api/schedules/${encodeURIComponent(id)}/action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, reason, expected_hold_id: expectedHoldId }),
        });
        reportScheduleOutcome(action, outcome);
        return outcome;
    }

    function reportScheduleOutcome(action, outcome) {
        const toast = scheduleOutcomeToast(action, outcome);
        if (toast) showToast(...toast);
    }

    mount.addEventListener('click', async (event) => {
        const btn = event.target.closest('[data-act]');
        if (!btn || busy || btn.dataset.act === 'schedule-text') return;
        const act = btn.dataset.act;
        const id = btn.dataset.id || '';
        if (act === 'bg-open') { openDashboardTab?.('consciousness'); return; }
        if (act === 'task-control') {
            // S3 (Q2/HQ1): owner product-wide parity — the SAME three-action
            // dropdown as the Chat card (one shared module: same actions,
            // endpoint bindings, request-id retry, and typed refusals).
            // Dismissing the menu continues the run. The durable detail decides
            // whether a cancel intent is pending (then only the hard escalation
            // is offered and hurry is never shown).
            let stored = null;
            try {
                stored = await getJson(`/api/tasks/${encodeURIComponent(id)}`);
            } catch (exc) {
                if (exc?.status !== 404) showToast(`Could not refresh task state: ${exc?.message || exc}`, 'error');
            }
            // A work's current episode offers that work's controls: Stop reaches every episode root still
            // running, Stop earlier attempt the earlier ones, each at the server's exact target.
            const work = btn.dataset.root === '1' ? workOfEpisode(lastCensus, id) : null;
            if (work && openWorkControls(btn, work, { cancelPending: taskCancelPending(stored), withResume: true,
                afterAction: refresh })) return;
            openTaskControlMenu(btn, {
                cancelPending: taskCancelPending(stored),
                budgetPaused: btn.dataset.budgetPaused === '1',
                resume: btn.dataset.resume === '1',
                // A child's Pause would not name its own tree (the server refuses it).
                wholeTree: btn.dataset.root === '1',
                busy: taskControlBusy(id),
                onAction: async (action) => {
                    busy = true;
                    try {
                        if (action === ACTION_HURRY) {
                            // Local toast acknowledgement only — never a chat message.
                            await hurryTaskAction(id);
                            return;
                        }
                        if (action === ACTION_RESUME) {
                            await resumeTaskAction(id);
                            return;
                        }
                        // Same declared semantics as the chat card (v6.82): the
                        // task AND its live subtree, so stopping an orchestrator
                        // never orphans its running subagents. A soft stop answers
                        // 202 with the intent open; an immediate stop answers 200
                        // once torn down or 202 pending while the intent is still
                        // being driven — either way the refresh shows honest state.
                        await requestStop(id, action);
                    } catch (exc) {
                        // A 404 is the documented completion race (the run
                        // finished on its own); the refresh tells that story.
                        if (exc?.status !== 404) {
                            showToast(`Action failed: ${exc?.message || exc}`, 'error');
                        }
                    } finally {
                        busy = false;
                        await refresh();
                    }
                },
            });
            return;
        }
        busy = true;
        btn.disabled = true;
        try {
            if (act === 'schedule-delete') {
                const noun = btn.dataset.note === '1' ? 'reminder' : 'schedule';
                const confirmedDelete = await openConfirmDialog({
                    title: `Delete ${noun}`,
                    body: `Delete this ${noun}? It stays restorable for a while and is removed after the retention window once its work has settled. This does not cancel work already started.`,
                    confirmLabel: 'Delete',
                    danger: true,
                });
                if (!confirmedDelete) return;
                await scheduleAction(id, 'delete', 'owner deleted the schedule from Activity');
            } else if (act === 'schedule-toggle') {
                // The button already carries the action it means; no full-record
                // round trip, so a stale read can never overwrite runtime fields.
                const action = btn.dataset.action === 'disable' ? 'disable' : 'restore';
                await scheduleAction(id, action, `owner chose ${action} from Activity`, btn.dataset.holdId || '');
            } else if (act === 'bg-toggle') {
                const on = btn.dataset.enabled === '1';
                // Reuse the existing direct control command (same as the chat header
                // toggle); /bg is a control slash-command, not a chat message to the agent.
                ws?.send?.({ type: 'command', cmd: `/bg ${on ? 'stop' : 'start'}` });
                await new Promise((resolve) => setTimeout(resolve, 400));
            }
        } catch (exc) {
            // A 404 is the documented completion race (the run finished on its own)
            // and the refresh below tells that story. Anything else is a real
            // failure — a refused cancel must not read as a silent no-op click.
            if (exc?.status !== 404) {
                showToast(`Action failed: ${exc?.message || exc}`, 'error');
            }
        } finally {
            busy = false;
            btn.disabled = false;
            await refresh();
        }
    });

    const shown = (event) => {
        if (event?.detail?.tab === 'activity') refresh();
    };
    window.addEventListener('ouro:dashboard-subtab-shown', shown);

    return { refresh, destroy() {
        destroyed = true;
        clearTimeout(migrationTimer);
        window.removeEventListener('ouro:dashboard-subtab-shown', shown);
    } };
}
