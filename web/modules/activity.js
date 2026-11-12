// Activity dashboard subtab (P4): a single observability + minimal-control view for
// cron/scheduled tasks, what is running/queued now, background consciousness and the services
// tasks kept running. Management is DIRECT mechanical control via existing APIs (cancel a task, enable/
// disable/delete/restore schedules, start/stop background consciousness, stop one kept service). Every
// lifecycle button NAMES its action to the one audited server seam, so the owner's
// buttons and the agent's tool mean exactly the same thing.

import { fetchJson, isStoreMigrating, readStateJson } from './api_client.js';
import { setInlineStatus } from './ui_helpers.js';
import { errorWords } from './ui_primitives.js';
import { fmt, tr } from './i18n.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { taskCancelPending } from './log_events.js';
import { waitingModel } from './project_activity.js';
import {
    memberState, openWorkControls, resumeTargets, workControls, workLine, workMenuActions, workOfEpisode, workRowView,
} from './work_status.js';
import { censusTaskFacts, censusTaskPhase, paintTaskPhase } from './task_phase_chip.js';
import {
    ACTION_HURRY,
    ACTION_RESUME,
    heldWorkFacts,
    hurryTaskAction,
    isRootTaskRow,
    openTaskControlMenu,
    requestStop,
    resumeTaskAction,
    taskControlActions,
    taskControlBusy,
    taskControlTriggerLabel,
} from './task_control_menu.js';
import { markView, statusMarkHtml } from './status_mark.js';
import { whenLabel } from './storage_format.js';
import { showToast } from './toast.js';
import { normalizeConsciousness } from './consciousness_api.js';
import { allowanceText } from './consciousness_words.js';

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

// Census phases that only guess at a parked row; the confirmed work whose live episode the row is says what is true.
const PARKED_PHASES = new Set(['budget_paused', 'budget_pausing']);

/** What remains of an answered root, from its confirmed `work` when one is given, else its census phase. */
function lateState(a, work) {
    if (work?.activity === 'unknown') return 'status unknown';
    if (work?.activity === 'working' && work.transition === 'pausing') return 'pausing';
    if (work?.activity === 'paused') return 'paused';
    return { budget_paused: 'paused', budget_pausing: 'pausing', unknown: 'status unknown' }[a.phase] || 'running';
}

/** An answered root whose late work remains (its upkeep, or a review its Pause deferred) names
 * its own work, when it answered, and that only the work after the answer waits — never a generic turn.
 * A confirmed `work` says that state, then its own line (who paused it, what still runs or nobody can confirm). */
function lateWorkHtml(a, late, work = null) {
    const at = new Date(String(late.answered_at || ''));
    const iso = Number.isNaN(at.getTime()) ? '' : at.toISOString();
    const when = iso ? ` <time datetime="${esc(iso)}" title="${esc(iso)}">${esc(whenLabel(iso))}</time>` : '';
    const line = work ? workLine(work) : '';
    // The state leads: a narrow row's ellipsis may cut the time, never what waits.
    return { name: esc(late.title || 'Answered task'),
        sub: `Remaining work ${lateState(a, work)} · answer delivered${when}${line ? ` · ${esc(line)}` : ''}` };
}

/** The name of the work a Presence turn's id belongs to ('' when no work names it): its correspondent's words never are. */
function presenceWorkName(census, taskId) {
    const works = Array.isArray(census?.works) ? census.works : [];
    return String(works.find((work) => String(work?.episode?.task_id || '') === taskId)?.name || '');
}

/** A census-only live root (a direct turn, a live inline Presence turn, or an answered root whose late work remains):
 * its state in the chip's own words (never the census code), and the shared control —
 * Resume only where its work's Resume control targets it (`resume`, work_status.resumeTargets).
 * `slot` marks where the caller paints a known outcome (`ownOutcomePhase`). A Presence turn says its `work`'s name and
 * line (a recorded Stop, what runs beside it) and offers no control before its first durable write (`cancelable`
 * false: no door takes it yet). `work` is the confirmed work whose live episode this row is: a parked census phase
 * only guesses, so that work's word and line say it, the row's own outcome still leading, as on a queue row. A work's
 * episode is offered that work's controls only (the click handler, never the generic menu), and a parked row whose
 * work lists none (an unbound start no door reaches) shows no trigger at all. */
export function liveActivityRowHtml(a, nowMs = Date.now(), slot = null, resume = false, work = null) {
    const names = { direct_chat: 'Direct turn', managed_task: 'Managed task' };
    const started = Number(a.started_at) || 0;
    const spent = started > 0 ? elapsedWords(nowMs / 1000 - started) : '';
    const elapsed = spent ? ` · ${spent}` : '';
    const truth = work && PARKED_PHASES.has(a.phase) ? workRowView(work) : null;
    const late = a.late_work && typeof a.late_work === 'object' ? lateWorkHtml(a, a.late_work, truth && work) : null;
    const own = !late && slot ? ownOutcomePhase(a) : null;
    const outcome = own && truth ? { ...own, secondary: truth.word } : own;
    const presence = a.kind === 'presence';
    // A Presence turn says its work's name; any other run the name its card shows (the producer's suggested_name).
    const named = !late && (presence ? work?.name : String(a.suggested_name || '').trim());
    const line = presence && work ? workLine(work) : truth?.secondary || '';
    const said = (text) => `<span data-i18n-skip>${esc(text)}</span>`;
    // The trigger says what a press will offer: its work's controls, else the card-state set the click opens.
    const actions = work ? workMenuActions(workControls(work), { resume: true })
        : taskControlActions({ budgetPaused: a.phase === 'budget_paused', resume, wholeTree: true });
    const offers = !truth || actions.length > 0;
    const look = truth || ownCensusView(a).view;
    const mark = late || outcome ? '' : stateMark(look);
    const trigger = (presence && a.cancelable === false) || !offers ? ''
        : `<div class="ui-row-actions activity-row-actions"><button type="button" class="btn btn-sm btn-default" data-act="task-control" data-id="${esc(a.activity_id || '')}" data-root="1"${a.phase === 'budget_paused' ? ' data-budget-paused="1"' : ''}${resume ? ' data-resume="1"' : ''} data-task-control-direct="1">${esc(taskControlTriggerLabel(actions))}</button></div>`;
    return `<div class="ui-row activity-row">
                <span class="ui-row-title activity-name"${named ? ' data-i18n-authored' : ''}>${late ? late.name : esc(named || names[a.kind] || 'Live turn')}</span>
                <span class="ui-row-meta activity-sub">${late ? late.sub : `${mark}${outcome ? `${slot(outcome)}${line ? ` · ${said(line)}` : ''}`
                    : said([truth ? truth.word : censusWords(a), line].filter(Boolean).join(' · '))}${elapsed}`}</span>
                ${trigger}
            </div>`;
}

/** The shared status mark of a row's state, beside its word: the work's own shape, the owner's move in amber. */
function stateMark(view) {
    return statusMarkHtml('activity-mark', markView({ mark: view?.mark, owner: Boolean(view?.owner) }));
}

// The same words, read loosely: whitespace collapsed and a closing full stop dropped (a derived name drops it too).
const plainWords = (value) => String(value || '').replace(/\s+/g, ' ').trim().replace(/\.$/, '');

/** A row's one disclosure: the whole order (or a schedule's instruction) when its title is not already those words,
 * and the record's codes. `key` keeps it open across re-renders; `data-schedule-text` marks a schedule's. */
function rowDetailsHtml({ key, open = false, text = '', title = '', facts = '', scheduleId = null }) {
    const whole = String(text || '').trim();
    const more = Boolean(whole) && plainWords(whole) !== plainWords(title);
    if (!more && !facts) return '';
    const schedule = scheduleId === null ? '' : ` data-schedule-text="${esc(scheduleId)}"`;
    const act = scheduleId === null ? '' : ` data-act="schedule-text" data-id="${esc(scheduleId)}"`;
    return `<details class="ui-disclosure activity-details" data-open-key="${esc(key)}"${schedule}${open ? ' open' : ''}>
        <summary${act}>Details</summary>
        ${more ? `<p class="activity-order" data-i18n-authored>${esc(whole).replace(/\r?\n/g, '<br>')}</p>` : ''}
        ${facts ? `<p class="activity-facts">${facts}</p>` : ''}
    </details>`;
}

/** A stored UTC schedule instant for the owner in ONE zone: the schedule's own (`timeZone`, the zone its cadence is
 * said in), else this viewer's. One fixed English format inside the English line (never the browser's locale, which
 * mixed two languages in one line); the exact UTC instant stays in `datetime`/`title`. Records stay UTC; an
 * unparseable value is shown raw rather than guessed. */
export function scheduleInstantHtml(value, { timeZone, includeYear = false } = {}) {
    const raw = String(value || '');
    const parsed = new Date(raw);
    if (!raw || Number.isNaN(parsed.getTime())) return esc(raw);
    // Storage's sentence shape ("16 Oct at 18:10", storage_format.whenLabel), in the given zone.
    const date = parsed.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', ...(includeYear ? { year: 'numeric' } : {}), timeZone });
    const time = parsed.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone });
    const exact = parsed.toISOString().replace('.000Z', 'Z');
    return `<time datetime="${esc(exact)}" title="${esc(exact)}">${esc(fmt('{date} at {time}', { date, time }))}</time>`;
}

/** This viewer's zone: a schedule in another one names its zone beside its times. */
export const viewerTimeZone = () => Intl.DateTimeFormat().resolvedOptions().timeZone || '';

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
// withdrawn rows, fired one-shots and switched-off one-shots whose moment has
// passed (the server's `time_passed_on`) remain reachable as finished.
export function isRetainedSchedule(s) {
    const status = scheduleStatus(s);
    return status === 'consumed' || (status === 'disabled' && Boolean(s.time_passed_on))
        || (status === 'delete_pending' && Boolean(s.withdrawal) && s.withdrawal.work_in_flight !== true);
}

/** The retained group's label, by what it holds: finished one-shots, withdrawn rows, or both. */
function retainedLabel(rows) {
    const withdrawn = rows.filter((s) => scheduleStatus(s) === 'delete_pending').length;
    const finished = rows.length - withdrawn;
    return finished && withdrawn ? `Finished and withdrawn (${rows.length})`
        : withdrawn ? `Withdrawn (${withdrawn})` : `Finished (${finished})`;
}

function scheduleRunHtml(s, timeZone) {
    const outcome = s.outcome;
    const parts = [];
    if (outcome?.state === 'pending_observation') {
        parts.push('last run: outcome not observed yet');
    } else if (outcome?.state) {
        const quiet = outcome.quiet === true && outcome.notice !== true;
        const state = quiet && outcome.state === 'completed' ? tr('schedule.outcome.quiet', 'ran · no message') : outcome.state;
        // The line is the host's notice, else the run's own words (its sentence, or the end of its answer).
        const line = outcome.line ? ` · ${outcome.notice ? `notice: ${esc(outcome.line)}` : `<span data-i18n-authored>${esc(outcome.line)}</span>`}` : '';
        parts.push(`last run: ${esc(state)}${outcome.ended_at ? ` · ${scheduleInstantHtml(outcome.ended_at, { timeZone, includeYear: true })}` : ''}${line}`);
    }
    // Why the row did not do its job, as stored (a note whose chat write was not confirmed is consumed and never
    // retried, so this is the only place it says so) — once: a run whose own line already says it is not repeated.
    const error = String(s.last_error || '').trim();
    if (error && error !== String(outcome?.line || '').trim()) parts.push(`last error: ${esc(error)}`);
    if (s.previous_run_open) {
        parts.push(`previous run still open${s.previous_run_open.task_id ? `: ${esc(s.previous_run_open.task_id)}` : ''}`);
    }
    return parts.map((line) => `<span class="activity-sub">${line}</span>`).join('');
}

export function scheduleRowHtml(s, expanded = false, viewerZone = viewerTimeZone()) {
    const trigger = s.trigger || {};
    const once = String(trigger.type || 'cron') === 'once';
    const zone = String(s.timezone || '').trim();
    const timeZone = zone || undefined;
    // One zone per row: a cron runs in its record's zone (the server's when none is stored), its cadence and times
    // are said there, and that zone is named on the row only when it is not this viewer's. Its code goes to Details.
    const cadence = once ? `Once on ${scheduleInstantHtml(trigger.run_at, { timeZone })}` : esc(s.cadence || trigger.expr || s.cron || '');
    const timing = `${cadence}${zone && zone !== viewerZone ? ` (${esc(zone)})` : ''}`;
    const note = ['note', 'notify'].includes(s.kind);
    const next = s.next_run_at ? scheduleInstantHtml(s.next_run_at, { timeZone }) : '';
    const runs = ['active', 'waiting'].includes(scheduleStatus(s));  // a switched-off or withdrawn row has no next run
    const status = scheduleStatus(s);
    const enabled = status === 'active';
    const consumed = status === 'consumed';
    const pendingDelete = status === 'delete_pending';
    const id = esc(s.id || '');
    // A due occurrence that waits (capacity, a missing folder, an unknown fact) says why.
    const wait = s.waiting_detail ?? s.hold ?? s.waiting;
    const waitText = typeof wait === 'string' ? wait : [wait?.reason, wait?.detail].filter(Boolean).join(': ');
    const waiting = waitText ? ` · <span class="activity-tag">waiting: ${esc(waitText)}</span>` : '';
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
    // The main line: what kind, when, its next run, its state and what holds it; the codes and ids are its Details.
    const sub = `${note ? 'Reminder · ' : s.kind === 'task' ? 'Task · ' : ''}${timing}${next && runs ? ` · next ${next}` : ''} · ${esc(statusLabel)}${hold.reason ? ` · ${esc(explanations[hold.reason] || hold.reason)}${hold.detail ? `: ${esc(hold.detail)}` : ''}` : ''}${followupWait ? ` · ${esc(explanations[followupWait] || followupWait)}` : ''}${waiting}`;
    const withdrawalDetail = withdrawal ? [
        withdrawal.at ? `withdrawn ${scheduleInstantHtml(withdrawal.at, { timeZone, includeYear: true })}` : '',
        withdrawal.by ? `by ${esc(withdrawal.by)}` : '',
        withdrawal.reason ? esc(withdrawal.reason) : '',
    ].filter(Boolean).join(' · ') : '';
    const facts = [
        once ? '' : `cron ${esc(trigger.expr || s.cron || '')}`, zone ? `time zone ${esc(zone)}` : once ? '' : 'server time zone',
        `relation ${esc(relation)}${binding}${address}`,
        s.deadline_at ? `deadline ${scheduleInstantHtml(s.deadline_at, { timeZone, includeYear: true })}` : '',
        s.completed_at ? 'already fired' : '', s.skill ? `from skill ${esc(s.skill)}` : '', withdrawalDetail,
    ].filter(Boolean).join(' · ');
    // The exact hold release stays available wherever a hold exists, including on a
    // deleted row whose accepted task still needs it; it never re-enables the row.
    // Each lifecycle button carries the past tense of its own word for the outcome it reports.
    const holdControl = !hold.hold_id ? ''
        : relation === 'unknown'
            ? '<span class="activity-tag">resolve relationship in conversation</span>'
            : !s.hold_persisted
                ? '<span class="activity-tag">hold persistence pending</span>'
                : `<button type="button" class="btn btn-sm btn-default" data-done="restored" data-act="schedule-toggle" data-id="${id}" data-action="restore" data-hold-id="${esc(hold.hold_id || '')}">Restore hold</button>`;
    // Restore cancels a withdrawal the server calls restorable (not fired, not held);
    // exact hold release is a separate action.
    const canRestore = !hold.hold_id && (withdrawal ? withdrawal.restorable === true : !(once && s.completed_at));
    const restoreWithdrawal = canRestore
        ? `<button type="button" class="btn btn-sm btn-default" data-done="restored" data-act="schedule-toggle" data-id="${id}" data-action="restore">Restore</button>` : '';
    const lifecycle = pendingDelete
        ? `${holdControl}${restoreWithdrawal}`
        : status === 'waiting'
            ? '<span class="activity-tag">waiting for existing control</span>'
            : hold.hold_id
                ? holdControl
                : consumed
                    ? '<span class="activity-tag">consumed once · history</span>'
                    : `<button type="button" class="btn btn-sm btn-default" data-done="${enabled ? 'disabled' : 'enabled'}" data-act="schedule-toggle" data-id="${id}" data-action="${enabled ? 'disable' : 'restore'}">${enabled ? 'Disable' : 'Enable'}</button>`;
    const text = (note ? s.notification?.text : s.task?.text || s.task?.objective) || s.description || s.objective_preview || '';
    const title = s.display_name || s.name || s.id || 'schedule';
    return `<div class="ui-row activity-row activity-schedule">
        <span class="ui-row-title activity-name" data-i18n-authored>${esc(title)}</span>
        <div class="ui-row-meta activity-lines"><span class="activity-sub">${sub}</span>${scheduleRunHtml(s, timeZone)}</div>
        <div class="ui-row-actions activity-row-actions">${lifecycle}
           <button type="button" class="btn btn-sm btn-danger" data-act="schedule-delete" data-id="${id}" data-note="${note ? '1' : ''}">Delete</button></div>
        ${rowDetailsHtml({ key: `schedule:${s.id || ''}`, open: expanded, text, title, facts, scheduleId: String(s.id || '') })}
    </div>`;
}

// What one lifecycle response means for the owner: ``[text, level]`` or null.
// The response says two separable things: whether the schedule CHANGED, and
// whether both audit facts landed. A change whose outcome record was lost is
// neither a clean success nor a failure, and saying either would be wrong.
// `word` is the pressed button's own word and `done` its past tense (Enable sends
// `restore` and reads "enabled"), so the feedback names what the owner pressed.
export function scheduleOutcomeToast(action, outcome, { word = action, done: doneWord = '' } = {}) {
    if (!outcome || typeof outcome !== 'object' || typeof outcome.changed !== 'boolean') {
        return [`Schedule ${word}: the server did not say what happened.`, 'error'];
    }
    const detail = String(outcome.detail || outcome.status || 'no status reported');
    if (outcome.changed === false) return [`Schedule ${word} did not change anything: ${detail}`, 'error'];
    // Restriction refresh can change bookkeeping while REFUSING release.
    // Only typed applied outcomes may produce an applied/success message.
    if (!['updated', 'withdrawn', 'deleted', 'delete_deferred',
        'hold_released', 'changed_audit_incomplete'].includes(outcome.status)) {
        return [`Schedule ${word} refused: ${detail}`, 'error'];
    }
    if (outcome.audit !== 'recorded') return [`Schedule ${word} applied, but its audit record is incomplete: ${detail}`, 'warn'];
    if (outcome.status === 'delete_deferred') return [`Schedule deletion is pending: ${detail}`, 'info'];
    if (outcome.ok !== true) return [`Schedule ${word} changed, but did not finish successfully: ${detail}`, 'warn'];
    const done = outcome.status === 'withdrawn' ? 'withdrawn' : doneWord || `${action}d`;
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

/** One service a task kept running, as custody reads it now (`GET /api/services/kept`): its name leads, then its state,
 * the task it came from, since when and the address its log printed; Stop ends that one launch. */
export function keptServiceRowHtml(service) {
    const name = String(service?.name || 'service');
    const id = String(service?.launch_id || '');
    const captureOnly = service?.state === 'capture_owed';
    const action = captureOnly ? 'Save outputs' : 'Stop';
    const origin = String(service?.task_title || '').trim() || String(service?.task_id || '').trim();
    const since = whenLabel(service?.since);
    const facts = [
        `Kept by a task · ${service?.state === 'running' ? 'running' : service?.state === 'capture_owed'
            ? 'ended, declared outputs not captured yet' : 'status unknown'}`,
        origin ? `from ${esc(origin)}` : '',
        since ? `since ${esc(since)}` : '',
        service?.address ? esc(service.address) : '',
    ].filter(Boolean).join(' · ');
    return `<div class="ui-row activity-row" data-kept-service="${esc(id)}">
            <span class="ui-row-title activity-name" data-i18n-authored>${esc(name)}</span>
            <span class="ui-row-meta activity-sub">${facts}</span>
            <div class="ui-row-actions activity-row-actions">
                <button type="button" class="btn btn-default btn-sm" data-act="kept-stop" data-id="${esc(id)}" data-name="${esc(name)}" data-capture-only="${captureOnly ? '1' : '0'}" aria-label="${action} ${esc(name)}">${action}</button>
            </div>
        </div>`;
}

/** The kept services under Background: every row, then what the read could not see. An unreadable list says so. */
export function keptServicesHtml(data) {
    if (!data || !Array.isArray(data.services)) {
        return '<div class="activity-empty">Kept services could not be read. Whether any still run is unknown.</div>';
    }
    const rows = data.services.map(keptServiceRowHtml);
    if (data.complete === false) {
        rows.push(`<div class="activity-empty">Kept services may be missing from this list: ${esc(data.note || 'custody could not be read whole')}.</div>`);
    }
    return rows.join('');
}

/** The Stop confirmation: what a stop does, never a promise its outputs will be kept (the receipt says). */
export function keptStopConfirmBody(name, captureOnly = false) {
    if (captureOnly) return `Save outputs from ${name}? This service was reported ended. This rechecks it, stops any remaining processes and tries to save its declared outputs for the original task. The result will show what was saved.`;
    return `Stop ${name}? A task started it and asked to keep it running. Stopping it ends its processes and keeps its log, then tries to capture the outputs it declared for that task; the result will say whether they were kept. Nothing else will be stopped.`;
}

/** A Stop receipt in owner words, with its toast tone: a stop whose declared outputs were not all kept says so. */
export function keptStopToast(receipt, name) {
    const subject = receipt?.name || name || 'The service';
    const outputs = receipt?.outputs || {};
    switch (receipt?.outcome) {
    case 'stopped':
        if (outputs.state === 'captured') return [`Stopped ${subject}. Its log and declared outputs are kept.`, 'success'];
        if (outputs.state === 'none_declared') return [`Stopped ${subject}. Its log is kept.`, 'success'];
        return [outputs.state === 'unavailable'
            ? `Stopped ${subject}. Its log is kept; no declared outputs were captured: ${outputs.detail || 'none recorded'}.`
            : `Stopped ${subject}. Its log is kept, but not every declared output was captured: ${
                outputs.detail || 'the capture did not finish'}.`, 'warning'];
    case 'not_found':
        if (!receipt.outputs) return [`${subject} was no longer running; nothing was stopped.`, 'info'];
        return receipt.ok ? [`${subject} was no longer running; its declared outputs are now kept.`, 'success']
            : [`${subject} was no longer running; not every declared output was captured: ${
                outputs.detail || 'the capture did not finish'}.`, 'warning'];
    case 'not_confirmed': return [`${subject} is not confirmed stopped: ${receipt.detail || 'some processes may still run'}.`, 'warning'];
    default: return [`Whether ${subject} stopped is unknown: ${receipt?.detail || 'custody could not be read'}.`, 'error'];
    }
}

export function initActivity({ mount, ws, openDashboardTab = null } = {}) {
    if (!mount) return { refresh: () => {} };
    let busy = false;
    let refreshRevision = 0;
    let migrationTimer = null, destroyed = false;
    // Remembered across re-renders: whether each disclosure (a group, a row's Details) is open, by its key. A refresh
    // rebuilds the markup, so a poll that silently re-collapsed what the owner just opened (and dropped the focus they
    // had inside it), or reopened what they folded, would read as the app undoing their action.
    let openState = new Map();
    const isOpen = (key, byDefault = false) => openState.get(key) ?? byDefault;
    const openIf = (key, byDefault = false) => (isOpen(key, byDefault) ? ' open' : '');
    // One Dashboard tab body; each section is a card whose rows are trays (docs/DESIGN.md principle 2).
    mount.innerHTML = `<div class="dashboard-body scroll-fade-y activity-body">
        <section class="ui-card activity-section" data-activity-section="queue"><h3 class="dashboard-card-title">Running &amp; queued</h3></section>
        <section class="ui-card activity-section" data-activity-section="background"><h3 class="dashboard-card-title">Background</h3></section>
        <section class="ui-card activity-section" data-activity-section="schedules"><h3 class="dashboard-card-title">Scheduled</h3></section>
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
    // A group of one list by its records' state (paused work, finished schedules): its label and count from the caller.
    const groupHtml = (key, label, rows, { attrs = '', open = false } = {}) => `<details class="ui-disclosure activity-group" data-open-key="${key}"${attrs}${openIf(key, open)}>
            <summary>${esc(label)}</summary>
            <div class="activity-group-rows">${rows.join('')}</div>
        </details>`;
    // The list reads by state: what works now, what waits, what is queued, then paused work under its count (open
    // until the owner folds it: its rows keep their Resume and Stop one press away).
    const RANKS = { working: 0, waiting: 1, unknown: 1, queued: 2, paused: 3 };

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
            const ownId = String(q.id || t.id || '');
            const id = esc(ownId);
            // A Presence turn's words are its correspondent's, never its name: its saved work says its work's name.
            // Any other run is named as its card is (the producer's suggested_name), never by its whole order.
            const named = t._presence_turn ? presenceWorkName(census, ownId) : String(t.suggested_name || t.title || '').trim();
            const label = esc(named || (t._presence_turn ? 'Live turn' : q.type || ownId || 'task'));
            const spent = kind === 'running' ? elapsedWords(q.runtime_sec) : '';
            const rt = spent ? ` · ${spent}` : '';
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
            const meta = `${outcome ? phaseSlot(outcome) : `${stateMark(view)}${esc(kindLabel)}`}${secondary ? ` · ${esc(secondary)}` : ''}${heldFacts ? ` · ${esc(heldFacts)}` : ''}${q.type ? ` · ${esc(q.type)}` : ''}${rt}`;
            const hold = t.project_admission_hold;
            const projectWait = hold?.label ? ` · ${esc(hold.label)}: ${esc(hold.detail || hold.reason)}` : '';
            // The trigger says what the press offers: a root's work controls, else the card-state set (click handler).
            const root = phases.has(ownId) || isRootTaskRow(t, ownId);
            const work = root ? workOfEpisode(census, ownId) : null;
            const budgetPaused = parked || resumable.has(ownId);
            const actions = work ? workMenuActions(workControls(work), { resume: true })
                : taskControlActions({ budgetPaused, resume: resumable.has(ownId), wholeTree: root });
            // The whole order is the record: it waits behind the row's Details (owner seed S2).
            const order = t._presence_turn ? '' : String(t.description || t.objective || t.text || '');
            // The held cause is what the owner reads before Resume: it wraps, never ellipsized.
            const html = `<div class="ui-row activity-row${projected?.activity === 'waiting' ? ' activity-held' : ''}">
                <span class="ui-row-title activity-name"${named ? ' data-i18n-authored' : ''}>${label}</span>
                <span class="ui-row-meta activity-sub">${meta}${projectWait}</span>
                <div class="ui-row-actions activity-row-actions">
                    <button type="button" class="btn btn-sm btn-default" data-act="task-control" data-id="${id}"${budgetPaused ? ' data-budget-paused="1"' : ''}${resumable.has(ownId) ? ' data-resume="1"' : ''}${root ? ' data-root="1"' : ''} data-task-control-direct="1">${esc(taskControlTriggerLabel(actions))}</button>
                </div>
                ${rowDetailsHtml({ key: `order:${ownId}`, open: isOpen(`order:${ownId}`), text: order, title: named })}
            </div>`;
            return { html, rank: RANKS[projected?.activity] ?? (kind === 'pending' ? RANKS.queued : RANKS.working) };
        };
        // Queue facts keep their runtime/budget controls. The census adds only
        // missing identities; queued direct turns can appear in both sources. A live Presence turn its
        // conversation lane carries (the Host's admission facts, `_presence_run`) is its place's conversation:
        // only the census lists it, by its work, and a published queue row it already left never stands for it.
        const queued = running.filter((q) => !q?.task?._presence_run);
        const known = new Set([...queued, ...pending].map((q) => String(q.id || q.task?.id || '')));
        const live = activities.filter((a) => a && !known.has(String(a.activity_id || '')));
        const items = [...queued.map((q) => row(q, 'running')), ...pending.map((q) => row(q, 'pending')),
            ...live.map((a) => {
                const work = workOfEpisode(census, a.activity_id);
                return { html: liveActivityRowHtml(a, Date.now(), phaseSlot, resumable.has(String(a.activity_id || '')), work),
                    rank: a.phase === 'budget_paused' || work?.activity === 'paused' ? RANKS.paused : RANKS.working };
            })];
        if (!items.length) {
            return census?.active_chat_activities_complete === true
                ? '<div class="activity-empty">Nothing running or queued.</div>'
                : '<div class="activity-empty">Queue empty; live turns unknown.</div>';
        }
        const ordered = items.map((item, index) => ({ ...item, index })).sort((a, b) => a.rank - b.rank || a.index - b.index);
        const paused = ordered.filter((item) => item.rank === RANKS.paused).map((item) => item.html);
        return [...ordered.filter((item) => item.rank !== RANKS.paused).map((item) => item.html),
            paused.length ? groupHtml('paused', `Paused (${paused.length})`, paused, { open: true }) : ''].join('');
    }

    function renderBg(stateData, kept) {
        if (typeof stateData?.bg_consciousness_enabled !== 'boolean') throw new Error('Background state unavailable');
        const enabled = stateData.bg_consciousness_enabled;
        const bg = (stateData && stateData.bg_consciousness_state) || {};
        const detail = esc(bg.detail || (enabled ? 'enabled' : 'disabled'));
        const facts = [
            bg.level ? `autonomy ${esc(bg.level)}` : '',
            bg.last_wake_at ? `last wake ${esc(whenLabel(bg.last_wake_at))}${bg.last_wake_outcome ? ` (${esc(bg.last_wake_outcome)})` : ''}` : '',
            // The money line in the Consciousness tab's own words ("≥ $1.50 of $20.00 in the last 24 hours").
            esc(allowanceText(normalizeConsciousness(bg, { enabled }))),
            Number.isFinite(Number(bg.max_tasks)) ? esc(fmt('own tasks {running}/{max}', { running: Number(bg.tasks_running || 0), max: Number(bg.max_tasks) })) : '',
        ].filter(Boolean).join(' · ');
        return `<div class="ui-row activity-row">
            <span class="ui-row-title activity-name">Consciousness</span>
            <span class="ui-row-meta activity-sub">${enabled ? 'enabled' : 'disabled'}${detail ? ` · ${detail}` : ''}${facts ? ` · ${facts}` : ''}</span>
            <div class="ui-row-actions activity-row-actions">
                <button type="button" class="btn btn-sm btn-default" data-act="bg-toggle" data-enabled="${enabled ? '1' : '0'}"${ws ? '' : ' disabled'}>${enabled ? 'Stop' : 'Start'}</button>
                <button type="button" class="btn btn-sm btn-ghost" data-act="bg-open" aria-label="Open Dashboard › Consciousness"${openDashboardTab ? '' : ' hidden'}>Details</button>
            </div>
        </div>${keptServicesHtml(kept)}`;
    }

    // The standing promises first: what will run, then what waits or is held, then what is withdrawn or switched off.
    const SCHEDULE_RANKS = { active: 0, waiting: 1, held: 1, delete_pending: 2 };

    function renderSchedules(data) {
        if (!Array.isArray(data?.tasks)) throw new Error('Schedules unavailable');
        const tasks = data.tasks;
        if (!tasks.length) return '<div class="activity-empty">No scheduled tasks.</div>';
        // Finished and withdrawn rows stay reachable without padding the standing schedules.
        const standing = [];
        const retained = [];
        for (const s of tasks) (isRetainedSchedule(s) ? retained : standing).push(s);
        const rank = (s) => SCHEDULE_RANKS[scheduleStatus(s)] ?? 3;
        const rowOf = (s) => scheduleRowHtml(s, isOpen(`schedule:${s.id || ''}`));
        const parts = standing.length
            ? standing.map((s, index) => ({ s, index })).sort((a, b) => rank(a.s) - rank(b.s) || a.index - b.index).map(({ s }) => rowOf(s))
            : ['<div class="activity-empty">No active or disabled schedules.</div>'];
        if (retained.length) parts.push(groupHtml('history', retainedLabel(retained), retained.map(rowOf), { attrs: ' data-activity-history' }));
        return parts.join('');
    }

    function rememberDisclosure() {
        openState = new Map([...mount.querySelectorAll('details[data-open-key]')]
            .map((node) => [node.dataset.openKey, Boolean(node.open ?? node.hasAttribute('open'))]));
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
        // Focus lives inside a collapsed group (a retained row, paused work): open it rather
        // than focusing a control the owner cannot see.
        for (let details = target.closest?.('details'); details; details = details.parentElement?.closest?.('details')) {
            if (!details.open) details.open = true;
        }
        target.focus?.();
    }

    async function refresh() {
        if (destroyed) return;
        clearTimeout(migrationTimer); migrationTimer = null;
        const revision = ++refreshRevision;
        rememberDisclosure();
        const focused = focusedControl();
        const focusedKey = focused && { act: focused.dataset.act, id: focused.dataset.id || '', holdId: focused.dataset.holdId || '' };
        // A refresh says nothing while it reads (aria-busy only): a line that came and went under each heading moved
        // every row below it (DESIGN principle 6). Only the first read and a failure speak.
        sections.forEach(({ root, status, loaded }) => {
            root.setAttribute('aria-busy', 'true');
            if (!loaded) setInlineStatus(status, 'Loading…');
        });
        const results = await Promise.allSettled([
            // This view renders only the queue; queue_only skips the whole
            // task-results scan server-side (v6.9x P2).
            getJson('/api/tasks?queue_only=1'),
            readStateJson(),
            getJson('/api/schedules'),
            getJson('/api/services/kept'),
        ]);
        if (destroyed || revision !== refreshRevision) return;
        // A request can take long enough for the owner to move focus elsewhere.
        // Only restore the original control when it is still the active owner;
        // if this render detaches it, activeElement is still the original node
        // immediately before replacement.  A different active element means
        // the owner made a newer choice while the request was in flight.
        const mayRestoreFocus = !focusedKey || document.activeElement === focused;
        const census = results[1].status === 'fulfilled' ? results[1].value : null;
        const kept = results[3].status === 'fulfilled' ? results[3].value : null;
        const renderers = [(data) => renderQueue(data?.queue, census), (data) => renderBg(data, kept), renderSchedules];
        sections.forEach((section, index) => {
            const { root, status, content } = section;
            root.removeAttribute('aria-busy');
            try {
                const result = results[index];
                if (result.status === 'rejected') throw result.reason;
                // Disclosure is an owner choice too.  Capture it at the render
                // boundary, after the async reads, so opening/closing a group or
                // a row's Details while a poll is pending is not undone by a stale snapshot.
                rememberDisclosure();
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
    async function scheduleAction(id, action, reason, expectedHoldId = '', words = {}) {
        const outcome = await fetchJson(`/api/schedules/${encodeURIComponent(id)}/action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, reason, expected_hold_id: expectedHoldId }),
        });
        const toast = scheduleOutcomeToast(action, outcome, words);
        if (toast) showToast(...toast);
        return outcome;
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
                if (exc?.status !== 404) showToast(`Could not refresh task state. ${errorWords(exc).text}`, 'error');
            }
            // A work's current episode offers that work's controls: Stop reaches every episode root still
            // running, Stop earlier attempt the earlier ones, each at the server's exact target. It offers
            // only those: a work that lists none opens nothing, never the generic menu below, whose Stop or
            // Resume a door that work does not list would refuse.
            const work = btn.dataset.root === '1' ? workOfEpisode(lastCensus, id) : null;
            if (work) {
                openWorkControls(btn, work, { cancelPending: taskCancelPending(stored), withResume: true,
                    afterAction: refresh });
                return;
            }
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
                // The outcome speaks the pressed button's word (Enable sends `restore` and says "enabled").
                await scheduleAction(id, action, `owner chose ${action} from Activity`, btn.dataset.holdId || '',
                    { word: btn.textContent.trim().toLowerCase() || action, done: btn.dataset.done || '' });
            } else if (act === 'kept-stop') {
                const name = btn.dataset.name || 'this service';
                const captureOnly = btn.dataset.captureOnly === '1';
                const action = captureOnly ? 'Save outputs' : 'Stop';
                const confirmed = await openConfirmDialog({
                    title: `${action} ${name}`,
                    body: keptStopConfirmBody(name, captureOnly),
                    confirmLabel: action,
                    danger: true,
                });
                if (!confirmed) return;
                const receipt = await fetchJson(`/api/services/kept/${encodeURIComponent(id)}/stop`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ reason: captureOnly ? 'the owner retried saving its outputs from Activity › Background'
                        : 'the owner stopped it from Activity › Background' }),
                });
                showToast(...keptStopToast(receipt, name));
            } else if (act === 'bg-toggle') {
                const on = btn.dataset.enabled === '1';
                // Reuse the existing direct control command (same as the chat header
                // toggle); /bg is a control slash-command, not a chat message to the agent.
                ws?.send?.({ type: 'command', cmd: `/bg ${on ? 'stop' : 'start'}`, ui_control: { name: 'consciousness_toggle' } });
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
