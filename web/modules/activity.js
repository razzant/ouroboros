// Activity dashboard subtab (P4): a single observability + minimal-control view for
// cron/scheduled tasks, what is running/queued now, and background consciousness.
// Management is DIRECT mechanical control via existing APIs (cancel a task, enable/
// disable/delete/restore schedules, start/stop background consciousness). Every
// lifecycle button NAMES its action to the one audited server seam, so the owner's
// buttons and the agent's tool mean exactly the same thing. Skill rows retain an
// owner override in the existing schedule store, so a lifecycle resync cannot undo it.

import { fetchJson } from './api_client.js';
import { setInlineStatus } from './ui_helpers.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { taskCancelPending } from './log_events.js';
import {
    ACTION_HURRY,
    ACTION_RESUME,
    TASK_CONTROL_TRIGGER_LABEL,
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

/** A census-only live root (a direct turn, or an answered root whose late work the owner
 * paused, D10): its phase, and the shared control — Resume once the census says Paused. */
export function liveActivityRowHtml(a, nowMs = Date.now()) {
    const names = { direct_chat: 'Direct turn', managed_task: 'Managed task' };
    const started = Number(a.started_at) || 0;
    const elapsed = started > 0 ? ` · ${Math.max(0, Math.round(nowMs / 1000 - started))}s` : '';
    return `<div class="activity-row">
                <div class="activity-row-main">
                    <span class="activity-name">${esc(names[a.kind] || 'Live turn')}</span>
                    <span class="activity-sub">${esc(a.phase || '')}${elapsed}</span>
                </div>
                <div class="activity-row-actions">
                    <button type="button" class="btn btn-xs btn-danger" data-act="task-control" data-id="${esc(a.activity_id || '')}" data-root="1"${a.phase === 'budget_paused' ? ' data-budget-paused="1"' : ''}>${esc(TASK_CONTROL_TRIGGER_LABEL)}</button>
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

function isSkillManaged(s) {
    return Boolean(s && (String(s.source || '') === 'skill_manifest' || String(s.skill || '')));
}

// The server names the lifecycle word; these fall back for an older payload
// so a row is never silently promoted to "active" by a missing field.
export function scheduleStatus(s) {
    const once = String((s.trigger || {}).type || 'cron') === 'once';
    if (s.status) return String(s.status);
    if (s.delete_requested_at) return 'delete_pending';
    if (once && s.completed_at) return 'consumed';
    if (isSkillManaged(s) && ['disabled', 'deleted'].includes(String(s.manual_override || ''))) return 'suppressed';
    return s.enabled === false ? 'disabled' : 'active';
}

// Retained rows — a fired one-shot, a suppressed skill row — are history, not
// schedules wearing a disabled flag. A deleted row still finishing accepted
// work is not history: it stays in the standing list until it disappears.
export function isRetainedSchedule(s) {
    return ['consumed', 'suppressed'].includes(scheduleStatus(s));
}

export function scheduleRowHtml(s) {
    const managed = isSkillManaged(s);
    const trigger = s.trigger || {};
    const once = String(trigger.type || 'cron') === 'once';
    // One-shot rows have no cron: show the fire instant + a "one-shot" tag. A cron
    // expression runs in its record's zone, or the server's when none is stored.
    const timing = once
        ? `one-shot · at/after ${scheduleInstantHtml(trigger.run_at)}`
        : `${esc(trigger.expr || s.cron || '')} (${s.timezone ? esc(s.timezone) : 'server time zone'})`;
    // A note (kind "notify") is shown to the owner as written, with no task: its
    // row names it by its own words rather than by a schedule label.
    const note = String(s.kind || '') === 'notify';
    const noteText = note ? String(s.notification?.text || '') : '';
    const next = s.next_run_at ? scheduleInstantHtml(s.next_run_at) : '';
    const status = scheduleStatus(s);
    const enabled = status === 'active';
    const consumed = status === 'consumed';
    const suppressed = status === 'suppressed';
    const pendingDelete = status === 'delete_pending';
    const id = esc(s.id || '');
    // A due occurrence that waits (capacity, a missing folder, an unknown fact) says why.
    const waiting = s.hold && s.hold.reason
        ? ` · <span class="activity-tag" title="${esc(s.hold.detail || '')}">waiting: ${esc(s.hold.reason)}</span>` : '';
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
        followup_result_persistence_unavailable: 'Task record could not be saved',
    };
    const pendingDetail = pendingDelete ? ` · deletion waits for ${relation === 'independent'
        ? 'its accepted run to start' : 'its task to finish'}` : '';
    const statusLabel = pendingDelete ? 'deletion pending' : status;
    const sub = `${note ? 'reminder · ' : ''}${timing}${next && !consumed && !pendingDelete ? ` · next ${next}` : ''} · ${esc(statusLabel)}${pendingDetail} · ${esc(relation)}${binding}${s.deadline_at ? ` · deadline ${scheduleInstantHtml(s.deadline_at, { includeYear: true })}` : ''}${hold.reason ? ` · ${esc(explanations[hold.reason] || hold.reason)}` : ''}${s.followup_wait ? ` · ${esc(explanations[s.followup_wait] || s.followup_wait)}` : ''}${s.completed_at ? ' · already fired' : ''}${managed && s.skill ? ` · ${esc(s.skill)}` : ''}${waiting}${failed}`;
    // The exact hold release stays available wherever a hold exists, including on a
    // deleted row whose accepted task still needs it; it never re-enables the row.
    const holdControl = !hold.hold_id ? ''
        : relation === 'unknown'
            ? '<span class="activity-tag">resolve relationship in conversation</span>'
            : !s.hold_persisted
                ? '<span class="activity-tag">hold persistence pending</span>'
                : `<button type="button" class="btn btn-xs btn-default" data-act="schedule-toggle" data-id="${id}" data-action="restore" data-hold-id="${esc(hold.hold_id || '')}">Restore hold</button>`;
    // A consumed one-shot cannot be re-armed, so it carries no Enable: the
    // only honest control left is removing the receipt. A suppressed skill
    // row offers Restore, which asks the server to re-evaluate the skill.
    // A skill row that is disabled WITHOUT the owner's marker is held back by
    // its skill's readiness and re-arms on resync; an Enable button there
    // would offer to lift a suppression nobody applied. A pending deletion can
    // be explicitly cancelled; naming it Enable would hide that extra effect.
    const cancelDelete = !hold.hold_id && !(once && s.completed_at)
        ? `<button type="button" class="btn btn-xs btn-default" data-act="schedule-toggle" data-id="${id}" data-action="restore">Cancel deletion</button>` : '';
    const readinessHeld = managed && !enabled && !suppressed;
    const lifecycle = pendingDelete
        ? `${holdControl}${cancelDelete}`
        : status === 'waiting'
            ? '<span class="activity-tag">waiting for existing control</span>'
            : hold.hold_id
                ? holdControl
                : consumed
                    ? '<span class="activity-tag">consumed once · history</span>'
                    : readinessHeld
                        ? '<span class="activity-tag">disabled by skill readiness</span>'
                        : `<button type="button" class="btn btn-xs btn-default" data-act="schedule-toggle" data-id="${id}" data-action="${suppressed || !enabled ? 'restore' : 'disable'}">${enabled ? 'Disable' : (suppressed ? 'Restore' : 'Enable')}</button>`;
    return `<div class="activity-row activity-schedule${enabled || hold.hold_id || s.followup_wait ? '' : ' off'}">
        <div class="activity-row-main">
            <span class="activity-name"${noteText ? ` title="${esc(noteText)}"` : ''}>${esc(noteText || s.name || s.id || 'schedule')}</span>
            <span class="activity-sub">${sub}</span>
        </div>
        <div class="activity-row-actions">${lifecycle}
           <button type="button" class="btn btn-xs btn-danger" data-act="schedule-delete" data-id="${id}" data-managed="${managed ? '1' : ''}" data-note="${note ? '1' : ''}">Delete</button></div>
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
    if (!['updated', 'deleted', 'delete_deferred', 'suppressed', 'restored_not_ready',
        'hold_released', 'changed_audit_incomplete'].includes(outcome.status)) {
        return [`Schedule ${action} refused: ${detail}`, 'error'];
    }
    if (outcome.audit !== 'recorded') return [`Schedule ${action} applied, but its audit record is incomplete: ${detail}`, 'warn'];
    if (outcome.status === 'delete_deferred') return [`Schedule deletion is pending: ${detail}`, 'info'];
    if (outcome.status === 'restored_not_ready') {
        return [`Schedule suppression lifted, but it is not ready to run: ${detail}`, 'warn'];
    }
    if (outcome.ok !== true) return [`Schedule ${action} changed, but did not finish successfully: ${detail}`, 'warn'];
    // Name what actually happened: a delete on a skill row is a durable
    // suppression, and saying "deleted" would claim a removal that did not occur.
    const done = outcome.status === 'suppressed' ? 'suppressed (kept disabled until restored)' : `${action}d`;
    // Lifecycle actions govern FUTURE dispatch; a run already admitted keeps
    // going. Silence here would let the owner read the button as a stop.
    if (outcome.running_or_queued === true) {
        return [`Schedule ${done}. A task it already admitted is queued or running and was not cancelled.`, 'info'];
    }
    if (outcome.running_or_queued === null || outcome.running_or_queued === undefined) {
        return [`Schedule ${done}. Whether a task it already started is still running is unknown.`, 'info'];
    }
    return outcome.status === 'suppressed' ? [`Schedule ${done}.`, 'info'] : null;
}

export function initActivity({ mount, ws } = {}) {
    if (!mount) return { refresh: () => {} };
    let busy = false;
    let refreshRevision = 0;
    // Remembered across re-renders: the owner's retained-history disclosure state.
    let historyOpen = false;
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

    function renderQueue(queue, census) {
        if (!Array.isArray(queue?.running) || !Array.isArray(queue?.pending)) throw new Error('Queue unavailable');
        const { running, pending } = queue;
        // #322: the snapshot already carries the pause truth — a member's own
        // _budget_pause row, or a root fence covering its tree.
        const liveFences = ((queue && queue.budget_root_fences) || [])
            .filter((f) => f && ['active', 'paused'].includes(String(f.status || '')));
        const fencedRoots = new Set(liveFences.map((f) => String(f.root_task_id || '')));
        // The owner's Pause rides the same latch, typed by its cause: not money.
        const ownerFencedRoots = new Set(liveFences.filter((f) => f.cause === 'owner_pause')
            .map((f) => String(f.root_task_id || '')));
        // #1196: a row whose root fence was lifted keeps a durable HOLD instead —
        // nothing dispatches it until an explicit selection is recorded, so
        // showing it as plain "queued" would promise work that cannot start.
        const heldRow = (t) => Boolean(t && t._budget_pause_hold && !t._budget_pause_hold.selected);
        const rowBudgetPaused = (q, t, kind) => kind === 'pending' && Boolean(
            (t && t._budget_pause)
            || heldRow(t)
            || fencedRoots.has(String((t && (t.root_task_id || t.id)) || q.id || '')));
        const row = (q, kind) => {
            const t = (q && q.task) || {};
            const id = esc(q.id || t.id || '');
            const label = esc(t.title || t.objective || t.text || q.type || id || 'task');
            const rt = kind === 'running' && q.runtime_sec != null ? ` · ${Math.round(q.runtime_sec)}s` : '';
            const paused = rowBudgetPaused(q, t, kind);
            // The owner's Restart holds never-started work under the same
            // hold carrier (restart_retention.py): same Resume, its own words.
            // Work saved before an application stop (saved_work_hold, #1563) has
            // the census's same "after restart" cause; it is never a money pause.
            const restartHeld = heldRow(t)
                && ['owner_restart_hold', 'saved_work_hold'].includes(t._budget_pause_hold.reason);
            const waiting = q.owner_wait;
            const warmSleeping = kind === 'running' && waiting?.state === 'waiting'
                && waiting.reason === 'sleep' && waiting.sleep?.mode === 'warm';
            const sleeping = t._budget_pause?.reason === 'sleep' || warmSleeping;
            const ownerPaused = t._budget_pause?.reason === 'owner'
                || ownerFencedRoots.has(String(t.root_task_id || t.id || q.id || ''));
            const phase = (census?.active_chat_activities || []).find((entry) =>
                entry.activity_id === String(t.root_task_id || t.id || q.id || ''))?.phase;
            const resumable = paused && (!ownerPaused || phase === 'budget_paused');
            const kindLabel = phase === 'unknown' ? 'pause status unknown'
                : phase === 'budget_pausing' ? 'pausing' : restartHeld ? 'held after Restart'
                : (paused ? (ownerPaused ? 'paused' : sleeping ? 'sleeping' : 'paused (budget)') : sleeping ? 'sleeping' : kind);
            const meta = `${esc(kindLabel)}${q.type ? ` · ${esc(q.type)}` : ''}${rt}`;
            const hold = t.project_admission_hold;
            const projectWait = hold?.label ? ` · ${esc(hold.label)}: ${esc(hold.detail || hold.reason)}` : '';
            return `<div class="activity-row">
                <div class="activity-row-main">
                    <span class="activity-name">${label}</span>
                    <span class="activity-sub">${meta}${projectWait}</span>
                </div>
                <div class="activity-row-actions">
                    <button type="button" class="btn btn-xs btn-danger" data-act="task-control" data-id="${id}"${resumable ? ' data-budget-paused="1"' : ''}${isRootTaskRow(t, q.id || t.id) ? ' data-root="1"' : ''}>${esc(TASK_CONTROL_TRIGGER_LABEL)}</button>
                </div>
            </div>`;
        };
        // Queue facts keep their runtime/budget controls. The census adds only
        // missing identities; queued direct turns can appear in both sources.
        const known = new Set([...running, ...pending].map((q) => String(q.id || q.task?.id || '')));
        const live = (Array.isArray(census?.active_chat_activities) ? census.active_chat_activities : [])
            .filter((a) => a && !known.has(String(a.activity_id || '')));
        const parts = [...running.map((q) => row(q, 'running')), ...pending.map((q) => row(q, 'pending')),
            ...live.map((a) => liveActivityRowHtml(a))];
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
            enabled && bg.next_wake_at ? `next wake ${esc(formatWhen(bg.next_wake_at))}` : '',
            bg.last_wake_at ? `last wake ${esc(formatWhen(bg.last_wake_at))}${bg.last_wake_outcome ? ` (${esc(bg.last_wake_outcome)})` : ''}` : '',
            allowanceLabel(bg.spent_24h_usd, bg.daily_usd, bg.unknown_unmetered, bg.integrity_degraded)
                ? `allowance ${esc(allowanceLabel(bg.spent_24h_usd, bg.daily_usd, bg.unknown_unmetered, bg.integrity_degraded))} (24 h)` : '',
            Number.isFinite(Number(bg.max_tasks)) ? `tasks ${Number(bg.tasks_running || 0)}/${Number(bg.max_tasks)}` : '',
        ].filter(Boolean).join(' · ');
        return `<div class="activity-row">
            <div class="activity-row-main">
                <span class="activity-name">Background consciousness</span>
                <span class="activity-sub">${enabled ? 'enabled' : 'disabled'}${detail ? ` · ${detail}` : ''}${facts ? ` · ${facts}` : ''}</span>
            </div>
            <div class="activity-row-actions">
                <button type="button" class="btn btn-xs btn-default" data-act="bg-toggle" data-enabled="${enabled ? '1' : '0'}"${ws ? '' : ' disabled'}>${enabled ? 'Stop' : 'Start'}</button>
            </div>
        </div>`;
    }

    function renderSchedules(data) {
        if (!Array.isArray(data?.tasks)) throw new Error('Schedules unavailable');
        const tasks = data.tasks;
        if (!tasks.length) return '<div class="activity-empty">No scheduled tasks.</div>';
        // Retained rows — a fired one-shot, a suppressed skill row — are history,
        // not schedules wearing a disabled flag. They stay reachable (and
        // restorable) but collapsed, so the standing schedules are the list.
        const standing = [];
        const retained = [];
        for (const s of tasks) (isRetainedSchedule(s) ? retained : standing).push(s);
        const parts = standing.length
            ? standing.map(scheduleRowHtml)
            : ['<div class="activity-empty">No active or disabled schedules.</div>'];
        if (retained.length) {
            // A refresh rebuilds this markup, so the disclosure carries the state
            // the owner LEFT it in: a poll that silently re-collapses the history
            // they just opened (and drops the focus they had inside it) reads as
            // the app undoing their action.
            parts.push(`<details class="activity-history" data-activity-history${historyOpen ? ' open' : ''}>
                <summary>History &amp; suppressed (${retained.length})</summary>
                ${retained.map(scheduleRowHtml).join('')}
            </details>`);
        }
        return parts.join('');
    }

    function rememberDisclosure() {
        const details = mount.querySelector('[data-activity-history]');
        if (details) historyOpen = Boolean(details.open ?? details.hasAttribute('open'));
    }

    function focusedControl() {
        const active = document.activeElement;
        return active && mount.contains?.(active) ? active.closest('[data-act]') : null;
    }

    function restoreFocus(previous) {
        if (!previous) return;
        const target = [...mount.querySelectorAll('[data-act]')].find((el) => (
            el.dataset.act === previous.act && (el.dataset.id || '') === previous.id));
        if (!target) return;
        // Focus lives inside the disclosure for a retained row: open it rather
        // than focusing a control the owner cannot see.
        const details = target.closest?.('[data-activity-history]');
        if (details && !details.open) details.open = true;
        target.focus?.();
    }

    async function refresh() {
        const revision = ++refreshRevision;
        rememberDisclosure();
        const focused = focusedControl();
        const focusedKey = focused && { act: focused.dataset.act, id: focused.dataset.id || '' };
        sections.forEach(({ root, status, loaded }) => {
            root.setAttribute('aria-busy', 'true');
            setInlineStatus(status, loaded ? 'Refreshing… Previously loaded values shown.' : 'Loading…');
        });
        const results = await Promise.allSettled([
            // This view renders only the queue; queue_only skips the whole
            // task-results scan server-side (v6.9x P2).
            getJson('/api/tasks?queue_only=1'),
            getJson('/api/state'),
            getJson('/api/schedules'),
        ]);
        if (revision !== refreshRevision) return;
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
                section.loaded = true;
                setInlineStatus(status, '');
            } catch {
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
        if (!btn || busy) return;
        const act = btn.dataset.act;
        const id = btn.dataset.id || '';
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
            openTaskControlMenu(btn, {
                cancelPending: taskCancelPending(stored),
                budgetPaused: btn.dataset.budgetPaused === '1',
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
                        // never orphans its running subagents. Soft stop answers
                        // 202 with the intent open; immediate answers after the
                        // teardown — either way the refresh shows honest state.
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
                // A skill-declared schedule cannot be removed: the skill's manifest
                // would recreate it. Delete SUPPRESSES it durably, and the dialog
                // says so before anything is sent.
                const managedRow = btn.dataset.managed === '1';
                const noun = btn.dataset.note === '1' ? 'reminder' : 'schedule';
                const confirmedDelete = await openConfirmDialog({
                    title: managedRow ? 'Suppress skill schedule' : `Delete ${noun}`,
                    body: managedRow
                        ? 'This schedule is declared by an installed skill and cannot be removed; Delete keeps it suppressed until you Restore it. Suppress it?'
                        : `Delete this ${noun}?`,
                    confirmLabel: managedRow ? 'Suppress' : 'Delete',
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

    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event?.detail?.tab === 'activity') refresh();
    });

    return { refresh };
}
