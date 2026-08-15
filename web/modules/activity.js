// Activity dashboard subtab (P4): a single observability + minimal-control view for
// cron/scheduled tasks, what is running/queued now, and background consciousness.
// Management is DIRECT mechanical control via existing APIs (cancel a task, enable/
// disable/delete a MANUAL schedule, start/stop background consciousness). Skill-managed
// schedules are READ-ONLY ("managed by skill") because the lifecycle resync would
// overwrite a direct toggle (supervisor/queue.py) — control those via the skill itself.

// Remote placement is PRESENTATION-only here: task lifecycle stays queue-owned,
// and the SSH retry is a connection-level reconnect that never replays work.

import { apiClient, apiFetch, cancelTask } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { showToast } from './toast.js';
import {
    mergeRemoteTaskState,
    normalizeRemoteTaskState,
    reduceRemoteConnectionEvent,
    remoteActionErrorText,
    remoteDetailText,
    remotePlacementFromTask,
    remoteReconnectNotice,
    remoteStateDetails,
    remoteStateLabel,
    remoteStateNote,
    remoteStateSummary,
    remoteTaskActions,
} from './remote_task_state.js';

function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, (c) => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

async function getJson(url) {
    try {
        const resp = await apiFetch(url, { cache: 'no-store' });
        if (resp && typeof resp.json === 'function') {
            if (resp.ok === false) return null;
            return await resp.json();
        }
        return resp;
    } catch {
        return null;
    }
}

// A schedule synced from a skill manifest is reconciled from skill readiness, so a
// direct enable/disable/delete here would be temporary/misleading — show it read-only.
function isSkillManaged(s) {
    return Boolean(s && (String(s.source || '') === 'skill_manifest' || String(s.skill || '')));
}

function taskIdOf(task = {}) {
    return String(task.task_id || task.id || '');
}

function renderRemoteDetails(state) {
    const details = remoteStateDetails(state);
    if (!details.length) return '';
    return `<details class="activity-remote-details">
        <summary>Diagnostics &amp; logs</summary>
        ${details.map((item) => {
            // Only a same-origin gateway path is turned into a link; anything
            // else stays inert text, so a remote-supplied ref cannot navigate.
            const href = String(item.value?.download_url || item.value?.url || '');
            const link = /^\/api\/[A-Za-z0-9_?&=./%-]+$/.test(href)
                ? `<a href="${esc(href)}" target="_blank" rel="noopener">Open full log</a>`
                : '';
            return `<div class="activity-remote-evidence">
                <div><strong>${esc(item.label)}</strong>${link}</div>
                <pre>${esc(remoteDetailText(item.value))}</pre>
            </div>`;
        }).join('')}
    </details>`;
}

function renderRemoteStatus(state) {
    if (!state) return '';
    const tone = ['degraded', 'disconnected'].includes(state.status) ? 'warn' : state.status;
    return `<span class="activity-connection-state" data-state="${esc(tone)}">${esc(remoteStateLabel(state.status))}</span>`;
}

function renderRemoteActions(state) {
    const actions = remoteTaskActions(state);
    return [
        actions.canReconnect
            ? `<button type="button" class="btn btn-xs btn-default" data-act="connection-reconnect" data-id="${esc(state.connectionId)}" data-task-id="${esc(state.taskId)}">Reconnect</button>`
            : '',
        actions.canCancel
            ? `<button type="button" class="btn btn-xs btn-danger" data-act="task-cancel" data-id="${esc(state.taskId)}">Cancel</button>`
            : '',
    ].filter(Boolean).join('');
}

export function initActivity({ mount, ws } = {}) {
    if (!mount) return { refresh: () => {} };
    let busy = false;
    let tasksData = null;
    let schedulesData = null;
    let stateData = null;
    let notice = null;
    let remoteStates = new Map();

    function durableTasks() {
        return Array.isArray(tasksData?.tasks) ? tasksData.tasks : [];
    }

    function taskFor(taskId) {
        return durableTasks().find((task) => taskIdOf(task) === taskId)
            || { task_id: taskId, status: remoteStates.get(taskId)?.taskStatus || '' };
    }

    // Remembering the merged state per task is what lets a live frame survive the
    // next /api/tasks poll: the durable row alone can only derive a coarse status.
    function stateForTask(task) {
        const taskId = taskIdOf(task);
        const previous = remoteStates.get(taskId);
        const state = previous
            ? mergeRemoteTaskState(previous, {}, task)
            : normalizeRemoteTaskState({}, task);
        if (taskId) remoteStates.set(taskId, state);
        return state;
    }

    // The fan-out (a connection-wide frame touching every task on it) lives in
    // remote_task_state.js; this is only the local Map swap.
    function applyConnectionEvent(event) {
        const { states } = reduceRemoteConnectionEvent(remoteStates, event, taskFor);
        remoteStates = states;
    }

    function renderTaskRow(task, kind, queueRow = null) {
        const id = taskIdOf(task) || String(queueRow?.id || '');
        const placement = remotePlacementFromTask(task);
        const remote = placement ? stateForTask({ ...task, task_id: id }) : null;
        const runtime = kind === 'running' && queueRow?.runtime_sec != null
            ? ` · ${Math.round(queueRow.runtime_sec)}s`
            : '';
        const type = queueRow?.type ? ` · ${esc(queueRow.type)}` : '';
        const label = task.title || task.description || task.objective || task.text
            || queueRow?.type || id || 'task';
        const remoteMeta = remote ? ` · SSH ${esc(remoteStateSummary(remote))}` : '';
        const note = remote ? remoteStateNote(remote) : '';
        // A generic Cancel is only for LOCAL rows; a remote row's actions come
        // from remoteTaskActions so the two surfaces cannot disagree on them.
        const genericCancel = remote
            ? ''
            : `<button type="button" class="btn btn-xs btn-danger" data-act="task-cancel" data-id="${esc(id)}">Cancel</button>`;
        return `<div class="activity-row${remote ? ' activity-row-remote' : ''}" data-task-id="${esc(id)}">
            <div class="activity-row-main">
                <span class="activity-name">${esc(label)}</span>
                <span class="activity-sub">${esc(kind)}${type}${runtime}${remoteMeta}</span>
                ${note ? `<span class="activity-sub activity-remote-note">${esc(note)}</span>` : ''}
                ${renderRemoteDetails(remote)}
            </div>
            <div class="activity-row-actions">
                ${renderRemoteStatus(remote)}
                ${remote ? renderRemoteActions(remote) : ''}
                ${genericCancel}
            </div>
        </div>`;
    }

    function renderQueue(queue) {
        const running = Array.isArray(queue?.running) ? queue.running : [];
        const pending = Array.isArray(queue?.pending) ? queue.pending : [];
        const durableById = new Map(durableTasks().map((task) => [taskIdOf(task), task]));
        const row = (entry, kind) => {
            const queued = entry?.task || {};
            const id = String(entry?.id || queued.id || '');
            // The queue row is thin; the durable row carries the metadata the
            // placement read needs, so merge before deciding local vs remote.
            return renderTaskRow({ ...durableById.get(id), ...queued, task_id: id }, kind, entry);
        };
        const parts = [
            ...running.map((entry) => row(entry, 'running')),
            ...pending.map((entry) => row(entry, 'pending')),
        ];
        return parts.length ? parts.join('') : '<div class="activity-empty">Nothing running or queued.</div>';
    }

    function renderRemoteTasks(queue) {
        const queuedIds = new Set([
            ...(Array.isArray(queue?.running) ? queue.running : []),
            ...(Array.isArray(queue?.pending) ? queue.pending : []),
        ].map((entry) => String(entry?.id || entry?.task?.id || '')));
        const rows = durableTasks()
            .filter((task) => remotePlacementFromTask(task) && !queuedIds.has(taskIdOf(task)))
            .slice(0, 10)
            .map((task) => renderTaskRow(task, String(task.status || 'unknown')));
        return rows.length ? rows.join('') : '<div class="activity-empty">No recent SSH tasks.</div>';
    }

    function renderBg(stateData) {
        const enabled = Boolean(stateData && stateData.bg_consciousness_enabled);
        const bg = (stateData && stateData.bg_consciousness_state) || {};
        const detail = esc(bg.detail || bg.last_idle_reason || (enabled ? 'running' : 'disabled'));
        return `<div class="activity-row">
            <div class="activity-row-main">
                <span class="activity-name">Background consciousness</span>
                <span class="activity-sub">${enabled ? 'enabled' : 'disabled'}${detail ? ` · ${detail}` : ''}</span>
            </div>
            <div class="activity-row-actions">
                <button type="button" class="btn btn-xs btn-default" data-act="bg-toggle" data-enabled="${enabled ? '1' : '0'}"${ws ? '' : ' disabled'}>${enabled ? 'Stop' : 'Start'}</button>
            </div>
        </div>`;
    }

    function renderSchedules(data) {
        const tasks = (data && Array.isArray(data.tasks)) ? data.tasks : [];
        if (!tasks.length) return '<div class="activity-empty">No scheduled tasks.</div>';
        return tasks.map((s) => {
            const managed = isSkillManaged(s);
            const cron = esc((s.trigger && s.trigger.expr) || s.cron || '');
            const next = esc(s.next_run_at || '');
            const enabled = s.enabled !== false;
            const id = esc(s.id || '');
            const sub = `${cron}${next ? ` · next ${next}` : ''}${managed && s.skill ? ` · ${esc(s.skill)}` : ''}`;
            const actions = managed
                ? '<span class="activity-tag">managed by skill</span>'
                : `<button type="button" class="btn btn-xs btn-default" data-act="schedule-toggle" data-id="${id}">${enabled ? 'Disable' : 'Enable'}</button>
                   <button type="button" class="btn btn-xs btn-danger" data-act="schedule-delete" data-id="${id}">Delete</button>`;
            return `<div class="activity-row${enabled ? '' : ' off'}">
                <div class="activity-row-main">
                    <span class="activity-name">${esc(s.name || s.id || 'schedule')}</span>
                    <span class="activity-sub">${sub}</span>
                </div>
                <div class="activity-row-actions">${actions}</div>
            </div>`;
        }).join('');
    }

    // render() is separate from refresh() so a live connection_state frame can
    // repaint without re-polling three endpoints.
    function render() {
        mount.innerHTML = `
            <div class="activity-scroll">
                ${notice ? `<div class="activity-notice" data-tone="${esc(notice.tone)}" role="status">${esc(notice.text)}</div>` : ''}
                <div class="activity-section">
                    <h3 class="activity-h">Running &amp; queued</h3>
                    ${renderQueue(tasksData?.queue)}
                </div>
                <div class="activity-section">
                    <h3 class="activity-h">SSH tasks</h3>
                    ${renderRemoteTasks(tasksData?.queue)}
                </div>
                <div class="activity-section">
                    <h3 class="activity-h">Background</h3>
                    ${renderBg(stateData)}
                </div>
                <div class="activity-section">
                    <h3 class="activity-h">Scheduled (cron)</h3>
                    ${renderSchedules(schedulesData)}
                </div>
            </div>
        `;
    }

    async function refresh() {
        if (!tasksData) mount.innerHTML = '<div class="activity-loading">Loading activity…</div>';
        const [sched, tasks, st] = await Promise.all([
            getJson('/api/schedules'),
            // NOT queue_only=1 (v6.9x P2): that skips the task-results scan
            // server-side, and this view no longer renders only the queue — the
            // SSH section lists recent remote tasks that already LEFT it, which
            // only the durable rows carry. limit=50 keeps that scan bounded
            // (it is also the server default) and is wide enough to surface them.
            getJson('/api/tasks?limit=50'),
            getJson('/api/state'),
        ]);
        schedulesData = sched;
        tasksData = tasks;
        stateData = st;
        render();
    }

    async function findSchedule(id) {
        const data = await getJson('/api/schedules');
        const tasks = (data && Array.isArray(data.tasks)) ? data.tasks : [];
        return tasks.find((s) => String(s.id) === String(id)) || null;
    }

    mount.addEventListener('click', async (event) => {
        const btn = event.target.closest('[data-act]');
        if (!btn || busy) return;
        const act = btn.dataset.act;
        const id = btn.dataset.id || '';
        const taskId = btn.dataset.taskId || '';
        busy = true;
        btn.disabled = true;
        notice = null;
        try {
            if (act === 'task-cancel') {
                // Same declared semantics as the chat card's "Cancel run" (v6.82):
                // the task AND its live subtree, so cancelling an orchestrator here
                // never orphans its running subagents.
                const confirmedCancel = await openConfirmDialog({
                    title: 'Cancel task',
                    body: 'Cancel this task and all its subagents?',
                    confirmLabel: 'Cancel task',
                    cancelLabel: 'Keep running',
                    danger: true,
                });
                if (!confirmedCancel) return;
                // The endpoint answers only once the teardown is done, so a
                // successful response means the subtree really is cancelled; a
                // refusal throws and is surfaced below — and the refresh below
                // already sees terminal state, so no settling delay is needed.
                await cancelTask(id, { cascade: true });
                notice = { tone: 'ok', text: 'Task and subtree cancelled.' };
            } else if (act === 'connection-reconnect') {
                applyConnectionEvent({
                    connection_id: id,
                    task_id: taskId,
                    status: 'connecting',
                    phase: 'connect',
                    completion: 'testing',
                });
                render();
                const result = await apiClient.connectionReconnect(id);
                applyConnectionEvent({ ...result, connection_id: id, task_id: taskId });
                notice = { tone: 'ok', text: remoteReconnectNotice(remoteStates.get(taskId)) };
            } else if (act === 'schedule-delete') {
                const confirmedDelete = await openConfirmDialog({
                    title: 'Delete schedule',
                    body: 'Delete this schedule?',
                    confirmLabel: 'Delete',
                    danger: true,
                });
                if (!confirmedDelete) return;
                await apiFetch(`/api/schedules/${encodeURIComponent(id)}`, { method: 'DELETE' });
            } else if (act === 'schedule-toggle') {
                // Read-modify-write the FULL record (upsert replaces by id; never drop
                // timezone/trigger/task/source) with the flipped enabled flag.
                const rec = await findSchedule(id);
                if (rec) {
                    await apiFetch('/api/schedules', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ...rec, enabled: !(rec.enabled !== false) }),
                    });
                }
            } else if (act === 'bg-toggle') {
                const on = btn.dataset.enabled === '1';
                // Reuse the existing direct control command (same as the chat header
                // toggle); /bg is a control slash-command, not a chat message to the agent.
                ws?.send?.({ type: 'command', cmd: `/bg ${on ? 'stop' : 'start'}` });
                await new Promise((resolve) => setTimeout(resolve, 400));
            }
        } catch (exc) {
            // A 404 is the documented completion race; the refresh below tells that
            // story. Anything else is real and reaches BOTH the row and the toast.
            if (exc?.status !== 404) {
                showToast(`Action failed: ${exc?.message || exc}`, 'error');
                notice = { tone: 'warn', text: remoteActionErrorText(exc) };
            }
            // A failed reconnect must ALSO land in the task's own state, not only in
            // the banner, so the row keeps showing the typed reason after the refresh
            // below repaints it. Order is free: the notice above reads the exception
            // directly, not the reduced state.
            if (act === 'connection-reconnect') {
                applyConnectionEvent({
                    ...(exc?.body || {}),
                    connection_id: id,
                    task_id: taskId,
                    status: 'degraded',
                });
            }
        } finally {
            busy = false;
            await refresh();
        }
    });

    if (ws && typeof ws.on === 'function') {
        ws.on('connection_state', (event) => {
            applyConnectionEvent(event);
            if (tasksData) render();
        });
    }

    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event?.detail?.tab === 'activity') refresh();
    });

    return { refresh };
}
