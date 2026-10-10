// Pure presentation helpers for the sidebar activity dots.
//
// The server's active_chat_activities census is the only source of liveness.
// This module deliberately owns no timer, request, socket or durable state. A
// caller supplies the previous projection when reconciling a partial census.

import { activeModelWaits, mergeModelWaits } from './model_wait.js';
import { censusTaskPhase } from './task_phase_chip.js';

function activityId(row) {
    return String(row?.activity_id || '').trim();
}

function projectId(row) {
    return String(row?.project_id || '').trim();
}

function isChildActivity(row) {
    if (row?.is_child === true || row?.child === true) return true;
    const parent = String(row?.parent_task_id || '').trim();
    if (!parent) return false;
    const root = String(row?.root_task_id || '').trim();
    return !root || root !== activityId(row);
}

function waitingModel(row) {
    const waits = row?.model_waits;
    if (!waits || typeof waits !== 'object') return false;
    // The same admission rule the chat card applies: a malformed wait row is
    // dropped here exactly as there, so the sidebar cannot go static on a row
    // the card would never show.
    const admitted = mergeModelWaits({}, waits);
    return activeModelWaits(admitted, false, Number(row?.task_attempt) || 0).length > 0;
}

/**
 * Summarize one or more census rows into one accessible marker state.
 *
 * The summary intentionally contains no counts. An independently working row
 * wins motion; a wait on that same row suppresses its motion while remaining in
 * the label so mixed projects keep both facts available to assistive
 * technology and the marker tooltip.
 */
export function summarizeProjectActivities(rows = []) {
    const phases = new Set();
    const waits = new Set();
    const unknown = new Set();
    let motion = false;
    let queued = false;
    const order = ['working', 'thinking', 'finalizing', 'queued', 'budget_pausing', 'budget_paused', 'unknown'];
    const ordered = (Array.isArray(rows) ? rows : []).filter(row => row && typeof row === 'object')
        .sort((a, b) => order.indexOf(a.phase) - order.indexOf(b.phase));
    for (const row of ordered) {
        const view = censusTaskPhase(row, null, !row._activityUnconfirmed, waitingModel(row));
        const label = [view.text, view.secondary].filter(Boolean).join(' · ');
        (view.waiting ? waits : view.phase === 'unknown' ? unknown : phases).add(label);
        motion ||= view.motion || Boolean(view.secondaryMotion);
        queued ||= view.phase === 'queued';
    }
    const parts = [...phases, ...unknown, ...waits];
    const waiting = waits.size > 0;
    const state = motion ? 'working' : waiting ? 'waiting'
        : queued ? 'queued' : parts.length ? 'unknown' : 'idle';
    return {
        state,
        motion,
        waiting,
        label: parts.join(' · '),
    };
}

/** Build project and direct-conversation summaries from census rows. */
export function buildProjectActivityIndex(rows = []) {
    const byProjectRows = new Map();
    const seen = new Set();
    for (const row of Array.isArray(rows) ? rows : []) {
        const id = activityId(row);
        if (!id || seen.has(id) || isChildActivity(row)) continue;
        seen.add(id);
        const pid = projectId(row);
        if (pid) byProjectRows.set(pid, [...(byProjectRows.get(pid) || []), row]);
    }
    const byProject = new Map();
    for (const [pid, projectRows] of byProjectRows) {
        byProject.set(pid, summarizeProjectActivities(projectRows));
    }
    const aggregateSummary = summarizeProjectActivities([...byProjectRows.values()].flat());
    return {
        byProject,
        aggregate: aggregateSummary,
    };
}

/**
 * Reconcile a census against the previous rows. A complete supervisor-ready
 * response authorizes absence-based clearing. Partial, unavailable or
 * disconnected responses retain omitted rows as unknown, with no motion.
 * Positive rows in a partial census still describe their own observed state.
 */
export function reconcileProjectActivityCensus(previous = new Map(), data = {}) {
    const prior = previous instanceof Map ? previous : new Map();
    const incoming = data?.active_chat_activities;
    const complete = Array.isArray(incoming) && data.active_chat_activities_complete === true
        && data.supervisor_ready === true;
    const next = new Map();
    if (!complete) {
        for (const [id, row] of prior) next.set(id, { ...row, _activityUnconfirmed: true });
    }
    for (const row of Array.isArray(incoming) ? incoming : []) {
        const id = activityId(row);
        if (id && !isChildActivity(row)) next.set(id, { ...row, _activityUnconfirmed: false });
    }
    return {
        rows: next,
        complete,
    };
}
