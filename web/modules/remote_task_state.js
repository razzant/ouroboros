// Pure reducers for remote (SSH-placed) state — no DOM, no fetch, no imports.
// This is the ONE place that decides what a remote status MEANS; connections_ui,
// activity, and chat only render what these functions return. Keeping the
// decisions here is why the same task can be shown in two surfaces without two
// slightly different notions of "degraded".

const CONNECTION_STATES = new Set([
    'connecting', 'ready', 'degraded', 'disconnected', 'unknown',
]);
const CANCELLABLE_TASK_STATES = new Set([
    'requested', 'connecting', 'scheduled', 'pending', 'running',
]);
const TERMINAL_TASK_STATES = new Set([
    'completed', 'failed', 'cancelled', 'rejected', 'rejected_duplicate',
    'interrupted', 'timed_out', 'timeout',
]);
const DETAIL_TEXT_LIMIT = 4000;
const DETAIL_COLLECTION_LIMIT = 32;

// The gateway's typed 503 for a build that carries no ssh transport at all
// (ouroboros/gateway/connections.py). It is NOT "try again later": there is
// nothing to restart and nothing to wait for, so every surface must render it
// as a settled fact rather than as pending work.
export const REMOTE_TRANSPORT_UNAVAILABLE = 'remote_transport_unavailable';
export const REMOTE_TRANSPORT_UNAVAILABLE_NOTE = (
    'The remote SSH transport is not part of this build, so connection actions '
    + 'cannot run. Saved connections and their trust history are untouched.'
);

function text(value, limit = 512) {
    return String(value ?? '').replace(/\0/g, '').slice(0, limit);
}

// What a bounded collection says about the part it dropped. A depth cap already
// announces itself as '[truncated]'; breadth must do the same, because a silent cap
// makes a shortened diagnostic indistinguishable from a complete one — the single
// worst ambiguity to leave in a forensic payload. Mirrors
// `gateway/connections.py::_omission_note`.
const OMITTED_KEY = '…';

function omissionNote(total, kept) {
    return `${total - kept} of ${total} entries omitted (bounded live projection)`;
}

function bounded(value, depth = 0) {
    if (depth >= 4) return '[truncated]';
    if (typeof value === 'string') return text(value, DETAIL_TEXT_LIMIT);
    if (value == null || ['boolean', 'number'].includes(typeof value)) return value;
    if (Array.isArray(value)) {
        const kept = value.slice(0, DETAIL_COLLECTION_LIMIT)
            .map((item) => bounded(item, depth + 1));
        return value.length > DETAIL_COLLECTION_LIMIT
            ? [...kept, omissionNote(value.length, DETAIL_COLLECTION_LIMIT)]
            : kept;
    }
    if (typeof value === 'object') {
        const entries = Object.entries(value);
        const kept = Object.fromEntries(
            entries.slice(0, DETAIL_COLLECTION_LIMIT)
                .map(([key, item]) => [text(key, 128), bounded(item, depth + 1)]),
        );
        if (entries.length > DETAIL_COLLECTION_LIMIT) {
            kept[OMITTED_KEY] = omissionNote(entries.length, DETAIL_COLLECTION_LIMIT);
        }
        return kept;
    }
    return text(value, DETAIL_TEXT_LIMIT);
}

/**
 * True when this payload is the typed "no ssh transport in this build" refusal.
 * Accepts a thrown fetch error (`.body`), a raw wire payload (`error_code`), or
 * an already-normalized state (`errorCode`), so no caller has to remember which
 * spelling it is holding.
 */
export function isRemoteTransportUnavailable(source) {
    if (!source || typeof source !== 'object') return false;
    const body = source.body && typeof source.body === 'object' ? source.body : source;
    return text(body.errorCode || body.error_code || '', 128) === REMOTE_TRANSPORT_UNAVAILABLE;
}

/** The honest human sentence for a state a retry cannot move. */
export function remoteStateNote(state = {}) {
    return isRemoteTransportUnavailable(state) ? REMOTE_TRANSPORT_UNAVAILABLE_NOTE : '';
}

/**
 * One rendering of a failed owner action, including the gateway's typed next
 * step. Used by every remote surface so an error never degrades to "[object
 * Object]" in one place and a full sentence in another.
 */
export function remoteActionErrorText(error) {
    const body = (error && typeof error === 'object' && error.body && typeof error.body === 'object')
        ? error.body
        : (error && typeof error === 'object' ? error : {});
    if (isRemoteTransportUnavailable(error)) return REMOTE_TRANSPORT_UNAVAILABLE_NOTE;
    const message = body.error || (error && error.message) || String(error ?? '');
    const next = text(body.action || '', 200);
    return `${text(message, 2000)}${next ? ` · Next: ${next}` : ''}`;
}

/**
 * The remote placement of a task, read from the SEALED workspace ref in task
 * metadata. `executor_ref` is only a fallback projection of the same fact
 * (RWS v2 §3.1: the ref is persisted, the executor ref is derived).
 */
export function remotePlacementFromTask(task = {}) {
    const metadata = task?.metadata && typeof task.metadata === 'object' ? task.metadata : {};
    const sealed = metadata._sealed_workspace_ref && typeof metadata._sealed_workspace_ref === 'object'
        ? metadata._sealed_workspace_ref
        : {};
    const executor = metadata.executor_ref && typeof metadata.executor_ref === 'object'
        ? metadata.executor_ref
        : {};
    const connectionId = text(
        sealed.connection_id
        || (executor.type === 'ssh' ? (executor.connection_id || executor.id) : ''),
    );
    if (!connectionId) return null;
    return {
        connectionId,
        projectId: text(task.project_id || metadata.project_id || ''),
        workspaceId: text(sealed.workspace_id || executor.workspace_id || ''),
    };
}

/**
 * The admission evidence Home ACTUALLY writes into a task record: the remote
 * preflight summary sealed into metadata when the placement was admitted
 * (`workspace_admission.placement_preflight_summary`, reaching the browser through
 * `/api/tasks` and `/api/tasks/{id}` — `outcomes.public_task_result` passes task
 * metadata through). Its remote arm is target IDENTITY (`placement: 'ssh'` plus the
 * host/release the target reported), so its presence means the TARGET answered when
 * this task was admitted. A remote preflight that could NOT be taken discloses an
 * `error` instead of raising — the placement is already admitted by then, and a task
 * must not be lost because its context block could not be decorated — so that field
 * is the negative half of the same evidence.
 *
 * `answered` and `refused` are deliberately not one boolean: "the target answered",
 * "the target refused", and "there is no evidence either way" are three states, and
 * collapsing the last two is what lets a surface invent SSH health it never had.
 */
function remoteAdmissionEvidence(task = {}) {
    const metadata = task?.metadata && typeof task.metadata === 'object' ? task.metadata : {};
    const preflight = (
        metadata.workspace_preflight && typeof metadata.workspace_preflight === 'object'
    )
        ? metadata.workspace_preflight
        : null;
    if (!preflight) return { answered: false, refused: false };
    return {
        answered: text(preflight.placement) === 'ssh' && !preflight.error,
        refused: Boolean(preflight.error),
    };
}

function inferredStatus(task = {}) {
    const taskStatus = text(task.status || '').toLowerCase();
    const { answered, refused } = remoteAdmissionEvidence(task);
    if (['requested', 'connecting', 'recovery_required'].includes(taskStatus)) {
        return refused ? 'degraded' : 'connecting';
    }
    if (['scheduled', 'pending', 'running'].includes(taskStatus)) {
        if (answered) return 'ready';
        return refused ? 'degraded' : 'unknown';
    }
    // A generic task failure says nothing about SSH health. Only a task whose
    // ADMISSION could not reach the target is projected as degraded; model/test
    // failures after a target that answered, and completed tasks after a reload,
    // deliberately remain unknown without a live connection_state frame.
    if (taskStatus === 'failed' && refused) return 'degraded';
    return 'unknown';
}

export function normalizeRemoteTaskState(event = {}, task = {}) {
    const placement = remotePlacementFromTask(task) || {};
    const rawStatus = text(event.status || '').toLowerCase();
    const hasTypedStatus = CONNECTION_STATES.has(rawStatus);
    const status = hasTypedStatus ? rawStatus : inferredStatus(task);
    const taskStatus = text(task.status || '').toLowerCase();
    const diagnostic = event.diagnostic && typeof event.diagnostic === 'object'
        ? bounded(event.diagnostic)
        : null;
    const rawLogRefs = Array.isArray(event.log_refs)
        ? event.log_refs.filter((item) => item && typeof item === 'object')
        : [];
    const logRefs = rawLogRefs.slice(0, DETAIL_COLLECTION_LIMIT).map((item) => bounded(item));
    // The gateway's own TOTAL wins when it capped before we saw the list; otherwise
    // what arrived IS the total. Either way the reader can tell `logRefs` apart from
    // "all the log refs there are" — `logRefsCount > logRefs.length` is the whole test.
    const logRefsCount = Math.max(
        Number(event.log_refs_count) || 0,
        rawLogRefs.length,
    );
    return {
        taskId: text(event.task_id || task.task_id || task.id || ''),
        connectionId: text(event.connection_id || placement.connectionId || ''),
        projectId: text(event.project_id || placement.projectId || ''),
        status,
        phase: text(event.phase || (status === 'connecting' ? 'admission' : '')),
        completion: text(event.completion || taskStatus || ''),
        errorCode: text(event.error_code || task.reason_code || ''),
        action: text(event.action || ''),
        diagnostic,
        logRefs,
        logRefsCount,
        taskStatus,
        // Where the status came from, because `remoteTaskActions` may only offer a
        // Reconnect against a state that rests on real connection evidence. A
        // DERIVED status that says something ('ready'/'connecting'/'degraded') rests
        // on the admission evidence above; a derived 'unknown' says nothing, and
        // nothing is not a basis for offering repair.
        stateSource: hasTypedStatus
            ? 'live'
            : (status === 'unknown' ? 'derived' : 'admission'),
    };
}

export function mergeRemoteTaskState(previous = {}, event = {}, task = {}) {
    const next = normalizeRemoteTaskState(event, task);
    const has = (key) => Object.prototype.hasOwnProperty.call(event || {}, key);
    return {
        ...previous,
        ...next,
        status: has('status') ? next.status : (previous.status || next.status),
        phase: has('phase') ? next.phase : (previous.phase || next.phase),
        completion: has('completion')
            ? next.completion
            : (previous.completion || next.completion),
        errorCode: has('error_code')
            ? next.errorCode
            : (has('status') && next.status === 'ready'
                ? ''
                : (previous.errorCode || next.errorCode)),
        action: has('action')
            ? next.action
            : (has('status') && next.status === 'ready'
                ? ''
                : (previous.action || next.action)),
        stateSource: has('status')
            ? next.stateSource
            : (previous.stateSource || next.stateSource),
        diagnostic: next.diagnostic || previous.diagnostic || null,
        // The list and its total move together — a retained list beside a fresh
        // count would disclose an omission that does not match what is shown.
        logRefs: next.logRefs.length ? next.logRefs : (previous.logRefs || []),
        logRefsCount: next.logRefs.length
            ? next.logRefsCount
            : (previous.logRefsCount || next.logRefsCount),
        taskStatus: next.taskStatus || previous.taskStatus || '',
    };
}

export function remoteTaskActions(state = {}) {
    const taskStatus = text(state.taskStatus || '').toLowerCase();
    const hasConnectionEvidence = ['live', 'admission'].includes(state.stateSource);
    const terminal = TERMINAL_TASK_STATES.has(taskStatus);
    return {
        canCancel: Boolean(state.taskId && CANCELLABLE_TASK_STATES.has(taskStatus || state.completion)),
        canReconnect: Boolean(
            state.connectionId
            && hasConnectionEvidence
            // A build without the transport cannot reconnect: offering the
            // button would be a retry loop over a structural absence.
            && !isRemoteTransportUnavailable(state)
            && (!terminal || taskStatus === 'failed')
            && ['degraded', 'disconnected', 'unknown'].includes(state.status),
        ),
        terminal,
    };
}

/**
 * Fold one `connection_state` frame into a `taskId → state` Map.
 *
 * The frame may name a task (`task_id`) or only a connection. A connection-wide
 * frame fans out to EVERY task currently placed on that connection — both Chat
 * and Activity need exactly that, so the walk lives here instead of being
 * written twice with two slightly different notions of "which tasks".
 *
 * `taskFor(taskId)` supplies the durable task row behind an id (or a minimal
 * stand-in) so status derivation keeps working for tasks the caller has not
 * loaded. Returns a NEW Map plus the ids that changed, so the caller re-renders
 * only those; the input Map is never mutated.
 */
export function reduceRemoteConnectionEvent(states, event = {}, taskFor = () => ({})) {
    const next = new Map(states instanceof Map ? states : []);
    const explicitTaskId = text(event.task_id || '');
    if (explicitTaskId) {
        next.set(explicitTaskId, mergeRemoteTaskState(
            next.get(explicitTaskId),
            { ...event, task_id: explicitTaskId },
            taskFor(explicitTaskId) || {},
        ));
        return { states: next, taskIds: [explicitTaskId] };
    }
    const connectionId = text(event.connection_id || '');
    if (!connectionId) return { states: next, taskIds: [] };
    const taskIds = [...next.keys()].filter(
        (taskId) => next.get(taskId)?.connectionId === connectionId,
    );
    for (const taskId of taskIds) {
        next.set(taskId, mergeRemoteTaskState(
            next.get(taskId),
            { ...event, task_id: taskId },
            taskFor(taskId) || {},
        ));
    }
    return { states: next, taskIds };
}

/**
 * What a successful reconnect means for THIS task. A reconnect repairs the
 * connection; it never replays work, so a task that already reached a terminal
 * status must be told so explicitly rather than left looking resumable.
 */
export function remoteReconnectNotice(state = {}) {
    return remoteTaskActions(state).terminal
        ? 'Connection is ready. This task already finished and was not replayed — start a new task to retry it.'
        : 'Connection reconnected and reconciled.';
}

export function remoteStateLabel(status = '') {
    return ({
        connecting: 'Connecting',
        ready: 'Ready',
        degraded: 'Degraded',
        disconnected: 'Disconnected',
        unknown: 'Unknown',
    })[status] || 'Unknown';
}

export function remoteStateSummary(state = {}) {
    return [
        remoteStateLabel(state.status),
        state.phase ? `phase: ${state.phase}` : '',
        state.completion ? `completion: ${state.completion}` : '',
        state.errorCode ? `error: ${state.errorCode}` : '',
    ].filter(Boolean).join(' · ');
}

export function remoteStateDetails(state = {}) {
    const parts = [];
    if (state.diagnostic) parts.push({ label: 'Diagnostic', value: state.diagnostic });
    const logRefs = state.logRefs || [];
    for (const ref of logRefs) {
        const label = text(ref.stream || ref.name || ref.kind || 'Full log', 80);
        parts.push({ label, value: ref });
    }
    // A bounded list gets a visible last row rather than just ending. Otherwise the
    // owner reads "these are the logs" where the truth is "these are some of them",
    // and a missing log ref is exactly what they would go looking for next.
    const omitted = (Number(state.logRefsCount) || logRefs.length) - logRefs.length;
    if (omitted > 0) {
        parts.push({
            label: 'Omitted log references',
            value: { omitted, total: Number(state.logRefsCount) || logRefs.length },
        });
    }
    return parts;
}

export function remoteDetailText(value, limit = 12000) {
    const rendered = JSON.stringify(value, null, 2) || '';
    return rendered.length > limit
        ? `${rendered.slice(0, limit)}\n… diagnostic preview truncated`
        : rendered;
}
