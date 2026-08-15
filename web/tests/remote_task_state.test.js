import test from 'node:test';
import assert from 'node:assert/strict';

import {
    REMOTE_TRANSPORT_UNAVAILABLE,
    isRemoteTransportUnavailable,
    mergeRemoteTaskState,
    normalizeRemoteTaskState,
    reduceRemoteConnectionEvent,
    remoteActionErrorText,
    remoteReconnectNotice,
    remoteDetailText,
    remotePlacementFromTask,
    remoteStateDetails,
    remoteStateLabel,
    remoteStateNote,
    remoteStateSummary,
    remoteTaskActions,
} from '../modules/remote_task_state.js';

// A durable remote task row exactly as `/api/tasks` returns it: the SEALED
// placement plus the admission preflight Home writes beside it. The preflight's
// remote arm is target identity, so it IS the admission evidence — nothing else in
// a task record says "the target answered when this was admitted".
const remoteTask = {
    task_id: 'task-1',
    project_id: 'project-1',
    status: 'requested',
    metadata: {
        _sealed_workspace_ref: {
            kind: 'ssh',
            connection_id: 'connection-1',
            workspace_id: 'workspace-1',
        },
    },
};

/** The same row after admission consulted the target successfully. */
const admittedRemoteTask = {
    ...remoteTask,
    metadata: {
        ...remoteTask.metadata,
        workspace_preflight: {
            schema_version: 1,
            workspace_root: 'ssh://connection-1/srv/work/app',
            placement: 'ssh',
            connection_id: 'connection-1',
            host_id: 'host-1',
            canonical_root: '/srv/work/app',
            release_id: 'execd-1.2.3',
        },
    },
};

/** And after an admission whose target could not be consulted (disclosed, not raised). */
const refusedRemoteTask = {
    ...remoteTask,
    metadata: {
        ...remoteTask.metadata,
        workspace_preflight: {
            schema_version: 1,
            workspace_root: 'ssh://connection-1/srv/work/app',
            error: 'remote_workspace_unavailable: Remote workspace broker is not configured.',
        },
    },
};

test('durable requested SSH task derives connecting without changing local task shape', () => {
    assert.deepEqual(remotePlacementFromTask(remoteTask), {
        connectionId: 'connection-1',
        projectId: 'project-1',
        workspaceId: 'workspace-1',
    });
    const state = normalizeRemoteTaskState({}, remoteTask);
    assert.equal(state.status, 'connecting');
    assert.equal(state.taskStatus, 'requested');
    assert.deepEqual(remoteTaskActions(state), {
        canCancel: true,
        canReconnect: false,
        terminal: false,
    });
});

// The persisted WorkspaceRef is the placement authority; executor_ref is only a
// derived projection of it (RWS v2 §3.1), and its ssh discriminator is "ssh".
test('placement falls back to the derived ssh executor projection, and local tasks have none', () => {
    assert.deepEqual(
        remotePlacementFromTask({
            task_id: 'task-2',
            metadata: { executor_ref: { type: 'ssh', connection_id: 'connection-2' } },
        }),
        { connectionId: 'connection-2', projectId: '', workspaceId: '' },
    );
    assert.equal(remotePlacementFromTask({ task_id: 'task-3' }), null);
    assert.equal(
        remotePlacementFromTask({
            task_id: 'task-4',
            metadata: { executor_ref: { type: 'docker_exec', id: 'container-1' } },
        }),
        null,
    );
});

test('typed remote failure preserves bounded diagnostics and full-log references', () => {
    const state = normalizeRemoteTaskState({
        type: 'connection_state',
        connection_id: 'connection-1',
        task_id: 'task-1',
        status: 'degraded',
        phase: 'connect',
        completion: 'failed',
        error_code: 'permission_denied',
        diagnostic: {
            domain: 'filesystem',
            code: 'permission_denied',
            details: { stderr: 'permission denied' },
        },
        log_refs: [{ stream: 'stderr', blob_id: 'log-1', size: 5000 }],
    }, { ...remoteTask, status: 'failed' });
    assert.equal(
        remoteStateSummary(state),
        'Degraded · phase: connect · completion: failed · error: permission_denied',
    );
    assert.equal(remoteStateDetails(state).length, 2);
    assert.deepEqual(remoteTaskActions(state), {
        canCancel: false,
        canReconnect: true,
        terminal: true,
    });
});

test('connection-wide reconnect retains task identity and never implies task replay', () => {
    const failed = normalizeRemoteTaskState({
        connection_id: 'connection-1',
        task_id: 'task-1',
        status: 'degraded',
        completion: 'failed',
        error_code: 'ssh_timeout',
    }, { ...remoteTask, status: 'failed' });
    const ready = mergeRemoteTaskState(failed, {
        connection_id: 'connection-1',
        task_id: 'task-1',
        status: 'ready',
        completion: 'reconciled',
    }, { ...remoteTask, status: 'failed' });
    assert.equal(ready.status, 'ready');
    assert.equal(ready.errorCode, '');
    assert.equal(ready.taskStatus, 'failed');
    assert.equal(remoteTaskActions(ready).terminal, true);
    assert.equal(remoteTaskActions(ready).canReconnect, false);
    const refreshed = mergeRemoteTaskState(ready, {}, {
        ...remoteTask,
        status: 'failed',
    });
    assert.equal(refreshed.status, 'ready');
    assert.equal(refreshed.taskStatus, 'failed');
});

test('generic model/test failure and completed reload never invent SSH health', () => {
    const failedAfterAdmission = {
        ...admittedRemoteTask,
        status: 'failed',
        reason_code: 'model_error',
    };
    assert.equal(normalizeRemoteTaskState({}, failedAfterAdmission).status, 'unknown');
    // "Unknown" is not evidence, so it may not carry a Reconnect: an SSH repair is
    // not the answer to a model failure on a target that answered fine.
    const derived = normalizeRemoteTaskState({}, failedAfterAdmission);
    assert.equal(derived.stateSource, 'derived');
    assert.equal(remoteTaskActions(derived).canReconnect, false);
    assert.equal(normalizeRemoteTaskState({}, {
        ...admittedRemoteTask,
        status: 'completed',
    }).status, 'unknown');
    // A task whose ADMISSION could not reach the target is the one honest degraded.
    assert.equal(normalizeRemoteTaskState({}, {
        ...refusedRemoteTask,
        status: 'failed',
        reason_code: 'ssh_timeout',
    }).status, 'degraded');
});

// M5: this is the whole reason the derivation exists. After a reload there are no
// live frames, only durable rows — and the reducer used to read two metadata keys
// (`remote_admission`, `_remote_admission_evidence`) that NO Python producer ever
// wrote. Both evidence branches were therefore permanently false, every derived
// status collapsed to 'unknown', `stateSource` was always 'derived', and
// `canReconnect` was structurally unreachable. It now reads the preflight summary
// that admission really seals, so a reloaded page says what the target actually did.
test('a reloaded page derives remote status from the preflight admission really seals', () => {
    const running = normalizeRemoteTaskState({}, { ...admittedRemoteTask, status: 'running' });
    assert.equal(running.status, 'ready');
    assert.equal(running.stateSource, 'admission');
    assert.equal(remoteTaskActions(running).canCancel, true);

    const admitting = normalizeRemoteTaskState({}, { ...remoteTask, status: 'requested' });
    assert.equal(admitting.status, 'connecting');
    assert.equal(admitting.phase, 'admission');
    assert.equal(admitting.stateSource, 'admission');

    // The failure that IS about the connection: degraded, admission-backed, and
    // reconnectable — the button appears exactly where a reconnect is the fix.
    const unreachable = normalizeRemoteTaskState({}, { ...refusedRemoteTask, status: 'failed' });
    assert.equal(unreachable.status, 'degraded');
    assert.equal(unreachable.stateSource, 'admission');
    assert.deepEqual(remoteTaskActions(unreachable), {
        canCancel: false,
        canReconnect: true,
        terminal: true,
    });

    // A remote row with no preflight at all (a pre-RWS record) stays honest rather
    // than claiming health: nothing derived, nothing offered.
    const legacy = normalizeRemoteTaskState({}, { ...remoteTask, status: 'completed' });
    assert.equal(legacy.status, 'unknown');
    assert.equal(legacy.stateSource, 'derived');
    assert.equal(remoteTaskActions(legacy).canReconnect, false);

    // A LOCAL preflight summary is not remote admission evidence, even though it
    // sits under the same metadata key.
    const localish = normalizeRemoteTaskState({}, {
        ...remoteTask,
        status: 'running',
        metadata: {
            ...remoteTask.metadata,
            workspace_preflight: { schema_version: 1, git: { head: 'abc' } },
        },
    });
    assert.equal(localish.status, 'unknown');
    assert.equal(localish.stateSource, 'derived');
});

test('diagnostic previews are bounded before rendering', () => {
    const state = normalizeRemoteTaskState({
        status: 'degraded',
        diagnostic: { message: 'x'.repeat(9000) },
        log_refs: Array.from({ length: 100 }, (_, idx) => ({ name: `log-${idx}` })),
    }, remoteTask);
    assert.equal(state.diagnostic.message.length, 4000);
    assert.equal(state.logRefs.length, 32);
    assert.match(remoteDetailText({ value: 'x'.repeat(20000) }), /preview truncated$/);
});

test('all connection contract states have stable UI labels', () => {
    assert.deepEqual(
        ['connecting', 'ready', 'degraded', 'disconnected', 'unknown']
            .map(remoteStateLabel),
        ['Connecting', 'Ready', 'Degraded', 'Disconnected', 'Unknown'],
    );
});

// A build without the ssh transport is a SETTLED fact, not a transient outage:
// no Reconnect button may be offered, because there is nothing to reconnect to
// and pressing it would only reproduce the same typed 503.
test('a transport-less build never offers reconnect and states the reason', () => {
    const degraded = normalizeRemoteTaskState({
        connection_id: 'connection-1',
        task_id: 'task-1',
        status: 'degraded',
        error_code: REMOTE_TRANSPORT_UNAVAILABLE,
        action: 'await_remote_transport',
    }, { ...remoteTask, status: 'failed' });
    assert.equal(degraded.errorCode, REMOTE_TRANSPORT_UNAVAILABLE);
    assert.equal(remoteTaskActions(degraded).canReconnect, false);
    assert.match(remoteStateNote(degraded), /not part of this build/);
    // The same failure with an ordinary transport error DOES stay retryable.
    const timedOut = normalizeRemoteTaskState({
        connection_id: 'connection-1',
        task_id: 'task-1',
        status: 'degraded',
        error_code: 'ssh_timeout',
    }, { ...remoteTask, status: 'failed' });
    assert.equal(remoteTaskActions(timedOut).canReconnect, true);
    assert.equal(remoteStateNote(timedOut), '');
});

test('the transport refusal is recognised in every spelling it arrives in', () => {
    assert.equal(
        isRemoteTransportUnavailable({ body: { error_code: REMOTE_TRANSPORT_UNAVAILABLE } }),
        true,
    );
    assert.equal(isRemoteTransportUnavailable({ error_code: REMOTE_TRANSPORT_UNAVAILABLE }), true);
    assert.equal(isRemoteTransportUnavailable({ errorCode: REMOTE_TRANSPORT_UNAVAILABLE }), true);
    assert.equal(isRemoteTransportUnavailable({ error_code: 'ssh_timeout' }), false);
    assert.equal(isRemoteTransportUnavailable(null), false);
    assert.equal(isRemoteTransportUnavailable('remote_transport_unavailable'), false);
});

// Chat and the Activity subtab both fold live frames into a taskId → state map.
// The walk lives in ONE reducer so they cannot disagree about which tasks a
// connection-wide frame touches.
test('a task-scoped frame touches exactly that task', () => {
    const before = new Map([
        ['task-1', normalizeRemoteTaskState({}, remoteTask)],
        ['task-9', normalizeRemoteTaskState({}, {
            ...remoteTask,
            task_id: 'task-9',
            metadata: { _sealed_workspace_ref: { kind: 'ssh', connection_id: 'other' } },
        })],
    ]);
    const { states, taskIds } = reduceRemoteConnectionEvent(
        before,
        { task_id: 'task-1', connection_id: 'connection-1', status: 'ready' },
        () => remoteTask,
    );
    assert.deepEqual(taskIds, ['task-1']);
    assert.equal(states.get('task-1').status, 'ready');
    // Untouched neighbour, and the input map is never mutated.
    assert.equal(states.get('task-9').status, 'connecting');
    assert.equal(before.get('task-1').status, 'connecting');
    assert.notEqual(states, before);
});

test('a connection-wide frame fans out to that connection only', () => {
    const before = new Map([
        ['task-1', normalizeRemoteTaskState({}, remoteTask)],
        ['task-2', normalizeRemoteTaskState({}, { ...remoteTask, task_id: 'task-2' })],
        ['task-9', normalizeRemoteTaskState({}, {
            ...remoteTask,
            task_id: 'task-9',
            metadata: { _sealed_workspace_ref: { kind: 'ssh', connection_id: 'other' } },
        })],
    ]);
    const { states, taskIds } = reduceRemoteConnectionEvent(
        before,
        { connection_id: 'connection-1', status: 'disconnected', error_code: 'ssh_eof' },
        (taskId) => ({ ...remoteTask, task_id: taskId }),
    );
    assert.deepEqual(taskIds.sort(), ['task-1', 'task-2']);
    assert.equal(states.get('task-1').status, 'disconnected');
    assert.equal(states.get('task-2').status, 'disconnected');
    assert.equal(states.get('task-9').status, 'connecting');
});

test('a frame naming neither a task nor a connection changes nothing', () => {
    const before = new Map([['task-1', normalizeRemoteTaskState({}, remoteTask)]]);
    const { states, taskIds } = reduceRemoteConnectionEvent(before, { status: 'ready' });
    assert.deepEqual(taskIds, []);
    assert.equal(states.get('task-1').status, 'connecting');
    // A non-Map input is tolerated as empty rather than throwing mid-frame.
    assert.deepEqual(reduceRemoteConnectionEvent(null, { task_id: 'task-1' }).taskIds, ['task-1']);
});

test('a reconnect never claims a finished task was replayed', () => {
    const live = normalizeRemoteTaskState({ status: 'ready' }, { ...remoteTask, status: 'running' });
    assert.equal(remoteReconnectNotice(live), 'Connection reconnected and reconciled.');
    const finished = normalizeRemoteTaskState({ status: 'ready' }, { ...remoteTask, status: 'failed' });
    assert.match(remoteReconnectNotice(finished), /was not replayed/);
});

test('one rendering of a failed owner action, including the typed next step', () => {
    assert.equal(
        remoteActionErrorText({
            body: { error: 'remote host identity differs', action: 'retrust' },
        }),
        'remote host identity differs · Next: retrust',
    );
    assert.equal(
        remoteActionErrorText({ message: 'HTTP 500' }),
        'HTTP 500',
    );
    // The transport refusal wins over the generic rendering: its own copy already
    // explains the situation, and "Next: await_remote_transport" would not.
    assert.match(
        remoteActionErrorText({
            body: {
                error: 'remote ssh transport is not available in this build',
                error_code: REMOTE_TRANSPORT_UNAVAILABLE,
                action: 'await_remote_transport',
            },
        }),
        /^The remote SSH transport is not part of this build/,
    );
});

// The reducer bounds the same two things the gateway does, and must leave the same
// trace: a shortened diagnostic that looks complete is the one thing a forensic
// payload must never be. Mirrors `gateway/connections.py::_omission_note`.
test('bounded collections leave a visible trace of what they dropped', () => {
    const state = normalizeRemoteTaskState({
        status: 'degraded',
        diagnostic: {
            details: {
                env: Object.fromEntries(
                    Array.from({ length: 40 }, (_, index) => [`KEY_${index}`, index]),
                ),
                lines: Array.from({ length: 40 }, (_, index) => `line ${index}`),
            },
        },
        log_refs: Array.from({ length: 40 }, (_, index) => ({ name: `log-${index}` })),
        log_refs_count: 40,
    }, remoteTask);

    assert.match(state.diagnostic.details.env['…'], /of 40 entries omitted/);
    assert.equal(state.diagnostic.details.lines.length, 33);
    assert.match(state.diagnostic.details.lines.at(-1), /of 40 entries omitted/);
    assert.equal(state.logRefs.length, 32);
    assert.equal(state.logRefsCount, 40);
    // The details list ends with the omission, so the owner does not read 32 refs as
    // "the logs" when there are 40.
    const details = remoteStateDetails(state);
    assert.equal(details.at(-1).label, 'Omitted log references');
    assert.deepEqual(details.at(-1).value, { omitted: 8, total: 40 });

    // A complete list gets no extra row, and the server's total cannot make what
    // arrived look short.
    const intact = normalizeRemoteTaskState({
        status: 'degraded',
        log_refs: [{ stream: 'stderr' }],
    }, remoteTask);
    assert.equal(intact.logRefsCount, 1);
    assert.deepEqual(
        remoteStateDetails(intact).map((part) => part.label),
        ['stderr'],
    );

    // A merge keeps the list and its total together: a retained list beside a fresh
    // count would disclose an omission that does not match what is shown.
    const merged = mergeRemoteTaskState(state, { status: 'ready' }, remoteTask);
    assert.equal(merged.logRefs.length, 32);
    assert.equal(merged.logRefsCount, 40);
});
