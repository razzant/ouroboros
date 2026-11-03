// The host's start as the Main and Project headers read it from /api/state: Starting… until the
// body states readiness, and the boot phases that say which part of the start runs or failed.

// В9: one /api/state body's `supervisor_ready`, null when it states nothing. Only
// true ends Starting…; a `supervisor_error` is not readiness and is not read here.
export function supervisorReady(data) {
    return typeof data?.supervisor_ready === 'boolean' ? data.supervisor_ready : null;
}

// The host's boot phases (the `boot` object of /api/state, the same one /api/health carries), null
// when the body states none: `phase`, `dialogue_ready`, `tasks_ready`, `steps`, `error`.
export function bootPhase(data) {
    const boot = data?.boot;
    return boot && typeof boot === 'object' && typeof boot.phase === 'string' ? boot : null;
}

// The Starting… header's words for a boot phase: the chat accepts messages while the
// background tasks are still being restored, or the start failed (its cause rides the
// title). Nothing known about the phase keeps the plain Starting….
export function startingStatus(boot) {
    const failedStep = Array.isArray(boot?.steps) ? boot.steps.find((step) => step?.state === 'failed') : null;
    const title = String(boot?.error || (failedStep ? `${failedStep.name}: ${failedStep.error || ''}` : '') || '');
    if (boot?.phase === 'failed') return { kind: 'starting', text: 'Start failed', title };
    if (boot?.dialogue_ready === true) {
        return { kind: 'starting', text: 'Chat ready, restoring background tasks', title };
    }
    return { kind: 'starting', text: 'Starting…', title };
}

// One chat instance's view of the start: the readiness and phases the last snapshot that stated
// them said, both forgotten on a disconnect (the reconnect's snapshot proves them again).
// `status()` is the header reducer's input; `ready()` answers the room's readiness question.
export function createHostStart() {
    let ready = false;
    let boot = null;
    return {
        read(data) {
            ready = supervisorReady(data) ?? ready;
            boot = bootPhase(data) ?? boot;
        },
        forget() {
            ready = false;
            boot = null;
        },
        ready: () => ready,
        status: () => ({ supervisorStarting: !ready, bootPhase: boot }),
    };
}
