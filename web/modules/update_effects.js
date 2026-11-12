// Preflight observations and the restart's fixed effect on tasks: the server's pin rule owns the restart decision.

// What an acknowledged Update or Replace restart does to this installation's own tasks: one rule, the same on
// every surface that shows the dialog (ARCHITECTURE §9 "Shutdown doors", §10 invariant 31;
// supervisor/restart_retention.py). Static on purpose: the returns are named at the stop and decided at boot,
// so a count taken here would be a prediction.
const TASK_EFFECTS = [
    'Running tasks stop and continue by themselves after the restart, from their last saved step; waiting and '
    + 'sleeping tasks and the runnable queue return too, and schedules are kept, so there is no need to pause '
    + 'anything first.',
    'Tasks already paused or held stay so until they are resumed. Not continued: a place conversation cut '
    + 'mid-turn, a task with nothing saved to continue from, and services that tasks started. A task the restart holds '
    + 'instead of continuing waits for Resume.',
];

export function updateEffectsText(preflight, { replace = false } = {}) {
    const effects = preflight?.effects?.target_sha === preflight?.merge_plan?.target_sha
        ? preflight?.effects || {} : {};
    const engine = (replace ? effects.replacement_engine : effects.engine) || {};
    const version = (pin) => pin?.version || 'unknown';
    const parts = [...TASK_EFFECTS, `Claudexor: ${version(engine.serving)} → ${version(engine.target)}.`];
    if (engine.action === 'replace') {
        parts.push('Restart will replace the shared engine and end all its active runs, including other work.');
    } else if (engine.action === 'keep') {
        parts.push('This pin keeps the serving engine running.');
    } else {
        parts.push('The engine transition could not be confirmed; active runs may end.');
    }
    const active = effects.active_work;
    parts.push(active?.status === 'known'
        ? `Currently active engine work: ${active.model} model call(s), ${active.non_model} other job(s).`
        : 'Currently active engine work: unknown.');
    if (engine.direction === 'backward') {
        parts.push(`Engine data is forward-only. After this engine stops, cold start needs a body pinning ${version(engine.serving)} or newer.`);
    } else {
        parts.push('Engine data is forward-only; restoring code does not downgrade it.');
    }
    if (effects.provisional && !replace) parts.push('The final pin may change during reviewed resolution.');
    const runtime = effects.runtime || {};
    parts.push(`This install: Python ${runtime.installed?.python || 'unknown'}, SQLite ${runtime.installed?.sqlite || 'unknown'}.`);
    parts.push(`Bundle described by this update: Python ${runtime.bundle?.python || 'unknown'}, SQLite ${runtime.bundle?.sqlite || 'unknown'}.`);
    parts.push('In-app Update keeps the interpreter and SQLite. Install a newer runtime separately to change them.');
    return parts.join('\n\n');
}
