// Preflight observations only: the server's pin rule owns the restart decision.
// `candidate`: for Restart to update, the id of the prepared update shown; its effects must name it.
export function updateEffectsText(preflight, { replace = false, candidate = null } = {}) {
    const restart = candidate !== null;
    const bound = restart ? Boolean(candidate) && preflight?.effects?.candidate_id === candidate
        : preflight?.effects?.target_sha === preflight?.merge_plan?.target_sha;
    const effects = bound ? preflight?.effects || {} : {};
    const engine = (replace ? effects.replacement_engine : effects.engine) || {};
    const version = (pin) => pin?.version || 'unknown';
    const parts = [`Claudexor: ${version(engine.serving)} → ${version(engine.target)}.`];
    if (engine.action === 'replace') {
        parts.push('Restart will replace the shared engine and end all its active runs, including other work.');
    } else if (engine.action === 'keep') {
        parts.push('This pin keeps the serving engine running.');
    } else {
        parts.push('The engine transition could not be confirmed; active runs may end.');
    }
    const active = effects.active_work;
    parts.push(active?.status === 'known'
        ? `Currently active: ${active.model} model call(s), ${active.non_model} other job(s).`
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
    // The in-place lanes never change the interpreter; Restart to update says what its preparation selected.
    const changes = restart ? runtime.changed_by_apply : false;
    if (changes === true) {
        parts.push(`Restart to update changes them: the new start runs Python ${runtime.selected?.python || 'unknown'}, SQLite ${runtime.selected?.sqlite || 'unknown'}.`);
    } else if (changes === false) {
        parts.push(`${restart ? 'Restart to update' : 'In-app Update'} keeps the interpreter and SQLite. Install a newer runtime separately to change them.`);
    } else {
        parts.push('Whether Restart to update changes the interpreter and SQLite is unknown.');
    }
    return parts.join('\n\n');
}
