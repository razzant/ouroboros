// Preview adapter inserted only in the variant-B page's main_waiting.js response.
let ownTargets = [];
function resumeTargetsInMain(data) {
    const out = [];
    for (const work of data?.works || []) {
        if (Number(work.room) !== 1 || work.project_id || !work.confirmed) continue;
        const resumes = work.controls.filter(c => c.action === 'resume');
        for (const turn of work.turn) {
            if (turn.kind !== 'action' || turn.wait_id) continue;
            const matches = resumes.filter(c => c.target === turn.task_id);
            if (matches.length !== 1) continue;
            const control = matches[0];
            if (control.scope === 'single') {
                if (control.target === work.episode.task_id) continue;
            } else if (control.scope || control.target !== work.episode.task_id) continue;
            out.push({kind: 'own', id: control.target, at: 1000 * work.since, control});
        }
    }
    return out;
}
function ownResume(target) {
    if (target.control.scope === 'single') return null;
    const card = messagesDiv.querySelector('.chat-live-card[data-task-id="' + CSS.escape(target.id) + '"]');
    const controls = Array.from(card?.querySelectorAll('[data-resume-run]') || [])
        .filter(button => button.closest('.chat-live-card') === card && !button.disabled);
    return controls.length === 1 ? controls[0] : null;
}
async function revealOwn(target) {
    if (target.control.scope === 'single') {
        window.dispatchEvent(new CustomEvent('ouro:open-dashboard', {detail: {tab: 'activity'}}));
        const selector = '[data-activity-section="queue"] [data-act="task-control"][data-resume="1"][data-id="'
            + CSS.escape(target.id) + '"]:not([data-root="1"])';
        const limit = performance.now() + 8000;
        while (performance.now() < limit) {
            const nodes = document.querySelectorAll(selector);
            if (nodes.length === 1) {
                const control = nodes[0];
                for (let group = control.closest('details'); group; group = group.parentElement?.closest('details')) {
                    group.open = true;
                }
                if (control.getClientRects().length) {
                    control.scrollIntoView({block: 'center'});
                    control.focus({preventScroll: true});
                    return;
                }
            }
            await new Promise(done => requestAnimationFrame(done));
        }
        showToast('Held helper unavailable in Activity.', 'error');
        return;
    }
    let control = ownResume(target);
    if (!control) {
        const found = await revealOwnTask(target.id);
        if (!found) return;
        control = found.target && ownResume(target);
        if (!control) {
            showToast(found.state === 'absent' ? 'No message for this task in Main.' : 'Task control unavailable.', 'info');
            return;
        }
    }
    const card = control.closest('.chat-live-card');
    land(card);
    if (!onScreen(control)) land(control);
    control.focus({preventScroll: true});
}
