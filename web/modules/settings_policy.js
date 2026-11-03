// The Settings policy-state lines: what Access, supervisor and review enforcement are in
// the current process and will be for the next task, and what Cyber Pro Access does to a
// Blocking choice. Facts come from the gateway's `policy_state`; the draft selects are read
// only to say what a selection would mean before Save.

export function policyValueLabel(value) {
    const labels = {
        light: 'Light', advanced: 'Advanced', pro: 'Pro', cyber_pro: 'Cyber Pro',
        full: 'Full', off: 'Off', advisory: 'Advisory', blocking: 'Blocking',
    };
    return labels[String(value || '').trim().toLowerCase()] || String(value || 'Unknown');
}

// Review findings never block actions under Cyber Pro Access (review_helpers.review_enforcement_blocks).
// Blocking in force now is a fact of the current process; a saved or drafted Blocking is
// conditional on the selected Access. An unknown current Access states nothing about now.
export function cyberReviewNote(state, { enforcement = '', access = '' } = {}) {
    const nowAccess = state?.access?.current_process || state?.access?.effective;
    const nextAccess = access || state?.access?.configured;
    const nowBlocking = (state?.review?.current_process || state?.review?.effective) === 'blocking';
    const nextBlocking = (enforcement || state?.review?.next_task || state?.review?.configured) === 'blocking';
    const nowCyber = nowAccess === 'cyber_pro';
    if (nowCyber && nowBlocking && nextBlocking && nextAccess === 'cyber_pro') {
        return 'Cyber Pro: Blocking is not enforced; review findings do not block actions';
    }
    const parts = nowCyber && nowBlocking ? ['Cyber Pro: Blocking is not enforced now'] : [];
    if (nextBlocking && nextAccess === 'cyber_pro') {
        parts.push('selected Blocking would not be enforced under the selected Cyber Pro');
    } else if (nextBlocking && nowCyber && ['light', 'advanced', 'pro'].includes(nextAccess)) {
        parts.push(`selected Blocking would be enforced only after a restart under the selected ${policyValueLabel(nextAccess)}`);
    }
    const text = parts.join('; ');
    return text.charAt(0).toUpperCase() + text.slice(1);
}

export function syncPolicyState(root, meta) {
    const state = meta?.policy_state;
    if (!state) return;
    const render = (key, text) => {
        const node = root?.querySelector(`[data-policy-state="${key}"]`);
        if (node) node.textContent = text;
    };
    const access = state.access || {};
    render('access', access.restart_required
        ? `Saved: ${policyValueLabel(access.configured)} · Current process: ${policyValueLabel(access.current_process || access.effective)} · After restart: ${policyValueLabel(access.configured)} · Restart required`
        : `Current process: ${policyValueLabel(access.current_process || access.effective)} · After restart: ${policyValueLabel(access.configured)}`);
    const suffix = (item) => item.active_task_snapshot
        ? `Saved: ${policyValueLabel(item.configured)} · Current process: ${policyValueLabel(item.current_process || item.effective)} · Next task: ${policyValueLabel(item.next_task || item.configured)} · Current task keeps its start snapshot`
        : `Current process: ${policyValueLabel(item.current_process || item.effective)} · Next task: ${policyValueLabel(item.next_task || item.configured)}`;
    render('supervisor', suffix(state.supervisor || {}));
    // The consciousness level's line, only where the server reports it (an older one has one level).
    if (state.supervisor_consciousness) render('supervisor_consciousness', suffix(state.supervisor_consciousness));
    const cyber = cyberReviewNote(state, { enforcement: root?.querySelector('#s-review-enforcement')?.value,
        access: root?.querySelector('#s-runtime-mode')?.value });
    render('review', suffix(state.review || {}) + (cyber ? ` · ${cyber}` : ''));
}
