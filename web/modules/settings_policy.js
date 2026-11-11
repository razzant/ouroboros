// The Settings policy-state lines: what Access, supervisor and review enforcement are in
// the current process and will be for the next task, and what Cyber Pro Access does to a
// Blocking choice. Facts come from the gateway's `policy_state`; the draft selects are read
// only to say what a selection would mean before Save. A managed deployment's part
// (`policy_state.deployment`, also the wizard's `bootstrap.deployment`) is read here once:
// a control offers only what its policy file allows, and what it fixes is said, not edited.

/** What a managed deployment's policy file fixes, or null outside one. */
export const deploymentOf = (meta) => meta?.policy_state?.deployment || null;

const RANKS = { access: ['light', 'advanced', 'pro', 'cyber_pro'], coverage: ['off', 'light', 'full'] };

/** The options of Access (a ceiling) or a Supervisor level (a floor) the deployment allows, in their order. */
export function allowedOptions(options, deployment, axis) {
    const ranks = RANKS[axis];
    const bound = ranks.indexOf(axis === 'access' ? deployment?.access_max : deployment?.min_coverage);
    if (bound < 0) return options;
    return options.filter(({ value }) => (axis === 'access' ? ranks.indexOf(value) <= bound : ranks.indexOf(value) >= bound));
}

/** The one sentence a fixed or bounded choice carries; '' where the deployment leaves it free. */
export function deploymentNote(deployment, axis) {
    const bound = axis === 'access' ? deployment?.access_max : deployment?.min_coverage;
    if (!bound) return '';
    const level = policyValueLabel(bound);
    if (allowedOptions(RANKS[axis].map((value) => ({ value })), deployment, axis).length === 1) {
        return `Set by your organization: ${level}`;
    }
    return axis === 'access' ? `Your organization allows up to ${level}` : `Your organization requires at least ${level}`;
}

/**
 * Offer a segmented choice (`target`, its hidden input) with only the options the deployment allows: drawn again by
 * the one renderer (`render`) and bound by `bind` on its parent, which holds its input and no other choice. A choice
 * left with one option is no choice: its row is hidden and its policy line says the fixed value. Unmanaged, the full
 * choice stays as drawn.
 */
export function offerAllowedChoices(root, { target, options, axis, modifier = '' }, deployment, { render, bind }) {
    const group = root?.querySelector(`[data-effort-target="${target}"]`);
    if (!group) return;
    const allowed = allowedOptions(options, deployment, axis);
    const drawn = Array.from(group.querySelectorAll('[data-effort-value]'), (button) => button.dataset.effortValue);
    let current = group;
    if (drawn.join(' ') !== allowed.map(({ value }) => value).join(' ')) {
        const holder = group.parentElement;
        group.outerHTML = render({ target, options: allowed, modifier });
        current = holder.querySelector(`[data-effort-target="${target}"]`);
        bind(holder);
    }
    current.hidden = allowed.length === 1;
    return allowed.length === 1;
}

/** What the Safety exits a deployment closes (`allowed_exits` names the open ones) mean for a message; '' when it
 * closes none or outside one. A refusal or an unchecked hold an exit could still lift is no fixed fact to state. */
export function safetyExitsNote(deployment) {
    const open = deployment?.allowed_exits;
    if (!Array.isArray(open)) return '';
    const facts = {
        repeat_after_dangerous: 'Ouroboros cannot override a DANGEROUS refusal on its own',
        release_after_dangerous: 'your approval cannot override a DANGEROUS refusal',
        release_unjudged: 'your approval cannot send a message the Supervisor has not checked',
    };
    const fact = Object.entries(facts).filter(([name]) => !open.includes(name)).map(([, text]) => text).join('; ');
    return fact ? `Set by your organization: ${fact}.` : '';
}

/** Whether the organization's Supervisor answers through this connection (`kind` or `kind@name`) or the local server. */
export const checkerUses = (deployment, connectionId) => (connectionId === 'local'
    ? Boolean(deployment?.checker?.local) : (deployment?.checker?.connections || []).includes(connectionId));

/** Whether the checker's built-in routes read this flat setting now (`checker.fields`: a built-in key's field, the
 * compatible lane's dedicated or legacy pair): that one field is shown, never edited; its row stays the employee's. */
export const checkerReads = (deployment, settingKey) => (deployment?.checker?.fields || []).includes(settingKey);

/** The sentence beside fields the checker reads, by their labels; '' for none. */
export const checkerFieldsNote = (labels) => (labels.length
    ? `${labels.join(' and ')}: set by your organization, its Safety Supervisor uses ${labels.length > 1 ? 'them' : 'it'}.` : '');

/** Show `input` (one flat setting outside an API-key row) as the checker reads it: read-only with `note`, or as it was. */
export function fixCheckerField(input, note, deployment, settingKey, label) {
    if (!input) return;
    const fixed = checkerReads(deployment, settingKey);
    input.readOnly = fixed;
    if (note) Object.assign(note, { hidden: !fixed, textContent: fixed ? checkerFieldsNote([label]) : '' });
}

/** Which required extension a custom setting is fixed for (``deployment.extensions``), else ''. */
export const organizationSettingOwner = (deployment, key) => ((deployment?.extensions?.required || [])
    .find((row) => (row?.settings || []).includes(key))?.name || '');

/** A secret row whose value the organization fixes: read-only, no Show, Clear or Remove, and one sentence saying
 * whose value it is. Its masked value stays in the field, so a save carries it back unchanged. */
export function fixOrganizationSecret(row, deployment, key) {
    const owner = organizationSettingOwner(deployment, key);
    if (!row || !owner) return;
    row.querySelectorAll('input').forEach((input) => { input.readOnly = true; });
    row.querySelectorAll('[data-row-secret-toggle], [data-row-secret-clear], [data-custom-secret-remove]')
        .forEach((button) => { button.hidden = true; });
    const note = row.ownerDocument.createElement('div');
    note.className = 'settings-inline-note';
    note.dataset.organizationSetting = key;
    note.textContent = `Set by your organization for ${owner}; its value is not shown.`;
    (row.querySelector('.secret-input-row')?.parentElement || row).appendChild(note);
}

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
    // One value said once ("Advisory now and for new tasks"); the facts are spelled out only where they differ.
    // A value the deployment fixes is said as its; a bound it sets follows the facts.
    const deployment = state.deployment || null;
    const bounded = (axis, text) => {
        const note = deploymentNote(deployment, axis);
        return note.startsWith('Set by') ? note : [text, note].filter(Boolean).join(' · ');
    };
    const access = state.access || {};
    const now = policyValueLabel(access.current_process || access.effective);
    const later = policyValueLabel(access.configured);
    render('access', bounded('access', access.restart_required
        ? `Saved: ${later} · Current process: ${now} · After restart: ${later} · Restart required`
        : now === later ? `${now} now and after restart` : `Current process: ${now} · After restart: ${later}`));
    const suffix = (item) => {
        const saved = policyValueLabel(item.configured);
        const current = policyValueLabel(item.current_process || item.effective);
        const next = policyValueLabel(item.next_task || item.configured);
        const running = item.active_task_snapshot ? '; the running task keeps the value it started with' : '';
        if (current === next && (!item.active_task_snapshot || saved === current)) return `${current} now and for new tasks${running}`;
        return item.active_task_snapshot
            ? `Saved: ${saved} · Current process: ${current} · Next task: ${next} · Current task keeps its start snapshot`
            : `Current process: ${current} · Next task: ${next}`;
    };
    render('supervisor', bounded('coverage', suffix(state.supervisor || {})));
    // The consciousness level's line, only where the server reports it (an older one has one level); a floor
    // the deployment sets binds it too.
    if (state.supervisor_consciousness || deployment?.min_coverage) {
        render('supervisor_consciousness', bounded('coverage', state.supervisor_consciousness ? suffix(state.supervisor_consciousness) : ''));
    }
    const cyber = cyberReviewNote(state, { enforcement: root?.querySelector('#s-review-enforcement')?.value,
        access: root?.querySelector('#s-runtime-mode')?.value });
    render('review', suffix(state.review || {}) + (cyber ? ` · ${cyber}` : ''));
}
