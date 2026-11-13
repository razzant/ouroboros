// What the owner's policy says in Settings rows: the facts of the gateway's `policy_state` (Access, the Safety levels,
// review enforcement) and a managed deployment's part of it (`policy_state.deployment`, also the wizard's
// `bootstrap.deployment`), read here once. A control offers only what the policy file allows and its row says whose
// limit that is; what it fixes is said, never edited and never hidden.
import { rowById } from './settings_rows.js';

// The Safety scale's shown words, from the row that draws them. The setup wizard imports this module, so it reads the
// words from the data module and never from consciousness_words.js, which brings the chat modules with it.
const SAFETY_LEVELS = rowById('OUROBOROS_SAFETY_MODE').control.options;

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
    const level = policyValueLabel(bound, axis);
    if (allowedOptions(RANKS[axis].map((value) => ({ value })), deployment, axis).length === 1) {
        return `Set by your organization: ${level}`;
    }
    return axis === 'access' ? `Your organization allows up to ${level}` : `Your organization requires at least ${level}`;
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

/** Which required extension a custom setting is fixed for (``deployment.extensions``), else ''. */
export const organizationSettingOwner = (deployment, key) => ((deployment?.extensions?.required || [])
    .find((row) => (row?.settings || []).includes(key))?.name || '');

/** A stored policy value as the owner reads it. `light` is a word of two scales: the Safety scale's is asked for by
 *  its axis (`coverage`), else it is Access's. */
export function policyValueLabel(value, axis = '') {
    const key = String(value || '').trim().toLowerCase();
    const level = SAFETY_LEVELS.find((item) => item.value === key);
    if (level && (axis === 'coverage' || key !== 'light')) return level.label;
    const labels = { light: 'Light', advanced: 'Advanced', pro: 'Pro', cyber_pro: 'Cyber Pro' };
    return labels[key] || String(value || 'Unknown');
}

const AXES = { OUROBOROS_RUNTIME_MODE: 'access', OUROBOROS_SAFETY_MODE: 'coverage', OUROBOROS_CONSCIOUSNESS_SAFETY_MODE: 'coverage' };

/**
 * The rows of Permissions and checks as this installation's policy leaves them (`rows`: copies of the descriptors,
 * changed in place). A managed deployment's ceiling on Access and floor under a Safety level leave only the options
 * it allows, and the row says whose limit it is.
 */
export function applyPolicyToRows(rows, settings, described = (id) => null) {
    const state = settings?._meta?.policy_state || null;
    const deployment = state?.deployment || null;
    for (const row of rows) {
        const base = described(row.id) || row;
        const axis = AXES[row.id];
        if (axis) {
            const note = deploymentNote(deployment, axis);
            const options = allowedOptions(base.control.options, deployment, axis);
            row.meta = note || base.meta;
            row.control = { ...base.control, options: options.map((option) => (
                note && axis === 'coverage' ? { ...option, note: `${option.note} ${note}.` } : { ...option })) };
        }
    }
    return rows;
}
