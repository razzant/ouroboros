import { APP_SETTINGS_ORIGIN, apiClient, fetchJson } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { saveCheckerSettings, saveSafetyLevel, saveSafetyRules } from './consciousness_api.js';
import { isDoorId } from './settings_pages.js';
import { rowById } from './settings_rows.js';
import { errorWords, escapeHtmlAttr, setResult } from './ui_primitives.js';

/** Read the complete custom-key draft, including blank and removed rows. No paint. */
export function readCustomSecretDraft(root) {
    return Array.from(root.querySelectorAll('[data-custom-secret-row]'), (row) => {
        const input = row.querySelector('[data-custom-secret-value]');
        return {
            key: row.querySelector('[data-custom-secret-key]')?.value || '',
            value: input?.value || '',
            appliedValue: input?.dataset.appliedValue || '',
            originalKey: row.dataset.originalKey || '',
            clear: input?.dataset.forceClear === '1',
            removed: row.dataset.removeCustomSecret === '1',
        };
    });
}

/** Compile the existing custom-key wire shape and return every local error. */
export function collectCustomSecretDraft(rows, knownKeys = []) {
    const values = {};
    const errors = [];
    const known = new Set(knownKeys);
    const seen = new Map();
    rows.forEach((row, index) => {
        const key = row.key.trim().toUpperCase();
        const fail = (message, field = 'key') => errors.push({ index, field, message });
        if (row.removed) {
            if (row.originalKey) values[row.originalKey] = '';
            return;
        }
        if (!key) fail('Enter a key name, or remove this row.');
        else if (!/^[A-Z][A-Z0-9_]{2,}$/.test(key)) fail('Use at least three letters, numbers or underscores, starting with a letter.');
        else if (key.startsWith('OUROBOROS_') || known.has(key)) fail('This name is a built-in setting. Use its existing field.');
        else if (seen.has(key)) {
            fail(`Duplicate key ${key}. Keep one row for this key.`);
            errors.push({ index: seen.get(key), field: 'key', message: `Duplicate key ${key}. Keep one row for this key.` });
        } else if (row.originalKey && key !== row.originalKey && row.value === row.appliedValue && !row.clear) {
            fail('Enter a value for the new key, or keep the saved name.', 'value');
        } else if (row.clear) values[key] = '';
        else if (row.value && row.value !== row.appliedValue) values[key] = row.value;
        else if (!row.originalKey) fail('Enter a value for this key, or remove this row.', 'value');
        if (key) seen.set(key, index);
    });
    return { values, errors };
}

/** A Settings action in flight: disabled and announced busy, the same for Save and a key's Test. */
export function setButtonBusy(button, busy) {
    if (!button) return;
    button.disabled = busy;
    if (busy) button.setAttribute('aria-busy', 'true');
    else button.removeAttribute('aria-busy');
}

/** Saved/unsaved/unknown are server receipts, never inferred from HTTP success. */
export function settingsWriteFailure(error, label = 'Settings') {
    const receipt = error?.body || error?.payload;
    const detail = error?.message || String(error);
    if (receipt?.saved === true) return { unknown: false, text: `${label} was saved, but a later step failed: ${detail}` };
    if (receipt?.saved === false) return { unknown: false, text: `${label} was not changed: ${detail}` };
    return { unknown: true, text: `${label} outcome is unknown: ${detail}` };
}

const SEARCH_GAPS = { no_credentials: 'no API key', package_missing: 'package not installed', legacy_openai_base_url: 'OPENAI_BASE_URL is set' };

/** The web search preview in owner words: the sources in the order they are tried (each named by `name`, the
 *  Source select's own word for its id) and a saved model a source ignores stay in the main line; the models, the
 *  sources not available now and the credential note are Details. */
export function webSearchPreviewWords(data, name = (id) => id) {
    if (data?.error) return { main: String(data.error), details: '' };
    const legs = data?.legs || [];
    const ignored = data?.ignored_legacy_model ? ` Anthropic does not apply the saved model ${data.ignored_legacy_model}; edit Source or Model to change it.`
        : data?.unapplied_model ? ` No built-in search serves the saved model ${data.unapplied_model}; Auto uses the recommended models.` : '';
    // provider_default names the source's recommended model (search_routes.SEARCH_MODELS), not the owner's.
    const models = legs.filter((leg) => leg.model)
        .map((leg) => `${name(leg.source)} · ${leg.model}${leg.model_source === 'provider_default' ? ' (recommended)' : ''}`);
    const unavailable = (data?.unavailable || [])
        .map((leg) => `${name(leg.source)} (${SEARCH_GAPS[leg.unavailable_reason] || 'not ready'})`);
    return {
        main: `${legs.length ? `Uses ${legs.map((leg) => name(leg.source)).join(', then ')}.` : 'No search source is ready.'}${ignored}`,
        details: [models.length ? `${models.join('; ')}.` : '', unavailable.length ? `Not available now: ${unavailable.join(', ')}.` : '',
            'Uses saved account credentials.'].filter(Boolean).join(' '),
    };
}

/* The one Settings section recipe (DESIGN.md principle 7; the 06.11 convergence): a title, one authored lead in
   owner words, right under it the one Details disclosure that holds the section's former copy unchanged (so it opens
   next to the line it explains, not screens below in a long section), then the controls. Quiet actions dock in the title row. Every page's sections come from here, so they
   read as one scheme. `titleId` lets a lone control name itself by the title; `card: false` is a section
   that already sits inside a card (setup's step), which a second card would outline (DESIGN principle 2). */
export function settingsSection({ title, lead = '', details = '', body = '', actions = '', id = '', titleId = '', className = '', attrs = '', card = true }) {
    const h3 = `<h3${titleId ? ` id="${titleId}"` : ''}>${title}</h3>`;
    const head = actions ? `<div class="settings-section-head">${h3}<div class="settings-toolbar">${actions}</div></div>` : h3;
    return `
        <div class="form-section${card ? ' ui-card' : ''}${className ? ` ${className}` : ''}"${id ? ` id="${id}"` : ''}${attrs ? ` ${attrs}` : ''}>
            ${head}
            ${lead ? `<p class="settings-section-lead">${lead}</p>` : ''}
            ${details ? `<details class="ui-disclosure settings-section-details"><summary>Details</summary><div class="settings-section-copy">${details}</div></details>` : ''}
            ${body}
        </div>`;
}

/* ---- Rows that save themselves (settings_rows.js describes a row; this draws it and is its one writer) ------------
   A row's write in flight, the value waiting behind it and its last outcome live here, keyed by row id, never in the
   page's DOM: a page torn down before the answer and bound again shows what happened and sends nothing twice. */

let lastSettings = null;         // the last GET /api/settings: values, `_meta.effects`, `_meta.tokens`, the setup contract
let settingsEpoch = 0;           // a read that started before a newer read or write never puts an old document back
const ROW_STATES = new Map();    // row id -> { row, busy, job, waiting, outcome, seen, typed, reread, sent }
const BOUND = new Map();         // row id -> the painters of that row on screen
const ENTRY_QUEUES = new Map();  // list key -> the tail of its entry writes
const HOST_STATES = new Map();   // host codec -> its last snapshot (`state`: on, off, or why it is neither)

function rowState(row) {
    if (!ROW_STATES.has(row.id)) {
        ROW_STATES.set(row.id, { row, busy: false, job: null, waiting: null, outcome: null, seen: false, typed: null, reread: false, sent: undefined });
    }
    return ROW_STATES.get(row.id);
}
const repaintRow = (id) => BOUND.get(id)?.forEach((paint) => paint());

// The composer's list and the circle and place Model fields learn the tiers and keys from the newest document
// (model_catalog_store.js): a read, or a row's landed write.
const announce = (data) => globalThis.document?.dispatchEvent?.(new CustomEvent('settings-document:applied', { detail: data }));

/** One fresh read of the settings document; the helper keeps the newest for values, effects, tokens and the setup
 *  contract. A row whose write was not confirmed says it saved once a read shows the value it sent. */
export async function readSettings() {
    const epoch = ++settingsEpoch;
    const data = await fetchJson('/api/settings', { cache: 'no-store' });
    if (epoch !== settingsEpoch && lastSettings) return data;
    lastSettings = data;
    announce(data);
    ROW_STATES.forEach((state) => {
        if (state.busy || state.outcome?.state !== 'unknown' || !sameValue(currentValue(state.row, data), state.sent)) return;
        Object.assign(state, { outcome: { ...state.outcome, state: 'saved', message: '', details: '' }, reread: false, typed: null, seen: false });
        repaintRow(state.row.id);
    });
    return data;
}
let firstRead = null;  // writes that come before any read wait on one read
const ensureRead = () => (firstRead ||= readSettings().catch(() => null).finally(() => { firstRead = null; }));

/** Forget every row's standing and the last read (a signed-out page, a test). */
export function forgetSettingRows() {
    lastSettings = null;
    settingsEpoch += 1;
    firstRead = null;
    [ROW_STATES, BOUND, ENTRY_QUEUES, HOST_STATES].forEach((map) => map.clear());
}

/** What the setup contract already says about a row's control (`control.from: 'setup:<name>'`): an option list such
 *  as `runtimeModes`, or the bounds of a field such as `totalBudget`. The row's own words win. */
function contractOf(from) {
    const name = String(from || '').replace(/^setup:/, '');
    const contract = lastSettings?._meta?.setup_contract || {};
    if (!name) return {};
    if (Array.isArray(contract[name])) {
        return { options: contract[name].map((option) => ({ value: option.value, label: option.buttonLabel || option.label })) };
    }
    const field = Object.values(contract).flat().find((item) => item?.stateKey === name);
    return field ? { min: field.min, max: field.max, step: field.step } : {};
}
const controlOf = (row) => ({ ...contractOf(row.control?.from), ...row.control });
const isTyped = (row) => ['number', 'text', 'secret', 'pair'].includes(row.control?.kind);
const truthy = (value) => value === true || /^(true|1|yes|on)$/i.test(String(value ?? '').trim());
const isUnlimited = (value) => String(value ?? '').trim().toLowerCase() === 'unlimited';

function currentValue(row, settings = lastSettings) {
    const codec = CODECS[row.codec];
    if (codec?.read) return codec.read(settings, row);
    if (!settings) return undefined;
    return row.keys.length > 1 ? row.keys.map((key) => settings[key]) : settings[row.keys[0]];
}

/** The keys a row's value writes. `only` (Undo) keeps those of them: a key the receipt did not name is left to
 *  whoever changed it since. */
function bodyOf(row, value, only = null) {
    const codec = CODECS[row.codec];
    const body = codec?.body ? codec.body(value, row)
        : row.keys.length > 1 ? Object.fromEntries(row.keys.map((key, i) => [key, value?.[i]])) : { [row.keys[0]]: value };
    (only ? row.keys.filter((key) => !only.includes(key)) : []).forEach((key) => { delete body[key]; });
    return body;
}

/** Whether a read shows the value a write sent. An entry of a list or a map is the same when every field is: a field a
 *  read masks (a key's secret) never equals the value typed, so such a write stays unconfirmed. */
function sameValue(a, b) {
    if (Array.isArray(a) || Array.isArray(b)) {
        return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((value, i) => sameValue(value, b[i]));
    }
    if (a === undefined || a === null || b === undefined || b === null) return a === b;
    if (typeof a === 'object' || typeof b === 'object') {
        return typeof a === 'object' && typeof b === 'object'
            && [...new Set([...Object.keys(a), ...Object.keys(b)])].every((key) => sameValue(a[key], b[key]));
    }
    const [x, y] = [String(a).trim(), String(b).trim()];
    return x === y || (x !== '' && y !== '' && Number(x) === Number(y));
}

// When a change applies is the server's fact (`_meta.effects`); the client keeps no table of keys. A key the map does
// not name (an older server) applies to new tasks, the server's own default; several keys take the latest of them.
const EFFECT_ORDER = ['now', 'next_call', 'next_wake', 'next_task', 'local_model', 'restart'];
const EFFECT_WORDS = {
    next_call: 'applies from the next model call', next_wake: 'applies from the next wake', next_task: 'applies to new tasks',
    restart: 'applies after restart', local_model: 'applies when the local model is restarted',
};
function effectOf(row) {
    const effects = lastSettings?._meta?.effects || {};
    const ranks = row.keys.map((key) => (EFFECT_ORDER.includes(effects[key]) ? EFFECT_ORDER.indexOf(effects[key]) : 3));
    return EFFECT_ORDER[ranks.length ? Math.max(...ranks) : 3];
}

/** The server's answer to a write that landed, as an Outcome. `changes` is the receipt (old, new, token per key);
 *  without it (an older server, `receipt: 'unavailable'`) the outcome carries no old value and no token. */
function savedOutcome(row, value, data = {}) {
    const base = { effect: effectOf(row), withheld: '', message: (data?.warnings || []).join(' '), code: '',
        references: Array.isArray(data?.connection_references) ? data.connection_references : [] };
    const ignored = data?.ignored_keys;
    if (row.keys.some((key) => (Array.isArray(ignored) ? ignored.includes(key) : Boolean(ignored && Object.hasOwn(ignored, key))))) {
        return { ...base, state: 'refused', code: 'ignored_key', message: 'This setting is not changed from here.' };
    }
    const changes = data?.changes && typeof data.changes === 'object' ? data.changes : null;
    const entries = row.keys.map((key) => changes?.[key] || null);
    const receipt = entries.some(Boolean);
    const outcome = { ...base, state: 'saved', withheld: entries.find((entry) => entry?.withheld)?.withheld || '' };
    if (data?.no_changes === true || (changes && !receipt && data.receipt !== 'unavailable')) outcome.state = 'unchanged';
    if (receipt && !outcome.withheld) {
        // A key the receipt does not name did not change: it was what the last read showed, and nothing of it landed.
        const now = currentValue(row);
        const side = (name) => entries.map((entry, i) => (entry ? entry[name] : name === 'old' ? now?.[i] ?? value?.[i] : undefined));
        [outcome.old, outcome.new] = row.keys.length > 1 ? [side('old'), side('new')] : [entries[0].old, entries[0].new];
        const absent = entries.map((entry) => entry?.old_absent === true);
        outcome.oldAbsent = row.keys.length > 1 ? absent : absent[0];
    }
    const tokens = Object.fromEntries(row.keys.map((key, i) => [key, entries[i]?.token]).filter(([, token]) => token));
    if (Object.keys(tokens).length) outcome.token = row.keys.length > 1 ? tokens : tokens[row.keys[0]];
    const running = data?.restart_state?.running;
    if (running && Object.hasOwn(running, row.keys[0])) outcome.running = running[row.keys[0]];
    return outcome;
}

/** A write that threw, through `settingsWriteFailure`: saved with a failed after-step, refused, or unknown; a 409
 *  `settings_changed_elsewhere` is its own state. The raw text is kept only where it says more than the sentence. */
function failureOutcome(row, value, error) {
    const body = error?.body || error?.payload || {};
    const code = String(body.code || body.reason_code || '');
    const words = errorWords(error);
    const bare = (text) => String(text || '').replace(/[.\s]+$/, '');
    const details = bare(words.details) && bare(words.details) !== bare(words.text) ? words.details : '';
    const base = { effect: effectOf(row), withheld: '', message: words.text, details, code };
    if (error?.status === 409 && code === 'settings_changed_elsewhere') return { ...base, state: 'changed_elsewhere' };
    if (settingsWriteFailure(error).unknown) return { ...base, state: 'unknown' };
    if (body.saved !== true) return { ...base, state: 'refused' };
    const failed = body.post_commit_failed ? `The ${body.post_commit_failed} step failed afterwards.` : words.text;
    return { ...savedOutcome(row, value, body), state: 'saved_then_failed', message: failed, details: words.details || words.text, code };
}

const refusal = (row, message, code = 'invalid_value') => ({ state: 'refused', effect: effectOf(row), withheld: '', message, code });
const unchanged = (row) => ({ state: 'unchanged', effect: effectOf(row), withheld: '', message: '', code: '' });

async function attempt(row, value, call) {
    try { return savedOutcome(row, value, await call()); } catch (error) { return failureOutcome(row, value, error); }
}

// An owner endpoint (or the desktop's bridge) answers `ok: true` once it wrote; any other answer wrote nothing.
function ownerResult(result, fallback) {
    if (result?.ok === true) return result;
    throw Object.assign(new Error(result?.error || fallback), { body: { saved: false, ...(result || {}) } });
}

/** The owner's Access write: every surface writes through the one owner endpoint (it asks in CONFIRMS.raiseAccess). */
export async function saveRuntimeMode(mode) {
    return ownerResult(await apiClient.ownerRuntimeMode(mode), 'Runtime mode change failed.');
}

/** Reviewed-skill auto-grant: the desktop's own bridge asks and writes where it exists, else the owner endpoint. */
export async function saveAutoGrant(enabled) {
    const bridge = globalThis.window?.pywebview?.api?.request_auto_grant_reviewed_skills_change;
    return ownerResult(bridge ? await bridge(Boolean(enabled)) : await apiClient.ownerAutoGrant(Boolean(enabled)),
        'Reviewed-skill auto-grant change was cancelled.');
}

/** Main's context mode, through its owner endpoint. */
export async function saveContextMode(mode) {
    return ownerResult(await apiClient.ownerContextMode(mode), 'Context mode change failed.');
}

// A host control answers its own snapshot; the row shows the last one it saw.
const hostCodec = (get, set) => ({
    host: true, get, set,
    read: (settings, row) => { const snapshot = HOST_STATES.get(row.codec); return snapshot ? snapshot.state === 'on' : undefined; },
});

/** How a row's value is read and sent when it is not "one key equals one value". */
export const CODECS = {
    bool: { read: (settings, row) => (settings ? truthy(settings[row.keys[0]]) : undefined), body: (value, row) => ({ [row.keys[0]]: Boolean(value) }) },
    autostart: hostCodec(() => apiClient.desktopAutostart(), (on) => apiClient.setDesktopAutostart(on)),
    background: hostCodec(() => apiClient.desktopBackground(), (on) => apiClient.setDesktopBackground(on)),
};

async function readHost(row) {
    try { HOST_STATES.set(row.codec, await CODECS[row.codec].get()); } catch { /* the row stays as it was drawn */ }
    repaintRow(row.id);
}

// A failed host write reads the host again, because a refusal may land between two writes (settings_autostart.js).
async function hostWrite(row, value) {
    const codec = CODECS[row.codec];
    const wanted = value ? 'on' : 'off';
    const outcome = (snapshot) => {
        HOST_STATES.set(row.codec, snapshot);
        return snapshot?.state === wanted ? { state: 'saved', effect: 'now', withheld: '', new: Boolean(value), message: '', code: '' }
            : { ...refusal(row, snapshot?.reason || 'This computer did not change it.', 'host_refused'), effect: 'now' };
    };
    try { return outcome(await codec.set(Boolean(value))); } catch (error) {
        try {
            const after = outcome(await codec.get());
            return after.state === 'saved' ? after : { ...after, message: errorWords(error).text };
        } catch { return { state: 'unknown', effect: 'now', withheld: '', message: errorWords(error).text, code: '' }; }
    }
}

const preconditionOf = (row, token) => (!token ? null : typeof token === 'string' ? { [row.keys[0]]: token } : token);

/** Every way a row is written. Each answers an Outcome and never throws. */
export const WRITERS = {
    generic(row, value, { ifUnchanged = '', unset = [], only = null } = {}) {
        const precondition = preconditionOf(row, ifUnchanged);
        const body = { ...bodyOf(row, value, only), ...(precondition ? { if_unchanged: precondition } : {}) };
        if (unset.length) {
            unset.forEach((key) => { delete body[key]; });
            body.unset = unset;
        }
        return attempt(row, value, () => apiClient.saveSettings(body, { headers: APP_SETTINGS_ORIGIN }));
    },
    // Every key of SAFETY_MODEL_KEYS: the owner's checker transaction. A change that leaves no reviewer asks first,
    // then the same settings go again with the owner's confirmation.
    async checker(row, value, { ifUnchanged = '', only = null, ask = openConfirmDialog } = {}) {
        const send = (allowEmptyReviewPool) => attempt(row, value, () => saveCheckerSettings(bodyOf(row, value, only),
            { allowEmptyReviewPool, ifUnchanged: preconditionOf(row, ifUnchanged), headers: APP_SETTINGS_ORIGIN }));
        const outcome = await send(false);
        if (outcome.code !== 'empty_review_pool') return outcome;
        return (await CONFIRMS.emptyReviewPool(row, outcome, null, ask)) ? send(true) : unchanged(row);
    },
    runtimeMode: (row, value) => attempt(row, value, () => saveRuntimeMode(value)),
    autoGrant: (row, value) => attempt(row, value, () => saveAutoGrant(truthy(value))),
    contextMode: (row, value) => attempt(row, value, () => saveContextMode(value)),
    safetyMode: (row, value) => attempt(row, value, () => saveSafetyLevel(value,
        row.keys[0] === 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE' ? 'consciousness' : 'your_work', { headers: APP_SETTINGS_ORIGIN })),
    // Your rules: the owner's text through its own audited endpoint (the page's Save, settings_safety.js).
    safetyRules: (row, value) => attempt(row, value, () => saveSafetyRules(value, { headers: APP_SETTINGS_ORIGIN })),
    language: (row, value) => attempt(row, value, () => apiClient.saveUiLanguage(String(value ?? ''))),
    host: hostWrite,
    // A device-local store (theme, size, notifications): its codec's `write` applies at once on this device.
    async device(row, value) {
        try {
            await CODECS[row.codec].write(value, row);
            return { state: 'saved', effect: 'now', withheld: '', new: value, message: '', code: '' };
        } catch (error) { return { ...refusal(row, errorWords(error).text, 'device_refused'), effect: 'now' }; }
    },
};
// Only these writers' endpoints take `if_unchanged`, so only their rows offer Undo.
export const GUARDED_WRITERS = new Set(['generic', 'checker']);

/** Rows a page leaves out: each reads the last settings document. */
export const HIDDEN_WHEN = {
    managedDeployment: (settings) => Boolean(settings?._meta?.policy_state?.deployment),
};

const optionRank = (row, value) => (controlOf(row).options || []).findIndex((option) => String(option.value) === String(value));
function valueWords(row, value) {
    if (Array.isArray(value)) return value.map((part) => valueWords(row, part)).join(' to ');
    const control = controlOf(row);
    if (control.noLimit && isUnlimited(value)) return 'no limit';
    const words = (control.options || []).find((option) => String(option.value) === String(value))?.label ?? String(value ?? '');
    // An empty value is said in a word, the row's own (`control.empty`) where it has one: "was none", never "was ".
    return words === '' ? control.empty || 'empty' : words;
}

/** The changes that ask first. Each answers whether to go on; a change that needs no question answers true at once. */
export const CONFIRMS = {
    async raiseAccess(row, next, current, ask = openConfirmDialog) {
        if (optionRank(row, next) <= optionRank(row, current)) return true;
        // The desktop's bridge is confirmation only; every surface then writes through the one owner endpoint.
        const native = globalThis.window?.pywebview?.api?.confirm_runtime_mode_change;
        if (native) return (await native(next))?.confirmed === true;
        return ask({ title: 'Raise Access',
            body: `Raise Access from ${valueWords(row, current)} to ${valueWords(row, next)}? The change takes effect after restart.`,
            confirmLabel: 'Raise Access' });
    },
    lowerSafety(row, next, current, ask = openConfirmDialog) {
        if (optionRank(row, next) >= optionRank(row, current)) return true;
        return ask({ title: 'Lower a safety level',
            body: `Lower the safety level for ${String(row.name || '').toLowerCase()} from ${valueWords(row, current)} to ${valueWords(row, next)}?\n\n`
                + 'Fixed rules still guard protected files at every level, and every step the lower level lets through is recorded.',
            confirmLabel: 'Lower the level', danger: true });
    },
    openNetwork(row, next, current, ask = openConfirmDialog) {
        const host = String((Array.isArray(next) ? next[0] : next) ?? '').trim();
        if (!host || /^(localhost|127\.|::1$|\[::1\]$)/i.test(host)) return true;
        return ask({ title: 'Open to the network',
            body: 'Other devices on your network will reach Ouroboros with the network password. The change takes effect after restart.',
            confirmLabel: 'Open to the network' });
    },
    autoGrantOn(row, next, current, ask = openConfirmDialog) {
        // The desktop's bridge asks in its own dialog when it writes (saveAutoGrant).
        if (!truthy(next) || globalThis.window?.pywebview?.api?.request_auto_grant_reviewed_skills_change) return true;
        return ask({ title: 'Give reviewed skills their keys?',
            body: 'A skill gets the keys and permissions it asks for after a review of exactly that version.',
            confirmLabel: 'Turn on' });
    },
    removeEntry(row, next, current, ask = openConfirmDialog) {
        return ask({ title: `Remove ${row?.name || 'this entry'}?`, body: 'It is removed at once. There is no Undo.', confirmLabel: 'Remove', danger: true });
    },
    // `refused` is the server's empty_review_pool refusal; its sentence says what would be left.
    emptyReviewPool(row, refused, current, ask = openConfirmDialog) {
        return ask({ title: 'Save without reviewers?', body: refused?.message || 'No agent is left to review.',
            confirmLabel: 'Save without reviewers', danger: true });
    },
};

function patchSettings(row, value, outcome) {
    settingsEpoch += 1;
    if (!lastSettings || !row.keys.length || CODECS[row.codec]?.host || row.control?.kind === 'secret' || outcome.withheld === 'secret') return;
    const landed = outcome.new !== undefined ? outcome.new : value;
    row.keys.forEach((key, i) => {
        if (row.keys.length === 1) lastSettings[key] = landed;
        else if (landed?.[i] !== undefined) lastSettings[key] = landed[i];  // a key the receipt did not name stays as read
    });
    const tokens = lastSettings._meta?.tokens;
    if (tokens && outcome.token) Object.assign(tokens, typeof outcome.token === 'string' ? { [row.keys[0]]: outcome.token } : outcome.token);
    announce(lastSettings);
}

async function writeOnce(row, state, value, options) {
    if (state.reread) {
        // An unknown outcome may still commit (a timed-out worker goes on): read before anything else is sent.
        let fresh;
        try { fresh = await readSettings(); } catch (error) {
            return { state: 'unknown', effect: effectOf(row), withheld: '', message: errorWords(error).text, code: '' };
        }
        state.reread = false;
        if (sameValue(currentValue(row, fresh), value)) return { state: 'saved', effect: effectOf(row), withheld: '', message: '', code: '' };
    }
    let outcome = await (WRITERS[row.writer || 'generic'] || WRITERS.generic)(row, value, options);
    if (outcome.state === 'unknown' || outcome.state === 'changed_elsewhere') {
        state.reread = outcome.state === 'unknown';
        try {
            const fresh = await readSettings();
            if (state.reread && sameValue(currentValue(row, fresh), value)) {
                state.reread = false;
                outcome = { ...outcome, state: 'saved', message: '', details: '' };
            }
        } catch { /* the row keeps saying what it knows */ }
    }
    return outcome;
}

async function drain(row, state, first) {
    let job = first;
    while (job) {
        Object.assign(state, { busy: true, job });
        repaintRow(row.id);
        const outcome = job.local || await writeOnce(row, state, job.value, job.options);
        const landed = outcome.state === 'saved' || outcome.state === 'saved_then_failed';
        if (landed) patchSettings(row, job.value, outcome);
        if (!job.local) state.sent = job.value;
        // The answer stands as the row's result unless a newer value is already on its way.
        if (!state.waiting) {
            Object.assign(state, { outcome, seen: false });
            if (landed || outcome.state === 'unchanged' || outcome.state === 'changed_elsewhere') state.typed = null;
        }
        job.settle.forEach((settle) => settle(outcome));
        if (landed) {
            globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:settings-updated',
                { detail: { reason: 'setting saved', source: 'settings-row', row: row.id, page: row.page, keys: row.keys } }));
        }
        job = state.waiting;
        state.waiting = null;
    }
    Object.assign(state, { busy: false, job: null });
    repaintRow(row.id);
}

// One row is one queue: a write in flight is never doubled, and a newer value replaces the one waiting behind it.
function enqueue(row, job) {
    const state = rowState(row);
    state.row = row;
    return new Promise((settle) => {
        job.settle = [settle];
        if (!state.busy) { void drain(row, state, job); return; }
        if (state.waiting) job.settle.push(...state.waiting.settle);
        state.waiting = job;
    });
}

/**
 * Write one row and answer its Outcome. While a write of the row is in flight a newer value waits and is sent after
 * it, alone. `confirmed` skips the row's question (Undo); `ifUnchanged` is the token Undo sends as its precondition;
 * `only` names the keys Undo writes; `ask` stands in for the shared confirm dialog.
 */
export async function commitSetting(row, value, { confirmed = false, ifUnchanged = '', unset = [], only = null, ask = openConfirmDialog } = {}) {
    if (!lastSettings) await ensureRead();  // effects, tokens and a contract field's bounds
    const question = CONFIRMS[row.confirm];
    if (question && !confirmed && !(await question(row, value, currentValue(row), ask))) {
        repaintRow(row.id);
        return unchanged(row);
    }
    return enqueue(row, { value, options: { ifUnchanged, unset, only, ask } });
}

function entryList(document) {
    let parsed = document;
    if (typeof parsed === 'string') { try { parsed = JSON.parse(parsed || 'null'); } catch { return null; } }
    const list = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.items) ? parsed.items : null;
    return list ? { list, wrap: (next) => (Array.isArray(parsed) ? next : { ...parsed, items: next }) } : null;
}
const entryId = (entry) => String(entry?.id || entry?.subagent_id || entry?.slug || entry?.name || '');

/**
 * Change one entry of a list setting (`MCP_SERVERS`, `OUROBOROS_SUBAGENTS`, ...): a fresh read, the one entry
 * changed (`change`: fields to merge, a function of the entry, or null to remove it), then that key posted with the
 * read's token as `if_unchanged`. A list that changed meanwhile is refused and never retried here. The
 * `empty_review_pool` refusal asks the owner and repeats the same body with the flag. `shown` is the list on screen
 * for a list never saved, written whole; `checker` is the editor's adapter for a roster change that must go through
 * the Safety checker's transaction: given the body, it answers the server's answer, or null to leave it to the
 * generic save.
 */
export function writeEntry(key, id, change, { confirmed = false, ask = openConfirmDialog, name = '', shown = null, checker = null } = {}) {
    const row = { id: key, keys: [key], name };
    const run = async () => {
        if (change === null && !confirmed && !(await CONFIRMS.removeEntry(row, null, null, ask))) return unchanged(row);
        let fresh;
        try { fresh = await readSettings(); } catch (error) { return refusal(row, errorWords(error).text, 'settings_unread'); }
        const stored = entryList(fresh[key]) || entryList(shown);
        if (!stored) return refusal(row, 'This list could not be read.', 'settings_unread');
        const index = stored.list.findIndex((entry) => entryId(entry) === String(id));
        const before = index < 0 ? null : stored.list[index];
        if (!before && typeof change !== 'function') return { ...unchanged(row), state: 'changed_elsewhere' };
        const after = typeof change === 'function' ? change(before) : change === null ? null : { ...before, ...change };
        const next = after === null || after === undefined ? stored.list.filter((_, i) => i !== index)
            : index < 0 ? [...stored.list, after] : stored.list.map((entry, i) => (i === index ? after : entry));
        const token = fresh._meta?.tokens?.[key];
        const body = { [key]: stored.wrap(next), ...(token ? { if_unchanged: { [key]: token } } : {}) };
        const post = (sent) => attempt(row, sent[key],
            () => checker?.({ key, id, body: sent, fresh }) ?? apiClient.saveSettings(sent, { headers: APP_SETTINGS_ORIGIN }));
        let outcome = await post(body);
        if (outcome.state === 'refused' && outcome.code === 'empty_review_pool') {
            if (!(await CONFIRMS.emptyReviewPool(row, outcome, null, ask))) return unchanged(row);
            outcome = await post({ ...body, allow_empty_review_pool: true });  // the same roster, never a newer one
        }
        if (outcome.state === 'saved' || outcome.state === 'saved_then_failed') {
            patchSettings(row, stored.wrap(next), { ...outcome, new: undefined });
            globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:settings-updated',
                { detail: { reason: 'entry saved', source: 'settings-row', row: key, keys: [key] } }));
        }
        return outcome;
    };
    const queued = (ENTRY_QUEUES.get(key) || Promise.resolve()).then(run);
    ENTRY_QUEUES.set(key, queued.catch(() => {}));
    return queued;
}

/** What a row says after a write (the one wording of every row): `{ text, tone, undo, note, details }`, or null for
 *  nothing. A switch, strip or select that acts now is its own confirmation; a refusal, an unknown outcome and a
 *  warning always speak. Without a receipt (an older server, a receipt that could not be taken) it is `Saved`, with no
 *  Undo; Undo also needs the installed value's token and a writer whose endpoint checks it. */
export function outcomeLine(row, outcome) {
    if (!outcome || outcome.state === 'unchanged') return null;
    const say = (text, tone, more = {}) => ({ text, tone, undo: false, note: '', details: '', ...more });
    if (outcome.state === 'refused') return say(`Not saved. ${outcome.message}`, 'danger', { details: outcome.details || '' });
    if (outcome.state === 'unknown') return say('Not confirmed yet', 'warn');
    if (outcome.state === 'changed_elsewhere') return say('Changed on another device. This is the current value.', 'warn');
    if (outcome.state === 'saved_then_failed') return say(`Saved. ${outcome.message}`, 'warn', { details: outcome.details || '' });
    if (!isTyped(row) && outcome.effect === 'now' && !outcome.message) return null;
    const known = outcome.old !== undefined;
    const parts = ['Saved'];
    if ((known || outcome.withheld) && EFFECT_WORDS[outcome.effect]) parts.push(EFFECT_WORDS[outcome.effect]);
    // A switch has one other value and a restart row says both values in its meta: neither repeats it here.
    if (known && row.control?.kind !== 'switch' && outcome.effect !== 'restart') parts.push(`was ${valueWords(row, outcome.old)}`);
    const undo = known && !outcome.withheld && Boolean(outcome.token) && GUARDED_WRITERS.has(row.writer || 'generic');
    return say(parts.join(' · '), outcome.message || outcome.effect === 'restart' ? 'warn' : 'ok', { undo, note: outcome.message || '' });
}

const rowDomId = (row) => `setting-${row.id}`;
const NUMBER_RULE = (min, max) => (min !== null && max !== null ? `Enter a number from ${min} to ${max}.`
    : min !== null ? `Enter a number, ${min} or more.` : max !== null ? `Enter a number, ${max} or less.` : 'Enter a number.');
const bound = (value) => (value === undefined || value === null || value === '' || !Number.isFinite(Number(value)) ? null : Number(value));

/** Typed text as the value to send, or the sentence that refuses it here. A bound that may be `unlimited` goes to
 *  the server as typed: it refuses a blank, a zero or a word and says why. */
function parseTyped(row, text) {
    const control = controlOf(row);
    if (control.kind === 'pair') {
        const sides = [].concat(text).map((part) => parseTyped({ ...row, control: { ...control, kind: 'number' } }, part));
        return sides.find((side) => side.refuse) || { value: sides.map((side) => side.value) };
    }
    if (control.kind !== 'number') return { value: text };
    const trimmed = String(text ?? '').trim();
    const number = trimmed === '' ? NaN : Number(trimmed);
    if (control.noLimit) return { value: Number.isFinite(number) ? number : trimmed };
    const [min, max] = [bound(control.min), bound(control.max)];
    if (!Number.isFinite(number) || (control.integer && !Number.isInteger(number)) || (min !== null && number < min) || (max !== null && number > max)) return { refuse: NUMBER_RULE(min, max) };
    return { value: number };
}

/** The sign at the end of a row that leads somewhere, one for every page (settings.css draws it as the home's two
 *  glyphs): › opens a page of Settings, ↗ (`data-out`) leaves Settings. `to` is a Settings path or a door id. */
export const doorSign = (to = '') => `<span class="settings-row-chevron" aria-hidden="true"${isDoorId(to) ? ' data-out' : ''}>›</span>`;

function controlHtml(row, control, ctx) {
    const id = rowDomId(row);
    const described = [`${id}-meta`, `${id}-result`, ctx.describedBy?.(row) || ''].filter(Boolean).join(' ');
    // An empty text field says its state in the row's own word ("None"), as an empty number says "No limit".
    const emptyWord = control.kind === 'text' && control.empty ? control.empty[0].toUpperCase() + control.empty.slice(1) : '';
    const input = (index, label = '') => `<input ${index ? '' : `id="${id}" `}class="ui-control" data-setting-control autocomplete="off"`
        + ` type="${control.kind === 'secret' ? 'password' : 'text'}"`
        + `${control.kind === 'number' || control.kind === 'pair' ? ` inputmode="${control.noLimit ? 'numeric' : 'decimal'}"` : ''}`
        + `${control.noLimit ? ' placeholder="No limit"' : emptyWord ? ` placeholder="${escapeHtmlAttr(emptyWord)}"` : ''}${label ? ` aria-label="${escapeHtmlAttr(label)}"` : ''} aria-describedby="${described}">`;
    const unit = control.unit ? `<span class="settings-row-unit">${escapeHtmlAttr(control.unit)}</span>` : '';
    const field = (body) => `<span class="settings-row-field">${control.unit === '$' ? unit + body : body + unit}</span>`;
    const named = `aria-labelledby="${id}-name" aria-describedby="${described}"`;
    const options = control.options || [];
    switch (control.kind) {
    case 'number': return field(input(0))
        + (control.noLimit ? '<button type="button" class="ui-segment settings-row-no-limit" data-setting-no-limit aria-pressed="false">No limit</button>' : '');
    case 'text': case 'secret': return input(0);
    case 'pair': return field(input(0, control.labels?.[0] || `${row.name}, first`) + input(1, control.labels?.[1] || `${row.name}, second`));
    case 'select': return `<select id="${id}" class="ui-control" data-setting-control aria-describedby="${described}">${options.map((option) =>
        `<option value="${escapeHtmlAttr(option.value)}">${escapeHtmlAttr(option.label ?? option.value)}</option>`).join('')}</select>`;
    case 'switch': return `<label class="ui-switch"><input type="checkbox" role="switch" id="${id}" data-setting-control aria-checked="false" ${named}>`
        + '<span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span></label>';
    case 'strip': return `<div class="ui-segment-group" role="group" ${named}>${options.map((option) =>
        `<button type="button" class="ui-segment" data-setting-value="${escapeHtmlAttr(option.value)}" aria-pressed="false">${escapeHtmlAttr(option.label ?? option.value)}</button>`).join('')}</div>`;
    // Options that each need a line of meaning: one under another, a radio group; the chosen one is `aria-checked`.
    case 'choices': return `<div class="settings-choices" role="radiogroup" ${named}>${options.map((option) =>
        `<button type="button" class="settings-choice" role="radio" aria-checked="false" data-setting-value="${escapeHtmlAttr(option.value)}">`
        + `<span class="settings-choice-name">${escapeHtmlAttr(option.label ?? option.value)}<span class="settings-choice-mark" data-setting-mark></span></span>`
        + `<span class="settings-choice-note">${escapeHtmlAttr(option.note || '')}</span></button>`).join('')}</div>`;
    case 'door': return `${ctx.doorValue?.(row) || ''}${doorSign(control.to)}`;  // a door may say its value before its sign
    default: return ctx.custom?.(row) || '';
    }
}

/** One row as HTML: its name, one meta line with the row's result after it, its control at the end, and its More.
 *  `ctx.describedBy(row)` adds ids to the control's description; `ctx.custom(row)` draws a `custom` control. */
export function settingsRow(row, ctx = {}) {
    const control = controlOf(row);
    const id = rowDomId(row);
    const name = escapeHtmlAttr(row.name);
    const meta = `<span id="${id}-meta" data-setting-meta>${escapeHtmlAttr(row.meta || '')}</span>`;
    const attrs = `class="ui-row settings-row" data-setting-row="${escapeHtmlAttr(row.id)}" data-kind="${escapeHtmlAttr(control.as || control.kind || 'custom')}"${control.noLimit ? ' data-no-limit' : ''}`;
    if (control.kind === 'door') {  // the whole row is the press: phrasing content only
        return `<button type="button" ${attrs} data-setting-door><span class="ui-row-title">${name}</span>`
            + `<span class="ui-row-meta">${meta}</span><span class="ui-row-actions">${controlHtml(row, control, ctx)}</span></button>`;
    }
    const labelled = ['number', 'text', 'secret', 'select'].includes(control.kind);
    return `<div ${attrs}>
        ${labelled ? `<label class="ui-row-title" for="${id}">${name}</label>` : `<p class="ui-row-title" id="${id}-name">${name}</p>`}
        <div class="ui-row-meta">${meta}<span class="ui-result settings-row-result" id="${id}-result" data-setting-result role="status" aria-live="polite"></span></div>
        <div class="ui-row-actions">${controlHtml(row, control, ctx)}</div>
        ${row.more ? `<details class="ui-disclosure settings-row-more"><summary>More</summary><p class="settings-row-copy">${escapeHtmlAttr(row.more)}</p></details>` : ''}
    </div>`;
}

// The row's result line: the sentence, then Undo as a link inside it, then a warning the server sent. No timer ends it.
function paintResult(result, line) {
    const signature = line ? `${line.tone}|${line.undo}|${line.text}|${line.note}|${line.details}` : '';
    if (!result || (result.dataset.settingLine || '') === signature) return;
    result.dataset.settingLine = signature;
    const tail = line?.note ? `. ${line.note}` : '';
    setResult(result, line ? { text: line.undo ? `${line.text} · ` : line.text + tail, tone: line.tone, details: line.details } : {});
    if (!line?.undo) return;
    const doc = result.ownerDocument;
    const undo = doc.createElement('button');
    undo.setAttribute('type', 'button');
    undo.className = 'consciousness-inline-link ui-hit-area';
    undo.setAttribute('data-setting-undo', '');
    undo.textContent = 'Undo';
    result.append(undo);
    if (tail) result.append(doc.createTextNode(tail));
}

function paintRow(node, row) {
    const state = rowState(row);
    const control = controlOf(row);
    const stored = currentValue(row);
    const shown = state.busy && !state.job.local ? state.job.value : stored;
    const forced = Boolean(state.outcome && !state.seen && state.outcome.state === 'changed_elsewhere');
    const doc = node.ownerDocument || globalThis.document;
    const controls = node.querySelectorAll('[data-setting-control]');
    // The value still in force while a restart waits: the write's answer, the row's codec, else the server's list.
    const running = state.outcome?.running ?? CODECS[row.codec]?.running?.(lastSettings, row)
        ?? lastSettings?._meta?.restart_state?.running?.[row.keys[0]];
    const waits = running !== undefined && stored !== undefined && !sameValue(running, stored);
    if (row.hiddenWhen) node.hidden = Boolean(HIDDEN_WHEN[row.hiddenWhen]?.(lastSettings));
    if (isTyped(row)) {
        const text = (value) => (value === undefined ? undefined : control.kind === 'secret' || (control.noLimit && isUnlimited(value)) ? '' : String(value ?? ''));
        const texts = state.typed !== null ? [].concat(state.typed) : [].concat(Array.isArray(shown) ? shown.map(text) : text(shown));
        controls.forEach((input, i) => {
            // A field the owner is in, or has typed into, is never rewritten under them.
            const held = !forced && (doc?.activeElement === input || input.dataset.settingDirty === '1');
            if (!held && texts[i] !== undefined && input.value !== texts[i]) input.value = texts[i];
        });
        const none = node.querySelector('[data-setting-no-limit]');
        if (none && shown !== undefined) {
            none.setAttribute('aria-pressed', String(isUnlimited(shown)));
            none.classList.toggle('active', isUnlimited(shown));
        }
    } else if (control.kind === 'select') {
        if (shown !== undefined && controls[0] && controls[0].value !== String(shown ?? '')) controls[0].value = String(shown ?? '');
    } else if (control.kind === 'switch') {
        if (shown !== undefined && controls[0]) {
            controls[0].checked = truthy(shown);
            controls[0].setAttribute('aria-checked', String(truthy(shown)));
        }
    } else if (control.kind === 'strip') {
        node.querySelectorAll('[data-setting-value]').forEach((button) => {
            const chosen = shown !== undefined && String(button.dataset.settingValue) === String(shown ?? '');
            button.classList.toggle('active', chosen);
            button.setAttribute('aria-pressed', String(chosen));
        });
    } else if (control.kind === 'choices') {
        // One tab stop, on the chosen option (the first while none is); while a restart waits the option in force
        // and the one chosen say so in words.
        const options = [...node.querySelectorAll('[data-setting-value]')];
        const at = options.findIndex((button) => shown !== undefined && String(button.dataset.settingValue) === String(shown ?? ''));
        options.forEach((button, i) => {
            button.setAttribute('aria-checked', String(i === at));
            button.tabIndex = i === Math.max(at, 0) ? 0 : -1;
            const words = !waits ? '' : sameValue(button.dataset.settingValue, running) ? 'Now' : i === at ? 'After restart' : '';
            const mark = button.querySelector('[data-setting-mark]');
            if (mark && mark.textContent !== words) mark.textContent = words;
        });
    }
    // A write in flight: the control is disabled and busy; nothing else is drawn for "saving". Disabling moves the
    // focus away, so a control that had it takes it back when the answer comes, unless the owner has moved on.
    [...controls, ...node.querySelectorAll('[data-setting-value]'), ...node.querySelectorAll('[data-setting-no-limit]')].forEach((element) => {
        if (state.busy) {
            if (!element.dataset.settingBusy && doc?.activeElement === element) element.dataset.settingRefocus = '1';
            element.disabled = true;
            element.dataset.settingBusy = '1';
            element.setAttribute('aria-busy', 'true');
        } else if (element.dataset.settingBusy) {
            element.disabled = false;
            delete element.dataset.settingBusy;
            element.removeAttribute('aria-busy');
            if (element.dataset.settingRefocus) {
                delete element.dataset.settingRefocus;
                const active = doc?.activeElement;
                if (!active || active === doc.body || active === element) element.focus?.();
            }
        }
    });
    // A strip or a select whose options carry a line of meaning says the chosen one's (a `choices` row and a page's own
    // `custom` control draw each option's line themselves).
    const noted = ['strip', 'select'].includes(control.kind)
        && (control.options || []).find((option) => String(option.value) === String(shown ?? ''))?.note;
    const meta = waits ? `Running: ${valueWords(row, running)} · after restart: ${valueWords(row, stored)}` : noted || row.meta || '';
    const metaNode = node.querySelector('[data-setting-meta]');
    if (metaNode && metaNode.textContent !== meta) metaNode.textContent = meta;
    paintResult(node.querySelector('[data-setting-result]'), outcomeLine(row, state.outcome));
    if (state.outcome) state.seen = true;
}

// A value from a control: the value already there (or already on its way) sends nothing.
function commitValue(row, value, ctx) {
    const state = rowState(row);
    const base = state.busy ? state.job.value : currentValue(row);
    if (lastSettings && !state.reread && sameValue(base, value)) {
        if (state.busy && state.waiting) {
            state.job.settle.push(...state.waiting.settle);
            state.waiting = null;
        } else if (!state.busy) Object.assign(state, { outcome: null, typed: null });
        repaintRow(row.id);
        return Promise.resolve(unchanged(row));
    }
    return commitSetting(row, value, { ask: ctx.ask });
}

function commitTyped(row, text, ctx) {
    const state = rowState(row);
    const parsed = parseTyped(row, text);
    if (row.control?.kind === 'secret' && text === '') return Promise.resolve(unchanged(row));
    state.typed = text;  // kept here, not in the page: a refusal after the page is gone restores it
    if (parsed.refuse) return enqueue(row, { value: undefined, options: {}, local: refusal(row, parsed.refuse) });
    return commitValue(row, parsed.value, ctx);
}

/**
 * Wire the rows drawn in `container`: a typed value, a switch and a select commit on `change` (for a field: leaving it
 * or Enter); a strip option and "No limit" on the press. Each row shows its result line and its Undo. `ctx.rows` names
 * descriptors that are not in settings_rows.js; `ctx.ask` stands in for the confirm dialog; `ctx.openDoor(row)` takes
 * a door's press. Answers `{ paint, leave, dispose }`: `paint()` after a fresh read (it never touches a focused or
 * typed field), `leave()` when the page is left (a result the owner saw ends there), `dispose()` when it is torn down.
 */
export function bindSettingRows(container, ctx = {}) {
    const find = (id) => ctx.rows?.find((row) => row.id === id) || rowById(id);
    const bindings = [];
    container.querySelectorAll('[data-setting-row]').forEach((node) => {
        const row = find(node.dataset.settingRow);
        if (!row) return;
        const paint = () => {
            if (node.isConnected === false) { BOUND.get(row.id)?.delete(paint); return; }
            paintRow(node, row);
        };
        if (!BOUND.has(row.id)) BOUND.set(row.id, new Set());
        BOUND.get(row.id).add(paint);
        bindings.push({ row, node, paint });
        paint();
        if (CODECS[row.codec]?.host && !HOST_STATES.has(row.codec)) void readHost(row);
    });
    const of = (event) => {
        const node = event.target?.closest?.('[data-setting-row]');
        return node ? bindings.find((binding) => binding.node === node) : null;
    };
    const onInput = (event) => {
        const binding = of(event);
        if (!binding || !isTyped(binding.row) || !event.target.matches?.('[data-setting-control]')) return;
        event.target.dataset.settingDirty = '1';
        // An edit of the row starts a new subject: its last result goes.
        rowState(binding.row).outcome = null;
        binding.paint();
    };
    const onChange = (event) => {
        const binding = of(event);
        if (!binding || !event.target.matches?.('[data-setting-control]')) return;
        const { row, node } = binding;
        if (row.control?.kind === 'switch') { void commitValue(row, Boolean(event.target.checked), ctx); return; }
        if (!isTyped(row)) { void commitValue(row, event.target.value, ctx); return; }
        const inputs = node.querySelectorAll('[data-setting-control]');
        inputs.forEach((input) => { delete input.dataset.settingDirty; });
        void commitTyped(row, row.control.kind === 'pair' ? Array.from(inputs, (input) => input.value) : event.target.value, ctx);
    };
    const onClick = (event) => {
        const binding = of(event);
        if (!binding) return;
        const { row, node } = binding;
        const state = rowState(row);
        const hit = (selector) => event.target.closest?.(selector);
        if (hit('[data-setting-undo]')) {
            // The inverse of what the receipt changed: its keys alone, each on the token it installed.
            const { old, oldAbsent, token } = state.outcome || {};
            const only = typeof token === 'string' ? row.keys : Object.keys(token || {});
            const unset = row.keys.filter((key, i) => only.includes(key) && (Array.isArray(oldAbsent) ? oldAbsent[i] : oldAbsent));
            state.typed = null;
            if (old !== undefined && token) void commitSetting(row, old, { confirmed: true, ifUnchanged: token, unset, only, ask: ctx.ask });
        } else if (hit('[data-setting-no-limit]')) {
            state.typed = null;
            node.querySelectorAll('[data-setting-control]').forEach((input) => { delete input.dataset.settingDirty; });
            void commitValue(row, 'unlimited', ctx);
        } else if (hit('[data-setting-value]')) {
            void commitValue(row, hit('[data-setting-value]').dataset.settingValue, ctx);
        } else if (hit('[data-setting-door]')) ctx.openDoor?.(row);
    };
    // A `choices` group as a radio group: the arrows, Home and End move between its options. A press chooses, never
    // the arrow itself: a choice is written at once and may ask first.
    const onKeydown = (event) => {
        const option = event.target?.closest?.('[role="radio"][data-setting-value]');
        const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
        if (!option || !of(event) || !(step || event.key === 'Home' || event.key === 'End')) return;
        const options = [...option.parentElement.querySelectorAll('[data-setting-value]')].filter((each) => !each.disabled);
        const next = event.key === 'Home' ? options[0] : event.key === 'End' ? options[options.length - 1]
            : options[(options.indexOf(option) + step + options.length) % options.length];
        event.preventDefault();
        next?.focus?.();
    };
    container.addEventListener('input', onInput);
    container.addEventListener('change', onChange);
    container.addEventListener('click', onClick);
    container.addEventListener('keydown', onKeydown);
    const leave = () => bindings.forEach(({ row, paint }) => {
        const state = rowState(row);
        if (state.seen && !state.busy) Object.assign(state, { outcome: null, typed: null });
        paint();
    });
    return {
        paint: () => bindings.forEach(({ paint }) => paint()),
        leave,
        dispose() {
            leave();
            bindings.forEach(({ row, paint }) => BOUND.get(row.id)?.delete(paint));
            container.removeEventListener('input', onInput);
            container.removeEventListener('change', onChange);
            container.removeEventListener('click', onClick);
            container.removeEventListener('keydown', onKeydown);
        },
    };
}
