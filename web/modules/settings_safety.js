// The Safety Supervisor in Settings. Permissions and checks, "Who watches its actions": the two levels are rows that
// save themselves (settings_rows.js; settings_controls.js writes them through the owner endpoint and asks before a
// level is lowered); this module says the facts beside them: which model checks, who re-checks, what a managed
// deployment closes, and how many checks were let through. Your rules is its own page: one text with its own Save,
// written by the row helper through the owner endpoint (its writer `safetyRules`) and read again by Ouroboros and the
// Supervisor from the next task or turn.
import { apiClient } from './api_client.js';
import { recheckFromSettings } from './consciousness_api.js';
import { parseModelSource } from './route_editor_primitives.js';
import { commitSetting, outcomeLine } from './settings_controls.js';
import { deploymentOf, safetyExitsNote } from './settings_policy.js';
import { rowById } from './settings_rows.js';
import { setResult } from './ui_primitives.js';

export const SAFETY_RULES_KEY = 'OUROBOROS_SAFETY_OWNER_RULES';
export const SAFETY_MODEL_KEY = 'OUROBOROS_MODEL_SAFETY';

const modelName = (value) => parseModelSource(value).model || String(value || '');

/** An Agents row by the handle the roster shows (its route's model), else its stored id. */
export function agentRowHandle(rowId, settings = {}) {
    let items = [];
    try {
        const raw = settings.OUROBOROS_SUBAGENTS;
        const value = typeof raw === 'string' ? JSON.parse(raw || '{}') : raw;
        items = Array.isArray(value?.items) ? value.items : [];
    } catch { items = []; }
    const row = items.find((item) => String(item?.subagent_id || '') === rowId);
    return row?.route?.target_id ? modelName(row.route.target_id) : rowId;
}

/** Who re-checks a flagged action, from the saved Agents roster (the marked row and the rows that may be marked). */
export function secondCheckWords(recheck, settings = {}) {
    if (!recheck.rowId) {
        return recheck.known && !recheck.eligibleRows.length
            ? 'Off. No agent can re-check yet: it needs an API model with Packet delivery.' : 'Off';
    }
    if (recheck.offReason === 'row_disabled') return 'Off: the marked agent is disabled';
    if (recheck.offReason === 'delegation_off') return 'Off: Delegation is off in Agents';
    return `${agentRowHandle(recheck.rowId, settings)} re-checks flagged actions`;
}

/**
 * The facts beside the two levels, from one settings document: `supervisor` (the model that checks: the Safety
 * Supervisor's own, else the Light model), `secondCheck` and whether no replay has measured it (`unmeasured`), and a
 * managed deployment's part: its checker decides alone and is said, never offered for change (`fixed`), the exits it
 * closes (`exits`) and whether its own rules apply too (`organizationRules`).
 */
export function safetyFacts(settings = {}) {
    const deployment = deploymentOf(settings?._meta);
    const own = String(settings?.[SAFETY_MODEL_KEY] ?? '').trim();
    if (deployment?.checker) {
        return { fixed: true, supervisor: `${modelName(deployment.checker.model)} · Set by your organization`,
            secondCheck: 'None. Your organization’s Supervisor decides alone.', unmeasured: false,
            exits: safetyExitsNote(deployment), organizationRules: Boolean(deployment.organization_rules) };
    }
    const recheck = recheckFromSettings(settings);
    return { fixed: false, supervisor: own ? modelName(own) : 'The Light model', secondCheck: secondCheckWords(recheck, settings),
        // Said wherever the second check is offered, the default Off included: no replay has measured it.
        unmeasured: Boolean(recheck.known && !recheck.measured),
        exits: safetyExitsNote(deployment), organizationRules: Boolean(deployment?.organization_rules) };
}

const SKIP_TAIL = 2000;

/** Waved-through checks among the inspected recent events (the audit of a level below + Commands) and the checks a
 * spare model answered (``spareWords``), counted within 24 h; a full tail younger than a day says the span it read,
 * never a day it did not inspect. The read is an event tail in the compact view: the page asks for it when it is shown
 * (`ouro:settings-page-shown`). */
export async function refreshSafetySkipCounter(root, { logsTail = apiClient.logsTail, now = Date.now() } = {}) {
    const el = root?.querySelector?.('[data-safety-skips]');
    const spares = root?.querySelector?.('[data-safety-spares]');
    if (!el && !spares) return;
    try {
        const data = await logsTail('events', SKIP_TAIL, 'compact');
        const entries = data?.entries || [];
        const cutoff = now - 24 * 3600 * 1000;
        const times = entries.map((entry) => Date.parse(String(entry?.ts || ''))).filter(Number.isFinite);
        const recent = (type) => entries.filter((entry) => {
            if (String(entry?.type || '') !== type) return false;
            const time = Date.parse(String(entry?.ts || ''));
            return Number.isFinite(time) && time >= cutoff;
        });
        const n = recent('safety_mode_skip').length;
        const oldest = times.length ? Math.min(...times) : null;
        const span = entries.length >= SKIP_TAIL && oldest !== null && oldest > cutoff
            ? `since ${new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false }).format(oldest)} (the latest ${SKIP_TAIL.toLocaleString('en-US')} events)`
            : 'in the last 24 h';
        if (el) {
            el.textContent = n > 0
                ? `${n} safety ${n === 1 ? 'check' : 'checks'} let through ${span}.`
                : `No safety checks let through ${span}.`;
        }
        if (spares) spares.textContent = spareWords(recent('safety_route_fallback'), recent('safety_verdict').length, span);
    } catch {
        if (el) el.textContent = '';
        if (spares) spares.textContent = '';
    }
}

/** The checks a spare model answered because the Supervisor's own model gave none (``safety_route_fallback`` events,
 * by the model that answered), the walks no spare answered, and the checks with a recorded verdict in the same span
 * (``safety_verdict`` events; one holds both verdicts when the second check is on): counts of events, never a share
 * (a walk is written before its reply is read). */
export function spareWords(walks, verdicts, span) {
    const checked = `${verdicts} ${verdicts === 1 ? 'check' : 'checks'} with a recorded verdict in the same span.`;
    if (!walks.length) return `No safety check went to a spare model ${span}. ${checked}`;
    const byModel = new Map();
    for (const walk of walks) {
        const model = walk?.answered?.model;
        if (model) byModel.set(modelName(model), (byModel.get(modelName(model)) || 0) + 1);
    }
    const answered = [...byModel.values()].reduce((sum, count) => sum + count, 0);
    const missed = walks.length - answered;
    return [
        answered ? `${answered} safety ${answered === 1 ? 'check was' : 'checks were'} answered by a spare model ${span}, because`
            + ` the Supervisor's own model gave no answer: ${[...byModel].map(([name, count]) => `${name} ×${count}`).join(', ')}.` : '',
        missed ? `${missed} ${missed === 1 ? 'check' : 'checks'} went to the spare models ${answered ? 'and' : span + ' and'} got no answer.` : '',
        checked,
    ].filter(Boolean).join(' ');
}

const RULES_SAVED = 'Saved · applies from the next task or turn';
let unsavedRules = null;  // text typed and not saved survives leaving the page, in this window

/** What Save says: the page's own sentence once saved, else the row helper's words (a refusal, an unknown outcome,
 *  a failed after-step); `{ text, tone, details }`. */
export function rulesSaveWords(outcome, row = rowById(SAFETY_RULES_KEY)) {
    if (outcome?.state === 'saved' || outcome?.state === 'unchanged') {
        return { text: outcome.message ? `${RULES_SAVED}. ${outcome.message}` : RULES_SAVED, tone: outcome.message ? 'warn' : 'ok', details: '' };
    }
    const line = outcomeLine(row, outcome);
    return { text: line?.text || 'Not saved.', tone: line?.tone || 'danger', details: line?.details || '' };
}

/** Your rules, as a page: the text, Save and its result. `mount(container, { ctx })` of settings_pages.js. */
export function mountRulesPage(container, { ctx, save = commitSetting } = {}) {
    const row = rowById(SAFETY_RULES_KEY);
    container.innerHTML = `<p class="settings-page-lead">Standing rules in your own words. Ouroboros reads them every time; the Safety Supervisor reads them before actions.</p>
        <section class="ui-card settings-card" data-setting-row="${SAFETY_RULES_KEY}">
            <textarea id="settings-rules" class="ui-control settings-rules-text" rows="10" spellcheck="false" aria-label="Your rules"
                aria-describedby="settings-rules-note settings-rules-result" data-setting-control></textarea>
            <div class="ui-action-row">
                <button type="button" class="btn btn-save" data-editor-save>Save</button>
                <span class="ui-result" id="settings-rules-result" data-editor-result role="status" aria-live="polite"></span>
            </div>
            <p class="settings-card-note" id="settings-rules-note">Running work keeps the text it started with. Rules for one chat or channel live with that place, in Circles.</p>
            <p class="settings-card-note" data-rules-organization hidden>Your organization’s rules apply too; its administrator sets them.</p>
        </section>`;
    const field = container.querySelector('#settings-rules');
    const button = container.querySelector('[data-editor-save]');
    const result = container.querySelector('[data-editor-result]');
    let stored = null;  // the saved text as last read; null until a read answers
    let gone = false;
    const say = (words) => setResult(result, words || {});
    if (unsavedRules !== null) {
        field.value = unsavedRules;
        say({ text: 'Not saved yet', tone: 'warn' });
    }
    async function read() {
        let settings;
        try { settings = await ctx.readSettings(); } catch { return; }  // the field keeps what it shows
        if (gone) return;
        stored = String(settings?.[SAFETY_RULES_KEY] ?? '');
        container.querySelector('[data-rules-organization]').hidden = !deploymentOf(settings?._meta)?.organization_rules;
        if (unsavedRules === null && field.value !== stored) field.value = stored;
    }
    const onInput = () => {
        unsavedRules = stored !== null && field.value === stored ? null : field.value;
        say(null);  // an edit starts a new subject: the last result goes
    };
    const onSave = async () => {
        if (button.disabled) return;
        const text = field.value;
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
        const outcome = await save(row, text);
        const landed = outcome.state === 'saved' || outcome.state === 'saved_then_failed' || outcome.state === 'unchanged';
        if (landed) {
            stored = text;
            if (field.value === text) unsavedRules = null;
        }
        if (gone) return;
        button.disabled = false;
        button.removeAttribute('aria-busy');
        // A refusal leaves the text in the field: nothing here rewrites it.
        say(rulesSaveWords(outcome, row));
    };
    const onShown = (event) => { if (event.detail?.path === 'rules') void read(); };
    field.addEventListener('input', onInput);
    button.addEventListener('click', onSave);
    globalThis.window?.addEventListener?.('ouro:settings-page-shown', onShown);
    void read();
    return { unmount() {
        gone = true;
        field.removeEventListener('input', onInput);
        button.removeEventListener('click', onSave);
        globalThis.window?.removeEventListener?.('ouro:settings-page-shown', onShown);
    } };
}

/** Forget the unsaved text (a test). */
export function forgetUnsavedRules() { unsavedRules = null; }
