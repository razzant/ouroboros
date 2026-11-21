import { markDesktopEditor } from './desktop_room_session.js';
// The Safety Supervisor in Settings. Permissions and checks, "Who watches its actions": the two levels are rows that
// save themselves (settings_rows.js; settings_controls.js writes them through the owner endpoint and asks before a
// level is lowered); this module says the facts beside them: which model checks, who re-checks, what a managed
// deployment closes, and how many checks were let through. Your rules is its own page: one text with its own Save,
// written by the row helper through the owner endpoint (its writer `safetyRules`) and read again by Ouroboros and the
// Supervisor from the next task or turn.
import { apiClient } from './api_client.js';
import { recheckFromSettings } from './consciousness_api.js';
import { parseModelSource } from './route_editor_primitives.js';
import { commitSetting, outcomeLine, paintWithRow, settingsReadMessage, settingsSnapshot } from './settings_controls.js';
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
const NOT_SAVED = { text: 'Not saved yet', tone: 'warn' };
let unsavedRules = null;  // text typed and not saved survives leaving the page, in this window
// What the owner typed last, in whichever mount of this page (`typedRules`), is safe from a window replacement only when
// it is the persisted text however every save still in flight or unconfirmed (`writing`) ends; `savedRules` is the text
// a read or a landed save last showed persisted. Saves line up in commitSetting's one row queue across mounts, which
// sends the newest value waiting behind the one in flight and answers the values it replaced with that one. Each save
// hands the queue its own record as a ticket and learns whose answer it got (`outcome.answers`): it is reconciled with the
// save the queue ran, never with the text its own press carried. The queue carries no text; the records stay here.
let typedRules = null, savedRules = null;
const writing = new Set();  // `{ text, open }`: open while in flight, then kept only while its outcome stays unknown

/** Whether a save whose answer never came still counts against the words in the field: such a request may complete
 *  after anything and put its own words back, so even words typed back to what is persisted are not safe while one
 *  of other words counts. A field that follows what is persisted (`typedRules` null) holds nothing: no words of the
 *  owner's are at risk. */
function unsettledRules() {
    return typedRules !== null && typedRules === savedRules
        && [...writing].some((write) => !write.open && write.text !== typedRules);
}

/** Weigh the newest typed words against what may be persisted: while a save in flight or unconfirmed could leave them
 *  unsaved they stay held, and the page restores them when it opens again — even words typed back to the saved text. */
function holdRules() {
    const open = typedRules !== null && (typedRules !== savedRules || [...writing].some((write) => write.text !== typedRules));
    unsavedRules = open ? typedRules : null;
    markDesktopEditor('Your rules', open);
}

/** A read showed `text` persisted: that is what is saved now. It proves no request ended — a save whose answer never
 *  came may still complete after the read and put its words back — so nothing is retired here; a save is settled by
 *  its own answer (`rulesSettled`) or given up by the owner (`releaseUnsettledRules`). */
function rulesRead(text) {
    savedRules = text;
    holdRules();
}

/** A save settled with `outcome`, the answer of the save the queue ran: one whose own text was replaced while it waited
 *  is answered by the one that replaced it, so its words never went out and it counts for nothing. Landed says the
 *  text was persisted at that moment; the server's own answer settles the save it is about — no other. A read alone
 *  (`outcome.byRead`, settings_controls.js) settles it only when nothing was sent for it (writeOnce's read before
 *  the send); a read after a write whose own answer was lost (`outcome.unanswered`) settles nothing, for that request
 *  was sent, may still land over newer words, and keeps counting until its own answer or the owner's give-up
 *  (`releaseUnsettledRules`). A read of what landed retires nothing (`rulesRead`), and an earlier save of other words
 *  may still land too. Refused, or changed elsewhere (the fresh read has what is persisted): nothing was written, so
 *  words typed back to the persisted text are saved after all. Unconfirmed: it may yet land, and keeps counting.
 *  Returns the save the answer is about. */
function rulesSettled(write, outcome) {
    const answered = typeof outcome.answers?.text === 'string' ? outcome.answers : write;
    const sent = answered.text;
    const { state } = outcome;
    write.open = false;
    if (answered !== write) writing.delete(write);
    if (state === 'saved' || state === 'saved_then_failed' || state === 'unchanged') {
        savedRules = sent;
        // The server's answer settles the save it is about, and only that one: another save of the same words,
        // answered later, is no proof that this one completed — it may still land after it and put these words back.
        // A read alone settles it only when nothing was sent for it: a save whose own answer was lost
        // (`outcome.unanswered`) was sent, may still land over newer words, and keeps counting until its own answer
        // or the owner's give-up.
        if (answered === write && !outcome.unanswered) writing.delete(write);
    } else if (state === 'refused' || state === 'changed_elsewhere') {
        writing.delete(write);
        if (state === 'changed_elsewhere') savedRules = settingsSnapshot() ? String(settingsSnapshot()[SAFETY_RULES_KEY] ?? '') : null;
    }
    holdRules();
    return answered;
}

/** The owner stops waiting for the saves whose answer never came: they accept that such a save may still land and
 *  replace what is saved. Nothing is sent, no words change and no request is claimed ended — the form only stops
 *  holding the owner's words against it, until they type new words or press Save again. */
function releaseUnsettledRules() {
    for (const write of writing) if (!write.open) writing.delete(write);
    holdRules();
}

/** What Save says: the page's own sentence once saved, else the row helper's words (a refusal, an unknown outcome,
 *  a failed after-step); `{ text, tone, details }`. */
export function rulesSaveWords(outcome, row = rowById(SAFETY_RULES_KEY)) {
    if (outcome?.state === 'saved' || outcome?.state === 'unchanged') {
        return { text: outcome.message ? `${RULES_SAVED}. ${outcome.message}` : RULES_SAVED, tone: outcome.message ? 'warn' : 'ok', details: '' };
    }
    const line = outcomeLine(row, outcome);
    return { text: line?.text || 'Not saved.', tone: line?.tone || 'danger', details: line?.details || '' };
}

/** Your rules, as a page: the text, Save and its result, and — while a save whose answer never came still counts —
 *  the line that says so, with the owner's way to stop waiting for it. `mount(container, { ctx })` of settings_pages.js. */
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
            <p class="settings-card-note" data-rules-unsettled hidden>An older save may still land and replace what is saved.
                <button type="button" class="consciousness-inline-link ui-hit-area" data-rules-give-up>Stop waiting for it</button></p>
        </section>`;
    const field = container.querySelector('#settings-rules');
    const button = container.querySelector('[data-editor-save]');
    const result = container.querySelector('[data-editor-result]');
    const standing = container.querySelector('[data-rules-unsettled]');
    let stored = null;  // the saved text as last read; null until a read answers
    let gone = false;
    let locked = false;
    // The standing line is not a save's result: it says a risk that outlives every result, and its link is the owner's
    // way to give this form's draft up. Both follow the words, so both are repainted wherever the result is.
    const paintStanding = () => { standing.hidden = !unsettledRules(); };
    const say = (words) => { setResult(result, words || {}); paintStanding(); };
    const unsaved = () => (unsavedRules === null ? null : NOT_SAVED);
    // Until a read answers the text is no value: not operable, and it says it is being read or could not be.
    const lock = () => {
        const words = stored === null ? settingsReadMessage() : '';
        if (!words && !locked) return;
        locked = Boolean(words);
        field.disabled = button.disabled = locked;
        say(locked ? { text: words } : unsaved());
    };
    const adopt = (settings) => {
        stored = String(settings?.[SAFETY_RULES_KEY] ?? '');
        container.querySelector('[data-rules-organization]').hidden = !deploymentOf(settings?._meta)?.organization_rules;
        // A page with nothing unsaved follows what is persisted; one holding words weighs them against it.
        if (unsavedRules === null && field.value !== stored) { field.value = stored; typedRules = null; }
        rulesRead(stored);
    };
    // Until a read answers, the text follows every read: one that lands is adopted, the others say what they found.
    const follow = () => { if (stored === null && settingsSnapshot()) adopt(settingsSnapshot()); lock(); };
    if (unsavedRules !== null) {
        field.value = unsavedRules;
        say(unsaved());
    }
    follow();  // the last read at once, else it says it is reading; this page's own read follows
    const offRead = paintWithRow(SAFETY_RULES_KEY, follow);
    async function read() {
        let settings;
        try { settings = await ctx.readSettings(); } catch { settings = settingsSnapshot(); }  // failed: the last read, if any
        if (gone) return;
        // A read a newer document outran (a save or a newer read landed meanwhile) says nothing of what is persisted
        // now: the page follows the newest document, never the older answer.
        if (settings && settings !== settingsSnapshot()) settings = settingsSnapshot();
        if (settings) adopt(settings);
        lock();
    }
    const onInput = () => {
        typedRules = field.value;
        holdRules();
        say(null);  // an edit starts a new subject: the last result goes
    };
    const onSave = async () => {
        if (button.disabled) return;
        const text = field.value;
        const write = { text, open: true };
        writing.add(write);  // from the press: words typed back to the saved text meanwhile are unsaved if this lands
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
        let outcome;
        try { outcome = await save(row, text, { ticket: write }); } catch { outcome = { state: 'unknown', message: '' }; }  // a save that threw is not confirmed
        const landed = outcome.state === 'saved' || outcome.state === 'saved_then_failed' || outcome.state === 'unchanged';
        // Whatever the owner typed meanwhile, here or in a page opened since, is weighed against how this ended: kept and
        // held when unsaved against it, so leaving and reopening shows their actual words, not the text this save carried.
        const answered = rulesSettled(write, outcome);
        if (landed) stored = answered.text;
        if (gone) return;
        button.disabled = false;
        button.removeAttribute('aria-busy');
        // A refusal leaves the text in the field: nothing here rewrites it. What landed is the save the queue ran, and the
        // line is about the words in the field now: other words (one it replaced, an edit since the press) are not saved
        // by it, and words an earlier save without an answer may still overwrite are not confirmed.
        const doubted = [...writing].some((other) => !other.open && other.text !== field.value);
        say(!landed ? rulesSaveWords(outcome, row) : field.value !== answered.text ? NOT_SAVED
            : doubted ? rulesSaveWords({ state: 'unknown' }, row) : rulesSaveWords(outcome, row));
    };
    const onShown = (event) => { if (event.detail?.path === 'rules') void read(); };
    standing.querySelector('[data-rules-give-up]').addEventListener('click', () => { releaseUnsettledRules(); say(null); });
    field.addEventListener('input', onInput);
    button.addEventListener('click', onSave);
    globalThis.window?.addEventListener?.('ouro:settings-page-shown', onShown);
    void read();
    return { unmount() {
        gone = true;
        offRead();
        field.removeEventListener('input', onInput);
        button.removeEventListener('click', onSave);
        globalThis.window?.removeEventListener?.('ouro:settings-page-shown', onShown);
    } };
}

/** Forget the typed words, what is known persisted and every save in flight, and release the window replacement (a test). */
export function forgetUnsavedRules() {
    unsavedRules = typedRules = savedRules = null;
    writing.clear();
    markDesktopEditor('Your rules', false);
}
