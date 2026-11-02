// Settings › Behavior › Safety Supervisor: two coverage levels over one checker — "Your work" (what the owner
// asked for) and "Consciousness" (what Ouroboros does on its own, places included) — the checker and second-check
// facts, and the owner's rules. Every value here is owner-only: the generic settings save never carries it. Save
// writes the checker's changed settings (from Models and Agents) in one owner transaction BEFORE the generic save,
// and each changed level and the rules through their audited owner endpoints after it (consciousness_api.js).
import { renderSegmentedField } from './page_header.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { apiClient } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { settingsWriteFailure, syncEffortSegments } from './settings_controls.js';
import { parseModelSource } from './route_editor_primitives.js';
import { modelRoleMap } from './model_roles.js';
import { subagentSettingsFingerprint } from './subagents_settings.js';
import {
    SAFETY_MODEL_KEYS, markRecheckRow, parseRoster, recheckFromSettings, saveCheckerSettings, saveSafetyLevel, saveSafetyRules,
} from './consciousness_api.js';
import { SAFETY_KINDS, SAFETY_LEVELS, SAFETY_RANK, listWords, safetyLevelOf } from './consciousness_words.js';
import { bindRecheckMarks, recheckDraft, recheckSaved, syncRecheckState } from './safety_recheck.js';

export const CONSCIOUSNESS_SAFETY_KEY = 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE';
export const SAFETY_RULES_KEY = 'OUROBOROS_SAFETY_OWNER_RULES';
export const SAFETY_MODEL_KEY = 'OUROBOROS_MODEL_SAFETY';

const LEVEL_OPTIONS = SAFETY_LEVELS.map(({ value, label }) => ({ value, label }));
const LOWERING_TITLE = 'Lowering a level asks for confirmation.';

/** The section body under its title (settings_ui.js keeps the `<h3>`). */
export function safetySectionHtml() {
    const notes = SAFETY_LEVELS.map((level) => `<br><code>${escapeHtml(level.label)}</code> &mdash; ${escapeHtml(level.note)}`).join('');
    return `
        <div class="settings-section-copy">
            A second model judges a step before it runs. Your work: what you asked for. Consciousness: what it does on its own, places included.${notes}
            <br><strong>Configuration authority:</strong> outside Cyber Pro, Ouroboros cannot lower its own checks; Cyber Pro also lets it configure Supervisor coverage. Changes apply on the next task; every step a lower level lets through is recorded.
        </div>
        <div class="safety-levels">
            <div class="settings-effort-card">
                <label>${escapeHtml(SAFETY_KINDS.your_work)}</label>
                <input id="s-safety-mode" type="hidden" value="full">
                ${renderSegmentedField({ target: 's-safety-mode', title: LOWERING_TITLE, options: LEVEL_OPTIONS })}
                <div class="settings-inline-note" data-policy-state="supervisor" role="status" aria-live="polite"></div>
            </div>
            <div class="settings-effort-card" data-safety-consciousness-row>
                <label>${escapeHtml(SAFETY_KINDS.consciousness)}</label>
                <input id="s-consciousness-safety-mode" type="hidden" value="full">
                ${renderSegmentedField({ target: 's-consciousness-safety-mode', title: LOWERING_TITLE, options: LEVEL_OPTIONS })}
                <div class="settings-inline-note" data-policy-state="supervisor_consciousness" role="status" aria-live="polite"></div>
                <p class="safety-line" data-safety-check-fully-line hidden><button type="button" class="consciousness-inline-link ui-hit-area" data-safety-check-fully>Check consciousness fully</button></p>
            </div>
        </div>
        <p class="safety-line" data-safety-checker></p>
        <p class="safety-line" data-safety-second-check></p>
        <p class="safety-line" data-safety-unmeasured hidden>Not measured yet: no replay of real checks exists.</p>
        <div class="form-field ui-field">
            <label for="s-safety-rules">Your rules</label>
            <textarea id="s-safety-rules" class="ui-control consciousness-owner-words" rows="3" spellcheck="false" aria-describedby="s-safety-rules-help"></textarea>
            <div class="settings-inline-note ui-field-help" id="s-safety-rules-help">Ouroboros and the Safety Supervisor read these in every task and check. Rules for one place live in that place's notes.</div>
        </div>
        <div id="s-safety-skip-counter" class="settings-inline-note"></div>`;
}

/** The one in-page link to another Settings tab («Change in Models»); the section's binder opens the tab. */
export const settingsLinkHtml = (tab, label) => `<button type="button" class="consciousness-inline-link ui-hit-area" data-settings-goto="${escapeHtml(tab)}">${escapeHtml(label)}</button>`;

const levelLabel = (value) => safetyLevelOf(value)?.label || String(value || 'Unknown');
const modelName = (value) => parseModelSource(value).model || String(value || '');

/** «Checked by: <the Safety Supervisor model, or the Light model when empty>». */
export function checkerText(settings = {}) {
    const own = String(settings[SAFETY_MODEL_KEY] ?? '').trim();
    return `Checked by: ${own ? modelName(own) : 'the Light model'}`;
}

/** The second-check line from the saved Agents roster (the marked row and the rows that may be marked). */
export function secondCheckText(recheck, settings = {}) {
    if (!recheck.rowId) {
        return recheck.known && !recheck.eligibleRows.length
            ? 'Second check: Off. No Agents row can re-check yet: it needs an API model row with Packet delivery.'
            : 'Second check: Off';
    }
    if (recheck.offReason === 'row_disabled') return 'Second check: Off (the marked row is disabled)';
    if (recheck.offReason === 'delegation_off') return 'Second check: Off (Delegation is off in Agents)';
    return `Second check: ${agentRowHandle(recheck.rowId, settings)} re-checks flagged actions`;
}

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

function paintLines(root, settings) {
    const meta = settings?._meta || {};
    const recheck = recheckFromSettings(settings);
    const set = (selector, html) => { const node = root.querySelector(selector); if (node) node.innerHTML = html; };
    set('[data-safety-checker]', `${escapeHtml(checkerText(settings))} · ${settingsLinkHtml('models', 'Change in Models')}`);
    set('[data-safety-second-check]', `${escapeHtml(secondCheckText(recheck, settings))}${recheck.known
        ? ` · ${settingsLinkHtml('agents', 'Change in Agents')}` : ''}`);
    // Said wherever the second check is offered, the default Off included: no replay has measured it.
    const unmeasured = root.querySelector('[data-safety-unmeasured]');
    if (unmeasured) unmeasured.hidden = !(recheck.known && !recheck.measured);
    syncRecheckState(recheck, root);
}

function syncCheckFully(root) {
    const line = root.querySelector('[data-safety-check-fully-line]');
    const input = root.querySelector('#s-consciousness-safety-mode');
    if (line && input) line.hidden = input.disabled || input.value === 'full';
}

/** Paint the section from a loaded settings document (values, facts, lines). */
export function applySafetySettings(root, settings = {}) {
    const yourWork = root.querySelector('#s-safety-mode');
    if (yourWork) yourWork.value = settings.OUROBOROS_SAFETY_MODE || 'full';
    const consciousness = root.querySelector('#s-consciousness-safety-mode');
    // A server that does not report the second level yet could only change "Your work" through it.
    const known = CONSCIOUSNESS_SAFETY_KEY in settings;
    if (consciousness) {
        consciousness.value = settings[CONSCIOUSNESS_SAFETY_KEY] || 'full';
        consciousness.disabled = !known;
        root.querySelectorAll('[data-effort-target="s-consciousness-safety-mode"] [data-effort-value]')
            .forEach((button) => { button.disabled = !known; });
        if (!known) {
            const note = root.querySelector('[data-policy-state="supervisor_consciousness"]');
            if (note) note.textContent = 'This server does not report a separate level for consciousness yet.';
        }
    }
    const rules = root.querySelector('#s-safety-rules');
    if (rules) {
        rules.value = String(settings[SAFETY_RULES_KEY] ?? '');
        rules.disabled = !(SAFETY_RULES_KEY in settings);
    }
    syncEffortSegments(root);
    paintLines(root, settings);
    syncCheckFully(root);
    void refreshSafetySkipCounter(root);
}

const MODEL_ROLE_MAPS = new Set(['OUROBOROS_MODEL_ACCOUNTS', 'OUROBOROS_MODEL_CONTEXT_WINDOWS', 'OUROBOROS_MODEL_PROCESSING_PREFERENCES']);
const sortedJson = (map) => JSON.stringify(Object.fromEntries(Object.entries(map).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))));

/** One checker value as Save compares it with the loaded document: a model string, a Local flag, a per-slot map. */
function checkerValueText(key, value) {
    if (key === 'OUROBOROS_SUBAGENTS') return subagentSettingsFingerprint(value);
    if (MODEL_ROLE_MAPS.has(key)) return sortedJson(modelRoleMap(value));
    if (key.startsWith('USE_LOCAL_')) return String(value === true || /^(true|1|yes|on)$/i.test(String(value ?? '').trim()));
    return String(value ?? '').trim();
}

/**
 * Take the checker's changes out of the generic body: every key of SAFETY_MODEL_KEYS whose draft differs from the
 * loaded document, the roster carrying the second-check mark. Outside Cyber Pro the generic save refuses any
 * change to the checker they resolve (the Safety Supervisor slot, else Light, else Main), so they travel in one
 * owner transaction instead; an unchanged key stays where it was and is never sent there. Returns
 * `{settings, saved, allowEmptyReviewPool}` (the keys to send, the sentence once saved, the owner's empty-pool
 * confirmation when the roster moved) or null.
 */
export function takeCheckerChanges(body, current = {}, mark = recheckDraft()) {
    const settings = {};
    for (const key of SAFETY_MODEL_KEYS) {
        if (key !== 'OUROBOROS_SUBAGENTS' && key in body
            && checkerValueText(key, body[key]) !== checkerValueText(key, current[key])) settings[key] = body[key];
    }
    const markChanged = mark !== null && mark !== recheckSaved();
    const roster = parseRoster(body.OUROBOROS_SUBAGENTS ?? (markChanged ? current.OUROBOROS_SUBAGENTS : null));
    const nextRoster = roster && markChanged ? markRecheckRow(roster, mark) : roster;
    let allowEmptyReviewPool = false;
    if (nextRoster && checkerValueText('OUROBOROS_SUBAGENTS', nextRoster) !== checkerValueText('OUROBOROS_SUBAGENTS', current.OUROBOROS_SUBAGENTS)) {
        settings.OUROBOROS_SUBAGENTS = nextRoster;
        // The roster's own confirmation (an empty review pool) travels with it to the owner transaction.
        allowEmptyReviewPool = body.allow_empty_review_pool === true;
        delete body.allow_empty_review_pool;
    }
    const keys = Object.keys(settings);
    if (!keys.length) return null;
    keys.forEach((key) => { delete body[key]; });
    const model = String(settings[SAFETY_MODEL_KEY] ?? '');
    const saved = [
        SAFETY_MODEL_KEY in settings ? (model ? `Safety Supervisor model saved as ${modelName(model)}.` : 'Safety Supervisor uses the Light model.') : '',
        markChanged && settings.OUROBOROS_SUBAGENTS ? (mark ? 'Second check saved.' : 'Second check turned off.') : '',
    ];
    return { settings, saved: saved.filter(Boolean).join(' '), allowEmptyReviewPool };
}

/**
 * The owner-only draft: both levels, the rules, and the checker's changes, which this takes OUT of the generic
 * body (`body`, as collected) so Save writes them first through their owner transaction (`saveSafetyChecker`).
 */
export function safetyOwnerDraft(root, body = {}, current = {}) {
    return {
        yourWork: root.querySelector('#s-safety-mode')?.value || 'full',
        consciousness: root.querySelector('#s-consciousness-safety-mode')?.value || 'full',
        rules: root.querySelector('#s-safety-rules')?.value ?? null,
        checker: takeCheckerChanges(body, current),
    };
}

/** Save's first write: the checker's changes in one owner transaction; a refusal throws the server's sentence. */
export async function saveSafetyChecker(draft) {
    if (!draft?.checker) return false;
    await saveCheckerSettings(draft.checker.settings, { allowEmptyReviewPool: draft.checker.allowEmptyReviewPool });
    return true;
}

/** «Models», «Agents» or «Models and Agents»: where the changes the checker transaction saved were made. */
export function checkerTabsText(draft) {
    const keys = Object.keys(draft?.checker?.settings || {});
    return listWords([keys.some((key) => key !== 'OUROBOROS_SUBAGENTS') ? 'Models' : '', keys.includes('OUROBOROS_SUBAGENTS') ? 'Agents' : '']);
}

/** The generic save's answer as the whole Save's effect: the checker's model, read live, applies at its next check;
 *  an Agents row (OUROBOROS_SUBAGENTS) applies from the next task. */
export function checkerSaveEffect(data, draft) {
    if (!draft?.checker) return data;
    const keys = Object.keys(draft.checker.settings || {});
    return { ...data, no_changes: false,
        ...(keys.some((key) => key !== 'OUROBOROS_SUBAGENTS') ? { immediate_changed: true } : {}),
        ...(keys.includes('OUROBOROS_SUBAGENTS') ? { next_task_changed: true } : {}) };
}

async function confirmLowering(kindLabel, current, next, askOwner) {
    const ok = await askOwner({
        title: 'Lower a safety level',
        body: `Lower the safety level for ${kindLabel.toLowerCase()} from ${levelLabel(current)} to ${levelLabel(next)}?\n\n`
            + 'Fixed rules still guard protected files at every level, and every step the lower level lets through is recorded.',
        confirmLabel: 'Lower the level',
        danger: true,
    });
    if (!ok) throw Object.assign(new Error('The change was not confirmed.'), { body: { saved: false } });
}

/**
 * After the generic save, write each changed level and the rules through their endpoints, in order: the two
 * levels (lowering asks first), then the rules. Returns what was saved (the checker's sentence, saved before the
 * generic save, leads), what failed and whether any outcome is unknown; one failure never stops the others.
 */
export async function saveSafetyOwnerChanges(draft, current = {}, { askOwner = openConfirmDialog } = {}) {
    const saved = draft.checker?.saved ? [draft.checker.saved] : [];
    const errors = [];
    let unknown = false;
    const run = async (label, step) => {
        try {
            const message = await step();
            if (message) saved.push(message);
        } catch (error) {
            const failure = settingsWriteFailure(error, label);
            errors.push(failure.text);
            unknown ||= failure.unknown;
        }
    };
    for (const [kind, next, key] of [['your_work', draft.yourWork, 'OUROBOROS_SAFETY_MODE'],
        ['consciousness', draft.consciousness, CONSCIOUSNESS_SAFETY_KEY]]) {
        if (kind === 'consciousness' && !(key in current)) continue;
        const now = current[key] || 'full';
        if (!next || next === now) continue;
        const kindLabel = SAFETY_KINDS[kind];
        await run(`Safety for ${kindLabel.toLowerCase()}`, async () => {
            if ((SAFETY_RANK[next] ?? 2) < (SAFETY_RANK[now] ?? 2)) await confirmLowering(kindLabel, now, next, askOwner);
            await saveSafetyLevel(next, kind);
            return `Safety for ${kindLabel.toLowerCase()} saved as ${levelLabel(next)}.`;
        });
    }
    if (draft.rules !== null && SAFETY_RULES_KEY in current && draft.rules !== String(current[SAFETY_RULES_KEY] ?? '')) {
        await run('Your rules', async () => { await saveSafetyRules(draft.rules); return 'Your rules saved.'; });
    }
    return { saved: saved.join(' '), error: errors.join(' '), unknown };
}

/** Waved-through checks among the inspected recent events (the Light/Off audit), counted within 24 h. */
export async function refreshSafetySkipCounter(root, { logsTail = apiClient.logsTail, now = Date.now() } = {}) {
    const el = root?.querySelector?.('#s-safety-skip-counter');
    if (!el) return;
    try {
        const data = await logsTail('events', 2000);
        const cutoff = now - 24 * 3600 * 1000;
        const n = (data?.entries || []).filter((entry) => {
            if (String(entry?.type || '') !== 'safety_mode_skip') return false;
            const time = Date.parse(String(entry?.ts || ''));
            return Number.isFinite(time) && time >= cutoff;
        }).length;
        // The tail may cover less than a day, including when it has no matches.
        el.textContent = n > 0
            ? `${n} safety check(s) waved through among inspected recent events dated within 24h.`
            : 'No waved-through safety checks found among inspected recent events dated within 24h.';
    } catch {
        el.textContent = '';
    }
}

/**
 * Bind the section once: "Check consciousness fully" raises only that row to Full at once through the owner
 * endpoint (raising never asks), the in-page links open the tab they name, and the Agents rows' second-check
 * marks keep their one-row rule.
 */
export function bindSafetySettings(root, { save = saveSafetyLevel } = {}) {
    const section = root.querySelector('[data-safety-consciousness-row]')?.closest('.form-section') || root;
    const onClick = async (event) => {
        const goto = event.target.closest?.('[data-settings-goto]');
        if (goto && section.contains(goto)) { root.activateSettingsTab?.(goto.dataset.settingsGoto); return; }
        if (event.target.closest?.('[data-effort-target="s-consciousness-safety-mode"]')) {
            queueMicrotask(() => syncCheckFully(root));
            return;
        }
        const button = event.target.closest?.('[data-safety-check-fully]');
        if (!button || button.disabled) return;
        const note = root.querySelector('[data-policy-state="supervisor_consciousness"]');
        button.disabled = true;
        try {
            await save('full', 'consciousness');
            const input = root.querySelector('#s-consciousness-safety-mode');
            if (input) input.value = 'full';
            syncEffortSegments(root);
            syncCheckFully(root);
            globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:settings-updated', { detail: { reason: 'consciousness checked fully', source: 'safety' } }));
        } catch (error) {
            if (note) note.textContent = settingsWriteFailure(error, 'Safety for consciousness').text;
        } finally {
            button.disabled = false;
        }
    };
    section.addEventListener('click', onClick);
    const disposeMarks = bindRecheckMarks(root);
    return () => { section.removeEventListener('click', onClick); disposeMarks(); };
}
