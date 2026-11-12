// The shared Models editor for Settings and onboarding. Model strings remain
// the routing authority; account and context choices belong to the exact role.
// Catalog arrival only enriches choices. It never authors an assignment.
import { apiClient, fetchJson } from './api_client.js';
import { SAFETY_MODEL_KEYS } from './consciousness_api.js';
import { API_PROVIDER_LABELS, harnessPresentation } from './harness_presentation.js';
import { controlIcon } from './page_icons.js';
import { MODEL_CATALOG_TIMEOUT_MS, catalogReadNote, mergeModelCatalog, refreshModelCatalog, watchAccountModelCatalog, withConnectionCatalogs } from './settings_catalog.js';
import { CODECS, CONFIRMS, bindSettingRows, commitSetting, doorSign, outcomeLine, setButtonBusy, settingsRow, webSearchPreviewWords } from './settings_controls.js';
import { registerSettingsPage } from './settings_pages.js';
import { HAND_WRITTEN, rowsOf } from './settings_rows.js';
import { setResult } from './ui_primitives.js';
import { accountRows, bindStatusSurface, claudexorStatus } from './claudexor_status_store.js';
import { parseModelSource, composeModelSource, apiConnections, configuredApiProviders, indexProfilesByHarness,
    profileOptionsFor, routeChoiceGroups, selectHtml, mintStableId, unavailableConnectionState, API_CHOICE_PREFIX,
    DEFAULT_API_PROVIDER, PROCESSING_PREFERENCE_KEY, MODEL_PROCESSING_PREFERENCES_KEY, PROCESSING_CHOICES,
    processingSelectHtml, processingIntentLabel, processingCapabilityNote, catalogModelOptions, priceWords, tierPriced,
    KEY_PROBLEM_WORDS, routeKeyProblem, keyProblemLine, apiProviderLabel } from './route_editor_primitives.js';
import { revealNewRow } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { catalogPrograms } from './model_program.js';
import { modelChooserHtml, bindModelChoosers, updateModelChooser } from './model_chooser.js';

export const MODEL_ACCOUNTS_KEY = 'OUROBOROS_MODEL_ACCOUNTS';
export const MODEL_CONTEXT_KEY = 'OUROBOROS_MODEL_CONTEXT_WINDOWS';

export { parseModelSource, composeModelSource } from './route_editor_primitives.js';

export function modelRoleMap(value) {
    if (typeof value === 'string') {
        try { value = JSON.parse(value); } catch (_) { return {}; }
    }
    return value && typeof value === 'object' && !Array.isArray(value) ? { ...value } : {};
}

/**
 * The row's stored source spelling (`openrouter`, `openai`, `subscription:x`,
 * `inherit`) as the shared select vocabulary. Mapping happens only here and in
 * `sourceFromChoice`, so the stored model strings never learn a new spelling.
 */
export function sourceChoice(source) {
    const raw = String(source || '');
    if (!raw || raw === 'inherit' || raw.startsWith('subscription:')) return raw;
    return `${API_CHOICE_PREFIX}${raw}`;
}

export function sourceFromChoice(choice) {
    const raw = String(choice || '');
    return raw.startsWith(API_CHOICE_PREFIX)
        ? raw.slice(API_CHOICE_PREFIX.length) || DEFAULT_API_PROVIDER : raw;
}

/**
 * The Models source select. One vocabulary with Agents: the same groups, the
 * same order, the same configured-only API list; only the "Uses Main" entry and
 * the absent agent-session group are specific to a model role.
 * @param {{sources?: Array, providers?: Array<{id:string,label?:string}>, connections?: Array,
 *   current?: string, catalogKnown?: boolean, accountsKnown?: boolean, providerProfiles?: object}} args
 *   `providers` is `configuredApiProviders()` output, `connections` `apiConnections()` output,
 *   `current` a stored row source.
 */
export function modelSourceGroups({ sources = [], providers = [], connections = [], current = '', inheritLabel = 'Uses Main',
    catalogKnown = false, accountsKnown = false, providerProfiles = {}, hasConfiguredAccounts = false, offerInherit = false } = {}) {
    return [
        ...(current === 'inherit' || offerInherit ? [{ options: [{ value: 'inherit', label: inheritLabel }] }] : []),
        ...routeChoiceGroups({ modelSources: sources, providers, connections, providerProfiles,
            currentChoice: current === 'inherit' ? '' : sourceChoice(current),
            catalogKnown, accountsKnown, hasConfiguredAccounts, includeSessions: false }),
    ];
}

// A window source as the owner says it: where the value comes from, and who says another value.
const WINDOW_SOURCES = { user_setting: ['set in Settings', 'Settings say'], owner_ack: ['your confirmation', 'your confirmation says'],
    provider_metadata: ['from the server', 'the server says'], local_health: ['the local serving window', 'the local server says'] };
const tokenWords = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : String(n));

/** The route's window in owner words, from the projection's typed fields (never catalogue arithmetic): the window,
 *  where it comes from, the usable input, a source that disagrees and an old confirmation no longer used; the
 *  observation stamps stay in the projection's record. */
export function modelContextNote(preview) {
    const window = preview?.window;
    if (!window?.tokens) return 'Context limit not known for this route.';
    const usable = Number.isFinite(window.usable) ? window.usable : window.tokens - (window.reserve || 0);
    const other = window.contradiction;
    const retired = Number(window.retired_ack?.window_tokens) || 0;
    return `Window ${tokenWords(window.tokens)} tokens (${WINDOW_SOURCES[window.source]?.[0] || window.source}) · ${tokenWords(usable)} usable for input`
        + (other ? ` · ${WINDOW_SOURCES[other.source]?.[1] || `${other.source} says`} ${tokenWords(other.tokens)}` : '')
        + (retired ? ` · old confirmation ${tokenWords(retired)} not used` : '');
}

// The slot an empty role uses: Main, except the Safety Supervisor, which checks with the Light model (and so
// with Main when Light is empty too). A contract row may name its own (`inherits`).
const INHERITS = Object.freeze({ safety: 'light' });
export const inheritTargetOf = (slot) => String(slot?.inherits || INHERITS[slot?.slot] || 'main');

export function modelRolesHost(id) {
    return `<div id="${escapeHtml(id)}" class="model-role-editor"></div>`;
}

/** Settings shows this under Main, before Save: what a new Main does to work already running (owner, 2026-11-03),
 *  in one sentence (owner 04.11, 1A: work with its own model is the work's own fact). */
export const MAIN_FOLLOW_NOTE = 'Running work that follows Main switches at its next call.';

/** In-memory form controller; refreshes have no settings or login side effects. `mainNote`: Settings' Main note.
 *  `only` (`{ slot, index }`) draws one role alone, as a form of labelled fields (a role's page in Settings): its
 *  source may always return to the role it follows, and a fallback past the saved ones is a new, unsaved one. */
export function createModelRolesEditor({ hostId, store = claudexorStatus,
    doc = () => document, onChange = () => {}, catalogFetch = fetchJson, showContext = true, mainNote = false, only = null } = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    let slots = [];
    let offeredSlots = [];  // every slot in contract order; a `managed` one (a deployment's checker) is shown, never a row
    let rows = [];
    let settings = {};
    let providerProfiles = {};
    let apiProviders = [];
    let connections = [];
    let mainChoice = [];  // `_meta.main_connection_choice`: an unauthored Main waits for the owner's key
    let catalog = { items: [], model_sources: [], read_state: 'not_read' };
    let loaded = false;
    let destroyed = false;
    let disposeStatus = null;
    let disposeChoosers = () => {};
    let validationAttempted = false;
    let processingPreference = '', processingTouched = false;
    const catalogs = new Map();
    const requests = new Map();
    const responseEdits = new WeakSet();
    const host = () => getDoc()?.getElementById(hostId);

    function rowState(slot, value, index = -1) {
        const account = modelRoleMap(settings[MODEL_ACCOUNTS_KEY])[slot.slot];
        const context = modelRoleMap(settings[MODEL_CONTEXT_KEY])[slot.slot];
        const processing = modelRoleMap(settings[MODEL_PROCESSING_PREFERENCES_KEY])[slot.slot];
        const parsed = parseModelSource(value);
        if (!parsed.model && !['main', 'fallback'].includes(slot.slot)) parsed.source = 'inherit';
        return { id: index < 0 ? slot.slot : mintStableId('fallback', rows.map((row) => row.id)), slot,
            ...parsed, account: String(index < 0 ? account || '' : account?.[index] || ''),
            context: index < 0 ? context || 0 : context?.[index] || 0,
            processing_preference: String(index < 0 ? processing || '' : processing?.[index] || ''),
            local: settings[`USE_LOCAL_${slot.slot.toUpperCase()}`] === true
                || settings[`USE_LOCAL_${slot.slot.toUpperCase()}`] === 'true' };
    }

    function collect() {
        if (!loaded) return {};
        const result = {};
        const accounts = modelRoleMap(settings[MODEL_ACCOUNTS_KEY]);
        const windows = modelRoleMap(settings[MODEL_CONTEXT_KEY]);
        const processing = modelRoleMap(settings[MODEL_PROCESSING_PREFERENCES_KEY]);
        for (const slot of slots) {
            const matching = rows.filter((row) => row.slot.slot === slot.slot);
            const fallback = slot.slot === 'fallback';
            const values = matching.map((row) => composeModelSource(row.source, row.model));
            result[slot.settingKey] = fallback ? values.join(', ') : values[0] || '';
            const pins = matching.map((row) => sourceId(row) ? row.account : '');
            const contexts = matching.map((row) => Number(row.context || 0));
            const preferences = matching.map((row) => row.processing_preference);
            if (pins.some(Boolean) || slot.slot in accounts) accounts[slot.slot] = fallback ? pins : pins[0] || '';
            if (contexts.some(Boolean) || slot.slot in windows) windows[slot.slot] = fallback ? contexts : contexts[0] || 0;
            if (preferences.some(Boolean) || slot.slot in processing) processing[slot.slot] = fallback ? preferences : preferences[0] || '';
            if (slot.settingsToggleId) result[`USE_LOCAL_${slot.slot.toUpperCase()}`] = matching[0]?.local || false;
        }
        if (Object.keys(accounts).length) result[MODEL_ACCOUNTS_KEY] = accounts;
        if (Object.keys(windows).length) result[MODEL_CONTEXT_KEY] = windows;
        // Wait pickers use showContext:false and author model/account only.
        if (showContext && Object.keys(processing).length) result[MODEL_PROCESSING_PREFERENCES_KEY] = processing;
        if (showContext && (processingTouched || PROCESSING_PREFERENCE_KEY in settings)) result[PROCESSING_PREFERENCE_KEY] = processingPreference;
        return result;
    }

    function changed(slot = '') { onChange(collect(), { slot }); }
    function rowErrors(row) {
        const errors = [];
        if (row.processing_preference && !PROCESSING_CHOICES.includes(row.processing_preference)) errors.push({ field: '[data-model-role-processing]', message: `${row.slot.label}: choose Standard, Fast, Economy or inherited processing.` });
        if (['main', 'fallback'].includes(row.slot.slot) && !row.model.trim()) {
            errors.push({ field: '[data-model-role-model]', message: `${row.slot.label}: choose a model${row.slot.slot === 'fallback' ? ' or remove this fallback' : ''}.` });
        }
        const contextInput = getDoc()?.getElementById(`${inputIdFor(row)}-context`);
        if (contextInput?.validity?.badInput || !Number.isSafeInteger(Number(row.context || 0)) || Number(row.context || 0) < 0) {
            errors.push({ field: '[data-model-role-context]', message: `${row.slot.label}: context window must be a positive whole number, or Auto.` });
        }
        return errors;
    }
    function validateAll() { return [
        ...(processingPreference && !PROCESSING_CHOICES.includes(processingPreference) ? ['Choose Standard, Fast, Economy or route defaults for Processing.'] : []),
        ...rows.flatMap((row) => rowErrors(row).map(({ message }) => message)),
    ]; }
    // The row an empty role resolves to, along its inherit chain (safety → light → main).
    function inheritedRow(row) {
        let current = row;
        for (let step = 0; step < 4 && current?.source === 'inherit'; step += 1) {
            const target = inheritTargetOf(current.slot);
            current = rows.find((entry) => entry.slot.slot === target && entry !== current)
                || rows.find((entry) => entry.slot.slot === 'main');
        }
        return current;
    }
    const inheritName = (row) => {
        const target = slots.find((slot) => slot.slot === inheritTargetOf(row.slot));
        return String(target?.label || 'Main').replace(/ Model$/, '');
    };
    const emptyUsesText = (row) => (inheritTargetOf(row.slot) === 'main' ? 'Empty uses Main' : `Empty uses the ${inheritName(row)} model`);
    function effectiveSource(row) { return row.source === 'inherit' ? inheritedRow(row)?.source || 'openrouter' : row.source; }
    function responseModel(row) {
        const selected = row.source === 'inherit' ? inheritedRow(row) : row;
        return selected ? composeModelSource(selected.source, selected.model) : '';
    }
    function sourceId(row) { const source = effectiveSource(row); return source.startsWith('subscription:') ? source.slice(13) : ''; }
    function catalogKey(row) { return JSON.stringify([sourceId(row), row.account]); }
    // A saved source whose key is gone stays selectable and labelled; discovery
    // only widens the list, it never rewrites the owner's assignment.
    function sourceGroupsFor(row, facts = {}) {
        return modelSourceGroups({ sources: catalog.model_sources, providers: apiProviders, connections, providerProfiles, current: row.source,
            inheritLabel: `Uses ${inheritName(row)}`, offerInherit: Boolean(only) && !['main', 'fallback'].includes(row.slot.slot), ...facts });
    }
    // A saved API source that cannot send says why under its row; the select keeps it chosen.
    function keyNote(row) {
        if (row.local || !row.model || row.source === 'inherit' || sourceId(row)
            || apiProviders.some((provider) => provider.id === row.source)) return '';
        return choiceNote(row) || KEY_PROBLEM_WORDS[unavailableConnectionState(row.source, connections)];
    }
    // Several keys of Main's provider and no built-in one: the owner picks Main's, nothing is guessed (#2969).
    function choiceNote(row) {
        const keys = row.slot.slot === 'main' ? mainChoice.map((id) => connections.find((entry) => entry.id === id)?.label || id) : [];
        return keys.length > 1 ? `Main is not set yet. Choose the key it uses: ${keys.slice(0, -1).join(', ')} or ${keys.at(-1)}.` : '';
    }
    const onMaintenanceSettled = () => {
        for (const row of new Map(rows.map(row => [catalogKey(row), row])).values()) void refreshRow(row, {force: true});
    };
    function itemsFor(row) {
        return sourceId(row) ? (catalogs.get(catalogKey(row))?.items || [])
            : catalog.items.filter((item) => parseModelSource(item.value || item.id).source === row.source);
    }
    // A fallback with no tier of its own keeps its caller's (model_slots.task_processing_preference): Main's or a lead's.
    const mainTier = () => (rows.find((entry) => entry.slot.slot === 'main')?.processing_preference
        ?? modelRoleMap(settings[MODEL_PROCESSING_PREFERENCES_KEY]).main) || processingPreference;
    const tieredProcessing = (row) => tierPriced(row.processing_preference || processingPreference)
        || (row.slot.slot === 'fallback' && !row.processing_preference && tierPriced(mainTier()));
    function currentItem(row) {
        const model = row.source === 'inherit' ? inheritedRow(row)?.model : row.model;
        // An account pins a subscription's rows; an API source's rows name none, so a stale pin never hides them.
        const matches = itemsFor(row).filter((item) => (parseModelSource(item.value || item.id).model === model
            || item.id === model) && (!row.account || !sourceId(row) || item.credential_profile_id === row.account));
        return matches.length === 1 ? matches[0] : undefined;
    }

    function inputIdFor(row) {
        return row.slot.slot === 'fallback' && rows.find((entry) => entry.slot === row.slot) !== row
            ? `${hostId}-${row.id}` : row.slot.inputId;
    }

    function detailsHtml(row) {
        if (!showContext) return '';
        return `<details class="model-role-details ui-disclosure"><summary><span>${only ? 'More: processing, window, longest answer' : 'Context &amp; processing'} · <span data-processing-summary>${escapeHtml(processingIntentLabel(row.processing_preference, processingPreference))}</span></span></summary>
            <label class="ui-field">Processing ${processingSelectHtml(`data-model-role-processing aria-label="${escapeHtml(row.slot.label)} processing"`, row.processing_preference)}</label>
            <div class="ui-field-help">Uses the same model and reasoning effort. An explicit native service choice takes precedence.</div>
            <div class="ui-field-help" data-processing-capability></div>
            <div class="model-role-context">
                <label class="ui-field">Window <input id="${escapeHtml(inputIdFor(row))}-context" class="ui-control" data-model-role-context type="number" min="0" step="1"
                    placeholder="Auto" value="${escapeHtml(row.context || '')}" aria-label="${escapeHtml(row.slot.label)} context window">
                    <span class="ui-field-help" data-model-context-note>${escapeHtml(modelContextNote(row.windowPreview))}</span></label>
                <button type="button" class="btn btn-default" data-window-clear hidden>Clear confirmation</button>
            </div>
            <div class="model-role-context">
                <label class="ui-field">Maximum response <input class="ui-control" data-response-limit type="number" min="0" step="1" placeholder="Auto" aria-label="${escapeHtml(row.slot.label)} maximum response">
                    <span class="ui-field-help" data-response-note>Auto. Applies to this exact model and server; blank restores metadata.</span></label>
                <button type="button" class="btn btn-default" data-response-apply>Apply maximum</button>
            </div></details>`;
    }

    // A role the deployment fixes: its model said in its place, no control to edit and nothing collected.
    const fixedGroupHtml = (slot) => `<section class="model-role-group" data-model-role-group="${escapeHtml(slot.slot)}" data-model-role-fixed>
        <div class="model-role-head"><h4>${escapeHtml(slot.label.replace(/ Model$/, ''))}</h4><span class="ui-chip skills-chip" data-tone="neutral">Set by your organization</span></div>
        <p class="model-role-copy">${escapeHtml(parseModelSource(settings[slot.settingKey]).model || String(settings[slot.settingKey] || ''))}. Its administrator changes it.</p>
    </section>`;

    function rowHtml(row, index, total) {
        const isFallback = row.slot.slot === 'fallback';
        const inputId = inputIdFor(row);
        const described = mainNote && row.slot.slot === 'main' ? ` aria-describedby="${escapeHtml(hostId)}-main-note"` : '';
        // Alone on its page a role is a form: each control under its own word.
        const field = (label, control) => (only ? `<div class="ui-field model-role-field"><span>${label}</span>${control}</div>` : control);
        const source = selectHtml(`data-model-role-source aria-label="${escapeHtml(row.slot.label)} source"`, sourceGroupsFor(row), sourceChoice(row.source));
        const model = modelChooserHtml(`id="${escapeHtml(inputId)}" data-model-role-model aria-label="${escapeHtml(row.slot.label)}${isFallback ? ` ${index + 1}` : ''}"${described}`, row.model, `${hostId}-${row.id}-models`, [], { placeholder: row.slot.slot === 'main' ? 'Choose a model' : emptyUsesText(row) });
        const account = `<select class="ui-control" data-model-role-account aria-label="${escapeHtml(row.slot.label)} account" ${sourceId(row) ? '' : 'hidden'}></select>`;
        const local = only && row.slot.settingsToggleId ? `<label class="local-toggle ui-field ui-field-inline"><input class="ui-checkbox" id="${escapeHtml(row.slot.settingsToggleId)}" type="checkbox" data-model-local ${row.local ? 'checked' : ''}> Use the local model for this role</label>` : '';
        return `<div class="model-role-row" data-model-role="${escapeHtml(row.id)}"${only ? ' data-solo' : ''}>
            <div class="model-role-controls">
                ${field('Source', source)}${field('Model', model)}${only ? `<div class="ui-field model-role-field" data-model-role-account-field ${sourceId(row) ? '' : 'hidden'}><span>Account</span>${account}</div>` : account}
                ${isFallback && !only ? `<span class="model-role-order"><button type="button" class="btn btn-default" data-model-up aria-label="Move fallback up" ${index === 0 ? 'disabled' : ''}>↑</button><button type="button" class="btn btn-default" data-model-down aria-label="Move fallback down" ${index === total - 1 ? 'disabled' : ''}>↓</button><button type="button" class="btn btn-default" data-model-remove aria-label="Remove fallback">Remove</button></span>` : ''}
            </div>${local}
            <div class="model-role-notes"><span id="${escapeHtml(hostId)}-${escapeHtml(row.id)}-status" class="model-role-meta ui-field-help" data-model-role-status></span>${only ? '' : detailsHtml(row)}</div>
            <div class="ui-status ui-field-help" id="${escapeHtml(hostId)}-${escapeHtml(row.id)}-error" data-model-role-error data-tone="error" hidden></div>
            ${only ? `${mainNote && row.slot.slot === 'main' ? `<p class="model-role-copy" id="${escapeHtml(hostId)}-main-note" data-main-follow-note>${escapeHtml(MAIN_FOLLOW_NOTE)}</p>` : ''}${only.actions || ''}${detailsHtml(row)}` : ''}
        </div>`;
    }

    // The one row a role's page draws: the slot's own, or the fallback at `only.index`.
    const soleRow = () => rows.filter((row) => row.slot.slot === only.slot)[only.index || 0] || null;

    function render() {
        const element = host();
        if (!element || destroyed || !loaded) return;
        disposeChoosers();
        if (only) {
            const slot = offeredSlots.find((entry) => entry.slot === only.slot);
            const row = slot && !slot.managed ? soleRow() : null;
            element.innerHTML = slot?.managed ? fixedGroupHtml(slot) : row ? rowHtml(row, only.index || 0, 1) : '';
        } else element.innerHTML = (showContext ? `<section class="model-role-group"><label class="ui-field">Processing
            ${processingSelectHtml('data-global-processing aria-label="Global processing"', processingPreference, { global: true })}</label>
            <p class="model-role-copy">Applies to new tasks. Fast permits accelerated paid service within existing limits. Economy may fall back to the ordinary rate. Model and reasoning effort stay the same.</p>
            <details class="model-role-details ui-disclosure"><summary>Details</summary><div class="ui-field-help">Keep route defaults preserves existing native choices. Explicit native service choices take precedence; the requested mode is not proof of the mode served.</div></details></section>` : '') + offeredSlots.map((slot) => {
            if (slot.managed) return fixedGroupHtml(slot);
            const matching = rows.filter((row) => row.slot.slot === slot.slot);
            const local = matching[0]?.local || false;
            return `<section class="model-role-group" data-model-role-group="${escapeHtml(slot.slot)}">
                <div class="model-role-head"><h4 title="${escapeHtml(slot.note || '')}">${escapeHtml(slot.label.replace(/ Model$/, ''))}</h4>
                    ${slot.settingsToggleId ? `<label class="local-toggle ui-field ui-field-inline"><input class="ui-checkbox" id="${escapeHtml(slot.settingsToggleId)}" type="checkbox" data-model-local aria-label="${escapeHtml(slot.label)} local runtime" ${local ? 'checked' : ''}> Local</label>` : ''}
                    ${slot.slot === 'fallback' ? '<button type="button" class="btn btn-default" data-model-add>Add fallback</button>' : ''}
                </div>
                ${slot.slot === 'fallback' ? '<p class="model-role-copy">Tried in this order when the acting model refuses, cannot be reached or loses its answer. A wait starts only when none of them answers.</p>' : ''}
                ${matching.map((row, index) => rowHtml(row, index, matching.length)).join('')}
                ${mainNote && slot.slot === 'main' ? `<p class="model-role-copy" id="${escapeHtml(hostId)}-main-note" data-main-follow-note>${escapeHtml(MAIN_FOLLOW_NOTE)}</p>` : ''}
            </section>`;
        }).join('');
        element.querySelector('[data-global-processing]')?.addEventListener('change', (event) => {
            processingPreference = event.target.value; processingTouched = true; changed(); updateCatalogViews();
        });
        bindRows(element);
        disposeChoosers = bindModelChoosers(element);
        updateCatalogViews();
        for (const row of only ? [soleRow()].filter(Boolean) : rows) void refreshRow(row);
    }

    function updateCatalogViews() {
        const element = host();
        if (!element || destroyed) return;
        const profiles = indexProfilesByHarness(store.snapshot);
        for (const row of rows) {
            const node = element.querySelector(`[data-model-role="${row.id}"]`);
            if (!node) continue;
            const responseIdentity = JSON.stringify([responseModel(row), row.local, row.account]);
            const windowIdentity = JSON.stringify([responseIdentity, row.context]);
            if (row.windowIdentity !== windowIdentity) {
                row.windowPreview = null;
                if (showContext) void responseLimit(row, node);
                row.windowIdentity = windowIdentity;
            }
            if (row.responseIdentity && row.responseIdentity !== responseIdentity) {
                const field = node.querySelector('[data-response-limit]');
                if (field) { field.value = ''; responseEdits.delete(field); }
                const note = node.querySelector('[data-response-note]');
                if (note) note.textContent = 'Route changed; open details to read its maximum, or apply a new value.';
            }
            row.responseIdentity = responseIdentity;
            const source = node.querySelector('[data-model-role-source]');
        const sourceHtml = selectHtml('', sourceGroupsFor(row, {
                catalogKnown: catalog.sources_read_state === 'ok', accountsKnown: store.accountsKnown,
                hasConfiguredAccounts: accountRows(store.snapshot || {}).length > 0,
            }), sourceChoice(row.source));
            const options = sourceHtml.slice(sourceHtml.indexOf('>') + 1, sourceHtml.lastIndexOf('</select>'));
            if (source.innerHTML !== options) source.innerHTML = options;
            const account = node.querySelector('[data-model-role-account]');
            account.hidden = !sourceId(row);
            const accountField = node.querySelector('[data-model-role-account-field]');
            if (accountField) accountField.hidden = account.hidden;
            const credentialHarness = catalog.model_sources.find((entry) => entry.id === sourceId(row))?.credentialHarness || '';
            const accountHtml = selectHtml('', [{ options: profileOptionsFor(profiles[credentialHarness], row.account, { accountsKnown: store.accountsKnown && Boolean(credentialHarness), labelled: Boolean(only) }) }], row.account);
            const accountOptions = accountHtml.slice(accountHtml.indexOf('>') + 1, accountHtml.lastIndexOf('</select>'));
            if (account.innerHTML !== accountOptions) account.innerHTML = accountOptions;
            updateModelChooser(node.querySelector('[data-model-role-model]'), catalogModelOptions(itemsFor(row).map((item) => ({
                ...item, value: parseModelSource(item.value || item.id).model, ...(tieredProcessing(row) ? { price: null } : {}),
            }))), row.local ? [] : catalogPrograms(catalogs.get(catalogKey(row)), {source: sourceId(row), profile: row.account, sources: catalog.model_sources}));
            if (showContext) node.querySelector('[data-model-context-note]').textContent = modelContextNote(row.windowPreview);
            if (showContext) node.querySelector('[data-processing-summary]').textContent = processingIntentLabel(row.processing_preference, processingPreference);
            if (showContext) node.querySelector('[data-processing-capability]').textContent = processingCapabilityNote(row.processing_preference || processingPreference, currentItem(row)?.processing,
                catalog.model_sources.find((source) => source.id === sourceId(row))?.processingPreferences);
            const status = node.querySelector('[data-model-role-status]');
            const rowCatalog = catalogs.get(catalogKey(row));
            // A subscription row reports its own account read in full; an API row
            // sits under the section banner, so it keeps to the compact form. An
            // API row whose own catalog (OpenRouter's) states a price says it.
            const own = Boolean(sourceId(row) && rowCatalog);
            const note = catalogReadNote(own ? rowCatalog : catalog, { compact: !own });
            // A model stored with its ` (local)` label runs on the local lane, as the runtime reads it (slot_lane_label).
            const local = row.local || / \(local\)$/.test(String(row.model || ''));
            const choosing = !local && Boolean(choiceNote(row)) && keyNote(row) === choiceNote(row);
            status.classList.toggle('ui-status', choosing);
            if (choosing) status.dataset.tone = 'warn'; else delete status.dataset.tone;
            const price = local || keyNote(row) || note || !row.model || sourceId(row) || tieredProcessing(row) ? ''
                : priceWords(currentItem(row)?.price, { unit: true });
            status.textContent = local ? 'Uses the local runtime.'
                : keyNote(row) || note || (!row.model ? (row.slot.slot === 'main' ? 'Choose a model to continue.' : `Uses ${inheritName(row)}.`)
                    : sourceId(row) ? (rowCatalog?.read_state === 'ok' && !rowCatalog.items.length
                        ? 'No models were listed for this source. Your selection is kept.'
                        : 'Uses your subscription. No API key required.') : price || 'Price unknown');
            // The price is already composed through the translation seam: the overlay leaves it alone and reads the rest.
            if (price) status.dataset.i18nFmt = 'model price'; else delete status.dataset.i18nFmt;
            const errors = validationAttempted ? rowErrors(row) : [];
            const message = node.querySelector('[data-model-role-error]');
            Object.assign(message, { textContent: errors.map((error) => error.message).join(' '), hidden: !errors.length });
            for (const selector of ['[data-model-role-model]', '[data-model-role-context]']) {
                const field = node.querySelector(selector);
                field?.setAttribute('aria-describedby', `${status.id} ${message.id}`);
                field?.setAttribute('aria-invalid', String(errors.some((error) => error.field === selector)));
            }
        }
    }

    async function refreshRow(row, { force = false } = {}) {
        if (!sourceId(row) || destroyed) return;
        const key = catalogKey(row);
        if (requests.has(key)) {
            if (force) requests.get(key).again = true;
            return requests.get(key).promise;
        }
        if (!force && catalogs.has(key)) return;
        const query = new URLSearchParams({ source_id: sourceId(row) });
        if (row.account) query.set('credential_profile_id', row.account);
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), MODEL_CATALOG_TIMEOUT_MS);
        const promise = Promise.resolve().then(() => catalogFetch(`/api/model-catalog?${query}`, { cache: 'no-store', signal: controller.signal }))
            .then((data) => { if (!destroyed) {
                if (!Array.isArray(data?.items)) throw new Error(data?.error || 'Model catalog response has no model list');
                catalogs.set(key, mergeModelCatalog(catalogs.get(key), data));
                if (Array.isArray(data.model_sources)) catalog.model_sources = mergeModelCatalog(catalog, data).model_sources;
            } })
            .catch((error) => { if (!destroyed) catalogs.set(key, mergeModelCatalog(catalogs.get(key), {
                read_state: 'transport', errors: [{ error: error.message || String(error) }],
            })); })
            .finally(() => {
                clearTimeout(timer); const again = requests.get(key)?.again; requests.delete(key); updateCatalogViews();
                if (again && !destroyed) void refreshRow(row, {force: true});
            });
        requests.set(key, { promise, controller });
        return promise;
    }

    async function responseLimit(row, node, apply = false) {
        // A queued disclosure preview must not supersede an explicit Apply.
        if (!apply && row.responseApplying) return;
        const field = node.querySelector('[data-response-limit]');
        const note = node.querySelector('[data-response-note]');
        const value = Number(field.value || 0);
        if (apply && (field.validity.badInput || !Number.isSafeInteger(value) || value < 0)) {
            note.textContent = 'Maximum response must be a positive whole number, or Auto.';
            field.setAttribute('aria-invalid', 'true'); return;
        }
        const model = responseModel(row);
        if (!model || model.endsWith('::')) { note.textContent = 'Choose a model to check its maximum response.'; return; }
        const identity = JSON.stringify([model, row.local, row.account, row.context]);
        const draft = field.value;
        const sequence = row.responseSequence = (row.responseSequence || 0) + 1;
        if (apply) row.responseApplying = sequence;
        const current = () => !destroyed && sequence === row.responseSequence && node.isConnected
            && identity === JSON.stringify([responseModel(row), row.local, row.account, row.context]);
        let saved = false;
        try {
            const data = await apiClient.responseLimitPreview({ model, local: row.local, account: row.account,
                role: row.slot.slot, window: Number(row.context || 0) });
            if (!current()) return;
            if (!data.response_limit || !data.route) throw new Error('Maximum response is not checked');
            row.windowPreview = data;
            node.querySelector('[data-model-context-note]').textContent = modelContextNote(data);
            node.querySelector('[data-window-clear]').hidden = !data.window_ack;
            let evidence = data.response_limit;
            if (apply) {
                const result = await apiClient.ownerCapabilityAck({ ...data.route, route_fp: evidence.route_fp, max_output_tokens: value });
                if (!current()) return;
                if (!result.ok) throw new Error(result.error || 'Maximum response could not be saved');
                evidence = result.ack;
                saved = true;
            }
            // Edits can precede the deferred toggle which starts this read.
            if (!apply && !responseEdits.has(field) && field.value === draft) field.value = evidence.source === 'owner_ack' ? String(evidence.max_output_tokens || '') : '';
            field.setAttribute('aria-invalid', 'false');
            const source = evidence.source === 'owner_ack' ? 'your maximum' : 'model metadata';
            const endpoint = data.route.base_url ? new URL(data.route.base_url) : null;
            const server = endpoint ? ` Server: ${endpoint.host}${endpoint.pathname.replace(/\/$/, '')}.` : '';
            note.textContent = `${evidence.max_output_tokens && !evidence.stale ? `${evidence.max_output_tokens.toLocaleString('en-US')} tokens · ${source}` : 'Auto: maximum response unknown'}.${server} Context window is separate.${data.route.provider === 'claudexor' ? ' Planning allowance only; the native engine controls its response.' : ''}`;
        } catch (error) { if (current()) note.textContent = error.message; }
        finally {
            if (row.responseApplying === sequence) row.responseApplying = 0;
            if (saved && current()) void responseLimit(row, node);
        }
    }

    function bindRows(element) {
        for (const row of rows) {
            const node = element.querySelector(`[data-model-role="${row.id}"]`);
            if (!node) continue;  // a role's page draws one row
            node.querySelector('[data-model-local]')?.addEventListener('change', (event) => {
                rows.filter((entry) => entry.slot === row.slot).forEach((entry) => { entry.local = event.target.checked; });
                changed(row.slot.slot); updateCatalogViews();
            });
            node.querySelector('[data-model-role-source]').addEventListener('change', (event) => {
                row.source = sourceFromChoice(event.target.value); row.model = ''; row.account = ''; row.context = 0;
                changed(row.slot.slot); render();
                host()?.querySelector(`[data-model-role="${row.id}"] [data-model-role-model]`)?.focus();
            });
            node.querySelector('.model-role-details')?.addEventListener('toggle', (event) => {
                if (event.target.open) void responseLimit(row, node);
            });
            node.querySelector('[data-window-clear]')?.addEventListener('click', async () => {
                const preview = row.windowPreview;
                if (!preview?.window_route) return;
                try {
                    const result = await apiClient.ownerCapabilityAck({ ...preview.window_route, window_tokens: 0 });
                    if (!result.ok) throw new Error(result.error || 'Confirmation could not be cleared');
                    if (!destroyed && node.isConnected) await responseLimit(row, node);
                } catch (error) {
                    if (!destroyed && node.isConnected) node.querySelector('[data-model-context-note]').textContent = error.message;
                }
            });
            node.querySelector('[data-response-apply]')?.addEventListener('click', () => { void responseLimit(row, node, true); });
            node.querySelector('[data-response-limit]')?.addEventListener('input', (event) => { responseEdits.add(event.target); });
            const input = getDoc().getElementById(inputIdFor(row));
            input.addEventListener('input', () => {
                row.model = input.value;
                const cap = node.querySelector('[data-response-limit]');
                if (cap) { cap.value = ''; responseEdits.delete(cap); }
                const capNote = node.querySelector('[data-response-note]');
                if (capNote) capNote.textContent = 'Route changed; open details to read its maximum, or apply a new value.';
                if (row.source === 'inherit' && row.model) row.source = effectiveSource(row);
                if (!row.model && !['main', 'fallback'].includes(row.slot.slot)) row.source = 'inherit';
                if (row.model.includes('::')) Object.assign(row, parseModelSource(row.model));
                changed(row.slot.slot); updateCatalogViews();
                for (const item of rows) void refreshRow(item);
            });
            node.querySelector('[data-model-role-account]').addEventListener('change', (event) => {
                row.account = event.target.value; changed(row.slot.slot); updateCatalogViews(); void refreshRow(row);
            });
            node.querySelector('[data-model-role-context]')?.addEventListener('input', (event) => {
                row.context = event.target.value; changed(row.slot.slot); updateCatalogViews();
            });
            node.querySelector('[data-model-role-processing]')?.addEventListener('change', (event) => {
                row.processing_preference = event.target.value; changed(row.slot.slot); updateCatalogViews();
            });
            for (const [selector, delta] of [['[data-model-up]', -1], ['[data-model-down]', 1]]) {
                node.querySelector(selector)?.addEventListener('click', () => {
                    const index = rows.indexOf(row);
                    [rows[index], rows[index + delta]] = [rows[index + delta], rows[index]];
                    changed(); render();
                    host()?.querySelector(`[data-model-role="${row.id}"] ${selector}`)?.focus();
                });
            }
            node.querySelector('[data-model-remove]')?.addEventListener('click', () => {
                const siblings = rows.filter((entry) => entry.slot === row.slot);
                const next = siblings[siblings.indexOf(row) + 1] || siblings[siblings.indexOf(row) - 1];
                rows = rows.filter((entry) => entry !== row); changed(); render();
                // The caret moves to the fallback now in this place, else to Add (DESIGN "List editors").
                const shown = next && host()?.querySelector(`[data-model-role="${next.id}"]`);
                if (shown) revealNewRow(shown, shown.querySelector('[data-model-role-model]'));
                else host()?.querySelector('[data-model-add]')?.focus();
            });
        }
        element.querySelectorAll('[data-model-role-group]').forEach((group) => {
            const slot = slots.find((entry) => entry.slot === group.dataset.modelRoleGroup);
            group.querySelector('[data-model-local]')?.addEventListener('change', (event) => {
                rows.filter((row) => row.slot === slot).forEach((row) => { row.local = event.target.checked; });
                changed(); updateCatalogViews();
            });
            group.querySelector('[data-model-add]')?.addEventListener('click', () => {
                const row = rowState(slot, '', rows.filter((entry) => entry.slot === slot).length);
                row.account = ''; row.context = 0; row.processing_preference = '';
                rows.push(row); changed(); render();
                const added = host()?.querySelector(`[data-model-role="${row.id}"]`);
                revealNewRow(added, added?.querySelector('[data-model-role-model]'));
            });
        });
    }

    return {
        // `keep`: a role's page holding typed fields keeps that role as typed; the rest follows the read (keys, Main, processing).
        load(value, contract = {}, { keep = false } = {}) {
            const kept = keep && only && loaded ? soleRow() : null;
            validationAttempted = false;
            settings = { ...value }; offeredSlots = contract.modelSlots || offeredSlots;
            slots = offeredSlots.filter((slot) => !slot.managed);
            processingPreference = String(settings[PROCESSING_PREFERENCE_KEY] || ''); processingTouched = false;
            providerProfiles = contract.providerProfiles || providerProfiles;
            connections = apiConnections(settings, providerProfiles);
            mainChoice = Array.isArray(settings._meta?.main_connection_choice) ? settings._meta.main_connection_choice : [];
            apiProviders = configuredApiProviders(settings, providerProfiles);
            rows = [];
            for (const slot of slots) {
                if (slot.slot === 'fallback') {
                    String(settings[slot.settingKey] || '').split(',').map((entry) => entry.trim()).filter(Boolean)
                        .forEach((entry, index) => rows.push(rowState(slot, entry, index)));
                } else rows.push(rowState(slot, settings[slot.settingKey]));
            }
            // A role's page past the saved fallbacks holds a new one, unsaved until its Save.
            const fallback = only?.slot === 'fallback' && slots.find((slot) => slot.slot === 'fallback');
            if (fallback && !soleRow()) rows.push(Object.assign(rowState(fallback, '', rows.length), { account: '', context: 0, processing_preference: '' }));
            const at = kept ? rows.indexOf(soleRow()) : -1;
            if (at >= 0) rows[at] = Object.assign(kept, { slot: rows[at].slot });
            loaded = true; render();
        },
        mount() {
            getDoc()?.addEventListener?.('harness-maintenance:settled', onMaintenanceSettled);
            // The prices are composed words the overlay leaves alone: a language change says them again.
            globalThis.window?.addEventListener?.('ouro:language-changed', updateCatalogViews);
            if (!disposeStatus) disposeStatus = bindStatusSurface(store, { elementId: hostId, doc: getDoc,
                listener: updateCatalogViews });
            render();
        },
        adoptCatalog(data = {}, { connectionCatalogs } = {}) {
            // The wizard supplies its already-merged general snapshot plus unsaved server reads.
            catalog = connectionCatalogs ? withConnectionCatalogs(data, connectionCatalogs) : mergeModelCatalog(catalog, data);
            if (Array.isArray(data.items)) {
                for (const row of rows) void refreshRow(row, { force: true });
            }
            updateCatalogViews();
        },
        collect,
        validateAll,
        /** The one role of `only` as its Save writes it, its refusals (said under it), and the slot a deployment fixes. */
        entry() {
            const row = only && loaded ? soleRow() : null;
            return row && { value: composeModelSource(row.source, row.model), account: sourceId(row) ? row.account : '',
                context: Number(row.context || 0), processing: row.processing_preference, local: Boolean(row.local),
                errors: rowErrors(row).map(({ message }) => message), slot: row.slot };
        },
        noteSaveAttempt() { validationAttempted = true; updateCatalogViews(); },
        validate() {
            return validateAll()[0] || '';
        },
        destroy() {
            getDoc()?.removeEventListener?.('harness-maintenance:settled', onMaintenanceSettled);
            globalThis.window?.removeEventListener?.('ouro:language-changed', updateCatalogViews);
            destroyed = true; disposeStatus?.(); disposeStatus = null;
            disposeChoosers();
            for (const request of requests.values()) request.controller.abort();
        },
    };
}

/* ---- Settings › Models (docs/DESIGN.md "Settings") ----------------------------------------------------------------
   The list (`models`): the Main model, the fallbacks in their order, a line per other role with the route it uses
   now, then the rows around them and the fold. A role's page (`models/role/<slot>`, `models/role/fallback/<n>`): its
   form with its own Save. The words of a role are its hand-written row (settings_rows.js). A Save changes the role's
   own keys on a fresh read: the checker's keys (consciousness_api.js SAFETY_MODEL_KEYS) go through the owner's checker
   transaction and the rest through the generic save, each with the tokens of that read. */

const MAP_KEYS = [MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_KEY, MODEL_PROCESSING_PREFERENCES_KEY];
const MAP_FIELDS = { [MODEL_ACCOUNTS_KEY]: ['account', ''], [MODEL_CONTEXT_KEY]: ['context', 0], [MODEL_PROCESSING_PREFERENCES_KEY]: ['processing', ''] };
const FALLBACKS_KEY = 'OUROBOROS_MODEL_FALLBACKS';
const ROLE_ORDER = ['light', 'vision', 'consciousness', 'safety'];
const truthy = (value) => value === true || /^(true|1|yes|on)$/i.test(String(value ?? '').trim());
// A key's value as one text: what is sent, and what a read is compared with when an answer never came.
const canon = (key, value) => JSON.stringify(key.startsWith('USE_LOCAL_') ? truthy(value) : MAP_KEYS.includes(key) ? modelRoleMap(value) : String(value ?? ''));
CODECS.modelKeys = {
    read: (settings, row) => (settings ? row.keys.map((key) => canon(key, settings[key])) : undefined),
    body: (value, row) => Object.fromEntries(row.keys.map((key, i) => [key, JSON.parse(value[i])])),
};
const landed = (outcome) => outcome.state === 'saved' || outcome.state === 'saved_then_failed';

/** Write the keys of `body` that differ from the read `fresh`: the checker's first, then the rest. Answers the
 *  Outcome of the last write; `partly` marks a second write that did not land after the first did. */
export async function writeModelKeys(id, body, fresh, { ask } = {}) {
    const changed = Object.keys(body).filter((key) => canon(key, body[key]) !== canon(key, fresh[key]));
    const parts = [['checker', changed.filter((key) => SAFETY_MODEL_KEYS.includes(key))], ['generic', changed.filter((key) => !SAFETY_MODEL_KEYS.includes(key))]];
    let last = null;
    for (const [writer, keys] of parts) {
        if (!keys.length) continue;
        const tokens = Object.fromEntries(keys.map((key) => [key, fresh._meta?.tokens?.[key]]).filter(([, token]) => token));
        const outcome = await commitSetting({ id: `${id}#${writer}`, keys, writer, codec: 'modelKeys', control: { kind: 'custom' } },
            keys.map((key) => canon(key, body[key])), { ifUnchanged: Object.keys(tokens).length ? tokens : '', ask });
        if (!landed(outcome)) return last ? { ...outcome, partly: true } : outcome;
        last = outcome;
    }
    return last || { state: 'unchanged', effect: 'now', withheld: '', message: '', code: '' };
}

// What a Save says (PINS section 5, an editor's line): when it applies, a refusal, another device's change.
function saveWords(outcome, saved = 'Saved') {
    if (outcome.state === 'unchanged') return { text: 'Nothing to save.', tone: 'muted' };
    const line = outcomeLine({ id: 'models', keys: [], control: { kind: 'custom' } }, { ...outcome, old: undefined, withheld: 'structure' });
    const words = line ? { text: line.note ? `${line.text}. ${line.note}` : line.text, tone: line.tone, details: line.details } : { text: 'Saved', tone: 'ok' };
    if (outcome.partly) return { ...words, text: `Partly saved. ${words.text}`, tone: 'danger' };
    return landed(outcome) && saved !== 'Saved' ? { ...words, text: words.text.replace(/^Saved/, saved) } : words;
}

/** The fallbacks of a read, each with its entries of the per-role maps. */
export function fallbackEntries(settings) {
    const maps = MAP_KEYS.map((key) => modelRoleMap(settings[key]).fallback);
    return String(settings[FALLBACKS_KEY] || '').split(',').map((value) => value.trim()).filter(Boolean).map((value, index) => ({
        value, ...Object.fromEntries(MAP_KEYS.map((key, i) => [MAP_FIELDS[key][0], (Array.isArray(maps[i]) ? maps[i][index] : '') || MAP_FIELDS[key][1]])),
    }));
}

/** The keys that hold `list` as the fallbacks: the order, and the fallback entry of each per-role map that has one. */
export function fallbackBody(fresh, list) {
    const body = { [FALLBACKS_KEY]: list.map((entry) => entry.value).join(', ') };
    for (const key of MAP_KEYS) {
        const map = modelRoleMap(fresh[key]);
        const values = list.map((entry) => entry[MAP_FIELDS[key][0]] || MAP_FIELDS[key][1]);
        if (values.some(Boolean) || 'fallback' in map) body[key] = { ...map, fallback: values };
    }
    return body;
}

/** One role's keys with its form's values (`entry`: the editor's), every other role's entry as the read `fresh` has
 *  it. A fallback is the one at `index`; null when the fallbacks are no longer the ones the form was read from. */
export function roleBody(fresh, entry, index = 0, readFallbacks = '') {
    const slot = entry.slot.slot;
    const body = entry.slot.settingsToggleId ? { [`USE_LOCAL_${slot.toUpperCase()}`]: entry.local } : {};
    if (slot === 'fallback') {
        const list = fallbackEntries(fresh);
        if (index > list.length || list.map((item) => item.value).join(',') !== readFallbacks) return null;
        list[index] = entry;
        return { ...body, ...fallbackBody(fresh, list) };
    }
    body[entry.slot.settingKey] = entry.value;
    for (const key of MAP_KEYS) {
        const map = modelRoleMap(fresh[key]);
        const value = entry[MAP_FIELDS[key][0]] || MAP_FIELDS[key][1];
        if (value || slot in map) body[key] = { ...map, [slot]: value };
    }
    return body;
}

// The slot wait the concurrency row names: a finite value, below 0 shown as 0 because
// `model_concurrency._max_slot_wait_sec` counts it so, else the 180 s default. Only the sentence follows it.
export function slotWaitSeconds(saved) {
    const wait = Number.parseFloat(saved);
    return Number.isFinite(wait) ? Math.max(0, wait) : 180;
}

const slotsOf = (settings) => (Array.isArray(settings?._meta?.setup_contract?.modelSlots) ? settings._meta.setup_contract.modelSlots : []);
const wordsOf = (id) => HAND_WRITTEN.find((row) => row.id === id) || { id, keys: [], name: id, meta: '' };
const shortModel = (model) => String(model || '').slice(String(model || '').lastIndexOf('/') + 1);

/** A stored route in the owner's words: `{ name, source, mark }`; an empty one follows another role (`follows`). */
export function routeWords(value, { local = false, account = '', follows = 'Main' } = {}) {
    const { source, model } = parseModelSource(value);
    if (local) return { name: 'Local model', source: 'On this computer', mark: '' };
    if (!String(value || '').trim()) return { name: `${follows} model`, source: '', mark: '', follows: true };
    const harness = source.startsWith('subscription:') ? source.slice(13) : '';
    const kind = source.split('@')[0];
    return { name: shortModel(model), mark: harness, source: harness ? `${harnessPresentation(harness).label} subscription · ${account ? `account ${account}` : 'your accounts in turn'}`
        : `${API_PROVIDER_LABELS[kind] || kind}${source.includes('@') ? ` (${source.split('@')[1]})` : ''} · by API key` };
}

// A subscription's own mark; any other route takes the plain one.
function markHtml(harness) {
    const mark = harness ? harnessPresentation(harness) : null;
    return mark ? `<svg viewBox="${mark.viewBox}" width="1em" height="1em" fill="currentColor" aria-hidden="true" focusable="false"><path d="${mark.path}"></path></svg>` : controlIcon('globe', 18);
}
const CONTEXT_WORDS = { nano: 'Nano', low: 'Low', max: 'Max' };
const SEARCH_SOURCES = [['auto', 'Auto'], ['openai', 'OpenAI'], ['openrouter', 'OpenRouter'], ['anthropic', 'Anthropic'], ['ddgs', 'DuckDuckGo']];

function mountModels(container, { path, ctx }) {
    const custom = (row) => `<button type="button" class="btn btn-default btn-sm" id="btn-refresh-model-catalog" data-models-action>${escapeHtml(row.control.label)}</button>`;
    const doors = [...ROLE_ORDER, 'websearch'].map((slot) => ({ ...wordsOf(`models:${slot}`), control: { kind: 'door', to: `models/role/${slot}` } }));
    const door = (row) => settingsRow(row).replace('data-setting-door', `data-setting-door data-entry-id="${row.id.split(':')[1]}"`);
    const plain = rowsOf('models', { fold: false }).map((row) => settingsRow(row, { custom })).join('');
    container.innerHTML = `<p class="settings-page-lead">Which model does which work.</p>
        <section class="ui-card settings-card" data-setting-row="models:main">
            <div class="models-main" data-entry-id="main"></div>
            <p class="settings-card-note">${escapeHtml(wordsOf('models:main').meta)} ${escapeHtml(MAIN_FOLLOW_NOTE)}</p>
        </section>
        <section class="ui-card settings-card" data-setting-row="models:fallbacks">
            <div class="settings-card-head"><h3 class="settings-card-title">If Main cannot answer</h3>
                <span class="ui-result" data-entry-result role="status" aria-live="polite"></span>
                <div class="ui-section-actions"><button type="button" class="btn btn-default btn-sm" data-fallback-add>Add fallback</button></div></div>
            <p class="settings-card-lead">${escapeHtml(wordsOf('models:fallbacks').meta)}</p>
            <div class="settings-rows" data-models-fallbacks></div>
        </section>
        <section class="ui-card settings-card"><h3 class="settings-card-title">Other work</h3>
            <p class="settings-card-lead">Each uses Main until you give it a model of its own.</p>
            <div class="settings-rows">${doors.map(door).join('')}</div></section>
        <section class="ui-card settings-card"><div class="settings-rows">${plain}</div></section>
        <section class="ui-card settings-card"><details class="ui-disclosure" data-settings-fold>
            <summary>More: processing, prompt cache, requests at once</summary>
            <div class="settings-rows" data-models-fold></div></details></section>`;
    // The catalogue's own result (settings_catalog.js writes it by this id) stands in its row's meta line.
    container.querySelector('[data-setting-row="models:catalogue"]')?.querySelector('.ui-row-meta')
        ?.appendChild(Object.assign(container.ownerDocument.createElement('span'), { id: 'settings-model-catalog-status', className: 'ui-result settings-row-result' }));
    const bound = bindSettingRows(container, { rows: doors, ask: ctx.ask, custom, openDoor: (row) => ctx.openSettings(row.control.to) });
    const foldHost = container.querySelector('[data-models-fold]');
    const entryResult = container.querySelector('[data-entry-result]');
    let fold = null;
    let shown = '', drawnFallbacks = '';   // the fallbacks as drawn: a press acts on these, never on a list that changed meanwhile
    // A row's value at its end, before its chevron: made once, then only its words change.
    const say = (id, text, follows = false) => {
        const node = container.querySelector(`[data-setting-row="${id}"]`);
        let span = node?.querySelector('[data-row-value]');
        if (node && !span) {
            span = Object.assign(container.ownerDocument.createElement('span'), { className: 'settings-row-value' });
            span.setAttribute('data-row-value', '');
            const box = node.querySelector('.ui-row-actions');
            box.insertBefore(span, box.firstChild);
        }
        if (span && span.textContent !== text) span.textContent = text;
        if (span && follows) span.setAttribute('data-follows', ''); else span?.removeAttribute('data-follows');
    };
    // A role whose key cannot send says so in its line, in place of its description, until it can again: the door's
    // own words, so the row painter paints them; the tone is the line's mark.
    const plainMeta = new Map(doors.map((row) => [row.id, row.meta]));
    const warn = (id, text) => {
        const door = doors.find((row) => row.id === id);
        if (door) door.meta = text || plainMeta.get(id);
        const meta = container.querySelector(`[data-setting-row="${id}"]`)?.querySelector('[data-setting-meta]');
        if (text) meta?.setAttribute('data-tone', 'warn'); else meta?.removeAttribute('data-tone');
    };
    const paint = (settings) => {
        const slots = Object.fromEntries(slotsOf(settings).map((slot) => [slot.slot, slot]));
        const accounts = modelRoleMap(settings[MODEL_ACCOUNTS_KEY]);
        const local = (slot) => Boolean(slots[slot]?.settingsToggleId) && truthy(settings[`USE_LOCAL_${slot.toUpperCase()}`]);
        const route = (slot, follows) => routeWords(settings[slots[slot]?.settingKey], { follows, account: accounts[slot], local: local(slot) });
        const profiles = settings._meta?.setup_contract?.providerProfiles || {};
        const keyWords = (value, onLocal) => {
            const problem = onLocal ? '' : routeKeyProblem(value, settings, profiles);
            return problem ? keyProblemLine(problem, apiProviderLabel(parseModelSource(value).source, profiles, apiConnections(settings, profiles))) : '';
        };
        // Several keys of Main's provider and no built-in one: Main waits for the owner's choice (#2969), said here too.
        const choice = Array.isArray(settings._meta?.main_connection_choice) ? settings._meta.main_connection_choice : [];
        const labels = choice.length > 1 ? choice.map((id) => apiConnections(settings, profiles).find((entry) => entry.id === id)?.label || id) : [];
        const main = route('main');
        const mainNote = labels.length ? `Choose the key it uses: ${labels.slice(0, -1).join(', ')} or ${labels.at(-1)}.`
            : keyWords(settings[slots.main?.settingKey], local('main'));
        container.querySelector('.models-main').innerHTML = `<span class="settings-head-glyph models-main-glyph" aria-hidden="true">${markHtml(main.mark)}</span>
            <div class="models-main-words"><span class="models-kicker">${escapeHtml(wordsOf('models:main').name)}</span>
                <span class="models-main-name">${escapeHtml(main.follows || labels.length ? 'Not chosen yet' : main.name)}</span><span class="models-main-meta"${mainNote ? ' data-tone="warn"' : ''}>${escapeHtml(mainNote || main.source)}</span></div>
            <button type="button" class="btn btn-default" data-entry-open>Change</button>`;
        const list = fallbackEntries(settings);
        shown = list.map((entry) => entry.value).join(',');
        const localFallback = truthy(settings.USE_LOCAL_FALLBACK);
        const fallbackHtml = list.map((entry, index) => {
            const words = routeWords(entry.value, { account: entry.account, local: localFallback });
            const note = keyWords(entry.value, localFallback);
            return `<div class="ui-row settings-head" data-entry-id="fallback/${index + 1}" data-actions>
                <span class="settings-head-glyph settings-head-order" aria-hidden="true">${index + 1}</span>
                <button type="button" class="settings-head-open ui-hit-area" data-entry-open><span class="ui-row-title">${escapeHtml(words.name)}</span><span class="ui-row-meta"${note ? ' data-tone="warn"' : ''}>${escapeHtml(note || words.source)}</span></button>
                <div class="ui-row-actions"><button type="button" class="btn btn-ghost btn-sm" data-fallback-move="-1" aria-label="Move fallback ${index + 1} up" ${index ? '' : 'disabled'}>↑</button>
                    <button type="button" class="btn btn-ghost btn-sm" data-fallback-move="1" aria-label="Move fallback ${index + 1} down" ${index < list.length - 1 ? '' : 'disabled'}>↓</button>
                    <button type="button" class="btn btn-ghost btn-sm" data-fallback-remove>Remove</button>${doorSign()}</div></div>`;
        }).join('') || '<p class="settings-card-lead">No fallback yet.</p>';
        if (fallbackHtml !== drawnFallbacks) container.querySelector('[data-models-fallbacks]').innerHTML = drawnFallbacks = fallbackHtml;
        for (const slot of ROLE_ORDER) {
            const words = route(slot, slot === 'safety' ? 'Light' : 'Main');
            say(`models:${slot}`, words.name, words.follows);
            warn(`models:${slot}`, keyWords(settings[slots[slot]?.settingKey], local(slot)));
        }
        say('models:websearch', Object.fromEntries(SEARCH_SOURCES)[settings.OUROBOROS_WEBSEARCH_BACKEND || 'auto'] || String(settings.OUROBOROS_WEBSEARCH_BACKEND));
        say('models:context', CONTEXT_WORDS[String(settings.OUROBOROS_CONTEXT_MODE || '').toLowerCase()] || String(settings.OUROBOROS_CONTEXT_MODE || ''));
        // The fold's rows are drawn once their values are read (the wait a request over the limit takes is the install's).
        if (!fold) {
            const rows = rowsOf('models', { fold: true }).map((row) => (row.id !== 'OUROBOROS_MODEL_MAX_CONCURRENCY' ? row
                : { ...row, more: row.more.replace('{wait}', String(slotWaitSeconds(settings.OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC))) }));
            foldHost.innerHTML = rows.map((row) => settingsRow(row)).join('');
            fold = bindSettingRows(foldHost, { rows, ask: ctx.ask });
        }
        fold.paint();
        bound.paint();
    };
    const read = async () => { try { paint(await ctx.readSettings()); } catch { /* the page keeps what it shows */ } };
    // The fallbacks as one write: `change` answers the next list from the fresh one, or null to leave it.
    const writeFallbacks = async (button, change, saved) => {
        setResult(entryResult, {});
        setButtonBusy(button, true);
        let outcome = { state: 'changed_elsewhere', effect: 'now', withheld: '', message: '', code: '' };
        try {
            const fresh = await ctx.readSettings();
            const list = fallbackEntries(fresh);
            if (list.map((entry) => entry.value).join(',') === shown) outcome = await writeModelKeys('models:fallbacks', fallbackBody(fresh, change(list)), fresh, { ask: ctx.ask });
        } catch (error) { outcome = { state: 'refused', effect: 'now', withheld: '', message: error?.message || String(error), code: '' }; }
        await read();
        setButtonBusy(button, false);
        setResult(entryResult, saveWords(outcome, saved));
    };
    const onClick = async (event) => {
        const hit = (selector) => event.target.closest?.(selector);
        if (hit('[data-models-action]')) { void refreshModelCatalog({ button: hit('[data-models-action]') }); return; }
        if (hit('[data-fallback-add]')) { ctx.openSettings(`models/role/fallback/${(shown ? shown.split(',').length : 0) + 1}`); return; }
        const head = hit('[data-entry-id]');
        const index = head?.dataset.entryId.startsWith('fallback/') ? Number(head.dataset.entryId.split('/')[1]) - 1 : -1;
        const move = hit('[data-fallback-move]');
        if (move) {
            const to = index + Number(move.dataset.fallbackMove);
            await writeFallbacks(move, (list) => { [list[index], list[to]] = [list[to], list[index]]; return list; }, 'Order saved');
            container.querySelector(`[data-entry-id="fallback/${to + 1}"] [data-fallback-move="${move.dataset.fallbackMove}"]:not([disabled])`)?.focus?.();
        } else if (hit('[data-fallback-remove]')) {
            const name = `fallback ${index + 1}, ${head.querySelector('.ui-row-title')?.textContent || ''}`;
            if (!(await CONFIRMS.removeEntry({ name }, null, null, ctx.ask))) return;
            await writeFallbacks(hit('[data-fallback-remove]'), (list) => list.filter((_, i) => i !== index), 'Removed');
            (container.querySelector(`[data-entry-id="fallback/${index + 1}"] [data-entry-open]`)
                || [...container.querySelectorAll('[data-entry-id^="fallback/"] [data-entry-open]')].at(-1)
                || container.querySelector('[data-fallback-add]'))?.focus?.();
        } else if (head && !hit('[data-setting-door]')) ctx.openSettings(`models/role/${head.dataset.entryId}`);
    };
    container.addEventListener('click', onClick);
    const onShown = (event) => { if (event.detail?.path === path) { void read(); void refreshModelCatalog(); } };
    const onSaved = (event) => { if (event.detail?.source === 'settings-row') void read(); };
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    globalThis.window?.addEventListener('ouro:settings-updated', onSaved);
    void read();
    return { unmount() {
        container.removeEventListener('click', onClick);
        globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown);
        globalThis.window?.removeEventListener('ouro:settings-updated', onSaved);
        bound.dispose(); fold?.dispose();
    } };
}

const SAVE_ROW = `<div class="ui-action-row"><button type="button" class="btn btn-save" data-editor-save>Save</button>
    <button type="button" class="btn btn-default" data-editor-discard hidden>Discard changes</button>
    <span class="ui-result" data-editor-result role="status" aria-live="polite"></span></div>`;
let hosts = 0;

/** A role's page: the shared editor drawing that one role, read when the page is shown and whenever it is shown again
 *  without unsaved text; its Save writes the role's keys on a fresh read. */
function mountRole(container, { path, ctx }, slot, index) {
    const hostId = `settings-model-role-${++hosts}`;
    container.innerHTML = `<p class="settings-page-lead">${escapeHtml(wordsOf(`models:${slot === 'fallback' ? 'fallbacks' : slot}`).meta)}</p>
        <section class="ui-card settings-card models-role">${modelRolesHost(hostId)}</section>`;
    let dirty = false;
    let readFallbacks = '';
    const editor = createModelRolesEditor({ hostId, mainNote: true, only: { slot, index, actions: SAVE_ROW },
        onChange: () => { dirty = true; container.querySelector('[data-editor-discard]')?.removeAttribute('hidden'); setResult(container.querySelector('[data-editor-result]'), {}); } });
    const say = (words) => setResult(container.querySelector('[data-editor-result]'), words);
    // A typed form keeps its fields while the page around them follows the read (a key added in Accounts, a new Main);
    // the fallback list a Save checks stays the one the typing began from. Nothing is drawn while the owner is in it.
    const typing = () => dirty && container.contains(globalThis.document?.activeElement);
    let asRead = null;  // the role as read when the typing began: a Save writes only what the owner changed from it
    const read = async () => {
        if (typing()) return;
        let settings;
        try { settings = await ctx.readSettings(); } catch { return; }
        if (typing()) return;
        if (!dirty) readFallbacks = fallbackEntries(settings).map((entry) => entry.value).join(',');
        editor.load(settings, { ...settings._meta?.setup_contract, modelSlots: slotsOf(settings).map((each) => ({ ...each, inputId: each.settingsInputId })) }, { keep: dirty });
        if (dirty) container.querySelector('[data-editor-discard]')?.removeAttribute('hidden');  // drawn again, still unsaved
        else asRead = editor.entry();
    };
    const save = async (button) => {
        editor.noteSaveAttempt();
        const entry = editor.entry();
        if (!entry || entry.errors.length) { say({ text: entry ? entry.errors[0] : 'This role is not shown yet.', tone: 'danger' }); return; }
        setButtonBusy(button, true);
        let outcome = { state: 'changed_elsewhere', effect: 'now', withheld: '', message: '', code: '' };
        try {
            const fresh = await ctx.readSettings();
            // A key the typing left as it was read keeps what is stored now: a Main chosen in the composer meanwhile stays.
            const mine = roleBody(fresh, entry, index, readFallbacks);
            const before = mine && asRead ? roleBody(fresh, asRead, index, readFallbacks) : null;
            const body = mine && Object.fromEntries(Object.entries(mine).filter(([key, value]) => !before || canon(key, value) !== canon(key, before[key])));
            if (body) outcome = await writeModelKeys(`models:${slot}`, body, fresh, { ask: ctx.ask });
        } catch (error) { outcome = { state: 'refused', effect: 'now', withheld: '', message: error?.message || String(error), code: '' }; }
        if (landed(outcome) || outcome.state === 'changed_elsewhere' || outcome.state === 'unchanged') { dirty = false; await read(); }
        setButtonBusy(button, false);
        say(saveWords(outcome));
    };
    const onClick = (event) => {
        const hit = (selector) => event.target.closest?.(selector);
        if (hit('[data-editor-save]')) void save(hit('[data-editor-save]'));
        else if (hit('[data-editor-discard]')) { dirty = false; void read(); }
    };
    const onCatalog = (event) => editor.adoptCatalog(event.detail);
    // The catalogue this role's model list comes from, read after the settings: when the page opens and whenever it is
    // shown again (a return, a reconnect). Opening and the shell's "shown" for it are one read.
    const accountCatalog = watchAccountModelCatalog();
    let catching = null;
    const catchUp = () => (catching ||= read().then(() => { accountCatalog.arm(); return refreshModelCatalog(); }).finally(() => { catching = null; }));
    const onShown = (event) => { if (event.detail?.path === path) void catchUp(); };
    container.addEventListener('click', onClick);
    globalThis.document?.addEventListener('settings-model-catalog:updated', onCatalog);
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    editor.mount();
    void catchUp();
    return { unsaved: () => dirty, unmount() {
        container.removeEventListener('click', onClick);
        globalThis.document?.removeEventListener('settings-model-catalog:updated', onCatalog);
        globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown);
        accountCatalog.dispose();
        editor.destroy();
    } };
}

/** Web search's page: its source and model with their own Save, and the sources a search would try, in order. */
function mountWebSearch(container, { path, ctx }) {
    container.innerHTML = `<p class="settings-page-lead">${escapeHtml(wordsOf('models:websearch').meta)}</p>
        <section class="ui-card settings-card models-role"><div class="models-form">
            <label class="ui-field"><span>Source</span><select id="s-websearch-source" class="ui-control">${SEARCH_SOURCES.map(([value, label]) => `<option value="${value}">${label}</option>`).join('')}</select></label>
            <label class="ui-field"><span>Model</span><input id="s-websearch-model" class="ui-control" type="text" placeholder="Recommended for each source" autocomplete="off" spellcheck="false">
                <span class="ui-field-help">Only the built-in web search uses it. Empty follows the model Ouroboros recommends for each source.</span></label></div>
            <div class="ui-field-help" data-websearch-preview role="status"></div>
            ${SAVE_ROW}</section>`;
    const [source, model, preview] = ['#s-websearch-source', '#s-websearch-model', '[data-websearch-preview]'].map((selector) => container.querySelector(selector));
    let dirty = false;
    let previews = 0;
    // An edited native Anthropic id takes the routed spelling, so dispatch recognizes the new intent.
    const selection = () => {
        const text = model.value.trim();
        return { backend: source.value, model: dirty && source.value === 'anthropic' && text && !text.includes('::') && !text.startsWith('anthropic/') ? `anthropic::${text}` : text };
    };
    const showPreview = async () => {
        const mine = ++previews;
        try {
            const data = await apiClient.webSearchPreview(dirty ? selection() : {});
            if (mine !== previews) return;
            const { main, details } = webSearchPreviewWords(data, (id) => Object.fromEntries(SEARCH_SOURCES)[id] || id);
            preview.innerHTML = `${escapeHtml(main)}${details ? ` <details class="ui-disclosure"><summary>Details</summary>${escapeHtml(details)}</details>` : ''}`;
        } catch { if (mine === previews) preview.textContent = 'Search routes could not be checked. Your choices are kept.'; }
    };
    let asRead = {};  // the choice as read when the typing began: a Save writes only what the owner changed from it
    const read = async () => {
        if (dirty) { void showPreview(); return; }  // a typed choice stays; the routes it would take are read again
        let settings;
        try { settings = await ctx.readSettings(); } catch { return; }
        source.value = settings.OUROBOROS_WEBSEARCH_BACKEND || 'auto';
        model.value = String(settings.OUROBOROS_WEBSEARCH_MODEL || '');
        asRead = { OUROBOROS_WEBSEARCH_BACKEND: source.value, OUROBOROS_WEBSEARCH_MODEL: model.value };
        void showPreview();
    };
    const say = (words) => setResult(container.querySelector('[data-editor-result]'), words);
    const edited = () => { dirty = true; container.querySelector('[data-editor-discard]').hidden = false; say({}); void showPreview(); };
    const onClick = async (event) => {
        const button = event.target.closest?.('[data-editor-save]');
        if (event.target.closest?.('[data-editor-discard]')) { dirty = false; container.querySelector('[data-editor-discard]').hidden = true; void read(); return; }
        if (!button) return;
        setButtonBusy(button, true);
        let outcome;
        try {
            const fresh = await ctx.readSettings();
            const wanted = { OUROBOROS_WEBSEARCH_BACKEND: selection().backend, OUROBOROS_WEBSEARCH_MODEL: selection().model };
            const body = Object.fromEntries(Object.entries(wanted).filter(([key, value]) => value !== asRead[key]));
            outcome = await writeModelKeys('models:websearch', body, fresh, { ask: ctx.ask });
        } catch (error) { outcome = { state: 'refused', effect: 'now', withheld: '', message: error?.message || String(error), code: '' }; }
        setButtonBusy(button, false);
        if (landed(outcome) || outcome.state === 'unchanged') { dirty = false; container.querySelector('[data-editor-discard]').hidden = true; await read(); }
        say(saveWords(outcome));
    };
    const onShown = (event) => { if (event.detail?.path === path) void read(); };
    source.addEventListener('change', edited);
    model.addEventListener('input', edited);
    container.addEventListener('click', onClick);
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    void read();
    return { unsaved: () => dirty, unmount() { container.removeEventListener('click', onClick); globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown); } };
}

registerSettingsPage('models', {
    title: (path) => {
        const [, , slot, n] = path.split('/');
        return !slot ? 'Models' : slot === 'fallback' ? `Fallback ${n || ''}`.trim() : wordsOf(`models:${slot}`).name;
    },
    mount: (container, page) => {
        const [, , slot, n] = page.path.split('/');
        return !slot ? mountModels(container, page) : slot === 'websearch' ? mountWebSearch(container, page)
            : mountRole(container, page, slot, slot === 'fallback' ? Math.max(0, Number(n || 1) - 1) : 0);
    },
});
