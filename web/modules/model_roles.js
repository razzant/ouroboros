// The shared Models editor for Settings and onboarding. Model strings remain
// the routing authority; account and context choices belong to the exact role.
// Catalog arrival only enriches choices. It never authors an assignment.
import { apiClient, fetchJson } from './api_client.js';
import { MODEL_CATALOG_TIMEOUT_MS, catalogReadNote, mergeModelCatalog } from './settings_catalog.js';
import { accountRows, bindStatusSurface, claudexorStatus } from './claudexor_status_store.js';
import { parseModelSource, composeModelSource, configuredApiProviders, indexProfilesByHarness,
    profileOptionsFor, routeChoiceGroups, selectHtml, mintStableId, API_CHOICE_PREFIX,
    DEFAULT_API_PROVIDER } from './route_editor_primitives.js';
import { PROCESSING_PREFERENCE_KEY, MODEL_PROCESSING_PREFERENCES_KEY, PROCESSING_CHOICES,
    processingSelectHtml, processingIntentLabel, processingCapabilityNote, catalogModelOptions } from './route_editor_primitives.js';
import { revealNewRow } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
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
 * @param {{sources?: Array, providers?: Array<{id:string,label?:string}>, current?: string,
 *   catalogKnown?: boolean, accountsKnown?: boolean, providerProfiles?: object}} args
 *   `providers` is `configuredApiProviders()` output, `current` a stored row source.
 */
export function modelSourceGroups({ sources = [], providers = [], current = '',
    catalogKnown = false, accountsKnown = false, providerProfiles = {}, hasConfiguredAccounts = false } = {}) {
    return [
        ...(current === 'inherit' ? [{ options: [{ value: 'inherit', label: 'Uses Main' }] }] : []),
        ...routeChoiceGroups({ modelSources: sources, providers, providerProfiles,
            currentChoice: current === 'inherit' ? '' : sourceChoice(current),
            catalogKnown, accountsKnown, hasConfiguredAccounts, includeSessions: false }),
    ];
}

export function modelContextNote(item, override = 0) {
    if (Number(override) > 0) return `${Number(override).toLocaleString('en-US')} tokens · set by you; the provider may accept less.`;
    const window = Number(item?.max_context_window || item?.context_window || 0);
    return window > 0 ? `Auto: ${window.toLocaleString('en-US')} tokens · advertised for this route.`
        : 'Auto: context limit not known for this route.';
}

export function modelRolesHost(id) {
    return `<div id="${escapeHtml(id)}" class="model-role-editor"></div>`;
}

/** In-memory form controller; refreshes have no settings or login side effects. */
export function createModelRolesEditor({ hostId, store = claudexorStatus,
    doc = () => document, onChange = () => {}, catalogFetch = fetchJson, showContext = true } = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    let slots = [];
    let rows = [];
    let settings = {};
    let providerProfiles = {};
    let apiProviders = [];
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
    function effectiveSource(row) { return row.source === 'inherit' ? rows.find((entry) => entry.slot.slot === 'main')?.source || 'openrouter' : row.source; }
    function responseModel(row) {
        const selected = row.source === 'inherit' ? rows.find((entry) => entry.slot.slot === 'main') : row;
        return selected ? composeModelSource(selected.source, selected.model) : '';
    }
    function sourceId(row) { const source = effectiveSource(row); return source.startsWith('subscription:') ? source.slice(13) : ''; }
    function catalogKey(row) { return JSON.stringify([sourceId(row), row.account]); }
    // A saved source whose key is gone stays selectable and labelled; discovery
    // only widens the list, it never rewrites the owner's assignment.
    function sourceGroupsFor(row, facts = {}) {
        return modelSourceGroups({ sources: catalog.model_sources, providers: apiProviders,
            providerProfiles, current: row.source, ...facts });
    }
    function itemsFor(row) {
        return sourceId(row) ? (catalogs.get(catalogKey(row))?.items || [])
            : catalog.items.filter((item) => parseModelSource(item.value || item.id).source === row.source);
    }
    function currentItem(row) {
        const model = row.source === 'inherit' ? rows.find((entry) => entry.slot.slot === 'main')?.model : row.model;
        const matches = itemsFor(row).filter((item) => (parseModelSource(item.value || item.id).model === model
            || item.id === model) && (!row.account || item.credential_profile_id === row.account));
        return matches.length === 1 ? matches[0] : undefined;
    }

    function inputIdFor(row) {
        return row.slot.slot === 'fallback' && rows.find((entry) => entry.slot === row.slot) !== row
            ? `${hostId}-${row.id}` : row.slot.inputId;
    }

    function detailsHtml(row) {
        if (!showContext) return '';
        return `<details class="model-role-details"><summary>Context &amp; processing · <span data-processing-summary>${escapeHtml(processingIntentLabel(row.processing_preference, processingPreference))}</span></summary>
            <label class="ui-field">Processing ${processingSelectHtml(`data-model-role-processing aria-label="${escapeHtml(row.slot.label)} processing"`, row.processing_preference)}</label>
            <div class="ui-field-help">Uses the same model and reasoning effort. An explicit native service choice takes precedence.</div>
            <div class="ui-field-help" data-processing-capability></div>
            <div class="model-role-context">
                <label class="ui-field">Window <input id="${escapeHtml(inputIdFor(row))}-context" class="ui-control" data-model-role-context type="number" min="0" step="1"
                    placeholder="Auto" value="${escapeHtml(row.context || '')}" aria-label="${escapeHtml(row.slot.label)} context window"></label>
                <span data-model-context-note>${escapeHtml(modelContextNote(currentItem(row), row.context))}</span>
            </div>
            <div class="model-role-context">
                <label class="ui-field">Maximum response <input class="ui-control" data-response-limit type="number" min="0" step="1" placeholder="Auto" aria-label="${escapeHtml(row.slot.label)} maximum response"></label>
                <button type="button" class="btn btn-default" data-response-apply>Apply maximum</button>
                <span class="ui-field-help" data-response-note>Auto. Applies to this exact model and server; blank restores metadata.</span>
            </div></details>`;
    }

    function rowHtml(row, index, total) {
        const isFallback = row.slot.slot === 'fallback';
        const inputId = inputIdFor(row);
        return `<div class="model-role-row" data-model-role="${escapeHtml(row.id)}">
            <div class="model-role-controls">
                ${selectHtml(`data-model-role-source aria-label="${escapeHtml(row.slot.label)} source"`, sourceGroupsFor(row), sourceChoice(row.source))}
                ${modelChooserHtml(`id="${escapeHtml(inputId)}" data-model-role-model aria-label="${escapeHtml(row.slot.label)}${isFallback ? ` ${index + 1}` : ''}"`, row.model, `${hostId}-${row.id}-models`, [], { placeholder: row.slot.slot === 'main' ? 'Choose a model' : 'Empty uses Main' })}
                <select class="ui-control" data-model-role-account aria-label="${escapeHtml(row.slot.label)} account" ${sourceId(row) ? '' : 'hidden'}></select>
                ${isFallback ? `<span class="model-role-order"><button type="button" class="btn btn-default" data-model-up aria-label="Move fallback up" ${index === 0 ? 'disabled' : ''}>↑</button><button type="button" class="btn btn-default" data-model-down aria-label="Move fallback down" ${index === total - 1 ? 'disabled' : ''}>↓</button><button type="button" class="btn btn-default" data-model-remove aria-label="Remove fallback">Remove</button></span>` : ''}
            </div>
            <div class="model-role-notes"><span id="${escapeHtml(hostId)}-${escapeHtml(row.id)}-status" class="model-role-meta ui-field-help" data-model-role-status></span>${detailsHtml(row)}</div>
            <div class="ui-status ui-field-help" id="${escapeHtml(hostId)}-${escapeHtml(row.id)}-error" data-model-role-error data-tone="error" hidden></div>
        </div>`;
    }

    function render() {
        const element = host();
        if (!element || destroyed || !loaded) return;
        disposeChoosers();
        element.innerHTML = (showContext ? `<section class="model-role-group"><label class="ui-field">Processing
            ${processingSelectHtml('data-global-processing aria-label="Global processing"', processingPreference, { global: true })}</label>
            <p class="model-role-copy">Applies to new tasks. Fast permits accelerated paid service within existing limits. Economy may fall back to the ordinary rate. Model and reasoning effort stay the same.</p>
            <div class="ui-field-help">Keep route defaults preserves existing native choices. Explicit native service choices take precedence; the requested mode is not proof of the mode served.</div></section>` : '') + slots.map((slot) => {
            const matching = rows.filter((row) => row.slot.slot === slot.slot);
            const local = matching[0]?.local || false;
            return `<section class="model-role-group" data-model-role-group="${escapeHtml(slot.slot)}">
                <div class="model-role-head"><h4 title="${escapeHtml(slot.note || '')}">${escapeHtml(slot.label.replace(/ Model$/, ''))}</h4>
                    ${slot.settingsToggleId ? `<label class="local-toggle ui-field ui-field-inline"><input class="ui-checkbox" id="${escapeHtml(slot.settingsToggleId)}" type="checkbox" data-model-local aria-label="${escapeHtml(slot.label)} local runtime" ${local ? 'checked' : ''}> Local</label>` : ''}
                    ${slot.slot === 'fallback' ? '<button type="button" class="btn btn-default" data-model-add>Add fallback</button>' : ''}
                </div>
                ${slot.slot === 'fallback' ? '<p class="model-role-copy">Tried in this order when the acting model refuses, cannot be reached or loses its answer. A wait starts only when none of them answers.</p>' : ''}
                ${matching.map((row, index) => rowHtml(row, index, matching.length)).join('')}
            </section>`;
        }).join('');
        element.querySelector('[data-global-processing]')?.addEventListener('change', (event) => {
            processingPreference = event.target.value; processingTouched = true; changed(); updateCatalogViews();
        });
        bindRows(element);
        disposeChoosers = bindModelChoosers(element);
        updateCatalogViews();
        for (const row of rows) void refreshRow(row);
    }

    function updateCatalogViews() {
        const element = host();
        if (!element || destroyed) return;
        const profiles = indexProfilesByHarness(store.snapshot);
        for (const row of rows) {
            const node = element.querySelector(`[data-model-role="${row.id}"]`);
            if (!node) continue;
            const responseIdentity = JSON.stringify([responseModel(row), row.local, row.account]);
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
            const credentialHarness = catalog.model_sources.find((entry) => entry.id === sourceId(row))?.credentialHarness || '';
            const accountHtml = selectHtml('', [{ options: profileOptionsFor(profiles[credentialHarness], row.account, { accountsKnown: store.accountsKnown && Boolean(credentialHarness) }) }], row.account);
            const accountOptions = accountHtml.slice(accountHtml.indexOf('>') + 1, accountHtml.lastIndexOf('</select>'));
            if (account.innerHTML !== accountOptions) account.innerHTML = accountOptions;
            updateModelChooser(node.querySelector('[data-model-role-model]'), catalogModelOptions(itemsFor(row).map((item) => ({
                ...item, value: parseModelSource(item.value || item.id).model,
            }))));
            if (showContext) node.querySelector('[data-model-context-note]').textContent = modelContextNote(currentItem(row), row.context);
            if (showContext) node.querySelector('[data-processing-summary]').textContent = processingIntentLabel(row.processing_preference, processingPreference);
            if (showContext) node.querySelector('[data-processing-capability]').textContent = processingCapabilityNote(row.processing_preference || processingPreference, currentItem(row)?.processing,
                catalog.model_sources.find((source) => source.id === sourceId(row))?.processingPreferences);
            const status = node.querySelector('[data-model-role-status]');
            const rowCatalog = catalogs.get(catalogKey(row));
            // A subscription row reports its own account read in full; an API row
            // sits under the section banner, so it keeps to the compact form.
            const own = Boolean(sourceId(row) && rowCatalog);
            const note = catalogReadNote(own ? rowCatalog : catalog, { compact: !own });
            status.textContent = row.local ? 'Uses the local runtime.'
                : note || (!row.model ? (row.slot.slot === 'main' ? 'Choose a model to continue.' : 'Uses Main.')
                    : sourceId(row) ? (rowCatalog?.read_state === 'ok' && !rowCatalog.items.length
                        ? 'No models were listed for this source. Your selection is kept.'
                        : 'Uses your subscription. No API key required.') : '');
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
        if (requests.has(key) || (!force && catalogs.has(key))) return requests.get(key)?.promise;
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
            .finally(() => { clearTimeout(timer); requests.delete(key); updateCatalogViews(); });
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
        const identity = JSON.stringify([model, row.local, row.account]);
        const draft = field.value;
        const sequence = row.responseSequence = (row.responseSequence || 0) + 1;
        if (apply) row.responseApplying = sequence;
        const current = () => !destroyed && sequence === row.responseSequence && node.isConnected
            && identity === JSON.stringify([responseModel(row), row.local, row.account]);
        try {
            const data = await apiClient.responseLimitPreview({ model, local: row.local, account: row.account });
            if (!current()) return;
            if (!data.response_limit || !data.route) throw new Error('Maximum response is not checked');
            let evidence = data.response_limit;
            if (apply) {
                const result = await apiClient.ownerCapabilityAck({ ...data.route, route_fp: evidence.route_fp, max_output_tokens: value });
                if (!current()) return;
                if (!result.ok) throw new Error(result.error || 'Maximum response could not be saved');
                evidence = result.ack;
            }
            // Edits can precede the deferred toggle which starts this read.
            if (!apply && !responseEdits.has(field) && field.value === draft) field.value = evidence.source === 'owner_ack' ? String(evidence.max_output_tokens || '') : '';
            field.setAttribute('aria-invalid', 'false');
            const source = evidence.source === 'owner_ack' ? 'your maximum' : 'model metadata';
            const endpoint = data.route.base_url ? new URL(data.route.base_url) : null;
            const server = endpoint ? ` Server: ${endpoint.host}${endpoint.pathname.replace(/\/$/, '')}.` : '';
            note.textContent = `${evidence.max_output_tokens && !evidence.stale ? `${evidence.max_output_tokens.toLocaleString('en-US')} tokens · ${source}` : 'Auto: maximum response unknown'}.${server} Context window is separate.${data.route.provider === 'claudexor' ? ' Planning allowance only; the native engine controls its response.' : ''}`;
        } catch (error) { if (current()) note.textContent = error.message; }
        finally { if (row.responseApplying === sequence) row.responseApplying = 0; }
    }

    function bindRows(element) {
        for (const row of rows) {
            const node = element.querySelector(`[data-model-role="${row.id}"]`);
            node.querySelector('[data-model-role-source]').addEventListener('change', (event) => {
                row.source = sourceFromChoice(event.target.value); row.model = ''; row.account = ''; row.context = 0;
                changed(row.slot.slot); render();
                host()?.querySelector(`[data-model-role="${row.id}"] [data-model-role-model]`)?.focus();
            });
            node.querySelector('.model-role-details')?.addEventListener('toggle', (event) => {
                if (event.target.open) void responseLimit(row, node);
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
                rows = rows.filter((entry) => entry !== row); changed(); render();
                host()?.querySelector('[data-model-add]')?.focus();
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
        load(value, contract = {}) {
            validationAttempted = false;
            settings = { ...value }; slots = contract.modelSlots || slots;
            processingPreference = String(settings[PROCESSING_PREFERENCE_KEY] || ''); processingTouched = false;
            providerProfiles = contract.providerProfiles || providerProfiles;
            apiProviders = configuredApiProviders(settings, providerProfiles);
            rows = [];
            for (const slot of slots) {
                if (slot.slot === 'fallback') {
                    String(settings[slot.settingKey] || '').split(',').map((entry) => entry.trim()).filter(Boolean)
                        .forEach((entry, index) => rows.push(rowState(slot, entry, index)));
                } else rows.push(rowState(slot, settings[slot.settingKey]));
            }
            loaded = true; render();
        },
        mount() {
            if (!disposeStatus) disposeStatus = bindStatusSurface(store, { elementId: hostId, doc: getDoc,
                listener: updateCatalogViews });
            render();
        },
        adoptCatalog(data = {}) {
            catalog = mergeModelCatalog(catalog, data);
            if (Array.isArray(data.items)) {
                for (const row of rows) void refreshRow(row, { force: true });
            }
            updateCatalogViews();
        },
        collect,
        validateAll,
        noteSaveAttempt() { validationAttempted = true; updateCatalogViews(); },
        validate() {
            return validateAll()[0] || '';
        },
        destroy() {
            destroyed = true; disposeStatus?.(); disposeStatus = null;
            disposeChoosers();
            for (const request of requests.values()) request.controller.abort();
        },
    };
}
