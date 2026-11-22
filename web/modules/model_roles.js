// The shared Models editor for Settings and onboarding. Model strings remain
// the routing authority; account and context choices belong to the exact role.
// Catalog arrival only enriches choices. It never authors an assignment.
import { apiClient, fetchJson } from './api_client.js';
import { SAFETY_MODEL_KEYS } from './consciousness_api.js';
import { API_PROVIDER_LABELS, harnessPresentation } from './harness_presentation.js';
import { sourceMark } from './model_source_mark.js';
import { bindSourcePickers } from './source_picker.js';
import { createScopedModelCatalog, modelSlotCatalogQuery, catalogReadNote, invalidateModelCatalogReads, mergeModelCatalog, refreshModelCatalog, watchAccountModelCatalog, withConnectionCatalogs } from './settings_catalog.js';
import { CODECS, CONFIRMS, bindSettingRows, commitSetting, doorSign, outcomeLine, setButtonBusy, settingsRow, searchModelAfterSourceSwitch, searchModelForKey, settingsReadMessage, settingsSnapshot, syncSearchKey, webSearchPreviewWords } from './settings_controls.js';
import { registerSettingsPage } from './settings_pages.js';
import { HAND_WRITTEN, rowsOf } from './settings_rows.js';
import { setResult } from './ui_primitives.js';
import { accountRows, bindStatusSurface, claudexorStatus } from './claudexor_status_store.js';
import { parseModelSource, composeModelSource, apiConnections, configuredApiProviders, indexProfilesByHarness,
    profileOptionsFor, routeChoiceGroups, selectHtml, mintStableId, unavailableConnectionState, API_CHOICE_PREFIX,
    DEFAULT_API_PROVIDER, PROCESSING_PREFERENCE_KEY, MODEL_PROCESSING_PREFERENCES_KEY, MODEL_ACCOUNTS_KEY, PROCESSING_CHOICES,
    processingSelectHtml, processingIntentLabel, catalogModelOptions, modelPriceWords, withoutPrice, tierPriced,
    KEY_PROBLEM_WORDS, PRICE_UNKNOWN, routeKeyProblem, keyProblemLine, apiProviderLabel } from './route_editor_primitives.js';
import { buildModelIndex, describeRoute, windowWords } from './chat_model_routes.js';
import { CONNECTION_CONDITION, dropTransport, modelCatalog, revisionIn, watchModelCatalog } from './model_catalog_store.js';
import { canChooseTier, confirmMainUltrafast, redundantStandard, routeTiers, tierBinding, tierChoices, tierNamed, tierWord } from './processing_choice.js';
import { fmt, translateString } from './i18n.js';
import { revealNewRow } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { catalogPrograms } from './model_program.js';
import { modelChooserHtml, bindModelChoosers, updateModelChooser } from './model_chooser.js';

export { MODEL_ACCOUNTS_KEY };
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
// A size refusal's window by how its number was known (the projection's `basis`, `context_window.WindowLimit.basis`).
// `exact_local_count`: a local instance counted the request Ouroboros refused before sending, no server answered;
// `anchored_estimate`: Ouroboros derived it from a numberless refusal, an anchored estimate of the refused request or a
// declared total the refusal confirmed. A projection that names no basis, or one not in this table, claims none.
const REFUSAL_BASES = { provider_stated: 'stated by the provider in a refusal',
    exact_local_count: 'derived from an exact local count before sending',
    anchored_estimate: 'derived from a refusal that stated no number', unrecorded: 'from an earlier refusal, basis not recorded' };
const sourceWords = (window) => (window.source !== 'refusal' ? WINDOW_SOURCES[window.source]?.[0] || window.source
    : Object.hasOwn(REFUSAL_BASES, window.basis) ? REFUSAL_BASES[window.basis] : 'from a refusal, basis unknown');
const tokenWords = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : String(n));

/** The route's window in owner words, from the projection's typed fields (never catalogue arithmetic): the window,
 *  where it comes from (a refusal's number: stated by the provider, derived, or its basis not recorded), the usable
 *  input, a source that disagrees and an old confirmation no longer used; the observation stamps stay in the
 *  projection's record. */
export function modelContextNote(preview) {
    const window = preview?.window;
    if (!window?.tokens) return 'Context limit not known for this route.';
    const usable = Number.isFinite(window.usable) ? window.usable : window.tokens - (window.reserve || 0);
    const other = window.contradiction;
    const retired = Number(window.retired_ack?.window_tokens) || 0;
    return `Window ${tokenWords(window.tokens)} tokens (${sourceWords(window)}) · ${tokenWords(usable)} usable for input`
        + (other ? ` · ${WINDOW_SOURCES[other.source]?.[1] || `${other.source} says`} ${tokenWords(other.tokens)}` : '')
        + (retired ? ` · old confirmation ${tokenWords(retired)} not used` : '');
}

// The slot an empty role uses: Main, except the Safety Supervisor, which checks with the Light model (and so
// with Main when Light is empty too). A contract row may name its own (`inherits`).
const INHERITS = Object.freeze({ safety: 'light' });
export const inheritTargetOf = (slot) => String(slot?.inherits || INHERITS[slot?.slot] || 'main');
/** An empty Safety Supervisor's option as the runtime reads it (model_slots.model_role_option): from Safety, Light and
 *  Main in turn the first one stated, the chain ending at the first role whose model is set (`set`). */
export function safetyChain(entries, valueOf) {
    for (const slot of ['safety', 'light', 'main']) {
        const entry = entries.find((each) => each.slot === slot);
        if (!entry) continue;
        const value = valueOf(entry);
        if (value) return value;
        if (entry.set || slot === 'main') return '';
    }
    return '';
}

export function modelRolesHost(id) {
    return `<div id="${escapeHtml(id)}" class="model-role-editor"></div>`;
}

/** Settings shows this under Main, before Save: what a new Main does to work already running (owner, 2026-11-03),
 *  in one sentence (owner 04.11, 1A: work with its own model is the work's own fact). */
export const MAIN_FOLLOW_NOTE = 'Running work that follows Main switches at its next call.';

/** In-memory form controller; refreshes have no settings or login side effects. `mainNote`: Settings' Main note.
 *  `only` (`{ slot, index }`) draws one role alone, as a form of labelled fields (a role's page in Settings): its
 *  source may always return to the role it follows, and a fallback past the saved ones is a new, unsaved one.
 *  A row's Processing offers what its own route says where it is bound (its source, model and account; DESIGN
 *  "Composer owner controls"); `tierStore` holds a key model's own endpoints, asked for a role's page alone, and
 *  `askUltrafast` is the question before Main's draft takes Ultrafast, which only Main's page offers. */
export function createModelRolesEditor({ hostId, store = claudexorStatus,
    doc = () => document, onChange = () => {}, catalogFetch = fetchJson, supportsScopes, legacyCatalog = () => modelCatalog.catalog(), readApiCatalog = null, apiCatalogAllowed = () => true, showContext = true, mainNote = false, only = null,
    tierStore = modelCatalog, askUltrafast = confirmMainUltrafast } = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    let slots = [];
    let offeredSlots = [];  // every slot in contract order; a `managed` one (a deployment's checker) is shown, never a row
    let rows = [];
    let settings = {};
    let providerProfiles = {};
    let apiProviders = [];
    let connections = [];
    let settingsRevisions = [];  // each API connection's revision as the loaded read gives it (`_meta.connections`)
    let mainChoice = [];  // `_meta.main_connection_choice`: an unauthored Main waits for the owner's key
    let catalog = { items: [], model_sources: [], read_state: 'not_read' };
    let loaded = false;
    let destroyed = false;
    let disposeStatus = null;
    let disposeChoosers = () => {};
    let disposeTiers = null;
    let validationAttempted = false;
    let processingPreference = '', processingTouched = false;
    let sources = null;
    let connectionLists = {};

    const catalogs = createScopedModelCatalog({
        supportsScopes: () => supportsScopes?.(settings) ?? true,
        legacyCatalog,
        fetchCatalog: (url, init) => url.includes('?provider_id=') && readApiCatalog
            ? readApiCatalog(url, init) : catalogFetch(url, init),
        onUpdate(query, data) {
            if (destroyed) return;
            if (query === 'view=sources') sources = data;
            else if (!visibleRows().some(row => catalogKey(row) === query || (query === 'provider_id=local' && row.local))) return;
            updateCatalogViews();
        },
    });
    const descriptors = () => sources?.model_sources ?? catalog.model_sources;
    const visibleRows = () => only ? [soleRow()].filter(Boolean) : rows;
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
        if (row.processing_preference && !tierBase(row).includes(row.processing_preference)) errors.push({ field: '[data-model-role-processing]', message: `${row.slot.label}: choose Standard, Fast, Flex or inherited processing.` });
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
        ...(processingPreference && !PROCESSING_CHOICES.includes(processingPreference) ? ['Choose Standard, Fast, Flex or route defaults for Processing.'] : []),
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
    // An empty role says the route it uses now, as the runtime resolves it: the model, an account and a tier it takes.
    const emptyUsesText = (row) => {
        const used = inheritedRow(row);
        const name = used?.model ? String(used.model).slice(String(used.model).lastIndexOf('/') + 1) : '';
        const account = row.slot.slot === 'safety' && !heldLocal(row) && sourceId(row) ? heldOption(row, 'account') : '';
        const tier = heldOption(row, 'processing_preference');
        const facts = [heldLocal(row) ? 'local' : '', account && `account ${account}`, tierNamed(tier) ? tierWord(tier) : ''].filter(Boolean);
        const head = inheritTargetOf(row.slot) === 'main' ? 'Empty uses Main' : `Empty uses the ${inheritName(row)} model`;
        return name ? `${head}: ${[name, ...facts].join(' · ')}` : head;
    };
    function effectiveSource(row) { return row.source === 'inherit' ? inheritedRow(row)?.source || 'openrouter' : row.source; }
    // An empty Safety Supervisor's account and tier, as the runtime reads them (model_slots.model_role_option): its
    // own, else Light's, else Main's, the chain ending at the first role with a model. Other roles hold their own.
    function heldOption(row, field) {
        if (row.slot.slot !== 'safety' || row[field]) return row[field];
        return safetyChain(rows.map((entry) => ({ slot: entry.slot.slot, set: entry.source !== 'inherit' && Boolean(entry.model),
            value: entry[field] })), (entry) => entry.value);
    }
    // Where an empty role runs (route_choice.model_role_slot_override): its own box over a cloud Main puts Main's model
    // on the local model; an empty Supervisor takes Light's route, Light's box included.
    function heldLocal(row) {
        if (row.source !== 'inherit') return Boolean(row.local);
        const main = rows.find((entry) => entry.slot.slot === 'main');
        if (row.slot.slot !== 'safety') return Boolean(row.local || main?.local);
        const light = rows.find((entry) => entry.slot.slot === 'light');
        return light && light.source !== 'inherit' && light.model ? Boolean(light.local) : Boolean(light?.local || main?.local);
    }
    // The revision of the API connection the row's route sends to, as this page read it ('' none, null not named).
    const sourceRevision = (row) => (heldLocal(row) ? '' : revisionIn(settings, { model: responseModel(row), use_local: false }));
    function responseModel(row) {
        const selected = row.source === 'inherit' ? inheritedRow(row) : row;
        return selected ? composeModelSource(selected.source, selected.model) : '';
    }
    function sourceId(row) { const source = effectiveSource(row); return source.startsWith('subscription:') ? source.slice(13) : ''; }
    function catalogKey(row) {
        return modelSlotCatalogQuery(effectiveSource(row), heldOption(row, 'account'));
    }
    // A saved source whose key is gone stays selectable and labelled; discovery
    // only widens the list, it never rewrites the owner's assignment.
    function sourceGroupsFor(row, facts = {}) {
        return modelSourceGroups({ sources: descriptors(), providers: apiProviders, connections, providerProfiles, current: row.source,
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
    function refreshCatalogs({ force = true, descriptors = true, subscriptionsOnly = false, joinPending = false } = {}) {
        if (destroyed) return;
        const reads = descriptors ? [catalogs.read('view=sources', { force, joinPending })] : [];
        const shown = new Map(visibleRows().map(row => [catalogKey(row), row]));
        for (const row of shown.values()) {
            if (!subscriptionsOnly || sourceId(row)) reads.push(refreshRow(row, { force, joinPending }));
        }
        return Promise.all(reads);
    }
    const onMaintenanceSettled = () => refreshCatalogs({ descriptors: Boolean(only) || !showContext, subscriptionsOnly: true });
    function itemsFor(row) {
        const source = effectiveSource(row);
        const scoped = catalogs.get(catalogKey(row));
        if (!row.local && Object.hasOwn(connectionLists, source)) return connectionLists[source];
        if (row.source === 'inherit' && !sourceId(row)) return [];
        const items = scoped?.items ?? catalog.items;
        return items.filter(item => !item.use_local && parseModelSource(item.value || item.id).source === source
            && (!sourceId(row) || !row.account || item.credential_profile_id === row.account
                || (scoped && !item.credential_profile_id)));
    }
    // A fallback with no tier of its own keeps its caller's (model_slots.task_processing_preference): Main's or a lead's.
    const mainTier = () => (rows.find((entry) => entry.slot.slot === 'main')?.processing_preference
        ?? modelRoleMap(settings[MODEL_PROCESSING_PREFERENCES_KEY]).main) || processingPreference;
    const tieredProcessing = (row) => tierPriced(row.processing_preference || processingPreference)
        || (row.slot.slot === 'fallback' && !row.processing_preference && tierPriced(mainTier()));
    // What a row's route says of each tier where it is bound (processing_choice.js `routeTiers`); null for a local
    // role or one without a model. A key's row is told by the general catalog. A subscription row's accounts are told
    // by the read of its own source and account alone, and only while that read is this moment's: it answered, it is
    // of the engine store the general catalog names, and no account both reads know has another fingerprint in one
    // than in the other (whichever answered later). Otherwise nothing is told of its accounts; what its transport
    // cannot carry stays the general read's word, never a guess here.
    const tierBase = (row) => (row.slot.slot === 'main' ? [...PROCESSING_CHOICES, 'ultrafast'] : PROCESSING_CHOICES);
    const indexes = new WeakMap();
    const indexOf = (read) => { if (!indexes.has(read)) indexes.set(read, buildModelIndex({ catalog: read })); return indexes.get(read); };
    const transports = new WeakMap();  // a general catalog → its transports' word alone, no account's
    const transportsOf = (read) => {
        const said = 'current' in read ? read.current : read;
        if (!transports.has(read)) transports.set(read, { current: said ? { store_id: said.store_id, model_sources: said.model_sources } : null });
        return indexOf(transports.get(read));
    };
    const prints = (index, route) => new Map((index.tiers(route)?.accounts || []).filter((entry) => entry.fingerprint).map((entry) => [entry.id, entry.fingerprint]));
    function rowTiers(row) {
        if (heldLocal(row)) {  // the local model: its adapter says what it can send, whatever model it loads
            const route = { model: responseModel(row) || 'local', use_local: true };
            const index = indexOf(catalog);
            const said = describeRoute(index, route);
            return { index, route, account: '', said, ...routeTiers({ index, route, account: '', sourceLabel: said.sourceLabel }) };
        }
        const model = responseModel(row);
        if (!model || model.endsWith('::')) return null;
        const [route, account, general] = [{ model, use_local: false }, sourceId(row) ? heldOption(row, 'account') : '', indexOf(catalog)];
        let index = general;
        if (sourceId(row)) {
            const own = catalogs.get(catalogKey(row));
            const scoped = own?.current ? indexOf(own) : null;
            const known = scoped && prints(general, route);
            index = scoped && !(scoped.storeId && general.storeId && scoped.storeId !== general.storeId)
                && [...prints(scoped, route)].every(([id, print]) => (known.get(id) || print) === print) ? scoped : transportsOf(catalog);
        } else {
            // A single-role page may have only the API connection's scoped read, not a whole-catalog read.
            const own = catalogs.get(catalogKey(row));
            if (own?.current) index = indexOf(own);
        }
        const said = describeRoute(sourceId(row) ? general : index, route);
        return { index, route, account, said, ...routeTiers({ index, route, account, detail: tierStore.detail(route), sourceLabel: said.sourceLabel }) };
    }
    // What a question about Main's Ultrafast is asked for: the row's route and account, with the store and the
    // account's fingerprint where its read names them.
    const rowBinding = (row) => {
        const tiers = rowTiers(row);
        return tiers ? tierBinding(tiers.index, tiers.route, tiers.account, indexOf(catalog)) : '';
    };
    const tierIdentity = (row) => JSON.stringify([rowBinding(row), sourceRevision(row)]);
    // Main's page alone offers Ultrafast, beyond one already stored; any other row keeps what it holds.
    const tierOptions = (row, said = null) => tierChoices(only && row.slot.slot === 'main' ? tierBase(row) : PROCESSING_CHOICES, said, row.processing_preference);
    const drawnTiers = new WeakMap();  // a Processing select → the options last written into it
    /** The question before Main's draft takes Ultrafast, for the route it has now. A yes stands only for the binding
     *  it was asked for, and only while the row still has it when the answer comes. */
    async function confirmUltrafast(row, carry = false) {
        const before = row && rowTiers(row);
        if (!only || !before) return false;
        const asked = tierIdentity(row);
        const revision = row.bindingRevision;
        const processing = row.processing_preference;
        const current = () => !destroyed && soleRow() === row && revision === row.bindingRevision
            && processing === row.processing_preference && asked === tierIdentity(row);
        // Reuse the role's addressed detail read, including one already started by the picker. Never show that
        // answer under a model that replaced the one this press asked about (even if it later came back).
        if (before.said.lane === 'api') await tierStore.readDetail(before.route);
        if (!current()) return false;
        const tiers = rowTiers(row);
        if (!carry && !canChooseTier('ultrafast', tiers)) return false;
        const yes = await askUltrafast({ model: tiers.said.short, source: [tiers.said.sourceLabel, tiers.accountLabel].filter(Boolean).join(' · '),
            explain: tiers.explain('ultrafast'), priceSaid: Boolean(tiers.priceOf('ultrafast')), pinned: Boolean(tiers.account), doc: getDoc() });
        const confirmed = Boolean(yes) && current() && (carry || canChooseTier('ultrafast', rowTiers(row) || {}));
        if (confirmed) row.ultrafastFor = asked;
        return confirmed;
    }
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
        return `<details class="model-role-details ui-disclosure"><summary><span>${only ? 'More: processing, context window, max output' : 'Context &amp; processing'} · <span data-processing-summary>${escapeHtml(processingIntentLabel(row.processing_preference, processingPreference))}</span></span></summary>
            <label class="ui-field" data-processing-field>Processing ${processingSelectHtml(`data-model-role-processing aria-label="${escapeHtml(row.slot.label)} processing" aria-describedby="${escapeHtml(inputIdFor(row))}-processing-note"`, row.processing_preference, { choices: tierOptions(row) })}</label>
            <div class="ui-field-help">Uses the same model and reasoning effort. An explicit native service choice takes precedence.</div>
            <div class="ui-field-help" id="${escapeHtml(inputIdFor(row))}-processing-note" data-processing-capability></div>
            <div class="model-role-context">
                <label class="ui-field">Context window <input id="${escapeHtml(inputIdFor(row))}-context" class="ui-control" data-model-role-context type="number" min="0" step="1"
                    placeholder="Auto" value="${escapeHtml(row.context || '')}" aria-label="${escapeHtml(row.slot.label)} context window">
                    <span class="ui-field-help" data-model-context-note>${escapeHtml(modelContextNote(row.windowPreview))}</span></label>
                <button type="button" class="btn btn-default" data-window-clear hidden>Clear confirmation</button>
            </div>
            <div class="model-role-context">
                <label class="ui-field">Max output tokens <input class="ui-control" data-response-limit type="number" min="0" step="1" placeholder="Auto" aria-label="${escapeHtml(row.slot.label)} max output tokens">
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
            <p class="model-role-copy">Applies to new tasks. Fast permits accelerated paid service within existing limits. Flex may fall back to the ordinary rate. Model and reasoning effort stay the same.</p>
            <details class="model-role-details ui-disclosure"><summary>Details</summary><div class="ui-field-help">Keep route defaults preserves existing native choices. Explicit native service choices take precedence; the requested mode is not proof of the mode served.</div></details></section>` : '') + offeredSlots.map((slot) => {
            if (slot.managed) return fixedGroupHtml(slot);
            const matching = rows.filter((row) => row.slot.slot === slot.slot);
            const local = matching[0]?.local || false;
            // What the role does, said once and visibly under its name (the contract's note; Fallback's own sentence).
            const note = slot.slot === 'fallback' ? 'Tried in this order when the acting model refuses, cannot be reached or loses its answer. A wait starts only when none of them answers.' : slot.note;
            return `<section class="model-role-group" data-model-role-group="${escapeHtml(slot.slot)}">
                <div class="model-role-head"><h4>${escapeHtml(slot.label.replace(/ Model$/, ''))}</h4>
                    ${slot.settingsToggleId ? `<label class="local-toggle ui-field ui-field-inline"><input class="ui-checkbox" id="${escapeHtml(slot.settingsToggleId)}" type="checkbox" data-model-local aria-label="${escapeHtml(slot.label)} local runtime" ${local ? 'checked' : ''}> Local</label>` : ''}
                    ${slot.slot === 'fallback' ? '<button type="button" class="btn btn-default" data-model-add>Add fallback</button>' : ''}
                </div>
                ${note ? `<p class="model-role-copy ui-field-help" data-model-role-note>${escapeHtml(note)}</p>` : ''}
                ${matching.map((row, index) => rowHtml(row, index, matching.length)).join('')}
                ${mainNote && slot.slot === 'main' ? `<p class="model-role-copy" id="${escapeHtml(hostId)}-main-note" data-main-follow-note>${escapeHtml(MAIN_FOLLOW_NOTE)}</p>` : ''}
            </section>`;
        }).join('');
        element.querySelector('[data-global-processing]')?.addEventListener('change', (event) => {
            processingPreference = event.target.value; processingTouched = true; changed(); updateCatalogViews();
        });
        bindRows(element);
        const unbindModels = bindModelChoosers(element);
        const unbindSources = bindSourcePickers(element);
        disposeChoosers = () => { unbindModels(); unbindSources(); };
        updateCatalogViews();
        for (const row of visibleRows()) void refreshRow(row);
    }

    function updateCatalogViews() {
        const element = host();
        if (!element || destroyed) return;
        const profiles = indexProfilesByHarness(store.snapshot);
        let asked = null;  // the one route whose own endpoints this pass asks for, once per model, after the paint
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
                row.bindingRevision = (row.bindingRevision || 0) + 1;
                row.ultrafastFor = null;
                const field = node.querySelector('[data-response-limit]');
                if (field) { field.value = ''; responseEdits.delete(field); }
                const note = node.querySelector('[data-response-note]');
                if (note) note.textContent = 'Route changed; open details to read its maximum, or apply a new value.';
            }
            row.responseIdentity = responseIdentity;
            const source = node.querySelector('[data-model-role-source]');
            const sourceHtml = selectHtml('', sourceGroupsFor(row, {
                catalogKnown: (sources?.read_state ?? catalog.sources_read_state) === 'ok', accountsKnown: store.accountsKnown,
                hasConfiguredAccounts: accountRows(store.snapshot || {}).length > 0,
            }), sourceChoice(row.source));
            const options = sourceHtml.slice(sourceHtml.indexOf('>') + 1, sourceHtml.lastIndexOf('</select>'));
            if (source.innerHTML !== options) source.innerHTML = options;
            const account = node.querySelector('[data-model-role-account]');
            account.hidden = !sourceId(row);
            const accountField = node.querySelector('[data-model-role-account-field]');
            if (accountField) accountField.hidden = account.hidden;
            const credentialHarness = descriptors().find((entry) => entry.id === sourceId(row))?.credentialHarness || '';
            const accountHtml = selectHtml('', [{ options: profileOptionsFor(profiles[credentialHarness], row.account, { accountsKnown: store.accountsKnown && Boolean(credentialHarness), labelled: Boolean(only) }) }], row.account);
            const accountOptions = accountHtml.slice(accountHtml.indexOf('>') + 1, accountHtml.lastIndexOf('</select>'));
            if (account.innerHTML !== accountOptions) account.innerHTML = accountOptions;
            updateModelChooser(node.querySelector('[data-model-role-model]'), catalogModelOptions(itemsFor(row).map((item) => ({
                ...withoutPrice(item, { tiered: tieredProcessing(row), media: row.slot.slot === 'vision' }),
                value: parseModelSource(item.value || item.id).model,
            }))), row.local ? [] : catalogPrograms(catalogs.get(catalogKey(row)), {source: sourceId(row), profile: row.account, sources: descriptors()}));
            if (showContext) node.querySelector('[data-model-context-note]').textContent = modelContextNote(row.windowPreview);
            const heldTier = heldOption(row, 'processing_preference');  // a Safety Supervisor's from the role it uses
            const effectiveTier = row.processing_preference || heldTier || (row.slot.slot === 'fallback' ? mainTier() : processingPreference);
            if (showContext) {
                const tiers = rowTiers(row);
                const select = node.querySelector('[data-model-role-processing]');
                const field = node.querySelector('[data-processing-field]');
                const inherited = row.slot.slot === 'fallback' ? [effectiveTier, processingPreference] : [effectiveTier];
                const focused = select === getDoc()?.activeElement;
                const hidden = Boolean(only && redundantStandard(row.processing_preference, inherited, tiers || undefined));
                if (hidden && focused) field.closest('details')?.querySelector('summary')?.focus();
                field.hidden = hidden;
                node.querySelector('[data-processing-summary]').textContent = processingIntentLabel(row.processing_preference, effectiveTier);
                // A role's page offers what its route says; the whole list of roles keeps each role's tiers as they are.
                const html = processingSelectHtml('', row.processing_preference, { choices: tierOptions(row, only ? tiers : null) });
                const options = html.slice(html.indexOf('>') + 1, html.lastIndexOf('</select>'));
                if (drawnTiers.get(select) !== options) { drawnTiers.set(select, options); select.innerHTML = options; }
                // The tier the role runs on, its own or the one it inherits, as its route says it (and its price by key).
                const note = node.querySelector('[data-processing-capability]');
                note.textContent = field.hidden ? translateString('Standard is the only listed tier for this route.')
                    : tiers && (effectiveTier || tiers.said.lane === 'api') ? tiers.explain(effectiveTier) : '';
                note.dataset.i18nSkip = '';  // composed through tr()/fmt(): it names a model and its source
                // Asked again when the held answers were dropped for another connection under the same value.
                const asking = `${tiers?.route.model}|${tierStore.detailEpoch?.() ?? ''}`;
                if (only && tiers && currentItem(row) && row.tierAsked !== asking) asked = [row, tiers.route, asking];
            }
            const status = node.querySelector('[data-model-role-status]');
            const rowCatalog = catalogs.get(catalogKey(row));
            // A subscription row reports its own account read in full; an API row
            // sits under the section banner, so it keeps to the compact form. An
            // API row whose own catalog (OpenRouter's) states a price says it.
            const own = Boolean(rowCatalog);
            const sourceIssue = sources?.errors?.some(error => error.code === 'scoped_catalog_unsupported')
                ? 'This core lists all models together. Press Read again in Models. '
                : sources && ['failed', 'transport'].includes(sources.read_state) ? 'Subscription sources could not be read. ' : '';
            const note = catalogReadNote(own ? rowCatalog : catalog, { compact: !sourceId(row) });
            // A model stored with its ` (local)` label runs on the local lane, as the runtime reads it (slot_lane_label).
            const local = row.local || / \(local\)$/.test(String(row.model || ''));
            const choosing = !local && Boolean(choiceNote(row)) && keyNote(row) === choiceNote(row);
            status.classList.toggle('ui-status', choosing);
            if (choosing) status.dataset.tone = 'warn'; else delete status.dataset.tone;
            const catalogPrice = local || keyNote(row) || note || !row.model || sourceId(row) || tieredProcessing(row) ? ''
                : modelPriceWords(withoutPrice(currentItem(row), { media: row.slot.slot === 'vision' }), { unit: true });
            const price = only && catalogPrice && !effectiveTier && effectiveSource(row).split('@')[0] === 'openrouter'
                ? fmt('Catalog rate · {price}', { price: catalogPrice }) : catalogPrice;
            const unlisted = rowCatalog?.read_state === 'ok' && !rowCatalog.items.length;
            const plain = local ? 'Uses the local runtime.'
                : keyNote(row) || note || (!row.model ? (row.slot.slot === 'main' ? 'Choose a model to continue.' : `Uses ${inheritName(row)}.`)
                    : sourceId(row) ? (unlisted ? 'No models were listed for this source. Your selection is kept.'
                        : 'Uses your subscription. No API key required.') : price || PRICE_UNKNOWN);
            // The model's window beside how it is paid, in the model list's words (DESIGN "Onboarding density"): the
            // window this route is sized with when the server knows one, else the one its catalog publishes; an
            // unknown window is omitted. Where it comes from stays in the fold below.
            const routed = !local && !keyNote(row) && !note && Boolean(row.model);  // the line is about the model's own route
            // A local role may keep a cloud id in its slot: its window is the local model's own row, never that id's.
            const listed = local ? (catalogs.get('provider_id=local')?.items ?? catalog.items).find(item => item.use_local) : currentItem(row);
            const size = local || (routed && !(sourceId(row) && unlisted)) ? windowWords(row.windowPreview?.window?.tokens
                || listed?.max_context_window || listed?.context_window) : '';
            const sized = size && fmt('{window} window', { window: size });
            const modelFacts = !sized ? plain : routed && !sourceId(row) ? `${price || translateString(PRICE_UNKNOWN)} · ${sized}`
                : `${translateString(plain)} ${sized}.`;
            status.textContent = (translateString(sourceIssue.trim()) + ' ' + modelFacts).trim();
            // A price or a window is composed through the translation seam: the overlay leaves such a line alone.
            if (price || sized) status.dataset.i18nFmt = 'model price'; else delete status.dataset.i18nFmt;
            const errors = validationAttempted ? rowErrors(row) : [];
            const message = node.querySelector('[data-model-role-error]');
            Object.assign(message, { textContent: errors.map((error) => error.message).join(' '), hidden: !errors.length });
            for (const selector of ['[data-model-role-model]', '[data-model-role-context]']) {
                const field = node.querySelector(selector);
                field?.setAttribute('aria-describedby', `${status.id} ${message.id}`);
                field?.setAttribute('aria-invalid', String(errors.some((error) => error.field === selector)));
            }
        }
        if (asked) { asked[0].tierAsked = asked[2]; void tierStore.readDetail(asked[1]); }
    }

    function refreshRow(row, { force = false, joinPending = false } = {}) {
        if (destroyed) return;
        const key = catalogKey(row);
        if (!key) return;
        if ((row.local || / \(local\)$/.test(row.model)) && readApiCatalog && apiCatalogAllowed())
            void catalogs.read('provider_id=local', { force, joinPending });
        if (!sourceId(row) && (!readApiCatalog || !apiCatalogAllowed()
            || !apiProviders.some(provider => provider.id === effectiveSource(row)))) return;
        return catalogs.read(key, { force, joinPending });
    }

    async function responseLimit(row, node, apply = false) {
        // A queued disclosure preview must not supersede an explicit Apply.
        if (!apply && row.responseApplying) return;
        const field = node.querySelector('[data-response-limit]');
        const note = node.querySelector('[data-response-note]');
        const value = Number(field.value || 0);
        if (apply && (field.validity.badInput || !Number.isSafeInteger(value) || value < 0)) {
            note.textContent = 'Max output tokens must be a positive whole number, or Auto.';
            field.setAttribute('aria-invalid', 'true'); return;
        }
        const model = responseModel(row);
        if (!model || model.endsWith('::')) { note.textContent = 'Choose a model to check its max output tokens.'; return; }
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
            if (!data.response_limit || !data.route) throw new Error('Max output tokens are not checked');
            row.windowPreview = data;
            node.querySelector('[data-model-context-note]').textContent = modelContextNote(data);
            updateCatalogViews();  // the role line says this window
            node.querySelector('[data-window-clear]').hidden = !data.window_ack;
            let evidence = data.response_limit;
            if (apply) {
                const result = await apiClient.ownerCapabilityAck({ ...data.route, route_fp: evidence.route_fp, max_output_tokens: value });
                if (!current()) return;
                if (!result.ok) throw new Error(result.error || 'Max output tokens could not be saved');
                evidence = result.ack;
                saved = true;
            }
            // Edits can precede the deferred toggle which starts this read.
            if (!apply && !responseEdits.has(field) && field.value === draft) field.value = evidence.source === 'owner_ack' ? String(evidence.max_output_tokens || '') : '';
            field.setAttribute('aria-invalid', 'false');
            const source = evidence.source === 'owner_ack' ? 'your maximum' : 'model metadata';
            const endpoint = data.route.base_url ? new URL(data.route.base_url) : null;
            const server = endpoint ? ` Server: ${endpoint.host}${endpoint.pathname.replace(/\/$/, '')}.` : '';
            note.textContent = `${evidence.max_output_tokens && !evidence.stale ? `${evidence.max_output_tokens.toLocaleString('en-US')} tokens · ${source}` : 'Auto: max output tokens unknown'}.${server} Context window is separate.${data.route.provider === 'claudexor' ? ' Planning allowance only; the native engine controls its response.' : ''}`;
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
                changed(row.slot.slot); updateCatalogViews(); void refreshRow(row);
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
                for (const item of visibleRows()) void refreshRow(item);
            });
            node.querySelector('[data-model-role-account]').addEventListener('change', (event) => {
                row.account = event.target.value; changed(row.slot.slot); updateCatalogViews(); void refreshRow(row);
            });
            node.querySelector('[data-model-role-context]')?.addEventListener('input', (event) => {
                row.context = event.target.value; changed(row.slot.slot); updateCatalogViews();
            });
            node.querySelector('[data-model-role-processing]')?.addEventListener('change', async (event) => {
                const picked = event.target.value;
                if (picked && !tierOptions(row, only ? rowTiers(row) : null).some((option) => option.value === picked && !option.disabled)) {
                    event.target.value = row.processing_preference;
                    return;
                }
                // Main's Ultrafast asks first; a cancel leaves the draft, and its select, as they were.
                if (picked === 'ultrafast' && row.slot.slot === 'main' && !(await confirmUltrafast(row))) {
                    if (!destroyed) event.target.value = row.processing_preference;
                    return;
                }
                row.processing_preference = picked; changed(row.slot.slot); updateCatalogViews();
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
                changed(); updateCatalogViews(); for (const row of visibleRows()) void refreshRow(row);
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
            // A connection read anew under another revision: the transport limits held for it are not this one's.
            const revisionsBefore = loaded ? JSON.stringify(settingsRevisions) : null;
            settingsRevisions = (Array.isArray(settings._meta?.connections) ? settings._meta.connections : []).map((row) => [row?.id, row?.revision]);
            if (revisionsBefore !== null && revisionsBefore !== JSON.stringify(settingsRevisions)) {
                const before = new Map(JSON.parse(revisionsBefore));
                const changed = new Set(settingsRevisions.filter(([id, revision]) => before.get(id) !== revision).map(([id]) => id)
                    .concat([...before.keys()].filter((id) => !settingsRevisions.some(([seen]) => seen === id))));
                catalog = dropTransport(catalog, changed);
                for (const id of changed) catalogs.invalidate(modelSlotCatalogQuery(id));
                // A read already out answers for the earlier source and is not adopted; a role's page reads anew.
                invalidateModelCatalogReads();
                if (only) void refreshModelCatalog();
            }
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
            loaded = true;
            // A kept draft includes the browser's unfinished number, selection and open Details, not just row values.
            if (at >= 0 && host()?.querySelector(`[data-model-role="${kept.id}"]`)) {
                updateCatalogViews();
                void refreshRow(kept);
            } else render();
        },
        mount() {
            getDoc()?.addEventListener?.('harness-maintenance:settled', onMaintenanceSettled);
            // The prices are composed words the overlay leaves alone: a language change says them again.
            globalThis.window?.addEventListener?.('ouro:language-changed', updateCatalogViews);
            if (!disposeStatus) disposeStatus = bindStatusSurface(store, { elementId: hostId, doc: getDoc,
                listener: updateCatalogViews });
            // A role's page hears its model's own endpoints arrive.
            if (only && showContext && !disposeTiers) disposeTiers = tierStore.subscribe(updateCatalogViews);
            render();
        },
        refreshCatalogs,
        resetCatalogs() { catalogs.reset(); sources = null; connectionLists = {}; },
        adoptSources(data) { sources = data; updateCatalogViews(); },
        adoptCatalog(data = {}, { connectionCatalogs } = {}) {
            // The wizard supplies its already-merged general snapshot plus explicitly read connection lists.
            connectionLists = connectionCatalogs || {};
            catalog = connectionCatalogs ? withConnectionCatalogs(data, connectionCatalogs) : mergeModelCatalog(catalog, data);
            // An updated general read cannot stand in for the selected account. Re-read just the visible
            // subscription scope when its own evidence failed or belongs to another identity.
            if (Array.isArray(data.items)) {
                for (const row of visibleRows()) {
                    const own = catalogs.get(catalogKey(row));
                    if (!sourceId(row) || !own) continue;
                    const tiers = rowTiers(row);
                    if (!own.current || (tiers && tiers.index !== indexOf(own)))
                        void refreshRow(row, { force: true, joinPending: true });
                }
            }
            updateCatalogViews();
        },
        /** A rejected pinned binding retires that scope's held evidence, not the saved assignment. A request
         *  started before the refusal cannot put it back; the next visit to the scope reads it again. */
        invalidateAccountCatalog({ value, account }) {
            const key = modelSlotCatalogQuery(parseModelSource(value).source, account);
            if (catalogs.invalidate(key)) {
                const row = visibleRows().find(candidate => catalogKey(candidate) === key);
                if (row) void refreshRow(row);
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
                errors: rowErrors(row).map(({ message }) => message), slot: row.slot,
                // the revision of the key its route sends to, as this page shows it ('' none, null not named)
                revision: sourceRevision(row), tierBinding: row.processing_preference === 'ultrafast' ? rowBinding(row) : '' };
        },
        /** Whether a Save must first ask about Main's Ultrafast: the draft holds it for a route nobody was asked about
         *  (the model or the account changed since the question, or it came past it), unless it is the stored one,
         *  left as `read` (the role's `entry()` as read) had it. `confirmUltrafast()` is that question. */
        ultrafastUnasked(read = null) {
            const row = only && loaded ? soleRow() : null;
            if (!row || row.slot.slot !== 'main' || row.processing_preference !== 'ultrafast') return false;
            const entry = this.entry();
            if (read && ['value', 'account', 'local', 'processing'].every((key) => read[key] === entry[key])) return false;
            return row.ultrafastFor !== tierIdentity(row) || (read?.processing !== 'ultrafast' && !canChooseTier('ultrafast', rowTiers(row) || {}));
        },
        confirmUltrafast: (carry = false) => confirmUltrafast(soleRow(), carry),
        /** Whether the role's route now offers Ultrafast to be asked about (so `confirmUltrafast()` opens its
         *  question), and what that route says of it. */
        ultrafastOffered: () => Boolean(only && loaded && soleRow() && canChooseTier('ultrafast', rowTiers(soleRow()) || {})),
        ultrafastWords: () => (only && loaded && soleRow() && rowTiers(soleRow())?.explain('ultrafast')) || '',
        noteSaveAttempt() { validationAttempted = true; updateCatalogViews(); },
        validate() {
            return validateAll()[0] || '';
        },
        destroy() {
            getDoc()?.removeEventListener?.('harness-maintenance:settled', onMaintenanceSettled);
            globalThis.window?.removeEventListener?.('ouro:language-changed', updateCatalogViews);
            destroyed = true; disposeStatus?.(); disposeStatus = null;
            disposeTiers?.(); disposeTiers = null;
            disposeChoosers();
            catalogs.dispose();
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
 *  Outcome of the last write; `partly` marks a second write that did not land after the first did. `unless` names
 *  keys that are not written and must still be as `fresh` read them (their tokens go as conditions of each write,
 *  ARCHITECTURE §7: a condition is checked per key under the lock): changed meanwhile, the write is refused. `held`
 *  are conditions the page took when it showed the choice (the connection a route was confirmed on), sent as shown. */
export async function writeModelKeys(id, body, fresh, { ask, unless = [], held = {} } = {}) {
    const changed = Object.keys(body).filter((key) => canon(key, body[key]) !== canon(key, fresh[key]));
    const parts = [['checker', changed.filter((key) => SAFETY_MODEL_KEYS.includes(key))], ['generic', changed.filter((key) => !SAFETY_MODEL_KEYS.includes(key))]];
    let last = null;
    for (const [writer, keys] of parts) {
        if (!keys.length) continue;
        const tokens = Object.fromEntries([...keys, ...unless.filter((key) => !changed.includes(key))]
            .map((key) => [key, fresh._meta?.tokens?.[key]]).filter(([, token]) => token).concat(Object.entries(held)));
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

const slotsOf = (settings) => (Array.isArray(settings?._meta?.setup_contract?.modelSlots) ? settings._meta.setup_contract.modelSlots : []);
const wordsOf = (id) => HAND_WRITTEN.find((row) => row.id === id) || { id, keys: [], name: id, meta: '' };
const shortModel = (model) => String(model || '').slice(String(model || '').lastIndexOf('/') + 1);

/** A stored route in the owner's words: `{ name, source, mark, label }`; an empty one follows another role
 *  (`follows`). A subscription's name comes from the held catalogue's descriptor when one exists — the opaque
 *  source id is no readable name — and `label` is that name alone, so a caller can mark it authored
 *  (`data-i18n-authored`) while the interface words around it stay translatable. Without a descriptor the old
 *  generic label stands. No read starts here. */
export function routeWords(value, { local = false, account = '', follows = 'Main', modelSources = [] } = {}) {
    const { source, model } = parseModelSource(value);
    if (local) return { name: 'Local model', source: 'On this computer', mark: '', label: '' };
    if (!String(value || '').trim()) return { name: `${follows} model`, source: '', mark: '', label: '', follows: true };
    const harness = source.startsWith('subscription:') ? source.slice(13) : '';
    const kind = source.split('@')[0];
    if (!harness) return { name: shortModel(model), mark: '', label: '',
        source: `${API_PROVIDER_LABELS[kind] || kind}${source.includes('@') ? ` (${source.split('@')[1]})` : ''} · by API key` };
    // Only a descriptor names the credential harness; the opaque id itself never picks a brand (sourceMark's rule).
    const label = (modelSources || []).find((entry) => entry?.id === harness)?.label || harnessPresentation(harness).label;
    return { name: shortModel(model), mark: harness, label,
        source: `${label} subscription · ${account ? `account ${account}` : 'your accounts in turn'}` };
}

// The mark of the source a route goes through: a subscription's own, a provider's own, else its initial; a local
// model and an unset route have none. Never a generic glyph (a globe read as "the internet", owner 13.11).
function routeMark(value, { local = false, modelSources = [] } = {}) {
    if (local || !String(value || '').trim()) return '';
    const { source } = parseModelSource(value);
    if (source.startsWith('subscription:')) {
        const sourceId = source.slice(13);
        const descriptor = (modelSources || []).find((entry) => entry?.id === sourceId);
        const label = descriptor?.label || sourceId;
        return sourceMark({ lane: 'subscription', credentialHarness: descriptor?.credentialHarness || '', label });
    }
    const kind = source.split('@')[0];
    return sourceMark({ kind, label: API_PROVIDER_LABELS[kind] || kind });
}
const CONTEXT_WORDS = { nano: 'Nano', low: 'Low', max: 'Max' };
const SEARCH_SOURCES = [['auto', 'Auto'], ['openai', 'OpenAI'], ['openrouter', 'OpenRouter'], ['anthropic', 'Anthropic'],
    ['ddgs', 'DuckDuckGo'], ['deepseek', 'DeepSeek']];

function mountModels(container, { path, ctx }) {
    const custom = (row) => `<button type="button" class="btn btn-default btn-sm" id="btn-refresh-model-catalog" data-models-action>${escapeHtml(row.control.label)}</button>`;
    const doors = [...ROLE_ORDER, 'websearch'].map((slot) => ({ ...wordsOf(`models:${slot}`), control: { kind: 'door', to: `models/role/${slot}` } }));
    const door = (row) => settingsRow(row).replace('data-setting-door', `data-setting-door data-entry-id="${row.id.split(':')[1]}"`);
    const plain = rowsOf('models', { fold: false }).map((row) => settingsRow(row, { custom })).join('');
    // The catalogue a model sheet already holds: its descriptors brand the routes they name, and no read of its
    // own starts for a mark (the home's Models door reads the same held list, settings_home.js).
    const heldSources = () => {
        const sources = modelCatalog.catalog()?.model_sources;
        return Array.isArray(sources) ? sources : [];
    };
    // A subscription's name is an authored label (a product's or the owner's); the words around it are chrome.
    // Two elements, never one text beside a child: a translation may rewrite the words, never the name.
    const sourceWordsHtml = (words) => (words.label
        ? `<span data-i18n-authored>${escapeHtml(words.label)}</span><span>${escapeHtml(words.source.slice(words.label.length))}</span>`
        : escapeHtml(words.source));
    container.innerHTML = `<p class="settings-page-lead">Which model does which work.</p>
        <section class="ui-card settings-card" data-setting-row="models:main">
            <h3 class="settings-card-title">${escapeHtml(wordsOf('models:main').name)}</h3>
            <p class="settings-card-lead">${escapeHtml(wordsOf('models:main').meta)}</p>
            <div class="settings-rows models-main" data-entry-id="main"></div>
            <p class="settings-card-note">${escapeHtml(MAIN_FOLLOW_NOTE)}</p>
        </section>
        <section class="ui-card settings-card" data-setting-row="models:fallbacks">
            <div class="settings-card-head"><h3 class="settings-card-title">Fallback models</h3>
                <span class="ui-result" data-entry-result role="status" aria-live="polite"></span>
                <div class="ui-section-actions"><button type="button" class="btn btn-default btn-sm" data-fallback-add>Add fallback</button></div></div>
            <p class="settings-card-lead">${escapeHtml(wordsOf('models:fallbacks').meta)}</p>
            <div class="settings-rows" data-models-fallbacks></div>
        </section>
        <section class="ui-card settings-card"><h3 class="settings-card-title">Other roles</h3>
            <p class="settings-card-lead">Each uses Main until you give it a model of its own.</p>
            <div class="settings-rows">${doors.map(door).join('')}</div></section>
        <section class="ui-card settings-card"><div class="settings-rows">${plain}</div></section>
        <section class="ui-card settings-card"><details class="ui-disclosure" data-settings-fold>
            <summary>More: processing, prompt cache, requests at once</summary>
            <div class="settings-rows" data-models-fold></div></details></section>`;
    // The catalogue's own result (settings_catalog.js writes it by this id) stands under its row's words and door, as
    // every row's result does (settings.css "A row's result").
    container.querySelector('[data-setting-row="models:catalogue"]')
        ?.appendChild(Object.assign(container.ownerDocument.createElement('span'), { id: 'settings-model-catalog-status', className: 'ui-result settings-row-result' }));
    const bound = bindSettingRows(container, { rows: doors, ask: ctx.ask, custom, openDoor: (row) => ctx.openSettings(row.control.to) });
    const foldHost = container.querySelector('[data-models-fold]');
    const entryResult = container.querySelector('[data-entry-result]');
    let fold = null;
    let shown = '', drawnFallbacks = '';   // the fallbacks as drawn: a press acts on these, never on a list that changed meanwhile
    let mainPainted = null;                // Main's route as last drawn: { value, account, local, note }
    let mainMetaShown = null;              // the meta line last given to the DOM (a late descriptor compares against it)
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
    // Main's two catalogue-decided slots: the mark of its source and the source line beside it. Only these change
    // when a descriptor arrives late; an unchanged result writes nothing (and the card is never redrawn for it).
    const mainMetaHtml = (value, account, local, note) => {
        if (note) return `<span>${escapeHtml(note)}</span>`;
        const words = routeWords(value, { account, local, modelSources: heldSources() });
        return `${routeMark(value, { local, modelSources: heldSources() })}<span>${sourceWordsHtml(words)}</span>`;
    };
    function paintMainMeta() {
        if (!mainPainted) return;
        const meta = container.querySelector('.models-main-meta');
        if (!meta) return;
        const html = mainMetaHtml(mainPainted.value, mainPainted.account, mainPainted.local, mainPainted.note);
        if (html === mainMetaShown) return;
        meta.innerHTML = mainMetaShown = html;
    }
    const paint = (settings) => {
        const slots = Object.fromEntries(slotsOf(settings).map((slot) => [slot.slot, slot]));
        const accounts = modelRoleMap(settings[MODEL_ACCOUNTS_KEY]);
        const local = (slot) => Boolean(slots[slot]?.settingsToggleId) && truthy(settings[`USE_LOCAL_${slot.toUpperCase()}`]);
        const route = (slot, follows) => routeWords(settings[slots[slot]?.settingKey], { follows, account: accounts[slot],
            local: local(slot), modelSources: heldSources() });
        const profiles = settings._meta?.setup_contract?.providerProfiles || {};
        const keyWords = (value, onLocal) => {
            const problem = onLocal ? '' : routeKeyProblem(value, settings, profiles);
            return problem ? keyProblemLine(problem, apiProviderLabel(parseModelSource(value).source, profiles, apiConnections(settings, profiles))) : '';
        };
        // Several keys of Main's provider and no built-in one: Main waits for the owner's choice (#2969), said here too.
        const choice = Array.isArray(settings._meta?.main_connection_choice) ? settings._meta.main_connection_choice : [];
        const labels = choice.length > 1 ? choice.map((id) => apiConnections(settings, profiles).find((entry) => entry.id === id)?.label || id) : [];
        const main = route('main');
        // An empty Safety Supervisor names what it checks with now: Light's model, else Main's, with the account and
        // tier the runtime takes for it along the same chain (Safety → Light → Main).
        const safetyWords = () => {
            const valueOf = (slot) => String(settings[slots[slot]?.settingKey] ?? '').trim();
            if (valueOf('safety')) return route('safety');
            const chain = ['safety', 'light', 'main'];
            const at = chain.findIndex((slot) => slot === 'main' || valueOf(slot));
            const entries = chain.map((slot) => ({ slot, set: Boolean(valueOf(slot)) }));
            const option = (key) => String(safetyChain(entries, (entry) => modelRoleMap(settings[key])[entry.slot]) || '');
            // Light's route, its box included; an empty Light's box over a cloud Main puts Main's model on the local model.
            const onLocal = at === 1 ? local('light') : local('light') || local('main');
            const words = routeWords(valueOf(chain[at]), { local: onLocal });
            const [account, tier] = [option(MODEL_ACCOUNTS_KEY), option(MODEL_PROCESSING_PREFERENCES_KEY)];
            return { follows: true, name: [words.name, account && !onLocal && parseModelSource(valueOf(chain[at])).source.startsWith('subscription:')
                ? `account ${account}` : '', tierNamed(tier) ? tierWord(tier) : '', at === 1 ? 'as Light' : 'as Main, via Light'].filter(Boolean).join(' · ') };
        };
        const mainNote = labels.length ? `Choose the key it uses: ${labels.slice(0, -1).join(', ')} or ${labels.at(-1)}.`
            : keyWords(settings[slots.main?.settingKey], local('main'));
        mainPainted = { value: String(settings[slots.main?.settingKey] || ''), account: accounts.main || '', local: local('main'), note: mainNote };
        mainMetaShown = mainMetaHtml(mainPainted.value, mainPainted.account, mainPainted.local, mainPainted.note);
        // The answer is one row: the model, under it the source with its own mark (or what is wrong with its key), and
        // the chevron of a row that opens a level. The role's name is the object's title above.
        container.querySelector('.models-main').innerHTML = `<button type="button" class="ui-row settings-row models-main-row" data-kind="door" data-entry-open${mainNote || main.source ? '' : ' data-no-meta'}>
            <span class="ui-row-title models-main-name">${escapeHtml(main.follows || labels.length ? 'Not chosen yet' : main.name)}</span>
            <span class="ui-row-meta models-main-meta"${mainNote ? ' data-tone="warn"' : ''}>${mainMetaShown}</span>
            <span class="ui-row-actions">${doorSign('models/role/main')}</span></button>`;
        const list = fallbackEntries(settings);
        shown = list.map((entry) => entry.value).join(',');
        const localFallback = truthy(settings.USE_LOCAL_FALLBACK);
        const fallbackHtml = list.map((entry, index) => {
            const words = routeWords(entry.value, { account: entry.account, local: localFallback, modelSources: heldSources() });
            const note = keyWords(entry.value, localFallback);
            return `<div class="ui-row settings-head" data-entry-id="fallback/${index + 1}" data-actions>
                <span class="settings-head-glyph settings-head-order" aria-hidden="true">${index + 1}</span>
                <button type="button" class="settings-head-open ui-hit-area" data-entry-open><span class="ui-row-title">${escapeHtml(words.name)}</span><span class="ui-row-meta"${note ? ' data-tone="warn"' : ''}>${note ? escapeHtml(note) : sourceWordsHtml(words)}</span></button>
                <div class="ui-row-actions"><button type="button" class="btn btn-ghost btn-sm" data-fallback-move="-1" aria-label="Move fallback ${index + 1} up" ${index ? '' : 'disabled'}>↑</button>
                    <button type="button" class="btn btn-ghost btn-sm" data-fallback-move="1" aria-label="Move fallback ${index + 1} down" ${index < list.length - 1 ? '' : 'disabled'}>↓</button>
                    <button type="button" class="btn btn-ghost btn-sm" data-fallback-remove>Remove</button>${doorSign()}</div></div>`;
        }).join('') || '<p class="settings-card-lead">No fallback yet.</p>';
        if (fallbackHtml !== drawnFallbacks) container.querySelector('[data-models-fallbacks]').innerHTML = drawnFallbacks = fallbackHtml;
        for (const slot of ROLE_ORDER) {
            const words = slot === 'safety' ? safetyWords() : route(slot, 'Main');
            say(`models:${slot}`, words.name, words.follows);
            warn(`models:${slot}`, keyWords(settings[slots[slot]?.settingKey], local(slot)));
        }
        say('models:websearch', Object.fromEntries(SEARCH_SOURCES)[settings.OUROBOROS_WEBSEARCH_BACKEND || 'auto'] || String(settings.OUROBOROS_WEBSEARCH_BACKEND));
        say('models:context', CONTEXT_WORDS[String(settings.OUROBOROS_CONTEXT_MODE || '').toLowerCase()] || String(settings.OUROBOROS_CONTEXT_MODE || ''));
        // The fold's rows are drawn once their values are read.
        if (!fold) {
            const rows = rowsOf('models', { fold: true });
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
    const onShown = (event) => { if (event.detail?.path === path) void read(); };
    const onSaved = (event) => { if (event.detail?.source === 'settings-row') void read(); };
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    globalThis.window?.addEventListener('ouro:settings-updated', onSaved);
    // A descriptor adopted after the row was drawn (a model sheet's read, Read again) brands the Main route it
    // names: only that row's mark and source line change, and the route drawn from the same settings stays the one
    // compared, so a different Main alone reaches this row through its own read.
    const offCatalog = watchModelCatalog(paintMainMeta);
    void read();
    return { unmount() {
        container.removeEventListener('click', onClick);
        globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown);
        globalThis.window?.removeEventListener('ouro:settings-updated', onSaved);
        offCatalog();
        bound.dispose(); fold?.dispose();
    } };
}

// Where a settings document has a role bound: each field of the role's form (`entry()`) beside the key that stores
// it, and that key's stored value for the role.
const ROUTE_FIELDS = [['value', (slot) => slot.settingKey], ['local', (slot) => `USE_LOCAL_${slot.slot.toUpperCase()}`], ['account', () => MODEL_ACCOUNTS_KEY]];
const boundRoute = (settings, slot) => ({ value: String(settings[slot.settingKey] ?? ''),
    local: truthy(settings[`USE_LOCAL_${slot.slot.toUpperCase()}`]), account: String(modelRoleMap(settings[MODEL_ACCOUNTS_KEY])[slot.slot] || '') });

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
    const editor = createModelRolesEditor({ hostId, supportsScopes: doc => doc._meta?.setup_contract?.scopedModelCatalogs === true, readApiCatalog: fetchJson, mainNote: true, only: { slot, index, actions: SAVE_ROW },
        // The page's own confirming seam (the shared dialog), as its other questions use.
        askUltrafast: (facts) => confirmMainUltrafast({ ...facts, dialog: ctx.ask ? (question) => ctx.ask(question) : undefined }),
        onChange: () => { dirty = true; container.querySelector('[data-editor-discard]')?.removeAttribute('hidden'); setResult(container.querySelector('[data-editor-result]'), {}); } });
    const say = (words) => setResult(container.querySelector('[data-editor-result]'), words);
    // A typed form keeps its fields while the page around them follows the read (a key added in Accounts, a new Main);
    // the fallback list a Save checks stays the one the typing began from. Nothing is drawn while the owner is in it.
    const typing = () => dirty && container.contains(globalThis.document?.activeElement);
    let asRead = null;  // the role as read when the typing began: a Save writes only what the owner changed from it
    let readRoute = null;  // and where that read had the role bound (`boundRoute`)
    const read = async () => {
        if (typing()) return;
        let settings;
        try { settings = await ctx.readSettings(); } catch { return; }
        if (typing()) return;
        if (!dirty) readFallbacks = fallbackEntries(settings).map((entry) => entry.value).join(',');
        editor.load(settings, { ...settings._meta?.setup_contract, modelSlots: slotsOf(settings).map((each) => ({ ...each, inputId: each.settingsInputId })) }, { keep: dirty });
        if (dirty) container.querySelector('[data-editor-discard]')?.removeAttribute('hidden');  // drawn again, still unsaved
        else { asRead = editor.entry(); readRoute = asRead && boundRoute(settings, asRead.slot); }
    };
    const save = async (button) => {
        editor.noteSaveAttempt();
        const entry = editor.entry();
        if (!entry || !asRead || entry.errors.length) { say({ text: entry?.errors[0] || 'This role is not shown yet.', tone: 'danger' }); return; }
        // Main's Ultrafast is asked for the route it is saved with: once at its choice, and here only when the model
        // or the account changed since; one stored and left as read never asks.
        setButtonBusy(button, true);
        let outcome = { state: 'changed_elsewhere', effect: 'now', withheld: '', message: '', code: '' };
        let moved = '';  // why the save was not sent: Main moved under this page (its words)
        try {
            if (editor.ultrafastUnasked(asRead)) {
                const carry = asRead.processing === 'ultrafast';
                if (carry || editor.ultrafastOffered()) {
                    if (!(await editor.confirmUltrafast(carry))) {
                        setButtonBusy(button, false);
                        say({ text: 'Not saved: Ultrafast was not confirmed.', tone: 'muted' });
                        return;
                    }
                } else {
                    setButtonBusy(button, false);
                    say({ text: `Not saved: Ultrafast is not offered for this model here. ${editor.ultrafastWords()} Choose another tier for Main, or Default.`.replace(/\s+/g, ' '), tone: 'warn' });
                    return;
                }
            }
            const fresh = await ctx.readSettings();
            // A key the typing left as it was read keeps what is stored now: a Main chosen in the composer meanwhile stays.
            const mine = roleBody(fresh, entry, index, readFallbacks);
            const before = mine && asRead ? roleBody(fresh, asRead, index, readFallbacks) : null;
            const body = mine && Object.fromEntries(Object.entries(mine).filter(([key, value]) => !before || canon(key, value) !== canon(key, before[key])));
            // Main's model, Local switch, account and tier are one binding: a save that writes any of them stands on all
            // of them as this page showed them (its read), and on their tokens under the lock (fresh read), so a tier
            // or model chosen elsewhere meanwhile is never carried along unasked. Ultrafast binds its connection too.
            const routeKeys = ROUTE_FIELDS.map(([, keyOf]) => keyOf(entry.slot));
            const binds = Boolean(body) && slot === 'main' && [...routeKeys, MODEL_PROCESSING_PREFERENCES_KEY].some((key) => key in body);
            const ultrafast = entry.processing === 'ultrafast';
            const unless = binds ? [...routeKeys, MODEL_PROCESSING_PREFERENCES_KEY] : [];
            // The key the Ultrafast was shown and confirmed on: its revision as this page showed it, never the fresh
            // read's, so a key, base or surface changed since is refused here or under the lock (CONNECTION_CONDITION).
            const shownSource = binds && ultrafast ? entry.revision : '';
            if (binds) {
                const now = boundRoute(fresh, entry.slot);
                const currentTier = modelRoleMap(fresh[MODEL_PROCESSING_PREFERENCES_KEY]).main || '';
                const draft = editor.entry();
                if (currentTier !== asRead.processing && currentTier !== entry.processing) {
                    moved = `Not saved. Main's processing is ${tierWord(currentTier)} now, set on another device. Check it, then save again.`;
                } else if (!draft || ROUTE_FIELDS.some(([field]) => draft[field] !== entry[field]
                        || (now[field] !== readRoute[field] && now[field] !== entry[field]))) {
                    moved = 'Not saved. Main changed on another device. Check the current model and account, then save again.';
                } else if (ultrafast && (draft.processing !== entry.processing || draft.tierBinding !== entry.tierBinding || editor.ultrafastUnasked(asRead))) {
                    moved = 'Not saved. Main changed after confirmation. Check the current model and account, then choose its tier again.';
                } else if (shownSource === null) {
                    moved = 'Not saved. The key this model goes through could not be verified. Reopen the page, then save again.';
                } else if (shownSource && revisionIn(fresh, { model: entry.value, use_local: false }) !== shownSource) {
                    moved = 'Not saved. The key this model goes through changed on another device. Its tiers are read again: check them, then save again.';
                }
                // A field edited here can already equal the fresh GET. It then drops out of the POST body,
                // but its token must still protect the binding, along with the tokens of the fields we write.
                if (!moved && !unless.every((key) => fresh._meta?.tokens?.[key])) {
                    throw new Error('Could not verify Main’s settings. Reopen the page before saving.');
                }
            }
            if (body && !moved) {
                outcome = await writeModelKeys(`models:${slot}`, body, fresh, { ask: ctx.ask, unless,
                    held: { ...(shownSource ? { [CONNECTION_CONDITION]: shownSource } : {}),
                        ...(binds && ultrafast ? { ultrafast: entry.tierBinding } : {}) } });
                if (binds && ultrafast && entry.account && outcome.state === 'changed_elsewhere') {
                    editor.invalidateAccountCatalog(entry);
                    await refreshModelCatalog();  // refreshes the general evidence and tells the role to reread its scope
                }
            }
        } catch (error) { outcome = { state: 'refused', effect: 'now', withheld: '', message: error?.message || String(error), code: '' }; }
        if (landed(outcome) || outcome.state === 'changed_elsewhere' || outcome.state === 'unchanged') { dirty = false; await read(); }
        setButtonBusy(button, false);
        const words = moved ? { text: moved, tone: 'warn' } : saveWords(outcome);
        say(words);
    };
    const onClick = (event) => {
        const hit = (selector) => event.target.closest?.(selector);
        if (hit('[data-editor-save]')) void save(hit('[data-editor-save]'));
        else if (hit('[data-editor-discard]')) { dirty = false; void read(); }
    };
    const accountCatalog = watchAccountModelCatalog(({ firstConfirmation }) =>
        editor.refreshCatalogs({ subscriptionsOnly: true, joinPending: firstConfirmation }));
    let reading = null, shownOnce = false;
    // The shell announces the first opening as it mounts. Join that read; on return or reconnect
    // read settings again even if a catalog from the previous showing is still answering.
    const catchUp = (force = false) => (reading ||= read().then(() => {
        accountCatalog.arm();
        void editor.refreshCatalogs({ force });
    }).finally(() => { reading = null; }));
    const onShown = (event) => {
        if (event.detail?.path !== path) return;
        void catchUp(shownOnce);
        shownOnce = true;
    };
    container.addEventListener('click', onClick);
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    editor.mount();
    void catchUp();
    return { unsaved: () => dirty, unmount() {
        container.removeEventListener('click', onClick);
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
                <span class="ui-field-help">Only the built-in web search uses it. Empty follows the model Ouroboros recommends for each source.</span></label>
            <label class="ui-field" hidden><span>Key</span><select id="s-websearch-key" class="ui-control" hidden></select>
                <span class="ui-field-help">DeepSeek search takes no model: choose the key it uses.</span></label></div>
            <div class="ui-field-help" data-websearch-preview role="status"></div>
            ${SAVE_ROW}</section>`;
    const [source, model, preview] = ['#s-websearch-source', '#s-websearch-model', '[data-websearch-preview]'].map((selector) => container.querySelector(selector));
    // DeepSeek takes no model: under it the Key field, with its own help, stands in for Model (owner decision, 11 Nov 2026).
    const key = container.querySelector('#s-websearch-key');
    let searchKeys = {};  // the DeepSeek keys of Accounts the Key select offers
    let dirty = false;
    let previews = 0;
    const save = container.querySelector('[data-editor-save]');
    const discard = container.querySelector('[data-editor-discard]');
    const formControls = [source, model, key, save, discard].filter(Boolean);
    const paintReadState = (settings = null) => {
        const unread = settings ? '' : settingsReadMessage();
        formControls.forEach((control) => { control.disabled = Boolean(unread); });
        if (unread && !dirty) preview.textContent = unread;
    };
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
    let asRead = null;  // the choice as read when the typing began (null before a read): a Save writes only what changed
    const adopt = (settings) => {
        if (!settings) { source.value = ''; paintReadState(); return; }
        // A typed choice stays, one begun while this read was pending too; the routes it would take and the keys of
        // Accounts (where the Key select's pointer leads) are read again for it. Only the first read fills the fields
        // under typing, so a Save never writes a Source the owner was not shown.
        if (!dirty || !asRead) {
            source.value = settings.OUROBOROS_WEBSEARCH_BACKEND || 'auto';
            model.value = String(settings.OUROBOROS_WEBSEARCH_MODEL || '');
            asRead = { OUROBOROS_WEBSEARCH_BACKEND: source.value, OUROBOROS_WEBSEARCH_MODEL: model.value };
        }
        const profiles = settings?._meta?.setup_contract?.providerProfiles || {};
        searchKeys = { connections: apiConnections(settings, profiles), providers: configuredApiProviders(settings, profiles),
            providerProfiles: profiles };
        syncSearchKey(source, model, key, searchKeys);
        paintReadState(settings);
        void showPreview();
    };
    const read = async () => {
        let settings;
        try { settings = await ctx.readSettings(); } catch { settings = settingsSnapshot(); }
        adopt(settings);
    };
    const say = (words) => setResult(container.querySelector('[data-editor-result]'), words);
    const edited = () => { dirty = true; syncSearchKey(source, model, key, searchKeys);
        container.querySelector('[data-editor-discard]').hidden = false; say({}); void showPreview(); };
    const onClick = async (event) => {
        const button = event.target.closest?.('[data-editor-save]');
        if (event.target.closest?.('[data-editor-discard]')) { dirty = false; container.querySelector('[data-editor-discard]').hidden = true; void read(); return; }
        if (!button) return;
        setButtonBusy(button, true);
        let outcome;
        try {
            const fresh = await ctx.readSettings();
            const wanted = { OUROBOROS_WEBSEARCH_BACKEND: selection().backend, OUROBOROS_WEBSEARCH_MODEL: selection().model };
            const body = Object.fromEntries(Object.entries(wanted).filter(([key, value]) => value !== asRead?.[key]));
            outcome = await writeModelKeys('models:websearch', body, fresh, { ask: ctx.ask });
            if (landed(outcome)) asRead = wanted;  // the new baseline before the reread, for typing that reread keeps
        } catch (error) { outcome = { state: 'refused', effect: 'now', withheld: '', message: error?.message || String(error), code: '' }; }
        setButtonBusy(button, false);
        if (landed(outcome) || outcome.state === 'unchanged') { dirty = false; container.querySelector('[data-editor-discard]').hidden = true; await read(); }
        say(saveWords(outcome));
    };
    const onShown = (event) => { if (event.detail?.path === path) void read(); };
    // Leaving DeepSeek drops only the key spelling written for it; the Key select spells the chosen one.
    source.addEventListener('change', () => { model.value = searchModelAfterSourceSwitch(source.value, model.value); edited(); });
    key?.addEventListener('change', () => { model.value = searchModelForKey(key.value, model.value); edited(); });
    model.addEventListener('input', edited);
    container.addEventListener('click', onClick);
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    adopt(settingsSnapshot());
    void read();
    return { unsaved: () => dirty, unmount() { container.removeEventListener('click', onClick); globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown); } };
}

registerSettingsPage('models', {
    title: (path) => {
        const [, , slot, n] = path.split('/');
        return !slot ? 'Models' : slot === 'fallback' ? `Fallback model ${n || ''}`.trim() : `${wordsOf(`models:${slot}`).name.replace(/ model$/, '')} model`;
    },
    mount: (container, page) => {
        const [, , slot, n] = page.path.split('/');
        return !slot ? mountModels(container, page) : slot === 'websearch' ? mountWebSearch(container, page)
            : mountRole(container, page, slot, slot === 'fallback' ? Math.max(0, Number(n || 1) - 1) : 0);
    },
});
