// Neutral route-editor primitives shared by Available subagents and Review
// lanes. Semantic owners keep their own schemas: reviewer rows serialize
// `api_chat` + `profile_id`; task actors serialize `api_model` +
// `credential_profile_id`.

import { accountRows, accountName } from './claudexor_status_store.js';
import { formatRelativeAge } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { modelChooserHtml, updateModelChooserOptions } from './model_chooser.js';

export const ROUTE_KIND_API_MODEL = 'api_model';
export const ROUTE_KIND_AGENT_SESSION = 'agent_session';
// Legacy single API choice. Retained as an accepted input spelling; every
// encoder now emits the provider-qualified `api:<provider>` form instead.
export const API_ROUTE_CHOICE = 'api';
export const API_CHOICE_PREFIX = 'api:';
export const DEFAULT_API_PROVIDER = 'openrouter';
export const EFFORT_CHOICES = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'];
export const PROCESSING_PREFERENCE_KEY = 'OUROBOROS_PROCESSING_PREFERENCE';
export const MODEL_PROCESSING_PREFERENCES_KEY = 'OUROBOROS_MODEL_PROCESSING_PREFERENCES';
export const PROCESSING_CHOICES = ['standard', 'fast', 'economy'];

export function processingLabel(value) {
    return { standard: 'Standard', fast: 'Fast', economy: 'Economy', mixed: 'Mixed', unknown: 'Unknown' }[value] || 'Route default';
}

export function processingIntentLabel(value, inherited = '') {
    return value ? `${processingLabel(value)} (override)`
        : inherited ? `${processingLabel(inherited)} (from Models)` : 'Route default (inherited)';
}

export function processingSelectHtml(attrs, selected, { global = false } = {}) {
    return selectHtml(attrs, [{ options: [
        { value: '', label: global ? 'Keep route defaults' : 'Use global setting' },
        ...PROCESSING_CHOICES.map((value) => ({ value, label: processingLabel(value) })),
    ] }], selected || '');
}

/** Intent controls never change model/effort or claim the route served this mode. */
export function processingDetailsHtml(attrs, selected, inherited = '') {
    return `<details class="model-role-details" data-processing-details><summary>Processing · <span data-processing-summary>${escapeHtml(processingIntentLabel(selected, inherited))}</span></summary>
        <label class="ui-field">Processing ${processingSelectHtml(attrs, selected)}</label>
        <div class="ui-field-help">Uses the same model and reasoning effort. An explicit native service choice takes precedence.</div></details>`;
}

export function processingCapabilityNote(preference, capability, transportPreferences) {
    if (preference && Array.isArray(transportPreferences) && !transportPreferences.includes(preference)) return `${processingLabel(preference)} processing is not advertised by this transport. Ordinary service remains available.`;
    return preference && Array.isArray(capability?.modes) && !capability.modes.includes(preference)
        ? `${processingLabel(preference)} is not advertised for this route. Ordinary service may be used; the model and effort stay the same.` : '';
}

/** Display only an existing execution receipt, keeping requested and observed separate. */
export function processingExecutionText(processing) {
    if (!processing || !('observed' in processing)) return '';
    const parts = [processing.observed === 'unknown' ? 'Applied processing not reported'
        : `Applied processing: ${processingLabel(processing.observed)}`];
    if (processing.requested) parts.push(`requested ${processingLabel(processing.requested)}`);
    if (processing.submittedNative) parts.push(`submitted service ${processing.submittedNative}`);
    if (processing.reason) parts.push(String(processing.reason));
    return parts.join(' · ');
}

export function parseModelSource(value) {
    const raw = String(value || '').trim();
    if (raw.startsWith('claudexor::')) {
        const target = raw.slice('claudexor::'.length);
        const split = target.indexOf('=');
        return { source: `subscription:${split < 0 ? target : target.slice(0, split)}`,
            model: split < 0 ? '' : target.slice(split + 1) };
    }
    const split = raw.indexOf('::');
    return split < 0 ? { source: 'openrouter', model: raw }
        : { source: raw.slice(0, split), model: raw.slice(split + 2) };
}

export function composeModelSource(source, model) {
    const value = String(model || '').trim();
    if (!value) return '';
    if (value.includes('::')) return value;
    if (source.startsWith('subscription:')) return `claudexor::${source.slice(13)}=${value}`;
    return source === 'openrouter' ? value : `${source}::${value}`;
}

// Owner-facing order of the direct API providers. OpenRouter first because an
// unprefixed model id routes through it; the rest follow the settings order.
export const API_PROVIDER_ORDER = ['openrouter', 'openai', 'anthropic', 'deepseek',
    'zai', 'minimax', 'cloudru', 'gigachat', 'openai-compatible'];

// Fallback names for providers the setup contract does not describe (GigaChat
// has no profile spec). The contract's label wins whenever it exists.
const API_PROVIDER_LABELS = {
    openrouter: 'OpenRouter', openai: 'OpenAI', anthropic: 'Anthropic', deepseek: 'DeepSeek',
    zai: 'Z.ai (GLM)', minimax: 'MiniMax', cloudru: 'Cloud.ru Foundation Models', gigachat: 'GigaChat',
    'openai-compatible': 'OpenAI-compatible endpoint',
};

// Credential shapes that make a provider usable, mirroring the runtime's
// PROVIDER_CREDENTIAL_GROUPS: any ONE group whose every key is present counts.
// Only the fields the transport cannot start without are listed here; optional
// region/scope/base-url companions are not evidence of a missing key.
const API_PROVIDER_CREDENTIALS = {
    openrouter: [['OPENROUTER_API_KEY']],
    openai: [['OPENAI_API_KEY']],
    anthropic: [['ANTHROPIC_API_KEY']],
    deepseek: [['DEEPSEEK_API_KEY']],
    zai: [['ZAI_API_KEY']],
    minimax: [['MINIMAX_API_KEY']],
    cloudru: [['CLOUDRU_FOUNDATION_MODELS_API_KEY']],
    gigachat: [['GIGACHAT_CREDENTIALS'], ['GIGACHAT_USER', 'GIGACHAT_PASSWORD']],
    'openai-compatible': [['OPENAI_COMPATIBLE_BASE_URL'], ['OPENAI_BASE_URL']],
};

/** Every settings key that can make an API provider selectable. */
export const API_PROVIDER_CREDENTIAL_KEYS = [...new Set(
    API_PROVIDER_ORDER.flatMap((id) => API_PROVIDER_CREDENTIALS[id].flat()),
)];

export function apiProviderLabel(id, providerProfiles = {}) {
    const key = String(id || '');
    return String(providerProfiles?.[key]?.label || API_PROVIDER_LABELS[key] || key);
}

/**
 * Providers the owner can actually send to, in owner-facing order.
 * A masked placeholder such as `***set***` is a stored credential; only an
 * absent or blank value withdraws the provider from the list.
 */
export function configuredApiProviders(settings = {}, providerProfiles = {}) {
    const present = (key) => String(settings?.[key] ?? '').trim() !== '';
    return API_PROVIDER_ORDER
        .filter((id) => API_PROVIDER_CREDENTIALS[id].some((group) => group.every(present)))
        .map((id) => ({ id, label: apiProviderLabel(id, providerProfiles) }));
}

/** Account binding is supported by agent sessions and subscription model routes. */
export function routeSupportsAccount(route) {
    return route?.kind === ROUTE_KIND_AGENT_SESSION
        || (['api_model', 'api_chat'].includes(route?.kind)
            && parseModelSource(route?.target_id).source.startsWith('subscription:'));
}

/**
 * Source ids are opaque; only the model-sources envelope names their credential owner.
 * `modelSources` names subscription sources, the options object names API providers.
 * @param {object} route
 * @param {Array<{id:string,label?:string,credentialHarness?:string}>} [modelSources]
 * @param {{providerProfiles?: object}} [options] setup-contract provider profiles, for labels
 * @returns {{source:string, sourceLabel:string, subscription:boolean, model:string,
 *   harness:string, provider:string, providerLabel:string}} `model` is always the model
 *   part alone: the `openai::`/`claudexor::<source>=` prefix lives in `provider`/`source`.
 */
export function routeModelFields(route, modelSources = [], { providerProfiles = {} } = {}) {
    if (route?.kind === ROUTE_KIND_AGENT_SESSION) {
        return { ...splitSessionTarget(route.target_id), subscription: false,
            provider: '', providerLabel: '' };
    }
    const parsed = parseModelSource(route?.target_id);
    const subscription = parsed.source.startsWith('subscription:');
    const source = subscription ? parsed.source.slice(13) : '';
    const descriptor = modelSources.find((entry) => entry.id === source);
    const provider = subscription ? '' : (parsed.source || DEFAULT_API_PROVIDER);
    return { source, sourceLabel: descriptor?.label || source, subscription,
        model: parsed.model,
        provider, providerLabel: provider ? apiProviderLabel(provider, providerProfiles) : '',
        harness: subscription ? String(descriptor?.credentialHarness || '') : '' };
}

/** The editor composes the stored spelling; the owner never types a `::` prefix. */
export function routeTargetFromModel(route, model) {
    if (route?.kind === ROUTE_KIND_AGENT_SESSION) {
        return composeSessionTarget(splitSessionTarget(route.target_id).harness, model);
    }
    const { source, subscription, provider } = routeModelFields(route);
    if (subscription) {
        return composeModelSource(`subscription:${source}`, model) || `claudexor::${source}=`;
    }
    return composeModelSource(provider, model) || emptyApiTarget(provider);
}

/** The transient draft of an API route whose model is still empty. */
function emptyApiTarget(provider) {
    return !provider || provider === DEFAULT_API_PROVIDER ? '' : `${provider}::`;
}

/** A source change clears only source-bound fields, never the caller's delivery kind. */
export function changeRouteChoice(route, choice, { apiKind = ROUTE_KIND_API_MODEL } = {}) {
    if (encodeRouteChoice({ route }) === choice) return { ...route };
    const decoded = decodeRouteChoice(choice, { apiKind });
    return { kind: decoded.kind, target_id: decoded.harness || (decoded.source
        ? `claudexor::${decoded.source}=` : emptyApiTarget(decoded.provider)) };
}

/** Catalog values are provider-tagged the same way routes are; scope to this route's own. */
function routeCatalogItems(route, items = []) {
    const fields = routeModelFields(route);
    const pin = route?.credential_profile_id || route?.profile_id || '';
    const wanted = fields.subscription ? `subscription:${fields.source}` : fields.provider;
    return items.filter((item) => (!wanted || parseModelSource(item?.value || item?.id || item).source === wanted)
        && (!pin || !item?.credential_profile_id || item.credential_profile_id === pin));
}

/**
 * One suggestion per model: the label names the model and makes no account claim
 * (DESIGN.md §7). Availability, the reading account and its observation time are
 * account facts, so they never travel on a model option. Two row facts do: what an
 * alias resolves to (`resolved_model`, shown only while every supplying row agrees)
 * and a row known only from the engine's frozen list (`origin: "hint"` on every
 * supplying row; absent origin is live). The value stays the row id either way.
 */
export function catalogModelOptions(items = []) {
    const values = new Map();
    for (const item of items) {
        const value = String(item?.value || item?.id || item);
        const name = String(item?.name || item?.label || '');
        if (!values.has(value)) values.set(value, { value, label: value, named: false, live: false, resolved: new Set() });
        const current = values.get(value);
        if (name && !current.named) Object.assign(current, { label: name, named: true });
        if (item?.origin !== 'hint') current.live = true;
        const resolved = item?.resolved_model;
        if (typeof resolved === 'string' && resolved && resolved !== value) current.resolved.add(resolved);
    }
    return [...values.values()].map(({ value, label, live, resolved }) => {
        const named = resolved.size === 1 ? `${value} → ${[...resolved][0]}` : label;
        return { value, label: live ? named : `${named} (shipped list)` };
    });
}

/** Suggestions carry the model alone; the source select already names the provider. */
export function routeModelSuggestions(route, items = []) {
    return routeCatalogItems(route, items)
        .map((item) => parseModelSource(String(item?.value || item?.id || item)).model);
}

/** Catalog suggestions, not an entitlement or context claim for the selected account. */
export function routeModelInputHtml(attrs, route, items, listId, { placeholder = 'Choose a model' } = {}) {
    const values = catalogModelOptions(routeCatalogItems(route, items).map((item) => {
        const value = String(item?.value || item?.id || item);
        return { ...(typeof item === 'object' ? item : {}), value: parseModelSource(value).model };
    }));
    return modelChooserHtml(attrs, routeModelFields(route).model, listId, values, { placeholder });
}

/** Catalog repaint owns suggestions and native option labels, never a draft node. */
export function updateRouteControlOptions(current, desired) {
    for (const field of current.querySelectorAll('select')) {
        const marker = [...field.attributes].find((attr) => attr.name.startsWith('data-'));
        if (!marker) continue;
        const next = [...desired.querySelectorAll('select')]
            .find((node) => node.getAttribute(marker.name) === marker.value);
        if (next && field.innerHTML !== next.innerHTML) {
            const value = field.value;
            field.innerHTML = next.innerHTML;
            field.value = value;
        }
    }
    updateModelChooserOptions(current, desired);
}

export function mintStableId(prefix, takenIds) {
    const taken = new Set(takenIds || []);
    for (let attempt = 0; attempt < 1000; attempt += 1) {
        const candidate = `${prefix}_${Math.random().toString(36).slice(2, 8)}`;
        if (!taken.has(candidate)) return candidate;
    }
    return `${prefix}_${Date.now().toString(36)}`;
}

// A roster row is NAMED by a projection of its route, never by a stored label
// (which rots once the owner re-points the row). JS twin of
// ouroboros/configured_subagents.py — engine_identity / subagent_handle /
// roster_handles / validate_unique_engines — held together by one parity table,
// web/tests/fixtures/subagent_handle_parity.json. Facts are EFFECTIVE, exactly
// what a task snapshot freezes: `inherited` is the global processing preference
// a row without its own value runs under, and an omitted session access is full.
function engineIdentity(row, inherited = '') {
    const route = row?.route || {};
    return [
        String(route.kind || ''), String(route.target_id || '').trim(),
        String(route.credential_profile_id || '').trim(), String(row?.effort || ''),
        String(row?.processing_preference || inherited || ''),
        route.kind === ROUTE_KIND_AGENT_SESSION ? String(row?.access || 'full') : '',
    ];
}

/** Route target plus THIS row's facets, each omitted at its baseline (full access, standard processing). */
export function subagentHandle(row, inherited = '') {
    const [, target, pin, effort, processing, access] = engineIdentity(row, inherited);
    return [target, effort, access === 'full' ? '' : access, pin ? `@${pin}` : '',
        processing === 'standard' ? '' : processing].filter(Boolean).join('/');
}

/** Stored id -> the label the live roster shows; twins are told apart by their stored key. */
export function rosterHandles(rows, inherited = '') {
    const base = (rows || []).map((row) => subagentHandle(row, inherited));
    return new Map((rows || []).map((row, index) => [
        String(row?.subagent_id || ''),
        base.indexOf(base[index]) === base.lastIndexOf(base[index])
            ? base[index] : `${base[index]}~${String(row?.subagent_id || '')}`,
    ]));
}

/** Index of the EARLIER row of the same kind sharing this row's handle, else -1 — a save-time rule; reads stay tolerant. */
export function sameEngineAs(rows, index, inherited = '') {
    const key = (row) => `${row?.route?.kind || ''}\n${subagentHandle(row, inherited)}`;
    return (rows || []).findIndex((row, other) => other < index && key(row) === key(rows[index]));
}

export function composeSessionTarget(harness, model) {
    const h = String(harness || '').trim();
    const m = String(model || '').trim();
    return m ? `${h}=${m}` : h;
}

export function splitSessionTarget(target) {
    const raw = String(target || '');
    const eq = raw.indexOf('=');
    if (eq < 0) return { harness: raw, model: '' };
    return { harness: raw.slice(0, eq), model: raw.slice(eq + 1) };
}

/** Effort already encoded in a Cursor/Agy compound model slug, if any. */
export function compoundSessionEffort(target) {
    const { harness, model } = splitSessionTarget(target);
    if (!['cursor', 'agy'].includes(String(harness || '')) || !model) return '';
    const compound = model.toLowerCase().endsWith('-fast') ? model.slice(0, -5) : model;
    const encoded = compound.slice(compound.lastIndexOf('-') + 1).toLowerCase();
    return EFFORT_CHOICES.includes(encoded) ? encoded : '';
}

/** Return the encoded effort only when a separate field contradicts it. */
export function compoundSessionEffortConflict(target, effort) {
    const encoded = compoundSessionEffort(target);
    const requested = String(effort || '').trim().toLowerCase();
    return encoded && requested && encoded !== requested ? encoded : '';
}

/** Choice values never contain `::`; the editor composes that spelling itself. */
export function encodeRouteChoice(row) {
    if (row?.route?.kind === ROUTE_KIND_AGENT_SESSION) {
        return `session:${splitSessionTarget(row.route.target_id).harness}`;
    }
    const { source, subscription, provider } = routeModelFields(row?.route);
    return subscription ? `subscription:${source}` : `${API_CHOICE_PREFIX}${provider}`;
}

export function decodeRouteChoice(value, { apiKind = ROUTE_KIND_API_MODEL } = {}) {
    const raw = String(value || '');
    if (raw.startsWith('session:')) {
        return { kind: ROUTE_KIND_AGENT_SESSION, harness: raw.slice('session:'.length) };
    }
    if (raw.startsWith('subscription:')) return { kind: apiKind, source: raw.slice(13) };
    // The bare legacy `api` choice is the OpenRouter lane it always meant.
    return { kind: apiKind, provider: (raw.startsWith(API_CHOICE_PREFIX)
        && raw.slice(API_CHOICE_PREFIX.length)) || DEFAULT_API_PROVIDER };
}

/** The chip text for a route's source: who serves it, in the owner's words. */
export function sourceIdentityLabel(route, {
    modelSources = [], providerProfiles = {}, harnesses = [],
} = {}) {
    if (route?.kind === ROUTE_KIND_AGENT_SESSION) {
        const { harness } = splitSessionTarget(route?.target_id);
        const descriptor = (harnesses || []).find((entry) => entry?.id === harness);
        return `${descriptor?.display_name || harness} · agent`;
    }
    const fields = routeModelFields(route, modelSources, { providerProfiles });
    return fields.subscription ? `${fields.sourceLabel || fields.source} · model`
        : `API · ${fields.providerLabel}`;
}

export function normalizeRouteSpec(route, {
    apiKind = ROUTE_KIND_API_MODEL,
    apiAliases = ['api_model', 'api_chat', 'api'],
} = {}) {
    const input = route && typeof route === 'object' ? route : {};
    const kind = input.kind === ROUTE_KIND_AGENT_SESSION
        ? ROUTE_KIND_AGENT_SESSION
        : (apiAliases.includes(String(input.kind || '')) ? apiKind : String(input.kind || apiKind));
    return {
        kind,
        target_id: String(input.target_id || ''),
        credential_pin: String(input.credential_pin
            || input.credential_profile_id || input.profile_id || ''),
    };
}

export function serializeRouteSpec(route, {
    apiKind = ROUTE_KIND_API_MODEL,
    credentialField = 'credential_profile_id',
} = {}) {
    const normalized = normalizeRouteSpec(route, { apiKind });
    const out = {
        kind: normalized.kind === ROUTE_KIND_AGENT_SESSION
            ? ROUTE_KIND_AGENT_SESSION : apiKind,
        target_id: normalized.target_id,
    };
    if (routeSupportsAccount(out) && normalized.credential_pin) {
        out[credentialField] = normalized.credential_pin;
    }
    return out;
}

function undiscoveredLabel(value, known) {
    return `${value} (${known ? 'not in discovery' : 'not checked'})`;
}

/**
 * The one grouped source select every editor draws: Models, Available subagents
 * and every review lane. Group vocabulary and order are identical everywhere;
 * a surface that cannot deliver a group omits it instead of renaming it.
 * @param {object} args
 * @param {Array} [args.harnesses] discovered agent harnesses
 * @param {Array} [args.modelSources] discovered subscription model sources
 * @param {Array<{id:string,label?:string}>} [args.providers] configuredApiProviders() output
 * @param {string} [args.currentChoice] the saved choice, so it survives a read gap
 * @param {boolean} [args.catalogKnown] whether discovery actually answered
 * @param {boolean} [args.accountsKnown] whether the accounts store actually answered
 * @param {boolean} [args.includeSessions] false where an agent session cannot be delivered
 * @param {boolean} [args.includeSubscriptions] false where a subscription model cannot be delivered
 * @param {object} [args.providerProfiles] setup-contract profiles, for a saved provider's label
 */
export function routeChoiceGroups({
    harnesses = [], modelSources = [], providers = [], currentChoice = '',
    catalogKnown = true, accountsKnown = true, includeSessions = true,
    includeSubscriptions = true, providerProfiles = {}, hasConfiguredAccounts = false,
} = {}) {
    const sessionValues = (harnesses || [])
        .filter((harness) => harness && harness.id)
        .map((harness) => ({
            value: `session:${harness.id}`,
            label: `${harness.display_name || harness.id} (agent)`,
            disabled: harness.status && harness.status !== 'ok' && !harness.enabled,
        }));
    const savedChoice = String(currentChoice || '');
    if (savedChoice.startsWith('session:')
        && !sessionValues.some((option) => option.value === savedChoice)) {
        sessionValues.push({
            value: savedChoice,
            label: undiscoveredLabel(savedChoice.slice('session:'.length), catalogKnown),
        });
    }
    const modelValues = modelSources.map((source) => ({
        value: `subscription:${source.id}`, label: `${source.label || source.id} (model)`,
    }));
    if (savedChoice.startsWith('subscription:')
        && !modelValues.some((option) => option.value === savedChoice)) {
        modelValues.push({ value: savedChoice, label: `${savedChoice.slice(13)} (not checked)` });
    }
    // Only providers whose credential is stored are offered. A saved choice
    // whose key is gone stays selectable and says so, so a save cannot silently
    // rewrite the assignment to the first listed provider.
    const apiValues = (providers || []).filter((provider) => provider && provider.id)
        .map((provider) => ({ value: `${API_CHOICE_PREFIX}${provider.id}`,
            label: provider.label || apiProviderLabel(provider.id, providerProfiles) }));
    if (savedChoice.length > API_CHOICE_PREFIX.length && savedChoice.startsWith(API_CHOICE_PREFIX)
        && !apiValues.some((option) => option.value === savedChoice)) {
        apiValues.push({ value: savedChoice,
            label: `${apiProviderLabel(savedChoice.slice(API_CHOICE_PREFIX.length), providerProfiles)} (no key)` });
    }
    apiValues.push({ value: '', disabled: true, label: 'Add a key in Accounts for more' });
    return [
        ...(includeSubscriptions ? [{ label: 'Subscriptions · models', options: modelValues.length
            ? modelValues
            : [{ value: '', disabled: true, label: catalogKnown && accountsKnown
                ? (hasConfiguredAccounts
                    ? 'No model sources listed — refresh Model Catalog'
                    : 'No model sources listed — connect one in Accounts')
                : catalogKnown ? 'No model sources listed; accounts have not been checked'
                    : 'Model sources have not been read — use Refresh Model Catalog' }] }] : []),
        { label: 'API keys', options: apiValues },
        ...(includeSessions ? [sessionValues.length
            ? { label: 'Agents · sessions', options: sessionValues }
            : { label: 'Agents · sessions', options: [{
                value: '',
                disabled: true,
                label: catalogKnown
                    ? 'None available — no agent sources were listed'
                    : 'Could not be listed — see the service banner above',
            }] }] : []),
    ];
}

// The PIN-SIDE projection of `accountRows`: one payload, one reader, so the
// select that pins a route and the Accounts tab that lists the same account
// call it one name. A second walk over `profiles.profiles` is how the pin
// option came to say `codex-default` for the row Accounts calls by its email.
export function indexProfilesByHarness(payload) {
    const byHarness = {};
    for (const row of accountRows(payload)) {
        if (row.kind !== 'profile' || !row.profile_id) continue;
        (byHarness[row.harness] = byHarness[row.harness] || []).push({
            id: row.profile_id,
            enabled: row.enabled,
            name: accountName(row),
        });
    }
    return byHarness;
}

export function profileEntry(entry) {
    if (typeof entry === 'string') return { id: entry, enabled: true, name: entry };
    const id = String(entry?.id || '');
    return { id, enabled: entry?.enabled !== false, name: String(entry?.name || '') || id };
}

/** Native model discovery is per account; an unread account is not an empty catalog. */
export function accountScopedModelCatalog(harness, pin = '') {
    const envelope = harness?.model_catalog;
    if (!Array.isArray(envelope?.accounts)) return harness;
    const accounts = envelope.accounts.filter((account) => !pin || account.credentialProfileId === pin);
    const gaps = accounts.filter((account) => !account.catalog);
    const error = gaps.map((account) => account.problem?.message || 'Account model list could not be read').join('; ')
        || ((!pin || !accounts.length) && envelope.partial ? 'Some account model lists could not be read' : '');
    return { ...harness, models: (harness.models || []).filter((item) => !pin || item.credential_profile_id === pin),
        models_error: error };
}

export function harnessModelsKnown(harness, catalogKnown = true) {
    return Boolean(catalogKnown) && !String(harness?.models_error || '');
}

export function modelsGapNote(harness, catalogKnown = true) {
    return catalogKnown && String(harness?.models_error || '')
        ? 'model list could not be read' : '';
}

export function sessionModelOptions(harness, currentModel, { catalogKnown = true } = {}) {
    const models = harness?.models || [];
    const options = [
        { value: '', label: 'Engine default model' },
        ...catalogModelOptions(models),
    ];
    if (currentModel && !options.some((option) => option.value === currentModel)) {
        options.push({
            value: currentModel,
            label: undiscoveredLabel(currentModel, harnessModelsKnown(harness, catalogKnown)),
        });
    }
    return options;
}

export function profileOptionsFor(profiles, savedPin, { accountsKnown = true } = {}) {
    const options = [
        { value: '', label: 'Account: automatic rotation' },
        ...(profiles || []).map(profileEntry).filter((profile) => profile.id).map((profile) => ({
            // The VALUE stays the id — it is what the setting stores and what
            // pins the route. Only the label speaks the owner's name for the
            // account, with the stored id appended when they differ.
            value: profile.id,
            label: `Account: ${profile.name}${profile.name !== profile.id ? ` · ${profile.id}` : ''}`
                + ` (pinned)${profile.enabled ? '' : ' (disabled)'}`,
        })),
    ];
    if (savedPin && !options.some((option) => option.value === savedPin)) {
        options.push({
            value: savedPin,
            label: `Account: ${undiscoveredLabel(savedPin, accountsKnown)}`,
        });
    }
    return options;
}

export function selectHtml(attrs, groups, selected) {
    const options = (groups || []).map((group) => {
        const body = (group.options || []).map((option) => {
            const isSelected = option.value === selected ? ' selected' : '';
            const disabled = option.disabled ? ' disabled' : '';
            return `<option value="${escapeHtml(option.value)}"${isSelected}${disabled}>${escapeHtml(option.label)}</option>`;
        }).join('');
        return group.label
            ? `<optgroup label="${escapeHtml(group.label)}">${body}</optgroup>` : body;
    }).join('');
    return `<select class="ui-control" ${attrs}>${options}</select>`;
}

export function effortSelectHtml(attrs, selected, surfaceDefault = 'route default') {
    const options = [
        { value: '', label: 'Default effort' },
        ...EFFORT_CHOICES.map((effort) => ({ value: effort, label: effort })),
    ];
    return selectHtml(
        `${attrs} title="Reasoning effort — default: ${escapeHtml(surfaceDefault)}"`,
        [{ label: '', options }],
        selected || '',
    );
}

export function describeExecutionEvidence(entry) {
    if (!entry || typeof entry !== 'object') return '';
    if ('requested_model' in entry || 'applied_model' in entry) {
        const parts = [];
        const route = String(entry.route || '');
        if (route) parts.push(route === 'api_model' ? 'API model' : `${route} session`);
        // Last-actual evidence is APPLIED telemetry only. Older receipts may
        // retain the requested route while omitting what the harness actually
        // served; never dress that requested value up as execution truth.
        const model = String(entry.applied_model || '');
        if (model) parts.push(model);
        else if (entry.requested_model) parts.push('model not disclosed');
        const account = String(entry.applied_profile || '');
        if (account) parts.push(`account ${account}`);
        const when = formatRelativeAge(Date.parse(entry.ts || ''), 'just now');
        const processing = processingExecutionText(entry.processing);
        if (processing) parts.push(processing);
        if (when) parts.push(when);
        if (entry.outcome) parts.push(`${entry.outcome}${entry.failure_code ? ` (${entry.failure_code})` : ''}`);
        if (entry.fallback?.model) parts.push(`fallback replied: ${entry.fallback.model}`);
        if ('occurred_at' in entry) parts.push(entry.occurred_at || `observed ${entry.observed_at || entry.ts}; occurrence time unknown`);
        return parts.join(' · ');
    }
    const effective = entry.effective || entry;
    const parts = [];
    const route = String(effective.route || effective.kind || '');
    if (route.startsWith(ROUTE_KIND_AGENT_SESSION)) {
        const harness = route.slice(ROUTE_KIND_AGENT_SESSION.length).replace(/^:/, '')
            || splitSessionTarget(effective.target_id || '').harness;
        parts.push(harness ? `${harness} session` : 'agent session');
    } else if (route) {
        parts.push('API model');
    }
    if (effective.model) parts.push(String(effective.model));
    const account = effective.credential_profile_id || effective.profile_id;
    if (account) parts.push(`account ${account}`);
    if (effective.access) parts.push(`access ${effective.access}`);
    const when = formatRelativeAge(Date.parse(entry.ts || ''), 'just now');
    const processing = processingExecutionText(effective.processing || entry.processing);
    if (processing) parts.push(processing);
    if (when) parts.push(when);
    return parts.join(' · ');
}
