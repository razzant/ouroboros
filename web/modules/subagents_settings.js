// Available subagents editor shared by Settings and first-run onboarding.
// Reviewer policy stays in reviewer_slots.js; only route presentation is shared.
import {
    FACET_ACCOUNTS, FACET_CATALOG, FACET_QUOTA, READ_OK, accountRows,
    bindStatusSurface, boundedStatusRefresh, claudexorStatus,
} from './claudexor_status_store.js';
import { renderSegmentedField } from './page_header.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import {
    EFFORT_CHOICES, ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL,
    compoundSessionEffortConflict, configuredApiProviders, changeRouteChoice, routeModelFields,
    routeModelInputHtml, routeTargetFromModel, routeSupportsAccount, effortSelectHtml,
    encodeRouteChoice, indexProfilesByHarness, mintStableId, profileOptionsFor,
    routeChoiceGroups, selectHtml, serializeRouteSpec, sessionModelOptions, updateRouteControlOptions,
    PROCESSING_CHOICES, PROCESSING_PREFERENCE_KEY, processingDetailsHtml, processingIntentLabel, accountScopedModelCatalog,
} from './route_editor_primitives.js';
import { modelChooserHtml, bindModelChoosers } from './model_chooser.js';
import { mergeModelCatalog, catalogReadNote, mergeHarnessModelCatalog } from './settings_catalog.js';
import { harnessMap, rowIdentity, rowMeta, rowStatus, sessionRouteVerdict } from './subagent_status_primitives.js';
import { revealNewRow } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
export const MAX_AVAILABLE_SUBAGENTS = 10;
export const SUBAGENT_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;
const SETTING_KEYS = new Set(['enabled', 'items']);
const ROW_KEYS = new Set(['subagent_id', 'name', 'recommended_use', 'route', 'effort', 'processing_preference', 'access']);
const ROUTE_KEYS = new Set(['kind', 'target_id', 'credential_profile_id']);
function ownUnknownKeys(value, allowed) {
    return Object.keys(value || {}).filter((key) => !allowed.has(key));
}
function canonicalRow(row) {
    const route = serializeRouteSpec(row?.route || {}, {
        apiKind: ROUTE_KIND_API_MODEL,
        credentialField: 'credential_profile_id',
    });
    route.target_id = String(route.target_id || '').trim();
    if (route.credential_profile_id !== undefined) {
        route.credential_profile_id = String(route.credential_profile_id || '').trim();
        if (!route.credential_profile_id) delete route.credential_profile_id;
    }
    // Drop legacy `name`; identity uses subagent_id and description uses recommended_use.
    return {
        subagent_id: String(row?.subagent_id || '').trim(),
        recommended_use: String(row?.recommended_use || ''),
        route,
        ...(row?.effort ? { effort: String(row.effort).trim().toLowerCase() } : {}),
        ...(row?.processing_preference ? { processing_preference: String(row.processing_preference).trim().toLowerCase() } : {}),
        ...(row?.access === 'full' ? { access: 'full' } : {}),
    };
}
function attachUiKeys(setting, previousItems = []) {
    const previous = new Map((previousItems || []).map(
        (row) => [String(row.subagent_id || ''), row._uiKey],
    ));
    const taken = new Set();
    setting.items = setting.items.map((row) => {
        let key = previous.get(String(row.subagent_id || '')) || '';
        if (!key || taken.has(key)) key = mintStableId('actor_row', taken);
        taken.add(key);
        return { ...row, _uiKey: key };
    });
    return setting;
}
/** Parse without replacing malformed saved bytes with an empty list. */
export function parseAvailableSubagentsSetting(value) {
    if (value === undefined || value === null || value === '') {
        return { setting: null, error: 'Available subagents configuration was not loaded' };
    }
    let input = value;
    if (typeof input === 'string') {
        try {
            input = JSON.parse(input);
        } catch (error) {
            return { setting: null, error: `saved value is not valid JSON: ${error.message || error}` };
        }
    }
    if (!input || typeof input !== 'object' || Array.isArray(input)) {
        return { setting: null, error: 'saved value must be an object' };
    }
    const settingUnknown = ownUnknownKeys(input, SETTING_KEYS);
    if (settingUnknown.length) {
        return { setting: null, error: `saved value has unknown field: ${settingUnknown[0]}` };
    }
    if (typeof input.enabled !== 'boolean' || !Array.isArray(input.items)) {
        return { setting: null, error: 'saved value needs a boolean enabled flag and an items list' };
    }
    if (input.items.length > MAX_AVAILABLE_SUBAGENTS) {
        return { setting: null, error: `saved value has more than ${MAX_AVAILABLE_SUBAGENTS} rows` };
    }
    const canonicalItems = [];
    for (const [index, row] of input.items.entries()) {
        if (!row || typeof row !== 'object' || Array.isArray(row)) {
            return { setting: null, error: `row ${index + 1} must be an object` };
        }
        const rowUnknown = ownUnknownKeys(row, ROW_KEYS);
        if (rowUnknown.length) {
            return { setting: null, error: `row ${index + 1} has unknown field: ${rowUnknown[0]}` };
        }
        if (typeof row.subagent_id !== 'string') {
            return { setting: null, error: `row ${index + 1} stable ID must be a string` };
        }
        if (row.name !== undefined && typeof row.name !== 'string') {
            return { setting: null, error: `row ${index + 1} name must be a string` };
        }
        if (typeof row.recommended_use !== 'string') {
            return { setting: null, error: `row ${index + 1} recommended use must be a string` };
        }
        if (row.effort !== undefined && row.effort !== null && typeof row.effort !== 'string') {
            return { setting: null, error: `row ${index + 1} effort must be a string` };
        }
        if (row.processing_preference != null && typeof row.processing_preference !== 'string') return { setting: null, error: `row ${index + 1} processing must be a string` };
        if (!row.route || typeof row.route !== 'object' || Array.isArray(row.route)) {
            return { setting: null, error: `row ${index + 1} needs a route object` };
        }
        const routeUnknown = ownUnknownKeys(row.route, ROUTE_KEYS);
        if (routeUnknown.length) {
            return { setting: null, error: `row ${index + 1} route has unknown field: ${routeUnknown[0]}` };
        }
        if (typeof row.route.kind !== 'string') {
            return { setting: null, error: `row ${index + 1} route kind must be a string` };
        }
        if (typeof row.route.target_id !== 'string') {
            return { setting: null, error: `row ${index + 1} route target must be a string` };
        }
        if (row.route.credential_profile_id !== undefined
            && row.route.credential_profile_id !== null
            && typeof row.route.credential_profile_id !== 'string') {
            return { setting: null, error: `row ${index + 1} account pin must be a string` };
        }
        const routeKind = row.route.kind.trim().toLowerCase();
        if (![ROUTE_KIND_API_MODEL, ROUTE_KIND_AGENT_SESSION].includes(routeKind)) {
            return { setting: null, error: `row ${index + 1} has unsupported route kind` };
        }
        if (row.access !== undefined && !['workspace_write', 'full'].includes(row.access)) {
            return { setting: null, error: `row ${index + 1} access must be workspace_write or full` };
        }
        if (row.access === 'full' && routeKind !== ROUTE_KIND_AGENT_SESSION) {
            return { setting: null, error: `row ${index + 1} full access requires an Agent session` };
        }
        if (!routeSupportsAccount({ ...row.route, kind: routeKind })
            && String(row.route.credential_profile_id || '').trim()) {
            return { setting: null, error: `row ${index + 1} has an account pin on an API route` };
        }
        canonicalItems.push(canonicalRow({
            ...row,
            route: { ...row.route, kind: routeKind },
        }));
    }
    const setting = { enabled: input.enabled, items: canonicalItems };
    const errors = validateAvailableSubagentsSetting(setting);
    if (errors.length) return { setting: null, error: `saved value is invalid: ${errors[0]}` };
    return {
        setting,
        error: '',
    };
}
// Shared card/list errors; ordered `ids` assigns duplicate errors to the later row.
function rowErrors(row, index, ids) {
    const errors = [];
    const id = String(row?.subagent_id || '').trim();
    if (!SUBAGENT_ID_PATTERN.test(id)) {
        errors.push('needs a stable ID using letters, numbers, ., _ or - (maximum 64 characters).');
    } else if (ids.has(id)) {
        errors.push(`repeats stable ID “${id}”.`);
    }
    ids.add(id);
    const route = row?.route || {};
    if (row?.access !== undefined && !['workspace_write', 'full'].includes(row.access)) {
        errors.push('access must be Working files or Full system access.');
    } else if (row?.access === 'full' && route.kind !== ROUTE_KIND_AGENT_SESSION) {
        errors.push('can use Full system access only with an Agent session.');
    }
    if (![ROUTE_KIND_API_MODEL, ROUTE_KIND_AGENT_SESSION].includes(route.kind)) {
        errors.push('must use API model or Agent session.');
    }
    if (!routeModelFields(route).model.trim() && route.kind !== ROUTE_KIND_AGENT_SESSION
        || !String(route.target_id || '').trim()) {
        errors.push('needs a model or agent-session route.');
    }
    if (!routeSupportsAccount(route) && route.credential_profile_id) {
        errors.push('can pin an account only for a subscription model or Agent session.');
    }
    if (route.kind === ROUTE_KIND_AGENT_SESSION) {
        const target = String(route.target_id || '');
        const parts = target.split('=');
        if (/\s|:/.test(target) || parts.length > 2
            || !SUBAGENT_ID_PATTERN.test(parts[0] || '')
            || (parts.length === 2 && !parts[1])) {
            errors.push('needs its agent-session target as harness or harness=model, without whitespace or legacy :effort.');
        }
    }
    if (row?.effort && !EFFORT_CHOICES.includes(String(row.effort))) {
        errors.push('has an unsupported reasoning effort.');
    }
    if (row?.processing_preference && !PROCESSING_CHOICES.includes(row.processing_preference)) errors.push('needs Standard, Fast, Economy or inherited processing.');
    const encodedEffort = route.kind === ROUTE_KIND_AGENT_SESSION
        ? compoundSessionEffortConflict(route.target_id, row?.effort) : '';
    if (encodedEffort) {
        errors.push(`effort “${row.effort}” conflicts with compound route effort “${encodedEffort}”.`);
    }
    return errors.map((text) => `Subagent ${index + 1} ${text}`);
}
function listLevelErrors(setting) {
    return setting.items.length > MAX_AVAILABLE_SUBAGENTS
        ? [`Available subagents supports at most ${MAX_AVAILABLE_SUBAGENTS} rows.`] : [];
}
export function validateAvailableSubagentsSetting(setting) {
    if (!setting || typeof setting.enabled !== 'boolean' || !Array.isArray(setting.items)) {
        return ['Available subagents configuration is not loaded.'];
    }
    const errors = listLevelErrors(setting);
    const ids = new Set();
    setting.items.forEach((row, index) => errors.push(...rowErrors(row, index, ids)));
    return errors;
}
export function buildAvailableSubagentsSetting(setting) {
    return {
        enabled: Boolean(setting?.enabled),
        items: (setting?.items || []).map((row) => {
            const out = canonicalRow(row);
            out.subagent_id = out.subagent_id.trim();
            out.route.target_id = out.route.target_id.trim();
            if (out.route.credential_profile_id) {
                out.route.credential_profile_id = out.route.credential_profile_id.trim();
            }
            if (out.effort) out.effort = out.effort.trim();
            return out;
        }),
    };
}
export function subagentSettingsFingerprint(value) {
    const parsed = parseAvailableSubagentsSetting(value);
    return parsed.setting
        ? JSON.stringify(buildAvailableSubagentsSetting(parsed.setting))
        : JSON.stringify(value ?? null);
}
export function availableSubagentsSavePayload({ loaded = false, parseError = '', setting } = {}) {
    if (!loaded || parseError) return {};
    return { OUROBOROS_SUBAGENTS: buildAvailableSubagentsSetting(setting) };
}
/** Preview the current Settings draft without turning its generated actor rows into owner input. */
export function availableSubagentsPreviewPayload(settingsDraft, subscriptionsConnected) {
    const payload = {
        ...(settingsDraft || {}),
        subscriptionsConnected: Boolean(subscriptionsConnected),
    };
    delete payload.OUROBOROS_SUBAGENTS;
    return payload;
}
export function generatedPreviewCanReplace({
    dirty = false, outerDraftClean = true, parsedSetting = null,
} = {}) {
    return !dirty && outerDraftClean && Boolean(parsedSetting);
}
function diagnosticsText(diagnostics, out = []) {
    if (!diagnostics) return out;
    if (typeof diagnostics === 'string') {
        if (diagnostics.trim()) out.push(diagnostics.trim());
        return out;
    }
    if (Array.isArray(diagnostics)) {
        diagnostics.forEach((item) => diagnosticsText(item, out));
        return out;
    }
    if (typeof diagnostics !== 'object') return out;
    const message = diagnostics.message || diagnostics.detail || diagnostics.error;
    if (message) {
        const code = String(diagnostics.code || '').trim();
        out.push(`${code ? `${code}: ` : ''}${String(message)}`);
        return out;
    }
    Object.values(diagnostics).forEach((item) => diagnosticsText(item, out));
    return out;
}
function connectedHarnessIds(snapshot) {
    return new Set(accountRows(snapshot)
        .filter((row) => row?.enabled !== false
            && String(row?.status?.verification || '') === 'passed')
        .map((row) => String(row.harness || '')));
}
function focusSnapshot(host, doc) {
    const active = doc?.activeElement;
    if (!active || !host?.contains?.(active)) return null;
    const row = active.closest?.('[data-subagent-row]');
    return {
        rowId: row?.dataset?.subagentRow || '',
        field: active.dataset?.subagentField || '',
        start: typeof active.selectionStart === 'number' ? active.selectionStart : null,
        end: typeof active.selectionEnd === 'number' ? active.selectionEnd : null,
        scrollTop: host.scrollTop,
    };
}
function restoreFocus(host, saved) {
    if (!saved) return;
    const rows = host.querySelectorAll?.('[data-subagent-row]') || [];
    const row = [...rows].find((item) => item.dataset?.subagentRow === saved.rowId);
    const field = row?.querySelector?.(`[data-subagent-field="${saved.field}"]`);
    if (field?.focus) field.focus({ preventScroll: true });
    if (saved.start !== null && field?.setSelectionRange) {
        field.setSelectionRange(saved.start, saved.end);
    }
    host.scrollTop = saved.scrollTop;
}
export function availableSubagentRowMarkup(row, state, index = 0) {
    const ordinal = index + 1;
    const rowKey = row._uiKey || row.subagent_id;
    const headingId = `available-subagent-${rowKey}-heading`;
    const session = row.route.kind === ROUTE_KIND_AGENT_SESSION;
    const split = routeModelFields(row.route, state.modelSources);
    const harnesses = harnessMap(state.snapshot);
    const routeGroups = routeChoiceGroups({
        harnesses: state.catalogKnown ? (state.snapshot?.harnesses || []) : [],
        modelSources: state.modelSources, providers: state.providers,
        providerProfiles: state.providerProfiles, currentChoice: encodeRouteChoice(row),
        catalogKnown: state.catalogKnown, accountsKnown: state.accountsKnown,
    });
    const modelOptions = sessionModelOptions(accountScopedModelCatalog(harnesses[split.harness], row.route.credential_profile_id), split.model, {
        catalogKnown: state.catalogKnown,
    });
    const profileOptions = profileOptionsFor(
        (indexProfilesByHarness(state.snapshot)[split.harness]) || [],
        row.route.credential_profile_id || '',
        { accountsKnown: state.accountsKnown && Boolean(split.harness) },
    );
    const status = rowStatus(row, state);
    const errors = rowErrors(row, index, new Set());
    const meta = rowMeta(row, state, errors);
    const invalid = Boolean(row._uiAttempted) && errors.length > 0;
    const identity = rowIdentity(row, state);
    const routeIdentity = harnessIdentityMarkup(identity.harnessId, {
        label: identity.label, channel: identity.channel,
        className: 'available-subagent-route-identity',
    });
    return `
        <article class="available-subagent-row" data-subagent-row="${escapeHtml(rowKey)}" aria-labelledby="${escapeHtml(headingId)}"${invalid ? ' data-invalid' : ''}>
            <div class="available-subagent-head">
                <h4 class="available-subagent-heading" id="${escapeHtml(headingId)}">Subagent ${ordinal}</h4>
                <div class="available-subagent-route-identity-wrap">${routeIdentity}</div>
                <span class="settings-inline-status" data-subagent-status data-tone="${escapeHtml(status.tone)}" title="${escapeHtml(status.text)}">${escapeHtml(status.label)}</span>
                <div class="available-subagent-actions">
                    <button type="button" class="btn btn-default" data-subagent-duplicate aria-label="Duplicate Subagent ${ordinal}">Duplicate</button>
                    <button type="button" class="btn btn-default" data-subagent-remove aria-label="Remove Subagent ${ordinal}">Remove</button>
                </div>
            </div>
            <label class="available-subagent-purpose ui-field">Description
                <textarea class="ui-control" data-subagent-field="recommended_use" rows="1" aria-label="Description for Subagent ${ordinal}" placeholder="When should Ouroboros choose this subagent?">${escapeHtml(row.recommended_use)}</textarea>
            </label>
            <div class="available-subagent-route">
                ${selectHtml(`data-subagent-field="route" aria-label="Source for Subagent ${ordinal}"`, routeGroups, encodeRouteChoice(row))}
                ${session
                    ? modelChooserHtml(`data-subagent-field="model" aria-label="Agent session model for Subagent ${ordinal}"`, split.model, `actor-${rowKey}-models`, modelOptions, { placeholder: 'Engine default model' })
                    : routeModelInputHtml(`data-subagent-field="model" aria-label="${split.subscription ? 'Subscription' : 'API'} model for Subagent ${ordinal}"`, row.route, state.apiModels, `actor-${rowKey}-models`)}
                ${routeSupportsAccount(row.route)
                    ? selectHtml(`data-subagent-field="account" aria-label="Account for Subagent ${ordinal}"`, [{ label: '', options: profileOptions }], row.route.credential_profile_id || '')
                    : ''}
                ${effortSelectHtml(`data-subagent-field="effort" aria-label="Reasoning effort for Subagent ${ordinal}"`, row.effort || '', 'route default')}
            </div>
            ${processingDetailsHtml(`data-subagent-field="processing_preference" aria-label="Processing for Subagent ${ordinal}"`, row.processing_preference, state.processingPreference)}
            ${session ? `<div class="ui-field">
                <label for="actor-${escapeHtml(rowKey)}-access">Access</label>
                ${selectHtml(`id="actor-${escapeHtml(rowKey)}-access" data-subagent-field="access" aria-label="Access for Subagent ${ordinal}"`, [{ label: '', options: [
                    { value: 'workspace_write', label: 'Working files (default)' },
                    { value: 'full', label: 'Full system access' },
                ] }], row.access || 'workspace_write')}
                <div class="ui-field-help" id="actor-${escapeHtml(rowKey)}-access-help">Full system access can reach outside the working folder. The selected agent must support it.</div>
            </div>` : ''}
            <div id="actor-${escapeHtml(rowKey)}-meta" class="available-subagent-meta ui-field-help" data-subagent-meta${meta.tone ? ` data-tone="${escapeHtml(meta.tone)}"` : ''} title="${escapeHtml(meta.text)}"${meta.text ? '' : ' hidden'}>${escapeHtml(meta.text)}</div>
        </article>`;
}
export function availableSubagentsRenderSignature(state, nowMs = Date.now()) {
    return JSON.stringify([
        state.loaded,
        state.parseError,
        state.setting,
        state.saveAttempted,
        state.baseline,
        state.source,
        diagnosticsText(state.diagnostics),
        state.statusError,
        state.catalogKnown,
        state.accountsKnown,
        state.quotaKnown,
        state.snapshot?.harnesses || [],
        accountRows(state.snapshot),
        state.snapshot?.quota || [],
        state.snapshot?.subagent_last_delegation || null,
        (state.setting?.items || []).map((row) => row?.route?.kind === ROUTE_KIND_AGENT_SESSION
            ? sessionRouteVerdict(row, state, nowMs).text : ''),
        state.apiModels, state.modelSources, state.modelCatalogNote, state.processingPreference,
        state.providers, state.providerProfiles,
    ]);
}
/** One isolated editor instance; Settings keeps a singleton wrapper below. */
export function createAvailableSubagentsEditor({
    hostId = 'available-subagents-editor',
    doc = () => (typeof document === 'undefined' ? null : document),
    win = () => (typeof window === 'undefined' ? null : window),
    store = claudexorStatus,
    onChange = () => {},
    onDirtyChange = () => {},
    onJudged = () => {},
    isOuterDraftClean = () => true,
    onGeneratedApply = () => {},
    allowUnloadedOmission = false,
    previewGenerated = null,
    baseline = 'saved',
} = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    const getWin = typeof win === 'function' ? win : () => win;
    let disposeChoosers = () => {};
    const state = {
        loaded: false,
        destroyed: false,
        parseError: '',
        unloadedOmissionAllowed: false,
        setting: { enabled: true, items: [] },
        source: '',
        diagnostics: [],
        dirty: false,
        saveAttempted: false,
        baseline: baseline === 'generated' ? 'generated' : 'saved',
        statusError: '',
        catalogKnown: false,
        accountsKnown: false,
        quotaKnown: false,
        snapshot: null,
        apiModels: [], modelSources: [], modelCatalogNote: '', processingPreference: '',
        // Providers with a stored key, named by the contract (setSourceContext).
        providers: [], providerProfiles: {},
        signature: '',
        statusDisposer: null,
        catalogDisposer: null,
        previewSignature: '',
        previewGeneration: 0,
    };
    function host() {
        return getDoc()?.getElementById?.(hostId) || null;
    }

    function adoptStatus() {
        state.statusError = store?.error || '';
        state.catalogKnown = store?.facet?.(FACET_CATALOG) === READ_OK;
        state.accountsKnown = store?.facet?.(FACET_ACCOUNTS) === READ_OK;
        state.quotaKnown = store?.facet?.(FACET_QUOTA) === READ_OK;
        const snapshot = store?.snapshot;
        state.snapshot = snapshot ? { ...snapshot, harnesses: snapshot.harnesses?.map((harness) =>
            mergeHarnessModelCatalog(state.snapshot?.harnesses?.find((old) => old.id === harness.id), harness)) } : null;
    }

    function validationErrors() {
        if (!state.loaded) {
            // Unloaded fields may be omitted; malformed saved or repair bytes
            // must block saving until repaired.
            if (state.unloadedOmissionAllowed) return [];
            return [state.parseError
                || 'Available subagents draft is still loading. Retry the preview before finishing.'];
        }
        if (state.parseError) return [state.parseError];
        return validateAvailableSubagentsSetting(state.setting);
    }

    // Reconcile verdicts in place to preserve the caret. Structural errors always
    // show; row errors follow an attempted save, keeping hints for new rows.
    function renderValidation() {
        const container = host();
        if (!container) return;
        const structural = !state.loaded || Boolean(state.parseError);
        const shown = structural ? validationErrors()
            : (state.saveAttempted ? listLevelErrors(state.setting) : []);
        const ids = new Set();
        state.setting.items.forEach((row, index) => {
            const rowErrs = state.loaded ? rowErrors(row, index, ids) : [];
            const judged = Boolean(row._uiAttempted) && rowErrs.length > 0;
            if (judged && !structural) shown.push(...rowErrs);
            const el = container.querySelector(`[data-subagent-row="${row._uiKey || row.subagent_id}"]`);
            if (!el) return;
            const processingSummary = el.querySelector('[data-processing-summary]');
            if (processingSummary) processingSummary.textContent = processingIntentLabel(row.processing_preference, state.processingPreference);
            el.toggleAttribute('data-invalid', judged);
            el.querySelectorAll('[data-subagent-field]').forEach((field) => {
                const prefix = `actor-${row._uiKey || row.subagent_id}`;
                field.setAttribute('aria-describedby', `${prefix}-meta${field.dataset.subagentField === 'access' ? ` ${prefix}-access-help` : ''}`);
                if (field.dataset.subagentField !== 'recommended_use') field.setAttribute('aria-invalid', String(judged));
            });
            const status = rowStatus(row, state);
            const statusEl = el.querySelector('[data-subagent-status]');
            if (statusEl) {
                Object.assign(statusEl, { textContent: status.label, title: status.text });
                statusEl.dataset.tone = status.tone;
            }
            const meta = rowMeta(row, state, rowErrs);
            const metaEl = el.querySelector('[data-subagent-meta]');
            if (!metaEl) return;
            Object.assign(metaEl, { hidden: !meta.text, textContent: meta.text, title: meta.text });
            if (meta.tone) metaEl.dataset.tone = meta.tone;
            else delete metaEl.dataset.tone;
        });
        const box = container.querySelector('[data-subagents-validation]');
        if (box) Object.assign(box, { hidden: !shown.length, textContent: shown[0] || '' });
        // The host mirrors this verdict in whatever it said about the roster.
        if (state.saveAttempted) onJudged(!shown.length);
    }

    // Save validates existing rows; later additions stay fresh. Advance the
    // signature after in-place validation so status ticks skip repainting.
    function noteSaveAttempt() {
        state.saveAttempted = true;
        state.setting.items.forEach((row) => { row._uiAttempted = true; });
        renderValidation();
        state.signature = availableSubagentsRenderSignature(state);
    }

    function markDirty({ structural = false } = {}) {
        if (!state.dirty) {
            state.dirty = true;
            onDirtyChange(true);
        }
        // Preserve text inputs on late status updates; structural changes still repaint.
        state.signature = structural ? '' : availableSubagentsRenderSignature(state);
        renderValidation();
        onChange(buildAvailableSubagentsSetting(state.setting));
    }

    function bindRows(container) {
        container.querySelectorAll?.('[data-subagent-row]').forEach((rowElement) => {
            const row = state.setting.items.find(
                (item) => (item._uiKey || item.subagent_id) === rowElement.dataset.subagentRow,
            );
            if (!row) return;
            rowElement.querySelector('[data-subagent-field="recommended_use"]')?.addEventListener('input', (event) => {
                row.recommended_use = String(event.target.value || '');
                markDirty();
            });
            rowElement.querySelector('[data-subagent-field="route"]')?.addEventListener('change', (event) => {
                row.route = changeRouteChoice(row.route, event.target.value);
                if (row.route.kind !== ROUTE_KIND_AGENT_SESSION) delete row.access;
                markDirty({ structural: true });
                paint();
            });
            rowElement.querySelector('[data-subagent-field="model"]')?.addEventListener(
                'input',
                (event) => {
                    const previous = encodeRouteChoice(row);
                    row.route.target_id = routeTargetFromModel(row.route, event.target.value);
                    const structural = previous !== encodeRouteChoice(row);
                    if (structural) delete row.route.credential_profile_id;
                    markDirty({ structural });
                    if (structural) paint(); else renderValidation(); // keeps the disclosed id current, caret intact
                },
            );
            rowElement.querySelector('[data-subagent-field="account"]')?.addEventListener('change', (event) => {
                const pin = String(event.target.value || '');
                if (pin) row.route.credential_profile_id = pin;
                else delete row.route.credential_profile_id;
                markDirty({ structural: true });
                paint();
            });
            rowElement.querySelector('[data-subagent-field="effort"]')?.addEventListener('change', (event) => {
                const effort = String(event.target.value || '');
                if (effort) row.effort = effort;
                else delete row.effort;
                markDirty();
            });
            rowElement.querySelector('[data-subagent-field="processing_preference"]')?.addEventListener('change', (event) => {
                if (event.target.value) row.processing_preference = event.target.value;
                else delete row.processing_preference;
                markDirty();
            });
            rowElement.querySelector('[data-subagent-field="access"]')?.addEventListener('change', (event) => {
                if (event.target.value === 'full') row.access = 'full';
                else delete row.access;
                markDirty();
            });
            rowElement.querySelector('[data-subagent-duplicate]')?.addEventListener('click', () => {
                if (state.setting.items.length >= MAX_AVAILABLE_SUBAGENTS) return;
                const copy = canonicalRow(row);
                copy.subagent_id = mintStableId(`${row.subagent_id || 'subagent'}_copy`,
                    state.setting.items.map((item) => item.subagent_id));
                copy._uiKey = mintStableId('actor_row',
                    state.setting.items.map((item) => item._uiKey));
                state.setting.items.splice(state.setting.items.indexOf(row) + 1, 0, copy);
                markDirty({ structural: true });
                paint();
                revealRow(copy._uiKey);
            });
            rowElement.querySelector('[data-subagent-remove]')?.addEventListener('click', () => {
                const index = state.setting.items.indexOf(row);
                if (index >= 0) state.setting.items.splice(index, 1);
                markDirty({ structural: true });
                paint();
            });
        });
    }

    function paint({ discoveryOnly = false } = {}) {
        const container = host();
        if (!container || state.destroyed) return false;
        const nextSignature = availableSubagentsRenderSignature(state);
        if (nextSignature === state.signature) return false;
        const focused = focusSnapshot(container, getDoc());
        state.signature = nextSignature;
        const errors = validationErrors();
        const diagnostics = diagnosticsText(state.diagnostics);
        const source = state.source ? `Source: ${state.source}.` : '';
        const readProblem = [state.statusError
            ? 'Live agent availability could not be read. Saved rows remain unchanged.' : '', state.modelCatalogNote].filter(Boolean).join(' ');
        if (discoveryOnly && container.querySelector('.available-subagents-list')) {
            state.setting.items.forEach((row, index) => {
                const el = container.querySelector(`[data-subagent-row="${row._uiKey || row.subagent_id}"]`);
                if (!el) return;
                const template = getDoc().createElement('template');
                template.innerHTML = availableSubagentRowMarkup(row, state, index);
                const desired = template.content.firstElementChild;
                updateRouteControlOptions(el, desired);
                el.querySelector('.available-subagent-route-identity-wrap').innerHTML = desired.querySelector('.available-subagent-route-identity-wrap').innerHTML;
            });
            container.querySelector('.available-subagents-source').textContent = [source, readProblem].filter(Boolean).join(' ');
            Object.assign(container.querySelector('[data-subagents-diagnostics]'), { textContent: diagnostics.join(' · '), hidden: !diagnostics.length });
            renderValidation();
            return true;
        }
        disposeChoosers();
        container.innerHTML = `
            <div class="available-subagents-toolbar">
                <label class="local-toggle ui-field ui-field-inline">
                    <input class="ui-checkbox" type="checkbox" data-subagents-enabled aria-label="Available subagents enabled" ${state.setting.enabled ? 'checked' : ''} ${state.loaded ? '' : 'disabled'}>
                    Enabled
                </label>
                <span class="available-subagents-count">${state.setting.items.length}/${MAX_AVAILABLE_SUBAGENTS}</span>
                <button type="button" class="btn btn-default" data-subagent-add
                    ${!state.loaded || state.setting.items.length >= MAX_AVAILABLE_SUBAGENTS ? 'disabled' : ''}>Add subagent</button>
            </div>
            <div class="available-subagents-source">${escapeHtml([source, readProblem].filter(Boolean).join(' '))}</div>
            <div data-subagents-diagnostics class="available-subagents-diagnostics" ${diagnostics.length ? '' : 'hidden'}>${escapeHtml(diagnostics.join(' · '))}</div>
            <div data-subagents-validation class="available-subagents-diagnostics" data-tone="error" ${errors.length ? '' : 'hidden'}>${escapeHtml(errors[0] || '')}</div>
            <div class="available-subagents-list">
                ${state.loaded
                    ? state.setting.items.map((row, index) => availableSubagentRowMarkup(row, state, index)).join('')
                        || '<div class="available-subagents-empty">No subagents configured. Add one, or leave the list empty to make no actors available.</div>'
                    : '<div class="available-subagents-empty">The saved configuration could not be loaded, so this editor will not replace it.</div>'}
            </div>`;
        container.querySelector('[data-subagents-enabled]')?.addEventListener('change', (event) => {
            state.setting.enabled = Boolean(event.target.checked);
            markDirty();
        });
        container.querySelector('[data-subagent-add]')?.addEventListener('click', () => {
            if (state.setting.items.length >= MAX_AVAILABLE_SUBAGENTS) return;
            const id = mintStableId('subagent', state.setting.items.map((row) => row.subagent_id));
            const uiKey = mintStableId('actor_row', state.setting.items.map((row) => row._uiKey));
            state.setting.items.push({
                subagent_id: id,
                recommended_use: '',
                route: { kind: ROUTE_KIND_API_MODEL, target_id: '' },
                _uiKey: uiKey,
            });
            markDirty({ structural: true });
            paint();
            revealRow(uiKey);
        });
        bindRows(container);
        disposeChoosers = bindModelChoosers(container);
        restoreFocus(container, focused);
        renderValidation();
        return true;
    }

    // After restoring prior focus, reveal the new row and focus its Description.
    function revealRow(uiKey) {
        const row = host()?.querySelector?.(`[data-subagent-row="${uiKey}"]`);
        revealNewRow(row, row?.querySelector?.('[data-subagent-field="recommended_use"]'));
    }

    function load(value, { source = '', diagnostics = [], allowOmission = false } = {}) {
        // Invalidate old previews so late responses cannot overwrite newly loaded rows.
        state.previewGeneration += 1;
        state.previewSignature = '';
        const parsed = parseAvailableSubagentsSetting(value);
        state.loaded = Boolean(parsed.setting);
        state.parseError = parsed.error;
        state.unloadedOmissionAllowed = Boolean(
            allowUnloadedOmission && allowOmission && !parsed.setting,
        );
        if (parsed.setting) state.setting = attachUiKeys(parsed.setting, state.setting.items);
        state.source = String(source || '');
        state.diagnostics = diagnostics;
        state.dirty = false;
        state.saveAttempted = false;
        state.signature = '';
        onDirtyChange(false);
        paint();
        return { loaded: state.loaded, error: state.parseError };
    }

    function applyGeneratedPreview(response) {
        const parsed = parseAvailableSubagentsSetting(response?.available_subagents);
        state.source = String(response?.source || state.source || 'onboarding_default');
        state.diagnostics = response?.diagnostics || [];
        let outerDraftClean = false;
        try {
            outerDraftClean = Boolean(isOuterDraftClean());
        } catch (error) {
            outerDraftClean = false;
        }
        const canApply = generatedPreviewCanReplace({
            dirty: state.dirty,
            outerDraftClean,
            parsedSetting: parsed.setting,
        });
        const sameAssignment = parsed.setting && state.loaded
            && JSON.stringify(buildAvailableSubagentsSetting(parsed.setting)) === JSON.stringify(buildAvailableSubagentsSetting(state.setting));
        if (canApply) {
            state.loaded = true;
            state.parseError = '';
            state.saveAttempted = false;
            if (!sameAssignment) state.setting = attachUiKeys(parsed.setting, state.setting.items);
            onDirtyChange(false);
            onGeneratedApply(buildAvailableSubagentsSetting(state.setting));
        } else if (!parsed.setting && !state.loaded) {
            state.parseError = parsed.error;
        }
        state.signature = '';
        paint({ discoveryOnly: state.loaded && (!canApply || sameAssignment) });
        return { applied: canApply, error: parsed.error };
    }

    function setPreviewFailure(error) {
        const message = String(
            error?.body?.detail || error?.body?.error || error?.message || error,
        );
        const code = String(error?.body?.code || '').trim();
        state.diagnostics = [
            `${code ? `${code}: ` : ''}${message}`,
            ...diagnosticsText(error?.body?.diagnostics),
        ];
        if (!state.loaded) {
            state.parseError = `Available subagents preview failed: ${state.diagnostics.join(' · ')}`;
        }
        state.signature = '';
        paint({ discoveryOnly: state.loaded });
    }

    async function reloadStatus() {
        await boundedStatusRefresh(store);
        adoptStatus();
        paint({ discoveryOnly: true });
        // Preview enrichment is bounded and only adopted for the same clean generation.
        void maybeRefreshGeneratedPreview({ force: true });
    }

    async function maybeRefreshGeneratedPreview({ force = false } = {}) {
        if (typeof previewGenerated !== 'function' || state.source !== 'undecided' || state.dirty) {
            return false;
        }
        let outerDraftClean = false;
        try {
            outerDraftClean = Boolean(isOuterDraftClean());
        } catch (error) {
            outerDraftClean = false;
        }
        if (!outerDraftClean) return false;
        const connected = state.accountsKnown
            ? [...connectedHarnessIds(state.snapshot)].sort()
            : [];
        const signature = JSON.stringify([state.accountsKnown, connected]);
        if (!force && signature === state.previewSignature) return false;
        state.previewSignature = signature;
        const generation = ++state.previewGeneration;
        try {
            const response = await previewGenerated({
                subscriptionsConnected: state.accountsKnown && connected.length > 0,
            });
            if (generation !== state.previewGeneration) return false;
            // Preserve unsaved candidate provenance for future clean status refreshes.
            const result = applyGeneratedPreview({ ...response, source: state.source });
            if (!result.applied) state.previewSignature = '';
            return result.applied;
        } catch (error) {
            if (generation !== state.previewGeneration) return false;
            setPreviewFailure(error);
            return false;
        }
    }

    function mount({ bindStatus = true } = {}) {
        adoptStatus();
        if (bindStatus && !state.statusDisposer) {
            state.statusDisposer = bindStatusSurface(store, {
                elementId: hostId,
                includeModels: true,
                doc: getDoc,
                win: getWin,
                listener: () => {
                    adoptStatus();
                    paint({ discoveryOnly: true });
                    void maybeRefreshGeneratedPreview();
                },
            });
        }
        if (!state.catalogDisposer) {
            const target = getDoc();
            const onCatalog = (event) => {
                const catalog = mergeModelCatalog({ items: state.apiModels, model_sources: state.modelSources }, event?.detail);
                state.modelSources = catalog.model_sources;
                state.apiModels = catalog.items;
                state.modelCatalogNote = catalogReadNote(catalog);
                state.signature = '';
                paint({ discoveryOnly: true });
            };
            target?.addEventListener?.('settings-model-catalog:updated', onCatalog);
            state.catalogDisposer = () => target?.removeEventListener?.('settings-model-catalog:updated', onCatalog);
        }
        state.signature = '';
        paint();
    }

    function destroy() {
        state.destroyed = true;
        state.previewGeneration += 1;
        disposeChoosers();
        state.statusDisposer?.();
        state.catalogDisposer?.();
        state.statusDisposer = null;
        state.catalogDisposer = null;
    }

    return {
        mount,
        destroy,
        load,
        paint,
        reloadStatus,
        refreshGeneratedPreview: maybeRefreshGeneratedPreview,
        applyGeneratedPreview,
        setPreviewFailure,
        validate: validationErrors,
        noteSaveAttempt,
        collect: () => availableSubagentsSavePayload(state),
        get setting() { return buildAvailableSubagentsSetting(state.setting); },
        get loaded() { return state.loaded; },
        get dirty() { return state.dirty; },
        get parseError() { return state.parseError; },
        setProcessingPreference(value) { state.processingPreference = String(value || ''); paint({ discoveryOnly: true }); },
        setSourceContext({ settings = {}, providerProfiles = {} } = {}) {
            state.providerProfiles = providerProfiles || {};
            state.providers = configuredApiProviders(settings, state.providerProfiles);
            paint({ discoveryOnly: true });
        },
    };
}

export function availableSubagentsEditorHost(hostId = 'available-subagents-editor') {
    return `<div id="${escapeHtml(hostId)}" class="available-subagents-editor">
        <div class="available-subagents-empty">Loading Available subagents…</div>
    </div>`;
}

export function renderSubagentsSection() {
    return `
        <div class="form-section" id="subagents-section">
            <h3>Available subagents</h3>
            <div class="settings-section-copy">
                Describe when Ouroboros should choose each numbered subagent, then select how it runs.
                Internal references stay stable automatically. A route that is unavailable stays saved
                and returns an explicit refusal instead of silently changing actor or model. An unpinned
                session row may rotate among compatible healthy accounts for that same route.
            </div>
            ${availableSubagentsEditorHost()}
            <div class="settings-effort-card">
                <label>Allow mutative subagents</label>
                <input id="s-allow-mutative-subagents" type="hidden" value="on">
                ${renderSegmentedField({
                    target: 's-allow-mutative-subagents',
                    title: 'Applies on the next task; no restart required.',
                    options: [
                        { value: 'off', label: 'Off' },
                        { value: 'auto', label: 'Auto' },
                        { value: 'on', label: 'On' },
                    ],
                })}
                <div class="settings-inline-note">
                    Whether a subagent may write in an isolated worktree, external workspace, or
                    from-scratch project. Read-only subagents remain available. Auto follows runtime
                    mode; this applies to new child tasks without a restart.
                </div>
            </div>
            <div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="s-active-subagents">Active subagents per root</label>
                    <input class="ui-control" id="s-active-subagents" type="number" min="1" max="500" value="6">
                    <div class="settings-inline-note">How many children one root task may run at once.</div>
                </div>
                <div class="form-field ui-field">
                    <label for="s-subagent-depth">Subagent depth</label>
                    <input class="ui-control" id="s-subagent-depth" type="number" min="0" max="10" value="3">
                    <div class="settings-inline-note">How deep the chain may nest. <code>0</code> turns delegation off entirely.</div>
                </div>
            </div>
            <details class="settings-subsection" id="delegation-advanced">
                <summary>Advanced — where subagents check out their work</summary>
                <div class="settings-subsection-body">
                    <div class="form-grid two">
                        <div class="form-field ui-field">
                            <label for="s-subagent-worktree-root">Subagent worktree root</label>
                            <input class="ui-control" id="s-subagent-worktree-root" type="text" placeholder="~/Ouroboros/subagent_worktrees">
                        </div>
                        <div class="form-field ui-field">
                            <label for="s-subagent-projects-root">Subagent projects root (genesis)</label>
                            <input class="ui-control" id="s-subagent-projects-root" type="text" placeholder="~/Ouroboros/projects">
                        </div>
                    </div>
                    <div class="settings-inline-note">
                        Leave either root blank for its default under <code>~/Ouroboros/</code>.
                        Genesis projects are durable; worktrees follow the GC retention setting.
                    </div>
                </div>
            </details>
        </div>`;
}

let settingsEditor = null;

function settingsSource(settings) {
    return String(settings?._meta?.available_subagents?.source
        || settings?._meta?.available_subagents_source
        || settings?.OUROBOROS_SUBAGENTS_SOURCE
        || 'configured');
}

export function availableSubagentsLoadValue(settings) {
    const raw = settings?.OUROBOROS_SUBAGENTS;
    if (raw !== undefined && raw !== null && raw !== '') return raw;
    return settings?._meta?.available_subagents?.candidate ?? raw;
}

/** Whether this response carries owner or repair bytes that must be fixed in-place. */
export function availableSubagentsHasExplicitDraft(settings) {
    const raw = settings?.OUROBOROS_SUBAGENTS;
    if (raw !== undefined && raw !== null && raw !== '') return true;
    const meta = settings?._meta?.available_subagents;
    return meta != null
        && Object.prototype.hasOwnProperty.call(meta, 'candidate')
        && meta.candidate !== undefined
        && meta.candidate !== null
        && meta.candidate !== '';
}

export function initSubagentsSection({
    onChange,
    onJudged,
    isOuterDraftClean,
    onGeneratedApply,
    previewGenerated = null,
    store = claudexorStatus,
} = {}) {
    destroySubagentsSection();
    settingsEditor = createAvailableSubagentsEditor({
        store,
        onChange: typeof onChange === 'function' ? onChange : () => {},
        onJudged: typeof onJudged === 'function' ? onJudged : () => {},
        isOuterDraftClean: typeof isOuterDraftClean === 'function' ? isOuterDraftClean : () => true,
        onGeneratedApply: typeof onGeneratedApply === 'function' ? onGeneratedApply : () => {},
        allowUnloadedOmission: true,
        previewGenerated,
    });
    settingsEditor.mount();
}

export function applySubagentsSettings(settings) {
    if (!settingsEditor) return;
    const meta = settings?._meta?.available_subagents || {};
    settingsEditor.setProcessingPreference(settings?.[PROCESSING_PREFERENCE_KEY]);
    settingsEditor.load(availableSubagentsLoadValue(settings), {
        source: settingsSource(settings),
        diagnostics: meta.diagnostics || meta.diagnostic || [],
        allowOmission: !availableSubagentsHasExplicitDraft(settings),
    });
}

export async function reloadSubagentsSection() { await settingsEditor?.reloadStatus(); }

export function destroySubagentsSection() {
    settingsEditor?.destroy();
    settingsEditor = null;
}

export function collectSubagentsSettings() { return settingsEditor?.collect() || {}; }
export function setSubagentsProcessingPreference(value) { settingsEditor?.setProcessingPreference(value); }
/** The providers the roster picker may offer, from the loaded settings document. */
export function setSubagentsSourceContext(settings, providerProfiles) { settingsEditor?.setSourceContext({ settings, providerProfiles }); }

export function validateSubagentsDraft() { return settingsEditor?.validate() || ['Available subagents editor is not loaded.']; }

/** Settings' Save button: the draft's own errors become visible from here on. */
export function noteSubagentsSaveAttempt() { settingsEditor?.noteSaveAttempt(); }
// Compatibility name; the signature covers the actor list.
export const renderSignature = availableSubagentsRenderSignature;
