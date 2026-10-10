// Settings/onboarding editor of the one subagent catalog; a row marked Reviewer is in the review pool.

import { apiFetch } from './api_client.js';
import {
    FACET_ACCOUNTS, FACET_CATALOG, FACET_QUOTA, READ_OK, accountRows,
    bindStatusSurface, boundedStatusRefresh, claudexorStatus,
} from './claudexor_status_store.js';
import { renderSegmentedField } from './page_header.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import {
    EFFORT_CHOICES, ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL,
    compoundSessionEffort, compoundSessionEffortConflict, configuredApiProviders, changeRouteChoice, routeModelFields,
    routeModelInputHtml, routeTargetFromModel, routeSupportsAccount, effortSelectHtml,
    encodeRouteChoice, indexProfilesByHarness, mintStableId, profileOptionsFor, describeExecutionEvidence,
    routeChoiceGroups, sameEngineAs, selectHtml, serializeRouteSpec, sessionModelOptions, updateRouteControlOptions,
    PROCESSING_CHOICES, PROCESSING_PREFERENCE_KEY, processingDetailsHtml, processingIntentLabel, accountScopedModelCatalog,
} from './route_editor_primitives.js';
import { modelChooserHtml, bindModelChoosers } from './model_chooser.js';
import { mergeModelCatalog, catalogReadNote, mergeHarnessModelCatalog } from './settings_catalog.js';
import {
    harnessMap, reviewTwinAllowed, rowIdentity, rowMeta, rowStatus, rowStatusReason, rowTaskRun, sessionRouteVerdict,
} from './subagent_status_primitives.js';
import { revealNewRow } from './ui_helpers.js';
import { effortLabel } from './effort_levels.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const MAX_AVAILABLE_SUBAGENTS = 26;
export const SUBAGENT_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;
export const ALLOW_EMPTY_REVIEW_POOL = 'allow_empty_review_pool';
const MINTED_FROM = { review_lane: 'From a former review lane', factory_default: 'Factory reviewer' };

const SETTING_KEYS = new Set(['enabled', 'items']);
const ROW_KEYS = new Set([
    'subagent_id', 'name', 'recommended_use', 'route', 'effort', 'processing_preference', 'access', 'enabled',
    'review_eligible', 'delivery', 'minted_from',
]);
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
    // `name` is retired (owner decision 1=A): a legacy value parses and is
    // DROPPED — a row is named by its route-derived handle, subagent_id is a
    // hidden stored join key, and recommended_use is the one semantic field.
    // `enabled` is written only when the owner switched the row OFF: an
    // untouched roster keeps its exact canonical bytes and fingerprint. The
    // review fields follow the same rule: only a non-default value is written.
    const marked = row?.review_eligible === true;
    return {
        subagent_id: String(row?.subagent_id || '').trim(),
        recommended_use: String(row?.recommended_use || ''),
        route,
        ...(row?.effort ? { effort: String(row.effort).trim().toLowerCase() } : {}),
        ...(row?.processing_preference ? { processing_preference: String(row.processing_preference).trim().toLowerCase() } : {}),
        ...(route.kind === ROUTE_KIND_AGENT_SESSION ? { access: row?.access ?? 'full' } : {}),
        ...(row?.enabled === false ? { enabled: false } : {}),
        ...(marked ? { review_eligible: true } : {}),
        ...(route.kind === ROUTE_KIND_API_MODEL && row?.delivery === 'packet' ? { delivery: 'packet' } : {}),
        ...(row?.minted_from ? { minted_from: String(row.minted_from) } : {}),
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

// The SHAPE verdict for one saved row, in the owner's words (`row N …`), or ''
// when the bytes can be canonicalized. Semantic and list-wide rules (stable ID
// spelling, route completeness, effort conflicts) belong to `rowErrors` below,
// which judges the live draft; this one answers only "can this be loaded".
function rowParseError(row, index) {
    const at = `row ${index + 1}`;
    if (!row || typeof row !== 'object' || Array.isArray(row)) return `${at} must be an object`;
    const rowUnknown = ownUnknownKeys(row, ROW_KEYS);
    if (rowUnknown.length) return `${at} has unknown field: ${rowUnknown[0]}`;
    if (typeof row.subagent_id !== 'string') return `${at} stable ID must be a string`;
    if (row.name !== undefined && typeof row.name !== 'string') return `${at} name must be a string`;
    if (typeof row.recommended_use !== 'string') return `${at} recommended use must be a string`;
    // Absent means enabled; anything other than a real boolean is refused
    // rather than coerced, so a typo can never read as "switched off".
    if (row.enabled !== undefined && typeof row.enabled !== 'boolean') return `${at} enabled must be true or false`;
    if (row.effort != null && typeof row.effort !== 'string') return `${at} effort must be a string`;
    if (row.processing_preference != null && typeof row.processing_preference !== 'string') return `${at} processing must be a string`;
    if (!row.route || typeof row.route !== 'object' || Array.isArray(row.route)) return `${at} needs a route object`;
    const routeUnknown = ownUnknownKeys(row.route, ROUTE_KEYS);
    if (routeUnknown.length) return `${at} route has unknown field: ${routeUnknown[0]}`;
    if (typeof row.route.kind !== 'string') return `${at} route kind must be a string`;
    if (typeof row.route.target_id !== 'string') return `${at} route target must be a string`;
    if (row.route.credential_profile_id != null && typeof row.route.credential_profile_id !== 'string') return `${at} account pin must be a string`;
    const routeKind = row.route.kind.trim().toLowerCase();
    if (![ROUTE_KIND_API_MODEL, ROUTE_KIND_AGENT_SESSION].includes(routeKind)) return `${at} has unsupported route kind`;
    if (row.access !== undefined && !['workspace_write', 'full'].includes(row.access)) return `${at} access must be workspace_write or full`;
    if (row.access !== undefined && routeKind !== ROUTE_KIND_AGENT_SESSION) return `${at} access requires an Agent session`;
    if (!routeSupportsAccount({ ...row.route, kind: routeKind })
        && String(row.route.credential_profile_id || '').trim()) return `${at} has an account pin on an API route`;
    if (row.review_eligible !== undefined && typeof row.review_eligible !== 'boolean') return `${at} reviewer mark must be true or false`;
    if (row.delivery !== undefined && !['native', 'packet'].includes(row.delivery)) return `${at} delivery must be native or packet`;
    if (row.delivery !== undefined && routeKind !== ROUTE_KIND_API_MODEL) return `${at} delivery requires an API model`;
    if (row.minted_from !== undefined && !Object.prototype.hasOwnProperty.call(MINTED_FROM, row.minted_from)) return `${at} has an unknown origin`;
    return '';
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
        const rowError = rowParseError(row, index);
        if (rowError) return { setting: null, error: rowError };
        canonicalItems.push(canonicalRow({
            ...row,
            route: { ...row.route, kind: row.route.kind.trim().toLowerCase() },
        }));
    }
    const setting = { enabled: input.enabled, items: canonicalItems };
    const errors = validateAvailableSubagentsSetting(setting);
    if (errors.length) return { setting: null, error: `saved value is invalid: ${errors[0]}` };
    return { setting, error: '' };
}

// One row's owner-facing errors, named the way the card is ("Subagent N"); the
// list validator and the per-row display read this one source. `ids` accumulates
// in list order so a repeated stable ID blames the later row. `rows` (with their inherited
// processing) ride only on the save of an EDITED roster: twins saved earlier load and re-save.
function rowErrors(row, index, ids, rows = null, inherited = '') {
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
    } else if (row?.access !== undefined && route.kind !== ROUTE_KIND_AGENT_SESSION) {
        errors.push('can select access only with an Agent session.');
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
    // A session slug's level refuses a contradicting pin, as the server does. An API-wrapped
    // `claudexor::cursor=…` row keeps a stored pin readable: the name wins when it runs.
    const encodedEffort = route.kind === ROUTE_KIND_AGENT_SESSION
        ? compoundSessionEffortConflict(route.target_id, row?.effort) : '';
    if (encodedEffort) {
        errors.push(`effort “${row.effort}” conflicts with compound route effort “${encodedEffort}”.`);
    }
    const twin = rows && String(route.target_id || '').trim() ? sameEngineAs(rows, index, inherited) : -1;
    if (twin >= 0 && !reviewTwinAllowed(rows[twin], row)) {
        errors.push(`runs the same engine as Subagent ${twin + 1} — change its model, effort, access, account or processing, mark both as Reviewer for a repeated review, or remove it.`);
    }
    return errors.map((text) => `Subagent ${index + 1} ${text}`);
}

function listLevelErrors(setting) {
    return setting.items.length > MAX_AVAILABLE_SUBAGENTS
        ? [`Available subagents supports at most ${MAX_AVAILABLE_SUBAGENTS} rows.`] : [];
}

/** The rows that review: marked Reviewer and switched on, in catalog order. */
export function reviewPoolRows(setting) {
    return (setting?.items || []).filter((row) => row?.review_eligible === true && row?.enabled !== false);
}

/** `subagent_runtime.PACKET_ONLY_POOL_WARNING`, the save warning of a pool no row of which reads the repository. */
export const PACKET_ONLY_POOL_WARNING = 'Every reviewer is a Packet row, so none reads the repository and the coupling question (how the change fits the rest of the code) goes unanswered: with Blocking review, every commit to Ouroboros itself stops as “not performed”. Mark a reviewer that reads the work itself, or choose Advisory.';

/** A non-empty pool whose every row is Packet (`subagent_runtime.coupling_unanswerable`). */
export function packetOnlyReviewPool(setting) {
    const pool = reviewPoolRows(setting);
    return pool.length > 0 && pool.every((row) => row?.route?.kind === ROUTE_KIND_API_MODEL && row?.delivery === 'packet');
}

/** The empty-pool refusal of a judged catalog draft (the server applies the same rule). */
export function reviewPoolErrors(setting, { judged = false, allowEmpty = false } = {}) {
    const items = setting?.items || [];
    if (!judged || allowEmpty || !items.length || reviewPoolRows(setting).length) return [];
    return [items.some((row) => row?.review_eligible === true)
        ? 'Every row marked Reviewer is switched off. Switch one on, or tick “Save without reviewers”.'
        : 'No row is marked Reviewer. Mark at least one row, or tick “Save without reviewers”.'];
}

/** The price of one full call of this row (the words `## Review` prints), or its non-money equivalent;
 * a reading reviewer makes several calls; unknown is never zero. */
export function reviewCostText(row, cost = null) {
    if (routeSupportsAccount(row?.route || {})) return 'uses a session seat and time';
    if (!cost) return 'price appears after saving';
    const usd = cost.usd_per_review;
    if (cost.basis !== 'route_tariff' || typeof usd !== 'number' || !Number.isFinite(usd)) return 'cost unknown';
    if (usd === 0) return 'no API cost per review';
    const text = `≈$${usd < 0.01 ? usd.toFixed(4) : usd.toFixed(2)} per full call (route tariff)`;
    return row?.delivery === 'packet' ? text : `${text}; a reading reviewer makes several`;
}

/** What actually ran the last time this row reviewed, with its review record (the card's "Last review"). */
export function lastReviewRunText(entry) {
    const text = describeExecutionEvidence(entry?.effective || !entry?.observed_model ? entry
        : { ...entry, effective: { model: entry.observed_model } });
    if (!text) return '';
    const record = String(entry?.review_record_id || entry?.record_id || '');
    return `${text}${record ? ` · record ${record}` : ''}`;
}

const reviewCostKey = (row, inherited) => JSON.stringify([
    row?.route?.kind || '', String(row?.route?.target_id || ''), row?.processing_preference || inherited || '',
]);
const UNKNOWN_COST = Object.freeze({ usd_per_review: null, basis: 'unknown' });

// Each review fact has the one place its meaning gives it (docs/DESIGN.md §6):
// the mark itself is the checkbox; a marked row switched off is a current
// exception under the head; a marked twin's repeat stays visible; the price or
// seat is the row's Review cost in Details and, for a marked API row, beside
// its delivery choice; the last review is history in Details.
function reviewCost(row, state) {
    return reviewCostText(row, state.reviewCosts?.get(reviewCostKey(row, state.processingPreference)) || null);
}

function reviewException(row) {
    return row.review_eligible === true && row.enabled === false ? 'Switched off, so not in the review pool.' : '';
}

function reviewRepeat(row, state, index) {
    const items = state.setting?.items || [];
    const twin = row.review_eligible === true ? sameEngineAs(items, index, state.processingPreference) : -1;
    return twin >= 0 && items[twin]?.review_eligible === true
        ? `Repeat of Subagent ${twin + 1}: another independent run of the same model, not a different reviewer.` : '';
}

function lastReview(row, state) {
    const pool = state.reviewPool || {};
    return lastReviewRunText((pool.pool || []).find((item) => item?.subagent_id === row.subagent_id)?.last_execution
        || pool.last_executions?.[row.subagent_id]);
}

// The effort select's empty option is Auto: the owner's chat range, whose top a marked row reviews at.
function effortDefaultLabel(row) {
    return row.review_eligible === true ? 'Auto (reviews at the top of the chat range)' : 'Auto (chat range)';
}

// A level in the model name (`cursor=…-xhigh`, `claudexor::agy=…-high`) is the row's effort: the
// facet reads it instead of offering a select, and the owner's pick of such a model clears a pin.
const namedRowEffort = (row) => compoundSessionEffort(row?.route?.target_id);
function effortFacetHtml(row, ordinal) {
    const named = namedRowEffort(row);
    if (named) {
        return `<span class="ui-control available-subagent-readonly" data-subagent-effort-named="${escapeHtml(named)}" aria-label="Reasoning effort for Subagent ${ordinal}">${escapeHtml(effortLabel(named))} · in the model name</span>`;
    }
    return effortSelectHtml(`data-subagent-field="effort" aria-label="Reasoning effort for Subagent ${ordinal}"`, row.effort || '', 'the chat range', effortDefaultLabel(row));
}

/**
 * The lanes-to-pool migration receipt that decided the shown document (`GET /api/review-pool`
 * `migration`), split by meaning: `current` is what still acts (a failed migration, an
 * environment catalog in force), `history` the conversion receipt kept behind "Reviewer origin".
 * A `history` receipt describes an earlier document the owner has since re-saved, so it says nothing.
 */
function migrationNote(migration) {
    if (!migration || migration.source === 'history') return { current: '', history: '' };
    const snapshot = migration.snapshot ? `snapshot ${migration.snapshot}` : 'no snapshot was written';
    if (migration.outcome === 'error') {
        return { current: `Review migration failed: ${migration.error || 'reason not recorded'}; your lanes were kept; ${snapshot}. Mark reviewers here and save to finish.`, history: '' };
    }
    const current = migration.source === 'environment' ? 'The catalog from the environment runs instead of these rows.' : '';
    if (migration.outcome === 'factory') {
        return { current, history: `This install had no review settings, so factory reviewers were set up (rows whose origin is “${MINTED_FROM.factory_default}”); ${snapshot}.` };
    }
    return { current, history: `Rows whose origin is “${MINTED_FROM.review_lane}” were converted from your review lanes; ${snapshot} keeps their previous value.` };
}

/** The pool rows whose model has no credentials in this install (`pool_without_credentials`); every row is the loud fact. */
function credentialsNote(payload) {
    const missing = payload.pool_without_credentials || [];
    const pool = payload.pool || [];
    if (!missing.length) return '';
    if (pool.length && missing.length === pool.length) {
        return `No pool row has credentials: none of the ${pool.length} reviewer model${pool.length === 1 ? '' : 's'} has an API key in this install, so every review will fail until a key is added or a reviewer with one is marked.`;
    }
    return `No credentials in this install for ${missing.join(', ')}: those seats cannot answer.`;
}

function reviewPoolSummary(state) {
    const items = state.setting?.items || [];
    const pool = reviewPoolRows(state.setting).length;
    const marked = items.filter((row) => row?.review_eligible === true).length;
    let empty = '';
    if (items.length && !marked) empty = 'No row is marked Reviewer, so reviews will not run and will report “not performed”.';
    else if (marked && !pool) empty = 'Every row marked Reviewer is switched off, so reviews will not run and will report “not performed”.';
    const payload = state.reviewPool || {};
    const migration = migrationNote(payload.migration);
    let note = payload.load_error || '';
    if (!note && payload.config_error) note = `The saved review pool has an error: ${payload.config_error}`;
    note = [note, migration.current, credentialsNote(payload)].filter(Boolean).join(' ');
    return {
        count: `Reviewers: ${pool}`,
        // Said when it matters: Delegation off leaves review on for the marked rows.
        stays: marked && !state.setting?.enabled ? 'Delegation is off; rows marked Reviewer still review.' : '',
        empty, confirm: Boolean(items.length && !pool), note, history: migration.history,
    };
}

// Saved/draft intent is ONE editor fact (docs/DESIGN.md §6): the cards carry availability only.
// Its slot stays in the toolbar in every state, so the first edit never re-wraps what sits above the rows.
const INTENT = {
    draft: { label: 'Unsaved changes', tone: 'neutral' },
    generated: { label: 'Generated draft', tone: 'neutral' },
    saved: { label: 'Saved', tone: 'ok' },
};

function editorIntent(state, hasPageDirtyIndicator) {
    // Settings already names the dirty draft in its Save bar; keep this slot empty
    // there. The wizard has no page indicator and still needs the editor's word.
    if (state.dirty && hasPageDirtyIndicator) return { label: '', title: '', tone: 'neutral' };
    const intent = INTENT[state.dirty ? 'draft' : state.baseline] || INTENT.saved;
    return { ...intent, title: intent.label };
}

function reviewDeliveryHtml(row, state, index) {
    if (row.route.kind !== ROUTE_KIND_API_MODEL) return '';
    // Rendered for every API row and shown while it is marked: the mark toggles it in place.
    return `<div class="ui-field available-subagent-field" data-subagent-delivery-field${row.review_eligible === true ? '' : ' hidden'}>
                    <label class="ui-field">Review delivery ${selectHtml(`data-subagent-field="delivery" aria-label="Review delivery for Subagent ${index + 1}"`, [{ label: '', options: [
                        { value: 'native', label: 'Reads the work itself' },
                        { value: 'packet', label: 'Packet — for models without tool calling' },
                    ] }], row.delivery === 'packet' ? 'packet' : 'native')}</label>
                    <span class="ui-field-help" data-subagent-delivery-cost>${escapeHtml(reviewCost(row, state))}</span>
                </div>`;
}

export function validateAvailableSubagentsSetting(setting, { uniqueEngines = false, processingPreference = '' } = {}) {
    if (!setting || typeof setting.enabled !== 'boolean' || !Array.isArray(setting.items)) {
        return ['Available subagents configuration is not loaded.'];
    }
    const ids = new Set();
    const rows = uniqueEngines ? setting.items : null;
    return [...listLevelErrors(setting), ...setting.items.flatMap((row, index) => rowErrors(row, index, ids, rows, processingPreference))];
}

export function buildAvailableSubagentsSetting(setting) {
    return {
        enabled: Boolean(setting?.enabled),
        items: (setting?.items || []).map(canonicalRow),
    };
}

export function subagentSettingsFingerprint(value) {
    const parsed = parseAvailableSubagentsSetting(value);
    return parsed.setting
        ? JSON.stringify(buildAvailableSubagentsSetting(parsed.setting))
        : JSON.stringify(value ?? null);
}

export function availableSubagentsSavePayload({ loaded = false, parseError = '', setting, allowEmptyReviewPool = false } = {}) {
    if (!loaded || parseError) return {};
    const built = buildAvailableSubagentsSetting(setting);
    // The owner's explicit "Save without reviewers": a request flag, never a stored setting.
    const emptyPool = built.items.length > 0 && !reviewPoolRows(built).length;
    return { OUROBOROS_SUBAGENTS: built, ...(allowEmptyReviewPool && emptyPool ? { [ALLOW_EMPTY_REVIEW_POOL]: true } : {}) };
}

/** Preview the current Settings draft without turning its generated actor rows into owner input. */
export function availableSubagentsPreviewPayload(settingsDraft, subscriptionsConnected) {
    const { OUROBOROS_SUBAGENTS: _roster, ...draft } = settingsDraft || {};
    return { ...draft, subscriptionsConnected: Boolean(subscriptionsConnected) };
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

// The row disclosures whose open state outlives a repaint, by kind.
const DISCLOSURES = { details: 'data-subagent-details', processing: 'data-processing-details' };
const disclosureKind = (node) => Object.keys(DISCLOSURES).find((kind) => node?.hasAttribute?.(DISCLOSURES[kind])) || '';

function focusSnapshot(host, doc) {
    const active = doc?.activeElement;
    if (!active || !host?.contains?.(active)) return null;
    const row = active.closest?.('[data-subagent-row]');
    return {
        rowId: row?.dataset?.subagentRow || '',
        field: active.dataset?.subagentField || '',
        summary: active.tagName === 'SUMMARY' ? disclosureKind(active.parentElement) : '',
        start: typeof active.selectionStart === 'number' ? active.selectionStart : null,
        end: typeof active.selectionEnd === 'number' ? active.selectionEnd : null,
        scrollTop: host.scrollTop,
    };
}

function restoreFocus(host, saved) {
    if (!saved) return;
    const rows = host.querySelectorAll?.('[data-subagent-row]') || [];
    const row = [...rows].find((item) => item.dataset?.subagentRow === saved.rowId);
    const field = saved.field ? row?.querySelector?.(`[data-subagent-field="${saved.field}"]`)
        : saved.summary ? row?.querySelector?.(`[${DISCLOSURES[saved.summary]}] > summary`) : null;
    if (field?.focus) field.focus({ preventScroll: true });
    if (saved.start !== null && field?.setSelectionRange) {
        field.setSelectionRange(saved.start, saved.end);
    }
    host.scrollTop = saved.scrollTop;
}

// Account evidence for a saved model's availability qualifier: only a confirmed
// Accounts read narrows the catalog's carriers; the catalog read still labels it.
function verifiedAccountSnapshot(state) {
    return state.accountsKnown ? state.snapshot : null;
}

// Session access choices; one list, so a further native profile lands in one place.
const ACCESS_CHOICES = [
    { value: 'full', label: 'Full system access' },
    { value: 'workspace_write', label: 'Working files' },
];
const ACCESS_HELP = 'Full system access (the default) can reach outside the working folder. The selected agent must support it. Explicit task restrictions still apply.';

// History, provenance and the row's secondary facts, behind one explicit disclosure
// (docs/DESIGN.md §6): Last review and Last task run are different events.
function rowDetailsHtml(row, state, rowKey) {
    const session = row.route.kind === ROUTE_KIND_AGENT_SESSION;
    const review = lastReview(row, state);
    const task = rowTaskRun(row, state);
    const stored = session ? '' : String(row.route.target_id || '').trim();
    const minted = MINTED_FROM[row.minted_from] || '';
    const fact = (label, value, attrs) => `<div${value ? '' : ' hidden'}${attrs ? ` ${attrs}` : ''}><dt>${label}</dt><dd>${escapeHtml(value)}</dd></div>`;
    return `<details class="model-role-details available-subagent-details" data-subagent-details>
                <summary>Details &amp; history</summary>
                <dl class="available-subagent-facts">
                    <div><dt>Review cost</dt><dd data-subagent-review-facts>${escapeHtml(reviewCost(row, state))}</dd></div>
                    ${fact('Last review', review, 'data-subagent-last-review')}
                    ${fact('Last task run', task, 'data-subagent-last-task')}
                    ${session ? '' : fact('Stored as', stored, 'data-subagent-stored')}
                    ${minted ? fact('Origin', minted, 'data-subagent-minted') : ''}
                    ${session ? `<div><dt>Access</dt><dd id="actor-${escapeHtml(rowKey)}-access-help">${ACCESS_HELP}</dd></div>` : ''}
                </dl>
            </details>`;
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
        catalogKnown: state.catalogKnown, snapshot: verifiedAccountSnapshot(state), pin: row.route.credential_profile_id,
    });
    const profileOptions = profileOptionsFor(
        (indexProfilesByHarness(state.snapshot)[split.harness]) || [],
        row.route.credential_profile_id || '',
        { accountsKnown: state.accountsKnown && Boolean(split.harness), labelled: true },
    );
    const status = rowStatus(row, state);
    const reason = rowStatusReason(status);
    const exception = reviewException(row);
    const repeat = reviewRepeat(row, state, index);
    const errors = rowErrors(row, index, new Set());
    const meta = rowMeta(row, state, errors);
    const invalid = Boolean(row._uiAttempted) && errors.length > 0;
    const identity = rowIdentity(row, state);
    const routeIdentity = harnessIdentityMarkup(identity.harnessId, {
        label: identity.label, channel: identity.channel,
        className: 'available-subagent-route-identity',
    });
    const field = (label, control) => `<label class="ui-field available-subagent-field">${label} ${control}</label>`;
    // Head: the title side wraps within itself; the Reviewer mark and the actions own
    // the end slot, so no fact of any length moves them (docs/DESIGN.md §6).
    return `
        <article class="available-subagent-row" data-subagent-row="${escapeHtml(rowKey)}" aria-labelledby="${escapeHtml(headingId)}"${invalid ? ' data-invalid' : ''}>
            <div class="available-subagent-head">
                <div class="available-subagent-title">
                    <label class="available-subagent-enable" title="Owner switch: a switched-off subagent keeps its configuration and stays editable; no new delegation selects it and it does not review."><input class="ui-checkbox" type="checkbox" data-subagent-field="enabled" aria-label="Subagent ${ordinal} enabled for new work"${row.enabled === false ? '' : ' checked'}></label>
                    <h4 class="available-subagent-heading" id="${escapeHtml(headingId)}">Subagent ${ordinal}</h4>
                    <div class="available-subagent-route-identity-wrap">${routeIdentity}</div>
                    <span class="settings-inline-status" data-subagent-status data-tone="${escapeHtml(status.tone)}" title="${escapeHtml(status.text)}">${escapeHtml(status.label)}</span>
                </div>
                <div class="available-subagent-actions">
                    <label class="available-subagent-reviewer"><input class="ui-checkbox" type="checkbox" data-subagent-field="review_eligible" aria-label="Subagent ${ordinal} reviews"${row.review_eligible === true ? ' checked' : ''}> Reviewer</label>
                    <button type="button" class="btn btn-default" data-subagent-duplicate aria-label="Duplicate Subagent ${ordinal}">Duplicate</button>
                    <button type="button" class="btn btn-default" data-subagent-remove aria-label="Remove Subagent ${ordinal}">Remove</button>
                </div>
            </div>
            <div class="available-subagent-status-reason" data-subagent-status-reason${reason ? '' : ' hidden'}>${escapeHtml(reason)}</div>
            <div class="available-subagent-status-reason" data-subagent-review-exception${exception ? '' : ' hidden'}>${escapeHtml(exception)}</div>
            <div class="available-subagent-meta" data-subagent-review-notes${repeat ? '' : ' hidden'}>${escapeHtml(repeat)}</div>
            <div class="available-subagent-route">
                ${field('Source', selectHtml(`data-subagent-field="route" aria-label="Source for Subagent ${ordinal}"`, routeGroups, encodeRouteChoice(row)))}
                <div class="ui-field available-subagent-field available-subagent-field-model">Model ${session
                    ? modelChooserHtml(`data-subagent-field="model" aria-label="Agent session model for Subagent ${ordinal}"`, split.model, `actor-${rowKey}-models`, modelOptions, { placeholder: 'Engine default model' })
                    : routeModelInputHtml(`data-subagent-field="model" aria-label="${split.subscription ? 'Subscription' : 'API'} model for Subagent ${ordinal}"`, row.route, state.apiModels, `actor-${rowKey}-models`)}</div>
                ${routeSupportsAccount(row.route)
                    ? field('Account', selectHtml(`data-subagent-field="account" aria-label="Account for Subagent ${ordinal}"`, [{ label: '', options: profileOptions }], row.route.credential_profile_id || ''))
                    : ''}
                ${field('Reasoning effort', `<span class="available-subagent-effort-facet" data-subagent-effort-facet data-named="${escapeHtml(namedRowEffort(row))}">${effortFacetHtml(row, ordinal)}</span>`)}
                ${session ? field('Access', selectHtml(`id="actor-${escapeHtml(rowKey)}-access" data-subagent-field="access" aria-label="Access for Subagent ${ordinal}"`, [{ label: '', options: ACCESS_CHOICES }], row.access || 'full')) : ''}
                ${reviewDeliveryHtml(row, state, index)}
            </div>
            <label class="available-subagent-purpose ui-field">Description
                <textarea class="ui-control" data-subagent-field="recommended_use" rows="1" aria-label="Description for Subagent ${ordinal}" placeholder="When should Ouroboros choose this subagent?">${escapeHtml(row.recommended_use)}</textarea>
            </label>
            <div id="actor-${escapeHtml(rowKey)}-meta" class="available-subagent-meta ui-field-help" data-subagent-meta${meta.qualifier ? ' data-availability-qualifier' : ''}${meta.tone ? ` data-tone="${escapeHtml(meta.tone)}"` : ''}${meta.text ? '' : ' hidden'}>${escapeHtml(meta.text)}</div>
            ${processingDetailsHtml(`data-subagent-field="processing_preference" aria-label="Processing for Subagent ${ordinal}"`, row.processing_preference, state.processingPreference)}
            ${rowDetailsHtml(row, state, rowKey)}
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
        state.providers, state.providerProfiles, state.reviewPool, state.allowEmptyReviewPool,
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
    hasPageDirtyIndicator = false,
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
        // GET /api/review-pool facts about the SAVED catalog; prices are keyed by
        // the loaded route so an edited row never shows its old route's price.
        reviewPool: null, reviewCosts: new Map(), loadedItems: [], loadedFingerprint: '',
        allowEmptyReviewPool: false,
        // `${_uiKey}:${kind}` of each open row disclosure: a repaint or a reload re-opens it.
        openDisclosures: new Set(),
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
            // Only an omitted response field may stay out of an unrelated save;
            // malformed saved bytes or an explicit repair candidate must report errors.
            if (state.unloadedOmissionAllowed) return [];
            return [state.parseError
                || 'Available subagents draft is still loading. Retry the preview before finishing.'];
        }
        if (state.parseError) return [state.parseError];
        return [
            ...validateAvailableSubagentsSetting(state.setting, { uniqueEngines: state.dirty, processingPreference: state.processingPreference }),
            ...poolErrors(),
        ];
    }

    // Judged like the server: a generated draft, or one that differs from what was loaded.
    function poolErrors() {
        const judged = state.baseline === 'generated'
            || JSON.stringify(buildAvailableSubagentsSetting(state.setting)) !== state.loadedFingerprint;
        return reviewPoolErrors(state.setting, { judged, allowEmpty: state.allowEmptyReviewPool });
    }

    function renderPoolSummary(container) {
        const summary = reviewPoolSummary(state);
        const count = container.querySelector('[data-review-pool-count]');
        if (count) count.textContent = summary.count;
        const stays = container.querySelector('[data-review-pool-stays]');
        if (stays) Object.assign(stays, { textContent: summary.stays, hidden: !summary.stays });
        const empty = container.querySelector('[data-review-pool-empty]');
        if (empty) empty.hidden = !summary.empty;
        const emptyText = container.querySelector('[data-review-pool-empty-text]');
        if (emptyText) emptyText.textContent = summary.empty;
        const confirm = container.querySelector('[data-review-pool-confirm]');
        if (confirm) confirm.hidden = !summary.confirm;
        const note = container.querySelector('[data-review-pool-note]');
        if (note) Object.assign(note, { textContent: summary.note, hidden: !summary.note });
        const history = container.querySelector('[data-review-pool-history]');
        if (history) history.hidden = !summary.history;
        const historyText = container.querySelector('[data-review-pool-history-text]');
        if (historyText) historyText.textContent = summary.history;
        const intent = editorIntent(state, hasPageDirtyIndicator);
        const intentEl = container.querySelector('[data-subagents-intent]');
        if (intentEl) {
            Object.assign(intentEl, { textContent: intent.label, title: intent.title });
            intentEl.dataset.tone = intent.tone;
        }
    }

    // Patch verdicts and inherited intent in place, preserving the caret.
    // Structural errors always show; row errors follow an attempted save.
    // The effort facet follows the model in place (select <-> the named level), so a model edit
    // keeps its caret and any open Details; a rebuilt select is bound like the first one.
    function bindEffortFacet(rowElement, row) {
        rowElement.querySelector('[data-subagent-effort-facet] [data-subagent-field="effort"]')?.addEventListener?.('change', (event) => {
            const value = String(event.target.value || '');
            if (value) row.effort = value;
            else delete row.effort;
            markDirty();
        });
    }
    function syncEffortFacet(el, row, index) {
        const facet = el.querySelector('[data-subagent-effort-facet]');
        if (!facet) return;
        const named = namedRowEffort(row);
        const current = facet.dataset.named || '';
        if (named === current && (named || facet.querySelector('[data-subagent-field="effort"]'))) {
            const effortDefault = el.querySelector('[data-subagent-field="effort"] option[value=""]');
            if (!named && effortDefault) effortDefault.textContent = effortDefaultLabel(row);
            return;
        }
        facet.innerHTML = effortFacetHtml(row, index + 1);
        if (named) facet.dataset.named = named;
        else delete facet.dataset.named;
        if (!named) bindEffortFacet(el, row);
    }

    function renderValidation() {
        const container = host();
        if (!container) return;
        const structural = !state.loaded || Boolean(state.parseError);
        const shown = structural ? validationErrors()
            : (state.saveAttempted ? [...listLevelErrors(state.setting), ...poolErrors()] : []);
        renderPoolSummary(container);
        const ids = new Set();
        state.setting.items.forEach((row, index) => {
            const rowErrs = state.loaded ? rowErrors(row, index, ids, state.dirty ? state.setting.items : null, state.processingPreference) : [];
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
            // Every line below the head is patched in place: a Reviewer or row switch never rebuilds the card.
            const show = (selector, text) => {
                const node = el.querySelector(selector);
                if (node) Object.assign(node, { textContent: text, hidden: !text });
            };
            show('[data-subagent-status-reason]', rowStatusReason(status));
            show('[data-subagent-review-exception]', reviewException(row));
            show('[data-subagent-review-notes]', reviewRepeat(row, state, index));
            const delivery = el.querySelector('[data-subagent-delivery-field]');
            if (delivery) delivery.hidden = row.review_eligible !== true;
            const cost = reviewCost(row, state);
            for (const selector of ['[data-subagent-review-facts]', '[data-subagent-delivery-cost]']) {
                const node = el.querySelector(selector);
                if (node) node.textContent = cost;
            }
            syncEffortFacet(el, row, index);
            for (const [selector, text] of [['[data-subagent-last-review]', lastReview(row, state)],
                ['[data-subagent-last-task]', rowTaskRun(row, state)],
                ['[data-subagent-stored]', String(row.route.target_id || '').trim()]]) {
                const fact = el.querySelector(selector);
                if (!fact) continue;
                fact.hidden = !text;
                const value = fact.querySelector?.('dd');
                if (value) value.textContent = text;
            }
            const meta = rowMeta(row, state, rowErrs);
            const metaEl = el.querySelector('[data-subagent-meta]');
            if (!metaEl) return;
            Object.assign(metaEl, { hidden: !meta.text, textContent: meta.text });
            metaEl.toggleAttribute('data-availability-qualifier', Boolean(meta.qualifier));
            if (meta.tone) metaEl.dataset.tone = meta.tone;
            else delete metaEl.dataset.tone;
        });
        const box = container.querySelector('[data-subagents-validation]');
        if (box) Object.assign(box, { hidden: !shown.length, textContent: shown[0] || '' });
        // The host mirrors this verdict in whatever it said about the roster.
        if (state.saveAttempted) onJudged(!shown.length);
    }

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
        // Text and fixed choices hold their draft; status keeps the same nodes.
        state.signature = structural ? '' : availableSubagentsRenderSignature(state);
        renderValidation();
        onChange(buildAvailableSubagentsSetting(state.setting));
    }

    // A new row's hidden keys are neutral: a label copied from its source would rot with the route.
    const mintRowKeys = () => ({
        subagent_id: mintStableId('subagent', state.setting.items.map((item) => item.subagent_id)),
        _uiKey: mintStableId('actor_row', state.setting.items.map((item) => item._uiKey)),
    });

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
            // Held as a draft like every other field: the section's Save is the
            // one writer, and `false` is stored only while the box is cleared.
            rowElement.querySelector('[data-subagent-field="enabled"]')?.addEventListener('change', (event) => {
                if (event.target.checked) delete row.enabled;
                else row.enabled = false;
                markDirty();
            });
            // A mark is patched in place like the row switch (its delivery field, effort default,
            // review lines and its twins' repeat notes): rebuilding would move the box under the pointer.
            rowElement.querySelector('[data-subagent-field="review_eligible"]')?.addEventListener('change', (event) => {
                if (event.target.checked) row.review_eligible = true;
                else delete row.review_eligible;
                markDirty();
            });
            rowElement.querySelectorAll('details').forEach((details) => {
                const kind = disclosureKind(details);
                if (!kind) return;
                const key = `${rowElement.dataset.subagentRow}:${kind}`;
                details.open = state.openDisclosures.has(key);
                details.addEventListener('toggle', () => {
                    if (details.open) state.openDisclosures.add(key);
                    else state.openDisclosures.delete(key);
                });
            });
            rowElement.querySelector('[data-subagent-field="delivery"]')?.addEventListener('change', (event) => {
                if (event.target.value === 'packet') row.delivery = 'packet';
                else delete row.delivery;
                markDirty();
            });
            rowElement.querySelector('[data-subagent-field="route"]')?.addEventListener('change', (event) => {
                row.route = changeRouteChoice(row.route, event.target.value);
                if (row.route.kind !== ROUTE_KIND_AGENT_SESSION) delete row.access;
                if (namedRowEffort(row)) delete row.effort;  // the owner's pick of a named model retires the pin
                markDirty({ structural: true });
                paint();
            });
            rowElement.querySelector('[data-subagent-field="model"]')?.addEventListener(
                'input',
                (event) => {
                    const previous = encodeRouteChoice(row);
                    row.route.target_id = routeTargetFromModel(row.route, event.target.value);
                    if (namedRowEffort(row)) delete row.effort;  // the owner's pick of a named model retires the pin
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
            for (const field of ['processing_preference', 'access']) {
                rowElement.querySelector(`[data-subagent-field="${field}"]`)?.addEventListener('change', (event) => {
                    const value = String(event.target.value || '');
                    if (value) row[field] = value;
                    else delete row[field];
                    markDirty();
                });
            }
            bindEffortFacet(rowElement, row);
            rowElement.querySelector('[data-subagent-duplicate]')?.addEventListener('click', () => {
                if (state.setting.items.length >= MAX_AVAILABLE_SUBAGENTS) return;
                // A copy IS the same engine, so it is born a judged draft: its card
                // names the twin until one engine field changes. The owner made the
                // copy, so it carries no factory or migration origin.
                const { minted_from: _origin, ...source } = canonicalRow(row);
                const copy = { ...source, ...mintRowKeys(), _uiAttempted: true };
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
            Object.assign(container.querySelector('[data-subagents-read-problem]'), { textContent: readProblem, hidden: !readProblem });
            Object.assign(container.querySelector('[data-subagents-diagnostics]'), { textContent: diagnostics.join(' · '), hidden: !diagnostics.length });
            renderValidation();
            return true;
        }
        disposeChoosers();
        const pool = reviewPoolSummary(state);
        const intent = editorIntent(state, hasPageDirtyIndicator);
        // Lines a row edit can toggle (the empty pool, the validation summary) sit below the rows:
        // a conditional line never moves the control that caused it (docs/DESIGN.md §6).
        container.innerHTML = `
            <div class="available-subagents-toolbar">
                <label class="local-toggle ui-field ui-field-inline" title="Off: new tasks do not delegate to these rows; rows marked Reviewer still review.">
                    <input class="ui-checkbox" type="checkbox" data-subagents-enabled aria-label="Delegation to these subagents" ${state.setting.enabled ? 'checked' : ''} ${state.loaded ? '' : 'disabled'}>
                    Delegation
                </label>
                <span class="available-subagents-count">${state.setting.items.length}/${MAX_AVAILABLE_SUBAGENTS}</span>
                <span class="available-subagents-count" data-review-pool-count>${escapeHtml(pool.count)}</span>
                <span class="settings-inline-status available-subagents-intent" data-subagents-intent data-tone="${intent.tone}" title="${escapeHtml(intent.title)}">${intent.label}</span>
                <button type="button" class="btn btn-default" data-subagent-add
                    ${!state.loaded || state.setting.items.length >= MAX_AVAILABLE_SUBAGENTS ? 'disabled' : ''}>Add subagent</button>
            </div>
            <div class="available-subagents-source" data-review-pool-stays ${pool.stays ? '' : 'hidden'}>${escapeHtml(pool.stays)}</div>
            <div class="available-subagents-diagnostics" data-review-pool-note ${pool.note ? '' : 'hidden'}>${escapeHtml(pool.note)}</div>
            <details class="model-role-details" data-review-pool-history ${pool.history ? '' : 'hidden'}><summary>Reviewer origin</summary>
                <div class="available-subagents-source" data-review-pool-history-text>${escapeHtml(pool.history)}</div></details>
            <div class="available-subagents-source" data-subagents-read-problem ${readProblem ? '' : 'hidden'}>${escapeHtml(readProblem)}</div>
            <div data-subagents-diagnostics class="available-subagents-diagnostics" ${diagnostics.length ? '' : 'hidden'}>${escapeHtml(diagnostics.join(' · '))}</div>
            <div class="available-subagents-list">
                ${state.loaded
                    ? state.setting.items.map((row, index) => availableSubagentRowMarkup(row, state, index)).join('')
                        || '<div class="available-subagents-empty">No subagents configured. Add one, or leave the list empty to make no actors available.</div>'
                    : '<div class="available-subagents-empty">The saved configuration could not be loaded, so this editor will not replace it.</div>'}
            </div>
            <div class="available-subagents-review-empty" data-review-pool-empty data-tone="warn" ${pool.empty ? '' : 'hidden'}>
                <span data-review-pool-empty-text>${escapeHtml(pool.empty)}</span>
                <label class="local-toggle ui-field-inline" data-review-pool-confirm ${pool.confirm ? '' : 'hidden'}><input class="ui-checkbox" type="checkbox" data-review-pool-allow-empty ${state.allowEmptyReviewPool ? 'checked' : ''}> Save without reviewers</label>
            </div>
            <div data-subagents-validation class="available-subagents-diagnostics" data-tone="error" ${errors.length ? '' : 'hidden'}>${escapeHtml(errors[0] || '')}</div>`;
        container.querySelector('[data-subagents-enabled]')?.addEventListener('change', (event) => {
            state.setting.enabled = Boolean(event.target.checked);
            markDirty();
        });
        container.querySelector('[data-review-pool-allow-empty]')?.addEventListener('change', (event) => {
            state.allowEmptyReviewPool = Boolean(event.target.checked);
            state.signature = availableSubagentsRenderSignature(state);
            renderValidation();
            onChange(buildAvailableSubagentsSetting(state.setting));
        });
        container.querySelector('[data-subagent-add]')?.addEventListener('click', () => {
            if (state.setting.items.length >= MAX_AVAILABLE_SUBAGENTS) return;
            const row = { recommended_use: '', route: { kind: ROUTE_KIND_API_MODEL, target_id: '' }, ...mintRowKeys() };
            state.setting.items.push(row);
            markDirty({ structural: true });
            paint();
            revealRow(row._uiKey);
        });
        bindRows(container);
        disposeChoosers = bindModelChoosers(container);
        restoreFocus(container, focused);
        renderValidation();
        return true;
    }

    // Reveal the new row after the painter restores previous focus.
    function revealRow(uiKey) {
        const row = host()?.querySelector?.(`[data-subagent-row="${uiKey}"]`);
        revealNewRow(row, row?.querySelector?.('[data-subagent-field="recommended_use"]'));
    }

    function load(value, { source = '', diagnostics = [], allowOmission = false } = {}) {
        // Invalidate a preview launched for the previous settings document.
        // A late response must never overwrite a freshly loaded configured row.
        state.previewGeneration += 1;
        state.previewSignature = '';
        const parsed = parseAvailableSubagentsSetting(value);
        state.loaded = Boolean(parsed.setting);
        state.parseError = parsed.error;
        state.unloadedOmissionAllowed = Boolean(
            allowUnloadedOmission && allowOmission && !parsed.setting,
        );
        if (parsed.setting) state.setting = attachUiKeys(parsed.setting, state.setting.items);
        const loadedSetting = parsed.setting ? buildAvailableSubagentsSetting(parsed.setting) : null;
        state.loadedItems = loadedSetting?.items || [];
        state.loadedFingerprint = loadedSetting ? JSON.stringify(loadedSetting) : '';
        state.reviewPool = null;
        state.reviewCosts = new Map();
        state.allowEmptyReviewPool = false;
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
        // Generated rows are enrichment, never a second unbounded gate on the
        // Settings critical path. The response is generation- and clean-gated.
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
            // Retain migration provenance for later clean refreshes; onboarding keeps the endpoint source.
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
        applyOwnerPreview(response) {
            const parsed = parseAvailableSubagentsSetting(response?.available_subagents);
            if (!parsed.setting) return { applied: false, error: parsed.error };
            load(parsed.setting, { source: 'configured_by_owner', diagnostics: response?.diagnostics || [] });
            markDirty({ structural: true }); paint();
            return { applied: true, error: '' };
        },
        setPreviewFailure,
        validate: validationErrors,
        noteSaveAttempt,
        collect: () => availableSubagentsSavePayload(state),
        /** Adopt GET /api/review-pool for the catalog loaded last. */
        setReviewPool(payload) {
            state.reviewPool = payload && typeof payload === 'object' ? payload : null;
            const costs = state.reviewPool?.row_costs || {};
            // A saved row the read did not price (it failed, or the pool has an error) is unknown, never free.
            state.reviewCosts = new Map(state.reviewPool ? state.loadedItems.map((row) => [
                reviewCostKey(row, state.processingPreference), costs[row.subagent_id] || UNKNOWN_COST]) : []);
            paint({ discoveryOnly: true });
        },
        get allowEmptyReviewPool() { return ALLOW_EMPTY_REVIEW_POOL in availableSubagentsSavePayload(state); },
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
                Rows marked Reviewer form the review pool: outside Cyber Pro each of them reviews every change
                to Ouroboros itself; in Cyber Pro the agent composes the panel from the pool and records why.
                Clearing a subagent's checkbox keeps its configuration and stops new tasks and reviews
                from choosing it. Internal references stay stable automatically. A route that is unavailable stays saved
                and returns an explicit refusal instead of silently changing actor or model. An unpinned
                session row may rotate among compatible healthy accounts for that same route.
            </div>
            <div class="settings-inline-note">
                Saved reviewer changes apply from the next task: a task that is already running keeps the
                reviewers it started with. A reviewer on a subscription waits for capacity rather than
                silently falling back to API spend.
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
    const meta = settings?._meta?.available_subagents;
    return [settings?.OUROBOROS_SUBAGENTS,
        meta != null && Object.prototype.hasOwnProperty.call(meta, 'candidate')
            ? meta.candidate : undefined,
    ].some(value => value !== undefined && value !== null && value !== '');
}

export function initSubagentsSection({
    onChange,
    hasPageDirtyIndicator = false,
    onJudged,
    isOuterDraftClean,
    onGeneratedApply,
    previewGenerated = null,
    store = claudexorStatus,
} = {}) {
    destroySubagentsSection();
    settingsEditor = createAvailableSubagentsEditor({
        store,
        hasPageDirtyIndicator,
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

/** Price, last-run and pool facts for the saved catalog; a failed read is said, never guessed. */
export async function reloadReviewPool({ isCurrent = () => true } = {}) {
    let payload;
    try {
        const resp = await apiFetch('/api/review-pool', { cache: 'no-store' });
        payload = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(payload.error || `HTTP ${resp.status}`);
    } catch (error) {
        payload = { load_error: `Review pool facts could not be read: ${error.message || error}` };
    }
    if (isCurrent()) settingsEditor?.setReviewPool(payload);
}

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
// Compatibility name for callers of the actor-list signature.
export const renderSignature = availableSubagentsRenderSignature;
