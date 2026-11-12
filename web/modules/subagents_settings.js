// The one agents catalog; a row marked Reviewer is in the review pool. A card is a head and a form: setup draws both
// for every row and saves them with its Finish; Settings draws the heads as a list and one form on an agent's page,
// where every change is one write of the roster (settings_controls.js writeEntry).

import { APP_SETTINGS_ORIGIN, apiClient, apiFetch } from './api_client.js';
import {
    FACET_ACCOUNTS, FACET_CATALOG, FACET_QUOTA, READ_OK, accountRows,
    bindStatusSurface, boundedStatusRefresh, claudexorStatus,
} from './claudexor_status_store.js';
import { markRecheckRow, recheckFromSettings, saveCheckerSettings } from './consciousness_api.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import {
    EFFORT_CHOICES, ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL,
    compoundSessionEffort, compoundSessionEffortConflict, apiConnections, configuredApiProviders, changeRouteChoice, routeModelFields,
    routeModelInputHtml, routeTargetFromModel, routeSupportsAccount, effortSelectHtml,
    encodeRouteChoice, indexProfilesByHarness, mintStableId, profileOptionsFor,
    routeChoiceGroups, sameEngineAs, selectHtml, serializeRouteSpec, sessionModelOptions, subagentHandle, updateRouteControlOptions,
    PROCESSING_CHOICES, PROCESSING_PREFERENCE_KEY, processingDetailsHtml, processingIntentLabel, accountScopedModelCatalog,
    tierPriced } from './route_editor_primitives.js';
import { catalogPrograms } from './model_program.js';
import { modelChooserHtml, bindModelChoosers } from './model_chooser.js';
import { mergeModelCatalog, catalogReadNote, mergeHarnessModelCatalog, refreshModelCatalog, watchAccountModelCatalog } from './settings_catalog.js';
import {
    harnessMap, lastReviewRunText, reviewCostText, reviewTwinAllowed, rowIdentity, rowMeta, rowStatus, rowStatusReason, rowTaskRun,
    sessionRouteVerdict,
} from './subagent_status_primitives.js';
import { bindRecheckMarks, recheckMarkHtml, syncRecheckState } from './safety_recheck.js';
import { CODECS, CONFIRMS, bindSettingRows, doorSign, outcomeLine, setButtonBusy, settingsRow, writeEntry } from './settings_controls.js';
import { registerSettingsPage } from './settings_pages.js';
import { rowById, rowsOf } from './settings_rows.js';
import { revealNewRow } from './ui_helpers.js';
import { setResult } from './ui_primitives.js';
import { effortLabel } from './effort_levels.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { currentLanguage } from './i18n.js';

export const MAX_AVAILABLE_SUBAGENTS = 26;
const KEY = 'OUROBOROS_SUBAGENTS';
export const SUBAGENT_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;
export const ALLOW_EMPTY_REVIEW_POOL = 'allow_empty_review_pool';
const MINTED_FROM = { review_lane: 'From a former review lane', factory_default: 'Factory reviewer' };

const SETTING_KEYS = new Set(['enabled', 'items']);
const ROW_KEYS = new Set([
    'subagent_id', 'name', 'recommended_use', 'route', 'effort', 'processing_preference', 'access', 'enabled',
    'review_eligible', 'delivery', 'minted_from', 'rechecks_flags',
]);
const ROUTE_KEYS = new Set(['kind', 'target_id', 'credential_profile_id']);

function ownUnknownKeys(value, allowed) {
    return Object.keys(value || {}).filter((key) => !allowed.has(key));
}

function canonicalRow(row) {
    const route = serializeRouteSpec(row?.route || {}, { apiKind: ROUTE_KIND_API_MODEL, credentialField: 'credential_profile_id' });
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
        // The second check's mark is the owner's (written only by its owner endpoint): kept as loaded.
        ...(row?.rechecks_flags === true ? { rechecks_flags: true } : {}),
    };
}

function attachUiKeys(setting, previousItems = []) {
    const previous = new Map((previousItems || []).map((row) => [String(row.subagent_id || ''), row._uiKey]));
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
    if (row.access !== undefined && !['readonly', 'workspace_write', 'full'].includes(row.access)) return `${at} access must be readonly, workspace_write or full`;
    if (row.access !== undefined && routeKind !== ROUTE_KIND_AGENT_SESSION) return `${at} access requires an Agent session`;
    if (!routeSupportsAccount({ ...row.route, kind: routeKind })
        && String(row.route.credential_profile_id || '').trim()) return `${at} has an account pin on an API route`;
    if (row.review_eligible !== undefined && typeof row.review_eligible !== 'boolean') return `${at} reviewer mark must be true or false`;
    if (row.delivery !== undefined && !['native', 'packet'].includes(row.delivery)) return `${at} delivery must be native or packet`;
    if (row.delivery !== undefined && routeKind !== ROUTE_KIND_API_MODEL) return `${at} delivery requires an API model`;
    if (row.minted_from !== undefined && !Object.prototype.hasOwnProperty.call(MINTED_FROM, row.minted_from)) return `${at} has an unknown origin`;
    if (row.rechecks_flags !== undefined && typeof row.rechecks_flags !== 'boolean') return `${at} second-check mark must be true or false`;
    return '';
}

/** Parse without replacing malformed saved bytes with an empty list. */
export function parseAvailableSubagentsSetting(value) {
    if (value === undefined || value === null || value === '') return { setting: null, error: 'Available subagents configuration was not loaded' };
    let input = value;
    const refuse = (error) => ({ setting: null, error });
    if (typeof input === 'string') {
        try { input = JSON.parse(input); } catch (error) { return refuse(`saved value is not valid JSON: ${error.message || error}`); }
    }
    if (!input || typeof input !== 'object' || Array.isArray(input)) return refuse('saved value must be an object');
    const settingUnknown = ownUnknownKeys(input, SETTING_KEYS);
    if (settingUnknown.length) return refuse(`saved value has unknown field: ${settingUnknown[0]}`);
    if (typeof input.enabled !== 'boolean' || !Array.isArray(input.items)) return refuse('saved value needs a boolean enabled flag and an items list');
    if (input.items.length > MAX_AVAILABLE_SUBAGENTS) return refuse(`saved value has more than ${MAX_AVAILABLE_SUBAGENTS} rows`);
    const canonicalItems = [];
    for (const [index, row] of input.items.entries()) {
        const rowError = rowParseError(row, index);
        if (rowError) return refuse(rowError);
        canonicalItems.push(canonicalRow({ ...row, route: { ...row.route, kind: row.route.kind.trim().toLowerCase() } }));
    }
    const setting = { enabled: input.enabled, items: canonicalItems };
    const errors = validateAvailableSubagentsSetting(setting);
    if (errors.length) return refuse(`saved value is invalid: ${errors[0]}`);
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
    if (row?.access !== undefined && !['readonly', 'workspace_write', 'full'].includes(row.access)) {
        errors.push('access must be Read only, Working files or Full system access.');
    } else if (row?.access !== undefined && route.kind !== ROUTE_KIND_AGENT_SESSION) {
        errors.push('can select access only with an Agent session.');
    }
    if (![ROUTE_KIND_API_MODEL, ROUTE_KIND_AGENT_SESSION].includes(route.kind)) errors.push('must use API model or Agent session.');
    if (!routeModelFields(route).model.trim() && route.kind !== ROUTE_KIND_AGENT_SESSION
        || !String(route.target_id || '').trim()) errors.push('needs a model or agent-session route.');
    if (!routeSupportsAccount(route) && route.credential_profile_id) errors.push('can pin an account only for a subscription model or Agent session.');
    if (route.kind === ROUTE_KIND_AGENT_SESSION) {
        const target = String(route.target_id || '');
        const parts = target.split('=');
        if (/\s|:/.test(target) || parts.length > 2
            || !SUBAGENT_ID_PATTERN.test(parts[0] || '')
            || (parts.length === 2 && !parts[1])) {
            errors.push('needs its agent-session target as harness or harness=model, without whitespace or legacy :effort.');
        }
    }
    if (row?.effort && !EFFORT_CHOICES.includes(String(row.effort))) errors.push('has an unsupported reasoning effort.');
    if (row?.processing_preference && !PROCESSING_CHOICES.includes(row.processing_preference)) errors.push('needs Standard, Fast, Economy or inherited processing.');
    // A session slug's level refuses a contradicting pin, as the server does. An API-wrapped
    // `claudexor::cursor=…` row keeps a stored pin readable: the name wins when it runs.
    const encodedEffort = route.kind === ROUTE_KIND_AGENT_SESSION
        ? compoundSessionEffortConflict(route.target_id, row?.effort) : '';
    if (encodedEffort) errors.push(`effort “${row.effort}” conflicts with compound route effort “${encodedEffort}”.`);
    const twin = rows && String(route.target_id || '').trim() ? sameEngineAs(rows, index, inherited) : -1;
    if (twin >= 0 && !reviewTwinAllowed(rows[twin], row)) {
        errors.push(`runs the same engine as Subagent ${twin + 1} — change its model, effort, access, account or processing, mark both as Reviewer for a repeated review, or remove it.`);
    }
    return errors.map((text) => `Subagent ${index + 1} ${text}`);
}

const listLevelErrors = (setting) => (setting.items.length > MAX_AVAILABLE_SUBAGENTS
    ? [`Available subagents supports at most ${MAX_AVAILABLE_SUBAGENTS} rows.`] : []);

/** The rows that review: marked Reviewer and switched on, in catalog order. */
export function reviewPoolRows(setting) {
    return (setting?.items || []).filter((row) => row?.review_eligible === true && row?.enabled !== false);
}

/** Pool rows running an earlier pool row's engine (`sameEngineAs`), grouped under the row they copy. */
export function reviewPoolCopies(setting, inherited = '') {
    const items = setting?.items || [];
    const pool = reviewPoolRows(setting);
    const groups = new Map();
    pool.forEach((row, index) => {
        const first = pool[sameEngineAs(pool, index, inherited)];
        if (first) groups.set(first, [...(groups.get(first) || []), row]);
    });
    return [...groups].map(([kept, copies]) => ({ keep: items.indexOf(kept), copies: copies.map((row) => items.indexOf(row)) }));
}

const subagentList = (indices) => `Subagents ${indices.slice(0, -1).map((index) => index + 1).join(', ')} and ${indices.at(-1) + 1}`;

/** The offer under a pool of copies: what they cost when a review asks every copy, and how many
 * independent seats the pool has. How many seats a review asks follows the strictness, never this line. */
export function reviewCopiesText(groups, poolSize) {
    if (!groups.length) return '';
    const independent = poolSize - groups.reduce((sum, group) => sum + group.copies.length, 0);
    const named = groups.map((group) => subagentList([group.keep, ...group.copies])).join('; ');
    const cost = groups.length === 1
        ? `are copies of one model: a review that asks all of them pays for ${groups[0].copies.length + 1} runs of it`
        : 'are copies of one model per group: a review that asks every copy pays for each one';
    return `${named} ${cost} and gets no second opinion. The pool has ${independent} independent seat${independent === 1 ? '' : 's'}, with or without the copies.`;
}

/** `subagent_runtime.PACKET_ONLY_POOL_WARNING`, the save warning of a pool no row of which reads the repository. */
export const PACKET_ONLY_POOL_WARNING = 'Every reviewer is a Packet row, so none reads the repository and the coupling question (how the change fits the rest of the code) goes unanswered: every review of a commit to Ouroboros itself ends without a verdict. Mark a reviewer that reads the work itself.';

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

function reviewPoolSummary(state, settings = false) {
    const items = state.setting?.items || [];
    const pool = reviewPoolRows(state.setting).length;
    const marked = items.filter((row) => row?.review_eligible === true).length;
    // An empty pool, no rows at all included, reviews on the main model alone (`review_pool_slots`).
    const why = marked ? 'Every row marked Reviewer is switched off' : 'No row is marked Reviewer';
    const empty = state.loaded && !pool ? `${why}, so the main model reviews alone: one model, no second opinion.` : '';
    const groups = state.loaded ? reviewPoolCopies(state.setting, state.processingPreference) : [];
    const payload = state.reviewPool || {};
    const migration = migrationNote(payload.migration);
    let note = payload.load_error || '';
    if (!note && payload.config_error) note = `The saved review pool has an error: ${payload.config_error}`;
    note = [note, migration.current, credentialsNote(payload)].filter(Boolean).join(' ');
    return {
        count: `Reviewers: ${pool}`,
        // Said when it matters: Delegation off leaves review on for the marked rows.
        stays: !marked || state.setting?.enabled ? '' : settings ? 'Helpers are off; agents marked Reviewer still review.'
            : 'Delegation is off; rows marked Reviewer still review.',
        // Settings asks in a dialog when the server refuses (writeEntry); setup ticks the box before its Finish.
        empty, confirm: Boolean(items.length && !pool) && !settings, note, history: migration.history,
        copies: reviewCopiesText(groups, pool),
    };
}

// Saved/draft intent is ONE editor fact of setup (docs/DESIGN.md §6): the cards carry availability only.
// Its slot stays in the toolbar in every state, so the first edit never re-wraps what sits above the rows.
const INTENT = {
    draft: { label: 'Unsaved changes', tone: 'neutral' },
    generated: { label: 'Generated draft', tone: 'neutral' },
    automatic: { label: 'Automatic', tone: 'neutral' },
    saved: { label: 'Saved', tone: 'ok' },
};

function editorIntent(state) {
    if (!state.loaded) return { label: '', title: '', tone: 'neutral' };  // nothing loaded claims no intent
    // The read says what is unsaved: an undecided candidate, or its own catalog following Main (`automatic`).
    const word = state.dirty ? 'draft'
        : state.source === 'undecided' ? 'generated' : state.automatic ? 'automatic' : state.baseline;
    const intent = INTENT[word] || INTENT.saved;
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
    if (!setting || typeof setting.enabled !== 'boolean' || !Array.isArray(setting.items)) return ['Available subagents configuration is not loaded.'];
    const ids = new Set();
    const rows = uniqueEngines ? setting.items : null;
    return [...listLevelErrors(setting), ...setting.items.flatMap((row, index) => rowErrors(row, index, ids, rows, processingPreference))];
}

export function buildAvailableSubagentsSetting(setting) {
    return { enabled: Boolean(setting?.enabled), items: (setting?.items || []).map(canonicalRow) };
}

export function availableSubagentsSavePayload({ loaded = false, parseError = '', setting, allowEmptyReviewPool = false } = {}) {
    if (!loaded || parseError) return {};
    const built = buildAvailableSubagentsSetting(setting);
    // The owner's explicit "Save without reviewers": a request flag, never a stored setting.
    const emptyPool = built.items.length > 0 && !reviewPoolRows(built).length;
    return { OUROBOROS_SUBAGENTS: built, ...(allowEmptyReviewPool && emptyPool ? { [ALLOW_EMPTY_REVIEW_POOL]: true } : {}) };
}

export function generatedPreviewCanReplace({ dirty = false, parsedSetting = null } = {}) {
    return !dirty && Boolean(parsedSetting);
}

// Every sentence of a diagnostics value, however it is nested; a coded one leads with its code.
function diagnosticsText(diagnostics, out = []) {
    if (typeof diagnostics === 'string') { if (diagnostics.trim()) out.push(diagnostics.trim()); return out; }
    if (!diagnostics || typeof diagnostics !== 'object') return out;
    const message = Array.isArray(diagnostics) ? '' : diagnostics.message || diagnostics.detail || diagnostics.error;
    const code = String(diagnostics.code || '').trim();
    if (message) out.push(`${code ? `${code}: ` : ''}${String(message)}`);
    else Object.values(diagnostics).forEach((item) => diagnosticsText(item, out));
    return out;
}

function connectedHarnessIds(snapshot) {
    return new Set(accountRows(snapshot).filter((row) => row?.enabled !== false && String(row?.status?.verification || '') === 'passed')
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
        rowId: row?.dataset?.subagentRow || '', field: active.dataset?.subagentField || '',
        summary: active.tagName === 'SUMMARY' ? disclosureKind(active.parentElement) : '',
        start: typeof active.selectionStart === 'number' ? active.selectionStart : null,
        end: typeof active.selectionEnd === 'number' ? active.selectionEnd : null,
    };
}

function restoreFocus(host, saved) {
    if (!saved) return;
    const rows = host.querySelectorAll?.('[data-subagent-row]') || [];
    const row = [...rows].find((item) => item.dataset?.subagentRow === saved.rowId);
    const field = saved.field ? row?.querySelector?.(`[data-subagent-field="${saved.field}"]`)
        : saved.summary ? row?.querySelector?.(`[${DISCLOSURES[saved.summary]}] > summary`) : row?.querySelector?.('[data-editor-save]');
    if (field?.focus) field.focus({ preventScroll: true });
    if (saved.start !== null && field?.setSelectionRange) field.setSelectionRange(saved.start, saved.end);
}

// Session access choices; one list, so a further native profile lands in one place.
const ACCESS_CHOICES = [{ value: 'full', label: 'Full system access' }, { value: 'workspace_write', label: 'Working files' }, { value: 'readonly', label: 'Read only' }];
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
    return `<details class="model-role-details available-subagent-details ui-disclosure" data-subagent-details>
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

const chip = (text, tone = 'neutral') => `<span class="ui-chip settings-chip agent-chip" data-tone="${tone}">${escapeHtml(text)}</span>`;
// An on/off that writes at once is the shared switch; one that is a field of a form with its own Save is a checkbox.
function toggleHtml(field, label, checked, live, recheckId = '') {
    const attrs = `type="checkbox" data-subagent-field="${field}"${recheckId ? ` data-recheck-mark="${escapeHtml(recheckId)}"` : ''} aria-label="${escapeHtml(label)}"${checked ? ' checked' : ''}`;
    return live ? `<label class="ui-switch"><input ${attrs} role="switch" aria-checked="${checked}"${field === 'enabled' ? ' data-entry-switch' : ''}><span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span></label>`
        : `<input class="ui-checkbox" ${attrs}>`;
}

/** An agent's name where rows carry no number: its model in short, then the facets its handle spells (effort, access, pin). */
export function agentName(row, state = {}) {
    const target = String(row?.route?.target_id || '').trim();
    const model = String(routeModelFields(row?.route || {}, state.modelSources).model || '').trim();
    const short = model.slice(model.lastIndexOf('/') + 1) || (row?.route?.kind === ROUTE_KIND_AGENT_SESSION ? target : '');
    return short ? short + subagentHandle(row, state.processingPreference).slice(target.length) : 'New agent';
}
// Sentences number the cards of setup ("Subagent 2"); a page of Settings has no numbers and names the agent instead.
const namedRows = (text, state) => String(text || '').replace(/Subagents? (\d+(?:(?:, | and )\d+)*)/g,
    (_all, list) => list.replace(/\d+/g, (n) => agentName(state.setting.items[n - 1], state)));

/** A card's head. In Settings' list it is the whole entry: the mark of its source, the name that opens the agent's
 *  page with what it is for under it, and the switch. In setup it leads the form below it. */
function subagentHeadHtml(row, state, index, list = false) {
    const ordinal = index + 1;
    const status = rowStatus(row, state);
    const identity = rowIdentity(row, state);
    const mark = harnessIdentityMarkup(identity.harnessId, { label: identity.label, channel: identity.channel, className: 'available-subagent-route-identity' });
    if (list) {
        const name = agentName(row, state);
        return `<span class="settings-head-glyph" aria-hidden="true">${mark}</span>
            <button type="button" class="settings-head-open ui-hit-area" data-subagent-field="open">
                <span class="ui-row-title">${escapeHtml(name)}${row.review_eligible === true ? chip('Reviewer') : ''}${status.tone === 'warn' ? chip(status.label, 'warn') : ''}</span>
                <span class="ui-row-meta">${escapeHtml([identity.label, row.recommended_use.trim()].filter(Boolean).join(' · '))}</span>
            </button>
            <div class="ui-row-actions">${toggleHtml('enabled', `Available: ${name}`, row.enabled !== false, true)}${doorSign()}</div>`;
    }
    const rowKey = row._uiKey || row.subagent_id;
    const reason = rowStatusReason(status);
    // The title side wraps within itself; the Reviewer mark and the actions own
    // the end slot, so no fact of any length moves them (docs/DESIGN.md §6).
    return `<div class="available-subagent-head">
                <div class="available-subagent-title">
                    <label class="available-subagent-enable" title="Owner switch: a switched-off subagent keeps its configuration and stays editable; no new delegation selects it and it does not review."><input class="ui-checkbox" type="checkbox" data-subagent-field="enabled" aria-label="Subagent ${ordinal} enabled for new work"${row.enabled === false ? '' : ' checked'}></label>
                    <h4 class="available-subagent-heading" id="available-subagent-${escapeHtml(rowKey)}-heading">Subagent ${ordinal}</h4>
                    <div class="available-subagent-route-identity-wrap">${mark}</div>
                    <span class="settings-inline-status" data-subagent-status data-tone="${escapeHtml(status.tone)}" title="${escapeHtml(status.text)}">${escapeHtml(status.label)}</span>
                </div>
                <div class="available-subagent-actions">
                    <label class="available-subagent-reviewer"><input class="ui-checkbox" type="checkbox" data-subagent-field="review_eligible" aria-label="Subagent ${ordinal} reviews"${row.review_eligible === true ? ' checked' : ''}> Reviewer</label>${recheckMarkHtml(row, ordinal)}
                    <button type="button" class="btn btn-default" data-subagent-duplicate aria-label="Duplicate Subagent ${ordinal}">Duplicate</button>
                    <button type="button" class="btn btn-default" data-subagent-remove aria-label="Remove Subagent ${ordinal}">Remove</button>
                </div>
            </div>
            <div class="available-subagent-status-reason" data-subagent-status-reason${reason ? '' : ' hidden'}>${escapeHtml(reason)}</div>
            ${reviewLinesHtml(row, state, index)}`;
}

// What the marks mean for this row now: a marked row switched off, a marked twin's repeat.
function reviewLinesHtml(row, state, index) {
    const exception = reviewException(row);
    const repeat = reviewRepeat(row, state, index);
    return `<div class="available-subagent-status-reason" data-subagent-review-exception${exception ? '' : ' hidden'}>${escapeHtml(exception)}</div>
            <div class="available-subagent-meta" data-subagent-review-notes${repeat ? '' : ' hidden'}>${escapeHtml(repeat)}</div>`;
}

/** A card's form, in parts: how the agent runs, what it is for, its one meta line, then Processing and Details. Setup
 *  stacks them under the head; an agent's page in Settings lays the same parts out in its cards. */
function subagentFormParts(row, state, index) {
    const ordinal = index + 1;
    const rowKey = row._uiKey || row.subagent_id;
    const session = row.route.kind === ROUTE_KIND_AGENT_SESSION;
    const split = routeModelFields(row.route, state.modelSources);
    const harnesses = harnessMap(state.snapshot);
    const routeGroups = routeChoiceGroups({
        harnesses: state.catalogKnown ? (state.snapshot?.harnesses || []) : [],
        modelSources: state.modelSources, providers: state.providers, connections: state.connections,
        providerProfiles: state.providerProfiles, currentChoice: encodeRouteChoice(row),
        catalogKnown: state.catalogKnown, accountsKnown: state.accountsKnown,
    });
    // Only a confirmed Accounts read narrows the catalog's carriers; the catalog read still labels the model.
    const modelOptions = sessionModelOptions(accountScopedModelCatalog(harnesses[split.harness], row.route.credential_profile_id), split.model,
        { catalogKnown: state.catalogKnown, snapshot: state.accountsKnown ? state.snapshot : null, pin: row.route.credential_profile_id });
    const profileOptions = profileOptionsFor((indexProfilesByHarness(state.snapshot)[split.harness]) || [], row.route.credential_profile_id || '',
        { accountsKnown: state.accountsKnown && Boolean(split.harness), labelled: true });
    const meta = rowMeta(row, state, rowErrors(row, index, new Set()));
    const field = (label, control) => `<label class="ui-field available-subagent-field">${label} ${control}</label>`;
    return {
        route: `<div class="available-subagent-route">
                ${field('Source', selectHtml(`data-subagent-field="route" aria-label="Source for Subagent ${ordinal}"`, routeGroups, encodeRouteChoice(row)))}
                <div class="ui-field available-subagent-field available-subagent-field-model">Model ${session
                    ? modelChooserHtml(`data-subagent-field="model" aria-label="Agent session model for Subagent ${ordinal}"`, split.model, `actor-${rowKey}-models`, modelOptions, { placeholder: 'Engine default model', programs: catalogPrograms(harnesses[split.harness], {harness: split.harness, profile: row.route.credential_profile_id}) })
                    : routeModelInputHtml(`data-subagent-field="model" aria-label="${split.subscription ? 'Subscription' : 'API'} model for Subagent ${ordinal}"`, row.route, tierPriced(row.processing_preference || state.processingPreference)
                        ? state.apiModels.map((item) => ({ ...item, price: null })) : state.apiModels, `actor-${rowKey}-models`, {programs: split.subscription ? catalogPrograms(state.modelCatalog, {source: split.source, profile: row.route.credential_profile_id, sources: state.modelSources}) : []})}</div>
                ${routeSupportsAccount(row.route)
                    ? field('Account', selectHtml(`data-subagent-field="account" aria-label="Account for Subagent ${ordinal}"`, [{ label: '', options: profileOptions }], row.route.credential_profile_id || ''))
                    : ''}
                ${field('Reasoning effort', `<span class="available-subagent-effort-facet" data-subagent-effort-facet data-named="${escapeHtml(namedRowEffort(row))}">${effortFacetHtml(row, ordinal)}</span>`)}
                ${session ? field('Access', selectHtml(`id="actor-${escapeHtml(rowKey)}-access" data-subagent-field="access" aria-label="Access for Subagent ${ordinal}"`, [{ label: '', options: ACCESS_CHOICES }], row.access || 'full')) : ''}
                ${reviewDeliveryHtml(row, state, index)}
            </div>`,
        purpose: (label) => `<label class="available-subagent-purpose ui-field">${label}
                <textarea class="ui-control" data-subagent-field="recommended_use" rows="1" aria-label="Description for Subagent ${ordinal}" placeholder="When should Ouroboros choose this subagent?">${escapeHtml(row.recommended_use)}</textarea>
            </label>`,
        meta: `<div id="actor-${escapeHtml(rowKey)}-meta" class="available-subagent-meta ui-field-help" data-subagent-meta${meta.qualifier ? ' data-availability-qualifier' : ''}${meta.tone ? ` data-tone="${escapeHtml(meta.tone)}"` : ''}${meta.text ? '' : ' hidden'}>${escapeHtml(meta.text)}</div>`,
        more: `${processingDetailsHtml(`data-subagent-field="processing_preference" aria-label="Processing for Subagent ${ordinal}"`, row.processing_preference, state.processingPreference)}
            ${rowDetailsHtml(row, state, rowKey)}`,
    };
}

/** Setup's card: the head, then the form. */
export function availableSubagentRowMarkup(row, state, index = 0) {
    const rowKey = row._uiKey || row.subagent_id;
    const invalid = Boolean(row._uiAttempted) && rowErrors(row, index, new Set()).length > 0;
    const form = subagentFormParts(row, state, index);
    return `
        <article class="available-subagent-row" data-subagent-row="${escapeHtml(rowKey)}" aria-labelledby="available-subagent-${escapeHtml(rowKey)}-heading"${invalid ? ' data-invalid' : ''}>
            ${subagentHeadHtml(row, state, index)}
            ${form.route}
            ${form.purpose('Description')}
            ${form.meta}
            ${form.more}
        </article>`;
}

/** One agent's page in Settings: what is switched at once, what it is for, how it runs with the form's Save, and its
 *  removal. `live` is a saved agent; in an unsaved one the on/off and the mark are fields of the form. */
function agentPageHtml(row, state, index, live) {
    const form = subagentFormParts(row, state, index);
    const status = rowStatus(row, state);
    const saved = state.loadedItems.find((item) => item.subagent_id === row.subagent_id);
    const recheck = saved?.rechecks_flags === true
        || (saved?.route.kind === ROUTE_KIND_API_MODEL && saved.delivery === 'packet' && saved.enabled !== false);
    const use = (field, title, meta, checked) => `<div class="ui-row settings-row agent-use" data-kind="switch"><p class="ui-row-title">${title}</p>
        <p class="ui-row-meta"><span>${escapeHtml(meta)}</span></p><div class="ui-row-actions">${toggleHtml(field, title, checked, live, field === 'rechecks_flags' ? row.subagent_id : '')}</div></div>`;
    const card = (title, body, lead = '') => `<section class="ui-card settings-card"><h3 class="settings-card-title agents-card-title">${title}</h3>
        ${lead ? `<p class="settings-card-lead">${lead}</p>` : ''}${body}</section>`;
    return `<p class="settings-page-lead">${escapeHtml(rowIdentity(row, state).label)}</p>
        <div class="settings-stack agent-page" data-subagent-row="${escapeHtml(row._uiKey || row.subagent_id)}" data-entry-id="${escapeHtml(row.subagent_id)}">
            ${card('In use', `<div class="settings-rows">
                ${use('enabled', 'Available', 'Off keeps the agent and its notes; Ouroboros does not call it.', row.enabled !== false)}
                ${use('review_eligible', 'Reviewer', state.tickCaption, row.review_eligible === true)}
                ${live && recheck ? use('rechecks_flags', 'Re-checks flagged actions', 'The second look at a step the Safety Supervisor flags. One agent holds it.', saved.rechecks_flags === true) : ''}
            </div>
            <p class="ui-status agent-status" data-subagent-status data-tone="${escapeHtml(status.tone)}">${escapeHtml(status.text)}</p>
            ${reviewLinesHtml(row, state, index)}`)}
            ${card('What it is good at', form.purpose(''), 'Ouroboros reads this when it chooses a helper.')}
            ${card('Runs on', `${form.route}${form.meta}${form.more}
            <div data-subagents-validation class="available-subagents-diagnostics" data-tone="error" hidden></div>
            <div class="ui-action-row"><button type="button" class="btn btn-save" data-editor-save>Save</button>
                ${live ? `<button type="button" class="btn btn-default" data-editor-discard${state.dirty ? '' : ' hidden'}>Discard changes</button>` : ''}
                <span class="ui-result" data-editor-result role="status" aria-live="polite"></span></div>`)}
            <div class="ui-action-row settings-foot">${live ? '<button type="button" class="btn btn-default" data-subagent-duplicate>Duplicate</button>' : ''}
                <button type="button" class="btn ${live ? 'btn-danger' : 'btn-default'}" data-subagent-remove data-editor-remove>${live ? 'Remove' : 'Discard'}</button></div>
        </div>`;
}

export function availableSubagentsRenderSignature(state, nowMs = Date.now()) {
    return JSON.stringify([
        state.loaded, state.parseError, state.setting, state.saveAttempted, state.baseline, state.automatic, state.source,
        diagnosticsText(state.diagnostics), state.statusError, state.catalogKnown, state.accountsKnown, state.quotaKnown,
        state.snapshot?.harnesses || [], accountRows(state.snapshot), state.snapshot?.quota || [],
        state.snapshot?.subagent_last_delegation || null,
        (state.setting?.items || []).map((row) => row?.route?.kind === ROUTE_KIND_AGENT_SESSION
            ? sessionRouteVerdict(row, state, nowMs).text : ''),
        state.apiModels, state.modelSources, state.modelCatalog, state.modelCatalogNote, state.processingPreference,
        state.providers, state.connections, state.providerProfiles, state.reviewPool, state.allowEmptyReviewPool,
        state.filter, state.tickCaption, state.unsavedId,
        currentLanguage(),  // the suggestions' prices are composed words: another language is another paint
    ]);
}

// Settings keeps these while the app is open: an agent page's unsaved form, each agent's name for its page title,
// the copy Duplicate hands to the page it opens, and the agent just added, which the list then shows and focuses.
const DRAFTS = new Map();
const NAMES = new Map();
let seed = null;
let reveal = '';
const GONE = new Error('The agent is no longer on the list.');
const LIVE_FLAGS = ['enabled', 'review_eligible', 'rechecks_flags'];
const landed = (outcome) => outcome.state === 'saved' || outcome.state === 'saved_then_failed';
const FILTERS = [['all', 'All'], ['reviewers', 'Reviewers'], ['off', 'Off'], ['unavailable', 'Unavailable']];
const inFilter = (row, state, key) => key === 'all' || (key === 'reviewers' ? row.review_eligible === true
    : key === 'off' ? row.enabled === false : rowStatus(row, state).tone === 'warn');

/** What a roster write says, in the one wording of every row; a list that changed elsewhere says so. */
export function rosterResult(outcome) {
    if (outcome?.state === 'changed_elsewhere') return { text: 'The list changed elsewhere. This is how it stands now.', tone: 'warn' };
    const line = outcomeLine({ control: { kind: 'switch' }, writer: 'checker' }, { ...outcome, token: '' });
    return line ? { text: [line.text, line.note].filter(Boolean).join('. '), tone: line.tone, details: line.details } : {};
}

/** One editor instance. Setup draws every card (head and form) and saves with its Finish; with `page` (Settings:
 *  `{ entry, ctx, reload }`) it draws the list of heads (`entry` '') or one agent's page ('new', or its id). */
export function createAvailableSubagentsEditor({
    hostId = 'available-subagents-editor',
    doc = () => (typeof document === 'undefined' ? null : document),
    win = () => (typeof window === 'undefined' ? null : window),
    store = claudexorStatus,
    onChange = () => {},
    previewGenerated = null,
    baseline = 'saved',
    page = null,
} = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    const getWin = typeof win === 'function' ? win : () => win;
    let disposeChoosers = () => {};
    const state = {
        loaded: false, destroyed: false, parseError: '', setting: { enabled: true, items: [] }, source: '', diagnostics: [],
        dirty: false, saveAttempted: false, baseline: baseline === 'generated' ? 'generated' : 'saved', automatic: false,
        statusError: '', catalogKnown: false, accountsKnown: false, quotaKnown: false, snapshot: null,
        apiModels: [], modelSources: [], modelCatalogNote: '', processingPreference: '',
        // Sources with a stored key, and every connection for a saved one's label (setSourceContext).
        providers: [], connections: [], providerProfiles: {},
        signature: '', statusDisposer: null, catalogDisposer: null, previewSignature: '', previewGeneration: 0,
        // GET /api/review-pool facts about the SAVED catalog; prices are keyed by
        // the loaded route so an edited row never shows its old route's price.
        reviewPool: null, reviewCosts: new Map(), loadedItems: [], loadedFingerprint: '', allowEmptyReviewPool: false,
        // `${_uiKey}:${kind}` of each open row disclosure: a repaint or a reload re-opens it.
        openDisclosures: new Set(),
        // Settings: the list's filter, the Reviewer caption of the saved strictness, the agent this page edits (an
        // unsaved one by `unsavedId`), and the last write's result, which stays through a repaint.
        filter: 'all', tickCaption: '', entryId: '', unsavedId: '', result: null,
    };
    const listView = Boolean(page) && !page.entry;
    const current = () => state.setting.items.find((row) => row.subagent_id === state.entryId) || null;
    // In Settings a saved agent's on/off and marks write at once; an unsaved one keeps them as fields of its form.
    const live = (row) => Boolean(page) && row.subagent_id !== state.unsavedId;
    const said = (text) => (page ? namedRows(text, state) : text);

    const host = () => getDoc()?.getElementById?.(hostId) || null;

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
        // Malformed saved bytes and a preview that has not arrived are both said, never saved over.
        if (!state.loaded) return [state.parseError || 'Available subagents draft is still loading. Retry the preview before finishing.'];
        if (state.parseError) return [state.parseError];
        return [
            ...validateAvailableSubagentsSetting(state.setting, { uniqueEngines: state.dirty, processingPreference: state.processingPreference }),
            ...poolErrors(),
        ];
    }

    // Judged like the server: a generated draft, or one that differs from what was loaded. A write of Settings is
    // judged by the server itself, whose refusal asks (writeEntry).
    function poolErrors() {
        if (page) return [];
        const judged = state.baseline === 'generated'
            || JSON.stringify(buildAvailableSubagentsSetting(state.setting)) !== state.loadedFingerprint;
        return reviewPoolErrors(state.setting, { judged, allowEmpty: state.allowEmptyReviewPool });
    }

    function renderPoolSummary(container) {
        const summary = reviewPoolSummary(state, Boolean(page));
        const intent = editorIntent(state);
        // Each line of the pool: its words where it has any, and whether it shows.
        for (const [name, text, shown] of [['count', summary.count], ['stays', summary.stays, summary.stays],
            ['empty', null, summary.empty], ['empty-text', summary.empty], ['confirm', null, summary.confirm],
            ['copies', null, summary.copies], ['copies-text', said(summary.copies)], ['note', summary.note, summary.note],
            ['history', null, summary.history], ['history-text', summary.history]]) {
            const node = container.querySelector(`[data-review-pool-${name}]`);
            if (!node) continue;
            if (text !== null) node.textContent = text;
            if (shown !== undefined) node.hidden = !shown;
        }
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
            if (event.target.value) row.effort = String(event.target.value); else delete row.effort;
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
        if (named) facet.dataset.named = named; else { delete facet.dataset.named; bindEffortFacet(el, row); }
    }

    /** Patch every verdict in place; answers how many errors it shows. */
    function renderValidation() {
        const container = host();
        if (!container) return 0;
        const structural = !state.loaded || Boolean(state.parseError);
        const shown = structural ? validationErrors()
            : (state.saveAttempted ? [...listLevelErrors(state.setting), ...poolErrors()] : []);
        renderPoolSummary(container);
        if (listView) return shown.length;  // a head holds no form: its facts are drawn with it
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
                // A page of Settings has room for the sentence; a card's head keeps the word, the sentence its title.
                Object.assign(statusEl, page ? { textContent: status.text } : { textContent: status.label, title: status.text });
                statusEl.dataset.tone = status.tone;
            }
            // Every line below the head is patched in place: a Reviewer or row switch never rebuilds the card.
            const show = (selector, text) => {
                const node = el.querySelector(selector);
                if (node) Object.assign(node, { textContent: text, hidden: !text });
            };
            show('[data-subagent-status-reason]', rowStatusReason(status));
            show('[data-subagent-review-exception]', reviewException(row));
            show('[data-subagent-review-notes]', said(reviewRepeat(row, state, index)));
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
            Object.assign(metaEl, { hidden: !meta.text, textContent: said(meta.text) });
            metaEl.toggleAttribute('data-availability-qualifier', Boolean(meta.qualifier));
            if (meta.tone) metaEl.dataset.tone = meta.tone;
            else delete metaEl.dataset.tone;
        });
        const box = container.querySelector('[data-subagents-validation]');
        if (box) Object.assign(box, { hidden: !shown.length, textContent: said(shown[0] || '') });
        return shown.length;
    }

    function noteSaveAttempt() {
        state.saveAttempted = true;
        state.setting.items.forEach((row) => { row._uiAttempted = true; });
        renderValidation();
        state.signature = availableSubagentsRenderSignature(state);
    }

    function markDirty({ structural = false } = {}) {
        state.dirty = true;
        // Text and fixed choices hold their draft; status keeps the same nodes.
        state.signature = structural ? '' : availableSubagentsRenderSignature(state);
        showResult(null);  // an edit starts a new subject: the last result goes
        host()?.querySelector?.('[data-editor-discard]')?.removeAttribute('hidden');  // an agent's page says it holds unsaved text
        renderValidation();
        onChange(buildAvailableSubagentsSetting(state.setting));
    }

    // A new row's hidden keys are neutral: a label copied from its source would rot with the route.
    const mintRowKeys = () => ({
        subagent_id: mintStableId('subagent', state.setting.items.map((item) => item.subagent_id)),
        _uiKey: mintStableId('actor_row', state.setting.items.map((item) => item._uiKey)),
    });

    function showResult(outcome) {
        state.result = outcome ? rosterResult(outcome) : null;
        setResult(host()?.querySelector?.('[data-entry-result], [data-editor-result]'), state.result || {});
    }

    // Settings: one write of the roster. A fresh read, the one entry changed, and the same body again after the
    // last-reviewer question (writeEntry); every write is the checker's transaction, and `edit` changes what no
    // single entry holds (the roster's own switch, the one re-check mark, the copies). A list never saved is written
    // as shown, without this page's unsaved form. The page then shows what the server holds.
    async function write(id, change, { edit = null, name = '', busy = null, settle = null, quiet = false } = {}) {
        const doc = getDoc();
        const held = busy && doc?.activeElement === busy ? focusSnapshot(host(), doc) || busy : null;  // disabling drops its focus
        setButtonBusy(busy, true);
        if (!quiet) showResult(null);  // a new write is a new subject: its result, when it comes, is seen to come
        let outcome;
        try {
            outcome = await writeEntry(KEY, id, change, { name, ask: page.ctx.ask,
                shown: page.entry ? JSON.parse(state.loadedFingerprint || 'null') : buildAvailableSubagentsSetting(state.setting),
                checker: ({ body }) => saveCheckerSettings({ [KEY]: edit ? edit(body[KEY]) : body[KEY] }, {
                    allowEmptyReviewPool: body[ALLOW_EMPTY_REVIEW_POOL] === true, ifUnchanged: body.if_unchanged, headers: APP_SETTINGS_ORIGIN }) });
        } catch (error) { if (error !== GONE) throw error; outcome = { state: 'changed_elsewhere' }; }
        if (landed(outcome)) settle?.();
        state.html = '';  // drawn again from what the server holds: a refused or declined switch returns to its stored value
        await page.reload();
        setButtonBusy(busy, false);
        if (held && doc.activeElement === doc.body) { if (held === busy) busy.focus?.(); else restoreFocus(host(), held); }
        if (!quiet) showResult(outcome);
        return outcome;
    }

    // An agent page's Save: the form goes into its own entry of the list read afresh. A saved agent keeps the on/off
    // and the marks the server holds (its switches write those); a new one brings its own.
    async function saveEntry(button) {
        const row = current();
        state.saveAttempted = row._uiAttempted = true;
        if (renderValidation()) return;
        const fresh = !live(row);
        const outcome = await write(row.subagent_id, (before) => {
            if (Boolean(before) === fresh) throw GONE;  // removed elsewhere, or its new id was taken: nothing is put back
            const next = canonicalRow(row);
            if (before) for (const flag of LIVE_FLAGS) { delete next[flag]; if (before[flag] !== undefined) next[flag] = before[flag]; }
            return next;
        }, { busy: button, settle: () => { state.dirty = false; } });
        if (!fresh || !landed(outcome)) return;
        state.dirty = false;  // the new agent is on the list now: this page keeps no form, and the list shows it
        reveal = row.subagent_id;
        page.ctx.openSettings('agents');
    }

    function bindRows(container) {
        container.querySelectorAll?.('[data-subagent-row]').forEach((rowElement) => {
            const row = state.setting.items.find((item) => (item._uiKey || item.subagent_id) === rowElement.dataset.subagentRow);
            if (!row) return;
            const open = () => page.ctx.openSettings(`agents/${row.subagent_id}`);
            rowElement.querySelector('[data-subagent-field="open"]')?.addEventListener('click', open);
            if (listView) rowElement.addEventListener('click', (event) => { if (!event.target.closest?.('button, .ui-switch')) open(); });
            rowElement.querySelector('[data-subagent-field="recommended_use"]')?.addEventListener('input', (event) => {
                row.recommended_use = String(event.target.value || '');
                markDirty();
            });
            // The switch and the Reviewer mark: setup holds each as a draft like every other field (stored only when it
            // is not the default) and patches its lines in place; in Settings a saved agent's is one write of that
            // entry, never of an open form.
            for (const [field, base] of [['enabled', true], ['review_eligible', false]]) {
                rowElement.querySelector(`[data-subagent-field="${field}"]`)?.addEventListener('change', (event) => {
                    const on = Boolean(event.target.checked);
                    if (live(row)) return void write(row.subagent_id, { [field]: on }, { busy: event.target });
                    if (on === base) delete row[field];
                    else row[field] = on;
                    return markDirty();
                });
            }
            rowElement.querySelector('[data-subagent-field="rechecks_flags"]')?.addEventListener('change', (event) => {
                const mark = event.target.checked ? row.subagent_id : '';
                void write(row.subagent_id, {}, { busy: event.target, edit: (roster) => markRecheckRow(roster, mark) });
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
            rowElement.querySelector('[data-subagent-field="route"]')?.addEventListener('change', (event) => {
                row.route = changeRouteChoice(row.route, event.target.value);
                if (row.route.kind !== ROUTE_KIND_AGENT_SESSION) delete row.access;
                if (namedRowEffort(row)) delete row.effort;  // the owner's pick of a named model retires the pin
                markDirty({ structural: true });
                paint();
            });
            rowElement.querySelector('[data-subagent-field="model"]')?.addEventListener('input', (event) => {
                const previous = encodeRouteChoice(row);
                row.route.target_id = routeTargetFromModel(row.route, event.target.value);
                if (namedRowEffort(row)) delete row.effort;  // the owner's pick of a named model retires the pin
                const structural = previous !== encodeRouteChoice(row);
                if (structural) delete row.route.credential_profile_id;
                markDirty({ structural });
                if (structural) paint(); else renderValidation(); // keeps the disclosed id current, caret intact
            });
            rowElement.querySelector('[data-subagent-field="account"]')?.addEventListener('change', (event) => {
                const pin = String(event.target.value || '');
                if (pin) row.route.credential_profile_id = pin;
                else delete row.route.credential_profile_id;
                markDirty({ structural: true });
                paint();
            });
            // Processing, Access and Delivery: a plain choice, stored unless it is the default (inherit; Reads the work itself).
            for (const field of ['processing_preference', 'access', 'delivery']) {
                rowElement.querySelector(`[data-subagent-field="${field}"]`)?.addEventListener('change', (event) => {
                    const value = String(event.target.value || '');
                    if (value && value !== 'native') row[field] = value;
                    else delete row[field];
                    markDirty();
                    if (field === 'processing_preference') paint();  // its suggestions' prices follow the tier
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
                if (page) { seed = copy; page.ctx.openSettings('agents/new'); return; }  // an unsaved agent's own page
                state.setting.items.splice(state.setting.items.indexOf(row) + 1, 0, copy);
                markDirty({ structural: true });
                paint();
                revealRow(copy._uiKey);
            });
            rowElement.querySelector('[data-subagent-remove]')?.addEventListener('click', async (event) => {
                if (page) {  // Settings asks first (writeEntry), removes at once and returns to the list
                    const beside = [1, -1].map((step) => state.setting.items[state.setting.items.indexOf(row) + step]).find(Boolean);
                    const next = live(row) ? beside || { subagent_id: '+' } : null;
                    if (next && !landed(await write(row.subagent_id, null, { name: agentName(row, state), busy: event.target }))) return;
                    state.dirty = false;
                    reveal = next?.subagent_id || '';  // the caret goes to the agent now in its place, else to Add
                    page.ctx.openSettings('agents');
                    return;
                }
                const index = state.setting.items.indexOf(row);
                if (index >= 0) state.setting.items.splice(index, 1);
                markDirty({ structural: true });
                paint();
                // The caret moves to the entry now in this place, else to Add (DESIGN "List editors").
                const next = state.setting.items[Math.min(index, state.setting.items.length - 1)];
                if (next) revealRow(next._uiKey || next.subagent_id);
                else host()?.querySelector('[data-subagent-add]')?.focus();
            });
        });
    }

    function paint({ discoveryOnly = false } = {}) {
        const container = host();
        if (!container || state.destroyed) return false;
        const nextSignature = availableSubagentsRenderSignature(state);
        if (nextSignature === state.signature || (page && !state.loaded && !state.parseError)) return false;  // Settings draws once it has read
        const focused = focusSnapshot(container, getDoc());
        state.signature = nextSignature;
        const errors = validationErrors();
        const diagnostics = diagnosticsText(state.diagnostics);
        const readProblem = [state.statusError
            ? 'Live agent availability could not be read. Saved rows remain unchanged.' : '', state.modelCatalogNote].filter(Boolean).join(' ');
        if (discoveryOnly && container.querySelector('.agent-page, .available-subagents-toolbar')) {
            state.setting.items.forEach((row, index) => {
                const el = container.querySelector(`[data-subagent-row="${row._uiKey || row.subagent_id}"]`);
                if (!el) return;
                const template = getDoc().createElement('template');
                template.innerHTML = availableSubagentRowMarkup(row, state, index);
                const desired = template.content.firstElementChild;
                updateRouteControlOptions(el, desired);
                const identity = el.querySelector('.available-subagent-route-identity-wrap');
                if (identity) identity.innerHTML = desired.querySelector('.available-subagent-route-identity-wrap').innerHTML;
            });
            Object.assign(container.querySelector('[data-subagents-read-problem]'), { textContent: readProblem, hidden: !readProblem });
            Object.assign(container.querySelector('[data-subagents-diagnostics]'), { textContent: diagnostics.join(' · '), hidden: !diagnostics.length });
            renderValidation();
            return true;
        }
        disposeChoosers();
        const scroller = container.closest?.('.settings-page'), top = scroller?.scrollTop;  // Settings scrolls in the level: a redraw keeps the place
        const pool = reviewPoolSummary(state, Boolean(page));
        const intent = editorIntent(state);
        const items = state.setting.items;
        const full = !state.loaded || items.length >= MAX_AVAILABLE_SUBAGENTS;
        const unloaded = '<div class="available-subagents-empty">The saved configuration could not be loaded, so this editor will not replace it.</div>';
        const facts = `<div class="available-subagents-source" data-subagents-read-problem ${readProblem ? '' : 'hidden'}>${escapeHtml(readProblem)}</div>
            <div data-subagents-diagnostics class="available-subagents-diagnostics" ${diagnostics.length ? '' : 'hidden'}>${escapeHtml(diagnostics.join(' · '))}</div>`;
        const above = `<div class="available-subagents-source" data-review-pool-stays ${pool.stays ? '' : 'hidden'}>${escapeHtml(pool.stays)}</div>
            <div class="available-subagents-diagnostics" data-review-pool-note ${pool.note ? '' : 'hidden'}>${escapeHtml(pool.note)}</div>
            <details class="model-role-details ui-disclosure" data-review-pool-history ${pool.history ? '' : 'hidden'}><summary>Reviewer origin</summary>
                <div class="available-subagents-source" data-review-pool-history-text>${escapeHtml(pool.history)}</div></details>
            ${facts}`;
        // Lines a row edit can toggle (the empty pool, the validation summary) sit below the rows:
        // a conditional line never moves the control that caused it (docs/DESIGN.md §6).
        const below = `<div class="available-subagents-review-empty" data-review-pool-empty data-tone="warn" ${pool.empty ? '' : 'hidden'}>
                <span data-review-pool-empty-text>${escapeHtml(pool.empty)}</span>
                <label class="local-toggle ui-field-inline" data-review-pool-confirm ${pool.confirm ? '' : 'hidden'}><input class="ui-checkbox" type="checkbox" data-review-pool-allow-empty ${state.allowEmptyReviewPool ? 'checked' : ''}> Save without reviewers</label>
            </div>
            <div class="available-subagents-review-copies" data-review-pool-copies data-tone="warn" ${pool.copies ? '' : 'hidden'}>
                <span data-review-pool-copies-text>${escapeHtml(said(pool.copies))}</span>
                <button type="button" class="btn btn-default" data-review-pool-keep-one>Keep one per model</button>
            </div>
            <div data-subagents-validation class="available-subagents-diagnostics" data-tone="error" ${errors.length ? '' : 'hidden'}>${escapeHtml(errors[0] || '')}</div>`;
        if (page?.entry) {
            const row = current();
            container.innerHTML = !state.loaded ? unloaded : !row ? '<p class="available-subagents-empty">This agent is no longer on the list.</p>'
                : `${agentPageHtml(row, state, items.indexOf(row), live(row))}${facts}`;
            container.querySelector('[data-editor-save]')?.addEventListener('click', (event) => void saveEntry(event.target));
            container.querySelector('[data-editor-discard]')?.addEventListener('click', () => { state.dirty = false; void page.reload(); });
        } else if (page) {
            const count = (key) => items.filter((row) => inFilter(row, state, key)).length;
            const heads = items.map((row, index) => (inFilter(row, state, state.filter)
                ? `<div class="ui-row settings-head" data-subagent-row="${escapeHtml(row._uiKey)}" data-entry-id="${escapeHtml(row.subagent_id)}">${subagentHeadHtml(row, state, index, true)}</div>` : '')).join('');
            const html = `<section class="ui-card settings-card">
                <div class="settings-card-head"><h3 class="settings-card-title agents-card-title">${items.length} agent${items.length === 1 ? '' : 's'}</h3>
                    <span class="ui-result" data-entry-result role="status" aria-live="polite"></span>
                    <div class="ui-section-actions"><button type="button" class="btn btn-default btn-sm" data-subagent-add ${full ? 'disabled' : ''}>Add agent</button></div></div>
                <div class="agents-filters">${FILTERS.filter(([key]) => key === 'all' || key === state.filter || count(key)).map(([key, label]) =>
                    `<button type="button" class="filter-chip${key === state.filter ? ' active' : ''}" data-agents-filter="${key}" aria-pressed="${key === state.filter}">${label} ${count(key)}</button>`).join('')}</div>
                ${state.source === 'undecided' || state.automatic ? `<p class="available-subagents-source" data-agents-automatic>Automatic: this list follows your Main model and accounts. Your first change here saves it as yours.</p>` : ''}
                ${above}
                <div class="settings-rows">${!state.loaded ? unloaded : heads || `<p class="available-subagents-empty">${items.length ? 'No agent here.'
                    : 'No agents yet, so the main model is the one agent. Add one to choose others.'}</p>`}</div>
                ${below}
                <p class="available-subagents-source">A change applies to new tasks: a task that is already running keeps the reviewers it started with.${
                    items.length >= MAX_AVAILABLE_SUBAGENTS ? ` The list holds at most ${MAX_AVAILABLE_SUBAGENTS} agents.` : ''}</p>
            </section>`;
            if (html === state.html) return true;  // nothing in view changed (a status tick, the pool read): focus and listeners stay
            container.innerHTML = state.html = html;
            container.querySelectorAll('[data-agents-filter]').forEach((button) => button.addEventListener('click', () => {
                Object.assign(state, { filter: button.dataset.agentsFilter, signature: '' });
                paint();
                host()?.querySelector(`[data-agents-filter="${state.filter}"]`)?.focus({ preventScroll: true });
            }));
        } else {
            container.innerHTML = `
            <div class="available-subagents-toolbar">
                <label class="local-toggle ui-field ui-field-inline" title="Off: new tasks do not delegate to these rows; rows marked Reviewer still review.">
                    <input class="ui-checkbox" type="checkbox" data-subagents-enabled aria-label="Delegation to these subagents" ${state.setting.enabled ? 'checked' : ''} ${state.loaded ? '' : 'disabled'}>
                    Delegation
                </label>
                <span class="available-subagents-count">${items.length}/${MAX_AVAILABLE_SUBAGENTS}</span>
                <span class="available-subagents-count" data-review-pool-count>${escapeHtml(pool.count)}</span>
                <span class="settings-inline-status available-subagents-intent" data-subagents-intent data-tone="${intent.tone}" title="${escapeHtml(intent.title)}">${intent.label}</span>
                <button type="button" class="btn btn-default" data-subagent-add ${full ? 'disabled' : ''}>Add subagent</button>
            </div>
            ${above}
            <div class="available-subagents-list">
                ${state.loaded
                    ? items.map((row, index) => availableSubagentRowMarkup(row, state, index)).join('')
                        || '<div class="available-subagents-empty">No subagents configured, so the main model is the one agent. Add one to choose others.</div>'
                    : unloaded}
            </div>
            ${below}`;
        }
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
        // The copies leave the list, the first row of each model stays: a draft edit in setup; in Settings it asks,
        // then removes at once the copies of the list the server holds then.
        container.querySelector('[data-review-pool-keep-one]')?.addEventListener('click', async (event) => {
            const copiesOf = (roster) => new Set(reviewPoolCopies(roster, state.processingPreference).flatMap((group) => group.copies));
            const copies = copiesOf(state.setting);
            if (!copies.size) return;
            const without = (roster, gone) => ({ ...roster, items: roster.items.filter((_row, index) => !gone.has(index)) });
            if (page) {
                const button = event.target;
                if (await CONFIRMS.removeEntry({ name: copies.size === 1 ? '1 copy' : `${copies.size} copies` }, null, null, page.ctx.ask)) {
                    void write('', () => null, { busy: button, edit: (roster) => without(roster, copiesOf(roster)) });
                }
                return;
            }
            state.setting.items = without(state.setting, copies).items;
            markDirty({ structural: true });
            paint();
        });
        container.querySelector('[data-subagent-add]')?.addEventListener('click', () => {
            if (items.length >= MAX_AVAILABLE_SUBAGENTS) return;
            if (page) { page.ctx.openSettings('agents/new'); return; }
            const row = { recommended_use: '', route: { kind: ROUTE_KIND_API_MODEL, target_id: '' }, ...mintRowKeys() };
            state.setting.items.push(row);
            markDirty({ structural: true });
            paint();
            revealRow(row._uiKey);
        });
        if (scroller) scroller.scrollTop = top;
        bindRows(container);
        disposeChoosers = bindModelChoosers(container);
        restoreFocus(container, focused);
        renderValidation();
        setResult(container.querySelector('[data-entry-result], [data-editor-result]'), state.result || {});
        const added = listView && reveal ? items.find((row) => row.subagent_id === reveal) : null;
        if (added) { reveal = ''; revealRow(added._uiKey); }
        else if (listView && reveal === '+' && state.loaded && !items.length) { reveal = ''; container.querySelector('[data-subagent-add]')?.focus(); }
        return true;
    }

    // Reveal the new row after the painter restores previous focus.
    function revealRow(uiKey) {
        const row = host()?.querySelector?.(`[data-subagent-row="${uiKey}"]`);
        revealNewRow(row, row?.querySelector?.('[data-subagent-field="recommended_use"], [data-subagent-field="open"]'));
    }

    // An agent's page over the list just read: its unsaved form stays (this page's, or the one kept when it was left),
    // while a saved agent's on/off and marks are the stored ones. A new agent takes keys the list does not hold.
    function openEntry(draft) {
        const items = state.setting.items;
        if (page.entry === 'new') {
            const { subagent_id: _id, _uiKey: _key, ...form } = seed || draft || { recommended_use: '', route: { kind: ROUTE_KIND_API_MODEL, target_id: '' } };
            seed = null;
            items.push({ ...form, ...mintRowKeys() });
            Object.assign(state, { entryId: items.at(-1).subagent_id, unsavedId: items.at(-1).subagent_id, dirty: true });
            return;
        }
        state.entryId = page.entry;
        const at = items.findIndex((row) => row.subagent_id === page.entry);
        if (at < 0 || !draft) return;
        const stored = Object.fromEntries(LIVE_FLAGS.filter((flag) => items[at][flag] !== undefined).map((flag) => [flag, items[at][flag]]));
        const { enabled: _on, review_eligible: _mark, rechecks_flags: _recheck, ...form } = draft;
        items[at] = { ...form, ...stored, _uiKey: items[at]._uiKey };
        state.dirty = true;
    }

    function load(value, { source = '', diagnostics = [], automatic = false } = {}) {
        // Invalidate a preview launched for the previous settings document.
        // A late response must never overwrite a freshly loaded configured row.
        state.previewGeneration += 1;
        state.previewSignature = '';
        const draft = page?.entry && state.dirty ? current() : null;
        const parsed = parseAvailableSubagentsSetting(value);
        state.loaded = Boolean(parsed.setting);
        state.parseError = parsed.error;
        if (parsed.setting) state.setting = attachUiKeys(parsed.setting, state.setting.items);
        const loadedSetting = parsed.setting ? buildAvailableSubagentsSetting(parsed.setting) : null;
        Object.assign(state, {
            loadedItems: loadedSetting?.items || [], loadedFingerprint: loadedSetting ? JSON.stringify(loadedSetting) : '',
            reviewPool: null, reviewCosts: new Map(), allowEmptyReviewPool: false, source: String(source || ''),
            automatic: automatic === true, diagnostics, dirty: false, saveAttempted: false, signature: '',
        });
        if (page) state.setting.items.forEach((row) => NAMES.set(row.subagent_id, agentName(row, state)));
        if (page?.entry && state.loaded) { openEntry(draft || DRAFTS.get(page.entry)); DRAFTS.delete(page.entry); }  // the form lives in this page now
        paint();
        return { loaded: state.loaded, error: state.parseError };
    }

    function applyGeneratedPreview(response) {
        const parsed = parseAvailableSubagentsSetting(response?.available_subagents);
        state.source = String(response?.source || state.source || 'onboarding_default');
        state.diagnostics = response?.diagnostics || [];
        const canApply = generatedPreviewCanReplace({ dirty: state.dirty, parsedSetting: parsed.setting });
        const sameAssignment = parsed.setting && state.loaded
            && JSON.stringify(buildAvailableSubagentsSetting(parsed.setting)) === JSON.stringify(buildAvailableSubagentsSetting(state.setting));
        if (canApply) {
            Object.assign(state, { loaded: true, parseError: '', saveAttempted: false });
            if (!sameAssignment) state.setting = attachUiKeys(parsed.setting, state.setting.items);
        } else if (!parsed.setting && !state.loaded) state.parseError = parsed.error;
        state.signature = '';
        paint({ discoveryOnly: state.loaded && (!canApply || sameAssignment) });
        return { applied: canApply, error: parsed.error };
    }

    function setPreviewFailure(error) {
        const message = String(error?.body?.detail || error?.body?.error || error?.message || error);
        const code = String(error?.body?.code || '').trim();
        state.diagnostics = [`${code ? `${code}: ` : ''}${message}`, ...diagnosticsText(error?.body?.diagnostics)];
        if (!state.loaded) state.parseError = `Available subagents preview failed: ${state.diagnostics.join(' · ')}`;
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
        if (typeof previewGenerated !== 'function' || state.source !== 'undecided' || state.dirty) return false;
        const connected = state.accountsKnown ? [...connectedHarnessIds(state.snapshot)].sort() : [];
        const signature = JSON.stringify([state.accountsKnown, connected]);
        if (!force && signature === state.previewSignature) return false;
        state.previewSignature = signature;
        const generation = ++state.previewGeneration;
        try {
            const response = await previewGenerated({ subscriptionsConnected: state.accountsKnown && connected.length > 0 });
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

    function adoptCatalog(data) {
        const catalog = mergeModelCatalog(state.modelCatalog || {items: state.apiModels, model_sources: state.modelSources}, data);
        state.modelCatalog = catalog;
        state.modelSources = catalog.model_sources;
        state.apiModels = catalog.items;
        state.modelCatalogNote = catalogReadNote(catalog);
        state.signature = '';
        paint({discoveryOnly: true});
    }

    // The suggestions' prices are composed words the overlay leaves alone: a language change paints them again.
    const onLanguageChanged = () => { if (!state.destroyed) paint(); };

    function mount({ bindStatus = true } = {}) {
        getWin()?.addEventListener?.('ouro:language-changed', onLanguageChanged);
        adoptStatus();
        if (bindStatus && !state.statusDisposer) {
            state.statusDisposer = bindStatusSurface(store, { elementId: hostId, includeModels: true, doc: getDoc, win: getWin,
                listener: () => {
                    adoptStatus();
                    paint({ discoveryOnly: true });
                    void maybeRefreshGeneratedPreview();
                } });
        }
        if (!state.catalogDisposer) {
            const target = getDoc();
            const onCatalog = (event) => adoptCatalog(event?.detail);
            target?.addEventListener?.('settings-model-catalog:updated', onCatalog);
            state.catalogDisposer = () => target?.removeEventListener?.('settings-model-catalog:updated', onCatalog);
        }
        state.signature = '';
        paint();
    }

    function destroy() {
        // An agent page that is left keeps its unsaved form for the next opening; a clean one keeps nothing.
        if (page?.entry) { if (state.dirty && current()) DRAFTS.set(page.entry, current()); else DRAFTS.delete(page.entry); }
        Object.assign(state, { destroyed: true, previewGeneration: state.previewGeneration + 1 });
        getWin()?.removeEventListener?.('ouro:language-changed', onLanguageChanged);
        disposeChoosers();
        for (const disposer of ['statusDisposer', 'catalogDisposer']) { state[disposer]?.(); state[disposer] = null; }
    }

    function setSourceContext({ settings = {}, providerProfiles = {} } = {}) {
        state.providerProfiles = providerProfiles || {};
        state.connections = apiConnections(settings, state.providerProfiles);
        state.providers = configuredApiProviders(settings, state.providerProfiles);
        paint({ discoveryOnly: true });
    }

    return {
        mount,
        destroy,
        load,
        paint,
        write,
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
        /** Settings: the document just read. A never-saved roster shows the candidate the server proposes. */
        adopt(settings) {
            const meta = settings?._meta?.available_subagents || {};
            state.processingPreference = String(settings?.[PROCESSING_PREFERENCE_KEY] || '');
            state.tickCaption = reviewerTickCaption(settings?.OUROBOROS_RUNTIME_MODE);
            setSourceContext({ settings, providerProfiles: settings?._meta?.setup_contract?.providerProfiles });
            return load(availableSubagentsLoadValue(settings), {
                source: String(meta.source || 'configured'), diagnostics: meta.diagnostics || meta.diagnostic || [], automatic: meta.automatic === true });
        },
        setPreviewFailure, adoptCatalog,
        validate: validationErrors,
        noteSaveAttempt,
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
        setSourceContext,
    };
}

export function availableSubagentsEditorHost(hostId = 'available-subagents-editor') {
    return `<div id="${escapeHtml(hostId)}" class="available-subagents-editor"><div class="available-subagents-empty">Loading Available subagents…</div></div>`;
}

// What the Reviewer tick means under the SAVED Access, one line each (IMPL_SPEC §5): Cyber Pro composes freely
// (review_helpers.cyber_review_authority).
export const REVIEWER_TICK_CAPTIONS = Object.freeze({
    advisory: 'Counts when chosen: for each check Ouroboros asks one, several or all reviewers and writes why; it never skips a check that is owed.',
    cyber_pro: 'Trusted to judge: Ouroboros decides whom to ask, if anyone; reviews inform and never block.',
});
export const reviewerTickCaption = (access) => REVIEWER_TICK_CAPTIONS[String(access || '').trim().toLowerCase() === 'cyber_pro'
    ? 'cyber_pro' : 'advisory'];

export function availableSubagentsLoadValue(settings) {
    const raw = settings?.OUROBOROS_SUBAGENTS;
    if (raw !== undefined && raw !== null && raw !== '') return raw;
    return settings?._meta?.available_subagents?.candidate ?? raw;
}

/** Price, last-run and pool facts for the saved catalog; a failed read is said, never guessed. */
export async function readReviewPool() {
    try {
        const resp = await apiFetch('/api/review-pool', { cache: 'no-store' });
        const payload = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(payload.error || `HTTP ${resp.status}`);
        return payload;
    } catch (error) { return { load_error: `Review pool facts could not be read: ${error.message || error}` }; }
}

// The last-reviewer question of every roster write, in the owner's words (the server's sentence names a request flag).
CONFIRMS.emptyReviewPool = (row, refused, current, ask) => ask({ title: 'Save without reviewers?', confirmLabel: 'Save without reviewers',
    danger: true, body: 'No agent marked Reviewer would be on, so the main model reviews alone: one model, no second opinion.' });
// Helpers may change files: stored as 'true', 'false' or nothing (Auto follows Access: config.py
// get_allow_mutative_subagents). An older spelling of yes or no reads as the same choice.
CODECS.helperMutation = { read(settings, row) {
    const text = String(settings?.[row.keys[0]] ?? '').trim().toLowerCase();
    return !settings ? undefined : /^(1|true|yes|on)$/.test(text) ? 'true' : /^(0|false|no|off)$/.test(text) ? 'false' : '';
} };

let roots = 0;
/** Settings › Agents (settings_pages.js): the list (`agents`), or one agent's page (`agents/<id>`, `agents/new`). It
 *  reads when it is opened and whenever the shell says it is shown again; an unsaved form survives each read. */
function mountAgents(container, { path, ctx }) {
    const entry = path.split('/')[1] || '';
    const hostId = `agents-root-${++roots}`;
    const root = `<div id="${hostId}" class="settings-stack agents-root"></div>`;
    container.innerHTML = entry ? root : `<p class="settings-page-lead">Helpers Ouroboros may hand work to, and reviewers whose word counts.</p>
        <section class="ui-card settings-card">${settingsRow({ ...rowById(KEY), control: { kind: 'switch' } })}</section>
        ${root}
        <section class="ui-card settings-card"><details class="ui-disclosure" data-settings-fold>
            <summary>More: limits for helpers and their folders</summary>
            <div class="settings-rows" data-agents-fold></div>
        </details></section>`;
    const unbindRechecks = bindRecheckMarks(container);
    const foldRows = container.querySelector('[data-agents-fold]');
    let fold = null;
    const roster = container.querySelector(`[data-setting-row="${KEY}"]`);
    const delegation = roster?.querySelector('[data-setting-control]');
    let reading = null;
    const read = async () => {
        let settings;
        // A failed read keeps what the page showed (the next showing reads again); with nothing shown yet it says so.
        try { settings = await ctx.readSettings(); } catch { if (!editor.loaded) editor.load(undefined); return; }
        syncRecheckState(recheckFromSettings(settings), container);
        editor.adopt(settings);
        // The fold's rows are drawn once their values are read: a row the page was opened for takes the focus filled.
        if (foldRows && !fold) foldRows.innerHTML = rowsOf('agents', { fold: true }).map((row) => settingsRow(row)).join('');
        fold = fold || (foldRows ? bindSettingRows(foldRows) : null);
        fold?.paint();
        if (delegation) Object.assign(delegation, { checked: editor.setting.enabled, disabled: !editor.loaded }).setAttribute('aria-checked', String(editor.setting.enabled));
        void readReviewPool().then((payload) => editor.setReviewPool(payload));
        void editor.reloadStatus().then(() => accountCatalog?.arm());
    };
    // One read at a time: opening a page and the shell's "shown" for it are one read.
    const reload = () => (reading ||= read().finally(() => { reading = null; }));
    const editor = createAvailableSubagentsEditor({
        hostId, page: { entry, ctx, reload },
        previewGenerated: entry ? null : ({ subscriptionsConnected }) => apiClient.previewOnboardingSubagents({ subscriptionsConnected }),
    });
    delegation?.addEventListener('change', async () => {
        const enabled = delegation.checked;
        const outcome = await editor.write('', () => null, { busy: delegation, quiet: true, edit: (roster) => ({ ...roster, enabled }) });
        setResult(roster.querySelector('[data-setting-result]'), rosterResult(outcome));
    });
    // The model chooser of an agent's form: its catalogue is read when the form opens and whenever it is shown again.
    let catching = null;
    const catchUp = () => { if (entry) catching ||= refreshModelCatalog().finally(() => { catching = null; }); };
    const onShown = (event) => { if (event.detail?.path === path) { void reload(); catchUp(); } };
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    editor.mount();
    // A sign-in that settles while an agent's form is open renews its model list, as on a role's page. The watch is
    // armed by the form's first account read, so the answer the form opened with is not a change.
    const accountCatalog = entry ? watchAccountModelCatalog() : null;
    void reload();
    catchUp();
    return { unmount() {
        globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown);
        accountCatalog?.dispose();
        unbindRechecks(); fold?.dispose(); editor.destroy();
    } };
}

registerSettingsPage('agents', {
    title: (path) => { const id = path.split('/')[1]; return !id ? 'Agents' : id === 'new' ? 'New agent' : NAMES.get(id) || 'Agent'; },
    mount: mountAgents,
});
