// Status and meta projection of one Available-subagents card: pure functions
// of the row, the editor's state and the shared Claudexor snapshot, with no
// DOM, so the editor's markup and its in-place painter read one source and
// node tests pin the words without a browser. Dispatch remains authoritative;
// this module only decides which positive/negative facts a card may honestly
// claim.

import { accountRows, familyLabel, nextUpAccount } from './claudexor_status_store.js';
import {
    ROUTE_KIND_AGENT_SESSION,
    describeExecutionEvidence,
    harnessModelsKnown,
    modelsGapNote,
    routeModelFields,
    sameEngineAs,
    sourceIdentityLabel,
    splitSessionTarget,
    accountScopedModelCatalog,
    sessionModelMatches,
    CLAUDE_BASE_QUALIFIER,
    routeSupportsAccount,
} from './route_editor_primitives.js';

/**
 * The card's source chip (owner decision 4A): the provider behind an API row,
 * the model source behind a subscription, the agent behind a session. A
 * retained daemon product name is evidence only while the catalog read is
 * known — during a gap `familyLabel` falls back to the presentation catalog.
 */
export function rowIdentity(row, state = {}) {
    const session = row?.route?.kind === ROUTE_KIND_AGENT_SESSION;
    // `harness` is the session's own harness OR the subscription source's
    // credential harness; a direct API route has neither and shows the channel
    // mark. Never split the stored target here: only the model-sources catalog
    // maps an opaque source id to a harness.
    const fields = routeModelFields(row?.route, state.modelSources, {
        providerProfiles: state.providerProfiles, connections: state.connections,
    });
    const label = sourceIdentityLabel(row?.route, {
        modelSources: state.modelSources,
        providerProfiles: state.providerProfiles, connections: state.connections,
        harnesses: [{
            id: fields.harness,
            display_name: familyLabel(fields.harness, state.snapshot, { catalogKnown: state.catalogKnown }),
        }],
    });
    return session || fields.subscription
        ? { harnessId: fields.harness || fields.source, label, channel: '' }
        : { harnessId: 'api', label, channel: 'api' };
}

export function harnessMap(snapshot) {
    return Object.fromEntries((snapshot?.harnesses || [])
        .filter((harness) => harness?.id)
        .map((harness) => [String(harness.id), harness]));
}

// The same field ranges as claudexor_observations._INSTANT. Date.parse rolls an
// impossible day forward (Feb 29 of a non-leap year becomes Mar 1), so the day is
// checked against its month here, as Python's datetime does.
const INSTANT = /^(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/;

/** A still-future engine instant in ms, else NaN: invalid timing is no live block. */
function liveInstantMs(value, nowMs) {
    const parts = INSTANT.exec(typeof value === 'string' ? value : '');
    if (!parts) return NaN;
    const [year, month, day] = parts.slice(1, 4).map(Number);
    const leap = (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
    if (day > [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]) return NaN;
    const at = Date.parse(value);
    return at > nowMs ? at : NaN;
}

// One window that measured this request (finite ratio, every model, the kind):
// evidence of an applicable observation, never a headroom policy.
function applicableMeasurement(constraint, requestKind) {
    if (!constraint || typeof constraint !== 'object') return false;
    const ratio = constraint.used_ratio;
    const models = constraint.applies_to_models;
    const prefixes = Object.hasOwn(constraint, 'applies_to_model_prefixes') ? constraint.applies_to_model_prefixes : [];
    const kinds = Object.hasOwn(constraint, 'applies_to_request_kinds') ? constraint.applies_to_request_kinds : ['text', 'image'];
    return typeof ratio === 'number' && ratio >= 0 && ratio <= 1
        && (models === undefined || models === null || (Array.isArray(models) && !models.length))
        && Array.isArray(prefixes) && !prefixes.length && Array.isArray(kinds) && kinds.includes(requestKind);
}

// Simultaneous rows of ONE subject are constraints, never alternatives: a dated,
// still-live engine block wins; available needs every row fresh and available,
// no absence and one applicable measured window.
function subjectQuotaVerdict(rows, absent, requestKind, nowMs) {
    const blocks = rows.filter((row) => ['exhausted', 'cooldown'].includes(row.availability?.state)
        && !Number.isNaN(liveInstantMs(row.availability.resets_at, nowMs)));
    if (blocks.length) {
        const earliest = blocks.reduce((first, row) => (Date.parse(row.availability.resets_at)
            < Date.parse(first.availability.resets_at) ? row : first));
        return { state: blocks.some((row) => row.availability.state === 'exhausted') ? 'exhausted' : 'cooldown',
            resets_at: earliest.availability.resets_at };
    }
    if (rows.length && !absent && rows.every((row) => row.freshness === 'fresh' && row.availability?.state === 'available')
        && rows.some((row) => Array.isArray(row.constraints)
            && row.constraints.some((constraint) => applicableMeasurement(constraint, requestKind)))) {
        return { state: 'available', resets_at: null };
    }
    return { state: 'unknown', resets_at: null };
}

/** Engine verdict for an honored exact scope, as in claudexor_observations.
 * Catalog/account facts remain useful when quota scope is unknown; neither a
 * model-name substring nor a raw ratio certifies this actor's availability,
 * and a pin answers only to its exact subject (a null one belongs to no pin).
 * Each full subject (harness, credential route, id) is judged alone; observed
 * subjects are alternatives, and an unknown one prevents an all-blocked claim.
 */
export function routeQuotaFact(snapshot, harness, model, profileId = '', nowMs = Date.now(), requestKind = 'text') {
    const scope = snapshot?.availability_scope;
    const unknown = { known: false, exhausted: false, state: 'unknown', resets_at: null };
    if (!scope || !Object.hasOwn(scope, 'model') || scope.model !== (model || null)
        || scope.requestKind !== requestKind || scope.basis !== 'quota_block') return unknown;
    const pin = String(profileId || '');
    const subjectRows = (rows) => (Array.isArray(rows) ? rows : []).filter((row) => row?.subject
        && typeof row.subject === 'object' && row.subject.harness === harness && (!pin || row.subject.subject_id === pin));
    const key = (row) => JSON.stringify([row.subject.harness, row.subject.credential_route ?? null, row.subject.subject_id ?? null]);
    const rows = subjectRows(snapshot?.quota);
    const absent = new Set(subjectRows(snapshot?.quota_absences).map(key));
    const verdicts = [...new Set([...rows.map(key), ...absent])]
        .map((subject) => subjectQuotaVerdict(rows.filter((row) => key(row) === subject), absent.has(subject), requestKind, nowMs));
    if (verdicts.some((verdict) => verdict.state === 'available')) return { known: true, exhausted: false, state: 'available', resets_at: null };
    if (!verdicts.length || verdicts.some((verdict) => verdict.state === 'unknown')) return unknown;
    const earliest = verdicts.reduce((first, verdict) => (Date.parse(verdict.resets_at) < Date.parse(first.resets_at) ? verdict : first));
    return { known: true, exhausted: true, resets_at: earliest.resets_at,
        state: verdicts.some((verdict) => verdict.state === 'exhausted') ? 'exhausted' : 'cooldown' };
}

// One verdict per branch: the short `label` (what a card head has room for),
// the status `tone`, the full `text` (the sentence a tooltip carries) and the
// `reason` a branch names beyond its label (a pin, a model). The four are
// decided together so a reader never re-derives tone or specificity from prose.
const AVAILABLE = ['Available', 'ok'];
const NOT_CHECKED = ['Not checked', 'neutral'];
const UNAVAILABLE = ['Unavailable', 'warn'];
const NO_ACCOUNT = ['No account', 'warn'];
const LIMIT = ['Limit reached', 'warn'];

function verdict([label, tone], text, { specific = false } = {}) {
    return { label, tone, text, reason: specific ? text : '' };
}

// Actors store the pin as `credential_profile_id`, reviewer rows as
// `profile_id`; the verdict reads both, or a pinned reviewer row would be
// judged as an unpinned one.
function routePin(route) {
    return String(route?.credential_profile_id || route?.profile_id || '');
}

function sessionRouteAvailability(row, state, nowMs) {
    const { harness, model } = splitSessionTarget(row?.route?.target_id);
    if (!state?.catalogKnown || !state?.accountsKnown) {
        return verdict(NOT_CHECKED, 'Agent session · live availability not checked');
    }
    const pin = routePin(row?.route);
    const harnessEntry = accountScopedModelCatalog(harnessMap(state.snapshot)[harness], pin);
    if (!harnessEntry) return verdict(UNAVAILABLE, `${harness} · currently unavailable`);
    if (!harnessModelsKnown(harnessEntry, state.catalogKnown)) {
        return verdict(NOT_CHECKED, `${harness} · model availability not checked`);
    }
    const matches = sessionModelMatches(harnessEntry, model);
    if (model && !matches.length) {
        return verdict(UNAVAILABLE, `${harness} · selected model ${model} currently unavailable`, { specific: true });
    }

    const rows = accountRows(state.snapshot).filter((account) => account.harness === harness);
    if (pin) {
        const account = rows.find((candidate) => String(candidate.profile_id || '') === pin);
        if (!account || account.enabled === false
            || String(account?.status?.verification || '') !== 'passed') {
            return verdict(UNAVAILABLE, `${harness} · pinned account ${pin} currently unavailable`, { specific: true });
        }
        if (!state.quotaKnown) return verdict(NOT_CHECKED, `${harness} · pinned account ready; quota not checked`);
        const quota = routeQuotaFact(state.snapshot, harness, model, pin, nowMs);
        if (quota.exhausted) return verdict(LIMIT, `${harness} · pinned account ${pin} limit reached`, { specific: true });
        if (!quota.known) return verdict(NOT_CHECKED, `${harness} · pinned account ready; quota availability not proven`);
        return verdict(AVAILABLE, `${harness} · available now`);
    }

    // The owner's switch refuses, exactly as dispatch reads it; the aggregate doctor
    // `status` describes only the default credential store and dispatch ignores it
    // (`subagent_route_health.route_health`), so it never paints a card Unavailable.
    if (harnessEntry.enabled === false) return verdict(UNAVAILABLE, `${harness} · currently unavailable`);
    const usable = rows.filter((account) => account.enabled !== false
        && String(account?.status?.verification || '') === 'passed');
    if (!usable.length) return verdict(NO_ACCOUNT, `${harness} · no usable account currently`);
    // "Some account carries this model" and "some account is usable" are two
    // questions, and `gpt-5.4` — listed only by an unverified account — used to
    // pass both while no single account could answer yes to BOTH. An
    // account-view catalog stamps each entry with the account that carries it,
    // so the two sets are intersected here; a legacy catalog carries no such
    // provenance (empty `carriers`) and keeps the older, weaker rule.
    const applicable = sessionModelMatches(harnessEntry, model, { snapshot: state.snapshot });
    if (matches.length && !applicable.length) {
        return verdict(NO_ACCOUNT, `${harness} · no usable account currently carries ${model}`, { specific: true });
    }
    if (!state.quotaKnown) return verdict(NOT_CHECKED, `${harness} · account ready; quota not checked`);
    const pool = nextUpAccount(state.snapshot, harness);
    if (pool?.kind === 'none' || pool?.kind === 'api_key_route') {
        return verdict(NO_ACCOUNT, `${harness} · no usable subscription account currently`);
    }
    const carriers = new Set(applicable.map((match) => match.profile));
    const attributed = carriers.size && !carriers.has('');
    // Absences are subjects too: a non-carrier's gap must not hide the carriers' block.
    const carried = (rows) => (rows || []).filter((row) => carriers.has(String(row?.subject?.subject_id || '')));
    const quotaSnapshot = attributed ? { ...state.snapshot, quota: carried(state.snapshot.quota),
        quota_absences: carried(state.snapshot.quota_absences) } : state.snapshot;
    const quota = routeQuotaFact(quotaSnapshot, harness, model, '', nowMs);
    if (quota.exhausted) return verdict(LIMIT, `${harness} · all known accounts reached a limit`, { specific: true });
    if (quota.known) return verdict(AVAILABLE, `${harness} · available now`);
    return verdict(NOT_CHECKED, `${harness} · live availability not checked`);
}

// Catalog evidence is independent of whether quota scope was checked. Reuse
// the account-scoped matcher; an unusable sibling cannot supply the qualifier.
function sessionCatalogQualifier(row, state) {
    if (!state?.catalogKnown || !state?.accountsKnown) return '';
    const { harness, model } = splitSessionTarget(row?.route?.target_id);
    const pin = routePin(row?.route);
    const catalog = accountScopedModelCatalog(harnessMap(state.snapshot)[harness], pin);
    if (!catalog || !harnessModelsKnown(catalog, state.catalogKnown)) return '';
    const matches = sessionModelMatches(catalog, model, { snapshot: state.snapshot, pin });
    return matches.length && matches.every((match) => match.basis === 'claude_1m_base') ? CLAUDE_BASE_QUALIFIER : '';
}

export function sessionRouteVerdict(row, state, nowMs = Date.now()) {
    const live = sessionRouteAvailability(row, state, nowMs);
    const qualifier = sessionCatalogQualifier(row, state);
    return qualifier ? { ...live, text: `${live.text} · ${qualifier}`, qualifier } : live;
}

// The card head carries the AVAILABILITY axis alone: one short word and its
// tone, with the full sentence (plus any model-list gap note) as its title.
// Saved/draft intent is one editor fact, never repeated per card
// (`subagents_settings.js` toolbar). An API model's availability is only ever
// known when a child starts, so its word says exactly that.
export function rowStatus(row, state) {
    if (row.route.kind !== ROUTE_KIND_AGENT_SESSION) {
        // The sentence names the SOURCE the owner picked, not a bare channel:
        // "API model" alone left two rows on different providers reading
        // identically (owner decision 4A).
        const fields = routeModelFields(row.route, state.modelSources, {
            providerProfiles: state.providerProfiles, connections: state.connections,
        });
        return {
            label: 'Checked at start', tone: 'neutral', reason: '',
            text: `${fields.subscription ? 'Subscription model' : `${fields.providerLabel} API model`} · availability is checked when a child starts`,
        };
    }
    const live = sessionRouteVerdict(row, state);
    const { harness } = splitSessionTarget(row.route.target_id);
    const gap = modelsGapNote(accountScopedModelCatalog(harnessMap(state.snapshot)[harness], routePin(row.route)), state.catalogKnown);
    return { ...live, text: [live.text, gap].filter(Boolean).join(' · ') };
}

// A row that will not run says WHY in a visible line under its head when the
// reason names more than the head word already does (a pin, a model, a
// limit): the title is the desktop's tooltip, and a phone, the Telegram mini
// app or a touch screen has no hover. A bare "currently unavailable" repeats
// the head and stays in the title.
export function rowStatusReason(status) {
    return status?.tone === 'warn' || status?.tone === 'error' ? String(status.reason || '') : '';
}

const ROUTE_HINT = 'Choose how this subagent runs: an API model or an agent session.';

// Two rows on one engine are a repeated review when both are marked, or a review
// row minted beside the owner's own delegation row; any other twin is a copy slip.
export function reviewTwinAllowed(first, second) {
    return (first?.review_eligible === true && second?.review_eligible === true)
        || Boolean(first?.minted_from) !== Boolean(second?.minted_from);
}

function executionFor(snapshot, subagentId) {
    const history = snapshot?.subagent_last_delegation;
    const receipt = history?.latest_by_subagent?.[subagentId] || history;
    if (!receipt || typeof receipt !== 'object') return null;
    return String(receipt.selected_subagent_id || '') === String(subagentId || '')
        ? receipt : null;
}

// ONE meta line under the controls, for the CURRENT row only, in priority:
// the row's own error once the owner tried to save THIS row (`_uiAttempted`,
// stamped by the save attempt on the rows that existed then — an entry added
// afterwards is fresh again); the neutral hint while its route is still
// unchosen (a fresh entry is an invitation, not an error); a twin; the
// catalog qualifier of a session model (Claude `[1m]` listed only by its base,
// even when quota availability is Not checked); nothing.
// History and the stored spelling live in the card's Details (`rowTaskRun`).
export function rowMeta(row, state, errors) {
    if (row._uiAttempted && errors.length) return { text: errors[0], tone: 'error' };
    const session = row.route?.kind === ROUTE_KIND_AGENT_SESSION;
    // An empty draft (`openai::` with no model yet) is still an invitation.
    if (!String(row.route?.target_id || '').trim()
        || (!session && !routeModelFields(row.route).model.trim())) return { text: ROUTE_HINT, tone: '' };
    // A twin is SAID even while it blocks nothing: the editor judges twins only
    // once the roster is edited, so one saved earlier never blocks an unrelated Save.
    const items = state.setting?.items || [];
    const twin = sameEngineAs(items, items.indexOf(row), state.processingPreference);
    if (twin >= 0 && !reviewTwinAllowed(items[twin], row)) return { text: `Runs the same engine as Subagent ${twin + 1} — change one of them to tell them apart.`, tone: '' };
    // The independent catalog qualifier stays readable where no hover exists.
    const qualifier = session ? sessionCatalogQualifier(row, state) : '';
    return qualifier ? { text: qualifier, tone: '', qualifier: true } : { text: '', tone: '' };
}

/**
 * The row's last delegated task run (`subagent_last_delegation`), or ''. It is
 * judged against the row's CURRENT settings: a receipt from other settings
 * says so instead of passing for the present route (docs/DESIGN.md §7), and a
 * failed run stays history, never a status.
 */
export function rowTaskRun(row, state) {
    const receipt = executionFor(state.snapshot, row.subagent_id);
    const evidence = describeExecutionEvidence(receipt);
    if (!evidence) return '';
    const session = row.route?.kind === ROUTE_KIND_AGENT_SESSION;
    const identity = receipt?.identity;
    const sameRoute = identity && identity.kind === row.route.kind
        && identity.target_id === row.route.target_id
        && identity.credential_profile_id === String(routePin(row.route) || '')
        && (!session || identity.access === String(row.access || 'full'))
        && identity.effort === String(row.effort || '')
        && identity.processing_preference === String(row.processing_preference || state.processingPreference || '');
    const identityComplete = identity && ['kind', 'target_id', 'credential_profile_id', 'effort',
        'processing_preference', ...(session ? ['access'] : [])].every((key) => typeof identity[key] === 'string');
    const qualifier = identityComplete ? (sameRoute ? '' : 'Earlier settings')
        : identity ? 'Settings not fully reported' : 'Settings not reported';
    return [qualifier, evidence].filter(Boolean).join(' · ');
}

/** What one review by this row actually cost — the median of its measured reviews with their count
 * and range (the words `## Review` prints) — or its non-money equivalent; unknown is never zero. */
export function reviewCostText(row, cost = null) {
    if (routeSupportsAccount(row?.route || {})) return 'uses a session seat and time';
    if (!cost) return 'price appears after saving';
    if (cost.basis === 'local_route') return 'no API cost per review';
    const usd = cost.usd_per_review;
    if (cost.basis !== 'observed_reviews' || typeof usd !== 'number' || !Number.isFinite(usd)) return 'cost unknown';
    const high = Number(cost.high);
    if (usd === 0 && !(high > 0)) return 'no API cost per review';
    const money = (value) => `$${value < 0.01 ? value.toFixed(4) : value.toFixed(2)}`;
    const reviews = Number(cost.reviews) || 0;
    return reviews < 2 ? `≈${money(usd)} per review (1 measured)`
        : `≈${money(usd)} per review (median of ${reviews} measured, ${money(Number(cost.low))}–${money(high)})`;
}

/** What actually ran the last time this row reviewed, with its review record (the card's "Last review"). */
export function lastReviewRunText(entry) {
    const text = describeExecutionEvidence(entry?.effective || !entry?.observed_model ? entry
        : { ...entry, effective: { model: entry.observed_model } });
    if (!text) return '';
    const record = String(entry?.review_record_id || entry?.record_id || '');
    return `${text}${record ? ` · record ${record}` : ''}`;
}
