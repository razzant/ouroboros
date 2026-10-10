// Status and meta projection of one Available-subagents card: pure functions
// of the row, the editor's state and the shared Claudexor snapshot, with no
// DOM, so the editor's markup and its in-place painter read one source and
// node tests pin the words without a browser. Dispatch remains authoritative;
// this module only decides which positive/negative facts a card may honestly
// claim.

import { accountRows, familyLabel, nextUpAccount, quotaConstraintFact } from './claudexor_status_store.js';
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
        providerProfiles: state.providerProfiles,
    });
    const label = sourceIdentityLabel(row?.route, {
        modelSources: state.modelSources,
        providerProfiles: state.providerProfiles,
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

function modelScopeMatches(model, aliases) {
    const routeModel = String(model || '').trim().toLowerCase();
    const scopes = (Array.isArray(aliases) ? aliases : [])
        .map((value) => String(value || '').trim().toLowerCase()).filter(Boolean);
    if (!scopes.length || !routeModel) return true;
    return scopes.some((scope) => scope === routeModel
        || routeModel.includes(scope) || scope.includes(routeModel));
}

/** UI projection of the same positive quota facts dispatch checks again. */
function routeQuotaFact(snapshot, harness, model, profileId = '', nowMs = Date.now()) {
    const pin = String(profileId || '');
    let observed = false;
    let usable = false;
    let spent = false;
    let unknown = false;
    for (const row of snapshot?.quota || []) {
        const subject = row?.subject || {};
        if (String(subject.harness || '') !== String(harness || '')) continue;
        if (pin && String(subject.subject_id || '') !== pin) continue;
        if (String(row?.freshness || '') !== 'fresh') continue;
        observed = true;
        const facts = (row?.constraints || [])
            .filter((constraint) => modelScopeMatches(model, constraint?.applies_to_models))
            .map((constraint) => quotaConstraintFact(constraint, nowMs));
        const spentHere = facts.some((fact) => fact.exhausted);
        if (spentHere) spent = true;
        else if (facts.some((fact) => fact.unknown)) unknown = true;
        else usable = true;
    }
    return { known: usable || (observed && !unknown), exhausted: spent && !usable && !unknown };
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

export function sessionRouteVerdict(row, state, nowMs = Date.now()) {
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
    const available = (text, applicable = matches) => {
        const baseOnly = applicable.length && applicable.every((match) => match.basis === 'claude_1m_base');
        return { ...verdict(AVAILABLE, `${text}${baseOnly ? ` · ${CLAUDE_BASE_QUALIFIER}` : ''}`),
            ...(baseOnly ? { qualifier: CLAUDE_BASE_QUALIFIER } : {}) };
    };
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
        return available(`${harness} · available now`);
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
    if (!attributed && (pool?.kind === 'profile' || pool?.kind === 'native')) {
        return available(`${harness} · compatible account selected; exact model quota checked at start`, applicable);
    }
    const quotaSnapshot = attributed ? { ...state.snapshot, quota: (state.snapshot.quota || []).filter((row) => carriers.has(String(row?.subject?.subject_id || ''))) } : state.snapshot;
    const quota = routeQuotaFact(quotaSnapshot, harness, model, '', nowMs);
    if (quota.exhausted) return verdict(LIMIT, `${harness} · all known accounts reached a limit`, { specific: true });
    if (quota.known) return available(`${harness} · available now`, applicable);
    return verdict(NOT_CHECKED, `${harness} · live availability not checked`);
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
            providerProfiles: state.providerProfiles,
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
// conditional qualifier of an Available session verdict (a Claude `[1m]`
// judged by its listed base); nothing.
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
    // A conditional availability qualifier must stay readable where no hover exists.
    const qualifier = session ? sessionRouteVerdict(row, state).qualifier : '';
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
