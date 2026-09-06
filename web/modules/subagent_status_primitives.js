// Status and meta projection of one Available-subagents card: pure functions
// of the row, the editor's state and the shared Claudexor snapshot, with no
// DOM, so the editor's markup and its in-place painter read one source and
// node tests pin the words without a browser. Dispatch remains authoritative;
// this module only decides which positive/negative facts a card may honestly
// claim.

import { accountRows, nextUpAccount, quotaConstraintFact } from './claudexor_status_store.js';
import {
    ROUTE_KIND_AGENT_SESSION,
    describeExecutionEvidence,
    harnessModelsKnown,
    modelsGapNote,
    splitSessionTarget,
} from './route_editor_primitives.js';

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

function modelIsPresent(harness, model) {
    if (!model) return true;
    return (harness?.models || []).some((entry) =>
        String(entry?.id || entry?.value || entry || '') === String(model));
}

// One verdict per branch: the short `label` (what a card head has room for),
// the status `tone` and the full `text` (the sentence a tooltip carries). The
// three are decided together so a reader never re-derives tone from prose.
const AVAILABLE = ['Available', 'ok'];
const NOT_CHECKED = ['Not checked', 'neutral'];
const UNAVAILABLE = ['Unavailable', 'warn'];
const NO_ACCOUNT = ['No account', 'warn'];
const LIMIT = ['Limit reached', 'warn'];

function verdict([label, tone], text) {
    return { label, tone, text };
}

export function sessionRouteVerdict(row, state, nowMs = Date.now()) {
    const { harness, model } = splitSessionTarget(row?.route?.target_id);
    if (!state?.catalogKnown || !state?.accountsKnown) {
        return verdict(NOT_CHECKED, 'Agent session · live availability not checked');
    }
    const harnessEntry = harnessMap(state.snapshot)[harness];
    if (!harnessEntry) return verdict(UNAVAILABLE, `${harness} · currently unavailable`);
    if (!harnessModelsKnown(harnessEntry, state.catalogKnown)) {
        return verdict(NOT_CHECKED, `${harness} · model availability not checked`);
    }
    if (!modelIsPresent(harnessEntry, model)) {
        return verdict(UNAVAILABLE, `${harness} · selected model ${model} currently unavailable`);
    }

    const rows = accountRows(state.snapshot).filter((account) => account.harness === harness);
    const pin = String(row?.route?.credential_profile_id || '');
    if (pin) {
        const account = rows.find((candidate) => String(candidate.profile_id || '') === pin);
        if (!account || account.enabled === false
            || String(account?.status?.verification || '') !== 'passed') {
            return verdict(UNAVAILABLE, `${harness} · pinned account ${pin} currently unavailable`);
        }
        if (!state.quotaKnown) return verdict(NOT_CHECKED, `${harness} · pinned account ready; quota not checked`);
        const quota = routeQuotaFact(state.snapshot, harness, model, pin, nowMs);
        if (quota.exhausted) return verdict(LIMIT, `${harness} · pinned account ${pin} limit reached`);
        if (!quota.known) return verdict(NOT_CHECKED, `${harness} · pinned account ready; quota availability not proven`);
        return verdict(AVAILABLE, `${harness} · available now`);
    }

    if (harnessEntry.enabled === false
        || (harnessEntry.status && String(harnessEntry.status) !== 'ok')) {
        return verdict(UNAVAILABLE, `${harness} · currently unavailable`);
    }
    if (!rows.some((account) => account.enabled !== false
        && String(account?.status?.verification || '') === 'passed')) {
        return verdict(NO_ACCOUNT, `${harness} · no usable account currently`);
    }
    if (!state.quotaKnown) return verdict(NOT_CHECKED, `${harness} · account ready; quota not checked`);
    const pool = nextUpAccount(state.snapshot, harness);
    if (pool?.kind === 'none' || pool?.kind === 'api_key_route') {
        return verdict(NO_ACCOUNT, `${harness} · no usable subscription account currently`);
    }
    if (pool?.kind === 'profile' || pool?.kind === 'native') {
        return verdict(AVAILABLE, `${harness} · compatible account selected; exact model quota checked at start`);
    }
    const quota = routeQuotaFact(state.snapshot, harness, model, '', nowMs);
    if (quota.exhausted) return verdict(LIMIT, `${harness} · all known accounts reached a limit`);
    if (quota.known) return verdict(AVAILABLE, `${harness} · available now`);
    return verdict(NOT_CHECKED, `${harness} · live availability not checked`);
}

// The card head has room for two short words — the intent axis and the
// availability axis — with one dot whose tone is the worse of the two; the
// full sentence of each axis (plus any model-list gap note) rides the title.
// Both axes are structural: the intent word and tone follow the editor's
// dirty flag and its baseline (what was loaded: saved bytes, or a generated
// draft in the wizard), never a parse of prose; an API model's availability
// is only ever known when a child starts, so its second word says exactly
// that.
const INTENT = {
    draft: { word: 'Draft', tone: 'neutral', text: 'Draft intent' },
    generated: { word: 'Generated', tone: 'neutral', text: 'Generated draft' },
    saved: { word: 'Saved', tone: 'ok', text: 'Saved intent' },
};
const TONE_RANK = { ok: 0, neutral: 1, warn: 2, error: 3 };
const worseTone = (a, b) => (TONE_RANK[b] > TONE_RANK[a] ? b : a);

function intentAxis(state) {
    return INTENT[state.dirty ? 'draft' : state.baseline] || INTENT.saved;
}

export function rowStatus(row, state) {
    const intent = intentAxis(state);
    if (row.route.kind !== ROUTE_KIND_AGENT_SESSION) {
        return {
            label: `${intent.word} · Checked at start`,
            tone: worseTone(intent.tone, 'neutral'),
            text: `${intent.text} · API model · availability is checked when a child starts`,
        };
    }
    const live = sessionRouteVerdict(row, state);
    const { harness } = splitSessionTarget(row.route.target_id);
    const gap = modelsGapNote(harnessMap(state.snapshot)[harness], state.catalogKnown);
    return {
        label: `${intent.word} · ${live.label}`,
        tone: worseTone(intent.tone, live.tone),
        text: [`${intent.text} · ${live.text}`, gap].filter(Boolean).join(' · '),
    };
}

const ROUTE_HINT = 'Choose how this subagent runs: an API model or an agent session.';

function executionFor(snapshot, subagentId) {
    const receipt = snapshot?.subagent_last_delegation;
    if (!receipt || typeof receipt !== 'object') return null;
    return String(receipt.selected_subagent_id || '') === String(subagentId || '')
        ? receipt : null;
}

// ONE meta line under the controls, in priority: the row's own error once the
// owner tried to save THIS row (`_uiAttempted`, stamped by the save attempt on
// the rows that existed then — an entry added afterwards is fresh again); the
// neutral hint while its route is still unchosen (a fresh entry is an
// invitation, not an error); the last actual run; nothing.
export function rowMeta(row, state, errors) {
    if (row._uiAttempted && errors.length) return { text: errors[0], tone: 'error' };
    if (!String(row.route?.target_id || '').trim()) return { text: ROUTE_HINT, tone: '' };
    const evidence = describeExecutionEvidence(executionFor(state.snapshot, row.subagent_id));
    return { text: evidence ? `Last actual run: ${evidence}` : '', tone: '' };
}
