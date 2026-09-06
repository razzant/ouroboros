// Review lanes UI (phase 6.2/6.3, revised per owner finding #6) — the rows in
// Agents → Review lanes over the ONE structured setting
// (OUROBOROS_REVIEWER_SLOTS; the settings key keeps its name, the section does
// not — D-10 moved these rows out of the Models tab and renamed them).
//
// Shape rules are the owner's:
//  * ONE flat picker per row (decision 1=B replaced the earlier two-level
//    source+route pair): the roster rows from Available subagents lead as
//    references, then the inline channels — "API model" plus one entry per
//    login-capable harness — never the full flat model catalog (hundreds of
//    options made the list unusable, finding #6). A referenced row's route/
//    model/effort/account are shown as READ-ONLY derived facts (the roster
//    stays their SSOT); the stored forms are mutually exclusive:
//    {slot_id, route, effort} XOR {slot_id, subagent_id, effort}.
//  * On the API route the model id is a FREE-TEXT input with a datalist of
//    catalog suggestions — the same catalog-assisted entry the model cards
//    use. On a harness route the MODEL is a dropdown fed by Claudexor
//    discovery. No invalid combinations can be composed.
//  * The provider shown for a delegated row is the HARNESS NAME (codex,
//    claude, cursor, …) — never "Claudexor", and never a `provider::model`
//    string syntax.
//  * Effort is the EXISTING per-model effort mechanism, one dropdown per row.
// Capability badges DISPLAY facts; they configure nothing.
//
// Pure helpers live at the top and are node-tested without a DOM.

import { apiFetch } from './api_client.js';
import { bindStatusSurface, boundedStatusRefresh, claudexorStatus } from './claudexor_status_store.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import { formatRelativeAge, revealNewRow } from './ui_helpers.js';
import * as routeEditor from './route_editor_primitives.js';
import {
    availableSubagentsLoadValue,
    parseAvailableSubagentsSetting,
} from './subagents_settings.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const ROUTE_KIND_API = 'api_chat';
export const ROUTE_KIND_SESSION = routeEditor.ROUTE_KIND_AGENT_SESSION;

// Select-value prefix marking a configured-subagent reference in the one
// flat reviewer picker (decision 1=B); everything else is a route choice.
export const SUBAGENT_CHOICE_PREFIX = 'subagent:';
// The route select's single API entry. The target model id never lives in
// the select — display always matches the stored target (finding #6c: a
// fresh row used to DISPLAY the first catalog model while storing '').
export const API_ROUTE_CHOICE = routeEditor.API_ROUTE_CHOICE;

export const EFFORT_CHOICES = routeEditor.EFFORT_CHOICES;

// ---------------------------------------------------------------------------
// Pure helpers.
// ---------------------------------------------------------------------------

export function mintSlotId(prefix, takenIds) {
    return routeEditor.mintStableId(prefix, takenIds);
}

export function encodeRouteChoice(row) {
    return routeEditor.encodeRouteChoice(row);
}

export function decodeRouteChoice(value) {
    return routeEditor.decodeRouteChoice(value, { apiKind: ROUTE_KIND_API });
}

// Claudexor's own reviewer-panel spelling: harness[=model]. Never '::'.
export function composeSessionTarget(harness, model) {
    return routeEditor.composeSessionTarget(harness, model);
}

export function splitSessionTarget(target) {
    return routeEditor.splitSessionTarget(target);
}

// `catalogKnown` / `accountsKnown` = the matching facet of the status payload
// was actually READ (claudexor_status_store.facetReadState). Only then does the
// absence of a harness/model/account mean anything about that harness/model/
// account. With the daemon stopped, unreachable, or that one read refused, the
// backend answers `harnesses: []` by construction, and this UI used to decorate
// EVERY saved row with "(not in discovery)" — an accusation nobody had earned,
// and the exact screen the owner reported (2026-08-08). The saved value keeps
// its option either way (that guard is what stops a Save from erasing a pin);
// only the LABEL's claim follows the facet: "(not in discovery)" is a VERDICT —
// it says we looked and did not find it — and is licensed ONLY by a facet that
// was read. An unread facet says "(not checked)": we did not look, and the
// owner's saved value is almost certainly still there. The two facets are
// INDEPENDENT: a failed account read must not silence the catalog's own honest
// verdict.
export function routeChoiceGroups({ harnesses = [], currentChoice = '', catalogKnown = true, apiLabel } = {}) {
    return routeEditor.routeChoiceGroups({ harnesses, currentChoice, catalogKnown, apiLabel });
}

export function indexProfilesByHarness(payload) {
    return routeEditor.indexProfilesByHarness(payload);
}

// One index entry, whichever spelling it arrived in: the index emits
// `{id, enabled}` objects; older call sites and tests still hand plain id
// strings, which read as enabled (the same fail-open rule as the index).
function profileEntry(entry) {
    return routeEditor.profileEntry(entry);
}

export function buildReviewerSlotsSetting(state) {
    const rowOut = (row) => {
        // The two stored forms are mutually exclusive: a configured-subagent
        // reference never duplicates route knobs (the roster row is their
        // SSOT), and an empty explicit effort is OMITTED so the roster row's
        // own effort keeps deciding.
        if (row.subagent_id) {
            const out = {
                slot_id: String(row.slot_id || ''),
                subagent_id: String(row.subagent_id),
            };
            if (row.effort) out.effort = String(row.effort);
            return out;
        }
        const out = {
            slot_id: String(row.slot_id || ''),
            route: routeEditor.serializeRouteSpec(row.route, {
                apiKind: ROUTE_KIND_API,
                credentialField: 'profile_id',
            }),
        };
        if (row.effort) out.effort = String(row.effort);
        return out;
    };
    const advisory = state.advisory || {};
    let advisoryOut;
    if (advisory.subagent_id) {
        advisoryOut = {
            enabled: advisory.enabled !== false,
            subagent_id: String(advisory.subagent_id),
        };
        if (advisory.effort) advisoryOut.effort = String(advisory.effort);
    } else {
        // The advisory row is on the SHARED closed route vocabulary now:
        // api_chat (routed catalog model, bounded native inspection episode)
        // or agent_session. The retired legacy kind 'api' (Claude-SDK
        // spellings) still parses server-side but is never WRITTEN again.
        advisoryOut = {
            enabled: advisory.enabled !== false,
            route: { kind: advisory.route?.kind === ROUTE_KIND_SESSION ? ROUTE_KIND_SESSION : ROUTE_KIND_API,
                     target_id: String(advisory.route?.target_id || '') },
            effort: advisory.route?.kind === ROUTE_KIND_SESSION
                ? String(advisory.effort || '') : (advisory.effort || 'low'),
        };
        if (advisoryOut.route.kind === ROUTE_KIND_SESSION && advisory.route?.profile_id) {
            advisoryOut.route.profile_id = String(advisory.route.profile_id);
        }
    }
    const setting = {
        triad: (state.triad || []).map(rowOut),
        scope: (state.scope || []).map(rowOut),
        advisory: advisoryOut,
    };
    // The deep self-review singleton is OPTIONAL server-side (absent = the
    // packed row synthesized from OUROBOROS_MODEL_DEEP_SELF_REVIEW). An
    // UNTOUCHED synthesized or empty placeholder (`materialized: false`) is
    // OMITTED: the runtime then synthesizes the identical row, and an
    // unrelated save never writes the key's value into the setting behind the
    // owner's back. Editing the row (or loading a SAVED one) materializes it.
    // Same two stored forms as every row, minus slot_id (fixed identity) and
    // minus `enabled` (no standing gate to switch off).
    if (state.deepReview && state.deepReview.materialized !== false) {
        setting.deep_review = rowOut({ ...state.deepReview, slot_id: '' });
        delete setting.deep_review.slot_id;
    }
    return JSON.stringify(setting);
}

export function deepReviewMetaNotes(row) {
    // The deep row's two owner-facing facts beside its badge: an untouched
    // synthesized row is shown but not written, and a blanked model box is a
    // typed save refusal (owner fork 3 = A) — said HERE, before the 400.
    const notes = [];
    if (row?.synthesizedFrom && row.materialized === false) {
        notes.push(`Not saved as a row yet — shown from ${row.synthesizedFrom}; edit it to store it as the deep_review row (an untouched row is not written)`);
    }
    if (row?.materialized !== false && !row?.subagent_id && row?.route?.kind !== ROUTE_KIND_SESSION
        && !String(row?.route?.target_id || '').trim()) {
        notes.push('Model id required — an empty model id is refused at save; enter one, or pick a configured subagent or an agent');
    }
    return notes;
}

export function deepReviewDeliveryNote(row, { roster = [], rosterKnown = true, harnesses = {}, catalogKnown = true } = {}) {
    // The deep-review row's ONE difference from the advisory, said where the
    // owner picks: an API MODEL here is the packed review (one large-context
    // call carrying the Atlas + memory), not an inspection episode; only a
    // configured subagent on an API model runs the native episode.
    if (row?.subagent_id) {
        const ref = (roster || []).find((item) => String(item.subagent_id || '') === String(row.subagent_id || ''));
        if (!ref) {
            return rosterKnown
                ? 'Delivery follows the roster row — none exists with this ID, so the review will refuse rather than reroute'
                : 'Delivery follows the roster row — the roster could not be read, so it is not shown';
        }
        return ref.route?.kind === ROUTE_KIND_SESSION
            ? 'Agent session — reads the repository with its own tools (reads not host-observed); the memory whitelist reaches it inline byte-exact'
            : 'Native inspection episode — reads the repository with host read-only tools (reads host-observed); the memory whitelist reaches it inline byte-exact';
    }
    if (row?.route?.kind === ROUTE_KIND_SESSION) {
        return `${capabilityBadge(row, harnesses, { catalogKnown })} — reads not host-observed`;
    }
    return 'One packed review — the repository Atlas plus the full memory whitelist in a single large-context call (the advisory’s API model runs an inspection episode instead)';
}

// ---------------------------------------------------------------------------
// Configured-subagent references (decision 5A). The roster list reuses the
// Available-subagents data source: settings.js hands the loaded settings
// document to adoptSubagentRoster below, parsed by the SAME parser that
// section uses — never a second schema.
// ---------------------------------------------------------------------------

export function advisoryReferenceTransition(advisory, subagentId) {
    // Crossing from an inline branch drops the inline effort ('' = the roster
    // row's own effort decides — the api default 'low' must not silently
    // override the reference); a reference-to-reference switch keeps an
    // explicit owner override (sol delta-review finding).
    return {
        subagent_id: String(subagentId || ''),
        effort: advisory?.subagent_id ? String(advisory.effort || '') : '',
    };
}

export function encodeReviewerChoice(row) {
    return row?.subagent_id
        ? SUBAGENT_CHOICE_PREFIX + String(row.subagent_id)
        : encodeRouteChoice(row);
}

export function reviewerChoiceGroups({
    roster = [], rosterKnown = true, row = {}, harnesses = [],
    catalogKnown = true, apiLabel,
} = {}) {
    // Decision 1=B: ONE flat picker — the Available-subagents references lead
    // (facts-first labels, decision 2=A), then the inline channels. A saved
    // reference that fell out of the roster keeps its option (the same
    // survive-the-save rule the roster select carried); an inline row passes
    // its own choice down so an undiscovered harness stays displayable too.
    const groups = [];
    const savedId = String(row?.subagent_id || '');
    const rosterOptions = subagentOptionsFor(roster, savedId, { rosterKnown })
        .map((option) => ({ ...option, value: SUBAGENT_CHOICE_PREFIX + option.value }));
    if (rosterOptions.length) {
        groups.push({ label: 'Available subagents', options: rosterOptions });
    }
    const currentChoice = savedId ? '' : encodeRouteChoice(row);
    groups.push(...routeChoiceGroups({ harnesses, currentChoice, catalogKnown, apiLabel }));
    return groups;
}

export function subagentOptionLabel(row) {
    const route = row?.route || {};
    const parts = [`#${String(row?.subagent_id || '')}`];
    if (route.kind === ROUTE_KIND_SESSION) {
        const split = splitSessionTarget(route.target_id);
        parts.push(split.harness || 'agent session');
        if (split.model) parts.push(split.model);
    } else {
        parts.push('API');
        if (route.target_id) parts.push(route.target_id);
    }
    if (row?.effort) parts.push(row.effort);
    // Free text is a caption, never identity: one line, bounded, with the
    // characters that could visually reorder or break the facts stripped
    // (bidi controls, newlines).
    const use = String(row?.recommended_use || '')
        .replace(/[\u202A-\u202E\u2066-\u2069\u200E\u200F\u061C]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    // Code POINTS, not UTF-16 units: a slice must never split a surrogate pair.
    const points = Array.from(use);
    const hint = points.length > 48 ? `${points.slice(0, 45).join('')}…` : use;
    return parts.join(' · ') + (hint ? ` — ${hint}` : '');
}

export function subagentOptionsFor(roster, savedId, { rosterKnown = true } = {}) {
    // Same survive-the-save rule as profileOptionsFor: the select's value must
    // EXIST as an option, or the browser silently redraws the row as the first
    // roster entry and the next Save really rewires the reviewer. And the
    // absence claim follows provenance: only a roster that was actually READ
    // may say a saved reference is not in it.
    // Label contract (owner decision 2=A): DERIVED FACTS lead — channel first,
    // then target and effort — so the delivery is visible BEFORE selection and
    // free-text intent can never disguise it; the description is a trimmed,
    // sanitized single-line caption after the facts.
    const options = (roster || []).map((row) => ({
        value: String(row.subagent_id || ''),
        label: subagentOptionLabel(row),
    }));
    const saved = String(savedId || '');
    if (saved && !options.some((option) => option.value === saved)) {
        options.push({
            value: saved,
            label: `${saved} (${rosterKnown ? 'not in the roster' : 'not checked'})`,
        });
    }
    return options;
}

export function describeSubagentReference(subagentId, roster, { rosterKnown = true } = {}) {
    // The DERIVED facts, disclosed read-only (never editable knobs): the
    // roster row is the SSOT for a referenced reviewer's route/model/effort/
    // account, so this line only reports what that row says.
    const row = (roster || []).find(
        (item) => String(item.subagent_id || '') === String(subagentId || ''));
    if (!row) {
        return rosterKnown
            ? 'Configured subagent — no roster row with this ID exists; the saved reference is kept and the review will refuse rather than reroute'
            : 'Configured subagent — the roster could not be read, so its route is not shown; the saved reference is unchanged';
    }
    const route = row.route || {};
    const parts = [];
    if (route.kind === ROUTE_KIND_SESSION) {
        const split = splitSessionTarget(route.target_id);
        parts.push(split.harness ? `${split.harness} session` : 'agent session');
        if (split.model) parts.push(split.model);
        if (route.credential_profile_id) parts.push(`account ${route.credential_profile_id}`);
    } else {
        parts.push(route.target_id ? `API model ${route.target_id}` : 'API model (unset)');
    }
    if (row.effort) parts.push(`effort ${row.effort}`);
    return `Runs as ${parts.join(' · ')} — from its roster row under Available subagents`;
}

export function describeLastExecution(entry) {
    if (!entry || typeof entry !== 'object') return '';
    const effective = entry.effective || {};
    const parts = [];
    // APPLIED facts only: a session run whose telemetry predates the engine
    // receipt shows honest absence, never the requested value as applied.
    // One quiet line (owner feedback): the route is spelled only when it says
    // more than the row's own delivery badge (which agent really ran),
    // and the timestamp is humanized — the raw route + ISO instant stay
    // recoverable in the line's tooltip (lastRunMetaTitle).
    const route = String(effective.route || '');
    if (route.startsWith('agent_session')) {
        const harness = route.slice('agent_session'.length).replace(/^:/, '');
        parts.push(harness ? `${harness} session` : 'agent session');
        parts.push(effective.model || 'model not disclosed');
    } else if (effective.model) {
        parts.push(effective.model);
    }
    if (effective.profile_id) parts.push(`account ${effective.profile_id}`);
    if (effective.access) parts.push(`access ${effective.access}`);
    // No applied effort is rendered: none exists upstream, so the key is not emitted.
    if (effective.verdict_method && effective.verdict_method !== 'structured'
        && effective.verdict_method !== 'strict_parse') {
        parts.push(`verdict via ${effective.verdict_method.replace(/_/g, ' ')}`);
    }
    const deltas = Array.isArray(entry.capability_delta) ? entry.capability_delta.length : 0;
    if (deltas) parts.push(`${deltas} capability delta${deltas === 1 ? '' : 's'} disclosed`);
    const when = formatRelativeAge(Date.parse(entry.ts || ''), 'just now');
    if (when) parts.push(when);
    return parts.join(' · ');
}

function lastRunMetaTitle(entry) {
    // The tooltip keeps the raw facts the visible line compresses away.
    const route = String(entry?.effective?.route || 'api_chat');
    const ts = String(entry?.ts || '');
    return `UI projection of capability_delta (D22) — ran as ${route}${ts ? ` at ${ts}` : ''}`;
}

export function harnessModelsKnown(harness, catalogKnown = true) {
    // Discovery is TWO reads, not one. The catalog read can land — daemon
    // globally `running`, `catalogKnown` true — while THIS harness's model
    // list refuses: the endpoint answers `models: []` with a typed
    // `models_error` for exactly that harness (claudexor_accounts.py, the
    // per-harness `harness_models` probe). Reading the empty list as discovery
    // then labelled a saved model "(not in discovery)", which claims a
    // successful discovery proved its absence — while no discovery happened.
    return routeEditor.harnessModelsKnown(harness, catalogKnown);
}

export function modelsGapNote(harness, catalogKnown = true) {
    // The typed gap, SAID rather than left as a silently short list. Only when
    // the catalog itself was read: with the catalog unread the section note
    // above already explains everything, and this would be a second sentence
    // for the same silence.
    return routeEditor.modelsGapNote(harness, catalogKnown);
}

export function sessionModelOptions(harness, currentModel, { catalogKnown = true } = {}) {
    // The ONE model-options fragment for a session row's model select (triad,
    // scope, advisory, and the Subagents section import it too). "Engine
    // default model" is the empty tail; a SAVED model discovery no longer
    // lists keeps a "(not in discovery)" option, or the browser silently
    // redraws the select as the first entry and the next Save erases the pin
    // (same rule as profileOptionsFor / routeChoiceGroups).
    return routeEditor.sessionModelOptions(harness, currentModel, { catalogKnown });
}

export function profileOptionsFor(profiles, savedPin, { accountsKnown = true } = {}) {
    // Mirrors the model list's own rule: a SAVED pin the daemon no longer discovers
    // (account signed out, daemon down, profile renamed) matched no option, so the
    // select fell back to its first entry and redrew the row as "automatic rotation".
    // The pin only LOOKED gone — until the owner saved the panel, which then really
    // did delete it, silently widening which account the reviewer may spend.
    // A DISABLED account stays selectable, labeled "(disabled)": the engine's
    // typed refusal is the authority on whether a pinned run may use it
    // (D-U6), so the label is honesty, not a gate — hiding or greying the
    // option would be a second, client-side gate the design rejected.
    return routeEditor.profileOptionsFor(profiles, savedPin, { accountsKnown });
}

export function pinnedAccountWarning({ triad = [], scope = [], advisory = null, deepReview = null,
                                       profilesByHarness = {}, accountsKnown = false } = {}) {
    // A removed (or signed-out) account must not silently reroute the row that
    // pinned it. `profileOptionsFor` already keeps such a pin selectable, so
    // the row itself never changes under the owner — this is the ONE actionable
    // sentence that says why it now reads "(not in discovery)".
    //
    // Only an ACTUALLY READ account list licenses the claim (BIBLE P1): with
    // the accounts facet unread every pin would look missing, and the tab's
    // service banner is already saying nobody could be asked.
    if (!accountsKnown) return '';
    const missing = [];
    const rows = [...triad, ...scope, ...(advisory ? [{ ...advisory, slot_id: 'advisory' }] : []),
                  ...(deepReview ? [{ ...deepReview, slot_id: 'deep_review' }] : [])];
    for (const row of rows) {
        // A configured-subagent reference carries no pin of its own — the
        // roster row is that route's SSOT, checked on its own surface.
        if (row?.subagent_id) continue;
        const route = row?.route || {};
        if (route.kind !== ROUTE_KIND_SESSION) continue;
        const pin = String(route.profile_id || '');
        if (!pin) continue;
        const harness = splitSessionTarget(route.target_id).harness;
        if ((profilesByHarness[harness] || []).some((entry) => profileEntry(entry).id === pin)) continue;
        const label = `${harness} · ${pin}`;
        if (!missing.includes(label)) missing.push(label);
    }
    if (!missing.length) return '';
    return `${missing.length === 1 ? 'A review row is' : `${missing.length} review rows are`} `
        + `pinned to an account the agent service no longer lists (${missing.join(', ')}). `
        + 'Those rows are shown as-is and will refuse rather than reroute — pick another '
        + 'account or automatic rotation below, or sign that account back in under Accounts.';
}

export function capabilityBadge(row, harnessesById, { catalogKnown = true } = {}) {
    // DISPLAY-only facts: never a control (6.2).
    if (row.route.kind === ROUTE_KIND_SESSION) {
        // "route not discovered" is a claim about the ROUTE; with no successful
        // discovery it is a claim about the read. Say nothing rather than
        // something false — the one section note above already says why.
        if (!catalogKnown) return 'agent session — retrieves context with its own tools';
        const harness = harnessesById?.[splitSessionTarget(row.route.target_id).harness];
        const status = harness ? (harness.status || 'unknown') : 'not discovered';
        return `agent session — retrieves context with its own tools · route ${status}`;
    }
    return 'API delivery';
}

export function advisoryRouteTransition(prev, decoded, memory = {}) {
    // Route-kind switching must not WIPE a stored target (finding #7c): the
    // old handler wrote target_id:'' on every change, so flipping to a
    // session and back — or just touching the select — silently discarded the
    // saved advisory target on the next Save. Each kind remembers the last
    // route it held; a kind the user returns to is restored, and the stored
    // route only changes when the user actually picks something else.
    const current = (prev && typeof prev === 'object' && prev.kind)
        ? prev : { kind: ROUTE_KIND_API, target_id: '' };
    const next = { ...memory };
    if (decoded.kind === ROUTE_KIND_SESSION) {
        const prevHarness = current.kind === ROUTE_KIND_SESSION
            ? splitSessionTarget(current.target_id).harness : '';
        if (prevHarness === decoded.harness) return { route: current, memory: next };
        if (current.kind === ROUTE_KIND_SESSION) next.session = { ...current };
        else next.api = { ...current };
        const stash = next.session;
        const route = (stash && splitSessionTarget(stash.target_id).harness === decoded.harness)
            ? { ...stash }
            : { kind: ROUTE_KIND_SESSION, target_id: decoded.harness };
        return { route, memory: next };
    }
    if (current.kind !== ROUTE_KIND_SESSION) return { route: current, memory: next };
    next.session = { ...current };
    return {
        route: next.api ? { ...next.api } : { kind: ROUTE_KIND_API, target_id: '' },
        memory: next,
    };
}

// ---------------------------------------------------------------------------
// DOM section (Agents → Review lanes). State is module-local; collect is synchronous.
// ---------------------------------------------------------------------------

const state = {
    loaded: false,
    configError: '',
    loadError: '',
    source: '',
    triad: [],
    scope: [],
    advisory: { enabled: true, route: { kind: ROUTE_KIND_API, target_id: '' }, effort: 'low', subagent_id: '' },
    // The deep self-review singleton; `synthesizedFrom` names the legacy model
    // key when the server showed a row that is not saved yet (Save stores it).
    deepReview: { route: { kind: ROUTE_KIND_API, target_id: '' }, effort: '', subagent_id: '', synthesizedFrom: '', materialized: false },
    limits: { triad: 10, scope: 4, advisory: 1, deep_review: 1 },
    lastExecutions: {},
    catalogModels: [],
    harnesses: [],
    profilesByHarness: {},
    // The configured-subagent roster (OUROBOROS_SUBAGENTS items) the reference
    // selects offer. Display-only here; the Available-subagents section owns
    // editing. rosterKnown follows the same provenance rule as the facets: an
    // unreadable roster licenses no absence claim about a saved reference.
    roster: [],
    rosterKnown: false,
    // PER-FACET provenance: the route/model lists come from the CATALOG facet,
    // the account pins from the ACCOUNTS facet, and one can be authoritative
    // while the other was never read. Only an `ok` facet may license a row-level
    // "(not in discovery)" label.
    catalogKnown: false,
    accountsKnown: false,
    store: claudexorStatus,
    disposers: [],
    onChange: () => {},
};

// Per-kind memory for the singleton route selects (see advisoryRouteTransition,
// which is singleton-generic despite its name): seeded from the LOADED setting
// so a kind round-trip restores the saved target.
const advisoryRouteMemory = { api: null, session: null };
const deepReviewRouteMemory = { api: null, session: null };

// The two single-row categories, rendered and bound by ONE renderer/binder
// (`singletonHtml` / `bindSingletonEvents`) parameterized by this table: the
// advisory (with its Enabled switch and the api-route `low` default) and the
// deep self-review row (no switch; '' = the Behavior-tab deep effort).
const SINGLETONS = {
    advisory: {
        attr: 'advisory', stateKey: 'advisory', lastKey: 'advisory_slot_1', ariaName: 'Advisory',
        rowId: 'reviewer-advisory-row', enabledToggle: true, apiEffortDefault: 'low', apiEffortLabel: 'low',
        apiLabel: 'API model (inspection episode)', apiPlaceholder: 'provider/model-id — empty = default',
        memory: advisoryRouteMemory, badgeOnReference: false,
        badge: (row) => capabilityBadge({ route: row.route || {} }, harnessesById(), { catalogKnown: state.catalogKnown }),
        extraMeta: () => [],
    },
    deepReview: {
        attr: 'deep-review', stateKey: 'deepReview', lastKey: 'deep_review_slot_1', ariaName: 'Deep self-review',
        rowId: 'reviewer-deep-review-row', enabledToggle: false, apiEffortDefault: '', apiEffortLabel: 'deep self-review effort',
        apiLabel: 'API model (one packed review)', apiPlaceholder: 'provider/model-id',
        memory: deepReviewRouteMemory, badgeOnReference: true, materializeOnEdit: true,
        badge: (row) => deepReviewDeliveryNote(row, {
            roster: state.roster, rosterKnown: state.rosterKnown, harnesses: harnessesById(), catalogKnown: state.catalogKnown,
        }),
        extraMeta: deepReviewMetaNotes,
    },
};

// The multi-row categories the editor paints. ONE table drives rendering,
// lookup, adding and removal — a category is a table entry, never another
// `group === 'scope' ? … : …` ternary (three of those used to encode it).
export const CATEGORIES = {
    triad: {
        stateKey: 'triad', limitKey: 'triad', idPrefix: 'triad',
        rowsId: 'reviewer-triad-rows', limitId: 'reviewer-triad-limit', addId: 'btn-add-triad-slot',
        surfaceDefault: 'review effort', empty: 'No triad slots configured.',
    },
    scope: {
        stateKey: 'scope', limitKey: 'scope', idPrefix: 'scope',
        rowsId: 'reviewer-scope-rows', limitId: 'reviewer-scope-limit', addId: 'btn-add-scope-slot',
        surfaceDefault: 'scope review effort', empty: 'No scope slots configured.',
    },
};

function categoryRows(group) {
    return state[CATEGORIES[group].stateKey];
}

export function renderReviewerSlotsSection() {
    return `
        <div class="form-section" id="reviewer-slots-section">
            <h3>Review lanes</h3>
            <div class="settings-section-copy">
                Who reviews each commit: the triad rows, the scope rows, and one optional advisory
                pre-reviewer. Each row picks its reviewer from one list — a configured subagent
                from Available subagents above, an API model, or an agent on your subscription —
                plus its own reasoning effort.
            </div>
            <div class="settings-inline-note">
                Saved lane changes apply from the next task: a task that is already running
                keeps the reviewer configuration it started with.
            </div>
            <div class="settings-inline-note">
                Rows routed to a subscription never fall back to API spend: if every eligible window
                is exhausted, the review waits for capacity. Commit, plan, scope, advisory, skill
                review and task acceptance all follow their configured rows — task acceptance runs
                the triad rows on their own delivery (API packet, configured-subagent inspection
                episode, or agent session), so an all-subscription triad puts every substantive
                task's acceptance panel on the subscription as well.
            </div>
            <div id="reviewer-slots-error" class="ui-status" data-tone="error" hidden></div>
            <div id="reviewer-slots-pins" class="settings-inline-status" data-tone="warn" hidden></div>
            <datalist id="reviewer-api-model-catalog"></datalist>
            <div class="reviewer-slots-group">
                <div class="reviewer-slots-head">
                    <h4 class="reviewer-slots-heading">Triad slots <span class="muted" id="reviewer-triad-limit" title="The commit gate's real ceiling"></span></h4>
                    <button type="button" class="btn btn-default" id="btn-add-triad-slot">Add triad slot</button>
                </div>
                <div id="reviewer-triad-rows" class="reviewer-slot-rows"></div>
            </div>
            <div class="reviewer-slots-group">
                <div class="reviewer-slots-head">
                    <h4 class="reviewer-slots-heading">Scope slots <span class="muted" id="reviewer-scope-limit" title="The scope pool's real width"></span></h4>
                    <button type="button" class="btn btn-default" id="btn-add-scope-slot">Add scope slot</button>
                </div>
                <div class="settings-inline-note">An agent row reads the repository with its own read-only tools instead of
                    being handed one assembled pack. Its verdict is authoritative once that agent's context window is
                    confirmed at 200K or more; Ouroboros does not attest which files the agent opened.</div>
                <div id="reviewer-scope-rows" class="reviewer-slot-rows"></div>
            </div>
            <div class="reviewer-slots-group">
                <h4 class="reviewer-slots-heading">Advisory pre-reviewer</h4>
                <div id="reviewer-advisory-row" class="reviewer-slot-rows"></div>
            </div>
            <div class="settings-inline-note">
                Disabling the advisory is a standing decision with a constitutional consequence:
                every reviewed commit then records an <strong>audited bypass</strong> instead of an
                advisory verdict. Nothing is skipped silently.
            </div>
            <div class="reviewer-slots-group">
                <h4 class="reviewer-slots-heading">Deep self-review</h4>
                <div class="settings-inline-note">
                    Who runs <code>/review</code>, the whole-system review against BIBLE.md. An API model here
                    receives ONE packed review — the repository Atlas plus the full memory whitelist in a single
                    large-context call (unlike the advisory, whose API model runs an inspection episode). A
                    configured subagent on an API model reads the repository itself in a native
                    inspection episode with host-observed reads; an agent on your subscription reads it in its
                    own session (reads not host-observed). Either way the memory whitelist reaches the reviewer
                    inline byte-exact — memory is never receipt-checked. The row's effort outranks the Behavior-tab deep
                    self-review effort; every report starts with a provenance header naming the delivery.
                </div>
                <div id="reviewer-deep-review-row" class="reviewer-slot-rows"></div>
            </div>
        </div>
    `;
}

function harnessesById() {
    const map = {};
    for (const h of state.harnesses) map[h.id] = h;
    return map;
}

function selectHtml(attrs, groups, selected) {
    return routeEditor.selectHtml(attrs, groups, selected);
}

function effortSelectHtml(attrs, selected, surfaceDefault) {
    // Compact closed state (owner feedback on field proportions): the wordy
    // "Default (scope review effort)" label made this select as wide as the
    // model field. Which setting the default follows moves to the tooltip.
    return routeEditor.effortSelectHtml(attrs, selected, surfaceDefault);
}

export function reviewerRouteIdentityMarkup(route, harnesses = {}, {
    catalogKnown = false,
} = {}) {
    if (route?.kind !== ROUTE_KIND_SESSION) {
        return harnessIdentityMarkup('api', {
            channel: 'api',
            className: 'reviewer-slot-route-identity',
        });
    }
    const split = splitSessionTarget(route.target_id);
    const harness = harnesses?.[split.harness];
    return harnessIdentityMarkup(split.harness, {
        label: catalogKnown ? String(harness?.display_name || '') : '',
        className: 'reviewer-slot-route-identity',
    });
}

function reviewerPickerHtml(attrs, row, { apiLabel } = {}) {
    // The one flat reviewer picker (decision 1=B): roster references first,
    // then the inline channels.
    const groups = reviewerChoiceGroups({
        roster: state.roster,
        rosterKnown: state.rosterKnown,
        row,
        harnesses: state.harnesses,
        catalogKnown: state.catalogKnown,
        apiLabel,
    });
    return selectHtml(attrs, groups, encodeReviewerChoice(row));
}

function subagentIdentityMarkup(row) {
    // Identity follows the DERIVED roster route: a session reference shows its
    // harness, anything else the API channel — same shared markup direct rows use.
    const rosterRow = (state.roster || []).find(
        (item) => String(item.subagent_id || '') === String(row.subagent_id || ''));
    const route = rosterRow?.route?.kind === ROUTE_KIND_SESSION
        ? { kind: ROUTE_KIND_SESSION, target_id: rosterRow.route.target_id }
        : { kind: ROUTE_KIND_API, target_id: '' };
    return reviewerRouteIdentityMarkup(route, harnessesById(), { catalogKnown: state.catalogKnown });
}

function rowHtml(row, group) {
    const { catalogKnown, accountsKnown } = state;
    if (row.subagent_id) {
        const last = state.lastExecutions[row.slot_id];
        const lastText = last ? describeLastExecution(last) : '';
        const metaParts = [describeSubagentReference(row.subagent_id, state.roster, { rosterKnown: state.rosterKnown })];
        if (lastText) metaParts.push(`Last run: ${lastText}`);
        return `
        <div class="reviewer-slot-row" data-slot-group="${group}" data-slot-id="${escapeHtml(row.slot_id)}">
            ${subagentIdentityMarkup(row)}
            <div class="reviewer-slot-controls">
                ${reviewerPickerHtml('data-slot-route aria-label="Reviewer"', row)}
                ${effortSelectHtml('data-slot-effort aria-label="Reasoning effort"', row.effort || '', 'subagent default')}
                <button type="button" class="btn btn-default" data-slot-remove title="Remove this slot">Remove</button>
            </div>
            <div class="reviewer-slot-meta muted"${last ? ` title="${escapeHtml(lastRunMetaTitle(last))}"` : ''}>${escapeHtml(metaParts.join(' · '))}</div>
        </div>
    `;
    }
    const session = row.route.kind === ROUTE_KIND_SESSION;
    const split = session ? splitSessionTarget(row.route.target_id) : { harness: '', model: '' };
    const harness = session ? harnessesById()[split.harness] : null;
    const modelOptions = sessionModelOptions(harness, split.model, { catalogKnown });
    const profiles = session ? (state.profilesByHarness[split.harness] || []) : [];
    const profileOptions = profileOptionsFor(profiles, row.route.profile_id, { accountsKnown });
    const last = state.lastExecutions[row.slot_id];
    const lastText = last ? describeLastExecution(last) : '';
    // ONE quiet meta line per row (owner feedback): the delivery badge and the
    // last-run projection share it; nothing is dropped — the raw route + ISO
    // timestamp live in the tooltip.
    const metaParts = [capabilityBadge(row, harnessesById(), { catalogKnown })];
    const modelsGap = session ? modelsGapNote(harness, catalogKnown) : '';
    if (modelsGap) metaParts.push(modelsGap);
    if (lastText) metaParts.push(`Last run: ${lastText}`);
    const surfaceDefault = CATEGORIES[group]?.surfaceDefault || 'review effort';
    return `
        <div class="reviewer-slot-row" data-slot-group="${group}" data-slot-id="${escapeHtml(row.slot_id)}">
            ${reviewerRouteIdentityMarkup(row.route, harnessesById(), { catalogKnown })}
            <div class="reviewer-slot-controls">
                ${reviewerPickerHtml('data-slot-route aria-label="Reviewer"', row)}
                ${session ? '' : `<input data-slot-custom-api list="reviewer-api-model-catalog" placeholder="provider/model-id" value="${escapeHtml(row.route.target_id || '')}" spellcheck="false" aria-label="API model id">`}
                ${session ? selectHtml('data-slot-model aria-label="Harness model"', [{ label: '', options: modelOptions }], split.model) : ''}
                ${session && profileOptions.length > 1 ? selectHtml('data-slot-profile aria-label="Credential account"', [{ label: '', options: profileOptions }], row.route.profile_id || '') : ''}
                ${effortSelectHtml('data-slot-effort aria-label="Reasoning effort"', row.effort, surfaceDefault)}
                <button type="button" class="btn btn-default" data-slot-remove title="Remove this slot">Remove</button>
            </div>
            <div class="reviewer-slot-meta muted"${last ? ` title="${escapeHtml(lastRunMetaTitle(last))}"` : ''}>${escapeHtml(metaParts.join(' · '))}</div>
        </div>
    `;
}

function singletonHtml(spec) {
    // ONE renderer for both single-row categories (advisory, deep self-review):
    // the same picker, model/account/effort controls and meta line the triad
    // rows use, parameterized by the SINGLETONS entry — never a second copy
    // per category.
    const row = state[spec.stateKey];
    const { catalogKnown, accountsKnown } = state;
    const a = spec.attr;
    const last = state.lastExecutions[spec.lastKey];
    const lastText = last ? describeLastExecution(last) : '';
    const enabled = spec.enabledToggle
        ? `<label class="local-toggle"><input type="checkbox" data-${a}-enabled ${row.enabled !== false ? 'checked' : ''}> Enabled</label>`
        : '';
    const meta = (parts) => `<div class="reviewer-slot-meta muted"${last ? ` title="${escapeHtml(lastRunMetaTitle(last))}"` : ''}>${escapeHtml(parts.join(' · '))}</div>`;
    if (row.subagent_id) {
        const metaParts = [
            describeSubagentReference(row.subagent_id, state.roster, { rosterKnown: state.rosterKnown }),
            ...(spec.badgeOnReference ? [spec.badge(row)] : []),
            ...spec.extraMeta(row),
        ];
        if (lastText) metaParts.push(`Last run: ${lastText}`);
        return `
        <div class="reviewer-slot-row" data-${a}-row>
            ${subagentIdentityMarkup(row)}
            <div class="reviewer-slot-controls">
                ${enabled}
                ${reviewerPickerHtml(`data-${a}-route aria-label="${spec.ariaName} reviewer"`, row)}
                ${effortSelectHtml(`data-${a}-effort aria-label="${spec.ariaName} effort"`, row.effort || '', 'subagent default')}
            </div>
            ${meta(metaParts)}
        </div>
    `;
    }
    // Both singletons share the closed route vocabulary (api_chat |
    // agent_session), labeled neutrally, never a vendor name (the retired
    // Claude-SDK 'api' kind is parse-only migration now).
    const session = row.route?.kind === ROUTE_KIND_SESSION;
    const split = session ? splitSessionTarget(row.route.target_id) : { harness: '', model: '' };
    // Session branch: the SAME model-options fragment the triad rows use —
    // rewriting it here would lose the "(not in discovery)" guard and let a
    // Save with the daemon down erase the owner's model. Api branch: the same
    // catalog-assisted free-text entry the triad rows use.
    const modelOptions = session
        ? sessionModelOptions(harnessesById()[split.harness], split.model, { catalogKnown }) : [];
    const profiles = session ? (state.profilesByHarness[split.harness] || []) : [];
    const profileOptions = session
        ? profileOptionsFor(profiles, row.route?.profile_id, { accountsKnown }) : [];
    const metaParts = [spec.badge(row), ...spec.extraMeta(row)];
    const modelsGap = session ? modelsGapNote(harnessesById()[split.harness], catalogKnown) : '';
    if (modelsGap) metaParts.push(modelsGap);
    if (lastText) metaParts.push(`Last run: ${lastText}`);
    return `
        <div class="reviewer-slot-row" data-${a}-row>
            ${reviewerRouteIdentityMarkup(row.route, harnessesById(), { catalogKnown })}
            <div class="reviewer-slot-controls">
                ${enabled}
                ${reviewerPickerHtml(`data-${a}-route aria-label="${spec.ariaName} reviewer"`, row, { apiLabel: spec.apiLabel })}
                ${session
                    ? selectHtml(`data-${a}-model aria-label="${spec.ariaName} harness model"`, [{ label: '', options: modelOptions }], split.model)
                    : `<input data-${a}-api-model list="reviewer-api-model-catalog" placeholder="${spec.apiPlaceholder}" value="${escapeHtml(row.route?.target_id || '')}" spellcheck="false" aria-label="${spec.ariaName} model id">`}
                ${session && profileOptions.length > 1 ? selectHtml(`data-${a}-profile aria-label="${spec.ariaName} credential account"`, [{ label: '', options: profileOptions }], row.route?.profile_id || '') : ''}
                ${effortSelectHtml(
                    `data-${a}-effort aria-label="${spec.ariaName} effort"`,
                    session ? row.effort : (row.effort === spec.apiEffortDefault ? '' : row.effort),
                    session ? 'route default' : spec.apiEffortLabel,
                )}
            </div>
            ${meta(metaParts)}
        </div>
    `;
}

function renderRows() {
    const errorBox = document.getElementById('reviewer-slots-error');
    if (errorBox) {
        errorBox.hidden = !(state.configError || state.loadError);
        errorBox.textContent = state.configError
            ? `Saved reviewer-slot configuration is invalid and blocks reviews: ${state.configError}. `
              + 'To repair it, add at least one triad slot and one scope slot from the group headers below, then Save'
              + (state.triad.length && state.scope.length ? '.' : ' — Save will report the missing rows.')
            : (state.loadError
                ? `Could not reach the reviewer-slot settings — ${state.loadError}. Your saved configuration is unchanged; retry when the connection is back.`
                : '');
    }
    // Why the agent service could not be read is the TAB's banner now (one
    // place, not one per section). What stays here is the fact only this
    // section knows: a row pinned to an account that is really gone.
    const pinsBox = document.getElementById('reviewer-slots-pins');
    if (pinsBox) {
        const text = pinnedAccountWarning(state);
        pinsBox.hidden = !text;
        pinsBox.textContent = text;
    }
    const singles = Object.values(SINGLETONS).map((spec) => [spec, document.getElementById(spec.rowId)]);
    const boxes = Object.entries(CATEGORIES).map(([group, cat]) => [group, cat, document.getElementById(cat.rowsId)]);
    if (singles.some(([, box]) => !box) || boxes.some(([, , box]) => !box)) return;
    for (const [group, cat, box] of boxes) {
        box.innerHTML = categoryRows(group).map((row) => rowHtml(row, group)).join('')
            || `<div class="muted">${cat.empty}</div>`;
        // The count stays in the heading; what the limit MEANS moved to the
        // span's title (owner feedback: headers carried parenthetical jargon).
        const limitEl = document.getElementById(cat.limitId);
        if (limitEl) limitEl.textContent = `${categoryRows(group).length}/${state.limits[cat.limitKey]}`;
        const addEl = document.getElementById(cat.addId);
        if (addEl) addEl.disabled = categoryRows(group).length >= state.limits[cat.limitKey];
    }
    for (const [spec, box] of singles) box.innerHTML = singletonHtml(spec);
    const datalist = document.getElementById('reviewer-api-model-catalog');
    if (datalist) {
        datalist.innerHTML = state.catalogModels
            .map((id) => `<option value="${escapeHtml(id)}"></option>`).join('');
    }
    bindRowEvents();
}

function findRow(group, slotId) {
    return (categoryRows(group) || []).find((row) => row.slot_id === slotId) || null;
}

function bindRowEvents() {
    const section = document.getElementById('reviewer-slots-section');
    if (!section) return;
    section.querySelectorAll('.reviewer-slot-row[data-slot-id]').forEach((rowEl) => {
        const group = rowEl.dataset.slotGroup;
        const row = findRow(group, rowEl.dataset.slotId);
        if (!row) return;
        rowEl.querySelector('[data-slot-route]')?.addEventListener('change', (event) => {
            const value = String(event.target.value || '');
            if (value.startsWith(SUBAGENT_CHOICE_PREFIX)) {
                // A reference pick. The inline route stays stashed on the row
                // object — picking a channel again restores it.
                row.subagent_id = value.slice(SUBAGENT_CHOICE_PREFIX.length);
                renderRows();
                state.onChange();
                return;
            }
            row.subagent_id = '';
            if (!row.route?.kind) row.route = { kind: ROUTE_KIND_API, target_id: '' };
            const decoded = decodeRouteChoice(value);
            if (decoded.kind === ROUTE_KIND_SESSION) {
                const prevHarness = row.route.kind === ROUTE_KIND_SESSION
                    ? splitSessionTarget(row.route.target_id).harness : '';
                if (prevHarness !== decoded.harness) {
                    row.route = { kind: ROUTE_KIND_SESSION, target_id: decoded.harness, profile_id: '' };
                }
            } else if (row.route.kind !== ROUTE_KIND_API) {
                // A session spec is not an API model id: the input starts
                // empty (placeholder visible), display matching state.
                row.route = { kind: ROUTE_KIND_API, target_id: '' };
            }
            renderRows();
            state.onChange();
        });
        rowEl.querySelector('[data-slot-custom-api]')?.addEventListener('input', (event) => {
            row.route.target_id = String(event.target.value || '').trim();
            state.onChange();
        });
        rowEl.querySelector('[data-slot-model]')?.addEventListener('change', (event) => {
            const split = splitSessionTarget(row.route.target_id);
            row.route.target_id = composeSessionTarget(split.harness, event.target.value);
            state.onChange();
        });
        rowEl.querySelector('[data-slot-profile]')?.addEventListener('change', (event) => {
            row.route.profile_id = String(event.target.value || '');
            state.onChange();
        });
        rowEl.querySelector('[data-slot-effort]')?.addEventListener('change', (event) => {
            row.effort = String(event.target.value || '');
            state.onChange();
        });
        rowEl.querySelector('[data-slot-remove]')?.addEventListener('click', () => {
            const rows = categoryRows(group);
            const index = rows.indexOf(row);
            if (index >= 0) rows.splice(index, 1);
            renderRows();
            state.onChange();
        });
    });
    for (const spec of Object.values(SINGLETONS)) bindSingletonEvents(section, spec);
}

function bindSingletonEvents(section, spec) {
    // ONE binder for both single-row categories; the row object is the
    // category's state entry and the per-kind route memory is the spec's.
    const el = section.querySelector(`[data-${spec.attr}-row]`);
    if (!el) return;
    const a = spec.attr;
    const row = state[spec.stateKey];
    // An edit MATERIALIZES an optional singleton: from here on the save
    // payload carries it (an untouched placeholder is omitted, see
    // buildReviewerSlotsSetting).
    const edited = () => {
        if (spec.materializeOnEdit) row.materialized = true;
        state.onChange();
    };
    el.querySelector(`[data-${a}-enabled]`)?.addEventListener('change', (event) => {
        row.enabled = Boolean(event.target.checked);
        edited();
    });
    el.querySelector(`[data-${a}-route]`)?.addEventListener('change', (event) => {
        const value = String(event.target.value || '');
        if (value.startsWith(SUBAGENT_CHOICE_PREFIX)) {
            const next = advisoryReferenceTransition(row, value.slice(SUBAGENT_CHOICE_PREFIX.length));
            row.subagent_id = next.subagent_id;
            row.effort = next.effort;
            if (spec.materializeOnEdit) row.materialized = true;
            renderRows();
            state.onChange();
            return;
        }
        row.subagent_id = '';
        if (!row.route?.kind) row.route = { kind: ROUTE_KIND_API, target_id: '' };
        const result = advisoryRouteTransition(row.route, decodeRouteChoice(value), spec.memory);
        row.route = result.route;
        Object.assign(spec.memory, result.memory);
        if (row.route.kind !== ROUTE_KIND_SESSION && !row.effort && spec.apiEffortDefault) {
            row.effort = spec.apiEffortDefault;
        }
        if (spec.materializeOnEdit) row.materialized = true;
        renderRows();
        state.onChange();
    });
    // Mirrors of the triad-row model/api/profile handlers. These controls
    // exist only on their own kind's branch of singletonHtml, so each
    // querySelector binds at most one of them per render.
    el.querySelector(`[data-${a}-model]`)?.addEventListener('change', (event) => {
        const split = splitSessionTarget(row.route?.target_id);
        row.route.target_id = composeSessionTarget(split.harness, event.target.value);
        edited();
    });
    el.querySelector(`[data-${a}-api-model]`)?.addEventListener('input', (event) => {
        row.route.target_id = String(event.target.value || '').trim();
        edited();
    });
    el.querySelector(`[data-${a}-profile]`)?.addEventListener('change', (event) => {
        row.route.profile_id = String(event.target.value || '');
        edited();
    });
    el.querySelector(`[data-${a}-effort]`)?.addEventListener('change', (event) => {
        const selected = String(event.target.value || '');
        // Empty means "the surface's default": the api default (low for the
        // advisory, the Behavior-tab deep effort for deep self-review), the
        // route/roster-row default on a session or subagent reference.
        row.effort = selected
            || (row.subagent_id || row.route?.kind === ROUTE_KIND_SESSION ? '' : spec.apiEffortDefault);
        edited();
    });
}

function addRow(group) {
    const cat = CATEGORIES[group];
    const rows = categoryRows(group);
    if (rows.length >= state.limits[cat.limitKey]) return;
    // Ids are unique across EVERY category: one identity space, one history per row.
    const taken = Object.keys(CATEGORIES).flatMap((g) => categoryRows(g)).map((row) => row.slot_id);
    rows.push({
        slot_id: mintSlotId(cat.idPrefix, taken),
        route: { kind: ROUTE_KIND_API, target_id: '' },
        subagent_id: '',
        effort: '',
    });
    renderRows();
    // The Add button sits in the group's header while the new row lands at
    // the group's end — reveal it there and hand the caret to its picker.
    const added = document.getElementById(group === 'scope' ? 'reviewer-scope-rows' : 'reviewer-triad-rows')
        ?.lastElementChild;
    revealNewRow(added, added?.querySelector?.('[data-slot-route]'));
    state.onChange();
}

export async function reloadReviewerSlots() {
    try {
        const resp = await apiFetch('/api/reviewer-slots', { cache: 'no-store' });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        state.loadError = '';
        state.configError = String(data.config_error || '');
        state.source = String(data.source || '');
        state.limits = data.limits || state.limits;
        state.lastExecutions = data.last_executions || {};
        // Rows spread as-is so a subagent_id reference rides along; a direct
        // route gets its own object. The advisory's non-session kind is
        // normalized to the shared api_chat spelling here — the retired legacy
        // 'api' (Claude-SDK) kind is parse-only server-side and this UI never
        // writes it again.
        const rowIn = (row) => ({ ...row, route: { ...(row.route || {}) } });
        state.triad = Array.isArray(data.triad) ? data.triad.map(rowIn) : [];
        state.scope = Array.isArray(data.scope) ? data.scope.map(rowIn) : [];
        state.advisory = data.advisory ? rowIn(data.advisory) : state.advisory;
        if (state.advisory.route?.kind !== ROUTE_KIND_SESSION) {
            state.advisory.route = { ...(state.advisory.route || {}), kind: ROUTE_KIND_API };
        }
        // Seed the per-kind route memory with the SAVED route, so switching
        // the advisory select away and back restores it (finding #7c).
        advisoryRouteMemory[state.advisory.route?.kind === ROUTE_KIND_SESSION ? 'session' : 'api']
            = { ...(state.advisory.route || {}) };
        // The deep self-review singleton: the server answers with the saved row
        // or, unsaved, the row synthesized from the legacy model key (labeled
        // so); beside a config_error that synthesized row is a legacy-derived
        // REPAIR PLACEHOLDER — no row is effective until the setting is repaired.
        // The label rides the state, never the saved bytes.
        const deep = data.deep_review ? rowIn(data.deep_review) : {};
        state.deepReview = {
            route: deep.route?.kind === ROUTE_KIND_SESSION ? deep.route : { ...(deep.route || {}), kind: ROUTE_KIND_API },
            effort: String(deep.effort || ''),
            subagent_id: String(deep.subagent_id || ''),
            synthesizedFrom: String(deep.synthesized_from || ''),
            // Only a SAVED row is materialized on load; a synthesized one (or
            // no row at all, e.g. beside a config_error on an older server)
            // stays an omitted placeholder until the owner edits it.
            materialized: Boolean(data.deep_review) && !deep.synthesized_from,
        };
        deepReviewRouteMemory[state.deepReview.route?.kind === ROUTE_KIND_SESSION ? 'session' : 'api']
            = { ...(state.deepReview.route || {}) };
        // The VIEW loaded even when the saved value is invalid — that is exactly the
        // state the owner repairs from, and treating it as "not loaded" made the
        // save drop the repair (see collectReviewerSlots).
        state.loaded = true;
    } catch (error) {
        // A transport failure is NOT a verdict on the saved configuration: the
        // config-error banner accuses the owner's settings of blocking review, and
        // a network blip must never say that. Separate field, separate sentence.
        state.loaded = false;
        state.loadError = `could not load reviewer slots: ${error.message || error}`;
    }
    // ONE status read for the whole app (the accounts panel and the Subagents
    // section share this request; `includeModels` is sticky, so no later read
    // downgrades the snapshot these selects depend on). Awaited only for a
    // BOUNDED beat: this read can wake a cold Claudexor daemon and walk
    // per-harness model discovery, and awaiting it outright held the Save
    // button (loadSettings awaits this function) hostage for the whole probe.
    // A warm daemon settles inside the beat and keeps the exact old
    // semantics; a cold one keeps refreshing in the background and the status
    // surface binding repaints these rows when the snapshot lands.
    await boundedStatusRefresh(state.store);
    adoptStatusSnapshot();
    renderRows();
}

function adoptStatusSnapshot() {
    // Each facet answers for itself. A never-read catalog and a never-read
    // account store are separate gaps, and neither is evidence that a saved
    // route or pin no longer exists. The tab's ONE service banner explains the
    // gap — every facet it lost, named — so this section adds no second
    // sentence about it; a facet that WAS read keeps its authoritative list.
    state.catalogKnown = state.store.catalogKnown;
    state.accountsKnown = state.store.accountsKnown;
    const snapshot = state.store.snapshot || {};
    state.harnesses = state.catalogKnown && Array.isArray(snapshot.harnesses) ? snapshot.harnesses : [];
    state.profilesByHarness = state.accountsKnown ? indexProfilesByHarness(snapshot) : {};
}

// The roster the «Configured subagent» selects offer, adopted from the SAME
// settings document (and parser) the Available-subagents section loads —
// settings.js calls this from applySettings, right beside that section's own
// load. A roster that could not be parsed licenses no absence claim about a
// saved reference (rosterKnown=false), exactly like an unread status facet.
export function adoptSubagentRoster(settings) {
    const parsed = parseAvailableSubagentsSetting(availableSubagentsLoadValue(settings));
    state.roster = parsed.setting ? parsed.setting.items : [];
    state.rosterKnown = Boolean(parsed.setting);
    renderRows();
}

export function initReviewerSlots({ onChange, store = claudexorStatus } = {}) {
    destroyReviewerSlots();
    state.onChange = typeof onChange === 'function' ? onChange : () => {};
    state.store = store;
    // Seed from whatever the shared store already holds, so a render triggered
    // before the first notify (the model-catalog event) is not rendered from a
    // blank derived state.
    adoptStatusSnapshot();
    // Follow the shared read: when the daemon comes up while Settings is open
    // the rows stop claiming "(not in discovery)" without a page reload. That
    // promise needs the SHARED surface binding — a bare subscribe() carries no
    // visibility predicate, and the store never polls for a subscriber that
    // cannot say it is on screen, so nothing ever arrived to react to. Only a
    // change this section RENDERS repaints: a repaint on every poll tick would
    // drop the caret out of the API-model field mid-typing.
    let signature = '';
    state.disposers.push(bindStatusSurface(state.store, {
        elementId: 'reviewer-triad-rows',
        includeModels: true,
        listener: () => {
            adoptStatusSnapshot();
            const next = JSON.stringify([state.catalogKnown, state.accountsKnown,
                state.harnesses, state.profilesByHarness]);
            if (next === signature) return;
            signature = next;
            renderRows();
        },
    }));
    for (const [group, cat] of Object.entries(CATEGORIES)) {
        document.getElementById(cat.addId)?.addEventListener('click', () => addRow(group));
    }
    const onCatalog = (event) => {
        const items = event?.detail?.items || [];
        state.catalogModels = items.map((item) => String(item.value || item.id || '')).filter(Boolean);
        renderRows();
    };
    document.addEventListener('settings-model-catalog:updated', onCatalog);
    state.disposers.push(() => document.removeEventListener('settings-model-catalog:updated', onCatalog));
    // The initial load is driven by settings.js loadSettings(), which awaits
    // reloadReviewerSlots() BEFORE taking the clean-draft baseline — otherwise
    // the async arrival of the rows would read as an unsaved edit.
}

export function destroyReviewerSlots() {
    for (const dispose of state.disposers.splice(0)) {
        try { dispose(); } catch (err) { /* a broken disposer must not block the rest */ }
    }
}

// #126, pure and node-tested: what the settings save sends for reviewer slots.
// Never author the setting from an UNLOADED view (an unrelated save must not
// overwrite the owner's configuration with an empty page), and a transport
// failure is not a verdict on the saved value either. But a LOADED view always
// sends what it shows — including an empty triad/scope. The old empty-set
// guard silently dropped the key, so deleting every row Saved "successfully"
// while saving nothing; now the backend's own 400 («triad needs at least one
// slot») surfaces through the existing failed-save status. Validation SSOT
// stays on the backend — no client-side duplicate.
export function reviewerSlotsSavePayload({ loaded = false, loadError = '', triad = [], scope = [], advisory, deepReview } = {}) {
    if (loadError || !loaded) return {};
    return { OUROBOROS_REVIEWER_SLOTS: buildReviewerSlotsSetting({ triad, scope, advisory, deepReview }) };
}

export function collectReviewerSlots() {
    // A config_error is NOT the unloaded case — the stored value is already
    // invalid and blocking review, the endpoint returns no rows with it, and
    // refusing here made the documented repair path swallow the owner's
    // replacement rows and still report success.
    return reviewerSlotsSavePayload(state);
}
