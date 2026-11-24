// Harness identity presentation SSOT.
//
// Brand vector provenance (paths and viewBoxes preserved byte-for-byte from
// Claudexor's HarnessLogoData.swift): Claude, Cursor and OpenCode are from
// Simple Icons; Codex/OpenAI is from SVGL. Antigravity's is Google's own one-colour
// mark (antigravity.google/assets/image/brand/antigravity-icon__one-color.svg), not
// Claudexor's or Simple Icons'. The marks are monochrome and inherit currentColor.
// Product names and marks remain the property of their owners.
//
// This module presents identity: harness and source marks and names, the route grammar selections and keys are read
// with (a local connection's name masked for display), compact model names, and the words of an executor's and a
// coordinator's recorded facts. It decides no route availability, status or review authority; which executor and
// request a card states across frames and attempts is executor_card_facts.js's.

import { escapeHtmlAttr as escapeHtml, escapeHtmlText } from './utils.js';
import { fmt } from './i18n.js';

export const HARNESS_MARKS = Object.freeze({
    codex: Object.freeze({
        label: 'Codex',
        viewBox: '0 0 256 260',
        path: 'M239.184 106.203a64.716 64.716 0 0 0-5.576-53.103C219.452 28.459 191 15.784 163.213 21.74A65.586 65.586 0 0 0 52.096 45.22a64.716 64.716 0 0 0-43.23 31.36c-14.31 24.602-11.061 55.634 8.033 76.74a64.665 64.665 0 0 0 5.525 53.102c14.174 24.65 42.644 37.324 70.446 31.36a64.72 64.72 0 0 0 48.754 21.744c28.481.025 53.714-18.361 62.414-45.481a64.767 64.767 0 0 0 43.229-31.36c14.137-24.558 10.875-55.423-8.083-76.483Zm-97.56 136.338a48.397 48.397 0 0 1-31.105-11.255l1.535-.87 51.67-29.825a8.595 8.595 0 0 0 4.247-7.367v-72.85l21.845 12.636c.218.111.37.32.409.563v60.367c-.056 26.818-21.783 48.545-48.601 48.601Zm-104.466-44.61a48.345 48.345 0 0 1-5.781-32.589l1.534.921 51.722 29.826a8.339 8.339 0 0 0 8.441 0l63.181-36.425v25.221a.87.87 0 0 1-.358.665l-52.335 30.184c-23.257 13.398-52.97 5.431-66.404-17.803ZM23.549 85.38a48.499 48.499 0 0 1 25.58-21.333v61.39a8.288 8.288 0 0 0 4.195 7.316l62.874 36.272-21.845 12.636a.819.819 0 0 1-.767 0L41.353 151.53c-23.211-13.454-31.171-43.144-17.804-66.405v.256Zm179.466 41.695-63.08-36.63L161.73 77.86a.819.819 0 0 1 .768 0l52.233 30.184a48.6 48.6 0 0 1-7.316 87.635v-61.391a8.544 8.544 0 0 0-4.4-7.213Zm21.742-32.69-1.535-.922-51.619-30.081a8.39 8.39 0 0 0-8.492 0L99.98 99.808V74.587a.716.716 0 0 1 .307-.665l52.233-30.133a48.652 48.652 0 0 1 72.236 50.391v.205ZM88.061 139.097l-21.845-12.585a.87.87 0 0 1-.41-.614V65.685a48.652 48.652 0 0 1 79.757-37.346l-1.535.87-51.67 29.825a8.595 8.595 0 0 0-4.246 7.367l-.051 72.697Zm11.868-25.58 28.138-16.217 28.188 16.218v32.434l-28.086 16.218-28.188-16.218-.052-32.434Z',
    }),
    claude: Object.freeze({
        label: 'Claude Code',
        viewBox: '0 0 24 24',
        path: 'm4.7144 15.9555 4.7174-2.6471.079-.2307-.079-.1275h-.2307l-.7893-.0486-2.6956-.0729-2.3375-.0971-2.2646-.1214-.5707-.1215-.5343-.7042.0546-.3522.4797-.3218.686.0608 1.5179.1032 2.2767.1578 1.6514.0972 2.4468.255h.3886l.0546-.1579-.1336-.0971-.1032-.0972L6.973 9.8356l-2.55-1.6879-1.3356-.9714-.7225-.4918-.3643-.4614-.1578-1.0078.6557-.7225.8803.0607.2246.0607.8925.686 1.9064 1.4754 2.4893 1.8336.3643.3035.1457-.1032.0182-.0728-.164-.2733-1.3539-2.4467-1.445-2.4893-.6435-1.032-.17-.6194c-.0607-.255-.1032-.4674-.1032-.7285L6.287.1335 6.6997 0l.9957.1336.419.3642.6192 1.4147 1.0018 2.2282 1.5543 3.0296.4553.8985.2429.8318.091.255h.1579v-.1457l.1275-1.706.2368-2.0947.2307-2.6957.0789-.7589.3764-.9107.7468-.4918.5828.2793.4797.686-.0668.4433-.2853 1.8517-.5586 2.9021-.3643 1.9429h.2125l.2429-.2429.9835-1.3053 1.6514-2.0643.7286-.8196.85-.9046.5464-.4311h1.0321l.759 1.1293-.34 1.1657-1.0625 1.3478-.8804 1.1414-1.2628 1.7-.7893 1.36.0729.1093.1882-.0183 2.8535-.607 1.5421-.2794 1.8396-.3157.8318.3886.091.3946-.3278.8075-1.967.4857-2.3072.4614-3.4364.8136-.0425.0304.0486.0607 1.5482.1457.6618.0364h1.621l3.0175.2247.7892.522.4736.6376-.079.4857-1.2142.6193-1.6393-.3886-3.825-.9107-1.3113-.3279h-.1822v.1093l1.0929 1.0686 2.0035 1.8092 2.5075 2.3314.1275.5768-.3218.4554-.34-.0486-2.2039-1.6575-.85-.7468-1.9246-1.621h-.1275v.17l.4432.6496 2.3436 3.5214.1214 1.0807-.17.3521-.6071.2125-.6679-.1214-1.3721-1.9246L14.38 17.959l-1.1414-1.9428-.1397.079-.674 7.2552-.3156.3703-.7286.2793-.6071-.4614-.3218-.7468.3218-1.4753.3886-1.9246.3157-1.53.2853-1.9004.17-.6314-.0121-.0425-.1397.0182-1.4328 1.9672-2.1796 2.9446-1.7243 1.8456-.4128.164-.7164-.3704.0667-.6618.4008-.5889 2.386-3.0357 1.4389-1.882.929-1.0868-.0062-.1579h-.0546l-6.3385 4.1164-1.1293.1457-.4857-.4554.0608-.7467.2307-.2429 1.9064-1.3114Z',
    }),
    cursor: Object.freeze({
        label: 'Cursor',
        viewBox: '0 0 24 24',
        path: 'M11.503.131 1.891 5.678a.84.84 0 0 0-.42.726v11.188c0 .3.162.575.42.724l9.609 5.55a1 1 0 0 0 .998 0l9.61-5.55a.84.84 0 0 0 .42-.724V6.404a.84.84 0 0 0-.42-.726L12.497.131a1.01 1.01 0 0 0-.996 0M2.657 6.338h18.55c.263 0 .43.287.297.515L12.23 22.918c-.062.107-.229.064-.229-.06V12.335a.59.59 0 0 0-.295-.51l-9.11-5.257c-.109-.063-.064-.23.061-.23',
    }),
    opencode: Object.freeze({
        label: 'OpenCode',
        viewBox: '0 0 24 24',
        path: 'M22 24H2V0h20zM17 4.8H7v14.4h10z',
    }),
    agy: Object.freeze({
        label: 'Antigravity',
        viewBox: '0 0 180 180',
        path: 'M144.248 149.062C151.748 154.688 162.998 150.938 152.685 140.625C121.748 110.625 128.31 28.125 89.8727 28.125C51.4352 28.125 57.9977 110.625 27.0602 140.625C15.8102 151.875 27.9977 154.688 35.4977 149.062C64.5602 129.375 62.6852 94.6875 89.8727 94.6875C117.06 94.6875 115.185 129.375 144.248 149.062Z',
    }),
});

export const GENERIC_HARNESS_MARK = Object.freeze({
    viewBox: '0 0 24 24',
    path: 'M12 2 22 12 12 22 2 12Z',
});

const GENERIC_LABELS = Object.freeze({ copilot: 'GitHub Copilot' });
// Names of the API providers (a task card's route rows, the route editor's fallback when the setup contract
// does not describe one, e.g. GigaChat). The contract's label wins wherever it exists.
export const API_PROVIDER_LABELS = Object.freeze({
    openrouter: 'OpenRouter', openai: 'OpenAI', anthropic: 'Anthropic', deepseek: 'DeepSeek',
    zai: 'Z.ai (GLM)', minimax: 'MiniMax', cloudru: 'Cloud.ru Foundation Models', gigachat: 'GigaChat',
    'openai-compatible': 'OpenAI-compatible endpoint',
});

// The vendors' plan ids as the owner's words; an id not listed here stays itself (honest, never guessed). The raw
// id is a fact of the account's Resources panel.
const PLAN_LABELS = Object.freeze({
    free: 'Free', plus: 'Plus', pro: 'Pro', prolite: 'Pro Lite', promax: 'Pro Max', max: 'Max', team: 'Team',
    business: 'Business', enterprise: 'Enterprise', claude_pro: 'Pro', claude_max: 'Max', claude_team: 'Team',
    claude_enterprise: 'Enterprise', self_serve_business_usage_based: 'Business usage-based',
});

/** The plan's word for a vendor plan id, or the id itself. */
export function planLabel(plan) {
    const raw = String(plan ?? '').trim();
    return PLAN_LABELS[raw.toLowerCase()] || raw;
}
const API_IDS = new Set(['api', 'api_chat', 'api_model', 'native']);

/** Return presentation facts only; callers retain status and evidence ownership. */
export function harnessPresentation(harnessId, { label = '', channel = '' } = {}) {
    const rawId = String(harnessId ?? '').trim();
    const normalized = rawId.toLowerCase();
    const isApi = channel === 'api' || API_IDS.has(normalized);
    if (isApi) {
        return Object.freeze({
            kind: 'channel',
            id: 'api',
            harnessId: null,
            label: String(label || '').trim() || 'API',
            known: true,
            generic: true,
            ...GENERIC_HARNESS_MARK,
        });
    }
    const branded = HARNESS_MARKS[normalized];
    const visibleLabel = String(label || '').trim()
        || branded?.label || GENERIC_LABELS[normalized] || rawId || 'Unknown harness';
    return Object.freeze({
        ...(branded || GENERIC_HARNESS_MARK),
        kind: 'harness',
        id: normalized || rawId,
        harnessId: rawId,
        label: visibleLabel,
        known: Boolean(branded || GENERIC_LABELS[normalized]),
        generic: !branded,
    });
}

/** The presentation facts of a saved subscription model source (`claudexor::<source>=…`): the one visual answer of
 *  its name and mark every consumer draws. A descriptor held for THIS source is the record — its label is
 *  authoritative, and its `credentialHarness` names the program whose mark presents it, an explicitly empty one
 *  naming none; with no label of its own, an unbound source keeps its id as its name. With no descriptor held (the
 *  list not read or late, or a read that no longer lists this saved source) the drawing still describes the saved
 *  source, never its right to call it: an id that exactly matches a HARNESS_MARKS entry wears that program's own
 *  name and mark, any other id the neutral mark beside its own name — never a letter, a substring or an alias
 *  guess. Such an id names no program (`harnessId` empty), so a consumer that draws the mark from it cannot read
 *  the spelling back into one. Presentation only: access keeps the descriptor's binding
 *  (route_editor_primitives.routeModelFields). */
export function sourcePresentation(sourceId, modelSources = []) {
    const id = String(sourceId ?? '').trim();
    const descriptor = (Array.isArray(modelSources) ? modelSources : []).find((entry) => entry?.id === id);
    if (descriptor) {
        const binding = String(descriptor.credentialHarness ?? '');
        return harnessPresentation(binding, { label: String(descriptor.label || '').trim() || (binding ? '' : id) });
    }
    // Without a descriptor the id is opaque, so only its OWN exact key may speak for a program: `harnessPresentation`
    // folds case and reserves the API channel words for a genuinely identified program, so through it `CLAUDE` would
    // borrow Claude's brand and `native` would be called API. An id no key of its own names keeps its own word beside
    // the neutral mark and claims no program — the shape an explicitly unbound record already gives.
    if (Object.prototype.hasOwnProperty.call(HARNESS_MARKS, id)) return harnessPresentation(id);
    const known = Object.prototype.hasOwnProperty.call(GENERIC_LABELS, id) ? GENERIC_LABELS[id] : '';
    return harnessPresentation('', { label: known || id });
}

/** Decorative currentColor mark plus adjacent, always-visible escaped text. */
export function harnessIdentityMarkup(harnessId, {
    label = '', channel = '', className = '',
} = {}) {
    const presentation = harnessPresentation(harnessId, { label, channel });
    const classes = ['harness-identity', className].filter(Boolean).join(' ');
    const dataIdentity = presentation.kind === 'channel' ? 'api' : presentation.harnessId;
    return `<span class="${escapeHtml(classes)}" data-presentation-kind="${presentation.kind}" data-harness-identity="${escapeHtml(dataIdentity || '')}">`
        + `<svg class="harness-identity-mark" viewBox="${presentation.viewBox}" width="1em" height="1em" fill="currentColor" aria-hidden="true" focusable="false"><path d="${presentation.path}"></path></svg>`
        + `<span class="harness-identity-label">${escapeHtml(presentation.label)}</span></span>`;
}

/** Shared visible identity for one optional named harness account. */
export function harnessAccountIdentityMarkup(harnessId, {
    label = '', profile = '', className = '',
} = {}) {
    const identity = harnessIdentityMarkup(harnessId, { label, className });
    return identity + (profile ? ` (${escapeHtml(profile)})` : '');
}

// The meta line's separator is TEXT, not a CSS gap: it has to survive a copy
// out of the page and be read aloud, and a flex gap does neither. One rule for
// every part boundary, here and in `renderLiveCardMeta`, which joins its own
// parts with the same string and receives this block as one of them.
export const META_PART_SEPARATOR = ' \u00b7 ';

export function joinMetaParts(parts) {
    return parts.filter(Boolean).join(META_PART_SEPARATOR);
}

/** The source identity as elements: a static word stays translatable chrome, an owner's or product's
 *  name is authored, and each separator is one skipped element — so no dot fuses into a name and the
 *  overlay never reports it as a key. */
export function identityPartsHtml(parts) {
    return (parts || []).filter((part) => part && String(part.text || '').trim()).map((part) => (part.kind === 'authored'
        ? `<span data-i18n-authored>${escapeHtml(part.text)}</span>`
        : `<span>${escapeHtml(part.text)}</span>`)).join(`<span class="harness-identity-sep" data-i18n-skip>${escapeHtml(META_PART_SEPARATOR)}</span>`);
}

// E2 (v6.39 UI): compact model name for the subagent label — drop the provider prefix
// ("anthropic/claude-sonnet-4.6" -> "claude-sonnet-4.6") and mark a local route once. Shared SSOT
// reused by the chat live-card headline (web/modules/chat.js).
// `local` is a recorded lane (`used_local`, `use_local_model`, a selection's own spelling) and alone decides when given.
// Without one, a route spelled `… (local)` is local, and so is an older id whose own model part says so; a provider or
// connection head never is (`openai@local-team::remote-a` names a remote model).
export function compactModel(model = '', local) {
    const m = String(model || '').trim();
    if (!m) return '';
    const spelled = / \(local\)$/i.test(m), id = spelled ? m.slice(0, -' (local)'.length).trim() : m;
    // Provider-prefixed IDs use either "provider/model" (OpenRouter) or "provider::model"
    // (direct providers, e.g. openai::gpt-5.5, cloudru::…); show just the model part. Take
    // whatever follows the LAST '/' or '::' separator.
    const slash = id.lastIndexOf('/');
    const dcolon = id.lastIndexOf('::');
    const short = id.slice(Math.max(slash >= 0 ? slash + 1 : 0, dcolon >= 0 ? dcolon + 2 : 0));
    return (typeof local === 'boolean' ? local : spelled || /local/i.test(short)) ? `${short} (local)` : short;
}

/** The harness a route spelling names (`harness[=model][:effort]`, the configured-row grammar); '' when its head is
 *  no harness id, as an API model target (`openai::gpt-5`, `anthropic/claude-x`, `claudexor::codex=gpt-5`) never is. */
export function routeHarness(route) {
    const head = String(route || '').split('=')[0].trim().toLowerCase();
    const harness = head.includes('::') ? '' : head.split(':')[0];
    return /^[a-z0-9][a-z0-9_.-]{0,63}$/.test(harness) ? harness : '';
}

// A named connection's stored name, the twin of provider_models._CONNECTION_NAME: the resolver refuses any other.
const CONNECTION_NAME = /^[a-z0-9][a-z0-9_-]{0,31}$/;
const ROUTE_KINDS = new Set(['claudexor', ...Object.keys(API_PROVIDER_LABELS)]);

/** The route grammar's head, the twin of provider_models.parse_model_route that the Models editor
 *  (route_editor_primitives.parseModelSource), a key's label (connection_labels.routeConnectionId) and a card
 *  (displayApiModelTarget, routeIdentity) share: `kind::model`, `kind@name::model` a named API connection (`@default`
 *  the built-in one), a name outside the canonical grammar or a `claudexor@` head `valid: false`, and a head naming no
 *  known kind an `opaque` OpenRouter id read whole. `connection` is the name as written; the model part is returned as
 *  written and a trailing ` (local)` stays the caller's. It reads, never rewrites, a route. */
export function apiRouteParts(route) {
    const text = String(route || '').trim(), sep = text.indexOf('::');
    const head = sep < 0 ? '' : text.slice(0, sep), at = head.indexOf('@');
    const kind = at < 0 ? head : head.slice(0, at), model = text.slice(sep + 2);
    if (sep < 0 || !ROUTE_KINDS.has(kind)) return { kind: 'openrouter', connection: '', named: false, model: text, valid: true, opaque: true };
    if (at < 0) return { kind, connection: '', named: false, model, valid: true, opaque: false };
    const connection = head.slice(at + 1);
    return { kind, connection, named: true, model, valid: kind !== 'claudexor' && CONNECTION_NAME.test(connection), opaque: false };
}

/** Mask any local API connection suffix in a known provider head (`kind@name::model` -> `kind::model`) for display.
 *  Opaque ids, `@` in model names, and one trailing ` (local)` mark stay intact; a connection-headed route with no
 *  model, a name the resolver refuses (`openai@Bad_Name::m`) or a `claudexor@…` head returns '' rather than leaking
 *  the connection segment or naming a model the route never selected. */
export function displayApiModelTarget(raw) {
    const trimmed = String(raw || '').trim();
    const local = trimmed.endsWith(' (local)');
    const parts = apiRouteParts(local ? trimmed.slice(0, -' (local)'.length) : trimmed), model = parts.model.trim();
    if (!parts.named) return trimmed;
    return parts.valid && model ? `${parts.kind}::${model}${local ? ' (local)' : ''}` : '';
}

// The route grammar's provider and model, as provider_models.parse_model_route reads a route: a known kind heads
// `kind::model`, and a bare or unknown-headed id is OpenRouter's (`sol` is `openrouter::sol`). A local connection name
// and the lane are no part of it; a caller compares lanes apart.
export function routeIdentity(route) {
    const text = displayApiModelTarget(String(route || '').replace(/ \(local\)$/i, ''));
    return apiRouteParts(text).opaque ? `openrouter::${text}` : text;
}

/** A selection's carrier as the wire defines it, `{kind, target_id}` and nothing else (a row's account pin,
 *  local API connection name, or other setting never travels); null when it names no known kind and a target. */
export function selectedRoute(route) {
    if (typeof route?.target_id !== 'string' || !['api_model', 'agent_session'].includes(route.kind)) return null;
    const target = route.kind === 'api_model' ? displayApiModelTarget(route.target_id) : route.target_id.trim();
    return target ? { kind: route.kind, target_id: target } : null;
}

/** The model a selection asked of this executor, as a compact line names it (its lane is its own route spelling's): a
 *  session row's for its own harness only, an API row's for no harness. */
export function selectedModel(selected, harness = '') {
    const own = String(harness || '').trim().toLowerCase();
    const model = selected?.kind === 'api_model' ? (own ? '' : selected.model)
        : selected?.kind === 'agent_session' && own && selected.harness === own ? selected.model : '';
    return compactModel(model, / \(local\)$/i.test(model));
}

/** The lane a card under an API selection knows for the model it names before any solve: the one its frames recorded
 *  (`kept`, the coordinator route recordedCoordinator compacts with `use_local_model`: an API child's own model) when it
 *  names this model, else the selection's own spelling for the same route, and none for another: a changed model's
 *  words never make one. Without an API selection, undefined, so the name keeps compactModel's own reading. */
export function selectionLane(selected, model, kept = '') {
    if (selected?.kind !== 'api_model' || !model) return undefined;
    const name = compactModel(model, false);
    if (kept === name || kept === `${name} (local)`) return kept !== name;
    return routeIdentity(model) === routeIdentity(selected.model) && / \(local\)$/i.test(selected.model);
}

/** Normalize coordinator route provenance from a recorded snapshot or frame: the words a frame stamped
 *  (supervisor_route.cognitive_route_provenance, what dispatch tells the nanny), else the same words read from an
 *  older frame's snapshot. */
export function formatCoordinatorProvenance(source) {
    if (!source) return '';
    if (typeof source === 'string') {
        const trimmed = source.trim();
        if (!trimmed) return '';
        const lower = trimmed.toLowerCase();
        if (lower === 'inherited' || lower === 'parent') return 'inherited';
        if (lower === 'parent_switch') return "parent's own choice";
        if (lower === 'fallback' || lower === 'parent_fallback' || lower === "parent's fallback") {
            return "parent's fallback";
        }
        if (lower.startsWith('chosen by the delegator')) return trimmed;
        if (lower.startsWith('delegator:')) {
            const row = trimmed.slice('delegator:'.length).trim();
            return row ? `chosen by the delegator (${row})` : 'chosen by the delegator';
        }
        return trimmed;
    }
    if (typeof source !== 'object') return '';
    const snap = source.parent_cognitive_route && typeof source.parent_cognitive_route === 'object'
        ? source.parent_cognitive_route
        : null;
    const chosenRow = String(
        snap?.supervisor_row || snap?.supervisor_subagent_id
        || snap?.selected_supervisor_snapshot?.handle || snap?.selected_supervisor_snapshot?.selected_subagent_id
        || snap?.row || source.supervisor_row || source.supervisor_subagent_id
        || source.selected_supervisor_snapshot?.handle || source.selected_supervisor_snapshot?.selected_subagent_id
        || '',
    ).trim();
    const stamped = String(source.coordinatorProvenance || source.coordinator_provenance || '').trim();
    const explicit = stamped || String(snap?.chosen_by || source.chosen_by || '').trim();
    const lower = explicit.toLowerCase();
    if (lower === 'delegator') {
        return chosenRow ? `chosen by the delegator (${chosenRow})` : 'chosen by the delegator';
    }
    if (lower === 'fallback' || lower === 'parent_fallback' || lower === "parent's fallback") {
        return "parent's fallback";
    }
    if (lower === 'inherited' || lower === 'parent') {
        // An inherited snapshot names the role its parent handed down (`follows_role`); a stamped word is already whole.
        const followed = stamped ? '' : String(snap?.follows_role || source.follows_role || '').trim();
        return followed ? `inherited (follows ${followed})` : 'inherited';
    }
    return explicit ? formatCoordinatorProvenance(explicit) : '';
}

const cognitiveSnapshot = (evt) => (evt?.parent_cognitive_route && typeof evt.parent_cognitive_route === 'object'
    ? evt.parent_cognitive_route : null);
const statedBilling = (evt, snap = cognitiveSnapshot(evt)) => String(
    evt?.coordinator_billing || evt?.nanny_billing || evt?.cost_kind || snap?.billing || snap?.cost_kind || '',
).trim();

/** The supervising agent's own recorded route and billing basis, for a delegated child's Coordinator row:
 *  never the executor's, so neither reads as the model the child's work ran on. */
export function coordinatorFacts(evt) {
    const snap = cognitiveSnapshot(evt);
    const rawModel = String(evt?.coordinator_route || evt?.model || snap?.model || '').trim();
    const lane = evt?.use_local_model ?? snap?.use_local_model;  // recorded, it decides with the route's spelling; else unknown
    return { route: compactModel(rawModel, lane == null ? undefined : Boolean(lane) || rawModel.endsWith(' (local)')),
        billing: statedBilling(evt, snap) || 'unknown' };
}

/** A coordinator's facts in one phrase, as the card's Coordinator row and Dashboard Logs say them: its route with any
 *  provenance, then its billing basis; nothing without a route. */
export function coordinatorLine({ route = '', billing = '', provenance = '' } = {}) {
    return route ? joinMetaParts([provenance ? `${route} (${provenance})` : route, billing && `bills as ${billing}`]) : '';
}

/** The coordinator facts a frame itself records (route, billing basis, provenance), each only when it states one, so a
 *  card keeps them past a later chip that carries none and never invents a basis no frame gave; null when none. */
export function recordedCoordinator(evt) {
    const snap = cognitiveSnapshot(evt), billing = statedBilling(evt, snap), provenance = formatCoordinatorProvenance(evt);
    if (!(snap || billing || provenance || String(evt?.coordinator_route || '').trim() || typeof evt?.use_local_model === 'boolean')) return null;
    const { route } = coordinatorFacts(evt);
    return { ...(route && { route }), ...(billing && { billing }), ...(provenance && { provenance }) };
}

/** Card identity consumes projected facts; it does not infer execution from marks.
 *  The chip's detail sentence is card text too, never a hover-only title: the
 *  `Executor` row of the card's details (chat_activity.renderLiveCardMeta), which
 *  its one disclosure opens by pointer, keyboard or touch, so the compact line
 *  never reflows when the card opens. Text-mode escaping keeps the markup equal to
 *  its serialization, which the meta line's no-op check compares, so a quote never
 *  forces a rebuild. `model` is the executor's own model fact (executorModelText),
 *  never the coordinating model: that one is a labelled row of the details. */
export function executorIdentityMarkup(chip, { model = '' } = {}) {
    if (!chip) return '';
    const identity = `<span class="harness-chip chat-live-executor-chip" title="${escapeHtml(executorTitle(chip))}">`
        + harnessIdentityMarkup(chip.harness, { label: chip.label || '', className: 'chat-live-executor-identity' }) + '</span>';
    const words = modelWords(model);  // a fact with no model (`{model: ''}`) adds no part
    return joinMetaParts([identity, words ? `<span class="chat-live-meta-text">${escapeHtmlText(words)}</span>` : '']);
}

/** An executor's detail sentence in the reader's language where its record types one: a selection that ended before any
 *  dispatch (log_events' `undispatched`) is worded at paint from its recorded harness and request, with any report it kept;
 *  every other sentence stays as recorded. */
export function executorTitle(chip) {
    if (!chip?.undispatched) return chip?.title || '';
    const name = harnessPresentation(chip.harness).label, model = chip.undispatched.model;
    const words = model ? fmt('Selected {name}, model {model} (requested); no dispatch was recorded before it ended', { name, model })
        : fmt('Selected {name}; no dispatch was recorded before it ended', { name });
    return chip.lastUpdate ? `${words}. ${chip.lastUpdate}` : words;
}

/** A compact line's model fact in the reader's language: a typed request (executorModelFact's `requested`) is translated
 *  where it is painted (a card is no overlay root, i18n.SKIP_ROOTS); a model identifier, a plain string included, stays
 *  verbatim whatever its words. */
export function modelWords(fact) {
    if (typeof fact !== 'object' || !fact) return String(fact || '');
    return fact.requested ? fmt('requested {model}', { model: fact.model }) : String(fact.model || '');
}

/** A task attempt as cards compare it (`task_attempt`: a frame's integer, an observation's decimal string), or null. */
export const taskAttempt = (value) => (/^\d+$/.test(String(value ?? '')) ? Number(value) : null);
