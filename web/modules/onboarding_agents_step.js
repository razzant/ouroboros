// Shared onboarding account connection and install-preview controller.
// Login and account truth remain in the same controller/store as Settings;
// draft compilation is read-only and completion is one atomic transaction.

import { apiClient } from './api_client.js';
import { ACCOUNTS_UNSUPPORTED_STATUS, claudexorStatus, accountRows, familyLabel } from './claudexor_status_store.js';
import { LOGIN_CARD_FULL, SIGN_IN_SCOPE_NOTE, createLoginCardController } from './harness_login_cards.js';
import { BOOTSTRAP_HARNESSES, familyStatus } from './harness_accounts.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import { bindFamilyPrograms } from './model_program.js';
import {
    availableSubagentsEditorHost,
    createAvailableSubagentsEditor,
} from './subagents_settings.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { PROCESSING_PREFERENCE_KEY, MODEL_PROCESSING_PREFERENCES_KEY } from './route_editor_primitives.js';

// Every supported task harness in the linear Available-subagents compiler; reviewer policy stays a separate core-only projection (Agy is task-only, never omitted here).
export const AGENT_FAMILIES = BOOTSTRAP_HARNESSES.map((harness) => ({ harness }));

const REVIEW_CAPABLE_AGENT_HARNESSES = new Set(['claude', 'codex', 'cursor']);

// Completion must not hang on an engine that is down: three attempts over roughly 1.2 seconds convert a
// transport blip into a proven cancel; anything longer stops being a blip.
const LOGIN_RELEASE_RETRIES = 2;
const LOGIN_RELEASE_RETRY_MS = 600;

// The three rungs, in the owner's own logic. `tone` is the one-word verdict the
// eye lands on first; `title` is the action; `body` is the honest consequence.
export const VALUE_LADDER = [
    {
        tone: 'Runs',
        title: 'One Codex subscription',
        body: 'Run models and agents without an API key. Codex is the recommended starting point.',
    },
    {
        tone: 'Better',
        title: 'Add API keys or other agents',
        body: 'API keys and local models work on their own or alongside subscriptions. '
            + 'Claude Code, Cursor, and Antigravity remain agent connections.',
    },
    {
        tone: 'Best',
        title: 'Add several accounts',
        body: 'They rotate on their own. When one account’s window is spent, the '
            + 'next compatible account can pick the work up. A pinned role waits for its account.',
    },
];

export const LADDER_FOOTNOTE =
    'Riding a plan is not free — it moves that work onto a subscription you already '
    + 'pay for instead of adding per-call API charges. What moves is exactly what you '
    + 'route: commit, plan, skill and result reviews all take their reviewers from the review pool '
    + '(the rows marked Reviewer), so an all-subscription pool puts each review on the subscription; '
    + 'Settings → Agents shows what one review by each API row actually cost (a wave costs the reviewers '
    + 'it asks), and a session spends minutes of your window per review instead.';

// ---------------------------------------------------------------------------
// Pure helpers.
// ---------------------------------------------------------------------------

export function connectedHarnesses(snapshot) {
    // Which preset families the engine vouches a signed-in account for, in family order, from `accountRows` — the SSOT
    // projection — so this step and the accounts panel never disagree about "connected"; an OFF account is skipped, absent fail-open.
    const passed = new Set(
        accountRows(snapshot)
            .filter((row) => row?.enabled !== false
                && String(row?.status?.verification || '') === 'passed')
            .map((row) => String(row.harness || '')),
    );
    return AGENT_FAMILIES.map((f) => f.harness).filter((harness) => passed.has(harness));
}

export function familyLabels(harnesses, snapshot, { catalogKnown = false } = {}) {
    // ONE authority, shared with the Agents tab: prefer the engine's own display_name, fall back to
    // the bootstrap product names. Passing the snapshot keeps this pure and node-testable; without
    // it the wizard would print a raw `claude` the day the engine renames a family or adds a fourth.
    return (harnesses || []).map(
        (harness) => familyLabel(harness, snapshot, { catalogKnown }),
    );
}

function joinLabels(labels) {
    if (labels.length <= 1) return labels[0] || '';
    return `${labels.slice(0, -1).join(', ')} and ${labels[labels.length - 1]}`;
}

/**
 * The declaration the wizard posts with the completion payload.
 *
 * `subscriptionsConnected` is a REQUEST ("go look at the daemon"), never an authority: the endpoint
 * re-reads live account state. It is set from what this step OBSERVED, so an unread account facet
 * declares nothing (BIBLE P1: a gap is not a zero).
 */
export function subscriptionDeclaration({ connected = [], skipPresets = false } = {}) {
    return {
        subscriptionsConnected: (connected || []).length > 0,
        skipSubscriptionPresets: Boolean(skipPresets),
    };
}

/**
 * True when the engine's declared model sources (the catalog's `model_sources`) are known and none of
 * the connected accounts is one: these accounts run agents, and no model discovery can change that.
 * An empty list is unknown, never "none".
 */
export function agentsOnlyAccess(connected = [], sources = []) {
    const serving = new Set((sources || []).map((source) => String(source?.credentialHarness || '')));
    return connected.length > 0 && serving.size > 0 && !connected.some((harness) => serving.has(harness));
}

const READ_WORDS = { not_read: 'is not available yet', partial: 'is incomplete', stale: 'is out of date' };

/**
 * What the Accounts step's access row and its API section badge say for the current reads: facts in, words out.
 * `continueBlocked` is the Continue button's own state, so the note never offers a step the button refuses.
 */
export function accessRowState({ connected = [], sources = [], read = 'ok', checking = false, readErrors = '',
    previewError = '', configured = false, continueBlocked = false } = {}) {
    const agentsOnly = !configured && agentsOnlyAccess(connected, sources);
    const note = [checking ? 'Checking model sources…' : '',
        !checking && read !== 'ok' && connected.length
            ? `The model list ${READ_WORDS[read] || 'could not be read'}. ${readErrors} Your choices are kept.` : '',
        previewError ? `Automatic assignments could not be prepared: ${previewError} `
            + (continueBlocked ? 'Retry.' : 'Retry, or Continue to choose models yourself.') : ''];
    return {
        note: note.filter(Boolean).join(' '),
        // A healthy read that names the connected accounts agents only leaves discovery nothing to retry.
        retryHidden: (!connected.length || (agentsOnly && read === 'ok')) && !previewError,
        // Accounts the engine names agents only leave this section the way forward, not "Optional"; while the list is
        // unread it stays one of the ways the reason beside Continue names, as with nothing connected.
        badge: configured ? 'Configured' : agentsOnly ? '' : 'Optional',
    };
}

/** Why Accounts cannot continue while nothing connected can serve a model role, with the way forward. */
export function mainSourceReason(connected = [], sources = [], { snapshot, catalogKnown = false } = {}) {
    if (!connected.length) return 'Connect Codex, enter an API key, or choose a local model before continuing.';
    if (!agentsOnlyAccess(connected, sources)) {
        return 'Could not confirm that a connected account can run Ouroboros’s own model. '
            + 'Retry model discovery, add an API key, or choose a local model.';
    }
    const serving = sources.map((source) => String(source.label || source.id)).join(' or ');
    return `${joinLabels(familyLabels(connected, snapshot, { catalogKnown }))} can run agents; `
        + `Ouroboros itself still needs a model. Connect ${serving}, add an API key, or choose a local model.`;
}

/** Open provider/local/model draft shared by preview and final completion. */
export function onboardingSettingsDraft({
    state = {}, providerFields = [], budgetFields = [], modelSlots = [],
    trim = (value) => String(value || '').trim(),
} = {}) {
    const clean = (value) => trim(value);
    return {
        OUROBOROS_CONNECTIONS: state.connections || [],
        ...Object.fromEntries(providerFields.map(
            (field) => [field.settingKey, clean(state[field.stateKey])],
        )),
        // The saved legacy address, so readiness and discovery read the pair the runtime reads;
        // no wizard field edits it, and completion keeps the saved value.
        ...(clean(state.legacyCompatibleBaseUrl) ? { OPENAI_BASE_URL: clean(state.legacyCompatibleBaseUrl) } : {}),
        ...Object.fromEntries(budgetFields.filter((field) => state[field.stateKey] !== '').map(
            (field) => [field.settingKey, Number(state[field.stateKey] || 0)],
        )),
        OUROBOROS_SKILLS_REPO_PATH: clean(state.skillsRepoPath),
        LOCAL_MODEL_SOURCE: clean(state.localSource),
        LOCAL_MODEL_FILENAME: clean(state.localFilename),
        LOCAL_MODEL_CONTEXT_LENGTH: Number(state.localContextLength || 0),
        LOCAL_MODEL_N_GPU_LAYERS: Number(state.localGpuLayers || 0),
        LOCAL_MODEL_CHAT_FORMAT: clean(state.localChatFormat),
        LOCAL_ROUTING_MODE: clean(state.localSource)
            ? (clean(state.localRoutingMode) || 'cloud') : 'cloud',
        ...Object.fromEntries(modelSlots.map(
            (slot) => [slot.settingKey, clean(state[slot.stateKey])],
        )),
        OUROBOROS_MODEL_ACCOUNTS: state.modelAccounts || {},
        OUROBOROS_MODEL_CONTEXT_WINDOWS: state.modelContextWindows || {},
        ...(state.processingPreference !== undefined ? { [PROCESSING_PREFERENCE_KEY]: clean(state.processingPreference) } : {}),
        ...(state.modelProcessingPreferences !== undefined ? { [MODEL_PROCESSING_PREFERENCES_KEY]: state.modelProcessingPreferences } : {}),
        OUROBOROS_RUNTIME_MODE: clean(state.runtimeMode) || 'advanced',
    };
}

export const UNCONFIRMED_COMPLETION_TEXT =
    'Onboarding was not confirmed: the server answered without the completion receipt.';

export const MALFORMED_RECEIPT_CODE = 'malformed_completion_receipt';
export const COMPLETION_REPLY_LOST_CODE = 'completion_reply_lost';

// What "Check status" says about the attempt it read back (docs/CLIENT_CONTRACT.md §4).
export const SAVE_NOT_CONFIRMED_TEXT = 'This save is not confirmed. Your draft remains here. '
    + 'You can check again or choose Retry save.';
export const SAVE_STORED_TEXT = 'This save is stored. Setup completion and restart requirements are not yet confirmed.';
// A retry refused before writing anything, while an earlier save's outcome was still open.
export const EARLIER_SAVE_KEPT_TEXT = 'This retry was not saved. The earlier save may still be finishing; Check status reads that save.';
// That earlier save is confirmed, but the owner changed the values since it was sent.
export const EARLIER_SAVE_OTHER_VALUES_TEXT = 'The earlier save is confirmed, but it does not contain the values shown now. '
    + 'Start Ouroboros to save them.';

/**
 * How the wizard READS the completion endpoint's answer. Pure, so every branch is asserted without a transport.
 *
 * A 2xx is NOT a completion by itself: the success envelope carries the saved runtime mode and whether a restart is
 * needed, so a shape-blind `ok` would announce a finished setup while discarding them. A body that will not parse is
 * UNKNOWN, not empty — substituting `{}` made a 200 carrying HTML (a proxy error page, a login redirect) look like
 * success, because `{}` is truthy. Neither that body nor an error page the endpoint did not write says whether the
 * save landed: both read as `saved: null`, which only this attempt's read-back settles.
 *
 * @returns {{receipt: object}|{failure: object}}
 */
export function readCompletionAnswer({ status = 0, ok = false, parsed = false, data = null } = {}) {
    const body = data && typeof data === 'object' ? data : {};
    if (!ok) {
        // TYPED refusals (daemon unreachable, a seat no live model id satisfies,
        // an install that stopped being first-run) keep their code, the engine's
        // own sentence, the skip offer, and whether bytes reached disk.
        return {
            failure: {
                message: String(body.error || `HTTP ${status}`),
                status,
                code: String(body.code || ''),
                detail: String(body.detail || ''),
                canSkip: Boolean(body.can_skip),
                // Tri-state, as the server sends it: `null` is the typed 503
                // `settings_save_timeout` — the save body kept running in the
                // server past its bound, so whether the bytes landed is
                // genuinely UNKNOWN. Collapsing it to `false` would offer a
                // blind re-save over a transaction that may already be on disk.
                saved: parsed && typeof body.saved === 'boolean' ? body.saved : null,
                // `post_commit_failed` is what the server actually sends (see
                // the gateway contract); `stage` was a name this reader made up,
                // so a real post-commit envelope parsed to '' and the owner lost
                // the one word saying WHICH step failed after their settings had
                // already landed. Both spellings are read so a stored or proxied
                // older body is not silently dropped.
                stage: String(body.post_commit_failed || body.stage || ''),
            },
        };
    }
    if (!parsed
        || body.ok !== true
        || typeof body.runtime_mode !== 'string'
        || typeof body.restart_required !== 'boolean') {
        return {
            failure: {
                message: UNCONFIRMED_COMPLETION_TEXT,
                status,
                code: MALFORMED_RECEIPT_CODE,
                detail: '',
                canSkip: false,
                saved: null,
                stage: '',
            },
        };
    }
    return { receipt: body };
}

/** A fresh id for ONE explicit save, made before sending so a lost answer can still be asked about. */
export function completionAttemptId(cryptoImpl = globalThis.crypto) {
    const bytes = cryptoImpl.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/** POST the completion and read its answer; a request that got no answer at all is unknown, too. */
export async function postCompletion(endpoint, payload, fetchImpl = fetch) {
    let response;
    try {
        response = await fetchImpl(endpoint, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        });
    } catch (error) {
        return { failure: { message: `The answer to this save did not arrive (${String(error?.message || error)}).`,
            status: 0, code: COMPLETION_REPLY_LOST_CODE, detail: '', canSkip: false, saved: null, stage: '' } };
    }
    let data = null;
    try { data = await response.json(); } catch { data = null; }
    const parsed = Boolean(data) && typeof data === 'object';
    return readCompletionAnswer({ status: response.status, ok: response.ok, parsed, data: parsed ? data : {} });
}

/**
 * What became of ONE attempt (`OnboardingAttemptResponse`), as the wizard acts on it: only its own
 * receipt completes; a recorded failure is shown like the POST answer it was; stored and unconfirmed
 * keep the draft. Readiness is never asked: a provider saved earlier already passes it.
 */
export function completionAttemptOutcome(data) {
    const body = data && typeof data === 'object' ? data : {};
    const receipt = body.saved === true && body.receipt
        ? readCompletionAnswer({ status: 200, ok: true, parsed: true, data: body.receipt }).receipt : null;
    if (receipt) return { state: 'completed', receipt };
    if (body.saved === false || (body.saved === true && body.post_commit_failed)) {
        return { state: 'failed', failure: { message: String(body.error || 'The save did not complete.'), status: 200,
            code: String(body.code || ''), detail: '', canSkip: false, saved: body.saved, stage: String(body.post_commit_failed || '') } };
    }
    return body.saved === true ? { state: 'stored', text: SAVE_STORED_TEXT } : { state: 'unconfirmed', text: SAVE_NOT_CONFIRMED_TEXT };
}

/** Read back one attempt by its id; an unreadable answer stays unknown and claims nothing. */
export async function readCompletionAttempt(attemptId, fetchImpl = fetch) {
    let response;
    let data = null;
    try {
        response = await fetchImpl(`/api/onboarding/complete?attempt=${encodeURIComponent(attemptId)}`,
            { cache: 'no-store', headers: { Accept: 'application/json' } });
        data = await response.json();
    } catch (error) {
        if (!response) return { state: 'unreadable', text: `Could not check the save status: ${String(error?.message || error)}. Try again in a moment.` };
    }
    if (!response.ok || !data || typeof data !== 'object') {
        return { state: 'unreadable', text: `Could not check the save status: ${String(data?.error || `HTTP ${response.status}`)}. Try again in a moment.` };
    }
    return completionAttemptOutcome(data);
}

export function completionFailureNotice(error) {
    // What the wizard SHOWS when completion did not succeed, and whether the finish-without-agent-defaults escape is
    // genuinely on offer. A typed refusal (engine unreachable, no seat satisfying a live model id, an install that
    // stopped being first-run) carries its own sentence in `detail`, and collapsing it into one generic "failed to
    // save" would leave the owner with a wizard that will not finish and no way to know why.
    //
    // `saved` is read HONESTLY: a refusal persisted nothing and a post-commit failure did persist, so the escape hatch
    // is offered only when nothing is on disk. `saved === null` is the third state (503 `settings_save_timeout`, or an
    // answer that never arrived or was not the endpoint's): the wizard offers "Check status" there rather than a retry — a second write over an unknown first is what it forbids.
    const message = String(error?.message || error || 'Failed to save onboarding settings.');
    const detail = String(error?.detail || '').trim();
    const saved = error?.saved === null ? null : Boolean(error?.saved);
    const parts = [message];
    if (detail) parts.push(detail);
    if (saved) {
        parts.push('Your settings WERE written'
            + `${error?.stage ? ` — the step that failed afterwards was ${String(error.stage)}` : ''}`
            + '. Restart Ouroboros to pick them up rather than filling this in again.');
    } else if (saved === null) {
        parts.push('The save result is unknown. Your draft has not been discarded. '
            + 'Use "Check status" to read the result of this save.');
    }
    return {
        code: String(error?.code || ''),
        saved,
        saveUnknown: saved === null,
        canSkip: Boolean(error?.canSkip) && saved === false,
        text: parts.join(' — '),
    };
}

export function agentsOutcomeText(connected = [], {
    accountsKnown = true,
    accountsUnsupported = false,
    catalogKnown = false,
    skipPresets = false,
    snapshot,
} = {}) {
    // What the owner is told BEFORE finishing. Every verb is conditional,
    // because the compiler can still refuse a seat no live model id satisfies.
    if (accountsUnsupported) {
        return 'Subscriptions can’t be connected until a newer Claudexor runs. Your draft is kept; '
            + 'configure API access to continue.';
    }
    if (!accountsKnown) {
        return 'Accounts could not be checked. Your draft is kept; refresh the connection '
            + 'or configure API access to continue.';
    }
    // Nothing connected: the rows and the reason beside Continue already say what to do.
    if (!connected.length) return '';
    const labels = joinLabels(familyLabels(connected, snapshot, { catalogKnown }));
    if (skipPresets) {
        return `Signed in to ${labels}, but you chose to `
            + 'skip the automatic subscription preset. The Available subagents draft below '
            + `is still saved exactly as you edit it; other agent defaults stay unchanged. ${SIGN_IN_SCOPE_NOTE}`;
    }
    const reviewerHarnesses = connected.filter(
        (harness) => REVIEW_CAPABLE_AGENT_HARNESSES.has(harness),
    );
    const taskOnlyHarnesses = connected.filter(
        (harness) => !REVIEW_CAPABLE_AGENT_HARNESSES.has(harness),
    );
    const clauses = [
        `Signed in to ${labels}. When you finish, `
            + 'Ouroboros will try to add subscription-backed choices to Available subagents '
            + `through ${connected.length === 1 ? 'it' : 'them'}.`,
    ];
    if (reviewerHarnesses.length) {
        const reviewerLabels = familyLabels(reviewerHarnesses, snapshot, { catalogKnown });
        clauses.push(`${joinLabels(reviewerLabels)} can `
            + 'also move commit, plan, skill and result review.');
    }
    if (taskOnlyHarnesses.length) {
        const taskOnlyLabels = familyLabels(taskOnlyHarnesses, snapshot, { catalogKnown });
        clauses.push(`${joinLabels(taskOnlyLabels)} `
            + `${taskOnlyHarnesses.length === 1 ? 'is task-only and does' : 'are task-only and do'} `
            + 'not change reviewer routes.');
    }
    // What Main still needs is said once, beside Continue, and only while it is true (`mainSourceReason`).
    if (connected.includes('codex')) clauses.push('Codex can also supply your model roles without an API key.');
    clauses.push(SIGN_IN_SCOPE_NOTE);
    return clauses.join(' ');
}

export function familyStatusText(snapshot, harness, { accountsKnown = true, accountsUnsupported = false } = {}) {
    if (accountsUnsupported) return { tone: 'muted', text: ACCOUNTS_UNSUPPORTED_STATUS };
    if (!accountsKnown) return { tone: 'muted', text: 'Not checked' };
    const status = familyStatus(accountRows(snapshot).filter((row) => row.harness === harness));
    return { tone: status.tone, text: status.label };
}

export function subagentPreviewStatusSignature(view, snapshot) {
    return JSON.stringify([
        view?.reads?.catalog || '',
        view?.reads?.accounts || '',
        (snapshot?.harnesses || []).map((harness) => [
            String(harness?.id || ''),
            String(harness?.models_error || ''),
            (harness?.models || []).map((model) => String(model?.id || model?.value || model || '')),
        ]),
        accountRows(snapshot).map((row) => [
            String(row?.harness || ''),
            String(row?.profileId || row?.profile_id || row?.id || ''),
            row?.enabled !== false,
            String(row?.status?.verification || ''),
        ]),
    ]);
}

// ---------------------------------------------------------------------------
// The rotation diagram: ONE static SVG, no script, no animation.
// ---------------------------------------------------------------------------

// Ids are page-unique by prefix: a duplicate marker id silently repaints someone else's arrows.
const ARROW_ID = 'ouro-agents-rotation-arrow';

export function rotationDiagramSvg() {
    // Top to bottom in one glance: the API/local source runs the main agent, the stack of agent accounts runs subagents (and
    // eligible plans run reviews), and both feed the one Ouroboros in the middle. The cycle glyph beside the stack plus its
    // caption say what rotation MEANS — "one window spent, the next takes over". Stacked rather than side-by-side because the
    // figure lives in a ~360px column (a wide composition scaled its labels past reading); the space is authored at roughly the
    // rendered width, so 12px CSS text lands at 12px on screen, and strokes are `currentColor`, the accent spent once on the point.
    return `
<svg class="agents-rotation-svg" viewBox="0 0 360 200" role="presentation" aria-hidden="true" focusable="false">
  <defs>
    <marker id="${ARROW_ID}" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path class="agents-rotation-fill" d="M0 1 L7 4 L0 7 Z"></path>
    </marker>
  </defs>
  <g class="agents-rotation-ink">
    <rect x="2" y="0" width="356" height="38" rx="9"></rect>
    <text x="16" y="17">API key or local model</text>
    <text x="16" y="32" class="agents-rotation-sub">runs the main agent</text>

    <path d="M180 40 V58" marker-end="url(#${ARROW_ID})"></path>

    <rect x="2" y="62" width="356" height="40" rx="11" class="agents-rotation-core"></rect>
    <text x="180" y="80" text-anchor="middle" class="agents-rotation-core-label">Ouroboros</text>
    <text x="180" y="95" text-anchor="middle" class="agents-rotation-sub">one agent, one budget</text>

    <path d="M180 122 V104" marker-end="url(#${ARROW_ID})"></path>

    <rect x="2" y="124" width="356" height="76" rx="9" class="agents-rotation-group"></rect>
    <text x="16" y="142">Agent plans</text>
    <text x="16" y="157" class="agents-rotation-sub">run subagents · eligible plans run reviews</text>
    <rect x="16" y="163" width="102" height="15" rx="5"></rect>
    <text x="24" y="175" class="agents-rotation-sub">Account 1</text>
    <rect x="16" y="181" width="102" height="15" rx="5"></rect>
    <text x="24" y="193" class="agents-rotation-sub">Account 2</text>

    <path d="M146 167 a13 13 0 1 1 -10 21" marker-end="url(#${ARROW_ID})"></path>

    <text x="182" y="175" class="agents-rotation-sub">one window spent —</text>
    <text x="182" y="191" class="agents-rotation-sub">the next takes over</text>
  </g>
</svg>`;
}

// ---------------------------------------------------------------------------
// View.
// ---------------------------------------------------------------------------

// A short viewport drops the ARTWORK (web/onboarding.css), whose every fact is
// in this text anyway; the ladder's own words stay at every size.
export function ladderHtml() {
    const rungs = VALUE_LADDER.map((rung) => `
        <li class="agent-rung">
            <span class="agent-rung-tone">${escapeHtml(rung.tone)}</span>
            <span class="agent-rung-text">
                <span class="agent-rung-title">${escapeHtml(rung.title)}</span>
                <span class="agent-rung-body">${escapeHtml(rung.body)}</span>
            </span>
        </li>
    `).join('');
    return `
        <div class="agent-ladder-layout">
            <ol class="agent-ladder">${rungs}</ol>
            <figure class="agents-rotation-figure">${rotationDiagramSvg()}</figure>
        </div>
        <div class="agent-ladder-note">${escapeHtml(LADDER_FOOTNOTE)}</div>
    `;
}

export function familyRowHtml(family, { status, connected = false, connectable = true, program = '', missing = null, profile = '', choices = null, picked = '', newAccount = false }) {
    // Equivalent rows (docs/DESIGN.md §6): only status text and action differ; a missing program is said and its press installs it first.
    // `choices` is the explicit pick of an existing saved account (§7): no profile is guessed onto the one press, a switched-off account
    // is listed as a disabled option — never used or re-enabled — and an empty choice cannot act; `newAccount` is the row's one press
    // that starts a NEW unnamed sign-in instead. A fresh row starts on the placeholder, never on the first option.
    const absent = Boolean(missing) && !connected;
    const said = absent ? { tone: 'muted', text: 'Not installed where Claudexor runs' } : status;
    const action = connected || newAccount ? 'Add another' : absent ? (missing.installable ? 'Install & connect' : '') : 'Connect';
    const pick = choices?.some((choice) => choice.id === picked && !choice.disabled) ? picked : '';
    const accountSelect = choices?.length ? `<select class="ui-control agent-account-choice" data-agent-account aria-label="Account">`
        + `${pick ? '' : '<option value="" disabled selected>Account…</option>'}`
        + `${choices.map((choice) => `<option value="${escapeHtml(choice.id)}"${choice.disabled ? ' disabled' : ''}${choice.id === pick ? ' selected' : ''}>${escapeHtml(choice.id)}</option>`).join('')}</select>` : '';
    return `
        <div class="agent-family-row" data-agent-family="${escapeHtml(family.harness)}">
            <span class="agent-family-identity">
                <span class="agent-family-heading">
                ${harnessIdentityMarkup(family.harness, { label: family.label, className: 'agent-family-name' })}
                <span class="agent-family-purpose">${family.harness === 'codex' ? 'Models + agents · Recommended' : 'Agents'}</span>
                </span>
                <span class="agent-family-status" data-tone="${escapeHtml(said.tone)}">${escapeHtml(said.text)}</span>
                ${program}
            </span>
            ${connectable && action ? `${accountSelect}<button type="button" class="btn btn-default"${accountSelect && !pick && !newAccount ? ' disabled' : ''} data-agent-connect="${escapeHtml(family.harness)}"${newAccount ? ' data-agent-new' : ''}${!connected && profile ? ` data-agent-profile="${escapeHtml(profile)}"` : ''}>
                ${escapeHtml(action)}
            </button>` : ''}
        </div>
    `;
}

export function familyListHtml(snapshot, {
    accountsKnown = true,
    accountsUnsupported = false,
    catalogKnown = false,
    programLine = () => '',
    programMissing = () => null,
    others = '',  // the engine's other programs it lacks, each installed by its own press (model_program.js `others`)
    chosen = {},  // the account each family's row is standing on (§7), so a rebuild re-renders the same pick
} = {}) {
    const connected = new Set(accountsKnown ? connectedHarnesses(snapshot) : []);
    return AGENT_FAMILIES.map((family) => {
        const own = accountRows(snapshot).filter((row) => row.harness === family.harness), missing = programMissing(family.harness);
        const connectedFamily = connected.has(family.harness);
        // The one press may carry a named profile only from a CURRENT roster read, and only for a
        // single account the owner has not switched off: a remembered row is no one's choice, and a
        // disabled account is installed for, never silently signed back in (§7). Elsewhere the saved
        // accounts become an explicit choice; a missing program installs from the row's press only
        // where that single usable account (or a read empty) leaves nothing to choose between.
        // The shortcut is NATURAL only on a fresh row: an answer the owner gave that this roster no
        // longer offers must not turn the sole survivor into the press's silent profile — the survivor
        // renders as that explicit choice (placeholder until the owner names it), never as a substitute.
        const remembered = String(chosen?.[family.harness] || '');
        const sole = own.length === 1 ? String(own[0].profile_id || '') : '';
        const profile = accountsKnown && own.length === 1 && own[0].enabled !== false
            && (!remembered || remembered === sole) ? sole : '';
        const row = Boolean(missing) && !connectedFamily && accountsKnown && own.length <= 1 && (own.length === 0 || profile);
        // The picker names every saved account, the switched-off ones among them as disabled options (§7).
        const choices = accountsKnown && !connectedFamily && !profile && own.length && !missing
            ? own.map((candidate) => ({ id: String(candidate.profile_id || ''), disabled: candidate.enabled === false }))
                .filter((choice) => choice.id) : null;
        // A remembered pick survives only while the CURRENT roster still offers it ENABLED — never a removed or switched-off stand-in.
        const picked = choices?.some((choice) => choice.id === chosen?.[family.harness] && !choice.disabled)
            ? chosen[family.harness] : '';
        // Nothing enabled left to choose with the program installed: the ordinary Add another is the separate step to a NEW account (§7).
        const newAccount = Boolean(choices) && !choices.some((choice) => !choice.disabled);
        // The press is withheld for an absent program the engine cannot install, and while the program is missing
        // with saved accounts but no single usable default: the line installs only, and a Connect there would
        // start an unnamed second account (§7). A connected family keeps its Add another — its sign-in proved the program runs.
        const pressable = !accountsUnsupported && !(missing && !missing.installable)
            && !(Boolean(missing) && !connectedFamily && !row && own.length > 0);
        return familyRowHtml({ ...family, label: familyLabel(family.harness, snapshot, { catalogKnown }) }, { connectable: pressable,
            status: familyStatusText(snapshot, family.harness, { accountsKnown, accountsUnsupported }), connected: connectedFamily,
            program: programLine(family.harness, { install: Boolean(missing) && !row }),
            missing: row ? missing : null, profile, choices, picked, newAccount,
        });
    }).join('') + others;
}

export function agentsStepHtml({ compact = false, showRoster = true } = {}) {
    // The static skeleton. Everything that moves (family rows, the service note, the outcome
    // sentence) is patched in place afterwards, so a status tick never rebuilds the login card.
    return `
        ${compact ? '' : `<div class="panel-card agent-ladder-card">
            <h3>What an agent plan changes</h3>
            ${ladderHtml()}
        </div>`}
        <div class="panel-card" id="agents-accounts">
            <h3>Subscriptions</h3>
            <div id="agents-status-note" class="agent-service-note" hidden></div>
            <div id="agents-family-list" class="agent-family-list"></div>
            <div class="wizard-inline-note agent-limits-note">Subscription limits still apply. This connection does not enable paid credits or change your provider's spending limits.</div>
            <div id="agents-login-host"></div>
            <div id="agents-outcome" class="agent-outcome"></div>
        </div>
        ${showRoster ? `<div class="panel-card" id="agents-available-subagents-card">
            <h3>Available subagents</h3>
            <div class="agent-ladder-note">
                This generated draft is what Ouroboros will see. Describe when each numbered
                subagent is useful, then adjust its route, model, effort, or account pin if needed.
            </div>
            ${availableSubagentsEditorHost('onboarding-available-subagents')}
        </div>` : ''}
    `;
}

// ---------------------------------------------------------------------------
// The controller.
// ---------------------------------------------------------------------------

/**
 * @param {object} options
 * @param {Function} [options.doc]        `document`, or a getter for it
 * @param {object}   [options.store]      the shared Claudexor status store
 * @param {Function} [options.fetchImpl]  transport for the login card
 * @param {Function} [options.isVisible]  is the Agents step on screen right now
 * @param {Function} [options.onChange]   called with the connected harness list
 * @param {Function} [options.previewPayload] current open provider/local draft
 * @param {Function} [options.previewTransport] injectable preview request
 * @param {Function} [options.onSubagentsChange] receives the editable canonical list
 * @param {object}   [options.providerProfiles] setup-contract provider names
 * @returns {object} controller
 */
export function createAgentsStep({
    doc = () => (typeof document === 'undefined' ? null : document),
    store = claudexorStatus,
    fetchImpl = null,
    isVisible = () => true,
    onChange = () => {},
    previewPayload = () => ({}),
    previewTransport = (payload) => apiClient.previewOnboardingSubagents(payload),
    onSubagentsChange = () => {}, readApiCatalog, apiCatalogAllowed, supportsScopes,
    providerProfiles = {},
    onSetupPreview = () => {},
    onPreviewStatus = () => {},
    onStatus = () => {},
    maintenanceApi = apiClient,
} = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    const state = {
        connected: [],
        skipPresets: false,
        disposed: false,
        unsubscribe: null,
        pageHideBound: null,
        pageHideTarget: null,
        listHtml: null,
        previewGeneration: 0,
        accountChoice: {},
        previewAppliedSignature: '',
        previewPending: false,
        previewFailure: null,
        previewStatusSignature: '',
    };

    function el(id) {
        return getDoc()?.getElementById(id) || null;
    }

    function accountsKnown() {
        return Boolean(store.accountsKnown);
    }

    // FULL, not compact (phase 2 disclosed the gap): compact omits the paste-code entry and the
    // collapsed terminal fallback, so a Claude login whose localhost callback cannot complete —
    // exactly the remote/headless first run — would have no way to finish. Onboarding mounts the
    // same card Settings mounts.
    function createLogin() {
        return createLoginCardController({
            host: () => el('agents-login-host'),
            store,
            mode: LOGIN_CARD_FULL,
            doc: getDoc,
            ...(fetchImpl ? { fetchImpl } : {}),
            ...programs.loginHooks(),
            // The rows' own decision (§7), honoured inside the card too — an open name face's submit included.
            signInWithheld: (harness) => {
                const missing = programs.missing(harness);
                if (missing && !missing.installable) {
                    return `${familyLabel(harness, store.snapshot || {}, { catalogKnown: Boolean(store.catalogKnown) })} is not installed where Claudexor runs and cannot be installed here.`;
                }
                // The competing-install fence lives HERE, at the one boundary every sign-in of this
                // step passes (Connect, Add another, the name form, a retry): while this family's
                // install is unresolved and its program is not observed installed, a new create would
                // orphan one of the two jobs.
                if (programs.competingInstall?.(harness)) {
                    return `${familyLabel(harness, store.snapshot || {}, { catalogKnown: Boolean(store.catalogKnown) })} is still being installed. Connect once that install finishes.`;
                }
                return '';
            },
            refuseDisabled: true,
            onSettled: () => { store.refresh(); void programs.read(); paint(); },
        });
    }

    // Each family's program line (docs/DESIGN.md §7), the same one an opened model chooser shows.
    const programs = bindFamilyPrograms({ harnesses: AGENT_FAMILIES.map((f) => f.harness), store, api: maintenanceApi,
        visible: () => isVisible() && Boolean(el('agents-family-list')), onChange: () => paint() });
    let login = createLogin();
    const subagents = createAvailableSubagentsEditor({
        hostId: 'onboarding-available-subagents',
        doc: getDoc,
        win: () => getDoc()?.defaultView,
        store,
        onChange: onSubagentsChange,
        readApiCatalog, apiCatalogAllowed, supportsScopes,
        baseline: 'generated',
    });

    function previewRequest({ includeVisibleRoster = false } = {}) {
        return {
            ...(previewPayload() || {}),
            ...(subagents.dirty || (includeVisibleRoster && subagents.loaded) ? { OUROBOROS_SUBAGENTS: subagents.setting } : {}),
            ...subscriptionDeclaration({
                connected: state.connected,
                skipPresets: state.skipPresets,
            }),
        };
    }

    function currentPreviewSignature() {
        return JSON.stringify(previewRequest());
    }

    async function refreshSubagentsPreview({ force = false, replaceReviewers = false } = {}) {
        if (state.disposed) return false;
        const payload = previewRequest({ includeVisibleRoster: replaceReviewers });
        const signature = JSON.stringify(payload);
        if (!force && signature === state.previewAppliedSignature && subagents.loaded) return true;
        state.previewPending = true;
        state.previewFailure = null;
        const generation = ++state.previewGeneration;
        onPreviewStatus();
        try {
            const response = await previewTransport(payload);
            if (state.disposed || generation !== state.previewGeneration) return false;
            const result = replaceReviewers ? subagents.applyOwnerPreview(response)
                : subagents.dirty ? { applied: true } : subagents.applyGeneratedPreview(response);
            if (!result.applied) {
                state.previewFailure = { detail: result.error || 'Available subagents preview was not applied.' };
                return false;
            }
            state.previewAppliedSignature = signature;
            onSubagentsChange(subagents.setting);
            onSetupPreview(response, { replaceReviewers });
            state.previewAppliedSignature = currentPreviewSignature();
            return true;
        } catch (error) {
            if (state.disposed || generation !== state.previewGeneration) return false;
            state.previewFailure = { code: error?.body?.code || '', canSkip: Boolean(error?.body?.can_skip),
                detail: String(error?.body?.detail || error?.message || error) };
            subagents.setPreviewFailure(error);
            return false;
        } finally {
            if (generation === state.previewGeneration) {
                state.previewPending = false;
                if (!state.disposed) onPreviewStatus();
            }
        }
    }

    function invalidateGeneratedPreview() {
        state.previewGeneration += 1;
        state.previewAppliedSignature = '';
        state.previewPending = false;
        state.previewFailure = null;
        onPreviewStatus();
    }

    function ensureLogin() {
        if (!login || login.disposed) login = createLogin();
        return login;
    }

    function paint() {
        if (state.disposed) return;
        const snapshot = store.snapshot;
        const known = accountsKnown();
        const unsupported = Boolean(store.accountsUnsupported);
        const list = el('agents-family-list');
        if (list) {
            // The standing rows hold the owner's own account picks: read them BEFORE the list is replaced, so a later rebuild renders that same account (§7).
            list.querySelectorAll('[data-agent-account]').forEach((select) => {
                const row = select.closest?.('[data-agent-family]');
                if (row && select.value) state.accountChoice[row.getAttribute('data-agent-family')] = select.value;
            });
            // Rebuild ONLY on a real change: the store settles every few seconds, and replacing unchanged rows is churn.
            const html = familyListHtml(snapshot, {
                accountsKnown: known,
                accountsUnsupported: unsupported,
                catalogKnown: Boolean(store.catalogKnown),
                programLine: programs.line,
                programMissing: programs.missing,
                others: programs.others(),
                chosen: state.accountChoice,
            });
            if (html !== state.listHtml) {
                state.listHtml = html;
                rebuildList(list, html);
            }
        }
        const note = el('agents-status-note');
        if (note) {
            const unavailable = store.unavailableNote('accounts');
            note.hidden = !unavailable;
            note.textContent = unavailable ? unavailable.text : '';
            if (unavailable) note.dataset.tone = unavailable.tone;
        }
        const outcome = el('agents-outcome');
        if (outcome) {
            outcome.textContent = agentsOutcomeText(state.connected, {
                accountsKnown: known,
                accountsUnsupported: unsupported,
                catalogKnown: Boolean(store.catalogKnown),
                skipPresets: state.skipPresets,
                snapshot,
            });
        }
        login?.render();
    }

    // A rebuild keeps the focused control of the same family (docs/DESIGN.md principle 6).
    function rebuildList(list, html) {
        const active = getDoc()?.activeElement;
        const control = list.contains(active) && ['data-agent-connect', 'data-program-action', 'data-agent-account'].find((name) => active.hasAttribute(name));
        const home = control && active.closest('[data-program], [data-agent-family], [data-programs-unknown]');
        const where = home && ['program', 'agentFamily'].map((key) => home.dataset[key] && `[data-${key === 'program' ? 'program' : 'agent-family'}="${home.dataset[key]}"]`).find(Boolean);
        list.innerHTML = html;
        // Naming the account opens the row's own press, and the answer is remembered the moment it is given: a
        // rebuild that replaces the step's DOM cannot read the standing row, so the pick would fall back to
        // whatever an earlier paint saved; the new-account press is never the named pick's gate (§7).
        list.querySelectorAll('[data-agent-account]').forEach((select) => select.addEventListener('change', () => {
            const row = select.closest?.('[data-agent-family]');
            if (row) state.accountChoice[row.getAttribute('data-agent-family')] = select.value;
            const press = row?.querySelector('[data-agent-connect]');
            if (press && !press.hasAttribute('data-agent-new')) press.disabled = !select.value;
        }));
        list.querySelectorAll('[data-agent-connect]').forEach((button) => button.addEventListener('click', () => {
            if (state.disposed) return;
            // The competing-install fence is the card's own `signInWithheld` boundary above, so
            // every press of this row (Connect, Add another) meets it and none can bypass it.
            // An explicit account choice names the profile the press starts with; the remembered single
            // default (data-agent-profile) is the other path. No choice and no default ever starts an unnamed
            // login — the press waits; the separate new-account press is the one unnamed sign-in this row offers.
            const pick = button.closest?.('[data-agent-family]')?.querySelector('[data-agent-account]')?.value;
            if (pick === '' && !button.hasAttribute('data-agent-new')) return;
            ensureLogin().start(button.getAttribute('data-agent-connect'), pick || button.dataset?.agentProfile || '');
        }));
        programs.bind(list);
        if (home) list.querySelector(control === 'data-agent-account' && where
            ? `${where} [data-agent-account]` : `${where || '[data-programs-unknown]'} [${control}="${active.getAttribute(control)}"]`)?.focus({ preventScroll: true });
    }

    function adopt(view) {
        const next = view && view.reads && view.reads.accounts === 'ok'
            ? connectedHarnesses(store.snapshot)
            : [];
        const changed = next.join(',') !== state.connected.join(',');
        const statusSignature = subagentPreviewStatusSignature(view, store.snapshot);
        const statusChanged = statusSignature !== state.previewStatusSignature;
        state.previewStatusSignature = statusSignature;
        state.connected = next;
        programs.observe();
        paint();
        if (changed) onChange([...state.connected]);
        onStatus({ catalogChanged: changed || statusChanged });
        if (changed || statusChanged) refreshSubagentsPreview({ force: statusChanged });
    }

    /** Mount into the step DOM the wizard just rendered. */
    function mount() {
        if (state.disposed) return;
        if (!state.unsubscribe) {
            state.unsubscribe = store.subscribe(adopt, { visible: isVisible });
            const pageHideTarget = getDoc()?.defaultView;
            if (pageHideTarget?.addEventListener) {
                // The wizard never unmounts itself, so this ONE listener is released on a real
                // unload as well as by dispose(). A bfcache hide (`persisted`) is NOT an unload:
                // the page can come back, and tearing down here would restore a wizard whose
                // Agents step no longer reads anything.
                state.pageHideTarget = pageHideTarget;
                state.pageHideBound = (event) => {
                    if (event?.persisted !== true) detach();
                };
                pageHideTarget.addEventListener('pagehide', state.pageHideBound);
            }
        }
        // A remount rebuilds the DOM the wizard just replaced.
        state.listHtml = null;
        programs.observe({ remount: true });
        paint();
        subagents.mount();
        // The wizard's API keys are typed on the Accounts step, so the provider list is re-derived
        // on every entry into a step that shows these rows — never at construction, when no key exists yet.
        applySourceContext();
        refreshSubagentsPreview();
        store.refresh();
    }

    /** The roster editor offers the providers the current draft has keys for. */
    function applySourceContext() {
        subagents.setSourceContext({
            settings: previewPayload() || {},
            providerProfiles,
        });
    }

    /**
     * Ask the login controller for its exact custody result: cleanup and local departure are deliberately separate,
     * `retained` means another cancel is pointless, and `unknown` stays retryable within the wizard's bounded window.
     *
     * @returns {Promise<'released'|'retained'|'unknown'>}
     */
    async function dispose() {
        if (state.disposed) return 'unknown';
        // The controller remains the custody SSOT even after its own recovery-face Close:
        // repeated dispose returns its remembered honest detach status without another request.
        return login ? login.dispose() : 'released';
    }

    /** Resolve sign-in custody as far as the bounded completion window proves. */
    async function disposeForCompletion() {
        // A known retained job cannot be improved by repeating cancel; only an unknown transport
        // result remains retryable. Local detach still makes no claim the daemon released custody.
        let custody = await dispose();
        for (let attempt = 0; custody === 'unknown' && attempt < LOGIN_RELEASE_RETRIES; attempt += 1) {
            await new Promise((resolve) => setTimeout(resolve, LOGIN_RELEASE_RETRY_MS));
            custody = await dispose();
        }
        if (custody === 'retained') {
            console.warn('onboarding: the agent engine could not confirm that the sign-in process stopped; '
                + 'the next Connect will return to that attempt.');
        } else if (custody === 'unknown') {
            console.warn('onboarding: sign-in cleanup remained unknown after '
                + `${LOGIN_RELEASE_RETRIES + 1} attempts; the agent engine still owns that job `
                + 'and the next Connect will recover its current state.');
        }
        detach();
        return custody;
    }

    /** Leave locally without claiming the daemon released process custody. */
    function detach() {
        if (state.disposed) return;
        state.disposed = true;
        login?.detach();
        subagents.destroy();
        programs.dispose();
        if (state.unsubscribe) state.unsubscribe();
        state.unsubscribe = null;
        if (state.pageHideBound && state.pageHideTarget?.removeEventListener) {
            state.pageHideTarget.removeEventListener('pagehide', state.pageHideBound);
        }
        state.pageHideBound = null;
        state.pageHideTarget = null;
        state.listHtml = null;
        login = null;
    }

    return {
        mount,
        paint,
        dispose,
        disposeForCompletion,
        detach,
        get connected() { return [...state.connected]; },
        // The payload the family names are spoken from. The wizard's review step renders the SAME
        // families one screen later, and without this it fell back to the bootstrap product names —
        // so an engine rename showed up here and silently un-renamed itself in the summary.
        get snapshot() { return store?.snapshot || null; },
        get catalogKnown() { return Boolean(store?.catalogKnown); },
        get accountsKnown() { return accountsKnown(); },
        get reads() { return store.reads; },
        refreshStatus() { return store.refresh(); },
        adoptCatalog: subagents.adoptCatalog, adoptSources: subagents.adoptSources, refreshCatalogs: subagents.refreshCatalogs,
        resetCatalogs() { subagents.resetCatalogs(); },
        get availableSubagents() { return subagents.setting; },
        get allowEmptyReviewPool() { return subagents.allowEmptyReviewPool; },
        setProcessingPreference(value) { subagents.setProcessingPreference(value); },
        /** Re-derive the provider list after the owner edits Accounts. */
        setSourceContext(context) {
            if (context) subagents.setSourceContext(context);
            else applySourceContext();
        },
        get generatedPreviewReady() {
            if (subagents.dirty) return true;
            try {
                return subagents.loaded && !state.previewPending && !state.previewFailure
                    && state.previewAppliedSignature === currentPreviewSignature();
            } catch (error) {
                return false;
            }
        },
        get previewPending() { return state.previewPending; },
        get previewError() { return state.previewFailure?.detail || ''; },
        get previewFailure() { return state.previewFailure; },
        validateSubagents() { return subagents.validate(); },
        // Finish is the wizard's commit: the roster then shows its own errors
        // beside the rows they name when the owner steps back here.
        noteSaveAttempt() { subagents.noteSaveAttempt(); },
        refreshSubagentsPreview,
        invalidateGeneratedPreview,
        setSkipPresets(value, { replaceReviewers = false } = {}) {
            const next = Boolean(value);
            let refreshed = Promise.resolve(true);
            if (next !== state.skipPresets) {
                state.skipPresets = next;
                refreshed = refreshSubagentsPreview({ force: true, replaceReviewers });
            } else if (replaceReviewers || (!subagents.dirty && !state.previewPending)) {
                refreshed = refreshSubagentsPreview({ force: true, replaceReviewers });
            }
            paint();
            return refreshed;
        },
        declaration({ skipPresets = state.skipPresets } = {}) {
            return subscriptionDeclaration({ connected: state.connected, skipPresets });
        },
    };
}
