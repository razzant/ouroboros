// Shared onboarding account connection and install-preview controller.
// Login and account truth remain in the same controller/store as Settings;
// draft compilation is read-only and completion is one atomic transaction.

import { apiClient } from './api_client.js';
import { claudexorStatus, accountRows, familyLabel } from './claudexor_status_store.js';
import { LOGIN_CARD_FULL, createLoginCardController } from './harness_login_cards.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import {
    availableSubagentsEditorHost,
    createAvailableSubagentsEditor,
} from './subagents_settings.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { PROCESSING_PREFERENCE_KEY, MODEL_PROCESSING_PREFERENCES_KEY } from './route_editor_primitives.js';

// Every supported task harness in the linear Available-subagents compiler.
// Reviewer policy remains a separate core-only projection: Agy is task-only and
// must never be omitted here merely because it creates no reviewer seats.
export const AGENT_FAMILIES = [
    { harness: 'codex' },
    { harness: 'claude' },
    { harness: 'cursor' },
    { harness: 'agy' },
];

const REVIEW_CAPABLE_AGENT_HARNESSES = new Set(['claude', 'codex', 'cursor']);

// Completion must not hang on an engine that is down. Three attempts over
// roughly 1.2 seconds convert a transport blip into a proven cancel; anything
// longer stops being a blip.
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
    + 'route: commit, plan, skill review and task acceptance all run on the review pool '
    + '(the rows marked Reviewer), so an all-subscription pool also puts each substantive task\'s '
    + 'acceptance panel on the subscription: on the API it measured about 12 s and '
    + '$0.07 per model row per task; a session spends minutes of your window per task instead.';

// ---------------------------------------------------------------------------
// Pure helpers.
// ---------------------------------------------------------------------------

export function connectedHarnesses(snapshot) {
    // Which preset families the daemon vouches a signed-in account for, in the
    // stable family order. `accountRows` is the SSOT projection of the status
    // payload (native CLI session + verified credential profiles alike), so
    // this step and the accounts panel can never disagree about "connected".
    // An account the owner switched OFF is skipped however healthy its login:
    // rotation never takes it, so counting it here would declare a family
    // connected on a pool the engine will not use. Absent stays connected —
    // the same fail-open `enabled` projection the accounts panel applies.
    const passed = new Set(
        accountRows(snapshot)
            .filter((row) => row?.enabled !== false
                && String(row?.status?.verification || '') === 'passed')
            .map((row) => String(row.harness || '')),
    );
    return AGENT_FAMILIES.map((f) => f.harness).filter((harness) => passed.has(harness));
}

export function harnessAccountCount(snapshot, harness) {
    return accountRows(snapshot).filter(
        (row) => row.harness === harness
            && row?.enabled !== false
            && String(row?.status?.verification || '') === 'passed',
    ).length;
}

export function familyLabels(harnesses, snapshot, { catalogKnown = false } = {}) {
    // ONE authority, shared with the Agents tab: prefer the engine's own
    // display_name, fall back to the bootstrap product names. Passing the
    // snapshot keeps this pure and node-testable like the helpers around it;
    // without it the wizard would print a raw `claude` the day the engine
    // renames a family or adds a fourth.
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
 * `subscriptionsConnected` is a REQUEST ("go look at the daemon"), never an
 * authority: the endpoint re-reads live account state and compiles the preset
 * from live discovery. It is set from what this step actually OBSERVED, so an
 * unread account facet declares nothing (BIBLE P1: a gap is not a zero).
 */
export function subscriptionDeclaration({ connected = [], skipPresets = false } = {}) {
    return {
        subscriptionsConnected: (connected || []).length > 0,
        skipSubscriptionPresets: Boolean(skipPresets),
    };
}

/** Open provider/local/model draft shared by preview and final completion. */
export function onboardingSettingsDraft({
    state = {}, providerFields = [], budgetFields = [], modelSlots = [],
    trim = (value) => String(value || '').trim(),
} = {}) {
    const clean = (value) => trim(value);
    return {
        ...Object.fromEntries(providerFields.map(
            (field) => [field.settingKey, clean(state[field.stateKey])],
        )),
        ...Object.fromEntries(budgetFields.filter((field) => state[field.stateKey] !== '').map(
            (field) => [field.settingKey, Number(state[field.stateKey] || 0)],
        )),
        OUROBOROS_REVIEW_ENFORCEMENT: clean(state.reviewEnforcement) || 'advisory',
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
    'Onboarding was not confirmed: the server answered without the completion '
    + 'receipt. Nothing here can tell whether your settings were saved — reload '
    + 'this page; if setup is still shown, finish it again.';

export const MALFORMED_RECEIPT_CODE = 'malformed_completion_receipt';

/**
 * How the wizard READS the completion endpoint's answer. Pure, so every branch
 * is asserted without a transport.
 *
 * A 2xx is NOT a completion by itself. The success envelope carries the two
 * facts everything downstream needs — the saved runtime mode and whether it
 * needs a restart — so a shape-blind `ok` would announce a finished setup while
 * silently discarding them. A body that will not parse is UNKNOWN, not empty:
 * substituting `{}` made a 200 carrying HTML (a proxy error page, a login
 * redirect) look like success, because `{}` is truthy.
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
                saved: body.saved === null ? null : Boolean(body.saved),
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
                saved: false,
                stage: '',
            },
        };
    }
    return { receipt: body };
}

export function completionFailureNotice(error) {
    // What the wizard SHOWS when completion did not succeed, and whether the
    // finish-without-agent-defaults escape is genuinely on offer.
    //
    // The endpoint answers typed refusals — the daemon is unreachable, a seat
    // no live model id satisfies, an install that stopped being first-run — and
    // each carries the engine's own sentence in `detail`. Collapsing those into
    // one generic "failed to save" leaves the owner holding a wizard that will
    // not finish and no way to know why, which is precisely the state the
    // atomic endpoint was built to avoid.
    //
    // `saved` is read HONESTLY. A refusal persisted nothing; a failure in a
    // post-commit stage did persist, and telling the owner otherwise would
    // repeat one layer up the exact dishonesty the atomic write removed. The
    // escape hatch is only offered when nothing is on disk — once settings
    // exist, "finish without agent defaults" would be a second write, not an
    // alternative to the first.
    //
    // `saved === null` is the third state (503 `settings_save_timeout`): the
    // save is still running in the server, so neither "written" nor "nothing
    // saved" is true yet. The wizard offers "Check status" for that case
    // instead of a retry, which would be a second write over an unknown first.
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
        parts.push('Whether your settings were saved is unknown — the save is still '
            + 'running in the server. Use "Check status" to see whether it landed '
            + 'before finishing setup again.');
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
    catalogKnown = false,
    skipPresets = false,
    snapshot,
} = {}) {
    // What the owner is told BEFORE finishing. Every verb is conditional,
    // because the compiler can still refuse a seat no live model id satisfies.
    if (!accountsKnown) {
        return 'Accounts could not be checked. Your draft is kept; refresh the connection '
            + 'or configure API access to continue.';
    }
    if (!connected.length) {
        return 'Connect Codex to start without an API key, or add API access below. '
            + 'Other subscriptions remain available for agents.';
    }
    const labels = joinLabels(familyLabels(connected, snapshot, { catalogKnown }));
    if (skipPresets) {
        return `${labels} ${connected.length === 1 ? 'is' : 'are'} connected, but you chose to `
            + 'skip the automatic subscription preset. The Available subagents draft below '
            + 'is still saved exactly as you edit it; other agent defaults stay unchanged.';
    }
    const reviewerHarnesses = connected.filter(
        (harness) => REVIEW_CAPABLE_AGENT_HARNESSES.has(harness),
    );
    const taskOnlyHarnesses = connected.filter(
        (harness) => !REVIEW_CAPABLE_AGENT_HARNESSES.has(harness),
    );
    const clauses = [
        `${labels} ${connected.length === 1 ? 'is' : 'are'} connected. When you finish, `
            + 'Ouroboros will try to add subscription-backed choices to Available subagents '
            + `through ${connected.length === 1 ? 'it' : 'them'}.`,
    ];
    if (reviewerHarnesses.length) {
        const reviewerLabels = familyLabels(reviewerHarnesses, snapshot, { catalogKnown });
        clauses.push(`${joinLabels(reviewerLabels)} can `
            + 'also move commit, scope, advisory, plan, skill review, and task acceptance.');
    }
    if (taskOnlyHarnesses.length) {
        const taskOnlyLabels = familyLabels(taskOnlyHarnesses, snapshot, { catalogKnown });
        clauses.push(`${joinLabels(taskOnlyLabels)} `
            + `${taskOnlyHarnesses.length === 1 ? 'is task-only and does' : 'are task-only and do'} `
            + 'not change reviewer routes.');
    }
    clauses.push(connected.includes('codex')
        ? 'Codex can also supply your model roles without an API key.'
        : 'Main still needs Codex, an API key, or a local model.');
    return clauses.join(' ');
}

export function familyStatusText(snapshot, harness, { accountsKnown = true } = {}) {
    if (!accountsKnown) return { tone: 'muted', text: 'Not checked' };
    const count = harnessAccountCount(snapshot, harness);
    if (!count) return { tone: 'muted', text: 'Not connected' };
    return {
        tone: 'ok',
        text: count === 1 ? 'Connected' : `${count} accounts connected · they rotate`,
    };
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

// Ids are page-unique by prefix: the wizard mounts this beside other inline SVG
// one day, and a duplicate marker id silently repaints someone else's arrows.
const ARROW_ID = 'ouro-agents-rotation-arrow';

export function rotationDiagramSvg() {
    // Top to bottom in one glance: the API/local source runs the main agent,
    // the stack of agent accounts runs subagents (and eligible plans run
    // reviews), and both feed the
    // one Ouroboros in the middle. The cycle glyph beside the stack plus its
    // caption say what rotation MEANS — "one window spent, the next takes
    // over". Stacked rather than side-by-side because the figure lives in a
    // ~360px column of the step: a wide composition scaled its labels down to
    // an unreadable size, and readability at a glance is the whole point.
    //
    // The coordinate space is authored at roughly the rendered width, so the
    // 12px CSS text lands at 12px on screen instead of being scaled by the
    // viewBox. Strokes are `currentColor` (the figure inherits the wizard's
    // ink); the accent is spent once, on the thing everything points at.
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

// NOTE ON ELEMENTS: nothing load-bearing here is a `<p>`. The wizard's
// short-viewport rule hides `.panel-card p` wholesale (it is how the flow stays
// unscrolled on a laptop), and the ladder is the one thing on this step that
// must survive that adaptation — what gets hidden instead is the ARTWORK, whose
// every fact is in this text anyway.
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

export function familyRowHtml(family, { status, connected = false }) {
    // Equivalent rows (docs/DESIGN.md §6): no family gets extra weight for being
    // first or already connected — only its status text and its action change.
    return `
        <div class="agent-family-row" data-agent-family="${escapeHtml(family.harness)}">
            <span class="agent-family-identity">
                <span class="agent-family-heading">
                ${harnessIdentityMarkup(family.harness, {
                    label: family.label,
                    className: 'agent-family-name',
                })}
                <span class="agent-family-purpose">${family.harness === 'codex' ? 'Models + agents · Recommended' : 'Agents'}</span>
                </span>
                <span class="agent-family-status" data-tone="${escapeHtml(status.tone)}">${escapeHtml(status.text)}</span>
            </span>
            <button type="button" class="btn btn-secondary" data-agent-connect="${escapeHtml(family.harness)}">
                ${escapeHtml(connected ? 'Add another' : 'Connect')}
            </button>
        </div>
    `;
}

export function familyListHtml(snapshot, {
    accountsKnown = true,
    catalogKnown = false,
} = {}) {
    const connected = new Set(accountsKnown ? connectedHarnesses(snapshot) : []);
    return AGENT_FAMILIES.map((family) => familyRowHtml({
        ...family,
        label: familyLabel(family.harness, snapshot, { catalogKnown }),
    }, {
        status: familyStatusText(snapshot, family.harness, { accountsKnown }),
        connected: connected.has(family.harness),
    })).join('');
}

export function agentsStepHtml({ compact = false, showRoster = true } = {}) {
    // The static skeleton. Everything that moves (family rows, the service
    // note, the outcome sentence) is patched in place afterwards, so a status
    // tick never rebuilds the login card mounted between them.
    return `
        ${compact ? '' : `<div class="panel-card agent-ladder-card">
            <h3>What an agent plan changes</h3>
            ${ladderHtml()}
        </div>`}
        <div class="panel-card" id="agents-accounts">
            <h3>Subscriptions</h3>
            <div id="agents-status-note" class="agent-service-note" hidden></div>
            <div id="agents-family-list" class="agent-family-list"></div>
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
    onSubagentsChange = () => {},
    providerProfiles = {},
    onSetupPreview = () => {},
    onPreviewStatus = () => {},
    onStatus = () => {},
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

    // FULL, not compact (phase 2 disclosed the gap): compact omits the
    // paste-code entry and the collapsed terminal fallback, so a Claude login
    // whose localhost callback cannot complete — the owner's browser on another
    // machine, exactly the remote/headless first run — would have no way to
    // finish. A dead end is worst at first run, so onboarding mounts the same
    // card Settings mounts.
    function createLogin() {
        return createLoginCardController({
            host: () => el('agents-login-host'),
            store,
            mode: LOGIN_CARD_FULL,
            doc: getDoc,
            ...(fetchImpl ? { fetchImpl } : {}),
            onSettled: () => { store.refresh(); paint(); },
        });
    }

    let login = createLogin();
    const subagents = createAvailableSubagentsEditor({
        hostId: 'onboarding-available-subagents',
        doc: getDoc,
        win: () => getDoc()?.defaultView,
        store,
        onChange: onSubagentsChange,
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
        const list = el('agents-family-list');
        if (list) {
            // Rebuild ONLY on a real change. The status store settles every few
            // seconds; replacing three unchanged rows on every tick is pointless
            // churn, and it opens a window where a click lands on a button that
            // is being replaced.
            const html = familyListHtml(snapshot, {
                accountsKnown: known,
                catalogKnown: Boolean(store.catalogKnown),
            });
            if (html !== state.listHtml) {
                state.listHtml = html;
                list.innerHTML = html;
                bindConnectButtons(list);
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
                catalogKnown: Boolean(store.catalogKnown),
                skipPresets: state.skipPresets,
                snapshot,
            });
        }
        login?.render();
    }

    function bindConnectButtons(list) {
        list.querySelectorAll('[data-agent-connect]').forEach((button) => {
            button.addEventListener('click', () => {
                if (state.disposed) return;
                ensureLogin().start(button.getAttribute('data-agent-connect'), '');
            });
        });
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
        paint();
        onStatus();
        if (changed) onChange([...state.connected]);
        if (changed || statusChanged) refreshSubagentsPreview({ force: statusChanged });
    }

    /** Mount into the step DOM the wizard just rendered. */
    function mount() {
        if (state.disposed) return;
        if (!state.unsubscribe) {
            state.unsubscribe = store.subscribe(adopt, { visible: isVisible });
            const pageHideTarget = getDoc()?.defaultView;
            if (pageHideTarget?.addEventListener) {
                // The wizard never unmounts itself, so the ONE listener this
                // controller holds is released on a real unload as well as by
                // dispose(). A bfcache hide (`persisted`) is NOT an unload: the
                // page can come back, and tearing down here would restore a
                // wizard whose Agents step no longer reads anything.
                state.pageHideTarget = pageHideTarget;
                state.pageHideBound = (event) => {
                    if (event?.persisted !== true) detach();
                };
                pageHideTarget.addEventListener('pagehide', state.pageHideBound);
            }
        }
        // A remount rebuilds the DOM the wizard just replaced.
        state.listHtml = null;
        paint();
        subagents.mount();
        // The wizard's API keys are typed on the Accounts step, so the provider
        // list is re-derived on every entry into a step that shows these rows —
        // never once at construction, when no key exists yet.
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
     * Ask the login controller for its exact custody result.
     *
     * Cleanup and local departure are deliberately separate. `retained` means
     * another cancel is pointless; `unknown` remains retryable for the wizard's
     * existing bounded window. Neither result is rewritten as release proof.
     *
     * @returns {Promise<'released'|'retained'|'unknown'>}
     */
    async function dispose() {
        if (state.disposed) return 'unknown';
        // The controller remains the custody SSOT even after its own
        // recovery-face Close: repeated dispose returns its remembered honest
        // detach status without sending another lifecycle request.
        return login ? login.dispose() : 'released';
    }

    /** Resolve sign-in custody as far as the bounded completion window proves. */
    async function disposeForCompletion() {
        // A known retained job cannot be improved by repeating cancel; only an
        // unknown transport result remains retryable. Local detach still makes
        // no claim that the daemon released process custody.
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
        // The payload the family names are spoken from. The wizard's review
        // step renders the SAME families one screen later, and without this it
        // fell back to the bootstrap product names — so an engine rename showed
        // up on the Agents step and then silently un-renamed itself in the
        // summary. Same store, same snapshot, one spelling.
        get snapshot() { return store?.snapshot || null; },
        get catalogKnown() { return Boolean(store?.catalogKnown); },
        get accountsKnown() { return accountsKnown(); },
        get reads() { return store.reads; },
        refreshStatus() { return store.refresh(); },
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
