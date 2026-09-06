// The wizard is served as a real ES module (`GET /onboarding`), so it uses the
// SPA's shared helpers directly instead of carrying private copies that could
// drift from them (the escaping is a security contract, and the request wrapper
// is the gateway boundary) — and the Agents step imports the SAME login
// machinery and the SAME status store the rest of the app uses.
import { fetchJson } from './api_client.js';
import {
    agentsStepHtml,
    completionFailureNotice,
    createAgentsStep,
    familyLabels,
    onboardingSettingsDraft,
    readCompletionAnswer,
} from './onboarding_agents_step.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { installAltMenuSuppression, installDesktopShellLinkInterceptor } from './ui_helpers.js';

(() => {
        // The wizard is its own document inside the overlay iframe, so the SPA's
        // Alt menu-lock guard cannot see its keyboard events — install our own.
        installAltMenuSuppression();
        // Same two-document pattern for the desktop shell: the Agents step's
        // primary "Open sign-in link" is a target="_blank" anchor, a silent
        // no-op in the embedded WebView without the shell link interceptor.
        // When framed, the pywebview bridge lives on the PARENT window; the
        // installer resolves it lazily and stays inert in ordinary browsers.
        installDesktopShellLinkInterceptor();
        const bootstrap = window.__OURO_ONBOARDING_BOOTSTRAP__ || {};
        const SETUP_CONTRACT = bootstrap.contract || {};
        const HOST_MODE = bootstrap.hostMode || 'desktop';
        const LOCAL_RUNTIME_CONTROLS = Boolean(bootstrap.supportsLocalRuntimeControls);
        const STEP_ORDER = bootstrap.stepOrder || (SETUP_CONTRACT.steps || []).map((step) => step.id);
        const STEP_META = Object.fromEntries((SETUP_CONTRACT.steps || []).map((step) => [step.id, step]));
        const PROVIDER_FIELDS = SETUP_CONTRACT.providerFields || [];
        const PROVIDER_PROFILES = SETUP_CONTRACT.providerProfiles || {};
        // The backend contract exports active slots only. Heavy remains a
        // bounded stored-value migration input and never enters this editor.
        const MODEL_SLOTS = SETUP_CONTRACT.modelSlots || [];
        const REVIEW_MODES = SETUP_CONTRACT.reviewModes || [];
        const RUNTIME_MODES = SETUP_CONTRACT.runtimeModes || [];
        const LOCAL_ROUTING_MODES = SETUP_CONTRACT.localRoutingModes || [];
        const BUDGET_FIELDS = SETUP_CONTRACT.budgetFields || [];
        const LOCAL_FIELDS = [
            ['local-source', 'localSource', 'Model Source', 'Qwen/Qwen2.5-7B-Instruct-GGUF or /absolute/path/model.gguf', 'Use either a HuggingFace repo ID or a local absolute GGUF path.', 'field field-full'],
            ['local-filename', 'localFilename', 'GGUF Filename', 'qwen2.5-7b-instruct-q3_k_m.gguf', 'Required only for HuggingFace repo IDs. Leave empty when the source is a direct filesystem path.', 'field field-full'],
            ['local-context', 'localContextLength', 'Context Length', '', '', 'field', 'number', '2048', '1024'],
            ['local-gpu-layers', 'localGpuLayers', 'GPU Layers', '', '', 'field', 'number', '', '1'],
            ['local-chat-format', 'localChatFormat', 'Chat Format', 'Leave empty for auto-detect', '', 'field field-full'],
        ];
        const MODEL_DEFAULTS = bootstrap.modelDefaults || {};
        const LOCAL_PRESETS = bootstrap.localPresets || {};
        const MODEL_SUGGESTIONS = bootstrap.modelSuggestions || [];
        const INITIAL_STATE = bootstrap.initialState || {};
        // What a stored credential looks like in INITIAL_STATE: a marker saying
        // "configured", not the secret. Posting it back means "leave it alone";
        // the shared server-side validator resolves it to the stored value.
        const SECRET_PLACEHOLDER = bootstrap.secretPlaceholder || '';
        const root = document.getElementById('root');

    const state = Object.assign({
        currentStep: STEP_ORDER[0],
        error: '',
        saving: false,
        modelsDirty: false,
        localSourceOpen: Boolean(INITIAL_STATE.localSource),
        moreProvidersOpen: Boolean(
            INITIAL_STATE.cloudruKey || INITIAL_STATE.minimaxKey || INITIAL_STATE.deepseekKey
            || INITIAL_STATE.compatibleBaseUrl || INITIAL_STATE.compatibleApiKey,
        ),
        localStatusText: 'Status: Offline',
        localStatusTone: 'muted',
        localTestResult: '',
        localTestTone: 'muted',
        localRuntimeReady: false,
        // Agents step: what the step OBSERVED (never a guess), the owner's
        // explicit "finish without agent defaults" choice, and the typed reason
        // a completion attempt refused to write the preset.
        agentsConnected: [],
        availableSubagents: null,
        skipSubscriptionPresets: false,
        presetFailure: null,
        // Set when completion answered 503 `settings_save_timeout`: the save is
        // still running in the server, so the wizard offers "Check status"
        // instead of a blind retry (a second write over an unknown first).
        saveUnknown: false,
        // Set once completion SUCCEEDED but the receipt says the saved runtime
        // mode needs a restart, in the one shell that cannot restart anything.
        completedRestartMode: '',
    }, INITIAL_STATE);

    let localStatusPollStarted = false;
    let agentsStep = null;

    function trim(value) {
        return String(value || '').trim();
    }

    function formatUsd(value) {
        const num = Number(value);
        return Number.isFinite(num) ? `$${num.toFixed(2)}` : '$0.00';
    }

    function hasLocalModel() {
        return trim(state.localSource).length > 0;
    }

    // "group" comes from the shared setup contract: "more" fields live inside
    // the collapsed "More options" disclosure; everything else stays in the
    // always-visible access grid. Inputs are mounted in the DOM either way.
    function primaryProviderFields() {
        return PROVIDER_FIELDS.filter((field) => field.group !== 'more');
    }

    function moreProviderFields() {
        return PROVIDER_FIELDS.filter((field) => field.group === 'more');
    }

    function hasMoreProviderValue() {
        return moreProviderFields().some((field) => trim(state[field.stateKey]).length > 0);
    }

    // A credential prefilled from disk arrives as SECRET_PLACEHOLDER, never as
    // the value, so "is this configured?" can no longer be answered by length.
    // The length rule stays for what the owner types here: it is the client
    // mirror of the server's too-short check.
    function isConfiguredCredential(value) {
        const text = trim(value);
        if (!text) return false;
        return text === SECRET_PLACEHOLDER || text.length >= 10;
    }

        function isLocalFilesystemSource(value) {
            const text = trim(value);
            return text.startsWith('/') || text.startsWith('~');
        }

        function optionByValue(items, value) {
            return (items || []).find((item) => item.value === value) || {};
        }

        function detectProviderProfile() {
            const configured = Object.fromEntries(PROVIDER_FIELDS.map((field) => [
                field.settingKey,
                isConfiguredCredential(state[field.stateKey]),
            ]));
            const hasOpenrouter = configured.OPENROUTER_API_KEY;
            const hasCompatible = trim(state.compatibleBaseUrl).length > 0;
            const direct = [
                ['OPENAI_API_KEY', 'openai'],
                ['CLOUDRU_FOUNDATION_MODELS_API_KEY', 'cloudru'],
                ['MINIMAX_API_KEY', 'minimax'],
                ['DEEPSEEK_API_KEY', 'deepseek'],
                ['ANTHROPIC_API_KEY', 'anthropic'],
            ].filter(([settingKey]) => configured[settingKey]);
            if (hasOpenrouter) return 'openrouter';
            if (hasCompatible) return 'openai-compatible';
            if (direct.length > 1) return 'direct-multi';
            if (direct.length === 1) return direct[0][1];
            if (hasLocalModel()) return 'local';
            return 'openrouter';
        }

    function activeProviderProfile() {
        const profile = detectProviderProfile();
        state.providerProfile = profile;
        return profile;
    }

        function profileLabel(profile) {
            return PROVIDER_PROFILES[profile]?.label || PROVIDER_PROFILES.openrouter?.label || 'OpenRouter';
        }

        function reviewLabel(mode) {
            return optionByValue(REVIEW_MODES, mode).label || 'Advisory';
        }

        function runtimeModeLabel(mode) {
            return optionByValue(RUNTIME_MODES, mode).label || 'Advanced';
        }

        function localRoutingLabel(mode) {
            return optionByValue(LOCAL_ROUTING_MODES, mode).label || 'Cloud models only';
        }

    function nextButtonShouldBeDisabled() {
        if (state.saving) return true;
        if (state.currentStep === 'summary') return false;
        return Boolean(validateCurrentStep());
    }

    function syncCurrentStepActionState() {
        const next = document.getElementById('next-btn');
        if (next) next.disabled = nextButtonShouldBeDisabled();
    }

    function markStepEdited() {
        state.error = '';
        // The preview owns generated rows only. Invalidate any response that
        // started against the previous provider/local/model/settings draft;
        // an owner-edited actor list ignores this through its own dirty gate.
        agentsStep?.invalidateGeneratedPreview();
        syncCurrentStepActionState();
    }

    function applyPresetSelection(presetId) {
        state.localPreset = presetId;
        state.localSourceOpen = Boolean(presetId);
        if (!presetId) {
            state.localSource = '';
            state.localFilename = '';
            state.localContextLength = 16384;
            state.localGpuLayers = -1;
            state.localChatFormat = '';
            state.localRoutingMode = 'cloud';
            return;
        }
        if (presetId === 'custom') {
            if (!trim(state.localSource)) {
                state.localSource = '';
                state.localFilename = '';
            }
            return;
        }
        const preset = LOCAL_PRESETS[presetId];
        if (!preset) return;
        state.localSource = preset.source;
        state.localFilename = preset.filename;
        state.localContextLength = preset.contextLength;
        state.localChatFormat = preset.chatFormat || '';
        if (activeProviderProfile() === 'local') {
            state.localRoutingMode = 'all';
        } else if (state.localRoutingMode === 'cloud') {
            state.localRoutingMode = 'fallback';
        }
    }

    function detectLocalPresetSelection() {
        const source = trim(state.localSource);
        const filename = trim(state.localFilename);
        if (!source && !filename) return '';
        for (const [presetId, preset] of Object.entries(LOCAL_PRESETS)) {
            if (source === trim(preset.source) && filename === trim(preset.filename)) {
                return presetId;
            }
        }
        return 'custom';
    }

    function applyModelDefaults(force) {
        if (state.modelsDirty && !force) return;
        const defaults = MODEL_DEFAULTS[activeProviderProfile()] || MODEL_DEFAULTS.openrouter || {};
        state.mainModel = defaults.main || '';
        state.lightModel = defaults.light || '';
        state.fallbackModel = defaults.fallback || '';
        state.modelsDirty = false;
    }

        function validateProvidersStep() {
            const keyValues = PROVIDER_FIELDS.map((field) => [field, trim(state[field.stateKey])]);
            const localSource = trim(state.localSource);
            const localFilename = trim(state.localFilename);
            // Only a credential the owner authored in THIS form is length-checked —
            // the same authorship rule validate_setup_payload applies server-side.
            // The wizard prefills every provider field from disk, so a stored value
            // the owner never touched arrives here unchanged; rejecting it blocks
            // Next/Save on the providers step and the offending value becomes the
            // one value the wizard can never replace.
            const shortKey = keyValues.find(([field, value]) => value && (field.inputType || 'password') === 'password' && value.length < 10 && value !== trim(INITIAL_STATE[field.stateKey]));
            if (shortKey) return `${shortKey[0].label.replace(' API Key', '')} API key looks too short.`;
            const hasRemote = keyValues.some(([field, value]) => value && !['OPENAI_COMPATIBLE_API_KEY', 'MINIMAX_REGION'].includes(field.settingKey));
            if (!hasRemote && !localSource) {
                return 'Enter at least one remote key or a local model source before continuing.';
            }
            if (trim(state.minimaxRegion) && !['global_en', 'cn_zh'].includes(trim(state.minimaxRegion).toLowerCase())) {
                return 'MiniMax Region must be global_en or cn_zh.';
            }
            if (localSource && !hasRemote && trim(state.localRoutingMode) === 'cloud') {
                return 'Local-only setups must route at least one model to the local runtime.';
            }
        if (localSource && localSource.includes('/') && !isLocalFilesystemSource(localSource) && !localFilename) {
            return 'Local HuggingFace sources need a GGUF filename.';
        }
        if (localSource && (!Number.isInteger(Number(state.localContextLength)) || Number(state.localContextLength) <= 0)) {
            return 'Local context length must be a positive integer.';
        }
        if (localSource && !Number.isInteger(Number(state.localGpuLayers))) {
            return 'Local GPU layers must be an integer.';
        }
        return '';
    }

    function validateModelsStep() {
        // Only Main is required; the remaining active slots are optional or
        // already carry a default. Don't force the owner to fill every slot.
        if (!trim(state.mainModel)) {
            return 'Confirm the Main model before starting Ouroboros.';
        }
        return '';
    }

    function validateReviewStep() {
        if (!['advisory', 'blocking'].includes(trim(state.reviewEnforcement))) {
            return 'Choose advisory or blocking review mode.';
        }
        return '';
    }

    function validateBudgetStep() {
        for (const field of BUDGET_FIELDS) {
            const value = Number(state[field.stateKey]);
            const min = Number(field.min || 0.01);
            if (!Number.isFinite(value) || value < min) {
                return `${field.title || field.label || 'Budget'} must be greater than zero.`;
            }
        }
        return '';
    }

    function validateCurrentStep() {
        // 'agents' is deliberately absent from this per-step navigation gate: a
        // subscription is an amplifier, never an admission requirement. Final
        // completion separately validates the visible canonical actor draft.
        if (state.currentStep === 'providers') return validateProvidersStep();
        if (state.currentStep === 'models') return validateModelsStep();
        if (state.currentStep === 'review_mode') return validateReviewStep();
        if (state.currentStep === 'budget') return validateBudgetStep();
        return '';
    }

    function nextStep() {
        const error = validateCurrentStep();
        state.error = error;
        if (error) {
            render();
            return;
        }
        if (state.currentStep === 'providers') applyModelDefaults(false);
        if (['providers', 'models', 'review_mode', 'budget'].includes(state.currentStep)) {
            // Preview is enrichment, never a navigation gate. Refresh in the
            // background after the step has a valid complete draft; Finish
            // checks the receipt only if generated rows still own the editor.
            if (agentsStep) void agentsStep.refreshSubagentsPreview({ force: true });
        }
        const index = STEP_ORDER.indexOf(state.currentStep);
        if (index >= 0 && index < STEP_ORDER.length - 1) {
            state.currentStep = STEP_ORDER[index + 1];
        }
        state.error = '';
        render();
    }

    function previousStep() {
        const index = STEP_ORDER.indexOf(state.currentStep);
        if (index > 0) state.currentStep = STEP_ORDER[index - 1];
        state.error = '';
        render();
    }

    const apiRequest = fetchJson;

    function renderLocalStatus() {
        const statusEl = document.getElementById('wizard-local-status');
        const stopButton = document.getElementById('wizard-local-stop');
        const testButton = document.getElementById('wizard-local-test');
        const resultEl = document.getElementById('wizard-local-test-result');
        if (statusEl) {
            statusEl.textContent = state.localStatusText || 'Status: Offline';
            statusEl.dataset.tone = state.localStatusTone || 'muted';
        }
        if (stopButton) stopButton.disabled = !state.localRuntimeReady;
        if (testButton) testButton.disabled = !state.localRuntimeReady;
        if (resultEl) {
            resultEl.hidden = !state.localTestResult;
            resultEl.dataset.tone = state.localTestTone || 'muted';
            resultEl.textContent = state.localTestResult || '';
        }
    }

    function setLocalTestResult(text, tone = 'muted') {
        state.localTestResult = text || '';
        state.localTestTone = tone;
        renderLocalStatus();
    }

    async function updateLocalStatus() {
        if (!LOCAL_RUNTIME_CONTROLS) return;
        try {
            const data = await apiRequest('/api/local-model/status', { cache: 'no-store' });
            const isReady = data.status === 'ready';
            let text = 'Status: ' + ((data.status || 'offline').charAt(0).toUpperCase() + (data.status || 'offline').slice(1));
            if (data.status === 'ready' && data.context_length) text += ` (ctx: ${data.context_length})`;
            if (data.status === 'downloading' && data.download_progress) text += ` ${Math.round(data.download_progress * 100)}%`;
            if (data.error) text += ` - ${data.error}`;
            state.localRuntimeReady = isReady;
            state.localStatusText = text;
            state.localStatusTone = isReady ? 'ok' : (data.status === 'error' ? 'error' : 'muted');
            renderLocalStatus();
        } catch (error) {
            state.localRuntimeReady = false;
            state.localStatusText = `Status: Error - ${error.message}`;
            state.localStatusTone = 'error';
            renderLocalStatus();
        }
    }

    function readLocalModelBody() {
        return {
            source: trim(state.localSource),
            filename: trim(state.localFilename),
            port: 8766,
            n_gpu_layers: parseInt(state.localGpuLayers, 10),
            n_ctx: parseInt(state.localContextLength, 10) || 16384,
            chat_format: trim(state.localChatFormat),
        };
    }

    function startLocalStatusPolling() {
        if (!LOCAL_RUNTIME_CONTROLS || localStatusPollStarted) return;
        localStatusPollStarted = true;
        updateLocalStatus();
        setInterval(updateLocalStatus, 3000);
    }

    function renderLocalControls() {
        if (!LOCAL_RUNTIME_CONTROLS) return '';
        return `
            <div class="wizard-runtime-strip">
                <button type="button" class="btn btn-ghost" id="wizard-local-start">Start local runtime</button>
                <button type="button" class="btn btn-ghost" id="wizard-local-stop" disabled>Stop</button>
                <button type="button" class="btn btn-ghost" id="wizard-local-test" disabled>Test tool calling</button>
                <span id="wizard-local-status" class="wizard-runtime-status">Status: Offline</span>
            </div>
            <div id="wizard-local-test-result" class="wizard-test-result"></div>
        `;
    }

    function summaryRows() {
        const rows = [
            ['Detected setup', profileLabel(activeProviderProfile())],
            ['Review mode', reviewLabel(state.reviewEnforcement)],
            ['Runtime mode', runtimeModeLabel(state.runtimeMode)],
            ['Total budget', formatUsd(state.totalBudget)],
            ['Per-task cost cap', formatUsd(state.perTaskCostUsd)],
            ['Main', trim(state.mainModel)],
            ['Light', trim(state.lightModel) || '(uses Main)'],
            ['Fallback', trim(state.fallbackModel)],
        ];
        if (trim(state.openrouterKey)) rows.splice(1, 0, ['OpenRouter', 'configured']);
        if (trim(state.openaiKey)) rows.splice(1, 0, ['OpenAI', 'configured']);
        if (trim(state.cloudruKey)) rows.splice(1, 0, ['Cloud.ru', 'configured']);
        if (trim(state.minimaxKey)) rows.splice(1, 0, ['MiniMax', 'configured']);
        if (trim(state.deepseekKey)) rows.splice(1, 0, ['DeepSeek', 'configured']);
        if (trim(state.anthropicKey)) rows.splice(1, 0, ['Anthropic', 'configured']);
        if (hasLocalModel()) {
            rows.splice(
                1,
                0,
                ['Local source', trim(state.localSource) + (trim(state.localFilename) ? ` / ${trim(state.localFilename)}` : '')],
                ['Local routing', localRoutingLabel(state.localRoutingMode)],
            );
        }
        rows.push(['Agents', agentsSummaryValue()]);
        if (trim(state.skillsRepoPath)) {
            rows.push(['Skills repo', trim(state.skillsRepoPath)]);
        }
        return rows;
    }

    function agentsSummaryValue() {
        // Through the step's own snapshot, so this line and the Agents step one
        // screen earlier spell a family the same way. Without it the summary
        // fell back to the bootstrap names and quietly undid an engine rename.
        const labels = familyLabels(state.agentsConnected, agentsStep?.snapshot, {
            catalogKnown: Boolean(agentsStep?.catalogKnown),
        });
        const actorCount = (agentsStep?.availableSubagents?.items
            || state.availableSubagents?.items || []).length;
        const actors = `${actorCount} Available subagent${actorCount === 1 ? '' : 's'}`;
        if (!labels.length) return `${actors} · API/local access only`;
        if (state.skipSubscriptionPresets) {
            return `${actors} · ${labels.join(', ')} connected · automatic subscription preset skipped`;
        }
        return `${actors} · ${labels.join(', ')} connected`;
    }

    function shouldOfferPresetSkip() {
        // The endpoint's own escape hatch, surfaced exactly when it can change
        // the outcome: something is connected to move onto, or a completion
        // attempt already refused the preset and said the skip is available.
        return state.agentsConnected.length > 0 || Boolean(state.presetFailure);
    }

        function providerKeyField({ id, label, placeholder, value, note, inputType }) {
            const type = inputType || 'password';
            return `
                <div class="field">
                <div class="field-label-row">
                    <label for="${escapeHtml(id)}">${escapeHtml(label)}</label>
                    <button class="field-clear" data-clear="${escapeHtml(id)}" type="button">Clear</button>
                </div>
                <input id="${escapeHtml(id)}" type="${escapeHtml(type)}" placeholder="${escapeHtml(placeholder)}" value="${escapeHtml(value)}">
                <div class="field-note">${escapeHtml(note)}</div>
            </div>
            `;
        }

        function localInputField([id, stateKey, label, placeholder, note, className, type = 'text', min = '', step = '']) {
            const clear = ['local-source', 'local-filename', 'local-chat-format'].includes(id)
                ? `<button class="field-clear" data-clear="${id}" type="button">Clear</button>`
                : '';
            return `
                <div class="${className}">
                    <div class="field-label-row"><label for="${id}">${label}</label>${clear}</div>
                    <input id="${id}" type="${type}" ${min ? `min="${min}"` : ''} ${step ? `step="${step}"` : ''} placeholder="${placeholder}" value="${escapeHtml(state[stateKey])}">
                    ${note ? `<div class="field-note">${note}</div>` : ''}
                </div>
            `;
        }

        function renderProvidersStep() {
        const selectedProfile = activeProviderProfile();
        const localPreset = trim(state.localPreset);
        const localSourceOpen = state.localSourceOpen || hasLocalModel();
        const moreProvidersOpen = state.moreProvidersOpen || hasMoreProviderValue();
        return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">${escapeHtml(STEP_META.providers.title)}</h2>
                    <p class="step-copy">${escapeHtml(STEP_META.providers.copy)}</p>
                </div>
            </div>
                <div class="panel-card">
                    <h3>Keys first, routing second</h3>
                    <p>${escapeHtml(PROVIDER_PROFILES[selectedProfile]?.providerCopy || '')}</p>
                </div>
                <div class="field-grid">
                    ${primaryProviderFields().map((field) => providerKeyField({
                        ...field,
                        value: state[field.stateKey],
                    })).join('')}
                </div>
            <details class="wizard-collapse" data-collapse="more-providers" ${moreProvidersOpen ? 'open' : ''}>
                <summary>
                    <span>More options</span>
                    <span class="selection-badge">${hasMoreProviderValue() ? 'Configured' : 'Optional'}</span>
                </summary>
                <div class="wizard-collapse-body">
                    <div class="field-grid">
                        ${moreProviderFields().map((field) => providerKeyField({
                            ...field,
                            value: state[field.stateKey],
                        })).join('')}
                    </div>
                </div>
            </details>
            <details class="wizard-collapse" data-collapse="local-model" ${localSourceOpen ? 'open' : ''}>
                <summary>
                    <span>Local model settings</span>
                    <span class="selection-badge">${hasLocalModel() ? 'Configured' : 'Optional'}</span>
                </summary>
                <div class="wizard-collapse-body">
                    <div class="field-grid">
                        <div class="field">
                            <div class="field-label-row">
                                <label for="local-preset">Preset</label>
                                <button class="field-clear" data-clear="local-preset" type="button">Clear</button>
                            </div>
                                <select id="local-preset">
                                    <option value="" ${localPreset === '' ? 'selected' : ''}>None</option>
                                    ${Object.entries(LOCAL_PRESETS).map(([id, preset]) => `<option value="${escapeHtml(id)}" ${localPreset === id ? 'selected' : ''}>${escapeHtml(preset.label)}</option>`).join('')}
                                    <option value="custom" ${localPreset === 'custom' ? 'selected' : ''}>Custom source</option>
                                </select>
                            <div class="field-note">Most people can ignore this. Open it only if you want local GGUF routing.</div>
                        </div>
                        <div class="field">
                                <div class="field-label-row"><label>Local routing</label></div>
                                <div class="selection-row">
                                    ${LOCAL_ROUTING_MODES.map((mode) => `<button class="selection-pill ${state.localRoutingMode === mode.value ? 'active' : ''}" data-local-mode="${escapeHtml(mode.value)}" type="button">${escapeHtml(mode.buttonLabel || mode.label)}</button>`).join('')}
                                </div>
                                <div class="field-note">Ignored unless a local model source is configured below.</div>
                            </div>
                            ${LOCAL_FIELDS.map(localInputField).join('')}
                        </div>
                    ${renderLocalControls()}
                </div>
            </details>
        `;
    }

    function renderAgentsStep() {
        // SKIPPABLE by construction: `validateCurrentStep` returns '' for this
        // step, so Continue is never disabled and no field here can block
        // finishing. An agent plan is an amplifier, never an admission gate.
        return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">${escapeHtml(STEP_META.agents.title)}</h2>
                    <p class="step-copy">${escapeHtml(STEP_META.agents.copy)}</p>
                </div>
            </div>
            ${agentsStepHtml()}
        `;
    }

    function bindAgentsStep() {
        if (!agentsStep) {
            agentsStep = createAgentsStep({
                isVisible: () => state.currentStep === 'agents',
                onChange: (connected) => { state.agentsConnected = connected; },
                previewPayload: () => onboardingSettingsDraft({ state, providerFields: PROVIDER_FIELDS, budgetFields: BUDGET_FIELDS, modelSlots: MODEL_SLOTS, trim }),
                onSubagentsChange: (setting) => { state.availableSubagents = setting; },
            });
        }
        agentsStep.setSkipPresets(state.skipSubscriptionPresets);
        agentsStep.mount();
        syncCurrentStepActionState();
    }

    function modelSuggestionField({ id, label, value, note }) {
        return `
            <div class="field wizard-model-field" data-wizard-model-field>
                <label for="${escapeHtml(id)}">${escapeHtml(label)}</label>
                <input id="${escapeHtml(id)}" value="${escapeHtml(value)}" autocomplete="off" spellcheck="false" data-wizard-model-input>
                <div class="wizard-model-suggestions" hidden></div>
                <div class="field-note">${escapeHtml(note)}</div>
            </div>
        `;
    }

        function renderCompatibleModelLoader() {
            return `
            <div class="panel-card" id="compatible-model-loader">
                <h3>Load models from endpoint</h3>
                <p class="field-note">Fetch the model list from your configured URL, then click a model to fill all empty slots.</p>
                <div class="compatible-model-actions">
                    <button type="button" class="btn btn-secondary" id="load-compatible-models">Load models</button>
                    <span id="compatible-load-status" class="field-note compatible-load-status"></span>
                </div>
                <div id="compatible-model-list" class="compatible-model-list" hidden></div>
            </div>
            `;
        }

        function renderModelsStep() {
            const profile = activeProviderProfile();
            return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">${escapeHtml(STEP_META.models.title)}</h2>
                    <p class="step-copy">${escapeHtml(STEP_META.models.copy)}</p>
                </div>
            </div>
                <div class="panel-card">
                    <h3>Current profile</h3>
                    <p>${escapeHtml(PROVIDER_PROFILES[profile]?.modelCopy || '')}</p>
                </div>
                ${profile === 'openai-compatible' ? renderCompatibleModelLoader() : ''}
                <div class="grid two">
                    ${MODEL_SLOTS.map((slot) => modelSuggestionField({
                        id: slot.inputId,
                        label: slot.label,
                        value: state[slot.stateKey],
                        note: slot.note,
                    })).join('')}
                </div>
            <div class="wizard-inline-note">Direct providers use explicit <code>provider::model</code> values, including <code>minimax::MiniMax-M3</code>, <code>minimax::MiniMax-M2.7</code>, <code>deepseek::deepseek-v4-pro</code> and <code>deepseek::deepseek-v4-flash</code>. OpenAI-compatible endpoints use <code>openai-compatible::your-model-name</code>. Plain slash-form model IDs stay router-style by design.</div>
        `;
    }

    function renderReviewModeStep() {
        const runtimeMode = trim(state.runtimeMode) || 'advanced';
        const runtimeModeCopy = HOST_MODE === 'desktop'
            ? 'Separate axis from review enforcement. This first-run choice becomes the boot baseline before Ouroboros starts; later elevation requires native launcher confirmation.'
            : 'Separate axis from review enforcement. Web/Docker onboarding saves this through the owner endpoint; the selected mode becomes active after restart.';
        return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">${escapeHtml(STEP_META.review_mode.title)}</h2>
                    <p class="step-copy">${escapeHtml(STEP_META.review_mode.copy)}</p>
                </div>
                </div>
                <div class="wizard-choice-grid">
                    ${REVIEW_MODES.map((mode) => `
                        <button type="button" class="wizard-choice ${escapeHtml(mode.className || mode.value)} ${state.reviewEnforcement === mode.value ? 'active' : ''}" data-review-mode="${escapeHtml(mode.value)}">
                            <span class="tone">${escapeHtml(mode.tone)}</span>
                            <h3>${escapeHtml(mode.label)}</h3>
                            <p>${escapeHtml(mode.copy)}</p>
                        </button>
                    `).join('')}
                </div>
            <div class="panel-card runtime-mode-card">
                <h3>Runtime mode</h3>
                    <p class="field-note">${escapeHtml(runtimeModeCopy)}</p>
                    <div class="wizard-choice-grid three">
                        ${RUNTIME_MODES.map((mode) => `
                            <button type="button" class="wizard-choice ${escapeHtml(mode.className || mode.value)} ${runtimeMode === mode.value ? 'active' : ''}" data-runtime-mode="${escapeHtml(mode.value)}">
                                <span class="tone">${escapeHtml(mode.tone)}</span>
                                <h3>${escapeHtml(mode.label)}</h3>
                                <p>${escapeHtml(mode.copy)}</p>
                            </button>
                        `).join('')}
                    </div>
                <div class="field">
                    <div class="field-label-row">
                        <label for="skills-repo-path">External skills repo (optional)</label>
                        <button class="field-clear" data-clear="skills-repo-path" type="button">Clear</button>
                    </div>
                    <input id="skills-repo-path" type="text" placeholder="~/Ouroboros/skills or /absolute/path/to/skills" value="${escapeHtml(state.skillsRepoPath || '')}">
                    <div class="field-note">Optional. Extra discovery root on top of the in-data-plane <code>data/skills/{native,clawhub,external}/</code> tree. Leave empty if you do not maintain your own skills checkout — Ouroboros never clones/pulls this directory.</div>
                </div>
            </div>
        `;
    }

        function renderBudgetStep() {
            return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">${escapeHtml(STEP_META.budget.title)}</h2>
                    <p class="step-copy">${escapeHtml(STEP_META.budget.copy)}</p>
                </div>
                </div>
                <div class="grid two">
                    ${BUDGET_FIELDS.map((field) => `
                        <div class="panel-card">
                            <h3>${escapeHtml(field.title)}</h3>
                            <div class="field">
                                <label for="${escapeHtml(field.inputId)}">${escapeHtml(field.label)}</label>
                                <input id="${escapeHtml(field.inputId)}" type="number" min="${escapeHtml(field.min || '0.01')}" step="${escapeHtml(field.step || 'any')}" value="${escapeHtml(state[field.stateKey])}">
                                <div class="field-note">${escapeHtml(field.note)}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

    function renderSummaryStep() {
        const summary = summaryRows().map(([label, value]) => `
            <div class="summary-kv">
                <strong>${escapeHtml(label)}</strong>
                <span>${escapeHtml(value)}</span>
            </div>
        `).join('');
        return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">${escapeHtml(STEP_META.summary.title)}</h2>
                    <p class="step-copy">${escapeHtml(STEP_META.summary.copy)}</p>
                </div>
            </div>
            <div class="summary-card">${summary}</div>
        `;
    }

    function renderRestartRequiredScreen() {
        return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">Setup saved — restart to apply it</h2>
                    <p class="step-copy">Everything you entered is on disk. One choice needs a fresh start before it is live.</p>
                </div>
            </div>
            <div class="panel-card">
                <h3>Runtime mode takes effect at the next boot</h3>
                <p>Ouroboros saved <code>${escapeHtml(state.completedRestartMode)}</code> as the runtime mode for the next boot, but this
                process is still running the mode it started with. Restart Ouroboros to run in the mode you chose.
                Opening the app now works — it simply runs in the previous mode until you do.</p>
                <div class="wizard-runtime-strip">
                    <button type="button" class="btn btn-primary" id="open-app-btn">Open Ouroboros in the current mode</button>
                </div>
            </div>
        `;
    }

    function renderStepContent() {
        if (state.completedRestartMode) return renderRestartRequiredScreen();
        if (state.currentStep === 'providers') return renderProvidersStep();
        if (state.currentStep === 'agents') return renderAgentsStep();
        if (state.currentStep === 'models') return renderModelsStep();
        if (state.currentStep === 'review_mode') return renderReviewModeStep();
        if (state.currentStep === 'budget') return renderBudgetStep();
        return renderSummaryStep();
    }

    function stepCards() {
        return STEP_ORDER.map((stepId, index) => {
            const active = stepId === state.currentStep;
            const done = STEP_ORDER.indexOf(state.currentStep) > index;
            const meta = STEP_META[stepId];
            return `
                <div class="wizard-step ${active ? 'active' : ''} ${done ? 'done' : ''}">
                    <div class="wizard-step-index">Step ${index + 1}</div>
                    <p class="wizard-step-title">${escapeHtml(meta.title)}</p>
                    <p class="wizard-step-copy">${escapeHtml(meta.railCopy || '')}</p>
                </div>
            `;
        }).join('');
    }

    function render() {
        const meta = STEP_META[state.currentStep];
        const index = STEP_ORDER.indexOf(state.currentStep);
        // An UNKNOWN save (503 settings_save_timeout: the write may still be
        // running in the server) makes "Check status" the primary action and
        // the re-submit an explicit, secondary "Retry save" — never the default
        // button beside it, which re-POSTed a second write over the first.
        const saveUnknown = state.currentStep === 'summary' && state.saveUnknown;
        const nextLabel = state.currentStep === 'summary'
            ? (state.saving ? 'Saving...' : (saveUnknown ? 'Retry save' : 'Start Ouroboros'))
            : 'Continue';
        root.innerHTML = `
            <div class="wizard-shell">
                <div class="wizard-header">
                    <div>
                        <h1 class="wizard-title">Ouroboros</h1>
                        <p class="wizard-subtitle">Shared desktop and web onboarding with the same model, review, and budget flow in both hosts.</p>
                    </div>
                    <div class="wizard-badge">Step ${index + 1} of ${STEP_ORDER.length}</div>
                </div>
                <div class="wizard-steps">${stepCards()}</div>
                <div class="wizard-content">
                    ${renderStepContent()}
                    ${state.completedRestartMode ? '' : `
                    <div class="wizard-footer">
                        <div class="footer-copy">${escapeHtml(meta.footer)}</div>
                        <div class="footer-actions">
                            <button class="btn btn-secondary" id="back-btn" type="button" ${index === 0 || state.saving ? 'disabled' : ''}>Back</button>
                            ${state.currentStep === 'summary' && shouldOfferPresetSkip() ? `
                                <button class="btn btn-secondary" id="skip-presets-btn" type="button" ${state.saving ? 'disabled' : ''}>Finish without subscription presets</button>
                            ` : ''}
                            <button class="btn ${saveUnknown ? 'btn-secondary' : 'btn-primary'}" id="next-btn" type="button" ${nextButtonShouldBeDisabled() ? 'disabled' : ''}>${escapeHtml(nextLabel)}</button>
                            ${saveUnknown ? `
                                <button class="btn btn-primary" id="check-save-btn" type="button" ${state.saving ? 'disabled' : ''}>Check status</button>
                            ` : ''}
                        </div>
                    </div>
                    <div class="wizard-error">${escapeHtml(state.error)}</div>
                    `}
                </div>
            </div>
        `;
        bindEvents();
        renderLocalStatus();
    }

        function bindClearButtons() {
            const clearActions = Object.fromEntries(PROVIDER_FIELDS.map((field) => [
                field.id,
                () => { state[field.stateKey] = ''; },
            ]));
            Object.assign(clearActions, {
                'local-preset': () => {
                    state.localPreset = '';
                    state.localSource = '';
                state.localFilename = '';
                state.localRoutingMode = 'cloud';
                state.localSourceOpen = false;
            },
            'local-source': () => {
                state.localSource = '';
                state.localPreset = detectLocalPresetSelection();
            },
            'local-filename': () => {
                state.localFilename = '';
                state.localPreset = detectLocalPresetSelection();
                },
                'local-chat-format': () => { state.localChatFormat = ''; },
                'skills-repo-path': () => { state.skillsRepoPath = ''; },
            });
        root.querySelectorAll('[data-clear]').forEach((button) => {
            button.addEventListener('click', () => {
                const target = button.getAttribute('data-clear');
                if (clearActions[target]) clearActions[target]();
                state.error = '';
                agentsStep?.invalidateGeneratedPreview();
                render();
            });
        });
    }

    function bindProvidersStep() {
        // Scoped per-collapse binding: a bare `.wizard-collapse` selector only
        // reaches the FIRST details element, which silently drops the toggle
        // persistence of every later collapse on the step.
        const collapseStateKeys = {
            'more-providers': 'moreProvidersOpen',
            'local-model': 'localSourceOpen',
        };
        Object.entries(collapseStateKeys).forEach(([collapseId, stateKey]) => {
            const details = root.querySelector(`[data-collapse="${collapseId}"]`);
            if (details) {
                details.addEventListener('toggle', () => {
                    state[stateKey] = details.open;
                });
            }
        });
            const localPreset = document.getElementById('local-preset');
            const localSource = document.getElementById('local-source');
        const localFilename = document.getElementById('local-filename');
        const localContext = document.getElementById('local-context');
        const localGpuLayers = document.getElementById('local-gpu-layers');
        const localChatFormat = document.getElementById('local-chat-format');

        function bindStateInput(input, key, after = null) {
            if (!input) return;
            input.addEventListener('input', () => {
                state[key] = input.value;
                if (after) after(input);
                markStepEdited();
            });
        }

            PROVIDER_FIELDS.forEach((field) => {
                bindStateInput(document.getElementById(field.id), field.stateKey);
            });
        if (localPreset) localPreset.addEventListener('change', () => {
            applyPresetSelection(localPreset.value);
            state.error = '';
            agentsStep?.invalidateGeneratedPreview();
            render();
        });
        bindStateInput(localSource, 'localSource', () => {
            state.localPreset = detectLocalPresetSelection();
            if (localPreset) localPreset.value = state.localPreset || '';
            state.localSourceOpen = true;
            if (trim(state.localSource) && activeProviderProfile() === 'local' && trim(state.localRoutingMode) === 'cloud') {
                state.localRoutingMode = 'all';
            }
        });
        bindStateInput(localFilename, 'localFilename', () => {
            state.localPreset = detectLocalPresetSelection();
            if (localPreset) localPreset.value = state.localPreset || '';
        });
        bindStateInput(localContext, 'localContextLength');
        bindStateInput(localGpuLayers, 'localGpuLayers');
        bindStateInput(localChatFormat, 'localChatFormat');
        root.querySelectorAll('[data-local-mode]').forEach((button) => {
            button.addEventListener('click', () => {
                state.localRoutingMode = button.getAttribute('data-local-mode');
                state.error = '';
                agentsStep?.invalidateGeneratedPreview();
                render();
            });
        });
        if (LOCAL_RUNTIME_CONTROLS) {
            startLocalStatusPolling();
            document.getElementById('wizard-local-start')?.addEventListener('click', async () => {
                const body = readLocalModelBody();
                if (!body.source) {
                    state.error = 'Enter a local model source before starting the local runtime.';
                    render();
                    return;
                }
                setLocalTestResult('', 'muted');
                try {
                    const resp = await fetch('/api/local-model/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body),
                    });
                    const data = await resp.json().catch(() => ({}));
                    if (resp.status === 412 && data.error === 'runtime_missing') {
                        setLocalTestResult(
                            'Local runtime (llama-cpp-python) is not installed.\n' +
                            'Go to Settings → Advanced → Local Model Runtime\n' +
                            'and click "Install Local Runtime".\n\n' +
                            'Manual: ' + (data.hint || 'pip install llama-cpp-python[server]'),
                            'error'
                        );
                    } else if (data.error) {
                        setLocalTestResult(`Start failed: ${data.error}`, 'error');
                    } else {
                        updateLocalStatus();
                    }
                } catch (error) {
                    setLocalTestResult(`Start failed: ${error.message}`, 'error');
                }
            });
            document.getElementById('wizard-local-stop')?.addEventListener('click', async () => {
                try {
                    await apiRequest('/api/local-model/stop', { method: 'POST' });
                    updateLocalStatus();
                } catch (error) {
                    setLocalTestResult(`Stop failed: ${error.message}`, 'error');
                }
            });
            document.getElementById('wizard-local-test')?.addEventListener('click', async () => {
                setLocalTestResult('Running tests...', 'muted');
                try {
                    const result = await apiRequest('/api/local-model/test', { method: 'POST' });
                    const lines = [];
                    lines.push(`${result.chat_ok ? '✓' : '✗'} Basic chat${result.tokens_per_sec ? ` (${result.tokens_per_sec} tok/s)` : ''}`);
                    lines.push(`${result.tool_call_ok ? '✓' : '✗'} Tool calling`);
                    if (result.details && !result.success) lines.push(result.details);
                    setLocalTestResult(lines.join('\n'), result.success ? 'ok' : 'warn');
                } catch (error) {
                    setLocalTestResult(`Test failed: ${error.message}`, 'error');
                }
            });
        }
        syncCurrentStepActionState();
    }

        function bindCompatibleModelLoader() {
            const loadBtn = document.getElementById('load-compatible-models');
            if (!loadBtn) return;
            loadBtn.addEventListener('click', async () => {
                const baseUrl = trim(state.compatibleBaseUrl).replace(/\/+$/, '');
                const apiKey = trim(state.compatibleApiKey);
                const statusEl = document.getElementById('compatible-load-status');
                const listEl = document.getElementById('compatible-model-list');
                if (!baseUrl) {
                    if (statusEl) statusEl.textContent = 'Go back and enter a base URL first.';
                    return;
                }
                if (statusEl) statusEl.textContent = 'Loading…';
                loadBtn.disabled = true;
                try {
                    let models;
                    if (HOST_MODE === 'web') {
                        const resp = await fetch('/api/openai-compatible/models', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ baseUrl, apiKey }),
                            cache: 'no-store',
                        });
                        const data = await resp.json().catch(() => ({}));
                        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
                        models = (data.models || []).map((m) => trim(m)).filter(Boolean).sort();
                    } else {
                        if (!window.pywebview?.api?.fetch_compatible_models) {
                            throw new Error('Desktop model-fetch bridge unavailable.');
                        }
                        const result = await window.pywebview.api.fetch_compatible_models({ baseUrl, apiKey });
                        if (result?.error) throw new Error(result.error);
                        models = (result?.models || []).map((m) => trim(m)).filter(Boolean).sort();
                    }
                    if (!models.length) throw new Error('No models returned by endpoint.');
                    if (statusEl) statusEl.textContent = `${models.length} model${models.length === 1 ? '' : 's'} found — click one to fill empty slots.`;
                    if (listEl) {
                        listEl.hidden = false;
                        listEl.innerHTML = models.map((m) =>
                            `<button type="button" class="selection-pill" data-apply-model="${escapeHtml(m)}">${escapeHtml(m)}</button>`
                        ).join('');
                    }
                } catch (err) {
                    const msg = String(err?.message || err || 'Unknown error');
                    if (statusEl) statusEl.textContent = `Failed: ${msg}`;
                    if (listEl) { listEl.hidden = true; listEl.innerHTML = ''; }
                } finally {
                    loadBtn.disabled = false;
                }
            });
            if (!root._compatModelListenerBound) {
                root._compatModelListenerBound = true;
                root.addEventListener('click', (event) => {
                    const pill = event.target.closest('[data-apply-model]');
                    if (!pill) return;
                    const modelId = `openai-compatible::${pill.dataset.applyModel}`;
                    if (!trim(state.mainModel)) state.mainModel = modelId;
                    if (!trim(state.lightModel)) state.lightModel = modelId;
                    if (!trim(state.fallbackModel)) state.fallbackModel = modelId;
                    state.modelsDirty = true;
                    agentsStep?.invalidateGeneratedPreview();
                    render();
                });
            }
        }

        function bindModelsStep() {
            const modelInputMap = Object.fromEntries(MODEL_SLOTS.map((slot) => [slot.inputId, slot.stateKey]));
            bindCompatibleModelLoader();
            function suggestionMatches(query) {
                const needle = trim(query).toLowerCase();
                return MODEL_SUGGESTIONS
                    .filter((model) => !needle || String(model).toLowerCase().includes(needle))
                    .slice(0, 8);
            }
        function closeSuggestions(exceptInput = null) {
            root.querySelectorAll('.wizard-model-suggestions').forEach((panel) => {
                if (exceptInput && panel.parentElement?.querySelector('input') === exceptInput) return;
                panel.hidden = true;
                panel.innerHTML = '';
            });
        }
        function renderSuggestions(input) {
            const panel = input.closest('[data-wizard-model-field]')?.querySelector('.wizard-model-suggestions');
            if (!panel) return;
            const matches = suggestionMatches(input.value);
            if (!matches.length) {
                panel.hidden = true;
                panel.innerHTML = '';
                return;
            }
            panel.innerHTML = matches.map((model) => (
                `<button type="button" class="wizard-model-suggestion" data-value="${escapeHtml(model)}">${escapeHtml(model)}</button>`
            )).join('');
            panel.hidden = false;
        }
        Object.entries(modelInputMap).forEach(([id, key]) => {
            const input = document.getElementById(id);
            if (!input) return;
            input.addEventListener('focus', () => {
                closeSuggestions(input);
                renderSuggestions(input);
            });
            input.addEventListener('input', () => {
                state[key] = input.value;
                state.modelsDirty = true;
                state.error = '';
                agentsStep?.invalidateGeneratedPreview();
                closeSuggestions(input);
                renderSuggestions(input);
                syncCurrentStepActionState();
            });
            input.addEventListener('change', () => {
                void agentsStep?.refreshSubagentsPreview({ force: true });
            });
        });
        root.querySelectorAll('.wizard-model-suggestions').forEach((panel) => {
            panel.addEventListener('mousedown', (event) => {
                const button = event.target.closest('.wizard-model-suggestion');
                if (!button) return;
                event.preventDefault();
                const input = panel.parentElement?.querySelector('input');
                if (!input) return;
                input.value = button.dataset.value || '';
                input.dispatchEvent(new Event('input', { bubbles: true }));
                closeSuggestions();
            });
        });
        if (root.dataset.modelSuggestionOutsideListener !== '1') {
            root.dataset.modelSuggestionOutsideListener = '1';
            document.addEventListener('mousedown', (event) => {
                if (!root.contains(event.target) || !event.target.closest('[data-wizard-model-field]')) {
                    root.querySelectorAll('.wizard-model-suggestions').forEach((panel) => {
                        panel.hidden = true;
                        panel.innerHTML = '';
                    });
                }
            });
        }
        syncCurrentStepActionState();
    }

    function bindReviewModeStep() {
        root.querySelectorAll('[data-review-mode]').forEach((button) => {
            button.addEventListener('click', () => {
                state.reviewEnforcement = button.getAttribute('data-review-mode');
                state.error = '';
                agentsStep?.invalidateGeneratedPreview();
                render();
            });
        });
        root.querySelectorAll('[data-runtime-mode]').forEach((button) => {
            button.addEventListener('click', () => {
                state.runtimeMode = button.getAttribute('data-runtime-mode');
                state.error = '';
                agentsStep?.invalidateGeneratedPreview();
                render();
            });
        });
        const skillsInput = document.getElementById('skills-repo-path');
        if (skillsInput) skillsInput.addEventListener('input', () => { state.skillsRepoPath = skillsInput.value; markStepEdited(); });
        syncCurrentStepActionState();
    }

        function bindBudgetStep() {
            BUDGET_FIELDS.forEach((field) => {
                const input = document.getElementById(field.inputId);
                if (input) input.addEventListener('input', () => { state[field.stateKey] = input.value; markStepEdited(); });
            });
            syncCurrentStepActionState();
        }

    // --- Completion ---------------------------------------------------------
    // ONE completion path on every host (D-8). The wizard runs against a live
    // gateway everywhere, so it posts the single atomic transaction and then
    // tells whichever shell embeds this page that setup is done; only the
    // announcement differs (embedded frame / desktop setup window / browser tab).
    //
    // The two legacy fallbacks are GONE. `POST /api/settings` + `POST
    // /api/owner/runtime-mode` was the pair whose failure between the two writes
    // left providers saved and runtime mode not, and the desktop `save_wizard`
    // bridge existed only to author the fresh-install `light` safety coverage
    // that the endpoint now authors itself, on its own server-side freshness
    // proof. Keeping either as a "not deployed yet" hedge meant a first run
    // could still silently take a non-atomic path.

    const ONBOARDING_COMPLETE_ENDPOINT = '/api/onboarding/complete';

    function announceCompletion(result) {
        const restartRequired = Boolean(result?.restart_required);
        const runtimeMode = trim(result?.runtime_mode) || trim(state.runtimeMode) || 'advanced';
        if (window.parent && window.parent !== window) {
            // Target our own origin explicitly (paired with the receiver's
            // origin check) instead of broadcasting to any embedding page.
            const targetOrigin = window.location.origin === 'null'
                ? (window.parent?.location?.origin ?? '*')
                : window.location.origin;
            window.parent.postMessage({
                type: 'ouroboros:onboarding-complete',
                restart_required: restartRequired,
                runtime_mode: runtimeMode,
            }, targetOrigin);
            return;
        }
        if (window.pywebview?.api?.onboarding_finished) {
            // Desktop setup window: the launcher owns both this window and the
            // managed server process, so it closes the window and recycles the
            // server itself instead of asking the owner to restart the app.
            window.pywebview.api.onboarding_finished({
                ok: true,
                restart_required: restartRequired,
                runtime_mode: runtimeMode,
            });
            return;
        }
        // PLAIN BROWSER TAB. Nothing here owns the server process, so a receipt
        // that says a restart is required cannot be discharged by navigating —
        // and redirecting anyway drops the owner into an app running a runtime
        // mode DIFFERENT from the one they just chose, with nothing on screen
        // saying so. The other two shells each show this (the overlay's restart
        // card, the launcher's own recycle); this one used to be the only place
        // the flag was silently thrown away.
        if (restartRequired) {
            showBrowserRestartRequired(runtimeMode);
            return;
        }
        window.location.replace('/');
    }

    function showBrowserRestartRequired(runtimeMode) {
        state.saving = false;
        state.completedRestartMode = trim(runtimeMode) || 'advanced';
        render();
    }

    async function checkSaveStatus() {
        // Completion answered 503 `settings_save_timeout`: its body kept running
        // in the server past the shared writer bound, so whether the bytes
        // landed is UNKNOWN (`saved: null`) and a retry would be a second write.
        // Re-read the readiness probe the overlay itself gates on — 204 means a
        // startup-ready provider is on disk, i.e. the transaction landed — and
        // proceed exactly as a completion receipt would; otherwise stay open.
        state.error = '';
        render();
        let status = 0;
        try {
            const response = await fetch('/api/onboarding', {
                method: 'GET', headers: { Accept: 'application/json' },
            });
            status = response.status;
        } catch (error) {
            state.error = `Could not check the save status: ${String(error?.message || error)}. Try again in a moment.`;
            render();
            return;
        }
        if (status === 204) {
            state.saveUnknown = false;
            await agentsStep?.disposeForCompletion();
            agentsStep = null;
            // The receipt's `restart_required` never arrived: the boot-pinned
            // mode is what the page loaded with, so a changed choice is treated
            // as needing a restart — the honest side to err on.
            announceCompletion({
                runtime_mode: state.runtimeMode,
                restart_required: trim(state.runtimeMode) !== trim(INITIAL_STATE.runtimeMode),
            });
            return;
        }
        state.error = 'Setup is not complete yet — the save may still be running in the '
            + 'server. Check again in a moment; if it never completes, finish setup again.';
        render();
    }

    async function completeOnboardingAtomically(payload) {
        // The atomic completion (D-8): server-side fresh-install proof, the
        // structural provider gate, optional agent-preset compilation, a single
        // persist, then supervisor start.
        const response = await fetch(ONBOARDING_COMPLETE_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        // A body that will not parse is UNKNOWN, not empty. Substituting `{}`
        // made a 200 carrying HTML (a proxy error page, a login redirect) look
        // like a completion, and `{}` being truthy meant the caller announced
        // success with `restart_required` silently gone.
        let data = null;
        let parsed = true;
        try {
            data = await response.json();
        } catch (err) {
            parsed = false;
        }
        if (!data || typeof data !== 'object') { data = {}; parsed = false; }
        // ONE reader for both answers (typed refusal and success envelope); the
        // branches live in onboarding_agents_step.js so every one is node-tested.
        const answer = readCompletionAnswer({
            status: response.status, ok: response.ok, parsed, data,
        });
        if (answer.failure) {
            const error = new Error(answer.failure.message);
            Object.assign(error, answer.failure);
            throw error;
        }
        return answer.receipt;
    }

    async function saveWizardPayload(payload) {
        const result = await completeOnboardingAtomically(payload);
        await agentsStep?.disposeForCompletion();
        agentsStep = null;
        announceCompletion(result);
        return 'ok';
    }

    async function saveWizard({ skipPresets = false } = {}) {
        if (skipPresets) {
            state.skipSubscriptionPresets = true;
            await agentsStep?.setSkipPresets(true);
        }
        const providersError = validateProvidersStep();
        const modelsError = validateModelsStep();
        const reviewError = validateReviewStep();
        const budgetError = validateBudgetStep();
        agentsStep?.noteSaveAttempt?.();
        const subagentsError = agentsStep?.validateSubagents?.()?.[0] || '';
        const previewError = agentsStep && !agentsStep.generatedPreviewReady
            ? (agentsStep.previewPending
                ? 'Available subagents are still updating from your latest setup choices. Try Finish again in a moment.'
                : `Available subagents could not be refreshed from your latest setup choices${agentsStep.previewError ? `: ${agentsStep.previewError}` : '.'}`)
            : '';
        state.error = providersError || modelsError || reviewError || budgetError
            || subagentsError || previewError;
        if (state.error) {
            render();
            return;
        }
        state.saving = true;
        state.error = '';
        render();
        const payload = {
            // Observations are not authority: the endpoint re-proves eligibility.
            subscriptionsConnected: state.agentsConnected.length > 0,
            skipSubscriptionPresets: state.skipSubscriptionPresets,
            ...onboardingSettingsDraft({ state, providerFields: PROVIDER_FIELDS, budgetFields: BUDGET_FIELDS, modelSlots: MODEL_SLOTS, trim }),
            // Completion validates this visible draft and never replaces it.
            OUROBOROS_SUBAGENTS: agentsStep?.availableSubagents
                || state.availableSubagents,
        };
        try {
            await saveWizardPayload(payload);
        } catch (error) {
            // The wizard STAYS OPEN with the real reason. A typed preset refusal
            // wrote nothing, so the offered escape is honest: finish without the
            // agent defaults and keep everything editable in Settings.
            const notice = completionFailureNotice(error);
            state.saving = false;
            state.presetFailure = notice.canSkip ? { code: notice.code } : null;
            state.saveUnknown = Boolean(notice.saveUnknown);
            state.error = notice.text;
            render();
        }
    }

    function bindEvents() {
        if (state.completedRestartMode) {
            document.getElementById('open-app-btn')?.addEventListener('click', () => {
                window.location.replace('/');
            });
            return;
        }
        bindClearButtons();
        document.getElementById('back-btn')?.addEventListener('click', previousStep);
        document.getElementById('next-btn')?.addEventListener('click', () => {
            if (state.currentStep === 'summary') saveWizard();
            else nextStep();
        });
        document.getElementById('skip-presets-btn')?.addEventListener('click', () => {
            saveWizard({ skipPresets: true });
        });
        document.getElementById('check-save-btn')?.addEventListener('click', () => {
            checkSaveStatus();
        });
        if (state.currentStep === 'providers') bindProvidersStep();
        if (state.currentStep === 'agents') bindAgentsStep();
        if (state.currentStep === 'models') bindModelsStep();
        if (state.currentStep === 'review_mode') bindReviewModeStep();
        if (state.currentStep === 'budget') bindBudgetStep();
        syncCurrentStepActionState();
    }

    applyModelDefaults(false);
    render();
})();
