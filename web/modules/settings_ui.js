import { renderPageHeader, renderSegmentedField, renderTabStrip } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { renderAgentAccountsSection, renderAgentsServiceBanner } from './harness_accounts.js';
import { renderReviewerSlotsSection } from './reviewer_slots.js';
import { renderSubagentsSection } from './subagents_settings.js';

// Reads as a sequence: keys → secrets → which API models → who among the agents
// does what → behavior → technical. "Agents", not "Coding agents" (D-10): the
// same subscriptions build presentations and run arbitrary tasks, so the
// narrower word named only one of their uses.
const SETTINGS_TABS = [
    { value: 'providers', label: 'Providers' },
    { value: 'secrets', label: 'Secrets' },
    { value: 'models', label: 'Models' },
    { value: 'agents', label: 'Agents' },
    { value: 'behavior', label: 'Behavior' },
    { value: 'advanced', label: 'Advanced' },
    { value: 'about', label: 'About' },
];
// Guard markers: renderTabStrip emits behavior/advanced tabs at runtime.

const MODEL_CARDS = [
    ['Main', 'Primary reasoning model.', 's-model', 's-local-main', 'google/gemini-3.8-flash'],
    ['Light', 'Fast summaries, lightweight internal work, reflections, and the default Fast scout. Empty uses Main.', 's-model-light', 's-local-light', 'openai/gpt-5.6-luna'],
    ['Vision', 'Caption and VLM lane. Empty uses Main.', 's-model-vision', '', ''],
    ['Consciousness', 'High-horizon background consciousness. Empty uses Main.', 's-model-consciousness', 's-local-consciousness', ''],
    ['Fallback', 'Resilience and degraded path (comma-separated chain).', 's-model-fallback', 's-local-fallback', 'openai/gpt-5.6-luna'],
];

// 6.3: Review and Scope Review efforts moved to per-slot dropdowns in
// Agents → Review lanes. Behavior keeps the surface-level lanes.
const EFFORT_FIELDS = [
    ['s-effort-task', 'Task / Chat', 'medium'],
    ['s-effort-evolution', 'Evolution', 'high'],
    ['s-effort-deep-self-review', 'Deep Self-Review', 'high'],
    ['s-effort-consciousness', 'Consciousness', 'high'],
];

function providerCard({ id, title, icon, hint, body, open = false }) {
    return `
        <details class="settings-provider-card" data-provider-card="${id}" ${open ? 'open' : ''}>
            <summary>
                <div class="settings-provider-title">
                    ${icon ? `<img src="${icon}" alt="" class="settings-provider-icon">` : ''}
                    <span>${title}</span>
                </div>
                <span class="settings-provider-hint">${hint || ''}</span>
            </summary>
            <div class="settings-provider-body">
                ${body}
            </div>
        </details>
    `;
}

function secretField({ id, settingKey, label, placeholder }) {
    return `
        <div class="form-field">
            <label>${label}</label>
            <div class="secret-input-row">
                <input id="${id}" data-secret-setting="${settingKey}" class="secret-input" type="password" placeholder="${placeholder}">
                <button type="button" class="btn btn-default secret-toggle" data-target="${id}">Show</button>
                <button type="button" class="btn btn-default secret-clear" data-target="${id}">Clear</button>
            </div>
        </div>
    `;
}

function plainField({ id, label, placeholder }) {
    return `<div class="form-field"><label>${label}</label><input id="${id}" placeholder="${placeholder}"></div>`;
}

const PROVIDER_CARDS = [
    {
        id: 'openrouter', title: 'OpenRouter', icon: '/static/providers/openrouter.ico', hint: 'Default multi-model router', open: true,
        fields: [{ id: 's-openrouter', settingKey: 'OPENROUTER_API_KEY', label: 'OpenRouter API Key', placeholder: 'sk-or-...' }],
        // testInputs must cover BOTH the secret and the plain fields of the card:
        // the probe runs against what is typed, before anything is saved.
        testProvider: 'openrouter', testInputs: { 's-openrouter': 'OPENROUTER_API_KEY' },
    },
    {
        id: 'openai', title: 'OpenAI', icon: '/static/providers/openai.svg', hint: 'Official OpenAI API',
        fields: [{ id: 's-openai', settingKey: 'OPENAI_API_KEY', label: 'OpenAI API Key', placeholder: 'sk-...' }],
        testProvider: 'openai', testInputs: { 's-openai': 'OPENAI_API_KEY' },
        note: 'Use model values like <code>openai::gpt-5.6-terra</code> in the Models tab to route models directly here. If OpenRouter is absent and the shipped defaults are still untouched, Ouroboros auto-remaps them to official OpenAI defaults.',
    },
    {
        id: 'compatible', title: 'OpenAI Compatible', icon: '/static/providers/openai-compatible.svg', hint: 'Custom OpenAI-style endpoint',
        fields: [
            { id: 's-openai-compatible-key', settingKey: 'OPENAI_COMPATIBLE_API_KEY', label: 'API Key', placeholder: 'Compatible provider key' },
            { id: 's-openai-compatible-base-url', label: 'Base URL', placeholder: 'https://provider.example/v1' },
        ],
        testProvider: 'openai-compatible',
        testInputs: {
            's-openai-compatible-key': 'OPENAI_COMPATIBLE_API_KEY',
            's-openai-compatible-base-url': 'OPENAI_COMPATIBLE_BASE_URL',
        },
        note: 'Use this card for custom base URLs. Built-in web search only works with the official OpenAI Responses API, so keep <code>OPENAI_BASE_URL</code> empty when you want <code>web_search</code>.',
    },
    {
        // advanced: rendered inside the collapsed "More providers" section.
        // Inputs stay mounted either way — settings.js applyInputValue has no
        // null guard, so every input id must always exist in the DOM.
        id: 'cloudru', title: 'Cloud.ru Foundation Models', icon: '/static/providers/cloudru.svg', hint: 'Cloud.ru OpenAI-compatible runtime', advanced: true,
        fields: [
            { id: 's-cloudru-key', settingKey: 'CLOUDRU_FOUNDATION_MODELS_API_KEY', label: 'API Key', placeholder: 'Cloud.ru Foundation Models API key' },
            { id: 's-cloudru-base-url', label: 'Base URL', placeholder: 'https://foundation-models.api.cloud.ru/v1' },
        ],
        testProvider: 'cloudru',
        testInputs: {
            's-cloudru-key': 'CLOUDRU_FOUNDATION_MODELS_API_KEY',
            's-cloudru-base-url': 'CLOUDRU_FOUNDATION_MODELS_BASE_URL',
        },
    },
    {
        id: 'minimax', title: 'MiniMax', icon: '', hint: 'Direct regional OpenAI-compatible runtime', advanced: true,
        fields: [
            { id: 's-minimax-key', settingKey: 'MINIMAX_API_KEY', label: 'API Key', placeholder: 'MiniMax API key' },
            { id: 's-minimax-region', label: 'Region', placeholder: 'global_en or cn_zh' },
        ],
        testProvider: 'minimax',
        testInputs: { 's-minimax-key': 'MINIMAX_API_KEY', 's-minimax-region': 'MINIMAX_REGION' },
        note: 'Use <code>minimax::MiniMax-M3</code> or <code>minimax::MiniMax-M2.7</code> in the Models tab. Leave Region empty for <code>global_en</code>; use <code>cn_zh</code> for the China endpoint.',
    },
    {
        id: 'deepseek', title: 'DeepSeek', icon: '', hint: 'Direct OpenAI-compatible runtime (v4 family)', advanced: true,
        fields: [
            { id: 's-deepseek-key', settingKey: 'DEEPSEEK_API_KEY', label: 'API Key', placeholder: 'sk-...' },
        ],
        testProvider: 'deepseek',
        testInputs: { 's-deepseek-key': 'DEEPSEEK_API_KEY' },
        note: 'Use <code>deepseek::deepseek-v4-pro</code> or <code>deepseek::deepseek-v4-flash</code> in the Models tab. Blocking deep/scope review in Max context mode additionally needs the owner 1M-window acknowledgement.',
    },
    {
        id: 'gigachat', title: 'GigaChat', icon: '/static/providers/gigachat.svg', hint: 'Sber GigaChat via the gigachat library', advanced: true,
        fields: [
            { id: 's-gigachat-credentials', settingKey: 'GIGACHAT_CREDENTIALS', label: 'Authorization Key', placeholder: 'Base64 client_id:secret (OAuth)' },
            { id: 's-gigachat-scope', label: 'Scope', placeholder: 'GIGACHAT_API_PERS' },
            { id: 's-gigachat-user', label: 'User (basic auth, optional)', placeholder: 'username' },
            { id: 's-gigachat-password', settingKey: 'GIGACHAT_PASSWORD', label: 'Password (basic auth, optional)', placeholder: 'password' },
            { id: 's-gigachat-base-url', label: 'Base URL', placeholder: 'https://api.giga.chat/v1' },
            { id: 's-gigachat-verify-ssl', label: 'Verify SSL Certs', placeholder: 'true / false' },
        ],
        testProvider: 'gigachat',
        testInputs: {
            's-gigachat-credentials': 'GIGACHAT_CREDENTIALS',
            's-gigachat-scope': 'GIGACHAT_SCOPE',
            's-gigachat-user': 'GIGACHAT_USER',
            's-gigachat-password': 'GIGACHAT_PASSWORD',
            's-gigachat-base-url': 'GIGACHAT_BASE_URL',
            's-gigachat-verify-ssl': 'GIGACHAT_VERIFY_SSL_CERTS',
        },
        note: 'Use model values like <code>gigachat::GigaChat-2-Max</code> in the Models tab to route directly through GigaChat. Authenticate with either an Authorization Key (OAuth, scope <code>GIGACHAT_API_PERS</code>, <code>GIGACHAT_API_B2B</code>, or <code>GIGACHAT_API_CORP</code>) or User + Password.',
    },
    {
        id: 'anthropic', title: 'Anthropic', icon: '/static/providers/anthropic.png', hint: 'Direct Anthropic API access',
        fields: [{ id: 's-anthropic', settingKey: 'ANTHROPIC_API_KEY', label: 'Anthropic API Key', placeholder: 'sk-ant-...' }],
        testProvider: 'anthropic', testInputs: { 's-anthropic': 'ANTHROPIC_API_KEY' },
        note: 'Use model values like <code>anthropic::claude-sonnet-5</code> in the Models tab to route models directly through Anthropic.',
    },
];

// {backend provider id: {DOM input id: settings key}} — settings.js reads the
// live input values through this map to build the unsaved-credential overrides.
export const PROVIDER_TEST_INPUTS = Object.fromEntries(
    PROVIDER_CARDS.filter((card) => card.testProvider).map((card) => [card.testProvider, card.testInputs]),
);

function providerSettingsCard(spec) {
    const fields = (spec.fields || [])
        .map((field) => field.settingKey ? secretField(field) : plainField(field))
        .join('');
    const test = spec.testProvider ? `
            <div class="settings-action-row">
                <span class="settings-inline-status" data-provider-test-status="${spec.testProvider}" role="status" aria-live="polite" aria-atomic="true"></span>
                <button type="button" class="btn btn-default" data-provider-test="${spec.testProvider}" title="Sends one short model request. Provider charges may apply.">Test</button>
            </div>
        ` : '';
    return providerCard({
        id: spec.id,
        title: spec.title,
        icon: spec.icon,
        hint: spec.hint,
        open: spec.open,
        body: `<div class="form-row">${fields}</div>${test}${spec.note ? `<div class="settings-inline-note">${spec.note}</div>` : ''}`,
    });
}

function modelCard({ title, copy, inputId, toggleId, defaultValue }) {
    const toggle = toggleId ? `<label class="local-toggle"><input type="checkbox" id="${toggleId}"> Local</label>` : '';
    return `
        <div class="settings-model-card">
            <div class="settings-model-header">
                <div>
                    <h4>${title}</h4>
                    <p>${copy}</p>
                </div>
                ${toggle}
            </div>
            <div class="model-picker" data-model-picker>
                <input
                    id="${inputId}"
                    value="${defaultValue}"
                    autocomplete="off"
                    spellcheck="false"
                >
                <div class="model-picker-results" hidden></div>
            </div>
        </div>
    `;
}

// The owner-facing subset of ouroboros/config.py EFFORT_SCALE: `minimal` is a
// valid runtime tier (bench adapters / agent-side switch_model use it) but is
// deliberately NOT offered as an owner slot default — sub-`low` thinking is a
// per-call tactical choice, not a standing configuration. xhigh/max/ultra adapt
// down to each route's real ceiling (exact-route request-wire recovery on API
// routes, per-model resolution on delegated ones); the adaptation is disclosed
// in usage, and a cold route whose provider rejects without naming supported
// tiers remains the PR-disclosed limit of the two-send recovery rail.
const EFFORT_OPTIONS = [
    { value: 'none', label: 'None' },
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
    { value: 'xhigh', label: 'X-High' },
    { value: 'max', label: 'Max' },
    { value: 'ultra', label: 'Ultra' },
];

function effortField({ id, label, defaultValue }) {
    return `
        <div class="settings-effort-card">
            <label>${label}</label>
            <input id="${id}" type="hidden" value="${defaultValue}">
            ${renderSegmentedField({ target: id, options: EFFORT_OPTIONS })}
        </div>
    `;
}

export const SECRET_KEYS = [
    ['OPENROUTER_API_KEY', 'OpenRouter API Key', 'sk-or-...'],
    ['OPENAI_API_KEY', 'OpenAI API Key', 'sk-...'],
    ['OPENAI_COMPATIBLE_API_KEY', 'OpenAI-compatible API Key', 'Compatible provider key'],
    ['CLOUDRU_FOUNDATION_MODELS_API_KEY', 'Cloud.ru Foundation Models API Key', 'Cloud.ru key'],
    ['GIGACHAT_CREDENTIALS', 'GigaChat Authorization Key', 'Base64 client_id:secret'],
    ['GIGACHAT_PASSWORD', 'GigaChat Password (basic auth)', 'password'],
    ['ANTHROPIC_API_KEY', 'Anthropic API Key', 'sk-ant-...'],
    ['MINIMAX_API_KEY', 'MiniMax API Key', 'MiniMax key'],
    ['DEEPSEEK_API_KEY', 'DeepSeek API Key', 'sk-...'],
    ['GITHUB_TOKEN', 'GitHub Token', 'ghp_...'],
    ['OUROBOROS_NETWORK_PASSWORD', 'Network Password', 'Required for LAN/Docker binds'],
];

function secretSettingsSection() {
    return `
        <section class="settings-card">
            <h3>Stored Secrets</h3>
            <div class="settings-section-copy">
                Central place for API keys, bridge tokens, passwords, and future skill-requested secrets.
                Skills only receive grant-only keys after explicit human approval.
            </div>
            <div class="form-grid two">
                ${SECRET_KEYS.map(([key, label, placeholder]) => secretField({
                    id: `s-secret-${key.toLowerCase().replace(/_/g, '-')}`,
                    settingKey: key,
                    label,
                    placeholder,
                })).join('')}
            </div>
        </section>
        <section class="settings-card">
            <h3>Requested By Skills</h3>
            <div class="settings-section-copy">
                Secrets requested by installed skills appear here only when a skill asks for them.
            </div>
            <div id="skill-requested-secrets" class="settings-secret-list">
                <div class="muted">No skill-requested secrets.</div>
            </div>
        </section>
        <section class="settings-card">
            <div class="settings-card-head">
                <div>
                    <h3>Custom Keys</h3>
                    <div class="settings-section-copy">
                        Optional key/value storage for future skills. Use uppercase names such as <code>SLACK_WEBHOOK_URL</code>.
                    </div>
                </div>
                <button type="button" class="btn btn-default btn-sm" id="btn-add-custom-secret">Add custom key</button>
            </div>
            <div id="custom-secrets-list" class="settings-secret-list settings-custom-secret-list"></div>
        </section>
    `;
}

export function renderSettingsPage() {
    return `
        ${renderPageHeader({
            title: 'Settings',
            icon: PAGE_ICONS.settings,
            description: 'Configure providers, secrets, models, behavior, source control, and runtime controls.',
            tabsHtml: `
                <div class="settings-tabs-bar">
                    <button type="button" class="settings-mobile-back" data-settings-back hidden>Settings</button>
                    ${renderTabStrip({
                        items: SETTINGS_TABS,
                        active: 'providers',
                        dataAttr: 'data-settings-tab',
                        ariaLabel: 'Settings sections',
                        stripClass: 'settings-tabs',
                        tabClass: 'settings-tab',
                    })}
                </div>
            `,
        })}
        <div class="settings-shell">
            <div class="settings-scroll scroll-fade-y">
                <section class="settings-panel active" data-settings-panel="providers">
                    <div class="settings-section-copy">
                        Configure remote providers and the optional network gate. Secret fields now have explicit
                        <code>Clear</code> actions so masked values can be removed intentionally.
                    </div>
                    ${PROVIDER_CARDS.filter((card) => !card.advanced).map(providerSettingsCard).join('')}
                    <details class="settings-more-providers" id="settings-more-providers">
                        <summary>
                            <span class="settings-provider-title"><span>More providers</span></span>
                            <span class="settings-provider-hint">Cloud.ru Foundation Models, MiniMax, DeepSeek, and GigaChat</span>
                        </summary>
                        <div class="settings-more-providers-body">
                            ${PROVIDER_CARDS.filter((card) => card.advanced).map(providerSettingsCard).join('')}
                        </div>
                    </details>
                    <!-- Agent accounts moved to the Agents tab (D-10): they are
                         not a remote API provider, and the owner had to hunt for
                         the Add-account button under an unrelated row. -->
                    <div class="form-section compact">
                        <h3>Legacy Compatibility</h3>
                        <div class="form-row">
                            <div class="form-field">
                                <label>Legacy OpenAI Base URL</label>
                                <input id="s-openai-base-url" placeholder="https://api.openai.com/v1 or compatible endpoint">
                            </div>
                        </div>
                        <div class="settings-inline-note">Backward-compatibility escape hatch for older installs. For new custom providers, use the dedicated <code>OpenAI Compatible</code> card instead.</div>
                    </div>
                    <div class="form-section compact">
                        <h3>Network Gate</h3>
                        <div class="form-row">${secretField({
                            id: 's-network-password',
                            settingKey: 'OUROBOROS_NETWORK_PASSWORD',
                            label: 'Network Password (optional)',
                            placeholder: 'Leave blank to keep the network surface open',
                        })}</div>
                        <div class="form-row">
                            <div class="form-field">
                                <label>Server Bind Host</label>
                                <input id="s-server-host" placeholder="127.0.0.1 or 0.0.0.0">
                                <div class="settings-inline-note">Use <code>127.0.0.1</code> for this machine only. Use <code>0.0.0.0</code> for LAN/Docker access with a Network Password in the same save. Specific LAN IP binds are manual/env-only.</div>
                            </div>
                        </div>
                        <div class="settings-inline-note">Adds a password wall only for non-localhost app and API access. If you expose Ouroboros on LAN or Docker, set a password before sharing the URL.</div>
                        <div id="settings-lan-hint" class="settings-lan-hint" hidden></div>
                    </div>
                </section>

                <section class="settings-panel" data-settings-panel="secrets">
                    ${secretSettingsSection()}
                </section>

                <section class="settings-panel" data-settings-panel="models">
                    <div class="form-section">
                        <h3>Model Routing</h3>
                        <div class="settings-section-copy">
                            These fields are cloud model IDs. Enable <code>Local</code> to route that model
                            through the GGUF server configured in Advanced.
                        </div>
                        <div class="settings-action-row">
                            <span id="settings-model-catalog-status" class="settings-inline-status" role="status" aria-live="polite" aria-atomic="true">Model catalog is optional and failure-tolerant.</span>
                            <button type="button" class="btn btn-default" id="btn-refresh-model-catalog">Refresh Model Catalog</button>
                        </div>
                        <div class="settings-model-grid">
                            ${MODEL_CARDS.map(([title, copy, inputId, toggleId, defaultValue]) => modelCard({ title, copy, inputId, toggleId, defaultValue })).join('')}
                        </div>
                    </div>

                    <!-- Review lanes and Delegation moved to the Agents tab
                         (D-10): they answer "who does the work", not "which API
                         model id". One capability, one section — no control here
                         duplicates one there. The deep self-review reviewer is a
                         Review lanes row too (R7); its former model field's key,
                         OUROBOROS_MODEL_DEEP_SELF_REVIEW, survives only as the
                         backend's invisible migration source for that row. -->

                    <div class="form-section">
                        <h3>Other Model Slots</h3>
                        <div class="form-grid two">
                            <div class="form-field">
                                <label>Web Search Model</label>
                                <input id="s-websearch-model" placeholder="gpt-5.2">
                                <div class="settings-inline-note">OpenAI model for <code>web_search</code>. Requires <code>OPENAI_API_KEY</code> and an empty Legacy Base URL.</div>
                            </div>
                        </div>
                    </div>
                </section>

                <section class="settings-panel" data-settings-panel="agents">
                    <div class="settings-section-copy">
                        The agents Ouroboros delegates to, in dependency order: the subscription
                        accounts they run on, the Available subagents built from those accounts
                        and API routes, and the review lanes that reference those subagents.
                        API keys stay in Providers; global model lanes stay in Models, while
                        each Available subagent owns its route here.
                    </div>
                    <!-- ONE service banner for the whole tab: the single place a
                         daemon or runtime problem is explained, instead of the
                         scattering of "(not in discovery)" the owner reported. -->
                    ${renderAgentsServiceBanner()}
                    ${renderAgentAccountsSection()}
                    ${renderSubagentsSection()}
                    ${renderReviewerSlotsSection()}
                </section>

                <section class="settings-panel" data-settings-panel="behavior">
                    <div class="form-section">
                        <h3>Reasoning Effort</h3>
                        <div class="settings-section-copy">Controls how deeply the model thinks per task type. Higher effort = slower but more thorough.</div>
                        <div class="settings-effort-grid">
                            ${EFFORT_FIELDS.map(([id, label, defaultValue]) => effortField({ id, label, defaultValue })).join('')}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Review Enforcement</h3>
                        <div class="settings-section-copy"><code>Advisory</code> keeps review visible but non-blocking. <code>Blocking</code> stops commits and reviewed-skill activation when critical findings remain unresolved.</div>
                        <div class="settings-effort-card">
                            <label>Enforcement Mode</label>
                            <input id="s-review-enforcement" type="hidden" value="advisory">
                            ${renderSegmentedField({
                                target: 's-review-enforcement',
                                modifier: 'data-enforcement-group',
                                options: [
                                    { value: 'advisory', label: 'Advisory' },
                                    { value: 'blocking', label: 'Blocking' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Task Result Review</h3>
                        <div class="settings-section-copy">Auto and Required run the root-owned review panel for queued/headless work and substantive direct results. Pure conversation and routing controls are skipped; Required additionally enforces the selected Advisory or Blocking improvement policy.</div>
                        <div class="settings-effort-card">
                            <label>Task Result Review</label>
                            <input id="s-task-review-mode" type="hidden" value="auto">
                            ${renderSegmentedField({
                                target: 's-task-review-mode',
                                modifier: 'data-task-review-group',
                                options: [
                                    { value: 'off', label: 'Off' },
                                    { value: 'auto', label: 'Auto' },
                                    { value: 'required', label: 'Required' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Max Review Cycles</h3>
                        <div class="settings-section-copy">One shared cap on paid review cycles per task for plan review, task acceptance (improvement passes = cycles &minus; 1), the commit gate (paid triad+scope cycles per root task, shared across the whole task tree; an unchanged diff never buys a new review after a recorded verdict block regardless of this number &mdash; blocking enforcement refuses it for free, while a pure advisory line records no verdict blocks and its no-new-spend guarantee is the free replay after exhaustion, with a loud durable disclosure) and skill review (paid panel runs per root task or per manual snapshot; identical snapshots replay free). <code>&infin;</code> removes the cap; the task's own deadline, budget and lifecycle rails still bind.</div>
                        <div class="settings-effort-card">
                            <label>Max Review Cycles</label>
                            <input id="s-review-max-cycles" type="hidden" value="2">
                            ${renderSegmentedField({
                                target: 's-review-max-cycles',
                                modifier: 'data-review-cycles-group',
                                options: [
                                    { value: '1', label: '1' },
                                    { value: '2', label: '2' },
                                    { value: '3', label: '3' },
                                    { value: '5', label: '5' },
                                    { value: 'unlimited', label: '\u221E' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Image Input</h3>
                        <div class="settings-section-copy">Auto sends images inline to vision-capable models and captions them for blind models. Caption always uses text captions; Inline refuses caption fallback; Off emits placeholders.</div>
                        <div class="settings-effort-card">
                            <label>Image Input Mode</label>
                            <input id="s-image-input-mode" type="hidden" value="auto">
                            ${renderSegmentedField({
                                target: 's-image-input-mode',
                                modifier: 'data-image-input-group',
                                options: [
                                    { value: 'auto', label: 'Auto' },
                                    { value: 'caption', label: 'Caption' },
                                    { value: 'inline', label: 'Inline' },
                                    { value: 'off', label: 'Off' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Skills</h3>
                        <div class="settings-section-copy">
                            Closed-loop skill development can auto-grant the keys and host permissions a skill declares after a fresh executable review for the current content hash.
                            Leave this off when every skill permission should require a separate human approval.
                        </div>
                        <label class="local-toggle" title="Applies only after a fresh executable skill review and only to manifest-declared grants for that exact content hash.">
                            <input type="checkbox" id="s-auto-grant-reviewed-skills">
                            Auto-grant reviewed skills' keys and permissions
                        </label>
                    </div>

                    <div class="form-section">
                        <h3>Context Mode</h3>
                        <div class="settings-section-copy">
                            Working-context size profile (separate axis from Runtime Mode and Review Enforcement).
                            <code>Max</code> inlines ARCHITECTURE and DEVELOPMENT in full &mdash; for ~1M-context models (today's behavior).
                            <code>Low</code> fits ~200K / local models: ARCHITECTURE becomes a navigation map (read full sections on demand), DEVELOPMENT stays full for normal runnable tasks unless a structured non-development caller opts out, and memory compacts sooner. It never changes the model or reasoning effort, and never lowers the review context floor.
                            <br><strong>Human controlled:</strong> saved via the owner endpoint; saves immediately (no restart), and lowering to Low requires Ouroboros to be idle.
                        </div>
                        <div class="settings-effort-card">
                            <label>Context Mode</label>
                            <input id="s-context-mode" type="hidden" value="max">
                            ${renderSegmentedField({
                                target: 's-context-mode',
                                title: 'Saves immediately; no restart required. Lowering to Low requires Ouroboros to be idle.',
                                options: [
                                    { value: 'low', label: 'Low' },
                                    { value: 'max', label: 'Max' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Prompt Cache TTL</h3>
                        <div class="settings-section-copy">
                            How long provider prompt caches keep this agent's stable context warm (Anthropic-family routes; other providers manage caching implicitly).
                            <code>1h</code> keeps the large stable prefix cached across long waits and review cycles &mdash; cache writes bill at 2&times; base input instead of 1.25&times;, but one wait longer than 5 minutes already pays that back on big contexts.
                            <code>5m</code> is the provider default tier as an explicit value; <code>Default</code> sends bare markers (provider default, callers may still declare their own TTL).
                            One honest global: it applies to every lane, including review and safety prompts.
                        </div>
                        <div class="settings-effort-card">
                            <label>Prompt Cache TTL</label>
                            <input id="s-prompt-cache-ttl" type="hidden" value="1h">
                            ${renderSegmentedField({
                                target: 's-prompt-cache-ttl',
                                options: [
                                    { value: 'default', label: 'Default' },
                                    { value: '5m', label: '5m' },
                                    { value: '1h', label: '1h' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Safety Supervisor</h3>
                        <div class="settings-section-copy">
                            Coverage of the LLM safety-supervisor layer (a separate axis from Runtime Mode).
                            <code>Full</code> &mdash; every guarded tool call gets the LLM safety check.
                            <code>Light</code> keeps the LLM check only for integration-policy tools; conditional shell/verify fall to the deterministic guards. Light is the default for new DESKTOP setups (authored by the first-run wizard); existing installs, web and Docker keep Full.
                            <code>Off</code> makes no LLM safety calls. In every mode the deterministic registry sandbox, protected-path policy, and light-mode guards STAY ON &mdash; the LLM supervisor is a layer, not the floor. Lowering coverage emits a durable audit event per waved-through call.
                            <br><strong>Human controlled:</strong> saved via the owner endpoint (the agent cannot lower its own supervision); applies on the next task.
                        </div>
                        <div class="settings-effort-card">
                            <label>Safety Supervisor</label>
                            <input id="s-safety-mode" type="hidden" value="full">
                            ${renderSegmentedField({
                                target: 's-safety-mode',
                                title: 'Owner-only. Lowering coverage prompts for confirmation.',
                                options: [
                                    { value: 'full', label: 'Full' },
                                    { value: 'light', label: 'Light' },
                                    { value: 'off', label: 'Off' },
                                ],
                            })}
                            <div id="s-safety-skip-counter" class="settings-section-copy"></div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Update Channel</h3>
                        <div class="settings-section-copy">
                            Chooses which official branch Update checks. Your local work branch stays <code>ouroboros</code>.
                            <code>Stable</code> follows released code on <code>main</code> (default),
                            <code>QA</code> follows <code>ouroboros-stable</code>, and
                            <code>Development</code> follows <code>ouroboros</code>.
                        </div>
                        <div class="settings-effort-card">
                            <label>Official Update Source</label>
                            <input id="s-update-channel" type="hidden" value="stable">
                            ${renderSegmentedField({
                                target: 's-update-channel',
                                modifier: 'data-update-channel-group',
                                title: 'Applies immediately; no restart required.',
                                options: [
                                    { value: 'stable', label: 'Stable' },
                                    { value: 'qa', label: 'QA' },
                                    { value: 'development', label: 'Development' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Runtime Mode</h3>
                        <div class="settings-section-copy">
                            Separate axis from Review Enforcement. Controls how far Ouroboros is allowed to self-modify.
                            <code>Light</code> blocks repo self-modification but allows reviewed + enabled skills to run.
                            <code>Advanced</code> is the default &mdash; self-modify the evolutionary layer; protected core/contract/release files stay guarded by the shared runtime-mode policy.
                            <code>Pro</code> can edit protected core/contract/release surfaces, but commits still go through the normal triad + scope review gate; Advanced remains limited to the evolutionary layer.
                            <br><strong>Human controlled:</strong> desktop builds ask the launcher for native confirmation before saving a mode change.
                            Web/Docker sessions save mode changes through the owner endpoint; the new mode takes effect after restart.
                        </div>
                        <div class="settings-effort-card">
                            <label>Runtime Mode</label>
                            <input id="s-runtime-mode" type="hidden" value="advanced">
                            ${renderSegmentedField({
                                target: 's-runtime-mode',
                                modifier: 'data-runtime-mode-group',
                                title: 'Runtime mode changes require native launcher confirmation and restart.',
                                options: [
                                    { value: 'light', label: 'Light' },
                                    { value: 'advanced', label: 'Advanced' },
                                    { value: 'pro', label: 'Pro' },
                                ],
                            })}
                        </div>
                    </div>

                    <!-- Allow Mutative Subagents moved to Agents → Delegation: one
                         subagent story in one place, and two controls over one
                         setting would have carried two drafts. -->

                    <div class="form-section">
                        <h3>Post-Task Self-Evolution</h3>
                        <div class="settings-section-copy">
                            After an eligible task, Ouroboros can optionally run one reviewed self-improvement cycle: the worker asks a light model whether to promote a backlog item, writes a durable request, and the supervisor starts a one-shot campaign later on an idle tick if all gates pass.
                            <br><strong>Human controlled:</strong> the agent cannot self-enable this (shell/browser/settings self-elevation is blocked). These controls apply on the next task.
                        </div>
                        <div class="settings-effort-card">
                            <label>Self-Improvement Trigger</label>
                            <input id="s-post-task-evolution-mode" type="hidden" value="off">
                            ${renderSegmentedField({
                                target: 's-post-task-evolution-mode',
                                options: [
                                    { value: 'off', label: 'Off' },
                                    { value: 'llm', label: 'After Each Task (LLM decides)' },
                                    { value: 'every_n', label: 'Every N Tasks' },
                                ],
                            })}
                            <div class="settings-inline-note"><strong>Counts every eligible task, including trivial chats.</strong> <code>Every N=1</code> means Ouroboros considers self-improvement after every task, then runs the actual cycle later on an idle supervisor tick.</div>
                        </div>
                        <div class="form-row">
                            <div class="form-field">
                                <div data-evo-every-n-row>
                                <label>Every N Tasks</label>
                                <input id="s-evo-cadence-n" type="number" min="1" step="1" placeholder="3">
                                <div class="settings-inline-note">Visible only when Self-Improvement Trigger = Every N Tasks.</div>
                                </div>
                            </div>
                            <div class="form-field">
                                <label>Per-Cycle Budget Reserve (USD)</label>
                                <input id="s-evo-budget" placeholder="0">
                                <div class="settings-inline-note">Minimum remaining global budget required to start a post-task cycle. <code>0</code> = rely on the normal gates. Running cycles still inherit the global per-task hard cost cap and the supervisor's reserved-budget floor.</div>
                            </div>
                        </div>
                        <div class="form-field">
                            <label>Standing Objective (optional)</label>
                            <input id="s-evo-objective" placeholder="(none) — e.g. prioritize test coverage and latency">
                            <div class="settings-inline-note">Optional steer appended to every evolution cycle objective. It never overrides the LLM-first promotion; leave empty for pure LLM choice.</div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Background Cognition</h3>
                        <div class="settings-section-copy">
                            Cadence for Ouroboros's background cognition loop. These values are read at startup; save them, then restart for the new timing to take effect.
                        </div>
                        <div class="form-row">
                            <div class="form-field">
                                <label>BG Wakeup Min (sec)</label>
                                <input id="s-bg-wakeup-min" type="number" min="1" step="1" placeholder="30">
                            </div>
                            <div class="form-field">
                                <label>BG Wakeup Max (sec)</label>
                                <input id="s-bg-wakeup-max" type="number" min="1" step="1" placeholder="7200">
                            </div>
                            <div class="form-field">
                                <label>BG Max Rounds</label>
                                <input id="s-bg-max-rounds" type="number" min="1" step="1" placeholder="10">
                            </div>
                        </div>
                        <div class="settings-inline-note"><strong>Applies after restart:</strong> BG Wakeup Min/Max and BG Max Rounds are read when the background cognition loop starts.</div>
                    </div>

                    <div class="form-section">
                        <h3>External Skills Repo</h3>
                        <div class="settings-section-copy">
                            Optional EXTRA discovery path on top of the in-data-plane
                            <code>data/skills/{native,clawhub,external}/</code> tree.
                            Ouroboros scans this for additional skill packages without
                            cloning or pulling them. Leave empty to use only the data plane.
                        </div>
                        <div class="form-row">
                            <div class="form-field">
                                <label>Skills Repo Path</label>
                                <input id="s-skills-repo-path" placeholder="~/Ouroboros/skills or /absolute/path/to/skills">
                                <div class="settings-inline-note">Absolute or <code>~</code>-prefixed path. Ouroboros never clones/pulls this directory — you manage it yourself.</div>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>ClawHub Marketplace</h3>
                        <div class="settings-section-copy">
                            Always-on surface for installing community skills from
                            <a href="https://clawhub.ai" target="_blank" rel="noopener">clawhub.ai</a>.
                            The Skills page exposes a Marketplace tab; every install is
                            staged, OpenClaw frontmatter is translated into the
                            Ouroboros manifest shape, and the standard tri-model review runs
                            automatically before the skill becomes executable. Plugins (Node)
                            are filtered out — only skill packages are installable.
                        </div>
                        <div class="form-row">
                            <div class="form-field">
                                <label>Registry URL</label>
                                <input id="s-clawhub-registry-url" placeholder="https://clawhub.ai/api/v1">
                                <div class="settings-inline-note">Override only for self-hosted mirrors. Hostname must be <code>clawhub.ai</code> or localhost.</div>
                            </div>
                        </div>
                    </div>
                </section>

                <section class="settings-panel" data-settings-panel="advanced">
                    <div class="form-section">
                        <div class="settings-card-head">
                            <div>
                                <h3>MCP Servers</h3>
                                <div class="settings-section-copy">
                                    External Model Context Protocol tool servers. MCP is a base-runtime client:
                                    it borrows tools from trusted HTTP/SSE servers or local stdio processes and exposes them as non-core
                                    <code>mcp_&lt;server&gt;__&lt;tool&gt;</code> tools after refresh. Changes are hot-reloadable.
                                    Treat server descriptions and results as untrusted third-party data.
                                </div>
                            </div>
                            <div class="settings-toolbar">
                                <button type="button" class="btn btn-default btn-sm" id="btn-mcp-add-server">Add server</button>
                                <button type="button" class="btn btn-default btn-sm" id="btn-mcp-refresh-all">Refresh all</button>
                            </div>
                        </div>
                        <div class="form-grid two">
                            <label class="local-toggle">
                                <input type="checkbox" id="s-mcp-enabled">
                                Enable MCP client
                            </label>
                            <div class="form-field">
                                <label>Per-tool timeout (s)</label>
                                <input id="s-mcp-tool-timeout" type="number" min="1" value="60">
                            </div>
                        </div>
                        <div id="mcp-global-status" class="settings-inline-status">Checking MCP status…</div>
                        <div id="mcp-servers-list" class="mcp-servers-list"></div>
                    </div>

                    <div class="form-section">
                        <h3>Source Control</h3>
                        <div class="settings-section-copy">Repository metadata for GitHub integration. Tokens live in Secrets; this is not secret.</div>
                        <div class="form-row">
                            <div class="form-field">
                                <label>GitHub Repo</label>
                                <input id="s-gh-repo" placeholder="owner/repo-name">
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Local Model Runtime</h3>
                        <div class="settings-section-copy">Only fill this in when you want Ouroboros to start and route to a GGUF model on this machine.</div>
                        <div class="form-grid two">
                            <div class="form-field">
                                <label>Model Source</label>
                                <input id="s-local-source" placeholder="bartowski/Llama-3.3-70B-Instruct-GGUF or /path/to/model.gguf">
                            </div>
                            <div class="form-field">
                                <label>GGUF Filename (for HF repos)</label>
                                <input id="s-local-filename" placeholder="Llama-3.3-70B-Instruct-Q4_K_M.gguf">
                            </div>
                        </div>
                        <div class="form-grid four">
                            <div class="form-field">
                                <label>Port</label>
                                <input id="s-local-port" type="number" value="8766">
                            </div>
                            <div class="form-field">
                                <label>GPU Layers (-1 = all)</label>
                                <input id="s-local-gpu-layers" type="number" value="-1">
                            </div>
                            <div class="form-field">
                                <label>Context Length</label>
                                <input id="s-local-ctx" type="number" value="16384">
                            </div>
                            <div class="form-field">
                                <label>Chat Format</label>
                                <input id="s-local-chat-format" placeholder="auto-detect">
                            </div>
                        </div>
                        <div class="settings-toolbar">
                            <button class="btn btn-primary" id="btn-local-start">Start</button>
                            <button class="btn btn-primary" id="btn-local-stop">Stop</button>
                            <button class="btn btn-primary" id="btn-local-test">Test Tool Calling</button>
                        </div>
                        <div id="local-model-status" class="settings-inline-status">Status: Offline</div>
                        <div id="local-model-progress-wrap" class="local-model-progress-wrap local-model-hidden" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
                            <div id="local-model-progress-bar" class="local-model-progress-bar"></div>
                        </div>
                        <button class="btn btn-secondary local-model-install-btn local-model-hidden" id="btn-local-install-runtime">Install Local Runtime</button>
                        <div id="local-model-test-result" class="settings-test-result"></div>
                    </div>

                    <div class="form-section">
                        <h3>Runtime Limits</h3>
                        <!-- Active Subagents / Root and Subagent Depth moved to
                             Agents → Delegation (D-10): they bound the agents,
                             not the process pool. Max Workers stays: it is
                             runtime worker processes, not an agent setting. -->
                        <div class="settings-section-copy">Workers control parallel task capacity. Task liveness is governed automatically by progress, deadlines, the absolute ceiling, and the reaper. Budget limits control runtime cost thresholds. How many subagents a task may run, and how deep they may nest, live in <code>Agents</code>.</div>
                        <div class="form-grid two">
                            <div class="form-field">
                                <label>Max Workers</label>
                                <input id="s-workers" type="number" min="1" max="50" value="10">
                            </div>
                            <div class="form-field">
                                <label>Concurrent Presence Conversations</label>
                                <input id="s-presence-max-active" type="number" min="1" max="20" value="2">
                            </div>
                            <div class="form-field">
                                <label>Tool Timeout (s)</label>
                                <input id="s-tool-timeout" type="number" value="600">
                            </div>
                            <div class="form-field">
                                <label>Total Budget (USD)</label>
                                <input id="s-total-budget" type="number" min="0.01" step="any" value="200.0">
                            </div>
                            <div class="form-field">
                                <label>Per-Task Cost Cap (USD)</label>
                                <input id="s-settings-per-task-cost" type="number" min="0.01" step="any" value="50.0">
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Cleanup</h3>
                        <!-- The two subagent path roots moved to Agents →
                             Delegation → Advanced (D-10). GC Retention stays: it
                             governs every disposable runtime artifact, not just
                             subagent worktrees. -->
                        <div class="settings-section-copy">
                            <strong>GC Retention</strong> is the single age knob (days) for all disposable runtime artifacts the startup garbage collector removes: acting-subagent worktrees, terminal task drives, and leftover service logs (hard max 365). Genesis projects are durable and never auto-removed. Where subagents check out that work is set in <code>Agents</code>.
                        </div>
                        <div class="form-grid two">
                            <div class="form-field">
                                <label>GC Retention (days)</label>
                                <input id="s-gc-retention-days" type="number" min="1" max="365" value="7">
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Extension Settings</h3>
                        <div class="settings-section-copy">
                            Live extensions can register reviewed, host-rendered settings sections.
                            Sections appear here after the owning skill is reviewed, enabled, and loaded.
                        </div>
                        <div id="extension-settings-sections" class="settings-extension-sections">
                            <div class="muted">No extension settings registered.</div>
                        </div>
                    </div>

                    <div class="form-section danger">
                        <h3>Danger Zone</h3>
                        <div class="settings-inline-note">Reset still uses the current restart-based flow. This clears runtime data but keeps the repo.</div>
                        <button class="btn btn-danger" id="btn-reset">Reset All Data</button>
                    </div>
                </section>

                <section class="settings-panel" data-settings-panel="about">
                    <div class="about-body">
                        <img src="/static/logo.jpg" class="about-logo" alt="Ouroboros">
                        <div>
                            <h1 class="about-title">Ouroboros</h1>
                            <p id="about-version" class="about-version"></p>
                        </div>
                        <p class="about-desc">
                            A self-creating AI agent. Not a tool, but a becoming digital personality
                            with its own constitution, persistent identity, and background consciousness.
                            Born February 16, 2026.
                        </p>
                        <div class="about-credits">
                            <span>Created by <strong>Anton Razzhigaev</strong> &amp; <strong>Andrew Kaznacheev</strong></span>
                            <div class="about-links">
                                <a href="https://t.me/abstractDL" target="_blank" rel="noopener noreferrer">@abstractDL</a>
                                <a href="https://github.com/razzant/ouroboros" target="_blank" rel="noopener noreferrer">GitHub</a>
                            </div>
                        </div>
                    </div>
                </section>
            </div>

            <div class="settings-footer">
                <div class="settings-footer-actions">
                    <button type="button" class="btn btn-secondary" id="btn-reload-settings">Reload Settings</button>
                    <button class="btn btn-save" id="btn-save-settings">Save Settings</button>
                    <button type="button" class="btn btn-secondary" id="btn-restart-now" hidden
                        title="Restart the agent process to apply the saved changes">Restart now</button>
                </div>
                <div class="settings-footer-status">
                    <span id="settings-unsaved-indicator" class="settings-inline-status settings-unsaved-indicator" aria-hidden="true">Unsaved changes</span>
                    <div id="settings-status" class="settings-inline-status"></div>
                </div>
            </div>
        </div>
    `;
}

export function bindSettingsTabs(root, options = {}) {
    const tabs = Array.from(root.querySelectorAll('.settings-tab'));
    const panels = Array.from(root.querySelectorAll('.settings-panel'));
    const scrollRoot = root.querySelector('.settings-scroll');
    const state = options.state || null;
    const onActivate = typeof options.onActivate === 'function' ? options.onActivate : null;

    // All viewports use horizontal tab pills; mobile back remains DOM-only for compat.
    function activate(tabName) {
        root.dataset.activeSettingsTab = tabName;
        let activeButton = null;
        tabs.forEach((button) => {
            const isActive = button.dataset.settingsTab === tabName;
            button.classList.toggle('active', isActive);
            if (isActive) activeButton = button;
        });
        panels.forEach((panel) => {
            panel.classList.toggle('active', panel.dataset.settingsPanel === tabName);
        });
        if (scrollRoot) scrollRoot.scrollTop = 0;
        if (state) state.settingsActiveSubtab = tabName;
        // Keep active pill visible in the horizontal strip.
        if (activeButton && typeof activeButton.scrollIntoView === 'function') {
            activeButton.scrollIntoView({
                behavior: 'auto',
                inline: 'center',
                block: 'nearest',
            });
        }
        if (onActivate) onActivate(tabName);
        window.dispatchEvent(new CustomEvent('ouro:settings-subtab-shown', { detail: { tab: tabName } }));
    }

    tabs.forEach((button) => {
        button.addEventListener('click', () => activate(button.dataset.settingsTab));
    });
    root.activateSettingsTab = activate;
    if (state && !state.settingsActiveSubtab) state.settingsActiveSubtab = 'providers';
    root.dataset.activeSettingsTab = state?.settingsActiveSubtab || 'providers';
}

export function bindSecretInputs(root) {
    root.querySelectorAll('.secret-input').forEach((input) => {
        input.addEventListener('input', () => {
            if (input.value.trim()) delete input.dataset.forceClear;
        });
    });

    root.querySelectorAll('.secret-toggle').forEach((button) => {
        button.addEventListener('click', () => {
            const target = root.querySelector(`#${button.dataset.target}`);
            if (!target) return;
            const nextType = target.type === 'password' ? 'text' : 'password';
            target.type = nextType;
            button.textContent = nextType === 'password' ? 'Show' : 'Hide';
        });
    });

    root.querySelectorAll('.secret-clear').forEach((button) => {
        button.addEventListener('click', () => {
            const target = root.querySelector(`#${button.dataset.target}`);
            if (!target) return;
            target.value = '';
            target.type = 'password';
            target.dataset.forceClear = '1';
            const toggle = root.querySelector(`.secret-toggle[data-target="${button.dataset.target}"]`);
            if (toggle) toggle.textContent = 'Show';
            // Programmatic value changes fire no 'input' event, but a Clear is
            // an edit like any other: the provider-test verdict-expiry listener
            // must see it, or a stale OK keeps vouching for a cleared key.
            target.dispatchEvent(new Event('input', { bubbles: true }));
        });
    });
}
