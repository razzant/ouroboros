// Provider-card markup shared by Settings and the builtin wizard. Assets resolve beside this module.
import { escapeHtmlAttr as escapeHtml } from './utils.js';

// A provider card: its head is the card's disclosure plus its own Add key (docs/DESIGN.md §6: a
// section-level action belongs in its group's header), its body the provider's key rows.
function providerCard(card, body) {
    const bodyId = `settings-provider-${card.id}-body`;
    const titleId = `settings-provider-${card.id}-title`;
    return `
        <section class="settings-provider-card" data-provider-card="${card.id}">
            <div class="settings-provider-head">
                <button type="button" class="settings-provider-toggle" data-provider-toggle aria-expanded="${card.open ? 'true' : 'false'}" aria-controls="${bodyId}">
                    <span class="settings-provider-title" id="${titleId}">
                        ${card.icon ? `<img src="${new URL(card.icon, import.meta.url).href}" alt="" class="settings-provider-icon">` : ''}
                        <span>${card.title}</span>
                    </span>
                    <span class="settings-provider-hint">${card.hint || ''}</span>
                </button>
                <button type="button" class="btn btn-default" data-connection-add="${card.kind}" aria-describedby="${titleId}">Add key</button>
            </div>
            <div class="settings-provider-body" id="${bodyId}"${card.open ? '' : ' hidden'}>
                ${body}
            </div>
        </section>
    `;
}

// API keys: one card per provider kind, one row per key. The built-in key ("Default") edits today's flat
// settings (`settingKey`; its inputs keep the `id` the setup contract names); a key added in the card is one
// OUROBOROS_CONNECTIONS row whose fields are its `rowField`s (ouroboros/provider_models.py PROVIDER_KINDS).
// `credential`: a plain field Remove clears on the built-in key besides its secrets; `required`: an added
// key cannot be saved without it.
export const PROVIDER_CARDS = [
    {
        id: 'openrouter', kind: 'openrouter', title: 'OpenRouter', icon: '../providers/openrouter.ico', hint: 'Default multi-model router', open: true,
        fields: [{ id: 's-openrouter', rowField: 'api_key', settingKey: 'OPENROUTER_API_KEY', secret: true, label: 'API key', placeholder: 'sk-or-...' }],
    },
    {
        id: 'openai', kind: 'openai', title: 'OpenAI', icon: '../providers/openai.svg', hint: 'Official OpenAI API',
        fields: [{ id: 's-openai', rowField: 'api_key', settingKey: 'OPENAI_API_KEY', secret: true, label: 'API key', placeholder: 'sk-...' }],
        note: 'Pick OpenAI as the source in Models or Agents to route a role through this key. If OpenRouter is absent and the shipped defaults are still untouched, Ouroboros auto-remaps them to official OpenAI defaults.',
    },
    {
        id: 'compatible', kind: 'openai-compatible', title: 'OpenAI Compatible', icon: '../providers/openai-compatible.svg', hint: 'Custom OpenAI-style endpoint',
        fields: [
            { id: 's-openai-compatible-base-url', rowField: 'base_url', settingKey: 'OPENAI_COMPATIBLE_BASE_URL', label: 'Base URL', placeholder: 'https://provider.example/v1', credential: true, required: true },
            { id: 's-openai-compatible-key', rowField: 'api_key', settingKey: 'OPENAI_COMPATIBLE_API_KEY', secret: true, label: 'API key', placeholder: 'No key' },
        ],
        note: 'Use this card for custom base URLs. Built-in web search only works with the official OpenAI Responses API, so keep <code>OPENAI_BASE_URL</code> empty when you want <code>web_search</code>.',
    },
    {
        // advanced: rendered inside the collapsed "More providers" section. Inputs stay mounted either
        // way — settings.js applyInputValue has no null guard, so every input id must always exist in the DOM.
        id: 'cloudru', kind: 'cloudru', title: 'Cloud.ru Foundation Models', icon: '../providers/cloudru.svg', hint: 'Cloud.ru OpenAI-compatible runtime', advanced: true,
        fields: [
            { id: 's-cloudru-key', rowField: 'api_key', settingKey: 'CLOUDRU_FOUNDATION_MODELS_API_KEY', secret: true, label: 'API key', placeholder: 'Cloud.ru Foundation Models API key' },
            { id: 's-cloudru-base-url', rowField: 'base_url', settingKey: 'CLOUDRU_FOUNDATION_MODELS_BASE_URL', label: 'Base URL', placeholder: 'https://foundation-models.api.cloud.ru/v1' },
        ],
    },
    {
        id: 'minimax', kind: 'minimax', title: 'MiniMax', icon: '', hint: 'Direct regional OpenAI-compatible runtime', advanced: true,
        fields: [
            { id: 's-minimax-key', rowField: 'api_key', settingKey: 'MINIMAX_API_KEY', secret: true, label: 'API key', placeholder: 'MiniMax API key' },
            { id: 's-minimax-region', rowField: 'region', settingKey: 'MINIMAX_REGION', label: 'Region', placeholder: 'global_en or cn_zh' },
        ],
        note: 'Pick MiniMax as the source in Models or Agents, then choose MiniMax-M3 or MiniMax-M3.1-Flash. Leave Region empty for <code>global_en</code>; use <code>cn_zh</code> for the China endpoint.',
    },
    {
        id: 'deepseek', kind: 'deepseek', title: 'DeepSeek', icon: '', hint: 'Direct OpenAI-compatible runtime (v4 family)', advanced: true,
        fields: [{ id: 's-deepseek-key', rowField: 'api_key', settingKey: 'DEEPSEEK_API_KEY', secret: true, label: 'API key', placeholder: 'sk-...' }],
        note: 'Pick DeepSeek as the source in Models or Agents, then choose deepseek-v4-pro or deepseek-v4-flash.',
    },
    {
        id: 'zai', kind: 'zai', title: 'Z.ai (GLM)', icon: '', hint: 'Direct OpenAI-compatible runtime', advanced: true,
        fields: [
            { id: 's-zai-key', rowField: 'api_key', settingKey: 'ZAI_API_KEY', secret: true, label: 'API key', placeholder: 'sk-...' },
            { id: 's-zai-plan', rowField: 'plan', settingKey: 'ZAI_PLAN', label: 'Plan', placeholder: 'payg or coding' },
        ],
        note: 'Pick Z.ai as the source in Models or Agents, then choose glm-5.3 or glm-5.3-flash. Coding Plan subscribers: set Plan to <code>coding</code>; the default <code>payg</code> is pay-as-you-go, and a Coding Plan key tested there reports No credits.',
    },
    {
        id: 'gigachat', kind: 'gigachat', title: 'GigaChat', icon: '../providers/gigachat.svg', hint: 'Sber GigaChat via the gigachat library', advanced: true,
        fields: [
            { id: 's-gigachat-credentials', rowField: 'credentials', settingKey: 'GIGACHAT_CREDENTIALS', secret: true, label: 'Authorization key', placeholder: 'Base64 client_id:secret (OAuth)' },
            { id: 's-gigachat-scope', rowField: 'scope', settingKey: 'GIGACHAT_SCOPE', label: 'Scope', placeholder: 'GIGACHAT_API_PERS' },
            { id: 's-gigachat-user', rowField: 'user', settingKey: 'GIGACHAT_USER', label: 'User (basic auth, optional)', placeholder: 'username', credential: true },
            { id: 's-gigachat-password', rowField: 'password', settingKey: 'GIGACHAT_PASSWORD', secret: true, label: 'Password (basic auth, optional)', placeholder: 'password' },
            { id: 's-gigachat-base-url', rowField: 'base_url', settingKey: 'GIGACHAT_BASE_URL', label: 'Base URL', placeholder: 'https://api.giga.chat/v1' },
            { id: 's-gigachat-verify-ssl', rowField: 'verify_ssl_certs', settingKey: 'GIGACHAT_VERIFY_SSL_CERTS', label: 'Verify SSL certs', placeholder: 'true / false' },
        ],
        note: 'Pick GigaChat as the source in Models or Agents to route a role through this account. Authenticate with either an Authorization Key (OAuth, scope <code>GIGACHAT_API_PERS</code>, <code>GIGACHAT_API_B2B</code>, or <code>GIGACHAT_API_CORP</code>) or User + Password.',
    },
    {
        id: 'anthropic', kind: 'anthropic', title: 'Anthropic', icon: '../providers/anthropic.png', hint: 'Direct Anthropic API access',
        fields: [{ id: 's-anthropic', rowField: 'api_key', settingKey: 'ANTHROPIC_API_KEY', secret: true, label: 'API key', placeholder: 'sk-ant-...' }],
        note: 'Pick Anthropic as the source in Models or Agents to route a role directly through this key.',
    },
];

/** Every provider secret Accounts renders: Secrets and skill requests never draw a second input for one. */
export const PROVIDER_SECRET_KEYS = PROVIDER_CARDS.flatMap((card) => card.fields.filter((field) => field.secret)
    .map((field) => field.settingKey));

const TEST_TITLE = 'Sends one short model request. Provider charges may apply.';

function connectionFieldHtml(card, field, { stem, value, builtin, keyName }) {
    const id = builtin ? field.id : `${stem}-${field.rowField.replace(/_/g, '-')}`;
    // The visible label stays short inside its card; the field's accessible name says whose key it is.
    const attrs = `id="${id}" name="${builtin && field.secret ? field.settingKey : id}"${builtin && field.secret ? ` data-secret-setting="${field.settingKey}"` : ''}`
        + ` data-connection-field="${field.rowField}" placeholder="${escapeHtml(field.placeholder)}" aria-label="${escapeHtml([card.title, keyName, field.label].filter(Boolean).join(' '))}"`
        + `${builtin ? '' : ` value="${escapeHtml(value)}" data-applied-value="${escapeHtml(value)}"`} autocomplete="off" spellcheck="false"`;
    if (!field.secret) return `<div class="form-field ui-field"><label for="${id}">${field.label}</label><input ${attrs} type="text" class="ui-control"></div>`;
    return `
        <div class="form-field ui-field">
            <label for="${id}">${field.label}</label>
            <div class="secret-input-row">
                <input ${attrs} class="secret-input ui-control" type="password">
                <button type="button" class="btn btn-default secret-toggle" data-target="${id}">Show</button>
                <button type="button" class="btn btn-default secret-clear" data-target="${id}">Clear</button>
            </div>
        </div>`;
}

/**
 * One API key row (docs/DESIGN.md §6 account-row anatomy): name and status, the meta line, Test /
 * Disable / Remove docked right, then the key's own fields and its quiet result lines. Without `row.uid` it
 * is the card's built-in key ("Default"); an added key's row carries its stored row values. The status, meta
 * and action labels are painted by settings_connections.js from the loaded document.
 */
export function connectionRowHtml(card, row = {}) {
    const builtin = !row.uid;
    const stem = builtin ? `s-${card.id}-default` : `conn-${row.uid}`;
    const keyName = builtin ? '' : String(row.label || row.name || '');
    const values = row.values || {};
    const text = (value) => (value === undefined || value === null ? '' : String(value));
    const fields = [
        builtin ? '' : `<div class="form-field ui-field"><label for="${stem}-label">Name</label><input id="${stem}-label" class="ui-control" type="text" data-connection-label value="${escapeHtml(text(row.label))}" placeholder="e.g. Claude credits" aria-label="${escapeHtml(`${card.title} key name`)}" autocomplete="off" spellcheck="false"></div>`,
        ...card.fields.map((field) => connectionFieldHtml(card, field, { stem, value: text(values[field.rowField]), builtin, keyName })),
    ].join('');
    const identity = builtin ? ' data-builtin="1"'
        : ` data-uid="${escapeHtml(row.uid)}" data-name="${escapeHtml(text(row.name))}"${row.draft ? ' data-draft="1"' : ''}`;
    return `
        <div class="harness-account-row connection-row" data-connection-row data-kind="${card.kind}"${identity} data-enabled="${row.enabled === false ? '0' : '1'}">
            <div class="harness-account-main">
                <strong data-connection-title${builtin ? '' : ' data-i18n-skip'}>${builtin ? 'Default' : escapeHtml(keyName || `New ${card.title} key`)}</strong>
                <span class="ui-status" data-tone="neutral" data-connection-status hidden></span>
            </div>
            <div class="harness-account-meta muted" data-connection-meta hidden></div>
            <div class="harness-account-actions">
                <button type="button" class="btn btn-default" data-connection-test${builtin ? ` data-provider-test="${card.kind}"` : ''} title="${TEST_TITLE}">Test</button>
                <button type="button" class="btn btn-default" data-connection-toggle>${row.enabled === false ? 'Enable' : 'Disable'}</button>
                <button type="button" class="btn btn-default" data-connection-remove>Remove</button>
            </div>
            <div class="connection-fields">
                <div class="form-row">${fields}</div>
                <div class="settings-inline-status" data-connection-test-status${builtin ? ` data-provider-test-status="${card.kind}"` : ''} role="status" aria-live="polite" aria-atomic="true"></div>
                <div class="connection-note" data-connection-cache hidden></div>
            </div>
        </div>`;
}

export function providerSettingsCard(card) {
    return providerCard(card, `<div class="connection-rows" data-connection-rows="${card.kind}">${connectionRowHtml(card)}</div>`
        + `${card.note ? `<div class="settings-inline-note">${card.note}</div>` : ''}`);
}
