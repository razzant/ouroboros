// Provider-card markup shared by Settings and the builtin wizard. Assets resolve beside this module.
import { escapeHtmlAttr as escapeHtml } from './utils.js';

/** A provider's own mark (its icon beside this module), or '' when it has none. */
export const providerIconHtml = (card) => (card?.icon ? `<img src="${new URL(card.icon, import.meta.url).href}" alt="" class="settings-provider-icon">` : '');

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
                        ${providerIconHtml(card)}
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
// key cannot be saved without it. `choices`: a field that is one of a few stored words, chosen and never typed
// (the key choices below); a card's `testNote` says what its Test sends when that differs from the common sentence.
export const PROVIDER_CARDS = [
    {
        id: 'openrouter', kind: 'openrouter', title: 'OpenRouter', icon: '../providers/openrouter.ico', hint: 'Default multi-model router', open: true,
        fields: [{ id: 's-openrouter', rowField: 'api_key', settingKey: 'OPENROUTER_API_KEY', secret: true, label: 'API key', placeholder: 'sk-or-...' }],
    },
    {
        id: 'openai', kind: 'openai', title: 'OpenAI', icon: '../providers/openai.svg', hint: 'Official OpenAI API',
        fields: [
            { id: 's-openai', rowField: 'api_key', settingKey: 'OPENAI_API_KEY', secret: true, label: 'API key', placeholder: 'sk-...' },
            // Which OpenAI API this key's requests use. A key saved before the choice existed stores nothing and keeps
            // Chat Completions (`absent`); a key being created starts on Responses (`seed`) and stores it, as the server
            // seeds a new key. `older`: the flat setting (and its name in the wizard's state) that marks an older setup
            // whose address this key reaches: the built-in key created beside it starts on `absent`. A choice's
            // `profile` names the setup profile whose model defaults the wizard proposes on that option, where the
            // server offers one. `setupState` names the stored word in the wizard's state. What each option says is
            // OpenAI's own statement (function-calling and migration guides, read 2026-11-16), with no promise of
            // speed or price.
            { id: 's-openai-api-surface', rowField: 'api_surface', settingKey: 'OPENAI_API_SURFACE', setupState: 'openaiApiSurface', label: 'API',
                absent: 'chat_completions', seed: 'responses', older: { settingKey: 'OPENAI_BASE_URL', setupState: 'legacyCompatibleBaseUrl' },
                choices: [
                    { value: 'chat_completions', label: 'Chat Completions', note: 'The earlier API, still supported. Some newer OpenAI models cannot call tools through it.' },
                    { value: 'responses', label: 'Responses', profile: 'openai-responses', note: 'OpenAI’s current API. Some newer models call tools only through it.' },
                ] },
        ],
        testNote: 'Test checks a short tool call through the API chosen above. Provider charges may apply.',
        note: 'Pick OpenAI as the source in Models or Agents to route a role through this key. If OpenRouter is absent and the shipped defaults are still untouched, Ouroboros auto-remaps them to official OpenAI defaults.',
    },
    {
        id: 'compatible', kind: 'openai-compatible', title: 'OpenAI Compatible', icon: '../providers/openai-compatible.svg', hint: 'Custom OpenAI-style endpoint',
        fields: [
            { id: 's-openai-compatible-base-url', rowField: 'base_url', settingKey: 'OPENAI_COMPATIBLE_BASE_URL', label: 'Base URL', placeholder: 'https://provider.example/v1', credential: true, required: true },
            { id: 's-openai-compatible-key', rowField: 'api_key', settingKey: 'OPENAI_COMPATIBLE_API_KEY', secret: true, label: 'API key', placeholder: 'No key' },
        ],
        note: 'Enter your provider’s Base URL, then choose this connection in Models or Agents. For coding with OpenCode Go, use <code>https://opencode.ai/inference/go/openai/v1</code> and your Console API key. For built-in <code>web_search</code>, use the official OpenAI Responses API and leave <code>OPENAI_BASE_URL</code> empty.',
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
        note: 'Pick DeepSeek as the source in Models or Agents, then choose deepseek-v4-pro or deepseek-flash.',
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

const TEST_TITLE = 'Runs a short, bounded test. Provider charges may apply.';

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

const text = (value) => (value === undefined || value === null ? '' : String(value));

/* ---- A choice among a key's fields (OpenAI's API) -------------------------------------------------------------------
   Drawn as the strip (docs/DESIGN.md "Controls and editable choices") over a hidden input that holds what travels, so a
   host reads, tests and saves it like every other field. The input holds the stored word, untouched while the owner
   leaves the choice alone: a key saved before the field existed stores nothing and shows `absent`, and pressing the
   option such a key already shows puts back what was stored, so only a real change is sent. A key being created
   (`data-fresh`) starts on the word the server would seed it with (`choiceStart`) and stores the option shown, so what
   was seen is what is saved. A key's own page, an added key's row in a card and the wizard's flat key draw it; the
   built-in key's row in a card does not, and there the stored word travels unchanged (settings_connections.js). */

/** The option a choice field shows for a stored word: that word, `absent` for none, '' for a word that is none of them. */
export function choiceShown(field, stored) {
    const word = text(stored).trim().toLowerCase();
    if (!word) return field.absent;
    return field.choices.some((choice) => choice.value === word) ? word : '';
}

/** What a key's choices say in one line, by the stored words `read(field)` answers: `Responses API`. */
export function choiceWords(card, read) {
    return (card?.fields || []).filter((field) => field.choices).map((field) => {
        const chosen = field.choices.find((choice) => choice.value === choiceShown(field, read(field)));
        return chosen ? `${chosen.label} ${field.label}` : `Unknown ${field.label}`;
    });
}

/** The word a key being created starts on: the field's `seed`, or `absent` for the built-in key of an older setup
 *  (`older`: the value of the flat setting that marks one). */
export const choiceStart = (field, older = '') => (text(older).trim() ? field.absent : field.seed);

// The line under the strip: what the chosen option means, or that the stored word is none of the options.
const choiceNote = (field, shown, word) => (shown ? field.choices.find((choice) => choice.value === shown).note
    : `Saved value “${text(word).trim()}” is none of these. Choose one.`);
const choiceField = (holder) => PROVIDER_CARDS.find((card) => card.kind === holder.dataset.connectionChoice)
    .fields.find((field) => field.choices && field.rowField === holder.dataset.choiceField);

/** One choice field. `value` is what travels now and `stored` what the key holds; `start` (a key being created) is
 *  the word it starts on while no option was pressed; `held` a key the owner may not change. */
export function connectionChoiceHtml(card, field, { id, value = '', stored = value, start = '', held = false, keyName = '' } = {}) {
    const seeded = Boolean(start) && !value;
    const word = seeded ? start : value;
    const shown = choiceShown(field, word);
    const options = field.choices.map((choice) => `<button type="button" class="ui-segment${choice.value === shown ? ' active' : ''}"`
        + ` data-connection-choice-value="${choice.value}" aria-pressed="${choice.value === shown}"${held ? ' disabled' : ''}>${choice.label}</button>`).join('');
    return `
        <div class="form-field ui-field" data-connection-choice="${card.kind}" data-choice-field="${field.rowField}">
            <label>${field.label}</label>
            <input type="hidden" id="${id}" data-connection-field="${field.rowField}" value="${escapeHtml(word)}" data-applied-value="${escapeHtml(stored)}"${start ? ' data-fresh="1"' : ''}${seeded ? ' data-seeded="1"' : ''}>
            <div class="ui-segment-group" role="group" aria-label="${escapeHtml([card.title, keyName, field.label].filter(Boolean).join(' '))}" aria-describedby="${id}-note">${options}</div>
            <div class="ui-field-help" id="${id}-note" data-connection-choice-note${shown ? '' : ' data-tone="error"'}>${escapeHtml(choiceNote(field, shown, word))}</div>
        </div>`;
}

function paintChoice(holder) {
    const input = holder.querySelector('[data-connection-field]');
    const field = choiceField(holder);
    const shown = choiceShown(field, input.value);
    holder.querySelectorAll('[data-connection-choice-value]').forEach((option) => {
        option.classList.toggle('active', option.dataset.connectionChoiceValue === shown);
        option.setAttribute('aria-pressed', String(option.dataset.connectionChoiceValue === shown));
        option.disabled = Boolean(input.readOnly);
    });
    const note = holder.querySelector('[data-connection-choice-note]');
    note.textContent = choiceNote(field, shown, input.value);
    if (shown) delete note.dataset.tone; else note.dataset.tone = 'error';
}

/** Paint every choice field under `root` from its input, after a host put a stored value there or held the field. */
export function paintConnectionChoices(root) {
    root.querySelectorAll('[data-connection-choice]').forEach(paintChoice);
}

/** The choice fields of a key being created: each starts on `choiceStart` (`older(field)` answers the older setup's
 *  value for the built-in key), which is stored with the key (`data-seeded` until the owner presses an option: a host
 *  need not send a choice nobody made without the key it belongs to). */
export function seedConnectionChoices(root, older = () => '') {
    root.querySelectorAll('[data-connection-choice]').forEach((holder) => {
        const input = holder.querySelector('[data-connection-field]');
        const field = choiceField(holder);
        input.dataset.fresh = '1';
        if (!input.value) Object.assign(input, { value: choiceStart(field, older(field)) }).dataset.seeded = '1';
        paintChoice(holder);
    });
}

/** Press an option of a choice strip; answers whether the field changed. A saved key whose stored word already shows
 *  the pressed option takes that word back, so the press that undoes a change leaves nothing to send. */
export function pressConnectionChoice(option) {
    const holder = option.closest('[data-connection-choice]');
    const input = holder.querySelector('[data-connection-field]');
    const stored = input.dataset.appliedValue ?? '';
    const pressed = option.dataset.connectionChoiceValue;
    const next = input.dataset.fresh !== '1' && pressed === choiceShown(choiceField(holder), stored) ? stored : pressed;
    if (next === input.value) return false;
    input.value = next;
    delete input.dataset.seeded;
    paintChoice(holder);
    return true;
}

/**
 * The choices of the first-run wizard's built-in keys, which are flat fields of its setup contract (`providerFields`,
 * `state[field.stateKey]`). A kind's choice stands under its key while the draft holds one and joins the draft's
 * settings: the word the install stored (`initial[field.setupState]`) unless the owner presses another. A key typed
 * where the install had none (`initial` holds no key of that kind: the server's fact, never a reading of the field)
 * is a key being created: it starts on `choiceStart` and the draft stores that word with it.
 * `held(kind, settingKey)`: the owner may not change it.
 */
export function createFlatKeyChoices({ state, initial = {}, providerFields = [], held = () => false, onChange = () => {} } = {}) {
    const entries = PROVIDER_CARDS.flatMap((card) => {
        const key = providerFields.find((each) => card.fields.some((field) => field.secret && field.settingKey === each.settingKey));
        return key ? card.fields.filter((field) => field.choices).map((field) => ({ card, field, key })) : [];
    });
    const start = ({ key, field }) => (text(initial[key.stateKey]).trim() ? '' : choiceStart(field, initial[field.older?.setupState]));
    const present = ({ key }) => Boolean(text(state[key.stateKey]).trim());
    const stored = ({ field }) => text(initial[field.setupState]).trim();
    const word = (entry) => text(state[entry.field.setupState]).trim() || start(entry);
    return {
        /** Whether a contract field is one of these choices: it is drawn as its strip, never as a text field. */
        owns: (providerField) => entries.some(({ field }) => field.settingKey === providerField.settingKey),
        /** The strips under the flat key field of `settingKey`, hidden while the draft holds no key of that kind. */
        html: (settingKey) => entries.filter(({ key }) => key.settingKey === settingKey).map((entry) => {
            const strip = connectionChoiceHtml(entry.card, entry.field, { id: `${entry.key.id}-${entry.field.rowField.replace(/_/g, '-')}`,
                value: text(state[entry.field.setupState]).trim(), stored: stored(entry), start: start(entry),
                held: held(entry.card.kind, entry.field.settingKey) });
            return `<div data-flat-key-choice="${entry.field.settingKey}"${present(entry) ? '' : ' hidden'}>${strip}</div>`;
        }).join(''),
        /** Wire the strips drawn under `root`; their listeners go with their nodes at the next render. */
        bind(root) {
            for (const entry of entries) {
                const holder = root.querySelector(`[data-flat-key-choice="${entry.field.settingKey}"]`);
                if (!holder) continue;
                // The strip stands only under a key: it comes with the first character typed and goes with the last.
                root.querySelector(`#${entry.key.id}`)?.addEventListener('input', (event) => { holder.hidden = !text(event.target.value).trim(); });
                holder.addEventListener('click', (event) => {
                    const option = event.target?.closest?.('[data-connection-choice-value]');
                    if (!option || !pressConnectionChoice(option)) return;
                    state[entry.field.setupState] = holder.querySelector('[data-connection-field]').value;
                    onChange();
                });
            }
        },
        /** The settings these choices add to the draft: a choice that differs from the stored word, of a kind whose key
         *  the draft holds. An untouched choice of a saved key sends nothing. */
        draft: () => Object.fromEntries(entries.filter((entry) => present(entry) && word(entry) !== stored(entry))
            .map((entry) => [entry.field.settingKey, word(entry)])),
        /** The setup profile whose model defaults a proposal on one key takes (`openai-responses`), '' for its kind's
         *  own: what the key's choice says, the built-in key's as this draft holds it and an added key's (`name`) as its
         *  row in `state.connections` stores it, none for the earlier API (connectionDefaults). */
        profileOf: (kind, name = '') => {
            const card = PROVIDER_CARDS.find((each) => each.kind === kind);
            const row = name ? (state.connections || []).find((each) => each?.kind === kind && each.name === name) : null;
            return (card?.fields || []).filter((field) => field.choices).map((field) => {
                const entry = entries.find((each) => each.field === field);
                const said = name ? row?.[field.rowField] : entry && present(entry) ? word(entry) : '';
                return field.choices.find((choice) => choice.value === choiceShown(field, said))?.profile || '';
            }).find(Boolean) || '';
        },
        /** What the choices of the key in `stateKey` say in the summary (`Responses API`); none without that key. */
        words: (stateKey) => entries.filter((entry) => entry.key.stateKey === stateKey && present(entry))
            .flatMap((entry) => choiceWords(entry.card, () => word(entry))),
    };
}

/** A key's form, the same in setup's card and on the key's own page in Settings: its name (an added key) and the
 *  fields of its kind. Without `row.uid` it is the card's built-in key ("Default"). `choices` adds the kind's choice
 *  fields (the key's own page). */
export function connectionFieldsHtml(card, row = {}, { choices = false } = {}) {
    const builtin = !row.uid;
    const stem = builtin ? `s-${card.id}-default` : `conn-${row.uid}`;
    const keyName = builtin ? '' : String(row.label || row.name || '');
    const values = row.values || {};
    return [
        builtin ? '' : `<div class="form-field ui-field"><label for="${stem}-label">Name</label><input id="${stem}-label" class="ui-control" type="text" data-connection-label value="${escapeHtml(text(row.label))}" placeholder="e.g. Claude credits" aria-label="${escapeHtml(`${card.title} key name`)}" autocomplete="off" spellcheck="false"></div>`,
        ...card.fields.map((field) => (!field.choices ? connectionFieldHtml(card, field, { stem, value: text(values[field.rowField]), builtin, keyName })
            : !choices ? '' : connectionChoiceHtml(card, field, { id: builtin ? field.id : `${stem}-${field.rowField.replace(/_/g, '-')}`,
                value: text(values[field.rowField]), start: row.draft ? field.seed : '', keyName }))),
    ].join('');
}

/**
 * One API key row (docs/DESIGN.md §6 account-row anatomy): name and status, the meta line, Test /
 * Disable / Remove docked right, then the key's own fields and its quiet result lines. Without `row.uid` it
 * is the card's built-in key ("Default"); an added key's row carries its stored row values. The status, meta
 * and action labels are painted by settings_connections.js from the loaded document.
 */
export function connectionRowHtml(card, row = {}) {
    const builtin = !row.uid;
    const keyName = builtin ? '' : String(row.label || row.name || '');
    // An added key carries its kind's choices (OpenAI's API) as on its own page; the built-in key's stand on its page.
    const fields = connectionFieldsHtml(card, row, { choices: !builtin });
    const identity = builtin ? ' data-builtin="1"'
        : ` data-uid="${escapeHtml(row.uid)}" data-name="${escapeHtml(text(row.name))}"${row.draft ? ' data-draft="1"' : ''}`;
    return `
        <div class="harness-account-row connection-row" data-connection-row data-kind="${card.kind}"${identity} data-enabled="${row.enabled === false ? '0' : '1'}">
            <div class="harness-account-main">
                <strong data-connection-title${builtin ? '' : ' data-i18n-skip'}>${builtin ? 'Default' : escapeHtml(keyName || `New ${card.title} key`)}</strong>
                <span class="ui-status" data-tone="neutral" data-connection-status hidden></span>
                <span class="ui-chip skills-chip" data-tone="neutral" data-connection-fixed hidden>Set by your organization</span>
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
                <div class="connection-note" data-connection-fixed-fields hidden></div>
            </div>
        </div>`;
}

export function providerSettingsCard(card) {
    return providerCard(card, `<div class="connection-rows" data-connection-rows="${card.kind}">${connectionRowHtml(card)}</div>`
        + `${card.note ? `<div class="settings-inline-note">${card.note}</div>` : ''}`);
}
