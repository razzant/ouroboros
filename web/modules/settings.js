import { refreshModelCatalog } from './settings_catalog.js';
import { bindEffortSegments, syncEffortSegments, readCustomSecretDraft, collectCustomSecretDraft, paintSettingsFieldErrors, settingsWriteFailure } from './settings_controls.js';
import { bindLocalModelControls } from './settings_local_model.js';
import { applyMcpSettings, collectMcpSettings, initMcpSettings, validateMcpSettings } from './mcp_settings.js';
import { adoptSubagentRoster, collectReviewerSlots, initReviewerSlots, reloadReviewerSlots, validateReviewerSlots, noteReviewerSlotsSaveAttempt, discardReviewerSlotsDraft, setReviewerProcessingPreference, setReviewerSourceContext } from './reviewer_slots.js';
import {
    applySubagentsSettings,
    availableSubagentsPreviewPayload,
    collectSubagentsSettings,
    initSubagentsSection,
    noteSubagentsSaveAttempt,
    reloadSubagentsSection,
    subagentSettingsFingerprint,
    validateSubagentsDraft,
    setSubagentsProcessingPreference,
    setSubagentsSourceContext,
} from './subagents_settings.js';
import { initHarnessAccounts } from './harness_accounts.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { PROVIDER_TEST_INPUTS, SECRET_KEYS, bindSecretInputs, bindSettingsTabs, renderSettingsPage } from './settings_ui.js';
import { showToast } from './toast.js';
import { bindThemeSegments } from './theme.js';
import { escapeHtmlAttr as escapeHtml, formatDualVersion } from './utils.js';
import { apiClient, apiFetch, cleanExtensionRoute, extensionRoutePath } from './api_client.js';
import { claudexorStatus } from './claudexor_status_store.js';
import { createModelRolesEditor, modelRoleMap } from './model_roles.js';
import { PROCESSING_PREFERENCE_KEY, MODEL_PROCESSING_PREFERENCES_KEY } from './route_editor_primitives.js';
import { collectSafeFieldValues, normalizeTone, renderSafeField, setInlineStatus, revealNewRow } from './ui_helpers.js';
import { extensionActionStatus } from './extension_status_text.js';
import { currentLanguage, setLanguage, storedLanguage } from './i18n.js';
import { isReasoningVisible, REASONING_VISIBILITY_EVENT, setReasoningVisible } from './log_events.js';

let markSettingsDirty = () => {};
const BASE_SECRET_KEYS = new Set(SECRET_KEYS.map(([key]) => key));
const pendingExtensionSettings = new Set();
let setupContract = {};

const INPUT_FIELDS = [
    ['s-openai-base-url', 'OPENAI_BASE_URL'], ['s-openai-compatible-base-url', 'OPENAI_COMPATIBLE_BASE_URL'], ['s-cloudru-base-url', 'CLOUDRU_FOUNDATION_MODELS_BASE_URL'],
    ['s-gigachat-scope', 'GIGACHAT_SCOPE'], ['s-gigachat-user', 'GIGACHAT_USER'], ['s-gigachat-base-url', 'GIGACHAT_BASE_URL'], ['s-gigachat-verify-ssl', 'GIGACHAT_VERIFY_SSL_CERTS'],
    ['s-minimax-region', 'MINIMAX_REGION'],
    ['s-server-host', 'OUROBOROS_SERVER_HOST', '127.0.0.1'],
    // 6.1: OUROBOROS_REVIEW_MODELS / OUROBOROS_SCOPE_REVIEW_MODELS are no
    // longer authored here — the Review lanes section composes the ONE
    // structured setting; the comma keys stay a backend-derived projection.
    // R7: OUROBOROS_MODEL_DEEP_SELF_REVIEW is not authored here either — the
    // deep self-review row lives in Review lanes; the key is the backend's
    // invisible migration source for that row.
    ['s-skills-repo-path', 'OUROBOROS_SKILLS_REPO_PATH'],
    ['s-clawhub-registry-url', 'OUROBOROS_CLAWHUB_REGISTRY_URL'], ['s-websearch-model', 'OUROBOROS_WEBSEARCH_MODEL'], ['s-gh-repo', 'GITHUB_REPO'],
    ['s-local-source', 'LOCAL_MODEL_SOURCE'], ['s-local-filename', 'LOCAL_MODEL_FILENAME'], ['s-local-chat-format', 'LOCAL_MODEL_CHAT_FORMAT'],
    ['s-subagent-worktree-root', 'OUROBOROS_SUBAGENT_WORKTREE_ROOT'], ['s-subagent-projects-root', 'OUROBOROS_SUBAGENT_PROJECTS_ROOT'],
    ['s-evo-budget', 'OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD', '0'],
    ['s-consciousness-daily-usd', 'OUROBOROS_CONSCIOUSNESS_DAILY_USD', '20'],  // float: NUMBER_FIELDS would truncate 20.5 to 20
    ['s-evo-objective', 'OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE', ''],
];
const VALUE_FIELDS = [
    // 6.3: Review / Scope Review efforts are per-slot rows in Agents → Review
    // lanes now; their global keys remain backend defaults, no longer UI-authored.
    ['s-effort-task', 'OUROBOROS_EFFORT_TASK', 'medium'], ['s-effort-evolution', 'OUROBOROS_EFFORT_EVOLUTION', 'high'],
    ['s-effort-consciousness', 'OUROBOROS_EFFORT_CONSCIOUSNESS', ''], ['s-effort-deep-self-review', 'OUROBOROS_EFFORT_DEEP_SELF_REVIEW', 'high'],
    ['s-consciousness-autonomy', 'OUROBOROS_CONSCIOUSNESS_AUTONOMY', 'act'],
    ['s-review-enforcement', 'OUROBOROS_REVIEW_ENFORCEMENT', 'advisory'], ['s-task-review-mode', 'OUROBOROS_TASK_REVIEW_MODE', 'auto'], ['s-runtime-mode', 'OUROBOROS_RUNTIME_MODE', 'advanced'],
    // Shared paid-review-cycle cap (plan review / task acceptance / commit gate);
    // the ∞ segment saves the string "unlimited" (SSOT: ouroboros/review_cycles.py).
    ['s-review-max-cycles', 'OUROBOROS_REVIEW_MAX_CYCLES', '2'],
    ['s-update-channel', 'OUROBOROS_UPDATE_CHANNEL', 'stable'],
    ['s-context-mode', 'OUROBOROS_CONTEXT_MODE', 'max'], ['s-image-input-mode', 'OUROBOROS_IMAGE_INPUT_MODE', 'auto'],
    ['s-safety-mode', 'OUROBOROS_SAFETY_MODE', 'full'],
    ['s-prompt-cache-ttl', 'OUROBOROS_PROMPT_CACHE_TTL', '1h'],
];
const _SAFETY_MODE_RANK = { full: 2, light: 1, off: 0 };
const NUMBER_FIELDS = [
    ['s-workers', 'OUROBOROS_MAX_WORKERS', 10], ['s-presence-max-active', 'OUROBOROS_PRESENCE_MAX_ACTIVE', 2], ['s-active-subagents', 'OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT', 6], ['s-subagent-depth', 'OUROBOROS_MAX_SUBAGENT_DEPTH', 3, true],
    ['s-tool-timeout', 'OUROBOROS_TOOL_TIMEOUT_SEC', 600], ['s-local-port', 'LOCAL_MODEL_PORT', 8766], ['s-local-gpu-layers', 'LOCAL_MODEL_N_GPU_LAYERS', -1, true],
    ['s-local-ctx', 'LOCAL_MODEL_CONTEXT_LENGTH', 16384], ['s-gc-retention-days', 'OUROBOROS_GC_RETENTION_DAYS', 7],
    ['s-bg-wakeup-min', 'OUROBOROS_BG_WAKEUP_MIN', 900], ['s-bg-wakeup-max', 'OUROBOROS_BG_WAKEUP_MAX', 14400],
    ['s-consciousness-max-tasks', 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS', 2, true],  // 0 = never starts tasks: a choice, not unset
];

function setupModelSlots() {
    return Array.isArray(setupContract.modelSlots) ? setupContract.modelSlots : [];
}

function byId(id) {
    return document.getElementById(id);
}

function applyInputValue(id, value) {
    const el = byId(id);
    el.value = value === undefined || value === null ? '' : value;
    // Server-applied snapshot (secrets arrive MASKED): lets the provider-test
    // handler tell an owner edit apart from the mask, which must never be sent
    // back as a credential.
    el.dataset.appliedValue = el.value;
}

function applyCheckboxValue(id, value) {
    byId(id).checked = isTruthySetting(value);
}

function isTruthySetting(value) {
    const normalized = String(value ?? '').trim().toLowerCase();
    return value === true || ['true', '1', 'yes', 'on'].includes(normalized);
}

// A loading, validation or editor owner may update its own status. A later
// message from anyone else drops that ownership, protecting the newer result.
function setStatus(text, tone = 'ok', owner = '', subject = '') {
    const status = byId('settings-status');
    status.textContent = text;
    status.dataset.tone = tone;
    if (owner) status.dataset.owner = owner;
    else delete status.dataset.owner;
    if (subject) status.dataset.subject = subject;
    else delete status.dataset.subject;
}

function setButtonBusy(button, busy) {
    if (!button) return;
    button.disabled = busy;
    if (busy) button.setAttribute('aria-busy', 'true');
    else button.removeAttribute('aria-busy');
}

function policyValueLabel(value) {
    const labels = {
        light: 'Light', advanced: 'Advanced', pro: 'Pro', cyber_pro: 'Cyber Pro',
        full: 'Full', off: 'Off', advisory: 'Advisory', blocking: 'Blocking',
    };
    return labels[String(value || '').trim().toLowerCase()] || String(value || 'Unknown');
}

function syncPolicyState(root, meta) {
    const state = meta?.policy_state;
    if (!state) return;
    const render = (key, text) => {
        const node = root?.querySelector(`[data-policy-state="${key}"]`);
        if (node) node.textContent = text;
    };
    const access = state.access || {};
    render('access', access.restart_required
        ? `Saved: ${policyValueLabel(access.configured)} · Current process: ${policyValueLabel(access.current_process || access.effective)} · After restart: ${policyValueLabel(access.configured)} · Restart required`
        : `Current process: ${policyValueLabel(access.current_process || access.effective)} · After restart: ${policyValueLabel(access.configured)}`);
    const suffix = (item) => item.active_task_snapshot
        ? `Saved: ${policyValueLabel(item.configured)} · Current process: ${policyValueLabel(item.current_process || item.effective)} · Next task: ${policyValueLabel(item.next_task || item.configured)} · Current task keeps its start snapshot`
        : `Current process: ${policyValueLabel(item.current_process || item.effective)} · Next task: ${policyValueLabel(item.next_task || item.configured)}`;
    render('supervisor', suffix(state.supervisor || {}));
    render('review', suffix(state.review || {}));
}

function readInt(id, fallback) {
    const value = parseInt(byId(id).value, 10);
    return Number.isNaN(value) ? fallback : value;
}

function resetSecretClearFlags(root) {
    root.querySelectorAll('.secret-input').forEach((input) => {
        delete input.dataset.forceClear;
        input.type = 'password';
    });
    root.querySelectorAll('.secret-toggle').forEach((button) => {
        button.textContent = 'Show';
    });
}

function applySecretInputs(root, settings) {
    root.querySelectorAll('[data-secret-setting]').forEach((input) => {
        applyInputValue(input.id, settings[input.dataset.secretSetting]);
    });
}


function customSecretRow(key = '', value = '') {
    const id = `custom-secret-${Math.random().toString(36).slice(2)}`;
    const ordinal = document.querySelectorAll('[data-custom-secret-row]').length + 1;
    const row = document.createElement('div');
    row.className = 'settings-custom-secret-row';
    row.dataset.customSecretRow = '1';
    row.dataset.originalKey = key;
    row.innerHTML = `
        <div class="form-field ui-field settings-custom-secret-key"><label for="${id}-key">Custom key ${ordinal}</label><input id="${id}-key" name="custom-key" type="text" class="ui-control" data-custom-secret-key value="${escapeHtml(key)}" placeholder="SLACK_WEBHOOK_URL" spellcheck="false"></div>
        <div class="form-field ui-field settings-custom-secret-value"><label for="${id}">Value for ${escapeHtml(key || `custom key ${ordinal}`)}</label><div class="secret-input-row">
            <input id="${id}" name="custom-value" data-custom-secret-value class="secret-input ui-control" type="password" value="${escapeHtml(value || '')}" placeholder="Secret value">
            <button type="button" class="btn btn-default secret-toggle" data-target="${id}" data-row-secret-toggle>Show</button>
            <button type="button" class="btn btn-default secret-clear" data-target="${id}" data-row-secret-clear>Clear</button>
        </div></div>
        <button type="button" class="btn btn-default settings-custom-secret-remove" data-custom-secret-remove>Remove</button>`;
    bindSecretInputs(row);
    row.querySelector('[data-custom-secret-value]').dataset.appliedValue = value;
    row.querySelector('[data-custom-secret-key]').addEventListener('input', (event) => {
        row.querySelector(`label[for="${id}"]`).textContent = `Value for ${event.target.value.trim() || `custom key ${ordinal}`}`;
    });
    row.querySelector('[data-custom-secret-remove]')?.addEventListener('click', () => {
        if (row.dataset.originalKey) { row.dataset.removeCustomSecret = '1'; row.hidden = true; }
        else row.remove();
        markSettingsDirty();
    });
    return row;
}

function renderCustomSecrets(root, settings) {
    const host = root.querySelector('#custom-secrets-list');
    if (!host) return;
    host.innerHTML = '';
    const keys = Array.isArray(settings?._meta?.custom_secret_keys) ? settings._meta.custom_secret_keys : [];
    keys.forEach((key) => host.appendChild(customSecretRow(key, settings[key] || '')));
    if (!keys.length) host.innerHTML = '<div class="muted">No custom keys yet.</div>';
}

function renderRequestedSkillSecrets(root, skills, settings) {
    const host = root.querySelector('#skill-requested-secrets');
    if (!host) return;
    const keys = [];
    (Array.isArray(skills) ? skills : []).forEach((skill) => {
        (skill?.grants?.requested_keys || []).forEach((key) => {
            const normalized = String(key || '').trim();
            if (normalized && !BASE_SECRET_KEYS.has(normalized)) keys.push(normalized);
        });
    });
    const unique = Array.from(new Set(keys)).sort((a, b) => a.localeCompare(b));
    if (!unique.length) { host.innerHTML = '<div class="muted">No skill-requested secrets.</div>'; return; }
    host.innerHTML = '';
    unique.forEach((key, idx) => {
        const id = `requested-secret-${idx}`;
        const el = document.createElement('div');
        el.className = 'settings-requested-secret-row';
        el.innerHTML = `<div class="form-field ui-field"><label for="${id}">${escapeHtml(key)}</label><div class="secret-input-row">
            <input id="${id}" name="${escapeHtml(key)}" data-secret-setting="${escapeHtml(key)}" class="secret-input ui-control" type="password" value="${escapeHtml(settings[key] || '')}" placeholder="Secret value">
            <button type="button" class="btn btn-default secret-toggle" data-target="${id}" data-row-secret-toggle>Show</button>
            <button type="button" class="btn btn-default secret-clear" data-target="${id}" data-row-secret-clear>Clear</button>
        </div></div>`;
        el.querySelector('.secret-input').dataset.appliedValue = settings[key] || '';
        bindSecretInputs(el); host.appendChild(el);
    });
}

// A declarative extension form must show what is STORED. Rendering the bare
// schema and posting it overwrote real values with the schema's first option —
// the bundled Telegram skill unbound its owner chat id that way. The host reads
// the current values from the SAME route it posts to; a plugin with no GET
// handler (404/405) keeps today's empty form, and any other outcome is an
// unknown read whose Save must not overwrite what we could not see.
const EXTENSION_VALUES_UNREADABLE = 'Current values could not be read; Save is disabled so it does not overwrite them. Use Reload Settings to retry.';

const extensionFormKey = (section, component, idx) =>
    `${section.key || `${section.skill}:${section.section_id}`}:${component.id || idx}`;

/** Read one form's stored values from its own route. Exported for node tests. */
export async function readExtensionFormValues(skill, route) {
    try {
        const resp = await apiFetch(extensionRoutePath(skill, route));
        if (resp.status === 404 || resp.status === 405) return { values: {}, blocked: false };
        if (!resp.ok) return { values: {}, blocked: true };
        const data = await resp.json();
        if (!data || typeof data !== 'object' || Array.isArray(data)) return { values: {}, blocked: true };
        return { values: data, blocked: false };
    } catch {
        return { values: {}, blocked: true };
    }
}

export async function renderExtensionSettingsSections(root, sections, { isCurrent = () => true } = {}) {
    const host = root.querySelector('#extension-settings-sections');
    if (!host) return;
    const items = Array.isArray(sections) ? sections : [];
    if (!items.length) {
        host.innerHTML = '<div class="muted">No extension settings registered.</div>';
        return;
    }
    const hydrated = new Map();
    await Promise.all(items.flatMap((section) => (Array.isArray(section.render?.components) ? section.render.components : [])
        .map(async (component, idx) => {
            const rawRoute = component.route || component.api_route || '';
            // Only a component with fields has values to read; an action's route is a
            // side effect and must not be probed on every Settings load.
            if (!['form', 'action'].includes(String(component.type || '')) || !cleanExtensionRoute(rawRoute)) return;
            if (!(Array.isArray(component.fields) && component.fields.length)) return;
            hydrated.set(extensionFormKey(section, component, idx),
                await readExtensionFormValues(section.skill || '', rawRoute));
        })));
    // A newer load or an owner edit may have landed while those reads were in
    // flight; a stale hydration must never overwrite the newer render.
    if (!isCurrent()) return;
    const formSpecs = new Map();
    const componentHtml = (section, component, idx) => {
        const type = String(component.type || '');
        if (type === 'markdown') {
            return `<div class="settings-section-copy">${escapeHtml(component.text || '')}</div>`;
        }
        if (type === 'json') {
            return `<details class="widget-json"><summary>${escapeHtml(component.label || 'JSON')}</summary><pre>${escapeHtml(JSON.stringify(component.value || component.data || {}, null, 2))}</pre></details>`;
        }
        if (type === 'form' || type === 'action') {
            const fields = Array.isArray(component.fields) ? component.fields : [];
            const rawRoute = component.route || component.api_route || '';
            if (!cleanExtensionRoute(rawRoute)) {
                return '<div class="settings-inline-note">Invalid extension settings route.</div>';
            }
            const formKey = extensionFormKey(section, component, idx);
            formSpecs.set(formKey, component);
            const { values = {}, blocked = false } = hydrated.get(formKey) || {};
            const disabled = Boolean(component.disabled);
            const fieldOptions = {
                disabled,
                fieldClass: 'form-field ui-field',
                inlineClass: 'settings-extension-checkbox ui-field ui-field-inline',
                helpClass: 'settings-inline-note ui-field-help',
            };
            return `
                <form class="settings-extension-form" data-extension-settings-form data-extension-settings-key="${escapeHtml(formKey)}" data-skill="${escapeHtml(section.skill || '')}" data-route="${escapeHtml(rawRoute)}"${blocked ? ' data-extension-settings-blocked="1"' : ''}>
                    <div class="form-grid two">${fields.map((field) => renderSafeField(field, values, fieldOptions)).join('')}</div>
                    <button class="btn btn-primary btn-sm" type="submit"${disabled || blocked ? ' disabled' : ''}>${escapeHtml(component.submit_label || component.label || 'Save')}</button>
                    <div class="settings-inline-status" data-extension-settings-status${blocked ? ` data-tone="${normalizeTone('warn')}"` : ''}>${blocked ? escapeHtml(EXTENSION_VALUES_UNREADABLE) : ''}</div>
                </form>
            `;
        }
        return `<div class="settings-inline-note">Unsupported extension settings component ${idx + 1}: ${escapeHtml(type || 'unknown')}</div>`;
    };
    host.innerHTML = items.map((section) => {
        const title = escapeHtml(section.title || section.section_id || section.key || 'Extension settings');
        const skill = escapeHtml(section.skill || '');
        const components = Array.isArray(section.render?.components) ? section.render.components : [];
        return `
            <article class="settings-extension-section">
                <div class="settings-extension-section-head">
                    <strong>${title}</strong>
                    ${skill ? `<span class="settings-inline-note">from ${skill}</span>` : ''}
                </div>
                <div class="settings-extension-components">
                    ${components.length ? components.map((component, idx) => componentHtml(section, component, idx)).join('') : '<div class="muted">No declarative components.</div>'}
                </div>
            </article>
        `;
    }).join('');
    host.querySelectorAll('[data-extension-settings-form]').forEach((form) => {
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const status = form.querySelector('[data-extension-settings-status]');
            const skill = form.dataset.skill || '';
            const route = form.dataset.route || '';
            const formKey = form.dataset.extensionSettingsKey || `${skill}:${route}`;
            const spec = formSpecs.get(formKey) || {};
            const requestKey = `${skill}:${route}`;
            if (!skill || !route || spec.disabled || form.dataset.extensionSettingsBlocked
                || pendingExtensionSettings.has(requestKey)) return;
            const values = collectSafeFieldValues(form, spec.fields || []);
            const button = form.querySelector('button[type="submit"]');
            const idleLabel = spec.submit_label || spec.label || 'Save';
            pendingExtensionSettings.add(requestKey);
            if (button) {
                button.disabled = true;
                button.textContent = spec.busy_label || 'Saving…';
            }
            setInlineStatus(status, 'Saving...', 'muted');
            try {
                const cleanRoute = cleanExtensionRoute(route);
                if (!cleanRoute) throw new Error('invalid extension settings route');
                const resp = await apiFetch(extensionRoutePath(skill, route), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(values),
                });
                const data = await resp.json().catch(() => ({}));
                if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);
                const outcome = extensionActionStatus(data);
                setInlineStatus(status, outcome.text, outcome.tone);
            } catch (err) {
                setInlineStatus(status, err.message || String(err), 'danger');
            } finally {
                pendingExtensionSettings.delete(requestKey);
                if (button) {
                    button.disabled = Boolean(spec.disabled);
                    button.textContent = idleLabel;
                }
            }
        });
    });
}

function collectSecretValue(id, body) {
    const input = byId(id);
    if (!input) return;
    const settingKey = input.dataset.secretSetting;
    if (!settingKey) return;
    if (input.dataset.forceClear === '1') {
        body[settingKey] = '';
        return;
    }
    const value = input.value;
    if (value && value !== input.dataset.appliedValue) body[settingKey] = value;
}


/**
 * Pure predicate (v6.82 P2): should the collapsed Settings "More providers"
 * section auto-open? True only for a USABLE credential — a provider API key,
 * a GigaChat OAuth credential, or a COMPLETE GigaChat basic-auth pair. Base
 * URLs/scope/TLS fields always carry shipped defaults and never count.
 * Exported for dependency-free node tests.
 */
export function moreProvidersCredentialConfigured({
    cloudruKey = '', minimaxKey = '', deepseekKey = '', gigachatCredentials = '', gigachatUser = '', gigachatPassword = '',
} = {}) {
    const has = (v) => Boolean(String(v ?? '').trim());
    return has(cloudruKey)
        || has(minimaxKey)
        || has(deepseekKey)
        || has(gigachatCredentials)
        || (has(gigachatUser) && has(gigachatPassword));
}

export function providerTestStatusText(result = {}) {
    if (result?.ok === true) return 'Works';
    const reason = String(result?.error || '').trim();
    return reason ? `Not ready — ${reason}` : 'Not ready';
}

export function providerTestNetworkErrorStatus() {
    return 'Not ready';
}

export function providerTestResultIsCurrent({
    sentGeneration, currentGeneration, sentFingerprint, currentFingerprint,
} = {}) {
    return sentGeneration === currentGeneration && sentFingerprint === currentFingerprint;
}

// Decision 16=A (#285): the settings "Restart now" action reuses the existing
// owner command contract — the same WS `/restart` the chat header sends. The
// whole confirm-and-send flow lives here (node-tested, panic-flow precedent):
// the click handler only injects real deps. queue:false keeps a disconnected
// page from silently queueing a destructive command for a later reconnect.
export async function confirmAndSendRestart({ openConfirmDialog: confirmDialog, ws: socket }) {
    const confirmed = await confirmDialog({
        title: 'Restart agent',
        body: 'All running and queued tasks stop, then the agent process restarts.\nSaved settings apply after the restart.',
        confirmLabel: 'Restart',
        danger: true,
    });
    if (!confirmed) return 'cancelled';
    const result = socket?.send?.({ type: 'command', cmd: '/restart' }, { queue: false });
    return result?.status === 'sent' ? 'sent' : 'not_connected';
}

function bindLanguageSegments(page) {
    const buttons = Array.from(page.querySelectorAll('[data-language-group] [data-language-value]'));
    const sync = (lang) => buttons.forEach((button) => {
        const on = button.dataset.languageValue === lang;
        button.classList.toggle('active', on);
        button.setAttribute('aria-pressed', String(on));
    });
    sync(currentLanguage() || storedLanguage());
    window.addEventListener('ouro:language-changed', (event) => sync(event.detail?.language || 'en'));
    buttons.forEach((button) => button.addEventListener('click', async () => {
        const lang = button.dataset.languageValue;
        if (lang === currentLanguage()) return;
        await setLanguage(lang);
        apiClient.saveUiPreferences({ language: lang }).catch(() => showToast('Language choice could not be saved.', 'error'));
    }));
}

/** Settings -> Behavior display toggle for the agent's reasoning rows. Like the
    theme control it applies + persists on its own and must never touch the
    settings draft, so its change event stops before the page-level dirty
    listener sees it. */
function bindReasoningToggle(page) {
    const box = page.querySelector('#ui-show-reasoning');
    if (!box) return;
    const sync = () => { box.checked = isReasoningVisible(); };
    box.addEventListener('change', (event) => {
        event.stopPropagation();
        const show = setReasoningVisible(box.checked);
        apiClient.saveUiPreferences({ show_reasoning: show })
            .catch(() => showToast('Reasoning display choice could not be saved.', 'error'));
    });
    window.addEventListener(REASONING_VISIBILITY_EVENT, sync);
    sync();
}

export function initSettings({ state, setBeforePageLeave, ws } = {}) {
    const page = document.createElement('div');
    page.id = 'page-settings';
    page.className = 'page app-page-glass';
    page.innerHTML = renderSettingsPage();
    document.getElementById('content').appendChild(page);

    const activateSettingsTab = (tabName) => {
        if (typeof page.activateSettingsTab === 'function') {
            page.activateSettingsTab(tabName);
        }
    };
    const disposeSettingsTabs = bindSettingsTabs(page, { state });
    bindSecretInputs(page);
    bindEffortSegments(page);
    // Theme and language are owner-local UI preferences, not part of the settings
    // draft: each applies on click and persists on its own, never marking the page dirty.
    bindThemeSegments(page);
    bindLanguageSegments(page);
    bindReasoningToggle(page);
    const disposeLocalModel = bindLocalModelControls({ state });
    // Best-effort About version from /api/health.
    apiFetch('/api/health')
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then((d) => {
            const verEl = document.getElementById('about-version');
            if (verEl) verEl.textContent = formatDualVersion(d);
        })
        .catch(() => { /* about version is best-effort */ });
    let currentSettings = {};
    let extensionRefreshPending = false;
    let settingsLoaded = false;
    let settingsBaseline = '';
    let settingsDirty = false;
    let draftRevision = 0;
    let loadSequence = 0;
    let settingsSaving = false;
    let saveOutcomeUnknown = false;
    let validationAttempted = false;
    const providerTestGenerations = new Map();
    const providerTestsInFlight = new Set();
    const modelRoles = createModelRolesEditor({ hostId: 'settings-model-roles',
        onChange: (settings) => { syncProcessingPreference(settings); onSettingsEdited(); } });
    modelRoles.mount();
    initMcpSettings({ onChange: onSettingsEdited });
    initReviewerSlots({ onChange: () => onSettingsEdited() });
    initSubagentsSection({
        onChange: (setting) => { adoptSubagentRoster({ OUROBOROS_SUBAGENTS: setting }); onSettingsEdited(); },
        // A judged roster may clear only the validation footer it authored.
        // A cadence or other field error keeps its typed subject and survives.
        onJudged: (clean) => {
            const status = byId('settings-status');
            if (clean && status.dataset.owner === 'validation'
                    && status.dataset.subject === 'subagents') setStatus('', 'ok');
        },
        isOuterDraftClean: () => !settingsDirty,
        onGeneratedApply: () => {
            if (settingsLoaded && !settingsDirty) setSettingsCleanBaseline();
        },
        previewGenerated: ({ subscriptionsConnected }) => apiClient.previewOnboardingSubagents(
            availableSubagentsPreviewPayload(collectBody(), subscriptionsConnected),
        ),
    });
    initHarnessAccounts();

    function syncProcessingPreference(settings) {
        setSubagentsProcessingPreference(settings[PROCESSING_PREFERENCE_KEY]);
        setReviewerProcessingPreference(settings[PROCESSING_PREFERENCE_KEY], modelRoleMap(settings[MODEL_PROCESSING_PREFERENCES_KEY]));
    }

    function syncSettingsLoadState() {
        const saveBtn = byId('btn-save-settings');
        if (saveBtn) {
            saveBtn.disabled = !settingsLoaded || settingsSaving || saveOutcomeUnknown;
            saveBtn.title = settingsLoaded
                ? (saveOutcomeUnknown ? 'Reload Settings to check the previous save before saving again.' : '')
                : 'Reload current settings successfully before saving.';
        }
    }

    function syncRuntimeModeBridgeState() {
        const hasBridge = Boolean(window.pywebview?.api?.confirm_runtime_mode_change);
        const group = document.querySelector('[data-runtime-mode-group]');
        if (group) {
            group.title = hasBridge
                ? 'Runtime mode changes require native launcher confirmation and restart.'
                : 'Runtime mode changes are saved through the owner endpoint and take effect after restart.';
        }
        document.querySelectorAll('[data-runtime-mode-group] [data-effort-value]').forEach((button) => {
            button.disabled = false;
        });
    }

    function syncPostTaskEvolutionUi() {
        const mode = byId('s-post-task-evolution-mode')?.value || 'off';
        page.querySelectorAll('[data-evo-every-n-row]').forEach((row) => {
            row.hidden = mode !== 'every_n';
        });
    }

    // Top-level keys in sorted order: the dirty check compares these strings,
    // and the status-settle baseline fold below inserts keys AFTER the fact —
    // equality must not depend on object insertion order. Nested values keep
    // native stringify (both sides build them through the same code path).
    function stableSerializeDraft(draft) {
        return JSON.stringify(Object.fromEntries(
            Object.entries(draft).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)),
        ));
    }

    function snapshotSettingsDraft() {
        return stableSerializeDraft({
            ...collectBody(),
            // Raw controls retain invalid/empty values and owner-only settings
            // that the transport payload intentionally normalizes or omits.
            controls: Array.from(page.querySelectorAll('input[id^="s-"], select[id^="s-"], textarea[id^="s-"], [data-secret-setting], [data-model-role-context]'),
                (input) => [input.id, input.value, input.checked, input.validity?.badInput || false, input.dataset.forceClear || '']),
            customSecrets: readCustomSecretDraft(page),
        });
    }

    function setSettingsCleanBaseline() {
        settingsBaseline = snapshotSettingsDraft();
        settingsDirty = false;
        const indicator = byId('settings-unsaved-indicator');
        if (indicator) indicator.classList.remove('is-visible');
    }

    function updateSettingsDirtyState() {
        const nextDirty = settingsLoaded && settingsBaseline
            ? snapshotSettingsDraft() !== settingsBaseline : draftRevision > 0;
        if (nextDirty === settingsDirty) return;
        settingsDirty = nextDirty;
        const indicator = byId('settings-unsaved-indicator');
        if (indicator) indicator.classList.toggle('is-visible', settingsDirty);
    }

    function onSettingsEdited() {
        draftRevision += 1;
        if (validationAttempted) renderValidation();
        updateSettingsDirtyState();
    }

    let baselineSettleDisposer = null;
    function armCleanBaselineOnStatusSettle(revision) {
        // The sections' Claudexor status probe is fire-and-forget, so the
        // baseline can be taken before the store-gated collectors have their
        // facts — and their output changes when a snapshot lands (the accounts
        // facet, the later include-models upgrade). Absent owner edits, no
        // store arrival may read as an unsaved change; and every owner edit
        // flips settingsDirty through its own input handler BEFORE any store
        // notify, so re-baselining while the draft is clean can never mask
        // one. Deliberately NOT a one-shot on everSettled: an earlier
        // model-less read may have settled the store long before the upgrade
        // this page's collectors actually feed on. A BARE subscription, not a
        // status surface: this observer must react to snapshots the sections'
        // own surfaces fetch, never arm the polling chain itself.
        baselineSettleDisposer?.();
        baselineSettleDisposer = claudexorStatus.subscribe(() => {
            // CLEAN drafts only. A late availability repaint may change status
            // copy but never the canonical actor draft; re-baselining a DIRTY
            // page would still absorb the owner's real row edit into the clean
            // baseline, so it remains forbidden.
            // After an edit, any availability-only difference remains in the
            // draft comparison until the next load/save; it cannot absorb edits.
            if (revision === draftRevision && !settingsDirty && settingsLoaded) setSettingsCleanBaseline();
        });
    }

    function discardUnsavedSettingsDraft() {
        applySettings(currentSettings || {});
        discardReviewerSlotsDraft();
        renderCustomSecrets(page, currentSettings || {});
        validationAttempted = false;
        paintSettingsFieldErrors(page, []);
        setSettingsCleanBaseline();
        setStatus('', 'ok');
    }

    function syncAutoGrantBridgeState() {
        const hasBridge = Boolean(window.pywebview?.api?.request_auto_grant_reviewed_skills_change);
        const checkbox = byId('s-auto-grant-reviewed-skills');
        const label = checkbox?.closest('.local-toggle');
        if (checkbox) checkbox.disabled = false;
        if (label) {
            label.title = hasBridge
                ? 'Requires native confirmation. Applies only after a fresh executable skill review and only to manifest-declared grants for that exact content hash.'
                : 'Uses the owner endpoint. Applies only after a fresh executable skill review and only to manifest-declared grants for that exact content hash.';
        }
    }

    function applySettings(s) {
        setupContract = s?._meta?.setup_contract || setupContract || {};
        // A settings (re)load replaces the values every provider verdict was
        // earned against — programmatic assignment fires no 'input' events, so
        // the expiry listener cannot see it; expire the verdicts here.
        Object.keys(PROVIDER_TEST_INPUTS).forEach((provider) => {
            providerTestGenerations.set(provider, (providerTestGenerations.get(provider) || 0) + 1);
        });
        page.querySelectorAll('[data-provider-test-status]').forEach((el) => setInlineStatus(el, '', 'muted'));
        applySecretInputs(page, s);
        INPUT_FIELDS.forEach(([id, key, fallback = '']) => applyInputValue(id, fallback && !s[key] ? fallback : s[key]));
        VALUE_FIELDS.forEach(([id, key, fallback]) => { byId(id).value = s[key] || fallback; });
        modelRoles.load(s, { ...setupContract, modelSlots: setupModelSlots().map((slot) => ({
            ...slot, inputId: slot.settingsInputId,
        })) });
        applyCheckboxValue('s-auto-grant-reviewed-skills', s.OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS);
        // Owner-facing mutative-subagents control shows the EFFECTIVE state when it
        // is binary-representable: an explicit value, or unset in advanced/pro
        // (every acting surface on = "On"). Unset in LIGHT mode is surface-aware
        // (external_workspace/genesis stay on, self_worktree off — see
        // config.get_allow_mutative_subagents), so neither Off nor On is truthful
        // there: it displays as "Auto". Picking Auto saves the empty value
        // (collectBody maps any non-on/off segment to ''), so the mode default
        // keeps deciding.
        const rawMutative = String(s.OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS ?? '').trim().toLowerCase();
        const runtimeMode = String(s.OUROBOROS_RUNTIME_MODE || 'advanced').trim().toLowerCase();
        const mutativeInput = byId('s-allow-mutative-subagents');
        mutativeInput.dataset.rawValue = rawMutative;
        delete mutativeInput.dataset.effortTouched;
        mutativeInput.value =
            ({ true: 'on', false: 'off' }[rawMutative] || (runtimeMode === 'light' ? 'auto' : 'on'));
        // The actor list lives next to it in Agents → Available subagents.
        applySubagentsSettings(s);
        syncProcessingPreference(s);
        // The Review-lanes «Configured subagent» selects reference the SAME
        // roster; adopt it from the same loaded document.
        adoptSubagentRoster(s);
        // …and both editors offer the API providers THIS document has a
        // credential for, named by the setup contract. Derived from the loaded
        // settings, so a key added under Accounts shows up on the next load
        // rather than being typed as a prefix (docs/DESIGN.md §7).
        setReviewerSourceContext({ settings: s, providerProfiles: setupContract.providerProfiles });
        setSubagentsSourceContext(s, setupContract.providerProfiles);
        // Post-task evolution: one owner-facing selector maps to enable + cadence.
        const evoEnabled =
            ({ true: 'on', '1': 'on', on: 'on', false: 'off', '0': 'off', off: 'off' }[
                String(s.OUROBOROS_POST_TASK_EVOLUTION ?? '').trim().toLowerCase()] || 'off') === 'on';
        const evoCadence = String(s.OUROBOROS_POST_TASK_EVOLUTION_CADENCE || 'llm').trim().toLowerCase();
        // Use the SAME strict shape as the backend (^every_n:[1-9]\d*$) so a stale/
        // malformed value (e.g. every_nonsense, every_n:0) displays as llm — never as
        // Every-N:3, which a later Save would silently persist as periodic evolution.
        const everyNMatch = evoCadence.match(/^every_n:([1-9]\d*)$/);
        if (!evoEnabled) {
            byId('s-post-task-evolution-mode').value = 'off';
        } else if (everyNMatch) {
            byId('s-post-task-evolution-mode').value = 'every_n';
            byId('s-evo-cadence-n').value = everyNMatch[1];
        } else {
            byId('s-post-task-evolution-mode').value = 'llm';
        }
        NUMBER_FIELDS.forEach(([id, key, fallback, allowFalsy]) => {
            const value = s[key];
            if (allowFalsy ? value !== null && value !== undefined : value) byId(id).value = value;
            else byId(id).value = fallback;
        });
        (Array.isArray(setupContract.budgetFields) ? setupContract.budgetFields : []).forEach((field) => {
            const id = field.settingsInputId;
            const input = byId(id);
            if (!input) return;
            input.min = field.min || '0.01';
            input.step = field.step || 'any';
            input.value = s[field.settingKey] ?? field.default ?? '';
        });
        applyMcpSettings(s);
        syncMoreProvidersDisclosure();
        resetSecretClearFlags(page);
        syncEffortSegments(page);
        syncRuntimeModeBridgeState();
        syncPolicyState(page, s?._meta);
        syncPostTaskEvolutionUi();
        refreshSafetySkipCounter();  // fire-and-forget; fills the 24h audited-skip note
    }

    function syncMoreProvidersDisclosure() {
        // Auto-open the collapsed "More providers" section when a usable
        // provider CREDENTIAL inside it is configured, so a set-up
        // A configured provider in this section is never hidden. Non-secret inputs
        // (base URLs, scope, verify-ssl) always carry shipped defaults and
        // must NOT count as "configured". Runs after applySettings; never
        // force-closes an owner-opened section.
        const wrapper = byId('settings-more-providers');
        if (!wrapper) return;
        const value = (id) => {
            const input = byId(id);
            return input ? input.value : '';
        };
        if (moreProvidersCredentialConfigured({
            cloudruKey: value('s-cloudru-key'),
            minimaxKey: value('s-minimax-key'),
            deepseekKey: value('s-deepseek-key'),
            gigachatCredentials: value('s-gigachat-credentials'),
            gigachatUser: value('s-gigachat-user'),
            gigachatPassword: value('s-gigachat-password'),
        })) wrapper.open = true;
    }

    function _renderNetworkHint(meta) {
        const hint = document.getElementById('settings-lan-hint');
        if (!hint || !meta) return;
        if (meta.reachability === 'loopback_only') {
            hint.innerHTML = 'Bound to <code>localhost</code>: only accessible from this machine. Set Server Bind Host to <code>0.0.0.0</code>, save, and restart for LAN access.';
            hint.dataset.tone = 'info';
            hint.hidden = false;
        } else if (meta.reachability === 'lan_reachable') {
            const url = escapeHtml(meta.recommended_url || '');
            const warning = escapeHtml(meta.warning || '');
            hint.innerHTML = `LAN URL: <a href="${url}" target="_blank" rel="noopener">${url}</a>${warning ? ' — <strong>' + warning + '</strong>' : ''}`;
            hint.dataset.tone = meta.warning ? 'warn' : 'ok';
            hint.hidden = false;
        } else if (meta.reachability === 'host_ip_unknown') {
            const url = escapeHtml(meta.recommended_url || '');
            const warning = escapeHtml(meta.warning || '');
            hint.innerHTML = `Server is listening on non-localhost but LAN IP could not be detected automatically. Try <code>${url}</code>.${warning ? ' <strong>' + warning + '</strong>' : ''}`;
            hint.dataset.tone = 'warn';
            hint.hidden = false;
        } else {
            hint.hidden = true;
        }
    }

    async function loadSettings() {
        const sequence = ++loadSequence;
        const revision = draftRevision;
        const [data, extData] = await Promise.all([
            apiClient.settings(),
            apiClient.extensions().catch(() => ({})),
        ]);
        if (!data || typeof data !== 'object' || Array.isArray(data) || data.error) {
            throw new Error(data?.error || 'The server did not return a settings document.');
        }
        const sections = Array.isArray(extData?.live?.settings_sections)
            ? extData.live.settings_sections
            : [];
        if (sequence !== loadSequence || revision !== draftRevision) return false;
        currentSettings = data;
        applySettings(data);
        renderRequestedSkillSecrets(page, extData.skills || [], data);
        renderCustomSecrets(page, data);
        // This confirmed document can already be edited and saved. Optional
        // reviewer/status reads must not hold its baseline or Save capability.
        settingsLoaded = true;
        saveOutcomeUnknown = false;
        validationAttempted = false;
        paintSettingsFieldErrors(page, []);
        setSettingsCleanBaseline();
        armCleanBaselineOnStatusSettle(revision);
        _renderNetworkHint(data._meta);
        syncSettingsLoadState();
        // Extension settings forms read their stored values before rendering:
        // that is optional enrichment too, and it must not delay the clean
        // baseline above or absorb an owner edit made while it was pending.
        const isCurrent = () => sequence === loadSequence && revision === draftRevision;
        await Promise.all([renderExtensionSettingsSections(page, sections, { isCurrent }), reloadReviewerSlots({ isCurrent }), reloadSubagentsSection()]);
        if (sequence !== loadSequence || revision !== draftRevision) {
            updateSettingsDirtyState();
            return false;
        }
        // Optional enrichment belongs in a still-clean baseline, never in an
        // owner edit made while one of those reads was pending.
        setSettingsCleanBaseline();
        return true;
    }

    async function reloadSettingsWithFeedback() {
        if (settingsSaving) return;
        if (settingsDirty && !(await confirmDiscardSettings('reload Settings'))) return;
        const reloadSequence = loadSequence + 1;
        setStatus('Loading settings...', 'muted', 'load');
        try {
            const applied = await loadSettings();
            if (!applied && byId('settings-status').dataset.owner === 'load') {
                setStatus('Settings were not reloaded because your draft changed while loading. Your edits are kept.', 'warn', 'load');
            }
            try {
                await refreshModelCatalog({ button: byId('btn-refresh-model-catalog') });
                if (applied && byId('settings-status').dataset.owner === 'load' && !settingsDirty && !settingsSaving && !saveOutcomeUnknown) setStatus('Settings loaded', 'ok');
            } catch (error) {
                if (applied && byId('settings-status').dataset.owner === 'load' && !settingsDirty && !settingsSaving && !saveOutcomeUnknown) setStatus(
                    `Settings loaded. Model catalog refresh failed: ${error.message || error}`,
                    'warn'
                );
            }
        } catch (error) {
            if (reloadSequence !== loadSequence) return;
            settingsLoaded = false;
            syncSettingsLoadState();
            setStatus(
                `Failed to load current settings. Save is disabled until reload succeeds: ${error.message || error}`,
                'warn'
            );
        }
    }

    async function refreshSettingsAfterExtensionChange(reason = 'skills changed') {
        if (extensionRefreshPending || settingsSaving || saveOutcomeUnknown) return;
        if (settingsDirty) {
            setStatus(`Settings changed externally (${reason}). Reload after saving or discarding your draft.`, 'warn');
            return;
        }
        extensionRefreshPending = true;
        try {
            if (!(await loadSettings())) return;
            setStatus('Settings refreshed', 'ok');
        } catch (error) {
            setStatus(`Settings refresh failed: ${error.message || error}`, 'warn');
        } finally {
            extensionRefreshPending = false;
        }
    }

    function collectBody() {
        const fieldValue = (id) => byId(id)?.value || '';
        const mutativeInput = byId('s-allow-mutative-subagents');
        const rawMutative = String(mutativeInput?.dataset?.rawValue ?? '').trim().toLowerCase();
        const mutativeTouched = mutativeInput?.dataset?.effortTouched === '1';
        const body = {
            OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS: byId('s-auto-grant-reviewed-skills')?.checked ? 'true' : 'false',
            OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS: mutativeTouched
                ? ({ on: 'true', off: 'false' }[mutativeInput?.value] ?? '')
                : (rawMutative ? ({ true: 'true', false: 'false' }[rawMutative] ?? rawMutative) : ''),
            ...collectMcpSettings(),
            // 6.1: the ONE structured reviewer-slot setting; {} until the rows
            // view has loaded, so an unrelated save cannot blank it.
            ...collectReviewerSlots(),
            // Saved config and live availability are independent: a loaded
            // actor list is collected even when status is down; only an
            // unloaded/unparseable editor omits the key on an unrelated save.
            ...collectSubagentsSettings(),
        };
        Object.assign(body, modelRoles.collect());
        INPUT_FIELDS.forEach(([id, key, fallback = '']) => {
            const value = fieldValue(id).trim();
            body[key] = key === 'OUROBOROS_SERVER_HOST' ? value || fallback : value || '';
        });
        VALUE_FIELDS
            // Owner-only keys travel through their audited owner endpoints, never
            // the generic settings POST (safety_mode joined runtime/context, r4).
            .filter(([, key]) => key !== 'OUROBOROS_RUNTIME_MODE' && key !== 'OUROBOROS_CONTEXT_MODE' && key !== 'OUROBOROS_SAFETY_MODE')
            .forEach(([id, key]) => { body[key] = fieldValue(id); });
        NUMBER_FIELDS.forEach(([id, key, fallback]) => { body[key] = readInt(id, fallback); });
        (Array.isArray(setupContract.budgetFields) ? setupContract.budgetFields : []).forEach((field) => {
            const id = field.settingsInputId;
            const input = byId(id);
            if (!input) return;
            const raw = String(input.value || '').trim();
            const parsed = Number(raw);
            const value = Number.isFinite(parsed) && parsed > 0 ? parsed : raw;
            if (String(value) !== String(currentSettings?.[field.settingKey] ?? field.default)) {
                body[field.settingKey] = value;
            }
        });
        // Post-task evolution: compose the legacy enable + cadence settings from
        // the single owner-facing selector.
        const evoCadMode = byId('s-post-task-evolution-mode').value;
        body.OUROBOROS_POST_TASK_EVOLUTION = evoCadMode === 'off' ? 'false' : 'true';
        body.OUROBOROS_POST_TASK_EVOLUTION_CADENCE = evoCadMode === 'every_n'
            ? `every_n:${Math.max(1, parseInt(byId('s-evo-cadence-n').value, 10) || 3)}`
            : 'llm';

        page.querySelectorAll('[data-secret-setting]').forEach((input) => {
            collectSecretValue(input.id, body);
        });
        Object.assign(body, collectCustomSecrets().values);
        return body;
    }

    function collectCustomSecrets() {
        const customKeys = new Set(currentSettings?._meta?.custom_secret_keys || []);
        return collectCustomSecretDraft(readCustomSecretDraft(page),
            Object.keys(currentSettings).filter((key) => !customKeys.has(key)));
    }

    function validationSummary(errors) {
        return errors.length > 1 ? `${errors[0]} (${errors.length} fields need attention.)` : errors[0] || '';
    }

    function collectValidation() {
        const fields = validateMcpSettings();
        const cadence = byId('s-evo-cadence-n');
        if (byId('s-post-task-evolution-mode')?.value === 'every_n' && !/^[1-9]\d*$/.test(cadence.value.trim())) {
            fields.push({ input: cadence, message: 'Every-N cadence needs a whole number ≥ 1.' });
        }
        page.querySelectorAll('input[id^="s-"], select[id^="s-"], textarea[id^="s-"]').forEach((input) => {
            if (input === cadence || fields.some((error) => error.input === input)
                    || input.hasAttribute('data-model-role-context') || !input.willValidate || input.validity.valid) return;
            const label = input.labels?.[0]?.textContent || input.name || input.id;
            fields.push({ input, message: `${label.trim()}: ${input.validationMessage}` });
        });
        const rows = [...page.querySelectorAll('[data-custom-secret-row]')];
        collectCustomSecrets().errors.forEach(({ index, field, message }) => {
            if (rows[index]?.dataset.judged === '1') fields.push({ input: rows[index].querySelector(`[data-custom-secret-${field}]`), message });
        });
        const groups = [
            ['fields', fields.map(({ message }) => message)],
            ['models', modelRoles.validateAll()],
            ['subagents', validateSubagentsDraft().map((error) => `Available subagents: ${error}`)],
            ['reviewers', validateReviewerSlots()],
        ];
        const messages = groups.flatMap(([, rows]) => rows).filter(Boolean);
        const subject = groups.find(([, rows]) => rows.some(Boolean))?.[0] || '';
        return { fields, messages, subject };
    }

    function renderValidation() {
        const { fields, messages, subject } = collectValidation();
        paintSettingsFieldErrors(page, fields);
        if (byId('settings-status').dataset.owner === 'validation') {
            if (messages.length) setStatus(validationSummary(messages), 'warn', 'validation', subject);
            else setStatus('', 'ok');
        }
        return { messages, subject };
    }

    async function confirmDiscardSettings(action) {
        return openConfirmDialog({
            title: 'Unsaved settings',
            body: `You have unsaved settings changes. Discard them and ${action}?`,
            confirmLabel: 'Discard and continue', cancelLabel: 'Stay',
        });
    }

    async function saveRuntimeModeViaNativeBridgeIfNeeded(nextMode) {
        const currentMode = currentSettings?.OUROBOROS_RUNTIME_MODE || 'advanced';
        if (nextMode === currentMode) return null;
        // The native bridge is confirmation-only.  Older shells do not expose
        // that method; fall back to the same in-app dialog so they never receive
        // the legacy mutating request_runtime_mode_change call (which cannot
        // represent Cyber Pro). Every surface writes through one owner endpoint.
        const nativeConfirm = window.pywebview?.api?.confirm_runtime_mode_change;
        const confirmed = nativeConfirm
            ? (await nativeConfirm(nextMode))?.confirmed === true
            : await openConfirmDialog({
                title: 'Change runtime mode',
                body: `Change Ouroboros runtime mode from ${currentMode} to ${nextMode}? The change takes effect after restart.`,
                confirmLabel: 'Change mode',
            });
        if (!confirmed) {
            const result = { ok: false, saved: false, error: 'Runtime mode change cancelled.' };
            throw Object.assign(new Error(result.error), { body: result });
        }
        const result = await apiClient.ownerRuntimeMode(nextMode);
        if (!result || result.ok !== true) {
            throw Object.assign(new Error(result?.error || 'Runtime mode change failed.'), { body: result });
        }
        return result;
    }

    async function saveAutoGrantViaNativeBridgeIfNeeded(nextEnabled) {
        const currentEnabled = isTruthySetting(currentSettings?.OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS);
        if (nextEnabled === currentEnabled) return null;
        const bridge = window.pywebview?.api?.request_auto_grant_reviewed_skills_change;
        // Browser-side confirm only; the pywebview bridge path stays untouched.
        const result = bridge
            ? await bridge(nextEnabled)
            : ((await openConfirmDialog({
                title: 'Reviewed-skill auto-grant',
                body: `${nextEnabled ? 'Enable' : 'Disable'} reviewed-skill auto-grant? It only applies after a fresh executable review for the current content hash.`,
                confirmLabel: nextEnabled ? 'Enable' : 'Disable',
            }))
                ? await apiClient.ownerAutoGrant(nextEnabled)
                : { ok: false, saved: false, error: 'Reviewed-skill auto-grant change cancelled.' });
        if (!result || result.ok !== true) {
            throw Object.assign(new Error(result?.error || 'Reviewed-skill auto-grant change was cancelled.'), { body: result });
        }
        return result;
    }

    async function saveSafetyModeViaOwnerEndpointIfNeeded(next) {
        // Owner-only, dropped from the generic /api/settings POST — saved through the
        // dedicated audited endpoint. Confirm on LOWERING coverage (full > light > off).
        const current = currentSettings?.OUROBOROS_SAFETY_MODE || 'full';
        if (next === current) return null;
        const lowering = (_SAFETY_MODE_RANK[next] ?? 2) < (_SAFETY_MODE_RANK[current] ?? 2);
        if (lowering) {
            const ok = await openConfirmDialog({
                title: 'Lower safety supervisor',
                body: `Lower the LLM safety supervisor from ${current} to ${next}?\n\n` +
                    `The deterministic sandbox, protected-path policy, and light-mode guards STAY ON in every mode. ` +
                    `Only the LLM safety-check layer is reduced, and every waved-through check is logged as an audit event.`,
                confirmLabel: 'Lower safety mode',
                danger: true,
            });
            if (!ok) throw Object.assign(new Error('Safety mode change was not confirmed.'), { body: { saved: false } });
        }
        const result = await apiClient.ownerSafetyMode(next);
        if (!result || result.ok !== true) {
            throw Object.assign(new Error(result?.error || 'Safety mode change failed.'), { body: result });
        }
        return result;
    }

    async function refreshSafetySkipCounter() {
        // 24h count of durable safety_mode_skip audit events, so the owner sees how much
        // the reduced coverage actually waved through.
        const el = byId('s-safety-skip-counter');
        if (!el) return;
        try {
            const data = await apiClient.logsTail('events', 2000);
            const cutoff = Date.now() - 24 * 3600 * 1000;
            const n = (data?.entries || []).filter((e) => {
                if (String(e?.type || '') !== 'safety_mode_skip') return false;
                const t = Date.parse(String(e?.ts || ''));
                return Number.isFinite(t) && t >= cutoff;
            }).length;
            // Honest window note: the count scans the recent events tail (2000), so a
            // very busy day can undercount — say "recent", never overclaim exactness.
            el.textContent = n > 0
                ? `${n} safety check(s) waved through in the last 24h (audited; recent events window).`
                : 'No safety checks waved through in the last 24h (recent events window).';
        } catch {
            el.textContent = '';
        }
    }

    async function saveContextModeViaOwnerEndpointIfNeeded(next) {
        const current = currentSettings?.OUROBOROS_CONTEXT_MODE || 'max';
        if (next === current) return null;
        const result = await apiClient.ownerContextMode(next);
        if (!result || result.ok !== true) {
            throw Object.assign(new Error(result?.error || 'Context mode change failed.'), { body: result });
        }
        return result;
    }

    markSettingsDirty = onSettingsEdited;
    syncSettingsLoadState();
    syncRuntimeModeBridgeState();
    syncAutoGrantBridgeState();
    reloadSettingsWithFeedback();

    if (typeof setBeforePageLeave === 'function') {
        // app.js showPage() awaits every beforePageLeave handler, so the async
        // dialog is legal here. Rapid double-navigation cannot double-fire the
        // discard: opening a second dialog resolves the first as false (stay),
        // so at most one confirmed leave runs discardUnsavedSettingsDraft().
        setBeforePageLeave(async ({ from }) => {
            if (from !== 'settings') return true;
            if (settingsSaving) return false;
            if (!settingsDirty) return true;
            const leave = await confirmDiscardSettings('leave Settings');
            if (leave) discardUnsavedSettingsDraft();
            return leave;
        });
    }

    page.addEventListener('input', onSettingsEdited);
    page.addEventListener('change', onSettingsEdited);
    page.addEventListener('click', (event) => {
        if (event.target.closest('[data-effort-value], .secret-clear, [data-row-secret-clear], [data-custom-secret-remove]')) {
            queueMicrotask(() => {
                syncPostTaskEvolutionUi();
                onSettingsEdited();
            });
        }
    });
    byId('btn-add-custom-secret')?.addEventListener('click', () => {
        const host = byId('custom-secrets-list');
        if (!host) return;
        if (host.querySelector('.muted')) host.innerHTML = '';
        const row = customSecretRow();
        host.appendChild(row);
        revealNewRow(row, row.querySelector('[data-custom-secret-key]'));
        markSettingsDirty();
    });

    window.addEventListener('ouro:skill-lifecycle', (event) => {
        const action = String(event.detail?.action || 'skills changed');
        refreshSettingsAfterExtensionChange(action);
    });
    window.addEventListener('ouro:settings-updated', (event) => {
        if (event.detail?.source === 'settings') return;
        const action = String(event.detail?.reason || 'settings changed');
        refreshSettingsAfterExtensionChange(action);
    });
    if (ws && typeof ws.on === 'function') {
        ws.on('extension_lifecycle', (event) => {
            const action = String(event?.action || 'extension lifecycle');
            refreshSettingsAfterExtensionChange(action);
        });
    }

    window.addEventListener('ouro:page-shown', (event) => {
        if (event.detail?.page === 'settings') refreshSettingsAfterExtensionChange('settings page shown');
    });

    const onModelCatalog = (event) => modelRoles.adoptCatalog(event.detail);
    const beforeUnload = (event) => {
        if (settingsDirty || settingsSaving) { event.preventDefault(); event.returnValue = ''; }
    };
    window.addEventListener('beforeunload', beforeUnload);
    document.addEventListener('settings-model-catalog:updated', onModelCatalog);
    window.addEventListener('pagehide', (event) => {
        if (event.persisted) return;
        disposeSettingsTabs();
        window.removeEventListener('beforeunload', beforeUnload);
        disposeLocalModel();
        baselineSettleDisposer?.();
        modelRoles.destroy();
        document.removeEventListener('settings-model-catalog:updated', onModelCatalog);
    });

    // Provider readiness probe: one short model request against the card draft.
    page.querySelector('[data-settings-panel="providers"]')?.addEventListener('click', async (event) => {
        const button = event.target instanceof Element ? event.target.closest('[data-provider-test]') : null;
        if (!button) return;
        const provider = button.dataset.providerTest;
        if (providerTestsInFlight.has(provider)) return;
        const status = page.querySelector(`[data-provider-test-status="${provider}"]`);
        const collectOverrides = () => {
            const overrides = {};
            for (const [inputId, settingKey] of Object.entries(PROVIDER_TEST_INPUTS[provider] || {})) {
                const input = byId(inputId);
                const value = (input?.value || '').trim();
                // Only owner-edited fields become overrides: saved secrets render
                // as MASKED placeholders (gateway mask_settings_secret), and echoing
                // a mask back as the credential would fail every already-saved key.
                // An untouched field means "test the saved value server-side"; an
                // edited-to-empty field (Clear included) sends an explicit empty
                // override so the probe tests the visible draft, not the old key.
                if (value !== (input?.dataset.appliedValue ?? '').trim()) {
                    overrides[settingKey] = value;
                }
            }
            return overrides;
        };
        const overrides = collectOverrides();
        const sentFingerprint = JSON.stringify(overrides);
        const sentGeneration = providerTestGenerations.get(provider) || 0;
        providerTestsInFlight.add(provider);
        setButtonBusy(button, true);
        if (status) setInlineStatus(status, 'Testing…', 'muted');
        const resultIsCurrent = () => providerTestResultIsCurrent({
            sentGeneration,
            currentGeneration: providerTestGenerations.get(provider) || 0,
            sentFingerprint,
            currentFingerprint: JSON.stringify(collectOverrides()),
        });
        try {
            const data = await apiClient.providerTest({ provider_id: provider, overrides });
            if (status && resultIsCurrent()) {
                setInlineStatus(status, providerTestStatusText(data), data?.ok ? 'ok' : 'danger');
            }
        } catch (_error) {
            if (status && resultIsCurrent()) {
                setInlineStatus(status, providerTestNetworkErrorStatus(), 'danger');
            }
        } finally {
            providerTestsInFlight.delete(provider);
            setButtonBusy(button, false);
        }
    });

    // A displayed verdict is only good for the draft it tested: the moment any
    // field of that card changes, the old OK/Failed would sit beside values it
    // never saw — clear it instead of letting it vouch for the new draft.
    page.querySelector('[data-settings-panel="providers"]')?.addEventListener('input', (event) => {
        const target = event.target;
        if (!(target instanceof Element) || !target.id) return;
        for (const [provider, inputs] of Object.entries(PROVIDER_TEST_INPUTS)) {
            if (target.id in inputs) {
                providerTestGenerations.set(
                    provider,
                    (providerTestGenerations.get(provider) || 0) + 1,
                );
                const status = page.querySelector(`[data-provider-test-status="${provider}"]`);
                if (status) setInlineStatus(status, '', 'muted');
                break;
            }
        }
    });

    byId('btn-refresh-model-catalog').addEventListener('click', async (event) => {
        await refreshModelCatalog({ button: event.currentTarget });
    });

    byId('btn-reload-settings')?.addEventListener('click', async () => {
        await reloadSettingsWithFeedback();
    });

    // #285: true from a restart-required save until the restart command is
    // actually sent — keeps the Restart now affordance across later saves.
    let restartPending = false;

    byId('btn-save-settings').addEventListener('click', async () => {
        if (settingsSaving || saveOutcomeUnknown) return;
        if (!settingsLoaded) {
            setStatus('Reload current settings successfully before saving.', 'warn');
            return;
        }
        // The owner just tried to commit the draft — every Save click is one,
        // whichever validation aborts it below — so from here the roster shows
        // its own errors beside the rows they name, not only in this status.
        noteSubagentsSaveAttempt();
        noteReviewerSlotsSaveAttempt();
        modelRoles.noteSaveAttempt();
        page.querySelectorAll('[data-custom-secret-row]').forEach((row) => { row.dataset.judged = '1'; });
        validationAttempted = true;
        const { messages: errors, subject } = renderValidation();
        if (errors.length) {
            setStatus(validationSummary(errors), 'warn', 'validation', subject);
            return;
        }
        const body = collectBody();
        loadSequence += 1;
        const sentRevision = draftRevision;
        const ownerDraft = {
            runtime: byId('s-runtime-mode').value || 'advanced',
            autoGrant: Boolean(byId('s-auto-grant-reviewed-skills').checked),
            context: byId('s-context-mode').value || 'max',
            safety: byId('s-safety-mode').value || 'full',
        };
        const subagentsChanged = subagentSettingsFingerprint(body.OUROBOROS_SUBAGENTS)
            !== subagentSettingsFingerprint(currentSettings?.OUROBOROS_SUBAGENTS);

        // Phase 1 of the save: the button goes busy and the status says so.
        // Capability probes on review-route changes make a save take seconds;
        // an idle "Save Settings" over that window reads as a dead click.
        const saveButton = byId('btn-save-settings');
        settingsSaving = true;
        setButtonBusy(saveButton, true);
        setStatus('Saving…', 'muted');
        // A pending restart LATCHES: a later save that needs no restart must
        // not hide the button while the process still runs the old config.
        if (!restartPending) byId('btn-restart-now')?.setAttribute('hidden', '');
        let saved = false;
        try {
            const data = await apiClient.saveSettings(body);
            if (data?.status !== 'saved') {
                const error = new Error(data?.error || 'The server did not confirm the settings save.');
                error.body = data;
                throw error;
            }
            saved = true;
            let runtimeModeResult = null;
            let runtimeModeError = '';
            let autoGrantResult = null;
            let autoGrantError = '';
            let contextModeResult = null;
            let contextModeError = '';
            let safetyModeResult = null;
            let safetyModeError = '';
            try {
                runtimeModeResult = await saveRuntimeModeViaNativeBridgeIfNeeded(ownerDraft.runtime);
            } catch (error) {
                const failure = settingsWriteFailure(error, "Runtime mode");
                runtimeModeError = failure.text;
                saveOutcomeUnknown ||= failure.unknown;
            }
            try {
                autoGrantResult = await saveAutoGrantViaNativeBridgeIfNeeded(ownerDraft.autoGrant);
            } catch (error) {
                const failure = settingsWriteFailure(error, "Reviewed-skill auto-grant");
                autoGrantError = failure.text;
                saveOutcomeUnknown ||= failure.unknown;
            }
            try {
                contextModeResult = await saveContextModeViaOwnerEndpointIfNeeded(ownerDraft.context);
            } catch (error) {
                const failure = settingsWriteFailure(error, "Context mode");
                contextModeError = failure.text;
                saveOutcomeUnknown ||= failure.unknown;
            }
            try {
                safetyModeResult = await saveSafetyModeViaOwnerEndpointIfNeeded(ownerDraft.safety);
            } catch (error) {
                const failure = settingsWriteFailure(error, "Safety mode");
                safetyModeError = failure.text;
                saveOutcomeUnknown ||= failure.unknown;
            }
            const ownerError = runtimeModeError || autoGrantError || contextModeError || safetyModeError;
            const draftKept = ownerError || sentRevision !== draftRevision || !(await loadSettings());
            syncAutoGrantBridgeState();
            let statusMsg;
            let statusType = 'ok';
            if (data.no_changes) {
                statusMsg = 'No changes detected';
            } else if (data.restart_required) {
                statusMsg = 'Settings saved. Some changes require a restart to take effect';
                statusType = 'warn';
            } else if (data.immediate_changed && data.next_task_changed) {
                statusMsg = 'Settings saved. Some changes took effect immediately; others apply on the next task';
            } else if (data.immediate_changed) {
                statusMsg = 'Settings saved. Changes took effect immediately';
            } else if (data.next_task_changed) {
                statusMsg = 'Settings saved. Changes take effect on the next task';
            } else {
                // Reachable when the only changed keys are retired no-ops:
                // the warning below carries the honest story.
                statusMsg = 'Settings saved';
            }
            if (subagentsChanged && data.agent_task_running) {
                statusMsg += '. Available subagents take effect for new child tasks; '
                    + 'the current task keeps its existing routes';
            }
            if (data.warnings && data.warnings.length) {
                statusMsg += ' ⚠️ ' + data.warnings.join(' | ');
                statusType = 'warn';
            }
            if (runtimeModeResult?.restart_required) {
                statusMsg = `${statusMsg} Runtime mode saved as ${runtimeModeResult.runtime_mode}; restart required.`;
                statusType = 'warn';
            }
            if (runtimeModeError) {
                statusMsg = `${statusMsg} ${runtimeModeError}`;
                statusType = 'warn';
            }
            if (autoGrantResult) {
                statusMsg = `${statusMsg} Reviewed-skill auto-grant ${autoGrantResult.enabled ? 'enabled' : 'disabled'}.`;
            }
            if (contextModeResult?.context_mode) {
                statusMsg = `${statusMsg} Context mode saved as ${contextModeResult.context_mode}.`;
            }
            if (contextModeError) {
                statusMsg = `${statusMsg} ${contextModeError}`;
                statusType = 'warn';
            }
            if (safetyModeResult?.safety_mode) {
                statusMsg = `${statusMsg} Safety supervisor saved as ${safetyModeResult.safety_mode}.`;
            }
            if (safetyModeError) {
                statusMsg = `${statusMsg} ${safetyModeError}`;
                statusType = 'warn';
            }
            if (autoGrantError) {
                statusMsg = `${statusMsg} ${autoGrantError}`;
                statusType = 'warn';
            }
            if (draftKept) {
                statusMsg += saveOutcomeUnknown ? '. Your draft is kept. Reload Settings to check before saving again.' : '. Your current draft is kept.';
                statusType = 'warn';
            }
            setStatus(statusMsg, statusType);
            if (data.restart_required || runtimeModeResult?.restart_required) {
                restartPending = true;
            }
            if (restartPending) byId('btn-restart-now')?.removeAttribute('hidden');
            window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { reason: 'settings saved', source: 'settings' } }));
        } catch (e) {
            const receipt = e?.body || e?.payload;
            const confirmedSaved = saved || receipt?.saved === true;
            saveOutcomeUnknown = !confirmedSaved && receipt?.saved !== false;
            setStatus(confirmedSaved
                ? `Settings were saved, but a later step failed: ${e.message}. Your draft is kept.`
                : saveOutcomeUnknown
                    ? `Save outcome unknown: ${e.message}. Your draft is kept. Reload Settings to check before saving again.`
                    : `Settings were not saved: ${e.message}. Your draft is kept.`, 'warn');
        } finally {
            settingsSaving = false;
            setButtonBusy(saveButton, false);
            syncSettingsLoadState();
        }
    });

    byId('btn-restart-now')?.addEventListener('click', async () => {
        const outcome = await confirmAndSendRestart({ openConfirmDialog, ws });
        if (outcome === 'sent') {
            restartPending = false;
            byId('btn-restart-now')?.setAttribute('hidden', '');
            setStatus('Restart requested. If the agent refuses, the reason appears in the main chat.', 'muted');
        } else if (outcome === 'not_connected') {
            setStatus('Not connected — the restart command was not sent.', 'warn');
        }
    });

    byId('btn-reset').addEventListener('click', async () => {
        const confirmedReset = await openConfirmDialog({
            title: 'Reset runtime data',
            body: 'This will delete all runtime data (state, memory, logs, settings) and restart.\nThe repo (agent code) will be preserved.\nYou will need to re-enter your provider settings.\n\nContinue?',
            confirmLabel: 'Delete and restart',
            danger: true,
        });
        if (!confirmedReset) return;
        try {
            const res = await apiFetch('/api/reset', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'ok') {
                await openConfirmDialog({
                    title: 'Reset complete',
                    body: 'Deleted: ' + (data.deleted.join(', ') || 'nothing') + '\nRestarting...',
                    alert: true,
                });
            } else {
                await openConfirmDialog({
                    title: 'Reset failed',
                    body: 'Error: ' + (data.error || 'unknown'),
                    alert: true,
                });
            }
        } catch (e) {
            showToast('Reset failed: ' + e.message, 'error');
        }
    });

    return {
        activateTab: activateSettingsTab,
        page,
    };
}
