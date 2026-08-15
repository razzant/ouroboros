import { refreshModelCatalog } from './settings_catalog.js';
import { bindEffortSegments, syncEffortSegments } from './settings_controls.js';
import { bindLocalModelControls } from './settings_local_model.js';
import { applyMcpSettings, collectMcpSettings, initMcpSettings } from './mcp_settings.js';
import { collectReviewerSlots, initReviewerSlots, reloadReviewerSlots } from './reviewer_slots.js';
import {
    applySubagentsSettings,
    collectSubagentsSettings,
    initSubagentsSection,
    reloadSubagentsSection,
} from './subagents_settings.js';
import { initHarnessAccounts } from './harness_accounts.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { SECRET_KEYS, bindSecretInputs, bindSettingsTabs, renderSettingsPage } from './settings_ui.js';
import { showToast } from './toast.js';
import { escapeHtmlAttr as escapeHtml, formatDualVersion } from './utils.js';
import { apiClient, apiFetch, cleanExtensionRoute, extensionRoutePath } from './api_client.js';
import { initConnectionsUI } from './connections_ui.js';
import { collectSafeFieldValues, renderSafeField, setInlineStatus } from './ui_helpers.js';

let markSettingsDirty = () => {};
const BASE_SECRET_KEYS = new Set(SECRET_KEYS.map(([key]) => key));
const pendingExtensionSettings = new Set();
let setupContract = {};

const INPUT_FIELDS = [
    ['s-openai-base-url', 'OPENAI_BASE_URL'], ['s-openai-compatible-base-url', 'OPENAI_COMPATIBLE_BASE_URL'], ['s-cloudru-base-url', 'CLOUDRU_FOUNDATION_MODELS_BASE_URL'],
    ['s-gigachat-scope', 'GIGACHAT_SCOPE'], ['s-gigachat-user', 'GIGACHAT_USER'], ['s-gigachat-base-url', 'GIGACHAT_BASE_URL'], ['s-gigachat-verify-ssl', 'GIGACHAT_VERIFY_SSL_CERTS'],
    // 6.4 (D10): the Claude Code model picker is gone with claude_code_edit;
    // CLAUDE_CODE_MODEL stays a backend setting for the api-route advisory.
    ['s-minimax-region', 'MINIMAX_REGION'],
    ['s-server-host', 'OUROBOROS_SERVER_HOST', '127.0.0.1'],
    // 6.1: OUROBOROS_REVIEW_MODELS / OUROBOROS_SCOPE_REVIEW_MODELS are no
    // longer authored here — the Review lanes section composes the ONE
    // structured setting; the comma keys stay a backend-derived projection.
    ['s-deep-self-review-model', 'OUROBOROS_MODEL_DEEP_SELF_REVIEW'], ['s-skills-repo-path', 'OUROBOROS_SKILLS_REPO_PATH'],
    ['s-clawhub-registry-url', 'OUROBOROS_CLAWHUB_REGISTRY_URL'], ['s-websearch-model', 'OUROBOROS_WEBSEARCH_MODEL'], ['s-gh-repo', 'GITHUB_REPO'],
    ['s-local-source', 'LOCAL_MODEL_SOURCE'], ['s-local-filename', 'LOCAL_MODEL_FILENAME'], ['s-local-chat-format', 'LOCAL_MODEL_CHAT_FORMAT'],
    ['s-subagent-worktree-root', 'OUROBOROS_SUBAGENT_WORKTREE_ROOT'], ['s-subagent-projects-root', 'OUROBOROS_SUBAGENT_PROJECTS_ROOT'],
    ['s-evo-budget', 'OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD', '0'],
    ['s-evo-objective', 'OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE', ''],
];
const VALUE_FIELDS = [
    // 6.3: Review / Scope Review efforts are per-slot rows in Agents → Review
    // lanes now; their global keys remain backend defaults, no longer UI-authored.
    ['s-effort-task', 'OUROBOROS_EFFORT_TASK', 'medium'], ['s-effort-evolution', 'OUROBOROS_EFFORT_EVOLUTION', 'high'],
    ['s-effort-consciousness', 'OUROBOROS_EFFORT_CONSCIOUSNESS', 'high'], ['s-effort-deep-self-review', 'OUROBOROS_EFFORT_DEEP_SELF_REVIEW', 'high'],
    ['s-review-enforcement', 'OUROBOROS_REVIEW_ENFORCEMENT', 'advisory'], ['s-task-review-mode', 'OUROBOROS_TASK_REVIEW_MODE', 'auto'], ['s-runtime-mode', 'OUROBOROS_RUNTIME_MODE', 'advanced'],
    ['s-update-channel', 'OUROBOROS_UPDATE_CHANNEL', 'stable'],
    ['s-context-mode', 'OUROBOROS_CONTEXT_MODE', 'max'], ['s-image-input-mode', 'OUROBOROS_IMAGE_INPUT_MODE', 'auto'],
    ['s-safety-mode', 'OUROBOROS_SAFETY_MODE', 'full'],
    ['s-prompt-cache-ttl', 'OUROBOROS_PROMPT_CACHE_TTL', '1h'],
];
const _SAFETY_MODE_RANK = { full: 2, light: 1, off: 0 };
const NUMBER_FIELDS = [
    ['s-workers', 'OUROBOROS_MAX_WORKERS', 10], ['s-active-subagents', 'OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT', 6], ['s-subagent-depth', 'OUROBOROS_MAX_SUBAGENT_DEPTH', 2, true],
    ['s-tool-timeout', 'OUROBOROS_TOOL_TIMEOUT_SEC', 600], ['s-local-port', 'LOCAL_MODEL_PORT', 8766], ['s-local-gpu-layers', 'LOCAL_MODEL_N_GPU_LAYERS', -1, true],
    ['s-local-ctx', 'LOCAL_MODEL_CONTEXT_LENGTH', 16384], ['s-gc-retention-days', 'OUROBOROS_GC_RETENTION_DAYS', 7],
    ['s-bg-wakeup-min', 'OUROBOROS_BG_WAKEUP_MIN', 30], ['s-bg-wakeup-max', 'OUROBOROS_BG_WAKEUP_MAX', 7200], ['s-bg-max-rounds', 'OUROBOROS_BG_MAX_ROUNDS', 10],
];

function setupModelSlots() {
    return Array.isArray(setupContract.modelSlots) ? setupContract.modelSlots : [];
}

function byId(id) {
    return document.getElementById(id);
}

function applyInputValue(id, value) {
    byId(id).value = value === undefined || value === null ? '' : value;
}

function applyCheckboxValue(id, value) {
    byId(id).checked = isTruthySetting(value);
}

function syncHeavyModelPlaceholder() {
    // Owner decision: an empty Heavy slot legally inherits Main, so the empty
    // field names the model it actually resolves to instead of looking unset.
    // Display-only — nothing here writes settings.
    const heavy = byId('s-model-heavy');
    const main = byId('s-model');
    if (!heavy || !main) return;
    const mainValue = String(main.value || '').trim();
    heavy.placeholder = mainValue ? `inherits Main (${mainValue})` : 'inherits Main';
}

function isTruthySetting(value) {
    const normalized = String(value ?? '').trim().toLowerCase();
    return value === true || ['true', '1', 'yes', 'on'].includes(normalized);
}

function setStatus(text, tone = 'ok') {
    const status = byId('settings-status');
    status.textContent = text;
    status.dataset.tone = tone;
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


function wireSecretRow(row) {
    const input = row.querySelector('.secret-input');
    const toggle = row.querySelector('[data-row-secret-toggle]');
    const clear = row.querySelector('[data-row-secret-clear]');
    if (input) input.addEventListener('input', () => { if (input.value.trim()) delete input.dataset.forceClear; });
    if (toggle && input) toggle.addEventListener('click', () => { input.type = input.type === 'password' ? 'text' : 'password'; toggle.textContent = input.type === 'password' ? 'Show' : 'Hide'; });
    if (clear && input) clear.addEventListener('click', () => { input.value = ''; input.type = 'password'; input.dataset.forceClear = '1'; if (toggle) toggle.textContent = 'Show'; markSettingsDirty(); });
}

function customSecretRow(key = '', value = '') {
    const id = `custom-secret-${Math.random().toString(36).slice(2)}`;
    const row = document.createElement('div');
    row.className = 'settings-custom-secret-row';
    row.dataset.customSecretRow = '1';
    row.innerHTML = `
        <div class="form-field settings-custom-secret-key"><label>Key</label><input data-custom-secret-key value="${escapeHtml(key)}" placeholder="SLACK_WEBHOOK_URL" spellcheck="false"></div>
        <div class="form-field settings-custom-secret-value"><label>Value</label><div class="secret-input-row">
            <input id="${id}" data-custom-secret-value class="secret-input" type="password" value="${escapeHtml(value || '')}" placeholder="Secret value">
            <button type="button" class="settings-ghost-btn" data-row-secret-toggle>Show</button>
            <button type="button" class="settings-ghost-btn" data-row-secret-clear>Clear</button>
        </div><div class="settings-inline-note" data-custom-secret-error hidden></div></div>
        <button type="button" class="settings-ghost-btn settings-custom-secret-remove" data-custom-secret-remove>Remove</button>`;
    wireSecretRow(row);
    row.querySelector('[data-custom-secret-remove]')?.addEventListener('click', () => { row.dataset.removeCustomSecret = '1'; row.hidden = true; markSettingsDirty(); });
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
        el.innerHTML = `<div class="form-field"><label>${escapeHtml(key)}</label><div class="secret-input-row">
            <input id="${id}" data-secret-setting="${escapeHtml(key)}" class="secret-input" type="password" value="${escapeHtml(settings[key] || '')}" placeholder="Secret value">
            <button type="button" class="settings-ghost-btn" data-row-secret-toggle>Show</button>
            <button type="button" class="settings-ghost-btn" data-row-secret-clear>Clear</button>
        </div></div>`;
        wireSecretRow(el); host.appendChild(el);
    });
}

function renderExtensionSettingsSections(root, sections) {
    const host = root.querySelector('#extension-settings-sections');
    if (!host) return;
    const items = Array.isArray(sections) ? sections : [];
    if (!items.length) {
        host.innerHTML = '<div class="muted">No extension settings registered.</div>';
        return;
    }
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
            const formKey = `${section.key || `${section.skill}:${section.section_id}`}:${component.id || idx}`;
            formSpecs.set(formKey, component);
            const disabled = Boolean(component.disabled);
            const fieldOptions = {
                disabled,
                fieldClass: 'form-field',
                inlineClass: 'settings-extension-checkbox',
                helpClass: 'settings-inline-note',
            };
            return `
                <form class="settings-extension-form" data-extension-settings-form data-extension-settings-key="${escapeHtml(formKey)}" data-skill="${escapeHtml(section.skill || '')}" data-route="${escapeHtml(rawRoute)}">
                    <div class="form-grid two">${fields.map((field) => renderSafeField(field, {}, fieldOptions)).join('')}</div>
                    <button class="btn btn-primary btn-sm" type="submit"${disabled ? ' disabled' : ''}>${escapeHtml(component.submit_label || component.label || 'Save')}</button>
                    <div class="settings-inline-status" data-extension-settings-status></div>
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
            if (!skill || !route || spec.disabled || pendingExtensionSettings.has(requestKey)) return;
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
                setInlineStatus(status, data.message || 'Saved.', 'ok');
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
    if (value && !value.includes('...')) body[settingKey] = value;
}

// Fallback picker pills mirror config defaults plus useful direct-provider ids.
const SETTINGS_FALLBACK_MODELS = [
    'x-ai/grok-4.5',
    'google/gemini-3.6-flash',
    'openai/gpt-5.6-terra',
    'openai/gpt-5.6-sol',
    'openai/gpt-5.6-luna',
    'openai::gpt-5.6-terra',
    'openai::gpt-5.6-sol',
    'openai::gpt-5.6-luna',
    'anthropic/claude-sonnet-5',
    'anthropic/claude-opus-5',
    'anthropic::claude-sonnet-5',
    'anthropic::claude-opus-5',
    'anthropic::claude-opus-4-6',
    'deepseek/deepseek-v4-pro',
    'minimax::MiniMax-M3',
    'minimax::MiniMax-M2.7',
];

let settingsModelCatalogItems = SETTINGS_FALLBACK_MODELS.map((value) => ({ value, label: 'Suggested model' }));

/**
 * Pure predicate (v6.82 P2): should the collapsed Settings "More providers"
 * section auto-open? True only for a USABLE credential — a provider API key,
 * a GigaChat OAuth credential, or a COMPLETE GigaChat basic-auth pair. Base
 * URLs/scope/TLS fields always carry shipped defaults and never count.
 * Exported for dependency-free node tests.
 */
export function moreProvidersCredentialConfigured({
    cloudruKey = '', minimaxKey = '', gigachatCredentials = '', gigachatUser = '', gigachatPassword = '',
} = {}) {
    const has = (v) => Boolean(String(v ?? '').trim());
    return has(cloudruKey)
        || has(minimaxKey)
        || has(gigachatCredentials)
        || (has(gigachatUser) && has(gigachatPassword));
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
    bindSettingsTabs(page, { state });
    bindSecretInputs(page);
    bindEffortSegments(page);
    bindLocalModelControls({ state });
    initConnectionsUI({ root: page, apiClient, ws });
    // Best-effort About version from /api/health.
    apiFetch('/api/health')
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then((d) => {
            const verEl = document.getElementById('about-version');
            if (verEl) verEl.textContent = formatDualVersion(d);
        })
        .catch(() => { /* about version is best-effort */ });
    let currentSettings = {};
    let claudeCodePollStarted = false;
    let extensionRefreshPending = false;
    // Runtime errors must surface even before ANTHROPIC_API_KEY is configured.
    let claudeRuntimeHasError = false;
    let settingsLoaded = false;
    let settingsBaseline = '';
    let settingsDirty = false;
    initMcpSettings({ onChange: updateSettingsDirtyState });
    initReviewerSlots({ onChange: () => updateSettingsDirtyState() });
    initSubagentsSection({ onChange: () => updateSettingsDirtyState() });
    initHarnessAccounts();

    function anthropicKeyConfigured() {
        const input = byId('s-anthropic');
        if (!input) return Boolean(String(currentSettings.ANTHROPIC_API_KEY || '').trim());
        if (input.dataset.forceClear === '1') return false;
        const liveValue = String(input.value || '').trim();
        if (liveValue) return true;
        return Boolean(String(currentSettings.ANTHROPIC_API_KEY || '').trim());
    }

    function shouldShowClaudeRuntimeCard() {
        return anthropicKeyConfigured() || claudeRuntimeHasError;
    }

    function renderClaudeCodeUi() {
        const panel = byId('settings-claude-code-panel');
        const note = byId('settings-claude-code-copy');
        const button = byId('btn-claude-code-install');
        const visible = shouldShowClaudeRuntimeCard();
        if (panel) panel.hidden = !visible;
        if (note) note.hidden = !visible;
        if (!visible) return;
        if (button && button.dataset.busy !== '1' && button.dataset.ready !== '1') {
            button.disabled = false;
            button.textContent = 'Repair Runtime';
        }
    }

    function syncSettingsLoadState() {
        const saveBtn = byId('btn-save-settings');
        if (saveBtn) {
            saveBtn.disabled = !settingsLoaded;
            saveBtn.title = settingsLoaded
                ? ''
                : 'Reload current settings successfully before saving.';
        }
    }

    function syncRuntimeModeBridgeState() {
        const hasBridge = Boolean(window.pywebview?.api?.request_runtime_mode_change);
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

    function snapshotSettingsDraft() {
        return JSON.stringify({
            ...collectBody(),
            OUROBOROS_RUNTIME_MODE_DRAFT: byId('s-runtime-mode')?.value || 'advanced',
            OUROBOROS_CONTEXT_MODE_DRAFT: byId('s-context-mode')?.value || 'max',
        });
    }

    function setSettingsCleanBaseline() {
        settingsBaseline = snapshotSettingsDraft();
        settingsDirty = false;
        const indicator = byId('settings-unsaved-indicator');
        if (indicator) indicator.classList.remove('is-visible');
    }

    function updateSettingsDirtyState() {
        if (!settingsLoaded || !settingsBaseline) return;
        const nextDirty = snapshotSettingsDraft() !== settingsBaseline;
        if (nextDirty === settingsDirty) return;
        settingsDirty = nextDirty;
        const indicator = byId('settings-unsaved-indicator');
        if (indicator) indicator.classList.toggle('is-visible', settingsDirty);
    }

    function discardUnsavedSettingsDraft() {
        closeSettingsModelPickers();
        applySettings(currentSettings || {});
        setSettingsCleanBaseline();
        setStatus('', 'ok');
    }

    function applyClaudeCodeStatus(payload = {}) {
        const button = byId('btn-claude-code-install');
        const status = byId('settings-claude-code-status');
        const ready = Boolean(payload.ready);
        const installed = Boolean(payload.installed);
        const busy = Boolean(payload.busy);
        const error = String(payload.error || '').trim();
        // Backend error state controls visibility without an API key.
        claudeRuntimeHasError = Boolean(error);
        const message = String(payload.message || '').trim()
            || (ready ? 'Claude runtime ready.' : (installed ? 'Claude runtime available but not ready.' : 'Claude runtime not available.'));
        const tone = ready ? 'ok' : (error ? 'error' : (installed ? 'muted' : 'error'));
        if (status) {
            status.textContent = message;
            status.dataset.tone = tone;
        }
        if (button) {
            button.dataset.busy = busy ? '1' : '0';
            button.dataset.ready = ready ? '1' : '0';
            button.dataset.installed = installed ? '1' : '0';
            button.disabled = busy;
            button.textContent = busy ? 'Repairing...' : (ready ? 'Runtime OK' : 'Repair Runtime');
        }
        renderClaudeCodeUi();
    }

    async function refreshClaudeCodeStatus() {
        // Poll even without API key; backend separates no_api_key from errors.
        try {
            const resp = await apiFetch('/api/claude-code/status', { cache: 'no-store' });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
            applyClaudeCodeStatus(data);
        } catch (error) {
            applyClaudeCodeStatus({
                installed: false,
                ready: false,
                busy: false,
                error: String(error?.message || error || ''),
                message: `Claude runtime status check failed: ${String(error?.message || error || '')}`,
            });
        }
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

    function startClaudeCodePolling() {
        if (claudeCodePollStarted) return;
        claudeCodePollStarted = true;
        refreshClaudeCodeStatus();
        setInterval(() => {
            refreshClaudeCodeStatus();
        }, 3000);
    }

    function applySettings(s) {
        setupContract = s?._meta?.setup_contract || setupContract || {};
        applySecretInputs(page, s);
        INPUT_FIELDS.forEach(([id, key, fallback = '']) => applyInputValue(id, fallback && !s[key] ? fallback : s[key]));
        VALUE_FIELDS.forEach(([id, key, fallback]) => { byId(id).value = s[key] || fallback; });
        setupModelSlots().forEach((slot) => {
            applyInputValue(slot.settingsInputId, s[slot.settingKey]);
            if (slot.settingsToggleId) applyCheckboxValue(slot.settingsToggleId, s[`USE_LOCAL_${slot.slot.toUpperCase()}`]);
        });
        syncHeavyModelPlaceholder();
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
        // The delegation route lives next to it in Agents → Delegation.
        applySubagentsSettings(s);
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
        const [data, extData] = await Promise.all([
            apiClient.settings(),
            apiClient.extensions().catch(() => ({})),
        ]);
        const sections = Array.isArray(extData?.live?.settings_sections)
            ? extData.live.settings_sections
            : [];
        currentSettings = data;
        applySettings(data);
        renderExtensionSettingsSections(page, sections);
        renderRequestedSkillSecrets(page, extData.skills || [], data);
        renderCustomSecrets(page, data);
        // Await the reviewer rows and the Subagents accounts BEFORE the clean
        // baseline: their async arrival must not read as an unsaved owner edit.
        await Promise.all([reloadReviewerSlots(), reloadSubagentsSection()]);
        setSettingsCleanBaseline();
        closeSettingsModelPickers();
        _renderNetworkHint(data._meta);
        renderClaudeCodeUi();
        settingsLoaded = true;
        markSettingsDirty = updateSettingsDirtyState;
        syncSettingsLoadState();
        startClaudeCodePolling();
    }

    async function reloadSettingsWithFeedback() {
        setStatus('Loading settings...', 'muted');
        settingsLoaded = false;
        syncSettingsLoadState();
        try {
            await loadSettings();
            try {
                await refreshModelCatalog();
                setStatus('Settings loaded', 'ok');
            } catch (error) {
                setStatus(
                    `Settings loaded. Model catalog refresh failed: ${error.message || error}`,
                    'warn'
                );
            }
        } catch (error) {
            settingsLoaded = false;
            syncSettingsLoadState();
            setStatus(
                `Failed to load current settings. Save is disabled until reload succeeds: ${error.message || error}`,
                'warn'
            );
        }
    }

    async function refreshSettingsAfterExtensionChange(reason = 'skills changed') {
        if (extensionRefreshPending) return;
        if (settingsDirty) {
            setStatus(`Settings changed externally (${reason}). Reload after saving or discarding your draft.`, 'warn');
            return;
        }
        extensionRefreshPending = true;
        try {
            await loadSettings();
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
            // Same rule for the delegated-subagent route: {} until the accounts
            // read succeeded, so an unrelated save cannot turn delegation off
            // because this page could not reach the daemon.
            ...collectSubagentsSettings(),
        };
        setupModelSlots().forEach((slot) => {
            body[slot.settingKey] = fieldValue(slot.settingsInputId);
            if (slot.settingsToggleId) body[`USE_LOCAL_${slot.slot.toUpperCase()}`] = Boolean(byId(slot.settingsToggleId)?.checked);
        });
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
        page.querySelectorAll('[data-custom-secret-row]').forEach((row) => {
            const keyInput = row.querySelector('[data-custom-secret-key]');
            const valueInput = row.querySelector('[data-custom-secret-value]');
            const key = (keyInput?.value || '').trim().toUpperCase();
            const error = row.querySelector('[data-custom-secret-error]');
            if (!key) return;
            if (!/^[A-Z][A-Z0-9_]{2,}$/.test(key)) { if (error) { error.hidden = false; error.textContent = 'Use uppercase letters, numbers, and underscores.'; } return; }
            if (row.dataset.removeCustomSecret === '1' || valueInput?.dataset.forceClear === '1') { body[key] = ''; return; }
            const value = valueInput?.value || '';
            if (value && !value.includes('...')) body[key] = value;
        });

        return body;
    }

    async function saveRuntimeModeViaNativeBridgeIfNeeded() {
        const nextMode = byId('s-runtime-mode').value || 'advanced';
        const currentMode = currentSettings?.OUROBOROS_RUNTIME_MODE || 'advanced';
        const bridge = window.pywebview?.api?.request_runtime_mode_change;
        if (nextMode === currentMode) {
            return bridge ? await bridge(nextMode) : await apiClient.ownerRuntimeMode(nextMode);
        }
        // Only the browser-side confirm is migrated to the in-house dialog; the
        // desktop pywebview bridge path above stays exactly as it was.
        const result = bridge
            ? await bridge(nextMode)
            : ((await openConfirmDialog({
                title: 'Change runtime mode',
                body: `Change Ouroboros runtime mode from ${currentMode} to ${nextMode}? The change takes effect after restart.`,
                confirmLabel: 'Change mode',
            }))
                ? await apiClient.ownerRuntimeMode(nextMode)
                : { ok: false, error: 'Runtime mode change cancelled.' });
        if (!result || result.ok !== true) {
            throw new Error(result?.error || 'Runtime mode change was cancelled.');
        }
        return result;
    }

    async function saveAutoGrantViaNativeBridgeIfNeeded() {
        const checkbox = byId('s-auto-grant-reviewed-skills');
        if (!checkbox) return null;
        const nextEnabled = Boolean(checkbox.checked);
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
                : { ok: false, error: 'Reviewed-skill auto-grant change cancelled.' });
        if (!result || result.ok !== true) {
            throw new Error(result?.error || 'Reviewed-skill auto-grant change was cancelled.');
        }
        return result;
    }

    async function saveSafetyModeViaOwnerEndpointIfNeeded() {
        // Owner-only, dropped from the generic /api/settings POST — saved through the
        // dedicated audited endpoint. Confirm on LOWERING coverage (full > light > off).
        const input = byId('s-safety-mode');
        if (!input) return null;
        const next = input.value || 'full';
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
            if (!ok) throw new Error('Safety mode change was not confirmed.');
        }
        const result = await apiClient.ownerSafetyMode(next);
        if (!result || result.ok !== true) {
            throw new Error(result?.error || 'Safety mode change failed.');
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

    // A pinned scope reviewer's route has no other reachable path to Capability
    // Evidence: the settings save probes it and returns the SAME needs_ack contract the
    // Max gate uses, so reuse that flow verbatim. Without rendering it the owner only
    // ever sees commits blocked by SCOPE_REVIEW_SUB_FLOOR telling them to owner-ack a
    // route the UI never offered. Declining leaves the slot fail-closed, as before.
    async function ackReviewCapabilityNotices(notices) {
        const pending = (Array.isArray(notices) ? notices : [])
            .filter((notice) => notice?.needs_ack?.model);
        let acked = 0;
        for (const notice of pending) {
            const ack = notice.needs_ack;
            const seen = Number(notice.window_tokens || 0);
            // Each delivery is judged by ITS OWN floor: the api row by the
            // constitutional 1M, a RETRIEVING row by the 200K session floor. Asking
            // about 1M for a retrieving row would demand a confirmation its own gate
            // never wanted, so the floor rides with the notice.
            const floor = Number(notice.floor_tokens || 0) || 1000000;
            const floorText = floor.toLocaleString('en-US');
            // A STALE record can report a full 1M and still not authorize, so say WHY
            // the ack is being asked for — otherwise the prompt reads "this route
            // reports 1000000 tokens, please confirm 1000000 tokens".
            const reading = !(seen > 0)
                ? 'no window metadata'
                : (notice?.needs_ack?.evidence?.stale
                    ? `${seen} tokens from an EXPIRED reading the provider could not re-confirm`
                    : `${seen} tokens`);
            const confirmed = await openConfirmDialog({
                title: 'Confirm scope-reviewer context window',
                body: `Scope review is fail-closed unless its reviewer's ${floorText}-token context `
                    + `window is currently known, and this route reports ${reading}.\n\n`
                    + `Confirm that this reviewer supports a ${floorText}-token context window?\n`
                    + `provider: ${ack.provider || '(default)'}\nmodel: ${ack.model}\n`
                    + `base_url: ${ack.base_url || '(default)'}\n\n`
                    + 'This applies only to this exact model/provider. Cancelling leaves scope '
                    + 'review blocking commits on this route.',
                confirmLabel: 'Confirm window',
            });
            if (!confirmed) continue;
            await apiClient.ownerCapabilityAck({
                provider: ack.provider, model: ack.model, base_url: ack.base_url,
                window_tokens: floor, note: 'owner-confirmed scope reviewer window',
            });
            acked += 1;
        }
        return acked;
    }

    async function saveContextModeViaOwnerEndpointIfNeeded() {
        const input = byId('s-context-mode');
        if (!input) return null;
        const next = input.value || 'max';
        const current = currentSettings?.OUROBOROS_CONTEXT_MODE || 'max';
        // A displayed `low` that is a system AUTO-DOWNGRADE is not an owner selection
        // (see the chat toggle): re-picking Low must still POST so the idempotent
        // endpoint clears the derived flag, otherwise an install whose route cannot be
        // confirmed >=1M has no reachable way to declare Low and every commit blocks.
        const derivedLow = String(currentSettings?.OUROBOROS_CONTEXT_MODE_AUTO_LOW || '')
            .trim().toLowerCase() === 'true';
        if (next === current && !(next === 'low' && derivedLow)) return null;
        // Owner-only + hot-apply (saves immediately, no restart). Max needs the active model's
        // 1M-token window confirmed; on a 409 needs_ack, share the chat-toggle's ack flow
        // (CW8) — confirm, POST the route-scoped capability-ack, retry — instead of a
        // generic failure.
        try {
            const result = await apiClient.ownerContextMode(next);
            if (!result || result.ok !== true) {
                throw new Error(result?.error || 'Context mode change failed.');
            }
            return result;
        } catch (e) {
            const ack = (e && e.status === 409 && e.body && e.body.needs_ack) ? e.body.needs_ack : null;
            if (!(next === 'max' && ack && ack.model)) {
                throw e;
            }
            const confirmed = await openConfirmDialog({
                title: 'Confirm 1M-token context window',
                body: `${(e.body && e.body.error) || 'Max context mode needs a confirmed 1M-token window.'}\n\n` +
                    `Confirm that this model supports a 1,000,000-token context window?\n` +
                    `provider: ${ack.provider || '(default)'}\nmodel: ${ack.model}\n` +
                    `base_url: ${ack.base_url || '(default)'}\n\n` +
                    `This applies only to this exact model/provider and is removed if you change it.`,
                confirmLabel: 'Confirm window',
            });
            if (!confirmed) {
                throw new Error('Max context mode was not confirmed.');
            }
            // Throws on a non-ok ack (surfaced by the save handler's catch).
            await apiClient.ownerCapabilityAck({
                provider: ack.provider, model: ack.model, base_url: ack.base_url,
                window_tokens: 1000000, note: 'owner-confirmed via settings save',
            });
            const retry = await apiClient.ownerContextMode(next);
            if (!retry || retry.ok !== true) {
                throw new Error(retry?.error || 'Context mode change failed after confirmation.');
            }
            return retry;
        }
    }

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
            if (from !== 'settings' || !settingsDirty) return true;
            const leave = await openConfirmDialog({
                title: 'Unsaved settings',
                body: 'You have unsaved settings changes. Discard them and leave Settings?',
                confirmLabel: 'Discard and leave',
                cancelLabel: 'Stay',
            });
            if (leave) discardUnsavedSettingsDraft();
            return leave;
        });
    }

    byId('s-anthropic')?.addEventListener('input', () => {
        renderClaudeCodeUi();
        if (anthropicKeyConfigured()) {
            startClaudeCodePolling();
            refreshClaudeCodeStatus();
        }
    });

    page.addEventListener('input', updateSettingsDirtyState);
    page.addEventListener('change', updateSettingsDirtyState);
    page.addEventListener('click', (event) => {
        if (event.target.closest('[data-effort-value], .secret-clear, [data-row-secret-clear], [data-custom-secret-remove]')) {
            queueMicrotask(() => {
                syncPostTaskEvolutionUi();
                updateSettingsDirtyState();
            });
        }
    });
    byId('btn-add-custom-secret')?.addEventListener('click', () => {
        const host = byId('custom-secrets-list');
        if (!host) return;
        if (host.querySelector('.muted')) host.innerHTML = '';
        const row = customSecretRow();
        host.appendChild(row);
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        row.querySelector('[data-custom-secret-key]')?.focus();
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

    function closeSettingsModelPickers(exceptPicker = null) {
        page.querySelectorAll('[data-model-picker]').forEach((picker) => {
            if (picker === exceptPicker) return;
            const panel = picker.querySelector('.model-picker-results');
            if (!panel) return;
            panel.hidden = true;
            panel.innerHTML = '';
        });
    }

    function renderSettingsModelPicker(input) {
        const picker = input.closest('[data-model-picker]');
        const panel = picker?.querySelector('.model-picker-results');
        if (!picker || !panel) return;
        const needle = String(input.value || '').trim().toLowerCase();
        let items = settingsModelCatalogItems
            .filter((item) => {
                const haystack = `${item.value} ${item.label || ''} ${item.provider || ''}`.toLowerCase();
                return !needle || haystack.includes(needle);
            })
            .slice(0, 8);
        if (!items.length && needle) {
            items = settingsModelCatalogItems.slice(0, 8);
        }
        if (!items.length) {
            panel.hidden = true;
            panel.innerHTML = '';
            return;
        }
        panel.innerHTML = items.map((item) => `
            <button type="button" class="model-picker-item" data-value="${escapeHtml(item.value)}">
                <span class="model-picker-item-value">${escapeHtml(item.value)}</span>
                <span class="model-picker-item-label">${escapeHtml(item.label || item.provider || 'Catalog model')}</span>
            </button>
        `).join('');
        panel.hidden = false;
    }

    page.addEventListener('focusin', (event) => {
        const input = event.target instanceof Element
            ? event.target.closest('[data-model-picker] input')
            : null;
        if (!input) return;
        const picker = input.closest('[data-model-picker]');
        closeSettingsModelPickers(picker);
        renderSettingsModelPicker(input);
    });
    page.dataset.modelPickerBound = '1';

    page.addEventListener('input', (event) => {
        const input = event.target instanceof Element
            ? event.target.closest('[data-model-picker] input')
            : null;
        if (!input) return;
        const picker = input.closest('[data-model-picker]');
        closeSettingsModelPickers(picker);
        renderSettingsModelPicker(input);
    });

    // Keep the Heavy inherits-Main hint live while Main is edited (typed
    // 'input') or picked from the catalog dropdown (dispatched 'change').
    for (const eventType of ['input', 'change']) {
        page.addEventListener(eventType, (event) => {
            if (event.target instanceof Element && event.target.id === 's-model') {
                syncHeavyModelPlaceholder();
            }
        });
    }

    page.addEventListener('mousedown', (event) => {
        const item = event.target instanceof Element
            ? event.target.closest('.model-picker-item')
            : null;
        if (item) {
            const picker = item.closest('[data-model-picker]');
            const input = picker?.querySelector('input');
            if (input) {
                event.preventDefault();
                input.value = item.dataset.value || '';
                closeSettingsModelPickers();
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
            return;
        }
        if (!(event.target instanceof Element) || !event.target.closest('[data-model-picker]')) {
            closeSettingsModelPickers();
        }
    });

    document.addEventListener('settings-model-catalog:updated', (event) => {
        const items = Array.isArray(event.detail?.items) ? event.detail.items : [];
        settingsModelCatalogItems = items.length
            ? items.map((item) => ({
                value: item.value || item.id || '',
                label: item.label || item.provider || 'Catalog model',
                provider: item.provider || '',
            })).filter((item) => item.value)
            : SETTINGS_FALLBACK_MODELS.map((value) => ({ value, label: 'Suggested model' }));
        page.querySelectorAll('[data-model-picker]').forEach((picker) => {
            const panel = picker.querySelector('.model-picker-results');
            if (panel && !panel.hidden) {
                const input = picker.querySelector('input');
                renderSettingsModelPicker(input);
            }
        });
    });

    page.addEventListener('click', (event) => {
        if (event.target.closest('.secret-clear[data-target="s-anthropic"]')) {
            queueMicrotask(() => {
                renderClaudeCodeUi();
                refreshClaudeCodeStatus();
            });
        }
    });

    byId('btn-claude-code-install')?.addEventListener('click', async () => {
        applyClaudeCodeStatus({
            installed: false,
            ready: false,
            busy: true,
            message: 'Repairing Claude runtime...',
            error: '',
        });
        try {
            const resp = await apiFetch('/api/claude-code/install', { method: 'POST' });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
            applyClaudeCodeStatus(data);
            setStatus(data.repaired ? 'Claude runtime repaired' : 'Claude runtime up to date', 'ok');
        } catch (error) {
            const message = String(error?.message || error || '');
            applyClaudeCodeStatus({
                installed: false,
                ready: false,
                busy: false,
                error: message,
                message: `Claude runtime repair failed: ${message}`,
            });
            setStatus('Claude runtime repair failed', 'warn');
        }
    });

    byId('btn-refresh-model-catalog').addEventListener('click', async () => {
        await refreshModelCatalog();
    });

    byId('btn-reload-settings')?.addEventListener('click', async () => {
        await reloadSettingsWithFeedback();
    });

    byId('btn-save-settings').addEventListener('click', async () => {
        if (!settingsLoaded) {
            setStatus('Reload current settings successfully before saving.', 'warn');
            return;
        }
        // Validate Every-N cadence before save: malformed N must NOT silently coerce
        // into a valid (e.g. every-task) cadence. Abort with a visible error instead.
        if (byId('s-post-task-evolution-mode')?.value === 'every_n'
            && !/^[1-9]\d*$/.test((byId('s-evo-cadence-n')?.value || '').trim())) {
            setStatus('Every-N cadence needs a whole number ≥ 1.', 'warn');
            return;
        }
        const body = collectBody();

        try {
            const data = await apiClient.saveSettings(body);
            let runtimeModeResult = null;
            let runtimeModeError = '';
            let autoGrantResult = null;
            let autoGrantError = '';
            let contextModeResult = null;
            let contextModeError = '';
            let safetyModeResult = null;
            let safetyModeError = '';
            try {
                runtimeModeResult = await saveRuntimeModeViaNativeBridgeIfNeeded();
            } catch (error) {
                runtimeModeError = error.message || String(error);
            }
            try {
                autoGrantResult = await saveAutoGrantViaNativeBridgeIfNeeded();
            } catch (error) {
                autoGrantError = error.message || String(error);
            }
            try {
                contextModeResult = await saveContextModeViaOwnerEndpointIfNeeded();
            } catch (error) {
                contextModeError = error.message || String(error);
            }
            try {
                safetyModeResult = await saveSafetyModeViaOwnerEndpointIfNeeded();
            } catch (error) {
                safetyModeError = error.message || String(error);
            }
            let reviewAcks = 0;
            let reviewAckError = '';
            try {
                reviewAcks = await ackReviewCapabilityNotices(data.review_capability_notices);
            } catch (error) {
                reviewAckError = error.message || String(error);
            }
            await loadSettings();
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
            } else {
                statusMsg = 'Settings saved. Changes take effect on the next task';
            }
            if (data.warnings && data.warnings.length) {
                statusMsg += ' ⚠️ ' + data.warnings.join(' | ');
                statusType = 'warn';
            }
            if (data.context_mode_downgraded) {
                // The new model can't sustain Max, so context mode auto-dropped to Low.
                statusMsg = `${statusMsg} ${data.notice || 'Context mode switched to Low.'}`;
                statusType = 'warn';
            }
            if (runtimeModeResult?.restart_required) {
                statusMsg = `${statusMsg} Runtime mode saved as ${runtimeModeResult.runtime_mode}; restart required.`;
                statusType = 'warn';
            }
            if (runtimeModeError) {
                statusMsg = `${statusMsg} Runtime mode was not changed: ${runtimeModeError}`;
                statusType = 'warn';
            }
            if (autoGrantResult) {
                statusMsg = `${statusMsg} Reviewed-skill auto-grant ${autoGrantResult.enabled ? 'enabled' : 'disabled'}.`;
            }
            if (contextModeResult?.context_mode) {
                statusMsg = `${statusMsg} Context mode saved as ${contextModeResult.context_mode}.`;
            }
            if (contextModeError) {
                statusMsg = `${statusMsg} Context mode was not changed: ${contextModeError}`;
                statusType = 'warn';
            }
            if (safetyModeResult?.safety_mode) {
                statusMsg = `${statusMsg} Safety supervisor saved as ${safetyModeResult.safety_mode}.`;
            }
            if (safetyModeError) {
                statusMsg = `${statusMsg} Safety mode was not changed: ${safetyModeError}`;
                statusType = 'warn';
            }
            if (autoGrantError) {
                statusMsg = `${statusMsg} Reviewed-skill auto-grant was not changed: ${autoGrantError}`;
                statusType = 'warn';
            }
            if (reviewAcks > 0) {
                statusMsg = `${statusMsg} Confirmed the required context window for ${reviewAcks} scope-review route(s).`;
            }
            if (reviewAckError) {
                statusMsg = `${statusMsg} The scope-reviewer window confirmation was not saved: ${reviewAckError}`;
                statusType = 'warn';
            }
            setStatus(statusMsg, statusType);
            window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { reason: 'settings saved', source: 'settings' } }));
        } catch (e) {
            setStatus('Failed to save: ' + e.message, 'warn');
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
