import { refreshModelCatalog } from './settings_catalog.js';
import { bindEffortSegments, syncEffortSegments } from './settings_controls.js';
import { bindLocalModelControls } from './settings_local_model.js';
import { applyMcpSettings, collectMcpSettings, initMcpSettings } from './mcp_settings.js';
import { SECRET_KEYS, bindSecretInputs, bindSettingsTabs, renderSettingsPage } from './settings_ui.js';
import { showToast } from './toast.js';
import { escapeHtmlAttr as escapeHtml, formatDualVersion } from './utils.js';
import { apiClient, apiFetch, cleanExtensionRoute, extensionRoutePath } from './api_client.js';

let markSettingsDirty = () => {};
const BASE_SECRET_KEYS = new Set(SECRET_KEYS.map(([key]) => key));
let setupContract = {};

const INPUT_FIELDS = [
    ['s-openai-base-url', 'OPENAI_BASE_URL'], ['s-openai-compatible-base-url', 'OPENAI_COMPATIBLE_BASE_URL'], ['s-cloudru-base-url', 'CLOUDRU_FOUNDATION_MODELS_BASE_URL'],
    ['s-gigachat-scope', 'GIGACHAT_SCOPE'], ['s-gigachat-user', 'GIGACHAT_USER'], ['s-gigachat-base-url', 'GIGACHAT_BASE_URL'], ['s-gigachat-verify-ssl', 'GIGACHAT_VERIFY_SSL_CERTS'],
    ['s-server-host', 'OUROBOROS_SERVER_HOST', '127.0.0.1'], ['s-claude-code-model', 'CLAUDE_CODE_MODEL', 'opus[1m]'],
    ['s-review-models', 'OUROBOROS_REVIEW_MODELS'], ['s-scope-review-models', 'OUROBOROS_SCOPE_REVIEW_MODELS'], ['s-skills-repo-path', 'OUROBOROS_SKILLS_REPO_PATH'],
    ['s-clawhub-registry-url', 'OUROBOROS_CLAWHUB_REGISTRY_URL'], ['s-websearch-model', 'OUROBOROS_WEBSEARCH_MODEL'], ['s-gh-repo', 'GITHUB_REPO'],
    ['s-local-source', 'LOCAL_MODEL_SOURCE'], ['s-local-filename', 'LOCAL_MODEL_FILENAME'], ['s-local-chat-format', 'LOCAL_MODEL_CHAT_FORMAT'],
];
const VALUE_FIELDS = [
    ['s-effort-task', 'OUROBOROS_EFFORT_TASK', 'medium'], ['s-effort-evolution', 'OUROBOROS_EFFORT_EVOLUTION', 'high'], ['s-effort-review', 'OUROBOROS_EFFORT_REVIEW', 'medium'],
    ['s-effort-consciousness', 'OUROBOROS_EFFORT_CONSCIOUSNESS', 'high'], ['s-effort-scope-review', 'OUROBOROS_EFFORT_SCOPE_REVIEW', 'high'],
    ['s-review-enforcement', 'OUROBOROS_REVIEW_ENFORCEMENT', 'advisory'], ['s-task-review-mode', 'OUROBOROS_TASK_REVIEW_MODE', 'auto'], ['s-runtime-mode', 'OUROBOROS_RUNTIME_MODE', 'advanced'],
];
const NUMBER_FIELDS = [
    ['s-workers', 'OUROBOROS_MAX_WORKERS', 5], ['s-soft-timeout', 'OUROBOROS_SOFT_TIMEOUT_SEC', 600], ['s-hard-timeout', 'OUROBOROS_HARD_TIMEOUT_SEC', 1800],
    ['s-tool-timeout', 'OUROBOROS_TOOL_TIMEOUT_SEC', 600], ['s-local-port', 'LOCAL_MODEL_PORT', 8766], ['s-local-gpu-layers', 'LOCAL_MODEL_N_GPU_LAYERS', -1, true],
    ['s-local-ctx', 'LOCAL_MODEL_CONTEXT_LENGTH', 16384],
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
    const fieldHtml = (field) => {
        const name = escapeHtml(field.name || '');
        const label = escapeHtml(field.label || field.name || '');
        const placeholder = escapeHtml(field.placeholder || '');
        const type = String(field.type || 'text');
        if (type === 'textarea') {
            return `<label class="form-field"><span>${label}</span><textarea name="${name}" placeholder="${placeholder}"></textarea></label>`;
        }
        if (type === 'checkbox') {
            return `<label class="settings-extension-checkbox"><input type="checkbox" name="${name}"><span>${label}</span></label>`;
        }
        return `<label class="form-field"><span>${label}</span><input name="${name}" type="${escapeHtml(type)}" placeholder="${placeholder}"></label>`;
    };
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
            return `
                <form class="settings-extension-form" data-extension-settings-form data-skill="${escapeHtml(section.skill || '')}" data-route="${escapeHtml(rawRoute)}">
                    <div class="form-grid two">${fields.map(fieldHtml).join('')}</div>
                    <button class="btn btn-primary btn-sm" type="submit">${escapeHtml(component.submit_label || component.label || 'Save')}</button>
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
            if (!skill || !route) return;
            const values = {};
            new FormData(form).forEach((value, key) => { values[key] = value; });
            form.querySelectorAll('input[type="checkbox"]').forEach((input) => {
                values[input.name] = input.checked;
            });
            if (status) {
                status.textContent = 'Saving...';
                status.dataset.tone = 'muted';
            }
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
                if (status) {
                    status.textContent = data.message || 'Saved.';
                    status.dataset.tone = 'ok';
                }
            } catch (err) {
                if (status) {
                    status.textContent = err.message || String(err);
                    status.dataset.tone = 'danger';
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
    'google/gemini-3.5-flash',
    'anthropic/claude-sonnet-4.6',
    'anthropic/claude-opus-4.8',
    'anthropic::claude-opus-4-8',
    'anthropic/claude-opus-4.7',
    'anthropic::claude-opus-4-7',
    'anthropic::claude-opus-4-6',
    'anthropic::claude-sonnet-4-6',
    'openai::gpt-5.5',
    'openai::gpt-5.5-mini',
    'openai/gpt-5.5',
    'anthropic/claude-opus-4.6',
];

let settingsModelCatalogItems = SETTINGS_FALLBACK_MODELS.map((value) => ({ value, label: 'Suggested model' }));

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

    function snapshotSettingsDraft() {
        return JSON.stringify({
            ...collectBody(),
            OUROBOROS_RUNTIME_MODE_DRAFT: byId('s-runtime-mode')?.value || 'advanced',
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
            applyCheckboxValue(slot.settingsToggleId, s[`USE_LOCAL_${slot.slot.toUpperCase()}`]);
        });
        applyCheckboxValue('s-auto-grant-reviewed-skills', s.OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS);
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
        resetSecretClearFlags(page);
        syncEffortSegments(page);
        syncRuntimeModeBridgeState();
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
        const body = {
            OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS: byId('s-auto-grant-reviewed-skills')?.checked ? 'true' : 'false',
            ...collectMcpSettings(),
        };
        setupModelSlots().forEach((slot) => {
            body[slot.settingKey] = fieldValue(slot.settingsInputId);
            body[`USE_LOCAL_${slot.slot.toUpperCase()}`] = Boolean(byId(slot.settingsToggleId)?.checked);
        });
        INPUT_FIELDS.forEach(([id, key, fallback = '']) => {
            const value = fieldValue(id).trim();
            body[key] = key === 'OUROBOROS_SERVER_HOST' ? value || fallback : value || (key === 'CLAUDE_CODE_MODEL' ? fallback : '');
        });
        VALUE_FIELDS
            .filter(([, key]) => key !== 'OUROBOROS_RUNTIME_MODE')
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
        const result = bridge
            ? await bridge(nextMode)
            : (confirm(`Change Ouroboros runtime mode from ${currentMode} to ${nextMode}? The change takes effect after restart.`)
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
        const result = bridge
            ? await bridge(nextEnabled)
            : (confirm(`${nextEnabled ? 'Enable' : 'Disable'} reviewed-skill auto-grant? It only applies after a fresh executable review for the current content hash.`)
                ? await apiClient.ownerAutoGrant(nextEnabled)
                : { ok: false, error: 'Reviewed-skill auto-grant change cancelled.' });
        if (!result || result.ok !== true) {
            throw new Error(result?.error || 'Reviewed-skill auto-grant change was cancelled.');
        }
        return result;
    }

    syncSettingsLoadState();
    syncRuntimeModeBridgeState();
    syncAutoGrantBridgeState();
    reloadSettingsWithFeedback();

    if (typeof setBeforePageLeave === 'function') {
        setBeforePageLeave(({ from }) => {
            if (from !== 'settings' || !settingsDirty) return true;
            const leave = confirm('You have unsaved settings changes. Discard them and leave Settings?');
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
            queueMicrotask(updateSettingsDirtyState);
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
        const body = collectBody();

        try {
            const data = await apiClient.saveSettings(body);
            let runtimeModeResult = null;
            let runtimeModeError = '';
            let autoGrantResult = null;
            let autoGrantError = '';
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
            if (autoGrantError) {
                statusMsg = `${statusMsg} Reviewed-skill auto-grant was not changed: ${autoGrantError}`;
                statusType = 'warn';
            }
            setStatus(statusMsg, statusType);
            window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { reason: 'settings saved', source: 'settings' } }));
        } catch (e) {
            setStatus('Failed to save: ' + e.message, 'warn');
        }
    });

    byId('btn-reset').addEventListener('click', async () => {
        if (!confirm('This will delete all runtime data (state, memory, logs, settings) and restart.\nThe repo (agent code) will be preserved.\nYou will need to re-enter your provider settings.\n\nContinue?')) return;
        try {
            const res = await apiFetch('/api/reset', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'ok') alert('Deleted: ' + (data.deleted.join(', ') || 'nothing') + '\nRestarting...');
            else alert('Error: ' + (data.error || 'unknown'));
        } catch (e) {
            showToast('Reset failed: ' + e.message, 'error');
        }
    });

    return {
        activateTab: activateSettingsTab,
        page,
    };
}
