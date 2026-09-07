import { refreshModelCatalog } from './settings_catalog.js';
import { bindEffortSegments, syncEffortSegments } from './settings_controls.js';
import { bindLocalModelControls } from './settings_local_model.js';
import { applyMcpSettings, collectMcpSettings, initMcpSettings } from './mcp_settings.js';
import { adoptSubagentRoster, collectReviewerSlots, initReviewerSlots, reloadReviewerSlots } from './reviewer_slots.js';
import {
    applySubagentsSettings,
    availableSubagentsPreviewPayload,
    collectSubagentsSettings,
    initSubagentsSection,
    noteSubagentsSaveAttempt,
    reloadSubagentsSection,
    subagentSettingsFingerprint,
    validateSubagentsDraft,
} from './subagents_settings.js';
import { initHarnessAccounts } from './harness_accounts.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { PROVIDER_TEST_INPUTS, SECRET_KEYS, bindSecretInputs, bindSettingsTabs, renderSettingsPage } from './settings_ui.js';
import { showToast } from './toast.js';
import { escapeHtmlAttr as escapeHtml, formatDualVersion } from './utils.js';
import { apiClient, apiFetch, cleanExtensionRoute, extensionRoutePath } from './api_client.js';
import { claudexorStatus } from './claudexor_status_store.js';
import { collectSafeFieldValues, renderSafeField, setInlineStatus, revealNewRow } from './ui_helpers.js';
import { extensionActionStatus } from './extension_status_text.js';

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
    ['s-evo-objective', 'OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE', ''],
];
const VALUE_FIELDS = [
    // 6.3: Review / Scope Review efforts are per-slot rows in Agents → Review
    // lanes now; their global keys remain backend defaults, no longer UI-authored.
    ['s-effort-task', 'OUROBOROS_EFFORT_TASK', 'medium'], ['s-effort-evolution', 'OUROBOROS_EFFORT_EVOLUTION', 'high'],
    ['s-effort-consciousness', 'OUROBOROS_EFFORT_CONSCIOUSNESS', 'high'], ['s-effort-deep-self-review', 'OUROBOROS_EFFORT_DEEP_SELF_REVIEW', 'high'],
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
    ['s-bg-wakeup-min', 'OUROBOROS_BG_WAKEUP_MIN', 30], ['s-bg-wakeup-max', 'OUROBOROS_BG_WAKEUP_MAX', 7200], ['s-bg-max-rounds', 'OUROBOROS_BG_MAX_ROUNDS', 10],
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

// `owner` names the surface a message belongs to (today only the Available
// subagents roster claims one); a later message from anyone else drops it, so
// an owner may clear its own stale message but never a newer one.
function setStatus(text, tone = 'ok', owner = '') {
    const status = byId('settings-status');
    status.textContent = text;
    status.dataset.tone = tone;
    if (owner) status.dataset.owner = owner;
    else delete status.dataset.owner;
}

function setButtonBusy(button, busy) {
    if (!button) return;
    button.disabled = busy;
    if (busy) button.setAttribute('aria-busy', 'true');
    else button.removeAttribute('aria-busy');
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
    if (clear && input) clear.addEventListener('click', () => {
        input.value = ''; input.type = 'password'; input.dataset.forceClear = '1';
        if (toggle) toggle.textContent = 'Show';
        markSettingsDirty();
        // Programmatic value changes fire no 'input' event, but a Clear is an
        // edit like any other: the provider-test verdict listener must see it.
        input.dispatchEvent(new Event('input', { bubbles: true }));
    });
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
            <button type="button" class="btn btn-default" data-row-secret-toggle>Show</button>
            <button type="button" class="btn btn-default" data-row-secret-clear>Clear</button>
        </div><div class="settings-inline-note" data-custom-secret-error hidden></div></div>
        <button type="button" class="btn btn-default settings-custom-secret-remove" data-custom-secret-remove>Remove</button>`;
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
            <button type="button" class="btn btn-default" data-row-secret-toggle>Show</button>
            <button type="button" class="btn btn-default" data-row-secret-clear>Clear</button>
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
    if (value && !value.includes('...')) body[settingKey] = value;
}

// Fallback picker pills mirror config defaults plus useful direct-provider ids.
const SETTINGS_FALLBACK_MODELS = [
    'google/gemini-3.8-flash',
    'x-ai/grok-4.6',
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
    'deepseek::deepseek-v4-pro',
    'deepseek::deepseek-v4-flash',
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
    let extensionRefreshPending = false;
    let settingsLoaded = false;
    let settingsBaseline = '';
    let settingsDirty = false;
    const providerTestGenerations = new Map();
    const providerTestsInFlight = new Set();
    initMcpSettings({ onChange: updateSettingsDirtyState });
    initReviewerSlots({ onChange: () => updateSettingsDirtyState() });
    initSubagentsSection({
        onChange: () => updateSettingsDirtyState(),
        // The roster's section line and the footer message it owns read one
        // verdict: when the judged rows come clean, the footer clears with the
        // line and the tint — unless someone else has written the footer since.
        onJudged: (clean) => {
            if (clean && byId('settings-status').dataset.owner === 'subagents') setStatus('', 'ok');
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

    let baselineSettleDisposer = null;
    function armCleanBaselineOnStatusSettle() {
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
            // Disclosed residual: a cold-daemon settle landing AFTER an owner
            // edit stays inside the unsaved-changes diff until the next save —
            // rare (the reloads wait a bounded beat for the probe first) and
            // fail-safe (an over-eager indicator, never a lost edit).
            if (!settingsDirty && settingsLoaded) setSettingsCleanBaseline();
        });
    }

    function discardUnsavedSettingsDraft() {
        closeSettingsModelPickers();
        applySettings(currentSettings || {});
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
        setupModelSlots().forEach((slot) => {
            applyInputValue(slot.settingsInputId, s[slot.settingKey]);
            if (slot.settingsToggleId) applyCheckboxValue(slot.settingsToggleId, s[`USE_LOCAL_${slot.slot.toUpperCase()}`]);
        });
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
        // The Review-lanes «Configured subagent» selects reference the SAME
        // roster; adopt it from the same loaded document.
        adoptSubagentRoster(s);
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
        // Await reviewer config and the Available-subagents bounded status beat BEFORE the clean
        // baseline: their async arrival must not read as an unsaved owner edit.
        // (The Claudexor status probe inside them is fire-and-forget — a cold
        // daemon must not hold the Save button — so its LATER settlement is
        // re-baselined below.)
        await Promise.all([reloadReviewerSlots(), reloadSubagentsSection()]);
        // Mark the document loaded before taking the baseline. A generated
        // preview may settle in the microtask between these statements; its
        // clean-gated callback must be allowed to fold that exact draft into
        // the baseline rather than leave a false unsaved change behind.
        settingsLoaded = true;
        setSettingsCleanBaseline();
        armCleanBaselineOnStatusSettle();
        closeSettingsModelPickers();
        _renderNetworkHint(data._meta);
        markSettingsDirty = updateSettingsDirtyState;
        syncSettingsLoadState();
    }

    async function reloadSettingsWithFeedback() {
        setStatus('Loading settings...', 'muted');
        settingsLoaded = false;
        syncSettingsLoadState();
        try {
            await loadSettings();
            try {
                await refreshModelCatalog({ button: byId('btn-refresh-model-catalog') });
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
            // Saved config and live availability are independent: a loaded
            // actor list is collected even when status is down; only an
            // unloaded/unparseable editor omits the key on an unrelated save.
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
        if (next === current) return null;
        const result = await apiClient.ownerContextMode(next);
        if (!result || result.ok !== true) {
            throw new Error(result?.error || 'Context mode change failed.');
        }
        return result;
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
        if (!settingsLoaded) {
            setStatus('Reload current settings successfully before saving.', 'warn');
            return;
        }
        // The owner just tried to commit the draft — every Save click is one,
        // whichever validation aborts it below — so from here the roster shows
        // its own errors beside the rows they name, not only in this status.
        noteSubagentsSaveAttempt();
        // Validate Every-N cadence before save: malformed N must NOT silently coerce
        // into a valid (e.g. every-task) cadence. Abort with a visible error instead.
        if (byId('s-post-task-evolution-mode')?.value === 'every_n'
            && !/^[1-9]\d*$/.test((byId('s-evo-cadence-n')?.value || '').trim())) {
            setStatus('Every-N cadence needs a whole number ≥ 1.', 'warn');
            return;
        }
        const subagentErrors = validateSubagentsDraft();
        if (subagentErrors.length) {
            setStatus(`Available subagents: ${subagentErrors[0]}`, 'warn', 'subagents');
            return;
        }
        const body = collectBody();
        const subagentsChanged = subagentSettingsFingerprint(body.OUROBOROS_SUBAGENTS)
            !== subagentSettingsFingerprint(currentSettings?.OUROBOROS_SUBAGENTS);

        // Phase 1 of the save: the button goes busy and the status says so.
        // Capability probes on review-route changes make a save take seconds;
        // an idle "Save Settings" over that window reads as a dead click.
        const saveButton = byId('btn-save-settings');
        setButtonBusy(saveButton, true);
        setStatus('Saving…', 'muted');
        // A pending restart LATCHES: a later save that needs no restart must
        // not hide the button while the process still runs the old config.
        if (!restartPending) byId('btn-restart-now')?.setAttribute('hidden', '');
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
            if (data.restart_required || runtimeModeResult?.restart_required) {
                restartPending = true;
            }
            if (restartPending) byId('btn-restart-now')?.removeAttribute('hidden');
            window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { reason: 'settings saved', source: 'settings' } }));
        } catch (e) {
            setStatus('Failed to save: ' + e.message, 'warn');
        } finally {
            setButtonBusy(saveButton, false);
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
