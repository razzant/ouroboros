import './api_types.js';

/**
 * Single browser-side gateway client. Keep backend calls here so UI modules
 * depend on named boundary helpers rather than raw transport details.
 */
export async function apiFetch(url, init = {}) {
    return fetch(url, init);
}

export async function fetchJson(url, init = {}, options = {}) {
    const response = await apiFetch(url, init);
    let data = null;
    try {
        data = await response.json();
    } catch {
        data = { error: `non-json response (HTTP ${response.status})` };
    }
    if (!response.ok || (options.rejectOkFalse && data && data.ok === false)) {
        const message = (data && (data.error || data.message)) || `HTTP ${response.status}`;
        const error = new Error(message);
        error.status = response.status;
        error.body = data;
        error.payload = data;
        throw error;
    }
    return data;
}

export function jsonPost(url, payload = {}, options = {}) {
    return fetchJson(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    }, options);
}

export function cleanExtensionRoute(value) {
    const route = String(value || '').trim().replace(/^\/+/, '');
    const parts = route.split('/').filter(Boolean);
    if (!route || route.includes('\\') || parts.some((part) => part === '.' || part === '..')) {
        return '';
    }
    return parts.map(encodeURIComponent).join('/');
}

export function extensionRoutePrefix(skill) {
    return `/api/extensions/${encodeURIComponent(skill)}/`;
}

export function extensionRoutePath(skill, route, params = null) {
    const cleanRoute = cleanExtensionRoute(route);
    if (!cleanRoute) return '';
    const query = params instanceof URLSearchParams && String(params) ? `?${params}` : '';
    return `${extensionRoutePrefix(skill)}${cleanRoute}${query}`;
}

export const apiClient = {
    /** @returns {Promise<import('./api_types.js').HealthResponse>} */
    health: () => fetchJson('/api/health', { cache: 'no-store' }),
    /** @returns {Promise<import('./api_types.js').StateResponse>} */
    state: () => fetchJson('/api/state', { cache: 'no-store' }),
    settings: () => fetchJson('/api/settings', { cache: 'no-store' }),
    /** @returns {Promise<import('./api_types.js').UiPreferencesResponse>} */
    uiPreferences: () => fetchJson('/api/ui/preferences', { cache: 'no-store' }),
    saveUiPreferences: (payload) => jsonPost('/api/ui/preferences', payload),
    saveSettings: (payload) => fetchJson('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    }),
    ownerRuntimeMode: (mode) => jsonPost('/api/owner/runtime-mode', { mode }),
    ownerAutoGrant: (enabled) => jsonPost('/api/owner/auto-grant', { enabled: Boolean(enabled) }),
    ownerContextMode: (mode) => jsonPost('/api/owner/context-mode', { mode }),
    /** @returns {Promise<import('./api_types.js').OwnerScopeReviewFloorResponse>} */
    ownerScopeReviewFloor: (floor) => jsonPost('/api/owner/scope-review-floor', { floor }),
    ownerCapabilityAck: (payload) => jsonPost('/api/owner/capability-ack', payload),
    /** @returns {Promise<import('./api_types.js').OpenAICompatibleModelsResponse>} */
    openAICompatibleModels: (payload) => jsonPost('/api/openai-compatible/models', payload),
    extensions: () => fetchJson('/api/extensions', { cache: 'no-store' }),
    skillLifecycleQueue: () => fetchJson('/api/skills/lifecycle-queue', { cache: 'no-store' }),
    /** @returns {Promise<import('./api_types.js').SkillDeleteResponse>} */
    deleteSkill: (skill, payloadRoot) => jsonPost(`/api/skills/${encodeURIComponent(skill)}/delete`, {
        payload_root: payloadRoot,
    }),
    skillGrants: (skill, items) => jsonPost(`/api/skills/${encodeURIComponent(skill)}/grants`, { items }),
    chatHistory: (limit = 1000) => fetchJson(`/api/chat/history?limit=${encodeURIComponent(limit)}`, { cache: 'no-store' }),
    projectFromTask: (taskId, id, name, objectiveHint = '') => jsonPost('/api/projects/from-task', { task_id: taskId, id, name, objective_hint: objectiveHint }),
    updateStatus: () => fetchJson('/api/update/status', { cache: 'no-store' }),
    updatePreflight: () => jsonPost('/api/update/preflight', {}),
    updateApply: (strategy) => jsonPost('/api/update/apply', { strategy }),
};
