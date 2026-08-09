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

/**
 * Cancel a task. With {cascade:true} the server also cancels the task's live
 * subtree and answers only once that teardown has finished; without it the
 * request stays the synchronous single-task cancel (no body — headless compat).
 * Shared by the Chat live-card "Cancel run" action and the Activity tab.
 * @param {string} taskId
 * @param {{cascade?: boolean}} [options]
 * @returns {Promise<import('./api_types.js').TaskCancelResponse>}
 */
export function cancelTask(taskId, { cascade = false } = {}) {
    const url = `/api/tasks/${encodeURIComponent(taskId)}/cancel`;
    return cascade ? jsonPost(url, { cascade: true }) : fetchJson(url, { method: 'POST' });
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

export function updateStrategyForPlan(plan = {}) {
    if (!plan.available) return '';
    const kind = String(plan.kind || '');
    if (!['clean', 'conflicting'].includes(kind)) return '';
    return kind === 'clean' ? 'auto_merge' : 'assisted';
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
    // DEPRECATED (v6.80.0): the value is stored but nothing consults it — BIBLE P3
    // scope-review applicability follows the owner context mode. Kept as a frozen
    // contract surface; the response carries an explicit deprecation notice.
    ownerScopeReviewFloor: (floor) => jsonPost('/api/owner/scope-review-floor', { floor }),
    /** @returns {Promise<import('./api_types.js').OwnerSafetyModeResponse>} */
    ownerSafetyMode: (mode) => jsonPost('/api/owner/safety-mode', { mode }),
    logsTail: (name, limit = 2000) => fetchJson(`/api/logs/${encodeURIComponent(name)}?limit=${encodeURIComponent(limit)}`, { cache: 'no-store' }),
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
    /** @param {import('./api_types.js').ProjectCreateRequest} payload */
    projectCreate: (payload) => jsonPost('/api/projects', payload),
    projectUpdate: (projectId, name) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/update`, { name }),
    /** @returns {Promise<import('./api_types.js').ProjectDeleteResponse>} */
    projectDelete: (projectId) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/delete`, {}),
    /** @returns {Promise<import('./api_types.js').FsDirsResponse>} */
    fsDirs: (path = '') => fetchJson(`/api/fs/dirs${path ? `?path=${encodeURIComponent(path)}` : ''}`, { cache: 'no-store' }),
    /**
     * Recent task results (newest first, server-sorted by ts).
     * @returns {Promise<import('./api_types.js').TaskListResponse>}
     */
    tasks: (limit = 50) => fetchJson(`/api/tasks?limit=${encodeURIComponent(limit)}`, { cache: 'no-store' }),
    /** One task's durable/effective result record. */
    task: (taskId) => fetchJson(`/api/tasks/${encodeURIComponent(taskId)}`, { cache: 'no-store' }),
    /**
     * One task's diff. The response carries the RAW patch: the client derives the
     * file list, per-file status and +/- counts from those same bytes.
     * @param {string} taskId
     * @returns {Promise<import('./api_types.js').TaskDiffResponse>}
     */
    taskDiff: (taskId) => fetchJson(`/api/tasks/${encodeURIComponent(taskId)}/diff`, { cache: 'no-store' }),
    updateStatus: () => fetchJson('/api/update/status', { cache: 'no-store' }),
    updateCheck: () => jsonPost('/api/update/check', {}),
    /** @returns {Promise<import('./api_types.js').UpdatePreflightResponse>} */
    updatePreflight: () => jsonPost('/api/update/preflight', {}),
    /** @returns {Promise<import('./api_types.js').UpdateApplySuccessResponse>} */
    updateApply: (strategy, plan = {}, { confirmRecovery = false } = {}) => jsonPost('/api/update/apply', {
        strategy,
        expected_base_sha: String(plan.base_sha || ''),
        expected_target_sha: String(plan.target_sha || ''),
        ...(confirmRecovery ? { confirm_recovery: true } : {}),
    }),
};
