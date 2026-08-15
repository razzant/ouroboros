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
    /**
     * Finish first-run onboarding in ONE atomic owner-scoped save (D-8):
     * settings + runtime mode + safety default + the completion fact land
     * together, or nothing does. The install-time agent preset and its marker
     * ride the same write only when the response says `preset.applied` —
     * `not_requested`, `skipped_by_owner` and `not_install_time` are ordinary
     * successes that persist no preset.
     * @param {import('./api_types.js').OnboardingCompleteRequest} payload
     * @returns {Promise<import('./api_types.js').OnboardingCompleteResponse>}
     */
    completeOnboarding: (payload) => jsonPost('/api/onboarding/complete', payload),
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
    projectFromTask: (taskId, id, name, objectiveHint = '') => jsonPost('/api/projects/from-task', { task_id: taskId, id, name, objective_hint: objectiveHint }),
    /** @param {import('./api_types.js').ProjectCreateRequest} payload */
    projectCreate: (payload) => jsonPost('/api/projects', payload),
    // A bare string stays the rename shorthand; an object is forwarded verbatim so
    // additive registry fields need no new client method. The server's allowed-field
    // set remains the authority and answers a typed 400 for anything it does not take.
    // A remote REBIND rides this same door as {connection_id, remote_root}: a second
    // named method would be a second client-side spelling of one endpoint.
    /** @param {string|import('./api_types.js').ProjectUpdateRequest} update */
    projectUpdate: (projectId, update) => jsonPost(
        `/api/projects/${encodeURIComponent(projectId)}/update`,
        typeof update === 'string' ? { name: update } : (update || {}),
    ),
    /** @returns {Promise<import('./api_types.js').ProjectDeleteResponse>} */
    projectDelete: (projectId) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/delete`, {}),
    /** @returns {Promise<import('./api_types.js').FsDirsResponse>} */
    fsDirs: (path = '') => fetchJson(`/api/fs/dirs${path ? `?path=${encodeURIComponent(path)}` : ''}`, { cache: 'no-store' }),
    // Owner-only connections surface (RWS v2, D6). Every route below is behind the
    // owner session gate (ouroboros/server_auth.py); a build without the ssh
    // transport answers a typed 503 `remote_transport_unavailable`.
    /** @returns {Promise<import('./api_types.js').ConnectionListResponse>} */
    connections: () => fetchJson('/api/owner/connections', { cache: 'no-store' }),
    /**
     * @param {import('./api_types.js').ConnectionAddRequest} payload
     * @returns {Promise<import('./api_types.js').ConnectionActionResponse>}
     */
    connectionAdd: (payload) => jsonPost('/api/owner/connections', payload),
    /** @returns {Promise<import('./api_types.js').ConnectionActionResponse>} */
    connectionTest: (connectionId) => jsonPost(`/api/owner/connections/${encodeURIComponent(connectionId)}/test`, {}),
    /** @returns {Promise<import('./api_types.js').ConnectionActionResponse>} */
    connectionBootstrap: (connectionId) => jsonPost(`/api/owner/connections/${encodeURIComponent(connectionId)}/bootstrap`, {}),
    /** @returns {Promise<import('./api_types.js').ConnectionActionResponse>} */
    connectionReconnect: (connectionId) => jsonPost(`/api/owner/connections/${encodeURIComponent(connectionId)}/reconnect`, {}),
    // Retrust demands confirm:true plus the exact old/new identity pair a live probe
    // observed; the gateway refuses a blind re-pin.
    /** @returns {Promise<import('./api_types.js').ConnectionActionResponse>} */
    connectionRetrust: (connectionId, payload) => jsonPost(`/api/owner/connections/${encodeURIComponent(connectionId)}/retrust`, payload),
    /** @returns {Promise<import('./api_types.js').ConnectionActionResponse>} */
    connectionRetire: (connectionId) => fetchJson(`/api/owner/connections/${encodeURIComponent(connectionId)}`, { method: 'DELETE' }),
    /** @returns {Promise<import('./api_types.js').ConnectionDirsResponse>} */
    connectionDirs: (connectionId, path = '') => fetchJson(
        `/api/owner/connections/${encodeURIComponent(connectionId)}/dirs${path ? `?path=${encodeURIComponent(path)}` : ''}`,
        { cache: 'no-store' },
    ),
    // Middleware route (ouroboros/server_auth.py), not a gateway endpoint: exchanges
    // the Network Password for an HttpOnly owner session cookie.
    ownerLogin: (password) => jsonPost('/auth/login', { password, next: '/' }),
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
