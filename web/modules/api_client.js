import './api_types.js';

/**
 * Single browser-side gateway client. Keep backend calls here so UI modules
 * depend on named boundary helpers rather than raw transport details.
 */
export async function apiFetch(url, init = {}) {
    return fetch(url, init);
}

/**
 * Read an owner-visible sentence out of a gateway error body. Several routes
 * answer with a STRUCTURED error (`{error: {code, message, ...}}`), and
 * interpolating that object straight into an Error message produced the
 * literal `[object Object]` in a toast — the owner saw an alarm carrying no
 * fact at all. Prefer the object's own sentence, then its code, and serialize
 * only as a last resort so nothing is silently swallowed.
 */
function errorText(data) {
    for (const value of [data?.error, data?.message]) {
        if (typeof value === 'string' && value.trim()) return value;
        if (value && typeof value === 'object') {
            const inner = value.message || value.detail || value.code;
            if (typeof inner === 'string' && inner.trim()) return inner;
            try { return JSON.stringify(value); } catch { /* fall through */ }
        }
    }
    return '';
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
        const message = errorText(data) || `HTTP ${response.status}`;
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
 * Run the read-only publication preflight for one selected skill. Domain
 * states, including repairable findings, remain successful JSON responses;
 * only transport/admission refusals reject through fetchJson.
 * @param {string} skill
 * @returns {Promise<import('./api_types.js').SkillPublishPreflightResponse>}
 */
export function skillPublishPreflight(skill) {
    return jsonPost(`/api/skills/${encodeURIComponent(skill)}/publish-preflight`, {});
}

/**
 * Create one ordinary managed task through the shared task gateway.
 * @param {import('./api_types.js').TaskCreateRequest} payload
 * @returns {Promise<import('./api_types.js').TaskCreateResponse>}
 */
export function createTask(payload) {
    return jsonPost('/api/tasks', payload);
}

/**
 * Cancel a task. With {cascade:true} the server also cancels the task's live
 * subtree and answers only once that teardown has finished; without it the
 * request stays the synchronous single-task cancel (no body — headless compat).
 * S3 (Q1/Q2): stopPolicy "finalize_then_cancel" requests the soft
 * finalize-then-stop episode (202 acknowledgement, cancel_state "pending");
 * "immediate" (or absent) keeps today's hard cancel byte-identical.
 * Shared by the Chat live-card stop control and the Activity tab.
 * @param {string} taskId
 * @param {{cascade?: boolean, stopPolicy?: string}} [options]
 * @returns {Promise<import('./api_types.js').TaskCancelResponse>}
 */
export function cancelTask(taskId, { cascade = false, stopPolicy = '' } = {}) {
    const url = `/api/tasks/${encodeURIComponent(taskId)}/cancel`;
    const policy = String(stopPolicy || '');
    const body = {
        ...(cascade ? { cascade: true } : {}),
        ...(policy && policy !== 'immediate' ? { stop_policy: policy } : {}),
    };
    return Object.keys(body).length ? jsonPost(url, body) : fetchJson(url, { method: 'POST' });
}

/** URL for one published immutable source handle. */
export function taskSourceDownloadUrl(taskId, ref) {
    const path = typeof ref?.path === 'string' ? ref.path : '';
    if (!taskId || ref?.root !== 'artifact_store' || ref?.kind !== 'task_source'
        || !/^[0-9a-f]{64}$/.test(ref?.sha256 || '')
        || !/^source_handles\/(tool_results|context_checkpoints)\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(path)
        || !Number.isSafeInteger(ref?.size) || ref.size < 0) return '';
    const name = path.split('/').at(-1);
    return `/api/tasks/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(name)}?source=${encodeURIComponent(path)}`;
}

export async function resumeTask(taskId) {
    return fetchJson(`/api/tasks/${encodeURIComponent(taskId)}/resume`, { method: 'POST' });
}

/**
 * Owner hurry (S3, HQ1): the text-free typed task-local acceleration control.
 * The body carries ONLY the client-generated stable request_id (reuse the same
 * id on retry — the acknowledgement is idempotent). This path never creates a
 * chat message anywhere; the durable facts are the typed owner-mailbox control
 * and the owner_hurry task-result projection.
 * @param {string} taskId
 * @param {string} requestId
 * @returns {Promise<import('./api_types.js').TaskHurryResponse>}
 */
export function hurryTask(taskId, requestId) {
    return jsonPost(
        `/api/tasks/${encodeURIComponent(taskId)}/hurry`,
        { request_id: String(requestId || '') },
        { rejectOkFalse: true },
    );
}

/**
 * Fetch one task's durable detail record, or null when unreachable — the
 * shared reconcile read used by the cancel/stop card flows.
 * @param {string} taskId
 * @returns {Promise<import('./api_types.js').TaskDetailResponse|null>}
 */
export async function fetchTaskDetail(taskId) {
    const resp = await apiFetch(`/api/tasks/${encodeURIComponent(taskId)}`);
    return (resp && typeof resp.json === 'function' && resp.ok !== false) ? resp.json() : null;
}

/**
 * Strict task-detail read for consumers that must tell a genuinely absent
 * record (404 → null) apart from a failed read (rejects). The lenient
 * fetchTaskDetail above keeps its every-failure→null contract for reconcile
 * flows that treat all misses alike.
 * @param {string} taskId
 * @returns {Promise<import('./api_types.js').TaskDetailResponse|null>}
 */
export async function fetchTaskDetailStrict(taskId) {
    const resp = await apiFetch(`/api/tasks/${encodeURIComponent(taskId)}`);
    if (resp && typeof resp.json === 'function') {
        if (resp.ok !== false) return resp.json();
        if (resp.status === 404) return null;
    }
    throw new Error(`task detail read failed (HTTP ${resp?.status ?? 'no response'})`);
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
    /**
     * @param {import('./api_types.js').OnboardingSubagentsPreviewRequest} payload
     * @returns {Promise<import('./api_types.js').OnboardingSubagentsPreviewResponse>}
     */
    previewOnboardingSubagents: (payload) => jsonPost('/api/onboarding/subagents/preview', payload),
    ownerRuntimeMode: (mode) => jsonPost('/api/owner/runtime-mode', { mode }),
    ownerAutoGrant: (enabled) => jsonPost('/api/owner/auto-grant', { enabled: Boolean(enabled) }),
    ownerContextMode: (mode) => jsonPost('/api/owner/context-mode', { mode }),
    /** @returns {Promise<import('./api_types.js').OwnerSafetyModeResponse>} */
    ownerSafetyMode: (mode) => jsonPost('/api/owner/safety-mode', { mode }),
    logsTail: (name, limit = 2000) => fetchJson(`/api/logs/${encodeURIComponent(name)}?limit=${encodeURIComponent(limit)}`, { cache: 'no-store' }),
    ownerCapabilityAck: (payload) => jsonPost('/api/owner/capability-ack', payload),
    /** @returns {Promise<import('./api_types.js').OpenAICompatibleModelsResponse>} */
    openAICompatibleModels: (payload) => jsonPost('/api/openai-compatible/models', payload),
    /**
     * @param {import('./api_types.js').ProviderTestRequest} payload
     * @returns {Promise<import('./api_types.js').ProviderTestResponse>}
     */
    providerTest: (payload) => jsonPost('/api/providers/test', payload),
    extensions: () => fetchJson('/api/extensions', { cache: 'no-store' }),
    /**
     * Widgets page cards: live extension UI tabs projected from the loader
     * snapshot (no skill discovery), each stamped with the owning skill's
     * payload `revision`.
     * @returns {Promise<import('./api_types.js').WidgetsResponse>}
     */
    widgets: () => fetchJson('/api/widgets', { cache: 'no-store' }),
    skillPublishPreflight,
    createTask,
    skillLifecycleQueue: () => fetchJson('/api/skills/lifecycle-queue', { cache: 'no-store' }),
    /** @returns {Promise<import('./api_types.js').SkillDeleteResponse>} */
    deleteSkill: (skill, payloadRoot) => jsonPost(`/api/skills/${encodeURIComponent(skill)}/delete`, {
        payload_root: payloadRoot,
    }),
    skillGrants: (skill, items) => jsonPost(`/api/skills/${encodeURIComponent(skill)}/grants`, { items }),
    /**
     * @param {string} skill
     * @param {import('./api_types.js').OwnerSkillPresenceRuntimeRequest} payload
     * @returns {Promise<import('./api_types.js').OwnerSkillPresenceRuntimeResponse>}
     */
    savePresenceRuntime: (skill, payload) => jsonPost(
        `/api/owner/skills/${encodeURIComponent(skill)}/presence-runtime`,
        payload,
    ),
    projectFromTask: (taskId, id, name, objectiveHint = '') => jsonPost('/api/projects/from-task', { task_id: taskId, id, name, objective_hint: objectiveHint }),
    /** @param {import('./api_types.js').ProjectCreateRequest} payload */
    projectCreate: (payload) => jsonPost('/api/projects', payload),
    projectUpdate: (projectId, name) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/update`, { name }),
    /** @returns {Promise<import('./api_types.js').ProjectDeleteResponse>} */
    projectDelete: (projectId) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/delete`, {}),
    /** @returns {Promise<import('./api_types.js').FsDirsResponse>} */
    fsDirs: (path = '') => fetchJson(`/api/fs/dirs${path ? `?path=${encodeURIComponent(path)}` : ''}`, { cache: 'no-store' }),
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
