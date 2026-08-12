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
    let unparseable = false;
    try {
        data = await response.json();
    } catch {
        data = { error: `non-json response (HTTP ${response.status})` };
        // A 2xx whose body cannot be parsed is NOT an answer, and returning it as
        // one made `{error: …}` look like a payload: `threadOps.bases` handed that
        // object back, `listed.ok` was false, and branch-off rendered an EMPTY base
        // offer and asked the owner to type one (I16). A body we could not read is
        // a transport failure whatever the status line says.
        unparseable = true;
    }
    if (unparseable || !response.ok || (options.rejectOkFalse && data && data.ok === false)) {
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

/**
 * The one place a thread's route prefix is spelled. Six T3 routes hang off it,
 * and hand-building the path per call is how one of them ends up missing an
 * `encodeURIComponent` on a project id the owner typed.
 */
export function threadPath(projectId, threadId) {
    return `/api/projects/${encodeURIComponent(projectId)}/threads/${encodeURIComponent(threadId)}`;
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
    /**
     * The projects list, optionally INCLUDING archived threads.
     *
     * `/api/state` carries the sidebar's own copy, where archived threads are
     * hidden. This is the one call that can ask for them, and it is the only way
     * `threadRestore` is reachable at all: a thread no surface can show is a
     * thread no owner can restore (T3R-8).
     */
    projectsList: (includeArchived = false) => fetchJson(
        includeArchived ? '/api/projects?include_archived=1' : '/api/projects',
        { cache: 'no-store' },
    ),
    /** @param {import('./api_types.js').ProjectCreateRequest} payload */
    projectCreate: (payload) => jsonPost('/api/projects', payload),
    projectUpdate: (projectId, name) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/update`, { name }),
    /** @returns {Promise<import('./api_types.js').ProjectDeleteResponse>} */
    projectDelete: (projectId) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/delete`, {}),
    /**
     * The owner's YES to a git_init_required offer: start tracking the project's
     * working folder with one attach-snapshot commit. Never called automatically —
     * admission raises the offer and stops; only the owner answers it.
     * @returns {Promise<import('./api_types.js').ProjectInitGitResponse>}
     */
    projectInitGit: (projectId) => jsonPost(`/api/projects/${encodeURIComponent(projectId)}/init-git`, {}),
    /**
     * Create a new empty thread in a project (a chat sharing its folder).
     * @returns {Promise<import('./api_types.js').ThreadResponse>}
     */
    projectThreadCreate: (projectId, name = '') => jsonPost(
        `/api/projects/${encodeURIComponent(projectId)}/threads`, name ? { name } : {},
    ),
    /** @returns {Promise<import('./api_types.js').ThreadResponse>} */
    projectThreadUpdate: (projectId, threadId, name) => jsonPost(
        `/api/projects/${encodeURIComponent(projectId)}/threads/${encodeURIComponent(threadId)}/update`,
        { name },
    ),
    /**
     * Fork a thread. The source is untouched: the new thread stores a cursor
     * into its rows, never a copy.
     * @returns {Promise<import('./api_types.js').ThreadResponse>}
     */
    projectThreadFork: (projectId, threadId) => jsonPost(
        `/api/projects/${encodeURIComponent(projectId)}/threads/${encodeURIComponent(threadId)}/fork`, {},
    ),
    /**
     * The bases this thread may branch off from (A8) — current branch, other
     * branches, tags, plus the always-present "exactly as it is now" entry.
     * @returns {Promise<import('./api_types.js').ThreadBranchBasesResponse>}
     */
    threadBranchBases: (projectId, threadId) => fetchJson(
        `${threadPath(projectId, threadId)}/branch-bases`, { cache: 'no-store' },
    ),
    /**
     * BRANCH OFF (A7): give this thread its own git worktree from `baseRef`
     * (a branch, tag, commit-ish, or '@snapshot' for "exactly as it is now").
     * @returns {Promise<import('./api_types.js').ThreadWorktreeResponse>}
     */
    threadBranchOff: (projectId, threadId, baseRef = '') => jsonPost(
        `${threadPath(projectId, threadId)}/branch-off`, baseRef ? { base_ref: baseRef } : {},
    ),
    /**
     * MERGE BACK (A7/A9). A conflict comes back as ok:false with `conflicts`;
     * the merge is already aborted by then and the thread keeps its branch.
     *
     * `acknowledgeCheckoutDirty` IS the owner's answer to the `checkout_dirty`
     * refusal, mirroring `threadWorktreeRemove`'s `acknowledgeUnmerged`. Without
     * it the server's only escape from that refusal had NO producer in any client
     * code, so one stray `build.log` in a checkout made merge-back permanently
     * unreachable — the exact failure the refusal was written to prevent.
     * @returns {Promise<import('./api_types.js').ThreadWorktreeResponse>}
     */
    threadMergeBack: (projectId, threadId, acknowledgeCheckoutDirty = false) => jsonPost(
        `${threadPath(projectId, threadId)}/merge-back`,
        acknowledgeCheckoutDirty ? { acknowledge_checkout_dirty: true } : {},
    ),
    /**
     * What removing this checkout would DESTROY — read before offering removal
     * (A10), never after.
     * @returns {Promise<import('./api_types.js').ThreadWorktreeResponse>}
     */
    threadWorktree: (projectId, threadId) => fetchJson(
        `${threadPath(projectId, threadId)}/worktree`, { cache: 'no-store' },
    ),
    /**
     * Remove this thread's checkout. `acknowledgeUnmerged` IS the owner's
     * consent: without it, unmerged work refuses with its inspection attached.
     * @returns {Promise<import('./api_types.js').ThreadWorktreeResponse>}
     */
    threadWorktreeRemove: (projectId, threadId, acknowledgeUnmerged = false) => jsonPost(
        `${threadPath(projectId, threadId)}/worktree/remove`,
        { acknowledge_unmerged: !!acknowledgeUnmerged },
    ),
    /**
     * A branched thread's own checkout diff (A13) — same envelope as taskDiff.
     * @returns {Promise<import('./api_types.js').ThreadDiffResponse>}
     */
    threadDiff: (projectId, threadId) => fetchJson(
        `${threadPath(projectId, threadId)}/diff`, { cache: 'no-store' },
    ),
    /**
     * Archive a thread: HIDE it. Nothing is removed and `threadRestore` puts it
     * back. A thread with a task still running stays visible until that task is
     * terminal — the answer says so via `visible_until_terminal`.
     * @returns {Promise<import('./api_types.js').ThreadLifecycleResponse>}
     */
    threadArchive: (projectId, threadId) => jsonPost(`${threadPath(projectId, threadId)}/archive`, {}),
    /** @returns {Promise<import('./api_types.js').ThreadLifecycleResponse>} */
    threadRestore: (projectId, threadId) => jsonPost(`${threadPath(projectId, threadId)}/restore`, {}),
    /**
     * Delete a thread: fence routing, cancel its tasks, then tombstone. The id is
     * never reused and the journal rows honestly remain. The checkout goes with
     * the thread (`worktree_removed`) — a tombstoned thread is invisible on every
     * surface, so one left behind is a folder and a branch nothing can reach.
     *
     * Two refusals, and only one of them is a question. Work at RISK (unmerged
     * commits, changes to tracked files, an unreadable checkout) refuses with
     * `checkout_holds_work` and names the removal route; nothing overrides it. A
     * checkout holding only ignored or untracked files answers
     * `checkout_holds_rebuildable_files`, and `acknowledgeUnmerged` is the owner's
     * yes to exactly that — the same argument shape `threadWorktreeRemove` and
     * `threadMergeBack` already take. Without a producer here the server's escape
     * would have no client at all and one `node_modules/` would make deleting a
     * thread a three-step detour (T3R2-H6 is precisely that class of defect).
     * @returns {Promise<import('./api_types.js').ThreadLifecycleResponse>}
     */
    threadDelete: (projectId, threadId, acknowledgeUnmerged = false) => jsonPost(
        `${threadPath(projectId, threadId)}/delete`,
        { acknowledge_unmerged: !!acknowledgeUnmerged },
    ),
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
