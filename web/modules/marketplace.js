/** ClawHub marketplace UI inside the Skills page. */

import {
    getPending,
    getPendingBySlug,
    setPending,
    startLifecyclePoller,
} from './lifecycle_card.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { jsonPost } from './api_client.js';
import { renderToneBadge } from './ui_helpers.js';
import {
    boundedText,
    emitSkillLifecycle,
    escapeHtmlAttr as escapeHtml,
    fetchJson,
    formatCompactNumber,
    grantReady,
    isRateLimitError,
    preflightFailed,
    preflightFailedStale,
    renderHubCard,
    renderSkillRepairPrompt,
    reviewReady,
    reviewTone,
    safeExternalHrefAttr,
    topReviewFinding,
} from './utils.js';

function installErrorCopy(message) {
    return isRateLimitError(message)
        ? `${message} Click Install again later to retry.`
        : message;
}

const safeExternalUrl = safeExternalHrefAttr;

const MARKETPLACE_SEARCH_LIMIT = 16;

/**
 * Ask which version to update a skill to (pure decision helper, node-tested
 * with an injected dialog). window.prompt is dead on the macOS desktop shell
 * (pywebview/WKWebView answers null silently), so this rides the in-house
 * input dialog. Contract preserved from the prompt() flow: cancel skips the
 * update entirely; confirmed-empty means "latest" (the server treats a
 * body without `version` as latest — gateway/marketplace.py).
 */
export async function promptUpdateVersion(slug, latest, { dialogImpl = openConfirmDialog } = {}) {
    const answer = await dialogImpl({
        title: `Update ${slug}`,
        body: `Update ${slug} to which version?\nLeave empty for latest (${latest || 'unknown'}).`,
        input: true,
        initialValue: latest || '',
        confirmLabel: 'Update',
    });
    if (!answer?.confirmed) return { confirmed: false, version: '' };
    return { confirmed: true, version: String(answer.value || '').trim() };
}


function controlsTemplate() {
    return `
        <div class="marketplace-controls">
            <input type="search" id="mp-query" class="marketplace-search ui-control" aria-label="Search ClawHub skills"
                   placeholder="Search ClawHub skills by name or summary…" autocomplete="off">
            <button class="btn btn-primary" data-mp-search>Search</button>
        </div>
        <div class="marketplace-filters">
            <label class="marketplace-filter-toggle">
                <input type="checkbox" id="mp-only-official">
                <span class="marketplace-filter-track" aria-hidden="true"></span>
                <span>Official only</span>
            </label>
        </div>
    `;
}


function paneTemplate({ includeControls = true } = {}) {
    return `
        <div class="marketplace-shell">
            ${includeControls ? controlsTemplate() : ''}
            <div id="mp-status" class="muted marketplace-status"></div>
            <div id="mp-results" class="marketplace-results"></div>
            <div id="mp-pagination" class="marketplace-pagination" hidden></div>
        </div>
    `;
}


function statusBadgeForReview(status) {
    const tone = status === 'clean' ? 'ok'
        : status === 'warnings' ? 'warn'
        : status === 'blockers' ? 'danger'
        : 'muted';
    return renderToneBadge(status || 'pending', tone);
}

function hasInstalledUiTab(installed) {
    return installed?.has_ui_tab === true;
}

export function lifecycleFor(summary, installed, pending, { installedUnavailable = false } = {}) {
    if (installedUnavailable) {
        return {
            tone: 'warn', label: 'Installed state unavailable',
            hint: '',
            action: '', button: 'Unavailable', disabled: true,
        };
    }
    if (pending) {
        if (pending.failed === true) {
            return {
                tone: pending.tone || 'danger',
                label: pending.label || 'Failed',
                hint: pending.message || '',
                action: pending.retry_action || '',
                button: pending.retry_label || 'Retry',
                disabled: !pending.retry_action,
            };
        }
        return {
            tone: pending.tone || 'warn',
            label: pending.label || 'Working',
            hint: pending.message || '',
            action: '',
            button: pending.label || 'Working...',
            disabled: true,
        };
    }
    if (!installed) {
        return {
            tone: 'muted',
            label: 'Not installed',
            hint: 'Install runs the adapter and starts security review automatically.',
            action: 'install',
            button: 'Install',
        };
    }
    if (installed.load_error) {
        return {
            tone: 'danger',
            label: 'Install needs fix',
            hint: installed.load_error,
            action: 'fix',
            button: 'Repair and run',
        };
    }
    if (installed.review_status === 'blockers' && !reviewReady(installed, { requireFresh: true })) {
        const finding = topReviewFinding(installed);
        return {
            tone: 'danger',
            label: 'Review blockers',
            hint: finding || 'Review has blocker findings; ask Ouroboros to repair the skill payload.',
            action: 'fix',
            button: 'Repair and run',
        };
    }
    // #335: a deterministic preflight FAIL persists as pending — Re-review
    // would fail the same way. The repair path fixes the payload instead.
    if (preflightFailed(installed) && !reviewReady(installed, { requireFresh: true })) {
        const finding = topReviewFinding(installed);
        return {
            tone: 'danger',
            label: 'Preflight failed',
            hint: finding || 'The deterministic preflight failed; ask Ouroboros to repair the skill payload.',
            action: 'fix',
            button: 'Repair and run',
        };
    }
    // D11: the recorded preflight FAIL is stale for the current payload bytes —
    // the cheap Re-review (which reruns the preflight) stays primary; the card
    // adds a secondary Repair based on the last recorded preflight.
    if (preflightFailedStale(installed) && !reviewReady(installed, { requireFresh: true })) {
        const finding = topReviewFinding(installed);
        return {
            tone: 'warn',
            label: 'Review stale',
            hint: finding
                ? `Last recorded preflight: ${finding}`
                : 'The last recorded preflight failed for a previous version of this skill; Re-review reruns it.',
            action: 'review',
            button: 'Re-review',
        };
    }
    if (!reviewReady(installed, { requireFresh: true })) {
        const finding = topReviewFinding(installed);
        return {
            tone: 'warn',
            label: installed.review_stale ? 'Review stale' : `Review ${installed.review_status || 'pending'}`,
            hint: finding || 'Review must pass before this skill can run.',
            action: 'review',
            button: installed.review_stale ? 'Re-review' : 'Review',
        };
    }
    if (!grantReady(installed)) {
        const missing = [
            ...(installed.grants?.missing_keys || []),
            ...(installed.grants?.missing_permissions || []),
        ];
        return {
            tone: 'warn',
            label: 'Needs grants',
            hint: missing.length ? `Missing: ${missing.join(', ')}` : 'Human key and permission grants required.',
            action: 'grant',
            button: 'Grant',
        };
    }
    if (!installed.enabled) {
        return {
            tone: 'ok',
            label: 'Ready',
            hint: 'Fresh executable review. Turn it on when you want the skill available.',
            action: 'enable',
            button: 'Enable',
        };
    }
    if (installed.type === 'extension' && hasInstalledUiTab(installed)) {
        return {
            tone: 'ok',
            label: 'Enabled',
            hint: 'Extension skills expose tools/routes and may add Widgets after loading.',
            action: 'widgets',
            button: 'Open widgets',
        };
    }
    if (installed.type === 'extension') {
        return {
            tone: 'ok',
            label: 'Enabled',
            hint: 'Extension is active. This skill does not expose a widget.',
            action: 'disable',
            button: 'Disable',
        };
    }
    return {
        tone: 'ok',
        label: 'Enabled',
        hint: 'Skill is enabled.',
        action: 'disable',
        button: 'Disable',
    };
}

function buildHealPrompt(installed, summary) {
    const findings = Array.isArray(installed?.review_findings) ? installed.review_findings : [];
    const diagnostics = {
        name: installed?.name || installed?.provenance?.sanitized_name || '',
        slug: summary?.slug || installed?.provenance?.slug || '',
        source: 'clawhub',
        payload_root: installed?.payload_root || '',
        type: installed?.type || 'unknown',
        initial_enabled: Boolean(installed?.enabled),
        content_hash: installed?.content_hash || '',
        review_status: installed?.review_status || 'pending',
        review_stale: Boolean(installed?.review_stale),
        load_error: boundedText(installed?.load_error || 'none', 2000),
        review_findings: findings.slice(0, 12).map((finding) => ({
            item: boundedText(finding.item || finding.check || finding.title || 'finding', 200),
            verdict: boundedText(finding.verdict || finding.severity || '', 80),
            reason: boundedText(finding.reason || finding.message || JSON.stringify(finding), 1200),
        })),
    };
    return renderSkillRepairPrompt(
        'Repair and run the ClawHub skill selected in the Marketplace UI.',
        JSON.stringify(diagnostics, null, 2),
    );
}


// D11 secondary affordance with the primary's pending discipline: a queued or
// running lifecycle job suppresses the stale-Repair button entirely (the card's
// pending chip already explains the state), so a concurrent repair cannot be
// enqueued while other work runs. Exported for renderer-level tests.
export function staleRepairSecondaryHtml(slug, installed, pending) {
    if (pending || !installed) return '';
    if (!preflightFailedStale(installed) || preflightFailed(installed)) return '';
    return `<button class="btn btn-default" data-mp-action="fix" data-slug="${escapeHtml(slug)}" title="Repair based on the last recorded preflight — the payload changed since that run">Repair and run</button>`;
}


function summaryCard(summary, installedMap, isPlugin, installedUnavailable = false) {
    const slug = summary.slug;
    const pending = getPending(slug);
    const installed = installedMap.get(slug);
    const installedAtVersion = installed?.provenance?.version || installed?.version || '';
    const isInstalled = !!installed;
    const updateAvailable = isInstalled
        && !installedUnavailable
        && summary.latest_version
        && installedAtVersion
        && summary.latest_version !== installedAtVersion;
    const downloads = formatCompactNumber(summary.stats?.downloads);
    const stars = formatCompactNumber(summary.stats?.stars);
    const license = summary.license || 'no-license';
    const homepageHref = safeExternalUrl(summary.homepage);
    const reviewBadge = isInstalled ? statusBadgeForReview(installed.review_status) : '';
    const lifecycle = lifecycleFor(summary, installed, pending, { installedUnavailable });
    const primaryHtml = isPlugin
        ? `<button class="btn btn-default" disabled title="OpenClaw Node/TypeScript plugins are not installable in Ouroboros. Use a Python port or MCP bridge.">Plugin</button>`
        : `<button class="btn btn-primary marketplace-next-action"
                   data-mp-action="${escapeHtml(lifecycle.action)}"
                   data-slug="${escapeHtml(slug)}"
                   ${lifecycle.disabled || !lifecycle.action ? 'disabled' : ''}>${escapeHtml(lifecycle.button)}</button>`;
    const secondaryHtml = isPlugin || installedUnavailable
        ? ''
        : isInstalled
            ? `
                ${staleRepairSecondaryHtml(slug, installed, pending)}
                ${updateAvailable ? `<button class="btn btn-default" data-mp-update="${escapeHtml(slug)}">Update</button>` : ''}
                ${installed.enabled && installed.type === 'extension' ? `<button class="btn btn-default" data-mp-action="disable" data-slug="${escapeHtml(slug)}">Disable</button>` : ''}
                <button class="btn btn-default" data-mp-uninstall="${escapeHtml(slug)}" data-name="${escapeHtml(installed.name || '')}">Uninstall</button>
            `
            : '';
    const badgesHtml = `
        ${isPlugin
        ? '<span class="skills-badge skills-badge-danger">plugin unsupported</span>'
        : ''}
        ${updateAvailable ? `<span class="skills-badge skills-badge-warn">update v${escapeHtml(summary.latest_version)}</span>` : ''}
        ${reviewBadge}
    `;
    const metaHtml = `
        <span>downloads: ${downloads}</span>
        <span>stars: ${stars}</span>
        <span>license: ${escapeHtml(license)}</span>
        ${homepageHref ? `<a href="${homepageHref}" target="_blank" rel="noopener noreferrer">homepage</a>` : ''}
        ${(summary.os || []).length ? `<span>os: ${(summary.os || []).map((o) => escapeHtml(o)).join(', ')}</span>` : ''}
    `;
    return renderHubCard(summary, { pending, installed, lifecycle, primaryHtml, secondaryHtml, badgesHtml, metaHtml, official: Boolean(summary.badges?.official) });
}


function renderResults(host, summaries, installedMap, registryCount, diagnostics) {
    if (!summaries.length) {
        const query = String(diagnostics?.query || '').trim();
        const official = Boolean(diagnostics?.official);
        const mode = query ? 'matching your search' : 'in the marketplace browse list';
        const officialText = official ? ' official' : '';
        if (registryCount > 0) {
            host.innerHTML = `<div class="muted">No installable${officialText} skills found ${mode}.</div>`;
        } else {
            const attempts = Array.isArray(diagnostics?.attempts) && diagnostics.attempts.length
                ? `<details class="marketplace-debug"><summary>Registry diagnostics</summary><pre>${escapeHtml(JSON.stringify(diagnostics.attempts, null, 2))}</pre></details>`
                : '';
            host.innerHTML = `
                <div class="muted">
                    No installable${officialText} skills found ${mode}.
                </div>
                ${attempts}
            `;
        }
        return;
    }
    host.innerHTML = summaries
        .map((s) => summaryCard(s, installedMap, !!s.is_plugin, diagnostics?.installedUnavailable))
        .join('');
}


function renderPagination(host, { query, limit, count, cursor, hasPrevious, nextCursor }) {
    const searchMode = Boolean(String(query || '').trim());
    if (searchMode || (!nextCursor && !hasPrevious)) {
        host.hidden = true;
        host.innerHTML = '';
        return;
    }
    host.hidden = false;
    host.innerHTML = `
        <button class="btn btn-default" data-mp-prev ${hasPrevious ? '' : 'disabled'}>Prev</button>
        <span class="muted">${cursor ? 'cursor page' : 'first page'} · ${count} shown</span>
        <button class="btn btn-default" data-mp-next ${nextCursor ? '' : 'disabled'}>Next</button>
    `;
}


function showStatus(host, message, tone) {
    const el = host.querySelector('#mp-status');
    if (!el) return;
    el.dataset.tone = tone || '';
    el.textContent = message || '';
}


async function loadInstalled({ signal: externalSignal } = {}) {
    // The primary installed read is bounded only by its caller (a newer
    // refresh); the optional live-details enrichment has its own 3 s bound,
    // so a slow installed-skill list never turns installed state unavailable.
    const enrichment = new AbortController();
    const timer = setTimeout(() => enrichment.abort(), 3000);
    const onExternalAbort = () => enrichment.abort();
    if (externalSignal) {
        if (externalSignal.aborted) enrichment.abort();
        else externalSignal.addEventListener('abort', onExternalAbort, { once: true });
    }
    try {
        const [data, catalog] = await Promise.all([
            fetchJson('/api/marketplace/clawhub/installed', externalSignal ? { signal: externalSignal } : {}),
            fetchJson('/api/extensions', { signal: enrichment.signal }).catch(() => null),
        ]);
        if (!Array.isArray(data?.skills)) throw new Error('Installed skills response is unavailable.');
        const uiTabSkills = new Set(
            (catalog?.live?.ui_tabs || [])
                .map((tab) => String(tab?.skill || tab?.skill_name || tab?.extension || ''))
                .filter(Boolean)
        );
        const byName = new Map();
        for (const skill of catalog?.skills || []) {
            if (skill.name) byName.set(skill.name, { ...skill, has_ui_tab: uiTabSkills.has(skill.name) });
        }
        const map = new Map();
        for (const skill of data.skills || []) {
            const merged = { ...skill, ...(byName.get(skill.name) || {}) };
            const provSlug = skill.provenance?.slug;
            if (provSlug) map.set(provSlug, merged);
        }
        return { map, available: true, enrichmentAvailable: Array.isArray(catalog?.skills) };
    } catch (err) {
        if (err?.name !== 'AbortError') {
            console.warn('marketplace: installed lookup failed', err);
        }
        return { map: null, available: false, error: err };
    } finally {
        clearTimeout(timer);
        if (externalSignal) externalSignal.removeEventListener('abort', onExternalAbort);
    }
}


async function runSearch(state, { signal } = {}) {
    const params = new URLSearchParams();
    const query = String(state.query || '').trim();
    if (query) params.set('q', query);
    params.set('limit', String(query ? MARKETPLACE_SEARCH_LIMIT : state.limit));
    if (!query && state.cursor) params.set('cursor', state.cursor);
    if (state.onlyOfficial) params.set('official', '1');
    return fetchJson(`/api/marketplace/clawhub/search?${params.toString()}`, { signal });
}


export function initMarketplace(pane, controlsHost = null) {
    pane.innerHTML = paneTemplate({ includeControls: !controlsHost });
    if (controlsHost) {
        controlsHost.innerHTML = controlsTemplate();
    }

    const state = {
        query: '',
        limit: 25,
        onlyOfficial: false,
        results: [],
        installedMap: new Map(),
        installedUnavailable: true,
        // Installed reads asked for (owed) and the newest one a landed read covers (read):
        // a superseding search, which does not ask, still reads while owed > read.
        installedOwed: 0,
        installedRead: 0,
        enrichmentUnavailable: false,
        catalogLoaded: false,
        cursor: '',
        cursorHistory: [],
        nextCursor: '',
        registryPath: 'packages',
        registryAttempts: [],
    };

    const controlsRoot = controlsHost || pane;
    const queryInput = controlsRoot.querySelector('#mp-query');
    const onlyOfficial = controlsRoot.querySelector('#mp-only-official');
    const searchBtn = controlsRoot.querySelector('[data-mp-search]');
    const resultsHost = pane.querySelector('#mp-results');
    const paginationHost = pane.querySelector('#mp-pagination');

    let debounceTimer = null;
    // Abort + token guard prevent slow stale searches from overwriting fresh UI.
    let activeController = null;
    let refreshToken = 0;
    let destroyed = false;

    function syncControlsForMode() {
        const searchMode = Boolean(String(state.query || '').trim());
        onlyOfficial.title = searchMode
            ? 'Filters enriched search results to skills marked official.'
            : '';
    }

    function renderCurrentResults() {
        if (destroyed || !state.catalogLoaded) return;
        renderResults(resultsHost, state.results, state.installedMap, state.results.length, {
            query: state.query,
            official: state.onlyOfficial,
            registryPath: state.registryPath,
            attempts: state.registryAttempts,
            installedUnavailable: state.installedUnavailable,
        });
    }

    function showActionFailure(slug, message, tone = 'danger') {
        if (destroyed) return;
        const cardVisible = Array.from(resultsHost.querySelectorAll('[data-slug]'))
            .some(card => card.dataset.slug === slug);
        if (!cardVisible) showStatus(pane, `${slug}: ${message}`, tone);
    }

    async function refresh({ installed: readInstalled = true } = {}) {
        if (destroyed) return;
        syncControlsForMode();
        const query = String(state.query || '').trim();
        showStatus(pane, query ? `Searching for "${query}"…` : 'Browsing ClawHub…', 'muted');
        if (activeController) {
            try { activeController.abort(); } catch (_) { /* ignore */ }
        }
        const myController = new AbortController();
        activeController = myController;
        const myToken = ++refreshToken;
        // Typing searches the registry only; the installed state it already
        // knows is kept, and an owed, unknown or unavailable one is read again.
        if (readInstalled) state.installedOwed += 1;
        const owedAtStart = state.installedOwed;
        const withInstalled = state.installedOwed > state.installedRead || state.installedUnavailable;
        try {
            const [catalog, installed] = await Promise.all([
                runSearch(state, { signal: myController.signal }).then(data => ({ data }), error => ({ error })),
                withInstalled ? loadInstalled({ signal: myController.signal }) : null,
            ]);
            if (destroyed || myToken !== refreshToken) return;
            if (installed) {
                state.installedUnavailable = !installed.available;
                state.enrichmentUnavailable = installed.available && !installed.enrichmentAvailable;
                if (installed.available) {
                    state.installedMap = installed.map;
                    state.installedRead = Math.max(state.installedRead, owedAtStart);
                }
            }
            state.installedMap.pendingBySlug = getPendingBySlug();
            if (catalog.error) throw catalog.error;
            const data = catalog.data;
            if (!Array.isArray(data?.results)) throw new Error('Marketplace response is unavailable.');
            state.results = data.results;
            state.catalogLoaded = true;
            state.nextCursor = data.next_cursor || '';
            state.registryPath = data.registry_path || 'packages';
            state.registryAttempts = data.registry_attempts || [];
            const registryWarnings = Array.isArray(data.registry_warnings) ? data.registry_warnings : [];
            renderCurrentResults();
            renderPagination(paginationHost, {
                query,
                limit: state.limit,
                count: state.results.length,
                cursor: state.cursor,
                hasPrevious: state.cursorHistory.length > 0,
                nextCursor: state.nextCursor,
            });
            const mode = query ? 'search' : 'browse';
            const official = state.onlyOfficial ? ' · official only' : '';
            if (state.installedUnavailable) {
                showStatus(pane, 'Installed skills could not be read. Previous installed details are shown where available; Refresh to retry.', 'warn');
            } else if (state.enrichmentUnavailable) {
                showStatus(pane, `${state.results.length} skills · live widget details unavailable. Refresh to retry.`, 'warn');
            } else if (registryWarnings.length) {
                showStatus(pane, `${state.results.length} skill${state.results.length === 1 ? '' : 's'} · ${mode}${official} · ${state.registryPath} · ${registryWarnings[0]}`, 'warn');
            } else {
                showStatus(pane, `${state.results.length} skill${state.results.length === 1 ? '' : 's'} · ${mode}${official} · ${state.registryPath}`, 'muted');
            }
        } catch (err) {
            if (destroyed || err?.name === 'AbortError' || myToken !== refreshToken) return;
            const rawMessage = String(err?.body?.error || err?.message || err || '');
            const firstLine = rawMessage.split('\n').map((line) => line.trim()).filter(Boolean)[0] || 'Marketplace request failed';
            const timeout = /timed out|timeout/i.test(rawMessage);
            const message = timeout
                ? 'ClawHub did not respond in time. Try again, or search by name to narrow the request.'
                : firstLine.replace(/^Error:\s*/i, '');
            showStatus(pane, `${message}${state.catalogLoaded ? ' · Showing previous results.' : ''}`, 'danger');
            renderCurrentResults();
            paginationHost.hidden = true;
        } finally {
            if (activeController === myController) activeController = null;
        }
    }

    function scheduleRefresh(immediate, options = {}) {
        if (destroyed) return;
        if (options.installed !== false) state.installedOwed += 1;  // survives a replaced timer
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => refresh(options), immediate ? 0 : 300);
    }

    pane._marketplaceRefresh = () => {
        clearTimeout(debounceTimer);
        return refresh();
    };

    const disposeLifecycle = startLifecyclePoller(() => {
        state.installedMap.pendingBySlug = getPendingBySlug();
        renderCurrentResults();
    });
    pane._marketplaceDestroy = () => {
        destroyed = true;
        clearTimeout(debounceTimer);
        activeController?.abort();
        disposeLifecycle();
    };

    async function toggleInstalledSkill(installed, enabled) {
        return jsonPost(`/api/skills/${encodeURIComponent(installed.name)}/toggle`, { enabled });
    }

    async function runLifecycleAction(slug, action) {
        const summary = state.results.find((item) => item.slug === slug) || { slug };
        const installed = state.installedMap.get(slug);
        if (action === 'widgets') {
            document.querySelector('[data-nav-page="widgets"]')?.click();
            return;
        }
        if (action === 'disable' && installed) {
            setPending(slug, { label: 'Turning off', tone: 'warn', message: 'Disabling skill…' });
            const result = await toggleInstalledSkill(installed, false);
            showStatus(pane, `${slug} disabled`, 'ok');
            emitSkillLifecycle('disable', installed.name, result);
            return;
        }
        if (action === 'enable' && installed) {
            setPending(slug, { label: 'Enabling', tone: 'warn', message: 'Turning skill on…' });
            const result = await toggleInstalledSkill(installed, true);
            showStatus(pane, `${slug} enabled`, 'ok');
            emitSkillLifecycle('enable', installed.name, result);
            return;
        }
        if (action === 'grant' && installed) {
            const items = [
                ...(installed.grants?.missing_keys || installed.grants?.requested_keys || []),
                ...(installed.grants?.missing_permissions || installed.grants?.requested_permissions || []),
            ];
            if (!items.length) throw new Error('No grant keys or permissions reported for this skill.');
            const ok = await openConfirmDialog({
                title: `Grant access to ${installed.name}`,
                body: `Grant ${installed.name} access to these keys and permissions?\n\n${items.join('\n')}\n\nOnly grant access to reviewed skills you trust.`,
                confirmLabel: 'Grant access',
            });
            if (!ok) return;
            const bridge = window.pywebview?.api?.request_skill_key_grant;
            setPending(slug, { label: 'Granting', tone: 'warn', message: 'Waiting for human confirmation…' });
            const result = bridge
                ? await bridge(installed.name, items)
                : await jsonPost(`/api/skills/${encodeURIComponent(installed.name)}/grants`, {
                    items, expected_content_hash: installed.content_hash,
                });
            if (!result?.ok) throw new Error(result?.error || 'Skill grant was cancelled.');
            showStatus(pane, `${slug} grant saved`, 'ok');
            emitSkillLifecycle('grant', installed.name, result);
            return;
        }
        if (action === 'fix' && installed) {
            const ok = await openConfirmDialog({
                title: `Repair and run ${installed.name || slug}`,
                body: `Start a repair task for ${installed.name || slug}? Ouroboros will repair the selected skill, review it, enable it when its prerequisites are ready, and test the result. The repaired skill stays running unless you stop or disable it. If the task cannot start, chat will show why.`,
                confirmLabel: 'Repair and run',
            });
            if (!ok) return;
            setPending(slug, { label: 'Repair requested', tone: 'warn', message: 'Sending repair request…' });
            await jsonPost('/api/command', {
                cmd: buildHealPrompt(installed, summary),
                task_constraint: { mode: 'normal', skill_name: installed.name || '', payload_root: installed.payload_root || '', allow_enable: false, allow_review: true },
                visible_task_id: `skill_repair_${installed.name || slug}`,
            });
            showStatus(pane, `${slug}: repair request sent — watch for its live card`, 'ok');
            emitSkillLifecycle('repair', installed.name || slug);
            document.querySelector('[data-nav-page="chat"]')?.click();
            return;
        }
        if (action === 'review' && installed) {
            setPending(slug, { label: 'Reviewing', tone: 'warn', message: 'Running skill review…' });
            const result = await jsonPost(`/api/skills/${encodeURIComponent(installed.name)}/review`);
            showStatus(
                pane,
                `${slug}: review ${result.status}${result.error ? ` — ${result.error}` : ''}`,
                reviewTone(result.status, result.error),
            );
            emitSkillLifecycle('review', installed.name, result);
            return;
        }
        if (action === 'update' && installed) {
            const retryVersion = getPending(slug)?.retry_version || '';
            const target = getPending(slug)?.target || installed.name;
            setPending(slug, {
                label: 'Updating',
                tone: 'warn',
                message: 'Updating skill…',
                target,
                retry_version: retryVersion,
            });
            const body = retryVersion ? { version: retryVersion } : {};
            const result = await jsonPost(`/api/marketplace/clawhub/update/${encodeURIComponent(target)}`, body);
            if (!result.ok) throw new Error(result.error || 'update failed');
            showStatus(pane, `Updated ${slug} — review ${result.review_status}`, reviewTone(result.review_status));
            emitSkillLifecycle('update', installed.name, result);
            return;
        }
        if (action === 'install') {
            setPending(slug, { label: 'Installing', tone: 'warn', message: 'Downloading, adapting, and reviewing…' });
            const result = await jsonPost('/api/marketplace/clawhub/install', { slug, auto_review: true });
            if (!result.ok) throw new Error(result.error || 'install failed');
            const installedName = result.sanitized_name;
            const requestedGrants = result.provenance?.requested_key_grants || [];
            if (['clean', 'warnings'].includes(result.review_status) && installedName && !requestedGrants.length) {
                showStatus(pane, `Installed ${slug}; review passed. Enable it from the card when ready.`, 'ok');
            } else if (['clean', 'warnings'].includes(result.review_status) && requestedGrants.length) {
                showStatus(pane, `Installed ${slug}; grant required before enabling`, 'warn');
            } else if (result.review_error) {
                showStatus(pane, `Installed ${slug}; review could not finish: ${result.review_error}`, 'warn');
            } else {
                showStatus(pane, `Installed ${slug}; review ${result.review_status || 'pending'}`, reviewTone(result.review_status));
            }
            emitSkillLifecycle('install', installedName || slug, result);
        }
    }

    queryInput.addEventListener('input', (event) => {
        state.query = event.target.value || '';
        state.cursor = '';
        state.cursorHistory = [];
        scheduleRefresh(false, { installed: false });
    });
    queryInput.addEventListener('keydown', (event) => {
        // Enter triggers the same immediate search as the button.
        if (event.key === 'Enter') {
            event.preventDefault();
            scheduleRefresh(true);
        }
    });
    onlyOfficial.addEventListener('change', () => {
        state.onlyOfficial = onlyOfficial.checked;
        state.cursor = '';
        state.cursorHistory = [];
        scheduleRefresh(true);
    });
    searchBtn.addEventListener('click', () => {
        // Explicit Search starts from a cursorless first page.
        state.cursor = '';
        state.cursorHistory = [];
        scheduleRefresh(true);
    });

    paginationHost.addEventListener('click', (event) => {
        const prev = event.target.closest('[data-mp-prev]');
        const next = event.target.closest('[data-mp-next]');
        if (prev) {
            state.cursor = state.cursorHistory.pop() || '';
            scheduleRefresh(true);
        } else if (next) {
            if (state.nextCursor) {
                state.cursorHistory.push(state.cursor || '');
                state.cursor = state.nextCursor;
            }
            scheduleRefresh(true);
        }
    });

    resultsHost.addEventListener('click', async (event) => {
        const actionBtn = event.target.closest('[data-mp-action]');
        const updateBtn = event.target.closest('[data-mp-update]');
        const uninstallBtn = event.target.closest('[data-mp-uninstall]');
        if ((actionBtn || updateBtn || uninstallBtn) && state.installedUnavailable) {
            showStatus(pane, 'Installed skills could not be read. Refresh before retrying this action.', 'warn');
            return;
        }
        if (actionBtn) {
            const slug = actionBtn.dataset.slug;
            const action = actionBtn.dataset.mpAction;
            if (!slug || !action) return;
            actionBtn.disabled = true;
            let failedMessage = '';
            try {
                await runLifecycleAction(slug, action);
            } catch (err) {
                failedMessage = action === 'install'
                    ? installErrorCopy(err.message || String(err))
                    : (err.message || String(err));
                const tone = action === 'install' && isRateLimitError(failedMessage) ? 'warn' : 'danger';
                setPending(slug, {
                    ...getPending(slug),
                    label: `${action} failed`,
                    tone,
                    message: failedMessage,
                    failed: true,
                    retry_action: action,
                    retry_label: action === 'install' ? 'Retry install' : `Retry ${action}`,
                });
                showActionFailure(slug, failedMessage, tone);
            } finally {
                if (!failedMessage) setPending(slug, null);
                actionBtn.disabled = false;
                // Coalesce action refreshes through the same abort/token guard.
                if (!failedMessage) scheduleRefresh(true);
            }
            return;
        }
        if (updateBtn) {
            updateBtn.disabled = true;
            const slug = updateBtn.dataset.mpUpdate;
            const installed = state.installedMap.get(slug);
            const sanitized = installed?.name;
            if (!sanitized) {
                showStatus(pane, `Cannot update ${slug}: no provenance found`, 'danger');
                updateBtn.disabled = false;
                return;
            }
            // Empty version means latest; cancel skips update. slug/installed/
            // sanitized/latest are all captured BEFORE the await below: the
            // 5-second lifecycle poller re-renders the results DOM under the
            // open dialog, so nothing may be read off the row afterwards.
            const summary = state.results.find((s) => s.slug === slug);
            const latest = summary?.latest_version || '';
            const asked = await promptUpdateVersion(slug, latest);
            if (!asked.confirmed) {
                updateBtn.disabled = false;
                return;
            }
            const targetVersion = asked.version;
            showStatus(pane, `Updating ${slug}${targetVersion ? ` → v${targetVersion}` : ' (latest)'}…`, 'muted');
            setPending(slug, {
                label: 'Updating',
                tone: 'warn',
                message: 'Updating skill…',
                target: sanitized,
                retry_version: targetVersion,
            });
            let updateError = '';
            try {
                const body = targetVersion ? { version: targetVersion } : {};
                const result = await jsonPost(`/api/marketplace/clawhub/update/${encodeURIComponent(sanitized)}`, body);
                if (!result.ok) {
                    throw new Error(result.error || 'update failed');
                } else {
                    showStatus(pane, `Updated ${slug} — review ${result.review_status}`, reviewTone(result.review_status));
                    setPending(slug, null);
                    emitSkillLifecycle('update', sanitized, result);
                }
            } catch (err) {
                updateError = err.message || String(err);
                setPending(slug, {
                    label: 'Failed',
                    tone: 'danger',
                    message: updateError,
                    failed: true,
                    retry_action: 'update',
                    retry_label: 'Retry update',
                    target: sanitized,
                    retry_version: targetVersion,
                });
            } finally {
                updateBtn.disabled = false;
                await pane._marketplaceRefresh();
                if (updateError) showActionFailure(slug, updateError);
            }
            return;
        }
        if (uninstallBtn) {
            const slug = uninstallBtn.dataset.mpUninstall;
            const sanitized = uninstallBtn.dataset.name;
            const ok = await openConfirmDialog({
                title: `Uninstall ${slug}`,
                body: `Uninstall ${slug}? This deletes data/skills/clawhub/${sanitized}/.`,
                confirmLabel: 'Uninstall',
                danger: true,
            });
            if (!ok) return;
            uninstallBtn.disabled = true;
            try {
                await jsonPost(`/api/marketplace/clawhub/uninstall/${encodeURIComponent(sanitized)}`);
                showStatus(pane, `Uninstalled ${slug}`, 'ok');
                emitSkillLifecycle('uninstall', sanitized);
            } catch (err) {
                showStatus(pane, `Uninstall error: ${err.message}`, 'danger');
            } finally {
                uninstallBtn.disabled = false;
                scheduleRefresh(true);
            }
        }
    });

    return refresh();
}
