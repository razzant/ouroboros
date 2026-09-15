import { initMarketplace } from './marketplace.js';
import { initOuroborosHub } from './ouroboroshub.js';
import { bindTabStrip, renderPageHeader, renderTabStrip } from './page_header.js';
import { bindMenu } from './ui_interactions.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { PAGE_ICONS } from './page_icons.js';
import { showToast } from './toast.js';
import { apiClient, apiFetch } from './api_client.js';
import { patchInstalledSkillEnrichment, renderInstalledSkillCard, renderSkillHubBadges } from './skill_card_renderer.js';
import { runSkillPublishFlow } from './skill_publish_flow.js';
import { installedTime } from './ui_helpers.js';
import {
    boundedText,
    emitSkillLifecycle,
    escapeHtmlAttr as escapeHtml,
    fetchJson,
    grantReady,
    renderSkillRepairPrompt,
    reviewTone,
    reviewReady,
} from './utils.js';

const SKILLS_TABS = [
    { value: 'installed', label: 'My skills', pillId: 'skills-tab-pill-installed', tabId: 'skills-tab-installed', panelId: 'skills-pane-installed' },
    { value: 'marketplace', label: 'ClawHub', pillId: 'skills-tab-pill-marketplace', tabId: 'skills-tab-marketplace', panelId: 'skills-pane-marketplace' },
    { value: 'ouroboroshub', label: 'OuroborosHub', pillId: 'skills-tab-pill-ouroboroshub', tabId: 'skills-tab-ouroboroshub', panelId: 'skills-pane-ouroboroshub' },
];
const LIFECYCLE_VISIBLE_STATUSES = new Set(['queued', 'running', 'failed']);

/** Installed skills UI: review, grant, enable, repair, update, uninstall, delete. */

function skillsPageTemplate() {
    return `
        <section class="page app-page-glass" id="page-skills">
            ${renderPageHeader({
                title: 'Skills',
                icon: PAGE_ICONS.skills,
                description: 'Skills extend Ouroboros with new tools, routes, and widgets. Each skill is reviewed for safety before you turn it on.',
                actionsHtml: '<button id="skills-refresh" class="btn btn-default btn-sm">Refresh</button>',
                tabsHtml: renderTabStrip({
                    items: SKILLS_TABS,
                    active: 'installed',
                    dataAttr: 'data-tab',
                    activeClass: 'is-active',
                    ariaLabel: 'Skills views',
                    stripClass: 'skills-tabs',
                    tabClass: 'skills-tab',
                }),
            })}
            <div class="skills-search-chrome" id="skills-pane-marketplace-chrome" data-chrome-pane="marketplace" hidden></div>
            <div class="skills-search-chrome" id="skills-pane-ouroboroshub-chrome" data-chrome-pane="ouroboroshub" hidden></div>
            <div class="skills-scroll scroll-fade-y">
                <div class="skills-tab-panel" id="skills-pane-installed" data-pane="installed" role="tabpanel" aria-labelledby="skills-tab-installed">
                <div id="skills-status" class="muted" role="status" aria-live="polite"></div>
                <div id="skills-list" class="skills-list"></div>
                <div id="skills-empty" class="muted" hidden>
                    No skills yet. Browse <b>ClawHub</b> or
                    <b>OuroborosHub</b> to add one, or import a custom
                    package from the Files tab.
                </div>
            </div>
                <div class="skills-tab-panel" id="skills-pane-marketplace" data-pane="marketplace" role="tabpanel" aria-labelledby="skills-tab-marketplace" hidden></div>
                <div class="skills-tab-panel" id="skills-pane-ouroboroshub" data-pane="ouroboroshub" role="tabpanel" aria-labelledby="skills-tab-ouroboroshub" hidden></div>
            </div>
        </section>
    `;
}


function isMissingGrantLoadError(skill) {
    return !grantReady(skill) && String(skill.load_error || '').includes('missing owner grants');
}

function sortSkillsForDisplay(skills) {
    return [...skills].sort((a, b) => {
        if (a.lifecycle_virtual && !b.lifecycle_virtual) return -1;
        if (!a.lifecycle_virtual && b.lifecycle_virtual) return 1;
        return installedTime(b) - installedTime(a) || String(a.name || '').localeCompare(String(b.name || ''));
    });
}


// OuroborosHub catalog snapshot for the display-only My-skills sync badges
// (hub_sync verdict). Fetched once per Skills page open and reused across
// re-renders; fail-soft — a failed fetch only hides catalog-derived badges.
const hubCatalog = { promise: null, available: false, byName: new Map(), generation: 0 };

function loadHubCatalog(force = false) {
    if (force) hubCatalog.promise = null;
    if (!hubCatalog.promise) {
        // Generation guard: only the NEWEST request may commit its snapshot,
        // so a slow older response cannot overwrite fresher catalog state.
        const generation = ++hubCatalog.generation;
        hubCatalog.promise = fetchJson('/api/marketplace/ouroboroshub/catalog')
            .then((data) => {
                if (generation !== hubCatalog.generation) return;
                const byName = new Map();
                for (const row of data.results || []) {
                    const key = String(row.sanitized_name || row.slug || '');
                    if (key && !byName.has(key)) byName.set(key, row);
                }
                hubCatalog.byName = byName;
                hubCatalog.available = true;
            })
            .catch(() => {
                if (generation !== hubCatalog.generation) return;
                hubCatalog.byName = new Map();
                hubCatalog.available = false;
            });
    }
    return hubCatalog.promise;
}

let skillsRenderGeneration = 0;
let skillsSnapshot = null;


async function fetchSkills() {
    const extResp = await apiClient.extensions();
    if (!Array.isArray(extResp?.skills)) throw new Error('Installed skills response is unavailable.');
    return { skills: extResp.skills, live: extResp.live || {} };
}


function lifecycleEventsFromQueue(queueResp) {
    const events = Array.isArray(queueResp?.events) ? queueResp.events : [];
    const active = queueResp?.active;
    if (!active || typeof active !== 'object') return events;
    const activeId = String(active.id || '');
    const deduped = activeId
        ? events.filter((event) => String(event?.id || '') !== activeId)
        : events;
    return [...deduped, active];
}


function mergeLifecycleEvents(skills, events) {
    const out = skills.map((skill) => ({ ...skill }));
    const byName = new Map(out.map((skill) => [skill.name, skill]));
    const names = new Set(byName.keys());
    const processedTargets = new Set();
    for (const event of [...events].reverse()) {
        const name = event.target;
        if (!name) continue;
        if (processedTargets.has(name)) continue;
        processedTargets.add(name);
        if (!LIFECYCLE_VISIBLE_STATUSES.has(event.status)) continue;
        if (names.has(name)) {
            // The skill already has a real card. Annotate it with the in-flight
            // transition so it can show "Disabling…/Enabling…" instead of a stale
            // clean toggle while the (serialized) lifecycle lane works through it.
            // Events are reversed → newest first, so the first wins per skill.
            const existing = byName.get(name);
            if (existing) {
                existing.lifecycle_status = event.status;
                existing.lifecycle_kind = event.kind || existing.lifecycle_kind || '';
                existing.lifecycle_pending = event.status !== 'failed';
                if (event.status === 'failed' && event.error) existing.lifecycle_error = event.error;
            }
            continue;
        }
        names.add(name);
        out.unshift({
            name,
            description: event.message || event.error || 'Skill lifecycle operation',
            version: '—',
            type: 'skill',
            enabled: false,
            review_status: 'pending',
            review_stale: true,
            permissions: [],
            load_error: event.status === 'failed' ? event.error : '',
            source: event.source || 'external',
            lifecycle_kind: event.kind || '',
            lifecycle_status: event.status,
            lifecycle_pending: event.status !== 'failed',
            lifecycle_error: event.error || '',
            lifecycle_virtual: true,
            grants: { all_granted: true },
        });
    }
    return out;
}


function updateQueueBadges(events) {
    const latestByTarget = new Map();
    const untargeted = [];
    for (const event of [...events].reverse()) {
        const target = event.target || '';
        if (!target) {
            untargeted.push(event);
            continue;
        }
        if (!latestByTarget.has(target)) latestByTarget.set(target, event);
    }
    const actionable = [...latestByTarget.values(), ...untargeted]
        .filter((event) => LIFECYCLE_VISIBLE_STATUSES.has(event.status));
    const bySource = new Map();
    for (const event of actionable) {
        const source = event.source === 'ouroboroshub' ? 'ouroboroshub'
            : event.source === 'clawhub' ? 'marketplace'
            : 'installed';
        bySource.set(source, (bySource.get(source) || 0) + 1);
    }
    for (const [id, count] of bySource.entries()) {
        const el = document.getElementById(`skills-tab-pill-${id}`);
        if (!el) continue;
        el.hidden = !count;
        el.textContent = count ? String(count) : '';
    }
    for (const id of ['installed', 'marketplace', 'ouroboroshub']) {
        if (bySource.has(id)) continue;
        const el = document.getElementById(`skills-tab-pill-${id}`);
        if (!el) continue;
        el.hidden = true;
        el.textContent = '';
    }
}


async function renderSkillsList(container, emptyEl, reviewingSkills = new Set(), repairingSkills = new Set(), interactions = {}) {
    if (!container.isConnected) return;
    const renderGeneration = ++skillsRenderGeneration;
    const current = () => renderGeneration === skillsRenderGeneration && container.isConnected;
    const catalogSettled = loadHubCatalog();
    const status = document.getElementById('skills-status');
    // Keep primary rows pristine: merging a newer terminal queue into already
    // annotated cards would retain old lifecycle_pending/error fields.
    const snapshot = { ...skillsSnapshot, rawSkills: null, live: {} };
    const reads = { state: 'loading', queue: 'loading' };
    const labels = { state: 'Skill settings', queue: 'Lifecycle progress' };
    const catalogOptions = () => ({
        githubTokenConfigured: snapshot.githubTokenConfigured,
        hubCatalogByName: hubCatalog.byName,
        hubCatalogAvailable: hubCatalog.available,
    });
    const projected = () => sortSkillsForDisplay(mergeLifecycleEvents(snapshot.rawSkills, lifecycleEventsFromQueue(snapshot.queue)));
    function showReadState() {
        if (!status) return;
        status.className = 'muted';
        status.textContent = Object.entries(reads).filter(([, state]) => state !== 'ready')
            .map(([name, state]) => state === 'loading' ? `${labels[name]} loading…`
                : `${labels[name]} could not be refreshed. Refresh to retry; previous details are retained where available.`).join(' ');
    }
    function applyEnrichment() {
        if (!current() || !snapshot.rawSkills) return;
        skillsSnapshot = snapshot;
        const skills = projected();
        const cards = new Map(Array.from(container.querySelectorAll(':scope > .skills-card')).map(card => [card.dataset.skill, card]));
        for (const skill of [...skills].reverse()) {
            const card = cards.get(skill.name);
            if (card) {
                patchInstalledSkillEnrichment(card, skill, reviewingSkills, repairingSkills, snapshot.live,
                    catalogOptions(), interactions.menuFor?.(card) || card);
                cards.delete(skill.name);
            } else {
                const template = document.createElement('template');
                template.innerHTML = renderInstalledSkillCard(skill, reviewingSkills, repairingSkills, snapshot.live, catalogOptions());
                container.insertBefore(template.content.firstElementChild, container.firstElementChild);
            }
        }
        for (const card of cards.values()) {
            interactions.beforeReplace?.(card);
            card.remove();
        }
        if (emptyEl) emptyEl.hidden = skills.length > 0;
        updateQueueBadges(lifecycleEventsFromQueue(snapshot.queue));
        showReadState();
    }
    function settleOptional(name, data) {
        if (!current()) return;
        const valid = name === 'state' ? typeof data?.github_token_configured === 'boolean'
            : Array.isArray(data?.events) || (data?.active && typeof data.active === 'object');
        reads[name] = valid ? 'ready' : 'failed';
        if (valid) {
            if (name === 'state') snapshot.githubTokenConfigured = data.github_token_configured;
            else snapshot.queue = data;
        }
        applyEnrichment();
    }
    // Neither optional read participates in the primary list/Refresh promise.
    apiClient.state().then(data => settleOptional('state', data), () => settleOptional('state', null));
    apiClient.skillLifecycleQueue().then(data => settleOptional('queue', data), () => settleOptional('queue', null));
    if (status) {
        status.className = 'muted';
        status.textContent = 'Loading installed skills…';
    }
    try {
        const primary = await fetchSkills();
        if (!current()) return;
        snapshot.rawSkills = primary.skills;
        snapshot.live = primary.live;
    } catch (err) {
        if (!current()) return;
        if (emptyEl) emptyEl.hidden = true;
        if (status) {
            status.className = 'skills-load-error';
            status.textContent = `Could not load installed skills: ${err.message || err}.${skillsSnapshot ? ' Showing the previous list; Refresh to retry.' : ' Refresh to retry.'}`;
        }
        return;
    }
    if (!current()) return;
    skillsSnapshot = snapshot;
    const skills = projected();
    updateQueueBadges(lifecycleEventsFromQueue(snapshot.queue));
    showReadState();
    if (emptyEl) emptyEl.hidden = skills.length > 0;
    // A menu may have opened while the primary read was pending. Close it at
    // the actual replacement boundary, while its owner is still connected.
    interactions.beforeReplace?.();
    container.innerHTML = skills.map((skill) => renderInstalledSkillCard(
        skill, reviewingSkills, repairingSkills, snapshot.live, catalogOptions(),
    )).join('');
    catalogSettled.then(() => {
        if (!current()) return;
        // Optional badges never recreate a card, its open menu or edited field.
        const byName = new Map(projected().map((skill) => [skill.name, skill]));
        container.querySelectorAll('[data-skill-hub-badges]').forEach((node) => {
            const skill = byName.get(node.dataset.skillHubBadges);
            if (skill) node.innerHTML = renderSkillHubBadges(skill, catalogOptions());
        });
    }).catch(() => {});
}


async function postWithFeedback(url, body) {
    const resp = await apiFetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
    });
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) {
        throw new Error(payload.error || `HTTP ${resp.status}`);
    }
    return payload;
}

function buildHealPrompt(skill) {
    const findings = Array.isArray(skill.review_findings) ? skill.review_findings : [];
    const diagnostics = {
        name: boundedText(skill.name, 200),
        source: boundedText(skill.source || 'unknown', 80),
        payload_root: boundedText(skill.payload_root || '', 300),
        type: boundedText(skill.type || 'unknown', 80),
        initial_enabled: Boolean(skill.enabled),
        content_hash: skill.content_hash || '',
        review_status: boundedText(skill.review_status || 'pending', 80),
        review_stale: Boolean(skill.review_stale),
        load_error: boundedText(skill.load_error || 'none', 2000),
        review_findings: findings.slice(0, 12).map((finding) => ({
            item: boundedText(finding.item || finding.check || finding.title || 'finding', 200),
            verdict: boundedText(finding.verdict || finding.severity || '', 80),
            reason: boundedText(finding.reason || finding.message || JSON.stringify(finding), 1200),
        })),
    };
    return renderSkillRepairPrompt(
        'Repair and run the installed Ouroboros skill selected in the Skills UI.',
        JSON.stringify(diagnostics, null, 2),
    );
}


function attachActionHandlers(container, renderFn, reviewingSkills, repairingSkills, ctx = {}) {
    let activeMenu = null;
    function closeSkillMenus(options = {}) {
        activeMenu?.binding.close(options);
    }
    function openSkillMenu(trigger) {
        if (activeMenu?.trigger === trigger) {
            closeSkillMenus({ restoreFocus: true });
            return;
        }
        closeSkillMenus();
        const owner = trigger.closest('.skills-card-menu');
        const popover = owner?.querySelector('.skills-card-menu-dialog');
        if (!popover) return;
        popover.classList.add('ui-popup');
        document.body.appendChild(popover);
        popover.show();
        trigger.setAttribute('aria-expanded', 'true');
        popover.addEventListener('click', onClick);
        const binding = bindMenu(popover, { anchor: trigger, onClose: () => {
            popover.removeEventListener('click', onClick);
            popover.close();
            if (owner.isConnected) owner.appendChild(popover);
            else popover.remove();
            trigger.setAttribute('aria-expanded', 'false');
            activeMenu = null;
        } });
        activeMenu = { trigger, binding, popover };
    }

    async function requestMissingKeyGrants(name, items, expectedContentHash = '') {
        const cleanItems = (items || []).map((k) => String(k || '').trim()).filter(Boolean);
        if (!cleanItems.length) return;
        const revision = expectedContentHash || (await fetchSkills()).skills.find((skill) => skill.name === name)?.content_hash;
        if (!revision) throw new Error('Skill revision is unavailable. Refresh and retry.');
        const ok = await openConfirmDialog({
            title: `Grant access to ${name}`,
            body: `Grant access to these keys and permissions for ${name}?\n\n${cleanItems.join('\n')}\n\nOnly grant access to reviewed skills you trust.`,
            confirmLabel: 'Grant access',
        });
        if (!ok) throw Object.assign(new Error('Skill grant cancelled.'), { cancelled: true });
        const bridge = window.pywebview?.api?.request_skill_key_grant;
        const result = bridge
            ? await bridge(name, cleanItems)
            : await apiClient.skillGrants(name, cleanItems, revision);
        if (!result?.ok) {
            throw new Error(result?.error || 'Skill grant was cancelled.');
        }
        return result;
    }

    async function savePresenceRuntime(form, reset = false) {
        const name = String(form?.dataset?.skillName || '');
        const expected = String(form?.dataset?.stateFingerprint || '');
        if (!name || !expected) throw new Error('Presence runtime state is unavailable. Refresh and retry.');
        const modelValue = reset ? '' : String(form.elements.model_slot?.value || '').trim();
        const roundsValue = reset ? '' : String(form.elements.inline_max_rounds?.value || '').trim();
        const rounds = roundsValue === '' ? null : Number(roundsValue);
        if (rounds !== null && (!Number.isInteger(rounds) || rounds < 1)) {
            throw new Error('Inline rounds must be a positive whole number.');
        }
        await apiClient.savePresenceRuntime(name, {
            expected_state_fingerprint: expected,
            runtime_overrides: {
                model_slot: modelValue || null,
                inline_max_rounds: rounds,
            },
            ...(!reset ? { workspace_root: String(form.elements.workspace_root?.value || '').trim() } : {}),
        });
        showToast(`${name}: Presence runtime ${reset ? 'reset' : 'saved'} for new turns`, 'ok');
    }

    async function triggerSkillAction(name, action, options = {}) {
        if (!name || !action) return false;
        if (action === 'open_widgets') {
            document.querySelector('[data-nav-page="widgets"]')?.click();
            return false;
        }
        if (action === 'submit_hub') {
            // The clicked card is the selected-skill identity. It may disappear
            // from the passive inventory after a manifest edit, while the
            // selected preflight can still return an agent-repairable state.
            const outcome = await runSkillPublishFlow(name);
            if (!outcome.started) return false;
            showToast(`${name}: publication task ${outcome.task?.task_id || ''} created`, 'ok');
            emitSkillLifecycle('submit_hub', name);
            if (typeof ctx.showPage === 'function') {
                ctx.showPage('chat');
            } else {
                document.querySelector('[data-nav-page="chat"]')?.click();
            }
            return;
        }
        if (action === 'retry_install') {
            showToast(`${name}: retrying ClawHub install (this may take ~30s)`, 'muted');
            const result = await postWithFeedback('/api/marketplace/clawhub/install', {
                slug: name,
                overwrite: true,
                auto_review: true,
            });
            const tail = result.review_status ? ` — review ${result.review_status}` : '';
            showToast(
                result.ok
                    ? `${name}: install retried${tail}`
                    : `${name}: install retry failed — ${result.error || 'unknown'}`,
                result.ok ? 'ok' : 'danger',
            );
            if (result.ok) emitSkillLifecycle('retry_install', name, result);
            return;
        }

        const { skills } = await fetchSkills();
        const skill = (skills || []).find((item) => item.name === name);
        if (!skill) throw new Error('Skill not found in current catalogue.');

        if (action === 'review' || action === 'rereview') {
            const ok = await openConfirmDialog({
                title: action === 'rereview' ? `Re-review ${name}` : `Review ${name}`,
                body: `Run security review for ${name}? It can take a few minutes and runs in the background.`,
                confirmLabel: action === 'rereview' ? 'Re-review' : 'Run review',
            });
            if (!ok) return false;
            await reviewSkillInBackground(name);
            return;
        }

        if (action === 'grant') {
            const grants = skill.grants || {};
            const keys = (options.keys || '').split(',').map((k) => k.trim()).filter(Boolean);
            const missingKeys = Array.isArray(grants.missing_keys) ? grants.missing_keys : (grants.requested_keys || []);
            const missingPermissions = Array.isArray(grants.missing_permissions) ? grants.missing_permissions : (grants.requested_permissions || []);
            const missing = keys.length ? keys : [...missingKeys, ...missingPermissions];
            const result = await requestMissingKeyGrants(name, missing, skill.content_hash);
            if (result) {
                showToast(`${name}: requested grants saved`, 'ok');
                emitSkillLifecycle('grant', name, result);
            }
            return;
        }

        if (action === 'approve_enable') {
            const grants = skill.grants || {};
            const keys = (options.keys || '').split(',').map((k) => k.trim()).filter(Boolean);
            const missingKeys = Array.isArray(grants.missing_keys) ? grants.missing_keys : (grants.requested_keys || []);
            const missingPermissions = Array.isArray(grants.missing_permissions) ? grants.missing_permissions : (grants.requested_permissions || []);
            const missing = keys.length ? keys : [...missingKeys, ...missingPermissions];
            if (missing.length) await requestMissingKeyGrants(name, missing, skill.content_hash);
            await toggleSkillEnabled(name, true);
            return;
        }

        if (action === 'repair') {
            if (repairingSkills.has(name)) {
                showToast(`${name}: repair is already being queued`, 'muted');
                return false;
            }
            const ok = await openConfirmDialog({
                title: `Repair and run ${name}`,
                body: `Start a repair task for ${name}? Ouroboros will repair the selected skill, review it, enable it when its prerequisites are ready, and test the result. The repaired skill stays running unless you stop or disable it. If the task cannot start, chat will show why.`,
                confirmLabel: 'Repair and run',
                danger: true,
            });
            if (!ok) return false;
            repairingSkills.add(name);
            renderFn();
            try {
                const prompt = buildHealPrompt(skill);
                await postWithFeedback('/api/command', {
                    cmd: prompt,
                    task_constraint: { mode: 'normal', skill_name: skill.name || name, payload_root: skill.payload_root || '', allow_enable: false, allow_review: true },
                    visible_task_id: `skill_repair_${name}`,
                });
                showToast(`${name}: repair request sent to Ouroboros`, 'ok');
                emitSkillLifecycle('repair', name);
                if (typeof ctx.showPage === 'function') {
                    ctx.showPage('chat');
                } else {
                    document.querySelector('[data-nav-page="chat"]')?.click();
                }
            } finally {
                repairingSkills.delete(name);
                renderFn();
            }
            return;
        }

    }

    async function toggleSkillEnabled(name, wantsEnabled) {
        const result = await postWithFeedback(
            `/api/skills/${encodeURIComponent(name)}/toggle`,
            { enabled: wantsEnabled }
        );
        const actionLabels = {
            extension_loaded: 'live',
            extension_unloaded: 'stopped',
            extension_already_live: '',
            extension_inactive: '',
            extension_load_error: 'load failed',
        };
        const friendlyAction = actionLabels[result.extension_action];
        const tail = friendlyAction ? ` — ${friendlyAction}` : '';
        showToast(`${name} ${wantsEnabled ? 'turned on' : 'turned off'}${tail}`, 'ok');
        emitSkillLifecycle(wantsEnabled ? 'enable' : 'disable', name, result);
        return result;
    }

    async function reviewSkillInBackground(name) {
        if (reviewingSkills.has(name)) return null;
        reviewingSkills.add(name);
        renderFn();
        try {
            showToast(`${name}: security review started; this can take a few minutes`, 'muted');
            const result = await postWithFeedback(
                `/api/skills/${encodeURIComponent(name)}/review`,
                {}
            );
            const findings = result.findings?.length ?? 0;
            const errorTail = result.error ? ` — ${result.error}` : '';
            showToast(
                `${name}: review ${result.status}${findings ? ` (${findings} findings)` : ''}${errorTail}`,
                reviewTone(result.status, result.error)
            );
            emitSkillLifecycle('review', name, result);
            return result;
        } finally {
            reviewingSkills.delete(name);
            renderFn();
        }
    }

    async function attestSkillReviewInBackground(name, expectedContentHash) {
        // Owner-attestation: SKIP only the expensive LLM review for the owner's own skill.
        // The deterministic preflight floor still runs server-side (a 409 surfaces here as a
        // thrown error caught by the click handler). Reuses the reviewingSkills lock + spinner.
        if (reviewingSkills.has(name)) return null;
        reviewingSkills.add(name);
        renderFn();
        try {
            showToast(`${name}: skipping LLM review (owner attestation)…`, 'warn');
            const result = await postWithFeedback(
                `/api/owner/skills/${encodeURIComponent(name)}/attest-review`,
                { expected_content_hash: expectedContentHash }
            );
            showToast(`${name}: review skipped — owner-attested`, 'warn');
            emitSkillLifecycle('attest_review', name, result);
            return result;
        } finally {
            reviewingSkills.delete(name);
            renderFn();
        }
    }

    // Checkbox toggle uses change so keyboard and mouse activation match.
    const onChange = async (event) => {
        const target = event.target;
        if (!target || !target.classList || !target.classList.contains('skills-toggle')) {
            return;
        }
        const name = target.dataset.skill;
        if (!name) return;
        const wantsEnabled = Boolean(target.checked);
        let refreshNeeded = true;
        target.disabled = true;
        target.dataset.skillPending = 'true';
        try {
            if (wantsEnabled) {
                let current = (await fetchSkills()).skills.find((skill) => skill.name === name);
                if (!current) throw new Error('Skill not found in current catalogue.');
                if ((current.review_status === 'blockers' && !reviewReady(current)) || (current.load_error && !isMissingGrantLoadError(current))) {
                    throw new Error('Repair this skill before enabling it.');
                }
                if (!reviewReady(current)) {
                    throw new Error('Run review and wait for a fresh executable review before enabling this skill.');
                }
                if (!grantReady(current)) {
                    const grants = current.grants || {};
                    const missingKeys = Array.isArray(grants.missing_keys) ? grants.missing_keys : (grants.requested_keys || []);
                    const missingPermissions = Array.isArray(grants.missing_permissions) ? grants.missing_permissions : (grants.requested_permissions || []);
                    const missing = [...missingKeys, ...missingPermissions];
                    await requestMissingKeyGrants(name, missing, current.content_hash);
                }
            }
            await toggleSkillEnabled(name, wantsEnabled);
            target.setAttribute('aria-checked', wantsEnabled ? 'true' : 'false');
        } catch (err) {
            if (err.cancelled) refreshNeeded = false;
            // Roll back to server-truth state on failed enable/disable.
            target.checked = !wantsEnabled;
            target.setAttribute('aria-checked', (!wantsEnabled).toString());
            showToast(`${name}: ${err.message || err}`, (err.message || '').includes('cancel') ? 'warn' : 'danger');
        } finally {
            target.disabled = false;
            delete target.dataset.skillPending;
            if (refreshNeeded) renderFn();
        }
    };
    const onKeydown = (event) => {
        const actionTarget = event.target.closest?.('[data-skill-action]');
        if (!actionTarget) return;
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        actionTarget.click();
    };
    const onSubmit = async (event) => {
        const form = event.target.closest?.('[data-presence-runtime-form]');
        if (!form) return;
        event.preventDefault();
        const submit = form.querySelector('button[type="submit"]');
        if (submit) submit.disabled = true;
        try {
            await savePresenceRuntime(form);
        } catch (err) {
            showToast(`${form.dataset.skillName || 'Skill'}: ${err.message || err}`, 'danger');
        } finally {
            if (submit) submit.disabled = false;
            renderFn();
        }
    };
    const onClick = async (event) => {
        // A portalled menu keeps this same action dispatcher. Restore to the
        // trigger before a menu action opens its own confirm/input dialog.
        if (event.target.closest('.skills-menu-item')) closeSkillMenus({ restoreFocus: true });
        const resetRuntime = event.target.closest('[data-presence-runtime-reset]');
        if (resetRuntime) {
            const form = resetRuntime.closest('[data-presence-runtime-form]');
            resetRuntime.disabled = true;
            try {
                await savePresenceRuntime(form, true);
            } catch (err) {
                showToast(`${form?.dataset?.skillName || 'Skill'}: ${err.message || err}`, 'danger');
            } finally {
                resetRuntime.disabled = false;
                renderFn();
            }
            return;
        }
        const menuTrigger = event.target.closest('[data-skill-menu-trigger]');
        if (menuTrigger) {
            openSkillMenu(menuTrigger);
            return;
        }
        if (event.target.closest('[data-skill-menu-close]')) {
            closeSkillMenus();
            return;
        }
        const actionTarget = event.target.closest('[data-skill-action]');
        if (actionTarget) {
            const name = actionTarget.dataset.skill;
            const action = actionTarget.dataset.skillAction;
            if (action === 'repair' && repairingSkills.has(name)) {
                return;
            }
            actionTarget.disabled = true;
            let refreshNeeded = true;
            try {
                refreshNeeded = await triggerSkillAction(name, action, { keys: actionTarget.dataset.keys || '' }) !== false;
            } catch (err) {
                if (err.cancelled) refreshNeeded = false;
                showToast(`${name}: ${err.message || err}`, (err.message || '').includes('cancel') ? 'warn' : 'danger');
            } finally {
                actionTarget.disabled = false;
                if (refreshNeeded) renderFn();
            }
            return;
        }
        const target = event.target.closest('button[data-skill]');
        if (!target) return;
        if (target.classList.contains('skills-toggle')) {
            // Checkbox handler above owns current toggles; ignore legacy buttons.
            return;
        }
        const name = target.dataset.skill;
        if (target.classList.contains('skills-review')) {
            if (reviewingSkills.has(name)) return;
            target.disabled = true;
            try {
                await reviewSkillInBackground(name);
            } catch (err) {
                showToast(`${name}: ${err.message || err}`, 'danger');
            } finally {
                target.disabled = false;
                renderFn();
            }
            return;
        }
        if (target.classList.contains('skills-attest-review')) {
            if (reviewingSkills.has(name)) return;
            let current;
            try {
                current = (await fetchSkills()).skills.find((skill) => skill.name === name);
            } catch (err) {
                showToast(`${name}: ${err.message || err}`, 'danger');
                return;
            }
            if (!current?.content_hash) {
                showToast(`${name}: skill revision unavailable; refresh and retry`, 'warn');
                return;
            }
            const ok = await openConfirmDialog({
                title: `Skip review for ${name}`,
                body: `Skip the expensive LLM security review for ${name}? The deterministic safety preflight still runs and refuses an unsafe or invalid skill. Owner-attestation is logged for audit — only skip review for a skill you authored or fully trust.`,
                confirmLabel: 'Skip review',
                danger: true,
            });
            if (!ok) return;
            target.disabled = true;
            try {
                await attestSkillReviewInBackground(name, current.content_hash);
            } catch (err) {
                showToast(`${name}: ${err.message || err}`, 'danger');
            } finally {
                target.disabled = false;
                renderFn();
            }
            return;
        }
        target.disabled = true;
        let refreshNeeded = true;
        try {
            if (target.classList.contains('skills-next-toggle')) {
                const wantsEnabled = target.dataset.enabled === 'true';
                await toggleSkillEnabled(name, wantsEnabled);
            } else if (target.classList.contains('skills-grant')) {
                const keys = (target.dataset.keys || '').split(',').map((k) => k.trim()).filter(Boolean);
                if (!keys.length) {
                    showToast(`${name}: no requested keys or permissions to grant`, 'warn');
                } else {
                    const result = await requestMissingKeyGrants(name, keys);
                    // Grant may persist even if live extension reconcile fails.
                    const reason = result.extension_reason;
                    const action = result.extension_action;
                    const loadError = result.load_error;
                    if (reason === 'reconcile_call_failed') {
                        showToast(
                            `${name}: grant saved, but server reconcile failed \u2014 toggle disable/enable to retry`,
                            'warn'
                        );
                    } else if (loadError) {
                        showToast(
                            `${name}: grant saved, but extension load failed: ${loadError}`,
                            'warn'
                        );
                    } else if (action === 'extension_loaded') {
                        showToast(`${name}: grant saved and extension loaded`, 'ok');
                    } else {
                        showToast(`${name}: requested grants saved`, 'ok');
                    }
                }
            } else if (target.classList.contains('skills-update')) {
                const source = target.dataset.source === 'ouroboroshub' ? 'ouroboroshub' : 'clawhub';
                showToast(`${name}: updating from ${source === 'ouroboroshub' ? 'OuroborosHub' : 'ClawHub'} (this may take ~30s)`, 'muted');
                const url = source === 'ouroboroshub'
                    ? `/api/marketplace/ouroboroshub/update/${encodeURIComponent(name)}`
                    : `/api/marketplace/clawhub/update/${encodeURIComponent(name)}`;
                const body = {};
                const result = await postWithFeedback(url, body);
                const tail = result.review_status ? ` — review ${result.review_status}` : '';
                showToast(
                    result.ok
                        ? `${name}: updated${tail}`
                        : `${name}: update failed — ${result.error || 'unknown'}`,
                    result.ok ? 'ok' : 'danger',
                );
            } else if (target.classList.contains('skills-submit-hub')) {
                if (target.dataset.submitDisabled === 'true') {
                    showToast(`${name}: submit disabled — ${target.dataset.submitReason || 'unknown reason'}`, 'warn');
                    refreshNeeded = false;
                    return;
                }
                refreshNeeded = await triggerSkillAction(name, 'submit_hub') !== false;
            } else if (target.classList.contains('skills-uninstall')) {
                const source = target.dataset.source === 'ouroboroshub' ? 'ouroboroshub' : 'clawhub';
                const ok = await openConfirmDialog({
                    title: `Uninstall ${name}`,
                    body: `Uninstall ${name}? This deletes data/skills/${source}/${name}/.`,
                    confirmLabel: 'Uninstall',
                    danger: true,
                });
                if (!ok) {
                    refreshNeeded = false;
                    return;
                }
                const url = source === 'ouroboroshub'
                    ? `/api/marketplace/ouroboroshub/uninstall/${encodeURIComponent(name)}`
                    : `/api/marketplace/clawhub/uninstall/${encodeURIComponent(name)}`;
                const result = await postWithFeedback(url, {});
                showToast(
                    result.ok ? `${name}: uninstalled` : `${name}: uninstall failed — ${result.error}`,
                    result.ok ? 'ok' : 'danger',
                );
                if (result.ok) emitSkillLifecycle('uninstall', name, result);
            } else if (target.classList.contains('skills-delete-local')) {
                const payloadRoot = target.dataset.payloadRoot || `skills/external/${name}`;
                const current = (await fetchSkills()).skills.find((skill) => skill.name === name);
                if (!current?.content_hash) throw new Error('Skill revision is unavailable. Refresh and retry.');
                const ok = await openConfirmDialog({
                    title: `Delete ${name}`,
                    body: `Delete ${name}? This deletes data/${payloadRoot}/ and data/state/skills/${name}/.`,
                    confirmLabel: 'Delete',
                    danger: true,
                });
                if (!ok) {
                    refreshNeeded = false;
                    return;
                }
                const result = await apiClient.deleteSkill(name, payloadRoot, current.content_hash);
                showToast(
                    result.ok ? `${name}: deleted` : `${name}: delete failed — ${result.error}`,
                    result.ok ? 'ok' : 'danger',
                );
                if (result.ok) emitSkillLifecycle('delete', name, result);
            }
        } catch (err) {
            if (err.cancelled) refreshNeeded = false;
            showToast(`${name}: ${err.message || err}`, 'danger');
        } finally {
            target.disabled = false;
            closeSkillMenus();
            if (refreshNeeded) renderFn();
        }
    };
    const handlers = [['change', onChange], ['keydown', onKeydown], ['submit', onSubmit], ['click', onClick]];
    handlers.forEach(([type, handler]) => container.addEventListener(type, handler));
    return { closeMenus: closeSkillMenus,
        menuFor: card => card.contains(activeMenu?.trigger) ? activeMenu.popover : null,
        beforeReplace(card) { if (!card || card.contains(activeMenu?.trigger)) closeSkillMenus(); },
        destroy() {
        closeSkillMenus();
        handlers.forEach(([type, handler]) => container.removeEventListener(type, handler));
    } };
}


function activateTab(tabName) {
    const panels = document.querySelectorAll('.skills-tab-panel');
    const chromeRows = document.querySelectorAll('.skills-search-chrome');
    panels.forEach((panel) => {
        panel.hidden = panel.dataset.pane !== tabName;
    });
    chromeRows.forEach((row) => {
        row.hidden = row.dataset.chromePane !== tabName;
    });
}


async function renderMarketplacePane() {
    const pane = document.getElementById('skills-pane-marketplace');
    if (!pane) return;
    if (pane.dataset.bootstrapped === 'true') {
        // Tab entry refreshes installed state without simulating Search.
        if (typeof pane._marketplaceRefresh === 'function') {
            return pane._marketplaceRefresh();
        }
        return;
    }
    pane.innerHTML = '<div class="muted">Loading marketplace…</div>';
    try {
        const loading = initMarketplace(pane, document.getElementById('skills-pane-marketplace-chrome'));
        pane.dataset.bootstrapped = 'true';
        await loading;
    } catch (err) {
        pane.dataset.bootstrapped = '';
        pane.innerHTML = `<div class="skills-load-error">Failed to load marketplace UI: ${escapeHtml(err.message || err)}</div>`;
    }
}


async function renderOuroborosHubPane() {
    const pane = document.getElementById('skills-pane-ouroboroshub');
    if (!pane) return;
    if (pane.dataset.bootstrapped === 'true') {
        if (typeof pane._ouroboroshubRefresh === 'function') {
            return pane._ouroboroshubRefresh();
        }
        return;
    }
    pane.innerHTML = '<div class="muted">Loading OuroborosHub…</div>';
    try {
        const loading = initOuroborosHub(pane, document.getElementById('skills-pane-ouroboroshub-chrome'));
        pane.dataset.bootstrapped = 'true';
        await loading;
    } catch (err) {
        pane.dataset.bootstrapped = '';
        pane.innerHTML = `<div class="skills-load-error">Failed to load OuroborosHub UI: ${escapeHtml(err.message || err)}</div>`;
    }
}


export function initSkills(ctx) {
    const page = document.createElement('div');
    page.innerHTML = skillsPageTemplate();
    document.getElementById('content').appendChild(page.firstElementChild);

    const container = document.getElementById('skills-list');
    const emptyEl = document.getElementById('skills-empty');
    const refreshBtn = document.getElementById('skills-refresh');
    const reviewingSkills = new Set();
    const repairingSkills = new Set();
    let activeTab = 'installed';
    let refreshGeneration = 0;
    let actions;
    let destroyed = false;

    const renderFn = () => {
        if (destroyed) return;
        return renderSkillsList(container, emptyEl, reviewingSkills, repairingSkills, actions);
    };
    const refreshActive = async () => {
        if (destroyed) return;
        const generation = ++refreshGeneration;
        refreshBtn.disabled = true;
        refreshBtn.classList.add('is-loading');
        const originalText = refreshBtn.textContent || 'Refresh';
        refreshBtn.textContent = 'Refreshing';
        try {
            await Promise.all([
                activeTab === 'marketplace' ? renderMarketplacePane()
                    : activeTab === 'ouroboroshub' ? renderOuroborosHubPane() : renderFn(),
                new Promise((resolve) => setTimeout(resolve, 250)),
            ]);
        } catch (err) {
            if (generation === refreshGeneration) showToast(`Skills view failed: ${err.message || err}`, 'danger');
            console.warn('skills: render failed', err);
        } finally {
            if (generation !== refreshGeneration) return;
            refreshBtn.disabled = false;
            refreshBtn.classList.remove('is-loading');
            refreshBtn.textContent = originalText === 'Refreshing' ? 'Refresh' : originalText;
        }
    };

    refreshBtn.addEventListener('click', refreshActive);
    actions = attachActionHandlers(container, renderFn, reviewingSkills, repairingSkills, ctx);

    const tabs = bindTabStrip(document.querySelector('#page-skills .skills-tabs'), {
        dataAttr: 'data-tab', activeClass: 'is-active',
        onChange(tabName) {
            actions.closeMenus();
            activeTab = tabName;
            activateTab(tabName);
            refreshActive();
        },
    });

    const onPageShown = (event) => {
        actions.closeMenus();
        if (event.detail?.page === 'skills') {
            // Fresh catalog snapshot once per page open; re-renders reuse it.
            tabs.select(activeTab);
            loadHubCatalog(true);
            refreshActive();
        }
    };
    window.addEventListener('ouro:page-shown', onPageShown);
    return { destroy() {
        destroyed = true;
        refreshGeneration += 1;
        skillsRenderGeneration += 1;
        actions.destroy();
        document.getElementById('skills-pane-marketplace')?._marketplaceDestroy?.();
        document.getElementById('skills-pane-ouroboroshub')?._ouroboroshubDestroy?.();
        tabs.destroy();
        refreshBtn.removeEventListener('click', refreshActive);
        window.removeEventListener('ouro:page-shown', onPageShown);
    } };
}
