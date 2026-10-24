// The Skills page: three views of one page — Installed (what Ouroboros can do
// now), Browse (what else it could do, from every connected source) and Sources
// (where skills come from and how much each place is trusted). Installed is the
// default and never waits for a source or GitHub: it reads the local listing and
// the serialized lifecycle queue, and every owner action keeps its own words,
// confirmation and refusal. Browse and Sources live in their own modules.
import { bindPageHeader, bindTabStrip, renderPageHeader, renderTabStrip } from './page_header.js';
import { bindMenu } from './ui_interactions.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { showToast } from './toast.js';
import { apiClient, apiFetch, skillSourcesApi } from './api_client.js';
import {
    fillFindingsOnOpen, installedSkillModel, installedSourceLabel, isMadeHere, patchInstalledSkillEnrichment,
    renderInstalledSkillCard,
} from './skill_card_renderer.js';
import { createSkillBrowse } from './skill_browse.js';
import { createSkillSources, unconfirmedHubRows } from './skill_sources_view.js';
import { openSkillDetailSheet } from './skill_detail_sheet.js';
import {
    openCompareDialog, openOfficialExplanation, openPublishDestinationDialog, openSkipReviewDialog, openUpdateDialog, updateWithoutReview,
} from './skill_dialogs.js';
import { connectsServices, plural, popularityText, repoLabel, routeExternalLink, versionText } from './skill_presentation.js';
import { openConnectorPlaces } from './place_sheet.js';
import { runSkillPublishFlow } from './skill_publish_flow.js';
import { installedTime } from './ui_helpers.js';
import {
    boundedText, emitSkillLifecycle, escapeHtmlAttr as escapeHtml, grantReady, renderSkillRepairPrompt,
    reviewTone, reviewReady,
} from './utils.js';

const SKILLS_TABS = [
    { value: 'installed', label: 'Installed', pillId: 'skills-tab-pill-installed', tabId: 'skills-tab-installed', panelId: 'skills-pane-installed' },
    { value: 'browse', label: 'Browse', pillId: 'skills-tab-pill-browse', tabId: 'skills-tab-browse', panelId: 'skills-pane-browse' },
    { value: 'sources', label: 'Sources', tabId: 'skills-tab-sources', panelId: 'skills-pane-sources' },
];
const LIFECYCLE_VISIBLE_STATUSES = new Set(['queued', 'running', 'failed']);
const INSTALLED_FILTERS = [
    ['all', 'All'], ['needs', 'Needs you'], ['updates', 'Updates'], ['off', 'Off'], ['made', 'Made here'],
];
const WORK_POLL_MS = 3000;
const REVIEW_POLL_MS = 15000;

function skillsPageTemplate() {
    return `
        <section class="page app-page-glass" id="page-skills">
            ${renderPageHeader({
                page: 'skills',
                actionsHtml: '<button id="skills-update-all" class="btn btn-default btn-sm" hidden>Update all</button><button id="skills-refresh" class="btn btn-default btn-sm">Refresh</button>',
                tabsHtml: renderTabStrip({
                    items: SKILLS_TABS, active: 'installed', dataAttr: 'data-tab', activeClass: 'is-active',
                    ariaLabel: 'Skills views', stripClass: 'skills-tabs', tabClass: 'skills-tab',
                }),
            })}
            <div class="skills-search-chrome" data-chrome-pane="installed">
                <input type="search" class="ui-control skills-search" id="skills-installed-search" placeholder="Find an installed skill" aria-label="Find an installed skill" autocomplete="off">
                <div class="skills-filter-strip" id="skills-installed-filters" role="group" aria-label="Show"></div>
            </div>
            <div class="skills-search-chrome" data-chrome-pane="browse" id="skills-browse-chrome" hidden></div>
            <div class="skills-scroll scroll-fade-y">
                <div class="skills-tab-panel" id="skills-pane-installed" data-pane="installed" role="tabpanel" aria-labelledby="skills-tab-installed">
                    <div id="skills-status" class="ui-status skills-list-status" role="status" aria-live="polite"></div>
                    <div id="skills-list" class="skills-list"></div>
                    <p id="skills-filter-empty" class="skills-empty" hidden>No installed skill matches.</p>
                    <div id="skills-empty" class="skills-empty" hidden>
                        <p>No skills yet. Pick one in Browse, or ask Ouroboros: “find a skill for …”.</p>
                        <button type="button" class="btn btn-primary btn-sm" data-skills-goto="browse">Browse skills</button>
                    </div>
                </div>
                <div class="skills-tab-panel" id="skills-pane-browse" data-pane="browse" role="tabpanel" aria-labelledby="skills-tab-browse" hidden></div>
                <div class="skills-tab-panel" id="skills-pane-sources" data-pane="sources" role="tabpanel" aria-labelledby="skills-tab-sources" hidden></div>
            </div>
        </section>
    `;
}

export function lifecycleEventsFromQueue(queueResp) {
    const events = Array.isArray(queueResp?.events) ? queueResp.events : [];
    const active = queueResp?.active;
    if (!active || typeof active !== 'object') return events;
    const activeId = String(active.id || '');
    const deduped = activeId
        ? events.filter((event) => String(event?.id || '') !== activeId)
        : events;
    return [...deduped, active];
}

export function mergeLifecycleEvents(skills, events) {
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
            // The skill already has a real card: annotate it with the in-flight
            // transition so it shows "Turning off…" instead of a stale toggle
            // while the serialized lifecycle lane works through it.
            const existing = byName.get(name);
            if (existing) {
                existing.lifecycle_status = event.status;
                existing.lifecycle_kind = event.kind || existing.lifecycle_kind || '';
                existing.lifecycle_message = event.message || '';
                existing.lifecycle_pending = event.status !== 'failed';
                if (event.status === 'failed' && event.error) existing.lifecycle_error = event.error;
            }
            continue;
        }
        names.add(name);
        out.unshift({
            name,
            description: event.message || event.error || 'Skill lifecycle operation',
            version: '',
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

/** In-flight lifecycle work, newest event per skill (the Installed tab pill). */
export function inFlightCount(events) {
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
    return [...latestByTarget.values(), ...untargeted]
        .filter((event) => event.status === 'queued' || event.status === 'running').length;
}

function sortSkillsForDisplay(skills) {
    return [...skills].sort((a, b) => {
        if (a.lifecycle_virtual && !b.lifecycle_virtual) return -1;
        if (!a.lifecycle_virtual && b.lifecycle_virtual) return 1;
        return installedTime(b) - installedTime(a) || String(a.name || '').localeCompare(String(b.name || ''));
    });
}

/** Does one installed row belong in a filter of the Installed strip? */
export function matchesInstalledFilter(skill, filter, model = installedSkillModel(skill)) {
    if (filter === 'needs') return model.needs;
    if (filter === 'updates') return model.update;
    if (filter === 'off') return !skill.enabled && !skill.lifecycle_virtual;
    if (filter === 'made') return isMadeHere(skill);
    return true;
}

/** «All 62 · Needs you 10 · Updates 12 · Off 41 · Made here 43», as facts of the loaded list. */
export function installedFilterCounts(skills) {
    const counts = Object.fromEntries(INSTALLED_FILTERS.map(([key]) => [key, 0]));
    for (const skill of skills) {
        const model = installedSkillModel(skill);
        for (const [key] of INSTALLED_FILTERS) if (matchesInstalledFilter(skill, key, model)) counts[key] += 1;
    }
    return counts;
}

/** «Update all» covers unmodified skills from trusted rows only (SPEC §7). */
export function updateAllCandidates(skills, sources = null) {
    const trust = new Map((sources || []).map((source) => [source.id, source.trust]));
    return skills.filter((skill) => {
        const install = skill.install;
        if (!install?.update_available?.digest || install.modified || install.base_known === false || install.withdrawn || skill.lifecycle_pending) return false;
        if (install.basis !== 'trusted') return false;
        const rowTrust = trust.get(install.source?.id);
        return rowTrust === undefined || rowTrust === 'trusted';
    });
}

/**
 * Where a skill can be published (B10): back to where it came from (a hub entry
 * that points at its author's repository goes there), a git source the owner
 * added, or the public hub — its shared catalog, or, for a skill that came from
 * the hub, a new entry beside the original.
 */
export function publishDestinations(skill, sources = []) {
    const out = [];
    const origin = skill.install?.source;
    const originRow = sources.find((source) => source.id === origin?.id);
    const originKind = originRow?.kind || origin?.kind;
    const authorRepo = String(skill.install?.release?.repo || '');
    if (origin?.id && ['hub', 'git'].includes(originKind)) {
        out.push(authorRepo
            ? { value: `origin:${origin.id}`, repository: repoLabel(authorRepo), label: `Back to ${repoLabel(authorRepo)} — its author’s repository`,
                detail: 'Suggest your changes to its author as a new version. The author decides.' }
            : { value: `origin:${origin.id}`, repository: repoLabel(originRow?.url || origin.url), label: `Back to ${origin.name || originRow?.name} — where it came from`,
                detail: 'Suggest your changes as a new version. The maintainers decide.' });
    }
    for (const source of sources.filter((row) => row.kind === 'git')) {
        if (source.id === origin?.id) continue;
        out.push({ value: `source:${source.id}`, repository: repoLabel(source.url), label: `${source.name} — your own source`,
            detail: origin ? 'Publish your version now, under the same name, for your own installs.' : `Publish now. Every Ouroboros that reads ${source.name} gets it at once.` });
    }
    // The hub's two rows are one repository: a skill from it goes back as its origin, or beside it as a new skill.
    const community = sources.find((row) => row.kind === 'hub' && row.part === 'community');
    if (community && originKind === 'hub') {
        out.push({ value: `new:${community.id}`, repository: repoLabel(community.url), label: `${community.name} — as a new skill`,
            detail: 'Ask to list it next to the original under another name.' });
    } else if (community) {
        out.push({ value: `source:${community.id}`, repository: repoLabel(community.url), label: `${community.name} — share with everyone`,
            detail: 'Ask to list it in the shared catalog. New skills and new authors are read by the maintainer first.' });
    }
    return out;
}

/** A chosen destination as the publish preflight and task carry it: `{source_id, mode}`. */
export function destinationFromChoice(value) {
    const [mode, sourceId] = String(value || '').split(':');
    if (!sourceId) return null;
    return { source_id: sourceId, mode: mode === 'origin' ? 'origin' : mode === 'new' ? 'new_entry' : 'entry' };
}

export function buildHealPrompt(skill) {
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

async function postWithFeedback(url, body) {
    const resp = await apiFetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Prefer: 'respond-async' },
        body: JSON.stringify(body || {}),
    });
    const payload = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(payload.error || `HTTP ${resp.status}`);
    return payload;
}

async function fetchSkills() {
    const extResp = await apiClient.extensions();
    if (!Array.isArray(extResp?.skills)) throw new Error('Installed skills response is unavailable.');
    return { skills: extResp.skills, live: extResp.live || {} };
}

export function initSkills(ctx = {}) {
    const page = document.createElement('div');
    page.innerHTML = skillsPageTemplate();
    document.getElementById('content').appendChild(page.firstElementChild);

    const root = document.getElementById('page-skills');
    const container = document.getElementById('skills-list');
    const status = document.getElementById('skills-status');
    const emptyEl = document.getElementById('skills-empty');
    const filterEmpty = document.getElementById('skills-filter-empty');
    const filtersEl = document.getElementById('skills-installed-filters');
    const searchInput = document.getElementById('skills-installed-search');
    const refreshBtn = document.getElementById('skills-refresh');
    const updateAllBtn = document.getElementById('skills-update-all');
    const reviewingSkills = new Set();
    const repairingSkills = new Set();
    let activeTab = 'installed';
    let visible = false;
    let destroyed = false;
    let renderGeneration = 0;
    let filter = 'all';
    let snapshot = null;   // { rawSkills, live, queue, githubTokenConfigured }
    let activeMenu = null;
    let sheet = null;
    let pollTimer = 0;
    let lastListingRead = 0;
    let firstStartAsked = false;

    // ---- shared reads (Browse and Sources reuse them) -------------------------------------------
    const cache = { sources: null, sourcesValue: null, sourcesProblem: '', catalog: null, catalogValue: null };
    const store = {
        sources(force = false) {
            if (force || !cache.sources) {
                cache.sources = skillSourcesApi.list().then((data) => {
                    const list = Array.isArray(data) ? data : (Array.isArray(data?.sources) ? data.sources : []);
                    cache.sourcesValue = list;
                    cache.sourcesProblem = String(data?.problem || '');
                    return list;
                }).catch((error) => { cache.sources = null; throw error; });
            }
            return cache.sources;
        },
        catalog({ refresh = false } = {}) {
            if (refresh || !cache.catalog) {
                cache.catalog = skillSourcesApi.catalog({ refresh }).then((data) => { cache.catalogValue = data; return data; })
                    .finally(() => { setTimeout(() => { cache.catalog = null; }, 0); });
            }
            return cache.catalog;
        },
    };
    const projected = () => (snapshot?.rawSkills
        ? sortSkillsForDisplay(mergeLifecycleEvents(snapshot.rawSkills, lifecycleEventsFromQueue(snapshot.queue))) : []);
    const installedByName = () => (snapshot?.rawSkills ? new Map(projected().map((skill) => [skill.name, skill])) : null);
    const officialSource = () => (cache.sourcesValue || []).find((source) => source.kind === 'hub' && source.part === 'official') || null;
    // The catalog entry an installed skill came from (its author, full text), when the catalog was read.
    const catalogEntry = (skill) => (cache.catalogValue?.items || [])
        .find((item) => item.source_id === skill?.install?.source?.id && item.name === skill?.name) || null;
    const cardOptions = () => ({ githubTokenConfigured: snapshot?.githubTokenConfigured });

    // ---- Installed view -------------------------------------------------------------------------
    function paintFilters(skills) {
        const counts = skills ? installedFilterCounts(skills) : null;
        filtersEl.innerHTML = INSTALLED_FILTERS.map(([key, label]) => {
            const count = counts ? counts[key] : null;
            const hide = key !== 'all' && count === 0 && filter !== key;
            return hide ? '' : `<button type="button" class="filter-chip skills-category${key === filter ? ' active' : ''}" aria-pressed="${key === filter ? 'true' : 'false'}" data-installed-filter="${key}">${escapeHtml(label)}${count === null ? '' : `<span class="skills-category-count">${count}</span>`}</button>`;
        }).join('');
    }

    function applyInstalledFilter() {
        const query = searchInput.value.trim().toLowerCase();
        const byName = installedByName() || new Map();
        let shown = 0;
        for (const card of container.querySelectorAll(':scope > .skills-card')) {
            const skill = byName.get(card.dataset.skill);
            const ok = !skill || (matchesInstalledFilter(skill, filter)
                && (!query || [skill.name, skill.description].some((text) => String(text || '').toLowerCase().includes(query))));
            card.hidden = !ok;
            if (ok) shown += 1;
        }
        const total = container.querySelectorAll(':scope > .skills-card').length;
        filterEmpty.hidden = !(total && !shown);
    }

    function paintChrome() {
        const skills = snapshot?.rawSkills ? projected() : null;
        paintFilters(skills);
        const candidates = skills ? updateAllCandidates(skills, cache.sourcesValue) : [];
        updateAllBtn.hidden = !(activeTab === 'installed' && candidates.length >= 2);
        updateAllBtn.textContent = `Update all (${candidates.length})`;
        const pill = document.getElementById('skills-tab-pill-installed');
        const count = inFlightCount(lifecycleEventsFromQueue(snapshot?.queue));
        if (pill) { pill.hidden = !count; pill.textContent = count ? String(count) : ''; }
    }

    function paintInstalled({ replace = false } = {}) {
        if (!snapshot?.rawSkills || destroyed) return;
        const skills = projected();
        const cards = new Map(Array.from(container.querySelectorAll(':scope > .skills-card')).map((card) => [card.dataset.skill, card]));
        let replacementFocus = null;
        if (replace) {
            closeMenus();
            container.innerHTML = skills.map((skill) => renderInstalledSkillCard(skill, reviewingSkills, repairingSkills, snapshot.live, cardOptions())).join('');
        } else {
            for (const skill of [...skills].reverse()) {
                const card = cards.get(skill.name);
                if (card) {
                    // A queue placeholder has no controls to enrich. Replace only this
                    // structural transition; real cards retain drafts and disclosures.
                    if (card.hasAttribute('data-lifecycle-virtual') !== Boolean(skill.lifecycle_virtual)) {
                        const ownsMenu = activeMenu?.card === card;
                        const focused = card.contains(document.activeElement)
                            || (ownsMenu && activeMenu.popover.contains(document.activeElement));
                        if (ownsMenu) closeMenus();
                        const template = document.createElement('template');
                        template.innerHTML = renderInstalledSkillCard(skill, reviewingSkills, repairingSkills, snapshot.live, cardOptions());
                        const fresh = template.content.firstElementChild;
                        card.replaceWith(fresh);
                        if (focused) replacementFocus = fresh;
                    } else {
                        patchInstalledSkillEnrichment(card, skill, reviewingSkills, repairingSkills, snapshot.live, cardOptions(),
                            activeMenu?.card === card ? activeMenu.popover : card);
                    }
                    cards.delete(skill.name);
                } else {
                    const template = document.createElement('template');
                    template.innerHTML = renderInstalledSkillCard(skill, reviewingSkills, repairingSkills, snapshot.live, cardOptions());
                    container.insertBefore(template.content.firstElementChild, container.firstElementChild);
                }
            }
            for (const card of cards.values()) {
                if (activeMenu?.card === card) closeMenus();
                card.remove();
            }
        }
        emptyEl.hidden = skills.length > 0;
        paintChrome();
        applyInstalledFilter();
        if (replacementFocus && !replacementFocus.hidden) {
            replacementFocus.querySelector('.skills-card-name')?.focus({ preventScroll: true });
        }
        browse?.repaintStates();
        schedulePoll();
    }

    async function readInstalled({ quiet = false } = {}) {
        if (destroyed) return;
        const generation = ++renderGeneration;
        const current = () => generation === renderGeneration && !destroyed;
        const next = { rawSkills: null, live: {}, queue: snapshot?.queue || null, githubTokenConfigured: snapshot?.githubTokenConfigured };
        const optional = { state: 'loading', queue: 'loading' };
        const showReads = () => {
            if (!current() || !snapshot?.rawSkills) return;
            const notes = Object.entries(optional).filter(([, value]) => value === 'failed')
                .map(([name]) => (name === 'state' ? 'Skill settings' : 'Lifecycle progress'));
            status.dataset.tone = notes.length ? 'warn' : 'muted';
            status.textContent = notes.length ? `${notes.join(' and ')} could not be refreshed. Refresh to retry; earlier facts are kept.` : '';
        };
        apiClient.state().then((data) => {
            if (!current()) return;
            const ok = typeof data?.github_token_configured === 'boolean';
            optional.state = ok ? 'ready' : 'failed';
            if (ok) next.githubTokenConfigured = data.github_token_configured;
            if (snapshot === next) paintInstalled();
            showReads();
        }, () => { if (current()) { optional.state = 'failed'; showReads(); } });
        apiClient.skillLifecycleQueue().then((data) => {
            if (!current()) return;
            const ok = Array.isArray(data?.events) || (data?.active && typeof data.active === 'object');
            optional.queue = ok ? 'ready' : 'failed';
            if (ok) next.queue = data;
            if (snapshot === next) paintInstalled();
            showReads();
        }, () => { if (current()) { optional.queue = 'failed'; showReads(); } });
        if (!quiet && !snapshot) { status.dataset.tone = 'muted'; status.textContent = 'Loading installed skills…'; }
        try {
            const primary = await fetchSkills();
            if (!current()) return;
            next.rawSkills = primary.skills;
            next.live = primary.live;
            lastListingRead = Date.now();
        } catch (err) {
            if (!current()) return;
            emptyEl.hidden = true;
            status.dataset.tone = 'danger';
            status.textContent = `Couldn’t refresh installed skills: ${err.message || err}.${snapshot ? ' Showing the previous list. Refresh to try again.' : ' Refresh to try again.'}`;
            return;
        }
        const firstPaint = !snapshot;
        snapshot = next;
        if (!quiet && !firstPaint) status.textContent = '';
        if (firstPaint) status.textContent = '';
        paintInstalled({ replace: firstPaint });
        showReads();
    }

    // Local lifecycle progress only (never GitHub): the queue while work runs, the
    // listing now and then while a review runs outside the lane.
    function schedulePoll() {
        clearTimeout(pollTimer);
        if (destroyed || !visible || !snapshot) return;
        const working = inFlightCount(lifecycleEventsFromQueue(snapshot.queue)) > 0;
        const reviewing = reviewingSkills.size > 0 || projected().some((skill) => /Reviewing/.test(installedSkillModel(skill).chip));
        if (!working && !reviewing) return;
        pollTimer = setTimeout(async () => {
            if (destroyed || !visible) return;
            if (Date.now() - lastListingRead >= REVIEW_POLL_MS || working) await readInstalled({ quiet: true });
            else schedulePoll();
        }, WORK_POLL_MS);
    }

    // ---- menus and the Official mark ------------------------------------------------------------
    function closeMenus(options = {}) { activeMenu?.binding.close(options); }

    function openSkillMenu(trigger) {
        if (activeMenu?.trigger === trigger) { closeMenus({ restoreFocus: true }); return; }
        closeMenus();
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
        activeMenu = { trigger, binding, popover, card: trigger.closest('.skills-card') };
    }

    // ---- owner actions on installed skills ------------------------------------------------------
    async function currentSkill(name) {
        const { skills } = await fetchSkills();
        const skill = skills.find((item) => item.name === name);
        if (!skill) throw new Error('Skill not found in the current list. Refresh and try again.');
        return skill;
    }

    async function requestMissingKeyGrants(name, items, expectedContentHash = '', enableAfterGrant = false) {
        const cleanItems = (items || []).map((k) => String(k || '').trim()).filter(Boolean);
        if (!cleanItems.length) return null;
        const revision = expectedContentHash || (await currentSkill(name)).content_hash;
        if (!revision) throw new Error('Skill revision is unavailable. Refresh and retry.');
        const ok = await openConfirmDialog({
            title: `Grant access to ${name}`,
            body: `Let ${name} use these keys and permissions?\n\n${cleanItems.join('\n')}\n\nOnly grant access to skills you trust.`,
            confirmLabel: 'Grant access',
        });
        if (!ok) throw Object.assign(new Error('Skill grant cancelled.'), { cancelled: true });
        // The desktop shell asks once more in a native dialog it owns.
        const nativeApi = window.pywebview?.api;
        // A named bridge capability opts in; older UI calling the old method still gets a terminal result.
        const result = nativeApi?.request_skill_key_grant_with_job
            ? await nativeApi.request_skill_key_grant_with_job(name, cleanItems, enableAfterGrant)
            : nativeApi?.request_skill_key_grant
                ? await nativeApi.request_skill_key_grant(name, cleanItems)
                : await apiClient.skillGrants(name, cleanItems, revision, enableAfterGrant);
        if (!result?.ok) throw new Error(result?.error || 'Skill grant was cancelled.');
        return result;
    }

    // A slow owner action answers its running job: the toast is the job's own message, the card follows the queue.
    function stillWorking(name, result) {
        if (!result?.pending) return false;
        showToast(`${name}: ${result.message || 'still working…'} It finishes in the background; you can leave this page.`, 'muted');
        readInstalled({ quiet: true });
        return true;
    }

    const missingGrantItems = (grants) => [...(grants.missing_keys || grants.requested_keys || []), ...(grants.missing_permissions || grants.requested_permissions || [])];

    async function toggleSkillEnabled(name, wantsEnabled) {
        const result = await postWithFeedback(`/api/skills/${encodeURIComponent(name)}/toggle`, { enabled: wantsEnabled });
        if (stillWorking(name, result)) return result;
        const failed = result.extension_action === 'extension_load_error';
        showToast(`${name} ${wantsEnabled ? 'is on' : 'is off'}${failed ? ' — it couldn’t start' : ''}`, failed ? 'warn' : 'ok');
        emitSkillLifecycle(wantsEnabled ? 'enable' : 'disable', name, result);
        return result;
    }

    // The server's typed refusal says why a skill cannot turn on; nothing is pre-judged here.
    async function turnOn(name) {
        const skill = await currentSkill(name);
        if (!grantReady(skill)) {
            const granted = await requestMissingKeyGrants(name, missingGrantItems(skill.grants || {}), skill.content_hash, true);
            if (stillWorking(name, granted)) return granted;  // its single job also owns the requested Enable
            if (granted?.setup_error) throw new Error(`Access granted, but ${granted.setup_error}`);
            if (granted?.enabled === true) {
                showToast(`${name} is on`, 'ok');
                emitSkillLifecycle('enable', name, granted);
                return granted;
            }
        }
        return toggleSkillEnabled(name, true);
    }

    async function reviewInBackground(name) {
        if (reviewingSkills.has(name)) return null;
        reviewingSkills.add(name);
        paintInstalled();
        try {
            showToast(`${name}: review started. It can take a while; you can leave this page.`, 'muted');
            const result = await postWithFeedback(`/api/skills/${encodeURIComponent(name)}/review`, {});
            const findings = result.findings?.length ?? 0;
            showToast(`${name}: review ${result.status}${findings ? ` (${plural(findings, 'finding')})` : ''}${result.error ? ` — ${result.error}` : ''}`,
                reviewTone(result.status, result.error));
            emitSkillLifecycle('review', name, result);
            return result;
        } finally {
            reviewingSkills.delete(name);
        }
    }

    async function attest(name) {
        const skill = await currentSkill(name);
        if (!skill.content_hash) throw new Error('Skill revision is unavailable. Refresh and retry.');
        const blocked = skill.review_status === 'blockers' && !reviewReady(skill);
        // The author is the catalog entry's (the host's cache, never the network).
        if (skill.install?.source?.id && !catalogEntry(skill)) await store.catalog().catch(() => null);
        const ok = await openSkipReviewDialog({
            name, sourceName: installedSourceLabel(skill), owner: catalogEntry(skill)?.owner || '',
            permissions: skill.permissions || [], label: blocked ? 'Turn on anyway' : 'Skip review',
        });
        if (ok !== true) return false;
        // Skip review skips only the model review: the deterministic file check still runs server-side.
        const result = await postWithFeedback(`/api/owner/skills/${encodeURIComponent(name)}/attest-review`, { expected_content_hash: skill.content_hash });
        if (stillWorking(name, result)) return true;
        showToast(`${name}: ${blocked ? 'turned on anyway' : 'review skipped'}. Your choice is recorded.`, 'warn');
        emitSkillLifecycle('attest_review', name, result);
        return true;
    }

    async function repair(name) {
        if (repairingSkills.has(name)) return false;
        const skill = await currentSkill(name);
        const ok = await openConfirmDialog({
            title: `Ask Ouroboros to fix ${name}?`,
            body: `Ouroboros starts a task to fix ${name}, reviews the result, turns it on when it is ready and tests it. The fixed skill counts as changed here. If the task cannot start, chat says why.`,
            confirmLabel: 'Ask Ouroboros',
        });
        if (!ok) return false;
        repairingSkills.add(name);
        paintInstalled();
        try {
            await postWithFeedback('/api/command', {
                cmd: buildHealPrompt(skill),
                task_constraint: { mode: 'normal', skill_name: skill.name || name, payload_root: skill.payload_root || '', allow_enable: false, allow_review: true },
                visible_task_id: `skill_repair_${name}`,
            });
            showToast(`${name}: Ouroboros is on it — see chat`, 'ok');
            emitSkillLifecycle('repair', name);
            if (typeof ctx.showPage === 'function') ctx.showPage('chat');
        } finally {
            repairingSkills.delete(name);
        }
        return true;
    }

    async function updateSkill(name, skipReview = false) {
        const skill = await currentSkill(name);
        if (skipReview) {
            const result = await updateWithoutReview(skill);
            if (!result) return false;
            showToast(`${name}: updating to ${skill.install.update_rejected.version} without review`, 'warn');
            emitSkillLifecycle('update', name, result);
            return true;
        }
        const update = skill.install?.update_available;
        if (!update?.digest) throw new Error('No update is waiting for this skill.');
        let mode = 'replace';
        // A copy changed here, or one whose original is unknown, asks what to keep (SPEC §7, A03).
        if (skill.install?.modified || skill.install?.base_known === false) {
            mode = await openUpdateDialog({
                name, sourceName: installedSourceLabel(skill), update,
                installedVersion: skill.install.release?.version, currentVersion: skill.version,
                baseKnown: skill.install.base_known !== false, editedByOwner: skill.review_gate?.blocking_reason === 'owner_edit',
            });
            if (!mode) return false;
        }
        // The host refuses when the candidate or the local files changed since this card was read.
        const result = await skillSourcesApi.updateSkill(name, mode, update.digest, skill.content_hash || '');
        const merge = result?.base_available === false
            ? 'Ouroboros is merging your changes into the update without the original version — see chat'
            : 'Ouroboros is updating it and keeping your changes — see chat';
        const words = { replace: `updating to ${versionText(update.version)}`, merge, keep: `you keep your version; ${versionText(update.version) || 'this version'} isn’t offered again` };
        showToast(`${name}: ${words[mode] || 'updated'}`, 'ok');
        emitSkillLifecycle('update', name, result);
        return true;
    }

    async function uninstall(name) {
        const skill = installedByName()?.get(name);
        const ok = await openConfirmDialog({
            title: `Uninstall ${name}?`,
            body: `Ouroboros loses ${name} and what it adds. Its settings and the keys you allowed are kept, so installing it again restores them.`,
            confirmLabel: 'Uninstall', danger: true,
        });
        if (!ok) return false;
        const result = await skillSourcesApi.uninstallSkill(name);
        if (result?.ok === false) throw new Error(result.error || 'Uninstall was refused.');
        showToast(`${name} uninstalled${skill?.install?.source?.name ? `; it is still in ${skill.install.source.name}` : ''}`, 'ok');
        emitSkillLifecycle('uninstall', name, result);
        return true;
    }

    async function deleteLocal(name, payloadRoot) {
        const skill = await currentSkill(name);
        if (!skill.content_hash) throw new Error('Skill revision is unavailable. Refresh and retry.');
        const ok = await openConfirmDialog({
            title: `Delete ${name}?`,
            body: `${name} was made on this computer${skill.published ? '' : ' and isn’t published anywhere'}, so deleting it can’t be undone.`,
            details: { summary: 'Show the folder', rows: [{ label: 'Folder', value: `data/${payloadRoot || skill.payload_root}` }] },
            confirmLabel: 'Delete', danger: true,
        });
        if (!ok) return false;
        const result = await apiClient.deleteSkill(name, payloadRoot || skill.payload_root, skill.content_hash);
        if (stillWorking(name, result)) return true;
        showToast(result.ok ? `${name} deleted` : `${name}: delete failed — ${result.error}`, result.ok ? 'ok' : 'danger');
        if (result.ok) emitSkillLifecycle('delete', name, result);
        return true;
    }

    async function publish(name) {
        const skill = installedByName()?.get(name) || await currentSkill(name);
        const sources = await store.sources().catch(() => []);
        const choices = publishDestinations(skill, sources);
        const choice = await openPublishDestinationDialog({ name, choices, changed: Boolean(skill.install) });
        if (!choice) return false;
        const destination = destinationFromChoice(choice.value);
        const outcome = await runSkillPublishFlow(name, { destination, privateRepository: choice.private });
        if (!outcome.started) return false;
        showToast(`${name}: publication task ${outcome.task?.task_id || ''} started — see chat`, 'ok');
        emitSkillLifecycle('submit_hub', name);
        if (typeof ctx.showPage === 'function') ctx.showPage('chat');
        return true;
    }

    // The skill as its card shows it: forgetting names the record the owner saw, never a newer one.
    const shownSkill = (name) => (snapshot?.rawSkills || []).find((item) => item.name === name) || null;

    async function forgetPublication(name) {
        const skill = shownSkill(name) || await currentSkill(name);
        const ok = await openConfirmDialog({
            title: `Forget the publication record of ${name}?`,
            body: 'This forgets only this computer’s note of what was sent; the request stays on GitHub.',
            confirmLabel: 'Forget the record',
        });
        if (!ok) return false;
        await postWithFeedback(`/api/skills/${encodeURIComponent(name)}/publication/clear`, { expected_published: skill.published || {} });
        showToast(`${name}: publication record forgotten`, 'ok');
        return true;
    }

    function openFindings(name) {
        const card = container.querySelector(`:scope > .skills-card[data-skill="${CSS.escape(name)}"]`);
        const details = card?.querySelector('.skills-details');
        if (!details) return;
        details.open = true;
        if (installedByName()?.get(name)?.install?.update_rejected) {
            const history = details.querySelector('.skills-review-history');
            if (history) history.open = true;
        }
        const findings = details.querySelector('[data-skill-findings]');
        if (findings) findings.open = true;
        (findings || details).scrollIntoView({ block: 'nearest' });
    }

    function openInstalledSheet(name, opener) {
        const skill = installedByName()?.get(name);
        if (!skill) return;
        const source = (cache.sourcesValue || []).find((row) => row.id === skill.install?.source?.id) || null;
        const catalogItem = catalogEntry(skill);
        const item = catalogItem || {
            id: `installed/${name}`, source_id: skill.install?.source?.id || '', source_name: installedSourceLabel(skill),
            name, description: skill.description || '', version: skill.version, permissions: skill.permissions || [],
            needs_keys: skill.grants?.requested_keys || [], official: Boolean(skill.install?.official), signals: null,
            category: skill.category || [],
        };
        sheet?.close();
        // The source's entry has the full text; a registry install has no entry to read back.
        const origin = skill.install?.source;
        sheet = openSkillDetailSheet({
            item, installed: skill, source, returnFocus: opener,
            actionsHtml: connectsServices(skill).length ? '<button type="button" class="btn btn-default btn-sm" data-sheet-action="open_place">Places</button>' : '',
            onAction: (action) => {
                if (action === 'open_place') { sheet?.close(); openConnectorPlaces(name, { returnFocus: opener }); }
            },
            readDetail: origin?.id && origin.kind !== 'registry' ? ({ signal } = {}) => skillSourcesApi.item(origin.id, name, { signal }) : null,
        });
    }

    /** One installed-skill action by name (cards, the menu, Browse and the sheet share it). */
    async function runInstalledAction(name, action, options = {}) {
        if (action === 'open_detail') { openInstalledSheet(name, options.node); return false; }
        if (action === 'open_widgets') { document.querySelector('[data-nav-page="widgets"]')?.click(); return false; }
        if (action === 'read_findings') { openFindings(name); return false; }
        if (action === 'find_in_browse') { selectTab('browse'); browse?.searchFor(name); return false; }
        if (action === 'review' || action === 'rereview') {
            const ok = await openConfirmDialog({
                title: `Review ${name}?`,
                body: `Ouroboros reviews ${name} on this computer: its files go to your review models. It can take a while and runs in the background.`,
                confirmLabel: 'Review',
            });
            if (!ok) return false;
            await reviewInBackground(name);
            return true;
        }
        if (action === 'attest_review') return attest(name);
        if (action === 'repair') return repair(name);
        if (action === 'update') return updateSkill(name);
        if (action === 'update_without_review') return updateSkill(name, true);
        if (action === 'uninstall') return uninstall(name);
        if (action === 'delete') return deleteLocal(name, options.payloadRoot);
        if (action === 'submit_hub') return publish(name);
        if (action === 'forget_publication') return forgetPublication(name);
        if (action === 'turn_on') return turnOn(name);
        if (action === 'grant') {
            const skill = await currentSkill(name);
            const keys = String(options.keys || '').split(',').map((k) => k.trim()).filter(Boolean);
            const result = await requestMissingKeyGrants(name, keys.length ? keys : missingGrantItems(skill.grants || {}), skill.content_hash);
            if (result && !stillWorking(name, result)) {
                const loaded = result.extension_action === 'extension_loaded';
                const problem = result.setup_error || result.load_error;
                showToast(`${name}: access granted${loaded ? ' and it started' : ''}${problem ? `, but it couldn’t start: ${problem}` : ''}`, problem ? 'warn' : 'ok');
                emitSkillLifecycle('grant', name, result);
            }
            return true;
        }
        return false;
    }

    async function updateAll() {
        const candidates = updateAllCandidates(projected(), cache.sourcesValue);
        if (candidates.length < 2) return;
        const lines = candidates.map((skill) => `${skill.name} ${versionText(skill.install.release?.version || skill.version)} → ${versionText(skill.install.update_available.version)}`);
        const ok = await openConfirmDialog({
            title: `Update ${plural(candidates.length, 'skill')}?`,
            body: `${lines.join(', ')}. Skills you changed here are not included; each of those asks what to keep.`,
            confirmLabel: `Update ${candidates.length}`,
        });
        if (!ok) return;
        let failed = 0;
        for (const skill of candidates) {
            try { await skillSourcesApi.updateSkill(skill.name, 'replace', skill.install.update_available.digest, skill.content_hash || ''); }
            catch (err) { failed += 1; showToast(`${skill.name}: ${err.message || err}`, 'danger'); }
        }
        showToast(failed ? `${candidates.length - failed} of ${candidates.length} updates started` : `${plural(candidates.length, 'update')} started`, failed ? 'warn' : 'ok');
        emitSkillLifecycle('update_all', '');
        await readInstalled({ quiet: true });
    }

    // ---- event wiring ---------------------------------------------------------------------------
    const onClick = async (event) => {
        if (routeExternalLink(event)) return;
        if (event.target.closest('.skills-menu-item')) closeMenus({ restoreFocus: true });
        const mark = event.target.closest('[data-official-mark]');
        if (mark) {
            await store.sources().catch(() => null);
            openOfficialExplanation(mark, (cache.sourcesValue || []).find((source) => source.id === mark.dataset.officialSource) || officialSource());
            return;
        }
        const goto = event.target.closest('[data-skills-goto]');
        if (goto) { selectTab(goto.dataset.skillsGoto); return; }
        const chip = event.target.closest('[data-installed-filter]');
        if (chip) { filter = chip.dataset.installedFilter; paintFilters(snapshot?.rawSkills ? projected() : null); applyInstalledFilter(); return; }
        const menuTrigger = event.target.closest('[data-skill-menu-trigger]');
        if (menuTrigger) { openSkillMenu(menuTrigger); return; }
        const actionTarget = event.target.closest('[data-skill-action]');
        if (!actionTarget || actionTarget.disabled || actionTarget.closest('#skills-pane-browse, #skills-pane-sources')) return;
        const name = actionTarget.dataset.skill;
        const action = actionTarget.dataset.skillAction;
        if (action === 'submit_hub' && actionTarget.dataset.submitDisabled === 'true') {
            showToast(`${name}: ${actionTarget.dataset.submitReason || 'publishing is not available now'}`, 'warn');
            return;
        }
        actionTarget.disabled = true;
        let refresh = true;
        try {
            refresh = await runInstalledAction(name, action, { keys: actionTarget.dataset.keys || '', payloadRoot: actionTarget.dataset.payloadRoot || '', node: actionTarget }) !== false;
        } catch (err) {
            if (err.cancelled) refresh = false;
            showToast(`${name}: ${err.message || err}`, err.cancelled ? 'warn' : 'danger');
        } finally {
            actionTarget.disabled = false;
            if (refresh) await readInstalled({ quiet: true });
        }
    };

    // The switch: change, so keyboard and pointer activation match.
    const onChange = async (event) => {
        const target = event.target;
        if (!target?.classList?.contains('skills-toggle')) return;
        const name = target.dataset.skill;
        if (!name) return;
        const wantsEnabled = Boolean(target.checked);
        let refresh = true;
        target.disabled = true;
        target.dataset.skillPending = 'true';
        try {
            if (wantsEnabled) await turnOn(name);
            else await toggleSkillEnabled(name, false);
            target.setAttribute('aria-checked', wantsEnabled ? 'true' : 'false');
        } catch (err) {
            if (err.cancelled) refresh = false;
            target.checked = !wantsEnabled;
            target.setAttribute('aria-checked', (!wantsEnabled).toString());
            showToast(`${name}: ${err.message || err}`, err.cancelled ? 'warn' : 'danger');
        } finally {
            target.disabled = false;
            delete target.dataset.skillPending;
            if (refresh) readInstalled({ quiet: true });
        }
    };
    // A findings block is rendered collapsed; its list is built on first open
    // from the row already in memory (`toggle` does not bubble: capture).
    const onToggle = (event) => fillFindingsOnOpen(event.target, snapshot?.rawSkills);
    const onSearch = () => applyInstalledFilter();

    // ---- Browse and Sources ---------------------------------------------------------------------
    const host = {
        installedByName,
        installedSkills: () => (snapshot?.rawSkills ? projected() : null),
        refreshInstalled: () => readInstalled({ quiet: true }),
        sources: (force) => store.sources(force),
        sourcesProblem: () => cache.sourcesProblem,
        catalog: (options) => store.catalog(options),
        reloadAll: async () => { await store.sources(true).catch(() => null); },
        runInstalledAction: async (name, action) => {
            try {
                const changed = await runInstalledAction(name, action);
                if (changed !== false) await readInstalled({ quiet: true });
            } catch (err) {
                if (!err.cancelled) showToast(`${name}: ${err.message || err}`, 'danger');
            }
        },
        confirmFirstStart: async (sourceId) => {
            const sources = await store.sources().catch(() => []);
            if (firstStartAsked || !unconfirmedHubRows(sources).some((row) => !sourceId || row.id === sourceId)) return;
            firstStartAsked = true;
            await sourcesView.confirmFirstStart();
        },
        compare: (item, installed) => openCompareDialog({
            mine: { name: installed?.name, official: installed?.install?.official, source: installedSourceLabel(installed || {}),
                version: installed?.version, state: installed?.enabled ? 'On' : 'Off', description: installed?.description },
            other: { name: item.name, official: item.official, source: item.source_name, owner: item.owner, version: item.version,
                popularity: popularityText(item), description: item.description,
                reviewed: (cache.sourcesValue || []).find((row) => row.id === item.source_id)?.trust !== 'trusted' },
        }),
        toast: (message, tone) => showToast(message, tone),
        showInstalled: (name) => { selectTab('installed'); searchInput.value = name; applyInstalledFilter(); },
        showMadeHere: () => { filter = 'made'; selectTab('installed'); },
        openBehaviorSettings: () => { if (typeof ctx.openSettingsTab === 'function') ctx.openSettingsTab('behavior'); },
    };
    const browse = createSkillBrowse({ pane: document.getElementById('skills-pane-browse'), chrome: document.getElementById('skills-browse-chrome'), host });
    const sourcesView = createSkillSources({ pane: document.getElementById('skills-pane-sources'), host });

    function activateTab(tabName) {
        root.querySelectorAll('.skills-tab-panel').forEach((panel) => { panel.hidden = panel.dataset.pane !== tabName; });
        root.querySelectorAll('.skills-search-chrome').forEach((row) => { row.hidden = row.dataset.chromePane !== tabName; });
    }

    async function showActive({ refresh = false } = {}) {
        if (activeTab === 'installed') return readInstalled({ quiet: Boolean(snapshot) });
        // Browse and Sources need the installed listing for every card's state.
        const listing = snapshot ? Promise.resolve() : readInstalled();
        if (activeTab === 'browse') await Promise.all([browse.show({ refresh }), listing]);
        else await Promise.all([sourcesView.show({ refresh }), listing]);
        if (!firstStartAsked) host.confirmFirstStart();
    }

    function selectTab(tabName) {
        closeMenus();
        activeTab = tabName;
        tabs.select(tabName);
        activateTab(tabName);
        paintChrome();
        showActive();
    }

    const tabs = bindTabStrip(root.querySelector('.skills-tabs'), {
        dataAttr: 'data-tab', activeClass: 'is-active',
        onChange(tabName) { closeMenus(); activeTab = tabName; activateTab(tabName); paintChrome(); showActive(); },
    });
    // On a desk Installed, Browse and Sources start the bar that Update all and Refresh end (DESIGN "Page header").
    const disposeBar = bindPageHeader(root.querySelector('.app-page-header'));

    const onRefresh = async () => {
        refreshBtn.disabled = true;
        refreshBtn.textContent = 'Refreshing';
        try {
            if (activeTab === 'installed') await readInstalled({ quiet: true });
            else if (activeTab === 'browse') await Promise.all([browse.refresh(), readInstalled({ quiet: true })]);
            else await sourcesView.refresh();
        } catch (err) {
            showToast(`Skills view failed: ${err.message || err}`, 'danger');
        } finally {
            refreshBtn.disabled = false;
            refreshBtn.textContent = 'Refresh';
        }
    };
    const onPageShown = (event) => {
        closeMenus();
        visible = event.detail?.page === 'skills';
        if (!visible) { clearTimeout(pollTimer); sheet?.close(); return; }
        tabs.select(activeTab);
        store.sources(true).catch(() => null);
        showActive();
    };
    let wsTimer = 0;
    const onLifecycle = () => {
        if (!visible) return;
        clearTimeout(wsTimer);
        wsTimer = setTimeout(() => readInstalled({ quiet: true }), 300);
    };
    const unsubscribe = typeof ctx.ws?.on === 'function' ? ctx.ws.on('extension_lifecycle', onLifecycle) : null;

    refreshBtn.addEventListener('click', onRefresh);
    updateAllBtn.addEventListener('click', updateAll);
    searchInput.addEventListener('input', onSearch);
    root.addEventListener('click', onClick);
    container.addEventListener('change', onChange);
    container.addEventListener('toggle', onToggle, true);
    window.addEventListener('ouro:page-shown', onPageShown);

    // Another surface asks for one installed skill's entry ({skill}), a Browse search ({query}) or a Browse category
    // ({category}): the dictation chooser's engines (dictation_chooser.js), a place's connector (place_sheet.js).
    const onOpenSkills = async (event) => {
        const { skill = '', query = '', category = '' } = event?.detail || {};
        closeMenus();
        if (!await ctx.showPage?.('skills')) return;
        if (!skill && category) { selectTab('browse'); await browse?.showCategory(category); return; }
        if (!skill) { selectTab('browse'); await browse?.searchFor(query); return; }
        selectTab('installed');
        const found = () => container.querySelector(`.skills-card[data-skill="${CSS.escape(skill)}"]`);
        if (found()) return found().scrollIntoView({ block: 'nearest' });
        const observer = new MutationObserver(() => {
            if (!found()) return;
            observer.disconnect();
            found().scrollIntoView({ block: 'nearest' });
        });
        observer.observe(document.getElementById('page-skills'), { childList: true, subtree: true });
        setTimeout(() => observer.disconnect(), 10000);  // an entry that never comes leaves the page as it is
    };
    window.addEventListener('ouro:open-skills', onOpenSkills);
    paintFilters(null);

    return { destroy() {
        destroyed = true;
        renderGeneration += 1;
        clearTimeout(pollTimer);
        clearTimeout(wsTimer);
        closeMenus();
        sheet?.close();
        browse.destroy();
        sourcesView.destroy();
        tabs.destroy();
        disposeBar();
        unsubscribe?.();
        refreshBtn.removeEventListener('click', onRefresh);
        updateAllBtn.removeEventListener('click', updateAll);
        searchInput.removeEventListener('input', onSearch);
        root.removeEventListener('click', onClick);
        container.removeEventListener('change', onChange);
        container.removeEventListener('toggle', onToggle, true);
        window.removeEventListener('ouro:page-shown', onPageShown);
        window.removeEventListener('ouro:open-skills', onOpenSkills);
    } };
}
