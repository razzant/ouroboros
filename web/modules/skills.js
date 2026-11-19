// The skill views of Extensions (extensions.js draws the place): Skills, the installed skills (what Ouroboros can do
// now), and the skill catalog's Browse (what else it could do, from every connected source) and Sources (where skills
// come from and how much each place is trusted) are three of its pages, never tabs; MCP servers is its fourth. The
// views stay one mounted block; the section says which one is on screen (`showView`, `hideViews`). Skills never waits
// for a source or GitHub: it reads the local listing and the serialized lifecycle queue, and every owner action keeps
// its own words, confirmation and refusal. Browse and Sources live in their own modules. A skill's own form opens over
// Skills as a level of the section; Accounts and Permissions open in Settings with Back to Extensions.
import { createExtensions } from './extensions.js';
import { bindMenu } from './ui_interactions.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { showToast } from './toast.js';
import { apiClient, apiFetch, skillSourcesApi } from './api_client.js';
import {
    continuableReview, fillFindingsOnOpen, installedSkillModel, installedSourceLabel, isMadeHere, patchInstalledSkillEnrichment,
    renderInstalledSkillCard,
    renderRequirementNotice,
} from './skill_card_renderer.js';
import { createSkillBrowse } from './skill_browse.js';
import { createSkillSources, unconfirmedHubRows } from './skill_sources_view.js';
import { openSkillDetailSheet } from './skill_detail_sheet.js';
import { getContributions } from './contributions.js';
import { getHostModules } from './host_modules.js';
import {
    connectSkillTools, openCompareDialog, openOfficialExplanation, openPublishDestinationDialog, openSkipReviewDialog, openUpdateDialog,
    updateWithoutReview,
} from './skill_dialogs.js';
import { plural, popularityText, repoLabel, routeExternalLink, skillTitle, versionText } from './skill_presentation.js';
import { runSkillPublishFlow } from './skill_publish_flow.js';
import { installedTime } from './ui_helpers.js';
import { readConnectedServers } from './mcp_settings.js';
import { bindScrollFade } from './scroll_fade.js';
import {
    boundedText, emitSkillLifecycle, escapeHtmlAttr as escapeHtml, grantReady, renderSkillRepairPrompt,
    reviewTone, reviewReady,
} from './utils.js';

const LIFECYCLE_VISIBLE_STATUSES = new Set(['queued', 'running', 'failed']);
const INSTALLED_FILTERS = [
    ['all', 'All'], ['needs', 'Needs you'], ['updates', 'Updates'], ['off', 'Off'], ['made', 'Made here'],
];
// The line under an Installed filter that shows no card: an empty Needs you is good news, not a search that missed.
const FILTER_EMPTY = 'No installed skill matches.';
const NOTHING_NEEDS_YOU = 'Nothing needs you now.';
const WORK_POLL_MS = 3000;
const REVIEW_POLL_MS = 15000;

// The band's end over a skill view: its own actions, where they stood beside the tabs.
export const SKILLS_ACTIONS = '<button id="skills-update-all" class="btn btn-default btn-sm" hidden>Update all</button><button id="skills-refresh" class="btn btn-default btn-sm">Refresh</button>';

// The three skill views as one block of the section's page column: each view's search row, then their one scroller.
export function skillsViewsTemplate() {
    return `
            <div class="skills-search-chrome" data-chrome-pane="installed">
                <input type="search" class="ui-control skills-search" id="skills-installed-search" placeholder="Find an installed skill" aria-label="Find an installed skill" autocomplete="off">
                <div class="skills-filter-strip" id="skills-installed-filters" role="group" aria-label="Show" data-scroll-x></div>
            </div>
            <div class="skills-search-chrome" data-chrome-pane="browse" id="skills-browse-chrome" hidden></div>
            <div class="skills-scroll scroll-fade-y">
                <div class="skills-tab-panel" id="skills-pane-installed" data-pane="installed">
                    <div id="skills-status" class="ui-status skills-list-status" role="status" aria-live="polite"></div>
                    <div id="skills-org-requirements" hidden></div>
                    <div id="skills-list" class="skills-list"></div>
                    <p id="skills-filter-empty" class="skills-empty" hidden>${FILTER_EMPTY}</p>
                    <div id="skills-empty" class="skills-empty" hidden>
                        <p>No skills yet. Pick one in Browse, or ask Ouroboros: “find a skill for …”.</p>
                        <button type="button" class="btn btn-primary btn-sm" data-skills-goto="browse">Browse skills</button>
                    </div>
                </div>
                <div class="skills-tab-panel" id="skills-pane-browse" data-pane="browse" hidden></div>
                <div class="skills-tab-panel" id="skills-pane-sources" data-pane="sources" hidden></div>
            </div>
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
export function installedFilterCounts(skills, pageModules = {}) {
    const counts = Object.fromEntries(INSTALLED_FILTERS.map(([key]) => [key, 0]));
    for (const skill of skills) {
        const model = installedSkillModel(skill, { pageModules });
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
        if (!installedSkillModel(skill).update) return false;  // the same change eligibility as its individual card
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
    // A hub read from a folder of its repository takes no entries; its pointer entries still go to their authors.
    const folderHub = originKind === 'hub' && Boolean(originRow?.path);
    if (origin?.id && ['hub', 'git'].includes(originKind) && (authorRepo || !folderHub)) {
        out.push(authorRepo
            ? { value: `origin:${origin.id}`, repository: repoLabel(authorRepo), label: `Back to ${repoLabel(authorRepo)} — its author’s repository`,
                detail: 'Suggest your changes to its author as a new version. The author decides.' }
            : { value: `origin:${origin.id}`, repository: repoLabel(originRow?.url || origin.url), label: `Back to ${origin.name || originRow?.name} — where it came from`,
                detail: 'Suggest your changes as a new version. The maintainers decide.' });
    }
    // A view pinned to a commit, or to a tag named in full, never reads a publication (the host refuses both).
    for (const source of sources.filter((row) => row.kind === 'git' && !row.commit && !String(row.ref || '').startsWith('refs/tags/'))) {
        if (source.id === origin?.id) continue;
        out.push({ value: `source:${source.id}`, repository: repoLabel(source.url), label: `${source.name} — your own source`,
            detail: origin ? 'Publish your version now, under the same name, for your own installs.' : `Publish now. Every Ouroboros that reads ${source.name} gets it at once.` });
    }
    // The hub's two rows are one repository: a skill from it goes back as its origin, or beside it as a new skill.
    const community = sources.find((row) => row.kind === 'hub' && row.part === 'community');
    if (community && originKind === 'hub' && !folderHub) {
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
    return { skills: extResp.skills, live: extResp.live || {}, deployment: extResp.deployment || null };
}

export function initSkills(ctx = {}) {
    // The section draws the place: its list of pages, the band and the page column these views stand in.
    const section = createExtensions({ ctx, viewsHtml: skillsViewsTemplate(), toolbarHtml: SKILLS_ACTIONS,
        views: { show: (view) => showView(view), hide: () => hideViews(), placeShown: () => placeShown(),
            rowsShown: () => readInstalled({ quiet: true, join: true }) },
        facts: () => sectionFacts() });

    const root = document.getElementById('page-skills');
    const viewsRoot = root.querySelector('[data-extensions-views]');   // the views' own box: their presses stay theirs
    const container = document.getElementById('skills-list');
    const status = document.getElementById('skills-status');
    const requirementsEl = document.getElementById('skills-org-requirements');
    const emptyEl = document.getElementById('skills-empty');
    const filterEmpty = document.getElementById('skills-filter-empty');
    const filtersEl = document.getElementById('skills-installed-filters');
    if (filtersEl) bindScrollFade(filtersEl, { axis: 'x' });  // a phone's one scrolling line fades where chips hide
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
    let requestedSkill = '';
    let openRequest = 0;
    let filterEmptySentence = FILTER_EMPTY;  // the English last written, so a translated line is not rewritten each paint
    let snapshot = null;   // { rawSkills, live, queue, githubTokenConfigured }
    let installedFailed = false;
    let activeMenu = null;
    let sheet = null;
    let pollTimer = 0;
    let lastListingRead = 0;
    let firstStartAsked = false;
    const contributions = getContributions();
    const hostModules = getHostModules();
    const onPageModuleFacts = () => {
        if (destroyed || document.hidden || !snapshot) return;
        if (visible) paintInstalled();
        else if (section.rowsOnScreen()) section.paintAnswers();
    };
    const stopContributionFacts = contributions?.onChange?.(onPageModuleFacts);
    const stopHostFacts = hostModules?.onFactsChange?.(onPageModuleFacts);

    // ---- shared reads (Browse and Sources reuse them) -------------------------------------------
    const cache = { sources: null, sourcesValue: null, sourcesError: null, sourcesProblem: '', sourceSuggestions: [], catalog: null, catalogValue: null };
    const sourceSnapshot = () => ({ rows: cache.sourcesValue, error: cache.sourcesError, problem: cache.sourcesProblem });
    // Browse says how an entry installs from its catalog entry, conditionally, never from this list: a read that
    // settles after its cards, a sheet or a dialog opened changes none of their words.
    function paintSourceFacts() {
        if (visible && activeTab === 'sources') sourcesView.repaint();
        section.paintAnswers();
    }
    const store = {
        // Arriving views join the place's read. An explicit refresh or write can supersede it.
        sources({ fresh = false, replace = false } = {}) {
            if (!replace && cache.sources) return cache.sources;
            if (!fresh && !replace && cache.sourcesValue && !cache.sourcesError) return Promise.resolve(cache.sourcesValue);
            const current = () => cache.sources === read && !destroyed;
            const read = skillSourcesApi.list().then((data) => {
                const list = Array.isArray(data) ? data : data?.sources;
                if (!Array.isArray(list)) throw new Error('Skill sources response is unavailable.');
                return { list, data };
            }).then(({ list, data }) => {
                if (current()) {
                    cache.sourcesValue = list;
                    cache.sourcesError = null;
                    cache.sourcesProblem = String(data?.problem || '');
                    cache.sourceSuggestions = Array.isArray(data?.suggestions) ? data.suggestions : [];
                    paintSourceFacts();
                }
                return list;
            }, (error) => {
                if (current()) {
                    cache.sourcesError = error;
                    paintSourceFacts();
                }
                throw error;
            }).finally(() => { if (cache.sources === read) cache.sources = null; });
            return (cache.sources = read);
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
    // The catalog entry an installed skill came from (its author, full text), when the catalog was read.
    const catalogEntry = (skill) => (cache.catalogValue?.items || [])
        .find((item) => item.source_id === skill?.install?.source?.id && item.name === skill?.name) || null;
    const pageModuleFacts = () => {
        const contributions = getContributions();
        const host = getHostModules();
        const facts = host?.pageModuleFacts?.() || { faults: contributions?.faults?.() || [], held: [] };
        return { ...facts, startHeldModule: (key) => host?.startHeldModule?.(key) === true };
    };
    const cardOptions = () => ({ githubTokenConfigured: snapshot?.githubTokenConfigured, pageModules: pageModuleFacts(),
        contributionVisibility: (name) => contributions?.visibilityForSkill?.(name) || [] });

    // What the section's list says about these views, from reads already made (never a read of its own).
    function sectionFacts() {
        return {
            installed: snapshot?.rawSkills ? snapshot.rawSkills.length : null,
            installedFailed,
            needs: snapshot?.rawSkills ? installedFilterCounts(projected(), pageModuleFacts()).needs : 0,
            working: inFlightCount(lifecycleEventsFromQueue(snapshot?.queue)),
            sources: Array.isArray(cache.sourcesValue) ? cache.sourcesValue.length : null,
            sourcesFailed: Boolean(cache.sourcesError),
        };
    }

    // ---- Installed view -------------------------------------------------------------------------
    function paintFilters(skills) {
        const counts = skills ? installedFilterCounts(skills, pageModuleFacts()) : null;
        filtersEl.innerHTML = INSTALLED_FILTERS.map(([key, label]) => {
            const count = counts ? counts[key] : null;
            const hide = key !== 'all' && count === 0 && filter !== key;
            return hide ? '' : `<button type="button" class="filter-chip skills-category${key === filter ? ' active' : ''}" aria-pressed="${key === filter ? 'true' : 'false'}" data-installed-filter="${key}">${escapeHtml(label)}${count === null ? '' : `<span class="skills-category-count">${count}</span>`}</button>`;
        }).join('');
    }

    function applyInstalledFilter() {
        const query = searchInput.value.trim().toLowerCase();
        const byName = installedByName() || new Map();
        const pageModules = pageModuleFacts();
        let shown = 0;
        let members = 0;
        for (const card of container.querySelectorAll(':scope > .skills-card')) {
            const skill = byName.get(card.dataset.skill);
            const member = !skill || matchesInstalledFilter(skill, filter, installedSkillModel(skill, { pageModules }));
            const ok = member && (!skill || !query
                || [skill.name, skill.description].some((text) => String(text || '').toLowerCase().includes(query)));
            card.hidden = !ok;
            if (member) members += 1;
            if (ok) shown += 1;
        }
        const total = container.querySelectorAll(':scope > .skills-card').length;
        const sentence = filter === 'needs' && !members ? NOTHING_NEEDS_YOU : FILTER_EMPTY;
        if (sentence !== filterEmptySentence) filterEmpty.textContent = filterEmptySentence = sentence;
        filterEmpty.hidden = !(total && !shown);
    }

    function paintChrome() {
        const skills = snapshot?.rawSkills ? projected() : null;
        paintFilters(skills);
        const candidates = skills ? updateAllCandidates(skills, cache.sourcesValue) : [];
        updateAllBtn.hidden = !(activeTab === 'installed' && candidates.length >= 2);
        updateAllBtn.textContent = `Update ${candidates.length}`;
        updateAllBtn.setAttribute('aria-label', `Update ${candidates.length} trusted skills`);
        // Work in flight and what needs the owner are said by the Skills row of the section's list.
        section.paintAnswers();
    }

    function paintInstalled({ replace = false } = {}) {
        if (!snapshot?.rawSkills || destroyed) return;
        const skills = projected();
        const pageFacts = pageModuleFacts();
        const options = cardOptions;
        const cards = new Map(Array.from(container.querySelectorAll(':scope > .skills-card')).map((card) => [card.dataset.skill, card]));
        let replacementFocus = null;
        if (replace) {
            closeMenus();
            container.innerHTML = skills.map((skill) => renderInstalledSkillCard(skill, reviewingSkills, repairingSkills, snapshot.live, options())).join('');
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
                        template.innerHTML = renderInstalledSkillCard(skill, reviewingSkills, repairingSkills, snapshot.live, options());
                        const fresh = template.content.firstElementChild;
                        card.replaceWith(fresh);
                        if (focused) replacementFocus = fresh;
                    } else {
                        patchInstalledSkillEnrichment(card, skill, reviewingSkills, repairingSkills, snapshot.live, options(),
                            activeMenu?.card === card ? activeMenu.popover : card);
                    }
                    cards.delete(skill.name);
                } else {
                    const template = document.createElement('template');
                    template.innerHTML = renderInstalledSkillCard(skill, reviewingSkills, repairingSkills, snapshot.live, options());
                    container.insertBefore(template.content.firstElementChild, container.firstElementChild);
                }
            }
            for (const card of cards.values()) {
                if (activeMenu?.card === card) closeMenus();
                card.remove();
            }
        }
        emptyEl.hidden = skills.length > 0;
        const notice = renderRequirementNotice(snapshot.deployment);  // a missing required package has no card
        if (requirementsEl.innerHTML !== notice) requirementsEl.innerHTML = notice;
        requirementsEl.hidden = !notice;
        paintChrome();
        applyInstalledFilter();
        if (replacementFocus && !replacementFocus.hidden) {
            replacementFocus.querySelector('.skills-card-name')?.focus({ preventScroll: true });
        }
        browse?.repaintStates();
        schedulePoll();
    }

    // The rail and a view shown together share one read. Refresh and lifecycle changes supersede an older read.
    let listingOnWay = null;
    const sayLoading = () => { status.dataset.tone = 'muted'; status.textContent = 'Loading installed skills…'; };
    function readInstalled({ quiet = false, join = false } = {}) {
        if (join && listingOnWay) {
            if (!quiet && !snapshot) sayLoading();
            return listingOnWay;
        }
        const read = readListing({ quiet });
        const settled = listed => {
            if (listingOnWay !== read) return;
            listingOnWay = null;
            if (listed) revealRequestedSkill();
            requestedSkill = '';  // a failure or an absent skill ends the request too
        };
        listingOnWay = read;
        read.then(settled, () => settled(false));
        return read;
    }

    async function readListing({ quiet = false } = {}) {
        if (destroyed) return;
        const generation = ++renderGeneration;
        const current = () => generation === renderGeneration && !destroyed;
        const next = { rawSkills: null, live: {}, queue: snapshot?.queue || null, githubTokenConfigured: snapshot?.githubTokenConfigured };
        const optional = { state: 'loading', queue: 'loading' };
        const showReads = () => {
            if (!current() || snapshot !== next) return;
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
            if (ok) {
                const connected = (job) => job?.kind === 'connect_mcp' && job.status === 'succeeded';
                const previous = lifecycleEventsFromQueue(next.queue).filter(connected);
                const justConnected = lifecycleEventsFromQueue(data).some((job) => connected(job)
                    && !previous.some((old) => old.id === job.id));
                next.queue = data;
                if (justConnected) readConnectedServers();
            }
            if (snapshot === next) paintInstalled();
            showReads();
        }, () => { if (current()) { optional.queue = 'failed'; showReads(); } });
        if (!quiet && !snapshot) sayLoading();
        try {
            const primary = await fetchSkills();
            if (!current()) return;
            next.rawSkills = primary.skills;
            next.live = primary.live;
            next.deployment = primary.deployment;
            lastListingRead = Date.now();
        } catch (err) {
            if (!current()) return;
            installedFailed = true;
            section.paintAnswers();
            emptyEl.hidden = true;
            status.dataset.tone = 'danger';
            status.textContent = `Couldn’t refresh installed skills: ${err.message || err}.${snapshot ? ' Showing the previous list. Refresh to try again.' : ' Refresh to try again.'}`;
            return;
        }
        const firstPaint = !snapshot;
        snapshot = next;
        installedFailed = false;
        paintInstalled({ replace: firstPaint });
        showReads();
        return true;
    }

    // Local progress only (never GitHub): lifecycle work and pending MCP listings,
    // plus the listing now and then while a review runs outside the lane.
    function schedulePoll() {
        clearTimeout(pollTimer);
        if (destroyed || !visible || document.hidden || !snapshot) return;
        const working = inFlightCount(lifecycleEventsFromQueue(snapshot.queue)) > 0;
        const listingTools = snapshot.rawSkills?.some((skill) => (skill.mcp_servers || [])
            .some((server) => server.configured && !server.listed && !server.error));
        const reviewing = reviewingSkills.size > 0 || projected().some((skill) => /Reviewing/.test(installedSkillModel(skill).chip));
        if (!working && !reviewing && !listingTools) return;
        pollTimer = setTimeout(async () => {
            if (destroyed || !visible || document.hidden) return;
            if (Date.now() - lastListingRead >= REVIEW_POLL_MS || working || listingTools) await readInstalled({ quiet: true });
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

    /** A new review of the current bytes, or (`body`: `SkillReviewRequest`) the continuation of a stopped one. */
    async function reviewInBackground(name, body = {}) {
        if (reviewingSkills.has(name)) return null;
        reviewingSkills.add(name);
        paintInstalled();
        try {
            showToast(`${name}: review ${body.continue_seats ? 'continuing' : 'started'}. It can take a while; you can leave this page.`, 'muted');
            const result = await postWithFeedback(`/api/skills/${encodeURIComponent(name)}/review`, body);
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
                // The press is my human's request; the prompt above is the host's words, not theirs (#2778).
                ui_control: { name: 'skill_repair' },
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
        const catalogItem = catalogEntry(skill);
        const item = catalogItem || {
            id: `installed/${name}`, source_id: skill.install?.source?.id || '', source_name: installedSourceLabel(skill),
            name, title: skillTitle(skill), description: skill.description || '', version: skill.version, permissions: skill.permissions || [],
            needs_keys: skill.grants?.requested_keys || [], official: Boolean(skill.install?.official), signals: null,
            category: skill.category || [],
        };
        sheet?.close();
        // The source's entry has the full text; a registry install has no entry to read back.
        const origin = skill.install?.source;
        sheet = openSkillDetailSheet({
            item, installed: skill, returnFocus: opener,
            readDetail: origin?.id && origin.kind !== 'registry' ? ({ signal } = {}) => skillSourcesApi.item(origin.id, name, { signal }) : null,
        });
    }

    /** One installed-skill action by name (cards, the menu, Browse and the sheet share it). */
    async function runInstalledAction(name, action, options = {}) {
        if (action === 'toggle_contribution') {
            const key = String(options.contributionKey || '');
            if (!key || !contributions?.setHidden) throw new Error('Composer control is unavailable.');
            const hidden = options.contributionHidden !== 'true';
            await contributions.setHidden(key, hidden);
            showToast(`${contributions.byKey(key)?.label || 'Composer control'} ${hidden ? 'hidden' : 'shown'}`, 'ok');
            return false;
        }
        if (action === 'open_detail') { openInstalledSheet(name, options.node); return false; }
        if (action === 'open_widgets') {
            if (options.opener) void ctx.showPage?.('widgets', { opener: options.opener });
            else void ctx.showPage?.('widgets');
            return false;
        }
        if (action === 'open_secrets') {
            // Accounts and keys reveals the row of the first key the button names (a stored key's row is its name).
            void ctx.openSettings?.('accounts', { row: String(options.keys || '').split(',')[0].trim() });
            return false;
        }
        // The skill's own form stays with the skill: a level over Skills; Back returns to its card, to the pressed control
        // while it still stands, else (an item of the card's menu is gone by then) to the menu's button.
        if (action === 'open_settings') {
            const pressed = options.node;
            section.go(`skill/${encodeURIComponent(name)}`, {
                opener: () => (pressed?.isConnected && pressed.getClientRects().length ? pressed
                    : viewsRoot.querySelector(`.skills-card[data-skill="${CSS.escape(name)}"] [data-skill-menu-trigger]`)),
            });
            return false;
        }
        if (action === 'read_findings') { openFindings(name); return false; }
        if (action === 'find_in_browse') {
            if (await section.open('browse', { opener: options.node }) && visible && activeTab === 'browse') void browse.searchFor(name);
            return false;
        }
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
        if (action === 'continue_review') {
            const saved = continuableReview(await currentSkill(name));
            if (!saved) { showToast(`${name}: no stopped review of these files is left to continue.`, 'warn'); return false; }
            const ok = await openConfirmDialog({
                title: `Continue the review of ${name}?`,
                body: 'The reviewer that stopped continues its saved work on the files it was reviewing. It uses your review models again and runs in the background.',
                confirmLabel: 'Continue review',
            });
            if (!ok) return false;
            await reviewInBackground(name, {
                review_reference: { record_id: saved.record_id, content_hash: saved.content_hash },
                continue_seats: saved.continuable_seats.map(({ seat_id, operation_id }) => ({ seat_id, operation_id })),
            });
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
        if (action === 'connect_tools') {
            const result = await connectSkillTools(await currentSkill(name));
            if (stillWorking(name, result)) return true;  // not confirmed yet: the card follows its job
            if (result?.ok) {
                readConnectedServers();
                showToast(`${name}: ${result.changed === false ? 'tools already configured' : 'tools configured'}. Its own instructions say what else they need.`, 'ok');
            }
            return Boolean(result);
        }
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
            body: `${lines.join(', ')}. Not included: skills you changed here and updates from sources that review each skill; each of those updates from its own card.`,
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
            // A Browse card's mark is explained from its own entry. Elsewhere only a mark bound to a source says how
            // that source installs, from its row as read now; an unbound mark borrows no other source's setting.
            const entry = mark.closest('.marketplace-card');  // only Browse draws these
            const bound = entry ? '' : mark.dataset.officialSource;
            if (bound) await store.sources().catch(() => null);
            openOfficialExplanation(mark, entry ? browse.officialSource(entry.dataset.item)
                : (bound && !cache.sourcesError && cache.sourcesValue?.find((row) => row.id === bound)) || null);
            return;
        }
        const goto = event.target.closest('[data-skills-goto]');
        if (goto) { void section.open(goto.dataset.skillsGoto, { opener: goto }); return; }
        const chip = event.target.closest('[data-installed-filter]');
        if (chip) { requestedSkill = ''; filter = chip.dataset.installedFilter; paintFilters(snapshot?.rawSkills ? projected() : null); applyInstalledFilter(); return; }
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
        // Disabling a focused button drops focus to the page in Chromium; the keyboard owner gets it back below.
        const hadFocus = document.activeElement === actionTarget;
        actionTarget.disabled = true;
        let refresh = true;
        try {
            if (action === 'start_host_module') {
                refresh = pageModuleFacts().startHeldModule(actionTarget.dataset.hostModuleKey || '') === true;
            } else {
                refresh = await runInstalledAction(name, action, { keys: actionTarget.dataset.keys || '', payloadRoot: actionTarget.dataset.payloadRoot || '', node: actionTarget,
                    contributionKey: actionTarget.dataset.contributionKey || '', contributionHidden: actionTarget.dataset.contributionHidden || '' }) !== false;
            }
        } catch (err) {
            if (err.cancelled) refresh = false;
            showToast(`${name}: ${err.message || err}`, err.cancelled ? 'warn' : 'danger');
        } finally {
            actionTarget.disabled = false;
            if (hadFocus && actionTarget.isConnected && (!document.activeElement || document.activeElement === document.body)) {
                actionTarget.focus({ preventScroll: true });
            }
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
    const onSearch = () => { requestedSkill = ''; applyInstalledFilter(); };

    function revealRequestedSkill() {
        if (!requestedSkill || !visible || document.hidden || activeTab !== 'installed') return;
        const card = container.querySelector(`.skills-card[data-skill="${CSS.escape(requestedSkill)}"]:not([data-lifecycle-virtual])`);
        if (!card) return;
        if (card.hidden) {
            searchInput.value = requestedSkill;
            applyInstalledFilter();
            if (card.hidden) {
                filter = 'all';
                paintFilters(projected());
                applyInstalledFilter();
            }
        }
        card.scrollIntoView({ block: 'nearest' });
    }

    // Opening Skills may already have started a fresh read. A read started before this request
    // cannot settle it: the requested card may just have been installed.
    function requestInstalledSkill(name, beforeOpen) {
        requestedSkill = name;
        void readInstalled({ quiet: true, join: renderGeneration > beforeOpen });
    }

    // ---- Browse and Sources ---------------------------------------------------------------------
    const host = {
        installedByName,
        installedSkills: () => (snapshot?.rawSkills ? projected() : null),
        live: () => snapshot?.live,
        refreshInstalled: () => readInstalled({ quiet: true }),
        sources: store.sources,
        sourceSnapshot,
        sourceSuggestions: () => cache.sourceSuggestions,
        catalog: (options) => store.catalog(options),
        // A source change moves the listing's source facts too (a re-added name's id, updates): read both again, and
        // the catalog under Browse when it is the page on screen (the first-start choice is made over it).
        reloadAll: async () => {
            await Promise.all([store.sources({ replace: true }).catch(() => null), readInstalled({ quiet: true })]);
            if (visible && activeTab === 'browse') await browse.show();
        },
        runInstalledAction: async (name, action, options) => {
            try {
                const changed = await runInstalledAction(name, action, options);
                if (changed !== false) await readInstalled({ quiet: true });
            } catch (err) {
                if (!err.cancelled) showToast(`${name}: ${err.message || err}`, 'danger');
            }
        },
        confirmFirstStart: async (sourceId, canOpen = () => true) => {
            // Browse and Sources have already requested these facts. A check after a failed visit is not another read.
            await cache.sources?.catch(() => null);
            const sources = cache.sourcesError ? [] : cache.sourcesValue || [];
            if (firstStartAsked || !canOpen() || !unconfirmedHubRows(sources).some((row) => !sourceId || row.id === sourceId)) return;
            await sourcesView.confirmFirstStart(() => {
                if (firstStartAsked || !canOpen()) return false;
                firstStartAsked = true;
                return true;
            });
        },
        compare: (item, installed) => openCompareDialog({
            mine: { name: installed?.name, official: installed?.install?.official, source: installedSourceLabel(installed || {}),
                version: installed?.version, state: installed?.enabled ? 'On' : 'Off', description: installed?.description },
            other: { name: item.name, official: item.official, source: item.source_name, owner: item.owner, version: item.version,
                popularity: popularityText(item), description: item.description },
        }),
        toast: (message, tone) => showToast(message, tone),
        pageModuleFacts,
        showInstalled: async (name, opener) => {
            const beforeOpen = renderGeneration;
            if (!await section.open('installed', { opener }) || !visible || activeTab !== 'installed') return;
            searchInput.value = name;
            applyInstalledFilter();
            requestInstalledSkill(name, beforeOpen);
        },
        showMadeHere: () => { filter = 'made'; return section.open('installed'); },
        // Skill sources open here as a level over Sources; the rest of Settings with Back to Extensions.
        openSettings: (path, options = {}) => section.fromPage(path, options),
    };
    const browse = createSkillBrowse({ pane: document.getElementById('skills-pane-browse'), chrome: document.getElementById('skills-browse-chrome'), host });
    const sourcesView = createSkillSources({ pane: document.getElementById('skills-pane-sources'), host });

    function activateTab(tabName) {
        root.querySelectorAll('.skills-tab-panel').forEach((panel) => { panel.hidden = panel.dataset.pane !== tabName; });
        root.querySelectorAll('.skills-search-chrome').forEach((row) => { row.hidden = row.dataset.chromePane !== tabName; });
    }

    async function showActive({ refresh = false } = {}) {
        if (activeTab === 'installed') return readInstalled({ quiet: Boolean(snapshot), join: true });
        const view = activeTab;
        const stillHere = () => !destroyed && visible && !document.hidden && activeTab === view;
        // Browse and Sources need the installed listing for every card's state.
        const listing = snapshot ? Promise.resolve() : readInstalled({ join: true });
        if (activeTab === 'browse') await Promise.all([browse.show({ refresh }), listing]);
        else await Promise.all([sourcesView.show({ refresh }), listing]);
        if (!firstStartAsked && stillHere()) void host.confirmFirstStart(undefined, stillHere);
    }

    // A view is the section's open page (extensions.js calls this when Skills, Browse or Sources comes on screen).
    function showView(view) {
        closeMenus();
        requestedSkill = '';
        activeTab = view;
        visible = true;
        activateTab(view);
        paintChrome();
        // The retained cards must match the selected filter before the new read answers.
        if (view === 'installed') applyInstalledFilter();
        showActive();
    }

    // No view is on screen: another page of the section, a level over a view, or another place.
    function hideViews() {
        closeMenus();
        requestedSkill = '';
        visible = false;
        clearTimeout(pollTimer);
        sheet?.close();
    }

    // Sources also supplies the trust facts used by the skill views.
    function placeShown() {
        store.sources({ fresh: true }).catch(() => null);
    }

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
    let wsTimer = 0;
    const needsListing = () => !destroyed && !document.hidden && (visible || section.rowsOnScreen());
    const onLifecycle = () => {
        if (!needsListing()) {
            listingOnWay = null;  // on return, read past the change instead of joining a read started before it
            return;
        }
        clearTimeout(wsTimer);
        const before = renderGeneration;
        wsTimer = setTimeout(() => {
            if (needsListing() && renderGeneration === before) void readInstalled({ quiet: true });
        }, 300);
    };
    const offLifecycle = contributions.onLifecycle(onLifecycle);
    const unsubscribe = ctx.ws?.on?.('extension_lifecycle', event => {
        if (event?.action === 'connect_mcp') readConnectedServers();
    });
    const onVisibilityChange = () => {
        if (document.hidden) { ++openRequest; requestedSkill = ''; }
        schedulePoll();
    };

    refreshBtn.addEventListener('click', onRefresh);
    updateAllBtn.addEventListener('click', updateAll);
    searchInput.addEventListener('input', onSearch);
    viewsRoot.addEventListener('click', onClick);
    container.addEventListener('change', onChange);
    container.addEventListener('toggle', onToggle, true);
    document.addEventListener('visibilitychange', onVisibilityChange);

    // Another surface asks for one installed skill's entry ({skill}), a Browse search ({query}) or a Browse category
    // ({category}): the dictation chooser's engines (dictation_chooser.js), a place's connector (place_sheet.js). The
    // explicit target opens regardless of the device's previously selected page.
    const onOpenSkills = async (event) => {
        const { skill = '', query = '', category = '' } = event?.detail || {};
        const request = ++openRequest;
        const beforeOpen = renderGeneration;
        requestedSkill = '';
        closeMenus();
        if (!await section.open(skill ? 'installed' : 'browse') || request !== openRequest || destroyed) return;
        if (!skill && category) { await browse?.showCategory(category); return; }
        if (!skill) { await browse?.searchFor(query); return; }
        if (!visible || activeTab !== 'installed') return;
        requestInstalledSkill(skill, beforeOpen);
    };
    window.addEventListener('ouro:open-skills', onOpenSkills);
    paintFilters(null);

    // `open` and `pressedAgain` are the app's doors to the section (app.js `openExtensions`, the sidebar's row).
    return { open: section.open, capture: section.capture, pressedAgain: section.pressedAgain, destroy() {
        destroyed = true;
        renderGeneration += 1;
        clearTimeout(pollTimer);
        clearTimeout(wsTimer);
        closeMenus();
        sheet?.close();
        browse.destroy();
        sourcesView.destroy();
        section.destroy();
        unsubscribe?.();
        offLifecycle();
        stopContributionFacts?.();
        stopHostFacts?.();
        refreshBtn.removeEventListener('click', onRefresh);
        updateAllBtn.removeEventListener('click', updateAll);
        searchInput.removeEventListener('input', onSearch);
        viewsRoot.removeEventListener('click', onClick);
        container.removeEventListener('change', onChange);
        container.removeEventListener('toggle', onToggle, true);
        document.removeEventListener('visibilitychange', onVisibilityChange);
        window.removeEventListener('ouro:open-skills', onOpenSkills);
    } };
}
