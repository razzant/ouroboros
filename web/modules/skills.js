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
import { apiClient, apiFetch, readStateJson, skillSourcesApi, withinRequestLifetime } from './api_client.js';
import {
    continuableReview, fillFindingsOnOpen, installedSkillModel, installedSkillRowKey, installedSourceLabel, isMadeHere, patchInstalledSkillEnrichment,
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
import { manifestUnread, opensDetail, plural, popularityText, repoLabel, routeExternalLink, skillTitle, versionText } from './skill_presentation.js';
import { runSkillPublishFlow } from './skill_publish_flow.js';
import { installedTime } from './ui_helpers.js';
import { readConnectedServers } from './mcp_settings.js';
import { bindScrollFade } from './scroll_fade.js';
import {
    boundedText, emitSkillLifecycle, escapeHtmlAttr as escapeHtml, grantReady, renderSkillRepairPrompt,
    reviewTone, reviewReady,
} from './utils.js';

const LIFECYCLE_VISIBLE_STATUSES = new Set(['queued', 'running', 'failed']);
// What a job's name-level row says of itself while more than one installed folder uses the name, and once the page that
// read the job beside several folders lists fewer (the queue names no directory, so it cannot say which one it was for).
const SHARED_NAME_ATTEMPT = 'Latest attempt on this name. More than one installed folder uses it; this is not one of them.';
const SEEN_SHARED_ATTEMPT = 'Latest attempt on this name. It was seen while more than one installed folder shared the name, so it can’t say which folder it was for.';
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

/**
 * The listed row is an installed extension that is On and affirmatively ready: the host lists it loaded with the
 * reason `ready` (a recovered restart shows as `ready` too), and nothing waits on it or on a process to exit.
 * A script or instruction, a missing or unfamiliar reason, an Off or still-unhealthy row proves no recovery.
 */
export function listedReadyOn(skill, pageModules = {}) {
    if (!skill || skill.lifecycle_virtual || skill.identity_collision || skill.type !== 'extension') return false;
    if (skill.enabled !== true || skill.load_error || skill.live_loaded !== true || skill.live_reason !== 'ready') return false;
    if (Object.keys(skill.retained_companions || {}).length) return false;
    if ((skill.mcp_servers || []).some((server) => server.configured && (server.error || !server.listed))) return false;
    return !installedSkillModel(skill, { pageModules }).needs;
}

/**
 * Observation marks of the failed jobs of one accepted queue read, by the queue's own job id: an earlier mark carries
 * over, a job first seen failed takes `nextTick()` now. A job the queue no longer lists keeps nothing, and one with no
 * id has no identity to mark, so no later listing can be said to answer it.
 */
export function tickFailures(events, earlier, nextTick) {
    const ticks = new Map();
    for (const event of events) {
        const id = typeof event?.id === 'string' ? event.id : '';
        if (id && event.status === 'failed') ticks.set(id, earlier?.get(id) ?? nextTick());
    }
    return ticks;
}

/**
 * The job ids one accepted queue read lists in any visible status — queued, running or failed. The name-level
 * ambiguity ids are pruned by this, not by the failed marks: a job remembered while its name was listed for more
 * than one folder is still a visible record of the queue while it runs, and dropping it before it fails would
 * hand it to whichever folder is listed next.
 */
export function visibleJobIds(events) {
    const ids = new Set();
    for (const event of events) {
        const id = typeof event?.id === 'string' ? event.id : '';
        if (id && LIFECYCLE_VISIBLE_STATUSES.has(event.status)) ids.add(id);
    }
    return ids;
}

// The failed Enable jobs a listing has answered are kept by the queue's job id and nothing else, in this tab's
// sessionStorage: a reload still knows the answer while the queue lists the job. Every accepted queue read drops the ids it
// no longer lists as failed, so the copy never outgrows the queue (80 jobs). A store that refuses leaves them to this page.
// The jobs this page read while their name was listed for more than one folder — of any visible status, a job still
// queued or running included — are kept the same way, under a key of their own (they are shown, where the answered
// ones are hidden): the queue names no directory, so such a job is never given to a folder listed later, and a reload
// of this tab still knows it. An accepted queue read drops one of these ids once the queue lists it in no visible
// status at all; a job merely not failed yet is still listed, so it is not forgotten before it can fail. A page
// without that memory (a new tab, or storage that refuses) has only the queue and the listing it reads now, which
// cannot say which folder a job was for.
const ANSWERED_FAILURES_KEY = 'ouroboros.skills.answeredFailures';
const NAME_LEVEL_FAILURES_KEY = 'ouroboros.skills.nameLevelFailures';

const tabStore = (storage) => {
    try { return storage === undefined ? globalThis.sessionStorage || null : storage; } catch { return null; }
};

/** The job ids this tab kept under `key` (the answered ones); what is not a non-empty string is no id. */
export function readAnsweredFailures(storage, key = ANSWERED_FAILURES_KEY) {
    try {
        const ids = JSON.parse(tabStore(storage)?.getItem(key) || '[]');
        return new Set(Array.isArray(ids) ? ids.filter((id) => typeof id === 'string' && id) : []);
    } catch { return new Set(); }
}

export function writeAnsweredFailures(ids, storage, key = ANSWERED_FAILURES_KEY) {
    try {
        const store = tabStore(storage);
        if (ids.size) store?.setItem(key, JSON.stringify([...ids]));
        else store?.removeItem(key);
    } catch { /* full or refused: the answers last for this page only */ }
}

export const readNameLevelFailures = (storage) => readAnsweredFailures(storage, NAME_LEVEL_FAILURES_KEY);
export const writeNameLevelFailures = (ids, storage) => writeAnsweredFailures(ids, storage, NAME_LEVEL_FAILURES_KEY);

/**
 * The kept ids that stay: all of them while no queue read has been accepted (`marks` unknown), then only those the
 * accepted queue still marks — its failed-job ticks for the answered ids, its visibly listed job ids (any visible
 * status) for the name-level ones. A queue that could not be read takes none away.
 */
export function carryAnsweredFailures(ids, marks) {
    return new Set(marks ? [...ids].filter((id) => marks.has(id)) : ids);
}

/**
 * `answered(event)` says a failed job was settled earlier. It is asked of each failed job by its queue id alone, before
 * the join by name, so two directories of one name, or their order, never decide which row carries it.
 * `superseded(event, row)` is the caller's proof that a failed job has now been answered by a newer listing; it is asked
 * only of an ordinary failed job of a name listed once, so neither an ambiguous identity nor a name-level attempt ever
 * makes a new answer. The history entry stays in the queue, only the card stops carrying the failure. A job names no
 * directory: while more than one directory carries its name, it stands as its own placeholder row, never on one of them,
 * so the order of the listing decides nothing. `nameLevel(event, shared)` is the caller's memory of that (`shared`: the
 * name is listed more than once now), asked of every visible job, one still in flight included: a job remembered so
 * stays that placeholder when fewer directories are listed later — while it is still queued or running, and after it
 * fails — with words that no longer say two folders exist, and is never given to one of them, so it never sets a
 * listed row's toggle, busy state or error. No listed row can answer it, so it stays until a newer job of its name or
 * the queue's own eviction. With no memory a job is a placeholder only while `shared`.
 */
export function mergeLifecycleEvents(skills, events, { answered = null, superseded = null, nameLevel = null } = {}) {
    const out = skills.map((skill) => ({ ...skill }));
    const byName = new Map();
    // A null entry means there is no unique installed row for that name.
    for (const row of out) byName.set(row.name, byName.has(row.name) ? null : row);
    const processedTargets = new Set();
    for (const event of [...events].reverse()) {
        const name = event.target;
        if (!name) continue;
        const existing = byName.get(name);
        const shared = existing === null;
        const visible = LIFECYCLE_VISIBLE_STATUSES.has(event.status);
        // Remember every visible job, even one shadowed by an active or newer job of the same name.
        const attempt = visible && (Boolean(nameLevel?.(event, shared)) || shared);
        if (processedTargets.has(name)) continue;
        processedTargets.add(name);
        // A settled or answered newest job still takes the name; it does not expose older failures.
        if (!visible || (event.status === 'failed' && answered?.(event))) continue;
        if (existing && !attempt) {
            // The skill already has a real card: annotate it with the in-flight
            // transition so it shows "Turning off…" instead of a stale toggle
            // while the serialized lifecycle lane works through it.
            // Neither a shared name nor an attempt gets here, so neither is answered by a listed row: no row can say which
            // directory the job was for.
            if (event.status === 'failed' && superseded?.(event, existing)) continue;
            existing.lifecycle_status = event.status;
            existing.lifecycle_kind = event.kind || existing.lifecycle_kind || '';
            existing.lifecycle_message = event.message || '';
            existing.lifecycle_job_id = typeof event.id === 'string' ? event.id : '';
            existing.lifecycle_pending = event.status !== 'failed';
            if (event.status === 'failed' && event.error) existing.lifecycle_error = event.error;
            continue;
        }
        // A queue-only record: its action, status and error. The queue's `source` is the caller, never the package's
        // origin, and nothing here is an installed version, review or grant, so none is invented.
        out.unshift({
            name,
            description: shared ? SHARED_NAME_ATTEMPT : attempt ? SEEN_SHARED_ATTEMPT : event.message || event.error || 'Skill lifecycle operation',
            load_error: event.status === 'failed' ? event.error : '',
            lifecycle_kind: event.kind || '',
            lifecycle_status: event.status,
            lifecycle_pending: event.status !== 'failed',
            lifecycle_error: event.error || '',
            lifecycle_virtual: true,
            ...(attempt ? { lifecycle_name_level: true } : {}),
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

async function fetchSkills(signal) {
    const extResp = await apiClient.extensions({ signal });
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
    let listingReadController = null;
    let filter = 'all';
    let requestedSkill = '';
    let openRequest = 0;
    let filterEmptySentence = FILTER_EMPTY;  // the English last written, so a translated line is not rewritten each paint
    let snapshot = null;   // { rawSkills, live, queue, githubTokenConfigured }
    let installedFailed = false;
    let activeMenu = null;
    let sheet = null;
    let sheetPending = null;  // the pending install the open sheet describes: its row key and queue job
    let pollTimer = 0;
    let hintTimer = 0;
    let lastListingRead = 0;
    // The order of observations, not clocks. A snapshot carries the tick at which its queue first showed each failed
    // job (`failureTicks`) and the tick at which its listing request began (`listingTick`): only a listing begun after
    // the failure was seen can answer it. Both commit with the snapshot, so a queue read that is not accepted, or a
    // listing that never lands, leaves the visible snapshot's marks as they were.
    let readTick = 0;
    let listingReadState = 'idle';
    let pendingListingHint = null;
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
    const failureTick = (id) => (typeof id === 'string' && id ? snapshot?.failureTicks?.get(id) : undefined);
    const failureAnswered = (event) => event.kind === 'enable' && snapshot?.listingTick > (failureTick(event.id) ?? Infinity);
    // A job a fresh listing has once answered stays answered, by its queue id alone, while the queue still lists it: a
    // later, different fault of the same row (another companion stopping), a second directory taking the name, the order
    // of the two, or a reload of this tab must not bring the resolved Enable's error back. The ids are the snapshot's own,
    // so they live no longer than the failed jobs of the accepted queue read; this tab's sessionStorage keeps a copy.
    const answeredFailure = (event) => typeof event.id === 'string' && Boolean(snapshot?.answeredFailures?.has(event.id));
    // A new answer needs a listing begun after the failure was seen, and that listing's one row of the name ready and On.
    const supersededFailure = (event, row) => {
        if (!failureAnswered(event) || !listedReadyOn(row, pageModuleFacts())) return false;
        snapshot.answeredFailures.add(event.id);
        writeAnsweredFailures(snapshot.answeredFailures);
        return true;
    };
    // A job of any visible status this page reads while its name is listed for more than one folder names none of them,
    // now or later: its queue id is kept (as the answered ones are), so a folder listed afterwards is never given it —
    // neither its toggle and busy state while it runs, nor its error after it fails — and it stays its own row in a
    // later listing as in a reload of this tab. A job with no id has no identity to keep.
    const nameLevelAttempt = (event, shared) => {
        const ids = snapshot?.nameLevelFailures;
        if (!ids || typeof event.id !== 'string' || !event.id) return shared;
        if (shared) ids.add(event.id);
        return ids.has(event.id);
    };
    // What a snapshot keeps of the answered and the name-level ids (a first one starts from this tab's copy); a change
    // reaches the copy too.
    const carryAnswers = (ids, marks, write = writeAnsweredFailures) => {
        const kept = carryAnsweredFailures(ids, marks);
        if (kept.size !== ids.size) write(kept);
        return kept;
    };
    const projected = () => {
        if (!snapshot?.rawSkills) return [];
        const ids = snapshot.nameLevelFailures;
        const size = ids.size;
        const rows = mergeLifecycleEvents(snapshot.rawSkills, lifecycleEventsFromQueue(snapshot.queue), {
            answered: answeredFailure, superseded: supersededFailure, nameLevel: nameLevelAttempt,
        });
        if (ids.size !== size) writeNameLevelFailures(ids);
        return sortSkillsForDisplay(rows);
    };
    // A card carrying a failed Enable whose row, as listed, is ready, though no listing begun after the failure was
    // seen has answered it yet: one more read settles it. (`listed` is the row as the listing gave it: the card's own
    // failure would make its model ask for the owner.) A name-level attempt is no such card: no listed row answers it.
    const awaitsNewerListing = (card, listed, facts) => card.lifecycle_status === 'failed' && card.lifecycle_kind === 'enable'
        && !card.lifecycle_name_level && snapshot.listingTick <= (failureTick(card.lifecycle_job_id) ?? -Infinity) && listedReadyOn(listed, facts);
    const installedByName = () => (snapshot?.rawSkills ? new Map(projected().map((skill) => [skill.name, skill])) : null);
    const cardRowKey = (card) => card.dataset.skillRow || card.dataset.skill;
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
        const byRow = new Map(projected().map((skill) => [installedSkillRowKey(skill), skill]));
        const pageModules = pageModuleFacts();
        let shown = 0;
        let members = 0;
        for (const card of container.querySelectorAll(':scope > .skills-card')) {
            const skill = byRow.get(cardRowKey(card));
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
        const cards = new Map(Array.from(container.querySelectorAll(':scope > .skills-card')).map((card) => [cardRowKey(card), card]));
        let replacementFocus = null;
        if (replace) {
            closeMenus();
            container.innerHTML = skills.map((skill) => renderInstalledSkillCard(skill, reviewingSkills, repairingSkills, snapshot.live, options())).join('');
        } else {
            for (const skill of [...skills].reverse()) {
                const key = installedSkillRowKey(skill);
                const card = cards.get(key);
                if (card) {
                    // A queue placeholder has no controls to enrich. Replace only this
                    // structural transition; real cards retain drafts and disclosures.
                    if (card.hasAttribute('data-lifecycle-virtual') !== Boolean(skill.lifecycle_virtual)
                        || (card.querySelector('.skills-card-name')?.tagName === 'BUTTON') !== opensDetail(skill)) {
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
                    cards.delete(key);
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
        retirePendingSheet(skills);
        browse?.repaintStates();
        schedulePoll();
    }

    // The preview of a queue-only install describes one queue job. The repaint whose rows no longer show that same job
    // as a pending install (failed, replaced, gone, or now a listed skill) closes it; focus then follows its binder to the
    // title this row key has now (`titleOf` in openInstalledSheet), if one is shown.
    function newestJobId(name) {
        const id = lifecycleEventsFromQueue(snapshot?.queue).filter((event) => event.target === name).pop()?.id;
        return typeof id === 'string' ? id : '';
    }
    function retirePendingSheet(rows) {
        if (!sheet || sheet.closed || !sheetPending) return;
        const row = rows.find((skill) => installedSkillRowKey(skill) === sheetPending.key);
        if (row?.lifecycle_virtual && opensDetail(row) && newestJobId(row.name) === sheetPending.job) return;
        sheet.close();
    }

    function clearListingTimers() {
        clearTimeout(pollTimer);
        clearTimeout(hintTimer);
        hintTimer = 0;
    }

    // The rail and a view shown together share one read. Refresh supersedes an older read; a hint (a change, the tab
    // shown again, the socket reopened) is read once, for the rail and the views alike, after the read on its way has
    // answered (`scheduleHintRead`): the section asks for its rows only as its list comes on screen (`rowsShown`).
    let listingOnWay = null;
    // The listing is wanted while a skill view or the section's list of pages is on screen.
    const needsListing = () => !destroyed && !document.hidden && (visible || section.rowsOnScreen());
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
        clearListingTimers();
        const hint = pendingListingHint;
        // A new listing reads its own optional facts; it does not leave the previous reads in flight.
        listingReadController?.abort();
        const controller = listingReadController = new AbortController();
        listingReadState = 'pending';
        const current = () => controller === listingReadController && !destroyed;
        const earlierTicks = snapshot?.failureTicks;
        const next = { rawSkills: null, live: {}, queue: snapshot?.queue || null, failureTicks: earlierTicks, visibleIds: snapshot?.visibleIds, githubTokenConfigured: snapshot?.githubTokenConfigured };
        const optional = { state: 'loading', queue: 'loading' };
        const showReads = () => {
            // Optional facts describe the new list only after it has been published.
            if (!current() || snapshot !== next) return;
            const notes = Object.entries(optional).filter(([, value]) => value === 'failed')
                .map(([name]) => (name === 'state' ? 'Skill settings' : 'Lifecycle progress'));
            status.dataset.tone = notes.length ? 'warn' : 'muted';
            status.textContent = notes.length ? `${notes.join(' and ')} could not be refreshed. Refresh to retry; earlier facts are kept.` : '';
        };
        readStateJson({ signal: controller.signal }).then((data) => {
            if (!current()) return;
            const ok = typeof data?.github_token_configured === 'boolean';
            optional.state = ok ? 'ready' : 'failed';
            if (ok) next.githubTokenConfigured = data.github_token_configured;
            if (snapshot === next) paintInstalled();
            showReads();
        }, () => { if (current()) { optional.state = 'failed'; showReads(); } });
        apiClient.skillLifecycleQueue({ signal: controller.signal }).then((data) => {
            if (!current()) return;
            const ok = Array.isArray(data?.events) || (data?.active && typeof data.active === 'object');
            optional.queue = ok ? 'ready' : 'failed';
            if (ok) {
                const events = lifecycleEventsFromQueue(data);
                const connected = (job) => job?.kind === 'connect_mcp' && job.status === 'succeeded';
                const previous = lifecycleEventsFromQueue(next.queue).filter(connected);
                const justConnected = events.some((job) => connected(job) && !previous.some((old) => old.id === job.id));
                next.queue = data;
                next.failureTicks = tickFailures(events, earlierTicks, () => ++readTick);
                next.visibleIds = visibleJobIds(events);
                if (next.answeredFailures) next.answeredFailures = carryAnswers(next.answeredFailures, next.failureTicks);
                // The ambiguity ids follow the queue's visible records, not its failed ones: a job still in flight is still listed.
                if (next.nameLevelFailures) next.nameLevelFailures = carryAnswers(next.nameLevelFailures, next.visibleIds, writeNameLevelFailures);
                if (justConnected) readConnectedServers();
            }
            if (snapshot === next) paintInstalled();
            showReads();
        }, () => { if (current()) { optional.queue = 'failed'; showReads(); } });
        if (!quiet && !snapshot) sayLoading();
        try {
            const listingTick = ++readTick;
            const primary = await withinRequestLifetime(fetchSkills, { signal: controller.signal });
            if (!current()) return;
            next.listingTick = listingTick;
            next.rawSkills = primary.skills;
            next.live = primary.live;
            next.deployment = primary.deployment;
            lastListingRead = Date.now();
            listingReadState = 'idle';
            if (pendingListingHint === hint) pendingListingHint = null;
        } catch (err) {
            if (!current()) return;
            listingReadState = 'failed';
            installedFailed = true;
            section.paintAnswers();
            emptyEl.hidden = true;
            const reason = err?.name === 'StateReadEnded' && err.reason === 'expired' ? 'the request took too long' : err.message || err;
            status.dataset.tone = 'danger';
            status.textContent = `Couldn’t refresh installed skills: ${reason}.${snapshot ? ' Showing the previous list. Refresh to try again.' : ' Refresh to try again.'}`;
            // New hints get their one trailing read even if the primary stalled. An old, unanswered hint
            // still retries at the slower failure cadence, not in a tight loop.
            if (pendingListingHint !== hint) scheduleHintRead();
            schedulePoll();
            return;
        }
        const firstPaint = !snapshot;
        // Decisions already made carry over while no accepted queue read has dropped their job — from the failed marks for
        // the answers, from the visible ones for the ambiguity, so a job still in flight is not forgotten before it can
        // fail. A first snapshot starts from this tab's copy: a reload does not bring back a failure a listing once
        // answered, nor move an attempt seen beside two folders onto the one left.
        next.answeredFailures = carryAnswers(snapshot?.answeredFailures ?? readAnsweredFailures(), next.failureTicks);
        next.nameLevelFailures = carryAnswers(snapshot?.nameLevelFailures ?? readNameLevelFailures(), next.visibleIds, writeNameLevelFailures);
        snapshot = next;
        installedFailed = false;
        paintInstalled({ replace: firstPaint });
        showReads();
        scheduleHintRead();
        return true;
    }

    function scheduleHintRead() {
        const canRead = () => needsListing() && listingReadState !== 'pending';
        // The first hint owns the deadline; newer hints cannot supersede an active read.
        if (!pendingListingHint || hintTimer || !canRead()) return;
        hintTimer = setTimeout(() => {
            hintTimer = 0;
            if (canRead()) readInstalled({ quiet: true });
        }, 300);
    }

    // Local progress only: pending work or a listing hint not yet covered by a successful read.
    function schedulePoll() {
        clearTimeout(pollTimer);
        if (destroyed || !visible || document.hidden || listingReadState === 'pending' || (!snapshot && !pendingListingHint)) return;
        const working = inFlightCount(lifecycleEventsFromQueue(snapshot?.queue)) > 0;
        const listingTools = snapshot?.rawSkills?.some((skill) => (skill.mcp_servers || [])
            .some((server) => server.configured && !server.listed && !server.error));
        const rows = projected();
        const reviewing = reviewingSkills.size > 0 || rows.some((skill) => installedSkillModel(skill).reviewing);
        const exiting = snapshot?.rawSkills?.some((skill) => Object.keys(skill.retained_companions || {}).length);
        const listed = new Map((snapshot?.rawSkills || []).map((skill) => [skill.name, skill]));
        const facts = pageModuleFacts();
        const confirming = rows.some((row) => awaitsNewerListing(row, listed.get(row.name), facts));
        if (!pendingListingHint && !working && !reviewing && !listingTools && !exiting && !confirming) return;
        const retrying = listingReadState === 'failed';
        pollTimer = setTimeout(async () => {
            if (destroyed || !visible || document.hidden) return;
            if (retrying || pendingListingHint || working || listingTools || confirming || Date.now() - lastListingRead >= REVIEW_POLL_MS) {
                await readInstalled({ quiet: true });
            } else schedulePoll();
        }, retrying ? REVIEW_POLL_MS : WORK_POLL_MS);
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
        // An attempt beside the folder of its name has no Details: the folder's card is the one that opens.
        const card = container.querySelector(`:scope > .skills-card[data-skill="${CSS.escape(name)}"]:not([data-lifecycle-virtual])`);
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
        const card = opener && (opener.closest('.skills-card') || (activeMenu?.popover.contains(opener) && activeMenu.card));
        const key = card ? cardRowKey(card) : name;
        const skill = projected().find((row) => installedSkillRowKey(row) === key);
        if (!skill || !opensDetail(skill)) return;
        const catalogItem = catalogEntry(skill);
        const item = catalogItem || {
            id: `installed/${name}`, source_id: skill.install?.source?.id || '', source_name: installedSourceLabel(skill),
            name, title: skillTitle(skill), description: skill.description || '', version: skill.version,
            official: Boolean(skill.install?.official), signals: null,
            category: skill.category || [],
        };
        sheet?.close();
        // What it can use is this loaded row's parsed manifest and grants, never the catalog's entry; a pending install, or
        // a folder whose manifest was not read, declares nothing we may state.
        const access = skill.lifecycle_virtual || manifestUnread(skill) ? null
            : { permissions: skill.permissions || [], needs_keys: skill.grants?.requested_keys || [] };
        // The source's entry has the full text; a registry install has no entry to read back.
        const origin = skill.install?.source;
        // A queue-only install's title may be replaced while its preview is open, so closing returns to the title this row
        // key has then (shown, never a hidden card or another key's); the job's own id says when it stops being this one.
        const pendingKey = skill.lifecycle_virtual ? installedSkillRowKey(skill) : '';
        const job = pendingKey ? newestJobId(name) : '';
        const titleOf = () => Array.from(container.querySelectorAll(':scope > .skills-card'))
            .find((listed) => cardRowKey(listed) === pendingKey && !listed.hidden)?.querySelector('.skills-card-name') || null;
        sheet = openSkillDetailSheet({
            item, installed: skill, returnFocus: pendingKey ? titleOf : opener, access,
            readDetail: origin?.id && origin.kind !== 'registry' ? ({ signal } = {}) => skillSourcesApi.item(origin.id, name, { signal }) : null,
        });
        sheetPending = job ? { key: pendingKey, job } : null;
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
        // The folder's own card is the skill asked for, and only it moves the owner's search and filter. An attempt of
        // the name alone is a queue record with nothing installed yet: it is brought into view where it stands.
        const entry = `.skills-card[data-skill="${CSS.escape(requestedSkill)}"]`;
        const card = container.querySelector(`${entry}:not([data-lifecycle-virtual])`);
        if (!card) {
            const attempt = container.querySelector(entry);
            if (attempt && !attempt.hidden) attempt.scrollIntoView({ block: 'nearest' });
            return;
        }
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
    // cannot settle it: the requested card may just have been installed. (`beforeOpen` is the read that was the
    // newest when the request began: each listing read has its own controller.)
    function requestInstalledSkill(name, beforeOpen) {
        requestedSkill = name;
        void readInstalled({ quiet: true, join: listingReadController !== beforeOpen });
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
            const beforeOpen = listingReadController;
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
        const listing = snapshot && !pendingListingHint ? Promise.resolve() : readInstalled({ quiet: Boolean(snapshot), join: true });
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
            if (!destroyed) showToast(`Skills view failed: ${err.message || err}`, 'danger');
        } finally {
            if (!destroyed) {
                refreshBtn.disabled = false;
                refreshBtn.textContent = 'Refresh';
            }
        }
    };
    const invalidateInstalled = () => {
        if (destroyed) return;
        // A successful read only consumes the hint it started with.
        pendingListingHint = Symbol();
        if (!needsListing()) listingOnWay = null;  // on return, read past the change instead of joining a read started before it
        scheduleHintRead();
    };
    // A lifecycle end reaches this page from the socket and from this window's own actions (contributions.js); health
    // and a reopened socket come from the socket alone.
    const offLifecycle = contributions.onLifecycle(invalidateInstalled);
    const stopSync = typeof ctx.ws?.on === 'function' ? [
        ctx.ws.on('extension_lifecycle', (event) => {
            if (event?.action === 'connect_mcp') readConnectedServers();
            invalidateInstalled();
        }),
        ctx.ws.on('extension_health_changed', invalidateInstalled),
        ctx.ws.on('open', invalidateInstalled),  // a lost hint is recovered by re-reading, not replayed
    ] : [];
    const onVisibilityChange = () => {
        if (document.hidden) { ++openRequest; requestedSkill = ''; clearListingTimers(); }
        else schedulePoll();
        invalidateInstalled();
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
        const beforeOpen = listingReadController;
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
        clearListingTimers();
        listingReadController?.abort();
        closeMenus();
        sheet?.close();
        browse.destroy();
        sourcesView.destroy();
        section.destroy();
        stopSync.forEach((off) => off?.());
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
