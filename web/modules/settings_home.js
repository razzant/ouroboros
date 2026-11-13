/** The home of Settings (docs/DESIGN.md "Settings"): Ouroboros's card, search, then six groups of rows, each a
 * question with its current answer; a row opens its page, or leaves Settings by its door (↗). The shell
 * (settings.js) shows it alone, or beside the page opened from it in the wide layout, where the row of that page is
 * marked current.
 *
 * Every answer comes from a read that is already made or passive: the settings document, the app's own `/api/state`
 * snapshot, the shared accounts store as it stands, and the status reads of tool servers, skills, places, updates,
 * storage and start at sign-in, none of which starts anything. Showing the home starts no daemon, no provider call,
 * no catalogue read and no disk walk; a fact that was not read says so. Ouroboros's card lists what waits for the
 * owner here (`needRows`), in its voice; search matches the words on screen and the words they replaced
 * (`searchSettings`), and its last line hands the words to Ouroboros as an unsent draft in Main. */

import { apiClient, fetchJson } from './api_client.js';
import { confirmAndSendRestart, headerBudgetPresentation } from './chat_activity.js';
import { routeSource, shortName } from './chat_model_routes.js';
import { accountName, accountRows, claudexorStatus, familyLabel, READ_OK } from './claudexor_status_store.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { normalizeConsciousness, parseRoster } from './consciousness_api.js';
import { headlineView } from './consciousness_dashboard.js';
import { consciousnessMark } from './consciousness_words.js';
import { refreshActionKind, verificationBadge } from './harness_accounts.js';
import { harnessPresentation } from './harness_presentation.js';
import { bootPhase, startingStatus, supervisorReady } from './host_starting_status.js';
import { fmt, translateString } from './i18n.js';
import { readNotifyPrefs } from './notifications.js';
import { PAGE_ICONS, SHELL_ICONS, controlIcon, fluentIcon } from './page_icons.js';
import { readSkills, skillForms } from './settings_extension_forms.js';
import { isDoorId, settingsPageFor } from './settings_pages.js';
import { policyValueLabel } from './settings_policy.js';
import { HAND_WRITTEN, HOME_ROWS, ROWS, SEARCH_DOORS } from './settings_rows.js';
import { getSpeech } from './speech_providers.js';
import { routeKeyProblem } from './route_editor_primitives.js';
import { readStateSnapshot } from './state_snapshots.js';
import { paintStatusMark, statusMarkHtml } from './status_mark.js';
import { formatBytes } from './storage_format.js';
import { availableSubagentsLoadValue } from './subagents_settings.js';
import { updateVerdict } from './updates.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export { isDoorId };

/** A page's title for `path`: its module's words, else the home row that opens it, else the path. */
export function settingsTitle(path) {
    const page = settingsPageFor(path);
    const named = typeof page?.title === 'function' ? page.title(path) : page?.title;
    return String(named || HOME_ROWS.find((row) => row.to === path)?.name || path);
}

// A fixed answer in the install's language; composed ones go through `fmt` with their numbers and names.
const NOT_CHECKED = 'Not checked';
const truthy = (value) => value === true || /^(true|1|yes|on)$/i.test(String(value ?? '').trim());
const count = (n, one, many) => fmt(n === 1 ? one : many, { n });

/** An account that needs sign-in again: enabled, and its login was checked and did not pass (harness_accounts.js). */
const needsSignIn = (row) => row.enabled !== false && verificationBadge(row).tone === 'error';
const serverName = (server) => String(server?.name || server?.label || server?.id || server?.slug || '');
const toolNeedsSignIn = (server) => server?.enabled !== false && Boolean(server?.sign_in)
    && (server.sign_in.state !== 'signed_in' || server.sign_in.reason === 'insufficient_scope');

/**
 * What waits for the owner here, one row each: `{ kind, id, text, note, owner, action, open | run }`. `facts` holds what
 * was read (`undefined`: not read yet, nothing is said; `null`: the read failed): `accounts` `{ read, rows, kind }`
 * from the shared accounts store, `tools` the tool servers' status, `settings` the settings document. An account row
 * stands only over an accounts facet that was really read. The list never says that nothing waits, and never what
 * was not read: that is the answer of the row it concerns.
 */
export function needRows(facts = {}) {
    const rows = [];
    const { accounts, tools, settings } = facts;
    // Only what needs the owner stands here: a fact that was not read, or whose read failed, is said by the row it
    // concerns (its answer on the home or its page), never as a line of the card with a Check.
    if (accounts?.read === READ_OK) {
        const byHarness = new Map();
        for (const row of accounts.rows.filter(needsSignIn)) byHarness.set(row.harness, [...(byHarness.get(row.harness) || []), row]);
        for (const [harness, list] of byHarness) {
            const label = familyLabel(harness, accounts.snapshot);
            rows.push({ kind: 'sign-in', id: harness, owner: true, action: 'Sign in', open: `accounts/${harness}`,
                text: list.length === 1 ? fmt('{label} account {name} needs sign-in again', { label, name: accountName(list[0]) })
                    : fmt('{n} {label} accounts need sign-in again', { n: list.length, label }) });
        }
    }
    for (const server of (tools?.servers || []).filter(toolNeedsSignIn)) {
        rows.push({ kind: 'tool-sign-in', id: String(server.id ?? serverName(server)), owner: true, action: 'Open',
            open: `tools/${encodeURIComponent(String(server.id ?? serverName(server)))}`,
            text: fmt('MCP server {name} needs sign-in', { name: serverName(server) }) });
    }
    const restart = settings?._meta?.restart_state;
    if (restart?.restart_required || restart?.restart_source_unknown_keys?.length) {
        const named = (restart.restart_keys || []).map((key) => ROWS.find((row) => row.keys.includes(key))?.name).filter(Boolean);
        const waiting = (restart.restart_keys || []).length;
        rows.push({ kind: 'restart', id: 'restart', owner: true, action: 'Restart', run: 'restart',
            text: !waiting ? fmt('The saved server address may need a restart')
                : named.length === waiting ? fmt('Restart to apply: {names}', { names: [...new Set(named)].join(', ') })
                    : count(waiting, 'Restart to apply {n} saved setting', 'Restart to apply {n} saved settings') });
    }
    if (restart?.local_model?.pending_keys?.length) {
        rows.push({ kind: 'local-model', id: 'local-model', owner: true, action: 'Open', open: 'models/local',
            text: fmt('Local model: saved changes wait for its own Stop, then Start') });
    }
    return rows;
}

/** Ouroboros's state for its card: `{ text, tone }`, from the socket and the last state snapshot. */
export function beingState({ connected = true, state = null } = {}) {
    if (!connected) return { text: fmt('Reconnecting…'), tone: 'muted' };
    if (supervisorReady(state) === false) {
        const starting = startingStatus(bootPhase(state));
        return { text: starting.text, tone: starting.kind === 'failed' ? 'danger' : 'muted' };
    }
    return { text: fmt('Online'), tone: 'ok' };
}

function accessWords(settings) {
    const access = settings?._meta?.policy_state?.access;
    if (!access) return fmt('Not reported');
    const now = policyValueLabel(access.current_process || access.effective);
    return access.restart_required ? fmt('{now} now, {later} after restart', { now, later: policyValueLabel(access.configured) })
        : fmt('Access: {now}', { now });
}

function agentsWords(settings) {
    const roster = parseRoster(availableSubagentsLoadValue(settings));
    if (!roster) return fmt(NOT_CHECKED);
    if (roster.enabled === false) return fmt('Delegation off');
    return fmt('{n} of {total} on', { n: roster.items.filter((item) => item?.enabled !== false).length, total: roster.items.length });
}

function accountsWords(accounts, settings) {
    // A document that does not list the connections says nothing about keys: no count is printed for it.
    const keys = Array.isArray(settings?._meta?.connections) ? settings._meta.connections.filter((row) => row?.configured).length : null;
    const keyWords = keys === null ? '' : count(keys, '{n} API key', '{n} API keys');
    // A count of accounts comes only from a roster that was really read and holds accounts; a store that was not
    // read (or a daemon that merely runs) prints no count, and the row says its API keys alone.
    const read = accounts?.read === READ_OK && accounts.rows.length > 0;
    if (!read && keys === null) return undefined;
    const known = read ? count(accounts.rows.length, '{n} account', '{n} accounts') : '';
    // An account that needs sign-in again is the owner's move, as its line of the card says.
    return { text: [known, keyWords].filter(Boolean).join(', '), owner: read && accounts.rows.some(needsSignIn) };
}

function toolsWords(tools) {
    if (tools.enabled === false) return fmt('MCP servers off');
    const servers = Array.isArray(tools.servers) ? tools.servers : [];
    if (!servers.length) return fmt('No MCP servers');
    // A server whose configuration could not be loaded is listed without its runtime facts: it is not on.
    const on = servers.filter((server) => server?.enabled === true && Object.hasOwn(server, 'tool_count')).length;
    return { text: fmt(servers.length === 1 ? '{on} of {n} server on' : '{on} of {n} servers on', { on, n: servers.length }),
        owner: servers.some(toolNeedsSignIn) };
}

function updateWords(update) {
    const verdict = updateVerdict(update);
    const words = { unchecked: NOT_CHECKED, check_failed: 'Could not check', unmanaged: 'Managed updates unavailable' }[verdict.state];
    return { text: words ? fmt(words) : String(verdict.headline || '').replace(/\.$/, ''), owner: ['available', 'available_unsafe'].includes(verdict.state) };
}

const START_WORDS = { on: 'Starts at sign-in', off: 'Starts by hand', other_copy: 'Another copy starts at sign-in', disabled_by_os: 'Start at sign-in is off in the system' };
/** This computer: how it starts where the host says so (the desktop app); on a host that does not (a browser, a
 *  server), who can open it, as the settings read reports what listens now. */
function computerWords(autostart, settings) {
    if (START_WORDS[autostart?.state]) return fmt(START_WORDS[autostart.state]);
    const reach = settings?._meta?.reachability;
    if (reach) return fmt(reach === 'loopback_only' ? 'Open only here' : 'Open to your network');
    return autostart === null || settings === null ? fmt(NOT_CHECKED) : undefined;
}

/**
 * The current answer of every home row, keyed by what the row opens: `{ text, owner }`, `owner` when the next move
 * is the owner's (a stop by the spending limit, a held consciousness, an update that waits, a sign-in). A row whose fact is not
 * read yet has no entry and keeps what it shows; one whose read failed says "Not checked". The Accounts row counts only
 * what was read: over an accounts roster that was not read, or whose read failed, it says its API keys alone, and it
 * has no entry when the settings document does not list them either.
 */
// Main's answer on the home: its model, or what keeps it from sending (a key choice it waits for, a key that cannot
// send), in the owner's colour.
function modelsAnswer(s) {
    const profiles = s._meta?.setup_contract?.providerProfiles || {};
    if (Array.isArray(s._meta?.main_connection_choice) && s._meta.main_connection_choice.length > 1) return { text: fmt('Choose its key'), owner: true };
    const problem = truthy(s.USE_LOCAL_MAIN) ? '' : routeKeyProblem(s.OUROBOROS_MODEL, s, profiles);
    if (problem) return { text: fmt({ off: 'Its key is off', removed: 'Its key was removed', 'no key': 'No key for it' }[problem]), owner: true };
    return shortName({ model: String(s.OUROBOROS_MODEL || ''), use_local: truthy(s.USE_LOCAL_MAIN) }) || fmt('Not chosen');
}

export function homeAnswers(facts = {}) {
    const { settings, state, accounts, tools, skills, places, update, storage, autostart, device = {} } = facts;
    const out = {};
    const say = (to, fact, words) => {
        if (fact === undefined) return;
        const answer = fact === null ? fmt(NOT_CHECKED) : words(fact);
        if (answer !== undefined) out[to] = typeof answer === 'string' ? { text: answer, owner: false } : answer;
    };
    say('models', settings, modelsAnswer);
    say('agents', settings, agentsWords);
    say('dashboard:consciousness', state, (s) => {
        const snapshot = normalizeConsciousness(s.bg_consciousness_state || {}, { enabled: s.bg_consciousness_enabled });
        return { text: headlineView(snapshot).text, owner: consciousnessMark(snapshot).hue === 'owner' };
    });
    say('dashboard:evolution', state, (s) => fmt(s.evolution_enabled === true ? 'On' : s.evolution_enabled === false ? 'Off' : 'State unknown'));
    say('page:circles', places, (p) => (Array.isArray(p.places) ? count(p.places.length, '{n} place', '{n} places') : fmt(NOT_CHECKED)));
    say('rules', settings, (s) => fmt(String(s.OUROBOROS_SAFETY_OWNER_RULES ?? '').trim() ? 'Written' : 'Empty'));
    say('permissions', settings, accessWords);
    say('dashboard:costs', state, (s) => {
        const budget = headerBudgetPresentation(s);
        return { text: budget.label, owner: budget.attention === true };
    });
    const accountsAnswer = accountsWords(accounts, settings);
    if (accountsAnswer !== undefined) out.accounts = accountsAnswer;
    say('tools', tools, toolsWords);
    say('page:skills', skills, (s) => (Array.isArray(s.skills) ? fmt('{n} installed', { n: s.skills.length }) : fmt(NOT_CHECKED)));
    say('appearance', device.theme, (theme) => (['system', 'light', 'dark'].includes(theme) ? fmt({ system: 'System', light: 'Light', dark: 'Dark' }[theme]) : undefined));
    say('notifications', device.notify, (prefs) => fmt(!prefs.enabled ? 'Off on this device' : prefs.sound ? 'On, with sound' : 'On, silent'));
    say('voice', device.dictation, (engine) => (engine ? fmt('Dictation: {engine}', { engine }) : fmt('Dictation not chosen')));
    // Until the update status answers, the row says the version that runs, which the state snapshot already holds.
    if (update === undefined && state?.version) out['dashboard:updates'] = { text: fmt('Version {version}', { version: String(state.version) }), owner: false };
    else say('dashboard:updates', update, updateWords);
    say('dashboard:storage', storage, (s) => (s.count ? formatBytes(s.count.bytes) : fmt('Not measured')));
    const computer = autostart === undefined ? undefined : computerWords(autostart, settings);
    if (computer !== undefined) out.computer = { text: computer, owner: false };
    return out;
}

const fold = (text) => String(text ?? '').normalize('NFKC').toLocaleLowerCase().trim();

/** Everything search can find, each with the words it answers to: the home's rows and the doors to pages no home row
 *  opens, every settings row (its name, meta, aliases, keys and its page's title) and the forms skills registered. */
export function searchEntries({ rows = ROWS, handWritten = HAND_WRITTEN, home = HOME_ROWS, doors = SEARCH_DOORS, forms = skillForms(), title = settingsTitle } = {}) {
    const entries = [...home, ...doors].map((row) => ({ id: row.id, name: row.name, where: '', to: row.to, row: '', glyph: row.glyph,
        words: [row.name, ...(row.aliases || [])] }));
    for (const row of [...rows, ...handWritten]) {
        const where = title(row.page);
        entries.push({ id: row.id, name: row.name || row.id, where, to: row.page, row: row.id,
            glyph: home.find((door) => door.to === String(row.page).split('/')[0])?.glyph || 'settings',
            words: [row.name, row.meta, ...(row.aliases || []), ...(row.keys || []), row.id, where] });
    }
    for (const form of forms) {
        for (const section of form.sections) {
            entries.push({ id: `skill:${form.skill}:${section.section_id || section.key || ''}`, name: String(section.title || form.name),
                where: form.name, to: `skill/${encodeURIComponent(form.skill)}`, row: '', glyph: 'skills',
                words: [section.title, form.name, form.skill] });
        }
    }
    return entries;
}

/** The entries whose words hold `query`, literally and case-folded, in English or as the install's translation
 *  shows them. No intents, no synonyms: an old name finds a row because the row lists it. */
export function searchSettings(query, entries = searchEntries(), translate = translateString) {
    const needle = fold(query);
    if (!needle) return [];
    return entries.filter((entry) => entry.words.some((word) => word
        && (fold(word).includes(needle) || fold(translate(String(word))).includes(needle))));
}

/** The words as an unsent draft in Main: Main is shown, and a draft that is already there stays as it is. */
export async function askOuroboros(words, { showPage, doc = globalThis.document } = {}) {
    if (await showPage?.('chat') === false) return false;
    const input = doc?.querySelector?.('#page-chat #chat-input');
    if (!input) return false;
    if (!String(input.value || '').trim()) {
        input.value = String(words);
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    input.focus?.({ preventScroll: true });
    return true;
}

/* ---- Drawing ----------------------------------------------------------------------------------------------- */

const OUT_ICON = fluentIcon('arrow-up-right', 14, 'regular');
const IN_ICON = fluentIcon('chevron-right', 14, 'regular');
// Glyphs the product's set (page_icons.js) does not carry: Fluent Filled (a row names a place or a thing).
// Models wears the cube (never a sparkle); Costs and Permissions wear a plain mark, the row's words say the value.
// Agents wears the network of one and its helpers (the owner's choice of 27.10 for subagents), Updates the arrow up.
const GLYPHS = Object.fromEntries(Object.entries({
    key: 'key',
    rules: 'text-bullet-list-square',
    growth: 'branch-fork',
    sun: 'dark-theme',
    moon: 'dark-theme',
    bell: 'alert',
    refresh: 'arrow-circle-up',
    agents: 'organization',
    activity: 'timeline',
    drive: 'hard-drive',
    monitor: 'desktop',
    cube: 'cube',
    wallet: 'wallet',
    shield: 'shield-checkmark',
}).map(([name, fluent]) => [name, fluentIcon(fluent, 18)]));
const glyphHtml = (name) => GLYPHS[name] || PAGE_ICONS[name] || controlIcon(name, 18);
// The chosen dictation engine's name ('' when none is chosen), from the page's speech hub as it stands.
function dictationWord() {
    try {
        const chosen = getSpeech().selected('stt');
        return String(chosen?.label || chosen?.key || '');
    } catch { return undefined; }
}

const markHtml = (presentation) => `<svg width="18" height="18" viewBox="${presentation.viewBox}" fill="currentColor" aria-hidden="true"><path d="${presentation.path}"/></svg>`;

/** The glyphs that say a fact: Main's source (its harness mark, else the cube) and the theme. */
function liveGlyphs({ settings, device = {} } = {}) {
    const out = {};
    if (settings) {
        const source = routeSource({ model: String(settings.OUROBOROS_MODEL || ''), use_local: truthy(settings.USE_LOCAL_MAIN) });
        const brand = source.startsWith('subscription:') ? harnessPresentation(source.slice('subscription:'.length)) : null;
        out.models = brand && !brand.generic ? markHtml(brand) : glyphHtml('cube');
    }
    if (device.theme) out.appearance = glyphHtml(device.theme === 'light' ? 'sun' : 'moon');
    return out;
}

function doorHtml({ to, name, glyph, answer = '', result = '' }) {
    return `<button type="button" class="settings-door section-row" data-settings-open="${escapeHtml(to)}"${isDoorId(to) ? ' data-settings-door-out data-section-out' : ''}${result ? ` data-settings-result="${escapeHtml(result)}"` : ''}>`
        + `<span class="settings-door-glyph section-row-glyph" aria-hidden="true" data-settings-glyph>${glyphHtml(glyph)}</span>`
        + `<span class="settings-door-words section-row-words"><span class="settings-door-title section-row-title">${escapeHtml(name)}</span>`
        + `<span class="settings-door-answer section-row-answer" data-settings-answer${result ? '' : ' data-i18n-fmt="settings.answer"'}>${escapeHtml(answer)}</span></span>`
        + `<span class="settings-door-end section-row-end" aria-hidden="true">${result ? '' : statusMarkHtml('settings-door-mark')}${isDoorId(to) ? OUT_ICON : IN_ICON}</span></button>`;
}

const groupHtml = (title, body) => `<section class="settings-group section-group">${title ? `<h3 class="settings-group-title section-group-title">${escapeHtml(title)}</h3>` : ''}`
    + `<div class="ui-card settings-door-list section-rows">${body}</div></section>`;

export function homeGroupsHtml(rows = HOME_ROWS) {
    return [...new Set(rows.map((row) => row.group))]
        .map((group) => groupHtml(group, rows.filter((row) => row.group === group).map(doorHtml).join(''))).join('');
}

/** Search results as rows, then the line that asks Ouroboros with the words. */
export function resultsHtml(query, found) {
    const hits = found.map((entry) => doorHtml({ to: entry.to, name: entry.name, glyph: entry.glyph, answer: entry.where, result: entry.id })).join('');
    const ask = `<button type="button" class="settings-door section-row" data-settings-ask data-settings-door-out data-section-out><span class="settings-door-glyph section-row-glyph" aria-hidden="true">${glyphHtml('chat')}</span>`
        + `<span class="settings-door-words section-row-words"><span class="settings-door-title section-row-title" data-i18n-fmt="settings.ask">${escapeHtml(fmt('Ask Ouroboros: “{words}”', { words: query.trim() }))}</span>`
        + `<span class="settings-door-answer section-row-answer">The words wait in Main as a draft you send.</span></span><span class="settings-door-end section-row-end" aria-hidden="true">${OUT_ICON}</span></button>`;
    return `${hits ? groupHtml('Settings', hits) : '<p class="settings-search-empty">No setting has these words.</p>'}${groupHtml('', ask)}`;
}

function needRowHtml(row) {
    const words = `settings-need-${row.kind}-${row.id}`.replace(/[^\w-]/g, '-');
    return `<div class="ui-row settings-need" data-needs="${escapeHtml(row.kind)}" data-need-id="${escapeHtml(row.id)}">`
        + `<span class="settings-need-lead">${statusMarkHtml('', row.owner ? { shape: 'dot', hue: 'owner' } : { shape: 'ring', hue: 'ink' })}</span>`
        + `<p class="ui-row-title" id="${words}" data-i18n-fmt="settings.need">${escapeHtml(row.text)}</p>`
        + `${row.note ? `<p class="ui-row-meta" data-i18n-fmt="settings.need">${escapeHtml(row.note)}</p>` : ''}`
        + `<div class="ui-row-actions"><button type="button" class="btn btn-default btn-sm" data-need-action aria-describedby="${words}">${escapeHtml(row.action)}</button></div></div>`;
}

// The facts the mounted home last read (its updates and storage reads among them), so another list of the same pages
// (the Dashboard's, dashboard.js) can say the same answers without a read of its own.
let seenFacts = null;
/** What the Settings home has read so far (`{}` before it is mounted); never starts a read. */
export const homeFactsSeen = () => seenFacts || {};

/**
 * Draw the home into `root` and take its presses. `open(path, { row, opener })` opens a row's page or door; `ctx`
 * is the shell's (`ws`, `showPage`, `readSettings`, `stateSnapshots`). Answers `{ mark(path), search(words), refresh(),
 * dispose() }`: `mark` names the row whose page is shown beside the home; `search` puts words into the search field and
 * shows their results, until the first press on them (words the owner typed stay); `refresh` reads the facts again
 * and repaints.
 */
export function mountSettingsHome(root, { open, ctx = {}, store = claudexorStatus, ask = openConfirmDialog } = {}) {
    root.innerHTML = `<div class="settings-column settings-home-column">
        <section class="ui-card settings-being settings-voice" data-settings-card aria-label="Ouroboros">
            <span class="settings-being-avatar" aria-hidden="true"></span>
            <div class="settings-being-words"><span class="settings-being-name" data-i18n-skip>Ouroboros</span>
                <span class="settings-being-state"><span class="ui-status" data-being-state></span><span class="settings-being-sep" aria-hidden="true" hidden>·</span>
                    <button type="button" class="settings-being-count" data-being-count aria-expanded="false" hidden></button></span></div>
            <div class="settings-needs" data-being-needs hidden></div>
        </section>
        <div class="ui-card settings-door-list section-rows settings-overview" data-settings-overview hidden>
            <button type="button" class="settings-door section-row" data-settings-open="" data-settings-overview-row>
                <span class="settings-door-glyph section-row-glyph settings-overview-glyph" aria-hidden="true"></span>
                <span class="settings-door-words section-row-words"><span class="settings-door-title section-row-title" data-i18n-skip>Ouroboros</span>
                    <span class="settings-door-answer section-row-answer" data-overview-answer></span></span>
                <span class="settings-door-end section-row-end" aria-hidden="true"></span></button>
        </div>
        <label class="settings-search">${SHELL_ICONS.search}<input type="search" id="settings-search" placeholder="Search settings" autocomplete="off" spellcheck="false" aria-label="Search settings"></label>
        <div class="settings-home-list section-list" data-settings-list>${homeGroupsHtml()}</div>
    </div>`;
    const find = (selector) => root.querySelector(selector);
    // The card and the search field keep their own references: the shell may move them (the field into the header,
    // the card into the Ouroboros pane of landing A), and they work wherever they stand.
    const cardEl = find('[data-settings-card]');
    const [needsEl, listEl, searchEl, countEl] = [find('[data-being-needs]'), find('[data-settings-list]'), find('#settings-search'), find('[data-being-count]')];
    const [stateEl, sepEl, overviewAnswer] = [find('[data-being-state]'), find('.settings-being-sep'), find('[data-overview-answer]')];
    const facts = { device: {} };
    seenFacts = facts;
    let current = '';
    let needs = [];
    let shownQuery = '';
    let handed = '';  // the words `search` put into the field, until the owner types there
    let serial = 0;
    let busy = '';
    let needsHtml = '';

    function paintCard() {
        const being = beingState({ connected: ctx.ws?.isConnected?.() !== false, state: facts.state });
        stateEl.textContent = being.text;
        stateEl.dataset.tone = being.tone;
        needs = needRows(facts);
        const waiting = needs.filter((row) => row.owner).length;
        countEl.textContent = waiting ? count(waiting, 'One thing needs you', '{n} things need you') : '';
        // The list's own Ouroboros row (two panes from the first showing, settings.js): its state, and how many
        // things need the owner when any do.
        if (overviewAnswer) overviewAnswer.textContent = waiting ? `${being.text} · ${countEl.textContent}` : being.text;
        countEl.hidden = !waiting;
        sepEl.hidden = !waiting;
        const html = needs.map(needRowHtml).join('') + (needs.length > 2
            ? `<button type="button" class="btn btn-ghost btn-sm settings-needs-more" data-needs-more>${escapeHtml(fmt('Show all {n}', { n: needs.length }))}</button>` : '');
        if (needsHtml !== html) needsEl.innerHTML = needsHtml = html;
        needsEl.hidden = !needs.length;
        needsEl.dataset.count = String(needs.length);
        needsEl.querySelectorAll('[data-need-action]').forEach((button) => {
            const row = button.closest('[data-needs]');
            const running = busy === `${row.dataset.needs}:${row.dataset.needId}`;
            button.disabled = running;
            if (running) button.setAttribute('aria-busy', 'true'); else button.removeAttribute('aria-busy');
        });
    }

    function paintRows() {
        if (shownQuery) return;
        const answers = homeAnswers(facts);
        const glyphs = liveGlyphs(facts);
        listEl.querySelectorAll('[data-settings-open]').forEach((row) => {
            const to = row.dataset.settingsOpen;
            const answer = answers[to];
            if (answer) {
                const words = row.querySelector('[data-settings-answer]');
                if (words.textContent !== answer.text) words.textContent = answer.text;
                paintStatusMark(row.querySelector('.settings-door-mark'), answer.owner ? { shape: 'dot', hue: 'owner' } : {});
            }
            const glyph = row.querySelector('[data-settings-glyph]');
            if (glyphs[to] && glyph.dataset.glyph !== glyphs[to]) {
                glyph.dataset.glyph = glyphs[to];
                glyph.innerHTML = glyphs[to];
            }
            if (to === current) row.setAttribute('aria-current', 'page'); else row.removeAttribute('aria-current');
        });
    }

    function paintList() {
        const query = String(searchEl.value || '').trim();
        if (query !== shownQuery) {
            shownQuery = query;
            listEl.innerHTML = query ? resultsHtml(query, searchSettings(query)) : homeGroupsHtml();
            if (query) listEl.dataset.searching = '1'; else delete listEl.dataset.searching;
        }
        paintRows();
    }
    const paint = () => { paintCard(); paintList(); };

    // The accounts store as it stands: this subscriber is never a visible surface, so it arms no poll and reads nothing.
    const readAccounts = (view = { reads: store.reads, snapshot: store.snapshot, everSettled: store.everSettled }) => {
        facts.accounts = { read: view.reads?.accounts, snapshot: view.snapshot, rows: accountRows(view.snapshot),
            kind: view.everSettled ? refreshActionKind(view.snapshot) : 'refresh' };
    };
    const READS = {
        settings: () => ctx.readSettings(),
        tools: () => fetchJson('/api/mcp/status', { cache: 'no-store' }),
        skills: () => readSkills(),
        places: () => fetchJson('/api/places', { cache: 'no-store' }),
        update: () => apiClient.updateStatus(),
        storage: () => fetchJson('/api/storage', { cache: 'no-store' }),
        autostart: () => apiClient.desktopAutostart(),
    };
    async function read(name, run) {
        try { facts[name] = (await READS[name]()) ?? null; } catch { facts[name] = null; }
        if (run === serial) paint();
    }
    async function readState(run) {
        const snapshots = ctx.stateSnapshots;
        if (!snapshots?.latest?.() && snapshots?.gate) {
            const request = await snapshots.gate(false);
            if (request) await readStateSnapshot(snapshots, request);
        }
        facts.state = snapshots?.latest?.() ?? null;
        if (run === serial) paint();
    }
    function refresh() {
        const run = ++serial;
        const storage = globalThis.localStorage;
        facts.device = { theme: globalThis.window?.ouroTheme?.choice, notify: readNotifyPrefs(storage), dictation: dictationWord() };
        facts.state = ctx.stateSnapshots?.latest?.() || facts.state;
        readAccounts();
        paint();
        void readState(run);
        return Promise.all(Object.keys(READS).map((name) => read(name, run)));
    }

    async function runNeed(row) {
        busy = `${row.kind}:${row.id}`;
        paintCard();
        try {
            if (row.run === 'restart') await confirmAndSendRestart({ openConfirmDialog: ask, ws: ctx.ws });
            else if (row.run) await read(row.run, serial);
        } finally {
            busy = '';
            paint();
        }
    }

    const onClick = (event) => {
        // The card takes its own presses (it may stand outside the home); the home's handler leaves them to it.
        if (event.currentTarget === root && cardEl?.contains?.(event.target)) return;
        const hit = (selector) => event.target.closest?.(selector);
        if (hit('[data-being-count], [data-needs-more]')) {
            const all = needsEl.dataset.all !== '1';
            if (all) needsEl.dataset.all = '1'; else delete needsEl.dataset.all;
            countEl.setAttribute('aria-expanded', String(all));
            return;
        }
        const need = hit('[data-need-action]')?.closest('[data-needs]');
        if (need) {
            const row = needs.find((each) => each.kind === need.dataset.needs && each.id === need.dataset.needId);
            if (row?.open) open(row.open, { opener: hit('[data-need-action]') });
            else if (row) void runNeed(row);
            return;
        }
        // Words the shell searched for an address no page has (`search`) served their purpose with the first press on
        // their results: the field is emptied, so a later return shows the home. Words the owner typed stay.
        const spend = () => {
            if (!handed || searchEl.value !== handed) return;
            const held = listEl.contains(root.ownerDocument?.activeElement);  // beside the page it opened, the pressed row keeps the focus
            searchEl.value = handed = '';
            paintList();
            if (held) listEl.querySelector('[aria-current="page"]')?.focus?.({ preventScroll: true });
        };
        if (hit('[data-settings-ask]')) { void askOuroboros(shownQuery, { showPage: ctx.showPage }); spend(); return; }
        const row = hit('[data-settings-open]');
        if (!row || !root.contains(row)) return;
        const found = row.dataset.settingsResult ? searchSettings(shownQuery).find((entry) => entry.id === row.dataset.settingsResult) : null;
        open(row.dataset.settingsOpen, { row: found?.row || '', opener: row });
        spend();
    };
    const onType = () => { handed = ''; paintList(); };
    // A setting saved while the home is on screen may start or end a wait for a restart: the document is read again.
    const onSaved = () => { if (root.getClientRects?.().length) void read('settings', serial); };
    const offStore = store.subscribe((view) => { readAccounts(view); paintCard(); paintRows(); }, { visible: () => false });
    root.addEventListener('click', onClick);
    cardEl?.addEventListener('click', onClick);
    searchEl.addEventListener('input', onType);
    globalThis.window?.addEventListener?.('ouro:settings-updated', onSaved);
    paint();
    return {
        mark(path) {
            current = String(path || '').split('/')[0];
            paintRows();
        },
        search(words) {
            searchEl.value = handed = String(words || '');
            paintList();
            searchEl.parentElement?.scrollIntoView?.({ block: 'start' });
        },
        refresh,
        dispose() {
            serial += 1;
            offStore();
            root.removeEventListener('click', onClick);
            cardEl?.removeEventListener('click', onClick);
            searchEl.removeEventListener('input', onType);
            globalThis.window?.removeEventListener?.('ouro:settings-updated', onSaved);
        },
    };
}
