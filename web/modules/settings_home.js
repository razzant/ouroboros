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
import { PAGE_ICONS, SHELL_ICONS, controlIcon, strokeIcon } from './page_icons.js';
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
const ACCESS_LEVELS = ['light', 'advanced', 'pro', 'cyber_pro'];

/** An account that needs sign-in again: enabled, and its login was checked and did not pass (harness_accounts.js). */
const needsSignIn = (row) => row.enabled !== false && verificationBadge(row).tone === 'error';
const serverName = (server) => String(server?.name || server?.label || server?.id || server?.slug || '');
const toolNeedsSignIn = (server) => server?.enabled !== false && Boolean(server?.sign_in)
    && (server.sign_in.state !== 'signed_in' || server.sign_in.reason === 'insufficient_scope');

/**
 * What waits for the owner here, one row each: `{ kind, id, text, note, owner, action, open | run }`. `facts` holds what
 * was read (`undefined`: not read yet, nothing is said; `null`: the read failed): `accounts` `{ read, rows, kind }`
 * from the shared accounts store, `tools` the tool servers' status, `settings` the settings document. An account row
 * stands only over an accounts facet that was really read; a facet that was not is a neutral row with its Check
 * (`kind: 'wake'` when only starting the agent daemon can read it, which the row then says). The list never says that
 * nothing waits.
 */
export function needRows(facts = {}) {
    const rows = [];
    const { accounts, tools, settings } = facts;
    if (accounts?.read === READ_OK) {
        const byHarness = new Map();
        for (const row of accounts.rows.filter(needsSignIn)) byHarness.set(row.harness, [...(byHarness.get(row.harness) || []), row]);
        for (const [harness, list] of byHarness) {
            const label = familyLabel(harness, accounts.snapshot);
            rows.push({ kind: 'sign-in', id: harness, owner: true, action: 'Sign in', open: `accounts/${harness}`,
                text: list.length === 1 ? fmt('{label} account {name} needs sign-in again', { label, name: accountName(list[0]) })
                    : fmt('{n} {label} accounts need sign-in again', { n: list.length, label }) });
        }
    } else if (accounts) {
        rows.push({ kind: 'unread', id: 'accounts', owner: false, action: 'Check', run: 'accounts', text: fmt('Accounts not checked'),
            note: accounts.kind === 'wake' ? fmt('Check starts the agent daemon.') : '' });
    }
    if (tools === null) rows.push({ kind: 'unread', id: 'tools', owner: false, action: 'Check', run: 'tools', text: fmt('Tool servers not checked') });
    for (const server of (tools?.servers || []).filter(toolNeedsSignIn)) {
        rows.push({ kind: 'tool-sign-in', id: String(server.id ?? serverName(server)), owner: true, action: 'Open',
            open: `tools/${encodeURIComponent(String(server.id ?? serverName(server)))}`,
            text: fmt('Tool server {name} needs sign-in', { name: serverName(server) }) });
    }
    if (settings === null) rows.push({ kind: 'unread', id: 'settings', owner: false, action: 'Check', run: 'settings', text: fmt('Saved settings not checked') });
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
    if (!accounts && keys === null) return undefined;
    const read = accounts?.read === READ_OK;
    const known = read ? count(accounts.rows.length, '{n} account', '{n} accounts') : fmt('Accounts not checked');
    // An account that needs sign-in again is the owner's move, as its line of the card says.
    return { text: [known, keyWords].filter(Boolean).join(', '), owner: read && accounts.rows.some(needsSignIn) };
}

function toolsWords(tools) {
    if (tools.enabled === false) return fmt('Tool servers off');
    const servers = Array.isArray(tools.servers) ? tools.servers : [];
    if (!servers.length) return fmt('No tool servers');
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
 * read yet has no entry and keeps what it shows; one whose read failed says "Not checked".
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
        return { text: budget.state === 'available' ? String(budget.label).replace(' / ', ' of ') : budget.label, owner: budget.attention === true };
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

const OUT_ICON = strokeIcon('<path d="M7 17 17 7"/><path d="M7 7h10v10"/>', 14);
const IN_ICON = strokeIcon('<path d="m9 18 6-6-6-6"/>', 14);
// Glyphs the product's set (page_icons.js) does not carry, as Lucide draws them.
const GLYPHS = Object.fromEntries(Object.entries({
    key: '<path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/>',
    rules: '<path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/>',
    growth: '<path d="M16 7h6v6"/><path d="m22 7-8.5 8.5-5-5L2 17"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
    moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    bell: '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    refresh: '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    drive: '<path d="M22 12H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><path d="M6 16h.01"/><path d="M10 16h.01"/>',
    monitor: '<rect width="20" height="14" x="2" y="3" rx="2"/><path d="M8 21h8"/><path d="M12 17v4"/>',
}).map(([name, body]) => [name, strokeIcon(body, 18)]));
const glyphHtml = (name) => GLYPHS[name] || PAGE_ICONS[name] || controlIcon(name, 18);
// The chosen dictation engine's name ('' when none is chosen), from the page's speech hub as it stands.
function dictationWord() {
    try {
        const chosen = getSpeech().selected('stt');
        return String(chosen?.label || chosen?.key || '');
    } catch { return undefined; }
}

/** A level as ascending bars, `lit` of `steps` (Access); a share as an arc of a ring (Costs). */
function barsHtml(lit, steps = 4) {
    const [width, gap] = [3.2, 2.2];
    const start = (24 - (steps * width + (steps - 1) * gap)) / 2;
    const bars = Array.from({ length: steps }, (_, i) => {
        const height = 6 + (12 / (steps - 1)) * i;
        return `<rect class="${i < lit ? 'settings-gauge-lit' : 'settings-gauge-rail'}" x="${(start + i * (width + gap)).toFixed(2)}" y="${(20 - height).toFixed(2)}" width="${width}" height="${height.toFixed(2)}" rx="1.6"/>`;
    }).join('');
    return `<svg class="settings-gauge" width="22" height="22" viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true">${bars}</svg>`;
}
function ringHtml(share) {
    const circumference = 2 * Math.PI * 9;
    const arc = (portion, name) => `<circle class="${name}" cx="12" cy="12" r="9" stroke-dasharray="${(portion * (300 / 360) * circumference).toFixed(2)} ${circumference.toFixed(2)}" transform="rotate(120 12 12)"/>`;
    return `<svg class="settings-gauge" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" aria-hidden="true">${arc(1, 'settings-gauge-rail')}${share > 0 ? arc(Math.max(0.03, Math.min(1, share)), 'settings-gauge-lit') : ''}</svg>`;
}
const markHtml = (presentation) => `<svg width="18" height="18" viewBox="${presentation.viewBox}" fill="currentColor" aria-hidden="true"><path d="${presentation.path}"/></svg>`;

/** The glyphs that say a fact: Main's source, the Access level, the share of the budget spent, the theme. */
function liveGlyphs({ settings, state, device = {} } = {}) {
    const out = {};
    if (settings) {
        const source = routeSource({ model: String(settings.OUROBOROS_MODEL || ''), use_local: truthy(settings.USE_LOCAL_MAIN) });
        const brand = source.startsWith('subscription:') ? harnessPresentation(source.slice('subscription:'.length)) : null;
        out.models = brand && !brand.generic ? markHtml(brand) : glyphHtml('sparkle');
        const access = settings._meta?.policy_state?.access;
        out.permissions = barsHtml(ACCESS_LEVELS.indexOf(String(access?.current_process || access?.effective || '')) + 1);
    }
    if (state) out['dashboard:costs'] = ringHtml(headerBudgetPresentation(state).fillPct / 100);
    if (device.theme) out.appearance = glyphHtml(device.theme === 'light' ? 'sun' : 'moon');
    return out;
}

function doorHtml({ to, name, glyph, answer = '', result = '' }) {
    return `<button type="button" class="settings-door" data-settings-open="${escapeHtml(to)}"${result ? ` data-settings-result="${escapeHtml(result)}"` : ''}>`
        + `<span class="settings-door-glyph" aria-hidden="true" data-settings-glyph>${glyphHtml(glyph)}</span>`
        + `<span class="settings-door-words"><span class="settings-door-title">${escapeHtml(name)}</span>`
        + `<span class="settings-door-answer" data-settings-answer${result ? '' : ' data-i18n-fmt="settings.answer"'}>${escapeHtml(answer)}</span></span>`
        + `<span class="settings-door-end" aria-hidden="true">${result ? '' : statusMarkHtml('settings-door-mark')}${isDoorId(to) ? OUT_ICON : IN_ICON}</span></button>`;
}

const groupHtml = (title, body) => `<section class="settings-group">${title ? `<h3 class="settings-group-title">${escapeHtml(title)}</h3>` : ''}`
    + `<div class="ui-card settings-door-list">${body}</div></section>`;

export function homeGroupsHtml(rows = HOME_ROWS) {
    return [...new Set(rows.map((row) => row.group))]
        .map((group) => groupHtml(group, rows.filter((row) => row.group === group).map(doorHtml).join(''))).join('');
}

/** Search results as rows, then the line that asks Ouroboros with the words. */
export function resultsHtml(query, found) {
    const hits = found.map((entry) => doorHtml({ to: entry.to, name: entry.name, glyph: entry.glyph, answer: entry.where, result: entry.id })).join('');
    const ask = `<button type="button" class="settings-door" data-settings-ask><span class="settings-door-glyph" aria-hidden="true">${glyphHtml('chat')}</span>`
        + `<span class="settings-door-words"><span class="settings-door-title" data-i18n-fmt="settings.ask">${escapeHtml(fmt('Ask Ouroboros: “{words}”', { words: query.trim() }))}</span>`
        + `<span class="settings-door-answer">The words wait in Main as a draft you send.</span></span><span class="settings-door-end" aria-hidden="true">${OUT_ICON}</span></button>`;
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
        <label class="settings-search">${SHELL_ICONS.search}<input type="search" id="settings-search" placeholder="Search settings" autocomplete="off" spellcheck="false" aria-label="Search settings"></label>
        <div class="settings-home-list" data-settings-list>${homeGroupsHtml()}</div>
    </div>`;
    const find = (selector) => root.querySelector(selector);
    const [needsEl, listEl, searchEl, countEl] = [find('[data-being-needs]'), find('[data-settings-list]'), find('#settings-search'), find('[data-being-count]')];
    const facts = { device: {} };
    let current = '';
    let needs = [];
    let shownQuery = '';
    let handed = '';  // the words `search` put into the field, until the owner types there
    let serial = 0;
    let busy = '';
    let needsHtml = '';

    function paintCard() {
        const being = beingState({ connected: ctx.ws?.isConnected?.() !== false, state: facts.state });
        const stateEl = find('[data-being-state]');
        stateEl.textContent = being.text;
        stateEl.dataset.tone = being.tone;
        needs = needRows(facts);
        const waiting = needs.filter((row) => row.owner).length;
        countEl.textContent = waiting ? count(waiting, 'One thing needs you', '{n} things need you') : '';
        countEl.hidden = !waiting;
        find('.settings-being-sep').hidden = !waiting;
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

    // The accounts store as it stands: this subscriber is never a visible surface, so it arms no poll.
    // A client that never read it reads it once on Check; a read that found the daemon unasked makes Check its start.
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
            else if (row.run === 'accounts') await (facts.accounts?.kind === 'wake' ? store.wake() : store.refresh());
            else if (row.run) await read(row.run, serial);
        } finally {
            busy = '';
            paint();
        }
    }

    const onClick = (event) => {
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
            searchEl.removeEventListener('input', onType);
            globalThis.window?.removeEventListener?.('ouro:settings-updated', onSaved);
        },
    };
}
