// Words and small markup pieces shared by every Skills surface: the Installed and
// Browse cards, the detail sheet, the Sources rows, the dialogs and the first-run
// wizard row. Pure functions over host facts; they never read the network or the
// page. Copy follows docs/DESIGN.md: facts in words, unknown is said (never 0),
// a status chip is a flat tint, a status sentence is dot + text, and only the
// three-dot loop moves.
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { strokeIcon } from './page_icons.js';
import * as pageIcons from './page_icons.js';
import { formatCompactNumber, safeExternalUrl } from './utils.js';
import { openExternalViaHostBridge } from './ui_helpers.js';
import { statusMarkHtml } from './status_mark.js';

const OFFICIAL_PATH = '<path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/><path d="m9 12 2 2 4-4"/>';
const WARNING_PATH = '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>';
export const OFFICIAL_GLYPH = strokeIcon(OFFICIAL_PATH, 14);
export const WARNING_GLYPH = strokeIcon(WARNING_PATH, 14);

export const plural = (count, one, many = `${one}s`) => `${count} ${count === 1 ? one : many}`;

/** A version label as published; labels are opaque, never ordered as numbers. */
export function versionText(version) {
    const text = String(version ?? '').trim();
    if (!text) return '';
    return /^v\d/i.test(text) ? text : `v${text}`;
}

/** The host's update outcome, shared by the card and exact-release confirmation. */
export function updateRejectionSentence(release) {
    const cause = {
        review_blockers: 'its review found problems',
        preflight_failed: 'its file checks failed',
        review_unfinished: "its review didn't finish",
        landing_failed: "it couldn't be installed",
    }[release.reason] || 'it did not complete';
    return `Update to ${release.version}: ${cause}; the previous version runs.`;
}

/** With no catalog candidate (including registries), the host pins and checks the rejected release. */
export function canUpdateWithoutReview(install) {
    return Boolean(install?.update_rejected?.digest
        && (!install.update_available || install.update_available.digest === install.update_rejected.digest));
}

/** Saved sources have no visibility fact; only the public hub is known by its address. */
export function publicationDisclosure(privateRepository, repository = '') {
    if (privateRepository === true) return "Everything in the skill's folder goes to your private repository.";
    const target = repoLabel(repository).replace(/\.git$/i, '');
    if (privateRepository === false || target.toLowerCase() === 'razzant/ouroboroshub') {
        return "Everything in the skill's folder becomes public.";
    }
    return `Everything in the skill's folder goes to ${target || 'the selected repository'}; whoever can read that repository can see it.`;
}

function timeOf(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value : 0;
    const parsed = Date.parse(String(value || ''));
    return Number.isFinite(parsed) ? parsed : 0;
}

/** «just now», «12 min ago», «5 h ago», «yesterday», «3 days ago», «7 weeks ago», «4 months ago». */
export function relativeAge(value, now = Date.now()) {
    const time = timeOf(value);
    if (!time) return '';
    const minutes = Math.max(0, (now - time) / 60000);
    if (minutes < 2) return 'just now';
    if (minutes < 60) return `${Math.round(minutes)} min ago`;
    const hours = minutes / 60;
    if (hours < 24) return `${Math.round(hours)} h ago`;
    const startOf = (ms) => new Date(ms).setHours(0, 0, 0, 0);
    const calendarDays = Math.round((startOf(now) - startOf(time)) / 86400000);
    if (calendarDays <= 1) return calendarDays === 1 ? 'yesterday' : `${Math.round(hours)} h ago`;
    const days = hours / 24;
    if (calendarDays < 14) return `${calendarDays} days ago`;
    if (days < 60) return `${Math.round(days / 7)} weeks ago`;
    if (days < 365) return `${Math.round(days / 30.4)} months ago`;
    return plural(Math.round(days / 365), 'year') + ' ago';
}

/** A calendar date for a recorded fact: «12 Oct». */
export function shortDate(value) {
    const time = timeOf(value);
    return time ? new Date(time).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) : '';
}

export function buttonHtml(label, { tone = 'default', size = 'sm', attrs = '', disabled = false } = {}) {
    const sizeClass = size ? ` btn-${size}` : '';
    return `<button type="button" class="btn btn-${tone}${sizeClass}"${attrs ? ` ${attrs}` : ''}${disabled ? ' disabled' : ''}>${escapeHtml(label)}</button>`;
}

/** Skip review: a neutral capsule whose stroke glyph carries the warning; the words say what it does. */
export function skipReviewButtonHtml(attrs = '', label = 'Skip review') {
    return `<button type="button" class="btn btn-default btn-sm skills-skip-btn"${attrs ? ` ${attrs}` : ''}><span class="skills-skip-glyph" aria-hidden="true">${WARNING_GLYPH}</span>${escapeHtml(label)}</button>`;
}

// A job that runs now is the shared moving mark (web/modules/status_mark.js) in its chip's or sentence's ink; a job
// goes on by itself, so its chip and sentence are neutral and amber stays for the owner's move.
const workMarkHtml = () => statusMarkHtml('', { shape: 'moving', hue: 'ink' });

/** A status chip: the chip IS the status (flat tint, never standing). */
export function chipHtml(label, tone = 'neutral', { work = false, attrs = '' } = {}) {
    return `<span class="ui-chip skills-chip" data-tone="${escapeHtml(tone)}"${attrs ? ` ${attrs}` : ''}>${work ? workMarkHtml() : ''}${escapeHtml(label)}</span>`;
}

/** A status sentence (dot + words); live work replaces the dot with the moving mark. */
export function statusHtml(textHtml, tone = 'muted', { work = false } = {}) {
    return `<span class="ui-status skills-status" data-tone="${escapeHtml(tone)}"${work ? ' data-work' : ''}>${work ? workMarkHtml() : ''}<span class="skills-status-text">${textHtml}</span></span>`;
}

/** One attention tray: one sentence (and an optional quiet note) with its actions, visible on touch. */
export function trayHtml({ sentence = '', note = '', actions = '', tone = '', attrs = '' } = {}) {
    if (!sentence) return '';
    return `<div class="skills-tray"${tone ? ` data-tone="${escapeHtml(tone)}"` : ''}${attrs ? ` ${attrs}` : ''}>
        <div class="skills-tray-row">
            <div class="skills-tray-text">${sentence}${note ? `<div class="skills-tray-note">${note}</div>` : ''}</div>
            ${actions ? `<div class="skills-tray-actions">${actions}</div>` : ''}
        </div>
    </div>`;
}

/** The Official mark: identity, quieter than the name; pressing it explains the promise. */
export function officialMarkHtml({ control = true, sourceId = '' } = {}) {
    const inner = `<span class="skills-mark-glyph" aria-hidden="true">${OFFICIAL_GLYPH}</span>Official`;
    return control
        ? `<button type="button" class="skills-mark" data-mark="official" data-official-mark${sourceId ? ` data-official-source="${escapeHtml(sourceId)}"` : ''} aria-haspopup="dialog">${inner}</button>`
        : `<span class="skills-mark" data-mark="official">${inner}</span>`;
}

/** The promise behind the mark, bound to the source that set it. */
export function officialPromise(source = null) {
    const name = source?.name || 'Ouroboros';
    const address = displayAddress(source?.url || '');
    const where = address ? `the ${name} source (${address})` : `the ${name} source`;
    const trust = source?.trust === 'review'
        ? 'Ouroboros still reviews it, because you chose to review each skill from that source.'
        : 'so it installs without review while you trust that source.';
    return `Official — the maintainer of Ouroboros vouches for this exact version. It comes from ${where}${source?.trust === 'review' ? '. ' : ', '}${trust}`;
}

/** An address as people read it: no scheme, no trailing slash. */
export function displayAddress(value) {
    return String(value || '').trim().replace(/^https?:\/\//i, '').replace(/\/+$/, '');
}

/** A repository address as its short name: «CrossRead/scholion» for a GitHub address. */
export function repoLabel(value) {
    return displayAddress(value).replace(/^github\.com\//i, '');
}

/** An external link the page routes through the host bridge (desktop shell, Telegram, browser). */
export function externalLinkHtml(url, label) {
    const target = safeExternalUrl(url);
    if (target === '#') return escapeHtml(label);
    return `<a class="skills-link" href="${escapeHtml(target)}" target="_blank" rel="noopener noreferrer" data-skill-external>${escapeHtml(label)} ↗</a>`;
}

/** Route a press on a `[data-skill-external]` link through the host while the gesture is live. */
export function routeExternalLink(event, openExternal = openExternalViaHostBridge) {
    const link = event?.target?.closest?.('a[data-skill-external]');
    if (!link) return false;
    event.preventDefault();
    Promise.resolve(openExternal(link.getAttribute('href'))).catch(() => {});
    return true;
}

// What a manifest permission lets a skill do, in plain words (detail sheet, dialogs).
export const PERMISSION_WORDS = Object.freeze({
    net: 'Internet',
    tool: 'New tools for Ouroboros',
    route: 'Its own pages inside Ouroboros',
    widget: 'A widget and a settings section',
    ws_handler: 'Live updates inside Ouroboros',
    fs: 'Files outside its own folder',
    subprocess: 'Runs programs on this computer',
    companion_process: 'A program that keeps running in the background',
    supervised_task: 'A program that keeps running in the background',
    read_settings: 'Reads the keys you allow',
    subscribe_event: 'Sees app events',
    inject_chat: 'Writes into your chat — with your permission',
    presence: 'Lives in a place for you: hears, speaks and acts there, with your permission',
    notify_owner: 'Sends you short notices — with your permission',
});

export function permissionWords(permissions = []) {
    const words = [];
    for (const permission of Array.isArray(permissions) ? permissions : []) {
        const base = String(permission || '').split(':')[0];
        const word = PERMISSION_WORDS[base];
        if (word && !words.includes(word)) words.push(word);
    }
    return words;
}

// Grantable host permissions as the request a skill makes («Wants your permission to …»).
const GRANT_PHRASES = Object.freeze({
    presence: 'live in a place for you: hear, speak and act there',
    inject_chat: 'write into your chat',
    notify_owner: 'send you short notices',
    subscribe_event: 'see chat events',
});

export function grantPhrases(permissions = []) {
    const phrases = [];
    for (const permission of permissions) {
        const base = String(permission || '').split(':')[0];
        const phrase = GRANT_PHRASES[base] || String(permission || '');
        if (phrase && !phrases.includes(phrase)) phrases.push(phrase);
    }
    return phrases;
}

/** One popularity fact with its noun; unknown is said, never a bare 0 (UX §2.4). */
export function popularityText(item = {}) {
    const stats = item.registry_stats && typeof item.registry_stats === 'object' ? item.registry_stats : null;
    if (stats && Object.values(stats).some((value) => typeof value === 'number')) {
        // A registry (ClawHub) counts stars and downloads, not votes.
        const parts = [];
        if (typeof stats.stars === 'number') parts.push(stats.stars > 0 ? `★ ${formatCompactNumber(stats.stars)}` : 'No stars yet');
        if (typeof stats.downloads === 'number' && stats.downloads > 0) parts.push(`${formatCompactNumber(stats.downloads)} downloads`);
        if (typeof stats.comments === 'number' && stats.comments > 0) parts.push(plural(stats.comments, 'comment'));
        return parts.join(' · ') || 'No data';
    }
    const signals = item.signals && typeof item.signals === 'object' ? item.signals : null;
    if (!signals) return item.source_id === 'builtin' ? '' : 'No data';
    if (typeof signals.votes !== 'number') return 'Votes not counted yet';
    return signals.votes > 0 ? `▲ ${plural(signals.votes, 'vote')}` : 'No votes yet';
}

/** The facts line: only exceptions and the one popularity number (UX §2.1 ⑤). */
export function factsText(item = {}) {
    const parts = [];
    const popularity = popularityText(item);
    if (popularity) parts.push(popularity);
    const keys = Array.isArray(item.needs_keys) ? item.needs_keys.length : 0;
    if (keys) parts.push(keys === 1 ? 'needs a key' : `needs ${keys} keys`);
    return parts.join(' · ');
}

/** «Ouroboros Community · by l1nkberry · CrossRead/scholion ↗ · updated 2 days ago». */
export function catalogWhoHtml(item = {}, { version = false, now = Date.now() } = {}) {
    if (item.source_id === 'builtin') return 'Comes with the app · updates with Ouroboros';
    const parts = [escapeHtml(item.source_name || item.source_id || '')];
    if (item.owner) parts.push(`by ${escapeHtml(item.owner)}`);
    // A pointer entry lives in its author's repository (SPEC §3.1): the who line links it.
    if (item.author_repo) parts.push(externalLinkHtml(item.author_repo, repoLabel(item.author_repo)));
    if (version && item.version) parts.push(escapeHtml(versionText(item.version)));
    const updated = relativeAge(item.updated_at, now);
    if (updated) parts.push(`updated ${escapeHtml(updated)}`);
    return parts.filter(Boolean).join(' · ');
}

/** The consequence of Install, said before it is pressed (UX §0.3 rule 2). */
export function trustSentence(source) {
    if (source?.kind === 'builtin') return 'Comes with the app';
    if (source?.trust === 'trusted') return `Installs without review — you trust ${source.name || 'this source'}`;
    return 'Ouroboros reviews it before it runs';
}

/** The one-line «about» of a source row: the source's kind in words. */
export function sourceAboutText(source = {}) {
    if (source.kind === 'builtin') return 'Skills that come with the app and update with Ouroboros.';
    if (source.kind === 'hub' && source.part === 'official') return 'The Ouroboros project’s own skills, marked Official by its maintainer.';
    if (source.kind === 'hub') return 'Skills shared by other people.';
    if (source.kind === 'registry') return 'OpenClaw’s public registry. Thousands of skills; they appear when you search.';
    if (source.kind === 'local') return 'A folder on this computer. Ouroboros reads every skill folder in it.';
    return 'A git repository with skills.';
}

/** The trust value of a source row: a setting, not a status. */
export function trustValueText(source = {}) {
    return source.trust === 'trusted' ? 'Installs without review' : 'Reviews each skill';
}

/**
 * Pass the host's SVG or named glyph spec to `renderSkillIcon`; legacy `{name}` specs also keep their glyph.
 * An absent or unknown icon uses the house plug. The label beside it always says what it is.
 */
export function skillIconHtml(icon, size = 18) {
    const render = pageIcons.renderSkillIcon;
    if (typeof render === 'function') {
        try { return render(icon || {}, { size }); } catch { /* the house glyph below */ }
    }
    return pageIcons.controlIcon(icon?.name || '', size);
}

/** The services a connector connects: the host's `connects` fact, a list of names derived from its place declaration. */
export function connectsServices(item) {
    const value = item?.connects;
    return Array.isArray(value) ? value.map((name) => String(name || '').trim()).filter(Boolean) : [];
}

/** «Connects Slack» for a connector, else ''. */
export function connectsText(item) {
    const services = connectsServices(item);
    return services.length ? `Connects ${services.join(', ')}` : '';
}

/** A skill that writes into the owner's chat without a place of its own: what arrives from it is the owner's. */
export function countsAsYoursText(item) {
    const permissions = Array.isArray(item?.permissions) ? item.permissions : [];
    const injects = permissions.some((permission) => String(permission || '').split(':')[0] === 'inject_chat');
    return injects && !connectsServices(item).length ? 'Messages from here count as yours' : '';
}

/** A connector's contribution supplies its public label; ids remain canonical. */
export const skillTitle = (skill) => skill.title || skill.contributes?.find((row) => row.provides === 'place' && row.label)?.label || skill.name;
