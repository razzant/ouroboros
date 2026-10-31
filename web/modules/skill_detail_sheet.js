// The skill detail sheet (UX §3, board B2c): everything needed to decide to
// install, or to understand an installed skill, on the delivered-document
// reader's geometry — a dialog over the page, a full sheet on a phone. The head
// comes from the card at once; the body reads the source's full entry and says
// so while it does. Close, Escape and the backdrop return focus to the opener.
import { bindDialogFocus } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { mountChatMarkdown, destroyChatMarkdown } from './chat_markdown.js';
import { mountConnectorPlaces } from './connector_places.js';
import { reviewSentence } from './skill_card_renderer.js';
import {
    catalogWhoHtml, connectsServices, displayAddress, externalLinkHtml, officialMarkHtml, permissionWords, popularityText,
    relativeAge, repoLabel, routeExternalLink, versionText,
} from './skill_presentation.js';

function section(title, html) {
    return html ? `<section class="skills-sheet-section"><h3>${escapeHtml(title)}</h3>${html}</section>` : '';
}

const note = (html) => (html ? `<p class="skills-sheet-note">${html}</p>` : '');

/** The one review sentence for a skill seen from the sheet (installed facts win). */
function sheetReviewSentence(item, installed, source) {
    if (installed) return reviewSentence(installed);
    if (source?.kind === 'builtin') return 'Comes with the app and updates with Ouroboros.';
    if (source?.trust === 'trusted') return `Installs without review: you trust ${source.name}.`;
    return 'Ouroboros will review it on this computer before it runs. You can skip the review.';
}

function versionsHtml(detail, item, installed) {
    const rows = Array.isArray(detail?.versions) && detail.versions.length
        ? detail.versions
        : (item.version ? [{ version: item.version, date: item.updated_at, current: true }] : []);
    const lines = rows.map((row) => [versionText(row.version), relativeAge(row.date), row.current ? 'current' : '', row.note || '']
        .filter(Boolean).map(escapeHtml).join(' · '));
    if (installed?.version) lines.unshift(`You have ${escapeHtml(versionText(installed.version))}.`);
    if (item.withdrawn) lines.push(`${escapeHtml(versionText(item.version))} was withdrawn${item.withdrawn.reason ? `: ${escapeHtml(item.withdrawn.reason)}` : '.'}`);
    return lines.length ? `<p>${lines.join('<br>')}</p>` : '';
}

function usesHtml(detail, item) {
    // The entry's own words come from its permission ids (the cards say the same);
    // an id this page has no word for keeps the source's label.
    const uses = Array.isArray(detail?.uses) ? detail.uses.filter((use) => use && typeof use === 'object') : null;
    const permissions = uses ? uses.filter((use) => use.kind !== 'key') : (item.permissions || []).map((id) => ({ id }));
    const words = [];
    for (const use of permissions) {
        const word = permissionWords([use.id])[0] || String(use.label || '');
        if (word && !words.includes(word)) words.push(word);
    }
    const keys = uses ? uses.filter((use) => use.kind === 'key').map((use) => String(use.id || '')).filter(Boolean) : (item.needs_keys || []);
    return `<p>${words.length ? words.map(escapeHtml).join(' · ') : 'Nothing beyond its own folder'}</p>${note(`Your keys: ${keys.length ? keys.map(escapeHtml).join(', ') : 'none'}`)}`;
}

function authorRepoHtml(item) {
    if (!item.author_repo) return '';
    const parts = [externalLinkHtml(item.author_repo, repoLabel(item.author_repo))];
    const stars = item.signals?.author_repo_stars;
    if (typeof stars === 'number') parts.push(`★ ${escapeHtml(String(stars))} ${stars === 1 ? 'star' : 'stars'} on ${escapeHtml(repoLabel(item.author_repo))}`);
    return `<p>${parts.join(' · ')}</p>${note('The source keeps a pointer to this repository; the author keeps the stars, issues and releases.')}`;
}

function linksHtml(detail, item) {
    const links = detail?.links && typeof detail.links === 'object' && !Array.isArray(detail.links) ? detail.links : {};
    const comments = item.signals?.comments;
    const rows = [
        [links.author_repo || item.author_repo, 'Author’s repository'],
        [links.discussion || item.signals?.discussion_url, `Discussion${typeof comments === 'number' ? ` (${comments})` : ''}`],
        [links.repo || item.repo_url, `${item.source_name || 'Source'} repository`],
    ].filter(([url]) => url);
    return rows.length ? `<p>${rows.map(([url, label]) => externalLinkHtml(url, label)).join(' · ')}</p>` : '';
}

function detailsHtml(item, installed) {
    const rows = [
        ['Source', [item.source_name, displayAddress(item.repo_url)].filter(Boolean).join(' · ')],
        ['Exact version', [versionText(item.version), item.digest || ''].filter(Boolean).join(' · ')],
        ['Installed folder', installed?.payload_root || ''],
        ['Ouroboros’s check', installed?.install?.check?.notes || ''],
    ].filter(([, value]) => value);
    if (!rows.length) return '';
    return `<details class="skills-details skills-sheet-details"><summary>Details</summary><dl class="skills-sheet-facts">${rows.map(([label, value]) =>
        `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl></details>`;
}

/**
 * Open the sheet. `item` is the catalog entry (or the installed skill seen as
 * one); `actionsHtml` mirrors the card's action row; `onAction(action, node)`
 * receives presses on `[data-sheet-action]`. `readDetail()` fetches the full
 * entry; without it the sheet shows what the card had.
 */
export function openSkillDetailSheet({ item, installed = null, source = null, actionsHtml = '', onAction = () => {}, readDetail = null, returnFocus = document.activeElement }) {
    const dialog = document.createElement('dialog');
    dialog.className = 'document-reader skills-detail-sheet';
    dialog.setAttribute('aria-labelledby', 'skills-sheet-title');
    dialog.setAttribute('aria-modal', 'true');
    const popularity = item.source_id === 'builtin' ? '' : popularityText(item);
    dialog.innerHTML = `<div class="document-reader-panel">
        <div class="document-reader-head">
            <div class="document-reader-identity">
                <h2 class="document-reader-title" id="skills-sheet-title">${escapeHtml(item.name)}${item.official ? ` ${officialMarkHtml({ control: false })}` : ''}</h2>
                <div class="document-reader-meta">${catalogWhoHtml(item, { version: true })}${popularity ? ` · ${escapeHtml(popularity)}` : ''}</div>
            </div>
            <div class="document-reader-actions" data-sheet-actions>${actionsHtml}</div>
            <button type="button" class="btn btn-default document-reader-close" data-sheet-close>Close</button>
        </div>
        <div class="document-reader-body skills-sheet-body is-quiet-focus" tabindex="-1" data-sheet-body><section class="skills-sheet-section" data-sheet-places hidden></section><div class="skills-sheet-content" data-sheet-content></div></div>
    </div>`;
    const body = dialog.querySelector('[data-sheet-body]');
    const content = dialog.querySelector('[data-sheet-content]');
    let disposePlaces = null;
    let closed = false;
    let disposeFocus = null;
    let controller = null;
    const close = () => {
        if (closed) return;
        closed = true;
        controller?.abort();
        disposePlaces?.();
        destroyChatMarkdown(content.querySelector('.skills-sheet-about'));
        // Close the modal first: WebKit's own close moves focus to what was focused before it opened
        // (a pressed button is not focused there), and the restore below then returns it to the opener.
        if (dialog.open && typeof dialog.close === 'function') dialog.close();
        disposeFocus?.();
        dialog.remove();
    };
    const paint = (detail, { reading = false, error = '' } = {}) => {
        const about = String(detail?.about || item.description || '');
        const aboutError = detail && !detail.about && detail.about_error
            ? note(`Its SKILL.md couldn’t be read (${escapeHtml(detail.about_error)}); this is the catalog’s description.`) : '';
        destroyChatMarkdown(content.querySelector('.skills-sheet-about'));
        content.innerHTML = [
            reading ? `<p class="skills-sheet-status" role="status">Reading ${escapeHtml(item.name)} from ${escapeHtml(item.source_name || 'its source')}…</p>` : '',
            error ? `<div class="skills-sheet-status is-error" role="alert"><p>Couldn’t read the full description from ${escapeHtml(item.source_name || 'its source')}: ${escapeHtml(error)}.</p><button type="button" class="btn btn-default btn-sm" data-sheet-retry>Try again</button></div>` : '',
            `<section class="skills-sheet-section"><h3>About</h3><div class="skills-sheet-about" data-i18n-authored></div>${aboutError}</section>`,
            section('What it can use', usesHtml(detail, item)),
            section('Review', `<p>${escapeHtml(sheetReviewSentence(item, installed, source))}</p>`),
            section('Author’s repository', authorRepoHtml(item)),
            section('Versions', versionsHtml(detail, item, installed)),
            section('Links', linksHtml(detail, item)),
            detailsHtml(item, installed),
        ].join('');
        mountChatMarkdown(body.querySelector('.skills-sheet-about'), about, { softBreaks: true });
    };
    const read = async () => {
        if (!readDetail) { paint(null); return; }
        controller?.abort();
        controller = typeof AbortController === 'function' ? new AbortController() : null;
        paint(null, { reading: true });
        try {
            const detail = await readDetail({ signal: controller?.signal });
            if (!closed) paint(detail);
        } catch (err) {
            if (!closed && err?.name !== 'AbortError') paint(null, { error: err?.message || String(err) });
        }
    };
    dialog.addEventListener('click', (event) => {
        if (routeExternalLink(event)) return;
        if (event.target === dialog || event.target.closest('[data-sheet-close]')) { close(); return; }
        if (event.target.closest('[data-sheet-retry]')) { read(); return; }
        const action = event.target.closest('[data-sheet-action]');
        if (action) onAction(action.dataset.sheetAction, action);
    });
    dialog.addEventListener('close', close);
    dialog.addEventListener('cancel', (event) => { event.preventDefault(); close(); });
    // Opened by a pointer, the reading region shows its focus ring from the first key, not before.
    dialog.addEventListener('keydown', () => body.classList.remove('is-quiet-focus'), { once: true });
    document.body.appendChild(dialog);
    if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    disposeFocus = bindDialogFocus(dialog, { initialFocus: body, returnFocus, onEscape: close });
    if (installed && connectsServices(installed).length) {
        const places = dialog.querySelector('[data-sheet-places]'); places.hidden = false;
        disposePlaces = mountConnectorPlaces(places, installed.name, { leave: close, returnFocus });
    }
    read();
    return {
        close,
        get closed() { return closed; },
        /** Repaint the head's action row from the card's current state. */
        setActions(html) { const host = dialog.querySelector('[data-sheet-actions]'); if (host) host.innerHTML = html; },
    };
}
