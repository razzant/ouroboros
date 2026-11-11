// Share (docs/DESIGN.md "Replies"): a saved Ouroboros message's or living card's action, beside Copy and in its row
// menu, that shows exactly what would leave before anything does. A message opens on its whole words as Text; Choose
// excerpt lets the owner tap whole parts or select exact words inside the preview (the feed's own selection is neither
// read nor changed); Card draws the same words as one picture (message_share_card.js); File is the whole message as
// one offline page (message_share_file.js). A card opens on its File (in Telegram, on its words as Text):
// the card itself as one page, read once from its own body (`GET /api/cards/{id}`), with its words as Text beside it.
// The File preview is the file itself. Only the ways this device really has are offered: Copy, Download (a picture or
// a file, outside Telegram) and Share… (Telegram's chat picker for Text in its Mini App, otherwise the system share
// sheet when it accepts exactly this payload). Nothing is sent, prepared or uploaded before a press; each outcome is
// said as it happened and a cancel stays quiet. Close, Escape or Back, Panic, leaving the page, and its room closing
// what it opened (chat_media.js holdTransient: the room covered or put away, the row released) close the sheet, and
// nothing that finishes later paints into it or sends.

import { fmt, fmtInto, tr } from './i18n.js';
import { bindDialogFocus } from './ui_interactions.js';
import { getMessageActions } from './message_actions.js';
import { controlIcon } from './page_icons.js';
import { mountChatMarkdown } from './chat_markdown.js';
import { readAppText, readCardBody } from './chat_cards.js';
import { copyTextToClipboard, downloadBlobViaHostBridge } from './ui_helpers.js';
import { extensionRoutePath, fetchJson } from './api_client.js';
import { hostCall, isTelegramHost } from './host_telegram.js';
import { blockWords, pickBlocks, shareBlocks, shareText, sliceBlocks, wordCount } from './message_share_model.js';
import { renderShareCard } from './message_share_card.js';
import { answerPage, appPage, componentsPage, drawnAnswer, shareFileName } from './message_share_file.js';

export const SHARE = 'share';
// Telegram's text message bound, in UTF-16 units (JavaScript's own length): the bot checks the same.
export const TELEGRAM_TEXT_LIMIT = 4096;
export const PREPARE_SHARE_ROUTE = extensionRoutePath('telegram', 'miniapp/prepare-share');
const TELEGRAM_FAILURES = {
    MESSAGE_EXPIRED: 'That prepared message expired. Press Share… again.',
    UNSUPPORTED: 'This Telegram app cannot share messages from here.',
    MESSAGE_SEND_FAILED: 'Telegram could not send it.',
};

const SHEET_HTML = `
    <div class="document-reader-panel">
        <header class="document-reader-head">
            <div class="document-reader-identity">
                <h2 class="document-reader-title">Share</h2>
                <div class="document-reader-meta message-share-lead"></div>
            </div>
            <div class="document-reader-actions">
                <div class="document-reader-views" role="group" aria-label="Format">
                    <button type="button" class="btn btn-default" data-share-format="text" aria-pressed="true">Text</button>
                    <button type="button" class="btn btn-default" data-share-format="card" aria-pressed="false">Card</button>
                    <button type="button" class="btn btn-default" data-share-format="file" aria-pressed="false">File</button>
                </div>
            </div>
            <button type="button" class="btn btn-default document-reader-close" data-share-action="close" aria-label="Close sharing">Close</button>
        </header>
        <div class="document-reader-body message-share-body is-quiet-focus" tabindex="0" role="region" aria-label="What would leave"></div>
        <footer class="message-share-foot">
            <div class="message-share-scope"><p class="message-share-count"></p></div>
            <div class="message-share-actions"></div>
            <p class="message-share-status" role="status" aria-live="polite" hidden></p>
        </footer>
    </div>`;

const reasonOf = (error) => String(error?.message || error || 'unknown error');

/** The prepared Telegram message for `text`: the bot's own `{id, expiresAt}` (skills/telegram, miniapp/prepare-share). */
export async function prepareTelegramShare(text, { signal = null, fetchImpl = undefined } = {}) {
    const body = await fetchJson(PREPARE_SHARE_ROUTE, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal,
    }, { fetchImpl });
    const id = String(body?.id || '');
    if (!id) throw new Error(tr('share.telegram_no_id', 'The bot returned no message to share.'));
    return { id, expiresAt: Number(body.expiration_date) || 0 };
}

/** The Telegram Mini App's own chat picker, where the client has it (Bot API 8.0). */
export function telegramShareAvailable({ win = globalThis.window, doc = globalThis.document } = {}) {
    const app = win?.Telegram?.WebApp;
    return isTelegramHost({ win, doc }) && typeof app?.shareMessage === 'function'
        && typeof app.isVersionAtLeast === 'function' && app.isVersionAtLeast('8.0') === true;
}

function canShare(win, data) {
    const nav = win?.navigator;
    if (typeof nav?.share !== 'function' || typeof nav.canShare !== 'function') return false;
    try { return nav.canShare(data) === true; } catch { return false; }
}

const canCopyImage = (win) => typeof win?.navigator?.clipboard?.write === 'function'
    && typeof win.ClipboardItem === 'function'
    && (typeof win.ClipboardItem.supports !== 'function' || win.ClipboardItem.supports('image/png') === true);

/** A card's plain words as the blocks Text and an excerpt read: a paragraph per blank line, a line break per line. */
function plainWords(doc, template, text) {
    for (const paragraph of String(text || '').split(/\n[ \t]*\n/)) {
        if (!paragraph.trim()) continue;
        const node = doc.createElement('p');
        paragraph.split('\n').forEach((line, index) => {
            if (index) node.append(doc.createElement('br'));
            node.append(doc.createTextNode(line));
        });
        template.content.append(node);
    }
}

/**
 * The File of one card `described` (`{id, revision, node}`, from its row): its own current body, then the page of its
 * kind, or `{refused}` with why there is none. A live skill widget runs only with its skill and is never a file.
 */
export async function cardFile(described, { doc, win, readCard = readCardBody, readApp = readAppText, alive = () => true,
    pages = { componentsPage, appPage } } = {}) {
    let body;
    try { body = await readCard(described.id, Number(described.revision) || 0); } catch (error) {
        return { refused: error?.status === 404 ? 'gone' : 'unreadable', reason: reasonOf(error) };
    }
    if (!alive()) return { refused: 'closed' };
    const material = described.node?.isConnected ? described.node : null;
    const name = shareFileName(body?.title);
    if (body?.kind === 'components') {
        return { ...(await pages.componentsPage(body, { doc, win, material })), kind: 'components', name, scripted: false };
    }
    if (body?.kind === 'app' && body.app?.url) {
        let html;
        try { html = await readApp(body.app.url, body.app.sha256); } catch (error) {
            return { refused: 'unreadable', reason: reasonOf(error) };
        }
        if (!alive()) return { refused: 'closed' };
        return { ...pages.appPage(body, html, { doc, win, material }), kind: 'app', name, scripted: true };
    }
    return { refused: body?.kind === 'widget' ? 'widget' : 'unsupported' };
}

/** The page's text range `range` inside `container` as model pieces `[{block, run, start, end}]`, in document order. */
export function selectedPieces(container, range) {
    const pieces = [];
    if (!container || !range) return pieces;
    for (const span of container.querySelectorAll('[data-share-run]')) {
        const node = span.firstChild;
        if (node?.nodeType !== 3) continue;
        let inside = false;
        try { inside = range.intersectsNode(node); } catch { inside = false; }
        if (!inside) continue;
        const start = range.startContainer === node ? range.startOffset : 0;
        const end = range.endContainer === node ? range.endOffset : node.data.length;
        if (end <= start) continue;
        pieces.push({ block: Number(span.closest('[data-share-block]')?.dataset.shareBlock), run: Number(span.dataset.shareRun), start, end });
    }
    return pieces;
}

/**
 * Register Share in `registry` (message_actions.js). It applies to a row whose host recorded its words
 * (`row.answerWords`) or its card (`row.card`). Returns the unregister; `close()`, `isOpen()` and `open(row, ctx)` are
 * on it for the page and tests.
 */
export function registerMessageShare({ registry = getMessageActions(), win = globalThis.window, doc = globalThis.document,
    render = mountChatMarkdown, card = renderShareCard, prepare = prepareTelegramShare, readCard = readCardBody,
    readApp = readAppText, pages = { answerPage, componentsPage, appPage, drawnAnswer } } = {}) {
    let session = null;
    let opened = 0;

    function status(current, text = '', tone = 'info') {
        const line = current.dialog.querySelector('.message-share-status');
        line.textContent = text;
        line.dataset.tone = tone;
        line.hidden = !text;
    }

    function releaseCard(current) {
        if (current.card?.url) win.URL?.revokeObjectURL?.(current.card.url);
        current.card = null;
    }

    function releaseFile(current) {
        current.file?.observer?.disconnect();
        current.file = null;
    }

    function close({ restoreFocus = true } = {}) {
        const current = session;
        if (!current) return;
        session = null;
        current.abort.abort();
        for (const dispose of current.disposers.splice(0)) {
            try { dispose(); } catch { /* already gone */ }
        }
        releaseCard(current);
        releaseFile(current);
        if (typeof current.dialog.close === 'function' && current.dialog.open) current.dialog.close();
        else current.dialog.removeAttribute('open');
        current.disposeFocus?.({ restoreFocus });
        current.dialog.remove();
    }

    const shownBlocks = (current) => current.excerpt || current.blocks;
    const words = (blocks) => wordCount(blockWords(blocks));
    // Inside Telegram (no file download there) and its chat picker (a client new enough to have one) are two facts.
    const inTelegram = () => isTelegramHost({ win, doc });
    const telegram = () => telegramShareAvailable({ win, doc });

    function setExcerpt(current, excerpt) {
        current.excerpt = excerpt && excerpt.length ? excerpt : null;
        current.text = shareText(shownBlocks(current));
        current.version += 1;
        releaseCard(current);
        status(current);
    }

    function ensureCard(current) {
        if (current.card?.key === current.version) return;
        releaseCard(current);
        const key = current.version;
        current.card = { key, pending: true };
        Promise.resolve()
            .then(() => card(shownBlocks(current), { excerpt: Boolean(current.excerpt), doc, win }))
            .then((result) => {
                // A later choice, another format or a closed sheet: this picture paints nothing.
                if (session !== current || current.card?.key !== key) return;
                if (result?.blob) {
                    const name = current.excerpt ? 'ouroboros-excerpt.png' : 'ouroboros-answer.png';
                    const file = typeof win.File === 'function' ? new win.File([result.blob], name, { type: 'image/png' }) : null;
                    current.card = { key, ...result, name, file, url: win.URL.createObjectURL(result.blob),
                        shareable: Boolean(file) && canShare(win, { files: [file] }) };
                } else current.card = { key, refused: result?.refused || 'unavailable', kinds: result?.kinds || [] };
                paint(current);
            }, (error) => {
                if (session !== current || current.card?.key !== key) return;
                current.card = { key, refused: 'failed', reason: reasonOf(error) };
                paint(current);
            });
    }

    // A message's file is made once the room has drawn its diagrams and charts (a bound later at most: what is still
    // drawing then leaves as its source, and the preview says so).
    async function answerFile(current, alive) {
        const undrawn = await (pages.drawnAnswer || drawnAnswer)(current.content, { alive });
        if (!alive()) return { refused: 'closed' };
        const made = pages.answerPage(current.content, { doc, win });
        return { ...made, kind: 'answer', name: shareFileName(made.title, 'ouroboros-answer'), scripted: false, undrawn };
    }

    // One file per open sheet: made at its first look (the whole message, or the card as its body is then) and kept.
    function ensureFile(current) {
        if (current.file) return;
        const key = (current.fileKey += 1);
        current.file = { key, pending: true };
        const alive = () => session === current && current.file?.key === key;
        Promise.resolve()
            .then(() => (current.cardSource
                ? cardFile(current.cardSource, { doc, win, readCard, readApp, alive, pages })
                : answerFile(current, alive)))
            .then((made) => {
                if (!alive()) return;  // closed, or another sheet: this file paints nothing
                if (made?.html) {
                    const BlobType = win.Blob || globalThis.Blob;
                    const blob = new BlobType([made.html], { type: 'text/html' });
                    const file = typeof win.File === 'function' ? new win.File([blob], made.name, { type: 'text/html' }) : null;
                    current.file = { key, ...made, blob, file, shareable: Boolean(file) && canShare(win, { files: [file] }) };
                } else current.file = { key, refused: made?.refused || 'unsupported', reason: made?.reason || '' };
                paint(current);
            }, (error) => {
                if (!alive()) return;
                current.file = { key, refused: 'failed', reason: reasonOf(error) };
                paint(current);
            });
    }

    // ---- the preview: the bytes that leave, or the parts to choose from
    function runSpan(run, index) {
        const span = doc.createElement('span');
        span.dataset.shareRun = String(index);
        const classes = ['bold', 'italic', 'code', 'strike'].filter((key) => run[key]).map((key) => `is-${key}`);
        if (run.link && run.href) classes.push('is-link');
        if (classes.length) span.className = classes.join(' ');
        span.textContent = run.text;
        return span;
    }

    function partRow(current, item) {
        const row = doc.createElement('div');
        row.className = `message-share-part is-${item.kind}`;
        row.dataset.shareBlock = String(item.index);
        row.setAttribute('role', 'checkbox');
        row.setAttribute('aria-checked', String(current.chosen.has(item.index)));
        row.tabIndex = 0;
        row.style.setProperty('--share-depth', String((item.depth || 0) + (item.quote || 0)));
        if (item.quote) row.classList.add('is-quoted');
        if (item.rtl) row.dir = 'rtl';
        const marker = doc.createElement('span');
        marker.className = 'message-share-marker';
        marker.setAttribute('aria-hidden', 'true');
        if (item.kind === 'item') {
            const box = item.task === 'done' ? '☑' : item.task === 'open' ? '☐' : '';
            marker.textContent = [item.marker === '•' && box ? '' : item.marker, box].filter(Boolean).join(' ');
        }
        const verbatim = ['code', 'diagram', 'chart', 'table'].includes(item.kind);
        const content = doc.createElement(verbatim ? 'pre' : 'div');
        content.className = 'message-share-words';
        if (item.kind === 'heading') content.dataset.level = String(item.level);
        if (item.kind === 'rule') content.append(doc.createElement('hr'));
        item.runs.forEach((run, index) => content.append(runSpan(run, index)));
        const check = doc.createElement('span');
        check.className = 'message-share-check';
        check.setAttribute('aria-hidden', 'true');
        check.innerHTML = controlIcon('check', 16);
        row.append(marker, content, check);
        return row;
    }

    function cardNote(state) {
        const note = doc.createElement('div');
        note.className = 'document-reader-status';
        const line = doc.createElement('p');
        note.append(line);
        if (!state || state.pending) {
            note.setAttribute('role', 'status');
            fmtInto(line, 'Drawing the card…');
            return note;
        }
        note.setAttribute('role', 'alert');
        note.classList.add('is-error');
        const names = { table: tr('share.kind_table', 'a table'), math: tr('share.kind_math', 'a formula'),
            diagram: tr('share.kind_diagram', 'a diagram'), chart: tr('share.kind_chart', 'a chart') };
        if (state.refused === 'unsupported') {
            fmtInto(line, 'A card cannot show {kinds}. Share it as Text, or choose an excerpt without it.',
                { kinds: (state.kinds || []).map((kind) => names[kind] || kind).join(', ') });
        } else if (state.refused === 'too_wide') fmtInto(line, 'This layout is too wide for one readable card. Share it as Text, or choose a simpler excerpt.');
        else if (state.refused === 'too_long') fmtInto(line, 'This is too long for one readable card. Choose an excerpt, or share it as Text.');
        else if (state.refused === 'failed') fmtInto(line, 'The card could not be drawn: {reason}', { reason: state.reason || '' });
        else fmtInto(line, 'This device cannot draw a card. Share it as Text.');
        return note;
    }

    function fileNote(state) {
        const note = doc.createElement('div');
        note.className = 'document-reader-status';
        const line = doc.createElement('p');
        note.append(line);
        if (!state || state.pending) {
            note.setAttribute('role', 'status');
            fmtInto(line, 'Making the file…');
            return note;
        }
        note.setAttribute('role', 'alert');
        note.classList.add('is-error');
        if (state.refused === 'widget') fmtInto(line, 'This card is a skill’s live widget: it works only with its skill, so a file cannot carry it. Share its words as Text.');
        else if (state.refused === 'gone') fmtInto(line, 'This card is no longer available.');
        else if (state.refused === 'unreadable') fmtInto(line, 'The card could not be read: {reason}', { reason: state.reason || '' });
        else if (state.refused === 'failed') fmtInto(line, 'The file could not be made: {reason}', { reason: state.reason || '' });
        else fmtInto(line, 'This card cannot be put in a file. Share its words as Text.');
        return note;
    }

    /** A page without script shows at its own height: it is read from here, never run. */
    function fitPreview(current, frame) {
        const file = current.file;
        const page = frame.contentDocument?.documentElement;
        if (session !== current || file?.frame !== frame || !page) return;
        const fit = () => frame.style.setProperty('--message-share-file-height', `${Math.ceil(page.scrollHeight)}px`);
        fit();
        file.observer?.disconnect();
        file.observer = typeof win.ResizeObserver === 'function' ? new win.ResizeObserver(fit) : null;
        file.observer?.observe(page);
    }

    // The file itself, in a frame of its own: a page with no script is read for its height and never run; an app's
    // page runs as it would anywhere, inside its frame, with a line on what it shows where scripts do not run.
    function filePreview(current) {
        const file = current.file;
        if (!file.frame) {
            const frame = doc.createElement('iframe');
            frame.className = 'message-share-file';
            frame.title = tr('share.file_preview', 'The file as it opens');
            frame.setAttribute('sandbox', file.scripted ? 'allow-scripts' : 'allow-same-origin');
            if (!file.scripted) frame.addEventListener('load', () => fitPreview(current, frame));
            frame.srcdoc = file.html;
            file.frame = frame;
            file.view = [frame];
            const said = doc.createElement('p');
            said.className = 'message-share-file-note';
            if (file.scripted) fmtInto(said, 'Where scripts do not run, as in a phone’s file preview, this file opens on the card’s words.');
            else if (file.undrawn?.length) fmtInto(said, 'A diagram or chart was still being drawn: the file shows its source instead.');
            if (said.textContent) file.view.push(said);
        }
        // Moving a frame loads it again: one already in place stays as it is.
        if (current.body.childNodes.length !== file.view.length || file.view.some((node, index) => current.body.childNodes[index] !== node)) {
            current.body.replaceChildren(...file.view);
        }
    }

    function paintBody(current) {
        const body = current.body;
        current.parts = null;
        if (current.mode === 'choose') {
            const parts = doc.createElement('div');
            parts.className = 'message-share-parts';
            parts.setAttribute('role', 'group');
            parts.setAttribute('aria-label', tr('share.parts', 'Parts of the message'));
            parts.setAttribute('data-i18n-authored', '');
            for (const item of current.blocks) parts.append(partRow(current, item));
            body.replaceChildren(parts);
            current.parts = parts;
        } else if (current.format === 'text') {
            const text = doc.createElement('pre');
            text.className = 'message-share-text';
            text.dir = 'auto';
            text.setAttribute('data-i18n-authored', '');
            text.textContent = current.text;
            body.replaceChildren(text);
        } else if (current.format === 'file') {
            if (current.file?.html) filePreview(current);
            else body.replaceChildren(fileNote(current.file));
        } else if (current.card?.blob) {
            const image = doc.createElement('img');
            image.className = 'message-share-card';
            image.src = current.card.url;
            image.width = current.card.width;
            image.height = current.card.height;
            image.alt = current.excerpt ? tr('share.card_alt_excerpt', 'Card picture of the excerpt')
                : tr('share.card_alt', 'Card picture of the message');
            body.replaceChildren(image);
        } else body.replaceChildren(cardNote(current.card));
    }

    // ---- the footer: what is chosen, and only the ways out this device really has
    function button(label, action, { primary = false, disabled = false, title = '' } = {}) {
        const node = doc.createElement('button');
        node.type = 'button';
        node.className = `btn ${primary ? 'btn-primary' : 'btn-default'}`;
        node.dataset.shareAction = action;
        node.textContent = label;
        if (title) node.title = title;
        node.disabled = disabled;
        return node;
    }

    /** How Share… passes the current result on here: 'telegram', 'web' or nothing. */
    function shareRoute(current) {
        if (current.format === 'text') {
            if (telegram()) return current.text.trim() && current.text.length <= TELEGRAM_TEXT_LIMIT ? 'telegram' : '';
            return canShare(win, { text: current.text }) ? 'web' : '';
        }
        return (current.format === 'file' ? current.file : current.card)?.shareable ? 'web' : '';
    }

    const chosenBlocks = (current) => (current.pressed || current.selection
        ? sliceBlocks(current.blocks, current.pressed || current.selection) : pickBlocks(current.blocks, current.chosen));

    /** The choice's count and Use excerpt, in place: a press on the footer keeps its button under the pointer. */
    function refreshChoice(current) {
        if (current.mode !== 'choose') return;
        const chosen = chosenBlocks(current);
        fmtInto(current.dialog.querySelector('.message-share-count'), '{n} of {total} words', { n: words(chosen), total: current.total });
        const use = current.dialog.querySelector('[data-share-action="use"]');
        if (use) use.disabled = !chosen.length;
    }

    // The count with what changes the choice, apart from the ways out: those keep together, the primary among them.
    function paintFoot(current) {
        const count = current.dialog.querySelector('.message-share-count');
        const scope = [count];
        const list = [];
        if (current.mode === 'choose') {
            list.push(button(tr('share.whole', 'Whole answer'), 'whole'), button(tr('share.use_excerpt', 'Use excerpt'), 'use', { primary: true }));
        } else {
            const total = current.total;
            const file = current.format === 'file';
            if (file) {
                const kind = current.cardSource ? current.file?.kind : 'answer';
                if (kind === 'answer') fmtInto(count, total === 1 ? 'Whole answer · {n} word' : 'Whole answer · {n} words', { n: total });
                else if (kind === 'components') fmtInto(count, 'The whole card, every tab and picture');
                else if (kind === 'app') fmtInto(count, 'The whole app, its code included');
                else count.textContent = '';
            } else if (current.excerpt) fmtInto(count, 'Excerpt · {n} of {total} words', { n: words(current.excerpt), total });
            else if (current.cardSource) fmtInto(count, total === 1 ? 'The card’s words · {n} word' : 'The card’s words · {n} words', { n: total });
            else if (total === 1) fmtInto(count, 'Whole answer · {n} word', { n: total });
            else fmtInto(count, 'Whole answer · {n} words', { n: total });
            // A file is always the whole message or card: an excerpt is Text's and Card's.
            if (!file) scope.push(button(tr('share.choose', 'Choose excerpt'), 'choose', { disabled: Boolean(current.telegram) }));
            const ready = current.format === 'text' || Boolean((file ? current.file : current.card)?.blob);
            if (current.format === 'text' || (current.format === 'card' && ready && canCopyImage(win))) list.push(button(tr('code.copy', 'Copy'), 'copy'));
            if (current.format !== 'text' && ready && !inTelegram()) list.push(button(tr('share.download', 'Download'), 'download'));
            const route = shareRoute(current);
            if (route) {
                list.push(button(tr('share.share', 'Share…'), 'share', { primary: true, disabled: Boolean(current.telegram),
                    title: route === 'telegram' ? tr('share.share_telegram', 'Share in Telegram') : '' }));
            }
            const quiet = current.dialog.querySelector('.message-share-status').hidden;
            if (quiet && file && ready && !list.length) {
                status(current, inTelegram()
                    ? tr('share.file_telegram', 'Telegram cannot take a file from here. Share the words as Text, or open Ouroboros in a browser to save the file.')
                    : tr('share.file_no_way', 'This app cannot save or share a file here. Share it as Text.'), 'warn');
            } else if (quiet && current.format === 'card' && ready && !list.length) {
                status(current, tr('share.card_no_way', 'This app cannot copy, save or share a picture here. Share it as Text.'), 'warn');
            } else if (quiet && current.format === 'text' && telegram() && current.text.length > TELEGRAM_TEXT_LIMIT) {
                status(current, fmt('Telegram takes up to {n} characters: choose an excerpt to share it there.', { n: TELEGRAM_TEXT_LIMIT }), 'warn');
            } else if (quiet && current.format === 'text' && inTelegram() && !route) {
                status(current, tr('share.telegram_old', 'This Telegram app cannot share messages from here. Copy the text, or update Telegram.'), 'warn');
            }
        }
        current.dialog.querySelector('.message-share-scope').replaceChildren(...scope);
        current.dialog.querySelector('.message-share-actions').replaceChildren(...list);
        refreshChoice(current);
    }

    /** The source's formats, the one in use pressed; while Telegram prepares a message the preview holds still. */
    function syncFormats(current) {
        for (const node of current.dialog.querySelectorAll('[data-share-format]')) {
            node.setAttribute('aria-pressed', String(current.mode === 'result' && node.dataset.shareFormat === current.format));
            node.disabled = Boolean(current.telegram);
        }
    }

    function paint(current) {
        if (session !== current) return;
        syncFormats(current);
        current.dialog.querySelector('.message-share-lead').textContent = current.mode === 'choose'
            ? tr('share.lead_choose', 'Tap the parts to share, or select exact words.')
            : tr('share.lead', 'Only this leaves. Your question and chat stay here.');
        if (current.mode === 'result' && current.format === 'card') ensureCard(current);
        if (current.mode === 'result' && current.format === 'file') ensureFile(current);
        paintBody(current);
        paintFoot(current);
    }

    // ---- the ways out, each in the press that asked for it
    async function copy(current, anchor) {
        if (current.format === 'card') {
            const blob = current.card?.blob;
            if (!blob) return;
            let write;
            try {
                // Built inside the press: an engine keeps a clipboard write only for the gesture that asked for it.
                write = win.navigator.clipboard.write([new win.ClipboardItem({ [blob.type || 'image/png']: blob })]);
            } catch (error) { write = Promise.reject(error); }
            try {
                await write;
                if (session === current) status(current, tr('share.image_copied', 'Picture copied.'), 'ok');
            } catch (error) {
                if (session === current) status(current, fmt('Copy failed: {reason}', { reason: reasonOf(error) }), 'error');
            }
            return;
        }
        try {
            await copyTextToClipboard(current.text, { win, doc, anchor });
            if (session === current) status(current, tr('share.copied', 'Copied.'), 'ok');
        } catch (error) {
            if (session === current) status(current, fmt('Copy failed: {reason}', { reason: reasonOf(error) }), 'error');
        }
    }

    async function download(current) {
        const file = current.format === 'file';
        const item = file ? current.file : current.card;
        if (!item?.blob) return;
        let result;
        try { result = await downloadBlobViaHostBridge(item.blob, item.name, { win, doc }); } catch (error) {
            result = { ok: false, error: reasonOf(error) };
        }
        if (session !== current) return;
        const reason = String(result?.error || 'unknown error');
        if (result?.ok && result.native) status(current, fmt('Saved to Downloads as {name}.', { name: result.filename || item.name }), 'ok');
        else if (result?.ok) status(current, tr('share.download_started', 'Download started.'), 'ok');
        else if (file) status(current, fmt('Could not save the file: {reason}', { reason }), 'error');
        else status(current, fmt('Could not save the picture: {reason}', { reason }), 'error');
    }

    function webShare(current) {
        const data = current.format === 'text' ? { text: current.text }
            : { files: [(current.format === 'file' ? current.file : current.card).file] };
        let shared;
        // Called in the press itself, with the payload canShare already accepted.
        try { shared = win.navigator.share(data); } catch (error) { shared = Promise.reject(error); }
        Promise.resolve(shared).then(() => {
            if (session === current) status(current, tr('share.handed', 'Passed to the share sheet.'), 'ok');
        }, (error) => {
            if (session !== current || error?.name === 'AbortError') return;  // the owner closed the sheet: not a failure
            status(current, fmt('Sharing did not start: {reason}', { reason: reasonOf(error) }), 'error');
        });
    }

    /** Stop hearing the last picker's outcome (a new press, or the sheet closing). */
    function forgetTelegram(current) {
        const wait = current.telegramWait;
        current.telegramWait = null;
        wait?.off();
    }

    // One press, one prepared message, one picker: Telegram tells the outcome, and nothing is sent again by itself.
    async function telegramShare(current) {
        const app = win.Telegram?.WebApp;
        forgetTelegram(current);
        const attempt = { text: current.text, version: current.version };
        current.telegram = attempt;  // the preview holds still until the picker has exactly these words
        status(current, tr('share.telegram_preparing', 'Preparing the message in Telegram…'));
        syncFormats(current);
        paintFoot(current);
        let prepared;
        try {
            prepared = await prepare(attempt.text, { signal: current.abort.signal });
        } catch (error) {
            if (session !== current || current.telegram !== attempt) return;
            current.telegram = null;
            status(current, fmt('Telegram could not prepare it: {reason}', { reason: reasonOf(error) }), 'error');
            syncFormats(current);
            paintFoot(current);
            return;
        }
        if (session !== current || current.telegram !== attempt) return;
        current.telegram = null;
        // Never a picker for other words than the preview shows.
        if (current.version !== attempt.version || current.mode !== 'result' || current.format !== 'text') {
            status(current);
            paint(current);
            return;
        }
        const offs = [];
        const wait = { off: () => { for (const off of offs.splice(0)) off(); } };
        current.telegramWait = wait;
        let outcome = '';
        const settle = (kind, code = '') => {
            if (session !== current || current.telegramWait !== wait) return;
            if (kind === 'sent') {
                outcome = kind;
                forgetTelegram(current);
                status(current, tr('share.telegram_sent', 'Sent in Telegram.'), 'ok');
            } else if (kind === 'failed') {
                if (outcome === 'sent') return;
                outcome = kind;
                forgetTelegram(current);
                if (code === 'USER_DECLINED') status(current);  // the owner closed the picker: no failure
                else if (TELEGRAM_FAILURES[code]) status(current, fmt(TELEGRAM_FAILURES[code]), code === 'MESSAGE_EXPIRED' ? 'warn' : 'error');
                else status(current, fmt('Telegram did not share it ({code}).', { code: code || 'UNKNOWN_ERROR' }), 'error');
            } else if (!outcome) {
                // The callback's `false`: not sent, its reason unknown until Telegram names one in its own event.
                outcome = kind;
                status(current, tr('share.telegram_not_sent', 'Not shared in Telegram.'));
            }
        };
        const listen = (event, handler) => {
            try { app.onEvent?.(event, handler); } catch { return; }
            offs.push(() => { try { app.offEvent?.(event, handler); } catch { /* gone */ } });
        };
        listen('shareMessageSent', () => settle('sent'));
        listen('shareMessageFailed', (event) => settle('failed', String(event?.error || '')));
        status(current);
        const called = hostCall(app, 'shareMessage', '8.0', prepared.id, (sent) => settle(sent === true ? 'sent' : 'not_sent'));
        if (!called) {
            forgetTelegram(current);
            status(current, fmt(TELEGRAM_FAILURES.UNSUPPORTED), 'error');
        }
        syncFormats(current);
        paintFoot(current);
    }

    function share(current) {
        const route = shareRoute(current);
        if (route === 'telegram' && !current.telegram) void telegramShare(current);
        else if (route === 'web') webShare(current);
    }

    function useExcerpt(current) {
        const pieces = current.pressed || current.selection;
        const excerpt = chosenBlocks(current);
        current.pressed = null;
        if (!excerpt.length) return;
        current.mode = 'result';
        current.selection = null;
        setExcerpt(current, !pieces && current.chosen.size === current.blocks.length ? null : excerpt);
        paint(current);
        current.body.focus({ preventScroll: true });
    }

    function toggle(current, row) {
        const index = Number(row.dataset.shareBlock);
        if (current.chosen.has(index)) current.chosen.delete(index); else current.chosen.add(index);
        row.setAttribute('aria-checked', String(current.chosen.has(index)));
        refreshChoice(current);
    }

    function onAction(current, action, target) {
        if (current.telegram && ['choose', 'whole', 'use'].includes(action)) return;  // the preview holds still
        if (action === 'close') close();
        else if (action === 'choose') { current.mode = 'choose'; current.selection = null; status(current); paint(current); }
        else if (action === 'whole') {
            current.mode = 'result';
            current.selection = null;
            current.chosen.clear();
            setExcerpt(current, null);
            paint(current);
        } else if (action === 'use') useExcerpt(current);
        else if (action === 'copy') void copy(current, target);
        else if (action === 'download') void download(current);
        else if (action === 'share') share(current);
    }

    function open(row, { anchor = null } = {}) {
        const cardSource = row?.answerWords ? null : row?.card || null;
        const source = cardSource ? String(cardSource.words || '') : String(row?.answerWords || '');
        if (!source && !cardSource) return false;
        close({ restoreFocus: false });
        // Read through the chat's own renderer into an inert template: nothing in it loads or runs. A card's words are
        // plain text (what Telegram gets), so they are read as they are.
        const template = doc.createElement('template');
        if (cardSource) plainWords(doc, template, source); else render(template, source);
        const blocks = shareBlocks(template.content);
        if (!blocks.length && !cardSource) throw new Error(tr('share.nothing', 'This message has no words to share.'));
        // A message has its words, their picture and its page; a card its page and its words. Inside Telegram (no file
        // leaves there) a card opens on its words, which its chat picker takes.
        const formats = cardSource ? (blocks.length ? ['file', 'text'] : ['file']) : ['text', 'card', 'file'];
        const format = cardSource && (!blocks.length || !inTelegram()) ? 'file' : 'text';
        const dialog = doc.createElement('dialog');
        dialog.className = 'document-reader message-share';
        dialog.setAttribute('aria-modal', 'true');
        dialog.innerHTML = SHEET_HTML;
        for (const node of dialog.querySelectorAll('[data-share-format]')) if (!formats.includes(node.dataset.shareFormat)) node.remove();
        if (formats.length < 2) dialog.querySelector('.document-reader-views')?.remove();
        opened += 1;
        const title = dialog.querySelector('.document-reader-title');
        title.id = `message-share-title-${opened}`;
        dialog.setAttribute('aria-labelledby', title.id);
        const bubble = cardSource?.node?.closest?.('.chat-bubble') || row.content?.parentElement || null;
        const current = {
            dialog, row, bubble, blocks, total: words(blocks), format, mode: 'result', chosen: new Set(),
            selection: null, pressed: null, dismissing: false, excerpt: null, text: shareText(blocks), version: 0, card: null,
            telegram: null, telegramWait: null, parts: null, body: dialog.querySelector('.message-share-body'),
            abort: new AbortController(), disposers: [], disposeFocus: null,
            content: row.content || null, cardSource, file: null, fileKey: 0,
        };
        current.disposers.push(() => forgetTelegram(current));
        const listen = (target, type, handler, options) => {
            if (!target?.addEventListener) return;
            target.addEventListener(type, handler, options);
            current.disposers.push(() => target.removeEventListener(type, handler, options));
        };
        listen(dialog, 'click', (event) => {
            if (session !== current) return;
            if (event.target === dialog) { close(); return; }  // the backdrop
            const format = event.target.closest?.('[data-share-format]')?.dataset.shareFormat;
            const action = event.target.closest?.('[data-share-action]');
            if (format) {
                if (current.telegram) return;
                current.mode = 'result';
                current.selection = null;
                current.format = format;
                status(current);
                paint(current);
            } else if (action) {
                if (!action.disabled) onAction(current, action.dataset.shareAction, action);
            } else {
                const part = event.target.closest?.('.message-share-part');
                // A drag that selected words inside the part is a selection, and a press that only put selected
                // words away is not a choice: neither is a tap.
                if (part && current.mode === 'choose' && !current.selection && !current.dismissing) toggle(current, part);
            }
        });
        listen(dialog, 'keydown', (event) => {
            current.body.classList.remove('is-quiet-focus');  // opened by a pointer: the ring from the first key
            const part = event.target.closest?.('.message-share-part');
            if (!part || event.target !== part || current.mode !== 'choose' || (event.key !== ' ' && event.key !== 'Enter')) return;
            event.preventDefault();
            toggle(current, part);
        });
        // Exact words are the preview's own selection, read as it changes. A press on the footer keeps what was
        // selected when it began: the shell clears a page selection under such a press (selection_lifecycle.js), and
        // that collapse is not the owner's new choice.
        listen(doc, 'selectionchange', () => {
            if (session !== current || current.mode !== 'choose' || !current.parts) return;
            const selection = doc.getSelection?.();
            const range = selection && !selection.isCollapsed && selection.rangeCount ? selection.getRangeAt(0) : null;
            const pieces = range ? selectedPieces(current.parts, range) : [];
            if (!pieces.length && current.pressed) return;
            current.selection = pieces.length ? pieces : null;
            refreshChoice(current);
        });
        listen(dialog, 'pointerdown', (event) => {
            const foot = Boolean(event.target.closest?.('.message-share-foot'));
            current.pressed = foot && current.selection ? current.selection : null;
            current.dismissing = !foot && Boolean(current.selection);
        }, true);
        // The press's click has run by the next task; a press abandoned elsewhere keeps nothing.
        listen(dialog, 'pointerup', () => {
            win.setTimeout?.(() => { if (session === current) { current.pressed = null; current.dismissing = false; } }, 0);
        }, true);
        listen(dialog, 'cancel', (event) => { event.preventDefault(); close(); });
        listen(dialog, 'close', () => { if (session === current) close({ restoreFocus: false }); });
        // Its room closes it with what else it opened: covered by another place, put away, or the row released.
        const release = row.holdTransient?.(() => { if (session === current) close({ restoreFocus: false }); });
        if (typeof release === 'function') current.disposers.push(release);
        doc.body.appendChild(dialog);
        session = current;
        paint(current);
        const returnFocus = anchor?.isConnected ? anchor : bubble;
        // The engine refocuses what was focused at showModal; WebKit does not focus a pressed button.
        if (returnFocus && returnFocus !== doc.activeElement) returnFocus.focus?.({ preventScroll: true });
        if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
        current.disposeFocus = bindDialogFocus(dialog, { initialFocus: current.body, returnFocus, onEscape: () => close(),
            preventScroll: true });
        return true;
    }

    const off = registry.register({
        key: SHARE, label: tr('share.label', 'Share'), title: tr('share.title', 'Share this message'), icon: 'share', order: 20,
        applies: (row) => Boolean(row?.answerWords || row?.card), run: (row, ctx) => { open(row, ctx); },
    });
    const onLeave = () => close({ restoreFocus: false });
    win?.addEventListener?.('ouro:panic', onLeave);
    win?.addEventListener?.('pagehide', onLeave);
    const unregister = () => {
        close({ restoreFocus: false });
        off();
        win?.removeEventListener?.('ouro:panic', onLeave);
        win?.removeEventListener?.('pagehide', onLeave);
    };
    unregister.close = close;
    unregister.isOpen = () => session !== null;
    unregister.open = open;
    return unregister;
}
