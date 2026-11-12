// The words a saved message shows, and where a search's query stands in them (docs/DESIGN.md "Search in a chat").
//
// A message's words are what the room's own renderer draws, never its stored source: `candidateBox` draws a search
// candidate (GET /api/chat/search) through the same functions its bubble uses (chat_text_bubble.js `bubbleWords`: the
// owner's literal words, Ouroboros's Markdown, a System sentence in the install's language; a Project ending's typed
// view, a links row, a question's Markdown, a file's shown name and caption) into an inert document, which loads, runs
// and fetches nothing; `wordPieces` reads those words from that box or from the landed row in the feed alike. A link's
// hidden destination, an image's address, a code block's language and controls are not words (an address the text
// itself shows is); a diagram or a chart is one opaque separator, never its source; a formula is its characters as
// KaTeX draws them, in reading order (its MathML), never its TeX source. One flow serves the candidate's confirmation,
// its snippet and the landed row's ranges: white space as one space, NFC per grapheme (`wordStream`), then Unicode 17
// full default case folding (C+F, never T: unicode_case_folding.js, the same in every browser engine), each folded
// unit mapped back to the characters it came from. A match starts and ends on whole graphemes and whole folds:
// `Straße` holds `STRASSE`, never `S` alone inside `ß`; `é` is never `e`.
import { CASE_FOLDING } from './unicode_case_folding.js';
import { bubbleWords } from './chat_text_bubble.js';
import { mountChatMarkdown, renderLatexIn } from './chat_markdown.js';
import { attachmentViews } from './chat_attachments.js';
import { fileNames, linksMessageHtml } from './chat_media.js';
import { PROJECT_ROW_TYPES, paintProjectEnding } from './project_answer.js';

// Python's str.split() white space, which the host also collapses in the query it binds (chat_search.py::_words).
const SPACE = /[\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/;
const SPACES = new RegExp(`${SPACE.source}+`, 'g');
// What of a row is its message's words: the text, the names an owner's attachments show and a question's question,
// never its author, time, controls, a quote of another message, a code block's language, a player, a photo's menu, a
// closed disclosure's body, what is hidden, a formula's TeX source (`annotation`) or what assistive technology does not
// read (a formula's drawn glyphs).
const WORDS = '.message, .chat-quiz-question, .chat-attachments';
const CHROME = '.sender, .msg-time, .chat-message-actions, .chat-quote, .md-code-language, button, .chat-media-player, '
    + '.chat-photo-actions, [hidden], [aria-hidden="true"], details:not([open]) > :not(summary), annotation';
// A file's shown name is words even where it stands in its card's button or its player's title.
const FILE_NAME = '.chat-file-name, .chat-media-title';
const FILE_NAME_HOSTS = '.chat-file-card, .chat-media-player';
const GROUP_ITEMS = '.chat-gallery-item, .chat-file-item';
// Drawn, not written: a diagram and a chart separate the words around them and hold none.
const OPAQUE = '.md-mermaid, .md-chart';
// Elements whose edges separate words as the stored text's line breaks do (a link button and display math stand
// apart too).
const EDGES = new Set(['BR', 'P', 'DIV', 'LI', 'UL', 'OL', 'PRE', 'BLOCKQUOTE', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
    'TABLE', 'TR', 'TD', 'TH', 'HR', 'FIGCAPTION', 'DL', 'DT', 'DD', 'SUMMARY']);
const EDGE_CLASS = '.chat-link-button, .katex-display';
const SNIPPET_CHARS = 160;
const LEAD_CHARS = 48;

/** A query as typed, as the search reads it: every run of white space one space, none at either end. */
export const queryWords = (value) => String(value ?? '').replace(SPACES, ' ').replace(/^ | $/g, '');

// ---- what a message shows -----------------------------------------------------------------------------------------

/** A box holding what `candidate` (`ChatSearchCandidate`) shows as its words, drawn into `doc` (an inert document) as
 *  the room draws its row. Media and files are never drawn: a photo or video is its caption, a file its card's name
 *  and its caption. */
export function candidateBox(candidate, doc) {
    const box = doc.createElement('div');
    const words = (className, html, rich = false) => {
        const node = doc.createElement('div');
        node.className = className;
        node.innerHTML = html;
        box.append(node);
        if (rich) renderLatexIn(node);
        return node;
    };
    const kind = candidate.msg_type || '';
    if (kind === 'photo' || kind === 'video' || kind === 'document') {
        const message = words('message', '');
        if (kind === 'document') {
            message.append(Object.assign(doc.createElement('div'), { className: 'chat-file-name' }));
            message.lastChild.textContent = fileNames(candidate).shown;
        }
        if (candidate.caption) message.append(doc.createTextNode(String(candidate.caption)));
        return box;
    }
    if (kind === 'links') {
        box.innerHTML = linksMessageHtml(candidate);
        return box;
    }
    if (kind === 'quiz') {  // the question as its card mounts it (chat_decision.js)
        const question = String(candidate.quiz?.question || '');
        if (question) {
            const node = words('chat-quiz-question', '');
            mountChatMarkdown(node, question);
            renderLatexIn(node);
        }
        return box;
    }
    const { role, systemType, text, markdown } = shownRow(candidate);
    const owned = role === 'user' && Array.isArray(candidate.attachments)
        ? candidate.attachments.filter((view) => view && typeof view === 'object') : [];
    if (owned.length) ownerAttachmentNames(words('chat-attachments', ''), owned);
    const { rendered, rich } = bubbleWords(text, role, { markdown, systemType, source: candidate.source || '',
        attachments: attachmentViews(owned), narration: candidate.narration,
        textPlaceholder: candidate.text_placeholder === true });
    const message = words(rich ? 'message ui-rich-content' : 'message', rendered, rich);
    if (systemType === 'project_completion_summary' && candidate.project_id) {
        paintProjectEnding(message, { role, completionNotice: candidate.completion_notice || null,
            taskName: candidate.task_name || '', projectName: candidate.project_name || '' });
    }
    return box;
}

/** The names an owner's attachments show above the owner's words, in the bubble's order (chat_attachments.js
 *  `buildAttachmentBlock`: images, players, files): a player's title and a file card's name; an available image shows
 *  only its picture. `owned` are the candidate's `{name, kind, available}`, named as `attachmentViews` names them. */
function ownerAttachmentNames(box, owned) {
    const views = attachmentViews(owned).map((view, at) => ({ name: view.name, available: owned[at].available === true,
        kind: owned[at].available === true && MEDIA_KINDS.has(owned[at].kind) ? owned[at].kind : 'file' }));
    const players = views.filter((view) => view.available && (view.kind === 'video' || view.kind === 'audio'));
    const files = views.filter((view) => !view.available || view.kind === 'file');
    for (const view of [...players, ...files]) {
        box.append(Object.assign(box.ownerDocument.createElement('div'), { className: 'chat-file-name',
            textContent: view.name }));
    }
}
const MEDIA_KINDS = new Set(['image', 'video', 'audio', 'file']);

/** The author and words a candidate's row is drawn with: a Project lifecycle row is its authored answer in
 *  Ouroboros's voice, else System text (project_answer.js `appendProjectRow`); every other row as history gives it. */
export function shownRow(candidate) {
    const systemType = String(candidate.system_type || '');
    if (!PROJECT_ROW_TYPES.has(systemType)) {
        return { role: candidate.role, systemType, text: String(candidate.text ?? ''), markdown: candidate.markdown === true };
    }
    const answer = String(candidate.completion_answer || '');
    return { role: answer ? 'assistant' : 'system', systemType, text: answer || String(candidate.text ?? ''),
        markdown: Boolean(answer || candidate.markdown === true) };
}

/** A row's message words in reading order as `wordStream` pieces: `[textNode, text]`, or `[null, ' ']` for an edge
 *  between words (a block, a line break, something drawn). `row` is a landed row or a `candidateBox`. */
export function wordPieces(row) {
    const pieces = [];
    const walk = (node) => {
        if (node.nodeType === 3) { if (node.data) pieces.push([node, node.data]); return; }
        if (node.nodeType !== 1) return;
        if (node.matches(CHROME)) {
            for (const name of node.matches(FILE_NAME_HOSTS) ? node.querySelectorAll(FILE_NAME) : []) {
                pieces.push([null, ' ']);
                for (const child of name.childNodes) walk(child);
                pieces.push([null, ' ']);
            }
            return;
        }
        if (node.matches(OPAQUE)) { pieces.push([null, ' ']); return; }
        const edge = EDGES.has(node.tagName) || node.matches(EDGE_CLASS);
        if (edge) pieces.push([null, ' ']);
        for (const child of node.childNodes) walk(child);
        if (edge) pieces.push([null, ' ']);
    };
    // A grouped photo or file (chat_media.js) is a row of its own inside its bubble's message: its words are its own.
    for (const box of row.matches?.(GROUP_ITEMS) ? [row] : row.querySelectorAll(WORDS)) {
        const outer = box.parentElement?.closest(`${WORDS}, ${CHROME}, ${OPAQUE}`);
        if (!outer || !row.contains(outer)) walk(box);
    }
    return pieces;
}

// ---- the words as a stream ----------------------------------------------------------------------------------------

/**
 * `pieces` as one stream of words: every run of white space one space (none at either end), NFC per grapheme. `text`
 * is that stream; its unit `i` comes from the raw units `from[i]` up to `to[i]` (a grapheme composes together), and
 * raw unit `r` is `places[r]`: `[node, offset]`, or null for an edge.
 */
export function wordStream(pieces) {
    let raw = '';
    const places = [];
    for (const [node, text] of pieces) {
        for (let index = 0; index < text.length; index += 1) {
            const space = !node || SPACE.test(text[index]);
            if (space && (!raw || raw.endsWith(' '))) continue;
            raw += space ? ' ' : text[index];
            places.push(node ? [node, index] : null);
        }
    }
    if (raw.endsWith(' ')) { raw = raw.slice(0, -1); places.pop(); }
    let text = '';
    const from = [], to = [];
    for (const cluster of (graphemes ||= new Intl.Segmenter('und', { granularity: 'grapheme' })).segment(raw)) {
        const composed = cluster.segment.normalize('NFC');
        for (let unit = 0; unit < composed.length; unit += 1) {
            from.push(cluster.index);
            to.push(cluster.index + cluster.segment.length);
        }
        text += composed;
    }
    return { text, from, to, places };
}

let graphemes = null, folds = null;
function foldOf(code) {
    if (!folds) {
        folds = new Map();
        for (const entry of CASE_FOLDING.split(';')) {
            const [code, to] = entry.split(':');
            folds.set(parseInt(code, 16), String.fromCodePoint(...to.split(' ').map((part) => parseInt(part, 16))));
        }
    }
    return folds.get(code);
}

/**
 * `text` case-folded by Unicode 17's full default folding, code point by code point (a code point the table does not
 * list is itself). With `map`, `start[i]` and `end[i]` are the units of `text` the folded unit `i` came from: an
 * expansion (ß → ss, ﬁ → fi, İ → i̇) maps each of its units to the one character, a surrogate pair maps whole.
 */
export function foldText(text, map = false) {
    let folded = '';
    const start = map ? [] : null, end = map ? [] : null;
    for (let at = 0; at < text.length;) {
        const code = text.codePointAt(at);
        const width = code > 0xffff ? 2 : 1;
        const to = foldOf(code) ?? text.slice(at, at + width);
        if (map) for (let unit = 0; unit < to.length; unit += 1) { start.push(at); end.push(at + width); }
        folded += to;
        at += width;
    }
    return { folded, start, end };
}

/** The query as words are matched: `queryWords`, NFC per grapheme and folded (an empty query is ''). */
export const searchKey = (value) => foldText(wordStream([[true, queryWords(value)]]).text).folded;

/**
 * Where `key` (`searchKey`) stands in `stream`, as `[start, end)` units of its text in reading order, without overlaps.
 * A match starts and ends on whole folds of whole graphemes, so part of a letter (the `s` of `ß`, the base of a
 * letter with a mark that does not compose) is never one.
 */
export function findIn(stream, key) {
    if (!key || !foldText(stream.text).folded.includes(key)) return [];  // most words hold no match: no map for them
    const { folded, start, end } = foldText(stream.text, true);
    const { from } = stream;
    const found = [];
    for (let at = folded.indexOf(key); at >= 0; at = folded.indexOf(key, at + 1)) {
        const last = at + key.length - 1;
        const begin = start[at], stop = end[last];
        if ((at && start[at - 1] === begin) || (last + 1 < folded.length && start[last + 1] === start[last])) continue;
        if ((begin && from[begin - 1] === from[begin]) || (stop < from.length && from[stop] === from[stop - 1])) continue;
        if (found.length && begin < found.at(-1)[1]) continue;
        found.push([begin, stop]);
    }
    return found;
}

/** The words around the first match as runs, `match` marking every occurrence in the window: about SNIPPET_CHARS
 *  characters from a little before the first, cut on word boundaries, a cut shown with an ellipsis. */
export function snippetOf({ text }, found) {
    if (!found.length) return [{ text: text.slice(0, SNIPPET_CHARS), match: false }];
    const [first, last] = found[0];
    let start = Math.max(0, first - LEAD_CHARS);
    if (start) {  // begin on a word: after the space before it, if the lead holds one
        const space = text.indexOf(' ', start);
        start = space >= 0 && space < first ? space + 1 : start;
    }
    let end = Math.max(last, Math.min(text.length, start + SNIPPET_CHARS));
    if (end < text.length) {
        const space = text.lastIndexOf(' ', end);
        end = space > last ? space : end;
    }
    const parts = start ? [{ text: '…', match: false }] : [];
    let cursor = start;
    for (const [begin, stop] of found) {
        if (begin < start || stop > end) continue;
        if (begin > cursor) parts.push({ text: text.slice(cursor, begin), match: false });
        parts.push({ text: text.slice(begin, stop), match: true });
        cursor = stop;
    }
    if (cursor < end) parts.push({ text: text.slice(cursor, end), match: false });
    if (end < text.length) parts.push({ text: '…', match: false });
    return parts;
}

/** Where a match's end at `place` is marked: a formula's characters are its MathML, which is not drawn, so a match
 *  in a formula marks the formula as drawn, whole (KaTeX's `.katex-html`, after its MathML). */
function drawnAt([node, offset], end) {
    const drawn = node.parentElement?.closest('.katex')?.querySelector(':scope > .katex-html');
    if (!drawn) return end ? [node, offset + 1] : [node, offset];
    return end ? [drawn, drawn.childNodes.length] : [drawn, 0];
}

/** DOM ranges of `found` in `stream`, first match first; a match whose end is not a character of the row (only an
 *  edge) has none. */
export function rangesOf(stream, found, doc = globalThis.document) {
    return found.map(([start, end]) => {
        const first = stream.places[stream.from[start]], last = stream.places[stream.to[end - 1] - 1];
        if (!first || !last) return null;
        const range = doc.createRange();
        range.setStart(...drawnAt(first, false));
        range.setEnd(...drawnAt(last, true));
        return range;
    }).filter(Boolean);
}

/** A landed row's matching words as DOM ranges, read exactly as its candidate was. */
export function rangesIn(row, key) {
    if (!row?.querySelectorAll || typeof globalThis.document?.createRange !== 'function') return [];
    const stream = wordStream(wordPieces(row));
    return rangesOf(stream, findIn(stream, key));
}
