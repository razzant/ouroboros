// Coordinates in the rendered message, independent of link destinations and wrapping.
// CSS-collapsed inline whitespace takes one UTF-16 unit. Preserved text keeps its offsets;
// <br> and block boundaries contribute LF. Controls and opaque renderer output contribute
// none of their own; a rendered diagram or formula reserves its source's units with no DOM
// run, so a passage of that source cannot be placed on other words. A formula holds the same
// units as source text, drawn, or shown as the TeX the renderer could not parse.
import { tr } from './i18n.js';
import { foldHides, PROJECT_ROW_TYPES } from './project_answer.js';

const BLOCK = 'p, div, blockquote, ul, ol, li, table, thead, tbody, tfoot, tr, td, th, pre, h1, h2, h3, h4, h5, h6';
const EXCLUDED = '.chat-quote, button, input, textarea, select, summary, [contenteditable=true], '
    + '[hidden], [aria-hidden=true], details:not([open]), .md-code-language, .md-image-label, .md-diagram-error-note, .md-mermaid, .md-chart, .katex, '
    + 'script, style, svg, canvas, iframe, img, video, audio';
const TEXT_TYPES = new Set(['', 'message', 'text', 'main_notice', 'proactive_message', 'relay_reply']);
const sourceRow = element => element?.closest?.('[data-reply-caption], .chat-bubble');

export function excerptBody(row) {
    const bubble = row?.closest?.('.chat-bubble');
    if (!bubble?.matches('.user, .assistant') || !row.dataset.rowAddress?.startsWith('row:')
        || bubble.matches('.progress, [data-portal="1"], [data-origin-projected="1"], [data-ephemeral="1"], [data-reply-excerpt-disabled="1"]')
        || bubble.querySelector('.chat-quiz-card, .chat-routing-card, .chat-relay-deck')) return null;
    if (row.matches('[data-reply-caption]')) return row;
    const kind = row.dataset.systemType || '';
    if (TEXT_TYPES.has(kind) || (row.classList.contains('assistant') && PROJECT_ROW_TYPES.has(kind))) {
        const body = row.querySelector(':scope > .message[data-reply-body]:not([hidden])');
        if (body) return body;
    }
    return [...row.querySelectorAll('[data-reply-caption]')].find(node => node.dataset.rowAddress === row.dataset.rowAddress) || null;
}

export function validExcerpt(value) {
    return typeof value?.text === 'string' && value.text.trim() && Number.isSafeInteger(value.start_utf16)
        && value.start_utf16 >= 0 && !/[\uD800-\uDFFF]/u.test(value.text);
}

/** A rendered formula's source with the delimiters the renderer consumed: KaTeX keeps the
 * formula's own TeX in its MathML annotation, and every delimiter pair this client renders
 * spans the same two units (`$$` as `\[`), so the reconstruction holds the fallback text's
 * coordinates. `null` when the node is not rendered math or keeps no readable source. */
function renderedMathSource(node) {
    if (!node.matches('.katex')) return null;
    const source = node.querySelector('.katex-mathml annotation')?.textContent;
    if (typeof source !== 'string') return null;
    return node.closest('.katex-display') ? `\\[${source}\\]` : `\\(${source}\\)`;
}

// Every delimiter this client renders (`$$`, `\[`, `\]`, `\(`, `\)`) is two units. A formula KaTeX
// could not parse shows its TeX alone, with no trace of which pair it had: the pair's units stay
// reserved as these, and no passage reaches across them.
const LOST_DELIMITER = '\uFFFC\uFFFC';

function bodyStream(body) {
    const runs = [], barriers = [], separators = [], lost = [], collapse = new Map();
    let text = '', lastBlock = null, lastEnd = null, space = null;
    function separator(value, left, right) {
        separators.push({ at: text.length, left, right });
        text += value;
    }
    function literal(node, from, to) {
        if (space && text && !text.endsWith('\n')) separator(' ', ...space);
        space = null;
        runs.push({ node, from, to, at: text.length });
        text += node.data.slice(from, to);
        lastEnd = [node, to];
    }
    function enterBlock(block, start) {
        if (lastBlock === block) return;
        space = null;
        if (lastEnd && !text.endsWith('\n')) separator('\n', lastEnd, start);
        lastBlock = block;
    }
    function reserve(node, value, end = node.childNodes.length) {
        // Source text that lived in this block: it has coordinates, but no selectable run or
        // separator of its own.
        enterBlock(node.closest(BLOCK), [node, 0]);
        if (space && text && !text.endsWith('\n')) separator(' ', ...space);
        space = null;
        text += value;
        lastEnd = [node, end];
    }
    function reserveMathSource(node, value) {
        const host = (node.closest('.katex-display') || node).parentElement;
        const style = body.ownerDocument.defaultView.getComputedStyle(host);
        const collapses = style.whiteSpaceCollapse === 'collapse' || /^(normal|nowrap)$/.test(style.whiteSpace);
        reserve(node, collapses ? value.replace(/[\t\r\n\f ]+/g, ' ') : value);
    }
    function lostDelimiter(node, end) {
        reserve(node, LOST_DELIMITER, end);
        lost.push(text.length - LOST_DELIMITER.length);
    }
    function walk(node) {
        if (node.nodeType === 3) {
            if (!node.length) return;
            const parent = node.parentElement, block = parent.closest(BLOCK);
            if (!collapse.has(parent)) {
                const style = body.ownerDocument.defaultView.getComputedStyle(parent);
                collapse.set(parent, style.whiteSpaceCollapse === 'collapse' || /^(normal|nowrap)$/.test(style.whiteSpace));
            }
            const collapses = collapse.get(parent);
            // Whitespace-only nodes between block elements don't introduce their own line.
            if (collapses && /^[\t\r\n\f ]+$/.test(node.data) && lastBlock !== block) return;
            enterBlock(block, [node, 0]);
            if (!collapses) { literal(node, 0, node.length); return; }
            let from = 0;
            for (const match of node.data.matchAll(/[\t\r\n\f ]+/g)) {
                if (from < match.index) literal(node, from, match.index);
                from = match.index + match[0].length;
                space = [space?.[0] || [node, match.index], [node, from]];
            }
            if (from < node.length) literal(node, from, node.length);
        } else if (node.nodeType === 1) {
            if (node.matches(EXCLUDED)) {
                barriers.push(node);
                // Mermaid's fallback contributes its literal code. While the diagram renders, reserve those
                // same units with no DOM run: a quote of the code cannot land on matching prose instead.
                const diagram = node.matches('.md-mermaid') ? String(node.dataset.mermaidSource ?? node.textContent) : '';
                if (diagram) {
                    enterBlock(node, [node, 0]);
                    text += diagram;
                    lastEnd = [node, node.childNodes.length];
                } else {
                    // Rendered math replaces its source text the same way: reserve the formula's
                    // source, as the collapsible prose that held it read it, with no DOM run.
                    const formula = renderedMathSource(node);
                    if (formula !== null) reserveMathSource(node, formula);
                }
                return;
            }
            if (node.matches('.katex-error')) {
                // The TeX stays selectable where its source text stood, between its delimiters' units.
                lostDelimiter(node, 0);
                for (const child of node.childNodes) walk(child);
                lostDelimiter(node, node.childNodes.length);
                return;
            }
            if (node.tagName === 'BR') {
                const parent = node.parentNode, index = Array.prototype.indexOf.call(parent.childNodes, node);
                space = null;
                separator('\n', [parent, index], [parent, index + 1]);
                return;
            }
            for (const child of node.childNodes) walk(child);
        }
    }
    walk(body);
    // Whether the units [start, end) take in a delimiter the page no longer shows.
    const crossesLost = (start, end) => lost.some(at => start < at + LOST_DELIMITER.length && end > at);
    return { text, runs, barriers, separators, crossesLost };
}

// Dragging the handles changes the range, not the rendered body. Mutations invalidate synchronously too:
// a caller in the same turn as a Markdown/caption write must not see yesterday's coordinates.
const renderedStreams = new WeakMap();
function renderedStream(body) {
    const Observer = body.ownerDocument.defaultView.MutationObserver;
    if (!Observer) return bodyStream(body);
    let cached = renderedStreams.get(body);
    if (!cached) {
        cached = { stream: null };
        cached.observer = new Observer(() => { cached.stream = null; });
        cached.observer.observe(body, { subtree: true, childList: true, characterData: true, attributes: true });
        renderedStreams.set(body, cached);
    }
    if (cached.observer.takeRecords().length) cached.stream = null;
    return cached.stream ||= bodyStream(body);
}

export function readExcerpt(row, range) {
    const body = excerptBody(row);
    if (!body || !range || !body.contains(range.startContainer) || !body.contains(range.endContainer)
        || foldHides(range, row)) return null;
    const stream = renderedStream(body);
    if (stream.barriers.some(node => range.intersectsNode(node))) return null;
    let start = Infinity, end = 0;
    for (const { node, at, from, to } of stream.runs) {
        if (!range.intersectsNode(node)) continue;
        const left = range.startContainer === node ? Math.max(range.startOffset, from) : from;
        const right = range.endContainer === node ? Math.min(range.endOffset, to) : to;
        if (left >= right) continue;
        start = Math.min(start, at + left - from);
        end = at + right - from;
    }
    for (const edge of stream.separators) {
        if (range.comparePoint(...edge.left) === 0 && range.comparePoint(...edge.right) === 0) {
            start = Math.min(start, edge.at);
            end = Math.max(end, edge.at + 1);
        }
    }
    if (stream.crossesLost(start, end)) return null;
    const result = { text: stream.text.slice(start, end), start_utf16: start };
    return validExcerpt(result) ? result : null;
}

export function excerptRange(row, excerpt) {
    const body = excerptBody(row);
    if (!body || !validExcerpt(excerpt)) return null;
    const stream = renderedStream(body), start = excerpt.start_utf16, end = start + excerpt.text.length;
    if (stream.text.slice(start, end) !== excerpt.text || stream.crossesLost(start, end)) return null;
    function boundary(unit, after) {
        const run = stream.runs.find(({ at, from, to }) => at <= unit && unit < at + to - from);
        if (run) return [run.node, run.from + unit - run.at + Number(after)];
        const edge = stream.separators.find(item => item.at === unit);
        return after ? edge?.right : edge?.left;
    }
    const first = boundary(start, false), last = boundary(end - 1, true);
    if (!first || !last) return null;
    const range = body.ownerDocument.createRange();
    range.setStart(...first);
    range.setEnd(...last);
    return stream.barriers.some(node => range.intersectsNode(node)) ? null : range;
}

// Each document has one quote landing. A later jump clears only its own paint or fallback selection.
const landedExcerpts = new WeakMap();

export function clearExcerptHighlight(doc) {
    landedExcerpts.get(doc)?.();
}

export function highlightExcerpt(range, duration) {
    const doc = range.startContainer.ownerDocument, win = doc.defaultView;
    clearExcerptHighlight(doc);
    let remove;
    if (win.CSS?.highlights && win.Highlight) {
        const mark = new win.Highlight(range);
        win.CSS.highlights.set('reply-excerpt', mark);
        remove = () => {
            if (win.CSS.highlights.get('reply-excerpt') === mark) win.CSS.highlights.delete('reply-excerpt');
        };
    } else {
        const selection = doc.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        const owned = selection.getRangeAt(0);
        const original = owned.cloneRange();
        remove = () => {
            if (selection.rangeCount === 1 && selection.getRangeAt(0) === owned
                && owned.compareBoundaryPoints(win.Range.START_TO_START, original) === 0
                && owned.compareBoundaryPoints(win.Range.END_TO_END, original) === 0)
                selection.removeAllRanges();
        };
    }
    const timer = win.setTimeout(clear, duration);
    function clear() {
        win.clearTimeout(timer);
        remove();
        landedExcerpts.delete(doc);
    }
    landedExcerpts.set(doc, clear);
}

// AppKit receives only a single-use menu ticket. The addressed range stays in its document.
const nativeClaims = new WeakMap();

function installNativeQuote(win) {
    if (!win || win.ouroReplyToQuote) return;
    win.ouroReplyToQuote = Object.freeze({
        available(ticket) {
            const claim = nativeClaims.get(win);
            if (!claim || !claim.current()) return false;
            if (ticket) {
                if (claim.ticket && claim.ticket !== ticket) return false;
                claim.ticket = ticket;
            }
            return true;
        },
        activate(ticket) {
            const claim = nativeClaims.get(win);
            if (!claim || claim.ticket !== ticket || !claim.current()) return false;
            nativeClaims.delete(win);
            return claim.stage();
        },
    });
}

// Select text lets touch devices use their own word handles on this row. Keeping the mode while the
// reader adjusts (or makes a fresh long press) matters on engines that don't show handles for a scripted range.
export function bindExcerptAction({ feed, present, choose }) {
    const doc = feed.ownerDocument, win = doc.defaultView;
    let button = null, candidate = null, touchRow = null, touchBody = null, pressing = false, secondary = false;
    installNativeQuote(win);
    const clearClaim = () => { if (nativeClaims.get(win)?.feed === feed) nativeClaims.delete(win); };
    function stage(picked) {
        if (!picked || !present() || !feed.contains(picked.row) || picked.address !== picked.row.dataset.rowAddress) return false;
        const range = excerptRange(picked.row, picked.excerpt);
        if (!range || foldHides(range, picked.row)) return false;
        choose(picked.row, picked.excerpt);
        doc.getSelection()?.removeAllRanges();
        hide(); leaveTouch(); clearClaim();
        return true;
    }
    const hideButton = () => { button?.remove(); button = null; pressing = false; };
    const hide = () => { hideButton(); candidate = null; };
    const leaveTouch = () => { if (touchRow) delete touchRow.dataset.selectText; touchRow = touchBody = null; };
    function refresh() {
        // Touch engines may dismiss native handles between pointerdown and click.
        if (pressing && present()) return;
        if (secondary && !nativeClaims.get(win)?.current()) { clearClaim(); hide(); return; }
        const selection = doc.getSelection?.();
        if (!selection || selection.rangeCount !== 1 || selection.isCollapsed || !present()) { hide(); return; }
        const range = selection.getRangeAt(0);
        const el = range.commonAncestorContainer;
        const row = sourceRow(el.nodeType === 1 ? el : el.parentElement);
        const excerpt = row && feed.contains(row) && readExcerpt(row, range);
        if (!excerpt) { hide(); return; }
        candidate = { row, address: row.dataset.rowAddress, excerpt };
        position(range);
    }
    function position(range) {
        const bounds = feed.getBoundingClientRect();
        const rect = [...range.getClientRects()].find(r => r.width && r.height && r.right > bounds.left && r.left < bounds.right
            && r.bottom > Math.max(0, bounds.top) && r.top < Math.min(win.innerHeight, bounds.bottom));
        if (!rect) { hideButton(); return; }
        if (!button) {
            button = doc.createElement('button');
            button.type = 'button';
            button.className = 'btn btn-float btn-sm ui-hit-area chat-reply-selection-action';
            button.textContent = tr('chat.reply.to_quote', 'Reply to quote');
            button.addEventListener('pointerdown', e => { pressing = true; e.preventDefault(); });
            button.addEventListener('pointercancel', () => { pressing = false; refresh(); });
            button.addEventListener('mousedown', e => e.preventDefault());
            button.addEventListener('click', () => {
                if (!stage(candidate)) { hide(); leaveTouch(); }
            });
            doc.body.append(button);
        }
        const width = button.offsetWidth, height = button.offsetHeight;
        button.style.setProperty('--quote-left', Math.max(4, Math.min(rect.left, win.innerWidth - width - 4)) + 'px');
        button.style.setProperty('--quote-top', Math.max(4, Math.min(win.innerHeight - height - 4,
            rect.top >= height + 6 ? rect.top - height - 4 : rect.bottom + 4)) + 'px');
    }
    function outside(event) {
        clearClaim();
        secondary = event.button === 2 || (event.button === 0 && event.ctrlKey);
        if (secondary && present()) {
            const selected = doc.getSelection?.();
            const range = selected?.rangeCount === 1 && !selected.isCollapsed && selected.getRangeAt(0);
            const row = sourceRow(event.target);
            if (range && row && feed.contains(row) && excerptBody(row)?.contains(event.target)
                && [...range.getClientRects()].some(rect => event.clientX >= rect.left && event.clientX <= rect.right
                    && event.clientY >= rect.top && event.clientY <= rect.bottom)) {
                const excerpt = readExcerpt(row, range), address = row.dataset.rowAddress;
                if (excerpt) nativeClaims.set(win, {
                    feed, stage: () => stage({ row, excerpt, address }),
                    current() {
                        const selection = doc.getSelection();
                        if (!present() || !feed.contains(row) || row.dataset.rowAddress !== address
                            || selection.rangeCount !== 1 || selection.isCollapsed) return false;
                        const now = readExcerpt(row, selection.getRangeAt(0));
                        return now?.start_utf16 === excerpt.start_utf16 && now.text === excerpt.text;
                    },
                });
            }
        }
        if (pressing && event.target !== button) { pressing = false; refresh(); }
        if (touchBody && !touchBody.contains(event.target) && event.target !== button) leaveTouch();
    }
    function escape(event) {
        secondary = false;
        clearClaim();
        if (event.key === 'Escape') { hide(); leaveTouch(); }
    }
    const pageChanged = () => { hide(); leaveTouch(); clearClaim(); };
    win?.addEventListener?.('ouro:page-shown', pageChanged);
    win?.addEventListener?.('pagehide', pageChanged);
    doc.addEventListener('selectionchange', refresh);
    doc.addEventListener('pointerdown', outside, true);
    doc.addEventListener('keydown', escape);
    const reposition = () => {
        if (!candidate || pressing) return;
        const selection = doc.getSelection?.();
        if (!present() || !feed.contains(candidate.row) || selection?.rangeCount !== 1 || selection.isCollapsed) {
            hide(); return;
        }
        position(selection.getRangeAt(0));
    };
    doc.addEventListener('scroll', reposition, true);
    win?.addEventListener?.('resize', reposition);
    const atPoint = (row, point) => {
        const chosen = point && sourceRow(doc.elementFromPoint?.(point.x, point.y));
        return chosen && row.contains(chosen) ? chosen : row;
    };
    return {
        canSelect: (row, point) => Boolean(excerptBody(atPoint(row, point))),
        select(row, point) {
            const body = excerptBody(atPoint(row, point));
            if (!body) return;
            leaveTouch();
            touchRow = row;
            touchBody = body;
            row.dataset.selectText = '1';
            delete row.dataset.rowPress;
            const stream = bodyStream(body);
            const caret = point && (doc.caretPositionFromPoint?.(point.x, point.y) || doc.caretRangeFromPoint?.(point.x, point.y));
            const node = caret?.offsetNode || caret?.startContainer;
            const run = stream.runs.find(item => item.node === node) || stream.runs.find(item => item.node.data.trim());
            if (!run) return;
            const at = run.node === node ? (caret.offset ?? caret.startOffset) : 0;
            const words = [...run.node.data.matchAll(/\S+/gu)];
            const word = words.find(m => m.index + m[0].length > at) || words.at(-1);
            if (!word) return;
            const range = doc.createRange(), selection = doc.getSelection();
            range.setStart(run.node, word.index);
            range.setEnd(run.node, word.index + word[0].length);
            selection.removeAllRanges(); selection.addRange(range);
        },
        destroy() {
            pageChanged();
            win?.removeEventListener?.('ouro:page-shown', pageChanged);
            win?.removeEventListener?.('pagehide', pageChanged);
            doc.removeEventListener('selectionchange', refresh);
            doc.removeEventListener('pointerdown', outside, true);
            doc.removeEventListener('keydown', escape);
            doc.removeEventListener('scroll', reposition, true);
            win?.removeEventListener?.('resize', reposition);
        },
    };
}
