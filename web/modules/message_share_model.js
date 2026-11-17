// What Share passes on (docs/DESIGN.md "Replies"): one Ouroboros message's own words, the source its Copy writes
// (chat.js records it), read as blocks through the chat's own sanitized Markdown rendering. The preview, an
// excerpt, the Text and the Card all come from this one model. Text is readable plain text: Markdown marks go, a
// link keeps its whole address beside its words, code keeps every space and line inside a fence. A table, a
// formula, a diagram or a chart stays in Text as its words or source; a card cannot draw them and says so. Nothing
// here touches the page: the walker reads any node tree with nodeType, tagName, childNodes and attributes.

const ELEMENT = 1;
const TEXT = 3;
const SKIPPED = new Set(['BUTTON', 'SVG', 'SCRIPT', 'STYLE', 'TEMPLATE', 'IMG', 'VIDEO', 'AUDIO', 'IFRAME', 'INPUT',
    'SELECT', 'TEXTAREA', 'NOSCRIPT']);
const INLINE = new Set(['A', 'ABBR', 'B', 'BDI', 'BDO', 'BR', 'CITE', 'CODE', 'DEL', 'DFN', 'EM', 'I', 'INS', 'KBD',
    'MARK', 'Q', 'S', 'SAMP', 'SMALL', 'SPAN', 'STRONG', 'SUB', 'SUP', 'TIME', 'U', 'VAR', 'WBR']);
// The chat's math delimiters (chat_markdown.js renderLatexIn): the unenhanced render keeps them as text.
const MATH = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)/;
const RTL = /[\p{Script=Hebrew}\p{Script=Arabic}\p{Script=Syriac}\p{Script=Thaana}\p{Script=Nko}]/u;
const STYLE_KEYS = ['bold', 'italic', 'code', 'strike', 'href', 'link'];

const tagOf = (node) => String(node?.tagName || '').toUpperCase();
const childrenOf = (node) => Array.from(node?.childNodes || []);
const classesOf = (node) => String(node?.getAttribute?.('class') ?? node?.className ?? '').split(/\s+/);
const hasClass = (node, name) => node?.nodeType === ELEMENT && classesOf(node).includes(name);
const sameStyle = (a, b) => STYLE_KEYS.every((key) => (a[key] || '') === (b[key] || ''));

/** A link's address worth passing on: an absolute web or mail address. The app's own relative routes stay its words. */
export function shareHref(value) {
    const text = String(value || '').trim();
    return /^(?:https?:\/\/|mailto:)\S/i.test(text) ? text : '';
}

function addRun(runs, text, style) {
    if (!text) return;
    const last = runs[runs.length - 1];
    if (last && sameStyle(last, style)) last.text += text;
    else runs.push({ ...style, text });
}

/** The inline runs of `nodes`, as the browser lays them out: prose whitespace collapsed, `<br>` a line break, inline
 * code keeping its own spaces, a link's runs sharing its number and address. */
function inlineRuns(nodes, style, ctx, runs = []) {
    for (const node of nodes) {
        if (node.nodeType === TEXT) {
            const raw = String(node.nodeValue ?? node.textContent ?? '');
            addRun(runs, style.code ? raw.replace(/\r?\n/g, ' ') : raw.replace(/[ \t\n\r\f]+/g, ' '), style);
            continue;
        }
        if (node.nodeType !== ELEMENT) continue;
        const tag = tagOf(node);
        if (SKIPPED.has(tag) || hasClass(node, 'md-checkbox') || node.getAttribute?.('aria-hidden') === 'true') continue;
        if (tag === 'BR') { addRun(runs, '\n', style); continue; }
        let next = style;
        if (tag === 'STRONG' || tag === 'B') next = { ...style, bold: true };
        else if (tag === 'EM' || tag === 'I') next = { ...style, italic: true };
        else if (tag === 'DEL' || tag === 'S') next = { ...style, strike: true };
        else if (tag === 'CODE' || tag === 'KBD' || tag === 'SAMP') next = { ...style, code: true };
        else if (tag === 'A') {
            ctx.links += 1;
            next = { ...style, link: ctx.links, href: shareHref(node.getAttribute?.('href')) };
        }
        inlineRuns(childrenOf(node), next, ctx, runs);
    }
    return runs;
}

/** Runs as one block shows them: nothing before the first word or after the last, no space beside a line break. */
export function tidyRuns(runs) {
    const out = [];
    for (const run of runs) {
        let text = String(run.text || '');
        const previous = out[out.length - 1];
        if (!run.code) {
            if (!previous) text = text.replace(/^[ \n]+/, '');
            else if (/[ \n]$/.test(previous.text)) text = text.replace(/^ +/, '');
            text = text.replace(/ *\n */g, '\n');
            if (text.startsWith('\n') && previous && !previous.code) previous.text = previous.text.replace(/ +$/, '');
        }
        if (!text) continue;
        if (previous && sameStyle(previous, run)) previous.text += text;
        else out.push({ ...run, text });
    }
    while (out.length) {
        const last = out[out.length - 1];
        if (!last.code) last.text = last.text.replace(/[ \n]+$/, '');
        if (last.text) break;
        out.pop();
    }
    return out.filter((run) => run.text);
}

const block = (kind, ctx, extra) => ({ kind, depth: ctx.depth, quote: ctx.quote, runs: [], ...extra });

function findDescendant(node, test, depth = 3) {
    for (const child of childrenOf(node)) {
        if (child.nodeType !== ELEMENT) continue;
        if (test(child)) return child;
        if (depth > 1) {
            const found = findDescendant(child, test, depth - 1);
            if (found) return found;
        }
    }
    return null;
}

function* rowsOf(node) {
    for (const child of childrenOf(node)) {
        if (child.nodeType !== ELEMENT) continue;
        if (tagOf(child) === 'TR') yield child;
        else yield* rowsOf(child);
    }
}

function codeBlock(node, ctx, out) {
    const code = tagOf(node) === 'PRE' ? (findDescendant(node, (el) => tagOf(el) === 'CODE', 1) || node)
        : findDescendant(node, (el) => tagOf(el) === 'CODE', 2);
    const language = classesOf(code).find((name) => name.startsWith('language-'))?.slice('language-'.length) || '';
    // The renderer ends a block's code with one line break of its own (marked); every other line is the author's.
    const text = String(code?.textContent ?? '').replace(/\n$/, '');
    out.push(block('code', ctx, { lang: language === 'plain' ? '' : language, runs: [{ text, code: true }] }));
}

function tableBlock(node, ctx, out) {
    const rows = [];
    for (const row of rowsOf(node)) {
        const cells = childrenOf(row).filter((cell) => ['TD', 'TH'].includes(tagOf(cell)))
            .map((cell) => runsText(tidyRuns(inlineRuns(childrenOf(cell), {}, ctx))).replace(/\n/g, ' '));
        if (cells.length) rows.push(cells.join(' | '));
    }
    if (rows.length) out.push(block('table', ctx, { runs: [{ text: rows.join('\n'), verbatim: true }] }));
}

function listBlocks(node, ordered, ctx, out) {
    const start = Number.parseInt(node.getAttribute?.('start') || '1', 10);
    let number = Number.isFinite(start) ? start : 1;
    for (const item of childrenOf(node)) {
        if (tagOf(item) !== 'LI') continue;
        // The task box is the item's own, at its head (a loose item holds it in its first paragraph).
        const box = findDescendant(item, (el) => hasClass(el, 'md-checkbox'), 2);
        const task = box ? (hasClass(box, 'is-checked') ? 'done' : 'open') : '';
        const marker = ordered ? `${number}.` : '•';
        number += 1;
        const inner = [];
        collectBlocks(item, { ...ctx, depth: ctx.depth + 1 }, inner);
        if (inner[0]?.kind === 'paragraph' && inner[0].depth === ctx.depth + 1) {
            Object.assign(inner[0], { kind: 'item', depth: ctx.depth, marker, task });
        } else {
            inner.unshift(block('item', ctx, { marker, task }));
        }
        out.push(...inner);
    }
}

function collectBlocks(container, ctx, out) {
    let inline = [];
    const flush = () => {
        const runs = tidyRuns(inlineRuns(inline, {}, ctx));
        inline = [];
        if (runs.length) out.push(block('paragraph', ctx, { runs }));
    };
    for (const node of childrenOf(container)) {
        if (node.nodeType === TEXT) { inline.push(node); continue; }
        if (node.nodeType !== ELEMENT) continue;
        const tag = tagOf(node);
        if (SKIPPED.has(tag)) continue;
        if (INLINE.has(tag)) { inline.push(node); continue; }
        flush();
        if (/^H[1-6]$/.test(tag)) {
            const runs = tidyRuns(inlineRuns(childrenOf(node), {}, ctx));
            if (runs.length) out.push(block('heading', ctx, { level: Number(tag[1]), runs }));
        } else if (tag === 'P') {
            const runs = tidyRuns(inlineRuns(childrenOf(node), {}, ctx));
            if (runs.length) out.push(block('paragraph', ctx, { runs }));
        } else if (tag === 'UL' || tag === 'OL') listBlocks(node, tag === 'OL', ctx, out);
        else if (tag === 'BLOCKQUOTE') collectBlocks(node, { ...ctx, quote: ctx.quote + 1 }, out);
        else if (tag === 'HR') out.push(block('rule', ctx, {}));
        else if (tag === 'PRE' || hasClass(node, 'md-code-block')) codeBlock(node, ctx, out);
        else if (hasClass(node, 'md-mermaid') || hasClass(node, 'md-chart')) {
            const kind = hasClass(node, 'md-mermaid') ? 'diagram' : 'chart';
            const text = String(node.getAttribute?.('data-mermaid-source') || node.textContent || '').replace(/\n$/, '');
            out.push(block(kind, ctx, { lang: kind === 'diagram' ? 'mermaid' : 'chart', runs: [{ text, code: true }] }));
        } else if (tag === 'TABLE' || hasClass(node, 'md-table-wrap')) tableBlock(node, ctx, out);
        else collectBlocks(node, ctx, out);  // a wrapper holds blocks of its own
    }
    flush();
}

/** The share model of a rendered message `root`: ordered blocks, each `{index, kind, depth, quote, runs, ...}`. */
export function shareBlocks(root) {
    const out = [];
    collectBlocks(root, { depth: 0, quote: 0, links: 0 }, out);
    return out.map((item, index) => {
        const words = item.runs.map((run) => run.text).join('');
        return { ...item, index, math: ['heading', 'paragraph', 'item', 'table'].includes(item.kind) && MATH.test(words),
            rtl: item.kind !== 'code' && textDirection(words) === 'rtl' };
    });
}

/** `rtl` when the first letter of `text` belongs to a right-to-left script. */
export function textDirection(text) {
    const letter = String(text || '').match(/\p{L}/u)?.[0] || '';
    return letter && RTL.test(letter) ? 'rtl' : 'ltr';
}

/** A block's words with each link's whole address after its words (unless the words are the address). */
export function runsText(runs) {
    let text = '';
    for (let index = 0; index < runs.length;) {
        const run = runs[index];
        if (!run.link) { text += run.text; index += 1; continue; }
        let words = '';
        while (index < runs.length && runs[index].link === run.link) { words += runs[index].text; index += 1; }
        text += words;
        const shown = words.trim();
        if (run.href && shown !== run.href && `mailto:${shown}` !== run.href) text += ` (${run.href})`;
    }
    return text;
}

const INDENT = '   ';

function fenced(text, lang) {
    const longest = Math.max(2, ...(String(text).match(/`{3,}/g) || []).map((run) => run.length));
    const fence = '`'.repeat(longest + 1);
    return `${fence}${lang || ''}\n${text}\n${fence}`;
}

function blockText(item) {
    const indent = INDENT.repeat(item.depth || 0);
    const indentLines = (text, first = indent, rest = indent) => text.split('\n')
        .map((line, index) => (line ? `${index ? rest : first}${line}` : '')).join('\n');
    if (item.kind === 'rule') return `${indent}———`;
    if (item.kind === 'code' || item.kind === 'diagram' || item.kind === 'chart') {
        return indentLines(fenced(item.runs.map((run) => run.text).join(''), item.lang));
    }
    if (item.kind === 'table') return indentLines(item.runs.map((run) => run.text).join(''));
    const words = runsText(item.runs);
    if (item.kind !== 'item') return indentLines(words);
    const box = item.task ? (item.task === 'done' ? '[x]' : '[ ]') : '';
    const lead = [item.marker === '•' && box ? '' : item.marker, box].filter(Boolean).join(' ');
    return indentLines(words, `${indent}${lead} `, `${indent}${' '.repeat(lead.length + 1)}`) || `${indent}${lead}`;
}

// Inside a list everything sits close to its item; a new top-level block keeps a blank line before it.
const tight = (previous, next) => (next.kind === 'item' && (previous.kind === 'item' || previous.depth > 0))
    || (next.depth > 0 && next.kind !== 'item');

/** The Text format of `blocks`: exactly the bytes Copy, Share and the Text preview pass on. */
export function shareText(blocks) {
    let text = '';
    let previous = null;
    for (const item of blocks) {
        let body = blockText(item);
        if (item.quote) {
            body = body.split('\n').map((line) => `${'>'.repeat(item.quote)} ${line}`.trimEnd()).join('\n');
        }
        if (previous) text += tight(previous, item) ? '\n' : '\n\n';
        text += body;
        previous = item;
    }
    return text;
}

/** How many words `text` holds, by the engine's word segmentation where it has one. */
export function wordCount(text, locale = undefined) {
    const source = String(text || '');
    const Segmenter = globalThis.Intl?.Segmenter;
    if (typeof Segmenter === 'function') {
        let count = 0;
        for (const part of new Segmenter(locale, { granularity: 'word' }).segment(source)) if (part.isWordLike) count += 1;
        return count;
    }
    return (source.match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) || []).length;
}

/** The words of `blocks` without the addresses Text adds, for counting. */
export const blockWords = (blocks) => blocks.map((item) => item.runs.map((run) => run.text).join('')).join('\n');

const copyBlock = (item) => ({ ...item, runs: item.runs.map((run) => ({ ...run })) });

/** The blocks whose indexes `chosen` holds, in their own order. */
export function pickBlocks(blocks, chosen) {
    return blocks.filter((item) => chosen.has(item.index)).map(copyBlock);
}

/**
 * Exactly the words `pieces` name, `[{block, run, start, end}]` in document order: each named block keeps its kind
 * and marks, with only the named part of each named run. Code keeps every character it was given.
 */
export function sliceBlocks(blocks, pieces) {
    const byBlock = new Map();
    for (const piece of pieces) {
        const item = blocks[piece.block];
        const run = item?.runs[piece.run];
        if (!run) continue;
        const text = run.text.slice(Math.max(0, piece.start), Math.max(0, piece.end));
        if (!text) continue;
        if (!byBlock.has(item.index)) byBlock.set(item.index, { ...item, runs: [] });
        byBlock.get(item.index).runs.push({ ...run, text });
    }
    return [...byBlock.values()].map((item) => {
        const runs = item.kind === 'code' || item.kind === 'diagram' || item.kind === 'chart' || item.kind === 'table'
            ? item.runs : tidyRuns(item.runs);
        return { ...item, runs };
    }).filter((item) => item.runs.length);
}

/** What a card cannot draw in `blocks`: 'table', 'math', 'diagram' or 'chart', each once, in this order. */
export function cardUnsupported(blocks) {
    const found = new Set();
    for (const item of blocks) {
        if (item.kind === 'table') found.add('table');
        else if (item.kind === 'diagram') found.add('diagram');
        else if (item.kind === 'chart') found.add('chart');
        if (item.math) found.add('math');
    }
    return ['table', 'math', 'diagram', 'chart'].filter((kind) => found.has(kind));
}
