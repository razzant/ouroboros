// The Card format of Share (docs/DESIGN.md "Replies"): the chosen blocks drawn on a canvas in the room's own colours
// and type, one PNG that the preview shows byte for byte. Nothing is fetched from outside the page (no fonts, images
// or addresses; the mark is the app's own icon) and no word is clipped or shrunk to fit: a table, a formula, a
// diagram, a chart or a text taller than one readable card is refused with its reason, and Text carries it whole.
// A link shows its words and its site; its whole address is Text's. A right-to-left paragraph is drawn as plain
// lines in its own direction, without bold or italic inside it.

import { tr } from './i18n.js';
import { cardUnsupported } from './message_share_model.js';

// A phone shows a picture across its width: at this width the card's type reads there near the size it is drawn, and
// at twice its width it still fits one screen. The PNG has three pixels for each, sharp on any screen.
export const CARD_WIDTH = 440;        // CSS px
export const CARD_MAX_HEIGHT = 880;
const SCALE = 3;
const PAD_X = 30;
const PAD_Y = 30;
const RADIUS = 22;
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace';
const LOGO = '/static/favicon.png';
const BODY = { size: 18, line: 28 };
const HEADINGS = { 1: { size: 22, line: 30 }, 2: { size: 22, line: 30 }, other: { size: 19, line: 27 } };
const CODE = { size: 14, line: 21, pad: 12 };
const BRAND = { size: 14, mark: 20, gap: 22 };
const BLOCK_GAP = 14;
const ITEM_GAP = 6;
const LEVEL = 24;
const QUOTE = 18;

/** The room's colours and type now, read from its tokens (docs/DESIGN.md: the card follows the theme it is made in). */
export function cardTheme(doc = globalThis.document, win = globalThis.window) {
    const root = win?.getComputedStyle?.(doc.documentElement);
    const token = (name, fallback) => String(root?.getPropertyValue?.(name) || '').trim() || fallback;
    return {
        background: token('--bg-primary', '#0f0f11'),
        tint: token('--accent-08', 'rgba(201, 53, 69, 0.08)'),
        text: token('--text-primary', '#efeff2'),
        meta: token('--text-meta', 'rgba(255, 255, 255, 0.7)'),
        link: token('--accent-light', '#fe7d85'),
        brand: token('--assistant-sender', '#feaaae'),
        rule: token('--field-border', 'rgba(255, 255, 255, 0.08)'),
        wash: token('--surface-capsule', 'rgba(255, 255, 255, 0.09)'),
        sans: String(win?.getComputedStyle?.(doc.body)?.fontFamily || '').trim() || 'system-ui, sans-serif',
    };
}

const fontOf = (theme, size, { weight = 400, italic = false, mono = false } = {}) => (
    `${italic ? 'italic ' : ''}${weight} ${size}px ${mono ? MONO : theme.sans}`);

function graphemes(text) {
    const Segmenter = globalThis.Intl?.Segmenter;
    if (typeof Segmenter === 'function') return Array.from(new Segmenter(undefined, { granularity: 'grapheme' }).segment(text), (part) => part.segment);
    return Array.from(text);
}

function siteOf(href) {
    try {
        const url = new URL(href);
        return /^https?:$/.test(url.protocol) ? url.hostname.replace(/^www\./, '') : '';
    } catch {
        return '';
    }
}

/** A block's runs as the card draws them: after a link's words, its site in the quiet ink when the words lack it. */
function cardRuns(runs) {
    const out = [];
    for (let index = 0; index < runs.length;) {
        const run = runs[index];
        if (!run.link) { out.push(run); index += 1; continue; }
        let words = '';
        while (index < runs.length && runs[index].link === run.link) { out.push(runs[index]); words += runs[index].text; index += 1; }
        const site = run.href ? siteOf(run.href) : '';
        if (site && !words.toLowerCase().includes(site.toLowerCase())) out.push({ text: ` (${site})`, meta: true });
    }
    return out;
}

/** Greedy lines of `runs` within `width`: words keep together, a word wider than the card breaks between graphemes. */
function wrapRuns(runs, width, styleOf, measure) {
    const lines = [];
    let line = [];
    let x = 0;
    let space = null;
    const newLine = () => { lines.push(line); line = []; x = 0; space = null; };
    const place = (text, style, size) => { line.push({ text, style, x, width: size }); x += size; };
    for (const run of runs) {
        const style = styleOf(run);
        for (const part of String(run.text).split(/(\n| +)/)) {
            if (!part) continue;
            if (part === '\n') { newLine(); continue; }
            if (!part.trim()) { if (line.length) space = { style, width: measure(' ', style.font) }; continue; }
            const size = measure(part, style.font);
            if (line.length && x + (space?.width || 0) + size > width) newLine();
            else if (space && line.length) place(' ', space.style, space.width);
            space = null;
            if (size <= width) { place(part, style, size); continue; }
            let chunk = '';
            for (const grapheme of graphemes(part)) {
                if (chunk && x + measure(chunk + grapheme, style.font) > width) {
                    place(chunk, style, measure(chunk, style.font));
                    newLine();
                    chunk = grapheme;
                } else chunk += grapheme;
            }
            if (chunk) place(chunk, style, measure(chunk, style.font));
        }
    }
    if (line.length || !lines.length) lines.push(line);
    return lines;
}

/** The card's drawing: lines of placed pieces and its shapes, measured with `measure(text, font)`. */
export function layoutCard(blocks, { theme, measure, excerpt = false, logo = false }) {
    const inner = CARD_WIDTH - 2 * PAD_X;
    const lines = [];
    const shapes = [];
    let y = PAD_Y;
    let previous = null;
    for (const item of blocks) {
        if (previous) y += item.kind === 'item' && previous.kind === 'item' ? ITEM_GAP : BLOCK_GAP;
        previous = item;
        const left = PAD_X + (item.depth || 0) * LEVEL + (item.quote || 0) * QUOTE;
        const top = y;
        if (item.kind === 'rule') {
            shapes.push({ type: 'rect', x: left, y: y + 6, w: PAD_X + inner - left, h: 1, color: theme.rule });
            y += 13;
        } else if (item.kind === 'code') {
            const font = fontOf(theme, CODE.size, { mono: true });
            const width = PAD_X + inner - left - 2 * CODE.pad;
            const codeLines = [];
            for (const source of item.runs.map((run) => run.text).join('').replace(/\t/g, '    ').split('\n')) {
                for (const wrapped of wrapRuns([{ text: source.replace(/ /g, '\u00a0') }], width,
                    () => ({ font, color: theme.text }), measure)) codeLines.push(wrapped);
            }
            const height = codeLines.length * CODE.line + 2 * CODE.pad;
            shapes.push({ type: 'rect', x: left, y, w: PAD_X + inner - left, h: height, color: theme.wash, radius: 10 });
            codeLines.forEach((pieces, index) => lines.push({ y: y + CODE.pad + index * CODE.line, height: CODE.line,
                pieces: pieces.map((piece) => ({ ...piece, x: piece.x + left + CODE.pad })) }));
            y += height;
        } else {
            const metric = item.kind === 'heading' ? (HEADINGS[item.level] || HEADINGS.other) : BODY;
            const weight = item.kind === 'heading' ? 700 : 400;
            const plain = fontOf(theme, metric.size, { weight });
            const box = item.task === 'done' ? '☑' : item.task === 'open' ? '☐' : '';
            const marker = item.kind === 'item'
                ? [item.marker === '•' && box ? '' : item.marker, box].filter(Boolean).join(' ') : '';
            const markerWidth = marker ? Math.max(26, measure(marker, plain) + 10) : 0;
            if (item.rtl) {
                // One base direction per paragraph: its lines in plain type, drawn from the right edge.
                const right = PAD_X + inner - (item.depth || 0) * LEVEL - (item.quote || 0) * QUOTE;
                if (marker) lines.push({ y, height: metric.line, rtl: true, right, pieces: [{ text: marker, style: { font: plain, color: theme.meta } }] });
                const words = cardRuns(item.runs).map((run) => run.text).join('');
                const wrapped = wrapRuns([{ text: words }], right - markerWidth - PAD_X, () => ({ font: plain, color: theme.text }), measure);
                wrapped.forEach((pieces, index) => lines.push({ y: y + index * metric.line, height: metric.line, rtl: true,
                    right: right - markerWidth,
                    pieces: [{ text: pieces.map((piece) => piece.text).join(''), style: { font: plain, color: theme.text } }] }));
                y += wrapped.length * metric.line;
            } else {
                if (marker) lines.push({ y, height: metric.line, pieces: [{ text: marker, x: left, width: markerWidth, style: { font: plain, color: theme.meta } }] });
                const textLeft = left + markerWidth;
                const styleOf = (run) => ({
                    font: fontOf(theme, run.code ? metric.size - 2 : metric.size,
                        { weight: run.bold || item.kind === 'heading' ? Math.max(weight, 600) : 400, italic: run.italic, mono: run.code }),
                    color: run.meta ? theme.meta : run.link && run.href ? theme.link : theme.text,
                    underline: Boolean(run.link && run.href), strike: Boolean(run.strike), wash: Boolean(run.code),
                });
                const wrapped = wrapRuns(cardRuns(item.runs), PAD_X + inner - textLeft, styleOf, measure);
                wrapped.forEach((pieces, index) => lines.push({ y: y + index * metric.line, height: metric.line,
                    pieces: pieces.map((piece) => ({ ...piece, x: piece.x + textLeft })) }));
                y += wrapped.length * metric.line;
            }
        }
        if (item.quote) {
            const x = item.rtl ? PAD_X + inner - (item.depth || 0) * LEVEL - 7 : left - QUOTE + 4;
            shapes.push({ type: 'rect', x, y: top, w: 3, h: y - top, color: theme.rule, radius: 1.5 });
        }
    }
    y += BRAND.gap;
    const brandFont = fontOf(theme, BRAND.size, { weight: 600 });
    const quietFont = fontOf(theme, BRAND.size);
    const brandLeft = PAD_X + (logo ? BRAND.mark + 8 : 0);
    const name = tr('share.card_brand', 'Ouroboros');
    const nameWidth = measure(name, brandFont);
    const brand = { y, height: BRAND.mark, mark: logo ? { x: PAD_X, y, size: BRAND.mark } : null, pieces: [
        { text: name, x: brandLeft, width: nameWidth, style: { font: brandFont, color: theme.brand } },
    ] };
    if (excerpt) {
        const label = ` · ${tr('share.card_excerpt', 'excerpt')}`;
        brand.pieces.push({ text: label, x: brandLeft + nameWidth, width: measure(label, quietFont), style: { font: quietFont, color: theme.meta } });
    }
    lines.push(brand);
    // A measured layout must fit horizontally too: nested indents, list markers, RTL lines and even one
    // oversized glyph can exceed the card. Refuse the whole picture rather than clipping any of them.
    const rightEdge = CARD_WIDTH - PAD_X;
    const outside = (x, width) => !Number.isFinite(x) || !Number.isFinite(width)
        || width < 0 || x < PAD_X || x + width > rightEdge + 0.5;
    const tooWide = shapes.some((shape) => outside(shape.x, shape.w))
        || lines.some((line) => line.pieces.some((piece) => {
            const width = piece.width ?? measure(piece.text, piece.style.font);
            return outside(line.rtl ? line.right - width : piece.x, width);
        }));
    return { lines, shapes, brand, height: Math.ceil(y + BRAND.mark + PAD_Y), tooWide };
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
}

function paint(ctx, plan, theme, logo) {
    roundRect(ctx, 0, 0, CARD_WIDTH, plan.height, RADIUS);
    ctx.fillStyle = theme.background;
    ctx.fill();
    ctx.fillStyle = theme.tint;
    ctx.fill();
    for (const shape of plan.shapes) {
        ctx.fillStyle = shape.color;
        roundRect(ctx, shape.x, shape.y, shape.w, shape.h, shape.radius || 0);
        ctx.fill();
    }
    ctx.textBaseline = 'middle';
    for (const line of plan.lines) {
        const middle = line.y + line.height / 2;
        if (line.mark && logo) {
            ctx.save();
            roundRect(ctx, line.mark.x, line.mark.y, line.mark.size, line.mark.size, 6);
            ctx.clip();
            ctx.drawImage(logo, line.mark.x, line.mark.y, line.mark.size, line.mark.size);
            ctx.restore();
        }
        for (const piece of line.pieces) {
            const { style } = piece;
            ctx.font = style.font;
            if (line.rtl) {
                ctx.direction = 'rtl';
                ctx.textAlign = 'right';
                ctx.fillStyle = style.color;
                ctx.fillText(piece.text, line.right, middle);
                ctx.direction = 'ltr';
                ctx.textAlign = 'left';
                continue;
            }
            if (style.wash && piece.text.trim()) {
                ctx.fillStyle = theme.wash;
                roundRect(ctx, piece.x - 3, line.y + 3, piece.width + 6, line.height - 6, 5);
                ctx.fill();
            }
            ctx.fillStyle = style.color;
            ctx.fillText(piece.text, piece.x, middle);
            const size = Number.parseFloat(String(style.font).match(/(\d+(?:\.\d+)?)px/)?.[1] || '16');
            if (style.underline && piece.text.trim()) ctx.fillRect(piece.x, middle + size * 0.55, piece.width, 1.25);
            if (style.strike && piece.text.trim()) ctx.fillRect(piece.x, middle, piece.width, 1.25);
        }
    }
}

function loadLogo(doc, src) {
    if (!src || typeof doc?.createElement !== 'function') return Promise.resolve(null);
    const image = doc.createElement('img');
    image.decoding = 'async';
    image.src = src;
    const decoded = typeof image.decode === 'function' ? image.decode() : Promise.reject(new Error('no decode'));
    const timeout = new Promise((resolve) => { setTimeout(() => resolve(null), 3000); });
    return Promise.race([decoded.then(() => image, () => null), timeout]);
}

/**
 * The card of `blocks` as `{blob, width, height}` (CSS pixels; the PNG has three times as many), or `{refused, kinds}`:
 * `unsupported` with what it cannot draw, `too_long`/`too_wide` beyond one readable card, `unavailable` without canvas.
 */
export async function renderShareCard(blocks, { excerpt = false, doc = globalThis.document, win = globalThis.window,
    logo = LOGO } = {}) {
    const kinds = cardUnsupported(blocks);
    if (kinds.length) return { refused: 'unsupported', kinds };
    const canvas = doc?.createElement?.('canvas');
    const ctx = canvas?.getContext?.('2d');
    if (!ctx || typeof canvas.toBlob !== 'function') return { refused: 'unavailable', kinds: [] };
    try { await doc.fonts?.ready; } catch { /* the system type still draws */ }
    const theme = cardTheme(doc, win);
    const mark = await loadLogo(doc, logo);
    const measure = (text, font) => { ctx.font = font; return ctx.measureText(text).width; };
    const plan = layoutCard(blocks, { theme, measure, excerpt, logo: Boolean(mark) });
    if (plan.tooWide) return { refused: 'too_wide', kinds: [] };
    if (plan.height > CARD_MAX_HEIGHT) return { refused: 'too_long', kinds: [] };
    canvas.width = Math.ceil(CARD_WIDTH * SCALE);
    canvas.height = Math.ceil(plan.height * SCALE);
    ctx.scale(SCALE, SCALE);
    paint(ctx, plan, theme, mark);
    const blob = await new Promise((resolve) => { canvas.toBlob(resolve, 'image/png'); });
    if (!blob) return { refused: 'unavailable', kinds: [] };
    return { blob, width: CARD_WIDTH, height: plan.height };
}
