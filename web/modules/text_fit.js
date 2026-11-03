// Fit a text into N lines of its element (docs/DESIGN.md "Replies"; the portal card's latest line). A CSS
// line clamp draws its ellipsis wherever the box ends, so a clamp that lands right after a full stop reads
// `редизайна.…` (four dots). This cuts on a word, drops the punctuation the cut leaves at the end, and only
// then adds the ellipsis. The full text stays on record (`fullText`) for copy, expansion and re-fitting; an
// element with no layout (a hidden page, a test DOM) simply shows the whole text and is fitted once it has
// a width. One ResizeObserver serves every fitted element; a width change re-fits on the next frame (a fit changes the
// element's own size, so a fit inside the observer's callback could loop in one frame: WebKit reports that as an
// error), a height change does not, and an element removed from the page leaves the observer (no element is kept
// alive by it). That frame's re-fit is heard (`onRefit`): a room places its reader again once its lines changed size
// (Main shown again had laid out its newest message before its lines were fitted at their width). Elements fitted in one
// batch (layout_batch.js: that frame's re-fit, the reading owner's write) are fitted in lockstep: every element's probe of
// a round is written before any is read, so a batch costs a few layouts of the page however many lines it holds (fitted
// one by one, every probe laid out the whole page: a long Main whose hundred work lines changed behind a page froze for
// seconds on return, and a page of history re-fitted each card once per row it replayed).
import { batch, measure } from './layout_batch.js';

const fits = new WeakMap();  // element -> { text, lines, width, onFit }
let observer = null;
const widened = new Set();  // elements whose width changed, fitted on the next frame
let frame = 0;
const refitted = new Set();  // who hears a frame's re-fit

/** `listener(elements)` hears the elements each frame re-fitted after their width changed; returns its disposer. */
export function onRefit(listener) {
    refitted.add(listener);
    return () => refitted.delete(listener);
}

const TAIL = /[\s.,;:!?…—–-]+$/u;

/** The words before a cut, without the punctuation or space the cut leaves at their end, plus `…`. */
export function ellipsize(prefix) {
    return `${String(prefix ?? '').replace(TAIL, '')}…`;
}

function lineHeight(style) {
    const value = parseFloat(style.lineHeight);
    return Number.isFinite(value) ? value : parseFloat(style.fontSize) * 1.35;
}

// A one-line (nowrap) element overflows sideways. Its text is measured with sub-pixel precision: a rounded
// scrollWidth can hide a fraction of a pixel that still makes the browser draw its own mid-word ellipsis.
function fitsInline(el, nowrap) {
    if (!nowrap) return true;
    if (typeof document === 'undefined' || typeof document.createRange !== 'function') return el.scrollWidth <= el.clientWidth;
    const range = document.createRange();
    range.selectNodeContents(el);
    return range.getBoundingClientRect().width <= el.getBoundingClientRect().width;
}

// A wrapping element overflows downwards, a one-line (nowrap) element sideways.
const inside = (job) => job.el.scrollHeight <= job.limit && fitsInline(job.el, job.nowrap);

// One element's fit, written as the reads it waits for: each `yield read` is measured with every other pending read
// (layout_batch.js), then the fit resumes with its value — its width, line height and whether the whole text fits,
// then a binary search on its words. The whole text is written at once, so whoever reads the element in the same write
// sees it (a card's change projection, chat_render_batch.js); only a cut waits for its reads. An element with no layout
// shows its whole text. In a batch an element built before it is mounted (a reply's quote) is fitted once the batch's
// writes have mounted it.
function* fitSteps(el, fit) {
    if (el.textContent !== fit.text) el.textContent = fit.text;
    const box = yield () => {
        if (!el.isConnected || !el.clientWidth) return null;
        const style = getComputedStyle(el);
        const job = { el, width: el.clientWidth, limit: fit.lines * lineHeight(style) + 1, nowrap: style.whiteSpace === 'nowrap' };
        job.whole = inside(job);
        return job;
    };
    fit.width = box ? box.width : 0;  // an element with no layout forgets its width, so showing it again fits it again
    if (!box || box.whole) { fit.onFit?.(false); return; }
    // Cut on whitespace only: a word is never split, a line never ends half a word.
    const tokens = fit.text.split(/(\s+)/);
    let low = 0, high = tokens.length;
    while (low < high) {
        const mid = Math.ceil((low + high) / 2);
        el.textContent = ellipsize(tokens.slice(0, mid).join(''));
        if (yield () => inside(box)) low = mid; else high = mid - 1;
    }
    el.textContent = ellipsize(tokens.slice(0, Math.max(low, 1)).join(''));
    fit.onFit?.(true);
}

function refit(el) {
    const fit = fits.get(el);
    if (!fit) return;
    const steps = fitSteps(el, fit);
    const step = (value) => {
        if (fits.get(el) !== fit) return;  // a newer fit of the element ends this one
        const { value: read, done } = steps.next(value);
        if (!done) measure(read, step);
    };
    step();
}

/**
 * Show `text` in at most `lines` lines of `el` (0 or Infinity = the whole text). `onFit(cut)` hears after every
 * fit whether text is hidden (a `Show all` control follows it). Returns the element.
 */
export function fitText(el, text, lines, onFit = null) {
    if (!el) return el;
    const value = String(text ?? '');
    if (!lines || lines === Infinity) {
        fits.delete(el);
        observer?.unobserve?.(el);
        el.textContent = value;
        return el;
    }
    fits.set(el, { text: value, lines, width: 0, onFit });
    if (!observer && typeof ResizeObserver === 'function') {
        observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                // A removed element reports its last size change here: it leaves the shared observer with it.
                if (!entry.target.isConnected) { observer.unobserve(entry.target); fits.delete(entry.target); continue; }
                const fit = fits.get(entry.target);
                if (fit && entry.target.clientWidth !== fit.width) widened.add(entry.target);
            }
            if (!widened.size || frame) return;
            const later = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (fn) => setTimeout(fn, 0);
            frame = later(() => {
                frame = 0;
                const due = [...widened];
                widened.clear();
                batch(() => due.forEach(refit));
                for (const listener of refitted) listener(due);
            });
        });
    }
    observer?.observe?.(el);
    refit(el);
    return el;
}

/** The text a fitted element stands for (its whole text), or its own text when it is not fitted. */
export function fullText(el) {
    return fits.get(el)?.text ?? el?.textContent ?? '';
}

/** True when the element currently shows less than its whole text. */
export function isCut(el) {
    const fit = fits.get(el);
    return Boolean(fit && el.textContent !== fit.text);
}
