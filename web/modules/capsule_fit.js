// Where an open composer capsule's strip stands (docs/DESIGN.md "Composer owner controls"): on the composer's line
// beside the other controls when it fits there, alone (they step aside, its own name too if need be) when it does not,
// scrolling inside itself only when nothing else fits. The fit is measured, never a breakpoint (a translation or another
// font needs none). The capsule machine (composer_owner_controls.js) opens and closes the capsule; this measures and
// writes the fit, fades a scrolling strip's hidden side, keeps what the owner moved in view, and holds the capsule's
// widths while its value gives way to the growing strip.

// The segments' side padding (`--capsule-seg-pad`, px): up to var(--space-2) when there is room
// (Context's three words take a little more air), never below var(--space-1) beside the other
// controls, down to 3 px when the strip stands alone (from 4 px down the pill's corners tighten
// with the padding, so the word's box stays inside it: `data-tight`). Below that the strip
// scrolls inside itself.
export const SEG_PAD = Object.freeze({ max: 8, beside: 4, alone: 3 });
// The fade at a scrolling strip's edge (var(--space-3)): what the owner moved stays clear of it.
const REVEAL_MARGIN_PX = 12;
export const px = (value) => Number.parseFloat(value) || 0;
// Fractional widths: whole-pixel offsets lose up to a pixel per element, enough to wrap the row.
export const widthOf = (node) => {
    const width = node?.getBoundingClientRect?.().width;
    return Number.isFinite(width) && width > 0 ? width : Number(node?.offsetWidth) || 0;
};

/** The fit of one capsule `el` (its `strip` of `segs`) in its composer `row`; `isOpen()` says whether it stands open. */
export function createCapsuleFit({ el, row, kind, strip, segs, padMax = SEG_PAD.max, isOpen, win, doc }) {
    const q = (selector) => el.querySelector(selector);
    const head = q('.chat-capsule-head');
    const glyph = q('.chat-capsule-glyph');
    const name = q('.chat-capsule-name');
    const value = q('.chat-capsule-value');
    const body = q('.chat-capsule-body');
    const styleOf = (node, pseudo) => { try { return win?.getComputedStyle?.(node, pseudo) || null; } catch { return null; } };
    // Right to left (the rendered direction) the strip starts at its right end: its levels run
    // leftwards to negative offsets and its scrollLeft runs from 0 down.
    const isRtl = () => styleOf(strip)?.direction === 'rtl';
    // What stays on the line beside the open strip: the other capsules, closed, and the row's
    // other controls, each with its row gap. Alone, they are hidden and measure 0: the width seen
    // before they stepped aside stays.
    let beside = { width: 0, count: 0 };
    function measurePeers(gap) {
        if (row.dataset?.ownerSolo) return beside;
        let width = 0;
        let count = 0;
        for (const node of Array.from(row.children || [])) {
            if (node === el || node.hidden) continue;
            const own = widthOf(node);
            if (own > 0) { width += own + gap; count += 1; }
        }
        beside = { width, count };
        return beside;
    }
    // The head as it stands open: the ring, then the name and the divider before the levels
    // unless the strip needs their room (the name is cached: a name that gave way measures 0).
    let nameWidth = 0;
    const measureName = () => { const shown = widthOf(name); if (shown > 0) nameWidth = shown; };
    function headWidths() {
        measureName();
        const headStyle = styleOf(head);
        const nameStyle = styleOf(name);
        const divider = styleOf(body, '::before');
        const bare = px(headStyle?.paddingLeft) + widthOf(glyph) + px(headStyle?.paddingRight);
        return {
            bare,
            label: bare + px(nameStyle?.marginLeft) + nameWidth + px(nameStyle?.marginRight)
                + px(divider?.marginLeft) + px(divider?.width) + px(divider?.marginRight),
        };
    }
    // Every input is independent of the padding it decides (words, head, borders, the row), so
    // a resize that follows a new padding finds the same answer.
    function measure() {
        const rowStyle = styleOf(row);
        const room = widthOf(row) - px(rowStyle?.paddingLeft) - px(rowStyle?.paddingRight) - 1;  // 1 px of slack
        if (!(room > 0)) return null;  // not laid out: a hidden pane, the unit-test DOM
        const words = segs.map((seg) => widthOf(seg.firstElementChild));
        // A segment is never narrower than its own minimum (Subagents' options keep a finger's width), so
        // the strip is each word with its padding or that minimum, whichever is wider.
        const floors = segs.map((seg) => px(styleOf(seg)?.minWidth));
        const stripAt = (pad) => words.reduce((sum, word, i) => sum + Math.max(word + 2 * pad, floors[i]), 0);
        const elStyle = styleOf(el);
        const stripStyle = styleOf(strip);
        // Besides the head and the words: the borders and the strip's end gutter (its optical
        // clearance from the rounded end).
        const frame = px(elStyle?.borderLeftWidth) + px(elStyle?.borderRightWidth)
            + px(stripStyle?.marginLeft) + px(stripStyle?.marginRight);
        const heads = headWidths();
        // The widest padding, from padMax down to `least`, at which the head and the strip fit `width`.
        const padIn = (width, headWidth, least) => {
            for (let pad = padMax; pad >= least; pad -= 1) if (headWidth + frame + stripAt(pad) <= width) return pad;
            return null;
        };
        const fitted = (mode, pad, label) => ({ mode, pad, label });
        const beside = measurePeers(px(rowStyle?.columnGap));
        let pad = padIn(room - beside.width, heads.label, beside.count ? SEG_PAD.beside : SEG_PAD.alone);
        if (pad !== null) return fitted('inline', pad, true);
        const alone = beside.count ? 'solo' : 'inline';
        pad = padIn(room, heads.label, SEG_PAD.alone);
        if (pad !== null) return fitted(alone, pad, true);
        pad = padIn(room, heads.bare, SEG_PAD.alone);
        if (pad !== null) return fitted(alone, pad, false);
        return fitted('overflow', SEG_PAD.alone, false);
    }
    function fit() {
        const fitted = isOpen() ? measure() : null;
        if (!fitted) return;
        el.style.setProperty('--capsule-seg-pad', `${fitted.pad}px`);
        el.dataset.fit = fitted.mode;
        el.dataset.tight = String(fitted.pad <= SEG_PAD.beside);
        el.dataset.label = String(fitted.label);
        if (fitted.mode !== 'inline') row.dataset.ownerSolo = kind;
        else if (row.dataset.ownerSolo === kind) delete row.dataset.ownerSolo;
        if (held) holdBody();
    }
    // A strip that scrolls fades on the side that hides levels (`start`/`end`: how far along its
    // own direction it is scrolled).
    function onStripScroll() {
        // The levels' own end, not scrollWidth: a bracket's hit area may reach past the last level.
        const last = segs[segs.length - 1];
        const rtl = isRtl();
        const width = Number(strip.clientWidth) || 0;
        const end = rtl ? width - Number(last?.offsetLeft) : Number(last?.offsetLeft) + Number(last?.offsetWidth);
        const scrollable = (Number.isFinite(end) && end > 0 ? end : Number(strip.scrollWidth) || 0) - width;
        const at = (rtl ? -1 : 1) * (Number(strip.scrollLeft) || 0);
        strip.dataset.scroll = scrollable <= 1 ? 'none' : at <= 1 ? 'start' : at >= scrollable - 1 ? 'end' : 'middle';
    }
    // ...and keeps what the owner moved (a handle, a mode; `left..right` in the strip) in view. The
    // view is `scrollLeft..scrollLeft + clientWidth` in offsetLeft's terms either way; it never
    // scrolls back past its start (0).
    function reveal(left, right) {
        if (el.dataset.fit !== 'overflow') return;
        const width = Number(strip.clientWidth) || 0;
        let to = null;
        if (left - REVEAL_MARGIN_PX < strip.scrollLeft) to = left - REVEAL_MARGIN_PX;
        else if (right + REVEAL_MARGIN_PX > strip.scrollLeft + width) to = right + REVEAL_MARGIN_PX - width;
        if (to !== null) strip.scrollLeft = isRtl() ? Math.min(0, to) : Math.max(0, to);
        onStripScroll();
    }
    // The value gives way while the strip grows, on its curve and from both real widths (style.css):
    // the capsule's width moves straight between its closed and open widths, so a line that holds
    // both never wraps in between. The widths are held while that runs, then follow the words again.
    let held = false;
    const heldWidth = (width) => `${Math.ceil(width * 64 + 1) / 64}px`;  // a hair over: layout rounds to 1/64 px
    function valueWidth() {
        try {  // the text keeps its whole width while the value gives way
            const range = doc.createRange();
            range.selectNodeContents(value);
            const width = range.getBoundingClientRect().width;
            if (width > 0) return width;
        } catch { /* no layout */ }
        return Number(value.scrollWidth) || 0;
    }
    function holdBody() {
        const stripStyle = styleOf(strip);
        const divider = el.dataset.label === 'false' ? null : styleOf(body, '::before');
        el.style.setProperty('--capsule-body-width', heldWidth(widthOf(strip) + px(stripStyle?.marginLeft) + px(stripStyle?.marginRight)
            + px(divider?.marginLeft) + px(divider?.width) + px(divider?.marginRight)));
    }
    function holdWidths() {
        if (!(widthOf(row) > 0)) return;  // not laid out: nothing moves
        held = true;
        el.style.setProperty('--capsule-value-width', heldWidth(valueWidth()));
        holdBody();
        void el.offsetWidth;  // the change starts from these widths
    }
    function releaseWidths() {
        if (!held) return;
        held = false;
        el.style.setProperty('--capsule-value-width', '');
        el.style.setProperty('--capsule-body-width', '');
    }
    // Alone, the strip appears and disappears at once (no width animation), so the other
    // controls never come back beside a strip that is still collapsing: the instant close is
    // flushed before the solo flag goes. The padding stays as fitted: beside them the strip is
    // still collapsing, and a wider padding now would push it onto a second line for those
    // frames (the next open fits again).
    function settleClosed() {
        if (el.dataset.fit && el.dataset.fit !== 'inline') void body.offsetWidth;
        delete el.dataset.fit;
        if (row.dataset?.ownerSolo === kind) delete row.dataset.ownerSolo;
    }
    return { isRtl, measureName, measure, fit, onStripScroll, reveal, holdWidths, releaseWidths, settleClosed };
}
