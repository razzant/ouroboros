/**
 * The one hint (docs/DESIGN.md "Hints"): what a control says about itself. It stands above its control and over the
 * page on the menus' floating material (ui.css), so showing it, changing its words or hiding it moves nothing. It
 * lives inside its control: the control's own press, hover and focus handling covers it, a reader hears it as the
 * control's description, and Tab reaches its one action right after the control.
 *
 * `say({ word, text, rule, action })` writes only the words that changed since it last said them, so a translation the
 * overlay put in their place stays; a row that watches its words for a refit leaves the hint's alone
 * (composer_owner_controls.js). `onAction()` runs the action.
 */
let made = 0;
const EDGE_PX = 8;
const GAP_PX = 8;  // ui.css: the hint stands var(--space-2) from its control

export function createHint(host, { doc = host.ownerDocument, onAction = () => {} } = {}) {
    const el = doc.createElement('div');
    el.className = 'ui-hint';
    el.id = `ui-hint-${++made}`;
    el.hidden = true;
    el.innerHTML = '<p class="ui-hint-line"><span class="ui-hint-word"></span><span class="ui-hint-text"></span></p>'
        + '<p class="ui-hint-rule"></p><button type="button" class="ui-hint-action ui-hit-area"></button>';
    const parts = {
        word: el.querySelector('.ui-hint-word'), text: el.querySelector('.ui-hint-text'),
        rule: el.querySelector('.ui-hint-rule'), action: el.querySelector('.ui-hint-action'),
    };
    const said = { word: null, text: null, rule: null, action: null };
    const write = (part, words = '') => {
        if (said[part] === words) return;
        said[part] = words;
        parts[part].textContent = words;
        parts[part].hidden = !words;
    };
    const onClick = () => onAction();
    parts.action.addEventListener('click', onClick);
    host.append(el);

    // Above its control while it fits there, else on the side with more room, and never past the window: where its
    // words need more than that room it scrolls inside itself. It starts where its control starts and moves back where
    // it would pass the far edge, never past the near one (right to left, the start is the right edge).
    const win = doc.defaultView;
    function place() {
        const width = Number(doc.documentElement?.clientWidth);
        if (!Number.isFinite(width) || width <= 0 || el.hidden) return;
        const view = win?.visualViewport;
        const top = Number(view?.offsetTop) || 0;
        const bottom = top + (Number(view?.height) || Number(doc.documentElement.clientHeight) || 0);
        const anchor = host.getBoundingClientRect();
        const natural = Number(el.scrollHeight) + 2;  // its words' height, whatever room it was given; plus the border
        const above = anchor.top - GAP_PX - EDGE_PX - top;
        const below = bottom - EDGE_PX - GAP_PX - anchor.bottom;
        const side = natural > above && below > above ? 'below' : 'above';
        if (el.dataset.side !== side) el.dataset.side = side;
        const room = Math.max(0, Math.floor(side === 'below' ? below : above));
        const cramped = String(natural > room);
        if (el.dataset.cramped !== cramped) el.dataset.cramped = cramped;
        el.style.setProperty('--ui-hint-room', cramped === 'true' ? `${room}px` : 'none');
        el.style.setProperty('--ui-hint-shift', '0px');
        const box = el.getBoundingClientRect();
        let dx = 0;
        if (box.right > width - EDGE_PX) dx = width - EDGE_PX - box.right;
        if (box.left + dx < EDGE_PX) dx = EDGE_PX - box.left;
        const rtl = win?.getComputedStyle?.(el)?.direction === 'rtl';
        if (dx) el.style.setProperty('--ui-hint-shift', `${Math.round(rtl ? -dx : dx)}px`);
    }
    // Its words can grow after they are written (the translation overlay rewrites every write in place), and the window
    // can change under it. A size it reports is placed on the next frame, outside the observer's delivery, so placing
    // never feeds the observer; so is every write, since a write and its translation can leave the size the observer
    // last saw while the place was measured in between.
    let frame = 0;
    const later = () => {
        if (!frame && typeof win?.requestAnimationFrame === 'function') frame = win.requestAnimationFrame(() => { frame = 0; place(); });
    };
    const resized = typeof win?.ResizeObserver === 'function' ? new win.ResizeObserver(later) : null;
    resized?.observe(el);
    win?.addEventListener?.('resize', place);
    win?.visualViewport?.addEventListener?.('resize', place);

    return {
        el,
        say({ word = '', text = '', rule = '', action = '' } = {}) {
            write('word', word);
            write('text', text);
            write('rule', rule);
            write('action', action);
            if (!el.hidden) { place(); later(); }
        },
        show() {
            if (!el.hidden) return;
            el.hidden = false;
            place();
            later();
        },
        hide() { el.hidden = true; },
        destroy() {
            parts.action.removeEventListener('click', onClick);
            resized?.disconnect();
            if (frame) win.cancelAnimationFrame(frame);
            win?.removeEventListener?.('resize', place);
            win?.visualViewport?.removeEventListener?.('resize', place);
            el.remove();
        },
    };
}
