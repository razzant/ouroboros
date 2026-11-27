/**
 * The one hint (docs/DESIGN.md "Hints"): what a control says about itself. It stands above its control and over the
 * page on the menus' floating material (ui.css), so showing it, changing its words or hiding it moves nothing. It
 * lives inside its control: the control's own press, hover and focus handling covers it, a reader hears it as the
 * control's description, and Tab reaches its one action right after the control.
 *
 * `say({ word, text, fact, rule, action })` writes only the words that changed since it last said them (`fact`, an
 * optional `{ word, text }` line under the first, is what the control observed apart from its value), so a translation the
 * overlay put in their place stays; a row that watches its words for a refit leaves the hint's alone
 * (composer_owner_controls.js). `onAction()` runs the action. Its consumer shows and hides it; `dismissOutside` lets a
 * press outside its control hide it too (a capsule's floating list is outside its host and still its own, so a capsule
 * keeps its hint), and `boundary` is the clipping box it stays inside. A hide asked for while focus moves into the
 * hint (a consumer hiding on its host's focusout) is not done: the words or the action taking that focus stay.
 */
import { bindScrollFade } from './scroll_fade.js';

let made = 0;
const EDGE_PX = 8;
const GAP_PX = 8;  // ui.css: the hint stands var(--space-2) from its control
// Words with no action that wait past an edge are a named keyboard region: the keyboard's one way in (WebKit reaches
// nothing else there). Words with an action keep the action as their stop, and reaching it scrolls them.
const WORDS_REGION = 'Hint';

export function createHint(host, { doc = host.ownerDocument, onAction = () => {}, boundary = null, dismissOutside = false } = {}) {
    const el = doc.createElement('div');
    // The shell is the words' focus frame (ui.css): a keyboard in them rings the shell while they keep fading.
    el.className = 'ui-hint scroll-fade-focus-frame';
    el.id = `ui-hint-${++made}`;
    el.hidden = true;
    // Its words live in their own body: only they scroll and fade, so the shell keeps the material, the frame and the
    // shadow (ui.css [data-cramped]; DESIGN principle 9: fade bodies, not shells).
    const body = doc.createElement('div');
    body.className = 'ui-hint-body scroll-fade-y';
    const line = doc.createElement('p');
    line.className = 'ui-hint-line';
    const word = doc.createElement('span');
    word.className = 'ui-hint-word';
    const text = doc.createElement('span');
    text.className = 'ui-hint-text';
    line.append(word, text);
    const factLine = doc.createElement('p');
    factLine.className = 'ui-hint-line ui-hint-fact';
    factLine.hidden = true;
    const factWord = doc.createElement('span');
    factWord.className = 'ui-hint-word ui-hint-fact-word';
    const factText = doc.createElement('span');
    factText.className = 'ui-hint-text ui-hint-fact-text';
    factLine.append(factWord, factText);
    const rule = doc.createElement('p');
    rule.className = 'ui-hint-rule';
    const action = doc.createElement('button');
    action.type = 'button';
    action.className = 'ui-hint-action ui-hit-area';
    body.append(line, factLine, rule, action);
    el.append(body);
    const parts = { word, text, rule, action, factWord, factText };
    const said = { word: null, text: null, rule: null, action: null, factWord: null, factText: null };
    const write = (part, words = '') => {
        if (said[part] === words) return;
        said[part] = words;
        parts[part].textContent = words;
        parts[part].hidden = !words;
    };
    const onClick = () => onAction();
    const outside = (event) => { if (!host.contains(event.target)) hide(); };
    // Where focus leaving something in its host goes, while that focusout is delivered: noted on the host before its
    // consumer hears it (captured), forgotten on the document after (bubbling).
    let entering = null;
    const noteEntry = (event) => {
        entering = event.relatedTarget && el.contains(event.relatedTarget) ? event.relatedTarget : null;
    };
    const forgetEntry = () => { entering = null; };
    // While it stands: any scroll (the document's or a scroller's, caught on its way down) can move its control
    // toward an edge, so it is placed again; with `dismissOutside`, a press elsewhere hides it.
    const listen = (on) => {
        const how = on ? 'addEventListener' : 'removeEventListener';
        doc[how]?.('scroll', later, { capture: true, passive: true });
        if (dismissOutside) doc[how]?.('pointerdown', outside, true);
        host[how]?.('focusout', noteEntry, true);
        doc[how]?.('focusout', forgetEntry);
    };
    const hide = () => {
        if (!el.hidden && entering) return;  // focus on its way into its words: hidden, they would drop it
        if (!el.hidden) listen(false);
        el.hidden = true;
        entering = null;
    };
    parts.action.addEventListener('click', onClick);
    host.append(el);
    // A cramped hint scrolls its own words (ui.css [data-cramped]); the cue follows them while they wait past an edge.
    let region = null;
    let disposeFade = () => {};
    const bindWords = () => {
        const name = said.action ? '' : WORDS_REGION;
        if (name === region) return;
        disposeFade();
        region = name;
        disposeFade = bindScrollFade(body, { region: name });
    };
    bindWords();

    // Above its control while it fits there, else on the side with more room, and never past the window: where its
    // words need more than that room it scrolls inside itself. It starts where its control starts and moves back where
    // it would pass the far edge, never past the near one (right to left, the start is the right edge).
    const win = doc.defaultView;
    // Its words' height, whatever room they were given: the body's own content plus the shell's padding and border.
    // Read from the body, not from the shell: a scrolling body leaves the shell reporting only the room it was given,
    // which would make the next placement think the words fit, grow the hint and make it cramped again (it breathes).
    const naturalHeight = () => {
        const style = win?.getComputedStyle?.(el);
        const padding = (name) => Number.parseFloat(style?.getPropertyValue?.(name)) || 0;
        return body.scrollHeight + padding('padding-top') + padding('padding-bottom') + 2;
    };
    function place() {
        const width = Number(doc.documentElement?.clientWidth);
        if (!Number.isFinite(width) || width <= 0 || el.hidden) return;
        const view = win?.visualViewport;
        let top = Number(view?.offsetTop) || 0;
        let bottom = top + (Number(view?.height) || Number(doc.documentElement.clientHeight) || 0);
        let left = 0, right = width;
        if (boundary) {  // A consumer inside a clipping box (a scroller, a collapse) supplies its visible content box.
            const clip = boundary.getBoundingClientRect();
            top = Math.max(top, clip.top + boundary.clientTop);
            bottom = Math.min(bottom, clip.top + boundary.clientTop + boundary.clientHeight);
            left = Math.max(left, clip.left + boundary.clientLeft);
            right = Math.min(right, clip.left + boundary.clientLeft + boundary.clientWidth);
            el.style.setProperty('--ui-hint-span', `${Math.max(0, Math.floor(right - left - 2 * EDGE_PX))}px`);  // no wider
        }
        const anchor = host.getBoundingClientRect();
        const natural = naturalHeight();
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
        if (box.right > right - EDGE_PX) dx = right - EDGE_PX - box.right;
        if (box.left + dx < left + EDGE_PX) dx = left + EDGE_PX - box.left;
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
    resized?.observe(body);
    if (boundary) resized?.observe(boundary);
    win?.addEventListener?.('resize', place);
    win?.visualViewport?.addEventListener?.('resize', place);

    return {
        el,
        say({ word = '', text = '', fact = null, rule = '', action = '' } = {}) {
            write('word', word);
            write('text', text);
            write('factWord', fact?.word || '');
            write('factText', fact?.text || '');
            factLine.hidden = !fact?.word;
            write('rule', rule);
            write('action', action);
            bindWords();
            if (!el.hidden) { place(); later(); }
        },
        show() {
            if (!el.hidden) return;
            el.hidden = false;
            listen(true);
            place();
            later();
        },
        hide,
        destroy() {
            entering = null;
            hide();
            disposeFade();
            parts.action.removeEventListener('click', onClick);
            resized?.disconnect();
            if (frame) win.cancelAnimationFrame(frame);
            win?.removeEventListener?.('resize', place);
            win?.visualViewport?.removeEventListener?.('resize', place);
            el.remove();
        },
    };
}
