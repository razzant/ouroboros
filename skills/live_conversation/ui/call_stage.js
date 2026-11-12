/** The call's two sizes, and its time. The chip is the composer row itself; the big view is the same
 * row grown upward into a tall glass panel docked where the composer lives: on a wide window the room
 * stays readable above it, below 980 px it takes the whole screen. The passage moves the one row: it
 * leaves the composer's flow at its own measured rectangle (a spacer keeps the composer's height), grows
 * to the panel's rectangle, and shrinks back to the spacer's before it returns to the flow, so the
 * glass and its wave stretch as one object. While it moves the capsule writes `data-call-move` and its
 * words wait; reduced motion switches at once. Escape minimises too, a convenience only: every surface
 * has the visible capsule. The call's time ticks once a second beside the state while the call is live.
 */
import { callClock } from './call_mark.js';

// Where the window is a phone's (the shell's mobile layout), the panel takes the whole screen.
const PHONE_BELOW = 980;
const PANEL_MAX = 500;
const PANEL_SHARE = 0.66;
const PHONE_GAP = 8;
// call.css moves the lifted row in 0.64 s; the words come back a little before it lands.
const MOVE_MS = 460;
const LAND_MS = 720;

/** The panel's rectangle for the chip's: over the composer on a wide window, the screen on a phone. */
export function panelBox(chip, { width, height }) {
    if (width < PHONE_BELOW) {
        return { left: PHONE_GAP, top: PHONE_GAP, width: width - 2 * PHONE_GAP, height: height - 2 * PHONE_GAP, radius: 30 };
    }
    const tall = Math.min(PANEL_MAX, Math.round(height * PANEL_SHARE), Math.max(chip.height, chip.bottom - PHONE_GAP));
    return { left: chip.left, top: chip.bottom - tall, width: chip.width, height: tall, radius: 28 };
}

export function createCallStage({ wrap, row, time, doc, env = globalThis, escape = () => {} }) {
    let big = false;
    let spacer = null;
    let homeRadius = 0;
    let covered = [];  // what a full-screen panel took out of reach while it covers the page
    let origin = { x: 0, y: 0 };  // a transformed ancestor's offset of the fixed row, measured once per lift
    let timers = [];
    let since = 0;
    let ticking = 0;
    const reduced = () => Boolean(env.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches);
    const viewport = () => ({ width: env.innerWidth || 0, height: env.innerHeight || 0 });
    const later = (fn, ms) => timers.push(env.setTimeout(fn, ms));
    const clear = () => { timers.forEach(timer => env.clearTimeout(timer)); timers = []; };
    const rect = element => {
        const box = element.getBoundingClientRect();
        return { left: box.left, top: box.top, width: box.width, height: box.height, bottom: box.top + box.height };
    };

    function place(box) {
        row.style.left = `${box.left - origin.x}px`;
        row.style.top = `${box.top - origin.y}px`;
        row.style.width = `${box.width}px`;
        row.style.height = `${box.height}px`;
        row.style.setProperty('--call-panel-radius', `${box.radius ?? box.height / 2}px`);
    }

    function lift(chip) {
        homeRadius = Number.parseFloat(env.getComputedStyle?.(row)?.borderTopLeftRadius) || chip.height / 2;
        spacer = doc.createElement('div');
        spacer.className = 'call-spacer';
        spacer.setAttribute('aria-hidden', 'true');
        spacer.style.height = `${chip.height}px`;
        row.before(spacer);
        row.classList.add('call-lifted');
        row.style.transition = 'none';
        origin = { x: 0, y: 0 };
        place({ ...chip, radius: homeRadius });
        const landed = rect(row);
        origin = { x: landed.left - chip.left, y: landed.top - chip.top };
        if (origin.x || origin.y) place({ ...chip, radius: homeRadius });
    }

    /** A panel that covers the whole screen takes the rest of the page out of reach (focus, screen readers). */
    function cover(on) {
        for (const element of covered) element.inert = false;
        covered = [];
        for (let node = row; on && node?.parentElement && node !== doc.body; node = node.parentElement) {
            for (const sibling of node.parentElement.children) {
                if (sibling !== node && !sibling.inert) { sibling.inert = true; covered.push(sibling); }
            }
        }
    }

    function land() {
        row.classList.remove('call-lifted');
        for (const name of ['left', 'top', 'width', 'height', 'transition', '--call-panel-radius']) row.style.removeProperty(name);
        spacer?.remove();
        spacer = null;
        delete wrap.dataset.callMove;
    }

    /** From the chip to the panel; the capsule renders its big view right after. */
    function grow() {
        if (big) return;
        clear();
        big = true;
        if (!spacer) lift(rect(row));
        const instant = reduced();
        if (instant) delete wrap.dataset.callMove; else wrap.dataset.callMove = '1';
        void row.offsetWidth;  // the chip's rectangle is the passage's first frame
        row.style.transition = instant ? 'none' : '';
        place(panelBox(rect(spacer), viewport()));
        cover(viewport().width < PHONE_BELOW);
        if (!instant) later(() => { delete wrap.dataset.callMove; }, MOVE_MS);
    }

    /** From the panel back to the chip; `instant` when the call ends or the room changes under it. */
    function shrink({ instant = false } = {}) {
        if (!big) return;
        clear();
        big = false;
        cover(false);
        if (!spacer) return;
        if (instant || reduced()) { land(); return; }
        wrap.dataset.callMove = '1';
        row.style.transition = '';
        const home = rect(spacer);
        place({ ...home, radius: homeRadius });
        later(() => { delete wrap.dataset.callMove; }, MOVE_MS - 40);
        later(land, LAND_MS);
    }

    const resized = () => {
        if (!big || !spacer) return;
        row.style.transition = 'none';
        place(panelBox(rect(spacer), viewport()));
    };
    const key = (event) => {
        if (big && event.key === 'Escape' && !event.defaultPrevented && !event.isComposing) escape();
    };
    env.addEventListener?.('resize', resized);
    doc.addEventListener?.('keydown', key);

    function tick() {
        const text = since ? callClock(since, Date.now()) : '';
        if (time.textContent !== text) time.textContent = text;
        time.hidden = !text;
    }

    return {
        get big() { return big; },
        grow,
        shrink,
        /** The call's time: it runs from the first live moment while the call is live or reconnecting. */
        clock(view) {
            const running = ['live', 'reconnecting'].includes(view.state) && view.liveSince;
            since = running ? view.liveSince : 0;
            if (running && !ticking) ticking = env.setInterval(tick, 1000);
            if (!running && ticking) { env.clearInterval(ticking); ticking = 0; }
            tick();
        },
        destroy() {
            clear();
            if (ticking) env.clearInterval(ticking);
            ticking = 0;
            big = false;
            cover(false);
            if (spacer) land();
            env.removeEventListener?.('resize', resized);
            doc.removeEventListener?.('keydown', key);
        },
    };
}
