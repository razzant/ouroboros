/** The shell's layout facts, read through one door.
 *
 * `web/theme.js` owns the switch: it runs before CSS, so the first paint is already the
 * right shell, writes `html[data-layout="docked"|"stack"]`, keeps it current on resize and
 * rotation, and publishes its number once as `window.ouroLayout`. Everything else
 * (app.js, client_surface.js, the layer stack, the drawer drag) asks this module and never
 * writes a media query of its own; the shell's CSS keys off the attribute
 * (tests/test_shell_layout_static.py). */

export const LAYOUTS = Object.freeze(['docked', 'stack']);

const win = () => globalThis.window ?? globalThis;
const rootOf = (doc = globalThis.document) => doc?.documentElement ?? null;

/** The published number and the current layout; theme.js must have run. */
export function layoutFacts(w = win()) {
    const facts = w?.ouroLayout;
    if (!facts) throw new Error('web/theme.js has not published window.ouroLayout');
    return facts;
}

/** `docked` or `stack`: the attribute theme.js painted, else what it resolves right now; a document with
 * neither (a test's fake DOM) is a window, so docked, as theme.js itself falls back. */
export function currentLayout({ root = rootOf(), w = win() } = {}) {
    const painted = root?.dataset?.layout;
    if (LAYOUTS.includes(painted)) return painted;
    const resolved = w?.ouroLayout?.layout;
    return LAYOUTS.includes(resolved) ? resolved : 'docked';
}

export function isStackLayout(options) {
    return currentLayout(options) === 'stack';
}

/** A pinch is magnifying the page: the visual viewport is narrower than the layout one, the root's client width
 * (`innerWidth` runs a pixel past the screen at an Interface size below 100 % and counts a classic scrollbar). A
 * keyboard only takes height, and a page zoom moves both together, so neither counts. While it holds, the room keeps its
 * size and the finger pans the magnified page (DESIGN "Touch and containment"). */
export function pinchMagnified(w = win()) {
    const vv = w?.visualViewport;
    const layoutWidth = Number(w?.document?.documentElement?.clientWidth) || 0;
    return Boolean(vv && layoutWidth && Number(vv.width) < layoutWidth - 1);
}

/** A window too short for the sidebar's zones pins its head and its foot over the one scroller (style.css). Their
 * measured heights are the scroll margin of the rows there, so a row reached with Tab or brought into view is never
 * left under either bar. Their border boxes are watched: the macOS title band and the safe area are their padding
 * and can arrive after the first paint (a Telegram inset, a page zoom), which a content box never reports. Returns
 * the disposer. */
export function measurePinnedBars(sidebar, { Observer = globalThis.ResizeObserver } = {}) {
    const head = sidebar?.querySelector('.sidebar-head');
    const foot = sidebar?.querySelector('.sidebar-foot-meta');
    if (!head || !foot || typeof Observer !== 'function') return () => {};
    const write = () => {
        sidebar.style.setProperty('--sidebar-head-h', `${head.getBoundingClientRect().height}px`);
        sidebar.style.setProperty('--sidebar-foot-h', `${foot.getBoundingClientRect().height}px`);
    };
    const observer = new Observer(write);
    observer.observe(head, { box: 'border-box' });
    observer.observe(foot, { box: 'border-box' });
    write();
    return () => observer.disconnect();
}

/** Hear the switch move; returns the disposer. The handler gets the new layout name. */
export function onLayoutChange(handler, { w = win() } = {}) {
    const listener = (event) => handler(event?.detail?.layout || currentLayout({ w }));
    w.addEventListener('ouro:layout-changed', listener);
    return () => w.removeEventListener('ouro:layout-changed', listener);
}
