/** The shell's layout facts, read through one door.
 *
 * `web/theme.js` owns the switch: it runs before CSS, so the first paint is already the
 * right shell, writes `html[data-layout="docked"|"stack"]`, keeps it current on resize and
 * rotation, and publishes its number once as `window.ouroLayout`. Everything else
 * (app.js, client_surface.js, the layer stack, the drawer drag) asks this module and never
 * writes a media query of its own; the shell's CSS keys off the attribute. Beside it the
 * primary input (`html[data-pointer]`) is read here too; the few components that keep their
 * own coarse-pointer query (the surface fact, a row's long press, the reader, the Interface
 * size) are a closed list (tests/test_shell_layout_static.py). */

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

/** The primary input theme.js paints beside the layout: `coarse` (a finger) or `fine` (a mouse or a trackpad); a
 * document with neither (a test's fake DOM) is a desktop window's, as theme.js itself falls back. */
export const POINTERS = Object.freeze(['fine', 'coarse']);

export function currentPointer({ root = rootOf(), w = win() } = {}) {
    const painted = root?.dataset?.pointer;
    if (POINTERS.includes(painted)) return painted;
    const resolved = w?.ouroLayout?.pointer;
    return POINTERS.includes(resolved) ? resolved : 'fine';
}

export function isCoarsePointer(options) {
    return currentPointer(options) === 'coarse';
}

/** A Project open over Main covers it whole: the stack layout's room, or the docked content column where the primary
 * input is a finger (DESIGN "Shell"). Beside Main (a desk) it does not. */
export function roomCoversMain(options) {
    return isStackLayout(options) || isCoarsePointer(options);
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

/** A window too short for the sidebar's zones pins its head over the one scroller (style.css); the rail stands beside
 * the column, never over its rows. The head's measured height is the scroll margin of the rows there, so a row reached
 * with Tab or brought into view is never left under it. Its border box is watched: the macOS title band and the safe
 * area are its padding and can arrive after the first paint (a Telegram inset, a page zoom), which a content box never
 * reports. The column is watched too: where the head and one Project row no longer fit in it with their clearance,
 * `data-bars-crowd` lets the head scroll away with the list (a pinned head would cover every row). Independently, the
 * rail and its groups are watched: where the rail's natural height (ignoring auto-margin stretch space on
 * `.nav-rail-system`) passes the column's available content height, `data-rail-crowd` unpins the rail so its lower
 * controls scroll cleanly into view. Returns the disposer. */
export function measurePinnedBars(sidebar, { Observer = globalThis.ResizeObserver } = {}) {
    const head = sidebar?.querySelector('.sidebar-head');
    if (!head || typeof Observer !== 'function') return () => {};
    const rail = sidebar.querySelector('.nav-rail');
    const px = (value) => parseFloat(value) || 0;
    const write = () => {
        const view = sidebar.ownerDocument?.defaultView;
        const sideStyle = view?.getComputedStyle?.(sidebar);
        const headH = head.getBoundingClientRect().height;
        sidebar.style.setProperty('--sidebar-head-h', `${headH}px`);
        const row = sidebar.querySelector('.nav-project-row')?.getBoundingClientRect().height || 48;
        const clearance = px(sideStyle?.getPropertyValue?.('--space-4'));
        const room = Number(sidebar.clientHeight) || 0;
        sidebar.toggleAttribute?.('data-bars-crowd', room > 0 && headH + row + 2 * clearance > room);
        if (rail) {
            const railStyle = view?.getComputedStyle?.(rail);
            const railRoom = room - px(sideStyle?.paddingTop) - px(sideStyle?.paddingBottom);
            let railNeed = px(railStyle?.paddingTop) + px(railStyle?.paddingBottom);
            let visible = 0;
            for (const child of rail.children || []) {
                const height = child.getBoundingClientRect?.().height || 0;
                if (!(height > 0)) continue;
                const childStyle = view?.getComputedStyle?.(child);
                const mt = child.classList?.contains?.('nav-rail-system') ? 0 : px(childStyle?.marginTop);
                railNeed += height + mt + px(childStyle?.marginBottom);
                visible += 1;
            }
            if (visible > 1) railNeed += (px(railStyle?.rowGap) || px(railStyle?.gap)) * (visible - 1);
            sidebar.toggleAttribute?.('data-rail-crowd', railRoom > 0 && railNeed > railRoom);
        }
    };
    const observer = new Observer(write);
    observer.observe(head, { box: 'border-box' });
    observer.observe(sidebar, { box: 'border-box' });
    if (rail) {
        observer.observe(rail, { box: 'border-box' });
        for (const child of rail.children || []) observer.observe(child, { box: 'border-box' });
    }
    write();
    return () => observer.disconnect();
}

/** Hear the switch move; returns the disposer. The handler gets the new layout name. */
export function onLayoutChange(handler, { w = win() } = {}) {
    const listener = (event) => handler(event?.detail?.layout || currentLayout({ w }));
    w.addEventListener('ouro:layout-changed', listener);
    return () => w.removeEventListener('ouro:layout-changed', listener);
}

/** Does a section (Settings, the Dashboard) hold its list beside the open page? The sheets draw that from a 40rem box
 * with a container query (ui.css `@container sectionplace`), and the section's module asks here before it treats a page
 * as standing beside the list. An engine without container queries (WebKit before 16, Chromium before 105) draws one
 * column at every width, so there the answer is no: a module that assumed two panes would hide Back and never show
 * the list. A document that cannot say (a test's fake DOM) is a current engine. */
export function sectionHoldsTwoPanes(page, { w = win(), root = rootOf() } = {}) {
    if (w?.CSS?.supports && !w.CSS.supports('container-type', 'inline-size')) return false;
    const rem = parseFloat(w?.getComputedStyle?.(root)?.fontSize) || 16;
    return (page?.getBoundingClientRect?.().width || 0) >= 40 * rem;
}
