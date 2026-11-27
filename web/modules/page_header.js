import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { revealPastFade, bindScrollFade } from './scroll_fade.js';
import { SHELL_ICONS } from './page_icons.js';
import { pageLabel } from './pages.js';
import { statusMarkHtml } from './status_mark.js';
import { currentLayout, onLayoutChange } from './layout.js';
import { fmt, translateString } from './i18n.js';

function classAttr(parts) {
    return parts.filter(Boolean).join(' ');
}

// The control is Ouroboros's own picture: press him, and his places open, with his card on top.
// The shell's ☰: one of the shell's icon controls (style.css .shell-icon-control), shown in the stack layout. A
// page's header carries it in its leading slot, where the drawer's × then sits; a room (Main, a Project) carries it
// as the composer row's first control instead, under the thumb (owner 19.10, 1A), as tall as the capsules beside it
// with the touch minimum as its finger target. Its status dot says what waits
// behind it, in the drawer the ☰ opens (app.js paints it and the name's words).
export function renderMobileNavToggle({ inComposer = false } = {}) {
    const classes = inComposer ? 'shell-icon-control mobile-nav-toggle composer-menu ui-hit-area' : 'shell-icon-control mobile-nav-toggle';
    return `
        <button class="${classes}" type="button" data-mobile-nav-toggle aria-label="Open navigation" aria-controls="primary-sidebar" aria-expanded="false"><span class="places-face" aria-hidden="true"></span>${statusMarkHtml()}</button>
    `;
}

/** A room's edge (style.css "A room's edges"): its two blur steps, then its tint, each on the shared edge curve. A
 * room's bar and its composer's area each carry it. */
export const ROOM_EDGE_HTML = ['soft', 'deep', 'tint']
    .map((step) => `<i class="room-edge" data-edge="${step}" aria-hidden="true"></i>`).join('');

/**
 * The one header of every page AND the Project panel (docs/DESIGN.md "Shell", plan P5): one row, one height,
 * one title style, one close control. `☰ · title · status … actions · ×`: the leading slot holds ☰ (the stack
 * layout shows it), the title row takes the room's status after the title (room_work.js), the toolbar and the
 * trailing slot end the row. A page has no caption line: the sidebar's row and its tabs name it. `named` says the
 * sidebar's chosen row already names this place: on a desk such a header shows no title (the h2 stays for readers)
 * and its lead slot takes the title's place with the page's own navigation (bindPageHeader); a header with nothing
 * to show draws no bar. A tab strip is written under the row, where the stack layout keeps it. A Project keeps its
 * name. No icon anywhere: the place is named by words, never a glyph (owner 5A).
 */
export function renderPageHeader({
    page = '',   // a registry id: the title is its label unless `title` says otherwise
    title,
    titleId = '',
    leadingHtml,
    toolbarHtml = '',
    trailingHtml = '',
    actionsHtml = '',
    tabsHtml = '',
    variant = '',
    className = '',
    toolbarClass = '',
    showMobileNav = true,
    named = true,
    leadHtml = '',  // the page's own navigation written straight into the lead (Settings): the bar is never empty
    roomEdge = false,  // a room's bar floats over its feed and carries the room's edge
} = {}) {
    const variantClass = variant ? `app-page-header-${escapeHtml(variant)}` : '';
    const words = title ?? pageLabel(page);
    const leading = leadingHtml !== undefined
        ? leadingHtml
        : (showMobileNav ? renderMobileNavToggle() : '');
    const toolbar = (toolbarHtml || actionsHtml)
        ? `<div class="${classAttr(['app-page-toolbar', 'app-page-actions', escapeHtml(toolbarClass)])}">${toolbarHtml || actionsHtml}</div>`
        : '';
    const trailing = trailingHtml
        ? `<div class="app-page-trailing">${trailingHtml}</div>`
        : '';
    const tabs = tabsHtml
        ? `<div class="app-page-tabs">${tabsHtml}</div>`
        : '';
    const facts = [
        named ? ' data-sidebar-names' : '',
        named && !toolbar && !trailing && !tabs && !leadHtml ? ' data-empty' : '',
        leadHtml ? ' data-lead' : '',
    ].join('');
    const id = titleId ? ` id="${escapeHtml(titleId)}"` : '';
    return `
        <div class="${classAttr(['page-header', 'app-page-header', variantClass, className])}"${facts}>${roomEdge ? ROOM_EDGE_HTML : ''}
            <div class="app-page-leading">${leading}</div>
            <div class="app-page-title-block">
                <div class="app-page-title-row">
                    <h2 class="app-page-title"${id}>${escapeHtml(words)}</h2>
                </div>${!leadHtml ? '<span class="app-page-return" data-page-return></span>' : ''}${named || leadHtml ? `<div class="app-page-lead">${leadHtml}</div>` : ''}
            </div>
            ${toolbar}
            ${trailing}
        </div>
        ${tabs}
    `;
}

/**
 * Back (docs/DESIGN.md "Page header", the Back control): one recipe for the whole app, a jump's return and a level's
 * Back alike. The shell's round icon control, as ☰ and × are (style.css .shell-icon-control: the chrome height on a
 * desk, the touch minimum in the stack layout), holding the chevron-left at 20 px; where it leads is its tooltip and
 * accessible name. A return to another place also draws that destination after the chevron on a desk (`‹ Settings`,
 * owner 27.11.2026; bindPageBack writes it), a level's Back stays the circle. `attrs` are the caller's hooks (a data
 * attribute its handler finds it by).
 */
export function renderBackControl({ to = '', title = to, attrs = '' } = {}) {
    const name = title ? `Back to ${title}` : 'Back';
    return `<button type="button" class="shell-icon-control shell-back" ${attrs} aria-label="${escapeHtml(name)}" title="${escapeHtml(name)}">${SHELL_ICONS.back}</button>`;
}

// A place is a page, a Project its own room: `settings:models` and `settings:tools` are one place, `project:a` and
// `project:b` two (the keys page_navigation.js writes).
const placeOf = (key) => (String(key).startsWith('project:') ? String(key) : String(key).split(':')[0]);

/** The destination's localized steps in the capsule, one node each (style.css draws the › between them). A Project's
 * name is the owner's: written as it is, in its own direction. */
function nameBack(button, steps, own, doc) {
    let label = button.querySelector('.shell-back-label');
    if (!steps) {
        label?.remove();
        button.removeAttribute('data-named');
        return;
    }
    if (!label) {
        label = doc.createElement('span');
        label.className = 'shell-back-label';
        button.append(label);
    }
    label.replaceChildren(...steps.map((part) => {
        const step = doc.createElement('span');
        step.textContent = part;
        return step;
    }));
    if (own) label.setAttribute('dir', 'auto'); else label.removeAttribute('dir');
    button.setAttribute('data-named', '');
}

/** An owed jump occupies the level's one Back slot until spent. Placement is a writer fact;
 * late children never decide layout through :has(:empty). A return to another place names it (nameBack) and marks the
 * header `data-return-named`, so the page's own name stands after it. */
export function bindPageBack({ layers, currentHeader, doc = document, win = window }) {
    let previous = null;
    function paint() {
        const header = currentHeader();
        if (previous && previous !== header) {
            previous.removeAttribute('data-return');
            previous.removeAttribute('data-return-named');
            previous.querySelector('[data-page-return]')?.replaceChildren();
        }
        previous = header;
        if (!header) return;
        const slot = header.querySelector('[data-page-return]');
        const column = slot?.closest('.section-lead-page');
        const primary = layers.page();  // the top page layer: a level opened after a jump peels before the jump
        const from = primary?.returnTo?.key;
        const crossing = Boolean(slot && primary?.back && from && placeOf(from) !== placeOf(primary.destination));
        if (primary?.back) header.setAttribute('data-return', column ? 'column' : 'leading');
        else header.removeAttribute('data-return');
        if (crossing) header.setAttribute('data-return-named', ''); else header.removeAttribute('data-return-named');
        if (!slot) return;
        let button = slot.querySelector('[data-shell-back]');
        const back = primary?.back;
        if (!back) { button?.remove(); return; }
        if (!button) {
            const holder = doc.createElement('span');
            holder.innerHTML = renderBackControl({ attrs: 'data-shell-back' });
            button = holder.firstElementChild;
            slot.append(button);
            button.addEventListener('click', event => layers.back(event.detail ? 'pointer' : 'keyboard', button.dataset.pageLayer));
        }
        button.dataset.pageLayer = primary.id;
        // One localized destination for the words, the accessible name and the tooltip, composed here once: the product's
        // words translated, the names a person gave (a Project's name, a Files path: the layer's `literal`) as written.
        // The page overlay never enters the control (data-i18n-authored), so nothing rewrites it afterwards.
        const own = String(from).startsWith('project:');
        const literal = new Set(back.literal || []);
        const steps = String(back.title || back.label).split(' › ').map((part) => (own || literal.has(part) ? part : translateString(part)));
        const name = fmt('Back to {place}', { place: steps.join(' › ') });
        button.setAttribute('data-i18n-authored', '');
        button.setAttribute('aria-label', name);
        button.title = name;
        nameBack(button, crossing ? steps : null, own, doc);
    }
    const off = layers.onChange(paint);
    win.addEventListener('ouro:page-shown', paint);
    win.addEventListener('ouro:language-changed', paint);
    paint();
    return () => { off(); win.removeEventListener('ouro:page-shown', paint); win.removeEventListener('ouro:language-changed', paint); };
}

/**
 * A page's own navigation in its bar (docs/DESIGN.md "Page header", owner 19.10 A): on a desk the bar of a page the
 * sidebar names starts with the page's tabs or location, where the hidden title stood, and ends with its actions, so
 * it has two ends as Main's bar has; in the stack layout the same nodes return under the row, where a phone keeps
 * them. `lead` defaults to what renderPageHeader wrote under the row (the tab strip); `actions` join the toolbar's
 * start. Nodes move and are never re-rendered, so their listeners, selection and focus stay theirs; the selected
 * tab is revealed again by its strip's own observer. Returns the disposer.
 */
export function bindPageHeader(header, { lead = null, actions = null } = {}) {
    const doc = header?.ownerDocument;
    const under = header?.nextElementSibling?.classList?.contains('app-page-tabs') ? header.nextElementSibling : null;
    const moves = [[lead || under?.firstElementChild, header?.querySelector('.app-page-lead'), 'append'],
        [actions, header?.querySelector('.app-page-toolbar'), 'prepend']]
        .filter(([node, bar]) => node && bar)
        .map(([node, bar, how]) => {
            // A mark holds each node's home under the row.
            const home = doc.createComment('');
            node.before(home);
            return { node, bar, how, home };
        });
    // A bar whose actions are already in it (Circles), or that holds the place's name alone (Widgets), still lays out
    // as the desk band of every page.
    if (!header) return () => {};
    let docked = null;
    const place = (layout) => {
        const active = doc.activeElement;
        const next = (typeof layout === 'string' ? layout : currentLayout()) === 'docked'
            || doc.documentElement?.dataset.telegramFullscreen === 'active';
        if (next === docked) return;
        docked = next;
        for (const { node, bar, how, home } of moves) {
            if (docked) bar[how](node); else home.after(node);
        }
        if (under) under.hidden = docked;
        header.toggleAttribute('data-lead', docked);  // the bar's two ends: style.css lays them on one line
        header.toggleAttribute('data-scroll-x', docked); // this band owns overflow, like its tab and folder rails
        header.classList.toggle('scroll-fade-x', docked);
        // An engine drops focus from a node it moves: the owner's place in the page stays where it was.
        if (active?.isConnected && doc.activeElement !== active) active.focus?.({ preventScroll: true });
    };
    place();
    const disposeFade = bindScrollFade(header, { axis: 'x' });
    const offLayout = onLayoutChange(place);
    doc.defaultView?.addEventListener?.('ouro:host-chrome', place);
    return () => { disposeFade(); offLayout(); doc.defaultView?.removeEventListener?.('ouro:host-chrome', place); };
}

/** A bar narrower than this (px) cannot hold the status, the consciousness chip, Restart, Panic and the budget. */
export const HEADER_FULL_ROW_MIN_WIDTH = 640;

/**
 * Main's Restart and budget by the room the bar has (owner 25.10, 4A): a bar wide enough keeps them in its row; the
 * stack layout, and a docked content column narrower than HEADER_FULL_ROW_MIN_WIDTH, move the same two nodes into the
 * room status's Shade (`shade.extras`, room_work.js), Restart first, so the bar keeps the status, the consciousness
 * chip and Panic, and no ⋯. Each control keeps its one handler and its one id. The bar's width is measured (a
 * ResizeObserver), never a viewport breakpoint. Returns the disposer.
 */
export function bindHeaderControlPlacement({ headerActions, budgetPill = null, shade = null, bar = headerActions?.closest?.('.app-page-header') } = {}) {
    const restart = headerActions?.querySelector?.('[data-chat-command="restart"]');
    const panic = headerActions?.querySelector?.('[data-chat-command="panic"]');
    const extras = shade?.extras;
    const toolbar = budgetPill?.parentElement || null;
    if (!restart || !extras) return () => {};
    let placed = '';
    const compact = (layout) => layout === 'stack'
        || (Number(bar?.clientWidth) > 0 && bar.clientWidth < HEADER_FULL_ROW_MIN_WIDTH);
    // Called with the layout name by the layout switch, with entries by the observer, with nothing at bind.
    const place = (arg) => {
        // A retained room out of sight has no width. Keep its controls where the return captured them.
        if (bar?.clientWidth === 0) return;
        const next = compact(typeof arg === 'string' ? arg : currentLayout()) ? 'shade' : 'row';
        if (next === placed) return;
        placed = next;
        if (next === 'shade') {
            extras.append(restart);
            if (budgetPill) extras.append(budgetPill);
        } else {
            headerActions.insertBefore(restart, panic);
            if (budgetPill && toolbar) toolbar.insertBefore(budgetPill, headerActions.nextSibling);
        }
        shade.update?.();
    };
    place();
    const offLayout = onLayoutChange(place);
    const Observer = bar?.ownerDocument?.defaultView?.ResizeObserver;
    const observer = Observer && bar ? new Observer(place) : null;
    observer?.observe(bar);
    return () => { offLayout(); observer?.disconnect(); };
}

// SSOT for a segmented single-select control, the strip (the settings effort/mode
// toggles): one generator owns the markup so every group renders identically — the
// sibling of renderTabStrip/.app-tab, but for in-card choices. The markup is all it
// writes: ui.css lays out the track (.ui-segment-group) and its options
// (.ui-segment), each as wide as its word, so nothing is counted, measured or
// written inline, and no hover title carries a sentence the owner needs. Buttons
// keep the data-effort-* hooks the controls layer binds on. `modifier` is a bare
// boolean data-attribute (e.g. 'data-place-state') scripts find the group by.
// A caller that knows the choice marks it here (`chosen`; else its reader marks it
// after the paint, as place_sheet.js does), names the group (`label`) and holds every
// option while a write is in flight (`busy`: disabled, the group aria-busy).
export function renderSegmentedField({ target, options = [], modifier = '', chosen, label = '', busy = false } = {}) {
    const tgt = String(target || '').trim();
    if (!tgt) {
        throw new Error('renderSegmentedField requires target');
    }
    const marked = chosen !== undefined && chosen !== null;
    const buttons = options.map((opt) => {
        const value = String(opt.value ?? '');
        const on = marked && value === String(chosen);
        const state = (marked ? ` aria-pressed="${on}"` : '') + (busy ? ' disabled' : '');
        return `<button type="button" class="ui-segment${on ? ' active' : ''}" data-effort-value="${escapeHtml(value)}"${state}>${escapeHtml(opt.label ?? value)}</button>`;
    }).join('');
    const mod = String(modifier || '').trim();
    const modAttr = mod ? ` ${mod}` : '';
    const named = (label ? ` role="group" aria-label="${escapeHtml(label)}"` : '') + (busy ? ' aria-busy="true"' : '');
    return `
        <div class="ui-segment-group" data-effort-group${modAttr} data-effort-target="${escapeHtml(tgt)}"${named}>
            ${buttons}
        </div>
    `;
}

export function renderTabStrip({
    items = [],
    active = '',
    dataAttr,
    activeClass = 'active',
    ariaLabel = 'Page views',
    stripClass = '',
    tabClass = '',
} = {}) {
    const attr = String(dataAttr || '').trim();
    if (!attr) {
        throw new Error('renderTabStrip requires dataAttr');
    }
    const selected = items.find((item) => !item.disabled && String(item.value ?? item.id ?? '') === String(active))
        || items.find((item) => !item.disabled);
    const buttons = items.map((item) => {
        const value = String(item.value ?? item.id ?? '');
        const isActive = item === selected;
        const pill = item.pillId
            ? `<span class="${classAttr(['app-tab-pill', item.pillClass || ''])}" id="${escapeHtml(item.pillId)}" hidden></span>`
            : '';
        return `
            <button
                type="button"
                class="${classAttr(['app-tab', tabClass, item.className || '', isActive ? activeClass : ''])}"
                ${attr}="${escapeHtml(value)}"
                role="tab"
                aria-selected="${isActive ? 'true' : 'false'}"
                ${item.disabled ? 'disabled' : ''}
                ${item.tabId ? `id="${escapeHtml(item.tabId)}"` : ''}
                ${item.panelId ? `aria-controls="${escapeHtml(item.panelId)}"` : ''}
            >
                ${escapeHtml(item.label ?? value)}
                ${pill}
            </button>
        `;
    }).join('');
    return `
        <div class="${classAttr(['app-tab-strip', 'scroll-fade-x', stripClass])}" role="tablist" aria-label="${escapeHtml(ariaLabel)}">
            ${buttons}
        </div>
    `;
}

/**
 * Bind one rendered strip. onChange owns domain loading/panel visibility.
 * select(value) synchronizes an external navigation without calling onChange;
 * user activation calls it once, only when the selection actually changes.
 */
export function revealStripChoice(strip, tab) {
    if (!tab || !strip.clientWidth) return;
    revealPastFade(strip, tab, { axis: 'x' });
}

export function bindTabStrip(strip, { dataAttr, activeClass = 'active', onChange } = {}) {
    if (!dataAttr) throw new Error('bindTabStrip requires dataAttr');
    let disposed = false;
    const disposeFade = bindScrollFade(strip, { axis: 'x' });
    const tabs = () => Array.from(strip.querySelectorAll(`[role="tab"][${dataAttr}]`))
        .filter((tab) => tab.closest('[role="tablist"]') === strip);
    // External navigation may select a tab before its page becomes interactive.
    // A disabled/hidden choice itself stays unavailable on either path.
    const selectable = (tab) => !tab.disabled && tab.getAttribute('aria-disabled') !== 'true'
        && !tab.matches('[hidden], [inert]');
    const enabled = (tab) => selectable(tab) && !tab.closest('[hidden], [inert]');
    let selected = null;
    // A vertical list (the Dashboard's) scrolls in its own box, the list or the rail around it: a chosen row out of
    // that box's view comes in just past its edge fade (as Settings' rail), and nothing beyond the
    // box moves (never the page). A row already clear of the fade never moves.
    function revealInScroller(tab) {
        const view = strip.ownerDocument.defaultView;
        const doc = strip.ownerDocument;
        let box = strip;
        while (box && box !== doc.body && box !== doc.documentElement) {
            const overflow = view.getComputedStyle(box).overflowY;
            if ((overflow === 'auto' || overflow === 'scroll') && box.scrollHeight > box.clientHeight) break;
            box = box.parentElement;
        }
        if (!box || box === doc.body || box === doc.documentElement) return;
        revealPastFade(box, tab);
    }
    function revealSelected() {
        if (disposed || !strip.clientWidth) return;
        const tab = tabs().find((item) => item.getAttribute(dataAttr) === selected);
        if (!tab) return;
        if (strip.getAttribute('aria-orientation') === 'vertical') { revealInScroller(tab); return; }
        revealStripChoice(strip, tab);  // only the strip moves, never its page or scroll panel
    }
    function select(value, { focus = false } = {}) {
        if (disposed) return false;
        const all = tabs();
        const allowed = focus ? enabled : selectable;
        const next = all.find((tab) => tab.getAttribute(dataAttr) === String(value) && allowed(tab));
        if (!next) return false;
        selected = next.getAttribute(dataAttr);
        all.forEach((tab) => {
            const active = tab === next;
            tab.classList.toggle(activeClass, active);
            tab.setAttribute('aria-selected', String(active));
            tab.tabIndex = active ? 0 : -1;
        });
        revealSelected();
        if (focus) next.focus({ preventScroll: true });
        return true;
    }
    function activate(tab, focus) {
        const value = tab.getAttribute(dataAttr);
        const changed = value !== selected;
        if (select(value, { focus }) && !disposed && changed) onChange?.(value, tab);
    }
    const onClick = (event) => {
        const tab = event.target.closest?.('[role="tab"]');
        if (tabs().includes(tab) && enabled(tab)) activate(tab, false);
    };
    const onKey = (event) => {
        if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
        const all = tabs().filter(enabled);
        const tab = event.target.closest?.('[role="tab"]');
        const index = all.indexOf(tab);
        if (index < 0) return;
        const vertical = strip.getAttribute('aria-orientation') === 'vertical';
        const previous = vertical ? 'ArrowUp' : 'ArrowLeft';
        const next = vertical ? 'ArrowDown' : 'ArrowRight';
        let target;
        if (event.key === 'Home') target = all[0];
        else if (event.key === 'End') target = all[all.length - 1];
        else if (event.key === previous) target = all[(index + all.length - 1) % all.length];
        else if (event.key === next) target = all[(index + 1) % all.length];
        else return; // Enter/Space keep the native button click, never a second callback.
        event.preventDefault();
        activate(target, true);
    };
    strip.addEventListener('click', onClick);
    strip.addEventListener('keydown', onKey);
    // Initialization may happen on a hidden page; reveal when it gets a real
    // width, and keep the selected tab reachable after a narrower resize.
    const Observer = strip.ownerDocument?.defaultView?.ResizeObserver;
    const observer = Observer ? new Observer(revealSelected) : null;
    observer?.observe(strip);
    const initial = tabs().find((tab) => selectable(tab) && tab.getAttribute('aria-selected') === 'true')
        || tabs().find(selectable);
    if (initial) select(initial.getAttribute(dataAttr));
    return {
        select,
        destroy() {
            disposed = true;
            disposeFade();
            strip.removeEventListener('click', onClick);
            strip.removeEventListener('keydown', onKey);
            observer?.disconnect();
        },
    };
}
