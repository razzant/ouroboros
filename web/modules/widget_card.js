/* Widgets card chrome for framed (module / route-iframe) cards: the effective
   launch policy (and whether it keeps the card running while Widgets is
   hidden), the card's ONE primary control (Start / Stop), the secondary
   launch-policy menu, and the facade a stopped card shows in place of its
   frame.
   widgets.js owns the registry and decides WHEN a card mounts or stops; this
   module only renders and reads the controls. Declarative cards are host-drawn
   and get none of this. */

import { PAGE_ICONS } from './page_icons.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { widgetKey } from './widget_list.js';
import { frameHeight, setFrameHeight } from './widget_module.js';

// Mirrors the validator's WIDGET_START_MODES (ouroboros/extension_ui_validation.py,
// the SSOT for the enum and the per-kind defaults).
export const WIDGET_START_MODES = ['auto', 'manual', 'retain'];
const KIND_DEFAULT_START = { declarative: 'auto', module: 'manual', iframe: 'manual' };
export const WIDGET_START_MODE_LABELS = {
    auto: 'Auto',
    manual: 'Manual',
    retain: 'Keep running',
};
// `tab.icon` is a glyph — an emoji or a symbol character. An identifier-like
// name (the `extension` default `register_ui_tab` stamps, or a named-icon set
// the host does not have) is not one; the facade shows the page's glyph instead.
const ICON_NAME = /^[a-z][a-z0-9_-]*$/i;

export function isFramedWidget(tab) {
    const kind = tab?.render?.kind;
    return kind === 'module' || kind === 'iframe';
}

/**
 * Effective launch policy of one card: the owner's override
 * (`ui_preferences.widget_start_mode[key]`) wins over the author's validated
 * `render.start`, which wins over the kind default (module/iframe → manual,
 * declarative → auto) for payloads registered before the validator filled it.
 * `retain` starts like `auto` and additionally keeps the card running while
 * the owner is on other pages (`isRetainedWidget`).
 */
export function effectiveStartMode(tab, prefs) {
    const owner = prefs?.widget_start_mode?.[widgetKey(tab)];
    if (WIDGET_START_MODES.includes(owner)) return owner;
    const author = tab?.render?.start;
    if (WIDGET_START_MODES.includes(author)) return author;
    return KIND_DEFAULT_START[tab?.render?.kind] || 'auto';
}

/**
 * A framed card the owner keeps running while Widgets is hidden. Only a frame
 * can be kept: a declarative card is host-drawn and always disposes on leave,
 * whatever an owner override says.
 */
export function isRetainedWidget(tab, prefs) {
    return isFramedWidget(tab) && effectiveStartMode(tab, prefs) === 'retain';
}

/** Whole-map replace payload for `POST /api/ui/preferences` (the `widget_order` shape). */
export function withWidgetStartMode(current, key, mode) {
    const next = current && typeof current === 'object' && !Array.isArray(current) ? { ...current } : {};
    next[key] = mode;
    return next;
}

// Head controls of a framed card: status (dot + text), the one primary button,
// and the launch-policy menu on the Skills card menu primitive
// (`.skills-card-menu` + `<dialog role="menu">`). The checked item is set by
// `syncWidgetCardControls` once the page knows the owner's preferences.
export function renderWidgetCardControls(tab) {
    if (!isFramedWidget(tab)) return '';
    const items = WIDGET_START_MODES.map((mode) => (
        `<button type="button" role="menuitemradio" class="skills-menu-item widgets-menu-item" data-widget-start-mode="${mode}" aria-checked="false"><span class="widgets-menu-check" aria-hidden="true">✓</span>${escapeHtml(WIDGET_START_MODE_LABELS[mode])}</button>`
    )).join('');
    return `<span class="ui-status" data-tone="neutral" data-widget-status hidden>Stopped</span>
        <button type="button" class="btn btn-primary btn-sm" data-widget-power>Start</button>
        <div class="skills-card-menu">
            <button type="button" class="skills-card-menu-trigger" aria-label="Launch policy" aria-haspopup="menu" aria-expanded="false" data-widget-menu-trigger>⋮</button>
            <dialog class="skills-card-menu-dialog" role="menu" aria-label="Launch policy">
                <div class="widgets-menu-heading">Launch policy</div>
                ${items}
            </dialog>
        </div>`;
}

const STATUS_TEXT = { starting: 'Starting…', running: 'Running', stopping: 'Stopping…' };

/**
 * Keep the head controls truthful. `state` is one of stopped | starting |
 * running | stopping — expressed through the button label, `disabled` while a
 * transition is in flight, and the status sentence; no state machine object.
 * A running card kept alive across pages (`mode === 'retain'`) says so: the
 * frame really keeps running while Widgets is hidden (the browser, not the
 * host, may pause its animation frames meanwhile — see CREATING_SKILLS).
 */
export function syncWidgetCardControls(card, state, mode = '') {
    const power = card?.querySelector('[data-widget-power]');
    if (!power) return;
    power.textContent = state === 'running' || state === 'stopping' ? 'Stop' : 'Start';
    power.disabled = state === 'starting' || state === 'stopping';
    const status = card.querySelector('[data-widget-status]');
    if (status) {
        status.hidden = state === 'stopped';
        status.dataset.tone = state === 'running' ? 'ok' : 'neutral';
        status.textContent = state === 'running' && mode === 'retain'
            ? 'Keeps running'
            : (STATUS_TEXT[state] || 'Stopped');
    }
    if (!mode) return;
    card.querySelectorAll('[data-widget-start-mode]').forEach((item) => {
        item.setAttribute('aria-checked', item.dataset.widgetStartMode === mode ? 'true' : 'false');
    });
}

/**
 * The stopped card's body: icon + title at the declared frame height (or the
 * 320 px floor — an auto-height module grows after Start; a known jump).
 * Idempotent: an existing facade is left alone, and so is a frame still in the
 * mount (a stop awaiting its acknowledgement keeps its iframe there).
 */
export function renderWidgetFacade(mount, tab) {
    if (!mount || mount.querySelector('[data-widget-facade], iframe')) return;
    const title = tab.title || tab.tab_id || tab.skill;
    const icon = String(tab.icon || '').trim();
    const glyph = !icon || ICON_NAME.test(icon) ? PAGE_ICONS.widgets : escapeHtml(icon);
    mount.innerHTML = `<div class="widgets-facade" data-widget-facade>
        <span class="widgets-facade-icon" aria-hidden="true">${glyph}</span>
        <strong class="widgets-facade-title">${escapeHtml(title)}</strong>
    </div>`;
    setFrameHeight(mount.firstElementChild, frameHeight(tab.render || {}));
}

let menusBound = false;

/**
 * Launch-policy menus: one delegated binding over the Widgets list, same
 * open/close behaviour as the Skills card menu (anchored non-modal `<dialog>`;
 * outside click, Escape and scroll close it). Selecting an item calls
 * `onSelectMode(cardKey, mode)`.
 */
export function bindWidgetCardMenus(list, onSelectMode) {
    if (!list || menusBound) return;
    menusBound = true;
    const closeMenus = (exceptMenu = null) => {
        list.querySelectorAll('.skills-card-menu').forEach((menu) => {
            if (menu === exceptMenu) return;
            const popover = menu.querySelector('.skills-card-menu-dialog');
            const trigger = menu.querySelector('[data-widget-menu-trigger]');
            // Focus goes back to the trigger when it was inside the closing menu
            // (Chromium does this for a <dialog>; WebKit does not).
            const hadFocus = Boolean(popover?.open && popover.contains(document.activeElement));
            if (popover?.open) popover.close();
            trigger?.setAttribute('aria-expanded', 'false');
            if (hadFocus) trigger?.focus({ preventScroll: true });
        });
    };
    list.addEventListener('click', (event) => {
        const trigger = event.target.closest('[data-widget-menu-trigger]');
        if (trigger) {
            const menu = trigger.closest('.skills-card-menu');
            const popover = menu?.querySelector('.skills-card-menu-dialog');
            const opening = !popover?.open;
            closeMenus(opening ? menu : null);
            if (!popover) return;
            trigger.setAttribute('aria-expanded', opening ? 'true' : 'false');
            if (opening) {
                popover.show();
                (popover.querySelector('[aria-checked="true"]') || popover.querySelector('[role="menuitemradio"]'))?.focus();
            } else {
                popover.close();
            }
            return;
        }
        const item = event.target.closest('[data-widget-start-mode]');
        if (!item) return;
        closeMenus();
        const key = item.closest('[data-widget-key]')?.dataset.widgetKey || '';
        if (key) onSelectMode(key, item.dataset.widgetStartMode || '');
    });
    document.addEventListener('click', (event) => {
        if (!event.target.closest?.('.widgets-card .skills-card-menu')) closeMenus();
    }, true);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeMenus();
    });
    window.addEventListener('scroll', () => closeMenus(), true);
}
