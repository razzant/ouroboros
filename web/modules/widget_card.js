/* Widgets card chrome: for framed (module / route-iframe) cards the effective
   launch policy (and whether it keeps the card running while Widgets is
   hidden), the card's ONE primary control (Start / Stop) and the facade a
   stopped card shows in place of its frame; for every card the card menu —
   the launch policy of a framed card and the owner's width steps (column
   spans: web/modules/widget_size.js).
   widgets.js owns the registry and decides WHEN a card mounts or stops; this
   module only renders and reads the controls. Declarative cards are host-drawn
   and get only the menu's width steps. */

import { PAGE_ICONS } from './page_icons.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { widgetKey } from './widget_list.js';
import { frameHeight, setFrameHeight } from './widget_module.js';
import { WIDGET_WIDTH_STEPS } from './widget_size.js';
import { bindMenu } from './ui_interactions.js';

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

// Head controls: a framed card's status (dot + text) and its one primary
// button, then the card menu on the Skills card menu primitive
// (`.skills-card-menu` + `<dialog role="menu">`): a framed card's launch policy
// and every card's width steps, each a radio group. The checked policy is set
// by `syncWidgetCardControls` once the page knows the owner's preferences; the
// checked width when the menu opens (`bindWidgetCardMenus`).
export function renderWidgetCardControls(tab) {
    const item = (attrs, label, role = 'menuitemradio') => (
        `<button type="button" role="${role}" class="skills-menu-item widgets-menu-item" ${attrs}><span class="widgets-menu-check" aria-hidden="true">${role === 'menuitemradio' ? '✓' : ''}</span>${escapeHtml(label)}</button>`
    );
    const group = (label, items) => `<div role="group" aria-label="${label}"><div class="widgets-menu-heading" aria-hidden="true">${label}</div>${items}</div>`;
    const framed = isFramedWidget(tab);
    const policy = framed ? group('Launch policy', WIDGET_START_MODES.map((mode) => (
        item(`data-widget-start-mode="${mode}" aria-checked="false"`, WIDGET_START_MODE_LABELS[mode])
    )).join('')) : '';
    const sizes = group('Size', `<p class="widgets-menu-note" data-widget-size-note hidden>Widths apply when the list is wide.</p>${
        WIDGET_WIDTH_STEPS.map(({ w, label }) => item(`data-widget-size="${w}" aria-checked="false"`, label)).join('')
    }${item('data-widget-size="reset"', 'Reset size', 'menuitem')}`);
    const power = framed ? `<span class="ui-status" data-tone="neutral" data-widget-status hidden>Stopped</span>
        <button type="button" class="btn btn-primary btn-sm" data-widget-power>Start</button>` : '';
    return `${power}
        <div class="skills-card-menu">
            <button type="button" class="skills-card-menu-trigger" aria-label="Widget options" aria-haspopup="menu" aria-expanded="false" data-widget-menu-trigger>⋮</button>
            <dialog class="skills-card-menu-dialog ui-popup" role="menu" aria-label="Widget options">${policy}${sizes}</dialog>
        </div>`;
}

const STATUS_TEXT = { starting: 'Starting…', running: 'Running', stopping: 'Stopping…' };

/**
 * Show a widget fault in the card's own status slot. Only the status span is
 * written: the lifecycle state is untouched, because the frame really is still
 * mounted and its Stop button must keep saying Stop. The status node persists
 * the fault for its keyed card until a real lifecycle transition clears it.
 */
export function setWidgetCardFault(card, text) {
    const status = card?.querySelector('[data-widget-status]');
    if (!status) return;
    status.hidden = false;
    status.dataset.tone = 'error';
    status.dataset.widgetFault = text;
    status.textContent = text;
}

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
        const fault = status.dataset.widgetFault;
        if (state === 'running' && fault) {
            status.hidden = false;
            status.dataset.tone = 'error';
            status.textContent = fault;
        } else {
            delete status.dataset.widgetFault;
            status.hidden = state === 'stopped';
            status.dataset.tone = state === 'running' ? 'ok' : 'neutral';
            status.textContent = state === 'running' && mode === 'retain'
                ? 'Keeps running'
                : (STATUS_TEXT[state] || 'Stopped');
        }
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

/**
 * Card-menu domain adapter over the shared keyboard/viewport menu: a launch
 * policy goes to `onSelectMode(key, mode)`, a width step (or Reset, `null`) to
 * `widths.setWidth(key, w)`; the menu opens with the card's current width
 * (`widths.widthOf(key)`) checked, and on the stacked column it says that
 * widths apply to the wide board only.
 */
export function bindWidgetCardMenus(list, onSelectMode, widths = null) {
    if (!list) return { close() {}, destroy() {} };
    let active = null;
    const close = () => active?.binding.close();
    const onClick = (event) => {
        const trigger = event.target.closest('[data-widget-menu-trigger]');
        if (!trigger) return;
        const wasOpen = active?.trigger === trigger;
        close();
        if (wasOpen) return;
        const card = trigger.closest('[data-widget-key]');
        const popover = trigger.closest('.skills-card-menu')?.querySelector('.skills-card-menu-dialog');
        const key = card?.dataset.widgetKey || '';
        if (!popover || !key) return;
        const width = widths?.widthOf(key) || 0;
        popover.querySelectorAll('[data-widget-size][role="menuitemradio"]').forEach((item) => {
            item.setAttribute('aria-checked', Number(item.dataset.widgetSize) === width ? 'true' : 'false');
        });
        popover.querySelector('[data-widget-size-note]')?.toggleAttribute('hidden', list.dataset.widgetLayout !== 'stack');
        // Capture the owning card before moving the popup outside clipped cards.
        const home = popover.parentNode;
        home.ownerDocument.body.append(popover);
        popover.show();
        trigger.setAttribute('aria-expanded', 'true');
        const onSelect = (selection) => {
            const item = selection.target.closest('[data-widget-start-mode], [data-widget-size]');
            if (!item) return;
            active?.binding.close({ restoreFocus: true });
            const size = item.dataset.widgetSize;
            if (!size) onSelectMode(key, item.dataset.widgetStartMode || '');
            else widths?.setWidth(key, size === 'reset' ? null : Number(size));
        };
        popover.addEventListener('click', onSelect);
        const binding = bindMenu(popover, {
            anchor: trigger,
            onClose() {
                popover.removeEventListener('click', onSelect);
                popover.close();
                if (home.isConnected) home.append(popover);
                else popover.remove();
                trigger.setAttribute('aria-expanded', 'false');
                active = null;
            },
        });
        active = { trigger, binding };
        (popover.querySelector('[aria-checked="true"]') || popover.querySelector('[role="menuitemradio"]'))?.focus({ preventScroll: true });
    };
    list.addEventListener('click', onClick);
    return {
        close,
        destroy() { close(); list.removeEventListener('click', onClick); },
    };
}
