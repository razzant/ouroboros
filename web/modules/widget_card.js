/* Widgets card chrome: for framed (module / route-iframe) cards the effective
   launch policy (and whether it keeps the card running while Widgets is
   hidden), the card's ONE primary control (Start / Stop) and the facade a
   stopped card shows in place of its frame; for every card the card menu —
   Move, the launch policy of a framed card, the owner's width steps (column
   spans) and heights (web/modules/widget_size.js) and Remove on a card kept
   from the chat; and the line under a card that says what its owner height
   keeps out of view (Show all / Back). widgets.js owns the registry and decides
   WHEN a card mounts or stops; this module only renders and reads the controls.
   Declarative cards are host-drawn and get only the menu's arrangement. */

import { PAGE_ICONS } from './page_icons.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { widgetKey } from './widget_list.js';
import { frameHeight, setFrameHeight } from './widget_module.js';
import { heightName, WIDGET_HEIGHT_PRESETS, WIDGET_WIDTH_STEPS } from './widget_size.js';
import { bindMenu } from './ui_interactions.js';

// The card menu's moves: where each one sends the card in the owner's order.
export const WIDGET_MOVES = Object.freeze([
    { where: 'start', label: 'Move to start' },
    { where: 'earlier', label: 'Move earlier' },
    { where: 'later', label: 'Move later' },
    { where: 'end', label: 'Move to end' },
]);

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

// Head controls: a framed card's status (dot + text) and its power button
// (Start; while it runs, Stop, which style.css leaves to the menu), then the
// card menu on the Skills card menu primitive (`.skills-card-menu` +
// `<dialog role="menu">`): Move, a framed card's launch policy, a skill's Shown in
// chat (its placed controls) and every card's width steps and heights, each a
// group, then a running framed card's Stop (rare, so never the board's loudest
// word) with the cause of a fault. The checked policy is set by
// `syncWidgetCardControls` once the page knows the owner's preferences; the
// checked width and height and the moves the card's place allows when the menu
// opens (`bindWidgetCardMenus`). The head draws the two parts apart
// (`renderWidgetCardParts`): the lifecycle controls may drop under the title's
// row, the menu stays with the move handle in one pair that never splits.
const menuItem = (attrs, label, role = 'menuitemradio') => (
    `<button type="button" role="${role}" class="skills-menu-item widgets-menu-item" ${attrs}><span class="widgets-menu-check" aria-hidden="true">${role === 'menuitemradio' || role === 'menuitemcheckbox' ? '✓' : ''}</span>${escapeHtml(label)}</button>`
);
const menuGroup = (label, items, attrs = '') => `<div role="group" aria-label="${label}"${attrs ? ` ${attrs}` : ''}><div class="widgets-menu-heading" aria-hidden="true">${label}</div>${items}</div>`;

/** The card menu's "Shown in chat" group: one checkbox per placed control of the skill, checked while it shows. */
export function renderWidgetContributionGroup(rows = []) {
    if (!rows.length) return '';
    return menuGroup('Shown in chat', rows.map((row) => {
        const hidden = row.hidden === true;
        const label = row.label || 'Skill control';
        const named = row.place ? `${label}, ${row.place.toLowerCase()}` : label;
        return menuItem(`data-widget-contribution-key="${escapeHtml(row.key || '')}" data-widget-contribution-hidden="${hidden ? 'true' : 'false'}" aria-checked="${hidden ? 'false' : 'true'}" aria-label="${escapeHtml(named)}" title="${escapeHtml(row.place || '')}"`,
            label, 'menuitemcheckbox');
    }).join(''), 'data-widget-contributions');
}

export function renderWidgetCardParts(tab, { contributionVisibility = [] } = {}) {
    const item = menuItem, group = menuGroup;
    const framed = isFramedWidget(tab);
    const moves = group('Move', WIDGET_MOVES.map(({ where, label }) => item(`data-widget-move="${where}"`, label, 'menuitem')).join(''));
    const policy = framed ? group('Launch policy', WIDGET_START_MODES.map((mode) => (
        item(`data-widget-start-mode="${mode}" aria-checked="false"`, WIDGET_START_MODE_LABELS[mode])
    )).join('')) : '';
    // Resize: where the primary input is a finger, the one card whose foot and grip show (widget_reorder.js).
    const sizes = group('Size', `<p class="widgets-menu-note" data-widget-size-note hidden>Widths apply when the list is wide.</p>${
        item('data-widget-adjust hidden', 'Resize', 'menuitem')}${
        WIDGET_WIDTH_STEPS.map(({ w, label }) => item(`data-widget-size="${w}" aria-checked="false"`, label)).join('')
    }${item('data-widget-size="reset"', 'Reset size', 'menuitem')}`);
    const composer = renderWidgetContributionGroup(contributionVisibility);
    // Natural height, the menu's heights, and the card's own height when the owner dragged one the menu does not list.
    const heights = group('Height', `${item('data-widget-height="0" aria-checked="false"', 'Natural height')}${
        WIDGET_HEIGHT_PRESETS.map((h) => item(`data-widget-height="${h}" aria-checked="false"`, heightName(h))).join('')
    }<button type="button" role="menuitemradio" class="skills-menu-item widgets-menu-item" data-widget-height="" data-widget-height-own aria-checked="false" hidden><span class="widgets-menu-check" aria-hidden="true">✓</span><span data-widget-height-label></span></button>`);
    // A card kept from the chat (`pinned_card`) is the owner's data: Remove takes it off the board.
    const remove = tab?.pinned_card ? `<div role="group" aria-label="Card">${item('data-widget-unpin', 'Remove', 'menuitem')}</div>` : '';
    const power = framed ? `<span class="ui-status" data-tone="neutral" data-widget-status hidden>Stopped</span>
        <button type="button" class="btn btn-primary btn-sm" data-widget-power>Start</button>` : '';
    const stop = framed ? item('data-widget-stop hidden', 'Stop', 'menuitem') : '';
    const menu = `<div class="skills-card-menu">
            <button type="button" class="btn btn-ghost btn-sm skills-card-menu-trigger" aria-label="Widget options" aria-haspopup="menu" aria-expanded="false" data-widget-menu-trigger>⋮</button>
            <dialog class="skills-card-menu-dialog ui-popup" role="menu" aria-label="Widget options">${moves}${policy}${composer}${sizes}${heights}${framed ? `<div role="group" aria-label="Run">${stop}<p class="widgets-menu-note" data-widget-fault-details hidden></p></div>` : ''}${remove}</dialog>
        </div>`;
    return { run: power, menu };
}

const STATUS_TEXT = { starting: 'Starting…', running: 'Running', stopping: 'Stopping…' };

/**
 * Show a widget fault in the card's own status slot, its raw cause (`details`)
 * in the card menu. Only the status span is written: the lifecycle state is
 * untouched, because a frame with a fault may still be mounted (Stop stays in
 * the menu). A mount that `ended` by itself keeps the fault while its card
 * settles stopped, where Start again restarts it. The status node persists the
 * fault for its keyed card until a real lifecycle transition clears it.
 */
export function setWidgetCardFault(card, text, { details = '', ended = false } = {}) {
    const status = card?.querySelector('[data-widget-status]');
    if (!status) return;
    status.hidden = false;
    status.dataset.tone = 'error';
    status.dataset.widgetFault = text;
    status.textContent = text;
    if (details) status.dataset.widgetFaultDetails = details;
    else delete status.dataset.widgetFaultDetails;
    if (ended) status.dataset.widgetEnded = 'true';
    else delete status.dataset.widgetEnded;
    const detail = card.querySelector('[data-widget-fault-details]');
    if (detail) { detail.textContent = details; detail.hidden = !details; }
}

/**
 * A stopped card's one line about why it did not start by itself (the start ledger, holds.js), in its status slot
 * until the next lifecycle transition rewrites it. An empty `text` leaves the slot as the transition left it.
 */
export function setWidgetCardNote(card, text) {
    const status = card?.querySelector('[data-widget-status]');
    if (!status || !text) return;
    status.hidden = false;
    status.dataset.tone = 'neutral';
    status.textContent = text;
}

/**
 * Keep the head controls truthful. `state` is one of stopped | starting |
 * running | stopping — expressed through the head's power button (Start, Stop
 * while the frame runs or stops, Start again after a mount that ended by
 * itself; `disabled` while a transition is in flight; crimson only for a plain
 * Start), the menu's Stop (only while it runs) and the status sentence; no state
 * machine object. A running card's power is `data-running`: style.css keeps Stop
 * in the card menu and hides it there (the board's fork: drop that one rule).
 * A running card kept alive across pages (`mode === 'retain'`) says so: the
 * frame really keeps running while Widgets is hidden (the browser, not the
 * host, may pause its animation frames meanwhile — see CREATING_SKILLS).
 */
export function syncWidgetCardControls(card, state, mode = '') {
    const power = card?.querySelector('[data-widget-power]');
    if (!power) return;
    const status = card.querySelector('[data-widget-status]');
    // A fault is kept while its frame still runs, and while a mount that ended by itself settles stopped.
    const fault = status?.dataset.widgetFault;
    const kept = Boolean(fault) && (state === 'running' || (status.dataset.widgetEnded === 'true' && state !== 'starting'));
    const again = kept && state === 'stopped';
    const running = state === 'running' || state === 'stopping';
    power.textContent = running ? 'Stop' : (again ? 'Start again' : 'Start');
    power.classList?.toggle('btn-primary', !running && !again);
    power.classList?.toggle('btn-default', running || again);
    power.toggleAttribute?.('data-running', running);
    power.disabled = state === 'starting' || state === 'stopping';
    const stop = card.querySelector('[data-widget-stop]');
    if (stop) stop.hidden = state !== 'running';
    if (status) {
        if (kept) {
            status.hidden = false;
            status.dataset.tone = 'error';
            status.textContent = fault;
        } else {
            delete status.dataset.widgetFault;
            delete status.dataset.widgetFaultDetails;
            delete status.dataset.widgetEnded;
            status.hidden = state === 'stopped';
            status.dataset.tone = state === 'running' ? 'ok' : 'neutral';
            status.textContent = state === 'running' && mode === 'retain'
                ? 'Keeps running'
                : (STATUS_TEXT[state] || 'Stopped');
        }
    }
    const detail = card.querySelector('[data-widget-fault-details]');
    if (detail) { detail.textContent = (kept && status.dataset.widgetFaultDetails) || ''; detail.hidden = !detail.textContent; }
    if (!mode) return;
    card.querySelectorAll('[data-widget-start-mode]').forEach((item) => {
        item.setAttribute('aria-checked', item.dataset.widgetStartMode === mode ? 'true' : 'false');
    });
}

// The line under a card that an owner height keeps from showing everything: how much is out of view, and
// Show all; once shown, the way Back. A frame not running has nothing to show.
export function syncWidgetCardMore(card, { expandable, expanded, hidden, h }) {
    const more = card.querySelector('[data-widget-more]');
    const button = more?.querySelector('[data-widget-expand]');
    if (!button) return;
    const show = Boolean(h) && (expanded || expandable);
    if (more.hidden === show) more.hidden = !show;
    if (card.hasAttribute('data-widget-more-shown') !== show) card.toggleAttribute('data-widget-more-shown', show);
    const text = expanded ? `Back to ${h} px` : (hidden > 0 ? `Show all · ${hidden.toLocaleString('en-US')} px more` : 'Show all');
    if (show && button.textContent !== text) button.textContent = text;
    if (button.getAttribute('aria-expanded') !== String(Boolean(expanded))) button.setAttribute('aria-expanded', String(Boolean(expanded)));
}

// A host-drawn card (declarative, kept from the chat) scrolls in its body under the owner's height `h` (0: none),
// unless shown all (`expanded`), and its line says what that height keeps out of view.
export function fitWidgetCardBody(card, h, expanded) {
    const body = card.querySelector('[data-widget-mount]');
    if (!body) return;
    const bounded = h > 0 && !expanded;
    if (bounded) body.style.setProperty('--widget-body-height', `${h}px`);
    else body.style.removeProperty('--widget-body-height');
    body.toggleAttribute('data-widget-bounded', bounded);
    const hidden = bounded ? Math.max(0, body.scrollHeight - body.clientHeight) : 0;
    syncWidgetCardMore(card, { expandable: hidden > 1 || (expanded && body.scrollHeight > h + 1), expanded: expanded && h > 0, hidden, h });
}

/**
 * The stopped card's body: icon + title as tall as its frame will be (`height`,
 * the board's: by default the declared height, or the 320 px loading height of a
 * module that sizes itself after Start). Idempotent: an existing facade keeps its
 * content and only takes a given `height`, and a frame still in the mount (a stop
 * awaiting its acknowledgement keeps its iframe there) is left alone.
 */
export function renderWidgetFacade(mount, tab, height = 0) {
    const existing = mount?.querySelector('[data-widget-facade], iframe');
    if (!mount || existing) {
        if (height && existing?.matches?.('[data-widget-facade]')) setFrameHeight(existing, height);
        return;
    }
    const title = tab.title || tab.tab_id || tab.skill;
    const icon = String(tab.icon || '').trim();
    const glyph = !icon || ICON_NAME.test(icon) ? PAGE_ICONS.widgets : escapeHtml(icon);
    mount.innerHTML = `<div class="widgets-facade" data-widget-facade>
        <span class="widgets-facade-icon" aria-hidden="true">${glyph}</span>
        <strong class="widgets-facade-title">${escapeHtml(title)}</strong>
    </div>`;
    setFrameHeight(mount.firstElementChild, height || frameHeight(tab.render || {}));
}

/**
 * Card-menu domain adapter over the shared keyboard/viewport menu: a launch
 * policy goes to `onSelectMode(key, mode)`, a width step to `widths.setWidth(key, w)`, Reset size to
 * `widths.resetSize(key)`, a height (0: Natural height) to `widths.setHeight(key, h)`, a move to
 * `onMove(key, where)`, a kept card's Remove to `onRemove(key)`, Resize (offered under a finger: `widths.canAdjust()`)
 * to `widths.adjust(key)`; the menu opens with the card's current width
 * (`widths.widthOf(key)`) and height (`widths.heightOf(key)`) checked, only the moves its place
 * (`placeOf(key)`: `{index, count}`) allows, and on the stacked column it says that widths apply to the wide board only.
 */
export function bindWidgetCardMenus(list, onSelectMode, widths = null, onRemove = () => {}, onStop = () => {},
    onToggleContribution = () => {}, { placeOf = () => null, onMove = () => {} } = {}) {
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
        popover.querySelector('[data-widget-adjust]')?.toggleAttribute('hidden', !widths?.canAdjust?.());
        const height = widths?.heightOf?.(key) || 0;
        const own = popover.querySelector('[data-widget-height-own]');
        const listed = height === 0 || WIDGET_HEIGHT_PRESETS.includes(height);
        if (own) {
            own.hidden = listed;
            own.dataset.widgetHeight = listed ? '' : String(height);
            own.querySelector('[data-widget-height-label]').textContent = listed ? '' : heightName(height);
        }
        popover.querySelectorAll('[data-widget-height]').forEach((item) => {
            item.setAttribute('aria-checked', item.dataset.widgetHeight !== '' && Number(item.dataset.widgetHeight) === height ? 'true' : 'false');
        });
        const place = placeOf(key);
        popover.querySelectorAll('[data-widget-move]').forEach((item) => {
            const earlier = ['start', 'earlier'].includes(item.dataset.widgetMove);
            item.hidden = !place || (earlier ? place.index <= 0 : place.index >= place.count - 1);
        });
        popover.querySelector('[role="group"][aria-label="Move"]')?.toggleAttribute('hidden', !place || place.count < 2);
        // Capture the owning card before moving the popup outside clipped cards.
        const home = popover.parentNode;
        home.ownerDocument.body.append(popover);
        popover.show();
        trigger.setAttribute('aria-expanded', 'true');
        const onSelect = (selection) => {
            const item = selection.target.closest('[data-widget-start-mode], [data-widget-size], [data-widget-height], [data-widget-move], [data-widget-unpin], [data-widget-stop], [data-widget-contribution-key], [data-widget-adjust]');
            if (!item) return;
            active?.binding.close({ restoreFocus: true });
            const size = item.dataset.widgetSize;
            if (item.hasAttribute('data-widget-adjust')) widths?.adjust?.(key);
            else if (item.hasAttribute('data-widget-stop')) onStop(key);
            else if (item.hasAttribute('data-widget-unpin')) onRemove(key);
            else if (item.hasAttribute('data-widget-contribution-key')) onToggleContribution(item.dataset.widgetContributionKey, item.dataset.widgetContributionHidden !== 'true');
            else if (item.dataset.widgetMove) onMove(key, item.dataset.widgetMove);
            else if (item.hasAttribute('data-widget-height')) widths?.setHeight?.(key, Number(item.dataset.widgetHeight) || 0);
            else if (!size) onSelectMode(key, item.dataset.widgetStartMode || '');
            else if (size === 'reset') widths?.resetSize?.(key);
            else widths?.setWidth(key, Number(size));
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
