import { escapeHtmlAttr as escapeHtml } from './utils.js';

function classAttr(parts) {
    return parts.filter(Boolean).join(' ');
}

export function renderMobileNavToggle() {
    return `
        <button class="mobile-nav-toggle" type="button" data-mobile-nav-toggle aria-label="Open navigation" aria-controls="primary-sidebar" aria-expanded="false">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h16"/></svg>
        </button>
    `;
}

export function renderPageHeader({
    title,
    icon = '',
    description = '',
    leadingHtml,
    toolbarHtml = '',
    trailingHtml = '',
    actionsHtml = '',
    tabsHtml = '',
    variant = '',
    className = '',
    showMobileNav = true,
} = {}) {
    const variantClass = variant ? `app-page-header-${escapeHtml(variant)}` : '';
    const iconHtml = icon ? `<span class="app-page-icon" aria-hidden="true">${icon}</span>` : '';
    const leading = leadingHtml !== undefined
        ? leadingHtml
        : (showMobileNav ? renderMobileNavToggle() : '');
    const descriptionHtml = description
        ? `<p class="app-page-description">${escapeHtml(description)}</p>`
        : '';
    const toolbar = (toolbarHtml || actionsHtml)
        ? `<div class="app-page-toolbar app-page-actions">${toolbarHtml || actionsHtml}</div>`
        : '';
    const trailing = trailingHtml
        ? `<div class="app-page-trailing">${trailingHtml}</div>`
        : '';
    const tabs = tabsHtml
        ? `<div class="app-page-tabs">${tabsHtml}</div>`
        : '';
    return `
        <div class="${classAttr(['page-header', 'app-page-header', variantClass, className])}">
            <div class="app-page-leading">${leading}</div>
            <div class="app-page-title-block">
                <div class="app-page-title-row">
                    ${iconHtml}
                    <h2 class="app-page-title">${escapeHtml(title)}</h2>
                </div>
                ${descriptionHtml}
            </div>
            ${toolbar}
            ${trailing}
            ${tabs}
        </div>
    `;
}

// SSOT for a segmented single-select control (the settings effort/mode toggles):
// one generator owns the markup so every group renders identically — the sibling
// of renderTabStrip/.app-tab, but for in-card segmented choices (.ui-segment).
// Buttons keep the data-effort-* hooks the controls layer binds on, plus the
// legacy .settings-effort-* classes that carry the settings layout and accents.
// The columns come from the choices actually rendered: the generator writes their
// number once as `--segment-count`, a fact of this rendering rather than a
// per-control setting, and settings.css derives equal columns from it. `modifier`
// is a bare boolean data-attribute (e.g. 'data-enforcement-group') for a per-group
// override such as accent colors.
export function renderSegmentedField({
    target,
    options = [],
    modifier = '',
    title = '',
} = {}) {
    const tgt = String(target || '').trim();
    if (!tgt) {
        throw new Error('renderSegmentedField requires target');
    }
    const buttons = options.map((opt) => {
        const value = String(opt.value ?? '');
        return `<button type="button" class="ui-segment settings-effort-btn" data-effort-value="${escapeHtml(value)}">${escapeHtml(opt.label ?? value)}</button>`;
    }).join('');
    const mod = String(modifier || '').trim();
    const modAttr = mod ? ` ${mod}` : '';
    const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
    return `
        <div class="ui-segment-group settings-effort-group" data-effort-group${modAttr} data-effort-target="${escapeHtml(tgt)}" style="--segment-count: ${options.length}"${titleAttr}>
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
        <div class="${classAttr(['app-tab-strip', stripClass])}" role="tablist" aria-label="${escapeHtml(ariaLabel)}">
            ${buttons}
        </div>
    `;
}

/**
 * Bind one rendered strip. onChange owns domain loading/panel visibility.
 * select(value) synchronizes an external navigation without calling onChange;
 * user activation calls it once, only when the selection actually changes.
 */
export function bindTabStrip(strip, { dataAttr, activeClass = 'active', onChange } = {}) {
    if (!dataAttr) throw new Error('bindTabStrip requires dataAttr');
    let disposed = false;
    const tabs = () => Array.from(strip.querySelectorAll(`[role="tab"][${dataAttr}]`))
        .filter((tab) => tab.closest('[role="tablist"]') === strip);
    const enabled = (tab) => !tab.disabled && tab.getAttribute('aria-disabled') !== 'true'
        && !tab.closest('[hidden], [inert]');
    let selected = null;
    function revealSelected() {
        if (disposed || !strip.clientWidth) return;
        const tab = tabs().find((item) => item.getAttribute(dataAttr) === selected);
        if (!tab) return;
        const bounds = strip.getBoundingClientRect();
        const rect = tab.getBoundingClientRect();
        const left = bounds.left + strip.clientLeft;
        const right = left + strip.clientWidth;
        // Scroll only this strip. scrollIntoView can also move the page or its
        // scroll panel when restoring an inactive page's saved selection.
        if (rect.left < left) strip.scrollLeft += rect.left - left;
        else if (rect.right > right) strip.scrollLeft += rect.right - right;
    }
    function select(value, { focus = false } = {}) {
        if (disposed) return false;
        const all = tabs();
        const next = all.find((tab) => tab.getAttribute(dataAttr) === String(value) && enabled(tab));
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
    const initial = tabs().find((tab) => enabled(tab) && tab.getAttribute('aria-selected') === 'true')
        || tabs().find(enabled);
    if (initial) select(initial.getAttribute(dataAttr));
    return {
        select,
        destroy() {
            disposed = true;
            strip.removeEventListener('click', onClick);
            strip.removeEventListener('keydown', onKey);
            observer?.disconnect();
        },
    };
}
