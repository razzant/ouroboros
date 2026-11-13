// Editable suggestions shared by the product's model editors. The input is
// the draft authority; catalogs enrich its list and never assign a value.
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { bindPopoverPosition } from './ui_interactions.js';
import { bindModelProgramFooter } from './model_program.js';

const choosers = new WeakMap();

/** `meta` is a fact beside the name (a catalog price): shown apart, never part of the label or the search. */
export function modelOptions(items = []) {
    const seen = new Set();
    return items.map((item) => typeof item === 'string' ? { value: item, label: item }
        : { value: String(item.value ?? item.id ?? ''), label: String(item.label || item.name || item.value || item.id || ''),
            ...(item.custom ? {custom: true} : {}), ...(item.meta ? {meta: String(item.meta)} : {}) })
        .filter((item) => !seen.has(item.value) && seen.add(item.value));
}

export function matchingModelOptions(items, query) {
    const search = String(query || '').trim().toLocaleLowerCase();
    return modelOptions(items).filter(({ value, label, custom }) => !custom && (!search
        || value.toLocaleLowerCase().includes(search) || label.toLocaleLowerCase().includes(search)));
}

function optionsHtml(items, listId) {
    return modelOptions(items).map(({ value, label, custom, meta }, index) =>
        `<div class="ui-model-option${meta ? ' ui-model-option-priced' : ''}" role="option" id="${escapeHtml(listId)}-${index}"
            data-model-value="${escapeHtml(value)}"${custom ? ' data-model-custom-option' : ''} aria-selected="false">${meta
            ? `<span class="ui-model-option-name">${escapeHtml(label || value)}</span> <span class="ui-model-option-meta" data-i18n-skip>${escapeHtml(meta)}</span>`
            : escapeHtml(label || value)}</div>`).join('');
}

export function modelChooserHtml(attrs, value, listId, items = [], { placeholder = 'Choose a model', labelUnlisted = true, programs = [] } = {}) {
    const options = `<div id="${escapeHtml(listId)}" role="listbox">${optionsHtml(items, listId)}</div>`;
    const footer = '<div class="ui-model-footer" data-model-footer role="group" aria-label="Supplying program" aria-live="polite" aria-atomic="true" hidden></div>';
    const popup = `<div class="ui-popup ui-model-options" data-model-popup hidden>${options}${footer}</div>`;
    return `<span class="ui-model-chooser"><input class="ui-control" type="text" ${attrs}
        data-model-chooser data-model-programs="${escapeHtml(JSON.stringify(programs))}" role="combobox" aria-autocomplete="list" aria-expanded="false"
        aria-controls="${escapeHtml(listId)}" value="${escapeHtml(value)}"
        placeholder="${escapeHtml(placeholder)}" autocomplete="off" spellcheck="false">
        ${popup}${labelUnlisted ? '<span class="ui-field-help" data-model-custom role="status" hidden>Custom ID · Not in this list</span>' : ''}</span>`;
}

// The name is the name element's text alone: the price beside it never joins the label, the search or a repaint.
function readOptions(list) {
    return [...list.querySelectorAll('[data-model-value]')].map((node) => ({
        value: node.dataset.modelValue, label: (node.querySelector('.ui-model-option-name') || node).textContent,
        custom: node.hasAttribute('data-model-custom-option'), meta: node.querySelector('.ui-model-option-meta')?.textContent || '',
    }));
}

export function updateModelChooser(input, items, programs) {
    choosers.get(input)?.update(items, programs);
}

/** Bind once per real input; dispose before its owner replaces or removes it. */
export function bindModelChooser(input) {
    if (!input || choosers.has(input)) return () => {};
    const doc = input.ownerDocument;
    const list = doc.getElementById(input.getAttribute('aria-controls'));
    if (!list) return () => {};
    const popup = list.closest('[data-model-popup]') || list;
    const owner = popup.parentElement;
    const footer = owner.querySelector('[data-model-footer]');
    list.setAttribute('aria-label', input.getAttribute('aria-label') || 'Models');
    let items = readOptions(list);
    let visible = [];
    let active = -1;
    let composing = false;
    let opened = false;
    let position = null;
    let restoringFocus = false;
    const programs = bindModelProgramFooter(input, footer, (restoreFocus) => {
        if (opened && restoreFocus) focusInput();
        position?.reposition();
    });
    programs.update(JSON.parse(input.dataset.modelPrograms || '[]'));
    const disposers = [];
    const listen = (target, type, listener) => {
        target.addEventListener(type, listener);
        disposers.push(() => target.removeEventListener(type, listener));
    };

    function highlight(index) {
        active = index;
        [...list.querySelectorAll('[role="option"]')].forEach((node, ordinal) => {
            node.setAttribute('aria-selected', String(ordinal === active));
            if (ordinal === active) {
                input.setAttribute('aria-activedescendant', node.id);
                node.scrollIntoView({ block: 'nearest' });
            }
        });
        if (active < 0) input.removeAttribute('aria-activedescendant');
    }

    function customLabel() {
        const label = owner.querySelector('[data-model-custom]');
        if (label) label.hidden = !String(input.value || '').trim() || items.some(item => !item.custom && item.value === String(input.value || '').trim());
    }
    customLabel();
    function render() {
        customLabel();
        const selected = visible[active]?.value;
        visible = matchingModelOptions(items, input.value);
        list.innerHTML = optionsHtml(visible, list.id)
            || (owner.querySelector('[data-model-custom]') ? '<div class="ui-model-empty">Custom ID · Not in this list</div>' : '<div class="ui-model-empty">No matching suggestions. You can keep this model ID.</div>');
        highlight(selected === undefined ? -1 : visible.findIndex((item) => item.value === selected));
        position?.reposition();
    }

    function close() {
        opened = false;
        programs.close();
        popup.hidden = true;
        input.setAttribute('aria-expanded', 'false');
        input.removeAttribute('aria-activedescendant');
        active = -1;
        position?.destroy(); position = null;
        if (popup.parentElement !== owner) owner.appendChild(popup);
        doc.removeEventListener('pointerdown', outside);
        doc.defaultView.removeEventListener('blur', close);
    }

    function open() {
        if (input.disabled) return;
        const wasOpen = opened;
        opened = true;
        popup.hidden = false;
        const portal = input.closest('dialog[open]') || doc.body;
        if (popup.parentElement !== portal) portal.appendChild(popup);
        input.setAttribute('aria-expanded', 'true');
        render();
        if (!position) position = bindPopoverPosition(popup, { anchor: input });
        doc.addEventListener('pointerdown', outside);
        doc.defaultView.addEventListener('blur', close);
        if (!wasOpen) programs.open();
    }

    function outside(event) {
        if (event.target !== input && !popup.contains(event.target)) close();
    }

    function focusInput() {
        restoringFocus = true;
        input.focus({ preventScroll: true });
        restoringFocus = false;
    }

    function tabPastPopup(event) {
        // Resolve from the input, not the portalled footer at the end of body.
        const controls = [...doc.querySelectorAll('a[href], button, input, select, textarea, summary, [tabindex]')]
            .filter(n => !popup.contains(n) && n.tabIndex >= 0 && !n.matches(':disabled')
                && !n.closest('[hidden], [inert]') && n.getClientRects().length
                && doc.defaultView.getComputedStyle(n).visibility !== 'hidden');
        const next = controls[controls.indexOf(input) + 1];
        event.preventDefault();
        close();
        if (next) next.focus({ preventScroll: true });
        else focusInput();
    }

    function choose(value) {
        input.value = value;
        // Existing domain input handlers retain serialization and source pins.
        input.dispatchEvent(new doc.defaultView.Event('input', { bubbles: true }));
        input.dispatchEvent(new doc.defaultView.Event('change', { bubbles: true }));
        close();
        focusInput();
    }

    listen(input, 'focus', () => { if (!restoringFocus) open(); });
    listen(input, 'click', () => { if (!opened) open(); });
    listen(input, 'input', () => { if (!composing) open(); });
    listen(input, 'compositionstart', () => { composing = true; });
    listen(input, 'compositionend', () => { composing = false; open(); });
    listen(input, 'blur', event => { if (!popup.contains(event.relatedTarget)) close(); });
    listen(input, 'keydown', (event) => {
        if (composing || event.isComposing) return;
        if (event.key === 'Escape' && opened) {
            event.preventDefault(); event.stopPropagation(); close();
        } else if (event.key === 'Tab') {
            const action = footer?.querySelector('button:not(:disabled), a[href]');
            if (opened && !event.shiftKey && action) {
                event.preventDefault(); highlight(-1); action.focus({ preventScroll: true });
            } else close();
        }
        else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            if (!opened) open();
            const delta = event.key === 'ArrowDown' ? 1 : -1;
            highlight(visible.length ? (active < 0 ? (delta > 0 ? 0 : visible.length - 1)
                : (active + delta + visible.length) % visible.length) : -1);
        } else if (event.key === 'Enter' && opened && active >= 0) {
            event.preventDefault(); choose(visible[active].value);
        }
    });
    // Keep focus/caret in the editable input; a touch scroll is still native.
    listen(popup, 'mousedown', (event) => event.preventDefault());
    listen(popup, 'focusout', () => {
        queueMicrotask(() => {
            if (opened && doc.activeElement !== input && !popup.contains(doc.activeElement)) close();
        });
    });
    listen(popup, 'keydown', event => {
        if (event.key === 'Escape') {
            event.preventDefault(); event.stopPropagation(); close(); focusInput();
        } else if (event.key === 'Tab' && footer?.contains(event.target)) {
            const actions = [...footer.querySelectorAll('button:not(:disabled), a[href]')];
            const next = actions[actions.indexOf(event.target) + (event.shiftKey ? -1 : 1)];
            if (next) { event.preventDefault(); next.focus({preventScroll: true}); }
            else if (event.shiftKey) { event.preventDefault(); focusInput(); }
            else tabPastPopup(event);
        }
    });
    listen(list, 'click', (event) => {
        const option = event.target.closest('[data-model-value]');
        if (option && list.contains(option)) choose(option.dataset.modelValue);
    });
    choosers.set(input, { footer, update(next, producers) {
        if (producers !== undefined) {
            input.dataset.modelPrograms = JSON.stringify(producers);
            programs.update(producers);
        }
        items = modelOptions(next);
        customLabel();
        if (opened && !composing) render();
    } });
    return () => {
        close();
        for (const dispose of disposers) dispose();
        programs.dispose();
        choosers.delete(input);
    };
}

export function bindModelChoosers(container) {
    const disposers = [...(container?.querySelectorAll('[data-model-chooser]') || [])].map(bindModelChooser);
    return () => { for (const dispose of disposers) dispose(); };
}

/** Refresh only discovery-owned options; actual editable nodes stay mounted. */
export function updateModelChooserOptions(current, desired) {
    for (const input of current.querySelectorAll('[data-model-chooser]')) {
        const id = input.getAttribute('aria-controls');
        const list = [...desired.querySelectorAll('[role="listbox"]')].find((node) => node.id === id);
        if (list) {
            const next = [...desired.querySelectorAll('[data-model-chooser]')].find((node) => node.getAttribute('aria-controls') === id);
            updateModelChooser(input, readOptions(list), JSON.parse(next?.dataset.modelPrograms || '[]'));
        }
    }
}
