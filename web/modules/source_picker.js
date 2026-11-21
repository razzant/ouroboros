// Engine/model source descriptors can arrive after a menu opens. The owner keeps its select/value/change
// contract; the accessible face and the open, portalled list refer to exact source values throughout.
import { bindOptionPress, bindPopoverPosition } from './ui_interactions.js';

let nextId = 0;

function updateSourceRows(select, list, rows) {
    const doc = list.ownerDocument;
    // Reuse rows by source ID while preserving group order, including ungrouped inheritance choices.
    const used = new Set(), groups = new Map([...list.children].filter(node => node.hasAttribute('role')
        && node.getAttribute('role') === 'group').map(node => [node.getAttribute('aria-label'), node]));
    const place = (parent, node, index) => {
        if (parent.children[index] !== node) parent.insertBefore(node, parent.children[index] || null);
    };
    let groupIndex = 0;
    function optionRow(option, parent, index) {
        const disabled = option.disabled || option.parentElement.disabled;
        const key = disabled ? option : option.value;
        let row = rows.get(key);
        if (!row) {
            row = doc.createElement('div');
            row.className = 'ui-model-option';
            row.id = 'source-option-' + ++nextId;
            row.setAttribute('role', 'option');
            rows.set(key, row);
        }
        if (row.textContent !== option.label) row.textContent = option.label;
        row.setAttribute('aria-selected', String(option.selected));
        row.setAttribute('aria-disabled', String(Boolean(disabled)));
        row.dataset.sourceValue = option.value;
        place(parent, row, index);
        used.add(key);
    }
    for (const entry of select.children) {
        if (entry.tagName === 'OPTGROUP') {
            let group = groups.get(entry.label);
            if (!group) {
                group = doc.createElement('div');
                group.setAttribute('role', 'group');
                group.setAttribute('aria-label', entry.label);
            }
            place(list, group, groupIndex++);
            groups.delete(entry.label);
            [...entry.children].forEach((option, index) => optionRow(option, group, index));
        } else optionRow(entry, list, groupIndex++);
    }
    for (const [key, row] of rows) if (!used.has(key)) { row.remove(); rows.delete(key); }
    for (const group of groups.values()) group.remove();
}

function bindSourcePicker(select) {
    const doc = select.ownerDocument, win = doc.defaultView;
    if (!win?.MutationObserver) return () => {};
    const button = doc.createElement('button');
    button.type = 'button';
    button.className = 'ui-control ui-source-picker';
    button.dataset.sourcePicker = '';
    button.setAttribute('role', 'combobox');
    button.setAttribute('aria-haspopup', 'listbox');
    button.setAttribute('aria-expanded', 'false');
    const list = doc.createElement('div');
    list.id = 'source-options-' + ++nextId;
    list.className = 'ui-popup ui-source-options';
    list.setAttribute('role', 'listbox');

    const oldTabIndex = select.getAttribute('tabindex');
    const field = select.parentElement;
    field.classList.add('ui-source-field');
    select.classList.add('ui-source-backing');
    select.tabIndex = -1;
    select.setAttribute('aria-hidden', 'true');
    select.before(button);
    let keyboardActive = false;
    let position = null, active = null, unbindPress = null, prefix = '', typedAt = 0;
    const rows = new Map();
    const enabled = () => [...select.options].filter(option => !option.disabled && !option.parentElement.disabled);
    const focus = () => button.focus({ preventScroll: true });
    function highlight(value, reveal = true) {
        rows.get(active)?.removeAttribute('data-active');
        active = rows.has(value) ? value : null;
        const row = rows.get(active);
        if (row) {
            row.dataset.active = '';
            button.setAttribute('aria-activedescendant', row.id);
            if (reveal) {
                const bounds = row.getBoundingClientRect(), top = list.getBoundingClientRect().top + list.clientTop;
                const bottom = top + list.clientHeight;
                if (bounds.bottom > bottom) list.scrollTop += Math.ceil(bounds.bottom - bottom);
                else if (bounds.top < top) list.scrollTop -= Math.ceil(top - bounds.top);
            }
        } else button.removeAttribute('aria-activedescendant');
    }
    function redraw() {
        const label = select.getAttribute('aria-label') || 'Source';
        button.setAttribute('aria-label', label);
        list.setAttribute('aria-label', label);
        const name = select.selectedOptions[0]?.label || label;
        if (button.textContent !== name) button.textContent = name;
        button.disabled = select.disabled;
        button.hidden = select.hidden;
        if (!list.isConnected) return;
        if (button.disabled || button.hidden) { close(); return; }
        updateSourceRows(select, list, rows);
        position?.reposition();
        highlight(active, keyboardActive);
    }
    function outside(event) {
        if (event.target !== button && !list.contains(event.target)) close();
    }
    function close() {
        position?.destroy();
        position = null;
        unbindPress?.();
        unbindPress = null;
        doc.removeEventListener('pointerdown', outside, true);
        win.removeEventListener('blur', close);
        rows.get(active)?.removeAttribute('data-active');
        list.remove();
        button.setAttribute('aria-expanded', 'false');
        button.removeAttribute('aria-controls');
        button.removeAttribute('aria-activedescendant');
        active = null; prefix = ''; keyboardActive = false;
    }
    function open() {
        if (list.isConnected) return;
        focus();
        (button.closest('dialog[open]') || doc.body).appendChild(list);
        button.setAttribute('aria-expanded', 'true');
        button.setAttribute('aria-controls', list.id);
        redraw();
        position = bindPopoverPosition(list, { anchor: button, field: button.closest('.ui-field') || button });
        highlight(select.value);
        unbindPress = bindOptionPress(list, '[data-source-value]', row => choose(row.dataset.sourceValue));
        doc.addEventListener('pointerdown', outside, true);
        win.addEventListener('blur', close);
    }
    function choose(value) {
        if (!enabled().some(option => option.value === value)) return;
        close();
        if (select.value === value) return;
        select.value = value;
        redraw();
        select.dispatchEvent(new win.Event('input', { bubbles: true }));
        select.dispatchEvent(new win.Event('change', { bubbles: true }));
    }
    button.addEventListener('click', () => list.isConnected ? close() : open());
    button.addEventListener('blur', close);
    button.addEventListener('keydown', event => {
        if (event.isComposing || event.ctrlKey || event.metaKey) return;
        const key = event.key;
        if (key === 'Tab') { close(); return; }
        if (key === 'Escape' && list.isConnected) {
            event.preventDefault(); event.stopPropagation(); close(); return;
        }
        if (key === 'Enter' || key === ' ') {
            event.preventDefault();
            if (!list.isConnected) open();
            else if (active !== null) choose(active);
            return;
        }
        const values = enabled();
        if (!values.length) return;
        if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(key)) {
            event.preventDefault();
            open();
            keyboardActive = true;
            const index = values.findIndex(option => option.value === active);
            let at = key === 'Home' ? 0 : values.length - 1;
            if (key === 'ArrowDown') at = Math.min(index + 1, values.length - 1);
            if (key === 'ArrowUp') at = index < 0 ? values.length - 1 : Math.max(0, index - 1);
            highlight(values[at].value);
        } else if (key.length === 1 && !event.altKey) {
            open();
            keyboardActive = true;
            const now = event.timeStamp, char = key.toLocaleLowerCase();
            prefix = now - typedAt < 650 && prefix !== char ? prefix + char : char;
            typedAt = now;
            const start = values.findIndex(option => option.value === active);
            const match = [...values.slice(start + 1), ...values.slice(0, start + 1)]
                .find(option => option.label.toLocaleLowerCase().startsWith(prefix));
            if (match) highlight(match.value);
        }
    });
    list.addEventListener('mousedown', event => event.preventDefault()); // keep keyboard/AT focus; touch scroll remains native
    list.addEventListener('wheel', () => { keyboardActive = false; }, { passive: true });
    list.addEventListener('pointerdown', () => { keyboardActive = false; });
    const observe = new win.MutationObserver(redraw);
    observe.observe(select, { childList: true, subtree: true, characterData: true, attributes: true,
        attributeFilter: ['label', 'selected', 'disabled', 'hidden', 'aria-label'] });
    select.addEventListener('change', redraw);
    select.addEventListener('focus', focus);
    redraw();
    return () => {
        close(); observe.disconnect();
        select.removeEventListener('focus', focus);
        select.removeEventListener('change', redraw);
        button.remove();
        field.classList.remove('ui-source-field');
        select.classList.remove('ui-source-backing');
        select.removeAttribute('aria-hidden');
        if (oldTabIndex === null) select.removeAttribute('tabindex');
        else select.setAttribute('tabindex', oldTabIndex);
    };
}

/** Bind the source selects owned by this form; the owner disposes before replacing its rows. */
export function bindSourcePickers(container) {
    const disposers = [...(container?.querySelectorAll('select[data-model-role-source], select[data-subagent-field="route"]') || [])]
        .map(bindSourcePicker);
    return () => disposers.forEach(dispose => dispose());
}
