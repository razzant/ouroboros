// Keyed repaint of the account families: `harness_accounts.js` paints them and `account_resources.js`
// keeps the list's reading place. Each family keeps its mounted login and maintenance views; each
// account keeps the controls the reader is using.
const actions = new WeakMap();

// Keep native activation and focus on the retained control, but do not let a
// poll turn a press into a different command. The revision also fences A→B→A.
export function bindAccountAction(control, command, invoke, hitArea = control) {
    let action = actions.get(control);
    if (!action) {
        action = { revision: 0 };
        actions.set(control, action);
        let pointer, key;
        hitArea.addEventListener('pointerdown', event => {
            if (event.isPrimary && event.button === 0) pointer = action.revision;
        });
        hitArea.addEventListener('pointercancel', event => {
            if (event.isPrimary) pointer = undefined;
        });
        control.addEventListener('keydown', event => {
            if (event.key !== ' ' && event.key !== 'Enter') return;
            if (!event.repeat) key = action.revision;
            else if (event.key === 'Enter') event.preventDefault();
        });
        control.addEventListener('blur', () => { key = undefined; });
        // Capture on the label as well as the input: cancelling the label's
        // original click prevents the browser from forwarding it to the switch.
        hitArea.addEventListener('click', event => {
            const observed = event.detail > 0 ? pointer : key;
            pointer = key = undefined;
            if (observed !== undefined && observed !== action.revision) event.preventDefault();
        }, true);
        control.onclick = event => {
            if (!event.defaultPrevented) action.invoke();
        };
    }
    if (action.command !== command) action.revision += 1;
    action.command = command;
    action.invoke = invoke;
}

function updateList(list, incoming, key, stationary, update) {
    const existing = new Map(Array.from(list.children, node => [node.dataset[key], node]));
    const wanted = Array.from(incoming.children);
    const keys = new Set(wanted.map(node => node.dataset[key]));
    for (const [id, node] of existing) if (!keys.has(id)) node.remove();
    let previous = null;
    for (const next of wanted) {
        const current = existing.get(next.dataset[key]);
        if (current) update(current, next);
        const node = current || next;
        const before = previous ? previous.nextElementSibling : list.firstElementChild;
        // Even an atomic move can cancel a native press in WebKit. Put its siblings
        // in order around the retained control without moving its own branch.
        if (node !== stationary && node !== before) {
            if (current && list.moveBefore) list.moveBefore(node, before);
            else list.insertBefore(node, before);
        }
        previous = node;
    }
}

function updatePart(current, next, selector) {
    const part = current.querySelector(selector);
    const wanted = next.querySelector(selector);
    if (part.innerHTML !== wanted.innerHTML) part.innerHTML = wanted.innerHTML;
}

function updateAttributes(current, next, names) {
    for (const name of names) {
        const value = next.getAttribute(name);
        if (current.getAttribute(name) === value) continue;
        if (value === null) current.removeAttribute(name);
        else current.setAttribute(name, value);
    }
}

function updateRow(row, next) {
    updateAttributes(row, next, ['class', 'data-kind', 'data-tone']);
    for (const selector of ['.harness-account-mark', '.harness-account-main', '.harness-account-meta']) {
        updatePart(row, next, selector);
    }
    const toggle = row.querySelector('[data-harness-toggle]');
    const wanted = next.querySelector('[data-harness-toggle]');
    updateAttributes(toggle, wanted, ['data-enabled', 'aria-label', 'aria-checked', 'aria-busy', 'disabled', 'checked']);
    toggle.checked = wanted.checked;
    const login = row.querySelector('[data-harness-login]');
    const nextLogin = next.querySelector('[data-harness-login]');
    updateAttributes(login, nextLogin, ['data-login-command', 'data-rare']);
    if (login.textContent !== nextLogin.textContent) login.textContent = nextLogin.textContent;
}

export function paintAccountFamilies(host, markup, control) {
    if (!host.firstElementChild) {
        host.innerHTML = markup;
        return;
    }
    const incoming = host.ownerDocument.createElement('div');
    incoming.innerHTML = markup;
    updateList(host, incoming, 'family', control?.closest('[data-family]'), (family, next) => {
        const card = family.querySelector('.agent-family-accounts');
        card.hidden = next.querySelector('.agent-family-accounts').hidden;
        const add = card.querySelector('[data-family-add]');
        const nextAdd = next.querySelector('[data-family-add]');
        if (!nextAdd) add?.remove();
        else if (!add) card.querySelector('.agent-family-head').append(nextAdd);
        else {
            updateAttributes(add, nextAdd, ['data-family-command']);
            if (add.textContent !== nextAdd.textContent) add.textContent = nextAdd.textContent;
        }
        updatePart(card, next, '.agent-family-id');
        updateList(card.querySelector('.agent-family-rows'), next.querySelector('.agent-family-rows'), 'profile',
            control?.closest('[data-harness][data-profile]'), updateRow);
    });
}
