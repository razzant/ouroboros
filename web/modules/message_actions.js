// The actions a saved chat row offers beside its own doors (docs/DESIGN.md "Replies"): one registry that both the
// bubble's control row (chat_reply.js builds it: Reply, then these, then Copy in the corner) and the row menu
// (chat_row_menu.js: after the row's own doors) read, so an action is drawn the same way in both and nowhere else.
// Host features register here (Read aloud) and skills' `message.action` contributions arrive through
// `bridgeContributedActions`. Never on hover only: a bubble shows an action as an icon control with its accessible
// name, the row menu in words.
import { fmt } from './i18n.js';
import { controlIcon } from './page_icons.js';
import { showToast } from './toast.js';
import { runContributionAction } from './contributions.js';

const ROLE_NAMES = ['user', 'assistant', 'system'];
const defaultNotice = (text, tone = 'info') => showToast(text, tone);

/**
 * An action is `{key, label, icon?, title?, order?, surface?, applies?(row), run(row, ctx), state?(row)}`. `row` is
 * `messageRow(bubble)`; `ctx` is `{anchor, notice(text, tone)}`; `surface: 'menu'` keeps an action to the row menu. `state(row)` may answer `{label, icon, title,
 * pressed}` for an action whose face follows its own state (Read aloud says Stop while it speaks): its owner calls
 * `changed(key)` when that state moves and every drawn control redraws. Lower `order` comes first, then registration.
 */
export function createMessageActions() {
    const actions = new Map();
    const listeners = new Set();
    let sequence = 0;
    let holding = 0;
    let pending = false;

    function changed(key = '') {
        if (holding) { pending = true; return; }
        for (const listener of [...listeners]) {
            try { listener(key); } catch (error) { console.error('message action listener failed', error); }
        }
    }

    const applies = (action, row) => {
        try { return Boolean(action.applies(row)); } catch { return false; }
    };

    return {
        /** Add one action; returns its unregister. A second action with the same key replaces the first. */
        register(action) {
            const key = String(action?.key || '');
            if (!key || typeof action.run !== 'function') throw new TypeError('A message action needs a key and run(row, ctx).');
            const entry = { applies: () => true, order: 0, ...action, key, seq: (sequence += 1) };
            actions.set(key, entry);
            changed(key);
            return () => {
                if (actions.get(key) !== entry) return;
                actions.delete(key);
                changed(key);
            };
        },
        /** The actions `row` offers, in order. */
        actionsFor(row) {
            return [...actions.values()].filter((action) => applies(action, row))
                .sort((a, b) => a.order - b.order || a.seq - b.seq);
        },
        onChange(listener) {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
        changed,
        /** Run `fn` with change notices held; one notice follows when anything changed. */
        batch(fn) {
            holding += 1;
            try { return fn(); } finally {
                holding -= 1;
                if (!holding && pending) { pending = false; changed(); }
            }
        },
    };
}

let shared = null;

/** The page's one registry. */
export function getMessageActions() {
    if (!shared) shared = createMessageActions();
    return shared;
}

/**
 * What an action sees of a saved bubble: its address, its author's role, its task, its plain words and its content
 * node (read it, never change it). Null for a bubble the host has not saved (no row address).
 */
export function messageRow(bubble) {
    const address = bubble?.dataset?.rowAddress || '';
    if (!address) return null;
    const content = bubble.querySelector(':scope > .message');
    return {
        address,
        role: ROLE_NAMES.find((name) => bubble.classList.contains(name)) || '',
        taskId: bubble.dataset.taskId || '',
        text: String(content?.textContent || bubble.dataset.rowWords || '').trim(),  // a living card: its fallback words
        content,
    };
}

/** The face an action shows on `row` now: its own words and glyph, or the ones its state names. */
export function presentAction(action, row) {
    let face = null;
    try { face = action.state?.(row) || null; } catch { face = null; }
    const label = face?.label || action.label;
    return { label, icon: face?.icon || action.icon || '', title: face?.title || action.title || label,
        pressed: face?.pressed === true };
}

/** Run one action on `row`; a throw or a rejection becomes one line naming the action. */
export function runMessageAction(action, row, { anchor = null, notice = defaultNotice } = {}) {
    const failed = (error) => notice(fmt('{label} did not work: {reason}',
        { label: action.label, reason: String(error?.message || error || 'unknown error') }), 'danger');
    try {
        const result = action.run(row, { anchor, notice });
        if (typeof result?.then === 'function') result.then(null, failed);
    } catch (error) { failed(error); }
}

/**
 * The registry's controls in every saved bubble's control row of `feed`, redrawn when the registry changes:
 * `sync(bubble)` after the row is built, `destroy()` with the room. `onDomWrite` is the room's stable-viewport seam.
 */
export function bindBubbleActions({ feed, registry = getMessageActions(), notice = defaultNotice,
    onDomWrite = (write) => write() }) {
    function control(bubble, action, face) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'chat-message-action ui-hit-area';
        button.dataset.messageAction = action.key;
        button.innerHTML = controlIcon(face.icon);
        button.title = face.title;
        button.setAttribute('aria-label', face.label);
        if (face.pressed) button.setAttribute('aria-pressed', 'true');
        button.addEventListener('click', () => {
            const row = messageRow(bubble);
            if (row) runMessageAction(action, row, { anchor: button, notice });
        });
        return button;
    }

    /** Draw `bubble`'s actions between its Reply and its Copy; true when the row changed. */
    function sync(bubble) {
        const strip = bubble?.querySelector?.(':scope > .chat-message-actions');
        const row = strip ? messageRow(bubble) : null;
        if (!row) return false;
        const wanted = registry.actionsFor(row).filter((action) => action.surface !== 'menu')
            .map((action) => [action, presentAction(action, row)]);
        const signature = JSON.stringify(wanted.map(([action, face]) => [action.key, face]));
        if ((strip.dataset.actionsSignature || '[]') === signature) return false;
        strip.dataset.actionsSignature = signature;
        for (const old of strip.querySelectorAll(':scope > .chat-message-action')) old.remove();
        const copy = strip.querySelector(':scope > .chat-message-copy');
        for (const [action, face] of wanted) strip.insertBefore(control(bubble, action, face), copy);
        // The time keeps clear of every control on its line (style.css, the bubble's control row).
        bubble.style.setProperty('--chat-message-extra-actions', wanted.length ? String(wanted.length) : '');
        return true;
    }

    const off = registry.onChange(() => onDomWrite(() => {
        let changed = false;
        for (const strip of feed.querySelectorAll('.chat-bubble > .chat-message-actions')) {
            changed = sync(strip.parentElement) || changed;
        }
        return changed;
    }));
    return { sync, destroy: off };
}

/**
 * Skills' `message.action` contributions as registry entries on Ouroboros's saved replies, after the host's own
 * actions and in the store's arrangement, kept in step with `store`. `draft` is the visible room's draft. Returns
 * the disposer.
 */
export function bridgeContributedActions({ store, registry = getMessageActions(), draft = null,
    run = runContributionAction }) {
    let unregister = [];
    let signature = null;
    const drop = () => { for (const off of unregister) off(); unregister = []; };
    function sync() {
        const items = store.byPlacement('message.action');
        const next = JSON.stringify(items);
        if (next === signature) return;
        signature = next;
        registry.batch(() => {
            drop();
            unregister = items.map((item, index) => registry.register({
                key: `contribution:${item.key}`, label: item.label, title: item.title || '', icon: item.icon || '',
                order: 1000 + index,
                applies: (row) => row?.role === 'assistant',
                run: (row, ctx) => run(item, { row, draft, notice: ctx.notice }),
            }));
        });
    }
    const off = store.onChange(sync);
    sync();
    return () => {
        off();
        registry.batch(drop);
        signature = null;
    };
}
