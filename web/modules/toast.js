import { haptic } from './haptics.js';
import { SHELL_ICONS } from './page_icons.js';
import { normalizeTone } from './ui_helpers.js';

function getStack(host = null) {
    let stack = host ? host.querySelector(':scope > .toast-stack--room') : document.getElementById('toast-stack');
    if (!stack) {
        stack = document.createElement('div');
        if (!host) stack.id = 'toast-stack';
        stack.className = host ? 'toast-stack toast-stack--room' : 'toast-stack';
        stack.setAttribute('aria-live', 'polite');
        stack.setAttribute('aria-relevant', 'additions');
        (host || document.body).appendChild(stack);
        // A window that grows or shrinks cuts the column or releases it with no notice coming or going.
        if (!host && typeof ResizeObserver === 'function') new ResizeObserver(() => markCut(stack)).observe(stack);
    }
    return stack;
}

// The page's stack ends above the screen's bottom and scrolls past it (style.css); cut there, it is reached by
// keyboard too, as a wide table is (docs/DESIGN.md). Measured when a notice comes or goes and when the column's box
// changes size; room stacks keep their place.
function markCut(stack) {
    if (stack.id !== 'toast-stack') return;
    const cut = stack.scrollHeight > stack.clientHeight;
    if (cut === stack.hasAttribute('tabindex')) return;
    if (cut) {
        stack.tabIndex = 0;
        stack.setAttribute('role', 'region');
        stack.setAttribute('aria-label', 'Scrollable notices');
    } else {
        ['tabindex', 'role', 'aria-label'].forEach((name) => stack.removeAttribute(name));
    }
}

/**
 * One toast; a press anywhere on it dismisses it, a scroll over it never does. `ttl` 0 keeps it until then. `action`
 * ({label, run}) adds one button that dismisses the toast and runs (a recovery notice's way back, host_modules.js).
 * `dismissLabel` adds an accessible close button, the words beside it in their own box; hover or focus pauses the
 * auto-dismiss timer. An error toast answers the owner's own action and plays Telegram's error haptic (haptics.js); a
 * notice the owner did not cause passes `{ haptic: false }`.
 * Optional `host` confines the stack above a room's composer. Its owner calls the returned node's `dispose()` on teardown.
 */
export function showToast(message, tone = 'info', { ttl = 6000, action = null, haptic: felt = true, dismissLabel = '', host = null } = {}) {
    const stack = getStack(host);
    const toast = document.createElement('div');
    const cleanTone = normalizeTone(tone || 'info', 'info');
    if (felt && cleanTone === 'danger') haptic('error');
    toast.className = `toast toast-${cleanTone}`;
    toast.setAttribute('role', cleanTone === 'danger' ? 'alert' : 'status');
    const words = dismissLabel ? document.createElement('span') : toast;
    if (words !== toast) {
        words.className = 'toast-words';
        toast.append(words);
    }
    words.textContent = message || '';
    let timer = 0;
    let removalTimer = 0;
    let hovered = false;
    let focused = false;
    const clear = () => { if (timer) { clearTimeout(timer); timer = 0; } };
    const dispose = () => {
        clear();
        clearTimeout(removalTimer);
        globalThis.window?.removeEventListener?.('pagehide', dispose);
        toast.remove();
        markCut(stack);
    };
    toast.dispose = dispose;
    const dismiss = () => {
        clear();
        globalThis.window?.removeEventListener?.('pagehide', dispose);
        if (toast.classList.contains('is-hiding')) return;
        // It fades where it stood, out of the column, while the toasts that stay glide into its place at once: the
        // shift is measured here, the glide is the stylesheet's (none with reduced motion).
        const stay = [...(stack.children || [])].filter((el) => el !== toast);
        const before = stay.map((el) => el.offsetTop);
        Object.assign(toast.style, { position: 'absolute', top: `${toast.offsetTop}px`, left: `${toast.offsetLeft}px`, width: `${toast.offsetWidth}px` });
        toast.classList.add('is-hiding');
        const moved = stay.filter((el, i) => {
            const shift = before[i] - el.offsetTop;
            if (shift) { el.style.setProperty('--toast-shift', `${shift}px`); el.setAttribute('data-gliding', ''); }
            return shift !== 0;
        });
        void stack.offsetHeight;  // the shifted places are where the glide starts
        moved.forEach((el) => el.removeAttribute('data-gliding'));
        removalTimer = setTimeout(dispose, 180);
    };
    const arm = () => {
        clear();
        if (ttl > 0 && !hovered && !focused && !toast.classList?.contains?.('is-hiding')) timer = setTimeout(dismiss, ttl);
    };
    if (action?.label && typeof action.run === 'function') {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-sm btn-default toast-action';
        button.textContent = action.label;
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            dismiss();
            action.run();
        });
        toast.append(button);
    }
    if (dismissLabel) {
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'shell-icon-control toast-dismiss ui-hit-area';
        close.setAttribute('aria-label', dismissLabel);
        close.innerHTML = SHELL_ICONS.close;
        close.addEventListener('click', (event) => {
            event.stopPropagation();
            dismiss();
        });
        toast.append(close);
    }
    stack.appendChild(toast);
    markCut(stack);
    if (ttl > 0) {
        toast.addEventListener('pointerenter', () => { hovered = true; clear(); });
        toast.addEventListener('pointerleave', () => { hovered = false; arm(); });
        toast.addEventListener('focusin', () => { focused = true; clear(); });
        toast.addEventListener('focusout', (event) => { focused = Boolean(toast.contains?.(event.relatedTarget)); arm(); });
        globalThis.window?.addEventListener?.('pagehide', dispose, { once: true });
        arm();
    }
    toast.addEventListener('click', dismiss);
    return toast;
}
