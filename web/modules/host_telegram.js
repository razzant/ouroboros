/** The Telegram Mini App host adapter (docs/DESIGN.md "Shell", plan P3).
 *
 * Active only inside Telegram: `html[data-ouroboros-host="telegram"]` (stamped by the sidecar proxy) or a non-empty
 * `Telegram.WebApp.initData`. The SDK's script tag is `async`, so the adapter binds when the global is there and
 * otherwise waits for the tag's load (and polls briefly, in case it loaded first). Every host call goes through one
 * gate, `call(target, method, minVersion, ...args)`: method presence and `isVersionAtLeast`, never a version
 * number in a screen. Binding is idempotent and `dispose()` releases every listener.
 *
 * What it does, once: `ready()`, `expand()`, `disableVerticalSwipes()` (7.7, owner Q5=A: nested scrollers on every
 * screen would otherwise minimise the app on a drag at the top); Telegram's BackButton is visible while the layer
 * stack is non-empty and its click pops one layer; the header, background and bottom bar take the room colour
 * (again on every theme change); the safe-area insets are mirrored to `--tg-safe-area-inset-*` and
 * `--tg-content-safe-area-inset-*` (ui.css combines them with `env()` into `--inset-top/bottom`); Telegram's
 * `colorScheme` is the "System" appearance source through theme.js's one hook; and when the visual viewport
 * collapses under ~100px while a field is focused (Telegram-iOS #2235) the field is blurred and `hideKeyboard()`
 * (9.1) asked for. NOT `viewportStableHeight → --vvh`: that rewrites the keyboard path and needs a device. */

export const KEYBOARD_COLLAPSE_PX = 100;
const SDK_SCRIPT = 'script[src*="telegram-web-app.js"]';
const LATE_POLL_MS = 250;
const LATE_POLL_LIMIT_MS = 10000;
const SIDES = ['top', 'bottom'];  // the shell reads vertical insets (ui.css --inset-top/bottom)

/** Inside Telegram: the host stamp, or launch data the SDK only carries there. */
export function isTelegramHost({ win = globalThis.window, doc = globalThis.document } = {}) {
    if (doc?.documentElement?.dataset?.ouroborosHost === 'telegram') return true;
    return Boolean(win?.Telegram?.WebApp?.initData);
}

/** One gate for every host method: present, new enough, and never throwing into the page. */
export function hostCall(target, method, minVersion, ...args) {
    const fn = target?.[method];
    if (typeof fn !== 'function') return false;
    if (minVersion && typeof target?.isVersionAtLeast === 'function' && !target.isVersionAtLeast(minVersion)) return false;
    try { fn.apply(target, args); return true; } catch { return false; }
}

const hexColor = (value) => {
    const text = String(value || '').trim();
    if (/^#[0-9a-f]{6}$/i.test(text)) return text.toLowerCase();
    if (/^#[0-9a-f]{3}$/i.test(text)) return `#${[...text.slice(1)].map((c) => c + c).join('')}`.toLowerCase();
    const rgb = text.match(/^rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)/i);
    if (rgb) return `#${rgb.slice(1, 4).map((n) => Number(n).toString(16).padStart(2, '0')).join('')}`;
    return '';
};

export function createTelegramHost({
    win = globalThis.window,
    doc = globalThis.document,
    layers = null,
    theme = win?.ouroTheme,
    roomColor = () => win?.getComputedStyle?.(doc.documentElement).getPropertyValue('--bg-primary'),
} = {}) {
    const disposers = [];
    let app = null;
    let bound = false;
    let disposed = false;
    let pollTimer = 0;
    let pollSince = 0;

    const webApp = () => win?.Telegram?.WebApp || null;
    const call = (method, minVersion, ...args) => hostCall(app, method, minVersion, ...args);
    const callOn = (member, method, minVersion, ...args) => {
        // A member (BackButton) is gated by the app's own version.
        if (minVersion && typeof app?.isVersionAtLeast === 'function' && !app.isVersionAtLeast(minVersion)) return false;
        return hostCall(app?.[member], method, '', ...args);
    };
    const on = (event, handler) => {
        if (typeof app?.onEvent !== 'function') return;
        try { app.onEvent(event, handler); } catch { return; }
        disposers.push(() => { try { app.offEvent?.(event, handler); } catch { /* gone */ } });
    };

    // ---- the room colour on Telegram's chrome: header (6.9 for a hex), background (6.1), bottom bar (7.10)
    function paintChrome() {
        const color = hexColor(roomColor());
        if (!color) return;
        call('setHeaderColor', '6.9', color);
        call('setBackgroundColor', '6.1', color);
        call('setBottomBarColor', '7.10', color);
    }

    // ---- safe areas: the SDK's objects mirrored once into the tokens ui.css combines with env()
    function mirrorInsets() {
        const root = doc?.documentElement;
        if (!root?.style?.setProperty) return;
        for (const [prefix, inset] of [['--tg-safe-area-inset-', app?.safeAreaInset], ['--tg-content-safe-area-inset-', app?.contentSafeAreaInset]]) {
            for (const side of SIDES) {
                const px = Math.max(0, Number(inset?.[side]) || 0);
                root.style.setProperty(`${prefix}${side}`, `${px}px`);
            }
        }
    }

    // ---- the host's Back control follows the layer stack and closes one layer
    const onBack = () => { layers?.back?.('host-back'); };
    function syncBack(snapshot = { size: layers?.size?.() || 0 }) {
        callOn('BackButton', snapshot.size > 0 ? 'show' : 'hide', '6.1');
    }

    // ---- keyboard recovery: a visual viewport under ~100px with a focused field (Telegram-iOS #2235)
    const editable = () => {
        const active = doc?.activeElement;
        return active && active !== doc.body && typeof active.blur === 'function'
            && (active.matches?.('input, textarea, select, [contenteditable]:not([contenteditable="false"])')) ? active : null;
    };
    function recoverKeyboard() {
        const height = Number(win?.visualViewport?.height ?? win?.innerHeight) || 0;
        if (height <= 0 || height >= KEYBOARD_COLLAPSE_PX) return false;
        const field = editable();
        if (!field) return false;
        field.blur();
        call('hideKeyboard', '9.1');
        return true;
    }

    // ---- the "System" appearance source: Telegram's colour scheme, not the OS's
    function bindTheme() {
        if (typeof theme?.setSystemSource !== 'function') return;
        const scheme = () => (app?.colorScheme === 'dark' ? 'dark' : app?.colorScheme === 'light' ? 'light' : '');
        const release = theme.setSystemSource(scheme);
        if (typeof release === 'function') disposers.push(release);
        on('themeChanged', () => { theme.refreshSystem?.(); paintChrome(); });
    }

    function attach() {
        if (bound || disposed) return bound;
        app = webApp();
        if (!app) return false;
        bound = true;
        call('ready');
        call('expand');
        call('disableVerticalSwipes', '7.7');
        mirrorInsets();
        on('safeAreaChanged', mirrorInsets);
        on('contentSafeAreaChanged', mirrorInsets);
        paintChrome();
        const onThemePaint = () => paintChrome();
        win?.addEventListener?.('ouro:theme-changed', onThemePaint);
        disposers.push(() => win?.removeEventListener?.('ouro:theme-changed', onThemePaint));
        bindTheme();
        if (callOn('BackButton', 'onClick', '6.1', onBack)) {
            disposers.push(() => callOn('BackButton', 'offClick', '6.1', onBack));
        }
        if (layers?.onChange) disposers.push(layers.onChange(syncBack));
        syncBack();
        disposers.push(() => callOn('BackButton', 'hide', '6.1'));
        const vv = win?.visualViewport;
        if (vv?.addEventListener) {
            vv.addEventListener('resize', recoverKeyboard);
            disposers.push(() => vv.removeEventListener('resize', recoverKeyboard));
        }
        on('viewportChanged', recoverKeyboard);
        return true;
    }

    function stopWaiting() {
        if (pollTimer) { win?.clearTimeout?.(pollTimer); pollTimer = 0; }
    }
    function waitForSdk() {
        // The async tag may still be loading: bind on its load, and poll in case it loaded before we looked.
        const script = doc?.querySelector?.(SDK_SCRIPT);
        if (script?.addEventListener) {
            const onLoad = () => attach();
            script.addEventListener('load', onLoad, { once: true });
            disposers.push(() => script.removeEventListener('load', onLoad));
        }
        pollSince = Date.now();
        const poll = () => {
            pollTimer = 0;
            if (disposed || attach()) return;
            if (Date.now() - pollSince < LATE_POLL_LIMIT_MS) pollTimer = win.setTimeout(poll, LATE_POLL_MS);
        };
        pollTimer = win.setTimeout(poll, LATE_POLL_MS);
    }

    /** Bind once; outside Telegram this is a no-op that reports `false`. */
    function bind() {
        if (disposed) return false;
        if (!isTelegramHost({ win, doc })) return false;
        if (attach()) return true;
        if (!pollTimer) waitForSdk();
        return false;
    }

    function dispose() {
        if (disposed) return;
        disposed = true;
        stopWaiting();
        while (disposers.length) { try { disposers.pop()(); } catch { /* already gone */ } }
        bound = false;
    }

    return { bind, dispose, bound: () => bound, active: () => isTelegramHost({ win, doc }), recoverKeyboard, paintChrome, mirrorInsets };
}
