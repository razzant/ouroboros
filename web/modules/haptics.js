/** Telegram's haptic feedback (docs/DESIGN.md "Shell", Haptics; owner 19.10, 3A): one meaning per kind, rare, never a
 * substitute for what the screen shows. Only inside Telegram, through its `HapticFeedback` (Bot API 6.1); in a
 * browser, the desktop app and an older Telegram this does nothing, so no screen depends on it.
 *
 * - `snap`: the drawer, released, latched open or closed: a light impact.
 * - `tick`: a value stepped or flipped (an Effort or Context level, Swarm armed or disarmed): the selection tick.
 * - `press`: a long press opened a row's menu: a medium impact, as the press is recognised.
 * - `threshold`: a swipe on a message reached its Reply point (again after pulling back under it): a medium impact,
 *   once per crossing, as Telegram's own swipe to reply.
 * - `stop`: the owner stopped a work (Stop now, or an earlier attempt): a rigid impact, once, on the press.
 * - `error`: the owner's own action did not go through (its error toast; toast.js): the error notification.
 * - `pickup`: a widget was picked up to be moved: a medium impact.
 * Nothing on send, copy or an ordinary toggle, and no `warning` (Telegram iOS plays a sound for it). */
import { hostCall, isTelegramHost } from './host_telegram.js';

const KINDS = Object.freeze({
    snap: ['impactOccurred', 'light'],
    tick: ['selectionChanged'],
    press: ['impactOccurred', 'medium'],
    threshold: ['impactOccurred', 'medium'],
    stop: ['impactOccurred', 'rigid'],
    error: ['notificationOccurred', 'error'],
    pickup: ['impactOccurred', 'medium'],
});

/** True when Telegram took the haptic. */
export function haptic(kind, { win = globalThis.window, doc = globalThis.document } = {}) {
    const spec = KINDS[kind];
    const app = win?.Telegram?.WebApp;
    if (!spec || !app?.HapticFeedback || !isTelegramHost({ win, doc })) return false;
    // A member of the app: gated by the app's own version.
    if (typeof app.isVersionAtLeast === 'function' && !app.isVersionAtLeast('6.1')) return false;
    const [method, ...args] = spec;
    return hostCall(app.HapticFeedback, method, '', ...args);
}
