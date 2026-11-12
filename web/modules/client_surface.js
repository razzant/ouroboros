/** Owner Surface Fact: raw sending-surface observables, measured AT SEND TIME
 * (the pywebview JS bridge appears asynchronously after page load, so a
 * load-time snapshot could misclassify the desktop shell as a browser tab).
 * Deliberately no device taxonomy — the model classifies from raw facts. */
import { isStackLayout } from './layout.js';
import { getInterfaceSize } from './interface_size.js';

/** Spread-ready chat-frame field: `{}` when the snapshot failed. */
export function clientSurfaceField() {
    const snap = clientSurfaceSnapshot();
    return snap ? { client_surface: snap } : {};
}

export function clientSurfaceSnapshot() {
    try {
        const snapshot = {
            pywebview: Boolean(window.pywebview?.api),
            ua: String(navigator.userAgent || ''),
            viewport: { w: Number(window.innerWidth) || 0, h: Number(window.innerHeight) || 0 },
            // The shell's own switch (layout.js): the model hears "narrow" exactly when the owner sees the drawer.
            narrow_layout: isStackLayout(),
            coarse_pointer: window.matchMedia('(pointer: coarse)').matches,
            // Pixels per CSS px: a browser's own zoom shows here, as does a high-density screen.
            device_pixel_ratio: Number(window.devicePixelRatio) || 1,
            captured_at: new Date().toISOString(),
        };
        // The owner's Interface size, where this page sets it (the desktop app, a touch screen); a browser's own
        // zoom is not the page's to know, and a size asked of the desktop window counts once the window reports it.
        const size = getInterfaceSize().state();
        if ((size.route === 'app' && !size.requested) || size.route === 'screen') snapshot.interface_size = size.percent;
        // The sender's IANA zone: the host names it beside UTC in Main's clock line
        // (owner 7A). A browser that cannot report one sends none — never a guess.
        const timezone = sendingTimeZone();
        if (timezone) snapshot.timezone = timezone;
        return snapshot;
    } catch {
        return null;  // absence is an honest gap; never a guessed default
    }
}

function sendingTimeZone() {
    try {
        const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        return typeof zone === 'string' ? zone : '';
    } catch {
        return '';
    }
}
