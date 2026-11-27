// A room's own news (its history read failed, a return landed inexactly, its project list was restored) is the page's
// ordinary notice (toast.js), told only while the room is the one the person looks at: shown, and not covered by its own
// work or consciousness sheet or by the setup wizard. chat.js adds the room's own visibility; chat_render_batch.js and
// room_archive.js decide when a notice is owed and withdraw it when the room is left or covered.
import { showToast } from './toast.js';
import { tr } from './i18n.js';

/** The notice itself, its tone the event's own: a failed read is a failure, anything else plain news. */
export const roomNotice = (text, kind) => showToast(text, kind === 'failed' ? 'danger' : 'info',
    { haptic: false, dismissLabel: tr('toast.dismiss', 'Dismiss notice') });

/** Whether the room is covered: `host` carries the open work or consciousness sheet's marker, `bar` the chip's. */
export const roomCovered = (host, bar) => () => Boolean(host?.matches?.('[data-rw-open], [data-cx-shade]')
    || bar?.hasAttribute?.('data-cx-shade') || globalThis.document?.getElementById?.('onboarding-overlay'));

/** Calls `changed` whenever a cover comes or goes, or the page's own layers move (its body's classes and children). */
export function watchRoomCover(host, bar, changed) {
    if (typeof MutationObserver !== 'function') return null;
    const watch = new MutationObserver(changed);
    if (host) watch.observe(host, { attributes: true, attributeFilter: ['data-rw-open', 'data-cx-shade'] });
    if (bar && bar !== host) watch.observe(bar, { attributes: true, attributeFilter: ['data-cx-shade'] });
    if (globalThis.document?.body) watch.observe(document.body, { attributes: true, attributeFilter: ['class'], childList: true });
    return watch;
}
