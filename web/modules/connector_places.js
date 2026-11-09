// The connector detail's Places section: how many places this connector has and the door to Circles, where every
// place is listed, added and adjusted (one home: circles.js). Nothing is listed twice.
import { failureText, readPlaces, PLACES_CHANGED_EVENT } from './consciousness_api.js';
import { openCircles } from './place_sheet.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { tr, fmt } from './i18n.js';

export function connectorPlacesHtml(count) {
    const words = count ? fmt(count === 1 ? '{n} place' : '{n} places', { n: count }) : tr('places.none', 'No place yet');
    return `<h3>Places</h3><p>${escapeHtml(words)} · <button type="button" class="consciousness-inline-link ui-hit-area" data-circles-open>${tr('circles.link', 'Circles ›')}</button></p>`;
}

export function mountConnectorPlaces(host, connector, { leave, read = readPlaces } = {}) {
    let closed = false, controller = null, version = 0;
    async function load() {
        controller?.abort(); controller = new AbortController(); const request = ++version;
        host.innerHTML = '<h3>Places</h3><p class="skills-sheet-note" role="status">Reading connected places…</p>';
        try {
            const value = await read({ signal: controller.signal });
            if (closed || request !== version) return;
            host.innerHTML = connectorPlacesHtml(value.places.filter((place) => place.connector === connector).length);
        } catch (error) {
            if (closed || request !== version) return;
            host.innerHTML = `<h3>Places</h3><p role="alert" data-i18n-authored>${escapeHtml(failureText(error))}</p><button type="button" class="btn btn-default btn-sm" data-connector-retry>Try again</button>`;
        }
    }
    const click = (event) => {
        if (event.target.closest('[data-connector-retry]')) { void load(); return; }
        if (event.target.closest('[data-circles-open]')) { leave(); openCircles(); }
    };
    const changed = () => { void load(); };
    host.addEventListener('click', click);
    window.addEventListener(PLACES_CHANGED_EVENT, changed);
    void load();
    return () => {
        if (closed) return;
        closed = true; version += 1; controller?.abort();
        host.removeEventListener('click', click); window.removeEventListener(PLACES_CHANGED_EVENT, changed);
    };
}
