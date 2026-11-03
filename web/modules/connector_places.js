// The connector detail's visible Places section uses the same rows as Settings.
import { readPlaces, PLACES_CHANGED_EVENT } from './consciousness_api.js';
import { settingsPlaceRowHtml } from './settings_consciousness.js';
import { openPlaceSheet } from './place_sheet.js';
import { connectPlaceButtonHtml, openConnectPlace } from './place_connect.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';

export function mountConnectorPlaces(host, connector, { leave, returnFocus, read = readPlaces } = {}) {
    let closed = false, controller = null, version = 0, places = [];
    async function load() {
        controller?.abort(); controller = new AbortController(); const request = ++version;
        host.innerHTML = '<h3>Places</h3><p class="skills-sheet-note" role="status">Reading connected places…</p>';
        try {
            const value = await read({ signal: controller.signal });
            if (closed || request !== version) return;
            places = value.places.filter((place) => place.connector === connector);
            host.innerHTML = `<h3>Places</h3>${places.length
                ? `<ul class="consciousness-place-rows">${places.map(settingsPlaceRowHtml).join('')}</ul><p class="skills-sheet-note">Each place has its own state, wakes and captured conversations.</p>`
                : `<p>No connected place for this connector.</p>${connectPlaceButtonHtml()}`}`;
        } catch (error) {
            if (closed || request !== version) return;
            host.innerHTML = `<h3>Places</h3><p role="alert" data-i18n-authored>${escapeHtml(error?.message || String(error))}</p><button type="button" class="btn btn-default btn-sm" data-connector-retry>Try again</button>`;
        }
    }
    const click = (event) => {
        if (event.target.closest('[data-connector-retry]')) { void load(); return; }
        // The shared control; this section's press filters the Connect flow to its own connector.
        if (event.target.closest('[data-place-connect]')) { leave(); openConnectPlace({ connector, returnFocus }); return; }
        const button = event.target.closest('[data-place-open]');
        const place = button && places.find((row) => row.id === button.dataset.placeOpen);
        if (place) { leave(); openPlaceSheet(place, { returnFocus }); }
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
