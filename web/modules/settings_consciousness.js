// Settings › Behavior › Consciousness (formerly Background Cognition): what Ouroboros does when nobody asked,
// on its own clock and in the places the owner connected. The autonomy level with its three notes, the daily
// allowance, how many of its own tasks may run, the places (each opens its sheet) and, under More, the wake
// interval bounds, the physical bound on concurrent conversations (moved here from Advanced, same key) and a
// pointer to its model. The fields keep their ids and keys, so settings.js reads and saves them as before;
// the places list reads `GET /api/places` on its own and writes nothing here (the sheet writes).
import { renderSegmentedField } from './page_header.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { parseModelSource } from './route_editor_primitives.js';
import { PLACES_CHANGED_EVENT, readPlaces } from './consciousness_api.js';
import { AUTONOMY_LEVELS, placeStateWord } from './consciousness_words.js';
import { skillIconHtml } from './skill_presentation.js';
import { openPlaceSheet } from './place_sheet.js';
import { connectPlaceButtonHtml, openConnectPlace } from './place_connect.js';
import { settingsLinkHtml } from './settings_safety.js';
import { tr, fmt } from './i18n.js';

/** The section body under its title (settings_ui.js keeps the `<h3>`). */
export function consciousnessSectionHtml() {
    return `
        <div class="settings-section-copy">What Ouroboros does when nobody asked: on its own clock, and in the places you connected.</div>
        <div class="settings-effort-card">
            <label>Autonomy</label>
            <input id="s-consciousness-autonomy" type="hidden" value="act">
            ${renderSegmentedField({ target: 's-consciousness-autonomy', options: AUTONOMY_LEVELS.map(({ value, label }) => ({ value, label })) })}
            <ul class="consciousness-levels">${AUTONOMY_LEVELS.map((level) => `<li><strong>${escapeHtml(level.label)}</strong> &mdash; ${escapeHtml(level.note)}</li>`).join('')}</ul>
        </div>
        <div class="form-grid two">
            <div class="form-field ui-field">
                <label for="s-consciousness-daily-usd">Daily allowance (USD)</label>
                <input id="s-consciousness-daily-usd" placeholder="20" class="ui-control" name="s-consciousness-daily-usd" type="text" aria-describedby="s-consciousness-daily-usd-help">
                <div class="settings-inline-note ui-field-help" id="s-consciousness-daily-usd-help">Spending cap over a rolling 24 hours: its own wakes, wakes in places, and the work they start. Settled dollars only; a subscription shows as unmetered.</div>
            </div>
            <div class="form-field ui-field">
                <label for="s-consciousness-max-tasks">Max concurrent tasks</label>
                <input id="s-consciousness-max-tasks" type="number" min="0" step="1" placeholder="2" class="ui-control" name="s-consciousness-max-tasks" aria-describedby="s-consciousness-max-tasks-help">
                <div class="settings-inline-note ui-field-help" id="s-consciousness-max-tasks-help">How many of its own tasks may run at once. Work you asked for never counts.</div>
            </div>
        </div>
        <div class="consciousness-places" data-consciousness-places>
            <div class="consciousness-places-head">
                <h4 class="consciousness-subhead">Places</h4>
                ${connectPlaceButtonHtml()}
            </div>
            <div class="consciousness-place-list" data-place-list><p class="settings-inline-note">Reading places…</p></div>
        </div>
        <details class="consciousness-more" data-consciousness-more>
            <summary><span class="consciousness-marker" aria-hidden="true">▸</span>More</summary>
            <div class="consciousness-more-body">
                <div class="form-grid two">
                    <div class="form-field ui-field">
                        <label for="s-bg-wakeup-min">Shortest wake interval (s)</label>
                        <input id="s-bg-wakeup-min" type="number" min="60" step="1" placeholder="900" class="ui-control" name="s-bg-wakeup-min" aria-describedby="s-bg-wakeup-help">
                    </div>
                    <div class="form-field ui-field">
                        <label for="s-bg-wakeup-max">Longest wake interval (s)</label>
                        <input id="s-bg-wakeup-max" type="number" min="60" step="1" placeholder="14400" class="ui-control" name="s-bg-wakeup-max" aria-describedby="s-bg-wakeup-help">
                    </div>
                </div>
                <div class="settings-inline-note" id="s-bg-wakeup-help">On its own clock it chooses when to wake, between these bounds. They apply without a restart.</div>
                <div class="form-grid two">
                    <div class="form-field ui-field">
                        <label for="s-presence-max-active">Concurrent conversations</label>
                        <input id="s-presence-max-active" type="number" min="1" max="20" value="2" class="ui-control" name="s-presence-max-active" aria-describedby="s-presence-max-active-help">
                        <div class="settings-inline-note ui-field-help" id="s-presence-max-active-help">Places it talks in at the same time. Others wait their turn.</div>
                    </div>
                    <div class="form-field ui-field">
                        <label id="s-consciousness-model-label">Model</label>
                        <div class="consciousness-model-value" aria-labelledby="s-consciousness-model-label"><span data-consciousness-model>Uses Main</span>
                            ${settingsLinkHtml('models', 'Change in Models')}</div>
                    </div>
                </div>
            </div>
        </details>`;
}

/** One row of the places list: icon, authored label, state word, open turns and connector readiness; the row's text is its accessible name. */
export function settingsPlaceRowHtml(place) {
    const count = Number.isInteger(place.unanswered) && place.unanswered > 0 ? fmt('{n} open turns', { n: place.unanswered }) : '';
    const readiness = place.readiness ? tr(`places.readiness.${place.readiness.text}`, place.readiness.text) : '';
    return `<li><button type="button" class="consciousness-place-row" data-place-open="${escapeHtml(place.id)}">
        <span class="place-glyph" aria-hidden="true">${skillIconHtml(place.icon, 18)}</span>
        <span class="consciousness-place-name" data-i18n-authored>${escapeHtml(place.label)}</span>
        <span class="consciousness-place-facts">${escapeHtml([placeStateWord(place.state), count, readiness].filter(Boolean).join(' · '))}</span>
    </button></li>`;
}

/** «Could not move an old connection: <detail>. Connect it again.», each distinct sentence once. */
export function migrationProblemLines(problems = []) {
    const lines = problems.map((problem) => {
        const why = (problem.detail || problem.code.replace(/_/g, ' ')).trim().replace(/\.+$/, '');
        return `Could not move an old connection${why ? `: ${why}` : ''}. Connect it again.`;
    });
    return [...new Set(lines)];
}

export function settingsPlacesHtml({ places = null, error = '', problems = [] } = {}) {
    if (error) return `<p class="settings-inline-note" data-tone="warn">${escapeHtml(error)}</p>`;
    if (!places) return '<p class="settings-inline-note">Reading places…</p>';
    // What the places migration could not carry over; said here only, never on the Dashboard.
    const moved = migrationProblemLines(problems).map((line) => `<p class="settings-inline-note" data-tone="warn">${escapeHtml(line)}</p>`).join('');
    if (!places.length) return `${moved}<p class="settings-inline-note">No place yet. Connect one from an installed connector.</p>`;
    return `${moved}<ul class="consciousness-place-rows">${places.map(settingsPlaceRowHtml).join('')}</ul>`;
}

/** «Uses Main», or the consciousness slot's model as the Models tab names it. */
export function consciousnessModelText(settings = {}) {
    const value = String(settings.OUROBOROS_MODEL_CONSCIOUSNESS ?? '').trim();
    return value ? parseModelSource(value).model || value : 'Uses Main';
}

/** Paint the parts a loaded settings document decides (the fields themselves are settings.js's). */
export function applyConsciousnessSettings(root, settings = {}) {
    const model = root.querySelector('[data-consciousness-model]');
    if (model) model.textContent = consciousnessModelText(settings);
}

/**
 * Bind once: the places list follows the server (read when Behavior shows, again after any place write),
 * a row opens its sheet, Connect opens the Connect flow, an in-page link opens the tab it names.
 */
export function bindConsciousnessSettings(root, { read = readPlaces, state = null } = {}) {
    const host = root.querySelector('[data-consciousness-places]');
    const list = root.querySelector('[data-place-list]');
    if (!host || !list) return () => {};
    const section = host.closest('.form-section') || host;
    let places = null;
    let problems = [];
    let generation = 0;
    const paint = (view) => { list.innerHTML = settingsPlacesHtml({ problems, ...view }); };
    const visible = () => (!state || state.activePage === 'settings') && root.dataset.activeSettingsTab === 'behavior';
    async function refresh() {
        const current = ++generation;
        try {
            const data = await read();
            if (current !== generation) return;
            places = data.places;
            problems = data.migrationProblems || [];
            paint({ places });
        } catch (error) {
            if (current === generation) paint({ places, error: fmt('Places could not be read: {reason}', { reason: error?.message || error }) });
        }
    }
    const onClick = (event) => {
        const goto = event.target.closest?.('[data-settings-goto]');
        if (goto && section.contains(goto)) { root.activateSettingsTab?.(goto.dataset.settingsGoto); return; }
        if (event.target.closest?.('[data-place-connect]')) { openConnectPlace({ returnFocus: event.target.closest('[data-place-connect]'), openSettingsTab: (tab) => root.activateSettingsTab?.(tab) }); return; }
        const row = event.target.closest?.('[data-place-open]');
        const place = row && (places || []).find((entry) => entry.id === row.dataset.placeOpen);
        if (place) openPlaceSheet(place, { returnFocus: row });
    };
    const onShown = (event) => {
        if (event?.detail?.tab === 'behavior' || (event?.detail?.page === 'settings' && visible())) void refresh();
    };
    const onChanged = () => { void refresh(); };
    section.addEventListener('click', onClick);
    window.addEventListener('ouro:settings-subtab-shown', onShown);
    window.addEventListener('ouro:page-shown', onShown);
    window.addEventListener(PLACES_CHANGED_EVENT, onChanged);
    if (visible()) void refresh();
    return () => {
        generation += 1;
        section.removeEventListener('click', onClick);
        window.removeEventListener('ouro:settings-subtab-shown', onShown);
        window.removeEventListener('ouro:page-shown', onShown);
        window.removeEventListener(PLACES_CHANGED_EVENT, onChanged);
    };
}
