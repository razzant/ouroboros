// Settings › Behavior › Consciousness (formerly Background Cognition): what Ouroboros does when nobody asked,
// on its own clock and in the places the owner connected. The autonomy level with its three notes, the daily
// allowance, how many of its own tasks may run, one link to Circles (where the places and their words live) and,
// under More, the wake interval bounds, the physical bound on concurrent conversations (moved here from Advanced,
// same key) and a pointer to its model. The fields keep their ids and keys, so settings.js reads and saves them as
// before; nothing here reads or writes a place.
import { renderSegmentedField } from './page_header.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { parseModelSource } from './route_editor_primitives.js';
import { AUTONOMY_LEVELS } from './consciousness_words.js';
import { openCircles } from './place_sheet.js';
import { settingsLinkHtml } from './settings_safety.js';
import { settingsSection } from './settings_controls.js';
import { tr } from './i18n.js';

/** The whole Behavior › Consciousness section (settings_controls.js settingsSection). */
export function consciousnessSectionHtml() {
    return settingsSection({
        title: 'Consciousness',
        id: 'settings-consciousness',
        lead: 'What Ouroboros does when nobody asked: on its own clock, and in the places you connected.',
        body: `<div class="settings-effort-card">
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
        <p class="settings-inline-note" data-consciousness-places>${tr('consciousness.places.circles', 'The places it lives in, their words and their circles:')} <button type="button" class="consciousness-inline-link ui-hit-area" data-circles-open>${tr('circles.link', 'Circles ›')}</button></p>
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
        </details>`,
    });
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

/** Bind once: an in-page link opens the Settings tab it names; the Circles link opens the Circles page. */
export function bindConsciousnessSettings(root) {
    const host = root.querySelector('[data-consciousness-places]');
    if (!host) return () => {};
    const section = host.closest('.form-section') || host;
    const onClick = (event) => {
        const goto = event.target.closest?.('[data-settings-goto]');
        if (goto && section.contains(goto)) { root.activateSettingsTab?.(goto.dataset.settingsGoto); return; }
        if (event.target.closest?.('[data-circles-open]')) openCircles();
    };
    section.addEventListener('click', onClick);
    return () => { section.removeEventListener('click', onClick); };
}
