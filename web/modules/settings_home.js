/** The home of Settings (docs/DESIGN.md "Settings"): six groups of rows, each a question with its current answer;
 * a row opens its page, or leaves Settings by its door (↗). The shell (settings.js) shows it alone, or beside the
 * page opened from it in the wide layout, where the row of that page is marked current. */

import { PAGE_ICONS, controlIcon, strokeIcon } from './page_icons.js';
import { HOME_ROWS } from './settings_rows.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

const OUT_ICON = strokeIcon('<path d="M7 17 17 7"/><path d="M7 7h10v10"/>', 14);
const IN_ICON = strokeIcon('<path d="m9 18 6-6-6-6"/>', 14);
export const isDoorId = (path) => /^(dashboard|page|main):/.test(String(path || ''));

function homeRowHtml(row) {
    return `<button type="button" class="settings-door" data-settings-open="${escapeHtml(row.to)}">`
        + `<span class="settings-door-glyph" aria-hidden="true">${PAGE_ICONS[row.glyph] || controlIcon(row.glyph, 18)}</span>`
        + `<span class="settings-door-words"><span class="settings-door-title">${escapeHtml(row.name)}</span>`
        + `<span class="settings-door-answer" data-settings-answer></span></span>`
        + `<span class="settings-door-end" aria-hidden="true">${isDoorId(row.to) ? OUT_ICON : IN_ICON}</span></button>`;
}

export function homeGroupsHtml(rows = HOME_ROWS) {
    const groups = [...new Set(rows.map((row) => row.group))];
    return groups.map((group) => `<section class="settings-group"><h3 class="settings-group-title">${escapeHtml(group)}</h3>`
        + `<div class="ui-card settings-door-list">${rows.filter((row) => row.group === group).map(homeRowHtml).join('')}</div></section>`).join('');
}

/** Draw the home into `root` and take its presses: `open(path, { opener })` for a row. Answers
 * `{ mark(path), dispose() }`: `mark` names the row whose page is shown beside the home. */
export function mountSettingsHome(root, { open }) {
    root.innerHTML = `<div class="settings-column settings-home-column">${homeGroupsHtml()}</div>`;
    const onClick = (event) => {
        const row = event.target.closest?.('[data-settings-open]');
        if (row && root.contains(row)) open(row.dataset.settingsOpen, { opener: row });
    };
    root.addEventListener('click', onClick);
    return {
        mark(path) {
            const first = String(path || '').split('/')[0];
            root.querySelectorAll('[data-settings-open]').forEach((row) => {
                if (row.dataset.settingsOpen === first) row.setAttribute('aria-current', 'page');
                else row.removeAttribute('aria-current');
            });
        },
        dispose() { root.removeEventListener('click', onClick); },
    };
}
