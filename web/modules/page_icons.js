import { FLUENT } from './fluent_glyphs.js';

// The house glyphs are Fluent System Icons on a 20×20 canvas, filled with `currentColor` (fluent_glyphs.js): a place
// or a thing wears a solid mark (Filled), an action inside a control a line glyph (Regular, drawn as a thin filled
// outline). Nav rows at 18px and header controls at 20px (style.css sizes them), controls and inline marks smaller.
// Its meaning is said in words beside it or by the accessible name of the control that holds it. The sidebar's rows
// are rendered from the page registry (pages.js), so no copy of these lives in index.html.
// `strokeIcon` stays for the hand-drawn Lucide marks elsewhere (24×24 stroke geometry).
export const strokeIcon = (body, size = 20) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;

/** Filled geometry on the 20×20 Fluent canvas; `data-glyph="fluent"` lets CSS keep it filled (ui.css). */
export const fillIcon = (body, size = 20) => `<svg width="${size}" height="${size}" viewBox="0 0 20 20" fill="currentColor" data-glyph="fluent" aria-hidden="true" focusable="false">${body}</svg>`;

/** A Fluent glyph by its name in the set and its style ('filled' for places and things, 'regular' for actions). */
export const fluentIcon = (name, size = 20, style = 'filled') => fillIcon(FLUENT[`${name}-20-${style}`] || '', size);

const filled = (name) => FLUENT[`${name}-20-filled`];
const regular = (name) => FLUENT[`${name}-20-regular`];

const PAGE_BODIES = {
    chat: filled('chat'),
    // Projects: "layer" — distinct from the Files folder glyph (the Projects header renders it: pages.js).
    projects: filled('layer'),
    // Circles: the people a circle gathers; its places wear their connectors' marks.
    circles: filled('people-community'),
    files: filled('folder'),
    skills: filled('hat-graduation'),
    widgets: filled('grid'),
    dashboard: filled('data-histogram'),
    settings: filled('settings'),
};

export const PAGE_ICONS = Object.fromEntries(Object.entries(PAGE_BODIES).map(([page, body]) => [page, fillIcon(body)]));

// The control glyphs a skill may name for a contributed control (`icon`, the catalog in
// ouroboros/extension_ui_validation.py CONTRIBUTION_ICONS, published as the projection's `icons`), plus the
// host's own state glyphs; names the page heads share draw those heads. The names are the contract; the
// pictures are Fluent. `sparkle` is kept as a name but draws the cube: no sparkle, wand or star marks a model.
const CONTROL_BODIES = {
    ...PAGE_BODIES,
    sparkle: filled('cube'),
    mic: filled('mic'),
    volume: filled('speaker-2'),
    list: filled('text-bullet-list-ltr'),
    plus: regular('add'),
    template: filled('board'),
    gauge: PAGE_BODIES.dashboard,
    check: regular('checkmark'),
    book: filled('book'),
    stop: filled('stop'),
    share: filled('share'),
    // Consciousness, its own time: the header's chip and the Dashboard's wakes.
    brain: filled('brain-circuit'),
    close: regular('dismiss'),
    chevron: regular('chevron-down'),
    'message-circle': filled('comment'),
    hash: filled('number-symbol'),
    mail: filled('mail'),
    network: filled('people-team'),
    inbox: filled('mail-inbox'),
    plug: filled('plug-connected'),
    globe: filled('globe'),
    bot: filled('bot'),
    webhook: filled('link'),
};

/** A control glyph by name at `size` px; a name outside the set draws the Skills glyph (a skill's control). */
export const controlIcon = (name, size = 14) => fillIcon(CONTROL_BODIES[name] || CONTROL_BODIES.skills, size);

/** Host-projected geometry or a named glyph. Keep the skill/place label visible beside it. */
export function renderSkillIcon(spec, { size = 16, className = '' } = {}) {
    const pixels = Number.isFinite(size) && size > 0 ? size : 16;
    const name = typeof spec === 'string' ? spec : spec?.name;
    // fillIcon already says aria-hidden; the shared prefix below adds the class only once.
    let svg = fillIcon(CONTROL_BODIES[name] || CONTROL_BODIES.plug, pixels).replace(' aria-hidden="true"', '');
    // icon_spec rebuilt this geometry on the server; do not fetch or sanitize it again.
    if (spec?.kind === 'svg' && typeof spec.svg === 'string' && spec.svg.startsWith('<svg ') && spec.svg.endsWith('</svg>')) {
        svg = spec.svg.replace('<svg ', `<svg width="${pixels}" height="${pixels}" `);
    }
    const css = String(className).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
    return svg.replace('<svg ', `<svg aria-hidden="true"${css ? ` class="${css}"` : ''} `);
}

// Fluent "pin" (Regular, 20×20 filled outline): a Project row's Pin, and the mark of a pinned effort level
// (effort_chip.js wraps it in its own 20×20 filled svg).
export const PIN_BODY = regular('pin');

// The shell's control glyphs, each once: back (‹), the one close (×), the one more (⋯), the
// chevron, the magnifier of Find a project, a Project row's Pin and its slashed Unpin, and the move grip (⠿) of a
// Widgets card. Fluent Regular (actions inside controls); a control's accessible name says what the glyph does.
export const SHELL_ICONS = {
    search: fillIcon(regular('search'), 18),
    back: fillIcon(regular('chevron-left')),
    close: fillIcon(regular('dismiss')),
    more: fillIcon(regular('more-horizontal')),
    chevron: fillIcon(regular('chevron-right'), 16),
    plus: fillIcon(regular('add'), 16),
    pin: fillIcon(PIN_BODY, 16),
    unpin: fillIcon(regular('pin-off'), 16),
    grip: fillIcon(regular('re-order-dots-vertical'), 16),
};
