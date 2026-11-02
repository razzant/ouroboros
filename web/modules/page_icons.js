// The house monochrome stroke vector (Lucide geometry, `currentColor`): nav rows at 20px
// (style.css sizes them), controls and inline marks smaller. Its meaning is said in words
// beside it or by the accessible name of the control that holds it. The sidebar's rows are
// rendered from the page registry (pages.js), so no copy of these lives in index.html.
export const strokeIcon = (body, size = 20) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;

const PAGE_BODIES = {
    chat: '<path d="M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z"/><path d="M7 11h10"/><path d="M7 15h6"/><path d="M7 7h8"/>',
    // Projects: Lucide "layers" — distinct from the Files folder glyph (the Projects header renders it: pages.js).
    projects: '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    files: '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>',
    skills: '<path d="M12 2v4"/><path d="M12 18v4"/><path d="M4.93 4.93l2.83 2.83"/><path d="M16.24 16.24l2.83 2.83"/><path d="M2 12h4"/><path d="M18 12h4"/><path d="M4.93 19.07l2.83-2.83"/><path d="M16.24 7.76l2.83-2.83"/>',
    widgets: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    dashboard: '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    settings: '<path d="M9.671 4.136a2.34 2.34 0 0 1 4.659 0 2.34 2.34 0 0 0 3.319 1.915 2.34 2.34 0 0 1 2.33 4.033 2.34 2.34 0 0 0 0 3.831 2.34 2.34 0 0 1-2.33 4.033 2.34 2.34 0 0 0-3.319 1.915 2.34 2.34 0 0 1-4.659 0 2.34 2.34 0 0 0-3.32-1.915 2.34 2.34 0 0 1-2.33-4.033 2.34 2.34 0 0 0 0-3.831A2.34 2.34 0 0 1 6.35 6.051a2.34 2.34 0 0 0 3.319-1.915"/><circle cx="12" cy="12" r="3"/>',
};

export const PAGE_ICONS = Object.fromEntries(Object.entries(PAGE_BODIES).map(([page, body]) => [page, strokeIcon(body)]));

// The control glyphs a skill may name for a contributed control (`icon`, the catalog in
// ouroboros/extension_ui_validation.py CONTRIBUTION_ICONS, published as the projection's `icons`), plus the
// host's own state glyphs; names the page heads share draw those heads.
const CONTROL_BODIES = {
    ...PAGE_BODIES,
    sparkle: '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
    mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/>',
    volume: '<path d="M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z"/><path d="M16 9a5 5 0 0 1 0 6"/><path d="M19.364 18.364a9 9 0 0 0 0-12.728"/>',
    list: '<path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/><path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/>',
    plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
    template: '<rect width="18" height="7" x="3" y="3" rx="1"/><rect width="9" height="7" x="3" y="14" rx="1"/><rect width="5" height="7" x="16" y="14" rx="1"/>',
    gauge: PAGE_BODIES.dashboard,
    check: '<path d="M20 6 9 17l-5-5"/>',
    book: '<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>',
    stop: '<rect width="12" height="12" x="6" y="6" rx="2"/>',
    // Consciousness, its own time: the header's chip and the Dashboard's wakes.
    brain: '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>',
    close: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
    'message-circle': '<path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5z"/>',
    hash: '<path d="M4 9h16M3 15h16M10 3 8 21M16 3l-2 18"/>',
    mail: '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m2 6 10 7 10-7"/>',
    network: '<rect x="9" y="2" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="16" y="16" width="6" height="6" rx="1"/><path d="M12 8v4M5 16v-4h14v4"/>',
    inbox: '<path d="m4 4-3 10v6h22v-6L20 4ZM1 14h6l2 3h6l2-3h6"/>',
    plug: '<path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0ZM12 17v4"/>',
    globe: '<circle cx="12" cy="12" r="10"/><ellipse cx="12" cy="12" rx="4" ry="10"/><path d="M2 12h20"/>',
    bot: '<rect x="4" y="7" width="16" height="14" rx="2"/><path d="M12 7V3H8M8 12v3M16 12v3M1 11v6M23 11v6"/>',
    webhook: '<circle cx="12" cy="5" r="3"/><circle cx="5" cy="18" r="3"/><circle cx="19" cy="18" r="3"/><path d="m10 8-3 7m1 3h8M14 8l3 7"/>',
};

/** A control glyph by name at `size` px; a name outside the set draws the Skills glyph (a skill's control). */
export const controlIcon = (name, size = 14) => strokeIcon(CONTROL_BODIES[name] || CONTROL_BODIES.skills, size);

/** Host-projected geometry or a named glyph. Keep the skill/place label visible beside it. */
export function renderSkillIcon(spec, { size = 16, className = '' } = {}) {
    const pixels = Number.isFinite(size) && size > 0 ? size : 16;
    const name = typeof spec === 'string' ? spec : spec?.name;
    let svg = strokeIcon(CONTROL_BODIES[name] || CONTROL_BODIES.plug, pixels);
    // icon_spec rebuilt this geometry on the server; do not fetch or sanitize it again.
    if (spec?.kind === 'svg' && typeof spec.svg === 'string' && spec.svg.startsWith('<svg ') && spec.svg.endsWith('</svg>')) {
        svg = spec.svg.replace('<svg ', `<svg width="${pixels}" height="${pixels}" `);
    }
    const css = String(className).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
    return svg.replace('<svg ', `<svg aria-hidden="true"${css ? ` class="${css}"` : ''} `);
}

// The shell's control glyphs, each once: the menu (☰), back (‹), the one close (×), the one more (⋯) and the
// chevron. Same stroke family as the page icons; a control's accessible name says what the glyph does.
export const SHELL_ICONS = {
    menu: strokeIcon('<path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h16"/>'),
    back: strokeIcon('<path d="m15 18-6-6 6-6"/>'),
    close: strokeIcon('<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'),
    more: strokeIcon('<circle cx="5" cy="12" r="1.6" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/><circle cx="19" cy="12" r="1.6" fill="currentColor" stroke="none"/>'),
    chevron: strokeIcon('<path d="m9 18 6-6-6-6"/>', 16),
    plus: strokeIcon('<path d="M12 5v14M5 12h14"/>', 16),
};
