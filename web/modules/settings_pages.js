/** The page registry (docs/DESIGN.md "Settings"): the pages of Settings and of Extensions, drawn by the shell that
 * opens them (settings.js, extensions.js; their one level mechanism is section_levels.js), which draws every page's
 * title and Back.
 *
 * A page module registers its family once, when it is imported: `prefix` is the first part of a Settings path
 * (`agents` serves `agents`, `agents/<id>` and `agents/new`), `title` a string or `(path) => string`, and
 * `mount(container, { path, row, ctx })` renders the body for that path and may return `{ unmount() }`. The shell
 * imports the page modules; a path no page serves opens the home with the path's last word searched.
 * `exists(path)` may verify a dynamic address before restoring it from device memory. Without it the registered
 * page and its children are trusted at landing. */

const PAGES = new Map();

export function registerSettingsPage(prefix, { title, mount, exists } = {}) {
    const key = String(prefix || '').trim();
    if (!key || typeof mount !== 'function') throw new TypeError('registerSettingsPage: a prefix and a mount');
    PAGES.set(key, { prefix: key, title, mount, exists });
}

/** The page registered under the longest prefix of `path`, whole parts only (`agents` serves `agents/a1`, never
 * `agentsx`); null when none does. */
export function settingsPageFor(path) {
    const target = String(path || '');
    let best = null;
    for (const page of PAGES.values()) {
        const serves = target === page.prefix || target.startsWith(`${page.prefix}/`);
        if (serves && (!best || page.prefix.length > best.prefix.length)) best = page;
    }
    return best;
}

/** A door that leaves Settings: a Dashboard tab, another page, Main, a page of Extensions (`dashboard:costs`, `page:skills`,
 *  `main:context`, `skills:tools/<id>`). */
export const isDoorId = (path) => /^(dashboard|page|main|skills):/.test(String(path || ''));

/** The first-level page of Extensions (extensions.js) that `path` stands on, '' for a page of Settings. Extensions'
 *  pages are `installed`, `tools` (MCP servers), `browse` and `sources`; `tools/<id>` and `tools/new` lie on MCP
 *  servers, `skill` and `skill/<name>` on Skills, `skill-sources` on Sources. Settings opens these addresses there. */
export function extensionsPageOf(path) {
    const target = String(path || '');
    const head = target.split('/')[0];
    if (head === 'tools') return 'tools';
    if (head === 'skill') return 'installed';
    if (target === 'skill-sources') return 'sources';
    return ['installed', 'browse', 'sources'].includes(target) ? target : '';
}

/** The arrow's destination in words for a reader; an internal row already names its own page. */
export function settingsDoorName(name, to) {
    const section = String(to).startsWith('dashboard:') ? 'Dashboard' : String(to).startsWith('main:') ? 'Main' : '';
    return section ? `${name}, ${section}` : name;
}

/** The registered prefixes, for search and the tests that walk every page. */
export function settingsPagePrefixes() {
    return [...PAGES.keys()];
}
