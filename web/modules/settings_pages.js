/** The Settings pages (docs/DESIGN.md "Settings"; the shell in settings.js draws every page's title and Back).
 *
 * A page module registers its family once, when it is imported: `prefix` is the first part of a Settings path
 * (`agents` serves `agents`, `agents/<id>` and `agents/new`), `title` a string or `(path) => string`, and
 * `mount(container, { path, row, ctx })` renders the body for that path and may return `{ unmount() }`. The shell
 * imports the page modules; a path no page serves opens the home with the path's last word searched. */

const PAGES = new Map();

export function registerSettingsPage(prefix, { title, mount } = {}) {
    const key = String(prefix || '').trim();
    if (!key || typeof mount !== 'function') throw new TypeError('registerSettingsPage: a prefix and a mount');
    PAGES.set(key, { prefix: key, title, mount });
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

/** A door that leaves Settings: a Dashboard tab, another page, Main (`dashboard:costs`, `page:skills`, `main:context`). */
export const isDoorId = (path) => /^(dashboard|page|main):/.test(String(path || ''));

/** The registered prefixes, for search and the tests that walk every page. */
export function settingsPagePrefixes() {
    return [...PAGES.keys()];
}
