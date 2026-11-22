/** The levels of a section with pages (docs/DESIGN.md §10 "A section with pages"): the one mechanism its two owners,
 * Settings (settings.js) and Extensions (extensions.js), open their pages with. A level is a page of the page registry
 * (settings_pages.js) drawn in the section's pane, or a page the section keeps mounted itself (`own(path)` answers
 * `{ el, show(row), hide() }`: Extensions' skill views). The first level stands beside the section's list from 40rem,
 * alone below; every later one lies over the one under it.
 *
 * What is shared is only the levels: opening one (unsaved or jump-retained pages come back as left),
 * drawing it, letting it go, the arrival on --motion-fade, the shell's layers (`layer`), the row a level was opened for
 * (`revealRow`), the signal that a page is shown (`announce`: `ouro:settings-page-shown`, `detail.path` and
 * `detail.place`) and whether a page is the one on screen (`onScreen(path)`, what a mounted page asks before a late
 * read repaints it). Each section keeps its own list, band words, landing, search, attention and doors. */

import { bindScrollFade } from './scroll_fade.js';
import { settingsPageFor } from './settings_pages.js';
import { resetSecretReveals } from './settings_secrets.js';
import { setResult } from './ui_primitives.js';

const quoted = (value) => String(value).replace(/["\\]/g, '\\$&');
export const isVisible = (el) => Boolean(el?.isConnected && el.getClientRects?.().length);

/**
 * `place` names the section (`settings`, `skills`) in level ids and in the shown signal; `pane` is the column levels
 * stand in; `shown()` says the section is the place on screen; `ctx` is what a registered page is mounted with; `title`
 * names a path; `onTitled(level)` hears that a page learned its name while it drew.
 */
export function createSectionLevels({ place, pane, layers = null, shown = () => false, ctx = {}, title = (path) => path,
    own = () => null, onTitled = () => {} } = {}) {
    const levels = [];
    const aside = new Map();  // path → a level with unsaved words or owned by a jump's return
    const layered = new Set();
    let serial = 0;
    const top = () => levels[levels.length - 1] || null;

    function pageElement(path) {
        const el = document.createElement('section');
        el.className = 'settings-page scroll-fade-y';
        el.dataset.settingsPage = path;
        el.tabIndex = -1;
        el.setAttribute('aria-label', title(path));
        el.innerHTML = '<div class="settings-column"><div class="settings-page-body"></div></div>';
        return el;
    }

    // `opener` is the control that opened the level, where focus returns: WebKit does not focus a pressed button.
    // A level put aside with unsaved words comes back as it was left; any other is drawn anew.
    function open(path, { row = '', jump = null, under = levels.length, opener = null, view = '' } = {}) {
        while (levels.length > under) close(levels.pop());
        const focused = document.activeElement === document.body ? null : document.activeElement;
        const kept = aside.get(path);
        aside.delete(path);
        const level = Object.assign(kept || { path, id: `${place}-level-${++serial}`, view }, { opener: opener || focused });
        if (jump) level.jump = jump;
        const mine = kept ? null : own(path);
        if (mine) Object.assign(level, { el: mine.el, own: mine });
        else if (!kept) {
            level.el = pageElement(path);
            level.disposeFade = bindScrollFade(level.el);
        }
        // Only the arriving page moves (settings.css): the one that leaves is gone at once.
        arrive(level.el);
        if (level.el.parentElement !== pane) pane.append(level.el);
        levels.push(level);
        if (mine) mine.show?.(row);
        else if (!kept) mount(level, row);
        return level;
    }

    function mount(level, row) {
        const found = settingsPageFor(level.path);
        const body = level.el.querySelector('.settings-page-body');
        const failed = (err) => {
            console.error(`${place}: a page failed to draw`, level.path, err);
            const line = document.createElement('p');
            line.className = 'ui-result';
            body.replaceChildren(line);
            setResult(line, { text: `This page could not be shown: ${err?.message || err}`, tone: 'danger' });
        };
        if (!found) { failed(new Error('no page serves this address')); return; }
        try {
            Promise.resolve(found.mount(body, { path: level.path, row, view: level.view || '', ctx })).then((handle) => {
                if (level.closed) { handle?.unmount?.(); return; }
                level.handle = handle;
                // A page that learns its name while it draws (a skill's form) is named again.
                level.el.setAttribute('aria-label', title(level.path));
                if (level === top()) onTitled(level);
            }, failed);
        } catch (err) { failed(err); }
    }

    const heldByJump = level => layers?.ids().some(id => layers.get(id)?.returnTo?.retains?.(level.el));
    function disposeLevel(level) {
        level.closed = true;
        if (level.own) return;  // the views' owner disposes their shared mounted block
        level.disposeFade?.();
        try { level.handle?.unmount?.(); } catch (err) { console.error(`${place}: a page failed to let go`, level.path, err); }
    }
    function releaseAside() {
        for (const [path, level] of aside) {
            if (level.handle?.unsaved?.() || heldByJump(level)) continue;
            aside.delete(path);
            disposeLevel(level);
        }
    }

    // Closing a clean page releases its bindings. Unsaved words or an actual jump keep the same nodes; the skill
    // views stay mounted under their own owner. Return custody is read from the shell, never another history here.
    function close(level) {
        resetSecretReveals(level.el);
        const at = document.activeElement;
        if (at && level.el.contains(at)) at.blur?.();
        level.watch?.disconnect();
        if (layered.delete(level.id)) layers?.remove(level.id);
        if (level.own) {
            level.el.hidden = true;
            level.own.hide?.();
        } else level.el.remove();
        if (level.handle?.unsaved?.() || heldByJump(level)) aside.set(level.path, level);
        else disposeLevel(level);
    }

    /** Close every level above `keep`; answers the one just above it (where focus may return from). */
    function closeLevels(keep) {
        const left = levels.length > keep ? levels[keep] : null;
        while (levels.length > keep) close(levels.pop());
        return left;
    }

    // The arriving page: held at its start state for its first paint, then it comes in (opacity and 8 px from the
    // right on --motion-fade, settings.css; reduced motion keeps the fade alone). Nothing waits on it.
    function arrive(el) {
        el.dataset.settingsArrive = 'wait';
        const run = () => {
            if (el.dataset.settingsArrive !== 'wait') return;
            el.dataset.settingsArrive = 'in';
            el.addEventListener('animationend', () => { delete el.dataset.settingsArrive; }, { once: true });
            setTimeout(() => { delete el.dataset.settingsArrive; }, 600);
        };
        if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => requestAnimationFrame(run));
        else delete el.dataset.settingsArrive;
    }

    /** Only the top level is shown. */
    function showTop() {
        const level = top();
        for (const each of levels) {
            if (each !== level && !each.el.hidden) resetSecretReveals(each.el);
            each.el.hidden = each !== level;
        }
        return level;
    }

    // A jump's landing stays under its return; the first level IS that return. A child opened afterwards goes above
    // it. Crossing 40rem changes only which local levels hide their parent, never the owed jump or its return.
    function layer(beside, onClose) {
        if (!layers) return;
        const owed = level => level.jump && layers.has(level.jump) ? level.jump : undefined;
        const wanted = !shown() ? [] : levels.filter((level, at) => at > 0 || !(beside || owed(level)));
        layers.batch(() => {
            for (const id of [...layered]) {
                if (wanted.some((level) => level.id === id)) continue;
                layers.remove(id);
                layered.delete(id);
            }
            for (const level of wanted) {
                layers.push({ id: level.id, el: level.el, page: true, modal: false, focus: false, opener: null,
                    before: owed(level),
                    back: { label: levels.indexOf(level) > 0 ? title(levels[levels.indexOf(level) - 1].path) : place === 'settings' ? 'Settings' : 'Extensions' },
                    onClose: () => onClose(level) });
                layered.add(level.id);
            }
        });
    }

    // The row a page was opened for: its fold opens, it scrolls into view, its control takes focus. A page drawn after
    // a read gets it when the row appears.
    function revealRow(level, id) {
        const find = () => level.el.querySelector(`[data-setting-row="${quoted(id)}"]`);
        const reveal = (node) => {
            for (let fold = node.closest('details'); fold; fold = fold.parentElement?.closest('details')) fold.open = true;
            node.scrollIntoView?.({ block: 'center' });
            const control = node.querySelector('[data-setting-control], [data-setting-value], button, input, select, textarea');
            (control || node).focus?.({ preventScroll: true });
        };
        const node = find();
        if (node || typeof MutationObserver !== 'function') { if (node) reveal(node); return; }
        level.watch = new MutationObserver(() => {
            const hit = find();
            if (!hit) return;
            level.watch.disconnect();
            reveal(hit);
        });
        level.watch.observe(level.el, { childList: true, subtree: true });
    }

    /** Says which page of this section is visible (`detail.path`, '' for the section's list): a page reads and
     *  repaints on it for its own path. Nothing is said while the section is put away. */
    function announce() {
        if (!shown()) return;
        window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: top()?.path || '', place } }));
    }

    /** The page the owner is looking at: this section is shown, `path` is its top level and the window is not hidden. */
    const onScreen = (path) => shown() && !globalThis.document?.hidden && top()?.path === String(path || '');

    // The page-navigation owner holds this snapshot. Restoring the levels neither makes a jump nor captures a new
    // return. Its reading owner restores the actual opener after the section has painted.
    function capture() {
        const kept = levels.map(level => ({ level, opener: level.opener, jump: level.jump }));
        return { retains: el => kept.some(({ level }) => level.el === el), release: releaseAside,
            restore() {
                let shared = 0;
                while (shared < kept.length && levels[shared] === kept[shared].level) shared++;
                closeLevels(shared);
                kept.forEach(({ level, opener, jump }, under) => {
                    const restored = under < shared ? levels[under] : open(level.path, { under, view: level.view });
                    // The outer return has its own opener; a later local Back still owes this level's original one.
                    Object.assign(restored, { opener, jump });
                });
            } };
    }

    function destroy() {
        for (const id of layered) layers?.remove(id);
        layered.clear();
        for (const level of new Set([...levels, ...aside.values()])) {
            level.watch?.disconnect();
            disposeLevel(level);
        }
        levels.length = 0;
        aside.clear();
    }

    return { levels, top, open, closeLevels, arrive, showTop, layer, revealRow, announce, onScreen, capture, destroy,
        indexOf: (path) => levels.findIndex((level) => level.path === path) };
}
