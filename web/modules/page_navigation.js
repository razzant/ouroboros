import { pageLabel } from './pages.js';

/** A jump is a page layer, never another history. The shell supplies the current retained place, the leave guard
 * and `leave`, which keeps what the room on screen holds (its selection) before a jump's blur would end it; the
 * destination commits synchronously with its return, after that guard has allowed the move. */
export function createPageNavigation({ layers, capture, canLeave, land = () => {}, leave = () => {}, doc = document } = {}) {
    let serial = 0;
    let pressed = null;
    const onPress = (event) => { pressed = event; };
    doc.addEventListener('click', onPress, true);

    function clear() {
        const returns = layers.ids().map(id => layers.get(id)).filter(layer => layer.returnTo);
        for (const layer of returns) layers.remove(layer.id);
        for (const layer of returns) layer.returnTo.release?.();
    }

    const currentPlace = (opener) => {
        // WebKit checkpoints microtasks between click listeners without focusing the pressed button.
        const target = pressed?.eventPhase ? pressed.target : null;
        return capture(opener || target?.closest?.('button, a, [role="button"], [role="tab"]') || target || doc.activeElement);
    };
    async function go(key, commit, { map = false, row = '', opener, origin = currentPlace(opener), current = () => true } = {}) {
        const fromKey = pressed?.eventPhase ? pressed.detail === 0 : false;  // a key's click (Enter, Space) counts none
        // A source owns its claim until this commit; covering its room then transfers ownership to the destination.
        if (await canLeave(key) === false || !current()) return false;
        // A jump lets go of the page it leaves. A map choice is pressed in the sidebar or drawer, which stays: its row
        // keeps the focus, the opener a layer the commit opens gives it back to (layers.js), and the room it leaves
        // keeps its selection (WebKit drops a selection with the focus a blur takes from its room); a jump's room keeps
        // its own before that blur.
        if (map !== true && origin.key !== key) { leave(); doc.activeElement?.blur?.(); }
        let id = null;
        layers.batch(() => {
            if (map === true) clear(); // sidebar/drawer; a section's own list and tabs keep the visit
            else if (!map && origin.key !== key) {
                id = `page-return-${++serial}`;
                layers.push({ id, page: true, modal: false, focus: false, opener: null,
                    returnTo: origin, destination: key, back: { label: origin.label, title: origin.title },
                    onClose: async (reason) => {
                        if (await canLeave(origin.key) === false) return false;
                        leave();
                        doc.activeElement?.blur?.();
                        const restored = layers.batch(() => {
                            layers.remove(id);
                            return origin.restore();
                        });
                        origin.release?.();
                        restored?.({ keyboard: ['keyboard', 'escape', 'shortcut'].includes(reason) });
                        return true;
                    } });
            }
            commit();
            if (!map && origin.key !== key && !row) land();
        });
        // A jump made from a key leaves focus on the destination's round Back (a row it asked for keeps its own); a
        // pointer's jump leaves focus where the page puts it.
        if (fromKey && id && !row) doc.querySelector?.(`[data-shell-back][data-page-layer="${id}"]`)?.focus?.({ preventScroll: true });
        return true;
    }
    return { go, clear, capture: currentPlace, destroy: () => doc.removeEventListener('click', onPress, true) };
}

/** Capture from the existing page, level and reading owners; each return holds its own retained snapshot. */
export function capturePagePlace({ state, navState, settingsControls, dashboardControls, skillsControls,
    projectInstances, projectPanelTitle, rooms, mainChat, logsControls, restorePlaces, commitPage, doc = document }, opener) {
    const page = state.activePage, projectId = navState.activeProjectId;
    const settings = page === 'settings' ? settingsControls?.capture(opener) : null;
    const dashboard = page === 'dashboard' ? dashboardControls?.capture() : null;
    const skills = page === 'skills' ? skillsControls?.capture(opener) : null;
    const project = projectId ? { id: projectId, name: projectPanelTitle.textContent } : null;
    const root = project ? projectInstances.get(projectId)?.page : doc.getElementById(`page-${page}`);
    const reading = restorePlaces.capture(root, opener);
    const feed = page === 'chat' ? (project ? projectInstances.get(projectId) : mainChat)?.captureReading()
        : state.dashboardActiveSubtab === 'logs' && page === 'dashboard' ? logsControls?.captureReading() : null;
    const label = project?.name || settings?.label || dashboard?.label || skills?.label || (page === 'chat' ? 'Main' : pageLabel(page));
    return { key: project ? `project:${projectId}` : settings ? `settings:${state.settingsPath}`
        : dashboard ? `dashboard:${state.dashboardActiveSubtab}` : skills?.key || page, label, title: settings?.title || dashboard?.title || skills?.title || label,
        retains: settings?.retains || skills?.retains, release: settings?.release || skills?.release,
        restore() {
            commitPage(page, { reveal: true, enter: () => {
                settings?.restore();
                dashboard?.restore();
                skills?.restore();
                if (project && projectInstances.has(projectId)) {
                    navState.activeProjectId = projectId;
                    projectPanelTitle.textContent = projectPanelTitle.title = project.name;
                    projectInstances.get(projectId).page.hidden = false;
                    rooms.show(projectId);
                }
            } });
            return (input) => { reading(input); feed?.(); };
        } };
}

/** The boot fragment names an injected page, once; navigation never writes a route back into the address. */
export function openInitialPage(showPage, current, win = window, doc = document) {
    const name = String(win.location.hash || '').replace(/^#/, '').trim();
    if (name && name !== current && doc.getElementById(`page-${name}`)) showPage(name, { map: true });
}

/** Section levels stay with their owner; crossing pages asks each registered leave guard in order. */
export async function canLeavePage(key, state, handlers) {
    const to = key.startsWith('project:') ? 'chat' : key.split(':')[0];
    if (to === state.activePage) return true;
    for (const handler of handlers) if (await handler({ from: state.activePage, to }) === false) return false;
    return true;
}
