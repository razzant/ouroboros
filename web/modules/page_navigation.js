import { pageLabel } from './pages.js';

/** A jump is a page layer, never another history. The shell supplies the current retained place and the leave
 * guard; the destination commits synchronously with its return, after that guard has allowed the move. */
export function createPageNavigation({ layers, capture, canLeave, land = () => {}, doc = document } = {}) {
    let serial = 0;
    let pressed = null;
    const onPress = (event) => { pressed = event; };
    doc.addEventListener('click', onPress, true);

    function clear() {
        const returns = layers.ids().map(id => layers.get(id)).filter(layer => layer.returnTo);
        for (const layer of returns) layers.remove(layer.id);
        for (const layer of returns) layer.returnTo.release?.();
    }

    const currentPlace = () => {
        // WebKit checkpoints microtasks between click listeners without focusing the pressed button.
        const target = pressed?.eventPhase ? pressed.target : null;
        return capture(target?.closest?.('button, a, [role="button"], [role="tab"]') || target || doc.activeElement);
    };
    async function go(key, commit, { map = false, row = '', origin = currentPlace(), current = () => true } = {}) {
        // A source owns its claim until this commit; covering its room then transfers ownership to the destination.
        if (await canLeave(key) === false || !current()) return false;
        if (origin.key !== key) doc.activeElement?.blur?.();
        layers.batch(() => {
            if (map === true) clear(); // sidebar/drawer; a section's own list and tabs keep the visit
            else if (!map && origin.key !== key) {
                const id = `page-return-${++serial}`;
                layers.push({ id, page: true, modal: false, focus: false, opener: null,
                    returnTo: origin, destination: key, back: { label: origin.label, title: origin.title },
                    onClose: async (reason) => {
                        if (await canLeave(origin.key) === false) return false;
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
    const skills = page === 'skills' ? skillsControls?.capture() : null;
    const project = projectId ? { id: projectId, name: projectPanelTitle.textContent } : null;
    const root = project ? projectInstances.get(projectId)?.page : doc.getElementById(`page-${page}`);
    const reading = restorePlaces.capture(root, opener);
    const feed = page === 'chat' ? (project ? projectInstances.get(projectId) : mainChat)?.captureReading()
        : state.dashboardActiveSubtab === 'logs' && page === 'dashboard' ? logsControls?.captureReading() : null;
    const label = project?.name || settings?.label || dashboard?.label || (page === 'chat' ? 'Main' : pageLabel(page));
    return { key: project ? `project:${projectId}` : settings ? `settings:${state.settingsPath}`
        : dashboard ? `dashboard:${state.dashboardActiveSubtab}` : page, label, title: settings?.title || label,
        retains: settings?.retains, release: settings?.release,
        restore() {
            commitPage(page, { reveal: true, enter: () => {
                settings?.restore();
                dashboard?.restore();
                skills?.();
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
