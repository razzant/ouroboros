// Catalog provenance and maintenance inside the shared model chooser. No model
// entitlement rule lives here: the catalog names its producer, inventory its actions.
import { createHarnessMaintenanceController, maintenanceLatestLine, maintenanceOperationLine, maintenanceOperationPending } from './harness_maintenance.js';
import { claudexorStatus } from './claudexor_status_store.js';
import { refreshMaintenanceCatalogs } from './settings_catalog.js';
import { harnessPresentation } from './harness_presentation.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

/** Account copies of the same list contribute one fact; distinct producer versions stay explicit. */
export function catalogPrograms(catalog = {}, { harness = '', label = '', source = '', profile = '', sources = [] } = {}) {
    const groups = new Map(), supplied = new Map();
    const inScope = (row) => row && (!profile || (row.credentialProfileId || row.credential_profile_id) === profile)
        && (!source || (row.source_id || row.source) === source);
    const modelKey = (row) => JSON.stringify([harness ? '' : row.source_id || row.source || '', row.id || row.value]);
    function add(row, owner = harness, name = label) {
        if (!inScope(row)) return;
        const id = row.source_id || row.source;
        const descriptor = sources.find((item) => item.id === id);
        owner = owner || row.credentialHarness || descriptor?.credentialHarness;
        const program = row.sourceProgram;
        if (!owner || !program?.path) return;
        if (!groups.has(owner)) groups.set(owner, { harness: owner, label: name || harnessPresentation(owner).label, programs: [], entrySources: [] });
        const group = groups.get(owner);
        if (!group.programs.some((item) => item.path === program.path && item.version === program.version)) group.programs.push(program);
        for (const model of row.models || []) {
            const key = modelKey({...model, source: id});
            if (!supplied.has(key)) supplied.set(key, new Set());
            supplied.get(key).add(owner);
        }
        return owner;
    }
    const envelope = catalog.model_catalog || catalog;
    if (Array.isArray(envelope.accounts)) for (const account of envelope.accounts) add(account.catalog);
    else add(envelope);
    for (const list of catalog.account_catalogs || []) {
        if (source && list.source !== source) continue;
        for (const account of list.accounts || []) add(account.catalog);
    }
    const choices = (catalog.items || catalog.models || []).filter(inScope).map((item) => {
        const owner = add(item);
        if (owner && catalog.items && item.source_id && item.source && item.source !== item.source_id) {
            const names = groups.get(owner).entrySources;
            if (!names.includes(item.source)) names.push(item.source);
        }
        return owner ? new Set([owner]) : supplied.get(modelKey(item)) || new Set();
    });
    return [...groups.values()].map((group) => ({...group,
        mixed: groups.size > 1 || choices.some((owners) => !owners.has(group.harness)),
    }));
}

export function modelProgramMarkup(program, view = {}) {
    const entry = view.entry;
    const operation = view.operation || entry?.operation;
    const pending = Boolean(view.request) || maintenanceOperationPending(operation);
    const action = entry?.actions?.[entry.selection?.path ? 'update' : 'install'];
    const needsUpdate = entry && entry.updateAvailable !== 'no';
    const disabled = Boolean(view.busy || view.inspectionPending || pending);
    const verb = entry?.selection?.path ? 'Update' : 'Install';
    const error = view.operationError || view.inspectionError || view.inventoryError;
    const versions = [...new Set(program.programs.map((item) => item.version || 'version unknown'))].join(', ');
    const subset = program.entrySources?.length ? `${program.entrySources.join(', ')} entries` : 'Models';
    let button = '';
    if (view.request && !operation) button = `<button type="button" class="btn btn-default" data-program-action="rejoin"${view.busy ? ' disabled' : ''}>${view.busy ? 'Sending…' : 'Retry same request'}</button>`;
    else if (pending && (view.operationError || operation?.termination === 'unconfirmed')) button = '<button type="button" class="btn btn-default" data-program-action="rejoin">Check operation</button>';
    else if ((needsUpdate && action?.available && !error) || pending) button = `<button type="button" class="btn btn-default" data-program-action="latest"${disabled ? ' disabled' : ''}>${verb}</button>`;
    return `<div class="harness-model-program ui-field-help" data-program="${escapeHtml(program.harness)}">
        <span>${program.mixed ? `${escapeHtml(subset)} supplied by` : 'List from'} ${escapeHtml(program.label)} ${escapeHtml(versions)}</span>
        ${needsUpdate ? `<span>${escapeHtml(maintenanceLatestLine(entry))}</span>` : ''}${button}
        ${needsUpdate && action && !action.available ? `<span>${escapeHtml(action.reason)}</span>` : ''}
        ${operation ? `<span role="status">${escapeHtml(maintenanceOperationLine(operation, {missing: !entry?.selection?.path && !operation.before?.path}))}</span>` : ''}
        ${error ? `<span data-tone="warn">${escapeHtml(error)}</span>` : ''}
        ${(operation?.limitations || []).map((line) => `<span>${escapeHtml(line)}</span>`).join('')}
        ${operation?.progress?.length ? `<span>${escapeHtml(operation.progress.at(-1))}</span>` : ''}
    </div>`;
}

let controller;
let disposeStatus;
const views = new Set();
function programsController() {
    if (!controller) controller = createHarnessMaintenanceController({
        onChange: () => { for (const view of views) view.render(); },
        onSettled: refreshMaintenanceCatalogs,
    });
    return controller;
}

/** Only open choosers hold a status subscription; ticks read operations, never inventory/newest. */
export function bindModelProgramFooter(input, footer, changed) {
    let programs = [], opened = false, signature = '';
    const view = { render() {
        const html = programs.map((program) => modelProgramMarkup(program, programsController().view(program.harness))).join('');
        footer.hidden = !programs.length;
        if (html === signature) return;
        const restoreFocus = footer.contains(input.ownerDocument.activeElement);
        signature = html;
        footer.innerHTML = html;
        changed(restoreFocus);
    } };
    const read = () => { for (const program of programs) void programsController().refresh({harness: program.harness, fresh: true, checkLatest: true}); };
    const click = (event) => {
        const button = event.target.closest('[data-program-action]');
        if (button) void programsController().act(button.closest('[data-program]').dataset.program, button.dataset.programAction);
    };
    footer.addEventListener('click', click);
    function close() {
        opened = false; views.delete(view);
        if (!views.size) { disposeStatus?.(); disposeStatus = null; }
    }
    return {
        update(next = []) {
            const same = JSON.stringify(programs) === JSON.stringify(next);
            const owners = programs.map((program) => program.harness).join();
            programs = next;
            if (opened || input.dataset.programPlacement === 'line') view.render();
            if (opened && !same && owners !== programs.map((program) => program.harness).join()) read();
        },
        open() {
            if (opened) return;
            opened = true; views.add(view); view.render(); read();
            if (!disposeStatus) disposeStatus = claudexorStatus.subscribe((status) => {
                if (!status.loading) programsController().poll();
            }, {visible: () => views.size > 0});
        },
        close,
        dispose() { close(); footer.removeEventListener('click', click); },
    };
}
