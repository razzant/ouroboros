// Catalog provenance and maintenance inside the shared model chooser. No model
// entitlement rule lives here: the catalog names its producer, inventory its actions.
import { byHandHtml, createHarnessMaintenanceController, currentMissing, maintenanceLatestLine, maintenanceOperationLine,
    maintenanceOperationPending, maintenanceProblemText, maintenanceVersionLine, missingProgram, programsUnknownHtml,
    wakeWords } from './harness_maintenance.js';
import { claudexorStatus, familyLabel } from './claudexor_status_store.js';
import { daemonAnswered } from './harness_accounts.js';
import { refreshMaintenanceCatalogs } from './settings_catalog.js';
import { harnessPresentation } from './harness_presentation.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

/** Account copies of the same list contribute one fact; distinct producer versions stay explicit. */
export function catalogPrograms(catalog = {}, { harness = '', label = '', source = '', profile = '', sources = catalog.model_sources || [] } = {}) {
    const groups = new Map(), supplied = new Map();
    const inScope = (row) => row && (!profile || (row.credentialProfileId || row.credential_profile_id) === profile)
        && (!source || (row.source_id || row.source) === source);
    const modelKey = (row) => JSON.stringify([harness ? '' : row.source_id || row.source || '', row.id || row.value, row.credentialProfileId || row.credential_profile_id || '']);
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
            const key = modelKey({...model, source: id, credentialProfileId: row.credentialProfileId});
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
        const owner = !catalog.account_catalogs && add(item);
        const owners = owner ? new Set([owner]) : supplied.get(modelKey(item)) || new Set();
        for (const name of owners) if (catalog.items && item.source_id && item.source && item.source !== item.source_id) {
            const names = groups.get(name).entrySources;
            if (!names.includes(item.source)) names.push(item.source);
        }
        return owners;
    });
    return [...groups.values()].map((group) => ({...group,
        mixed: groups.size > 1 || choices.some((owners) => !owners.has(group.harness)),
    }));
}

/** `lead` opens the line: the chooser names the list's producer; a family row names its program's own version.
 *  `install`: the line itself installs an absent program, as the chooser's does; false where the row's own press does. */
export function modelProgramMarkup(program, view = {}, lead = '', { install = true } = {}) {
    const entry = view.entry;
    const operation = view.operation || entry?.operation;
    const pending = Boolean(view.request) || maintenanceOperationPending(operation);
    const action = entry?.actions?.[entry.selection?.path ? 'update' : 'install'];
    const needsUpdate = entry && entry.updateAvailable !== 'no';
    const disabled = Boolean(view.busy || view.inspectionPending || pending);
    const [verb, press] = entry?.selection?.path ? ['Update', 'latest'] : ['Install', 'install'];
    const readError = view.inspectionError || view.inventoryError;
    const error = view.operationError || readError;
    // The facts of a read that failed, or of a newer list that left this program out, are no current basis for
    // an Install, an Update or advice to install by hand: the row keeps its reason and its way back.
    const standing = view.listed !== false && !readError;
    const versions = () => [...new Set(program.programs.map((item) => item.version || 'version unknown'))].join(', ');
    const subset = program.entrySources?.length ? `${program.entrySources.join(', ')} entries` : 'Models';
    lead ||= `${program.mixed ? `${subset} supplied by` : 'List from'} ${program.label} ${versions()}`;
    // An absent program keeps its reason and its recovery (Check again) as last read; only a read that still stands
    // (`currentMissing`: none refused, the program listed) offers its Install, by its family row's own press or, `install`,
    // this line's.
    const missing = pending || !standing ? null : missingProgram(entry, operation);
    const current = pending ? null : currentMissing(view);
    const latest = needsUpdate ? maintenanceLatestLine(entry) : '';
    let button = '';
    const checkAgain = `<button type="button" class="btn btn-default" data-program-action="inspect"${view.inspectionPending ? ' disabled' : ''}>${view.inspectionPending ? 'Checking…' : 'Check again'}</button>`;
    // The owner's unresolved press outranks any receipt: the row offers Retry same request, never a Check
    // operation labelled by an older operation's receipt — unless the controller says the resend is unsafe
    // (`retryBlocked`): the press stays on the record, the reason shows and the way on is a read-only Check again.
    const blocked = view.retryBlocked;
    const blockedText = typeof blocked === 'string' ? blocked : (blocked?.reason || '');
    if (view.request && blocked) button = checkAgain;
    else if (view.request) button = `<button type="button" class="btn btn-default" data-program-action="rejoin"${view.busy ? ' disabled' : ''}>${view.busy ? 'Sending…' : 'Retry same request'}</button>`;
    else if (pending && (view.operationError || operation?.termination === 'unconfirmed')) button = '<button type="button" class="btn btn-default" data-program-action="rejoin">Check operation</button>';
    // An operation under way keeps the press it made, disabled, whatever the last read said.
    else if (pending) button = `<button type="button" class="btn btn-default" data-program-action="${press}"${disabled ? ' disabled' : ''}>${verb}</button>`;
    else if (missing) button = (install && current?.installable ? `<button type="button" class="btn btn-default" data-program-action="install"${disabled ? ' disabled' : ''}>Install</button>` : '')
        + (missing.byHand || !current ? checkAgain : '');
    // Facts that no longer stand offer no new press: the way back is the row's own read again, also when no earlier
    // row of this program was read (`entry` null) and the refused read itself is what keeps the row standing.
    else if (!standing && (entry || readError)) button = checkAgain;
    // A press the engine never received keeps its reason and offers Update again; only a failed read withholds it.
    else if (needsUpdate && action?.available) button = `<button type="button" class="btn btn-default" data-program-action="${press}"${disabled ? ' disabled' : ''}>${verb}</button>`;
    const failure = maintenanceProblemText(operation?.problem);
    return `<div class="harness-model-program ui-field-help" data-program="${escapeHtml(program.harness)}">
        <span>${escapeHtml(lead)}${latest ? `\u00a0· <span class="harness-model-latest">${escapeHtml(latest)}</span>` : ''}</span>${button}
        ${needsUpdate && action && !action.available ? `<span>${escapeHtml(action.reason)}</span>` : ''}
        ${operation ? `<span role="status">${escapeHtml(maintenanceOperationLine(operation, {missing: !entry?.selection?.path && !operation.before?.path}))}</span>` : ''}
        ${blockedText ? `<span data-tone="warn">${escapeHtml(blockedText)}</span>` : ''}
        ${failure ? `<span data-tone="warn">${escapeHtml(failure)}</span>` : ''}
        ${missing?.byHand && byHandHtml(entry, operation) ? `<span>${byHandHtml(entry, operation)}</span>` : ''}
        ${error ? `<span data-tone="warn">${escapeHtml(error)}</span>` : ''}
        ${(operation?.limitations || []).map((line) => `<span>${escapeHtml(line)}</span>`).join('')}
        ${operation?.progress?.length ? `<span>${escapeHtml(operation.progress.at(-1))}</span>` : ''}
    </div>`;
}

let controller;
let disposeStatus;
const views = new Set();
function syncStatus() {
    const pending = [...views].some((view) => view.pending());
    if (pending && !disposeStatus) {
        disposeStatus = () => {}; // A subscription's first callback may render synchronously.
        disposeStatus = claudexorStatus.subscribe(() => {
            controller.poll();
        }, {visible: () => [...views].some((view) => view.pending())});
    } else if (!pending && disposeStatus) { disposeStatus(); disposeStatus = null; }
}
function programsController() {
    if (!controller) controller = createHarnessMaintenanceController({
        onChange: () => { for (const view of views) view.render(); syncStatus(); },
        onSettled: refreshMaintenanceCatalogs,
    });
    return controller;
}

/** Open choosers subscribe only while work is pending; opening itself observes the producer. */
export function bindModelProgramFooter(input, footer, changed) {
    let programs = [], opened = false, signature = '';
    const view = { pending: () => programs.some((program) => {
        const state = programsController().view(program.harness);
        return maintenanceOperationPending(state.operation);
    }), render() {
        const html = programs.map((program) => modelProgramMarkup(program, programsController().view(program.harness))).join('');
        footer.hidden = !programs.length;
        if (html === signature) return;
        const restoreFocus = footer.contains(input.ownerDocument.activeElement);
        signature = html;
        footer.innerHTML = html;
        changed(restoreFocus);
    } };
    const read = async () => {
        const answers = await Promise.all(programs.map((program) => programsController().refresh({harness: program.harness, fresh: true, checkLatest: true})));
        for (const program of programs) {
            const entry = answers.flatMap((answer) => answer?.harnesses || []).find((row) => row.harness === program.harness);
            if (entry && !maintenanceOperationPending(entry.operation) && program.programs.some((source) =>
                source.path !== entry.selection.path || source.version !== entry.selection.version)) {
                void refreshMaintenanceCatalogs(program.harness, entry.selection);
                break; // One catalog refresh covers every producer in this opened list.
            }
        }
    };
    const click = (event) => {
        const button = event.target.closest('[data-program-action]');
        if (button) void programsController().act(button.closest('[data-program]').dataset.program, button.dataset.programAction);
    };
    footer.addEventListener('click', click);
    function close() {
        opened = false; views.delete(view);
        syncStatus();
    }
    return {
        update(next = []) {
            const producers = (groups) => JSON.stringify(groups.map(({harness, programs}) => [harness, programs]));
            const same = producers(programs) === producers(next);
            programs = next;
            if (opened) view.render();
            if (opened && !same && programs.some((program) => {
                const selected = programsController().view(program.harness).entry?.selection;
                return program.programs.some((source) => source.path !== selected?.path || source.version !== selected?.version);
            })) read();
        },
        open() {
            if (opened) return;
            opened = true; views.add(view); view.render(); read();
            syncStatus();
        },
        close,
        dispose() { close(); footer.removeEventListener('click', click); },
    };
}

/** A setup family's line shows only with something to act on, once its newest-release read has answered; a refused
 *  read of this family is itself such a fact (its reason and Check again), while a refused whole list keeps its one
 *  notice in `others()`. The owner's unresolved press or operation keeps its reason and recovery through any
 *  read, also one that finds no program; a press error stays while its program is selected and the update it was
 *  about is still offered. Up to date stays silent, and so does an absent program the engine can install (its row
 *  says Install & connect); one it cannot install here, or whose install failed, says why and offers Check again. */
export function familyProgramShown(view = {}) {
    const entry = view.entry;
    const operation = view.operation || entry?.operation;
    if (view.request || maintenanceOperationPending(operation)) return true;
    // W5 (rl-20261127T050408): a refused read of THIS family is its own applicable fact — the row keeps the reason
    // and the read again, program selected or not — instead of staying silent behind the whole list's notice. Only
    // the engine-wide failure (`inventoryError`, spoken once in `others()`) is not repeated here; a shorter list
    // that merely left the program out (`listed === false`) says nothing either.
    if (view.inspectionError && !view.inventoryError) return true;
    // An absent program speaks only where the owner installs it by hand (the row's own press installs it
    // otherwise) — and not from a read that no longer stands, which would duplicate the whole list's own
    // notice. A terminal operation's receipt still speaks: a confirmed failure keeps its typed outcome.
    if (!entry?.selection?.path) {
        if (operation) return Boolean(missingProgram(entry, operation)?.byHand);
        if (view.inspectionError || view.inventoryError || view.listed === false) return false;
        return Boolean(missingProgram(entry, operation)?.byHand);
    }
    if (view.operationError) return entry.updateAvailable !== 'no';
    if (!entry.newest || entry.newest.status === 'not_read' || view.inspectionError || view.inventoryError) return false;
    return entry.updateAvailable !== 'no';
}

/** The setup wizard's family rows: the engine is asked only while it answers (now, when it comes up, after a sign-in
 *  or an operation settles), the newest release only for a family with a selected program; the step's tick polls. */
export function bindFamilyPrograms({ harnesses = [], store = claudexorStatus, visible = () => true, onChange = () => {},
    api, onSettled = refreshMaintenanceCatalogs } = {}) {
    let up = false;
    // The one predicate this surface reads the engine through, for what it draws and for what its press does: the
    // engine ANSWERS (`daemonAnswered`, so a facet that landed outranks a partial refusal of another) and the store
    // carries no live read error behind the row. A `running` an earlier visit left under a transport error is not a
    // serving engine: nothing is drawn from it, and the owner's press says the ordinary start it will actually make.
    const serving = () => !store.error && daemonAnswered(store.snapshot || {});
    const missing = (harness) => (serving() ? currentMissing(controller.view(harness)) : null);
    // This family's true pending intent, from the same standing view its row reads: the owner's unresolved press
    // or an operation under way. A family holding it keeps its row, and its independent Connect must not start a
    // competing new login while it stands.
    const pending = (harness) => {
        const view = controller.view(harness);
        return Boolean(view.request) || maintenanceOperationPending(view.operation || view.entry?.operation);
    };
    // A same-family install that has not resolved yet COMPETES with a new sign-in for that family: the
    // engine may be installing the program right now, and a second create would orphan one of the two jobs.
    // Only a program the engine currently READS as installed stands the fence down — a missing or unknown
    // current state keeps it closed.
    const competingInstall = (harness) => {
        if (!pending(harness)) return false;
        const view = controller.view(harness);
        const installed = serving() && Boolean(view.entry?.selection?.path) && view.listed !== false
            && !view.inspectionError && !view.inventoryError;
        return !installed;
    };
    let finding = false, findError = '';  // the owner's press that starts the agent daemon to list the programs
    async function find() {
        if (finding) return;
        finding = true; findError = ''; onChange();
        // An engine that already answers is only read again; a start is the owner's own, separately labelled act.
        if (!serving()) {
            const woke = await store.wake();
            if (!woke?.ok) {
                findError = `Could not start the agent daemon: ${woke?.error || 'request failed'}`;
                finding = false; onChange();
                return;
            }
        }
        await read();
        finding = false; onChange();
    }
    const controller = createHarnessMaintenanceController({ ...(api ? { api } : {}), visible, onChange,
        onSettled: (harness, operation) => { void onSettled(harness, operation); void read(); } });
    async function read() {
        if (!serving()) return null;
        const inventory = await controller.refresh({ fresh: true });
        const selected = (inventory?.harnesses || []).filter((entry) => harnesses.includes(entry.harness) && entry.selection?.path);
        return Promise.all(selected.map((entry) => controller.refresh({ harness: entry.harness, fresh: true, checkLatest: true })));
    }
    let checking = false;
    // The refused whole inventory's way back: ONE fresh inventory read. It starts nothing and sends nothing.
    async function checkInventory() {
        if (checking) return;
        checking = true; onChange();
        await controller.refresh({ fresh: true });
        checking = false; onChange();
    }
    return {
        read,
        /** The step's status tick: a first sight of the engine running reads; a remount reads again. Rows out of
         *  view (`visible`) observe nothing: their remount reads. */
        observe({ remount = false } = {}) {
            if (!visible()) return;
            const now = serving();
            if (now && (!up || remount)) void read();
            up = now;
            controller.poll();
        },
        /** `install`: the family's row cannot install its missing program without a second account, so this line does. */
        line(harness, { install = false } = {}) {
            const view = controller.view(harness);
            // A stopped engine hides nothing of the owner's own: an unresolved press and an operation under way
            // keep their row, their reason and their recovery. Everything else waits for the engine to run again.
            const ownerPending = pending(harness);
            // The owner's pressed Install keeps its line with its reason and recovery when the read before it
            // was refused or left the program out, like `others()` keeps a pressed program.
            const pressedAbsent = controller.pressed(harness) && !view.entry?.selection?.path
                && (Boolean(view.inspectionError || view.inventoryError || view.operationError) || view.listed === false);
            const shown = ownerPending || (serving() && (familyProgramShown(view)
                || (install && currentMissing(view)?.installable) || pressedAbsent));
            return shown ? modelProgramMarkup({ harness }, view,
                maintenanceVersionLine(view.entry, { unknownCurrent: view.unknownCurrent }), { install }) : '';
        },
        /** The family's program the running engine's current reads find absent (`currentMissing`), else null. */
        missing,
        /** This family's true pending intent (its row's own predicate): the owner's unresolved press or an
         *  operation under way. While it stands the row keeps its line, and Connect must not start a competing
         *  new login for the same family. */
        pending,
        /** Whether a new sign-in for this family would race an unresolved same-family install: the
         *  sign-in boundary uses it, so no button-level copy of the fence can bypass it. */
        competingInstall,
        /** The shared sign-in card's hooks: it says a missing program may be installed first; each answer reads that program
         *  again unless a current read shows it installed (absent, unread or refused, the sign-in may have installed it). */
        loginHooks: () => ({
            onAnswer: (harness) => {
                const view = controller.view(harness);
                // The answer may itself have started the agent daemon, and the store has not polled again:
                // its snapshot must not skip this program's own targeted read. Only a current read that
                // shows the program installed ends it.
                if (!(view.entry?.selection?.path && !view.inspectionError && !view.inventoryError)) void controller.refresh({ harness, fresh: true });
            },
            installing: (harness) => (missing(harness)?.installable ? familyLabel(harness, store.snapshot, { catalogKnown: Boolean(store.catalogKnown) }) : ''),
        }),
        /** One line for the engine's other programs (no subscription family) it lacks or this session installs: each its
         *  own Install, no sign-in or model promised; the press's operation then speaks there. */
        others() {
            // Stopped, the engine is asked nothing until the owner's press; a refused inventory keeps the engine's words.
            if (!store.snapshot) return '';  // nothing heard of the engine yet: nothing to say
            const row = (harness, view) => modelProgramMarkup({ harness }, view,
                `${harnessPresentation(harness).label} · ${maintenanceVersionLine(view.entry)}`);
            if (!serving()) {
                // The deliberate start says what it does: a reading the owner may ask again lives on the answered
                // engine's own refusal row, never behind a start.
                const notice = `<div class="agent-other-programs" data-programs-unknown>${programsUnknownHtml({
                    starts: wakeWords(store.snapshot), busy: finding, error: findError })}</div>`;
                // A stopped engine hides nothing of the owner's own: an unresolved press and an operation under
                // way keep their row and their receipt.
                const items = controller.harnesses.filter((harness) => !harnesses.includes(harness)).map((harness) => {
                    const view = controller.view(harness);
                    const operation = view.operation || view.entry?.operation;
                    return (view.request || maintenanceOperationPending(operation)) ? row(harness, view) : '';
                }).filter(Boolean);
                return items.length ? `${notice}<div class="agent-other-programs"><span>More programs:</span>${items.join('')}</div>` : notice;
            }
            // A refused whole inventory keeps its one notice and the way back that reads it again; a pressed
            // program's recovery renders beside it.
            const reason = controller.inventoryError();
            const items = controller.harnesses.filter((harness) => !harnesses.includes(harness)).map((harness) => {
                const view = controller.view(harness);
                const operation = view.operation || view.entry?.operation;
                // The owner's pending intent outranks the last list: an unresolved press or an operation under
                // way (a durable unconfirmed receipt included) keeps its row after a reload, when `pressed` is
                // false again.
                return view.request || maintenanceOperationPending(operation) || currentMissing(view) || controller.pressed(harness)
                    ? row(harness, view) : '';
            }).filter(Boolean);
            const notice = reason ? `<div class="agent-other-programs" data-programs-unknown>${programsUnknownHtml({ reason })}`
                + `<button type="button" class="btn btn-default" data-program-action="check-inventory"${checking ? ' disabled' : ''}>`
                + `${checking ? 'Checking…' : 'Check again'}</button></div>` : '';
            if (!notice && !items.length) return '';
            return `${notice}${items.length ? `<div class="agent-other-programs"><span>More programs:</span>${items.join('')}</div>` : ''}`;
        },
        bind(host) {
            host.querySelectorAll?.('[data-program-action]').forEach((control) => control.addEventListener('click', () => {
                if (control.dataset.programAction === 'find') return void find();
                // The whole list's way back: one fresh inventory read. It starts nothing and sends nothing.
                if (control.dataset.programAction === 'check-inventory') return void checkInventory();
                const harness = control.closest('[data-program]').dataset.program;
                controller.markPressed(harness);
                controller.act(harness, control.dataset.programAction);
            }));
        },
        dispose: () => controller.dispose(),
    };
}
