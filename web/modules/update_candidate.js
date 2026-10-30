/* The prepared Update beside the running app (UPDATE-TECHNICAL-PLAN §5.6, §6.1), worded for Dashboard › Updates.

   `/api/update/status` carries `candidate`: the one update this installation prepared in its own copy (its state,
   pins and checks, the activation's outcome and the actions its state allows), with `start_failure`, this checkout's
   last failed start, and `last_activation`, the last Restart to update's settled outcome, which outlives its record
   until the next preparation is admitted (the panel's Last attempt). Preparing never changes the running checkout;
   Restart to update (`activate`) is the one step that does. Every sentence here is read from those fields and the
   status around them, never guessed from a version. updates.js::updateVerdict is the one caller and decides which
   verdict wins; this module only words them. */

import { updateEffectsText } from './update_effects.js';
import { activationOutcomeKey, attemptOutcome, attemptTime } from './update_status.js';

const short = (sha) => String(sha || '').slice(0, 8);
// A version with its commit, "7.8.1 (88acfdbe)": either alone when the other is unknown, '' when both are.
const versioned = (version, sha) => (version && sha ? `${version} (${short(sha)})` : String(version || short(sha)));
const strings = (value) => (Array.isArray(value) ? value.map(String).filter(Boolean) : []);
const object = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? value : null);
const sentence = (text) => { const body = String(text || '').trim(); return !body || /[.!?]$/.test(body) ? body : `${body}.`; };
const cap = (text) => String(text).charAt(0).toUpperCase() + String(text).slice(1);
// The first few, then how many more of `total` (a bounded list counts what it did not carry).
const named = (items, limit = 3, total = items.length) => {
    const shown = items.slice(0, limit);
    return shown.join(', ') + (total > shown.length ? ` and ${total - shown.length} more` : '');
};

// The owner's words for the codes the server records (body_switch.qualify_return, finalize_on_boot); a code this
// table does not know stays as written, so no reason is lost. The exact codes stay in the Raw record.
const ENTRY_WORDS = { server: 'its server', app: 'its desktop app', launcher: 'its launcher', other: 'another process' };
const CODE_WORDS = {
    supervisor_not_ready: 'the supervisor did not become ready',
    native_host_unconfirmed: 'the desktop app did not confirm the new version',
    panic_stop: 'Panic stop is on',
    dependencies_changed: 'the Python packages changed',
    runtime_changed: 'the Python or SQLite runtime changed',
    engine_pin_changed: 'the engine version changed',
    engine_floor_unproven: 'an engine whose data only moves forward ran here, so the previous version may not read it',
    environment_switched: 'the Python packages were switched to the new version’s own',
    checkout_moved: 'the checkout moved after the switch',
    composition_unread: 'what the checkout holds now could not be read',
    composition_unrecorded: 'the switched files were not recorded, so a later edit cannot be told apart',
    later_edits: 'a later edit sits at a switched path',
    previous_objects_missing: 'files of the previous version are missing',
    successor_entered: (kind) => `the new version had already started ${ENTRY_WORDS[kind] || kind}, so it may have `
        + 'changed data',
    successor_changed_data: 'the new version changed data the previous one reads first',
    data_unobserved: 'what the new version did to the data was not recorded',
    successor_launcher_ran: 'a changed launcher ran',
    import_probe_failed: 'the new version did not import',
    environment_selection_failed: 'the system refused to switch the Python packages',
    successor_start_failed: 'the new version did not become ready',
    loaded_tree_unattested: 'this start did not verify the switched files before importing them',
    return_requested: 'a return to the previous version was requested',
    native_switch_failed: 'switching to the new app failed',
    no_app_bundle: 'no macOS desktop app was found for this installation',
    origin_custom: 'this app is not the official signed release, so it is never replaced automatically',
    origin_unknown: 'the origin of this app could not be read, so it is not replaced automatically',
    protocol_differs: 'this app hosts a different entry than the new version needs',
    protocol_unread: 'which entry this app hosts could not be read',
    entry_unread: 'whether this app has its entry could not be read',
    no_published_package: 'no published app hosts what the new version needs yet',
    signed_entry_missing: 'this app has no signed entry for the current source, so its restarts lose the app’s '
        + 'profile and notification identity',
    release_assets_unread: 'the published apps could not be read',
    exchange_unsupported: 'the app’s volume cannot swap two apps in one step',
    staged_runtime_unread: 'the new app’s Python did not run',
    environment_venv: 'this installation runs its packages in a virtual environment that is not its launcher’s '
        + '--runtime-venv, so the new app would not start with it',
    app_changed: 'another installer (such as Homebrew) replaced an app on disk just before the swap, so the apps were '
        + 'not exchanged',
    environment_system: 'this installation’s packages are not the app’s own, which the new app does not start with',
    app_not_exchanged: 'the new app was never put in place',
    native_switch_interrupted: 'switching to the new app was interrupted',
    exchange_failed: 'the new app could not be put in place',
    entry_missing: 'the new app has no entry to run the update',
    runtime_differs: 'the new app runs another Python version',
    download_failed: 'the new app could not be downloaded',
    channel_changed: 'the update channel changed after it was prepared',
};
function codeWords(code) {
    const text = String(code || '');
    const cut = text.indexOf(':');
    const head = cut < 0 ? text : text.slice(0, cut);
    const words = CODE_WORDS[head];
    const detail = cut < 0 ? '' : text.slice(cut + 1).trim();
    if (typeof words === 'function') return words(detail);
    if (!words) return text;
    return detail ? `${words} (${detail})` : words;
}

/** The candidate's target as a sentence's subject and as a mention inside one: the version its preparation pinned
 *  (a status without it: the latest check's, when it names the same commit), else its short commit id. A channel that
 *  moved on never renames it. Empty without a pinned target. */
export function candidateName(data = {}) {
    const candidate = object(data.candidate) || {};
    const sha = String(candidate.target_sha || '');
    if (!sha) return { subject: '', mention: '' };
    const version = String(candidate.target_version || '')
        || (String(data.latest_sha || '') === sha ? String(data.latest_version || '') : '');
    return version ? { subject: versioned(version, sha), mention: versioned(version, sha) }
        : { subject: `Update ${short(sha)}`, mention: `update ${short(sha)}` };
}

const MERGE_WORDS = {
    current: 'this checkout already has it: only the desktop app changes, the files stay exactly as they are',
    clean: 'merged with this installation’s own commits, without conflicts',
    fast_forward: 'fast-forward: no commits of this installation to merge',
    conflicting: 'conflicts with this installation’s own commits',
};

// A test file the official version requires (its floor) that the prepared update lacks is refused before any test runs
// (update_checks.selected_tests: `floor_missing`, also in `blocking`): it is missing, never a test that failed.
const lacking = (missing, limit) => `the update lacks test file(s) the official version requires: ${named(missing, limit)}`;

function checksText(candidate) {
    const checks = object(candidate.checks) || {};
    const tests = object(checks.tests) || {};
    if (candidate.state === 'checking') return 'running in the separate copy';
    if (checks.ok === true) {
        if (checks.skipped) return `not run (${checks.skipped})`;
        return Number.isInteger(tests.file_count) ? `passed: ${tests.file_count} selected test file(s)` : 'passed';
    }
    if (checks.ok === false) {
        if (tests.unverified) return 'did not complete, so nothing was verified';
        const missing = strings(tests.floor_missing);
        if (missing.length) return `not run: ${lacking(missing)}`;
        const blocking = strings(tests.blocking);
        return blocking.length ? `failed only with the update: ${named(blocking)}` : 'failed';
    }
    return '';
}

// The owner's work as the restart leaves it (update_prepare.carried_layers): what each layer changes on the update.
const STAGED_WORDS = { A: 'new, staged', D: 'deleted, staged' };  // otherwise staged
const FILES_WORDS = { A: 'new file', D: 'deleted' };  // otherwise edited
const CARRIED_ROWS = 10;  // listed as rows; the rest of the recorded paths follow in one block

/** Each carried path once, with what its layers do to it ("staged, then edited"), path order. */
function carriedPaths(carried) {
    const byPath = new Map();
    for (const [layer, words, otherwise] of [['index', STAGED_WORDS, 'staged'], ['files', FILES_WORDS, 'edited']]) {
        for (const entry of Array.isArray(object(carried[layer])?.paths) ? carried[layer].paths : []) {
            if (!Array.isArray(entry) || entry.length < 2) continue;
            const [status, path] = [String(entry[0]), String(entry[1])];
            byPath.set(path, [...(byPath.get(path) || []), words[status] || otherwise]);
        }
    }
    return [...byPath].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([path, done]) => ({ path,
        words: done.join(', then ') }));
}

/** How many carried files a layer count proves: every layer recorded in full, else null (unknown, never zero). */
function carriedCount(carried) {
    const complete = ['index', 'files'].every((layer) => Number.isInteger(object(carried?.[layer])?.count)
        && carried[layer].count === (Array.isArray(carried[layer].paths) ? carried[layer].paths.length : -1));
    return complete ? carriedPaths(carried).length : null;
}

/** Each layer's exact change count, `{staged, files}`, or null when a layer's count is not recorded. */
function carriedTotals(carried) {
    const [staged, files] = ['index', 'files'].map((layer) => object(carried?.[layer])?.count);
    return Number.isInteger(staged) && Number.isInteger(files) ? { staged, files } : null;
}

const CARRIED_MORE = 20;  // after the rows, at most this many more lines; the complete list is a download

/** Where the complete list beside the candidate's copy downloads (update_prepare.carried_list_path, linked by the
 *  gateway's update_activation.candidate_view by this candidate's id): only that download, never a path. */
function carriedList(carried) {
    const url = String(carried.list_url || '');
    return { url: url.startsWith('/api/files/download?carried_list=') ? url : '', path: String(carried.list || '') };
}

/** The rows of "Uncommitted after the restart": the first paths as rows, a few more in one block, and when the
 *  panel lists less than all, the exact totals with the complete list as its download. */
function carriedRows(candidate) {
    const carried = object(candidate.carried);
    const paths = carried ? carriedPaths(carried) : [];
    if (!paths.length) return [];
    const rows = [{ head: 'Uncommitted after the restart' },
        ...paths.slice(0, CARRIED_ROWS).map(({ path, words }) => ({ label: path, value: words }))];
    const rest = paths.slice(CARRIED_ROWS);
    // A layer recorded in part (update_prepare._CARRIED_LIMIT) is complete only in the list beside the copy.
    const partial = ['index', 'files'].some((layer) => (Number(object(carried[layer])?.count) || 0)
        > (Array.isArray(object(carried[layer])?.paths) ? carried[layer].paths.length : 0));
    const shown = rest.slice(0, CARRIED_MORE);
    const fewer = rest.length > shown.length || partial;
    if (shown.length) {
        rows.push({ label: fewer ? 'More' : `${rest.length} more`, code: true,
            value: shown.map(({ path, words }) => `${path}  ${words}`).join('\n') });
    }
    if (!fewer) return rows;
    const totals = carriedTotals(carried);
    const all = totals ? `${totals.staged + totals.files} change(s): ${totals.staged} staged, ${totals.files} in the `
        + 'files' : '';
    const list = carriedList(carried);
    if (list.url) {
        rows.push({ label: 'In all', value: `${all || 'More changes than shown'}.`,
            link: { href: list.url, name: list.path.split('/').pop(), text: 'Download the full list' } });
    } else {
        // No list beside the copy (a record made before it, or one already gone): only what the record holds.
        rows.push({ label: 'In all', value: all ? `${all}; the ones not shown are not recorded here.`
            : 'More changes than shown; they are not recorded here.' });
    }
    return rows;
}

const CONFLICT_LABELS = { merge: 'Conflicts with your local commits', index: 'Conflicts in your staged changes',
    files: 'Conflicts in your uncommitted files' };

/** The candidate's facts for its "This update" list: `{label, value, code?}` rows and `{head}` group headings. */
export function candidateFacts(data = {}, candidate = object(data.candidate)) {
    if (!candidate?.candidate_id) return [];
    const rows = [];
    const add = (label, value, extra = {}) => { if (value) rows.push({ label, value: String(value), ...extra }); };
    const checks = object(candidate.checks) || {};
    const carrying = ['checking', 'ready', 'activating'].includes(candidate.state);
    const files = carriedCount(object(candidate.carried));
    const totals = carriedTotals(object(candidate.carried));
    add('Prepared from', versioned(candidate.base_version, candidate.base_sha));
    add('Target', [versioned(candidate.target_version, candidate.target_sha),
        candidate.target_ref && `on ${candidate.target_ref}`].filter(Boolean).join(' '));
    add('Combined', MERGE_WORDS[candidate.merge_kind]);
    add('Uncommitted changes', !candidate.local_changes ? 'none'
        : candidate.without_local_changes
            ? `left out of this update; kept in Git at refs/ouroboros/update/${candidate.candidate_id}/k2`
            : carrying
                ? (files ? `${files} file(s) carried over, still uncommitted`
                    // Recorded in part: the exact change counts, never a guessed number of files.
                    : totals && totals.staged + totals.files ? `${totals.staged + totals.files} change(s) carried over `
                        + `(${totals.staged} staged, ${totals.files} in the files), still uncommitted`
                        : 'carried over (staged, unstaged and untracked), still uncommitted')
                : 'present; they stay in this checkout as they are');
    const native = object(candidate.native) || {};
    // A new app always runs on its own copy of the packages, prepared by its own Python, whatever the lock says.
    add('Python packages', native.status === 'staged' ? (native.runtime_venv
        ? `its own copy of the virtual environment ${native.runtime_venv}, which the new app starts with`
        : 'its own copy, prepared by the new app’s Python')
        : candidate.environment_same === true ? 'the same as the running app'
        : candidate.environment_same === false ? sentence(checks.environment_note || 'the dependency lock differs') : '');
    add('Checks', checksText(candidate));
    const tests = object(checks.tests) || {};
    add('Already failing before', named(strings(tests.already_failing), 6));
    add('Also failing in the official version', named(strings(tests.target_failing), 6));
    add('Your tests no longer in it', named(strings(tests.dropped_local_tests), 6));
    const uncovered = strings(tests.uncovered);  // the record keeps the first 50 and the exact count
    add('No automatic test covers', named(uncovered, 6, Math.max(uncovered.length, Number(tests.uncovered_count) || 0)));
    // An open conflict names the layer it is in: the merge with this installation's commits, or a layer of its own.
    for (const [step, paths] of Object.entries(object(candidate.layer_conflicts) || {})) {
        const listed = strings(paths);
        const more = Number(candidate.conflict_count) - listed.length;
        if (listed.length) {
            rows.push({ label: CONFLICT_LABELS[step] || `Conflicts (${step})`, code: true,
                value: [...listed, ...(more > 0 ? [`and ${more} more`] : [])].join('\n') });
        }
    }
    add('Desktop app', native.status === 'staged'
        ? `${native.previous_version ? `${native.previous_version} → ` : ''}${native.version || 'the new app'}: `
            + `${native.sha256 ? 'its checksum matches the release and ' : ''}its signature is the official one; it `
            + 'replaces this app at the restart'
        : native.status === 'manual' ? `${native.version || 'a new app'} is needed: ${codeWords(native.reason)}` : '');
    if (native.status === 'staged') {
        add('App location', native.installed);
        add('Previous app kept at', native.backup_app);
    }
    const adaptation = object(candidate.adaptation) || {};
    add('Adapted under review', adaptation.task_id && `task ${adaptation.task_id}${adaptation.resolution
        ? `, reviewed commit ${short(adaptation.resolution)}` : ''}`);
    add('Recorded', attemptTime(candidate.updated_at));
    if (carrying && !candidate.without_local_changes) rows.push(...carriedRows(candidate));
    return rows;
}

function startFailureFacts(failure, { head = true } = {}) {
    const rows = head ? [{ head: 'Last start' }] : [];
    const add = (label, value, extra = {}) => { if (value) rows.push({ label, value: String(value), ...extra }); };
    add('Failed', failure.cause === 'import_failed' ? 'this checkout’s code did not import' : failure.cause);
    add('These files', { healthy: 'started successfully before', unverified: 'never reached a successful start' }[failure.composition]);
    if (strings(failure.paths).length) rows.push({ label: 'Kept in place', value: strings(failure.paths).join('\n'), code: true });
    add('Rescue copy', failure.rescue);
    add('When', attemptTime(failure.at));
    if (failure.detail) rows.push({ label: 'Import error', value: String(failure.detail), code: true });
    return rows;
}

// What the owner's local work becomes in the prepared update, as one sentence (status `local_changes` names the
// checkout's own committed content; the candidate's `local_changes` its staged, unstaged and untracked work).
function carriedText(data, candidate) {
    // The merge counts the commits it keeps; without that count the status says whether there are any. An app-only
    // Update merges nothing.
    const counted = Number.isInteger(candidate.local_commits) ? candidate.local_commits : null;
    const commits = candidate.merge_kind !== 'current' && (counted === null ? data.local_changes === 'present' : counted > 0);
    const yourCommits = counted ? `Your ${counted} local commit(s) are` : 'Your local commits are';
    const files = carriedCount(object(candidate.carried));
    const yourFiles = files ? `your ${files} uncommitted file(s) are` : 'your staged, unstaged and untracked changes are';
    if (candidate.local_changes && candidate.without_local_changes) {
        return `Your uncommitted changes are left out: after the restart they are not in your files; Git keeps them at `
            + `refs/ouroboros/update/${candidate.candidate_id}/k2.${commits ? ` ${yourCommits} merged into it.` : ''}`;
    }
    if (commits && candidate.local_changes) return `${yourCommits} merged into it, and ${yourFiles} carried over and stay uncommitted.`;
    if (commits) return `${yourCommits} merged into it.`;
    if (candidate.local_changes) return `${cap(yourFiles)} carried over and stay uncommitted.`;
    return '';
}

/** The available update's hint while preparation is the next step (no candidate yet). */
export function prepareHint(data = {}) {
    const candidate = object(data.candidate) || {};
    const latest = data.latest_version || short(data.latest_sha) || 'the update';
    const files = Number(data.dirty_count) || 0;
    const commits = data.local_changes === 'present';
    const refusal = String(object(candidate.refusal)?.text || '');
    const parts = refusal ? [`The last preparation was refused: ${sentence(refusal)}`] : [];
    if (data.local_changes === 'unknown') parts.push('Could not compare this checkout with the official line.');
    if (commits && files) {
        parts.push(`Your local commits and ${files} uncommitted file(s) will be carried into ${latest}. `
            + 'Uncommitted files stay uncommitted.');
    } else if (commits) {
        parts.push(`Your local commits will be carried into ${latest}.`);
    } else if (files) {
        parts.push(`Your ${files} uncommitted file(s) will be carried into ${latest} and stay uncommitted.`);
    } else {
        parts.push(`Ouroboros prepares it in a separate copy; ${data.current_version || 'this version'} keeps running `
            + 'until you restart.');
    }
    return parts.join(' ');
}

/** A warning when the candidate record could not be read: the panel then claims nothing about it. */
export function candidateWarning(data = {}) {
    const candidate = object(data.candidate);
    if (candidate?.state !== 'unknown') return '';
    return `The prepared update could not be read${candidate.error ? `: ${candidate.error}` : '.'}`;
}

const act = (id, label, extra = {}) => ({ id, label, ...extra });

/**
 * The verdict for the prepared Update, or null when the panel's ordinary verdict applies (no candidate, none
 * prepared, or a record that could not be read). `base` carries the status chips and warnings. Each verdict adds
 * `secondary` (the quieter actions, in order) and `facts` / `factsOpen` for the "This update" list.
 */
export function candidateVerdict(data = {}, base = {}) {
    const candidate = object(data.candidate);
    if (!candidate) return null;
    const state = String(candidate.state || 'unknown');
    const activation = object(candidate.activation) || {};
    const failure = object(candidate.start_failure);
    const offered = new Set(strings(candidate.actions));
    const { subject, mention: target } = candidateName(data);
    const running = data.current_version ? String(data.current_version) : 'the running version';
    const Running = cap(running);
    const keeps = `${Running} keeps running. Nothing in this installation changes until you restart.`;
    const own = candidateFacts(data, candidate);
    // The facts: what its activation recorded first, then the candidate's, then this checkout's last failed start.
    const factsWith = (rows = []) => [...rows.filter(([, value]) => value).map(([label, value]) => ({ label,
        value: String(value) })), ...own, ...(failure ? startFailureFacts(failure, { head: own.length > 0 }) : [])];
    const make = (fields) => ({ ...base, facts: factsWith(), factsOpen: false, secondary: [], action: null, ...fields });
    const listed = (...ids) => ids.filter((id) => offered.has(id)).map((id) => ({
        prepare: act('prepare', 'Prepare again'),
        apply_without_local_changes: act('apply_without_local_changes', 'Prepare without local changes'),
        discard_candidate: act('discard_candidate', 'Discard'),
        activate: act('activate', 'Restart to update'),
    }[id]));
    const [again, ...rest] = listed('prepare', 'apply_without_local_changes', 'discard_candidate');
    // The version the update started from, by its own record: what a return runs again and Restore goes back to.
    const previousName = versioned(candidate.base_version, candidate.base_sha);
    const previousVersion = String(candidate.base_version || '') || 'the previous version';
    const name = subject || 'The update';
    // What happened at the restart is the Last attempt's (activationAttemptView): its why, its codes and its record.

    if (state === 'activating' && activation.outcome === 'unconfirmed') {
        // The boot's observation (body_adoption.finalize_on_boot): switched, never attested. Neither done nor failed.
        return make({ state: 'candidate_unconfirmed', tone: 'neutral',
            headline: `${name} is in place, but its start is unconfirmed.`,
            hint: 'This start did not verify the switched files before it imported them, so the update is counted '
                + 'neither done nor failed; a start that verifies them settles it. Nothing was rolled back.' });
    }
    if (state === 'activating' && activation.outcome === 'returning') {
        return make({ state: 'candidate_returning', tone: 'neutral',
            headline: `${name} did not start; Ouroboros is returning to ${previousName || 'the previous version'}…`,
            hint: 'Its compatibility with your current data is proven, so it restarts into it. Data is not rolled back.' });
    }
    if (activation.outcome === 'returned' && state !== 'ready') {  // a ready one returned before it ran: it may retry
        const local = Boolean(candidate.local_changes) || Number(candidate.local_commits) > 0;
        return make({ state: 'candidate_returned', tone: 'error',
            headline: `Ouroboros is back on ${previousName || 'the previous version'}. ${name} did not start.`,
            hint: [RETURNED_BEFORE_IT_RAN.has(String(activation.why || object(activation.decision)?.trigger || ''))
                ? 'It switched back before the new version ran, so your data is as it was.'
                : `It switched back on its own because ${previousVersion} was proven to read your current data; `
                    + 'nothing was rolled back.', local ? 'Your local changes are back as they were.' : '']
                .filter(Boolean).join(' '),
            action: again || null, secondary: rest });
    }
    if (activation.outcome === 'start_failed') {
        const previous = String(activation.previous || '');
        return make({ state: 'candidate_start_failed', tone: 'error',
            headline: `${name} did not start correctly. Nothing was rolled back.`,
            hint: `Ouroboros could not prove that ${previousVersion} can run on what is here now, so it did not `
                + 'switch back on its own. Your data and local changes are kept as they are.',
            action: previous ? act('restore', 'Restore previous version', { target: previous, name: previousName })
                : act('recovery', 'Choose a version to restore') });
    }
    if (failure) {
        const paths = strings(failure.paths);
        const why = failure.composition === 'healthy'
            ? 'These files started successfully before, so something outside them changed: the Python environment, '
                + 'the data or the system. Nothing was reset.'
            : failure.composition === 'unverified'
                ? `Local changes that never reached a successful start break it (${paths.length} path(s)). They stay in `
                    + `place${failure.rescue ? ` and were copied to ${failure.rescue}` : ''}; the last commit was not `
                    + 'started without them.'
                : '';
        return make({ state: 'start_failed', tone: 'error', factsOpen: true, factsTitle: own.length ? '' : 'Last start',
            headline: 'The last start failed: this checkout’s code did not import.',
            hint: [why, 'Restore starts a version you choose.'].filter(Boolean).join(' '),
            action: act('recovery', 'Choose a version to restore') });
    }
    const needed = object(candidate.app_needed);
    if (state === 'none' && needed && !data.available && offered.has('prepare')) {  // the source is current, the app is not
        return make({ state: 'app_needed', tone: 'warn',
            headline: `This desktop app${needed.installed_version ? ` (${needed.installed_version})` : ''} needs the `
                + 'official app that runs the current version.',
            hint: `${sentence(cap(codeWords(needed.reason)))} Prepare downloads and checks it beside this app; nothing `
                + 'changes until you restart, and your files stay exactly as they are.',
            action: act('prepare', 'Prepare app update') });
    }
    if (state === 'preparing') {
        return make({ state: 'candidate_preparing', tone: 'neutral',
            headline: target ? `Preparing ${target}…` : 'Preparing the update…', hint: keeps });
    }
    if (state === 'checking') {
        return make({ state: 'candidate_checking', tone: 'neutral', headline: `Checking ${target || 'the update'}…`,
            hint: `Combined with this installation; its checks run in its own copy. ${keeps}` });
    }
    if (state === 'ready') {
        const newer = Boolean(candidate.newer_available);
        const latest = data.latest_version || short(data.latest_sha) || 'the newer version';
        const carried = carriedText(data, candidate);
        const activate = listed('activate');
        const staged = object(candidate.native)?.status === 'staged'
            ? `Restart to update also replaces this desktop app${candidate.native.previous_version
                ? ` (${candidate.native.previous_version})` : ''} with ${candidate.native.version || 'the new app'}; the `
                + 'previous app stays beside it.' : '';
        const appOnly = candidate.merge_kind === 'current' && staged;
        // Its last Restart to update switched back before the new version ran (its packages could not be moved).
        const decision = object(activation.decision) || {};
        const retry = activation.outcome === 'returned' ? `The last restart did not switch: ${codeWords(decision.trigger
            || activation.why)}${decision.detail ? ` (${decision.detail})` : ''}. Nothing changed. If a program is using `
            + 'that folder, close it; Restart to update tries again.' : '';
        return make({ state: 'candidate_ready', tone: retry ? 'warn' : 'ok',
            headline: appOnly ? `${candidate.native.version ? `The desktop app ${candidate.native.version}`
                : 'The new desktop app'} is ready. Restart to switch to it.`
                : `${subject || 'The update'} is ready${carried && !candidate.without_local_changes
                    ? ' with your local changes' : ''}. Restart to switch to it.`,
            hint: [retry, carried, staged, newer ? `A newer version is available: ${latest}${data.latest_short_sha
                && data.latest_version ? ` (${data.latest_short_sha})` : ''}.` : '', `If it does not start, ${running} comes `
                + 'back on its own only when it is proven to read your current data; otherwise everything stays as it is '
                + 'for recovery.'].filter(Boolean).join(' '),
            action: newer ? act('prepare', `Prepare ${latest}`) : (activate[0] || null),
            secondary: newer ? activate : listed('discard_candidate') });
    }
    if (state === 'needs_adaptation') {
        // The step of the open conflict (the projection keys its paths by it) says which of the owner's layers it is.
        const layers = object(candidate.layer_conflicts) || {};
        const step = String(object(candidate.adaptation)?.step || Object.keys(layers)[0] || '');
        const paths = strings(layers[step] || candidate.conflict_paths);
        const what = { merge: 'Your local commits', index: 'Your staged changes', files: 'Your uncommitted changes' }[step]
            || 'Your local changes';
        const without = offered.has('apply_without_local_changes') ? ', or prepare without your uncommitted changes' : '';
        return make({ state: 'candidate_needs_adaptation', tone: 'warn', factsOpen: true,
            headline: `${what} conflict with ${target || 'the update'}. Nothing was changed.`,
            hint: `${paths.length ? `Conflicting: ${named(paths, 3, Number(candidate.conflict_count) || paths.length)}. `
                : ''}${Running} keeps running with your local changes. Resolve the conflict in this checkout and `
                + `prepare again${without}.`,
            action: again || null, secondary: rest });
    }
    if (state === 'adapting') {
        const task = object(candidate.adaptation)?.task_id;
        const paths = strings(candidate.conflict_paths);
        return make({ state: 'candidate_adapting', tone: 'neutral',
            headline: `Adapting your local changes to ${target || 'the update'}…`,
            hint: `A reviewed task resolves ${paths.length ? named(paths, 3, Number(candidate.conflict_count)
                || paths.length) : 'the conflict'} in the separate copy`
                + `${task ? ` (task ${task})` : ''}. ${keeps}`,
            secondary: listed('discard_candidate') });
    }
    if (state === 'needs_decision') {
        const native = object(candidate.native) || {};
        const app = String(candidate.reason || '').startsWith('native_app:');
        // The manual instruction is the owner's way on; without one, the reason itself is the fact.
        const said = String(native.instruction || candidate.question || '').trim();
        return make({ state: 'candidate_needs_decision', tone: 'warn', factsOpen: true,
            headline: app ? `${subject || 'The update'} needs another desktop app. Nothing was changed.`
                : 'This update needs your decision. Nothing was changed.',
            hint: [sentence(cap(said || (app ? codeWords(native.reason || String(candidate.reason).slice('native_app:'.length).trim())
                : ''))), app ? 'Then prepare it again.' : ''].filter(Boolean).join(' '),
            action: again || null, secondary: rest });
    }
    if (state === 'paused_budget') {
        return make({ state: 'candidate_paused', tone: 'warn',
            headline: `Preparing ${target || 'the update'} stopped at the spending limit. Nothing was changed.`,
            hint: `The review of its adaptation waits for budget. ${Running} keeps running; prepare it again once the limit allows.`,
            action: again || null, secondary: rest });
    }
    if (state === 'stale') {
        // An app staged by an earlier preparation was recorded without its contents (body_switch.identity_stale).
        const earlier = String(candidate.reason || '') === 'native_identity_stale';
        return make({ state: 'candidate_stale', tone: 'warn',
            headline: earlier ? 'This update was prepared by an earlier version of Ouroboros.'
                : 'This checkout changed after the update was prepared.',
            hint: earlier ? `Prepare it again so the app it brings is checked as this version checks apps. Nothing was `
                + `changed; ${running} keeps running.`
                : `Prepare it again to carry the current files. Nothing was changed; ${running} keeps running.`,
            action: again || null, secondary: rest });
    }
    if (state === 'activating') {
        return make({ state: 'candidate_activating', tone: 'neutral',
            headline: `Switching to ${target || 'the update'}…`,
            hint: 'Ouroboros restarts to switch to it. This page updates itself when the server is back.' });
    }
    if (state === 'failed') {
        const reason = String(candidate.reason || '');
        const tests = object(object(candidate.checks)?.tests);
        const blocking = strings(tests?.blocking);
        const missing = strings(tests?.floor_missing);
        const collision = reason.startsWith('ignored_path_collision:') ? reason.slice(reason.indexOf(':') + 1).trim() : '';
        if (reason.startsWith('activation:')) {  // the restart's other outcomes; the Last attempt keeps the record
            const ended = STARTED_WORDS[activation.outcome] || STARTED_WORDS.unknown;
            return make({ state: 'candidate_failed', tone: 'error', headline: ended.headline(name),
                hint: ended.hint(running), action: again || null, secondary: rest });
        }
        // A typed refusal is recorded as "CODE: its sentence": the sentence is the owner's, the code stays in the record.
        const refusal = /^[A-Z][A-Z0-9_]+: ([\s\S]+)$/.exec(reason);
        const checked = reason === 'checks_failed' || reason === 'checks_unverified';
        const hint = reason === 'checks_failed'
            ? (missing.length ? `No test was run: ${lacking(missing, 5)}.`
                : blocking.length ? `Tests that fail only with the update: ${named(blocking, 5)}.` : 'Its selected tests did not pass.')
            : reason === 'checks_unverified' ? 'Its checks did not complete, so it was neither passed nor failed. Prepare it again.'
            : collision ? `It adds files where this checkout keeps ignored files: ${collision}.`
                : sentence(refusal ? refusal[1] : codeWords(reason));
        return make({ state: 'candidate_failed', tone: 'error', factsOpen: checked,
            headline: reason === 'checks_failed' ? `${name} did not pass its checks. Nothing was changed.`
                : reason === 'checks_unverified' ? `${name} could not be verified. Nothing was changed.`
                : `${name} could not be prepared. Nothing was changed.`,
            hint, action: again || null, secondary: rest });
    }
    return null;  // none, activated, unknown or a state this panel does not know: the ordinary verdict speaks
}

// A return that happened before anything of the new tree ran (body_switch: the import probe, the app installer).
const RETURNED_BEFORE_IT_RAN = new Set(['import_probe_failed', 'app_not_exchanged', 'native_switch_interrupted',
    'native_switch_failed', 'exchange_failed', 'app_changed', 'environment_selection_failed']);
// The restart's outcomes that leave a failed candidate besides a return and a failed start.
const STARTED_WORDS = {
    not_applied: { headline: (name) => `${name} was not switched to. Nothing was changed.`,
        hint: (running) => `The restart ended before the switch, so ${running} still runs. Prepare it again to retry.` },
    superseded: { headline: (name) => `${name} was switched to, then this checkout moved to another commit.`,
        hint: () => 'Ouroboros does not count it as the running version; Last attempt keeps the record.' },
    unknown: { headline: (name) => `${name} did not finish starting.`, hint: () => 'Last attempt keeps the record.' },
};

/**
 * The last Restart to update's settled outcome as the panel's Last attempt (`candidate.last_activation`), in the
 * shape of update_status.js::updateAttemptView: `{id, when, tone, headline, text, details, raw, restoreTarget,
 * restoreName, open, terminal}`, or null. Names come from the record (its versions and commits), never from the
 * channel now; the raw record keeps every code.
 */
export function activationAttemptView(data = {}) {
    const live = object(data.candidate) || {};
    const observed = object(live.activation) || {};
    if (live.state === 'activating' && observed.outcome === 'unconfirmed') {
        // The boot's observation on the live record (an admission cleared any older retained outcome): neutral, with
        // the record one disclosure away; a start that attests the switched files settles it either way.
        const words = attemptOutcome(activationOutcomeKey('unconfirmed'));
        const rows = [['Why', codeWords(observed.why)], ['Update', versioned(live.target_version, live.target_sha)],
            ['Previous version', versioned(live.base_version, live.base_sha)], ['Your data', 'not rolled back']]
            .filter(([, value]) => value).map(([label, value]) => ({ label, value: String(value) }));
        return { id: `activation:${live.candidate_id || ''}:unconfirmed`, terminal: true, tone: words.tone,
            when: String(observed.observed_at || live.updated_at || ''), headline: words.headline,
            text: 'The checkout is at the update; a start that verifies its files settles it.', details: rows,
            raw: JSON.stringify(observed, null, 2), restoreTarget: '', restoreName: '', open: true };
    }
    const last = object(live.last_activation);
    if (!last?.outcome) return null;
    const outcome = String(last.outcome);
    const words = attemptOutcome(activationOutcomeKey(outcome));
    const newName = versioned(last.target_version, last.target_sha) || 'the update';
    // A return runs the previous commit again (``sha``); otherwise ``previous`` is the commit the update replaced.
    const previous = String((outcome === 'returned' ? last.sha : last.previous) || '');
    const oldName = versioned(last.base_version, previous) || 'the previous version';
    const native = object(last.native) || {};
    const rows = [];
    const add = (label, value) => { if (value) rows.push({ label, value: String(value) }); };
    const why = codeWords(last.why || last.trigger);
    let headline = words.headline;
    let text = '';
    if (outcome === 'adopted') {
        const appOnly = last.merge_kind === 'current' && native.version;
        const withWork = last.without_local_changes && last.local_changes ? ' without your uncommitted changes'
            : last.local_changes || Number(last.local_commits) > 0 ? ' with your local changes' : '';
        headline = appOnly ? `The desktop app was updated: ${native.previous_version || 'the previous app'} → ${native.version}.`
            : `Updated ${last.base_version || 'the previous version'} → ${last.target_version || short(last.target_sha)}`
                + `${withWork}.`;
        add('Previous version', oldName);
        add('Local commits', Number(last.local_commits) > 0 ? `${last.local_commits}, kept in history` : '');
        // The count its preparation recorded in full; without one, only that the work came along (never "none").
        add('Uncommitted files', last.without_local_changes || !last.local_changes ? ''
            : Number.isInteger(last.carried_files) ? `${last.carried_files}, still uncommitted`
                : 'carried over, still uncommitted');
        add('Left out', last.left_out ? `your uncommitted changes; Git keeps them at ${last.left_out}` : '');
        add('Desktop app', native.version ? `${native.previous_version ? `${native.previous_version} → ` : ''}`
            + `${native.version}${native.backup_app ? `; the previous app stays at ${native.backup_app}` : ''}` : '');
    } else if (outcome === 'returned') {
        add('Why', why);
        add('The system said', last.detail);
        add('Did not start', newName);
        add('Running again', oldName);
        add('Your data', 'not rolled back');
        add('Local changes', last.local_changes || Number(last.local_commits) > 0 ? 'back as they were' : '');
    } else if (outcome === 'start_failed') {
        add('Why', why);
        add('Not proven', strings(last.reasons).map(codeWords).join('; '));
        add('Previous version', oldName);
        add('Your data and local changes', 'kept, not rolled back');
    } else {
        text = outcome === 'not_applied' ? 'Restart to update ended before the switch; nothing was changed.'
            : outcome === 'superseded' ? 'The update was switched in, then this checkout moved to another commit.'
                : `The record names an outcome this panel does not know (${outcome}).`;
        add('Update', newName);
    }
    return { id: `activation:${last.candidate_id || ''}:${outcome}`, when: String(last.at || ''), terminal: true,
        tone: words.tone, headline, text, details: rows, raw: JSON.stringify(last, null, 2),
        // Restore is the explicit way back from an update that started (a failed one names it in its verdict too).
        restoreTarget: ['adopted', 'start_failed'].includes(outcome) ? previous : '', restoreName: oldName,
        open: outcome !== 'adopted' };
}

/** The confirmation for each candidate action that needs one: `openConfirmDialog` options, or null. Its quiet
 *  `details` hold the facts the action rests on, from the status and the candidate's record only. Restart to update
 *  also words the effects of `preflight`, read for this candidate (update_effects.js; unread or another's: unknown). */
export function candidateDialog(id, data = {}, preflight = null) {
    const { mention: target } = candidateName(data);
    const candidate = object(data.candidate) || {};
    const rowsOf = (pairs) => pairs.filter(([, value]) => value).map(([label, value]) => ({ label, value: String(value) }));
    if (id === 'prepare') {
        // Before a range is first prepared; the same range again, or an app alone, needs no second look.
        if (!data.available || !data.latest_sha || String(candidate.target_sha || '') === String(data.latest_sha)) return null;
        const latest = String(data.latest_version || '') || short(data.latest_sha);
        const files = Number(data.dirty_count) || 0;
        const local = data.local_changes === 'present' || files > 0;
        return { title: `Prepare ${versioned(data.latest_version, data.latest_sha)}?`, confirmLabel: 'Prepare update',
            body: `Ouroboros prepares ${latest}${local ? ' with your local work' : ''} in a separate copy and checks it `
                + `there. ${cap(data.current_version || 'this version')} keeps running, and nothing in this installation `
                + 'changes until you restart.'
                + (local ? '\n\nUncommitted files stay uncommitted; nothing is committed for you. If a change cannot be '
                    + 'carried over without changing what it does, Ouroboros stops and asks you.' : ''),
            details: { summary: 'Show details', rows: rowsOf([
                ['Range', `${short(data.current_sha) || data.current_short_sha || 'this version'} → ${short(data.latest_sha)}`
                    + `${Number(data.behind) ? `, ${data.behind} commit(s)` : ''}`],
                ['Local commits', { present: `with changes of their own since ${short(data.local_changes_base)}; merged with it`,
                    absent: 'none of their own', unknown: 'could not be compared with the official line' }[data.local_changes]],
                ['Uncommitted files', files ? `${files}, carried over and kept uncommitted` : 'none'],
                ['Conflicts', 'only what Git cannot combine goes to a reviewed task in the copy, which uses model budget'],
                ['Checks', 'selected tests run in the copy; Restart to update is offered only when they pass'],
            ]) } };
    }
    if (id === 'activate') {
        const native = object(candidate.native) || {};
        const staged = native.status === 'staged';
        const app = staged ? rowsOf([
            ['New app', native.version],
            ['This app', [native.previous_version, native.origin === 'official' ? 'from an official release' : '',
                native.installed && `at ${native.installed}`].filter(Boolean).join(', ')],
            ['Checksum', native.sha256 && `SHA-256 ${String(native.sha256).slice(0, 12)}…, matches the release`],
            ['Signature', 'valid and notarized, the official team’s'],
            // Homebrew's own record is never written: it keeps naming the version it installed.
            ['Installed by', native.manager && `${native.manager}${native.manager_version
                ? `, which installed ${native.manager_version}` : ''}; its record stays as it is`],
            ['Previous app kept at', native.backup_app],
        ]) : [];
        return { title: 'Restart to update?', confirmLabel: 'Restart to update',
            body: `Ouroboros restarts and starts ${target || 'the prepared update'}${data.local_changes === 'present'
                || (candidate.local_changes && !candidate.without_local_changes) ? ' with your local changes' : ''}. `
                + `${staged ? `This desktop app${native.previous_version ? ` (${native.previous_version})` : ''} is replaced `
                    + `with ${native.version || 'the new app'}; the previous app stays beside it. ` : ''}`
                + 'Running work is interrupted by the restart.\n\nIf it does not start, the previous version comes back '
                + 'on its own only when it is proven to read the data the new start wrote; otherwise your data and local '
                + 'changes stay as they are and this panel offers recovery. Data is never rolled back.\n\n'
                + updateEffectsText(preflight, { candidate: String(candidate.candidate_id || '') }),
            ...(app.length ? { details: { summary: 'Show app details', rows: app } } : {}) };
    }
    if (id === 'apply_without_local_changes') {
        return { title: 'Prepare without your uncommitted changes?', confirmLabel: 'Prepare without them', danger: true,
            body: 'The update is prepared without your staged, unstaged and untracked changes. After Restart to update '
                + 'they are not in your files; Git keeps them, and this panel names where once the update is ready.\n\n'
                + 'Your local commits are kept. Nothing changes until you restart.' };
    }
    if (id === 'discard_candidate') {
        return { title: 'Discard the prepared update?', confirmLabel: 'Discard',
            body: 'Its separate copy is removed. This installation does not change, and you can prepare the update again.' };
    }
    return null;
}

/** Paint the verdict's buttons into the action bar: the quieter ones first, the primary last, each naming its action
 *  (`data-action`, `data-target`). The same number of quieter buttons keeps its nodes, and with them the focus. */
export function paintVerdictActions(bar, primary, verdict) {
    const secondary = verdict.secondary || [];
    let held = [...bar.querySelectorAll('[data-secondary]')];
    if (held.length !== secondary.length) {
        held.forEach((node) => node.remove());
        held = secondary.map(() => {
            const button = (bar.ownerDocument || globalThis.document).createElement('button');
            button.type = 'button';
            button.className = 'btn btn-default';
            button.dataset.secondary = '';
            bar.insertBefore(button, primary);
            return button;
        });
    }
    const paint = (button, item) => {
        button.disabled = Boolean(item.disabled);
        button.textContent = item.label;
        button.dataset.action = item.id;
        button.dataset.target = item.target || '';
        button.dataset.name = item.name || '';
    };
    secondary.forEach((item, index) => paint(held[index], item));
    primary.hidden = !verdict.action;
    if (verdict.action) paint(primary, verdict.action);
    else primary.dataset.action = '';
}

/** Paint the "This update" list (`nodes`: section, details, title, list). It is rewritten only when its rows change,
 *  so a refresh keeps a selection; the verdict opens it where it must be read, once per state (`memo`). */
export function paintCandidateFacts(nodes, verdict, rowHtml, memo) {
    const rows = verdict.facts || [];
    nodes.section.hidden = !rows.length;
    if (!rows.length) return;
    nodes.title.textContent = verdict.factsTitle || 'This update';
    const html = rows.map(rowHtml).join('');
    if (html !== memo.html) nodes.list.innerHTML = memo.html = html;
    if (verdict.state !== memo.state) {
        memo.state = verdict.state;
        nodes.details.open = Boolean(verdict.factsOpen);
    }
}
