// Restart Claudexor (Accounts): the owner's one deliberate way to replace the engine process.
//
// The engine belongs to the installation: every program that uses it has its runs on it, so
// the header Restart leaves it alone and this action says, before anything stops, how many
// runs it ends and whose. The flow is preview -> the shared in-app dialog -> restart:
//
//  * the PREVIEW (GET) is passive and typed (`ClaudexorRestartPreview`, ./claudexor_restart_types.js); a count the server
//    could not read is said as unknown, never as zero;
//  * the dialog's body is one sentence of fact plus the pin warning when there is one; the
//    engine versions and the runs themselves open under its details;
//  * the RESTART (POST) carries the preview's `token`. When what would end changed in between,
//    the server refuses with the fresh preview and the dialog asks again with those facts;
//  * the status the answer carries is committed by the SHARED store (its single writer).
//
// Pure helpers up top are node-tested without a DOM.

import { apiFetch } from './api_client.js';
import { RESTART_ENDPOINT } from './claudexor_status_store.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { absoluteTime } from './utils.js';

export const RESTART_LABEL = 'Restart Claudexor';
export const RESTART_BUSY_LABEL = 'Restarting Claudexor…';
const STALE = 'restart_preview_stale';
const DETAIL_RUNS = 12; // the runs a dialog names one by one; the rest are counted

const count = (value) => (Number.isSafeInteger(value) && value >= 0 ? value : null);
const runs = (n) => `${n} run${n === 1 ? '' : 's'}`;

/** Offered once this page has read a daemon that is ours to manage and was ever started. */
export function restartOffered(payload) {
    const daemon = payload?.daemon || {};
    return Boolean(daemon.state) && daemon.state !== 'not_provisioned' && !daemon.ownership_problem;
}

/** The one sentence of fact: how many runs a restart ends and whose they are.
 *  @param {import('./claudexor_restart_types.js').ClaudexorRestartPreview} preview */
export function restartFactSentence(preview) {
    const facts = preview?.runs || {};
    const total = count(facts.total);
    const own = count(facts.ouroboros);
    const other = count(facts.other);
    if (preview?.engine?.state === 'not_started') {
        return 'Claudexor has not been started yet, so no runs end. This starts it.';
    }
    if (facts.status === 'known' && total !== null && own !== null && other !== null) {
        return total === 0
            ? 'Claudexor has no runs right now, so restarting it ends none.'
            : `Restarting Claudexor ends ${runs(total)}: ${own} of Ouroboros, ${other} of other programs.`;
    }
    if (facts.status === 'unattributed' && total !== null) {
        return `Restarting Claudexor ends ${runs(total)}. Ouroboros could not read its own run records, `
            + 'so it cannot say whose they are.';
    }
    const reason = preview?.engine?.state === 'serving'
        ? 'Claudexor did not list its runs' : 'Claudexor is not answering, so its runs could not be counted';
    const mine = own === null ? '' : ` Ouroboros's own records hold ${runs(own)} open.`;
    return `${reason}. If it is running, restarting ends every run on it, other programs' included.${mine}`;
}

const engineName = (engine) => (engine?.version
    ? `${engine.version}${engine.build_sha ? ` (${String(engine.build_sha).slice(0, 12)})` : ''}` : '');

function runRow(item) {
    const mine = item.owner === 'ouroboros';
    const since = absoluteTime(item.started_at);
    return {
        label: mine ? 'Ouroboros run' : 'Other program’s run',
        value: [item.harness, item.project, since ? `${item.state} since ${since}` : item.state,
            mine && item.kind, mine && item.task_id && `task ${item.task_id}`, item.run_id]
            .filter(Boolean).join(' · '),
    };
}

/** Everything the dialog shows for one preview. `changed`: the server refused the last one as stale.
 *  @param {import('./claudexor_restart_types.js').ClaudexorRestartPreview} preview */
export function restartConfirmCopy(preview, { changed = false } = {}) {
    const engine = preview?.engine || {};
    const facts = preview?.runs || {};
    const calls = preview?.model_calls || {};
    const warnings = (preview?.warnings || []).map((warning) => String(warning?.message || '')).filter(Boolean);
    const items = Array.isArray(facts.items) ? facts.items : [];
    const unlisted = Math.max(0, items.length - DETAIL_RUNS) + (count(facts.items_omitted) || 0);
    const serving = engine.state === 'serving' ? engineName(engine.serving) || 'version unknown'
        : engine.state === 'not_started' ? 'not started' : 'not answering';
    const rows = [
        { label: 'Serving engine', value: serving },
        { label: 'Engine this Ouroboros starts', value: engineName(engine.pin) || 'unknown' },
        ...(calls.status === 'known' ? [{ label: 'Model calls in flight',
            value: `${calls.active} active, ${count(calls.queued) ?? 'unknown'} queued; they end too` }] : []),
        ...(facts.error ? [{ label: 'Why the runs are not counted', value: String(facts.error) }] : []),
        ...items.slice(0, DETAIL_RUNS).map(runRow),
        ...(unlisted ? [{ label: 'More runs', value: `${runs(unlisted)} not listed here` }] : []),
    ];
    const ends = facts.status !== 'known' || (count(facts.total) || 0) > 0;
    return {
        title: 'Restart Claudexor?',
        body: [changed ? 'This changed while the question was open.' : '', restartFactSentence(preview), ...warnings]
            .filter(Boolean).join(' '),
        details: { summary: 'Engine and runs', rows },
        confirmLabel: RESTART_LABEL,
        danger: ends || warnings.length > 0,
    };
}

/** What the panel says after the restart answered: the stop and the start are separate facts.
 *  @param {import('./claudexor_restart_types.js').ClaudexorRestartReceipt} receipt */
export function restartOutcomeNote(receipt) {
    const error = receipt?.error ? `: ${receipt.error}` : '';
    const ended = count(receipt?.runs?.total);
    if (receipt?.stop === 'unconfirmed') {
        return receipt.serving
            ? { tone: 'warn', text: 'Claudexor did not confirm that it stopped. An engine is answering; it may be the same process.' }
            : { tone: 'error', text: `Claudexor did not confirm that it stopped, and no engine answers now${error}.` };
    }
    if (!receipt?.serving) {
        return { tone: 'error', text: receipt?.stop === 'stopped'
            ? `Claudexor stopped and did not start again${error}.` : `Claudexor did not start${error}.` };
    }
    if (receipt.stop === 'stopped') {
        return { tone: 'ok', text: ended ? `Claudexor restarted. ${runs(ended)} ended.` : 'Claudexor restarted.' };
    }
    return { tone: 'ok', text: 'Claudexor was not running. It is started now.' };
}

async function readPreview(fetchImpl) {
    const resp = await fetchImpl(RESTART_ENDPOINT);
    const data = await resp.json().catch(() => null);
    if (!resp.ok || typeof data?.token !== 'string' || !data.token) {
        throw new Error(String(data?.error || (resp.ok ? 'the answer could not be understood' : `HTTP ${resp.status}`)));
    }
    return data;
}

/**
 * The whole owner flow behind the button. Resolves the note the panel shows (`null` when the
 * owner cancelled): nothing is sent without a confirmation of the preview on screen, and a
 * refused stale preview asks again with the fresh one instead of acting on the old consent.
 * `onBusy(true|false)` brackets the request that actually restarts.
 */
export async function restartClaudexor({ store, fetchImpl = apiFetch, dialogImpl = openConfirmDialog,
    onBusy = () => {} } = {}) {
    let preview;
    try {
        preview = await readPreview(fetchImpl);
    } catch (error) {
        return { tone: 'error', text: `Could not check what a restart would end: ${error.message || error}. Nothing was stopped.` };
    }
    for (let changed = false; ; changed = true) {
        if (await dialogImpl(restartConfirmCopy(preview, { changed })) !== true) return null;
        onBusy(true);
        let outcome;
        try {
            outcome = await store.restart(preview.token);
        } finally {
            onBusy(false);
        }
        const answer = outcome?.data;
        if (answer?.restart) return restartOutcomeNote(answer.restart);
        if (outcome?.httpStatus === 409 && answer?.code === STALE && typeof answer.preview?.token === 'string') {
            preview = answer.preview;
            continue;
        }
        // A typed refusal stopped nothing; a lost or unreadable answer may hide a restart that ran.
        const refused = [400, 409, 503].includes(outcome?.httpStatus) && answer?.error;
        return { tone: 'error', text: refused
            ? `Claudexor was not restarted: ${outcome.error}`
            : `The restart did not answer: ${outcome?.error || 'request failed'}. Claudexor may have stopped; its status is above.` };
    }
}

/**
 * The panel's controller: one question and one restart at a time, the button's state and the
 * note of the last outcome (`null` until one exists, when the owner cancels, and once the
 * owner's next action in the section retires it through `clear`).
 */
export function createRestartController({ store, render = () => {}, onStart = () => {}, onSettled = () => {} } = {}) {
    const view = { asking: false, busy: false, note: null };
    async function run(deps = {}) {
        if (view.asking) return;
        view.asking = true;
        view.note = null;
        onStart();
        render();
        try {
            view.note = await restartClaudexor({ store, ...deps,
                onBusy: (busy) => { view.busy = busy; render(); } });
        } finally {
            view.asking = view.busy = false;
        }
        render();
        onSettled();
    }
    function paint(button, payload, blocked = false) {
        if (!button) return;
        button.hidden = !restartOffered(payload);
        button.textContent = view.busy ? RESTART_BUSY_LABEL : RESTART_LABEL;
        button.disabled = Boolean(view.busy || blocked);
    }
    return { run, paint, clear() { view.note = null; },
        get busy() { return view.busy; }, get note() { return view.note; } };
}
