// The first-run wizard's optional «Consciousness» row (Summary step, #1603) and the one ordinary Start it may
// send. The server offers the row only while setup has never completed here (`bootstrap.consciousnessOffer`,
// built from the completion record), with the limits consciousness runs under now. It starts unticked, lives
// in this page alone and never rides the save. Once the save is confirmed, a ticked row sends `/bg start`
// through `POST /api/command` at most once. Finish captures boot.since before saving; the command carries that
// same-boot precondition. Requests and body reads end with this page's lifetime, including when transport hangs.
// Then it reads `/api/state` until the control shows on. HTTP 200 only queues the Start; "on" is what the
// state shows. Nothing is stored and nothing is retried: a closed, reloaded or cancelled page sends nothing
// later, nor does one left after Finish and shown again from the back/forward cache, and a Start that did not
// happen is started again only from Dashboard › Consciousness.
import { apiFetch, withinRequestLifetime } from './api_client.js';
import { escapeHtmlAttr as escapeHtml, formatUsd2 } from './utils.js';
import { autonomyOf } from './consciousness_autonomy.js';

export const READY_WAIT_MS = 120_000;
export const CONFIRM_WAIT_MS = 120_000;
export const POLL_MS = 1000;
const STARTED_HOLD_MS = 1200;
const START_COMMAND = '/bg start';
const DOOR = 'Dashboard › Consciousness shows whether it is on; its Start is there.';

/** The bootstrap's offer as the row reads it; an older server's bootstrap without one offers nothing. */
export function normalizeOffer(offer) {
    const value = offer && typeof offer === 'object' ? offer : {};
    const number = (raw) => (raw === null || raw === undefined || raw === '' || !Number.isFinite(Number(raw)) ? null : Number(raw));
    return {
        offered: value.offered === true,
        enabled: typeof value.enabled === 'boolean' ? value.enabled : null,
        autonomy: String(value.autonomy || ''),
        dailyUsd: number(value.dailyUsd),
        maxTasks: number(value.maxTasks),
    };
}

/** The limits consciousness runs under now, one phrase each, in the words Settings and Costs use. */
export function limitPhrases(facts) {
    const level = autonomyOf(facts.autonomy);
    const autonomy = level ? `Autonomy: ${level.label}. ${level.note}` : 'Autonomy unknown.';
    const allowance = facts.dailyUsd === null ? 'Daily allowance unknown.'
        : facts.dailyUsd === 0 ? 'Daily allowance $0.00: it may not spend.'
            : `Up to ${formatUsd2(facts.dailyUsd)} a day in recorded spending. Subscription quota is separate.`;
    const tasks = facts.maxTasks === null ? 'Its own-task limit is unknown.'
        : facts.maxTasks === 0 ? 'It starts no tasks of its own.'
            : `At most ${facts.maxTasks} of its own tasks at once.`;
    return [autonomy, allowance, tasks];
}

/** The row: nothing when not offered; already on: the facts and the door; otherwise an unticked choice. */
export function offerRowHtml(facts, ticked = false) {
    if (!facts.offered) return '';
    const limits = `<span class="consciousness-offer-sub">${limitPhrases(facts).map(escapeHtml).join(' ')}</span>`;
    const where = '<span class="consciousness-offer-sub">Change the allowance in Dashboard › Costs. Its other settings, and Stop and Start, are in Dashboard › Consciousness.</span>';
    const value = facts.enabled === true
        ? '<span class="consciousness-offer-value">On already</span>'
        : `<label class="consciousness-offer-choice"><input type="checkbox" class="ui-checkbox" data-consciousness-start${ticked ? ' checked' : ''}>
            <span>Start consciousness with Ouroboros</span></label>
            <span class="consciousness-offer-sub">Between your requests it continues topics that interest it and you, notices opportunities and brings you suggestions.</span>`;
    return `<div class="summary-kv consciousness-offer-row"><strong>Consciousness</strong><span class="consciousness-offer">${value}${limits}${where}</span></div>`;
}

async function readJson(fetchImpl, url, signal, lifetimeMs) {
    try {
        return await withinRequestLifetime(async (transport) => {
            const response = await fetchImpl(url, { headers: { Accept: 'application/json' }, cache: 'no-store', signal: transport });
            const data = response.ok ? await response.json() : null;
            return { ok: Boolean(response.ok), status: response.status, data };
        }, { signal, ...(lifetimeMs === undefined ? {} : { lifetimeMs: Math.max(1, lifetimeMs) }) });
    } catch (error) {
        // `expired`: the lifetime this caller gave the read ran out first. That is the caller's clock, not an answer.
        return { ok: false, status: 0, data: null, expired: error?.name === 'StateReadEnded' && error.reason === 'expired' };
    }
}

async function postCommand(fetchImpl, expectedSince, signal) {
    try {
        return await withinRequestLifetime(async (transport) => {
            const response = await fetchImpl('/api/command', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: transport,
                body: JSON.stringify({ cmd: START_COMMAND, if_boot: expectedSince, ui_control: { name: 'setup_consciousness' } }),
            });
            const data = await response.json();
            return { accepted: response.ok && data?.status === 'ok', refused: !response.ok && data?.accepted === false,
                error: String(data?.error || `HTTP ${response.status}`) };
        }, { signal });
    } catch {
        return { accepted: false, refused: false, error: '' };  // abort/expiry cannot roll back a POST
    }
}

/**
 * The one ordinary Start a ticked row sends after a CONFIRMED save. Resolves `{outcome, reason, detail, sent}`:
 * outcome `started` | `not_started` | `unknown` | `cancelled`; `sent` names what reached the server:
 * `none`, `accepted` (HTTP 200: queued), `refused` (an answer that queued nothing) or `unanswered` (it may
 * have been queued). Nothing is retried. `wait(ms)` must resolve early once `cancelled()` turns true.
 */
export async function startConsciousnessAfterSave({
    fetchImpl = apiFetch, wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms)), now = Date.now,
    expectedSince = '', signal = null, cancelled = () => false, onPhase = () => {},
    readyWaitMs = READY_WAIT_MS, confirmWaitMs = CONFIRM_WAIT_MS, pollMs = POLL_MS,
} = {}) {
    const result = (outcome, reason, sent, detail = '') => ({ outcome, reason, sent, detail });
    const stopped = () => signal?.aborted || cancelled();
    if (stopped()) return result('cancelled', 'cancelled', 'none');
    if (!expectedSince) return result('not_started', 'boot_unknown', 'none');
    onPhase('waiting');
    // Each phase's window is checked before every read, and a read gets only what is left of it: a read the
    // window cut short is the window ending (not ready), while `unreachable` is a read that failed on its own.
    const readyBy = now() + readyWaitMs;
    for (;;) {
        if (stopped()) return result('cancelled', 'cancelled', 'none');
        const left = readyBy - now();
        if (left <= 0) return result('not_started', 'not_ready', 'none');
        const health = await readJson(fetchImpl, '/api/health', signal, left);
        if (stopped()) return result('cancelled', 'cancelled', 'none');
        if (health.expired) return result('not_started', 'not_ready', 'none');
        if (!health.ok) return result('not_started', 'unreachable', 'none');
        const boot = health.data?.boot || {};
        const current = String(boot.since || '');
        if (current !== expectedSince) return result('not_started', 'restarted', 'none');
        if (boot.phase === 'failed') return result('not_started', 'startup_failed', 'none', String(boot.error || ''));
        if (boot.dialogue_ready === true) break;
        await wait(pollMs);
    }
    onPhase('sending');
    if (stopped()) return result('cancelled', 'cancelled', 'none');
    const answer = await postCommand(fetchImpl, expectedSince, signal);
    if (stopped()) return result('cancelled', 'cancelled', answer.accepted ? 'accepted' : answer.refused ? 'refused' : 'unanswered');
    if (answer.refused) return result('not_started', 'refused', 'refused', answer.error);
    if (!answer.accepted) return result('unknown', 'unanswered', 'unanswered', answer.error);
    onPhase('confirming');
    const confirmBy = now() + confirmWaitMs;
    for (;;) {
        if (stopped()) return result('cancelled', 'cancelled', 'accepted');
        const left = confirmBy - now();
        if (left <= 0) return result('unknown', 'unconfirmed', 'accepted');
        const state = await readJson(fetchImpl, '/api/state', signal, left);
        if (stopped()) return result('cancelled', 'cancelled', 'accepted');
        if (state.ok && state.data?.bg_consciousness_enabled === true) return result('started', 'observed_on', 'accepted');
        await wait(pollMs);
    }
}

const PHASE_WORDS = {
    waiting: 'Waiting for Ouroboros to finish starting. Nothing is sent before it is ready.',
    sending: 'Sending the Start.',
    confirming: 'The Start was accepted. Waiting for consciousness to show as on.',
    started: 'Consciousness is on.',
};

const REASON_WORDS = {
    boot_unknown: 'The server identity could not be read when you chose Finish, so the Start was not sent.',
    not_ready: 'Ouroboros did not finish starting within two minutes, so the Start was not sent.',
    startup_failed: 'Ouroboros reported a startup failure, so the Start was not sent.',
    restarted: 'Ouroboros restarted before the Start could be sent, so nothing was sent.',
    unreachable: 'Ouroboros stopped answering before the Start could be sent, so nothing was sent.',
    left: 'You left this page before the Start could be sent, so nothing was sent.',
    refused: 'Ouroboros did not accept the Start, so nothing was queued.',
    unanswered: 'The Start was sent, but its acceptance could not be confirmed. It will not be sent again.',
    unconfirmed: 'The Start was accepted, but consciousness has not shown as on yet.',
};

/** The screen after a confirmed save: «Setup saved» first, then where the Start is, in its own words. */
export function runScreenHtml(run) {
    const done = run.phase === 'done';
    const headline = run.phase === 'started' ? 'Consciousness is on' : !done ? 'Starting consciousness…'
        : run.outcome === 'not_started' ? 'Consciousness was not started' : 'Consciousness is not confirmed';
    const words = done ? [REASON_WORDS[run.reason] || '', run.detail ? `(${run.detail})` : '', DOOR].filter(Boolean).join(' ')
        : PHASE_WORDS[run.phase] || PHASE_WORDS.waiting;
    const button = done || run.phase === 'started'
        ? '<button type="button" class="btn btn-primary" id="consciousness-open-btn">Open Ouroboros</button>'
        : `<button type="button" class="btn btn-default" id="consciousness-open-btn"${run.leaving ? ' disabled' : ''}>${run.leaving ? 'Opening…' : 'Open Ouroboros now'}</button>`;
    return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">Setup saved</h2>
                    <p class="step-copy">Your settings are on disk.</p>
                </div>
            </div>
            <div class="panel-card" data-consciousness-run="${escapeHtml(done ? run.outcome : run.phase)}">
                <h3>${escapeHtml(headline)}</h3>
                <div class="wizard-inline-note" role="status">${escapeHtml(words)}</div>
                <div class="wizard-runtime-strip">${button}</div>
            </div>
        `;
}

/** What a page left during its Start shows when it is shown again: what reached the server, and no new send. */
const afterLeaving = (sent) => (sent === 'accepted' ? { outcome: 'unknown', reason: 'unconfirmed', sent, detail: '' }
    : sent === 'unanswered' ? { outcome: 'unknown', reason: 'unanswered', sent, detail: '' }
        : sent === 'refused' ? { outcome: 'not_started', reason: 'refused', sent, detail: '' }
            : { outcome: 'not_started', reason: 'left', sent: 'none', detail: '' });

/**
 * The Summary row's controller. `mount` paints the row into its placeholder after each wizard render (the
 * tick survives renders); `afterSave` runs once the save is confirmed and calls `announce` (the wizard's
 * hand-off) only when the Start's result can be shown: at once without a tick, after the observed "on",
 * or after the owner continues past a Start that did not happen or is not confirmed. `leave` is every
 * pagehide, including entry into the back/forward cache, which keeps the page and its form alive.
 */
export function createConsciousnessOffer(offer, { onChange = () => {}, start = startConsciousnessAfterSave, hold = STARTED_HOLD_MS, fetchImpl = apiFetch } = {}) {
    const facts = normalizeOffer(offer);
    let ticked = false;
    let host = null;
    let run = null;
    let cancel = false;
    let disposed = false;
    let departed = false;  // the page was left since this Finish began: its Start is never sent later
    let expectedSince = '';
    const lifetime = new AbortController();
    let operation = null;  // the Start in progress, from the confirmed save until its outcome is known
    let used = false;  // one Start per page: a later confirmation of the same Finish never sends another
    let proceed = null;
    let wake = null;
    const wait = (ms) => new Promise((resolve) => {
        const timer = setTimeout(() => { wake = null; resolve(); }, ms);
        wake = () => { clearTimeout(timer); wake = null; resolve(); };
    });
    const change = (next) => { run = next; if (!disposed) onChange(); };

    function onToggle(event) {
        const box = event.target?.closest?.('[data-consciousness-start]');
        if (box) ticked = Boolean(box.checked);
    }

    const controller = {
        /** Capture this Finish's server before a save can lose its answer or cross a restart. Not persisted. */
        async beforeSave() {
            expectedSince = '';
            departed = false;  // a Finish chosen after coming back is a new choice; leaving before it retired nothing
            if (disposed || used || !controller.wanted()) return;
            const health = await readJson(fetchImpl, '/api/health', lifetime.signal);
            if (!disposed && health.ok) expectedSince = String(health.data?.boot?.since || '');
        },
        dispose() {
            disposed = true;
            lifetime.abort(); operation?.abort(); wake?.();
            const release = proceed; proceed = null; release?.();
            host?.removeEventListener?.('change', onToggle); host = null;
        },
        /** The page is being left: a Start not sent yet is retired for good, and the one in progress stops. A page
         * shown again keeps its settled result; only the owner's Open Ouroboros hands off. */
        leave() {
            departed = true;
            if (operation) { operation.abort(); wake?.(); }
        },
        mount(node) {
            host?.removeEventListener?.('change', onToggle);
            host = node || null;
            if (!host) return;
            host.addEventListener('change', onToggle);
            host.innerHTML = offerRowHtml(facts, ticked);
        },
        /** True only for a row the owner ticked in this page: the one Start this Finish may send. */
        wanted: () => facts.offered && facts.enabled !== true && ticked,
        running: () => run !== null,
        screenHtml: () => (run ? runScreenHtml(run) : ''),
        /** Open Ouroboros: past a settled result it hands off; during the wait it cancels what was not sent yet. */
        bindScreen(rootNode) {
            rootNode.querySelector('#consciousness-open-btn')?.addEventListener('click', () => {
                if (proceed) { const go = proceed; proceed = null; go(); return; }
                if (cancel) return;
                cancel = true;
                operation?.abort();
                wake?.();
                change({ ...run, leaving: true });
            });
        },
        snapshot: () => ({ ...facts, ticked, run: run ? { ...run } : null }),
        async afterSave({ isDisposed = () => false, announce }) {
            if (disposed || isDisposed() || used) return undefined;
            if (!controller.wanted()) return announce();
            used = true;
            let outcome = { outcome: 'cancelled', reason: 'cancelled', sent: 'none', detail: '' };  // left while saving
            if (!departed) {
                operation = new AbortController();
                change({ phase: 'waiting' });
                outcome = await start({ wait, fetchImpl, expectedSince, signal: operation.signal,
                    cancelled: () => cancel || departed || disposed || isDisposed(),
                    onPhase: (phase) => { if (!disposed && !isDisposed()) change({ ...run, phase }); } });
                operation = null;
            }
            if (disposed || isDisposed()) return undefined;
            if (outcome.outcome === 'cancelled' && departed && !cancel) outcome = afterLeaving(outcome.sent);
            if (outcome.outcome === 'started' && !cancel) {
                change({ ...outcome, phase: 'started' });
                await new Promise((resolve) => { proceed = resolve; void wait(hold).then(resolve); });
                proceed = null;
            } else if (outcome.outcome !== 'cancelled' && outcome.outcome !== 'started') {
                change({ ...outcome, phase: 'done' });
                await new Promise((resolve) => { proceed = resolve; });
            }
            if (disposed || isDisposed()) return undefined;
            wake?.();
            run = null;
            return announce();
        },
    };
    return controller;
}
