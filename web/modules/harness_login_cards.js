// Agent login cards — the host-neutral controller + view (phase 2 seam).
//
// «Красота-сначала»: the login card is LINK-FIRST — the engine disclosures the
// sign-in URL (and a one-time code for the codex device flow) through the job
// snapshot's transient overlay, and the card renders "Open sign-in link" +
// copy affordances + a live state line. A job that also accepts user input
// (the claude OAuth paste-code, disclosure flow `oauth_url_input`) shows an
// ALWAYS-VISIBLE code entry: claude's CLI only asks for the code when the
// browser cannot reach its localhost callback (e.g. another device), so the
// field cannot depend on a prompt this UI cannot observe — the input is
// OPTIONAL by design (the callback may complete on its own). The copy-paste
// `claudexor setup attach …` command survives ONLY as a collapsed Advanced
// affordance for engines that predate the disclosure modes or jobs whose link
// never arrives — never the primary UI. No terminal panel exists in this UI,
// and none may be added. Shape selection keys on the disclosure's typed FLOW
// (url-only `oauth_url`/`chatgpt` vs `oauth_url_input`; 3.3.7 final
// contract) — no harness-name branches, no boolean sidecar.
//
// This machinery used to be welded into the Providers-tab section
// (`harness_accounts.js`). It is extracted so the onboarding wizard mounts the
// SAME flow instead of reimplementing device codes, backoff and the
// verify-race — and so both surfaces inherit every fix once.
//
// TWO render modes:
//   * `full`    — exactly what Settings renders today: link/device code,
//                 the optional paste-code entry, the live state line, the
//                 verdict + the engine's own sentence, the collapsed Advanced
//                 terminal fallback (or an explicit external continuation),
//                 and Close.
//   * `compact` — Connect + progress + verified state + retry, for a wizard
//                 step. The optional paste-code entry and the delayed Advanced
//                 fallback stay full-only. Any available attach command is
//                 rendered directly in compact, while an explicitly selected
//                 external-terminal continuation is direct in BOTH modes, so
//                 neither mode strands the completion path.
//
// Pure helpers up top are node-tested without a DOM.

import { apiFetch } from './api_client.js';
import { harnessAccountIdentityMarkup } from './harness_presentation.js';
import {
    accountLoginConfirmed,
    claudexorPreparationLine,
    claudexorStatus,
    familyLabel,
} from './claudexor_status_store.js';
import { copyTextWithToast } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml, safeExternalHrefAttr } from './utils.js';

const JOB_POLL_MS = 3000;
export const JOB_POLL_MAX_DELAY_MS = 30000;
export const JOB_POLL_GIVE_UP_FAILURES = 10;

export const LOGIN_CARD_FULL = 'full';
export const LOGIN_CARD_COMPACT = 'compact';

export const TERMINAL_TRANSPORT_ERROR_CODES = Object.freeze([
    'terminal_transport_unavailable',
    'terminal_transport_unsupported',
    'terminal_transport_probe_failed',
    'terminal_transport_failed',
]);
const TERMINAL_TRANSPORT_ERRORS = new Set(TERMINAL_TRANSPORT_ERROR_CODES);

// ---------------------------------------------------------------------------
// Pure helpers.
// ---------------------------------------------------------------------------

// #125: job-poll pacing over CONSECUTIVE failures (a successful read — any
// parsed snapshot, pending included — resets the counter): 3s while healthy,
// exponential backoff 6/12/24/30…s capped at 30s while the daemon is not
// answering, and after 10 consecutive failures the chain gives up (~4 min)
// instead of hammering a dead daemon forever.
export function nextJobPollDelay(consecutiveFailures) {
    const failures = Math.max(0, Number(consecutiveFailures) || 0);
    if (failures >= JOB_POLL_GIVE_UP_FAILURES) return { delayMs: 0, giveUp: true };
    return {
        delayMs: Math.min(JOB_POLL_MS * 2 ** failures, JOB_POLL_MAX_DELAY_MS),
        giveUp: false,
    };
}

export function normalizeProfileName(raw) {
    // The ENGINE's registration contract is the authority: a profile id must
    // match ^[a-z0-9][a-z0-9_-]{0,63}$ (Claudexor profile-registration slug).
    // Lowercase, map every other character to '-', strip separators the slug
    // may not START with, cap at the engine's 64, and return '' for anything
    // that still cannot satisfy the contract (e.g. a name with no ASCII
    // alphanumerics at all) — an empty result makes the dialog/card ask again
    // instead of submitting a name the engine would refuse. Lives with the
    // login card; `harness_accounts.js` re-exports the established path.
    const mapped = String(raw || '').trim().toLowerCase().replace(/[^a-z0-9_-]/g, '-');
    const slug = mapped.replace(/^[-_]+/, '').slice(0, 64);
    return /^[a-z0-9][a-z0-9_-]{0,63}$/.test(slug) ? slug : '';
}

export function profileNameSubmission(raw) {
    // The same loop-until-stable rule `promptProfileName` (Add account)
    // applies: a submitted name STARTS a login only when it already IS its
    // normalized form; otherwise the normalized spelling is handed back,
    // editable, with a note — never rewritten silently. Pure for node tests.
    const typed = String(raw || '').trim();
    const normalized = normalizeProfileName(typed);
    if (!normalized) {
        return { profile: '', normalized: '',
            note: 'Enter a name that starts with a lowercase letter or digit — '
                + 'letters, digits, "-" and "_", at most 64 characters.' };
    }
    if (normalized !== typed) {
        return { profile: '', normalized,
            note: `"${typed}" will be saved as "${normalized}" — edit the name or press the button again to continue.` };
    }
    return { profile: normalized, normalized, note: '' };
}

export function attachShellLabel(shell) {
    if (shell === 'powershell') return 'PowerShell';
    if (shell === 'posix') return 'POSIX shell';
    return 'your terminal shell';
}

export function deviceCodeDisclosure(envelope) {
    // The transient read-time projection is canonical and envelope-level:
    // {job, cursor, sequence, deviceCode?}. It is never journaled by the daemon
    // and therefore must be read from the latest whole envelope, not recovered
    // recursively from an accidental double-wrap.
    const disclosure = envelope?.deviceCode;
    if (!disclosure || typeof disclosure !== 'object'
        || !disclosure.flow || !disclosure.verificationUrl) return null;
    return {
        url: String(disclosure.verificationUrl),
        code: String(disclosure.userCode || ''),
        flow: String(disclosure.flow),
    };
}

// Which face the login card shows, in priority order. STRUCTURED WINS: when
// the engine surfaces an OAuth device/link disclosure on ANY job the card
// renders it (link-first). There is deliberately NO attach face any more: the
// copy-paste command is a demoted, collapsed Advanced affordance under the
// waiting face (attachFallbackDue below), never a primary card body — the
// owner rejected terminal-first login outright. Pure for node tests.
export function loginCardFace(active) {
    if (!active) return 'none';
    if (active.needsProfile) return 'name';
    if (active.error) return 'error';
    if (active.verdict?.kind === 'recovery') return 'recovery';
    if (active.verdict?.kind === 'reconciled') return 'reconciled';
    if (active.verdict?.kind === 'unavailable') return 'unavailable';
    if (deviceCodeDisclosure(active.envelope || {})) return 'device';
    return 'progress';
}

function durableTerminalTransportError(active) {
    const code = active?.envelope?.job?.nativeCommand?.errorCode;
    return typeof code === 'string' && TERMINAL_TRANSPORT_ERRORS.has(code) ? code : '';
}

export function externalTerminalActionAvailable(active) {
    if (!active || active.transport === 'client_pty' || active.attachCommand) return false;
    const problemCode = String(active.problemCode || '');
    const actions = Array.isArray(active.requiredActions) ? active.requiredActions : [];
    if (TERMINAL_TRANSPORT_ERRORS.has(problemCode)) {
        return actions.includes('use_external_terminal');
    }
    return Boolean(durableTerminalTransportError(active));
}

export function loginRetryAvailable(active) {
    const code = String(active?.problemCode || '') || durableTerminalTransportError(active);
    if (code !== 'terminal_transport_unavailable'
        && code !== 'terminal_transport_unsupported') return true;
    const actions = Array.isArray(active?.requiredActions) ? active.requiredActions : [];
    return actions.includes('retry_setup_login');
}

// How long the card waits for the engine to disclose a sign-in link before
// offering the collapsed Advanced attach fallback.
export const ATTACH_FALLBACK_MS = 20000;

export function attachFallbackDue(active, nowMs) {
    // The demoted fallback: only for a job that HAS a copy-paste command (the
    // create answer issues one exactly for client_pty jobs — the POLL route's
    // per-job attach_command is deliberately never read), and only when the
    // primary path has nothing better: the engine predates the disclosure
    // modes (engineDegraded — the create answer said so, or the input route
    // 404'd), or no disclosure arrived within the window. A rendered
    // disclosure keeps the fallback hidden unless the engine is degraded:
    // link-first, always.
    if (!active || !active.attachCommand) return false;
    if (active.engineDegraded) return true;
    if (deviceCodeDisclosure(active.envelope || {})) return false;
    const started = Number(active.startedAtMs) || 0;
    return started > 0 && (Number(nowMs) - started) >= ATTACH_FALLBACK_MS;
}

export function loginInputSupport(envelope) {
    // Card shape 2 discriminator: can the user paste the browser's code back
    // into this job? The engine's typed disclosure FLOW is the structural
    // marker (3.3.7 final contract): `oauth_url_input` = sign-in link plus an
    // optional paste-code the daemon forwards (claude's manual-callback
    // path); `oauth_url`/`chatgpt` stay link-only. No boolean sidecar and no
    // harness-name fallback — the enum decides. The input is OPTIONAL by
    // design: claude's localhost callback may complete the login with no code
    // ever shown, so the field's presence never implies obligation.
    return deviceCodeDisclosure(envelope)?.flow === 'oauth_url_input';
}

// Job-failure reasons a stale post-login verification read can fabricate:
// codex clears its auth store when a login STARTS, so a probe in that window
// reads "not logged in" while the vendor login is succeeding — the owner's
// codex login SUCCEEDED while the card said "Login failed · completed". These
// verdicts are re-checked against live account status before the card is
// allowed to say "failed".
export const VERIFY_RACE_REASONS = ['capability_verification_failed', 'auth_not_ready'];

export function loginVerdict(envelope) {
    // State and verdict must never contradict: a non-terminal job has NO
    // verdict, a succeeded job is success, and a failed job is only a final
    // failure when its typed outcome reason is one no verification race can
    // fabricate. Everything else is 'recheck' — judged by live account
    // status, not by one stale read.
    const summary = jobStateSummary(envelope);
    if (!summary.terminal) return { kind: 'pending', reason: '' };
    const reason = terminationReason(envelope);
    const released = loginReleaseProven(envelope);
    if (reason === 'termination_unconfirmed') {
        return released
            ? { kind: 'reconciled', reason }
            : { kind: 'recovery', reason };
    }
    if (summary.succeeded) return { kind: 'success', reason: '' };
    const outcome = envelope?.job?.outcome || {};
    const verdictReason = String(outcome.reason || '') || summary.state;
    if (summary.state === 'failed'
        && (!outcome.reason || VERIFY_RACE_REASONS.includes(String(outcome.reason)))) {
        return { kind: 'recheck', reason: verdictReason };
    }
    return { kind: 'failure', reason: verdictReason };
}

export function failureText(reason) {
    return `Sign-in failed — ${String(reason || 'unknown reason').replace(/_/g, ' ')}.`;
}

// A bounded re-check that RAN OUT is not a failure. The verdict it re-checks
// was already judged unproven (a verification race), and the account row it
// waits for routinely lands a tick after the window closes — so the honest
// answer is "not confirmed yet", pointing at the two places the truth appears,
// never a hard "Sign-in failed" for a login that may well have succeeded.
export const UNCONFIRMED_TEXT =
    'Could not confirm the sign-in yet — check the account row above, or press Refresh.';

export function jobDetail(envelope) {
    // The engine's own SENTENCE about a settled job, beside the typed reason:
    // `message` on the canonical envelope's one bare job. The card used to
    // drop it entirely: a codex
    // login that ended `auth_not_ready` rendered only the fixed
    // UNCONFIRMED_TEXT, while the daemon had already said exactly what was
    // wrong ("native Codex session is not logged in") and that text was
    // sitting in the snapshot the card was holding. A typed reason names a
    // CATEGORY; this names the thing that happened, so it is the half the
    // owner needs and the half that reached no reader.
    const value = envelope?.job?.message;
    return typeof value === 'string' ? value.trim() : '';
}

export function loginStatusLine(envelope) {
    // The live state line for a non-terminal job — plain words mapped from
    // the typed state/phase, never raw enum spellings glued together (the old
    // "Login failed · completed" contradiction is structurally impossible
    // now: a terminal job renders a verdict, never this line).
    const summary = jobStateSummary(envelope);
    if (summary.terminal) return '';
    if (summary.phase === 'verifying') return 'Checking the sign-in…';
    if (deviceCodeDisclosure(envelope)) return 'Waiting for you to finish signing in in the browser…';
    if (summary.phase === 'awaiting_user' || summary.state === 'waiting_for_input') {
        return 'Waiting for the sign-in link…';
    }
    return 'Starting the sign-in…';
}

export async function submitLoginInput(jobId, value, { fetchImpl = apiFetch } = {}) {
    // ONE code line to the engine via the gateway proxy. Typed non-2xx
    // answers keep their meaning: 404 is a capability gap (`degraded` — the
    // engine predates the route, or reaped the job) and 409 is a typed input
    // CONFLICT (`conflict` carries the engine's code:
    // setup_input_not_applicable = the callback already completed and no code
    // is needed; setup_input_already_submitted = a repeat the server, being
    // authoritative over our double-submit guard, refused). Neither is a raw
    // error — the card maps both to friendly copy.
    try {
        const resp = await fetchImpl(`/api/claudexor/login/${encodeURIComponent(jobId)}/input`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value }),
        });
        const data = await resp.json().catch(() => ({}));
        if (resp.ok) return { ok: true, degraded: false, conflict: '', error: '' };
        if (resp.status === 404) {
            return { ok: false, degraded: true, conflict: '', error: String(data.error || 'input not supported') };
        }
        if (resp.status === 409) {
            return { ok: false, degraded: false, conflict: String(data.code || 'conflict'), error: String(data.error || 'input conflict') };
        }
        return { ok: false, degraded: false, conflict: '', error: String(data.error || `HTTP ${resp.status}`) };
    } catch (error) {
        return { ok: false, degraded: false, conflict: '', error: String(error?.message || error) };
    }
}

export const CONFIRM_POLL_MS = 2500;
export const CONFIRM_ATTEMPTS = 4;

export async function confirmLoginLive(harness, profileId, {
    store = claudexorStatus, attempts = CONFIRM_ATTEMPTS, delayMs = CONFIRM_POLL_MS,
    sleepImpl = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    isStale = () => false,
    // The read seam. It goes through the SHARED store by default, so the
    // account rows this re-check consults are the very rows on screen — one
    // request, one truth. Tests inject a direct reader.
    readStatus = null,
} = {}) {
    // The verify-race re-check: briefly re-poll live account status and let
    // IT decide, instead of declaring failure off the job's one stale
    // verification read. Bounded; answers with the last payload so the caller
    // can refresh the rows from the same reads.
    const read = typeof readStatus === 'function'
        ? readStatus
        : async () => {
            await store.refresh();
            return store.error ? null : store.snapshot;
        };
    let payload = null;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
        if (attempt > 0) await sleepImpl(delayMs);
        if (isStale()) return { confirmed: false, stale: true, payload };
        try {
            const data = await read();
            if (data) {
                payload = data;
                if (accountLoginConfirmed(data, harness, profileId)) {
                    return { confirmed: true, stale: false, payload };
                }
            }
        } catch (err) { /* transient; the next attempt retries */ }
    }
    return { confirmed: false, stale: false, payload };
}

export function resolvedJobProfileId(envelope) {
    // The profile a login job REALLY targets, read off the job's own record.
    // On a legacy engine a default login carries `profileId: null` — the
    // empty-string address the native pseudo-row answers to. On a unified
    // engine (plan §K.4) the engine resolves a default login onto its
    // bootstrap registry row and the job record names it, so the verify-race
    // must ask about THAT row: no row with the empty id exists there, and an
    // unresolved '' address would report every successful default login as
    // unconfirmed. A named login's job names the same profile the card
    // started with, so adopting the job's answer is safe on both engines.
    const raw = envelope?.job?.profileId;
    return typeof raw === 'string' ? raw : '';
}

export function pollResponseApplies(captured, current) {
    // Whether a poll answer may still be written onto the card. It belongs to
    // the job it was captured for (`captured === current`: the card may have
    // been closed, or reopened for another account, while the request was in
    // flight) and only while that job is UNSETTLED — a verdict, or the live
    // re-check that decides one, has already taken the card, and a snapshot
    // captured before the job settled would put a pending state back under it.
    // Pure so the ordering rule is asserted without a DOM or a timer.
    return Boolean(captured) && captured === current
        && !captured.verdict && !captured.confirming;
}

export function jobStateSummary(envelope) {
    // Every operation now returns the same browser envelope {job, ...metadata}.
    // There is deliberately no bare-job or nested-envelope compatibility read:
    // accepting both is what hid an accidental double-wrap and lost the
    // envelope-level device-code projection when the backend was corrected.
    const job = envelope?.job;
    const state = String(job?.state || '');
    const phase = String(job?.phase || '');
    const terminal = ['succeeded', 'failed', 'cancelled', 'timed_out',
        'interrupted_unknown', 'not_supported'].includes(state);
    return { state, phase, terminal, succeeded: state === 'succeeded' };
}

export const LOGIN_CUSTODY_RELEASED = 'released';
export const LOGIN_CUSTODY_RETAINED = 'retained';
export const LOGIN_CUSTODY_UNKNOWN = 'unknown';

function isJobEnvelope(value) {
    return Boolean(value && typeof value === 'object' && !Array.isArray(value)
        && value.job && typeof value.job === 'object' && !Array.isArray(value.job));
}

function hasJobState(value) {
    return isJobEnvelope(value) && Boolean(jobStateSummary(value).state);
}

function terminationReason(envelope) {
    return String(envelope?.job?.outcome?.reason || '');
}

/** The one proof predicate used by Start, Close, dispose, cancel and reconcile. */
export function loginReleaseProven(envelope, { absent = false } = {}) {
    if (absent) return true;
    const summary = jobStateSummary(envelope);
    if (!summary.terminal) return false;
    if (terminationReason(envelope) !== 'termination_unconfirmed') return true;
    return String(envelope?.job?.terminationReconciliation?.status || '') === 'empty';
}

function custodyResult(envelope, { absent = false, error = '', code = '', requiredActions = [] } = {}) {
    if (loginReleaseProven(envelope, { absent })) {
        return { status: LOGIN_CUSTODY_RELEASED, envelope: isJobEnvelope(envelope) ? envelope : null,
            absent, error, code, requiredActions };
    }
    if (hasJobState(envelope)) {
        return { status: LOGIN_CUSTODY_RETAINED, envelope, absent: false,
            error, code, requiredActions };
    }
    return { status: LOGIN_CUSTODY_UNKNOWN, envelope: null, absent: false,
        error, code, requiredActions };
}

/**
 * Cancel a login job and preserve the daemon's custody evidence. A 2xx alone
 * proves nothing: its canonical job body decides released versus retained;
 * malformed success and transport failures remain explicitly unknown.
 */
export async function cancelLoginJob(jobId, fetchImpl = apiFetch) {
    if (!jobId) return custodyResult(null, { absent: true });
    try {
        const resp = await fetchImpl(`/api/claudexor/login/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
        if (resp?.status === 404 || resp?.status === 410) {
            return custodyResult(null, { absent: true });
        }
        let data = null;
        try { data = await resp?.json?.(); } catch (error) { /* malformed body stays unknown */ }
        if (!resp?.ok) {
            return custodyResult(null, { error: String(data?.error || `HTTP ${resp?.status || 0}`) });
        }
        return custodyResult(data);
    } catch (error) {
        return custodyResult(null, { error: String(error?.message || error) });
    }
}

export async function reconcileLoginJob(jobId, fetchImpl = apiFetch) {
    if (!jobId) return custodyResult(null);
    try {
        const resp = await fetchImpl(`/api/claudexor/login/${encodeURIComponent(jobId)}/reconcile`, {
            method: 'POST',
        });
        if (resp?.status === 404 || resp?.status === 410) {
            return custodyResult(null, { absent: true });
        }
        let data = null;
        try { data = await resp?.json?.(); } catch (error) { /* malformed body stays unknown */ }
        const error = String(data?.error || (resp?.ok ? '' : `HTTP ${resp?.status || 0}`));
        const code = String(data?.code || '');
        const requiredActions = Array.isArray(data?.required_actions)
            ? data.required_actions.map(String) : [];
        if (resp?.status === 409) {
            return { status: LOGIN_CUSTODY_RETAINED, envelope: null, absent: false,
                error, code, requiredActions };
        }
        if (!resp?.ok) return custodyResult(null, { error, code, requiredActions });
        return custodyResult(data, { error, code, requiredActions });
    } catch (error) {
        return custodyResult(null, { error: String(error?.message || error) });
    }
}

export function loginCardHtml(active, nowMs = Date.now(), { mode = LOGIN_CARD_FULL, statusPayload = null } = {}) {
    // The card's whole body as a STRING — pure, so every face is asserted in
    // node without a DOM: the unsafe-URL refusal, the unconfirmed re-check,
    // and the rule that a verdict silences the live status line.
    if (!active) return '';
    const compact = mode === LOGIN_CARD_COMPACT;
    const summary = jobStateSummary(active.envelope || {});
    const disclosure = deviceCodeDisclosure(active.envelope || {});
    const face = loginCardFace(active);
    // NOT .panel-card: that class lives in onboarding.css, which the settings
    // pages never load — the card rendered with no frame at all (finding #4).
    const bits = [`<div class="harness-login-card" data-login-card data-login-mode="${escapeHtml(compact ? LOGIN_CARD_COMPACT : LOGIN_CARD_FULL)}">`,
        `<h4>Connect ${harnessAccountIdentityMarkup(active.harness, { label: active.familyLabel, profile: active.profile })}</h4>`];
    if (face === 'error') {
        bits.push(`<div class="settings-inline-note" data-tone="error">${escapeHtml(active.error)}</div>`);
        if (loginRetryAvailable(active)) {
            bits.push('<button type="button" class="btn btn-primary" data-login-retry>Try again</button>');
        }
    } else if (face === 'name') {
        // The engine refused to CREATE the default login and said why — its
        // sentence is the honest source of the reason and is shown verbatim.
        // The card turns that dead end into the action the refusal asks for:
        // name an account (same validation the Add-account dialog runs) and
        // start the NAMED login through the same standard flow.
        bits.push(`<div class="settings-inline-note" data-tone="error" data-login-engine-said>${escapeHtml(active.needsProfile.message)}</div>`);
        bits.push(`
            <div class="harness-code-entry" data-profile-name-entry>
                <label for="harness-profile-name-input">Name for the ${escapeHtml(active.needsProfile.familyLabel || active.harness)} account (e.g. work, backup). Lowercase letters, digits, "-" and "_" — anything else becomes "-".</label>
                <div class="harness-code-entry-row">
                    <input type="text" id="harness-profile-name-input" data-profile-name-input autocomplete="off" spellcheck="false"
                        placeholder="account name" value="${escapeHtml(active.profileNameValue || '')}">
                    <button type="button" class="btn btn-primary" data-profile-name-submit>Add account &amp; connect</button>
                </div>
                ${active.profileNameNote ? `<div class="settings-inline-status" data-tone="muted" data-profile-name-note>${escapeHtml(active.profileNameNote)}</div>` : ''}
            </div>
        `);
    } else if (face === 'device') {
        // LINK-FIRST: the prominent action is opening the disclosed sign-in
        // URL, with an explicit copy affordance (the macOS-app card pattern).
        // That link is now a PRIMARY click target built from engine-supplied
        // text, so it goes through the house helper: http/https only, escaped
        // by the helper itself. Anything else — javascript:, data:, a
        // malformed string — yields '' and renders NO clickable link at all.
        const href = safeExternalHrefAttr(disclosure.url);
        bits.push(href ? `
            <div class="harness-signin-actions">
                <a class="btn btn-primary" data-open-signin href="${href}" target="_blank" rel="noopener">Open sign-in link</a>
                <button type="button" class="btn btn-default" data-copy-signin-link>Copy link</button>
            </div>
        ` : `
            <div class="settings-inline-note" data-tone="error" data-unsafe-signin-link>
                The engine disclosed a sign-in link this app will not open (only http and https links are opened).
            </div>
        `);
        if (disclosure.code) {
            // The codex device flow: big selectable one-time code, explicit
            // Copy (never auto-copied).
            bits.push(`
                <div class="harness-device-code">
                    <p>Enter this one-time code on that page:</p>
                    <div class="harness-code" data-device-code>${escapeHtml(disclosure.code)}</div>
                    <button type="button" class="btn btn-default" data-copy-device-code>Copy code</button>
                </div>
            `);
        }
    }
    // The live state line and the verdict are mutually exclusive by
    // construction (loginStatusLine answers '' on a terminal job), so the
    // card can never say two contradicting things at once. The verdict slot
    // ALSO silences it explicitly: once a verdict (or its re-check) owns the
    // card, a snapshot that still reads pending — a poll response captured
    // before the job settled — must not print "Waiting for the sign-in link…"
    // underneath "Connected.".
    if (face !== 'error' && face !== 'name' && !active.verdict && !active.confirming) {
        const line = active.preparingRuntime
            ? claudexorPreparationLine(statusPayload)
            : loginStatusLine(active.envelope || {});
        if (line) bits.push(`<div class="settings-inline-status" data-tone="muted" data-login-state>${escapeHtml(line)}</div>`);
    }
    // Shape 2: the ALWAYS-VISIBLE paste-code entry for a job whose disclosure
    // flow is `oauth_url_input`. Claude's CLI prompts for the code only when
    // the browser cannot reach its localhost callback (e.g. the browser is on
    // another device), and this UI cannot observe that prompt — so the field
    // is unconditional while the job awaits the user, with the hint carrying
    // both the "if" and the optionality (no code is needed when the callback
    // completes on its own).
    if (!compact && face === 'device' && loginInputSupport(active.envelope || {}) && !summary.terminal) {
        const busy = active.inputBusy || active.inputSent;
        bits.push(`
            <div class="harness-code-entry" data-code-entry>
                <label for="harness-code-input">If the browser shows a code instead of finishing, paste it here (otherwise the sign-in completes on its own):</label>
                <div class="harness-code-entry-row">
                    <input type="text" id="harness-code-input" data-login-code-input autocomplete="off" spellcheck="false"
                        placeholder="sign-in code" value="${escapeHtml(active.inputValue || '')}"${active.inputSent ? ' disabled' : ''}>
                    <button type="button" class="btn btn-default" data-login-code-submit${busy ? ' disabled' : ''}>${active.inputBusy ? 'Sending…' : (active.inputSent ? 'Code sent' : 'Submit code')}</button>
                </div>
                ${active.inputError ? `<div class="settings-inline-note" data-tone="error">${escapeHtml(active.inputError)}</div>` : ''}
                ${active.inputSent ? `<div class="settings-inline-status" data-tone="muted">${escapeHtml(active.inputNote || 'Code sent — finishing the sign-in…')}</div>` : ''}
            </div>
        `);
    }
    // The verdict: success, a REAL failure with its typed reason, or the
    // in-between "confirming" state while live account status is re-checked.
    if (active.confirming) {
        const text = active.verdict?.kind === 'recovery'
            ? 'Checking whether the previous sign-in process stopped…'
            : 'Confirming the sign-in…';
        bits.push(`<div class="settings-inline-status" data-tone="muted" data-login-verdict>${text}</div>`);
    } else if (face === 'recovery') {
        bits.push(`<div class="settings-inline-status" data-tone="warn" data-login-verdict>
            Claudexor could not prove the previous sign-in process stopped, so it will not start another one yet.
        </div>`);
        const detail = String(active.verdict?.detail || '').trim();
        if (detail) bits.push(`<div class="settings-inline-note" data-login-detail>${escapeHtml(detail)}</div>`);
        if (!compact) {
            bits.push('<button type="button" class="btn btn-primary" data-login-reconcile>Check again</button>');
        }
    } else if (face === 'reconciled') {
        bits.push('<div class="settings-inline-status" data-tone="ok" data-login-verdict>The previous sign-in process is no longer blocking a new sign-in.</div>');
        if (!compact) {
            bits.push('<button type="button" class="btn btn-primary" data-login-retry>Connect again</button>');
        }
    } else if (face === 'unavailable') {
        bits.push('<div class="settings-inline-status" data-tone="muted" data-login-verdict>This sign-in job is no longer available. Refresh the account status before trying again.</div>');
    } else if (active.verdict?.kind === 'success') {
        bits.push('<div class="settings-inline-status" data-tone="ok" data-login-verdict>Connected.</div>');
    } else if (active.verdict?.kind === 'failure') {
        bits.push(`<div class="settings-inline-status" data-tone="error" data-login-verdict>${escapeHtml(failureText(active.verdict.reason))}</div>`);
    } else if (active.verdict?.kind === 'unconfirmed') {
        // The re-check window closed without the row appearing. Unknown, not
        // failed — worded so the owner looks where the answer actually lands.
        bits.push(`<div class="settings-inline-status" data-tone="warn" data-login-verdict>${escapeHtml(UNCONFIRMED_TEXT)}</div>`);
    }
    // A settled non-success verdict in COMPACT mode still offers the retry the
    // mode promises; in full mode the account row's own Connect button is the
    // retry, exactly as today.
    if (compact && loginRetryAvailable(active)
        && (active.verdict?.kind === 'failure' || active.verdict?.kind === 'unconfirmed')) {
        bits.push('<button type="button" class="btn btn-primary" data-login-retry>Try again</button>');
    }
    // …and beside that verdict, the engine's own explanation. The two verdict
    // texts above are FIXED constants (deliberately: one is a category, the
    // other an honest "unknown"), so without this line a settled login says
    // nothing about WHY — the daemon's sentence lived in the snapshot and was
    // rendered nowhere. Only for a settled non-success verdict: beside
    // "Connected." a stale message would contradict the outcome, and while the
    // job is pending the live status line already owns the card.
    if (!compact && (active.verdict?.kind === 'failure' || active.verdict?.kind === 'unconfirmed')) {
        // #125: a verdict minted client-side (the poll give-up) carries its own
        // sentence; a daemon-settled verdict keeps the engine's own message.
        const detail = String(active.verdict?.detail || '').trim() || jobDetail(active.envelope || {});
        if (detail) {
            bits.push(`<div class="settings-inline-note" data-login-detail>${escapeHtml(detail)}</div>`);
        }
    }
    if (externalTerminalActionAvailable(active)) {
        bits.push(`
            <div class="settings-inline-note" data-login-external-action>
                This sign-in cannot use the in-app terminal transport on this host.
            </div>
            <button type="button" class="btn btn-primary" data-login-external-terminal>Continue in external terminal</button>
        `);
    }
    // An explicit external-terminal continuation exposes its usable command
    // immediately in both render modes; compact also exposes any attach-capable
    // job directly because it has no Advanced wrapper. This is inert copy-paste
    // text for the labelled shell, never an embedded terminal.
    const directExternal = !!active.attachCommand && !summary.terminal
        && (active.transport === 'client_pty' || compact);
    if (directExternal) {
        const commandLabel = attachShellLabel(active.attachShell);
        bits.push(`
            <div class="harness-external-terminal" data-login-external-command>
                <p>Continue this sign-in in your own terminal (outside Ouroboros).</p>
                <p><strong>Command for ${escapeHtml(commandLabel)}:</strong></p>
                <pre class="harness-attach-command" data-attach-command>${escapeHtml(active.attachCommand)}</pre>
                <button type="button" class="btn btn-default" data-copy-attach>Copy command</button>
            </div>
        `);
    }
    // The demoted attach fallback: a collapsed Advanced affordance, only when
    // due (engine predates the disclosure modes, or no link within the
    // window) and only while the job can still be attached.
    if (!compact && !directExternal && !summary.terminal && attachFallbackDue(active, nowMs)) {
        const commandLabel = attachShellLabel(active.attachShell);
        bits.push(`
            <details class="harness-advanced" data-login-advanced${active.advancedOpen ? ' open' : ''}>
                <summary>Advanced: sign in from your own terminal</summary>
                <p>${active.engineDegraded
        ? (active.setupLoginSource === 'legacy_global_operation'
            ? 'This older agent service exposes only a global sign-in capability, not a per-agent host guarantee. Run this in your own terminal (outside Ouroboros); the card follows the progress automatically:'
            : 'This agent uses an external terminal for sign-in on this host. Run this outside Ouroboros; the card follows the progress automatically:')
        : 'No sign-in link arrived yet. You can run this in your own terminal (outside Ouroboros) instead; the card follows the progress automatically:'}</p>
                <p><strong>Command for ${escapeHtml(commandLabel)}:</strong></p>
                <pre class="harness-attach-command" data-attach-command>${escapeHtml(active.attachCommand)}</pre>
                <button type="button" class="btn btn-default" data-copy-attach>Copy command</button>
            </details>
        `);
    }
    if (!compact) {
        // A pressed Close queues behind the in-flight start transition (C7) —
        // possibly for a whole runtime install. The press must ANSWER at once,
        // or a working control reads as a dead one (owner report, 2026-08-27).
        bits.push(`<button type="button" class="btn btn-default" data-login-dismiss${active.closing ? ' disabled' : ''}>${active.closing ? 'Closing…' : 'Close'}</button>`);
    }
    bits.push('</div>');
    return bits.join('');
}

export function preserveCardFocus(host, swap, doc = typeof document === 'undefined' ? null : document) {
    // The card re-renders on EVERY 3-second poll tick by replacing its whole
    // DOM, and the replacement dropped the caret out of the paste-code field
    // mid-typing — a sign-in code cannot be typed into a field that dies every
    // three seconds. The focused entry and its selection are captured across
    // the swap and restored onto the element that replaced it.
    const prior = doc ? doc.activeElement : null;
    // The paste-code field AND the name-the-account field share the fate: a
    // re-render (poll tick, or the name face redrawing its normalization note)
    // replaces the input mid-typing.
    const marker = ['data-login-code-input', 'data-profile-name-input']
        .find((attr) => prior && host.contains?.(prior) && prior.hasAttribute?.(attr));
    const keep = Boolean(marker);
    const start = keep ? prior.selectionStart : null;
    const end = keep ? prior.selectionEnd : null;
    swap();
    if (!keep) return;
    const next = host.querySelector?.(`[${marker}]`);
    if (!next || next.disabled) return;
    next.focus?.();
    if (typeof next.setSelectionRange === 'function' && start !== null) {
        next.setSelectionRange(start, end === null ? start : end);
    }
}

// ---------------------------------------------------------------------------
// The controller. Owns the job lifecycle; the caller owns the DOM host.
// ---------------------------------------------------------------------------

/**
 * @param {object} options
 * @param {Element|Function} options.host  where the card renders (element or getter)
 * @param {object} [options.store]         the shared Claudexor status store
 * @param {string} [options.mode]          'full' | 'compact'
 * @param {Function} [options.fetchImpl]   transport
 * @param {Function} [options.onSettled]   called after a verdict lands / the card closes
 * @returns {object} controller with start/close/render/dispose/detach
 *
 * LIFECYCLE CONTRACT (public — two sibling branches mount this):
 *   * `close()` and `dispose()` resolve to a typed custody result:
 *     `released`, valid still-held `retained`, or evidence-poor `unknown`.
 *     A 2xx status alone is never release proof.
 *   * Neither is ever DROPPED. Transitions queue and run in order, so a close
 *     that arrives during the create POST is applied to the job that POST
 *     installs — it used to be answered `false` and forgotten, leaving a live
 *     server-side job nobody ever cancelled.
 *   * `detach()` is the narrow synchronous local exit: it starts no lifecycle
 *     request, clears timers/holds/host/active state, and permanently fences
 *     this controller. It does not turn retained/unknown custody into proof.
 *   * `start()` COALESCES duplicate starts for the same harness/profile onto
 *     one promise, so a double-click cannot create-then-cancel-then-create.
 */
export function createLoginCardController({
    host,
    store = claudexorStatus,
    mode = LOGIN_CARD_FULL,
    fetchImpl = apiFetch,
    doc = () => (typeof document === 'undefined' ? null : document),
    now = () => Date.now(),
    onSettled = () => {},
} = {}) {
    const getHost = typeof host === 'function' ? host : () => host;
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    const ctl = {
        active: null,
        jobTimer: 0,
        transitionChain: Promise.resolve(),
        pendingStart: null,
        releaseHold: null,
        disposed: false,
        detachedStatus: null,
    };

    function holdStatusPolling() {
        // A live login job keeps the shared status poll awake even when the
        // surface would otherwise be considered idle — the account rows are
        // exactly what the login is about to change.
        if (ctl.releaseHold || !store?.holdPolling) return;
        ctl.releaseHold = store.holdPolling('login-job');
    }

    function releaseStatusPolling() {
        if (!ctl.releaseHold) return;
        ctl.releaseHold();
        ctl.releaseHold = null;
    }

    function stopJobPolling() {
        if (ctl.jobTimer) clearTimeout(ctl.jobTimer);
        ctl.jobTimer = 0;
    }

    function clearHost() {
        const hostEl = getHost();
        if (hostEl) hostEl.innerHTML = '';
    }

    // While the create POST holds the transition (runtime ensure included),
    // no job polling exists yet, so nothing re-rendered the card as the
    // runtime moved through installing -> starting -> serving; the phase line
    // froze on its first words. A BARE subscription (no visibility predicate)
    // cannot by itself make the store poll — it rides whatever reads the
    // visible surfaces and the login hold cause.
    const releasePhaseFollow = store?.subscribe
        ? store.subscribe(() => { if (ctl.active?.preparingRuntime) render(); })
        : () => {};

    function render() {
        const hostEl = getHost();
        if (!hostEl) return;
        const active = ctl.active;
        if (!active) { hostEl.innerHTML = ''; return; }
        // Resolve at paint time: only a proven current catalog read may lend
        // its display name; a retained snapshot after a gap stays untrusted.
        active.familyLabel = familyLabel(active.harness, store?.snapshot || null, {
            catalogKnown: Boolean(store?.catalogKnown),
        });
        if (active.needsProfile) active.needsProfile.familyLabel = active.familyLabel;
        preserveCardFocus(hostEl, () => {
            // A failed refresh RETAINS the prior snapshot and records the
            // error; a retained snapshot is not phase evidence, and rendering
            // it kept a positive "Installing…" claim alive off a dead read.
            hostEl.innerHTML = loginCardHtml(active, now(), {
                mode,
                statusPayload: store?.error ? null : (store?.snapshot || null),
            });
        }, getDoc());
        wireLoginCard(hostEl, active);
    }

    function withLoginTransition(fn) {
        // ONE QUEUE for EVERY login lifecycle transition (C7): start, close and
        // shutdown. Each transition is an async section that reads and replaces
        // ctl.active across awaits (create POST while jobId is still '';
        // cancel DELETE while a start could run), so they must not interleave.
        //
        // It used to be a try-LOCK that DROPPED whatever arrived while another
        // transition ran, and dropping a close is not a small loss: a Close
        // pressed during the create POST answered "false" and vanished, the
        // POST then installed a live job, and no DELETE was ever issued — the
        // daemon kept running a sign-in for a card the owner had dismissed.
        // Queued instead, every transition runs in order and observes the world
        // the previous one left, so the close applies to the job it created.
        const run = ctl.transitionChain.then(() => fn(), () => fn());
        ctl.transitionChain = run.then(() => {}, () => {});
        return run;
    }

    function wireLoginCard(hostEl, active) {
        hostEl.querySelector('[data-login-retry]')?.addEventListener('click', () => {
            // NO preemptive stopJobPolling() here: if the C7 guard inside
            // start refuses (cancel unproven, job still live), the old poll
            // must stay armed — it is the recovery path that clears the
            // transient note and settles the real verdict. _startLocked stops
            // polling itself once the previous job is terminal or provably
            // cancelled.
            start(active.harness, active.profile, active.transport);
        });
        hostEl.querySelector('[data-login-external-terminal]')?.addEventListener('click', () => {
            // This is a NEW job through the same serialized release/custody
            // guard as every other start. Never mutate the active job into a
            // different transport and never auto-fallback.
            start(active.harness, active.profile, 'client_pty');
        });
        // Copy is a STEP of signing in here, so a non-secure origin or a
        // desktop shell without the async clipboard must not leave a dead
        // button: the shared helper falls back and always reports.
        hostEl.querySelector('[data-copy-signin-link]')?.addEventListener('click', () => {
            const disclosure = deviceCodeDisclosure(active.envelope || {});
            void copyTextWithToast(disclosure?.url || '', { okMessage: 'Sign-in link copied.' });
        });
        hostEl.querySelector('[data-copy-device-code]')?.addEventListener('click', () => {
            const disclosure = deviceCodeDisclosure(active.envelope || {});
            void copyTextWithToast(disclosure?.code || '', { okMessage: 'Code copied.' });
        });
        hostEl.querySelector('[data-copy-attach]')?.addEventListener('click', () => {
            void copyTextWithToast(active.attachCommand || '', { okMessage: 'Command copied.' });
        });
        const advanced = hostEl.querySelector('[data-login-advanced]');
        // Re-renders (every poll tick) must not slam the user's Advanced section
        // shut: the open state is carried on the job and re-applied.
        advanced?.addEventListener('toggle', () => { active.advancedOpen = advanced.open; });
        const codeInput = hostEl.querySelector('[data-login-code-input]');
        codeInput?.addEventListener('input', () => { active.inputValue = codeInput.value; });
        codeInput?.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') { event.preventDefault(); submitCodeFromCard(active); }
        });
        const nameInput = hostEl.querySelector('[data-profile-name-input]');
        nameInput?.addEventListener('input', () => { active.profileNameValue = nameInput.value; });
        nameInput?.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') { event.preventDefault(); submitProfileNameFromCard(active); }
        });
        hostEl.querySelector('[data-profile-name-submit]')?.addEventListener('click', () => submitProfileNameFromCard(active));
        hostEl.querySelector('[data-login-code-submit]')?.addEventListener('click', () => submitCodeFromCard(active));
        hostEl.querySelector('[data-login-reconcile]')?.addEventListener('click', () => reconcile(active));
        hostEl.querySelector('[data-login-dismiss]')?.addEventListener('click', () => {
            // Instant local acknowledgement: the close itself queues behind
            // whatever transition is in flight (C7), and during a runtime
            // install that wait is minutes — a silent queued click reads as a
            // broken button. The close promise is RETURNED for harnesses that
            // invoke the handler directly (the DOM discards listener returns),
            // and the flag resets on settle only when this card is still the
            // active one (a settled close usually cleared it).
            if (active.closing) return undefined;
            active.closing = true;
            render();
            return close(active).finally(() => {
                if (ctl.active === active && active.closing) {
                    active.closing = false;
                    render();
                }
            });
        });
    }

    function showRecovery(active, result, fallbackDetail = '') {
        if (result.envelope) active.envelope = result.envelope;
        active.absent = false;
        active.custodyStatus = result.status;
        active.error = '';
        active.confirming = false;
        active.verdict = {
            kind: 'recovery',
            reason: terminationReason(active.envelope) || 'custody_retained',
            detail: String(result.error || fallbackDetail || '').trim(),
        };
        stopJobPolling();
        releaseStatusPolling();
    }

    function detach() {
        // Owner-approved honest local detach. It is intentionally outside the
        // async transition queue: pagehide and the second recovery-face Close
        // are non-awaitable and must make every continuation inert NOW. The
        // monotone disposed flag plus active identity checks are the fence; a
        // second generation counter would duplicate that authority.
        if (!ctl.active && ctl.detachedStatus) {
            // Disposed or not: an empty slot with a remembered verdict answers
            // that verdict — the same invariant _closeLocked and dispose hold.
            // The per-card no-job close leaves {disposed:false, active:null,
            // detachedStatus}, and fabricating absent/released here would be
            // exactly the empty-slot release proof this seam forbids.
            ctl.disposed = true;
            stopJobPolling();
            releaseStatusPolling();
            clearHost();
            return ctl.detachedStatus;
        }
        const active = ctl.active;
        let result = active
            ? custodyResult(active.envelope, { absent: Boolean(active.absent) })
            : custodyResult(null, { absent: true });
        if (active?.custodyStatus === LOGIN_CUSTODY_UNKNOWN) {
            result = { ...result, status: LOGIN_CUSTODY_UNKNOWN };
        }
        ctl.disposed = true;
        ctl.pendingStart = null;
        stopJobPolling();
        releaseStatusPolling();
        // Same fence as the timers and the polling hold: a detached controller
        // must not keep reacting to store snapshots (the disposer is
        // idempotent, so a later dispose() releasing again is a no-op).
        try { releasePhaseFollow(); } catch (err) { /* detach must not throw */ }
        ctl.active = null;
        ctl.detachedStatus = result.status;
        clearHost();
        return result.status;
    }

    async function _closeLocked(expected, { shuttingDown = false } = {}) {
        const active = ctl.active;
        if (!active) {
            if (shuttingDown) { stopJobPolling(); releaseStatusPolling(); clearHost(); }
            // A remembered detach verdict outranks the empty-slot default: a
            // queued close that detached with UNKNOWN must not be overwritten
            // by a later shutdown answering absent/released for the same card
            // — that would fabricate release proof out of an empty slot.
            if (ctl.detachedStatus) return { status: ctl.detachedStatus };
            return custodyResult(null, { absent: true });
        }
        if (expected !== undefined && active !== expected) return custodyResult(null);
        if ((active.error || active.needsProfile) && !active.jobId && !shuttingDown) {
            // Re-proved at EXECUTION time, not at click time: a close queued
            // during an in-flight create observes the world that create left.
            // When it failed without a job id there is nothing a DELETE can
            // address. This is the PER-CARD half of detach() only: the full
            // detach permanently fences the controller, which silently dropped
            // a second Connect legitimately queued behind this close. The
            // verdict is still remembered so a dispose queued behind this
            // close answers it instead of fabricating release from the empty
            // slot; a later start opens a new custody story and clears it.
            let result = custodyResult(active.envelope, { absent: Boolean(active.absent) });
            if (active.custodyStatus === LOGIN_CUSTODY_UNKNOWN) {
                result = { ...result, status: LOGIN_CUSTODY_UNKNOWN };
            }
            stopJobPolling();
            releaseStatusPolling();
            ctl.active = null;
            ctl.detachedStatus = result.status;
            clearHost();
            onSettled();
            return result;
        }

        let result = custodyResult(active.envelope, { absent: Boolean(active.absent) });
        if (active.verdict?.kind === 'recovery'
            && active.custodyStatus !== LOGIN_CUSTODY_UNKNOWN) {
            // Repeating cancel cannot turn an already parsed terminal
            // unconfirmed result into process-empty proof. The host consumes
            // `retained` and chooses its explicit local detach policy.
            showRecovery(active, result);
            if (shuttingDown) clearHost(); else render();
            return result;
        }
        if (result.status !== LOGIN_CUSTODY_RELEASED && active.jobId) {
            result = await cancelLoginJob(active.jobId, fetchImpl);
            if (ctl.active !== active || ctl.disposed && !shuttingDown) return custodyResult(null);
            // A terminal response may have landed while DELETE was in flight.
            // The same release predicate judges that fresher envelope; no
            // separate "settled means gone" shortcut is allowed.
            const current = custodyResult(active.envelope, { absent: Boolean(active.absent) });
            if (!shuttingDown
                && result.status === LOGIN_CUSTODY_UNKNOWN
                && current.status === LOGIN_CUSTODY_RELEASED
                && active.verdict) {
                // The passive GET won the race and already rendered a truthful
                // terminal face while the DELETE was unconfirmed. Do not let
                // that older Close continuation erase the outcome it just
                // learned. A later explicit Close starts with the verdict
                // already present and still dismisses the settled card.
                stopJobPolling();
                releaseStatusPolling();
                render();
                return current;
            }
            if (result.status === LOGIN_CUSTODY_UNKNOWN
                && (current.status === LOGIN_CUSTODY_RELEASED
                    || (active.verdict?.kind === 'recovery'
                        && active.custodyStatus !== LOGIN_CUSTODY_UNKNOWN))) result = current;
            if (result.envelope) active.envelope = result.envelope;
            if (result.absent) active.absent = true;
        }

        if (result.status === LOGIN_CUSTODY_RETAINED) {
            showRecovery(active, result);
            if (shuttingDown) clearHost(); else render();
            return result;
        }
        if (result.status === LOGIN_CUSTODY_UNKNOWN) {
            active.custodyStatus = LOGIN_CUSTODY_UNKNOWN;
            active.error = 'Could not cancel this sign-in — it may still be running. '
                + 'Try dismissing again once the daemon is reachable.';
            if (shuttingDown) {
                stopJobPolling();
                releaseStatusPolling();
                clearHost();
            } else {
                render();
            }
            return result;
        }

        stopJobPolling();
        releaseStatusPolling();
        if (!shuttingDown && result.absent) {
            active.error = '';
            active.confirming = false;
            active.verdict = { kind: 'unavailable', reason: 'job_absent' };
            render();
            store?.refresh?.();
            onSettled();
            return result;
        }
        ctl.active = null;
        if (shuttingDown) { clearHost(); return result; }
        render();
        store?.refresh?.();
        onSettled();
        return result;
    }

    /**
     * Close the card, cancelling a still-live job first (C7).
     * @param {object} [expected] when given, a no-op unless it is still the
     *        active job — stale wiring from a rebuilt card must never clear a
     *        job it does not own.
     * @returns {Promise<string>} released | retained | unknown.
     */
    function close(expected = undefined) {
        if (ctl.disposed && !ctl.active && ctl.detachedStatus) {
            return Promise.resolve(ctl.detachedStatus);
        }
        const active = ctl.active;
        if ((active?.error || active?.needsProfile) && !active.jobId
            && (expected === undefined || expected === active)) {
            // A create that produced no usable identity has nothing this card
            // can address with DELETE. A typed refusal carries absent=true and
            // detaches as released; an untyped discovery/transport failure
            // preserves unknown. Either way the existing Close affordance is
            // usable through honest local detach.
            const result = detach();
            onSettled();
            return Promise.resolve(result);
        }
        if (active?.verdict?.kind === 'unavailable'
            && (expected === undefined || expected === active)) {
            // The absent response already released custody. Its distinct face
            // is informational, not a card the owner can never close.
            stopJobPolling();
            releaseStatusPolling();
            ctl.active = null;
            clearHost();
            onSettled();
            return Promise.resolve(LOGIN_CUSTODY_RELEASED);
        }
        if (active?.verdict?.kind === 'recovery'
            && (expected === undefined || expected === active)) {
            // The FIRST active Close already attempted cancellation and exposed
            // recovery. The SECOND visible Close is the distinct synchronous
            // local detach: zero create, DELETE or reconcile requests.
            const result = detach();
            onSettled();
            return Promise.resolve(result);
        }
        return withLoginTransition(() => _closeLocked(expected)).then((result) => result.status);
    }

    function reconcile(expected = undefined) {
        return withLoginTransition(async () => {
            const active = ctl.active;
            if (!active || ctl.disposed || (expected !== undefined && expected !== active)
                || active.verdict?.kind !== 'recovery') return custodyResult(null);
            stopJobPolling();
            active.confirming = true;
            render();
            const result = await reconcileLoginJob(active.jobId, fetchImpl);
            if (ctl.active !== active || ctl.disposed) return custodyResult(null);
            active.confirming = false;
            if (result.envelope) active.envelope = result.envelope;
            if (result.absent) active.absent = true;
            if (result.status === LOGIN_CUSTODY_RELEASED) {
                active.error = '';
                active.custodyStatus = LOGIN_CUSTODY_RELEASED;
                active.verdict = result.absent
                    ? { kind: 'unavailable', reason: 'job_absent' }
                    : { kind: 'reconciled', reason: '' };
                releaseStatusPolling();
                render();
                store?.refresh?.();
                onSettled();
                return result;
            }
            const detail = result.error || (result.status === LOGIN_CUSTODY_RETAINED
                ? 'Claudexor still cannot prove the previous sign-in process stopped. Check again later.'
                : 'Could not check the previous sign-in process. Check again when the daemon is reachable.');
            showRecovery(active, result, detail);
            render();
            return result;
        });
    }

    function submitProfileNameFromCard(active) {
        // The "name the account" face's submit: the same validation loop the
        // Add-account dialog runs (`promptProfileName`) — a name normalization
        // would rewrite is shown back, editable, before any login starts. A
        // stable name starts the NAMED login through the one standard start
        // path (create → poll → verdict), which replaces this face.
        if (!active?.needsProfile || ctl.active !== active || ctl.disposed) return;
        const answer = profileNameSubmission(active.profileNameValue);
        if (!answer.profile) {
            if (answer.normalized) active.profileNameValue = answer.normalized;
            active.profileNameNote = answer.note;
            render();
            return;
        }
        start(active.harness, answer.profile, active.transport);
    }

    async function submitCodeFromCard(active) {
        // Double-submit guard: one POST in flight at a time, and an accepted
        // code is final — the button stays disabled after success.
        if (!active || active.inputBusy || active.inputSent) return;
        const value = String(active.inputValue || '').trim();
        if (!value) return;
        // The job may have finished while the user typed: render the verdict
        // instead of posting into a dead job.
        if (jobStateSummary(active.envelope || {}).terminal) { render(); return; }
        active.inputBusy = true;
        active.inputError = '';
        render();
        const result = await submitLoginInput(active.jobId, value, { fetchImpl });
        if (ctl.active !== active) return;   // card closed while in flight
        active.inputBusy = false;
        if (result.ok) {
            active.inputSent = true;
        } else if (result.conflict === 'setup_input_not_applicable') {
            // Typed 409: the callback already completed (or the flow moved past
            // the code step) — no code is needed. An answer, not an error: quiet
            // note, and the job poll lands the verdict.
            active.inputSent = true;
            active.inputNote = 'No code needed — the sign-in is already completing.';
        } else if (result.conflict === 'setup_input_already_submitted') {
            // Typed 409: the server already has a code (it is authoritative over
            // our double-submit guard, e.g. a second browser tab).
            active.inputSent = true;
            active.inputNote = 'Code already received — finishing the sign-in.';
        } else if (result.degraded) {
            // The engine predates the input route, or reaped the job. Degrade to
            // the Advanced fallback when one exists; otherwise say what is known.
            if (!jobStateSummary(active.envelope || {}).terminal && active.attachCommand) {
                active.engineDegraded = true;
                active.inputError = 'This engine cannot accept the code here — see Advanced below.';
            } else {
                active.inputError = 'The engine did not accept the code (the sign-in may have already finished).';
            }
        } else {
            active.inputError = result.error || 'Code submission failed; try again.';
        }
        render();
    }

    function start(harness, profile, transport = '') {
        if (!harness || ctl.disposed) return Promise.resolve();
        // Duplicate starts are COALESCED, not merely serialized. The queue
        // guaranteed order, and order made an ordinary double-click into
        // create → cancel → create: the second start ran the C7 guard against
        // the job the first one had just installed, DELETEd it, and created
        // another — so the device link the owner was reading was invalidated
        // the moment it appeared. While a start for the same account is still
        // settling, every caller shares its promise.
        // The separator is spelled as a source escape rather than written as a
        // literal NUL byte: one raw NUL makes every ordinary search tool
        // classify this 53 KB module as binary, and `grep` then answers with
        // SILENCE instead of "no match", so a reviewer looking for a symbol
        // in here concludes it does not exist. Same key, same collisions.
        const key = `${harness}\u0000${profile || ''}\u0000${transport || ''}`;
        if (ctl.pendingStart && ctl.pendingStart.key === key) return ctl.pendingStart.promise;
        const promise = withLoginTransition(() => _startLocked(harness, profile, transport));
        const clear = () => {
            if (ctl.pendingStart && ctl.pendingStart.promise === promise) ctl.pendingStart = null;
        };
        // Both arms, and on the ORIGINAL promise: a derived one would be an
        // unhandled rejection for callers that fire and forget.
        promise.then(clear, clear);
        ctl.pendingStart = { key, promise };
        return promise;
    }

    async function _startLocked(harness, profile, transport = '') {
        // Re-read after the queue wait: a shutdown may have been queued ahead of
        // this start, and a disposed controller must not create a job nobody
        // will ever poll or cancel.
        if (ctl.disposed) return;
        // A new card is a new custody story: the remembered verdict belonged
        // to the card the queued close detached, not to this one.
        ctl.detachedStatus = null;
        // C7 (plan roast, accepted): a NEW login may start only once release of
        // the previous job is PROVEN (loginReleaseProven): a terminal snapshot
        // whose reason is not termination_unconfirmed, a reconciliation that
        // found the setup empty, or a job proven absent. A terminal
        // termination_unconfirmed snapshot keeps the job fenced, and a 2xx
        // cancel response alone is NOT proof of release — a second job beside a
        // possibly-live one races the daemon and orphans the old job
        // server-side. Centralized HERE so every entry point (Connect, Add
        // account, Retry) inherits the guard.
        const prev = ctl.active;
        if (prev && prev.jobId
            && !loginReleaseProven(prev.envelope, { absent: Boolean(prev.absent) })) {
            let result = await cancelLoginJob(prev.jobId, fetchImpl);
            if (ctl.active !== prev || ctl.disposed) return;
            const current = custodyResult(prev.envelope, { absent: Boolean(prev.absent) });
            if (result.status === LOGIN_CUSTODY_UNKNOWN
                && (current.status === LOGIN_CUSTODY_RELEASED
                    || (prev.verdict?.kind === 'recovery'
                        && prev.custodyStatus !== LOGIN_CUSTODY_UNKNOWN))) result = current;
            if (result.envelope) prev.envelope = result.envelope;
            if (result.absent) prev.absent = true;
            if (result.status === LOGIN_CUSTODY_RETAINED) {
                showRecovery(prev, result);
                render();
                return;
            }
            if (result.status === LOGIN_CUSTODY_UNKNOWN) {
                prev.custodyStatus = LOGIN_CUSTODY_UNKNOWN;
                prev.error = 'Could not cancel the active sign-in — it may still be running. '
                    + 'Retry from its card, or dismiss it first.';
                render();
                return;
            }
        }
        stopJobPolling();
        holdStatusPolling();
        // One start shape for every harness: the in-app flow is asked for and the
        // server/engine decide the hosting (loginFlow rides only for codex).
        const body = { harness, login_flow: 'device_auth' };
        if (profile) body.profile_id = profile;
        if (transport === 'client_pty') body.transport = transport;
        ctl.active = {
            harness, profile, transport, jobId: '', envelope: null, absent: false, custodyStatus: '',
            attachCommand: '', attachShell: '', setupLoginSource: '', error: '',
            problemCode: '', requiredActions: [],
            startedAtMs: now(), engineDegraded: false,
            inputValue: '', inputBusy: false, inputSent: false, inputError: '', inputNote: '',
            needsProfile: null, profileNameValue: '', profileNameNote: '',
            verdict: null, confirming: false, advancedOpen: false, preparingRuntime: true,
            closing: false,
        };
        const active = ctl.active;
        render();
        try {
            const resp = await fetchImpl('/api/claudexor/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json().catch(() => null);
            if (ctl.active !== active) return;
            if (!resp.ok) {
                const actions = Array.isArray(data?.required_actions)
                    ? data.required_actions.map(String) : [];
                const problemCode = typeof data?.code === 'string' ? data.code : '';
                active.problemCode = problemCode;
                active.requiredActions = actions;
                if (problemCode) {
                    // A typed daemon refusal before a durable job was returned
                    // proves this client owns no setup custody. Generic
                    // discovery/transport failures carry no code and remain
                    // honestly unknown.
                    active.absent = true;
                    active.custodyStatus = LOGIN_CUSTODY_RELEASED;
                }
                if (resp.status === 400 && !profile
                    && problemCode === 'credential_profile_required'
                    && actions.includes('add_named_account')) {
                    // The engine's typed verdict, not harness identity, family
                    // emptiness, status alone or prose, selects the name face.
                    // A named create and every unrelated 400/409 stay ordinary
                    // errors. The refusal also
                    // proves no job exists (the daemon answered the create with
                    // a refusal), so custody is honestly absent/released.
                    active.preparingRuntime = false;
                    active.needsProfile = {
                        message: String(data?.error || `HTTP ${resp.status}`),
                    };
                    releaseStatusPolling();
                    render();
                    return;
                }
                throw new Error(data?.error || `HTTP ${resp.status}`);
            }
            // A 2xx is not a created job. Without a job id there is nothing to
            // poll, nothing to cancel and nothing to report: the card used to
            // accept such an answer and sit "Starting the sign-in…" forever,
            // with no error and no way out. Fail here instead, where the error
            // face and its Try again already exist.
            const jobId = String(data?.job_id || '');
            if (!jobId) throw new Error('the sign-in service returned no job id');
            // Adopted BEFORE the body validation: a nonempty id names a job
            // the daemon may be running regardless of how malformed the rest
            // of the answer is, and a queued close must be able to DELETE it —
            // the no-job detach is only for creates that produced no id.
            active.jobId = jobId;
            if (!hasJobState(data)) throw new Error('the sign-in service returned no job');
            active.preparingRuntime = false;
            active.envelope = data;
            active.attachCommand = String(data.attach_command || '');
            active.attachShell = String(data.attach_shell || '');
            active.setupLoginSource = String(data.setup_login_source || '');
            // An engine that predates the disclosure modes never sends a link;
            // the demoted Advanced fallback may show right away (still collapsed).
            active.engineDegraded = data.disclosure_native === false
                && !!active.attachCommand;
            const verdict = loginVerdict(active.envelope);
            if (verdict.kind === 'pending') startJobPolling();
            else if (verdict.kind === 'recovery') {
                // Install the fence face even when a queued disposer has
                // already marked the controller inert. That queued cleanup
                // then consumes typed retained custody without issuing a
                // structurally-pointless second cancel.
                showRecovery(active, custodyResult(active.envelope));
                if (!ctl.disposed) {
                    render();
                    store?.refresh?.();
                    onSettled();
                }
            } else await settleVerdict(active, verdict);
        } catch (error) {
            if (ctl.active !== active) return;
            active.preparingRuntime = false;
            active.error = String(error.message || error);
            releaseStatusPolling();
        }
        render();
    }

    function startJobPolling() {
        stopJobPolling();
        // #125: a setTimeout CHAIN, not an interval — the next tick is armed only
        // after the previous round-trip settled, so overlapping polls (and the
        // out-of-order snapshot that overwrote a terminal one) are structurally
        // impossible; the pollResponseApplies ordering guard still stands for a
        // card closed/reopened mid-flight. Failures back off exponentially and
        // after 10 CONSECUTIVE failures the chain stops with an honest
        // "unconfirmed" verdict instead of hammering a dead daemon forever.
        let consecutiveFailures = 0;
        const schedule = (delayMs) => { ctl.jobTimer = setTimeout(tick, delayMs); };
        const tick = async () => {
            ctl.jobTimer = 0;
            const active = ctl.active;
            if (!active?.jobId) return;
            let envelope = null;
            let absent = false;
            let readOk = false;
            try {
                const resp = await fetchImpl(`/api/claudexor/login/${encodeURIComponent(active.jobId)}`, { cache: 'no-store' });
                const data = await resp.json().catch(() => null);
                // The CREATE answer is the only authority on whether this job has a
                // copy-paste command: it issues one exactly for a client_pty job,
                // while the poll route hands one back for every job including the
                // daemon-transport codex device flow, which must never show it.
                //
                // A poll answer without a `job` object is a PROTOCOL failure, not
                // a healthy read: the route answers {job, attach_command} on every
                // success, so a 2xx carrying no job tells us nothing about the
                // login. Counting it as a success reset the failure streak, and a
                // stream of such answers polled forever — the documented ten-failure
                // give-up was unreachable, and the card stayed pending for good.
                const job = isJobEnvelope(data) ? data.job : null;
                // …and an EMPTY job object is no more an answer than a missing
                // one. The gateway normalizes a non-object engine reply to `{}`
                // (`gateways/claudexor.py::setup_job_call`), so `{job:{}}` is
                // reachable on the real wire, not hypothetical: twelve such
                // polls left the verdict null and armed another timer, because
                // each one reset the failure streak. A snapshot with no state
                // says nothing about this login, so it counts as a failure.
                if (resp.status === 404 || resp.status === 410) {
                    absent = true;
                } else if (resp.ok && job && jobStateSummary(data).state) {
                    envelope = data;
                    readOk = true;
                }
            } catch (err) { /* transient poll failure; the chain retries with backoff */ }
            if (ctl.disposed || !pollResponseApplies(active, ctl.active)) return;
            if (absent) {
                active.absent = true;
                active.custodyStatus = LOGIN_CUSTODY_RELEASED;
                active.error = '';
                active.verdict = { kind: 'unavailable', reason: 'job_absent' };
                stopJobPolling();
                releaseStatusPolling();
                render();
                store?.refresh?.();
                onSettled();
                return;
            }
            // ANY answer that CARRIED A JOB (a pending snapshot included) resets the
            // failure streak — a legitimate multi-minute sign-in must never trip
            // give-up — and everything else counts toward the same bounded give-up.
            consecutiveFailures = readOk ? 0 : consecutiveFailures + 1;
            if (envelope) active.envelope = envelope;
            if (readOk && active.error) {
                // A successful job read re-establishes the true state: the
                // cancel-failure note (a Dismiss whose DELETE the daemon never
                // confirmed) is TRANSIENT and must not mask a live/terminal face
                // forever — a recovered daemon + succeeded login used to leave the
                // card stuck on the cancel-error face while the account row said
                // connected. Fatal START errors never reach this line: a job that
                // failed to create has no jobId and is never polled.
                active.error = '';
                active.custodyStatus = '';
            }
            const verdict = loginVerdict(active.envelope || {});
            if (verdict.kind !== 'pending') {
                settleVerdict(active, verdict);
                return;
            }
            const next = nextJobPollDelay(consecutiveFailures);
            if (next.giveUp) {
                // Contact with the job is lost, NOT the login: the sign-in may
                // well have completed. The EXISTING typed unconfirmed verdict path
                // owns this face. Deliberately NOT active.error — its Try-again
                // would start a SECOND login while the old jobId may still be
                // live; a fresh attempt goes through Close (which DELETEs a
                // non-terminal job) or the account row once the daemon answers.
                active.verdict = {
                    kind: 'unconfirmed',
                    reason: 'job_poll_lost_contact',
                    detail: 'Lost contact with the sign-in job — the sign-in may '
                        + 'still have completed. Press Refresh to re-check the account.',
                };
                releaseStatusPolling();
                render();
                store?.refresh?.();
                onSettled();
                return;
            }
            render();
            schedule(next.delayMs);
        };
        schedule(nextJobPollDelay(0).delayMs);
    }

    async function settleVerdict(active, verdict) {
        if (!active || ctl.active !== active || ctl.disposed) return;
        if (verdict.kind !== 'recheck') {
            if (verdict.kind === 'recovery') showRecovery(active, custodyResult(active.envelope));
            else {
                active.verdict = verdict;
                releaseStatusPolling();
            }
            render();
            store?.refresh?.();
            onSettled();
            return;
        }
        // The verify-race: never declare failure off one stale verification read
        // (the owner's codex login SUCCEEDED while the card said "Login failed ·
        // completed"). Live account status — the truth the rows themselves render
        // — briefly re-polled, is the judge.
        active.confirming = true;
        render();
        // The row address: the profile the card started with, upgraded to the
        // job's OWN resolved profile when the engine names one — a unified
        // engine resolves a default login onto its bootstrap registry row, and
        // only that row can confirm it (resolvedJobProfileId). On a legacy
        // engine the job carries null for a default login and the address
        // stays the empty string, byte-identical to today.
        const rowAddress = active.profile
            || resolvedJobProfileId(active.envelope || {}) || '';
        const check = await confirmLoginLive(active.harness, rowAddress, {
            store,
            isStale: () => ctl.active !== active,
        });
        if (ctl.active !== active || ctl.disposed) return;
        active.confirming = false;
        // A positive confirmation is MONOTONE evidence: a login SEEN online is
        // never un-seen by bookkeeping. The bounded re-check can run out a tick
        // before the account row lands, while a reading that arrived meanwhile
        // (the held status poll, an owner wake) already shows the login — so
        // the newest committed snapshot is judged with the same predicate
        // before "unconfirmed" may be said.
        const confirmed = check.confirmed
            || accountLoginConfirmed(store?.snapshot, active.harness, rowAddress);
        // An exhausted window is not a failure verdict: the job's own read was
        // already judged unproven, and the account row often appears a tick after
        // the bounded re-poll gives up. Say that, and say where to look.
        active.verdict = confirmed
            ? { kind: 'success', reason: '' }
            : { kind: 'unconfirmed', reason: verdict.reason };
        releaseStatusPolling();
        render();
        onSettled();
    }

    /**
     * Shut the controller down: wait for an in-flight transition (a create POST
     * owns the queue while its job id is still unknown), then run the SAME
     * proven-cancel path Close uses, and clear the host either way.
     *
     * It used to be synchronous and blind: it cleared `ctl.active`, released
     * the hold and returned — with no DELETE for a live job and the rendered
     * card still on screen. Both matter beyond Settings, because the onboarding
     * wizard mounts this controller on a step the owner can cancel mid-login.
     *
     * @returns {Promise<string>} typed custody status. The host chooses when to
     *        call synchronous `detach()` after consuming retained/unknown.
     */
    function dispose() {
        if (ctl.disposed && !ctl.active) {
            // Idempotent once locally detached: no second request, and the
            // remembered status prevents detach from becoming false proof.
            stopJobPolling();
            releaseStatusPolling();
            return Promise.resolve(ctl.detachedStatus || LOGIN_CUSTODY_RELEASED);
        }
        // Set FIRST: no new start may be queued behind the shutdown.
        ctl.disposed = true;
        ctl.pendingStart = null;
        try { releasePhaseFollow(); } catch (err) { /* disposal must not throw */ }
        return withLoginTransition(() => _closeLocked(undefined, { shuttingDown: true }))
            .then((result) => {
                if (!ctl.active) ctl.detachedStatus = result.status;
                return result.status;
            });
    }

    return {
        start,
        close,
        reconcile,
        render,
        dispose,
        detach,
        // The name-the-account face's submit seam (the card's own input/button
        // wiring calls the same function); exposed for node tests, which have
        // no DOM to click through.
        submitProfileName: () => submitProfileNameFromCard(ctl.active),
        get active() { return ctl.active; },
        get disposed() { return ctl.disposed; },
        get mode() { return mode; },
    };
}
