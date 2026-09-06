// Agent accounts (D30, regrouped in the Agents tab) — the owner-facing surface
// over the owned Claudexor daemon's account truth.
//
// The shape is the owner's (2026-08-08): "все акки клод кода должны быть
// эквивалентны", "позиции кнопок нелогичные… в каждой секции добавить кнопку",
// "текст про лимиты компактнее и понятнее". So:
//
//  * ONE CARD PER FAMILY (Claude Code / Codex / Cursor). The family name and
//    its aggregate status sit in the card header, and that card owns its own
//    Add-account button — the button used to hang off the native row, which is
//    why adding a Codex account meant hunting for a control under an unrelated
//    line.
//  * ROWS ARE ONE TYPE, on both engine generations. A UNIFIED engine (the
//    server-stamped `unified_accounts` fact) serves every account — migrated
//    default logins included — as a named registry row, so every row carries
//    the same name, Enabled toggle and Remove. A LEGACY engine still emits a
//    native pseudo-row per harness; it renders through the same two-line
//    layout, named by the identity the daemon observed, and only its ACTIONS
//    differ — no Remove and no toggle, because the legacy engine has no route
//    for either on that row.
//  * TWO LINES PER ROW. Line 1 is the one primary thing (the account) plus its
//    status; line 2 is muted metadata in human words — "38% used · resets in
//    2h", never a raw ISO instant.
//  * REMOVAL and the ENABLED toggle go through the ENGINE's own contract
//    (DELETE/PATCH /api/claudexor/credential-profiles/…). A legacy native CLI
//    login has neither button: that account belongs to the vendor's CLI, and
//    simulating an effect this app cannot have would be a lie.
//
// The status payload comes from the SHARED store (`claudexor_status_store.js`)
// — this section owns no poll — and the login card is the SHARED controller
// (`harness_login_cards.js`), so the onboarding wizard mounts the same flow.
//
// PROVENANCE IS PER FACET (the store's rule) and it OUTRANKS the aggregate:
// `daemon.state` can say `unreachable` while two of the three fanned-out reads
// landed, so no sentence here judges the daemon off the aggregate alone —
// `daemonAnswered` below is the one predicate, and the status line asks the
// facets before it repeats any runtime claim ("keeps running", "ready").
//
// Pure helpers up top are node-tested without a DOM.

import { apiFetch } from './api_client.js';
import {
    FACET_ACCOUNTS,
    FACET_CATALOG,
    FACET_QUOTA,
    FACET_SUBJECT,
    READ_FAILED,
    READ_INDETERMINATE,
    READ_OK,
    READ_TRANSPORT,
    READ_UNREAD,
    STATUS_FACETS,
    accountRows,
    bindStatusSurface,
    claudexorPreparationLine,
    claudexorStatus,
    facetReadState,
    familyLabel,
    nextUpAccount,
    quotaConstraintFact,
    readsFor,
    unifiedAccounts,
} from './claudexor_status_store.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import { createLoginCardController, normalizeProfileName, preserveCardFocus } from './harness_login_cards.js';
import { formatRelativeAge } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

// ---------------------------------------------------------------------------
// Pure helpers.
// ---------------------------------------------------------------------------

export function verificationBadge(profile, { known = true } = {}) {
    // Q2-а: both statuses are shown honestly — vendor-verified is trusted,
    // local-store presence stays labeled "not verified live" in WORDS, but in
    // a NEUTRAL tone (owner finding #2): the engine has no vendor probe for
    // some harnesses (cursor), so a warning-toned "not verified" there is a
    // permanent alarm nothing can clear — noise, not honesty. "local session"
    // is the daemon's own name for the route (next_up.route).
    //
    // The instant moved OUT of this label (owner finding: a row must not lead
    // with a raw ISO timestamp) — `accountMetaLine` humanizes it below.
    const status = profile?.status || profile || {};
    const source = String(status.verification_source || '');
    const verification = String(status.verification || '');
    const availability = String(status.availability || '');
    const badge = () => {
        if (source === 'vendor' && verification === 'passed') {
            return { tone: 'ok', label: 'Verified live' };
        }
        if (verification === 'passed') {
            // The claim is NARROWER than "signed in" and must stay narrower in
            // WORDS: local-store material has read `passed` a minute before a 401.
            return { tone: 'muted', label: 'Signed in — not verified live' };
        }
        if (verification === 'not_run') {
            // No probe ran, so this is unknown rather than a failed login.
            // A typed `availability=unknown` is stronger evidence: the engine
            // tried the auth-status probe but could not get a verdict (timeout,
            // malformed output, etc.). Keep that distinct from a clean
            // not-yet-verified profile so the owner is not nudged into a
            // needless re-login.
            return {
                tone: 'muted',
                label: availability === 'unknown' ? 'Login status unknown' : 'Not verified',
            };
        }
        if (verification) {
            return { tone: 'error', label: `Verification ${verification}` };
        }
        return { tone: 'muted', label: 'Not signed in' };
    };
    const value = badge();
    // `known` = the ACCOUNTS facet was really read. Otherwise this row is the
    // retained snapshot's memory of an account, and painting a green "Verified
    // live" over a read that never landed is the same lie as the banner's — the
    // panel used to say nothing could be listed while a stale row sat below it
    // dressed as verified. The row survives (it is the only Connect affordance
    // some harnesses have); only its claim is dated, and the green goes with it.
    if (known) return value;
    return { tone: 'muted', label: `${value.label} — last known` };
}

export function humanizeResetAt(resetsAt, nowMs = Date.now()) {
    // "resets in 2h", not "resets 2026-08-09T21:04:00Z". Absence is absence.
    const at = Date.parse(String(resetsAt || ''));
    if (!Number.isFinite(at)) return '';
    const minutes = Math.round((at - nowMs) / 60000);
    if (minutes <= 1) return 'in a moment';
    if (minutes < 60) return `in ${minutes}m`;
    const hours = Math.round(minutes / 60);
    if (hours < 48) return `in ${hours}h`;
    return `in ${Math.round(hours / 24)}d`;
}

export function quotaSummary(snapshots, harnessId, subjectId = '',
                             { quotaRead = READ_OK, nowMs = Date.now(),
                               fallbackSubjectIds = [], absences = [] } = {}) {
    // The exhausted window is SHOWN with its reset time, never hidden (Q2-б):
    // hiding it would make the D28 fallback to API money unexplainable. What
    // CHANGED is only the wording — the owner asked for the limit text to be
    // compact and understandable, so "window exhausted — resets
    // 2026-08-09T21:04:00Z" became "Limit reached · resets in 2h".
    //
    // `quotaRead` is the QUOTA facet's own provenance. A refused quota read is
    // not a zero and not a full window: it licenses no usage claim at all,
    // while the catalogue and account facets beside it stay authoritative.
    if (quotaRead !== READ_OK) {
        return { label: 'Limits not checked', exhausted: false, resetsAt: '', tone: 'muted' };
    }
    const freshRowsFor = (wantedSubject) => (snapshots || []).filter((snap) => {
        const subject = snap?.subject || {};
        if (String(subject.harness || '') !== String(harnessId)) return false;
        // EXACT subject, including the default account's empty id. The old
        // `!subjectId ||` wildcard made the native row match EVERY subject on the
        // harness, so the default account reported a named profile's exhausted
        // window — red styling and all — as its own.
        if (String(subject.subject_id || '') !== String(wantedSubject)) return false;
        // The RUNTIME ignores a stale reading ("an old reading must not block a lane",
        // subagents.py `_exhausted_window`), so a card that paints one red is
        // reporting a block that will not happen: the lane still dispatches. Same bar,
        // same answer, on both sides of the glass.
        return String(snap?.freshness || '') === 'fresh';
    });
    let rows = freshRowsFor(subjectId);
    // DUAL-KEYED SUBJECT (unified migration window, plan §K.3): the quota
    // journal is never rewritten, so right after the engine migrates a default
    // login onto its named registry row the row's fresh window can still be
    // keyed by the LEGACY empty/null subject until the next refresh re-keys
    // it. The caller names the legacy aliases this exact row may inherit
    // (`fallbackSubjectIds`); they are consulted ONLY when the exact subject
    // has no fresh window, so a re-keyed reading always wins and no other
    // account's window can be borrowed (the aliases still match the same
    // harness, exactly).
    for (const alias of fallbackSubjectIds) {
        if (rows.length) break;
        if (String(alias) === String(subjectId)) continue;
        rows = freshRowsFor(alias);
    }
    let worst = null;
    // The runtime's own bar, per snapshot: spent when a constraint is cooling down OR
    // its window is fully used with a future reset — ANY constraint, not just the highest
    // ratio. Reading exhaustion off `worst` alone hid a cooling constraint whenever
    // some other window happened to report a larger used_ratio, and dropped it
    // entirely when the cooling one reported no ratio at all.
    let exhausted = false;
    let unknown = false;
    let exhaustedResetsAt = '';
    const scopedSpent = [];
    for (const snap of rows) {
        for (const constraint of snap.constraints || []) {
            const used = Number(constraint.used_ratio);
            const fact = quotaConstraintFact(constraint, nowMs);
            const spent = fact.exhausted;
            const models = Array.isArray(constraint.applies_to_models)
                ? constraint.applies_to_models.filter(Boolean) : [];
            if (models.length) {
                // A non-null applies_to_models is a PER-MODEL cap — the daemon
                // schema's own words: "a model-specific cap never cools a
                // different model on the same subject" (@claudexor/schema
                // quota.ts). So it must never paint the whole account
                // exhausted, and its ratio is not the account's bar: a spent
                // scope becomes a compact note beside the account label.
                if (spent || fact.unknown) {
                    const label = String(constraint.label || constraint.id || models.join(', '));
                    scopedSpent.push(`${label} ${spent ? 'spent' : 'availability not proven'}`);
                }
                continue;
            }
            if (spent && !exhausted) {
                exhausted = true;
                exhaustedResetsAt = fact.resetsAt;
            }
            unknown ||= fact.unknown;
            if (!Number.isFinite(used)) continue;
            if (!worst || used > worst.used) {
                worst = { used, resetsAt: fact.unknown ? ''
                    : String(constraint.resets_at || constraint.cooldown_until || '') };
            }
        }
    }
    const note = [...new Set(scopedSpent)].join(', ');
    const resetsAt = exhausted ? (exhaustedResetsAt || worst?.resetsAt || '') : (worst?.resetsAt || '');
    const resets = humanizeResetAt(resetsAt, nowMs);
    let base = '';
    if (exhausted) base = `Limit reached${resets ? ` · resets ${resets}` : ''}`;
    else if (worst) {
        base = `${Math.min(100, Math.round(worst.used * 100))}% used${resets ? ` · resets ${resets}` : ''}`;
        if (unknown) base += ' · availability not proven';
    }
    // Match typed gaps by the exact current subject only: the legacy default
    // alias belongs to retained snapshots, never to another subject's
    // credential verdict. Claudexor suppresses snapshot-covered absences at
    // its response boundary; a contradictory future body stays visible here
    // rather than authorizing a stronger consumer-side interpretation.
    const absence = (Array.isArray(absences) ? absences : []).find((row) => {
            const subject = row?.subject;
            if (!subject || typeof subject !== 'object') return false;
            if (typeof subject.harness !== 'string') return false;
            if (subject.subject_id !== null && typeof subject.subject_id !== 'string') return false;
            return subject.harness === String(harnessId)
                && String(subject.subject_id || '') === String(subjectId);
        }) || null;
    const reason = typeof absence?.reason === 'string' ? absence.reason : '';
    const labels = {
        refresh_failed: 'Usage refresh failed',
        rate_limited: 'Usage check rate-limited',
        probe_skipped_rate_limited: 'Usage check paused after a rate limit',
        poll_paced: 'Usage check paced',
        not_logged_in: 'Usage unavailable · not signed in',
        auth_revoked: 'Usage unavailable · sign-in revoked',
    };
    const absenceLabel = reason ? (labels[reason] || 'Usage unavailable') : '';
    // Claudexor's absence detail is already redacted at the producer. Display
    // it as text only; it never chooses reason, tone, login action, or routing.
    const absenceDetail = reason && typeof absence?.detail === 'string'
        ? absence.detail.trim() : '';
    const gap = [absenceLabel, absenceDetail].filter(Boolean).join(' · ');
    if (!base && !note) {
        return {
            label: gap || 'Usage unavailable',
            exhausted: false,
            resetsAt: '',
            tone: 'muted',
        };
    }
    return {
        exhausted,
        resetsAt,
        tone: exhausted ? 'warn' : 'muted',
        label: [base, note, gap].filter(Boolean).join(' · '),
    };
}

// The profile-id alphabet lives with the login card now (its own "name the
// account" face applies the same validation the Add-account dialog does);
// re-exported so this module keeps its established import path.
export { normalizeProfileName };

export async function promptProfileName({ dialogImpl = openConfirmDialog, family = '' } = {}) {
    // pywebview's WKWebView implements no window.prompt — it answers null
    // silently, so the old prompt()-based Add-account flow was a dead button on
    // the desktop app. The in-house input dialog asks instead, and it loops
    // until the typed name already IS its normalized form: a name that
    // normalization would change ("Work" → "work") is shown back, editable,
    // BEFORE any login starts — never rewritten silently — and a name nothing
    // slug-legal survives of ("Работа") re-asks with the engine's contract
    // spelled out instead of offering an illegal all-separator id.
    let initialValue = '';
    let body = `Name for the additional ${family || 'agent'} account (e.g. work, backup).`
        + ' Lowercase letters, digits, "-" and "_" — anything else becomes "-".';
    for (;;) {
        const answer = await dialogImpl({ title: 'Add account', body, input: true, initialValue });
        if (!answer?.confirmed) return '';
        const raw = String(answer.value || '').trim();
        const normalized = normalizeProfileName(raw);
        if (!raw) return '';  // a confirmed BLANK keeps meaning "never mind"
        if (!normalized) {
            // A typed name nothing slug-legal survives of (engine contract
            // ^[a-z0-9][a-z0-9_-]{0,63}$ — e.g. no ASCII alphanumerics at
            // all): re-ask with the contract spelled out instead of
            // abandoning the add or submitting a name the engine refuses.
            initialValue = '';
            body = `"${raw}" cannot become an account name. `
                + 'Enter a name that starts with a lowercase letter or digit — '
                + 'letters, digits, "-" and "_", at most 64 characters.';
            continue;
        }
        if (normalized === raw) return normalized;
        initialValue = normalized;
        body = `"${raw}" will be saved as "${normalized}" — edit the name or continue.`;
    }
}

export function runtimeActionLabel(payload) {
    const state = String(payload?.daemon?.runtime?.state || '');
    if (state === 'error') return 'Fix & connect';
    if (state === 'missing') return 'Install & connect';
    if (state === 'update_available') return 'Update & connect';
    return 'Connect';
}

// The facets the status contract declares, restated here for this module's
// own iteration order. The AUTHORITY is the store's `STATUS_FACETS` — that is
// the literal tests/test_gateway_parity.py greps and compares with
// `ClaudexorStatusReads` — and a node pin welds this spelling to the store's,
// so the two cannot drift and this list inherits the contract transitively.
export const READ_FACETS = ['catalog', 'accounts', 'quota'];

export function unreadFacets(payload) {
    // Which facets did NOT answer, in contract order. Empty means everything
    // this payload promises was actually read. Derived through the STORE's one
    // reader — a second parse of the `reads` block here is exactly the
    // two-readers-disagreeing bug the store was extracted to end.
    return STATUS_FACETS.filter((facet) => facetReadState(payload, facet) !== READ_OK);
}

export function daemonAnswered(payload) {
    // Did the daemon ANSWER? A disjunction whose halves prove different things.
    // An authenticated `running` is positive evidence on its own — the
    // handshake happened. Anything else is NOT evidence of silence: a PARTIAL
    // refusal (quota times out while the catalog and the account store land) is
    // reported as `daemon.state = 'unreachable'`, so a predicate written on the
    // literal `running` called a daemon dead while its own accounts were on
    // screen — it kept a failed wake's error standing over them and made
    // Refresh offer to start something already answering. There a facet's own
    // `ok` is the evidence. What the aggregate can never be is the NEGATIVE
    // answer.
    if (String(payload?.daemon?.state || '') === 'running') return true;
    return STATUS_FACETS.some((facet) => facetReadState(payload, facet) === READ_OK);
}

// The Refresh button's honest label. It only ever RE-READS while the daemon is
// alive, but with a sleeping daemon a plain re-read returns the same nothing
// forever — so there it becomes an explicit owner action that STARTS the
// daemon, and the label says so rather than hiding the side effect. ONE
// predicate behind BOTH the label and the click, so they cannot drift apart
// again (they were written separately once, and did).
export function refreshActionKind(payload) {
    return daemonAnswered(payload) ? 'refresh' : 'wake';
}

export function refreshActionLabel(payload) {
    return refreshActionKind(payload) === 'refresh'
        ? 'Refresh'
        : 'Check accounts (starts the agent daemon)';
}

const capitalize = (text) => (text ? `${text.charAt(0).toUpperCase()}${text.slice(1)}` : text);

function facetGapNames(reads) {
    // The unread facets a sentence may NAME, in the store's own subjects.
    // `indeterminate` is excluded on the store's rule — it is not a verdict
    // ABOUT a facet, so naming subjects under it would invent the per-facet
    // accusation the coarse state exists to avoid — and `unread` is just this
    // client's first read still in flight.
    const named = STATUS_FACETS
        .filter((facet) => reads[facet] !== READ_OK
            && reads[facet] !== READ_INDETERMINATE && reads[facet] !== READ_UNREAD)
        .map((facet) => FACET_SUBJECT[facet] || facet);
    return joinSubjects(named);
}

export function daemonStatusLine(payload, { checking = false, reads = null } = {}) {
    const daemon = payload?.daemon || {};
    const runtime = daemon.runtime || {};
    const runtimeState = String(runtime.state || '');
    const status = String(daemon.state || 'unknown');
    // Nothing read yet and a read in flight: SAY so, and say what it costs. The
    // daemon re-probes every agent CLI on each read, so first paint is tens of
    // seconds — and an unexplained silent panel reads as "broken", not as
    // "loading" (owner report, 2026-08-08).
    if (checking && !daemon.state) {
        return { tone: 'muted', text: 'Checking Claudexor… the first read probes each agent CLI and can take a minute or more.' };
    }
    if (daemon.ownership_problem) {
        return { tone: 'error', text: `This daemon home is not managed from here: ${daemon.ownership_problem}` };
    }
    // A facet can fail WITHOUT the aggregate hearing about it: an envelope that
    // arrived in the wrong shape is a failed read, not an exception, so the
    // daemon still reports `running`. The panel then said "Claudexor ready" in
    // green while the row underneath said the accounts were not checked — one
    // screen, two contradictory claims, the reassuring one on top. The status
    // line asks the FACETS, not the aggregate, and it does so BEFORE the
    // runtime branches, which used to return above the facet logic and hide the
    // gaps entirely.
    const facetReads = reads || readsFor(payload);
    const unread = STATUS_FACETS.filter((facet) => facetReads[facet] !== READ_OK);
    const gapNames = facetGapNames(facetReads);
    const unreadTail = gapNames ? ` ${capitalize(gapNames)} were not read.` : '';
    if (runtimeState === 'installing') {
        // The branch condition IS the phase: say "installing", not a hedge
        // that makes a minutes-long download read like a sub-second probe.
        return { tone: 'muted', text: `${claudexorPreparationLine(payload)}${unreadTail}` };
    }
    if (runtimeState === 'error') {
        const detail = runtime.last_error ? `: ${runtime.last_error}.` : '.';
        return { tone: 'error', text: `Claudexor needs repair${detail} Connect retries automatically.${unreadTail}` };
    }
    // The staged-update line asserts the current engine is still SERVING. Only
    // say that when this reading actually saw it serve; otherwise the facets
    // own the line and the staged update is a footnote — "Engine X keeps
    // running" was a positive claim about a daemon that, in that window,
    // answered nothing, printed over a button offering to START it.
    if (runtimeState === 'update_staged' && !unread.length) {
        const target = runtime.staged_version || runtime.target_version || '?';
        const current = daemon.engine_version || runtime.version || '?';
        return { tone: 'warn', text: `Claudexor ${target} is ready and will activate after the daemon next restarts. Engine ${current} keeps running until then.` };
    }
    if (runtimeState === 'update_staged') {
        const target = runtime.staged_version || runtime.target_version || '?';
        const gap = gapNames ? `${capitalize(gapNames)} were not read` : 'The status answer did not complete';
        return { tone: 'warn', text: `${gap}${daemon.last_error ? `: ${daemon.last_error}` : ''}. Claudexor ${target} is staged and activates after the daemon next restarts.` };
    }
    if (status === 'running') {
        // A REAL refusal (a read that was made and did not land) demotes the
        // green line: "ready" would be an overclaim about the facets. A gap
        // that is merely `not_read` keeps the ready line — the daemon itself
        // is proven up, and the tab's banner note (the store's own sentence)
        // explains a read nobody made.
        const refused = unread.filter((facet) =>
            facetReads[facet] === READ_FAILED || facetReads[facet] === READ_TRANSPORT);
        if (refused.length) {
            const names = joinSubjects(refused.map((facet) => FACET_SUBJECT[facet] || facet));
            return { tone: 'warn', text: `Claudexor is running, but ${names} were not read${daemon.last_error ? `: ${daemon.last_error}` : ''}. What those cover is unknown.` };
        }
        return { tone: 'ok', text: `Claudexor ready (engine ${daemon.engine_version || '?'}) · home ${payload.config_dir || ''}` };
    }
    if (status === 'not_provisioned') {
        if (runtimeState === 'ready') {
            const version = runtime.version ? ` ${runtime.version}` : '';
            return { tone: 'muted', explainsUnread: true, text: `Claudexor${version} is ready. Connect an account to start Ouroboros’s own agent daemon.` };
        }
        return { tone: 'muted', explainsUnread: true, text: 'No accounts connected yet. Connect installs Claudexor and starts Ouroboros’s own agent daemon automatically.' };
    }
    if (status === 'stale') {
        // NOT a warning: the daemon is LAZY by design (the status read never
        // spawns it), so "home exists, nothing answering" is the ordinary idle
        // state, not a fault. Lead with what is true and what happens next; a
        // genuine RUNTIME fault renders through the `error` branch above.
        // Disclosed residual (both review lenses, 2026-08-08): `stale` is also
        // what a CRASHED daemon lands in — the state machine cannot tell the two
        // apart (the detail lives only in last_error, which the warn-toned line
        // never showed either), so the only thing a crash loses here is the
        // alarming tone. The sentence stays true for it: ensure_running restarts
        // a dead daemon on the next login or delegated run, and a crash mid-run
        // surfaces through that run's own typed failure, not this panel. Hence
        // no "yet" — that word would claim it had never started.
        const version = runtime.version ? ` ${runtime.version}` : '';
        return { tone: 'muted', explainsUnread: true, text: `Claudexor${version} is installed; the agent daemon is not running. It starts automatically on the next login or delegated run.` };
    }
    if (status === 'foreign_daemon') {
        return { tone: 'warn', text: 'Another daemon answered on the stale port (not ours — left untouched). The next login restarts our own daemon on a fresh port.' };
    }
    if (daemonAnswered(payload) && gapNames) {
        // A PARTIAL refusal: the aggregate says `unreachable` because one read
        // died, but the others landed and their rows are on screen right now.
        // Announcing a dead daemon above accounts it just handed over is the
        // same false verdict as an unread store rendered empty. NAME the facets
        // that did not answer — "what is shown below was read" was itself an
        // overclaim when the accounts facet is the one that failed.
        return { tone: 'warn', text: `${capitalize(gapNames)} were not read${daemon.last_error ? `: ${daemon.last_error}` : ''}. What those cover is unknown.` };
    }
    return { tone: 'error', text: `Daemon ${status}${daemon.last_error ? `: ${daemon.last_error}` : ''}` };
}

// The agent families a fresh install can connect BEFORE the daemon exists.
// Discovery needs a running daemon, and on first run there is none — so with
// nothing discovered the UI still offers a Connect affordance, and the first
// Connect is exactly what provisions the owned daemon (D30). Presentation
// only; the login flow itself stays harness-agnostic. agy (Antigravity) is
// deliberately NOT bootstrapped: it has no engine-default credential store,
// so a pre-discovery card could only refuse — its card appears from live
// discovery the moment the engine answers.
export const BOOTSTRAP_HARNESSES = ['codex', 'claude', 'cursor'];

// The display name comes from the store, which owns the payload it reads and is
// imported by BOTH this tab and the onboarding wizard. Re-exported so this
// module keeps its established import path.
export { familyLabel };

// Re-exported so the accounts surface keeps ONE import path for the payload
// projection it renders (the definition lives with the store that owns the
// payload).
export { accountRows };

export function bareRowStatusText(accountsRead) {
    // The verdict for a family with NO row. "no account connected" is a claim
    // about the ACCOUNT STORE, and it may only be made once that store was
    // actually read: an idle daemon is never asked, so the emptiness says
    // nothing (BIBLE P1 — a gap is not a zero). The Connect button stays in
    // every case; onboarding must remain reachable.
    if (accountsRead === READ_OK) return 'No account connected';
    if (accountsRead === READ_UNREAD) return 'Checking…';
    if (accountsRead === READ_TRANSPORT) return 'Not checked — the status request did not complete';
    if (accountsRead === READ_FAILED) return 'Not checked — the daemon did not answer this read';
    // The coarse state: the answer did not complete, and it does not say which
    // read was the one that failed — so the row claims nothing beyond that.
    // Without this branch a legacy payload's global refusal fell through to
    // "never asked", which is the opposite of what happened.
    if (accountsRead === READ_INDETERMINATE) return 'Not checked — the status answer did not complete';
    // NOT READ says nobody asked; it does NOT say why. "the agent daemon is not
    // running" named a cause this row cannot see — a runtime awaiting repair, a
    // foreign daemon on the stale port and an ownership problem all arrive here
    // as the same unread facet, and the tab's ONE banner is the place that
    // explains which of them it is.
    return 'Not checked — the agent daemon was never asked';
}

export function familyStatus(rows, { accountsRead = READ_OK } = {}) {
    // The aggregate lozenge in a card header: how many accounts ROTATION can
    // actually use. That claim may only count accounts that are BOTH signed in
    // and enabled — a disabled row is the owner's own exclusion, and a header
    // that counted it would over-promise the pool exactly the way counting a
    // cold row used to.
    if (!rows.length) return { tone: 'muted', label: bareRowStatusText(accountsRead) };
    // …and the SAME provenance rule the rows obey. These rows are the retained
    // snapshot's memory when the accounts facet did not land, so a green
    // "Connected" over them is the row badge's lie one level up — and the two
    // would contradict each other inside one card, the header claiming fresh
    // while the badge under it says last known.
    const known = accountsRead === READ_OK;
    const verdict = (tone, label) => (known
        ? { tone, label }
        : { tone: tone === 'error' ? 'error' : 'muted', label: `${label} — last known` });
    // "Need attention" obeys the same exclusion: a DISABLED account is the
    // owner's own removal from rotation, so its failed verification is not a
    // family-level alarm — rotation never takes it, and the row's own badge
    // still shows the error in place. Counting it here turned the header red
    // over an account the owner had already dealt with.
    const bad = rows.filter((row) => row.enabled !== false
        && verificationBadge(row).tone === 'error').length;
    if (bad) {
        return verdict('error', `${bad} of ${rows.length} need attention`);
    }
    // Enabled state is the structural owner choice. In particular, an engine
    // deliberately does not probe disabled rows and reports `not_run`; that
    // unknown verification must not hide the stronger fact that every account
    // has been excluded from rotation.
    if (rows.every((row) => row.enabled === false)) {
        return verdict('muted', `${rows.length} account${rows.length === 1 ? '' : 's'} · all disabled`);
    }
    const live = rows.filter((row) => String(row?.status?.verification || '') === 'passed'
        && row.enabled !== false).length;
    if (!live) {
        const notRun = rows.some((row) => String(row?.status?.verification || '') === 'not_run');
        if (notRun) return verdict('muted', `${rows.length} account${rows.length === 1 ? '' : 's'} · not verified`);
        return verdict('muted', `${rows.length} account${rows.length === 1 ? '' : 's'} · not signed in`);
    }
    if (live < rows.length) return verdict('ok', `${live} of ${rows.length} connected`);
    if (live === 1) return verdict('ok', 'Connected');
    return verdict('ok', `${live} accounts · rotating`);
}

export function nextUpBadge(payload, harness, { accountsRead = READ_OK } = {}) {
    // The family header's "Next up" badge: who an unpinned run would take,
    // read through the store's ONE dual-wire reader (accountPools first,
    // legacy harnessAccounts[].next_up second). '' = nothing to show — no
    // verdict in the payload, or an accounts read that did not land (a stale
    // routing claim dressed as current is the exact lie the facets exist to
    // stop). Every kind renders FAIL-SAFE: an unknown future kind becomes a
    // generic "unknown" state, never a crash and never a guess.
    if (accountsRead !== READ_OK) return '';
    const verdict = nextUpAccount(payload, harness);
    if (!verdict) return '';
    const kind = String(verdict.kind || '');
    if (kind === 'profile') {
        const id = String(verdict.profileId || '');
        return id ? `Next up: ${id}` : 'Next up: unknown';
    }
    if (kind === 'api_key_route') return 'Next up: API key (no subscription capacity)';
    if (kind === 'none') return '';
    if (kind === 'native') {
        // The legacy union's default-subject verdict. Its optional route says
        // whether the session or a configured key would actually serve it.
        return String(verdict.route || '') === 'api_key'
            ? 'Next up: API key' : 'Next up: default account';
    }
    return 'Next up: unknown';
}

export function accountName(row) {
    // The account's OWN name, in this order: the registry row's display name,
    // the identity the daemon observed (email), the machine id. The old
    // "Default CLI login" label claimed a TYPE — under the unified model every
    // account is a named row of one type, and even on a legacy engine the
    // honest name for the unnamed default is who it is signed in as. The
    // legacy pseudo-row keeps a neutral fallback when the daemon observed no
    // identity for it.
    const identity = row.identity || {};
    const named = String(row.display_name || '') || String(identity.email || '');
    if (named) return named;
    if (row.kind === 'native') return 'Default account';
    return String(row.profile_id || '') || 'Account';
}

// The quota-subject LEGACY ALIASES one row may inherit (see quotaSummary's
// dual-keyed fallback): only the unified engine's migrated default row — the
// reserved `<harness>-default` registry id, frozen contract §L.3 — may inherit
// the legacy empty/null subject its account was keyed by before migration.
// Every other row gets no alias: exact-subject matching stays the rule.
export function quotaSubjectAliases(row, payload) {
    if (!unifiedAccounts(payload)) return [];
    return row.profile_id === `${row.harness}-default` ? [''] : [];
}

export function accountMetaLine(row, payload, { quotaRead = READ_OK, nowMs = Date.now() } = {}) {
    // Line 2: everything that is NOT the account itself, in human words and at
    // muted ink. Order is the owner's — how much of the window is left, which
    // plan, who it is, when we last checked. (The former "Managed by the X
    // CLI" caption is gone: it described a separate account TYPE the unified
    // model retired, and on a legacy engine the row's actions already say
    // everything the caption did.)
    const parts = [];
    if (row.enabled === false) {
        // The one user-settable routing fact, stated where the rotation claim
        // lives: a disabled account is excluded from rotation however healthy
        // its login is.
        parts.push('disabled — excluded from rotation');
    }
    parts.push(quotaSummary(payload?.quota || [], row.harness, row.profile_id,
        { quotaRead, nowMs, fallbackSubjectIds: quotaSubjectAliases(row, payload),
          absences: payload?.quota_absences || [] }).label);
    const identity = row.identity || {};
    if (identity.plan) parts.push(String(identity.plan));
    // The email is metadata only while it is not already the row's name.
    if (identity.email && String(identity.email) !== accountName(row)) {
        parts.push(String(identity.email));
    }
    const at = Date.parse(String(row?.status?.last_verified_at || ''));
    if (Number.isFinite(at)) {
        const age = formatRelativeAge(at, 'just now');
        if (age) parts.push(`checked ${age}`);
    }
    return parts.filter(Boolean).join(' · ');
}

export function accountGroups(payload, {
    accountsRead = READ_OK,
    catalogKnown = false,
} = {}) {
    // One group per family, in a stable order: discovered families first (the
    // engine's own order), then any bootstrap family still missing, so a fresh
    // install shows all three cards and every one of them can be connected.
    const rows = accountRows(payload);
    const order = [];
    for (const harness of payload?.harnesses || []) {
        const id = String(harness?.id || '');
        if (id && !order.includes(id)) order.push(id);
    }
    for (const row of rows) if (!order.includes(row.harness)) order.push(row.harness);
    for (const id of BOOTSTRAP_HARNESSES) if (!order.includes(id)) order.push(id);
    return order.map((id) => {
        const own = rows.filter((row) => row.harness === id);
        return {
            harness: id,
            label: familyLabel(id, payload, { catalogKnown }),
            rows: own,
            status: familyStatus(own, { accountsRead }),
        };
    });
}

export function familyActionLabel(group, payload) {
    // The card's OWN button, and the fix for "позиции кнопок нелогичные": the
    // add affordance lives in the family header instead of hanging off one
    // privileged row. An empty family connects its default CLI login first
    // (carrying the runtime's install/repair intent); once a family has any
    // account, the button adds a NAMED one — which is what makes the accounts
    // equivalent instead of one-default-plus-extras.
    //
    // DISCLOSED RESIDUAL (adversarial review, 2026-08-09): unlike `rowActionLabel`
    // this deliberately does NOT hand the label to a runtime that needs
    // installing or repairing. The button's own action is to ASK FOR A NAME and
    // then start a login — a header reading "Fix & connect" that opens a
    // name-the-account dialog would misdescribe what the click does, and
    // dropping the name step would remove the add intent this card exists for.
    // The repair is a PREREQUISITE, not the destination: the login card
    // performs it in the foreground and reports it there, and the tab's service
    // banner already names the fault above.
    return group.rows.length ? 'Add account' : runtimeActionLabel(payload);
}

export function rowActionLabel(row, payload) {
    // A runtime that needs installing, repairing or updating owns the label —
    // that work happens first whatever the row wants. Otherwise the row says
    // what it is really offering: an account that HAS a session signs in again,
    // one that does not simply signs in. ("Connect" belongs to a family with no
    // account yet, where it is the first step rather than a repeat.)
    return rowLoginAction(row, payload).label;
}

/**
 * The engine's explicit auth-probe-unknown state is not permission to start a
 * new login. `availability=unknown` + `verification=not_run` means that the
 * probe could not decide; only an explicit `unavailable`/`failed` verdict may
 * offer the sign-in action. Older engines omit `availability`, so they retain
 * the legacy behavior. The top-level Refresh action re-runs the status probe.
 */
export function loginStatusUnknown(row) {
    const status = row?.status || {};
    return String(status.availability || '') === 'unknown'
        && String(status.verification || '') === 'not_run';
}

export function rowLoginAction(row, payload) {
    const runtime = runtimeActionLabel(payload);
    if (runtime !== 'Connect') return { label: runtime, refresh: false };
    if (loginStatusUnknown(row)) {
        // Keep recovery available for every harness.  An unknown probe is not
        // proof of logout, but it also must not strand a profile whose next
        // status read may become a clean login verdict.  The click handler
        // below runs the shared Refresh path instead of starting OAuth.
        return { label: 'Check status', refresh: true };
    }
    return {
        label: String(row?.status?.verification || '') === 'passed'
            ? 'Sign in again' : 'Sign in',
        refresh: false,
    };
}

// "agents", "agents and limits", "agents, accounts and limits".
function joinSubjects(names) {
    const list = names.filter(Boolean);
    if (list.length <= 1) return list[0] || '';
    return `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`;
}

const TONE_RANK = { ok: 0, muted: 0, warn: 1, error: 2 };

// A note whose only content is "we did not check" yields to a service line that
// EXPLAINS why nothing was read. A warn/error note reports a real read failure
// and keeps its place, because the service line cannot know which read died.
const GENERIC_FACET_NOTE_YIELDS = new Set(['muted']);

function faultOutranksReassurance(service, note) {
    // A MUTED note is a reassurance: "nothing below is missing or wrong". It
    // may not be the last word while the service line has a FAULT to report.
    // Every settled non-running state — runtime `error`, `foreign_daemon`, an
    // ownership problem, a recorded daemon `last_error` — leaves all three
    // facets unread, so the benign note used to be the ONLY sentence the owner
    // saw while the row buttons beside it offered "Fix & connect". The whole
    // error/warn vocabulary daemonStatusLine already speaks was unreachable
    // there. A warn/error note (a refused read, a dead request) is itself a
    // report and keeps its place.
    // Precedence is by SPECIFICITY, not by tone. A muted service line can still
    // be the more informative sentence: on a first run "No accounts connected
    // yet. Connect installs Claudexor…" is exactly what the owner needs, and it
    // was unreachable while only warn/error could win — every stopped state
    // leaves all three facets unread, so the generic note always spoke instead.
    // The generic note explains nothing the service line does not; it is the
    // FALLBACK for when the service line has nothing concrete to say.
    if (!note) return service;
    const serviceSpeaksFirst = Boolean(service) && (
        service.tone === 'error' || service.tone === 'warn' || service.explainsUnread === true
    );
    if (GENERIC_FACET_NOTE_YIELDS.has(note.tone) && serviceSpeaksFirst) {
        return { tone: service.tone, text: service.text };
    }
    return { tone: note.tone, text: note.text };
}

export function serviceBannerLine(store, { wakeError = '', wakeBusy = false } = {}) {
    if (wakeBusy) {
        return { tone: 'muted', text: 'Starting the agent daemon…' };
    }
    // THE service banner: one place on the tab that explains a daemon/runtime
    // problem, replacing the scattering of "(not in discovery)" the owner
    // reported. Provenance is PER FACET, so this line never collapses three
    // independent reads into one verdict where the wire tells it apart: a
    // refused quota read leaves the catalogue and accounts authoritative and
    // says exactly that. The producer stamps `reads` on every answer
    // (`claudexor_accounts.py`), so this is the shape the line renders live;
    // the coarse all-indeterminate rendering remains only for a legacy payload
    // without the block.
    //
    // Deliberately NOT built on the store's `facetGapClause`, which exists for a
    // surface that LEADS with one facet and must still name the others (the
    // Delegation note does exactly that for its model select). This line leads
    // with none: it enumerates every facet it lost, in that facet's own state,
    // so the shared clause would add a second "could not be read" about facets
    // the sentence above it has already named. Same authority, one phrasing.
    //
    // A WAKE ERROR leads outright: the owner PRESSED the button and it did not
    // work — silence there is the same class of dishonesty this banner exists
    // to remove (a typed 503 from a missing binary or foreign home, a 404 from
    // an older backend, a dead network). The rows and Connect stay put; the
    // error's LIFECYCLE is the panel's (it expires only when the daemon
    // provably answers — `daemonAnswered`, deliberately not the literal
    // `running`).
    if (wakeError) {
        return { tone: 'error', text: `Could not start the agent daemon: ${wakeError}` };
    }
    const reads = store.reads || {};
    const facets = [FACET_CATALOG, FACET_ACCOUNTS, FACET_QUOTA];
    const bad = facets.filter((facet) => reads[facet] !== READ_OK);
    if (!bad.length) {
        return daemonStatusLine(store.snapshot || {}, {
            checking: store.loading && !store.everSettled,
            reads,
        });
    }
    // NOTHING READ YET is not a gap to report — it is the first read in flight,
    // and what the owner needs then is its COST: the daemon re-probes every
    // agent CLI, so first paint is tens of seconds and a silent panel reads as
    // broken rather than as loading (owner report, 2026-08-08). A bare
    // "Reading…" would have thrown that sentence away.
    if (!store.everSettled) {
        return daemonStatusLine(store.snapshot || {}, { checking: store.loading, reads });
    }
    // The service line asks the FACETS too (invariant: facets outrank both the
    // aggregate and the runtime branches), so when it wins below it already
    // names the unread facets and gates its own positive claims ("keeps
    // running", "ready") on a reading that actually saw them.
    const service = daemonStatusLine(store.snapshot || {}, { reads });
    // All three unread in the same way: ONE sentence about the service, from
    // the shared vocabulary, with the subject widened to the whole tab —
    // naming just the accounts would under-report a gap that also swallowed
    // the agent catalogue and the limits. A runtime fault outranks it.
    //
    // The sentence is asked of the STORE, never assembled here, because the
    // detail beside it is the store's to resolve: a transport error when the
    // request itself died, and otherwise — for a read that was made and did not
    // land — the daemon's OWN `last_error`. That string is the only explanation
    // an `unreachable` answer carries, and a banner that called the copy factory
    // directly printed "could not be read" and dropped it.
    const states = new Set(bad.map((facet) => reads[facet]));
    if (bad.length === 3 && states.size === 1) {
        return faultOutranksReassurance(service,
            store.unavailableNote(bad[0], { subject: 'agents, accounts and limits' }));
    }
    // A PARTIAL gap: name EVERY facet that could not be read — one sentence per
    // distinct way they failed — and let the closing reassurance cover only the
    // facets that genuinely read. Reporting `bad[0]` alone and appending
    // "everything else was read normally" told the owner two of three failures
    // had landed fine; the backend stamps `reads` per facet on every answer,
    // which is exactly what makes a mixed verdict possible.
    const sentences = [];
    let tone = 'muted';
    for (const readState of states) {
        const group = bad.filter((facet) => reads[facet] === readState);
        const subjects = group.map((facet) => FACET_SUBJECT[facet] || facet);
        // Any facet of the group answers for it — they share the read state,
        // and asking the store keeps the daemon's own reason attached.
        const note = store.unavailableNote(group[0], { subject: joinSubjects(subjects) });
        if (!note) continue;
        sentences.push(note.text);
        if (TONE_RANK[note.tone] > TONE_RANK[tone]) tone = note.tone;
    }
    if (!sentences.length) return service;
    const readOk = facets
        .filter((facet) => reads[facet] === READ_OK)
        .map((facet) => FACET_SUBJECT[facet] || facet);
    const tail = readOk.length ? ` Your ${joinSubjects(readOk)} were read normally.` : '';
    // The SAME precedence as the full-gap branch above — the two are one
    // decision, and fixing only one half of it is how this class survives. The
    // backend stamps `reads` per facet on every answer, so a mixed verdict is
    // an ordinary state — and a muted "some facets were never asked · the rest
    // read normally" must not swallow a runtime that needs repair.
    return faultOutranksReassurance(service, { tone, text: `${sentences.join(' ')}${tail}` });
}

export async function removeAccount(harness, profileId, { fetchImpl = apiFetch } = {}) {
    // The engine owns the account record, so removal is ITS contract. A failure
    // is reported as a failure — nothing here pretends an account is gone.
    const url = `/api/claudexor/credential-profiles/${encodeURIComponent(harness)}`
        + `/${encodeURIComponent(profileId)}`;
    const resp = await fetchImpl(url, { method: 'DELETE' });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(String(data?.error || `HTTP ${resp.status}`));
    return data;
}

export async function setAccountEnabled(harness, profileId, enabled, { fetchImpl = apiFetch } = {}) {
    // The Enabled toggle is the engine's own PATCH contract (the one
    // user-settable routing control a registry row carries); a refusal is the
    // answer, and nothing here pretends the pool changed.
    const url = `/api/claudexor/credential-profiles/${encodeURIComponent(harness)}`
        + `/${encodeURIComponent(profileId)}`;
    const resp = await fetchImpl(url, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: Boolean(enabled) }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(String(data?.error || `HTTP ${resp.status}`));
    return data;
}

export function removeAccountConfirmBody(name, family) {
    return `The Claudexor binding for ${family} account "${name}" will be removed. `
        + 'Vendor or OS credential storage may remain signed in and is not changed by '
        + `Ouroboros; the deletion receipt says when it was retained. Reviewer rows `
        + 'and a Delegation pin pointing at this account stay visible and are shown as '
        + 'unavailable until you repoint them.';
}

export function vendorCredentialRetainedNotice(receipt, name, family) {
    const disposition = receipt?.vendorCredentialDisposition;
    if (!disposition || disposition.owner !== 'vendor'
        || disposition.state !== 'left_unchanged'
        || disposition.scope !== 'os_user') return '';
    return `Removed "${name}" from ${family}. Claudexor left vendor credential storage `
        + 'for this OS user unchanged; the vendor account may still be signed in outside Ouroboros.';
}

// ---------------------------------------------------------------------------
// DOM section.
// ---------------------------------------------------------------------------

const state = {
    store: claudexorStatus,
    loginCard: null,
    loginFamily: '',
    disposers: [],
    removeError: '',
    removeNotice: '',
    initialized: false,
    // The owner asked to start the daemon and the wake refused. Rendered by the
    // service banner and expired ONLY when the daemon provably answers
    // (`daemonAnswered` over a fresh reading) — a 200 that reports the daemon
    // still down must NOT wipe the reason before the owner can read it, and a
    // daemon that came up on its own must not leave the error standing over
    // its accounts. The wake POST itself is the STORE's (single writer).
    wakeError: '',
    wakeBusy: false,
};

/** The tab's ONE service banner; rendered above every section. */
export function renderAgentsServiceBanner() {
    return `<div id="agents-service-banner" class="agents-service-banner settings-inline-status" data-tone="muted">Checking the agent service…</div>`;
}

export function renderAgentAccountsSection() {
    return `
        <div class="form-section" id="harness-accounts-section">
            <h3>Accounts</h3>
            <div class="settings-section-copy">
                Agent subscriptions used by delegated subagents and review lanes. Unpinned work
                rotates across a family's enabled, signed-in accounts; a disabled account keeps its
                login and stays out of rotation. Accounts live in Ouroboros's own agent home; your
                personal logins are never read or imported.
            </div>
            <div id="harness-accounts-error" class="settings-inline-status" data-tone="error" hidden></div>
            <div id="harness-accounts-groups" class="agent-family-list"></div>
            <div id="harness-login-card"></div>
            <div class="settings-toolbar">
                <button type="button" class="btn btn-default" id="btn-harness-refresh">Refresh</button>
            </div>
        </div>
    `;
}

export function accountRowFacts(row, payload,
                                { accountsRead = READ_OK, quotaRead = READ_OK, nowMs = Date.now() } = {}) {
    // Each projection is gated by ITS OWN facet: the identity claim is the
    // ACCOUNTS read, the window is the QUOTA read, and one lands while the
    // other refuses. The panel used to render both off the retained snapshot
    // regardless, so after a refused read the banner said nothing could be
    // listed while a stale row sat underneath it showing "Verified live" and a
    // red exhausted window. Pure, because that rule is the thing worth pinning.
    //
    // The two-line anatomy is the owner's (D-10): line 1 is the account and its
    // status, line 2 is muted metadata in human words. `quotaSummary` carries
    // the quota gap itself — an unread window says "Limits not checked" rather
    // than dressing a remembered percentage as current, and it never paints the
    // row red, because the exhausted styling is a claim about RIGHT NOW and the
    // reset it promises may already have happened.
    return {
        badge: verificationBadge(row, { known: accountsRead === READ_OK }),
        quota: quotaSummary(payload?.quota || [], row.harness, row.profile_id,
            { quotaRead, nowMs, fallbackSubjectIds: quotaSubjectAliases(row, payload),
              absences: payload?.quota_absences || [] }),
        name: accountName(row),
        meta: accountMetaLine(row, payload, { quotaRead, nowMs }),
    };
}

function rowHtml(row, payload, facets = {}) {
    const { badge, quota, name, meta } = accountRowFacts(row, payload, facets);
    const loginAction = rowLoginAction(row, payload);
    // Row ACTIONS follow the engine's own routes, never the row's looks: a
    // named registry row (every row on a unified engine, the named profiles on
    // a legacy one) has the PATCH Enabled toggle and DELETE Remove; the legacy
    // native pseudo-row has neither, because that engine has no route for
    // either and a dead button would claim an effect this app cannot have.
    const rowActions = row.kind === 'native' ? '' : `
                <button type="button" class="btn btn-default" data-harness-toggle data-enabled="${row.enabled === false ? '0' : '1'}" title="${row.enabled === false ? 'Let rotation use this account again' : 'Keep the login, take this account out of rotation'}">${row.enabled === false ? 'Enable' : 'Disable'}</button>
                <button type="button" class="btn btn-default" data-harness-remove title="Ask the agent service to forget this account">Remove</button>`;
    return `
        <div class="harness-account-row${quota.exhausted ? ' harness-exhausted' : ''}" data-harness="${escapeHtml(row.harness)}" data-profile="${escapeHtml(row.profile_id)}" data-kind="${escapeHtml(row.kind)}">
            <div class="harness-account-main">
                <strong>${escapeHtml(name)}</strong>
                <span class="ui-status" data-tone="${badge.tone}">${escapeHtml(badge.label)}</span>
            </div>
            <div class="harness-account-meta muted">${escapeHtml(meta)}</div>
            <div class="harness-account-actions">
                <button type="button" class="btn btn-default" data-harness-login>${escapeHtml(loginAction.label)}</button>${rowActions}
            </div>
        </div>
    `;
}

export function harnessFamilyMarkup(group, payload, facets) {
    // An empty family is a ONE-LINE card: the header already carries the verdict
    // (familyStatus falls through to it), and printing the same sentence again
    // in the body just made the card twice as tall to say nothing new.
    const body = group.rows.map((row) => rowHtml(row, payload, facets)).join('');
    const nextUp = nextUpBadge(payload, group.harness,
        { accountsRead: facets?.accountsRead ?? READ_OK });
    return `
        <section class="agent-family-card" data-family="${escapeHtml(group.harness)}">
            <div class="agent-family-head">
                <div class="agent-family-id">
                    <h4>${harnessIdentityMarkup(group.harness, { label: group.label })}</h4>
                    <span class="ui-status" data-tone="${group.status.tone}">${escapeHtml(group.status.label)}</span>
                    ${nextUp ? `<span class="ui-status" data-tone="muted" data-next-up>${escapeHtml(nextUp)}</span>` : ''}
                </div>
                <button type="button" class="btn btn-default" data-family-add>${escapeHtml(familyActionLabel(group, payload))}</button>
            </div>
            <div class="agent-family-login" data-family-login="${escapeHtml(group.harness)}"></div>
            <div class="agent-family-rows">${body}</div>
        </section>
    `;
}

function renderRows() {
    // The wake error expires HERE, on the one condition that makes it moot: a
    // FRESH reading (store error retired) in which the daemon provably
    // answered. A refusal was only ever news while nothing answered; keeping
    // it over accounts the daemon just handed over would be the stale-error
    // twin of the stale-absence lie.
    if (state.wakeError && !state.store.error && daemonAnswered(state.store.snapshot)) {
        state.wakeError = '';
    }
    const host = document.getElementById('harness-accounts-groups');
    const banner = document.getElementById('agents-service-banner');
    if (banner) {
        const line = serviceBannerLine(state.store, {
            wakeError: state.wakeError,
            wakeBusy: state.wakeBusy,
        });
        banner.textContent = line.text;
        banner.dataset.tone = line.tone;
    }
    // The Refresh button says what pressing it does (one predicate feeds the
    // label AND the click — see initHarnessAccounts), and while a wake is in
    // flight it says that instead of inviting a second one.
    const refreshEl = document.getElementById('btn-harness-refresh');
    if (refreshEl) {
        refreshEl.textContent = state.wakeBusy
            ? 'Starting the agent daemon…'
            : refreshActionLabel(state.store.snapshot);
        refreshEl.disabled = Boolean(state.wakeBusy);
    }
    if (!host) return;
    const errorBox = document.getElementById('harness-accounts-error');
    if (errorBox) {
        const text = state.removeError || state.removeNotice;
        errorBox.hidden = !text;
        errorBox.textContent = text;
        errorBox.dataset.tone = state.removeError ? 'error' : 'warn';
    }
    const payload = state.store.snapshot || {};
    const accountsRead = state.store.facet(FACET_ACCOUNTS);
    const quotaRead = state.store.facet(FACET_QUOTA);
    // The family-mounted login card lives INSIDE this container, so the
    // innerHTML rebuild destroys a focused paste-code/name input before the
    // card's own render can see it. The capture must therefore wrap the
    // whole rebuild: same SSOT helper, one level up.
    preserveCardFocus(host, () => {
        host.innerHTML = accountGroups(payload, {
            accountsRead,
            catalogKnown: state.store.catalogKnown,
        })
            .map((group) => harnessFamilyMarkup(group, payload, { accountsRead, quotaRead })).join('');
        host.querySelectorAll('[data-harness-login]').forEach((button) => {
            button.addEventListener('click', () => {
                if (!state.initialized) return;
                const row = button.closest('[data-harness]');
                const rowData = row?.dataset || {};
                const rowModel = accountRows(payload).find((candidate) =>
                    String(candidate?.harness || '') === String(rowData.harness || '')
                    && String(candidate?.profile_id || '') === String(rowData.profile || '')
                );
                const action = rowLoginAction(rowModel, payload);
                if (action.refresh) {
                    state.store.refresh();
                    return;
                }
                startLogin(rowData.harness, rowData.profile);
            });
        });
        host.querySelectorAll('[data-harness-remove]').forEach((button) => {
            button.addEventListener('click', () => {
                const row = button.closest('[data-harness]');
                confirmRemoveAccount(row?.dataset.harness, row?.dataset.profile);
            });
        });
        host.querySelectorAll('[data-harness-toggle]').forEach((button) => {
            button.addEventListener('click', () => {
                const row = button.closest('[data-harness]');
                // The button carries the state it RENDERED for, so a click flips
                // exactly what the owner saw — never a re-read of a row the poll
                // may have replaced mid-click.
                toggleAccountEnabled(row?.dataset.harness, row?.dataset.profile,
                    button.dataset.enabled === '0');
            });
        });
        host.querySelectorAll('[data-family-add]').forEach((button) => {
            button.addEventListener('click', async () => {
                if (!state.initialized) return;
                // Captured before the await: the status poll replaces the cards
                // while the dialog is open, detaching this button's section.
                const card = button.closest('[data-family]');
                const harness = card?.dataset.family;
                const hasRows = Boolean(card?.querySelector('[data-harness]'));
                if (!hasRows) { startLogin(harness, ''); return; }
                const profile = await promptProfileName({
                    family: familyLabel(harness, state.store.snapshot || {}, {
                        catalogKnown: state.store.catalogKnown,
                    }),
                });
                if (profile) startLogin(harness, profile);
            });
        });
        state.loginCard?.render();
    });
}

async function toggleAccountEnabled(harness, profileId, enabled) {
    // No confirm dialog: the toggle is reversible in one click and destroys
    // nothing (the login material stays). Failures ride the same inline error
    // box as removal, and the fresh read repaints the row from engine truth.
    if (!harness || !profileId) return;
    state.removeError = '';
    state.removeNotice = '';
    try {
        await setAccountEnabled(harness, profileId, enabled);
    } catch (error) {
        state.removeError = `Could not ${enabled ? 'enable' : 'disable'} "${profileId}": `
            + `${error.message || error}. The account is unchanged.`;
    }
    await state.store.refresh();
    renderRows();
}

/**
 * Complete destructive flow behind a row's Remove button. Injectable deps let
 * the node suite drive the production handler; confirm mode authorizes only on
 * its documented strict boolean `true`, never the input mode's object shape.
 */
export async function confirmRemoveAccount(harness, profileId, {
    dialogImpl = openConfirmDialog,
    removeImpl = removeAccount,
    store = state.store,
    renderImpl = renderRows,
} = {}) {
    if (!harness || !profileId) return;
    const family = familyLabel(harness, store.snapshot || {}, {
        catalogKnown: store.catalogKnown,
    });
    const answer = await dialogImpl({
        title: 'Remove account',
        body: removeAccountConfirmBody(profileId, family),
        confirmLabel: 'Remove',
        danger: true,
    });
    if (answer !== true) return;
    state.removeError = '';
    state.removeNotice = '';
    try {
        const receipt = await removeImpl(harness, profileId);
        state.removeNotice = vendorCredentialRetainedNotice(
            receipt, profileId, family);
    } catch (error) {
        state.removeError = `Could not remove "${profileId}": ${error.message || error}. `
            + 'The account is unchanged.';
    }
    await store.refresh();
    renderImpl();
}

/**
 * OWNER action behind the Refresh button when the daemon is asleep: start it,
 * then take the fresh reading. The POST and the commit belong to the STORE
 * (single writer — `store.wake()` serializes against the poll in both orders);
 * what lives here is the error's LIFECYCLE: shown only while it still matters
 * (a refusal is not news once the daemon answers — an ordinary poll can commit
 * a live reading while the wake is in flight), and expired only by a daemon
 * that provably answered (see renderRows).
 */
export async function wakeDaemon() {
    if (state.wakeBusy) return;
    state.wakeBusy = true;
    state.wakeError = '';
    renderRows();
    let result;
    try {
        result = await state.store.wake();
    } finally {
        state.wakeBusy = false;
    }
    if (!result?.ok) {
        state.wakeError = daemonAnswered(state.store.snapshot)
            ? ''
            : String(result?.error || 'request failed');
    }
    renderRows();
}

// Harness ids are conservative tokens; escape defensively for the attribute
// selector without depending on the browser-only CSS.escape (node tests).
function familyLoginSelector(harness) {
    return `[data-family-login="${String(harness).replace(/["\\]/g, '\\$&')}"]`;
}

function ensureLoginCard() {
    if (state.loginCard && !state.loginCard.disposed) return state.loginCard;
    // `detach()` permanently fences one controller. Explicit Connect after a
    // destroy/re-init must therefore build a fresh controller instead of
    // reusing a cached disposed object whose start() correctly does nothing.
    state.loginCard = null;
    state.loginCard = createLoginCardController({
        host: () => (
            (state.loginFamily
                && document.querySelector?.(familyLoginSelector(state.loginFamily)))
            || document.getElementById('harness-login-card')
        ),
        store: state.store,
        // The Settings face is the FULL card: paste-code entry, engine detail,
        // the collapsed Advanced terminal fallback, Close.
        mode: 'full',
        onSettled: () => renderRows(),
    });
    return state.loginCard;
}

/**
 * Start (or restart) a login for one account row. Exported because the account
 * rows, the Add-account dialog and the browser smoke tests all drive it.
 */
export async function startLogin(harness, profile) {
    if (!harness || !state.initialized) return;
    state.loginFamily = String(harness);
    const card = ensureLoginCard();
    await card.start(harness, profile);
    // The card mounts inside the clicked family's block; make sure a long
    // account list has not left it above/below the viewport.
    document.querySelector?.(familyLoginSelector(state.loginFamily))
        ?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
}

/** Read the shared status once (the Refresh button, and the first paint). */
export function refreshHarnessStatus() {
    return state.store.refresh();
}

/**
 * Opening Agents is an explicit owner action. Refresh first so an old live
 * snapshot cannot hide a daemon reaped by a server restart; only the exact
 * already-provisioned idle state is then restarted. First-time installation,
 * foreign ownership and repair remain behind Connect.
 */
async function refreshHarnessStatusOnActivation() {
    const snapshot = await state.store.refresh();
    // The store deliberately retains the last answer after a failed GET so
    // the panel can keep useful context.  That retained snapshot is not fresh
    // evidence for an owner-triggered write: a transient outage must never
    // turn yesterday's `stale + ready` into a wake request.
    if (state.store.error) return null;
    const daemon = snapshot?.daemon || state.store.snapshot?.daemon || {};
    if (String(daemon.state || '') === 'stale'
        && String(daemon.runtime?.state || '') === 'ready'
        && !daemon.ownership_problem) {
        return wakeDaemon();
    }
    return null;
}

/**
 * Mount the section. The exported destroy seam is an honest local detach, so
 * remount never waits on or invents daemon release proof.
 */
export function initHarnessAccounts({ store = claudexorStatus } = {}) {
    return _init(store);
}

async function _init(store) {
    _destroy();
    state.store = store;
    state.removeError = '';
    state.removeNotice = '';
    state.wakeError = '';
    state.wakeBusy = false;
    ensureLoginCard();
    document.getElementById('btn-harness-refresh')
        ?.addEventListener('click', () => {
            if (!state.initialized) return;
            // A sleeping daemon cannot be re-read into existence: there the
            // button is the owner's explicit start. Live, it stays a plain
            // re-read. SAME predicate the LABEL uses (renderRows), so the two
            // cannot disagree again.
            return refreshActionKind(state.store.snapshot) === 'refresh'
                ? state.store.refresh()
                : wakeDaemon();
        });
    // The SHARED surface binding: the visibility predicate that lets this
    // section keep the poll armed, and the catch-up read when the panel becomes
    // reachable — one implementation, released by one disposer. It carries no
    // tab NAME on purpose, and this section is the proof: it moved from
    // Providers to Agents in this very sprint, so a hardcoded tab name would
    // have gone quietly dead on arrival while its comment still promised that
    // a daemon coming up is picked up without a reload.
    state.disposers.push(bindStatusSurface(state.store, {
        listener: () => renderRows(),
        elementId: 'harness-accounts-groups',
        onActivate: refreshHarnessStatusOnActivation,
    }));
    state.initialized = true;
    // The first read must not wait for the poll interval: init runs while the
    // page may not be visible yet, and the panel would sit on "Checking
    // daemon…" until the first tick (#125).
    state.store.refresh();
    renderRows();
    return true;
}

/**
 * Tear the exported Settings test/lifecycle seam down synchronously. This is a
 * local detach only: zero create/cancel/reconcile requests and no claim that a
 * daemon-owned process stopped. Production Settings remains mounted across
 * ordinary SPA navigation and never calls this as a leave hook.
 */
export function destroyHarnessAccounts() {
    return _destroy();
}

function _destroy() {
    for (const dispose of state.disposers.splice(0)) {
        try { dispose(); } catch (err) { /* a broken disposer must not block the rest */ }
    }
    state.initialized = false;
    const card = state.loginCard;
    if (!card) return true;
    card.detach();
    state.loginCard = null;
    return true;
}
