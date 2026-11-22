// Main-screen Update affordance (P2): one compact pill. While an update attempt runs it names the
// real stage; afterwards a failed or unconfirmed outcome stays on it until the owner hides it with the
// close button beside it (owner 3A, #1818); a confirmed success never wears it (owner 1B): that record
// lives in Dashboard -> Updates alone and cannot mask a newer version. Otherwise the pill appears when a
// managed update is available (status is populated by the boot-time check-on-restart). The pill is a
// pointer, not a second apply surface: clicking it opens Dashboard -> Updates, where the ONE apply flow
// (with its verified preflight and typed outcomes) lives. The former pill-local staged dialog was removed
// (owner decision 2026-08-31) so two hand-rolled apply controllers cannot drift apart again.
// With no capsule to show, the sidebar's version itself is the check for updates (owner 21.11: no Check button):
// a press runs the panel's own check (apiClient.updateCheck, one in flight for every surface on the same channel
// selection) and its answer stands in the capsule's place; it never starts an update.
// Until a status read says this install offers that check, the version is the plain text it is: never a control
// that would do nothing.

import { apiClient, updateStrategyForPlan } from './api_client.js';
import { SHELL_ICONS, fluentIcon } from './page_icons.js';
import { errorWords } from './ui_primitives.js';
import { fmt } from './i18n.js';
import { durationLabel, formatBytes } from './storage_format.js';

// Fail-soft wrapper around the api_client update helpers (the pill must never throw the app).
async function safe(fn) {
    try {
        return await fn();
    } catch {
        return null;
    }
}

// The capsule says one word (owner Q6=A; the concept of 21.11: «Обновить», no icon, no number): the versions and
// commit hashes stay in Dashboard → Updates and in the capsule's title, where updatePillDetail puts them.
export function updatePillText() {
    return 'Update';
}

export function updatePillDetail(status = {}) {
    const currentVersion = String(status.current_version || '');
    const latestVersion = String(status.latest_version || '');
    const currentSha = String(status.current_short_sha || status.current_sha || '').slice(0, 8);
    const latestSha = String(status.latest_short_sha || status.latest_sha || '').slice(0, 8);
    if (currentVersion && latestVersion && currentVersion === latestVersion) {
        return currentSha && latestSha
            ? `Update ${currentSha} → ${latestSha}`
            : `Update available${latestSha ? ` · ${latestSha}` : ''}`;
    }
    const current = currentVersion || currentSha;
    const latest = latestVersion || latestSha;
    return current && latest ? `Update ${current} → ${latest}` : 'Update available';
}

/** What an available update brings, in owner words (DESIGN principle 7): its versions, else its count of new changes;
 *  the commits they name are the record's (the Updates page's chips, the Apply dialog's Details). */
export function updateTarget(status = {}) {
    const current = status.current_version || 'unknown';
    const latest = status.latest_version || 'unknown';
    return latest !== current ? `${current} → ${latest}`
        : status.behind ? `${status.behind} new ${status.behind === 1 ? 'change' : 'changes'}` : latest;
}

// The Apply dialog's title names the versions only when the status read names the very range preflight pinned and both
// its versions are readable (never "unknown"); otherwise it is neutral, and the pinned commits lead its Details.
export const applyTitle = (status = {}, plan = {}) => (plan.base_sha && (status.running_sha ?? status.current_sha) === plan.base_sha
    && (status.checked_target_sha || status.latest_sha) === plan.target_sha && status.current_version && status.latest_version ? `Update: ${updateTarget(status)}` : 'Update');

// Shared preflight verification for the ONE apply surface (consumed by
// web/modules/updates.js): never lets an unverified plan reach updateApply.
export function verifiedUpdatePlan(preflight) {
    const plan = preflight?.merge_plan;
    if (!plan || typeof plan !== 'object') return null;
    const strategy = updateStrategyForPlan(plan);
    if (
        !strategy
        || !plan.base_sha
        || !plan.target_sha
        || !Number.isInteger(plan.local_dirty_count)
        || plan.local_dirty_count < 0
    ) return null;
    return { plan, strategy };
}

/** Only the saved Markdown for the confirmation's exact pins. The panel follows its own status reads. */
export function preflightDescription(preflight) {
    const plan = preflight?.merge_plan;
    const letter = preflight?.letter;
    return plan?.base_sha && plan.target_sha && letter?.description_current && letter.text
        && letter.key?.base_sha === plan.base_sha && letter.key?.target_sha === plan.target_sha
        ? String(letter.text) : '';
}

// The receipt's review keys this reader words; any other key is shown verbatim, never dropped.
const REVIEW_KEYS = new Set(['observed', 'declared', 'state', 'relation']);

/** One commit's review line: its observed verdict beside its record state and the subject relation its merge
 *  request saw, bound to this commit; a receipt that predates them says they were not recorded. */
function reviewLine(review) {
    const seen = 'state' in review || 'relation' in review
        ? `record ${review.state || 'not recorded'}, subject relation at the merge request ${review.relation || 'not recorded'}`
        : 'record state and subject relation at the merge request not recorded';
    const other = Object.entries(review).filter(([key]) => !REVIEW_KEYS.has(key))
        .map(([key, value]) => `; ${key} ${typeof value === 'string' ? value : JSON.stringify(value)}`).join('');
    return `Review observed ${review.observed || 'unknown'}, author declared ${review.declared || 'unknown'}; ${seen}${other}.`;
}

// A receipt's observed verdict that says its merge went in over a review FAIL or with no verdict. PASS, an absent
// or an unrecognized value is not counted: what is unknown names nothing.
const OVER_REVIEW = new Set(['FAIL', 'NOT_PERFORMED', 'QUORUM_FAILED', 'NOT_DISPATCHED', 'unknown']);
const plural = (count, word) => `${count} ${word}${count === 1 ? '' : 's'}`;
// Said wherever this install's own landed rows were not read for a range: unknown, never "none".
const NOT_READ = 'Not read for this range: whether its commits landed over a review FAIL or no verdict is unknown.';

/**
 * What a range brings over a review FAIL or no verdict, from its qualification: this install's own commits it
 * recorded so (`landed_over_review`, written when the check collected the range) and the first-parent merges whose
 * receipt observed it; a commit that is both counts once, as a commit. `line` is the one host sentence above the
 * description and `detail` the same with its items named, for Details; both say how many receipts were not read
 * (a range checked before every first-parent receipt was read) and are '' when the facts name nothing: absent is
 * unknown, never "clean".
 * @returns {{line: string, detail: string, names: string[], landed: object[]}}
 */
export function updateOverReview(facts) {
    const listed = Array.isArray(facts?.commits) ? facts.commits : [];
    const landed = Array.isArray(facts?.landed_over_review) ? facts.landed_over_review.filter((row) => row?.sha) : [];
    const own = new Set(landed.map((row) => row.sha));
    const merges = listed.filter((commit) => !own.has(commit.sha) && OVER_REVIEW.has(commit.receipt?.review?.observed));
    const unread = listed.filter((commit) => !commit.receipt && commit.body_read === false).length;
    const parts = [...(merges.length ? [plural(merges.length, 'merge')] : []),
        ...(landed.length ? [`${plural(landed.length, 'commit')} Ouroboros made on this computer`] : [])];
    const names = [...landed.map((row) => `${String(row.sha).slice(0, 8)} (made on this computer; ${row.outcome})`),
        ...merges.map((commit) => `${String(commit.sha).slice(0, 8)} (merge; ${
            commit.receipt.review.observed === 'FAIL' ? 'review FAIL' : 'no review verdict recorded'})`)];
    const said = (items) => [items.length ? `Went in over a review FAIL or no verdict: ${items.join(', ')}.` : '',
        unread ? `The review receipts of ${plural(unread, 'older commit')} were not read.` : ''].filter(Boolean);
    const blind = facts && !Array.isArray(facts.landed_over_review) ? [`Made on this computer: ${NOT_READ}`] : [];
    return { line: said(parts).join(' '), detail: [...said(names), ...blind].join(' '), names, landed };
}

/**
 * The exact range's evidence as the confirmation's detail rows: this install's own commits recorded over a review
 * outcome (or that they were not read), every first-parent commit's receipt
 * (declarations stay labelled as such; each observed verdict keeps its own record state and request-time
 * relation), the post-merge read and each observed run. Missing, unread and non-green stay distinct;
 * nothing here becomes a verdict.
 * @returns {Array<{label: string, value: string}>}
 */
export function preflightEvidenceRows(preflight) {
    const plan = preflight?.merge_plan || {};
    const facts = preflight?.qualification;
    // The range's commits lead its Details, known checks or not: the dialog's title says the versions (updateTarget).
    const range = { label: 'Commits', value: `${String(plan.base_sha || '').slice(0, 8)} → ${String(plan.target_sha || '').slice(0, 8)}` };
    if (!facts || facts.base_sha !== plan.base_sha || facts.target_sha !== plan.target_sha) {
        return [range, { label: 'Checks', value: 'Qualification for this range is unknown.' }];
    }
    const rows = [range, ...updateOverReview(facts).landed.map((row) => ({ label: String(row.sha).slice(0, 8),
        value: `${row.subject ? `${row.subject}\n` : ''}Ouroboros made it on this computer; ${row.outcome}${/[.!?]"?$/.test(row.outcome) ? '' : '.'}` }))];
    if (!Array.isArray(facts.landed_over_review)) {
        rows.push({ label: 'Made on this computer', value: NOT_READ });
    }
    const listed = facts.commits || [];
    const receipts = listed.filter((commit) => commit.receipt).length;
    const unread = listed.filter((commit) => !commit.receipt && commit.body_read === false).length;
    if (receipts < listed.length) {
        rows.push({ label: 'Review receipts', value: `${receipts} of the ${plural(listed.length, 'commit')} listed here carry one${
            unread ? `; ${unread} commit ${unread === 1 ? 'body was' : 'bodies were'} not read` : ''}. Without a receipt the review is unknown.` });
    }
    for (const commit of listed) {
        // The short SHA names the row; the subject leads its value, where a long one has room.
        const label = String(commit.sha).slice(0, 8);
        const lines = commit.subject ? [String(commit.subject)] : [];
        if (!commit.receipt) {
            lines.push(commit.body_read === false ? 'Commit body not read; qualification unknown.' : 'No qualification receipt.');
            rows.push({ label, value: lines.join('\n') });
            continue;
        }
        for (const check of commit.receipt.checks || []) {
            const line = `${check.name} ${check.status} (${check.source || 'unknown source'})${check.detail ? `; ${check.detail}` : ''}`;
            lines.push(/[.!?]$/.test(line) ? line : `${line}.`);
        }
        lines.push(reviewLine(commit.receipt.review || {}));
        rows.push({ label, value: lines.join('\n') });
    }
    const github = facts.github || {};
    rows.push({ label: 'GitHub checks', value: `${github.status || 'unknown'}${github.reason ? `; ${github.reason}` : ''}.` });
    for (const run of github.runs || []) {
        const covers = run.covered_shas?.length
            ? `\nThis run also covers: ${run.covered_shas.map((sha) => sha.slice(0, 8)).join(', ')}.` : '';
        rows.push({ label: `${String(run.sha).slice(0, 8)} post-merge`,
            value: `${run.conclusion || run.status || 'unknown'}${run.url ? ` ${run.url}` : ''}${covers}` });
    }
    if (github.missing_shas?.length) {
        rows.push({ label: 'No matching run observed', value: github.missing_shas.map((sha) => sha.slice(0, 8)).join(', ') });
    }
    return rows;
}

// The retained attempt's before/after samples as labelled rows of its Last attempt (owner 25.10: A). Only a number
// is a measurement: a zero says so, never an absent value, and a value the sample did not record leaves its row out
// (the raw record under the rows keeps its null). Sizes and durations read as the other Dashboard tabs write them
// (storage_format.js). Peak RSS keeps its word; lifetime CPU time and one interval's share are separate rows; the
// dispatch time runs from acceptance to the first model call, not to an answer. Each side is one recorded sample:
// history, never a verdict about the update.
const measured = (value, format) => (typeof value === 'number' ? format(value) : '');
const overrunWords = (overruns) => (overruns && typeof overruns === 'object' ? Object.entries(overruns)
    .map(([duty, count]) => `${duty.replace(/[-_]/g, ' ')} ×${count}`).join(', ') || 'none' : '');

function utcOffset(date) {
    const minutes = -date.getTimezoneOffset();
    const whole = Math.abs(minutes);
    const rest = whole % 60;
    return `UTC${minutes < 0 ? '-' : '+'}${Math.floor(whole / 60)}${rest ? `:${String(rest).padStart(2, '0')}` : ''}`;
}

function sampleRows(sample) {
    if (!sample) return [{ label: 'Sample', value: 'unobserved' }];
    const direct = sample.direct_message_first_dispatch;
    const when = attemptTime(sample.observed_at);
    const percent = measured(sample.cpu_percent, (value) => `${value.toFixed(1)} %`);
    const interval = measured(sample.cpu_interval_seconds, durationLabel);
    return [
        { label: 'Source', value: sample.source_sha || 'unknown source' },
        // The local minute to read; the exact recorded stamp is the raw record's.
        { label: 'Observed', value: when ? `${when} ${utcOffset(new Date(sample.observed_at))}` : sample.observed_at || 'unknown time' },
        { label: `RSS${sample.rss_measure === 'peak' ? ' peak' : ''}`, value: measured(sample.rss_bytes, formatBytes) },
        { label: 'CPU time, process lifetime', value: measured(sample.cpu_seconds, durationLabel) },
        // One measured half of the share still says which half is missing.
        { label: 'CPU share, sample interval',
            value: percent || interval ? `${percent || 'unobserved'} over ${interval || 'unobserved'}` : '' },
        { label: 'Duty overruns', value: overrunWords(sample.duty_overruns) },
        { label: 'Direct message to first model call', value: typeof direct?.seconds === 'number'
            ? `${durationLabel(direct.seconds)} from acceptance${direct.task_id ? ` · task ${direct.task_id}` : ''}` : '' },
    ].filter((row) => row.value);
}

/** `after_update` as rows: each side's heading (`head`), then its `{label, value}` rows; [] without facts. */
export function afterUpdateRows(facts) {
    if (!facts) return [];
    return [['before', 'Before the update'], ['after', 'After the update']].flatMap(([side, name]) => {
        const sample = facts[side] || null;
        return [{ head: sample ? `${name} (recorded sample, server process)` : name }, ...sampleRows(sample)];
    });
}

// The executor's real stages (no percentages or timers): the panel headline, and the pill's shorter word.
export const UPDATE_STAGE_LABELS = {
    draining_writers: 'Finishing current work…',
    stopping_workers: 'Stopping worker processes…',
    stopping_services: 'Stopping services…',
    preparing: 'Preparing the update…',
    applying: 'Applying the update…',
    checking: 'Checking the update and dependencies…',
    rolling_back: 'Restoring the previous version…',
    restart_requested: 'Restart requested…',
};

const PILL_STAGES = {
    draining_writers: 'Finishing work…',
    stopping_workers: 'Stopping work…',
    stopping_services: 'Stopping services…',
    preparing: 'Preparing update…',
    applying: 'Applying update…',
    checking: 'Checking update…',
    rolling_back: 'Restoring version…',
    restart_requested: 'Restart requested…',
};

// Only the server concludes an attempt; a reconnect, a version or a SHA never does. A word for the pill is the
// outcome's claim to wear it: a confirmed success has none (owner 1B), so its record stays in the panel alone.
const OUTCOMES = {
    applied: { tone: 'ok', pill: '', headline: 'Update completed.' },
    not_applied: { tone: 'error', pill: 'Update not applied', headline: 'No update was applied.' },
    rolled_back: { tone: 'error', pill: 'Previous version restored',
        headline: 'The update failed; the previous version was restored.' },
    failed: { tone: 'error', pill: 'Update failed', headline: 'The update did not complete.' },
    unknown: { tone: 'neutral', pill: 'Update outcome unconfirmed', headline: 'The last update outcome is unconfirmed.' },
};

const UNKNOWN_REASONS = {
    startup_unconfirmed: 'The restarted server did not confirm its startup checks.',
    running_source_mismatch: 'The running version is not the commit this update landed.',
    record_unreadable: 'The saved record of the last attempt could not be read.',
    state_unreadable: 'The saved state could not be read, so the last attempt is unknown.',
};

// A boot recovery that reported its own rollback unfinished. The retry comes from this reason, never from the branch's
// text, and only while that attempt's transaction is still open; it is no Restart or Restore control.
export const RECOVERY_RETRY = 'Quit and reopen the desktop app, or restart the server process for a web deployment, '
    + 'to retry the recovery.';
const FAILED_REASONS = {
    rollback_incomplete: { pill: 'Update recovery incomplete',
        headline: 'The update failed, and restoring the previous version did not finish.' },
};

// supervisor/update_merge._ASSISTED_PHASES (pinned by a source test): only these are a reviewed resolution at work.
const ASSISTED_PHASES = new Set(['materializing_assisted', 'assisted_resolution', 'committing_assisted']);
// supervisor/update_merge.MARKER_CLEANUP_RETRY_PHASE: the transaction recorded its completion (a finished rollback
// awaiting its restart included) and only cleanup is left, so an earlier unfinished report is history.
const COMPLETION_RECORDED_PHASE = 'marker_cleanup_retry';

/** Whether an update transaction phase is a reviewed resolution at work: the one reading for panel and record. */
export function assistedUpdatePhase(phase) {
    return ASSISTED_PHASES.has(String(phase || ''));
}

/** The still-active transaction this attempt is bound to (its `tx_attempt_id`), else null. */
export function boundUpdateTx(progress, tx) {
    return tx?.active && tx.attempt_id && tx.attempt_id === progress?.tx_attempt_id ? tx : null;
}

/** The attempt's one concluded reading: its outcome; a record without the field (an older shape) that failed is a failure. */
export function attemptOutcome(progress) {
    return progress?.outcome || (!progress?.active && progress?.result === 'failed' ? 'failed' : '');
}

function recoveryReport(progress) {
    return attemptOutcome(progress) === 'failed' ? FAILED_REASONS[progress.reason] || null : null;
}

/** A bound attempt whose boot recovery reported its rollback unfinished, still the transaction's state: its words, else null. */
export function unfinishedRecovery(progress, tx) {
    const known = recoveryReport(progress);
    const bound = boundUpdateTx(progress, tx);
    return known && bound && bound.phase !== COMPLETION_RECORDED_PHASE ? known : null;
}

const REFUSALS = {
    release_moved: 'The update changed after it was prepared.',
    late_local_changes: 'Local files changed while the update was starting.',
    replace_not_applied: 'Git refused the first checkout step.',
};

function attemptText(progress, recovery = false) {
    const blockers = Array.isArray(progress.blockers) ? progress.blockers.map(String) : [];
    // Only the drain refusal names active writers, and it returns before any handoff or teardown.
    const busy = blockers.length > 0 && blockers.every((item) => item.startsWith('active:'));
    const parts = [];
    if (progress.outcome === 'not_applied') {
        parts.push(busy ? `Work was still finishing: ${blockers.map((item) => item.slice('active:'.length)).join(', ')}.`
            : REFUSALS[progress.reason] || progress.error);
        // Owner 4A: only when that attempt's own executor proved it reached no teardown.
        if (busy && progress.work_stop === 'none' && progress.concluded_by === 'executor') {
            parts.push('This attempt stopped no running work.');
        }
        if (busy) parts.push('Try again when it finishes.');
    } else if (progress.outcome === 'unknown') {
        parts.push(UNKNOWN_REASONS[progress.reason]
            || 'No completion record arrived; the running version alone does not establish the outcome.');
    } else if (progress.outcome === 'applied') {
        parts.push('The restarted server confirmed this update.');
    } else if (!recovery) {
        parts.push(progress.error);
    }
    if (progress.work_stop === 'invoked') parts.push('Running work may have been interrupted.');
    if (progress.restart_required && progress.current_generation !== false) {
        parts.push('Ouroboros needs a restart before updates can continue.');
    }
    if (progress.keep_branch) parts.push(`Your local commits are kept on branch ${progress.keep_branch}.`);
    if (progress.stash_note && !recovery) parts.push(progress.stash_note);
    return parts.filter(Boolean).join(' ');
}

function attemptDetails(progress, outcome) {
    const rows = [];
    const add = (label, value) => {
        if (value) rows.push({ label, value: String(value) });
    };
    add('Base', String(progress.base_sha || '').slice(0, 12));
    add('Target', String(progress.target_sha || '').slice(0, 12));
    // The executor's stage in its own words (the progress label without its ellipsis); an unknown stage keeps its name.
    add('Stage', progress.stage ? (UPDATE_STAGE_LABELS[progress.stage] || progress.stage).replace(/…$/, '') : (outcome ? 'not started' : ''));
    add('Outcome', outcome);
    add('Reason', progress.reason);
    add('Blocked by', (Array.isArray(progress.blockers) ? progress.blockers : []).join(', '));
    add('Error', progress.error);
    if (outcome === 'failed' && FAILED_REASONS[progress.reason]) add('Recovery note', progress.stash_note);
    add('Saved local work', progress.keep_branch);
    add('Concluded by', { executor: 'this update', boot: 'the restarted server' }[progress.concluded_by]);
    return rows;
}

/**
 * One reading of the retained attempt for the pill and the panel: the stage while it runs, then the
 * server's conclusion or its named uncertainty. `tx` is the same status read's `update_tx`. `null` when
 * there is no attempt to show.
 */
export function updateAttemptView(progress, tx = null, { showRecoveryInstruction = true } = {}) {
    if (!progress || typeof progress !== 'object' || !progress.operation_id) return null;
    const key = attemptOutcome(progress);
    const base = {
        id: String(progress.operation_id),
        when: String(progress.outcome_at || progress.stage_started_at || progress.started_at || ''),
        details: attemptDetails(progress, key),
    };
    const retention = progress.retention_unconfirmed
        ? 'This observation could not be saved and may be unavailable after restart.' : '';
    const outcome = OUTCOMES[key];
    if (outcome && !progress.active) {
        const recovery = recoveryReport(progress);
        // A later recorded completion of the bound transaction outdates "did not finish"; its record stays failed.
        const named = recovery && boundUpdateTx(progress, tx)?.phase !== COMPLETION_RECORDED_PHASE ? recovery : outcome;
        // Exact diagnostics stay in Details, including an older note's retry wording. The current action is said once:
        // here, unless the panel's own verdict already gives a recovery instruction.
        const retry = showRecoveryInstruction && unfinishedRecovery(progress, tx) ? RECOVERY_RETRY : '';
        return { ...base, terminal: true, tone: outcome.tone, pill: named.pill, headline: named.headline,
            restoreTarget: key === 'applied' ? String(progress.base_sha || '') : '',
            text: [attemptText({ ...progress, outcome: key }, Boolean(recovery)), retry, retention].filter(Boolean).join(' ') };
    }
    // Not concluded: this server's stage, a restart it needs, a reviewed resolution, or a later server's checks.
    const stage = String(progress.stage || '');
    const here = progress.active || progress.current_generation !== false;
    const resolving = assistedUpdatePhase(boundUpdateTx(progress, tx)?.phase);
    const [pill, headline] = progress.result === 'assisted_started'
        // A resolver start whose transaction left the assisted phases (cleanup retry, stash recovery, a rollback or
        // gate block) is not a resolution at work, and nothing here says how it ended.
        ? (resolving ? ['Resolving update…', 'A reviewed resolution owns this update.']
            : ['Update not concluded', 'This update has not concluded yet.'])
        : progress.result === 'restart_required' && here
            ? ['Restart needed', 'The update needs a restart to finish.']
            : here
                ? [PILL_STAGES[stage] || 'Updating…', UPDATE_STAGE_LABELS[stage] || 'Updating…']
                : ['Confirming update…', 'Waiting for the restarted server to confirm the update…'];
    return { ...base, terminal: false, tone: 'neutral', pill, headline, text: retention };
}

/** A request that never reached Ouroboros in owner words (errorWords); the server's own words are kept as they are. */
export const failureWords = (err) => (err instanceof TypeError && err.response === undefined
    ? errorWords(err).text : String(err?.message || err));

export function attemptTime(value) {
    const date = new Date(value);
    if (!value || Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('en-GB', {
        day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false,
    }).format(date);
}

// One bounded per-browser preference: the one terminal attempt hidden from the pill. It is never sent anywhere
// and changes neither the server's record nor the panel; another browser shows the notice again.
const DISMISSED_KEY = 'ouroboros.update-attempt-dismissed';

// The settings row that chooses the update channel (settings_rows.js): its accepted save event is what marks a
// new selection, wherever the save happened (Settings page or the Dashboard's own card).
const CHANNEL_ROW = 'OUROBOROS_UPDATE_CHANNEL';

export function dismissedUpdateAttempt() {
    try {
        return globalThis.localStorage?.getItem(DISMISSED_KEY) || '';
    } catch {
        return '';
    }
}

export function dismissUpdateAttempt(id) {
    try {
        globalThis.localStorage?.setItem(DISMISSED_KEY, String(id || ''));
    } catch {
        // Storage refused: the pill keeps showing the result, which is the safe side.
    }
    globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:update-attempt-dismissed'));
}

/** The one reading of a check that did not complete, shared by every surface (the panel's own record and this
 * row's words): an answer that arrived unreadable was cut in transport — said so, never as silence or a
 * placeholder — a request that never reached Ouroboros keeps its owner words, and a readable answer its own
 * warnings. Each surface keeps its own presentation of the reason; this is the interpretation alone. */
export function checkFailureReason(answer, error = null) {
    if (error?.unreadable) return 'the response to the update check could not be read';
    if (error) return failureWords(error);
    return (Array.isArray(answer?.warnings) ? answer.warnings : []).join(', ') || 'the check did not complete';
}

// `checked_at` is minted once per completed check (the update contract), so on one channel a strictly newer stamp
// is a genuinely new observation while an equal one is the same check; words that do not parse name nothing newer.
function newerStamp(at, than) {
    const left = String(at || '');
    const right = String(than || '');
    if (!left || !right || left === right) return false;
    const a = Date.parse(left);
    const b = Date.parse(right);
    return Number.isNaN(a) || Number.isNaN(b) ? left > right : a > b;
}

// One attempt record's own moment: its start, its latest stage or its conclusion. Each stamp it carries is the
// server's ISO-8601 UTC, so the latest of them is the record's own as-of moment.
function attemptMoment(progress) {
    const record = progress && typeof progress === 'object' ? progress : {};
    return [record.started_at, record.stage_started_at, record.outcome_at]
        .map((stamp) => String(stamp || '')).filter(Boolean).sort().pop() || '';
}

/** Whether an arriving attempt record is the newer observation of the running attempt: another operation that
 * started later, or this operation at its own later stage or conclusion. A record the row cannot place in that
 * order (no identity, no comparable stamp) is not newer, so the attempt the row already read keeps the row. */
function newerAttempt(incoming, held) {
    const arriving = incoming && typeof incoming === 'object' ? incoming : {};
    const row = held && typeof held === 'object' ? held : {};
    const arrivingId = String(arriving.operation_id || '');
    if (!arrivingId) return false;
    const rowId = String(row.operation_id || '');
    if (!rowId) return true;
    return arrivingId === rowId
        ? newerStamp(attemptMoment(arriving), attemptMoment(row))
        : newerStamp(arriving.started_at, row.started_at);
}

/** What a finished check says in the version row. Only this check's own positive result counts: `current` (Up to
 * date) or `available` needs `check_ok: true` and a boolean `available` — a truthy flag, a tx-only answer and one
 * that carries no check result at all (a readable status without `check_ok`) are `failed`, with the reason in
 * words and the last good check the failure leaves standing (a later one ends it); the version stays the retry. An
 * install that offers no in-app update (unmanaged, its organization's) is `unoffered`: no failure, only the plain
 * version. */
export function checkOutcome(status, error = null) {
    if (!error && status && !updatesOffered(status)) return { kind: 'unoffered' };
    if (error || status?.check_ok !== true || typeof status?.available !== 'boolean') {
        return { kind: 'failed', reason: checkFailureReason(status, error), lastGood: String(status?.checked_at || '') };
    }
    return { kind: status.available ? 'available' : 'current' };
}

const checkFailure = (reason) => fmt('Could not check the official update channel: {reason}', { reason });

/** Whether this install offers the in-app check and update at all: an unmanaged checkout and a deployment its
 * organization updates show the version as plain text and only an attempt's or a restart's words (updates.js). */
export function updatesOffered(status) {
    return Boolean(status) && status.managed !== false && !status.managed_by_deployment
        && !(Array.isArray(status.warnings) && status.warnings.includes('managed_updates_unavailable'));
}

// Why such an install offers none, in the Updates panel's own words.
const unofferedWords = (status) => (status?.managed_by_deployment ? 'Managed by your organization'
    : 'Managed updates are unavailable for this checkout.');

// How long Up to date stands in the capsule's place before the row is at rest again.
export const UP_TO_DATE_MS = 4000;

export function initUpdateStatus({ showPage, openDashboardTab, ws, readStatus = () => apiClient.updateStatus(),
    checkNow = (selection) => apiClient.updateCheck(selection), upToDateMs = UP_TO_DATE_MS } = {}) {
    let latest = null;
    // The row's one ordered status, read on two axes that carry their own proof of being newer: the channel's check,
    // stamped once by the server (`checked_at`) so a strictly newer stamp is a newer observation and an equal one the
    // same check, and the running attempt, ordered by its own record (a later start for another operation, a later
    // stage or conclusion for this one). A passive read is the row's current truth and applies only while no check
    // has answered since its read began, and a read that names a different channel is that channel's own first
    // reading: nothing of the one before it survives. A press's answer takes an axis only while the row still holds
    // the selection that press named, while the answer itself names no other channel, and only where it is the newer
    // observation — so a reply delayed in transit, an earlier channel's answer or a partial answer never paints over
    // facts the row already read.
    let checkAnswered = 0;
    // What only a completed check establishes: a failed or partial answer leaves the last good one standing and never
    // creates one of them from its own incomplete reading. The tip facts are the check's own record, identified by
    // its stamp — an equal stamp is the same check and an older one an older observation; the checkout observations
    // are the row's freshest reading of the checkout, so an answer's own older sample never overwrites that reading
    // (a consumed offer, dirty state) at an equal stamp.
    const TIP_FACTS = ['available', 'safe_to_apply', 'behind', 'ahead', 'checked_at', 'checked_target_sha',
        'from_cache', 'latest_sha', 'latest_short_sha', 'latest_version', 'latest_message'];
    const CHECK_FACTS = ['available', 'safe_to_apply', 'behind', 'ahead', 'checked_at', 'checked_target_sha',
        'from_cache', 'latest_sha', 'latest_short_sha', 'latest_version', 'latest_message',
        'local_changes', 'local_changes_base', 'local_changes_reason'];
    const CHECKOUT_FACTS = ['dirty', 'local_changes', 'local_changes_base', 'local_changes_reason'];
    // The attempt axis: one live reading whose halves always travel together.
    const ATTEMPT_FACTS = ['update_progress', 'update_tx'];

    // A completed check's answer carries checkout observations sampled when the check ran. A passive read is a later
    // sample at the same stamp; a check's answer at that stamp is the earlier one. So held facts win where the held
    // stamp is strictly newer and, for a check's answer, also at an equal stamp; a read's facts win at an equal stamp,
    // and a strictly newer stamp's own facts win either way. A held reading with no stamp cannot be ordered above a
    // stamped answer — the first stamped Check supplies its own coherent checkout facts — and an unstamped answer
    // never outranks the row's own unstamped reading either.
    function checkoutObservations(status, passive = false) {
        const held = latest || {};
        const heldWins = passive ? newerStamp(held.checked_at, status.checked_at)
            : (held.checked_at ? !newerStamp(status.checked_at, held.checked_at) : !status.checked_at);
        const kept = {};
        for (const key of CHECKOUT_FACTS) {
            const value = heldWins && key in held ? held[key] : status[key];
            if (value !== undefined) kept[key] = value;
        }
        return kept;
    }

    let ownedNotice = null;
    let closeButton = null;
    let hideable = '';  // the terminal attempt the close button hides now; '' while it is not offered
    // The capsule and its close button: one notice whose end sits where the notice does (a hide on another page is
    // no ending). The close is drawn inside the capsule's end, so no other control moves when it comes and goes.
    function ensurePill() {
        let pill = document.getElementById('update-pill');
        if (!pill) {
            ownedNotice = document.createElement('span');
            ownedNotice.className = 'update-notice';
            ownedNotice.hidden = true;
            pill = document.createElement('button');
            pill.id = 'update-pill';
            pill.type = 'button';
            pill.className = 'btn btn-default btn-sm update-pill';
            pill.hidden = true;
            pill.addEventListener('click', () => {
                if (openDashboardTab) openDashboardTab('updates');
                else showPage?.('dashboard');
            });
            closeButton = document.createElement('button');
            closeButton.type = 'button';
            closeButton.className = 'update-notice-close ui-hit-area';
            closeButton.hidden = true;
            closeButton.setAttribute('aria-label', 'Hide this notice');
            closeButton.title = 'Hide this notice. The update record stays in Dashboard → Updates.';
            closeButton.innerHTML = SHELL_ICONS.close;
            closeButton.addEventListener('click', () => {
                if (!hideable) return;
                const held = document.activeElement === closeButton;
                dismissUpdateAttempt(hideable);  // the event repaints this notice at once
                // The control that ended the notice is gone: a keyboard reader lands on what is left of it — the
                // capsule it unmasked, the row's own check when it unmasked none — else on the sidebar's last
                // row, never on the page.
                if (held) {
                    const rows = document.body.querySelectorAll?.('#primary-sidebar [data-nav-page]') || [];
                    (pill.hidden ? (checker && !checker.hidden ? checker : [...rows].pop()) : pill)?.focus?.({ preventScroll: true });
                }
            });
            ownedNotice.append(pill, closeButton);
            // The row the page gives the version (pages.js renderNavFoot); an older page without it (v7.7.0's
            // sidebar) keeps the capsule beside the version itself, where it always stood.
            if (row) row.appendChild(ownedNotice);
            else if (version?.parentElement) version.parentElement.insertBefore(ownedNotice, version.nextSibling);
            else document.body.appendChild(ownedNotice);
        }
        return pill;
    }

    // The version row (pages.js renderNavFoot): the version in its check, the check's answer and its words for readers.
    const version = document.getElementById('nav-version');
    const row = version?.closest?.('.sidebar-foot-meta') || null;
    let checker = null;
    let glyph = null;
    let note = null;
    let said = null;
    // The channel epoch of this row's check in flight, or null. A check belongs to the selection the press named, so
    // an older epoch's unanswered press neither blocks the selection the row holds now nor clears a newer press's
    // busy state when it finally answers.
    let busyEpoch = null;
    let outcome = null;  // the last check's answer while it stands: checkOutcome's
    let outcomeTimer = 0;
    // The epoch of the row's channel selection: bumped by an accepted channel save, so a press's answer (a response
    // or a rejection) for the selection the row has left is not the row's on either axis.
    let channelEpoch = 0;
    // An accepted channel save names the row, never the value it saved, so which selection this row holds is unknown
    // until its own read names the channel again: while it is, the version is the plain text it was before the first
    // read — never a control whose press would key a selection the row cannot vouch for.
    let channelUnknown = false;
    function mountCheck() {
        if (!row) return;
        checker = Object.assign(document.createElement('button'), { type: 'button', id: 'nav-version-check',
            className: 'nav-version-check ui-hit-area', title: 'Check for updates' });
        // Its name is the action and the version (the version's own node, so a newer version renames it).
        checker.setAttribute('aria-labelledby', 'nav-version-check-word nav-version');
        checker.setAttribute('aria-describedby', 'nav-check-said');
        const word = Object.assign(document.createElement('span'), { id: 'nav-version-check-word', hidden: true });
        word.textContent = 'Check for updates';
        glyph = Object.assign(document.createElement('span'), { className: 'nav-version-glyph' });
        glyph.innerHTML = fluentIcon('arrow-clockwise', 14, 'regular');
        row.insertBefore(checker, version);
        checker.append(word, version, glyph);
        note = Object.assign(document.createElement('span'), { className: 'nav-check-note ui-status', hidden: true });
        said = Object.assign(document.createElement('span'), { id: 'nav-check-said', className: 'nav-check-said' });
        said.setAttribute('role', 'status');
        row.append(note, said);
        checker.addEventListener('click', press);
    }

    let saidWords = '';
    function say(words) {
        // Deduped by the authored source alone, never by the live region's text: the translation overlay rewrites
        // that text in place (i18n.js observes characterData), so comparing against it would re-announce the same
        // words on every visit and fight the overlay's own translation. A genuinely new authored reading writes.
        if (!said || saidWords === words) return;
        saidWords = words;
        said.textContent = words;
    }

    // One action per row: while a capsule stands the version is its plain text and the answer gives way to it; so it
    // is until a status read says this install offers the check at all, and while the row's channel is unknown after
    // an accepted save — no press names a selection the row cannot vouch for.
    function renderCheck() {
        if (!checker) return;
        const capsule = Boolean(ownedNotice && !ownedNotice.hidden);
        const plain = capsule || channelUnknown || !updatesOffered(latest);
        if (plain && version.parentElement === checker) {
            const held = document.activeElement === checker;
            row.insertBefore(version, checker);
            // The control that held focus is gone: a keyboard reader lands on the capsule, else on the last row —
            // and only onto something visible, only because this control held the focus.
            const rows = document.body.querySelectorAll?.('#primary-sidebar [data-nav-page]') || [];
            const landing = capsule ? ensurePill() : [...rows].pop();
            if (held && landing && !landing.hidden) landing.focus?.({ preventScroll: true });
        } else if (!plain && version.parentElement !== checker) {
            checker.insertBefore(version, glyph);
        }
        checker.hidden = plain;
        const busy = busyEpoch === channelEpoch;
        if (busy) checker.setAttribute('aria-busy', 'true');
        else checker.removeAttribute('aria-busy');
        const shown = !plain && !busy && ['current', 'failed'].includes(outcome?.kind) ? outcome : null;
        note.hidden = !shown;
        if (!shown) return;
        const failed = shown.kind === 'failed';
        if (failed) note.dataset.tone = 'error';
        else delete note.dataset.tone;
        note.textContent = failed ? 'Check failed' : 'Up to date';
        if (!failed) note.insertAdjacentHTML?.('afterbegin', fluentIcon('checkmark', 14, 'regular'));
        note.title = failed ? checkFailure(shown.reason) : '';
    }

    // A press runs the one check and nothing else (never a description, a preflight, an apply or a restart); a press
    // while its own selection's check is in flight starts nothing (api_client coalesces the same selection, and this
    // control waits for the check it started), nor does one while the version is plain text. The press names the
    // selection it believes current, so a coalesced answer is never another selection's.
    function applyCheckAnswer(status, error) {
        if (!status) return;  // the request never reached Ouroboros: every fact the row holds still stands
        // This answer owns the row's check axis: a passive read that began before it never paints over it.
        checkAnswered += 1;
        // The attempt and its transaction travel together, and an answer that cannot show itself newer than the
        // attempt the row read leaves both standing: a reply delayed in transit must not erase a stage, a reviewed
        // resolution or a needed restart that a later read already read.
        const newer = newerAttempt(status.update_progress, latest?.update_progress);
        const carried = {};
        for (const key of ATTEMPT_FACTS) {
            const value = newer ? status[key] : latest?.[key];
            if (value !== undefined) carried[key] = value;
        }
        // Only a completed check of the row's channel establishes its check facts, and only where the row does not
        // already hold a strictly newer check (another surface's, read here meanwhile): the newer observation speaks
        // for the channel, whatever order the answers arrive in.
        if (!error && status.check_ok === true && typeof status.available === 'boolean'
            && !newerStamp(latest?.checked_at, status.checked_at)) {
            const held = latest || {};
            const same = Boolean(held.checked_at) && held.checked_at === status.checked_at;
            const heldTip = {};
            if (same) for (const key of TIP_FACTS) if (key in held) heldTip[key] = held[key];
            latest = { ...status, ...heldTip, ...checkoutObservations(status), ...carried };
            return;
        }
        // A partial, failed or outrun answer keeps the last good check's facts standing and takes only the answer's
        // own live evidence: its attempt, its needed restart, the running version — reported independently of the
        // check, and never creating one of its facts from an incomplete reading. Its own checkout sample is an
        // observation like any other, reconciled on the stamp axis after the kept facts: a delayed failed POST never
        // reverts the checkout a later passive read sampled, and a sample the answer minted newer still wins.
        const answer = { ...status };
        for (const key of CHECK_FACTS) delete answer[key];
        const keep = {};
        for (const key of CHECK_FACTS) if (key in (latest || {})) keep[key] = latest[key];
        latest = { ...answer, ...keep, ...checkoutObservations(status), ...carried };
    }

    // A passive read's channel change retires the row's outcome for every kind: the words belong to the channel the
    // row has left. A failure also ends where a newer completed check succeeded — here or on another surface — or at
    // the first confirmed stamp where no good one stood before (the passive route proves a completed check by its
    // cache-proven `checked_at` alone; `check_ok` is null on status); the same stamp is the same check and keeps it.
    function retireOutcome(status) {
        if (!outcome) return;
        const channel = String(status?.update_channel || '');
        if (channel && outcome.channel && channel !== outcome.channel) { outcome = null; return; }
        if (outcome.kind !== 'failed') return;
        const firstEver = !outcome.lastGood && Boolean(status?.checked_at);
        if (newerStamp(status?.checked_at, outcome.lastGood) || firstEver) outcome = null;
    }

    async function press() {
        if (channelUnknown || checker.hidden || !updatesOffered(latest)) return;
        const epoch = channelEpoch;
        // One check in flight per selection: the row's current one. A press whose selection the row has already left
        // (an accepted save moved it on) no longer holds this row busy, so the selection it holds now is pressable.
        if (busyEpoch === epoch) return;
        const selection = String(latest?.update_channel || '');
        busyEpoch = epoch;
        outcome = null;
        clearTimeout(outcomeTimer);
        say('');
        renderCheck();
        let status = null;
        let error = null;
        try {
            status = await checkNow(selection);
        } catch (err) {
            error = err;
        }
        if (busyEpoch === epoch) busyEpoch = null;
        if (disposed) return;
        // The row's channel moved on while this press was in flight (an accepted save, and its own read naming the new
        // selection): this answer speaks of the selection the row has left, so it writes no outcome and clears no
        // busy state — the newer press's own answer owns both, and the save's read already observed the new state.
        if (epoch !== channelEpoch) return;
        const lastGood = String(latest?.checked_at || '');
        // The answer belongs to the selection its press named, on the epoch that named it: while the row still
        // holds both, it is this row's answer; once the row's channel has moved (the setting changed elsewhere and
        // a read showed it, or the row learned another one), the reply speaks of a channel the row has left, and it
        // contributes neither facts nor words — a stale rejection no less than a stale response: its own result is
        // not attributed to the channel the row holds now. A resolved POST that itself names another channel says
        // the check ran after a parallel change the row has not read: it is discarded the same way, and one passive
        // re-read — never a second POST — observes the state that change left behind.
        const answeredChannel = status && typeof status === 'object' ? String(status.update_channel || '') : '';
        const foreign = Boolean(selection) && Boolean(answeredChannel) && answeredChannel !== selection;
        const holdsSelection = epoch === channelEpoch && String(latest?.update_channel || '') === selection;
        const ours = holdsSelection && !foreign;
        if (ours) applyCheckAnswer(status, error);
        outcome = ours ? checkOutcome(status, error) : null;
        if (outcome) {
            outcome.channel = String(status?.update_channel || selection || '');
            if (outcome.kind === 'failed' && !outcome.lastGood) outcome.lastGood = lastGood;
            if (outcome.kind === 'current') {
                // The timer belongs to this result alone: a later answer replaces it, and this timer never clears
                // another's.
                const mine = outcome;
                outcomeTimer = setTimeout(() => {
                    if (outcome !== mine) return;
                    outcome = null;
                    renderPill();
                }, upToDateMs);
            }
        }
        if (foreign && holdsSelection) refresh();
        renderPill();
    }

    function renderPill() {
        const status = latest;
        const pill = ensurePill();
        const pillHeld = document.activeElement === pill;
        const closeHeld = Boolean(closeButton) && !closeButton.hidden && document.activeElement === closeButton;
        const attempt = updateAttemptView(status?.update_progress, status?.update_tx);
        // A stage always shows; a concluded attempt shows only if its outcome has a word for the pill (a success has
        // none) and this browser has not hidden it. What is hidden is the notice: the record stays in the panel.
        const shown = Boolean(attempt?.pill) && !(attempt.terminal && attempt.id === dismissedUpdateAttempt());
        // Content or uncommitted work of the checkout's own is a caveat about an AVAILABLE update, not a
        // count of commits ahead; while an attempt shows, its stage or outcome is the pill's one fact.
        // An install that offers no in-app update (unmanaged, its organization's) points at none.
        const available = status?.available === true && updatesOffered(status);
        const local = Boolean(!shown && available && (status.dirty || status.local_changes !== 'absent'));
        pill.classList.toggle('has-local', local);
        // An update on offer is the one crimson capsule (the concept of 21.11); its caveat and an attempt keep theirs.
        const offer = !shown && available && !local;
        pill.classList.toggle('btn-primary', offer);
        pill.classList.toggle('btn-default', !offer);
        hideable = shown && attempt.terminal ? attempt.id : '';
        if (closeButton) closeButton.hidden = !hideable;
        pill.classList.toggle('has-close', Boolean(hideable));
        if (shown) {
            pill.textContent = attempt.pill;
            pill.title = [attempt.headline, attempt.text].filter(Boolean).join(' ');
            pill.dataset.tone = attempt.tone;
        } else {
            delete pill.dataset.tone;
            if (available) {
                pill.textContent = updatePillText(status);
                pill.title = updatePillDetail(status);
            }
        }
        pill.hidden = !shown && !available;
        if (ownedNotice) ownedNotice.hidden = pill.hidden;
        renderCheck();
        // The pill that held the focus is going away: a keyboard reader lands on the row's own visible check, else
        // the sidebar's last row — only because this element itself held the focus, and nowhere else.
        if (pillHeld && pill.hidden) {
            const rows = document.body.querySelectorAll?.('#primary-sidebar [data-nav-page]') || [];
            (checker && !checker.hidden ? checker : [...rows].pop())?.focus?.({ preventScroll: true });
        }
        if (closeHeld && closeButton.hidden) {
            // The × this reader held is gone — a new attempt took the notice over, or its record changed shape: land
            // on what is left of the row (the capsule where it stood, the row's own check), never on the page.
            const rows = document.body.querySelectorAll?.('#primary-sidebar [data-nav-page]') || [];
            (pill.hidden ? (checker && !checker.hidden ? checker : [...rows].pop()) : pill)?.focus?.({ preventScroll: true });
        }
        syncSaid();
    }

    // The row's live region says what the row now shows, never a staler copy of it (aria-describedby names it): a
    // capsule's words where one stands (an update on offer, an attempt or a needed restart takes the row), else
    // this check's own answer, an install's plain reason, or nothing while it checks.
    function syncSaid() {
        const pill = document.getElementById('update-pill');
        say(pill && !pill.hidden ? (pill.title || pill.textContent)
            : outcome?.kind === 'failed' ? checkFailure(outcome.reason)
            : outcome?.kind === 'current' ? 'Up to date'
            : latest && !updatesOffered(latest) ? unofferedWords(latest) : '');
    }

    // A passive read's answer owns the row's live facts (its attempt record, its checkout observations — it
    // fetched them now) but cannot revert a settled check: where the row holds the newer stamped tip the read's
    // older tip facts stay out; an answer that cannot show itself newer never replaces what the row read. A read
    // that names a different channel than the one the row's facts belong to is the new channel's own first
    // reading: none of the old channel's stamped tip or checkout observations survive into it.
    function mergeRead(status) {
        const held = latest || {};
        if (held.update_channel && status.update_channel && held.update_channel !== status.update_channel) {
            return { ...status };
        }
        const merged = { ...status, ...checkoutObservations(status, true) };
        for (const key of TIP_FACTS) {
            if (key in held && (newerStamp(held.checked_at, status.checked_at) || !(key in status))) {
                merged[key] = held[key];
            }
        }
        if (!('update_channel' in status) && 'update_channel' in held) merged.update_channel = held.update_channel;
        return merged;
    }

    // The panel's coalescing (updates.js loadStatus): one read in flight; a newer notice owns the next read.
    let reading = null;
    let readAgain = false;
    let disposed = false;
    function refresh() {
        readAgain = true;
        if (reading) return reading;
        reading = (async () => {
            while (readAgain && !disposed) {
                readAgain = false;
                // A check that answers while this read runs owns the row's check axis; this answer cannot be ordered
                // against it (its snapshot may be older or newer — a restart, a new channel), so instead of being
                // discarded or painting over the settled check, one coalesced re-read observes the state the check
                // left behind. A channel saved while the read runs leaves the answer speaking for a selection the
                // row has left: the same one coalesced re-read serves it. No polling: the re-read runs once per
                // overlap, only when a check genuinely answered or the selection genuinely changed.
                const answered = checkAnswered;
                const epoch = channelEpoch;
                const status = await safe(readStatus);
                if (readAgain || disposed) continue; // superseded: never render an older answer
                if (answered !== checkAnswered || epoch !== channelEpoch) { readAgain = true; continue; }
                if (status) {
                    latest = mergeRead(status); // a failed read hides nothing already known
                    channelUnknown = false; // any readable status names the channel this install holds now
                    retireOutcome(latest);
                }
                renderPill();
            }
        })().finally(() => { reading = null; });
        return reading;
    }

    const disposers = [];
    const listen = (event, handler) => {
        const dispose = ws?.on?.(event, handler);
        if (typeof dispose === 'function') disposers.push(dispose);
    };
    listen('update_status_ready', refresh);
    listen('update_progress_changed', refresh);
    listen('open', (event = {}) => {
        if (event.previouslyConnected === true) refresh();
    });
    const onDismissed = () => renderPill();
    // A channel saved while an old offer stands: the accepted save is settings_controls.js's `ouro:settings-updated`
    // naming this row (both entries of the setting — the Settings page and the Dashboard's own card — emit it). The
    // old selection's check facts, offers and answers — visible and ARIA alike — are invalidated for every outcome
    // kind; the new selection owns nothing until its own check runs, and answers still in flight for the old
    // selection (a response or a rejection) stop being the row's through the epoch. The save names the row, never
    // the accepted value: which selection this row holds is unknown until its own read names it, so a stale channel
    // neither keys the next press nor blocks the one after the read.
    const onSettingsUpdated = (event) => {
        const detail = event?.detail || {};
        if (detail.source !== 'settings-row') return;
        const keys = Array.isArray(detail.keys) ? detail.keys : [];
        if (detail.row !== CHANNEL_ROW && !keys.includes(CHANNEL_ROW)) return;
        channelEpoch += 1;
        const carried = { ...(latest || {}) };
        for (const key of CHECK_FACTS) delete carried[key];
        delete carried.update_channel;  // the accepted selection is this row's only once its own read names it
        latest = carried;
        channelUnknown = true;
        outcome = null;
        clearTimeout(outcomeTimer);
        renderPill();
        refresh();
    };
    const onPageShown = (event) => {
        if (event?.detail?.page === 'chat') refresh();
    };
    window.addEventListener('ouro:update-attempt-dismissed', onDismissed);
    window.addEventListener('ouro:settings-updated', onSettingsUpdated);
    window.addEventListener('ouro:page-shown', onPageShown);
    mountCheck();
    // One synchronous paint of the row's starting state: mountCheck has moved the version into the control, and
    // until a status read says this install offers the check, the version is plain text — never a visible control
    // whose press would do nothing (the first read may take seconds, or never answer). No capsule is created here:
    // nothing has been read yet.
    renderCheck();
    refresh();

    return {
        refresh,
        dispose() {
            disposed = true;
            disposers.splice(0).forEach((dispose) => dispose());
            window.removeEventListener('ouro:update-attempt-dismissed', onDismissed);
            window.removeEventListener('ouro:settings-updated', onSettingsUpdated);
            window.removeEventListener('ouro:page-shown', onPageShown);
            clearTimeout(outcomeTimer);
            ownedNotice?.remove();
            if (checker) {  // the version stays, as plain text where the page put it
                row.insertBefore(version, checker);
                [checker, note, said].forEach((node) => node.remove());
            }
        },
    };
}
