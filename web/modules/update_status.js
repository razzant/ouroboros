// Main-screen Update affordance (P2): one compact pill. While an update attempt runs it names the
// real stage; afterwards a failed or unconfirmed outcome stays on it until the owner hides it with the
// close button beside it (owner 3A, #1818); a confirmed success never wears it (owner 1B): that record
// lives in Dashboard -> Updates alone and cannot mask a newer version. Otherwise the pill appears when a
// managed update is available (status is populated by the boot-time check-on-restart). The pill is a
// pointer, not a second apply surface: clicking it opens Dashboard -> Updates, where the ONE apply flow
// (with its verified preflight and typed outcomes) lives. The former pill-local staged dialog was removed
// (owner decision 2026-08-31) so two hand-rolled apply controllers cannot drift apart again.

import { apiClient, updateStrategyForPlan } from './api_client.js';
import { SHELL_ICONS } from './page_icons.js';

// Fail-soft wrapper around the api_client update helpers (the pill must never throw the app).
async function safe(fn) {
    try {
        return await fn();
    } catch {
        return null;
    }
}

// The capsule says one thing (owner Q6=A): the versions and commit hashes stay in Dashboard → Updates and in
// the capsule's title, where updatePillDetail puts them.
export function updatePillText() {
    return 'Update available';
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

// The plan pins the prose: only a description of exactly its range is shown. The owner's request is
// not prose: it stays the live status's own, so a task that saves or ends meanwhile is still followed.
export function bindPreflightDescription(status, preflight) {
    const plan = preflight?.merge_plan || {};
    const letter = preflight?.letter;
    const matches = letter?.key?.base_sha === plan.base_sha && letter?.key?.target_sha === plan.target_sha;
    const request = status?.letter?.request || null;
    return { ...status, preflight_range: plan, checked_target_sha: plan.target_sha,
        latest_sha: plan.target_sha, letter: matches ? { ...letter, request } : (request && { request }) };
}

/** The saved description of exactly these preflight pins, as Markdown to render; '' when there is none. */
export function preflightDescription(preflight) {
    const letter = bindPreflightDescription({}, preflight).letter;
    return letter?.description_current && letter.text ? String(letter.text) : '';
}

/**
 * The exact range's evidence as the confirmation's detail rows: every first-parent commit's receipt
 * (declarations stay labelled as such), the post-merge read and each observed run. Missing, unread and
 * non-green stay distinct; nothing here becomes a verdict.
 * @returns {Array<{label: string, value: string}>}
 */
export function preflightEvidenceRows(preflight) {
    const plan = preflight?.merge_plan || {};
    const facts = preflight?.qualification;
    if (!facts || facts.base_sha !== plan.base_sha || facts.target_sha !== plan.target_sha) {
        return [{ label: 'Checks', value: 'Qualification for this range is unknown.' }];
    }
    const rows = [];
    for (const commit of facts.commits || []) {
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
        const review = commit.receipt.review || {};
        lines.push(`Review observed ${review.observed || 'unknown'}, author declared ${review.declared || 'unknown'}.`);
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
// is a measurement, so null stays "unobserved", never 0. Peak RSS keeps its word; lifetime CPU time and one
// interval's share are separate rows; the dispatch time runs from acceptance to the first model call, not to an
// answer. Each side is one recorded sample: history, never a verdict about the update.
const measured = (value, unit) => (typeof value === 'number' ? `${value.toFixed(2)} ${unit}` : 'unobserved');

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
    const percent = measured(sample.cpu_percent, '%');
    const interval = measured(sample.cpu_interval_seconds, 'seconds');
    return [
        { label: 'Source', value: sample.source_sha || 'unknown source' },
        // The local minute to read, with the exact recorded stamp under it.
        when ? { label: 'Observed', value: `${when} ${utcOffset(new Date(sample.observed_at))}`, exact: sample.observed_at }
            : { label: 'Observed', value: sample.observed_at || 'unknown time' },
        { label: `RSS${sample.rss_measure === 'peak' ? ' peak' : ''}`,
            value: measured(typeof sample.rss_bytes === 'number' ? sample.rss_bytes / 1048576 : null, 'MiB') },
        { label: 'CPU time, process lifetime', value: measured(sample.cpu_seconds, 'seconds') },
        { label: 'CPU share, sample interval',
            value: percent === 'unobserved' && interval === 'unobserved' ? 'unobserved' : `${percent} over ${interval}` },
        { label: 'Duty overruns', value: sample.duty_overruns == null ? 'unobserved' : JSON.stringify(sample.duty_overruns) },
        { label: 'Direct message to first model call', value: typeof direct?.seconds === 'number'
            ? `${measured(direct.seconds, 'seconds')} from acceptance${direct.task_id ? ` · task ${direct.task_id}` : ''}`
            : 'unobserved' },
    ];
}

/** `after_update` as rows: each side's heading (`head`), then its `{label, value, exact?}` rows; [] without facts. */
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

const REFUSALS = {
    release_moved: 'The update changed after it was prepared.',
    late_local_changes: 'Local files changed while the update was starting.',
    replace_not_applied: 'Git refused the first checkout step.',
};

function attemptText(progress) {
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
    } else {
        parts.push(progress.error);
    }
    if (progress.work_stop === 'invoked') parts.push('Running work may have been interrupted.');
    if (progress.restart_required && progress.current_generation !== false) {
        parts.push('Ouroboros needs a restart before updates can continue.');
    }
    if (progress.keep_branch) parts.push(`Your local commits are kept on branch ${progress.keep_branch}.`);
    if (progress.stash_note) parts.push(progress.stash_note);
    return parts.filter(Boolean).join(' ');
}

function attemptDetails(progress, outcome) {
    const rows = [];
    const add = (label, value) => {
        if (value) rows.push({ label, value: String(value) });
    };
    add('Base', String(progress.base_sha || '').slice(0, 12));
    add('Target', String(progress.target_sha || '').slice(0, 12));
    add('Stage', progress.stage || (outcome ? 'not started' : ''));
    add('Outcome', outcome);
    add('Reason', progress.reason);
    add('Blocked by', (Array.isArray(progress.blockers) ? progress.blockers : []).join(', '));
    add('Error', progress.error);
    add('Saved local work', progress.keep_branch);
    add('Concluded by', { executor: 'this update', boot: 'the restarted server' }[progress.concluded_by]);
    return rows;
}

/**
 * One reading of the retained attempt for the pill and the panel: the stage while it runs, then the
 * server's conclusion or its named uncertainty. `null` when there is no attempt to show.
 */
export function updateAttemptView(progress) {
    if (!progress || typeof progress !== 'object' || !progress.operation_id) return null;
    // A record without the outcome field (an older shape) that failed is still a failure, never a stage.
    const key = progress.outcome || (!progress.active && progress.result === 'failed' ? 'failed' : '');
    const base = {
        id: String(progress.operation_id),
        when: String(progress.outcome_at || progress.stage_started_at || progress.started_at || ''),
        details: attemptDetails(progress, key),
    };
    const retention = progress.retention_unconfirmed
        ? 'This observation could not be saved and may be unavailable after restart.' : '';
    const outcome = OUTCOMES[key];
    if (outcome && !progress.active) {
        return { ...base, terminal: true, tone: outcome.tone, pill: outcome.pill, headline: outcome.headline,
            restoreTarget: key === 'applied' ? String(progress.base_sha || '') : '',
            text: [attemptText({ ...progress, outcome: key }), retention].filter(Boolean).join(' ') };
    }
    // Not concluded: this server's stage, a restart it needs, a reviewed resolution, or a later server's checks.
    const stage = String(progress.stage || '');
    const here = progress.active || progress.current_generation !== false;
    const [pill, headline] = progress.result === 'assisted_started'
        ? ['Resolving update…', 'A reviewed resolution owns this update.']
        : progress.result === 'restart_required' && here
            ? ['Restart needed', 'The update needs a restart to finish.']
            : here
                ? [PILL_STAGES[stage] || 'Updating…', UPDATE_STAGE_LABELS[stage] || 'Updating…']
                : ['Confirming update…', 'Waiting for the restarted server to confirm the update…'];
    return { ...base, terminal: false, tone: 'neutral', pill, headline, text: retention };
}

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

export function initUpdateStatus({ showPage, openDashboardTab, ws, readStatus = () => apiClient.updateStatus() } = {}) {
    let latest = null;

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
                // The control that ended the notice is gone: a keyboard reader lands on what is left of it (the
                // availability it unmasked), else on the sidebar's last row, never on the page.
                if (held) {
                    const rows = document.body.querySelectorAll?.('#primary-sidebar [data-nav-page]') || [];
                    (pill.hidden ? [...rows].pop() : pill)?.focus?.({ preventScroll: true });
                }
            });
            ownedNotice.append(pill, closeButton);
            const anchor = document.getElementById('nav-version');
            if (anchor && anchor.parentNode) {
                anchor.parentNode.insertBefore(ownedNotice, anchor.nextSibling);
            } else {
                document.body.appendChild(ownedNotice);
            }
        }
        return pill;
    }

    function renderPill(status) {
        const pill = ensurePill();
        const attempt = updateAttemptView(status?.update_progress);
        // A stage always shows; a concluded attempt shows only if its outcome has a word for the pill (a success has
        // none) and this browser has not hidden it. What is hidden is the notice: the record stays in the panel.
        const shown = Boolean(attempt?.pill) && !(attempt.terminal && attempt.id === dismissedUpdateAttempt());
        // Content or uncommitted work of the checkout's own is a caveat about an AVAILABLE update, not a
        // count of commits ahead; while an attempt shows, its stage or outcome is the pill's one fact.
        pill.classList.toggle('has-local', Boolean(!shown && status?.available
            && (status.dirty || status.local_changes !== 'absent')));
        hideable = shown && attempt.terminal ? attempt.id : '';
        if (closeButton) closeButton.hidden = !hideable;
        pill.classList.toggle('has-close', Boolean(hideable));
        if (shown) {
            pill.textContent = attempt.pill;
            pill.title = [attempt.headline, attempt.text].filter(Boolean).join(' ');
            pill.dataset.tone = attempt.tone;
        } else {
            delete pill.dataset.tone;
            if (status?.available) {
                pill.textContent = updatePillText(status);
                pill.title = updatePillDetail(status);
            }
        }
        pill.hidden = !shown && !status?.available;
        if (ownedNotice) ownedNotice.hidden = pill.hidden;
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
                const status = await safe(readStatus);
                if (readAgain || disposed) continue; // superseded: never render an older answer
                if (status) latest = status; // a failed read hides nothing already known
                renderPill(latest);
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
    const onDismissed = () => renderPill(latest);
    const onPageShown = (event) => {
        if (event?.detail?.page === 'chat') refresh();
    };
    window.addEventListener('ouro:update-attempt-dismissed', onDismissed);
    window.addEventListener('ouro:page-shown', onPageShown);
    refresh();

    return {
        refresh,
        dispose() {
            disposed = true;
            disposers.splice(0).forEach((dispose) => dispose());
            window.removeEventListener('ouro:update-attempt-dismissed', onDismissed);
            window.removeEventListener('ouro:page-shown', onPageShown);
            ownedNotice?.remove();
        },
    };
}
