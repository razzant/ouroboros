import { escapeHtmlAttr as escapeHtml, lastAppVersion, watchAppVersion } from './utils.js';
import { paintStatusMark, statusMarkHtml } from './status_mark.js';
import { mountVersionList } from './update_versions.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { showToast } from './toast.js';
import { apiClient, apiFetch } from './api_client.js';
import { RECOVERY_RETRY, UPDATE_STAGE_LABELS, afterUpdateRows, assistedUpdatePhase, attemptOutcome, attemptTime, bindPreflightDescription, failureWords, preflightDescription, preflightEvidenceRows, unfinishedRecovery, updateAttemptView, verifiedUpdatePlan } from './update_status.js';
import { destroyChatMarkdown, enhanceChatMarkdown, mountChatMarkdown } from './chat_markdown.js';
import { updateEffectsText } from './update_effects.js';
import { openTaskReference, taskReferenceHtml } from './task_reference.js';
import { mountTabSettings } from './dashboard.js';

// Known non-state warnings are folded into their verdict states; everything
// else is surfaced verbatim so a backend warning class can never vanish.
const STATE_WARNINGS = new Set(['official_status_requires_check', 'managed_updates_unavailable']);
// The headline's one mark per tone (status_mark.js), and the facts said beside it; the rest is the card's Details.
const TONE_MARKS = { ok: 'ok', warn: 'owner', error: 'failed' };
const HEADLINE_FACTS = new Set(['App', 'Official repo', 'Channel']);

// ONE relative-time vocabulary for this panel: the same four buckets carry the
// "checked …" age on the action row and the "written …" age on the letter, so
// two ages on one card can never disagree about what "3 h ago" means.
function relativeAge(iso) {
    if (!iso) return '';
    const then = Date.parse(iso);
    if (Number.isNaN(then)) return '';
    const minutes = Math.max(0, Math.round((Date.now() - then) / 60000));
    if (minutes < 2) return 'just now';
    if (minutes < 90) return `${minutes} min ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 36) return `${hours} h ago`;
    return new Date(then).toISOString().slice(0, 10);
}

function humanizeCheckedAt(iso) {
    const age = relativeAge(iso);
    return age ? `checked ${age}` : '';
}

function repoSlug(url) {
    const match = /github\.com[/:]([^/]+\/[^/.]+)/.exec(String(url || ''));
    return match ? match[1] : String(url || '');
}

function extraWarnings(data) {
    return (Array.isArray(data.warnings) ? data.warnings : [])
        .filter((w) => !STATE_WARNINGS.has(String(w).split(':', 1)[0]) && !STATE_WARNINGS.has(w));
}

const LOCAL_CHANGES = { absent: 'none', present: 'present', unknown: 'unknown' };

// Why an available update is a caution: content of the checkout's own, an
// unknown comparison, or uncommitted work (stashed and restored around the update).
function unsafeUpdateHint(data) {
    const parts = [];
    if (data.local_changes === 'present') {
        parts.push('This checkout changed since its last shared commit with the official line; applying merges those changes.');
    } else if (data.local_changes !== 'absent') {
        parts.push('Could not compare this checkout with the official line; applying merges any local changes.');
    }
    if (data.dirty) parts.push('Uncommitted changes are stashed and restored around the update.');
    return parts.join(' ');
}

function statusReadFailed(data) {
    return Array.isArray(data?.warnings)
        && data.warnings.some((warning) => String(warning).startsWith('status_error:'));
}


function progressVerdict(data, base) {
    const progress = data.update_progress;
    if (!progress?.operation_id || progress.current_generation === false) return null;
    if (progress.active || progress.result === 'restart_requested') {
        return {
            ...base, state: progress.result === 'restart_requested' ? 'restarting' : 'updating',
            tone: 'neutral',
            headline: UPDATE_STAGE_LABELS[progress.stage] || 'Updating…',
            hint: progress.stage_started_at ? `Stage started ${relativeAge(progress.stage_started_at)}.` : '',
            action: { id: 'update', label: 'Updating…', disabled: true },
        };
    }
    if (progress.result === 'restart_required') {
        return { ...base, state: 'restart_required', tone: 'warn',
            headline: 'The update needs a restart that could not be requested automatically.',
            hint: progress.error || 'Restart Ouroboros to finish.',
            action: { id: 'restart', label: 'Restart now' } };
    }
    if (progress.result === 'failed' && progress.restart_required) {
        return { ...base, state: 'restart_needed', tone: 'error',
            headline: 'The update did not complete.', hint: progress.error || 'Restart Ouroboros before trying again.',
            action: { id: 'restart', label: 'Restart now' } };
    }
    return null; // Retained conclusions do not hide the next explicitly requested action.
}

export function applyFailureText(err) {
    const body = err?.body || {};
    const parts = [failureWords(err)];
    if (body.reason) parts.push(`reason: ${body.reason}`);
    if (Array.isArray(body.blockers) && body.blockers.length) {
        parts.push(`blocked by: ${body.blockers.slice(0, 5).join(', ')}`);
    }
    if (body.rolled_back) parts.push('the update was rolled back');
    if (body.smoke) parts.push('the post-update smoke check failed');
    if (typeof body.estimated_wave_usd === 'number') {
        parts.push(`assisted review needs ~$${body.estimated_wave_usd}` + (
            typeof body.remaining_usd === 'number' ? ` of $${body.remaining_usd} remaining` : ''
        ));
    }
    if (body.stash_note) parts.push(body.stash_note);
    return parts.join(' · ');
}

// Verdict function: durable server state × transient client phase → one
// presentation descriptor (deterministic given status, phase, and the clock —
// humanizeCheckedAt reads Date.now for the "checked N ago" age). The button is always a real next action; facts
// travel as status line, hint, and chips (ARCHITECTURE §3: unavailable,
// divergent, dirty, unsafe, failed-check, rollback, and restart-required
// states stay visible — they are just not buttons). `appVersion` is the app (host) version About shows: the
// headline names the code an update changes, the App chip the app that a new download changes.
export function updateVerdict(data = {}, phase = '', appVersion = '') {
    const checkedAgo = humanizeCheckedAt(data.checked_at);
    const chips = appVersion ? [{ label: 'App', value: appVersion }] : [];
    if (data.official_repo_url) chips.push({ label: 'Official repo', value: repoSlug(data.official_repo_url) });
    if (data.target_ref) chips.push({ label: 'Target', value: data.target_ref });
    if (data.update_channel) chips.push({ label: 'Channel', value: data.update_channel });
    // Commit counts are history (the updater's own merges count too); whether the
    // checkout holds content of its own is the server's `local_changes` fact.
    const commits = [];
    if (data.behind) commits.push(`${data.behind} incoming`);
    if (data.ahead) commits.push(`${data.ahead} local`);
    if (commits.length) chips.push({ label: 'Commits', value: commits.join(' / ') });
    if (LOCAL_CHANGES[data.local_changes]) {
        chips.push({ label: 'Local changes', value: data.local_changes === 'present'
            ? `present since ${String(data.local_changes_base || '').slice(0, 8)}` : LOCAL_CHANGES[data.local_changes] });
    }
    if (data.dirty_count) chips.push({ label: 'Uncommitted', value: `${data.dirty_count} file(s)` });

    const warnings = extraWarnings(data);
    const base = { chips, warnings, checkedAgo };
    // `recoveryInstruction`: this verdict's hint is the quit/reopen instruction, so Last attempt does not repeat one.
    if (data.update_tx?.active && data.update_tx.local_work_recovery) {
        return { ...base, state: 'resolving', tone: 'warn', recoveryInstruction: true,
            headline: 'Local changes still need recovery confirmation.',
            hint: 'Quit and reopen the desktop app, or restart the server process for a web deployment. '
                + 'In-app Restart is deferred to preserve the current files. The saved update stash is retained.',
            action: { id: 'check', label: 'Check again' } };
    }
    const attempt = data.update_progress;
    const recovery = data.update_tx?.active && (
        ['corrupt', 'gate_blocked', 'marker_cleanup_retry'].includes(data.update_tx.phase)
        || attemptOutcome(attempt) === 'failed' || (!attempt?.active && attempt?.result === 'failed')
    );
    if (recovery) phase = ''; // Durable recovery always outranks an older progress observation.
    if (!recovery && !['restarting', 'restart_needed', 'restart_required'].includes(phase)) {
        const pendingRequest = ['checking', 'preflighting', 'updating'].includes(phase);
        const currentExecution = data.update_progress?.active || data.update_progress?.result === 'restart_requested';
        const progress = !pendingRequest || currentExecution ? progressVerdict(data, base) : null;
        if (progress) return progress;
    }

    if (phase === 'loading') return { ...base, state: 'loading', tone: 'neutral', headline: 'Loading update status…', hint: '', action: null };
    if (phase === 'checking') return { ...base, state: 'checking', tone: 'neutral', headline: 'Checking the official channel…', hint: '', action: { id: 'check', label: 'Checking…', disabled: true } };
    if (phase === 'preflighting') return { ...base, state: 'preflighting', tone: 'neutral', headline: 'Preparing the update…', hint: '', action: { id: 'update', label: 'Preparing…', disabled: true } };
    if (phase === 'updating') return { ...base, state: 'updating', tone: 'neutral', headline: 'Applying the update…', hint: '', action: { id: 'update', label: 'Updating…', disabled: true } };
    if (phase === 'restarting') return { ...base, state: 'restarting', tone: 'ok', headline: 'Restarting the server…', hint: 'This page updates itself when the server is back.', action: { id: 'restart', label: 'Restarting…', disabled: true } };
    if (phase === 'restart_required') return { ...base, state: 'restart_required', tone: 'warn', headline: 'The update needs a restart that could not be requested automatically.', hint: 'Restart Ouroboros to finish.', action: { id: 'restart', label: 'Restart now' } };
    // Distinct from restart_required (the update is applied, or recorded for the restart exit when
    // its target lacks the store, and only its restart is missing): here the operation did NOT
    // complete — the runtime just cannot continue without a restart (failed writer fence, failed
    // rollback, rollback landed without its restart).
    if (phase === 'restart_needed') return { ...base, state: 'restart_needed', tone: 'warn', headline: 'Ouroboros needs a restart before updates can continue.', hint: 'The last operation could not complete cleanly; restart, then retry.', action: { id: 'restart', label: 'Restart now' } };

    const unmanaged = data.managed === false
        || (Array.isArray(data.warnings) && data.warnings.includes('managed_updates_unavailable'));
    if (unmanaged) {
        return {
            ...base,
            state: 'unmanaged', tone: 'disabled',
            headline: 'Managed updates are unavailable for this checkout.',
            hint: 'Use git directly, or install a launcher-managed build.',
            action: null,
        };
    }
    if (data.update_tx?.active) {
        const txPhase = String(data.update_tx.phase || '');
        const task = data.update_tx.task_id ? ` (task ${data.update_tx.task_id})` : '';
        if (txPhase === 'corrupt') {
            // Boot recovery quarantines readable corruption when no live merge
            // is present. It deliberately leaves unreadable/rename-failed or
            // merge-owned markers fail-closed. The in-app restart is itself
            // deferred while the marker is corrupt, so do not offer a control
            // that cannot reach that boot recovery path.
            return {
                ...base,
                state: 'resolving', tone: 'error', recoveryInstruction: true,
                headline: 'The update transaction marker is corrupt.',
                hint: 'Quit and reopen Ouroboros so boot recovery can quarantine readable corruption when no merge is active. The in-app restart is deferred while this marker is corrupt. If this state remains after reopening, inspect the marker file (ouroboros-update-tx.json in the repository .git directory) manually, then check again.',
                action: { id: 'check', label: 'Check again' },
            };
        }
        const unfinished = unfinishedRecovery(attempt, data.update_tx);
        if (unfinished) {
            // The bound attempt's own boot report outranks the phase it left behind (an assisted one included); the
            // phase keeps its own control: pending_boot_smoke still allows the in-app Restart that reruns that recovery.
            return {
                ...base,
                state: 'resolving', tone: 'error', recoveryInstruction: true,
                headline: unfinished.headline,
                hint: RECOVERY_RETRY,
                action: txPhase === 'pending_boot_smoke' ? { id: 'restart', label: 'Restart now' }
                    : { id: 'check', label: 'Check again' },
            };
        }
        if (txPhase === 'pending_boot_smoke') {
            return {
                ...base,
                state: 'resolving', tone: 'warn',
                headline: 'An update is waiting for a restart to finish.',
                hint: 'Restart Ouroboros to run the post-update checks and complete it.',
                action: { id: 'restart', label: 'Restart now' },
            };
        }
        if (txPhase === 'gate_blocked' || txPhase === 'marker_cleanup_retry') {
            // The in-app restart defers in these phases (server_restart._safe_restart_serialized),
            // so only a new process reaches boot recovery: offer no Restart control.
            return {
                ...base,
                state: 'resolving', tone: 'error', recoveryInstruction: true,
                headline: 'An update attempt stopped mid-flight and needs recovery.',
                hint: `Phase: ${txPhase}. Quit and reopen the desktop app, or restart the server process for a web deployment: boot recovery retries the rollback/cleanup. The in-app Restart is deferred in this phase.`,
                action: { id: 'check', label: 'Check again' },
            };
        }
        if (assistedUpdatePhase(txPhase)) {
            return {
                ...base,
                state: 'resolving', tone: 'warn',
                headline: 'A conflicting update is being resolved under review.',
                hint: `Watch progress in chat${task}. Applying another update waits for this resolution.`,
                action: null,
            };
        }
        return {
            ...base,
            state: 'resolving', tone: 'warn',
            headline: 'An update transaction is still active.',
            hint: `Phase: ${txPhase || 'unknown'}. Another update waits until it settles.`,
            action: null,
        };
    }
    const requiresCheck = !data.from_cache && Array.isArray(data.warnings) && data.warnings.includes('official_status_requires_check');
    if (requiresCheck && !data.checked_at) {
        return {
            ...base,
            state: 'unchecked', tone: 'neutral',
            headline: 'Official update status has not been checked yet.',
            hint: '',
            action: { id: 'check', label: 'Check for updates' },
        };
    }
    // A failure says so in words, its codes are the record's (Details); the button keeps the name it had.
    if (data.check_ok === false) {
        const lastGood = relativeAge(data.checked_at);
        return {
            ...base, checkedAgo: '', codes: warnings,
            state: 'check_failed', tone: 'error',
            headline: 'Could not check the official update channel.',
            hint: lastGood ? `The last good check was ${lastGood}.` : 'Try again when the network is available.',
            warnings: [],
            action: { id: 'check', label: data.checked_at ? 'Check again' : 'Check for updates' },
        };
    }
    // The versions are the headline; the commits they name are the record's.
    const currentVersion = data.current_version || 'unknown';
    if (data.current_short_sha) chips.push({ label: 'Running commit', value: data.current_short_sha });
    if (data.available) {
        const latestVersion = data.latest_version || 'unknown';
        const unsafe = !data.safe_to_apply;
        if (data.latest_short_sha) chips.push({ label: 'Update commit', value: data.latest_short_sha });
        if (data.latest_message) chips.push({ label: 'Latest', value: data.latest_message });
        const target = latestVersion !== currentVersion ? `${currentVersion} → ${latestVersion}`
            : data.behind ? `${data.behind} new ${data.behind === 1 ? 'change' : 'changes'}` : latestVersion;
        return {
            ...base,
            state: unsafe ? 'available_unsafe' : 'available',
            tone: unsafe ? 'warn' : 'ok',
            headline: `Update available: ${target}`,
            hint: unsafe ? unsafeUpdateHint(data) : '',
            action: { id: 'update', label: `Update to ${latestVersion}` },
        };
    }
    // "Up to date" is only claimed over an actual check result: a fresh
    // check_ok, or a cache-carried timestamp of the last real check.
    if (data.check_ok === true || data.checked_at) {
        return {
            ...base,
            state: 'current', tone: 'ok',
            headline: `Up to date · ${currentVersion}`,
            hint: '',
            action: { id: 'check', label: 'Check again' },
        };
    }
    return {
        ...base, codes: warnings,
        state: 'unknown', tone: 'neutral',
        headline: 'Update status is unknown.',
        hint: 'Run a check to get the official status.',
        warnings: [],
        action: { id: 'check', label: 'Check for updates' },
    };
}

// --- The update description -------------------------------------------------
//
// Ouroboros writes the description of the update the panel is offering as
// ordinary work: when the owner presses Write/Rewrite here (one ordinary Main
// task, whose status the backend reads from its task result) or on its own
// initiative. A check never writes it. The backend hands it to the client
// inside the ordinary status payload as the additive `letter` key. It is never
// deleted once the update lands: the same text, relabelled, becomes "What
// changed in this version", and a text written for an older range keeps that
// range's label when a newer target appears. Description work and its failures
// stay separate from the primary installation action.

// Hidden where the description could only mislead: a checkout that managed
// updates do not own (unmanaged), an unknown status, a target no fetching check
// has named yet (unchecked), and the restart phase, whose served-SHA reload owns
// the card. Loading and checking KEEP it: the text is still the last known fact,
// and a passive refresh or a failed/running check must not blank it — that is
// what the renderer's content key is for.
const LETTER_HIDDEN_VERDICTS = new Set(['unmanaged', 'unknown', 'unchecked']);
const LETTER_HIDDEN_PHASES = new Set(['restarting']);
const LETTER_BUSY_PHASES = new Set(['preflighting', 'updating', 'restarting']);
const LETTER_RELATIONS = new Set(['pending', 'applied', 'superseded', 'other']);
// The request's own ordinary task statuses during which it may still save.
const LIVE_REQUEST_STATUSES = new Set(['requested', 'scheduled', 'running', 'interrupted', 'cancel_requested']);
// `applied` is the one relation whose label changes: the running version IS
// the description's target, so the text is history, not a preview.
const LETTER_LABELS = {
    pending: "What's new",
    applied: 'What changed in this version',
    superseded: "What's new",
    other: "What's new",
};

function noLetterView() {
    return {
        state: 'none',
        relation: '',
        markdown: '',
        meta: { authorVersion: '', targetVersion: '', writtenAt: '', ageText: '' },
        failure: null,
        label: '',
        note: '',
        covers: '',
        write: null,
        request: null,
    };
}

const shortRange = (key) => (key?.base_sha && key?.target_sha
    ? `${String(key.base_sha).slice(0, 8)} → ${String(key.target_sha).slice(0, 8)}` : '');
const sameRange = (a, b) => Boolean(a?.base_sha && a?.target_sha
    && a.base_sha === b?.base_sha && a.target_sha === b?.target_sha);

function requestView(letter) {
    const request = letter?.request;
    if (!request?.task_id) return null;
    const status = String(request.status || 'unknown');
    return { taskId: String(request.current_task_id || request.task_id), mode: String(request.mode || ''), key: request.key || {},
        requestedAt: String(request.requested_at || ''), status, saved: request.saved === true,
        live: LIVE_REQUEST_STATUSES.has(status) };
}

// What a finished request that saved nothing leaves to say; a saved one says nothing.
function endedRequestNote(request) {
    if (!request || request.live || request.saved) return '';
    const how = { completed: 'finished', failed: 'failed', cancelled: 'was stopped' }[request.status];
    return how ? `The last writing task ${how} without saving a description.`
        : 'The last writing task left no readable result; nothing was saved.';
}

/**
 * Description projector: durable server state × transient client phase → one
 * presentation descriptor, pure but for the clock `relativeAge` reads.
 *
 * `state: 'none'` hides the section; `'empty'` shows it without text, offering
 * Write for the update on offer. `write` is the one authoring control: Write
 * for a range nothing describes yet, Rewrite for the shown text's own range
 * (an applied one included), disabled while a requested task still runs.
 */
export function updateLetterView(data = {}, phase = '') {
    // No record yet still offers Write for an update on offer: the server checks the range.
    const letter = data?.letter && typeof data.letter === 'object' ? data.letter : {};
    if (data.preflight_range && letter.text && (letter.key?.base_sha !== data.preflight_range.base_sha
        || letter.key?.target_sha !== data.preflight_range.target_sha)) return noLetterView();
    if (LETTER_HIDDEN_PHASES.has(phase)) return noLetterView();
    if (LETTER_HIDDEN_VERDICTS.has(updateVerdict(data, phase).state)) return noLetterView();

    const markdown = String(letter.text || '').trim();
    const state = letter.state === 'failed' ? 'failed' : (markdown ? 'ready' : 'empty');
    const runningBase = String(data.running_sha || data.current_sha || '');
    const latestTarget = String(data.checked_target_sha || data.latest_sha || '');
    const offered = data.available && runningBase && latestTarget && runningBase !== latestTarget
        && (!letter.checked_key || sameRange(letter.checked_key, { base_sha: runningBase, target_sha: latestTarget }))
        ? { ...(letter.checked_key || {}), base_sha: runningBase, target_sha: latestTarget } : null;
    const request = requestView(letter);
    if (!markdown && !offered && !request?.live && state !== 'failed') return noLetterView();

    const relation = markdown && LETTER_RELATIONS.has(letter.relation) ? letter.relation : (markdown ? 'other' : '');
    const meta = {
        authorVersion: String(letter.author_version || ''),
        targetVersion: String(letter.target_version || ''),
        writtenAt: String(letter.written_at || ''),
        ageText: relativeAge(letter.written_at),
    };
    // The server compares the shown text's complete source range. Neither the
    // version label, a newer post-merge run nor mutable checkout HEAD moves it.
    const descriptionCurrent = letter.description_current === true;
    const failedKey = letter.latest_failed_key;
    const failure = state === 'failed' && !descriptionCurrent && (!failedKey
        || (failedKey.base_sha === runningBase && failedKey.target_sha === latestTarget))
        ? { kind: String(letter.error_kind || ''), text: String(letter.error_text || ''),
            failedAt: String(letter.failed_at || ''), key: failedKey || null } : null;
    // Write targets the update on offer; with none, Rewrite rephrases the shown text's own range.
    const writeKey = offered || (markdown && letter.key?.base_sha ? letter.key : null);
    const rewriting = Boolean(writeKey && markdown && sameRange(letter.key, writeKey));
    const write = writeKey ? {
        mode: rewriting ? 'rewrite' : 'write',
        key: writeKey,
        label: request?.live ? 'Writing…' : (rewriting ? 'Rewrite' : 'Write description'),
        disabled: Boolean(request?.live) || LETTER_BUSY_PHASES.has(phase),
    } : null;
    let note = '';
    if (request?.live) {
        // Once its own save landed, the shown text is no longer the one waiting to be replaced. The time
        // is the owner's press, a fixed fact: the task's own card (the note's door) shows what it does.
        const asked = attemptTime(request.requestedAt);
        note = request.saved ? `Ouroboros saved the description; its task (${request.taskId}) is still running.`
            : `Ouroboros is writing the description${asked ? ` you asked for on ${asked}` : ''} (task ${request.taskId}).${
                markdown ? ' The current text stays until the new one is saved.' : ''}`;
    } else if (failure) {
        const when = attemptTime(failure.failedAt);
        note = `Refresh failed${when ? ` on ${when}` : ''}. ${markdown
            ? 'Previous description kept.' : 'No description is available yet.'}`;
    } else if (!markdown) {
        note = endedRequestNote(request) || 'No description of this update yet.';
    } else if (offered && !descriptionCurrent) {
        note = `This description covers an earlier update (${shortRange(letter.key)}).`;
    } else if (!offered && !descriptionCurrent && relation !== 'applied') {
        note = 'This description was written for an earlier update.';
    }
    const ended = request && sameRange(request.key, writeKey) ? endedRequestNote(request) : '';
    if (ended && markdown && !request.live) note = note ? `${note} ${ended}` : ended;

    return {
        state,
        relation,
        markdown,
        meta,
        failure,
        descriptionCurrent,
        label: LETTER_LABELS[relation] || "What's new",
        note,
        covers: markdown ? shortRange(letter.key) : '',
        write,
        request,
    };
}

/** Content identity of a rendered description: re-render only when this changes.
 *  The markdown itself, never a length proxy — two different texts of equal length
 *  are a different description. */
function letterContentKey(view) {
    // Status and provenance updates must not replace the text being read.
    return view.markdown;
}

// Mirrors the two boot-recovery phases admitted by server.py's serialized
// restart path. A later backend phase remains safe: the UI falls through to
// its durable verdict instead of inventing another transient hold.
const RESTART_BOOT_PHASES = new Set(['pending_boot_smoke', 'applying_replace']);

// A reconnect proves that the browser reached a server generation after the
// restart request, but the managed-update boot finalizer can still own the
// durable transaction. Keep the synthetic phase until that boot-only state is
// gone; every other durable status is more truthful than "Restarting…".
export function restartStatusCanSettle(data, { afterBootNotice = false } = {}) {
    if (!data || typeof data !== 'object') return false;
    if (statusReadFailed(data)) return false;
    if (!data.update_tx?.active) return true;
    const txPhase = String(data.update_tx.phase || '');
    // update_status_ready is emitted after the boot finalizer returns. If it
    // deliberately leaves a boot phase durable (for example, a retry is still
    // required), that verdict now owns the UI instead of synthetic restarting.
    return afterBootNotice || !RESTART_BOOT_PHASES.has(txPhase);
}

// Updates is mounted for the SPA lifetime. Keep its reconnect episode local:
// update_status_ready is transient and can arrive from the old generation, so
// it may reconcile a restart only after the socket has actually reopened.
export function bindUpdateRefreshEvents({ ws, getPhase, reconcileRestart, loadStatus, getRequestTaskId = () => '' }) {
    let restartReconnected = false;
    let continuedTaskId = ''; // Event correlation only; the status response owns liveness.
    const disposers = [];
    const listen = (event, handler) => {
        const dispose = ws?.on?.(event, handler);
        if (typeof dispose === 'function') disposers.push(dispose);
    };

    listen('open', (event = {}) => {
        if (event.previouslyConnected !== true) return;
        if (getPhase() === 'restarting') {
            restartReconnected = true;
            reconcileRestart({ afterBootNotice: false });
        } else {
            loadStatus({ fetchRemote: false, preservePhase: ['preflighting', 'updating'].includes(getPhase()) });
        }
    });
    listen('update_progress_changed', () => {
        if (getPhase() === 'restarting') {
            if (restartReconnected) reconcileRestart({ afterBootNotice: false });
        } else {
            loadStatus({ fetchRemote: false, preservePhase: true });
        }
    });
    // The requested description task ends through the ordinary task lifecycle;
    // its own task_done (saved or not) is the one cue to read the status again.
    // A host retry ends under its own id, naming the request as its root.
    listen('log', (msg) => {
        const evt = msg?.data;
        const taskId = getRequestTaskId();
        // Continue may be pressed in another window or through another owner interface.
        // Its existing host log reaches every socket; the status reader resolves the new author.
        if (!evt || evt._historical || !taskId || getPhase() === 'restarting') return;
        const matches = (id) => id === taskId || (continuedTaskId && id === continuedTaskId);
        const continued = evt.type === 'owner_continue_admitted' && matches(evt.predecessor_task_id);
        const ended = evt.type === 'task_done'
            && (matches(evt.task_id) || (matches(evt.root_task_id) && !evt.parent_task_id));
        if (continued && typeof evt.task_id === 'string') continuedTaskId = evt.task_id;
        if (continued || ended) loadStatus({ fetchRemote: false, preservePhase: true });
    });
    listen('update_status_ready', () => {
        const phase = getPhase();
        if (phase === 'restarting') {
            if (restartReconnected) reconcileRestart({ afterBootNotice: true });
            return;
        }
        if (phase === '' || phase === 'loading') loadStatus({ fetchRemote: false });
    });

    return {
        beginRestarting() { restartReconnected = false; },
        dispose() { disposers.splice(0).forEach((dispose) => dispose()); },
    };
}

// The App chip follows the health read About uses: one that lands after a status rendered re-renders the panel;
// a panel still loading keeps its loading line until its own status render. Returns the disposer.
export function bindAppVersionRender({ getStatus, render }) {
    return watchAppVersion(() => { if (getStatus()) render(); });
}

export function initUpdates({ mount, state, ws }) {
    // The tab's one root and scroller (style.css "The Dashboard tab body"): the state of the channel leads as the
    // tab's headline with its facts and one next action; the last attempt, the record's facts, the description and
    // Recovery follow, each behind its own disclosure. The channel itself is chosen in the card below.
    const page = Object.assign(document.createElement('div'), { id: 'page-updates', className: 'dashboard-body scroll-fade-y' });
    page.innerHTML = `
            <section class="ui-card updates-card" id="updates-status-card">
                <div class="dashboard-hero">
                    <p class="dashboard-hero-kicker">Official updates</p>
                    <p class="dashboard-hero-figure"><span id="updates-summary">Loading update status…</span>${statusMarkHtml('consciousness-mark updates-mark')}</p>
                    <p class="dashboard-hero-meta" id="updates-meta"></p>
                    <p class="dashboard-hero-meta" id="updates-action-note"></p>
                    <div class="dashboard-hero-actions"><button type="button" class="btn btn-primary" id="btn-update-primary" hidden></button></div>
                </div>
                <p class="updates-hint" id="updates-hint" hidden></p>
                <div class="updates-attempt" id="updates-attempt" hidden>
                    <details class="ui-disclosure updates-attempt-details" id="updates-attempt-details">
                        <summary><span class="updates-attempt-line"><time id="updates-attempt-time">Last attempt</time><span class="ui-status" id="updates-attempt-headline"></span></span></summary>
                        <dl class="dashboard-facts" id="updates-attempt-facts"></dl>
                        <details class="ui-disclosure" id="updates-attempt-raw" hidden>
                            <summary id="updates-attempt-raw-label">Raw record</summary>
                            <pre class="ui-result-details" id="updates-attempt-raw-text"></pre>
                        </details>
                    </details>
                    <button type="button" class="btn btn-default btn-sm" id="updates-attempt-restore" hidden>Restore previous version</button>
                    <p class="updates-hint" id="updates-attempt-text"></p>
                </div>
                <details class="ui-disclosure updates-facts" id="updates-facts-more" hidden><summary>Details</summary><dl class="dashboard-facts" id="updates-facts"></dl></details>
                <section class="updates-letter" id="updates-letter" aria-labelledby="updates-letter-label" hidden>
                    <div class="updates-letter-head">
                        <div class="updates-letter-heading">
                            <h4 class="updates-letter-label" id="updates-letter-label"></h4>
                            <span class="updates-letter-meta"><time id="updates-letter-meta"></time></span>
                        </div>
                        <div class="updates-letter-actions">
                            <button type="button" class="btn btn-ghost btn-sm updates-letter-refresh" id="updates-letter-refresh"
                                aria-label="Check for updates" aria-describedby="updates-letter-resources">
                                <span aria-hidden="true">↻</span><span id="updates-letter-refresh-label">Check</span>
                            </button>
                            <button type="button" class="btn btn-default btn-sm updates-letter-write" id="updates-letter-write"
                                aria-describedby="updates-letter-resources" hidden></button>
                        </div>
                    </div>
                    <div class="updates-letter-note" id="updates-letter-note-line" hidden><span id="updates-letter-note" role="status" aria-live="polite"></span> <span id="updates-letter-task"></span></div>
                    <div class="updates-letter-body ui-rich-content" id="updates-letter-body"></div>
                    <button type="button" class="btn btn-ghost btn-sm ui-list-more" id="updates-letter-more" hidden>Read all</button>
                    <details class="ui-disclosure updates-letter-details" id="updates-letter-details">
                        <summary>Details</summary>
                        <p id="updates-letter-provenance"></p>
                        <p id="updates-letter-error" hidden></p>
                        <p id="updates-letter-resources">Check reads the official channel and never writes a description. Writing runs one ordinary Ouroboros task on your Main model, which can read the changes themselves; it may use subscription limits, local resources, or API budget.</p>
                    </details>
                </section>
                <details class="ui-disclosure updates-recovery">
                    <summary>Recovery</summary>
                    <p class="updates-recovery-copy">Replace the active checkout with the exact official version from the selected channel. A rescue copy is saved first, but this is intentionally more destructive than an ordinary update.</p>
                    <div class="updates-recovery-actions">
                        <button type="button" class="btn btn-danger btn-sm" id="btn-update-replace">Replace with the official version</button>
                        <button type="button" class="btn btn-default btn-sm" id="updates-promote">Save recovery point</button>
                    </div>
                    <p class="updates-branch" id="updates-current"></p>
                    <h4 class="updates-subhead">Restore an earlier version</h4>
                    <div id="updates-commits" class="updates-restore-list"></div>
                    <button type="button" class="btn btn-ghost btn-sm ui-list-more" id="updates-commits-more" hidden></button>
                    <h4 class="updates-subhead">Official releases</h4>
                    <div id="updates-official-tags" class="updates-restore-list"></div>
                </details>
            </section>
            <div class="dashboard-settings-cards" data-updates-settings></div>
    `;
    mount.appendChild(page);
    // The channel: a card of choices drawn once, out of every repaint below. Each settings read marks the chosen one.
    const channelCard = page.querySelector('[data-updates-settings]');
    mountTabSettings(channelCard, 'updates', { state, cards: [{ label: 'Which updates', solo: true }], onSaved: () => loadStatus({ fetchRemote: false }),
        onPaint: (settings, chosen = String(settings?.OUROBOROS_UPDATE_CHANNEL ?? '')) => channelCard.querySelectorAll('[data-setting-value]')
            .forEach((choice) => choice.setAttribute('aria-pressed', String(choice.dataset.settingValue === chosen))),
        custom: (row) => `<div class="settings-choices" role="group" aria-labelledby="setting-${row.id}-name">${row.control.options.map((option) =>
            `<button type="button" class="settings-choice dashboard-choice" data-setting-value="${option.value}" aria-pressed="false"><span class="settings-choice-name">${option.label}</span>`
            + `<span class="settings-choice-note">${option.note}</span></button>`).join('')}</div>` });

    const primaryBtn = page.querySelector('#btn-update-primary');
    let restartNeeded = false;  // panel-lifetime restart continuation (no durable marker exists for fence/rollback refusals)
    let replaceInFlight = false; // latch: a pending recovery request keeps Replace disabled across re-renders (tab reopen included)
    const replaceBtn = page.querySelector('#btn-update-replace');
    const dot = page.querySelector('.updates-mark');
    const summary = page.querySelector('#updates-summary');
    const hint = page.querySelector('#updates-hint');
    const factsMore = page.querySelector('#updates-facts-more');
    const facts = page.querySelector('#updates-facts');
    const attemptSection = page.querySelector('#updates-attempt');
    const attemptHeadline = page.querySelector('#updates-attempt-headline');
    const attemptWhen = page.querySelector('#updates-attempt-time');
    const attemptText = page.querySelector('#updates-attempt-text');
    const attemptFacts = page.querySelector('#updates-attempt-facts');
    const attemptRestore = page.querySelector('#updates-attempt-restore');
    const attemptRaw = page.querySelector('#updates-attempt-raw');
    const attemptRawLabel = page.querySelector('#updates-attempt-raw-label');
    const attemptRawText = page.querySelector('#updates-attempt-raw-text');
    const meta = page.querySelector('#updates-meta');
    const actionNote = page.querySelector('#updates-action-note');
    const current = page.querySelector('#updates-current');
    const commitsDiv = page.querySelector('#updates-commits');
    const commitsMore = page.querySelector('#updates-commits-more');
    const officialTagsDiv = page.querySelector('#updates-official-tags');
    const letterSection = page.querySelector('#updates-letter');
    const letterLabel = page.querySelector('#updates-letter-label');
    const letterMeta = page.querySelector('#updates-letter-meta');
    const letterMore = page.querySelector('#updates-letter-more');
    const letterWrite = page.querySelector('#updates-letter-write');
    const letterNote = page.querySelector('#updates-letter-note');
    const letterNoteLine = page.querySelector('#updates-letter-note-line');
    const letterTask = page.querySelector('#updates-letter-task');
    const letterBody = page.querySelector('#updates-letter-body');
    const letterRefresh = page.querySelector('#updates-letter-refresh');
    const letterRefreshLabel = page.querySelector('#updates-letter-refresh-label');
    const letterProvenance = page.querySelector('#updates-letter-provenance');
    const letterError = page.querySelector('#updates-letter-error');
    let letterRefreshing = false;
    let letterRefreshError = null;
    let letterRequesting = false;
    let letterRequestError = null;
    let letterDisposer = null, letterKey = null;
    // "Read all" offers the rest only when the clamped body hides some. A body in a hidden tab or page measures 0 x 0,
    // which says nothing, so that measure stays owed until the body has a height (its tab or the Dashboard shown, or
    // the next render). A new width refolds a folded text, so it owes the measure again.
    let letterMeasureOwed = false;
    let letterDoor = '';
    let latestStatus = null;
    let phase = 'loading';
    let restartReconcileInFlight = false;
    let restartReconcileQueued = false;
    let restartReconcileAfterBootNotice = false;

    // The headline's facts are one flat line (the app, the official repository, the channel); the rest of the record is
    // the card's Details. A group heading spans both columns of a facts list.
    const factLine = (chip) => escapeHtml(chip.label === 'App' ? `App ${chip.value}` : chip.label === 'Channel' ? `${chip.value} channel` : chip.value);
    const factRowHtml = (row) => (row.head ? `<dt class="updates-facts-head">${escapeHtml(row.head)}</dt>`
        : `<dt>${escapeHtml(row.label)}</dt><dd>${escapeHtml(row.value)}</dd>`);

    // The letter body hosts rendered markdown, so its enhancement owns real
    // resources (Chart instances, mermaid timers, a click handler). Release
    // them BEFORE the innerHTML that would orphan them.
    function releaseLetterBody() {
        if (letterDisposer) {
            letterDisposer();
            letterDisposer = null;
        } else {
            destroyChatMarkdown(letterBody);
        }
    }

    function measureLetter() {
        if (!letterMeasureOwed || letterBody.hidden || !letterBody.clientHeight) return;
        letterMeasureOwed = false;
        letterMore.hidden = letterBody.scrollHeight <= letterBody.clientHeight + 1;
    }

    /** The letter is a fact: no control the markdown pipeline adds may survive in it. */
    function stripLetterControls(root = letterBody) {
        root.querySelectorAll('button').forEach((control) => control.remove());
    }

    /** The Apply confirmation's copy of the description: the same pipeline and scrub as the panel. */
    function mountDescription(container, markdown) {
        const heading = document.createElement('h4');
        heading.className = 'updates-letter-label';
        heading.textContent = "What's new";
        const body = document.createElement('div');
        body.className = 'updates-letter-body ui-rich-content';
        container.append(heading, body);
        mountChatMarkdown(body, markdown);
        stripLetterControls(body);
        const dispose = enhanceChatMarkdown(body, { onDomWrite: (mutate) => { mutate(); stripLetterControls(body); } });
        stripLetterControls(body);
        return () => (typeof dispose === 'function' ? dispose() : destroyChatMarkdown(body));
    }

    // Called from render() on every phase change and status load. The head is
    // cheap text, but the body is re-rendered ONLY when its content key moves:
    // an unchanged letter keeps its DOM through a passive refresh and a running
    // check, and with it the owner's selection and any mounted chart.
    function renderLetter() {
        const view = updateLetterView(latestStatus || {}, phase);
        if (view.state === 'none') {
            if (letterKey !== null) {
                releaseLetterBody();
                letterBody.innerHTML = '';
                letterKey = null;
                letterMeasureOwed = false;
            }
            letterSection.hidden = true;
            return;
        }
        letterSection.hidden = false;
        letterSection.setAttribute('aria-busy', String(letterRefreshing || letterRequesting || Boolean(view.request?.live)));
        letterLabel.textContent = view.label;
        // When it was written is the heading's one fact; the range it covers is its provenance (Details).
        const written = view.markdown ? attemptTime(view.meta.writtenAt) : '';
        letterMeta.textContent = written ? `Written ${written}` : '';
        letterMeta.dateTime = view.meta.writtenAt;
        letterMeta.hidden = !written;
        // One control per intent: when the primary action already checks, the section adds none.
        letterRefresh.hidden = updateVerdict(latestStatus || {}, phase).action?.id === 'check';
        letterRefresh.disabled = letterRefreshing || LETTER_BUSY_PHASES.has(phase);
        letterRefreshLabel.textContent = letterRefreshing ? 'Checking…' : 'Check';
        letterRefresh.setAttribute('aria-label', letterRefreshing ? 'Checking for updates' : 'Check for updates');
        letterWrite.hidden = !view.write;
        letterWrite.disabled = !view.write || view.write.disabled || letterRequesting || letterRefreshing;
        letterWrite.textContent = letterRequesting ? 'Requesting…' : (view.write?.label || '');
        // A failed check is said once, by the headline; the description it left alone needs no second line.
        const note = letterRefreshing ? (view.markdown ? 'Checking the official channel; the description stays as it is.' : 'Checking the official channel…')
            : letterRequestError ? `Could not request a description: ${letterRequestError.text}` : view.note;
        letterNote.textContent = note;
        letterNoteLine.hidden = !note;
        // A running request's own note opens its task's card in Main; rewritten only when the task changes,
        // so a passive refresh keeps focus on it.
        const door = note === view.note && view.request?.live ? taskReferenceHtml(view.request.taskId) : '';
        if (door !== letterDoor) {
            letterTask.innerHTML = door;
            letterDoor = door;
        }
        const letter = latestStatus?.letter || {};
        const author = letter.author?.task_id ? ` in task ${letter.author.task_id}${letter.author.model
            ? ` (${letter.author.model})` : ''}${letter.author.requested ? ', at your request' : ''}` : '';
        letterProvenance.textContent = view.markdown ? `Written by Ouroboros${view.meta.authorVersion
            ? ` ${view.meta.authorVersion}` : ''}${author}${view.meta.targetVersion
            ? ` about ${view.meta.targetVersion}` : ''}.${view.covers ? ` Changes ${view.covers}.` : ''}`
            : (view.write ? `The update on offer: changes ${shortRange(view.write.key)}.` : '');
        letterProvenance.hidden = !letterProvenance.textContent;
        const lastFailure = letterRefreshError?.text || letter.error_text || letter.error_kind || '';
        const failedAt = attemptTime(letterRefreshError?.at || letter.failed_at);
        // Failed-attempt provenance must never borrow the shown
        // body's range, which may describe an earlier, already-applied update.
        const failedKey = letterRefreshError ? null : letter.latest_failed_key;
        const failedRange = failedKey?.base_sha && failedKey?.target_sha ? ` for changes ${shortRange(failedKey)}` : '';
        letterError.textContent = lastFailure
            ? `${letterRefreshError ? 'Last check failed' : 'Last failure'}${failedAt ? ` on ${failedAt}` : ''}${failedRange}: ${lastFailure}` : '';
        letterError.hidden = !lastFailure;
        const nextKey = letterContentKey(view);
        if (nextKey === letterKey) {
            if (letterMeasureOwed) requestAnimationFrame(measureLetter);
            return;
        }
        releaseLetterBody();
        letterKey = nextKey;
        letterBody.hidden = !view.markdown;
        // A new text opens at its first part; "Read all" offers the rest only when there is more than that part.
        letterBody.classList.remove('is-open');
        letterMore.textContent = 'Read all';
        letterMore.hidden = true;
        mountChatMarkdown(letterBody, view.markdown || '');
        letterMeasureOwed = Boolean(view.markdown);
        if (!view.markdown) return;
        requestAnimationFrame(measureLetter);
        // No anchored scroll to protect on this page, so markdown's deferred
        // writes (highlight, latex, mermaid, charts) run directly.
        // The authored body keeps no controls of its own. The shared
        // markdown pipeline adds controls to some blocks (a Copy button on a
        // fenced one, and another when a mermaid block DEGRADES asynchronously), so the
        // scrub runs after every write it makes, not once. The text stays untouched.
        stripLetterControls();
        letterDisposer = enhanceChatMarkdown(letterBody, {
            onDomWrite: (mutate) => {
                mutate();
                stripLetterControls();
            },
        });
        stripLetterControls();
    }

    letterTask.addEventListener('click', openTaskReference);

    // Check is the ordinary fetching check: it records range facts and never
    // writes prose, so an unchanged description stays exactly as it was.
    letterRefresh.addEventListener('click', async () => {
        if (letterRefreshing || letterRefresh.disabled) return;
        letterRefreshing = true;
        letterRefreshError = null;
        letterRequestError = null;
        renderLetter();
        const previous = latestStatus?.letter;
        const previousView = updateLetterView(latestStatus || {}, phase);
        try {
            await loadStatus({ fetchRemote: true, preservePhase: true });
            const current = latestStatus?.letter;
            if (!letterRefreshError && latestStatus?.check_ok === true && previousView.descriptionCurrent
                && updateLetterView(latestStatus, phase).descriptionCurrent && previous?.text === current?.text
                && previous?.written_at === current?.written_at
                && ['base_sha', 'target_sha', 'update_channel', 'target_ref'].every(
                    key => previous?.key?.[key] === current?.key?.[key])) {
                showToast('Description is up to date.', 'info');
            }
        } finally {
            letterRefreshing = false;
            render();
        }
    });

    // Write/Rewrite is the owner's explicit request: one ordinary Main task
    // reads the range and saves a description; nothing repeats on its own.
    letterWrite.addEventListener('click', async () => {
        const view = updateLetterView(latestStatus || {}, phase);
        if (letterRequesting || letterWrite.disabled || !view.write || view.write.disabled) return;
        letterRequesting = true;
        letterRequestError = null;
        renderLetter();
        try {
            await apiClient.updateDescription(view.write.mode, view.write.key);
        } catch (err) {
            // A request still running is shown by the status read below, not as an error.
            if (!(err?.status === 409 && err?.body?.task_id)) {
                letterRequestError = { text: String(err?.body?.error || err?.message || err) };
            }
        } finally {
            letterRequesting = false;
        }
        await loadStatus({ fetchRemote: false, preservePhase: true });
    });

    // The retained last attempt (#1818), apart from the primary action and kept until the next attempt.
    // While this server's attempt runs or awaits its restart, the headline above already names it.
    function renderAttempt(verdict) {
        const view = updateAttemptView(latestStatus?.update_progress, latestStatus?.update_tx,
            { showRecoveryInstruction: !verdict.recoveryInstruction });
        const above = !view?.terminal && ['updating', 'restarting', 'restart_required'].includes(verdict.state);
        attemptSection.hidden = !view || (above && !view.text);
        if (attemptSection.hidden) return;
        // One quiet line of the record: when, and how it ended with its tone; the rest opens under it.
        attemptHeadline.dataset.tone = view.tone;
        attemptHeadline.textContent = view.headline;
        attemptHeadline.hidden = above || view.headline === verdict.headline;
        const when = attemptTime(view.when);
        attemptWhen.textContent = when ? `Last attempt · ${when}` : 'Last attempt';
        attemptWhen.dateTime = view.when;
        attemptText.textContent = view.text;
        attemptText.hidden = !view.text;
        // A confirmed success's words are the record's (in its Details); any other outcome's stay in view under the line.
        if (view.tone === 'ok') attemptFacts.before(attemptText); else if (attemptText.parentElement !== attemptSection) attemptSection.append(attemptText);
        // The retained samples (after_update) join the rows only of the attempt they name, so they appear with this
        // section, which stays hidden while an update runs (owner 25.10: A). A record of another operation is never
        // read as this attempt's: it keeps only its exact text below, labelled so.
        const observed = latestStatus?.after_update || null;
        const own = Boolean(observed) && String(observed.operation_id || '') === view.id;
        attemptFacts.innerHTML = [...view.details, ...(own ? afterUpdateRows(observed) : [])].map(factRowHtml).join('');
        attemptRaw.hidden = !observed;
        attemptRawLabel.textContent = own ? 'Raw record' : 'Raw record (not matched to this attempt)';
        // Rewritten only when the record changes, so a refresh keeps the owner's selection in it.
        const raw = observed ? JSON.stringify(observed, null, 2) : '';
        if (attemptRawText.textContent !== raw) attemptRawText.textContent = raw;
        attemptRestore.hidden = !view.restoreTarget || Boolean(latestStatus?.managed_by_deployment);
        attemptRestore.dataset.target = view.restoreTarget || '';
        attemptRestore.disabled = replaceInFlight || Boolean(latestStatus?.update_tx?.active)
            || ['preflighting', 'updating', 'restarting'].includes(phase);
    }

    function render() {
        const verdict = updateVerdict(latestStatus || {}, phase, lastAppVersion());
        // A managed deployment's administrator replaces the serving program (deployment_policy): no update, replace,
        // restore or recovery point is offered here; a restart of the same program stays.
        const managed = String(latestStatus?.managed_by_deployment || '');
        if (managed) Object.assign(verdict, { headline: 'Managed by your organization', hint: managed,
            action: verdict.action?.id === 'restart' ? verdict.action : null });
        page.querySelector('.updates-recovery').hidden = Boolean(managed);
        paintStatusMark(dot, { shape: TONE_MARKS[verdict.tone] ? 'dot' : 'none', hue: TONE_MARKS[verdict.tone] || 'ink' });
        summary.textContent = verdict.headline;
        const hintText = [verdict.hint, ...(verdict.warnings || []).map((w) => `Warning: ${w}`)]
            .filter(Boolean).join(' · ');
        hint.textContent = hintText;
        hint.hidden = !hintText;
        const chips = verdict.chips || [];
        meta.innerHTML = chips.filter((chip) => HEADLINE_FACTS.has(chip.label)).map(factLine).join(' · ');
        const rows = [...chips.filter((chip) => !HEADLINE_FACTS.has(chip.label)),
            ...(verdict.codes || []).map((code) => ({ label: 'Reported', value: code }))];
        facts.innerHTML = rows.map(factRowHtml).join('');
        factsMore.hidden = !rows.length;
        actionNote.textContent = verdict.checkedAgo || '';
        actionNote.hidden = !verdict.checkedAgo;
        if (verdict.action) {
            // The crimson is this place's own action, an update; a check or a restart is a neutral capsule.
            primaryBtn.className = `btn ${verdict.action.id === 'update' ? 'btn-primary' : 'btn-default'}`;
            primaryBtn.hidden = false;
            primaryBtn.disabled = Boolean(verdict.action.disabled);
            primaryBtn.textContent = verdict.action.label;
            primaryBtn.dataset.action = verdict.action.id;
        } else {
            primaryBtn.hidden = true;
            primaryBtn.dataset.action = '';
        }
        // Replace gating fails CLOSED: a status read that failed (synthesized
        // status_error) proves nothing about the durable transaction, so the
        // recovery action stays disabled until a successful re-read.
        replaceBtn.disabled = replaceInFlight || statusReadFailed(latestStatus) || [
            'loading', 'checking', 'updating', 'preflighting', 'restarting',
            'restart_required', 'restart_needed', 'resolving', 'unmanaged',
        ].includes(verdict.state);
        renderAttempt(verdict);
        renderLetter();
    }

    function setPhase(next) {
        phase = next;
        render();
    }

    function enterRestarting() {
        restartRefresh.beginRestarting();
        setPhase('restarting');
    }

    async function reconcileRestartStatus({ afterBootNotice = false } = {}) {
        if (phase !== 'restarting') return;
        if (restartReconcileInFlight) {
            restartReconcileQueued = true;
            restartReconcileAfterBootNotice ||= afterBootNotice;
            return;
        }
        restartReconcileInFlight = true;
        let currentAfterBootNotice = afterBootNotice;
        try {
            do {
                restartReconcileQueued = false;
                restartReconcileAfterBootNotice = false;
                let data;
                try {
                    data = await apiClient.updateStatus();
                } catch {
                    // A failed read proves nothing. Stay in the honest
                    // restarting state and let the next reconnect/ready event
                    // provide another bounded chance to reconcile.
                    currentAfterBootNotice = restartReconcileAfterBootNotice;
                    continue;
                }
                if (phase !== 'restarting') return;
                latestStatus = data;
                if (restartStatusCanSettle(data, { afterBootNotice: currentAfterBootNotice })) {
                    restartNeeded = false;
                    setPhase('');
                    return;
                }
                render();
                currentAfterBootNotice = restartReconcileAfterBootNotice;
            } while (restartReconcileQueued && phase === 'restarting');
        } finally {
            restartReconcileInFlight = false;
        }
    }

    let statusRefreshPromise = null;
    let statusRefreshNext = null;
    function loadStatus({ fetchRemote = false, preservePhase = !fetchRemote && ['preflighting', 'updating'].includes(phase) } = {}) {
        statusRefreshNext = {
            fetchRemote: fetchRemote || Boolean(statusRefreshNext?.fetchRemote),
            preservePhase: preservePhase && (statusRefreshNext?.preservePhase ?? true),
        };
        if (statusRefreshPromise) return statusRefreshPromise;
        statusRefreshPromise = (async () => {
            while (statusRefreshNext) {
                const options = statusRefreshNext;
                statusRefreshNext = null;
                if (!options.preservePhase) setPhase(options.fetchRemote ? 'checking' : 'loading');
                try {
                    const data = await (options.fetchRemote ? apiClient.updateCheck() : apiClient.updateStatus());
                    if (options.fetchRemote) letterRefreshError = null;
                    // The explicit check owns release discovery. A queued progress
                    // refresh has no tags and must not erase that successful read.
                    if (options.fetchRemote || officialTagsDiv.childElementCount === 0) {
                        versions.renderReleases(data.official_tags || []);
                    }
                    if (statusRefreshNext) continue; // A newer notice owns the next read.
                    latestStatus = options.preservePhase && phase === 'preflighting' && latestStatus?.preflight_range
                        ? bindPreflightDescription(data, { merge_plan: latestStatus.preflight_range, letter: latestStatus.letter }) : data;
                    // Keep an outstanding apply/restart await. Server progress refines
                    // its label; a fresh page starts with no old local phase.
                    if (!options.preservePhase || !['updating', 'restarting', 'preflighting'].includes(phase)) {
                        setPhase(restartNeeded && !data?.update_tx?.active ? 'restart_needed' : '');
                    } else render();
                } catch (err) {
                    if (statusRefreshNext) continue;
                    if (options.fetchRemote) letterRefreshError = {
                        at: new Date().toISOString(), text: String(err.message || err),
                    };
                    latestStatus = { ...latestStatus, managed: true, warnings: [`status_error:${err.message || err}`], check_ok: false };
                    if (!options.preservePhase) setPhase(restartNeeded ? 'restart_needed' : '');
                    else render();
                }
            }
        })().finally(() => { statusRefreshPromise = null; });
        return statusRefreshPromise;
    }

    // The Recovery list of earlier versions and official releases (update_versions.js); a Restore press confirms here.
    const versions = mountVersionList({ commits: commitsDiv, more: commitsMore, releases: officialTagsDiv,
        branch: current, onRestore: rollback });

    // The owner decides by a version's name and date; the exact commit is the confirmation's Details.
    async function rollback({ target, label = target, when = '' }) {
        const confirmed = await openConfirmDialog({
            title: 'Restore an earlier version',
            body: `Go back to ${label}${when ? ` from ${when}` : ''}?\n\nRunning work will be interrupted, a rescue snapshot will be saved, and the server will restart. Nothing will be pushed. Engine data cannot be downgraded.`,
            details: { summary: 'Details', rows: [{ label: 'Exact target', value: target }] },
            confirmLabel: 'Restore',
            danger: true,
        });
        if (!confirmed) return;
        try {
            const resp = await apiFetch('/api/git/rollback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target }),
            });
            const data = await resp.json();
            if (data.status === 'ok') {
                showToast(`Rollback ${data.switch_deferred ? 'requested' : 'successful'}: ${data.message}. Server is restarting...`, 'success');
                enterRestarting();
            } else if (data.status === 'restart_required') {
                // A ROLLBACK, not an update: the update-specific restart_required headline
                // would lie here. A store-less target moves only at the restart exit.
                showToast(`Rollback ${data.switch_deferred ? 'recorded' : 'completed'}: ${data.message}. Restart Ouroboros to finish.`, 'error');
                restartNeeded = true;
                setPhase('restart_needed');
            } else {
                const suffix = data.restart_required
                    ? ' Runtime shutdown was incomplete; restart Ouroboros before retrying.'
                    : '';
                showToast(`Rollback failed: ${data.error || 'unknown error'}${suffix}`, 'error');
                if (data.restart_required) { restartNeeded = true; setPhase('restart_needed'); }
            }
        } catch (err) {
            showToast(`Rollback failed: ${failureWords(err)}`, 'error');
        }
    }

    async function applyUpdate() {
        if (!latestStatus?.available) return;
        setPhase('preflighting');
        try {
            const preflight = await apiClient.updatePreflight();
            const verified = verifiedUpdatePlan(preflight);
            if (!verified) {
                throw new Error(preflight?.merge_plan?.error || 'The update plan could not be verified. No files were changed.');
            }
            const { plan, strategy } = verified;
            latestStatus = bindPreflightDescription(latestStatus, preflight);
            render();
            const conflictCount = (plan.code_conflict_paths || []).length + (plan.doc_conflict_paths || []).length;
            const assisted = strategy === 'assisted';
            const description = preflightDescription(preflight);
            // The description reads as rendered prose in its own region; the consequences of
            // applying stay plain text below it, and the per-commit evidence sits in Details.
            const proceed = await openConfirmDialog({
                title: `Update ${String(plan.base_sha).slice(0, 8)} -> ${String(plan.target_sha).slice(0, 8)}`,
                content: description ? { label: "What's new", mount: (el) => mountDescription(el, description) } : null,
                body: (description ? '' : 'No description is available for this exact preflight range.\n\n') + (assisted
                    ? `${conflictCount} conflict(s) need resolution. Ouroboros will resolve the merge as a reviewed task (model spend applies); progress lands in chat, and the server restarts after the reviewed commit.`
                    : `Git applies this update directly (${plan.local_dirty_count || 0} uncommitted file(s) are stashed and restored). The server restarts when it lands.`)
                    + '\n\n' + updateEffectsText(preflight),
                details: { summary: 'Checks for this range', rows: preflightEvidenceRows(preflight) },
                confirmLabel: assisted ? 'Start reviewed resolution' : 'Update now',
            });
            if (!proceed) {
                setPhase('');
                // The pin served the confirmation only: the panel reads the record as it is now,
                // a description saved while the dialog was open included.
                await loadStatus({ fetchRemote: false, preservePhase: true });
                return;
            }
            setPhase('updating');
            const data = await apiClient.updateApply(strategy, plan);
            if (data.status === 'assisted_started') {
                // Said of the admission, which a later status read may already have outlived.
                showToast('Ouroboros started a reviewed resolution of the update merge. Watch progress in chat.', 'success');
                await loadStatus({ fetchRemote: false, preservePhase: true });
                setPhase('');
            } else if (data.status === 'restart_required') {
                showToast('Update landed, but automatic restart failed. Restart Ouroboros to finish.', 'error');
                setPhase('restart_required');
            } else {
                showToast('Restart requested to finish the update.', 'success');
                enterRestarting();
            }
        } catch (err) {
            showToast('Update request did not complete: ' + applyFailureText(err), 'error');
            // An error carrying restart_required does NOT mean the update
            // landed (a failed writer fence or a failed rollback also sets
            // it): re-read the durable transaction state, and when no marker
            // survived (writer-fence refusals leave none) keep an honest
            // restart continuation instead of restoring the ordinary action.
            if (err?.body?.restart_required) restartNeeded = true;
            await loadStatus({ preservePhase: false });
        }
    }

    async function replaceWithOfficial() {
        // In-flight latch, not a bare .disabled: a tab reopen re-renders the
        // panel mid-request, and render() would otherwise re-enable Replace
        // while this destructive recovery is still pending (final-review
        // finding, round 3).
        replaceInFlight = true;
        render();
        try {
            const preflight = await apiClient.updatePreflight();
            const plan = preflight?.merge_plan || {};
            if (!plan.base_sha || !plan.target_sha) {
                throw new Error(plan.error || 'Could not resolve an exact recovery target.');
            }
            const proceed = await openConfirmDialog({
                title: 'Replace with official version',
                body: 'Recovery will reset the active checkout to the exact official version. A rescue snapshot and local keep branch preserve your work.\n\n'
                    + updateEffectsText(preflight, { replace: true }),
                confirmLabel: 'Replace checkout', danger: true,
            });
            if (!proceed) return;
            const data = await apiClient.updateApply('replace', plan, { confirmRecovery: true });
            if (data.status === 'restart_required') {
                showToast('Recovery needs a restart that could not be requested. Restart Ouroboros to finish.', 'error');
                setPhase('restart_required');
            } else {
                // Neither outcome claims the version landed: a store-less target is checked out only
                // at the restart exit, and boot smoke may still roll it back.
                showToast('Restart requested to finish recovery with the official version.', 'success');
                enterRestarting();
            }
        } catch (err) {
            const restartRequired = Boolean(err?.body?.restart_required);
            const suffix = restartRequired ? ' Runtime shutdown was incomplete; restart Ouroboros before retrying.' : '';
            showToast(`Recovery failed: ${failureWords(err)}${suffix}`, 'error');
            if (restartRequired) restartNeeded = true;
            // Fail-closed: ANY replace failure (the tx-active 409 included)
            // re-reads durable state, and render() alone owns the Replace
            // gate — the catch never re-enables it over stale/unknown state.
            await loadStatus({ preservePhase: false });
        } finally {
            replaceInFlight = false;
            render();
        }
    }

    async function restartNow() {
        primaryBtn.disabled = true;
        try {
            const resp = await apiFetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cmd: '/restart', ui_control: { name: 'restart' } }),
            });
            if (!resp.ok) throw new Error(`restart command refused (HTTP ${resp.status})`);
            showToast('Restart requested.', 'success');
            enterRestarting();
        } catch (err) {
            showToast(`Restart failed: ${failureWords(err)}`, 'error');
            primaryBtn.disabled = false;
        }
    }

    primaryBtn.addEventListener('click', () => {
        const action = primaryBtn.dataset.action;
        if (action === 'check') {
            loadStatus({ fetchRemote: true });
            versions.load();
        } else if (action === 'update') {
            applyUpdate();
        } else if (action === 'restart') {
            restartNow();
        }
    });
    replaceBtn.addEventListener('click', replaceWithOfficial);
    attemptRestore.addEventListener('click', () => rollback({ target: attemptRestore.dataset.target, label: 'the version before this update' }));
    letterMore.addEventListener('click', () => {
        letterMore.textContent = letterBody.classList.toggle('is-open') ? 'Show less' : 'Read all';
    });
    page.querySelector('#updates-promote').addEventListener('click', async () => {
        const confirmedPromote = await openConfirmDialog({
            title: 'Save recovery point',
            body: 'Move this installation\'s local recovery branch (ouroboros-stable) to the current checkout now?\n\nThis is the fallback the runtime boots when the working branch breaks. Each successful start of a clean, launcher-managed working branch moves it, so a point saved here lasts until then (earlier versions stay under Restore). It does not publish anything and does not change the official QA feed of any install.',
            confirmLabel: 'Save recovery point',
        });
        if (!confirmedPromote) return;
        try {
            const resp = await apiFetch('/api/git/promote', { method: 'POST' });
            const data = await resp.json();
            if (data.status === 'ok') {
                showToast(data.message, 'success');
            } else {
                showToast(`Could not save the recovery point${data.error ? `: ${data.error}` : '.'}`, 'error');
            }
        } catch (err) {
            showToast(`Could not save the recovery point: ${failureWords(err)}`, 'error');
        }
    });

    // The panel is mounted once for the whole app lifetime (app.js), so this
    // binding deliberately lives for that same installation lifetime.
    const restartRefresh = bindUpdateRefreshEvents({
        ws,
        getPhase: () => phase,
        reconcileRestart: reconcileRestartStatus,
        loadStatus,
        getRequestTaskId: () => requestView(latestStatus?.letter)?.taskId || '',
    });
    bindAppVersionRender({ getStatus: () => latestStatus, render });  // same lifetime as the panel
    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event.detail?.tab !== 'updates' || state.activePage !== 'dashboard') return;
        requestAnimationFrame(measureLetter);
        loadStatus({ fetchRemote: false });
        versions.load();
    });
    // The sidebar's return to the Dashboard announces only the page (app.js showPage); its selected tab stays.
    window.addEventListener('ouro:page-shown', (event) => {
        if (event.detail?.page === 'dashboard') requestAnimationFrame(measureLetter);
    });
    // A narrower window or interface size folds more of a text that fitted; an open text keeps its Show less.
    window.addEventListener('resize', () => {
        if (letterBody.classList.contains('is-open')) return;
        letterMeasureOwed = true;
        requestAnimationFrame(measureLetter);
    });
}
