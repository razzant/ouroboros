import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { showToast } from './toast.js';
import { apiClient, apiFetch } from './api_client.js';
import { verifiedUpdatePlan } from './update_status.js';
import { destroyChatMarkdown, enhanceChatMarkdown, renderChatMarkdown } from './chat_markdown.js';

// Known non-state warnings are folded into their verdict states; everything
// else is surfaced verbatim so a backend warning class can never vanish.
const STATE_WARNINGS = new Set(['official_status_requires_check', 'managed_updates_unavailable']);

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

function statusReadFailed(data) {
    return Array.isArray(data?.warnings)
        && data.warnings.some((warning) => String(warning).startsWith('status_error:'));
}

// Verdict function: durable server state × transient client phase → one
// presentation descriptor (deterministic given status, phase, and the clock —
// humanizeCheckedAt reads Date.now for the "checked N ago" age). The button is always a real next action; facts
// travel as status line, hint, and chips (ARCHITECTURE §3: unavailable,
// divergent, dirty, unsafe, failed-check, rollback, and restart-required
// states stay visible — they are just not buttons).
export function updateVerdict(data = {}, phase = '') {
    const checkedAgo = humanizeCheckedAt(data.checked_at);
    const chips = [];
    if (data.official_repo_url) chips.push({ label: 'Official repo', value: repoSlug(data.official_repo_url) });
    if (data.target_ref) chips.push({ label: 'Target', value: data.target_ref });
    if (data.update_channel) chips.push({ label: 'Channel', value: data.update_channel, link: 'settings-advanced' });
    const divergence = [];
    if (data.behind) divergence.push(`${data.behind} incoming`);
    if (data.ahead) divergence.push(`${data.ahead} local`);
    if (data.dirty_count) divergence.push(`${data.dirty_count} dirty`);
    if (divergence.length) chips.push({ label: 'Divergence', value: divergence.join(' / ') });

    const warnings = extraWarnings(data);
    const base = { chips, warnings, checkedAgo };

    if (phase === 'loading') return { ...base, state: 'loading', tone: 'neutral', headline: 'Loading update status…', hint: '', action: null };
    if (phase === 'checking') return { ...base, state: 'checking', tone: 'neutral', headline: 'Checking the official channel…', hint: '', action: { id: 'check', label: 'Checking…', disabled: true } };
    if (phase === 'preflighting') return { ...base, state: 'preflighting', tone: 'neutral', headline: 'Preparing the update…', hint: '', action: { id: 'update', label: 'Preparing…', disabled: true } };
    if (phase === 'updating') return { ...base, state: 'updating', tone: 'neutral', headline: 'Applying the update…', hint: '', action: { id: 'update', label: 'Updating…', disabled: true } };
    if (phase === 'restarting') return { ...base, state: 'restarting', tone: 'ok', headline: 'Restarting the server…', hint: 'This page updates itself when the server is back.', action: { id: 'restart', label: 'Restarting…', disabled: true } };
    if (phase === 'restart_required') return { ...base, state: 'restart_required', tone: 'warn', headline: 'The update landed, but the automatic restart failed.', hint: 'Restart Ouroboros to finish.', action: { id: 'restart', label: 'Restart now' } };
    // Distinct from restart_required ("the update landed"): the operation did
    // NOT land — the runtime just cannot continue without a restart (failed
    // writer fence, failed rollback, rollback landed without its restart).
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
                state: 'resolving', tone: 'error',
                headline: 'The update transaction marker is corrupt.',
                hint: 'Quit and reopen Ouroboros so boot recovery can quarantine readable corruption when no merge is active. The in-app restart is deferred while this marker is corrupt. If this state remains after reopening, inspect the marker file (ouroboros-update-tx.json in the repository .git directory) manually, then check again.',
                action: { id: 'check', label: 'Check again' },
            };
        }
        if (txPhase === 'pending_boot_smoke') {
            return {
                ...base,
                state: 'resolving', tone: 'warn',
                headline: 'An update landed and is waiting for a restart to finish.',
                hint: 'Restart Ouroboros to run the post-update checks and complete it.',
                action: { id: 'restart', label: 'Restart now' },
            };
        }
        if (txPhase === 'gate_blocked' || txPhase === 'marker_cleanup_retry') {
            return {
                ...base,
                state: 'resolving', tone: 'error',
                headline: 'An update attempt stopped mid-flight and needs recovery.',
                hint: `Phase: ${txPhase}. Restart Ouroboros — boot recovery retries the rollback/cleanup.`,
                action: { id: 'restart', label: 'Restart now' },
            };
        }
        if (txPhase.includes('assisted')) {
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
    if (data.check_ok === false) {
        return {
            ...base,
            state: 'check_failed', tone: 'error',
            headline: 'Could not check the official update channel.',
            hint: warnings.length ? warnings.join(' · ') : 'Try again when the network is available.',
            warnings: [],
            action: { id: 'check', label: 'Check for updates' },
        };
    }
    const currentVersion = data.current_version || 'unknown';
    const currentSha = data.current_short_sha || '?';
    if (data.available) {
        const latestVersion = data.latest_version || 'unknown';
        const latestSha = data.latest_short_sha || '?';
        const unsafe = !data.safe_to_apply;
        if (data.latest_message) chips.push({ label: 'Latest', value: data.latest_message });
        return {
            ...base,
            state: unsafe ? 'available_unsafe' : 'available',
            tone: unsafe ? 'warn' : 'ok',
            headline: `Update available: ${currentVersion} (${currentSha}) -> ${latestVersion} (${latestSha})`,
            hint: unsafe ? 'Local commits or uncommitted changes diverge from the official line; applying merges them.' : '',
            action: { id: 'update', label: `Update to ${latestVersion}` },
        };
    }
    // "Up to date" is only claimed over an actual check result: a fresh
    // check_ok, or a cache-carried timestamp of the last real check.
    if (data.check_ok === true || data.checked_at) {
        return {
            ...base,
            state: 'current', tone: 'ok',
            headline: `Ouroboros is up to date at ${currentVersion} (${currentSha}).`,
            hint: '',
            action: { id: 'check', label: 'Check again' },
        };
    }
    return {
        ...base,
        state: 'unknown', tone: 'neutral',
        headline: 'Update status is unknown.',
        hint: warnings.length ? warnings.join(' · ') : 'Run a check to get the official status.',
        warnings: [],
        action: { id: 'check', label: 'Check for updates' },
    };
}

// --- The update letter -----------------------------------------------------
//
// Ouroboros writes ONE short markdown paragraph about the update the panel is
// offering, and the backend hands it to the client inside the ordinary status
// payload as the additive `letter` key. It is never deleted once the update
// lands: the same text, relabelled, becomes "What changed in this version".
// The letter is a fact, never an action — it adds no button and never touches
// the verdict.

// Hidden where the letter could only mislead: a checkout that managed updates
// do not own (unmanaged), a status the panel could not read (check_failed,
// unknown), a target no fetching check has named yet (unchecked), and the
// restart phase, whose served-SHA reload owns the card. Loading and checking
// KEEP it: the paragraph is still the last known fact, and a passive refresh
// or a running check must not blank it — that is what the letter renderer's
// content key is for.
const LETTER_HIDDEN_VERDICTS = new Set(['unmanaged', 'check_failed', 'unknown', 'unchecked']);
const LETTER_HIDDEN_PHASES = new Set(['restarting']);
const LETTER_RELATIONS = new Set(['pending', 'applied', 'superseded', 'other']);
// `applied` is the one relation whose label changes: the running version IS
// the letter's target, so the paragraph is history, not a preview.
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
    };
}

/**
 * Letter projector: durable server state × transient client phase → one
 * presentation descriptor, pure but for the clock `relativeAge` reads.
 *
 * `state: 'none'` means the section is hidden. A failed letter still renders
 * when the backend kept the previous good text for the same range (the
 * failure travels as a note line), and renders as the bare reason when it did
 * not — a letter that silently disappeared would look like an update with
 * nothing to say.
 */
export function updateLetterView(data = {}, phase = '') {
    const letter = data?.letter && typeof data.letter === 'object' ? data.letter : null;
    if (!letter) return noLetterView();
    if (LETTER_HIDDEN_PHASES.has(phase)) return noLetterView();
    if (LETTER_HIDDEN_VERDICTS.has(updateVerdict(data, phase).state)) return noLetterView();

    const state = letter.state === 'failed' ? 'failed' : (letter.state === 'ready' ? 'ready' : '');
    const markdown = String(letter.text || '').trim();
    // An unnamed state, or a "ready" letter with nothing in it, has no honest
    // rendering; the panel stays exactly as it was without one.
    if (!state || (state === 'ready' && !markdown)) return noLetterView();

    const relation = LETTER_RELATIONS.has(letter.relation) ? letter.relation : 'other';
    const meta = {
        authorVersion: String(letter.author_version || ''),
        targetVersion: String(letter.target_version || ''),
        writtenAt: String(letter.written_at || ''),
        ageText: relativeAge(letter.written_at),
    };
    const failure = state === 'failed'
        ? { kind: String(letter.error_kind || ''), text: String(letter.error_text || '') }
        : null;

    const notes = [];
    if (relation === 'applied' && meta.targetVersion && data.current_version
        && meta.targetVersion !== String(data.current_version)) {
        // The kept letter describes a version that this one includes but has moved past.
        notes.push(`written about ${meta.targetVersion}; the running version is ${data.current_version}`);
    }
    if (relation === 'superseded' || relation === 'other') {
        notes.push(meta.authorVersion && meta.targetVersion
            ? `written for ${meta.authorVersion} → ${meta.targetVersion}`
            : 'written for an earlier update');
    }
    if (failure) {
        const reason = failure.text || failure.kind || 'unknown reason';
        notes.push(markdown
            ? `rewriting this letter failed (${reason}); showing the last one that succeeded`
            : `Ouroboros could not write an update letter (${reason})`);
    }

    return {
        state,
        relation,
        markdown,
        meta,
        failure,
        label: LETTER_LABELS[relation],
        note: notes.join(' · '),
    };
}

/** Provenance sentence for the letter head: who wrote it, about what, when. */
function letterProvenance({ authorVersion, targetVersion, ageText }) {
    const who = authorVersion ? `written by Ouroboros ${authorVersion}` : 'written by Ouroboros';
    return [targetVersion ? `${who} about ${targetVersion}` : who, ageText]
        .filter(Boolean).join(' · ');
}

/** Content identity of a rendered letter: re-render only when this changes.
 *  The markdown itself, never a length proxy — two different paragraphs of equal length
 *  are a different letter, and one paragraph is small enough to compare outright. */
function letterContentKey(view) {
    return [view.state, view.relation, view.markdown].join('|');
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
export function bindUpdateRefreshEvents({ ws, getPhase, reconcileRestart, loadStatus }) {
    let restartReconnected = false;
    const disposers = [];
    const listen = (event, handler) => {
        const dispose = ws?.on?.(event, handler);
        if (typeof dispose === 'function') disposers.push(dispose);
    };

    listen('open', (event = {}) => {
        if (getPhase() !== 'restarting' || event.previouslyConnected !== true) return;
        restartReconnected = true;
        reconcileRestart({ afterBootNotice: false });
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

export function initUpdates({ mount, state, ws, openSettingsTab }) {
    const page = document.createElement('div');
    page.id = 'page-updates';
    page.className = 'settings-embedded-content';
    page.innerHTML = `
        <div class="updates-scroll">
            <section class="updates-card" id="updates-status-card">
                <div class="updates-card-title">Official Updates</div>
                <div class="updates-status">
                    <span class="updates-status-dot" id="updates-dot" data-tone="neutral"></span>
                    <span class="updates-headline" id="updates-summary">Loading update status...</span>
                </div>
                <div class="updates-hint" id="updates-hint" hidden></div>
                <div class="updates-meta" id="updates-meta"></div>
                <div class="settings-action-row updates-action-row">
                    <span class="updates-action-note" id="updates-action-note"></span>
                    <button class="btn btn-primary" id="btn-update-primary" hidden></button>
                </div>
                <section class="updates-letter" id="updates-letter" aria-labelledby="updates-letter-label" hidden>
                    <div class="updates-letter-head">
                        <h4 class="updates-letter-label" id="updates-letter-label"></h4>
                        <span class="updates-letter-meta" id="updates-letter-meta"></span>
                    </div>
                    <div class="updates-letter-note" id="updates-letter-note" hidden></div>
                    <div class="updates-letter-body ui-rich-content" id="updates-letter-body"></div>
                </section>
                <details class="updates-recovery">
                    <summary>Recovery</summary>
                    <p class="updates-recovery-copy">Replace the active checkout with the exact official version from the selected channel. A rescue copy is saved first, but this is intentionally more destructive than an ordinary update.</p>
                    <div class="updates-recovery-actions">
                        <button class="btn btn-danger btn-sm" id="btn-update-replace">Replace with Official Version (Recovery)</button>
                        <button class="btn btn-default btn-sm" id="updates-promote">Save recovery point</button>
                    </div>
                    <div class="updates-branch" id="updates-current"></div>
                    <h4 class="updates-subhead">Restore a previous version</h4>
                    <div id="updates-commits" class="updates-restore-list"></div>
                    <h4 class="updates-subhead">Official releases</h4>
                    <div id="updates-official-tags" class="updates-restore-list"></div>
                </details>
            </section>
        </div>
    `;
    mount.appendChild(page);

    const primaryBtn = page.querySelector('#btn-update-primary');
    let restartNeeded = false;  // panel-lifetime restart continuation (no durable marker exists for fence/rollback refusals)
    let replaceInFlight = false; // latch: a pending recovery request keeps Replace disabled across re-renders (tab reopen included)
    const replaceBtn = page.querySelector('#btn-update-replace');
    const dot = page.querySelector('#updates-dot');
    const summary = page.querySelector('#updates-summary');
    const hint = page.querySelector('#updates-hint');
    const meta = page.querySelector('#updates-meta');
    const actionNote = page.querySelector('#updates-action-note');
    const current = page.querySelector('#updates-current');
    const commitsDiv = page.querySelector('#updates-commits');
    const officialTagsDiv = page.querySelector('#updates-official-tags');
    const letterSection = page.querySelector('#updates-letter');
    const letterLabel = page.querySelector('#updates-letter-label');
    const letterMeta = page.querySelector('#updates-letter-meta');
    const letterNote = page.querySelector('#updates-letter-note');
    const letterBody = page.querySelector('#updates-letter-body');
    let letterDisposer = null;
    let letterKey = null;
    let latestStatus = null;
    let phase = 'loading';
    let restartReconcileInFlight = false;
    let restartReconcileQueued = false;
    let restartReconcileAfterBootNotice = false;

    function chipHtml(chip) {
        const body = `<strong>${escapeHtml(chip.label)}:</strong> ${escapeHtml(chip.value)}`;
        if (chip.link === 'settings-advanced') {
            return `<button type="button" class="updates-chip updates-chip-link" data-open-settings-advanced title="Change in Settings -> Advanced">${body}</button>`;
        }
        return `<span class="updates-chip">${body}</span>`;
    }

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

    /** The letter is a fact: no control the markdown pipeline adds may survive in it. */
    function stripLetterControls() {
        letterBody.querySelectorAll('button').forEach((control) => control.remove());
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
            }
            letterSection.hidden = true;
            return;
        }
        letterSection.hidden = false;
        letterLabel.textContent = view.label;
        letterMeta.textContent = letterProvenance(view.meta);
        letterNote.textContent = view.note;
        letterNote.hidden = !view.note;
        letterNote.className = view.state === 'failed'
            ? 'updates-letter-note updates-letter-note-failed'
            : 'updates-letter-note';
        const nextKey = letterContentKey(view);
        if (nextKey === letterKey) return;
        releaseLetterBody();
        letterKey = nextKey;
        letterBody.hidden = !view.markdown;
        letterBody.innerHTML = view.markdown ? renderChatMarkdown(view.markdown) : '';
        if (!view.markdown) return;
        // No anchored scroll to protect on this page, so markdown's deferred
        // writes (highlight, latex, mermaid, charts) run directly.
        // The card carries exactly ONE action and it is never inside the letter. The shared
        // markdown pipeline adds controls of its own to some blocks (a Copy button on a
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

    function render() {
        const verdict = updateVerdict(latestStatus || {}, phase);
        dot.dataset.tone = verdict.tone;
        summary.textContent = verdict.headline;
        const hintText = [verdict.hint, ...(verdict.warnings || []).map((w) => `Warning: ${w}`)]
            .filter(Boolean).join(' · ');
        hint.textContent = hintText;
        hint.hidden = !hintText;
        meta.innerHTML = (verdict.chips || []).map(chipHtml).join('');
        actionNote.textContent = verdict.checkedAgo || '';
        const actionRow = actionNote.parentElement;
        actionRow.hidden = !verdict.action && !verdict.checkedAgo;
        if (verdict.action) {
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

    async function loadStatus({ fetchRemote = false } = {}) {
        setPhase(fetchRemote ? 'checking' : 'loading');
        try {
            const data = await (fetchRemote ? apiClient.updateCheck() : apiClient.updateStatus());
            latestStatus = data;
            // A restart-required refusal (failed writer fence, failed rollback)
            // leaves NO durable marker, so the continuation lives in this
            // panel-lifetime flag: every refresh — tab reopen included —
            // re-applies it until the restart actually happens. A full page
            // reload honestly loses it (nothing durable exists server-side).
            setPhase(restartNeeded && !data?.update_tx?.active ? 'restart_needed' : '');
            renderOfficialTags(data.official_tags || []);
        } catch (err) {
            latestStatus = { managed: true, warnings: [`status_error:${err.message || err}`], check_ok: false };
            setPhase(restartNeeded ? 'restart_needed' : '');
        }
    }

    function renderRestoreRow({ label, date, message, target, restorable }) {
        const row = document.createElement('div');
        row.className = 'updates-restore-row';
        const when = (date || '').slice(0, 16).replace('T', ' ');
        row.innerHTML = `
            <span class="updates-restore-label">${escapeHtml(label)}</span>
            <span class="updates-restore-date">${escapeHtml(when)}</span>
            <span class="updates-restore-msg">${escapeHtml((message || '').slice(0, 96))}</span>
            ${restorable ? `<button class="btn btn-danger btn-xs" data-target="${escapeHtml(target)}">Restore</button>` : ''}
        `;
        if (restorable) row.querySelector('button').addEventListener('click', () => rollback(target));
        return row;
    }

    function renderOfficialTags(tags) {
        officialTagsDiv.innerHTML = '';
        (tags || []).forEach((tag) => {
            officialTagsDiv.appendChild(renderRestoreRow({
                label: tag.tag || '', date: '', message: (tag.sha || '').slice(0, 12),
                target: '', restorable: false,
            }));
        });
        if (!tags?.length) officialTagsDiv.innerHTML = '<div class="updates-empty">Run a check to load official releases.</div>';
    }

    async function loadVersions() {
        try {
            const resp = await apiFetch('/api/git/log', { cache: 'no-store' });
            if (!resp.ok) throw new Error('Git log API error ' + resp.status);
            const data = await resp.json();
            current.textContent = `Branch: ${data.branch || '?'} @ ${data.sha || '?'}`;
            // One restore list (owner decision 2026-08-31): tags that point at a
            // listed commit become labels on that commit's row; tags whose
            // target is older than the listed window keep their own row.
            const tagsBySha = new Map();
            (data.tags || []).forEach((tag) => {
                if (!tag.sha) return;
                const rows = tagsBySha.get(tag.sha) || [];
                rows.push(tag);
                tagsBySha.set(tag.sha, rows);
            });
            commitsDiv.innerHTML = '';
            const seenTagShas = new Set();
            (data.commits || []).forEach((commit) => {
                const tagged = tagsBySha.get(commit.sha) || [];
                tagged.forEach((tag) => seenTagShas.add(tag.sha));
                const tagNames = tagged.map((tag) => tag.tag).join(', ');
                commitsDiv.appendChild(renderRestoreRow({
                    label: tagNames || commit.short_sha || commit.sha?.slice(0, 8) || '?',
                    date: commit.date,
                    message: commit.message,
                    target: commit.sha,
                    restorable: true,
                }));
            });
            (data.tags || []).forEach((tag) => {
                if (tag.sha && seenTagShas.has(tag.sha)) return;
                commitsDiv.appendChild(renderRestoreRow({
                    label: tag.tag, date: tag.date, message: tag.message,
                    target: tag.tag, restorable: true,
                }));
            });
            if (!commitsDiv.children.length) commitsDiv.innerHTML = '<div class="updates-empty">No commits found</div>';
        } catch (err) {
            commitsDiv.innerHTML = `<div class="updates-empty updates-empty-error">Failed to load: ${escapeHtml(err.message || err)}</div>`;
            current.textContent = 'Branch: unknown';
        }
    }

    async function rollback(target) {
        const confirmed = await openConfirmDialog({
            title: 'Roll back',
            body: `Roll back to ${target}?\n\nA rescue snapshot of the current state will be saved. The server will restart.`,
            confirmLabel: 'Roll back',
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
                showToast(`Rollback successful: ${data.message}. Server is restarting...`, 'success');
                enterRestarting();
            } else if (data.status === 'restart_required') {
                // A ROLLBACK landed, not an update: the update-specific
                // restart_required headline would lie here.
                showToast(`Rollback completed: ${data.message}. Restart Ouroboros to finish.`, 'error');
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
            showToast('Rollback failed: ' + (err.message || err), 'error');
        }
    }

    function applyFailureText(err) {
        const body = err?.body || {};
        const parts = [String(err.message || err)];
        if (body.reason) parts.push(`reason: ${body.reason}`);
        if (Array.isArray(body.blockers) && body.blockers.length) {
            parts.push(`blocked by: ${body.blockers.slice(0, 5).join(', ')}`);
        }
        if (body.rolled_back) parts.push('the checkout was rolled back');
        if (body.smoke) parts.push('the post-update smoke check failed');
        if (typeof body.estimated_wave_usd === 'number') {
            parts.push(`assisted review needs ~$${body.estimated_wave_usd}` + (
                typeof body.remaining_usd === 'number' ? ` of $${body.remaining_usd} remaining` : ''
            ));
        }
        if (body.stash_note) parts.push(body.stash_note);
        return parts.join(' · ');
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
            const conflictCount = (plan.code_conflict_paths || []).length + (plan.doc_conflict_paths || []).length;
            const assisted = strategy === 'assisted';
            const proceed = await openConfirmDialog({
                title: `Update ${String(plan.base_sha).slice(0, 8)} -> ${String(plan.target_sha).slice(0, 8)}`,
                body: assisted
                    ? `${conflictCount} conflict(s) need resolution. Ouroboros will resolve the merge as a reviewed task (model spend applies); progress lands in chat, and the server restarts after the reviewed commit.`
                    : `Git applies this update directly (${plan.local_dirty_count || 0} local change(s) are stashed and restored). The server restarts when it lands.`,
                confirmLabel: assisted ? 'Start reviewed resolution' : 'Update now',
            });
            if (!proceed) {
                setPhase('');
                return;
            }
            setPhase('updating');
            const data = await apiClient.updateApply(strategy, plan);
            if (data.status === 'assisted_started') {
                showToast('Ouroboros is resolving the update merge under review. Watch progress in chat.', 'success');
                latestStatus = {
                    ...latestStatus,
                    update_tx: { active: true, phase: 'assisted_resolution', task_id: data.task_id || '' },
                };
                setPhase('');
            } else if (data.status === 'restart_required') {
                showToast('Update landed, but automatic restart failed. Restart Ouroboros to finish.', 'error');
                setPhase('restart_required');
            } else {
                showToast('Update applied. Server is restarting.', 'success');
                enterRestarting();
            }
        } catch (err) {
            showToast('Update failed: ' + applyFailureText(err), 'error');
            // An error carrying restart_required does NOT mean the update
            // landed (a failed writer fence or a failed rollback also sets
            // it): re-read the durable transaction state, and when no marker
            // survived (writer-fence refusals leave none) keep an honest
            // restart continuation instead of restoring the ordinary action.
            if (err?.body?.restart_required) restartNeeded = true;
            await loadStatus();
        }
    }

    async function replaceWithOfficial() {
        const proceed = await openConfirmDialog({
            title: 'Replace with official version',
            body: 'Recovery will replace the active checkout with the exact official version from the selected channel.\n\nA rescue snapshot and a local keep branch preserve a copy, but the active branch will be reset. Continue?',
            confirmLabel: 'Replace checkout',
            danger: true,
        });
        if (!proceed) return;
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
            const data = await apiClient.updateApply('replace', plan, { confirmRecovery: true });
            if (data.status === 'restart_required') {
                showToast('Recovery landed, but automatic restart failed. Restart Ouroboros to finish.', 'error');
                setPhase('restart_required');
            } else {
                showToast('Official version restored. Server is restarting.', 'success');
                enterRestarting();
            }
        } catch (err) {
            const restartRequired = Boolean(err?.body?.restart_required);
            const suffix = restartRequired ? ' Runtime shutdown was incomplete; restart Ouroboros before retrying.' : '';
            showToast('Recovery failed: ' + (err.message || err) + suffix, 'error');
            if (restartRequired) restartNeeded = true;
            // Fail-closed: ANY replace failure (the tx-active 409 included)
            // re-reads durable state, and render() alone owns the Replace
            // gate — the catch never re-enables it over stale/unknown state.
            await loadStatus();
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
                body: JSON.stringify({ cmd: '/restart' }),
            });
            if (!resp.ok) throw new Error(`restart command refused (HTTP ${resp.status})`);
            showToast('Restart requested.', 'success');
            enterRestarting();
        } catch (err) {
            showToast('Restart failed: ' + (err.message || err), 'error');
            primaryBtn.disabled = false;
        }
    }

    primaryBtn.addEventListener('click', () => {
        const action = primaryBtn.dataset.action;
        if (action === 'check') {
            loadStatus({ fetchRemote: true });
            loadVersions();
        } else if (action === 'update') {
            applyUpdate();
        } else if (action === 'restart') {
            restartNow();
        }
    });
    replaceBtn.addEventListener('click', replaceWithOfficial);
    meta.addEventListener('click', (event) => {
        if (event.target.closest?.('[data-open-settings-advanced]')) {
            openSettingsTab?.('advanced');
        }
    });
    page.querySelector('#updates-promote').addEventListener('click', async () => {
        const confirmedPromote = await openConfirmDialog({
            title: 'Save recovery point',
            body: 'Move this installation\'s local recovery branch (ouroboros-stable) to the current checkout?\n\nThis is the fallback the runtime boots when the working branch breaks. It does not publish anything and does not change the official QA feed of any install.',
            confirmLabel: 'Save recovery point',
        });
        if (!confirmedPromote) return;
        try {
            const resp = await apiFetch('/api/git/promote', { method: 'POST' });
            const data = await resp.json();
            if (data.status === 'ok') {
                showToast(data.message, 'success');
            } else {
                showToast('Error: ' + (data.error || 'unknown'), 'error');
            }
        } catch (err) {
            showToast('Failed: ' + (err.message || err), 'error');
        }
    });

    // The panel is mounted once for the whole app lifetime (app.js), so this
    // binding deliberately lives for that same installation lifetime.
    const restartRefresh = bindUpdateRefreshEvents({
        ws,
        getPhase: () => phase,
        reconcileRestart: reconcileRestartStatus,
        loadStatus,
    });
    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event.detail?.tab !== 'updates' || state.activePage !== 'dashboard') return;
        loadStatus({ fetchRemote: false });
        loadVersions();
    });
}
