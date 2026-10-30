import assert from 'node:assert/strict';
import test from 'node:test';

import { updateVerdict } from '../modules/updates.js';


test('a failed or never-run status read never fabricates "up to date" (R8)', () => {
    // status_error fall-through: managed, no warnings the state machine knows,
    // check_ok null, no checked_at — the old screen said "up to date" here.
    const verdict = updateVerdict({
        managed: true,
        warnings: ['status_error:boom'],
        check_ok: null,
        available: false,
        current_version: '6.113.4',
        current_short_sha: 'baba1acb',
    }, '');
    assert.equal(verdict.state, 'unknown');
    assert.notEqual(verdict.state, 'current');
    assert.match(verdict.hint, /status_error:boom/);
    assert.equal(verdict.action.id, 'check');
});


test('up to date is claimed only over a real check result, with its age', () => {
    const fresh = updateVerdict({ managed: true, check_ok: true, available: false, current_version: '1', current_short_sha: 'aaaa' }, '');
    assert.equal(fresh.state, 'current');
    const cached = updateVerdict({
        managed: true, check_ok: null, available: false,
        warnings: ['official_status_requires_check'],
        checked_at: new Date(Date.now() - 3 * 3600 * 1000).toISOString(),
        current_version: '1', current_short_sha: 'aaaa',
    }, '');
    assert.equal(cached.state, 'current');
    assert.equal(cached.checkedAgo, 'checked 3 h ago');
    const never = updateVerdict({
        managed: true, check_ok: null, available: false,
        warnings: ['official_status_requires_check'],
    }, '');
    assert.equal(never.state, 'unchecked');
});


test('restarting and restart_required stay distinct states with distinct actions', () => {
    const restarting = updateVerdict({}, 'restarting');
    assert.equal(restarting.state, 'restarting');
    assert.equal(restarting.action.disabled, true);
    assert.match(restarting.hint, /page updates itself/);
    assert.doesNotMatch(restarting.hint, /reload/i);
    const required = updateVerdict({}, 'restart_required');
    assert.equal(required.state, 'restart_required');
    assert.equal(required.action.id, 'restart');
    assert.equal(required.action.disabled, undefined);
    assert.equal(required.action.label, 'Restart now');
});


test('an unsafe update keeps ONE button plus a warning, never a second control', () => {
    const verdict = updateVerdict({
        managed: true, check_ok: true, available: true, safe_to_apply: false,
        ahead: 2, dirty: true, dirty_count: 1, local_changes: 'present', local_changes_base: 'c0ffee0123456789',
        current_version: '1', latest_version: '2',
        current_short_sha: 'aaaa', latest_short_sha: 'bbbb',
    }, '');
    assert.equal(verdict.state, 'available_unsafe');
    assert.equal(verdict.tone, 'warn');
    assert.equal(verdict.action.id, 'update');
    assert.match(verdict.action.label, /Update to 2/);
    const chip = (label) => verdict.chips.find((item) => item.label === label)?.value;
    assert.equal(chip('Commits'), '2 local');
    assert.equal(chip('Local changes'), 'present since c0ffee01');
    assert.equal(chip('Uncommitted'), '1 file(s)');
    // The in-place path (a status without `candidate`) stashes; a conflict keeps that stash, so no restore is promised.
    assert.match(verdict.hint, /changed since its last shared commit .* applying merges those changes\. Uncommitted changes go into a Git stash/);
    assert.match(verdict.hint, /if they conflict, they stay in that stash/);
    assert.doesNotMatch(verdict.hint, /stashed and restored/);
});


test('commits ahead with no content of their own are history, not a caution (#1718)', () => {
    const facts = { managed: true, check_ok: true, available: true, behind: 1, ahead: 45, dirty: false,
        current_version: '1', latest_version: '2', current_short_sha: 'aaaa', latest_short_sha: 'bbbb' };
    const merges = updateVerdict({ ...facts, safe_to_apply: true, local_changes: 'absent' }, '');
    assert.equal(merges.state, 'available');
    assert.equal(merges.tone, 'ok');
    assert.equal(merges.hint, '');
    assert.equal(merges.chips.find((chip) => chip.label === 'Commits').value, '1 incoming / 45 local');
    assert.equal(merges.chips.find((chip) => chip.label === 'Local changes').value, 'none');
    const unknown = updateVerdict({ ...facts, safe_to_apply: false, local_changes: 'unknown' }, '');
    assert.equal(unknown.state, 'available_unsafe');
    assert.match(unknown.hint, /^Could not compare this checkout with the official line/);
    assert.equal(unknown.chips.find((chip) => chip.label === 'Local changes').value, 'unknown');
    const dirtyOnly = updateVerdict({ ...facts, ahead: 0, safe_to_apply: false, local_changes: 'absent',
        dirty: true, dirty_count: 3 }, '');
    assert.equal(dirtyOnly.hint, 'Uncommitted changes go into a Git stash for the update and are applied again after the '
        + 'restart (staged ones come back unstaged); if they conflict, they stay in that stash.');
    assert.equal(dirtyOnly.chips.find((chip) => chip.label === 'Commits').value, '1 incoming');
});


test('an active assisted resolution is visible on re-entry and blocks the button', () => {
    const verdict = updateVerdict({
        managed: true, check_ok: true, available: true,
        update_tx: { active: true, phase: 'assisted_running', task_id: 'task-7' },
    }, '');
    assert.equal(verdict.state, 'resolving');
    assert.equal(verdict.action, null);
    assert.match(verdict.hint, /task-7/);
});


test('a failed check keeps the same actionable button label, not a dead state label', () => {
    const verdict = updateVerdict({ managed: true, check_ok: false, warnings: ['fetch_error:down'] }, '');
    assert.equal(verdict.state, 'check_failed');
    assert.equal(verdict.action.label, 'Check for updates');
    assert.match(verdict.hint, /fetch_error:down/);
});


test('unknown backend warning classes surface verbatim instead of vanishing (R7)', () => {
    const verdict = updateVerdict({
        managed: true, check_ok: true, available: false,
        warnings: ['target_ref_error:no shared stable release'],
    }, '');
    assert.deepEqual(verdict.warnings, ['target_ref_error:no shared stable release']);
});


test('non-assisted and corrupt update transactions are named honestly, never "under review"', () => {
    const rollback = updateVerdict({ managed: true, update_tx: { active: true, phase: 'rolling_back' } }, '');
    assert.equal(rollback.state, 'resolving');
    assert.match(rollback.headline, /transaction is still active/);
    assert.match(rollback.hint, /Phase: rolling_back/);
    const corrupt = updateVerdict({ managed: true, update_tx: { active: true, phase: 'corrupt' } }, '');
    assert.equal(corrupt.tone, 'error');
    assert.match(corrupt.headline, /corrupt/);
    const assisted = updateVerdict({ managed: true, update_tx: { active: true, phase: 'assisted_resolution', task_id: 't1' } }, '');
    assert.match(assisted.headline, /resolved under review/);
});

test('a corrupt marker names out-of-process boot recovery without offering the refused restart', () => {
    const corrupt = updateVerdict({ managed: true, update_tx: { active: true, phase: 'corrupt' } }, '');
    assert.equal(corrupt.action?.id, 'check');
    assert.equal(corrupt.action?.label, 'Check again');
    assert.match(corrupt.hint, /Quit and reopen Ouroboros/);
    assert.match(corrupt.hint, /in-app restart is deferred/i);
    assert.match(corrupt.hint, /If this state remains after reopening/);
    assert.match(corrupt.hint, /ouroboros-update-tx\.json/);
    assert.doesNotMatch(corrupt.hint, /restart will not clear/i);
    assert.doesNotMatch(corrupt.hint, /Replace with Official/);
});

test('a landed-but-unrestarted transaction keeps the Restart continuation across reloads', () => {
    const landed = updateVerdict({ managed: true, update_tx: { active: true, phase: 'pending_boot_smoke' } }, '');
    assert.match(landed.headline, /waiting for a restart/);
    assert.equal(landed.action?.id, 'restart');
});

test('a recovery the in-app Restart defers points at quitting and reopening, never at Restart', () => {
    // server_restart._safe_restart_serialized restarts only pending_boot_smoke and applying_replace.
    for (const txPhase of ['gate_blocked', 'marker_cleanup_retry']) {
        for (const phase of ['', 'restart_needed', 'restart_required']) {
            const view = updateVerdict({ managed: true, update_tx: { active: true, phase: txPhase } }, phase);
            assert.equal(view.tone, 'error');
            assert.doesNotMatch(view.headline, /update landed/i);
            assert.deepEqual(view.action, { id: 'check', label: 'Check again' });
            assert.match(view.hint, /Quit and reopen the desktop app, or restart the server process/);
            assert.match(view.hint, /in-app Restart is deferred/);
        }
    }
});

test('restart_needed is honest: no "update landed" claim, restart continuation kept', () => {
    const v = updateVerdict({ managed: true }, 'restart_needed');
    assert.doesNotMatch(v.headline, /update landed/i);
    assert.match(v.headline, /needs a restart/);
    assert.equal(v.action?.id, 'restart');
});

test('restart verdicts are true for an applied update and for a deferred store-less replace alike', () => {
    // A store-less replace moves its checkout only at the restart exit, so no restart verdict may
    // claim the version already landed; each keeps the Restart continuation.
    const phase = updateVerdict({ managed: true }, 'restart_required');
    const progress = updateVerdict({ managed: true, update_progress: {
        operation_id: 'op-1', active: false, result: 'restart_required', error: '' } }, '');
    const pending = updateVerdict({ managed: true, update_tx: { active: true, phase: 'pending_boot_smoke' } }, '');
    assert.equal(phase.headline, 'The update needs a restart that could not be requested automatically.');
    assert.equal(progress.headline, phase.headline);
    assert.equal(pending.headline, 'An update is waiting for a restart to finish.');
    for (const view of [phase, progress, pending]) {
        assert.doesNotMatch(view.headline, /landed|restored/i);
        assert.equal(view.action?.id, 'restart');
    }
});
