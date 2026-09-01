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
        ahead: 2, dirty_count: 1,
        current_version: '1', latest_version: '2',
        current_short_sha: 'aaaa', latest_short_sha: 'bbbb',
    }, '');
    assert.equal(verdict.state, 'available_unsafe');
    assert.equal(verdict.tone, 'warn');
    assert.equal(verdict.action.id, 'update');
    assert.match(verdict.action.label, /Update to 2/);
    const divergence = verdict.chips.find((chip) => chip.label === 'Divergence');
    assert.match(divergence.value, /2 local \/ 1 dirty/);
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
    const blocked = updateVerdict({ managed: true, update_tx: { active: true, phase: 'gate_blocked' } }, '');
    assert.equal(blocked.tone, 'error');
    assert.doesNotMatch(blocked.headline, /update landed/i);
    assert.equal(blocked.action?.id, 'restart');
    const cleanup = updateVerdict({ managed: true, update_tx: { active: true, phase: 'marker_cleanup_retry' } }, '');
    assert.equal(cleanup.action?.id, 'restart');
});

test('restart_needed is honest: no "update landed" claim, restart continuation kept', () => {
    const v = updateVerdict({ managed: true }, 'restart_needed');
    assert.doesNotMatch(v.headline, /update landed/i);
    assert.match(v.headline, /needs a restart/);
    assert.equal(v.action?.id, 'restart');
});
