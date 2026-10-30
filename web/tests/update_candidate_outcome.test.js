// The prepared Update after its preparation: the work it carries, the last Restart to update as the panel's Last
// attempt, and the indicator that follows it (UPDATE-TECHNICAL-PLAN §5.6, §6.1, §6.7; DESIGN §4 Update indicator).
// Invented values: 7.8.2, the hashes and the paths. The shapes are supervisor/update_prepare.py's projection and its
// retained activation outcome (_retain_activation); the reader/writer test at the end reads those producers.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import { activationAttemptView, candidateDialog, candidateName } from '../modules/update_candidate.js';
import { candidateNotice, dismissUpdateAttempt, initUpdateStatus, updateNotice } from '../modules/update_status.js';
import { updateVerdict } from '../modules/updates.js';
import { installDomDouble } from './dom_double.js';

const SHA = { base: '88acfdbe3c1f0a2b4d5e6f708192a3b4c5d6e7f8', target: 'd2f6a91c7e5b4a3928170f6e5d4c3b2a19087f6e',
    merged: '5e7c0b13aa00bb11cc22dd33ee44ff5566778899', newer: 'e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4' };
const STATUS = {
    managed: true, check_ok: true, checked_at: new Date().toISOString(),
    current_version: '7.8.1', current_short_sha: '88acfdbe', current_sha: SHA.base,
    latest_version: '7.8.2', latest_short_sha: 'd2f6a91c', latest_sha: SHA.target,
    official_repo_url: 'https://github.com/razzant/ouroboros', target_ref: 'managed/ouroboros', update_channel: 'development',
    behind: 6, ahead: 0, dirty: false, dirty_count: 0, local_changes: 'absent', available: true, safe_to_apply: true,
    warnings: [], update_tx: { active: false }, update_progress: {},
};
const LOCAL = { ahead: 2, dirty: true, dirty_count: 3, local_changes: 'present', local_changes_base: '80a38cb9aa00', safe_to_apply: false };
const candidate = (fields) => ({ candidate_id: 'update_d2f6a91c7e5b_4f9a1c', reason: '', target_sha: SHA.target,
    target_ref: 'managed/ouroboros', target_version: '7.8.2', base_version: '7.8.1', channel: 'development',
    base_sha: SHA.base, local_changes: false, without_local_changes: false, merge_kind: 'fast_forward', local_commits: 0,
    carried: {}, conflict_paths: [], conflict_count: 0, layer_conflicts: {}, environment_same: true,
    checks: { ok: true, tests: { file_count: 14 } }, activation: {}, newer_available: false,
    updated_at: '2026-10-28T09:00:00+00:00', actions: [], ...fields });
const CARRIED = { index: { count: 1, paths: [['M', 'web/modules/interface_size.js']] },
    files: { count: 2, paths: [['A', 'notes/local-ideas.md'], ['M', 'supervisor/queue.py']] } };
// The retained outcome after an adopted update, as update_prepare._retain_activation writes it.
const settled = (fields) => ({ candidate_id: 'update_d2f6a91c7e5b_4f9a1c', at: '2026-10-28T09:05:00+00:00',
    outcome: 'adopted', sha: SHA.merged, previous: SHA.base, trigger: '', target_sha: SHA.target, target_version: '7.8.2',
    base_version: '7.8.1', merge_kind: 'clean', local_commits: 2, local_changes: true, without_local_changes: false,
    carried_files: 3, note: `Update ${SHA.merged.slice(0, 12)} is running on ouroboros with the local changes it carried; `
        + `previous commit ${SHA.base.slice(0, 12)}.`, ...fields });
const after = (last, fields = {}) => ({ ...STATUS, available: false, current_version: '7.8.2', current_short_sha: '5e7c0b13',
    current_sha: SHA.merged, behind: 0, ...fields, candidate: { state: 'none', actions: ['prepare'], last_activation: last } });
const row = (view, label) => view.details.find((item) => item.label === label)?.value;

test('a confirmed success stays the Last attempt: versions, local work, Restore, and the ordinary verdict above it', () => {
    const last = settled({});
    const view = activationAttemptView(after(last));
    assert.deepEqual([view.terminal, view.tone, view.headline, view.open],
        [true, 'ok', 'Updated 7.8.1 → 7.8.2 with your local changes.', false]);
    assert.equal(row(view, 'Previous version'), '7.8.1 (88acfdbe)');
    assert.equal(row(view, 'Local commits'), '2, kept in history');
    assert.equal(row(view, 'Uncommitted files'), '3, still uncommitted');
    assert.deepEqual([view.restoreTarget, view.restoreName], [SHA.base, '7.8.1 (88acfdbe)'], 'the existing Restore, by name');
    assert.equal(view.id, 'activation:update_d2f6a91c7e5b_4f9a1c:adopted');
    assert.deepEqual(JSON.parse(view.raw), last, 'the exact record, note and codes, stays one disclosure away');
    // The verdict above it is the ordinary one: up to date now, or the newer version, never masked by the record.
    assert.equal(updateVerdict(after(last)).headline, 'Ouroboros is up to date at 7.8.2 (5e7c0b13).');
    const newer = after(last, { available: true, behind: 1, latest_version: '7.8.3', latest_short_sha: 'e1f2a3b4', latest_sha: SHA.newer });
    assert.equal(updateVerdict(newer).headline, 'Update available: 7.8.2 (5e7c0b13) -> 7.8.3 (e1f2a3b4)');
    assert.equal(updateNotice(after(last)), null, 'owner 1B: a confirmed success never wears the indicator');
    assert.equal(updateNotice(newer), null, 'the pill then says only that a newer version is available');
});

test('a success names only the local work its record proves: counts it recorded, refs it kept, never a guessed zero', () => {
    const plain = activationAttemptView(after(settled({ local_changes: false, local_commits: 0, merge_kind: 'fast_forward',
        carried_files: 0 })));
    assert.equal(plain.headline, 'Updated 7.8.1 → 7.8.2.');
    assert.deepEqual(plain.details.map((item) => item.label), ['Previous version'], 'no local work: no rows about it');
    const unread = activationAttemptView(after(settled({ carried_files: null })));
    assert.equal(row(unread, 'Uncommitted files'), 'carried over, still uncommitted', 'an unrecorded count is not zero');
    const left = activationAttemptView(after(settled({ without_local_changes: true, carried_files: 0,
        left_out: 'refs/ouroboros/update/update_d2f6a91c7e5b_4f9a1c/k2' })));
    assert.equal(left.headline, 'Updated 7.8.1 → 7.8.2 without your uncommitted changes.');
    assert.equal(row(left, 'Left out'), 'your uncommitted changes; Git keeps them at refs/ouroboros/update/update_d2f6a91c7e5b_4f9a1c/k2');
    assert.equal(row(left, 'Uncommitted files'), undefined);
    const app = activationAttemptView(after(settled({ merge_kind: 'current', local_changes: false, local_commits: null,
        native: { version: '7.8.0', previous_version: '7.7.0', backup_app: '/Users/o/Desktop/.Ouroboros-7.7.0-backup.app' } })));
    assert.equal(app.headline, 'The desktop app was updated: 7.7.0 → 7.8.0.');
    assert.equal(row(app, 'Desktop app'), '7.7.0 → 7.8.0; the previous app stays at /Users/o/Desktop/.Ouroboros-7.7.0-backup.app');
});

test('a return and a failed start are worded from their codes; every code stays in the raw record', () => {
    const returned = settled({ outcome: 'returned', sha: SHA.base, previous: SHA.merged, why: 'supervisor_not_ready',
        trigger: 'return_requested', reasons: [], local_changes: false, local_commits: 0 });
    const back = activationAttemptView(after(returned));
    assert.deepEqual([back.tone, back.headline, back.open, back.restoreTarget],
        ['error', 'The update failed; the previous version was restored.', true, '']);
    assert.deepEqual(back.details.map((item) => [item.label, item.value]), [['Why', 'the supervisor did not become ready'],
        ['Did not start', '7.8.2 (d2f6a91c)'], ['Running again', '7.8.1 (88acfdbe)'], ['Your data', 'not rolled back']],
    'no local work was there: no line says it came back');
    assert.equal(row(activationAttemptView(after({ ...returned, local_changes: true })), 'Local changes'), 'back as they were');
    for (const code of ['supervisor_not_ready', 'return_requested']) assert.match(back.raw, new RegExp(code));
    const failed = settled({ outcome: 'start_failed', sha: SHA.merged, previous: SHA.base, why: 'supervisor_not_ready',
        reasons: ['successor_entered:server', 'successor_changed_data:dialogue_store', 'engine_floor_unproven',
            'environment_switched', 'a_future_code:detail'] });
    const kept = activationAttemptView(after(failed));
    assert.deepEqual([kept.headline, kept.restoreTarget, kept.restoreName],
        ['The update did not complete.', SHA.base, '7.8.1 (88acfdbe)']);
    assert.equal(row(kept, 'Why'), 'the supervisor did not become ready');
    assert.equal(row(kept, 'Not proven'), 'the new version had already started its server, so it may have changed data; '
        + 'the new version changed data the previous one reads first (dialogue_store); an engine whose data only moves '
        + 'forward ran here, so the previous version may not read it; the Python packages were switched to the new '
        + 'version’s own; a_future_code:detail', 'a code this panel does not know stays as written');
    assert.equal(row(kept, 'Your data and local changes'), 'kept, not rolled back');
    const words = kept.details.map((item) => `${item.label} ${item.value}`).join(' ');
    assert.doesNotMatch(words, /successor_entered|engine_floor_unproven|environment_switched|supervisor_not_ready/);
    assert.match(kept.raw, /engine_floor_unproven/);
    const odd = activationAttemptView(after(settled({ outcome: 'from_the_future' })));
    assert.deepEqual([odd.tone, odd.headline], ['neutral', 'The last update outcome is unconfirmed.'], 'unknown is never success');
    assert.match(odd.text, /from_the_future/);
    assert.equal(activationAttemptView({ ...STATUS, candidate: { state: 'none' } }), null);
});

test('a refused package switch returned before anything ran: ready again, the system’s own words, Restart to update', () => {
    // body_switch: the rename the system refused, recorded as it said it (never a guessed holder); everything went back.
    const detail = "[WinError 5] Access is denied: 'C:\\Ouroboros\\data\\state\\python-userbase' -> "
        + "'C:\\Ouroboros\\data\\state\\python-userbase.previous-a1'";
    const decision = { outcome: 'returned', trigger: 'environment_selection_failed', reasons: [], retry: true, detail };
    const last = settled({ outcome: 'returned', sha: SHA.base, previous: SHA.target, trigger: 'environment_selection_failed',
        why: 'environment_selection_failed', detail, local_changes: false, local_commits: 0, carried_files: null });
    const data = { ...STATUS, candidate: candidate({ state: 'ready', reason: 'activation:returned', last_activation: last,
        activation: { outcome: 'returned', why: 'environment_selection_failed', decision },
        actions: ['activate', 'discard_candidate'] }) };
    const verdict = updateVerdict(data);
    assert.deepEqual([verdict.state, verdict.tone, verdict.action?.id], ['candidate_ready', 'warn', 'activate']);
    assert.match(verdict.hint, /^The last restart did not switch: the system refused to switch the Python packages \(\[WinError 5\] Access is denied: /);
    assert.match(verdict.hint, /\)\. Nothing changed\. If a program is using that folder, close it; Restart to update tries again\./);
    const view = activationAttemptView(data);
    assert.deepEqual([row(view, 'Why'), row(view, 'The system said'), row(view, 'Your data')],
        ['the system refused to switch the Python packages', detail, 'not rolled back']);
    // The same return on a candidate that is not ready (an older record) keeps the returned verdict's words.
    const older = updateVerdict({ ...data, candidate: { ...data.candidate, state: 'failed', actions: ['prepare', 'discard_candidate'] } });
    assert.equal(older.state, 'candidate_returned');
    assert.match(older.hint, /^It switched back before the new version ran, so your data is as it was\./);
});

test('the indicator follows the prepared update: its stage, ready, waiting for the owner, then its outcome', () => {
    const notice = (fields, extra = {}) => candidateNotice({ ...STATUS, ...extra, candidate: candidate(fields) });
    for (const state of ['preparing', 'checking', 'adapting']) {
        assert.deepEqual([notice({ state }).pill, notice({ state }).terminal], ['Preparing update…', false], state);
    }
    assert.equal(candidateNotice({ ...STATUS, candidate: { state: 'preparing', actions: [] } }).pill, 'Preparing update…');
    assert.equal(notice({ state: 'activating', activation: { adoption_id: 'a1' } }).pill, 'Restart requested…');
    assert.equal(notice({ state: 'activating', activation: { outcome: 'returning' } }).pill, 'Restoring version…');
    assert.deepEqual([notice({ state: 'ready' }).pill, notice({ state: 'ready' }).terminal], ['Update ready', false]);
    assert.equal(notice({ state: 'ready', merge_kind: 'current', native: { status: 'staged', version: '7.8.0' } }).pill,
        'App update ready');
    for (const state of ['needs_adaptation', 'needs_decision', 'paused_budget', 'stale']) {
        assert.deepEqual([notice({ state }).pill, notice({ state }).tone, notice({ state }).terminal],
            ['Update needs you', 'warn', false], state);
    }
    const prep = notice({ state: 'failed', reason: 'checks_failed' });
    assert.deepEqual([prep.pill, prep.tone, prep.terminal, prep.id],
        ['Update not prepared', 'error', true, 'candidate:update_d2f6a91c7e5b_4f9a1c:failed']);
    const returned = notice({ state: 'failed', reason: 'activation:returned', activation: { outcome: 'returned' } },
        { candidate: undefined });
    assert.deepEqual([returned.pill, returned.tone, returned.terminal], ['Previous version restored', 'error', true]);
    const retained = candidateNotice({ ...STATUS, candidate: { state: 'none', last_activation: settled({ outcome: 'start_failed' }) } });
    assert.deepEqual([retained.pill, retained.id], ['Update failed', 'activation:update_d2f6a91c7e5b_4f9a1c:start_failed']);
    assert.equal(candidateNotice({ ...STATUS, candidate: { state: 'none', last_activation: settled({}) } }).pill, '',
        'a success has no word');
    assert.equal(candidateNotice({ ...STATUS, candidate: { state: 'none', actions: ['prepare'] } }), null);
});

test('one notice wins: a running stage, then the prepared update, then the newer of two concluded outcomes', () => {
    const legacyFailed = { operation_id: 'op-old', active: false, outcome: 'failed', outcome_at: '2026-10-27T10:00:00+00:00' };
    const running = { operation_id: 'op-run', active: true, stage: 'applying', current_generation: true };
    const ready = { ...STATUS, candidate: candidate({ state: 'ready' }) };
    assert.equal(updateNotice({ ...ready, update_progress: running }).pill, 'Applying update…', 'a running stage first');
    assert.equal(updateNotice({ ...ready, update_progress: legacyFailed }).pill, 'Update ready', 'a live candidate beats a record');
    const newerSuccess = { ...STATUS, update_progress: legacyFailed, candidate: { state: 'none', last_activation: settled({}) } };
    assert.equal(updateNotice(newerSuccess), null, 'a newer confirmed success ends an older failure notice');
    const olderReturn = { ...STATUS, update_progress: { ...legacyFailed, outcome_at: '2026-10-29T10:00:00+00:00' },
        candidate: { state: 'none', last_activation: settled({ outcome: 'returned' }) } };
    assert.deepEqual([updateNotice(olderReturn).pill, updateNotice(olderReturn).id], ['Update failed', 'op-old'],
        'the newer concluded outcome is the one shown');
});

const settle = async () => {
    for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

test('the pill on the page: ready has no close; a return closes for this browser; then the newer version shows', async () => {
    const env = installDomDouble();
    const stored = new Map();
    const prior = { localStorage: globalThis.localStorage, dispatch: globalThis.window.dispatchEvent };
    globalThis.localStorage = { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, String(value)) };
    globalThis.window.dispatchEvent = (event) => globalThis.window.emit(event.type);
    let controller;
    try {
        env.doc.body.append(env.el('span', { attrs: { id: 'nav-version' } }));
        const listeners = new Map();
        const ws = { on(name, fn) { listeners.set(name, fn); return () => listeners.delete(name); } };
        let status = { ...STATUS, candidate: candidate({ state: 'ready', actions: ['activate', 'discard_candidate'] }) };
        controller = initUpdateStatus({ ws, readStatus: async () => structuredClone(status) });
        await settle();
        const pill = () => env.doc.getElementById('update-pill');
        const close = () => env.doc.body.querySelector('.update-notice-close');
        assert.deepEqual([pill().hidden, pill().textContent, close().hidden], [false, 'Update ready', true]);
        assert.equal(pill().dataset.tone, undefined);
        status = { ...STATUS, candidate: candidate({ state: 'failed', reason: 'activation:returned', actions: ['prepare'],
            activation: { outcome: 'returned' }, last_activation: settled({ outcome: 'returned' }) }) };
        listeners.get('update_progress_changed')();
        await settle();
        assert.deepEqual([pill().textContent, pill().dataset.tone, close().hidden], ['Previous version restored', 'error', false]);
        close().click();
        assert.deepEqual([...stored.values()], ['activation:update_d2f6a91c7e5b_4f9a1c:returned'], 'this browser, that outcome');
        assert.deepEqual([pill().hidden, pill().textContent], [false, 'Update available'], 'what it was hiding shows');
        dismissUpdateAttempt('');
    } finally {
        controller?.dispose();
        globalThis.window.dispatchEvent = prior.dispatch;
        globalThis.localStorage = prior.localStorage;
        env.restore();
    }
});

test('ready lists each carried path once with its layers; a long list is bounded and a partial record says so', () => {
    const ready = (fields) => updateVerdict({ ...STATUS, ...LOCAL, candidate: candidate({ state: 'ready', merge_kind: 'clean',
        local_changes: true, local_commits: 2, actions: ['activate', 'discard_candidate'], ...fields }) });
    const facts = ready({ carried: CARRIED }).facts;
    const head = facts.findIndex((item) => item.head === 'Uncommitted after the restart');
    assert.deepEqual(facts.slice(head + 1).map((item) => [item.label, item.value]), [
        ['notes/local-ideas.md', 'new file'], ['supervisor/queue.py', 'edited'], ['web/modules/interface_size.js', 'staged']]);
    assert.equal(facts.find((item) => item.label === 'Uncommitted changes').value, '3 file(s) carried over, still uncommitted');
    const both = ready({ carried: { index: { count: 1, paths: [['A', 'x.py']] }, files: { count: 1, paths: [['M', 'x.py']] } } });
    assert.equal(both.facts.find((item) => item.label === 'x.py').value, 'new, staged, then edited', 'one path, both layers');
    const many = Array.from({ length: 15 }, (_v, index) => ['M', `pkg/file_${String(index).padStart(2, '0')}.py`]);
    const long = ready({ carried: { index: { count: 0, paths: [] }, files: { count: 15, paths: many } } }).facts;
    const listed = long.slice(long.findIndex((item) => item.head) + 1);
    assert.equal(listed.length, 11, 'ten rows, then one block');
    assert.deepEqual([listed[10].label, listed[10].code, listed[10].value.split('\n').length], ['5 more', true, 5]);
    // A layer recorded in part (update_prepare._CARRIED_LIMIT): twenty more paths, then the exact totals with the
    // complete list beside the copy as its download by the candidate's id (gateway update_activation.candidate_view).
    const notes = Array.from({ length: 200 }, (_v, index) => ['A', `notes/n${String(index).padStart(3, '0')}.md`]);
    const list = '/Users/o/Ouroboros/subagent_worktrees/copies.noindex/update_d2f6a91c7e5b_4f9a1c.uncommitted.txt';
    const url = '/api/files/download?carried_list=update_d2f6a91c7e5b_4f9a1c';
    const recorded = { index: { count: 0, paths: [] }, files: { count: 250, paths: notes }, list, list_url: url };
    const sourced = ready({ carried: recorded }).facts;
    const shown = sourced.slice(sourced.findIndex((item) => item.head) + 1);
    assert.equal(shown.length, 12, 'ten rows, one block, the totals');
    const [block, all] = shown.slice(10);
    assert.deepEqual([block.label, block.code, block.value.split('\n').length], ['More', true, 20],
        'twenty more paths: never the 190 others inline');
    assert.deepEqual([all.label, all.value, all.link], ['In all', '250 change(s): 0 staged, 250 in the files.',
        { href: url, name: 'update_d2f6a91c7e5b_4f9a1c.uncommitted.txt', text: 'Download the full list' }]);
    assert.equal(sourced.find((item) => item.label === 'Uncommitted changes').value,
        '250 change(s) carried over (0 staged, 250 in the files), still uncommitted', 'exact counts, no file guess');
    // Only the candidate's own list download is linked: never another address, never a Files path (a phone without a
    // Files root could not open it).
    for (const other of ['https://example.com/x', `/api/files/download?path=${list.slice(1)}`, '']) {
        assert.equal(ready({ carried: { ...recorded, list_url: other } }).facts.at(-1).link, undefined, other);
    }
    // A record without the list (made before it, or the list already gone): the totals, and no invented link.
    const partial = ready({ carried: { index: { count: 250, paths: many.slice(0, 2) }, files: { count: 0, paths: [] } } });
    assert.deepEqual([partial.facts.at(-1).value, partial.facts.at(-1).link],
        ['250 change(s): 250 staged, 0 in the files; the ones not shown are not recorded here.', undefined]);
    // A complete record the panel shows whole needs no list.
    assert.ok(!long.some((item) => item.link || item.label === 'In all'), 'fifteen shown: nothing more to download');
    const none = ready({ carried: {}, local_changes: true });
    assert.ok(!none.facts.some((item) => item.head === 'Uncommitted after the restart'), 'no layers recorded: no rows');
    const left = ready({ without_local_changes: true, carried: CARRIED });
    assert.ok(!left.facts.some((item) => item.head), 'work left out is not listed as carried');
});

// body_adoption.finalize_on_boot's observation of a switched tree this start did not attest, as record_activation
// keeps it on the live record (state activating, adoption id kept, never retained as a concluded outcome).
const UNCONFIRMED = { adoption_id: 'a1b2c3d4e5f6', outcome: 'unconfirmed', why: 'loaded_tree_unattested',
    observed_at: '2026-10-28T09:07:00+00:00', note: 'The checkout is at 5e7c0b13aa00, but this process did not verify '
        + 'that tree before its imports; the adoption stays unconfirmed until a start that does.' };

test('a start that did not attest its switch is shown as unconfirmed: neutral, its record one disclosure away', () => {
    const data = { ...STATUS, candidate: candidate({ state: 'activating', reason: 'activation:unconfirmed',
        activation: UNCONFIRMED }) };
    const verdict = updateVerdict(data);
    assert.deepEqual([verdict.state, verdict.tone, verdict.action], ['candidate_unconfirmed', 'neutral', null]);
    assert.equal(verdict.headline, '7.8.2 (d2f6a91c) is in place, but its start is unconfirmed.');
    assert.match(verdict.hint, /counted neither done nor failed; a start that verifies them settles it/);
    assert.doesNotMatch(`${verdict.headline} ${verdict.hint}`, /unattested|adoption|Restart requested|did not (start|complete)|restored/,
        'no code and no terminal claim');
    const view = activationAttemptView(data);
    assert.deepEqual([view.tone, view.headline, view.id, view.restoreTarget, view.open],
        ['neutral', 'The last update outcome is unconfirmed.', 'activation:update_d2f6a91c7e5b_4f9a1c:unconfirmed', '', true]);
    assert.equal(view.details.find((row) => row.label === 'Why').value,
        'this start did not verify the switched files before importing them');
    assert.match(view.raw, /loaded_tree_unattested/);
    assert.match(view.raw, /a1b2c3d4e5f6/, 'the adoption it observed stays named in the record');
    const notice = candidateNotice(data);
    assert.deepEqual([notice.pill, notice.tone, notice.terminal, notice.id],
        ['Update outcome unconfirmed', 'neutral', true, 'activation:update_d2f6a91c7e5b_4f9a1c:unconfirmed']);
    // Genuinely in flight and genuinely returning stay what they are.
    const inFlight = { ...STATUS, candidate: candidate({ state: 'activating', activation: { adoption_id: 'a1b2c3d4e5f6' } }) };
    assert.deepEqual([updateVerdict(inFlight).state, candidateNotice(inFlight).pill, candidateNotice(inFlight).terminal],
        ['candidate_activating', 'Restart requested…', false]);
    assert.equal(activationAttemptView(inFlight), null, 'nothing was observed yet');
    const returning = { ...STATUS, candidate: candidate({ state: 'activating', activation: { outcome: 'returning' } }) };
    assert.deepEqual([updateVerdict(returning).state, candidateNotice(returning).pill], ['candidate_returning',
        'Restoring version…']);
    assert.equal(activationAttemptView(returning), null);
});

test('the prepared version names the update after the channel moved on, and a stale status cannot rename it', () => {
    const moved = { ...STATUS, latest_version: '7.8.3', latest_sha: SHA.newer, candidate: candidate({ state: 'ready' }) };
    assert.deepEqual(candidateName(moved), { subject: '7.8.2 (d2f6a91c)', mention: '7.8.2 (d2f6a91c)' });
    assert.deepEqual(candidateName({ ...moved, candidate: candidate({ state: 'ready', target_version: '' }) }),
        { subject: 'Update d2f6a91c', mention: 'update d2f6a91c' }, 'an older record without it names the commit only');
    assert.equal(updateVerdict({ ...moved, candidate: candidate({ state: 'activating' }) }, 'restarting').headline,
        'Restarting into 7.8.2 (d2f6a91c)…');
});

test('Prepare summarizes the range and the local work from the status, quietly, before the first preparation only', () => {
    const plain = candidateDialog('prepare', { ...STATUS, candidate: { state: 'none', actions: ['prepare'] } });
    assert.deepEqual([plain.title, plain.confirmLabel, plain.details.summary], ['Prepare 7.8.2 (d2f6a91c)?', 'Prepare update',
        'Show details']);
    assert.match(plain.body, /^Ouroboros prepares 7\.8\.2 in a separate copy and checks it there\. 7\.8\.1 keeps running/);
    assert.deepEqual(plain.details.rows.map((item) => [item.label, item.value]), [
        ['Range', '88acfdbe → d2f6a91c, 6 commit(s)'], ['Local commits', 'none of their own'], ['Uncommitted files', 'none'],
        ['Conflicts', 'only what Git cannot combine goes to a reviewed task in the copy, which uses model budget'],
        ['Checks', 'selected tests run in the copy; Restart to update is offered only when they pass']]);
    const local = candidateDialog('prepare', { ...STATUS, ...LOCAL, candidate: { state: 'none', actions: ['prepare'] } });
    assert.match(local.body, /with your local work/);
    assert.match(local.body, /Uncommitted files stay uncommitted; nothing is committed for you\./);
    assert.equal(local.details.rows.find((item) => item.label === 'Local commits').value,
        'with changes of their own since 80a38cb9; merged with it');
    assert.equal(local.details.rows.find((item) => item.label === 'Uncommitted files').value, '3, carried over and kept uncommitted');
    const unknown = candidateDialog('prepare', { ...STATUS, local_changes: 'unknown', candidate: { state: 'none' } });
    assert.equal(unknown.details.rows.find((item) => item.label === 'Local commits').value,
        'could not be compared with the official line');
    for (const value of [...plain.details.rows, ...local.details.rows].map((item) => item.value)) {
        assert.doesNotMatch(value, /Python|Claudexor|engine/i, 'no runtime claim the status cannot back');
    }
    assert.equal(candidateDialog('prepare', { ...STATUS, candidate: candidate({ state: 'failed' }) }), null,
        'the same range again needs no second look');
    assert.equal(candidateDialog('prepare', { ...STATUS, available: false, candidate: { state: 'none',
        app_needed: { status: 'needed', reason: 'signed_entry_missing' } } }), null, 'an app alone: no source range');
    const newer = candidateDialog('prepare', { ...STATUS, latest_version: '7.8.3', latest_sha: SHA.newer,
        candidate: candidate({ state: 'ready', newer_available: true }) });
    assert.equal(newer.title, 'Prepare 7.8.3 (e1f2a3b4)?', 'a newer range is a new summary');
});

test('an update staged with the earlier app identity is stale and prepared again, not a changed checkout', () => {
    const stale = (reason) => updateVerdict({ ...STATUS, candidate: candidate({ state: 'stale', reason,
        actions: ['prepare', 'discard_candidate'] }) });
    const earlier = stale('native_identity_stale');
    assert.equal(earlier.state, 'candidate_stale');
    assert.equal(earlier.headline, 'This update was prepared by an earlier version of Ouroboros.');
    assert.match(earlier.hint,
        /^Prepare it again so the app it brings is checked as this version checks apps\. Nothing was changed;/);
    assert.equal(stale('local_changed').headline, 'This checkout changed after the update was prepared.');
});

test('an app Homebrew installed and a --runtime-venv are named where the Update lists the app and the packages', () => {
    const native = { status: 'staged', version: '7.8.0', previous_version: '7.7.0',
        installed: '/Applications/Ouroboros.app', backup_app: '/Applications/.Ouroboros-7.7.0-backup.app',
        origin: 'official', manager: 'Homebrew Cask', manager_version: '7.7.0',
        runtime_venv: '/Users/o/ouroboros-venv' };
    const dialog = candidateDialog('activate', { ...STATUS, candidate: candidate({ state: 'ready', native }) });
    assert.equal(dialog.details.rows.find((item) => item.label === 'Installed by').value,
        'Homebrew Cask, which installed 7.7.0; its record stays as it is');
    const ready = candidate({ state: 'ready', native, actions: ['activate'] });
    const facts = updateVerdict({ ...STATUS, candidate: ready }).facts;
    assert.equal(facts.find((item) => item.label === 'Python packages').value,
        'its own copy of the virtual environment /Users/o/ouroboros-venv, which the new app starts with');
    const plain = candidateDialog('activate', { ...STATUS, candidate: candidate({ state: 'ready',
        native: { ...native, manager: undefined, manager_version: undefined } }) });
    assert.equal(plain.details.rows.find((item) => item.label === 'Installed by'), undefined, 'no manager, no row');
    // Another installer replaced an app just before the swap: the composition came back before the new version ran.
    const returned = updateVerdict({ ...STATUS, candidate: candidate({ state: 'failed', reason: 'activation:returned',
        activation: { outcome: 'returned', sha: SHA.base, previous: SHA.target, why: 'app_changed',
            decision: { trigger: 'app_changed' } }, actions: ['prepare', 'discard_candidate'] }) });
    assert.match(returned.hint, /^It switched back before the new version ran, so your data is as it was\.$/);
    const view = activationAttemptView(after(settled({ outcome: 'returned', sha: SHA.base, previous: SHA.target,
        why: 'app_changed', trigger: 'app_changed' })));
    assert.equal(row(view, 'Why'), 'another installer (such as Homebrew) replaced an app on disk just before the swap, '
        + 'so the apps were not exchanged');
});

test('Restart to update names the app it replaces and how the package was checked, from the staged record', () => {
    const native = { status: 'staged', version: '7.8.0', previous_version: '7.7.0', installed: '/Users/o/Desktop/Ouroboros.app',
        backup_app: '/Users/o/Desktop/.Ouroboros-update-7.8.0-x/Ouroboros-7.7.0-backup.app', origin: 'official',
        sha256: '41b9a890c2d4e6f8a0b2c4d6e8f0a2b4c6d8e0f2a4b6c8d0e2f4a6b8c0d2ef57' };
    const dialog = candidateDialog('activate', { ...STATUS, candidate: candidate({ state: 'ready', native }) });
    assert.match(dialog.body, /This desktop app \(7\.7\.0\) is replaced with 7\.8\.0; the previous app stays beside it\./);
    assert.deepEqual(dialog.details.rows.map((item) => [item.label, item.value]), [['New app', '7.8.0'],
        ['This app', '7.7.0, from an official release, at /Users/o/Desktop/Ouroboros.app'],
        ['Checksum', 'SHA-256 41b9a890c2d4…, matches the release'], ['Signature', 'valid and notarized, the official team’s'],
        ['Previous app kept at', native.backup_app]]);
    assert.equal(candidateDialog('activate', { ...STATUS, candidate: candidate({ state: 'ready' }) }).details, undefined,
        'a source-only update has no app details');
    const facts = updateVerdict({ ...STATUS, candidate: candidate({ state: 'ready', native, actions: ['activate'] }) }).facts;
    assert.equal(facts.find((item) => item.label === 'Desktop app').value, '7.7.0 → 7.8.0: its checksum matches the release '
        + 'and its signature is the official one; it replaces this app at the restart');
    assert.equal(facts.find((item) => item.label === 'App location').value, native.installed);
});

test('every field the panel and the pill read is one its producer writes', () => {
    const py = readFileSync(new URL('../../supervisor/update_prepare.py', import.meta.url), 'utf8');
    const keysOf = (text) => new Set([...text.matchAll(/"([a-z_]+)"/g)].map((match) => match[1]));
    const retainer = py.slice(py.indexOf('def _retain_activation('), py.indexOf('def _forget_activation('));
    const retained = keysOf(retainer);
    const panel = readFileSync(new URL('../modules/update_candidate.js', import.meta.url), 'utf8');
    const pill = readFileSync(new URL('../modules/update_status.js', import.meta.url), 'utf8');
    const lastReads = new Set([...panel.matchAll(/\blast\.([a-z_]+)/g)].map((match) => match[1]));
    assert.ok(lastReads.size >= 10, 'no last_activation reads found — the module moved');
    assert.deepEqual([...lastReads].filter((key) => !retained.has(key)), [], 'the Last attempt reads only what is retained');
    const nativeFacts = keysOf(py.slice(py.indexOf('_NATIVE_FACTS = ('), py.indexOf('def projection(')));
    const nativeReads = new Set([...panel.matchAll(/\bnative\.([a-z_]+)/g)].map((match) => match[1]));
    assert.deepEqual([...nativeReads].filter((key) => !nativeFacts.has(key) && !retained.has(key)), [],
        'the app facts read are the projected (or retained) ones');
    const emitted = keysOf(py.slice(py.indexOf('def projection(')));
    const pillReads = new Set([...pill.matchAll(/\bcandidate\.([a-z_]+)/g)].map((match) => match[1]));
    assert.deepEqual([...pillReads].filter((key) => !emitted.has(key)), [], 'the pill reads only projected fields');
    // The carried list and the boot's observation: written where the panel reads them.
    const layers = keysOf(py.slice(py.indexOf('def carried_layers('), py.indexOf('# The record')));
    for (const key of ['count', 'paths', 'list']) assert.ok(layers.has(key), `carried_layers writes ${key}`);
    const gateway = readFileSync(new URL('../../ouroboros/gateway/update_activation.py', import.meta.url), 'utf8');
    const linked = keysOf(gateway.slice(gateway.indexOf('def candidate_view('), gateway.indexOf('def _activate_fenced(')));
    for (const key of ['list_url', 'carried_list']) assert.ok(linked.has(key), `the gateway links the list by ${key}`);
    assert.ok(panel.includes("'/api/files/download?carried_list='"), 'the panel accepts that link only');
    const recorder = keysOf(py.slice(py.indexOf('def record_activation('), py.indexOf('def _retain_activation(')));
    for (const key of ['unconfirmed', 'observed_at', 'why']) assert.ok(recorder.has(key), `record_activation keeps ${key}`);
});

test('the checks name what failed only here, what the official version fails too, dropped tests and unknown coverage', () => {
    // supervisor/update_checks.selected_tests: blocking stays blocking when T fails it too (attribution, never a waiver).
    const tests = { blocking: ['tests/test_queue.py::test_drain'], already_failing: ['tests/test_net.py::test_dns'],
        target_failing: ['tests/test_queue.py::test_drain'], dropped_local_tests: ['tests/test_my_tool.py'],
        uncovered: ['notes/local-ideas.md', 'web/my_panel.js'], uncovered_count: 9 };
    const view = updateVerdict({ ...STATUS, ...LOCAL, candidate: candidate({ state: 'failed', reason: 'checks_failed',
        checks: { ok: false, tests }, actions: ['prepare', 'discard_candidate'] }) });
    const fact = (label) => view.facts.find((item) => item.label === label)?.value;
    assert.equal(view.headline, '7.8.2 (d2f6a91c) did not pass its checks. Nothing was changed.');
    assert.equal(fact('Checks'), 'failed only with the update: tests/test_queue.py::test_drain');
    assert.equal(fact('Also failing in the official version'), 'tests/test_queue.py::test_drain');
    assert.equal(fact('Already failing before'), 'tests/test_net.py::test_dns');
    assert.equal(fact('Your tests no longer in it'), 'tests/test_my_tool.py');
    assert.equal(fact('No automatic test covers'), 'notes/local-ideas.md, web/my_panel.js and 7 more');
    const unverified = updateVerdict({ ...STATUS, candidate: candidate({ state: 'failed', reason: 'checks_unverified',
        checks: { ok: false, tests: { unverified: 'the selected run did not complete' } }, actions: ['prepare'] }) });
    assert.equal(unverified.facts.find((item) => item.label === 'Checks').value, 'did not complete, so nothing was verified');
});

test('a test file the official version requires and the update lacks is worded as missing, never as a test that failed', () => {
    // supervisor/update_checks.selected_tests: the floor (the official target's own required test files) the prepared
    // update lacks is `floor_missing`, also its `blocking`, refused before any run: checks_failed with nothing run.
    // A measured failure keeps "fail only with the update"; a floor or a run that could not be read or finished is `unverified`.
    const shown = (reason, tests) => updateVerdict({ ...STATUS, ...LOCAL, candidate: candidate({ state: 'failed', reason,
        checks: { ok: false, tests }, actions: ['prepare', 'discard_candidate'] }) });
    const checks = (view) => view.facts.find((item) => item.label === 'Checks').value;
    const floor = ['tests/test_floor_one.py', 'tests/test_floor_two.py'];
    const lacks = 'the update lacks test file(s) the official version requires: tests/test_floor_one.py, tests/test_floor_two.py';

    const missing = shown('checks_failed', { floor_missing: floor, blocking: floor, failed: {}, changed: 2, local_tests: 1 });
    assert.equal(checks(missing), `not run: ${lacks}`);
    assert.equal(missing.hint, `No test was run: ${lacks}.`);
    assert.doesNotMatch(`${checks(missing)} ${missing.hint}`, /fail(?:s|ed)? only with the update/, 'nothing ran, so nothing failed');
    // As many as the measured-failure lists show, then how many more: five in the sentence, three in the fact.
    const many = Array.from({ length: 7 }, (_, index) => `tests/test_floor_${index}.py`);
    const long = shown('checks_failed', { floor_missing: many, blocking: many });
    assert.match(long.hint, /^No test was run: .*: tests\/test_floor_0\.py, .*, tests\/test_floor_4\.py and 2 more\.$/);
    assert.match(checks(long), /tests\/test_floor_2\.py and 4 more$/);

    const measured = shown('checks_failed', { blocking: ['tests/test_queue.py::test_drain'],
        failed: { 'tests/test_queue.py::test_drain': 'AssertionError' } });
    assert.equal(checks(measured), 'failed only with the update: tests/test_queue.py::test_drain');
    assert.equal(measured.hint, 'Tests that fail only with the update: tests/test_queue.py::test_drain.');

    // An unreadable declaration of the floor, like a run that did not finish, verified nothing: neither passed nor failed.
    const unknown = shown('checks_unverified', { unverified: "the official target's required floor is unknown: cannot be read",
        blocking: [], failed: {}, floor: { source: `${SHA.target}:ouroboros/test_selection.py`, error: 'cannot be read' } });
    assert.equal(checks(unknown), 'did not complete, so nothing was verified');
    assert.equal(unknown.hint, 'Its checks did not complete, so it was neither passed nor failed. Prepare it again.');

    // Status and actions are the failed checks' own, whatever the cause: Prepare again, with Discard beside it.
    const same = (view) => [view.state, view.tone, view.headline, view.action, view.secondary];
    assert.deepEqual(same(missing), same(measured));
    assert.deepEqual([missing.state, missing.tone, missing.headline, missing.action?.id, missing.secondary.map((item) => item.id)],
        ['candidate_failed', 'error', '7.8.2 (d2f6a91c) did not pass its checks. Nothing was changed.', 'prepare', ['discard_candidate']]);

    // The record above is the producer's own, and "No test was run" is its premise: the typed refusal writes these keys and
    // returns before any run.
    const producer = readFileSync(new URL('../../supervisor/update_checks.py', import.meta.url), 'utf8');
    const refusal = producer.slice(producer.indexOf('except FloorMissing'), producer.indexOf('root = _process_root()'));
    assert.ok(refusal.includes('"floor_missing": exc.missing') && refusal.includes('"blocking": exc.missing'), 'the refusal moved');
    assert.ok(!refusal.includes('run_tests'), 'a refusal that runs tests is no longer "No test was run"');
});
