import test from 'node:test';
import assert from 'node:assert/strict';

import {
    makeRemotePlacementRequest,
    remoteConnectionChoices,
    remoteConnectionsErrorCopy,
    remoteGitRootNote,
} from '../modules/project_create.js';

// The New Project dialog's REMOTE decisions live in pure functions so they can be
// asserted without a DOM: which connections may be offered, what the dialog says
// when none may be, what it may claim about the selected folder, and what actually
// crosses the wire. The DOM plumbing around them stays in the closure.

const _ready = (over = {}) => ({
    id: 'conn-1',
    name: 'Build box',
    ssh_alias: 'build',
    lifecycle: 'active',
    status: 'ready',
    bootstrap_compatible: true,
    health_fresh: true,
    ...over,
});

test('the picker offers only connections a placement could actually be admitted on', () => {
    const { options, placeholder } = remoteConnectionChoices([
        _ready(),
        _ready({ id: 'conn-2', name: 'Second', ssh_alias: 'two' }),
        // Each of these would be refused by admission, so offering it would move the
        // refusal to after the owner had already picked a folder.
        _ready({ id: 'never-bootstrapped', bootstrap_compatible: false }),
        _ready({ id: 'stale-health', health_fresh: false }),
        _ready({ id: 'degraded', status: 'degraded' }),
        _ready({ id: 'retired', lifecycle: 'retired' }),
    ]);
    assert.deepEqual(options, [
        { value: 'conn-1', label: 'Build box (build)' },
        { value: 'conn-2', label: 'Second (two)' },
    ]);
    assert.equal(placeholder, 'Choose a connection');
});

test('with no connection at all the picker says the only true thing: add one', () => {
    for (const rows of [[], undefined]) {
        const { options, placeholder } = remoteConnectionChoices(rows);
        assert.deepEqual(options, []);
        assert.match(placeholder, /No SSH connections yet/);
        assert.match(placeholder, /Settings → Connections/);
    }
});

// THE regression. Measured live: with an executor built against an older Home<->execd
// contract set, the picker said "run Bootstrap (or Test to refresh health)", the owner
// pressed Test, Test answered ok with `health_fresh: true`, and the connection stayed
// unselectable — only Bootstrap writes the contract-set stamp. The placeholder must now
// quote the SERVER's blocker sentence, which names Bootstrap and says in so many words
// that Test will change nothing.
test('the empty state names the one action that removes the CURRENT block', () => {
    const { options, placeholder } = remoteConnectionChoices([
        _ready({
            id: 'stale-execd',
            execd_outdated: true,
            blocked_by: 'remote_execd_outdated',
            blocker_action: 'bootstrap_connection',
            blocker_hint: 'Run Bootstrap in Settings → Connections — Test will report it '
                + 'healthy and change nothing.',
            blocker_rank: 3,
        }),
    ]);
    assert.deepEqual(options, []);
    assert.match(placeholder, /Run Bootstrap/);
    assert.match(placeholder, /change nothing/);
    // …and it must NOT offer Test as an alternative route, which is the exact text
    // that turned a refusal into a dead end.
    assert.doesNotMatch(placeholder, /or Test to refresh health/);
});

test('with several blocked connections the picker advises about the nearest one', () => {
    // Ranks come from the server's removal ladder: HIGHER means fewer steps left, so a
    // connection that needs one Test wins over one that needs Retrust then Bootstrap.
    const { placeholder } = remoteConnectionChoices([
        _ready({
            id: 'swapped',
            bootstrap_compatible: false,
            blocked_by: 'host_identity_changed',
            blocker_action: 'retrust_host',
            blocker_hint: 'A different host is answering this SSH alias.',
            blocker_rank: 1,
        }),
        _ready({
            id: 'restarted',
            health_fresh: false,
            blocked_by: 'connection_health_stale',
            blocker_action: 'test_connection',
            blocker_hint: 'This connection has not answered since Ouroboros started.',
            blocker_rank: 5,
        }),
    ]);
    assert.match(placeholder, /has not answered since Ouroboros started/);
    assert.doesNotMatch(placeholder, /different host/);
});

test('an unreadable connections list is explained by its cause, not its status code', () => {
    assert.match(
        remoteConnectionsErrorCopy({ body: { error_code: 'owner_auth_required' } }),
        /Unlock owner access/,
    );
    assert.match(
        remoteConnectionsErrorCopy({ body: { error_code: 'owner_auth_not_configured' } }),
        /Network Password/,
    );
    // The transport being absent is a RESTART, not an authentication problem — and it
    // must never read as "you have no connections".
    for (const code of ['remote_transport_unavailable', 'remote_service_unavailable']) {
        const copy = remoteConnectionsErrorCopy({ body: { error_code: code } });
        assert.match(copy, /SSH transport is not running/);
        assert.doesNotMatch(copy, /no connections/i);
    }
    // Anything untyped is passed through verbatim rather than paraphrased.
    assert.equal(
        remoteConnectionsErrorCopy({ body: { error: 'ssh: connect timed out' } }),
        'ssh: connect timed out',
    );
    assert.equal(remoteConnectionsErrorCopy(new Error('offline')), 'offline');
});

test('the picker never claims a git verdict the host has not given', () => {
    assert.equal(remoteGitRootNote(true), 'Git worktree root ✓');
    assert.equal(remoteGitRootNote(false), 'Not a Git worktree root');
    // Unknown says the check is still to come; the target's own `git rev-parse` runs
    // at creation and is the only authority.
    assert.match(remoteGitRootNote(null), /validated on the host before creation/);
    assert.match(remoteGitRootNote(undefined), /validated on the host before creation/);
});

test('the request carries the two halves of a placement and never a workspace_ref', () => {
    const payload = makeRemotePlacementRequest(' conn-1 ', ' /srv/work/app ');
    assert.deepEqual(payload, { connection_id: 'conn-1', remote_root: '/srv/work/app' });
    // The workspace identity is allocated by the TARGET at admission, so the browser
    // has nothing to put there and must not invent a half-filled ref.
    assert.equal('workspace_ref' in payload, false);
    assert.equal('kind' in payload, false);
    assert.equal('workspace_id' in payload, false);

    // Half a selection is no request at all: the dialog blocks on it rather than
    // sending a body the server would have to guess at.
    assert.equal(makeRemotePlacementRequest('', '/srv/work/app'), null);
    assert.equal(makeRemotePlacementRequest('conn-1', ''), null);
    assert.equal(makeRemotePlacementRequest('   ', '   '), null);
});
