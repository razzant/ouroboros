import test from 'node:test';
import assert from 'node:assert/strict';

import {
    connectionActionState,
    connectionBlocker,
    connectionStatusCopy,
    isSelectableRemoteConnection,
    renderConnectionRow,
    transportNoteMarkup,
} from '../modules/connections_ui.js';
import { REMOTE_TRANSPORT_UNAVAILABLE } from '../modules/remote_task_state.js';

// `utils.escapeHtmlText` escapes by round-tripping through a detached element, so
// the pure row renderers need the smallest possible stand-in for that one browser
// behaviour — textContent in, entity-escaped innerHTML out. No jsdom. Nothing is
// called during module evaluation, so installing it here is early enough.
globalThis.document = globalThis.document || {
    createElement: () => ({
        innerHTML: '',
        set textContent(value) {
            this.innerHTML = String(value ?? '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        },
    }),
};

test('project picker requires bootstrap evidence and fresh ready health', () => {
    assert.equal(isSelectableRemoteConnection({
        lifecycle: 'active',
        status: 'ready',
        bootstrap_compatible: true,
        health_fresh: true,
    }), true);
    assert.equal(isSelectableRemoteConnection({
        lifecycle: 'active',
        status: 'ready',
        expected_host_id: 'pinned-host',
        bootstrap_compatible: false,
        health_fresh: true,
    }), false);
    assert.equal(isSelectableRemoteConnection({
        lifecycle: 'active',
        status: 'ready',
        bootstrap_compatible: true,
        health_fresh: false,
    }), false);
    assert.equal(isSelectableRemoteConnection({
        lifecycle: 'active',
        status: 'degraded',
        expected_host_id: 'pinned-host',
        bootstrap_compatible: true,
        health_fresh: true,
    }), false);
    assert.equal(isSelectableRemoteConnection({
        lifecycle: 'retired',
        status: 'ready',
        bootstrap_compatible: true,
        health_fresh: true,
    }), false);
});

test('connection status copy keeps phase and typed error visible', () => {
    assert.equal(
        connectionStatusCopy({
            status: 'degraded',
            phase: 'handshake',
            error_code: 'incompatible_protocol',
        }),
        'degraded · phase: handshake · error: incompatible_protocol',
    );
});

test('connection status copy exposes neutralized SSH alias directives', () => {
    assert.equal(
        connectionStatusCopy({
            status: 'ready',
            phase: 'connect',
            warnings: [{
                code: 'ssh_alias_forwarding_neutralized',
                directives: ['localforward'],
            }],
        }),
        'ready · phase: connect · warning: ssh_alias_forwarding_neutralized',
    );
});

// A build whose ssh transport has not been assembled answers every
// transport-dependent owner route with a typed 503 (`remote_transport_unavailable`,
// gateway/connections.py). The card must SAY that, not spin and not go blank.
test('a build without the ssh transport disables exactly the transport actions', () => {
    const missing = { transportAvailable: false };
    for (const action of ['test', 'bootstrap', 'reconnect', 'retrust', 'retire']) {
        const state = connectionActionState(action, missing);
        assert.equal(state.disabled, true, `${action} must be disabled`);
        assert.match(state.reason, /not part of this build/);
    }
    // Listing and adding are pure owner-state operations: they keep working, so
    // the owner can still curate connections in a transport-less build.
    assert.deepEqual(
        connectionActionState('add', missing),
        { disabled: false, reason: '' },
    );
    // With the transport present nothing is blocked, and `loading` alone never
    // invents an explanation it does not have.
    assert.deepEqual(
        connectionActionState('bootstrap', { transportAvailable: true }),
        { disabled: false, reason: '' },
    );
    assert.deepEqual(
        connectionActionState('bootstrap', { loading: true, transportAvailable: true }),
        { disabled: true, reason: '' },
    );
});

test('the missing-transport state is rendered as a settled note, never a spinner', () => {
    assert.equal(transportNoteMarkup(true), '');
    const markup = transportNoteMarkup(false);
    assert.match(markup, /data-conn-transport-missing/);
    assert.match(markup, /role="note"/);
    assert.match(markup, /not part of this build/);
    assert.match(markup, /trust history are untouched/);
    assert.doesNotMatch(markup, /aria-busy="true"/);

    const row = renderConnectionRow(
        { id: 'conn-1', name: 'Production', ssh_alias: 'production', lifecycle: 'active' },
        { transportAvailable: false },
    );
    // Every transport button is present-but-refused with the reason attached, so
    // the owner learns WHY rather than watching a silent no-op.
    assert.equal((row.match(/ disabled/g) || []).length, 5);
    assert.equal((row.match(/title="The remote SSH transport/g) || []).length, 5);
    assert.match(row, /data-conn-action="bootstrap"/);
});

test('a retired connection offers no actions at all', () => {
    const row = renderConnectionRow(
        { id: 'conn-1', name: 'Old', ssh_alias: 'old', lifecycle: 'retired' },
        { transportAvailable: true },
    );
    assert.doesNotMatch(row, /data-conn-action/);
    assert.match(row, />Retired</);
});

test('the typed transport-unavailable code is the one shared spelling', () => {
    assert.equal(REMOTE_TRANSPORT_UNAVAILABLE, 'remote_transport_unavailable');
});

// A cap that leaves no trace reads exactly like data that was never there, and the
// one thing an operator reads a diagnostic for is what is NOT in it. The gateway
// sends `<name>_count` beside every bounded list, so "four warnings" and "four of
// nine warnings" stop being the same sentence.
test('an abbreviated warning list says how much it left out', () => {
    assert.equal(
        connectionStatusCopy({
            status: 'degraded',
            phase: 'connect',
            warnings: Array.from({ length: 6 }, (_, index) => ({ code: `w${index}` })),
            warnings_count: 9,
        }),
        'degraded · phase: connect · warning: w0 · warning: w1 · warning: w2 '
        + '· warning: w3 · +5 more warnings',
    );
    // Singular reads as singular.
    assert.match(
        connectionStatusCopy({
            status: 'ready',
            warnings: Array.from({ length: 5 }, (_, index) => ({ code: `w${index}` })),
        }),
        /\+1 more warning$/,
    );
    // Nothing is claimed when nothing was cut.
    assert.equal(
        connectionStatusCopy({ status: 'ready', warnings: [{ code: 'only' }], warnings_count: 1 }),
        'ready · warning: only',
    );
    // A total the server never sent cannot make the received list look short.
    assert.equal(
        connectionStatusCopy({ status: 'ready', warnings: [{ code: 'only' }] }),
        'ready · warning: only',
    );
});

test('the details panel shows every warning it received and names the gateway cap', () => {
    const row = renderConnectionRow({
        id: 'conn-1',
        name: 'Build',
        ssh_alias: 'build',
        lifecycle: 'active',
        status: 'degraded',
        warnings: Array.from({ length: 6 }, (_, index) => ({ code: `w${index}` })),
        warnings_count: 40,
        log_refs: [{ stream: 'stderr' }],
        log_refs_count: 1,
    });
    // The summary abbreviates; the panel does not.
    assert.match(row, /w5/);
    assert.match(row, /34 more entries omitted by the bounded live projection/);
    // The log refs were complete, so they get no note at all.
    assert.equal((row.match(/omitted by the bounded live projection/g) || []).length, 1);
});

// An execd whose contract set predates this Home build reports `ready` on every other
// measure and fails EVERY remote tool call (`gateway/connections._runtime_evidence_fields`
// derives `execd_outdated` from the durable `bootstrap_contract_set`). The row must say
// so, must not read green, and must not be offered as a place a new project can go —
// otherwise the surface agrees with the wrong half of the evidence.
test('an outdated executor is named in the row summary and never reads as ready', () => {
    // The blocker and its action come from the SERVER's ladder
    // (`remote_refusal_actions.connection_blocker`), which is why this row carries
    // `blocked_by`/`blocker_action`: a badge composed here would be a second opinion
    // about which action fixes which state, and the picker's version of that second
    // opinion is what sent the owner to press Test on a block Test cannot move.
    assert.equal(
        connectionStatusCopy({
            status: 'ready',
            phase: 'connect',
            lifecycle: 'active',
            bootstrap_compatible: true,
            health_fresh: true,
            execd_outdated: true,
            blocked_by: 'remote_execd_outdated',
            blocker_action: 'bootstrap_connection',
        }),
        'ready · phase: connect · blocked: remote_execd_outdated - bootstrap_connection',
    );
    // A current executor adds nothing: no noise where none is due.
    assert.equal(
        connectionStatusCopy({
            status: 'ready',
            phase: 'connect',
            lifecycle: 'active',
            bootstrap_compatible: true,
            health_fresh: true,
            execd_outdated: false,
        }),
        'ready · phase: connect',
    );
});

test('the blocker a row shows is exactly the blocker the server put on it', () => {
    const stale = {
        lifecycle: 'active',
        status: 'ready',
        bootstrap_compatible: true,
        health_fresh: true,
        execd_outdated: true,
        blocked_by: 'remote_execd_outdated',
        blocker_action: 'bootstrap_connection',
        blocker_hint: 'Run Bootstrap — Test will report it healthy and change nothing.',
        blocker_rank: 3,
    };
    // Non-null exactly when the row is not selectable, and null the moment it is:
    // "has no blocker" and "may carry a project" are ONE fact, not two opinions.
    assert.equal(connectionBlocker(stale)?.action, 'bootstrap_connection');
    assert.equal(connectionBlocker({ ...stale, execd_outdated: false, blocked_by: '' }), null);
    // Nothing is invented locally. A row with no server verdict yields no hint,
    // because re-deriving one here is how the two authorities drift apart.
    assert.equal(connectionBlocker({ lifecycle: 'active', status: 'degraded' }), null);
});

test('an outdated executor is not selectable for a new project, and Bootstrap restores it', () => {
    const stale = {
        lifecycle: 'active',
        status: 'ready',
        bootstrap_compatible: true,
        health_fresh: true,
        execd_outdated: true,
    };
    assert.equal(isSelectableRemoteConnection(stale), false);
    // The one action that fixes it clears exactly that flag and nothing else.
    assert.equal(isSelectableRemoteConnection({ ...stale, execd_outdated: false }), true);
});

test('an outdated executor row renders warn, not ok', () => {
    const markup = renderConnectionRow({
        id: 'conn-stale',
        name: 'Build host',
        ssh_alias: 'build-host',
        lifecycle: 'active',
        status: 'ready',
        execd_outdated: true,
        bootstrap_contract_set: 0,
        required_contract_set: 1,
        bootstrap_compatible: true,
        health_fresh: true,
        blocked_by: 'remote_execd_outdated',
        blocker_action: 'bootstrap_connection',
        blocker_hint: 'Run Bootstrap in Settings → Connections — Test will change nothing.',
        blocker_rank: 3,
    }, {});
    assert.ok(markup.includes('remote_execd_outdated'), 'the badge text must reach the row');
    assert.ok(
        markup.includes('Test will change nothing'),
        'the details panel must carry the sentence that says what will NOT help',
    );
    assert.ok(!markup.includes('status-ok'), 'a host that cannot serve must not read green');
    assert.ok(
        markup.includes('0 installed, 1 required'),
        'the details panel must name both contract sets',
    );
});
