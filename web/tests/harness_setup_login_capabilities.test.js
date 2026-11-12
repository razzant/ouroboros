// Focused rendering contract for setup-login transport capabilities. Controller
// lifecycle and custody behavior remain in harness_login_cards.test.js.

import assert from 'node:assert/strict';
import test from 'node:test';

import {
    attachShellLabel,
    LOGIN_CARD_COMPACT,
    loginCardHtml,
} from '../modules/harness_login_cards.js';

test('compact drops Advanced, paste-code and Close but keeps a completing command and retry', () => {
    const active = {
        harness: 'claude', profile: '', startedAtMs: 0, engineDegraded: true,
        attachCommand: 'claudexor setup attach j1', attachShell: 'powershell',
        setupLoginSource: 'per_harness', error: '', verdict: null, confirming: false,
        envelope: { job: { state: 'waiting_for_input' }, deviceCode: {
            flow: 'oauth_url_input', verificationUrl: 'https://example.test/signin', userCode: '' } },
    };
    const full = loginCardHtml(active, 999999);
    assert.ok(full.includes('data-login-code-input'), 'full keeps the optional paste-code entry');
    assert.ok(full.includes('data-login-advanced'), 'full keeps the collapsed terminal fallback');
    assert.ok(full.includes('Command for PowerShell'), 'the exposed syntax names its shell');
    assert.ok(full.includes('uses an external terminal for sign-in on this host'));
    assert.ok(full.includes('data-login-dismiss'));

    const compact = loginCardHtml(active, 999999, { mode: LOGIN_CARD_COMPACT });
    // The sign-in action itself survives — a card that cannot start the login
    // would be worse than none.
    assert.ok(compact.includes('data-open-signin'), 'compact keeps the sign-in link');
    assert.ok(compact.includes('data-login-state'), 'compact keeps the progress line');
    assert.ok(!compact.includes('data-login-code-input'));
    assert.ok(!compact.includes('data-login-advanced'));
    assert.ok(compact.includes('data-login-external-command'),
        'compact exposes the labelled attach command instead of stranding the flow');
    assert.ok(compact.includes('Command for PowerShell'));
    assert.ok(!compact.includes('data-login-dismiss'));
    assert.ok(compact.includes(`data-login-mode="${LOGIN_CARD_COMPACT}"`));

    // A settled non-success verdict offers Try again in compact (the wizard has
    // no account row behind it to retry from).
    const failed = loginCardHtml({ ...active, verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' } },
        999999, { mode: LOGIN_CARD_COMPACT });
    assert.ok(failed.includes('data-login-retry'));
    assert.ok(failed.includes('Could not confirm the sign-in yet'));

    const verified = loginCardHtml({ ...active, verdict: { kind: 'success', reason: '' } },
        999999, { mode: LOGIN_CARD_COMPACT });
    assert.ok(verified.includes('Signed in.'));
    assert.ok(!verified.includes('data-login-retry'), 'nothing to retry once verified');

    const explicitExternal = { ...active, transport: 'client_pty',
        envelope: { job: { state: 'running' } } };
    for (const options of [{}, { mode: LOGIN_CARD_COMPACT }]) {
        const html = loginCardHtml(explicitExternal, 999999, options);
        assert.ok(html.includes('data-login-external-command'), html);
        assert.ok(html.includes('Continue this sign-in in your own terminal'), html);
        assert.ok(html.includes('Command for PowerShell'), html);
        assert.ok(html.includes('claudexor setup attach j1'), html);
        assert.ok(!html.includes('data-login-advanced'),
            'an explicit continuation exposes the command instead of hiding it in Advanced');
    }
});

test('only exact terminal receipt codes/actions expose the explicit external continuation', () => {
    const preJob = {
        harness: 'one', profile: '', transport: '', error: 'helper unavailable',
        problemCode: 'terminal_transport_unavailable',
        requiredActions: ['use_external_terminal'], envelope: null,
    };
    for (const options of [{}, { mode: LOGIN_CARD_COMPACT }]) {
        const html = loginCardHtml(preJob, 1, options);
        assert.ok(html.includes('data-login-external-terminal'), html);
        assert.ok(html.includes('Continue in external terminal'), html);
        assert.ok(!html.includes('data-login-retry'),
            'unavailable without retry_setup_login must not loop the same request');
    }
    const probe = loginCardHtml({ ...preJob,
        problemCode: 'terminal_transport_probe_failed',
        requiredActions: ['retry_setup_login', 'use_external_terminal'],
    }, 1, { mode: LOGIN_CARD_COMPACT });
    assert.ok(probe.includes('data-login-external-terminal'));
    assert.ok(probe.includes('data-login-retry'));

    for (const errorCode of [
        'terminal_transport_unavailable', 'terminal_transport_unsupported',
        'terminal_transport_probe_failed', 'terminal_transport_failed',
    ]) {
        const durable = loginCardHtml({
            harness: 'one', profile: '', transport: '', error: '', requiredActions: [],
            verdict: { kind: 'failure', reason: errorCode }, confirming: false,
            envelope: { job: { state: 'failed', nativeCommand: { errorCode } } },
        }, 1, { mode: LOGIN_CARD_COMPACT });
        assert.ok(durable.includes('data-login-external-terminal'), errorCode);
        if (errorCode === 'terminal_transport_unavailable'
            || errorCode === 'terminal_transport_unsupported') {
            assert.ok(!durable.includes('data-login-retry'), errorCode);
        }
    }
    const nearMiss = loginCardHtml({ ...preJob,
        problemCode: 'terminal transport unavailable',
        error: 'terminal_transport_unavailable appears only in prose',
    }, 1);
    assert.ok(!nearMiss.includes('data-login-external-terminal'),
        'no text sniff or near-match may select the action');
});

test('attach shell labels are bounded and legacy global capability is disclosed', () => {
    assert.equal(attachShellLabel('powershell'), 'PowerShell');
    assert.equal(attachShellLabel('posix'), 'POSIX shell');
    assert.equal(attachShellLabel('future'), 'your terminal shell');
    const html = loginCardHtml({
        harness: 'one', profile: '', startedAtMs: 1, engineDegraded: true,
        attachCommand: 'CLAUDEXOR_CONFIG_DIR=/tmp/cfg claudexor setup attach j1',
        attachShell: 'posix', setupLoginSource: 'legacy_global_operation',
        error: '', verdict: null, confirming: false,
        envelope: { job: { state: 'waiting_for_input' } },
    }, 2);
    assert.match(html, /Command for POSIX shell/);
    assert.match(html, /older agent service exposes only a global sign-in capability/);
});
