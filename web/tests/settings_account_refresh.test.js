import assert from 'node:assert/strict';
import test from 'node:test';
import { accountCatalogRefreshKey } from '../modules/settings_catalog.js';

function statusView({ profileId = 'work', verification = 'passed' } = {}) {
    return {
        reads: { accounts: 'ok' },
        snapshot: {
            profiles: {
                profiles: [{
                    profile: { harness_id: 'codex', profile_id: profileId, enabled: true },
                    identity: { email: `${profileId}@example.test` },
                    status: { verification },
                }],
                harnessAccounts: [],
            },
        },
    };
}

test('catalog refresh key waits for a confirmed Accounts read', () => {
    assert.equal(accountCatalogRefreshKey({ reads: { accounts: 'unread' }, snapshot: {} }), null);
    assert.equal(accountCatalogRefreshKey({ reads: { accounts: 'failed' }, snapshot: {} }), null);
    assert.notEqual(accountCatalogRefreshKey(statusView()), null);
});

test('account login/status changes produce a new catalog refresh key', () => {
    const settled = accountCatalogRefreshKey(statusView());
    assert.equal(accountCatalogRefreshKey(statusView()), settled, 'an unchanged poll is quiet');
    assert.notEqual(accountCatalogRefreshKey(statusView({ verification: 'not_run' })), settled,
        'a reconnect/status transition is observable');
    assert.notEqual(accountCatalogRefreshKey(statusView({ profileId: 'personal' })), settled,
        'a newly settled account is observable');
});

test('an empty confirmed catalog still asks for an account on a clean install', async () => {
    const { routeChoiceGroups } = await import('../modules/route_editor_primitives.js');
    const empty = routeChoiceGroups({ catalogKnown: true, accountsKnown: true, hasConfiguredAccounts: false });
    assert.equal(empty[0].options[0].label, 'No model sources listed: connect one in Accounts');
    const stale = routeChoiceGroups({ catalogKnown: true, accountsKnown: true, hasConfiguredAccounts: true });
    assert.equal(stale[0].options[0].label, 'No model sources listed: press Read again in Models');
});
