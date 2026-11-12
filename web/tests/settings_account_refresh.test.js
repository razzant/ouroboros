import assert from 'node:assert/strict';
import test from 'node:test';

test('an empty confirmed catalog still asks for an account on a clean install', async () => {
    const { routeChoiceGroups } = await import('../modules/route_editor_primitives.js');
    const empty = routeChoiceGroups({ catalogKnown: true, accountsKnown: true, hasConfiguredAccounts: false });
    assert.equal(empty[0].options[0].label, 'No model sources listed — connect one in Accounts');
    const stale = routeChoiceGroups({ catalogKnown: true, accountsKnown: true, hasConfiguredAccounts: true });
    assert.equal(stale[0].options[0].label, 'No model sources listed — refresh Model Catalog');
});
