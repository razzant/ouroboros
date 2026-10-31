import test from 'node:test';
import assert from 'node:assert/strict';
import { receiptNotice } from '../modules/project_handoff.js';

// The Project work entry is gone from Main (the moved work keeps its own card: work_portal.js); the binding's
// receipt word still reaches the owner as one toast after Turn into project.
test('receipt words: durable and already delivered say nothing, every other word names its gap', () => {
    assert.equal(receiptNotice('durable'), '');
    assert.equal(receiptNotice('already_delivered'), '');
    assert.match(receiptNotice('unregistered'), /not protected against a restart/);
    assert.match(receiptNotice('unavailable'), /could not be sent/);
    assert.match(receiptNotice('origin_unproven'), /no recorded Main origin/);
    assert.match(receiptNotice(undefined), /unconfirmed/);
});
