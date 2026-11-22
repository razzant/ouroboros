import assert from 'node:assert/strict';
import test from 'node:test';
import { HOSTS, sequence } from './keyboard_sequence_fixture.js';

const PER_HOST = Number(process.env.KEYBOARD_SEQUENCES || 400);
const FIRST_SEED = Number(process.env.KEYBOARD_SEQUENCES_SEED || 1);

for (const kind of HOSTS) {
    test(`generated keyboard orders keep geometry and cosmetic invariants on ${kind} (${PER_HOST} seeds)`, () => {
        const failed = [];
        let checks = 0;
        for (let seed = FIRST_SEED; seed < FIRST_SEED + PER_HOST; seed++) {
            try {
                checks += sequence(kind, seed).checks;
            } catch (error) {
                failed.push({ seed, message: error.message });
            }
        }
        assert.ok(checks > 0);
        assert.deepEqual(failed.slice(0, 5), [], `${failed.length} of ${PER_HOST} ${kind} seeds failed`);
    });
}
