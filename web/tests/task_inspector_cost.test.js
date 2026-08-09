import assert from 'node:assert/strict';
import test from 'node:test';

import {
    formatElapsed,
    inspectorCostRows,
    inspectorFooter,
    taskIsRunning,
} from '../modules/task_inspector.js';

const row = (rows, label) => rows.find((item) => item.label === label);

test('a settled ledger renders the exact final cost', () => {
    const rows = inspectorCostRows({
        cost_accounting_status: 'available',
        cost_final: true,
        cost_usd: 0.0712,
        total_rounds: 5,
        prompt_tokens: 12000,
        completion_tokens: 900,
    });
    assert.equal(row(rows, 'Task cost').value, '$0.07');
    assert.equal(row(rows, 'Task cost').tone, 'final');
    assert.equal(row(rows, 'LLM rounds').value, '5');
    assert.equal(row(rows, 'Tokens in/out').value, '12000 / 900');
    assert.equal(row(rows, 'Accounting').value, 'final');
});

test('an unavailable ledger says Unavailable — never a convincing $0.00', () => {
    const rows = inspectorCostRows({
        cost_accounting_status: 'unavailable',
        cost_final: false,
        cost_usd: null,
        cost_accounting_error: 'ledger_unavailable',
    });
    assert.equal(row(rows, 'Task cost').value, 'Unavailable');
    assert.equal(row(rows, 'Task cost').tone, 'unavailable');
    assert.equal(row(rows, 'Accounting'), undefined);
    assert.ok(!rows.some((item) => item.value.includes('$0.00')));
});

test('a genuine zero cost is shown as $0.00 and is NOT confused with unavailable', () => {
    const rows = inspectorCostRows({ cost_accounting_status: 'available', cost_final: true, cost_usd: 0 });
    assert.equal(row(rows, 'Task cost').value, '$0.00');
    assert.equal(row(rows, 'Task cost').tone, 'final');
});

test('an unsettled reading is labelled pending', () => {
    const rows = inspectorCostRows({ cost_accounting_status: 'available', cost_final: false, cost_usd: 0.5 });
    assert.equal(row(rows, 'Task cost').value, '$0.50 (pending)');
    assert.equal(row(rows, 'Accounting').value, 'pending');
});

test('a record with no accounting evidence at all reports Unavailable', () => {
    const rows = inspectorCostRows({});
    assert.equal(rows.length, 1);
    assert.equal(rows[0].value, 'Unavailable');
    const declaredButEmpty = inspectorCostRows({ cost_usd: null });
    assert.equal(declaredButEmpty[0].value, 'Pending');
});

test('optional money fields appear only when they carry a real positive value', () => {
    const bare = inspectorCostRows({ cost_accounting_status: 'available', cost_final: true, cost_usd: 1 });
    for (const label of ['Reserved', 'Unresolved ≤', 'Unmetered calls']) {
        assert.equal(row(bare, label), undefined);
    }
    const full = inspectorCostRows({
        cost_accounting_status: 'available',
        cost_final: false,
        cost_usd: 1,
        reserved_usd: 0.25,
        unresolved_upper_bound_usd: 2.5,
        unknown_unmetered: 3,
    });
    assert.equal(row(full, 'Reserved').value, '$0.25');
    assert.equal(row(full, 'Unresolved ≤').value, '$2.50');
    assert.equal(row(full, 'Unmetered calls').value, '3');
    // Zeros are not noise-worthy rows, but they are also never invented.
    const zeros = inspectorCostRows({ cost_usd: 1, reserved_usd: 0, unknown_unmetered: 0 });
    assert.equal(row(zeros, 'Reserved'), undefined);
    assert.equal(row(zeros, 'Unmetered calls'), undefined);
});

test('token rows survive a half-known pair without fabricating the other half', () => {
    const rows = inspectorCostRows({ prompt_tokens: 42, completion_tokens: null });
    assert.equal(row(rows, 'Tokens in/out').value, '42 / —');
    assert.equal(row(inspectorCostRows({ completion_tokens: 7 }), 'Tokens in/out').value, '— / 7');
    assert.equal(row(inspectorCostRows({}), 'Tokens in/out'), undefined);
});

test('the cost tab exposes ONLY persisted decision-34 fields', () => {
    const rows = inspectorCostRows({
        cost_accounting_status: 'available',
        cost_final: true,
        cost_usd: 1,
        total_rounds: 2,
        prompt_tokens: 1,
        completion_tokens: 1,
        reserved_usd: 1,
        unresolved_upper_bound_usd: 1,
        unknown_unmetered: 1,
        // Fields deliberately NOT in v1: per-model rows and tool-call totals.
        per_model: [{ model: 'x', cost_usd: 1 }],
        tool_calls: 12,
    });
    const labels = rows.map((item) => item.label);
    assert.deepEqual(labels, [
        'Task cost', 'LLM rounds', 'Tokens in/out', 'Reserved', 'Unresolved ≤',
        'Unmetered calls', 'Accounting',
    ]);
});

test('elapsed formats real durations and refuses to invent a zero', () => {
    assert.equal(formatElapsed(4.52), '4.5s');
    assert.equal(formatElapsed(42.4), '42s');
    assert.equal(formatElapsed(72), '1m 12s');
    assert.equal(formatElapsed(3725), '1h 2m');
    assert.equal(formatElapsed(0), '0.0s');
    assert.equal(formatElapsed(null), '');
    assert.equal(formatElapsed(undefined), '');
    assert.equal(formatElapsed('nonsense'), '');
    assert.equal(formatElapsed(-5), '');
});

test('the footer combines parsed counts, cost and elapsed honestly', () => {
    const parsed = { files: [{}, {}], added: 38, removed: 12 };
    const footer = inspectorFooter({
        cost_accounting_status: 'available', cost_final: true, cost_usd: 0.42, duration_sec: 95,
    }, parsed, { status: 'ready' });
    assert.deepEqual(footer, {
        added: 38, removed: 12, files: 2, cost: '$0.42', elapsed: '1m 35s',
    });

    const unknown = inspectorFooter({ cost_accounting_status: 'unavailable' }, null, null);
    assert.deepEqual(unknown, {
        added: null, removed: null, files: null, cost: 'Unavailable', elapsed: 'Unavailable',
    });
});

test('footer counts are NULL unless a ready diff licenses a number', () => {
    // U2: `+0 −0` is the claim "this task changed nothing". For a diff that is
    // pending, blocked, or simply not fetched yet the truth is "we do not know",
    // and the two must never render as the same thing.
    const task = { cost_accounting_status: 'available', cost_final: true, cost_usd: 1, duration_sec: 5 };
    const empty = { files: [], added: 0, removed: 0 };
    for (const diff of [null, { status: 'pending' }, { status: 'blocked' }, { status: '' }]) {
        const footer = inspectorFooter(task, empty, diff);
        assert.equal(footer.added, null, JSON.stringify(diff));
        assert.equal(footer.removed, null);
        assert.equal(footer.files, null);
    }
    // A READY diff that really is empty says so with real zeroes.
    const ready = inspectorFooter(task, empty, { status: 'ready' });
    assert.deepEqual([ready.added, ready.removed, ready.files], [0, 0, 0]);
    // Parsed bytes with no status behind them cannot count either.
    assert.equal(inspectorFooter(task, { files: [{}], added: 3, removed: 0 }).added, null);
});

test('the inspector fetches the diff on open/tab/terminal only, never per state tick', async () => {
    // U3: the state poll runs every few seconds and the diff endpoint forks git on
    // the server, so the tick refreshes the task RECORD only. Pinned in the source
    // because the wiring is what carries the guarantee.
    const source = await import('node:fs/promises')
        .then((fs) => fs.readFile(new URL('../modules/task_inspector.js', import.meta.url), 'utf8'));
    const diffLoads = source.match(/load\([^)]*\{ diff: true \}\)/g) || [];
    // mount, the no-openRightPanel fallback, Changes-tab activation, terminal edge.
    assert.equal(diffLoads.length, 4);
    // The subscribeState callback's own load carries NO diff flag.
    assert.match(source, /subscribeState\(\(\) => \{[\s\S]*?load\(view\.taskId\);\s*\}\);/);
});

test('Changes-tab activation retries a BLOCKED diff, not only a missing one', async () => {
    // `blocked` covers transient conditions (a failed request, a projection that
    // moved under the read, a snapshot not recorded yet). Gating the refetch on
    // `!view.diff` alone left an open panel stuck on the first refusal until it was
    // closed and reopened; the never-per-state-tick rule above is unchanged.
    const source = await import('node:fs/promises')
        .then((fs) => fs.readFile(new URL('../modules/task_inspector.js', import.meta.url), 'utf8'));
    assert.match(
        source,
        /const retryable = !view\.diff \|\| String\(view\.diff\.status \|\| ''\) === 'blocked';/,
    );
    assert.match(source, /if \(activated && next === 'changes' && view\.taskId && retryable\)/);
});

test('drift and the missing-baseline sentence are read from the Changes module', async () => {
    // C3/C5 are ONE owner-facing rule each; the inspector imports both rather than
    // re-spelling them, so the two diff surfaces cannot drift apart.
    const source = await import('node:fs/promises')
        .then((fs) => fs.readFile(new URL('../modules/task_inspector.js', import.meta.url), 'utf8'));
    assert.match(
        source,
        /import \{ HEAD_DRIFT_NOTICE, NO_BASELINE_NOTICE, diffLacksBaselineOnly \} from '\.\/changes\.js'/,
    );
    assert.match(source, /head_advanced && String\(view\.diff\.source \|\| ''\) === 'mutation_baseline'/);
});

test('both diff surfaces disclose drift with the SAME owner-facing sentence', async () => {
    const { HEAD_DRIFT_NOTICE } = await import('../modules/changes.js');
    const source = await import('node:fs/promises')
        .then((fs) => fs.readFile(new URL('../modules/task_inspector.js', import.meta.url), 'utf8'));
    // The inspector imports the wording instead of re-spelling it (one fact, one place).
    assert.match(source, /^import \{ HEAD_DRIFT_NOTICE[^}]*\} from '\.\/changes\.js';$/m);
    assert.ok(!source.includes('HEAD differs from the task baseline;'));
    assert.equal(
        HEAD_DRIFT_NOTICE,
        'HEAD differs from the task baseline; showing the current projection for paths '
        + 'attributed during the task window',
    );
});

test('liveness follows the terminal status set', () => {
    assert.equal(taskIsRunning({ status: 'running' }), true);
    assert.equal(taskIsRunning({ status: 'scheduled' }), true);
    assert.equal(taskIsRunning({}), true);
    for (const status of ['completed', 'failed', 'cancelled', 'cancel_requested', 'rejected_duplicate']) {
        assert.equal(taskIsRunning({ status }), false, status);
    }
    assert.equal(taskIsRunning({ status: 'COMPLETED' }), false);
});
