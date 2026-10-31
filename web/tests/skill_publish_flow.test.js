import assert from 'node:assert/strict';
import test from 'node:test';

import { createTask, skillPublishPreflight } from '../modules/api_client.js';
import { renderConfirmDialogDetails } from '../modules/confirm_dialog.js';
import {
    buildSkillPublishTask,
    runSkillPublishFlow,
    skillPublishDetails,
    skillPublishDialogModel,
} from '../modules/skill_publish_flow.js';

function preflight(state = 'ready', overrides = {}) {
    return {
        ok: true,
        skill: 'Canonical-Skill',
        repository: 'Owner/Configured-Hub',
        state,
        publication_ready: state === 'ready' || state === 'warnings',
        task_start_allowed: state !== 'hard_block',
        snapshot_hash: 'abc123',
        review: { status: 'clean', stale: false, profile: 'full' },
        scanner: {
            status: 'ok',
            engine: 'betterleaks',
            version: '1.8.1',
            ruleset_sha256: 'rules-sha',
        },
        findings: [],
        omitted_count: 0,
        blocker_count: 0,
        warning_count: 0,
        audited_false_positive_count: 0,
        reason_code: '',
        summary: `${state} summary`,
        repair_hint: '',
        ...overrides,
    };
}

test('the browser renders all five backend-authored states without reclassifying reason_code', () => {
    const expected = {
        ready: { canStart: true, label: 'Publish' },
        warnings: { canStart: true, label: 'Publish with warnings' },
        needs_attention: { canStart: true, label: 'Ask Ouroboros to fix and publish' },
        repairable: { canStart: true, label: 'Ask Ouroboros to fix and publish' },
        hard_block: { canStart: false, label: 'OK' },
    };
    for (const [state, want] of Object.entries(expected)) {
        const model = skillPublishDialogModel(preflight(state, {
            reason_code: state === 'ready' ? 'scanner_missing' : 'arbitrary_reason',
        }));
        assert.equal(model.state, state);
        assert.equal(model.canStart, want.canStart);
        assert.equal(model.dialog.confirmLabel, want.label);
        assert.equal(model.dialog.alert, !want.canStart);
        assert.match(model.dialog.body, new RegExp(`${state} summary`));
    }
});

test('publication confirmation and task preserve public, private and unknown visibility', async () => {
    for (const privateRepository of [true, false, undefined, null]) {
        const disclosure = privateRepository === true
            ? "Everything in the skill's folder goes to your private repository."
            : privateRepository === false ? "Everything in the skill's folder becomes public."
                : "Everything in the skill's folder goes to Owner/Configured-Hub; whoever can read that repository can see it.";
        const result = await runSkillPublishFlow('demo', {
            privateRepository,
            destination: { source_id: 'my-skills', mode: 'entry' },
            preflightImpl: async () => preflight(),
            dialogImpl: async (model) => { assert.ok(model.body.includes(disclosure), model.body); return true; },
            createTaskImpl: async () => ({ ok: true }),
        });
        assert.equal(result.started, true);
        for (const field of ['description', 'constraints', 'context']) {
            assert.ok(result.payload[field].includes(disclosure), `${field}: ${result.payload[field]}`);
            assert.equal(result.payload[field].includes('becomes public'), privateRepository === false);
            assert.equal(result.payload[field].includes('private repository'), privateRepository === true);
        }
        assert.deepEqual(result.payload.metadata.skill_publish_target.destination, { source_id: 'my-skills', mode: 'entry' },
            'visibility copy adds no fields to the shared destination contract');
    }
});

test('the known public hub is disclosed as public without guessing visibility for other repositories', () => {
    for (const repository of ['razzant/OuroborosHub', 'https://github.com/razzant/OuroborosHub.git/', 'owner/private-hub']) {
        const model = skillPublishDialogModel(preflight('ready', { repository }));
        assert.equal(model.dialog.body.includes('becomes public'), repository !== 'owner/private-hub');
        assert.equal(model.dialog.body.includes('whoever can read that repository'), repository === 'owner/private-hub');
    }
});

test('a typed hard block renders and starts no task even without a canonical target', async () => {
    const blocked = preflight('hard_block', {
        ok: false,
        skill: '',
        repository: '',
        task_start_allowed: false,
        reason_code: 'authority_missing',
    });
    const dialogs = [];
    let creates = 0;
    const result = await runSkillPublishFlow('requested', {
        preflightImpl: async () => blocked,
        dialogImpl: async (options) => { dialogs.push(options); return true; },
        createTaskImpl: async () => { creates += 1; },
    });
    assert.equal(result.started, false);
    assert.equal(creates, 0);
    assert.equal(dialogs.length, 1);
    assert.equal(dialogs[0].title, 'Can’t publish');
    assert.equal(dialogs[0].alert, true);
});

test('explicit confirmation creates exactly one ordinary skill_publish task from canonical response fields', async () => {
    const calls = { preflight: [], dialog: [], create: [] };
    const selected = preflight('needs_attention', {
        skill: 'Canonical-Case',
        repository: 'ConfiguredOwner/ConfiguredRepo',
        publication_ready: false,
        task_start_allowed: true,
        blocker_count: 2,
        reason_code: 'review_blockers',
    });
    const result = await runSkillPublishFlow('stale-list-name', {
        preflightImpl: async (skill) => { calls.preflight.push(skill); return selected; },
        dialogImpl: async (options) => { calls.dialog.push(options); return true; },
        createTaskImpl: async (payload) => {
            calls.create.push(payload);
            return { ok: true, task_id: 'task-1', status: 'queued' };
        },
    });

    assert.equal(result.started, true);
    assert.deepEqual(calls.preflight, ['stale-list-name']);
    assert.equal(calls.dialog.length, 1);
    assert.equal(calls.create.length, 1);
    assert.equal(calls.dialog[0].confirmLabel, 'Ask Ouroboros to fix and publish');

    const payload = calls.create[0];
    assert.equal(payload.type, 'skill_publish');
    assert.equal(payload.source, 'web');
    assert.equal(payload.chat_id, 1);
    assert.equal(Object.hasOwn(payload, 'project_id'), false);
    assert.deepEqual(payload.metadata, {
        skill_publish_target: {
            skill: 'Canonical-Case',
            repository: 'ConfiguredOwner/ConfiguredRepo',
        },
    });
    assert.match(payload.expected_output, /Return only the validated GitHub pull-request URL/);
    assert.match(payload.constraints, /repository-provided checksum-pinned Betterleaks 1\.8\.1 installer/);
    assert.match(payload.constraints, /python -m ouroboros\.betterleaks_runtime install/);
    assert.match(payload.constraints, /No account or authentication change is authorized/);
    assert.match(payload.constraints, /missing or corrupt/);
    assert.match(payload.context, /explicitly confirmed publication/);
    assert.match(payload.context, /evidence for the next LLM turn/);
    assert.match(payload.context, /"state":"needs_attention"/);
    assert.match(payload.context, /"reason_code":"review_blockers"/);
    for (const forbidden of [
        'acceptance_claims',
        'workspace_root',
        'priority',
        'task_constraint',
        'skill_repair',
    ]) {
        assert.equal(Object.hasOwn(payload, forbidden), false, `${forbidden} must stay absent`);
    }
    assert.doesNotMatch(JSON.stringify(payload), /refusal.{0,20}success/i);
});

test('an explicitly bound Project keeps publication in its existing thread', async () => {
    const result = await runSkillPublishFlow('skill', {
        projectId: 'release-room',
        preflightImpl: async () => preflight(),
        dialogImpl: async () => true,
        createTaskImpl: async () => ({ task_id: 'project-publish' }),
    });
    assert.equal(result.payload.source, 'web');
    assert.equal(result.payload.project_id, 'release-room');
    assert.equal(Object.hasOwn(result.payload, 'chat_id'), false, 'admission resolves the registered room');
    const explicit = buildSkillPublishTask(preflight(), { projectId: 'release-room', chatId: 1042 });
    assert.equal(explicit.chat_id, 1042);
});

test('cancel, backdrop, Escape, and truthy glitches create zero tasks', async () => {
    for (const resolution of [false, undefined, null, 'true', 1, { confirmed: true }]) {
        let preflights = 0;
        let creates = 0;
        const result = await runSkillPublishFlow('skill', {
            preflightImpl: async () => { preflights += 1; return preflight('ready'); },
            dialogImpl: async () => resolution,
            createTaskImpl: async () => { creates += 1; },
        });
        assert.equal(result.started, false, `resolution ${JSON.stringify(resolution)} must cancel`);
        assert.equal(preflights, 1, 'selected preflight must run exactly once');
        assert.equal(creates, 0, 'cancel/glitch must start zero tasks');
    }
});

test('repairable scanner failure still starts one ordinary task after confirmation', async () => {
    let payload = null;
    const result = await runSkillPublishFlow('skill', {
        preflightImpl: async () => preflight('repairable', {
            publication_ready: false,
            task_start_allowed: true,
            scanner: { status: 'missing', engine: 'betterleaks', version: '', ruleset_sha256: '' },
            reason_code: 'scanner_missing',
            repair_hint: 'Use the repository-pinned recovery surface.',
        }),
        dialogImpl: async () => true,
        createTaskImpl: async (created) => { payload = created; return { task_id: 'repair-1' }; },
    });
    assert.equal(result.started, true);
    assert.equal(payload.type, 'skill_publish');
    assert.equal(Object.hasOwn(payload, 'task_constraint'), false);
});

test('known broken skill with empty capture facts remains an ordinary needs_attention task', async () => {
    let payload = null;
    const result = await runSkillPublishFlow('broken-skill', {
        preflightImpl: async () => preflight('needs_attention', {
            publication_ready: false,
            task_start_allowed: true,
            snapshot_hash: '',
            review: {},
            scanner: {},
            findings: [],
            reason_code: 'snapshot_payload_unreadable',
            repair_hint: 'The selected skill payload needs agent attention.',
        }),
        dialogImpl: async () => true,
        createTaskImpl: async (created) => { payload = created; return { task_id: 'repair-2' }; },
    });
    assert.equal(result.started, true);
    assert.equal(payload.type, 'skill_publish');
    assert.match(payload.context, /snapshot_payload_unreadable/);
});

test('typed non-2xx hard-block body is shown and never posted as a task', async () => {
    const error = new Error('blocked');
    error.body = preflight('hard_block', {
        ok: false,
        task_start_allowed: false,
        publication_ready: false,
        reason_code: 'source_unsupported',
    });
    let creates = 0;
    const result = await runSkillPublishFlow('skill', {
        preflightImpl: async () => { throw error; },
        dialogImpl: async () => true,
        createTaskImpl: async () => { creates += 1; },
    });
    assert.equal(result.started, false);
    assert.equal(creates, 0);
});

test('ok:false ready envelope alerts, creates no task, and cannot bypass through the builder', async () => {
    const failed = preflight('ready', { ok: false, task_start_allowed: true });
    const model = skillPublishDialogModel(failed);
    assert.equal(model.canStart, false);
    assert.equal(model.dialog.alert, true);
    assert.equal(model.dialog.title, 'Publish preflight unavailable');
    assert.throws(
        () => buildSkillPublishTask(failed),
        /does not authorize a task start/,
    );

    let creates = 0;
    const result = await runSkillPublishFlow('skill', {
        preflightImpl: async () => failed,
        dialogImpl: async () => true,
        createTaskImpl: async () => { creates += 1; },
    });
    assert.equal(result.started, false);
    assert.equal(creates, 0);
});

test('preflight details preserve safe finding fields and shared dialog escapes every value', () => {
    const hostile = '<script>alert("candidate")</script>';
    const response = preflight('warnings', {
        skill: hostile,
        repository: `Owner/${hostile}`,
        summary: hostile,
        repair_hint: hostile,
        findings: [{
            path: `outer!archive/${hostile}`,
            line: 7,
            detector: hostile,
            confidence: 'medium',
            reason: hostile,
            verification: 'not_attempted',
            disposition: 'warning',
        }],
    });
    const details = skillPublishDetails(response);
    const html = renderConfirmDialogDetails(details);
    assert.doesNotMatch(html, /<script>/);
    assert.match(html, /&lt;script&gt;/);
    assert.match(html, /outer!archive/);
    assert.match(html, /not_attempted/);
    assert.match(html, /warning/);
});

test('named API functions use the encoded selected route and ordinary task route once each', async () => {
    const originalFetch = globalThis.fetch;
    const calls = [];
    globalThis.fetch = async (url, init) => {
        calls.push({ url, init });
        return {
            ok: true,
            status: 200,
            json: async () => ({ ok: true, skill: 'A/B', repository: 'Owner/Repo' }),
        };
    };
    try {
        await skillPublishPreflight('A/B');
        await createTask(buildSkillPublishTask(preflight('ready')));
    } finally {
        globalThis.fetch = originalFetch;
    }
    assert.equal(calls.length, 2);
    assert.equal(calls[0].url, '/api/skills/A%2FB/publish-preflight');
    assert.equal(calls[1].url, '/api/tasks');
    assert.equal(calls[0].init.method, 'POST');
    assert.equal(calls[1].init.method, 'POST');
});

test('the destination the owner chose reaches the preflight and the task, and is absent when none was chosen', async () => {
    const destination = { source_id: 'ouroboros-community', mode: 'new_entry' };  // SkillPublishDestinationChoice
    const calls = { preflight: [], create: [] };
    const result = await runSkillPublishFlow('demo', {
        destination,
        preflightImpl: async (...args) => { calls.preflight.push(args); return preflight('ready', { skill: 'demo', repository: 'razzant/OuroborosHub' }); },
        dialogImpl: async () => true,
        createTaskImpl: async (payload) => { calls.create.push(payload); return { task_id: 't1' }; },
    });
    assert.equal(result.started, true);
    assert.deepEqual(calls.preflight, [['demo', destination]]);
    assert.deepEqual(calls.create[0].metadata.skill_publish_target, { skill: 'demo', repository: 'razzant/OuroborosHub', destination });
    // The publish tool reads the owner's choice from the task's target; the prose names only skill and repository.
    assert.doesNotMatch(calls.create[0].context, /destination/);

    const plain = { preflight: [] };
    await runSkillPublishFlow('demo', {
        preflightImpl: async (...args) => { plain.preflight.push(args); return preflight('ready', { skill: 'demo', repository: 'r/r' }); },
        dialogImpl: async () => false,
        createTaskImpl: async () => { throw new Error('cancelled flows start nothing'); },
    });
    assert.deepEqual(plain.preflight, [['demo']], 'without a chosen destination the request is unchanged');

    const originalFetch = globalThis.fetch;
    const bodies = [];
    globalThis.fetch = async (url, init) => { bodies.push([url, init.body]); return { ok: true, status: 200, json: async () => ({ ok: true }) }; };
    try {
        await skillPublishPreflight('demo', destination);
        await skillPublishPreflight('demo');
    } finally {
        globalThis.fetch = originalFetch;
    }
    assert.deepEqual(bodies, [
        ['/api/skills/demo/publish-preflight', JSON.stringify({ destination })],
        ['/api/skills/demo/publish-preflight', '{}'],
    ]);
});
