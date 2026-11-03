import assert from 'node:assert/strict';
import test from 'node:test';
import {
    createMySkillsDestination, firstStartTrustHtml, sourceProbeWarning, sourceProblemsHtml, sourceTarget, sourceViewSentence, updateWithoutReview,
} from '../modules/skill_dialogs.js';
import { skillSourcesApi } from '../modules/api_client.js';

test('rejected update confirms its exact version and sends only that release after literal confirmation', async () => {
    const skill = { name: 'demo', install: { update_rejected: { version: '2.0', digest: 'd'.repeat(64), reason: 'review_blockers' },
        update_available: { version: '2.0', digest: 'd'.repeat(64) } } };
    for (const decision of [false, null, undefined, { confirmed: true }, true]) {
        const calls = [];
        const result = await updateWithoutReview(skill, {
            dialog: async (model) => {
                assert.match(model.title, /demo to 2\.0 without review/);
                assert.match(model.body, /its review found problems; the previous version runs\./);
                assert.equal(model.confirmLabel, 'Update without review');
                return decision;
            },
            update: async (...args) => { calls.push(args); return { ok: true }; },
        });
        assert.deepEqual(calls, decision === true ? [['demo', 'replace', 'd'.repeat(64), '', { skipReview: true }]] : []);
        assert.deepEqual(result, decision === true ? { ok: true } : null);
    }
});

test('owner skip needs a rejected digest and is hidden only for a known different candidate', async () => {
    for (const install of [
        { update_rejected: { digest: 'rejected' }, update_available: { digest: 'newer' } },
        { update_rejected: { version: '2.0' } }, {},
    ]) {
        await updateWithoutReview({ name: 'demo', install }, {
            dialog: () => assert.fail('no exact rejected release is offered'),
            update: () => assert.fail('nothing was confirmed'),
        });
    }
    for (const [reason, cause] of Object.entries({ review_unfinished: "its review didn't finish", preflight_failed: 'its file checks failed', landing_failed: "it couldn't be installed" })) {
        await updateWithoutReview({ name: 'demo', install: {
            update_rejected: { version: '2.0', digest: 'd', reason }, update_available: { digest: 'd' },
        } }, { dialog: async (model) => { assert.ok(model.body.includes(cause)); return false; } });
    }
});

test('owner skip with no catalog candidate confirms and sends the rejected release for registry and unread sources', async () => {
    for (const kind of ['registry', 'git']) {
        for (const candidate of [null, undefined]) {
            for (const confirmed of [true, false, null, { confirmed: true }]) {
                const calls = [];
                const result = await updateWithoutReview({ name: 'demo', install: {
                    source: { kind }, update_rejected: { version: '2.0', digest: 'rejected', reason: 'review_unfinished' },
                    ...(candidate === undefined ? {} : { update_available: candidate }),
                } }, {
                    dialog: async (model) => { assert.match(model.title, /demo to 2\.0 without review/); return confirmed; },
                    update: async (...args) => { calls.push(args); return { ok: true }; },
                });
                assert.deepEqual(calls, confirmed === true ? [['demo', 'replace', 'rejected', '', { skipReview: true }]] : []);
                assert.deepEqual(result, confirmed === true ? { ok: true } : null);
            }
        }
    }
});

test('every repository address form the host accepts is sent as an address, whatever its host; anything else is a folder', () => {
    for (const url of ['https://gitverse.ru/team/skills', 'http://git.local/x', 'ssh://git@git.example.org:2222/t/s.git',
        'git@gitverse.ru:team/skills.git', 'deploy@git.example.org:team/skills.git', 'file:///srv/git/skills.git', 'FILE:///x']) {
        assert.deepEqual(sourceTarget(`  ${url} `), { url });
    }
    for (const path of ['/Users/me/skills', '~/skills', 'C:\\skills', 'gitverse.ru/team/skills']) {
        assert.deepEqual(sourceTarget(path), { path });
    }
    assert.deepEqual(sourceTarget('git@gitverse.ru:team/skills.git', { revision: ' v1.2 ', subfolder: ' skills/team ' }),
        { url: 'git@gitverse.ru:team/skills.git', revision: 'v1.2', subfolder: 'skills/team' });
    assert.deepEqual(sourceTarget('/srv/skills', { revision: 'main', subfolder: '' }), { path: '/srv/skills', revision: 'main' },
        'a folder keeps what the owner typed so the host can refuse it by name');
});

test('the checked view says what will be read and whether it can update', () => {
    const commit = 'f'.repeat(40);
    assert.equal(sourceViewSentence({}, commit), 'Follows the default branch (now at fffffff).');
    assert.equal(sourceViewSentence({ revision: 'v1.2', subfolder: 'skills/team' }, commit), 'Follows v1.2 (now at fffffff), reading the folder skills/team.');
    assert.equal(sourceViewSentence({ revision: 'a'.repeat(40) }, 'a'.repeat(40)), 'Stays at commit aaaaaaa: its skills don’t update.');
});

test('problems are a warning count over the exact reasons, closed on a row and open where the owner decides', () => {
    assert.equal(sourceProblemsHtml([]), '');
    const html = sourceProblemsHtml(['a <b>: c'], { open: true });
    assert.match(html, /^<details class="skills-source-problems" open><summary><span class="ui-status skills-status" data-tone="warn">/);
    assert.match(html, /1 problem<\/span>/);
    assert.match(html, /<li>a &lt;b&gt;: c<\/li>/);
    assert.doesNotMatch(sourceProblemsHtml(['x', 'y']), / open>/);
});

test('an unencrypted probe discloses the warning; a regular probe has none', () => {
    assert.equal(sourceProbeWarning({ warning: 'not_encrypted' }), 'This address is not encrypted: anyone on the network path could change what you install.');
    assert.equal(sourceProbeWarning({ kind: 'git' }), '');
});

test('first-start trust discloses that suggestions apply and keeps either supplied choice selected', () => {
    for (const trust of ['trusted', 'review']) {
        const html = firstStartTrustHtml([{ id: 'hub', name: 'Team <hub>', trust }]);
        assert.match(html, /Suggested choices apply until you choose\./);
        assert.match(html, new RegExp(`aria-checked="true" data-trust="${trust}"`));
        assert.equal((html.match(/aria-checked="true"/g) || []).length, 1);
        assert.match(html, /Team &lt;hub&gt;/);
    }
});

test('My skills selects the source from both create and connect answers', async () => {
    for (const created of [false, true]) {
        for (const privateValue of [true, false, undefined, 'true']) {
            const choice = await createMySkillsDestination(async () => ({ ok: true, created, private: privateValue,
                repository: 'owner/ouroboros-skills', source: { id: 'owner-skills', name: 'My renamed skills' } }));
            const visibility = privateValue === true ? ' (private)' : privateValue === false ? ' (public)' : '';
            assert.equal(choice.value, 'source:owner-skills');
            assert.equal(choice.private, typeof privateValue === 'boolean' ? privateValue : undefined);
            assert.equal(choice.repository, 'owner/ouroboros-skills');
            assert.match(choice.label, /My renamed skills/);
            assert.ok(choice.label.includes(`owner/ouroboros-skills${visibility}`));
            assert.equal(choice.status, `${created ? 'Created' : 'Connected'} owner/ouroboros-skills${visibility}.`);
        }
    }
});

test('My skills failures give the next step for typed refusals and network failures', async () => {
    const messages = { no_token: /Add GITHUB_TOKEN in Settings/, cannot_create: /allow repository access/, failed: /check your GitHub connection/ };
    for (const [code, expected] of Object.entries(messages)) {
        await assert.rejects(createMySkillsDestination(async () => ({ ok: false, code })), expected);
        await assert.rejects(createMySkillsDestination(async () => { throw { body: { code } }; }), expected);
    }
    await assert.rejects(createMySkillsDestination(async () => { throw new Error('offline'); }), /check your GitHub connection/);
});

test('owner skip and My skills API calls use the shared HTTP contract', async () => {
    const before = globalThis.fetch;
    const calls = [];
    globalThis.fetch = async (url, init) => { calls.push([url, JSON.parse(init.body)]); return { ok: true, json: async () => ({ ok: true }) }; };
    try {
        await skillSourcesApi.mySkills();
        await skillSourcesApi.updateSkill('demo', 'replace', 'a'.repeat(64), '', { skipReview: true });
        await skillSourcesApi.updateSkill('demo', 'replace', 'b'.repeat(64));
        assert.deepEqual(calls, [
            ['/api/skill-sources/my-skills', {}],
            ['/api/skills/demo/update', { mode: 'replace', digest: 'a'.repeat(64), skip_review: true }],
            ['/api/skills/demo/update', { mode: 'replace', digest: 'b'.repeat(64) }],
        ]);
    } finally { globalThis.fetch = before; }
});
