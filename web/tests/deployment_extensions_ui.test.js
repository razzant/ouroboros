// A deployment's required extensions on Skills and in Settings (skill_card_renderer.js, settings_policy.js over the
// server's requirement rows): a required card says so, keeps its switch and offers no removal, update or repair; an
// unmet requirement says why and what ordinary work does; a missing package is said above the list; a setting the
// organization fixes is named by the extension that requires it. Unmanaged rows (no `deployment`) stay as they were.
import assert from 'node:assert/strict';
import test from 'node:test';

import { installedSkillModel, renderInstalledSkillCard, renderRequirementNotice } from '../modules/skill_card_renderer.js';
import { organizationSettingOwner } from '../modules/settings_policy.js';

const PROCESS = { role: 'server', pid: 4242, started: 'Sun Nov  8 12:00:00 2026', observed_at: '2026-11-08T12:00:01Z' };
const MET = { name: 'error_tracker', path: '/srv/data/skills/external/error_tracker', content_hash: 'a'.repeat(64),
    settings: ['SENTRY_DSN'], state: 'met', reason: 'running', detail: 'Running in this process.', process: PROCESS,
    when_unmet: 'hold' };

function skill(overrides = {}) {
    return {
        name: 'error_tracker', type: 'extension', version: '0.2.0', description: 'Example error tracker',
        enabled: true, source: 'external', payload_root: 'skills/external/error_tracker', removal: 'delete',
        review_status: 'clean', review_stale: false, review_gate: { executable_review: true }, executable_review: true,
        permissions: ['net'], grants: { all_granted: true }, owner_attestable: true, live_loaded: true,
        install: { update_available: { version: '0.3.0', digest: 'b'.repeat(64) }, release: { version: '0.2.0' },
            source: { id: 'hub', name: 'Hub' } },
        ...overrides,
    };
}

const switchOf = (html) => html.match(/<input type="checkbox" class="skills-toggle"[^>]*>/)[0];

test('a required card names the organization, keeps its switch and offers no removal, update or repair', () => {
    const managed = renderInstalledSkillCard(skill({ deployment: MET }));
    assert.match(managed, /data-required-by-organization>Required by your organization</);
    assert.match(switchOf(managed), / checked /);
    assert.match(switchOf(managed), / disabled/, 'Off is the organization’s; the toggle refuses it too');
    for (const offer of ['data-skill-action="delete"', 'data-skill-action="uninstall"', 'data-skill-action="update"',
        'data-skill-action="repair"']) {
        assert.doesNotMatch(managed, new RegExp(offer), offer);
    }
    assert.doesNotMatch(managed, /data-required-unmet/, 'a met requirement needs no tray');
    assert.equal(installedSkillModel(skill({ deployment: MET })).needs, false);

    const personal = renderInstalledSkillCard(skill());
    assert.doesNotMatch(personal, /Required by your organization/);
    assert.doesNotMatch(switchOf(personal), / disabled/);
    assert.match(personal, /data-skill-action="delete"/);
    assert.match(personal, /data-skill-action="update"/);
});

test('an unmet requirement says why first, and whether ordinary work waits or goes on', () => {
    const unmet = { ...MET, state: 'unmet', reason: 'review', detail: 'It needs a review or your attestation of these exact bytes.' };
    const held = skill({ deployment: unmet, review_status: 'pending', review_gate: { executable_review: false }, executable_review: false });
    const html = renderInstalledSkillCard(held);
    const trays = html.split('data-skill-trays>')[1];
    assert.ok(trays.indexOf('data-required-unmet') < trays.indexOf('Ouroboros hasn’t reviewed it yet'), 'the requirement comes first');
    assert.match(trays, /Your organization requires it, and it isn’t running: It needs a review or your attestation/);
    assert.match(trays, /Ordinary work waits until it runs\. Stop, Panic and Settings stay available\./);
    assert.match(trays, /data-skill-action="review"/, 'the owner’s own next step stays offered');
    assert.equal(installedSkillModel(held).needs, true);
    const warned = renderInstalledSkillCard(skill({ deployment: { ...unmet, when_unmet: 'warn' } }));
    assert.match(warned, /Work goes on while it isn’t running\./);
});

test('the list says every unmet requirement, a missing package included, and nothing when all run', () => {
    const deployment = { when_unmet: 'hold', required: [MET, { ...MET, name: 'audit_log', state: 'unmet', reason: 'missing',
        detail: 'No package named audit_log is installed.' }] };
    const notice = renderRequirementNotice(deployment);
    assert.match(notice, /Your organization requires an extension that isn’t running here\./);
    assert.match(notice, /<li data-required-name="audit_log"><code data-i18n-authored>audit_log<\/code> · No package named audit_log is installed\.<\/li>/);
    assert.doesNotMatch(notice, /data-required-name="error_tracker"/, 'a running one is not listed');
    assert.match(notice, /Seen by the server process; each worker checks its own before it starts work\./);
    assert.equal(renderRequirementNotice({ when_unmet: 'hold', required: [MET] }), '');
    assert.equal(renderRequirementNotice(null), '');
});

test('a setting the organization fixes is named by the extension that requires it, and no other is', () => {
    const deployment = { extensions: { when_unmet: 'hold', required: [MET] } };
    assert.equal(organizationSettingOwner(deployment, 'SENTRY_DSN'), 'error_tracker');
    assert.equal(organizationSettingOwner(deployment, 'SLACK_WEBHOOK_URL'), '');
    assert.equal(organizationSettingOwner(null, 'SENTRY_DSN'), '', 'unmanaged: nothing is fixed');
});
