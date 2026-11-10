// A deployment's trusted extensions on Skills (skill_card_renderer.js over the server's `trust` rows, present only with
// an execution domain): an extension the service does not run says so as a neutral fact, offers nothing that would not
// make it run, and an On copy can still be turned off; a delivered one names the organization, keeps its switch and
// review, and offers no removal, update or repair. Rows without `trust` stay exactly as they were.
import assert from 'node:assert/strict';
import test from 'node:test';

import { installedSkillModel, renderInstalledSkillCard } from '../modules/skill_card_renderer.js';

const UNLISTED = { state: 'untrusted', reason: 'not_delivered', administered: false,
    detail: 'Only extensions your organization’s administrator delivers run here, and this one isn’t among them.' };
const CHANGED = { state: 'untrusted', reason: 'changed', administered: true,
    detail: 'Its files differ from the version your organization’s administrator delivered, so it doesn’t run.' };
const DELIVERED = { state: 'trusted', administered: true };

function skill(overrides = {}) {
    return {
        name: 'audit_widget', type: 'extension', version: '0.2.0', description: 'Example extension',
        enabled: true, source: 'external', payload_root: 'skills/external/audit_widget', removal: 'delete',
        review_status: 'clean', review_stale: false, review_gate: { executable_review: true }, executable_review: true,
        permissions: ['tool'], grants: { all_granted: true }, owner_attestable: true, live_loaded: true,
        install: { update_available: { version: '0.3.0', digest: 'b'.repeat(64) }, release: { version: '0.2.0' },
            source: { id: 'hub', name: 'Hub' } },
        ...overrides,
    };
}

// What the service lists for an extension it does not run: not live, its reason, the owner's sentence.
const notRun = (trust, overrides = {}) => skill({ trust, live_loaded: false, live_reason: 'deployment_untrusted',
    load_error: trust.detail, review_status: 'pending', review_gate: { executable_review: false }, executable_review: false,
    grants: { all_granted: false, missing_keys: ['AUDIT_TOKEN'] }, ...overrides });
const switchOf = (html) => html.match(/<input type="checkbox" class="skills-toggle"[^>]*>/)[0];
const NOWHERE = ['data-skill-action="review"', 'data-skill-action="rereview"', 'data-skill-action="attest_review"',
    'data-skill-action="grant"', 'data-skill-action="repair"', 'data-skill-action="update"', 'data-skill-action="turn_on"'];

test('an extension the service does not run says so neutrally and offers nothing that would not make it run', () => {
    const off = notRun(UNLISTED, { enabled: false });
    const model = installedSkillModel(off);
    assert.match(model.chip, /data-tone="neutral"[^>]*>Doesn’t run here</);
    assert.equal(model.needs, false, 'the organization’s choice is not the owner’s move');
    assert.equal(model.lockReason, 'Only extensions your organization delivers run here.');
    const html = renderInstalledSkillCard(off);
    const trays = html.split('data-skill-trays>')[1];
    assert.match(trays, /Only extensions your organization’s administrator delivers run here, and this one isn’t among them\./);
    assert.match(trays, /New extensions and their versions come from your organization’s administrator\./);
    assert.doesNotMatch(trays, /data-tone="(warn|danger)"/);
    for (const offer of NOWHERE) assert.doesNotMatch(html, new RegExp(offer), offer);
    assert.doesNotMatch(html, /data-needs-you/);
    assert.match(switchOf(html), / disabled/, 'turning it on would be refused');
    assert.match(html, /data-skill-action="delete"/, 'a copy the employee installed may still be removed');
    assert.doesNotMatch(switchOf(renderInstalledSkillCard(notRun(UNLISTED, { enabled: true }))), / disabled/,
        'an On copy can still be turned off');
});

test('another copy of a delivered name says why and offers no removal either', () => {
    const html = renderInstalledSkillCard(notRun(CHANGED, { enabled: false }));
    assert.match(html, /Its files differ from the version your organization’s administrator delivered, so it doesn’t run\./);
    for (const offer of [...NOWHERE, 'data-skill-action="delete"', 'data-skill-action="uninstall"']) {
        assert.doesNotMatch(html, new RegExp(offer), offer);
    }
});

test('a delivered extension names the organization, keeps its switch and review, and offers no removal or update', () => {
    const delivered = skill({ trust: DELIVERED });
    const html = renderInstalledSkillCard(delivered);
    assert.match(html, /data-delivered-by-organization>Delivered by your organization</);
    assert.doesNotMatch(html, /Required by your organization/);
    assert.doesNotMatch(switchOf(html), / disabled/, 'Off stays the employee’s');
    for (const offer of ['data-skill-action="delete"', 'data-skill-action="uninstall"', 'data-skill-action="update"',
        'data-skill-action="repair"']) {
        assert.doesNotMatch(html, new RegExp(offer), offer);
    }
    assert.match(html, /data-skill-action="review"|data-skill-action="rereview"/, 'its review stays the employee’s');
    const model = installedSkillModel(delivered);
    assert.equal(model.update, false);
    assert.equal(model.needs, false);
    const required = renderInstalledSkillCard(skill({ trust: DELIVERED, deployment: { state: 'met', when_unmet: 'hold' } }));
    assert.match(required, /Required by your organization/);
    assert.doesNotMatch(required, /Delivered by your organization/, 'required says the stronger fact once');
});

test('rows without trust are rendered as before', () => {
    const personal = renderInstalledSkillCard(skill());
    assert.doesNotMatch(personal, /Doesn’t run here|Delivered by your organization/);
    assert.match(personal, /data-skill-action="delete"/);
    assert.match(personal, /data-skill-action="update"/);
    assert.equal(installedSkillModel(skill()).update, true);
});
