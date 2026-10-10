import assert from 'node:assert/strict';
import test from 'node:test';

import { renderInstalledSkillCard, renderReviewFindingsList } from '../modules/skill_card_renderer.js';
import { preflightFailed, preflightFindingText, topReviewFinding } from '../modules/utils.js';

// #335: a deterministic preflight FAIL persists as review_status=pending.
// Re-review would deterministically fail again — the card must offer Repair.

const PREFLIGHT_REASON = JSON.stringify({
    skill: 'minecraft-widget',
    manifest: [
        { item: 'manifest_present', ok: true },
        { item: 'manifest_entry_exists', ok: false, detail: 'missing or escaping entry: plugin.py' },
    ],
    files: [],
    ok: false,
});

function preflightSkill(overrides = {}) {
    return {
        name: 'minecraft-widget',
        type: 'extension',
        version: '1.0.0',
        enabled: false,
        source: 'external',
        payload_root: 'skills/external/minecraft-widget',
        review_status: 'pending',
        review_stale: false,
        review_gate: { executable_review: false, preflight_failed: true, blocking_reason: 'review_pending' },
        executable_review: false,
        review_findings: [{
            item: 'skill_preflight', verdict: 'FAIL', severity: 'critical',
            reason: PREFLIGHT_REASON, model: 'deterministic_preflight',
        }],
        grants: { all_granted: true, missing_keys: [], missing_permissions: [] },
        permissions: [],
        ...overrides,
    };
}

test('preflight-failed external skill offers Repair as the primary action', () => {
    const html = renderInstalledSkillCard(preflightSkill());
    assert.match(html, /data-skill-action="repair"[^>]*>Repair and run</);
    assert.doesNotMatch(html, />Re-review</);
});

test('preflight-failed self-authored skill offers Repair too', () => {
    const html = renderInstalledSkillCard(preflightSkill({
        source: 'self_authored',
        is_self_authored: true,
    }));
    assert.match(html, /data-skill-action="repair"[^>]*>Repair and run</);
});

test('plain pending without a preflight failure keeps the Review CTA', () => {
    const html = renderInstalledSkillCard(preflightSkill({
        review_gate: { executable_review: false, blocking_reason: 'review_pending' },
        review_findings: [],
    }));
    assert.match(html, />Review</);
    assert.doesNotMatch(html, /data-skill-action="repair"[^>]*>Repair and run</);
});

test('a gate without the preflight key is never treated as a failure', () => {
    // Absence propagation: older backends do not send preflight_failed.
    assert.equal(preflightFailed({ review_gate: { executable_review: false } }), false);
    assert.equal(preflightFailed({}), false);
    assert.equal(preflightFailed({ review_gate: { preflight_failed: true } }), true);
});

test('Skip review is hidden when the preflight failed (it would 409)', () => {
    const html = renderInstalledSkillCard(preflightSkill({
        source: 'self_authored',
        is_self_authored: true,
    }));
    assert.doesNotMatch(html, /Skip review/);
    // ...and stays offered for the same skill without the preflight failure.
    const clean = renderInstalledSkillCard(preflightSkill({
        source: 'self_authored',
        is_self_authored: true,
        review_gate: { executable_review: false, blocking_reason: 'review_pending' },
        review_findings: [],
    }));
    assert.match(clean, /Skip review/);
});

test('the preflight finding renders a human-readable diagnosis, not raw JSON', () => {
    // The card carries the collapsed findings block; its list is built when opened.
    const html = renderInstalledSkillCard(preflightSkill());
    assert.match(html, /data-skill-findings="[^"]+"><summary[^>]*>1 review finding<\/summary><\/details>/);
    const opened = renderReviewFindingsList(preflightSkill());
    assert.match(opened, /missing or escaping entry: plugin\.py/);
    assert.doesNotMatch(opened, /manifest_present/); // passing rows are not noise

    const text = preflightFindingText({
        item: 'skill_preflight', verdict: 'FAIL', reason: PREFLIGHT_REASON,
    });
    assert.match(text, /^Preflight failed — manifest_entry_exists: missing or escaping entry: plugin\.py$/);

    // topReviewFinding (the marketplace hint) uses the same diagnosis.
    assert.match(topReviewFinding(preflightSkill()), /missing or escaping entry/);
});

test('an unparseable preflight reason falls back to the raw rendering', () => {
    assert.equal(preflightFindingText({ item: 'skill_preflight', reason: 'not json' }), '');
    assert.equal(preflightFindingText({ item: 'other', reason: PREFLIGHT_REASON }), '');
    const skill = preflightSkill({
        review_findings: [{ item: 'skill_preflight', verdict: 'FAIL', reason: 'exploded' }],
    });
    assert.doesNotMatch(renderInstalledSkillCard(skill), /exploded/); // built on open, not per render
    assert.match(renderReviewFindingsList(skill), /exploded/);
});

test('non-repairable native source keeps the Review CTA on a preflight failure', () => {
    const html = renderInstalledSkillCard(preflightSkill({
        source: 'native',
        payload_root: '',
    }));
    assert.match(html, />Review</);
    assert.doesNotMatch(html, /data-skill-action="repair"[^>]*>Repair and run</);
});

test('the status chip names the preflight failure instead of generic Needs review', () => {
    const html = renderInstalledSkillCard(preflightSkill());
    assert.match(html, />Preflight failed</);
    assert.doesNotMatch(html, />Needs review</);
});

test('the diagnosis covers widget, permission, and presence failures too', () => {
    const reason = JSON.stringify({
        manifest: [{ item: 'manifest_present', ok: true }],
        widgets: [{ item: 'widget_schema', ok: false, detail: 'bad schema: x' }],
        permissions: [{ item: 'permission_static', ok: false, detail: 'register_tool without tool permission' }],
        presence: [{ item: 'presence_profile', ok: false, detail: 'unknown slot' }],
        files: [],
        ok: false,
    });
    const text = preflightFindingText({ item: 'skill_preflight', verdict: 'FAIL', reason });
    assert.match(text, /widget_schema: bad schema: x/);
    assert.match(text, /permission_static: register_tool without tool permission/);
    // Three parts shown; the cap keeps the line readable.
    assert.match(text, /^Preflight failed — /);
});

test('a stale recorded FAIL keeps Re-review primary and offers Repair from the menu (D11)', () => {
    // The backend gate reports preflight_failed=false for a stale state (the
    // persisted failure belongs to the previous payload bytes) and carries the
    // typed preflight_failed_stale companion — the card keeps the cheap
    // Re-review primary, re-surfaces owner attestation, and additionally
    // offers Repair based on the last recorded preflight.
    const html = renderInstalledSkillCard(preflightSkill({
        source: 'self_authored',
        is_self_authored: true,
        review_stale: true,
        review_gate: {
            executable_review: false, preflight_failed: false,
            preflight_failed_stale: true, stale: true,
        },
    }));
    assert.match(html, /skills-primary-action[^>]*data-skill-action="rereview"[^>]*>Re-review</);
    assert.doesNotMatch(html, /skills-primary-action[^>]*>Repair and run</);
    assert.match(html, /skills-menu-item skills-repair-stale[^>]*data-skill-action="repair"[^>]*>Repair and run</);
    assert.match(html, /based on the last recorded preflight/);
    assert.match(html, /Skip review/);
});

test('a stale review without a recorded FAIL offers no Repair anywhere', () => {
    const html = renderInstalledSkillCard(preflightSkill({
        source: 'self_authored',
        is_self_authored: true,
        review_stale: true,
        review_gate: {
            executable_review: false, preflight_failed: false,
            preflight_failed_stale: false, stale: true,
        },
        review_findings: [],
    }));
    assert.match(html, />Re-review</);
    assert.doesNotMatch(html, /data-skill-action="repair"[^>]*>Repair and run</);
});

test('self-authored instruction skill offers Make runnable in the menu', () => {
    const html = renderInstalledSkillCard(preflightSkill({
        type: 'instruction',
        source: 'self_authored',
        is_self_authored: true,
        review_gate: { executable_review: false },
        review_findings: [],
    }));
    assert.match(html, /Make runnable/);
});

test('menu Review is hidden on a fresh repairable preflight failure (marketplace parity)', () => {
    const html = renderInstalledSkillCard(preflightSkill());
    assert.doesNotMatch(html, /skills-menu-item skills-review"/);
    // Non-repairable native keeps its Review affordance.
    const native = renderInstalledSkillCard(preflightSkill({ source: 'native', payload_root: '' }));
    assert.match(native, /skills-menu-item skills-review"/);
});
