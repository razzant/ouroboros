import assert from 'node:assert/strict';
import test from 'node:test';

import { renderInstalledSkillCard, renderReviewFindingsList } from '../modules/skill_card_renderer.js';
import { preflightFailed, preflightFindingText } from '../modules/utils.js';

// #335: a deterministic preflight FAIL persists as review_status=pending; the
// card's offers for it are pinned in skill_card_renderer.test.js. Here: the typed
// fact and the human-readable diagnosis.

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

test('a gate without the preflight key is never treated as a failure', () => {
    // Absence propagation: older backends do not send preflight_failed.
    assert.equal(preflightFailed({ review_gate: { executable_review: false } }), false);
    assert.equal(preflightFailed({}), false);
    assert.equal(preflightFailed({ review_gate: { preflight_failed: true } }), true);
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
});

test('an unparseable preflight reason falls back to the raw rendering', () => {
    assert.equal(preflightFindingText({ item: 'skill_preflight', reason: 'not json' }), '');
    assert.equal(preflightFindingText({ item: 'other', reason: PREFLIGHT_REASON }), '');
    const skill = preflightSkill({
        review_findings: [{ item: 'skill_preflight', verdict: 'FAIL', reason: 'exploded' }],
    });
    // The danger tray quotes the first problem once; the findings list itself is built on open.
    const card = renderInstalledSkillCard(skill);
    assert.equal(card.split('exploded').length - 1, 1);
    assert.doesNotMatch(card, /<li>/);
    assert.match(renderReviewFindingsList(skill), /exploded/);
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

