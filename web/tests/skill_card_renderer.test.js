import assert from 'node:assert/strict';
import test from 'node:test';

import { renderInstalledSkillCard } from '../modules/skill_card_renderer.js';

function skill(overrides = {}) {
    return {
        name: 'telegram',
        type: 'extension',
        version: '1.0.0',
        description: 'Telegram',
        enabled: false,
        source: 'native',
        review_status: 'clean',
        review_stale: false,
        review_gate: { executable_review: true },
        executable_review: true,
        grants: { all_granted: true, missing_keys: [], missing_permissions: [] },
        permissions: [],
        ...overrides,
    };
}

test('extension registration status says Loaded, not Active', () => {
    const html = renderInstalledSkillCard(skill({
        enabled: true,
        live_loaded: true,
        dispatch_live: true,
    }));

    assert.match(html, />Loaded</);
    assert.doesNotMatch(html, />Active</);
});

test('extension live state shows its answering process qualifier', () => {
    const html = renderInstalledSkillCard(skill({
        enabled: true,
        live_loaded: true,
        dispatch_live: true,
        process: 'worker',
    }));

    assert.match(html, />Loaded · worker</);
});

test('disabled conflicting skill is explained and cannot be enabled', () => {
    const html = renderInstalledSkillCard(skill({
        conflict: { code: 'skill_conflict', skills: ['telegram-bridge'], omitted: 0 },
    }));

    assert.match(html, /Conflicts with telegram-bridge/);
    assert.match(html, /Locked: conflicts with telegram-bridge/);
    assert.match(html, /class="skills-toggle"[^>]*disabled/);
});

test('enabled conflicting skill can still be disabled', () => {
    const html = renderInstalledSkillCard(skill({
        enabled: true,
        conflict: { code: 'skill_conflict', skills: ['telegram-bridge'], omitted: 0 },
    }));

    assert.match(html, /Conflicts with telegram-bridge/);
    assert.doesNotMatch(html, /class="skills-toggle"[^>]*disabled/);
});

test('conflict support preserves the existing Grant access action', () => {
    const html = renderInstalledSkillCard(skill({
        grants: {
            all_granted: false,
            requested_keys: ['TELEGRAM_BOT_TOKEN'],
            missing_keys: ['TELEGRAM_BOT_TOKEN'],
            missing_permissions: [],
        },
    }));

    assert.match(html, />Grant access</);
});

test('passive publish affordance trusts backend task_start_allowed independently of readiness', () => {
    const html = renderInstalledSkillCard(skill({
        source: 'external',
        payload_root: 'skills/external/telegram',
        submit_hub: {
            visible: true,
            publication_ready: false,
            task_start_allowed: true,
            state: 'needs_attention',
            reason: 'Review needs attention',
        },
    }));

    assert.match(html, />Publish to OuroborosHub</);
    assert.match(html, /data-submit-disabled="false"/);
    assert.match(html, /data-publication-ready="false"/);
    assert.match(html, /data-submit-state="needs_attention"/);
});

test('passive hard block stays visible but cannot start a task', () => {
    const html = renderInstalledSkillCard(skill({
        source: 'external',
        payload_root: 'skills/external/telegram',
        submit_hub: {
            visible: true,
            publication_ready: false,
            task_start_allowed: false,
            state: 'hard_block',
            reason: 'GitHub identity is unavailable',
        },
    }));

    assert.match(html, /data-submit-disabled="true"/);
    assert.match(html, /aria-disabled="true"/);
    assert.match(html, /GitHub identity is unavailable/);
});

test('task_start_allowed outranks a stale compatibility disabled flag', () => {
    const html = renderInstalledSkillCard(skill({
        source: 'external',
        payload_root: 'skills/external/telegram',
        submit_hub: {
            visible: true,
            publication_ready: false,
            task_start_allowed: true,
            disabled: true,
            state: 'needs_attention',
            reason: 'Agent work is needed',
        },
    }));

    assert.match(html, /data-submit-disabled="false"/);
    assert.match(html, /aria-disabled="false"/);
});

test('legacy review states keep selected preflight reachable', () => {
    const cases = [
        { review_status: 'pending' },
        { review_status: 'clean', review_stale: true },
        { review_status: 'blockers' },
        { review_status: 'clean', review_profile: 'owner_attested' },
    ];
    for (const reviewState of cases) {
        const html = renderInstalledSkillCard(skill({
            source: 'external',
            payload_root: 'skills/external/telegram',
            ...reviewState,
        }), new Set(), new Set(), {}, { githubTokenConfigured: true });
        assert.match(html, /data-submit-disabled="false"/);
        assert.match(html, /data-publication-ready="false"/);
        assert.match(html, />Publish to OuroborosHub</);
    }
});

test('reviewed Presence skill renders compact local runtime controls', () => {
    const html = renderInstalledSkillCard(skill({
        presence_runtime: {
            defaults: { model_slot: 'light', inline_max_rounds: 10 },
            overrides: { model_slot: 'main', inline_max_rounds: 7 },
            state_fingerprint: 'a'.repeat(64),
        },
    }));

    assert.match(html, /data-presence-runtime-form/);
    assert.match(html, /Presence runtime/);
    assert.match(html, /Reviewed default \(light\)/);
    assert.match(html, /name="inline_max_rounds"[^>]*value="7"/);
    assert.match(html, /data-presence-runtime-reset/);
    assert.match(html, /Applies to new Presence turns only/);
});

test('ordinary skill card does not render Presence runtime controls', () => {
    const html = renderInstalledSkillCard(skill());
    assert.doesNotMatch(html, /data-presence-runtime-form/);
});

// ---------------------------------------------------------------------------
// OuroborosHub sync badges (hubflow phase D): display-only, hub_sync verdict.
// ---------------------------------------------------------------------------

function hubOptions(rows, available = true) {
    return {
        hubCatalogByName: new Map(rows.map((row) => [row.sanitized_name, row])),
        hubCatalogAvailable: available,
    };
}

test('hub-bucket skill behind the catalog shows the Update available badge', () => {
    const html = renderInstalledSkillCard(skill({
        source: 'ouroboroshub',
        payload_root: 'skills/ouroboroshub/telegram',
        version: '1.0.0',
    }), new Set(), new Set(), {}, hubOptions([
        { slug: 'telegram', sanitized_name: 'telegram', latest_version: '2.0.0', identity_conflict: false },
    ]));
    assert.match(html, />Update available</);
    assert.doesNotMatch(html, />Published v/);
});

test('hub-bucket skill at the served version claims no update', () => {
    const html = renderInstalledSkillCard(skill({
        source: 'ouroboroshub',
        payload_root: 'skills/ouroboroshub/telegram',
        version: '2.0.0',
    }), new Set(), new Set(), {}, hubOptions([
        { slug: 'telegram', sanitized_name: 'telegram', latest_version: '2.0.0', identity_conflict: false },
    ]));
    assert.doesNotMatch(html, />Update available</);
});

test('byte-verified hub skill shows Published even without a catalog snapshot', () => {
    const html = renderInstalledSkillCard(skill({
        source: 'ouroboroshub',
        payload_root: 'skills/ouroboroshub/telegram',
        version: '2.0.0',
        official_hub_verified: true,
    }));
    assert.match(html, />Published v2\.0\.0</);
});

test('publish receipt unconfirmed by the catalog shows the Submitted PR badge', () => {
    const html = renderInstalledSkillCard(skill({
        source: 'external',
        payload_root: 'skills/external/telegram',
        version: '0.1.0',
        content_hash: 'b'.repeat(64),
        published: {
            slug: 'telegram',
            version: '0.1.0',
            content_hash: 'a'.repeat(64),
            repository: 'razzant/ouroboroshub',
            pr_number: 7,
            pr_url: 'https://github.com/razzant/ouroboroshub/pull/7',
            published_at: '2026-08-20T00:00:00Z',
        },
    }), new Set(), new Set(), {}, hubOptions([]));
    assert.match(html, />Submitted PR #7</);
});

test('without a catalog snapshot no catalog-derived badge is guessed', () => {
    const html = renderInstalledSkillCard(skill({
        source: 'ouroboroshub',
        payload_root: 'skills/ouroboroshub/telegram',
        version: '1.0.0',
        published: {
            slug: 'telegram',
            version: '1.0.0',
            content_hash: 'a'.repeat(64),
            repository: 'razzant/ouroboroshub',
            pr_number: 9,
            pr_url: 'https://github.com/razzant/ouroboroshub/pull/9',
            published_at: '2026-08-20T00:00:00Z',
        },
    }));
    assert.doesNotMatch(html, />Update available</);
    assert.doesNotMatch(html, />Submitted PR #/);
});

test('native cards never carry hub sync badges', () => {
    const html = renderInstalledSkillCard(skill(), new Set(), new Set(), {}, hubOptions([
        { slug: 'telegram', sanitized_name: 'telegram', latest_version: '9.9.9', identity_conflict: false },
    ]));
    assert.doesNotMatch(html, />Update available</);
    assert.doesNotMatch(html, />Published v/);
    assert.doesNotMatch(html, />Submitted PR #/);
});
