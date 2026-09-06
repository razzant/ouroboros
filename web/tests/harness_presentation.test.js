import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { familyLabel } from '../modules/claudexor_status_store.js';
import { harnessFamilyMarkup } from '../modules/harness_accounts.js';
import { createLoginCardController, loginCardHtml } from '../modules/harness_login_cards.js';
import {
    GENERIC_HARNESS_MARK,
    HARNESS_MARKS,
    harnessAccountIdentityMarkup,
    harnessIdentityMarkup,
    harnessPresentation,
} from '../modules/harness_presentation.js';
import {
    ROUTE_KIND_AGENT_SESSION,
    ROUTE_KIND_API_MODEL,
    serializeRouteSpec,
} from '../modules/route_editor_primitives.js';
import { availableSubagentRowMarkup } from '../modules/subagents_settings.js';

const OFFICIAL_PATH_SHA256 = Object.freeze({
    codex: '0ca6f41e3579cc59746a5b6b92b558f403bdc28ebf08f42be7943ce282700e1c',
    claude: '0442033dcc3824e52ffb0a07849c46becbeeacd75d1287f9081c00510e3bbf84',
    cursor: '85eaa79be69a55d712a4843bf8d65a217d0e5b648177c8e5f2ad67fe9a46d9ad',
    opencode: 'f4f11e1603a4a49ca387925840ed9ebb4505786d6db512225205e076e54b4676',
});

test('official Claudexor vector geometry stays exact', () => {
    assert.deepEqual(Object.keys(HARNESS_MARKS), ['codex', 'claude', 'cursor', 'opencode']);
    assert.equal(HARNESS_MARKS.codex.viewBox, '0 0 256 260');
    for (const id of ['claude', 'cursor', 'opencode']) {
        assert.equal(HARNESS_MARKS[id].viewBox, '0 0 24 24');
    }
    for (const [id, expected] of Object.entries(OFFICIAL_PATH_SHA256)) {
        assert.equal(createHash('sha256').update(HARNESS_MARKS[id].path).digest('hex'), expected);
    }
});

test('identity markup is local, monochrome, decorative, and text-always', () => {
    const html = harnessIdentityMarkup('claude');
    assert.match(html, /<svg\b[^>]*fill="currentColor"/);
    assert.match(html, /aria-hidden="true"/);
    assert.match(html, /focusable="false"/);
    assert.match(html, /<span class="harness-identity-label">Claude Code<\/span>/);
    assert.doesNotMatch(html, /(?:https?:|data:|url\()/i);
});

test('named account identity reuses the mark and escapes its visible profile', () => {
    const html = harnessAccountIdentityMarkup('codex', {
        label: 'Codex Live', profile: 'work<&"',
    });
    assert.match(html, /data-harness-identity="codex"/);
    assert.match(html, />Codex Live<\/span><\/span> \(work&lt;&amp;&quot;\)$/);
});

test('app and standalone onboarding sheets share the mark and long-label contract', () => {
    const sheets = [
        readFileSync(new URL('../style.css', import.meta.url), 'utf8'),
        readFileSync(new URL('../onboarding.css', import.meta.url), 'utf8'),
    ];
    for (const css of sheets) {
        assert.match(css, /\.harness-identity\s*\{[^}]*display:\s*inline-flex;/s);
        assert.match(css, /\.harness-identity\s*\{[^}]*gap:\s*var\(--space-1\);/s);
        assert.match(css, /\.harness-identity-mark\s*\{[^}]*flex:\s*0 0 1em;/s);
        assert.match(css, /\.harness-identity-mark\s*\{[^}]*fill:\s*currentColor;/s);
        assert.match(css, /\.harness-identity-label\s*\{[^}]*overflow-wrap:\s*anywhere;/s);
        assert.match(css, /\.available-subagent-route-identity-wrap\s*\{[^}]*min-width:\s*0;/s);
    }
});

test('unknown and Agy identities use the generic vector with readable escaped text', () => {
    const agy = harnessPresentation('agy');
    const future = harnessPresentation('future<&"');
    assert.equal(agy.label, 'Antigravity');
    assert.equal(agy.generic, true);
    assert.equal(agy.path, GENERIC_HARNESS_MARK.path);
    assert.equal(future.label, 'future<&"');
    assert.equal(future.known, false);
    assert.equal(future.path, GENERIC_HARNESS_MARK.path);

    const html = harnessIdentityMarkup('future<&"');
    assert.match(html, /future&lt;&amp;&quot;/);
    assert.doesNotMatch(html, /future<&"/);
});

test('direct API is a neutral channel presentation, never a harness identity', () => {
    const api = harnessPresentation('api_model');
    assert.deepEqual({ kind: api.kind, harnessId: api.harnessId, label: api.label }, {
        kind: 'channel', harnessId: null, label: 'API',
    });
    const html = harnessIdentityMarkup('api_model');
    assert.match(html, /data-presentation-kind="channel"/);
    assert.match(html, />API<\/span>/);
    const native = harnessPresentation('native', { label: 'API · native tool rounds' });
    assert.deepEqual({ kind: native.kind, harnessId: native.harnessId, label: native.label }, {
        kind: 'channel', harnessId: null, label: 'API · native tool rounds',
    });
});

test('Chat renders the executor evidence chip through the shared identity SSOT', () => {
    const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const logSource = readFileSync(new URL('../modules/log_events.js', import.meta.url), 'utf8');
    assert.match(chatSource, /harnessIdentityMarkup\(record\.executorChip\.harness/);
    assert.match(logSource, /harnessPresentation\(harness\)\.label/);
    assert.doesNotMatch(logSource, /HARNESS_CHIP_(?:ICON|NAME)/);
    assert.doesNotMatch(chatSource, /record\.executorChip\.icon/);
});

test('live daemon display names override catalog fallbacks without changing the mark', () => {
    const payload = { harnesses: [{ id: 'claude', display_name: 'Claude Code CLI' }] };
    assert.equal(familyLabel('claude', payload, { catalogKnown: true }), 'Claude Code CLI');
    const presentation = harnessPresentation('claude', {
        label: familyLabel('claude', payload, { catalogKnown: true }),
    });
    assert.equal(presentation.label, 'Claude Code CLI');
    assert.equal(presentation.path, HARNESS_MARKS.claude.path);
});

test('Agents and login surfaces use the shared identity markup with visible labels', () => {
    const groupHtml = harnessFamilyMarkup({
        harness: 'codex', label: 'Codex Live', rows: [],
        status: { tone: 'muted', label: 'No account connected' },
    }, { daemon: {} }, { accountsRead: 'ok', quotaRead: 'ok' });
    assert.match(groupHtml, /data-harness-identity="codex"/);
    assert.match(groupHtml, />Codex Live<\/span>/);

    const loginHtml = loginCardHtml({
        harness: 'cursor', familyLabel: 'Cursor Live', profile: '',
        envelope: null, error: 'offline', verdict: null,
    });
    assert.match(loginHtml, /data-harness-identity="cursor"/);
    assert.match(loginHtml, />Cursor Live<\/span>/);
});

test('an active login identity trusts only a currently proven catalog label', async () => {
    let catalogKnown = false;
    let snapshot = { harnesses: [{ id: 'codex', display_name: 'Stale daemon label' }] };
    const store = {
        get catalogKnown() { return catalogKnown; },
        get snapshot() { return snapshot; },
        holdPolling: () => () => {},
    };
    const host = {
        innerHTML: '', contains: () => false, querySelector: () => null,
        querySelectorAll: () => [],
    };
    const ctl = createLoginCardController({
        host, store,
        fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({ error: 'offline' }) }),
    });
    await ctl.start('codex', '');
    assert.match(host.innerHTML, />Codex<\/span>/);
    assert.doesNotMatch(host.innerHTML, /Stale daemon label/);

    snapshot = { harnesses: [{ id: 'codex', display_name: 'Codex Live' }] };
    catalogKnown = true;
    ctl.render();
    assert.match(host.innerHTML, />Codex Live<\/span>/);

    catalogKnown = false;
    ctl.render();
    assert.match(host.innerHTML, />Codex<\/span>/);
    assert.doesNotMatch(host.innerHTML, /Codex Live/);
    ctl.detach();
});

test('configured subagent marks surround native text controls without changing saved bytes', () => {
    const sessionRow = {
        subagent_id: 'builder', name: 'Builder', recommended_use: 'Implement changes.',
        route: {
            kind: ROUTE_KIND_AGENT_SESSION,
            target_id: 'codex=gpt-5.6-sol-high',
            credential_profile_id: 'work',
        },
    };
    const before = JSON.stringify(sessionRow);
    const state = {
        catalogKnown: true, accountsKnown: true, quotaKnown: true, statusError: '',
        snapshot: {
            harnesses: [{
                id: 'codex', display_name: 'Codex Live', status: 'ok', enabled: true,
                models: [{ id: 'gpt-5.6-sol-high' }],
            }],
            profiles: { profiles: [{
                profile: { harness_id: 'codex', profile_id: 'work', enabled: true },
                status: { verification: 'passed' },
            }] },
            quota: [],
        },
    };
    const html = availableSubagentRowMarkup(sessionRow, state);
    assert.match(html, /data-harness-identity="codex"/);
    assert.match(html, />Codex Live<\/span>/);
    assert.match(html, /<select[^>]*data-subagent-field="route"/);
    assert.equal(JSON.stringify(sessionRow), before);
    assert.deepEqual(serializeRouteSpec(sessionRow.route, {
        apiKind: ROUTE_KIND_API_MODEL,
        credentialField: 'credential_profile_id',
    }), sessionRow.route);

    const apiRow = {
        ...sessionRow,
        route: { kind: ROUTE_KIND_API_MODEL, target_id: 'openai/gpt-5.6-sol' },
    };
    const apiHtml = availableSubagentRowMarkup(apiRow, {
        ...state, snapshot: { harnesses: [], profiles: {}, quota: [] },
    });
    assert.match(apiHtml, /data-presentation-kind="channel"/);
    assert.match(apiHtml, /aria-label="API model for Subagent 1"/);
});

test('configured identity ignores stale daemon labels until the catalog read is proven', () => {
    const row = {
        subagent_id: 'builder', recommended_use: 'Implement changes.',
        route: {
            kind: ROUTE_KIND_AGENT_SESSION,
            target_id: 'codex=gpt-5.6-sol-high',
            credential_profile_id: '',
        },
    };
    const snapshot = {
        harnesses: [{
            id: 'codex', display_name: 'Stale daemon label',
            models: [{ id: 'gpt-5.6-sol-high' }],
        }],
        profiles: { harnessAccounts: [], profiles: [] },
        quota: [],
    };
    const gapHtml = availableSubagentRowMarkup(row, {
        catalogKnown: false, accountsKnown: false, quotaKnown: false,
        statusError: '', snapshot,
    });
    assert.match(gapHtml, />Codex<\/span>/);
    assert.doesNotMatch(gapHtml, /Stale daemon label/);

    const provenHtml = availableSubagentRowMarkup(row, {
        catalogKnown: true, accountsKnown: false, quotaKnown: false,
        statusError: '', snapshot,
    });
    assert.match(provenHtml, />Stale daemon label<\/span>/);
});

test('Chat, Logs, onboarding, and reviewer lanes consume the same mark owner', () => {
    const modules = ['chat.js', 'logs.js', 'onboarding_agents_step.js', 'reviewer_slots.js'];
    for (const name of modules) {
        const source = readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8');
        assert.match(source, /harnessIdentityMarkup/, `${name} bypasses harness presentation SSOT`);
    }
    const events = readFileSync(new URL('../modules/log_events.js', import.meta.url), 'utf8');
    assert.match(events, /harnessPresentation/);
    assert.doesNotMatch(events, /HARNESS_CHIP_(?:ICON|NAME)/);
    assert.doesNotMatch(events, /['"](?:◇|✳|▸|◆)['"]/);
});
