// One writer (docs/DESIGN.md "Settings"): a setting is written by the row helper, settings_controls.js, and by
// nothing else under web/modules. No other module posts to /api/settings or to the owner settings endpoints, and
// none reads the old save answer (`immediate_changed`, `next_task_changed`, `restart_required`) to say when a change
// applies: the helper says that from `_meta.effects`. An editor's Save goes through the helper too.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import test from 'node:test';

const MODULES = new URL('../modules/', import.meta.url);
const WRITER = 'settings_controls.js';

// The transports the helper calls, and the writers that are not Settings rows.
const ALLOWED_WRITES = {
    'api_client.js': 'the HTTP client: it defines the calls, the helper makes them',
    'consciousness_api.js': 'the owner endpoints\' client, called by the helper',
    'composer_owner_controls.js': 'the composer\'s Context capsule writes Main\'s value through its own owner endpoint',
    'subagents_settings.js': 'the roster\'s checker adapter, which the helper\'s writeEntry calls for every roster write',
};
// The same word for another fact: an update's progress, the wizard's completion answer, and the home's read of
// GET /api/settings `_meta` (a restart that waits, the Access that runs).
const ALLOWED_READS = {
    'updates.js': 'an update that needs a restart', 'update_status.js': 'an update that needs a restart',
    'onboarding_wizard.js': 'the wizard\'s completion answer', 'onboarding_overlay.js': 'the wizard\'s completion message',
    'onboarding_agents_step.js': 'the wizard\'s preview answer',
    'settings_home.js': '`_meta.restart_state` and `_meta.policy_state` of the read, never a save answer',
};
// Modules another lane is still converting to rows. Empty on the integrated tree: a name that no longer writes or
// reads fails below until it is removed.
const PENDING = {};

const WRITES = /apiClient\.saveSettings\(|\bowner(RuntimeMode|AutoGrant|ContextMode|SafetyMode)\(|\bsave(SafetyLevel|SafetyRules|CheckerSettings)\(|['"`]\/api\/owner\/(runtime-mode|auto-grant|context-mode|safety-mode|safety-rules|safety-model)\b|['"`]\/api\/settings['"`][^\n]*method:\s*['"]POST/;
const READS = /\b(immediate_changed|next_task_changed|restart_required)\b/;

/** What a module's source does that only the helper may: `write`, `read`, both, or nothing. Comments do not count. */
function offences(source) {
    const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '').replace(/\s\/\/ .*$/gm, '');
    return [WRITES.test(code) ? 'write' : '', READS.test(code) ? 'read' : ''].filter(Boolean);
}

test('the guard fires on a second writer and on a reader of the old answer, and is quiet on the helper\'s callers', () => {
    assert.deepEqual(offences("await apiClient.saveSettings({ TOTAL_BUDGET: 1 }, { headers });"), ['write']);
    assert.deepEqual(offences("await apiFetch('/api/owner/safety-rules', { method: 'POST' });"), ['write']);
    assert.deepEqual(offences("fetchJson('/api/settings', { method: 'POST', body });"), ['write']);
    assert.deepEqual(offences('if (data.next_task_changed) say("applies to new tasks");'), ['read']);
    assert.deepEqual(offences("const outcome = await commitSetting(row, value); // never apiClient.saveSettings(\nconst data = await fetchJson('/api/settings', { cache: 'no-store' });"), []);
    assert.deepEqual(offences('/* restart_required */ const row = settingsRow(rowById(id));'), []);
});

test('outside settings_controls.js no module writes a setting or reads the old save answer', () => {
    const found = {};
    for (const name of readdirSync(MODULES).filter((file) => file.endsWith('.js') && !file.endsWith('_types.js'))) {
        const kinds = offences(readFileSync(new URL(name, MODULES), 'utf8'));
        if (kinds.length) found[name] = kinds;
    }
    assert.deepEqual(found[WRITER], ['write'], 'the helper is the writer, and reads no old answer field');
    const excused = (name, kind) => name === WRITER || Object.hasOwn(PENDING, name)
        || Object.hasOwn(kind === 'write' ? ALLOWED_WRITES : ALLOWED_READS, name);
    const strangers = Object.entries(found).flatMap(([name, kinds]) => kinds.filter((kind) => !excused(name, kind)).map((kind) => `${name}: ${kind}`));
    assert.deepEqual(strangers, []);
    // The lists only say what is true: an excused module that no longer does it comes off its list.
    for (const name of Object.keys(ALLOWED_WRITES)) assert.ok(found[name]?.includes('write'), `${name} writes nothing: remove it from ALLOWED_WRITES`);
    for (const name of Object.keys(ALLOWED_READS)) assert.ok(found[name]?.includes('read'), `${name} reads none of them: remove it from ALLOWED_READS`);
    for (const [name, why] of Object.entries(PENDING)) assert.ok(found[name], `${name} is converted: remove it from PENDING (${why})`);
});
