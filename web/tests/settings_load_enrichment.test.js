// Settings paints and saves from the settings document alone. The installed-skill
// list (skill-requested keys, extension settings sections) is optional enrichment
// that lands later, only on an unchanged draft, and a hidden page never reads it.
// These drive the production loadSettings and its lifecycle listeners against
// their closure boundary, the way skills_read_state.test.js drives its reader.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { setImmediate as nextTurn } from 'node:timers/promises';

function source(from, until) {
    const text = readFileSync(new URL('../modules/settings.js', import.meta.url), 'utf8');
    const start = text.indexOf(from);
    const end = text.indexOf(until, start);
    assert.ok(start >= 0 && end > start, 'settings.js: source boundaries');
    return text.slice(start, end);
}

function deferred() {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

const SKILLS = [{ name: 'reveal-fixture', grants: { requested_keys: ['TEST_REVEAL_KEY'] } }];
const SECTIONS = [{ skill: 'reveal-fixture', section_id: 'config', render: { components: [] } }];

function settingsLoader({ activePage = 'settings', extensions } = {}) {
    const calls = { applied: 0, rows: [], sections: [], baselines: 0, dirtyChecks: 0, extensionReads: 0 };
    const context = vm.createContext({
        page: {}, loadSequence: 0, restartReadSequence: 0, draftRevision: 0,
        currentSettings: {}, settingsLoaded: false, saveOutcomeUnknown: true, validationAttempted: true,
        state: { activePage },
        apiClient: {
            settings: async () => ({ GITHUB_REPO: 'owner/repo', _meta: {} }),
            extensions: () => { calls.extensionReads += 1; return extensions(); },
        },
        resetSecretReveals() {}, syncRestartState() {}, renderCustomSecrets() {}, paintSettingsFieldErrors() {},
        armCleanBaselineOnStatusSettle() {}, _renderNetworkHint() {}, syncSettingsLoadState() {},
        applySettings() { calls.applied += 1; },
        setSettingsCleanBaseline() { calls.baselines += 1; },
        updateSettingsDirtyState() { calls.dirtyChecks += 1; },
        renderRequestedSkillSecrets(root, skills) { calls.rows.push(skills); },
        renderExtensionSettingsSections: async (root, sections, { isCurrent }) => { if (isCurrent()) calls.sections.push(sections); },
        reloadReviewPool: async () => {}, reloadSubagentsSection: async () => {},
    });
    vm.runInContext(source('    const settingsPageActive = ', '\n    async function reloadSettingsWithFeedback'), context);
    return { context, calls, load: () => context.loadSettings() };
}

test('the settings document paints and Save is usable while /api/extensions is held; the rows follow', async () => {
    const extensions = deferred();
    const view = settingsLoader({ extensions: () => extensions.promise });
    const loading = view.load();
    await nextTurn();
    assert.equal(view.calls.extensionReads, 1, 'the list read starts with the document read');
    assert.equal(view.calls.applied, 1, 'the document is applied without the list');
    assert.equal(view.context.settingsLoaded, true, 'Save is usable while the list is held');
    assert.equal(view.context.saveOutcomeUnknown, false);
    assert.equal(view.calls.baselines, 1, 'the clean baseline is taken from the document alone');
    assert.deepEqual(view.calls.rows, [], 'no row claims absence before the list arrives');
    extensions.resolve({ skills: SKILLS, live: { settings_sections: SECTIONS } });
    assert.equal(await loading, true);
    assert.deepEqual(view.calls.rows, [SKILLS], 'the skill-requested rows land after the release');
    assert.deepEqual(view.calls.sections, [SECTIONS]);
    assert.equal(view.calls.baselines, 2, 'the enrichment re-baselines the still-clean draft');
    assert.equal(view.calls.dirtyChecks, 0);

    // A failed list read is today's empty enrichment, never a failed document.
    const failing = settingsLoader({ extensions: () => Promise.reject(new Error('HTTP 503')) });
    assert.equal(await failing.load(), true);
    assert.equal(JSON.stringify(failing.calls.rows), '[[]]');
    assert.equal(JSON.stringify(failing.calls.sections), '[[]]');
});

test('an owner edit made before the enrichment arrives survives it: nothing is absorbed or overwritten', async () => {
    const extensions = deferred();
    const view = settingsLoader({ extensions: () => extensions.promise });
    const loading = view.load();
    await nextTurn();
    assert.equal(view.context.settingsLoaded, true);
    view.context.draftRevision += 1; // the owner typed into the painted document
    extensions.resolve({ skills: SKILLS, live: { settings_sections: SECTIONS } });
    assert.equal(await loading, false, 'the load reports that the draft was kept');
    assert.deepEqual(view.calls.rows, [], 'late rows do not replace what the owner may be editing');
    assert.deepEqual(view.calls.sections, []);
    assert.equal(view.calls.baselines, 1, 'the edited draft is never re-baselined as clean');
    assert.equal(view.calls.dirtyChecks, 1);
});

test('a newer load supersedes an older held list read; a hidden page reads no list at all', async () => {
    const olderRead = deferred();
    let reads = 0;
    const view = settingsLoader({
        extensions: () => (reads++ === 0 ? olderRead.promise : Promise.resolve({ skills: SKILLS, live: {} })),
    });
    const older = view.load();
    await nextTurn();
    const newer = view.load();
    assert.equal(await newer, true);
    assert.deepEqual(view.calls.rows, [SKILLS]);
    olderRead.resolve({ skills: [{ name: 'stale' }], live: {} });
    assert.equal(await older, false, 'the superseded load reports itself superseded');
    assert.deepEqual(view.calls.rows, [SKILLS], 'the superseded load wrote nothing');
    assert.equal(view.calls.baselines, 3, 'older document, newer document, newer enrichment');

    const hidden = settingsLoader({ activePage: 'chat', extensions: () => { throw new Error('must not read'); } });
    assert.equal(await hidden.load(), true);
    assert.equal(hidden.calls.extensionReads, 0, 'a hidden page (boot, background refresh) skips the list');
    assert.equal(hidden.context.settingsLoaded, true);
    assert.deepEqual(hidden.calls.rows, [], 'the hosts keep what they show; nothing claims absence');
    assert.equal(hidden.calls.baselines, 2);
});

test('skill lifecycle listeners reload only a visible Settings page', () => {
    const listeners = {}, refreshes = [];
    for (const activePage of ['chat', 'settings']) {
        const context = vm.createContext({
            window: { addEventListener: (name, handler) => { listeners[name] = handler; } },
            ws: { on: (name, handler) => { listeners[name] = handler; } },
            settingsPageActive: () => activePage === 'settings',
            refreshSettingsAfterExtensionChange: (reason) => { refreshes.push(`${activePage}:${reason}`); },
        });
        vm.runInContext(source("    window.addEventListener('ouro:skill-lifecycle'", '\n    // A confirmed Accounts facet'), context);
        listeners['ouro:skill-lifecycle']({ detail: { action: 'review' } });
        listeners.extension_lifecycle({ action: 'loaded' });
        listeners['ouro:settings-updated']({ detail: { reason: 'budget saved', source: 'costs' } });
    }
    assert.deepEqual(refreshes, ['chat:budget saved', 'settings:review', 'settings:loaded', 'settings:budget saved']);
});
