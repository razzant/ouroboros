// The list of skill forms (docs/DESIGN.md "Settings"): `skill` without an id is one door per skill that registered a
// settings form, so "where are the settings of my skills" has an answer in Settings and search has a page to land
// on. An empty list says that no installed skill has settings; a list that could not be read claims nothing.
import assert from 'node:assert/strict';
import test from 'node:test';

import { extensionRoutePath } from '../modules/api_client.js';
import { getContributions } from '../modules/contributions.js';
import '../modules/settings_extension_forms.js';
import { searchEntries, searchSettings } from '../modules/settings_home.js';
import { settingsPageFor } from '../modules/settings_pages.js';
import { Stub, settle, world } from './helpers/settings_dom.js';

const section = (skill, title, id = 'config') => ({ skill, section_id: id, key: `${skill}:${id}`, title, render: { components: [] } });
const INSTALLED = {
    skills: [{ name: 'live_conversation', title: 'Calls' }, { name: 'telegram', title: 'Telegram' }, { name: 'quiet_one', title: 'No form here' }],
    live: { settings_sections: [section('live_conversation', 'Calls (beta)'), section('telegram', 'Bot', 'bot'), section('telegram', 'Proxy', 'proxy')] },
};

async function mountList(t, answer) {
    const { env } = world(t);
    env.routes.set('/api/extensions', async () => answer());
    const opened = [];
    const state = { activePage: 'skills', settingsPath: 'skill' };
    const container = new Stub('div');
    const page = settingsPageFor('skill');
    assert.equal(typeof page.title === 'function' ? page.title('skill') : page.title, 'Skill settings');
    const handle = await page.mount(container, { path: 'skill', ctx: { openSettings: (path) => opened.push(path), state, onScreen: (path) => state.activePage === 'skills' && state.settingsPath === path } });
    await settle();
    return { env, container, opened, handle, state, doors: () => container.querySelectorAll('[data-setting-door]') };
}
const ok = (body) => ({ ok: true, status: 200, json: async () => body });

test('one door per skill that registered a form, named by its title, opening that skill\'s page', async (t) => {
    const { env, container, opened, handle, doors } = await mountList(t, () => ok(INSTALLED));
    const names = doors().map((door) => [door.dataset.settingRow, door.querySelector('.ui-row-title').textContent,
        door.querySelector('[data-setting-meta]').textContent]);
    // One section: the form's own title, with the skill's name under it. Several: the skill's name.
    assert.deepEqual(names, [['skill:live_conversation', 'Calls (beta)', 'Calls'], ['skill:telegram', 'Telegram', '']]);
    assert.doesNotMatch(container.textContent, /No installed skill has settings|No form here/);
    await container.fire('click', { target: doors()[1] });
    await container.fire('click', { target: doors()[0] });
    assert.deepEqual(opened, ['skill/telegram', 'skill/live_conversation']);
    // Shown again, the list is read again for its own path only; let go, it reads no more.
    const reads = () => env.calls.filter((url) => url === '/api/extensions').length;
    const before = reads();
    env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'skill/telegram' } }));
    await settle();
    assert.equal(reads(), before);
    env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'skill' } }));
    await settle();
    assert.equal(reads(), before + 1);
    handle.unmount();
    env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'skill' } }));
    await settle();
    assert.equal(reads(), before + 1);
});

test('a skill changed by anyone redraws the list on screen: the server says so or this page did it; hidden or let go, nothing is read', async (t) => {
    const installed = structuredClone(INSTALLED);
    const { env, handle, state, doors } = await mountList(t, () => ok(installed));
    env.routes.set('/api/ui/contributions', async () => ok({}));
    const heard = new Map();
    t.after(getContributions().attach({ ws: { on(type, fn) { heard.set(type, fn); return () => heard.delete(type); } }, target: env.window }));
    await settle();
    const reads = () => env.calls.filter((url) => url === '/api/extensions').length;
    assert.deepEqual(doors().map((door) => door.dataset.settingRow), ['skill:live_conversation', 'skill:telegram']);
    // Ouroboros or another window took a skill's form away: the server says so and this page did nothing.
    installed.live.settings_sections = installed.live.settings_sections.filter((entry) => entry.skill !== 'live_conversation');
    let before = reads();
    heard.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'live_conversation', action: 'disabled' });
    await settle();
    assert.equal(reads(), before + 1);
    assert.deepEqual(doors().map((door) => door.dataset.settingRow), ['skill:telegram']);
    // An action on this page reaches the list through the same signal.
    before = reads();
    env.window.dispatchEvent(new CustomEvent('ouro:skill-lifecycle', { detail: { action: 'review', name: 'telegram' } }));
    await settle();
    assert.equal(reads(), before + 1);
    // Another page is on screen: the list reads nothing (the shell's "shown" reads it on the way back, above).
    state.activePage = 'chat';
    heard.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'telegram', action: 'updated' });
    await settle();
    assert.equal(reads(), before + 1, 'a hidden Extensions reads nothing');
    state.activePage = 'skills';
    // A skill's form is open over the list: the list under it reads nothing either.
    state.settingsPath = 'skill/telegram';
    heard.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'telegram', action: 'updated' });
    await settle();
    assert.equal(reads(), before + 1, 'a list under an open form reads nothing');
    state.settingsPath = 'skill';
    handle.unmount();
    heard.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'telegram', action: 'enabled' });
    await settle();
    assert.equal(reads(), before + 1, 'a page that was let go reads no more');
});

test('a skill form the owner types into is not redrawn by a read that was on its way; untouched, it follows the server', async (t) => {
    let handle;
    t.after(() => handle?.unmount());
    const { env } = world(t);
    const component = { type: 'form', id: 'bot', route: 'save', submit_label: 'Save', fields: [{ name: 'chat_id', label: 'Chat ID', type: 'text' }] };
    const installed = { skills: [{ name: 'demo', title: 'Demo' }], live: { settings_sections: [{ ...section('demo', 'Demo'), render: { components: [component] } }] } };
    let stored = 'first';
    let gate = null;  // while set, the read of the form's stored values waits for it
    env.routes.set('/api/extensions', async () => ok(installed));
    env.routes.set(extensionRoutePath('demo', 'save'), async () => { await gate; return ok({ chat_id: stored }); });
    env.routes.set('/api/ui/contributions', async () => ok({}));
    const heard = new Map();
    t.after(getContributions().attach({ ws: { on(type, fn) { heard.set(type, fn); return () => heard.delete(type); } }, target: env.window }));
    const container = new Stub('div');
    container.replaceChildren = (...nodes) => { container.textContent = ''; container.append(...nodes); };  // the stub lacks it
    const state = { activePage: 'skills', settingsPath: 'skill/demo' };
    handle = await settingsPageFor('skill/demo').mount(container, { path: 'skill/demo', ctx: { openSettings() {}, state, onScreen: (path) => state.activePage === 'skills' && state.settingsPath === path } });
    await settle();
    const host = container.firstElementChild;
    const shown = () => host.querySelector('[name="chat_id"]')?.getAttribute('value');
    assert.equal(shown(), 'first');
    const changedElsewhere = () => heard.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'other', action: 'updated' });

    assert.equal(handle.unsaved(), false);
    // Untouched, the form follows the server: another window or Ouroboros changed a skill, the form is read again.
    stored = 'second';
    changedElsewhere();
    await settle();
    assert.equal(shown(), 'second');

    // A read is on its way when the owner starts typing: its answer is not drawn over the owner's words.
    stored = 'third';
    let release;
    gate = new Promise((resolve) => { release = resolve; });
    changedElsewhere();
    await settle();
    const field = host.querySelector('[name="chat_id"]');
    const form = field.closest('form');
    // The stub has neither reflected names, a named controls collection nor event propagation.
    field.name = field.getAttribute('name');
    field.value = 'my draft';
    form.elements = { namedItem: (name) => form.querySelectorAll('[name]').find(node => node.getAttribute('name') === name) };
    await form.fire('input', { type: 'input', target: field });
    release();
    gate = null;
    await settle();
    assert.equal(host.querySelector('[name="chat_id"]'), field, 'the field the owner types in stays the same node');
    assert.equal(shown(), 'second', 'the answer that arrived late was not drawn');
    assert.equal(field.value, 'my draft');
    assert.equal(handle.unsaved(), true, 'the shared levels must keep only forms with explicit unsaved edits');
    await form.fire('submit', { preventDefault() {} });
    assert.equal(handle.unsaved(), false, 'the matching successful Save releases the draft');
});

test('a read on its way when the owner leaves or covers a skill form draws nothing; shown again, it is read and drawn', async (t) => {
    let handle;
    // After hooks run in registration order: unmount while this world's window/document still exist.
    t.after(() => handle?.unmount());
    const { env } = world(t);
    const component = { type: 'form', id: 'bot', route: 'save', submit_label: 'Save', fields: [{ name: 'chat_id', label: 'Chat ID', type: 'text' }] };
    const installed = { skills: [{ name: 'demo', title: 'Demo' }], live: { settings_sections: [{ ...section('demo', 'Demo'), render: { components: [component] } }] } };
    let stored = 'first';
    let gate = null;  // while set, the read of the form's stored values waits for it
    env.routes.set('/api/extensions', async () => ok(installed));
    env.routes.set(extensionRoutePath('demo', 'save'), async () => { await gate; return ok({ chat_id: stored }); });
    env.routes.set('/api/ui/contributions', async () => ok({}));
    const heard = new Map();
    t.after(getContributions().attach({ ws: { on(type, fn) { heard.set(type, fn); return () => heard.delete(type); } }, target: env.window }));
    const container = new Stub('div');
    container.replaceChildren = (...nodes) => { container.textContent = ''; container.append(...nodes); };  // the stub lacks it
    // The section's fact (section_levels.js onScreen): Extensions is the place shown and this form its top level.
    const state = { activePage: 'skills', settingsPath: 'skill/demo' };
    const ctx = { openSettings() {}, state, onScreen: (path) => state.activePage === 'skills' && state.settingsPath === path };
    handle = await settingsPageFor('skill/demo').mount(container, { path: 'skill/demo', ctx });
    await settle();
    const host = container.firstElementChild;
    const shown = () => host.querySelector('[name="chat_id"]')?.getAttribute('value');
    assert.equal(shown(), 'first');
    const changedElsewhere = () => heard.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'other', action: 'updated' });
    const hold = () => { let release; gate = new Promise((resolve) => { release = resolve; }); return () => { release(); gate = null; }; };

    // Left: the owner went to another place while the read was on its way. Its answer draws nothing there.
    stored = 'second';
    let release = hold();
    changedElsewhere();
    await settle();
    state.activePage = 'chat';
    release();
    await settle();
    assert.equal(shown(), 'first', 'a read that lands after the owner left draws nothing');
    // A lifecycle signal while the place is hidden reads nothing at all.
    const reads = () => env.calls.filter((url) => url === '/api/extensions').length;
    const before = reads();
    changedElsewhere();
    await settle();
    assert.equal(reads(), before, 'a hidden form reads nothing');
    // Back: the shell says "shown" for this path, and the form is read and drawn.
    state.activePage = 'skills';
    env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'skill/demo', place: 'skills' } }));
    await settle();
    assert.equal(shown(), 'second', 'shown again, it is read and drawn');

    // Covered: another level lies over the form (its section's top is another path). The same: nothing is drawn.
    stored = 'third';
    release = hold();
    changedElsewhere();
    await settle();
    state.settingsPath = 'skill/demo/other';
    release();
    await settle();
    assert.equal(shown(), 'second', 'a read that lands under another level draws nothing');
    state.settingsPath = 'skill/demo';
    env.window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'skill/demo', place: 'skills' } }));
    await settle();
    assert.equal(shown(), 'third');
});

test('an empty list says that no installed skill has settings', async (t) => {
    const empty = await mountList(t, () => ok({ skills: [{ name: 'quiet_one' }], live: { settings_sections: [] } }));
    assert.equal(empty.doors().length, 0);
    assert.match(empty.container.textContent, /No installed skill has settings\./);
    empty.handle.unmount();
});

test('a list that could not be read claims nothing about the skills', async (t) => {
    const failed = await mountList(t, () => ({ ok: false, status: 503, json: async () => ({ error: 'the skills index is busy' }) }));
    assert.equal(failed.doors().length, 0);
    assert.doesNotMatch(failed.container.textContent, /No installed skill has settings/);
    assert.ok(failed.container.querySelector('.ui-result').textContent.trim(), 'it says the read failed');
    failed.handle.unmount();
});

test('search has a door to the list, by the names it had: Extension Settings, under Advanced; it leads to Extensions', () => {
    const entries = searchEntries({ forms: [] });
    for (const words of ['Skill settings', 'extension settings', 'ADVANCED', 'Settings of skills']) {
        const hit = searchSettings(words, entries).find((entry) => entry.id === 'door:skill');
        assert.deepEqual([hit?.to, hit?.row], ['skills:skill', ''], words);
    }
    // "Advanced" also names what moved to This computer: both are offered, neither guessed.
    assert.ok(searchSettings('advanced', entries).some((entry) => entry.id === 'home:computer'));
    assert.ok(!searchSettings('extension settings', searchEntries({ forms: [], doors: [] })).length, 'the door is what answers');
});
