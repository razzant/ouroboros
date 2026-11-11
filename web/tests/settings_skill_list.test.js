// The list of skill forms (docs/DESIGN.md "Settings"): `skill` without an id is one door per skill that registered a
// settings form, so "where are the settings of my skills" has an answer in Settings and search has a page to land
// on. An empty list says that no installed skill has settings; a list that could not be read claims nothing.
import assert from 'node:assert/strict';
import test from 'node:test';

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
    const container = new Stub('div');
    const page = settingsPageFor('skill');
    assert.equal(typeof page.title === 'function' ? page.title('skill') : page.title, 'Skill settings');
    const handle = await page.mount(container, { path: 'skill', ctx: { openSettings: (path) => opened.push(path) } });
    await settle();
    return { env, container, opened, handle, doors: () => container.querySelectorAll('[data-setting-door]') };
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

test('search has a door to the list, by the names it had: Extension Settings, under Advanced', () => {
    const entries = searchEntries({ forms: [] });
    for (const words of ['Skill settings', 'extension settings', 'ADVANCED', 'Settings of skills']) {
        const hit = searchSettings(words, entries).find((entry) => entry.id === 'door:skill');
        assert.deepEqual([hit?.to, hit?.row], ['skill', ''], words);
    }
    // "Advanced" also names what moved to This computer: both are offered, neither guessed.
    assert.ok(searchSettings('advanced', entries).some((entry) => entry.id === 'home:computer'));
    assert.ok(!searchSettings('extension settings', searchEntries({ forms: [], doors: [] })).length, 'the door is what answers');
});
