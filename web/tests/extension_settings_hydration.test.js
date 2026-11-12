// A declarative extension settings form used to render the bare schema: every
// select showed its FIRST option and every text field was empty, so pressing
// Save posted that blank form over the stored values (the bundled Telegram
// skill unbound its owner chat id that way). The host now reads the current
// values from the same route it posts to; these pin the outcomes of that read
// against the real renderer. A route without a GET (404/405) cannot say what is
// stored (#1552): its form starts locked until the owner chooses Edit anyway.
import assert from 'node:assert/strict';
import test from 'node:test';

import { readExtensionFormValues, renderExtensionSettingsSections } from '../modules/settings_extension_forms.js';

const UNREADABLE_NOTE = 'Current values could not be read; Save is disabled so it does not overwrite them. '
    + 'Open this page again to retry.';
const UNAVAILABLE_NOTE = 'Current values are unavailable. These are defaults. Sending this form may change saved settings.';

const FORM = {
    type: 'form',
    id: 'settings-form',
    route: 'save',
    submit_label: 'Save',
    fields: [
        {
            name: 'mode', label: 'Mode', type: 'select', default: 'safe',
            options: [{ label: 'Safe', value: 'safe' }, { label: 'Fast', value: 'fast' }],
        },
        { name: 'chat_id', label: 'Chat ID', type: 'text' },
        { name: 'token', label: 'Token', type: 'password' },
    ],
};

const sections = (component = FORM) => [{
    skill: 'demo', section_id: 'config', key: 'demo:config', title: 'Demo', render: { components: [component] },
}];

/** Minimal render target: the renderer only writes innerHTML and binds forms. */
function fakeHost() {
    const host = { innerHTML: '', querySelectorAll: () => [] };
    return { host, root: { querySelector: (sel) => (sel === '#extension-settings-sections' ? host : null) } };
}

function stubFetch(handler) {
    const calls = [];
    const prior = globalThis.fetch;
    globalThis.fetch = async (url, init) => {
        calls.push({ url, init });
        return handler(url, init);
    };
    return { calls, restore: () => { globalThis.fetch = prior; } };
}

const jsonResponse = (body, status = 200) => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
});

async function render(handler, { component = FORM, isCurrent } = {}) {
    const { host, root } = fakeHost();
    const fetcher = stubFetch(handler);
    try {
        await renderExtensionSettingsSections(root, sections(component), isCurrent ? { isCurrent } : undefined);
    } finally {
        fetcher.restore();
    }
    return { html: host.innerHTML, calls: fetcher.calls };
}

test('(a) a stored document hydrates the form, and a password never comes back', async () => {
    const { html, calls } = await render(() => jsonResponse({ mode: 'fast', chat_id: '42', token: 'leaked-secret' }));

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/extensions/demo/save');
    assert.ok(!calls[0].init?.method, 'the hydration read is a plain GET on the POST route');
    assert.match(html, /<option value="fast" selected>Fast<\/option>/);
    assert.doesNotMatch(html, /<option value="safe" selected>/);
    assert.match(html, /name="chat_id"[^>]* value="42"/);
    // Passwords are never hydrated by renderSafeField; nothing may leak into the DOM.
    assert.doesNotMatch(html, /leaked-secret/);
    assert.match(html, /type="submit">Save<\/button>/);
    assert.doesNotMatch(html, /data-extension-settings-blocked/);
    assert.doesNotMatch(html, new RegExp(UNREADABLE_NOTE.slice(0, 30)));
});

test('(b) a route with no GET handler reads as UNKNOWN: defaults shown locked, Save withheld until Edit anyway', async () => {
    for (const status of [404, 405]) {
        const { html } = await render(() => jsonResponse({ error: 'nope' }, status));
        assert.match(html, /data-extension-settings-blocked="1"/, `status ${status}`);
        assert.ok(html.includes(`data-extension-settings-read data-tone="warn">${UNAVAILABLE_NOTE}</div>`), `status ${status}`);
        assert.ok(html.indexOf(UNAVAILABLE_NOTE) < html.indexOf('name="mode"'), 'the warning comes before the values it qualifies');
        // The declared defaults are shown, never as the owner's values: every field is locked.
        assert.match(html, /<select class="ui-control" name="mode"[^>]* disabled>/, `status ${status}`);
        assert.match(html, /name="chat_id"[^>]* value="" disabled>/, `status ${status}`);
        assert.match(html, /type="submit" disabled>Save<\/button>/, `status ${status}`);
        assert.match(html, /type="button" data-extension-settings-edit-anyway>Edit anyway<\/button>/, `status ${status}`);
        assert.doesNotMatch(html, new RegExp(UNREADABLE_NOTE.slice(0, 30)), `status ${status}`);
    }
});

test('a successful empty object is a real empty document, and an action with fields is read like a form', async () => {
    const { html } = await render(() => jsonResponse({}));
    assert.match(html, /<option value="safe" selected>Safe<\/option>/);
    assert.match(html, /type="submit">Save<\/button>/);
    assert.doesNotMatch(html, /data-extension-settings-blocked|data-extension-settings-read|Edit anyway/);

    // `action` with fields is configuration too: no GET handler is the same unknown, not "stateless".
    const action = await render(() => jsonResponse({}, 405), { component: { ...FORM, type: 'action', submit_label: 'Apply' } });
    assert.equal(action.calls.length, 1);
    assert.match(action.html, /type="submit" disabled>Apply<\/button>/);
    assert.match(action.html, /Edit anyway/);
});

test('a fieldless action is never read and stays a plain enabled button', async () => {
    const { html, calls } = await render(() => jsonResponse({}, 405), {
        component: { type: 'action', id: 'refresh', route: 'refresh', label: 'Refresh status' },
    });
    assert.deepEqual(calls, [], 'an action route is a side effect, never probed');
    assert.match(html, /type="submit">Refresh status<\/button>/);
    assert.doesNotMatch(html, /data-extension-settings-blocked|data-extension-settings-read|Edit anyway/);
});

test('(c) an unreadable read disables Save and says why, in the shared warn tone', async () => {
    const unreadable = [
        ['5xx', () => jsonResponse({}, 500)],
        ['other 4xx', () => jsonResponse({}, 403)],
        ['non-JSON body', () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError('not json'); } })],
        ['non-object body', () => jsonResponse([1, 2, 3])],
        ['network error', () => { throw new TypeError('Failed to fetch'); }],
    ];
    for (const [label, handler] of unreadable) {
        const { html } = await render(handler);
        assert.match(html, /data-extension-settings-blocked="1"/, label);
        assert.match(html, /type="submit" disabled>Save<\/button>/, label);
        assert.match(html, /data-extension-settings-status data-tone="warn">/, label);
        assert.ok(html.includes(UNREADABLE_NOTE), label);
        // The form still renders; only Save is withheld, and nothing offers a way around it.
        assert.match(html, /<option value="safe" selected>Safe<\/option>/, label);
        assert.doesNotMatch(html, /Edit anyway|data-extension-settings-read/, label);
    }
});

test('a stale load never overwrites a newer render', async () => {
    const { html, calls } = await render(() => jsonResponse({ mode: 'fast' }), { isCurrent: () => false });
    assert.equal(calls.length, 1, 'the read still happens; only the write is withheld');
    assert.equal(html, '');
});

test('a component without a usable route is never read from', async () => {
    const { html, calls } = await render(() => jsonResponse({}), {
        component: { ...FORM, route: '../escape' },
    });
    assert.deepEqual(calls, []);
    assert.match(html, /Invalid extension settings route/);
});

test('readExtensionFormValues classifies every outcome of the read', async () => {
    const outcomes = [
        [jsonResponse({ mode: 'fast' }), { values: { mode: 'fast' }, read: 'ok' }],
        [jsonResponse({}), { values: {}, read: 'ok' }],
        [jsonResponse({}, 404), { values: {}, read: 'unsupported' }],
        [jsonResponse({}, 405), { values: {}, read: 'unsupported' }],
        [jsonResponse({}, 500), { values: {}, read: 'failed' }],
        [jsonResponse(null), { values: {}, read: 'failed' }],
    ];
    for (const [response, expected] of outcomes) {
        const fetcher = stubFetch(() => response);
        try {
            assert.deepEqual(await readExtensionFormValues('demo', 'save'), expected);
        } finally {
            fetcher.restore();
        }
    }
});

/** One rendered form as the handlers see it: its controls, named fields and listeners (no layout). */
function formDouble(fieldNames) {
    const control = (extra = {}) => ({
        disabled: true, dataset: {}, textContent: '', listeners: {},
        addEventListener(type, fn) { this.listeners[type] = fn; }, ...extra,
    });
    const inputs = Object.fromEntries(fieldNames.map((name) => [name, control({ value: `typed ${name}` })]));
    const parts = {
        '[data-extension-settings-edit-anyway]': control({ disabled: false }),
        'button[type="submit"]': control(),
        '[data-extension-settings-read]': control(),
        '[data-extension-settings-status]': control(),
    };
    const form = control({
        dataset: { extensionSettingsKey: 'demo:config:settings-form', skill: 'demo', route: 'save', extensionSettingsBlocked: '1' },
        elements: { namedItem: (name) => inputs[name] || null },
        querySelector: (selector) => parts[selector] || null,
    });
    return { form, inputs, parts };
}

test('Edit anyway is the owner\'s explicit choice: it unlocks the form, and only then does Save send every field', async () => {
    const component = { ...FORM, fields: [...FORM.fields, { name: 'fixed', label: 'Fixed', type: 'text', disabled: true }] };
    const { form, inputs, parts } = formDouble(['mode', 'chat_id', 'token', 'fixed']);
    const host = { innerHTML: '', querySelectorAll: (selector) => (selector === '[data-extension-settings-form]' ? [form] : []) };
    const root = { querySelector: (selector) => (selector === '#extension-settings-sections' ? host : null) };
    const posts = [];
    const fetcher = stubFetch((url, init) => {
        if (init?.method === 'POST') { posts.push(JSON.parse(init.body)); return jsonResponse({ message: 'Saved.' }); }
        return jsonResponse({ error: 'Method Not Allowed' }, 405);   // a POST-only legacy route
    });
    try {
        await renderExtensionSettingsSections(root, sections(component));
        await form.listeners.submit({ preventDefault() {} });
        assert.deepEqual(posts, [], 'locked: Save sends nothing before the owner chooses');

        const editAnyway = parts['[data-extension-settings-edit-anyway]'];
        editAnyway.listeners.click({ currentTarget: editAnyway });
        assert.deepEqual(Object.fromEntries(Object.entries(inputs).map(([name, input]) => [name, input.disabled])),
            { mode: false, chat_id: false, token: false, fixed: true }, 'a field the schema disables stays disabled');
        assert.equal(parts['button[type="submit"]'].disabled, false);
        assert.equal(editAnyway.disabled, true);
        assert.equal(form.dataset.extensionSettingsBlocked, undefined);
        const note = parts['[data-extension-settings-read]'];
        assert.equal(note.textContent, 'You chose to edit without reading the saved values. Review every field before sending.');
        assert.equal(note.dataset.tone, 'warn');

        await form.listeners.submit({ preventDefault() {} });
        assert.deepEqual(posts, [{ mode: 'typed mode', chat_id: 'typed chat_id', token: 'typed token', fixed: 'typed fixed' }],
            'every displayed field is sent, as an explicit replacement of what nobody could read');
        assert.equal(parts['[data-extension-settings-status]'].textContent, 'Saved.');
    } finally {
        fetcher.restore();
    }
});

test('a form of read-only fields is a status: its check refreshes those fields in place and saves nothing', async () => {
    const component = {
        type: 'form', id: 'connection', route: 'connection', method: 'POST', submit_label: 'Refresh status', busy_label: 'Checking…',
        fields: ['token_status', 'bridge_status', 'miniapp_status'].map((name) => ({ name, label: name, type: 'text', disabled: true })),
    };
    const reads = [
        { token_status: 'Not set (Settings → Secrets)', bridge_status: 'Waiting for a bot token', miniapp_status: 'Waiting for a bot token' },
        { token_status: 'Saved', bridge_status: 'Connected', miniapp_status: 'Ready', message: 'Status checked.' },
        { token_status: "Can't read saved settings now; the current token stays in use", bridge_status: 'Connected', miniapp_status: 'Ready', message: 'Status checked.' },
    ];
    const { html } = await render(() => jsonResponse(reads[0]), { component });
    assert.match(html, /class="btn btn-default" type="submit">Refresh status<\/button>/, 'a neutral check, not the green Save');
    assert.match(html, /name="bridge_status"[^>]* value="Waiting for a bot token"[^>]* disabled/);

    const { form, inputs, parts } = formDouble(['token_status', 'bridge_status', 'miniapp_status']);
    form.dataset = { extensionSettingsKey: 'demo:config:connection', skill: 'demo', route: 'connection' };
    const host = { innerHTML: '', querySelectorAll: (selector) => (selector === '[data-extension-settings-form]' ? [form] : []) };
    const root = { querySelector: (selector) => (selector === '#extension-settings-sections' ? host : null) };
    const events = [];
    const priorWindow = globalThis.window;
    globalThis.window = { dispatchEvent: (event) => events.push(event.type) };
    let answer = 0;
    const busy = [];
    const fetcher = stubFetch((url, init) => {
        if (init?.method !== 'POST') return jsonResponse(reads[0]);
        busy.push([parts['button[type="submit"]'].textContent, parts['[data-extension-settings-status]'].textContent]);
        answer += 1;
        return jsonResponse(reads[answer]);
    });
    try {
        await renderExtensionSettingsSections(root, sections(component));
        for (const expected of [reads[1], reads[2]]) {
            await form.listeners.submit({ preventDefault() {} });
            for (const name of ['token_status', 'bridge_status', 'miniapp_status']) assert.equal(inputs[name].value, expected[name]);
            assert.equal(parts['[data-extension-settings-status]'].textContent, 'Status checked.');
            assert.equal(parts['button[type="submit"]'].textContent, 'Refresh status');
        }
        assert.deepEqual(busy, [['Checking…', 'Checking…'], ['Checking…', 'Checking…']], 'never “Saving…”');
        assert.deepEqual(events.filter((type) => type === 'ouro:settings-updated'), [], 'a check announces no save');
    } finally {
        fetcher.restore();
        globalThis.window = priorWindow;
    }
});

test('an editable form still saves, keeps what the owner typed and announces the save', async () => {
    const { form, inputs } = formDouble(['mode', 'chat_id', 'token']);
    form.dataset = { extensionSettingsKey: 'demo:config:settings-form', skill: 'demo', route: 'save' };
    const host = { innerHTML: '', querySelectorAll: (selector) => (selector === '[data-extension-settings-form]' ? [form] : []) };
    const root = { querySelector: (selector) => (selector === '#extension-settings-sections' ? host : null) };
    const events = [];
    const priorWindow = globalThis.window;
    globalThis.window = { dispatchEvent: (event) => events.push(event.type) };
    const fetcher = stubFetch((url, init) => jsonResponse(init?.method === 'POST' ? { message: 'Saved.', chat_id: 'server' } : {}));
    try {
        await renderExtensionSettingsSections(root, sections());
        await form.listeners.submit({ preventDefault() {} });
        assert.equal(inputs.chat_id.value, 'typed chat_id', 'an editable field is never overwritten by the answer');
        assert.ok(events.includes('ouro:settings-updated'));
    } finally {
        fetcher.restore();
        globalThis.window = priorWindow;
    }
});

test('Telegram\'s Connection shows each long production sentence whole, posts what it shows and refreshes in place', async () => {
    const long = "Restoring the previous owner's menu button: Telegram did not confirm the restored menu button.";
    const names = ['token_status', 'bridge_status', 'miniapp_status'];
    // The schema skills/telegram/lib/telegram_connection.py registers: read-only multiline values.
    const component = {
        type: 'form', id: 'connection', route: 'connection', method: 'POST', submit_label: 'Refresh status', busy_label: 'Checking…',
        fields: [['token_status', 'Bot token'], ['bridge_status', 'Text bridge'], ['miniapp_status', 'Mini App']]
            .map(([name, label]) => ({ name, label, type: 'textarea', disabled: true })),
    };
    const reads = [
        { token_status: "Can't read saved settings now; the current token stays in use",
          bridge_status: 'Stopped: Telegram refused this token; save a new one in Secrets', miniapp_status: long },
        { token_status: 'Saved', bridge_status: 'Connecting with the saved token…',
          miniapp_status: 'Applying the token once the old Mini App has stopped', message: 'Status checked.' },
    ];
    const { html } = await render(() => jsonResponse(reads[0]), { component });
    for (const name of names) {
        const label = component.fields.find((field) => field.name === name).label;
        const whole = reads[0][name].replace(/'/g, '&#39;');
        assert.ok(html.includes(`<output class="ui-control ui-control-readonly" name="${name}" aria-label="${label}" tabindex="0">${whole}</output>`), name);
    }
    assert.doesNotMatch(html, /<textarea|resize|rows=/);

    const { form, inputs, parts } = formDouble(names);
    for (const name of names) Object.assign(inputs[name], { type: 'output', value: reads[0][name] });
    form.dataset = { extensionSettingsKey: 'demo:config:connection', skill: 'demo', route: 'connection' };
    const host = { innerHTML: '', querySelectorAll: (selector) => (selector === '[data-extension-settings-form]' ? [form] : []) };
    const root = { querySelector: (selector) => (selector === '#extension-settings-sections' ? host : null) };
    const events = [];
    const posts = [];
    const priorWindow = globalThis.window;
    globalThis.window = { dispatchEvent: (event) => events.push(event.type) };
    const fetcher = stubFetch((url, init) => {
        if (init?.method !== 'POST') return jsonResponse(reads[0]);
        posts.push(JSON.parse(init.body));
        return jsonResponse(reads[1]);
    });
    try {
        await renderExtensionSettingsSections(root, sections(component));
        await form.listeners.submit({ preventDefault() {} });
        assert.deepEqual(posts, [{ token_status: reads[0].token_status, bridge_status: reads[0].bridge_status, miniapp_status: long }],
            'the check posts exactly the status it showed, as the textareas did');
        for (const name of names) assert.equal(inputs[name].value, reads[1][name], `${name} refreshed in place`);
        assert.equal(parts['[data-extension-settings-status]'].textContent, 'Status checked.');
        assert.deepEqual(events.filter((type) => type === 'ouro:settings-updated'), [], 'a check announces no save');
    } finally {
        fetcher.restore();
        globalThis.window = priorWindow;
    }
});
