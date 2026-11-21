import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { createChatInstance } from '../modules/chat.js';
import { createUnconfirmedSends } from '../modules/chat_attachments.js';
import { desktopRoomStorage, finishDesktopStartup, openDesktopRoomSession, roomStorage } from '../modules/desktop_room_session.js';
import { createLayerStack } from '../modules/layers.js';
import { bindPageBack } from '../modules/page_header.js';
import { capturePagePlace, createPageNavigation } from '../modules/page_navigation.js';
import { createRoomRetirement, roomSocket } from '../modules/room_retire.js';
import { WS } from '../modules/ws.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

function surface(postMessage) {
    const meta = { content: 'one-navigation', remove() {} };
    const doc = { querySelector: () => meta, documentElement: { dataset: {} } };
    const host = Object.assign(new EventTarget(), { webkit: { messageHandlers: { ouroborosRoomSession: { postMessage } } } });
    host.top = host;
    return { host, doc };
}

function deferred() {
    let resolve, reject;
    const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
    return { promise, resolve, reject };
}

function chat(env, ws, projectId = '', chatId = 1) {
    return createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        chatId, projectId, idPrefix: projectId ? `pchat-${projectId}` : 'chat', mountEl: env.mount, asPanel: !!projectId,
        stateSnapshots: { begin: () => ({}), gate: async () => ({}), isCurrent: () => true, apply() {}, fail() {}, latest: () => null } });
}

// app.js boots the SPA on import. Run its exact consumer functions without booting an interface.
function appFunctions(...names) {
    const source = readFileSync(new URL('../app.js', import.meta.url), 'utf8').replace(/\r\n?/g, '\n');
    return names.map(name => {
        const match = source.match(new RegExp(`^(?:async )?function ${name}\\(.*?^}`, 'ms'));
        assert.ok(match, name);
        return match[0];
    }).join('\n');
}

function exactProjectReader() {
    return vm.runInNewContext(`${appFunctions('resolveProjectRoom')}\nresolveProjectRoom;`, {
        projectNavigationGeneration: 1, fmt: text => text,
    });
}

const jsonReply = (data, status = 200) => ({ status, ok: status === 200, json: async () => data });
const readyFrames = posts => posts.filter(post => post.op === 'ready');
function startupHydration(url, history, unexpected) {
    if (url === '/api/ui/preferences' || url === '/api/state') return jsonReply({});
    if (url === '/api/chat/history' || url === '/api/chat/history?chat_id=7') {
        history.push(url);
        return jsonReply({ messages: [], window: { complete: true } });
    }
    unexpected.push(url);  // preference/state readers can catch a fixture refusal
    assert.fail(`unexpected startup read: ${url}`);
}

test('ordinary tabs, Mini App and subframes use their own tab storage', () => {
    const { host, doc } = surface(() => assert.fail('unexpected native message'));
    const env = installDom();
    try {
        assert.equal(openDesktopRoomSession(host, { querySelector: () => null }), null);
        doc.documentElement.dataset.ouroborosHost = 'telegram';
        assert.equal(openDesktopRoomSession(host, doc), null);
        delete doc.documentElement.dataset.ouroborosHost;
        host.top = {};
        assert.equal(openDesktopRoomSession(host, doc), null);
        assert.equal(roomStorage(), globalThis.sessionStorage);
        let connected = 0;
        finishDesktopStartup(assert.fail, assert.fail, () => { connected++; });
        assert.equal(connected, 1);
    } finally { restoreDom(env.prior); }
});

test('the successor restores the room, names missing files, and reconciles the original frame without a socket write', async () => {
    let registryReads = 0;
    const env = installDom(async url => {
        if (url === '/api/projects?id=project7') {
            registryReads++;
            if (registryReads === 1) throw new TypeError('Connection closed during read');
            return { ok: true, json: async () => ({ projects: [{ id: 'project7', chat_id: 7, lifecycle: 'active' }] }) };
        }
        return { ok: true, json: async () => String(url).startsWith('/api/chat/history')
            ? { messages: [], window: { complete: true } }
            : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } };
    });
    const snapshot = deferred(), mountedRoom = deferred();
    const posts = [], sent = [];
    const { host, doc } = surface(message => {
        posts.push(message);
        if (message.op === 'open') return { mountProject: 'project7' };
        if (message.op === 'ready') { mountedRoom.resolve(); return snapshot.promise; }
        return true;
    });
    const frame = { type: 'chat', content: 'Earlier send', client_message_id: 'sent-once', sender_session_id: 'sender',
        project_id: 'project7', chat_id: 7, reply_to: 'row:7@a#1',
        attachments: [{ filename: 'stored-upload', display_name: 'sent.txt', mime: 'text/plain' }] };
    const quotedPhoto = '/api/files/download?upload=0123456789abcdef0123456789abcdef_photo.png';
    let room, retained;
    try {
        const opened = openDesktopRoomSession(host, doc, assert.fail);
        const ws = { on: () => () => {}, isConnected: () => true, send(frame) {
            sent.push({ ...frame, client_message_id: 'new-id' });
            return { status: 'sent', clientMessageId: 'new-id' };
        } };
        await Promise.resolve();
        assert.deepEqual(posts, [{ op: 'open', staged: true, ticket: 'one-navigation' }]);
        let connected = 0;
        const started = finishDesktopStartup(() => room, async (id, mounted, readProject) => {
            const { projects: [row] } = await readProject(id, () => true);
            assert.equal(row.id, 'project7');
            room = chat(env, ws, row.id, row.chat_id);
            mounted(room);
            return opened;  // the panel's history read can finish only after the native reply
        }, () => { connected++; });
        await Promise.race([mountedRoom.promise, started.then(() => assert.fail('started before the room mounted'))]);
        assert.equal(registryReads, 2);
        assert.deepEqual(posts.map(p => p.stage || p.op), ['open', 'mount', 'retry', 'ready']);
        assert.deepEqual(readyFrames(posts), [{ op: 'ready', project: 'project7', ticket: 'one-navigation' }]);
        assert.equal(connected, 0);
        retained = createUnconfirmedSends({ send: ws.send, root: () => null, onDomWrite: fn => fn(),
            storage: roomStorage(), storageKey: 'ouro_chat_unconfirmed:7', showToast: assert.fail });
        const send = env.mount.querySelector('#pchat-project7-send');
        assert.equal(room.composer.input.readOnly, true);
        assert.equal(room.composer.setDraft('not accepted before the native reply'), false);
        send.click();
        assert.deepEqual(sent, []);
        const announced = [];
        room.composer.input.addEventListener('input', (event) => announced.push([event.type, room.composer.input.value]));
        snapshot.resolve({ ouro_chat_active_project: 'project7', ouro_chat_session_id: 'sender', ouro_chat_draft: 'Main alone', 'ouro_chat_draft:7': '  Привет\n',
            'ouro_chat_files:7': '2', 'ouro_chat_reply:7': JSON.stringify({ chat_id: 7, role: 'assistant',
                address: frame.reply_to, snippet: 'Quoted words', media: { kind: 'image', available: true, url: quotedPhoto } }),
            'ouro_chat_unconfirmed:7': JSON.stringify([{ frame }]) });
        await started;
        await Promise.resolve();
        assert.equal(connected, 1);
        assert.equal(room.composer.getDraft(), '  Привет\n');
        // The restored draft announces itself as a field change: watchers that did not type — Send's
        // reserved Talk face among them — see the room's kept words instead of an empty field.
        assert.deepEqual(announced, [['input', '  Привет\n']], 'the restoration reached the field\'s watchers');
        assert.equal(room.composer.input.getAttribute('aria-describedby'), 'pchat-project7-input-reply');
        assert.equal(env.mount.querySelector('.chat-quote-thumb').src, quotedPhoto);
        const preview = env.mount.querySelector('#pchat-project7-attachment-preview');
        assert.match(preview.children[0].textContent, /2 files again/);
        const shown = [];
        retained.reconcile(entry => shown.push(entry.frame));
        assert.deepEqual(shown, [frame]);
        assert.deepEqual(sent, [], 'restoring a held frame never writes it');
        assert.equal(roomStorage().getItem('ouro_chat_draft'), 'Main alone');
        assert.equal(globalThis.sessionStorage.getItem('ouro_chat_draft:7'), null);
        assert.equal(room.composer.input.readOnly, false);
        room.composer.setDraft(`${room.composer.getDraft()}!`);
        await Promise.resolve();
        assert.deepEqual(posts.filter(x => x.op === 'update').flatMap(x => JSON.parse(x.changes))
            .filter(([key]) => key === 'ouro_chat_draft:7'), [['ouro_chat_draft:7', '  Привет\n!']]);
        send.click();
        assert.equal(sent.length, 1);
        assert.deepEqual([sent[0].chat_id, sent[0].project_id, sent[0].sender_session_id, sent[0].reply_to,
            sent[0].client_message_id], [7, 'project7', 'sender', frame.reply_to, 'new-id']);
        assert.equal(roomStorage().getItem('ouro_chat_draft'), 'Main alone');
    } finally { retained?.release(); room?.destroy(); restoreDom(env.prior); }
});

test('an empty Main must still mount an editable field before the shell can release its socket', async () => {
    const posts = [], final = deferred();
    const { host, doc } = surface(message => {
        posts.push(message.op);
        return message.op === 'open' ? { mountProject: '' } : final.promise;
    });
    openDesktopRoomSession(host, doc, assert.fail);
    await Promise.resolve();
    assert.deepEqual(posts, ['open']);
    let connected = 0;
    const main = { projectId: '', composer: { input: { isConnected: true } } };
    const startup = finishDesktopStartup(() => main, assert.fail, () => { connected++; });
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(posts, ['open', 'ready']);
    assert.equal(connected, 0);
    final.resolve({});
    await startup;
    assert.equal(connected, 1);
});

for (const boot of [{ phase: 'failed' }, { phase: 'ready', jobs: { dialogue_import: { state: 'failed' } } }]) test(
    `a reported ${boot.phase === 'failed' ? 'boot' : 'archive import'} failure ends only the Project restore`, async () => {
    const reads = [], posts = [], warnings = [];
    const env = installDom(async url => {
        reads.push(url);
        return url === '/api/health' ? jsonReply({ boot }) : jsonReply({ reason_code: 'migrating' }, 503);
    });
    try {
        const { host, doc } = surface(message => {
            posts.push(message);
            if (message.op === 'open') return { mountProject: 'p7' };
            return message.op === 'ready' ? {} : true;
        });
        openDesktopRoomSession(host, doc, text => warnings.push(text));
        const main = { projectId: '', composer: { input: { isConnected: true } } };
        const resolve = exactProjectReader();
        let connected = 0;
        await finishDesktopStartup(() => main, (id, _mounted, read, unavailable) =>
            resolve({ id }, 1, read, unavailable), () => { connected++; });
        assert.deepEqual(reads, ['/api/projects?id=p7', '/api/health']);
        assert.deepEqual(readyFrames(posts), [{ op: 'ready', project: 'p7', selection: true, ticket: 'one-navigation' }]);
        assert.equal(connected, 1);
        assert.equal(desktopRoomStorage().inputLocked(), true);
        assert.deepEqual(warnings, ['Startup reported a failure while restoring your last Project. Choose Main or another Project to continue.']);
    } finally { restoreDom(env.prior); }
});

test('failed room startup never connects the rejected window, even with no held input', async () => {
    const field = (projectId, props) => ({ projectId, composer: { input: { isConnected: true, ...props } } });
    const main = field(''), project = field('p7');
    for (const [why, shown, mounted, refuse] of [
        ['no active chat', null], ['composer failed to mount', { projectId: '' }],
        ['field detached', field('', { isConnected: false })], ['field disabled', field('', { disabled: true })],
        ['field read-only', field('', { readOnly: true })], ['another project returned', field('other'), field('other')],
        ['room changed after mount', main, project], ['native open refused', main, null, 'open'],
        ['native ready refused', main, null, 'ready'],
    ]) {
        const messages = [], warnings = [], { host, doc } = surface(message => {
            messages.push(message.op);
            if (message.op === refuse) return Promise.reject(new Error(why));
            return message.op === 'open' ? { mountProject: mounted ? 'p7' : '' } : {};
        });
        openDesktopRoomSession(host, doc, text => warnings.push(text));
        let connections = 0;
        await finishDesktopStartup(() => shown, (id, accept) => accept(mounted), () => { connections++; });
        assert.equal(connections, 0, why);
        assert.deepEqual(messages, refuse === 'ready' ? ['open', 'ready'] : ['open'], why);
        assert.equal(warnings.length, 1, why);
    }
});

test('a plain desktop snapshot keeps Main read-only while restoring its project; explicit Main wins immediately', async () => {
    for (const scenario of ['restore', 'navigate', 'legacy']) {
        const navigate = scenario === 'navigate';
        const firstHealth = deferred(), reads = [], posts = [];
        let first = true, healthSignal;
        const env = installDom(async (url, init) => {
            if (url === '/api/health' || url.startsWith('/api/projects?id=')) reads.push(url);
            return { ok: true, status: 200, json: async () => {
                if (url === '/api/health') {
                    if (first) { first = false; healthSignal = init.signal; return firstHealth.promise; }
                    if (scenario === 'legacy') throw new SyntaxError('empty health body');
                    return { status: 'ok' };
                }
                if (url.startsWith('/api/projects?id=')) return { projects: [{ id: 'p7', chat_id: 7, lifecycle: 'active' }] };
                return { messages: [], window: { complete: true } };
            } };
        });
        let main, project;
        try {
            const { host, doc } = surface(message => {
                posts.push(message.op);
                return { ouro_chat_active_project: 'p7', 'ouro_chat_draft:7': 'kept' };
            });
            openDesktopRoomSession(host, doc, assert.fail);
            const ws = { on: () => () => {}, isConnected: () => true, send: assert.fail };
            main = chat(env, ws);
            let connections = 0, current = true, restored;
            const started = finishDesktopStartup(() => project || main, async (id, mounted, read) => {
                restored = await read(id, () => current);
                if (current) { project = chat(env, ws, id, restored.projects[0].chat_id); mounted(project); }
            }, () => { connections++; });
            await new Promise(resolve => setImmediate(resolve));
            assert.equal(connections, 1);
            assert.deepEqual(reads, ['/api/health']);
            assert.equal(main.composer.input.readOnly, true);
            assert.equal(main.composer.setDraft('wrong address'), false);
            env.mount.querySelector('#chat-send').click();
            current = !navigate;
            if (navigate) {
                desktopRoomStorage().selectRoom(''); // Main's explicit navigation commit
                await started;  // does not need the outstanding health response
                assert.equal(healthSignal.aborted, true);
                assert.equal(main.composer.input.readOnly, false);
                main.composer.setDraft('Now in Main');
            }
            if (scenario === 'legacy') firstHealth.reject(new SyntaxError('empty health body'));
            else firstHealth.resolve({ boot: { phase: 'serving', steps: [] } });
            await started;
            await new Promise(resolve => setImmediate(resolve));
            assert.deepEqual(reads, navigate ? ['/api/health'] : ['/api/health', '/api/health', '/api/projects?id=p7']);
            assert.equal(restored?.projects[0].id, navigate ? undefined : 'p7');
            assert.equal(posts[0], 'open');
            assert.equal(roomStorage().getItem('ouro_chat_draft:7'), 'kept');
            assert.equal((project || main).composer.getDraft(), navigate ? 'Now in Main' : 'kept');
            assert.equal(roomStorage().getItem('ouro_chat_active_project'), navigate ? null : 'p7');
        } finally { project?.destroy(); main?.destroy(); restoreDom(env.prior); }
    }
});

test('source exec waits for readable startup past twelve seconds', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const turn = () => new Promise(resolve => setImmediate(resolve));
    const reads = [];
    let clock = 0, connected = false;
    const field = projectId => ({ projectId, composer: { input: { isConnected: true } } });
    let visible = field('');
    const id = 'p7', rows = { projects: [{ id }] }, slots = { ouro_chat_active_project: id, 'ouro_chat_draft:7': 'kept' };
    const env = installDom(async url => {
        reads.push([clock, url]);
        if (url === '/api/health') return jsonReply({ boot: { phase: 'serving',
            steps: [{ name: 'projects', state: clock < 13_300 ? 'running' : 'done' }] } });
        return jsonReply(rows);
    });
    try {
        const { host, doc } = surface(message => {
            assert.equal(message.op, 'open');
            return slots;
        });
        openDesktopRoomSession(host, doc, assert.fail);
        const started = finishDesktopStartup(() => visible, async (id, mounted, read) => {
            const data = await read(id, () => true);
            assert.equal(data?.projects[0].id, 'p7');
            visible = field(id); mounted(visible);
        }, () => { connected = true; });
        await turn();
        while (clock < 13_300) {
            assert.equal(connected, true);
            assert.equal(desktopRoomStorage().inputLocked(), true);
            clock += 700;
            t.mock.timers.tick(700);
            await turn();
        }
        await started;
        assert.equal(connected, true);
        assert.equal(desktopRoomStorage().inputLocked(), false);
        assert.equal(roomStorage().getItem('ouro_chat_draft:7'), 'kept');
        assert.ok(reads.some(([at, url]) => at === 13_300 && url === '/api/projects?id=p7'));
        assert.equal(reads.filter(([, url]) => url === '/api/projects?id=p7').length, 1);
    } finally { restoreDom(env.prior); }
});

for (const chooseMain of [false, true]) test(chooseMain ? 'explicit Main wins an outstanding staged Project lookup'
    : 'the staged shell stays responsive and restores its exact Project after a minute without advancing boot facts', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let lookups = 0, polls = 0, connects = 0, lastLookup, healthSignal;
    const turn = () => new Promise(resolve => setImmediate(resolve));
    const advance = async ms => { t.mock.timers.tick(ms); await turn(); };
    const body = (ms, payload) => new Promise(resolve => setTimeout(() => resolve(payload), ms));
    const id = 'p7', expected = { id, chat_id: 7, lifecycle: 'active' };
    const slots = { ouro_chat_active_project: id, 'ouro_chat_draft:7': 'kept' };
    const posts = [], warnings = [], history = [], unexpected = [], mountReply = deferred(), snapshot = deferred();
    const env = installDom(async (url, init) => {
        if (url === `/api/projects?id=${id}`) {
            lastLookup = init.signal;
            const migrating = ++lookups <= (chooseMain ? 1 : 4);
            return { ok: !migrating, status: migrating ? 503 : 200, json: () => body(11_000,
                migrating ? { reason_code: 'migrating' } : { projects: [expected] }) };
        }
        if (url === '/api/health') {
            healthSignal = init.signal;
            const jobs = ++polls % 2 ? { dialogue_import: { state: 'running', done: 1 } } : {};
            return { ok: true, json: () => body(chooseMain ? 20_000 : 10_000, { boot: { phase: 'ready', jobs } }) };
        }
        return startupHydration(url, history, unexpected);
    });
    let main, project;
    try {
        const { host, doc } = surface(message => {
            posts.push(message);
            if (message.stage === 'mount') return mountReply.promise;  // the receipt must not delay HTTP
            return message.op === 'open' ? { mountProject: id } : message.op === 'ready' ? snapshot.promise : true;
        });
        openDesktopRoomSession(host, doc, text => warnings.push(text));
        const ws = { on: () => () => {}, isConnected: () => false, send: assert.fail };
        main = chat(env, ws);
        const resolve = exactProjectReader();
        const started = finishDesktopStartup(() => project || main, async (key, mounted, read, unavailable) => {
            const row = await resolve({ id: key }, 1, read, unavailable);
            if (row) { project = chat(env, ws, row.id, row.chat_id); mounted(project); }
        }, () => { connects++; });
        const ready = { op: 'ready', project: id, selection: true, ticket: 'one-navigation' };
        await turn();
        assert.deepEqual(posts.map(p => p.stage || p.op), ['open', 'mount']);
        await advance(11_000);
        assert.deepEqual(readyFrames(posts), [ready], 'the first typed 503 is enough to present the shell');
        assert.deepEqual([polls, connects], [0, 0], 'wait for native acceptance, not health');
        assert.deepEqual(history, []);
        snapshot.resolve(slots);
        await turn();
        assert.equal(connects, 1);
        assert.equal(polls, 1, 'restoration continues in the presented document');
        assert.equal(main.composer.input.readOnly, true);
        assert.match(main.composer.input.placeholder, /Restoring the room/);
        assert.equal(main.composer.setDraft('wrong address'), false);
        env.mount.querySelector('#chat-send').click();
        assert.equal(main.composer.getDraft(), '');
        assert.deepEqual(history, ['/api/chat/history']);
        mountReply.resolve(false);  // an obsolete progress receipt is not the presentation receipt
        await advance(chooseMain ? 12_000 : 10_000);
        await advance(700);
        if (chooseMain) {
            assert.equal(healthSignal.aborted, true, 'a missed advisory health reply did not end restoration');
            assert.equal(lookups, 2);
            desktopRoomStorage().selectRoom('');
            await started;
            assert.equal(lastLookup.aborted, true);
            assert.equal(main.composer.input.readOnly, false);
            main.composer.setDraft('Main by choice');
            await advance(11_000);  // actual late body for p7
            assert.equal(project, undefined);
            assert.equal(main.composer.getDraft(), 'Main by choice');
            assert.equal(roomStorage().getItem('ouro_chat_active_project'), null);
            assert.deepEqual(history, ['/api/chat/history']);
        } else {
            for (let index = 0; index < 3; index++) {
                await advance(11_000);
                await advance(10_000);
                assert.equal(main.composer.input.readOnly, true);
                assert.equal(project, undefined);
                await advance(700);
            }
            await advance(11_000);
            assert.equal(lookups, 5); // 97.8 seconds since the first lookup, no advancing health facts
            assert.equal(polls, 4);
            assert.ok(project, 'positive exact registry response mounted the saved Project');
            await started;
            assert.equal(project.composer.getDraft(), 'kept');
            assert.equal(project.composer.input.readOnly, false);
            assert.equal(roomStorage().getItem('ouro_chat_active_project'), id);
            assert.deepEqual(history, ['/api/chat/history', '/api/chat/history?chat_id=7']);
        }
        assert.equal(roomStorage().getItem('ouro_chat_draft:7'), 'kept');
        assert.deepEqual(readyFrames(posts), [ready]);
        assert.deepEqual(posts.filter(p => p.op === 'project_importing').map(p => p.stage), ['mount']);
        assert.deepEqual(warnings, []);
        assert.deepEqual(unexpected, []);
    } finally { project?.destroy(); main?.destroy(); restoreDom(env.prior); }
});

test('a hung response body in staged startup ends that read and presents a locked room picker', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const turn = () => new Promise(resolve => setImmediate(resolve));
    let signal;
    const posts = [], warnings = [], stalled = deferred();
    const env = installDom(async (_url, init) => {
        assert.equal(signal, undefined, 'a hung read is not repeated');
        signal = init.signal;
        return { ok: true, json: () => stalled.promise };
    });
    try {
        const { host, doc } = surface(message => {
            posts.push(message);
            return message.op === 'open' ? { mountProject: 'p7' } : { ouro_chat_active_project: 'p7' };
        });
        openDesktopRoomSession(host, doc, msg => warnings.push(msg));
        const main = { projectId: '', composer: { input: { isConnected: true } } };
        const started = finishDesktopStartup(() => main, async (id, mounted, read, unavailable) => {
            try { if (await read(id, () => true)) mounted(main); }
            catch { unavailable('unreadable'); }
        }, () => {});
        await turn();
        assert.equal(signal.aborted, false);
        t.mock.timers.tick(12_000);
        await started;
        assert.equal(signal.aborted, true);
        assert.deepEqual(readyFrames(posts), [{ op: 'ready', project: 'p7', selection: true, ticket: 'one-navigation' }]);
        assert.match(warnings[0], /could not be read.*Choose Main or another Project/);
        stalled.resolve({ projects: [{ id: 'p7' }] });
        await turn();
        assert.equal(desktopRoomStorage().inputLocked(), true);
    } finally { restoreDom(env.prior); }
});

for (const [exit, detail] of [['leaving', 1], ['leaving', 0], ['settled', 1], ['settled', 0], ['deleting', 0], ['gone', 0]]) test(
    `Project → Accounts → ${detail ? 'pointer' : 'keyboard'} Back while ${exit} hands off only the returned room`, async t => {
    const env = installDom(), win = new EventTarget();
    const ws = { on: () => () => {}, isConnected: () => false, send: assert.fail };
    let drained, main, project, restored;
    const outgoing = surface(message => {
        if (message.op === 'drained') drained = JSON.parse(message.items);
        return message.op === 'open' ? {} : true;
    });
    const navDoc = Object.assign(new EventTarget(), { createElement: globalThis.document.createElement,
        getElementById: () => ({ classList: { add() {} } }), querySelectorAll: () => [] });
    const layers = createLayerStack({ win, doc: navDoc });
    const header = navDoc.createElement('div');
    header.innerHTML = '<span data-page-return></span>';
    const state = { activePage: 'chat', settingsPath: '' }, navState = { activeProjectId: null };
    const unbind = bindPageBack({ layers, doc: navDoc, win, currentHeader: () => state.activePage === 'settings' ? header : null });
    const returns = [];
    let nav;
    try {
        const opened = openDesktopRoomSession(outgoing.host, outgoing.doc, assert.fail);
        main = chat(env, roomSocket(ws));
        await finishDesktopStartup(() => main, assert.fail, () => {});
        await opened;
        project = chat(env, roomSocket(ws), 'p7', 7);
        const projectInstances = new Map([['p7', project]]), projectPanelTitle = { textContent: 'Project Seven' };
        const rooms = createRoomRetirement({ instances: projectInstances, cancelPaint: (_id, inst) => inst?.cancelHistoryPaint() });
        const sandbox = vm.createContext({ state, navState, rooms, projectInstances, mainChat: main, document: navDoc,
            projectNavigationGeneration: 0, projectReveals: new Map(), desktopRoomStorage, window: win,
            CustomEvent: class extends Event { constructor(type) { super(type); } },
            syncNavigationState() {}, updateUnreadBadge() {}, enterMain() {} });
        const commitPage = vm.runInContext(`${appFunctions('commitPage', 'closeProjectPanel')}\ncommitPage;`, sandbox);
        nav = createPageNavigation({ layers, doc: navDoc, canLeave: () => true, capture: opener => capturePagePlace({
            state, navState, rooms, mainChat: main, projectInstances, projectPanelTitle, doc: navDoc, commitPage,
            restorePlaces: { capture: root => ({ keyboard }) => returns.push([root, keyboard]) },
        }, opener) });
        const selected = t.mock.method(desktopRoomStorage(), 'selectRoom');
        roomStorage().setItem('ouro_chat_active_project', 'prior');
        await nav.go('chat', () => commitPage('chat'), { map: true });
        assert.equal(roomStorage().getItem('ouro_chat_active_project'), null, 'choosing Main in Main clears a prior selection');
        await nav.go('project:p7', () => commitPage('chat', { closeProject: false, enter: () => { navState.activeProjectId = 'p7'; } }), { map: true });
        project.composer.setDraft('  Kept in Project\n');
        await nav.go('settings:accounts', () => commitPage('settings', { enter: () => { state.settingsPath = 'accounts'; } }));
        assert.equal(state.activePage, 'settings');
        assert.equal(projectInstances.has('p7'), false);
        assert.equal(project.page.inert, true);
        const [[pid, inst, turn]] = rooms.exits();
        const motionEnded = () => rooms.settle(pid, inst, true, turn); // the actual shell's motion callback
        if (exit === 'deleting' || exit === 'gone') rooms.forget('p7');
        if (exit === 'settled' || exit === 'gone') {
            motionEnded();
            assert.equal(projectInstances.has('p7'), exit === 'settled');
        }
        const button = header.querySelector('[data-shell-back]');
        assert.equal(button.getAttribute('aria-label'), 'Back to Project Seven');
        for (const click of button.listeners.get('click')) click({ detail });
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(state.activePage, 'chat');
        const selectedRoom = exit === 'deleting' || exit === 'gone' ? null : 'p7';
        assert.equal(navState.activeProjectId, selectedRoom);
        assert.deepEqual(selected.mock.calls.map(call => call.arguments[0]), [null, 'p7', selectedRoom]);
        motionEnded();  // a stale exit must not hide the room it once held
        if (selectedRoom) {
            assert.deepEqual([...projectInstances.values()], [project]);
            assert.equal(project.page.hidden, false);
            assert.equal(project.page.inert, false);
            assert.equal(project.ws.paused, false);
            assert.deepEqual(returns, [[project.page, detail === 0]]);
        } else {
            assert.equal(projectInstances.size, 0);
            assert.deepEqual(returns, []);
        }
        outgoing.host.dispatchEvent(Object.assign(new Event('ouro:room-handoff'), { detail: 'normal-restart' }));
        assert.equal(drained.ouro_chat_active_project, selectedRoom || undefined);
        assert.equal(drained['ouro_chat_draft:7'], '  Kept in Project\n');
        assert.equal(drained.ouro_chat_draft, undefined);
        if (!selectedRoom) return;

        project.destroy(); main.destroy();
        const receiver = surface(message => {
            if (message.op === 'open') return { mountProject: drained.ouro_chat_active_project };
            if (message.op === 'ready') { assert.equal(message.project, 'p7'); return drained; }
            assert.equal(message.op, 'update');
            return true;
        });
        openDesktopRoomSession(receiver.host, receiver.doc, assert.fail);
        await finishDesktopStartup(() => restored, (id, mounted) => {
            restored = chat(env, ws, id, 7);  // a successor has the same room services, including state snapshots
            mounted(restored);
        }, () => {});
        assert.equal(restored.projectId, 'p7');
        assert.equal(restored.composer.getDraft(), '  Kept in Project\n');
        assert.equal(restored.composer.input.readOnly, false);
    } finally { unbind(); nav?.destroy(); layers.destroy(); restored?.destroy(); project?.destroy(); main?.destroy(); restoreDom(env.prior); }
});

for (const choice of ['back', 'missing', 'main', 'other']) test(`during migration, Settings leaves the pending Project for ${choice}`, async t => {
    const turn = () => new Promise(resolve => setImmediate(resolve));
    const health = deferred(), readingHealth = deferred(), returningRead = deferred();
    // The room's staged reply as another interface version may have written it: a field this page does not read.
    const reply = JSON.stringify({ address: 'row:7@a#1', chat_id: 7, role: 'assistant', snippet: 'Quoted', target_home: 'room' });
    const id = 'p7', reads = [], warnings = [], slots = { ouro_chat_active_project: id, 'ouro_chat_draft:7': 'kept', 'ouro_chat_reply:7': reply };
    const readsHistory = [], unexpected = [];
    const env = installDom(async url => {
        if (url === '/api/health') { readingHealth.resolve(); return health.promise; }
        if (url === `/api/projects?id=${id}`) {
            reads.push(id);
            return reads.length === 1 ? jsonReply({ reason_code: 'migrating' }, 503) : returningRead.promise;
        }
        return startupHydration(url, readsHistory, unexpected);
    });
    const win = new EventTarget(), ws = { on: () => () => {}, isConnected: () => false, send: assert.fail };
    const navDoc = Object.assign(new EventTarget(), { getElementById: () => ({ classList: { add() {} } }), querySelectorAll: () => [] });
    const layers = createLayerStack({ win, doc: navDoc });
    const state = { activePage: 'chat' }, navState = { activeProjectId: null };
    const projectInstances = new Map(), projectPanelTitle = { textContent: 'Project Seven' };
    const rooms = createRoomRetirement({ instances: projectInstances, cancelPaint: () => {} });
    let main, project, nav;
    try {
        const { host, doc } = surface(({ op }) => op === 'open' ? { mountProject: id } : op === 'ready' ? slots : true);
        openDesktopRoomSession(host, doc, text => warnings.push(text));
        main = chat(env, roomSocket(ws));
        const sandbox = vm.createContext({ state, navState, rooms, projectInstances, mainChat: main, document: navDoc,
            projectNavigationGeneration: 0, projectReveals: new Map(), desktopRoomStorage, window: win,
            CustomEvent: class extends Event { constructor(type) { super(type); } },
            syncNavigationState() {}, updateUnreadBadge() {}, enterMain() {} });
        const commitPage = vm.runInContext(`${appFunctions('commitPage', 'closeProjectPanel')}\ncommitPage;`, sandbox);
        const resolve = vm.runInContext(`${appFunctions('resolveProjectRoom')}\nresolveProjectRoom;`, sandbox);
        nav = createPageNavigation({ layers, doc: navDoc, canLeave: () => true, capture: (opener, capturePending) => capturePagePlace({
            state, navState, rooms, mainChat: main, projectInstances, projectPanelTitle, doc: navDoc, commitPage,
            pendingRoom: capturePending && desktopRoomStorage().returnToProject, restorePlaces: { capture: () => () => {} },
        }, opener) });
        const started = finishDesktopStartup(() => state.activePage === 'chat' && (project || main),
            async (key, mounted, read, unavailable) => {
                const origin = nav.capture(false), navigation = ++sandbox.projectNavigationGeneration;
                const row = await resolve({ id: key }, navigation, read, unavailable);
                if (!row) return;
                await nav.go(`project:${key}`, () => commitPage('chat', { initialRoom: true, closeProject: false, enter: () => {
                    navState.activeProjectId = row.id;
                    project = chat(env, ws, row.id, row.chat_id);
                    projectInstances.set(row.id, project);
                    mounted(project);
                } }), { origin, current: () => navigation === sandbox.projectNavigationGeneration });
            }, () => {});
        await readingHealth.promise;
        assert.equal(desktopRoomStorage().inputLocked(), true);
        await nav.go('settings:accounts', () => commitPage('settings'));
        assert.equal(layers.returnPage().back.title, 'your last Project', 'Back names the Project it resumes, never Main');
        health.resolve(jsonReply({ boot: { phase: 'ready' } }));
        await started;
        assert.equal(state.activePage, 'settings');
        assert.deepEqual(reads, [id]);
        assert.equal(roomStorage().getItem('ouro_chat_active_project'), id);

        assert.equal(layers.back('keyboard'), true);
        await turn();
        assert.equal(state.activePage, 'chat');
        assert.equal(desktopRoomStorage().inputLocked(), true);
        assert.equal(main.composer.setDraft('wrong address'), false);
        assert.deepEqual(reads, [id, id]);
        if (choice === 'main' || choice === 'other') {
            await nav.go(choice === 'main' ? 'chat' : 'project:p9', () => commitPage('chat', {
                enter: () => { if (choice === 'other') navState.activeProjectId = 'p9'; },
            }), { map: true });
        }
        returningRead.resolve(jsonReply({ projects: choice === 'missing' ? [] : [{ id, chat_id: 7, lifecycle: 'active' }] }));
        await turn();
        assert.equal(roomStorage().getItem('ouro_chat_draft:7'), 'kept');
        assert.equal(roomStorage().getItem('ouro_chat_draft'), null);
        assert.equal(roomStorage().getItem('ouro_chat_reply:7'), reply, 'the room keeps its reply whole, whichever room is shown');
        assert.equal(roomStorage().getItem('ouro_chat_reply'), null);
        assert.equal(desktopRoomStorage().inputLocked(), choice === 'missing');
        assert.equal(navState.activeProjectId, choice === 'back' ? id : choice === 'other' ? 'p9' : null);
        assert.equal(roomStorage().getItem('ouro_chat_active_project'), choice === 'main' ? null : choice === 'other' ? 'p9' : id);
        if (choice === 'back') {
            assert.equal(project.composer.getDraft(), 'kept');
            assert.match(project.composer.input.getAttribute('aria-describedby'), /-reply$/, 'its reply is staged again');
            assert.ok(!main.composer.input.getAttribute('aria-describedby'), 'and only in its own room');
            assert.equal(layers.returnPage().back.title, 'Main', 'the opened Project\'s own Back leads to Main');
            assert.equal(layers.back('keyboard'), true);  // the mounted Project does not capture its own pending lookup
            await turn();
            assert.equal(navState.activeProjectId, null);
            assert.deepEqual(reads, [id, id]);
        } else assert.equal(project, undefined);
        if (choice === 'missing') assert.match(warnings[0], /no longer available.*Choose Main or another Project/);
        else assert.deepEqual(warnings, []);
        assert.deepEqual(unexpected, []);
    } finally { nav?.destroy(); layers.destroy(); project?.destroy(); main?.destroy(); restoreDom(env.prior); }
});

test('unavailable staged Projects present a locked selection, not a Main draft or a lost window', async t => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const turn = () => new Promise(resolve => setImmediate(resolve));
    for (const [status, payload, words] of [
        [200, { projects: [] }, /no longer available/],
        [200, { projects: [{ id: 'p7', chat_id: 7, lifecycle: 'deleting' }] }, /being deleted/],
        [403, { error: 'not readable' }, /could not be read/],
    ]) {
        const history = [], unexpected = [];
        const env = installDom(async url => {
            return url === '/api/projects?id=p7' ? jsonReply(payload, status) : startupHydration(url, history, unexpected);
        }), posts = [], warnings = [];
        const { host, doc } = surface(message => {
            posts.push(message);
            if (message.op === 'open') return { mountProject: 'p7' };
            if (message.op === 'ready') return { ouro_chat_active_project: 'p7', 'ouro_chat_draft:7': 'kept in p7' };
            return true;
        });
        let main;
        try {
            openDesktopRoomSession(host, doc, text => warnings.push(text));
            main = chat(env, { on: () => () => {}, isConnected: () => true, send: assert.fail });
            let connected = 0;
            const resolve = exactProjectReader();
            await finishDesktopStartup(() => main, (id, _mount, read, unavailable) =>
                resolve({ id }, 1, read, unavailable), () => { connected++; });
            assert.equal(connected, 1);
            assert.match(warnings[0], words);
            assert.match(warnings[0], /Choose Main or another Project/);
            assert.deepEqual(readyFrames(posts), [{ op: 'ready', project: 'p7', selection: true, ticket: 'one-navigation' }]);
            assert.equal(main.composer.input.readOnly, true);
            assert.equal(main.composer.insertAtCursor('blocked'), false);
            assert.equal(roomStorage().getItem('ouro_chat_active_project'), 'p7');
            desktopRoomStorage().selectRoom('');
            assert.equal(main.composer.input.readOnly, false);
            assert.equal(main.composer.getDraft(), '');
            assert.equal(roomStorage().getItem('ouro_chat_draft:7'), 'kept in p7');
            await turn();
            t.mock.timers.tick(50);  // Main's normal initial history gate
            await turn();
            assert.deepEqual(history, ['/api/chat/history']);
            assert.deepEqual(unexpected, []);
        } finally { main?.destroy(); restoreDom(env.prior); }
    }
});

test('handoff snapshots the last edit, removal and original Send in their room before its update reply', async () => {
    const posts = [], warnings = [], update = deferred();
    const { host, doc } = surface(message => {
        posts.push(message);
        if (message.op === 'open') return { ouro_chat_draft: 'Main alone', 'ouro_chat_draft:7': 'before',
            'ouro_chat_reply:7': 'old quote', 'ouro_chat_files:7': '2' };
        return message.op === 'update' ? update.promise : true;
    });
    await openDesktopRoomSession(host, doc, text => warnings.push(text));
    const storage = roomStorage();
    const frame = { type: 'chat', content: 'sent once', client_message_id: 'original', sender_session_id: 'sender',
        chat_id: 7, project_id: 'p7', reply_to: 'row:7@a#1' };
    const pending = JSON.stringify([{ frame }]);
    storage.setItem('ouro_chat_unconfirmed:7', pending);
    storage.setItem('ouro_chat_draft:7', '  next words\n');
    storage.removeItem('ouro_chat_reply:7');
    storage.removeItem('ouro_chat_files:7');
    storage.editor('Files', true);
    host.dispatchEvent(Object.assign(new Event('ouro:room-handoff'), { detail: 'this-replacement' }));
    assert.deepEqual(posts.map(message => message.op), ['open', 'editor', 'update', 'drained']);
    const drained = posts.at(-1);
    assert.equal(drained.ticket, 'one-navigation');
    assert.equal(drained.handoff, 'this-replacement');
    assert.deepEqual(drained.editors, ['Files']);
    assert.deepEqual(JSON.parse(drained.items), { ouro_chat_draft: 'Main alone', 'ouro_chat_draft:7': '  next words\n',
        'ouro_chat_unconfirmed:7': pending });
    update.resolve(true);
    assert.deepEqual(warnings, []);
    host.dispatchEvent(new Event('ouro:room-input-unconfirmed'));
    assert.match(warnings.at(-1), /last input.*draft, reply and room history/i);
});

test('the handoff includes a pressed Send that finishes its upload after the new field mounts', async () => {
    const messages = [], upload = deferred();
    const { host, doc } = surface(message => {
        messages.push(message);
        return message.op === 'open' ? { ouro_chat_draft: 'ready to send' } : true;
    });
    await openDesktopRoomSession(host, doc, assert.fail);
    const storage = roomStorage();
    storage.trackSend(upload.promise.then(() => {
        storage.removeItem('ouro_chat_draft');
        storage.setItem('ouro_chat_unconfirmed', JSON.stringify([{ frame: { client_message_id: 'once' } }]));
    }));
    host.dispatchEvent(Object.assign(new Event('ouro:room-handoff'), { detail: 'active-transition' }));
    assert.deepEqual(messages.map(message => message.op), ['open']);
    upload.resolve();
    await new Promise(resolve => setImmediate(resolve));
    const snapshot = JSON.parse(messages.at(-1).items);
    assert.deepEqual(JSON.parse(snapshot.ouro_chat_unconfirmed), [{ frame: { client_message_id: 'once' } }]);
    assert.equal(snapshot.ouro_chat_draft, undefined);
});

test('a planned replacement holds ordinary text by ID and drops queued commands, including typed /restart', async t => {
    const { host, doc } = surface(message => message.op === 'resume_queue' ? false : {});
    await openDesktopRoomSession(host, doc, assert.fail);
    const env = installDom(), sent = [], dropped = [];
    const ws = new WS('unused');
    try {
        if (!globalThis.WebSocket) {
            globalThis.WebSocket = { OPEN: 1 };
            t.after(() => { delete globalThis.WebSocket; });
        }
        ws.ws = { readyState: WebSocket.OPEN, send: wire => sent.push(JSON.parse(wire)) };
        const chat = { type: 'chat', content: '/tmp/work folder', sender_session_id: 'sender', client_message_id: 'original' };
        const path = { ...chat, content: '/review/notes', client_message_id: 'second', chat_id: 7, reply_to: 'row:7@a#1' };
        const control = { type: 'command', cmd: '/bg status', ui_control: { name: 'consciousness_toggle' } };
        const typed = { ...chat, content: ' /RESTART ', client_message_id: 'command' };
        ws._pendingMessages.push(chat, control, path, typed);
        ws.on('outbound_dropped', frame => dropped.push([frame.clientMessageId, frame.held]));
        await ws._flushPendingMessages();
        assert.deepEqual(sent, []);
        assert.deepEqual(dropped, [['original', true], ['', false], ['second', true], ['command', false]]);
        assert.deepEqual(ws._pendingMessages, []);
    } finally { restoreDom(env.prior); }
});

test('desktop reconnect drains ordinary actions FIFO while new actions join the pending native decision', async t => {
    const first = deferred(), next = deferred(), permits = [first, next];
    const { host, doc } = surface(message => message.op === 'resume_queue' ? permits.shift().promise : {});
    await openDesktopRoomSession(host, doc, assert.fail);
    const env = installDom(), sent = [];
    const ws = new WS('unused');
    try {
        if (!globalThis.WebSocket) {
            globalThis.WebSocket = { OPEN: 1 };
            t.after(() => { delete globalThis.WebSocket; });
        }
        ws.connect = ws._scheduleReconnect = () => {};
        const chat = { type: 'chat', content: 'first', client_message_id: 'one' };
        const control = { type: 'command', cmd: '/status' };
        const later = { ...chat, content: 'next send', client_message_id: 'two' };
        const typed = { ...chat, content: '/review', client_message_id: 'three' };
        assert.equal(ws.send(chat).status, 'queued');
        assert.equal(ws.send(control).status, 'queued');
        assert.equal(ws.send(control, { queue: false }).status, 'failed');
        ws.ws = { readyState: WebSocket.OPEN, send: wire => sent.push(JSON.parse(wire)) };
        const drained = ws._flushPendingMessages();
        assert.equal(ws.send(later).status, 'queued');
        assert.equal(ws.send(typed).status, 'queued');
        assert.deepEqual(sent, []);
        const immediate = { type: 'command', cmd: '/status explicit' };
        assert.equal(ws.send(immediate, { queue: false }).status, 'sent');
        const stops = [];
        const previousBridge = globalThis.pywebview;
        globalThis.pywebview = { api: { panic_stop(ticket) { stops.push(ticket); return { status: 'stopping' }; } } };
        t.after(() => { if (previousBridge) globalThis.pywebview = previousBridge; else delete globalThis.pywebview; });
        assert.equal(ws.send({ type: 'chat', content: '/panic' }).status, 'intercepted');
        assert.deepEqual(stops, ['one-navigation']);
        assert.equal(permits.length, 1);
        first.resolve(true);
        await new Promise(resolve => setImmediate(resolve));
        assert.deepEqual(sent, [immediate, chat, control]);
        assert.equal(permits.length, 0);
        next.resolve(true);
        await drained;
        assert.deepEqual(sent, [immediate, chat, control, later, typed]);

        const logged = t.mock.method(console, 'error', () => {});
        ws.on('outbound_sent', () => { throw new Error('retired listener'); });
        const written = { ...chat, client_message_id: 'written-once' };
        assert.deepEqual(ws.send(written), { status: 'sent', clientMessageId: 'written-once' });
        assert.equal(logged.mock.callCount(), 1);
        assert.deepEqual(ws._pendingMessages, []);
        assert.deepEqual(sent, [immediate, chat, control, later, typed, written]);
    } finally { restoreDom(env.prior); }
});

test('a receipt arriving during the native open settles only its original held id', async () => {
    let deliver;
    const pending = new Promise(resolve => { deliver = resolve; });
    const { host, doc } = surface(message => message.op === 'open' ? pending : true);
    const opened = openDesktopRoomSession(host, doc, assert.fail);
    const kept = createUnconfirmedSends({ send: assert.fail, root: () => null, onDomWrite: fn => fn(),
        storage: roomStorage(), storageKey: 'ouro_chat_unconfirmed', showToast: assert.fail });
    const frame = { type: 'chat', sender_session_id: 'original-tab', content: 'same words' };
    kept.settle({ role: 'user', client_message_id: 'accepted', ingress_accepted: true, ingress_dispatched: true });
    deliver({ ouro_chat_unconfirmed: JSON.stringify(['accepted', 'unknown'].map(id =>
        ({ frame: { ...frame, client_message_id: id } }))) });
    await opened;
    const ids = [];
    kept.reconcile(entry => ids.push(entry.frame.client_message_id));
    assert.deepEqual(ids, ['unknown']);
    kept.release();
});

test('a socket change invalidates the old flush and a failed write keeps the unwritten tail', async t => {
    const replies = [];
    const { host, doc } = surface(message => message.op === 'open' ? {} : new Promise(resolve => replies.push(resolve)));
    await openDesktopRoomSession(host, doc, assert.fail);
    if (!globalThis.WebSocket) {
        globalThis.WebSocket = { OPEN: 1 };
        t.after(() => { delete globalThis.WebSocket; });
    }
    const ws = new WS('unused'), writes = [], dropped = [];
    const chat = { type: 'chat', sender_session_id: 's', client_message_id: 'held', content: 'words' };
    const controls = ['first', 'second'].map(content => ({ type: 'command', content }));
    ws.on('outbound_dropped', frame => dropped.push(frame.clientMessageId));
    ws._pendingMessages.push(chat, ...controls);
    ws._scheduleReconnect = () => {};
    ws.ws = { readyState: WebSocket.OPEN, send() { throw new Error('socket closed'); } };
    const stale = ws._flushPendingMessages();
    assert.deepEqual(ws._pendingMessages, [chat, ...controls]);
    const accept = wire => writes.push(JSON.parse(wire));
    ws.ws = { readyState: WebSocket.OPEN, send(wire) {
        if (writes.length) throw new Error('socket closed');
        accept(wire);
    } };
    const current = ws._flushPendingMessages();
    replies.shift()(false);
    await stale;
    assert.deepEqual(dropped, []);
    assert.deepEqual(ws._pendingMessages, [chat, ...controls]);
    replies.shift()(true);
    await current;
    assert.deepEqual(writes, [chat]);
    assert.deepEqual(ws._pendingMessages, controls);
    ws.ws.send = accept;
    const resumed = ws._flushPendingMessages();
    replies.shift()(true);
    await resumed;
    assert.deepEqual(writes, [chat, ...controls]);
});
