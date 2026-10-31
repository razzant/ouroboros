// Source proposal by readonly child 87500533; materialized and executed by parent.
import assert from 'node:assert/strict';
import test from 'node:test';
import { installCardDom, TextNode } from './helpers/card_dom.js';
import { applyPayload, createTranslator, setMissTransport } from '../modules/i18n.js';
import { normalizePlace, readPlaceRooms } from '../modules/consciousness_api.js';
import { wakesForText } from '../modules/consciousness_words.js';
import { mountPlaceHistory, capturedWindowNote, historySpeaker, historyFactWords, placeRoomsHtml, shownTime } from '../modules/place_history.js';
import { openPlaceSheet } from '../modules/place_sheet.js';
import { mountConnectorPlaces } from '../modules/connector_places.js';
const tick = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
const unsafeText = '<img src="x" onerror="bad()"> & "hello"';
const rawPlace = (id = 'one') => ({ place_id: id, connector: 'telegram-user-session', label: unsafeText, account_id: `account-${id}`, state: 'active' });
const place = normalizePlace(rawPlace());
const room = { room_chat_id: 5_000_000_123, address: unsafeText, unanswered_count: null, label: null, last_at: '', conversation_id: '', thread_id: '' };
const coverage = { discovery: 'arrivals', importing: false, gaps: false };
const index = (rooms = [room]) => ({ place_id: place.id, rooms, coverage });
const row = (id, text = unsafeText) => ({ history_id: id, role: 'user', text, sender_label: unsafeText, place_history: {kind: 'message'} });
const page = (messages, next = null) => ({ messages, next_cursor: next, has_more: next !== null, window: { complete: next === null, truncated_by: next ? ['quota'] : [] } });
function fixture(t) {
    const dom = installCardDom(); const { doc, win } = dom;
    // Missing fixture DOM mechanics, not a replacement for production behavior.
    doc.defaultView = win; win.getComputedStyle = () => ({ visibility: 'visible' });
    doc.getElementById = (id) => doc.body.querySelector(`[id="${id}"]`);
    doc.addEventListener = win.addEventListener; doc.removeEventListener = win.removeEventListener;
    const create = doc.createElement;
    doc.createElement = (tag) => {
        const node = create(tag);
        Object.defineProperties(node, { isConnected: { get: () => doc.body.contains(node) }, childElementCount: { get: () => node.children.length } });
        node.getClientRects = () => node.isConnected ? [{}] : [];
        node.scrollTop = 0; node.scrollHeight = 0;
        node.prepend = (...nodes) => { for (const child of [...nodes].reverse()) node.insertBefore(child, node.firstChild); };
        const matches = node.matches.bind(node);
        node.matches = (selector) => matches(selector.replace(/:disabled/g, '[disabled]'));
        return node;
    };
    doc.createDocumentFragment = () => doc.createElement('#fragment');
    doc.createTextNode = (text) => new TextNode(text, doc);
    doc.createTreeWalker = (root) => {
        const texts = []; const visit = (node) => { if (node.nodeType === 3) {
            Object.defineProperty(node, 'parentElement', { configurable: true, get: () => node.parentNode }); texts.push(node);
        } else (node.childNodes || []).forEach(visit); }; visit(root);
        let i = 0; return { nextNode: () => texts[i++] || null };
    };
    const requests = []; let answer = () => { throw new Error('Unexpected request: no network allowed'); };
    t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
        const request = { url: String(url), ...init }; requests.push(request);
        assert.equal(init.method || 'GET', 'GET', 'navigation must not write');
        const value = await answer(request);
        return { ok: true, status: 200, json: async () => value };
    });
    t.after(() => { for (const button of doc.body.querySelectorAll('[data-history-close], [data-place-close]')) button.click(); dom.restore(); });
    return { doc, requests, reply(fn) { answer = fn; }, q(selector) { return doc.body.querySelector(selector); } };
}
function mount(f, options = {}, currentPlace = place) {
    const host = f.doc.createElement('div'); f.doc.body.append(host);
    const reader = mountPlaceHistory(host, currentPlace, room, { onBack: () => reader.close(), ...options });
    return reader;
}

test('role-first names and facts never infer delivery or exact wake', () => {
    assert.deepEqual(['user', 'assistant', 'system'].map((role) => historySpeaker({role, sender_label: 'Alice'})), ['Alice', 'Ouroboros', 'System']);
    assert.deepEqual(historyFactWords({role: 'assistant'}), []);
    assert.deepEqual(historyFactWords({place_history: {delivery_state: 'authored'}}), ['Delivery not confirmed']);
    assert.deepEqual(historyFactWords({place_history: {attention: {decision:'focus_started'}}}), ['Focus started']);
});

test('a retained row without new fields renders byte for byte as on e3bd98c20', async (t) => {
    const f = fixture(t);
    const legacy = { history_id: 'chat:legacy', role: 'user', text: 'Already saved <plain> & unchanged.', sender_label: 'Maya',
        place_history: { kind: 'message', owner_confirmed: true, attention: { decision: 'heard', reason: 'paused' } } };
    const reader = mount(f, { readHistory: async () => page([legacy]) }); await tick();
    assert.equal(f.q('[data-history-messages]').innerHTML,
        `<article class="place-convo-row is-owner"><div class="place-convo-meta"><span class="place-convo-who" data-i18n-authored>Maya</span><span>(you)</span>
        <time data-i18n-authored datetime title></time>
        <span class="place-convo-fact">Heard while Paused</span></div><div class="chat-markdown place-convo-text place-history-plain" data-i18n-authored>Already saved &lt;plain&gt; &amp; unchanged.</div></article>`);
    reader.close();
});

test('delivery rows stamp the recorded account, as you for its owner, and nothing for old rows', async (t) => {
    const f = fixture(t);
    const sender = { id: 'account-id', label: unsafeText, kind: 'user' };
    const rows = [false, true, null].map((owner, i) => ({ ...row(`out-${i}`, 'My whole reply'), role: 'assistant',
        place_history: { kind: 'delivery', delivery_state: 'delivered', ...(owner === null ? {} : { sends_as: sender, sends_as_owner: owner }) } }));
    const reader = mount(f, { readHistory: async () => page(rows) }); await tick();
    const metas = [...f.doc.body.querySelectorAll('.place-convo-meta')];
    assert.ok(metas[0].textContent.includes(`as ${unsafeText}`));
    assert.equal(metas[0].querySelector('span[data-i18n-authored]').textContent, unsafeText);
    assert.match(metas[1].textContent, /as you/);
    assert.ok(!metas[1].textContent.includes(unsafeText));
    assert.doesNotMatch(metas[2].textContent, /as you|as /);
    assert.ok(metas.every((meta) => meta.textContent.includes('Ouroboros') && meta.textContent.includes('Delivered')));
    assert.equal(f.q('img'), null);
    reader.close();
});

test('heard facts sit beside the reason with core words and authored signal labels', async (t) => {
    const f = fixture(t);
    const heard = { ...row('heard'), place_history: { kind: 'message', attention: { decision: 'heard', reason: 'no_wake_match' },
        facts: ['direct_message', 'from_bot', 'signal.words'] } };
    const reader = mount(f, { readHistory: async () => page([heard,
        { ...heard, history_id: 'old', place_history: { ...heard.place_history, facts: undefined } },
        { ...heard, history_id: 'empty', place_history: { ...heard.place_history, facts: [] } },
        { ...heard, history_id: 'focus', place_history: { ...heard.place_history, attention: { decision: 'focus_started' } } },
    ]) }, { ...place, facts: [{ id: 'signal.words', label: unsafeText }], labels: { direct_message: 'Mail addressed to you' } });
    await tick();
    const metas = [...f.doc.body.querySelectorAll('.place-convo-meta')];
    assert.deepEqual([...metas[0].querySelectorAll('.place-convo-fact')].map((node) => node.textContent),
        ['Heard; no wake matched', 'Mail addressed to you', 'from a bot', unsafeText]);
    assert.equal(metas[0].querySelectorAll('.place-convo-fact[data-i18n-authored]').length, 2);
    assert.ok(!metas[0].textContent.includes('direct messages'), 'declared core labels also name the reader fact');
    assert.deepEqual(metas.slice(1).map((meta) => meta.querySelectorAll('.place-convo-fact').length), [1, 1, 1]);
    assert.ok(metas.slice(1).every((meta) => meta.querySelectorAll('[data-heard-fact]').length === 0));
    assert.match(metas[3].textContent, /Focus started/);
    assert.equal(f.q('img'), null);
    reader.close();
});

for (const labels of [null, { any_message: 'Every saved arrival', direct_message: 'Mail addressed to you' }]) {
    test(`heard messages show only specific facts (${labels ? 'declared labels' : 'core words'})`, async (t) => {
        const f = fixture(t);
        const rows = [['any_message', 'from_bot', 'direct_message'], [], ['any_message']].map((facts, i) => ({
            ...row(`heard-${i}`), place_history: { kind: 'message', attention: { decision: 'heard', reason: 'no_wake_match' }, facts },
        }));
        const reader = mount(f, { readHistory: async () => page(rows) }, { ...place, labels }); await tick();
        const metas = [...f.doc.body.querySelectorAll('.place-convo-meta')];
        assert.deepEqual(metas.map((meta) => [...meta.querySelectorAll('[data-heard-fact]')].map((node) => node.textContent)),
            [['from a bot', labels?.direct_message || 'addressed to it'], [], []]);
        assert.deepEqual([...metas[0].querySelectorAll('.place-convo-fact')].map((node) => node.textContent),
            ['Heard; no wake matched', 'from a bot', labels?.direct_message || 'addressed to it']);
        assert.equal(f.q('[data-heard-fact="direct_message"]').hasAttribute('data-i18n-authored'), Boolean(labels));
        reader.close();
    });
}

test('heard core facts describe one message while wake sentences keep their policy words', async (t) => {
    const f = fixture(t);
    const facts = ['direct_message', 'group_dm', 'mention', 'reply_to_me', 'my_thread', 'new_issue',
        'new_pull_request', 'review_requested', 'assigned', 'any_message', 'from_bot', 'legacy_addressed'];
    const heard = { ...row('heard'), place_history: { kind: 'message', attention: { decision: 'heard' }, facts } };
    const reader = mount(f, { readHistory: async () => page([heard]) }); await tick();
    assert.deepEqual([...f.doc.body.querySelectorAll('[data-heard-fact]')].map((node) => node.textContent),
        ['addressed to it', 'group message', 'mentions it', 'reply to it', 'in its thread', 'new issue',
            'new pull request', 'review requested', 'assigned to it', 'from a bot', 'addressed to it']);
    assert.equal(wakesForText(facts.map((fact) => ({ fact }))),
        'wakes for direct messages, group messages, mentions, replies, replies in its threads, new issues, new pull requests, review requests addressed to me, assignments addressed to me, every message and messages addressed to it · also wakes on bot messages');
    reader.close();
});

for (const late of [false, true]) test(`the sheet reader merges status labels (${late ? 'after' : 'before'} opening) and keeps authored nodes`, async (t) => {
    const f = fixture(t), status = deferred();
    const current = normalizePlace({ ...rawPlace(), source: { url: '/api/extensions/telegram-user-session/status' },
        facts: ['direct_message', 'from_bot', { id: 'signal.words', label: 'PR words' }], labels: { direct_message: 'Declared mail' } });
    const heard = { ...row('heard'), place_history: { kind: 'message', attention: { decision: 'heard' },
        facts: ['any_message', 'direct_message', 'from_bot', 'signal.words'] } };
    f.reply(({ url }) => url === current.source.url ? status.promise : page([heard]));
    const sheet = openPlaceSheet(current, { readRooms: async () => index() });
    t.after(() => sheet.close());
    const payload = { text: 'Ready', tone: 'ok', labels: { any_message: 'Every arrival', direct_message: unsafeText } };
    if (!late) status.resolve(payload);
    await tick();
    f.q(`[data-place-room="${room.room_chat_id}"]`).click(); await tick();
    const words = () => [...f.doc.body.querySelectorAll('[data-heard-fact]')].map((node) => node.textContent);
    assert.deepEqual(words(), [late ? 'Declared mail' : unsafeText, 'from a bot', 'PR words']);
    const message = f.q('.place-convo-text'), article = f.q('.place-convo-row');
    if (late) { status.resolve(payload); await tick(); }
    assert.deepEqual(words(), [unsafeText, 'from a bot', 'PR words']);
    assert.equal(f.q('.place-convo-text'), message);
    assert.equal(f.q('.place-convo-row'), article);
    assert.ok(f.q('[data-heard-fact="direct_message"]').hasAttribute('data-i18n-authored'));
    assert.equal(f.q('img'), null);
    assert.deepEqual(sheet.place.labels, { direct_message: 'Declared mail' });
    assert.equal(f.requests.filter(({ url }) => url === current.source.url).length, 1);
});

test('reader labels return to core words when absent without replacing the retained message', async (t) => {
    const f = fixture(t);
    const heard = { ...row('heard'), place_history: { kind: 'message', attention: { decision: 'heard' }, facts: ['direct_message'] } };
    const reader = mount(f, { readHistory: async () => page([heard]) }, { ...place, labels: { direct_message: 'Mail addressed to you' } });
    await tick();
    const fact = f.q('[data-heard-fact]'), message = f.q('.place-convo-text');
    assert.equal(fact.textContent, 'Mail addressed to you');
    assert.ok(fact.hasAttribute('data-i18n-authored'));
    reader.updatePlace(place);
    assert.equal(fact.textContent, 'addressed to it');
    assert.equal(fact.hasAttribute('data-i18n-authored'), false);
    assert.equal(f.q('.place-convo-text'), message);
    reader.close();
});

test('a held unchecked reply keeps its whole authored text below the System line, never a checked claim', async (t) => {
    const f = fixture(t);
    const heldText = `My reply\n\n${unsafeText}\n${'More of my words. '.repeat(300)}`;
    const held = { ...row('held', 'Reply held: mailbox is Silent'), role: 'system',
        place_history: { kind: 'notice', held_text: heldText, held: true, checked: false } };
    const rows = [held, ...[{ held: false }, { checked: true }, { checked: undefined }, { held_text: undefined }]
        .map((fields, i) => ({ ...held, history_id: `other-${i}`, place_history: { ...held.place_history, ...fields } }))];
    const reader = mount(f, { readHistory: async () => page(rows) }); await tick();
    const articles = [...f.doc.body.querySelectorAll('.place-convo-row')];
    assert.equal(articles[0].querySelector('.place-convo-who').textContent, 'System');
    assert.deepEqual(articles[0].children.slice(1).map((node) => node.textContent), ['Reply held: mailbox is Silent', 'held, unchecked', heldText.replaceAll('\n', '')]);
    assert.equal(articles[0].children.at(-1).querySelectorAll('br').length, 3, 'Markdown preserves the line breaks');
    assert.ok(articles[0].children.at(-1).getAttribute('data-i18n-authored') !== null);
    assert.ok(articles.slice(1).every((article) => !article.textContent.includes('held, unchecked') && !article.textContent.includes(heldText)));
    assert.equal(f.q('img'), null);
    reader.close();
});

test('new reader words translate while the sending account, signal and held words remain authored', async (t) => {
    const f = fixture(t);
    setMissTransport(() => Promise.resolve(null));
    applyPayload({ language: 'fr', english: false, entries: {
        'as {label}': { text: 'comme {label}' }, 'code:places.identity.as_you': { text: 'comme vous' },
        'code:places.history.held_unchecked': { text: 'retenu, non vérifié' },
        'code:places.wake.direct_message': { text: 'messages directs' },
        'code:places.heard.direct_message': { text: 'adressé à vous' },
        'code:places.heard.from_bot': { text: 'd’un bot' }, Settings: { text: 'Réglages' },
    } });
    t.after(() => applyPayload({ language: '', english: true, entries: {} }));
    const reader = mount(f, { readHistory: async () => page([
        { ...row('out', 'Settings'), role: 'assistant', place_history: { kind: 'delivery', sends_as: { id: 'id', label: 'Settings', kind: 'user' }, sends_as_owner: false } },
        { ...row('heard', 'Settings'), place_history: { kind: 'message', attention: { decision: 'heard' }, facts: ['any_message', 'direct_message', 'from_bot', 'signal.words'] } },
        { ...row('held', 'System line'), role: 'system', place_history: { kind: 'notice', held_text: 'Settings', held: true, checked: false } },
    ]) }, { ...place, facts: [{ id: 'signal.words', label: 'Settings' }] });
    await tick();
    createTranslator({ skipRoots: '[data-i18n-authored]', userContent: '', excludeSelector: '' }).applyTo(f.doc.body);
    const rows = [...f.doc.body.querySelectorAll('.place-convo-row')];
    assert.match(rows[0].textContent, /comme Settings/);
    assert.deepEqual([...rows[1].querySelectorAll('[data-heard-fact]')].map((node) => node.textContent), ['adressé à vous', 'd’un bot', 'Settings']);
    assert.equal(wakesForText([{ fact: 'direct_message' }]), 'wakes for messages directs');
    assert.match(rows[2].textContent, /retenu, non vérifiéSettings/);
    assert.ok(rows.every((node) => !node.textContent.includes('Réglages')));
    reader.close();
});

test('page, pending import, range gaps and unknown coverage have distinct words', () => {
    assert.doesNotMatch(capturedWindowNote({has_more: false, window:{complete:false,truncated_by:['page']}}), /unavailable|incomplete/);
    assert.match(capturedWindowNote({has_more:true,window:{complete:false,truncated_by:['page']}}), /Earlier saved/);
    assert.match(capturedWindowNote({window:{complete:false,truncated_by:['chat_importing']}}), /still being imported/);
    assert.match(capturedWindowNote({window:{complete:false,truncated_by:['page']},coverage:{spans:{chat:{gaps:['malformed_jsonl']}}}}), /unavailable/);
    assert.match(capturedWindowNote({}), /not fully reported/);
    assert.match(placeRoomsHtml({rooms:[],coverage}), /No saved conversation found in this index/);
    assert.match(placeRoomsHtml({rooms:[],coverage:{...coverage,importing:true}}), /still being imported/);
});

test('reader forwards large IDs, retries exact cursor and excludes progress/evidence', async (t) => {
    const f=fixture(t); const cursor='opaque /+?&cursor'; let attempt=0;
    f.reply(({url}) => {
        const query=new URL(url,'http://local').searchParams;
        assert.equal(query.get('chat_id'),String(room.room_chat_id));
        assert.equal(query.get('cursor'),attempt===0 ? null : cursor); attempt+=1;
        if (attempt===1) return page([row('new'),{...row('thought','hidden thought'),is_progress:true}, {history_id:'evidence',text:'hidden evidence',role:'system'}],cursor);
        if (attempt===2) throw new Error(unsafeText);
        if (attempt===3) return page([],cursor);
        return page([row('old','older'),row('new')]);
    });
    const reader=mount(f); await tick();
    assert.equal(f.q('img'),null);
    assert.equal(f.q('[data-history-messages]').childElementCount,1);
    f.q('[data-history-older]').click(); await tick();
    assert.equal(f.q('[data-history-status]').textContent,unsafeText);
    f.q('[data-history-retry]').click(); await tick();
    assert.match(f.q('[data-history-status]').textContent,/no page was skipped/);
    f.q('[data-history-retry]').click(); await tick();
    assert.equal(f.q('[data-history-messages]').childElementCount,2);
    assert.equal(f.q('[data-history-messages]').firstElementChild.querySelector('.chat-markdown').textContent,'older');
    assert.equal(f.q('[data-history-older]').hidden,true); assert.equal(attempt,4); reader.close();
});

test('partial ranges retry their own page token and keep gaps on other loaded ranges', async (t) => {
    const f=fixture(t); const requests=[]; let repaired=false;
    const reader=mount(f,{readHistory:async ({cursor})=>{
        requests.push(cursor);
        if (cursor==='same-page') { repaired=true; return {...page([row('new')],'older'),page_cursor:'same-page'}; }
        if (cursor==='older') return {...page([row('old')]),window:{complete:false,truncated_by:['page']}};
        return {...page([row('new')],'older'),page_cursor:'same-page',window:{complete:false,truncated_by:['chat_malformed_jsonl']}};
    }}); await tick();
    f.q('[data-history-older]').click(); await tick();
    assert.match(f.q('[data-history-coverage]').textContent,/unavailable/);
    f.q('[data-history-range]').click(); await tick();
    assert.equal(repaired,true); assert.deepEqual(requests,[null,'older','same-page']);
    assert.doesNotMatch(f.q('[data-history-coverage]').textContent,/unavailable/); reader.close();
});

test('room API validates identity, safe integers, unknown count and coverage', async (t) => {
    const f=fixture(t); const controller=new AbortController(); f.reply(()=>index());
    assert.deepEqual(await readPlaceRooms(place.id,{signal:controller.signal}),{rooms:[room],coverage});
    assert.equal(f.requests[0].signal,controller.signal); assert.equal(f.requests[0].cache,'no-store');
    for(const invalid of [{...index(),place_id:'different'},{...index(),rooms:null},{...index(),coverage:null},index([{...room,room_chat_id:Number.MAX_SAFE_INTEGER+1}]),index([{...room,room_chat_id:0}]),index([{...room,room_chat_id:String(room.room_chat_id)}]),index([{...room,unanswered_count:-1}])]) {
        f.reply(()=>invalid); await assert.rejects(readPlaceRooms(place.id),/could not be read/);
    }
});

test('Back disposal aborts pending read and ignores its late answer', async (t) => {
    const f=fixture(t); const pending=deferred(); let signal;
    const reader=mount(f,{readHistory: (request)=>{signal=request.signal;return pending.promise}});
    f.q('[data-history-back]').click(); reader.close(); assert.equal(signal.aborted,true);
    pending.resolve(page([row('late')])); await tick(); assert.equal(f.q('.place-convo'),null);
});

test('connector section lists zero/one/many without selecting or writing, and aborts on disposal', async (t) => {
    const f=fixture(t);
    for(const count of [0,1,2]) {
        const host=f.doc.createElement('section'); f.doc.body.append(host);
        const supplied=Array.from({length:count},(_,i)=>normalizePlace(rawPlace(String(i))));
        let signal;
        const dispose=mountConnectorPlaces(host,'telegram-user-session',{read:async (request)=>{signal=request.signal;return {places:[...supplied,normalizePlace({...rawPlace('foreign'),connector:'elsewhere'})]}}});
        await tick(); assert.equal(host.querySelectorAll('[data-place-open]').length,count);
        assert.equal(Boolean(host.querySelector('[data-place-connect]')),count===0);
        assert.equal(f.q('[data-place-state]'),null); dispose(); assert.equal(signal.aborted,true); host.remove();
    }
});

test('an empty connector section uses the shared Connect control, filtered to its own connector', async (t) => {
    const f=fixture(t); const host=f.doc.createElement('section'); f.doc.body.append(host); let left=0;
    const dispose=mountConnectorPlaces(host,'telegram-user-session',{leave:()=>{left+=1;},read:async ()=>({places:[]})});
    await tick();
    f.reply(()=>({places:[],connectors:[{connector:'telegram-user-session'},{connector:'elsewhere'}]}));
    host.querySelector('[data-place-connect]').click(); await tick();
    assert.equal(left,1,'the connector sheet leaves before Connect opens');
    assert.deepEqual([...f.doc.body.querySelectorAll('[data-connect-act]')].map((button)=>button.dataset.connector),['telegram-user-session']);
    assert.deepEqual(f.requests.map((request)=>request.url),['/api/places']);
    dispose();
});

test('translation preserves authored messages while chrome translates', async (t) => {
    const f=fixture(t); f.reply(()=>page([row('authored','Settings')])); const reader=mount(f); await tick();
    const chrome=f.doc.createElement('button'); chrome.textContent='Settings'; f.doc.body.append(chrome);
    setMissTransport(()=>Promise.resolve({ok:true}));
    applyPayload({language:'fr',english:false,revision:1,entries:{Settings:{text:'Translated chrome'}}});
    t.after(()=>{applyPayload({language:'',english:true,entries:{}});setMissTransport(null)});
    createTranslator({skipRoots:'[data-i18n-authored]',userContent:'',excludeSelector:''}).applyTo(f.doc.body);
    assert.equal(f.q('.chat-markdown').textContent,'Settings'); assert.equal(chrome.textContent,'Translated chrome'); reader.close();
});

const viewChanged = () => Object.assign(new Error('history_view_changed'), { status: 409, body: { reason_code: 'history_view_changed' } });
const shownTexts = (f) => [...f.q('[data-history-messages]').children].map((node) => node.querySelector('.chat-markdown').textContent);

test('a changed view rereads the newest page and replaces the shown ranges only once it arrives', async (t) => {
    const f = fixture(t); const requests = []; const fresh = deferred(); let busyOnce = true;
    const reader = mount(f, { readHistory: async ({ cursor }) => {
        requests.push(cursor);
        if (requests.length === 1) return page([row('live', 'newest live')], 'stored-order');
        if (cursor === 'stored-order') throw viewChanged();  // the room's first import changed its view
        if (busyOnce) { busyOnce = false; throw new Error('Saved history is busy.'); }
        return fresh.promise;
    } });
    await tick();
    f.q('[data-history-older]').click(); await tick(); await tick();
    assert.deepEqual(requests, [null, 'stored-order', null], 'the refused cursor is never resent');
    assert.equal(f.q('[data-history-status]').textContent, 'Saved history is busy.');
    assert.deepEqual(shownTexts(f), ['newest live'], 'a failed fresh read keeps the shown ranges');
    f.q('[data-history-retry]').focus(); f.q('[data-history-retry]').click(); await tick();
    assert.deepEqual(requests, [null, 'stored-order', null, null], 'Try again rereads the newest page');
    assert.deepEqual(shownTexts(f), ['newest live'], 'the shown ranges stay while the fresh read runs');
    fresh.resolve(page([row('imported', 'earlier imported'), row('live', 'newest live')], 'time-order')); await tick();
    assert.deepEqual(shownTexts(f), ['earlier imported', 'newest live']);
    assert.equal(f.q('[data-history-status]').textContent, 'Saved history changed, so it was read again from the newest messages.');
    assert.equal(f.q('[data-history-retry]').hidden, true);
    assert.equal(f.doc.activeElement, f.q('[data-history-older]'), 'focus stays in the reader');
    f.q('[data-history-older]').click(); await tick();
    assert.equal(requests.at(-1), 'time-order'); reader.close();
});

test('a changed view met by Retry this range replaces the chain too; closing ignores the late fresh page', async (t) => {
    const f = fixture(t); const requests = []; const fresh = deferred(); let signal;
    const reader = mount(f, { readHistory: async ({ cursor, signal: aborted }) => {
        requests.push(cursor);
        if (requests.length === 1) return { ...page([row('live', 'newest live')], 'older'), page_cursor: 'same-page', window: { complete: false, truncated_by: ['chat_malformed_jsonl'] } };
        if (cursor === 'same-page') throw viewChanged();
        signal = aborted; return fresh.promise;
    } });
    await tick();
    f.q('[data-history-range]').click(); await tick(); await tick();
    assert.deepEqual(requests, [null, 'same-page', null]);
    reader.close(); assert.equal(signal.aborted, true);
    fresh.resolve(page([row('late', 'late answer')])); await tick();
    assert.equal(f.q('.place-convo'), null);
});

test('a time outside the current year names its year, in rows and in the room list', async (t) => {
    const year = new Date().getFullYear();
    const f = fixture(t);
    f.reply(() => page([{ ...row('then', 'two years ago'), ts: `${year - 2}-06-15T12:00:00Z` }, { ...row('now', 'this year'), ts: `${year}-06-15T12:00:00Z` }]));
    const reader = mount(f); await tick();
    const times = [...f.doc.body.querySelectorAll('time')].map((node) => node.textContent);
    assert.match(times[0], new RegExp(String(year - 2)));
    assert.doesNotMatch(times[1], new RegExp(String(year)));
    assert.match(placeRoomsHtml({ rooms: [{ ...room, last_at: `${year - 1}-03-01T08:00:00Z` }], coverage }), new RegExp(String(year - 1)));
    reader.close();
});

test('the start is the record saved when it was read; Refresh history reads the newest page again', async (t) => {
    const f = fixture(t); const requests = [];
    const reader = mount(f, { readHistory: async ({ cursor }) => {
        requests.push(cursor);
        if (requests.length === 1) return page([row('b', 'newest')], 'older');
        if (cursor === 'older') return page([row('a', 'oldest as read')]);
        return page([row('z', 'imported since'), row('b', 'newest')], 'after-import');
    } });
    await tick();
    assert.equal(f.q('[data-history-refresh]').hidden, false);
    f.q('[data-history-older]').click(); await tick();
    assert.match(f.q('[data-history-coverage]').textContent.trim(), /^Start of the record saved when this was read\. Refresh history to include anything saved since\./);
    f.q('[data-history-refresh]').click();
    assert.equal(f.q('[data-history-refresh]').disabled, true, 'one read at a time');
    await tick();
    assert.deepEqual(requests, [null, 'older', null]);
    assert.deepEqual(shownTexts(f), ['imported since', 'newest']);
    assert.match(f.q('[data-history-coverage]').textContent, /Earlier saved records are available\./);
    assert.equal(f.q('[data-history-refresh]').disabled, false); reader.close();
});

test('a row shows the send time its arrival recorded; the title keeps when it was actually saved', async (t) => {
    const imported = { ...row('i', 'imported'), ts: '2024-03-01T10:00:00+00:00', place_history: { kind: 'message',
        sent_at: '2024-03-01T10:00:00+00:00', historical_import: { authorship: 'participant', received_at: '2026-10-26T09:00:00+00:00' } } };
    const late = { ...row('l', 'late live'), ts: '2026-10-26T09:05:00+00:00', place_history: { kind: 'message', sent_at: '2026-10-26T08:00:00+00:00' } };
    const receipt = { ...row('r', 'receipt'), ts: '2026-10-26T09:06:00+00:00', place_history: { kind: 'delivery' } };
    assert.deepEqual(shownTime(imported), { at: imported.ts, title: 'Sent 2024-03-01T10:00:00+00:00 · saved 2026-10-26T09:00:00+00:00' });
    assert.deepEqual(shownTime(late), { at: '2026-10-26T08:00:00+00:00', title: 'Sent 2026-10-26T08:00:00+00:00 · saved 2026-10-26T09:05:00+00:00' });
    assert.deepEqual(shownTime(receipt), { at: receipt.ts, title: receipt.ts }, 'no send time is invented');
    const f = fixture(t); f.reply(() => page([imported, late, receipt])); const reader = mount(f); await tick();
    const times = [...f.doc.body.querySelectorAll('time')];
    assert.deepEqual(times.map((node) => node.getAttribute('datetime')), [imported.ts, '2026-10-26T08:00:00+00:00', receipt.ts]);
    assert.match(times[0].textContent, /2024/); reader.close();
});
