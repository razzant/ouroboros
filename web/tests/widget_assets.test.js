import test from 'node:test';
import assert from 'node:assert/strict';
import { openWidgetAssetLease, maintainWidgetAssetLease } from '../modules/widget_assets.js';
import { planWidgetListPatch, widgetCardSignature } from '../modules/widget_list.js';

const flush = async () => { for (let i = 0; i < 6; i += 1) await new Promise(resolve => setImmediate(resolve)); };
const lease = { token: 'mount', base_url: '/api/widget-assets/mount/', expires_in: 120, renew_after: 40 };

test('the parent renews the same mount, releases once, and cancels future work', async t => {
    const priorWindow = globalThis.window, priorFetch = globalThis.fetch;
    const target = new EventTarget();
    target.location = { origin: 'https://host' };
    globalThis.window = target;
    const requests = [];
    globalThis.fetch = async (url, init) => { requests.push({url, ...init}); return {ok:true, json:async () => lease}; };
    t.mock.timers.enable({apis:['setTimeout']});
    try {
        const opened = await openWidgetAssetLease({skill:'demo',tab_id:'main',generation:'g1'});
        assert.equal(opened.baseUrl, 'https://host/api/widget-assets/mount/');
        let lost = 0;
        const release = maintainWidgetAssetLease(opened, () => lost++);
        t.mock.timers.tick(40000); await flush();
        assert.ok(requests[1].url.endsWith('/renew'));
        assert.deepEqual(JSON.parse(requests[1].body), {token:'mount'});
        release(); release(); await flush();
        assert.equal(requests.filter(r => r.url.endsWith('/revoke')).length, 1);
        assert.equal(requests.at(-1).keepalive, true);
        t.mock.timers.tick(120000); await flush();
        assert.equal(lost, 0);
        assert.equal(requests.length, 3);
    } finally { globalThis.window = priorWindow; globalThis.fetch = priorFetch; }
});

test('a lost renewal ends the mount; pagehide revokes and stops the lifecycle', async t => {
    const priorWindow = globalThis.window, priorFetch = globalThis.fetch;
    globalThis.window = new EventTarget();
    const urls = [];
    globalThis.fetch = async url => { urls.push(url); throw new Error('offline'); };
    t.mock.timers.enable({apis:['setTimeout']});
    try {
        let lost = 0;
        let release;
        release = maintainWidgetAssetLease(lease, () => { lost++; release(); });
        t.mock.timers.tick(40000); await flush();
        assert.equal(lost, 1);
        assert.equal(urls.filter(url => url.endsWith('/revoke')).length, 1);
        t.mock.timers.tick(120000); await flush();
        assert.equal(lost, 1);
        release = maintainWidgetAssetLease(lease, () => { lost++; release(); });
        window.dispatchEvent(new Event('pagehide')); await flush();
        assert.equal(lost, 2);
    } finally { globalThis.window = priorWindow; globalThis.fetch = priorFetch; }
});

test('same bytes with a new publication replace only an ESM mount; classic follows revision; rearrangement neither', () => {
    const classic = {key:'demo:main', revision:'same', generation:'g1', render:{kind:'module', entry:'widget.js'}};
    const esm = {key:'demo:esm', revision:'same', generation:'g1', render:{kind:'module', entry_type:'module', entry:'app/main.js'}};
    const republished = [classic, esm].map(tab => ({...tab, generation:'g2'}));
    assert.deepEqual(planWidgetListPatch([classic, esm], republished), {added:[], changed:['demo:esm'], removed:[]});
    assert.notEqual(widgetCardSignature(classic), widgetCardSignature({...classic, revision:'next'}));
    assert.equal(widgetCardSignature(esm), widgetCardSignature({...esm, order:2}));
});
