import './fixtures/call_modules.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { acquire, currentHolder } from '../modules/media_capture.js';
const { bindDeviceSettings } = await import('../../skills/live_conversation/ui/call_device_settings.js');
const { loadDevicePrefs, DEVICES_CHANGED } = await import('../../skills/live_conversation/ui/call_devices.js');

function fixture(getUserMedia) {
    const nodes = new Map();
    const node = selector => {
        if (!nodes.has(selector)) nodes.set(selector, {dataset:{},textContent:'',replaceChildren(...children) {this.children = children;}});
        return nodes.get(selector);
    };
    const root = Object.assign(new EventTarget(), {querySelector:node,isConnected:true,rects:[{}],getClientRects() {return this.rects;}});
    const doc = Object.assign(new EventTarget(), {createElement:() => ({}),hidden:false});
    const env = Object.assign(new EventTarget(), { document:doc, CustomEvent,
        navigator:{mediaDevices:{getUserMedia,enumerateDevices:async () => [{kind:'audioinput',deviceId:'mic',label:'Fixture mic'}]}},
        localStorage:{getItem:() => null,setItem() {}}, AudioContext:class {},
        setTimeout:() => 1,clearTimeout() {},cancelAnimationFrame() {} });
    const previous = { navigator:Object.getOwnPropertyDescriptor(globalThis,'navigator'), secure:globalThis.isSecureContext };
    Object.defineProperty(globalThis,'navigator',{configurable:true,value:env.navigator});
    globalThis.isSecureContext = true;
    const dispose = bindDeviceSettings(root,env);
    function emit(type,target) {
        const event = new Event(type);
        Object.defineProperty(event,'target',{value:target});
        root.dispatchEvent(event);
    }
    return {env,root,node,dispose,click:kind => emit('click',{closest:() => ({dataset:{callTest:kind}})}),emit,
        restore() { dispose(); if (previous.navigator) Object.defineProperty(globalThis,'navigator',previous.navigator);
            else delete globalThis.navigator; globalThis.isSecureContext = previous.secure; } };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('closing device settings releases pending permission and stops its late stream', async () => {
    let resolve, stopped = 0;
    const f = fixture(() => new Promise(done => {resolve = done;}));
    try {
        f.click('mic');
        assert.equal(currentHolder().owner, f.root);
        f.dispose();
        assert.equal(currentHolder(), null);
        resolve({getTracks:() => [{stop() {stopped++;}}]});
        await settle();
        assert.equal(stopped,1);
    } finally { f.restore(); }
});

test('leaving a retained form stops its test but keeps the controls bound for return', async () => {
    let resolve, stopped = 0;
    const f = fixture(() => new Promise(done => { resolve = done; }));
    const track = () => Object.assign(new EventTarget(), { stop() { stopped++; } });
    const levelShown = path => f.env.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path, place:'skills' } }));
    try {
        f.click('mic');
        levelShown('skill/live_conversation');
        assert.equal(currentHolder().owner, f.root, 'an announcement while still visible is not dismissal');
        f.root.isConnected = false;          // local Back retains the level but detaches its nodes
        levelShown('installed');
        assert.equal(currentHolder(), null, 'leave invalidates even a pending permission request');
        resolve({getTracks:() => [track()]});
        await settle();
        assert.equal(stopped, 1);

        f.root.isConnected = true;
        f.click('mic');
        assert.equal(currentHolder().owner, f.root, 'same form can start a test after returning');
        f.root.rects = [];                    // while the section is put away, the level can stay connected
        f.env.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'settings' } }));
        assert.equal(currentHolder(), null);
        resolve({getTracks:() => [track()]});
        await settle();
        assert.equal(stopped, 2);
    } finally { f.restore(); }
});

test('a test cannot steal call capture; changing the client choice announces it immediately', async () => {
    let captures = 0;
    const track = Object.assign(new EventTarget(), {stop() {}});
    const f = fixture(async () => {captures++;return {getTracks:() => [track]};});
    let lease;
    try {
        lease = await acquire({owner:'live-call',label:'Live Conversation',keepWhenHidden:true});
        f.click('mic'); await settle();
        assert.equal(captures,1);
        assert.equal(currentHolder().owner,'live-call');
        assert.match(f.node('[data-call-device-status]').textContent,/in use by Live Conversation/);
        const changes = [];
        f.env.addEventListener(DEVICES_CHANGED,event => changes.push(event.detail));
        f.emit('change',{dataset:{callDevice:'mic'},value:'mic'});
        assert.equal(loadDevicePrefs(f.env).mic,'mic');
        assert.equal(changes[0].mic,'mic');
        assert.equal(currentHolder().owner,'live-call');
    } finally { lease?.release(); f.restore(); }
});

test('a cancelled microphone test cannot attach to the next test while audio resume is pending', async () => {
    const f = fixture(async () => ({getTracks:() => [Object.assign(new EventTarget(), {stop() {}})]}));
    const contexts = [];
    f.env.requestAnimationFrame = () => 1;
    f.env.AudioContext = class {
        constructor() { this.analysers = 0; this.pending = new Promise(resolve => {this.finishResume = resolve;}); contexts.push(this); }
        resume() { return this.pending; }
        close() { return Promise.resolve(); }
        createMediaStreamSource() { return {connect() {}}; }
        createAnalyser() { this.analysers++; return {fftSize:512,getFloatTimeDomainData() {}}; }
    };
    try {
        f.click('mic'); await settle();
        f.click('mic'); // cancel while the first context is resuming
        f.click('mic'); await settle();
        assert.equal(contexts.length,2);
        contexts[0].finishResume(); await settle();
        assert.equal(contexts[1].analysers,0);
        contexts[1].finishResume(); await settle();
        assert.equal(contexts[1].analysers,1);
    } finally { contexts.forEach(context => context.finishResume()); f.restore(); }
});
