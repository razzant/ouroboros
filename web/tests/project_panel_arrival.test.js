// Docked, the Project panel comes in from the window's edge once its room has painted its words, its history or this
// browser's saved text of it (chat.js hasPaintedWords, chat_preview.js), or at its read's bound (DESIGN "Shell", owner
// 20.10 Q2 A). app.js is a boot script, so arrivePanel is lifted out of its source and run against stubbed module
// state and frames 16 ms apart, as project_history_ack.test.js lifts its functions.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../app.js', import.meta.url), 'utf8').replace(/\r\n?/g, '\n');
// The bound, the running arrival and the function, as app.js states them.
const from = source.indexOf('const PANEL_ARRIVAL_BOUND_MS = ');
const until = source.indexOf('\n}\n', source.indexOf('function arrivePanel(', from));
assert.ok(from >= 0 && until > from, 'arrivePanel and its bound are top-level in app.js');
const arrival = source.slice(from, until + 3);

function arrive(instance) {
    const frames = [], opened = [];
    let now = 0;
    const names = new Set();
    const context = vm.createContext({
        navState: { activeProjectId: 'p1' },
        projectInstances: new Map([['p1', instance]]),
        projectPanel: { classList: { contains: (name) => names.has(name), add: (name) => { names.add(name); opened.push(now); } } },
        isStackLayout: () => false, reducedMotion: () => false,
        afterMotion: () => ({ done: Promise.resolve() }),
        requestAnimationFrame: (fn) => frames.push(fn),
        performance: { now: () => now },
    });
    vm.runInContext(`${arrival}\narrivePanel('p1', 0);`, context);
    while (frames.length && !opened.length && now < 1000) { now += 16; frames.shift()(); }
    return opened[0];
}

test('docked, a room whose saved text painted comes in at once; one with no words waits for its read\'s bound', () => {
    assert.equal(arrive({ hasPaintedHistory: () => false, hasPaintedWords: () => true }), 16, 'it sets off at the first frame');
    const bound = Number(/const PANEL_ARRIVAL_BOUND_MS = (\d+);/.exec(arrival)[1]);
    const cold = arrive({ hasPaintedHistory: () => false, hasPaintedWords: () => false });
    assert.ok(cold >= bound && cold < bound + 32, `without words it sets off at the bound (${cold} ms)`);
});
