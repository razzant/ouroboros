import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
    WIDGET_FRAME_BORDER_RESERVE,
    WIDGET_FRAME_DEFAULT_HEIGHT,
    WIDGET_FRAME_MAX_HEIGHT,
} from '../modules/widgets.js';
import {
    moduleResizeScript,
} from '../modules/widget_frame.js';
import {
    classifyWidgetJobStatus,
    isRetryableWidgetError,
    readWidgetJobStatus,
    withWidgetRequestTimeout,
} from '../modules/widget_job.js';

function resizeHarness({
    floor = WIDGET_FRAME_DEFAULT_HEIGHT,
    maxHeight = 1000,
    borderReserve = WIDGET_FRAME_BORDER_RESERVE,
    initialHeight = 600,
    paddingBottom = 12,
    borderBottom = 4,
} = {}) {
    const state = { height: initialHeight };
    const messages = [];
    const sequence = [];
    const disposeCallbacks = [];
    const listeners = new Map();
    const styleWrites = { set: 0, remove: 0 };
    let observerCallback = null;
    let observerDisconnects = 0;
    const createStyle = (label, initial = {}) => {
        const declarations = new Map();
        if (initial.value) declarations.set('overflow-y', initial);
        return {
            getPropertyValue(name) { return declarations.get(name)?.value || ''; },
            getPropertyPriority(name) { return declarations.get(name)?.priority || ''; },
            setProperty(name, value, priority = '') {
                assert.equal(name, 'overflow-y');
                declarations.set(name, { value, priority });
                styleWrites.set += 1;
                sequence.push(`${label}:set:${value}:${priority}`);
            },
            removeProperty(name) {
                assert.equal(name, 'overflow-y');
                declarations.delete(name);
                styleWrites.remove += 1;
                sequence.push(`${label}:remove`);
            },
        };
    };
    const documentElement = { style: createStyle('html') };
    const root = {
        get scrollHeight() { return state.height; },
        getBoundingClientRect: () => ({ height: state.height, bottom: state.height }),
    };
    const body = {
        style: createStyle('body'),
        scrollHeight: 0,
        clientHeight: 0,
        getBoundingClientRect: () => ({ top: 0 }),
    };
    const document = {
        body,
        documentElement,
        getElementById: (id) => (id === 'root' ? root : null),
    };
    const overflowState = () => Object.fromEntries(
        [['html', documentElement], ['body', body]].map(([label, element]) => [label, {
            value: element.style.getPropertyValue('overflow-y'),
            priority: element.style.getPropertyPriority('overflow-y'),
        }]),
    );
    const window = {
        innerHeight: 768,
        parent: {
            postMessage(message) {
                messages.push(message);
                sequence.push('message');
            },
        },
        addEventListener(type, listener) { listeners.set(type, listener); },
        removeEventListener(type, listener) {
            if (listeners.get(type) === listener) listeners.delete(type);
        },
        __ouroWidgetOnDispose(callback) { disposeCallbacks.push(callback); },
    };
    class FakeResizeObserver {
        constructor(callback) { observerCallback = callback; }
        observe(target) { assert.equal(target, root); }
        disconnect() { observerDisconnects += 1; }
    }
    const getComputedStyle = (target) => {
        assert.equal(target, body);
        return {
            height: 'auto',
            paddingBottom: `${paddingBottom}px`,
            borderBottomWidth: `${borderBottom}px`,
        };
    };
    Function(
        'document',
        'window',
        'ResizeObserver',
        'getComputedStyle',
        moduleResizeScript('nonce', floor, maxHeight, borderReserve),
    )(document, window, FakeResizeObserver, getComputedStyle);
    return {
        messages,
        sequence,
        overflowState,
        styleWrites,
        listeners,
        observerDisconnects: () => observerDisconnects,
        resize(height) {
            state.height = height;
            observerCallback();
        },
        dispose() { disposeCallbacks.forEach((callback) => callback()); },
    };
}

test('widget frame contract keeps the bounded host geometry', () => {
    assert.equal(WIDGET_FRAME_DEFAULT_HEIGHT, 320);
    assert.equal(WIDGET_FRAME_MAX_HEIGHT, 8192);
    assert.equal(WIDGET_FRAME_BORDER_RESERVE, 2);
});

test('module auto-height owns only vertical overflow across cap transitions', () => {
    const harness = resizeHarness();
    assert.deepEqual(harness.overflowState(), {
        html: { value: 'hidden', priority: 'important' },
        body: { value: 'hidden', priority: 'important' },
    });
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 0 });
    assert.deepEqual(harness.messages.map((item) => item.height), [616]);
    assert.deepEqual(harness.sequence.slice(0, 3), [
        'html:set:hidden:important', 'body:set:hidden:important', 'message',
    ]);

    harness.resize(600);
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 0 });
    assert.equal(harness.messages.length, 1);

    harness.resize(1200);
    assert.deepEqual(harness.overflowState(), {
        html: { value: '', priority: '' }, body: { value: '', priority: '' },
    });
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
    assert.deepEqual(harness.messages.map((item) => item.height), [616, 1216]);

    harness.resize(1200);
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
    assert.equal(harness.messages.length, 2);

    harness.resize(500);
    assert.deepEqual(harness.overflowState(), {
        html: { value: 'hidden', priority: 'important' },
        body: { value: 'hidden', priority: 'important' },
    });
    assert.deepEqual(harness.styleWrites, { set: 4, remove: 2 });
    assert.deepEqual(harness.messages.map((item) => item.height), [616, 1216, 516]);
});

test('module overflow ownership covers floor equality, fixed-height no-op, and cleanup', () => {
    const floorCap = resizeHarness({ maxHeight: WIDGET_FRAME_DEFAULT_HEIGHT, initialHeight: 100 });
    assert.deepEqual(floorCap.overflowState(), {
        html: { value: '', priority: '' }, body: { value: '', priority: '' },
    });
    assert.deepEqual(floorCap.styleWrites, { set: 2, remove: 2 });

    const harness = resizeHarness();
    harness.dispose();
    assert.deepEqual(harness.overflowState(), {
        html: { value: '', priority: '' }, body: { value: '', priority: '' },
    });
    assert.deepEqual(harness.styleWrites, { set: 2, remove: 2 });
    assert.equal(harness.observerDisconnects(), 1);
    assert.equal(harness.listeners.has('load'), false);

    // Page host plus the framed mounts split out of it: the resize bridge wiring
    // lives in widget_module.js; the negative pin must hold across both files.
    const framedSource = ['widgets.js', 'widget_module.js']
        .map((name) => readFileSync(new URL(`../modules/${name}`, import.meta.url), 'utf8'))
        .join('\n');
    assert.match(framedSource, /const resizeBridge = autoHeight\s*\? moduleResizeScript\(/);
    assert.match(
        framedSource,
        /moduleResizeScript\(\s*nonce,\s*WIDGET_FRAME_DEFAULT_HEIGHT,\s*maxHeight,\s*WIDGET_FRAME_BORDER_RESERVE,/,
    );
    assert.doesNotMatch(framedSource, /scrolling="no"|syncModuleFrameScrolling/);
});

test('widget job retry classification distinguishes transport from terminal errors', () => {
    assert.equal(isRetryableWidgetError({ status: 408 }), true);
    assert.equal(isRetryableWidgetError({ status: 429 }), true);
    assert.equal(isRetryableWidgetError({ status: 503 }), true);
    assert.equal(isRetryableWidgetError({ name: 'TypeError' }), true);
    assert.equal(isRetryableWidgetError({ status: 400 }), false);
    assert.equal(isRetryableWidgetError({ status: 404 }), false);
    assert.equal(isRetryableWidgetError({ status: 200, retryable: false }), false);
    assert.equal(isRetryableWidgetError({ name: 'AbortError', retryable: true }), false);
});

test('widget jobs bound unknown status and reject a missing status', () => {
    assert.equal(classifyWidgetJobStatus('queued'), 'pending');
    assert.equal(classifyWidgetJobStatus('running'), 'pending');
    assert.equal(classifyWidgetJobStatus('done'), 'success');
    assert.equal(classifyWidgetJobStatus('failed'), 'failure');
    assert.equal(classifyWidgetJobStatus(''), 'invalid');
    assert.equal(classifyWidgetJobStatus(123), 'invalid');
    assert.equal(classifyWidgetJobStatus({}), 'invalid');
    assert.equal(classifyWidgetJobStatus([]), 'invalid');
    assert.equal(classifyWidgetJobStatus('mystery'), 'pending');
});

test('widget job status selection preserves explicit falsy status values', () => {
    assert.equal(readWidgetJobStatus({ status: 0, state: 'running' }), 0);
    assert.equal(readWidgetJobStatus({ status: false, state: 'running' }), false);
    assert.equal(readWidgetJobStatus({ status: '', state: 'running' }), '');
    assert.equal(readWidgetJobStatus({ state: 'running' }), 'running');
});

test('widget request timeout aborts the request and remains retryable', async () => {
    const controller = new AbortController();
    await assert.rejects(
        withWidgetRequestTimeout(
            (signal) => new Promise((_, reject) => {
                signal.addEventListener('abort', () => {
                    const error = new Error('aborted');
                    error.name = 'AbortError';
                    reject(error);
                }, { once: true });
            }),
            controller,
            5,
        ),
        (error) => error.code === 'WIDGET_REQUEST_TIMEOUT' && error.retryable === true,
    );
    assert.equal(controller.signal.aborted, true);
});

test('widget request timeout stays terminal when the task swallows abort', async () => {
    const controller = new AbortController();
    await assert.rejects(
        withWidgetRequestTimeout(
            () => new Promise((resolve) => setTimeout(() => resolve('late result'), 20)),
            controller,
            5,
        ),
        (error) => error.code === 'WIDGET_REQUEST_TIMEOUT' && error.retryable === true,
    );
    assert.equal(controller.signal.aborted, true);
});
