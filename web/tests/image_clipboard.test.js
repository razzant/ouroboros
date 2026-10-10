import assert from 'node:assert/strict';
import test from 'node:test';
import { fixture } from './helpers/media_dom.js';

for (const outcome of ['success', 'rejected', 'unavailable', 'no-item']) {
    test(`photo copy reports the binary clipboard outcome: ${outcome}`, async () => {
        const fx = fixture();
        const previous = { item: globalThis.ClipboardItem, timer: globalThis.setTimeout };
        const writes = [];
        let textWrites = 0;
        try {
            globalThis.setTimeout = () => 0;
            document.getElementById = (id) => document.body.children.find((node) => node.id === id);
            globalThis.ClipboardItem = outcome === 'no-item' ? undefined : class {
                constructor(values) { this.values = values; }
            };
            navigator.clipboard = {
                writeText: async () => { textWrites += 1; },
                ...(outcome === 'unavailable' ? {} : { write: async (items) => {
                    if (outcome === 'rejected') throw new Error('Permission denied');
                    writes.push(...items);
                } }),
            };
            const bubble = fx.controller.buildMediaBubble({
                type: 'photo', role: 'assistant', task_id: 't', image_base64: 'aGVsbG8=', mime: 'image/png',
            });
            await bubble.querySelector('[data-photo-action="copy"]').click();
            assert.equal(textWrites, 0, 'copy image never writes a URL');
            const toast = document.body.querySelector('.toast');
            if (outcome === 'success') {
                assert.equal(writes.length, 1);
                assert.equal(await writes[0].values['image/png'].text(), 'hello');
                assert.equal(toast.textContent, 'Image copied.');
                assert.ok(toast.classList.contains('toast-ok'));
            } else {
                assert.equal(writes.length, 0);
                assert.match(toast.textContent, /Could not copy image:/);
                assert.ok(toast.classList.contains('toast-danger'));
            }
            assert.ok(bubble.querySelector('[data-photo-action="open"]'));
            assert.ok(bubble.querySelector('[data-photo-action="download"]'));
        } finally {
            fx.controller.destroy();
            fx.restore();
            globalThis.ClipboardItem = previous.item;
            globalThis.setTimeout = previous.timer;
        }
    });
}
