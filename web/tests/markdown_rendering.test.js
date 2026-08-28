import assert from 'node:assert/strict';
import test from 'node:test';

function escapeText(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

globalThis.document = {
    createElement() {
        let inner = '';
        return {
            set textContent(value) {
                inner = escapeText(value);
            },
            get innerHTML() {
                return inner;
            },
        };
    },
};

const { renderMarkdown } = await import('../modules/utils.js');

test('helper references render as safe inline code without literal escapes', () => {
    const source = 'Inspect with `get_task_result(<id>)` / `peek_task(<id>)`.';
    const html = renderMarkdown(source);

    assert.equal(
        html,
        'Inspect with <code class="inline-code">get_task_result(&lt;id&gt;)</code> / ' +
            '<code class="inline-code">peek_task(&lt;id&gt;)</code>.',
    );
    assert.equal(html.includes('\\'), false);
    assert.equal(html.includes('(<id>)'), false);
});

test('renderer preserves intentional backslashes instead of globally stripping them', () => {
    const html = renderMarkdown('`get\\_task\\_result(\\<id>)`');

    assert.match(html, /get\\_task\\_result\(\\&lt;id&gt;\)/);
});
