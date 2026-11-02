import { projectMarkdownLines } from '../modules/utils.js';
import assert from 'node:assert/strict';
import test from 'node:test';

class TextElement {
    constructor(tagName) {
        this.tagName = String(tagName || '').toLowerCase();
        this._innerHTML = '';
        this._textContent = '';
        this.value = '';
    }

    set textContent(value) {
        this._textContent = String(value ?? '');
        this._innerHTML = this._textContent
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;');
    }

    get textContent() { return this._textContent; }

    set innerHTML(value) {
        this._innerHTML = String(value ?? '');
        if (this.tagName === 'textarea') {
            this.value = this._innerHTML
                .replace(/&lt;/gi, '<')
                .replace(/&gt;/gi, '>')
                .replace(/&quot;/gi, '"')
                .replace(/&#0*39;/g, "'")
                .replace(/&#x0*27;/gi, "'")
                .replace(/&amp;/gi, '&');
        }
    }

    get innerHTML() { return this._innerHTML; }
}

const priorDocument = globalThis.document;
globalThis.document = { createElement: (tagName) => new TextElement(tagName) };
const { renderMarkdown, renderMarkdownSafe } = await import('../modules/utils.js');

test.after(() => { globalThis.document = priorDocument; });

test('legacy renderMarkdown pins GFM-style pipe-table shape and quirks', () => {
    const source = [
        '| First | Second |',
        '| :--- | ---: |',
        '| one | two |',
        '| three | four |',
    ].join('\n');
    // The separator's colons are the author's alignment, written as marked writes it.
    assert.equal(
        renderMarkdown(source),
        '<div class="md-table-wrap"><table class="md-table"><thead><tr><th align="left">First</th><th align="right">Second</th></tr></thead><tbody><tr><td align="left">one</td><td align="right">two</td></tr><tr><td align="left">three</td><td align="right">four</td></tr></tbody></table></div>',
    );
    assert.equal(
        renderMarkdown('| a | b |\n| --- | :-: |\n| 1 | 2 |'),
        '<div class="md-table-wrap"><table class="md-table"><thead><tr><th>a</th><th align="center">b</th></tr></thead><tbody><tr><td>1</td><td align="center">2</td></tr></tbody></table></div>',
    );
    assert.equal(renderMarkdown('| lone | row |'), '| lone | row |');
});

test('legacy renderMarkdown routes links through safeExternalUrl', () => {
    assert.equal(
        renderMarkdown('[Web](https://example.com/docs) [Mail](mailto:owner@example.com) [Bad](javascript:alert(1))'),
        '<a href="https://example.com/docs" target="_blank" rel="noopener noreferrer" class="md-link">Web</a> <a href="mailto:owner@example.com" target="_blank" rel="noopener noreferrer" class="md-link">Mail</a> <a href="#" target="_blank" rel="noopener noreferrer" class="md-link">Bad</a>',
    );
});

test('legacy renderMarkdown pins emphasis, inline code, and fake headings', () => {
    assert.equal(
        renderMarkdown('# One\n## Two\n### Three\n**bold** *italic* ~~strike~~ `code`'),
        '<strong class="md-h1">One</strong>\n<strong class="md-h2">Two</strong>\n<strong class="md-h3">Three</strong>\n<strong>bold</strong> <em>italic</em> <del>strike</del> <code class="inline-code">code</code>',
    );
});

test('legacy renderMarkdown reads double-backtick spans and demotes deep headings', () => {
    assert.equal(
        renderMarkdown('#### Four\n##### Five\n``a `b` c`` and `d`'),
        '<strong class="md-h3">Four</strong>\n<strong class="md-h3">Five</strong>\n<code class="inline-code">a `b` c</code> and <code class="inline-code">d</code>',
    );
});

test('legacy renderMarkdown keeps a marker-led paragraph as prose', () => {
    const paragraph = 'summary ' + 'word '.repeat(30).trim();
    assert.equal(renderMarkdown(`## ${paragraph}`), paragraph);
    assert.equal(renderMarkdown('## Short heading'), '<strong class="md-h2">Short heading</strong>');
    // The cap measures what the reader sees: rendered inline tags and entities are not length.
    const seventy = 'x'.repeat(70);
    assert.equal(renderMarkdown(`## **${seventy}**`), `<strong class="md-h2"><strong>${seventy}</strong></strong>`);
    assert.equal(renderMarkdown('## Tom & Jerry <3'), '<strong class="md-h2">Tom &amp; Jerry &lt;3</strong>');
    const longUrl = 'https://example.com/' + 'segment/'.repeat(14);
    assert.match(renderMarkdown(`## [Short](${longUrl})`), /^<strong class="md-h2"><a href=/);
});

test('legacy renderMarkdown discards fenced-code language', () => {
    assert.equal(
        renderMarkdown('```javascript\nconst less = 1 < 2;\n```'),
        '<pre><code>const less = 1 &lt; 2;\n</code></pre>',
    );
});

test('legacy renderMarkdown escapes raw and stored entity text without decoding', () => {
    assert.equal(renderMarkdown('<b>& raw</b>'), '&lt;b&gt;&amp; raw&lt;/b&gt;');
    assert.equal(renderMarkdown('&lt;b&gt;&amp;amp;'), '&amp;lt;b&amp;gt;&amp;amp;amp;');
    assert.equal(renderMarkdown('`<tag>&`'), '<code class="inline-code">&lt;tag&gt;&amp;</code>');
});

test('renderMarkdownSafe exposes escaped pre/code fallback without DOMPurify', () => {
    const priorMarked = globalThis.marked;
    const priorPurify = globalThis.DOMPurify;
    globalThis.marked = { parse: () => '<p>must not be used</p>' };
    globalThis.DOMPurify = undefined;
    try {
        assert.equal(
            renderMarkdownSafe('<script>alert(1)</script> & text', { preClass: 'publisher-md' }),
            '<pre class="publisher-md"><code>&lt;script&gt;alert(1)&lt;/script&gt; &amp; text</code></pre>',
        );
    } finally {
        globalThis.marked = priorMarked;
        globalThis.DOMPurify = priorPurify;
    }
});

test('projectMarkdownLines follows the renderer heading rule, projects bullets and table rows, and leaves inline markup alone', () => {
    assert.equal(projectMarkdownLines('## summary\n\nbody'), 'summary —\n\nbody');
    assert.equal(projectMarkdownLines('## Only'), 'Only');
    assert.equal(projectMarkdownLines(`## ${'y'.repeat(81)}\nnext`), `${'y'.repeat(81)}\nnext`);
    assert.equal(projectMarkdownLines('## [Short](http://x/very/long/url/that/is/not/visible/text)\nnext'), '[Short](http://x/very/long/url/that/is/not/visible/text) —\nnext');
    assert.equal(projectMarkdownLines('**bold**\n- item'), '**bold**\nitem');
    // Line-anchored markers go while the newlines still exist; a fenced list stays code.
    assert.equal(projectMarkdownLines('| a | b |\n|---|---|\n| 1 | 2 |'), 'a b\n\n1 2');
    assert.equal(projectMarkdownLines('```md\n- kept\n```\n- item'), '```md\n- kept\n```\nitem');
    assert.equal(projectMarkdownLines('```sh\n# comment\nls\n```\nafter'), '```sh\n# comment\nls\n```\nafter');
    assert.equal(projectMarkdownLines('## Done —\nnext'), 'Done —\nnext');
    assert.equal(projectMarkdownLines('## Title\n```\ncode\n```'), 'Title\n```\ncode\n```');
    // One fence grammar with the renderer: indented and punctuation-labelled
    // openers keep the code literal in both views.
    assert.equal(projectMarkdownLines('   ```md\n## code\n```\nafter'), '   ```md\n## code\n```\nafter');
    // The renderer opens a fence on ANY line ending in ```<info>: a prefixed opener too,
    // and the fence closes on the next line that contains ```.
    assert.equal(projectMarkdownLines('prefix ```md\n## code\n```\n## after\ntext'), 'prefix ```md\n## code\n```\nafter —\ntext');
    assert.equal(projectMarkdownLines('```md\n## code\nend ```\n## after\ntext'), '```md\n## code\nend ```\nafter —\ntext');
    // An unclosed opener is ordinary text for the renderer, so its `##` line is a heading,
    // and a heading right before an unclosed opener is followed by text.
    assert.equal(projectMarkdownLines('```md\n## code\nmore'), '```md\ncode —\nmore');
    assert.match(renderMarkdown('```md\n## code\nmore'), /md-h2/);
    assert.equal(projectMarkdownLines('## Title\n```md\nmore'), 'Title —\n```md\nmore');
    assert.equal(projectMarkdownLines('## Title\n```md\nmore\n```'), 'Title\n```md\nmore\n```');
    // Trailing blanks and CRLF are ordinary fence syntax; the plain preview
    // normalizes line endings and trailing whitespace while preserving the code markers.
    assert.equal(projectMarkdownLines('```md   \n## code\n```'), '```md\n## code\n```');
    assert.match(renderMarkdown('```md   \n## code\n```'), /<pre>/);
    assert.equal(projectMarkdownLines('```md\r\n## code\r\n```'), '```md\n## code\n```');
    assert.match(renderMarkdown('```md\r\n## code\r\n```'), /<pre>/);
    // Linear on hostile brackets: 40k unmatched `[` inside one heading line.
    const brackets = Date.now(); projectMarkdownLines(`## ${'['.repeat(40000)}\nnext`);
    assert.ok(Date.now() - brackets < 500, 'link projection must stay linear');
    assert.equal(projectMarkdownLines('## [a](x) [b\nnext'), '[a](x) [b —\nnext');
    // `&lt;` typed raw is four visible characters (the renderer shows it literally).
    assert.equal(projectMarkdownLines(`## ${'x'.repeat(79)}&lt;\nnext`), `${'x'.repeat(79)}&lt;\nnext`);
    // An entity-like literal is visible as typed on the raw path (9 characters).
    assert.equal(projectMarkdownLines(`## ${'q'.repeat(72)}&abcdefg;\nnext`), `${'q'.repeat(72)}&abcdefg;\nnext`);
    assert.equal(projectMarkdownLines(`## ${'q'.repeat(71)}&abcdefg;\nnext`), `${'q'.repeat(71)}&abcdefg; —\nnext`);
    // A punctuation-labelled closed fence is code; an unclosed one remains prose.
    assert.equal(projectMarkdownLines('```md-js\n## code\n```'), '```md-js\n## code\n```');
    assert.equal(projectMarkdownLines('```md-js\n## code\ntext'), '```md-js\ncode —\ntext');
    assert.doesNotMatch(renderMarkdown('```md-js\n## code\n```'), /md-h2/);
    assert.doesNotMatch(renderMarkdown('   ```md\n## code\n```\nafter'), /md-h2/);
    // Stage-correct visibility: on the rendered path an unmatched `*` is a visible
    // character (80 x + `*` = 81 → prose); on the raw path a literal `<…>` or `&amp;`
    // is visible too, and a styled `**Steps:**` still ends in a colon.
    assert.equal(renderMarkdown(`## ${'x'.repeat(80)}*`), `${'x'.repeat(80)}*`); // prose keeps its text, not the markers
    assert.match(renderMarkdown(`## ${'x'.repeat(79)}*`), /^<strong class="md-h2">/);
    assert.equal(projectMarkdownLines(`## <${'x'.repeat(79)}>\nnext`), `<${'x'.repeat(79)}>\nnext`);
    assert.equal(projectMarkdownLines(`## ${'x'.repeat(76)}&amp;\nnext`), `${'x'.repeat(76)}&amp;\nnext`); // 76 + 5 literal chars = 81 → prose
    assert.equal(projectMarkdownLines(`## ${'x'.repeat(74)}&amp;\nnext`), `${'x'.repeat(74)}&amp; —\nnext`);
    assert.equal(projectMarkdownLines('## **Steps:**\nnext'), '**Steps:**\nnext');
    assert.equal(projectMarkdownLines('## *Done —*\nnext'), '*Done —*\nnext');
    // Raw `<okay>` is visible text (the renderer escapes it): a heading of 78 chars
    // including such a literal is still a heading on the raw path.
    const withAngle = `${'q'.repeat(72)}<okay>`;
    assert.equal(projectMarkdownLines(`## ${withAngle}\nnext`), `${withAngle} —\nnext`);
    assert.equal(projectMarkdownLines(`## ${'q'.repeat(75)}<okay>\nnext`), `${'q'.repeat(75)}<okay>\nnext`);
    // Linear on many headings, not only on one long line.
    const many = Array.from({ length: 20000 }, (_, i) => `## h${i}\ntext ${i}`).join('\n');
    const started = Date.now(); projectMarkdownLines(many);
    assert.ok(Date.now() - started < 1000, 'many headings must stay linear');
    // Raw span markers are not visible text: a bold 78-char heading is still a heading.
    const bold78 = `**${'z'.repeat(78)}**`;
    assert.equal(projectMarkdownLines(`## ${bold78}\nnext`), `${bold78} —\nnext`);
    assert.equal(renderMarkdown(`## ${bold78}`), `<strong class="md-h2"><strong>${'z'.repeat(78)}</strong></strong>`);
});
