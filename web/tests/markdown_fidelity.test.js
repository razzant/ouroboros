import assert from 'node:assert/strict';
import test from 'node:test';

// The compact renderer escapes through a detached element and reads link
// addresses through a textarea; this stub does both the way a browser does.
const priorDocument = globalThis.document;
globalThis.document = {
    createElement: (tagName) => ({
        set textContent(value) {
            this.innerHTML = String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
        },
        set innerHTML(value) {
            this._html = String(value ?? '');
            if (tagName === 'textarea') {
                this.value = this._html.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
            }
        },
        get innerHTML() { return this._html; },
    }),
};
const { projectMarkdownLines, renderMarkdown } = await import('../modules/utils.js');
test.after(() => { globalThis.document = priorDocument; });

// The #1367 fence: every compact substitution has something to grab inside it.
const FENCE_BODY = [
    '**literal stars** and x = a * b * c',
    '- literal dash',
    '# comment',
    '[x](https://example.com/y)',
    '| a | b |',
    '|---|---|',
    '| 1 | 2 |',
    '`tick` and ~~strike~~ and ``double``',
    'a &amp; b &lt; c &#42; d <div>',
].join('\n');

const escaped = (value) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

test('a fenced block is the author text in both compact consumers', () => {
    const source = `Before\n\`\`\`md\n${FENCE_BODY}\n\`\`\`\n**after**`;
    const expected = `Before\n<pre><code>${escaped(FENCE_BODY)}\n</code></pre>\n<strong>after</strong>`;
    assert.equal(renderMarkdown(source), expected);
    assert.equal(renderMarkdown(source, { inlineHeadingBreaks: true }), expected);
});

test('punctuation labels and normal fence whitespace preserve code and preview boundaries', () => {
    for (const label of ['c++', 'c#', 'objective-c', 'shell-session', 'md-js', 'f#', 'asp.net']) {
        for (const ending of ['\n', '   \n', '\t\r\n']) {
            const source = '```' + label + ending + FENCE_BODY + '\n```';
            assert.equal(renderMarkdown(source), `<pre><code>${escaped(FENCE_BODY)}\n</code></pre>`);
            assert.equal(renderMarkdown(source, { inlineHeadingBreaks: true }), renderMarkdown(source));
            assert.equal(projectMarkdownLines(source), '```' + label + '\n' + FENCE_BODY + '\n```');
        }
    }
});

test('an inline backtick mention cannot steal the next fenced block', () => {
    for (const lead of [
        'Wrap it in ```python``` blocks.\nLater:', 'Use ``` fences like this\n**bold** here',
        'Enclose code in triple backticks (```).\n', 'Wrap it with ```:', 'Close the fence with ```.\nThen:',
        'Mention ```<>&',
    ]) {
        const html = renderMarkdown(lead + '\n```python\nx = *a*\n```\n**after**');
        assert.deepEqual([...html.matchAll(/<pre><code>([\s\S]*?)<\/code><\/pre>/g)].map((m) => m[1]), ['x = *a*\n']);
        assert.ok(html.endsWith('<strong>after</strong>'));
    }
    // Attributes remain outside the compact grammar; no claim of CommonMark parity.
    assert.doesNotMatch(renderMarkdown('```python title=demo\n# prose\n```'), /<pre>/);
});

test('a later inline-code pass cannot hide an already parked fenced block', () => {
    for (const size of [4, 5]) {
        for (const label of ['', 'python']) {
            const ticks = '`'.repeat(size);
            const html = renderMarkdown(ticks + label + '\nx = *a*\n' + ticks);
            // Longer fences keep the compact renderer's stray outer ticks, but
            // their code must not disappear into a nested internal placeholder.
            assert.equal(html, '`'.repeat(size - 3) + '<pre><code>x = *a*\n</code></pre>' + '`'.repeat(size - 3));
            assert.doesNotMatch(html, /OUROBOROSCODE/);
        }
    }
    assert.equal(renderMarkdown('Use `literal` and **bold**'),
        'Use <code class="inline-code">literal</code> and <strong>bold</strong>');
});

test('an inline code span is literal and keeps formatting around it', () => {
    assert.equal(
        renderMarkdown('Use `**b** *i* ~~s~~ [x](https://e.com/) |a|` with **care** and `&amp;`'),
        'Use <code class="inline-code">**b** *i* ~~s~~ [x](https://e.com/) |a|</code> with <strong>care</strong> and '
            + '<code class="inline-code">&amp;amp;</code>',
    );
    // Emphasis around a span still applies; the span itself stays literal.
    assert.equal(renderMarkdown('**see `a*b*c`**'), '<strong>see <code class="inline-code">a*b*c</code></strong>');
    // A pipe inside a code span does not split a table cell.
    assert.match(renderMarkdown('| k | v |\n| --- | --- |\n| `a|b` | 2 |'), /<td><code class="inline-code">a\|b<\/code><\/td><td>2<\/td>/);
});

test('an author text holding the parking token reads back unchanged', () => {
    const source = 'OUROBOROSCODE0OUROBOROSCODE and `code` and OUROBOROSCODEX1OUROBOROSCODEX';
    assert.equal(
        renderMarkdown(source),
        'OUROBOROSCODE0OUROBOROSCODE and <code class="inline-code">code</code> and OUROBOROSCODEX1OUROBOROSCODEX',
    );
});

test('a hostile parking-token run stays literal and costs one scan', () => {
    // A stem grown one X per rescan of the text was quadratic, and past a long
    // run its restore pattern no longer compiled: both consumers threw.
    const run = `OUROBOROSCODE${'X'.repeat(100000)}`;
    const started = Date.now();
    const html = renderMarkdown(`${run} ${'`a` '.repeat(2000)}**b**`);
    assert.ok(Date.now() - started < 1000, 'the parking stem must be found in one scan');
    assert.equal(html, `${run} ${'<code class="inline-code">a</code> '.repeat(2000)}<strong>b</strong>`);
    assert.equal(projectMarkdownLines(`## \`x\` ${run}\nbody`), `\`x\` ${run}\nbody`);
    // Every short tag after the prefix taken: the stem still misses them all.
    const tags = Array.from({ length: 26 }, (_, i) => `OUROBOROSCODE${String.fromCharCode(65 + i)}0`).join(' ');
    assert.equal(renderMarkdown(`${tags} \`c\``), `${tags} <code class="inline-code">c</code>`);
});

test('the compact grammar the design names keeps its code literal', () => {
    // A fence with no language, closed mid-line, and a two-backtick span holding a
    // backtick: the edges of the grammar DESIGN.md promises for this renderer.
    assert.equal(renderMarkdown('```\n**x** <b>\n``` after'), '<pre><code>**x** &lt;b&gt;\n</code></pre> after');
    assert.equal(renderMarkdown('``a`**b**`` *c*'), '<code class="inline-code">a`**b**</code> <em>c</em>');
});

test('heading length counts code as visible text in the renderer and in previews', () => {
    // 76 visible characters inside the span: a heading. Stars inside code are
    // characters; the old pipeline turned them into tags and measured less.
    const short = '`' + '*'.repeat(4) + 'x'.repeat(72) + '`';
    assert.match(renderMarkdown(`## ${short}`), /^<strong class="md-h2"><code class="inline-code">\*{4}x{72}<\/code><\/strong>$/);
    const long = '`' + '**' + 'x'.repeat(40) + '**' + 'y'.repeat(37) + '`';
    assert.equal(renderMarkdown(`## ${long}`), `<code class="inline-code">**${'x'.repeat(40)}**${'y'.repeat(37)}</code>`);
    // The preview projection agrees: the 81-character line is prose (no separator).
    assert.equal(projectMarkdownLines(`## ${long}\nbody`), `${long}\nbody`);
    assert.equal(projectMarkdownLines(`## ${short}\nbody`), `${short} —\nbody`);
    // A link inside code is text, not a label with a hidden destination.
    assert.equal(projectMarkdownLines('## `[a](b)` label\nbody'), '`[a](b)` label —\nbody');
});

// Every anchor the compact renderer writes, in its one fixed shape: an href with
// nothing that could end the attribute, then the renderer's own attributes.
const ANCHOR = /^<a href="[^"<>`]*" target="_blank" rel="noopener noreferrer" class="md-link( md-image-ref)?">$/;
const anchorTags = (html) => html.match(/<a\b[^>]*>/g) || [];

test('code inside a link destination stays literal code and never reaches the attribute', () => {
    // The compact sinks (Skill Review report, task timeline, delegated rows) have
    // no sanitizer after this renderer: parked code restored into an escaped href
    // used to close the attribute with the code's own quote.
    const fence = '[open](https://example.com/```\n" onclick="alert(1)" data-x="\n```)';
    assert.equal(
        renderMarkdown(fence),
        '[open](https://example.com/<pre><code>" onclick="alert(1)" data-x="\n</code></pre>)',
    );
    const span = '![pic](https://example.com/`x" onerror="alert(1)`) then [docs](https://example.com/d)';
    const rendered = renderMarkdown(span, { inlineHeadingBreaks: true });
    assert.equal(
        rendered,
        '![pic](https://example.com/<code class="inline-code">x" onerror="alert(1)</code>) then '
            + '<a href="https://example.com/d" target="_blank" rel="noopener noreferrer" class="md-link">docs</a>',
    );
    for (const html of [renderMarkdown(fence), rendered]) {
        for (const tag of anchorTags(html)) assert.match(tag, ANCHOR);
    }
    // Code in a link's words is still that link's text.
    assert.equal(
        renderMarkdown('[see `a" b`](https://example.com/)'),
        '<a href="https://example.com/" target="_blank" rel="noopener noreferrer" class="md-link">'
            + 'see <code class="inline-code">a" b</code></a>',
    );
});

test('a Markdown image stays a visible, policy-checked link in the compact renderer', () => {
    assert.equal(
        renderMarkdown('Before ![diagram](https://example.com/a.png) after'),
        'Before <a href="https://example.com/a.png" target="_blank" rel="noopener noreferrer" class="md-link md-image-ref">'
            + 'Image: diagram</a> after',
    );
    assert.match(renderMarkdown('![x](javascript:alert(1))'), /^<a href="#" [^>]*class="md-link md-image-ref">Image: x<\/a>/);
    // An ordinary link is unchanged.
    assert.equal(
        renderMarkdown('[docs](https://example.com/)'),
        '<a href="https://example.com/" target="_blank" rel="noopener noreferrer" class="md-link">docs</a>',
    );
});

test('a compact destination is the address alone: its title is not in it, its parentheses are', () => {
    const anchor = (href, text, image = false) => `<a href="${href}" target="_blank" rel="noopener noreferrer" `
        + `class="md-link${image ? ' md-image-ref' : ''}">${text}</a>`;
    const cases = [
        ['![diagram](https://example.com/a.png "Architecture")', anchor('https://example.com/a.png', 'Image: diagram', true)],
        ["![diagram](https://example.com/a.png 'Architecture')", anchor('https://example.com/a.png', 'Image: diagram', true)],
        ['![diagram](https://example.com/a.png (Architecture))', anchor('https://example.com/a.png', 'Image: diagram', true)],
        ['[docs](https://example.com/ "Docs") next', `${anchor('https://example.com/', 'docs')} next`],
        ['![chart](https://example.com/wiki/Foo_(bar).png) after',
            `${anchor('https://example.com/wiki/Foo_(bar).png', 'Image: chart', true)} after`],
        ['See [wiki](https://example.com/wiki/Foo_(bar)).', `See ${anchor('https://example.com/wiki/Foo_(bar)', 'wiki')}.`],
        // A refused address is refused whole: nothing of it is left behind as text.
        ['![x](javascript:alert(1))', anchor('#', 'Image: x', true)],
        ['![x](javascript:alert(document.cookie) "t") after', `${anchor('#', 'Image: x', true)} after`],
        ['![x](data:image/png;base64,AAAA)', anchor('#', 'Image: x', true)],
        // A form this renderer does not read stays the author's text, never a guessed address.
        ['[a](https://example.com/a b)', '[a](https://example.com/a b)'],
        ['![a](https://example.com/a((b)))', '![a](https://example.com/a((b)))'],
        ['[a](https://example.com/a(b)', '[a](https://example.com/a(b)'],
        // Code in a title is still the author's code, never a dropped tooltip.
        ['![d](https://example.com/a.png "`x" onerror="y`")',
            '![d](https://example.com/a.png "<code class="inline-code">x" onerror="y</code>")'],
    ];
    for (const [source, expected] of cases) {
        const html = renderMarkdown(source);
        assert.equal(html, expected, source);
        for (const tag of anchorTags(html)) assert.match(tag, ANCHOR);
    }
});
