import assert from 'node:assert/strict';
import test from 'node:test';

import {
    TOKEN_TYPES,
    highlightLine,
    languageForPath,
    tokenizeLine,
} from '../modules/code_highlight.js';

const types = (line, lang) => tokenizeLine(line, lang).map((token) => token.type);
const texts = (line, lang) => tokenizeLine(line, lang).map((token) => token.text);

// ---------------------------------------------------------------------------
// XSS matrix — the whole point of tokenize-then-escape (decision 19)
// ---------------------------------------------------------------------------

const HOSTILE = [
    '<script>alert(1)</script>',
    '</span><script>alert(1)</script><span>',
    'x = "\\" onerror=\\"alert(1)"',
    "img = '<img src=x onerror=alert(1)>'",
    'a & b && c &amp; d',
    'print("a `b` c")',
    '<span class="tok-string">not really a token</span>',
    'value = "</code></div><script>fetch(\'/evil\')</script>"',
    '# comment with <b>bold</b> and "quotes" and `backticks`',
    'nested = "outer \'inner\' outer"  # trailing <i>comment</i>',
    '"""docstring with <svg onload=alert(1)> inside"""',
    'javascript:void(0)</textarea><script>x</script>',
];

const OWN_MARKUP = /<span class="tok-(?:keyword|self|string|number|comment|default)">|<\/span>/g;

test('no source byte can escape as markup, in any supported language', () => {
    const languages = ['python', 'markdown', 'js', 'json', 'css', 'yaml', 'plain'];
    for (const language of languages) {
        for (const line of HOSTILE) {
            const html = highlightLine(line, language);
            // Only highlighter-authored tags may appear.
            const tags = html.match(/<[^>]*>/g) || [];
            for (const tag of tags) {
                assert.ok(
                    /^<span class="tok-(?:keyword|self|string|number|comment|default)">$/.test(tag) || tag === '</span>',
                    `unexpected tag ${tag} for ${language}: ${line}`,
                );
            }
            // The invariant: remove the highlighter's OWN markup and nothing that
            // could start a tag, close an attribute, or open a template literal
            // is left — every source byte is an entity.
            const payload = html.replace(OWN_MARKUP, '');
            assert.ok(
                !/[<>"'`]/.test(payload),
                `unescaped character survived for ${language}: ${line} -> ${payload}`,
            );
            assert.ok(!/<script/i.test(html), `raw <script survived for ${language}: ${line}`);
        }
    }
});

test('every dangerous character is entity-escaped exactly once', () => {
    const html = highlightLine('a < b > c & d " e \' f ` g', 'plain');
    assert.equal(
        html,
        '<span class="tok-default">a &lt; b &gt; c &amp; d &quot; e &#39; f &#96; g</span>',
    );
    assert.ok(!html.includes('&amp;lt;'), 'double escaping would corrupt the source text');
});

test('source that looks like generated highlighter markup renders as text', () => {
    const line = '<span class="tok-keyword">def</span>';
    const html = highlightLine(line, 'python');
    // Strip the highlighter's own wrappers: what remains is the source line with
    // every angle bracket and quote turned into an entity, so the browser shows
    // the markup instead of adopting it.
    assert.equal(
        html.replace(OWN_MARKUP, ''),
        '&lt;span class=&quot;tok-keyword&quot;&gt;def&lt;/span&gt;',
    );
    // `def` and `"tok-keyword"` are still classified — that is fine: the class
    // comes from the tokenizer, never from bytes in the file.
    assert.ok(html.includes('<span class="tok-keyword">def</span>'));
    assert.ok(html.includes('<span class="tok-string">&quot;tok-keyword&quot;</span>'));
});

test('an unknown language escapes but never classifies', () => {
    assert.equal(languageForPath('notes/todo.unknownext'), 'plain');
    assert.equal(languageForPath('Makefile'), 'plain');
    assert.deepEqual(tokenizeLine('def x(): # not python here', 'plain'), [
        { type: 'default', text: 'def x(): # not python here' },
    ]);
    assert.equal(
        highlightLine('<b>&</b>', 'plain'),
        '<span class="tok-default">&lt;b&gt;&amp;&lt;/b&gt;</span>',
    );
});

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

test('tokens concatenate back to the exact input line', () => {
    const samples = [
        ['python', '    return self._value + 42  # keep'],
        ['markdown', '## Heading with `code` inside'],
        ['js', "const x = `tpl ${y}`; // note"],
        ['json', '{"a": 1, "b": true, "c": null}'],
        ['css', '.x { color: var(--code-string); width: 12px; } /* c */'],
        ['yaml', 'key: "value"  # trailing'],
    ];
    for (const [language, line] of samples) {
        assert.equal(texts(line, language).join(''), line, language);
        for (const token of tokenizeLine(line, language)) {
            assert.ok(TOKEN_TYPES.includes(token.type), `${language}: ${token.type}`);
        }
    }
});

test('python: keywords, self, strings, numbers and comments', () => {
    assert.deepEqual(types('def run(self, n=3):  # go', 'python'), [
        'keyword', 'default', 'self', 'default', 'number', 'default', 'comment',
    ]);
    // A `#` inside a string is not a comment; a quote inside a comment is not a string.
    assert.deepEqual(tokenizeLine('x = "a # b"', 'python'), [
        { type: 'default', text: 'x = ' },
        { type: 'string', text: '"a # b"' },
    ]);
    assert.deepEqual(tokenizeLine('# say "hi"', 'python'), [{ type: 'comment', text: '# say "hi"' }]);
    // Nested quote styles stay inside one string lexeme.
    assert.deepEqual(tokenizeLine(`t = "it's here"`, 'python'), [
        { type: 'default', text: 't = ' },
        { type: 'string', text: `"it's here"` },
    ]);
    // A single-line docstring is one string; an unterminated one colors only its
    // delimiter (a line tokenizer carries no cross-line state, by design).
    assert.deepEqual(tokenizeLine('"""one line."""', 'python'), [
        { type: 'string', text: '"""one line."""' },
    ]);
    assert.deepEqual(types('"""open', 'python'), ['string', 'default']);
});

test('markdown headings and inline code follow the prototype', () => {
    assert.deepEqual(tokenizeLine('# SYSTEM', 'markdown'), [{ type: 'keyword', text: '# SYSTEM' }]);
    assert.deepEqual(tokenizeLine('plain `code` tail', 'markdown'), [
        { type: 'default', text: 'plain ' },
        { type: 'string', text: '`code`' },
        { type: 'default', text: ' tail' },
    ]);
    assert.deepEqual(tokenizeLine('```python', 'markdown'), [{ type: 'comment', text: '```python' }]);
    // `#` mid-line is not a heading.
    assert.deepEqual(tokenizeLine('not # a heading', 'markdown'), [
        { type: 'default', text: 'not # a heading' },
    ]);
});

test('js / json / css / yaml classify their own shapes', () => {
    assert.deepEqual(types('const a = this.b; // x', 'js'), [
        'keyword', 'default', 'self', 'default', 'comment',
    ]);
    assert.deepEqual(tokenizeLine('/* whole line */', 'js'), [
        { type: 'comment', text: '/* whole line */' },
    ]);
    assert.deepEqual(types('"key": 12.5', 'json'), ['string', 'default', 'number']);
    assert.deepEqual(types('null', 'json'), ['keyword']);
    assert.deepEqual(types('@media (min-width: 10px)', 'css'), ['keyword', 'default', 'number', 'default']);
    assert.deepEqual(types('  --files-indent: 0;', 'css'), ['default', 'self', 'default', 'number', 'default']);
    assert.deepEqual(types('name: value  # c', 'yaml'), ['keyword', 'default', 'comment']);
    // A yaml key lexeme carries its own indent (and list dash): whitespace has no
    // color, so keeping it inside the lexeme costs nothing and avoids a lookbehind.
    assert.deepEqual(tokenizeLine('  - enabled: true', 'yaml'), [
        { type: 'keyword', text: '  - enabled' },
        { type: 'default', text: ': ' },
        { type: 'keyword', text: 'true' },
    ]);
});

test('empty and non-string input produce nothing', () => {
    assert.deepEqual(tokenizeLine('', 'python'), []);
    assert.deepEqual(tokenizeLine(null, 'python'), []);
    assert.equal(highlightLine('', 'python'), '');
});

test('extensions map to languages, unknown ones to plain', () => {
    assert.equal(languageForPath('/Users/o/ouroboros/loop.py'), 'python');
    assert.equal(languageForPath('prompts/SYSTEM.md'), 'markdown');
    assert.equal(languageForPath('web/app.js'), 'js');
    assert.equal(languageForPath('data/settings.json'), 'json');
    assert.equal(languageForPath('web/style.css'), 'css');
    assert.equal(languageForPath('deploy/compose.YAML'), 'yaml');
    assert.equal(languageForPath('archive.tar.gz'), 'plain');
    assert.equal(languageForPath('.env'), 'plain');
    assert.equal(languageForPath(''), 'plain');
    // An unknown language is not an error: it tokenizes as one escaped default
    // lexeme, which is what `'plain'` means downstream.
    assert.deepEqual(tokenizeLine('def f():', 'plain'), [{ type: 'default', text: 'def f():' }]);
});
