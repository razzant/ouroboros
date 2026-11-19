import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, readdirSync } from 'node:fs';
import { mountChatMarkdown } from '../modules/chat_markdown.js';

test('mounting Markdown carries presentation without wrapping the content or owning enhancement', () => {
    const classes = new Set(['card-body']);
    const host = { classList: { add: (name) => classes.add(name) }, innerHTML: 'old content' };
    // The renderer fallback also gets the contract; loading libraries cannot
    // determine whether a host has the CSS needed to display its output.
    mountChatMarkdown(host, '<tag>\nsecond line');
    assert.deepEqual([...classes], ['card-body', 'ui-rich-content']);
    assert.equal(host.innerHTML, '&lt;tag&gt;<br>second line');
    mountChatMarkdown(host, '');
    assert.equal(host.innerHTML, '', 'clearing content leaves no old rendered blocks');
    assert.equal(classes.size, 2, 'reusing a host does not grow a wrapper or another CSS role');
});

test('rich Markdown mounts use the seam or the existing message template contract', () => {
    const modules = new URL('../modules/', import.meta.url);
    // The existing message factory (a text row's bubble) builds one HTML template, preserving its
    // direct-child layout. Its explicit host is the one string-rendering seam.
    const bubble = readFileSync(new URL('chat_text_bubble.js', modules), 'utf8');
    assert.ok(bubble.includes("message${richMarkdown ? ' ui-rich-content' : ''}"));
    assert.match(bubble, /: renderChatMarkdown\(painted\);/);
    assert.equal([...bubble.matchAll(/\brenderChatMarkdown\s*\(/g)].length, 1,
        'a second raw render must not bypass the mounting seam inside chat_text_bubble.js');
    for (const file of readdirSync(modules).filter((name) => name.endsWith('.js') && !['chat_markdown.js', 'chat_text_bubble.js'].includes(name))) {
        const source = readFileSync(new URL(file, modules), 'utf8');
        assert.doesNotMatch(source, /\brenderChatMarkdown\b/,
            `${file}: mountChatMarkdown must attach the content contract with the HTML`);
    }
});
