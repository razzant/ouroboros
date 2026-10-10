// Producers whose DOM lives inside roots the overlay never enters (a transcript, a live card,
// rendered markdown) must read their own words through the translation seam: nothing else will
// ever translate them, and nothing will ever report them missing. This guard makes a new English
// literal in such a producer a failing test instead of a reviewer's finding.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// composer_owner_controls.js is not listed: the composer is overlay-translated chrome, and its
// dynamic title and aria-valuetext already go through fmt()/tr().
const PRODUCERS = ['task_phase_chip.js', 'chat_decision.js', 'chat_media.js', 'task_continue.js', 'chat_markdown.js', 'cancel_presentation.js',
    'effort_chip.js'];
// Literal owner-visible text handed straight to the DOM: an assignment or an attribute write.
const PATTERNS = [
    /\.(?:textContent|title|placeholder|innerText)\s*=\s*(['"`])([A-Z][^'"`\n]{2,})\1/,
    /setAttribute\(\s*['"](?:aria-label|title|placeholder)['"]\s*,\s*(['"`])([A-Z][^'"`\n]{2,})\1/,
    /\.(?:textContent|title)\s*=\s*[^;\n]*\?\s*(['"])([A-Z][^'"\n]{2,})\1\s*:/,
];
// Not language: symbols and identifiers a producer may write as they are.
const ALLOWED = new Set([]);

test('producers inside overlay-excluded roots hand no bare English to the DOM', async () => {
    const found = [];
    for (const name of PRODUCERS) {
        const source = await readFile(new URL(`../modules/${name}`, import.meta.url), 'utf8');
        source.split('\n').forEach((line, index) => {
            if (/\b(?:tr|fmt|tx)\(/.test(line)) return;
            for (const pattern of PATTERNS) {
                const match = pattern.exec(line);
                if (match && !ALLOWED.has(match[2])) found.push(`${name}:${index + 1}: ${match[2]}`);
            }
        });
    }
    assert.deepEqual(found, [], 'read these through tr()/fmt() (web/modules/i18n.js): the overlay never reaches them');
});

test('the guard itself sees a bare literal (it fails when the seam is removed)', () => {
    const line = "        button.textContent = 'Paused';";
    assert.ok(PATTERNS.some((pattern) => pattern.test(line)));
    assert.ok(!PATTERNS.some((pattern) => pattern.test("        button.textContent = label;")));
});
