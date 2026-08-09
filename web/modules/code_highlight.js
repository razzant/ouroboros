/**
 * Tiny in-house syntax highlighter (redesign decision 19).
 *
 * XSS-safe BY CONSTRUCTION, not by sanitizing afterwards:
 *
 *   1. the input is treated as PLAIN TEXT and split into typed lexemes;
 *   2. EVERY lexeme's text is escaped;
 *   3. only then is each escaped lexeme wrapped in `<span class="tok-…">`.
 *
 * No markup from the source can survive step 2, so a file containing
 * `<script>`, `" onerror="…`, a stray `&`, backticks, or text that itself looks
 * like generated highlighter markup renders as visible characters. There is no
 * path where source bytes reach the DOM unescaped.
 *
 * `escapeHtmlAttr` is used (not `escapeHtmlText`) for two reasons: it is a pure
 * string function, so this module is node-testable without a DOM, and it also
 * escapes quotes and backticks — strictly more than text position requires.
 *
 * Scope is deliberately small (P7): one regex pass per LINE, no parser state
 * carried across lines, no worker, no async. A construct that spans lines (a
 * Python docstring, a multi-line `/* … *\/` comment) colors its delimiters and
 * leaves the body default — an honest limitation of a line tokenizer, never a
 * wrong claim about the code.
 *
 * Colors come exclusively from the Phase-0 syntax tokens
 * (`--code-keyword/self/string/number/comment/default`) via the `.tok-*`
 * classes in web/style.css; this module never names a hue.
 */

import { escapeHtmlAttr as escapeHtml } from './utils.js';

/** Lexeme roles. Each maps to one `.tok-<type>` class and one CSS token. */
export const TOKEN_TYPES = Object.freeze(['keyword', 'self', 'string', 'number', 'comment', 'default']);

/**
 * Rule sources MUST NOT contain capturing groups: the combined regex wraps each
 * rule in exactly one group and identifies the winning rule by group index.
 * Earlier rules win, so comments and strings are listed before keywords.
 */
const RULES = Object.freeze({
    python: [
        ['comment', /#[^\n]*/],
        ['string', /"""[\s\S]*?"""|'''[\s\S]*?'''|"""|'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/],
        ['self', /\b(?:self|cls)\b/],
        ['keyword', /\b(?:def|class|async|await|return|if|elif|else|while|for|try|except|finally|import|from|raise|with|as|pass|yield|lambda|not|and|or|in|is|None|True|False|global|nonlocal|assert|del|break|continue)\b/],
        ['number', /\b\d+(?:\.\d+)?\b/],
    ],
    markdown: [
        // A heading line is one lexeme: the prototype colored the whole line.
        ['keyword', /^#{1,6}[^\n]*/],
        ['comment', /^ {0,3}(?:`{3,}|~{3,})[^\n]*/],
        ['string', /`[^`\n]*`/],
    ],
    js: [
        ['comment', /\/\/[^\n]*|\/\*[\s\S]*?\*\/|\/\*[\s\S]*/],
        ['string', /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`/],
        ['self', /\b(?:this|super)\b/],
        ['keyword', /\b(?:const|let|var|function|return|if|else|for|while|do|switch|case|break|continue|class|extends|new|delete|typeof|instanceof|in|of|try|catch|finally|throw|async|await|import|export|from|default|yield|void|null|undefined|true|false)\b/],
        ['number', /\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/],
    ],
    json: [
        ['string', /"(?:\\.|[^"\\])*"/],
        ['keyword', /\b(?:true|false|null)\b/],
        ['number', /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/],
    ],
    css: [
        ['comment', /\/\*[\s\S]*?\*\/|\/\*[\s\S]*/],
        ['string', /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/],
        // Custom properties read as the file's own identifiers, like `self`.
        ['self', /--[\w-]+/],
        ['keyword', /@[\w-]+|\b(?:important|inherit|initial|unset|var|calc)\b/],
        ['number', /#[0-9a-fA-F]{3,8}\b|-?\b\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|fr|ms|s|deg)?\b/],
    ],
    yaml: [
        ['comment', /#[^\n]*/],
        ['string', /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/],
        ['keyword', /^[ \t]*(?:- )?[\w.$/-]+(?=[ \t]*:)|\b(?:true|false|null|yes|no)\b/],
        ['number', /\b\d+(?:\.\d+)?\b/],
    ],
});

const EXTENSIONS = Object.freeze({
    py: 'python', pyi: 'python',
    md: 'markdown', markdown: 'markdown',
    js: 'js', mjs: 'js', cjs: 'js', jsx: 'js', ts: 'js', tsx: 'js',
    json: 'json', jsonl: 'json',
    css: 'css',
    yml: 'yaml', yaml: 'yaml',
});

const COMPILED = new Map();

function compiled(language) {
    if (COMPILED.has(language)) return COMPILED.get(language);
    const rules = RULES[language];
    if (!rules) {
        COMPILED.set(language, null);
        return null;
    }
    const spec = {
        types: rules.map(([type]) => type),
        regex: new RegExp(rules.map(([, re]) => `(${re.source})`).join('|'), 'g'),
    };
    COMPILED.set(language, spec);
    return spec;
}

/** Language key for a path, or `'plain'` when the extension is unknown. */
export function languageForPath(path) {
    const name = String(path || '').split('/').filter(Boolean).pop() || '';
    const dot = name.lastIndexOf('.');
    if (dot <= 0) return 'plain';
    return EXTENSIONS[name.slice(dot + 1).toLowerCase()] || 'plain';
}

/**
 * Split ONE line of plain text into typed lexemes. Unmatched spans become
 * `default`, so concatenating `token.text` in order rebuilds the input exactly.
 */
export function tokenizeLine(line, language = 'plain') {
    const text = typeof line === 'string' ? line : '';
    if (!text) return [];
    const spec = compiled(String(language));
    if (!spec) return [{ type: 'default', text }];

    const tokens = [];
    const { regex, types } = spec;
    regex.lastIndex = 0;
    let cursor = 0;
    let match = regex.exec(text);
    while (match) {
        if (match[0] === '') {
            // A rule that can match empty would loop forever; step past it.
            regex.lastIndex += 1;
            match = regex.exec(text);
            continue;
        }
        if (match.index > cursor) tokens.push({ type: 'default', text: text.slice(cursor, match.index) });
        let type = 'default';
        for (let group = 1; group < match.length; group += 1) {
            if (match[group] !== undefined) { type = types[group - 1]; break; }
        }
        tokens.push({ type, text: match[0] });
        cursor = match.index + match[0].length;
        match = regex.exec(text);
    }
    if (cursor < text.length) tokens.push({ type: 'default', text: text.slice(cursor) });
    return tokens;
}

/** Tokens -> escaped, wrapped HTML. The ONLY function that emits markup. */
export function highlightTokens(tokens) {
    return (Array.isArray(tokens) ? tokens : [])
        .map((token) => `<span class="tok-${TOKEN_TYPES.includes(token.type) ? token.type : 'default'}">${escapeHtml(token.text)}</span>`)
        .join('');
}

/** One line of source -> safe HTML. Empty input yields an empty string. */
export function highlightLine(line, language = 'plain') {
    return highlightTokens(tokenizeLine(line, language));
}
