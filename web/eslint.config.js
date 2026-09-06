// ESLint second layer of the browser no-undef gate — CI-only.
//
// The hermetic commit gate keeps its own dependency-free walker
// (tests/no_undef.test.js on vendored acorn, run by `node --test` through
// ouroboros/preflight_node.py); this config is the independent second opinion
// both CI jobs run after that suite (owner decision D-13 = A). It carries ONE
// rule, `no-undef`, so a divergence between the two layers is a scope-model
// bug in one of them, never a style disagreement. The install is
// lockfile-frozen (`npm ci` on web/package-lock.json) and eslint/globals are
// pinned to exact versions, so a registry move cannot change the verdict.
import globals from 'globals';

// Vendored runtime scripts index.html loads as classic <script> tags — globals
// BY DESIGN (same list as KNOWN_GLOBALS in tests/no_undef.test.js).
const VENDORED_SCRIPT_GLOBALS = {
    Chart: 'readonly',
    marked: 'readonly',
    DOMPurify: 'readonly',
    mermaid: 'readonly',
    hljs: 'readonly',
};

export default [
    // The vendored parser the hermetic gate ships is not ours to lint.
    { ignores: ['tests/vendor/**', 'tests/fixtures/**', 'node_modules/**'] },
    {
        // The ES modules the page loads.
        files: ['app.js', 'modules/**/*.js'],
        languageOptions: {
            ecmaVersion: 2022,
            sourceType: 'module',
            globals: { ...globals.browser, ...globals.es2022, ...VENDORED_SCRIPT_GLOBALS },
        },
        rules: { 'no-undef': 'error' },
    },
    {
        // The node --test suite: browser globals it stubs by name plus node's.
        files: ['tests/**/*.test.js'],
        languageOptions: {
            ecmaVersion: 2022,
            sourceType: 'module',
            globals: { ...globals.browser, ...globals.es2022, ...globals.node, ...VENDORED_SCRIPT_GLOBALS },
        },
        rules: { 'no-undef': 'error' },
    },
];
