// Every browser module must resolve every identifier it references.
//
// The class this pins: a call to a name that no longer exists anywhere — the
// onboarding wizard's `render()` kept calling `renderClaudeCliStatus()` after
// the Claude-runtime retirement deleted the function, so the first render
// painted, the save re-render threw ReferenceError, and the completion POST
// never left the browser ("Saving..." forever, issues #557/#607). `node
// --check` cannot see it (the file parses), the Python static pins cannot see
// it (they grep markup), and a browser test only sees it on the path it
// happens to drive. A static free-identifier walk over EVERY module sees it
// on every path, offline, in the same `node --test` lane CI and the hermetic
// commit gate already run — no network, no npm, no configuration to drift.
//
// Guarantee: over `web/app.js` and `web/modules/*.js` — the ES modules the
// page loads — every identifier READ at runtime resolves on the ES2022 scope
// chain or is one of the listed globals below. Not covered, by construction:
// inline <script> blocks in served HTML; anything reached only through a
// string (`eval`, `window[name]`, dynamic `import()` specifiers); `typeof name`
// (exempt by design — it never throws, matching ESLint's no-undef); and
// property names, which are not references.
//
// Scope model (ES2022, deliberately conservative): a name is DECLARED when it
// is a module-level import, a `var`/`let`/`const`/`function`/`class` binding,
// a function or catch parameter, or a class/named-function expression's own
// name. `var` and function declarations hoist to the enclosing function
// scope; everything else is block-scoped. Non-computed member properties,
// object keys, labels and export specifiers are not references. Computed
// keys and default values inside destructuring patterns (function, catch and
// declaration bindings alike) are expressions and are walked.
//
// The walker was validated during development against ESLint no-undef
// (browser+es2021 globals) over the module set and a corpus of ES2022 forms;
// the self-checks below are the in-tree proof it cannot rot into a no-op. The
// second test pins import LINKAGE: a stale named import is a load-time
// SyntaxError that kills the whole page, strictly worse than the dead-call
// class.

import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import * as acorn from './vendor/acorn.mjs';

const WEB_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');

// Globals the modules legitimately read without declaring. Browser/ECMAScript
// names are the standard surface; the last group is the vendored runtime
// scripts `index.html` loads as classic <script> tags (Chart.js, marked,
// DOMPurify, mermaid, highlight.js) — they are globals BY DESIGN, not
// undeclared references.
const KNOWN_GLOBALS = new Set([
    // ECMAScript
    'globalThis', 'undefined', 'NaN', 'Infinity', 'Object', 'Array', 'String', 'Number', 'Boolean',
    'Symbol', 'BigInt', 'Math', 'JSON', 'Date', 'RegExp', 'Error', 'TypeError', 'RangeError',
    'SyntaxError', 'ReferenceError', 'EvalError', 'URIError', 'AggregateError', 'Promise', 'Proxy',
    'Reflect', 'Map', 'Set', 'WeakMap', 'WeakSet', 'WeakRef', 'FinalizationRegistry', 'ArrayBuffer',
    'SharedArrayBuffer', 'DataView', 'Uint8Array', 'Int8Array', 'Uint8ClampedArray', 'Uint16Array',
    'Int16Array', 'Uint32Array', 'Int32Array', 'Float32Array', 'Float64Array', 'BigInt64Array',
    'BigUint64Array', 'Atomics', 'Intl', 'parseInt', 'parseFloat', 'isNaN', 'isFinite',
    'encodeURIComponent', 'decodeURIComponent', 'encodeURI', 'decodeURI', 'escape', 'unescape',
    'eval', 'Function', 'Iterator', 'structuredClone', 'queueMicrotask',
    // Browser
    'window', 'document', 'navigator', 'location', 'history', 'screen', 'console', 'alert',
    'confirm', 'prompt', 'fetch', 'Request', 'Response', 'Headers', 'FormData', 'URL',
    'URLSearchParams', 'AbortController', 'AbortSignal', 'Blob', 'File', 'FileReader',
    'TextEncoder', 'TextDecoder', 'WebSocket', 'EventSource', 'XMLHttpRequest', 'Event',
    'CustomEvent', 'KeyboardEvent', 'MouseEvent', 'PointerEvent', 'ClipboardEvent', 'DragEvent',
    'InputEvent', 'FocusEvent', 'MessageEvent', 'CloseEvent', 'ErrorEvent', 'PromiseRejectionEvent',
    'EventTarget', 'Node', 'Element', 'HTMLElement', 'HTMLInputElement', 'HTMLTextAreaElement',
    'HTMLSelectElement', 'HTMLButtonElement', 'HTMLAnchorElement', 'HTMLImageElement',
    'HTMLCanvasElement', 'HTMLIFrameElement', 'HTMLDialogElement', 'SVGElement', 'DocumentFragment',
    'Text', 'Range', 'Selection', 'DOMParser', 'XMLSerializer', 'MutationObserver',
    'ResizeObserver', 'IntersectionObserver', 'PerformanceObserver', 'performance',
    'requestAnimationFrame', 'cancelAnimationFrame', 'requestIdleCallback', 'cancelIdleCallback',
    'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'localStorage', 'sessionStorage',
    'indexedDB', 'crypto', 'Image', 'Audio', 'Notification', 'MediaSource', 'MediaRecorder',
    'ImageData', 'OffscreenCanvas', 'Path2D', 'DOMRect', 'DOMException', 'getComputedStyle',
    'matchMedia', 'atob', 'btoa',
    'CSS', 'ClipboardItem', 'Worker', 'SharedWorker', 'BroadcastChannel', 'MessageChannel',
    'ReadableStream', 'WritableStream', 'TransformStream', 'CompressionStream',
    'DecompressionStream', 'reportError',
    'devicePixelRatio', 'innerWidth', 'innerHeight', 'outerWidth', 'outerHeight', 'scrollX',
    'scrollY', 'pageXOffset', 'pageYOffset', 'visualViewport', 'speechSynthesis',
    'SpeechSynthesisUtterance', 'AudioContext', 'webkitAudioContext', 'HTMLMediaElement',
    'HTMLVideoElement', 'HTMLAudioElement', 'HTMLFormElement', 'HTMLLabelElement', 'HTMLDivElement',
    'HTMLSpanElement', 'HTMLUListElement', 'HTMLLIElement', 'HTMLTableElement', 'HTMLTemplateElement',
    'HTMLDetailsElement', 'HTMLOptionElement', 'HTMLStyleElement', 'HTMLScriptElement',
    'HTMLLinkElement', 'HTMLMetaElement', 'HTMLBodyElement', 'HTMLHeadElement', 'HTMLHtmlElement',
    'ShadowRoot', 'NodeList', 'HTMLCollection', 'NamedNodeMap', 'Attr', 'Comment', 'Document',
    'Window', 'Navigator', 'Storage', 'Location', 'History',
    // Vendored runtime scripts loaded by index.html as classic <script> tags.
    'Chart', 'marked', 'DOMPurify', 'mermaid', 'hljs',
]);

function moduleFiles() {
    const files = [join(WEB_DIR, 'app.js')];
    for (const name of readdirSync(join(WEB_DIR, 'modules')).sort()) {
        if (name.endsWith('.js')) files.push(join(WEB_DIR, 'modules', name));
    }
    return files;
}

// --- scope analysis -------------------------------------------------------

class Scope {
    constructor(parent, isFunction) {
        this.parent = parent;
        this.isFunction = isFunction;
        this.names = new Set();
    }
    functionScope() {
        let scope = this;
        while (!scope.isFunction && scope.parent) scope = scope.parent;
        return scope;
    }
    has(name) {
        for (let scope = this; scope; scope = scope.parent) {
            if (scope.names.has(name)) return true;
        }
        return false;
    }
}

function patternNames(pattern, out) {
    if (!pattern) return out;
    switch (pattern.type) {
        case 'Identifier': out.push(pattern.name); break;
        case 'ObjectPattern':
            for (const prop of pattern.properties) {
                patternNames(prop.type === 'RestElement' ? prop.argument : prop.value, out);
            }
            break;
        case 'ArrayPattern':
            for (const element of pattern.elements) patternNames(element, out);
            break;
        case 'RestElement': patternNames(pattern.argument, out); break;
        case 'AssignmentPattern': patternNames(pattern.left, out); break;
        default: break;
    }
    return out;
}

// Hoisting pass: `var` declarations bind in the nearest FUNCTION scope;
// `let`/`const`/`class` bind in the block they appear in. A function
// declaration binds in the scope of the statement list it appears in — the
// function body or the module for a top-level one, the BLOCK for a nested one
// (modules are strict code: `{ function f() {} } f()` is a ReferenceError).
function hoistDeclarations(body, scope) {
    for (const stmt of body) {
        if (!stmt) continue;
        switch (stmt.type) {
            case 'FunctionDeclaration':
                if (stmt.id) scope.names.add(stmt.id.name);
                break;
            case 'ClassDeclaration':
                if (stmt.id) scope.names.add(stmt.id.name);
                break;
            case 'VariableDeclaration': {
                const target = stmt.kind === 'var' ? scope.functionScope() : scope;
                for (const decl of stmt.declarations) for (const n of patternNames(decl.id, [])) target.names.add(n);
                break;
            }
            case 'ImportDeclaration':
                for (const spec of stmt.specifiers) scope.names.add(spec.local.name);
                break;
            case 'ExportNamedDeclaration':
            case 'ExportDefaultDeclaration':
                if (stmt.declaration) hoistDeclarations([stmt.declaration], scope);
                break;
            default: break;
        }
        hoistVarsDeep(stmt, scope);
    }
}

// `var` inside nested blocks/loops still hoists to the function scope; walk
// statements (not into nested functions, which get their own pass).
function hoistVarsDeep(node, scope) {
    if (!node || typeof node.type !== 'string') return;
    if (/Function/.test(node.type) || node.type === 'ClassDeclaration' || node.type === 'ClassExpression') return;
    if (node.type === 'VariableDeclaration' && node.kind === 'var') {
        for (const decl of node.declarations) for (const n of patternNames(decl.id, [])) scope.functionScope().names.add(n);
    }
    for (const key of Object.keys(node)) {
        if (key === 'type' || key === 'loc' || key === 'range') continue;
        const child = node[key];
        if (Array.isArray(child)) child.forEach((c) => c && typeof c.type === 'string' && hoistVarsDeep(c, scope));
        else if (child && typeof child.type === 'string') hoistVarsDeep(child, scope);
    }
}

function walk(node, scope, report, parent, key) {
    if (!node || typeof node.type !== 'string') return;
    switch (node.type) {
        case 'Program': {
            hoistDeclarations(node.body, scope);
            for (const stmt of node.body) walk(stmt, scope, report, node, 'body');
            return;
        }
        case 'ImportDeclaration':
        case 'ExportAllDeclaration':
            return;
        case 'ExportNamedDeclaration':
            if (node.declaration) walk(node.declaration, scope, report, node, 'declaration');
            // `export { a as b }` specifiers reference LOCAL names; a re-export
            // (`export { a } from './x.js'`) names bindings of the OTHER module.
            if (!node.source) for (const spec of node.specifiers) reference(spec.local, scope, report);
            return;
        case 'ExportDefaultDeclaration':
            walk(node.declaration, scope, report, node, 'declaration');
            return;
        case 'FunctionDeclaration':
        case 'FunctionExpression':
        case 'ArrowFunctionExpression': {
            const inner = new Scope(scope, true);
            if (node.type === 'FunctionExpression' && node.id) inner.names.add(node.id.name);
            // `arguments` binds in ordinary functions only; an arrow reads its
            // enclosing function's (a module-level arrow: ReferenceError).
            if (node.type !== 'ArrowFunctionExpression') inner.names.add('arguments');
            for (const param of node.params) {
                for (const n of patternNames(param, [])) inner.names.add(n);
            }
            for (const param of node.params) walkPatternDefaults(param, inner, report);
            if (node.body.type === 'BlockStatement') {
                hoistDeclarations(node.body.body, inner);
                for (const stmt of node.body.body) walk(stmt, inner, report, node.body, 'body');
            } else {
                walk(node.body, inner, report, node, 'body');
            }
            return;
        }
        case 'ClassDeclaration':
        case 'ClassExpression': {
            const inner = new Scope(scope, false);
            if (node.type === 'ClassExpression' && node.id) inner.names.add(node.id.name);
            if (node.superClass) walk(node.superClass, scope, report, node, 'superClass');
            for (const member of node.body.body) {
                if (member.computed) walk(member.key, inner, report, member, 'key');
                if (member.value) walk(member.value, inner, report, member, 'value');
                if (member.type === 'StaticBlock') {
                    const block = new Scope(inner, true);
                    hoistDeclarations(member.body, block);
                    for (const stmt of member.body) walk(stmt, block, report, member, 'body');
                }
            }
            return;
        }
        case 'BlockStatement': {
            const inner = new Scope(scope, false);
            hoistDeclarations(node.body, inner);
            for (const stmt of node.body) walk(stmt, inner, report, node, 'body');
            return;
        }
        case 'SwitchStatement': {
            walk(node.discriminant, scope, report, node, 'discriminant');
            const inner = new Scope(scope, false);
            for (const c of node.cases) hoistDeclarations(c.consequent, inner);
            for (const c of node.cases) {
                if (c.test) walk(c.test, inner, report, c, 'test');
                for (const stmt of c.consequent) walk(stmt, inner, report, c, 'consequent');
            }
            return;
        }
        case 'ForStatement':
        case 'ForInStatement':
        case 'ForOfStatement': {
            const inner = new Scope(scope, false);
            const head = node.type === 'ForStatement' ? node.init : node.left;
            if (head && head.type === 'VariableDeclaration') hoistDeclarations([head], inner);
            if (node.type === 'ForStatement') {
                if (node.init) walk(node.init, inner, report, node, 'init');
                if (node.test) walk(node.test, inner, report, node, 'test');
                if (node.update) walk(node.update, inner, report, node, 'update');
            } else {
                if (node.left.type !== 'VariableDeclaration') walk(node.left, inner, report, node, 'left');
                else for (const decl of node.left.declarations) walkPatternDefaults(decl.id, inner, report);
                walk(node.right, inner, report, node, 'right');
            }
            walk(node.body, inner, report, node, 'body');
            return;
        }
        case 'CatchClause': {
            const inner = new Scope(scope, false);
            if (node.param) for (const n of patternNames(node.param, [])) inner.names.add(n);
            // A destructuring catch parameter's computed keys and defaults are
            // expressions evaluated in the catch scope, exactly like function
            // parameters: `catch ({ [key()]: v = fallback() })` reads both.
            if (node.param) walkPatternDefaults(node.param, inner, report);
            hoistDeclarations(node.body.body, inner);
            for (const stmt of node.body.body) walk(stmt, inner, report, node.body, 'body');
            return;
        }
        case 'VariableDeclaration':
            for (const decl of node.declarations) {
                walkPatternDefaults(decl.id, scope, report);
                if (decl.init) walk(decl.init, scope, report, decl, 'init');
            }
            return;
        case 'MemberExpression':
            walk(node.object, scope, report, node, 'object');
            if (node.computed) walk(node.property, scope, report, node, 'property');
            return;
        case 'Property':
            if (node.computed) walk(node.key, scope, report, node, 'key');
            if (node.shorthand && node.value.type === 'Identifier' && parent && parent.type === 'ObjectExpression') {
                reference(node.value, scope, report);
                return;
            }
            walk(node.value, scope, report, node, 'value');
            return;
        case 'MethodDefinition':
        case 'PropertyDefinition':
            if (node.computed) walk(node.key, scope, report, node, 'key');
            if (node.value) walk(node.value, scope, report, node, 'value');
            return;
        case 'LabeledStatement':
            walk(node.body, scope, report, node, 'body');
            return;
        case 'BreakStatement':
        case 'ContinueStatement':
        case 'MetaProperty':
        case 'Literal':
        case 'ThisExpression':
        case 'Super':
        case 'TemplateElement':
        case 'EmptyStatement':
        case 'DebuggerStatement':
            return;
        case 'UnaryExpression':
            // `typeof name` never throws on an undeclared name (ESLint no-undef
            // exempts it too): a feature-detect must not turn the gate red.
            if (node.operator === 'typeof' && node.argument.type === 'Identifier') return;
            walk(node.argument, scope, report, node, 'argument');
            return;
        case 'Identifier':
            reference(node, scope, report);
            return;
        case 'AssignmentPattern':
        case 'ObjectPattern':
        case 'ArrayPattern':
        case 'RestElement':
            // A pattern in an assignment position ( [a, b] = ... ) references its
            // identifier targets; a member target (`({ a: obj.x } = o)`) is an
            // ordinary expression and is walked as one.
            for (const n of patternNames(node, [])) reference({ name: n, loc: node.loc }, scope, report);
            walkPatternDefaults(node, scope, report);
            walkAssignmentTargets(node, scope, report);
            return;
        default:
            break;
    }
    for (const childKey of Object.keys(node)) {
        if (childKey === 'type' || childKey === 'loc' || childKey === 'range' || childKey === 'start' || childKey === 'end') continue;
        const child = node[childKey];
        if (Array.isArray(child)) {
            for (const c of child) if (c && typeof c.type === 'string') walk(c, scope, report, node, childKey);
        } else if (child && typeof child.type === 'string') {
            walk(child, scope, report, node, childKey);
        }
    }
}

// Non-identifier targets of a destructuring ASSIGNMENT (`[o.x] = arr`,
// `({ k: o.y } = obj)`) are expressions: their object part is a reference.
function walkAssignmentTargets(pattern, scope, report) {
    if (!pattern) return;
    switch (pattern.type) {
        case 'AssignmentPattern': walkAssignmentTargets(pattern.left, scope, report); break;
        case 'ObjectPattern':
            for (const prop of pattern.properties) walkAssignmentTargets(prop.type === 'RestElement' ? prop.argument : prop.value, scope, report);
            break;
        case 'ArrayPattern': for (const element of pattern.elements) walkAssignmentTargets(element, scope, report); break;
        case 'RestElement': walkAssignmentTargets(pattern.argument, scope, report); break;
        case 'Identifier': break;
        default: walk(pattern, scope, report, null, null); break;
    }
}

// Default values inside destructuring patterns are expressions evaluated in
// the enclosing scope: `function f({ a = b })` references `b`.
function walkPatternDefaults(pattern, scope, report) {
    if (!pattern) return;
    switch (pattern.type) {
        case 'AssignmentPattern':
            walkPatternDefaults(pattern.left, scope, report);
            walk(pattern.right, scope, report, pattern, 'right');
            break;
        case 'ObjectPattern':
            for (const prop of pattern.properties) {
                if (prop.type === 'RestElement') walkPatternDefaults(prop.argument, scope, report);
                else {
                    if (prop.computed) walk(prop.key, scope, report, prop, 'key');
                    walkPatternDefaults(prop.value, scope, report);
                }
            }
            break;
        case 'ArrayPattern':
            for (const element of pattern.elements) walkPatternDefaults(element, scope, report);
            break;
        case 'RestElement':
            walkPatternDefaults(pattern.argument, scope, report);
            break;
        default: break;
    }
}

function reference(identifier, scope, report) {
    const name = identifier.name;
    if (scope.has(name) || KNOWN_GLOBALS.has(name)) return;
    const line = identifier.loc ? identifier.loc.start.line : '?';
    report.push(`${name} (line ${line})`);
}

export function undeclaredReferences(source, sourceFile = '<memory>') {
    const ast = acorn.parse(source, {
        ecmaVersion: 2024, sourceType: 'module', locations: true, allowHashBang: true,
    });
    const report = [];
    walk(ast, new Scope(null, true), report, null, null);
    return report.map((entry) => `${sourceFile}: ${entry}`);
}

// --- the pin ---------------------------------------------------------------

test('the checker itself catches an undeclared call (so it cannot rot into a no-op)', () => {
    const findings = undeclaredReferences(`
        import { helper } from './x.js';
        const state = { saving: false };
        function render() { helper(state); renderGone(); }
        export { render };
    `, 'probe.js');
    assert.deepEqual(findings, ['probe.js: renderGone (line 4)']);
});

test('arrows do not bind `arguments`, member targets of a destructuring assignment and catch-pattern expressions are walked, and `typeof` is exempt', () => {
    const findings = undeclaredReferences(`
        export const f = () => arguments;
        function g() { return () => arguments; }
        const o = {};
        ({ a: missingNested.x } = o);
        [missingArr.y] = [1];
        if (typeof maybeGlobal !== 'undefined') { g(); }
        try { g(); } catch ({ [missingKey()]: v }) { void v; }
        try { g(); } catch ({ v = missingDefault() }) { void v; }
        try { g(); } catch ([a = missingArrDefault()]) { void a; }
        export { o };
    `, 'forms2.js');
    assert.deepEqual(findings, [
        'forms2.js: arguments (line 2)',
        'forms2.js: missingNested (line 5)',
        'forms2.js: missingArr (line 6)',
        'forms2.js: missingKey (line 8)',
        'forms2.js: missingDefault (line 9)',
        'forms2.js: missingArrDefault (line 10)',
    ]);
});

test('a function declared inside a block is not visible outside it (strict-mode modules)', () => {
    const findings = undeclaredReferences(`
        if (true) { function onlyHere() {} onlyHere(); }
        onlyHere();
        function top() { { function inner() {} inner(); } return inner; }
        export { top };
    `, 'blocks.js');
    assert.deepEqual(findings, ['blocks.js: onlyHere (line 3)', 'blocks.js: inner (line 4)']);
});

test('the checker accepts every declaration form it must understand', () => {
    const findings = undeclaredReferences(`
        import def, { named as alias } from './x.js';
        import * as ns from './y.js';
        export const [a, { b, c: d = a }] = ns.pair;
        var hoisted = 1;
        let block = 2; const fixed = 3;
        function decl(p, { q, ...rest } = {}, ...more) {
            if (p) { var late = q; let inner = rest; return inner + more.length + late; }
            return late; // var hoists to the function scope
        }
        class K extends Object { static #s = 1; m(x) { return this.#s + x + (K ? 1 : 0); } static { hoisted += 1; } }
        const arrow = (u = hoisted) => u + block + fixed + def + alias;
        const named = function self(n) { return n ? self(n - 1) : 0; };
        for (const item of [a]) { console.log(item); }
        for (let i = 0; i < 1; i++) { continue; }
        for (const key in { d }) { void key; }
        try { throw new Error('e'); } catch ({ message }) { console.log(message); }
        try { throw new Error('e'); } catch ({ [fixed]: code = hoisted, ...meta }) { console.log(code, meta); }
        switch (b) { case 1: { const y = 1; void y; } break; default: break; }
        label: for (;;) { break label; }
        const obj = { a, decl, [fixed]: named, get g() { return arrow(); } };
        window.setTimeout(() => decl(K, obj), 0);
        export default class Default {}
        export { decl as exported };
        export { reexported } from './z.js';
    `, 'forms.js');
    assert.deepEqual(findings, []);
});

test('every browser module references only names it declares, imports, or the platform provides', () => {
    const findings = [];
    for (const file of moduleFiles()) {
        const rel = file.slice(WEB_DIR.length + 1);
        findings.push(...undeclaredReferences(readFileSync(file, 'utf8'), rel));
    }
    assert.deepEqual(
        findings, [],
        'Undeclared identifier(s) in web modules — a call to a deleted/renamed function '
        + 'throws ReferenceError at runtime on the path that reaches it:\n' + findings.join('\n'),
    );
});

// --- import linkage --------------------------------------------------------------

function moduleExports(ast) {
    const names = new Set();
    for (const node of ast.body) {
        if (node.type === 'ExportNamedDeclaration') {
            if (node.declaration) {
                const decl = node.declaration;
                if (decl.type === 'VariableDeclaration') for (const d of decl.declarations) for (const n of patternNames(d.id, [])) names.add(n);
                else if (decl.id) names.add(decl.id.name);
            }
            for (const spec of node.specifiers) names.add(spec.exported.type === 'Identifier' ? spec.exported.name : spec.exported.value);
        } else if (node.type === 'ExportDefaultDeclaration') {
            names.add('default');
        } else if (node.type === 'ExportAllDeclaration' && node.exported) {
            names.add(node.exported.type === 'Identifier' ? node.exported.name : node.exported.value);
        }
    }
    return names;
}

test('every relative named import and re-export resolves to an export of its target module', () => {
    const parsed = new Map();
    const parse = (file) => {
        if (!parsed.has(file)) {
            parsed.set(file, acorn.parse(readFileSync(file, 'utf8'), { ecmaVersion: 2024, sourceType: 'module', locations: true }));
        }
        return parsed.get(file);
    };
    const findings = [];
    for (const file of moduleFiles()) {
        const ast = parse(file);
        for (const node of ast.body) {
            const source = node.source && node.source.value;
            if (!source || !source.startsWith('.')) continue;
            const target = join(dirname(file), source);
            let targetExports;
            try { targetExports = moduleExports(parse(target)); } catch (err) {
                findings.push(`${file.slice(WEB_DIR.length + 1)}: cannot resolve '${source}' (${err.message})`);
                continue;
            }
            const wanted = [];
            if (node.type === 'ImportDeclaration') {
                for (const spec of node.specifiers) {
                    if (spec.type === 'ImportSpecifier') wanted.push(spec.imported.type === 'Identifier' ? spec.imported.name : spec.imported.value);
                    if (spec.type === 'ImportDefaultSpecifier') wanted.push('default');
                }
            } else if (node.type === 'ExportNamedDeclaration') {
                for (const spec of node.specifiers) wanted.push(spec.local.type === 'Identifier' ? spec.local.name : spec.local.value);
            }
            for (const name of wanted) {
                if (!targetExports.has(name)) findings.push(`${file.slice(WEB_DIR.length + 1)}: '${name}' is not exported by '${source}' (line ${node.loc.start.line})`);
            }
        }
    }
    assert.deepEqual(findings, [], 'Stale import(s) — a load-time SyntaxError that kills the whole page:\n' + findings.join('\n'));
});
