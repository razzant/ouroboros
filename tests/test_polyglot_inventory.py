"""Polyglot structural extraction (v6.33.0 WS3).

One generic tree-sitter path covers every language without a bespoke extractor
(Go/Rust/Java/Ruby/C/...). Python stays on the canonical stdlib ast. When
tree-sitter is unavailable the file is marked structural_unavailable VISIBLY,
never silently regex-guessed.
"""

from __future__ import annotations

import pathlib

import ouroboros.code_intelligence as ci
from ouroboros.code_intelligence import _file_fact


def _fact(tmp_path: pathlib.Path, name: str, body: str):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return _file_fact(tmp_path, p)


def test_go_symbols_and_calls(tmp_path):
    ff = _fact(tmp_path, "srv.go",
               "package main\nfunc Add(a int) int { return helper(a) }\n"
               "type Server struct{}\nfunc (s *Server) Start() { Add(1) }\n")
    assert ff.language == "go"
    kinds = {(s.kind, s.name) for s in ff.symbols}
    assert ("function", "Add") in kinds
    assert ("type", "Server") in kinds
    assert ("method", "Start") in kinds
    assert "helper" in {c.name for c in ff.call_sites}
    assert ff.disposition == "indexed"


def test_rust_symbols(tmp_path):
    ff = _fact(tmp_path, "lib.rs",
               "fn main() {}\nstruct P { x: i32 }\nimpl P { fn new() -> Self { todo!() } }\ntrait T { fn go(&self); }\n")
    kinds = {(s.kind, s.name) for s in ff.symbols}
    assert ("function", "main") in kinds
    assert ("struct", "P") in kinds
    assert ("trait", "T") in kinds


def test_java_symbols(tmp_path):
    ff = _fact(tmp_path, "App.java", "class Foo {\n  int bar() { return baz(); }\n}\ninterface Baz { void go(); }\n")
    kinds = {(s.kind, s.name) for s in ff.symbols}
    assert ("class", "Foo") in kinds
    assert ("method", "bar") in kinds
    assert ("interface", "Baz") in kinds


def test_javascript_uses_treesitter_and_keeps_imports(tmp_path):
    ff = _fact(tmp_path, "app.js",
               "import {x} from './m';\nfunction hello(){ return world(); }\nclass Widget {}\n")
    kinds = {(s.kind, s.name) for s in ff.symbols}
    assert ("function", "hello") in kinds
    assert ("class", "Widget") in kinds
    assert "world" in {c.name for c in ff.call_sites}
    assert "./m" in ff.imports  # extract_js_imports still feeds JS imports


def test_python_stays_on_ast(tmp_path):
    ff = _fact(tmp_path, "m.py", "async def go():\n    pass\nCONST = 1\n")
    by_name = {s.name: s.kind for s in ff.symbols}
    # async_function kind is an ast-only distinction tree-sitter would not produce.
    assert by_name.get("go") == "async_function"
    assert by_name.get("CONST") == "constant"


def test_visible_fallback_when_treesitter_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_ts_parser", lambda grammar: None)
    ff = _fact(tmp_path, "srv.go", "package main\nfunc Add() {}\n")
    assert ff.disposition == "structural_unavailable:go"
    assert ff.symbols == []  # NOT a silent regex guess


def test_non_code_file_discloses_unavailable_outline(tmp_path):
    ff = _fact(tmp_path, "notes.md", "# title\n")
    assert ff.disposition == "structural_unavailable:markdown"
    assert ff.symbols == []


def test_member_call_extracts_callee_not_receiver(tmp_path):
    """Regression (adversarial r1): a member/method call must record the FINAL
    identifier (the callee), not the receiver/namespace — obj.doThing() -> doThing,
    fmt.Sprintf() -> Sprintf — matching the Python ast path and the former JS regex."""
    go = _fact(tmp_path, "m.go",
               "package main\nimport \"fmt\"\nfunc F() { fmt.Sprintf(\"x\") }\n")
    go_calls = {c.name for c in go.call_sites}
    assert "Sprintf" in go_calls, go_calls
    assert "fmt" not in go_calls, go_calls

    js = _fact(tmp_path, "m.js",
               "function f(obj){ obj.doThing(); console.log('x'); }\n")
    js_calls = {c.name for c in js.call_sites}
    assert "doThing" in js_calls, js_calls
    assert "log" in js_calls, js_calls
    assert "console" not in js_calls and "obj" not in js_calls, js_calls


def test_exported_const_is_symbol_with_any_initializer_inside_namespace(tmp_path):
    ff = _fact(tmp_path, "util.ts", """export namespace util {
  export const objectKeys = typeof Object.keys === 'function'
    ? (obj: object) => Object.keys(obj)
    : (obj: object) => [];
  export let mutable = 1;
  export var older = 2;
}
""")
    symbols = {symbol.name: symbol for symbol in ff.symbols}
    assert symbols["objectKeys"].kind == "constant"
    assert symbols["objectKeys"].line_start == 2
    assert symbols["objectKeys"].line_end == 4
    assert symbols["mutable"].kind == "variable"
    assert symbols["older"].kind == "variable"
    assert any(call.name == "keys" and call.enclosing == "objectKeys" for call in ff.call_sites)
    assert "typeof" not in symbols


def test_name_fields_on_calls_or_properties_are_not_declarations(tmp_path):
    ff = _fact(tmp_path, "m.js", "const obj = { property: 1 };\nobj.run();\nobj.member = 2;\n")
    symbols = {symbol.name for symbol in ff.symbols}
    assert symbols == {"obj"}


def test_unknown_grammar_keeps_file_available_without_inventing_symbols(tmp_path):
    ff = _fact(tmp_path, "m.unknown", "export const objectKeys = f();\n")
    assert ff.language == "unknown"
    assert ff.disposition == "structural_unavailable:unknown"
    assert ff.symbols == []
    assert ff.call_sites == []
