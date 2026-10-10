"""Syntactic imports and their deliberately ambiguous filesystem candidates."""

import pytest
from tree_sitter_language_pack import get_parser

from ouroboros.code_import_candidates import import_candidates, import_specs, is_import_node


def _imports(language, source):
    tree = get_parser(language).parse(source.encode())
    assert not tree.root_node.has_error
    stack = [tree.root_node]
    rows = []
    while stack:
        node = stack.pop()
        if is_import_node(node, language):
            rows.append((node.type, import_specs(node, language)))
        stack.extend(reversed(node.named_children))
    return rows


@pytest.mark.parametrize("language,source,expected", [
    ("python", "import a.b as renamed, c\nfrom . import child as chosen\nfrom ..pkg import helper\n",
     [("import_statement", ["a.b", "c"]),
      ("import_from_statement", [".", ".child"]),
      ("import_from_statement", ["..pkg", "..pkg.helper"])]),
    ("typescript", 'import type { T } from "./util";\nexport { T as U } from "./util";\n',
     [("import_statement", ["./util"]), ("export_statement", ["./util"])]),
    ("typescript", 'import util = require("./util");',
     [("import_statement", ["./util"])]),
    ("javascript", 'import "./side"; const a = require("./util"); import("./lazy");',
     [("import_statement", ["./side"]), ("call_expression", ["./util"]),
      ("call_expression", ["./lazy"])]),
    ("javascript", 'obj.require("./x"); require(variable); import(`./${name}`);', []),
    ("go", 'package main\nimport ("fmt"; renamed "example.org/pkg")\n',
     [("import_spec", ["fmt"]), ("import_spec", ["example.org/pkg"])]),
    ("java", 'import pkg.Util;\nimport static pkg.Util.run;\n',
     [("import_declaration", ["pkg.Util"]),
      ("import_declaration", ["pkg.Util", "pkg.Util.run"])]),
    ("rust", 'use crate::util::{self, helper as chosen, nested::{A, B}};\n',
     [("use_declaration", ["crate::util", "crate::util::helper", "crate::util::nested::A",
                           "crate::util::nested::B"])]),
    ("rust", 'use foo::*;\nuse foo::{bar, *};\n',
     [("use_declaration", ["foo"]), ("use_declaration", ["foo", "foo::bar"])]),
])
def test_specifiers_come_from_syntax(language, source, expected):
    assert _imports(language, source) == expected


def test_relative_candidates_keep_extension_and_index_ambiguity():
    paths = ["pkg/util.js", "pkg/util.ts", "pkg/util/index.ts", "elsewhere/util.ts"]
    assert import_candidates("./util", "pkg/main.ts", paths, "typescript") == (
        "relative", ["pkg/util.js", "pkg/util.ts", "pkg/util/index.ts"])
    # Runtime .js can correspond to TypeScript source. Neither match wins.
    assert import_candidates("./util.js", "pkg/main.ts", paths, "typescript") == (
        "relative", ["pkg/util.js", "pkg/util.ts"])


def test_suffix_candidates_preserve_monorepo_ambiguity():
    paths = ["a/pkg/helper.py", "b/pkg/helper.py", "pkg/helper/__init__.py", "helper.py"]
    assert import_candidates("pkg.helper", "client.py", paths, "python") == (
        "suffix", ["a/pkg/helper.py", "b/pkg/helper.py", "pkg/helper/__init__.py"])


@pytest.mark.parametrize("prefix", ["", "src/", "packages/a/src/"])
def test_suffix_membership_uses_complete_path_components(prefix):
    hits = [prefix + path for path in (
        "util.py", "util.ts", "util/index.ts", "util/__init__.py", "util/mod.rs",
    )]
    misses = [prefix + path for path in (
        "notutil.py", "util.ts.bak", "util/child.py", "util/sub/index.ts", "utilindex.ts",
    )]
    assert import_candidates("util", "main.ts", hits + misses) == ("suffix", sorted(hits))


def test_relative_membership_does_not_match_deeper_suffixes():
    paths = ["pkg/util.ts", "src/pkg/util.ts", "pkg/deep/util.ts", "pkg/util/index.ts"]
    assert import_candidates("./util", "pkg/main.ts", iter(paths)) == (
        "relative", ["pkg/util.ts", "pkg/util/index.ts"])


def test_relative_python_joins_use_current_paths_only():
    importer = "pkg/consumer/main.py"
    assert import_candidates("..helper", importer, [], "python") == ("relative", [])
    assert import_candidates("..helper", importer, ["pkg/helper.py"], "python") == (
        "relative", ["pkg/helper.py"])
    assert import_candidates("..helper", importer, [], "python") == ("relative", [])
    assert import_candidates(".", "pkg/main.py", ["pkg/__init__.py"], "python") == (
        "relative", ["pkg/__init__.py"])
    assert import_candidates(".", "main.py", ["__init__.py"], "python") == (
        "relative", ["__init__.py"])


@pytest.mark.parametrize("spec,importer,language", [
    ("../../secret", "pkg/main.ts", "typescript"),
    ("...secret", "pkg/main.py", "python"),
    ("./secret", "../outside.ts", "typescript"),
    ("/secret", "main.ts", "typescript"),
    ("../secret", "main.ts", "typescript"),
])
def test_escaping_paths_never_become_suffix_candidates(spec, importer, language):
    assert import_candidates(spec, importer, ["secret.ts", "secret.py", "../secret.ts"], language)[1] == []


def test_dot_normalization_is_relative_and_case_sensitive():
    paths = ["pkg/util.ts", "pkg/Util.ts", "other/util.ts"]
    assert import_candidates("./sub/../util", "pkg/main.ts", paths) == (
        "relative", ["pkg/util.ts"])


def test_java_and_rust_are_conventional_path_candidates():
    assert import_candidates("pkg.Util", "src/Main.java", ["src/pkg/Util.java"], "java") == (
        "suffix", ["src/pkg/Util.java"])
    assert import_candidates("crate::util::helper", "src/main.rs", [
        "src/util.rs", "src/util/helper.rs", "src/util/mod.rs", "other/util.rs",
    ], "rust") == ("suffix", ["other/util.rs", "src/util.rs", "src/util/helper.rs", "src/util/mod.rs"])
    assert import_candidates("super::util::helper", "src/feature.rs", [
        "src/util.rs", "src/util/helper.rs", "elsewhere/util.rs",
    ], "rust") == ("relative", ["src/util.rs", "src/util/helper.rs"])


def test_unsupported_grammar_shape_has_no_invented_specifier():
    class Unknown:
        type = "unknown_import"
        text = b"import magic"
        named_children = ()

    assert not is_import_node(Unknown(), "unknown")
    assert import_specs(Unknown(), "unknown") == []
