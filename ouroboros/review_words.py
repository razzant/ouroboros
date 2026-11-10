"""Source words made visible before a review's diff, never a verdict or gate.

Python AST and the installed JavaScript grammar identify literals in complete
pre/post images. Locations pair edits; equal words cancel even when moved. This
is deliberately inclusive evidence, not whole-program data-flow analysis:
unclassified prose stays visible. Dynamic expressions remain placeholders.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser

HEADING = "## Words to the owner: before → after"


@dataclass(frozen=True)
class Literal:
    path: str
    place: str
    line: int
    text: str


@dataclass(frozen=True)
class WordChange:
    before: Literal | None
    after: Literal | None


@dataclass(frozen=True)
class ReviewWords:
    changes: tuple[WordChange, ...] = ()
    gaps: tuple[str, ...] = ()

    def omission(self, source: str = "the diff's file/blob identities") -> str:
        return (HEADING + "\n\n⚠️ OMISSION NOTE: source words omitted to fit the review packet; "
                f"read {source}.") if self.changes or self.gaps else ""

    def render(self, limit: int, source: str = "the diff's file/blob identities") -> str:
        from ouroboros.utils import truncate_within_limit

        if not self.changes and not self.gaps:
            return ""
        lines = [HEADING, "", "Source literals; expressions are {…}. Evidence only. " + f"Full sources: {source}.", ""]
        for change in self.changes:
            location = change.after or change.before
            kind = "changed" if change.before and change.after else "added" if change.after else "removed"
            before = json.dumps(change.before.text, ensure_ascii=False) if change.before else "∅"
            after = json.dumps(change.after.text, ensure_ascii=False) if change.after else "∅"
            lines.append(f"- `{location.path}:{location.line}` · `{location.place}` ({kind}): {before} → {after}")
        lines.extend(f"- Extraction unavailable: {gap}; inspect the source." for gap in self.gaps)
        # Even a zero layout hint must disclose an omission. The enclosing
        # packet measures this irreducible heading/source marker too.
        if limit < len(self.omission(source)):
            return self.omission(source)
        return truncate_within_limit("\n".join(lines), limit)


def words_material(source: str = "the diff's file/blob identities") -> list[str]:
    """The section's framing belongs to the existing review contract fingerprint."""
    sample = ReviewWords((WordChange(Literal("⟨path⟩", "⟨place⟩", 1, "⟨before⟩"),
                                    Literal("⟨path⟩", "⟨place⟩", 1, "⟨after⟩")),), ("⟨gap⟩",))
    return [sample.render(section_budget(), source), sample.omission(source)]


_CHAT_TURN_KEYS = frozenset({"role", "content", "name", "tool_calls", "tool_call_id"})


def has_words(path: str) -> bool:
    """A source whose literals a person may read; a minified bundle is machine output, nobody's words."""
    return path.endswith((".py", ".js", ".html")) and not path.endswith(".min.js")


def section_budget(window_tokens: int | None = None) -> int:
    """Use the existing review inline share; the enclosing packet still has to fit."""
    from ouroboros.runtime_limits import REVIEW_GOVERNANCE_INLINE_SHARE
    from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET

    return int((REVIEW_PROMPT_TOKEN_BUDGET if window_tokens is None else window_tokens)
               * 4 * REVIEW_GOVERNANCE_INLINE_SHARE)


def _prose(text: str) -> bool:
    token = text.strip()
    return (any(c.isalpha() for c in token)
            and not re.match(r"^(?:\.?\.?/|https?://|[A-Za-z]:\\)", token)
            and not re.fullmatch(r"(?:--?[\w-]+|\S*[/\\]\S*|[\w.-]*\.[\w-]+|\w*_\w*|[A-Z]+\d+)", token)
            and not re.fullmatch(r"\s*(?:--[\w-]+|[a-z-]+)\s*:[^\n]+[;}]\s*", text))


def _label(text: str) -> bool:
    """Prose a person reads: several words, or one capitalised word (`Resume`), never a lower-case code."""
    token = text.strip()
    return _prose(token) and (any(c.isspace() for c in token) or token[:1].isupper())


def _python_text(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "{…}" for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _python_text(node.left), _python_text(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _python_literals(path: str, source: str) -> list[Literal]:
    found = []
    tree = ast.parse(source)
    # Names used as keys remain keys when their literals live in a constant,
    # loop or a local helper's argument instead of directly in the lookup.
    owners, scoped_keys = {}, defaultdict(set)
    machine_modules, machine_calls = {"re", "subprocess"}, {"git", "run_cmd", "_git_bytes"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            machine_modules.update(alias.asname or alias.name for alias in node.names if alias.name in {"re", "subprocess"})
        if isinstance(node, ast.ImportFrom) and node.module in {"re", "subprocess"}:
            machine_calls.update(alias.asname or alias.name for alias in node.names)

    def model_input(name):
        # A prompt's name identifies model context, never words in its prose; `messages` does only as a call's
        # argument (a dictionary of error messages for people is words), and a chat turn by its API shape.
        return bool(re.search(r"(?:^|_)prompt(?:_|$)", name, re.I))

    def machine_call(node):
        name = ast.unparse(node.func)
        return name in machine_calls or name.split(".")[0] in machine_modules

    def key_uses(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = node
        owners[node] = scope
        keys = (node.keys if isinstance(node, ast.Dict) else [node.key] if isinstance(node, ast.DictComp)
                else [node.slice] if isinstance(node, ast.Subscript) else [])
        if isinstance(node, ast.Call) and ast.unparse(node.func).rsplit(".", 1)[-1] in {"get", "pop", "setdefault", "tr", "t"}:
            keys = node.args[:1]
        inner = []
        if isinstance(node, ast.Call):
            if machine_call(node):
                inner = node.args
            inner = [*inner, *(kw.value for kw in node.keywords if kw.arg == "messages" or model_input(kw.arg or ""))]
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(model_input(ast.unparse(target)) for target in targets):
                inner = [node.value]
        # A lookup's key is its own name. Named argv, regexes and model messages are not captions just because
        # their literals are assigned separately from the consuming call, wherever the name sits inside it.
        names = [key for key in keys if isinstance(key, ast.Name)]
        names += [part for key in inner if key is not None for part in ast.walk(key) if isinstance(part, ast.Name)]
        scoped_keys[scope].update(name.id for name in names)
        for child in ast.iter_child_nodes(node):
            key_uses(child, scope)

    key_uses(tree, tree)
    # A module constant used as a key inside a function is still a key; local
    # variables of the same name in unrelated functions do not inherit that use.
    module_keys = set().union(*scoped_keys.values())
    key_args = {node.name: {i for i, arg in enumerate([*node.args.posonlyargs, *node.args.args]) if arg.arg in scoped_keys[node]}
                for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def walk(node, place=()):
        key_names = module_keys if owners.get(node) is tree else scoped_keys[owners.get(node)]
        if isinstance(node, (ast.Raise, ast.Assert, ast.Import, ast.ImportFrom, ast.Compare)):
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return  # docstrings and inert expression strings, never rendered
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if model_input(node.name):
                return
            if not isinstance(node, ast.ClassDef):
                for default in [*node.args.defaults, *node.args.kw_defaults]:
                    walk(default, (*place, node.name, "defaults"))
            for child in node.body:
                walk(child, (*place, node.name))
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            names = {part.id for part in ast.walk(target) if isinstance(part, ast.Name)}
            if names and (names <= key_names or "__all__" in names or any(model_input(name) for name in names)):
                return
            walk(node.value, (*place, ast.unparse(target)))
            return
        if isinstance(node, ast.Dict):
            keys = {key.value for key in node.keys if isinstance(key, ast.Constant)}
            if {"role", "content"} <= keys <= _CHAT_TURN_KEYS and len(keys) == len(node.keys):
                return  # a model's chat turn; a System row to the owner carries `text`, not `content`
            for key, value in zip(node.keys, node.values):
                walk(value, (*place, ast.unparse(key) if key else "**"))
            return
        if isinstance(node, ast.Subscript):
            walk(node.value, place)  # the index is a key
            return
        if isinstance(node, (ast.For, ast.comprehension)):
            if not isinstance(node.target, ast.Name) or node.target.id not in key_names:
                walk(node.iter, place)
            for child in (node.body + node.orelse if isinstance(node, ast.For) else node.ifs):
                walk(child, place)
            return
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if re.search(r"(?:^|\.)(?:\w*_)?(?:log|logger|logging)\.", name, re.I):
                return
            if name.endswith(("Error", "Exception")):
                return
            if machine_call(node):
                return
            if isinstance(node.func, ast.Attribute):
                walk(node.func.value, place)
            method = name.rsplit(".", 1)[-1]
            keys = ({0} if method in {"tr", "t", "get", "pop", "setdefault"} else
                    {1} if method in {"getattr", "hasattr", "setattr", "delattr"} else key_args.get(name, set()))
            if method in {"encode", "decode", "startswith", "endswith", "child_by_field_name", "get_parser"}:
                keys = set(range(len(node.args)))
            for index, arg in enumerate(node.args):
                if index not in keys:
                    walk(arg, place)
            for keyword in node.keywords:
                if keyword.arg not in {"encoding", "errors"} and not model_input(keyword.arg or ""):
                    walk(keyword.value, (*place, keyword.arg or "**"))
            return
        text = _python_text(node)
        if text is not None:
            if _label(text):
                found.append(Literal(path, ".".join(place) or "module", node.lineno, text))
            # Conditional literal words inside an interpolation are words too.
            for child in ast.walk(node):
                if isinstance(child, ast.FormattedValue):
                    walk(child.value, place)
            return
        if isinstance(node, ast.AST):
            for child in ast.iter_child_nodes(node):
                walk(child, place)

    walk(tree)
    return found


class _HTMLWords(HTMLParser):
    """Readable HTML text/attributes; scripts use the same JavaScript extractor."""

    def __init__(self, path, place=(), line=1):
        super().__init__(convert_charrefs=True)
        self.path, self.place, self.line = path, place, line
        self.stack, self.found = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        place = attrs.get("id") or tag
        if tag not in {"input", "img", "br", "hr", "meta", "link", "source", "wbr", "area", "embed"}:
            self.stack.append((tag, place))
        for name in ("title", "alt", "placeholder", "aria-label", "value"):
            if _label(attrs.get(name) or ""):
                self.add(attrs[name], (*self.place, place, name))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def add(self, text, place):
        self.found.append(Literal(self.path, ".".join(place) or "html", self.line + self.getpos()[0] - 1, text))

    def handle_data(self, data):
        tags = [tag for tag, _ in self.stack]
        if "style" in tags:
            return
        if "script" in tags:
            self.found.extend(_javascript_literals(self.path, data, self.line + self.getpos()[0] - 1))
        elif _label(data):
            self.add(data.strip(), (*self.place, *(place for _, place in self.stack)))


def _html_literals(path, source, place=(), line=1):
    parser = _HTMLWords(path, place, line)
    parser.feed(source)
    return parser.found


def _js_decode(text):
    def decode(match):
        escape = match[0][1:]
        if escape.startswith(("u", "x")):
            return chr(int(escape[1:].strip("{}"), 16))
        return {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "\n": ""}.get(escape, escape)
    return re.sub(r"\\(?:u\{[\da-fA-F]+\}|u[\da-fA-F]{4}|x[\da-fA-F]{2}|.)", decode, text)


def _javascript_literals(path: str, source: str, line=1) -> list[Literal]:
    from tree_sitter_language_pack import get_parser

    raw = source.encode("utf-8")
    root = get_parser("javascript").parse(raw).root_node
    if root.has_error:
        raise ValueError("JavaScript syntax could not be parsed")
    found = []

    def text(node):
        return raw[node.start_byte:node.end_byte].decode("utf-8") if node else ""

    def literal_text(node):
        if node.type == "binary_expression" and text(node.child_by_field_name("operator")) == "+":
            left = literal_text(node.child_by_field_name("left"))
            right = literal_text(node.child_by_field_name("right"))
            return left + right if left is not None and right is not None else None
        if node.type not in {"string", "template_string"}:
            return None
        value = raw[node.start_byte + 1:node.end_byte - 1]
        for child in reversed(node.named_children):
            if child.type == "template_substitution":
                start, end = child.start_byte - node.start_byte - 1, child.end_byte - node.start_byte - 1
                value = value[:start] + "{…}".encode() + value[end:]
        return _js_decode(value.decode("utf-8"))

    def substitutions(node, place):
        for child in node.named_children:
            if child.type == "template_substitution":
                for expression in child.named_children:
                    walk(expression, place)
            else:
                substitutions(child, place)

    def walk(node, place=()):
        kind = node.type
        if kind in {"comment", "regex", "import_statement", "throw_statement"}:
            return
        if kind == "binary_expression" and text(node.child_by_field_name("operator")) in {"===", "!==", "==", "!="}:
            return
        if kind in {"function_declaration", "method_definition", "class_declaration"}:
            place = (*place, text(node.child_by_field_name("name")))
        if kind in {"variable_declarator", "pair", "assignment_expression"}:
            key = node.child_by_field_name({"variable_declarator": "name", "pair": "key", "assignment_expression": "left"}[kind])
            name = text(key).strip("\"'")
            if (name in {"style", "className", "selector"} or ".style." in name or ".dataset." in name
                    or name.endswith((".cssText", ".className"))):
                return
            value = node.child_by_field_name("right" if kind == "assignment_expression" else "value")
            if value:
                walk(value, (*place, name))
            return
        if kind == "call_expression":
            function = node.child_by_field_name("function")
            name = text(function)
            args = node.child_by_field_name("arguments")
            if not args or re.search(r"(?:^|\.)(?:console|log|logger)\.", name):
                return
            if args.type == "template_string":
                if name != "css":
                    walk(args, place)
                return
            # A chained call's receiver owns callbacks and literals too:
            # slots.map(callback).join('') must visit the map and its callback.
            walk(function, place)
            if (name.rsplit(".", 1)[-1] in {"querySelector", "querySelectorAll", "matches", "closest", "insertRule"}
                    or ".classList." in name):
                return
            children = args.named_children
            if name.endswith("setAttribute") and children and text(children[0]).strip("\"'") in {"class", "style", "id"}:
                return
            if name.rsplit(".", 1)[-1] in {"tr", "t"}:
                children = children[1:]
            for child in children:
                walk(child, place)
            return
        value = literal_text(node)
        if value is not None:
            substitutions(node, place)
            if re.match(r"\s*[\w-]+=[\"']", value):
                value = f"<span {value}></span>"  # an HTML attribute fragment, not prose
            if re.search(r"</?[a-zA-Z][^>]*>", value):
                found.extend(_html_literals(path, value, place, line + node.start_point[0]))
            elif _label(value):
                found.append(Literal(path, ".".join(place) or "module", line + node.start_point[0], value))
            return
        for child in node.named_children:
            walk(child, place)

    walk(root)
    return found


def source_literals(path: str, source: str) -> list[Literal]:
    if "tests" in path.replace("\\", "/").split("/"):
        return []
    if path.endswith(".py"):
        return _python_literals(path, source)
    if path.endswith(".js"):
        return _javascript_literals(path, source)
    if path.endswith(".html"):
        return _html_literals(path, source)
    return []


def compare_sources(sources) -> ReviewWords:
    """Compare ``(path, old source, new source)`` rows, including moved words."""
    old, new, gaps = [], [], []
    for path, before, after in sources:
        try:
            left, right = source_literals(path, before), source_literals(path, after)
        except Exception as exc:
            gaps.append(f"{path} ({type(exc).__name__}: {exc})")
            continue
        old.extend(left)
        new.extend(right)
    # Equal words at the same place cancel first, then words moved by a refactor.
    for key in (lambda x: (x.path, x.place, x.text), lambda x: x.text):
        available = defaultdict(list)
        for i, item in enumerate(new):
            available[key(item)].append(i)
        kept, consumed = [], set()
        for item in old:
            indices = available[key(item)]
            if indices:
                consumed.add(indices.pop(0))
            else:
                kept.append(item)
        old, new = kept, [item for i, item in enumerate(new) if i not in consumed]
    by_place = defaultdict(list)
    for item in new:
        by_place[item.path, item.place].append(item)
    changes, paired = [], set()
    for item in old:
        matches = by_place[item.path, item.place]
        match = matches.pop(0) if matches else None
        changes.append(WordChange(item, match))
        if match:
            paired.add(id(match))
    changes.extend(WordChange(None, item) for item in new if id(item) not in paired)
    return ReviewWords(tuple(changes), tuple(gaps))


def _postimage(before: str, patch: str) -> str:
    """Apply unified hunks to the blob, also for an unstored worktree postimage."""
    original, out, cursor = re.findall(r"[^\n]*\n|[^\n]+$", before), [], 0
    hunks = list(re.finditer(r"(?m)^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", patch))
    for i, hunk in enumerate(hunks):
        start = int(hunk[1]) - (hunk[2] != "0")
        out.extend(original[cursor:start])
        cursor = start
        old_left, new_left = int(hunk[2] or 1), int(hunk[4] or 1)
        body = patch[hunk.end():hunks[i + 1].start() if i + 1 < len(hunks) else len(patch)]
        rows = re.findall(r"[^\n]*\n|[^\n]+$", body)
        for n, row in enumerate(rows):
            if not old_left and not new_left:
                break
            if row.startswith("\\ No newline"):
                continue
            value = row[1:]
            if n + 1 < len(rows) and rows[n + 1].startswith("\\ No newline"):
                value = value.removesuffix("\n")
            if row.startswith((" ", "-")):
                if cursor >= len(original) or original[cursor] != value:
                    raise ValueError("diff preimage does not match its blob")
                cursor += 1
                old_left -= 1
            if row.startswith((" ", "+")):
                out.append(value)
                new_left -= 1
            if row[:1] not in " +-" or old_left < 0 or new_left < 0:
                raise ValueError("diff hunk does not match its declared lines")
        if old_left or new_left:
            raise ValueError("diff hunk is incomplete")
    out.extend(original[cursor:])
    return "".join(out)


def words_for_diff(repo_dir, diff: str) -> ReviewWords:
    """Read the diff's blob identities, never HEAD or a possibly changed checkout (a work folder's by its own Git)."""
    from ouroboros import work_domain

    sources, gaps = [], []
    for block in re.split(r"(?m)(?=^diff --git )", diff):
        path_match = re.search(r"(?m)^\+\+\+ (.+)$", block)
        if not path_match:
            continue
        path = path_match[1].split("\t", 1)[0]
        try:
            if path == "/dev/null":
                old_path = re.search(r"(?m)^--- (.+)$", block)
                if not old_path:
                    raise ValueError("deleted file has no preimage path")
                path = old_path[1].split("\t", 1)[0]
            if path.startswith('"'):
                path = ast.literal_eval(path)
                try:
                    path = path.encode("latin1").decode("utf-8")
                except UnicodeError:
                    pass  # core.quotePath=false keeps non-ASCII characters literal
            path = path[2:]
            if not has_words(path):
                continue
            if "tests" in path.split("/") or "@@ " not in block:
                continue
            index = re.search(r"(?m)^index ([0-9a-f]+)\.\.([0-9a-f]+)(?: (\d+))?$", block)
            if not index:
                raise ValueError("diff has no blob identities")
            if index[3] == "120000":
                continue
            if set(index[1]) == {"0"}:
                before = ""
            else:
                before = work_domain.run(["git", "cat-file", "blob", index[1]], cwd=repo_dir, check=True,
                                         capture_output=True, timeout=10).stdout.decode("utf-8")
            sources.append((path, before, _postimage(before, block)))
        except (SyntaxError, ValueError, OSError, subprocess.SubprocessError) as exc:
            gaps.append(f"{path} ({type(exc).__name__})")
    result = compare_sources(sources)
    return ReviewWords(result.changes, (*result.gaps, *gaps))
