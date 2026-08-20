"""Generate editbench v2 fixtures (workspace + expected) from REAL Ouroboros files.

Tasks:
  t2_surgical   — review_state.py (~1700 lines): 2 helper renames + 2 constant bumps.
  t3_blocks     — shell_parse.py: replace 3 whole functions with provided code.
  t4_move       — move collect_leading_env from shell_parse.py into git_shell_policy.py.
  t5_overhaul   — provider_models.py: flip every eligible double-quoted string literal
                  to single quotes (touches most lines).

Run once: python devtools/benchmarks/editbench/make_fixtures_v2.py
Validates every expected file (py_compile; token-level equivalence for t5) and
prints per-task change stats.
"""

from __future__ import annotations

import ast
import io
import pathlib
import py_compile
import shutil
import tokenize

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "fixtures_v2"


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _count_replace(content: str, old: str, new: str, expect: int, label: str) -> str:
    got = content.count(old)
    if got != expect:
        raise SystemExit(f"{label}: expected {expect} occurrences of {old!r}, found {got}")
    return content.replace(old, new)


def _compile_check(path: pathlib.Path) -> None:
    py_compile.compile(str(path), doraise=True)


# ---------------------------------------------------------------------------
# t2_surgical
# ---------------------------------------------------------------------------

def build_t2() -> None:
    # The v7 split moved the digest/timestamp bodies out of the review_state.py
    # facade into the records owner; the fixture follows the material it renames.
    src = _read("ouroboros/review_state_records.py")
    ws = OUT / "t2_surgical" / "workspace"
    exp = OUT / "t2_surgical" / "expected"
    _write(ws / "review_state_records.py", src)
    out = src
    out = _count_replace(out, "_stable_digest", "_content_digest", 3, "t2 rename1")
    out = _count_replace(out, "_max_iso_ts", "_latest_iso_ts", 1, "t2 rename2")
    out = _count_replace(out, "_MAX_RUN_HISTORY = 10", "_MAX_RUN_HISTORY = 25", 1, "t2 const1")
    out = _count_replace(out, "_REVIEW_ATTEMPT_TTL_SEC = 1800", "_REVIEW_ATTEMPT_TTL_SEC = 2400", 1, "t2 const2")
    _write(exp / "review_state_records.py", out)
    ast.parse(out)
    changed = sum(1 for a, b in zip(src.splitlines(), out.splitlines()) if a != b)
    print(f"t2_surgical: {changed} changed lines of {len(src.splitlines())}")


# ---------------------------------------------------------------------------
# t3_blocks
# ---------------------------------------------------------------------------

T3_NEW_STRIP = '''def strip_leading_env_assignments(argv: List[str]) -> List[str]:
    """Drop leading VAR=value assignment tokens from an argv list."""
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if "=" not in token or token.startswith("="):
            break
        key = token.partition("=")[0]
        if not key.replace("_", "").isalnum():
            break
        idx += 1
    return argv[idx:]'''

T3_NEW_CMDSTR = '''def shell_command_string(argv: List[str]) -> str:
    """Return the inline command payload of a ``sh -c ...`` style argv."""
    for idx, arg in enumerate(argv[1:], start=1):
        if arg == "-c" or (arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:]):
            return argv[idx + 1] if idx + 1 < len(argv) else ""
        if arg.startswith("--command="):
            return arg.partition("=")[2]
    return ""'''

T3_NEW_SUDO = '''def sudo_noninteractive_violation(argv: List[str]) -> bool:
    if argv and pathlib.PurePath(argv[0]).name.lower() in _SHELLS:
        inline = shell_command_string(argv)
        if inline:
            return sudo_noninteractive_violation(shell_argv(inline))
    for idx, token in enumerate(argv):
        command_name = pathlib.PurePath(token).name.lower()
        if command_name in {"sudoedit", "doas"}:
            return True
        if command_name != "sudo":
            continue
        has_noninteractive = False
        for option in _sudo_option_tokens(argv[idx + 1 :]):
            if option == "-S" or (option.startswith("-") and not option.startswith("--") and "S" in option[1:]):
                return True
            if option == "-n" or (option.startswith("-") and not option.startswith("--") and "n" in option[1:]):
                has_noninteractive = True
            if option.startswith("--non-interactive"):
                has_noninteractive = True
        if not has_noninteractive:
            return True
    return False'''

T3_REPLACEMENTS = {
    "strip_leading_env_assignments": T3_NEW_STRIP,
    "shell_command_string": T3_NEW_CMDSTR,
    "sudo_noninteractive_violation": T3_NEW_SUDO,
}


def _function_block(source: str, name: str) -> str:
    """Exact source text of a module-level function (no trailing blank lines)."""
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = node.lineno - 1
            if node.decorator_list:
                start = node.decorator_list[0].lineno - 1
            return "\n".join(lines[start:node.end_lineno])
    raise SystemExit(f"function {name} not found")


def build_t3() -> None:
    src = _read("ouroboros/shell_parse.py")
    ws = OUT / "t3_blocks" / "workspace"
    exp = OUT / "t3_blocks" / "expected"
    _write(ws / "shell_parse.py", src)
    out = src
    for name, new_block in T3_REPLACEMENTS.items():
        old_block = _function_block(out, name)
        if out.count(old_block) != 1:
            raise SystemExit(f"t3: function block {name} is not unique")
        out = out.replace(old_block, new_block)
    _write(exp / "shell_parse.py", out)
    _compile_check(exp / "shell_parse.py")
    changed = len(out.splitlines()) - len(src.splitlines())
    print(f"t3_blocks: 3 functions replaced (net {changed:+d} lines)")


# ---------------------------------------------------------------------------
# t4_move
# ---------------------------------------------------------------------------

_T4_UTILS_STUB = '''"""Minimal utils stub for the editbench t4 workspace (safe_relpath only)."""

from __future__ import annotations

import pathlib


def safe_relpath(path) -> str:
    """Best-effort display-relative path (stub of ouroboros.utils.safe_relpath)."""
    try:
        return str(pathlib.Path(path))
    except Exception:
        return str(path)
'''


def build_t4() -> None:
    sp_src = _read("ouroboros/shell_parse.py")
    gsp_src = _read("ouroboros/git_shell_policy.py")
    ws = OUT / "t4_move" / "workspace"
    exp = OUT / "t4_move" / "expected"
    for root in (ws, exp):
        _write(root / "ouroboros" / "__init__.py", "")
        _write(root / "ouroboros" / "utils.py", _T4_UTILS_STUB)
    _write(ws / "ouroboros" / "shell_parse.py", sp_src)
    _write(ws / "ouroboros" / "git_shell_policy.py", gsp_src)

    # expected shell_parse: delete the collect_leading_env block and ONE of the
    # surrounding blank-line pairs (keep exactly two blank lines between neighbors).
    fn_block = _function_block(sp_src, "collect_leading_env")
    needle = "\n\n\n" + fn_block + "\n\n\n"
    if sp_src.count(needle) != 1:
        raise SystemExit("t4: collect_leading_env block (with blank-line frame) not unique")
    sp_out = sp_src.replace(needle, "\n\n\n")
    _write(exp / "ouroboros" / "shell_parse.py", sp_out)
    _compile_check(exp / "ouroboros" / "shell_parse.py")

    # expected git_shell_policy: drop the import name, insert the private helper
    # immediately before _git_subcommand_and_args, rename the call site.
    gsp_out = _count_replace(gsp_src, "    collect_leading_env,\n", "", 1, "t4 import")
    private_block = fn_block.replace(
        "def collect_leading_env(", "def _collect_leading_env(", 1
    )
    anchor = "def _git_subcommand_and_args("
    if gsp_out.count(anchor) != 1:
        raise SystemExit("t4: anchor not unique")
    gsp_out = gsp_out.replace(anchor, private_block + "\n\n\n" + anchor, 1)
    gsp_out = _count_replace(
        gsp_out, "env_assigns, command = collect_leading_env(segment)",
        "env_assigns, command = _collect_leading_env(segment)", 1, "t4 callsite",
    )
    _write(exp / "ouroboros" / "git_shell_policy.py", gsp_out)
    _compile_check(exp / "ouroboros" / "git_shell_policy.py")
    print("t4_move: built (move + rename + import trim)")


# ---------------------------------------------------------------------------
# t5_overhaul
# ---------------------------------------------------------------------------

def _flip_quotes(source: str) -> tuple[str, int]:
    """Convert every eligible double-quoted STRING token to single quotes.

    Eligible: single-line, starts with '\"' or a prefix+'\"' (f/r/b/u...), not
    triple-quoted, and the raw inner text contains no single quote, no double
    quote, and no backslash.
    """
    lines = source.splitlines(keepends=True)
    flipped = 0
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    # Apply edits bottom-up so positions stay valid.
    for tok in reversed(tokens):
        if tok.type != tokenize.STRING:
            continue
        text = tok.string
        head_len = len(text) - len(text.lstrip("fFrRbBuU"))
        prefix, body = text[:head_len], text[head_len:]
        if not body.startswith('"') or body.startswith('"""'):
            continue
        inner = body[1:-1]
        if "'" in inner or '"' in inner or "\\" in inner:
            continue
        srow, scol = tok.start
        erow, ecol = tok.end
        if srow != erow:
            continue
        line = lines[srow - 1]
        new_text = prefix + "'" + inner + "'"
        lines[srow - 1] = line[:scol] + new_text + line[ecol:]
        flipped += 1
    return "".join(lines), flipped


def build_t5() -> None:
    src = _read("ouroboros/provider_models.py")
    ws = OUT / "t5_overhaul" / "workspace"
    exp = OUT / "t5_overhaul" / "expected"
    _write(ws / "provider_models.py", src)
    out, flipped = _flip_quotes(src)
    _write(exp / "provider_models.py", out)
    _compile_check(exp / "provider_models.py")
    # Semantic equivalence: identical AST payloads.
    if ast.dump(ast.parse(src)) != ast.dump(ast.parse(out)):
        raise SystemExit("t5: AST changed — quote flip is not semantics-preserving")
    changed = sum(1 for a, b in zip(src.splitlines(), out.splitlines()) if a != b)
    print(f"t5_overhaul: {flipped} strings flipped, {changed}/{len(src.splitlines())} lines changed")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    build_t2()
    build_t3()
    build_t4()
    build_t5()
    print(f"fixtures written under {OUT}")


if __name__ == "__main__":
    main()
