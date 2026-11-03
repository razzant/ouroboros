"""Materialize the FIXED editbench v2 workload: four real-file editing tasks.

The tasks are pinned data, not the product under test, and nothing imports them
at runtime. The four seeds ship beside this file as data, seeds_v2/<name>.py.txt:
the exact bytes of four Ouroboros modules on main on 2026-08-02 (SEED_COMMIT),
whose files reproduce every count the August v2 tables in README.md document.
Each seed's git blob id is recorded as provenance, never read: materializing
needs no git history. Prompts, oracle derivation and check scripts form ONE
identified version (WORKLOAD_VERSION); every seed and every materialized file is
pinned by sha256, so a run measures today's editing tools on the same tasks.

Tasks:
  t2_surgical   — review_state.py (1691 lines): 2 helper renames + 2 constant bumps.
  t3_blocks     — shell_parse.py (460 lines): replace 3 whole functions with given code.
  t4_move       — move collect_leading_env from shell_parse.py into git_shell_policy.py.
  t5_overhaul   — provider_models.py (423 lines): flip every eligible double-quoted
                  string literal to single quotes (208 literals on 119 lines).

Run: python devtools/benchmarks/editbench/make_fixtures_v2.py
It builds into a staging directory beside fixtures_v2/, validates the WHOLE set
(pins; the expected tree passes its oracle; the untouched workspace does not), and
only then swaps it in, keeping the replaced set as fixtures_v2.previous/. A failing
build leaves the published set as it was; an already valid set is left alone.
Changing a prompt, check, seed or derivation makes a new workload: bump
WORKLOAD_VERSION and re-pin from the digests the failure reports.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import tokenize

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "fixtures_v2"
SEEDS = HERE / "seeds_v2"
CHECKS = HERE / "checks"
CHECKER = HERE / "checker.py"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from checker import oracle_problems  # noqa: E402

WORKLOAD_VERSION = "v2.1"
SEED_COMMIT = "1b7e097e1e52e593967a9c3bab21f56b310a3157"  # main, 2026-08-02 06:13 +0300
# Path in SEED_COMMIT -> provenance and pin of seeds_v2/<name>.py.txt. "blob" is the
# git content address of those same bytes (`git cat-file blob <id>` shows them in a
# full clone); "sha256" is what a build checks.
SEED_FILES = {
    "ouroboros/review_state.py": {
        "blob": "b0249fd338533c135d9af66fd6d4b3bb41024bbf",
        "sha256": "7e04112a1b530034a0219a331d8a81f967511578a62f4fd97fbee04a2c79ac79",
    },
    "ouroboros/shell_parse.py": {
        "blob": "5acdfd6516fda1fb3c2436b49aca622c940f7532",
        "sha256": "56bec32e2d1794ff372c28c1604af2d5caf944b8a33e3e6e4675b7a9b6b0b2e6",
    },
    "ouroboros/git_shell_policy.py": {
        "blob": "405301314af38e136b571b08abe6bc9139aa8c77",
        "sha256": "be9d8553225f7607a0bee4cbe3f02528841199a532f05a3909ca889cbd1cfc3a",
    },
    "ouroboros/provider_models.py": {
        "blob": "660bbf2309ecdb417dfe3e6dbcdb7e4b5180c39d",
        "sha256": "cd73462c8b864d1eff4a007ac1fb6f6d32d068ce914bfc26bf93143ad5662f2f",
    },
}


# ---------------------------------------------------------------------------
# Task text. Part of the pinned workload: editing any of it is a version bump.
# ---------------------------------------------------------------------------

T2_PROMPT = """\
In review_state.py apply EXACTLY these four changes and nothing else (the file \
must stay byte-identical everywhere else — no reformatting, no other renames):
1. Rename the helper function `_stable_digest` to `_content_digest` — the def and every reference.
2. Rename the helper function `_max_iso_ts` to `_latest_iso_ts` — the def and every reference. \
Do NOT touch `_min_iso_ts`.
3. Change the constant `_MAX_RUN_HISTORY` from 10 to 25.
4. Change the constant `_REVIEW_ATTEMPT_TTL_SEC` from 1800 to 2400.
Do not create new files and do not commit."""

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

T3_PROMPT = f"""\
In shell_parse.py replace three functions with the new implementations given below. \
Each replacement must be verbatim exactly as given (same indentation), and the rest \
of the file must stay byte-identical. Do not create new files and do not commit.

Replace the function `strip_leading_env_assignments` with:

```python
{T3_NEW_STRIP}
```

Replace the function `shell_command_string` with:

```python
{T3_NEW_CMDSTR}
```

Replace the function `sudo_noninteractive_violation` with:

```python
{T3_NEW_SUDO}
```"""

T4_PROMPT = """\
Refactor: move the function `collect_leading_env` from ouroboros/shell_parse.py \
into ouroboros/git_shell_policy.py as a private helper. Exact spec — everything \
else must stay byte-identical:
1. In ouroboros/shell_parse.py: delete the entire `collect_leading_env` function \
(def line through its final `return`), leaving exactly two blank lines between the \
functions that surrounded it.
2. In ouroboros/git_shell_policy.py: remove `collect_leading_env,` from the \
`from ouroboros.shell_parse import (...)` list (delete that whole line).
3. In ouroboros/git_shell_policy.py: insert the function immediately BEFORE \
`def _git_subcommand_and_args(`, renamed to `_collect_leading_env` (body and \
docstring unchanged, byte-for-byte), with exactly two blank lines before it and \
exactly two blank lines after it.
4. Rename its single call site `collect_leading_env(segment)` to \
`_collect_leading_env(segment)`.
Do not touch ouroboros/utils.py or ouroboros/__init__.py. Do not create new files \
and do not commit."""

T5_PROMPT = """\
In provider_models.py convert string literals from double quotes to single quotes \
per this EXACT rule, and change nothing else:
- Applies to every SINGLE-LINE string literal written with double quotes, including \
prefixed ones (f"...", r"...", b"..." etc.).
- SKIP (leave unchanged): triple-quoted strings/docstrings, and any literal whose \
inner text contains a single quote ('), a double quote, or a backslash.
- Comments are not string literals — never edit comment text.
- The file's code semantics must be unchanged (same AST), only quote characters change.
There are a LOT of eligible literals — be systematic and cover the whole file. \
Do not create new files and do not commit."""

# Graded files, prompt and behavior check of each task. The runner reads these, so
# the task it sends is the task this module materialized and validated.
V2_TASKS: dict[str, dict] = {
    "t2_surgical": {"files": ["review_state.py"], "prompt": T2_PROMPT, "check": None},
    "t3_blocks": {"files": ["shell_parse.py"], "prompt": T3_PROMPT, "check": "t3_check.py"},
    "t4_move": {
        "files": ["ouroboros/shell_parse.py", "ouroboros/git_shell_policy.py",
                  "ouroboros/utils.py", "ouroboros/__init__.py"],
        "prompt": T4_PROMPT,
        "check": "t4_check.py",
    },
    "t5_overhaul": {"files": ["provider_models.py"], "prompt": T5_PROMPT, "check": "t5_check.py"},
}

# sha256 of every input that defines a task: its prompt, its check script and every
# materialized file (exact set). Generated data must match these byte for byte.
PINS: dict[str, dict] = {
    "t2_surgical": {
        "prompt": "1b46c4c0d7e51f23a42ebd0cd5d93a0c106c910543d4d09f0823d4ffc7c3257c",
        "check": None,
        "files": {
            "expected/review_state.py": "1e804098f4c2064b64ec8385e4e417e67def776c34f255217adaf6e1e5ad564b",
            "workspace/review_state.py": "7e04112a1b530034a0219a331d8a81f967511578a62f4fd97fbee04a2c79ac79",
        },
    },
    "t3_blocks": {
        "prompt": "2765f58708e16173618902b5139572d517f4e67cb5c5ca34a761ee303b55e1a3",
        "check": "4327115401071fe06449990d6acef7137d59913dc374b1e0845345c51cefd7e7",
        "files": {
            "expected/shell_parse.py": "2a2836bf535a214bfe545ea789a3710fbd94115fbec99bc2d1224902016ea7d9",
            "workspace/shell_parse.py": "56bec32e2d1794ff372c28c1604af2d5caf944b8a33e3e6e4675b7a9b6b0b2e6",
        },
    },
    "t4_move": {
        "prompt": "705350d698b697a06d1869ff2f79bafb52c976eb1c21076a84d3ee4365f92460",
        "check": "1c8868f558f796c8093db79530a6bbbe6bd4fa30b5d8af36bd0682004f7f2d73",
        "files": {
            "expected/ouroboros/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "expected/ouroboros/git_shell_policy.py": "e0a83be2e65aaf267b778dc74faeb0bea8a07a7c7e689356b6d36792c76e1f86",
            "expected/ouroboros/shell_parse.py": "dcbbfc77f90b98d40176ed9de09cffc5f32b1840cea002f5c7e2b40f1be40277",
            "expected/ouroboros/utils.py": "d4eb9a065d82aa650c0cb80135d0e3acbac7ca6ac2e30644baed3a3bc7a023f9",
            "workspace/ouroboros/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "workspace/ouroboros/git_shell_policy.py": "be9d8553225f7607a0bee4cbe3f02528841199a532f05a3909ca889cbd1cfc3a",
            "workspace/ouroboros/shell_parse.py": "56bec32e2d1794ff372c28c1604af2d5caf944b8a33e3e6e4675b7a9b6b0b2e6",
            "workspace/ouroboros/utils.py": "d4eb9a065d82aa650c0cb80135d0e3acbac7ca6ac2e30644baed3a3bc7a023f9",
        },
    },
    "t5_overhaul": {
        "prompt": "ec0725eb9fbaaf4473da742d122530eb92a8d5a921513b90bfcd38742c0b0022",
        "check": "75fa87f6811a0008279ea2b047b35f13ba72ca0f86b7c43832e12006c3b6c9d2",
        "files": {
            "expected/provider_models.py": "f35e2b29be6e29f019c889c59f09db63fb34019f1576908fb912126d9c005966",
            "workspace/provider_models.py": "cd73462c8b864d1eff4a007ac1fb6f6d32d068ce914bfc26bf93143ad5662f2f",
        },
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_argv(name: str) -> list[str] | None:
    """The behavior check of a v2 task, run with the candidate on PYTHONPATH."""
    check = V2_TASKS[name]["check"]
    return [sys.executable, str(CHECKS / check)] if check else None


def task_identity(name: str) -> dict:
    """Version and digest of one task, for a run manifest to attribute results to.

    The digest covers the task's inputs (graded files, check, every pin) and the
    shared grader, checker.py, so two runs with equal digests ran the same task
    and were scored by the same rules. It records the grader; it does not pin it.
    """
    spec = V2_TASKS[name]
    checker = _sha256(CHECKER.read_bytes())
    payload = {
        "workload": WORKLOAD_VERSION, "task": name, "seed_commit": SEED_COMMIT,
        "graded_files": spec["files"], "check": spec["check"], "pins": PINS[name],
        "checker_sha256": checker,
    }
    digest = _sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return {"version": WORKLOAD_VERSION, "digest": digest, "checker_sha256": checker}


# ---------------------------------------------------------------------------
# Oracle derivation (pure functions of the seed text)
# ---------------------------------------------------------------------------

def _count_replace(content: str, old: str, new: str, expect: int, label: str) -> str:
    got = content.count(old)
    if got != expect:
        raise ValueError(f"{label}: expected {expect} occurrences of {old!r}, found {got}")
    return content.replace(old, new)


def _compile_check(text: str, label: str) -> None:
    compile(text, label, "exec")  # in memory: no __pycache__ lands in the fixture


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
    raise ValueError(f"function {name} not found")


def _build_t2(seeds: dict[str, str]) -> dict[str, str]:
    src = seeds["ouroboros/review_state.py"]
    out = src
    out = _count_replace(out, "_stable_digest", "_content_digest", 6, "t2 rename1")
    out = _count_replace(out, "_max_iso_ts", "_latest_iso_ts", 3, "t2 rename2")
    out = _count_replace(out, "_MAX_RUN_HISTORY = 10", "_MAX_RUN_HISTORY = 25", 1, "t2 const1")
    out = _count_replace(out, "_REVIEW_ATTEMPT_TTL_SEC = 1800", "_REVIEW_ATTEMPT_TTL_SEC = 2400", 1, "t2 const2")
    _compile_check(out, "t2 expected")
    changed = sum(1 for a, b in zip(src.splitlines(), out.splitlines()) if a != b)
    print(f"t2_surgical: {changed} changed lines of {len(src.splitlines())}")
    return {"workspace/review_state.py": src, "expected/review_state.py": out}


def _build_t3(seeds: dict[str, str]) -> dict[str, str]:
    src = seeds["ouroboros/shell_parse.py"]
    out = src
    for name, new_block in T3_REPLACEMENTS.items():
        old_block = _function_block(out, name)
        if out.count(old_block) != 1:
            raise ValueError(f"t3: function block {name} is not unique")
        out = out.replace(old_block, new_block)
    _compile_check(out, "t3 expected")
    changed = len(out.splitlines()) - len(src.splitlines())
    print(f"t3_blocks: 3 functions replaced (net {changed:+d} lines)")
    return {"workspace/shell_parse.py": src, "expected/shell_parse.py": out}


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


def _build_t4(seeds: dict[str, str]) -> dict[str, str]:
    sp_src = seeds["ouroboros/shell_parse.py"]
    gsp_src = seeds["ouroboros/git_shell_policy.py"]
    # shell_parse: delete the collect_leading_env block and ONE of the surrounding
    # blank-line pairs (keep exactly two blank lines between its neighbors).
    fn_block = _function_block(sp_src, "collect_leading_env")
    needle = "\n\n\n" + fn_block + "\n\n\n"
    if sp_src.count(needle) != 1:
        raise ValueError("t4: collect_leading_env block (with blank-line frame) not unique")
    sp_out = sp_src.replace(needle, "\n\n\n")
    _compile_check(sp_out, "t4 expected shell_parse")
    # git_shell_policy: drop the import name, insert the private helper immediately
    # before _git_subcommand_and_args, rename the call site.
    gsp_out = _count_replace(gsp_src, "    collect_leading_env,\n", "", 1, "t4 import")
    private_block = fn_block.replace("def collect_leading_env(", "def _collect_leading_env(", 1)
    anchor = "def _git_subcommand_and_args("
    if gsp_out.count(anchor) != 1:
        raise ValueError("t4: anchor not unique")
    gsp_out = gsp_out.replace(anchor, private_block + "\n\n\n" + anchor, 1)
    gsp_out = _count_replace(
        gsp_out, "env_assigns, command = collect_leading_env(segment)",
        "env_assigns, command = _collect_leading_env(segment)", 1, "t4 callsite",
    )
    _compile_check(gsp_out, "t4 expected git_shell_policy")
    print("t4_move: built (move + rename + import trim)")
    files = {}
    for side, sp, gsp in (("workspace", sp_src, gsp_src), ("expected", sp_out, gsp_out)):
        files.update({
            f"{side}/ouroboros/__init__.py": "",
            f"{side}/ouroboros/utils.py": _T4_UTILS_STUB,
            f"{side}/ouroboros/shell_parse.py": sp,
            f"{side}/ouroboros/git_shell_policy.py": gsp,
        })
    return files


def _literal_spans(source: str):
    """(start, end) of every string literal, an f-string counted whole.

    Python 3.12+ tokenizes an f-string as FSTRING_START … FSTRING_END instead of
    one STRING token. Reading that whole span keeps the oracle identical on every
    interpreter and true to T5_PROMPT, which includes f"..." literals.
    """
    start_type = getattr(tokenize, "FSTRING_START", None)
    end_type = getattr(tokenize, "FSTRING_END", None)
    depth, opened = 0, None
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if start_type is not None and tok.type == start_type:
            depth += 1
            if depth == 1:
                opened = tok.start
        elif end_type is not None and tok.type == end_type:
            depth -= 1
            if depth == 0:
                yield opened, tok.end
        elif tok.type == tokenize.STRING and depth == 0:
            yield tok.start, tok.end


def _flip_quotes(source: str) -> tuple[str, int]:
    """Convert every eligible double-quoted string literal to single quotes.

    Eligible: single-line, starts with '\"' or a prefix+'\"' (f/r/b/u...), not
    triple-quoted, and the raw inner text contains no single quote, no double
    quote, and no backslash.
    """
    lines = source.splitlines(keepends=True)
    flipped = 0
    # Bottom-up, right to left, so earlier positions stay valid.
    for (srow, scol), (erow, ecol) in reversed(list(_literal_spans(source))):
        if srow != erow:
            continue
        line = lines[srow - 1]
        text = line[scol:ecol]
        head_len = len(text) - len(text.lstrip("fFrRbBuU"))
        prefix, body = text[:head_len], text[head_len:]
        if not body.startswith('"') or body.startswith('"""'):
            continue
        inner = body[1:-1]
        if "'" in inner or '"' in inner or "\\" in inner:
            continue
        lines[srow - 1] = line[:scol] + prefix + "'" + inner + "'" + line[ecol:]
        flipped += 1
    return "".join(lines), flipped


def _build_t5(seeds: dict[str, str]) -> dict[str, str]:
    src = seeds["ouroboros/provider_models.py"]
    out, flipped = _flip_quotes(src)
    _compile_check(out, "t5 expected")
    if ast.dump(ast.parse(src)) != ast.dump(ast.parse(out)):
        raise ValueError("t5: AST changed — quote flip is not semantics-preserving")
    changed = sum(1 for a, b in zip(src.splitlines(), out.splitlines()) if a != b)
    print(f"t5_overhaul: {flipped} strings flipped, {changed}/{len(src.splitlines())} lines changed")
    return {"workspace/provider_models.py": src, "expected/provider_models.py": out}


_BUILDERS = {"t2_surgical": _build_t2, "t3_blocks": _build_t3, "t4_move": _build_t4, "t5_overhaul": _build_t5}


# ---------------------------------------------------------------------------
# Validation and publication
# ---------------------------------------------------------------------------

def verify_task(name: str, root: pathlib.Path = OUT) -> list[str]:
    """Why ``root/<name>`` is not the pinned task; [] when it is. Reads only."""
    spec, pins = V2_TASKS[name], PINS[name]
    problems = []
    prompt_sha = _sha256(spec["prompt"].encode("utf-8"))
    if prompt_sha != pins["prompt"]:
        problems.append(f"prompt sha256 {prompt_sha} is not the pinned {pins['prompt']}")
    if spec["check"]:
        check = CHECKS / spec["check"]
        check_sha = _sha256(check.read_bytes()) if check.is_file() else "missing"
        if check_sha != pins["check"]:
            problems.append(f"checks/{spec['check']} sha256 {check_sha} is not the pinned {pins['check']}")
    task_root = pathlib.Path(root) / name
    found = ({p.relative_to(task_root).as_posix() for p in task_root.rglob("*") if p.is_file()}
             if task_root.is_dir() else set())
    pinned = set(pins["files"])
    problems += [f"missing {rel}" for rel in sorted(pinned - found)]
    problems += [f"unexpected {rel}" for rel in sorted(found - pinned)]
    for rel in sorted(pinned & found):
        actual = _sha256((task_root / rel).read_bytes())
        if actual != pins["files"][rel]:
            problems.append(f"{rel} sha256 {actual} is not the pinned {pins['files'][rel]}")
    return problems


def task_problems(name: str, root: pathlib.Path = OUT) -> list[str]:
    """Pins first; only a pinned set is worth running its oracle on."""
    task_root = pathlib.Path(root) / name
    return verify_task(name, root) or oracle_problems(
        task_root / "workspace", task_root / "expected", V2_TASKS[name]["files"],
        check_argv=check_argv(name), check_pythonpath=bool(V2_TASKS[name]["check"]),
    )


def seed_path(rel: str, seeds_dir: pathlib.Path = SEEDS) -> pathlib.Path:
    """The data file carrying seed ``rel``: seeds_v2/<name>.py.txt, never imported."""
    return pathlib.Path(seeds_dir) / (pathlib.PurePosixPath(rel).name + ".txt")


def _read_seeds(seeds_dir: pathlib.Path) -> tuple[dict[str, str], list[str]]:
    seeds, problems = {}, []
    for rel, pin in SEED_FILES.items():
        path = seed_path(rel, seeds_dir)
        data = path.read_bytes() if path.is_file() else None
        actual = _sha256(data) if data is not None else "missing"
        if actual != pin["sha256"]:
            problems.append(f"seed {rel}: {path} sha256 {actual} is not the pinned {pin['sha256']}")
            continue
        seeds[rel] = data.decode("utf-8")
    return seeds, problems


def _publish(staging: pathlib.Path, out: pathlib.Path) -> None:
    """Swap a validated staging tree in; the replaced tree becomes ``<out>.previous``.

    Two renames, not an atomic exchange. An OSError on the second puts the
    replaced tree back, but a crash between them leaves no ``<out>`` and the
    replaced tree in ``<out>.previous``: the runner then refuses (recorded
    fixture_preflight), and rerunning the generator publishes again and keeps
    that ``.previous``. A hard kill also leaves its ``<out>.staging-*``, which is
    safe to delete.
    """
    previous = out.with_name(out.name + ".previous")
    if out.exists():
        if previous.exists():
            shutil.rmtree(previous)  # one previous generation is kept, not a history
        os.replace(out, previous)
    try:
        os.replace(staging, out)
    except OSError:
        if previous.exists() and not out.exists():
            os.replace(previous, out)
        raise


def materialize(out: pathlib.Path = OUT, seeds_dir: pathlib.Path = SEEDS) -> list[str]:
    """Build, validate and publish the pinned workload; [] on success, else problems."""
    out = pathlib.Path(out)
    if not any(task_problems(name, out) for name in V2_TASKS):
        print(f"fixed workload {WORKLOAD_VERSION} already materialized and valid under {out}")
        return []
    seeds, problems = _read_seeds(seeds_dir)
    if problems:
        return problems
    built: dict[str, dict[str, str]] = {}
    for name, build in _BUILDERS.items():
        try:
            built[name] = build(seeds)
        except (ValueError, SyntaxError) as exc:
            problems.append(f"{name}: {exc}")
    if problems:
        return problems
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f"{out.name}.staging-", dir=out.parent))
    try:
        for name, files in built.items():
            for rel, text in files.items():
                path = staging / name / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text.encode("utf-8"))  # LF on every platform
        problems = [f"{name}: {p}" for name in V2_TASKS for p in task_problems(name, staging)]
        if problems:
            return problems
        _publish(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(f"fixed workload {WORKLOAD_VERSION} published under {out}")
    return []


def main() -> int:
    problems = materialize(OUT, SEEDS)
    for problem in problems:
        print(f"make_fixtures_v2: {problem}", file=sys.stderr)
    if problems:
        print(f"make_fixtures_v2: nothing published; {OUT} is unchanged", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
