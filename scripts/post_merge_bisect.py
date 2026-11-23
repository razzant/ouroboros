#!/usr/bin/env python3
"""Post-merge attribution: scope the historical probes, admit fresh exports, bisect their answers.

The check (scripts/post_merge_check.py) owns Git execution, worktrees and test execution. This
module uses supplied readers and runners, and needs only the standard library.
"""
from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
import subprocess

MAX_PROBES = 16
ORDINARY = ("100644", "100755")  # a regular file's modes: a symlink or a submodule is no ordinary blob
HOOK = "conftest.py"  # the common hook: its code runs for every test of its directory
PROTECTED = "tests/test_repo_health_smoke.py"  # the one protected path: its change reaches its importers
LAUNCHERS = ("scripts/run_tests.py", "scripts/safe_test.py", "ouroboros/preflight_node.py")
PYTEST_CONFIGS = ("pyproject.toml", "pytest.ini", ".pytest.ini", "pytest.toml", ".pytest.toml", "setup.cfg", "tox.ini")
PYTEST_SOURCES = (":(glob)tests/**/*.py", HOOK, *LAUNCHERS, *PYTEST_CONFIGS,
                  *(f":(glob)tests/**/{name}" for name in PYTEST_CONFIGS))


def grep_names(tree: str, output: str) -> list[str] | None:
    """The relative names a `git grep` read yields, None where the read is not one we can place.

    Git writes the names of a tree as `<tree>:<path>` — the tree exactly as it was asked for — and
    separates them with NUL under `-z`; a search of the checkout writes plain relative paths. A name
    that does not carry the tree it was read from belongs to no tree we asked about: the whole read
    is unreadable (None), never a wrong name. A read with no names is an empty answer.
    """
    prefix = f"{tree}:" if tree else ""
    records = output.split("\0") if "\0" in output else output.splitlines()
    names = []
    for record in records:
        if not record:
            continue
        if prefix:
            if not record.startswith(prefix):
                return None
            record = record[len(prefix):]
        names.append(record)
    return names


def bisect(chain: list[str], ids, probe, *, max_probes: int = MAX_PROBES) -> list[dict]:
    """The first commit of `chain` at which each id fails; chain[0] is the base, chain[-1] the head.

    Every id fails at the head. The base is probed first: an id failing there is pre-existing.
    The others share bisection intervals, and a probe at a midpoint runs every unresolved id whose
    interval contains it, so ids broken by one merge cost one bisection. `probe(sha, ids)` answers
    {id: "fail" | "pass" | "error"}, or {id: ("error", why)} where the probe knows why that id could
    not be answered there. Past `max_probes` probes the rest stays unattributed.
    """
    ids, verdicts, pending, used = sorted(set(ids)), {}, {}, 0
    if len(chain) < 2:
        return [{"id": nodeid, "verdict": "pre-existing", "why": "no commit since the base"} for nodeid in ids]
    if ids:
        used += 1
        answers = probe(chain[0], ids)
        for nodeid in ids:
            answer, why = _answer(answers.get(nodeid, "error"))
            if answer == "fail":
                verdicts[nodeid] = {"verdict": "pre-existing", "why": f"already fails at the base {chain[0][:12]}"}
            elif answer == "pass":
                pending[nodeid] = (0, len(chain) - 1)
            else:
                verdicts[nodeid] = {"verdict": "unattributed",
                                    "why": f"the probe at {chain[0][:12]} could not answer {nodeid}"
                                          + (f": {why}" if why else "")}
    while pending:
        for nodeid, (good, bad) in list(pending.items()):
            if bad - good == 1:
                verdicts[nodeid] = {"verdict": "culprit", "culprit": chain[bad], "last_good": chain[good]}
                del pending[nodeid]
        if pending and used >= max_probes:
            for nodeid, (good, bad) in pending.items():
                verdicts[nodeid] = {"verdict": "unattributed", "why": f"the probe budget ({max_probes}) ran out; "
                                    f"first bad after {chain[good][:12]}, at or before {chain[bad][:12]}"}
            pending = {}
        if not pending:
            break
        good, bad = min(pending.values(), key=lambda interval: (interval[0] - interval[1], interval))
        middle = (good + bad) // 2
        targets = sorted(nodeid for nodeid, (low, high) in pending.items() if low < middle < high)
        used += 1
        answers = probe(chain[middle], targets)
        for nodeid in targets:
            low, high = pending[nodeid]
            answer, why = _answer(answers.get(nodeid, "error"))
            if answer in ("fail", "pass"):
                pending[nodeid] = (low, middle) if answer == "fail" else (middle, high)
            else:
                verdicts[nodeid] = {"verdict": "unattributed",
                                    "why": f"the probe at {chain[middle][:12]} could not answer {nodeid}"
                                          + (f": {why}" if why else "")}
                del pending[nodeid]
    return [{"id": nodeid, **verdicts[nodeid]} for nodeid in ids]


def probe_files(ids, size_prefix: str) -> list[tuple[str, str]]:
    """(id, file) of each probed id that names a file: a test id and a size id name their module."""
    pairs = []
    for nodeid in ids:
        name = nodeid[len(size_prefix):] if nodeid.startswith(size_prefix) else nodeid
        if (path := name.split("::", 1)[0]).endswith(".py"):
            pairs.append((nodeid, path))
    return pairs


def protected_projection(head: str, revision: str, git) -> tuple[str, str]:
    """The protected entry's state and reason to leave it off disk. No reason means same ordinary blob."""
    try:
        there = _tree_entry(git, revision)
        if there is None:
            return "absent", f"`{PROTECTED}` is gone at this revision"
        here = _tree_entry(git, head)
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unreadable", f"Git could not read the tree entry for `{PROTECTED}`"
    if _same(here, there):
        return "same", ""
    return "different", _why(here, there, PROTECTED)


def comparable(pairs, *, revision: str, protection, git):
    """Keep independent pytest IDs; exclude the family from size collection, not just its output.

    The family is the protected path and its direct consumers or unreadable candidates. A hook or
    other shared dependency (including a consumer of a direct importer) withholds all pytest IDs.
    This is a comparability check, not an execution sandbox.
    """
    state, reason = protection
    kept = [nodeid for nodeid, _ in pairs]
    if not kept or not reason:
        return kept, {}, set()
    family = {PROTECTED}
    if state == "unreadable":
        return [], dict.fromkeys(kept, reason), family
    try:
        direct = _consumers(git, revision, {PROTECTED})
        family.update(direct)
        shared = next((path for path in direct if not _test_module(path)), None)
        if direct and not shared:
            # No transitive import graph: a second hop suffices to recognize a shared dependency.
            shared = next(iter(_consumers(git, revision, direct)), None)
    except (OSError, subprocess.SubprocessError, ValueError):
        return [], dict.fromkeys(kept, f"{reason}; its consumers in the questioned tree could not be read"), family
    if shared:
        return [], dict.fromkeys(kept, f"{reason}; `{shared}` may load its family for other tests"), family
    excluded = {nodeid: (reason if path == PROTECTED else f"{reason}; `{path}` {direct[path]}")
                for nodeid, path in pairs if path in family}
    return [nodeid for nodeid in kept if nodeid not in excluded], excluded, family


def node_blocker(git, revision, reason, family) -> str:
    """The Node lane uses the Python launchers, not pytest's collector or hooks."""
    launcher = next((path for path in LAUNCHERS if path in family), None)
    if not launcher:
        tests = {path for path in family if path.endswith(".py")}
        try:
            launcher = next(iter(_consumers(git, revision, {PROTECTED, *tests}, scope=LAUNCHERS)), None)
        except (OSError, subprocess.SubprocessError, ValueError):
            return f"{reason}; could not check the Node launchers"
    return f"{reason}; Node launcher `{launcher}` may load its family" if launcher else ""


def _tree_entry(git, revision):
    """Mode, kind and object ID of the protected path: no match means absent, no blob read."""
    line = git("ls-tree", revision, "--", PROTECTED)
    if not line:
        return None
    mode, kind, oid = line.split("\t", 1)[0].split()
    return mode, kind, oid


def _test_module(path):
    name = PurePosixPath(path).name
    return path.startswith("tests/") and name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _consumers(git, revision, paths, *, scope=PYTEST_SOURCES):
    """Inspect only name matches in test/collection sources, never the protected blob itself.

    A literal mention proves no import. Pytest's configuration slots and unreadable source matches
    stay uncertain; an inventory or other data file is not a collection source on either hop.
    """
    stems = {PurePosixPath(path).stem for path in paths}
    exclude = {PROTECTED, *paths}
    try:
        listed = git("grep", "-l", "-z", "-F", *(arg for stem in sorted(stems) for arg in ("-e", stem)),
                     revision, "--", *scope,
                     *(f":(exclude,literal){path}" for path in sorted(exclude)))
    except subprocess.CalledProcessError as error:
        if error.returncode != 1:
            raise
        return {}  # no matching files, not a read failure
    if (names := grep_names(revision, listed)) is None:
        raise ValueError("unreadable consumer names")
    found = {}
    for path in sorted(set(names) - exclude):
        if not path.endswith(".py"):
            if PurePosixPath(path).name in PYTEST_CONFIGS:
                found[path] = "may register a consumer"
            continue
        try:
            if _may_load(git("show", f"{revision}:{path}", strip=False), stems):
                found[path] = "may load it"
        except (OSError, subprocess.SubprocessError, SyntaxError, ValueError):
            found[path] = "could not be read for dependencies"
    return found


def _may_load(source, stems):
    """Static import or loader/plugin reference, not an inventory string. No code is run."""
    def names_module(text):
        return stems.intersection(text.replace("\\", "/").replace("/", ".").split("."))

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if names_module(getattr(node, "module", "") or "") or any(names_module(alias.name) for alias in node.names):
                return True
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name not in ("__import__", "import_module", "spec_from_file_location", "run_module", "run_path"):
                continue
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is None or not any(isinstance(target, ast.Name) and target.id == "pytest_plugins" for target in targets):
                continue
            try:
                ast.literal_eval(node.value)
            except (SyntaxError, ValueError):
                return True  # A dynamic plugin declaration in an already matched file is not proof of isolation.
        else:
            continue
        if any(isinstance(value, ast.Constant) and isinstance(value.value, str) and names_module(value.value)
               for value in ast.walk(node)):
            return True
    return False


def fresh_pytest_export(directory, start, read):
    """Run and read a newly written terminal export; keep the wrapper exit as a separate fact."""
    result = Path(directory) / "results.json"

    def stamp():
        try:
            stat = result.stat()
        except FileNotFoundError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size

    try:
        previous = stamp()
    except OSError:
        return None
    code = start()
    try:
        current = stamp()
    except OSError:
        return None
    if current is None or current == previous:
        return None  # A reused directory's old export does not describe this invocation.
    session = read(directory)
    return {**session, "runner_exit": code} if session["final"] else None


def _answer(value):
    """(answer, why): a probe's answer for one id, with the reason it carries when it has one."""
    if isinstance(value, tuple):
        return value[0], (value[1] if len(value) > 1 else "")
    return value, ""


def _same(here, there) -> bool:
    """Whether both reads name the same ordinary blob: the file is unchanged between the two trees."""
    return bool(here and there and here[0] in ORDINARY and there[0] in ORDINARY
                and here[1] == "blob" and there[1] == "blob" and here[2] == there[2])


def _why(here, there, path) -> str:
    """Why a file is not the head's: names and Git metadata, never the bytes of a historical tree."""
    if not here:
        return f"`{path}` is not in the head"
    if here[0] not in ORDINARY or there[0] not in ORDINARY or here[1] != "blob" or there[1] != "blob":
        return f"`{path}` is not an ordinary file in both trees"
    return f"`{path}` differs from the head (blob {here[2][:12]} vs {there[2][:12]})"
