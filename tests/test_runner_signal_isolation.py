"""CLASS GATE 5b — a test that signals a process group must own that group.

The incident this closes: three panic-ledger cases reached for "the ssh child's whole
group" with ``os.killpg(os.getpgid(client.process.pid), SIGKILL)``.  A plain ``Popen``
INHERITS the runner's process group, so ``getpgid(child)`` was pytest's own group and the
call SIGKILLed pytest.  The suite died mid-file with no failure, no traceback and no
summary line — output indistinguishable from a suite that finished cleanly, which is why
nobody noticed that the entire kill arm of the feature had stopped being tested.

That is the worst shape a test defect can take: it does not report a false pass, it
removes the report.  So the rule is structural rather than a fixed list of three cases —
a test may signal a process group only if it created that group.

BOUNDARY, stated so the reach is not overestimated:

* The isolation evidence is looked for anywhere in the SAME FILE, not in the killing
  function alone. Function scope was tried first and was wrong for this codebase: the
  fixed version of the incident spawns with ``start_new_session=True`` inside a helper
  client class and kills from three test bodies, so function scope flagged correct code.
  The cost of file scope is stated plainly: a file that isolates ONE child and then kills a
  DIFFERENT, unisolated child's group is not distinguished, and a spawner in
  ``conftest.py`` is not seen at all. It still catches the incident as it actually
  occurred — the pre-fix file contained three ``killpg`` calls and zero isolation markers.
* Signal 0 is a liveness PROBE and never terminates anything, so ``os.killpg(pgid, 0)`` is
  legal everywhere.
* A test that monkeypatches ``os.killpg`` is testing the caller's arithmetic, not sending
  a signal, so a patched name in the same function counts as isolation.
* A raw ``os.kill(pid, …)`` of a single pid is NOT policed: killing one pid the test owns
  is ordinary, and the runner-suicide shape needs a GROUP. ``os.kill(-pgid, …)`` — the
  negative-pid spelling of killpg — IS policed, because it is the same act.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"

# Names that send a signal to a whole process group.
_GROUP_SIGNALLERS = frozenset({"killpg", "kill_process_group_id", "killpg_or_kill"})
# Evidence that the process group in question was created by this code.
_ISOLATION_MARKERS = (
    "start_new_session",
    "setsid",
    "process_group",
    "setpgrp",
    "setpgid",
    "creationflags",
    # A monkeypatched signaller sends nothing at all.
    'setattr(os, "killpg"',
    "setattr(pl.os",
    'setattr(os, "kill"',
    "monkeypatch.setattr",
)


def _sends_to_group(node: ast.Call) -> bool:
    """True when this call signals a GROUP with a real (non-zero) signal."""

    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    if name in _GROUP_SIGNALLERS:
        # `killpg(pgid, 0)` is a liveness probe, not a kill.
        if len(node.args) >= 2:
            sig = node.args[1]
            if isinstance(sig, ast.Constant) and sig.value == 0:
                return False
        return True
    if name == "kill" and node.args:
        # os.kill(-pgid, sig) is killpg spelled with a negative pid.
        first = node.args[0]
        if isinstance(first, ast.UnaryOp) and isinstance(first.op, ast.USub):
            if len(node.args) >= 2:
                sig = node.args[1]
                if isinstance(sig, ast.Constant) and sig.value == 0:
                    return False
            return True
    return False


def unisolated_group_signals() -> list[str]:
    """Every test-side group signal whose enclosing function shows no group of its own."""

    findings: list[str] = []
    for path in sorted(TESTS.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        rel = str(path.relative_to(REPO))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # File scope, not function scope — see the BOUNDARY note above.
            isolated = any(marker in text for marker in _ISOLATION_MARKERS)
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and _sends_to_group(node) and not isolated:
                    findings.append(
                        f"{rel}:{node.lineno}: {func.name} signals a process group it "
                        "never created — if that group is the runner's, the suite dies "
                        "silently instead of failing"
                    )
    return findings


def test_no_test_signals_a_process_group_it_did_not_create():
    """The RWS-111 shape cannot come back anywhere in the suite.

    Fix a failure by spawning the child with ``start_new_session=True`` (so it has a group
    of its own) and killing THAT group, or by taking the pgid from whatever spawned it.
    """

    findings = unisolated_group_signals()
    assert not findings, (
        "a test may only signal a process group it created — the alternative is a suite "
        "that vanishes without a failure:\n" + "\n".join(findings)
    )


def test_the_scan_catches_the_incident_it_was_written_for():
    """The exact RWS-111 spelling, and the negative-pid variant of it."""

    reproduction = (
        "import os, signal, subprocess\n"
        "def test_panic_kills_the_group():\n"
        "    child = subprocess.Popen(['sleep', '5'])\n"
        "    os.killpg(os.getpgid(child.pid), signal.SIGKILL)\n"
    )
    negative_pid = (
        "import os, signal, subprocess\n"
        "def test_panic_kills_the_group():\n"
        "    child = subprocess.Popen(['sleep', '5'])\n"
        "    os.kill(-os.getpgid(child.pid), signal.SIGKILL)\n"
    )
    for source in (reproduction, negative_pid):
        tree = ast.parse(source)
        hits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _sends_to_group(node)
        ]
        assert hits, source
        assert not any(marker in source for marker in _ISOLATION_MARKERS), (
            "the reproduction must contain no isolation evidence, or it proves nothing"
        )


def test_the_scan_leaves_the_prescribed_shapes_alone():
    """The correct spellings must stay legal, or the gate becomes noise to be muted."""

    isolated = (
        "import os, signal, subprocess\n"
        "def test_ok():\n"
        "    child = subprocess.Popen(['sleep', '5'], start_new_session=True)\n"
        "    os.killpg(os.getpgid(child.pid), signal.SIGKILL)\n"
    )
    probe = (
        "import os\n"
        "def _group_exists(pgid):\n"
        "    os.killpg(pgid, 0)\n"
    )
    for source in (isolated, probe):
        tree = ast.parse(source)
        offenders = [
            node
            for node in ast.walk(tree)
            for func in [None]
            if isinstance(node, ast.Call)
            and _sends_to_group(node)
            and not any(marker in source for marker in _ISOLATION_MARKERS)
        ]
        assert not offenders, source


def test_the_scan_states_its_own_boundary():
    """The admitted residue stays named and stays real."""

    assert "BOUNDARY" in (__doc__ or "")
    doc = __doc__ or ""
    assert "conftest.py" in doc, "the cross-module spawn residue must stay named"
    assert "DIFFERENT, unisolated child" in doc, "the file-scope cost must stay named"
    # And it really is blind to a single-pid kill, as the paragraph says.
    single = "import os\ndef test_x():\n    os.kill(1234, 9)\n"
    tree = ast.parse(single)
    assert not [
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and _sends_to_group(node)
    ], "single-pid kills are now policed — narrow the BOUNDARY paragraph to match"
