"""OPEN-6: descriptor ownership across panic teardown (PR 79 regression).

The PR 79 CI symptom was that the SSH probe AFTER `broker.panic()` failed with
`OSError: [Errno 9] Bad file descriptor` while Python closed `subprocess.stdout`.
The cause is ownership, not timing, so it reproduces deterministically with plain
subprocesses and needs no Docker:

`os.close(stream.fileno())` closes the descriptor behind the back of the object
that owns it — the `FileIO` that `subprocess` created with `closefd=True`. That
object still believes it is open. The OS reuses the descriptor NUMBER for the next
child immediately, and when the stale object is finalized it closes a descriptor
that now belongs to a LIVE process. The next legitimate close then fails.

`test_the_donor_teardown_reproduces_errno_9` pins the mechanism (it would pass
even with the bug fixed elsewhere — it is the evidence, not the guard).
`test_release_child_streams_*` are the guards: they pin what the fix must keep
true, including that it does NOT flush, which is what made `os.close` tempting in
the fork path in the first place.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.remote_ssh import OpenSSHExecdTransport, _release_child_streams

pytestmark = pytest.mark.serial

_SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


def _spawn(**kwargs):
    return subprocess.Popen(
        _SLEEPER,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )


def _fds(process):
    return [process.stdin.fileno(), process.stdout.fileno(), process.stderr.fileno()]


def _kill(process):
    process.kill()
    process.wait(timeout=5)


def test_the_donor_teardown_reproduces_errno_9():
    """The evidence: stealing the fd breaks the NEXT child, not this one."""

    first = _spawn(bufsize=0)
    stolen = _fds(first)
    _kill(first)
    # The donor's `_panic_discard_process`, verbatim in shape.
    for stream in (first.stdin, first.stdout, first.stderr):
        try:
            os.close(stream.fileno())
        except (OSError, ValueError):
            pass
    stale = (first.stdin, first.stdout, first.stderr)
    assert all(not stream.closed for stream in stale), (
        "the owner objects must still believe they are open — that IS the bug"
    )

    second = _spawn(bufsize=0)
    try:
        assert set(stolen) & set(_fds(second)), "the OS did not reuse a descriptor"
        # Finalizing the stale objects closes the LIVE process's descriptors.
        for stream in stale:
            try:
                stream.close()
            except OSError:
                pass
        _kill(second)
        # Close ALL THREE, collecting errors instead of stopping at the first: a
        # stream left unclosed here still OWNS a descriptor number it no longer
        # holds, and finalizing it during a LATER test closes that number out from
        # under whatever process owns it by then. The evidence test must not be the
        # bug it documents.
        errnos = []
        for stream in (second.stdout, second.stderr, second.stdin):
            try:
                stream.close()
            except OSError as exc:
                errnos.append(exc.errno)
        assert 9 in errnos, errnos
    finally:
        if second.poll() is None:
            _kill(second)


def test_release_child_streams_leaves_a_reused_descriptor_usable():
    """The guard: after the fix, the next child closes cleanly."""

    first = _spawn(bufsize=0)
    released = _fds(first)
    _kill(first)
    _release_child_streams(first)
    stale = (first.stdin, first.stdout, first.stderr)
    assert all(stream.closed for stream in stale), (
        "the descriptor must be closed BY ITS OWNER, so nothing closes it twice"
    )

    second = _spawn(bufsize=0)
    try:
        assert set(released) & set(_fds(second)), "the OS did not reuse a descriptor"
        for stream in stale:
            stream.close()  # idempotent now
        _kill(second)
        # The whole point: no Errno 9 here.
        second.stdout.close()
        second.stderr.close()
        second.stdin.close()
    finally:
        if second.poll() is None:
            _kill(second)


def test_release_child_streams_does_not_flush_buffered_bytes(tmp_path):
    """Why the fix closes the RAW stream rather than the wrapper.

    A forked child must never push buffered bytes into the parent's SSH pipe, and
    panic must never wait on a flush. Closing the raw stream satisfies both.
    """

    sink = tmp_path / "received.bin"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,pathlib;pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
            str(sink),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        child.stdin.write(b"MUST-NOT-REACH-THE-PARENT-PIPE")

        _release_child_streams(child)

        child.wait(timeout=10)
        assert sink.read_bytes() == b""
    finally:
        if child.poll() is None:
            _kill(child)


def test_release_child_streams_is_immediate_with_a_full_buffer():
    """Panic may not block: releasing never waits on a flush."""

    child = _spawn()
    try:
        try:
            child.stdin.write(b"x" * 2_000_000)
        except (BrokenPipeError, OSError):
            pass
        _kill(child)

        started = time.monotonic()
        _release_child_streams(child)
        assert time.monotonic() - started < 2
    finally:
        if child.poll() is None:
            _kill(child)


def test_release_child_streams_tolerates_absent_and_dead_streams():
    assert _release_child_streams(None) is None
    process = SimpleNamespace(stdin=None, stdout=None, stderr=None)
    assert _release_child_streams(process) is None


def test_transport_panic_then_detach_leaves_a_new_child_usable(tmp_path):
    """End to end on the transport: panic, then the NEXT probe must be clean.

    This is the PR 79 sequence with the SSH parts faked out — panic tears down a
    real child, then a fresh child is spawned onto the freed descriptor numbers
    and must close without Errno 9.
    """

    transport = object.__new__(OpenSSHExecdTransport)
    transport.request = SimpleNamespace(
        connection={"id": "connection-1", "ssh_alias": "build"},
        server_generation="generation-1",
        drive_root=tmp_path,
    )
    transport._stop = threading.Event()
    transport._send_lock = threading.RLock()
    transport._sequence = 0
    transport._helper_lock = threading.RLock()
    # Its own process group, exactly as `spawn_supervised(new_process_group=True)`
    # gives the real transport — panic kills the GROUP, and a child sharing the
    # test runner's group would make this test kill pytest.
    child = _spawn(bufsize=0, start_new_session=True)
    released = _fds(child)
    transport._process = child
    transport._helper_process = None

    transport.panic()

    deadline = time.monotonic() + 5
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child.poll() is not None, "panic must kill the child"
    probe = _spawn(bufsize=0)
    try:
        assert set(released) & set(_fds(probe)), "the OS did not reuse a descriptor"
        # Finalize the torn-down child's stream objects while the probe is live:
        # this is the moment the donor's teardown clobbered the probe's pipes.
        for stream in (child.stdin, child.stdout, child.stderr):
            try:
                stream.close()
            except OSError:
                pass
        _kill(probe)
        probe.stdout.close()
        probe.stderr.close()
        probe.stdin.close()
    finally:
        if probe.poll() is None:
            _kill(probe)


def test_panic_does_not_wait_for_a_busy_send():
    """The whole reason line 725 is `acquire(blocking=False)` — and it was untested.

    Panic's best-effort `panic` frame is written under `_send_lock`, and the acquire is
    non-blocking so that a backpressured send in another thread cannot hold the emergency
    stop behind it. `_send_lock` is an RLock, which grants the panicking thread no
    reentrancy relief, and the acquire sits in the `if` of the `try` whose `finally`
    discards the child — so a blocking acquire would not merely DELAY teardown, it would
    skip it entirely, leaving the child alive.

    Nothing exercised that. The one test that constructs `_send_lock` never contends it,
    and the structurally similar `test_panic_does_not_wait_for_the_broker_state_lock`
    holds the BROKER's state lock over fake transports, so it cannot reach this one. The
    lane named for panic's local half never called `panic()` at all.
    """

    transport = object.__new__(OpenSSHExecdTransport)
    transport.request = SimpleNamespace(
        connection={"id": "connection-1", "ssh_alias": "build"},
        server_generation="generation-1",
        drive_root=None,
    )
    transport._stop = threading.Event()
    transport._send_lock = threading.RLock()
    transport._sequence = 0
    transport._helper_lock = threading.RLock()
    child = _spawn(bufsize=0, start_new_session=True)
    transport._process = child
    transport._helper_process = None

    holding = threading.Event()
    release = threading.Event()

    def hog():
        with transport._send_lock:
            holding.set()
            release.wait(10)

    sender = threading.Thread(target=hog, daemon=True)
    sender.start()
    try:
        assert holding.wait(5), "the contending sender never took the lock"
        started = time.monotonic()
        transport.panic()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        sender.join(timeout=5)

    assert elapsed < 1, f"panic waited {elapsed:.2f}s behind a busy send"
    deadline = time.monotonic() + 5
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child.poll() is not None, "panic skipped teardown when the send lock was held"


_DESCRIPTOR_CLOSERS = frozenset({"close", "closerange", "dup2"})
_TRANSPORT_SOURCES = (
    "ouroboros/remote_ssh.py",
    "ouroboros/execd.py",
    "ouroboros/remote_worker_proxy.py",
    "ouroboros/remote_browser_forward.py",
)


def _borrowed_descriptor_closes(source: str) -> list[tuple[int, str]]:
    """Every `os.close`-family call whose argument came from someone else's `fileno()`.

    An AST pass, not a line scan, and it follows ONE level of local aliasing: a name
    assigned from a `.fileno()` call anywhere in the same function is treated as a
    borrowed descriptor for the whole function. That is what makes the two-line spelling
    (`fd = stream.fileno()` then `os.close(fd)`) visible, which is the form the previous
    single-line `startswith("os.close(")` check could not see at all.

    BOUNDARY — forms this deliberately does NOT catch, so the claim stays honest:
    * a descriptor that travels through a container, an attribute (`self._fd`), a return
      value, or a function parameter — the alias map is function-local by design, because
      following descriptors across frames needs real dataflow and a gate that reports
      maybes is a gate people mute;
    * `os.close(int(text))` or any descriptor number that arrives as data;
    * a close performed by a C extension or by a library the transport calls.
    Those residues are covered by the RUNTIME probes in this module, which count real
    descriptors across a real panic — the two halves are complementary on purpose.
    """

    tree = ast.parse(source)
    offenders: list[tuple[int, str]] = []
    lines = source.splitlines()

    def _is_fileno(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fileno"
        )

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        borrowed: set[str] = set()
        for node in ast.walk(func):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if value is not None and _is_fileno(value):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            borrowed.add(target.id)
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            # `os.close(...)`, and an aliased `from os import close as _c` spelling.
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in _DESCRIPTOR_CLOSERS:
                continue
            if isinstance(fn, ast.Attribute) and not (
                isinstance(fn.value, ast.Name) and fn.value.id in {"os", "_os"}
            ):
                continue  # `stream.close()` owns its own object; that is legitimate
            for arg in node.args:
                if _is_fileno(arg) or (isinstance(arg, ast.Name) and arg.id in borrowed):
                    offenders.append((node.lineno, lines[node.lineno - 1].strip()))
                    break
    return offenders


def test_no_transport_teardown_path_steals_a_descriptor():
    """No statement in the transport closes a descriptor it borrowed from an object.

    The bug is one line away from returning and the safe primitive is not obvious at the
    call site, so the boundary is asserted structurally rather than trusted to review.
    Prose about the bug is fine; a statement that performs it is not.

    Widened from `remote_ssh.py` alone to every module that holds a live child's streams,
    because the descriptor-stealing shape is not special to the transport — it is special
    to owning a `Popen` you did not create.
    """

    repo = pathlib.Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for rel in _TRANSPORT_SOURCES:
        path = repo / rel
        if not path.exists():
            continue
        for lineno, text in _borrowed_descriptor_closes(path.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{lineno}: {text}")
    assert offenders == [], (
        "a teardown path closes a descriptor owned by someone else — the object will "
        "close it again and the second close lands on whatever now holds that number:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "snippet",
    [
        # The form the old single-line check caught.
        "import os\ndef f(s):\n    os.close(s.fileno())\n",
        # The two-line spelling it could not see.
        "import os\ndef f(s):\n    fd = s.fileno()\n    os.close(fd)\n",
        # Wrapped across lines.
        "import os\ndef f(s):\n    os.close(\n        s.fileno()\n    )\n",
        # Not at the start of a statement.
        "import os\ndef f(s):\n    if s:\n        os.close(s.fileno())\n",
        # A sibling primitive with the same consequence.
        "import os\ndef f(s):\n    fd = s.fileno()\n    os.dup2(fd, 1)\n",
        "import os\ndef f(s):\n    os.closerange(s.fileno(), s.fileno() + 1)\n",
    ],
)
def test_the_descriptor_scan_sees_the_indirect_spellings(snippet):
    """Each evasion of the previous line-based check is a finding now.

    Every case here was verified UNDETECTED by the `startswith("os.close(")` version.
    """

    assert _borrowed_descriptor_closes(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        # Closing an object you own is the prescribed pattern, not a violation.
        "def f(s):\n    s.close()\n",
        # A descriptor this code created itself is its own to close.
        "import os\ndef f(p):\n    fd = os.open(p, 0)\n    os.close(fd)\n",
        # Prose about the bug must stay legal, or the gate selects for silence.
        "import os\ndef f(s):\n    '''never os.close(s.fileno()) here'''\n    return 1\n",
    ],
)
def test_the_descriptor_scan_leaves_the_legal_shapes_alone(snippet):
    """The hardening must not turn into noise that gets muted by an allowlist."""

    assert _borrowed_descriptor_closes(snippet) == [], snippet


def test_the_descriptor_scan_states_its_own_boundary():
    """The admitted blind spots cannot be quietly deleted from the docstring."""

    doc = _borrowed_descriptor_closes.__doc__ or ""
    assert "BOUNDARY" in doc
    assert "self._fd" in doc, "the cross-frame residue must stay named"
    assert "function-local" in doc


def test_the_named_descriptor_blind_spots_are_really_blind():
    """And they cannot go stale in the other direction.

    If one of these starts being caught, the BOUNDARY paragraph is now wrong and must be
    narrowed — that is the point of asserting the blindness rather than assuming it.
    """

    through_attribute = "import os\nclass C:\n    def f(self, s):\n        self._fd = s.fileno()\n        os.close(self._fd)\n"
    through_parameter = "import os\ndef f(fd):\n    os.close(fd)\n"
    through_data = "import os\ndef f(t):\n    os.close(int(t))\n"
    for snippet in (through_attribute, through_parameter, through_data):
        assert _borrowed_descriptor_closes(snippet) == [], (
            "this form is now detected — narrow the BOUNDARY paragraph to match"
        )
