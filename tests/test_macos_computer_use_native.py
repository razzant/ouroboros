"""Native macOS qualification of quiet ``element_act`` and its fresh tree in ``unix_computer_use``.

Opt-in: runs only on macOS with ``OUROBOROS_MACOS_DESKTOP_QA=1`` in a logged-in desktop session
whose owner keeps the keyboard and pointer still for the run (allow several minutes). It compiles the
AppKit fixture ``tests/fixtures/macos_computer_use_qa.swift`` into this test's temporary directory and
starts two windows in two processes it alone owns and stops: a target with a field, Commit, a counter,
two buttons named "Twin" and a nested long command/URL, and a front app that receives synthetic keystrokes. Through the production
path - built-in install, out-of-process load and dispatch through ``ToolRegistry`` - the skill presses and
sets a value in the background target while the front app is typed into. Each fixture logs its own
state, every activation the workspace announces (a focus taken and given back between two samples
still shows), the key events it receives, pointer events and the clipboard's change count (never its
contents); the skill's answers, kept in ``receipts.jsonl``, are checked against those logs, not against
themselves. Three cases use foreground input: one hands the front to one fixture and lets the other take
it (targeted ``key``/``type_text`` with ``raise_policy="never"`` must refuse rather than take it back), one
sends Command-A spelled ``cmd+a`` and ``cmd+A`` and reads the modifiers the front app received, and one
clicks Commit at a point read off a downscaled screenshot. A fourth, only with ``OUROBOROS_MACOS_DESKTOP_QA=clipboard``
because it replaces the clipboard, pastes into the paste fixture under whatever keyboard layout is active (#2589);
another, on a qualified layout, sends cmd+g and cmd+} there and requires their menu items to fire (cmd+? is judged by
its event only, its menu recorded), and one waits with until_text for a status the target shows after a single press.
The calls run as an owner task's would
(``work_kind`` owner). A grant the run lacks becomes the skip reason, never a pass.
Further cases read a named window's rich tree, tell the window an ``observe=true`` action kept from a same-process
decoy that a default lookup finds and from the acted window's old title, keep a receipt apart from the read after a
window its commit closed, select a screenshot's display by app, and time one block of press-then-read samples (arm
B, C2 or C1; ``OUROBOROS_QA_SKILL_SEED_ROOT`` stages another skill payload on the same core for B).
``OUROBOROS_QA_FIXTURE_BINARY`` with ``OUROBOROS_QA_FIXTURE_SOURCE_SHA256`` runs a fixture built elsewhere from this
source on a machine without Swift (a clean VM guest). Pass every ``OUROBOROS_QA_*`` option AFTER the safe-test
boundary: exporting it before ``scripts/safe_test.py`` discards it. For example, in an authorized guest with
``PYTHON`` naming its test interpreter and the desktop opt-in already set:
``$PYTHON -I -S scripts/safe_test.py -- env OUROBOROS_QA_TIMING_ARM=C1 $PYTHON -m pytest tests/test_macos_computer_use_native.py::test_press_and_fresh_tree_timing_block``.
Place optional fixture/source-digest, seed-root and sample-count assignments after the same ``env`` too.
The Swift build test runs without the desktop opt-in and never starts a fixture.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
import uuid

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "tests" / "fixtures" / "macos_computer_use_qa.swift"
SKILL = "unix_computer_use"
DESKTOP = sys.platform == "darwin" and os.environ.get("OUROBOROS_MACOS_DESKTOP_QA") in ("1", "clipboard")
desktop_only = pytest.mark.skipif(not DESKTOP, reason="needs a logged-in macOS desktop and OUROBOROS_MACOS_DESKTOP_QA=1")
# The paste case replaces the clipboard, so it asks for that in so many words (the other cases leave it alone).
clipboard_only = pytest.mark.skipif(os.environ.get("OUROBOROS_MACOS_DESKTOP_QA") != "clipboard",
                                    reason="replaces the clipboard: needs OUROBOROS_MACOS_DESKTOP_QA=clipboard")
log = logging.getLogger(__name__)

# The registered public schema: process and window scope sit outside the element selector.
ACT_ARGS = {"app", "do", "select", "text", "window"}
SELECT_KEYS = {"role", "label", "identifier", "nth"}
# About ten seconds of keystrokes at 50 ms: long enough to span every action below (checked, not assumed).
TYPED = "quietlyactingonthebackgroundtarget" * 6


def _sha256(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(out_dir: pathlib.Path) -> pathlib.Path:
    prebuilt = os.environ.get("OUROBOROS_QA_FIXTURE_BINARY")
    if prebuilt:
        # A machine without a Swift toolchain (a clean VM guest) runs a binary built elsewhere from THIS source:
        # the caller names the source digest it was built from, and a different fixture source fails the run.
        assert os.environ.get("OUROBOROS_QA_FIXTURE_SOURCE_SHA256") == _sha256(SOURCE), \
            "OUROBOROS_QA_FIXTURE_BINARY was not built from this tests/fixtures/macos_computer_use_qa.swift"
        binary = out_dir / "QAFixture"
        shutil.copy2(prebuilt, binary)
        return binary
    swiftc = subprocess.run(["xcrun", "--find", "swiftc"], capture_output=True, text=True, check=False)
    if swiftc.returncode != 0:
        pytest.skip("xcrun swiftc is not installed: " + swiftc.stderr.strip())
    binary = out_dir / "QAFixture"
    # A full build, not -typecheck: some errors (a global captured before its declaration) surface only here.
    built = subprocess.run(["xcrun", "swiftc", "-swift-version", "5", "-Onone", "-o", str(binary), str(SOURCE)],
                           capture_output=True, text=True, timeout=300, check=False, cwd=out_dir)
    assert built.returncode == 0, built.stderr
    return binary


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("xcrun"), reason="the fixture is AppKit")
def test_the_fixture_builds(tmp_path):
    assert _build(tmp_path).stat().st_size > 0


class Owned:
    """One fixture process this test started and alone stops; its JSONL log is the independent record."""

    def __init__(self, binary: pathlib.Path, role: str, root: pathlib.Path) -> None:
        self.role = role
        self.title = f"ouroboros-qa-{uuid.uuid4().hex[:12]}-{role}"
        self.log = root / f"{role}.jsonl"
        self.err = root / f"{role}.stderr"
        with self.err.open("w", encoding="utf-8") as err:
            self.proc = subprocess.Popen(  # noqa: S603 - fixed argv, test-owned process
                [str(binary), role, self.title, str(self.log)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=err, start_new_session=True, text=True)
        self.pid = self.proc.pid

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def rows(self) -> list[dict]:
        """Finished rows only: a line the fixture is still writing (no newline yet) waits for the next read,
        while a finished line that is not JSON fails the run."""
        if not self.log.exists():
            return []
        data = self.log.read_bytes()
        finished = data[: data.rfind(b"\n") + 1].decode("utf-8")
        return [json.loads(line) for line in finished.splitlines() if line.strip()]

    def wait_for(self, *events: str, after: float = 0.0, timeout: float = 15.0, **match) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for row in self.rows():
                if row["event"] in events and row["t"] > after and all(row.get(k) == v for k, v in match.items()):
                    return row
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        raise AssertionError(f"{self.role} never logged {events} {match or ''} (exit {self.proc.poll()}, "
                             f"stderr {self.err.read_text(encoding='utf-8')[-500:]!r}): {self.rows()[-8:]}")

    def mark(self, label: str) -> dict:
        self.send(f"mark {label}")
        return self.wait_for("mark", label=label)

    def stop(self) -> None:
        try:
            self.send("quit")
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - a fixture that will not quit is killed
            self.proc.kill()
            self.proc.wait(timeout=10)


@pytest.fixture
def launch(tmp_path):
    binary = _build(tmp_path)
    started: list[Owned] = []

    def start(role: str) -> Owned:
        owned = Owned(binary, role, tmp_path)
        started.append(owned)
        owned.wait_for("ready")
        return owned

    try:
        yield start
    finally:
        for owned in reversed(started):
            owned.stop()


@pytest.fixture
def host(tmp_path, monkeypatch):
    """The production built-in install -> trust -> turn-on -> out-of-process load path."""
    from ouroboros import extension_loader
    from ouroboros.skill_builtin import _dependency_pass, install_builtin_skill
    from ouroboros.skill_loader import load_enabled
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from tests._shared import clean_extension_runtime_state

    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    data = tmp_path / "data"
    (data / "skills").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", data / "settings.json")
    clean_extension_runtime_state()
    # The payload under test: this checkout's skill, or (timing comparison only) another skill payload staged by the
    # same core, so two payloads are compared on one core, harness, interpreter and fixture.
    seed_root = pathlib.Path(os.environ.get("OUROBOROS_QA_SKILL_SEED_ROOT") or REPO / "skills")
    payload = {str(path.relative_to(seed_root / SKILL)): _sha256(path)
               for path in sorted((seed_root / SKILL).rglob("*")) if path.is_file() and "__pycache__" not in path.parts}
    with (tmp_path / "payload.jsonl").open("w", encoding="utf-8") as out:
        out.write(json.dumps({"seed_root": str(seed_root), "files": payload, "fixture_source_sha256": _sha256(SOURCE),
                              "python": sys.version, "executable": sys.executable}) + "\n")
    assert install_builtin_skill(SKILL, drive_root=data, seed_dir=seed_root, turn_on_when_ready=True,
                                 installed_by="owner")["ok"]
    _dependency_pass(data, [SKILL], log)
    assert load_enabled(data, SKILL) is True
    registry = ToolRegistry(repo_dir=REPO, drive_root=data)
    # An owner task's context. Work of unknown provenance counts as consciousness work, whose full Safety
    # Supervisor coverage holds every action while no safety model is reachable, which a hermetic run never has.
    registry.set_context(ToolContext(repo_dir=REPO, drive_root=data, task_metadata={"work_kind": "owner"}))

    receipts = tmp_path / "receipts.jsonl"  # every answer as received, on the fixtures' clock (system uptime)

    def call(name, **args):
        started = time.perf_counter()
        raw = str(registry.execute(extension_loader.extension_surface_name(SKILL, name), args))
        wall_ms = round((time.perf_counter() - started) * 1000, 3)
        with receipts.open("a", encoding="utf-8") as out:
            row = {"t": time.monotonic(), "tool": name, "args": args, "wall_ms": wall_ms,
                   "bytes": len(raw.encode("utf-8")), "answer": raw}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        call.last = row  # the raw answer's size and wall time, for the timing samples
        head, sep, _tail = raw.partition("\n\n⚠️ ")
        try:
            return json.loads(head if sep else raw)
        except ValueError:
            raise AssertionError(f"{name} answered without a receipt: {raw[:1500]}") from None

    def schema(name):
        tool = extension_loader.get_tool(extension_loader.extension_surface_name(SKILL, name))
        return None if tool is None else tool["schema"]

    try:
        yield call, schema
    finally:
        clean_extension_runtime_state()
        # The parent may name a durable evidence root for the native opt-in. Preserve the independent
        # process logs beside the tool receipts and timing samples even when pytest's tmp_path is released.
        evidence_root = os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT")
        if evidence_root:
            case = pathlib.Path(evidence_root) / "macos-computer-use" / (uuid.uuid4().hex + "-" + receipts.parent.name)
            case.mkdir(parents=True)
            for pattern in ("*.jsonl", "*.png"):
                for source in tmp_path.glob(pattern):
                    shutil.copy2(source, case / source.name)


def _grants(call) -> dict:
    caps = call("capabilities")
    assert caps.get("permission_state_verified") is True, caps
    return caps["permissions"]


def _interface(schema) -> None:
    """The registered element_act takes the arguments this test sends, or the run stops with its schema."""
    registered = schema("element_act")
    assert registered is not None, "element_act is not registered by the installed unix_computer_use"
    props = registered.get("properties", {})
    select = props.get("select", {}).get("properties", {})
    missing = sorted(ACT_ARGS - set(props)) + sorted(SELECT_KEYS - set(select))
    assert not missing, f"element_act schema lacks {missing}; registered: {json.dumps(registered)}"
    assert set(registered.get("required", [])) == {"app", "do", "select"}, registered


def _new_interface(schema) -> None:
    _interface(schema)
    for name, required in (("ax_tree", {"window", "text_limit"}),
                           ("element_act", {"observe", "text_limit"})):
        registered = schema(name)
        assert registered is not None, name
        missing = required - set(registered.get("properties", {}))
        assert not missing, f"{name} schema lacks {sorted(missing)}: {json.dumps(registered)}"


def _identified(tree: dict, identifier: str) -> dict:
    matches = [node for node in tree.get("nodes", []) if node.get("identifier") == identifier]
    assert len(matches) == 1, {"identifier": identifier, "matches": matches, "tree": tree}
    return matches[0]


def _act(call, target: Owned, do: str, *, text: str | None = None, **select) -> dict:
    args = {"app": f"pid:{target.pid}", "do": do, "select": select, "window": target.title}
    if text is not None:
        args["text"] = text
    return call("element_act", **args)


def _between(rows: list[dict], start: float, end: float, *events: str) -> list[dict]:
    return [row for row in rows if start < row["t"] < end and row["event"] in events]


def _input_ready(call, *, cliclick: bool = True, **also: str) -> None:
    """Foreground input needs the grants to post it and cliclick to send it (Command and one character go through
    osascript, cliclick=False); without them the case is skipped."""
    grants = _grants(call)
    required = {"accessibility": "trusted", "post_events": "granted", **also}
    lacking = {key: grants.get(key, "absent") for key, value in required.items() if grants.get(key) != value}
    if lacking:
        pytest.skip(f"unsupported here, not passed: this process lacks {lacking} (no grant is requested)")
    if cliclick and (call("capabilities").get("input") or {}).get("cliclick") is not True:
        pytest.skip("unsupported here, not passed: cliclick is not installed, so key, type_text and click send nothing")


def _hand_front(call, owned: Owned) -> dict:
    """Puts an owned fixture in front through activate_app, the switch an owner would make, and returns its own
    snapshot once it is the active application with its key window."""
    seen = owned.mark("handing")
    if not (seen["app_active"] and seen["key_window"] and seen["front_pid"] == owned.pid):
        switched = call("activate_app", app=f"pid:{owned.pid}")
        assert switched.get("ok") is True and (switched.get("target") or {}).get("verified") is True, switched
        if not seen["app_active"]:
            owned.wait_for("app_became_active", after=seen["t"])
        if not seen["key_window"]:
            owned.wait_for("window_became_key", after=seen["t"])
    held = owned.mark("in_front")
    assert held["app_active"] and held["key_window"] and held["front_pid"] == owned.pid, held
    return held


@desktop_only
def test_without_accessibility_element_act_refuses_and_the_target_is_untouched(host, launch):
    call, schema = host
    grants = _grants(call)
    if grants.get("accessibility") != "untrusted":
        pytest.skip(f"this case qualifies the untrusted refusal; Accessibility here is {grants.get('accessibility')!r}")
    _interface(schema)
    target = launch("target")
    before = target.mark("before")
    refused = _act(call, target, "press", identifier="qa.commit")
    after = target.mark("after")

    assert refused.get("ok") is not True and refused.get("dispatched") is False, refused
    assert "accessib" in json.dumps(refused).lower(), refused
    assert not _between(target.rows(), before["t"], after["t"], "commit", "app_became_active", "window_became_key")
    assert after["field"] == "" and after["count"] == 0


@desktop_only
def test_quiet_press_and_set_value_while_the_front_app_is_typed_into(host, launch):
    call, schema = host
    grants = _grants(call)
    # The helper's own preflight vocabulary (macos_native.op_perms); an absent or unknown grant is lacking too.
    required = {"accessibility": "trusted"}
    lacking = {key: grants.get(key, "absent") for key, value in required.items() if grants.get(key) != value}
    if lacking:
        pytest.skip(f"unsupported here, not passed: this process lacks {lacking} (no grant is requested)")
    _interface(schema)
    target = launch("target")
    front = launch("front")
    monitor = front.wait_for("pointer_monitor")
    if monitor.get("installed") is not True:
        pytest.skip("pointer observation unavailable; no non-interference verdict")
    try:
        front.wait_for("app_became_active", timeout=5)
    except AssertionError:  # an app started outside LaunchServices may be kept back: hand the front over
        handoff = call("activate_app", app=f"pid:{front.pid}")
        assert handoff.get("ok") is True and (handoff.get("target") or {}).get("verified") is True, handoff
        front.wait_for("app_became_active")

    # Every step runs even after a failure, and each failure keeps both answers, so one desktop
    # run explains all of what went wrong.
    problems: list[str] = []

    def check(label, condition, payload):
        if not condition:
            problems.append(f"{label}: {json.dumps(payload, ensure_ascii=False)[:1500]}")

    def quiet(label, result):
        check(label, result.get("ok") is True and result.get("dispatched") is True and result.get("action_outcome") == "completed"
              and (result.get("front_before") or {}).get("pid") == front.pid
              and (result.get("front_after") or {}).get("pid") == front.pid
              and "observation" not in result, result)  # a plain action (no observe) reads nothing afterwards

    def refused(label, result, reason):
        check(label, result.get("ok") is False and result.get("reason") == reason and result.get("dispatched") is False
              and "observation" not in result, result)

    t_before, f_before = target.mark("before"), front.mark("before")
    front.send(f"type 50 {TYPED}")
    started = front.wait_for("typing_started", "typing_unsupported", after=f_before["t"])
    if started["event"] == "typing_unsupported":
        pytest.skip(f"unsupported here, not passed: the fixture cannot post keystrokes ({started['reason']})")
    front.wait_for("front_text", after=started["t"])  # keystrokes are landing before the first action
    acting = front.mark("acting")
    quiet("set_value", _act(call, target, "set_value", text="1500", identifier="qa.field"))
    quiet("press", _act(call, target, "press", identifier="qa.commit"))
    refused("ambiguous", _act(call, target, "press", label="Twin"), "element_ambiguous")
    quiet("nth", _act(call, target, "press", label="Twin", nth=2))
    # Three nodes cannot reach the twins: a walk cut by max_nodes refuses even with nth.
    refused("incomplete walk", call("element_act", app=f"pid:{target.pid}", do="press", select={"label": "Twin", "nth": 1},
                                    window=target.title, max_nodes=3), "window_not_fully_read")
    refused("absent window", call("element_act", app=f"pid:{target.pid}", do="press", select={"identifier": "qa.commit"},
                                  window=target.title + "-absent"), "window_not_found")
    target.send("replace")
    target.wait_for("commit_replaced", after=t_before["t"])
    quiet("press after replace", _act(call, target, "press", identifier="qa.commit"))
    acted = front.mark("acted")

    ended = front.wait_for("typing_finished", "typing_stopped", after=started["t"], timeout=30)
    check("typing ran to the end", ended["event"] == "typing_finished", ended)
    t_after, f_after = target.mark("after"), front.mark("after")
    trows, frows = target.rows(), front.rows()
    # A keystroke landed before the first action and another after the last: typing_finished alone comes one
    # timer tick after the final character, so it cannot show that the typing outlasted the actions.
    landed = [row["t"] for row in frows if row["event"] == "front_text" and row["t"] > started["t"]]
    check("typing spanned every action", bool(landed) and landed[0] < acting["t"] and acted["t"] < landed[-1],
          {"acting": acting["t"], "acted": acted["t"], "first_keystroke": landed[:1], "last_keystroke": landed[-1:]})
    start, end = min(t_before["t"], f_before["t"]), max(t_after["t"], f_after["t"])

    commits = [(r["instance"], r["field"], r["count"]) for r in _between(trows, start, end, "commit")]
    check("target commits", commits == [(1, "1500", 1), (2, "1500", 2)], commits)
    check("twin", [r["identifier"] for r in _between(trows, start, end, "twin")] == ["qa.twinB"], trows)
    check("no keystroke reached the target", not _between(trows, start, end, "key_down"), trows)
    check("target never active or key", not _between(trows, start, end, "app_became_active", "window_became_key"), trows)
    check("front kept active and key", not _between(frows, start, end, "app_resigned_active", "window_resigned_key"), frows)
    check("no application activated", not _between(frows, start, end, "workspace_activated"), frows)
    check("every keystroke reached the front", f_after["text"] == TYPED, f_after)
    check("no pointer event", not _between(frows, start, end, "pointer_event"), frows)
    check("pointer unmoved", f_before["mouse"] == f_after["mouse"], [f_before["mouse"], f_after["mouse"]])
    check("clipboard unchanged", f_before["pasteboard_change_count"] == f_after["pasteboard_change_count"],
          [f_before["pasteboard_change_count"], f_after["pasteboard_change_count"]])
    check("target state", (t_after["field"], t_after["count"], t_after["status"], t_after["app_active"])
          == ("1500", 2, "committed:1500", False), t_after)

    # The positive control, after the quiet interval: a deliberate switch to the target is seen by the same monitors
    # that saw none above (it says nothing about the pointer monitor, which only the absence of pointer rows covers).
    c_before = front.mark("control_before")
    switched = call("activate_app", app=f"pid:{target.pid}")
    check("control switch", switched.get("ok") is True and (switched.get("target") or {}).get("verified") is True, switched)
    for owned, event, match in ((front, "app_resigned_active", {}), (front, "workspace_activated", {"activated_pid": target.pid}),
                                (target, "app_became_active", {})):
        try:
            owned.wait_for(event, after=c_before["t"], timeout=5, **match)
        except AssertionError as exc:
            check(f"control seen: {owned.role} {event}", False, str(exc))
    assert not problems, "\n".join(problems)


@desktop_only
def test_never_policy_leaves_the_front_with_the_app_that_took_it(host, launch):
    """SKILL.md's handoff: after a one-shot activate_app, targeted input with raise_policy="never" refuses once
    another application holds the front instead of taking it back. A is handed the front, B takes it, then key and
    type_text aimed at A must send nothing and switch nothing; the same key aimed at B, which is in front, must land
    in B, so the logs that show no keystroke are logs that would have shown one."""
    call, _schema = host
    _input_ready(call)
    handed, taker = launch("target"), launch("front")
    _hand_front(call, handed)
    _hand_front(call, taker)  # stands in for the owner switching to another application

    problems: list[str] = []

    def check(label, condition, payload):
        if not condition:
            problems.append(f"{label}: {json.dumps(payload, ensure_ascii=False)[:1500]}")

    a_before, b_before = handed.mark("before"), taker.mark("before")
    aimed = f"pid:{handed.pid}"
    refusals = {"key": call("key", keys="x", app=aimed, raise_policy="never"),
                "type_text": call("type_text", text="x", app=aimed, raise_policy="never")}
    time.sleep(1.0)  # a late switch or keystroke would land in the logs before the marks below
    a_after, b_after = handed.mark("after"), taker.mark("after")
    for name, refused in refusals.items():
        check(f"{name} refused", refused.get("ok") is False and refused.get("reason") == "target_lost"
              and refused.get("dispatched") is not True and not refused.get("observed_effect")
              and (refused.get("target") or {}).get("activated") is False
              and (refused.get("front") or {}).get("pid") == taker.pid, refused)
    arows, brows = handed.rows(), taker.rows()
    start, end = min(a_before["t"], b_before["t"]), max(a_after["t"], b_after["t"])
    check("A never active, key or typed into",
          not _between(arows, start, end, "app_became_active", "window_became_key", "key_down", "field_typed"), arows)
    check("B kept active and key", not _between(brows, start, end, "app_resigned_active", "window_resigned_key"), brows)
    check("no application activated", not _between(brows, start, end, "workspace_activated"), brows)
    check("no keystroke reached B", not _between(brows, start, end, "key_down", "front_text")
          and b_after["text"] == b_before["text"], [b_before, b_after])
    check("B still in front", b_after["app_active"] and b_after["front_pid"] == taker.pid
          and (call("capabilities").get("frontmost") or {}).get("pid") == taker.pid, b_after)

    # The control: the same key aimed at B, which is in front, is sent and lands there.
    sent = call("key", keys="k", app=f"pid:{taker.pid}", raise_policy="never")
    check("control sent", sent.get("ok") is True and sent.get("dispatched") is True
          and (sent.get("delivered_to") or {}).get("pid") == taker.pid
          and (sent.get("target") or {}).get("activated") is False, sent)
    try:
        landed = taker.wait_for("front_text", after=b_after["t"])
        check("control landed in B", landed["text"] == b_before["text"] + "k", landed)
    except AssertionError as exc:
        check("control landed in B", False, str(exc))
    assert not problems, "\n".join(problems)


@desktop_only
def test_command_a_arrives_as_command_a_however_its_letter_is_cased(host, launch):
    """key("cmd+a") and key("cmd+A") both name Command-A: a letter's case in a combination is not a Shift. The front
    fixture logs every key event it receives with its modifiers. Command-A is harmless there (the fixture has no
    menu that acts on it); no quit or logout key is ever sent. It is the Command chord program's: the characters the
    event types under Command are "a" in any layout (charactersIgnoringModifiers is the typing plane: "ф" in Russian)."""
    call, _schema = host
    _input_ready(call, cliclick=False)
    front = launch("front")
    _hand_front(call, front)
    problems: list[str] = []
    for keys in ("cmd+a", "cmd+A"):
        mark = front.mark(keys)
        sent = call("key", keys=keys, app=f"pid:{front.pid}", raise_policy="never")
        if not (sent.get("ok") is True and sent.get("dispatched") is True
                and (sent.get("delivered_to") or {}).get("pid") == front.pid):
            problems.append(f"{keys} not sent to the front fixture: {json.dumps(sent)[:1500]}")
            continue
        try:
            row = front.wait_for("key_down", after=mark["t"], timeout=5)
        except AssertionError as exc:
            problems.append(f"{keys} never arrived: {exc}")
            continue
        if (row["characters"], row["modifiers"], sent.get("method")) != ("a", ["cmd"], "command_chord"):
            problems.append(f"{keys} arrived as {row['modifiers']} + {row['characters']!r}, not Command-A: "
                            f"{json.dumps(row)} (sent {json.dumps(sent.get('cmd'))})")
    after = front.mark("after")
    assert not problems and after["text"] == "", "\n".join(problems) or after


@desktop_only
def test_a_point_read_off_a_screenshot_clicks_the_button_drawn_there(host, launch):
    """Observation-bound input: the centre of the fixture's own Commit button, converted into the pixels of a
    screenshot taken 512 px wide (so image->input is not the identity on any display), clicked with that
    screenshot's observation_id, presses Commit once. Where Commit is comes from the fixture's own geometry."""
    call, _schema = host
    _input_ready(call, screen_recording="granted")
    target = launch("target")
    placed = _hand_front(call, target)
    shot = call("screenshot", max_width=512, max_height=384)
    transform = shot.get("coord_transform") or {}
    assert shot.get("ok") is True and shot.get("observation_id") and transform.get("sx", 1) > 1, shot
    x, y, w, h = placed["commit_frame"]
    point = [round((x + w / 2 - transform["ox"]) / transform["sx"]),
             round((y + h / 2 - transform["oy"]) / transform["sy"])]
    before = target.mark("before")
    clicked = call("click", x=point[0], y=point[1], observation_id=shot["observation_id"],
                   app=f"pid:{target.pid}", raise_policy="never")
    try:
        pressed = target.wait_for("commit", after=before["t"], timeout=5)
    except AssertionError as exc:
        pressed = {"missing": str(exc)}
    time.sleep(0.5)  # a second press or a stray twin would show before the mark below
    after = target.mark("after")
    rows = _between(target.rows(), before["t"], after["t"], "commit", "twin")
    assert (clicked.get("ok") is True and clicked.get("dispatched") is True
            and clicked.get("observation_id") == shot["observation_id"] and clicked.get("mapped_from") == point
            and (clicked.get("delivered_to") or {}).get("pid") == target.pid), clicked
    assert [(r["event"], r.get("instance")) for r in rows] == [("commit", 1)] and after["count"] == 1, \
        {"point": point, "commit_frame": placed["commit_frame"], "transform": transform, "pressed": pressed,
         "rows": rows}


@desktop_only
@clipboard_only
def test_paste_arrives_as_the_active_layouts_command_v_and_pastes_through_the_edit_menu(host, launch):
    """#2589: type_text(via="paste") posts Command and the key that gives v under Command in the active layout. The
    paste fixture's own log, not the receipt, says what arrived (characters "v", exactly the Command modifier), which
    action its standard Edit menu performed and what text landed. With the window itself first responder the same
    chord arrives and nothing is pasted while the receipt still says dispatched: what the receipt cannot see."""
    call, _schema = host
    _input_ready(call)
    paste = launch("paste")
    _hand_front(call, paste)
    token = f"qa{uuid.uuid4().hex[:8]}Привет"
    paste.send(f"expect {token}")
    problems: list[str] = []

    def check(label, condition, payload):
        if not condition:
            problems.append(f"{label}: {json.dumps(payload, ensure_ascii=False)[:1500]}")

    for focus, lands in (("paste", True), ("window", False)):
        paste.send("clear")
        paste.send(f"focus {focus}")
        before = paste.mark(f"before-{focus}")
        sent = call("type_text", text=token, via="paste", app=f"pid:{paste.pid}", raise_policy="never")
        check(f"{focus}: dispatched to the fixture", sent.get("ok") is True and sent.get("dispatched") is True
              and sent.get("clipboard_replaced") is True and (sent.get("delivered_to") or {}).get("pid") == paste.pid, sent)
        time.sleep(1.0)  # a late key, action or text lands before the mark below
        after = paste.mark(f"after-{focus}")
        rows = _between(paste.rows(), before["t"], after["t"], "key_down", "paste_action", "menu_sent_action")
        keys = [(r["modifiers"], r.get("characters")) for r in rows if r["event"] == "key_down"]
        check(f"{focus}: one Command chord that reads as v", keys == [(["cmd"], "v")],
              {"keys": keys, "input_source": after.get("input_source"), "sent": sent.get("helper_stderr")})
        pasted = [r.get("action") for r in rows if r["event"] == "paste_action"]
        check(f"{focus}: Edit > Paste {'performed' if lands else 'not performed'}", pasted == (["paste:"] if lands else []), rows)
        check(f"{focus}: text", (after["text"], after.get("pasteboard_is_token")) == ((token if lands else ""), True), after)
    assert not problems, "\n".join(problems)


# The layouts the Command chord program is qualified on (12.11.2026 guest matrix): two ignore Shift under Command (ABC,
# Dvorak), three give Command+Shift characters of their own.
QUALIFIED_LAYOUTS = {f"com.apple.keylayout.{name}" for name in ("ABC", "Dvorak", "DVORAK-QWERTYCMD", "RussianWin", "Russian")}


@desktop_only
def test_command_shortcuts_reach_the_menu_by_the_character_command_gives(host, launch, tmp_path):
    """key("cmd+g"), key("cmd+}") and key("cmd+?") under the active layout, one of the qualified ones: cmd+g and cmd+}
    must be sent (never refused) and fire the paste fixture's own Find Next and Shortcut } items, the consumer's
    answer. cmd+? is checked by its event only: one key down with exactly Command and Shift that the layout itself reads
    as ? (its characters, or charactersIgnoringModifiers where the key types under Command alone what it typed with
    Shift: the layout ignored Shift). Whether the app's own Shortcut ? item fires is recorded, not judged: in the guest
    no ⌘? sent on key code 44 reached it, the base's cliclick included, and what took it is not established. So this
    case does not prove ⌘? reaches an app's command."""
    call, _schema = host
    _input_ready(call, cliclick=False)
    paste = launch("paste")
    _hand_front(call, paste)
    layout = (paste.mark("layout").get("input_source") or {}).get("layout")
    if layout not in QUALIFIED_LAYOUTS:
        pytest.skip(f"unsupported here, not passed: {layout!r} is not one of the qualified layouts {sorted(QUALIFIED_LAYOUTS)}")
    problems: list[str] = []
    for keys, char, modifiers, item in (("cmd+g", "g", ["cmd"], "Find Next"), ("cmd+}", "}", ["cmd", "shift"], "Shortcut }"),
                                        ("cmd+?", "?", ["cmd", "shift"], "Shortcut ?")):
        before = paste.mark(f"before-{keys}")
        sent = call("key", keys=keys, app=f"pid:{paste.pid}", raise_policy="never")
        time.sleep(1.0)  # a late key or menu action lands before the mark below
        after = paste.mark(f"after-{keys}")
        rows = _between(paste.rows(), before["t"], after["t"], "key_down", "find_next")
        downs = [r for r in rows if r["event"] == "key_down"]
        chosen = [r["sender"] for r in rows if r["event"] == "find_next"]
        seen = {"layout": layout, "receipt": {k: sent.get(k) for k in ("ok", "dispatched", "method", "reason", "helper_stderr")},
                "downs": [{k: r.get(k) for k in ("key_code", "modifiers", "characters", "chars_ignoring", "command_characters")}
                          for r in downs], "menu": chosen}
        sent_ok = (sent.get("ok") is True and sent.get("dispatched") is True and sent.get("method") == "command_chord"
                   and (sent.get("delivered_to") or {}).get("pid") == paste.pid)
        one = len(downs) == 1 and downs[0]["modifiers"] == modifiers
        if char != "?":
            if not (sent_ok and one and chosen == [item]):
                problems.append(f"{keys}: {json.dumps(seen, ensure_ascii=False)[:1500]}")
            continue
        row = downs[0] if one else {}
        ignored = bool(row) and row.get("command_characters") == row.get("characters")
        reads = row.get("chars_ignoring") if ignored else row.get("characters")
        if not (sent_ok and one and reads == "?"):
            problems.append(f"{keys}: {json.dumps(seen, ensure_ascii=False)[:1500]}")
        with (tmp_path / "diagnostics.jsonl").open("a", encoding="utf-8") as out:  # kept beside the receipts as evidence
            out.write(json.dumps({"case": keys, **seen, "judged": "event only", "menu_item_fired": chosen == [item],
                                  "gap": "whether ⌘? reaches an app's own command is not established here"},
                                 ensure_ascii=False) + "\n")
    assert not problems, "\n".join(problems)


@desktop_only
def test_until_text_waits_for_delayed_text_after_one_press(host, launch):
    """The target shows "committed:" a second after Commit (mode delay; "loading" until then). element_act with
    observe=true and until_text presses ONCE and rereads only that window until the text is there; an absent text read
    by ax_tree ends at its deadline as not_observed, or as unknown that names a deadline-cut latest read and the whole
    earlier tree it kept (the fixture's own log says the press count)."""
    call, schema = host
    _new_interface(schema)
    if _grants(call).get("accessibility") != "trusted":
        pytest.skip("unsupported here, not passed: this process lacks the Accessibility grant (none is requested)")
    target = launch("target")
    target.send("mode delay")
    target.wait_for("mode_set", mode="delay")
    before = target.mark("before")
    acted = call("element_act", app=f"pid:{target.pid}", do="press", select={"identifier": "qa.commit"},
                 window=target.title, observe=True, until_text="committed:", timeout_ms=5000)
    shown = target.wait_for("status_shown", after=before["t"], timeout=5)
    seen = acted.get("observation") or {}
    assert acted.get("action_outcome") == "completed" and acted.get("dispatched") is True, acted
    assert (seen.get("until_text") or {}).get("status") == "matched" and seen["until_text"]["reads"] >= 2, seen
    absent = call("ax_tree", app=f"pid:{target.pid}", window=target.title, until_text="never-shown-text", timeout_ms=1500)
    wait = absent.get("until_text") or {}
    after = target.mark("after")
    assert shown["count"] == after["count"] == 1, (shown, after)
    assert wait.get("status") == "not_observed" or (
        wait.get("status") == "unknown" and "timeout" in (wait.get("latest") or {}).get("truncated_by", [])
        and wait.get("tree_read", 0) >= 1 and absent.get("ok") is True), absent


@desktop_only
def test_named_window_tree_keeps_nested_unicode_command_and_url(host, launch):
    """One background-window tree must expose real hierarchy and the full synthetic fields on request."""
    call, schema = host
    if _grants(call).get("accessibility") != "trusted":
        pytest.skip("unsupported here, not passed: Accessibility is not trusted")
    _new_interface(schema)
    target, front = launch("target"), launch("front")
    if front.wait_for("pointer_monitor").get("installed") is not True:
        pytest.skip("pointer observation unavailable; no non-interference verdict")
    _hand_front(call, front)
    t_before, f_before = target.mark("tree_before"), front.mark("tree_before")
    args = {"app": f"pid:{target.pid}", "window": target.title, "max_nodes": 100, "timeout_ms": 5000}
    compact = call("ax_tree", **args)
    full = call("ax_tree", **args, text_limit=100000)
    t_after, f_after = target.mark("tree_after"), front.mark("tree_after")

    for tree in (compact, full):
        assert tree.get("ok") is True and tree.get("pid") == target.pid and tree.get("window") == target.title, tree
        assert not tree.get("truncated") and tree.get("truncated_by") == [], tree
        assert (tree.get("target") or tree.get("app") or {}).get("pid") == target.pid, tree
    command_short, command = _identified(compact, "qa.command"), _identified(full, "qa.command")
    url_short, url = _identified(compact, "qa.url"), _identified(full, "qa.url")
    assert len(t_before["command"]) > 200 and len(t_before["url"]) > 200
    assert command_short["value"] == t_before["command"][:200] and command_short["value_len"] == len(t_before["command"])
    assert url_short["url"] == t_before["url"][:200] and url_short["url_len"] == len(t_before["url"])
    assert command["value"] == t_before["command"] and "value_len" not in command
    assert url["url"] == t_before["url"] and "url_len" not in url
    assert command.get("description") == "Synthetic command", command
    assert url.get("title") == "Open synthetic reference" and url.get("description") == "Synthetic URL", url
    nodes = {node["id"]: node for node in full["nodes"]}
    group = _identified(full, "qa.command_group")
    assert "parent_id" not in full["nodes"][0] and group["id"] in nodes
    for child in (command, url):
        ancestry = set()
        current = child
        while "parent_id" in current:
            assert current["parent_id"] in nodes and current["parent_id"] not in ancestry, current
            ancestry.add(current["parent_id"])
            current = nodes[current["parent_id"]]
        assert group["id"] in ancestry, {"child": child, "group": group, "ancestry": ancestry}
    start, end = min(t_before["t"], f_before["t"]), max(t_after["t"], f_after["t"])
    assert not _between(target.rows(), start, end, "commit", "app_became_active", "window_became_key", "key_down")
    assert not _between(front.rows(), start, end, "app_resigned_active", "window_resigned_key", "workspace_activated", "pointer_event")
    assert f_after["app_active"] and f_after["front_pid"] == front.pid and f_before["mouse"] == f_after["mouse"]
    assert f_before["pasteboard_change_count"] == f_after["pasteboard_change_count"]


def _number(mark: dict, role: str) -> int | None:
    """The windowNumber (CGWindowID) the fixture itself reports for one of its windows: an id the skill does not supply."""
    return next((row.get("number") for row in mark.get("windows", []) if row.get("role") == role), None)


def _values(tree: dict, identifier: str) -> list:
    return [node.get("value") for node in tree.get("nodes", []) if node.get("identifier") == identifier]


def _evidence(tmp_path: pathlib.Path, name: str, row: dict) -> None:
    with (tmp_path / name).open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")


@desktop_only
@pytest.mark.parametrize("mode", ["retitle", "shift"])
def test_observation_reads_the_acted_window_not_a_new_default_or_its_old_title(host, launch, tmp_path, mode):
    """The read after an action is of the window the action was scoped to, kept from before the action, not looked up
    again. The target process has a second window, the decoy, with its own qa.sentinel. Commit (retitle) renames its
    own window and brings the decoy in front as the app's main window, or (shift) only brings the decoy forward. The
    fixture's log - its own sentinel, titles and windowNumbers - is the oracle. press with observe=true must read the
    acted window's new sentinel; afterwards the old title must name no window (retitle) and a default lookup must
    select the decoy: if it does not, the case did not tell a retained window from a fresh default lookup and fails.
    The decoy's promotion is deliberate focus stress inside the target process, so front quietness is not judged here
    (the quiet case does that); only that the tool activated nothing and typed nothing."""
    call, schema = host
    if _grants(call).get("accessibility") != "trusted":
        pytest.skip("unsupported here, not passed: Accessibility is not trusted")
    _new_interface(schema)
    target, front = launch("target"), launch("front")
    _hand_front(call, front)
    target.send("decoy")
    target.wait_for("decoy_ready")
    decoy_title = target.title + "-decoy"
    target.send(f"mode {mode}")
    target.wait_for("mode_set", mode=mode)
    app = f"pid:{target.pid}"
    problems: list[str] = []

    def check(label, condition, payload):
        if not condition:
            problems.append(f"{label}: {json.dumps(payload, ensure_ascii=False)[:2000]}")

    setup = _act(call, target, "set_value", text="1500", identifier="qa.field")
    check("setup", setup.get("ok") is True and setup.get("action_outcome") == "completed", setup)
    before = target.mark("press_before")
    acted_id, decoy_id = _number(before, "target"), _number(before, "decoy")
    pressed = call("element_act", app=app, do="press", select={"identifier": "qa.commit"}, window=target.title,
                   max_nodes=100, timeout_ms=5000, observe=True)
    try:
        commit = target.wait_for("commit", after=before["t"], timeout=5)
    except AssertionError as exc:
        commit = {"missing": str(exc)}
    time.sleep(0.5)  # a second commit would land before the mark below
    after = target.mark("press_after")
    commits = _between(target.rows(), before["t"], after["t"], "commit")
    check("exactly one commit", [(r["count"], r["field"]) for r in commits] == [(1, "1500")], commits)
    new_title = commit.get("title_after")
    expected = "retitled:1" if mode == "retitle" else "target:1"
    check("the fixture did its mode", commit.get("sentinel") == expected
          and (new_title == target.title + "-retitled" if mode == "retitle" else new_title == target.title), commit)

    check("receipt", pressed.get("ok") is True and pressed.get("dispatched") is True
          and pressed.get("action_outcome") == "completed"
          and (pressed.get("element") or {}).get("identifier") == "qa.commit"
          and (pressed.get("window") or {}).get("title") == target.title, pressed)
    observation = pressed.get("observation") or {}
    seen = observation.get("window") or {}
    check("observation complete", observation.get("ok") is True and not observation.get("truncated")
          and observation.get("truncated_by") == [], observation)
    check("observation is of the acted window, under its title now", seen.get("title") == new_title, seen)
    check("observation reads the acted window's sentinel and count", _values(observation, "qa.sentinel") == [expected]
          and _values(observation, "qa.count") == ["count:1"], observation)
    ids = {"fixture_acted": acted_id, "fixture_decoy": decoy_id, "receipt": (pressed.get("window") or {}).get("window_id"),
           "observation": seen.get("window_id")}
    id_proof = all(isinstance(v, int) and not isinstance(v, bool) for v in ids.values())
    if id_proof:
        check("window ids: receipt and observation are the acted window, not the decoy",
              ids["receipt"] == ids["observation"] == acted_id != decoy_id, ids)

    old = call("ax_tree", app=app, window=target.title, max_nodes=100, timeout_ms=5000)
    if mode == "retitle":
        check("the old title names no window now", old.get("ok") is False and old.get("reason") == "window_not_found", old)
    else:
        check("the unchanged title still names the acted window", old.get("ok") is True
              and _values(old, "qa.sentinel") == [expected], old)
    default = call("ax_tree", app=app, max_nodes=100, timeout_ms=5000)
    check("a default lookup now selects the decoy (else this case does not discriminate)", default.get("ok") is True
          and default.get("window") == decoy_title and _values(default, "qa.sentinel") == ["decoy"], default)
    if id_proof and isinstance(default.get("window_id"), int):
        check("default lookup id is the decoy's", default["window_id"] == decoy_id, {"default": default.get("window_id"), **ids})
    renamed = call("ax_tree", app=app, window=new_title, max_nodes=100, timeout_ms=5000)
    check("the acted window by its title now", renamed.get("ok") is True and _values(renamed, "qa.sentinel") == [expected], renamed)

    # Window refusals: nothing is done when the named window is absent or ambiguous (two windows titled as the decoy).
    target.send("twinwindow")
    target.wait_for("twin_window_ready")
    r_before = target.mark("refusals_before")
    for label, window, reason in (("absent window", target.title + "-absent", "window_not_found"),
                                  ("ambiguous window", decoy_title, "window_ambiguous")):
        refusal = call("element_act", app=app, do="press", select={"identifier": "qa.commit"}, window=window, observe=True)
        check(label, refusal.get("ok") is False and refusal.get("reason") == reason and refusal.get("dispatched") is False
              and (refusal.get("observation") or {}).get("ok") is False, refusal)
    time.sleep(0.5)
    r_after = target.mark("refusals_after")
    check("no action on refusal", not _between(target.rows(), r_before["t"], r_after["t"], "commit", "twin")
          and r_after["count"] == 1, r_after)
    rows = target.rows()
    check("the tool activated and typed nothing", not _between(rows, before["t"], r_after["t"], "app_became_active", "key_down"), rows[-12:])
    _evidence(tmp_path, "retained_window.jsonl", {
        "mode": mode, "ids": ids, "id_selection": "integer ids compared" if id_proof else "UNTESTED: an id was not an integer",
        "acted_title_before": target.title, "acted_title_after": new_title, "decoy_title": decoy_title,
        "observation_window": seen, "observation_sentinel": _values(observation, "qa.sentinel"),
        "old_title_lookup": {k: old.get(k) for k in ("ok", "reason", "window", "window_id")},
        "default_lookup": {k: default.get(k) for k in ("ok", "reason", "window", "window_id")},
        "default_sentinel": _values(default, "qa.sentinel"), "problems": problems})
    assert not problems, "\n".join(problems)


@desktop_only
def test_a_window_closed_by_its_commit_keeps_the_receipt_apart_from_the_read_after(host, launch, tmp_path):
    """Commit closes its own window. The action receipt stands (completed, one commit, no replay); the read after it is
    its own result. If AppKit still answers the closed window's elements, the failing-read branch was not triggered:
    the case says so and is skipped (UNTESTED), never invented as a failure or a pass."""
    call, schema = host
    if _grants(call).get("accessibility") != "trusted":
        pytest.skip("unsupported here, not passed: Accessibility is not trusted")
    _new_interface(schema)
    target, front = launch("target"), launch("front")
    _hand_front(call, front)
    target.send("mode close")
    target.wait_for("mode_set", mode="close")
    before = target.mark("close_before")
    pressed = call("element_act", app=f"pid:{target.pid}", do="press", select={"identifier": "qa.commit"},
                   window=target.title, max_nodes=100, timeout_ms=5000, observe=True)
    closed = target.wait_for("window_closed", after=before["t"], timeout=5)
    time.sleep(0.5)
    after = target.mark("close_after")
    commits = _between(target.rows(), before["t"], after["t"], "commit")
    observation = pressed.get("observation") or {}
    _evidence(tmp_path, "close_after_commit.jsonl", {"closed": closed, "commits": len(commits),
                                                     "receipt_ok": pressed.get("ok"), "observation": observation})
    assert len(commits) == 1 and after["count"] == 1 and not closed["visible"], {"commits": commits, "closed": closed}
    assert pressed.get("ok") is True and pressed.get("dispatched") is True and pressed.get("action_outcome") == "completed", pressed
    if observation.get("ok") is True:
        pytest.skip("untriggered, not passed: AppKit still answered the closed window's elements "
                    f"(visited {observation.get('visited')}), so a failed read after the action did not occur")
    assert observation.get("ok") is False and observation.get("reason") and observation.get("error"), observation


@desktop_only
def test_screenshot_of_a_named_app_captures_the_display_holding_its_window(host, launch, tmp_path):
    """app= selects the display of the resolved application's window (selected_by app, with that app); an app= that
    resolves to no running application is refused and nothing is captured. One display proves the selection path,
    not multi-display correctness."""
    call, _schema = host
    grants = _grants(call)
    lacking = {k: grants.get(k, "absent") for k, v in (("accessibility", "trusted"), ("screen_recording", "granted"))
               if grants.get(k) != v}
    if lacking:
        pytest.skip(f"unsupported here, not passed: this process lacks {lacking} (no grant is requested)")
    target = launch("target")
    shot = call("screenshot", app=f"pid:{target.pid}", max_width=512, max_height=384)
    for key in ("path", "full_resolution_path"):
        if shot.get(key) and pathlib.Path(shot[key]).is_file():
            shutil.copy2(shot[key], tmp_path / f"screenshot-app-{key}.png")
    display = shot.get("display") or {}
    assert shot.get("ok") is True and display.get("selected_by") == "app" and (shot.get("app") or {}).get("pid") == target.pid, shot
    unknown = call("screenshot", app=f"ouroboros-qa-no-such-application-{uuid.uuid4().hex[:8]}")
    assert unknown.get("ok") is False and not unknown.get("path") and unknown.get("error"), unknown
    _evidence(tmp_path, "screenshot.jsonl", {"app": shot, "unknown": unknown})


TIMING_ARMS = {"B": "reference payload: press, then a separate ax_tree",
               "C2": "candidate payload: press, then a separate ax_tree",
               "C1": "candidate payload: press with observe=true"}


@desktop_only
def test_press_and_fresh_tree_timing_block(host, launch, tmp_path):
    """Repeatable, opt-in performance probe for future skill changes on one fixture: press Commit, then read its window.

    ``OUROBOROS_QA_TIMING_ARM`` selects the arm of every sample in this block (B, C2 or C1; B is meant to run with
    ``OUROBOROS_QA_SKILL_SEED_ROOT`` naming the earlier skill payload, so both payloads run on this core, harness,
    interpreter and fixture) and ``OUROBOROS_QA_TIMING_SAMPLES`` how many (default 4); without an arm, C2 and C1
    alternate C2,C1,C1,C2. Only arguments both payloads take are used (no window or text_limit on ax_tree; default
    text_limit). The fixture's commit events and the count node are the oracle. Every sample is kept, failures
    included, with the first use of the block marked; what is timed is ToolRegistry wall time per call and per step,
    never model rounds, process counts or significance. A non-improvement is a result, not a failure."""
    call, schema = host
    if _grants(call).get("accessibility") != "trusted":
        pytest.skip("unsupported here, not passed: Accessibility is not trusted")
    arm = os.environ.get("OUROBOROS_QA_TIMING_ARM", "")
    count_n = int(os.environ.get("OUROBOROS_QA_TIMING_SAMPLES") or 4)
    if arm:
        assert arm in TIMING_ARMS, f"OUROBOROS_QA_TIMING_ARM must be one of {sorted(TIMING_ARMS)}"
        sequence = [arm] * count_n
    else:
        sequence = ["C2", "C1", "C1", "C2"]
    if "B" in sequence:
        _interface(schema)
    else:
        _new_interface(schema)
    target, front = launch("target"), launch("front")
    if front.wait_for("pointer_monitor").get("installed") is not True:
        pytest.skip("pointer observation unavailable; no non-interference verdict")
    _hand_front(call, front)
    app = f"pid:{target.pid}"
    problems: list[str] = []
    setup = _act(call, target, "set_value", text="1500", identifier="qa.field")
    assert setup.get("ok") is True and setup.get("action_outcome") == "completed", setup
    t_before, f_before = target.mark("timing_before"), front.mark("timing_before")
    acted_id = _number(t_before, "target")
    if "B" not in sequence:
        refused = call("element_act", app=app, do="press", select={"identifier": "qa.commit"}, window=target.title,
                       text_limit=100000)
        if not (refused.get("ok") is False and refused.get("reason") == "bad_option" and refused.get("dispatched") is False):
            problems.append(f"text_limit without observe was not refused: {json.dumps(refused)[:1500]}")
        if target.mark("refusal_check")["count"] != 0:
            problems.append("the refused call committed")

    samples: list[dict] = []
    timing = tmp_path / "native_timing.jsonl"
    expected_count = 0
    for index, method in enumerate(sequence):
        failures: list[str] = []
        before = target.mark(f"{index}_before")
        start = time.perf_counter()
        receipt = call("element_act", app=app, do="press", select={"identifier": "qa.commit"}, window=target.title,
                       max_nodes=100, timeout_ms=5000, **({"observe": True} if method == "C1" else {}))
        action = dict(call.last)
        if method == "C1":
            tree, tree_call = receipt.get("observation") or {}, None
        else:
            if "observation" in receipt:
                failures.append("a plain press carried an observation")
            tree = call("ax_tree", app=app, max_nodes=100, timeout_ms=5000)
            tree_call = dict(call.last)
        step_ms = round((time.perf_counter() - start) * 1000, 3)
        expected_count += 1
        try:
            commit = target.wait_for("commit", after=before["t"], timeout=5)
        except AssertionError as exc:
            commit = {"missing": str(exc)}
        after = target.mark(f"{index}_after")
        commits = [(r["instance"], r["field"], r["count"]) for r in _between(target.rows(), before["t"], after["t"], "commit")]
        if not (receipt.get("ok") is True and receipt.get("dispatched") is True and receipt.get("action_outcome") == "completed"
                and (receipt.get("front_before") or {}).get("pid") == front.pid
                and (receipt.get("front_after") or {}).get("pid") == front.pid
                and (receipt.get("window") or {}).get("title") == target.title):
            failures.append(f"receipt: {json.dumps(receipt)[:1500]}")
        if commits != [(1, "1500", expected_count)] or commit.get("count") != expected_count:
            failures.append(f"commits {commits} (expected one with count {expected_count}); {commit}")
        if not (tree.get("ok") is True and not tree.get("truncated") and tree.get("truncated_by") == []):
            failures.append(f"tree: {json.dumps(tree)[:1500]}")
        title_seen = (tree.get("window") or {}).get("title") if method == "C1" else tree.get("window")
        if title_seen != target.title:
            failures.append(f"read window {title_seen!r}, not {target.title!r}")
        id_seen = (tree.get("window") or {}).get("window_id") if method == "C1" else tree.get("window_id")
        # Integer ids are compared where both sides have one; a missing id proves nothing (never None == None), so the
        # window read is then shown by the acted window's own sentinel and count below, and id selection is UNTESTED.
        ids = {"fixture": acted_id, "receipt": (receipt.get("window") or {}).get("window_id"), "read": id_seen}
        present = {k: v for k, v in ids.items() if isinstance(v, int) and not isinstance(v, bool)}
        if len(set(present.values())) > 1:
            failures.append(f"window ids differ: {ids}")
        id_selection = ("compared" if len(present) == len(ids) else
                        f"UNTESTED: no integer id from {sorted(set(ids) - set(present))}" + (" (compared the rest)" if len(present) > 1 else ""))
        sentinel = _values(tree, "qa.sentinel")
        if sentinel != [f"target:{expected_count}"] or after.get("sentinel") != f"target:{expected_count}":
            failures.append(f"sentinel read {sentinel}, fixture {after.get('sentinel')!r}, expected target:{expected_count}")
        if _values(tree, "qa.count") != [f"count:{expected_count}"] or _values(tree, "qa.status") != ["committed:1500"]:
            failures.append(f"tree count {_values(tree, 'qa.count')} status {_values(tree, 'qa.status')}")
        sample = {"kind": "sample", "arm": method, "index": index, "first_use": index == 0,
                  "action_wall_ms": action["wall_ms"], "tree_wall_ms": tree_call["wall_ms"] if tree_call else None,
                  "step_ms": step_ms, "action_bytes": action["bytes"], "tree_bytes": tree_call["bytes"] if tree_call else None,
                  "fixture_count": after["count"], "tree_count": _values(tree, "qa.count"),
                  "fixture_sentinel": after.get("sentinel"), "tree_sentinel": sentinel, "window_ids": ids,
                  "id_selection": id_selection,
                  "failures": failures}
        samples.append(sample)
        with timing.open("a", encoding="utf-8") as out:
            out.write(json.dumps(sample) + "\n")
        problems.extend(f"sample {index} {method}: {failure}" for failure in failures)

    time.sleep(0.5)  # catch a late duplicate press or activation before the final marks
    t_after, f_after = target.mark("timing_after"), front.mark("timing_after")
    start, end = min(t_before["t"], f_before["t"]), max(t_after["t"], f_after["t"])
    quiet = {
        "commits": len(_between(target.rows(), start, end, "commit")) == expected_count,
        "target untouched otherwise": not _between(target.rows(), start, end, "twin", "app_became_active", "window_became_key", "key_down"),
        "front kept": not _between(front.rows(), start, end, "app_resigned_active", "window_resigned_key", "workspace_activated", "pointer_event"),
        "front still in front, pointer unmoved": bool(f_after["app_active"] and f_after["front_pid"] == front.pid
                                                       and f_before["mouse"] == f_after["mouse"]),
        "pasteboard count unchanged": f_before["pasteboard_change_count"] == f_after["pasteboard_change_count"],
        "target state": t_after["count"] == expected_count and t_after["field"] == "1500",
    }
    problems.extend(f"block: {label}" for label, held in quiet.items() if not held)
    summary = {}
    for method in sorted(set(sequence)):
        rows = [s for s in samples if s["arm"] == method]
        later = [s["step_ms"] for s in rows if not s["first_use"]]
        summary[method] = {"meaning": TIMING_ARMS[method], "n": len(rows), "failed": sum(bool(s["failures"]) for s in rows),
                           "id_selection": sorted({s["id_selection"] for s in rows}),
                           "first_use_step_ms": next((s["step_ms"] for s in rows if s["first_use"]), None),
                           "later_step_ms_median": statistics.median(later) if later else None,
                           "later_step_ms_range": [min(later), max(later)] if later else None}
    with timing.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"kind": "summary", "sequence": sequence, "quiet": quiet, "arms": summary,
                              "tool_wall_only": True, "model_round_measured": False, "process_count_measured": False,
                              "significance": None}) + "\n")
    assert not problems, "\n".join(problems)

