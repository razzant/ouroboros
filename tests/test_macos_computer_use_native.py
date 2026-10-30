"""Native macOS qualification of quiet ``element_act`` in the bundled ``unix_computer_use`` skill.

Opt-in: runs only on macOS with ``OUROBOROS_MACOS_DESKTOP_QA=1`` in a logged-in desktop session
whose owner keeps the keyboard and pointer still for the run (about half a minute). It compiles the
AppKit fixture ``tests/fixtures/macos_computer_use_qa.swift`` into this test's temporary directory and
starts two windows in two processes it alone owns and stops: a target with a field, Commit, a counter
and two buttons named "Twin", and a front app that receives synthetic keystrokes. Through the production
path - built-in install, out-of-process load and dispatch through ``ToolRegistry`` - the skill presses and
sets a value in the background target while the front app is typed into. Each fixture logs its own
state, every activation the workspace announces (a focus taken and given back between two samples
still shows), the key events it receives, pointer events and the clipboard's change count (never its
contents); the skill's answers, kept in ``receipts.jsonl``, are checked against those logs, not against
themselves. Three cases use foreground input: one hands the front to one fixture and lets the other take
it (targeted ``key``/``type_text`` with ``raise_policy="never"`` must refuse rather than take it back), one
sends Command-A spelled ``cmd+a`` and ``cmd+A`` and reads the modifiers the front app received, and one
clicks Commit at a point read off a downscaled screenshot. The calls run as an owner task's would
(``work_kind`` owner). A grant the run lacks becomes the skip reason, never a pass.
``test_the_fixture_builds`` alone runs without the opt-in: it compiles and starts nothing.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "tests" / "fixtures" / "macos_computer_use_qa.swift"
SKILL = "unix_computer_use"
DESKTOP = sys.platform == "darwin" and os.environ.get("OUROBOROS_MACOS_DESKTOP_QA") == "1"
desktop_only = pytest.mark.skipif(not DESKTOP, reason="needs a logged-in macOS desktop and OUROBOROS_MACOS_DESKTOP_QA=1")
log = logging.getLogger(__name__)

# The registered public schema: process and window scope sit outside the element selector.
ACT_ARGS = {"app", "do", "select", "text", "window"}
SELECT_KEYS = {"role", "label", "identifier", "nth"}
# About ten seconds of keystrokes at 50 ms: long enough to span every action below (checked, not assumed).
TYPED = "quietlyactingonthebackgroundtarget" * 6


def _build(out_dir: pathlib.Path) -> pathlib.Path:
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
    assert install_builtin_skill(SKILL, drive_root=data, seed_dir=REPO / "skills", turn_on_when_ready=True,
                                 installed_by="owner")["ok"]
    _dependency_pass(data, [SKILL], log)
    assert load_enabled(data, SKILL) is True
    registry = ToolRegistry(repo_dir=REPO, drive_root=data)
    # An owner task's context. Work of unknown provenance counts as consciousness work, whose full Safety
    # Supervisor coverage holds every action while no safety model is reachable, which a hermetic run never has.
    registry.set_context(ToolContext(repo_dir=REPO, drive_root=data, task_metadata={"work_kind": "owner"}))

    receipts = tmp_path / "receipts.jsonl"  # every answer as received, on the fixtures' clock (system uptime)

    def call(name, **args):
        raw = str(registry.execute(extension_loader.extension_surface_name(SKILL, name), args))
        with receipts.open("a", encoding="utf-8") as out:
            row = {"t": time.monotonic(), "tool": name, "args": args, "answer": raw}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
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


def _act(call, target: Owned, do: str, *, text: str | None = None, **select) -> dict:
    args = {"app": f"pid:{target.pid}", "do": do, "select": select, "window": target.title}
    if text is not None:
        args["text"] = text
    return call("element_act", **args)


def _between(rows: list[dict], start: float, end: float, *events: str) -> list[dict]:
    return [row for row in rows if start < row["t"] < end and row["event"] in events]


def _input_ready(call, **also: str) -> None:
    """Foreground input needs the grants to post it and cliclick to send it; without them the case is skipped."""
    grants = _grants(call)
    required = {"accessibility": "trusted", "post_events": "granted", **also}
    lacking = {key: grants.get(key, "absent") for key, value in required.items() if grants.get(key) != value}
    if lacking:
        pytest.skip(f"unsupported here, not passed: this process lacks {lacking} (no grant is requested)")
    if (call("capabilities").get("input") or {}).get("cliclick") is not True:
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
              and (result.get("front_after") or {}).get("pid") == front.pid, result)

    t_before, f_before = target.mark("before"), front.mark("before")
    front.send(f"type 50 {TYPED}")
    started = front.wait_for("typing_started", "typing_unsupported", after=f_before["t"])
    if started["event"] == "typing_unsupported":
        pytest.skip(f"unsupported here, not passed: the fixture cannot post keystrokes ({started['reason']})")
    front.wait_for("front_text", after=started["t"])  # keystrokes are landing before the first action
    acting = front.mark("acting")
    quiet("set_value", _act(call, target, "set_value", text="1500", identifier="qa.field"))
    quiet("press", _act(call, target, "press", identifier="qa.commit"))
    twins = _act(call, target, "press", label="Twin")
    check("ambiguous", twins.get("ok") is False and twins.get("reason") == "element_ambiguous"
          and twins.get("dispatched") is False, twins)
    quiet("nth", _act(call, target, "press", label="Twin", nth=2))
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
    menu that acts on it); no quit or logout key is ever sent."""
    call, _schema = host
    _input_ready(call)
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
        if (row["chars_ignoring"], row["modifiers"]) != ("a", ["cmd"]):
            problems.append(f"{keys} arrived as {row['modifiers']} + {row['chars_ignoring']!r}, not Command-A: "
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
