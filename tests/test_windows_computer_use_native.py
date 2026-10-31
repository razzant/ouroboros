"""Native Windows qualification of the bundled ``windows_computer_use`` skill.

Opt-in: runs only on Windows with ``OUROBOROS_WINDOWS_DESKTOP_QA=1`` in an
interactive desktop session, and needs network access for the real isolated
``comtypes`` install. It drives the production path end to end - built-in install
and trusted acceptance, the startup dependency step, out-of-process extension load
and tool dispatch through ``ToolRegistry`` - against a WinForms window and a WPF
window this test creates in its own PowerShell process. Only those windows are
read or acted on, and only that process is stopped; nothing else on the desktop
is touched.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import uuid

import pytest

pytestmark = [
    pytest.mark.serial,
    pytest.mark.skipif(
        sys.platform != "win32" or os.environ.get("OUROBOROS_WINDOWS_DESKTOP_QA") != "1",
        reason="needs an interactive Windows desktop and OUROBOROS_WINDOWS_DESKTOP_QA=1",
    ),
]

REPO = pathlib.Path(__file__).resolve().parents[1]
SKILL = "windows_computer_use"
log = logging.getLogger(__name__)

FIXTURE = r"""
param([string]$Title)
Add-Type -AssemblyName System.Windows.Forms, PresentationFramework
$form = New-Object System.Windows.Forms.Form
$form.Text = $Title
$form.Width = 520; $form.Height = 420
function Add($control, $name, $x, $y) {
  $control.Name = $name; $control.Left = $x; $control.Top = $y; $form.Controls.Add($control); $control
}
$amount = Add (New-Object System.Windows.Forms.TextBox) 'amountBox' 10 10
$amount.AccessibleName = 'Amount'
$status = Add (New-Object System.Windows.Forms.Label) 'statusLabel' 10 40
$status.Width = 300; $status.Text = 'idle'
$apply = Add (New-Object System.Windows.Forms.Button) 'applyButton' 10 70
$apply.Text = 'Apply'
$apply.Add_Click({ $status.Text = 'applied:' + $amount.Text })
$agree = Add (New-Object System.Windows.Forms.CheckBox) 'agreeBox' 10 100
$agree.Text = 'Agree'
$kind = Add (New-Object System.Windows.Forms.ComboBox) 'kindBox' 10 130
[void]$kind.Items.AddRange(@('alpha', 'beta'))
$items = Add (New-Object System.Windows.Forms.ListBox) 'itemsList' 10 160
[void]$items.Items.AddRange(@('one', 'two', 'three'))
$twinA = Add (New-Object System.Windows.Forms.Button) 'twinA' 200 70
$twinA.Text = 'Twin'
$twinB = Add (New-Object System.Windows.Forms.Button) 'twinB' 300 70
$twinB.Text = 'Twin'
$freeze = Add (New-Object System.Windows.Forms.Button) 'freezeButton' 200 100
$freeze.Text = 'Freeze'
$freeze.Add_Click({ Start-Sleep -Seconds 8 })
# PowerShell 5.1 hosts WinForms with .NET Framework's legacy accessibility: the
# combo box above has no ExpandCollapse provider and no WinForms element offers
# ScrollItem. A WPF window of this same process carries the controls whose
# automation peers implement both.
$wpf = New-Object System.Windows.Window
$wpf.Title = $Title + '-wpf'
$wpf.Width = 320; $wpf.Height = 240
$wpf.WindowStartupLocation = 'Manual'; $wpf.Left = 560; $wpf.Top = 40
$panel = New-Object System.Windows.Controls.StackPanel
$shape = New-Object System.Windows.Controls.ComboBox
$shape.Name = 'shapeBox'
[void]$shape.Items.Add('circle'); [void]$shape.Items.Add('square')
$planets = New-Object System.Windows.Controls.ListBox
$planets.Name = 'planetList'; $planets.Height = 60
# Every item realized, so the one scrolled to is in the tree while out of view.
[System.Windows.Controls.VirtualizingStackPanel]::SetIsVirtualizing($planets, $false)
foreach ($planet in 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn') { [void]$planets.Items.Add($planet) }
[void]$panel.Children.Add($shape); [void]$panel.Children.Add($planets)
$wpf.Content = $panel
$wpf.Show()
[System.Windows.Forms.Application]::Run($form)
"""


@pytest.fixture
def owned_window(tmp_path):
    """A WinForms window in a PowerShell process this test started and alone stops."""
    from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs

    title = f"ouroboros-uia-qa-{uuid.uuid4().hex}"
    script = tmp_path / "fixture.ps1"
    script.write_text(FIXTURE, encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, test-owned process
        ["powershell.exe", "-NoProfile", "-Sta", "-ExecutionPolicy", "Bypass", "-File", str(script), title],
        **subprocess_new_group_kwargs(),
    )
    try:
        yield title, proc.pid
    finally:
        kill_process_tree(proc)
        proc.wait(timeout=30)


@pytest.fixture
def host(tmp_path, monkeypatch):
    """The production built-in install -> trust -> dependency -> out-of-process load path."""
    from ouroboros import extension_loader
    from ouroboros.skill_builtin import _dependency_pass, install_builtin_skill
    from ouroboros.skill_loader import find_skill, load_enabled
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from tests._shared import clean_extension_runtime_state

    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    data = tmp_path / "data"
    (data / "skills").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", data / "settings.json")
    clean_extension_runtime_state()
    assert install_builtin_skill("unix_computer_use", drive_root=data, seed_dir=REPO / "skills")["code"] == "other_platform"
    assert install_builtin_skill(SKILL, drive_root=data, seed_dir=REPO / "skills", turn_on_when_ready=True,
                                 installed_by="owner")["ok"]
    assert find_skill(data, "unix_computer_use") is None
    assert _dependency_pass(data, [SKILL], log) == [SKILL]
    assert load_enabled(data, SKILL) is True
    state = json.loads((data / "state" / "skills" / SKILL / "deps.json").read_text(encoding="utf-8"))
    assert state["status"] == "installed", state
    tool = extension_loader.get_tool(extension_loader.extension_surface_name(SKILL, "window_list"))
    assert tool is not None and tool["out_of_process"] is True
    registry = ToolRegistry(repo_dir=REPO, drive_root=data)
    registry.set_context(ToolContext(repo_dir=REPO, drive_root=data))

    def call(name, **args):
        raw = registry.execute(extension_loader.extension_surface_name(SKILL, name), args)
        head, sep, _tail = str(raw).partition("\n\n⚠️ ")
        return json.loads(head if sep else raw)

    try:
        yield data, call
    finally:
        clean_extension_runtime_state()


def _hwnd(call, title, pid):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for row in call("window_list", limit=500)["windows"]:
            if row["title"] == title and row["pid"] == pid:
                return int(row["hwnd"], 16)
        time.sleep(0.5)
    raise AssertionError(f"the owned fixture window {title!r} never appeared")


def test_the_skill_reads_and_acts_on_its_own_window_through_the_host(host, owned_window):
    data, call = host
    title, pid = owned_window
    hwnd = _hwnd(call, title, pid)

    reading = call("window_read", hwnd=hwnd, shape="flat", max_nodes=400)
    assert reading["backend"] == "uia", reading.get("fallback_reason")
    nodes = {node.get("automation_id"): node for node in reading["items"]}
    assert nodes["amountBox"]["role"] == "edit" and nodes["applyButton"]["role"] == "button"
    assert nodes["agreeBox"]["state"]["checked"] == "off"
    assert nodes["applyButton"]["rect"]["width"] > 0 and "unread" not in nodes["applyButton"]

    def act(select, do, *, on=hwnd, **extra):
        return call("element_act", hwnd=on, select=select, do=do, **extra)

    # Every step runs even after a failure, and each failure keeps the skill's
    # own answer, so one desktop run explains all of what went wrong.
    problems = []

    def check(label, condition, payload):
        if not condition:
            problems.append(f"{label}: {json.dumps(payload, ensure_ascii=False)[:1500]}")

    def done(label, select, do, **extra):
        result = act(select, do, **extra)
        check(label, result.get("ok") is True, result)
        return result

    done("set_value", {"automation_id": "amountBox"}, "set_value", text="1500")
    found = call("element_find", hwnd=hwnd, select={"automation_id": "amountBox", "value_contains": "1500"})
    check("value read back", found.get("matched") == 1, found)
    done("invoke", {"automation_id": "applyButton"}, "invoke")
    applied = call("element_find", hwnd=hwnd, select={"name_contains": "applied:1500"}, timeout_ms=5000)
    check("invoke effect", applied.get("ok") is True and applied.get("matched") == 1, applied)
    toggle = act({"automation_id": "agreeBox"}, "toggle")
    check("toggle", bool(toggle.get("element_after")), toggle)
    toggled = call("window_read", hwnd=hwnd, shape="flat", text_contains="agree")
    check("toggle effect", any(node.get("state", {}).get("checked") == "on" for node in toggled["items"]), toggled)
    done("select", {"role": "list_item", "name": "two"}, "select")
    try:
        wpf = _hwnd(call, f"{title}-wpf", pid)
    except AssertionError as absent:
        problems.append(str(absent))
    else:
        done("expand", {"automation_id": "shapeBox"}, "expand", on=wpf)
        done("collapse", {"automation_id": "shapeBox"}, "collapse", on=wpf)
        done("scroll_into_view", {"role": "list_item", "name": "saturn"}, "scroll_into_view", on=wpf)
    # The WinForms combo box and list item lack these patterns (run 39058404823):
    # refused by name, with the patterns they do have, never clicked instead.
    for verb, select, pattern in (("expand", {"automation_id": "kindBox"}, "ExpandCollapsePattern"),
                                  ("scroll_into_view", {"role": "list_item", "name": "three"}, "ScrollItemPattern")):
        refused = act(select, verb)
        check(f"{verb} refused", refused.get("ok") is False and f"does not support {pattern}; patterns it does "
              "support: " in str(refused.get("error")), refused)
    focus = act({"automation_id": "amountBox"}, "focus")
    check("focus is not mutating", focus.get("mutating") is False, focus)

    twins = act({"role": "button", "name": "Twin"}, "focus")
    check("ambiguity", twins.get("ok") is False and twins.get("reason") == "ambiguous" and twins.get("matched") == 2, twins)
    done("nth", {"role": "button", "name": "Twin", "nth": 1}, "focus")
    # timeout_ms=0: one whole walk under the worker grace, so this absence check
    # does not depend on where the default wait's last poll falls.
    missing = act({"name": "no such element"}, "focus", timeout_ms=0)
    check("not found", missing.get("reason") == "not_found" and missing.get("searched_whole_window") is True, missing)
    unsupported = act({"automation_id": "statusLabel"}, "invoke")
    check("unsupported", unsupported.get("ok") is False and "does not support InvokePattern" in str(unsupported.get("error")), unsupported)

    classic = call("window_read", hwnd=hwnd, backend="win32", max_nodes=50)
    check("win32 fallback", classic.get("backend") == "win32" and classic.get("nodes_read", 0) > 1, classic)
    assert not problems, "\n".join(problems)


def test_a_busy_window_answers_bounded_instead_of_hanging(host, owned_window):
    data, call = host
    title, pid = owned_window
    hwnd = _hwnd(call, title, pid)

    started = time.monotonic()
    pressed = call("element_act", hwnd=hwnd, select={"automation_id": "freezeButton"}, do="invoke", timeout_ms=1000)
    reading = call("window_read", hwnd=hwnd, timeout_ms=1000)
    elapsed = time.monotonic() - started

    # Either answer is honest; silence or an unbounded wait is not.
    assert pressed["ok"] is True or "did not answer" in pressed.get("error", "")
    assert reading["ok"] is True and (reading["backend"] == "win32" or reading["truncated"] or reading["elapsed_ms"] < 1000)
    assert elapsed < 60


def test_the_ids_this_skill_uses_match_the_installed_type_library(host):
    """The constants in win_uia are compared with the comtypes-generated module of
    this machine's UIAutomationCore type library, in the skill's own environment."""
    from ouroboros.marketplace.isolated_deps import python_runtime_binary

    data, _call = host
    skill_dir = data / "skills" / "native" / SKILL
    python = python_runtime_binary(skill_dir)
    assert python is not None
    probe = skill_dir.parent.parent.parent / "uia_constants_probe.py"
    probe.write_text(
        "import importlib.util, json, sys\n"
        "import comtypes, comtypes.client\n"
        "from comtypes import GUID\n"
        "comtypes.client.GetModule(comtypes.typeinfo.LoadRegTypeLib(GUID('{944DE083-8FB8-45CF-BCB7-C477ACB2F897}'), 1, 0, 0))\n"
        "import comtypes.gen.UIAutomationClient as gen\n"
        "source = open(sys.argv[1], encoding='utf-8').read()\n"
        "names = [line.split(' = ')[0] for line in source.splitlines() if line.startswith('UIA_') and ' = ' in line]\n"
        "print(json.dumps({name: getattr(gen, name, None) for name in names}))\n",
        encoding="utf-8",
    )
    result = subprocess.run([str(python), str(probe), str(skill_dir / "win_uia.py")], capture_output=True,
                            text=True, timeout=120, check=True)
    library = json.loads(result.stdout)
    import tests._windows_computer_use  # noqa: F401
    from windows_computer_use import win_uia

    assert library and {name: getattr(win_uia, name) for name in library} == library
