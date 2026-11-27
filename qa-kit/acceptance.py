"""Packaging acceptance on Windows: what ran, and what only a person or a clean runner can say.

Two legs, each on its OWN disposable Windows runner and profile (the workflow runs them side by side), so the first is really clean:
  visible    a clean profile without system Git: the real installer wizard is opened as a person opens it (/LANG=en) and driven only
             through UI Automation InvokePattern (Install, Finish), inside the windows of the installer's own process tree; frames of its
             pages; the app's first-run wizard, owned by exactly the installed Ouroboros.exe; which Git ran.
  migration  the former user: the PUBLISHED old release ZIP (pinned bytes) creates the profile and a chat with the system Git it needs
             (its public loopback onboarding and Main chat APIs, normal Quit) -> the new package through the SAME ordinary installer wizard; only its Finish launches the
             app, and the window it opens must belong to the installed copy and show the former exchange (the Start shortcut is never
             used to make up for it) -> first in-app Update, from whichever known revision actually serves and only from the local
             mirror -> normal Quit -> cold start through the Start-menu .lnk, the exchange read again -> normal Quit -> the installer's
             own quiet uninstall -> the profile is byte-identical and the new ZIP copy opens it, the exchange read in its window.
             Every window is recorded with whose code it was: the process that owns it, and the body that served it. The official
             address is read from the local mirror by environment variables of this run's own launches only (no Git config file is
             written), from before the old release starts to the end of the leg; Git confirms that offline before the old release is
             launched, and again, for the profile's own configured source, before Update.

NO PHYSICAL INPUT. #4220's fallback helpers are replaced here by a refusal (global_input_forbidden, unconfirmed). The installer's
wizard stays InvokePattern only. WM_CLOSE is sent only to an addressed window of the app we started (its normal close gesture).
The migration seeds the former profile through v7.12.0's documented completion endpoint and an ordinary Main socket turn, not a
settings or database fixture. Nothing about this is evidence of using the old wizard's controls.

It subclasses #4220's ``scenario.QualificationRun`` (vendor/, unmodified) for the host checks, the first-run wizard, the chat turns,
the Update flow and the Git traces. Pixels are never judged: a frame is "pending_visual_inspection" or "unknown".
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from typing import Any, Callable

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "vendor" / "windows_update_4220_e21fee6")]

import chat_reply_witness as cw  # noqa: E402
import fixture_model as fm  # noqa: E402
import git_traces as gt  # noqa: E402
import host_win as hw  # noqa: E402
import judge as jd  # noqa: E402
import prepare_build as pb  # noqa: E402
import scenario as sc  # noqa: E402
import windows_host as wh  # noqa: E402
from uia_reading import flatten_tree_nodes, visible_rect  # noqa: E402

BRAND_PS, EXPLORER_PS = hw.PS_DIR / "brand-probe.ps1", hw.PS_DIR / "capture-frames.ps1"
CONSENT = "Keep Ouroboros running?"  # launcher_background.CONSENT_TITLE: the one-time question a first close asks
SETUP_TITLE = "Ouroboros \u2014 Setup"  # launcher_onboarding: the app's own first-run wizard window
BOOT_SEC, QUIT_SEC = 300.0, 150.0
APP_DEFAULT_SOURCE = "https://github.com/razzant/ouroboros"  # supervisor/git_ops.OFFICIAL_UPDATE_REMOTE_URL: used when no source is configured
MAX_FRAME_BYTES = 3 * 1024 * 1024  # Invoke-Acceptance.ps1 publishes no larger PNG ($_.Length -le 3MB): such a frame is not available
INSTALLED_HANDOFF = ("after_update_handoff", "cold_start_shortcut")  # the installed copy's window is the bundled Python's here
INSTALLED_STATES = ("installed_first_launch", *INSTALLED_HANDOFF)  # the contract's AppUserModel.ID is the installed copy's
WIZARD_FRAMES = ("window_wizard_ready", "window_wizard_progress", "window_wizard_finish")
FRAMES = {"migration": ("explorer_installer", "explorer_installed_exe", "explorer_start_shortcut", "caption_installed_first_launch",
                        "caption_after_update_handoff", "caption_cold_start_shortcut", "window_zip_reopen"),
          "visible": ("explorer_installer", "explorer_installed_exe", "explorer_start_shortcut", *WIZARD_FRAMES,
                      "window_visible_first_launch", "caption_visible_first_launch")}
WIZARD_TITLE = re.compile(r"^Setup - ", re.IGNORECASE)  # Inno's wizard under /LANG=en: "Setup - Ouroboros"
PRESSABLE = (("finish", "finish"), ("install", "ready"), ("next >", "next"))  # the only buttons ever invoked, by their standard captions
MAX_NEXT = 4
FIXTURE: dict[str, Any] = {}


def forbid_global_input() -> None:
    """Replace every physical-input helper of the vendored #4220 code by a refusal (idempotent). They are the fallbacks behind
    act_or_click(), reveal_wizard_control(), set_exact_text() and the chat reply witness's right-click; with them gone, a step that
    needs one raises Unread and ends unconfirmed."""
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise wh.Unread("global_input_forbidden")

    for name in ("click", "wheel", "select_all_and_replace_text", "_send_inputs"):
        setattr(wh, name, refuse)
    cw.reply_from_visible_leaf = lambda *_a, **_k: False  # no witness through a context menu; the UIA leaf reading stands alone


forbid_global_input()
BaseDriver = sc.UIADesktopDriver  # #4220's UIA driver as it is, kept for the installer's windows


class AdoptedProcess:
    """A launcher this script did not spawn, found through the installation's own lock file."""

    def __init__(self, pid: int) -> None:
        self.pid = int(pid)

    def poll(self) -> int | None:
        return None if hw.pid_alive(self.pid) else 0


class HandoffAwareDriver(BaseDriver):
    """#4220's UIA driver, following the launcher through a source handoff: the lock file names whoever holds the installation,
    a live new holder is adopted, and a launcher that ends is given time to be replaced."""

    app_root: pathlib.Path | None = None

    def _follow(self) -> None:
        pid = hw.lock_pid(type(self).app_root)
        if pid and pid != self.proc.pid and hw.pid_alive(pid):
            self.proc = AdoptedProcess(pid)

    def ensure_alive(self, phase: str) -> None:
        self._follow()
        deadline = time.monotonic() + 240.0
        while self.proc.poll() is not None:
            if time.monotonic() > deadline:
                raise sc.Failure(phase, "launcher_exited_unexpectedly")
            time.sleep(0.5)
            self._follow()


def fixture_b(*, harness_git: str, mirror: pathlib.Path, qa_root: pathlib.Path, bundle: pathlib.Path, candidate: str,
              doc_rel_path: str, marker: str) -> tuple[str, list[str], str]:
    """#4220's ``create_fixture_mirror_b`` with one more change in commit B: a comment appended to ``launcher.py``. Its B added only a
    docs file, which the launcher never loads, so an Update to it never hands off to the source launcher. Everything stays in the local
    bare mirror of the runner; its one push goes to that mirror."""
    env = pb._env()  # noqa: SLF001

    def git(*a: str, raw: bool = False) -> Any:
        done = subprocess.run([harness_git, *a], capture_output=True, text=not raw, env=env, timeout=300)
        if done.returncode:
            raise pb.Refusal("fixture_git_failed", subcommand=pb._subcommand(a), exit_code=done.returncode)  # noqa: SLF001
        return done.stdout if raw else done.stdout.strip()

    git("init", "--bare", str(mirror))
    for spec in ("HEAD:refs/heads/ouroboros", "+refs/tags/*:refs/tags/*"):
        git("-C", str(mirror), "fetch", "--no-tags", str(bundle), spec)
    for branch in ("ouroboros-stable", "main"):
        git("-C", str(mirror), "update-ref", f"refs/heads/{branch}", candidate)
    git("-C", str(mirror), "symbolic-ref", "HEAD", "refs/heads/ouroboros")
    tags = git("-C", str(mirror), "for-each-ref", "--format=%(refname)", "refs/tags").splitlines()
    work = qa_root / "fixture-work"
    git("clone", "--branch", "ouroboros", str(mirror), str(work))
    (work / doc_rel_path).parent.mkdir(parents=True, exist_ok=True)
    (work / doc_rel_path).write_text(f"# Packaging acceptance fixture\n\nMarker: {marker}\n", encoding="utf-8")
    if not (work / "launcher.py").is_file():
        raise pb.Refusal("fixture_launcher_py_missing")
    (work / "launcher.py").write_bytes((work / "launcher.py").read_bytes() + f"\n# QA fixture B handoff marker: {marker}\n".encode())
    git("-C", str(work), "add", doc_rel_path, "launcher.py")
    git("-C", str(work), *pb.FIXTURE_IDENTITY, "commit", "-q", "-m", "QA fixture B (synthetic local mirror only): a doc and a launcher.py comment")
    b_sha, parent = git("-C", str(work), "rev-parse", "HEAD"), git("-C", str(work), "rev-parse", "HEAD^")
    blob = git("-C", str(work), "cat-file", "blob", "HEAD:launcher.py", raw=True)
    changed = blob != git("-C", str(work), "cat-file", "blob", "HEAD^:launcher.py", raw=True)
    git("-C", str(work), "push", "-q", "origin", "HEAD:refs/heads/ouroboros")
    FIXTURE.update(marker=marker, changed=changed, launcher_sha=hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest())
    return b_sha, tags, parent


def same_address(a: Any, b: Any) -> bool:
    """One Git address, whatever the slashes and the case of a Windows path."""
    norm = lambda s: str(s).replace("\\", "/").rstrip("/").lower()  # noqa: E731
    return norm(a) == norm(b)


def manifest(root: pathlib.Path) -> dict[str, Any]:
    """Every regular file under ``root`` with size and SHA-256; what could not be read is listed, never skipped."""
    files: dict[str, list[Any]] = {}
    unreadable: list[str] = []
    if not root.exists():
        return {"files": files, "unreadable": unreadable}
    for folder, _dirs, names in os.walk(root, onerror=lambda error: unreadable.append(type(error).__name__)):  # a folder not listed is not "empty"
        for name in names:
            path = pathlib.Path(folder, name)
            try:
                files[path.relative_to(root).as_posix()] = [path.stat().st_size, wh.sha256_file(path)]
            except OSError:
                unreadable.append(name)
    return {"files": files, "unreadable": unreadable}


def static_tool_dirs(environ: Any = os.environ) -> list[str]:
    """The places the app adds to its PATH on Windows (ouroboros/platform_layer.common_tool_directories on the base branch), for a host
    without the candidate's source: Git\\cmd under ProgramFiles and under %LOCALAPPDATA%\\Programs, nodejs, npm, cargo. A mirror can drift;
    --candidate-root uses the candidate's own function, and the Git trace identity decides afterwards whichever list was used."""
    layout = (("ProgramFiles", ("Git", "cmd")), ("ProgramFiles", ("nodejs",)), ("LOCALAPPDATA", ("Programs", "Git", "cmd")),
              ("APPDATA", ("npm",)), ("USERPROFILE", (".cargo", "bin")))
    out = []
    for name, parts in layout:
        base = environ.get(name) or (r"C:\Program Files" if name == "ProgramFiles" else "")
        if base:
            out.append(str(pathlib.PureWindowsPath(base).joinpath(*parts)))
    return out


PATH_TOKEN = re.compile(r"(?i)(?:\b[a-z]:[\\/]|\\\\[A-Za-z0-9_.$-]+\\)[^\"<>|\r\n]*")  # C:\... or \\server\share\..., spaces included
USER_SEGMENT = re.compile(r"(?i)([\\/]Users[\\/])[^\\/\"<>|\r\n]+")


def strip_paths(text: str) -> str:
    """A path is replaced to the end of its quoted or unquoted run, spaces included (#4220's scrub only hides the drive prefix, which
    keeps a user name): losing the words after a path costs less than keeping half a profile name. A ``Users\name`` segment that
    survives in a relative path loses its name too."""
    return USER_SEGMENT.sub(r"\1<user>", PATH_TOKEN.sub("<path>", text))


def deep_scrub(value: Any) -> Any:
    """Every string in a published payload without a host path or a user name: the privacy guard then never has to stop a result."""
    if isinstance(value, dict):
        return {deep_scrub(k): deep_scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [deep_scrub(v) for v in value]
    return pb._scrub_text(strip_paths(value), 600) if isinstance(value, str) else value  # noqa: SLF001


def withhold_text(value: Any) -> Any:
    """Identifiers, numbers and fixed words pass; any other string (a message, a caption, a path) is withheld."""
    if isinstance(value, dict):
        return {k: withhold_text(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [withhold_text(v) for v in value]
    return value if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:+-]{0,80}", value) else "<withheld>"


# ---- the installer's wizard: UI Automation InvokePattern only, inside the installer's own process tree ----

def caption(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name or "").replace("&", "")).strip().lower()


def wizard_stage(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Which page the wizard shows, from its visible UIA nodes alone: a standard button that can be pressed (Finish, Install, Next >), a
    progress bar, or none of them ("other"). A node UIA did not read as on screen is not visible."""
    shown = [n for n in nodes if visible_rect(n) is not None]
    buttons: dict[str, dict[str, Any]] = {}
    for n in shown:
        if n.get("role") == "button" or "button" in str(n.get("class") or "").lower():
            buttons.setdefault(caption(n.get("name")), n)
    for want, stage in PRESSABLE:
        node = buttons.get(want)
        if node is not None and (node.get("state") or {}).get("enabled") is not False:
            return {"stage": stage, "node": node}
    if any("progress" in f"{n.get('role') or ''} {n.get('class') or ''}".lower() for n in shown):
        return {"stage": "progress", "node": None}
    return {"stage": "other", "node": None}


def page_signature(nodes: list[dict[str, Any]]) -> int:
    """The visible content of the page as one number, for equality inside this process only (never stored): consecutive "Next >" pages
    look alike by stage, and only their content tells that a press moved the wizard on."""
    return hash(tuple(sorted((str(n.get("role") or ""), str(n.get("name") or "")[:60]) for n in nodes if visible_rect(n) is not None)))


def note_buttons(out: dict[str, Any], nodes: list[dict[str, Any]]) -> None:
    """What the wizard's own buttons looked like to UIA, as fixed words only (diagnosis for the first run; never a path or a label)."""
    wanted = {want for want, _stage in PRESSABLE} | {"cancel", "back"}
    for n in nodes:
        word = caption(n.get("name"))
        if word in wanted:
            row = {"name": word, "role": str(n.get("role") or "")[:24], "class": str(n.get("class") or "")[:24],
                   "enabled": (n.get("state") or {}).get("enabled"), "on_screen": visible_rect(n) is not None}
            if row not in out["buttons_seen"] and len(out["buttons_seen"]) < 12:
                out["buttons_seen"].append(row)


def patterns_in(error: Any) -> list[str]:
    """The pattern names the skill lists when it refuses an invoke ('patterns it does support: A, B.'): fixed UIA words."""
    found = re.search(r"patterns it does support: ([A-Za-z, ]{0,160})", str(error or ""))
    return [w.strip() for w in found.group(1).split(",") if w.strip()][:10] if found else []


def drive_installer(root_pid: int, *, pids_of: Callable[[int], set[int] | None], windows_of: Callable[[int], list[Any]],
                    read: Callable[[Any], tuple[list[dict[str, Any]] | None, str | None]], invoke: Callable[[Any, dict[str, Any]], dict[str, Any]],
                    capture: Callable[[str, Any], bool], exit_code: Callable[[], int | None], reject: Callable[[str], None] = lambda _key: None,
                    before_finish: Callable[[], None] = lambda: None, sleep: Callable[[float], None] = time.sleep,
                    now: Callable[[], float] = time.monotonic, timeout: float = 1800.0, step: float = 1.5, settle: float = 25.0,
                    stall: float = 150.0) -> dict[str, Any]:
    """Drive the real installer wizard to its end with UI Automation InvokePattern, and report only what was seen.

    Scope: the windows of ``root_pid``'s own process tree (the extracted child owns the wizard). Input: ``invoke`` on the exact element of
    Finish, Install or Next > (UIA InvokePattern; the skill itself refuses a coordinate fallback). A page is "seen" when its controls were
    read; a frame counts only if the page was unchanged across the capture (a progress page: still the progress page; its text moves),
    and one that does not count is handed to ``reject`` so it is not kept; a press counts only if the page then changed. When invoke
    is unavailable the run stops there with the patterns UIA does offer: an exact skip, never another way to press."""
    out: dict[str, Any] = {"wizard_found": False, "stages_seen": [], "frames": {}, "pressed": [], "extra_pages": 0, "stopped": None,
                           "exit_code": None, "buttons_seen": [], "uia_problem": None, "invoke": None, "polls": 0}
    begun = now()
    deadline = begun + timeout
    prev, prev_sig, last_change, pending, finish_at = None, None, begun, None, None
    unread_since = None
    while now() < deadline:
        out["polls"] += 1
        code = exit_code()
        tree = pids_of(root_pid)
        wins = [w for pid in sorted({root_pid} | (tree or set())) for w in windows_of(pid)]
        wizard = [w for w in wins if WIZARD_TITLE.match(w.title)]
        if finish_at is not None:  # Finish was invoked: only the installer's exit is left to see
            if code is not None:
                out["exit_code"] = code
                return out
            if now() - finish_at > 180.0:
                out["stopped"] = "installer_did_not_exit_after_finish"
                return out
            sleep(step)
            continue
        if code is not None and not wizard:
            out["exit_code"], out["stopped"] = code, out["stopped"] or ("installer_exited_without_a_wizard" if not out["wizard_found"] else "installer_exited_early")
            return out
        if len(wizard) > 1 or (wizard and len(wins) > len(wizard)):
            out["stopped"] = "unexpected_window"  # a dialog or a second window: nothing is pressed, nothing is guessed
            return out
        if not wizard:
            if not out["wizard_found"] and now() - begun > 120.0:
                out["stopped"] = "no_wizard_window"
                return out
            sleep(step)
            continue
        win = wizard[0]
        out["wizard_found"] = True
        nodes, problem = read(win)
        if nodes is None:
            out["uia_problem"] = problem
            unread_since = now() if unread_since is None else unread_since
            if now() - unread_since > stall:  # unreadable all that time: one failed read during a long install is not this
                out["stopped"] = "uia_unreadable"
                return out
            sleep(step)
            continue
        unread_since = None
        st = wizard_stage(nodes)
        name = st["stage"]
        note_buttons(out, nodes)
        sig = page_signature(nodes)
        if name != prev:
            prev, last_change = name, now()
        if sig != prev_sig:  # any change of what the page shows: the press (if any) took effect
            prev_sig, pending = sig, None
        key = name if name != "next" else f"next_{out['extra_pages'] + 1}"
        frame = out["frames"].get(key)
        if name != "other" and (frame is None or (not frame["captured"] and frame["tries"] < 3)):
            if name not in out["stages_seen"]:
                out["stages_seen"].append(name)
            took = bool(capture(key, win))
            after, _problem = read(win)
            same = after is not None and wizard_stage(after)["stage"] == name and (name == "progress" or page_signature(after) == sig)
            if not (took and same):
                reject(key)  # whatever was written for it shows another page, or nothing: it must not count
            out["frames"][key] = {"captured": took and same, "tries": (frame or {}).get("tries", 0) + 1,
                                  **({} if same else {"note": "the page changed during the capture"})}
            continue  # decide on a fresh reading: the page may have moved on while the frame was taken
        if name == "other" and now() - last_change > stall:
            out["stopped"] = "stalled_on_unknown_page"
            return out
        node = st["node"]
        if node is None:
            sleep(step)
            continue
        if pending is not None and pending[0] == name:
            if now() - pending[1] > settle:
                out["stopped"] = "invoke_had_no_effect"  # pressed once, the page did not change: never pressed twice
                return out
            sleep(step)
            continue
        if name == "next" and out["extra_pages"] >= MAX_NEXT:
            out["stopped"] = "too_many_pages"
            return out
        if name == "finish":
            before_finish()  # copying has ended and nothing the installer would launch has started yet
        res = invoke(win, node)
        if not res.get("ok"):
            out["stopped"] = "invoke_unavailable"
            word = str(res.get("reason") or "")
            out["invoke"] = {"stage": name, "patterns_offered": patterns_in(res.get("error")),  # fixed words only, never the skill's own text
                             "reason": "pattern_missing" if patterns_in(res.get("error")) else word if re.fullmatch(r"[a-z_]{1,40}", word) else "refused"}
            return out
        out["pressed"].append(name)
        pending = (name, now())
        out["extra_pages"] += name == "next"
        if name == "finish":
            finish_at = now()
    out["stopped"] = "timeout"
    return out


def profile_http(port: int, path: str, body: dict[str, Any] | None = None) -> tuple[int | None, dict[str, Any]]:
    """Only the caller's verified loopback listener. No proxy, redirect, retry or response text in evidence."""
    addr = f"127.0.0.1:{port}"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.request("GET" if body is None else "POST", path, None if body is None else json.dumps(body),
                     {"Origin": f"http://{addr}", "Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read((1 << 20) + 1)
        data = json.loads(raw) if len(raw) <= 1 << 20 else None
        return response.status, data if isinstance(data, dict) else {}
    except (OSError, ValueError, http.client.HTTPException):
        return None, {}  # a POST may have landed; use only that attempt's readback below
    finally:
        conn.close()


def send_old_main_turn(port: int, prompt: str, *, timeout: float = 45.0) -> bool:
    """One ordinary Main message over the public /ws, no retries; only our correlated owner echo settles admission."""
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    addr, message_id = f"127.0.0.1:{port}", f"qa-{uuid.uuid4().hex}"
    try:
        with connect(f"ws://{addr}/ws", origin=f"http://{addr}", proxy=None, open_timeout=10, close_timeout=2) as socket:
            socket.send(json.dumps({"type": "chat", "content": prompt, "chat_id": 1, "client_message_id": message_id}))
            deadline = time.monotonic() + timeout
            while (left := deadline - time.monotonic()) > 0:
                row = json.loads(socket.recv(timeout=left))
                if (isinstance(row, dict) and row.get("type") == "chat" and row.get("role") == "user" and row.get("chat_id") == 1
                        and row.get("content") == prompt and row.get("client_message_id") == message_id and row.get("ingress_accepted") is True):
                    return True
    except (OSError, ValueError, WebSocketException):
        return False
    return False


class AcceptanceRun(sc.QualificationRun):
    def __init__(self, spec: dict[str, Any], args: argparse.Namespace, *, app_root_override: pathlib.Path | None = None) -> None:
        forbid_global_input()
        sc.UIADesktopDriver = HandoffAwareDriver  # type: ignore[misc]
        pb.create_fixture_mirror_b = fixture_b  # type: ignore[assignment]
        super().__init__(argparse.Namespace(candidate=spec["candidate_sha"], archive=spec["zip"]["path"], archive_sha256=spec["zip"]["sha256"],
                                            candidate_src=spec["candidate_root"], evidence=args.evidence, mirror_url=spec["mirror_url"], tag=args.tag),
                         app_root_override=app_root_override)  # a host's profile is ~/Ouroboros; only a test passes another
        self.cli, self.inst = args, spec["installer"]
        self.candidate_sha = self.candidate  # self.candidate is #4220's "A": the migration points it at the old release while that serves
        self.old, self.old_root = spec.get("old_release") or {}, None
        self.result["declared"].update(ui_driver="candidate_windows_computer_use_uia_only_no_physical_input",  # collected: counted at save
                                       screenshots_policy="addressed_captures_of_this_runs_own_windows_cut_to_the_window_or_row",
                                       code_signature_checked=False, smartscreen_tested=False)
        if args.mode == "migration":  # the old release's Git trace sinks exist only on the host, unjudged
            self.result["declared"].update(git_observation_sink="old_release_launch_only_local_files_unjudged_unpublished",
                                           git_identity_claim="not_claimed_in_this_leg",
                                           update_source_redirect="launch_environment_only_not_the_explorer_start" if self.by_explorer()
                                           else "launch_environment_only_for_the_whole_leg",
                                           old_profile_preparation="published_old_app_loopback_onboarding_and_main_chat")
        self.payload_root = pathlib.Path(spec["payload_root"]) if spec.get("payload_root") else None
        self.installer = pathlib.Path(self.inst["path"])
        self.install_dir = pathlib.Path(os.path.expandvars(self.inst["install_dir"]))
        self.lnk = pathlib.Path(os.path.expandvars(self.inst["start_menu_shortcut"]))
        self.private, self.frames_dir = self.evidence / "private", self.evidence / "frames"
        self.private.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        HandoffAwareDriver.app_root = self.app_root
        self.sinks = {"portable": (self.trace1, self.trace2_dir),
                      "installed": (self.qa_root / "git-trace1-installed.log", self.qa_root / "git-trace2-installed")}  # judged in the visible leg
        self.obs: dict[str, Any] = {}
        self.frames: dict[str, Any] = {}
        self.digests: dict[str, str] = {}
        self.cu: Any = None
        self.first_pid = 0
        self.token =f"QA-PROFILE-TOKEN-{fm._clean_tag(args.tag)}"  # noqa: SLF001
        self.result["facts"].update(observed=self.obs, frames=self.frames, identity={}, git={}, profile={}, route=self.opt("launch_route", "shellexecute"),
                                    mode=args.mode, physical_input="none: UIA verbs and addressed WM_CLOSE only",
                                    contract="read from the installer's source (ouroboros.iss, pack_windows_setup.ps1); not confirmed on Windows",
                                    acceptance_run_started=True, launch_attempts=[], launches_observed=[])
        self.held_before_finish: bool | None = None
        self.plain_env: dict[str, str] = {}  # the launch environment before the mirror redirect (set with it)

    def opt(self, name: str, default: Any = False) -> Any:
        return getattr(self.cli, name, default)

    def save(self, name: str, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("reasons"), list):  # #4220's outcome fields may hold a skill's or a control's own text
            payload = {**payload, "reasons": [{**r, **({"fields": withhold_text(r["fields"])} if r.get("fields") else {})} for r in payload["reasons"]]}
        if name == "result.json":  # what was actually kept, at the moment of writing
            kept = sum(f.get("state") == "pending_visual_inspection" for f in self.frames.values())
            payload = {**payload, "declared": {**(payload.get("declared") or {}), "screenshots_collected": kept > 0},
                       "facts": {**(payload.get("facts") or {}), "frames_kept": kept}}
        super().save(name, deep_scrub(payload))

    def asked(self, what: str) -> None:
        """A launch this run is about to ask for (a program, or the installer's Finish). Asking is not seeing it start."""
        self.result["facts"]["launch_attempts"].append(what)

    def saw(self, what: str) -> None:
        """A launch confirmed by what appeared: a window of it, the profile's lock taken, or the program's own exit."""
        self.result["facts"]["launches_observed"].append(what)

    # ---- recording: nothing defaults to pass
    def observe(self, name: str, state: str, *, required: bool = True, stop: bool = True, unknown_stops: bool | None = None, **detail: Any) -> None:
        """One decision. A known contrary fact (fail) stops the run; so does a required unknown, unless ``stop`` is False (the final gate
        decides) or ``unknown_stops`` is False (a process that could not be identified is unknown, not a defect: the run goes on and
        gathers the independent evidence, and the gate still keeps it from passing)."""
        self.obs[name] = {"state": state, "required": required, **{k: pb._scalar(v) for k, v in detail.items()}}  # noqa: SLF001
        if stop and state == "fail":
            raise sc.Failure(self.step, name, **detail)
        if (stop if unknown_stops is None else unknown_stops) and required and state == "unknown":
            raise sc.Unconfirmed(self.step, name, **detail)

    def required_frames(self) -> tuple[str, ...]:
        return FRAMES[self.cli.mode]

    def gate(self) -> None:
        failed = [n for n, o in self.obs.items() if o["state"] == "fail"]
        unproven = [n for n, o in self.obs.items() if o["required"] and o["state"] != "pass"]
        no_frame = [f for f in self.required_frames() if (self.frames.get(f) or {}).get("state") != "pending_visual_inspection"]
        if failed:
            raise sc.Failure("gate", "observations_failed", names=failed)
        if unproven or no_frame:
            raise sc.Unconfirmed("gate", "required_observations_or_frames_unproven", unproven=unproven, frames_missing=no_frame)

    def keep(self, fid: str, png: pathlib.Path, receipt: dict[str, Any], *, shell_icon: bool = False) -> bool:
        """Keep only bytes that match the native probe's reopened, decoded PNG; read back the exact path used for publication."""
        try:
            with png.open("rb") as image:
                data = image.read(MAX_FRAME_BYTES + 1)
        except OSError:
            self.lose(fid, "frame_file_missing_or_unreadable")
            return False
        if len(data) > MAX_FRAME_BYTES:
            self.lose(fid, "frame_over_the_publishing_limit")
            return False
        if (len(data) < 45 or data[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" or
                data[-12:] != b"\x00\x00\x00\x00IEND\xaeB`\x82"):
            self.lose(fid, "frame_invalid_png")
            return False
        width, height = (int.from_bytes(data[i:i + 4], "big") for i in (16, 20))
        digest = hashlib.sha256(data).hexdigest()
        method = receipt.get("method")
        accepted = ("shell_item_icon",) if shell_icon else ("window_icon",) if fid.startswith("own_icon_") else ("printwindow_composed", "printwindow_default")
        if (receipt.get("file") != png.name or receipt.get("bytes") != len(data) or receipt.get("sha256") != digest or
                (receipt.get("width"), receipt.get("height")) != (width, height) or
                not 0 < width <= 8192 or not 0 < height <= 8192 or method not in accepted):
            self.lose(fid, "frame_native_readback_mismatch")
            return False
        target = self.frames_dir / f"{'shell_item_' if shell_icon else ''}{fid}.png"
        # Exclusive create protects a previous accepted file or an unexpected path. The folder belongs to this run.
        made = False
        try:
            with target.open("xb") as output:
                made = True
                output.write(data)
            with target.open("rb") as image:
                if image.read(MAX_FRAME_BYTES + 1) != data:
                    raise OSError("frame content changed")
        except OSError:
            if made:
                target.unlink(missing_ok=True)
            self.lose(fid, "frame_staging_readback_failed")
            return False
        self.frames[fid] = {"state": "pending_visual_inspection", "file": target.name, "sha256": digest,
                            "bytes": len(data), "width": width, "height": height, "method": method}
        if shell_icon:
            self.frames[fid].update(surface="windows_shell_item_icon_not_explorer_row", explorer_row_state="unknown")
        return True

    def lose(self, fid: str, why: str) -> None:
        self.frames[fid] = {"state": "unknown", "reason": why if re.fullmatch(r"[a-z0-9_]{1,60}", why) else "unspecified"}

    def roots(self) -> dict[str, pathlib.Path]:
        if self.cli.mode != "migration":
            return {"installed": self.install_dir}
        return {**({"old": self.old_root} if self.old_root else {}), "portable": self.package_root, "installed": self.install_dir}

    def identify(self, pid: int) -> dict[str, Any]:
        """Which program ``pid`` is: its image (path classified, bytes compared with the package's when it has them), its command line as
        fixed words, its launcher by name. The image path is read FIRST: a brief launcher can be gone before the slow reads end. Taken
        separately from what the window or the taskbar shows."""
        path = hw.image_path(pid)
        info = hw.cim_process(pid)
        parent = str(info.get("parent") or "gone").lower()
        ident = {**jd.image(path, self.roots()), "pid": pid, "pid_read": bool(path), "parent": parent if re.fullmatch(r"[a-z0-9_.-]{1,40}", parent) else "other",
                 "command": jd.command(str(info.get("cmd") or ""), self.roots(), self.app_root / "repo")}
        if path and ident["copy"] in ("portable", "installed") and self.digests:  # the old release's bytes are checked when unpacked
            try:
                ident["bytes_equal_zip"] = wh.sha256_file(path) == self.digests.get(ident["rel"])
            except OSError:
                ident["bytes_equal_zip"] = None
        return ident

    def env(self, which: str) -> dict[str, str]:
        trace1, trace2 = self.sinks[which]
        trace2.mkdir(parents=True, exist_ok=True)
        return {**self.launch_env, **self.wh.trace_env(trace1, trace2)}

    # ---- clean inputs
    def dirty_inputs(self) -> list[str]:
        """What an earlier install or run left. A clean host has none of it; the uninstall key is read, never guessed."""
        paths = {"profile": self.app_root, "install_dir": self.install_dir, "start_shortcut": self.lnk, "git_traces_installed": self.sinks["installed"][0]}
        dirty = [name for name, path in paths.items() if path.exists()]
        if hw.uninstall_entry(self.inst["uninstall_key"]) is not None:
            dirty.append("uninstall_entry")
        return dirty

    def ensure_no_system_git(self) -> None:
        """No Git reachable by the app: directories, registry and where.exe (never a bare-name program), including the places the
        candidate adds to its PATH. The cwd of that lookup is our own private folder, which exists on any host. A present Git is refused,
        except the Git of a GitHub-hosted runner, and only on the explicit opt-in, by #4220's verified uninstall."""
        def probe() -> dict[str, Any]:
            if not self.args.candidate_src or not pathlib.Path(self.args.candidate_src, "ouroboros").is_dir():
                tools, source = static_tool_dirs(), "kit_static_mirror"
            else:
                tools, source = self.wh.candidate_tool_dirs(pathlib.Path(self.args.candidate_src), sys.executable, self.launch_env), "candidate_source"
            dirs = self.wh.lookup_dirs(self.launch_env["PATH"].split(os.pathsep), tools, self.install_dir)
            report = self.wh.git_lookup_report(dirs, self.install_dir, self.launch_env, self.private)
            return {**{k: report[k] for k in ("absent", "git_hit_count", "unreadable_dir_count")}, "tool_dirs_source": source}

        before = probe()
        facts = {"before": before, "condition": "absent"}
        self.result["facts"]["system_git"] = facts
        if before["unreadable_dir_count"]:
            raise sc.Unconfirmed("host_check", "system_git_lookup_folder_unreadable", unreadable=before["unreadable_dir_count"])
        if not before["absent"]:
            if not (self.opt("uninstall_runner_git") and self.wh.disposable_hosted_runner()):
                raise sc.Unconfirmed("host_check", "system_git_present_a_clean_windows_is_required")
            installation = self.wh.runner_git_installation()
            if installation is None:
                raise sc.Unconfirmed("host_check", "unexpected_foreign_git_installation", git_hits=before["git_hit_count"])
            done = self.wh.uninstall_runner_git(installation)
            facts["uninstall"] = {k: done[k] for k in ("done", "problem")}
            if not done["done"]:
                raise sc.Unconfirmed("host_check", "runner_git_not_uninstalled", problem=done["problem"])
            self.launch_env = self._build_launch_env()
            facts["condition"], facts["after"] = "uninstalled_on_disposable_hosted_runner", probe()
            if facts["after"]["unreadable_dir_count"] or not facts["after"]["absent"]:
                raise sc.Unconfirmed("host_check", "system_git_still_reachable", count=facts["after"]["git_hit_count"])
        self.observe("no_system_git_reachable_before_the_first_launch", "pass", condition=facts["condition"], tool_dirs=before["tool_dirs_source"])

    def host_basics(self) -> None:
        """A disposable GitHub-hosted Windows runner with an interactive desktop and WebView2; anything else is refused."""
        if not self.wh.is_windows() or not self.wh.disposable_hosted_runner():
            raise sc.Unconfirmed("host_check", "not_windows" if not self.wh.is_windows() else "not_a_disposable_hosted_runner")
        self.result["facts"].update(host_image=self.wh.host_image(), dpi_awareness=self.wh.make_dpi_aware())
        if not self.wh.screen_info().get("foreground_present"):
            raise sc.Unconfirmed("host_check", "no_interactive_desktop")
        if not self.wh.webview2_runtime():
            raise sc.Unconfirmed("host_check", "webview2_runtime_absent")

    # ---- frames
    def explorer(self, stage: str, frames: list[dict[str, str]]) -> None:
        """Explorer rows from its own new window, or a separate bounded Shell icon of the exact file. Unobserved remains unknown."""
        if self.opt("skip_explorer"):
            return
        todo = [f for f in frames if pathlib.Path(f["folder"]).is_dir()]
        for f in (f for f in frames if f not in todo):
            self.lose(f["id"], "folder_absent")
        if not todo:
            return
        out, spec = self.private / f"explorer-{stage}", self.private / f"explorer-{stage}.json"
        # Extract a .lnk's icon only after its resolved target and icon source have been checked against this install.
        shortcut_safe = all(self.obs.get(f"start_lnk.{field}", {}).get("state") == "pass" for field in ("target", "icon"))
        spec.write_text(json.dumps({"stage": stage, "frames": [
            {**f, "shell_item_icon": f["id"] != "explorer_start_shortcut" or shortcut_safe} for f in todo]}), encoding="utf-8")
        code, _tail = hw.run_ps(EXPLORER_PS, ["-SpecFile", spec, "-OutDir", out], timeout=900)
        rows = hw.read_json(out / "observations.json").get("frames") or {}
        for f in todo:
            fid = f["id"]
            row, png = rows.get(fid) or {}, out / f"{fid}.png"
            status = row.get("status")
            icon = out / f"shell_item_{fid}.png"
            receipt = row.get("png") or {}
            if status == "captured":
                self.keep(fid, png, receipt)
            elif status == "icon_only" and (fid != "explorer_start_shortcut" or shortcut_safe):
                if self.keep(fid, icon, receipt, shell_icon=True) and self.capture_reason(row.get("capture_reason")):
                    self.frames[fid]["explorer_row_reason"] = row["capture_reason"]
            else:
                stopped = str(row.get("stage") or "")
                self.lose(fid, self.capture_reason(row.get("capture_reason")) or (
                    f"explorer_stopped_at_{stopped}" if re.fullmatch(r"[a-z_]{1,30}", stopped) else f"explorer_script_exit_{code}".lower()))

    @staticmethod
    def capture_reason(value: Any) -> str:
        """A fixed code from the probe, never PowerShell's message, title or path."""
        return value if isinstance(value, str) and re.fullmatch(r"capture_[a-z0-9_]{1,51}", value) else ""

    def probe_window(self, label: str, pid: int, prefix: str, hwnd: int = 0, *, frames_only: bool = False) -> tuple[dict[str, Any], pathlib.Path, str]:
        """brand-probe.ps1 on ONE window of ONE process (``hwnd`` pins the exact one): its title, the icons it supplies, its
        AppUserModel properties, frames drawn by that window itself. It reads and captures; it clicks nothing."""
        out = self.private / f"brand-{label}"
        for number in range(2, 9):  # a retried frame needs its own fresh folder
            if not out.exists():
                break
            out = self.private / f"brand-{label}-{number}"
        code, _tail = hw.run_ps(BRAND_PS, ["-Mode", "window", "-OutDir", out, "-Label", label, "-ProcessId", pid, "-TitlePrefix", prefix,
                                           "-Hwnd", hwnd, *(["-FramesOnly"] if frames_only else [])], timeout=15 if frames_only else 60)
        obs = hw.read_json(out / "observations.json")
        why = ("capture_probe_timeout_or_unavailable" if code is None else f"capture_probe_exit_{code}" if code else
               "capture_probe_receipt_missing" if not obs else "capture_target_not_unique_or_absent" if not (obs.get("window") or {}).get("found") else
               "capture_no_png_receipt")
        return obs, out, why

    def keep_probe_frames(self, o: dict[str, Any], out: pathlib.Path, label: str, why: str, kinds: tuple[str, ...]) -> bool:
        by_id = {str(f.get("id")): f for f in o.get("frames") or [] if isinstance(f, dict)}
        sources = {"window": "window", "caption": "caption", "own_icon": "own-icon-big"}
        got = False
        for kind in kinds:
            source = sources[kind]
            fid = f"{kind}_{label}"
            receipt = by_id.get(source)
            if receipt:
                kept = self.keep(fid, out / f"{source}-{label}.png", receipt)
                got = got or (kept and kind == "window")
            else:
                self.lose(fid, self.capture_reason((o.get("frame_errors") or {}).get(source)) or why.lower())
        return got

    def brand(self, state: str, *, prefix: str = "Ouroboros v", judge: bool = True, hwnd: int = 0) -> None:
        """What Windows holds for the app's window in ``state``: its own title and icons (WM_GETICON, else its class's), AppUserModel.ID, the taskbar
        button as UI Automation names it, frames of the window. The installed copy's window label and icon after an Update are
        required; the taskbar stays "unknown" unless exactly one of its buttons is ours; it is never photographed."""
        if self.opt("skip_brand"):
            return
        self.driver.ensure_alive(f"brand_{state}")
        o, out, why = self.probe_window(state, self.driver.root_pid, prefix, hwnd)
        win, bar = jd.window(o), jd.taskbar(o)
        shown = win if judge else {**win, "label": "not_judged"}  # a setup window or the old release's: no Main title is expected of it
        self.result["facts"].setdefault("visible", {})[state] = {**shown, "taskbar": bar, "aumid_props": {k: bool(v) for k, v in (o.get("props") or {}).items()}}
        if judge:
            both = "pass" if set(win.values()) == {"pass"} else "fail" if "fail" in win.values() else "unknown"
            self.observe(f"window_label_and_icon.{state}", both, required=state in INSTALLED_HANDOFF, stop=False, **win)
            if state in INSTALLED_STATES:
                self.observe(f"aumid.{state}", jd.aumid(o, bar), required=False, stop=False, taskbar=bar["state"], taskbar_reason=bar.get("reason"),
                             taskbar_appid=bar.get("appid_state"), taskbar_name_has_product=bar.get("name_has_product"))
        self.keep_probe_frames(o, out, state, why, ("window", "caption", "own_icon"))

    def capture_wizard(self, key: str, win: Any) -> bool:
        """Frames of the installer's wizard page ``key``: the exact window every time; its title bar and the icon it supplies on its
        first page. Drawn by our own installer's window; nothing is clicked."""
        o, out, why = self.probe_window(f"wizard_{key}", win.pid, "Setup - ", win.hwnd, frames_only=True)
        return self.keep_probe_frames(o, out, f"wizard_{key}", why, ("window", "caption") if key == "ready" else ("window",))

    def reject_wizard(self, key: str) -> None:
        """A frame that does not show the page it is named after is removed, not published as that page."""
        for fid, frame in list(self.frames.items()):
            if fid.endswith(f"_wizard_{key}") and frame.get("state") == "pending_visual_inspection":
                if frame.get("file"):
                    (self.frames_dir / frame["file"]).unlink(missing_ok=True)
                self.lose(fid, "capture_wizard_frame_discarded_before_page_proven")

    # ---- the installer's wizard (the visible leg)
    def run_installer_wizard(self, proc: Any) -> dict[str, Any]:
        def windows_of(pid: int) -> list[Any]:
            try:
                return self.wh.list_own_windows(pid)
            except wh.Unread:
                return []

        def driver_for(win: Any) -> Any:
            return BaseDriver(cu=self.cu, proc=AdoptedProcess(win.pid), host_mod=wh)

        def read(win: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
            try:
                return flatten_tree_nodes(driver_for(win).read_tree(win.hwnd, phase="wizard", allow_truncated=True).get("root")), None
            except (sc.Unconfirmed, sc.Failure) as exc:
                return None, exc.code

        def invoke(win: Any, node: dict[str, Any]) -> dict[str, Any]:
            try:  # InvokePattern only: the skill refuses a coordinate fallback, and nothing else is tried here
                return driver_for(win)._element_act(win.hwnd, phase="wizard", select={"role": node.get("role"), "name": node.get("name")},  # noqa: SLF001
                                                    do="invoke", timeout_ms=3500)
            except (sc.Unconfirmed, sc.Failure) as exc:
                return {"ok": False, "error": exc.code}

        res = drive_installer(proc.pid, pids_of=hw.descendants, windows_of=windows_of, read=read, invoke=invoke, capture=self.capture_wizard,
                              reject=self.reject_wizard, before_finish=self.before_finish, exit_code=proc.poll, timeout=float(self.opt("wizard_minutes", 30.0)) * 60)
        if res["exit_code"] is None and res["stopped"] is None:
            try:
                res["exit_code"] = proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass
        return res

    def before_finish(self) -> None:
        """Copying has ended and Finish is not pressed yet: compare the installed tree, and see that nobody holds the profile."""
        self.compare_installed_payload()
        holder = hw.lock_pid(self.app_root)
        self.held_before_finish = bool(holder and hw.pid_alive(holder))
        self.asked("app_by_installer_finish")  # Finish is invoked next; whether an app starts is seen later, or not

    def run_ordinary_installer(self, env: dict[str, str]) -> None:
        """The installer opened as a person opens it, driven by UIA InvokePattern only; its Finish page launches the app."""
        self.asked("installer")
        proc = subprocess.Popen([a.replace("{installer}", str(self.installer)) for a in self.inst["visible_install_argv"]], cwd=str(self.private),
                                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.mark("installer_opened")
        self.result["facts"]["install"] = {"mode": "wizard_by_uia", "note": "the ordinary wizard, UIA InvokePattern only; Launch Ouroboros is checked by default"}
        res = self.run_installer_wizard(proc)
        if res["wizard_found"]:
            self.saw("installer_wizard_window")
        self.result["facts"]["install"].update(wizard_seen_in_this_leg=bool(res["stages_seen"]), finish_pressed="finish" in res["pressed"])  # before record_wizard may stop
        self.record_wizard(res)
        if "installed_files_equal_the_package_payload" not in self.obs:  # Finish was never reached: said, not assumed
            self.observe("installed_files_equal_the_package_payload", "unknown", stop=False, code="finish_page_not_reached")

    def record_wizard(self, res: dict[str, Any]) -> None:
        """Every stage is its own decision: a page that was not read, a frame that was not taken, a press that had no effect stays unknown."""
        keep =("wizard_found", "stages_seen", "pressed", "stopped", "exit_code", "buttons_seen", "uia_problem", "invoke", "polls", "extra_pages")
        self.result["facts"]["wizard"] = {k: res.get(k) for k in keep}
        self.result["facts"]["wizard"]["frames"] = {k: v.get("captured") for k, v in res["frames"].items()}
        seen, pressed = set(res["stages_seen"]), set(res["pressed"])
        self.observe("wizard_window_found_in_the_installer_tree", jd.tri(True) if res["wizard_found"] else "unknown", stop=False)
        for stage, name in (("ready", "install_page"), ("progress", "progress_page"), ("finish", "finish_page")):
            # the transient progress page is the visible leg's evidence; the migration does not depend on catching it
            self.observe(f"wizard_{name}_seen", "pass" if stage in seen else "unknown", required=stage != "progress" or self.cli.mode == "visible", stop=False)
        self.observe("wizard_install_invoked_through_uia", "pass" if "ready" in pressed else "unknown", stop=False)
        self.observe("wizard_finish_invoked_through_uia", "pass" if "finish" in pressed else "unknown", stop=False)
        code = res["exit_code"]
        self.observe("installer_exit_code", "unknown" if code is None else jd.tri(code in self.inst["success_exit_codes"]), stop=False, exit_code=code)
        if res["stopped"]:
            raise sc.Unconfirmed(self.step, f"wizard_{res['stopped']}", **{k: res[k] for k in ("stages_seen", "pressed")})

    # ---- processes
    def adopt_lock_holder(self, phase: str, before: int | None, wait_sec: float) -> int:
        """The live launcher the installation's lock file names. After a normal Quit the file is gone; a holder equal to ``before`` is only
        accepted when ``before`` was already dead (Windows recycles PIDs)."""
        was_alive = before is not None and hw.pid_alive(before)

        def holder() -> int | None:
            pid = hw.lock_pid(self.app_root)
            return pid if pid and hw.pid_alive(pid) and (pid != before or not was_alive) else None

        pid = sc.poll_until(wait_sec, holder, interval=0.1)
        if not pid:
            raise sc.Unconfirmed(phase, "no_launcher_took_the_installation_lock")
        self.driver = HandoffAwareDriver(cu=self.cu, proc=AdoptedProcess(int(pid)), host_mod=wh)
        return int(pid)

    def await_main_window(self, state: str) -> list[str]:
        """The Main window of the launcher that holds the lock. A "Setup Required" or "Startup Failed" window is a product failure (the
        driver raises it); a first-run Setup wizard for an adopted profile is one too."""
        deadline = time.monotonic() + BOOT_SEC
        while time.monotonic() < deadline:
            titles = [w.title for w in self.driver._owned_ouroboros_windows(state)]  # noqa: SLF001
            if any(jd.MAIN_TITLE.match(t) for t in titles):
                return titles
            if any(t == SETUP_TITLE for t in titles):
                raise sc.Failure(state, "wizard_shown_for_an_adopted_profile")
            time.sleep(0.5)
        raise sc.Unconfirmed(state, "main_window_not_observed")

    def await_setup_window(self, phase: str) -> Any:
        """The app's first-run wizard window (HWND and owner PID) of the launcher that holds the lock, by its exact title: another
        window whose title only starts with the product's name (an error dialog) is not it."""
        deadline = time.monotonic() + BOOT_SEC
        while time.monotonic() < deadline:
            wins = self.driver._owned_ouroboros_windows(phase)  # noqa: SLF001
            setup = [w for w in wins if w.title == SETUP_TITLE]
            if len(setup) == 1:
                return setup[0]
            if not setup and any(jd.MAIN_TITLE.match(w.title) for w in wins):
                raise sc.Failure(phase, "first_run_wizard_not_shown")
            time.sleep(0.5)
        raise sc.Unconfirmed(phase, "setup_window_not_observed")

    def setup_control_read(self, hwnd: int) -> bool | None:
        """The wizard's own footer button, read by UIA in that exact window. Not read is not established (never a failure)."""
        def look() -> bool:
            return len(self.driver.find_nodes(hwnd, phase="first_launch", automation_id="next-btn", require_visible=False, allow_truncated=True)) == 1

        try:
            return True if sc.poll_until(60.0, look, interval=1.0) else None
        except (sc.Unconfirmed, wh.Unread):
            return None

    def main_window(self, phase: str) -> tuple[int, dict[str, Any]]:
        """The Main window of whoever holds the profile, and the identity of the process that owns THAT window."""
        self.await_main_window(phase)
        hwnd = self.driver.acquire_main_window(phase=phase, timeout_sec=60.0)
        owner = self.identify(self.driver.root_pid)
        if not self.wh.window_owned_by(hwnd, owner["pid"]):
            raise sc.Unconfirmed(phase, "the_window_changed_its_owner_while_it_was_identified")
        self.result["facts"]["identity"][f"{phase}_window_owner"] = owner
        return hwnd, owner

    def read_pair_in_window(self, label: str, hwnd: int) -> None:
        """The former exchange as drawn in that very window: #4220's screen witness (UIA text leaves inside the Main feed). A window
        UIA cannot read leaves it unknown, which keeps the leg from passing; the durable history is a separate fact."""
        phase = f"pair_in_{label}"
        want = [("user", fm.prompt_text(1, self.tag)), ("assistant", fm.reply_text(1, self.tag))]
        witness, seen = self.message_witness(1), {"durable": False, "uia_gap": ""}

        def look() -> bool:
            self._observe_chat_state(hwnd, phase=phase, expected_durable=want, witness=witness, state=seen)
            return witness.complete() and not seen["uia_gap"]

        try:
            self.ensure_main_chat_open(phase=phase, click_sidebar=True)
            sc.poll_until(self.RETAINED_SEC, look, interval=0.5)
        except (sc.Unconfirmed, wh.Unread):
            pass
        self.observe(f"former_pair_read_in_the_window.{label}", "pass" if witness.complete() and not seen["uia_gap"] else "unknown", stop=False)

    def versions(self, state: str) -> None:
        """Whose code this window was, as facts: the process that holds the installation and its title's version (the shell), and
        the body that serves it (the API's version and running revision, the checkout's HEAD). Nothing here is a verdict."""
        health, status = self.api("/api/health") or {}, self.api("/api/update/status") or {}
        known = {"old_release": self.old.get("source_sha", ""), "candidate": self.candidate_sha, "update_target_b": self.b_sha}
        word = lambda v: str(v)[:24] if re.fullmatch(r"v?\d+\.\d+\.\d+[A-Za-z0-9.+-]*", str(v or "")) else "unread"  # noqa: E731
        try:
            titles = [m.group(0).split()[-1] for w in self.wh.list_own_windows(self.driver.root_pid) if (m := jd.MAIN_TITLE.match(w.title))]
        except wh.Unread:
            titles = []
        holder = self.identify(self.driver.root_pid)
        self.result["facts"].setdefault("versions", {})[state] = {
            "window_title_version": word(titles[0]) if len(titles) == 1 else "unread",
            "window_process": {"copy": holder.get("copy"), "kind": holder.get("kind"), "read": holder.get("pid_read")},
            "package_version_the_api_was_told": word(health.get("app_version")),
            "body_version": word(health.get("runtime_version")),
            "body_revision": jd.revision(status.get("running_sha"), known),
            "checkout_revision": jd.revision(pb.read_git_head(self.app_root / "repo" / ".git"), known)}

    def start(self, state: str, *, launch: bool = True) -> dict[str, Any]:
        """Open the real Start-menu .lnk (or take the launcher something else started), adopt the launcher that holds the lock, identify it at
        first sight, wait for its Main window."""
        before = hw.lock_pid(self.app_root)
        if launch:
            self.asked("app_by_start_shortcut")
            hw.start_shortcut(self.lnk, self.opt("launch_route", "shellexecute"), self.launch_env if self.opt("launch_route", "shellexecute") == "shellexecute" else None)
        self.mark(f"{state}_launched")
        pid = self.adopt_lock_holder(state, None if not launch else before, BOOT_SEC)
        if launch:
            self.saw("app_by_start_shortcut")
        first = self.identify(pid)
        self.result["facts"]["identity"][f"{state}_first_holder"] = first
        self.await_main_window(state)
        self.driver.acquire_main_window(phase=state, timeout_sec=60.0)
        return first

    def quit(self, label: str) -> None:
        """The owner's close gesture: WM_CLOSE to the Main window and, if the one-time 'Keep Ouroboros running?' appears, UIA InvokePattern on its
        Cancel (which quits). No key is sent. Settled by the process ending and by the cause the app itself recorded."""
        self.driver.ensure_alive(f"quit_{label}")
        hwnd, pid = self.driver.acquire_main_window(phase=f"quit_{label}", timeout_sec=90.0), self.driver.root_pid
        entry = self.data / "state" / "shutdown_entry.json"
        before = entry.stat().st_mtime_ns if entry.is_file() else None  # an earlier Quit's record must not speak for this one
        hw.post_close(hwnd, pid)
        asked, deadline = False, time.monotonic() + QUIT_SEC
        while hw.pid_alive(pid):
            if time.monotonic() > deadline:
                raise sc.Failure(f"quit_{label}", "process_did_not_exit_after_close", dialog_answered=asked)
            for w in self.wh.list_own_windows(pid):
                if w.title == CONSENT and not asked:
                    self.driver.act_or_click(w.hwnd, phase=f"quit_{label}", label="Cancel", query={"role": "button", "name": "Cancel"},
                                             verbs=("invoke",), single_press_strict=True)
                    asked = True
            time.sleep(0.5)
        doc = pb.read_json_retry(entry) if entry.is_file() and entry.stat().st_mtime_ns != before else None
        cause = doc.get("cause") if isinstance(doc, dict) and re.fullmatch(r"[a-z_]{1,40}", str(doc.get("cause"))) else None
        self.observe(f"normal_quit.{label}", "unknown" if cause is None else jd.tri(cause == "launcher_quit"), unknown_stops=False, cause=cause, dialog_answered=asked)

    def wait_b_shell_ready(self, *, stable_sec: float = 5.0, timeout: float = 300.0, interval: float = 1.0) -> int:
        """After the first Update: the app's own API says it runs B with no update transaction open, and ONE live launcher holds the lock on
        two reads at least ``stable_sec`` apart. Only then is the process looked at: pressing Update returns before the handoff happens."""
        seen: dict[str, Any] = {"pid": None, "since": 0.0}

        def ready() -> int | None:
            status = self.api("/api/update/status") or {}
            if not (status.get("running_sha") == self.b_sha and status.get("current_sha") == self.b_sha and (status.get("update_tx") or {}).get("active") is False):
                seen["pid"] = None
                return None
            pid = hw.lock_pid(self.app_root)
            if not (pid and hw.pid_alive(pid)):
                seen["pid"] = None
                return None
            moment = time.monotonic()
            if seen["pid"] != pid:
                seen.update(pid=pid, since=moment)
            return pid if moment - seen["since"] >= stable_sec else None

        pid = sc.poll_until(timeout, ready, interval=interval)
        if not pid:
            raise sc.Unconfirmed("handoff", "b_shell_not_ready_in_time")
        return int(pid)

    # ---- migration phases
    def unpack_old_release(self) -> None:
        """The published old release, by its pinned bytes: the archive, its executable and what its own manifest names. It is never
        rebuilt, patched or replaced by the candidate."""
        self.step = "old_release"
        archive = pathlib.Path(str(self.opt("old_zip", "") or "-"))
        if not archive.is_file() or wh.sha256_file(archive) != self.old["zip_sha256"]:
            raise sc.Unconfirmed("old_release", "old_release_zip_missing_or_not_the_pinned_bytes")
        root, manifest, _gits = pb.unpack_candidate_archive(archive, self.qa_root / "old-app")
        same = ((root / "Ouroboros.exe").is_file() and wh.sha256_file(root / "Ouroboros.exe") == self.old["exe_sha256"]
                and manifest.get("source_sha") == self.old["source_sha"] and manifest.get("managed_remote_url") == self.old["remote"])
        if not same:
            raise sc.Unconfirmed("old_release", "old_release_is_not_the_published_one")
        self.old_root = root
        self.result["facts"]["old_release"] = {"version": self.old["version"], "source_sha": self.old["source_sha"], "zip_sha256": self.old["zip_sha256"]}

    def prepare_mirror(self) -> None:
        """#4220's local bare mirror and fixture commit B, without its removal of the system Git: the former user keeps theirs."""
        self.step = "prepare_mirror"
        harness_git = shutil.which("git")
        if not harness_git or self.bundle is None:
            raise sc.Unconfirmed("prepare_mirror", "no_git_for_fixture_preparation")
        self.marker = f"OUROBOROS-QA-FIXTURE-B-{fm._clean_tag(self.tag)}"  # noqa: SLF001
        try:
            self.b_sha, _tags, parent = pb.create_fixture_mirror_b(harness_git=harness_git, mirror=self.mirror, qa_root=self.qa_root, bundle=self.bundle,
                                                                  candidate=self.candidate, doc_rel_path=sc.FIXTURE_DOC, marker=self.marker)
        except pb.Refusal as ref:
            raise sc.Unconfirmed("prepare_mirror", ref.reason, **ref.fields) from ref
        ancestor = subprocess.run([harness_git, "-C", str(self.mirror), "merge-base", "--is-ancestor", self.old["source_sha"], self.candidate],
                                  capture_output=True, timeout=120).returncode
        if parent != self.candidate or ancestor != 0:
            raise sc.Unconfirmed("prepare_mirror", "the_candidate_is_not_a_fast_forward_of_the_old_release_and_b", merge_base_exit=ancestor)
        self.result["facts"]["fixture"] = {"old": self.old["source_sha"], "a": self.candidate, "b": self.b_sha, "local_bare_mirror": True}

    def host(self) -> None:
        self.prepare_host_and_package()  # #4220: Windows, interactive, WebView2, fresh profile, ZIP digest, package names candidate and mirror
        if dirty := self.dirty_inputs():
            raise sc.Unconfirmed("host_check", "inputs_not_clean", names=dirty)
        if wh.sha256_file(self.installer) != self.inst["sha256"].lower():
            raise sc.Failure("inputs", "installer_digest_mismatch")
        version = (self.package_root / "_internal" / "VERSION").read_text(encoding="utf-8").strip()
        self.version = version
        self.observe("installer_name_carries_the_payload_version", jd.tri(self.installer.name == f"Ouroboros-Setup-{version}-windows-x64.exe"))
        with zipfile.ZipFile(self.args.archive) as z:
            for info in (i for i in z.infolist() if not i.is_dir()):
                with z.open(info) as handle:
                    self.digests[pb.inner_archive_path(info.filename).lower()] = hashlib.file_digest(handle, "sha256").hexdigest()
        self.result["facts"]["new_package"] = {"version": pb._scalar(version), "zip_sha256": self.args.archive_sha256.lower(),  # noqa: SLF001
                                               "installer_sha256": self.inst["sha256"].lower()}
        self.launch_env = self._build_launch_env()
        if self._probe_host_git(pathlib.Path(self.args.candidate_src))[1]["absent"]:
            raise sc.Unconfirmed("host_check", "the_old_release_needs_a_system_git_and_there_is_none")
        self.unpack_old_release()
        self.prepare_mirror()

    def official_addresses(self) -> list[str]:
        """Every spelling of the official address a program of this leg may be configured with: the published release's own and the app's
        default, each without and with ".git". Git takes the longest matching rule, so each one is read as exactly the mirror."""
        bare = {address.removesuffix(".git") for address in (self.old["remote"], APP_DEFAULT_SOURCE)}
        return sorted(bare | {f"{address}.git" for address in bare})

    def redirect_to_mirror(self) -> None:
        """Environment variables of this run's own launches only (no Git config file is written anywhere): every spelling of the official
        address is read from the local mirror. Never taken back, so every later program of the leg, the Update included, inherits it."""
        self.plain_env = dict(self.launch_env)  # without the rules: what a program NOT started by this run (real Explorer) would be given
        addresses = self.official_addresses()
        rules = {"GIT_CONFIG_COUNT": str(len(addresses))}
        for number, address in enumerate(addresses):
            rules.update({f"GIT_CONFIG_KEY_{number}": f"url.{self.args.mirror_url}.insteadOf", f"GIT_CONFIG_VALUE_{number}": address})
        self.launch_env = {**self.launch_env, **rules}

    def effective_source(self, address: str, cwd: pathlib.Path, env: dict[str, str] | None = None) -> str:
        """What Git would really read for ``address`` (a remote's name or a URL), asked of the former user's Git in the app's launch
        environment (or ``env``): ``ls-remote --get-url`` applies every url.<base>.insteadOf and contacts nobody. One fixed word:
        "mirror", "other" (another address) or "unread" (Git not found or no single answer). The address itself never leaves."""
        env = self.launch_env if env is None else env
        git = shutil.which("git", path=env.get("PATH"))
        if not git:
            return "unread"
        try:
            done = subprocess.run([git, "ls-remote", "--get-url", "--", address], cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, errors="replace", timeout=60)
        except (OSError, subprocess.SubprocessError):
            return "unread"
        lines = done.stdout.splitlines()
        if done.returncode or len(lines) != 1 or not lines[0]:
            return "unread"
        return "mirror" if same_address(lines[0], self.args.mirror_url) else "other"

    def old_profile_port(self, pid: int) -> int:
        """Bind writes to this freshly created profile, known old checkout and its own live loopback process."""
        port = 0
        try:
            raw = (self.data / "state" / "server_port").read_text(encoding="utf-8").strip()
            port = int(raw) if raw.isdigit() else int(json.loads(raw)["port"])
        except (OSError, ValueError, TypeError, KeyError):
            pass
        image = hw.image_path(pid)
        if (not image or not same_address(image, self.exe) or not hw.loopback_listener_owned(port, pid)
                or pb.read_git_head(self.app_root / "repo" / ".git") != self.old["source_sha"]):
            raise sc.Unconfirmed(self.step, "old_profile_loopback_or_revision_unverified")
        code, health = profile_http(port, "/api/health")
        if code != 200 or health.get("runtime_version") != self.old["version"]:
            raise sc.Unconfirmed(self.step, "old_profile_loopback_or_revision_unverified")
        return port

    def old_complete(self, port: int) -> None:
        """The v7.12.0 public completion contract. Save once; a lost answer only reads back that attempt."""
        assert self.model is not None
        endpoint = self.model.base_url
        if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}/v1", endpoint):
            raise sc.Unconfirmed(self.step, "old_fixture_endpoint_not_loopback")
        path, attempt = "/api/onboarding/complete", f"qa-{uuid.uuid4().hex}"
        status, receipt = profile_http(port, path, {"attempt_id": attempt, "OPENAI_COMPATIBLE_BASE_URL": endpoint,
            "OPENAI_COMPATIBLE_API_KEY": "", "OUROBOROS_MODEL": f"openai-compatible::{fm.MODEL_ID}",
            "skipSubscriptionPresets": True, "allow_empty_review_pool": True, "consciousness": False, "starter_skills": []})
        if status != 200 or receipt.get("ok") is not True or receipt.get("status") != "saved":
            if receipt.get("saved") is False:
                raise sc.Unconfirmed(self.step, "old_onboarding_refused", http_status=status)

            def readback() -> dict[str, Any] | None:
                code, same = profile_http(port, f"{path}?attempt={attempt}")
                saved = same.get("receipt")
                return saved if code == 200 and same.get("saved") is True and isinstance(saved, dict) else None

            receipt = sc.poll_until(45.0, readback, interval=1.0) or {}
            if receipt.get("ok") is not True or receipt.get("status") != "saved":
                raise sc.Unconfirmed(self.step, "old_onboarding_completion_unconfirmed")
        if receipt.get("restart_required") is not False:
            raise sc.Unconfirmed(self.step, "old_onboarding_runtime_not_ready_without_restart")

    def old_ready_for_chat(self, port: int) -> bool:
        code, health = profile_http(port, "/api/health")
        boot = health.get("boot")
        state = self.state_doc()
        if code != 200 or state is None or not isinstance(boot, dict) or not (boot.get("dialogue_ready") is True and boot.get("tasks_ready") is True):
            return False
        # Start can admit a host-origin first meeting in Main. Let it settle before sending the one owner turn;
        # the strict model fixture declines that greeting and the greeting cannot serve as our saved exchange.
        meeting = state.get("first_meeting") or {}
        if not isinstance(meeting, dict):
            return False
        if meeting.get("state") != "admitted":
            return meeting.get("state") not in ("owed", "admitting")
        task = str(meeting.get("task_id") or "")
        if not re.fullmatch(r"[0-9a-f]{8}", task):
            return False
        code, detail = profile_http(port, f"/api/tasks/{task}")
        return code == 200 and detail.get("status") in ("completed", "failed", "cancelled", "interrupted")

    def old_first_exchange(self, port: int) -> None:
        assert self.model is not None
        prompt, reply = fm.prompt_text(1, self.tag), fm.reply_text(1, self.tag)
        want = [("user", prompt), ("assistant", reply)]
        cursor = self.model.cursor()
        if not send_old_main_turn(port, prompt):
            raise sc.Unconfirmed(self.step, "old_main_owner_echo_unconfirmed")

        def answered() -> bool:
            return any(e.get("kind") == "completion" and e.get("slot") == 1 and e.get("status") == 200
                       for e in self.model.events_since(cursor))

        def saved_pair() -> bool:
            if not answered():
                return False
            ok, _, rows = self.durable_history()
            return ok and sc.has_exact_subsequence(rows, want)

        if not sc.poll_until(self.TURN_SEC, saved_pair, interval=0.5):
            raise sc.Unconfirmed(self.step, "old_fixture_reply_or_history_unconfirmed", fixture_completion_seen=answered())
        self.init_id = self.initialization_id()
        status, state = self.api("/api/update/status") or {}, self.state_doc() or {}
        revs = [status.get("running_sha"), status.get("current_sha"), state.get("current_sha"),
                pb.read_git_head(self.app_root / "repo" / ".git")]
        if not self.init_id or not all(revs):
            raise sc.Unconfirmed(self.step, "old_exchange_profile_identity_unreadable")
        self.observe("old_app_created_and_saved_the_fixture_exchange", jd.tri(set(revs) == {self.old["source_sha"]}),
                     accepted_main_echo=True, fixture_model_replied=True, history_via_api=True)

    def old_release_seed(self) -> None:
        """Launch the pinned old ZIP itself. Its own public APIs do the writes; expose its Main only for a normal Quit."""
        self.step = "old_release"
        self.redirect_to_mirror()
        read = [self.effective_source(address, self.private) for address in self.official_addresses()]
        self.observe("official_address_reads_from_the_mirror_before_the_old_release", "pass" if set(read) == {"mirror"} else "unknown",
                     spellings=len(read), read=sorted(set(read)))
        new = (self.exe, self.package_root, self.candidate)
        self.exe, self.package_root, self.candidate = self.old_root / "Ouroboros.exe", self.old_root, self.old["source_sha"]
        try:
            self.cu = self._load_uia_skill(pathlib.Path(self.args.candidate_src))
            self.asked("old_release_portable")
            spawn = self._launcher_spawn or (lambda cmd, **kw: subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=hw.NEW_GROUP, **kw))
            self.proc = spawn([str(self.exe)], cwd=str(self.exe.parent), env=self.env("portable"))
            self.driver = HandoffAwareDriver(cu=self.cu, proc=self.proc, host_mod=self.wh)
            win = self.await_setup_window(self.step)
            image = hw.image_path(win.pid)
            if not image or not same_address(image, self.exe) or not self.wh.window_owned_by(win.hwnd, win.pid):
                raise sc.Unconfirmed(self.step, "old_setup_window_not_bound_to_published_exe")
            self.saw("old_release_portable")
            self.old_complete(self.old_profile_port(win.pid))
            hw.post_close(win.hwnd, win.pid)  # dismiss the old wizard after its API has saved; no field or button driven
            if not sc.poll_until(20.0, lambda: not self.wh.window_owned_by(win.hwnd, win.pid), interval=0.1):
                raise sc.Unconfirmed(self.step, "old_setup_window_close_unconfirmed")
            self.await_main_window(self.step)
            port = self.old_profile_port(self.driver.root_pid)
            if not sc.poll_until(180.0, lambda: self.old_ready_for_chat(port), interval=1.0):
                raise sc.Unconfirmed(self.step, "old_dialogue_or_setup_meeting_not_ready")
            self.old_first_exchange(self.old_profile_port(self.driver.root_pid))
            (self.app_root / "repo" / ".git" / "qa-profile-token").write_text(self.token, encoding="utf-8")  # an unknown file in .git: git status stays clean
            self.versions("old_release_first_run")
            self.quit("old_release")
        finally:
            self.exe, self.package_root, self.candidate = new  # the redirect stays in the launch environment to the end of the leg

    def install(self) -> None:
        """The same ordinary wizard as the visible leg, over the former user's profile, with nobody holding it."""
        self.step = "install"
        holder = hw.lock_pid(self.app_root)
        if holder and hw.pid_alive(holder):
            raise sc.Unconfirmed("install", "the_app_was_running_before_the_installer_was_opened")
        self.run_ordinary_installer(self.launch_env)
        if self.obs["installed_files_equal_the_package_payload"]["state"] == "fail":
            raise sc.Failure("install", "installed_files_differ_from_the_package")

    def compare_installed_payload(self) -> None:
        """Exact equality of the installed tree with the package (the installer's own uninstaller files aside), read once, after
        copying has ended and before anything is launched: a first launch may write files (__pycache__) that were never shipped."""
        name = "installed_files_equal_the_package_payload"
        if not self.digests:
            return self.observe(name, "unknown", stop=False, code="no_package_to_compare_with")
        skip = [pat.lower() for pat in self.inst["files_not_in_zip"]]
        try:
            got = {p.relative_to(self.install_dir).as_posix().lower(): wh.sha256_file(p) for p in self.install_dir.rglob("*") if p.is_file()}
        except OSError:
            return self.observe(name, "unknown", stop=False, code="installed_tree_unreadable")
        got = {k: v for k, v in got.items() if not ("/" not in k and any(fnmatch.fnmatch(k, pat) for pat in skip))}
        bad = {"missing": len(set(self.digests) - set(got)), "extra": len(set(got) - set(self.digests)),
               "different": sum(1 for k in got if k in self.digests and got[k] != self.digests[k])}
        self.observe(name, jd.tri(not any(bad.values())), stop=False, **bad)

    def check_installed(self) -> None:
        i = self.inst
        entry = hw.uninstall_entry(i["uninstall_key"])
        self.observe("install_layout_exe_shortcut_uninstall_entry",
                     jd.tri((self.install_dir / "Ouroboros.exe").is_file() and self.lnk.is_file() and entry is not None
                            and wh.inside(hw.split_command(entry["UninstallString"])[0], self.install_dir)), uninstall_entry=entry is not None)
        self.read_shortcut()
        self.explorer("post_install", [
            {"id": "explorer_installed_exe", "folder": str(self.install_dir), "folder_label": "installed folder", "item": "Ouroboros.exe"},
            {"id": "explorer_start_shortcut", "folder": str(self.lnk.parent), "folder_label": "Start menu programs", "item": self.lnk.name}])

    def read_shortcut(self) -> None:
        out = self.private / "shortcut"
        hw.run_ps(BRAND_PS, ["-Mode", "shortcut", "-OutDir", out, "-ShortcutPath", self.lnk], timeout=180)
        lnk = hw.read_json(out / "observations.json").get("shortcut") or {}
        verdict = jd.shortcut(lnk, self.install_dir / "Ouroboros.exe", self.install_dir) if lnk.get("exists") else {
            "target": "unknown", "icon": "unknown", "icon_source": "unread", "aumid": "unknown"}
        for key in ("target", "icon", "aumid"):  # SOURCE of the icon only; its pixels are a frame, judged by a viewer
            self.observe(f"start_lnk.{key}", verdict[key], stop=False, icon_source=verdict["icon_source"] if key == "icon" else None)

    def installed_first_launch(self) -> None:
        """The window the installer's Finish opened, over the former profile. Nothing else is launched here: when Finish opens nothing,
        another copy or another profile, the leg ends; the Start shortcut is never used to make up for it."""
        self.step = "installed_first_launch"
        opened = "finish_opened_a_window_of_the_installed_copy"
        if self.held_before_finish is not False:
            raise sc.Unconfirmed(self.step, "the_profile_was_not_seen_free_before_finish")
        try:
            pid = self.adopt_lock_holder(self.step, None, BOOT_SEC)
            self.saw("app_by_installer_finish")
            self.result["facts"]["identity"]["finish_first_holder"] = self.identify(pid)
            hwnd, owner = self.main_window(self.step)
        except sc.Unconfirmed:
            self.observe(opened, "unknown", stop=False)
            raise
        self.first_pid = owner["pid"]
        self.observe(opened, jd.tri(jd.installed_window(owner)), copy=owner["copy"], kind=owner["kind"], read=owner["pid_read"])
        sc.poll_until(180.0, lambda: self.durable_history()[0] and self.api("/api/update/status") and self.state_doc(), interval=2.0)
        ok, complete, rows = self.durable_history()
        token = self.app_root / "repo" / ".git" / "qa-profile-token"
        try:
            kept: bool | None = token.read_text(encoding="utf-8") == self.token
        except FileNotFoundError:
            kept = False if token.parent.is_dir() else None
        except OSError:
            kept = None
        checks = {"history_rows_via_api": jd.saved_rows(ok, complete, sc.has_exact_subsequence(rows, [("user", fm.prompt_text(1, self.tag)),
                                                                                                    ("assistant", fm.reply_text(1, self.tag))])),
                  "same_initialization_id": jd.equals(self.initialization_id(), self.init_id),
                  "token_in_the_working_repo": kept}
        self.observe("former_chat_opened_by_the_installed_program", jd.tri(jd.and3(*checks.values())), **{k: jd.tri(v) for k, v in checks.items()})
        self.read_pair_in_window("finish_window", hwnd)
        self.versions("installed_first_launch")
        self.brand("installed_first_launch", hwnd=hwnd)

    def serving_revision(self) -> str:
        """Which known commit actually serves the profile now: the new launcher may have left the old checkout, or moved it to the
        candidate. The API's running and current revision and the checkout's HEAD must agree."""
        status = self.api("/api/update/status") or {}
        seen = {str(v or "") for v in (status.get("running_sha"), status.get("current_sha"), pb.read_git_head(self.app_root / "repo" / ".git"))}
        known = {"old_release": self.old["source_sha"], "candidate": self.candidate_sha}
        return jd.revision(seen.pop(), known) if len(seen) == 1 else "inconsistent"

    def configured_source(self, env: dict[str, str]) -> dict[str, str]:
        """Where Git, given ``env``, would read the profile's configured update source. The app re-points its remote at the managed
        manifest's address (its own official default when the manifest names none) before it fetches, so BOTH that address and the
        checkout's own remote are asked, offline. Words only."""
        git_dir = self.app_root / "repo" / ".git"
        found = {"manifest": "unread", "remote": "unread"}
        try:
            meta = json.loads((git_dir / "ouroboros-managed.json").read_text(encoding="utf-8"))
            config = (git_dir / "config").read_text(encoding="utf-8")
        except (OSError, ValueError):
            return found
        if not isinstance(meta, dict):
            return found
        name = str(meta.get("managed_remote_name") or "managed")
        found["manifest"] = self.effective_source(str(meta.get("managed_remote_url") or "").strip() or APP_DEFAULT_SOURCE, git_dir.parent, env)
        if jd.remote_url(config, name):  # Git would echo the name of an absent remote back as if it were an address: leave that unread
            found["remote"] = self.effective_source(name, git_dir.parent, env)
        return found

    def by_explorer(self) -> bool:
        """The shortcut is opened by the running shell, which gives the app its own environment: no redirect reaches that start."""
        return self.opt("launch_route", "shellexecute") == "explorer"

    def update_source_is_the_mirror(self) -> bool | None:
        """Both configured addresses in the launch environment, and the running app's passive status, which says where ITS Git reads the
        remote (``git remote get-url`` in the app's own environment); with a start by real Explorer ahead, both addresses also without
        the redirect. True only when every one is the local mirror; anything else, another address included, is None: the stand must
        not press Update, and that is not a product failure."""
        found = self.result["facts"]["update_source"] = {**self.configured_source(self.launch_env), "app_reports": "unread"}
        shown = str((self.api("/api/update/status") or {}).get("official_repo_url") or "")  # a passive read: it never fetches
        if shown:
            found["app_reports"] = "mirror" if same_address(shown, self.args.mirror_url) else "other"
        if self.by_explorer():
            found.update({f"{key}_for_explorer": word for key, word in self.configured_source(self.plain_env).items()})
        return True if set(found.values()) == {"mirror"} else None

    def update(self) -> None:
        self.step = "update_source"
        serving = self.serving_revision()
        # Both required before anything is pressed: an unknown starting point or any other source ends the leg here.
        self.observe("update_starts_from_the_revision_that_serves", "pass" if serving in ("old_release", "candidate") else "unknown", serving=serving)
        self.observe("update_source_is_the_local_mirror", jd.tri(self.update_source_is_the_mirror()))
        self.result["facts"]["update_from"] = serving
        candidate = self.candidate  # #4220's "A" is the revision that serves, for the Update only
        self.candidate = self.old["source_sha"] if serving == "old_release" else self.candidate_sha
        try:
            self.run_plain_check_and_update_a_to_b()
            self.step = "handoff"
            pid = self.wait_b_shell_ready()  # first the app's own evidence that B runs; only then the process, its path and its bytes
            ident = self.identify(pid)
            try:
                data = (self.app_root / "repo" / "launcher.py").read_bytes()
                is_b: bool | None = FIXTURE["marker"].encode() in data and hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == FIXTURE["launcher_sha"]
            except OSError:
                is_b = None
            self.result["facts"]["identity"]["after_update"] = ident
            self.observe("update_handed_off_to_the_updated_launcher_py", jd.tri(jd.after_update(ident, "installed", pid=pid, first_pid=self.first_pid, file_is_b=is_b)),
                         unknown_stops=False, profile_launcher_py_is_b=is_b, kind=ident.get("kind"), command=ident.get("command"), read=ident["pid_read"])
            self.verify_after_update_and_second_chat()
            served = (self.api("/api/health") or {}).get("runtime_version")
            self.observe("after_update_the_new_body_serves", jd.tri(jd.equals(served, self.version)), unknown_stops=False)
            self.versions("after_update")
            self.brand("after_update_handoff")
            self.quit("installed_after_update")
        finally:
            self.candidate = candidate

    def history_both(self, label: str) -> None:
        want = [("user", fm.prompt_text(1, self.tag)), ("assistant", fm.reply_text(1, self.tag)),
                ("user", fm.prompt_text(2, self.tag)), ("assistant", fm.reply_text(2, self.tag))]
        sc.poll_until(180.0, lambda: self.durable_history()[0] and self.state_doc(), interval=2.0)  # the app's API answers a while after its window
        ok, complete, rows = self.durable_history()
        state = self.state_doc() or {}
        saved = jd.saved_rows(ok, complete, sc.has_exact_subsequence(rows, want))
        good = jd.and3(saved, jd.equals(pb.read_git_head(self.app_root / "repo" / ".git"), self.b_sha),
                       jd.equals(state.get("current_sha"), self.b_sha), jd.equals(self.initialization_id(), self.init_id))
        self.observe(f"history_via_app_api.{label}", jd.tri(good), history_rows_via_api=jd.tri(saved))

    def final_holder(self) -> dict[str, Any]:
        """The lock holder once the Main window is up (after any handoff), identified; the chain's verdict is ``jd.cold_chain``: a brief frozen
        launcher that was not seen is unknown, a known other process is a fail."""
        pid = hw.lock_pid(self.app_root)
        return self.identify(pid) if pid and hw.pid_alive(pid) else {}

    def observe_start(self, end: str, chain: str, first: dict[str, Any], final: dict[str, Any], copy: str) -> None:
        """Two facts apart: who holds the profile at the END of a start (image, bytes and command line), and whether the whole chain from
        the frozen exe was SEEN. A first holder missed between two reads leaves the chain unknown (never a full pass) and casts no doubt
        on the end; a known contrary fact in either is a failure."""
        self.observe(end, jd.tri(jd.handed_off(final, copy)), unknown_stops=False, kind=final.get("kind"), command=final.get("command"), read=final.get("pid_read"))
        self.observe(chain, jd.tri(jd.cold_chain(first, final, copy)), unknown_stops=False, first=first["kind"], first_read=first["pid_read"],
                     final=final.get("kind"), final_read=final.get("pid_read"))

    def cold_start(self) -> None:
        self.step = "cold_start"
        if self.by_explorer():  # asked again after the Update, before the shell starts anything that could fetch
            found = self.result["facts"]["explorer_start_source"] = self.configured_source(self.plain_env)
            self.observe("explorer_start_reads_only_the_local_mirror", "pass" if set(found.values()) == {"mirror"} else "unknown")
        first = self.start("cold_start_shortcut")
        final = self.final_holder()
        self.result["facts"]["identity"]["cold_start_final"] = final
        self.observe_start("cold_start_ends_in_the_bundled_python_running_the_profile_launcher",
                           "cold_start_frozen_then_bundled_python_running_the_profile_launcher", first, final, "installed")
        self.history_both("cold_start")
        hwnd, _owner = self.main_window("cold_start")
        self.read_pair_in_window("cold_start_window", hwnd)
        self.versions("cold_start")
        self.brand("cold_start_shortcut", hwnd=hwnd)
        self.quit("installed_second")

    def uninstall_and_reopen(self) -> None:
        i, before = self.inst, manifest(self.app_root)
        self.step = "uninstall"
        quiet = (hw.uninstall_entry(i["uninstall_key"]) or {}).get("QuietUninstallString")
        if not quiet:
            raise sc.Unconfirmed("uninstall", "no_quiet_uninstall_string_in_the_registry_entry")
        argv = [*hw.split_command(quiet), "/SUPPRESSMSGBOXES", "/NORESTART", f"/LOG={self.private / 'uninstall.log'}"]
        self.asked("uninstaller")
        done = subprocess.run(argv, env=self.launch_env, capture_output=True, timeout=600)
        self.saw("uninstaller")
        self.observe("uninstaller_exit_code", jd.tri(done.returncode in i["success_exit_codes"]), exit_code=done.returncode)
        gone = lambda: not (self.install_dir / "Ouroboros.exe").exists() and not self.lnk.exists() and hw.uninstall_entry(i["uninstall_key"]) is None  # noqa: E731
        self.observe("program_shortcut_and_uninstall_entry_removed", jd.tri(bool(sc.poll_until(180.0, gone, interval=2.0))))
        after = manifest(self.app_root)
        self.result["facts"]["profile"] = {"files_before": len(before["files"]), "files_after": len(after["files"]),
                                           "unreadable": len(before["unreadable"]) + len(after["unreadable"])}
        self.observe("profile_bytes_unchanged_by_the_uninstall", jd.same_profile(before, after))
        self.step = "zip_reopen"
        prior = hw.lock_pid(self.app_root)
        self.asked("new_zip_copy")
        subprocess.Popen([str(self.exe)], cwd=str(self.exe.parent), env=self.launch_env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=hw.NEW_GROUP)
        pid = self.adopt_lock_holder("zip_reopen", prior, BOOT_SEC)
        self.saw("new_zip_copy")
        first = self.identify(pid)
        self.await_main_window("zip_reopen")
        self.driver.acquire_main_window(phase="zip_reopen", timeout_sec=60.0)
        final = self.final_holder()
        self.observe_start("zip_copy_ends_in_the_bundled_python_running_the_profile_launcher",
                           "zip_copy_frozen_then_bundled_python_running_the_profile_launcher", first, final, "portable")
        self.history_both("zip_reopen")
        hwnd, _owner = self.main_window("zip_reopen")  # the exact window the ZIP copy shows now: the exchange is read and framed in it
        self.read_pair_in_window("zip_reopen_window", hwnd)
        self.versions("zip_reopen")
        self.brand("zip_reopen", hwnd=hwnd)
        self.quit("zip_reopen")

    def judge_git(self, which: str, root: pathlib.Path, windows: tuple[str, ...]) -> None:
        """Every Git this copy ran is the bundled one, byte-equal to the SHIPPED payload, and the windows asked for hold their
        commands. A foreign or altered Git is a fail; anything not established (no shipped bytes included) is unknown."""
        if not self.gits:
            self.observe(f"bundled_git.{which}", "unknown", stop=False, code="no_shipped_git_bytes_to_compare_with")
            return
        try:
            for attempt in range(3):  # a Git starting during the read shows as a count mismatch: read again
                sinks = self.wh.read_trace_sinks(*self.sinks[which])
                found = gt.sessions(sinks[1], discarded=sinks[2])
                claim = gt.identity(gt.identity_lines(sinks[0]), found, package_root=str(root), git_root=str(root / "_internal" / "git-standalone"),
                                    package_exes=self.gits, folder_exes=lambda rel: self.wh.exe_digests(root, rel))
                if claim["state"] != "unconfirmed" or attempt == 2:
                    break
                time.sleep(3.0)
            bounds = {k: v for k, v in gt.bounds(self.result["marks"], self.attempt).items() if k in windows}
            wins = {k: gt.window(found, gt.when(a), gt.when(b), target=self.b_sha if k == "apply" else "") for k, (a, b) in bounds.items()}
            reqs = {k: v for k, v in gt.phase_requirements(wins).items() if k in windows}
        except wh.Unread as unread:
            self.observe(f"bundled_git.{which}", "unknown", stop=False, code=unread.code)
            return
        self.result["facts"]["git"][which] = {"identity": claim["state"], "requirements": reqs, "bytes_compared_with": "shipped_payload"}
        self.observe(f"bundled_git.{which}", jd.bundled_git(claim["state"], self.control_ok, reqs, windows), stop=False,
                     identity=claim["state"], requirements=reqs)

    def judge_git_traces(self, *, require: bool) -> None:  # #4220's single-sink judgement is replaced: judge_git reads each copy where its folder still exists
        if self.result["facts"]["git"]:
            self.save("git-traces.json", {"sinks": self.result["facts"]["git"]})

    def _run_all_steps(self) -> None:
        if self.cli.mode == "visible":
            return self.visible()
        self.host()
        with fm.FixtureModel(self.tag) as model:
            self.model = model
            self.explorer("pre_install", [
                {"id": "explorer_installer", "folder": str(self.installer.parent), "folder_label": "installer folder", "item": self.installer.name},
                {"id": "explorer_zip_copy_exe", "folder": str(self.package_root), "folder_label": "extracted ZIP copy", "item": "Ouroboros.exe"}])
            self.old_release_seed()
            self.install()
            self.installed_first_launch()
            self.check_installed()
            self.update()
            self.cold_start()
            self.uninstall_and_reopen()
        self.gate()

    # ---- the visible leg: a clean profile, no system Git, the installer as a person opens it, driven by UIA only
    def visible(self) -> None:
        self.step = "host_check"
        self.exe, self.package_root = self.install_dir / "Ouroboros.exe", self.install_dir
        self.host_basics()
        self.launch_env = self._build_launch_env()
        if dirty := self.dirty_inputs():
            raise sc.Unconfirmed("host_check", "inputs_not_clean", names=dirty)
        if self.payload_root is not None:  # the SHIPPED bytes: what the installed tree is compared with, Git included
            self.digests = {p.relative_to(self.payload_root).as_posix().lower(): wh.sha256_file(p) for p in self.payload_root.rglob("*") if p.is_file()}
            self.gits = pb._git_standalone_executables(self.payload_root)  # noqa: SLF001
        self.result["facts"]["new_package"] = {"installer_sha256": self.inst["sha256"].lower()}
        self.ensure_no_system_git()
        self.cu = self._load_uia_skill(pathlib.Path(self.args.candidate_src))
        self.explorer("pre_install", [{"id": "explorer_installer", "folder": str(self.installer.parent), "folder_label": "installer folder", "item": self.installer.name}])
        self.step = "installer_visible"
        self.mark("launched")  # the Git trace window for the first launch opens here
        self.run_ordinary_installer(self.env("installed"))
        self.step = "first_launch"
        pid = self.adopt_lock_holder("first_launch", None, 120.0)
        self.saw("app_by_installer_finish")
        ident = self.identify(pid)
        self.result["facts"]["identity"]["visible_first_launch"] = ident
        win = self.await_setup_window("first_launch")
        self.mark("setup_window_ready")  # the first-run clone is done before the wizard is presented: #4220's bootstrap window
        owner = ident if win.pid == ident["pid"] else self.identify(win.pid)  # the process that owns THIS window, whoever started it
        self.result["facts"]["identity"]["visible_setup_window_owner"] = owner
        self.observe("first_launch_reached_its_own_window_without_system_git", "pass")
        self.observe("first_launch_is_the_installed_frozen_exe", jd.tri(jd.frozen_of(owner, "installed")), unknown_stops=False, kind=owner["kind"],
                     copy=owner["copy"], read=owner["pid_read"], same_process_as_first_seen=win.pid == ident["pid"])
        if self.digests:
            self.observe("first_launch_exe_bytes_equal_the_payload", jd.tri(owner.get("bytes_equal_zip") if jd.read(owner) else None), unknown_stops=False)
        self.observe("first_launch_wizard_control_read_in_that_window", jd.tri(self.setup_control_read(win.hwnd)), stop=False)
        self.brand("visible_first_launch", prefix="Ouroboros", judge=False, hwnd=win.hwnd)
        self.check_installed()
        if self.gits:
            self.control_git_identity()
        self.observe("git_identity_control", "pass" if self.control_ok else "unknown", stop=False)
        self.judge_git("installed", self.install_dir, ("bootstrap",))
        self.gate()

    def run(self) -> int:
        code = super().run()
        if code == 0:
            self.result["code"], self.result["visual"] = "all_checks_passed_frames_pending_visual_inspection", "pending_visual_inspection"
            self.save("result.json", self.result)
        return code
