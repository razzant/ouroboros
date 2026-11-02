"""The system WKWebView composer-clearance check judges honestly without running a native window here.

The native run itself is operator-invoked on macOS (devtools/macos_wkwebview/README.md). These cases pin what its
verdict may conclude from recorded observations, how it sizes a short history, which files its source identity
covers, that an unsupported host never reads as a pass, and that the UI anchors it drives still exist.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from devtools.macos_wkwebview import composer_clearance as check
from tests._shared import retire_synthetic_retention

REPO = pathlib.Path(__file__).resolve().parents[1]


def _sample(width, surface, phase, composer, variant, *, extent, scroll, settled=True):
    """Feed top 0, client 640, composer top 560: the newest row ends at `extent` scrolled to the top."""
    end = {"lastBottom": extent - scroll, "composerTop": 560.0, "feedTop": 0.0, "clientHeight": 640.0,
           "scrollTop": scroll}
    return {"id": f"{width}-{surface}-{phase}-{composer}-{variant}", "width": width, "surface": surface,
            "phase": phase, "composer": composer, "variant": variant, "end": end, "innerHeight": 734,
            "settle": {"settled": settled, "frames": 22}}


def _observation():
    samples = []
    for width in check.WIDTHS:
        for surface in check.SURFACES:
            for phase in check.PHASES:
                for composer in check.COMPOSERS:
                    extent = 2000.0 if phase == "growth" else 600.0
                    samples.append(_sample(width, surface, phase, composer, "current", extent=extent,
                                           scroll=extent - 550.0))
            for composer in check.COMPOSERS:
                for variant in check.VARIANTS:
                    negative = _sample(width, surface, "history", composer, variant, extent=600.0, scroll=0.0)
                    samples.append({**negative, "restored": {"before": 50.0, "after": 50.0}})
    left = {"scrollTop": 50.0, "scrollHeight": 690, "clientHeight": 640, "lastBottom": 550.0, "composerTop": 560.0}
    returns = [{"width": width, "before": left, "shown": "171px", "hidden": "171px",
                "first": {"scrollTop": 50.0, "lastBottom": 550.0, "composerTop": 560.0},
                "presented": {"presentedFrame": 1, "scrollTop": 50.0, "lastBottom": 550.0, "composerTop": 560.0}}
               for width in check.WIDTHS]
    return {"status": "observed", "served_matches_source": True, "widths": list(check.WIDTHS), "samples": samples,
            "returns": returns, "engine_probe": {"flex_column_container": {"endPaddingScrolls": False}}}


def _find(observation, sample_id):
    return next(sample for sample in observation["samples"] if sample["id"] == sample_id)


def test_a_complete_run_whose_negatives_overlap_passes():
    result = check.judge(_observation())
    assert result["verdict"] == "PASS", result
    assert result["historical_defect"] == {"engine_leaves_end_padding_unscrollable": True,
                                           "old_padding_layout": "reproduced"}


def _overlap(obs):
    _find(obs, "1100-project-reopen-multiline-current")["end"]["lastBottom"] = 600.0


def _lost_reserve(obs):
    obs["returns"][1]["hidden"] = "92px"


def _late_first_frame(obs):
    obs["returns"][0]["presented"]["lastBottom"] = 590.0


@pytest.mark.parametrize("break_it", [_overlap, _lost_reserve, _late_first_frame])
def test_a_message_under_the_composer_or_a_lost_reserve_fails(break_it):
    observation = _observation()
    break_it(observation)
    result = check.judge(observation)
    assert result["verdict"] == "FAIL" and len(result["failures"]) == 1, result


def _negative_cleared(obs):
    sample = _find(obs, "390-main-history-empty-no-reserve")
    sample["end"].update(lastBottom=550.0, scrollTop=50.0)


def _old_cleared_where_padding_cannot_scroll(obs):
    _find(obs, "1100-main-history-multiline-old-padding")["end"].update(lastBottom=550.0, scrollTop=50.0)


def _not_short(obs):
    _find(obs, "1100-main-history-empty-current")["end"].update(lastBottom=300.0, scrollTop=0.0)


def _missing(obs):
    obs["samples"] = [s for s in obs["samples"] if s["id"] != "390-project-growth-empty-current"]


def _unsettled(obs):
    _find(obs, "390-main-reopen-empty-current")["settle"]["settled"] = False


def _served_elsewhere(obs):
    obs["served_matches_source"] = False


def _interrupted(obs):
    obs.update(status="deadline", errors=["the driver's 570 s deadline passed"])


def _left_away_from_newest(obs):
    """Run 1: a negative's clamp left Main at its top, under the composer, before it went behind Settings."""
    obs["returns"][0]["before"] = {**obs["returns"][0]["before"], "scrollTop": 0.0, "lastBottom": 600.0}
    obs["returns"][0]["presented"]["lastBottom"] = 600.0


def _departure_unrecorded(obs):
    del obs["returns"][1]["before"]


def _presentation_unobserved(obs):
    obs["returns"][0]["presented"] = None


def _clamp_kept(obs):
    _find(obs, "1100-project-history-multiline-old-padding")["restored"]["after"] = 0.0


@pytest.mark.parametrize("break_it", [_negative_cleared, _old_cleared_where_padding_cannot_scroll, _not_short,
                                      _missing, _unsettled, _served_elsewhere, _interrupted,
                                      _left_away_from_newest, _departure_unrecorded, _clamp_kept,
                                      _presentation_unobserved])
def test_a_condition_the_run_did_not_establish_is_inconclusive_never_a_pass(break_it):
    observation = _observation()
    break_it(observation)
    result = check.judge(observation)
    assert result["verdict"] == "INCONCLUSIVE" and not result["failures"] and result["gaps"], result


def test_an_engine_that_scrolls_end_padding_reports_the_old_defect_as_not_reproduced():
    observation = _observation()
    observation["engine_probe"]["flex_column_container"]["endPaddingScrolls"] = True
    for sample in observation["samples"]:
        if sample["variant"] == "old-padding":
            sample["end"].update(lastBottom=550.0, scrollTop=50.0)
    result = check.judge(observation)
    assert result["verdict"] == "PASS", result
    assert result["historical_defect"] == {"engine_leaves_end_padding_unscrollable": False,
                                           "old_padding_layout": "not reproduced"}


def test_the_first_frame_back_is_judged_as_presented_not_before_its_own_callbacks():
    """Native run 2 and its probes (WebKit 21624.1.16.11.4): Main left at its newest message reads scrollTop 0 at the
    start of the first frame back, because a hidden size container keeps no scroll position there, and that same
    rendering update lands it on the newest message before layout and paint. Only the presented state counts."""
    observation = _observation()
    for back in observation["returns"]:
        back["first"] = {"scrollTop": 0.0, "lastBottom": 471.0, "composerTop": 315.0}
        back["presented"] = {"presentedFrame": 1, "scrollTop": 178.0, "lastBottom": 293.0, "composerTop": 315.0}
    assert check.judge(observation)["verdict"] == "PASS"
    observation["returns"][1]["presented"].update(scrollTop=0.0, lastBottom=471.0)
    result = check.judge(observation)
    assert result["verdict"] == "FAIL" and "as presented (frame 1)" in result["failures"][0], result


def test_an_unsupported_host_is_its_own_verdict():
    result = check.judge({"status": "unsupported", "reason": "no display"})
    assert result == {"verdict": "UNSUPPORTED", "failures": [], "gaps": ["no display"]}


class _ClampingFeed:
    """One transcript as the engine treats it: a negative collapses the scroll range and clamps the position into it;
    removing the negative restores the range, never the position."""

    def __init__(self, top):
        self.top, self.negative = top, False

    def _range(self):
        return 0 if self.negative else 90

    def _geometry(self):
        return {"scrollTop": self.top, "scrollHeight": 640 + self._range(), "clientHeight": 640, "feedTop": 0.0,
                "lastBottom": 600.0 - self.top, "composerTop": 560.0}

    def page(self, name, timeout=45.0, **args):
        if name == "variant":
            self.negative = bool(args["css"])
            self.top = min(self.top, self._range())
            return self.negative
        if name == "restore":
            self.top = min(args["scrollTop"], self._range())
        if name == "measure":
            rest = self._geometry()
            self.top = self._range()
            return {"rest": rest, "end": self._geometry(), "innerHeight": 647}
        return {"settle": {"frames": 22, "settled": True}}.get(name, {"scrollTop": self.top})


def test_a_negative_hands_back_the_position_the_positive_flow_left():
    feed, report = _ClampingFeed(top=90), {"samples": []}
    check.Walk(feed, report, lambda: None, REPO, 1024).sample("main", check.MAIN_FEED, "history", "empty", "no-reserve")
    sample = report["samples"][0]
    assert sample["end"]["scrollTop"] == 0 and check.clears(sample["end"]) is False  # clamped, and overlapping
    assert feed.top == 90 and sample["restored"] == {"before": 90, "after": 90}  # the walk goes on from its own place
    assert check.at_newest(feed._geometry()) and check.clears(feed._geometry())


def test_the_fitted_window_puts_a_short_history_midway_through_the_reserve():
    measured = _sample(1100, "main", "history", "empty", "current", extent=300.0, scroll=0.0)
    assert not check.short_history(measured)
    assert check.fitted_height(measured, 500, 2000) == 500  # never below the window's floor
    height = check.fitted_height(measured, check.LOWEST_HEIGHT, 2000)
    shift = height - measured["innerHeight"]
    end = measured["end"]
    end.update(clientHeight=end["clientHeight"] + shift, composerTop=end["composerTop"] + shift)
    assert check.short_history(measured)
    assert abs((300.0 - end["composerTop"]) - (end["clientHeight"] - 300.0)) <= 1


def test_the_copy_holds_the_runtime_inputs_dirty_edits_included_and_nothing_else(tmp_path):
    runtime = ("VERSION", "server.py", "web/style.css", "web/modules/chat.js", "ouroboros/module.py",
               "ouroboros/uncommitted_new.py", "docs/ARCHITECTURE.md", "skills/demo/SKILL.md", "assets/icon.png")
    elsewhere = (".venv/bin/python", ".venv310/lib/x.py", "venv/x.py", ".ouroboros/state.json", ".local/x",
                 ".env", ".review-drive/a.json", ".review-evidence/run/report.json", "tests/test_x.py",
                 "devtools/tool.py", "scripts/build.py", "Makefile", "web/node_modules/a.js",
                 "ouroboros/__pycache__/module.cpython-310.pyc", "web/.DS_Store")
    for name in runtime + elsewhere:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name, encoding="utf-8")
    assert [path.as_posix() for path in check.product_files(tmp_path)] == sorted(runtime)
    before = check.source_identity(tmp_path)
    for name in elsewhere:
        (tmp_path / name).write_text("churn", encoding="utf-8")
    assert check.source_identity(tmp_path) == before
    (tmp_path / "web/style.css").write_text("changed", encoding="utf-8")
    after = check.source_identity(tmp_path)
    assert after["tree_sha256"] != before["tree_sha256"]
    assert after["layout_files"]["web/style.css"] != before["layout_files"]["web/style.css"]
    shipped = (REPO / "Ouroboros.spec").read_text(encoding="utf-8")
    assert all(f"('{name}', " in shipped for name in check.RUNTIME_INPUTS)  # what the app bundle ships


def test_a_host_without_macos_reports_unsupported_and_writes_its_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check.sys, "platform", "linux")
    out = tmp_path / "evidence"
    assert check.main(["--out", str(out)]) == check.EXIT_CODES["UNSUPPORTED"]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["judgement"]["verdict"] == "UNSUPPORTED"
    assert set(report["instrument"]) == {"composer_clearance.py", "composer_clearance.js"}
    assert "UNSUPPORTED" in capsys.readouterr().out


def test_an_output_inside_the_checkout_is_refused_before_anything_is_made(monkeypatch):
    monkeypatch.setattr(check.sys, "platform", "linux")
    local = REPO / ".composer-clearance-refused-output"
    with pytest.raises(SystemExit) as refused:
        check.main(["--out", str(local)])
    assert refused.value.code == 2 and not local.exists()


class _Child:
    """A contained child as run() stops it: Popen's poll, terminate and wait, and its container's reap and close."""

    def __init__(self, *, running=True, exit=143, left=""):
        self.returncode = None if running else exit
        self._exit, self._left = exit, left

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = self._exit

    def wait(self, timeout=None):
        return self.returncode

    def reap(self):
        return self._left

    def close(self):
        pass


def _run(monkeypatch, tmp_path, *, driver_exit=0, server=None, observation=None):
    """check.run() itself, its observe() seam answering at once: no copy, server or window."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(check.sys, "platform", "darwin")
    monkeypatch.setattr(check, "_pyobjc_present", lambda: True)
    monkeypatch.setattr(check, "ROOT_PARENT", tmp_path)

    def observe(root, out, timeout, report, teardown, children):
        children.append(("server", server or _Child(), server or _Child(), 20.0))
        driver = _Child(running=False, exit=driver_exit)
        children.append(("driver", driver, driver, 10.0))
        report["driver"] = {"exit": driver_exit}
        return observation or _observation()

    monkeypatch.setattr(check, "observe", observe)
    out = tmp_path / "out"
    out.mkdir()
    code = check.run(out, False, 60.0)
    return code, json.loads((out / "report.json").read_text(encoding="utf-8"))


def test_a_normal_run_passes_with_its_server_stopped_as_expected(tmp_path, monkeypatch):
    code, report = _run(monkeypatch, tmp_path)
    assert (code, report["judgement"]["verdict"]) == (0, "PASS"), report["judgement"]
    assert report["teardown"]["server"] == {"running_at_stop": True, "exit": 143, "left_running": None}
    assert not list(tmp_path.glob("ouroboros-wkwebview-*"))


@pytest.mark.parametrize("lifecycle", [
    {"driver_exit": -15},  # the driver died after saving an observed, all-clear observation
    {"server": _Child(left="pid 4242 is still running")},  # its container could not confirm the server gone
    {"server": _Child(running=False, exit=1)},  # the server ended before the run stopped it
])
def test_the_runs_own_processes_must_vouch_for_a_pass(tmp_path, monkeypatch, lifecycle):
    code, report = _run(monkeypatch, tmp_path, **lifecycle)
    assert (code, report["judgement"]["verdict"]) == (3, "INCONCLUSIVE"), report["judgement"]
    assert not report["judgement"]["failures"] and report["judgement"]["gaps"]
    if "root" in report:
        assert retire_synthetic_retention(report["root"], "composer clearance check root kept")


def test_a_cleanup_gap_keeps_the_root(tmp_path, monkeypatch):
    _, report = _run(monkeypatch, tmp_path, server=_Child(left="pid 4242 is still running"))
    assert (pathlib.Path(report["root"]) / "OUROBOROS_RETAINED.txt").is_file()
    assert retire_synthetic_retention(report["root"], "composer clearance check root kept")


def test_a_geometric_fail_and_an_unsupported_host_stand_whatever_the_driver_did(tmp_path, monkeypatch):
    overlapping = _observation()
    _overlap(overlapping)
    code, report = _run(monkeypatch, tmp_path / "fail", driver_exit=-15, observation=overlapping)
    assert (code, report["judgement"]["verdict"]) == (1, "FAIL")
    code, report = _run(monkeypatch, tmp_path / "unsupported", driver_exit=4,
                        observation={"status": "unsupported", "reason": "no display"})
    assert (code, report["judgement"]["verdict"]) == (4, "UNSUPPORTED")


def test_the_ui_anchors_the_check_drives_still_exist():
    sources = "\n".join(path.read_text(encoding="utf-8") for path in (REPO / "web").rglob("*")
                        if path.suffix in {".js", ".css", ".html"}
                        and not {"tests", "vendor", "node_modules"} & set(path.relative_to(REPO).parts))
    page = check.PAGE_SCRIPT.read_text(encoding="utf-8")
    for anchor in ('id="chat-messages"', "chat-input-area", "chat-bubble", "nav-project-row", "data-project-id",
                   "project-panel-close", "data-nav-page", "page-settings", "dockedMinWidth", "--chat-input-reserve",
                   "chat-messages::after"):
        assert anchor in sources, anchor
    for name in ("ready", "settle", "measure", "fill", "draft", "click", "emit", "variant", "leaveAndReturn",
                 "probe", "showing", "gone", "viewport"):
        assert f"{name}(" in page, name
