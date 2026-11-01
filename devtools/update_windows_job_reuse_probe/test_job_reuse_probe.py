"""Falsifying tests of the observation patch and the probe's bookkeeping (stdlib; no real child, never a Windows result).

They prove the patch only adds fields to events, cannot replace the original exception or skip the cleanup, and that
an unmeasured trial cannot read as measured. They do not emulate Windows Jobs.
"""
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import job_reuse_probe as probe  # noqa: E402
import wq_custody as patched  # noqa: E402

spec = importlib.util.spec_from_file_location("wq_custody_original", HERE / "wq_custody.original.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

DENIED = PermissionError(13, "Access is denied.", None, 5)


class Win:
    """Injected answers. ``fail`` maps a call to the exception it raises; ``log`` holds the order of the calls."""

    def __init__(self, fail=None):
        self.fail, self.log = dict(fail or {}), []

    def _call(self, name, value=None):
        self.log.append(name)
        if name in self.fail:
            raise self.fail[name]
        return value

    def job(self):
        return "job"

    def assign(self, job, handle):
        self._call("assign")

    def birth(self, handle):
        return self._call("birth", "777")

    def resume(self, handle):
        self._call("resume")

    def exit_code(self, handle):
        return self._call("exit_code")

    def in_job(self, handle, job):
        return self._call("in_job", False)

    def active(self, job):
        return self._call("active", 0)

    def pids(self, job):
        return self._call("pids", [])

    def terminate(self, job, code):
        self._call("terminate")

    def close(self, job):
        self._call("close")

    def snapshot(self):
        return {}

    def identity(self, pid, job=None, parents=None):
        return {"pid": pid, "alive": False, "birth": "1", "in_job": True}


class Proc:
    pid, _handle = 77, 1

    def __init__(self, win, argv=(), kill_error=None, returncode=0):
        self.win, self.argv, self.kill_error, self.returncode = win, argv, kill_error, returncode

    def kill(self):
        self.win.log.append("kill")
        if self.kill_error:
            raise self.kill_error

    def wait(self, timeout=None):
        self.win.log.append("wait")
        if self.returncode is None:
            raise subprocess.TimeoutExpired("x", timeout)
        return self.returncode


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="jrp-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def spawn(self, module, win, **proc):
        events = []
        raw = types.SimpleNamespace(emit=lambda event, **facts: events.append({"event": event, **facts}))
        custody = module.Custody(win, raw, "F", [])
        fake = types.SimpleNamespace(Popen=lambda *a, **k: Proc(win, **proc), PIPE=-1, DEVNULL=-3)
        raised = None
        with mock.patch.object(module, "subprocess", fake):
            try:
                custody.spawn("direct-server", ["python"], env={}, cwd=self.tmp, logs=self.tmp / "logs")
            except BaseException as exc:
                raised = exc
        return custody, events, raised


class PatchTests(Scratch):
    def test_each_failing_call_names_its_stage(self):
        for stage in ("assign", "birth", "resume"):
            _custody, events, raised = self.spawn(patched, Win({stage: DENIED}))
            self.assertIs(raised, DENIED)
            self.assertEqual(events[-1]["event"], "spawn_refused")
            self.assertEqual(events[-1]["stage"], stage)

    def test_readings_come_before_the_kill(self):
        win = Win({"assign": DENIED})
        _custody, events, _raised = self.spawn(patched, win)
        self.assertEqual(win.log, ["assign", "exit_code", "birth", "in_job", "in_job", "active", "pids", "kill", "wait"])
        observed = events[-1]["observed"]
        self.assertEqual(sorted(observed), ["birth", "exit_code", "in_any_job", "in_this_job", "job_active", "job_pids"])
        json.dumps(events[-1])

    def test_a_failing_reader_cannot_replace_the_refusal_or_skip_cleanup(self):
        fail = {"assign": DENIED, "exit_code": OSError(6, "x"), "in_job": KeyboardInterrupt(), "active": ValueError("y"),
                "pids": RuntimeError("z")}
        win = Win(fail)
        _custody, events, raised = self.spawn(patched, win)
        self.assertIs(raised, DENIED)
        self.assertEqual(win.log[-2:], ["kill", "wait"])
        observed = events[-1]["observed"]
        for key in ("exit_code", "in_this_job", "in_any_job", "job_active", "job_pids"):
            self.assertIn("read_error", observed[key], key)

    def test_a_win_without_readers_is_read_errors_not_an_attribute_failure(self):
        win = Win({"assign": DENIED})
        with mock.patch.object(Win, "exit_code", None):
            _custody, events, raised = self.spawn(patched, win)
        self.assertIs(raised, DENIED)
        self.assertIn("read_error", events[-1]["observed"]["exit_code"])

    def test_the_job_pid_reading_is_bounded(self):
        win = Win({"assign": DENIED})
        with mock.patch.object(Win, "pids", lambda self, job: list(range(1000))):
            _custody, events, _raised = self.spawn(patched, win)
        self.assertEqual(len(events[-1]["observed"]["job_pids"]), 32)

    def test_against_the_original_only_fields_and_read_calls_are_added(self):
        readers = {"exit_code", "in_job", "active", "pids"}
        for fail, proc in (({"assign": DENIED}, {}), ({"resume": DENIED}, {}), ({}, {})):
            old_win, new_win = Win(fail), Win(fail)
            _old, old_events, old_raised = self.spawn(original, old_win, **proc)
            _new, new_events, new_raised = self.spawn(patched, new_win, **proc)
            self.assertIs(old_raised, new_raised)
            self.assertEqual([e["event"] for e in old_events], [e["event"] for e in new_events])
            for old, new in zip(old_events, new_events):
                self.assertEqual(old, {key: new[key] for key in old})
                self.assertLessEqual(set(new) - set(old), {"stage", "since_stop_ms", "observed", "pid", "cleanup"})
            # birth is both a step and a reading here, so the mutating order is compared without it
            self.assertEqual([name for name in old_win.log if name != "birth"],
                             [name for name in new_win.log if name not in readers and name != "birth"])

    def test_a_failed_kill_keeps_the_refusal_and_is_disclosed_beside_it(self):
        kill = OSError(5, "kill")
        win = Win({"assign": DENIED})
        _custody, events, raised = self.spawn(patched, win, kill_error=kill)
        self.assertIs(raised, kill)  # as before: the cleanup's exception is what propagates
        self.assertEqual(win.log[-1], "kill")  # and as before, no wait follows a failed kill
        refused = events[-1]
        self.assertEqual((refused["event"], refused["stage"], refused["error"], refused["pid"]),
                         ("spawn_refused", "assign", repr(DENIED), 77))
        self.assertEqual(refused["cleanup"], {"kill": repr(kill), "wait": "not_run"})
        self.assertEqual(sorted(refused["observed"]), ["birth", "exit_code", "in_any_job", "in_this_job", "job_active",
                                                       "job_pids"])
        json.dumps(refused)

    def test_a_failed_wait_keeps_the_refusal_and_a_clean_run_reads_ok(self):
        _custody, events, raised = self.spawn(patched, Win({"resume": DENIED}), returncode=None)
        self.assertIsInstance(raised, subprocess.TimeoutExpired)
        self.assertEqual((events[-1]["stage"], events[-1]["error"]), ("resume", repr(DENIED)))
        self.assertEqual(events[-1]["cleanup"], {"kill": "ok", "wait": repr(raised)})
        _custody, events, raised = self.spawn(patched, Win({"birth": DENIED}))
        self.assertIs(raised, DENIED)
        self.assertEqual(events[-1]["cleanup"], {"kill": "ok", "wait": "ok"})

    def test_a_recording_failure_cannot_skip_the_cleanup(self):
        win, tried = Win({"assign": DENIED}), []

        def emit(event, **facts):
            tried.append(event)
            if event == "spawn_refused":
                raise OSError(28, "disk full")
        custody = patched.Custody(win, types.SimpleNamespace(emit=emit), "F", [])
        fake = types.SimpleNamespace(Popen=lambda *a, **k: Proc(win), PIPE=-1, DEVNULL=-3)
        with mock.patch.object(patched, "subprocess", fake), self.assertRaises(OSError):
            custody.spawn("direct-server", ["python"], env={}, cwd=self.tmp, logs=self.tmp / "logs")
        self.assertEqual(win.log[-2:], ["kill", "wait"])
        self.assertEqual(tried[-1], "spawn_refused")

    def test_since_stop_is_none_before_any_stop_and_measured_after_one(self):
        custody, events, _raised = self.spawn(patched, Win())
        self.assertIsNone(events[-1]["since_stop_ms"])
        custody.stop_all("x", deadline=1)
        fake = types.SimpleNamespace(Popen=lambda *a, **k: Proc(custody.win), PIPE=-1, DEVNULL=-3)
        with mock.patch.object(patched, "subprocess", fake):
            custody.spawn("next", ["python"], env={}, cwd=self.tmp, logs=self.tmp / "logs")
        self.assertGreaterEqual(events[-1]["since_stop_ms"], 0)


class ProbeWin(Win):
    """A Win for whole trials: ``refuse`` decides each ``assign``; a started process stays a Job member until the
    next terminate; ``hold_after_terminate`` keeps the count above 0 through that many terminates."""

    def __init__(self, refuse=lambda: False, hold_after_terminate=0):
        super().__init__()
        self.refuse, self.hold, self.terminates, self.member = refuse, hold_after_terminate, 0, False

    def pids(self, job):
        return [77] if self.member else []

    def terminate(self, job, code):
        self.log.append("terminate")
        self.terminates += 1
        self.member = False

    def active(self, job):
        return 1 if self.terminates and self.terminates <= self.hold else 0

    def resume(self, handle):
        self.member = True

    def assign(self, job, handle):
        self.log.append("assign")
        if self.refuse():
            raise DENIED


def fake_popen(win, run=True, returncode=0):
    """Starts nothing: the marker argument is touched as the child would have, and only when ``run``."""
    def popen(argv, **_kw):
        if run:
            pathlib.Path(argv[-1]).touch()
        return Proc(win, returncode=returncode)
    return types.SimpleNamespace(Popen=popen, PIPE=-1, DEVNULL=-3, TimeoutExpired=subprocess.TimeoutExpired)


class ProbeTests(Scratch):
    seq = 0

    def trial(self, win, arm, rechecks=0, **popen):
        self.seq += 1
        raw = probe.Recorder(self.tmp / "events.jsonl")
        self.addCleanup(raw.fh.close)
        with mock.patch.object(patched, "subprocess", fake_popen(win, **popen)), \
                mock.patch.object(probe.time, "sleep", lambda s: None), mock.patch.object(probe, "STEP_DEADLINE", 0.05):
            return probe.run_trial(win, raw, "C:\\py\\python.exe", {"seq": self.seq, "arm": arm}, self.tmp, rechecks)

    def test_the_plan_is_finite_within_its_bounds_and_repeatable(self):
        trials = probe.plan()
        self.assertEqual(trials, probe.plan())
        self.assertLessEqual(len(trials) + probe.MAX_RECHECKS, 32)
        self.assertLessEqual(probe.TOTAL_DEADLINE, 180)
        self.assertEqual({arm for arm in (t["arm"] for t in trials)}, set(probe.ARMS))
        self.assertNotEqual([t["arm"] for t in trials[:3]], ["fresh_job"] * 3)

    def test_every_arm_accepted_is_measured_by_the_marker(self):
        for arm in probe.ARMS:
            row = self.trial(ProbeWin(), arm)
            self.assertEqual(row["outcome"], "accepted", arm)
            self.assertEqual(row["teardown"]["status"], "confirmed")

    def test_terminated_arms_really_call_terminate_and_the_others_do_not(self):
        counts = {}
        for arm in probe.ARMS:  # the teardown always terminates once; an arm's own stop is the second
            win = ProbeWin()
            self.trial(win, arm)
            counts[arm] = win.terminates
        self.assertEqual(counts, {"fresh_job": 1, "emptied_no_terminate": 1, "terminated_empty": 2,
                                  "terminated_tree_immediate": 2, "terminated_tree_2s": 2})

    def test_a_refusal_keeps_its_stage_and_readings_and_spends_a_recheck(self):
        starts = iter([False, True, False])  # the tree assigned, the target refused, the recheck assigned
        win = ProbeWin(refuse=lambda: next(starts))
        row = self.trial(win, "terminated_tree_immediate", rechecks=1)
        self.assertEqual((row["outcome"], row["target"]["stage"]), ("refused", "assign"))
        self.assertIn("job_active", row["target"]["observed"])
        self.assertEqual(row["recheck"]["outcome"], "accepted")

    def test_no_recheck_when_the_allowance_is_spent(self):
        starts = iter([False, True])
        row = self.trial(ProbeWin(refuse=lambda: next(starts)), "terminated_tree_immediate", rechecks=0)
        self.assertEqual(row["outcome"], "refused")
        self.assertNotIn("recheck", row)

    def test_a_child_that_left_no_marker_or_exit_is_unverified(self):
        self.assertEqual(self.trial(ProbeWin(), "fresh_job", run=False)["outcome"], "unverified")
        self.assertEqual(self.trial(ProbeWin(), "fresh_job", returncode=None)["outcome"], "unverified")

    def test_an_unconfirmed_stop_never_reaches_the_target(self):
        win = ProbeWin(hold_after_terminate=1)
        row = self.trial(win, "terminated_tree_immediate")
        self.assertEqual(row["outcome"], "setup_unknown")
        self.assertNotIn("target", row)
        self.assertIn("close", win.log)

    def test_a_popen_failure_names_no_stage(self):
        win = ProbeWin()
        fake = fake_popen(win)
        fake.Popen = mock.Mock(side_effect=OSError(2, "no"))
        raw = probe.Recorder(self.tmp / "events.jsonl")
        self.addCleanup(raw.fh.close)
        custody = patched.Custody(win, raw, "F", [])
        with mock.patch.object(patched, "subprocess", fake):
            result = probe.attempt(custody, raw, "py", self.tmp, "target")
        self.assertEqual(result["outcome"], "popen_failed")
        self.assertNotIn("stage", result)

    def test_a_refused_start_whose_cleanup_failed_is_never_popen_failed_nor_complete(self):
        kill = PermissionError(13, "kill denied", None, 5)
        win = ProbeWin(refuse=lambda: True)
        raw = probe.Recorder(self.tmp / "events.jsonl")
        self.addCleanup(raw.fh.close)
        custody = patched.Custody(win, raw, "F", [])
        fake = fake_popen(win)
        fake.Popen = lambda argv, **_kw: Proc(win, kill_error=kill)
        with mock.patch.object(patched, "subprocess", fake):
            result = probe.attempt(custody, raw, "py", self.tmp, "target")
        self.assertEqual((result["outcome"], result["stage"], result["error"], result["pid"]),
                         ("refused", "assign", repr(DENIED), 77))
        self.assertEqual(result["cleanup"], {"kill": repr(kill), "wait": "not_run"})
        trials = [{"seq": 0, "arm": "fresh_job"}]
        row = {**trials[0], "outcome": "refused", "target": result, "teardown": {"status": "confirmed"}}
        summary = probe.summarize(trials, [row], {}, {})
        self.assertEqual(summary["measurement"], "incomplete")
        self.assertEqual(summary["cleanup_uncertain"][0]["cleanup"], result["cleanup"])
        self.assertTrue(summary["target_refused"])

    def test_a_terminated_arm_needs_its_terminate_to_have_succeeded(self):
        for arm in ("terminated_empty", "terminated_tree_immediate", "terminated_tree_2s"):
            win = ProbeWin()
            win.terminate = lambda job, code, win=win: (win.log.append("terminate"), (_ for _ in ()).throw(DENIED))
            row = self.trial(win, arm)
            self.assertIs(row["setup"]["ready"], False, arm)
            self.assertEqual(row["outcome"], "setup_unknown", arm)
            self.assertNotIn("target", row)

    def test_a_setup_refusal_is_not_a_target_refusal_and_reads_incomplete(self):
        win = ProbeWin(refuse=lambda: True)  # the tree's own start into the fresh Job is refused
        row = self.trial(win, "terminated_tree_immediate")
        self.assertEqual(row["outcome"], "setup_unknown")
        summary = probe.summarize([{"seq": 1, "arm": row["arm"]}], [row], {}, {})
        self.assertIs(summary["target_refused"], False)
        self.assertEqual(summary["measurement"], "incomplete")

    def test_only_a_fully_measured_plan_is_complete_and_nothing_says_pass(self):
        trials = [{"seq": 0, "arm": "fresh_job"}, {"seq": 1, "arm": "terminated_tree_immediate"}]
        good = {"teardown": {"status": "confirmed"}}
        accepted = {**trials[0], **good, "outcome": "accepted", "target": {"since_stop_ms": None}}
        refused = {**trials[1], **good, "outcome": "refused",
                   "target": {"outcome": "refused", "stage": "assign", "since_stop_ms": 0.1, "pid": 77,
                              "cleanup": probe.CLEANED}}
        summary = probe.summarize(trials, [accepted, refused], {}, {})
        self.assertEqual(summary["measurement"], "complete")
        self.assertTrue(summary["target_refused"])
        self.assertEqual(summary["refusal_stages"]["terminated_tree_immediate"], {"assign": 1})
        self.assertNotIn("PASS", json.dumps(summary))
        self.assertEqual(probe.summarize(trials, [accepted], {}, {})["measurement"], "incomplete")
        unknown = {**refused, "outcome": "setup_unknown"}
        self.assertEqual(probe.summarize(trials, [accepted, unknown], {}, {})["measurement"], "incomplete")
        lost = {**refused, "teardown": {"status": "unknown"}}
        self.assertEqual(probe.summarize(trials, [accepted, lost], {}, {})["measurement"], "incomplete")

    def test_the_total_deadline_stops_new_trials_and_reads_incomplete(self):
        raw = probe.Recorder(self.tmp / "events.jsonl")
        self.addCleanup(raw.fh.close)
        trials, rows = probe.run(ProbeWin(), raw, "py", self.tmp, deadline=0)
        self.assertEqual(rows, [])
        self.assertEqual(probe.summarize(trials, rows, {}, {})["measurement"], "incomplete")

    def test_manifest_holds_on_this_checkout_lf_or_crlf_and_refuses_a_change(self):
        self.assertTrue(probe.verified_files())
        copy = self.tmp / "copy"
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("__pycache__"))
        for path in copy.glob("*.py"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertTrue(probe.verified_files(copy))
        with (copy / "wq_custody.py").open("a") as fh:
            fh.write("# changed\n")
        with self.assertRaises(SystemExit):
            probe.verified_files(copy)

    def test_children_get_no_inherited_secret(self):
        with mock.patch.dict(probe.os.environ, {"SystemRoot": "C:\\W", "GITHUB_TOKEN": "x", "ACTIONS_RUNTIME_TOKEN": "y"}):
            self.assertEqual(probe.child_env().get("GITHUB_TOKEN"), None)
            self.assertEqual(probe.child_env()["SystemRoot"], "C:\\W")
            self.assertNotIn("ACTIONS_RUNTIME_TOKEN", probe.child_env())


if __name__ == "__main__":
    unittest.main()
