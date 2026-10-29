"""Falsifying tests of instrumentation logic, never simulated NTFS success claims."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
import uuid
import zipfile

import probe
from winapi import MOUNT_POINT, mount_buffer


class Records:
    def __init__(self):
        self.rows = []

    def emit(self, event, **facts):
        self.rows.append({"event": event, **facts})


class FakeProcess:
    pid = 123
    stdout = None

    def __init__(self):
        self.stdin = io.BytesIO()
        self.returncode = None

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("owned gate", timeout)
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


class Portable(unittest.TestCase):
    def setUp(self):
        # Retain fixtures as verification evidence in this equipment workspace.
        # Never use the system temporary directory or clean another run's files.
        evidence = Path(__file__).absolute().parent / ".review-evidence" / "portable-fixtures"
        self.root = evidence / (self._testMethodName + "-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.raw = Records()

    def test_finite_matrix_has_controls(self):
        rows = probe.cases()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(set(rows)), 18)
        for layout, op, _ in rows:
            self.assertIn((layout, op, "none"), rows)

    def test_failed_or_unknown_count_never_means_empty(self):
        for value in (None, {}, {"status": "error", "error": "query failed"},
                      {"status": "ok", "value": None}, {"status": "ok", "value": False},
                      {"status": "ok", "value": "0"}, {"status": "ok", "value": 1}):
            self.assertFalse(probe.confirmed_empty(value), value)
        self.assertTrue(probe.confirmed_empty({"status": "ok", "value": 0}))

    def test_observation_rejects_failed_return_stale_json_noise_and_missing_fields(self):
        data = {"schema": "ntfs-observation/1", "cached_package": {"generation": "B"},
                "selected_marker": {"status": "ok", "value": "B"}, "executable": "active/python.exe",
                "new_import": {"status": "ok", "value": {"generation": "B"}},
                "pyd": {"status": "ok", "value": {}}, "phase": "forward", "pid": 12}
        text = json.dumps(data)
        self.assertEqual(probe.parse_observation(7, text)["status"], "error")
        self.assertEqual(probe.parse_observation(None, text)["status"], "unknown")
        for bad in (text + "\n" + text, "noise\n" + text, "{}", "null", "["):
            self.assertEqual(probe.parse_observation(0, bad)["status"], "error")
        data["new_import"] = {"status": "error", "error": {"type": "ImportError"}}
        parsed = probe.parse_observation(0, json.dumps(data))
        self.assertEqual(parsed["status"], "error")
        self.assertEqual(parsed["value"], data)

    def test_os_refusal_retains_error_code_but_absent_source_is_error(self):
        for code, status in ((32, "refused"), (5, "refused"), (2, "error")):
            exc = OSError("injected native error")
            exc.winerror = code
            result = probe.attempt(Mock(side_effect=exc))
            self.assertEqual(result["status"], status)
            self.assertEqual(result["error"]["winerror"], code)

    def owned(self, win, proc):
        o = probe.Owned.__new__(probe.Owned)
        o.win, o.raw, o.label, o.job = win, self.raw, "test", "only-owned-job"
        o.proc, o.assigned, o.pump = proc, True, None
        o.out, o.err = io.BytesIO(), io.BytesIO()
        return o

    def test_unknown_teardown_is_retained_and_raises(self):
        win = Mock()
        win.active.side_effect = OSError("job query unavailable")
        proc = FakeProcess()
        proc.returncode = 0
        owned = self.owned(win, proc)
        with patch("shutil.rmtree") as remove:
            with self.assertRaises(probe.TeardownUnknown):
                owned.finish()
            remove.assert_not_called()
        win.terminate.assert_called_once_with("only-owned-job")
        self.assertEqual([r for r in self.raw.rows if r["event"] == "teardown"][0]["status"], "unknown")

    def test_gate_exit_alone_does_not_prove_descendants_exited(self):
        win = Mock()
        win.active.side_effect = [1, 0]
        proc = FakeProcess()
        proc.returncode = 0
        self.assertEqual(self.owned(win, proc).finish(), 0)
        win.terminate.assert_called_once_with("only-owned-job")

    def test_failed_job_assignment_never_releases_gate(self):
        win, proc = Mock(), FakeProcess()
        win.assign.side_effect = OSError("assignment refused")
        win.active.return_value = 0
        writes = []
        original_write = proc.stdin.write
        proc.stdin.write = lambda b: (writes.append(b), original_write(b))[1]
        with patch.object(probe.subprocess, "Popen", return_value=proc):
            with self.assertRaisesRegex(OSError, "assignment refused"):
                probe.Owned(win, self.raw, self.root, "gate", ["never-start.exe"], {})
        self.assertEqual(writes, [])
        self.assertEqual(proc.poll(), -9)

    def layout(self):
        fixed, a, b = (self.root / name for name in ("active", "A", "B"))
        fixed.mkdir()
        b.mkdir()
        (fixed / "generation.txt").write_text("A")
        (b / "generation.txt").write_text("B")
        return fixed, {"A": a, "B": b}, {"A": fixed, "B": b}

    def test_first_rename_refusal_never_attempts_second_step(self):
        fixed, homes, locations = self.layout()
        with patch.object(probe.os, "rename", side_effect=PermissionError("held")) as rename:
            probe.select(Mock(), self.raw, "rename", fixed, homes, locations, "B", "forward")
        rename.assert_called_once_with(fixed, homes["A"])
        self.assertEqual((fixed / "generation.txt").read_text(), "A")

    def test_second_rename_failure_reverse_uses_present_saved_source(self):
        fixed, homes, locations = self.layout()
        rename = os.rename

        def fail_incoming(src, dst):
            if src == homes["B"]:
                raise PermissionError("injected second-step refusal")
            rename(src, dst)

        with patch.object(probe.os, "rename", side_effect=fail_incoming):
            probe.select(Mock(), self.raw, "rename", fixed, homes, locations, "B", "forward")
        self.assertFalse(fixed.exists())
        self.assertEqual(locations["A"], homes["A"])
        probe.select(Mock(), self.raw, "rename", fixed, homes, locations, "A", "reverse")
        self.assertEqual((fixed / "generation.txt").read_text(), "A")
        self.assertEqual((homes["B"] / "generation.txt").read_text(), "B")

    def test_junction_remove_refusal_does_not_create_or_follow_target(self):
        fixed, homes, locations = self.layout()
        win = Mock()
        with patch.object(probe, "require_junction"), patch.object(probe.os, "rmdir", side_effect=PermissionError("held")) as rm:
            probe.select(win, self.raw, "junction-replace", fixed, homes, locations, "B", "forward")
        rm.assert_called_once_with(fixed)
        win.junction.assert_not_called()
        self.assertEqual((homes["B"] / "generation.txt").read_text(), "B")

    def test_repoint_refuses_nonjunction(self):
        fixed, homes, locations = self.layout()
        win = Mock()
        probe.select(win, self.raw, "junction-repoint", fixed, homes, locations, "B", "forward")
        win.reparse.assert_not_called()

    def test_missing_native_proof_never_establishes_holder(self):
        ready = {"event": "ready", "kind": "pyd", "observation": {"cached_package": {"generation": "A"}}}
        self.assertFalse(probe.holder_proven("pyd", ready, {}, "x", "x"))
        ready["observation"].update(pyd={"status": "ok", "value": {"roundtrip": True, "mapped_device": "A"}},
                                    expected_pyd_device={"status": "ok", "value": "B"})
        self.assertFalse(probe.holder_proven("pyd", ready, {}, "x", "x"))
        ready["kind"] = "redirector"
        self.assertFalse(probe.holder_proven("redirector", ready, {"status": "error"}, "x", "x"))

    def test_final_path_extended_prefix_does_not_hide_cwd_holder(self):
        with patch.object(probe.os.path, "realpath", side_effect=lambda x: x):
            self.assertTrue(probe.same_path("\\\\?\\C:\\active\\pkg", "C:\\active\\pkg"))

    def test_reparse_buffer_offsets_and_tag(self):
        target = "C:\\disposable\\generation-\u0411"
        data = mount_buffer(target)
        tag, length, reserved = struct.unpack_from("<IHH", data)
        a, n, b, m = struct.unpack_from("<HHHH", data, 8)
        self.assertEqual((tag, length, reserved), (MOUNT_POINT, len(data) - 8, 0))
        self.assertEqual(data[16+a:16+a+n].decode("utf-16-le"), "\\??\\" + target)
        self.assertEqual(data[16+b:16+b+m].decode("utf-16-le"), target)
        for bad in ("relative", "C:relative", "\\\\server\\share"):
            with self.assertRaises(ValueError):
                mount_buffer(bad)

    def test_wheel_records_and_two_real_console_entrypoints(self):
        path = self.root / "fixture.whl"
        probe.wheel(path, "B")
        with zipfile.ZipFile(path) as z:
            info = "holder_fixture-2.0.0.dist-info/"
            for row in z.read(info + "RECORD").decode().splitlines()[:-1]:
                name, digest, size = row.split(",")
                content = z.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
                self.assertEqual(digest, "sha256=" + expected)
                self.assertEqual(int(size), len(content))
            entries = z.read(info + "entry_points.txt").decode()
            self.assertIn("ntfs-observe = holder_fixture.entry:main", entries)
            self.assertIn("ntfs-hold = holder_fixture.entry:main", entries)

    def test_generated_entry_imports_child_from_unrelated_cwd(self):
        wheel = self.root / "fixture.whl"
        probe.wheel(wheel, "A")
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        bootstrap = r'''
import importlib, json, os, sys
from pathlib import Path

child, wheel = map(Path, sys.argv[1:])
source = child.parent
assert source != Path.cwd()
assert all(os.path.abspath(p) != str(source) for p in sys.path)
assert "probe" not in sys.modules and "winapi" not in sys.modules
sys.path.insert(0, str(wheel))
entry = importlib.import_module("holder_fixture.entry")
assert entry.__file__.startswith(str(wheel))
print("SOURCE_ABSENT_BEFORE_ENTRY", flush=True)
sys.argv = ["ntfs-observe", "--help"]
try:
    entry.main()
except SystemExit as exc:
    assert exc.code == 0, exc.code
else:
    raise AssertionError("real child argparse --help did not exit")
origins = {name: sys.modules[name].__file__ for name in ("probe", "winapi")}
assert all(Path(path) == source / (name + ".py") for name, path in origins.items())
print("CHILD_IMPORTS " + json.dumps(origins), flush=True)
'''
        env = {"NTFS_PROBE_CHILD": str(probe.SOURCE / "child.py")}
        # Windows needs these OS paths; provider/Python/user settings are excluded.
        env.update({key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ})
        command = [sys.executable, "-I", "-B", "-c", bootstrap,
                   str(probe.SOURCE / "child.py"), str(wheel)]
        (self.root / "command.json").write_text(json.dumps({"argv": command, "cwd": str(unrelated), "env": env}, indent=2))
        result = subprocess.run(command, cwd=unrelated, env=env, capture_output=True, text=True, timeout=30)
        (self.root / "stdout.txt").write_text(result.stdout)
        (self.root / "stderr.txt").write_text(result.stderr)
        (self.root / "returncode.txt").write_text(str(result.returncode) + "\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SOURCE_ABSENT_BEFORE_ENTRY", result.stdout)
        self.assertIn("CHILD_IMPORTS ", result.stdout)

    def test_semantics_copy_uses_its_own_present_source(self):
        win = Mock()

        def fake_junction(link, target):
            # Exercise the specimen sequence, not junction behavior on this OS.
            link.mkdir()
            (link / "sentinel.txt").write_text((target / "sentinel.txt").read_text())

        win.junction.side_effect = fake_junction
        with patch.object(probe, "require_junction"), patch.object(probe, "path_facts", return_value={}):
            probe.junction_semantics(win, self.raw, self.root / "semantics")
        copied = self.root / "semantics/copytree/copy/sentinel.txt"
        self.assertEqual(copied.read_text(), "must survive")
        self.assertEqual(len({call.args[0] for call in win.junction.call_args_list}), 4)

    def test_semantics_target_loss_aborts_and_preserves_evidence(self):
        win = Mock()
        win.junction.side_effect = lambda link, target: link.mkdir()

        def target_loss(link):
            (link.parent.parent / "target/sentinel.txt").unlink()

        with patch.object(probe, "require_junction"), patch.object(probe, "path_facts", return_value={}), patch("shutil.rmtree", side_effect=target_loss):
            with self.assertRaisesRegex(RuntimeError, "target changed"):
                probe.junction_semantics(win, self.raw, self.root / "semantics")
        self.assertTrue((self.root / "semantics/rmtree-link").exists())
        self.assertFalse((self.root / "semantics/copytree").exists())
        self.assertEqual(self.raw.rows[-1]["event"], "semantics_after")


if __name__ == "__main__":
    unittest.main()
