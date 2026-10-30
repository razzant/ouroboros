"""Falsifying tests of the equipment's own logic (stdlib unittest; portable, never a Windows result).

They prove that refuted, unmeasured, historical or rejected evidence cannot reach PASS or exit 0, that the
fixture faults fail for their intended reason, and that the builders write what the cases assume. They do
not emulate NTFS, Windows Jobs, the released app or the product; those are measured only on Windows.
Scratch lives in one private temporary directory per test, removed only by that test.
"""
import ast
import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import zipfile
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qualify  # noqa: E402
import wq_cases  # noqa: E402
import wq_custody  # noqa: E402
import wq_fixtures as fx  # noqa: E402
import wq_product  # noqa: E402
import wq_win  # noqa: E402
from wq_custody import Custody, classify, identities, known_roles, launcher_ids, unread  # noqa: E402
from wq_verdict import (EXIT, FAIL, MODES, NOT_RUN, PASS, REQUIRED, UNKNOWN, Case, CaseStop,  # noqa: E402
                        activation_request, closed_adoption, conjoin, exit_code, generation_seen, layers_equal,
                        prepared_environment, prepared_ready, same_path, select, settled, under, witness_for)

PRIOR_DRAFT = HERE.parent / "prior-draft" / "update_windows_qualification.py"
FUTURE_MODULE = '"""A module like ouroboros/gateway/router.py."""\nfrom __future__ import annotations\n\nVALUE = 1\n'


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="winqual-portable-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def python(self, code, *args, env=None, cwd=None, stdin=None):
        """A child interpreter that sees only what the test puts on its path."""
        base = {key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "WINQUAL"))}
        return subprocess.run([sys.executable, "-B", "-c", code, *map(str, args)], input=stdin, capture_output=True,
                              text=True, timeout=120, env={**base, **(env or {})}, cwd=cwd)


class VerdictTests(unittest.TestCase):
    def test_only_literal_true_passes_and_unmeasured_is_not_pass(self):
        case = Case("A")
        for truthy in (1, "yes", {"ok": True}, [True]):
            self.assertFalse(case.check("truthy", truthy))
        self.assertEqual(case.status, FAIL)
        unknown = Case("B")
        unknown.check("measured", True)
        self.assertIsNone(unknown.check("deadline", None))
        self.assertEqual(unknown.status, UNKNOWN)
        self.assertEqual(Case("C").status, UNKNOWN, "a case that measured nothing proves nothing")
        forced = Case("D")
        forced.not_run("blocked")
        forced.check("refuted", False)
        self.assertEqual(forced.status, FAIL, "a refuted check outranks NOT_RUN")
        good = Case("E")
        good.check("one", True)
        good.check("two", True)
        self.assertEqual(good.status, PASS)
        with self.assertRaises(CaseStop):
            Case("F").require("needed", None)
        claimed = Case("G", required=("first", "second"))
        claimed.check("first", True)
        self.assertEqual((claimed.status, claimed.missing), (UNKNOWN, ["second"]), "a skipped claim cannot pass")
        claimed.check("second", True)
        self.assertEqual(claimed.status, PASS)

    def test_selection_refuses_unknown_repeated_and_marks_subsets_partial(self):
        self.assertEqual(select(None), (REQUIRED, False))
        self.assertEqual(select("all"), (REQUIRED, False))
        self.assertEqual(select("C,A"), (("A", "C"), True))
        self.assertEqual(select(",".join(REQUIRED)), (REQUIRED, False))
        for bad in ("A,Z", "A,A", ",", "E"):
            with self.assertRaises(ValueError):
                select(bad)

    def test_an_incomplete_selection_never_exits_0_whatever_the_caller_says(self):
        self.assertEqual(exit_code({"A": PASS}, ("A",), False, True), EXIT["partial"])
        self.assertEqual(exit_code({name: PASS for name in REQUIRED}, REQUIRED[:-1], False, True), EXIT["partial"])

    def test_exit_code_is_zero_only_for_the_complete_passing_custodied_run(self):
        passing = {name: PASS for name in REQUIRED}
        self.assertEqual(exit_code(passing, REQUIRED, False, True), 0)
        self.assertEqual(exit_code(passing, REQUIRED, False, False), EXIT["custody"])
        self.assertEqual(exit_code(passing, REQUIRED, False, None), EXIT["custody"])
        self.assertEqual(exit_code({**passing, "C": FAIL}, REQUIRED, False, True), EXIT["fail"])
        self.assertEqual(exit_code({**passing, "D": UNKNOWN}, REQUIRED, False, True), EXIT["unknown_or_not_run"])
        missing = dict(passing)
        del missing["E.F"]
        self.assertEqual(exit_code(missing, REQUIRED, False, True), EXIT["unknown_or_not_run"])
        self.assertEqual(exit_code({"A": PASS}, ("A",), True, True), EXIT["partial"])
        self.assertEqual(exit_code({}, (), False, True), EXIT["unknown_or_not_run"])

    def test_false_verdict_and_rejected_activation_fail_the_process(self):
        self.assertNotEqual(qualify.demonstrate_false_verdict(), 0)
        done = subprocess.run([sys.executable, "-B", str(HERE / "qualify.py"), "--demonstrate-false-verdict"],
                              capture_output=True, text=True, timeout=120)
        self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn('"B": "FAIL"', done.stdout)

    def test_cli_refuses_bad_selection_and_measures_nothing_off_windows(self):
        self.assertEqual(qualify.main(["--cases", "A,nonsense"]), EXIT["usage"])
        if os.name != "nt":
            self.assertEqual(qualify.main([]), EXIT["unknown_or_not_run"])

    def test_activation_request_binds_status_candidate_and_adoption(self):
        good = {"status": "ok", "restarting": True, "candidate": {"candidate_id": "c1",
                                                                 "activation": {"adoption_id": "0123456789ab"}}}
        self.assertEqual(activation_request(200, good, "c1")[0], "0123456789ab")
        self.assertEqual(activation_request(409, good, "c1")[0], "", "a rejected activation is never accepted")
        self.assertEqual(activation_request(None, good, "c1")[0], "")
        self.assertEqual(activation_request(200, good, "c2")[0], "", "another candidate")
        self.assertEqual(activation_request(200, {**good, "restarting": False}, "c1")[0], "")
        bad_id = json.loads(json.dumps(good))
        bad_id["candidate"]["activation"]["adoption_id"] = "../../x"
        self.assertEqual(activation_request(200, bad_id, "c1")[0], "")
        self.assertEqual(activation_request(500, "Internal Server Error", "c1")[0], "")

    def test_only_this_attempts_closed_row_counts(self):
        rows = [{"type": "body_adoption_closed", "adoption_id": "aaaaaaaaaaaa", "outcome": "adopted"},
                {"type": "body_adoption_unconfirmed", "adoption_id": "bbbbbbbbbbbb"}]
        self.assertIsNone(closed_adoption(rows, "bbbbbbbbbbbb"), "an unconfirmed row is not a closed one")
        self.assertIsNone(closed_adoption(rows, "cccccccccccc"), "a historical row of another attempt")
        self.assertIsNone(closed_adoption(rows, ""))
        self.assertEqual(closed_adoption(rows, "aaaaaaaaaaaa")["outcome"], "adopted")
        self.assertIsNone(closed_adoption(rows + rows[:1], "aaaaaaaaaaaa"), "an ambiguous attempt is not evidence")

    def test_settlement_needs_record_and_closed_handoff_to_agree(self):
        adopted = {"state": "activated", "activation": {"outcome": "adopted"}}
        self.assertTrue(settled(adopted, {"outcome": "adopted"}, "adopted"))
        self.assertFalse(settled(adopted, {"outcome": "returned:import_probe_failed"}, "adopted"))
        self.assertIsNone(settled(None, {"outcome": "adopted"}, "adopted"))
        self.assertIsNone(settled(adopted, None, "adopted"), "a record alone (e.g. last_activation) is not enough")
        retry = {"state": "ready", "activation": {"outcome": "returned", "decision": {
            "trigger": "environment_selection_failed", "retry": True}}}
        self.assertTrue(settled(retry, {"outcome": "returned:environment_selection_failed"}, "returned_retry"))
        no_retry = json.loads(json.dumps(retry))
        no_retry["activation"]["decision"]["retry"] = False
        self.assertFalse(settled(no_retry, {"outcome": "returned:environment_selection_failed"}, "returned_retry"))
        probe = {"state": "failed", "activation": {"outcome": "returned", "decision": {"trigger": "import_probe_failed"}}}
        self.assertTrue(settled(probe, {"outcome": "returned:import_probe_failed"}, "returned"))
        self.assertFalse(settled(probe, {"outcome": "returned:import_probe_failed"}, "adopted"))

    def test_prepared_environment_must_be_the_measured_kind_beside_the_fixed_path(self):
        fixed = "D:\\wq\\U\\app root\\data\\state\\python-userbase"
        record = {"environment": {"same": False, "generation": {
            "kind": "user_site", "current": fixed.upper(), "user_base": fixed + ".update-abc"}}}
        self.assertTrue(prepared_environment(record, "user_site", fixed)[0])
        self.assertFalse(prepared_environment(record, "venv", fixed)[0], "the wrong selected environment")
        self.assertFalse(prepared_environment(record, "user_site", "D:\\other")[0])
        unchanged = json.loads(json.dumps(record))
        unchanged["environment"]["same"] = True
        self.assertFalse(prepared_environment(unchanged, "user_site", fixed)[0])
        elsewhere = json.loads(json.dumps(record))
        elsewhere["environment"]["generation"]["user_base"] = "D:\\elsewhere\\gen"
        self.assertFalse(prepared_environment(elsewhere, "user_site", fixed)[0], "not a sibling of the fixed path")
        self.assertFalse(prepared_environment({}, "user_site", fixed)[0])

    def test_windows_path_rules(self):
        self.assertTrue(same_path("C:\\A\\b\\", "c:/a/B"))
        self.assertTrue(same_path("\\\\?\\C:\\a\\b", "C:\\a\\b"))
        self.assertFalse(same_path("", ""))
        self.assertFalse(same_path("C:\\a\\b", "C:\\a\\bc"))
        self.assertTrue(under("C:\\a\\b\\c.txt", "C:\\A\\B"))
        self.assertFalse(under("C:\\a\\bc\\x", "C:\\a\\b"))
        self.assertTrue(under("C:\\a\\b", "C:\\a\\b"))

    def test_witness_binds_pid_birth_and_time(self):
        rows = [{"pid": 7, "birth": "100", "time_ns": 5, "where": "w"},
                {"pid": 7, "birth": "200", "time_ns": 9, "where": "w"}]
        self.assertEqual(witness_for(rows, {"pid": 7, "birth": "100"}, "w")["time_ns"], 5)
        self.assertIsNone(witness_for(rows, {"pid": 7, "birth": "300"}, "w"), "a reused PID is another process")
        self.assertIsNone(witness_for(rows, {"pid": 7, "birth": "100"}, "w", since_ns=6), "older than the attempt")
        seen = {"prefix": "C:\\v", "package": {"version": "2.0", "realpath": "C:\\v\\Lib\\site-packages\\p\\__init__.py"}}
        self.assertTrue(generation_seen(seen, "2.0", "C:\\v\\Lib\\site-packages", "c:\\V"))
        self.assertFalse(generation_seen(seen, "1.0", "C:\\v\\Lib\\site-packages"))
        self.assertFalse(generation_seen(seen, "2.0", "C:\\other"))
        self.assertFalse(generation_seen(seen, "2.0", "C:\\v\\Lib\\site-packages", "C:\\other"))
        self.assertIsNone(generation_seen(None, "2.0", "C:\\v"))

    def test_layers_compare_only_complete_readings_and_always_answer_a_pair(self):
        before = {"head": "a" * 40, "branch": "refs/heads/ouroboros", "index": "b" * 64, "status": "c" * 64,
                  "files": {"qualification/H.txt": "d" * 64}}
        self.assertEqual(layers_equal(before, dict(before)), (True, {"changed": []}))
        self.assertEqual(layers_equal(before, {**before, "index": "e" * 64}), (False, {"changed": ["index"]}))
        unreadable = ({"error": "git failed"}, None, {}, {**before, "head": ""}, {**before, "index": None},
                      {k: v for k, v in before.items() if k != "status"},
                      {**before, "files": {"qualification/H.txt": None}}, {**before, "files": None})
        for broken in unreadable:
            for pair in ((broken, before), (before, broken), (broken, broken)):
                same, facts = layers_equal(*pair)  # always a pair: callers unpack it
                self.assertIsNone(same, (pair, facts))
                self.assertIn("unmeasured", facts)

    def test_roles_from_command_lines(self):
        roles = classify([{"image": "C:\\x\\Ouroboros.exe", "cmdline": "\"C:\\x\\Ouroboros.exe\" --no-ui"},
                          {"image": "C:\\p\\python.exe", "cmdline": "python C:\\r\\launcher.py --seed-bundle x"},
                          {"image": "C:\\p\\python.exe", "cmdline": "python C:\\r\\server.py"},
                          {"image": "C:\\p\\git.exe", "cmdline": "git status"}, {}])
        self.assertEqual([len(roles[key]) for key in ("frozen", "launcher", "server", "other")], [1, 1, 1, 2])

    @unittest.skipUnless(PRIOR_DRAFT.is_file(), "the retained prior draft is not placed beside this copy")
    def test_prior_draft_was_fail_open_and_is_retained_unchanged(self):
        tree = ast.parse(PRIOR_DRAFT.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        returns = [node.value.value for node in ast.walk(main) if isinstance(node, ast.Return)]
        self.assertEqual(returns, [0, 0], "negative evidence: the retained draft returned 0 on every path")
        own = ast.parse((HERE / "qualify.py").read_text(encoding="utf-8"))
        own_main = next(node for node in own.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        constants = [node.value.value for node in ast.walk(own_main) if isinstance(node, ast.Return)
                     and isinstance(node.value, ast.Constant)]
        self.assertNotIn(0, constants, "the driver never returns a literal 0")


class FixtureTests(Scratch):
    def test_fault_is_inserted_after_future_imports_and_compiles_while_the_prior_prepend_does_not(self):
        edited = fx.insert_after_future(FUTURE_MODULE, fx.FAULT_LINES + fx.WITNESS_LINES)
        compile(edited, "router.py", "exec")
        lines = edited.splitlines()
        self.assertLess(lines.index("from __future__ import annotations"), lines.index(fx.FAULT_LINES[0]))
        prior = ('import ouroboros_qual_probe\nassert ouroboros_qual_probe.VERSION != "4.0", '
                 '"qualification: these packages break the tree"\n')
        with self.assertRaises(SyntaxError, msg="the retained draft's C fixture failed before its fault"):
            compile(prior + FUTURE_MODULE, "queue.py", "exec")
        crlf = fx.insert_after_future(FUTURE_MODULE.replace("\n", "\r\n"), fx.WITNESS_LINES)
        self.assertIn("annotations\r\n" + fx.WITNESS_LINES[0] + "\r\n", crlf)
        self.assertEqual(crlf.count("\n"), crlf.count("\r\n"), "the file keeps its own line endings")
        with self.assertRaises(SyntaxError):
            fx.insert_after_future(FUTURE_MODULE, ("def broken(:",))

    def write_package(self, version):
        root = self.tmp / f"site-{version}"
        for name, text in fx.package_files(version).items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text, encoding="utf-8")
        return root

    def test_fault_fails_only_on_the_new_version_with_the_intended_diagnostic(self):
        code = "import sys; sys.path.insert(0, sys.argv[1]); " + fx.FAULT_LINES[0].split("  #")[0]
        old = self.python(code, self.write_package("3.0"))
        self.assertEqual(old.returncode, 0, old.stderr)
        new = self.python(code, self.write_package("4.0"))
        self.assertNotEqual(new.returncode, 0)
        self.assertIn("cannot import name 'stable_api' from 'ouroboros_qual_probe'", new.stderr)

    def test_witness_hold_and_report_protocol(self):
        site = self.write_package("2.0")
        witness_dir = self.tmp / "witness"
        witness_dir.mkdir()
        done = self.python("import sys; sys.path.insert(0, sys.argv[1]); import ouroboros_qual_probe as q; "
                           "print(q.witness('unit'))", site, env={"WINQUAL_WITNESS_DIR": str(witness_dir)})
        self.assertEqual(done.returncode, 0, done.stderr)
        rows = [json.loads(path.read_text()) for path in witness_dir.glob("pkg-*.json")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["package"]["version"], "2.0")
        self.assertEqual(rows[0]["native"]["status"], "absent", "no native image without the wheel's pyd")
        self.assertTrue(rows[0]["package"]["realpath"].startswith(os.path.realpath(site)))
        quiet = self.python("import sys; sys.path.insert(0, sys.argv[1]); import ouroboros_qual_probe as q; "
                            "print(q.witness('unit'))", site, env={"WINQUAL_WITNESS_DIR": str(self.tmp / "absent")})
        self.assertEqual((quiet.returncode, quiet.stdout.strip()), (0, "None"), "a witness never raises into the product")
        held = self.tmp / "held dir"
        held.mkdir()
        session = self.python("import runpy, sys; sys.path.insert(0, sys.argv[1]); sys.argv[1:] = sys.argv[2:]; "
                              "runpy.run_module('ouroboros_qual_probe', run_name='__main__', alter_sys=True)",
                              site, "hold", "--cwd", held,
                              stdin=json.dumps({"id": 1, "op": "report"}) + "\n" + json.dumps({"id": 2, "op": "exit"}) + "\n")
        replies = [json.loads(line) for line in session.stdout.splitlines()]
        self.assertEqual([row["event"] for row in replies], ["ready", "report", "exit"], session.stderr)
        self.assertEqual([row.get("id") for row in replies[1:]], [1, 2])
        self.assertTrue(os.path.samefile(replies[0]["cwd"], held), "the holder's working directory is the held one")

    def test_wheel_record_tags_and_console_script(self):
        native = fx.build_wheel(self.tmp, "2.0", b"MZ-not-a-real-pyd")
        pure = fx.build_wheel(self.tmp, "3.0")
        self.assertTrue(native.name.endswith("-cp310-cp310-win_amd64.whl"))
        self.assertTrue(pure.name.endswith("-py3-none-any.whl"))
        with zipfile.ZipFile(native) as archive:
            names = set(archive.namelist())
            info = "ouroboros_qual_probe-2.0.dist-info/"
            self.assertIn("ouroboros_qual_probe/_bz2.pyd", names)
            self.assertIn("Root-Is-Purelib: false", archive.read(info + "WHEEL").decode())
            self.assertIn("ouroboros-qual-tool = ouroboros_qual_probe:main", archive.read(info + "entry_points.txt").decode())
            import base64
            import hashlib
            for line in archive.read(info + "RECORD").decode().splitlines():
                name, digest, size = line.split(",")
                if name.endswith("RECORD"):
                    continue
                data = archive.read(name)
                self.assertEqual(digest, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode()
                                 .rstrip("="))
                self.assertEqual(int(size), len(data))
        with self.assertRaises(FileExistsError):
            fx.build_wheel(self.tmp, "3.0")

    def test_lock_pin(self):
        self.assertEqual(fx.pin("a==1\nb==2\n", "4.0"), "a==1\nb==2\nouroboros-qual-probe==4.0\n")
        self.assertEqual(fx.pin("a==1\r\n", "1.0"), "a==1\r\nouroboros-qual-probe==1.0\r\n")
        with self.assertRaises(ValueError):
            fx.pin("ouroboros-qual-probe==1.0\n", "2.0")

    def test_target_commits_carry_only_the_declared_edits(self):
        repo = self.tmp / "candidate"
        (repo / "ouroboros" / "gateway").mkdir(parents=True)
        (repo / fx.ROUTER).write_text(FUTURE_MODULE, encoding="utf-8")
        (repo / fx.LOCK).write_text("# exported\nanyio==4.0\n", encoding="utf-8")
        (repo / "README.md").write_text("x\n", encoding="utf-8")
        env = {**os.environ, **fx.IDENTITY, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        for args in (["init", "--quiet"], ["add", "-A"], ["commit", "--quiet", "-m", "candidate"]):
            subprocess.run(["git", "-C", str(repo), *args], check=True, env=env, capture_output=True)
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        commits = fx.build_commits(repo, sha, self.tmp / "work")
        self.assertEqual(sorted(commits), ["S0", "T2", "T3", "T4", "T5"])

        def show(commit, path):
            return subprocess.run(["git", "-C", str(self.tmp / "work"), "show", f"{commit}:{path}"], capture_output=True,
                                  text=True, check=True).stdout

        def parent(commit):
            return subprocess.run(["git", "-C", str(self.tmp / "work"), "rev-parse", f"{commit}^"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        expected_parent = {"S0": sha, "T2": commits["S0"], "T3": commits["T2"], "T4": commits["T3"], "T5": commits["S0"]}
        for name, version in fx.VERSIONS.items():
            self.assertEqual(parent(commits[name]), expected_parent[name])
            router = show(commits[name], fx.ROUTER)
            compile(router, fx.ROUTER, "exec")
            self.assertEqual(show(commits[name], fx.LOCK).splitlines()[-1], f"ouroboros-qual-probe=={version}")
            self.assertEqual(fx.FAULT_LINES[0] in router, name == "T4")
            self.assertIn(fx.WITNESS_LINES[1], router)

    def test_provisioning_git_flow_seed_clone_mirror_mapping_and_ignored_stub(self):
        env = {**os.environ, **fx.IDENTITY, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

        def run(*args, cwd):
            return subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, text=True,
                                  check=True).stdout.strip()
        candidate = self.tmp / "candidate"
        (candidate / "ouroboros" / "gateway").mkdir(parents=True)
        (candidate / fx.ROUTER).write_text(FUTURE_MODULE, encoding="utf-8")
        (candidate / fx.LOCK).write_text("anyio==4.0\n", encoding="utf-8")
        run("init", "--quiet", "-b", "ouroboros", cwd=candidate)
        run("add", "-A", cwd=candidate)
        run("commit", "--quiet", "-m", "released source", cwd=candidate)
        released = run("rev-parse", "HEAD", cwd=candidate)
        internal = self.tmp / "released app" / "Ouroboros" / "_internal"
        internal.mkdir(parents=True)
        run("bundle", "create", str(internal / "repo.bundle"), "refs/heads/ouroboros", cwd=candidate)
        (candidate / "README.md").write_text("candidate\n", encoding="utf-8")
        run("add", "-A", cwd=candidate)
        run("commit", "--quiet", "-m", "candidate", cwd=candidate)
        sha = run("rev-parse", "HEAD", cwd=candidate)
        commits = fx.build_commits(candidate, sha, self.tmp / "fixture-work")
        url = "https://example.invalid/official/ouroboros.git"
        ctx = types.SimpleNamespace(run_root=self.tmp / "run", evidence=self.tmp / "run" / "evidence",
                                    git_dir=os.path.dirname(shutil.which("git")), wheelhouse=self.tmp / "wheelhouse",
                                    fixture_repo=self.tmp / "fixture-work", commits=commits, equipment=HERE,
                                    raw=types.SimpleNamespace(emit=lambda *a, **k: None),
                                    release={"internal": internal, "official_urls": [url],
                                             "manifest": {"source_sha": released}})
        with mock.patch.dict(os.environ, {"SystemRoot": os.environ.get("SystemRoot") or str(self.tmp / "Windows")}):
            mode = wq_cases.Mode(ctx, "U")
            mode.provision()
            mode_env = mode.env()
            repo = mode.install.repo
            self.assertEqual(wq_product.git(repo, "rev-parse", "HEAD", env=mode_env), commits["S0"])
            self.assertEqual(wq_product.git(repo, "status", "--porcelain", env=mode_env), "", "the stub is ignored")
            self.assertTrue((repo / "sitecustomize.py").is_file())
            self.assertEqual(wq_product.git(repo, "rev-parse", "ouroboros-stable", env=mode_env), released)
            self.assertEqual(wq_product.git(repo, "remote", env=mode_env), "", "no origin pointing at the seed")
            mode.mirror_to("T2")
            listed = wq_product.git(repo, "ls-remote", url, "refs/heads/ouroboros", env=mode_env)
            self.assertTrue(listed.startswith(commits["T2"]), "the official URL resolves to this mode's mirror")
            self.assertIn(" ", str(mode.install.app_root))

    def test_stub_label_and_checks_only_replacement(self):
        stub_tree = ast.parse((HERE / "stub" / "sitecustomize.py").read_text(encoding="utf-8"))
        label = next(node.value.value for node in stub_tree.body if isinstance(node, ast.Assign)
                     and getattr(node.targets[0], "id", "") == "LABEL")
        self.assertEqual(label, fx.STUB_LABEL)
        product = self.tmp / "checkout"
        (product / "supervisor").mkdir(parents=True)
        (product / "supervisor" / "__init__.py").write_text("", encoding="utf-8")
        (product / "supervisor" / "update_checks.py").write_text(
            "def workspace_checks(row):\n    return {'ok': False, 'real': True}\n"
            "def run_tests(*a):\n    return 'real run_tests'\n", encoding="utf-8")
        shutil.copyfile(HERE / "stub" / "sitecustomize.py", product / "sitecustomize.py")
        witness = self.tmp / "witness"
        witness.mkdir()
        code = ("import json; from supervisor.update_checks import workspace_checks, run_tests; "
                "print(json.dumps([workspace_checks({'candidate_id': 'c9'}), run_tests()]))")
        stubbed = self.python(code, env={"PYTHONPATH": str(product), "WINQUAL_CHECKS_STUB": "1",
                                         "WINQUAL_WITNESS_DIR": str(witness)}, cwd=self.tmp)
        checks, other = json.loads(stubbed.stdout)
        self.assertEqual((checks["ok"], checks["stub"], checks["candidate_id"]), (True, fx.STUB_LABEL, "c9"))
        self.assertEqual(other, "real run_tests", "only workspace_checks is replaced")
        starts = [json.loads(path.read_text()) for path in witness.glob("start-*.json")]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["prefix"], sys.prefix)
        plain = self.python(code, env={"PYTHONPATH": str(product)}, cwd=self.tmp)
        self.assertEqual(json.loads(plain.stdout)[0], {"ok": False, "real": True}, "inert without its switch")


class ProductReadTests(Scratch):
    def test_console_launcher_split_and_shebang(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("__main__.py", "print('x')\n")
        line = b'#!"C:\\app root\\runtime venv\\Scripts\\python.exe"\r\n'
        data = b"MZ stub bytes" + line + archive.getvalue()
        stub, shebang, rest = wq_product.console_launcher_parts(data)
        self.assertEqual(stub, b"MZ stub bytes")
        self.assertEqual(shebang, line.strip().decode())
        self.assertEqual(rest[:2], b"\r\n")
        self.assertTrue(zipfile.ZipFile(io.BytesIO(data)).testzip() is None)
        python = "c:\\APP ROOT\\runtime venv\\Scripts\\python.exe"
        self.assertTrue(wq_product.shebang_names(shebang, python))
        self.assertFalse(wq_product.shebang_names("#!C:\\app root\\runtime venv\\Scripts\\python.exe", python),
                         "a path with a space must be quoted")
        self.assertFalse(wq_product.shebang_names(shebang, "C:\\app root\\runtime venv.update-1\\Scripts\\python.exe"))
        self.assertIsNone(wq_product.console_launcher_parts(b"no archive here"))

    def test_inventory_identity_survives_rename_not_reinstall(self):
        site = self.tmp / "site-packages"
        (site / "ouroboros_qual_probe-2.0.dist-info").mkdir(parents=True)
        (site / "Some_Pkg-1.2.3.dist-info").mkdir()
        rows = wq_product.inventory(site)
        self.assertEqual({name: row["version"] for name, row in rows.items()},
                         {"ouroboros-qual-probe": "2.0", "some-pkg": "1.2.3"})
        moved = self.tmp / "moved"
        site.rename(moved)
        self.assertEqual(wq_product.inventory(moved), rows, "a rename keeps every identity")
        shutil.rmtree(moved / "Some_Pkg-1.2.3.dist-info")
        (moved / "Some_Pkg-1.2.3.dist-info").mkdir()
        self.assertFalse(wq_product.same_inventory(rows, wq_product.inventory(moved)), "a reinstall is a new directory")
        self.assertIsNone(wq_product.inventory(self.tmp / "absent"))
        self.assertFalse(wq_product.same_inventory(None, None), "an absent package path proves nothing")


class FlowTests(Scratch):
    """Mode steps over fake product answers: a rejected or historical activation can never pass."""

    def mode(self, install):
        mode = wq_cases.Mode.__new__(wq_cases.Mode)
        events = []
        mode.ctx = types.SimpleNamespace(raw=types.SimpleNamespace(emit=lambda *a, **k: events.append((a, k))),
                                         commits={"T2": "f" * 40})
        mode.label, mode.evidence, mode.install, mode.markers, mode.prepared_ids = "U", self.tmp, install, [], []
        mode.custody = types.SimpleNamespace(members=lambda: [], label="U", raw=mode.ctx.raw)
        return mode

    def fake(self, response, rows=(), record=None):
        return types.SimpleNamespace(
            server=lambda custody: {"pid": 1, "birth": "1", "port": 1},
            http=lambda *args, **kwargs: response, handoff=lambda: None, supervisor_rows=lambda: list(rows),
            candidate=lambda candidate_id: record,
            wait_ready=lambda *args, **kwargs: {"pid": 2, "birth": "2", "port": 1, "health": {}})

    def test_rejected_activation_fails_the_case(self):
        case = Case("A")
        mode = self.mode(self.fake((409, {"error": "the update candidate is failed, not ready",
                                          "reason": "candidate_not_ready", "outcome": "not_applied"})))
        with self.assertRaises(CaseStop):
            mode.activate(case, "c1", "adopted")
        self.assertEqual(case.status, FAIL)

    def test_historical_outcome_of_another_attempt_never_settles_this_one(self):
        accepted = (200, {"status": "ok", "restarting": True, "candidate": {
            "candidate_id": "c1", "activation": {"adoption_id": "0123456789ab"}}})
        history = [{"type": "body_adoption_closed", "adoption_id": "ffffffffffff", "outcome": "adopted"}]
        record = {"state": "activated", "activation": {"outcome": "adopted"}}
        case = Case("A")
        mode = self.mode(self.fake(accepted, history, record))
        with mock.patch.dict(wq_cases.DEADLINE, {"settle": 1}), self.assertRaises(CaseStop):
            mode.activate(case, "c1", "adopted")
        self.assertEqual(case.status, UNKNOWN)
        self.assertNotEqual(case.status, PASS)

    def test_matching_attempt_settles(self):
        accepted = (200, {"status": "ok", "restarting": True, "candidate": {
            "candidate_id": "c1", "activation": {"adoption_id": "0123456789ab"}}})
        rows = [{"type": "body_adoption_closed", "adoption_id": "0123456789ab", "outcome": "adopted",
                 "candidate": "c" * 40}]
        record = {"state": "activated", "activation": {"outcome": "adopted"}, "layers": {"new_commit": "c" * 40}}
        case = Case("A")
        mode = self.mode(self.fake(accepted, rows, record))
        mode.head = lambda: "c" * 40
        result = mode.activate(case, "c1", "adopted")
        self.assertEqual((result["adoption"], case.status), ("0123456789ab", PASS))

    def test_an_attempt_on_another_composition_fails(self):
        accepted = (200, {"status": "ok", "restarting": True, "candidate": {
            "candidate_id": "c1", "activation": {"adoption_id": "0123456789ab"}}})
        record = {"state": "activated", "activation": {"outcome": "adopted"}, "layers": {"new_commit": "c" * 40}}
        for closed_candidate, head in (("d" * 40, "c" * 40), ("c" * 40, "d" * 40), ("", "")):
            rows = [{"type": "body_adoption_closed", "adoption_id": "0123456789ab", "outcome": "adopted",
                     "candidate": closed_candidate}]
            case = Case("B")
            mode = self.mode(self.fake(accepted, rows, record))
            mode.head = lambda head=head: head
            mode.activate(case, "c1", "adopted")
            self.assertEqual(case.status, FAIL, (closed_candidate, head))

    def preparing(self, layers_reading):
        """``Mode.prepare`` over a fake serving process; ``layers_reading`` is what every Git read returns."""
        fixed = self.tmp / "python-userbase"
        site = wq_product.user_site(fixed)
        (site / "ouroboros_qual_probe-1.0.dist-info").mkdir(parents=True, exist_ok=True)
        generation = self.tmp / "python-userbase.update-c9"
        (wq_product.user_site(generation) / "ouroboros_qual_probe-2.0.dist-info").mkdir(parents=True, exist_ok=True)
        view = {"state": "ready", "candidate_id": "c9", "target_sha": "f" * 40, "channel": "development",
                "target_ref": "managed/ouroboros", "checks": {"stub": fx.STUB_LABEL}}
        record = {"environment": {"same": False, "generation": {"kind": "user_site", "current": str(fixed),
                                                                "user_base": str(generation)}}}
        answers = {("POST", "/api/update/apply"): (202, {"status": "preparing"}),
                   ("GET", "/api/update/status"): (200, {"candidate": view})}
        install = types.SimpleNamespace(
            http=lambda method, path, *args, **kwargs: answers[(method, path)],
            server=lambda custody: {"pid": 1, "birth": "1", "port": 1}, candidate=lambda cid: record,
            markers=lambda names: {}, repo=self.tmp / "repo")
        mode = self.mode(install)
        mode.mirror_to, mode.env, mode.expected_layers = (lambda name: None), (lambda **extra: {}), None
        case = Case("A")
        with mock.patch.object(wq_cases, "layers", lambda *args, **kwargs: dict(layers_reading)):
            mode.prepare(case, "T2", "user_site", fixed, site, {"pid": 1, "birth": "1"})
        return case, next(row for row in case.checks if row["check"] == "serving_untouched_by_preparation")

    def test_preparation_over_unreadable_layers_is_unmeasured_not_untouched(self):
        case, row = self.preparing({"error": "RuntimeError('git rev-parse HEAD: rc=128')"})
        self.assertIsNone(row["ok"], row)
        self.assertEqual(case.status, UNKNOWN)
        measured = {"head": "a" * 40, "branch": "refs/heads/ouroboros", "index": "b" * 64, "status": "c" * 64,
                    "files": {}}
        case, row = self.preparing(measured)
        self.assertIs(row["ok"], True, "the control: complete equal readings do pass")

    def test_unobserved_liveness_stays_unmeasured_at_its_callers(self):
        mode = self.mode(types.SimpleNamespace(endpoint_gone=lambda server: True, repo=self.tmp))
        mode.custody = types.SimpleNamespace(alive=lambda identity: None, members=lambda: [], label="U",
                                             stop_all=lambda reason: {"confirmed": True}, raw=mode.ctx.raw)
        mode.head = lambda: "a" * 40
        case = Case("E.U")
        with self.assertRaises(CaseStop):
            mode.cold(case, lambda: None, {"pid": 5, "birth": "5", "port": 1}, self.tmp / "fixed", "3.0",
                      self.tmp / "site", None)
        self.assertEqual((case.checks[-1]["check"], case.checks[-1]["ok"], case.status),
                         ("previous_endpoint_gone", None, UNKNOWN), "an unobserved process is not a stopped one")
        holder = {"identity": {"pid": 9, "birth": "9"}, "proc": types.SimpleNamespace(wait=lambda timeout: 0),
                  "io": types.SimpleNamespace(request=lambda op, timeout=60: {"event": "exit"})}
        self.assertEqual(mode.release_holder(holder), ({"event": "exit"}, None))
        mode.custody.alive = lambda identity: False
        self.assertIs(mode.release_holder(holder)[1], True)

    def test_every_mode_reaches_all_its_cases_and_only_the_terminal_export_decides_pass(self):
        """Regression found in r2: the export check was required from module start, so each case read UNKNOWN
        right after its step, ``_sequence`` blocked the next one, and every mode stopped after its first case.
        Round 4: the teardown must also report a complete drain of the holders' output before the export counts."""
        def run(export_ok, drain=True):
            seen, statuses = [], {}
            for label, names in MODES.items():
                cases = {name: Case(name, required=wq_cases.REQUIRED_CHECKS[name]) for name in names}

                def step_for(name):
                    def step(case):  # a flow that records every real claim of its case, all measured true
                        seen.append(name)
                        for check in wq_cases.REQUIRED_CHECKS[name]:
                            case.check(check, True)
                    return step
                wq_cases._sequence(None, cases, names, [(name, step_for(name)) for name in names])
                outcome = {"ok": export_ok, "copied": 7 if export_ok else 0, "errors": [] if export_ok else
                           ["disk full"], "missing_required": []}
                drained = {"complete": drain, "pumps": 2, "pending": [] if drain else ["D-holder.stdout"], "errors": []}
                mode = types.SimpleNamespace(custody=types.SimpleNamespace(
                    close=lambda drained=drained: {"status": "confirmed", "drain": drained}),
                    export=lambda outcome=outcome: outcome)
                wq_cases._close_and_export(mode, cases, {})
                for name, case in cases.items():
                    self.assertIn("evidence_exported", case.required, "bound at the mode's end")
                    statuses[name] = case.status
            return seen, statuses

        seen, statuses = run(True)
        self.assertEqual(seen, list(REQUIRED), "every selected case of every mode is reached")
        self.assertEqual(set(statuses.values()), {PASS})
        self.assertEqual(exit_code(statuses, REQUIRED, False, True), EXIT["qualified"])
        seen, statuses = run(False)
        self.assertEqual(seen, list(REQUIRED))
        self.assertEqual(set(statuses.values()), {UNKNOWN}, "a failed export leaves every case unmeasured")
        self.assertEqual(exit_code(statuses, REQUIRED, False, True), EXIT["unknown_or_not_run"])
        seen, statuses = run(True, drain=False)
        self.assertEqual(seen, list(REQUIRED))
        self.assertEqual(set(statuses.values()), {UNKNOWN}, "a copy made before the holders' output drained")
        self.assertEqual(exit_code(statuses, REQUIRED, False, True), EXIT["unknown_or_not_run"])
        self.assertFalse([name for checks in wq_cases.REQUIRED_CHECKS.values() for name in checks
                          if name in wq_cases.TERMINAL_CHECKS], "step readiness never waits for the export")

    def test_teardown_and_export_run_on_every_path_and_a_failed_export_is_not_pass(self):
        def broken_close():
            raise RuntimeError("close broke")
        cases = {"A": Case("A", required=("x", "evidence_exported")), "B": Case("B")}
        cases["A"].check("x", True)
        cases["B"].not_run("blocked")
        exported = {"ok": False, "copied": 0, "errors": ["copy refused"], "missing_required": ["data-logs"]}
        mode = types.SimpleNamespace(custody=types.SimpleNamespace(close=broken_close), export=lambda: exported)
        teardown = wq_cases._close_and_export(mode, cases, {"status": "confirmed"})
        self.assertEqual((teardown["status"], teardown["export"]["ok"]), ("unknown", False))
        self.assertEqual((cases["A"].status, cases["B"].status), (UNKNOWN, NOT_RUN))
        good = {"A": Case("A", required=("x", "evidence_exported"))}
        good["A"].check("x", True)
        mode = types.SimpleNamespace(custody=types.SimpleNamespace(close=lambda: {"status": "confirmed", "drain": {
            "complete": True, "pumps": 0, "pending": [], "errors": []}}),
                                     export=lambda: {"ok": True, "copied": 3, "errors": [], "missing_required": []})
        self.assertEqual(wq_cases._close_and_export(mode, good, {})["status"], "confirmed")
        self.assertEqual(good["A"].status, PASS)

    def test_an_equipment_error_mid_case_never_leaves_it_passing(self):
        cases = {name: Case(name) for name in ("entry.U", "A")}

        def crashing(case):
            case.check("measured_before_the_crash", True)
            raise FileNotFoundError("a record the case needed")
        wq_cases._sequence(None, cases, ("entry.U", "A"), (("entry.U", crashing), ("A", lambda case: None)))
        self.assertEqual((cases["entry.U"].status, cases["A"].status), (UNKNOWN, NOT_RUN))

    def test_every_required_check_is_emitted_by_the_case_code(self):
        source = (HERE / "wq_cases.py").read_text(encoding="utf-8")
        templates = {r"marker_(winqual-[a-z]+-[a-z]+)_written": '"{}"', r"settled_(adopted|returned_retry|returned)": '"{}"',
                     r"(venv_python|console_image)_holder_(?:acknowledged|survived|released)": '("{}"'}
        import re
        for case, names in wq_cases.REQUIRED_CHECKS.items():
            self.assertIn(case, REQUIRED)
            for name in names:
                if f'"{name}"' in source:
                    continue
                matched = [(pattern, form) for pattern, form in templates.items() if re.fullmatch(pattern, name)]
                self.assertTrue(matched, f"{case}: {name} is never emitted")
                pattern, form = matched[0]
                self.assertIn(form.format(re.fullmatch(pattern, name).group(1)), source, f"{case}: {name}")
        self.assertEqual(sorted(wq_cases.REQUIRED_CHECKS), sorted(REQUIRED))

    def test_sequence_blocks_after_a_non_pass_and_respects_partial_selection(self):
        cases = {name: Case(name) for name in ("entry.U", "A", "B")}
        steps = (("entry.U", lambda case: case.check("ok", True)), ("A", lambda case: case.require("x", False)),
                 ("B", lambda case: case.check("never", True)))
        wq_cases._sequence(None, cases, ("entry.U", "A", "B"), steps)
        self.assertEqual([cases[name].status for name in ("entry.U", "A", "B")], [PASS, FAIL, NOT_RUN])
        partial = {name: Case(name) for name in ("entry.U", "A")}
        wq_cases._sequence(None, partial, ("A",), (("entry.U", lambda case: None), ("A", lambda case: None)))
        self.assertEqual(partial["A"].status, NOT_RUN, "a case whose predecessor did not run cannot run")

    def test_installation_environment_carries_no_host_secret_or_python_override(self):
        host = {"SystemRoot": "C:\\Windows", "ANTHROPIC_API_KEY": "secret", "GITHUB_TOKEN": "t", "PYTHONPATH": "x",
                "PIP_INDEX_URL": "https://index", "OUROBOROS_PYTEST_ACTIVE": "1", "VIRTUAL_ENV": "v",
                "OUROBOROS_UPDATE_CHANNEL": "stable", "ACTIONS_RUNTIME_TOKEN": "a", "COMPUTERNAME": "runner"}
        ctx = types.SimpleNamespace(run_root=self.tmp, evidence=self.tmp / "evidence", git_dir="C:\\Git\\cmd",
                                    wheelhouse=self.tmp / "wheelhouse")
        with mock.patch.dict(os.environ, host, clear=False):
            env = wq_cases.Mode(ctx, "U").env(PYTHONUSERBASE="C:\\u")
        for key in ("ANTHROPIC_API_KEY", "GITHUB_TOKEN", "PYTHONPATH", "PIP_INDEX_URL", "OUROBOROS_PYTEST_ACTIVE",
                    "VIRTUAL_ENV", "OUROBOROS_UPDATE_CHANNEL", "ACTIONS_RUNTIME_TOKEN"):
            self.assertNotIn(key, env)
        self.assertEqual((env["PIP_NO_INDEX"], env["WINQUAL_CHECKS_STUB"], env["PYTHONUSERBASE"]), ("1", "1", "C:\\u"))
        self.assertIn(" ", env["OUROBOROS_APP_ROOT"], "the installation root holds a space")
        self.assertTrue(env["PIP_FIND_LINKS"].startswith("file:"))
        self.assertEqual(env["COMPUTERNAME"], "runner")


class FakeWin:
    """Injected Win32 answers for the custody owner; rows map pid -> identity dict or an exception."""

    def __init__(self, rows=(), job_pids=(), active=0, **errors):
        self.rows, self.job_pids, self.active_value, self.errors = dict(rows), list(job_pids), active, errors
        self.calls = {"terminate": 0, "close": 0}

    def _raise(self, name):
        if self.errors.get(name):
            raise self.errors[name]

    def job(self):
        return "job-handle"

    def identity(self, pid, job=None, parents=None):
        value = self.rows[pid]
        if isinstance(value, Exception):
            raise value
        return {"pid": pid, **value}

    def snapshot(self):
        self._raise("snapshot")
        return {pid: (0, "x.exe") for pid in self.rows}

    def pids(self, job):
        self._raise("pids")
        return list(self.job_pids)

    def terminate(self, job, code):
        self.calls["terminate"] += 1
        self._raise("terminate")

    def active(self, job):
        self._raise("active")
        return self.active_value

    def close(self, handle):
        self.calls["close"] += 1
        self._raise("close")


class CustodyTests(Scratch):
    def custody(self, win):
        return Custody(win, types.SimpleNamespace(emit=lambda *a, **k: None), "U", ["C:\\wq\\U"])

    def test_alive_is_false_only_when_measured(self):
        cases = ((None, {"alive": None, "open_error": 5}), (False, {"alive": False, "absent": True, "open_error": 87}),
                 (False, {"alive": True, "birth": "2", "in_job": True}), (True, {"alive": True, "birth": "1",
                                                                              "in_job": True}),
                 (None, {"alive": True, "birth": "1", "in_job": False}), (None, OSError("query failed")))
        for expected, row in cases:
            self.assertIs(self.custody(FakeWin({7: row})).alive({"pid": 7, "birth": "1"}), expected, row)

    def test_an_incomplete_census_is_none_and_stop_still_counts_the_job(self):
        win = FakeWin({7: {"alive": True}}, pids=RuntimeError("job process list incomplete (3/4)"),
                      terminate=OSError("terminate refused"))
        custody = self.custody(win)
        self.assertIsNone(custody.members())
        stopped = custody.stop_all("unit", deadline=1)
        self.assertEqual((stopped["before"], stopped["confirmed"], win.calls["terminate"]), (None, True, 1),
                         "the Job's active count is the primary custody evidence")
        self.assertIn("terminate refused", stopped["terminate"])
        self.assertFalse(self.custody(FakeWin(active=2)).stop_all("unit", deadline=0.3)["confirmed"])

    def test_escape_scan_discloses_unobserved_processes_without_judging_them(self):
        rows = {10: {"alive": None, "open_error": 5}, 11: OSError("read failed"),
                12: {"alive": True, "in_job": False, "image": "C:\\Windows\\System32\\svchost.exe"}}
        escaped, scan = self.custody(FakeWin(rows)).escaped()
        self.assertEqual((escaped, scan["unobserved"]), ([], 2))
        rows[13] = {"alive": True, "in_job": False, "image": "C:\\wq\\U\\app root\\x\\python.exe"}
        escaped, _scan = self.custody(FakeWin(rows)).escaped()
        self.assertEqual([row["pid"] for row in escaped], [13])
        self.assertEqual(self.custody(FakeWin(rows, snapshot=OSError("walk failed"))).escaped()[0], None)

    def test_close_always_closes_the_job_and_never_claims_what_it_could_not_measure(self):
        confirmed = self.custody(FakeWin({10: {"alive": None, "open_error": 5}}))
        self.assertEqual(confirmed.close()["status"], "confirmed", "a protected process is a disclosed gap")
        for name, error in (("snapshot", OSError("walk failed")), ("close", OSError("close failed"))):
            win = FakeWin({}, **{name: error})
            teardown = self.custody(win).close()
            self.assertEqual((teardown["status"], win.calls["close"]), ("unknown", 1), name)
        win = FakeWin({})
        custody = self.custody(win)

        def broken_stop(reason):
            raise RuntimeError("census exploded")
        custody.stop_all = broken_stop
        self.assertEqual((custody.close()["status"], win.calls["close"]), ("unknown", 1))

    def test_win_identity_tells_absent_from_denied_and_unreadable(self):
        win = wq_win.Win.__new__(wq_win.Win)
        win.OpenProcess, win.CloseHandle = (lambda *args: 0), (lambda handle: 1)
        for error, alive in ((wq_win.ERROR_INVALID_PARAMETER, False), (wq_win.ERROR_ACCESS_DENIED, None), (6, None)):
            with mock.patch.object(wq_win.c, "get_last_error", create=True, return_value=error):
                self.assertIs(win.identity(7)["alive"], alive, error)
        win.OpenProcess, win.GetProcessTimes = (lambda *args: 99), (lambda *args: 0)
        with mock.patch.object(wq_win.c, "get_last_error", create=True, return_value=5), \
                mock.patch.object(wq_win.c, "WinError", create=True, new=lambda code=None: OSError(code, "injected")):
            row = win.identity(7)
        self.assertIsNone(row["alive"], "a handle whose reads fail observed nothing")

    def test_win_snapshot_tells_the_end_of_the_table_from_a_failed_walk(self):
        win = wq_win.Win.__new__(wq_win.Win)
        win.CreateToolhelp32Snapshot, win.CloseHandle = (lambda *args: 42), (lambda handle: 1)
        for error, fails in ((wq_win.ERROR_NO_MORE_FILES, False), (31, True)):
            steps = iter([True, True, False])
            win.Process32FirstW = win.Process32NextW = lambda snap, entry: next(steps)
            with mock.patch.object(wq_win.c, "get_last_error", create=True, return_value=error), \
                    mock.patch.object(wq_win.c, "WinError", create=True, new=lambda code=None: OSError(code, "walk")):
                if fails:
                    with self.assertRaises(OSError):
                        win.snapshot()
                else:
                    self.assertEqual(len(win.snapshot()), 1)


class EvidenceTests(Scratch):
    def test_global_package_content_changes_are_seen_even_with_the_same_dist_info(self):
        site = self.tmp / "site-packages"
        (site / "pkg").mkdir(parents=True)
        (site / "pkg" / "__init__.py").write_text("A = 1\n", encoding="utf-8")
        (site / "pkg-1.0.dist-info").mkdir()
        (site / "pkg-1.0.dist-info" / "RECORD").write_text("pkg/__init__.py,,\n", encoding="utf-8")
        before, identities = wq_product.content_inventory(site), wq_product.inventory(site)
        (site / "pkg" / "__pycache__").mkdir()
        (site / "pkg" / "__pycache__" / "__init__.cpython-310.pyc").write_bytes(b"cache")
        (site / "pkg" / "loose.pyc").write_bytes(b"cache")
        self.assertEqual(wq_product.content_changes(before, wq_product.content_inventory(site)), [],
                         "the interpreter's bytecode cache is not a package change")
        (site / "pkg" / "__init__.py").write_text("A = 2\n", encoding="utf-8")  # in place, same size
        self.assertEqual(wq_product.inventory(site), identities, "dist-info identities cannot see this")
        self.assertEqual(wq_product.content_changes(before, wq_product.content_inventory(site)), ["pkg/__init__.py"])
        (site / "pkg-1.0.dist-info" / "RECORD").write_text("changed metadata\n", encoding="utf-8")
        self.assertIn("pkg-1.0.dist-info/RECORD", wq_product.content_changes(before, wq_product.content_inventory(site)))
        self.assertIsNone(wq_product.content_inventory(site, {"files": 1, "bytes": 10 ** 9}), "over the bound")
        self.assertIsNone(wq_product.content_inventory(self.tmp / "absent"))
        self.assertIsNone(wq_product.content_changes(None, before))

    def test_export_reports_missing_required_evidence_and_copy_failures(self):
        events = []
        ctx = types.SimpleNamespace(run_root=self.tmp / "run", evidence=self.tmp / "evidence",
                                    raw=types.SimpleNamespace(emit=lambda *a, **k: events.append(k)))
        mode = wq_cases.Mode(ctx, "U")
        mode.evidence.mkdir(parents=True)
        outcome = mode.export()
        self.assertFalse(outcome["ok"])
        self.assertEqual(sorted(outcome["missing_required"]), ["data-logs", "harness-logs", "witness"])
        for folder in (mode.install.data / "logs", mode.install.witness_dir, mode.install.logs):
            folder.mkdir(parents=True)
            (folder / "one.txt").write_text("x", encoding="utf-8")
        self.assertTrue(mode.export()["ok"])
        with mock.patch.object(wq_cases.shutil, "copy2", side_effect=OSError("disk full")):
            refused = mode.export()
        self.assertEqual((refused["ok"], len(refused["errors"])), (False, 3))


def fast_clock(step=0.0):
    """A stand-in for ``wq_cases.time`` only (never the process's own clock): sleeps return at once and, with
    ``step``, each ``monotonic`` read advances by that many seconds."""
    now = [time.monotonic()]

    def monotonic():
        now[0] += step
        return now[0] if step else time.monotonic()
    return types.SimpleNamespace(monotonic=monotonic, sleep=lambda seconds: None, time_ns=time.time_ns)


class MeasurementTests(Scratch):
    """Round 4 (Astra review rl-20261029T112322-f1b343ef83bb): a missing census, reading, answer or deadline is
    UNKNOWN, a measured refutation stays FAIL and the measured positive path still PASSes. The conditions are
    the ones of Astra's retained counterexamples (``test_review_counterexamples.py``, ``check_log_export_race.py``)."""

    OLD = {"pid": 10, "birth": "100", "alive": True, "image": "C:\\p\\python.exe",
           "cmdline": "python C:\\wq\\U\\app root\\repo\\launcher.py --seed-bundle x"}
    NEW = {**OLD, "pid": 11, "birth": "101"}
    UNREAD = {"pid": 12, "alive": None, "open_error": 5}
    NO_CMDLINE = {"pid": 13, "birth": "103", "alive": True, "image": "C:\\p\\python.exe", "cmdline": None}
    DEAD = {"pid": 14, "birth": "104", "alive": False, "image": "C:\\p\\git.exe", "exit_code": 0, "cmdline": None}
    ACCEPTED = (200, {"status": "ok", "restarting": True, "candidate": {
        "candidate_id": "c1", "activation": {"adoption_id": "0123456789ab"}}})
    CLOSED = [{"type": "body_adoption_closed", "adoption_id": "0123456789ab", "outcome": "adopted", "candidate": "c" * 40}]
    ACTIVATED = {"state": "activated", "activation": {"outcome": "adopted"}, "layers": {"new_commit": "c" * 40}}
    OLD_SERVER, NEW_SERVER = {"pid": 20, "birth": "200", "port": 1}, {"pid": 21, "birth": "201", "port": 1}

    def flow_mode(self, censuses, install):
        """A real ``Mode`` over a fake installation; each ``members()`` call answers the next census."""
        mode = wq_cases.Mode.__new__(wq_cases.Mode)
        events = types.SimpleNamespace(emit=lambda *a, **k: None)
        mode.ctx = types.SimpleNamespace(raw=events, commits={"S0": "a" * 40, "T2": "f" * 40})
        mode.label, mode.evidence, mode.markers, mode.prepared_ids, mode.expected_layers = "U", self.tmp, [], [], None
        census = iter(censuses)
        mode.custody = types.SimpleNamespace(members=lambda: next(census), label="U", raw=events)
        mode.install, mode.head, mode.snapshot = install, (lambda: "c" * 40), (lambda *a, **k: None)
        return mode

    def activation_install(self, response=ACCEPTED, record=ACTIVATED, ready=NEW_SERVER, old=OLD_SERVER):
        return types.SimpleNamespace(server=lambda custody: old, http=lambda *a, **k: response, handoff=lambda: None,
                                     supervisor_rows=lambda: list(self.CLOSED), candidate=lambda cid: record,
                                     wait_ready=lambda *a, **k: ready)

    # ------------------------------------------------------------------ 1: the census
    def test_an_incomplete_census_is_never_an_empty_one(self):
        self.assertEqual(launcher_ids([self.OLD, self.DEAD]), {(10, "100")}, "a dead member's command line is no gap")
        self.assertEqual(launcher_ids([]), set(), "a complete census without a launcher is measured empty")
        for census in (None, [self.OLD, self.UNREAD], [self.OLD, self.NO_CMDLINE]):
            self.assertIsNone(launcher_ids(census), census)
            self.assertIsNone(known_roles(census), census)
        self.assertIsNone(classify(None), "no census is not an empty role table")
        self.assertIsNone(identities(None))
        self.assertIsNone(identities([self.OLD, self.UNREAD]), "an unobserved member could be any process")
        self.assertEqual(identities([self.OLD, self.NO_CMDLINE, self.DEAD]), {(10, "100"), (13, "103")})
        self.assertEqual(unread([self.OLD, self.UNREAD, self.NO_CMDLINE, self.DEAD]), [self.UNREAD, self.NO_CMDLINE])

    def test_fresh_launcher_and_server_are_judged_only_between_measured_identities(self):
        def run(before, after, old=self.OLD_SERVER, new=self.NEW_SERVER):
            mode = self.flow_mode([before, after], self.activation_install(old=old, ready=new))
            case = Case("A")
            mode.fresh_generation(case, mode.activate(case, "c1", "adopted"))
            rows = {row["check"]: row["ok"] for row in case.checks}
            return rows["fresh_launcher"], rows["fresh_server"]
        self.assertEqual(run(None, [self.OLD]), (None, True), "Astra: a failed census before, the same launcher after")
        for before, after, expected in (([self.OLD], [self.NEW], True), ([self.OLD], [self.OLD], False),
                                        ([self.OLD], [self.OLD, self.NEW], False), ([self.OLD], [], False),
                                        ([self.OLD], None, None), ([self.NO_CMDLINE], [self.OLD], None),
                                        ([self.OLD, self.UNREAD], [self.NEW], None), ([self.OLD], [self.NEW, self.UNREAD], None)):
            self.assertIs(run(before, after)[0], expected, (before, after))
        self.assertIsNone(run([self.OLD], [self.NEW], old=None)[1], "an unread old server makes no server fresh")
        self.assertIs(run([self.OLD], [self.NEW], new=dict(self.OLD_SERVER))[1], False)
        self.assertIs(run([self.OLD], [self.NEW], new={**self.OLD_SERVER, "birth": "999"})[1], True, "new birth, same pid")

    def run_steps(self, function, mode, names, patches=()):
        """The real step closures ``names`` of ``function`` (Astra's harness for B, generalised): each runs on
        its own case as ``_sequence`` runs it, but none is blocked; only what the steps call is faked."""
        cases = {name: Case(name) for name in names}

        def sequence(_mode, _cases, _selected, steps):
            steps = dict(steps)
            for name in names:
                try:
                    steps[name](cases[name])
                except CaseStop:
                    pass
                except Exception as exc:  # as _sequence records it; the tests below assert it did not happen
                    cases[name].check("case_completed", None, error=repr(exc))
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(wq_cases, "_sequence", sequence))
            stack.enter_context(mock.patch.object(wq_cases, "time", fast_clock()))
            for target, name, value in patches:
                stack.enter_context(mock.patch.object(target, name, value))
            function(mode, {}, ())
        return {name: {row["check"]: row["ok"] for row in case.checks} for name, case in cases.items()}

    def case_b(self, censuses, final_server, serving_alive=True):
        """``launcher_not_replaced_without_switch`` and ``no_crash_loop`` of the real B step."""
        fixed = pathlib.Path("/unused/private/python-userbase")
        site = wq_cases.user_site(fixed)
        serving = {"pid": 50, "birth": "500", "port": 1}
        layer = {"head": "a" * 40, "branch": "ouroboros", "index": "b" * 64, "status": "c" * 64, "files": {}}
        refusal = {"adoption": "0123456789ab", "since": 1, "server": serving,
                   "record": {"activation": {"decision": {"error": {"step": "serving_aside", "winerror": 5}}}}}
        adopted = {"adoption": "0123456789ac", "since": 2, "server": {"pid": 51, "birth": "510", "port": 1}}
        replies, census, silent = iter([refusal, adopted]), iter(censuses), (lambda *a, **k: None)
        holder = {"ready": {"event": "ready", "cwd": str(site / fx.PACKAGE)}, "identity": {"pid": 60, "birth": "600"}}
        mode = types.SimpleNamespace(
            ctx=types.SimpleNamespace(release={"internal": pathlib.Path("/unused/app/_internal"),
                                               "python": pathlib.Path("/unused/python.exe")}),
            install=types.SimpleNamespace(user_base=fixed, server=lambda *a: final_server), entry=lambda *a: serving,
            prepare=lambda *a: {"path": fixed.with_name("python-userbase.update-c1"), "id": [1, 1], "candidate_id": "c1"},
            marker=silent, hold=lambda *a, **k: holder, layers=lambda: layer, activate=lambda *a: next(replies),
            custody=types.SimpleNamespace(members=lambda: next(census),
                                          alive=lambda identity: serving_alive if identity is serving else True),
            server_witness=lambda *a: {}, markers_kept=silent, release_holder=lambda *a: ({}, True), selected=silent,
            fresh_generation=silent, layers_carried=silent)
        rows = self.run_steps(wq_cases._user_site, mode, ("entry.U", "B"), (
            (wq_cases, "path_id", lambda path: [1, 1]), (wq_cases, "inventory", lambda site: {}),
            (wq_cases, "generation_seen", lambda *a: True)))["B"]
        self.assertNotIn("case_completed", rows)
        return rows["launcher_not_replaced_without_switch"], rows["no_crash_loop"]

    def test_b_keeps_its_launcher_and_its_server_only_by_measured_identity(self):
        serving, kept = {"pid": 50, "birth": "500", "port": 1}, [self.OLD]
        self.assertEqual(self.case_b([kept, kept], dict(serving)), (True, True), "the measured positive path")
        self.assertIs(self.case_b([kept, kept], {**serving, "birth": "501"})[1], False, "Astra: pid 50 reborn as 501")
        self.assertIs(self.case_b([kept, kept], {**serving, "pid": 52})[1], False)
        self.assertIs(self.case_b([kept, kept], None, serving_alive=False)[1], False, "the serving process measured gone")
        for alive in (None, True):  # no bound server read, and the process unobserved or still running
            self.assertIsNone(self.case_b([kept, kept], None, serving_alive=alive)[1], alive)
        self.assertIs(self.case_b([kept, [self.NEW]], serving)[0], False, "a replaced launcher")
        self.assertIs(self.case_b([[], []], serving)[0], False, "a measured census without any launcher keeps none")
        for censuses in ((None, kept), (kept, None), (None, None), ([self.OLD, self.UNREAD], kept),
                         (kept, [self.NO_CMDLINE, self.OLD])):
            self.assertIsNone(self.case_b(list(censuses), serving)[0], censuses)

    def entry_rows(self, census=..., rc=0, status=(200, {"running_sha": "a" * 40}), head="a" * 40,
                   ready={"pid": 30, "birth": "300", "port": 1, "health": {"runtime_version": "9.9"}}, boot_failed=False):
        """The real ``Mode.entry`` over fakes; ``census`` is what ``members()`` answers (default: the successor)."""
        internal, repo = pathlib.Path("/wq/released app/_internal"), pathlib.Path("/wq/U/app root/repo")
        successor = {"pid": 31, "birth": "301", "alive": True, "image": "/wq/python.exe",
                     "cmdline": f'"python" "{repo}/launcher.py" --no-ui --seed-bundle "{internal}"'}

        def wait(timeout):
            if rc is None:
                raise subprocess.TimeoutExpired("Ouroboros.exe", timeout)
            return rc
        install = types.SimpleNamespace(wait_ready=lambda *a, **k: ready, http=lambda *a, **k: status, repo=repo,
                                        witnesses=lambda: [], boot_failed=boot_failed)
        mode = self.flow_mode([[successor] if census is ... else census], install)
        mode.ctx.release, mode.ctx.candidate_version = {"internal": internal}, "9.9"
        mode.start_released, mode.env = (lambda *a: (types.SimpleNamespace(wait=wait), {}, None)), (lambda **extra: {})
        case = Case("entry.U")
        with mock.patch.object(wq_cases, "git", lambda *a, **k: head):
            try:
                mode.entry(case, (), pathlib.Path("/wq/site"), pathlib.Path("/wq/prefix"))
            except CaseStop:
                pass
        return {row["check"]: row["ok"] for row in case.checks}

    def test_entry_claims_keep_measured_from_unmeasured(self):
        rows = self.entry_rows()
        self.assertEqual([rows[name] for name in ("server_ready", "frozen_root_exited_after_handover",
                                                  "source_launcher_successor", "serving_tree_is_S0", "served_version")],
                         [True] * 5, "the measured positive path")
        self.assertEqual(self.entry_rows(ready=None), {"server_ready": None}, "Astra: a readiness deadline")
        self.assertEqual(self.entry_rows(ready=None, boot_failed=True), {"server_ready": False}, "a failed boot")
        self.assertIsNone(self.entry_rows(rc=None)["frozen_root_exited_after_handover"], "still running at 60 s")
        self.assertIs(self.entry_rows(rc=1)["frozen_root_exited_after_handover"], False)
        self.assertIs(self.entry_rows(census=[])["source_launcher_successor"], False, "measured: no successor")
        for census in (None, [self.UNREAD], [self.NO_CMDLINE]):
            self.assertIsNone(self.entry_rows(census=census)["source_launcher_successor"], census)
        self.assertIsNone(self.entry_rows(status=(None, "TimeoutError"))["serving_tree_is_S0"], "no status answer")
        self.assertIs(self.entry_rows(status=(None, "TimeoutError"), head="b" * 40)["serving_tree_is_S0"], False,
                      "a measured HEAD refutes it whatever the transport did")
        self.assertIs(self.entry_rows(status=(200, {"running_sha": "b" * 40}))["serving_tree_is_S0"], False)

    def cold_rows(self, seen=..., now=..., stop=True, rc=0, roles=..., boot_failed=False, ready=...):
        """The real ``Mode.cold`` (released form) over fakes: ``seen``/``roles``/``now`` are the three censuses."""
        repo = pathlib.Path("/wq/U/app root/repo")
        launcher = {**self.NEW, "cmdline": f'"python" "{repo}/launcher.py" --no-ui'}
        censuses = [[self.OLD] if seen is ... else seen, [launcher] if roles is ... else roles,
                    [launcher] if now is ... else now]
        cold = {"pid": 32, "birth": "302", "port": 1} if ready is ... else ready

        def wait(timeout):
            if rc is None:
                raise subprocess.TimeoutExpired("Ouroboros.exe", timeout)
            return rc
        install = types.SimpleNamespace(wait_ready=lambda *a, **k: cold, endpoint_gone=lambda server: True,
                                        witnesses=lambda: [], markers=lambda names: {}, repo=repo, boot_failed=boot_failed)
        mode = self.flow_mode(censuses, install)
        census, mode.env = mode.custody.members, (lambda **extra: {})
        mode.custody = types.SimpleNamespace(members=census, stop_all=lambda reason: {"confirmed": stop},
                                             alive=lambda identity: False, escaped=lambda: ([], {}))
        case = Case("E.U")
        with mock.patch.object(wq_cases, "path_id", lambda path: [1, 2]), \
                mock.patch.object(wq_cases, "inventory", lambda site: {}):
            try:
                mode.cold(case, lambda: (types.SimpleNamespace(wait=wait), {}, None), self.OLD_SERVER,
                          pathlib.Path("/wq/fixed"), "3.0", pathlib.Path("/wq/site"), None, released=True)
            except CaseStop:
                pass
        return {row["check"]: row["ok"] for row in case.checks}

    def test_cold_entry_claims_keep_measured_from_unmeasured(self):
        rows = self.cold_rows()
        self.assertEqual([rows[name] for name in ("previous_processes_stopped", "previous_endpoint_gone",
                                                  "no_process_outside_custody", "cold_ready",
                                                  "released_app_handed_over_again", "all_processes_new")],
                         [True] * 6, "the measured positive path")
        self.assertEqual(self.cold_rows(stop=False), {"previous_processes_stopped": None},
                         "the harness's own unconfirmed stop is unmeasured, never the product's failure")
        self.assertIsNone(self.cold_rows(ready=None)["cold_ready"])
        self.assertIs(self.cold_rows(ready=None, boot_failed=True)["cold_ready"], False)
        self.assertIsNone(self.cold_rows(rc=None)["released_app_handed_over_again"], "the frozen root ran on at 60 s")
        self.assertIsNone(self.cold_rows(roles=None)["released_app_handed_over_again"])
        self.assertIs(self.cold_rows(rc=1, roles=None)["released_app_handed_over_again"], False, "measured rc 1")
        self.assertIs(self.cold_rows(roles=[])["released_app_handed_over_again"], False)
        for seen, now in ((None, ...), ([self.OLD, self.UNREAD], ...), (..., None)):
            self.assertIsNone(self.cold_rows(seen=seen, now=now)["all_processes_new"], (seen, now))
        self.assertIs(self.cold_rows(seen=[self.NEW])["all_processes_new"], False, "a process that survived the stop")

    def closure_mode(self, root):
        """A stand-in for ``Mode`` whose every answer is the measured positive path of the step closures (the
        tests below change one answer at a time); paths are real directories under ``root``."""
        app_root, internal, bare = root / "app root", root / "released app" / "_internal", root / "bare py"
        venv, private = app_root / "runtime venv", app_root / "data" / "state" / "python-private"
        silent, server = (lambda *a, **k: None), {"pid": 30, "birth": "300", "port": 1, "image": str(bare / "python.exe")}
        layer = {"head": "a" * 40, "branch": "ouroboros", "index": "b" * 64, "status": "c" * 64, "files": {}}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(zipfile.ZipInfo("__main__.py", (1980, 1, 1, 0, 0, 0)), "print('x')\n")
        for scripts, generation in ((venv / "Scripts", venv), (root / "gen" / "Scripts", root / "gen")):
            scripts.mkdir(parents=True, exist_ok=True)
            (scripts / "ouroboros-qual-tool.exe").write_bytes(
                b"MZ stub" + f'#!"{generation / "Scripts" / "python.exe"}"\r\n'.encode() + archive.getvalue())
        holder = {"identity": {"pid": 60, "birth": "600"}, "io": types.SimpleNamespace(
            request=lambda op, timeout=60: {"id": 1, "package": {"version": "1.0"}})}
        site = wq_product.user_site(app_root / "data" / "state" / "python-userbase")
        holders = {
            "A-pyd-holder": {**holder, "ready": {"event": "ready", "package": {"version": "1.0"}, "native": {
                "status": "loaded", "realpath": str(site / fx.PACKAGE / "_bz2.pyd"), "mapped_device": "\\Device\\x"}}},
            "D-venv-python-holder": {**holder, "ready": {"event": "ready", "prefix": str(venv), "native": {"status": "loaded"}}},
            "D-console-image-holder": {**holder, "ready": {"event": "ready", "prefix": str(venv), "native": {"status": "loaded"}}}}
        images = [{"pid": 61, "birth": "601", "alive": True, "image": str(venv / "Scripts" / name), "cmdline": name}
                  for name in ("python.exe", "ouroboros-qual-tool.exe")]
        facts = {"prefix": str(private), "package": {"version": "5.0", "realpath": str(wq_cases.venv_site(private) / "p.py")}}
        bounded = {"C-fresh-observer": (0, json.dumps({"package": {"version": "3.0", "realpath": str(site / "p.py")}})),
                   "D-relocated-tool": (0, json.dumps({"prefix": str(venv), "executable": str(venv / "Scripts" / "python.exe"),
                                                       "package": {"version": "5.0", "realpath": str(wq_cases.venv_site(venv) / "p.py")}})),
                   "E.D-tool": (0, json.dumps({"prefix": str(venv), "package": {
                       "version": "5.0", "realpath": str(wq_cases.venv_site(venv) / "p.py")}})),
                   "F-owner-installs-base-lock": (0, ""), "module-cli-help": (0, ""), "module-entry-facts": (0, json.dumps(facts))}
        record = {"environment": {"generation": {}}, "activation": {"decision": {
            "detail": f"rc=1: ImportError: cannot import name 'stable_api' from '{fx.PACKAGE}'"}}}
        launchers = [{"pid": 70, "birth": "700", "alive": True, "image": str(venv / "Scripts" / "python.exe"),
                      "cmdline": f'"{venv / "Scripts" / "python.exe"}" "{app_root / "repo" / "launcher.py"}" --no-ui'},
                     {"pid": 71, "birth": "710", "alive": True, "image": str(internal / "python-standalone" / "python.exe"),
                      "cmdline": f'python "{app_root / "repo" / "launcher.py"}" --no-ui'}]
        return types.SimpleNamespace(
            root=root, ctx=types.SimpleNamespace(release={"internal": internal, "python": internal / "python.exe"},
                                                 bare_python=bare / "python.exe", bare_root=bare),
            install=types.SimpleNamespace(user_base=site.parent.parent, app_root=app_root, private=private,
                                          repo=app_root / "repo", witnesses=lambda: [], boot_failed=False,
                                          wait_ready=lambda *a, **k: server, endpoint_gone=lambda server: True),
            custody=types.SimpleNamespace(members=lambda: [*images, {**server, "alive": True, "ppid": 40,
                                                                     "cmdline": "python server.py"}],
                                          alive=lambda identity: True, stop_all=lambda reason: {"confirmed": True}),
            entry=lambda *a: server, channel_and_marker=silent, marker=silent, markers_kept=silent, seed_layers=silent,
            layers_carried=silent, selected=silent, fresh_generation=silent, snapshot=silent, layers=lambda: layer,
            hold=lambda label, argv, **env: holders[label], release_holder=lambda holder: ({}, True),
            prepare=lambda *a: {"path": root / "gen", "id": [1, 1], "candidate_id": "c1", "record": record,
                                "inventory": {fx.DIST: {"id": [7, 7]}}},
            activate=lambda *a: {"adoption": "0123456789ab", "since": 0, "server": server, "record": record,
                                 "old_members": [*launchers, {**server, "alive": True, "ppid": 40, "cmdline": "python server.py"}]},
            server_witness=lambda *a: None, start=lambda *a, **k: (None, {"pid": 40, "birth": "400"}, None),
            run_bounded=lambda label, argv, deadline, **kw: bounded[label], cold=lambda *a, **k: server)

    def closure_rows(self, function, names, **changes):
        mode = self.closure_mode(self.tmp / "U")
        for path, value in changes.items():  # "install.boot_failed" -> setattr(mode.install, "boot_failed", value)
            *parents, name = path.split(".")
            target = mode
            for parent in parents:
                target = getattr(target, parent)
            setattr(target, name, value)
        rows = self.run_steps(function, mode, names, (
            (wq_cases, "path_id", lambda path: [1, 1]),
            (wq_cases, "inventory", lambda site: {fx.DIST: {"id": [7, 7]}}),
            (wq_cases, "content_inventory", lambda root: {})))
        for name in names:
            self.assertNotIn("case_completed", rows[name], (name, rows[name]))
        return rows

    def test_a_deadline_or_silence_in_any_case_step_is_unmeasured(self):
        """One answer at a time turned missing in the real closures of U, D and F (C, A, entry.D, D, E.D, F, E.F)."""
        u, d, f = wq_cases._user_site, wq_cases._runtime_venv, wq_cases._bare_python
        rows = self.closure_rows(u, ("entry.U", "A", "C"))
        self.assertIs(rows["A"]["holder_survived_switch"], True, "the measured positive path")
        self.assertIs(rows["C"]["fresh_process_imports_restored_package"], True)
        silent_holder = types.SimpleNamespace(request=lambda op, timeout=60: None)
        holder = self.closure_mode(self.tmp / "U").hold  # the same root closure_rows uses
        changed = {**holder("A-pyd-holder", []), "io": silent_holder}
        rows = self.closure_rows(u, ("entry.U", "A"), hold=lambda label, argv, **env: changed)
        self.assertIsNone(rows["A"]["holder_survived_switch"], "a live holder that did not answer")
        rows = self.closure_rows(u, ("entry.U", "A"), **{"custody.alive": lambda identity: identity is not
                                                         changed["identity"]}, hold=lambda label, argv, **env: changed)
        self.assertIs(rows["A"]["holder_alive_before_switch"], False, "a holder measured gone fails the case")
        bounded = self.closure_mode(self.tmp / "U").run_bounded
        for label, timed_out, check, function, names in (
                ("C-fresh-observer", None, "fresh_process_imports_restored_package", u, ("entry.U", "C")),
                ("D-relocated-tool", None, "relocated_console_script_runs_new_generation", d, ("entry.D", "D")),
                ("E.D-tool", None, "console_script_after_cold_entry", d, ("entry.D", "E.D")),
                ("module-cli-help", None, "module_cli_entry_exits_0", f, ("F", "E.F")),
                ("module-entry-facts", None, "module_entry_redirected_before_imports", f, ("F", "E.F"))):
            for rc, expected in ((None, None), (1, False), (0, True)):
                answers = {label: (rc, bounded(label, [], 0)[1])}
                rows = self.closure_rows(function, names, run_bounded=lambda lab, argv, deadline, answers=answers,
                                         **kw: answers.get(lab) or bounded(lab, argv, deadline))
                self.assertIs(rows[names[-1]][check], expected, (label, rc))
        for rc, expected in ((None, None), (1, False)):
            answers = {"F-owner-installs-base-lock": (rc, "")}
            rows = self.closure_rows(f, ("F",), run_bounded=lambda lab, argv, deadline, answers=answers, **kw:
                                     answers.get(lab) or bounded(lab, argv, deadline))
            self.assertEqual(rows["F"], {"owner_installed_base_lock_into_interpreter": expected})
        start = {"schema": "winqual-start/1", "pid": 30, "birth": "300", "prefix": str(self.tmp / "U" / "app root" /
                 "runtime venv"), "base_prefix": str(self.tmp / "U" / "released app" / "_internal" / "python-standalone")}
        self.assertIsNone(self.closure_rows(d, ("entry.D",))["entry.D"]["server_runs_in_venv_on_app_python"],
                          "no start witness: the equipment's own observation is missing")
        self.assertIs(self.closure_rows(d, ("entry.D",), **{"install.witnesses": lambda: [start]})["entry.D"][
            "server_runs_in_venv_on_app_python"], True)
        self.assertIs(self.closure_rows(d, ("entry.D",), **{"install.witnesses": lambda: [{**start, "prefix": "C:\\x"}]})[
            "entry.D"]["server_runs_in_venv_on_app_python"], False)

    def test_d_holders_and_the_direct_entrys_parent_need_a_census_that_could_see_them(self):
        d, f = wq_cases._runtime_venv, wq_cases._bare_python
        rows = self.closure_rows(d, ("entry.D", "D"))["D"]
        self.assertEqual((rows["venv_python_holder_acknowledged"], rows["console_image_holder_acknowledged"]), (True, True))
        self.assertEqual(self.closure_rows(d, ("entry.D", "D"), **{"custody.members": lambda: None})["D"],
                         {"venv_python_holder_acknowledged": None}, "an unreadable census")
        self.assertEqual(self.closure_rows(d, ("entry.D", "D"), **{"custody.members": lambda: [self.UNREAD]})["D"],
                         {"venv_python_holder_acknowledged": None}, "an unread member may be the holder")
        self.assertEqual(self.closure_rows(d, ("entry.D", "D"), **{"custody.members": lambda: [self.OLD]})["D"],
                         {"venv_python_holder_acknowledged": False}, "a complete census without the holder image")
        rows = self.closure_rows(f, ("F", "E.F"))["E.F"]
        self.assertEqual([rows[name] for name in ("source_launcher_stopped", "direct_entry_ready",
                                                  "direct_command_is_waited_parent")], [True] * 3)
        for change, check, expected in (
                ({"custody.stop_all": lambda reason: {"confirmed": False}}, "source_launcher_stopped", None),
                ({"install.endpoint_gone": lambda server: False}, "source_launcher_stopped", False),
                ({"custody.members": lambda: None}, "direct_command_is_waited_parent", None),
                ({"custody.members": lambda: [{"pid": 30, "alive": None, "open_error": 5}]},
                 "direct_command_is_waited_parent", None),
                ({"custody.members": lambda: [{"pid": 30, "birth": "300", "alive": True, "ppid": 41, "cmdline": "x"}]},
                 "direct_command_is_waited_parent", False)):
            rows = self.closure_rows(f, ("F", "E.F"), **change)["E.F"]
            self.assertIs(rows[check], expected, (change, check))
        for boot_failed, expected in ((False, None), (True, False)):
            calls = iter([{"pid": 30, "birth": "300", "port": 1, "image": "x"}, None])  # F's server, then E.F's direct one
            rows = self.closure_rows(f, ("F", "E.F"), **{"install.wait_ready": lambda *a, **k: next(calls),
                                                          "install.boot_failed": boot_failed})
            self.assertIs(rows["E.F"]["direct_entry_ready"], expected, boot_failed)

    def test_d_product_launcher_is_the_venv_redirector_only_by_a_complete_census(self):
        """D's product holder: the launcher alive at the activation (its own census) runs from the venv redirector.
        Measured from that census only: an unread member or command line leaves it unmeasured."""
        d, root = wq_cases._runtime_venv, self.tmp / "U"  # the root closure_rows uses
        venv = root / "app root" / "runtime venv"
        own = {"pid": 70, "birth": "700", "alive": True, "image": str(venv / "Scripts" / "python.exe"),
               "cmdline": f'"{venv / "Scripts" / "python.exe"}" launcher.py --no-ui --seed-bundle x'}
        base = {**own, "pid": 71, "birth": "710", "image": "C:\\app\\_internal\\python-standalone\\python.exe"}
        holder = {**own, "pid": 72, "birth": "720", "cmdline": f'"{venv / "Scripts" / "python.exe"}" -m {fx.PACKAGE} hold'}
        accepted = self.closure_mode(root).activate()

        def check(members):
            rows = self.closure_rows(d, ("entry.D", "D"), activate=lambda *a: {**accepted, "old_members": members})
            return rows["D"]["product_launcher_runs_from_venv_redirector"]
        self.assertIs(check([own, base]), True, "the measured positive path")
        self.assertIs(check([base, holder]), False, "measured: no launcher on the redirector (a holder is no launcher)")
        self.assertIs(check([]), False, "a complete census without a launcher")
        self.assertIs(check([{**own, "alive": False, "exit_code": 0}, base]), False, "measured gone at the activation")
        self.assertIsNone(check(None), "no census")
        self.assertIsNone(check([own, self.UNREAD]), "an unread member")
        self.assertIsNone(check([{**own, "cmdline": None}]), "an unread command line")
        self.assertIn("product_launcher_runs_from_venv_redirector", wq_cases.REQUIRED_CHECKS["D"])

    def ten_cases(self, drained=True, reborn=False):
        """All three modes through the REAL step closures, the real ``_sequence`` and ``_close_and_export``. The
        ``Mode`` methods (each tested above) are stand-ins that record their own claims as measured; every other
        answer is the measured positive path. ``reborn``: B's crash-loop sample sees a new birth on the serving
        pid. ``drained=False``: a holder's output is still pending at every teardown."""
        mode = self.closure_mode(self.tmp / "seq")
        root, install, internal, bare = mode.root, mode.install, mode.ctx.release["internal"], mode.ctx.bare_root
        venv, private = install.app_root / "runtime venv", install.private
        u_site = wq_product.user_site(install.user_base)
        contexts = {"U": (u_site, internal / "python-standalone", None),
                    "D": (wq_cases.venv_site(venv), venv, internal / "python-standalone"),
                    "F": (wq_cases.venv_site(private), private, bare)}
        state, current, witnesses, starts, pids = {}, {}, {}, [], iter(range(100, 1000))
        (venv / "ouroboros-runtime-venv.json").write_text(json.dumps({"state": "ready"}), encoding="utf-8")
        (root / "pgen").mkdir(parents=True, exist_ok=True)
        (root / "pgen" / "ouroboros-private-environment.json").write_text(json.dumps({"repo_dir": str(install.repo)}))
        (root / "pgen" / "pyvenv.cfg").write_text(f"home = {bare}\n", encoding="utf-8")

        def record(case, *names):
            for name in names:
                case.check(name, True)

        def serve(version, site, prefix, base=None, image="python.exe"):
            pid = next(pids)
            witnesses[pid] = {"pid": pid, "birth": str(pid * 10), "prefix": str(prefix), "base_prefix": str(base or prefix),
                              "package": {"version": version, "realpath": str(site / fx.PACKAGE / "__init__.py")}}
            starts.append({"schema": "winqual-start/1", **{key: witnesses[pid][key] for key in ("pid", "birth", "prefix",
                                                                                                 "base_prefix")}})
            current["server"] = {"pid": pid, "birth": str(pid * 10), "port": 1, "image": image, "health": {}}
            return current["server"]

        def here(version):
            return serve(version, *contexts[state["label"]], image=str(mode.ctx.bare_python) if state["label"] == "F"
                         else "python.exe")

        def entry(case, extra, site, prefix):
            record(case, *wq_cases._ENTRY)
            state["version"] = "1.0"
            return here("1.0")

        def prepare(case, target, kind, fixed, site, server):
            record(case, *wq_cases._PREPARE)
            return {"path": root / ("pgen" if kind == "private" else "gen"), "id": [1, 1], "candidate_id": f"c-{target}",
                    "record": {"environment": {"generation": {"pip": "ensurepip" if kind == "private" else None}}},
                    "inventory": {fx.DIST: {"id": [7, 7]}}}

        def activate(case, candidate_id, expect):
            record(case, "activation_accepted", f"settled_{expect}", "attempt_bound_to_candidate_composition",
                   *(("adopted_composition",) if expect == "adopted" else ()), "ready_after_settlement")
            if expect == "adopted":
                state["version"] = fx.VERSIONS[candidate_id.split("-")[-1]]
            old, new = current.get("server"), here(state["version"])
            if reborn and expect == "returned_retry":
                current["server"] = {**new, "birth": new["birth"] + "1"}
            decision = ({"error": {"step": "serving_aside", "winerror": 32}} if expect == "returned_retry" else
                        {"detail": f"rc=1: ImportError: cannot import name 'stable_api' from '{fx.PACKAGE}' (x)"})
            return {"adoption": f"{new['pid']:012x}", "since": 0, "record": {"activation": {"decision": decision}},
                    "server": new, "old_server": old, "old_launchers": set(), "old_members": mode.custody.members(),
                    "trace": []}

        def cold(case, starter, server, fixed, version, site, prefix, released=False):
            record(case, *wq_cases._COLD, *(("released_app_handed_over_again",) if released else ()), "owner_layers_carried")
            return here(version)
        launcher = {"pid": 70, "birth": "700", "alive": True, "image": "python.exe", "cmdline": "python launcher.py --no-ui"}
        venv_launcher = {"pid": 71, "birth": "710", "alive": True, "image": str(venv / "Scripts" / "python.exe"),
                         "cmdline": "python launcher.py --no-ui"}  # D: the product launcher on the venv redirector
        images = [{"pid": 61, "birth": "601", "alive": True, "image": str(venv / "Scripts" / name), "cmdline": name}
                  for name in ("python.exe", "ouroboros-qual-tool.exe")]
        b_holder = {"identity": {"pid": 62, "birth": "620"}, "io": None, "ready": {"event": "ready",
                                                                                   "cwd": str(u_site / fx.PACKAGE)}}
        hold, waits = mode.hold, iter([lambda: serve("1.0", bare / "Lib" / "site-packages", bare, bare, str(mode.ctx.bare_python)),
                                       lambda: here("5.0")])
        mode.custody.members = lambda: [launcher, venv_launcher, *images, {**current["server"], "alive": True,
                                                                           "ppid": 40, "cmdline": "python server.py"}]
        install.server, install.wait_ready, install.witnesses = (lambda custody: current["server"]), (
            lambda *a, **k: next(waits)()), (lambda: list(starts))
        mode.hold = lambda label, argv, **env: b_holder if label == "B-cwd-holder" else hold(label, argv, **env)
        mode.entry, mode.prepare, mode.activate, mode.cold = entry, prepare, activate, cold
        mode.server_witness = lambda server, since_ns=0: witnesses.get((server or {}).get("pid"))
        mode.channel_and_marker = lambda case, name: record(case, "channel_development", f"marker_{name}_written")
        mode.marker = lambda case, name: record(case, f"marker_{name}_written")
        mode.seed_layers = lambda case: record(case, "owner_layers_seeded")
        mode.selected = lambda case, result, prepared, fixed, serving_id: record(
            case, "prepared_directory_selected", *(("serving_directory_kept_aside",) if serving_id is not None else ()))
        mode.fresh_generation = lambda case, result: record(case, "fresh_launcher", "fresh_server")
        mode.layers_carried = lambda case, head=None: record(case, "owner_layers_carried")
        mode.markers_kept = lambda case: record(case, "data_markers_kept")
        drain = {"complete": drained, "pumps": 3, "pending": [] if drained else ["logs/D-console-image-holder.stdout"],
                 "errors": []}
        teardown = types.SimpleNamespace(custody=types.SimpleNamespace(close=lambda: {"status": "confirmed", "drain": drain}),
                                         export=lambda: {"ok": True, "copied": 9, "errors": [], "missing_required": []})
        statuses, rows = {}, {}
        with mock.patch.object(wq_cases, "path_id", lambda path: [1, 1]), \
                mock.patch.object(wq_cases, "inventory", lambda site: {fx.DIST: {"id": [7, 7]}}), \
                mock.patch.object(wq_cases, "content_inventory", lambda root: {}), \
                mock.patch.object(wq_cases, "time", fast_clock()):
            for label, function in (("U", wq_cases._user_site), ("D", wq_cases._runtime_venv), ("F", wq_cases._bare_python)):
                state["label"] = label
                cases = {name: Case(name, required=wq_cases.REQUIRED_CHECKS[name]) for name in MODES[label]}
                function(mode, cases, REQUIRED)  # the real _sequence
                wq_cases._close_and_export(teardown, cases, {})
                statuses.update({name: case.status for name, case in cases.items()})
                rows.update({name: case.as_dict() for name, case in cases.items()})
        return statuses, exit_code(statuses, REQUIRED, False, True), rows

    def test_the_ten_cases_reach_pass_through_the_real_step_closures(self):
        """The round-3 regression class (every mode stopping after its first case) checked on the step code itself:
        with every observation measured, all ten cases PASS and the run exits 0."""
        statuses, code, rows = self.ten_cases()
        unmeasured = {name: [row["check"] for row in case["checks"] if row["ok"] is not True]
                      for name, case in rows.items()}
        self.assertEqual((statuses, code), ({name: PASS for name in REQUIRED}, EXIT["qualified"]), unmeasured)
        self.assertEqual({name: case["missing"] for name, case in rows.items() if case["missing"]}, {})
        statuses, code, _rows = self.ten_cases(drained=False)
        self.assertEqual((set(statuses.values()), code), ({UNKNOWN}, EXIT["unknown_or_not_run"]), "undrained output")
        statuses, code, _rows = self.ten_cases(reborn=True)
        self.assertEqual(([statuses[name] for name in MODES["U"]], code), ([PASS, PASS, FAIL, NOT_RUN, NOT_RUN],
                                                                           EXIT["fail"]), "Astra 2 inside the sequence")
        self.assertEqual({statuses[name] for name in (*MODES["D"], *MODES["F"])}, {PASS})

    # ------------------------------------------------------------------ 3: deadlines and silence
    def test_a_readiness_deadline_is_unmeasured_and_only_a_failed_boot_is_refuted(self):
        custody = types.SimpleNamespace(raw=types.SimpleNamespace(emit=lambda *a, **k: None), label="U",
                                        members=lambda: [])
        install = wq_product.Install(self.tmp / "U", 1)
        install.server = lambda custody: {"pid": 2, "birth": "2", "port": 1}
        failed, ready = (503, {"status": "error", "boot": {"phase": "failed"}}), (200, {"status": "ok", "boot": {"phase": "ready"}})
        with mock.patch.object(wq_product, "time", fast_clock()):
            for answer, expected in (((200, {"status": "ok", "boot": {"phase": "serving"}}), None), (failed, False),
                                     ((None, "ConnectionRefusedError(61)"), None), ((500, "Internal Server Error"), None),
                                     ((200, {"status": "ok", "boot": "ready"}), None)):
                install.http = lambda *a, answer=answer, **k: answer
                server = install.wait_ready(custody, 0.05)
                self.assertEqual((server, wq_cases._ready(install, server)), (None, expected), answer)
            answers = iter([failed, ready])  # a failed generation, then a ready one: no early verdict
            install.http = lambda *a, **k: next(answers)
            server = install.wait_ready(custody, 5)
            self.assertEqual((server["pid"], wq_cases._ready(install, server)), (2, True))
            install.server, install.boot_failed = (lambda custody: None), True  # no new server; nothing carried over
            self.assertIsNone(wq_cases._ready(install, install.wait_ready(custody, 0.05)))

    def preparing(self, views=None, post=(202, {"status": "preparing"}), markers=lambda names: {}, record=...,
                  reads=None, alive=True, clock=None, prepared_ids=()):
        """The real ``Mode.prepare``; ``views`` are successive candidate views (the last repeats), ``reads``
        successive bound-server reads (default: the serving process itself)."""
        fixed = self.tmp / "python-userbase"
        site = wq_product.user_site(fixed)
        (site / "ouroboros_qual_probe-1.0.dist-info").mkdir(parents=True, exist_ok=True)
        generation = self.tmp / "python-userbase.update-c9"
        (wq_product.user_site(generation) / "ouroboros_qual_probe-2.0.dist-info").mkdir(parents=True, exist_ok=True)
        serving = {"pid": 1, "birth": "1"}
        ready = {"state": "ready", "candidate_id": "c9", "target_sha": "f" * 40, "channel": "development",
                 "target_ref": "managed/ouroboros", "checks": {"stub": fx.STUB_LABEL}}
        views, reads = list(views or [ready]), list(reads or [{**serving, "port": 1}])
        if record is ...:
            record = {"environment": {"same": False, "generation": {"kind": "user_site", "current": str(fixed),
                                                                    "user_base": str(generation)}}}

        def http(method, path, *args, **kwargs):
            if method == "POST":
                return post
            return 200, {"candidate": views.pop(0) if len(views) > 1 else views[0]}
        install = types.SimpleNamespace(http=http, candidate=lambda cid: record, markers=markers, repo=self.tmp / "repo",
                                        server=lambda custody: reads.pop(0) if len(reads) > 1 else reads[0])
        mode = self.flow_mode([], install)
        mode.prepared_ids = list(prepared_ids)
        mode.mirror_to, mode.env = (lambda name: None), (lambda **extra: {})
        mode.custody.alive = lambda identity: alive
        complete = {"head": "a" * 40, "branch": "refs/heads/ouroboros", "index": "b" * 64, "status": "c" * 64, "files": {}}
        case = Case("A")
        with mock.patch.object(wq_cases, "layers", lambda *args, **kwargs: dict(complete)), \
                mock.patch.object(wq_cases, "time", clock or fast_clock()), \
                mock.patch.dict(wq_cases.DEADLINE, {"prepare": 0.2 if clock is None else 2700}):
            try:
                mode.prepare(case, "T2", "user_site", fixed, site, serving)
            except CaseStop:
                pass
        return case, {row["check"]: row["ok"] for row in case.checks}

    def test_preparation_tells_a_measured_answer_from_silence_and_a_deadline(self):
        case, rows = self.preparing()
        self.assertEqual((case.status, rows["prepared_ready"], rows["serving_untouched_by_preparation"]),
                         (PASS, True, True), "the measured positive path")
        for post, expected in (((None, "TimeoutError"), None), ((409, {"error": "busy"}), False),
                               ((202, {"status": "refused"}), False)):
            self.assertEqual(self.preparing(post=post)[1], {"prepare_accepted": expected}, post)
        for views, expected in (([{"state": "preparing"}], None), ([{"state": "checking", "candidate_id": "c9"}], None),
                                ([{}], None), ([{"state": "checking"}, {"state": "failed", "candidate_id": "c9"}], False),
                                ([{"state": "none", "refusal": "another preparation runs"}], False)):
            self.assertIs(self.preparing(views=views)[1]["prepared_ready"], expected, views)
        stale = {"state": "ready", "candidate_id": "c1"}
        self.assertIsNone(self.preparing(views=[stale], prepared_ids=["c1"])[1]["prepared_ready"], "only the last attempt")
        self.assertIsNone(self.preparing(record=None)[1]["prepared_beside_fixed_path"], "the record could not be read")
        self.assertIs(self.preparing(record={"environment": {}})[1]["prepared_beside_fixed_path"], False)
        for alive, expected in ((False, False), (None, None)):  # no bound server for 60 s
            rows = self.preparing(views=[{"state": "preparing"}], reads=[None], alive=alive, clock=fast_clock(step=30))[1]
            self.assertIs(rows["prepared_ready"], expected, alive)

    def test_serving_is_untouched_only_by_measured_readings(self):
        rows = self.preparing(markers=lambda names: None)[1]
        self.assertIsNone(rows["serving_untouched_by_preparation"], "two unanswered marker reads are not equal ones")
        answers = iter([{"winqual": True}, None])
        self.assertIsNone(self.preparing(markers=lambda names: next(answers))[1]["serving_untouched_by_preparation"])
        answers = iter([{"winqual": True}, {"winqual": False}])
        self.assertIs(self.preparing(markers=lambda names: next(answers))[1]["serving_untouched_by_preparation"], False)
        for read, alive, expected in (({"pid": 1, "birth": "2", "port": 1}, True, False), (None, False, False),
                                      (None, None, None), (None, True, None)):
            reads = [{"pid": 1, "birth": "1", "port": 1}, read]  # the loop's read, then the comparison's
            rows = self.preparing(reads=reads, alive=alive)[1]
            self.assertIs(rows["serving_untouched_by_preparation"], expected, (read, alive))
        self.assertEqual([conjoin(), conjoin(True, None), conjoin(None, False), conjoin(True, 1)], [True, None, False, False])
        self.assertEqual([prepared_ready(view) for view in (None, {"state": "ready", "stale": "x"}, {"state": "ready"},
                                                            {"state": "refused"}, {"server_ended": True}, {"state": "none"})],
                         [None, None, True, False, False, None])

    def test_activation_silence_and_unsettled_records_are_unmeasured(self):
        mode = self.flow_mode([[self.OLD]], self.activation_install(response=(None, "ConnectionResetError(54)")))
        case = Case("A")
        with self.assertRaises(CaseStop):
            mode.activate(case, "c1", "adopted")
        self.assertEqual((case.checks[-1]["check"], case.status), ("activation_accepted", UNKNOWN), "no answer")
        for boot_failed, status in ((False, UNKNOWN), (True, FAIL)):
            install = self.activation_install(ready=None)
            install.boot_failed, case = boot_failed, Case("A")
            with self.assertRaises(CaseStop):
                self.flow_mode([[self.OLD]], install).activate(case, "c1", "adopted")
            self.assertEqual((case.checks[-1]["check"], case.status), ("ready_after_settlement", status), boot_failed)
        unsettled = {**self.ACTIVATED, "state": "activating"}
        case = Case("A")
        with mock.patch.object(wq_cases, "time", fast_clock()), mock.patch.dict(wq_cases.DEADLINE, {"settle": 0.05}), \
                self.assertRaises(CaseStop):
            self.flow_mode([[self.OLD]], self.activation_install(record=unsettled)).activate(case, "c1", "adopted")
        self.assertEqual((case.checks[-1]["check"], case.checks[-1]["ok"], case.status), ("settled_adopted", None, UNKNOWN))
        activating = {"state": "activating", "activation": {}}
        self.assertIsNone(settled(activating, {"outcome": "adopted"}, "adopted"), "closed, the record not yet settled")
        self.assertIs(settled(activating, {"outcome": "returned:import_probe_failed"}, "adopted"), False)
        self.assertIsNone(settled(activating, {"outcome": "returned:import_probe_failed"}, "returned"))
        with self.assertRaises(ValueError):
            settled(activating, {}, "nonsense")
        self.assertIsNone(prepared_environment(None, "user_site", "C:\\x")[0], "an unread record is not a refuted one")

    def test_an_owner_action_or_a_holder_that_did_not_answer_is_unmeasured(self):
        for post, seen, expected in ((None, "stable", None), (200, None, None), (200, "stable", False),
                                     (409, "stable", False), (200, "development", True)):
            mode = self.flow_mode([], types.SimpleNamespace(
                set_channel=lambda channel, post=post, seen=seen: (post in (200, 201) and seen == channel,
                                                                   {"post": post, "seen": seen}),
                add_marker=lambda name: (True, {})))
            case = Case("A")
            with contextlib.suppress(CaseStop):
                mode.channel_and_marker(case, "winqual-a-before")
            self.assertIs(case.checks[0]["ok"], expected, (post, seen))

        def deadline(timeout):
            raise subprocess.TimeoutExpired("holder", timeout)
        mode = self.flow_mode([], None)
        mode.custody = types.SimpleNamespace(alive=lambda identity: True)
        holder = {"identity": {"pid": 9, "birth": "9"}, "proc": types.SimpleNamespace(wait=deadline),
                  "io": types.SimpleNamespace(request=lambda op, timeout=60: None)}
        self.assertEqual(mode.release_holder(holder), (None, None), "a holder still running at the deadline")

    # ------------------------------------------------------------------ 4: the holders' output and the export
    def drained_export(self, name, release_after=None, fails=False, allowance=5):
        """Astra's log-export race through the real ``Custody.close`` and ``_close_and_export``: the holder
        is gone, its last 54 bytes are still in the pipe, released ``release_after`` seconds into the teardown."""
        events = types.SimpleNamespace(emit=lambda *a, **k: None)
        ctx = types.SimpleNamespace(run_root=self.tmp / name, evidence=self.tmp / name / "evidence", raw=events)
        mode = wq_cases.Mode(ctx, "U")
        for folder in (mode.install.logs, mode.install.witness_dir, mode.install.data / "logs", mode.evidence):
            folder.mkdir(parents=True)
        transcript = mode.install.logs / "holder.stdout"
        transcript.touch()
        entered, release = threading.Event(), threading.Event()

        class Stream:
            used = False

            def readline(self):
                if self.used:
                    return b""
                self.used = True
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("fixture output never released")
                if fails:
                    raise OSError("the pipe broke")
                return b'{"event":"ready","evidence":"buffered holder output"}\n'
        mode.custody = Custody(FakeWin(), events, "U", [str(self.tmp)])
        mode.custody.pumps.append(wq_custody.Interactive(types.SimpleNamespace(stdout=Stream()), transcript))
        self.assertTrue(entered.wait(5))
        timer = threading.Timer(release_after, release.set) if release_after is not None else None
        if timer:
            timer.start()
        case = Case("A", required=("x",))
        case.check("x", True)
        try:
            with mock.patch.object(wq_custody, "TEARDOWN_ALLOWANCE", allowance):
                teardown = wq_cases._close_and_export(mode, {"A": case}, {})
        finally:
            release.set()  # leave no fixture thread waiting
            mode.custody.pumps[0].pump.join(15)
        exported = mode.evidence / "harness-logs" / "holder.stdout"
        return teardown, case.status, exported.stat().st_size

    def test_the_export_holds_the_holders_last_output_or_does_not_count(self):
        teardown, status, size = self.drained_export("released", release_after=0.3)
        self.assertEqual((teardown["drain"]["complete"], teardown["export"]["ok"], status, size), (True, True, PASS, 54),
                         "the drain waited for the pipe's end, then the copy held every byte")
        teardown, status, size = self.drained_export("pending", allowance=0.3)
        self.assertEqual((teardown["drain"]["complete"], teardown["export"]["ok"], status, size), (False, False, UNKNOWN, 0))
        self.assertEqual(len(teardown["drain"]["pending"]), 1)
        teardown, status, _size = self.drained_export("broken", release_after=0.1, fails=True)
        self.assertEqual((teardown["drain"]["complete"], status), (False, UNKNOWN))
        self.assertIn("the pipe broke", teardown["drain"]["errors"][0])
        # Astra's own condition: a teardown that reports no drain at all (its fake close) is not drained
        ctx = types.SimpleNamespace(run_root=self.tmp / "nodrain", evidence=self.tmp / "nodrain" / "evidence",
                                    raw=types.SimpleNamespace(emit=lambda *a, **k: None))
        mode = wq_cases.Mode(ctx, "U")
        for folder in (mode.install.logs, mode.install.witness_dir, mode.install.data / "logs", mode.evidence):
            folder.mkdir(parents=True)
        mode.custody = types.SimpleNamespace(close=lambda: {"status": "confirmed"})
        self.assertIs(wq_cases._close_and_export(mode, {}, {})["export"]["ok"], False)
        mode.custody = None  # custody never opened: no holder was started, nothing to drain
        self.assertIs(wq_cases._close_and_export(mode, {}, {})["export"]["ok"], True)

    def test_every_interactive_start_is_tracked_for_the_teardown_drain(self):
        class SpawnWin(FakeWin):
            def assign(self, job, handle):
                pass

            def birth(self, handle):
                return "777"

            def resume(self, handle):
                pass
        events = types.SimpleNamespace(emit=lambda *a, **k: None)
        custody = Custody(SpawnWin(), events, "U", [str(self.tmp)])
        procs = iter([types.SimpleNamespace(pid=77, _handle=1, stdout=io.BytesIO(b'{"event": "ready"}\n'),
                                            stdin=io.BytesIO()), types.SimpleNamespace(pid=78, _handle=2)])
        fake = types.SimpleNamespace(Popen=lambda *a, **k: next(procs), PIPE=-1, DEVNULL=-3)
        with mock.patch.object(wq_custody, "subprocess", fake):
            _proc, _identity, holder = custody.spawn("H", ["holder"], env={}, cwd=self.tmp, logs=self.tmp / "logs",
                                                     interactive=True)
            custody.spawn("tool", ["tool"], env={}, cwd=self.tmp, logs=self.tmp / "logs")
        self.assertEqual(custody.pumps, [holder], "the interactive start, and only it")
        teardown = custody.close()
        self.assertEqual((teardown["status"], teardown["drain"]), ("confirmed", {
            "complete": True, "pumps": 1, "pending": [], "errors": []}))
        self.assertEqual((self.tmp / "logs" / "H.stdout").read_bytes(), b'{"event": "ready"}\n')

    # ------------------------------------------------------------------ 5: each mode's own roots and ports
    def test_every_mode_runs_on_its_own_root_and_its_own_ports(self):
        ctx = types.SimpleNamespace(run_root=self.tmp / "run", evidence=self.tmp / "run" / "evidence",
                                    git_dir="C:\\Git\\cmd", wheelhouse=self.tmp / "wheelhouse")
        host = {"SystemRoot": "C:\\Windows", "OUROBOROS_HOST_SERVICE_PORT": "8767", "OUROBOROS_SERVER_PORT": "8765"}
        with mock.patch.dict(os.environ, host):
            modes = {label: wq_cases.Mode(ctx, label) for label in MODES}
            envs = {label: mode.env() for label, mode in modes.items()}
        ports, mains = [], [int(env["OUROBOROS_SERVER_PORT"]) for env in envs.values()]
        for label, mode in modes.items():
            env = envs[label]
            self.assertEqual(mode.root, ctx.run_root / label)
            for key in ("USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "OUROBOROS_APP_ROOT",
                        "PIP_CACHE_DIR", "GIT_CONFIG_GLOBAL", "WINQUAL_WITNESS_DIR"):
                self.assertTrue(pathlib.Path(env[key]).is_relative_to(mode.root), (label, key, env[key]))
            main, service = int(env["OUROBOROS_SERVER_PORT"]), int(env["OUROBOROS_HOST_SERVICE_PORT"])
            self.assertEqual(main, mode.install.requested_port, "what the mode reads is what it asked for")
            self.assertNotIn(service, range(8765, 8775), "the product's defaults (8767 collided in a real fixture)")
            self.assertFalse(any(other < service <= other + 9 for other in mains), "inside a main port's fallback window")
            ports += [main, service]
        self.assertEqual(len(set(ports)), 6, "no two services of the three modes share a port")



class BindingTests(Scratch):
    """The product commit and the equipment commit stay two identities (``qualify.bind_equipment``, ``--bind-only``)
    and the job is the pushed equipment commit carrying its own pins (the workflow and ``run.ps1``, read as text:
    no PowerShell on the authoring host)."""

    GIT = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@invalid", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

    def git(self, repo, *args):
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120,
                              env={**os.environ, **self.GIT})
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.strip()

    def commit(self, repo, files, message):
        for name, data in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_bytes(data)
            self.git(repo, "add", "--", name)
        self.git(repo, "commit", "--quiet", "--no-verify", "-m", message)
        return self.git(repo, "rev-parse", "HEAD")

    def branch(self, name="repo", extra=None, template=None, workflow=b"name: q\n"):
        """``(repo, candidate, its tree, equipment commit)``: a product commit and one equipment commit on top."""
        repo = self.tmp / name
        repo.mkdir()
        self.git(repo, "init", "--quiet")
        candidate = self.commit(repo, {"server.py": b"print('product')\n", "supervisor/git_ops.py": b"X = 1\n"},
                                "product")
        files = {f"{qualify.EQUIPMENT_DIR}/qualify.py": b"# equipment\n", qualify.WORKFLOW: workflow,
                 f"{qualify.EQUIPMENT_DIR}/update-windows-qualification.yml": workflow if template is None else template}
        files.update(extra or {})
        return repo, candidate, self.git(repo, "rev-parse", f"{candidate}^{{tree}}"), self.commit(repo, files, "equipment")

    def test_both_identities_are_recorded_apart(self):
        import hashlib
        repo, candidate, tree, equipment = self.branch()
        bound = qualify.bind_equipment(repo, equipment, candidate, tree)
        self.assertEqual(bound["product"], {"sha": candidate, "tree": tree})
        self.assertEqual((bound["equipment"]["commit"], bound["equipment"]["parent"]), (equipment, candidate))
        self.assertNotIn(bound["equipment"]["tree"], (tree, candidate, equipment))
        self.assertEqual(bound["equipment"]["changed_paths"], sorted(bound["equipment"]["files"]))
        self.assertEqual(bound["equipment"]["files"][f"{qualify.EQUIPMENT_DIR}/qualify.py"]["sha256"],
                         hashlib.sha256(b"# equipment\n").hexdigest(), "the bytes that ran, by content")

    def test_an_equipment_commit_that_is_not_exactly_candidate_plus_equipment_refuses(self):
        refused = {
            "a product file rides on the equipment commit": self.branch("r1", extra={"server.py": b"changed\n"}),
            "the running workflow is not the tested copy": self.branch("r2", template=b"name: other\n")}
        repo, candidate, tree, equipment = self.branch("r3")
        refused.update({
            "another tree": (repo, candidate, "0" * 40, equipment),
            "another candidate": (repo, equipment, self.git(repo, "rev-parse", f"{equipment}^{{tree}}"), equipment),
            "a job naming another commit": (repo, candidate, tree, candidate),
            "a placeholder": (repo, "REPLACE_WITH_FROZEN_CANDIDATE_COMMIT", tree, equipment)})
        repo4 = self.tmp / "r4"  # the equipment commit without its workflow file
        repo4.mkdir()
        self.git(repo4, "init", "--quiet")
        first = self.commit(repo4, {"server.py": b"p\n"}, "product")
        alone = self.commit(repo4, {f"{qualify.EQUIPMENT_DIR}/qualify.py": b"# e\n"}, "equipment without workflow")
        refused["no workflow file"] = (repo4, first, self.git(repo4, "rev-parse", f"{first}^{{tree}}"), alone)
        repo7 = self.tmp / "r7"  # the workflow without the copy the portable tests read
        repo7.mkdir()
        self.git(repo7, "init", "--quiet")
        first7 = self.commit(repo7, {"server.py": b"p\n"}, "product")
        untested = self.commit(repo7, {f"{qualify.EQUIPMENT_DIR}/qualify.py": b"# e\n", qualify.WORKFLOW: b"name: q\n"},
                               "equipment without its tested workflow copy")
        refused["no tested workflow copy"] = (repo7, first7, self.git(repo7, "rev-parse", f"{first7}^{{tree}}"), untested)
        repo5, candidate5, tree5, equipment5 = self.branch("r5")  # a merge of the candidate and the equipment
        self.git(repo5, "checkout", "--quiet", "--detach", candidate5)
        self.git(repo5, "merge", "--quiet", "--no-ff", "--no-edit", equipment5)
        refused["two parents, the candidate one of them"] = (repo5, candidate5, tree5, self.git(repo5, "rev-parse", "HEAD"))
        repo6, candidate6, tree6, equipment6 = self.branch("r6")  # the checkout moved past the commit the job names
        self.commit(repo6, {f"{qualify.EQUIPMENT_DIR}/later.txt": b"later\n"}, "a later equipment commit")
        refused["a checkout past the named commit"] = (repo6, candidate6, tree6, equipment6)
        for label, (repo_x, candidate_x, tree_x, equipment_x) in refused.items():
            with self.subTest(label), self.assertRaises(SystemExit):
                qualify.bind_equipment(repo_x, equipment_x, candidate_x, tree_x)

    def test_bind_only_refuses_unbound_pins_and_writes_both_identities(self):
        repo, candidate, tree, equipment = self.branch()
        checkout, inputs, run_root = self.tmp / "candidate", self.tmp / "inputs.json", self.tmp / "run"
        self.git(self.tmp, "clone", "--quiet", "--no-checkout", str(repo), str(checkout))
        self.git(checkout, "checkout", "--quiet", "--detach", candidate)

        def bind(sha, tree_value, at=checkout):
            inputs.write_text(json.dumps({"candidate": {"sha": sha, "tree": tree_value}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return qualify.main(["--bind-only", "--inputs", str(inputs), "--run-root", str(run_root), "--candidate",
                                     str(at), "--equipment-repo", str(repo), "--equipment-commit", equipment])
        self.assertEqual(bind("REPLACE_WITH_FROZEN_CANDIDATE_COMMIT", "REPLACE_WITH_FROZEN_CANDIDATE_TREE"), EXIT["usage"])
        self.assertEqual(bind(candidate, tree, at=repo), EXIT["usage"], "a candidate checkout at the equipment commit")
        self.assertFalse((run_root / "evidence" / "bindings.json").exists(), "nothing recorded for a refusal")
        self.assertEqual(bind(candidate, tree), 0)
        bound = json.loads((run_root / "evidence" / "bindings.json").read_text(encoding="utf-8"))
        self.assertEqual((bound["product"]["sha"], bound["equipment"]["commit"]), (candidate, equipment))

    def test_a_run_binds_the_equipment_before_it_reads_the_release(self):
        """The driver's own run path (``bind_inputs``) refuses a foreign equipment commit before it reads anything of
        the released app or the bare interpreter (neither exists here), and carries both identities when bound."""
        repo, candidate, tree, equipment = self.branch(extra={"supervisor/git_ops.py": b"X = 2\n"})
        checkout, inputs = self.tmp / "candidate", self.tmp / "inputs.json"
        self.git(self.tmp, "clone", "--quiet", "--no-checkout", str(repo), str(checkout))
        self.git(checkout, "checkout", "--quiet", "--detach", candidate)
        inputs.write_text(json.dumps({"candidate": {"sha": candidate, "tree": tree}, "release": {}, "python": {}}),
                          encoding="utf-8")
        events = []
        args = types.SimpleNamespace(inputs=str(inputs), candidate=str(checkout), equipment_repo=str(repo),
                                     equipment_commit=equipment, release_dir=str(self.tmp / "absent"),
                                     bare_python=str(self.tmp / "absent" / "python.exe"), run_root=str(self.tmp / "run"),
                                     wheelhouse=str(self.tmp))
        with self.assertRaises(SystemExit) as refused:
            qualify.bind_inputs(args, types.SimpleNamespace(emit=lambda *a, **k: events.append((a, k))))
        self.assertIn("supervisor/git_ops.py", str(refused.exception), "refused for the product file it carries")
        self.assertEqual(events, [], "nothing bound, nothing read")

    def test_the_job_runs_on_the_pushed_equipment_commit_with_the_pins_it_carries(self):
        workflow = "\n".join(line for line in (HERE / "update-windows-qualification.yml").read_text(
            encoding="utf-8").splitlines() if not line.lstrip().startswith("#"))  # what GitHub reads, not the notes
        run = (HERE / "run.ps1").read_text(encoding="utf-8")
        self.assertIn("branches: ['qualification/update-windows-*']", workflow)
        for needed in ("fetch-depth: 0", "persist-credentials: false", "-EquipmentCommit $env:GITHUB_SHA",
                       "if: always()", "contents: read"):
            self.assertIn(needed, workflow)
        for gone in ("candidate_sha", "candidate_tree", "candidate_ref", "pull_request", "concurrency", "secrets."):
            self.assertNotIn(gone, workflow)
        # Unbound pins refuse, the candidate is checked out on its own and both identities are bound, all before
        # the 772 MB release download; the driver then gets the same identity arguments.
        order = [run.index(marker) for marker in ("does not bind the candidate", "checkout --quiet --detach $CandidateSha",
                                                  "--bind-only @identity", "$inputs.release.url")]
        self.assertEqual(order, sorted(order))
        self.assertIn("+ $identity", run)
        inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
        self.assertEqual((inputs["release"]["size"], inputs["release"]["sha256"], inputs["release"]["source_sha"]),
                         (772126896, "44782370dacfed116363d94c6ce92103a3184f0a42476efa56c7000d6249e2a8",
                          "6f3ddff05ff3e6c6f8d007ff96358df6af872e83"), "the owner-fixed released input is not rebound")


if __name__ == "__main__":
    unittest.main()
