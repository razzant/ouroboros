"""SM1's direct adoption through ``scenarios.run_sm1`` without a lane server: every identity in the chain, swapped.

The scene is what the product leaves after one SM1 task, built from its own pieces: real Git with a linked candidate,
the stub author's real release projection, the review ledger's real writers, the adoption's real log writer and a real
loopback serve of the served bytes. A real ``body_adoption`` and ``request_restart`` answer the negative qualification
paths. This pins the stand's oracle: the positive path over a real server, real tests and a real browser is the keyless
rehearsal (``test_stub_sm1_end_to_end_on_a_real_isolated_server``), which this file does not replace or claim.
"""
from __future__ import annotations

import http.server
import json
import pathlib
import re
import shutil
import subprocess
import threading
import time
import types

import pytest

from devtools.e2e_live import adoption, scenarios, stub_lane
from ouroboros import body_adoption, body_candidate, preflight_qualification, preflight_receipts
from ouroboros import review_ledger as ledger
from tests.body_candidate_support import candidate_commit, git, isolate, make_ctx, make_serving
from tests.system_e2e.harness import ArtifactOracle, wait_until
from tests.test_review_ledger import _facts, _panel
from tests._cases import case_folders

REPO = pathlib.Path(__file__).resolve().parents[1]
REAL_EVIDENCE = preflight_qualification.adoption_test_evidence  # ``isolate`` stubs it for switching tests
TASK = "sm1-task"
OTHER_SHA = "c" * 40
SCENE_START_ACCENT = "#c93545"   # the scene's own seed accent, never SM1's: the executing checkout may already serve it
DARK_ROOT, LIGHT_ROOT = ":root {", ':root[data-theme="light"] {'
# Where an author's edit can stop short of the palette: only ``--accent``, in these theme blocks.
SHORT_EDITS = {"accent-alone": (DARK_ROOT, LIGHT_ROOT), "dark-only": (DARK_ROOT,)}


def _accent_alone(css: str, *openings: str) -> str:
    for opening in openings:
        start = css.index(opening)
        end = css.index("\n}", start)
        block, found = re.subn(r"(?m)^(\s*--accent:\s*)[^;]+;", rf"\g<1>{scenarios.SM1_NEW_ACCENT};", css[start:end], count=1)
        assert found == 1, opening
        css = css[:start] + block + css[end:]
    return css


def _release_inputs(source: pathlib.Path, dest: pathlib.Path) -> None:
    """The release carriers and the shared palette the scripted author edits, as ``source`` holds them."""
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS

    for rel in sorted({*CARRIER_SPAN_PATHS, scenarios.SM1_CSS_PATH}):
        if (source / rel).is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / rel, dest / rel)


class _Host:
    """One SM1 task's durable outcome; each ``fault`` swaps the one identity a test is about."""

    def __init__(self, root: pathlib.Path, fault: str = "", source: pathlib.Path = REPO) -> None:
        self.fault, self.lane = fault, root / "lane"
        self.clone, self.data = self.lane / "clone", self.lane / "data"
        _release_inputs(source, self.clone)
        css = self.clone / scenarios.SM1_CSS_PATH   # inside a candidate a real SM1 landed, copying it alone makes no change
        css.write_text(scenarios.css_with_accent(css.read_text(encoding="utf-8"), SCENE_START_ACCENT), encoding="utf-8")
        self.seed_css = css.read_text(encoding="utf-8")
        for args in (("init", "-q", "-b", "ouroboros"), ("config", "core.autocrlf", "false"), ("config", "user.name", "t"),
                     ("config", "user.email", "t@e.invalid"), ("add", "-A"), ("commit", "-q", "-m", "seed")):
            self.git(*args)
        self.old = self.git("rev-parse", "HEAD")
        self.result = scenarios.SM1_STUB_FINAL
        self.sha, self.tree = self._author()
        self._durable_rows()
        self.served_body = None   # None: the checkout's bytes, as the real static mount serves them
        self.served_sha = self.old
        self.proc = types.SimpleNamespace(poll=lambda: 1 if fault == "server_exited" else None)

    def run_the_task(self) -> dict:
        """The answer to the stand's submit: everything above becomes true once the task has run and the server re-exec'd."""
        if self.fault not in {"checkout_not_moved", "no_request", "restart_refused", "restart_plain", "closed_abandoned",
                              "unconfirmed", "no_settlement", "handoff_other_commit", "closed_other_id"}:
            self.git("merge", "-q", "--ff-only", self.sha)
        self.served_sha = self.old if self.fault == "state_serves_old" else self.git("rev-parse", "HEAD")
        return {"task_id": TASK}

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=str(self.clone), check=True, capture_output=True, text=True).stdout.strip()

    def served_stylesheet(self) -> bytes:
        return self.served_body if self.served_body is not None else (self.clone / scenarios.SM1_CSS_PATH).read_bytes()

    def _author(self) -> tuple[str, str]:
        """The scripted author's writes in its linked candidate, committed as ``commit_reviewed`` does there."""
        self.candidate = self.lane / "worktrees" / f"body_{TASK}"
        self.git("worktree", "add", "-q", "-b", f"candidate/{TASK}", str(self.candidate), self.old)
        for step in scenarios.sm1_stub_script(self.clone, self.data)["agent"]:
            if isinstance(step, dict) and step.get("tool") == "write_file":
                (self.candidate / step["arguments"]["path"]).write_text(step["arguments"]["content"], encoding="utf-8")
        if self.fault in SHORT_EDITS:
            (self.candidate / scenarios.SM1_CSS_PATH).write_text(_accent_alone(self.seed_css, *SHORT_EDITS[self.fault]), encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=str(self.candidate), check=True)
        subprocess.run(["git", "commit", "-q", "-m", scenarios.SM1_COMMIT_MESSAGE], cwd=str(self.candidate), check=True)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(self.candidate), check=True, capture_output=True, text=True).stdout.strip()
        return sha, self.git("rev-parse", f"{sha}^{{tree}}")

    def _journal(self, name: str, *rows: dict) -> None:
        path = self.data / "logs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as sink:
            sink.writelines(json.dumps(row) + "\n" for row in rows)

    def _durable_rows(self) -> None:
        fault, sha = self.fault, self.sha
        record = ledger.write_record(self.data, ledger.build_commit_gate_record(_facts(_panel(), task_id=TASK)))
        ledger.note_commit_landing(self.data, record["record_id"], None, {"commit_sha": sha, "tree_sha": self.tree, "parent_sha": self.old})
        self.record_id = record["record_id"]
        landing = (f"OK: committed to candidate/{TASK}: {scenarios.SM1_COMMIT_MESSAGE} [not pushed: commit is on candidate "
                   f"branch candidate/{TASK}; publication stays an explicit Git/PR step]\nreview_record_id: {self.record_id}")
        request = {"type": "tool_call", "tool": "request_restart", "status": "ok", "args": {"reason": scenarios.SM1_RESTART_REASON, "adopt_commit": sha},
                   "result_preview": f"Restart requested. It adopts candidate commit {sha[:12]} onto ouroboros once this generation's readers have stopped."}
        if fault == "restart_refused":
            request.update(status="blocked", result_preview="⚠️ RESTART_BLOCKED: ADOPTION_TESTS_REQUIRED: no qualifying test run of exactly it is on record")
        if fault == "restart_plain":
            request.update(args={"reason": "just restart"}, result_preview="Restart requested. The candidate is NOT adopted by this restart; it stays retained.")
        if fault == "request_other_commit":   # the root asked for another commit; this one's handoff came from elsewhere
            request.update(args={"reason": scenarios.SM1_RESTART_REASON, "adopt_commit": OTHER_SHA},
                           result_preview=f"Restart requested. It adopts candidate commit {OTHER_SHA[:12]} onto ouroboros once this generation's readers have stopped.")
        tools = [{"type": "tool_call_started", "tool": "commit_reviewed"},
                 {"type": "tool_call", "tool": "commit_reviewed", "status": "ok", "args": {"paths": []}, "result_preview": landing}]
        self._journal("tools.jsonl", *tools, *([] if fault == "no_request" else [request]))
        self._journal("events.jsonl", {"type": "llm_usage", "task_id": TASK, "prompt_tokens": 11, "completion_tokens": 3})
        reviewed = [OTHER_SHA if fault == "candidate_other_commit" else sha]
        copies = self.data / "state" / "private_copies"
        copies.mkdir(parents=True, exist_ok=True)
        (copies / f"body_{TASK}.json").write_text(json.dumps({
            "kind": "body_candidate", "task_id": TASK, "candidate_id": f"body_{TASK}", "path": str(self.candidate),
            "branch": f"candidate/{TASK}", "base_sha": self.old, "reviewed_commits": reviewed}), encoding="utf-8")
        tests = {"tier": "selected", "source": "receipt", "head": OTHER_SHA if fault == "qualification_other_head" else sha,
                 "tree": OTHER_SHA if fault == "qualification_other_tree" else self.tree, "key": "selected-key", "phase": "post_commit",
                 "base": self.old, "changed": 3, "rule": "floor", "file_count": 2}
        self.task_row = {"status": "completed", "result": self.result}
        if fault == "evolution_beside":
            self._adoption_rows(OTHER_SHA, "evolution-adoption", dict(tests, head=OTHER_SHA), "adopted")
        if fault not in {"no_request", "restart_refused", "restart_plain"}:
            handoff_sha = OTHER_SHA if fault == "handoff_other_commit" else sha
            outcome = {"closed_abandoned": "abandoned:not_armed"}.get(fault, "adopted")
            self._adoption_rows(handoff_sha, "sm1-adoption", tests, outcome)

    def _adoption_rows(self, sha: str, adoption_id: str, tests: dict, outcome: str) -> None:
        handoff = {"id": adoption_id, "cand": sha, "old": self.old}
        body_adoption._log(self.data, "body_adoption_authorized", handoff, branch="ouroboros", paths=3)
        if self.fault in {"unconfirmed", "unconfirmed_then_adopted"} and adoption_id == "sm1-adoption":
            # A boot that did not confirm it; the product keeps the handoff for the next start (``body_adoption._waiting``).
            body_adoption._log(self.data, "body_adoption_unconfirmed", handoff, reason="native_host")
        if self.fault in {"unconfirmed", "no_settlement"} and adoption_id == "sm1-adoption":
            return
        closed_id = "another-adoption" if self.fault == "closed_other_id" and adoption_id == "sm1-adoption" else adoption_id
        body_adoption._log(self.data, "body_adoption_closed", {**handoff, "id": closed_id}, outcome=outcome, phase="adopted",
                           events=[{"phase": phase} for phase in ("authorized", "armed", "switched", "adopted")], tests=tests)
        observed = OTHER_SHA if self.fault == "verify_other_sha" and adoption_id == "sm1-adoption" else sha
        if self.fault != "no_verify" or adoption_id != "sm1-adoption":
            self._journal("events.jsonl", {"type": "restart_verify", "ok": observed == sha, "expected_sha": sha, "observed_sha": observed})
        self._journal("events.jsonl", {"type": "worker_boot", "git_sha": sha})
        if self.fault != "no_task_record" and adoption_id == "sm1-adoption":
            self.task_row.setdefault(body_adoption.ADOPTIONS_KEY, []).append(
                {"adoption_id": closed_id, "outcome": outcome, "commit_sha": sha, "previous_sha": self.old, "branch": "ouroboros"})


class _Oracle(ArtifactOracle):
    """The scene's task row stands in for the store; every other durable read is the stand's real one."""

    def __init__(self, data_root, host) -> None:
        super().__init__(data_root)
        self.host = host

    def task_result(self, task_id: str) -> dict:
        return dict(self.host.task_row)

    def task_drive(self, task_id: str):
        return self


class _Page:
    """What a browser computes from the stylesheet this scene serves, in the theme asked for: the palette the scripted
    author really landed, never a canned accent, so a one-theme or token-only edit fails this chain too."""

    def __init__(self, host: _Host) -> None:
        self.host, self.scheme = host, ""

    def color_scheme(self, scheme):
        self.scheme = scheme

    def computed_property(self, _selector, name):
        return scenarios.css_theme_tokens(self.host.served_stylesheet().decode("utf-8"))[self.scheme].get(name, "")

    goto = screenshot = close = lambda self, *a, **k: None


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """A factory: ``(host, ctx)`` of one scene, its bytes served over real loopback HTTP."""
    served = []

    def build(fault: str = "", *, ready_timeout: float = 1.5, stub: bool = True, source: pathlib.Path = REPO):
        host = _Host(tmp_path, fault, source)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib callback name
                if self.path == "/api/health":
                    version = (host.clone / "VERSION").read_text(encoding="utf-8").strip()
                    body = json.dumps({"runtime_version": version}).encode("utf-8")
                elif self.path in scenarios.SM1_SERVED_STYLESHEETS:
                    body = host.served_stylesheet()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        http_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=http_server.serve_forever, daemon=True).start()
        served.append(http_server)
        server = types.SimpleNamespace(
            base_url=f"http://127.0.0.1:{http_server.server_address[1]}", proc=host.proc,
            wait_task=lambda task_id, timeout: {"status": "completed"}, wait_for_health=lambda timeout: True,
            current_sha=lambda: host.served_sha[:8])
        monkeypatch.setattr(scenarios, "_api", lambda *a, **k: host.run_the_task())

        def never_restarted():
            raise AssertionError("the adoption's own re-exec is the restart: the stand starts no second lifecycle")

        ctx = scenarios.LaneContext(
            server=server, clone=host.clone, data_root=host.data, oracle=_Oracle(host.data, host),
            harness=types.SimpleNamespace(wait_until=wait_until, wait_durable_result=lambda oracle, task_id, timeout: host.task_row),
            ui_resolver=lambda url: (_Page(host), ""), ui_reason="", shots=tmp_path / "shots", log=lambda _m: None,
            task_timeout=1, restart=never_restarted, stub=stub, ready_timeout=ready_timeout)
        return host, ctx

    yield build
    for http_server in served:
        http_server.shutdown()
        http_server.server_close()


ADOPTION_CHECKS = {"adoption_requested_for_reviewed_commit", "adoption_qualified_by_real_tests", "adoption_settled_and_restart_verified",
                   "adoption_recorded_on_task", "adoption_serves_commit", "adoption_serves_committed_stylesheets"}


def _failed(ctx) -> set:
    return {name for name, ok in ctx.checks.items() if not ok}


def _blue_checkout(root: pathlib.Path) -> pathlib.Path:
    """The executing checkout's release inputs with SM1's accent already in them: what a candidate holds once a real
    SM1 landed, and what this file reads when that candidate's own qualification runs it."""
    blue = root / "blue-input"
    _release_inputs(REPO, blue)
    css = blue / scenarios.SM1_CSS_PATH
    css.write_text(scenarios.css_with_accent(css.read_text(encoding="utf-8"), scenarios.SM1_NEW_ACCENT), encoding="utf-8")
    return blue


@pytest.mark.parametrize("fault,blue_input", [("", False), ("evolution_beside", False), ("unconfirmed_then_adopted", False), ("", True)])
def test_the_whole_chain_binds_one_commit_from_the_gate_to_the_served_bytes(lane, tmp_path, fault, blue_input):
    """Positive: every check, on the commit the gate's record names, with qualification recorded as it happened. An
    evolution cycle's own adoption beside it (another id, another commit) speaks for nothing here. A boot that left the
    handoff unconfirmed is no end: the next one's ``adopted`` settles it. The scene starts from its own accent, so a
    checkout that already serves SM1's blue still yields a real change."""
    host, ctx = lane(fault, source=_blue_checkout(tmp_path) if blue_input else REPO)
    scenarios.run_sm1(ctx)
    assert _failed(ctx) == set(), json.dumps(ctx.facts["direct_adoption"], indent=1)
    assert ADOPTION_CHECKS <= ctx.checks.keys()
    facts = ctx.facts["direct_adoption"]
    assert facts["reason"] == "adopted" and facts["gate"]["commit_sha"] == host.sha == ctx.facts["commit_sha"]
    assert facts["handoff"]["unconfirmed"] == (["native_host"] if fault == "unconfirmed_then_adopted" else []), "kept as a fact"
    assert ctx.facts["accent_before"] == SCENE_START_ACCENT != scenarios.SM1_NEW_ACCENT
    assert facts["gate"]["record_id"] == host.record_id and facts["gate"]["tree_sha"] == host.tree
    assert facts["candidate"]["reviewed_head"] == host.sha and facts["handoff"]["adoption_id"] == "sm1-adoption"
    assert facts["handoff"]["phases"] == ["authorized", "armed", "switched", "adopted"]
    assert facts["qualification"] == {"tier": "selected", "source": "receipt", "head": host.sha, "tree": host.tree, "key": "selected-key",
                                      "phase": "post_commit", "base": host.old, "changed": 3, "rule": "floor", "file_count": 2, "bound": True}
    assert facts["served"]["attestation"]["repo_head"] == host.sha and facts["serving"]["at_commit"] is True
    assert [row["route"] for row in facts["served"]["stylesheets"]] == list(scenarios.SM1_SERVED_STYLESHEETS)
    assert all(row["matches"] and row["status"] == 200 for row in facts["served"]["stylesheets"])

    def requested(adopt_commit: str) -> bool:   # the same facts, the request's own argument swapped
        return adoption.adoption_checks({**facts, "request": {**facts["request"], "adopt_commit": adopt_commit}})[
            "adoption_requested_for_reviewed_commit"]
    assert requested(host.sha) and requested(host.sha[:12]), "the tool resolves an abbreviation to the same commit"
    assert not any(map(requested, (OTHER_SHA, host.old, "HEAD", ""))), "a request for another commit or a ref"


def test_full_evidence_from_the_tasks_own_proof_qualifies_too(lane):
    """No ``source == proof`` gate and no ``selected`` promise: what the product accepted is recorded."""
    host, _ctx = lane()
    full = {"tier": "full", "source": "proof", "head": host.sha, "tree": host.tree}
    assert adoption.qualification_fact(full, host.sha, host.tree) == {**full, "bound": True}
    assert adoption.qualification_fact({**full, "tier": "skipped"}, host.sha, host.tree)["bound"] is False
    assert adoption.qualification_fact({}, host.sha, host.tree) == {"bound": False}


@pytest.mark.parametrize("fault,reason,failed", [
    ("handoff_other_commit", "timeout:authorization", {"adoption_requested_for_reviewed_commit", "adoption_qualified_by_real_tests",
                                                        "adoption_settled_and_restart_verified", "adoption_recorded_on_task"}),
    ("closed_other_id", "timeout:settlement", {"adoption_qualified_by_real_tests", "adoption_settled_and_restart_verified", "adoption_recorded_on_task"}),
    # Unconfirmed is no dead end (the next boot may settle it), so only the deadline names it.
    ("unconfirmed", "adoption_unconfirmed:native_host", {"adoption_qualified_by_real_tests", "adoption_settled_and_restart_verified",
                                                         "adoption_recorded_on_task"}),
])
def test_one_wrong_identity_fails_exactly_its_own_stage(lane, fault, reason, failed):
    host, ctx = lane(fault)
    scenarios.run_sm1(ctx)
    facts = ctx.facts["direct_adoption"]
    assert facts["reason"] == reason, facts
    not_served = {"adoption_serves_commit", "adoption_serves_committed_stylesheets"} if reason != "adopted" else set()
    ui = {"ui_app_palette", "ui_onboarding_palette", "ui_computed_style"} if "adoption_serves_commit" in failed | not_served else set()
    if fault == "checkout_not_moved":   # the unmoved checkout also serves the old stylesheet bytes
        not_served = {"adoption_serves_committed_stylesheets"}
    assert _failed(ctx) == failed | not_served | ui, json.dumps(facts, indent=1)


def test_a_stale_stylesheet_fails_the_byte_check_alone(lane):
    host, ctx = lane()
    host.served_body = host.git("show", f"{host.sha}:{scenarios.SM1_CSS_PATH}").encode("utf-8") + b"\n/* right palette, wrong bytes */"
    scenarios.run_sm1(ctx)
    assert _failed(ctx) == {"adoption_serves_committed_stylesheets"}


@pytest.mark.parametrize("fault", sorted(SHORT_EDITS))
def test_an_authors_edit_that_stops_short_of_the_palette_fails_the_chain_after_a_clean_adoption(lane, fault):
    """Everything up to the served bytes is right, so only the palette can say it: the real rehearsal reached exactly this
    with its dark-only edit. Its first ``--accent`` carried the target, and the source check said so while Light kept the old."""
    _host, ctx = lane(fault)
    scenarios.run_sm1(ctx)
    failed = _failed(ctx)
    assert failed - {"committed_css_carries_new_accent"} == {"ui_app_palette", "ui_onboarding_palette", "ui_computed_style"}, failed
    assert ("committed_css_carries_new_accent" in failed) is (fault != "accent-alone"), "the source reads both themes too"
    problems = ctx.facts["palette_problems"]
    assert ("--accent is #c93545" in problems["app"]["light"]) is (fault == "dark-only")
    for document in ("app", "onboarding"):
        for theme in scenarios.SM1_THEMES:
            assert "--accent-10 keeps its seed value" in problems[document][theme], "the ladder stayed in every case"


def test_a_serving_checkout_whose_status_git_cannot_read_is_not_clean(tmp_path):
    """A corrupt index: ``git status`` exits 128 and prints nothing, which read as no entries, so clean, while HEAD still
    resolves to the commit. An unknown worktree is no serving proof; the failure stays in the fact."""
    repo = tmp_path / "serving"
    repo.mkdir()
    (repo / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "serving")):
        git(repo, *args)
    sha = git(repo, "rev-parse", "HEAD")
    assert adoption._serving_fact(repo, sha)["clean"] is True, "the readable twin is clean"
    (repo / ".git" / "index").write_bytes(b"not an index")
    fact = adoption._serving_fact(repo, sha)
    assert fact["at_commit"] is True and fact["clean"] is False and "index" in fact["status_error"], fact


@pytest.mark.parametrize("fault,reason,refusal", [
    ("restart_refused", "restart_refused:ADOPTION_TESTS_REQUIRED", "ADOPTION_TESTS_REQUIRED"),
    ("no_request", "adoption_not_requested", None),
    ("restart_plain", "restart_without_adoption", ""),
    ("request_other_commit", "restart_for_other_commit", None),
    ("closed_abandoned", "adoption_closed:abandoned:not_armed", None),
    ("verify_other_sha", "restart_verify_failed", None),
    ("server_exited", "server_exited:1", None),
])
def test_a_dead_end_ends_the_wait_at_once_with_its_own_reason_and_the_commit_kept(lane, fault, reason, refusal):
    """Long budget, immediate answer: nothing waits out a refusal. A refused state stays its own fact, and only the
    served stage needs the adoption, so the commit's own checks keep their verdicts."""
    host, ctx = lane(fault, ready_timeout=60.0)
    started = time.monotonic()
    scenarios.run_sm1(ctx)
    assert time.monotonic() - started < 30, "a dead end must not wait out the deadline"
    facts = ctx.facts["direct_adoption"]
    assert facts["reason"] == reason, facts
    if refusal is not None:
        assert facts["request"]["refusal"] == refusal and facts["request"]["accepted"] is (fault == "restart_plain")
    assert ctx.checks["commit_landed"] and ctx.checks["committed_css_carries_new_accent"] and ctx.checks["adoption_serves_commit"] is False
    assert subprocess.run(["git", "cat-file", "-e", f"{host.sha}^{{commit}}"], cwd=str(host.clone)).returncode == 0
    assert facts["candidate"]["reviewed_head"] == host.sha, "the reviewed commit stays on its candidate"
    assert ctx.checks["ui_computed_style"] is False, "a served accent never stands in for the adoption"


@pytest.mark.parametrize("turns", [True, False], ids=["then_adopted", "never_adopted"])
def test_the_wait_ends_on_the_tasks_adopted_row_never_on_its_unconfirmed_one(lane, monkeypatch, turns):
    """An unconfirmed boot gives the task an ``unconfirmed`` row of the adoption (``body_adoption._waiting``); the boot
    that adopts archives ``adopted`` and verifies its restart before ``_publish_on_task`` replaces that row. The store
    here serves that stale row while the closure and the verification are already durable, then the final row: the old
    wait stopped at any row of the adoption and failed ``adoption_recorded_on_task`` on the stale one. A row that never
    turns keeps the deadline's own stage."""
    host, ctx = lane("unconfirmed_then_adopted")
    final, seen = host.task_row[body_adoption.ADOPTIONS_KEY][-1], []

    def task_result(_task_id):
        durable = (any(row.get("outcome") == "adopted" for row in ctx.oracle.supervisor_rows("body_adoption_closed"))
                   and any(row.get("ok") for row in ctx.oracle.events("restart_verify")))
        row = final if turns and seen else {**final, "outcome": "unconfirmed"}
        seen.append((row["outcome"], durable))
        return {**host.task_row, body_adoption.ADOPTIONS_KEY: [row]}

    monkeypatch.setattr(ctx.oracle, "task_result", task_result)
    scenarios.run_sm1(ctx)
    facts = ctx.facts["direct_adoption"]
    assert seen[0] == ("unconfirmed", True), "the stale row was read with the closure and the verification durable"
    if turns:
        assert seen == [("unconfirmed", True), ("adopted", True)] and facts["reason"] == "adopted", (seen, facts["reason"])
        assert _failed(ctx) == set() and facts["on_task"]["outcome"] == "adopted"
    else:
        assert {outcome for outcome, _ in seen} == {"unconfirmed"} and facts["reason"] == "timeout:task_record", (seen, facts["reason"])
        assert _failed(ctx) == {"adoption_recorded_on_task", "adoption_serves_commit", "adoption_serves_committed_stylesheets",
                                "ui_app_palette", "ui_onboarding_palette", "ui_computed_style"}


def test_an_exhausted_script_completes_but_never_counts(lane):
    host, ctx = lane()
    host.task_row["result"] = "Stub: no scripted step left for role probe."
    scenarios.run_sm1(ctx)
    assert ctx.checks["http_terminal_completed"] and ctx.checks["durable_terminal_completed"]
    assert _failed(ctx) == {"stub_scripted_final_reached"} and "no scripted step left" in ctx.facts["task_final_head"]


def test_the_paid_path_is_not_judged_by_the_scripts_final(lane):
    host, ctx = lane(stub=False)
    host.task_row["result"] = "The accent is changed, reviewed and its adoption is requested."
    scenarios.run_sm1(ctx)
    assert _failed(ctx) == set() and "stub_scripted_final_reached" not in ctx.checks


def test_the_scripted_author_adopts_exactly_what_the_gate_produced_then_finishes_without_waiting(lane):
    host, ctx = lane()
    script = scenarios.sm1_stub_script(host.clone, host.data)["agent"]
    assert [getattr(step, "__name__", None) or step.get("tool") for step in script[-3:]] == ["commit_reviewed", "step", "sm1_final_step"]
    adopt, final = script[-2:]
    landed = f"prompt\nOK: committed to candidate/{TASK}: m [not pushed]\nreview_record_id: {host.record_id}\n"
    assert adopt(landed) == {"tool": "request_restart", "arguments": {"reason": scenarios.SM1_RESTART_REASON, "adopt_commit": host.sha}}
    assert not any(key.startswith("skip_") for step in script if isinstance(step, dict) for key in step.get("arguments", {}))
    for refused in ("prompt\n⚠️ TESTS_PREFLIGHT_BLOCKED: tests must pass", "OK: committed to x: m\nreview_record_id: rl-20260101T000000-000000000000\n"):
        assert adopt(refused)["final"].startswith(scenarios.SM1_SCRIPT_ERROR)
    swapped = json.loads((host.data / "state" / "private_copies" / f"body_{TASK}.json").read_text(encoding="utf-8"))
    (host.data / "state" / "private_copies" / f"body_{TASK}.json").write_text(json.dumps({**swapped, "reviewed_commits": [OTHER_SHA]}), encoding="utf-8")
    assert "not a candidate's newest reviewed commit" in adopt(landed)["final"]
    assert final(landed + "Restart requested. It adopts candidate commit abc.") == {"final": scenarios.SM1_STUB_FINAL}
    blocked = final(landed + "⚠️ RESTART_BLOCKED: ADOPTION_TESTS_REQUIRED: no qualifying run")["final"]
    assert blocked.startswith(scenarios.SM1_SCRIPT_ERROR) and "ADOPTION_TESTS_REQUIRED" in blocked
    assert final(landed)["final"].startswith(scenarios.SM1_SCRIPT_ERROR), "no answer is no acceptance"
    assert final("Restart requested. an older one\n" + landed + "RESTART_UNAVAILABLE: request 1")["final"].startswith(scenarios.SM1_SCRIPT_ERROR)
    assert "machinery" in stub_lane.MACHINERY_ONLY and "not independent semantic review" in stub_lane.MACHINERY_ONLY


def test_the_paid_prompt_asks_for_the_exact_adoption_and_for_finishing_first():
    prompt = scenarios.sm1_prompt()
    assert "request_restart" in prompt and "adopt_commit" in prompt and "do not wait" in prompt
    assert "skip_" not in prompt.lower() and "bumped in the same diff" in prompt


def _red_or_passed_receipt(data: pathlib.Path, candidate: pathlib.Path, sha: str, result: str) -> None:
    tree = git(candidate, "rev-parse", f"{sha}^{{tree}}")
    identity = {"schema": preflight_receipts.RECEIPT_SCHEMA, "head": sha, "tree": tree, "index_tree": tree,
                "workload": "w", "installation": "i", "source": "s"}
    document = {"schema": preflight_receipts.RECEIPT_SCHEMA, "identity": identity, "result": result, "authenticity": "unsigned",
                "lanes": {"node": "absent", "parallel_workers": 2, "admitted_workers": 2, "serial": "passed"},
                "pass_seconds": {"parallel": 1, "serial": 1}, "phase": "post_commit", "task_id": "t",
                "created_at": "2026-11-02T00:00:00+00:00"}
    path = data / "state" / preflight_receipts.RECEIPTS_DIR / f"run-{sha[:12]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("evidence", ["absent", "red"])
def test_the_real_owner_refuses_a_commit_without_qualifying_tests_and_keeps_it_unadopted(tmp_path, monkeypatch, evidence):
    """A reviewed candidate commit whose post-commit run is missing or red: the real ``request_restart`` refuses with
    the typed code, the serving tree and the candidate stay as they are, no handoff exists, and the stand reads exactly
    that. The red record's flipped twin passes the same real reader, so the refusal is about the run's result."""
    from ouroboros.tools.registry import ToolRegistry

    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(preflight_qualification, "adoption_test_evidence", REAL_EVIDENCE)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    ctx = make_ctx(serving, data, "root-sm1")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    old = git(serving, "rev-parse", "HEAD")
    sha = candidate_commit(candidate, files={"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    body_candidate.record_reviewed_commit(ctx, sha)
    if evidence == "red":
        _red_or_passed_receipt(data, candidate, sha, "passed")
        assert preflight_qualification.receipt_for_commit(data, candidate, sha)["tier"] == "full", "the reader's positive control"
        _red_or_passed_receipt(data, candidate, sha, "failed")
    registry = ToolRegistry(repo_dir=serving, drive_root=data)
    registry.set_context(ctx)

    text = registry.execute("request_restart", {"reason": scenarios.SM1_RESTART_REASON, "adopt_commit": sha})

    assert "RESTART_BLOCKED: ADOPTION_TESTS_REQUIRED" in text, text
    assert body_adoption.read(data) == {} and git(serving, "rev-parse", "HEAD") == old
    assert git(candidate, "merge-base", "--is-ancestor", sha, "HEAD") == "" and git(candidate, "rev-parse", "HEAD") == sha
    fact = adoption.restart_request_fact([{"tool": "request_restart", "result_preview": text, "args": {"adopt_commit": sha}}])
    assert (fact["accepted"], fact["refusal"]) == (False, "ADOPTION_TESTS_REQUIRED")
    assert scenarios.sm1_final_step(f"OK: committed to c: m\n{text}")["final"].startswith(scenarios.SM1_SCRIPT_ERROR)


EXACT, UNKNOWN, LATE_ACK = "<the reviewed commit>", "0" * 40, "<the host's acceptance of the request it holds arrives>"
ADOPT, PLAIN, REASON_ONLY = {"reason": "adopt reviewed fix", "adopt_commit": EXACT}, {"reason": "just restart"}, {"reason": "a different reason"}
# ``(task attempt, request_restart arguments)`` in call order, and whether the request the host then holds adopts EXACT.
# With LATE_ACK the host answers no intent itself (a lost ACK), and accepts the held request only at that step.
REQUESTS = {
    "adopt_then_reason_only_retry": ([(1, ADOPT), (1, REASON_ONLY)], True),
    "unconfirmed_adopt_then_late_ack_then_reason_only_retry": ([(1, ADOPT), LATE_ACK, (1, REASON_ONLY)], True),
    "refused_then_adopt": ([(1, {"reason": "adopt", "adopt_commit": UNKNOWN}), (1, ADOPT)], True),
    "plain_then_adopt_retry": ([(1, PLAIN), (1, ADOPT)], False),
    "plain_then_next_attempt_refused_then_adopts": ([(1, PLAIN), (2, {"reason": "adopt", "adopt_commit": UNKNOWN}), (2, ADOPT)], True),
    "adopt_then_next_attempt_plain": ([(1, ADOPT), (2, PLAIN)], False),
}


def test_the_request_is_the_one_the_host_holds_from_the_rows_the_real_loop_wrote(subtests, tmp_path, monkeypatch):
    """``request_restart`` through the real loop: the stand reads the rows the product wrote. In one task attempt the
    first call the host answers sends the request and every later call only reads its receipt, an acceptance or a
    ``RESTART_<STATUS>: request <id>`` (``control_runtime._request_restart``; a refusal before anything was sent leaves
    the attempt free); a new attempt holds its own (``owner_wait``). Ground truth is the product's: the restart the task's
    end sends binds the handoff only when the request held adopts that commit (``body_adoption.bind_restart``). Reading
    the newest call's arguments turned an exact adoption followed by a reason-only retry into
    ``restart_without_adoption``, and an ordinary restart followed by an ignored ``adopt_commit`` into a wait for an
    adoption that never comes."""
    case_paths = case_folders(tmp_path)
    for calls, adopts in REQUESTS.values():
        tmp_path = next(case_paths)
        with subtests.test(calls=calls, adopts=adopts), monkeypatch.context() as monkeypatch:
            from ouroboros import loop_tool_execution
            from ouroboros.tools import control_events
            from ouroboros.tools.registry import ToolRegistry
            from tests._shared import write_queue_document
            from tests.body_candidate_support import host_restart_ack

            serving = make_serving(tmp_path)
            data = isolate(monkeypatch, tmp_path)
            monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
            monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
            monkeypatch.setattr(control_events, "_PROMOTE_CONFIRM_TIMEOUT_SEC", 0)   # a silent host is a lost ACK at once
            ctx = make_ctx(serving, data, "root-sm1")
            body_candidate.prepare(ctx)
            sha = candidate_commit(pathlib.Path(ctx.repo_dir), files={"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
            body_candidate.record_reviewed_commit(ctx, sha)
            old = git(serving, "rev-parse", "HEAD")
            sent = host_restart_ack(ctx, data, status=None if LATE_ACK in calls else "accepted")
            registry = ToolRegistry(repo_dir=serving, drive_root=data)
            registry.set_context(ctx)
            requests = [step for step in calls if step != LATE_ACK]
            for index, step in enumerate(calls):
                if step == LATE_ACK:   # what ``host_restart_ack`` publishes, for the one request already held
                    held = {key: ctx.pending_restart_request[key] for key in ("request_id", "task_id", "task_attempt", "reason")}
                    write_queue_document(data / "state" / "queue_snapshot.json", {"restart_intent": {**held, "status": "accepted"}})
                    continue
                attempt, arguments = step
                ctx.task_attempt = attempt
                arguments = {key: sha if value == EXACT else value for key, value in arguments.items()}
                loop_tool_execution._execute_single_tool(registry, {"id": f"restart-{index}", "function": {
                    "name": "request_restart", "arguments": json.dumps(arguments)}}, data / "logs", "root-sm1")
            rows = scenarios.tool_result_rows([json.loads(line) for line in (data / "logs" / "tools.jsonl").read_text(encoding="utf-8").splitlines()])
            assert [row["tool"] for row in rows] == ["request_restart"] * len(requests), rows
            assert len([event for event in sent if event.get("type") == "restart_request"]) == len({a for a, _ in requests}), "one request per attempt"
            if LATE_ACK in calls:   # the receipt branch, as the loop wrote it: the request's own row, then a replay's
                assert "RESTART_UNCONFIRMED: request " in rows[0]["result_preview"], rows[0]
                assert rows[-1]["result_preview"].startswith("Restart requested. The previously acknowledged request is unchanged."), rows[-1]

            body_adoption.bind_restart(lambda **_kw: (True, "ok"), data, ctx.pending_restart_reason)(
                reason="agent_restart_request", unsynced_policy="rescue_and_block")
            handoff = body_adoption.read(data)
            assert (handoff.get("cand") == sha and handoff.get("restart_bound") is True) is adopts, handoff

            fact = adoption.restart_request_fact(rows)
            lane_view = types.SimpleNamespace(data_root=data, clone=serving, oracle=ArtifactOracle(data), server=None)
            facts = adoption.observe_direct_adoption(lane_view, "root-sm1", rows, {"commit_sha": sha}, timeout=0, stylesheet="VERSION",
                                                     routes=(), poll=0)
            assert facts["reason"] == ("timeout:settlement" if adopts else "restart_without_adoption"), facts
            assert adoption.adoption_checks(facts)["adoption_requested_for_reviewed_commit"] is adopts
            assert fact["accepted"] is True and adoption.names_commit(fact["adopt_commit"], sha) is adopts, fact
            assert fact["adopt_commit"] in {"", sha} and fact["task_attempt"] == requests[-1][0], fact
            assert git(serving, "rev-parse", "HEAD") == old, "nothing here restarts or adopts"
