"""Task-policy and custody contracts of the current (ordinary-task) smoke adapter.

The old direct marketplace three-wave/thread-drain helper is retired. Its
unsettled-wave prohibition now has an actual-consumer witness in
system_e2e/test_skill_smoke_author; server lifetime, not a pytest-owned review thread,
owns the worker. There is deliberately no provider retry/reset after ambiguity.
"""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import json
import runpy
import shutil

import pytest

from ouroboros.tools import ToolContext, ToolRegistry
from tests.skill_smoke_author import AUTHOR_TOOLS, REVIEW_CYCLES, author_policy, author_settings
import tests.test_skill_smoke_official as smoke


def test_effective_catalog_and_dispatch_withhold_mutation_execution_and_delegation(tmp_path):
    registry = ToolRegistry(Path(__file__).resolve().parents[1], tmp_path)
    disabled = author_policy(registry._ctx.repo_dir, tmp_path)
    context = ToolContext(repo_dir=registry._ctx.repo_dir, drive_root=tmp_path)
    context.task_id = "ci-author-policy"
    context.task_metadata = {"task_contract": {"disabled_tools": disabled}}
    registry.set_context(context)
    names = {schema["function"]["name"] for schema in registry.schemas()}
    assert names == AUTHOR_TOOLS
    from ouroboros.tools.registry_guards import _capability_resource_guard_result
    for name in ("write_file", "run_command", "run_script", "toggle_skill", "skill_exec",
                 "delegate_start", "schedule_subagent", "start_task", "switch_model",
                 "skill_owner_action", "configure_presence", "verify_and_record"):
        result = _capability_resource_guard_result(context, name, {})
        assert result is not None and result.status == "blocked", name


def test_two_paid_cycles_and_finite_author_window_are_the_live_consumers_budget():
    settings = author_settings(smoke._REVIEW_MODEL)
    assert REVIEW_CYCLES == smoke._REVIEW_WAVES == settings["OUROBOROS_REVIEW_MAX_CYCLES"] == 2
    assert settings["OUROBOROS_RUNTIME_MODE"] == "advanced"
    assert settings["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
    assert smoke._REVIEW_MODEL == "deepseek/deepseek-v4.1-flash"
    assert settings["OUROBOROS_MODEL"] == settings["OUROBOROS_MODEL_LIGHT"] == smoke._REVIEW_MODEL
    roster = json.loads(settings["OUROBOROS_SUBAGENTS"])
    assert roster["enabled"] is True and len(roster["items"]) == 1
    reviewer = roster["items"][0]
    assert reviewer["route"] == {"kind": "api_model", "target_id": smoke._REVIEW_MODEL}
    assert reviewer["effort"] == "low" and reviewer["delivery"] == "native" and reviewer["review_eligible"] is True
    assert settings["OUROBOROS_EFFORT_MIN"] == settings["OUROBOROS_EFFORT_TASK"] == settings["OUROBOROS_EFFORT_MAX"] == "low"
    assert settings["TOTAL_BUDGET"] == settings["OUROBOROS_PER_TASK_COST_USD"] == 10
    assert smoke._REVIEW_AUTHOR_TIMEOUT_SEC > 2 * smoke._REVIEW_WINDOW_SEC
    assert smoke._REVIEW_TEST_TIMEOUT_SEC > smoke._REVIEW_AUTHOR_TIMEOUT_SEC


def test_no_import_oracle_detects_import_even_when_loader_catches_exception(tmp_path):
    from tests.system_e2e.test_skill_smoke_author import assert_not_imported
    plugin = tmp_path / "plugin.py"
    plugin.write_text("from pathlib import Path\nPath(__file__).with_name('imported.marker').write_text('yes')\nraise RuntimeError('caught by loader')\n")
    assert_not_imported(tmp_path)
    with pytest.raises(RuntimeError, match="caught by loader"):
        runpy.run_path(str(plugin))
    with pytest.raises(AssertionError, match="payload was imported"):
        assert_not_imported(tmp_path)


@pytest.mark.parametrize("start_error", [False, True])
@pytest.mark.parametrize("export_error", ["none", "write", "parse"])
def test_failed_tree_reap_retains_candidate_data_and_blocks_ci_enable(tmp_path, monkeypatch, capsys, start_error, export_error):
    """No sockets/server: the actual CI entry must not reach enable after failed teardown."""
    import tests.skill_smoke_author as author
    import tests.candidate_checkout as checkouts
    from ouroboros.skill_loader import load_enabled
    source = tmp_path / "source"
    payload = source / "skills/ouroboroshub/probe"
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text("---\nname: probe\ntype: extension\nentry: plugin.py\n---\n")
    (payload / "plugin.py").write_text("def register(api): pass\n")
    candidate = checkouts.CandidateCheckout(None, tmp_path / "repo")
    candidate.path.mkdir()

    @contextmanager
    def checkout(*_args, **_kwargs):
        yield candidate

    class Container:
        def __init__(self, **kwargs):
            assert kwargs == {"lifeline": True, "overload": True}

        def spawn(self, *_args, **_kwargs):
            return SimpleNamespace(wait=lambda **_kw: 0)

        def reap(self):
            return "unconfirmed descendant"

        def close(self):
            pass

    def initialize(self, clone, data, settings, **_kwargs):
        self.clone, self.data_root, self.settings_path = clone, data, settings
        self.proc, self.base_url = None, "unused"

    def ready(self, _timeout):
        if start_error:
            raise ValueError("original readiness failure")

    monkeypatch.setattr(checkouts, "candidate_checkout", checkout)
    monkeypatch.setattr(author, "verify_checkout", lambda *_args: None)
    monkeypatch.setattr(author, "require_candidate_interpreter", lambda *_args: None)
    monkeypatch.setattr(author, "ProcessContainer", Container)
    monkeypatch.setattr(author.IsolatedServer, "__init__", initialize)
    monkeypatch.setattr(author.IsolatedServer, "stop", lambda _self: None)
    monkeypatch.setattr(author.SmokeAuthorServer, "_wait_ready", ready)
    monkeypatch.setattr(author.SmokeAuthorServer, "_read_authoritative_settings", lambda _self: {})
    monkeypatch.setattr(author.SmokeAuthorServer, "_patch_settings_ports", lambda _self: None)
    monkeypatch.setattr(author.SmokeAuthorServer, "_env", lambda _self: {})
    monkeypatch.setattr(author.SmokeAuthorServer, "wait_task", lambda *_args, **_kw: {"status": "completed"})
    monkeypatch.setattr(author, "_api", lambda *_args, **_kw: {"task_id": "probe-task"})
    monkeypatch.setattr(author, "seed_owner_state", lambda *_args: None)
    monkeypatch.setattr(author, "author_policy", lambda *_args: [])
    if export_error == "write":
        def fail_write(*_args):
            raise OSError("synthetic export failure")
        monkeypatch.setattr(author, "write_record", fail_write)
    if export_error == "parse":
        logs = tmp_path / "author/data/logs"
        def fail_parse(self, _timeout):
            logs.mkdir(parents=True, exist_ok=True)
            (logs / "tools.jsonl").write_text("malformed JSON")
            ready(self, _timeout)
        monkeypatch.setattr(author.SmokeAuthorServer, "_wait_ready", fail_parse)
    monkeypatch.setattr(smoke, "DATA_DIR", source)
    monkeypatch.setenv("OPENROUTER_API_KEY", "stub-not-a-credential")
    run = tmp_path / "run"  # stands in for the whole battery's scratch root
    run.mkdir()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(run))
    enabled = []
    monkeypatch.setattr(smoke, "save_enabled", lambda *args: enabled.append(args))
    try:
        with pytest.raises(checkouts.CandidateError, match="unconfirmed descendant") as raised:
            smoke.test_review_grants_and_enable("probe", None, lambda _slug: None, tmp_path)
        assert candidate.unproven == 1 and candidate.path.exists()
        assert (tmp_path / "author/data").exists()
        assert not enabled and not load_enabled(tmp_path / "author/data", "probe")
        if export_error != "write":
            evidence = json.loads((tmp_path / "author/author-evidence.json").read_text())
            assert evidence["server_stopped"] is False
            assert "unconfirmed descendant" in evidence["teardown_failure"]
            if export_error == "parse":
                assert "author_tool_calls" in evidence["evidence_errors"][0]
        else:
            assert "SKILL_SMOKE_EVIDENCE_INCOMPLETE" in capsys.readouterr().err
        # A failed export keeps the author's own tree, never the run around it (#1908).
        assert (tmp_path / "author/OUROBOROS_RETAINED.txt").is_file() is (export_error != "none")
        assert not (run / "OUROBOROS_RETAINED.txt").exists()
        if start_error:
            assert isinstance(raised.value.__cause__, ValueError)
            assert str(raised.value.__cause__) == "original readiness failure"
        checkouts._retire_checkout(candidate.path, candidate, True)
        assert candidate.path.exists(), "unproven checkout was deleted"
    finally:
        # The reap and the export faults are stubs: no process ever ran from this tree.
        from tests._shared import retire_synthetic_retention
        try:
            json.loads("malformed JSON")
        except ValueError as exc:
            parse = f"author_tool_calls: {type(exc).__name__}: {exc}"
        faults = ("local: OSError: synthetic export failure", "export: OSError: synthetic export failure", parse)
        retire_synthetic_retention(candidate.path, "1 process tree(s) started from this checkout were never proven gone")
        retire_synthetic_retention(tmp_path / "author/data", "CANDIDATE_RETAINED: tree reap: unconfirmed descendant")
        retire_synthetic_retention(tmp_path / "author", *(
            f"test diagnostics incomplete: SKILL_SMOKE_EVIDENCE_INCOMPLETE: {fault}" for fault in faults))


def test_export_keeps_full_judgments_but_not_credentials_or_private_runtime(tmp_path, monkeypatch):
    from tests.skill_smoke_author import persist_author_evidence
    root = tmp_path / "scratch/author"
    root.mkdir(parents=True)
    destination = tmp_path / "durable"
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(destination))
    credential = "synthetic-credential-not-a-token-pattern"
    judgment = "Complete reviewer judgment " + "evidence " * 2000
    actor = {"raw_text": judgment + credential, "parsed_items": [{"reason": judgment}],
             "response_ref": "/private/runtime/ref.json", "operation_state": "settled"}
    evidence = {"task_id": "task", "original_hash": "hash", "server_stopped": True,
                "author_answer": "Grounded argument " + credential, "author_trace": "skill_review(rebuttal)",
                "review_history": [{"status": "blockers", "paid": True, "raw_actor_records": [actor]},
                                   {"status": "clean", "paid": True, "rebuttal_sha256": "arg", "raw_actor_records": [actor]}],
                "review": {"raw_result": judgment, "raw_actor_records": [actor], "findings": [{"reason": judgment}]},
                "settings": {"OPENROUTER_API_KEY": credential}, "author_tool_calls": [
                    {"tool": "skill_review", "args": {"skill": "probe", "review_rebuttal": judgment + credential}}]}
    assert not persist_author_evidence(root, evidence, credential)
    shutil.rmtree(root.parent)  # only this test's disposable input
    files = list(destination.glob("skill-smoke-author/*/author-evidence.json"))
    assert len(files) == 1 and not root.exists()
    text = files[0].read_text()
    exported = json.loads(text)
    assert credential not in text and "/private/runtime" not in text
    assert "settings" not in exported and "author_tool_calls" not in exported
    assert exported["review"]["raw_result"] == judgment
    assert exported["review_history"][0]["raw_actor_records"][0]["parsed_items"][0]["reason"] == judgment
    assert exported["review_history"][1]["rebuttal_sha256"] == "arg"
    assert exported["review_calls"][0]["arguments"]["review_rebuttal"] == judgment + "***"


@pytest.mark.parametrize("export_error", [False, True])
def test_author_entry_summary_uses_paid_verdict_and_failed_export_stays_red(tmp_path, monkeypatch, capsys, export_error):
    import tests.skill_smoke_author as author
    payload = tmp_path / "source/skills/ouroboroshub/probe"
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text("---\nname: probe\ntype: instruction\n---\nProbe text\n")
    server = SimpleNamespace(start=lambda: None, stop=lambda: None, base_url="unused",
                             wait_task=lambda *_args, **_kw: {"status": "completed", "result": "author reasoning"})
    monkeypatch.setattr(author, "SmokeAuthorServer", lambda *_args, **_kw: server)
    monkeypatch.setattr(author, "_api", lambda *_args, **_kw: {"task_id": "task"})
    monkeypatch.setattr(author, "author_policy", lambda *_args: [])
    monkeypatch.setattr(author, "seed_owner_state", lambda *_args: None)
    monkeypatch.setattr(author, "load_review_waves", lambda *_args, **_kw: [{"paid": True, "status": "warnings"}])
    gate = []
    def assert_gate(*_args):
        gate.append(True)
        return {"status": "warnings"}
    monkeypatch.setattr(author, "assert_review_evidence", assert_gate)
    if export_error:
        def fail_write(*_args):
            raise OSError("synthetic export failure")
        monkeypatch.setattr(author, "write_record", fail_write)
    args = dict(candidate=SimpleNamespace(path=tmp_path / "repo"), source_data=tmp_path / "source",
                root=tmp_path / "author", name="probe", model="stub", timeout_sec=1,
                credential=("OPENROUTER_API_KEY", "stub-not-a-credential"))
    run = tmp_path / "run"  # stands in for the whole battery's scratch root
    run.mkdir()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(run))
    try:
        if export_error:
            with pytest.raises(AssertionError, match="SKILL_SMOKE_EVIDENCE_INCOMPLETE"):
                author.run_author_review(**args)
            assert not gate
        else:
            assert author.run_author_review(**args)["review"]["status"] == "warnings"
            assert gate == [True]
            assert (tmp_path / "author/author-evidence.json").exists()
        assert "final=warnings" in capsys.readouterr().out
        assert (tmp_path / "author/OUROBOROS_RETAINED.txt").is_file() is export_error
        assert not (run / "OUROBOROS_RETAINED.txt").exists()
    finally:
        from tests._shared import retire_synthetic_retention
        retire_synthetic_retention(tmp_path / "author", *(  # the injected export fault, nothing else
            f"test diagnostics incomplete: SKILL_SMOKE_EVIDENCE_INCOMPLETE: {stage}: OSError: synthetic export failure"
            for stage in ("local", "export")))


@pytest.mark.parametrize("source_missing", [False, True])
def test_actual_dispatch_source_preserves_long_rebuttal_before_scratch_cleanup(tmp_path, monkeypatch, source_missing):
    import tests.skill_smoke_author as author
    from ouroboros.tool_call_log import new_invocation, persist_dispatch_source, invocation_fields
    from ouroboros.utils import sanitize_tool_args_for_log
    source = tmp_path / "source"
    payload = source / "skills/ouroboroshub/probe"
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text("---\nname: probe\ntype: instruction\n---\nProbe\n")
    root = tmp_path / "author"
    destination = tmp_path / "durable"
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(destination))
    argument = "Exact evidence, including the final qualification. " * 100
    args = {"skill": "probe", "review_rebuttal": argument}
    preview = sanitize_tool_args_for_log("skill_review", args)
    assert preview["review_rebuttal"].startswith("<TRUNCATED:")

    def finish(*_args, **_kwargs):
        data = root / "data"
        logs = data / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        invocation = new_invocation("call", {}, 1)
        persist_dispatch_source({}, logs, "task", invocation, "skill_review", args)
        assert invocation["args_source_status"] == "ready"
        if source_missing:
            shutil.rmtree(data / "observability")
        row = {"type": "tool_call", "task_id": "task", "tool": "skill_review",
               "status": "ok", "args": preview, **invocation_fields(invocation)}
        (logs / "tools.jsonl").write_text(json.dumps(row) + "\n")
        return {"status": "completed", "result": "Finished"}

    server = SimpleNamespace(start=lambda: None, stop=lambda: None, base_url="unused", wait_task=finish)
    monkeypatch.setattr(author, "SmokeAuthorServer", lambda *_args, **_kw: server)
    monkeypatch.setattr(author, "_api", lambda *_args, **_kw: {"task_id": "task"})
    monkeypatch.setattr(author, "author_policy", lambda *_args: [])
    monkeypatch.setattr(author, "seed_owner_state", lambda *_args: None)
    monkeypatch.setattr(author, "load_review_waves", lambda *_args, **_kw: [{"paid": True, "status": "clean"}])
    gate = []
    monkeypatch.setattr(author, "assert_review_evidence", lambda *_args: gate.append(True) or {"status": "clean"})
    kwargs = dict(candidate=SimpleNamespace(path=tmp_path / "repo"), source_data=source,
                  root=root, name="probe", model="stub", timeout_sec=1,
                  credential=("OPENROUTER_API_KEY", "stub-not-a-credential"))
    if source_missing:
        with pytest.raises(AssertionError, match="SKILL_SMOKE_EVIDENCE_INCOMPLETE"):
            author.run_author_review(**kwargs)
        assert not gate
    else:
        author.run_author_review(**kwargs)
        assert gate == [True]
    shutil.rmtree(root)
    exported = json.loads(next(destination.glob("skill-smoke-author/*/author-evidence.json")).read_text())
    if source_missing:
        assert exported["evidence_errors"]
    else:
        assert exported["review_calls"][0]["arguments"]["review_rebuttal"] == argument
