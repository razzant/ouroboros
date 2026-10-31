"""#1908 producer contract: a declared diagnostic destination outlives the scratch release.

A producer states where its own diagnostics go before the work starts. The launcher's shared
channel keeps that destination outside the disposable scratch its owner releases on green AND
red; an explicit operator destination stays exactly where it was exported. These are contract
and real-producer checks; the browser lane proves the same seam end to end under safe_test.
"""
import pytest

from ouroboros.test_environment import DIAGNOSTIC_OUTPUT_KEYS
from tests._shared import producer_evidence_dir


@pytest.mark.serial
def test_managed_launchers_preserve_explicit_producer_destinations(tmp_path, monkeypatch):
    """Exercise the actual scrub/launcher, not just a producer in the parent env."""
    import json
    import sys
    from ouroboros.preflight_runner import _preflight_env
    from ouroboros.test_evidence import diagnostic_directory
    from ouroboros.test_environment import DIAGNOSTIC_OUTPUT_KEYS
    from tests.test_test_storage import REPO, _safe

    chosen = diagnostic_directory(tmp_path / "selected", "explicit-output-contract")
    selected = {key: str(chosen / key) for key in DIAGNOSTIC_OUTPUT_KEYS}
    for key, path in selected.items():
        monkeypatch.setenv(key, path)
    monkeypatch.setenv("OUROBOROS_MODEL", "must-not-cross-test-boundary")
    env = _preflight_env(tmp_path / "native", REPO)
    assert {key: env[key] for key in selected} == selected
    assert "OUROBOROS_MODEL" not in env
    script = ("import json,os; from pathlib import Path; "
              f"keys={list(selected)!r}; "
              "out={key:os.environ[key] for key in keys}; "
              "assert 'OUROBOROS_MODEL' not in os.environ; print(json.dumps(out))")
    result, evidence = _safe(tmp_path, [sys.executable, "-c", script])
    assert result.returncode == 0, result.stderr
    raw = next(evidence.glob("command-*/stdout.bin")).read_text(encoding="utf-8")
    assert json.loads(raw) == selected


@pytest.mark.parametrize("key", DIAGNOSTIC_OUTPUT_KEYS)
def test_native_explicit_output_in_scratch_retains_it(tmp_path, monkeypatch, key):
    from ouroboros.preflight_runner import _preflight_env
    from ouroboros.test_environment import RETENTION_MARKER
    root = tmp_path / "native"
    monkeypatch.setenv(key, str(root / "chosen"))
    _preflight_env(root, tmp_path)
    marker = root / RETENTION_MARKER
    assert key in marker.read_text(encoding="utf-8")
    # This test spawned no process and wrote no diagnostic; retire its synthetic marker.
    marker.unlink()


@pytest.mark.parametrize("fault", ["injected", "real"])
def test_failed_export_keeps_the_producers_tree_not_the_whole_run(tmp_path, monkeypatch, fault):
    """#1908: a test's injected export fault never holds the battery; a real one still keeps it."""
    import tests.skill_smoke_author as author
    from ouroboros.safe_test_supervisor import release_scratch
    from ouroboros.test_environment import RETENTION_MARKER, declare_scratch
    from tests._shared import retire_synthetic_retention

    run = tmp_path / "run"  # the launcher's root of a whole battery
    (run / "b0").mkdir(parents=True)
    declare_scratch(run / "b0", "pytest-basetemp")
    producer = run / "b0" / "t" / "author"  # one test's own tree
    producer.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(run))
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(tmp_path / "evidence"))
    if fault == "injected":
        def fail(*_args):
            raise OSError("synthetic export failure")
        monkeypatch.setattr(author, "write_record", fail)
    else:  # the durable destination is a file, so the export cannot be created
        (tmp_path / "evidence").write_text("not a directory", encoding="utf-8")
    errors = author.persist_author_evidence(producer, {"task_id": "probe"}, "credential")
    reasons = [f"test diagnostics incomplete: {error}" for error in errors]
    try:
        assert errors and (producer / RETENTION_MARKER).read_text(encoding="utf-8").splitlines() == reasons
        assert not (run / RETENTION_MARKER).exists(), "the producer kept the whole run"
        if fault == "injected":
            # What the test that injected it does; nothing else may be left holding the run.
            assert retire_synthetic_retention(producer, *reasons)
        released = release_scratch(run)
        assert released["retained"] == ([] if fault == "injected" else [str(run)]), released
        assert run.exists() is (fault == "real")
    finally:
        retire_synthetic_retention(producer, *reasons)  # this test is the real producer's only consumer


def test_a_producer_whose_own_tree_cannot_be_marked_keeps_the_run(tmp_path, monkeypatch):
    from ouroboros.test_environment import RETENTION_MARKER
    from ouroboros.test_evidence import diagnostics_incomplete

    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(run))
    diagnostics_incomplete("export: OSError: disk full", tmp_path / "vanished")
    marker = run / RETENTION_MARKER
    assert marker.read_text(encoding="utf-8") == "test diagnostics incomplete: export: OSError: disk full\n"
    marker.unlink()  # this test's own synthetic run root


def test_explicit_destination_wins_over_the_shared_channel(tmp_path, monkeypatch):
    chosen = (tmp_path / "chosen").resolve()
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(tmp_path / "channel"))
    monkeypatch.setenv("FAMILY_EVIDENCE_DIR", str(chosen))
    assert producer_evidence_dir(tmp_path / "fallback", "family", env="FAMILY_EVIDENCE_DIR") == chosen


def test_channel_keeps_a_producer_destination_outside_released_scratch(tmp_path, monkeypatch):
    scratch, evidence = (tmp_path / "scratch").resolve(), (tmp_path / "evidence").resolve()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(scratch))
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(evidence))
    destination = producer_evidence_dir(scratch / "tmp-path", "family", env="FAMILY_EVIDENCE_DIR")
    assert destination.is_relative_to(evidence)
    assert not destination.is_relative_to(scratch)


def test_one_invocation_shares_a_destination_and_workers_do_not(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(tmp_path / "evidence"))
    first = producer_evidence_dir(tmp_path / "worker0/test0", "family")
    assert first == producer_evidence_dir(tmp_path / "worker0/test0", "family")
    assert first != producer_evidence_dir(tmp_path / "worker1/test0", "family")


def test_bare_pytest_keeps_the_producers_own_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("OUROBOROS_TEST_EVIDENCE_ROOT", raising=False)
    monkeypatch.delenv("FAMILY_EVIDENCE_DIR", raising=False)
    destination = producer_evidence_dir(tmp_path / "fallback", "family", env="FAMILY_EVIDENCE_DIR")
    assert destination.name == "fallback" and destination.is_dir()


def test_real_project_card_producer_resolves_outside_scratch(tmp_path, monkeypatch):
    from tests.test_project_work_card_browser import _evidence_dir

    scratch, evidence = (tmp_path / "scratch").resolve(), (tmp_path / "evidence").resolve()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(scratch))
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(evidence))
    monkeypatch.delenv("OUROBOROS_BROWSER_EVIDENCE_OUT", raising=False)
    assert _evidence_dir(scratch / "tmp-path").is_relative_to(evidence)


def test_real_history_producer_resolves_outside_scratch(tmp_path, monkeypatch):
    pytest.importorskip("playwright.sync_api")
    from tests.test_chat_history_paging_browser import _evidence_dir

    scratch, evidence = (tmp_path / "scratch").resolve(), (tmp_path / "evidence").resolve()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(scratch))
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(evidence))
    monkeypatch.delenv("HISTORY_UI_EVIDENCE_DIR", raising=False)
    assert _evidence_dir(scratch / "tmp-path").is_relative_to(evidence)


@pytest.mark.parametrize("family", ["main", "quiz", "skill", "quiz-response"])
def test_main_and_quiz_producers_keep_the_shared_and_explicit_destinations(tmp_path, monkeypatch, family):
    from tests.test_main_consumer_journeys_browser import _controls_evidence
    from tests.test_ui_quiz_comment_only_browser import _evidence_dir
    from tests.test_ui_smoke_skill_lifecycle import _evidence_dir as skill_evidence
    from tests.test_quiz_response_browser import _evidence_dir as response_evidence

    def destination():
        if family == "main":
            return _controls_evidence(tmp_path / "scratch", "stage")
        if family == "quiz-response":
            return response_evidence(tmp_path / "scratch")
        return (skill_evidence if family == "skill" else _evidence_dir)(tmp_path / "scratch")

    key = "OUROBOROS_UI_SCREENSHOT_DIR" if family == "skill" else "OUROBOROS_BROWSER_EVIDENCE_OUT"
    if family == "quiz-response":
        key = "QUIZ_BROWSER_EVIDENCE"
    monkeypatch.delenv(key, raising=False)
    shared = tmp_path / "shared"
    monkeypatch.setenv("OUROBOROS_TEST_EVIDENCE_ROOT", str(shared))
    path = destination()
    assert path.is_relative_to(shared) and path == destination()
    explicit = tmp_path / "selected"
    monkeypatch.setenv(key, str(explicit))
    assert destination() == (explicit / "stage" if family == "main" else explicit)


@pytest.mark.serial
def test_managed_launchers_do_not_enable_optional_outputs(tmp_path, monkeypatch):
    import json
    import sys
    from ouroboros.preflight_runner import _preflight_env
    from tests.test_test_storage import REPO, _safe

    optional = set(DIAGNOSTIC_OUTPUT_KEYS) - {"OUROBOROS_UI_EVIDENCE_DIR"}
    for key in DIAGNOSTIC_OUTPUT_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert optional.isdisjoint(_preflight_env(tmp_path / "native", REPO))
    script = ("import json,os; "
              f"keys={sorted(optional)!r}; "
              "print(json.dumps([key for key in keys if key in os.environ]))")
    result, evidence = _safe(tmp_path, [sys.executable, "-c", script])
    assert result.returncode == 0, result.stderr
    raw = next(evidence.glob("command-*/stdout.bin")).read_text(encoding="utf-8")
    assert json.loads(raw) == []
