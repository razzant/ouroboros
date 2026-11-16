"""The stability tests' evidence (``tests/_stability_evidence.py``) follows the test-evidence contract: a run publishes
its facts and launcher output, while the copy of the server's data root, a runtime archive, stays with the run."""
from __future__ import annotations

import json

from ouroboros.test_evidence import EVIDENCE_ENV, run_diagnostics
from tests._stability_evidence import retain_server_evidence


def test_published_diagnostics_carry_facts_and_launcher_output_but_never_the_data_root(tmp_path, monkeypatch):
    run = tmp_path / "evidence-run"
    run.mkdir()
    monkeypatch.setenv(EVIDENCE_ENV, str(run))
    data = tmp_path / "server" / "data"
    (data / "state").mkdir(parents=True)
    (data / "state" / "tasks.json").write_text('{"exact": "model and tool sources"}', encoding="utf-8")
    retain_server_evidence(data, {"teardown": "confirmed"}, b"launcher bytes")
    published, _left, gaps = run_diagnostics(run)
    names = sorted(path.name for path in published)
    assert not gaps and "facts.json" in names and "launcher-output.bin" in names, names
    assert not any("data" in path.relative_to(run).parts for path in published), published
    (kept,) = (run / "stability-server-data").glob("*/data/state/tasks.json")  # retained raw, unpublished, whole
    assert kept.read_text(encoding="utf-8") == '{"exact": "model and tool sources"}'
    (facts,) = (run / "stability-server").glob("*/facts.json")
    assert json.loads(facts.read_text(encoding="utf-8")) == {"teardown": "confirmed"}
