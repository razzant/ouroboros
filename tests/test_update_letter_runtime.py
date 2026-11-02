"""The Runtime context carries the update letter with its qualification in counts; the whole record has a reader."""
from __future__ import annotations

import json

from ouroboros import store_import, update_letter
from ouroboros.tools.query_data import _query_data
from ouroboros.tools.registry import ToolContext

BASE, TARGET = "a" * 40, "b" * 40
CHECKS = [{"name": "full_battery", "status": "passed", "source": "author_declared", "detail": "x" * 400},
          {"name": "cost_contract", "status": "unknown", "source": "missing", "detail": "No execution evidence."}]


def _record(root):
    commits = [{"sha": f"{n:040x}", "subject": f"commit {n} " + "s" * 200, "date": "2026-10-27T10:00:00+00:00",
                "receipt": {"schema": 1, "checks": CHECKS, "review": {"observed": "pass", "declared": "pass"}}
                if n % 2 else None, "body_read": True} for n in range(40)]
    github = {"status": "partial", "reason": "GitHub response page budget reached.", "missing_shas": ["c" * 40],
              "runs": [{"sha": commits[0]["sha"], "status": "completed", "conclusion": "success"},
                       {"sha": commits[1]["sha"], "status": "completed", "conclusion": "failure"},
                       {"sha": commits[2]["sha"], "status": "in_progress", "conclusion": None}]}
    qualification = {"base_sha": BASE, "target_sha": TARGET, "commits": commits, "github": github}
    record = {"state": "ready", "text": "WHAT THIS UPDATE BRINGS", "key": {"base_sha": BASE, "target_sha": TARGET},
              "author_version": "7.8.0", "target_version": "7.9.0", "written_at": "2026-10-27T10:00:00+00:00",
              "qualification": qualification, "checked_head_sha": BASE}
    path = root / "state" / "update_letter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


def test_the_runtime_letter_keeps_its_text_and_summarizes_the_qualification(tmp_path):
    record = _record(tmp_path)
    state = {"managed_update_cache": {"latest_sha": TARGET, "available": True, "checked_at": "2026-10-27T10:05:00Z"}}
    fact = update_letter.official_update_projection(BASE, drive_root=tmp_path, state=state)
    letter = fact["letter"]
    assert fact["status"] == "update_available" and letter["text"] == "WHAT THIS UPDATE BRINGS"
    assert letter["relation"] == "pending" and letter["target_version"] == "7.9.0"
    assert letter["qualification"] == {
        "base_sha": BASE, "target_sha": TARGET, "commits": 40, "with_receipt": 20,
        "checks": {"full_battery": {"passed (author_declared)": 20}, "cost_contract": {"unknown (missing)": 20}},
        "reviews": {"observed": {"pass": 20}, "declared": {"pass": 20}},
        "post_merge": {"read": "partial", "reason": "GitHub response page budget reached.",
                       "runs": {"success": 1, "failure": 1, "in_progress": 1}, "missing": 1},
        "full": "query_data(op='json', kind='update_letter')"}
    assert len(json.dumps(letter)) * 10 < len(json.dumps(record["qualification"]))
    # The panel and the letter's author read the record itself, untouched.
    panel = update_letter.project_letter(update_letter.read_record(tmp_path), head_sha=BASE, latest_sha=TARGET)
    assert panel["qualification"] == record["qualification"]
    assert json.loads((tmp_path / "state" / "update_letter.json").read_text(encoding="utf-8")) == record


def test_a_letter_without_qualification_is_unchanged(tmp_path):
    from ouroboros.update_qualification import runtime_letter

    assert runtime_letter(None) is None and runtime_letter({"text": "t"}) == {"text": "t"}
    assert runtime_letter({"text": "t", "qualification": None}) == {"text": "t", "qualification": None}


def test_query_data_reads_the_whole_letter_from_the_installation_root(tmp_path):
    store_import.ensure_current_store(tmp_path, with_children=False)
    record = _record(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller")
    result = json.loads(_query_data(ctx, op="json", kind="update_letter", max_bytes=1_000_000))
    assert result["status"] == "ok" and result["data"] == record and result["source"]["plane"] == "canonical"
    refused = json.loads(_query_data(ctx, op="json", kind="update_letter", child_task_id="second"))
    assert refused["status"] == "error" and "installation-wide" in refused["message"]
