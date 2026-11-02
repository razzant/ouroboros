"""Running-generation ranges, same-range reuse, and the origin of a shown text versus a failed read."""
import json

import pytest

from ouroboros import server_process, update_letter as ul
from tests.test_update_letter import _capture_for, _commit, _git, _record, _status, history_repo  # noqa: F401

pytestmark = pytest.mark.serial


def test_server_baseline_is_captured_once_even_when_initial_identity_is_unknown(tmp_path, monkeypatch):
    from ouroboros.gateway import state

    monkeypatch.setattr(server_process, "_source_baseline", None)
    monkeypatch.setattr(state, "_git_checkout_identity", lambda repo: ("branch", "a" * 40))
    assert server_process.capture_server_source_baseline(tmp_path) == "a" * 40
    monkeypatch.setattr(state, "_git_checkout_identity", lambda repo: ("branch", "b" * 40))
    assert server_process.capture_server_source_baseline(tmp_path) == "a" * 40
    monkeypatch.setattr(server_process, "_source_baseline", None)
    monkeypatch.setattr(state, "_git_checkout_identity", lambda repo: (None, ""))
    assert server_process.capture_server_source_baseline(tmp_path) == ""
    monkeypatch.setattr(state, "_git_checkout_identity", lambda repo: ("branch", "b" * 40))
    assert server_process.capture_server_source_baseline(tmp_path) == ""


def test_runtime_status_preserves_checkout_and_only_reads_captured_baseline(monkeypatch):
    monkeypatch.setattr(server_process, "_source_baseline", "a" * 40)
    status = _status(current_sha="b" * 40, latest_sha="", checked_target_sha="c" * 40, available=False)
    bound = ul.runtime_status(status)
    assert bound["current_sha"] == "b" * 40 and bound["running_sha"] == "a" * 40
    assert bound["available"] is False and bound["latest_sha"] == ""
    assert "running_sha" not in status
    assert ul._key_from_status(bound)["target_sha"] == "c" * 40


def _stored_material(drive, key):
    (drive / "state").mkdir(parents=True, exist_ok=True)
    ul.atomic_write_json(ul.material_path(drive), {"schema": 1, "key": key, "text": "commits", "text_bytes": 7})


def test_same_range_reuses_its_material_and_never_touches_the_text(tmp_path, monkeypatch):
    from ouroboros.update_qualification import range_qualification
    record = _record(revision=2)
    record["qualification"] = range_qualification({"base_sha": "a" * 40, "target_sha": "b" * 40,
        "commits": [{"sha": "b" * 40, "subject": "change", "body": ""}]}, "")
    ul.atomic_write_json(ul.record_path(tmp_path), record)
    _stored_material(tmp_path, record["key"])
    monkeypatch.setattr(ul, "_default_git", lambda: lambda cmd: (1, "", ""))
    calls = []
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: calls.append(a))
    for _ in range(2):
        checked = ul.refresh_after_check(_status(), drive_root=tmp_path)
        assert (checked["text"], checked["revision"]) == (record["text"], 2)
    assert calls == []
    # An old record whose last attempt failed keeps its shown text through every check.
    failed = _record(key={**record["key"], "target_sha": "c" * 40}, state="failed", text="", last_good=record,
                     qualification=record["qualification"])
    ul.atomic_write_json(ul.record_path(tmp_path), failed)
    assert ul.refresh_after_check(_status(), drive_root=tmp_path)["last_good"]["text"] == record["text"]
    assert calls == []


def test_running_baseline_collects_cumulative_commits_despite_disk_advance(history_repo, tmp_path, monkeypatch):  # noqa: F811
    repo, base = history_repo["repo"], history_repo["base"]
    middle, target = history_repo["c1"], history_repo["c4"]
    monkeypatch.setattr(ul, "_default_git", lambda: _capture_for(repo))
    collect = ul.collect_range_material
    calls = []
    monkeypatch.setattr(ul, "collect_range_material", lambda b, t, **k: calls.append(collect(b, t, **k)) or calls[-1])
    drive = tmp_path / "drive"
    status = _status(current_sha=middle, running_sha=base, latest_sha=target)
    first = ul.refresh_after_check(status, drive_root=drive)
    assert first["checked_key"]["base_sha"] == base and first["state"] == "none"
    assert {r["sha"] for r in calls[0]["commits"]} == set(_git(repo, "rev-list", f"{base}..{target}").splitlines())
    saved = ul.save_description("Cumulative description", base, target, 0, author={"task_id": "t", "model": "m"},
                                drive_root=drive, git=_capture_for(repo))
    assert saved["outcome"] == "saved"
    # Disk consumed the target; the running generation did not. Reuse A→C.
    landed = dict(status, current_sha=target, latest_sha="", checked_target_sha=target, available=False)
    ul.refresh_after_check(landed, drive_root=drive)
    assert len(calls) == 1
    panel = ul.project_letter_for_panel(landed, drive_root=drive)
    assert panel["relation"] == "pending" and panel["description_current"] is True
    context = ul.official_update_projection(target, drive_root=drive, state={"managed_update_cache": {
        "latest_sha": target, "available": False, "checked_at": "now"}})
    assert context["status"] == "up_to_date" and context["letter"]["relation"] == "applied"
    # Same target, new running generation: B→C, not A→C.
    ul.refresh_after_check(dict(status, running_sha=middle), drive_root=drive)
    assert len(calls) == 2 and calls[-1]["base_sha"] == middle
    # Target advances again: always the full baseline→target range.
    (repo / "new.txt").write_text("new", encoding="utf-8")
    newest = _commit(repo, "new target")
    ul.refresh_after_check(dict(status, latest_sha=newest), drive_root=drive)
    assert calls[-1]["base_sha"] == base and calls[-1]["target_sha"] == newest
    assert middle in {r["sha"] for r in calls[-1]["commits"]}
    material = json.loads(ul.material_path(drive).read_text(encoding="utf-8"))
    assert material["key"]["target_sha"] == newest and f"- {middle} " in material["text"]
    request = ul.author_request_text(ul.read_record(drive), {
        "mode": "write", "key": ul.read_record(drive)["checked_key"], "revision_at_request": 1})
    assert f"{base}..{newest}" in request and "expected_revision=1" in request
    # The saved text keeps its own range while the newer target waits for its own description.
    assert ul.read_record(drive)["key"]["target_sha"] == target


def test_an_unreadable_new_range_keeps_shown_origin_and_names_its_own(tmp_path, monkeypatch):
    good = _record()
    ul.atomic_write_json(ul.record_path(tmp_path), good)
    status = _status(current_sha="b" * 40, running_sha="b" * 40, latest_sha="c" * 40)
    monkeypatch.setattr(ul, "collect_range_material",
                        lambda *a, **k: (_ for _ in ()).throw(ul.MaterialUnavailable("git log failed (rc=128)")))
    monkeypatch.setattr(ul, "_default_git", lambda: lambda cmd: (1, "", ""))
    ul.refresh_after_check(status, drive_root=tmp_path)
    view = ul.project_letter_for_panel(status, drive_root=tmp_path)
    assert view["relation"] == "applied" and view["state"] == "ready"
    assert view["key"] == good["key"] and view["text"] == good["text"]
    assert view["written_at"] == good["written_at"] and view["failed_at"]
    assert view["latest_failed_key"] == ul._key_from_status(status) and view["error_kind"] == "material_unavailable"
    assert view["description_current"] is False


def test_unknown_baseline_never_uses_disk_head_or_reads_a_range(tmp_path, monkeypatch):
    good = _record()
    ul.atomic_write_json(ul.record_path(tmp_path), good)
    calls = []
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(ul, "_default_git", lambda: lambda cmd: (1, "", ""))
    record = ul.refresh_after_check(_status(running_sha=""), drive_root=tmp_path)
    assert record["material_error"]["kind"] == "runtime_source_unavailable" and calls == []
    assert record["text"] == good["text"] and record["key"] == good["key"]
    assert json.loads(ul.record_path(tmp_path).read_text(encoding="utf-8"))["checked_key"]["base_sha"] == ""
