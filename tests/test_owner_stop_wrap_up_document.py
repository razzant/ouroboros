"""#1564 A: the owner reads every row and saved text Wrap up kept, whole, from the task's
Files: the task detail names ``wrap_up_document`` and the existing file route writes it on read
(``owner_stop_receipt.write_wrap_up_document``). Nothing is written on the Stop path.

Offline fixtures only. The author does not execute this suite; the parent uses
safe_test in its external test root, with the existing test admission overlay.
"""
from __future__ import annotations

import json

from ouroboros import delegate_collection as collection
from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_terminal as terminal
from ouroboros import owner_stop_receipt as osr
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_delegate_body_collection import started
from tests.test_delegate_body_collection import terminal as detail
from tests.test_owner_stop_retained_work import (
    CHILD_FIELD,
    EVENT_FIELD,
    _capture,
    _child,
    _client,
    _collect,
    _download,
    _retained_long,
    _wrap_up,
)


def _spill(root, tid, rid, payload: bytes):
    """Publish PAYLOAD as RID's complete saved output, collected from its current engine report."""
    from ouroboros.delegate_output import retain_terminal_bytes

    artifact = retain_terminal_bytes(root, tid, rid, payload.decode("utf-8"), payload)
    custody.emit(root, custody.OUTPUT_SPILLED, {"run_id": rid, "task_id": tid, "artifact": artifact["source"]["path"],
        "source": artifact["source"], "staged": True, "full_content": True, "sha256": artifact["sha256"],
        "terminal_sha": current.run(root, rid).terminal_source["sha256"]})
    return artifact


def _reported(root, rid, output, tid="kid"):
    """A stopped run of TID whose engine report carries OUTPUT as its primary output."""
    entry = started(root, tid=tid, rid=rid, root_task_id="parent1")
    report = detail(rid, text="")
    report["summary"]["state"] = "cancelled"
    report["primaryOutput"] = output
    collection.observe_run(root, entry, report)
    return entry


def _drive_state(root):
    """Every path under the drive with its mode, and each file's size and mtime, apart from the
    task store's own SQLite files, which a reading connection may touch."""
    import stat

    state = {}
    for path in root.rglob("*"):
        if ".sqlite" not in path.name:
            info = path.lstat()
            state[path.relative_to(root).as_posix()] = (info.st_mode,) if stat.S_ISDIR(info.st_mode) else (
                info.st_mode, info.st_size, info.st_mtime_ns)
    return state


def test_task_detail_names_the_wrap_up_document_the_file_route_writes_on_read(tmp_path, monkeypatch):
    """Check 5: a settled failed root's detail names ``wrap_up_document`` beside more than eight ordinary
    files, counting, sizing and hashing nothing; the existing file route writes it on read from the
    stored receipt (a range and HEAD agree with the whole). Nothing is stored or listed, the other
    files and refusals keep their responses, and a child's detail names no root receipt."""
    from ouroboros.artifacts import collect_task_artifact_records, store_task_artifact_bytes
    from ouroboros.gateway import task_archive

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="Its answer in its own words.", terminal_origin="model_final")
    _collect(tmp_path, monkeypatch)
    intent, stop = _wrap_up(tmp_path)
    _capture(tmp_path, "parent1", stop)
    for index in range(9):
        store_task_artifact_bytes(tmp_path, "parent1", f"report-{index}.txt", b"ordinary %d" % index)
    files = collect_task_artifact_records(tmp_path, "parent1")
    write_task_result(tmp_path, "parent1", "failed", artifacts=files)
    client = _client(tmp_path)
    task = client.get("/api/tasks/parent1").json()
    assert task["status"] == "failed"
    assert task["wrap_up_document"] == {"name": "work-saved-at-wrap-up-parent1.md", "request_id": intent["request_id"]}
    assert sorted(row["name"] for row in task["artifacts"]) == [f"report-{index}.txt" for index in range(9)]
    saved, response = _download(client)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["x-ouroboros-artifact-identity"] == "unmeasured"
    assert "x-ouroboros-artifact-sha256" not in response.headers
    text = response.content.decode("utf-8")
    assert text.startswith(f"# Work saved at Wrap up\n\nTask parent1, Stop request {intent['request_id']}.")
    assert "- Subtask kid (Stopped): its answer is saved\n  - Run run-one stopped: full output saved" in text
    assert "### Subtask kid\n\nIts answer, in its own words:\n\n> Its answer in its own words." in text
    assert ("### Subtask kid, run run-one\n\nFull output, in the run's own words (from its saved output):"
            "\n\n> saved child answer") in text
    url = f"/api/tasks/parent1/artifacts/{saved['name']}"
    ranged = client.get(url, params={"wrap_up": saved["request_id"]}, headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and ranged.content == response.content[:100]
    assert ranged.headers["content-range"] == f"bytes 0-99/{len(response.content)}"
    head = client.head(url, params={"wrap_up": saved["request_id"]})
    assert head.status_code == 200 and int(head.headers["content-length"]) == len(response.content)
    assert collect_task_artifact_records(tmp_path, "parent1") == files, "written on read: nothing is stored"
    assert client.get("/api/tasks/parent1").json()["artifacts"] == task["artifacts"]
    assert client.get(url).status_code == 404, "without its selector the name is no task file"
    if task_archive.CONFINED:  # an ordinary file keeps the route's platform boundary (#1297)
        assert client.get("/api/tasks/parent1/artifacts/report-0.txt").content == b"ordinary 0"
    stop_id, doc = saved["request_id"], saved["name"]
    # One selector per request: archive and wrap_up refuse each other (under either name) and relpath/source.
    for name, params, status, code in (
            (doc, {"wrap_up": "ci_other"}, 404, "wrap_up_document_unavailable"),
            (doc, {"wrap_up": stop_id, "relpath": doc}, 400, "wrap_up_document_invalid"),
            (doc, {"wrap_up": stop_id, "source": "x"}, 400, "wrap_up_document_invalid"),
            (doc, {"wrap_up": stop_id, "archive": "reports"}, 400, "artifact_archive_invalid"),
            ("reports.zip", {"archive": "reports", "wrap_up": stop_id}, 400, "artifact_archive_invalid"),
            ("reports.zip", {"archive": "reports", "source": "x"}, 400, "artifact_archive_invalid"),
            (doc, {"relpath": doc, "source": "x"}, 400, "artifact_relpath_invalid")):
        refused = client.get(f"/api/tasks/parent1/artifacts/{name}", params=params)
        assert (refused.status_code, refused.json().get("reason_code")) == (status, code), (name, params)
    foreign = client.get("/api/tasks/parent1/artifacts/work-saved-at-wrap-up-kid.md",
                         params={"wrap_up": saved["request_id"]})
    assert foreign.status_code == 404
    assert "wrap_up_document" not in client.get("/api/tasks/kid").json(), "a child's detail names no root receipt"


def test_a_large_saved_output_reaches_the_owner_whole_and_a_damaged_or_lost_one_is_named(tmp_path, monkeypatch):
    """Check 1: an output above 4.5 MiB reaches the owner's download whole, its last sentence
    included, and a byte range of the download is the same bytes. Damaged, then lost, it is a
    named gap beside the engine report's own preview; nothing is invented."""
    from ouroboros.delegate_output import staged_source_path

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="kept answer", terminal_origin="model_final")
    tail = "TAIL-SENTINEL: the saved output's last sentence."
    entry = _retained_long(tmp_path, whole="long saved line\n" * 300_000 + tail)
    assert entry.output_source["size"] > 4.5 * 1024 * 1024
    _intent, stop = _wrap_up(tmp_path)
    _capture(tmp_path, "parent1", stop)
    write_task_result(tmp_path, "parent1", "failed")
    client = _client(tmp_path)
    _saved, response = _download(client)
    assert response.status_code == 200, response.text
    assert response.content.endswith(f"> {tail}\n".encode("utf-8"))
    assert response.text.count("> long saved line\n") == 300_000
    ending = _download(client, headers={"Range": "bytes=-200"})[1]
    assert ending.status_code == 206 and ending.content == response.content[-200:]
    staged = staged_source_path(tmp_path, entry.output_source)
    staged.write_bytes(staged.read_bytes()[:-2] + b"!}")  # the recorded size, other bytes
    text = _download(client)[1].text
    assert "The text from its saved output could not be read: its bytes no longer match what was recorded." in text
    assert tail not in text
    assert "Partial output, in the run's own words (from the engine's last report):\n\n> The long answer begins" in text
    staged.unlink()
    assert ("The text from its saved output could not be read: it was not found where it is kept."
            in _download(client)[1].text)


def test_earlier_outputs_newer_reports_and_later_collections_keep_their_labels(tmp_path, monkeypatch):
    """Check 2: a complete output saved before a newer, uncollected report stays in the download,
    labelled earlier, beside that report's partial words. An output the run's own receipt
    published after Wrap up reads as collected after Wrap up; the list stays what Wrap up saw."""
    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="kid words", terminal_origin="model_final")
    entry = _collect(tmp_path, monkeypatch, text="first full answer")
    revision = detail(entry.run_id, text="later partial words", seq=5)
    revision["summary"]["state"] = "cancelled"
    revision["primaryOutput"].update(bytes=100_000, truncated=True)
    collection.observe_run(tmp_path, entry, revision)
    _intent, stop = _wrap_up(tmp_path)
    run = _capture(tmp_path, "parent1", stop)[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"][0]
    assert run["output_state"] == "retained_earlier_observation"
    write_task_result(tmp_path, "parent1", "failed")
    client = _client(tmp_path)
    text = _download(client)[1].text
    listed = "- Run run-one stopped: an earlier output saved, the latest not collected"
    assert listed in text
    assert ("### Subtask kid, run run-one\n\nFull output, in the run's own words (from an earlier saved output):"
            "\n\n> first full answer\n\nPartial output, in the run's own words (from the engine's last report):"
            "\n\n> later partial words\n") in text
    assert text.count("> first full answer") == 1, "the report the earlier output was collected from is not repeated"
    artifact = _spill(tmp_path, "kid", entry.run_id, json.dumps({"primary_output": {"text": "later whole answer"}}).encode())
    terminal.refresh_terminal_reconciliation(tmp_path, "kid")  # a no-op when the spill already re-published it
    published = load_task_result(tmp_path, "kid")["delegate_terminal_reconciliation"]["terminal_runs"][0]
    assert published["staged_output"]["sha256"] == artifact["sha256"] and published["other_staged_outputs"]
    text = _download(client)[1].text
    assert listed in text, "the list stays the one Wrap up recorded"
    assert ("Full output, in the run's own words (from its saved output, not in the Wrap up snapshot):"
            "\n\n> later whole answer") in text
    assert "> first full answer" in text and "> later partial words" in text


def test_cancelled_child_salvage_reaches_the_owner_whole_without_its_private_path(tmp_path, monkeypatch):
    """Check 3: a natively cancelled child's result is the host's kill note: a bounded preview
    naming a private copy. The download quotes that note as recorded, claiming no author, with
    the path replaced, then the full copy its salvage owner kept, tail included; an API child's
    recorded salvage is quoted once. Without the copy the note says so; nothing is invented."""
    import pathlib

    from ouroboros import observability

    write_task_result(tmp_path, "parent1", "running")
    full = "salvaged words " * 400 + "SALVAGE-TAIL."
    monkeypatch.setattr(observability, "latest_llm_response_text", lambda *_a: full)
    note = observability.salvaged_output_note(tmp_path, "kid", preserve_root=tmp_path)
    path = observability.preserved_salvage_path(tmp_path, "kid")
    assert path and path in note and "SALVAGE-TAIL." not in note, "the note is a bounded preview naming its copy"
    _child(tmp_path, result="Running task cancelled and worker terminated." + note)
    api_text = "api words " * 500 + "API-TAIL."
    write_task_result(tmp_path, "api-kid", "failed", root_task_id="parent1", parent_task_id="parent1",
                      result=api_text, terminal_origin="host_salvage",
                      terminal_salvage_path=observability.preserve_salvaged_output(tmp_path, "api-kid", api_text))
    _intent, stop = _wrap_up(tmp_path)
    _capture(tmp_path, "parent1", stop)
    write_task_result(tmp_path, "parent1", "failed")
    client = _client(tmp_path)
    text = _download(client)[1].text
    assert str(tmp_path) not in text and "salvaged/" not in text and ".txt" not in text, "no private path"
    assert "- Subtask kid (Stopped): a saved text with no recorded author" in text
    assert ("### Subtask kid\n\nIts saved task text; the author is not recorded:\n\n"
            "> Running task cancelled and worker terminated.\n>\n"
            "> Last persisted intermediate model message (salvaged best-effort, unreviewed; full copy below):\n") in text
    assert f"{osr._KEPT_COPY}\n\n> salvaged words" in text and text.count("SALVAGE-TAIL.") == 1
    assert ("### Subtask api-kid\n\nIts last unreviewed message, kept by Ouroboros (not a final answer):"
            "\n\n> api words") in text and text.count("API-TAIL.") == 1
    pathlib.Path(path).unlink()
    text = _download(client)[1].text
    assert "unreviewed; full copy not readable here):" in text and "SALVAGE-TAIL." not in text
    assert osr._KEPT_COPY not in text


def test_a_copy_only_a_retained_placement_holds_is_unverified_on_stop_and_read_on_download(tmp_path, monkeypatch):
    """The Stop path only stats the canonical copy, so a copy only a retained placement holds is
    unverified there, never called unavailable; the owner's download reads it through the verifying
    reader's placements (after child-drive cleanup too). Gone from there, the download names the gap
    and quotes the engine report it still has."""
    import shutil

    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
    from ouroboros.delegate_output import staged_source_path

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path)
    entry = _collect(tmp_path, monkeypatch)
    canonical = staged_source_path(tmp_path, entry.output_source)
    placed = task_artifact_dir_path(tmp_path / "task_drives" / "kid", "kid").joinpath(
        *entry.output_source["path"].split("/"))
    placed.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(canonical), str(placed))
    assert read_actor_source_bytes(tmp_path, "kid", entry.output_source), "the existing reader recovers it"
    run = _capture(tmp_path, "parent1")[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"][0]
    assert run["output_state"] == "retained_complete" and run["gaps"] == ["retained_output_unverified"]
    owner = osr.retained_work_fields(load_task_result(tmp_path, "parent1"))[EVENT_FIELD]["text"]
    assert "whether the saved output is still readable is unknown" in owner
    write_task_result(tmp_path, "parent1", "failed")
    client = _client(tmp_path)
    saved, response = _download(client)
    assert saved["request_id"] == "", "a receipt that recorded no Stop is named by an empty request"
    assert "Full output, in the run's own words (from its saved output):\n\n> saved child answer" in response.text
    placed.unlink()
    text = _download(client)[1].text
    assert "The text from its saved output could not be read: it was not found where it is kept." in text
    assert "Full output, in the run's own words (from the engine's last report):\n\n> saved child answer" in text


def test_wrap_up_document_opens_only_refs_their_publishing_task_publishes(tmp_path, monkeypatch):
    """Check 6: a ref the stored receipt names that its publishing task's own receipt does not
    publish (forged, foreign) is never opened, nor is an entry naming no exact ref (no digest, no
    record, no list); the download names each instead. A receipt without rows, or whose task id
    the route would refuse, names no saved work."""
    from ouroboros import artifacts

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="kid words", terminal_origin="model_final")
    entry = _collect(tmp_path, monkeypatch)
    _intent, stop = _wrap_up(tmp_path)
    audit = _capture(tmp_path, "parent1", stop)
    forged = {"kind": "task_source", "root": "artifact_store", "task_id": "other", "size": 2, "sha256": "f" * 64,
              "path": "source_handles/tool_results/forged-" + "f" * 64 + ".json"}
    run = audit[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"][0]
    run["other_staged_outputs"] = [forged, {"path": "../../outside.json"}, "malformed", {"sha256": "not-a-digest"}]
    run["other_terminal_observations"] = "not a list"
    write_task_result(tmp_path, "parent1", "failed", delegate_terminal_reconciliation=audit)
    reads = []
    real = artifacts.read_actor_source_bytes
    monkeypatch.setattr(artifacts, "read_actor_source_bytes",
                        lambda *a, **k: reads.append(a[2].get("path")) or real(*a, **k))
    text = _download(_client(tmp_path))[1].text
    assert ("The text from another retained saved output is not published by the task that saved it, so it is not read."
            in text)
    assert set(reads) == {entry.output_source["path"]}, "only the published saved output is opened"
    assert text.count("The text from") == 1 and "> saved child answer" in text
    assert text.count("A record of another retained saved output is malformed, so it is not opened.") == 3
    assert text.count("A record of another retained engine report is malformed, so it is not opened.") == 1
    rows = {"task_id": "parent1", "runs": [{"run_id": "r"}]}
    assert osr.wrap_up_document_view({}) == {} and osr.wrap_up_document_view(
        {"delegate_terminal_reconciliation": {CHILD_FIELD: {**rows, "runs": []}}}) == {}
    refused = {"delegate_terminal_reconciliation": {CHILD_FIELD: {**rows, "task_id": "../x"}}}
    assert osr.wrap_up_document_view(refused) == {}, "a name the route would refuse is never offered"


def test_a_wrong_shape_or_missing_text_is_named_and_a_report_of_no_output_stays_quiet(tmp_path, monkeypatch):
    """Correction 2: a verified saved output of valid JSON in the wrong shape is a named gap, as is
    an engine report that claims output it does not hold; an empty saved output is said to hold no
    text, while a report that recorded no output adds nothing beyond the list's own words."""
    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="kid words", terminal_origin="model_final")
    _reported(tmp_path, "run-shape", None)
    _spill(tmp_path, "kid", "run-shape", b"[1, 2, 3]")
    _reported(tmp_path, "run-claim", {"text": "", "bytes": 500, "truncated": True})
    _reported(tmp_path, "run-empty", {"text": "", "bytes": 0, "truncated": False})
    _spill(tmp_path, "kid", "run-empty", json.dumps({"primary_output": {"text": ""}}).encode("utf-8"))
    _reported(tmp_path, "run-quiet", None)
    _intent, stop = _wrap_up(tmp_path)
    _capture(tmp_path, "parent1", stop)
    write_task_result(tmp_path, "parent1", "failed")
    text = _download(_client(tmp_path))[1].text

    def section(rid):
        head = f"### Subtask kid, run {rid}\n"
        return text.split(head)[1].split("###")[0].strip() if head in text else ""

    assert section("run-shape") == "The text from its saved output could not be read: its record has an unexpected shape."
    assert section("run-claim") == ("The text from the engine's last report could not be read: "
                                    "it reports output it does not hold.")
    assert section("run-empty") == "Its saved output holds no text."
    assert section("run-quiet") == "", "a report that recorded no output adds nothing"
    assert "- Run run-quiet stopped: no output reported" in text


def test_reading_the_wrap_up_document_writes_nothing_to_the_drive(tmp_path, monkeypatch):
    """Correction 1: the owner's real GETs (detail, whole document, a range, HEAD) leave every path
    under the drive as they found it, the task store's own SQLite files aside. A drive without an
    observability store still has none, and an existing salvage store keeps the modes it had: the
    lookup creates no directory and re-applies no private mode; only the salvage writer does."""
    import shutil
    import stat

    from ouroboros import observability

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="Running task cancelled and worker terminated.")
    _collect(tmp_path, monkeypatch)
    _intent, stop = _wrap_up(tmp_path)
    _capture(tmp_path, "parent1", stop)
    write_task_result(tmp_path, "parent1", "failed")
    client = _client(tmp_path)
    saved = client.get("/api/tasks/parent1").json()["wrap_up_document"]  # the detail's first reads come first
    url, params = f"/api/tasks/parent1/artifacts/{saved['name']}", {"wrap_up": saved["request_id"]}
    store = tmp_path / "observability"
    shutil.rmtree(store, ignore_errors=True)

    def read_all():
        assert client.get("/api/tasks/parent1").json()["wrap_up_document"] == saved
        assert client.get(url, params=params, headers={"Range": "bytes=0-49"}).status_code == 206
        assert client.head(url, params=params).status_code == 200
        response = client.get(url, params=params)
        assert response.status_code == 200, response.text
        return response.text

    before = _drive_state(tmp_path)
    read_all()
    assert not store.exists(), "the salvage lookup creates no store"
    assert _drive_state(tmp_path) == before
    observability.preserve_salvaged_output(tmp_path, "kid", "kept salvage copy")  # the writer owns creation
    kept = store / observability.SALVAGED_OUTPUT_DIR
    for directory in (store, kept):
        directory.chmod(0o750)
    before = _drive_state(tmp_path)
    assert "kept salvage copy" in read_all()
    assert _drive_state(tmp_path) == before, "nothing is created, rewritten or re-moded"
    if observability.posix_private_modes_supported():
        assert {stat.S_IMODE(store.stat().st_mode), stat.S_IMODE(kept.stat().st_mode)} == {0o750}


def test_root_two_published_revisions_delayed_stale_audit_and_timestamped_later_collection(tmp_path, monkeypatch):
    """Defect 3: two root revisions published before Wrap up are both frozen in the root snapshot
    (labelled 'its saved output' and 'another retained saved output', never 'earlier' or
    'collected after Wrap up'); a delayed stale audit writing after Wrap up is labelled
    'another retained saved output, not in the Wrap up snapshot'; only genuine timestamps prove
    'collected after Wrap up'."""
    write_task_result(tmp_path, "parent1", "running")
    entry = _collect(tmp_path, monkeypatch, tid="parent1", rid="run-root", text="root rev 1")
    terminal.record_terminal_reconciliation(
        tmp_path, "parent1", terminal.terminal_custody_disclosure(tmp_path, "parent1"))
    stale_audit = terminal.terminal_custody_disclosure(tmp_path, "parent1")

    rev2 = detail("run-root", text="root rev 2", seq=5)
    rev2["summary"]["state"] = "cancelled"
    collection.observe_run(tmp_path, entry, rev2)
    _spill(tmp_path, "parent1", "run-root", json.dumps({"primary_output": {"text": "root rev 2"}}).encode("utf-8"))
    terminal.record_terminal_reconciliation(
        tmp_path, "parent1", terminal.terminal_custody_disclosure(tmp_path, "parent1"))

    _intent, stop = _wrap_up(tmp_path)
    audit = _capture(tmp_path, "parent1", stop)
    root_run = audit[CHILD_FIELD]["runs"][0]
    assert root_run["other_staged_outputs"] and root_run["other_terminal_observations"]
    write_task_result(tmp_path, "parent1", "failed")
    client = _client(tmp_path)
    doc = _download(client)[1].text
    assert "Full output, in the run's own words (from its saved output):\n\n> root rev 2" in doc
    assert "Full output, in the run's own words (from another retained saved output):\n\n> root rev 1" in doc
    assert "collected after Wrap up" not in doc and "not in the Wrap up snapshot" not in doc
    assert "an earlier saved output" not in doc

    # A delayed audit publishes a third uncollected/older ref after Wrap up without timestamps.
    extra = _spill(tmp_path, "parent1", "run-root",
                   json.dumps({"primary_output": {"text": "delayed audit output"}}).encode("utf-8"))
    delayed = dict(stale_audit)
    delayed_run = dict(delayed["terminal_runs"][0])
    delayed_run["staged_output"] = {**delayed_run["staged_output"], **extra["source"], "sha256": extra["sha256"]}
    delayed["terminal_runs"] = [delayed_run]
    terminal.record_terminal_reconciliation(tmp_path, "parent1", delayed)
    doc_after_delayed = _download(client)[1].text
    assert ("Full output, in the run's own words (from another retained saved output, not in the Wrap up snapshot):"
            "\n\n> delayed audit output") in doc_after_delayed
    assert "collected after Wrap up" not in doc_after_delayed

    # Genuine timestamps on both receipt and source prove 'collected after Wrap up'.
    stored = load_task_result(tmp_path, "parent1")
    stored_audit = stored["delegate_terminal_reconciliation"]
    stored_audit[CHILD_FIELD]["captured_at"] = "2026-10-21T19:00:00+00:00"
    for r in stored_audit["terminal_runs"]:
        if r["run_id"] == "run-root":
            r["staged_output"]["at"] = "2026-10-21T19:05:00+00:00"
    write_task_result(tmp_path, "parent1", "failed", delegate_terminal_reconciliation=stored_audit)
    doc_timestamped = _download(client)[1].text
    assert ("Full output, in the run's own words (from another retained saved output, collected after Wrap up):"
            "\n\n> delayed audit output") in doc_timestamped
