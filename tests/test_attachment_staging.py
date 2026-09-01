"""First-class attachment access (v6.52.0, P1).

Covers the shared staging substrate (stage_task_attachments), its secret-skip /
bound behavior, the READY artifact_store manifest lines, native image-block
injection via build_user_content, the collect_task_artifact_records exclusion of
staged inputs, and the desktop chat reroute.

Parallel-safe: every test isolates state under tmp_path; no shared global state.
"""

from __future__ import annotations

import base64
import pathlib
import queue

import pytest


# A 1x1 transparent PNG.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _drive(tmp_path):
    drive = tmp_path / "data"
    drive.mkdir(parents=True, exist_ok=True)
    return drive


def _attach_dir(drive, task_id):
    return drive / "task_results" / "artifacts" / task_id / "attachments"


class TestStageTaskAttachments:
    def test_stages_into_artifact_store(self, tmp_path):
        from ouroboros.artifacts import stage_task_attachments

        drive = _drive(tmp_path)
        src = tmp_path / "report.txt"
        src.write_text("hello", encoding="utf-8")

        manifest = stage_task_attachments(drive, "task01", [{"path": str(src), "label": "Report"}])

        assert len(manifest) == 1
        entry = manifest[0]
        assert entry["ordinal"] == 0
        assert entry["status"] == "staged"
        assert entry["reason"] == ""
        assert entry["root"] == "artifact_store"
        assert entry["relpath"].startswith("attachments/")
        assert entry["label"] == "Report"
        assert entry["is_image"] is False
        staged = _attach_dir(drive, "task01") / entry["relpath"].split("/", 1)[1]
        assert staged.is_file()
        assert staged.read_text(encoding="utf-8") == "hello"

    def test_child_drive_and_shared_drive_both_work(self, tmp_path):
        """The relpath is identical regardless of which drive it is staged into;
        the caller resolves it against task['drive_root'] at read time."""
        from ouroboros.artifacts import stage_task_attachments

        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n", encoding="utf-8")

        shared = _drive(tmp_path)
        child = tmp_path / "child_drive"
        child.mkdir()

        m_shared = stage_task_attachments(shared, "tshared", [{"path": str(src)}])
        m_child = stage_task_attachments(child, "tchild", [{"path": str(src)}])

        assert m_shared and m_child
        assert m_shared[0]["relpath"].startswith("attachments/")
        assert m_child[0]["relpath"].startswith("attachments/")
        # Each landed in its OWN drive's artifact store.
        assert (_attach_dir(shared, "tshared") / m_shared[0]["relpath"].split("/", 1)[1]).is_file()
        assert (_attach_dir(child, "tchild") / m_child[0]["relpath"].split("/", 1)[1]).is_file()

    def test_manifest_has_no_bare_absolute_path(self, tmp_path):
        from ouroboros.artifacts import stage_task_attachments

        drive = _drive(tmp_path)
        src = tmp_path / "doc.md"
        src.write_text("x", encoding="utf-8")

        manifest = stage_task_attachments(drive, "task02", [{"path": str(src), "label": "Doc"}])
        entry = manifest[0]
        # No manifest field leaks the absolute source path.
        for value in entry.values():
            assert str(src) != str(value)
        assert "/" not in entry["relpath"].split("/", 1)[1]  # single attachments/ component

    def test_secret_source_skipped_credentials(self, tmp_path):
        from ouroboros.artifacts import stage_task_attachments

        drive = _drive(tmp_path)
        good = tmp_path / "ok.txt"
        good.write_text("fine", encoding="utf-8")
        secret = tmp_path / "credentials.json"
        secret.write_text("{\"token\": \"x\"}", encoding="utf-8")

        manifest = stage_task_attachments(
            drive, "task03", [{"path": str(secret)}, {"path": str(good)}]
        )
        labels = {m["label"] for m in manifest if m["status"] == "staged"}
        assert "ok.txt" in labels
        assert "credentials.json" not in labels
        assert len(manifest) == 2
        assert manifest[0]["status"] == "rejected"
        assert manifest[0]["reason"] == "secret_source"
        assert manifest[0]["ordinal"] == 0

    def test_secret_source_skipped_ssh_dir(self, tmp_path):
        from ouroboros.artifacts import stage_task_attachments

        drive = _drive(tmp_path)
        ssh = tmp_path / ".ssh"
        ssh.mkdir()
        key = ssh / "id_rsa"
        key.write_text("PRIVATE", encoding="utf-8")

        manifest = stage_task_attachments(drive, "task04", [{"path": str(key)}])
        # #447 G10: the rejection row names the exact rule that fired.
        assert manifest == [{
            "ordinal": 0,
            "status": "rejected",
            "reason": "secret_source",
            "label": "id_rsa",
            "rule": "credential/control directory component '.ssh'",
        }]

    def test_image_source_marked_is_image(self, tmp_path):
        from ouroboros.artifacts import stage_task_attachments

        drive = _drive(tmp_path)
        img = tmp_path / "pic.png"
        img.write_bytes(_PNG_BYTES)

        manifest = stage_task_attachments(drive, "task05", [{"path": str(img)}])
        assert len(manifest) == 1
        assert manifest[0]["is_image"] is True
        assert manifest[0]["mime"] == "image/png"

    def test_missing_and_nonfile_skipped(self, tmp_path):
        from ouroboros.artifacts import stage_task_attachments

        drive = _drive(tmp_path)
        adir = tmp_path / "adir"
        adir.mkdir()
        manifest = stage_task_attachments(
            drive, "task06", [{"path": str(tmp_path / "nope.txt")}, {"path": str(adir)}]
        )
        assert [row["reason"] for row in manifest] == ["source_missing", "source_not_file"]
        assert [row["ordinal"] for row in manifest] == [0, 1]

    def test_large_file_skipped(self, tmp_path, monkeypatch):
        import ouroboros.artifacts as art

        drive = _drive(tmp_path)
        big = tmp_path / "big.bin"
        big.write_bytes(b"\0" * 1024)
        monkeypatch.setattr(art, "_MAX_STAGED_ATTACHMENT_BYTES", 512)
        manifest = art.stage_task_attachments(drive, "task07", [{"path": str(big)}])
        assert manifest[0]["status"] == "rejected"
        assert manifest[0]["reason"] == "file_too_large"

    def test_max_count_bound(self, tmp_path, monkeypatch):
        import ouroboros.artifacts as art

        drive = _drive(tmp_path)
        monkeypatch.setattr(art, "_MAX_STAGED_ATTACHMENTS", 2)
        items = []
        for i in range(5):
            f = tmp_path / f"f{i}.txt"
            f.write_text(str(i), encoding="utf-8")
            items.append({"path": str(f)})
        manifest = art.stage_task_attachments(drive, "task08", items)
        assert len(manifest) == 5
        assert [row["status"] for row in manifest] == [
            "staged", "staged", "rejected", "rejected", "rejected",
        ]
        assert {row["reason"] for row in manifest[2:]} == {"attachment_limit_exceeded"}

    def test_copy_failure_is_a_typed_row(self, tmp_path, monkeypatch):
        import ouroboros.artifacts as art

        drive = _drive(tmp_path)
        source = tmp_path / "copy-me.txt"
        source.write_text("payload", encoding="utf-8")
        monkeypatch.setattr(
            art.shutil,
            "copy2",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )

        manifest = art.stage_task_attachments(drive, "copy-fail", [{"path": str(source)}])

        assert manifest == [{
            "ordinal": 0,
            "status": "rejected",
            "reason": "copy_failed",
            "label": "copy-me.txt",
        }]

    def test_retry_reuses_staged_file_without_duplicate(self, tmp_path):
        from ouroboros.artifacts import remove_staged_attachments, stage_task_attachments

        drive = _drive(tmp_path)
        source = tmp_path / "same.txt"
        source.write_text("same bytes", encoding="utf-8")
        first = stage_task_attachments(drive, "retry-stage", [{"path": str(source)}])
        second = stage_task_attachments(drive, "retry-stage", [{"path": str(source)}])

        assert first[0]["relpath"] == second[0]["relpath"]
        assert len(list(_attach_dir(drive, "retry-stage").iterdir())) == 1
        # Rejecting/cancelling the repeated call cannot unlink the already
        # admitted file owned by the first call.
        assert remove_staged_attachments(second) == 0
        assert pathlib.Path(first[0]["abs_path"]).is_file()
        assert not any(key.startswith("_") for row in second for key in row)
        assert remove_staged_attachments(first) == 1

    def test_changed_source_versions_never_overwrite_prior_delivery(self, tmp_path):
        from ouroboros.artifacts import remove_staged_attachments, stage_task_attachments

        drive = _drive(tmp_path)
        source = tmp_path / "mutable.txt"
        source.write_text("version one", encoding="utf-8")
        first = stage_task_attachments(drive, "versioned-stage", [{"path": str(source)}])
        first_path = pathlib.Path(first[0]["abs_path"])
        source.write_text("version two", encoding="utf-8")
        second = stage_task_attachments(drive, "versioned-stage", [{"path": str(source)}])
        second_path = pathlib.Path(second[0]["abs_path"])
        assert second_path != first_path
        assert second_path.read_text(encoding="utf-8") == "version two"

        source.write_text("version three", encoding="utf-8")
        third = stage_task_attachments(drive, "versioned-stage", [{"path": str(source)}])
        third_path = pathlib.Path(third[0]["abs_path"])
        assert third_path != second_path
        assert second_path.read_text(encoding="utf-8") == "version two"
        assert third_path.read_text(encoding="utf-8") == "version three"

        assert remove_staged_attachments(third) == 1
        assert second_path.read_text(encoding="utf-8") == "version two"

    def test_distinct_sources_no_clobber(self, tmp_path):
        from ouroboros.artifacts import stage_task_attachments

        drive = _drive(tmp_path)
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        (d1 / "same.txt").write_text("one", encoding="utf-8")
        (d2 / "same.txt").write_text("two", encoding="utf-8")

        manifest = stage_task_attachments(
            drive, "task09", [{"path": str(d1 / "same.txt")}, {"path": str(d2 / "same.txt")}]
        )
        assert len(manifest) == 2
        rels = {m["relpath"] for m in manifest}
        assert len(rels) == 2  # distinct destinations

    @pytest.mark.parametrize("memory_mode", ["shared", "forked"])
    def test_inherited_inputs_materialize_in_child_artifact_store(
        self, tmp_path, memory_mode,
    ):
        from ouroboros.artifacts import (
            materialize_inherited_attachment_manifest,
            stage_task_attachments,
        )
        from ouroboros.tools.core import _read_file
        from ouroboros.tools.registry import ToolContext

        parent_drive = _drive(tmp_path)
        source = tmp_path / "authority.txt"
        source.write_text("inherited-readable", encoding="utf-8")
        parent_manifest = stage_task_attachments(
            parent_drive, "parent-task", [{"path": str(source), "label": "authority"}],
        )
        child_drive = (
            parent_drive if memory_mode == "shared" else tmp_path / "forked-child"
        )
        child_drive.mkdir(parents=True, exist_ok=True)

        child_manifest, error = materialize_inherited_attachment_manifest(
            parent_manifest, child_drive, "child-task",
        )

        assert error == ""
        assert str(parent_manifest[0]["abs_path"]) not in str(child_manifest)
        ctx = ToolContext(repo_dir=tmp_path, drive_root=child_drive, task_id="child-task")
        assert "inherited-readable" in _read_file(
            ctx, child_manifest[0]["relpath"], root="artifact_store",
        )

    def test_child_contract_and_external_work_order_use_materialized_manifest(self, tmp_path):
        from ouroboros.subagent_work_order import compile_external_work_order
        from ouroboros.tools.control import _build_child_subagent_contract

        parent_path = str(tmp_path / "parent" / "attachments" / "authority.txt")
        child_path = str(tmp_path / "child" / "attachments" / "authority.txt")
        parent_contract = {
            "objective": "parent",
            "attachment_manifest": [{
                "ordinal": 0, "status": "staged", "reason": "", "label": "authority",
                "root": "artifact_store", "relpath": "attachments/authority.txt",
                "abs_path": parent_path, "mime": "text/plain", "is_image": False,
            }],
        }
        child_manifest = [{**parent_contract["attachment_manifest"][0], "abs_path": child_path}]
        contract = _build_child_subagent_contract({
            "tid": "child", "objective": "inspect input", "expected_output": "report",
            "parent_contract": parent_contract, "attachment_manifest": child_manifest,
        })

        assert contract["attachment_manifest"] == child_manifest
        assert parent_path not in str(contract)
        work_order = compile_external_work_order({
            "id": "child", "objective": "inspect input", "task_contract": contract,
        })
        assert "INHERITED TASK INPUTS" in work_order
        # The canonical work-order renderer serializes the manifest with
        # Python's repr; Windows paths therefore carry escaped backslashes.
        assert repr(child_path) in work_order
        assert repr(parent_path) not in work_order


class TestComposeTaskTextManifestLines:
    def test_ready_read_file_lines(self):
        from ouroboros.gateway.tasks import _compose_task_text

        manifest = [
            {"label": "Pic", "root": "artifact_store", "relpath": "attachments/pic.png",
             "mime": "image/png", "is_image": True},
            {"label": "Doc", "root": "artifact_store", "relpath": "attachments/doc.txt",
             "mime": "text/plain", "is_image": False},
        ]
        text = _compose_task_text(
            "Do the thing",
            workspace_root=None,
            workspace_mode="",
            memory_mode="shared",
            workspace_preflight={},
            attachments=manifest,
        )
        assert "[ATTACHMENTS]" in text
        assert "- Pic (image): read_file(root='artifact_store', path='attachments/pic.png')" in text
        assert "read_file(root='artifact_store', path='attachments/doc.txt')" in text
        assert "status=staged, ordinal=0" in text
        # Never a bare absolute path.
        assert "/attachments/pic.png:" not in text


class TestBuildUserContentAttachmentImages:
    def test_emits_native_image_blocks(self, tmp_path):
        from ouroboros.artifacts import stage_task_attachments
        from ouroboros.context import build_user_content

        drive = _drive(tmp_path)
        img = tmp_path / "shot.png"
        img.write_bytes(_PNG_BYTES)
        manifest = stage_task_attachments(drive, "tcontent", [{"path": str(img), "label": "Shot"}])
        images = [m for m in manifest if m["is_image"]]

        task = {
            "id": "tcontent",
            "drive_root": str(drive),
            "text": "look at this",
            "attachment_images": images,
        }
        content = build_user_content(task)
        assert isinstance(content, list)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1
        block = image_blocks[0]
        assert block["image_url"]["url"].startswith("data:image/png;base64,")
        assert block["_source_path"].endswith(".png")
        # The lead text block carries the message text.
        assert any(b.get("type") == "text" and "look at this" in b.get("text", "") for b in content)

    def test_image_base64_backward_compat_still_works(self):
        from ouroboros.context import build_user_content

        task = {
            "id": "tlegacy",
            "text": "hi",
            "image_base64": base64.b64encode(_PNG_BYTES).decode("ascii"),
            "image_mime": "image/png",
            "image_caption": "a cat",
        }
        content = build_user_content(task)
        assert isinstance(content, list)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1

    def test_cap_respected(self, tmp_path, monkeypatch):
        import ouroboros.context_budget as cb
        from ouroboros.artifacts import stage_task_attachments
        from ouroboros.context import build_user_content

        monkeypatch.setattr(cb, "MAX_LIVE_IMAGE_BLOCKS", 1)
        drive = _drive(tmp_path)
        items = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            p.write_bytes(_PNG_BYTES)
            items.append({"path": str(p), "label": f"img{i}"})
        manifest = stage_task_attachments(drive, "tcap", items)
        task = {
            "id": "tcap",
            "drive_root": str(drive),
            "text": "many",
            "attachment_images": [m for m in manifest if m["is_image"]],
        }
        content = build_user_content(task)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1  # capped; rest stay manifest-readable

    def test_non_image_attachment_not_injected(self, tmp_path):
        """A staged non-image (binary/text) yields no attachment_images entry, so
        build_user_content injects no image block for it."""
        from ouroboros.artifacts import stage_task_attachments
        from ouroboros.context import build_user_content

        drive = _drive(tmp_path)
        blob = tmp_path / "data.bin"
        blob.write_bytes(b"\x00\x01\x02binary")
        manifest = stage_task_attachments(drive, "tbin", [{"path": str(blob)}])
        assert manifest and manifest[0]["is_image"] is False
        images = [m for m in manifest if m["is_image"]]
        assert images == []

        task = {"id": "tbin", "drive_root": str(drive), "text": "t", "attachment_images": images}
        content = build_user_content(task)
        # No images -> plain text content.
        assert content == "t"


class TestCollectArtifactRecordsExcludesAttachments:
    def test_attachments_not_recorded_as_deliverables(self, tmp_path):
        from ouroboros.artifacts import (
            collect_task_artifact_records,
            copy_file_to_task_artifacts,
            stage_task_attachments,
            task_artifact_dir_path,
        )

        drive = _drive(tmp_path)
        # A staged INPUT attachment.
        src = tmp_path / "input.txt"
        src.write_text("in", encoding="utf-8")
        stage_task_attachments(drive, "tcollect", [{"path": str(src)}])

        # A real produced deliverable.
        deliverable = tmp_path / "output.txt"
        deliverable.write_text("out", encoding="utf-8")

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.drive_root = str(drive)
        ctx.task_id = "tcollect"
        copy_file_to_task_artifacts(ctx, str(deliverable))

        records = collect_task_artifact_records(drive, "tcollect")
        names = {r["name"] for r in records}
        assert "output.txt" in names
        assert "input.txt" not in names
        # Defensive: no recorded path is under attachments/.
        adir = task_artifact_dir_path(drive, "tcollect")
        for r in records:
            rel = str(r["path"]).replace(str(adir), "").lstrip("/")
            assert not rel.startswith("attachments/")


class _FakeChatAgent:
    """Captures the task dict _run_chat_task builds, without running the LLM."""

    def __init__(self):
        self.task = None

    def handle_task(self, task):
        self.task = task
        return []


class TestDesktopChatFullSetStaging:
    """v6.52.0 (P1, full desktop unify): the WHOLE desktop attachment set routes
    through the shared substrate via task['metadata']['chat_attachment_uploads']."""

    def test_image_plus_pdf_both_staged_and_manifest_rendered(self, tmp_path, monkeypatch):
        import supervisor.workers as workers

        drive = _drive(tmp_path)
        monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
        # This test pins the attachment-staging seam, not the process-lifetime
        # event bus. Use a local queue so the turn never touches get_event_q():
        # a prior TestClient lifespan in the same xdist worker leaves
        # _EVENT_Q_SHUTDOWN latched (server shutdown calls shutdown_event_q()),
        # and the real get_event_q() then raises before agent.handle_task ever
        # runs, leaving agent.task None (seen on the ubuntu CI shard; same
        # isolation precedent as tests/test_inflight_indicator_seams.py::
        # _patch_workers).
        monkeypatch.setattr(workers, "get_event_q", lambda: queue.Queue())
        # Avoid the proactive namer spinning a real thread/LLM in this unit test
        # (it is a local `from ouroboros.project_naming import ...`, so patch source).
        import ouroboros.project_naming as project_naming
        monkeypatch.setattr(project_naming, "spawn_proactive_namer", lambda *a, **k: None)

        # Two uploads on disk: an image and a non-image PDF.
        img_src = tmp_path / "photo.png"
        img_src.write_bytes(_PNG_BYTES)
        pdf_src = tmp_path / "report.pdf"
        pdf_src.write_bytes(b"%PDF-1.4\n%fake\n")

        agent = _FakeChatAgent()
        workers._run_chat_task(
            agent,
            1,
            "look at these",
            None,
            task_metadata={
                "chat_attachment_uploads": [
                    {"path": str(img_src), "label": "photo.png", "mime": "image/png"},
                    {"path": str(pdf_src), "label": "report.pdf", "mime": "application/pdf"},
                ]
            },
        )

        task = agent.task
        assert task is not None
        # Both files staged under the artifact_store attachments dir.
        adir = _attach_dir(drive, task["id"])
        staged_names = sorted(p.name for p in adir.iterdir() if p.is_file())
        assert any(n.endswith(".png") for n in staged_names)
        assert any(n.endswith(".pdf") for n in staged_names)
        # attachment_images carries ONLY the image.
        imgs = task.get("attachment_images")
        assert isinstance(imgs, list) and len(imgs) == 1
        assert imgs[0]["is_image"] is True
        assert imgs[0]["relpath"].startswith("attachments/")
        # drive_root is set so build_user_content can resolve the relpaths.
        assert task["drive_root"] == str(drive)
        # The READY read_file manifest is appended for EVERY file (image + pdf).
        assert "[ATTACHMENTS]" in task["text"]
        assert "[END_ATTACHMENTS]" in task["text"]
        assert "read_file(root='artifact_store'" in task["text"]
        assert task["text"].count("read_file(root='artifact_store'") == 2

    def test_no_uploads_keeps_legacy_inline_image(self, tmp_path, monkeypatch):
        """Backward-compat: with no chat_attachment_uploads, the single-image base64
        seam is untouched (image_base64 stays, nothing is staged)."""
        import supervisor.workers as workers

        drive = _drive(tmp_path)
        monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
        # Same event-bus isolation as test_image_plus_pdf_both_staged_and_
        # manifest_rendered: a latched _EVENT_Q_SHUTDOWN from a prior TestClient
        # lifespan in this xdist worker must not abort the turn before
        # handle_task.
        monkeypatch.setattr(workers, "get_event_q", lambda: queue.Queue())
        import ouroboros.project_naming as project_naming
        monkeypatch.setattr(project_naming, "spawn_proactive_namer", lambda *a, **k: None)

        b64 = base64.b64encode(_PNG_BYTES).decode("ascii")
        agent = _FakeChatAgent()
        workers._run_chat_task(agent, 1, "hi", (b64, "image/png", "a shot"))

        task = agent.task
        assert task is not None
        assert task.get("image_base64") == b64
        assert task.get("image_mime") == "image/png"
        assert "attachment_images" not in task
        assert "[ATTACHMENTS]" not in task["text"]

    def test_initial_ui_turn_is_atomic_on_staging_rejection(self, tmp_path, monkeypatch):
        import supervisor.workers as workers

        drive = _drive(tmp_path)
        monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
        notices = []
        monkeypatch.setattr(
            workers, "send_with_budget", lambda _chat_id, text, **_kwargs: notices.append(text),
        )
        import ouroboros.projects_registry as projects_registry

        bindings = []
        monkeypatch.setattr(
            projects_registry,
            "bind_task_to_project",
            lambda *args, **kwargs: bindings.append((args, kwargs)),
        )
        agent = _FakeChatAgent()

        workers._run_chat_task(
            agent,
            1,
            "must have the file",
            task_metadata={
                "project_id": "project-one",
                "chat_attachment_uploads": [
                    {"path": str(tmp_path / "missing.txt"), "label": "missing.txt"},
                ],
            },
        )

        assert agent.task is None
        assert bindings == []
        assert notices and "source_missing" in notices[-1]
        assert not (drive / "task_results" / "artifacts").exists()


class TestChatAttachmentUploads:
    """ws._chat_attachment_uploads: confine to validated data/uploads/ basenames."""

    def _setup(self, tmp_path, monkeypatch):
        import ouroboros.gateway.ws as ws

        data_dir = tmp_path / "data"
        uploads = data_dir / "uploads"
        uploads.mkdir(parents=True)
        monkeypatch.setattr(ws, "DATA_DIR", str(data_dir))
        return ws, uploads

    def test_resolves_validated_uploads(self, tmp_path, monkeypatch):
        ws, uploads = self._setup(tmp_path, monkeypatch)
        (uploads / "abc_photo.png").write_bytes(_PNG_BYTES)
        (uploads / "def_report.pdf").write_bytes(b"%PDF-1.4\n")

        specs = ws._chat_attachment_uploads([
            {"filename": "abc_photo.png", "display_name": "photo.png", "mime": "image/png"},
            {"filename": "def_report.pdf", "display_name": "report.pdf", "mime": "application/pdf"},
        ])
        assert len(specs) == 2
        assert all(s["path"].startswith(str(uploads)) for s in specs)
        labels = {s["label"] for s in specs}
        assert labels == {"photo.png", "report.pdf"}

    def test_rejects_traversal_and_missing(self, tmp_path, monkeypatch):
        ws, uploads = self._setup(tmp_path, monkeypatch)
        # A secret OUTSIDE uploads that a traversal attachment would try to read.
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")
        (uploads / "present.txt").write_text("ok", encoding="utf-8")

        specs = ws._chat_attachment_uploads([
            {"filename": "../secret.txt", "display_name": "x"},        # traversal: basename'd -> secret.txt, not in uploads
            {"filename": "/etc/passwd", "display_name": "y"},          # absolute: basename'd -> passwd, missing
            {"filename": "missing.txt", "display_name": "z"},          # not on disk
            {"filename": "present.txt", "display_name": "present"},    # the only valid one
        ])
        assert len(specs) == 4
        assert [row["path"] for row in specs[:3]] == ["", "", ""]
        assert specs[3]["label"] == "present"
        assert specs[3]["path"] == str((uploads / "present.txt").resolve(strict=False))

    def test_non_list_returns_empty(self, tmp_path, monkeypatch):
        ws, _ = self._setup(tmp_path, monkeypatch)
        assert ws._chat_attachment_uploads(None) == []
        assert ws._chat_attachment_uploads("nope") == []
        assert ws._chat_attachment_uploads([]) == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
