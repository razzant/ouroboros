"""v6.73.0 Project Origin Invariant — the start-message-loss class is closed.

Covers the four legs of the meta-fix:
1. ``bind_task_to_project`` REQUIRES a typed origin (ref-by-value or closed-enum
   absence) and enriches a ref-less binding one-way on a same-project re-bind.
2. Producers pass the ingress-captured ref BY VALUE (promote/route/ensure) even
   when the LLM rewrote the message text — no content-derived identity lookup.
3. The Project lens synthesizes the start message from the binding's own
   ``source_text`` when the canonical row is not among the emitted rows
   (rotation OR tail-quota pruning), without duplicates.
4. The context reader follows the frozen legacy cursor across chat.jsonl rotations
   (the old dialogue writer that advanced it is retired).
"""

from __future__ import annotations


import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import (
    bind_task_to_project,
    create_project,
    project_binding_for_task,
)


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    """Q10=A auto-provisions a genesis workspace for file-less project promotes;
    keep it out of the real ~/Ouroboros/projects."""
    monkeypatch.setenv(
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
        str(tmp_path_factory.mktemp("projects_root")),
    )

OWNER_TEXT = "сделай мне детализированную 3d игру про робота"


def _ref(chat_id=1, cmid="owner-msg-1", ts="2026-07-19T21:10:38Z", text=OWNER_TEXT):
    return build_owner_message_ref(
        chat_id=chat_id, client_message_id=cmid, ts=ts, text=text,
    )


# ---------------------------------------------------------------- binder

def test_bind_requires_typed_origin(tmp_path):
    create_project(tmp_path, "alpha")
    with pytest.raises(TypeError):
        bind_task_to_project(tmp_path, "t1", "alpha")  # origin omitted → impossible
    with pytest.raises(ValueError, match="exactly one of"):
        bind_task_to_project(tmp_path, "t1", "alpha", origin={})
    with pytest.raises(ValueError, match="exactly one of"):
        bind_task_to_project(
            tmp_path, "t1", "alpha",
            origin={"ref": _ref(), "text": OWNER_TEXT, "absent": "system"},
        )
    with pytest.raises(ValueError, match="absence reason"):
        bind_task_to_project(tmp_path, "t1", "alpha", origin={"absent": "because"})
    with pytest.raises(ValueError, match="non-empty"):
        bind_task_to_project(
            tmp_path, "t1", "alpha", origin={"ref": {"chat_id": 1}, "text": OWNER_TEXT},
        )


def test_bind_cross_thread_requires_matching_text(tmp_path):
    create_project(tmp_path, "alpha")
    # Cross-thread origin (main chat 1 != project chat) without text → refused.
    with pytest.raises(ValueError, match="text"):
        bind_task_to_project(tmp_path, "t1", "alpha", origin={"ref": _ref()})
    # Text that does not match the ref's integrity hash → refused.
    with pytest.raises(ValueError, match="integrity"):
        bind_task_to_project(
            tmp_path, "t1", "alpha", origin={"ref": _ref(), "text": "different words"},
        )
    row = bind_task_to_project(
        tmp_path, "t1", "alpha", origin={"ref": _ref(), "text": OWNER_TEXT},
    )
    assert row["source_ref"]["client_message_id"] == "owner-msg-1"
    assert row["source_text"] == OWNER_TEXT
    assert "origin_absent" not in row


def test_rebind_enriches_refless_binding_one_way(tmp_path):
    create_project(tmp_path, "alpha")
    first = bind_task_to_project(
        tmp_path, "t1", "alpha", origin={"absent": "post_hoc_unresolved"},
    )
    assert first["origin_absent"] == "post_hoc_unresolved"
    enriched = bind_task_to_project(
        tmp_path, "t1", "alpha", origin={"ref": _ref(), "text": OWNER_TEXT},
    )
    assert enriched["source_ref"] == _ref()
    assert enriched["source_text"] == OWNER_TEXT
    assert "origin_absent" not in enriched
    assert project_binding_for_task(tmp_path, "t1")["source_text"] == OWNER_TEXT
    # A stored valid ref is immutable: a different ref never replaces it.
    other = _ref(cmid="other-msg", text="other text")
    unchanged = bind_task_to_project(
        tmp_path, "t1", "alpha", origin={"ref": other, "text": "other text"},
    )
    assert unchanged["source_ref"] == _ref()
    # Different-project re-bind still refuses.
    create_project(tmp_path, "beta")
    with pytest.raises(ValueError, match="immutable"):
        bind_task_to_project(tmp_path, "t1", "beta", origin={"absent": "system"})


# ---------------------------------------------------------------- producers


# ---------------------------------------------------------------- lens fallback

def _history(tmp_path, chat_id, n_human=None):
    from ouroboros.gateway.history import make_chat_history_endpoint

    endpoint = make_chat_history_endpoint(tmp_path)
    params = {"chat_id": str(chat_id)}
    if n_human is not None:
        params["n_human"] = str(n_human)
    resp = asyncio.run(endpoint(SimpleNamespace(query_params=params)))
    return json.loads(resp.body.decode("utf-8"))["messages"]


# ---------------------------------------------------------------- ingress capture


class _ImmediateThread:
    """``threading.Thread`` whose target has run to its end when ``start()`` returns.

    Tests patch it over the global class, so every caller meets it, not only the code under test: Windows
    ``subprocess`` drains a child's pipes through reader threads it then ``join()``s and polls with ``is_alive()``."""

    def __init__(self, target, args=(), kwargs=None, daemon=False, **_ignored):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        """Nothing to wait for: ``start()`` returned only after the target did."""

    def is_alive(self):
        return False


def test_ensure_worker_falls_back_to_task_record_origin(tmp_path, monkeypatch):
    """SCOPE r1 critical: a QUEUED task self-scoping mid-run must recover its
    origin from the persisted task record (ctx.task_metadata has none)."""
    import supervisor.workers as workers
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    write_task_result(
        tmp_path, "queued-1", STATUS_RUNNING,
        origin_message_ref=_ref(), origin_message_text=OWNER_TEXT,
    )
    ctx = SimpleNamespace(RUNNING={})
    workers.ensure_project_scope(
        {"task_id": "queued-1", "project_id": "robot-city", "project_name": "Robot City"},
        ctx,
    )
    binding = project_binding_for_task(tmp_path, "queued-1")
    assert binding["source_ref"] == _ref()
    assert binding["source_text"] == OWNER_TEXT


def test_sanitize_task_event_caps_origin_text_in_both_copies(tmp_path):
    """Triad r2 critical: the metadata mirror of a direct turn must not carry an
    unbounded origin text into events.jsonl."""
    from ouroboros.utils import sanitize_task_for_event

    big = "x" * 9000
    task = {
        "id": "t1", "text": big, "origin_message_text": big,
        "metadata": {"origin_message_text": big, "client_message_id": "m1"},
    }
    sanitized = sanitize_task_for_event(task, tmp_path, threshold=4000)
    assert len(sanitized["origin_message_text"]) < 5000
    assert len(sanitized["metadata"]["origin_message_text"]) < 5000
    assert task["metadata"]["origin_message_text"] == big  # original untouched


def test_origin_stub_failure_is_loud_typed_event(tmp_path, monkeypatch):
    """Triad r8: a failed early-origin persist is LOUD — warning + durable typed
    anomaly (deliberately non-fatal: the owner's task outlives its start message)."""
    import ouroboros.agent as agent_mod

    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise OSError("disk full")

    monkeypatch.setattr(agent_mod, "write_task_result", _boom)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    agent_mod._persist_early_origin_stub(tmp_path, {
        "id": "direct-x", "chat_id": 1,
        "origin_message_ref": _ref(), "origin_message_text": OWNER_TEXT,
    })
    assert calls["n"] == 2  # bounded retry
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["type"] == "origin_stub_persist_failed"
    assert rows[0]["task_id"] == "direct-x"


def test_lens_over_cap_origins_emit_disclosed_omission_note(tmp_path):
    """Triad r3: past the synthesis cap the omission is DISCLOSED (count + durable
    source named), never a silent cut."""
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "logs" / "progress.jsonl").write_text("", encoding="utf-8")
    project = create_project(tmp_path, "manyorigins", name="Many")
    for index in range(12):
        text = f"origin message number {index}"
        bind_task_to_project(
            tmp_path, f"task-{index}", "manyorigins",
            origin={
                "ref": _ref(cmid=f"owner-{index}", ts=f"2026-07-19T21:{index:02d}:00Z", text=text),
                "text": text,
            },
        )
    view = _history(tmp_path, int(project["chat_id"]))
    synthesized = [m for m in view if m.get("origin_projected")]
    assert len(synthesized) == 10
    notes = [m for m in view if m.get("system_type") == "origin_omission"]
    assert len(notes) == 1
    assert "2 more" in notes[0]["text"]
    assert "project_task_bindings.json" in notes[0]["text"]


# ---------------------------------------------------------------- attachment-bearing origins

def _attachment_origin(tmp_path, *, cmid="owner-att-1", text="Посмотри\n\n[Attached file: one.png]",
                       source="web", placeholder=False, name="one.png"):
    """A canonical Main row with recorded attachment refs, as the ingress writes it."""
    from ouroboros import chat_uploads
    from supervisor.message_bus import log_chat
    from tests.test_chat_attachments import PNG

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "progress.jsonl").touch()
    _path, stored = chat_uploads.store_upload(PNG, name, data_dir=tmp_path)
    row = log_chat("in", 1, 1, text, source=source, client_message_id=cmid, drive_root=tmp_path,
                   require_write=True, message_meta={"attachments": [stored], "text_placeholder": placeholder})
    return row, _ref(cmid=cmid, ts=row["ts"], text=text)


def _later_project_rows(tmp_path, chat_id, count):
    from supervisor.message_bus import log_chat

    for index in range(count):
        log_chat("in", chat_id, 1, f"later project message {index}", source="web",
                 client_message_id=f"later-{index}", drive_root=tmp_path)


def test_an_attachment_origin_keeps_its_recorded_media_when_quota_prunes_its_row(tmp_path):
    """The binding records the refs its canonical row recorded (never guessed from the text), so the
    quota-pruned copy renders as the row does; once the row is emitted again it adopts the copy:
    same id, origin, text, views and provenance."""
    from ouroboros.chat_uploads import attachment_views, stored_refs

    canonical, ref = _attachment_origin(tmp_path)
    project = create_project(tmp_path, "gallery", name="Gallery")
    binding = bind_task_to_project(tmp_path, "root-task", "gallery", origin={"ref": ref, "text": canonical["text"]})
    assert binding["source_attachments"] == stored_refs(canonical["attachments"])
    assert binding["source_channel"] == "web" and "source_text_placeholder" not in binding
    chat = int(project["chat_id"])
    _later_project_rows(tmp_path, chat, 3)

    whole = [m for m in _history(tmp_path, chat, n_human=50) if m.get("client_message_id") == "owner-att-1"]
    pruned = [m for m in _history(tmp_path, chat, n_human=1) if m.get("client_message_id") == "owner-att-1"]
    assert len(whole) == 1 and not whole[0].get("origin_projected"), "the canonical row, emitted once"
    assert len(pruned) == 1 and pruned[0]["origin_projected"] is True, "quota pruned it: the binding's copy"
    copy, row = pruned[0], whole[0]
    assert copy["attachments"] == row["attachments"] == attachment_views(canonical["attachments"], tmp_path)
    assert copy["attachments"][0]["available"] is True and copy["attachments"][0]["kind"] == "image"
    for key in ("text", "source", "client_message_id", "origin_id"):
        assert copy[key] == row[key], key
    assert "text_placeholder" not in copy and "text_placeholder" not in row
