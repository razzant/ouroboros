import threading

import pytest

from ouroboros.improvement_backlog import (
    append_backlog_items,
    backlog_path,
    format_backlog_digest,
    load_backlog_items,
)


@pytest.fixture(autouse=True)
def _stub_semantic_dedup(monkeypatch):
    """Neutralize the semantic-dedup LLM call for this module.

    These tests exercise backlog/groom LOGIC, not the dedup detector. Seeding via
    ``append_backlog_items`` runs the C9.2 semantic-redirect pre-pass, which calls
    ``ouroboros.semantic_dedup.find_semantic_duplicate_id`` once per fingerprint-MISS
    that has candidates — a real light-model NETWORK call. With no API key it
    retry-storms (minutes, non-deterministic) before failing open to ``None``; that
    made ``test_groom_backlog_rejects_invented_items`` alone ~129s (~40% of the whole
    suite), because ``_seed_many`` seeds before any mock is installed. Stub the detector
    to its own fail-open default (``None`` = no duplicate, exactly what the doomed call
    eventually returns for the distinct seeded items) so the module is network-free and
    deterministic. The dedup contract itself is covered by ``test_semantic_dedup_v6370``.
    """
    import ouroboros.semantic_dedup as semantic_dedup

    monkeypatch.setattr(semantic_dedup, "find_semantic_duplicate_id", lambda *a, **k: None)


def test_append_and_load_backlog_items(tmp_path):
    added = append_backlog_items(tmp_path, [{
        "summary": "Resolve recurring review blocker: tests_affected",
        "category": "review",
        "source": "review_evidence",
        "task_id": "task-1",
        "evidence": "Fix the missing test before commit",
        "context": "blocked commit",
        "proposed_next_step": "Run plan_task for the narrow fix.",
    }])

    assert added == 1
    path = backlog_path(tmp_path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# Improvement Backlog" in text
    assert "Resolve recurring review blocker: tests_affected" in text

    items = load_backlog_items(tmp_path)
    assert len(items) == 1
    assert items[0]["category"] == "review"
    assert items[0]["task_id"] == "task-1"


def test_append_recurrence_bumps_count_not_drop(tmp_path):
    # A: a repeat of the same item is NOT dropped; its count/last_seen are bumped.
    item = {
        "summary": "Reduce recurring task friction around SHELL_EXIT_ERROR",
        "category": "process",
        "source": "execution_reflection",
        "task_id": "task-2",
        "evidence": "SHELL_EXIT_ERROR",
    }
    assert append_backlog_items(tmp_path, [item]) == 1
    assert append_backlog_items(tmp_path, [item]) == 1  # recurrence recorded, not dropped
    items = load_backlog_items(tmp_path)
    assert len(items) == 1
    assert int(items[0]["count"]) == 2


def test_append_concurrent_writers_do_not_drop_entries(tmp_path):
    item_a = {
        "summary": "Investigate recurring grep timeout",
        "category": "tooling",
        "source": "execution_reflection",
        "task_id": "task-a",
        "evidence": "TOOL_TIMEOUT",
        "fingerprint": "fp-a",
        "id": "ibl-fp-a",
    }
    item_b = {
        "summary": "Resolve recurring review blocker: tests_affected",
        "category": "review",
        "source": "review_evidence",
        "task_id": "task-b",
        "evidence": "Fix the missing test before commit",
        "fingerprint": "fp-b",
        "id": "ibl-fp-b",
    }

    results = []

    def _append(item):
        results.append(append_backlog_items(tmp_path, [item]))

    t1 = threading.Thread(target=_append, args=(item_a,))
    t2 = threading.Thread(target=_append, args=(item_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(results) == [1, 1]
    items = load_backlog_items(tmp_path)
    fingerprints = {item["fingerprint"] for item in items}
    assert {"fp-a", "fp-b"}.issubset(fingerprints)


def test_format_backlog_digest_includes_omission_note(tmp_path):
    for idx in range(7):
        append_backlog_items(tmp_path, [{
            "summary": f"Item {idx}",
            "category": "process",
            "source": "execution_reflection",
            "task_id": f"task-{idx}",
            "evidence": f"marker-{idx}",
            "fingerprint": f"fp-{idx}",
            "id": f"ibl-fp-{idx}",
            "created_at": f"2026-04-14T09:0{idx}:00+00:00",
        }])

    digest = format_backlog_digest(tmp_path, limit=3, max_chars=2500)
    assert "## Improvement Backlog" in digest
    assert "open_items: 7" in digest
    assert "⚠️ OMISSION NOTE: 4 additional open backlog items not shown" in digest


def test_priority_and_count_sort_surfaces_important_old_item(tmp_path):
    # B: a high-priority OLD item outranks a newer low-priority junk burst.
    append_backlog_items(tmp_path, [{
        "summary": "Critical infra fix", "category": "infra", "source": "x",
        "evidence": "e", "fingerprint": "fp-hi", "id": "ibl-hi", "priority": "high",
        "created_at": "2026-01-01T00:00:00+00:00",
    }])
    for i in range(5):
        append_backlog_items(tmp_path, [{
            "summary": f"junk {i}", "category": "misc", "source": "x",
            "evidence": f"e{i}", "fingerprint": f"fp-{i}", "id": f"ibl-{i}",
            "priority": "low", "created_at": f"2026-05-0{i + 1}T00:00:00+00:00",
        }])
    digest = format_backlog_digest(tmp_path, limit=1)
    assert "ibl-hi" in digest
    assert "priority=high" in digest


def test_close_backlog_items_by_id_and_task(tmp_path):
    from ouroboros.improvement_backlog import close_backlog_items

    append_backlog_items(tmp_path, [{
        "summary": "do x", "category": "c", "source": "s", "evidence": "e",
        "fingerprint": "fp1", "id": "ibl-1", "task_id": "t-9",
    }])
    assert close_backlog_items(tmp_path, ids=["ibl-1"]) == 1
    item = load_backlog_items(tmp_path)[0]
    assert item["status"] == "done"
    assert item.get("closed_at")
    assert format_backlog_digest(tmp_path) == ""  # closed items excluded
    # closing again is a no-op
    assert close_backlog_items(tmp_path, task_id="t-9") == 0


def test_append_preserves_item_without_fingerprint(tmp_path):
    # A parser-valid hand-added item without a fingerprint must survive append.
    from ouroboros.improvement_backlog import backlog_path, ensure_backlog_file

    ensure_backlog_file(tmp_path)
    p = backlog_path(tmp_path)
    p.write_text(p.read_text(encoding="utf-8") + "\n### manual-no-fp\n- status: open\n- summary: keep me\n", encoding="utf-8")
    append_backlog_items(tmp_path, [{
        "summary": "brand new item", "category": "c", "source": "s",
        "evidence": "e", "fingerprint": "fpx", "id": "ibl-x",
    }])
    ids = {it.get("id") for it in load_backlog_items(tmp_path)}
    assert "manual-no-fp" in ids  # not silently dropped
    assert "ibl-x" in ids


def _seed_many(tmp_path, n):
    for i in range(n):
        append_backlog_items(tmp_path, [{
            "summary": f"item {i}", "category": "c", "source": "s",
            "evidence": f"e{i}", "fingerprint": f"fp-{i}", "id": f"ibl-{i}",
        }])


def _patch_groom_llm(monkeypatch, content):
    import ouroboros.config as cfg
    import ouroboros.llm as llmmod
    import ouroboros.llm_observability as obs

    monkeypatch.setattr(llmmod, "LLMClient", lambda *a, **k: object())
    monkeypatch.setattr(cfg, "get_light_model", lambda: "x")
    monkeypatch.setattr(obs, "chat_observed", lambda *a, **k: ({"content": content}, {}))


def test_groom_backlog_rejects_invented_items(tmp_path, monkeypatch):
    from ouroboros import improvement_backlog as ib

    _seed_many(tmp_path, 35)
    before = len(load_backlog_items(tmp_path))
    _patch_groom_llm(monkeypatch, '[{"summary":"invented survivor","category":"x","source":"y","count":"1"}]')
    assert ib.groom_backlog(tmp_path, cap=30) == 0  # fail closed
    assert len(load_backlog_items(tmp_path)) == before  # NOT wiped


def test_groom_backlog_floor_aborts_excessive_drop(tmp_path, monkeypatch):
    import json as _json

    from ouroboros import improvement_backlog as ib

    _seed_many(tmp_path, 35)
    items = load_backlog_items(tmp_path)
    keep = [{"id": it["id"], "fingerprint": it["fingerprint"], "summary": it["summary"]} for it in items[:3]]
    before = len(items)
    _patch_groom_llm(monkeypatch, _json.dumps(keep))
    assert ib.groom_backlog(tmp_path, cap=30) == 0  # below cap//2 floor -> abort
    assert len(load_backlog_items(tmp_path)) == before


def test_groom_backlog_keeps_existing_subset(tmp_path, monkeypatch):
    import json as _json

    from ouroboros import improvement_backlog as ib

    _seed_many(tmp_path, 35)
    items = load_backlog_items(tmp_path)
    keep = [{"id": it["id"], "fingerprint": it["fingerprint"], "summary": it["summary"], "status": "open"} for it in items[:20]]
    _patch_groom_llm(monkeypatch, _json.dumps(keep))
    assert ib.groom_backlog(tmp_path, cap=30) == 20
    assert len(load_backlog_items(tmp_path)) == 20


def test_groom_backlog_preserves_no_fingerprint_item(tmp_path, monkeypatch):
    import json as _json

    from ouroboros import improvement_backlog as ib
    from ouroboros.improvement_backlog import backlog_path, ensure_backlog_file

    _seed_many(tmp_path, 34)
    ensure_backlog_file(tmp_path)
    p = backlog_path(tmp_path)
    p.write_text(p.read_text(encoding="utf-8") + "\n### manual-no-fp\n- status: open\n- summary: keep me groom\n", encoding="utf-8")
    items = load_backlog_items(tmp_path)
    nofp = [it for it in items if it["id"] == "manual-no-fp"][0]
    others = [it for it in items if it["id"] != "manual-no-fp"][:19]
    keep = [{"id": it["id"], "fingerprint": it.get("fingerprint", ""), "summary": it["summary"]}
            for it in (others + [nofp])]
    _patch_groom_llm(monkeypatch, _json.dumps(keep))
    assert ib.groom_backlog(tmp_path, cap=30) == 20
    ids = {it["id"] for it in load_backlog_items(tmp_path)}
    assert "manual-no-fp" in ids  # no-fingerprint item preserved through grooming


def test_serialize_preserves_hand_added_custom_field(tmp_path, monkeypatch):
    import json as _json

    from ouroboros import improvement_backlog as ib
    from ouroboros.improvement_backlog import backlog_path, ensure_backlog_file

    _seed_many(tmp_path, 34)
    ensure_backlog_file(tmp_path)
    p = backlog_path(tmp_path)
    p.write_text(
        p.read_text(encoding="utf-8")
        + "\n### manual-x\n- status: open\n- summary: keep me\n- owner_note: do-not-lose-this\n",
        encoding="utf-8",
    )
    # A new append re-serializes the whole file; the custom field must survive.
    append_backlog_items(tmp_path, [{
        "summary": "trigger reserialize", "category": "c", "source": "s",
        "evidence": "e", "fingerprint": "fp-z", "id": "ibl-z",
    }])
    manual = {it["id"]: it for it in load_backlog_items(tmp_path)}["manual-x"]
    assert manual.get("owner_note") == "do-not-lose-this"

    # And it must also survive a grooming pass (no-fingerprint passthrough).
    items = load_backlog_items(tmp_path)
    keep = [{"id": it["id"], "fingerprint": it["fingerprint"], "summary": it["summary"]}
            for it in items if it.get("fingerprint")][:20]
    _patch_groom_llm(monkeypatch, _json.dumps(keep))
    ib.groom_backlog(tmp_path, cap=30)
    manual2 = {it["id"]: it for it in load_backlog_items(tmp_path)}.get("manual-x")
    assert manual2 and manual2.get("owner_note") == "do-not-lose-this"


def test_groom_backlog_aborts_on_concurrent_change(tmp_path, monkeypatch):
    import json as _json

    from ouroboros import improvement_backlog as ib

    _seed_many(tmp_path, 35)
    items = load_backlog_items(tmp_path)
    keep = [{"id": it["id"], "fingerprint": it["fingerprint"], "summary": it["summary"]} for it in items[:20]]

    def racing_chat(*a, **k):
        # A concurrent append lands DURING the lock-free LLM call.
        append_backlog_items(tmp_path, [{
            "summary": "concurrent", "category": "c", "source": "s",
            "evidence": "e", "fingerprint": "fp-new", "id": "ibl-new",
        }])
        return ({"content": _json.dumps(keep)}, {})

    import ouroboros.config as cfg
    import ouroboros.llm as llmmod
    import ouroboros.llm_observability as obs
    monkeypatch.setattr(llmmod, "LLMClient", lambda *a, **k: object())
    monkeypatch.setattr(cfg, "get_light_model", lambda: "x")
    monkeypatch.setattr(obs, "chat_observed", racing_chat)

    assert ib.groom_backlog(tmp_path, cap=30) == 0  # aborts on concurrent change
    ids = {it["id"] for it in load_backlog_items(tmp_path)}
    assert "ibl-new" in ids  # concurrent append survived (no lost update)
    assert len(load_backlog_items(tmp_path)) == 36


def test_recurrence_reopens_done_item(tmp_path):
    from ouroboros.improvement_backlog import close_backlog_items

    item = {"summary": "flaky", "category": "c", "source": "s", "evidence": "e",
            "fingerprint": "fp3", "id": "ibl-3"}
    append_backlog_items(tmp_path, [item])
    close_backlog_items(tmp_path, ids=["ibl-3"])
    assert load_backlog_items(tmp_path)[0]["status"] == "done"
    append_backlog_items(tmp_path, [item])  # recurs after being closed
    reopened = load_backlog_items(tmp_path)[0]
    assert reopened["status"] == "open"
    assert int(reopened["count"]) == 2
