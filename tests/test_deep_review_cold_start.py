"""Cold-start density probe of the packed deep self-review (owner decisions R60/R61).

Contract under test:
- a required set that does not fit under the COLD cap triggers exactly ONE bounded
  probe send on the exact model, ONE rebuild under the recalibrated cap, and the
  review then proceeds;
- a warm store (fresh exact-model witness) never probes: the refusal is the typed
  ``deep_self_review_pack_unfit`` whose text asks the owner to switch the row;
- a failed probe leaves the cold cap standing and yields the same typed refusal;
- the refusal never falls back to a retrieving delivery;
- an exact-model witness stays authoritative for 90 days.
"""
from __future__ import annotations

import datetime
from unittest import mock

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.deep_self_review import run_deep_self_review
from ouroboros.outcomes import REASON_DEEP_SELF_REVIEW_PACK_UNFIT
from ouroboros.reviewer_slot_config import DEEP_REVIEW_SLOT_ID, ConfiguredReviewerSlot


def _packed_row(model: str) -> ConfiguredReviewerSlot:
    return ConfiguredReviewerSlot(slot_id=DEEP_REVIEW_SLOT_ID, kind="api_chat", target_id=model)


def _unfit(limit: int) -> tuple[str, dict]:
    manifest = {
        "status": "budget_omitted",
        "unassembled_required": [{"path": "docs/ARCHITECTURE.md", "disposition": "budget_omitted",
                                  "reason": "required file does not fit", "tokens": 109_000}],
        "selected": [{"path": "BIBLE.md", "disposition": "full", "tokens": 12_000}],
    }
    return "", {"file_count": 0, "total_chars": 0,
                "skipped": [f"FATAL: required artifact could not be assembled (cap {limit})"],
                "context_manifest": manifest}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "ARCHITECTURE.md").write_text("# Architecture\n" + ("architecture prose. " * 3000), encoding="utf-8")
    (root / "BIBLE.md").write_text("# BIBLE\n" + ("principle text. " * 2000), encoding="utf-8")
    return root


@pytest.fixture
def drive(tmp_path):
    root = tmp_path / "drive"
    (root / "state").mkdir(parents=True)
    (root / "memory").mkdir()
    return root


def _run(repo, drive, model, llm, build, chat, monkeypatch, *, window_ok=True):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    progress: list[str] = []
    with (
        mock.patch("ouroboros.deep_self_review.build_review_pack", side_effect=build),
        mock.patch("ouroboros.llm_observability.chat_observed", side_effect=chat),
        mock.patch("ouroboros.deep_self_review._run_retrieving_review") as retrieving,
    ):
        text, usage = run_deep_self_review(
            repo_dir=repo, drive_root=drive, llm=llm, emit_progress=lambda text, *, incident=None: progress.append(text), slot=_packed_row(model),
        )
    return text, usage, progress, retrieving


def test_cold_store_probes_once_then_rebuilds_and_proceeds(repo, drive, monkeypatch):
    model = "probe/cold-model-a"
    limits: list[int] = []
    calls: list[dict] = []

    def build(repo_dir, drive_root, fixed_prompt_tokens=0, hard_budget_reduction=0, input_token_limit=0):
        limits.append(input_token_limit)
        if len(limits) == 1:
            return _unfit(input_token_limit)
        return "y" * 4_000, {"file_count": 5, "total_chars": 4_000, "skipped": []}

    def chat(llm, **kwargs):
        calls.append(kwargs)
        if kwargs.get("call_type") == "deep_self_review_density_probe":
            chars = sum(len(m["content"]) for m in kwargs["messages"])
            return {"content": "OK"}, {"prompt_tokens": int(chars / 4 * 0.9), "cost": 0.0}
        return {"content": "Review result."}, {"cost": 0.0}

    text, usage, progress, retrieving = _run(repo, drive, model, mock.Mock(), build, chat, monkeypatch)

    assert text.endswith("\n\nReview result."), text
    assert [c.get("call_type") for c in calls] == ["deep_self_review_density_probe", "deep_self_review"]
    probe = calls[0]
    assert probe["max_tokens"] == 256 and probe["reasoning_effort"] == "low"
    assert probe["model"] == model and probe["tools"] is None
    assert len(limits) == 2 and limits[1] > limits[0], limits
    density, source = ce.resolve_review_token_density(drive, model)
    assert source == "measured" and density < ce.COLD_START_TOKEN_DENSITY
    assert not retrieving.called


def test_warm_store_never_probes_and_refuses_typed(repo, drive, monkeypatch):
    model = "probe/warm-model-b"
    ce.record_token_density(drive, model, prompt_chars=400_000, prompt_tokens=95_000, source="dispatch_usage")
    calls: list[dict] = []

    def build(repo_dir, drive_root, fixed_prompt_tokens=0, hard_budget_reduction=0, input_token_limit=0):
        return _unfit(input_token_limit)

    def chat(llm, **kwargs):
        calls.append(kwargs)
        return {"content": "OK"}, {"prompt_tokens": 1, "cost": 0.0}

    text, usage, progress, retrieving = _run(repo, drive, model, mock.Mock(), build, chat, monkeypatch)

    assert calls == [], "a warm store must not spend a probe"
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == REASON_DEEP_SELF_REVIEW_PACK_UNFIT == "deep_self_review_pack_unfit"
    assert "pack unfit" in text and "switch the `deep_review` reviewer row" in text
    assert "No automatic fallback runs" in text and "token density" in text and "measured" in text
    assert not retrieving.called


def test_failed_probe_keeps_the_cold_cap_and_refuses_typed(repo, drive, monkeypatch):
    model = "probe/cold-model-c"
    limits: list[int] = []

    def build(repo_dir, drive_root, fixed_prompt_tokens=0, hard_budget_reduction=0, input_token_limit=0):
        limits.append(input_token_limit)
        return _unfit(input_token_limit)

    def chat(llm, **kwargs):
        raise RuntimeError("provider down")

    text, usage, progress, retrieving = _run(repo, drive, model, mock.Mock(), build, chat, monkeypatch)

    assert len(limits) == 1, "no rebuild without a new witness"
    assert usage["reason_code"] == REASON_DEEP_SELF_REVIEW_PACK_UNFIT
    assert "cold_conservative" in text and any("probe failed" in p for p in progress)
    assert not retrieving.called


def test_fitting_pack_never_probes(repo, drive, monkeypatch):
    model = "probe/cold-model-d"
    calls: list[dict] = []

    def build(repo_dir, drive_root, fixed_prompt_tokens=0, hard_budget_reduction=0, input_token_limit=0):
        return "y" * 4_000, {"file_count": 5, "total_chars": 4_000, "skipped": []}

    def chat(llm, **kwargs):
        calls.append(kwargs)
        return {"content": "Review result."}, {"cost": 0.0}

    text, usage, progress, retrieving = _run(repo, drive, model, mock.Mock(), build, chat, monkeypatch)

    assert [c.get("call_type") for c in calls] == ["deep_self_review"]
    assert text.endswith("\n\nReview result.")


def test_exact_model_witness_governs_for_ninety_days(drive):
    model = "probe/idle-model-e"
    thirty_days_ago = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)).isoformat()
    ce.record_token_density(drive, model, prompt_chars=400_000, prompt_tokens=95_000, source="dispatch_usage")
    store = ce._load(drive)
    store["token_density"][model]["pairs"][0]["observed_at"] = thirty_days_ago
    assert ce._save(drive, store)
    density, source = ce.resolve_review_token_density(drive, model)
    assert source == "measured" and density == pytest.approx(0.95 * ce.MEASURED_DENSITY_SAFETY_FACTOR, rel=1e-6)
    assert ce._TOKEN_DENSITY_TTL_SEC == 90 * 24 * 3600.0


def test_probe_without_usage_leaves_the_cold_cap_and_says_so(repo, drive, monkeypatch):
    model = "probe/cold-model-f"
    limits: list[int] = []

    def build(repo_dir, drive_root, fixed_prompt_tokens=0, hard_budget_reduction=0, input_token_limit=0):
        limits.append(input_token_limit)
        return _unfit(input_token_limit)

    def chat(llm, **kwargs):
        return {"content": None}, {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    text, usage, progress, retrieving = _run(repo, drive, model, mock.Mock(), build, chat, monkeypatch)

    assert len(limits) == 1
    assert usage["reason_code"] == REASON_DEEP_SELF_REVIEW_PACK_UNFIT
    assert any("returned no usage" in p for p in progress)
    assert ce.resolve_review_token_density(drive, model)[1] == "cold_conservative"
    assert not retrieving.called
