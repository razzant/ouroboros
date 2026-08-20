"""Child-drive skill_payload resolution + OpenRouter-only budget drift.

Two regressions covered:

1. v6.70.0 granted read/list/search on ``root=skill_payload`` to read-only
   subagents, but the path layer still resolved payloads against the child's
   isolated drive (``data/state/headless_tasks/<tid>/data``), which has no
   ``skills/`` tree — every scout was blinded with a bare "Directory not
   found" (2026-07-21 anime_studio audit swarm). Resolution now targets the
   canonical data root while the verb matrix keeps children read-only.

2. ``budget_drift_alert`` compared the ALL-provider ledger delta against the
   single OpenRouter key's usage delta, so real direct-provider spend
   (Anthropic advisory) latched the alert at ~88% while nothing was wrong.
   Drift now compares the OpenRouter-only settled delta, rebaselines on a key
   change, and suppresses the comparison on a quarantined ledger.
"""
from __future__ import annotations

import json
import pathlib

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canonical_root(tmp_path: pathlib.Path, name: str = "data") -> pathlib.Path:
    canonical = tmp_path / name
    payload = canonical / "skills" / "ouroboroshub" / "alpha"
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text("---\nname: alpha\n---\nalpha payload marker\n", encoding="utf-8")
    (payload / "plugin.py").write_text("def generate_audio():\n    return 'alpha-audio'\n", encoding="utf-8")
    return canonical


def _make_child_drive(canonical: pathlib.Path, task_id: str = "child0001") -> pathlib.Path:
    child = canonical / "state" / "headless_tasks" / task_id / "data"
    for rel in ("memory", "logs", "state", "task_results"):
        (child / rel).mkdir(parents=True, exist_ok=True)
    return child


def _readonly_child_registry(
    repo: pathlib.Path,
    child_drive: pathlib.Path,
    canonical: pathlib.Path | None,
    *,
    metadata_only: bool = False,
) -> ToolRegistry:
    registry = ToolRegistry(repo_dir=repo, drive_root=child_drive)
    ctx_kwargs: dict = {}
    if canonical is not None and not metadata_only:
        ctx_kwargs["budget_drive_root"] = str(canonical)
    if canonical is not None and metadata_only:
        ctx_kwargs["task_metadata"] = {"budget_drive_root": str(canonical)}
    registry.set_context(
        ToolContext(
            repo_dir=repo,
            drive_root=child_drive,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
            **ctx_kwargs,
        )
    )
    return registry


# ---------------------------------------------------------------------------
# Fix 1: skill_payload resolves against the canonical data root
# ---------------------------------------------------------------------------

class TestChildDriveSkillPayload:
    def test_canonical_data_root_precedence(self, tmp_path):
        from ouroboros.tool_access import canonical_data_root

        class _Ctx:
            drive_root = tmp_path / "child"
            budget_drive_root = str(tmp_path / "canonical")
            task_metadata = {"budget_drive_root": str(tmp_path / "meta")}

        assert canonical_data_root(_Ctx()) == (tmp_path / "meta").resolve()

        class _CtxNoMeta:
            drive_root = tmp_path / "child"
            budget_drive_root = str(tmp_path / "canonical")
            task_metadata: dict = {}

        assert canonical_data_root(_CtxNoMeta()) == (tmp_path / "canonical").resolve()

        class _CtxBare:
            drive_root = tmp_path / "child"
            budget_drive_root = ""
            task_metadata: dict = {}

        assert canonical_data_root(_CtxBare()) == (tmp_path / "child").resolve()

    def test_readonly_child_reads_canonical_payload(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        canonical = _make_canonical_root(tmp_path)
        child = _make_child_drive(canonical)
        registry = _readonly_child_registry(repo, child, canonical)

        read = registry.execute("read_file", {
            "root": "skill_payload", "bucket": "ouroboroshub", "skill_name": "alpha",
            "path": "SKILL.md",
        })
        assert "alpha payload marker" in read

        listing = registry.execute("list_files", {
            "root": "skill_payload", "bucket": "ouroboroshub", "skill_name": "alpha",
            "path": ".",
        })
        assert "LIST_FILES_ERROR" not in listing
        rendered = json.dumps(json.loads(listing))
        assert "SKILL.md" in rendered and "plugin.py" in rendered

        search = registry.execute("search_code", {
            "root": "skill_payload", "bucket": "ouroboroshub", "skill_name": "alpha",
            "query": "generate_audio",
        })
        assert "SEARCH_ERROR" not in search
        assert "plugin.py" in search

    def test_metadata_only_canonical_root_fallback(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        canonical = _make_canonical_root(tmp_path)
        child = _make_child_drive(canonical)
        registry = _readonly_child_registry(repo, child, canonical, metadata_only=True)

        read = registry.execute("read_file", {
            "root": "skill_payload", "bucket": "ouroboroshub", "skill_name": "alpha",
            "path": "SKILL.md",
        })
        assert "alpha payload marker" in read

    def test_empty_budget_root_falls_back_to_drive_root(self, tmp_path):
        """A root task (drive_root == canonical) keeps its previous behavior."""
        repo = tmp_path / "repo"
        repo.mkdir()
        canonical = _make_canonical_root(tmp_path)
        registry = _readonly_child_registry(repo, canonical, None)

        read = registry.execute("read_file", {
            "root": "skill_payload", "bucket": "ouroboroshub", "skill_name": "alpha",
            "path": "SKILL.md",
        })
        assert "alpha payload marker" in read

    def test_no_cross_root_leak_between_isolated_runs(self, tmp_path):
        """A child bound to canonical root A must never read root B's payloads."""
        repo = tmp_path / "repo"
        repo.mkdir()
        root_a = _make_canonical_root(tmp_path / "runA")
        root_b = _make_canonical_root(tmp_path / "runB")
        (root_b / "skills" / "ouroboroshub" / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\n---\nROOT-B ONLY marker\n", encoding="utf-8"
        )
        child = _make_child_drive(root_a)
        registry = _readonly_child_registry(repo, child, root_a)

        read = registry.execute("read_file", {
            "root": "skill_payload", "bucket": "ouroboroshub", "skill_name": "alpha",
            "path": "SKILL.md",
        })
        assert "alpha payload marker" in read
        assert "ROOT-B ONLY" not in read

    def test_child_write_stays_denied(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        canonical = _make_canonical_root(tmp_path)
        child = _make_child_drive(canonical)
        registry = _readonly_child_registry(repo, child, canonical)

        # The tool is not even exposed to the readonly profile...
        assert registry.get_schema_by_name("write_file") is None
        # ...and a forced dispatch is refused before any handler runs.
        result = registry.execute("write_file", {
            "root": "skill_payload", "bucket": "ouroboroshub", "skill_name": "alpha",
            "path": "SKILL.md", "content": "pwned",
        })
        assert "⚠️" in result
        payload_file = canonical / "skills" / "ouroboroshub" / "alpha" / "SKILL.md"
        assert "pwned" not in payload_file.read_text(encoding="utf-8")

    def test_path_escape_still_blocked(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        canonical = _make_canonical_root(tmp_path)
        secret = canonical / "state" / "skills" / "alpha" / "grants.json"
        secret.parent.mkdir(parents=True, exist_ok=True)
        secret.write_text('{"secret": true}', encoding="utf-8")
        child = _make_child_drive(canonical)
        registry = _readonly_child_registry(repo, child, canonical)

        read = registry.execute("read_file", {
            "root": "skill_payload", "bucket": "ouroboroshub", "skill_name": "alpha",
            "path": "../../../state/skills/alpha/grants.json",
        })
        assert "secret" not in read.replace("⚠️", "")
        assert "⚠️" in read

    def test_native_bucket_stays_rejected(self, tmp_path):
        """resolve_skill_payload_target keeps native out of the data-plane
        resolver (native skills are git-tracked; read them via system_repo)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        canonical = _make_canonical_root(tmp_path)
        child = _make_child_drive(canonical)
        registry = _readonly_child_registry(repo, child, canonical)

        read = registry.execute("read_file", {
            "root": "skill_payload", "bucket": "native", "skill_name": "alpha",
            "path": "SKILL.md",
        })
        assert "⚠️" in read


# ---------------------------------------------------------------------------
# Fix 2: OpenRouter-only budget drift
# ---------------------------------------------------------------------------

class TestBudgetDriftOpenRouterOnly:
    def _setup_state(self, tmp_path, monkeypatch, *, or_settled: float, accounted: float,
                     calls: int = 50, integrity_degraded: bool = False):
        from supervisor import state as sup_state

        sup_state.init(tmp_path, total_budget_limit=0.0)
        monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key-1")

        breakdown = {
            "accounted_usd": accounted,
            "physical_calls": calls,
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "cached_tokens": 0,
            "settled_usd": accounted,
            "confirmed_usd": accounted,
            "estimated_usd": 0.0,
            "reserved_usd": 0.0,
            "unresolved_upper_bound_usd": 0.0,
            "unknown_unmetered": 0,
            "cost_final": True,
            "attempt_counts": {"settled": calls},
            "integrity_degraded": integrity_degraded,
            "by_provider": {"openrouter": {"settled_usd": or_settled}},
        }
        import ouroboros.usage_accounting as ua

        monkeypatch.setattr(ua, "ensure_legacy_imported", lambda *_a, **_k: None)
        monkeypatch.setattr(ua, "usage_breakdown", lambda *_a, **_k: dict(breakdown))
        return sup_state

    def _seed_session(self, sup_state, *, total_snap: float, or_snap: float):
        def _mut(st):
            st["session_total_snapshot"] = total_snap
            st["session_spent_snapshot"] = 0.0
            st["session_openrouter_settled_snapshot"] = or_snap
            st["session_openrouter_key_fp"] = sup_state._openrouter_key_fingerprint()
            st["openrouter_last_check_call"] = -1
        sup_state.update_state(_mut)

    def test_direct_provider_spend_does_not_alert(self, tmp_path, monkeypatch):
        """$100 direct-Anthropic + $30 OpenRouter vs $30 key usage → no drift."""
        sup_state = self._setup_state(tmp_path, monkeypatch, or_settled=30.0, accounted=130.0)
        self._seed_session(sup_state, total_snap=1000.0, or_snap=0.0)
        monkeypatch.setattr(
            sup_state, "check_openrouter_ground_truth",
            lambda: {"total_usd": 1030.0, "daily_usd": 30.0},
        )

        sup_state.update_budget_from_usage({})
        st = sup_state.load_state()
        assert st["budget_drift_alert"] is False
        assert st["budget_drift_pct"] is not None
        assert st["budget_drift_pct"] < 5.0
        assert st["openrouter_ledger_settled_usd"] == 30.0

    def test_real_openrouter_drift_still_alerts(self, tmp_path, monkeypatch):
        """Ledger says $40 OpenRouter, key says $100 → genuine drift fires."""
        sup_state = self._setup_state(tmp_path, monkeypatch, or_settled=40.0, accounted=40.0)
        self._seed_session(sup_state, total_snap=1000.0, or_snap=0.0)
        monkeypatch.setattr(
            sup_state, "check_openrouter_ground_truth",
            lambda: {"total_usd": 1100.0, "daily_usd": 100.0},
        )

        sup_state.update_budget_from_usage({})
        st = sup_state.load_state()
        assert st["budget_drift_alert"] is True
        events = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in events.splitlines() if line.strip()]
        drift = [r for r in rows if r.get("type") == "budget_drift_warning"]
        assert drift and drift[-1]["our_delta"] == 40.0 and drift[-1]["or_delta"] == 100.0
        assert "all_provider_delta" in drift[-1]

    def test_key_change_rebaselines_without_alert(self, tmp_path, monkeypatch):
        sup_state = self._setup_state(tmp_path, monkeypatch, or_settled=40.0, accounted=40.0)
        self._seed_session(sup_state, total_snap=1000.0, or_snap=0.0)
        monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key-2-DIFFERENT")
        monkeypatch.setattr(
            sup_state, "check_openrouter_ground_truth",
            lambda: {"total_usd": 5555.0, "daily_usd": 1.0},
        )

        sup_state.update_budget_from_usage({})
        st = sup_state.load_state()
        assert st["budget_drift_alert"] is False
        assert st["budget_drift_pct"] is None
        assert st["session_total_snapshot"] == 5555.0
        assert st["session_openrouter_settled_snapshot"] == 40.0
        assert st["session_openrouter_key_fp"] == sup_state._openrouter_key_fingerprint()

    def test_missing_snapshot_rebaselines(self, tmp_path, monkeypatch):
        """Pre-upgrade state (no OpenRouter-only snapshot) skips one cycle."""
        sup_state = self._setup_state(tmp_path, monkeypatch, or_settled=40.0, accounted=40.0)

        def _mut(st):
            st["session_total_snapshot"] = 1000.0
            st["session_spent_snapshot"] = 0.0
            st["session_openrouter_settled_snapshot"] = None
            st["session_openrouter_key_fp"] = ""
            st["openrouter_last_check_call"] = -1
        sup_state.update_state(_mut)
        monkeypatch.setattr(
            sup_state, "check_openrouter_ground_truth",
            lambda: {"total_usd": 1100.0, "daily_usd": 100.0},
        )

        sup_state.update_budget_from_usage({})
        st = sup_state.load_state()
        assert st["budget_drift_alert"] is False
        assert st["budget_drift_pct"] is None
        assert st["session_openrouter_settled_snapshot"] == 40.0

    def test_integrity_degraded_suppresses_comparison(self, tmp_path, monkeypatch):
        sup_state = self._setup_state(
            tmp_path, monkeypatch, or_settled=40.0, accounted=40.0, integrity_degraded=True,
        )
        self._seed_session(sup_state, total_snap=1000.0, or_snap=0.0)
        monkeypatch.setattr(
            sup_state, "check_openrouter_ground_truth",
            lambda: {"total_usd": 1100.0, "daily_usd": 100.0},
        )

        sup_state.update_budget_from_usage({})
        st = sup_state.load_state()
        assert st["budget_drift_alert"] is False
        assert st["budget_drift_pct"] is None

    def test_status_text_uses_openrouter_only_delta(self, tmp_path, monkeypatch):
        sup_state = self._setup_state(tmp_path, monkeypatch, or_settled=30.0, accounted=130.0)
        self._seed_session(sup_state, total_snap=1000.0, or_snap=0.0)
        monkeypatch.setattr(
            sup_state, "check_openrouter_ground_truth",
            lambda: {"total_usd": 1030.0, "daily_usd": 30.0},
        )
        sup_state.update_budget_from_usage({})

        text = sup_state.status_text(sup_state.load_state(), [], {})
        assert "budget_drift" in text
        assert "openrouter tracked: $30.00" in text
        # The all-provider delta ($130) must not masquerade as the tracked side.
        assert "tracked: $130.00" not in text
