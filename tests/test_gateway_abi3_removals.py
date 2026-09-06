"""ABI 7.0 (ABI-3): per-alias removal pins for the five gateway compat aliases.

One test class per alias (F11 axes: declaration / producer / stored tolerance /
migration surface), per docs/v7next/ABI3_GATEWAY_ALIAS_INVENTORY.md. These pins
are the REMOVAL side; the read-tolerance side lives in
tests/test_cost_projection.py and the endpoint behavior in
tests/test_ui_preferences_api.py / tests/test_gateway_history.py.
"""

from __future__ import annotations

import json
from typing import get_type_hints

import pytest


class TestCostAliasRemoval:
    def test_declaration_gone_from_chat_outbound(self):
        from ouroboros.gateway.contracts import ChatOutbound

        hints = set(get_type_hints(ChatOutbound, include_extras=True))
        assert "cost_usd" not in hints
        assert "cost_usd_with_children" not in hints
        assert {"accounted_upper_bound_usd",
                "accounted_upper_bound_usd_with_children"} <= hints

    def test_ssot_emitters_never_emit_the_alias(self):
        from ouroboros.cost_projection import (
            carry_cost_meta,
            cost_projection,
            with_cost_aliases,
        )

        legacy_source = {"cost_usd": 1.0, "cost_usd_with_children": 2.0,
                         "cost_final": True}
        for out in (with_cost_aliases(legacy_source),
                    carry_cost_meta(legacy_source),
                    cost_projection(legacy_source)):
            assert "cost_usd" not in out and "cost_usd_with_children" not in out
            assert out["accounted_upper_bound_usd"] == 1.0

    def test_live_root_projection_emits_honest_names_only(self, tmp_path):
        from ouroboros.cost_projection import live_root_cost_projection

        out = live_root_cost_projection(
            "t1", {"metadata": {}}, {}, tmp_path)
        # Root with an empty ledger returns {}; a non-root returns {} — either
        # way the alias never appears. Exercise the unavailable branch too.
        assert "cost_usd" not in out and "cost_usd_with_children" not in out

    def test_admission_failure_record_stamps_the_honest_name(self):
        # gateway/tasks.py admission-failure producer switched off the alias.
        import inspect

        from ouroboros.gateway import tasks as gateway_tasks

        source = inspect.getsource(gateway_tasks)
        assert "cost_usd=0.0" not in source
        assert "accounted_upper_bound_usd=0.0" in source

    def test_stored_legacy_record_still_reads(self, tmp_path):
        from ouroboros.cost_projection import cost_projection
        from ouroboros.task_results import load_task_result, write_task_result

        # A record authored by an older release (raw legacy field passthrough).
        write_task_result(tmp_path, "legacy", "completed",
                          cost_usd=1.25, cost_final=True)
        stored = load_task_result(tmp_path, "legacy")
        assert cost_projection(stored)["accounted_upper_bound_usd"] == 1.25


class TestTelegramChatIdRemoval:
    def test_declaration_gone_from_all_outbound_frames(self):
        from ouroboros.gateway.contracts import (
            ChatOutbound,
            DocumentOutbound,
            PhotoOutbound,
            VideoOutbound,
        )

        for cls in (ChatOutbound, PhotoOutbound, VideoOutbound, DocumentOutbound):
            hints = get_type_hints(cls, include_extras=True)
            assert "telegram_chat_id" not in hints, cls.__name__
            assert "transport" in hints, cls.__name__

    def test_no_runtime_producer_left(self):
        # The history mapper was the ONLY emitter; grep-level absence pin.
        import inspect

        from ouroboros.gateway import history as gateway_history

        source = inspect.getsource(gateway_history)
        assert '"telegram_chat_id": ' not in source

    def test_legacy_stored_row_replays_without_reemitting_the_key(self, tmp_path):
        from ouroboros.gateway.history import _collect_chat_rows

        chat = tmp_path / "logs" / "chat.jsonl"
        chat.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": "2026-01-01T00:00:00+00:00", "direction": "in",
            "text": "legacy row", "type": "", "telegram_chat_id": 5,
            "chat_id": 1,
        }
        chat.write_text(json.dumps(row) + "\n", encoding="utf-8")
        rows, _quota = _collect_chat_rows(
            chat, tmp_path / "logs" / "archive", 10,
            lambda entry_chat, entry=None: True, {})
        assert len(rows) == 1
        assert rows[0]["text"] == "legacy row"
        # Stored tolerance: the legacy key is read-and-ignored — the outbound
        # history record never re-emits it (ABI-3), and replay is not rejected.
        assert "telegram_chat_id" not in rows[0]


class TestUiPreferenceAliasRemoval:
    def test_declaration_gone_from_response_contract(self):
        from ouroboros.gateway.contracts import UiPreferencesResponse

        hints = set(get_type_hints(UiPreferencesResponse, include_extras=True))
        assert "project_last_viewed" not in hints
        assert "project_hidden" not in hints
        assert "project_seen_revision" in hints

    def test_defaults_and_known_keys_dropped_the_aliases(self):
        from ouroboros.gateway.ui_preferences import DEFAULT_UI_PREFERENCES

        assert "project_last_viewed" not in DEFAULT_UI_PREFERENCES
        assert "project_hidden" not in DEFAULT_UI_PREFERENCES

    def test_stored_legacy_keys_are_ignored_not_fatal(self, tmp_path):
        from ouroboros.gateway.ui_preferences import _normalize_preferences

        prefs = _normalize_preferences({
            "widget_order": ["a"],
            "project_last_viewed": {"p": "2026-01-01T00:00:00Z"},
            "project_hidden": {"p": True},
        })
        assert prefs["widget_order"] == ["a"]
        assert "project_last_viewed" not in prefs
        assert "project_hidden" not in prefs


class TestAliasProducerFanOutSweep:
    """Ф3.1 fix-round-2: fan-out-complete producer pin over the WHOLE runtime
    tree (every ``ouroboros/**/*.py`` and ``supervisor/**/*.py``).

    No production code emits a retired gateway alias key in an emission-shaped
    AST position — a dict-literal key, a subscript assignment, a keyword
    argument on ANY call (receipt constructors, event emitters, ``dict()``
    builders), or anything on a ``write_task_result`` call (the durable ABI-3
    store; NO allowlist there). Legacy READS stay legal and are naturally
    invisible to this scan: ``resolve_cost_pair``/``.get``/``in``/``.pop``
    never author a key. Generic passthrough (projections, deep copies) is
    structurally invisible to ANY static scan — the projection-boundary
    runtime pin (``TestProjectionBoundaryNormalization``) covers that shape
    by feeding stored legacy bytes through the outbound projections.

    The allowlist is PER-SITE and COUNT-ANCHORED (fix-round-3):
    ``(posix path, alias, enclosing scope) -> (reason, emission count)``. A
    new emission in an already-allowlisted FILE but a different function is
    NOT allowlisted; a NEW emission inside an already-allowlisted FUNCTION
    breaks its count anchor and fails; a stale row (no emission matches it)
    FAILS the test — the list can only shrink honestly. Every surviving row
    is an INTERNAL non-gateway plane that merely shares the spelling:
    physical usage-ledger rows, llm/usage observability events (converted to
    the honest name at the /api/logs projection boundary on replay),
    review/evidence receipt schemas, evolution checkpoint records, custody
    settlement events, reflection/consciousness records. ``outcomes.py``
    (loop-outcome usage snapshot) and the subagent envelope are deliberately
    GONE from this list (fix-round-2), and the evolution campaign HISTORY
    row producer is gone since fix-round-3 (it reaches /api/state, so it
    stamps the honest name and the state projection boundary normalizes
    stored legacy rows).
    """

    RETIRED_ALIASES = frozenset({
        "cost_usd", "cost_usd_with_children", "telegram_chat_id",
        "project_last_viewed", "project_hidden",
    })
    # (posix path, alias, enclosing scope) -> (why this INTERNAL plane
    # legitimately keeps the spelling, exact emission count at these sites).
    INTERNAL_PLANE_ALLOWLIST = {
        # physical usage ledger rows / legacy usage import (P7 monetary authority)
        ("ouroboros/usage_accounting.py", "cost_usd", "record_unmetered_external_dispatch"): ("ledger unmetered dispatch row", 1),
        ("ouroboros/usage_accounting.py", "cost_usd", "record_subscription_session"): ("ledger subscription session row", 1),
        ("ouroboros/usage_accounting.py", "cost_usd", "terminalize_abandoned_attempt"): ("ledger settlement transition", 1),
        ("ouroboros/usage_accounting.py", "cost_usd", "settle_attempt"): ("ledger settlement transition", 1),
        ("ouroboros/usage_accounting.py", "cost_usd", "_terminalize_failed_attempt"): ("ledger settlement transition", 2),
        ("ouroboros/usage_accounting.py", "cost_usd", "execute_physical_attempt"): ("ledger settlement call", 1),
        ("ouroboros/usage_accounting.py", "cost_usd", "execute_physical_attempt_async"): ("ledger settlement call", 1),
        ("ouroboros/usage_legacy_import.py", "cost_usd", "_ensure_legacy_imported_locked"): ("legacy usage.json ledger import rows", 2),
        ("ouroboros/usage_compaction.py", "cost_usd", "_build_candidate"): ("ledger compaction baseline-group row (CPL4-C6; exact-decimal string sum)", 1),
        ("ouroboros/tools/search.py", "cost_usd", "_web_search"): ("ledger settlement call (web search attempt)", 1),
        # usage/observability event streams (events.jsonl, live log frames;
        # /api/logs replay converts to the honest name at the boundary)
        ("ouroboros/loop_llm_call.py", "cost_usd", "call_llm_with_retry"): ("llm_round usage event rows", 2),
        ("ouroboros/post_task_synthesis.py", "cost_usd", "_run_chat_consolidation"): ("chat_block_consolidation event row", 1),
        ("ouroboros/post_task_synthesis.py", "cost_usd", "_run_reflection"): ("reflection generation gate args", 1),
        ("ouroboros/consciousness.py", "cost_usd", "_think_scoped"): ("consciousness thought receipt row", 2),
        ("supervisor/events_evolution_done.py", "cost_usd", "_handle_evolution_task_done"): ("internal lifecycle/checkpoint call kwargs + supervisor.jsonl observability row", 3),
        # review/evidence receipt schemas (internal review plane)
        ("ouroboros/triad_review.py", "cost_usd", "to_dict"): ("triad review receipt serialization", 1),
        ("ouroboros/triad_review.py", "cost_usd", "_actor_record"): ("triad review actor record", 1),
        ("ouroboros/skill_loader.py", "cost_usd", "to_dict"): ("skill review outcome receipt serialization", 1),
        ("ouroboros/skill_loader.py", "cost_usd", "load_review_state"): ("skill review state load", 1),
        ("ouroboros/skill_review.py", "cost_usd", "_run_deterministic_preflight"): ("skill review preflight receipt", 1),
        ("ouroboros/skill_review.py", "cost_usd", "_persist_reviewed_outcome"): ("skill review outcome receipt", 1),
        ("ouroboros/skill_owner_attestation.py", "cost_usd", "run_owner_attestation"): ("owner attestation review receipt", 1),
        ("ouroboros/tools/claude_advisory_review.py", "cost_usd", "_run_advisory_native"): ("advisory review receipt", 1),
        ("ouroboros/tools/delegate_terminal_evidence.py", "cost_usd", "_reported_cost"): ("delegate terminal evidence rows", 4),
        ("ouroboros/tools/preflight_review_run.py", "cost_usd", "_llm_extract_advisory_items"): ("advisory preflight usage receipt", 1),
        ("ouroboros/tools/preflight_review_run.py", "cost_usd", "_advisory_failure"): ("internal advisory failure adapter; physical charges remain in usage/custody, not gateway fields", 1),
        ("ouroboros/tools/preflight_review_run.py", "cost_usd", "_run_advisory_delegated"): ("advisory preflight receipt", 1),
        ("ouroboros/tools/preflight_review_run.py", "cost_usd", "_run_claude_advisory"): ("advisory preflight receipt", 4),
        ("ouroboros/tools/review_admission.py", "cost_usd", "triad_not_dispatched_records"): ("review admission receipt", 1),
        ("ouroboros/tools/review_helpers.py", "cost_usd", "build_scope_actor_record"): ("review usage receipt", 1),
        ("ouroboros/tools/scope_review.py", "cost_usd", "_scope_oversize_result"): ("scope review receipt", 1),
        ("ouroboros/tools/scope_review.py", "cost_usd", "run_scope_review"): ("scope review receipt", 6),
        ("ouroboros/tools/parallel_review.py", "cost_usd", "_run_scope"): ("scope review receipt", 1),
        # evolution checkpoint plane (durable state files, never a gateway payload;
        # the campaign HISTORY row producer left this list in fix-round-3)
        ("ouroboros/evolution_checkpoints.py", "cost_usd", "build_solve_capability_digest"): ("evolution capability digest", 1),
        ("ouroboros/evolution_checkpoints.py", "cost_usd", "append_evolution_checkpoint"): ("evolution checkpoint records", 1),
        # custody settlement events
        ("ouroboros/delegate_custody.py", "cost_usd", "settle_run"): ("custody SETTLED event row", 1),
        # reflection records
        ("ouroboros/reflection.py", "cost_usd", "generate_reflection"): ("task reflection record", 1),
    }

    @staticmethod
    def _emission_hits():
        import ast
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parents[1]
        aliases = TestAliasProducerFanOutSweep.RETIRED_ALIASES
        dict_hits: list = []
        writer_kwarg_hits: list = []

        def visit(node, rel, scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope = node.name
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and key.value in aliases:
                        dict_hits.append((rel, key.value, scope, key.lineno))
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value in aliases
                    ):
                        dict_hits.append((rel, target.slice.value, scope, target.lineno))
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                for kw in node.keywords:
                    if kw.arg in aliases:
                        if name == "write_task_result":
                            writer_kwarg_hits.append((rel, kw.arg, scope, node.lineno))
                        else:
                            dict_hits.append((rel, kw.arg, scope, node.lineno))
                if name == "write_task_result":
                    for arg in [kw.value for kw in node.keywords if kw.arg is None] + list(node.args):
                        if isinstance(arg, ast.Dict):
                            for key in arg.keys:
                                if isinstance(key, ast.Constant) and key.value in aliases:
                                    writer_kwarg_hits.append((rel, key.value, scope, node.lineno))
            for child in ast.iter_child_nodes(node):
                visit(child, rel, scope)

        for package in ("ouroboros", "supervisor"):
            for path in sorted((repo_root / package).rglob("*.py")):
                rel = path.relative_to(repo_root).as_posix()
                visit(ast.parse(path.read_text(encoding="utf-8")), rel, "<module>")
        return dict_hits, writer_kwarg_hits

    def test_no_task_result_writer_passes_a_retired_alias(self):
        _, writer_kwarg_hits = self._emission_hits()
        assert writer_kwarg_hits == [], (
            "write_task_result call sites must stamp honest names only "
            f"(ABI-3); offending sites: {writer_kwarg_hits!r}"
        )

    def test_every_alias_key_emission_is_an_allowlisted_internal_plane(self):
        import collections

        dict_hits, _ = self._emission_hits()
        counts = collections.Counter(
            (rel, alias, scope) for rel, alias, scope, _lineno in dict_hits
        )
        unexpected = [
            hit for hit in dict_hits
            if (hit[0], hit[1], hit[2]) not in self.INTERNAL_PLANE_ALLOWLIST
        ]
        assert unexpected == [], (
            "new emission-shaped occurrence of a retired gateway alias; either "
            "cut the producer over to the honest name or (only for a genuinely "
            "internal non-gateway plane) add a PER-SITE allowlist row: "
            f"{unexpected!r}"
        )
        drifted = {
            site: {"actual": count, "anchored": self.INTERNAL_PLANE_ALLOWLIST[site][1]}
            for site, count in sorted(counts.items())
            if site in self.INTERNAL_PLANE_ALLOWLIST
            and count != self.INTERNAL_PLANE_ALLOWLIST[site][1]
        }
        assert drifted == {}, (
            "emission-count drift inside an allowlisted function — a NEW "
            "emission in an already-allowlisted scope is NOT allowlisted "
            f"(fix-round-3 anchor): {drifted!r}"
        )
        stale = sorted(set(self.INTERNAL_PLANE_ALLOWLIST) - set(counts))
        assert stale == [], (
            f"stale allowlist rows (no emission matches them any more): {stale!r}"
        )

    def test_no_gateway_alias_survives_outside_the_cost_pair(self):
        """The three non-cost aliases have zero emission-shaped occurrences at
        all — no allowlist, no exceptions."""
        dict_hits, _ = self._emission_hits()
        non_cost = [
            hit for hit in dict_hits
            if hit[1] in {"telegram_chat_id", "project_last_viewed", "project_hidden"}
        ]
        assert non_cost == []

    def test_the_public_projection_planes_are_not_allowlisted(self):
        """Fix-round-2 pin: no allowlist row may name a plane whose data
        reaches the public task-result projection — the loop-outcome usage
        snapshot, the subagent envelope, or the projection module itself."""
        banned_files = {
            "ouroboros/outcomes.py", "ouroboros/subagents.py",
            "ouroboros/agent_task_pipeline.py",
        }
        offending = sorted(
            row for row in self.INTERNAL_PLANE_ALLOWLIST if row[0] in banned_files
        )
        assert offending == []


def _retired_alias_paths_deep(payload):
    """Every path in *payload* whose KEY is a retired cost alias, recursively.

    This is the generic-passthrough catcher no AST sweep can be: it inspects
    the actual outbound bytes, so a projection or deep copy that carried a
    stored legacy spelling through shows up regardless of how the code
    spelled the copy."""
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"cost_usd", "cost_usd_with_children"}:
                    found.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload, "$")
    return found


class TestProjectionBoundaryNormalization:
    """Ф3.1 fix-round-2: the ABI carries NO alias — outbound payloads built
    from STORED LEGACY rows contain only the honest cost names.

    Stored tolerance is read-side only: the legacy spelling resolves
    (deprecated-wins) and is NORMALIZED at the projection boundary
    (public_task_result / task detail / the list row) and at re-write
    (write_task_result strips aliases from the merged existing row)."""

    LEGACY_ROW = {
        "_schema_version": 1,
        "task_id": "legacy-cost",
        "status": "completed",
        "result": "done",
        "ts": "2026-01-01T00:00:00Z",
        "cost_usd": 1.5,
        "cost_usd_with_children": 2.75,
        "cost_final": True,
        "cost_accounting_status": "available",
        "subagent_envelope": {
            "task_id": "legacy-cost",
            "status": "completed",
            # Fix-round-3: the alias sits on the ACTUALLY SUPPORTED producer
            # path (build_subagent_envelope embeds the stored usage snapshot)
            # — envelope.usage.cost_usd — beside the envelope-root spelling.
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "rounds": 3,
                      "cost_usd": 0.125},
            "cost_usd": 0.25,
        },
        "loop_outcome": {
            "reason_code": "final_answer",
            "usage": {"cost_usd": 0.5, "prompt_tokens": 1,
                      "completion_tokens": 2, "total_rounds": 3},
        },
    }

    def _write_legacy(self, data):
        results = data / "task_results"
        results.mkdir(parents=True, exist_ok=True)
        (results / "legacy-cost.json").write_text(
            json.dumps(self.LEGACY_ROW), encoding="utf-8"
        )

    def test_public_task_result_emits_honest_names_only(self):
        from ouroboros.outcomes import public_task_result

        out = public_task_result(dict(self.LEGACY_ROW))
        assert _retired_alias_paths_deep(out) == []
        assert out["accounted_upper_bound_usd"] == 1.5
        assert out["accounted_upper_bound_usd_with_children"] == 2.75
        assert out["cost_final"] is True
        assert out["subagent_envelope"]["accounted_upper_bound_usd"] == 0.25
        assert out["subagent_envelope"]["usage"]["accounted_upper_bound_usd"] == 0.125
        assert out["loop_outcome"]["usage"]["accounted_upper_bound_usd"] == 0.5
        # And the stored input was not mutated (projection, not conversion).
        assert self.LEGACY_ROW["cost_usd"] == 1.5
        assert self.LEGACY_ROW["subagent_envelope"]["usage"]["cost_usd"] == 0.125

    def test_task_detail_and_list_row_emit_honest_names_only(self, tmp_path):
        import asyncio
        from types import SimpleNamespace

        from ouroboros.gateway.tasks import api_task_get, api_tasks_list

        data = tmp_path / "data"
        self._write_legacy(data)
        detail_request = SimpleNamespace(
            path_params={"task_id": "legacy-cost"},
            app=SimpleNamespace(state=SimpleNamespace(drive_root=data)),
        )
        detail = json.loads(
            asyncio.run(api_task_get(detail_request)).body.decode("utf-8")
        )
        assert _retired_alias_paths_deep(detail) == []
        assert detail["accounted_upper_bound_usd"] == 1.5
        assert detail["subagent_envelope"]["accounted_upper_bound_usd"] == 0.25
        assert detail["subagent_envelope"]["usage"]["accounted_upper_bound_usd"] == 0.125
        assert detail["loop_outcome"]["usage"]["accounted_upper_bound_usd"] == 0.5

        list_request = SimpleNamespace(
            query_params={},
            path_params={},
            app=SimpleNamespace(state=SimpleNamespace(drive_root=data)),
        )
        payload = json.loads(
            asyncio.run(api_tasks_list(list_request)).body.decode("utf-8")
        )
        assert _retired_alias_paths_deep(payload) == []
        row = next(r for r in payload["tasks"] if r["task_id"] == "legacy-cost")
        assert row["accounted_upper_bound_usd"] == 1.5
        assert row["accounted_upper_bound_usd_with_children"] == 2.75

    def test_rewrite_normalizes_the_stored_row_to_honest_names(self, tmp_path):
        """write_task_result strips aliases from the merged EXISTING row after
        deprecated-wins, and a fresh honest write is never outranked by the
        stored legacy spelling."""
        from ouroboros.task_results import load_task_result, write_task_result

        results = tmp_path / "task_results"
        results.mkdir(parents=True, exist_ok=True)
        (results / "legacy-rw.json").write_text(json.dumps({
            "_schema_version": 1, "task_id": "legacy-rw", "status": "running",
            "ts": "2026-01-01T00:00:00Z", "cost_usd": 1.0, "cost_final": False,
        }), encoding="utf-8")

        write_task_result(tmp_path, "legacy-rw", "completed",
                          accounted_upper_bound_usd=7.0, cost_final=True)
        stored = load_task_result(tmp_path, "legacy-rw")
        assert "cost_usd" not in stored and "cost_usd_with_children" not in stored
        assert stored["accounted_upper_bound_usd"] == 7.0
        assert stored["cost_final"] is True

    def test_rewrite_honors_a_legacy_mutators_edit_then_strips_it(self, tmp_path):
        """A legacy spelling arriving IN the write itself still wins the pair
        (deprecated-wins: the mutator's edit is honored) but is persisted
        under the honest name only."""
        from ouroboros.task_results import load_task_result, write_task_result

        write_task_result(tmp_path, "legacy-mut", "running",
                          accounted_upper_bound_usd=1.0)
        write_task_result(tmp_path, "legacy-mut", "completed",
                          **{"cost_usd": 9.0})
        stored = load_task_result(tmp_path, "legacy-mut")
        assert "cost_usd" not in stored
        assert stored["accounted_upper_bound_usd"] == 9.0

    def test_rewrite_normalizes_the_nested_public_cost_planes(self, tmp_path):
        """Fix-round-3: write_task_result's rewrite normalizes the KNOWN
        nested public planes too — a stored legacy subagent envelope (root
        and usage) and loop-outcome usage leave the next persisted row under
        honest names only."""
        from ouroboros.task_results import load_task_result, write_task_result

        results = tmp_path / "task_results"
        results.mkdir(parents=True, exist_ok=True)
        (results / "legacy-deep.json").write_text(
            json.dumps(self.LEGACY_ROW).replace("legacy-cost", "legacy-deep"),
            encoding="utf-8",
        )
        write_task_result(tmp_path, "legacy-deep", "completed", result="rewritten")
        stored = load_task_result(tmp_path, "legacy-deep")
        assert _retired_alias_paths_deep(stored) == []
        assert stored["subagent_envelope"]["accounted_upper_bound_usd"] == 0.25
        assert stored["subagent_envelope"]["usage"]["accounted_upper_bound_usd"] == 0.125
        assert stored["loop_outcome"]["usage"]["accounted_upper_bound_usd"] == 0.5

    def test_envelope_builder_normalizes_the_stored_usage_snapshot(self):
        """Fix-round-3: build_subagent_envelope normalizes the stored usage
        snapshot BEFORE embedding it — the envelope's usage plane carries the
        honest name only, and the legacy amount still resolves into the
        envelope's own accounted_upper_bound_usd."""
        from ouroboros.subagents import build_subagent_envelope

        envelope = build_subagent_envelope(
            task_id="child-1",
            status="completed",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "cost_usd": 0.75},
        )
        assert _retired_alias_paths_deep(envelope) == []
        assert envelope["usage"]["accounted_upper_bound_usd"] == 0.75
        assert envelope["accounted_upper_bound_usd"] == 0.75


class TestEvolutionHistoryPlane:
    """Ф3.1 fix-round-3: the evolution campaign history row reaches the
    public ``/api/state`` payload — its producer stamps the honest name
    (``update_evolution_campaign_after_task``, sweep allowlist row removed)
    and the state projection boundary resolves STORED legacy rows
    deprecated-wins, emitting honest names only."""

    def test_state_projection_normalizes_stored_legacy_history_rows(self):
        from ouroboros.gateway.state import _evolution_state_public

        snapshot = {
            "enabled": True,
            "campaign": {
                "id": "c1",
                "history": [
                    {"task_id": "old", "cost_usd": 1.5,
                     "cost_accounting_status": "available"},
                    {"task_id": "new", "accounted_upper_bound_usd": 2.5,
                     "cost_accounting_status": "available"},
                ],
            },
        }
        out = _evolution_state_public(snapshot)
        assert _retired_alias_paths_deep(out) == []
        rows = {row["task_id"]: row for row in out["campaign"]["history"]}
        assert rows["old"]["accounted_upper_bound_usd"] == 1.5
        assert rows["new"]["accounted_upper_bound_usd"] == 2.5
        # Copy-on-write: the shared supervisor snapshot row is untouched.
        assert snapshot["campaign"]["history"][0]["cost_usd"] == 1.5

    def test_diverged_stored_pair_resolves_deprecated_wins(self):
        from ouroboros.gateway.state import _evolution_state_public

        out = _evolution_state_public({"campaign": {"history": [
            {"task_id": "t", "cost_usd": 9.0, "accounted_upper_bound_usd": 1.0},
        ]}})
        assert out["campaign"]["history"][0]["accounted_upper_bound_usd"] == 9.0
        assert "cost_usd" not in out["campaign"]["history"][0]


class TestApiV1ShimRemoval:
    def test_module_is_gone(self):
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("ouroboros.contracts.api_v1")

    def test_gateway_contracts_is_the_sole_ssot(self):
        from ouroboros.gateway import contracts

        assert "ChatOutbound" in contracts.__all__
        assert "HTTP_ENDPOINTS" in contracts.__all__
