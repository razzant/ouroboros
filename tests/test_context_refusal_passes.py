"""The refusal path sizes its model-free passes on this refusal and this conversation (17.10.2026).

86aa died on a 933k-token request the host had estimated at 678k with another task's density,
so no pass knew how much to shed. These pins keep the target measured on the conversation's
own anchor (unknown when nothing comparable measures it) and each pass shedding only what its
earlier steps left of that target: the gradual order the owner asked for, never a fixed share.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
import pytest

from ouroboros import loop_model_call as lmc
from ouroboros.context_anchor import REFUSED_INPUTS_KEY
from tests.test_context_economy_window import live_wait as wait_fixture
from tests.test_context_economy_window import main_call as main_fixture
from tests.test_context_economy_window import setup as gateway_fixture

setup, live_wait, main_call = gateway_fixture, wait_fixture, main_fixture  # the real Main call fixtures


def _fit(estimate, *, anchor=874_829, accepted=921_840, reply=65_536, stated_total=0):
    """``stated_total``: the round sent this request inside the reply reservation of that stated, unproved total."""
    return SimpleNamespace(measurement=SimpleNamespace(
        estimated_input_tokens=estimate, anchor_input_tokens=anchor, accepted_input_tokens=accepted,
        capacity_total_tokens=None, window_reading_tokens=None, reply_allowance_tokens=reply,
        capacity_deficit_tokens=None, probes_reply_reservation=bool(stated_total), window={"tokens": stated_total}))


def _ctx(message=""):
    return SimpleNamespace(accumulated_usage={"_last_llm_provider_message": message}, active_context_mode="max",
                           context_fit_plan=SimpleNamespace(provider="claudexor", model="codex=gpt-6-astra"))


def _target(ctx, fit, refused=None):
    return lmc._refusal_target(ctx, fit, refused)[0]


def _refused_size(ctx):
    (size,) = ctx.accumulated_usage[REFUSED_INPUTS_KEY].values()
    return size


def test_start_refusals_and_useful_answers_keep_their_own_size_and_order(tmp_path, monkeypatch):
    """A warm density is not a window. Raw witnesses can clear older equal/smaller refusals, not later ones."""
    from dataclasses import asdict, replace
    from datetime import timedelta
    from ouroboros import context_window as cw
    from ouroboros import capability_evidence as ce
    from ouroboros import usage_accounting as ua
    from ouroboros.context_anchor import record_refused_input
    from ouroboros.response_limits import dispatch_target
    from tests.test_first_contact_seams import stamped

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    model, cap, r = "openai-compatible::mock-model", 65_536, "turn:round:1"
    target = dispatch_target(model, {"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:1/v1",
                                    "OPENAI_COMPATIBLE_API_KEY": "fixture-not-a-credential"})
    scope = cw.first_contact_scope(target, cap)
    assert scope
    plan = SimpleNamespace(route_fp="r", window_witness_bound=True, provider="openai-compatible", model=model)
    physical = ua.PhysicalAttemptContext("owner_max", "max", "fresh_model_usage", "not-the-scope", r,
        None, None, False, False, first_contact_scope=scope, view_kind="owner_turn", preferred_mode="max",
        first_contact_plan=(scope, False))
    payload = {"model": target["resolved_model"], "messages": [{"role": "user", "content": "sample"}],
               "tools": [{"type": "function", "function": {"name": "inspect_fixture",
                          "parameters": {"type": "object", "properties": {}}}}]}
    assert cw.signed_first_contact(physical, target, payload, cap).first_contact_scope == scope
    assert not cw.signed_first_contact(physical, target, {**payload, "tools": []}, cap).first_contact_scope
    for ineligible in ({**target, "provider": "local"}, {**target, "provider": "claudexor"},
                       {**target, "provider": "gigachat"}, {**target, "api_key": ""},
                       {**target, "provider": "openai", "api_surface": "responses"}):
        assert not cw.signed_first_contact(physical, ineligible, payload, cap).first_contact_scope
    image = {**payload, "messages": [{"role": "user", "content": [{"type": "image_url"}]}]}
    assert not cw.signed_first_contact(physical, target, image, cap).first_contact_scope
    assert not any(name.startswith("first_contact_") for name in asdict(physical))

    def capture(order, raw, status=None):
        return ua.PhysicalAttemptCapture(str(order), model, plan.provider, "settled", "canonical_json_v1",
            max_completion_tokens=cap, physical_context=replace(physical, first_contact_dispatch=(order, ce.utc_now_iso())),
            candidate_manifest_ref={"fixture": True}, prompt_tokens_bounded_estimate=raw, provider_status_code=status)

    def refuse(candidate, round_id=r):
        usage = {"_last_llm_failure": {"physical_attempt_id": candidate.attempt_id, "failure_code": "context_overflow"},
                 **stamped()}
        record_refused_input(usage, plan, candidate.prompt_tokens_bounded_estimate + 11_000,
                             measured=True, candidate=candidate, round_id=round_id)

    def answer(candidate, **fields):
        cw.record_start_answer({"execution_id": "turn", "_last_llm_call_meta": {
            "physical_attempt_id": candidate.attempt_id, "round_id": r, **fields}}, candidate, 1)

    f0, f1, ok = capture(1, 548_000, 400), capture(3, 410_000, 400), capture(4, 420_000)
    refuse(f0)
    refuse(f1)
    answer(capture(2, 420_000))  # older than F1, although it publishes later
    assert cw.first_contact_limits(target, cap) == (410_000, 0, 410_000)
    answer(ok, provider_body_error=True)
    assert cw.first_contact_limits(target, cap) == (410_000, 0, 410_000)
    answer(ok)
    assert cw.first_contact_limits(target, cap) == (548_000, 0, 548_000)
    answer(capture(7, 410_000))  # no new 410k refusal has reached the store yet
    refuse(capture(6, 410_000, 400))  # older on the wire, despite publishing after the covering answer
    assert cw.first_contact_limits(target, cap) == (548_000, 0, 548_000)
    evidence = ce.CapabilityEvidence(0, "unprobeable", "none", "r")
    assert cw.resolve_window(evidence, cap, root=tmp_path).usable is None
    wrote = ce._store_path(tmp_path).read_bytes()
    refuse(f1)
    answer(ok)
    assert ce._store_path(tmp_path).read_bytes() == wrote  # replay neither teaches nor refreshes TTL
    assert cw.first_contact_limits(target, cap - 1) == (0, 0, 0)
    assert cw.first_contact_limits({**target, "api_key": "another-synthetic-key"}, cap) == (0, 0, 0)
    answer(capture(8, 548_000))  # same raw input suffices to disprove a refusal
    assert cw.first_contact_limits(target, cap) == (0, 0, 0)
    assert not cw.record_start_refusal({"_last_llm_failure": {"physical_attempt_id": "3",
        "failure_code": "context_overflow"}, **stamped()}, replace(f1, state="released"), r)
    later = capture(9, 550_000, 400)
    refuse(later)
    assert cw.first_contact_limits(target, cap) == (550_000, 0, 550_000)
    written, future = ce._store_path(tmp_path).read_bytes(), ce.utc_now() + timedelta(hours=25)
    monkeypatch.setattr(ce, "utc_now", lambda: future)
    refuse(later)  # exact replay after expiry supplies neither a fresh write nor a hint
    assert ce._store_path(tmp_path).read_bytes() == written
    assert cw.first_contact_limits(target, cap) == (0, 0, 0)


def test_a_saturated_start_log_still_proves_the_live_refusal(tmp_path, monkeypatch):
    """Traffic may bound the set, never erase a live 548k refusal nor revive one already answered (R5)."""
    from dataclasses import replace
    from ouroboros import context_window as cw
    from ouroboros import capability_evidence as ce
    from ouroboros import usage_accounting as ua
    from ouroboros.context_anchor import record_refused_input
    from ouroboros.response_limits import dispatch_target
    from tests.test_first_contact_seams import stamped

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    model, cap, r = "openai-compatible::mock-model", 65_536, "turn:round:1"
    target = dispatch_target(model, {"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:1/v1",
                                    "OPENAI_COMPATIBLE_API_KEY": "fixture-not-a-credential"})
    scope = cw.first_contact_scope(target, cap)
    plan = SimpleNamespace(route_fp="r", window_witness_bound=True, provider="openai-compatible", model=model)
    physical = ua.PhysicalAttemptContext("owner_max", "max", "fresh_model_usage", "not-the-scope", r,
        None, None, False, False, first_contact_scope=scope, view_kind="owner_turn", preferred_mode="max",
        first_contact_plan=(scope, False))

    def capture(order, raw, status=None):
        return ua.PhysicalAttemptCapture(str(order), model, plan.provider, "settled", "canonical_json_v1",
            max_completion_tokens=cap, physical_context=replace(physical, first_contact_dispatch=(order, ce.utc_now_iso())),
            candidate_manifest_ref={"fixture": True}, prompt_tokens_bounded_estimate=raw, provider_status_code=status)

    def refuse(order, raw):
        candidate = capture(order, raw, 400)
        usage = {"_last_llm_failure": {"physical_attempt_id": candidate.attempt_id, "failure_code": "context_overflow"},
                 **stamped()}
        record_refused_input(usage, plan, raw + 11_000, measured=True, candidate=candidate, round_id=r)

    def answer(order, raw):
        candidate = capture(order, raw)
        cw.record_start_answer({"execution_id": "turn", "_last_llm_call_meta": {
            "physical_attempt_id": candidate.attempt_id, "round_id": r}}, candidate, 1)

    def stored():
        return ce._read(tmp_path).get(cw._START_HINTS, {}).get(scope, [])

    refuse(1, 548_000)
    for index in range(20):
        refuse(2 + index, 100_000 + 1_000 * index)  # the churn a recency-only bound used to answer with
    assert cw.first_contact_limits(target, cap) == (100_000, 0, 100_000)
    answer(22, 420_000)  # a legitimate smaller start lands: it answers those refusals, not the 548k one
    assert cw.first_contact_limits(target, cap) == (548_000, 0, 548_000)
    assert len(stored()) <= 3 * cw._START_ROWS + 1
    for index in range(30):
        answer(24 + index, 90_000)  # unrelated traffic carries the 420k answer out of the set
    assert cw.first_contact_limits(target, cap) == (548_000, 0, 548_000)  # an answered refusal does not come back
    assert len(stored()) <= 3 * cw._START_ROWS + 1
    answer(54, 548_000)  # the refused size itself is proven acceptable on this same route
    assert cw.first_contact_limits(target, cap) == (0, 0, 0)


def test_the_bound_start_set_keeps_its_extrema_and_their_proofs():
    """The trimming rule alone: what the reader derives survives; replies past the newest sixteen fold into one cover
    that still answers what they answered, so an answered row never outlives its answer."""
    from ouroboros import context_window as cw

    def row(order, raw, *, answered=False, form="ordinary", rnd="r1"):
        return {"attempt": str(order), "order": order, "at": "2026-11-21T00:00:00+00:00", "round": rnd,
                "raw": raw, "form": form, "accepted": answered}

    big, rung, proof = row(1, 548_000), row(3, 500_000, rnd="r3"), row(4, 380_000, answered=True, form="maps", rnd="r3")
    lower, lower_proof = row(5, 470_000, rnd="r5"), row(6, 300_000, form="maps", rnd="r5")  # the maps send refused too
    ordered = ([big, rung, proof, lower, lower_proof] + [row(10 + index, 100_000 + 1_000 * index) for index in range(20)]
               + [row(30, 420_000, answered=True)] + [row(40 + index, 90_000, answered=True) for index in range(30)])
    trimmed = cw._trim_start_rows(ordered)

    assert {kept["raw"]: maps for kept, maps in cw._pending_starts(trimmed)} == {
        548_000: False, 500_000: True, 470_000: True}
    orders = {kept["order"] for kept in trimmed}
    assert {proof["order"], lower_proof["order"]} <= orders  # each rung keeps its proof, even a refused maps send
    assert not orders & ({30} | set(range(10, 30)))  # answered refusals and their folded answer leave
    (cover,) = [kept for kept in trimmed if kept["form"] == "cover"]
    assert (cover["order"], cover["raw"], cover["at"], cover["accepted"]) == (53, 420_000, ordered[0]["at"], True)
    assert all(any(source is kept for source in ordered) for kept in trimmed if kept is not cover)  # TTL intact
    assert trimmed == sorted(trimmed, key=lambda kept: kept["order"])
    assert len(trimmed) <= 3 * cw._START_ROWS + 1


def test_the_wire_profile_still_names_every_transport():
    """Only the start hint draws the transport line: pricing and finalization still get a wire name (R6)."""
    from ouroboros.llm_fallback import _wire_surface
    from ouroboros.provider_models import OPENAI_API_SURFACES as SURFACES

    for provider in ("local", "claudexor", "gigachat", "openai-compatible", ""):
        assert _wire_surface({"provider": provider}) == SURFACES["chat_completions"]
    assert _wire_surface({"provider": "openai", "api_surface": "responses"}) == SURFACES["responses"]
    assert _wire_surface({"provider": "anthropic"}) == "messages"


@pytest.mark.parametrize("scenario", ["enough", "maps", "none"])
def test_first_contact_preference_reaches_the_next_real_chat_request(tmp_path, monkeypatch, scenario):
    """Real fitting, loop, attempt settlement and Chat HTTP; the second chat has no provider-usage anchor."""
    import json
    import re
    import shutil
    from pathlib import Path
    from ouroboros import capability_evidence as ce, context, context_fit as cf, context_window as cw, dialogue_store
    from ouroboros.chronicle_lead import LEAD_RULE_VERSION
    from ouroboros import review_body_fact
    from ouroboros.memory import Memory
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.response_limits import dispatch_target
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry
    from tests import _turn_shapes
    from tests._first_contact import ANSWER, _episode, _run
    from tests._stand import lay_rooms, room_needle, FIRST_ROOM
    from tests.system_e2e.harness import ScriptedStubModel, StubHttpError

    class Numberless(ScriptedStubModel):
        def __init__(self):
            super().__init__(final_answer=ANSWER)
            self.window_tokens, self.limit, self.wire = 0, None, []
            original = self._server.RequestHandlerClass

            class Handler(original):
                def _send(self, payload, *, stream=False, status=200):
                    if status == 400:
                        payload = {"error": {"code": "1261", "message": "Prompt too long",
                                             "type": "invalid_request_error"}}
                    super()._send(payload, stream=stream, status=status)

            self._server.RequestHandlerClass = Handler

        def _completion(self, body):
            if body.get("tools"):
                raw = cf.bounded_prompt_tokens_for_payload(body, 0)
                text = json.dumps(body["messages"], ensure_ascii=False)
                if self.limit is None:
                    # E refuses the uncut view, but can hold its ordinary telling plus a real required book.
                    # M refuses its first two sends (the whole view, then memory's pass) and holds anything smaller
                    # than the second: the books rung, the Body Map at its parts, is the first request it accepts.
                    self.limit = {"enough": raw - 1_000, "maps": 0, "none": 100_000}[scenario]
                elif scenario == "maps" and len(self.wire) == 1:
                    self.limit = raw  # memory's pass is refused at its own size; a smaller request is held
                rooms = tuple(i for i in range(16) if room_needle(FIRST_ROOM + i) in text)
                self.wire.append(dict(raw=raw, status=400 if raw >= self.limit else 200,
                    # The resident Body Map whole (its own heading), or at its parts and addresses.
                    books="## Body map\\n" in text, parts="## Body map (parts and addresses)" in text,
                    tools=sorted(tool["function"]["name"] for tool in body["tools"]), bible="owed on body changes" in text,
                    development="Hermetic preflight isolation" in text, story=len(rooms), rooms=rooms,
                    leads=text.count("settled its mock work"),  # every room's telling lead (``told_page``)
                    cap=body.get("max_tokens"), owner="fixture 3896" in text,
                    facts=re.findall(r"\[CONTEXT_FACTS\][^|]+\| mode ([^|]+)", text)))
                if raw >= self.limit:
                    raise StubHttpError(400, "Prompt too long")
            result = super()._completion(body)
            result["usage"] = None
            return result

    with Numberless() as server:
        seed = _episode(tmp_path, monkeypatch, server)
        drive, repo = seed.drive, seed.tools._ctx.repo_dir
        shutil.copytree(Path(__file__).resolve().parents[1] / "docs/development", repo / "docs/development")
        page = _turn_shapes._page
        def told_page(room, ids, text):
            return {**page(room, ids, f"Room {room} settled its mock work.\n\n{text}"),
                    "metadata": {"lead_rule": LEAD_RULE_VERSION}}
        # Fixture input, not a patched fit: the whole page has a genuine shorter telling beside its address.
        with monkeypatch.context() as shapes:
            shapes.setattr(_turn_shapes, "_page", told_page)
            # E's story is large enough that the hint's fold leaves the required handbook room under E's limit
            # (whole books once gave that slack; the resident Body Map does not).
            lay_rooms(drive, 16, 290_000 if scenario == "enough" else 100_000)
        memory = Memory(drive_root=drive, repo_dir=repo)
        env = SimpleNamespace(repo_dir=repo, drive_root=drive, repo_path=lambda p: repo / p, drive_path=lambda p: drive / p)
        if scenario == "enough":
            # The same foreign subject is normally served with the Development map. Mock only the Git fact;
            # a directly required handbook then adds its real chapters to both the projection and physical send.
            foreign = tmp_path / "foreign"
            foreign.mkdir()
            body_fact = review_body_fact.body_fact
            monkeypatch.setattr(review_body_fact, "body_fact", lambda root, **kw: (
                review_body_fact.BodyFact(review_body_fact.BODY_FALSE, "remote_chain", "fixture foreign subject")
                if Path(root) == foreign else body_fact(root, **kw)))

        def keyed(key):
            settings_path = drive / "settings.json"
            settings = {**json.loads(settings_path.read_text()), "OPENAI_COMPATIBLE_API_KEY": key}
            settings_path.write_text(json.dumps(settings))
            monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", key)
            return settings

        def send(room, key, *, required=False, dispatch_key=None, answered=scenario != "none"):
            settings = keyed(key)
            # M/N use a short A2 to exercise the just-under-refusal trap. In E only memory may pay the difference.
            short = scenario != "enough" and room in {7102, 7105, 7106}
            task = {**seed.task, "id": f"first-contact-{room}", "chat_id": room, "_is_direct_chat": True,
                    "text": f"fixture 3896 {room} " + "synthetic titles " * (4 if short else 300)}
            if scenario == "enough":
                task["workspace_root"] = str(foreign)
            row = dialogue_store.append(drive, "chat", {"direction": "in", "source": "web", "user_id": 1,
                "chat_id": room, "client_message_id": task["id"], "ts": ce.utc_now_iso(), "text": task["text"]}).row
            task["origin_message_ref"], _ = owner_row_source(row)
            if required:
                task["task_contract"] = {"context_requires_development": True}
            registry = ToolRegistry(repo_dir=repo, drive_root=drive)
            registry._ctx.task_metadata = task
            registry._ctx.task_id, registry._ctx.budget_drive_root = task["id"], drive
            registry._ctx.is_direct_chat, registry._ctx.current_chat_id = True, room
            start = len(server.wire)
            context.build_llm_messages(env, memory, task, ctx=registry._ctx, tool_registry=registry,
                tool_schemas=initial_tool_schemas(registry, context_mode="max"))
            plan = registry._ctx.context_fit_plan
            if dispatch_key:  # the key rotates between this fit and the first real POST
                keyed(dispatch_key)
            text, usage, _ = _run(SimpleNamespace(plan=plan, task=task, drive=drive, tools=registry))
            assert (ANSWER in str(text)) is answered, (server.wire, usage.get("_last_llm_error"))
            # Recovery steps memory and the Body Map, never a tool: every request of this chat carries the same tools.
            assert len({tuple(row["tools"]) for row in server.wire[start:]}) == 1, server.wire[start:]
            sends[room] = registry._ctx.context_fit_plan
            return plan, server.wire[start:], dispatch_target(task["model"], settings), str(text)

        sends = {}

        first, a1, target_a, _ = send(7101, "synthetic-a-3896")
        refused, before_maps, _any = cw.first_contact_limits(target_a, a1[0]["cap"])  # (Max wire, before maps, any form)
        assert refused == a1[0]["raw"], (a1, ce._read(drive).get("main_first_contact"))
        assert before_maps == (a1[1]["raw"] if scenario != "enough" else 0), (a1, ce._read(drive).get("main_first_contact"))
        second, a2, _, _ = send(7102, "synthetic-a-3896")
        if scenario == "enough":  # the handbook a chat requires is no memory's debt: the same tellings fold
            required, with_handbook, _, _ = send(7106, "synthetic-a-3896", required=True)
            assert required.core.development_required and with_handbook[0]["development"], with_handbook
            assert not a2[0]["development"] and required.core.required_book_tokens() > 0
            assert [row["status"] for row in with_handbook] == [200], with_handbook
            assert with_handbook[0]["books"] and with_handbook[0]["story"] == a2[0]["story"], (a2, with_handbook)
            assert with_handbook[0]["rooms"] == a2[0]["rooms"], (a2, with_handbook)
            print({"required A2": with_handbook})
        other, b1, _, _ = send(7103, "synthetic-b-3896")
        if scenario == "maps":  # the followed maps refusal trades no frame for a required handbook either
            required, with_handbook, _, said = send(7104, "synthetic-a-3896", required=True, answered=False)
            assert required.start == ("max", "first_contact") and required.core.development_required
            assert all(row["development"] for row in with_handbook), with_handbook  # whole on every request
            assert with_handbook[0]["books"] and with_handbook[0]["story"] == a2[0]["story"], (a2, with_handbook)
            # M holds less than the tellings plus the handbook: memory never pays it (owner 1.11); the books rung
            # steps only the Map, and the terminal names the owner's choices instead of dropping the requirement.
            assert with_handbook[-1]["parts"] and all(row["status"] == 400 for row in with_handbook), with_handbook
            assert "a new task without this requirement" in said, said
        # A's fact shaped the fit; B resolved at dispatch: B's first POST is the ordinary view, one plan, no extra call.
        hinted, rotated, _, _ = send(7105, "synthetic-a-3896", dispatch_key="synthetic-b-3896")
        assert hinted.first_contact_normal is not None, scenario
        assert rotated[0]["books"] and rotated[0]["story"] == a1[0]["story"], (a1, rotated)
        # Its own facts only: the unsent hinted preparation's row is gone, the ordinary view's row is B1's.
        assert len(rotated[0]["facts"]) == 1 and rotated[0]["facts"] == b1[0]["facts"], (b1, rotated)
        assert rotated[0]["story"] > a2[0]["story"], (a2, rotated)  # not A2's folded view
        assert sends[7105].first_contact_normal is None
        assert not sends[7105].projection(sends[7105].initial_mode).memory_facts.get("first_contact_preference")
        if scenario == "none":  # a refused Body-Map-parts send is no Max bound, and the next start keeps its tellings
            _third, a3, _, _ = send(7106, "synthetic-a-3896")
            assert a3[0]["story"] == a2[0]["story"] and a3[0]["books"] and not a3[0]["parts"], (a2, a3)
            assert a3[0]["leads"] == a2[0]["leads"] == 16, (a2, a3)
            # Max with A2's story: A2's size plus the Map's step back to whole (as A1's own sends measured it) plus
            # what four more chats add to memory (the same Low form at the same level on both chats: their last sends).
            step = max(a["raw"] - b["raw"] for a in a1 for b in a1 if a["books"] and b["parts"] and a["story"] == b["story"])
            grown = a3[-1]["raw"] - a2[-1]["raw"]
            assert a3[-1]["parts"] and a2[-1]["parts"] and a3[-1]["leads"] == a2[-1]["leads"], (a2, a3)
            assert step > 0 and a2[0]["raw"] < a3[0]["raw"] <= a2[0]["raw"] + step + grown + 500, (a1, a2, a3, step, grown)
        keyed("synthetic-a-3896")
    print({"limit": server.limit, "A1": a1, "A2": a2, "B1": b1})
    if scenario == "none":
        assert a1 and a2 and b1 and all(row["status"] == 400 for row in a1 + a2 + b1), (a1, a2, b1)
        assert a2[0]["raw"] >= min(row["raw"] for row in a1) - 2_000, (a1, a2)
    else:
        assert [row["status"] for row in a1] == ([400, 400, 200] if scenario == "maps" else [400, 200]), a1
        # M: the books rung's Body Map at its parts is what A1's round finally sent; memory's pass was refused.
        # That first numberless pass of the running view takes every telling (a1[1]["leads"] == 0) while the Map is
        # still whole: base behaviour, razzant/ouroboros#4321, pinned neither away nor as desired here.
        assert scenario != "maps" or (a1[2]["parts"] and not a1[1]["parts"] and a1[2]["story"] == a1[1]["story"]), a1
        # M: one accepted send. My human's Max at its tellings still stood above the refused Max a parts send
        # followed, so A2 starts Low, temporarily, for the window: the Map at its parts, every telling, A1's tools.
        assert [row["status"] for row in a2] == [200], (a1, a2)
    lowered = scenario != "enough"  # E's fold gets under its refusal (no parts send followed): Max, the Map whole
    if lowered:  # every room at its telling, smaller than every size this wire refused
        assert a2[0]["story"] == 0 and a2[0]["leads"] == 16 and a2[0]["raw"] < _any, (a1, a2, _any)
        assert a2[0]["parts"] and not a2[0]["books"] and a2[0]["tools"] == a1[0]["tools"], (a1, a2)
    else:
        assert a2[0]["books"] and not a2[0]["parts"], a2
    assert b1[0]["status"] == 400 and b1[0]["books"] and b1[0]["story"] == a1[0]["story"], (a1, b1)
    assert a2[0]["story"] < a1[0]["story"]
    assert second.start == (("low", "first_contact_maps") if lowered else ("max", "first_contact")), second.start
    assert all(row["bible"] and row["owner"] and row["cap"] == a1[0]["cap"] for row in a1 + a2 + b1)
    assert all(plan.window_record.usable is None and plan.preferred_mode == "max" for plan in (first, second, other))
    assert second.projection(second.initial_mode).memory_facts["first_contact_preference"]


def test_the_target_is_measured_on_this_conversation_or_stays_unknown():
    ctx = _ctx()
    # 86aa r1482: E = its raw 848,751 at its own density 874,829/795,458; A = the largest input Codex accepted.
    assert lmc._refusal_target(ctx, _fit(933_440)) == (933_440 - 921_840 + 115_230, 115_230)  # + ceil(A/8), L33 2=A
    assert ctx.accumulated_usage[REFUSED_INPUTS_KEY]  # recorded: the facts stop stating free room
    assert _target(_ctx(), _fit(900_000)) is None  # E at or below A contradicts the reading
    assert _target(_ctx(), _fit(933_440, anchor=None)) is None  # a cold count sizes nothing
    # A numberless refusal of a request sent inside the reply reservation of a stated total is that total's own: it
    # is measured against the total (input + reply - T, and the total's eighth), never against the last accepted
    # input, whatever that was (a jump of 80,000 cut 124,668 where 5,536 were over; an earlier larger acceptance
    # left the target unknown, which takes whole categories).
    for accepted in (160_000, 260_000):
        inside = _fit(240_000, anchor=160_000, accepted=accepted, stated_total=300_000)
        assert lmc._refusal_target(_ctx(), inside) == (240_000 + 65_536 - 300_000 + 37_500, 37_500)
    assert lmc._refusal_target(_ctx(), _fit(240_000, anchor=160_000, accepted=160_000)) == (
        240_000 - 160_000 + 20_000, 20_000)  # outside any reservation: E - A and the eighth of A, as before
    # Outside the reservation band, an anchored refusal can still measure E - A. Only THAT recovery uses the
    # physical window's eighth; an unsized refusal keeps the established admission-frame slack and order.
    for anchor, accepted, target, slack in ((250_000, 250_000, 53_000, 50_000),
                                           (None, 250_000, None, 57_168), (250_000, 254_000, None, 57_168)):
        fit = _fit(253_000, anchor=anchor, accepted=accepted)
        fit.measurement.capacity_total_tokens, fit.measurement.window_boundary_tokens = 457_344, 400_000
        assert lmc._refusal_target(_ctx(), fit) == (target, slack)
    stated = _ctx("This model's maximum context length is 400,000 tokens.")
    assert _target(stated, _fit(390_000, accepted=None, reply=20_000)) == 10_000 + 50_000
    # Anthropic and Gemini wordings state both numbers; the provider's own count beats the estimate,
    # and a reply the refused request reserved, where stated, is the one the retry reserves again.
    for message, size, reply, limit in (
            ("prompt is too long: 215000 tokens > 200000 maximum", 215_000, 8_192, 200_000),
            ("input length and `max_tokens` exceed context limit: 190000 + 65536 > 200000", 190_000, 65_536, 200_000),
            ("The input token count (1050000) exceeds the maximum number of tokens allowed (1048576).",
             1_050_000, 8_192, 1_048_576),
            ("This model's maximum context length is 128000 tokens. However, your messages resulted in "
             "131000 tokens.", 131_000, 8_192, 128_000)):
        refusal = _ctx(message)
        assert _target(refusal, _fit(1, anchor=None, accepted=None, reply=8_192)) == (
            max(0, size + reply - limit) + -(-limit // 8)), message
        assert _refused_size(refusal) == size, message
    # A local serving instance that refused before sending gives its own exact counts (#1018 lane).
    local = SimpleNamespace(input_tokens=34_000, context_window_tokens=32_768)
    assert _target(_ctx(), _fit(30_000, anchor=None, accepted=None, reply=8_192), local) == (
        34_000 + 8_192 - 32_768 + 4_096)
    assert _target(_ctx(), _fit(30_000, anchor=None, accepted=None, reply=8_192)) is None  # cold, numberless
    # A stated limit beside a COLD estimate measures nothing (chars/4 under-counts code and Cyrillic): unknown;
    # with the provider's own count of the refused request it measures even the first refusal (review r2, 18.10).
    assert _target(stated, _fit(390_000, anchor=None, accepted=None, reply=20_000)) is None


def test_a_cited_limit_declared_for_the_route_recognises_size_sizes_the_pass_and_teaches_nothing(tmp_path, monkeypatch):
    """15.11 18:20: Alibaba refused a request estimated at 819,929 with «Range of input length should be [1, 983616]»
    behind the gateway's «Provider returned error». 983,616 is that endpoint's declared ``max_prompt_tokens``, so the
    round recognises a size refusal by equality and its passes run on the same route, sized on an input cap (the
    reply is not subtracted again); nothing is written to the window record, the measured size included. A total
    on the record recognises alike without the catalogue read. A typed cause, a number at or below the reply
    allowance, a capture of another route or a route with no matching declaration recognise nothing."""
    from ouroboros import capability_evidence as ce, pricing
    from ouroboros.context_anchor import cited_limit_refusal
    from ouroboros.context_window import WindowLimit, WindowRecord

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    declared = WindowLimit(1_000_000, "total", "provider_metadata")
    plan = SimpleNamespace(provider="openrouter", model="qwen/qwen3.8-max-0902", route_fp="scope", window_witness_bound=True,
                           window_record=WindowRecord("scope", total=declared, reserve=65_536, declared=(declared,)))
    fit = SimpleNamespace(measurement=SimpleNamespace(**vars(_fit(819_929, anchor=803_073, accepted=803_073).measurement),
                                                      route_fp="route", measurement_density=1.0))
    refused = SimpleNamespace(max_completion_tokens=65_536, prompt_tokens_bounded_estimate=819_929,
                              physical_context=SimpleNamespace(route_fp="route", round_id="other"))
    reads, catalogue = [], {}
    monkeypatch.setattr(pricing, "get_pricing", lambda **route: reads.append(route) or dict(catalogue))

    def stamped(sentence, *, speaker="Alibaba", untyped=True):
        return {"_last_llm_error_kind": "bad_request", "_last_llm_provider_message": sentence, "_last_llm_provider_answer": {
            "speaker": speaker, "sentence": sentence, "untyped": untyped,
            "said": [["", "Provider returned error", False], [speaker, sentence, False]]}}

    bracket = "Range of input length should be [1, 983616]"
    usage = stamped(bracket)
    assert cited_limit_refusal(usage, refused, fit.measurement, plan) is False  # no catalogue row: the walk, informed
    assert reads == [{"provider": "openrouter", "model": "qwen/qwen3.8-max-0902"}] and "basis" not in usage["_last_llm_provider_answer"]
    catalogue["alibaba"] = pricing.PricingSchedule((1.0, None, None, 2.0), source={"context_length": 1_000_000,
                                                                                   "max_prompt_tokens": 983_616})
    assert cited_limit_refusal(usage, refused, fit.measurement, plan) is True
    answer = usage["_last_llm_provider_answer"]
    assert (answer["basis"], answer["limit"], answer["limit_kind"], answer["speaker"]) == (
        "number_equals_declared", 983_616, "input", "Alibaba")
    assert answer["declaration"] == "the endpoint catalogue (alibaba max_prompt_tokens)" and "untyped" not in {
        key for key, value in answer.items() if value}
    # What the round then tells the next candidate and the owner follows the round's verdict, not the first filing.
    from ouroboros.loop_transport import provider_terminal_fallback_text
    from ouroboros.primary_route_observation import failed_attempt_facts

    row = {"model": plan.model, "failure_code": "bad_request", "status_code": 400}
    unread = failed_attempt_facts({**stamped(bracket), "_last_llm_failure": row}, plan.model, False)
    assert "established no size cause" in unread and "size refusal" not in unread
    told = failed_attempt_facts({**usage, "_last_llm_error_kind": "context_overflow", "_last_llm_failure": row}, plan.model, False)
    assert ("read it as a size refusal by a cited 983,616 equal to the input limit of the endpoint catalogue" in told
            and "established no size cause" not in told and f'Alibaba said "{bracket}"' in told)
    long = failed_attempt_facts({**stamped("too long " * 100), "_last_llm_failure": row}, plan.model, False)
    assert long.count("[cut here]") == 1 and "[cut here]" not in told  # a cut sentence says so; a whole one does not
    assert "Logs" in long.split("[cut here]")[1] and "Logs" not in told  # and where the rest is kept
    terminal = dict(is_context_overflow=True, is_transport_wait=False, waited_sec=0.0, is_deadline_exhausted=False)
    ended = provider_terminal_fallback_text({**usage, "_last_llm_error_kind": "context_overflow"}, **terminal)
    assert f'Alibaba said: "{bracket}"' in ended  # the one-route terminal keeps the provider's own sentence
    assert "said:" not in provider_terminal_fallback_text({"_last_llm_error_kind": "context_overflow"}, **terminal)
    ctx = SimpleNamespace(accumulated_usage=usage, active_context_mode="max", context_fit_plan=plan)
    near = SimpleNamespace(**{**vars(refused), "prompt_tokens_bounded_estimate": 950_000})
    assert lmc._refusal_target(ctx, fit, near) == (0 + 122_952, 122_952)  # under the input cap: only the slack is asked
    assert usage[REFUSED_INPUTS_KEY] and not ce._load(tmp_path).get("main_windows")  # this task knows; no record row
    reads.clear()
    total = stamped("This model's maximum context length is 1000000 tokens. However, you requested 1 output tokens and "
                    "your prompt contains at least 1000000 input tokens.", speaker="Cloudflare")
    assert cited_limit_refusal(total, refused, fit.measurement, plan) is True and reads == []  # the record's own total
    assert (total["_last_llm_provider_answer"]["limit_kind"], total["_last_llm_provider_answer"]["declaration"]) == (
        "total", "the window record (provider_metadata)")
    ctx = SimpleNamespace(accumulated_usage=total, active_context_mode="max", context_fit_plan=plan)
    assert lmc._refusal_target(ctx, fit, near) == (950_000 + 65_536 - 1_000_000 + 125_000, 125_000)  # a total holds the reply
    for quiet, capture in ((stamped(bracket, untyped=False), refused),                      # a typed cause of another kind
                           (stamped("Image exceeds 8000 pixels; the limit is 65536"), refused),  # not above the reply sent
                           (stamped("Range of input length should be [1, 983615]"), refused),    # equal to no declaration
                           (stamped(bracket), SimpleNamespace(physical_context=SimpleNamespace(route_fp="other")))):
        assert cited_limit_refusal(quiet, capture, fit.measurement, plan) is False
        assert "basis" not in quiet["_last_llm_provider_answer"]

    # The round itself: the refusal as the transport raised it reaches the same-route passes only through the seam.
    import httpx
    import openai

    from ouroboros import loop as loop_mod, primary_route_observation as pro, project_placement, vision_routing
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.loop_transport import stamp_owner_provider_message

    raw = 'data: {"error":{"code":"invalid_parameter_error","message":"' + bracket + '","type":"invalid_request_error"}}\n\n'
    founding = openai.OpenAI(api_key="x", base_url="https://openrouter.ai/api/v1")._make_status_error_from_response(httpx.Response(
        400, json={"error": {"message": "Provider returned error", "code": 400, "metadata": {"raw": raw, "provider_name": "Alibaba"}}},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")))

    def refuse(ctx, disposition, **_kw):
        filed = classify_llm_exception(founding)
        ctx.accumulated_usage.update(_last_llm_error_kind=filed.kind, _last_llm_failure={
            "model": ctx.active_model, "failure_code": filed.kind, "status_code": 400})
        stamp_owner_provider_message(ctx.accumulated_usage, founding, filed)
        return None, 0.0

    entered = []
    for module, name, value in (
            (loop_mod, "_measure_round_main_fit", lambda ctx, **_kw: SimpleNamespace(measurement=fit.measurement)),
            (loop_mod, "_dispatch_round_model", refuse), (loop_mod, "last_physical_attempt_capture", lambda: refused),
            (lmc, "_fit_route_tool_ceiling", lambda ctx: False), (lmc, "_append_routing_receipts", lambda ctx: False),
            (lmc, "_project_wake_input", lambda ctx, overflowed=False: False),
            (lmc, "_recover_context_overflow", lambda ctx, failed, cost, **kw: entered.append(kw) or ("answered", cost, "max")),
            (pro, "observe_primary_route", lambda ctx: ""), (project_placement, "append_placement_notice", lambda *a, **k: None),
            (vision_routing, "retry_refused_image_round", lambda ctx, cap: None)):
        monkeypatch.setattr(module, name, value)

    def round_once():
        ctx = SimpleNamespace(accumulated_usage={}, tools=SimpleNamespace(_ctx=SimpleNamespace()), context_fit_plan=plan,
                              messages=[{"role": "user", "content": "hi"}], active_model=plan.model, active_context_mode="max",
                              defer_resource_wait=False, attempt_cap=None, refused_density=None, round_idx=3)
        return lmc._call_round_model_once(ctx)[0], ctx.accumulated_usage

    answered, round_usage = round_once()
    assert answered == "answered" and len(entered) == 1 and round_usage["_last_llm_error_kind"] == "context_overflow"
    assert round_usage["_last_llm_provider_answer"]["basis"] == "number_equals_declared"
    catalogue.clear()  # no declaration equals the cited number: the round walks, informed
    answered, round_usage = round_once()
    assert answered is None and len(entered) == 1 and round_usage["_last_llm_error_kind"] == "bad_request"


def test_a_providers_requested_total_holds_the_reply_once():
    """Review r3 (18.10): "you requested (about) N tokens" is the input plus the reply that request reserved.

    Read as input, it added the reply twice: 96,072 instead of 30,536 on OpenAI's own wording."""
    openai = _ctx("This model's maximum context length is 200000 tokens. However, you requested 205536 tokens "
                  "(140000 in the messages, 65536 in the completion). Please reduce the length of the messages "
                  "or completion.")
    assert lmc._refusal_target(openai, _fit(1, anchor=None, accepted=None, reply=20_000)) == (5_536 + 25_000, 25_000)
    assert _refused_size(openai) == 140_000
    openrouter = _ctx("This endpoint's maximum context length is 128000 tokens. However, you requested about 129782 "
                      "tokens (928 of text input, 854 of tool input, 128000 in the output). Please reduce the length "
                      "of either one, or use the \"middle-out\" transform to compress your prompt automatically.")
    assert _target(openrouter, _fit(1, anchor=None, accepted=None, reply=20_000)) == 1_782 + 16_000
    assert _refused_size(openrouter) == 1_782
    parts = _ctx("This endpoint's maximum context length is 200000 tokens. However, you requested about 237000 tokens "
                 "(205000 of text input, 32000 in the output).")
    assert _target(parts, _fit(205_000, accepted=None, reply=32_000)) == 37_000 + 25_000  # anchored E agrees
    # OpenAI counts tool definitions as their own share ("in the functions"); the input is the total less the
    # reply whatever parts it lists (review r4, 18.10: the messages share alone shed 16,000 of a 38,000 need).
    tools = _ctx("This model's maximum context length is 128000 tokens. However, you requested 150000 tokens "
                 "(110000 in the messages, 25000 in the functions, and 15000 in the completion). Please reduce the "
                 "length of the messages, functions, or completion.")
    assert _target(tools, _fit(1, anchor=None, accepted=None, reply=20_000)) == 22_000 + 16_000
    assert _refused_size(tools) == 135_000
    whole = _ctx("This model's maximum context length is 128000 tokens. However, your messages resulted in 140000 "
                 "tokens (115000 in the messages, 25000 in the functions). Please reduce the length of the messages "
                 "or functions.")
    assert _target(whole, _fit(1, anchor=None, accepted=None, reply=8_192)) == 140_000 + 8_192 - 128_000 + 16_000
    assert _refused_size(whole) == 140_000
    # A bare total already holds its reply: the excess is the total over the limit, the input the rest.
    bare = _ctx("This endpoint's maximum context length is 400000 tokens. However, you requested about 410000 tokens")
    assert _target(bare, _fit(300_000, anchor=None, accepted=None, reply=20_000)) == 10_000 + 50_000
    assert _refused_size(bare) == 390_000


def test_a_refusal_is_measured_on_the_bytes_it_refused_and_a_rebound_view_is_not_asked_twice():
    """Astra 11(ii) and R4 A4: the refused size is the last physical attempt's own (its capture's estimate of the bytes
    it sent), not the candidate a rebind narrowed since: 950k refused stays recorded as 950k beside a 600k view. Beside
    an unchanged view the measured excess holds; beside a rebound one, what that view already dropped is not asked."""
    fit = SimpleNamespace(measurement=SimpleNamespace(**vars(_fit(600_000, accepted=900_000).measurement),
                                                      measurement_density=1.0))
    sent = SimpleNamespace(prompt_tokens_bounded_estimate=950_000, physical_context=SimpleNamespace(route_fp="r"))
    for rebound, target in ((False, 50_000 + 112_500), (True, 112_500)):  # E - A = 950k - 900k; slack ceil(A/8)
        ctx = _ctx()
        assert lmc._refusal_target(ctx, fit, sent, rebound=rebound) == (target, 112_500)
        assert _refused_size(ctx) == 950_000


def test_a_numberless_first_refusal_drops_the_slack_and_a_further_one_the_rest(tmp_path, monkeypatch):
    """R4 A2: a first-step refusal with no number and no anchor beside a known window, the view unchanged and the
    host's own estimate fitting (no deficit), measures no excess. The first pass asks memory for the owner's
    return-after-fold slack, ceil(W/8): a strictly smaller retry that keeps most of memory's detail, never the
    shortest view at once; after a further refusal memory's rest takes everything it can give."""
    import ouroboros.loop as loop
    from ouroboros import context_fit as cf, loop_messages, memory_view as mv
    from tests import _memory_view_synthetic as syn

    words = [syn.conversation(i, words=40, word_chars=2_000) for i in range(30)]
    core = cf.ContextCore("Core", "", "", "", "", "", '"Continue."',
                          memory_view_json=mv.snapshot_json(syn.snapshot(room_facts=syn.room("1"), conversations=words)))
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    evidence = SimpleNamespace(route_fp="probe", status="confirmed", stale=False, window_tokens=1_000_000, source="fixture")
    plan = cf.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"id": "a2", "_is_direct_chat": True},
        preferred_mode="max", route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"}, evidence))
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode=plan.initial_mode, tool_schemas=[], task_id="a2",
        tools=SimpleNamespace(_ctx=SimpleNamespace()), messages=plan.messages_for(plan.initial_mode),
        event_queue=None, drive_logs=tmp_path, round_idx=1, accumulated_usage={})

    def facts():
        shown = ctx.context_fit_plan.projection(ctx.active_context_mode)
        return shown.memory_facts["floor"]["steps"].get("F6w", 0), shown.memory_facts["blocks"]["room_tokens"]

    fit = SimpleNamespace(measurement=SimpleNamespace(capacity_total_tokens=1_000_000, measurement_density=1.0,
        reprojection_input_tokens=None, estimated_input_tokens=0, capacity_deficit_tokens=0, route_fp="probe"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *_a, **_kw: fit)
    monkeypatch.setattr(loop_messages, "append_context_facts", lambda _trial: False)  # this stub prices no refresh
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    folded, room = facts()
    assert folded == 0 and room > 500_000  # the start held every word
    slack = lmc._refusal_target(_ctx(), SimpleNamespace(measurement=SimpleNamespace(**{
        **vars(_fit(0, anchor=None).measurement), "capacity_total_tokens": 1_000_000, "measurement_density": 1.0})))[1]
    assert slack == 125_000
    lmc._apply_refusal_step(ctx, "floor", fit, None, slack)
    first, after_first = facts()
    assert 0 < first < 1_200 and slack <= room - after_first < 2 * slack  # about the slack, most words still whole
    lmc._apply_refusal_step(ctx, "memory_rest", fit, None, slack)
    assert facts()[0] == 1_200 and facts()[1] < after_first - slack  # the rest: every word by address


def test_a_refused_first_send_retries_the_view_its_route_rebind_narrowed(main_call, monkeypatch, tmp_path):
    """The path of 07.11 at once (window C, A and 1.8): an Auto start with no known window sends its whole memory and
    Codex refuses one input string's length, its route naming the refusing account but no model. The receipt names the
    requested model, so the rebind binds that account and its known window narrows the view; that view is retried
    first on the same route. The configured chain never starts, no refusal pass folds memory, and memory keeps every
    word the narrowed view kept instead of its shortest forms."""
    import json

    from ouroboros import capability_evidence as ce, context, memory_view as mv
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
    from tests import _memory_view_synthetic as syn
    from tests.test_context_economy_window import MODEL, ROUTE, _answer, _drained, _native, _run
    from tests.test_llm_claudexor import result

    words = syn.live_scale(groups=60, word_chars=1_000)
    ctx, gateway, tools, events, plan, windows = _native(main_call, monkeypatch, tmp_path, owner_mode="max",
                                                          view=mv.snapshot_json(words))
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "", "fallback": ["account-b"]}))  # Main runs Auto
    windows.update({"account-a": 0, "account-b": 400_000})

    def route(task, **_kw):  # the account an operation's receipt binds (``model_account_options``), else Auto
        options = ce.model_account_options(MODEL, role=task["model_role"], model_route=task.get("model_route"))
        account = options["credential_profile_id"] if options.get("account_fingerprint") else "account-a"
        bound = {"source_id": "codex", "credential_profile_id": account, "account_fingerprint": f"fingerprint-{account[-1]}"}
        return {"model": MODEL, "provider": "claudexor", "options": bound}, SimpleNamespace(
            route_fp=ce.route_fingerprint(provider="claudexor", model=MODEL, base_url="", options=bound), stale=False,
            status="confirmed" if windows[account] else "unprobeable", window_tokens=windows[account], source="advertised",
            source_id="codex", credential_profile_id=account, account_fingerprint=bound["account_fingerprint"])
    monkeypatch.setattr(context, "_context_fit_route", route)
    built = plan(0)
    ctx.messages[:] = built.messages_for(built.initial_mode)
    refusing = {"source": "codex", "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b", "model": None}
    gateway.results = [
        result(outcome="failed", route=refusing, problem={"code": "invalid_request", "message": "Codex model request was "
               "refused (HTTP 400).", "context": {"httpStatus": 400, "vendorCode": "string_above_max_length",
                                                  "parameter": "input[3].content[0].text"}}),
        _answer({**ROUTE, "credentialProfileId": "account-b", "accountFingerprint": "fingerprint-b"},
                {"role": "assistant", "content": "Done."})]
    gateway.dispatch = ["response_received"] * 2
    text, _usage, _trace = _run(ctx, tools, events, built)
    assert text == "Done." and len(gateway.creates) == 2  # the narrowed view answered
    first, retry = (payload for payload, _key in gateway.uploads)
    # Auto again, no chain: the unknown start named no account; the retry names the account its rebound plan rests on
    # (account-b's known window sized it) as a preference, never the fallback's pin (W2 H5).
    assert (first["account"], retry["account"]) == ({"mode": "auto"}, {"mode": "auto", "preferredProfileId": "account-b"})
    assert len(json.dumps(retry["messages"])) < len(json.dumps(first["messages"]))  # strictly smaller, same route
    rebound = tools._ctx.context_fit_plan
    assert rebound.model_route["accountFingerprint"] == "fingerprint-b" and rebound.window_record.usable is not None
    kinds = [row.get("checkpoint_kind") for row in _drained(events)]
    assert "context_fit_route_rebound" in kinds and "context_memory_coarsened" not in kinds  # no pass folded memory
    folded = rebound.projection(rebound.initial_mode).memory_facts["floor"]["steps"].get("F6w", 0)
    assert 0 < folded < 60 * 33  # narrowed for the account's window, never every word


def test_the_size_parser_reads_input_reply_and_total_apart():
    from ouroboros.capability_evidence import _parse_ctx_requested

    assert _parse_ctx_requested("However, you requested 205536 tokens (140000 in the messages, 65536 in the "
                                "completion).") == (140_000, 65_536, 205_536)
    assert _parse_ctx_requested("you requested about 129782 tokens (928 of text input, 854 of tool input, "
                                "128000 in the output)") == (1_782, 128_000, 129_782)
    assert _parse_ctx_requested("you requested about 237000 tokens (32000 in the output)") == (205_000, 32_000, 237_000)
    assert _parse_ctx_requested("you requested about 410000 tokens") == (0, 0, 410_000)  # a total, never an input
    assert _parse_ctx_requested("you requested 150000 tokens (110000 in the messages, 25000 in the functions, "
                                "and 15000 in the completion)") == (135_000, 15_000, 150_000)
    assert _parse_ctx_requested("your messages resulted in 140000 tokens (115000 in the messages, 25000 in the "
                                "functions)") == (140_000, 0, 0)  # the whole input, not its messages share
    assert _parse_ctx_requested("you requested 140000 tokens (115000 in the messages, 25000 in the functions)") == (
        140_000, 0, 140_000)  # no reply share: the parts are the input
    assert _parse_ctx_requested("context limit: 190000 + 65536 > 200000") == (190_000, 65_536, 0)
    assert _parse_ctx_requested("The input token count (1050000) exceeds the limit") == (1_050_000, 0, 0)
    assert _parse_ctx_requested("Your input exceeds the context window of this model") == (0, 0, 0)


def test_bodies_shed_only_what_the_host_copies_left_of_the_measured_excess(monkeypatch):
    """Audit of early stops (06.11): shown bodies pay the measured excess only; the return-after-fold
    slack is memory's to pay, so bodies the host copies already covered stay in place."""
    calls, shed = [], {}

    def address(_ctx, _fit, *, rung, goal_tokens=None):
        calls.append((rung, goal_tokens))
        return SimpleNamespace(status="applied" if rung in shed else "no_eligible", reclaimed_tokens=shed.get(rung, 0))

    monkeypatch.setattr(lmc, "_run_emergency_address_pass", address)
    first = ("host_copies", "bodies")
    shed.update(host_copies=6_000, bodies=70_000)  # target 126,830 = excess 11,600 + slack 115,230
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830, 115_230) is True
    assert calls == [("host_copies", None), ("bodies", 5_600)]  # the rest of the excess, never the slack
    calls.clear()
    shed.update(host_copies=60_000)
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830, 115_230) is True
    assert calls == [("host_copies", None)]  # superseded host state paid the excess: every body stays
    calls.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830) is True  # the slack read from the window
    assert calls == [("host_copies", None)]
    calls.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), None) is True
    assert calls == [("host_copies", None), ("bodies", None)]  # unknown: each step takes its category
    calls.clear()
    shed.clear()
    assert lmc._apply_refusal_pass(_ctx(), first, _fit(1), 126_830, 115_230) is False  # nothing changed, no retry


def test_bodies_pay_the_excess_and_the_measured_floor_owns_the_books_decision(monkeypatch):
    """Shown bodies pay only the excess; the measured first floor gets the rest and the slack.
    Unsized recovery and the second pass keep the ordinary ladder."""
    import ouroboros.loop as loop

    calls, sizes = [], []
    monkeypatch.setattr(lmc, "_coarsen_memory_view", lambda _ctx, _fit, *, saving=None, slack=0, telling_floor=False: (
        calls.append(("floor", saving, slack, telling_floor)) or True))
    monkeypatch.setattr(lmc, "_reproject_actual_overflow_low",
                        lambda ctx, **_kw: calls.append(("books",)) or setattr(ctx, "active_context_mode", "low"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda _ctx, **_kw: _fit(sizes.pop(0)))
    monkeypatch.setattr(lmc, "_run_emergency_address_pass", lambda _ctx, _fit, *, rung, goal_tokens=None: (
        calls.append((rung, goal_tokens)) or SimpleNamespace(status="applied", reclaimed_tokens=goal_tokens or 0)))
    first, second = lmc.REFUSAL_PASSES
    assert first == ("host_copies", "bodies", "floor") and second == ("books", "unseen_bodies")
    stated = _ctx("This model's maximum context length is 400,000 tokens.")  # slack 50,000
    sizes[:] = [830_000]  # after memory narrowed
    assert lmc._apply_refusal_pass(stated, first, _fit(900_000, accepted=None), 70_000, 50_000) is True
    assert calls == [("host_copies", None), ("bodies", 20_000), ("floor", 0, 50_000, True)]
    assert stated.active_context_mode == "max"  # the ordinary ladder didn't decide to replace its books
    calls.clear()
    sizes[:] = [880_000, 830_000]  # before and after the books step
    assert lmc._apply_refusal_pass(stated, second, _fit(900_000, accepted=None), 70_000, 50_000) is True
    assert calls == [("books",), ("unseen_bodies", 20_000)]  # books first, then only what they left
    assert stated.active_context_mode == "low"
    unsized = _fit(900_000, accepted=None)  # no window reading anywhere: no slack, the whole target is excess
    calls.clear()
    sizes[:] = [800_000]
    assert lmc._apply_refusal_pass(_ctx(), first, unsized, None) is True
    # Unknown: memory's detail pays all it can, a whole-or-outline section nothing without a measured excess.
    assert calls == [("host_copies", None), ("bodies", None), ("floor", 0, lmc.WHOLE_CATEGORY, False)]
    calls.clear()
    sizes[:] = [800_000, 790_000, 780_000]  # the books step's before and after, then memory's rest
    assert lmc._apply_refusal_pass(_ctx(), second, unsized, None) is True
    # After a further refusal memory's rest may take everything, whole-or-outline sections included.
    assert calls == [("books",), ("unseen_bodies", None), ("floor", lmc.WHOLE_CATEGORY, 0, False)]
    calls.clear()
    sizes[:] = [800_000]
    known = SimpleNamespace(measurement=SimpleNamespace(**{**vars(unsized.measurement), "capacity_deficit_tokens": 18_887}))
    assert lmc._apply_refusal_pass(_ctx(), first, known, None) is True
    assert calls[-1] == ("floor", 18_887, 0, False)  # a capacity estimate alone doesn't change unsized refusal order


def test_a_failed_candidates_walk_keeps_the_rounds_refusal_and_the_conversation_facts():
    """Review r2 (18.10): restoring the primary's route-fit state after a failed fallback erased the round's
    size-refusal stamp (so the renewal never ran) and the per-model anchors. Route-fit keys still restore."""
    from ouroboros.context_anchor import ACCEPTED_INPUTS_KEY, INPUT_ANCHOR_KEY, REFUSED_INPUTS_KEY

    usage = {"_context_capacity_total_tokens": 872_000, INPUT_ANCHOR_KEY: {"a": {}}}
    snapshot = lmc._snapshot_context_fit_usage(usage)
    usage.update({"_context_capacity_total_tokens": 128_000, lmc.CONTEXT_REFUSAL_ROUND_KEY: 7,
                  INPUT_ANCHOR_KEY: {"a": {}, "b": {}}, ACCEPTED_INPUTS_KEY: {"b": 90_000}, REFUSED_INPUTS_KEY: {"b": 130_000}})
    lmc._restore_context_fit_usage(usage, snapshot)
    assert usage["_context_capacity_total_tokens"] == 872_000  # route-fit state is the primary's again
    assert usage[lmc.CONTEXT_REFUSAL_ROUND_KEY] == 7 and set(usage[INPUT_ANCHOR_KEY]) == {"a", "b"}
    assert usage[ACCEPTED_INPUTS_KEY] == {"b": 90_000} and usage[REFUSED_INPUTS_KEY] == {"b": 130_000}


def test_a_local_pre_send_refusal_uses_the_reply_its_exact_request_reserved():
    """Review r2 (18.10): a 131,072 window, 125,000 exact input and the 8,192 reply floor of the refused
    request exceed by 2,120, not by the estimate's 14,688-token reply."""
    local = SimpleNamespace(input_tokens=125_000, context_window_tokens=131_072, max_completion_tokens=8_192)
    assert _target(_ctx(), _fit(100_000, anchor=None, accepted=None, reply=14_688), local) == 2_120 + -(-131_072 // 8)


def test_books_pay_remaining_room_before_any_unseen_body(monkeypatch):
    """The owner's 22 October priority: memory and host books yield before chosen bodies; since 06.11 books
    wait for the second pass, after a further refusal."""
    import ouroboros.loop as loop

    calls, sizes, unseen = [], [], {}
    monkeypatch.setattr(lmc, "_reproject_actual_overflow_low",
                        lambda ctx, **_kw: calls.append(("books",)) or setattr(ctx, "active_context_mode", "low"))
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda _ctx, **_kw: _fit(sizes.pop(0)))
    monkeypatch.setattr(lmc, "_run_emergency_address_pass", lambda _ctx, _fit, *, rung, goal_tokens=None: (
        calls.append((rung, goal_tokens)) or SimpleNamespace(status=unseen.get("status", "no_eligible"),
                                                             reclaimed_tokens=unseen.get("shed", 0))))
    stated = _ctx("This model's maximum context length is 400,000 tokens.")  # slack 50,000
    second = lmc.REFUSAL_PASSES[1]
    sizes[:] = [880_000, 810_000]  # before and after the books step
    assert lmc._apply_refusal_pass(stated, second, _fit(900_000, accepted=None), 70_000) is True
    assert calls == [("books",)]  # the books paid the whole goal: unseen bodies stay
    assert stated.active_context_mode == "low"
    calls.clear()
    paid = _ctx("This model's maximum context length is 400,000 tokens.")
    unseen.update(status="applied", shed=50_000)
    sizes[:] = [880_000, 860_000]
    assert lmc._apply_refusal_pass(paid, second, _fit(900_000, accepted=None), 70_000) is True
    assert calls == [("books",), ("unseen_bodies", 50_000)]  # then only what the books left


@pytest.mark.parametrize("useful", [False, True], ids=["map-saves-nothing", "map-steps-to-parts"])
def test_a_measured_second_pass_cannot_hide_an_unhelpful_book_switch(tmp_path, monkeypatch, useful):
    """The books rung gives only the Body Map's step to its parts and addresses in the Low frame: where that saves
    nothing (a Map with no shorter form) the measured rung changes nothing and stays Max, so the pass goes on to unseen
    bodies; where it saves (a large Map, its short parts form and a reader for its full text) the start is Low with the
    Map at its parts at the same memory level, and nothing else is spent."""
    import json

    from ouroboros import context_fit as cf, loop, memory_view as mv
    from tests import _memory_view_synthetic as syn
    from tests._context_fit_fixtures import _refusal_ctx

    body_map = "## Body map\n\n" + "Whole map material. " * (4_000 if useful else 1)
    core = cf.ContextCore("Core", "", body_map, "", "", "", '"Assignment"', memory_view_json=mv.snapshot_json(syn.snapshot()),
                          **({"body_map_parts": "## Body map (parts and addresses)\n\n- Core: `docs/core.md`",
                              "section_root": str(tmp_path / "sections"), "section_task_id": "books"} if useful else {}))
    evidence = SimpleNamespace(route_fp="books", status="confirmed", stale=False, window_tokens=200_000)
    plan = cf.build_context_fit_plan(None, core, {"type": "task"}, preferred_mode="max",
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"}, evidence))
    ctx = _refusal_ctx(tmp_path, plan)
    ctx.accumulated_usage = {}
    shown = ctx.messages[0]

    def measure(ctx, **_kw):
        return cf.measure_main_fit(ctx.context_fit_plan, ctx.messages, [],
            profile=lmc._main_context_profile(ctx.context_fit_plan, ctx.active_context_mode),
            rendered_mode=ctx.active_context_mode, round_id="books")

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    unseen = []
    def shed_unseen(current, _fit, *, rung, goal_tokens=None):
        unseen.append((rung, current.active_context_mode, goal_tokens))
        return SimpleNamespace(status="applied", reclaimed_tokens=goal_tokens)

    monkeypatch.setattr(lmc, "_run_emergency_address_pass", shed_unseen)
    assert lmc._apply_refusal_pass(ctx, lmc.REFUSAL_PASSES[1], measure(ctx), 2_000, 1_000)
    if useful:
        assert ctx.active_context_mode == "low" and ctx.messages[0] != shown and not unseen
        low, held = ctx.context_fit_plan.projection("low").memory_facts, plan.projection("max").memory_facts
        assert [row["section_id"] for row in low["host_sections_addressed"]] == ["body_map"]
        sent = json.dumps(ctx.messages[0], ensure_ascii=False)
        assert "## Body map (parts and addresses)" in sent and "Whole map material." not in sent
        assert (low["floor"]["steps"], low["floor"]["runs"]) == (held["floor"]["steps"], held["floor"]["runs"])
    else:
        assert unseen == [("unseen_bodies", "max", 2_000)]
        assert ctx.context_fit_plan is plan and ctx.messages[0] == shown


@pytest.mark.parametrize("refused", [False, True])
def test_a_just_chosen_body_pays_last_and_only_what_a_providers_refusal_left(monkeypatch, refused):
    """Owner 2A: an estimate never cuts a chosen read. After a provider's refusal memory, books and unasked
    bodies go first; a chosen body that fits alone pays only what they left (everything when nothing measures it)."""
    shed, chosen = {}, []
    monkeypatch.setattr(lmc, "_apply_refusal_step", lambda _ctx, step, _fit, _target, _slack=None: shed.get(step))
    monkeypatch.setattr(lmc, "_run_emergency_address_pass", lambda _ctx, _fit, *, rung, goal_tokens=None: (
        chosen.append((rung, goal_tokens)) or SimpleNamespace(status="applied", reclaimed_tokens=1)))
    first, second = lmc.REFUSAL_PASSES
    for steps, target, paid, expected in (
            (second, None, {}, [("chosen_bodies", None)]),  # nothing else changed an unmeasured refusal
            (second, None, {"books": 0}, []),  # the books changed the candidate: its retry answers first
            (second, 100_000, {"books": 30_000, "unseen_bodies": 10_000}, [("chosen_bodies", 60_000)]),
            (second, 100_000, {"books": 100_000}, []),  # the books paid the whole measured target
            (first, None, {}, [])):  # the first pass never reaches a chosen body
        shed.clear(), shed.update(paid), chosen.clear()
        changed = lmc._apply_refusal_pass(_ctx(), steps, _fit(1), target, None, refused)
        assert chosen == (expected if refused else [])
        assert changed is bool(paid or chosen)


def _books_core(tmp_path, form="parts"):
    """A Max core whose Body Map Max holds whole (``_reference_book_fixtures.booked``: the real renderer's Map and its
    parts form) beside a reader for its full text; ``form`` takes the step away: no parts form, or an actor with no
    reader at all (no store to retain the full text in, and no ``read_file`` for the Map's own source)."""
    from ouroboros import context_fit as cf, memory_view as mv
    from tests import _memory_view_synthetic as syn
    from tests._reference_book_fixtures import booked

    snapshot = syn.snapshot(story=syn.pages(2, 200), room_facts=syn.room())
    core = booked(cf.ContextCore("Core", "", "", "", "", "", '"Assignment"', memory_view_json=mv.snapshot_json(snapshot),
                                 section_root=str(tmp_path / "sections"), section_task_id="books01"),
                  tmp_path, body="Whole map material. " * 40, readable=form != "no-reader")
    return replace(core, **{"parts": {}, "no-parts": {"body_map_parts": ""}, "no-reader": {"section_root": ""}}[form])


def _books_ctx(tmp_path, monkeypatch, core, *, mode="max", window=1_000_000):
    from ouroboros import capability_evidence as ce, context_fit as cf
    import ouroboros.loop as loop

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    plan = cf.build_context_fit_plan(None, core, {"type": "task"}, preferred_mode=mode,
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "test"},
            ce.CapabilityEvidence(window, "confirmed" if window else "unprobeable", "metadata", "books-route")))
    assert plan.initial_mode == mode
    ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode=mode, messages=plan.messages_for(mode),
        tools=SimpleNamespace(_ctx=SimpleNamespace()), tool_schemas=[], accumulated_usage={}, task_id="books01",
        event_queue=None, drive_logs=tmp_path / "logs", round_idx=2)

    def measure(ctx, **_kwargs):
        return cf.measure_main_fit(ctx.context_fit_plan, ctx.messages, [],
            profile=lmc._main_context_profile(ctx.context_fit_plan, ctx.active_context_mode),
            rendered_mode=ctx.active_context_mode, round_id="books")

    events = []
    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda _queue, _task, _logs, event, *_a, **_kw: events.append(event))
    monkeypatch.setattr(lmc, "invalidate_task_cache_splits", lambda *_a: None)
    return ctx, measure, events


@pytest.mark.parametrize("ceiling,mode", [(0, "low"), (200_000, "low"), (50_000, "nano")])
def test_books_recovery_keeps_the_current_refusal_frame_and_selected_mode(tmp_path, monkeypatch, ceiling, mode):
    """The books rung from Max: the Body Map steps to its parts (a real Map, its parts form and a reader for its full
    text), in the frame the refused window selects."""
    from ouroboros.context_window import record_observation

    ctx, measure, events = _books_ctx(tmp_path, monkeypatch, _books_core(tmp_path))
    plan = ctx.context_fit_plan
    if ceiling:
        record_observation(tmp_path, plan.route_fp, refused=ceiling)
    lmc._reproject_actual_overflow_low(ctx)
    assert ctx.active_context_mode == ctx.tools._ctx.active_context_mode == mode
    assert lmc.map_at_parts(ctx.context_fit_plan.projection(mode))
    assert [event["checkpoint_kind"] for event in events] == ["context_fit_low_retry"]
    from ouroboros.transcript_prefix import SCOPED_SANCTION_ATTR  # the next send names this rebuild (C2 S4)
    assert getattr(ctx.tools._ctx, SCOPED_SANCTION_ATTR) == {"system": "low_retry"}
    assert ctx.messages[0] == ctx.context_fit_plan.projection(mode).system_message()
    assert ctx.context_fit_plan.window_record is not None
    # A refused total charges the whole reply reserve; the stated one, not refused yet, the reply floor
    # (``WindowRecord.admitted``): the round's frame is that line plus the reserve.
    fit = measure(ctx).measurement
    assert fit.capacity_total_tokens - fit.response_reserve_tokens == (ceiling - fit.response_reserve_tokens if ceiling else 1_000_000 - 8_192)
    assert ctx.context_fit_plan.window_record.source.startswith("refusal" if ceiling else "metadata")


@pytest.mark.parametrize("form", ["parts", "no-parts", "no-reader"])
def test_a_books_rung_that_cannot_step_the_map_changes_and_counts_nothing(tmp_path, monkeypatch, form):
    """With no parts form or no reader for the Map's full text the rung has nothing to give: the
    mode, the plan and the transcript stay, no ``context_fit_low_retry`` checkpoint tells the owner the Map was
    shortened, and the step is no contribution; with both, the same refusal steps the Map and counts its saving."""
    ctx, measure, events = _books_ctx(tmp_path, monkeypatch, _books_core(tmp_path, form))
    plan, shown = ctx.context_fit_plan, json.dumps(ctx.messages)
    shed = lmc._apply_refusal_step(ctx, "books", measure(ctx), None, 0)
    if form == "parts":
        assert ctx.active_context_mode == "low" and lmc.map_at_parts(ctx.context_fit_plan.projection("low"))
        assert shed and shed > 0 and [event["checkpoint_kind"] for event in events] == ["context_fit_low_retry"]
    else:
        assert shed is None and events == []
        assert ctx.active_context_mode == "max" and ctx.context_fit_plan is plan and json.dumps(ctx.messages) == shown


@pytest.mark.parametrize("form", ["parts", "no-reader"])
def test_an_already_low_frame_on_a_route_with_no_window_steps_the_map_at_a_numberless_refusal(tmp_path, monkeypatch, form):
    """An owner-Low plan on a route that states no window meets a refusal that names no size; the
    books rung works in that frame too: the Map steps to its parts, the mode stays Low, the saving counts and the
    owner is told. With no reader for the Map's full text nothing steps, nothing is claimed."""
    ctx, measure, events = _books_ctx(tmp_path, monkeypatch, _books_core(tmp_path, form), mode="low", window=0)
    plan = ctx.context_fit_plan
    assert plan.window_record is None or plan.window_record.usable is None  # no window known on this route
    assert not lmc.map_at_parts(plan.projection("low"))  # an unknown window shortens nothing at the start
    before = plan.projection("low").memory_facts["floor"].get("steps")
    shed = lmc._apply_refusal_step(ctx, "books", measure(ctx), None, 0)
    assert ctx.active_context_mode == "low"
    if form == "parts":
        rebound = ctx.context_fit_plan.projection("low")
        assert lmc.map_at_parts(rebound) and shed and shed > 0
        assert rebound.memory_facts["floor"].get("steps") == before  # only the Map stepped
        assert [event["checkpoint_kind"] for event in events] == ["context_fit_low_retry"]
    else:
        assert shed is None and events == [] and ctx.context_fit_plan is plan
