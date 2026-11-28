"""Count-only observation and startup admission of the Nano floor for one captured local text root.

N is the request at the end of the ordinary physical-shortage walk, before
uncovered owner words yield: each host form the sender's own last memory rung
may take in this tool frame, at the address its capture would publish. B removes
the six budgeted memory sources from that rendering. Neither copy is an inference
candidate, a memory write or a retained source.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from ouroboros.section_floor import (
    _system_blocks, predicted_section, retain_section, section_views, with_program_history,
)

MEMORY_SECTIONS = ("identity", "shared_understanding", "scratchpad", "marks", "owner_rules", "knowledge_index")
_MEMORY_HOST = frozenset(MEMORY_SECTIONS) - {"marks"}
# The typed cause (``reason_code``) of a count the server's one slot declined: no connection opened and nothing was
# measured, so it is neither an estimate nor a lost counter. ``reason`` keeps the count it met as ``<count>:<code>``.
COUNTER_BUSY = "counter_busy"
SOURCE_READ_INCOMPLETE = "source_read_incomplete"


def measurement_episode(ctx, plan):
    """Observe the first ordinary root round per process entry, not a voice or direct chat."""
    if (getattr(plan, "preferred_mode", "") != "nano" or not getattr(ctx, "active_use_local", False)
            or getattr(plan, "model_role", "") != "main" or getattr(plan, "view_kind", "") != "owner_turn" or not plan.core):
        return ""
    from ouroboros.body_candidate import continuation_predecessor
    from ouroboros.loop_round_limits import COLD_RETURN_ATTR, start_unread

    slot = ctx.tools._ctx
    if (ctx.task_type != "task" or getattr(slot, "is_direct_chat", False)
            or (getattr(slot, "task_metadata", None) or {}).get("_is_direct_chat")):
        return ""
    if entry := getattr(slot, "_nano_measurement_entry", None):
        return entry[1] if entry[0] == ctx.round_idx else ""
    cold = bool(getattr(slot, COLD_RETURN_ATTR, False))
    continued = bool(continuation_predecessor(slot))
    if not (cold or start_unread(slot, ctx.accumulated_usage)):
        return ""
    episode = ("cold_resume_of_continue" if continued else "cold_resume") if cold else ("continue" if continued else "cold_start")
    slot._nano_measurement_entry = (ctx.round_idx, episode)
    return episode


def held_memory_sections(core):
    """The budgeted sources this capture carries at a revision a continuation saved, in ``MEMORY_SECTIONS`` order.

    A cold return keeps the overview its work held (``context_sections.build_knowledge_sections``) and the index rows
    of its held heads beside every other source read now. Empty for a fresh Start or Continue."""
    saved = getattr(core, "held_note_resume", None) or {}
    held = set()
    if (saved.get("companions") or {}).get("shared_understanding"):
        held.add("shared_understanding")
    if any(entry.get("index_row") for entry in saved.get("entries") or ()):
        held.add("knowledge_index")
    return [name for name in MEMORY_SECTIONS if name in held]


def rules_digest(parts):
    from hashlib import sha256

    owner = next((part for part in parts if part.section_id == "owner_rules"), None)
    return sha256(owner.full_text.encode("utf-8")).hexdigest() if owner is not None else None


def completeness_gap(core, snapshot) -> str:
    """Return ``""`` when every captured source read cleanly (known-empty sources allowed), else a typed gap."""
    if not snapshot or not snapshot.active:
        return "memory_snapshot_unavailable"
    if core is None:
        return f"{SOURCE_READ_INCOMPLETE}:core"
    if failures := tuple(getattr(core, "read_failures", ()) or ()):
        return f"{SOURCE_READ_INCOMPLETE}:{failures[0]}"
    if getattr(core, "reference_book_errors", ()):
        return f"{SOURCE_READ_INCOMPLETE}:reference_books"
    if (snapshot.owner_words_sources or {}).get("gaps"):
        return f"{SOURCE_READ_INCOMPLETE}:owner_words"
    room = snapshot.room or {}
    if room.get("origin_gaps") or (room.get("facts") or {}).get("gaps"):
        return f"{SOURCE_READ_INCOMPLETE}:room"
    if any((group.get("facts") or {}).get("gaps") for group in snapshot.conversations or ()):
        return f"{SOURCE_READ_INCOMPLETE}:conversations"
    if any(not item.get("text_sha256") for item in room.get("lane1") or ()):
        return f"{SOURCE_READ_INCOMPLETE}:room_rows"
    return ""


def _origin(plan):
    """Whose root this capture is, with no address: ``main`` is chat 1 with no project shelf, any other ``project``;
    and the budgeted sources it holds at a saved revision. Empty when the capture has no memory view to say."""
    from ouroboros.memory_view import MAIN_ROOM, snapshot_from_json

    core = plan.core
    snapshot = snapshot_from_json(core.memory_view_json) if core is not None and core.memory_view_json else None
    if snapshot is None:
        return {}
    project = any(getattr(part, "section_id", "") == "project_knowledge_index" for part in core.dynamic_head_sections)
    return {"room": "main" if str(snapshot.spec.room_id) == MAIN_ROOM and not project else "project",
            "held_sections": held_memory_sections(core), "rules_sha256": rules_digest(core.semi_stable_sections)}


def _render(plan, rows, *, window, tool_names, world_short=False, whole_world=False):
    """``(pair, forms, gap)``: the paired canonical messages and the lawful forms they took, or a typed scope gap."""
    from ouroboros import memory_floor as floor
    from ouroboros.context_fit import _governance_blocks
    from ouroboros.memory_view import snapshot_from_json, story_is_whole
    from ouroboros.memory_view_work import model_room

    core = plan.core
    snapshot = snapshot_from_json(core.memory_view_json) if core and core.memory_view_json else None
    if not snapshot or not snapshot.active:
        return None, {}, "memory_snapshot_unavailable"
    if snapshot.spec.role != "integrator" or snapshot.spec.work_room_levels != 2:
        return None, {}, "outside_root_scope"
    if gap := completeness_gap(core, snapshot):
        return None, {}, gap
    projection = plan.projection("nano")
    if projection.user_content_json is not None:
        return None, {}, "owner_input_already_addressed"
    if not rows or rows[0].get("role") != "system" or rows[0].get("content") != projection.system_message()["content"]:
        return None, {}, "unbound_system_projection"
    if not (core.section_root and core.section_task_id):
        return None, {}, "host_section_reader_unavailable"
    host = (*core.semi_stable_sections, *core.dynamic_head_sections, *core.dynamic_sections)
    if not any(part.section_id == "owner_rules" for part in host):
        return None, {}, "owner_rule_boundary_unavailable"
    governance = _governance_blocks(None, core, mode="low")
    sections = (*host, *(governance[0] if not isinstance(governance[0], str) else ()))
    reader = "read_file" in tuple(tool_names or ())
    already_shown = tuple(row for row in projection.memory_facts.get("host_sections_addressed", ())
                          if (reader and not whole_world) or row.get("section_id") != "world")
    # Nothing is written here: a step is priced at the address its capture would publish. WORLD yields to its retained
    # reader only as the sender's exact-shortage rung takes it, with ``read_file`` in the frame it sends.
    receipts, _ = floor._host_steps(sections, already_shown, measure=lambda _: 0, window=window,
        reserve=0, ratio=1, saving=0, root=core.section_root, task_id=core.section_task_id, measure_floor=True,
        retain=predicted_section, exact_local_pressure=world_short and reader and not whole_world)
    forms = section_views(sections, receipts)
    shown = floor.render_view_for_mode(snapshot, mode="nano", owner_mode="nano", window_tokens=window,
        known_window=True, output_reserve=plan.output_reserve_tokens, ratio=1, non_memory_tokens=0,
        tool_names=tool_names, shortest=True)
    # Render the identical floor receipt both times, and filter only the ordinary marks producer.
    parts = dict(floor_note=shown.note, opening_address=core.opening_program_address)
    room, program = model_room(snapshot, shown.level, **parts)
    without_marks, _ = model_room(snapshot, shown.level, **parts,
                                   observe=lambda key, text: "" if key == ("marks",) else text)
    tail = with_program_history(rows[1:], program)
    whole = story_is_whole(shown.level)
    def system(text, omit=(), *, views=forms):
        return {**rows[0], "content": _system_blocks(core, governance, shown.story, text, "nano", views, whole, omit=omit)}

    # P1's two lawful forms of the durable index (whole, or the outline naming every active note), and whether
    # this floor rests on WORLD's retained reader.
    taken = {"knowledge_index_form": "outline" if "knowledge_index" in forms else "whole",
             "world_form": "reader" if "world" in forms else "whole"}
    pair = ([system(room), *tail], [system(without_marks, _MEMORY_HOST), *tail])
    if "knowledge_index" in forms:
        full = {name: text for name, text in forms.items() if name != "knowledge_index"}
        pair += ([system(room, views=full), *tail],)
    return pair, taken, ""


def render_minimum(plan, rows, *, window, tool_names):
    """Render paired canonical messages from this capture; return a typed scope gap instead of guessing."""
    pair, _taken, gap = _render(plan, rows, window=window, tool_names=tool_names)
    return pair[:2] if pair is not None else None, gap


def _sample(payload, observed):
    from ouroboros.local_model_server import input_fingerprint

    tokens = observed.get("input_tokens")
    provenance = observed.get("tokenizer_template_provenance") or {}
    if (observed.get("supported") is not True or observed.get("input_is_exact") is not True
            or type(tokens) is not int or tokens <= 0 or provenance.get("source") != "serving_llama_formatter"
            or observed.get("native_input_sha256") != input_fingerprint(payload) or not provenance.get("rendered_prompt_sha256")):
        return None
    return {"input_tokens": tokens, "native_input_sha256": observed["native_input_sha256"],
            "rendered_prompt_sha256": provenance["rendered_prompt_sha256"]}


def _binding(observed):
    formatter = observed.get("tokenizer_template_provenance") or {}
    return {**{key: observed.get(key) for key in ("process_id", "context_window", "raw_context_window")},
            "formatter": {key: formatter.get(key) for key in ("source", "formatter", "library_version", "chat_format", "launched_n_ctx")}}


def _verified_binding(observed, *, expected_window=None):
    """Verify the serving binding and its effective loaded window, rejecting allocator padding or unverified limits."""
    from ouroboros.local_model_server import effective_window

    if not (observed.get("output_limit_enforced") is True and observed.get("reasoning_included_in_limit") is True):
        return None, "reply_limit_unverified"
    binding = _binding(observed)
    window, raw = binding["context_window"], binding["raw_context_window"]
    if (type(window) is not int or type(raw) is not int or not 0 < window <= raw
            or type(binding["process_id"]) is not int or binding["process_id"] <= 0
            or any(value is None for value in binding["formatter"].values())):
        return None, "serving_binding_unavailable"
    launched = binding["formatter"]["launched_n_ctx"]
    if type(launched) is not int or launched <= 0 or window != effective_window(launched, raw):
        return None, "effective_window_unproven"
    if expected_window is not None and window != expected_window:
        return None, "outside_qualified_window"
    return binding, ""


def _gap(result, code, *, count="", detail=""):
    """A typed gap: ``reason_code`` is the stable cause; ``reason`` adds the count it met and any detail."""
    return {**result, "reason_code": code, "reason": (f"{count}:" if count else "") + code + (f":{detail}" if detail else "")}


def _counter_gap(result, count, observed):
    """A count's own refusal, passed through unchanged as the typed cause (``COUNTER_BUSY`` among them)."""
    code, _, detail = str(observed.get("reason") or "exact_measurement_unavailable").partition(":")
    return _gap(result, code, count=count, detail=detail)


class _CounterUnavailable(Exception):
    def __init__(self, status: str, *, unreachable: bool):
        self.status = status
        self.unreachable = unreachable


def qualified_local_window(record):
    """Whether a window record is the qualified 80k one (``nano_publication.QUALIFIED_WINDOW``) with a usable input:
    the window Nano's exact start counts for and a Send's own row is said once on (``memory_view_work``)."""
    from ouroboros.nano_publication import QUALIFIED_WINDOW

    return (record is not None and record.usable is not None
            and (record.total.tokens if record.total else record.total_equivalent) == QUALIFIED_WINDOW)


def local_root_start_candidate(task, route, record):
    """Cheap route and task prerequisites shared by early rejection and the captured exact walk: the 80k local Main
    route and a root of my own focus, queued or my human's direct turn (Send in Main); never a wake, Presence, a
    helper or a nanny (``memory_view.view_role``)."""
    from ouroboros.memory_view import view_role

    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else (
        task.get("task_metadata") if isinstance(task.get("task_metadata"), dict) else {})
    return (route.get("provider") == "local" and route.get("use_local") is True
            and (route.get("model_role") or "main") == "main" and qualified_local_window(record)
            and task.get("type") == "task" and not task.get("parent_task_id")
            and view_role({**task, "metadata": meta}) == "integrator"
            and task.get("delegation_role") in (None, "", "root")
            and meta.get("delegation_role") in (None, "", "root"))


def _fresh_local_root_snapshot(core, task, state):
    """Return the complete captured view for an ordinary fresh Web/Main 80k local root, else None: a queued root at its
    two eighths of work room or my human's direct turn at its one (``context_budget.start_room_levels``)."""
    from ouroboros.context_fit import _request_rows, messages_carry_native_images
    from ouroboros.memory_view import MAIN_ROOM, snapshot_from_json, start_room
    from ouroboros.owner_wait import fresh_input

    if (not local_root_start_candidate(task, state.route, state.window_record)
            or not fresh_input(task) or core.held_note_resume is not None or not (core.section_root and core.section_task_id)):
        return None
    snapshot = snapshot_from_json(core.memory_view_json) if core.memory_view_json else None
    if (not snapshot or snapshot.spec.role != "integrator" or snapshot.spec.work_room_levels != start_room(task, "integrator")
            or str(snapshot.spec.room_id) != MAIN_ROOM
            or any(getattr(part, "section_id", "") == "project_knowledge_index" for part in core.dynamic_head_sections)
            or not any(part.section_id == "owner_rules" for part in (
                *core.semi_stable_sections, *core.dynamic_head_sections, *core.dynamic_sections))
            or completeness_gap(core, snapshot)
            or messages_carry_native_images(_request_rows([], core.user_content_json, core.supplementary_messages_json))):
        return None
    return snapshot


def _exact_start_fit(core, state, schemas, project, snapshot, user_json: str):
    """Walk deterministic Nano projections on the serving formatter up to the last reachable lawful form."""
    from ouroboros.context_budget import context_mode_limits
    from ouroboros.context_fit import MinimumViewUnfit, _request_rows, estimate_context_prompt_tokens
    from ouroboros.context_window import WindowLimit, WindowRecord
    from ouroboros.llm import LLMClient
    from ouroboros.local_model import get_manager
    from ouroboros.local_model_server import input_fingerprint
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
    from ouroboros.memory_floor import uncovered_owner_rows
    from ouroboros.nano_publication import QUALIFIED_WINDOW

    preview = SimpleNamespace(**{**vars(state), "known_window": False, "ratio": 0.01})
    has_reader = "read_file" in {row["function"]["name"] for row in schemas if (row.get("function") or {}).get("name")}
    failed_sections: set[str] = set()

    def trial_retain(part, *, root, task_id):
        if part.section_id in failed_sections:
            raise OSError("section capture previously failed")
        return predicted_section(part, root=root, task_id=task_id)

    def real_retain(part, *, root, task_id):
        try:
            return retain_section(part, root=root, task_id=task_id)
        except (OSError, ValueError):
            failed_sections.add(part.section_id)
            raise

    if get_manager().serving_context_evidence().get("confirmed") is not True:
        raise _CounterUnavailable("unavailable", unreachable=True)
    uncut_contents, uncut_mode = project(preview, user_json, retain=trial_retain)
    uncut_sys, uncut_facts, uncut_prog = uncut_contents["nano"]
    uncut_rows = _request_rows(uncut_sys, user_json, core.supplementary_messages_json, room_data=uncut_prog)
    actual, ceiling, prepare, measure = LLMClient(api_key="unused")._local_count_io(
        uncut_rows, schemas, MAIN_LOOP_MAX_TOKENS, wait=True)

    def checked_count(payload, expected_binding=None):
        obs = measure(payload)
        if str(obs.get("reason") or "").startswith(("measurement_", "counter_")):
            raise _CounterUnavailable("unavailable", unreachable=True)
        sample = _sample(payload, obs)
        cand_binding, gap = _verified_binding(obs, expected_window=QUALIFIED_WINDOW)
        live = get_manager().serving_context_evidence()
        if live.get("confirmed") is not True:
            raise _CounterUnavailable("unavailable", unreachable=True)
        if sample is None or cand_binding is None or gap:
            raise _CounterUnavailable(gap or "unavailable", unreachable=sample is None)
        if (live.get("process_id") != cand_binding["process_id"]
                or live.get("context_window") != cand_binding["context_window"]
                or (expected_binding is not None and cand_binding != expected_binding)):
            raise _CounterUnavailable("unavailable", unreachable=True)
        return sample, cand_binding

    first_sample, binding = checked_count(actual)
    expected_instance = str(getattr(state.evidence, "window_instance_id", "") or "")
    if expected_instance and expected_instance != str(binding["process_id"]):
        raise _CounterUnavailable("unavailable", unreachable=True)
    window = binding["context_window"]
    caller_ceiling = min(ceiling, state.output_reserve) if state.output_reserve > 0 else ceiling
    _target, reserve = context_mode_limits(
        "nano", "nano", caller_ceiling, working_reserve_tokens=state.working_reserve)
    measured_total = WindowLimit(window, "total", "local_health", instance_id=str(binding["process_id"]))
    serving_record = (replace(state.window_record, total=measured_total, reserve=reserve)
                      if state.window_record is not None
                      else WindowRecord(str(getattr(state.evidence, "route_fp", "") or ""),
                                        total=measured_total, reserve=reserve))
    state.counter_status = "exact"
    input_limit = serving_record.usable
    if input_limit is None or input_limit <= 0:
        raise MinimumViewUnfit("Protected owner words and the minimum view cannot fit the usable input",
                               fixed_tokens=first_sample["input_tokens"], memory_tokens=0, reserve_tokens=reserve,
                               window_tokens=serving_record.total_equivalent or window, ratio=1.0)
    if (first_sample["input_tokens"] <= input_limit
            and not any(row.get("predicted") for row in uncut_facts.get("host_sections_addressed", ()))):
        state.window_record, state.output_reserve = serving_record, caller_ceiling
        return uncut_contents, uncut_mode

    cache: dict[tuple[str, tuple[str, ...]], int] = {(first_sample["native_input_sha256"], ()): first_sample["input_tokens"]}

    def count_payload(payload) -> int:
        key = (input_fingerprint(payload), tuple(sorted(failed_sections)))
        if key not in cache:
            cache[key] = checked_count(payload, binding)[0]["input_tokens"]
        return cache[key]

    raw_uncut = max(1, estimate_context_prompt_tokens(uncut_rows, schemas, provider="local"))
    measured_ratio = first_sample["input_tokens"] / raw_uncut
    last_tokens, last_raw = first_sample["input_tokens"], raw_uncut
    has_uncovered = bool(uncovered_owner_rows(snapshot))
    rungs = [("keep_owner", False, lambda: measured_ratio), ("before_f7", False, lambda: measured_ratio)]
    if has_reader:
        rungs.append(("before_f7", True, lambda: measured_ratio))
    rungs.append(("before_f7_outline", has_reader, lambda: measured_ratio))
    if has_uncovered:
        rungs.extend([("calibrated_f7", has_reader, lambda: last_tokens / last_raw),
                      ("minimum", has_reader, lambda: measured_ratio)])

    for floor_mode, pressure, ratio_fn in rungs:
        rung_state = SimpleNamespace(**{
            **vars(state), "known_window": True, "output_reserve": caller_ceiling,
            "window_record": serving_record, "ratio": ratio_fn(),
        })
        cand_contents, _ = project(rung_state, user_json, retain=trial_retain, exact_local_pressure=pressure,
                                   defer_unfit=True, floor_mode=floor_mode)
        cand_sys, _, cand_prog = cand_contents["nano"]
        cand_rows = _request_rows(cand_sys, user_json, core.supplementary_messages_json, room_data=cand_prog)
        cand_payload = prepare(cand_rows)
        last_tokens = count_payload(cand_payload)
        last_raw = max(1, estimate_context_prompt_tokens(cand_rows, schemas, provider="local"))
        if last_tokens <= input_limit:
            real_contents, real_mode = project(rung_state, user_json, retain=real_retain, exact_local_pressure=pressure,
                                               defer_unfit=True, floor_mode=floor_mode)
            real_sys, real_facts, real_prog = real_contents["nano"]
            if any(row.get("predicted") for row in real_facts.get("host_sections_addressed", ())):
                raise _CounterUnavailable("unavailable", unreachable=False)
            real_rows = _request_rows(real_sys, user_json, core.supplementary_messages_json, room_data=real_prog)
            real_payload = prepare(real_rows)
            last_tokens = (last_tokens if input_fingerprint(real_payload) == input_fingerprint(cand_payload)
                           else count_payload(real_payload))
            if last_tokens <= input_limit:
                state.window_record, state.output_reserve = serving_record, caller_ceiling
                return real_contents, real_mode
            last_raw = max(1, estimate_context_prompt_tokens(real_rows, schemas, provider="local"))

    req_book = 0
    if core.development_required:
        final_mode = "minimum" if has_uncovered else "before_f7_outline"
        no_book_state = SimpleNamespace(**{
            **vars(state), "known_window": True, "output_reserve": caller_ceiling,
            "window_record": serving_record, "ratio": measured_ratio,
        })
        nb_contents, _ = project(
            no_book_state, user_json, core_override=replace(core, development_required=False),
            retain=trial_retain, exact_local_pressure=has_reader, defer_unfit=True, floor_mode=final_mode)
        nb_sys, _, nb_prog = nb_contents["nano"]
        nb_tokens = count_payload(prepare(_request_rows(nb_sys, user_json, core.supplementary_messages_json, room_data=nb_prog)))
        if nb_tokens <= input_limit:
            req_book = max(1, last_tokens - nb_tokens)
    # The counted request is the whole input this start needs, the required handbook inside it, as the estimated walk's
    # refusal names it (``memory_floor.fit_required_book``): its book part is said beside, never taken out of the sum.
    unfit = MinimumViewUnfit(
        "This task's directly required handbook cannot fit beside its ordinary view" if req_book
        else "Protected owner words and the minimum view cannot fit the usable input",
        fixed_tokens=last_tokens, memory_tokens=0, reserve_tokens=reserve,
        window_tokens=serving_record.total_equivalent or window, ratio=1.0, required_book_tokens=req_book)
    unfit.server_program = ((state.route.get("_resolved_target") or {}).get("ollama_record") or {}).get("protocol")
    raise unfit


def _start_control_reason(owner):
    """Read this unpaid start's controls; a contention retry can itself observe the interruption."""
    from ouroboros._usage_wait import retry_presend
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import dispatch_control_reason

    try:
        return retry_presend(lambda: dispatch_control_reason(owner, before_task_start=True),
                             check=False, before_task_start=True)
    except PhysicalDispatchInterrupted as control:
        control.before_task_start = True
        raise


def _control_after_count(task):
    """A control accepted while the start's count ran binds before the loop: no counted view enters it."""
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import current_model_wait

    owner = current_model_wait()
    if owner is None or owner.task_id != str(task.get("id") or ""):
        return
    if reason := _start_control_reason(owner):
        control = PhysicalDispatchInterrupted(reason)
        control.before_task_start = True
        raise control


def _reference_web_start_origin(core, snapshot):
    """Bind the duplicated Runtime prose to the first input of this attested Web turn, without a source read."""
    from ouroboros.context_sections import reference_runtime_origin
    from ouroboros.memory_view_work import TURN_ORIGIN_KEY

    room = snapshot.room or {}
    if not (origin := room.get(TURN_ORIGIN_KEY)):
        return core
    address, ref = origin["address"], origin["ref"]
    item = next((row for row in room["lane1"] if row["address"] == address), None)
    if item is None or core.user_content_json != json.dumps(item["text"], ensure_ascii=False, sort_keys=True):
        return core
    for n, part in enumerate(core.dynamic_sections):
        if part.section_id == "runtime":
            selected = reference_runtime_origin(part, ref, item["text"], address)
            if selected is not part:
                parts = (*core.dynamic_sections[:n], selected, *core.dynamic_sections[n + 1:])
                return replace(core, dynamic_sections=parts, dynamic_text="\n\n".join(parts))
            break
    return core


def walk_local_root_start(env, core, task, state, schemas, project):
    """Admit a fresh Web/Main Nano root (queued, or my human's direct turn) on a qualified 80k local server by exact
    formatter count before replacing the working message (``rescued_start``) or refusing ``MinimumViewUnfit`` on an
    approximate estimate. Whatever a count leaves (a start, a refusal, an unproven count the ordinary estimate then
    decides), Pause, Stop, a deadline or Wrap up accepted while it ran bind first."""
    from ouroboros.context_fit import MinimumViewUnfit
    from ouroboros.context_input_selection import rescued_start
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import current_model_wait

    snapshot = _fresh_local_root_snapshot(core, task, state)
    if snapshot is None:
        return None
    start_core = _reference_web_start_origin(core, snapshot)

    def exact_project(frame, text, *, core_override=None, **options):
        override = start_core if core_override is None and start_core is not core else core_override
        return project(frame, text, core_override=override, **options)

    def fit(text):
        return _exact_start_fit(start_core, state, schemas, exact_project, snapshot, text)

    while True:
        try:
            try:
                outcome = (*fit(core.user_content_json), {})
            except MinimumViewUnfit as unfit:
                outcome = rescued_start(env, task, core.user_content_json, unfit, "nano", fit)
        except MinimumViewUnfit as refused:
            outcome = refused
        except _CounterUnavailable as unavailable:
            state.counter_status, outcome = unavailable.status, None  # None: the ordinary walk decides
            if unavailable.unreachable:
                preview = SimpleNamespace(**{**vars(state), "known_window": False, "ratio": 0.01})
                outcome = (*project(preview, core.user_content_json), {})
        except PhysicalDispatchInterrupted as control:
            owner = current_model_wait()
            if owner is None or owner.task_id != str(task.get("id") or ""):
                raise
            if control.control_reason == "owner_pause":
                # Bind the ordinary prestart fence, not an exact continuation of a preview. A fence
                # already released while unwinding the counter no longer parks the task.
                reason = _start_control_reason(owner)
                if not reason:
                    continue
                if reason != control.control_reason:
                    newer = PhysicalDispatchInterrupted(reason)
                    newer.before_task_start = True
                    raise newer from control
            control.before_task_start = True
            raise
        _control_after_count(task)  # the one prestart probe for every outcome a count reached
        if isinstance(outcome, MinimumViewUnfit):
            raise outcome
        return (*outcome, start_core if state.counter_status == "exact" else core) if outcome is not None else None


def _count_floor_forms(render, prepare, counted, window, reply_floor, *, world_reader):
    """Select by exact counts; the selected N and B always share the same WORLD and other background."""
    pair, taken, reason = render(world_short=False)
    if pair is None:
        return None, reason

    def forms(pair, taken, prefix):
        n, b, *whole = (prepare(messages) for messages in pair)
        named = "whole_index" if whole else "minimum"
        payload = whole[0] if whole else n
        full, gap = counted(prefix + named, payload)
        if gap:
            return None, gap
        return (payload, b, {**taken, "knowledge_index_form": "whole"}, full, full, n if whole else None), None

    first, gap = forms(pair, taken, "")
    if gap:
        return None, gap
    if first[3]["input_tokens"] + reply_floor <= window:
        return first[:5], None
    variants = [("", first, taken)]
    if world_reader and taken["world_form"] == "whole":
        shorter, world_forms, _ = render(world_short=True)
        if world_forms.get("world_form") == "reader":
            second, gap = forms(shorter, world_forms, "world_reader_")
            if gap:
                return None, gap
            omitted = max(0, first[3]["input_tokens"] - second[3]["input_tokens"])
            if omitted:
                world_forms = {**world_forms, "world_omitted_tokens": omitted}
                second = (*second[:2], {**second[2], "world_omitted_tokens": omitted}, *second[3:])
            if second[3]["input_tokens"] + reply_floor <= window:
                return second[:5], None
            if omitted:
                variants = [("world_reader_", second, world_forms)]
    best = min((variant[:5] for _, variant, _ in variants), key=lambda item: item[3]["input_tokens"])
    for prefix, (_, b, _, _, full, n), taken in variants:
        if n is None:
            continue
        minimum, gap = counted(prefix + "minimum", n)
        if gap:
            return None, gap
        candidate = n, b, taken, minimum, full
        if minimum["input_tokens"] + reply_floor <= window:
            return candidate, None
        if minimum["input_tokens"] < best[3]["input_tokens"]:
            best = candidate
    return best, None


def measure_minimum(context, rows, actual, actual_measurement, caller_max, prepare, measure):
    """Count N/B with the serving formatter and retain only the observations usable by the startup consumer.

    ``prepare`` uses the ordinary local wire projection and this attempt's clock line. ``measure`` can only count;
    no member of the pair is returned to the dispatch path. The ordinary physical candidate is never mutated.
    The window is the launched limit (``local_model_server.effective_window``), never the backend's padded
    allocation; a binding that cannot show both leaves the floor unknown rather than borrowing that padding.
    """
    from ouroboros.context_budget import _reply_floor
    from ouroboros.llm_helper_window import window_reply_candidate
    from ouroboros.local_model_server import input_fingerprint

    plan, episode = context.nano_floor_plan
    actual_sample = _sample(actual, actual_measurement)
    result = {"schema": 1, "scope": episode, "status": "unknown", "core_sha256": plan.core_sha256,
              "floor_kind": "physical_shortage_before_uncovered_owner_words", "growth_guard": "not_enforced",
              "work_headroom_basis": "additional_after_this_prepared_work_tail", "memory_sections": MEMORY_SECTIONS,
              **_origin(plan),
              "actual": {**(actual_sample or {"native_input_sha256": input_fingerprint(actual), "input_tokens": None}),
                         "sent_max_tokens": actual.get("max_tokens")}}
    if actual_sample is None:
        return _counter_gap(result, "actual", actual_measurement)
    binding, binding_gap = _verified_binding(actual_measurement)
    if binding_gap:
        return _gap(result, binding_gap, count="actual" if binding_gap == "reply_limit_unverified" else "")
    window = binding["context_window"]
    if any(isinstance(block, dict) and "image" in str(block.get("type", ""))
           for row in rows for block in (row.get("content") if isinstance(row.get("content"), list) else ())):
        return _gap(result, "nontext_input")
    names = [tool["function"]["name"] for tool in actual.get("tools", ()) if (tool.get("function") or {}).get("name")]
    result.update(binding, counter_confirmed=True)
    ceiling, reply_floor = _reply_floor(caller_max, True)
    result.update(caller_max_tokens=ceiling, reply_floor_tokens=reply_floor)

    def counted(name, payload):
        observed = measure(payload)
        if (sample := _sample(payload, observed)) is None:
            return None, _counter_gap(result, name, observed)
        cand_binding, cand_gap = _verified_binding(observed)
        if cand_gap == "reply_limit_unverified":
            return None, _gap(result, "reply_limit_unverified", count=name)
        if cand_binding != binding or cand_gap:
            return None, _gap(result, "serving_binding_changed", count=name)
        return sample, None

    # A prospective write compares the same complete WORLD in every branch.
    # Unlike a send observation, this count measures even an unfitting view.
    whole_world = episode == "publication_candidate"
    try:
        selected, gap = _count_floor_forms(
            lambda world_short: _render(plan, rows, window=window, tool_names=names,
                                        world_short=world_short, whole_world=whole_world),
            prepare, counted, window, reply_floor, world_reader="read_file" in names and not whole_world)
        if selected is None:
            if isinstance(gap, dict):
                return gap
            code, _, detail = gap.partition(":")
            return _gap(result, code, detail=detail)
        n, b, taken, minimum, full = selected
    except (OSError, ValueError) as error:
        return _gap(result, "minimum_render_unavailable", detail=type(error).__name__)
    result.update(**taken, **{name: {"native_input_sha256": input_fingerprint(payload), "input_tokens": None}
                              for name, payload in (("minimum", n), ("without_memory", b))})
    without, gap = counted("without_memory", b)
    if gap:
        return gap
    count = minimum["input_tokens"]
    # Apply the existing exact Nano reply allocator to N, not B. The handbook is
    # fully counted in headroom even when the existing allocator sizes reply before charging it.
    frame, short, _, _ = window_reply_candidate({**n, "max_tokens": caller_max}, "max_tokens", context, window,
        {"input_tokens": count, "context_window": window}, helper=False)
    actual_count, actual_reply = actual_sample["input_tokens"], actual["max_tokens"]
    return {**result, "minimum": minimum, "without_memory": without, "whole_index": full, "status": "exact",
            "memory_tokens": count - without["input_tokens"], "whole_memory_tokens": full["input_tokens"] - without["input_tokens"],
            "minimum_reply_allowance_tokens": frame["max_tokens"], "minimum_reply_floor_shortfall": short,
            "actual_work_headroom_at_wire_allowance_tokens": window - actual_count - actual_reply,
            "minimum_work_headroom_at_reply_floor_tokens": window - count - reply_floor,
            "minimum_work_headroom_at_wire_allowance_tokens": window - count - frame["max_tokens"]}


def observation_failed(context, actual, error):
    """The entry round's evidence when its observation itself failed: unknown with a typed cause, never a count.

    Only the sender uses it, so a defect in a count-only projection cannot cost the request it rides; a prospective
    publication calls ``measure_minimum`` directly and stays closed on the same failure."""
    from ouroboros.local_model_server import input_fingerprint

    plan, episode = context.nano_floor_plan
    return _gap({"schema": 1, "scope": episode, "status": "unknown", "core_sha256": plan.core_sha256,
                 "actual": {"native_input_sha256": input_fingerprint(actual), "input_tokens": None,
                            "sent_max_tokens": actual.get("max_tokens")}},
                "observation_failed", detail=type(error).__name__)


def bound_evidence(context, evidence):
    return replace(context, nano_floor_plan=None, nano_floor_evidence=evidence)
