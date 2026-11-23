"""Record synthetic inputs through the UI's Python producers, without a server.

Regenerate from the repo root: `python -I -S scripts/safe_test.py --
<dependency-python> -m scripts.generate_card_gallery_fixture` (one command).
The freshness test uses `--stdout` in a fresh interpreter: producers importing
clock readers during recording must not leave them in pytest. `frames` is the
ordered WebSocket sequence. `responses` holds whole JSON bodies in replay order;
`status` is present for a non-200 response. A task card's state holds one list per
read the chat makes (`history`, `state`, `detail`), each in replay order, from a data
root of its own whose path the recording names `/gallery-data`. A connection card's
state holds the token its scene types and the host's `identity` answer for it, which
the scene hands the card's own check instead of posting. Existing
work/consciousness recordings stay in web/tests/fixtures and are read directly by
gallery consumers.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import queue
import re
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import UUID

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/card_gallery.json"
STAMP = "2026-11-20T12:00:00Z"
EPOCH = datetime.fromisoformat(STAMP.replace("Z", "+00:00")).timestamp()
GALLERY_DATA = "/gallery-data"  # what a recording calls the temporary data root its producers wrote into
TASK_NAME = "Compare the project summary before and after moving its supporting material into the reader"
TASK_LINES = ("Reading both saved reports first: the summary, then the material it points to.",
              "Comparing the two summaries at the working width and in a narrow room.",
              "Both read alike now; checking the longest names once more before I answer.")
# The keys of the settings answer Settings › Agents reads; the others name this machine's folders and network.
AGENT_KEYS = ("OUROBOROS_SUBAGENTS", "OUROBOROS_RUNTIME_MODE", "OUROBOROS_PROCESSING_PREFERENCE",
              "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT",
              "OUROBOROS_MAX_SUBAGENT_DEPTH")
LONG_TITLE = "Compare the rendering of the project summary before and after moving its supporting material into the reader"
LONG_TEXT = "\n\n".join((
    "## What should stay visible?\n\nThe summary is meant to tell a returning reader what changed, "
    "what they need to do next, and which decisions remain open.",
    "## What moves into details?\n\nKeep the exact source, the entire response, and the observations that "
    "explain its limitations. The technical explanation can open separately without shortening the original.",
    "- Leave the decision visible at the working width.\n- Keep the original wording selectable.\n"
    "- Show the same explanation when this message is read again.",
    "## Next step\n\nBefore accepting the change, compare this result on a wide display and in a narrow "
    "room. Repeat with a reader who has not seen the earlier work.",
))


def _request(root, **path_params):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=root)),
                           path_params=path_params, query_params={})


def _json_body(response):
    return json.loads(response.body)


def _quiz_cases(root, bridge, frames):
    from ouroboros import owner_quiz

    cases = {}
    options = [{"label": "Keep the summary", "detail": "Leave the current reading order.", "recommended": True},
               {"label": "Open the whole report first", "detail": "Make the source the first thing you read."}]
    for name in ("open", "long", "waiting", "closed", "answered"):
        frames.clear()
        quiz_id, task_id = f"gallery-quiz-{name}", f"gallery-question-{name}"
        question = LONG_TEXT if name == "long" else "Which result should lead the report?"
        waiting = name == "waiting"
        choices = [] if name == "open" else options
        assumption = "Keep the current summary meanwhile." if not waiting else ""
        owner_quiz.record_asked(root, task_id, quiz_id=quiz_id, question=question,
                                options=[option["label"] for option in choices],
                                option_details=[option["detail"] for option in choices],
                                recommended_index=0 if choices else None, assumption=assumption,
                                wait_for_answer=waiting, chat_id=1)
        ok, reason = bridge.send_quiz(1, quiz_id, question, choices, assumption=assumption,
                                      task_id=task_id, wait_for_answer=waiting)
        if not ok or not frames:
            raise RuntimeError(f"Could not record quiz {name}: {reason}")
        if name == "closed":
            closed = owner_quiz.record_closed(root, task_id, quiz_id=quiz_id, actor_task_id=task_id,
                                              reason="The comparison no longer needs a choice.")
            if not closed["ok"]:
                raise RuntimeError(f"Could not settle quiz {quiz_id}: {closed}")
            block = closed["block"]
            bridge.send_quiz_state(quiz_id, task_id, block["state"], chat_id=1,
                                   closed_reason=block["closed_reason"], closed_at=block["closed_at"],
                                   closed_actor=block["closed_actor"], closed_request_id=block["closed_request_id"],
                                   closed_by_task_id=block["closed_by_task_id"],
                                   question_delivery=owner_quiz.question_delivery_projection(block))
        if name == "answered":
            answer = owner_quiz.record_answered(root, task_id, quiz_id=quiz_id, request_id="gallery-choice",
                                                 option_index=1, comment="Open the report first.\nKeep the full text.")
            if not answer["ok"]:
                raise RuntimeError(f"Could not settle quiz {quiz_id}: {answer}")
            block = answer["block"]
            bridge.send_quiz_state(quiz_id, task_id, block["state"], block["answered_index"], 1, block["comment"],
                                   question_delivery=owner_quiz.question_delivery_projection(block))
        cases[name] = {"frames": list(frames)}
    return cases


def _living_cases(root, bridge, frames):
    from ouroboros import card_store
    from ouroboros.gateway import cards
    from supervisor import card_delivery

    cases = {}
    for index, name in enumerate(("long", "updated", "missing"), 1):
        frames.clear()
        task_id = f"gallery-card-{name}"
        body = {"components": [{"type": "markdown", "text": LONG_TEXT if name == "long" else "The comparison is ready."}]}
        with patch.object(card_store.uuid, "uuid4", return_value=UUID(hex=f"{index:016x}" + "0" * 16)):
            record = card_store.write_card(root, chat_id=1, task_id=task_id, title=LONG_TITLE,
                                           fallback_text="Comparison of the two saved reports.", kind="components", body=body)
        card_delivery.send_card(bridge, record, 1, task_id=task_id, interactive=False)
        card_id = record["card_id"]
        if name == "missing":
            # The timeline still knows the published card; the body is no longer in its store.
            (root / card_store.CARDS_DIR / f"{card_id}.json").unlink()
        response = asyncio.run(cards.api_card_get(_request(root, card_id=card_id)))
        case = {"responses": [_json_body(response)]}
        if response.status_code != 200:
            case["status"] = response.status_code
        if name == "updated":
            newer = card_store.write_card(root, chat_id=1, task_id=task_id, title="Comparison complete",
                                          fallback_text="Both reports were checked.", kind="components", replace=card_id,
                                          body={"components": [{"type": "callout", "text": "Both reports were checked."}]})
            card_delivery.send_card_state(bridge, newer)
            case["responses"].append(_json_body(asyncio.run(cards.api_card_get(_request(root, card_id=card_id)))))
        cases[name] = {"frames": list(frames), **case}
    return cases


def _review_cases(root):
    from ouroboros import review_ledger
    from ouroboros.gateway import task_reviews
    from ouroboros.review_projection import compact_review_projection, task_ledger_records
    from tests.test_review_record_card_projection import _gate_record

    cases = {}
    for name in ("empty", "waiting", "failure", "missing_record"):
        task_id = f"gallery-review-{name}"
        if name != "empty":
            record = _gate_record(task_id, opus_findings=12, opus_pending=name == "waiting")
            record.record_id = f"rl-fixture-{name}"
            record.brief["goal"] = LONG_TITLE if name == "failure" else "Compare the two reports"
            review_ledger.write_record(root, record)
            if name == "missing_record":
                review_ledger.record_path(root, record.record_id).unlink()
        # This is the `review_projection` field of task detail, not a replacement for task detail.
        records, omitted = task_ledger_records({"id": task_id}, root)
        projection = compact_review_projection([], records)
        projection["review_records_omitted"] = omitted
        response = task_reviews._history(_request(root, task_id=task_id))
        cases[name] = {"task_id": task_id, "review_projection": projection, "responses": [_json_body(response)]}
    return cases


def _widget_cases(root):
    from ouroboros import card_store, extension_loader
    from ouroboros.gateway import widgets

    request = _request(root)
    with patch.object(extension_loader, "live_widget_projection", return_value=[]):
        empty = _json_body(asyncio.run(widgets.api_widgets(request)))
        card_store.pin_card(root, "card_0000000000000001")
        pinned = _json_body(asyncio.run(widgets.api_widgets(request)))
    return {"empty": {"responses": [empty]}, "pinned_long": {"responses": [pinned]}}


def _login_cases():
    from ouroboros.gateway import claudexor_accounts

    # Fixture input is the daemon's synthetic job. The Python gateway wraps it;
    # pending transport, script errors and typed provider fields aren't inferred.
    job = {"jobId": "gallery-login", "harness": "codex", "action": "login"}
    return {name: {"responses": [claudexor_accounts._login_job_response({**job, **fields}, job_id=job["jobId"])]}
            for name, fields in {
                "waiting": {"state": "waiting_for_input", "phase": "awaiting_user"},
                "failed": {"state": "failed", "phase": "completed", "outcome": {"reason": "vendor_program_failed"},
                           "message": "The sign-in window closed before the account was connected."},
            }.items()}


def _skill_cases(root):
    from ouroboros import extension_registry_state
    from ouroboros.gateway import contributions, extensions
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state
    from ouroboros.tools import github

    # No live extension, bundled offers or configured tokens in this installation.
    with (patch.object(extensions, "snapshot", return_value={"extensions": [], "tools": [], "routes": [],
                       "ws_handlers": [], "ui_tabs": [], "settings_sections": [], "contributions": []}),
          patch.object(extension_registry_state, "live_contribution_problems", return_value={}),
          patch.object(contributions, "_uninstalled_builtin_offers", return_value=[]),
          patch.object(github, "github_token_from_env_or_settings", return_value="")):
        empty = extensions._build_extensions_index(root, repo_path="")
        name = "report-reader"
        folder = root / "skills" / "external" / name
        folder.mkdir(parents=True)
        manifest = folder / "SKILL.md"
        manifest.write_bytes((f'---\nname: {name}\ntype: instruction\nversion: 1.0.0\n'
                              f'description: "{LONG_TITLE}."\n---\n\n# Report reader\n').encode("utf-8"))
        instant = datetime.fromisoformat(STAMP.replace("Z", "+00:00")).timestamp()
        os.utime(manifest, (instant, instant))  # the Extensions producer reads the manifest's mtime
        skill = find_skill(root, name, repo_path="")
        save_review_state(root, name, SkillReviewState(status="pending", content_hash=skill.content_hash,
                          findings=[{"item": "skill_preflight", "verdict": "FAIL", "severity": "critical",
                                     "reason": "The packaged instructions could not be validated."}]))
        blocked = extensions._build_extensions_index(root, repo_path="")
    return {"empty": {"responses": [empty]}, "review_blocked": {"responses": [blocked]}}


def _agents_cases():
    """Settings › Agents: the settings answer for a synthetic roster (its own keys, see AGENT_KEYS) beside the review
    pool the same roster gives. An empty pool alone does not say which agents the page lists; the roster does."""
    from ouroboros import config, reviewer_slot_config, server_maintenance
    from ouroboros.gateway import settings

    row = {"subagent_id": "gallery-critic", "recommended_use": LONG_TITLE,
           "route": {"kind": "agent_session", "target_id": "codex=gpt-6.2-sol"}}
    cases = {}
    # No installed review history, migration files or pricing are inputs to this recording.
    with TemporaryDirectory(prefix="card-gallery-settings-") as temporary, \
            patch.object(reviewer_slot_config, "reviewer_slot_last_executions", return_value={}), \
            patch.object(server_maintenance, "review_pool_migration_records", return_value={}), \
            patch.object(settings, "_review_pool_costs", return_value={}):
        path = Path(temporary) / "settings.json"
        request = SimpleNamespace(query_params={}, app=SimpleNamespace(state=SimpleNamespace(
            drive_root=Path(temporary), port_file=Path(temporary) / "server_port", bind_host="127.0.0.1")))
        for name, items in (("empty_pool", [row]), ("disabled_reviewer", [{**row, "enabled": False, "review_eligible": True}])):
            roster = json.dumps({"enabled": True, "items": items})
            path.write_text(json.dumps({"OUROBOROS_SUBAGENTS": roster}), encoding="utf-8")
            with patch.object(config, "SETTINGS_PATH", path):
                answer = _json_body(asyncio.run(settings.api_settings_get(request)))
            meta = answer["_meta"]
            kept = {key: answer[key] for key in AGENT_KEYS if key in answer}
            kept["_meta"] = {"available_subagents": meta["available_subagents"], "setup_contract": meta["setup_contract"],
                             **{part: {key: meta[part][key] for key in AGENT_KEYS if key in meta[part]}
                                for part in ("effects", "tokens")}}
            cases[name] = {"settings": kept, "responses": [settings.review_pool_payload({"OUROBOROS_SUBAGENTS": roster})]}
    return cases


def _connection_cases():
    """A skill's declared connection card in the setup's Skills step: the starter list the host serves for the fixture
    skill of tests/test_onboarding_connections.py (its card's addresses are example.test), and the host's identity
    check of a token, which that skill's own function answers (`1:` accepted as demo_bot, anything else refused),
    never a service."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import onboarding_connections, skill_starter, store
    from tests.test_onboarding_connections import _skill

    offer = [{"id": "builtin/demo", "name": "demo", "source_id": "builtin", "category": ["messaging"],
              "needs_keys": ["DEMO_TOKEN", "DEMO_CODE"]},
             {"id": "builtin/other", "name": "other", "source_id": "builtin", "category": ["messaging"], "needs_keys": []}]
    check = "/api/onboarding/connection/identity"
    cases = {}
    with TemporaryDirectory(prefix="card-gallery-setup-") as temporary, ExitStack() as restore:
        root = Path(temporary)
        restore.callback(store.close_connections_under, root)
        skills = _skill(root / "skills")
        for target, value in (("ouroboros.skill_builtin.bundled_skills_dir", lambda: skills),
                              ("ouroboros.skill_source_catalog.snapshot_for", lambda *_a, **_k: (None, {})),
                              ("ouroboros.config.get_auto_grant_enabled", lambda: True)):
            restore.enter_context(patch(target, value))
        restore.enter_context(patch.object(skill_starter, "_offer", lambda *_a, **_k: [dict(item) for item in offer]))
        restore.enter_context(patch.object(skill_starter, "starter_ids", lambda *_a, **_k: []))
        groups = skill_starter.starter_groups(root, rows=[])
        app = Starlette(routes=[Route(check, endpoint=onboarding_connections.api_connection_identity, methods=["POST"])])
        with TestClient(app) as client:
            for name, token in (("qr", "1:gallery-synthetic-token"), ("refused", "2:gallery-synthetic-token")):
                answer = client.post(check, json={"skill": "demo", "token": token})
                cases[name] = {"responses": [groups], "token": token,
                               "identity": {"status": answer.status_code, "body": answer.json()}}
    return cases


def _provider_card_cases():
    """The setup wizard's provider card in its providers panel: the bootstrap the host serves for a settings document
    (the parts that panel reads), without a compatible server and with one whose key the bootstrap only marks as set."""
    from ouroboros import config, store
    from ouroboros.gateway import onboarding_host
    from ouroboros.provider_models import PROVIDER_KINDS

    compatible = PROVIDER_KINDS["openai-compatible"]
    documents = {
        "closed": {},
        "saved": {
            compatible.flat_key("base_url"): "https://llm.example.test/v1",
            compatible.flat_key("api_key"): "sk-gallery-not-a-key",
        },
    }
    cases = {}
    with TemporaryDirectory(prefix="card-gallery-providers-") as temporary, ExitStack() as restore:
        root = Path(temporary)
        restore.callback(store.close_connections_under, root)
        path = root / "settings.json"
        request = SimpleNamespace(query_params={}, headers={}, app=SimpleNamespace(state=SimpleNamespace(drive_root=root)))
        for name, document in documents.items():
            path.write_text(json.dumps(document), encoding="utf-8")
            with patch.object(config, "SETTINGS_PATH", path):
                bootstrap = _json_body(asyncio.run(onboarding_host.api_onboarding_bootstrap(request)))["bootstrap"]
            contract = bootstrap["contract"]
            cases[name] = {"responses": [{
                "initialState": bootstrap["initialState"], "deployment": bootstrap["deployment"],
                "contract": {part: contract[part] for part in ("providerFields", "budgetFields", "modelSlots")}}]}
    return cases


def _clock():
    """A producer's clock that moves one second each time it is read, from STAMP."""
    ticks = itertools.count()
    return lambda: "2026-11-20T12:%02d:%02d+00:00" % divmod(next(ticks), 60)


@contextmanager
def _task_world(task_id):
    """One task's own data root: the real chat bridge, send handler and stores, the queue maps the census reads, the
    clock above in every module that reads `utc_now_iso` and identifiers counted from one (a terminal projection
    mints a uuid4 token). Yields (root, broadcast frames, say, deliver, queue maps)."""
    import uuid

    from ouroboros import store, utils
    from ouroboros.agent import OuroborosAgent
    from supervisor import events_chat_delivery, message_bus
    from supervisor import queue as queue_maps

    with TemporaryDirectory(prefix="card-gallery-task-") as temporary, ExitStack() as restore:
        root = Path(temporary)
        restore.callback(store.close_connections_under, root)
        identifiers = itertools.count(1)
        restore.enter_context(patch.object(uuid, "uuid4", lambda: UUID(int=next(identifiers))))
        clock, original = _clock(), utils.utc_now_iso
        readers = [module for module in list(sys.modules.values())
                   if isinstance(module, ModuleType) and getattr(module, "utc_now_iso", None) is original]
        for module in readers:
            restore.enter_context(patch.object(module, "utc_now_iso", clock))
        (root / "logs").mkdir()
        (root / "logs" / "chat.jsonl").touch()
        frames, events = [], queue.Queue()
        bridge = message_bus.LocalChatBridge()
        bridge._broadcast_fn = frames.append
        for target, name, value in ((message_bus, "DATA_DIR", root), (message_bus, "_BRIDGE", bridge),
                                    (message_bus, "load_state", lambda: {"owner_id": 1}),
                                    (message_bus, "publish_event", lambda *_a, **_k: None),
                                    (events_chat_delivery, "_bound_project_chat_id", lambda *_a: 0),
                                    (queue_maps, "PENDING", []), (queue_maps, "RUNNING", {})):
            restore.enter_context(patch.object(target, name, value))
        agent = SimpleNamespace(_last_progress_ts=None, _event_queue=events, _current_chat_id=1, _current_task_id=task_id,
                                _subagent_progress_meta=lambda _event: {},
                                tools=SimpleNamespace(_ctx=SimpleNamespace(task_attempt=1)))
        delivery = SimpleNamespace(DRIVE_ROOT=root, RUNNING={task_id: {"task": {"id": task_id, "_attempt": 1}}},
                                   send_with_budget=message_bus.send_with_budget, append_jsonl=lambda *_a: None)

        def deliver(event):  # a worker's send_message event, through the supervisor's own send handler
            events_chat_delivery._handle_send_message(dict(event), delivery)

        def say(text):  # the model's narration: the worker's emit, then that handler
            OuroborosAgent._emit_progress(agent, text, narration=True)
            deliver(events.get_nowait())

        yield root, frames, say, deliver, queue_maps


def _normalize_data_paths(value, *root_spellings):
    """Name only this recording's root and its descendants, on decoded JSON values."""
    delimiters = r"\s\"'`<>()\[\]{},;="
    patterns = []
    for spelling in sorted(set(root_spellings), key=lambda root: (-len(root), root)):
        windows = bool(PureWindowsPath(spelling).drive)
        if windows:
            prefix = r"[/\\]".join(map(re.escape, spelling.replace("\\", "/").split("/")))
            path = rf"(?i:{prefix})(?P<tail>[/\\][^{delimiters}]*)?"
        else:
            path = re.escape(spelling)
        separators = r"/\\" if windows else "/"
        patterns.append((re.compile(rf"(?<![^{delimiters}]){path}(?=$|[{delimiters}{separators}:])"), windows))

    def normalize(item):
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, str):
            for pattern, windows in patterns:
                if windows:
                    item = pattern.sub(lambda match: GALLERY_DATA + (match["tail"] or "").replace("\\", "/"), item)
                else:
                    item = pattern.sub(GALLERY_DATA, item)
        return item

    return normalize(value)


def _task_case(task_id, *, state="done", answer=LONG_TEXT, usage=None):
    """One root task card in Main through the producers a live task runs: its narration through the worker's emit and
    the supervisor's send handler; a finished task's answer through the end-of-task pipeline and that handler; a
    running or paused one as the queue keeps it; then what the page reads (history, census, the task's detail) from
    the gateway's own readers. A running task's next narration is its live frame."""
    from ouroboros import agent_task_pipeline
    from ouroboros.dialogue_import import export_for_older_body
    from ouroboros.gateway import tasks as task_gateway
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.post_task_checkpoint import set_root_post_task_checkpoint
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, write_task_result
    from tests._work_projection_shared import state_fields

    task = {"id": task_id, "type": "task", "chat_id": 1, "root_task_id": task_id, "title": TASK_NAME,
            "text": TASK_NAME, "suggested_name": TASK_NAME}
    with _task_world(task_id) as (root, frames, say, deliver, queue_maps):
        write_task_result(root, task_id, STATUS_SCHEDULED, chat_id=1, root_task_id=task_id,
                          suggested_name=TASK_NAME, result="Task accepted and scheduled.")
        for line in TASK_LINES[:2]:
            say(line)
        env = SimpleNamespace(drive_root=root, repo_dir=root)
        if state == "running":
            write_task_result(root, task_id, STATUS_RUNNING, chat_id=1, root_task_id=task_id, result="running")
            queue_maps.RUNNING[task_id] = {"task": task, "started_at": EPOCH, "attempt": 1}
        elif state == "paused":  # the owner's Pause, as the queue keeps a paused root
            queue_maps.PENDING.append({**task, "_budget_pause": {"reason": "owner", "paused_at": STAMP}})
        else:
            ended = []
            with patch.object(agent_task_pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None):
                agent_task_pipeline.emit_task_results(env, None, None, ended, dict(task), answer, usage or {"rounds": 6},
                                                      {"tool_calls": [], "reasoning_notes": []}, start_time=EPOCH,
                                                      drive_logs=root / "logs")
            for event in ended:
                if event.get("type") == "send_message":
                    deliver(event)
            # The root's post-task work done (its free host-facts row is left out: the stored row names the
            # recording's temporary folders, files_rescued, so its row address would differ on every machine).
            set_root_post_task_checkpoint(env, task, "completed")
        export_for_older_body(root)  # the history reader still reads the chain files
        history = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"limit": "50"})))
        detail = task_gateway._task_get_response(_request(root, task_id=task_id))
        census = {"supervisor_ready": True, **state_fields(root)}
        seen = len(frames)
        if state == "running":
            say(TASK_LINES[2])  # arrives while the page is open: the scene's next frame
        case = {"task_id": task_id, "frames": frames[seen:], "history": [_json_body(history)], "state": [census],
                "detail": [_json_body(detail)], **({"detail_status": detail.status_code} if detail.status_code != 200 else {})}
        return _normalize_data_paths(case, str(root.resolve()), str(root))


def _task_card_cases():
    failed = {"rounds": 2, "execution_status": "infra_failed", "reason_code": "provider_unavailable"}
    return {"running": _task_case("gallery-task-running", state="running"),
            "done": _task_case("gallery-task-done"),
            "failed": _task_case("gallery-task-failed", answer="Kept the comparison made so far; the model provider did not answer.",
                                 usage=failed),
            "paused": _task_case("gallery-task-paused", state="paused")}


def _share_cases():
    """Share's Card on a finished task's answer: one that fits a card, and one whose table the card cannot draw."""
    fits = ("Keep the summary first.\n\n## Why\n\n1. **The decision stays visible** at the working width.\n"
            "2. **The source opens** separately, whole.\n\nCheck it with `make check` before you send it.")
    table = ("Both reports, side by side:\n\n| Report | Pages | Opens with |\n| --- | ---: | --- |\n"
             "| Summary | 2 | the decision |\n| Full report | 14 | the source |")
    return {"card": _task_case("gallery-share-card", answer=fits), "table": _task_case("gallery-share-table", answer=table)}


def corpus_bytes():
    """Record in a dedicated process; a lazy import can capture a synthetic clock."""
    from ouroboros import card_store, owner_quiz, review_ledger, store
    from supervisor import card_delivery, message_bus, state

    with TemporaryDirectory(prefix="card-gallery-") as temporary, ExitStack() as restore:
        root = Path(temporary)
        restore.callback(store.close_connections_under, root)
        # Patch time/identity at the producer seam only, and leave outbound transports disconnected.
        for module in (owner_quiz, message_bus, card_store, card_delivery, review_ledger):
            restore.enter_context(patch.object(module, "utc_now_iso", return_value=STAMP))
        restore.enter_context(patch.object(message_bus, "DATA_DIR", root))
        for module in (message_bus, state):
            restore.enter_context(patch.object(module, "load_state", return_value={"owner_id": 1}))
        for module in (message_bus, card_delivery):
            restore.enter_context(patch.object(module, "publish_event", return_value=None))
        frames = []
        bridge = message_bus.LocalChatBridge()
        bridge._broadcast_fn = frames.append
        data = {
            "_note": "Generated by scripts/generate_card_gallery_fixture.py from synthetic inputs; do not edit.",
            "quiz": _quiz_cases(root, bridge, frames),
            "living_card": _living_cases(root, bridge, frames),
            "reviews": _review_cases(root),
            "widgets": _widget_cases(root),
            "login": _login_cases(),
            "skills": _skill_cases(root),
            "agents": _agents_cases(),
            "connection": _connection_cases(),
            "provider_card": _provider_card_cases(),
        }
    # Outside the patches above: each task card's world pins its own clock in every module that reads one.
    data["task_card"] = _task_card_cases()
    data["share"] = _share_cases()
    from tests._work_projection_shared import work_fields

    data["work_reads"] = {"incomplete_read": work_fields([], complete=False),
                          "confirmed_absent": work_fields([], complete=True)}
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Record the synthetic card gallery from its Python producers.")
    parser.add_argument("--stdout", action="store_true", help="write UTF-8/LF bytes to stdout without updating the fixture")
    args = parser.parse_args(argv)
    data = corpus_bytes()
    if args.stdout:
        sys.stdout.buffer.write(data)
    else:
        FIXTURE.write_bytes(data)
        print(FIXTURE.name)


if __name__ == "__main__":
    main()
